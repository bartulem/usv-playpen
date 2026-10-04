"""
@author: bartulem
Unit tests for ``usv_playpen.modeling.torus_geodesics`` -- the three torus
distance geometries (flat-torus wrapped, density-ratio graph geodesic, and
decoder-Jacobian pullback graph geodesic).

The graph geodesics are validated against analytic ground truth wherever
possible: the density geodesic must recover the flat graph metric at
``density_exponent=0`` and grow through low-density regions; the pullback metric
must equal ``J^T J`` for a known linear decoder, reduce to the flat graph under
the identity decoder, and stretch distances along a deliberately stretched axis.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from usv_playpen import os_utils
from usv_playpen.modeling.manifold_metric import _geodesic_distance_matrix
from usv_playpen.modeling.modeling_torus_geodesics import (
    _snap_to_grid,
    build_torus_geodesic_context,
    density_geodesic_matrix,
    flat_torus_distance_matrix,
    geodesic_mae_columns,
    make_qlvm_decode_fn_from_model_cell,
    make_qlvm_decode_fn_from_source,
    resolve_geodesic_decoder_source,
    resolve_manifold_column_names,
    per_event_geodesic_error,
    pullback_geodesic_matrix,
    pullback_metric_at_nodes,
    torus_distance,
    torus_grid,
    torus_kde_density,
)
from usv_playpen.processing.qlvm_latents import condition_quantile_value, load_model_cell
from usv_playpen.processing.qlvm_model import decode_lattice_atlas, decoder_forward, torus_basis_forward
from tests.processing.test_qlvm_latents import _make_model_cell, _phase11_bins


class TestFlat:
    def test_matches_manifold_metric_and_wraps(self):
        """Flat-torus matrix equals the pipeline's wrap-aware distance, is a
        symmetric zero-diagonal matrix, and takes the short way round the seam."""

        rng = np.random.default_rng(0)
        nodes = rng.random((20, 2))
        d = flat_torus_distance_matrix(nodes, period=1.0)
        np.testing.assert_allclose(
            d, _geodesic_distance_matrix(nodes, metric='torus', period=1.0))
        assert np.allclose(np.diag(d), 0.0)
        assert np.allclose(d, d.T)
        seam = np.array([[0.05, 0.5], [0.95, 0.5]])
        assert abs(flat_torus_distance_matrix(seam, period=1.0)[0, 1] - 0.10) < 1e-9


class TestGrid:
    def test_shape_and_range(self):
        """The grid tiles ``[0, period)^2`` at cell centres (nothing on the seam)."""

        g = torus_grid(7, period=1.0)
        assert g.shape == (49, 2)
        assert g.min() > 0.0 and g.max() < 1.0


class TestKDE:
    def test_nonnegative_and_periodic(self):
        """The wrap-aware KDE is non-negative and (near) identical at the two
        sides of the seam, which are physically the same torus point."""

        rng = np.random.default_rng(1)
        samples = (rng.normal(0.5, 0.06, size=(600, 2))) % 1.0
        seam = np.array([[0.0, 0.5], [1.0 - 1e-6, 0.5]])
        d = torus_kde_density(samples, seam, period=1.0)
        assert np.all(d >= 0.0)
        assert abs(d[0] - d[1]) < 1e-3 * max(d.max(), 1e-9)


class TestDensityGeodesic:
    def test_uniform_density_is_exponent_invariant(self):
        """With uniform density every edge weight is just the flat length
        regardless of the exponent, so the geodesic matrix cannot depend on it."""

        nodes = torus_grid(10, period=1.0)
        dens = np.ones(nodes.shape[0])
        d1 = density_geodesic_matrix(nodes, dens, period=1.0, k=8, density_exponent=1.0)
        d3 = density_geodesic_matrix(nodes, dens, period=1.0, k=8, density_exponent=3.0)
        assert np.all(np.isfinite(d1))
        np.testing.assert_allclose(d1, d3)

    def test_low_density_region_lengthens_paths(self):
        """Introducing a low-density region can only raise path costs, and more
        so at a larger inverse-density exponent."""

        nodes = torus_grid(15, period=1.0)
        uniform = np.ones(nodes.shape[0])
        barrier = uniform.copy()
        barrier[np.abs(nodes[:, 0] - 0.5) < 0.08] = 0.02      # low-density stripe
        d_uni = density_geodesic_matrix(nodes, uniform, period=1.0, k=8)
        d_bar1 = density_geodesic_matrix(nodes, barrier, period=1.0, k=8, density_exponent=1.0)
        d_bar2 = density_geodesic_matrix(nodes, barrier, period=1.0, k=8, density_exponent=2.0)
        fin = np.isfinite(d_uni) & np.isfinite(d_bar1) & np.isfinite(d_bar2)
        assert d_bar1[fin].max() > d_uni[fin].max()
        assert d_bar2[fin].max() >= d_bar1[fin].max()


class TestPullback:
    def test_metric_is_JtJ_for_linear_decoder(self):
        """For a linear decoder ``g(z) = A z`` the pullback metric is the
        constant ``A^T A`` at every node (J is constant)."""

        A = jnp.array([[2.0, 0.0], [0.0, 0.5], [1.0, 1.0]])   # (3, 2)
        decode = lambda z: A @ z
        nodes = torus_grid(5, period=1.0)
        g = pullback_metric_at_nodes(nodes, decode)
        ata = np.asarray(A.T @ A)
        assert g.shape == (nodes.shape[0], 2, 2)
        for i in range(g.shape[0]):
            np.testing.assert_allclose(g[i], ata, atol=1e-5)

    def test_identity_decoder_reduces_to_flat_graph(self):
        """The identity decoder gives ``G = I``, so each edge weight is the flat
        length -- the pullback geodesic must equal the flat-length graph geodesic
        (the uniform density geodesic)."""

        decode = lambda z: z
        nodes = torus_grid(9, period=1.0)
        d_pull = pullback_geodesic_matrix(nodes, decode, period=1.0, k=8)
        d_flat_graph = density_geodesic_matrix(
            nodes, np.ones(nodes.shape[0]), period=1.0, k=8, density_exponent=0.0)
        np.testing.assert_allclose(d_pull, d_flat_graph, atol=1e-9)

    def test_anisotropic_stretch_costs_more_along_stretched_axis(self):
        """A decoder that stretches x by 3 and y by 1 (``G = diag(9, 1)``) makes
        an x-separated target farther than an equally flat-distant y-separated
        target."""

        A = jnp.array([[3.0, 0.0], [0.0, 1.0]])
        decode = lambda z: A @ z
        nodes = torus_grid(13, period=1.0)

        def nearest(pt):
            return int(np.argmin(np.sum((nodes - np.asarray(pt)) ** 2, axis=1)))

        src = nearest([0.5, 0.5])
        tgt_x = nearest([0.5 + 0.23, 0.5])
        tgt_y = nearest([0.5, 0.5 + 0.23])
        d = pullback_geodesic_matrix(nodes, decode, period=1.0, k=8, sources=[src])[0]
        assert d[tgt_x] > d[tgt_y]
        assert d[tgt_x] > 2.0 * d[tgt_y]      # ~3x stretch, comfortably > 2x


class TestDispatcher:
    def test_flat_route_matches_direct(self):
        nodes = torus_grid(6, period=1.0)
        np.testing.assert_allclose(
            torus_distance(nodes, 'flat', period=1.0),
            flat_torus_distance_matrix(nodes, period=1.0))

    def test_missing_inputs_and_unknown_method_raise(self):
        nodes = torus_grid(6, period=1.0)
        with pytest.raises(ValueError):
            torus_distance(nodes, 'density_geodesic')            # no density
        with pytest.raises(ValueError):
            torus_distance(nodes, 'pullback_geodesic')           # no decode_fn / tensors
        with pytest.raises(ValueError):
            torus_distance(nodes, 'not_a_method')


class TestPerEventHelpers:
    def test_snap_to_grid_hits_own_nodes_and_wraps(self):
        """Every grid node snaps to itself, and a point just below the seam
        snaps to the last (nearest) node, not across the wrap."""

        n = 10
        grid = torus_grid(n, period=1.0)
        np.testing.assert_array_equal(_snap_to_grid(grid, n, 1.0), np.arange(n * n))
        near_seam = _snap_to_grid(np.array([[1.0 - 1e-6, 0.5]]), n, 1.0)[0]
        assert near_seam // n == n - 1

    def test_context_shapes_and_pullback_gate(self):
        """The context carries a grid, node densities, and the density matrix;
        the pullback matrix is present only when a decoder is supplied."""

        rng = np.random.default_rng(3)
        emb = rng.random((300, 2))
        ctx = build_torus_geodesic_context(emb, n_per_dim=12, period=1.0, k=8)
        assert ctx.grid.shape == (144, 2)
        assert ctx.density.shape == (144,)
        assert ctx.density_matrix.shape == (144, 144)
        assert ctx.pullback_matrix is None
        a = jnp.array([[2.0, 0.0], [0.0, 1.0]])
        ctx2 = build_torus_geodesic_context(emb, decode_fn=lambda z: a @ z,
                                            n_per_dim=12, period=1.0, k=8)
        assert ctx2.pullback_matrix.shape == (144, 144)

    def test_per_event_zero_when_pred_equals_true(self):
        """A prediction equal to the truth snaps to the same node, so its
        geodesic error is exactly zero under both geometries."""

        rng = np.random.default_rng(4)
        emb = rng.random((300, 2))
        a = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        ctx = build_torus_geodesic_context(emb, decode_fn=lambda z: a @ z,
                                           n_per_dim=15, period=1.0, k=8)
        pts = rng.random((40, 2))
        assert np.allclose(
            per_event_geodesic_error(pts, pts, ctx, method='density_geodesic'), 0.0)
        assert np.allclose(
            per_event_geodesic_error(pts, pts, ctx, method='pullback_geodesic'), 0.0)

    def test_mae_columns_and_nan_without_decoder(self):
        """`geodesic_mae_columns` returns both keys; the pullback column is NaN
        when the context has no decoder, while density stays finite."""

        rng = np.random.default_rng(5)
        emb = rng.random((300, 2))
        yp, yt = rng.random((50, 2)), rng.random((50, 2))
        ctx_no_dec = build_torus_geodesic_context(emb, n_per_dim=12, period=1.0, k=8)
        cols = geodesic_mae_columns(yp, yt, ctx_no_dec)
        assert set(cols) == {'density_geodesic_mae', 'pullback_geodesic_mae'}
        assert np.isfinite(cols['density_geodesic_mae'])
        assert np.isnan(cols['pullback_geodesic_mae'])
        a = jnp.array([[3.0, 0.0], [0.0, 1.0]])
        ctx_dec = build_torus_geodesic_context(emb, decode_fn=lambda z: a @ z,
                                               n_per_dim=12, period=1.0, k=8)
        assert np.isfinite(geodesic_mae_columns(yp, yt, ctx_dec)['pullback_geodesic_mae'])

    def test_none_context_yields_nan_columns(self):
        """A ``None`` context (geometry unavailable / disabled) yields NaN for
        both columns without raising, so callers can add them unconditionally."""

        rng = np.random.default_rng(6)
        yp, yt = rng.random((10, 2)), rng.random((10, 2))
        assert np.all(np.isnan(
            per_event_geodesic_error(yp, yt, None, method='density_geodesic')))
        cols = geodesic_mae_columns(yp, yt, None)
        assert np.isnan(cols['density_geodesic_mae'])
        assert np.isnan(cols['pullback_geodesic_mae'])



class TestModelCellDecoder:
    """The pullback decoder built from a QLVM model package cell (the v3 route)."""

    def test_matches_the_embedding_decoder(self, tmp_path):
        """The cell's decode function reproduces, on lattice points, exactly the
        decoder call the embedding makes (``decoder_forward`` of the torus basis of
        ``lattice % 1``, as in ``qlvm_model.embed_data``) and the lattice atlas;
        it is periodic in z and has a finite pullback metric."""

        rng = np.random.default_rng(31)
        grid = np.ones((8, 8), dtype=np.int64)
        cell = _make_model_cell(tmp_path, rng, masking_type="none", floor=0.2, fine_grid=grid, coarse_grid=grid)
        decode_fn = make_qlvm_decode_fn_from_model_cell(str(cell))
        model = load_model_cell(str(cell))
        points = np.asarray(model['lattice'])[[0, 1, 7, 20]]
        embedding_call = np.asarray(
            decoder_forward(torus_basis_forward(jnp.asarray(points) % 1), model['params'])).reshape(len(points), -1)
        atlas = np.asarray(decode_lattice_atlas(jnp.asarray(points), model['params'])).reshape(len(points), -1)
        mine = np.stack([np.asarray(decode_fn(jnp.asarray(point))) for point in points])
        assert mine.shape == (4, 128 * 128)
        np.testing.assert_allclose(mine, embedding_call, atol=1e-6)
        np.testing.assert_allclose(mine, atlas, atol=1e-6)
        np.testing.assert_allclose(np.asarray(decode_fn(jnp.asarray(points[2] + 1.0))), mine[2], atol=1e-6)
        assert np.isfinite(pullback_metric_at_nodes(points[:2], decode_fn)).all()

    def test_conditional_cell_is_refused(self, tmp_path):
        """A conditional cell's decoder is not a function of z alone, so without a
        condition quantile that fixes its conditioning value it is refused."""

        rng = np.random.default_rng(32)
        grid = np.ones((8, 8), dtype=np.int64)
        cell = _make_model_cell(tmp_path, rng, masking_type="none", floor=0.2, fine_grid=grid, coarse_grid=grid,
                                condition={"name": "bandwidth", "decode": "exact"},
                                bins=_phase11_bins("bandwidth", 0.05, 0.95, 0.01))
        with pytest.raises(ValueError, match="conditional cell"):
            make_qlvm_decode_fn_from_model_cell(str(cell))

    def test_conditional_cell_decodes_at_the_corpus_quantile(self, tmp_path):
        """With a condition quantile, a conditional cell decodes every z at the one
        conditioning value that quantile of its training distribution gives (here
        the exact-decode median 0.5 of edges 0.05 / 0.5 / 0.95 with equal rows),
        appended to the torus basis exactly as the embedding appends it; the
        pullback metric is finite and differs from a decode at another quantile."""

        rng = np.random.default_rng(34)
        grid = np.ones((8, 8), dtype=np.int64)
        bins = _phase11_bins("bandwidth", 0.05, 0.95, 0.01)
        cell = _make_model_cell(tmp_path, rng, masking_type="none", floor=0.2, fine_grid=grid, coarse_grid=grid,
                                condition={"name": "bandwidth", "decode": "exact"}, bins=bins)
        decode_fn = make_qlvm_decode_fn_from_model_cell(str(cell), condition_quantile=0.5)
        model = load_model_cell(str(cell))
        c = condition_quantile_value(model['condition_bins'], "exact", 0.5)
        assert float(c) == pytest.approx(0.5, abs=1e-6)
        z = jnp.asarray([0.3, 0.7])
        basis = jnp.concatenate([torus_basis_forward(z[None, :]), jnp.full((1, 1), c)], axis=-1)
        expected = np.asarray(decoder_forward(basis, model['params'])).reshape(-1)
        np.testing.assert_allclose(np.asarray(decode_fn(z)), expected, atol=1e-6)
        nodes = np.array([[0.3, 0.7], [0.6, 0.2]])
        median_metric = pullback_metric_at_nodes(nodes, decode_fn)
        assert np.isfinite(median_metric).all()
        low_fn = make_qlvm_decode_fn_from_model_cell(str(cell), condition_quantile=0.0)
        assert not np.allclose(pullback_metric_at_nodes(nodes, low_fn), median_metric)

    def test_source_resolution(self):
        """With pullback_metric on, the decoder is the production cell of the map the
        manifold columns name, from the os_utils constants (no settings path the
        experimenter re-keying could rewrite): the regular, a conditional or the
        squeak cell, carried with the condition quantile; off, there is no pullback
        decoder. A path key a user's older settings may still carry
        (decoder_model_cell_directory, the retired decoder_weights_npz_path) is
        ignored; a block without pullback_metric or pullback_condition_quantile is a
        settings error, as are columns naming no QLVM map and a quantile outside
        [0, 1]."""

        on = {'pullback_metric': True, 'pullback_condition_quantile': 0.5}
        regular = f"{os_utils.QLVM_MODEL_PACKAGE_ROOT}/{os_utils.QLVM_PRODUCTION_MODEL_CELLS['qlvm']}"
        duration = f"{os_utils.QLVM_MODEL_PACKAGE_ROOT}/{os_utils.QLVM_PRODUCTION_MODEL_CELLS['qlvm_duration']}"
        squeak = f"{os_utils.QLVM_SQUEAK_PACKAGE_ROOT}/{os_utils.QLVM_SQUEAK_PRODUCTION_CELL}"
        assert resolve_geodesic_decoder_source(on, ['qlvm1', 'qlvm2']) == ('model_cell', regular, 0.5)
        assert resolve_geodesic_decoder_source(
            on, ['qlvm_duration1', 'qlvm_duration2']) == ('model_cell', duration, 0.5)
        assert resolve_geodesic_decoder_source(
            {**on, 'pullback_condition_quantile': 0.25},
            ['qlvm_squeak1', 'qlvm_squeak2']) == ('model_cell', squeak, 0.25)
        for prefix in os_utils.QLVM_MAPS:
            source = resolve_geodesic_decoder_source(on, [f"{prefix}1", f"{prefix}2"])
            assert source[1] == os_utils.qlvm_production_cell_directory(prefix)
        assert resolve_geodesic_decoder_source({'pullback_metric': False}, ['qlvm1', 'qlvm2']) is None
        assert resolve_geodesic_decoder_source({'pullback_metric': False}, ['vae1', 'vae2']) is None
        assert resolve_geodesic_decoder_source(
            {**on, 'decoder_model_cell_directory': '/elsewhere'}, ['qlvm1', 'qlvm2']) == ('model_cell', regular, 0.5)
        with pytest.raises(KeyError):
            resolve_geodesic_decoder_source({'decoder_model_cell_directory': '/c'}, ['qlvm1', 'qlvm2'])
        with pytest.raises(KeyError):
            resolve_geodesic_decoder_source({'pullback_metric': True}, ['qlvm1', 'qlvm2'])
        with pytest.raises(ValueError, match="has no decoder"):
            resolve_geodesic_decoder_source(on, ['vae1', 'vae2'])
        with pytest.raises(ValueError, match="must be in"):
            resolve_geodesic_decoder_source({**on, 'pullback_condition_quantile': -0.1}, ['qlvm1', 'qlvm2'])

    def test_manifold_columns_prefer_the_pickle_record(self):
        """The input pickle's recorded manifold columns win over the settings; a
        pickle without that record falls back to the current setting."""

        vocal = {'usv_manifold_column_names': ['qlvm1', 'qlvm2']}
        recorded = {'analysis_specific': {'usv_manifold_column_names': ['qlvm_entropy1', 'qlvm_entropy2']}}
        assert resolve_manifold_column_names(recorded, vocal) == ['qlvm_entropy1', 'qlvm_entropy2']
        assert resolve_manifold_column_names(None, vocal) == ['qlvm1', 'qlvm2']
        assert resolve_manifold_column_names({'analysis_specific': {}}, vocal) == ['qlvm1', 'qlvm2']
        assert resolve_manifold_column_names({}, vocal) == ['qlvm1', 'qlvm2']

    def test_source_builder_uses_the_cell(self, tmp_path):
        """Building from a model-cell source gives the cell's decoder; any other
        source kind (including the retired 'npz') is refused."""

        rng = np.random.default_rng(33)
        grid = np.ones((8, 8), dtype=np.int64)
        cell = _make_model_cell(tmp_path, rng, masking_type="none", floor=0.2, fine_grid=grid, coarse_grid=grid)
        z = jnp.asarray([0.3, 0.7])
        from_source = np.asarray(make_qlvm_decode_fn_from_source(('model_cell', str(cell), 0.5))(z))
        from_cell = np.asarray(make_qlvm_decode_fn_from_model_cell(str(cell))(z))
        np.testing.assert_allclose(from_source, from_cell, atol=1e-7)
        for kind in ('npz', 'zip'):
            with pytest.raises(ValueError, match="unknown decoder source"):
                make_qlvm_decode_fn_from_source((kind, 'x', 0.5))
