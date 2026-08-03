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

from usv_playpen.modeling.manifold_metric import _geodesic_distance_matrix
from usv_playpen.modeling.modeling_torus_geodesics import (
    _snap_to_grid,
    build_torus_geodesic_context,
    density_geodesic_matrix,
    flat_torus_distance_matrix,
    geodesic_mae_columns,
    per_event_geodesic_error,
    pullback_geodesic_matrix,
    pullback_metric_at_nodes,
    torus_distance,
    torus_grid,
    torus_kde_density,
)


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
