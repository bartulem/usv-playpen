"""
@author: bartulem
Unit tests for the cluster-geometry helpers
(`derive_cluster_centers_empirically`, `derive_cluster_geometry`,
`usv_in_circle`) that back the CNN saliency phase and any other
region-conditional analysis on the USV manifold.

The tests synthesise per-cluster point clouds with KNOWN centres on both
the Euclidean plane and the unit torus, so the recovered geometry can be
checked against analytic expectations rather than against a frozen
snapshot. Wrap-boundary cases are exercised explicitly because the
torus codepath uses 3x3 lattice replication and any regression there is
silent on standard fixtures.
"""

from __future__ import annotations

import numpy as np
import pytest

from usv_playpen.modeling.acoustic_manifold_geometry import (
    derive_cluster_centers_empirically,
    derive_cluster_geometry,
    usv_in_circle,
)


# ---------------------------------------------------------------------------
# Synthetic-data builders
# ---------------------------------------------------------------------------


def _make_euclidean_clusters(rng_seed: int = 0,
                             scale: float = 0.05,
                             n_per_cluster: int = 500) -> tuple[np.ndarray, np.ndarray, dict]:
    """Build a (Y, labels, true_centres) fixture of four tight Gaussian
    blobs on the Euclidean plane plus a sparse noise label (label 0)."""

    rng = np.random.default_rng(rng_seed)
    true_centres = {
        1: np.array([0.20, 0.20]),
        2: np.array([0.80, 0.20]),
        3: np.array([0.20, 0.80]),
        4: np.array([0.80, 0.80]),
    }
    Y_chunks, label_chunks = [], []
    for lbl, c in true_centres.items():
        Y_chunks.append(rng.normal(loc=c, scale=scale, size=(n_per_cluster, 2)))
        label_chunks.append(np.full(n_per_cluster, lbl, dtype=np.int64))
    # Noise label 0 — sparse, should be dropped by default.
    Y_chunks.append(rng.uniform(0.0, 1.0, size=(30, 2)))
    label_chunks.append(np.zeros(30, dtype=np.int64))

    Y = np.concatenate(Y_chunks, axis=0)
    labels = np.concatenate(label_chunks, axis=0)
    return Y, labels, true_centres


def _make_torus_wrap_cluster(rng_seed: int = 0,
                             scale: float = 0.04,
                             period: float = 1.0,
                             n_per_cluster: int = 500) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a single cluster whose true centre sits at the origin (a
    wrap-boundary corner of the unit torus). Returns `(Y, labels, true_centre)`.

    Roughly half of the synthesised points end up with each coordinate
    just above 0 and half just below `period` after the modulo wrap, so
    a naive (non-wrap-aware) KDE on the canonical cell would put the
    estimated mode near `period/2` — the antipode of the true centre.
    """

    rng = np.random.default_rng(rng_seed)
    true_centre = np.array([0.0, 0.0])
    raw = rng.normal(loc=true_centre, scale=scale, size=(n_per_cluster, 2))
    wrapped = np.mod(raw, period)
    labels = np.ones(n_per_cluster, dtype=np.int64)
    return wrapped, labels, true_centre


# ---------------------------------------------------------------------------
# derive_cluster_centers_empirically
# ---------------------------------------------------------------------------


class TestDeriveClusterCenters:

    def test_euclidean_recovers_known_centres_and_drops_noise(self):
        """KDE-mode centres on tight Gaussian blobs should land within a
        small tolerance of the true centres, and label 0 should NOT appear
        in the returned dict because `drop_label=0` is the default."""

        Y, labels, true_centres = _make_euclidean_clusters()
        out = derive_cluster_centers_empirically(
            Y, labels,
            metric='euclidean',
            grid_resolution=300,
        )

        assert set(out.keys()) == set(true_centres.keys()), (
            f"expected labels {sorted(true_centres.keys())}, got {sorted(out.keys())}"
        )
        for lbl, c_true in true_centres.items():
            c_est = out[lbl]
            assert c_est.shape == (2,)
            err = float(np.linalg.norm(c_est - c_true))
            assert err < 0.02, f"label {lbl}: centre estimate off by {err:.4f}"

    def test_drop_label_none_keeps_every_label(self):
        """Passing `drop_label=None` should retain label 0 in the output
        (provided it has enough points)."""

        rng = np.random.default_rng(0)
        true_centres = {0: np.array([0.3, 0.3]), 1: np.array([0.7, 0.7])}
        Y_chunks, label_chunks = [], []
        for lbl, c in true_centres.items():
            Y_chunks.append(rng.normal(loc=c, scale=0.04, size=(400, 2)))
            label_chunks.append(np.full(400, lbl, dtype=np.int64))
        Y = np.concatenate(Y_chunks, axis=0)
        labels = np.concatenate(label_chunks, axis=0)

        out = derive_cluster_centers_empirically(
            Y, labels, drop_label=None, grid_resolution=300,
        )
        assert set(out.keys()) == {0, 1}

    def test_skip_labels_below_min_points(self):
        """Labels with fewer than `min_points_per_label` finite points
        should be silently dropped — they cannot support a KDE."""

        rng = np.random.default_rng(0)
        Y_dense = rng.normal(loc=[0.5, 0.5], scale=0.05, size=(400, 2))
        Y_sparse = rng.normal(loc=[0.1, 0.1], scale=0.05, size=(5, 2))
        Y = np.vstack([Y_dense, Y_sparse])
        labels = np.concatenate([np.full(400, 7), np.full(5, 8)])

        out = derive_cluster_centers_empirically(
            Y, labels, min_points_per_label=20, grid_resolution=200,
        )
        assert 7 in out
        assert 8 not in out

    def test_nan_labels_are_ignored(self):
        """NaN-valued labels must not appear as a cluster in the output."""

        rng = np.random.default_rng(0)
        Y = rng.normal(loc=[0.5, 0.5], scale=0.05, size=(400, 2))
        labels = np.full(400, 3.0, dtype=np.float64)
        # Inject NaN labels — these are not biological vocalisations and
        # should be skipped, not crash the KDE fit.
        nan_extra = rng.uniform(0.0, 1.0, size=(50, 2))
        Y = np.vstack([Y, nan_extra])
        labels = np.concatenate([labels, np.full(50, np.nan)])

        out = derive_cluster_centers_empirically(
            Y, labels, drop_label=0, grid_resolution=200,
        )
        assert set(out.keys()) == {3.0}

    def test_torus_wrap_boundary_recovers_origin(self):
        """A cluster whose true centre is at the wrap-boundary corner
        `(0, 0)` should be recovered near `(0, 0)` (or, equivalently,
        near `(period, period)`) on torus — but never near the antipode
        `(period/2, period/2)` which is what a naive KDE on the canonical
        cell would produce."""

        period = 1.0
        Y, labels, _ = _make_torus_wrap_cluster(period=period)
        out = derive_cluster_centers_empirically(
            Y, labels,
            drop_label=None,
            metric='torus',
            period=period,
            grid_resolution=400,
        )
        c_est = out[1]
        # Wrap-aware distance to the true centre at the origin must be
        # small — measure under the torus metric so a recovery at
        # `(0.98, 0.02)` counts as close to `(0, 0)`.
        d_axis = np.minimum(c_est, period - c_est)
        wrap_aware_err = float(np.linalg.norm(d_axis))
        assert wrap_aware_err < 0.05, (
            f"torus KDE missed the wrap-boundary cluster: estimate={c_est}, "
            f"wrap-aware error={wrap_aware_err:.4f}"
        )

    def test_torus_wrap_recovers_seam_with_recommended_bandwidth(self):
        """Regression: with the docstring-recommended ``kde_bandwidth=0.8`` a
        TIGHT seam-straddling cluster must still recover the seam centre. Fitting
        the KDE on the 9x lattice-replicated cloud (the previous approach) made
        ``gaussian_kde`` infer a period-scale bandwidth, over-smoothing the density
        so its peak collapsed to the cell centre ``(0.5, 0.5)``."""

        period = 1.0
        Y, labels, _ = _make_torus_wrap_cluster(scale=0.03, period=period)
        out = derive_cluster_centers_empirically(
            Y, labels,
            drop_label=None,
            metric='torus',
            period=period,
            kde_bandwidth=0.8,
            grid_resolution=400,
        )
        c_est = out[1]
        d_axis = np.minimum(c_est % period, period - (c_est % period))
        wrap_aware_err = float(np.linalg.norm(d_axis))
        # The cell-centre collapse would give err ~0.7; require seam recovery.
        assert wrap_aware_err < 0.1, (
            f"seam cluster collapsed with bw=0.8: estimate={c_est}, "
            f"wrap-aware error={wrap_aware_err:.4f}"
        )

    def test_validates_metric_and_period(self):
        """`metric` must be one of `'euclidean'` / `'torus'` and `period`
        must be positive when torus — the shared validator raises."""

        rng = np.random.default_rng(0)
        Y = rng.normal(loc=[0.5, 0.5], scale=0.05, size=(200, 2))
        labels = np.full(200, 1)
        with pytest.raises(ValueError, match='manifold_metric must be one of'):
            derive_cluster_centers_empirically(Y, labels, metric='spherical')


# ---------------------------------------------------------------------------
# derive_cluster_geometry
# ---------------------------------------------------------------------------


class TestDeriveClusterGeometry:

    def test_adaptive_radii_match_alpha_dnn_over_two(self):
        """In adaptive mode each cluster's radius must equal
        `alpha * d_nn(i) / 2`."""

        centres = {
            'a': np.array([0.0, 0.0]),
            'b': np.array([1.0, 0.0]),
            'c': np.array([0.0, 3.0]),
        }
        alpha = 0.5
        geo = derive_cluster_geometry(
            centres, alpha=alpha, mode='adaptive', metric='euclidean',
        )
        # Nearest neighbour distances: a->b=1, b->a=1, c->a=3.
        expected_dnn = {'a': 1.0, 'b': 1.0, 'c': 3.0}
        for k, dnn in expected_dnn.items():
            assert geo[k]['nearest_neighbour_distance'] == pytest.approx(dnn)
            assert geo[k]['radius'] == pytest.approx(alpha * dnn / 2.0)

    def test_uniform_radius_is_global_minimum(self):
        """Uniform mode gives every cluster the same radius sized by the
        globally tightest pair: `alpha * min_i d_nn(i) / 2`."""

        centres = {
            'a': np.array([0.0, 0.0]),
            'b': np.array([1.0, 0.0]),
            'c': np.array([0.0, 3.0]),
        }
        alpha = 0.5
        geo = derive_cluster_geometry(
            centres, alpha=alpha, mode='uniform', metric='euclidean',
        )
        expected_radius = alpha * 1.0 / 2.0  # min d_nn is 1.0
        radii = [geo[k]['radius'] for k in centres.keys()]
        assert all(r == pytest.approx(expected_radius) for r in radii)
        # `nearest_neighbour_distance` should still be the per-centre raw
        # value for diagnostic purposes.
        assert geo['c']['nearest_neighbour_distance'] == pytest.approx(3.0)

    def test_alpha_one_produces_tangent_no_overlap(self):
        """With `alpha = 1.0` every pair of adjacent circles is tangent
        (the sum of their radii equals their centre-to-centre distance);
        for any pair the sum of radii must not exceed the distance."""

        centres = {
            0: np.array([0.0, 0.0]),
            1: np.array([1.0, 0.0]),
            2: np.array([0.0, 1.0]),
            3: np.array([2.0, 2.0]),
        }
        geo = derive_cluster_geometry(centres, alpha=1.0, mode='adaptive')
        labels = list(centres.keys())
        for i, li in enumerate(labels):
            for j in range(i + 1, len(labels)):
                lj = labels[j]
                d = float(np.linalg.norm(centres[li] - centres[lj]))
                r_sum = geo[li]['radius'] + geo[lj]['radius']
                assert r_sum <= d + 1e-9, (
                    f"pair {(li, lj)} overlap: d={d:.4f}, r_sum={r_sum:.4f}"
                )

    def test_max_radius_clamps(self):
        """Per-cluster radii must be capped at `max_radius` when supplied."""

        centres = {
            'a': np.array([0.0, 0.0]),
            'b': np.array([10.0, 0.0]),
        }
        # Without cap the adaptive radius would be 0.5 * 10 / 2 = 2.5.
        geo = derive_cluster_geometry(
            centres, alpha=0.5, mode='adaptive', max_radius=1.0,
        )
        assert geo['a']['radius'] == pytest.approx(1.0)
        assert geo['b']['radius'] == pytest.approx(1.0)
        # nearest_neighbour_distance is preserved verbatim even after clamp.
        assert geo['a']['nearest_neighbour_distance'] == pytest.approx(10.0)

    def test_torus_metric_picks_wrap_neighbour(self):
        """On a unit torus the nearest neighbour of `(0.05, 0.5)` is
        `(0.95, 0.5)` (distance 0.10 via wrap), not `(0.5, 0.5)`
        (distance 0.45 on the canonical cell)."""

        centres = {
            'a': np.array([0.05, 0.5]),
            'b': np.array([0.50, 0.5]),
            'c': np.array([0.95, 0.5]),
        }
        geo = derive_cluster_geometry(
            centres, alpha=1.0, mode='adaptive', metric='torus', period=1.0,
        )
        # On torus, a-c distance is 0.10; a-b is 0.45.
        assert geo['a']['nearest_neighbour_distance'] == pytest.approx(0.10, abs=1e-9)
        assert geo['c']['nearest_neighbour_distance'] == pytest.approx(0.10, abs=1e-9)

    def test_raises_on_invalid_alpha(self):
        centres = {'a': np.array([0.0, 0.0]), 'b': np.array([1.0, 0.0])}
        with pytest.raises(ValueError):
            derive_cluster_geometry(centres, alpha=0.0)
        with pytest.raises(ValueError):
            derive_cluster_geometry(centres, alpha=-0.5)

    def test_raises_on_invalid_mode(self):
        centres = {'a': np.array([0.0, 0.0]), 'b': np.array([1.0, 0.0])}
        with pytest.raises(ValueError):
            derive_cluster_geometry(centres, alpha=0.5, mode='quadratic')

    def test_raises_on_fewer_than_two_centres(self):
        with pytest.raises(ValueError):
            derive_cluster_geometry({'a': np.array([0.0, 0.0])}, alpha=0.5)


# ---------------------------------------------------------------------------
# usv_in_circle
# ---------------------------------------------------------------------------


class TestUsvInCircle:

    def test_euclidean_inside_outside_boundary(self):
        """Boundary points (`||y - c|| == r`) are inside (closed disk);
        strictly farther points are outside."""

        Y = np.array([
            [0.5, 0.5],  # at centre
            [0.6, 0.5],  # inside  (d=0.10)
            [0.7, 0.5],  # on boundary (d=0.20)
            [0.8, 0.5],  # outside (d=0.30)
            [0.5, 1.5],  # far outside
        ])
        mask = usv_in_circle(Y, np.array([0.5, 0.5]), radius=0.2)
        assert mask.tolist() == [True, True, True, False, False]

    def test_radius_zero_is_all_false(self):
        """A non-positive radius collapses to an empty membership set."""

        Y = np.array([[0.0, 0.0], [0.1, 0.1]])
        assert not usv_in_circle(Y, np.array([0.0, 0.0]), radius=0.0).any()
        assert not usv_in_circle(Y, np.array([0.0, 0.0]), radius=-1.0).any()

    def test_torus_wraps_distance(self):
        """A point at `(0.98, 0.5)` is at wrap-distance 0.07 from a
        centroid at `(0.05, 0.5)` on the unit torus, well inside a
        radius-0.1 circle — even though the canonical-cell distance
        is 0.93."""

        Y = np.array([
            [0.98, 0.50],  # wrap-distance 0.07 -> inside
            [0.15, 0.50],  # straight distance 0.10 -> on boundary
            [0.50, 0.50],  # straight distance 0.45 -> outside
        ])
        mask = usv_in_circle(
            Y, np.array([0.05, 0.50]), radius=0.10,
            metric='torus', period=1.0,
        )
        assert mask.tolist() == [True, True, False]

    def test_membership_shape_matches_input(self):
        """The returned mask must be 1-D with length matching `Y.shape[0]`."""

        Y = np.random.default_rng(0).uniform(0.0, 1.0, size=(123, 2))
        mask = usv_in_circle(Y, np.array([0.5, 0.5]), radius=0.3)
        assert mask.shape == (123,)
        assert mask.dtype == bool
