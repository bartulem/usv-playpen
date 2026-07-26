"""
@author: bartulem
Unit tests for the manifold-metric helpers in
``usv_playpen.modeling.manifold_metric`` — the single source of truth for
distance, centroid, dispersion, and (sin, cos) encoding on the two
supported USV manifolds (flat ``'euclidean'`` and periodic ``'torus'``).

The tests assert analytic invariants rather than frozen snapshots:
euclidean reductions must collapse to the textbook arithmetic mean /
L2 norm, while the torus path must respect the wrap. The headline torus
case is the circular mean of two points straddling the seam
(``{0.05, 0.95}`` on a unit period), whose correct centre is ``0.0`` even
though the arithmetic mean is ``0.5`` (the antipode). The JAX mirrors are
checked for numerical agreement with the NumPy versions, and the
``sin_cos`` encode/decode pair is checked for round-trip identity.
"""

from __future__ import annotations

import numpy as np
import pytest

from usv_playpen.modeling.manifold_metric import (
    _validate_metric_period,
    angle_decode_jax,
    circular_mean,
    dcor_prediction_truth,
    distance_correlation,
    manifold_prediction_metrics,
    pairwise_distance,
    resolve_manifold_metric,
    signed_diff,
    signed_diff_jax,
    sin_cos_encode_jax,
    torus_embed,
    total_dispersion,
    _fit_von_mises_kappa,
    macro_von_mises_logscore,
    inverse_region_frequency_weights,
)


# Validation


class TestValidateMetricPeriod:

    def test_accepts_supported_metrics(self):
        """Both canonical tags with a positive finite period must pass
        silently (the helper returns ``None`` and raises nothing)."""

        assert _validate_metric_period('euclidean', 1.0) is None
        assert _validate_metric_period('torus', 2.5) is None

    def test_rejects_unknown_metric(self):
        """An unsupported metric tag is a settings bug, not a soft
        fallback — the helper raises ``ValueError``."""

        with pytest.raises(ValueError):
            _validate_metric_period('spherical', 1.0)

    @pytest.mark.parametrize("bad_period", [0.0, -1.0, np.inf, np.nan])
    def test_rejects_non_positive_or_nonfinite_period(self, bad_period):
        """A non-positive or non-finite period is rejected even on
        euclidean, so a caller cannot mis-specify on a euclidean run and
        silently flip to torus later."""

        with pytest.raises(ValueError):
            _validate_metric_period('torus', bad_period)


# signed_diff


class TestSignedDiff:

    def test_euclidean_is_plain_difference(self):
        """On euclidean the signed difference is exactly ``a - b``."""

        a = np.array([[0.9, 0.1], [0.2, 0.7]])
        b = np.array([[0.1, 0.9], [0.7, 0.2]])
        out = signed_diff(a, b, metric='euclidean', period=1.0)
        np.testing.assert_allclose(out, a - b)

    def test_torus_takes_shortest_wrap_direction(self):
        """``0.9 - 0.1`` on a unit torus is ``-0.2`` (wrap forward),
        never ``+0.8`` (the long way round)."""

        a = np.array([0.9, 0.1])
        b = np.array([0.1, 0.9])
        out = signed_diff(a, b, metric='torus', period=1.0)
        np.testing.assert_allclose(out, np.array([-0.2, 0.2]), atol=1e-12)

    def test_torus_result_within_half_period_band(self):
        """Every component of the wrap-aware diff lies in the
        ``(-period/2, period/2]`` band."""

        rng = np.random.default_rng(0)
        period = 3.0
        a = rng.uniform(0.0, period, size=(200, 2))
        b = rng.uniform(0.0, period, size=(200, 2))
        out = signed_diff(a, b, metric='torus', period=period)
        assert np.all(out > -period / 2 - 1e-9)
        assert np.all(out <= period / 2 + 1e-9)

    def test_jax_mirror_agrees_with_numpy(self):
        """``signed_diff_jax`` reproduces ``signed_diff`` up to
        floating-point noise on both metrics."""

        rng = np.random.default_rng(1)
        a = rng.uniform(0.0, 1.0, size=(50, 2))
        b = rng.uniform(0.0, 1.0, size=(50, 2))
        for metric in ('euclidean', 'torus'):
            np_out = signed_diff(a, b, metric=metric, period=1.0)
            jax_out = np.asarray(signed_diff_jax(a, b, metric=metric, period=1.0))
            np.testing.assert_allclose(jax_out, np_out, atol=1e-6)


# pairwise_distance


class TestPairwiseDistance:

    def test_euclidean_is_l2_norm(self):
        """Euclidean distance is the row-wise L2 norm of ``a - b``."""

        a = np.array([[0.0, 0.0], [3.0, 0.0]])
        b = np.array([[0.0, 4.0], [0.0, 0.0]])
        out = pairwise_distance(a, b, metric='euclidean', period=10.0)
        np.testing.assert_allclose(out, np.array([4.0, 3.0]))

    def test_torus_uses_shortest_path(self):
        """On a unit torus the distance between ``0.98`` and ``0.02`` per
        axis is ``0.04`` (via the seam), not ``0.96``."""

        a = np.array([[0.98, 0.98]])
        b = np.array([[0.02, 0.02]])
        out = pairwise_distance(a, b, metric='torus', period=1.0)
        np.testing.assert_allclose(out, np.array([np.sqrt(2) * 0.04]), atol=1e-12)

    def test_symmetric(self):
        """``d(a, b) == d(b, a)`` under both metrics."""

        rng = np.random.default_rng(2)
        a = rng.uniform(0.0, 1.0, size=(30, 2))
        b = rng.uniform(0.0, 1.0, size=(30, 2))
        for metric in ('euclidean', 'torus'):
            d_ab = pairwise_distance(a, b, metric=metric, period=1.0)
            d_ba = pairwise_distance(b, a, metric=metric, period=1.0)
            np.testing.assert_allclose(d_ab, d_ba, atol=1e-12)


# circular_mean


class TestCircularMean:

    def test_euclidean_is_arithmetic_mean(self):
        """On euclidean the unweighted centroid is the column mean."""

        Y = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
        out = circular_mean(Y, metric='euclidean', period=10.0)
        np.testing.assert_allclose(out, Y.mean(axis=0))

    def test_euclidean_weighted_mean(self):
        """Weights produce the renormalised weighted mean."""

        Y = np.array([[0.0, 0.0], [10.0, 10.0]])
        w = np.array([3.0, 1.0])
        out = circular_mean(Y, metric='euclidean', period=100.0, weights=w)
        np.testing.assert_allclose(out, np.array([2.5, 2.5]))

    def test_torus_recovers_seam_straddling_centre(self):
        """The circular mean of points straddling the wrap seam sits at
        the seam (~0.0), not at the arithmetic-mean antipode (~0.5)."""

        Y = np.array([[0.05, 0.05], [0.95, 0.95]])
        out = circular_mean(Y, metric='torus', period=1.0)
        # Wrap-aware distance of the estimate to the true centre (0, 0).
        d_axis = np.minimum(out, 1.0 - out)
        assert float(np.linalg.norm(d_axis)) < 1e-9

    def test_torus_result_in_canonical_cell(self):
        """The torus centroid is folded into ``[0, period)``."""

        rng = np.random.default_rng(3)
        period = 2.0
        Y = rng.uniform(0.0, period, size=(100, 2))
        out = circular_mean(Y, metric='torus', period=period)
        assert np.all(out >= 0.0)
        assert np.all(out < period)

    def test_zero_weight_sum_raises(self):
        """A zero (or negative) weight sum cannot be renormalised and
        must raise ``ValueError``."""

        Y = np.array([[0.0, 0.0], [1.0, 1.0]])
        with pytest.raises(ValueError):
            circular_mean(Y, metric='euclidean', period=1.0,
                          weights=np.array([0.0, 0.0]))


# total_dispersion


class TestTotalDispersion:

    def test_single_point_has_zero_dispersion(self):
        """A degenerate one-point cloud sits exactly on its own
        centroid, so the squared dispersion is zero."""

        Y = np.array([[0.3, 0.7]])
        for metric in ('euclidean', 'torus'):
            assert total_dispersion(Y, metric=metric, period=1.0) == pytest.approx(0.0)

    def test_matches_sum_of_squared_distances_euclidean(self):
        """Unweighted euclidean dispersion equals the sum of squared
        distances to the arithmetic centroid."""

        Y = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0]])
        centroid = Y.mean(axis=0)
        expected = float(np.sum((Y - centroid) ** 2))
        out = total_dispersion(Y, metric='euclidean', period=100.0)
        assert out == pytest.approx(expected)

    def test_zero_weight_sum_raises(self):
        """A zero weight sum propagates a ``ValueError`` (the centroid
        call raises first, but a non-uniform path is also guarded)."""

        Y = np.array([[0.0, 0.0], [1.0, 1.0]])
        with pytest.raises(ValueError):
            total_dispersion(Y, metric='euclidean', period=1.0,
                             weights=np.array([0.0, 0.0]))


# sin_cos encode / decode


class TestSinCosEncodeDecode:

    def test_encode_shape_doubles_last_axis(self):
        """Encoding maps ``(..., D)`` to ``(..., 2D)``."""

        y = np.zeros((7, 2))
        enc = np.asarray(sin_cos_encode_jax(y, period=1.0))
        assert enc.shape == (7, 4)

    def test_encode_interleaves_sin_cos_per_axis(self):
        """The encoding layout is ``(s_1, c_1, s_2, c_2)``; at the origin
        every sine is 0 and every cosine is 1."""

        y = np.zeros((1, 2))
        enc = np.asarray(sin_cos_encode_jax(y, period=1.0))[0]
        np.testing.assert_allclose(enc, np.array([0.0, 1.0, 0.0, 1.0]), atol=1e-12)

    def test_round_trip_recovers_coordinates(self):
        """``angle_decode_jax(sin_cos_encode_jax(y))`` returns ``y`` mod
        period for coordinates already in the canonical cell."""

        rng = np.random.default_rng(4)
        period = 1.5
        y = rng.uniform(0.0, period, size=(64, 2))
        dec = np.asarray(angle_decode_jax(sin_cos_encode_jax(y, period), period))
        np.testing.assert_allclose(dec, y, atol=1e-6)

    def test_decode_in_canonical_cell(self):
        """Decoded coordinates are folded into ``[0, period)``."""

        rng = np.random.default_rng(5)
        period = 2.0
        raw = rng.standard_normal((40, 4))
        dec = np.asarray(angle_decode_jax(raw, period))
        assert np.all(dec >= 0.0)
        assert np.all(dec < period)

    def test_decode_is_scale_invariant(self):
        """``atan2`` ignores positive radial scaling, so an
        under-confident output decodes to the same angle as a unit one."""

        period = 1.0
        y = np.array([[0.2, 0.8]])
        enc = np.asarray(sin_cos_encode_jax(y, period))
        dec_unit = np.asarray(angle_decode_jax(enc, period))
        dec_scaled = np.asarray(angle_decode_jax(0.3 * enc, period))
        np.testing.assert_allclose(dec_unit, dec_scaled, atol=1e-6)


# torus_embed


class TestTorusEmbed:

    def test_shape_doubles_last_axis(self):
        """The 4-D embedding maps ``(N, D)`` to ``(N, 2D)``."""

        Y = np.zeros((10, 2))
        emb = torus_embed(Y, period=1.0)
        assert emb.shape == (10, 4)

    def test_identical_points_embed_to_zero_distance(self):
        """Two independently constructed but numerically equal torus points
        map to equal embedded points (so their embedded distance is zero)."""

        point_a = np.array([[0.3, 0.6]])
        point_b = np.array([[0.1 + 0.2, 0.2 + 0.4]])
        np.testing.assert_allclose(point_a, point_b)
        emb_a = torus_embed(point_a, period=1.0)
        emb_b = torus_embed(point_b, period=1.0)
        assert np.allclose(emb_a, emb_b)
        np.testing.assert_allclose(np.linalg.norm(emb_a[0] - emb_b[0]), 0.0, atol=1e-12)

    def test_embedded_distance_monotone_in_torus_distance(self):
        """Embedded Euclidean distance is a strictly monotone function of
        toroidal distance: a wrap-near pair must embed closer than a
        wrap-far pair even when their canonical-cell coordinates suggest
        the opposite."""

        period = 1.0
        base = np.array([[0.0, 0.0]])
        near = np.array([[0.98, 0.0]])   # torus distance 0.02
        far = np.array([[0.40, 0.0]])    # torus distance 0.40
        emb_base = torus_embed(base, period)
        d_near = np.linalg.norm(torus_embed(near, period) - emb_base)
        d_far = np.linalg.norm(torus_embed(far, period) - emb_base)
        assert d_near < d_far

    def test_chord_formula_single_axis(self):
        """For one axis the embedded distance equals the unit-circle
        chord length ``2 |sin(pi * d_torus / period)|``."""

        period = 1.0
        a = np.array([[0.0, 0.0]])
        b = np.array([[0.10, 0.0]])
        emb_d = np.linalg.norm(torus_embed(a, period) - torus_embed(b, period))
        expected = 2.0 * abs(np.sin(np.pi * 0.10 / period))
        assert emb_d == pytest.approx(expected, abs=1e-12)

    def test_rejects_non_positive_period(self):
        """A non-positive period is rejected at the embedding boundary."""

        with pytest.raises(ValueError):
            torus_embed(np.zeros((3, 2)), period=0.0)


# resolve_manifold_metric


class TestResolveManifoldMetric:

    def test_reads_and_validates_settings(self):
        """The resolver reads the two canonical keys and returns the
        ``(metric, period)`` pair with the period coerced to float."""

        settings = {'vocal_features': {'usv_manifold_metric': 'torus',
                                       'usv_manifold_period': 2}}
        metric, period = resolve_manifold_metric(settings)
        assert metric == 'torus'
        assert isinstance(period, float)
        assert period == 2.0

    def test_propagates_validation_error(self):
        """An invalid metric in settings raises through the resolver."""

        settings = {'vocal_features': {'usv_manifold_metric': 'bogus',
                                       'usv_manifold_period': 1.0}}
        with pytest.raises(ValueError):
            resolve_manifold_metric(settings)

    def test_missing_key_raises_keyerror(self):
        """Strict key access (no ``.get`` defaults) means a missing key
        surfaces as ``KeyError``, flagging a settings-file bug."""

        with pytest.raises(KeyError):
            resolve_manifold_metric({'vocal_features': {'usv_manifold_metric': 'torus'}})


# Distance correlation (the torus selection score)


class TestDistanceCorrelation:
    """`distance_correlation` + `dcor_prediction_truth` — the wrap-aware,
    model-based score used for torus feature selection."""

    @staticmethod
    def _wound(n, seed):
        """A target that winds around the torus: each axis is the wrapped
        atan2 of two fixed linear projections of X, so X -> Y is a genuine
        (non-linear, periodic) dependence."""

        g = np.random.default_rng(seed)
        X = g.standard_normal((n, 4))
        w = g.standard_normal((4, 4))
        thx = np.arctan2(X @ w[0], X @ w[1]); thy = np.arctan2(X @ w[2], X @ w[3])
        Y = np.column_stack([(thx / (2 * np.pi)) % 1.0, (thy / (2 * np.pi)) % 1.0])
        return X, Y

    def test_distance_correlation_independent_vs_dependent(self):
        """dCor is near-zero for independent variables and high for a
        perfectly dependent (identical) one; a constant input gives 0."""

        rng = np.random.default_rng(0)
        from usv_playpen.modeling.manifold_metric import _geodesic_distance_matrix
        A = rng.random((300, 2)); B = rng.random((300, 2))
        da = _geodesic_distance_matrix(A, metric='torus', period=1.0)
        db = _geodesic_distance_matrix(B, metric='torus', period=1.0)
        assert distance_correlation(da, da) == pytest.approx(1.0, abs=1e-9)   # self
        assert distance_correlation(da, db) < 0.15                            # independent (+ finite-n bias)
        const = np.zeros((300, 300))
        assert distance_correlation(da, const) == 0.0                         # degenerate -> 0

    def test_dcor_prediction_truth_signal_vs_noise(self):
        """On a wound torus target, a near-perfect prediction scores ~1 and an
        independent prediction scores ~0."""

        _, Y = self._wound(1500, 1)
        rng = np.random.default_rng(2)
        Y_perfect = (Y + 0.01 * rng.standard_normal(Y.shape)) % 1.0
        Y_noise = rng.random(Y.shape)
        d_good = dcor_prediction_truth(Y_perfect, Y, metric='torus', period=1.0, random_state=0)
        d_bad = dcor_prediction_truth(Y_noise, Y, metric='torus', period=1.0, random_state=0)
        assert d_good > 0.9
        assert d_bad < 0.15

    def test_distance_correlation_scale_invariant(self):
        """dCor is invariant to scaling either distance matrix — the property
        that ultimately makes the decoded score insensitive to ridge
        magnitude-shrinkage (the atan2 embedding scale drops out)."""

        rng = np.random.default_rng(3)
        from usv_playpen.modeling.manifold_metric import _geodesic_distance_matrix
        A = _geodesic_distance_matrix(rng.random((250, 2)), metric='euclidean', period=1.0)
        B = _geodesic_distance_matrix(rng.random((250, 2)), metric='euclidean', period=1.0)
        base = distance_correlation(A, B)
        assert distance_correlation(7.0 * A, B) == pytest.approx(base, abs=1e-9)
        assert distance_correlation(A, 0.013 * B) == pytest.approx(base, abs=1e-9)

    def test_dcor_deterministic(self):
        """A fixed `random_state` makes the subsampled estimate reproducible."""

        _, Y = self._wound(1000, 5)
        Yp = (Y + 0.05 * np.random.default_rng(6).standard_normal(Y.shape)) % 1.0
        a = dcor_prediction_truth(Yp, Y, metric='torus', period=1.0, random_state=0)
        b = dcor_prediction_truth(Yp, Y, metric='torus', period=1.0, random_state=0)
        assert a == b


class TestManifoldPredictionMetrics:
    """`manifold_prediction_metrics` — the shared bundle scorer for the
    non-fitted baselines (the `null_model_free` empirical-density draw and the
    forward-selection Step-0 baseline)."""

    _KEYS = {
        'r2_spatial', 'euclidean_mae', 'euclidean_rmse', 'euclidean_mae_weighted',
        'euclidean_mae_raw', 'mahalanobis_mae', 'mae_x', 'mae_y',
        'pearson_x', 'pearson_y', 'spearman_x', 'spearman_y', 'dcor_xy',
        'vm_logscore',
    }

    def test_bundle_keys_and_raw_equals_mae(self):
        """Returns exactly the fitted-strategy key set; with no snap, the raw
        and snapped MAE coincide."""

        rng = np.random.default_rng(0)
        Y = rng.random((200, 2)); Yp = rng.random((200, 2))
        bundle = manifold_prediction_metrics(Y, Yp, metric='euclidean', period=1.0)
        assert set(bundle) == self._KEYS
        assert bundle['euclidean_mae_raw'] == bundle['euclidean_mae']

    def test_perfect_prediction(self):
        """An exact prediction scores `r2_spatial == 1`, zero error, and unit
        per-axis correlation. On the torus the selection score is `vm_logscore`
        (the von Mises log-likelihood), which an exact prediction MAXIMISES (a
        large, finite value -- not pinned at 1.0), so it must comfortably exceed a
        shuffled prediction's score; `dcor_xy` is `nan` on the torus."""

        rng = np.random.default_rng(1)
        Y = rng.random((300, 2))
        bundle = manifold_prediction_metrics(Y, Y.copy(), metric='torus', period=1.0)
        assert bundle['r2_spatial'] == pytest.approx(1.0, abs=1e-9)
        assert bundle['euclidean_mae'] == pytest.approx(0.0, abs=1e-9)
        assert bundle['pearson_x'] == pytest.approx(1.0, abs=1e-9)
        assert np.isnan(bundle['dcor_xy'])          # dcor is not computed on the torus
        shuffled = Y[rng.permutation(len(Y))]
        vm_shuffled = manifold_prediction_metrics(Y, shuffled, metric='torus', period=1.0)['vm_logscore']
        assert np.isfinite(bundle['vm_logscore'])
        assert bundle['vm_logscore'] > vm_shuffled

    def test_selection_score_key_by_geometry(self):
        """The selection score lives under a geometry-specific, honestly-named
        key: `dcor_xy` (real distance correlation) on euclidean with `vm_logscore`
        NaN, and `vm_logscore` (von Mises log-lik) on the torus with `dcor_xy`
        NaN."""

        rng = np.random.default_rng(2)
        Y = rng.random((200, 2)); Yp = rng.random((200, 2))
        euc = manifold_prediction_metrics(Y, Yp, metric='euclidean', period=1.0)
        assert np.isfinite(euc['dcor_xy'])
        assert np.isnan(euc['vm_logscore'])
        tor = manifold_prediction_metrics(Y, Yp, metric='torus', period=1.0)
        assert np.isfinite(tor['vm_logscore'])
        assert np.isnan(tor['dcor_xy'])

    def test_density_draw_correlations_are_finite(self):
        """A uniform draw has real geometry, so the per-axis correlations and the
        torus selection score `vm_logscore` are defined finite-sample chance
        values — unlike the old constant-centroid baseline, whose correlations
        were NaN."""

        rng = np.random.default_rng(3)
        Y = rng.random((400, 2))
        draw = Y[rng.choice(len(Y), size=len(Y), replace=True)]
        bundle = manifold_prediction_metrics(Y, draw, metric='torus', period=1.0)
        for key in ('pearson_x', 'pearson_y', 'spearman_x', 'spearman_y', 'vm_logscore'):
            assert np.isfinite(bundle[key])

    def test_mahalanobis_requires_cov(self):
        """`mahalanobis_mae` is `nan` without a training inverse-covariance and
        finite once one is supplied."""

        rng = np.random.default_rng(4)
        Y = rng.random((100, 2)); Yp = rng.random((100, 2))
        assert np.isnan(
            manifold_prediction_metrics(Y, Yp, metric='euclidean', period=1.0)['mahalanobis_mae']
        )
        assert np.isfinite(
            manifold_prediction_metrics(
                Y, Yp, metric='euclidean', period=1.0, train_cov_inv=np.eye(2),
            )['mahalanobis_mae']
        )


class TestFitVonMisesKappa:
    """The global von Mises concentration MLE (`_fit_von_mises_kappa`)."""

    def test_recovers_known_concentration(self):
        """Sampling residuals from a von Mises(0, kappa) and re-fitting recovers
        kappa to within sampling tolerance."""
        rng = np.random.default_rng(0)
        for true_kappa in (0.5, 4.0, 20.0):
            residuals = rng.vonmises(0.0, true_kappa, size=40000)
            est = _fit_von_mises_kappa(residuals)
            assert est == pytest.approx(true_kappa, rel=0.15)

    def test_zero_concentration_for_uniform_residuals(self):
        """Uniform (fully dispersed) residuals have ~zero mean resultant length,
        so the MLE returns 0 (no concentration)."""
        rng = np.random.default_rng(1)
        uniform = rng.uniform(-np.pi, np.pi, size=40000)
        assert _fit_von_mises_kappa(uniform) == pytest.approx(0.0, abs=0.05)

    def test_saturates_for_near_perfect_residuals(self):
        """Essentially-zero residuals (a near-perfect fit) saturate kappa at the
        numerical upper bound rather than diverging."""
        assert _fit_von_mises_kappa(np.zeros(1000)) == 1e6

    def test_empty_returns_zero(self):
        """An empty residual array returns 0 (no data -> no concentration)."""
        assert _fit_von_mises_kappa(np.empty(0)) == 0.0


class TestMacroVonMisesLogscore:
    """The macro (per-region) von Mises log-score (`macro_von_mises_logscore`)."""

    def test_higher_for_better_prediction(self):
        """A prediction closer to the truth scores strictly higher (higher is
        better) than a poorer one."""
        rng = np.random.default_rng(2)
        Y = rng.random((400, 2))
        good = (Y + rng.normal(0, 0.01, Y.shape)) % 1.0
        poor = (Y + rng.normal(0, 0.2, Y.shape)) % 1.0
        s_good = macro_von_mises_logscore(good, Y, None, metric='torus', period=1.0)
        s_poor = macro_von_mises_logscore(poor, Y, None, metric='torus', period=1.0)
        assert np.isfinite(s_good) and np.isfinite(s_poor)
        assert s_good > s_poor

    def test_pooled_when_no_labels(self):
        """`region_labels=None` and an all-NaN label array both fall back to the
        pooled single-region score (identical value)."""
        rng = np.random.default_rng(3)
        Y = rng.random((300, 2))
        Yp = (Y + rng.normal(0, 0.05, Y.shape)) % 1.0
        pooled = macro_von_mises_logscore(Yp, Y, None, metric='torus', period=1.0)
        all_nan = macro_von_mises_logscore(
            Yp, Y, np.full(len(Y), np.nan), metric='torus', period=1.0)
        assert pooled == pytest.approx(all_nan, abs=1e-9)

    def test_macro_drops_regions_below_floor(self):
        """A region with fewer than `min_region_events` labelled events is
        excluded from the macro average (so the score reflects only well-sampled
        regions)."""
        rng = np.random.default_rng(4)
        Y = rng.random((300, 2))
        Yp = (Y + rng.normal(0, 0.05, Y.shape)) % 1.0
        labels = np.zeros(300)
        labels[:5] = 1.0                 # region 1 has only 5 events
        # Share one kappa across both calls (otherwise each re-fits it on its own
        # sample and the two disagree by sampling noise). With a floor of 50,
        # region 1 is dropped -> macro == the pooled mean over the region-0 events.
        r = signed_diff(Y, Yp, metric='torus', period=1.0) * (2 * np.pi)
        kappa = _fit_von_mises_kappa(r.ravel())
        macro = macro_von_mises_logscore(
            Yp, Y, labels, metric='torus', period=1.0, min_region_events=50, kappa=kappa)
        region0_only = macro_von_mises_logscore(
            Yp[5:], Y[5:], None, metric='torus', period=1.0, kappa=kappa)
        assert macro == pytest.approx(region0_only, abs=1e-9)

    def test_kappa_reuse(self):
        """Passing a pre-fit `kappa` reuses it instead of re-fitting (the two
        paths agree when the passed kappa equals the fitted one)."""
        rng = np.random.default_rng(5)
        Y = rng.random((200, 2))
        Yp = (Y + rng.normal(0, 0.05, Y.shape)) % 1.0
        r = signed_diff(Y, Yp, metric='torus', period=1.0) * (2 * np.pi)
        kappa = _fit_von_mises_kappa(r.ravel())
        auto = macro_von_mises_logscore(Yp, Y, None, metric='torus', period=1.0)
        reused = macro_von_mises_logscore(Yp, Y, None, metric='torus', period=1.0, kappa=kappa)
        assert auto == pytest.approx(reused, abs=1e-9)


class TestInverseRegionFrequencyWeights:
    """The equal-region reweighting factors (`inverse_region_frequency_weights`)."""

    def test_inverse_frequency(self):
        """Each labelled row's factor is 1 / count[its region]."""
        labels = np.array([0.0, 0.0, 0.0, 1.0])   # region 0 x3, region 1 x1
        factors = inverse_region_frequency_weights(labels)
        np.testing.assert_allclose(factors, [1 / 3, 1 / 3, 1 / 3, 1.0])

    def test_equal_total_weight_per_region(self):
        """Summing the factors within each region gives the same total across
        regions (the whole point: no region dominates the fit)."""
        labels = np.array([0.0] * 10 + [1.0] * 2)
        factors = inverse_region_frequency_weights(labels)
        assert factors[labels == 0.0].sum() == pytest.approx(factors[labels == 1.0].sum())

    def test_unlabelled_get_median_factor(self):
        """NaN-region rows receive the median labelled factor (neither dropped
        nor over-weighted)."""
        labels = np.array([0.0, 0.0, 0.0, 1.0, np.nan])
        factors = inverse_region_frequency_weights(labels)
        assert factors[-1] == pytest.approx(np.median([1 / 3, 1 / 3, 1 / 3, 1.0]))

    def test_all_unlabelled_is_uniform(self):
        """With no labelled rows the factors are all 1.0 (uniform -> no
        reweighting)."""
        factors = inverse_region_frequency_weights(np.full(6, np.nan))
        np.testing.assert_array_equal(factors, np.ones(6))
