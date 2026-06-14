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
    pairwise_distance,
    resolve_manifold_metric,
    signed_diff,
    signed_diff_jax,
    sin_cos_encode_jax,
    torus_embed,
    total_dispersion,
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
