"""
@author: bartulem
Unit tests for ``usv_playpen.neural_modeling.deviance_metrics``.

Coverage: the calibrated explained-deviance statistic and its two fast paths. The batched and binned
scorers exist only to make the shuffle nulls affordable, so what matters is that they agree with the
scalar path they replace -- exactly for the batched one, and to a stated tolerance for the binned one.
The sign-blindness of the statistic is asserted directly, because the whole three-way verdict rests on
the calibration slope rather than the score.
"""

from __future__ import annotations

import numpy as np
import pytest

from usv_playpen.neural_modeling.deviance_metrics import (
    area_under_roc,
    batched_calibrated_explained_deviance,
    binned_calibrated_explained_deviance,
    calibrated_explained_deviance,
    intercept_only_deviance,
)


def _draw(n_frames=4000, n_draws=40, rate=0.08, seed=0):
    """A frozen predictor plus a block of independent relabellings, the shape a null draws."""
    generator = np.random.default_rng(seed)
    eta = generator.normal(size=n_frames) * 1.2
    labels = (generator.random((n_frames, n_draws)) < rate).astype(float)
    return eta, labels


class TestCalibratedExplainedDeviance:

    def test_perfect_ordering_scores_high_and_chance_scores_near_zero(self):
        """A predictor that orders the frames explains deviance; an unrelated one does not."""
        generator = np.random.default_rng(1)
        eta = generator.normal(size=6000)
        informative = (generator.random(6000) < 1.0 / (1.0 + np.exp(-(1.5 * eta - 2.0)))).astype(float)
        unrelated = (generator.random(6000) < 0.12).astype(float)

        informative_score, informative_slope = calibrated_explained_deviance(eta, informative, 30)
        unrelated_score, _unrelated_slope = calibrated_explained_deviance(eta, unrelated, 30)

        assert informative_score > 0.05
        assert informative_slope > 0.0
        assert abs(unrelated_score) < 0.01

    def test_the_score_is_blind_to_sign_but_the_slope_is_not(self):
        """The reason the verdict gates on the slope: a reliably BACKWARDS filter explains deviance
        exactly as well as a correct one, so the score alone cannot tell them apart."""
        generator = np.random.default_rng(2)
        eta = generator.normal(size=6000)
        labels = (generator.random(6000) < 1.0 / (1.0 + np.exp(-(1.5 * eta - 2.0)))).astype(float)

        forward_score, forward_slope = calibrated_explained_deviance(eta, labels, 30)
        reversed_score, reversed_slope = calibrated_explained_deviance(-eta, labels, 30)

        assert forward_score == pytest.approx(reversed_score, rel=1e-6)
        assert forward_slope > 0.0
        assert reversed_slope < 0.0

    def test_degenerate_responses_return_nan(self):
        """All-spike and no-spike frames leave no deviance to explain."""
        eta = np.random.default_rng(3).normal(size=500)
        assert np.isnan(calibrated_explained_deviance(eta, np.zeros(500), 30)[0])
        assert np.isnan(calibrated_explained_deviance(eta, np.ones(500), 30)[0])

    def test_constant_predictor_returns_nan(self):
        """With no spread there is nothing to standardize and nothing to calibrate."""
        labels = (np.random.default_rng(4).random(500) < 0.1).astype(float)
        assert np.isnan(calibrated_explained_deviance(np.full(500, 2.5), labels, 30)[0])


class TestBatchedMatchesScalar:

    def test_batched_equals_the_scalar_path(self):
        """The batched scorer is an optimisation, not an approximation: it must agree to
        floating-point tolerance on every draw."""
        eta, labels = _draw()
        scalar = np.array([calibrated_explained_deviance(eta, labels[:, d], 30)[0]
                           for d in range(labels.shape[1])])
        batched = batched_calibrated_explained_deviance(eta, labels, 30)

        assert np.allclose(scalar, batched, atol=1e-10, equal_nan=True)

    def test_batched_marks_degenerate_draws_nan(self):
        eta, labels = _draw(n_draws=6)
        labels[:, 0] = 0.0
        labels[:, 3] = 1.0
        scores = batched_calibrated_explained_deviance(eta, labels, 30)

        assert np.isnan(scores[0])
        assert np.isnan(scores[3])
        assert np.all(np.isfinite(scores[[1, 2, 4, 5]]))


class TestBinnedApproximation:

    def test_binned_tracks_the_exact_path_within_a_small_fraction_of_the_null_spread(self):
        """The binned scorer approximates only the PREDICTOR-side sums; the responses enter exactly.
        The tolerance that matters is the error relative to the null's own spread, since that is what
        decides whether a draw changes side of the observed value."""
        eta, labels = _draw(n_frames=20000, n_draws=60)
        exact = batched_calibrated_explained_deviance(eta, labels, 30)
        binned = binned_calibrated_explained_deviance(eta, labels, 30, 2000)

        error = np.nanmax(np.abs(exact - binned))
        assert error / np.nanstd(exact) < 0.05

    def test_more_bins_never_makes_it_worse(self):
        eta, labels = _draw(n_frames=20000, n_draws=40)
        exact = batched_calibrated_explained_deviance(eta, labels, 30)
        coarse = np.nanmax(np.abs(exact - binned_calibrated_explained_deviance(eta, labels, 30, 100)))
        fine = np.nanmax(np.abs(exact - binned_calibrated_explained_deviance(eta, labels, 30, 4000)))

        assert fine <= coarse


class TestSupportingMetrics:

    def test_ties_take_the_average_rank(self):
        """The WHEN predictor is an integer spike count in a 50 ms window, so most pairs are tied.
        Breaking ties by sort position would read the arbitrary order as discrimination: here every
        score is identical, so the only defensible answer is chance."""
        assert area_under_roc(np.ones(6), np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])) == pytest.approx(0.5)

    def test_a_half_tied_split_sits_between_chance_and_certainty(self):
        # positives {1, 2} against negatives {0, 1}: three pairs won outright and one tied,
        # so 3.5 of 4 -- a tie-breaking implementation would score this 1.0 or 0.75 depending on
        # which way the sort happened to fall
        scores = np.array([1.0, 2.0, 0.0, 1.0])
        labels = np.array([1.0, 1.0, 0.0, 0.0])
        assert area_under_roc(scores, labels) == pytest.approx(0.875)

    def test_area_under_roc_extremes(self):
        scores = np.array([0.1, 0.2, 0.3, 0.4])
        assert area_under_roc(scores, np.array([0.0, 0.0, 1.0, 1.0])) == pytest.approx(1.0)
        assert area_under_roc(scores, np.array([1.0, 1.0, 0.0, 0.0])) == pytest.approx(0.0)

    def test_intercept_only_deviance_stays_finite_at_a_degenerate_rate(self):
        """The rate is clipped to 1e-6 so a zero-spike epoch cannot produce log(0). The deviance is then
        small but strictly positive, which is what keeps a D^2 denominator usable rather than NaN."""
        degenerate = intercept_only_deviance(np.zeros(100), 0.0)
        assert np.isfinite(degenerate)
        assert 0.0 < degenerate < 1e-3

    def test_intercept_only_deviance_grows_with_a_real_base_rate(self):
        labels = (np.arange(100) < 10).astype(float)
        assert intercept_only_deviance(labels, 0.1) > 50.0
