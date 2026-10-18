"""
@author: bartulem
Unit tests for ``usv_playpen.neural_modeling.deviance_metrics``.

Coverage: the calibrated explained-deviance statistic and its two fast paths. The batched and binned
scorers exist only to make the shuffle nulls affordable, so what matters is that they agree with the
scalar path they replace -- exactly for the batched one, and to a stated tolerance for the binned one.
The sign-blindness of the statistic is asserted directly, because the whole three-way verdict rests on
the calibration slope rather than the score.

The cross-fitted rule: blocks are contiguous within each group; a planted predictor
scores positive with a positive scale and an unrelated one near zero; a reversed predictor keeps the
negative scale that the verdict reads; the batched null path agrees with the scalar one exactly at
``n_bins = 0`` and within tolerance when binned; the router refuses an unknown scheme and reproduces the
in-sample rule when asked for it.
"""

from __future__ import annotations

import numpy as np
import pytest

from usv_playpen.neural_modeling.deviance_metrics import (
    area_under_roc,
    batched_calibrated_explained_deviance,
    batched_crossfitted_explained_deviance,
    binned_calibrated_explained_deviance,
    calibrated_explained_deviance,
    contiguous_blocks,
    crossfitted_explained_deviance,
    frozen_model_null_scores,
    frozen_model_score,
    intercept_only_deviance,
)


def _draw(n_frames=4000, n_draws=40, rate=0.08, seed=0):
    """
    Description
    -----------
    Builds a frozen predictor plus a block of independent Bernoulli relabellings, the shape a shuffle
    null scores.

    Parameters
    ----------
    n_frames (int)
        Number of frames.
    n_draws (int)
        Number of independent relabellings (columns).
    rate (float)
        Spike probability of every relabelled frame.
    seed (int)
        Seed for ``np.random.default_rng``.

    Returns
    -------
    eta (np.ndarray)
        The predictor, Gaussian with standard deviation 1.2, shape ``(n_frames,)``.
    labels (np.ndarray)
        The 0/1 relabellings as float, shape ``(n_frames, n_draws)``.
    """
    generator = np.random.default_rng(seed)
    eta = generator.normal(size=n_frames) * 1.2
    labels = (generator.random((n_frames, n_draws)) < rate).astype(float)
    return eta, labels


class TestCalibratedExplainedDeviance:

    """The scalar calibrated explained-deviance statistic and its calibration slope."""
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

    """The batched scorer that makes the shuffle null affordable, against the scalar path."""
    def test_batched_equals_the_scalar_path(self):
        """The batched scorer is an optimisation, not an approximation: it must agree to
        floating-point tolerance on every draw."""
        eta, labels = _draw()
        scalar = np.array([calibrated_explained_deviance(eta, labels[:, d], 30)[0]
                           for d in range(labels.shape[1])])
        batched = batched_calibrated_explained_deviance(eta, labels, 30)

        assert np.allclose(scalar, batched, atol=1e-10, equal_nan=True)

    def test_batched_marks_degenerate_draws_nan(self):
        """An all-zero and an all-one relabelling must score NaN while the other draws stay finite, so a
        degenerate draw is dropped from the null instead of counted as a value."""
        eta, labels = _draw(n_draws=6)
        labels[:, 0] = 0.0
        labels[:, 3] = 1.0
        scores = batched_calibrated_explained_deviance(eta, labels, 30)

        assert np.isnan(scores[0])
        assert np.isnan(scores[3])
        assert np.all(np.isfinite(scores[[1, 2, 4, 5]]))


class TestBinnedApproximation:

    """The predictor-binned approximation of the batched scorer."""
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
        """The worst-case error at 4,000 bins must not exceed that at 100 bins, so the bin count is a knob that
        monotonically trades speed for accuracy."""
        eta, labels = _draw(n_frames=20000, n_draws=40)
        exact = batched_calibrated_explained_deviance(eta, labels, 30)
        coarse = np.nanmax(np.abs(exact - binned_calibrated_explained_deviance(eta, labels, 30, 100)))
        fine = np.nanmax(np.abs(exact - binned_calibrated_explained_deviance(eta, labels, 30, 4000)))

        assert fine <= coarse


class TestSupportingMetrics:

    """The tie-aware AUROC and the intercept-only deviance used alongside the main statistic."""
    def test_ties_take_the_average_rank(self):
        """The vocal-occurrence predictor is an integer spike count in a 50 ms window, so most pairs are tied.
        Breaking ties by sort position would read the arbitrary order as discrimination: here every
        score is identical, so the only defensible answer is chance."""
        assert area_under_roc(np.ones(6), np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])) == pytest.approx(0.5)

    def test_a_half_tied_split_sits_between_chance_and_certainty(self):
        """Positives {1, 2} against negatives {0, 1} give three pairs won outright and one tied, so the AUROC
        must be 3.5 of 4 = 0.875; a tie-breaking implementation would score 1.0 or 0.75 depending on which way
        the sort happened to fall."""
        scores = np.array([1.0, 2.0, 0.0, 1.0])
        labels = np.array([1.0, 1.0, 0.0, 0.0])
        assert area_under_roc(scores, labels) == pytest.approx(0.875)

    def test_area_under_roc_extremes(self):
        """Perfectly separated scores must give AUROC 1.0 and perfectly reversed ones 0.0, pinning the
        orientation of the metric."""
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
        """A 10% base rate over 100 frames must give a deviance above 50, so the denominator scales with the
        information actually present."""
        labels = (np.arange(100) < 10).astype(float)
        assert intercept_only_deviance(labels, 0.1) > 50.0


CROSSFIT = {"scheme": "crossfit", "n_blocks": 5}


class TestCrossfittedScoring:

    """The cross-fitted calibration rule, its contiguous blocks, its batched null path and the scheme router."""
    @staticmethod
    def _planted(seed, gain, n_rows=6000, family="bernoulli", n_groups=1):
        """
        Description
        -----------
        Builds a smoothed Gaussian predictor and a response planted on it (Bernoulli through a logistic link,
        or Poisson through a log link), with an optional per-group offset in the response level.

        Parameters
        ----------
        seed (int)
            Seed for ``np.random.default_rng``.
        gain (float)
            Coefficient of the predictor in the linear predictor; negative or zero gives no/reversed signal.
        n_rows (int)
            Number of frames.
        family (str)
            ``"bernoulli"`` or anything else for Poisson.
        n_groups (int)
            Number of equal contiguous groups; each adds 0.4 times its index to the linear predictor.

        Returns
        -------
        x (np.ndarray)
            The predictor, shape ``(n_rows,)``.
        y (np.ndarray)
            The planted response as float, shape ``(n_rows,)``.
        group (np.ndarray)
            The integer group label of every frame.
        """
        generator = np.random.default_rng(seed)
        x = np.convolve(generator.normal(size=n_rows), np.ones(25) / 5.0, mode="same")
        group = np.repeat(np.arange(n_groups), n_rows // n_groups)
        level = 0.4 * group
        if family == "bernoulli":
            y = (generator.random(n_rows) < 1.0 / (1.0 + np.exp(-(gain * x - 2.0 + level)))).astype(float)
        else:
            y = generator.poisson(np.exp(gain * x - 1.0 + level)).astype(float)
        return x, y, group

    def test_blocks_are_contiguous_within_each_group(self):
        """Blocks must be contiguous and non-decreasing inside each group and cover every block id, so a
        cross-fit fold never interleaves autocorrelated neighbouring frames between fit and score."""
        group = np.repeat([0, 1], [10, 7])
        blocks = contiguous_blocks(group, 3)
        np.testing.assert_array_equal(blocks[:10], [0, 0, 0, 0, 1, 1, 1, 2, 2, 2])
        assert np.all(np.diff(blocks[10:]) >= 0)
        assert set(blocks[10:]) == {0, 1, 2}

    @pytest.mark.parametrize("family", ["bernoulli", "poisson"])
    def test_a_planted_predictor_scores_and_an_unrelated_one_does_not(self, family):
        """For both families, a planted predictor must score above 0.02 with a positive scale, and the same
        predictor against a shuffled response must score near zero, so cross-fitting keeps signal and adds no
        spurious score."""
        x, y, group = self._planted(1, 0.8, family=family, n_groups=2)
        score, slope = crossfitted_explained_deviance(x, y, group, 5, family, 40)
        assert score > 0.02
        assert slope > 0.0
        shuffled = np.random.default_rng(2).permutation(y)
        null_score, _ = crossfitted_explained_deviance(x, shuffled, group, 5, family, 40)
        assert abs(null_score) < 0.01

    def test_a_reversed_predictor_keeps_its_negative_scale(self):
        """The verdict reads the SIGN of the scale; cross-fitting must not hide a reversed filter."""
        x, y, group = self._planted(3, 0.8)
        _score, slope = crossfitted_explained_deviance(-x, y, group, 5, "bernoulli", 40)
        assert slope < 0.0

    def test_the_batched_null_matches_the_scalar_rule(self):
        """The batched cross-fitted null must equal the scalar rule exactly at ``n_bins = 0`` and within 2e-4 at
        500 bins, so the null is drawn from the same statistic the observed value uses."""
        eta, labels = _draw(n_frames=3000, n_draws=12, rate=0.1, seed=5)
        scalar = np.array([crossfitted_explained_deviance(eta, labels[:, k], np.zeros(eta.size, dtype=np.int64),
                                                          5, "bernoulli", 40)[0] for k in range(labels.shape[1])])
        exact = batched_crossfitted_explained_deviance(eta, labels, 5, 40, 0)
        binned = batched_crossfitted_explained_deviance(eta, labels, 5, 40, 500)
        np.testing.assert_allclose(exact, scalar, atol=1e-7)
        np.testing.assert_allclose(binned, scalar, atol=2e-4)

    def test_the_router_refuses_an_unknown_scheme_and_reproduces_in_sample(self):
        """An unknown ``calibration.scheme`` must raise ``ValueError`` in both router entry points, and the
        ``in_sample`` and ``crossfit`` schemes must reproduce their underlying functions exactly."""
        x, y, group = self._planted(6, 0.8)
        with pytest.raises(ValueError, match=r"calibration\.scheme"):
            frozen_model_score(x, y, group, 40, "bernoulli", 1.0, {"scheme": "bootstrap", "n_blocks": 5})
        with pytest.raises(ValueError, match=r"calibration\.scheme"):
            frozen_model_null_scores(x, y[:, None], 40, 0, {"scheme": "bootstrap", "n_blocks": 5})
        in_sample = frozen_model_score(x, y, group, 40, "bernoulli", 1.0, {"scheme": "in_sample", "n_blocks": 5})
        assert in_sample == calibrated_explained_deviance(x, y, 40)
        crossfit = frozen_model_score(x, y, group, 40, "bernoulli", 1.0, CROSSFIT)
        assert crossfit == crossfitted_explained_deviance(x, y, group, 5, "bernoulli", 40)
