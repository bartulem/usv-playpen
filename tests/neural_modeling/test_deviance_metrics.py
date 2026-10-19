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

The guards and cores, against known answers: one Newton step from zero equals the closed-form step
(logistic) or the damped step (Poisson), and a singular Hessian stops the solver at its current
coefficients; every scorer's degenerate-input guard (flat predictor, one class, nothing to explain, a
training split that leaves a group without events or non-events) returns NaN as documented; with zero
Newton steps the batched, binned and batched cross-fitted scorers return the closed-form score of the
zero calibration; and both routers send every scheme, family and group count to the scorer they name.
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
    encoding_metrics,
    frozen_model_null_scores,
    frozen_model_score,
    intercept_only_deviance,
    newton_logistic,
    newton_poisson,
    pooled_calibrated_explained_deviance,
    pooled_poisson_explained_deviance,
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


class TestNewtonCores:
    """The two Newton solvers: one exact step from zero, and the singular-Hessian stop."""

    def test_one_logistic_step_is_the_closed_form_newton_step_from_zero(self):
        """
        Description
        -----------
        From ``beta = 0`` every fitted mean is 0.5 and every weight 0.25, so a single
        unpenalised Newton step is ``(0.25 X'X)^-1 X'(y - 0.5)`` exactly; with
        ``n_steps = 1`` the loop runs out after that step and returns it.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        rng = np.random.default_rng(0)
        x = np.column_stack([rng.normal(size=50), np.ones(50)])
        y = (rng.random(50) < 0.3).astype(float)
        expected = np.linalg.solve(0.25 * x.T @ x, x.T @ (y - 0.5))
        np.testing.assert_allclose(
            newton_logistic(x, y, np.zeros(2), 1), expected, rtol=1e-12
        )

    def test_a_singular_logistic_hessian_stops_at_the_current_coefficients(self):
        """
        Description
        -----------
        An all-zero design column with no ridge makes the Hessian exactly singular; the
        solver must stop rather than raise, returning the coefficients it had (zeros).

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        x = np.column_stack([np.ones(6), np.zeros(6)])
        y = np.array([0.0, 1.0, 0.0, 1.0, 1.0, 0.0])
        np.testing.assert_array_equal(
            newton_logistic(x, y, np.zeros(2), 10), [0.0, 0.0]
        )

    @pytest.mark.parametrize(
        ("counts", "expected"),
        [
            (np.array([0.0, 1.0, 2.0, 5.0]), 1.0),
            (np.array([10.0, 10.0, 10.0, 10.0]), 2.0),
        ],
    )
    def test_one_poisson_step_is_the_damped_newton_step_from_zero(
        self, counts, expected
    ):
        """
        Description
        -----------
        For an intercept-only Poisson fit from ``beta = 0`` every mean is 1, so the Newton
        step is ``sum(counts - 1) / n``: 1.0 for counts ``{0, 1, 2, 5}`` (taken whole) and
        9.0 for four tens, which the damping caps at a largest change of 2.

        Parameters
        ----------
        counts (np.ndarray)
            The four observed counts.
        expected (float)
            The intercept after one damped step.

        Returns
        -------
        None
        """
        beta = newton_poisson(np.ones((4, 1)), counts, np.zeros(1), 1)
        assert beta[0] == pytest.approx(expected, rel=1e-9)

    def test_a_singular_poisson_hessian_stops_at_the_current_coefficients(self):
        """
        Description
        -----------
        Two identical columns of magnitude 1e4 swamp the 1e-12 jitter, so the Hessian is
        exactly singular in float64; the solver must stop and return zeros, not raise.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        x = np.full((10, 2), 1e4)
        np.testing.assert_array_equal(
            newton_poisson(x, np.arange(10.0), np.zeros(2), 10), [0.0, 0.0]
        )


class TestDegenerateInputsReturnNan:
    """Every scorer's documented guard: no spread, no events or nothing to explain gives NaN."""

    def test_auroc_with_one_class_is_nan(self):
        """
        Description
        -----------
        With only positives (or only negatives) there is no pair to rank, so the AUROC is
        undefined and must come back NaN.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        assert np.isnan(area_under_roc(np.array([0.1, 0.5, 0.9]), np.ones(3)))
        assert np.isnan(area_under_roc(np.array([0.1, 0.5, 0.9]), np.zeros(3)))

    def test_pooled_poisson_with_nothing_to_explain_is_nan(self):
        """
        Description
        -----------
        A count of exactly one in every bin is fitted perfectly by the intercept (the
        Newton start ``mu = 1`` is already the optimum), so the baseline deviance is zero
        and the explained fraction must be NaN, with the gain left at zero.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        x = np.random.default_rng(0).normal(size=40)
        score, slope = pooled_poisson_explained_deviance(
            x, np.ones(40), np.zeros(40, dtype=np.int64), 40, 1.0
        )
        assert np.isnan(score)
        assert slope == 0.0

    def test_pooled_calibrated_with_a_flat_predictor_is_nan(self):
        """
        Description
        -----------
        A constant predictor has no spread to standardise, so the pooled calibrated score
        and slope must both be NaN.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        y = np.array([0.0, 1.0] * 10)
        score, slope = pooled_calibrated_explained_deviance(
            np.full(20, 0.3), y, np.repeat([0, 1], 10), 40
        )
        assert np.isnan(score)
        assert np.isnan(slope)

    @pytest.mark.parametrize(
        ("eta", "labels", "spike_rate"),
        [
            (np.full(5, 1.0), np.array([0.0, 1.0, 0.0, 1.0, 1.0]), 0.6),
            (np.linspace(-1.0, 1.0, 5), np.zeros(5), 0.0),
            (np.zeros(0), np.zeros(0), np.nan),
        ],
    )
    def test_encoding_metrics_degenerate_roster(self, eta, labels, spike_rate):
        """
        Description
        -----------
        A flat predictor, a spike-free response or an empty scoring set returns the full
        roster with every model statistic NaN, while the bookkeeping (``spike_rate``,
        ``n_frames``, ``n_spikes``) still reports what was there.

        Parameters
        ----------
        eta (np.ndarray)
            The frozen predictor.
        labels (np.ndarray)
            The 0/1 labels.
        spike_rate (float)
            The expected reported spike rate (NaN for an empty set).

        Returns
        -------
        None
        """
        metrics = encoding_metrics(eta, labels, 30)
        for key in (
            "deviance_explained",
            "calibration_slope",
            "calibration_intercept",
            "log_loss",
            "llr_per_spike",
            "brier",
            "auroc",
        ):
            assert np.isnan(metrics[key])
        np.testing.assert_equal(metrics["spike_rate"], spike_rate)
        assert metrics["n_frames"] == labels.size
        assert metrics["n_spikes"] == int(labels.sum())

    def test_every_scorer_returns_nan_on_an_empty_scoring_set_without_warning(self):
        """
        Description
        -----------
        An empty scoring set must trip each scorer's own degenerate-input guard silently:
        the predictor spread is taken only when there is something to take it over, so
        numpy's 'Degrees of freedom <= 0' warning (an error under the suite's filter) never
        fires. The scalar scorers return ``(nan, nan)`` and the batched ones one NaN per
        draw (three draws here).

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        empty = np.zeros(0)
        empty_index = np.zeros(0, dtype=np.int64)
        empty_draws = np.zeros((0, 3))
        scalar_results = (
            pooled_poisson_explained_deviance(empty, empty, empty_index, 30, 1.0),
            calibrated_explained_deviance(empty, empty, 30),
            pooled_calibrated_explained_deviance(empty, empty, empty_index, 30),
            crossfitted_explained_deviance(empty, empty, empty_index, 2, "bernoulli", 30),
        )
        for score, slope in scalar_results:
            assert np.isnan(score)
            assert np.isnan(slope)
        batched_results = (
            batched_calibrated_explained_deviance(empty, empty_draws, 30),
            binned_calibrated_explained_deviance(empty, empty_draws, 30, 10),
            batched_crossfitted_explained_deviance(empty, empty_draws, 2, 30, 10),
        )
        for scores in batched_results:
            assert scores.shape == (3,)
            assert np.all(np.isnan(scores))

    def test_batched_and_binned_with_a_flat_predictor_are_all_nan(self):
        """
        Description
        -----------
        A constant predictor leaves every draw undefined in the three batched scorers, so
        each returns one NaN per draw rather than raising.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        _eta, labels = _draw(n_frames=200, n_draws=4)
        flat = np.full(200, 0.7)
        for scores in (
            batched_calibrated_explained_deviance(flat, labels, 30),
            binned_calibrated_explained_deviance(flat, labels, 30, 20),
            batched_crossfitted_explained_deviance(flat, labels, 5, 30, 0),
        ):
            assert scores.shape == (4,)
            assert np.all(np.isnan(scores))

    def test_batched_and_binned_with_only_degenerate_draws_are_all_nan(self):
        """
        Description
        -----------
        When every draw is all-zero or all-one, no draw is solvable, the Newton loop stops
        at once, and every score must be NaN.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        eta = np.random.default_rng(1).normal(size=200)
        labels = np.column_stack([np.zeros(200), np.ones(200), np.zeros(200)])
        for scores in (
            batched_calibrated_explained_deviance(eta, labels, 30),
            binned_calibrated_explained_deviance(eta, labels, 30, 20),
            batched_crossfitted_explained_deviance(eta, labels, 5, 30, 0),
        ):
            assert np.all(np.isnan(scores))

    def test_crossfit_with_a_flat_predictor_is_nan(self):
        """
        Description
        -----------
        The scalar cross-fitted scorer returns NaN for both outputs when the predictor
        has no spread.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        y = np.array([0.0, 1.0] * 50)
        score, slope = crossfitted_explained_deviance(
            np.ones(100), y, np.zeros(100, dtype=np.int64), 5, "bernoulli", 30
        )
        assert np.isnan(score)
        assert np.isnan(slope)

    @pytest.mark.parametrize("missing", ["events", "non_events"])
    def test_crossfit_with_a_group_degenerate_in_a_training_split_is_nan(self, missing):
        """
        Description
        -----------
        Group 1's events (or its non-events) all sit in its first contiguous block, so the
        training rows used to score that block leave group 1 with no events (or no
        non-events) to estimate its intercept from; the score must then be NaN, as the
        docstring states, rather than an extrapolated intercept.

        Parameters
        ----------
        missing (str)
            ``'events'`` to confine group 1's spikes to one block, ``'non_events'`` to
            confine its silences.

        Returns
        -------
        None
        """
        rng = np.random.default_rng(2)
        x = rng.normal(size=200)
        y = np.r_[(rng.random(100) < 0.3).astype(float), np.zeros(100)]
        group = np.repeat([0, 1], 100)
        y[100:120] = 1.0
        if missing == "non_events":
            y[100:] = 1.0 - y[100:]
        score, slope = crossfitted_explained_deviance(x, y, group, 2, "bernoulli", 30)
        assert np.isnan(score)
        assert np.isnan(slope)

    def test_crossfit_poisson_with_nothing_to_explain_is_nan(self):
        """
        Description
        -----------
        Counts of exactly one everywhere are matched by the cross-fitted intercept, whose
        log-likelihood then equals the saturated one; the denominator is zero, so the
        score must be NaN while the mean scale (zero) is still reported.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        x = np.random.default_rng(3).normal(size=40)
        score, slope = crossfitted_explained_deviance(
            x, np.ones(40), np.zeros(40, dtype=np.int64), 4, "poisson", 40
        )
        assert np.isnan(score)
        assert slope == 0.0


class TestZeroNewtonSteps:
    """With no Newton steps the calibration stays at zero, which has a closed-form score."""

    def test_batched_and_binned_score_the_zero_model(self):
        """
        Description
        -----------
        With ``n_steps = 0`` the slope and intercept stay at zero, so every fitted
        probability is 0.5 and the model deviance is ``2 n log 2``; the score must be
        ``1 - 2 n log 2 / D0`` with ``D0`` the deviance of the draw's own rate, for the
        exact and the binned path alike (the binned path is exact at a zero slope).

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        eta, labels = _draw(n_frames=500, n_draws=3, rate=0.2, seed=4)
        rate = labels.mean(axis=0)
        baseline = -2.0 * 500 * (rate * np.log(rate) + (1.0 - rate) * np.log1p(-rate))
        expected = 1.0 - 2.0 * 500 * np.log(2.0) / baseline
        np.testing.assert_allclose(
            batched_calibrated_explained_deviance(eta, labels, 0), expected, rtol=1e-12
        )
        np.testing.assert_allclose(
            binned_calibrated_explained_deviance(eta, labels, 0, 50),
            expected,
            rtol=1e-12,
        )

    def test_batched_crossfit_scores_the_zero_model_against_the_crossfitted_rate(self):
        """
        Description
        -----------
        With ``n_steps = 0`` each block is scored at probability 0.5 (log-likelihood
        ``-n_test log 2``) against the reference at the training rows' rate; the score
        must equal that closed form summed over the five contiguous blocks.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        eta, labels = _draw(n_frames=500, n_draws=3, rate=0.2, seed=5)
        blocks = contiguous_blocks(np.zeros(500, dtype=np.int64), 5)
        ll_model = np.zeros(3)
        ll_reference = np.zeros(3)
        for block in range(5):
            test = blocks == block
            train_rate = labels[~test].mean(axis=0)
            intercept = np.log(train_rate / (1.0 - train_rate))
            ll_model -= test.sum() * np.log(2.0)
            ll_reference += intercept * labels[test].sum(
                axis=0
            ) - test.sum() * np.logaddexp(0.0, intercept)
        expected = (ll_model - ll_reference) / (-ll_reference)
        np.testing.assert_allclose(
            batched_crossfitted_explained_deviance(eta, labels, 5, 0, 0),
            expected,
            rtol=1e-10,
        )


class TestSchemeRouting:
    """The two routers dispatch every scheme/family/group combination to the right scorer."""

    def test_in_sample_poisson_routes_to_the_pooled_poisson_scorer(self):
        """
        Description
        -----------
        ``in_sample`` with the Poisson family must return exactly what
        ``pooled_poisson_explained_deviance`` returns for the same inputs.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        x, y, group = TestCrossfittedScoring._planted(
            7, 0.8, family="poisson", n_groups=2
        )
        routed = frozen_model_score(
            x, y, group, 40, "poisson", 2.0, {"scheme": "in_sample", "n_blocks": 5}
        )
        assert routed == pooled_poisson_explained_deviance(x, y, group, 40, 2.0)

    def test_in_sample_bernoulli_over_several_groups_routes_to_the_pooled_scorer(self):
        """
        Description
        -----------
        ``in_sample`` Bernoulli over more than one group must use per-group intercepts,
        i.e. return exactly ``pooled_calibrated_explained_deviance``.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        x, y, group = TestCrossfittedScoring._planted(8, 0.8, n_groups=2)
        routed = frozen_model_score(
            x, y, group, 40, "bernoulli", 1.0, {"scheme": "in_sample", "n_blocks": 5}
        )
        assert routed == pooled_calibrated_explained_deviance(x, y, group, 40)

    def test_the_null_router_dispatches_each_scheme(self):
        """
        Description
        -----------
        ``frozen_model_null_scores`` must return exactly the batched cross-fitted scores
        under ``crossfit``, and under ``in_sample`` the binned scores when ``n_bins > 0``
        and the exact batched scores when ``n_bins = 0``.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        eta, labels = _draw(n_frames=1000, n_draws=5, seed=6)
        in_sample = {"scheme": "in_sample", "n_blocks": 5}
        np.testing.assert_array_equal(
            frozen_model_null_scores(eta, labels, 30, 0, CROSSFIT),
            batched_crossfitted_explained_deviance(eta, labels, 5, 30, 0),
        )
        np.testing.assert_array_equal(
            frozen_model_null_scores(eta, labels, 30, 100, in_sample),
            binned_calibrated_explained_deviance(eta, labels, 30, 100),
        )
        np.testing.assert_array_equal(
            frozen_model_null_scores(eta, labels, 30, 0, in_sample),
            batched_calibrated_explained_deviance(eta, labels, 30),
        )
