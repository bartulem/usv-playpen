"""
@author: bartulem
Unit tests for ``usv_playpen.neural_modeling.neural_vocal_occurrence``.

Coverage: the window counting and quiet tiling, the per-session and cross-fitted quiet levels, the class
rates and the equal-prior posterior, the class-balanced leave-one-session-out decoding gain, and its
circular-shift null. The properties asserted are the ones the design turns on -- tiles never overlap a
guarded USV, a held-out window never sets its own quiet level, the gain does not depend on how many quiet
tiles were drawn and is bounded by log 2, it is sign-free so suppressed units are not penalised, and the
null recounts both classes and refits.
"""

from __future__ import annotations

import numpy as np
import pytest

from usv_playpen.neural_modeling.deviance_metrics import (
    area_under_roc,
    calibrate_intercept,
    newton_logistic,
)
from usv_playpen.neural_modeling.neural_vocal_occurrence import (
    EQUAL_PRIOR_LOG,
    class_balanced_mean,
    counts_in_windows,
    held_out_quiet_levels,
    occurrence_context,
    occurrence_log_posterior,
    occurrence_rates,
    quiet_tile_edges,
    session_quiet_levels,
    vocal_occurrence_shift_null,
    vocal_occurrence_statistic,
)
from usv_playpen.neural_modeling.shift_null_inference import shift_range_seconds

AXIS = {"ridge_frac": 0.02, "amplitude_calibration": {"enabled": True, "alpha_grid": [0.1, 0.25, 0.5, 0.75, 1.0, 1.5]},
        "mean_count_floor": 1e-3}
CROSSFIT = 5


def _unit(strength, n_sessions=3, n_positive=200, n_negative=200, seed=0, rate_scale=None):
    """
    Description
    -----------
    Builds a synthetic unit whose peri-onset windows carry ``strength`` more Poisson count than its quiet
    tiles, with an optional per-session rate multiplier so the per-session quiet level is exercised.
    Windows are laid in time order within each session, positives and tiles interleaved.

    Parameters
    ----------
    strength (float)
        Extra mean count of a positive window over the negative mean of 2.0.
    n_sessions (int)
        Number of sessions.
    n_positive (int)
        Peri-onset windows per session.
    n_negative (int)
        Quiet tiles per session.
    seed (int)
        Seed for ``np.random.default_rng``.
    rate_scale (list[float] | None)
        Per-session multiplier of both means; ``None`` means 1.0 everywhere.

    Returns
    -------
    counts (np.ndarray)
        Window counts across all sessions.
    labels (np.ndarray)
        1.0 for peri-onset windows, 0.0 for quiet tiles.
    sessions (np.ndarray)
        Session index of every window.
    times (np.ndarray)
        Window left edge in seconds.
    """
    generator = np.random.default_rng(seed)
    counts, labels, sessions, times = [], [], [], []
    for session in range(n_sessions):
        scale = 1.0 if rate_scale is None else rate_scale[session]
        positive = generator.poisson(scale * (2.0 + strength), n_positive).astype(float)
        negative = generator.poisson(scale * 2.0, n_negative).astype(float)
        counts.append(np.concatenate([positive, negative]))
        labels.append(np.concatenate([np.ones(n_positive), np.zeros(n_negative)]))
        sessions.append(np.full(n_positive + n_negative, session))
        times.append(generator.permutation(n_positive + n_negative).astype(float))
    return (np.concatenate(counts), np.concatenate(labels), np.concatenate(sessions), np.concatenate(times))


def _statistic(counts, labels, sessions, times, axis=AXIS, single_session_blocks=5):
    """
    Description
    -----------
    Runs the occurrence statistic with folds built from the same windows.

    Parameters
    ----------
    counts (np.ndarray)
        Window counts.
    labels (np.ndarray)
        Window labels.
    sessions (np.ndarray)
        Session index per window.
    times (np.ndarray)
        Window left edge per window.
    axis (dict)
        The occurrence settings block.
    single_session_blocks (int)
        Blocks a single training session is cut into.

    Returns
    -------
    result (dict)
        The statistic's result.
    """
    folds = occurrence_context(labels, sessions, times, single_session_blocks)
    return vocal_occurrence_statistic(counts, labels, sessions, times, folds, axis, CROSSFIT)


class TestWindowCounts:

    """Spike counting in fixed-width windows."""
    def test_counts_are_half_open(self):
        """Windows are half-open ``[start, start + width)``: a spike on a shared edge belongs to the later window
        only, so no spike is counted twice or dropped."""
        spikes = np.array([1.0, 1.5, 2.0, 2.5])
        counts = counts_in_windows(spikes, np.array([1.0, 2.0]), 1.0)

        assert counts.tolist() == [2.0, 2.0]      # [1,2) takes 1.0 and 1.5; [2,3) takes 2.0 and 2.5

    def test_empty_windows_score_zero(self):
        """Windows holding no spike must count 0.0 rather than be skipped, so counts stay aligned to their edges."""
        assert counts_in_windows(np.array([5.0]), np.array([0.0, 1.0]), 0.5).tolist() == [0.0, 0.0]


class TestQuietTiles:

    """Tiling the quiet negative class around guarded USVs."""
    def test_tiles_avoid_the_guard_band_around_every_usv(self):
        """The guard is the kinematic encoding's quiet definition, so a tile is silent of BOTH animals and carries a
        forward guard -- the negative class is 'no USV for seconds', not 'not during a USV'."""
        edges = quiet_tile_edges(np.array([100.0]), np.array([100.5]), 300.0, 0.05,
                                 history_pre_seconds=4.0, clean_post_seconds=2.0)

        assert not np.any((edges + 0.05 > 98.0) & (edges < 104.5))
        assert edges.min() >= 0.0
        assert edges.max() + 0.05 <= 300.0 + 1e-9

    def test_tiles_do_not_overlap(self):
        """Sorted tile starts must be at least one width apart, so no spike can be counted in two negative windows."""
        edges = quiet_tile_edges(np.array([50.0, 150.0]), np.array([50.2, 150.2]), 300.0, 0.05, 4.0, 2.0)
        gaps = np.diff(np.sort(edges))

        assert np.all(gaps >= 0.05 - 1e-9)      # no spike can be counted twice

    def test_overlapping_guards_are_merged(self):
        """Two USVs a second apart have overlapping guard bands; merging them is what stops the tiler
        emitting a negative-width gap between them."""
        edges = quiet_tile_edges(np.array([100.0, 101.0]), np.array([100.1, 101.1]), 300.0, 0.05, 4.0, 2.0)

        assert not np.any((edges + 0.05 > 98.0) & (edges < 105.1))

    def test_a_session_with_no_usvs_tiles_throughout(self):
        """A 10 s session with no USVs tiles into exactly ten 1 s windows, so a silent session still contributes
        negatives over its whole length."""
        edges = quiet_tile_edges(np.array([]), np.array([]), 10.0, 1.0, 4.0, 2.0)
        assert edges.size == 10


class TestQuietLevels:

    """The quiet level each window's class rates are expressed against."""
    def test_the_training_level_is_each_sessions_quiet_mean(self):
        """A session's level is the mean count of its quiet tiles, so one pair of class rates serves sessions
        that fire at different speeds."""
        counts = np.array([10.0, 12.0, 2.0, 4.0, 100.0, 104.0, 20.0, 24.0])
        sessions = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        labels = np.array([1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
        levels = session_quiet_levels(counts, labels, sessions, np.arange(8), 1e-3)
        np.testing.assert_allclose(levels, [3.0] * 4 + [22.0] * 4)

    def test_a_session_silent_in_quiet_tiles_stays_finite(self):
        """The failure the old standardisation needed a floor for: a unit silent in every quiet tile. The level
        floor keeps its rates finite."""
        counts = np.array([5.0, 7.0, 0.0, 0.0])
        labels = np.array([1.0, 1.0, 0.0, 0.0])
        levels = session_quiet_levels(counts, labels, np.zeros(4, dtype=int), np.arange(4), 1e-3)
        assert np.all(levels == 1e-3)

    def test_a_held_out_window_never_sets_its_own_level(self):
        """The held-out level is cross-fitted: changing one quiet tile's count must leave its own block's level
        unchanged and move the other blocks'."""
        generator = np.random.default_rng(3)
        counts = generator.poisson(2.0, 100).astype(float)
        labels = np.zeros(100)
        times = np.arange(100, dtype=float)
        rows = np.arange(100)
        before = held_out_quiet_levels(counts, labels, times, rows, 5, 1e-3)
        changed = counts.copy()
        changed[0] += 50.0
        after = held_out_quiet_levels(changed, labels, times, rows, 5, 1e-3)
        np.testing.assert_allclose(after[:20], before[:20])
        assert np.all(after[20:] > before[20:])


class TestRatesAndPosterior:

    """The two class rates and the equal-prior posterior."""
    def test_rates_are_count_over_exposure(self):
        """With no ridge, each class rate is its total count over its total quiet level, so a peri-onset window
        at twice the quiet mean reads as rate 2."""
        counts = np.array([4.0, 4.0, 2.0, 2.0])
        labels = np.array([1.0, 1.0, 0.0, 0.0])
        rates = occurrence_rates(counts, labels, np.full(4, 2.0), 0.0)
        np.testing.assert_allclose(rates, [1.0, 2.0])

    def test_a_zero_amplitude_returns_the_prior(self):
        """``alpha = 0`` removes the contrast between the classes, so the posterior is the equal prior and the
        decoder earns nothing."""
        log_posterior = occurrence_log_posterior(np.array([0.0, 3.0, 9.0]), np.array([1.0, 4.0]),
                                                 np.full(3, 2.0), 0.0)
        np.testing.assert_allclose(log_posterior, EQUAL_PRIOR_LOG)

    def test_the_class_balanced_mean_ignores_the_class_sizes(self):
        """Duplicating the quiet windows must not move the class-balanced mean, which is what keeps the gain
        independent of the tile cap."""
        values = np.array([0.4, 0.2, -0.1, 0.1])
        labels = np.array([1.0, 1.0, 0.0, 0.0])
        doubled = class_balanced_mean(np.concatenate([values, values[2:]]), np.concatenate([labels, labels[2:]]))
        assert class_balanced_mean(values, labels) == pytest.approx(doubled)
        assert np.isnan(class_balanced_mean(values[:2], labels[:2]))


class TestFittingPrimitives:

    """The ridge Newton logistic fit, intercept calibration and AUROC used by the statistic."""
    def test_ridge_leaves_unpenalized_columns_free(self):
        """Intercepts must be able to reach a session's base rate; shrinking them would push the rate
        difference into the slope, which is the coefficient under test."""
        generator = np.random.default_rng(0)
        x = generator.normal(size=400)
        design = np.column_stack([x, np.ones(400)])
        labels = (generator.random(400) < 1.0 / (1.0 + np.exp(-(1.2 * x + 2.0)))).astype(float)

        beta = newton_logistic(design, labels, np.array([0.0, 0.0]), 60)
        heavy = newton_logistic(design, labels, np.array([1e6, 0.0]), 60)

        assert beta[0] == pytest.approx(1.2, abs=0.4)
        assert abs(heavy[0]) < 0.01                 # slope crushed
        assert heavy[1] == pytest.approx(np.log(labels.mean() / (1 - labels.mean())), abs=0.2)

    def test_calibrate_intercept_recovers_the_base_rate_with_a_zero_offset(self):
        """With a zero offset the calibrated intercept must equal the log-odds of the 30% base rate, so the
        intercept carries the rate and nothing else."""
        labels = np.concatenate([np.ones(300), np.zeros(700)])
        intercept = calibrate_intercept(np.zeros(1000), labels, 50)

        assert intercept == pytest.approx(np.log(0.3 / 0.7), abs=1e-6)

    def test_auroc_extremes_and_ties(self):
        """AUROC must be 1.0 for perfect order, 0.0 for reversed order and 0.5 for all-tied scores, pinning its
        orientation and its tie handling."""
        labels = np.array([0.0, 0.0, 1.0, 1.0])
        assert area_under_roc(np.array([1.0, 2.0, 3.0, 4.0]), labels) == pytest.approx(1.0)
        assert area_under_roc(np.array([3.0, 4.0, 1.0, 2.0]), labels) == pytest.approx(0.0)
        assert area_under_roc(np.ones(4), labels) == pytest.approx(0.5)


class TestVocalOccurrenceStatistic:

    """The leave-one-session-out occurrence decoder on synthetic units."""
    def test_a_responsive_unit_decodes_and_a_flat_one_does_not(self):
        """A unit with a 3-count peri-onset elevation must earn a clear gain and AUROC, while a flat unit sits
        near zero gain and chance AUROC; both stay below the log 2 ceiling of an equal prior."""
        strong = _statistic(*_unit(3.0))
        flat = _statistic(*_unit(0.0, seed=1))
        assert 0.05 < strong["gain"] < np.log(2.0)
        assert strong["auroc"] > 0.7
        assert abs(flat["gain"]) < 0.02
        assert flat["auroc"] == pytest.approx(0.5, abs=0.06)

    def test_the_gain_does_not_depend_on_how_many_tiles_were_drawn(self):
        """The prior is fixed and the gain class-balanced, so the same windows with every quiet tile drawn three
        more times (at new times) must leave the gain essentially unchanged, where a frequency prior would
        have moved it."""
        counts, labels, sessions, times = _unit(3.0, seed=2)
        quiet = labels < 0.5
        copies = [(counts[quiet], labels[quiet], sessions[quiet], times[quiet] + 0.25 * (k + 1)) for k in range(3)]
        many = _statistic(np.concatenate([counts] + [c[0] for c in copies]),
                          np.concatenate([labels] + [c[1] for c in copies]),
                          np.concatenate([sessions] + [c[2] for c in copies]),
                          np.concatenate([times] + [c[3] for c in copies]))
        few = _statistic(counts, labels, sessions, times)
        assert many["gain"] == pytest.approx(few["gain"], abs=0.01)

    def test_the_gain_is_sign_free_so_suppression_competes_equally(self):
        """A suppressed unit must decode about as well as an excited one; the direction is reported separately."""
        counts, labels, sessions, times = _unit(3.0)
        elevated = _statistic(counts, labels, sessions, times)
        suppressed = _statistic(counts, 1.0 - labels, sessions, times)
        assert suppressed["gain"] == pytest.approx(elevated["gain"], rel=0.25)
        assert elevated["response_sign"] == "elevated"
        assert suppressed["response_sign"] == "suppressed"

    def test_sessions_at_different_rates_share_one_pair_of_class_rates(self):
        """Sessions firing at 1x, 3x and 6x the same relationship: the per-session quiet level must leave one
        pair of class rates serviceable for all three."""
        result = _statistic(*_unit(3.0, rate_scale=[1.0, 3.0, 6.0]))
        assert result["gain"] > 0.05
        assert result["auroc"] > 0.65
        assert len(result["per_fold"]) == 3

    def test_two_sessions_tune_the_amplitude_on_blocks(self):
        """With two sessions each fold has one training session; its amplitude is tuned on blocks of it, and
        without event times the fold falls back to 1.0."""
        _counts, labels, sessions, times = _unit(3.0, n_sessions=2)
        folds = occurrence_context(labels, sessions, times, 5)
        assert all(len(fold["inner_splits"]) == 5 for fold in folds)
        assert not any(occurrence_context(labels, sessions, None, 5)[0]["inner_splits"] for _ in [0])

    def test_every_fold_is_reported(self):
        """Four sessions must give four per-fold records in session order and a total of 1,600 windows."""
        result = _statistic(*_unit(2.0, n_sessions=4))
        assert [fold["session"] for fold in result["per_fold"]] == [0, 1, 2, 3]
        assert result["n_windows"] == 1600

    def test_labels_other_than_zero_and_one_are_refused(self):
        """The decoder has exactly two classes."""
        with pytest.raises(ValueError, match="must be 1"):
            occurrence_context(np.array([0.0, 2.0, 1.0, 0.0]), np.array([0, 0, 1, 1]), np.arange(4.0), 5)


class TestVocalOccurrenceShiftNull:
    """The null the vocal-occurrence axis is tested against: shift each session's spike train, recount BOTH
    classes, refit. Seeded per draw, which is what lets draws be split across workers and topped up by the
    escalation ladder."""

    GUARD = 20.0

    @staticmethod
    def _unit(n_sessions=2, duration=300.0, n_usvs=60, n_tiles=120, width=0.1, seed=0):
        """
        Description
        -----------
        Builds a unit with uniform background spikes plus a three-spike burst inside every peri-onset window,
        so the spikes are ALIGNED to its USVs -- exactly what a circular shift should destroy.

        Parameters
        ----------
        n_sessions (int)
            Number of sessions.
        duration (float)
            Session length in seconds.
        n_usvs (int)
            Peri-onset windows (positives) per session.
        n_tiles (int)
            Quiet tiles (negatives) per session.
        width (float)
            Window width in seconds.
        seed (int)
            Seed for ``np.random.default_rng``.

        Returns
        -------
        windows (dict)
            ``counts``, ``labels``, ``session_index``, ``edges`` and ``width`` of all windows.
        spikes (dict)
            Sorted spike times in seconds per session slot.
        durations (dict)
            Session duration in seconds per session slot.
        """
        generator = np.random.default_rng(seed)
        counts, labels, sessions, edges, spikes, durations = [], [], [], [], {}, {}
        for slot in range(n_sessions):
            usv_edges = np.sort(generator.uniform(10.0, duration - 10.0, n_usvs))
            tile_edges = np.sort(generator.uniform(10.0, duration - 10.0, n_tiles))
            background = generator.uniform(0.0, duration, 900)
            burst = np.repeat(usv_edges, 3) + generator.uniform(0.0, width, 3 * n_usvs)
            spikes[slot] = np.sort(np.concatenate([background, burst]))
            durations[slot] = duration
            window_edges = np.concatenate([usv_edges, tile_edges])
            counts.append(counts_in_windows(spikes[slot], window_edges, width))
            labels.append(np.concatenate([np.ones(n_usvs), np.zeros(n_tiles)]))
            sessions.append(np.full(window_edges.size, slot))
            edges.append(window_edges)
        windows = {"counts": np.concatenate(counts), "labels": np.concatenate(labels),
                   "session_index": np.concatenate(sessions), "edges": np.concatenate(edges), "width": width}
        return windows, spikes, durations

    def _null(self, windows, spikes, durations, start, n_draws, seed=7):
        """
        Description
        -----------
        Runs ``vocal_occurrence_shift_null`` with folds built from the windows and the module settings.

        Parameters
        ----------
        windows (dict)
            The window arrays from ``_unit``.
        spikes (dict)
            Spike times per session slot.
        durations (dict)
            Session durations per session slot.
        start (int)
            Global index of the first draw.
        n_draws (int)
            Number of draws.
        seed (int)
            Base seed of the per-draw generators.

        Returns
        -------
        null (np.ndarray)
            The ``n_draws`` null statistics.
        """
        folds = occurrence_context(windows["labels"], windows["session_index"], windows["edges"], 5)
        return vocal_occurrence_shift_null(windows, spikes, durations, start, n_draws, seed, self.GUARD, folds,
                                           AXIS, CROSSFIT)

    def _score(self, windows, counts):
        """
        Description
        -----------
        Scores a set of window counts with the occurrence statistic at the module settings.

        Parameters
        ----------
        windows (dict)
            The window arrays from ``_unit``.
        counts (np.ndarray)
            Window counts aligned with ``windows``.

        Returns
        -------
        gain (float)
            The statistic in nats per window.
        """
        return _statistic(counts, windows["labels"], windows["session_index"], windows["edges"])["gain"]

    def test_any_partition_of_the_draws_gives_the_same_values(self):
        """Draw d is seeded by its global index, so splitting [0, 6) as [0, 2) + [2, 6) -- across workers,
        or across escalation steps -- cannot change a single value."""
        windows, spikes, durations = self._unit()
        whole = self._null(windows, spikes, durations, 0, 6)
        split = np.concatenate([self._null(windows, spikes, durations, 0, 2),
                                self._null(windows, spikes, durations, 2, 4)])
        np.testing.assert_array_equal(whole, split)

    def test_a_shift_destroys_real_alignment(self):
        """A unit whose spikes are aligned to its USVs must score above every one of 25 shift-null draws."""
        windows, spikes, durations = self._unit(seed=1)
        observed = self._score(windows, windows["counts"])
        null = self._null(windows, spikes, durations, 0, 25)
        assert observed > float(null.max())

    def test_both_classes_are_recounted(self):
        """Freezing the quiet tiles and shifting only the USVs would be a different null. A draw must equal
        the recount-BOTH reference built from the same generator, and differ from the USVs-only one."""
        windows, spikes, durations = self._unit(seed=2)
        drawn = self._null(windows, spikes, durations, 0, 1)[0]
        generator = np.random.default_rng(7)
        both, usvs_only = windows["counts"].copy(), windows["counts"].copy()
        for slot in sorted(spikes):
            low, high = shift_range_seconds(durations[slot], self.GUARD)
            shifted = np.sort((spikes[slot] + generator.uniform(low, high)) % durations[slot])
            mask = windows["session_index"] == slot
            recounted = counts_in_windows(shifted, windows["edges"][mask], windows["width"])
            both[mask] = recounted
            usvs_only[mask & (windows["labels"] == 1)] = recounted[windows["labels"][mask] == 1]
        assert drawn == pytest.approx(self._score(windows, both), rel=0, abs=0)
        assert self._score(windows, usvs_only) != pytest.approx(drawn)
