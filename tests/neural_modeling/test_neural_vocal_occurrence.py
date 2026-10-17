"""
@author: bartulem
Unit tests for ``usv_playpen.neural_modeling.neural_vocal_occurrence``.

Coverage: the window counting and quiet tiling, the per-session standardization and its sigma floor, the
LOSO transfer statistic, and its circular-shift null. The properties asserted are the ones the design turns
on -- tiles never
overlap a guarded call, standardization is per session, the slope is shared while intercepts are not, and
the statistic is sign-free so suppressed units are not penalised.
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
    counts_in_windows,
    quiet_tile_edges,
    standardize_per_session,
    vocal_occurrence_shift_null,
    vocal_occurrence_statistic,
)
from usv_playpen.neural_modeling.shift_null_inference import shift_range_seconds


class TestWindowCounts:

    def test_counts_are_half_open(self):
        spikes = np.array([1.0, 1.5, 2.0, 2.5])
        counts = counts_in_windows(spikes, np.array([1.0, 2.0]), 1.0)

        assert counts.tolist() == [2.0, 2.0]      # [1,2) takes 1.0 and 1.5; [2,3) takes 2.0 and 2.5

    def test_empty_windows_score_zero(self):
        assert counts_in_windows(np.array([5.0]), np.array([0.0, 1.0]), 0.5).tolist() == [0.0, 0.0]


class TestQuietTiles:

    def test_tiles_avoid_the_guard_band_around_every_call(self):
        """The guard is the kinematic encoding's quiet definition, so a tile is silent of BOTH animals and carries a
        forward guard -- the negative class is 'no call for seconds', not 'not during a call'."""
        edges = quiet_tile_edges(np.array([100.0]), np.array([100.5]), 300.0, 0.05,
                                 history_pre_seconds=4.0, clean_post_seconds=2.0)

        assert not np.any((edges + 0.05 > 98.0) & (edges < 104.5))
        assert edges.min() >= 0.0
        assert edges.max() + 0.05 <= 300.0 + 1e-9

    def test_tiles_do_not_overlap(self):
        edges = quiet_tile_edges(np.array([50.0, 150.0]), np.array([50.2, 150.2]), 300.0, 0.05, 4.0, 2.0)
        gaps = np.diff(np.sort(edges))

        assert np.all(gaps >= 0.05 - 1e-9)      # no spike can be counted twice

    def test_overlapping_guards_are_merged(self):
        """Two calls a second apart have overlapping guard bands; merging them is what stops the tiler
        emitting a negative-width gap between them."""
        edges = quiet_tile_edges(np.array([100.0, 101.0]), np.array([100.1, 101.1]), 300.0, 0.05, 4.0, 2.0)

        assert not np.any((edges + 0.05 > 98.0) & (edges < 105.1))

    def test_a_session_with_no_calls_tiles_throughout(self):
        edges = quiet_tile_edges(np.array([]), np.array([]), 10.0, 1.0, 4.0, 2.0)
        assert edges.size == 10


class TestStandardization:

    def test_scaling_is_per_session_and_taken_from_the_negatives(self):
        counts = np.array([10.0, 12.0, 2.0, 4.0, 100.0, 104.0, 20.0, 24.0])
        sessions = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        labels = np.array([1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0])

        standardized, floored = standardize_per_session(counts, sessions, labels, 0.05)

        # what standardization guarantees: each session's quiet tiles become mean 0, unit SD
        for session in (0, 1):
            quiet = standardized[(sessions == session) & (labels < 0.5)]
            assert np.mean(quiet) == pytest.approx(0.0, abs=1e-9)
            assert np.std(quiet) == pytest.approx(1.0, abs=1e-9)
        # the predictor now reads in SDs of each session's OWN resting variability, so it is the
        # session's spread rather than its rate that sets the scale
        assert floored == 0

    def test_the_sigma_floor_catches_a_unit_silent_during_quiet(self):
        """The failure it was calibrated for: zero quiet-tile SD produced z-values around 2e10 for
        exactly the most vocal-exclusive units."""
        counts = np.array([5.0, 7.0, 0.0, 0.0])
        sessions = np.zeros(4, dtype=int)
        labels = np.array([1.0, 1.0, 0.0, 0.0])

        standardized, floored = standardize_per_session(counts, sessions, labels, 0.05)

        assert floored == 1
        assert np.all(np.isfinite(standardized))
        assert np.abs(standardized).max() < 1e4


class TestFittingPrimitives:

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
        labels = np.concatenate([np.ones(300), np.zeros(700)])
        intercept = calibrate_intercept(np.zeros(1000), labels, 50)

        assert intercept == pytest.approx(np.log(0.3 / 0.7), abs=1e-6)

    def test_auroc_extremes_and_ties(self):
        labels = np.array([0.0, 0.0, 1.0, 1.0])
        assert area_under_roc(np.array([1.0, 2.0, 3.0, 4.0]), labels) == pytest.approx(1.0)
        assert area_under_roc(np.array([3.0, 4.0, 1.0, 2.0]), labels) == pytest.approx(0.0)
        assert area_under_roc(np.ones(4), labels) == pytest.approx(0.5)


class TestVocalOccurrenceStatistic:

    @staticmethod
    def _unit(strength, n_sessions=3, per_session=400, seed=0, rate_scale=None):
        """A synthetic unit whose prevocal windows carry `strength` SDs more count than its quiet tiles,
        with an optional per-session rate multiplier so the shared-slope logic is exercised."""
        generator = np.random.default_rng(seed)
        counts, labels, sessions = [], [], []
        for session in range(n_sessions):
            scale = 1.0 if rate_scale is None else rate_scale[session]
            positive = generator.poisson(scale * (2.0 + strength), per_session // 2).astype(float)
            negative = generator.poisson(scale * 2.0, per_session // 2).astype(float)
            counts.append(np.concatenate([positive, negative]))
            labels.append(np.concatenate([np.ones(per_session // 2), np.zeros(per_session // 2)]))
            sessions.append(np.full(per_session, session))
        return (np.concatenate(counts), np.concatenate(labels), np.concatenate(sessions))

    def test_a_predictive_unit_scores_above_a_null_one(self):
        strong = vocal_occurrence_statistic(*self._unit(3.0), 0.02, 0.05, 40, 40)
        null = vocal_occurrence_statistic(*self._unit(0.0, seed=1), 0.02, 0.05, 40, 40)

        assert strong["d2"] > 0.05
        assert strong["auroc"] > 0.7
        assert abs(null["d2"]) < 0.02
        assert null["auroc"] == pytest.approx(0.5, abs=0.06)

    def test_the_statistic_is_sign_free_so_suppression_competes_equally(self):
        """About a fifth of responders are suppressed; the score must not penalise them, and the
        direction is reported separately."""
        elevated = vocal_occurrence_statistic(*self._unit(3.0), 0.02, 0.05, 40, 40)
        counts, labels, sessions = self._unit(3.0)
        suppressed = vocal_occurrence_statistic(counts, 1.0 - labels, sessions, 0.02, 0.05, 40, 40)

        assert suppressed["d2"] == pytest.approx(elevated["d2"], rel=0.35)
        assert elevated["response_sign"] == "elevated"
        assert suppressed["response_sign"] == "suppressed"

    def test_a_per_session_rate_difference_does_not_break_the_shared_slope(self):
        """Sessions firing at 1x, 3x and 6x the same relationship: per-session standardization plus
        per-session intercepts must leave one slope serviceable for all three."""
        result = vocal_occurrence_statistic(*self._unit(3.0, rate_scale=[1.0, 3.0, 6.0]), 0.02, 0.05, 40, 40)

        assert result["d2"] > 0.05
        assert result["auroc"] > 0.65
        assert len(result["per_fold"]) == 3

    def test_every_fold_is_reported(self):
        result = vocal_occurrence_statistic(*self._unit(2.0, n_sessions=4), 0.02, 0.05, 40, 40)
        assert [fold["session"] for fold in result["per_fold"]] == [0, 1, 2, 3]
        assert result["n_windows"] == 1600


class TestVocalOccurrenceShiftNull:
    """The null the vocal-occurrence axis is tested against: shift each session's spike train, recount BOTH classes,
    refit. Promoted from the pilot driver with per-draw seeds, which is what lets draws be split across
    workers and topped up by the escalation ladder."""

    RIDGE, FLOOR, IRLS, CALIBRATION, GUARD = 0.02, 0.05, 30, 40, 20.0

    @staticmethod
    def _unit(n_sessions=2, duration=300.0, n_calls=60, n_tiles=120, width=0.1, seed=0):
        """Uniform background spikes plus a burst inside every prevocal window, so the unit's spikes are
        ALIGNED to its calls -- exactly what a shift should destroy."""
        generator = np.random.default_rng(seed)
        counts, labels, sessions, edges, spikes, durations = [], [], [], [], {}, {}
        for slot in range(n_sessions):
            call_edges = np.sort(generator.uniform(10.0, duration - 10.0, n_calls))
            tile_edges = np.sort(generator.uniform(10.0, duration - 10.0, n_tiles))
            background = generator.uniform(0.0, duration, 900)
            burst = np.repeat(call_edges, 3) + generator.uniform(0.0, width, 3 * n_calls)
            spikes[slot] = np.sort(np.concatenate([background, burst]))
            durations[slot] = duration
            window_edges = np.concatenate([call_edges, tile_edges])
            counts.append(counts_in_windows(spikes[slot], window_edges, width))
            labels.append(np.concatenate([np.ones(n_calls), np.zeros(n_tiles)]))
            sessions.append(np.full(window_edges.size, slot))
            edges.append(window_edges)
        windows = {"counts": np.concatenate(counts), "labels": np.concatenate(labels),
                "session_index": np.concatenate(sessions), "edges": np.concatenate(edges), "width": width}
        return windows, spikes, durations

    def _null(self, windows, spikes, durations, start, n_draws, seed=7):
        return vocal_occurrence_shift_null(windows, spikes, durations, start, n_draws, seed, self.GUARD, self.RIDGE,
                               self.FLOOR, self.IRLS, self.CALIBRATION)

    def test_any_partition_of_the_draws_gives_the_same_values(self):
        """Draw d is seeded by its global index, so splitting [0, 6) as [0, 2) + [2, 6) -- across workers,
        or across escalation steps -- cannot change a single value."""
        windows, spikes, durations = self._unit()
        whole = self._null(windows, spikes, durations, 0, 6)
        split = np.concatenate([self._null(windows, spikes, durations, 0, 2),
                                self._null(windows, spikes, durations, 2, 4)])
        np.testing.assert_array_equal(whole, split)

    def test_a_shift_destroys_real_alignment(self):
        windows, spikes, durations = self._unit(seed=1)
        observed = vocal_occurrence_statistic(windows["counts"], windows["labels"], windows["session_index"], self.RIDGE,
                                  self.FLOOR, self.IRLS, self.CALIBRATION)["nats_per_window"]
        null = self._null(windows, spikes, durations, 0, 25)
        assert observed > float(null.max())

    def test_both_classes_are_recounted(self):
        """Freezing the quiet tiles and shifting only the calls would be a different null. A draw must equal
        the recount-BOTH reference built from the same generator, and differ from the calls-only one."""
        windows, spikes, durations = self._unit(seed=2)
        drawn = self._null(windows, spikes, durations, 0, 1)[0]
        generator = np.random.default_rng(7)
        both, calls_only = windows["counts"].copy(), windows["counts"].copy()
        for slot in sorted(spikes):
            low, high = shift_range_seconds(durations[slot], self.GUARD)
            shifted = np.sort((spikes[slot] + generator.uniform(low, high)) % durations[slot])
            mask = windows["session_index"] == slot
            recounted = counts_in_windows(shifted, windows["edges"][mask], windows["width"])
            both[mask] = recounted
            calls_only[mask & (windows["labels"] == 1)] = recounted[windows["labels"][mask] == 1]

        def score(counts):
            return vocal_occurrence_statistic(counts, windows["labels"], windows["session_index"], self.RIDGE, self.FLOOR,
                                  self.IRLS, self.CALIBRATION)["nats_per_window"]

        assert drawn == pytest.approx(score(both), rel=0, abs=0)
        assert score(calls_only) != pytest.approx(drawn)
