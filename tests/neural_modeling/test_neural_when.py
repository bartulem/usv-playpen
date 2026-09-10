"""
@author: bartulem
Unit tests for ``usv_playpen.neural_modeling.neural_when``.

Coverage: the window counting and quiet tiling, the per-session standardization and its sigma floor, and
the LOSO transfer statistic. The properties asserted are the ones the WHEN design turns on -- tiles never
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
from usv_playpen.neural_modeling.neural_when import (
    counts_in_windows,
    quiet_tile_edges,
    standardize_per_session,
    when_statistic,
)


class TestWindowCounts:

    def test_counts_are_half_open(self):
        spikes = np.array([1.0, 1.5, 2.0, 2.5])
        counts = counts_in_windows(spikes, np.array([1.0, 2.0]), 1.0)

        assert counts.tolist() == [2.0, 2.0]      # [1,2) takes 1.0 and 1.5; [2,3) takes 2.0 and 2.5

    def test_empty_windows_score_zero(self):
        assert counts_in_windows(np.array([5.0]), np.array([0.0, 1.0]), 0.5).tolist() == [0.0, 0.0]


class TestQuietTiles:

    def test_tiles_avoid_the_guard_band_around_every_call(self):
        """The guard is the claim-1 quiet definition, so a tile is silent of BOTH animals and carries a
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


class TestWhenStatistic:

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
        strong = when_statistic(*self._unit(3.0), 0.02, 0.05, 40, 40)
        null = when_statistic(*self._unit(0.0, seed=1), 0.02, 0.05, 40, 40)

        assert strong["d2"] > 0.05
        assert strong["auroc"] > 0.7
        assert abs(null["d2"]) < 0.02
        assert null["auroc"] == pytest.approx(0.5, abs=0.06)

    def test_the_statistic_is_sign_free_so_suppression_competes_equally(self):
        """About a fifth of responders are suppressed; the score must not penalise them, and the
        direction is reported separately."""
        elevated = when_statistic(*self._unit(3.0), 0.02, 0.05, 40, 40)
        counts, labels, sessions = self._unit(3.0)
        suppressed = when_statistic(counts, 1.0 - labels, sessions, 0.02, 0.05, 40, 40)

        assert suppressed["d2"] == pytest.approx(elevated["d2"], rel=0.35)
        assert elevated["response_sign"] == "elevated"
        assert suppressed["response_sign"] == "suppressed"

    def test_a_per_session_rate_difference_does_not_break_the_shared_slope(self):
        """Sessions firing at 1x, 3x and 6x the same relationship: per-session standardization plus
        per-session intercepts must leave one slope serviceable for all three."""
        result = when_statistic(*self._unit(3.0, rate_scale=[1.0, 3.0, 6.0]), 0.02, 0.05, 40, 40)

        assert result["d2"] > 0.05
        assert result["auroc"] > 0.65
        assert len(result["per_fold"]) == 3

    def test_every_fold_is_reported(self):
        result = when_statistic(*self._unit(2.0, n_sessions=4), 0.02, 0.05, 40, 40)
        assert [fold["session"] for fold in result["per_fold"]] == [0, 1, 2, 3]
        assert result["n_windows"] == 1600
