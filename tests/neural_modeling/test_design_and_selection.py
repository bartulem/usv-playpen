"""
@author: bartulem
Unit tests for the frame bookkeeping in ``neural_design_assembly`` and the selection rules in
``kinematic_encoding`` / ``main_neural_encoding_dispatcher``.

Coverage: the quiet/vocal frame definitions and their DISJOINTNESS, which is the property the whole
design rests on -- it is what lets one model be scored on every session's vocal frames; the lagged
design's orientation; the paired 1SE acceptance; the block-CV gap; and the representative-fold rule
including the even-count tie-break that a median alone cannot resolve.
"""

from __future__ import annotations

import json
import pathlib
from itertools import pairwise

import numpy as np
import pytest

from usv_playpen.neural_modeling.deviance_metrics import (
    pooled_calibrated_explained_deviance,
)
from usv_playpen.neural_modeling.kinematic_encoding import (
    block_resample_anchors,
    filter_band,
    fit_quiet_model,
    forward_select,
    inner_folds,
    quiet_block_sessions,
)
from usv_playpen.neural_modeling.main_neural_encoding_dispatcher import (
    representative_fold,
)
from usv_playpen.neural_modeling.neural_design_assembly import (
    lagged_design,
    quiet_anchor_frames,
    spike_labels_at_frames,
    vocal_span_frames,
)
from usv_playpen.neural_modeling.quiet_to_vocal_transfer import (
    combine_folds,
    leave_one_session_out_scores,
)

FPS = 150.0
HISTORY = 4.0
POST = 2.0
N_LAGS = int(HISTORY * FPS)


class TestFrameDefinitions:

    def test_quiet_excludes_the_guard_band_around_every_call(self):
        """A call forbids [start - clean_post, stop + history_pre], so a quiet anchor's whole 4 s
        history is vocalization-free by construction."""
        starts, stops = np.array([100.0]), np.array([100.5])
        quiet = quiet_anchor_frames(starts, stops, 60_000, FPS, HISTORY, POST)

        assert not np.any((quiet >= int((100.0 - POST) * FPS)) & (quiet < int((100.5 + HISTORY) * FPS)))
        assert np.any(quiet < int((100.0 - POST) * FPS))
        assert np.any(quiet >= int((100.5 + HISTORY) * FPS))

    def test_quiet_and_vocal_frames_never_overlap(self):
        """THE load-bearing property. Because fitting touches quiet anchors only, no model ever sees a
        vocal frame from ANY session -- which is what makes the pooled transfer legitimate everywhere
        and lets a single representative model be scored on every session."""
        generator = np.random.default_rng(0)
        starts = np.sort(generator.uniform(20.0, 340.0, 40))
        stops = starts + generator.uniform(0.03, 0.20, 40)

        quiet = quiet_anchor_frames(starts, stops, 60_000, FPS, HISTORY, POST)
        _kept, vocal, _pointer = vocal_span_frames(starts, stops, FPS, 60_000, N_LAGS)

        assert np.intersect1d(quiet, vocal).size == 0

    def test_vocal_spans_cover_the_call_and_drop_calls_without_full_history(self):
        starts = np.array([0.5, 100.0])           # the first has no room for 4 s of history
        stops = np.array([0.6, 100.2])
        kept, frames, pointer = vocal_span_frames(starts, stops, FPS, 60_000, N_LAGS)

        assert kept.tolist() == [False, True]      # `kept` is a MASK over the input calls
        assert pointer.size == 2
        assert frames.min() >= int(100.0 * FPS)
        assert frames.max() < int(np.ceil(100.2 * FPS))


class TestLaggedDesign:

    def test_the_last_column_is_the_anchor_frame_itself(self):
        """Column 0 is the OLDEST lag and the final column is the anchor, so a filter plotted against
        'time before the frame' must be read right-to-left."""
        series = np.arange(1000, dtype=float).reshape(-1, 1)
        design = lagged_design(series, np.array([600, 700]), N_LAGS)

        assert design.shape == (2, N_LAGS)
        assert design[0, -1] == pytest.approx(600.0)
        assert design[0, 0] == pytest.approx(600.0 - (N_LAGS - 1))

    def test_anchors_without_full_history_are_rejected(self):
        series = np.arange(1000, dtype=float).reshape(-1, 1)
        with pytest.raises(ValueError, match="n_lags-1"):
            lagged_design(series, np.array([10]), N_LAGS)


class TestSpikeLabels:

    def test_labels_are_binary_and_mark_the_right_frames(self):
        labels = spike_labels_at_frames(np.array([5, 5, 9]), np.array([4, 5, 6, 9]), 20)

        assert labels.tolist() == [0.0, 1.0, 0.0, 1.0]   # a multi-spike frame is still 1


class TestBlockCrossValidation:

    def test_blocks_are_separated_by_at_least_the_requested_gap(self):
        """The gap must exceed the predictor history or a validation frame's lags reach into a training
        block. Trimming is per END, so the caller passes half the required separation."""
        session = {"quiet": np.arange(600, 50_600), "spike_frames": np.array([1]),
                   "n_frames": 60_000, "fps": FPS,
                   "feature_time_series": np.zeros((60_000, 2)), "feature_names": ["a", "b"]}
        required = max(3.0, HISTORY)
        blocks = quiet_block_sessions(session, "s", 5, int(np.ceil(required * FPS / 2.0)))
        pieces = list(blocks.values())

        assert len(blocks) == 5
        for earlier, later in pairwise(pieces):
            assert later["quiet"][0] - earlier["quiet"][-1] >= required * FPS

    def test_too_few_usable_blocks_is_reported_not_hidden(self):
        session = {"quiet": np.arange(600, 1000), "spike_frames": np.array([1]),
                   "n_frames": 60_000, "fps": FPS,
                   "feature_time_series": np.zeros((60_000, 2)), "feature_names": ["a"]}
        assert len(quiet_block_sessions(session, "s", 5, 300)) == 0

    def test_a_single_pool_session_yields_an_empty_training_set(self):
        """Why the block fallback exists at all: leave-one-session-out on one session would fit on
        nothing, and at min_courtship_sessions = 2 those units are in the cohort."""
        assert inner_folds(["only"]) == [([], "only")]


class TestRepresentativeFold:

    @staticmethod
    def _fold(quiet, vocal_frames, no_model=False):
        return {"quiet_score": quiet, "vocal_frames": np.zeros(vocal_frames), "no_model": no_model}

    def test_odd_counts_take_the_median(self):
        folds = [self._fold(0.09287, 14934), self._fold(0.21020, 14683), self._fold(0.16110, 4920)]
        assert representative_fold(folds) == 2          # cl0401: the +0.16110 fold

    def test_even_counts_break_the_tie_on_vocal_frames(self):
        """The two straddling folds are EQUIDISTANT from the median by construction -- the median is
        their midpoint -- so 'closest to the median' can never separate them."""
        folds = [self._fold(0.02019, 10447), self._fold(0.03100, 10695), self._fold(0.04678, 8564),
                 self._fold(0.04255, 4745), self._fold(0.02950, 27131), self._fold(0.04752, 7592)]
        assert representative_fold(folds) == 1          # cl0499: 10,695 frames beats 4,745

    def test_folds_without_a_model_are_ignored(self):
        folds = [self._fold(0.0, 0, no_model=True), self._fold(0.05, 100), self._fold(0.09, 100)]
        assert representative_fold(folds) == 1

    def test_no_usable_fold_returns_minus_one(self):
        assert representative_fold([self._fold(0.0, 0, no_model=True)]) == -1


class TestFilterBand:
    """The reported band: refit the SELECTED model on block resamples and take the per-lag spread."""

    @staticmethod
    def _session(n_frames=4000, seed=0):
        generator = np.random.default_rng(seed)
        feature = np.cumsum(generator.normal(size=n_frames)) / 20.0
        rate = 1.0 / (1.0 + np.exp(-(0.8 * feature - 2.0)))
        spikes = np.flatnonzero(generator.random(n_frames) < rate)
        return {"feature_time_series": feature[:, None], "n_frames": n_frames,
                "spike_frames": spikes, "quiet": np.arange(20, n_frames - 5)}

    @staticmethod
    def _encoding():
        return {"significance_model": {"lambda_group": 0.0, "lambda_smooth": 1000.0,
                                       "lambda_ridge": 0.01},
                "smoothness_order": 1, "negatives_per_positive": 10, "max_train_total": 4000,
                "solver": {"max_iter": 300, "tol": 1e-6},
                "subsampling_correction": "log_prior_offset"}

    def test_block_resampling_keeps_the_count_and_the_ordering(self):
        anchors = np.arange(1000)
        drawn = block_resample_anchors(anchors, 50, np.random.default_rng(0))
        assert drawn.size == anchors.size
        assert np.all(np.diff(drawn) >= 0)
        assert set(drawn.tolist()) <= set(anchors.tolist())

    def test_blocks_are_contiguous_runs_not_scattered_frames(self):
        """The unit of resampling is a block, because quiet frames a few tens of milliseconds apart
        share almost their whole predictor. Resampling frames independently would treat hundreds of
        thousands of them as independent observations and report a band an order of magnitude too
        narrow."""
        drawn = block_resample_anchors(np.arange(2000), 100, np.random.default_rng(1))
        steps = np.diff(np.unique(drawn))
        # a contiguous-block draw is mostly consecutive; an iid draw over 2000 from 2000 would not be
        assert np.mean(steps == 1) > 0.9

    def test_an_empty_anchor_set_is_returned_unchanged(self):
        assert block_resample_anchors(np.array([]), 10, np.random.default_rng(0)).size == 0

    def test_the_band_has_one_filter_per_resample_and_summarises_them(self):
        sessions = {"a": self._session(seed=0), "b": self._session(seed=1)}
        band = filter_band(sessions, ["a", "b"], [0], 5, np.random.default_rng(0),
                           self._encoding(), 4, message_output=lambda *_a: None)

        assert band["filters"].shape == (4, 1, 5)
        assert band["mean"].shape == (1, 5)
        assert np.allclose(band["mean"], band["filters"].mean(axis=0))
        assert np.allclose(band["sd"], band["filters"].std(axis=0, ddof=1))
        assert np.all(band["low"] <= band["high"])
        assert band["intercepts"].size == 4

    def test_the_feature_set_is_frozen_across_resamples(self):
        """The band must show how much the COEFFICIENTS move, not how much the selection moves."""
        sessions = {"a": self._session(seed=2), "b": self._session(seed=3)}
        band = filter_band(sessions, ["a", "b"], [0], 5, np.random.default_rng(0),
                           self._encoding(), 3, message_output=lambda *_a: None)
        assert band["filters"].shape[1] == 1        # one feature in, one feature out, every time


class TestSelectionGuards:
    """`selection_method` and `acceptance_rule` name the procedure `forward_select` IS. Unchecked they
    are decoration -- a run could declare one thing in its settings, be reported as such, and have run
    the other all along."""

    @staticmethod
    def _settings(**overrides):
        selection = {"screen_alpha": 0.01, "selection_method": "forward_d2",
                     "acceptance_rule": "paired_1se", **overrides}
        return {"kinematic_encoding": {"feature_selection": selection}}

    def test_an_unimplemented_selection_method_is_refused(self):
        with pytest.raises(ValueError, match="forward"):
            forward_select({}, ["a", "b"], [0], 5, np.random.default_rng(0),
                           self._settings(selection_method="marginal_d2"), ["f"],
                           message_output=lambda *_a: None)

    def test_an_unimplemented_acceptance_rule_is_refused(self):
        with pytest.raises(ValueError, match="paired"):
            forward_select({}, ["a", "b"], [0], 5, np.random.default_rng(0),
                           self._settings(acceptance_rule="unpaired_1se"), ["f"],
                           message_output=lambda *_a: None)

    def test_the_shipped_settings_declare_what_the_code_implements(self):
        with (pathlib.Path(__file__).resolve().parents[2] / "src" / "usv_playpen"
              / "_parameter_settings" / "neural_modeling_settings.json").open() as handle:
            selection = json.load(handle)["kinematic_encoding"]["feature_selection"]
        assert selection["selection_method"] == "forward_d2"
        assert selection["acceptance_rule"] == "paired_1se"
        # the stability-selection knobs described a method that was rejected before it was built
        for gone in ("stability_selection", "pi_threshold", "n_bootstraps", "subsample_size",
                     "lambda_group_path"):
            assert gone not in selection


class TestSolverSettings:

    def test_tol_reaches_the_solver(self):
        """`tol` sat in settings unpassed, and the solver's own default was the SAME value -- so it
        was ignored while looking exactly like it worked. A loose tolerance must stop the fit early,
        or the key is decoration again."""
        session = TestFilterBand._session(seed=5)
        encoding = TestFilterBand._encoding()
        encoding["solver"] = {"max_iter": 4000, "tol": 1e-9}
        tight, _rate = fit_quiet_model({"a": session}, ["a"], [0], 5, np.random.default_rng(0),
                                       encoding, message_output=lambda *_a: None)
        encoding["solver"] = {"max_iter": 4000, "tol": 1.0}
        loose, _rate = fit_quiet_model({"a": session}, ["a"], [0], 5, np.random.default_rng(0),
                                       encoding, message_output=lambda *_a: None)
        assert loose.n_iter_ < tight.n_iter_

    def test_the_shipped_solver_block_has_no_learning_rate(self):
        """FISTA takes its step from a backtracking Lipschitz estimate; there is no learning rate to
        set, and the key named a solver parameter that does not exist."""
        with (pathlib.Path(__file__).resolve().parents[2] / "src" / "usv_playpen"
              / "_parameter_settings" / "neural_modeling_settings.json").open() as handle:
            solver = json.load(handle)["kinematic_encoding"]["solver"]
        assert "learning_rate" not in solver
        assert set(solver) == {"max_iter", "tol", "calibration_steps", "null_calibration_bins"}


class TestLeaveOneSessionOut:
    """The guard against one session carrying the transfer, matching `min_leave_one_fold_out` on the
    decoding claims. A descriptor, never a gate."""

    @staticmethod
    def _pooled(n_per_session, slopes, seed=0):
        """Sessions whose eta predicts labels at the given per-session strength, on different baselines."""
        rng = np.random.default_rng(seed)
        eta, labels, index = [], [], []
        for session, (count, strength) in enumerate(zip(n_per_session, slopes, strict=True)):
            values = rng.normal(size=count)
            rate = 1.0 / (1.0 + np.exp(-(strength * values - 1.0 - 0.5 * session)))
            eta.append(values)
            labels.append((rng.random(count) < rate).astype(np.float64))
            index.append(np.full(count, session))
        return (np.concatenate(eta), np.concatenate(labels), np.concatenate(index),
                [f"s{i}" for i in range(len(n_per_session))])

    def test_a_session_carrying_the_result_shows_up_as_a_low_minimum(self):
        """The whole purpose: drop the one session that carries the signal and the pooled score falls.
        Dropping any of the others leaves it roughly where it was."""
        eta, labels, index, ids = self._pooled([4000, 4000, 4000], [2.5, 0.0, 0.0], seed=1)
        scores = leave_one_session_out_scores(eta, labels, index, ids, 40)
        assert min(scores, key=lambda k: scores[k]["score"]) == "s0"
        assert scores["s0"]["score"] < scores["s1"]["score"]
        assert scores["s0"]["score"] < scores["s2"]["score"]

    def test_a_shared_effect_survives_every_drop(self):
        """The reassuring case, and the one the descriptor exists to certify: no single session is
        load-bearing, so the minimum stays positive."""
        eta, labels, index, ids = self._pooled([3000, 3000, 3000], [1.2, 1.2, 1.2], seed=2)
        scores = leave_one_session_out_scores(eta, labels, index, ids, 40)
        assert min(entry["score"] for entry in scores.values()) > 0.0
        assert all(entry["slope"] > 0.0 for entry in scores.values())

    def test_it_re_pools_rather_than_scoring_the_dropped_session_alone(self):
        """The reason this is not just the per-session diagnostics already reported. Each drop must be
        the pooled calibration on the REMAINING sessions -- which is what makes it robust on a unit
        with one thin session, where the per-session fit is the fragile construction pooling avoids."""
        eta, labels, index, ids = self._pooled([3000, 3000, 60], [1.5, 1.5, 1.5], seed=3)
        scores = leave_one_session_out_scores(eta, labels, index, ids, 40)
        keep = index != 2
        expected, expected_slope = pooled_calibrated_explained_deviance(eta[keep], labels[keep],
                                                                       index[keep], 40)
        assert scores["s2"]["score"] == pytest.approx(expected)
        assert scores["s2"]["slope"] == pytest.approx(expected_slope)

    def test_one_session_yields_nothing_rather_than_a_misleading_number(self):
        """Dropping the only session leaves nothing to pool, so there is no leave-one-out statistic --
        the caller reports NaN rather than a value computed on an empty set."""
        eta, labels, index, ids = self._pooled([2000], [1.5], seed=4)
        assert leave_one_session_out_scores(eta, labels, index, ids, 40) == {}


class TestTransferGuards:

    @staticmethod
    def _settings(**overrides):
        return {"kinematic_encoding": {"transfer": {"calibration_grain": "pooled_across_folds",
                                                    "d2_reference_rate": "scored_frames",
                                                    **overrides}}}

    def test_fold_averaged_scoring_is_refused(self):
        """Pooling is what stopped a 372-call session's null -- mean four times its own observed --
        swamping the average, and it flipped cl0401 from p 0.065 to 9.9e-06."""
        with pytest.raises(ValueError, match="ONE calibration"):
            combine_folds([], {}, [], {}, self._settings(calibration_grain="per_fold"),
                          message_output=lambda *_a: None)

    def test_a_non_local_reference_rate_is_refused(self):
        with pytest.raises(ValueError, match="own spike rate"):
            combine_folds([], {}, [], {}, self._settings(d2_reference_rate="training_sessions"),
                          message_output=lambda *_a: None)

    def test_the_screen_slope_gate_lives_with_the_screen(self):
        """It governs which FEATURES may enter selection. The verdict's slope condition is a
        different thing and is definitional -- its sign separates transformation from
        regime-switching -- so it is deliberately not a setting."""
        with (pathlib.Path(__file__).resolve().parents[2] / "src" / "usv_playpen"
              / "_parameter_settings" / "neural_modeling_settings.json").open() as handle:
            encoding = json.load(handle)["kinematic_encoding"]
        assert encoding["feature_selection"]["screen_require_positive_slope"] is True
        assert "require_positive_slope" not in encoding["transfer"]
        assert set(encoding["transfer"]) == {"calibration_grain", "d2_reference_rate"}
