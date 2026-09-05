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

from itertools import pairwise

import numpy as np
import pytest

from usv_playpen.neural_modeling.kinematic_encoding import (
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
        nothing, and at min_sessions = 2 those units are in the cohort."""
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
