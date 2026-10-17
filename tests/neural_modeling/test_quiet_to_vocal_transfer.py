"""
@author: bartulem
Tests for the quiet-to-vocal transfer's universe and grain.

Coverage:
- transfer_frames: the USV-span universe passes spans through; the bout window merges USVs closer than
  the inter-bout threshold, adds the run-up, drops frames without a full history and every quiet anchor.
- bin_frame_matrix / counts_per_frame: contiguous runs cut into full bins, multi-spike frames counted.
- pooled_poisson_explained_deviance: positive on a planted gain, ~0 on shuffled counts, NaN on a flat
  predictor, the gain's sign recovered.
- combine_folds under the count grain: a planted unit passes, the per-frame descriptors survive, the
  diagnostics carry no bin arrays; the frame grain is untouched.
- The shipped transfer block carries the ruled values.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from usv_playpen.neural_modeling.deviance_metrics import (
    pooled_poisson_explained_deviance,
)
from usv_playpen.neural_modeling.kinematic_encoding import (
    FrameFilterModel,
    linear_predictor_at_frames,
)
from usv_playpen.neural_modeling.quiet_to_vocal_transfer import (
    bin_frame_matrix,
    combine_folds,
    counts_per_frame,
    score_fold,
    transfer_frames,
)

FPS = 100.0
TRANSFER = {"calibration_grain": "pooled_across_folds", "d2_reference_rate": "scored_frames",
            "universe": "bout_window", "pre_bout_seconds": 1.0, "inter_bout_threshold_seconds": 0.2,
            "grain": "count", "bin_seconds": 0.1}


def spans(usvs: list, fps: float = FPS) -> tuple[np.ndarray, np.ndarray]:
    """Span frames and offsets for USVs given as (start, stop) seconds, as vocal_span_frames lays them out."""
    frames, pointer = [], [0]
    for start, stop in usvs:
        block = np.arange(round(start * fps), round(stop * fps))
        frames.append(block)
        pointer.append(pointer[-1] + block.size)
    return np.concatenate(frames), np.asarray(pointer)


class TestTransferFrames:
    def test_the_span_universe_passes_the_spans_through(self):
        frames, pointer = spans([(2.0, 2.1), (2.15, 2.25)])
        out = transfer_frames(frames, pointer, FPS, 1000, 50, np.arange(0, 100), {**TRANSFER, "universe": "usv_span"})
        np.testing.assert_array_equal(out, frames)

    def test_the_bout_window_merges_close_usvs_and_adds_the_run_up(self):
        """Two USVs 50 ms apart are one bout; the window runs from 1 s before the first to the end of the
        second, including the gap between them."""
        frames, pointer = spans([(5.0, 5.1), (5.15, 5.25)])
        out = transfer_frames(frames, pointer, FPS, 2000, 50, np.zeros(0, dtype=np.int64), TRANSFER)
        np.testing.assert_array_equal(out, np.arange(400, 525))

    def test_a_long_gap_starts_a_new_bout_with_its_own_run_up(self):
        frames, pointer = spans([(5.0, 5.1), (8.0, 8.1)])
        out = transfer_frames(frames, pointer, FPS, 2000, 50, np.zeros(0, dtype=np.int64), TRANSFER)
        expected = np.concatenate([np.arange(400, 510), np.arange(700, 810)])
        np.testing.assert_array_equal(out, expected)

    def test_quiet_anchors_and_short_history_are_removed(self):
        """A quiet anchor inside the run-up would be both a fitting frame and a scoring frame; the first
        n_lags - 1 frames have no full history."""
        frames, pointer = spans([(0.6, 0.7)])
        quiet = np.arange(0, 20)
        out = transfer_frames(frames, pointer, FPS, 1000, 30, quiet, TRANSFER)
        assert out.min() >= 29
        assert not np.intersect1d(out, quiet).size
        assert out.max() == 69

    def test_an_unknown_universe_is_refused(self):
        frames, pointer = spans([(1.0, 1.1)])
        with pytest.raises(ValueError, match=r"transfer\.universe"):
            transfer_frames(frames, pointer, FPS, 500, 10, np.zeros(0, dtype=np.int64), {**TRANSFER, "universe": "bouts"})


class TestBinsAndCounts:
    def test_runs_are_cut_into_full_bins_only(self):
        frames = np.concatenate([np.arange(10, 35), np.arange(100, 112)])
        bins = bin_frame_matrix(frames, 10)
        assert bins.shape == (3, 10)
        np.testing.assert_array_equal(bins[0], np.arange(10, 20))
        np.testing.assert_array_equal(bins[2], np.arange(100, 110))

    def test_multi_spike_frames_count_more_than_once(self):
        counts = counts_per_frame(np.array([3, 3, 3, 7, 999]), 10)
        assert counts[3] == 3
        assert counts[7] == 1
        assert counts.sum() == 4


class TestPooledPoisson:
    @staticmethod
    def _bins(seed, gain):
        generator = np.random.default_rng(seed)
        x = generator.normal(size=4000)
        session = np.repeat(np.arange(2), 2000)
        rate = np.exp(np.log(0.3) + gain * x + 0.3 * session)
        return x, generator.poisson(rate), session

    def test_a_planted_gain_is_recovered_with_its_sign(self):
        x, counts, session = self._bins(0, 0.5)
        d2, slope = pooled_poisson_explained_deviance(x, counts, session, 30, 1.0)
        assert d2 > 0.05
        assert slope > 0.3
        x, counts, session = self._bins(1, -0.5)
        _d2, slope = pooled_poisson_explained_deviance(x, counts, session, 30, 1.0)
        assert slope < -0.3

    def test_shuffled_counts_explain_nothing(self):
        x, counts, session = self._bins(2, 0.5)
        d2, _slope = pooled_poisson_explained_deviance(x, np.random.default_rng(3).permutation(counts), session, 30, 1.0)
        assert abs(d2) < 0.01

    def test_a_flat_predictor_is_nan(self):
        d2, slope = pooled_poisson_explained_deviance(np.ones(100), np.ones(100, dtype=int), np.zeros(100, dtype=int), 30, 1.0)
        assert np.isnan(d2)
        assert np.isnan(slope)


class TestCountGrainCombine:
    """A two-session unit whose vocal-period spiking follows a known instantaneous filter of one feature,
    scored by the frozen filter through score_fold and combine_folds under both grains."""

    N_LAGS = 5

    @classmethod
    def _session(cls, seed, planted):
        generator = np.random.default_rng(seed)
        n_frames = 6000
        # A STATIONARY slow feature (smoothed noise at unit variance), as z-scored kinematics are. A random
        # walk drifts across several SD within a session, so a contiguous held-out block would sit outside
        # the range its cross-fitted calibration was estimated on.
        raw = np.convolve(generator.normal(size=n_frames + 99), np.ones(100), mode="valid")
        feature = (raw - raw.mean()) / raw.std()
        eta = -2.0 + 1.2 * feature if planted else np.full(n_frames, -2.0)
        counts = generator.poisson(0.4 * np.exp(np.clip(eta, -6, 2)))
        spike_frames = np.repeat(np.arange(n_frames), counts)
        return {"feature_time_series": feature[:, None], "n_frames": n_frames, "spike_frames": spike_frames,
                "fps": FPS, "quiet": np.arange(cls.N_LAGS, 1000)}

    @classmethod
    def _run(cls, planted, grain):
        sessions = {"a": cls._session(10, planted), "b": cls._session(11, planted)}
        filters = np.zeros(cls.N_LAGS)
        filters[-1] = 1.2
        model = FrameFilterModel(filters, -2.0, True, 1)
        transfer = {**TRANSFER, "grain": grain, "universe": "usv_span"}
        encoding = {"chunk_rows": 2000, "solver": {"calibration_steps": 30}, "transfer": transfer,
                    "calibration": {"scheme": "crossfit", "n_blocks": 5}}
        settings = {"kinematic_encoding": encoding, "null": {"n_shuffles": 60, "shuffle_guard_seconds": 1.0,
                                                             "shuffle_seed": 0},
                    "significance": {"escalation_ladder": []}}
        frames = {sid: np.arange(1500, 5500) for sid in sessions}
        results = [score_fold(model, sessions[sid], [0], frames[sid], cls.N_LAGS, 0.1, encoding,
                              linear_predictor_at_frames, transfer) for sid in sessions]
        return combine_folds(results, sessions, list(sessions), frames, settings, message_output=lambda *_a: None)

    def test_a_planted_unit_passes_under_the_count_grain(self):
        out = self._run(planted=True, grain="count")
        assert out["grain"] == "count"
        assert out["n_bins"] == 2 * (4000 // 10)
        assert out["bin_width_frames"] == 10
        assert out["score"] > 0.05
        assert out["slope"] > 0.0
        assert out["at_floor"]
        assert set(out["leave_one_session_out"]) == {"a", "b"}

    def test_a_flat_unit_does_not(self):
        out = self._run(planted=False, grain="count")
        assert out["p"] > 0.05

    def test_the_diagnostics_keep_the_per_frame_descriptors_and_drop_the_arrays(self):
        out = self._run(planted=True, grain="count")
        fold = out["folds"][0]
        for key in ("auroc", "spike_rate", "n_frames", "metrics", "level", "n_bins"):
            assert key in fold
        for key in ("eta", "labels", "bin_eta", "bin_counts"):
            assert key not in fold

    def test_the_frame_grain_is_the_old_statistic(self):
        out = self._run(planted=True, grain="frame")
        assert out["grain"] == "frame"
        assert out["n_bins"] == 0
        assert out["score"] > 0.02
        assert out["n_frames"] == 8000


class TestShippedTransferSettings:
    def test_the_ruled_values_are_shipped(self):
        """Ruled 2026-10-14 after the two-day test: bout window with a 4 s run-up at the repo's inter-bout
        threshold, Poisson counts per 100 ms bin."""
        with (pathlib.Path(__file__).resolve().parents[2] / "src" / "usv_playpen"
              / "_parameter_settings" / "neural_modeling_settings.json").open() as handle:
            block = json.load(handle)["kinematic_encoding"]["transfer"]
        assert block["universe"] == "bout_window"
        assert block["grain"] == "count"
        assert block["pre_bout_seconds"] == 4.0
        assert block["bin_seconds"] == 0.1
        assert block["inter_bout_threshold_seconds"] == pytest.approx(0.1155, abs=1e-4)
        assert set(block) == {"calibration_grain", "d2_reference_rate", "universe", "pre_bout_seconds",
                              "inter_bout_threshold_seconds", "grain", "bin_seconds"}
