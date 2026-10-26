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
- pooled_transfer_null under the frame grain: the memory budget sets only how many draws run together.
- The shipped transfer block carries the chosen values.
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
    pooled_transfer_null,
    score_fold,
    transfer_frames,
)

FPS = 100.0
TRANSFER = {"calibration_grain": "pooled_across_folds", "d2_reference_rate": "scored_frames",
            "universe": "bout_window", "pre_bout_seconds": 1.0, "inter_bout_threshold_seconds": 0.2,
            "grain": "count", "bin_seconds": 0.1, "null_block_megabytes": 64.0}


def spans(usvs: list, fps: float = FPS) -> tuple[np.ndarray, np.ndarray]:
    """
    Description
    -----------
    Builds span frames and their offsets for USVs given as (start, stop) seconds, laid out the way
    ``vocal_span_frames`` returns them.

    Parameters
    ----------
    usvs (list)
        ``(start, stop)`` pairs in seconds, one per USV.
    fps (float)
        Frame rate used to convert seconds to frame indices.

    Returns
    -------
    frames (np.ndarray)
        Concatenated span frames of every USV.
    pointer (np.ndarray)
        Offsets into ``frames``: USV ``i`` occupies ``frames[pointer[i]:pointer[i + 1]]``.
    """
    frames, pointer = [], [0]
    for start, stop in usvs:
        block = np.arange(round(start * fps), round(stop * fps))
        frames.append(block)
        pointer.append(pointer[-1] + block.size)
    return np.concatenate(frames), np.asarray(pointer)


class TestTransferFrames:
    """The frame universe the frozen quiet filter is scored on during vocal periods."""
    def test_the_span_universe_passes_the_spans_through(self):
        """Under ``universe = "usv_span"`` the output must be the USV span frames unchanged, so the span
        universe adds no run-up and merges nothing."""
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
        """USVs 3 s apart (beyond the inter-bout threshold) must form two bouts, each with its own 1 s run-up and
        nothing filled in between, so silence between bouts is never scored as vocal."""
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
        """An unrecognised universe must raise ``ValueError`` naming ``transfer.universe`` rather than fall back
        to a default frame set."""
        frames, pointer = spans([(1.0, 1.1)])
        with pytest.raises(ValueError, match=r"transfer\.universe"):
            transfer_frames(frames, pointer, FPS, 500, 10, np.zeros(0, dtype=np.int64), {**TRANSFER, "universe": "bouts"})


class TestBinsAndCounts:
    """Cutting contiguous frame runs into bins and counting spikes per frame."""
    def test_runs_are_cut_into_full_bins_only(self):
        """A 25-frame and a 12-frame run at bin width 10 must give three full bins (two, then one), dropping the
        partial remainders and never letting a bin straddle the gap between runs."""
        frames = np.concatenate([np.arange(10, 35), np.arange(100, 112)])
        bins = bin_frame_matrix(frames, 10)
        assert bins.shape == (3, 10)
        np.testing.assert_array_equal(bins[0], np.arange(10, 20))
        np.testing.assert_array_equal(bins[2], np.arange(100, 110))

    def test_multi_spike_frames_count_more_than_once(self):
        """Three spikes on one frame must count 3, and a spike past the last frame must be dropped, so the count
        grain sees every spike inside the session and nothing outside it."""
        counts = counts_per_frame(np.array([3, 3, 3, 7, 999]), 10)
        assert counts[3] == 3
        assert counts[7] == 1
        assert counts.sum() == 4


class TestPooledPoisson:
    """The pooled-across-sessions Poisson explained deviance used by the count grain."""
    @staticmethod
    def _bins(seed, gain):
        """
        Description
        -----------
        Builds two 2,000-bin sessions of Poisson counts whose log-rate is linear in a Gaussian predictor,
        with a 0.3 log-rate offset in the second session.

        Parameters
        ----------
        seed (int)
            Seed for ``np.random.default_rng``.
        gain (float)
            Log-rate coefficient of the predictor.

        Returns
        -------
        x (np.ndarray)
            The predictor, shape ``(4000,)``.
        counts (np.ndarray)
            The Poisson counts per bin.
        session (np.ndarray)
            Session index (0 or 1) of every bin.
        """
        generator = np.random.default_rng(seed)
        x = generator.normal(size=4000)
        session = np.repeat(np.arange(2), 2000)
        rate = np.exp(np.log(0.3) + gain * x + 0.3 * session)
        return x, generator.poisson(rate), session

    def test_a_planted_gain_is_recovered_with_its_sign(self):
        """A +0.5 planted gain must score D2 above 0.05 with slope above 0.3, and a -0.5 gain must give slope
        below -0.3, so the count grain keeps the sign the verdict reads."""
        x, counts, session = self._bins(0, 0.5)
        d2, slope = pooled_poisson_explained_deviance(x, counts, session, 30, 1.0)
        assert d2 > 0.05
        assert slope > 0.3
        x, counts, session = self._bins(1, -0.5)
        _d2, slope = pooled_poisson_explained_deviance(x, counts, session, 30, 1.0)
        assert slope < -0.3

    def test_shuffled_counts_explain_nothing(self):
        """Counts permuted against the predictor must explain essentially no deviance (|D2| < 0.01), so the
        cross-session offset alone cannot produce a score."""
        x, counts, session = self._bins(2, 0.5)
        d2, _slope = pooled_poisson_explained_deviance(x, np.random.default_rng(3).permutation(counts), session, 30, 1.0)
        assert abs(d2) < 0.01

    def test_a_flat_predictor_is_nan(self):
        """A constant predictor must return NaN for both D2 and slope, since there is no spread to calibrate."""
        d2, slope = pooled_poisson_explained_deviance(np.ones(100), np.ones(100, dtype=int), np.zeros(100, dtype=int), 30, 1.0)
        assert np.isnan(d2)
        assert np.isnan(slope)


class TestCountGrainCombine:
    """A two-session unit whose vocal-period spiking follows a known instantaneous filter of one feature,
    scored by the frozen filter through score_fold and combine_folds under both grains."""

    N_LAGS = 5

    @classmethod
    def _session(cls, seed, planted):
        """
        Description
        -----------
        Builds one synthetic session: a stationary unit-variance slow feature and spikes drawn as Poisson
        counts per frame, either following the planted filter on that feature or at a flat rate.

        Parameters
        ----------
        seed (int)
            Seed for ``np.random.default_rng``.
        planted (bool)
            True to drive the rate with the feature; False for a flat rate.

        Returns
        -------
        session (dict)
            ``feature_time_series``, ``n_frames``, ``spike_frames``, ``fps`` and ``quiet`` (the quiet anchors).
        """
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
        """
        Description
        -----------
        Scores a frozen filter matching the planted one on two synthetic sessions with ``score_fold`` and
        combines the folds with ``combine_folds`` under the requested grain.

        Parameters
        ----------
        planted (bool)
            Whether the sessions follow the planted filter.
        grain (str)
            ``"count"`` or ``"frame"``.

        Returns
        -------
        result (dict)
            The combined transfer result.
        """
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
        """A planted unit under the count grain must report 800 bins of 10 frames, a positive score and slope at
        the null floor, and one leave-one-session-out entry per session."""
        out = self._run(planted=True, grain="count")
        assert out["grain"] == "count"
        assert out["n_bins"] == 2 * (4000 // 10)
        assert out["bin_width_frames"] == 10
        assert out["score"] > 0.05
        assert out["slope"] > 0.0
        assert out["at_floor"]
        assert set(out["leave_one_session_out"]) == {"a", "b"}

    def test_a_flat_unit_does_not(self):
        """A unit with no planted filter must not pass (p > 0.05), so the count grain does not manufacture
        transfer."""
        out = self._run(planted=False, grain="count")
        assert out["p"] > 0.05

    def test_the_diagnostics_keep_the_per_frame_descriptors_and_drop_the_arrays(self):
        """Each stored fold must keep its scalar descriptors and drop the per-frame and per-bin arrays, so the
        per-unit artifact stays small."""
        out = self._run(planted=True, grain="count")
        fold = out["folds"][0]
        for key in ("auroc", "spike_rate", "n_frames", "metrics", "level", "n_bins"):
            assert key in fold
        for key in ("eta", "labels", "bin_eta", "bin_counts"):
            assert key not in fold

    def test_the_frame_grain_is_the_old_statistic(self):
        """The frame grain must report zero bins and score all 8,000 frames, so switching the grain leaves the
        previous per-frame statistic intact."""
        out = self._run(planted=True, grain="frame")
        assert out["grain"] == "frame"
        assert out["n_bins"] == 0
        assert out["score"] > 0.02
        assert out["n_frames"] == 8000

    @pytest.mark.parametrize("grain", ["count", "frame"])
    def test_a_session_with_no_transfer_frames_is_scored_as_empty_not_raised(self, grain):
        """A held-out session or block in which the animal did not vocalize has no transfer frames. The
        fold's own vocal score must come back empty -- NaN scores, ``n_frames`` 0, empty arrays -- rather
        than raising on an empty division, which crashed four folds of a one-session unit whose USVs all
        fell in its first block."""
        session = self._session(12, True)
        filters = np.zeros(self.N_LAGS)
        filters[-1] = 1.2
        model = FrameFilterModel(filters, -2.0, True, 1)
        transfer = {**TRANSFER, "grain": grain, "universe": "usv_span"}
        encoding = {"chunk_rows": 2000, "solver": {"calibration_steps": 30}, "transfer": transfer,
                    "calibration": {"scheme": "crossfit", "n_blocks": 5}}
        result = score_fold(model, session, [0], np.zeros(0, dtype=np.int64), self.N_LAGS, 0.1, encoding,
                            linear_predictor_at_frames, transfer)
        assert result["n_frames"] == 0
        assert np.isnan(result["fold_score"])
        assert np.isnan(result["fold_slope"])
        assert result["eta"].size == 0
        assert result["labels"].size == 0
        if grain == "count":
            assert result["n_bins"] == 0
            assert result["bin_counts"].size == 0

    def test_a_block_with_its_spikes_in_one_piece_is_left_out_and_named(self):
        """A third session whose few spikes all sit at the start of its transfer frames cannot be
        cross-fitted, which made the pooled transfer NaN on one-session units. It must be left out of the
        score, named in ``left_out_sessions`` with its reason, and leave the score finite."""
        sessions = {"a": self._session(10, True), "b": self._session(11, True), "c": self._session(13, True)}
        sessions["c"]["spike_frames"] = np.array([1510, 1540, 1600])
        filters = np.zeros(self.N_LAGS)
        filters[-1] = 1.2
        model = FrameFilterModel(filters, -2.0, True, 1)
        transfer = {**TRANSFER, "grain": "count", "universe": "usv_span"}
        encoding = {"chunk_rows": 2000, "solver": {"calibration_steps": 30}, "transfer": transfer,
                    "calibration": {"scheme": "crossfit", "n_blocks": 5}}
        settings = {"kinematic_encoding": encoding, "null": {"n_shuffles": 20, "shuffle_guard_seconds": 1.0,
                                                             "shuffle_seed": 0},
                    "significance": {"escalation_ladder": []}}
        frames = {sid: np.arange(1500, 5500) for sid in sessions}
        results = [score_fold(model, sessions[sid], [0], frames[sid], self.N_LAGS, 0.1, encoding,
                              linear_predictor_at_frames, transfer) for sid in sessions]
        out = combine_folds(results, sessions, list(sessions), frames, settings, message_output=lambda *_a: None)
        assert out["left_out_sessions"] == {"c": "events_in_one_piece"}
        assert np.isfinite(out["score"])
        assert out["score"] > 0.05

    def test_splitting_the_frame_grain_null_into_memory_blocks_changes_nothing(self):
        """The block size is a memory budget, not a statistical choice: a budget holding three draws' labels
        must give exactly the null of a budget holding all of them, including a short final block."""
        sessions = {"a": self._session(10, True), "b": self._session(11, True)}
        filters = np.zeros(self.N_LAGS)
        filters[-1] = 1.2
        model = FrameFilterModel(filters, -2.0, True, 1)
        transfer = {**TRANSFER, "grain": "frame", "universe": "usv_span"}
        encoding = {"chunk_rows": 2000, "solver": {"calibration_steps": 30}, "transfer": transfer,
                    "calibration": {"scheme": "crossfit", "n_blocks": 5}}
        frames = {sid: np.arange(1500, 5500) for sid in sessions}
        results = [score_fold(model, sessions[sid], [0], frames[sid], self.N_LAGS, 0.1, encoding,
                              linear_predictor_at_frames, transfer) for sid in sessions]
        total_frames = sum(int(result["eta"].size) for result in results)

        def null_with(block_megabytes):
            """The 10-draw pooled null under a given memory budget."""
            return pooled_transfer_null(results, sessions, list(sessions), frames, 10, 0, 1.0, 30,
                                        {**transfer, "null_block_megabytes": block_megabytes},
                                        encoding["calibration"])

        whole = null_with(64.0)
        blocked = null_with(total_frames * 8 * 3 / (1024 * 1024))
        assert whole.shape == (10,)
        np.testing.assert_allclose(blocked, whole)


class TestShippedTransferSettings:
    """The transfer block of the shipped settings file."""
    def test_the_shipped_values(self):
        """The shipped choice: bout window with a 4 s run-up at the repo's inter-bout
        threshold, Poisson counts per 100 ms bin."""
        with (pathlib.Path(__file__).resolve().parents[2] / "src" / "usv_playpen"
              / "_parameter_settings" / "neural_modeling_settings.json").open() as handle:
            block = json.load(handle)["kinematic_encoding"]["transfer"]
        assert block["universe"] == "bout_window"
        assert block["grain"] == "count"
        assert block["pre_bout_seconds"] == 4.0
        assert block["bin_seconds"] == 0.1
        assert block["inter_bout_threshold_seconds"] == pytest.approx(0.1155, abs=1e-4)
        assert block["null_block_megabytes"] == 64
        assert set(block) == {"calibration_grain", "d2_reference_rate", "universe", "pre_bout_seconds",
                              "inter_bout_threshold_seconds", "grain", "bin_seconds", "null_block_megabytes"}
