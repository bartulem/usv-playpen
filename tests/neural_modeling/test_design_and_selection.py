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

from usv_playpen.neural_modeling import main_neural_encoding_dispatcher as dispatcher
from usv_playpen.neural_modeling.deviance_metrics import (
    calibrated_explained_deviance,
    pooled_calibrated_explained_deviance,
)
from usv_playpen.neural_modeling.kinematic_encoding import (
    FrameFilterModel,
    block_resample_anchors,
    filter_band,
    fit_quiet_model,
    forward_select,
    inner_folds,
    lag_basis,
    linear_predictor_at_frames,
    quiet_block_sessions,
)
from usv_playpen.neural_modeling.main_neural_encoding_dispatcher import (
    combine,
    load_settings,
    per_fold_roster,
    representative_fold,
    write_fold_artifact,
)
from usv_playpen.neural_modeling.neural_artifacts import read_unit_artifact
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
    """The quiet-anchor and vocal-span frame sets, and the disjointness between them."""

    def test_quiet_excludes_the_guard_band_around_every_usv(self):
        """A USV forbids [start - clean_post, stop + history_pre], so a quiet anchor's whole 4 s
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

    def test_vocal_spans_cover_the_usv_and_drop_usvs_without_full_history(self):
        """A vocal span runs from onset to offset in frames, and a USV too close to the recording start
        for a full 4 s history is masked out instead of being scored on a truncated design."""
        starts = np.array([0.5, 100.0])           # the first has no room for 4 s of history
        stops = np.array([0.6, 100.2])
        kept, frames, pointer = vocal_span_frames(starts, stops, FPS, 60_000, N_LAGS)

        assert kept.tolist() == [False, True]      # `kept` is a MASK over the input USVs
        assert pointer.size == 2
        assert frames.min() >= int(100.0 * FPS)
        assert frames.max() < int(np.ceil(100.2 * FPS))


class TestLaggedDesign:
    """Orientation and history requirement of the lagged predictor design."""

    def test_the_last_column_is_the_anchor_frame_itself(self):
        """Column 0 is the OLDEST lag and the final column is the anchor, so a filter plotted against
        'time before the frame' must be read right-to-left."""
        series = np.arange(1000, dtype=float).reshape(-1, 1)
        design = lagged_design(series, np.array([600, 700]), N_LAGS)

        assert design.shape == (2, N_LAGS)
        assert design[0, -1] == pytest.approx(600.0)
        assert design[0, 0] == pytest.approx(600.0 - (N_LAGS - 1))

    def test_anchors_without_full_history_are_rejected(self):
        """An anchor earlier than ``n_lags - 1`` would need lags before frame 0; it must raise rather
        than wrap around or zero-pad the history."""
        series = np.arange(1000, dtype=float).reshape(-1, 1)
        with pytest.raises(ValueError, match="n_lags-1"):
            lagged_design(series, np.array([10]), N_LAGS)


class TestSpikeLabels:
    """The per-frame spike labels the Bernoulli fit is trained on."""

    def test_labels_are_binary_and_mark_the_right_frames(self):
        """Labels are 1 exactly on frames holding at least one spike and stay binary on multi-spike
        frames, since the model is a per-frame Bernoulli, not a count model."""
        labels = spike_labels_at_frames(np.array([5, 5, 9]), np.array([4, 5, 6, 9]), 20)

        assert labels.tolist() == [0.0, 1.0, 0.0, 1.0]   # a multi-spike frame is still 1


class TestBlockCrossValidation:
    """The within-session block fallback used when the leave-one-session-out pool is too small."""

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
        """When the quiet frames cannot fill the requested blocks plus gaps, no blocks are returned, so
        the caller sees the shortfall instead of training on overlapping pieces."""
        session = {"quiet": np.arange(600, 1000), "spike_frames": np.array([1]),
                   "n_frames": 60_000, "fps": FPS,
                   "feature_time_series": np.zeros((60_000, 2)), "feature_names": ["a"]}
        assert len(quiet_block_sessions(session, "s", 5, 300)) == 0

    def test_a_single_pool_session_yields_an_empty_training_set(self):
        """Why the block fallback exists at all: leave-one-session-out on one session would fit on
        nothing, and at min_courtship_sessions = 2 those units are in the cohort."""
        assert inner_folds(["only"]) == [([], "only")]


class TestRepresentativeFold:
    """The rule that picks the one fold whose model is reported as the unit's representative."""

    @staticmethod
    def _fold(quiet, vocal_frames, no_model=False):
        """
        Description
        -----------
        Build the minimal fold record ``representative_fold`` reads.

        Parameters
        ----------
        quiet (float)
            The fold's held-out quiet score.
        vocal_frames (int)
            Number of vocal frames the fold scored (only the length of the array is used).
        no_model (bool)
            Whether the fold found no model; defaults to ``False``.

        Returns
        -------
        fold (dict)
            ``quiet_score``, a zero ``vocal_frames`` array of that length, and ``no_model``.
        """
        return {"quiet_score": quiet, "vocal_frames": np.zeros(vocal_frames), "no_model": no_model}

    def test_odd_counts_take_the_median(self):
        """With an odd number of usable folds the fold holding the median quiet score is chosen."""
        folds = [self._fold(0.09287, 14934), self._fold(0.21020, 14683), self._fold(0.16110, 4920)]
        assert representative_fold(folds) == 2          # the +0.16110 fold is the median

    def test_even_counts_break_the_tie_on_vocal_frames(self):
        """The two straddling folds are EQUIDISTANT from the median by construction -- the median is
        their midpoint -- so 'closest to the median' can never separate them."""
        folds = [self._fold(0.02019, 10447), self._fold(0.03100, 10695), self._fold(0.04678, 8564),
                 self._fold(0.04255, 4745), self._fold(0.02950, 27131), self._fold(0.04752, 7592)]
        assert representative_fold(folds) == 1          # 10,695 vocal frames beats 4,745

    def test_folds_without_a_model_are_ignored(self):
        """A no-model fold carries no filter to report, so it must not take part in the median."""
        folds = [self._fold(0.0, 0, no_model=True), self._fold(0.05, 100), self._fold(0.09, 100)]
        assert representative_fold(folds) == 1

    def test_no_usable_fold_returns_minus_one(self):
        """With no fold holding a model the sentinel -1 is returned rather than an index into a fold
        with nothing to report."""
        assert representative_fold([self._fold(0.0, 0, no_model=True)]) == -1

    def test_folds_with_a_nan_quiet_score_are_ignored(self):
        """A held-out block with no quiet spikes scores NaN. Sorting with a NaN key gives an arbitrary
        order, so such a fold could otherwise land in the middle and become the representative; it must
        be left out, and the median taken over the folds that can be ranked."""
        folds = [self._fold(np.nan, 900), self._fold(-0.029, 100), self._fold(np.nan, 900),
                 self._fold(-0.004, 100), self._fold(-0.043, 100)]
        assert representative_fold(folds) == 1          # median of -0.043, -0.029, -0.004

    def test_nan_folds_do_not_count_toward_the_even_tie_break(self):
        """Three NaN folds and two finite ones leave an even count of two; the tie-break on vocal
        frames decides between the finite pair only."""
        folds = [self._fold(0.014, 200), self._fold(-0.002, 300), self._fold(np.nan, 5000),
                 self._fold(np.nan, 5000), self._fold(np.nan, 5000)]
        assert representative_fold(folds) == 1

    def test_every_model_fold_nan_returns_minus_one(self):
        """With models but no finite quiet score anywhere, there is nothing to rank, so -1 is returned
        rather than an index into a fold whose score is undefined."""
        assert representative_fold([self._fold(np.nan, 10), self._fold(np.nan, 20)]) == -1


class TestFilterBand:
    """The reported band: refit the SELECTED model on block resamples and take the per-lag spread."""

    @staticmethod
    def _session(n_frames=4000, seed=0):
        """
        Description
        -----------
        Build one synthetic quiet session whose spikes follow a logistic function of a slow
        random-walk feature, so a filter fit on it has real signal to find.

        Parameters
        ----------
        n_frames (int)
            Session length in frames; defaults to 4000.
        seed (int)
            Seed for ``np.random.default_rng``; defaults to 0.

        Returns
        -------
        session (dict)
            ``feature_time_series`` (n_frames x 1), ``n_frames``, ``spike_frames`` and ``quiet``
            anchor frames.
        """
        generator = np.random.default_rng(seed)
        feature = np.cumsum(generator.normal(size=n_frames)) / 20.0
        rate = 1.0 / (1.0 + np.exp(-(0.8 * feature - 2.0)))
        spikes = np.flatnonzero(generator.random(n_frames) < rate)
        return {"feature_time_series": feature[:, None], "n_frames": n_frames,
                "spike_frames": spikes, "quiet": np.arange(20, n_frames - 5)}

    @staticmethod
    def _encoding():
        """
        Description
        -----------
        Return a small ``kinematic_encoding`` settings block (raw per-frame basis, short solver
        budget) that keeps the quiet fits in these tests fast.

        Parameters
        ----------
        None

        Returns
        -------
        encoding (dict)
            A self-contained ``kinematic_encoding`` block with every key ``fit_quiet_model`` reads.
        """
        return {"raw_lag_penalties": {"lambda_group": 0.0, "lambda_smooth": 1000.0,
                                      "lambda_ridge": 0.01},
                "smoothness_order": 1, "negatives_per_positive": 10, "max_train_total": 4000,
                "solver": {"max_iter": 300, "tol": 1e-6},
                "subsampling_correction": "log_prior_offset",
                "calibration": {"scheme": "crossfit", "n_blocks": 5},
                "session_intercepts": {"compute": True, "indicator_scale": 10.0, "max_sweeps": 30, "tolerance": 1e-4,
                                       "newton_max_steps": 25, "newton_step_tolerance": 1e-10},
                "temporal_basis": {"type": "none", "n_splines": 8, "spline_order": 3, "l2_lag_alpha": 100.0,
                                   "lambda_ridge": 0.1, "lambda_smooth": 0.0, "lambda_group": 0.0,
                                   "smoothness_order": 2, "spike_end_correction": "none"}}

    def test_block_resampling_keeps_the_count_and_the_ordering(self):
        """A block resample keeps the anchor count, stays sorted and draws only real anchors, so the
        refit sees a dataset of the same size and structure."""
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
        """An empty anchor set must come back empty rather than raise inside the block sampler."""
        assert block_resample_anchors(np.array([]), 10, np.random.default_rng(0)).size == 0

    def test_the_band_has_one_filter_per_resample_and_summarises_them(self):
        """The band stores one filter per resample, and its mean, sd (ddof=1) and low/high summaries are
        computed from exactly those filters, so the reported band is reproducible from what is stored."""
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
        """
        Description
        -----------
        Build a minimal settings dict with a valid ``feature_selection`` block, with any key
        overridden by keyword.

        Parameters
        ----------
        **overrides (dict)
            ``feature_selection`` keys to replace (e.g. ``selection_method``).

        Returns
        -------
        settings (dict)
            ``{"kinematic_encoding": {"feature_selection": ...}}``.
        """
        selection = {"screen_alpha": 0.01, "selection_method": "forward_d2",
                     "acceptance_rule": "paired_1se", **overrides}
        return {"kinematic_encoding": {"feature_selection": selection}}

    def test_an_unimplemented_selection_method_is_refused(self):
        """Declaring a selection method the code does not implement must raise before any fitting."""
        with pytest.raises(ValueError, match="forward"):
            forward_select({}, ["a", "b"], [0], 5, np.random.default_rng(0),
                           self._settings(selection_method="marginal_d2"), ["f"],
                           message_output=lambda *_a: None)

    def test_an_unimplemented_acceptance_rule_is_refused(self):
        """Declaring an acceptance rule other than the paired 1SE must raise before any fitting."""
        with pytest.raises(ValueError, match="paired"):
            forward_select({}, ["a", "b"], [0], 5, np.random.default_rng(0),
                           self._settings(acceptance_rule="unpaired_1se"), ["f"],
                           message_output=lambda *_a: None)

    def test_the_shipped_settings_declare_what_the_code_implements(self):
        """The shipped block names the implemented method and rule, and carries none of the
        stability-selection knobs of the method that was never built."""
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
    """The solver keys in the shipped settings are real and reach the solver."""

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


class TestFoldsWithoutAModel:
    """A fold that finds no model is the ORDINARY case at cohort scale -- an untuned unit produces one on
    every fold. Each such fold must still save a fold file, or `combine` stops with "fold artifact
    missing" and the unit saves nothing, and the per-fold lists must not read keys a no-model fold never
    carries."""

    N_LAGS = 5
    UNIT_ID = "imec1_cl0000_ch000_good"
    UNIT_UID = f"m_20250101_{UNIT_ID}"

    @classmethod
    def _model_fold(cls, fold_index, test_id, names, indices, seed):
        """
        Description
        -----------
        Build a fold record for a fold that found a model, carrying every key the fold writer and
        ``per_fold_roster`` read.

        Parameters
        ----------
        fold_index (int)
            The fold's position.
        test_id (str)
            The held-out session id.
        names (list[str])
            Selected feature names.
        indices (list[int])
            Selected feature indices.
        seed (int)
            Seed for the random filter coefficients.

        Returns
        -------
        fold (dict)
            A complete model fold with ``len(indices) * N_LAGS`` coefficients.
        """
        rng = np.random.default_rng(seed)
        return {"fold_index": fold_index, "test_id": test_id, "no_model": False,
                "selected": list(names), "selected_indices": list(indices),
                "path": [{"step": 0, "candidate": names[0], "decision": "ANCHOR"}],
                "screen": [{"feature": index, "survived": True} for index in indices],
                "coefficients": rng.normal(size=len(indices) * cls.N_LAGS), "intercept": -3.5,
                "quiet_score": 0.03, "quiet_p": 1e-5, "quiet_metrics": {"auroc": 0.7},
                "fit_seconds": 9.0, "n_iter": 2500}

    @staticmethod
    def _no_model_fold(fold_index, test_id):
        """
        Description
        -----------
        Build a fold record with ONLY the keys a no-model fold is guaranteed to carry: no quiet
        metrics, fit time, iteration count or filter -- and no quiet p, which one of the two no-model
        branches used to omit.

        Parameters
        ----------
        fold_index (int)
            The fold's position.
        test_id (str)
            The held-out session id.

        Returns
        -------
        fold (dict)
            A no-model fold whose screen table holds one near-miss feature (p = 5e-3).
        """
        return {"fold_index": fold_index, "test_id": test_id, "no_model": True,
                "selected": [], "path": [], "quiet_score": 0.0,
                "screen": [{"feature": 4, "survived": False, "p": 5e-3}]}

    @classmethod
    def _round_trip(cls, tmp_path, folds):
        """
        Description
        -----------
        Write the folds through the real fold writer and load them back exactly as `combine` loads
        them.

        Parameters
        ----------
        tmp_path (pathlib.Path)
            Directory the fold ``.npz`` files are written to.
        folds (list[dict])
            Fold records to write.

        Returns
        -------
        loaded (list[dict])
            The folds as read back from disk, in input order.
        """
        unit = {"unit_id": cls.UNIT_ID, "unit_uid": cls.UNIT_UID}
        for fold in folds:
            write_fold_artifact(fold, unit, fold["fold_index"], str(tmp_path), load_settings(),
                                message_output=lambda *_a: None)
        return [dict(np.load(tmp_path / f"{cls.UNIT_UID}_fold{fold['fold_index']}.npz", allow_pickle=True))
                for fold in folds]

    def _mixed(self):
        """
        Description
        -----------
        Build three folds -- model, no model, model -- with different selected feature sets.

        Parameters
        ----------
        None

        Returns
        -------
        folds (list[dict])
            Fold 0 selects ``a, b``; fold 1 has no model; fold 2 selects ``a``.
        """
        return [self._model_fold(0, "s0", ["a", "b"], [0, 1], seed=1),
                self._no_model_fold(1, "s1"),
                self._model_fold(2, "s2", ["a"], [0], seed=2)]

    def test_a_no_model_fold_still_writes_the_file_combine_looks_for(self, tmp_path):
        """A no-model fold must still write its ``.npz``, or ``combine`` stops on a missing fold and the
        untuned unit saves nothing."""
        write_fold_artifact(self._no_model_fold(0, "s0"), {"unit_id": self.UNIT_ID, "unit_uid": self.UNIT_UID},
                            0, str(tmp_path), load_settings(), message_output=lambda *_a: None)
        assert (tmp_path / f"{self.UNIT_UID}_fold0.npz").exists()

    def test_folds_are_named_by_unit_uid_so_units_sharing_a_unit_id_do_not_collide(self, tmp_path):
        """Two units with the same probe/cluster/channel name on different days must write different
        fold files into one cohort directory."""
        for date in ("20250101", "20250102"):
            write_fold_artifact(self._no_model_fold(0, "s0"),
                                {"unit_id": self.UNIT_ID, "unit_uid": f"m_{date}_{self.UNIT_ID}"},
                                0, str(tmp_path), load_settings(), message_output=lambda *_a: None)
        assert len(list(tmp_path.glob(f"*{self.UNIT_ID}_fold0.npz"))) == 2

    def test_combine_refuses_folds_written_under_other_settings(self, tmp_path):
        """A fold made under other settings must not be combined into a result labelled with these."""
        sessions = ["s0", "s1"]
        unit = {"unit_id": self.UNIT_ID, "unit_uid": self.UNIT_UID, "mouse_id": "m",
                "rec_date": 20250101, "courtship_sessions": sessions, "vocal_sessions": sessions}
        other = load_settings()
        other["null"]["shuffle_seed"] = int(other["null"]["shuffle_seed"]) + 1
        for fold_index, session in enumerate(sessions):
            write_fold_artifact(self._no_model_fold(fold_index, session), unit, fold_index, str(tmp_path),
                                other, message_output=lambda *_a: None)
        with pytest.raises(ValueError, match="other settings"):
            combine(unit, load_settings(), str(tmp_path), str(tmp_path), message_output=lambda *_a: None)

    def test_every_fold_is_listed_in_order_with_no_model_folds_recorded_as_absent(self, tmp_path):
        """Position i must stay fold i. Dropping the no-model fold would pair fold 2's filter with the
        wrong held-out session."""
        roster = per_fold_roster(self._round_trip(tmp_path, self._mixed()), self.N_LAGS)
        assert all(len(values) == 3 for values in roster.values())
        assert roster["no_model_per_fold"] == [False, True, False]
        assert [entry["test_session"] for entry in roster["filters_per_fold"]] == ["s0", "s1", "s2"]
        assert roster["quiet_p_per_fold"][1] == 1.0
        assert roster["quiet_metrics_per_fold"][1] is None
        assert roster["n_iter_per_fold"][1] is None
        assert np.isnan(roster["fit_seconds_per_fold"][1])
        assert roster["filters_per_fold"][1]["filter"] is None
        # the near-miss is what a figure of a marginal unit needs, so the screen table survives
        assert roster["screen_per_fold"][1][0]["p"] == 5e-3

    def test_every_folds_filter_is_kept_with_its_own_features(self, tmp_path):
        """Every fold's filter is kept, not only the representative's. Folds select different features,
        so each filter carries its own names and its own shape."""
        folds = self._mixed()
        roster = per_fold_roster(self._round_trip(tmp_path, folds), self.N_LAGS)
        first, last = roster["filters_per_fold"][0], roster["filters_per_fold"][2]
        assert first["features"] == ["a", "b"]
        assert first["filter"].shape == (2, self.N_LAGS)
        assert last["features"] == ["a"]
        assert last["filter"].shape == (1, self.N_LAGS)
        np.testing.assert_array_equal(first["filter"].ravel(), folds[0]["coefficients"])
        assert first["intercept"] == -3.5

    def test_combine_writes_a_unit_where_no_fold_found_a_model(self, tmp_path, monkeypatch):
        """The untuned unit must reach the cohort table as a result -- quiet p 1.0, transfer p NaN, a
        no_model flag -- not vanish as a failed job."""
        sessions = ["s0", "s1"]
        unit = {"unit_id": self.UNIT_ID, "unit_uid": self.UNIT_UID, "mouse_id": "m",
                "rec_date": 20250101, "courtship_sessions": sessions, "vocal_sessions": sessions}
        self._round_trip(tmp_path, [self._no_model_fold(0, "s0"), self._no_model_fold(1, "s1")])
        monkeypatch.setattr(dispatcher, "assemble_unit_sessions",
                            lambda *_a, **_k: {s: {"fps": 150.0, "feature_names": ["a", "b"], "n_frames": 1500,
                                                   "spike_frames": np.array([10, 20, 900, 1400]),
                                                   "quiet": np.array([20, 30, 900]), "window": (0, 1000)}
                                               for s in sessions})
        combine(unit, load_settings(), str(tmp_path), str(tmp_path), message_output=lambda *_a: None)
        stored = read_unit_artifact(str(tmp_path), unit["unit_uid"])["kinematic_encoding"]
        # three spikes inside the window [0, 1000), two of them on quiet anchors (20 and 900)
        assert stored["spike_counts_per_session"]["s0"] == {"spikes": 3, "quiet_spikes": 2,
                                                            "seconds": pytest.approx(1000 / 150.0)}
        assert stored["no_model"] is True
        assert stored["quiet_p"] == 1.0
        assert np.isnan(stored["transfer_p"])
        assert stored["no_model_per_fold"] == [True, True]
        assert stored["screen_per_fold"][0][0]["p"] == 5e-3


class TestLeaveOneSessionOut:
    """The guard against one session carrying the transfer, matching `min_leave_one_fold_out` on the
    decoding analyses. A descriptor, never a gate."""

    @staticmethod
    def _pooled(n_per_session, slopes, seed=0):
        """
        Description
        -----------
        Build pooled sessions whose eta predicts labels at the given per-session strength, on
        different per-session baselines.

        Parameters
        ----------
        n_per_session (list[int])
            Number of frames per session.
        slopes (list[float])
            Per-session strength of the eta-to-label relation (0 means no signal).
        seed (int)
            Seed for ``np.random.default_rng``; defaults to 0.

        Returns
        -------
        pooled (tuple)
            ``(eta, labels, session_index, session_ids)`` concatenated over sessions.
        """
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
    """Guards that refuse transfer configurations the scoring does not implement."""

    @staticmethod
    def _settings(**overrides):
        """
        Description
        -----------
        Build a minimal settings dict with a valid ``transfer`` block, with any key overridden by
        keyword.

        Parameters
        ----------
        **overrides (dict)
            ``transfer`` keys to replace.

        Returns
        -------
        settings (dict)
            ``{"kinematic_encoding": {"transfer": ...}}``.
        """
        return {"kinematic_encoding": {"transfer": {"calibration_grain": "pooled_across_folds",
                                                    "d2_reference_rate": "scored_frames",
                                                    **overrides}}}

    def test_fold_averaged_scoring_is_refused(self):
        """The transfer is calibrated once on the pooled folds; averaging per-fold scores lets a small
        session's noisy null swamp the average, so the fold-averaged grain is refused."""
        with pytest.raises(ValueError, match="ONE calibration"):
            combine_folds([], {}, [], {}, self._settings(calibration_grain="per_fold"),
                          message_output=lambda *_a: None)

    def test_a_non_local_reference_rate_is_refused(self):
        """D2 must be referenced to the scored session's own spike rate; a training-session rate would
        credit the model for a rate difference it did not predict."""
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
        assert set(encoding["transfer"]) == {"calibration_grain", "d2_reference_rate", "universe",
                                             "inter_bout_threshold_seconds", "grain", "bin_seconds",
                                             "null_block_megabytes"}


class TestTemporalBasis:
    """The optional spline lag basis: the modeling pipeline's layer on the neural quiet fit. It exists for
    compute -- measured at 50-250x less fit time with the same held-out D2, filter shape and transfer on
    three PAG units -- and must therefore hand every consumer the same kind of object the raw solver does:
    a per-frame filter."""

    N_LAGS = 60

    @staticmethod
    def _basis_settings(kind="bspline", alpha=100.0, correction="none"):
        """
        Description
        -----------
        Build a ``temporal_basis`` settings block.

        Parameters
        ----------
        kind (str)
            Basis type; defaults to ``"bspline"``.
        alpha (float)
            The lag-ridge ``l2_lag_alpha``; defaults to 100.0.
        correction (str)
            The ``spike_end_correction``; defaults to ``"none"``.

        Returns
        -------
        basis (dict)
            A complete ``temporal_basis`` block with 8 cubic splines.
        """
        return {"type": kind, "n_splines": 8, "spline_order": 3, "l2_lag_alpha": alpha,
                "lambda_ridge": 0.1, "lambda_smooth": 0.0, "lambda_group": 0.0, "smoothness_order": 2,
                "spike_end_correction": correction}

    @classmethod
    def _encoding(cls, kind):
        """
        Description
        -----------
        Build a ``kinematic_encoding`` block for the planted-filter sessions with the given basis.

        Parameters
        ----------
        kind (str)
            Basis type passed to ``_basis_settings`` (``"none"`` or ``"bspline"``).

        Returns
        -------
        encoding (dict)
            A complete ``kinematic_encoding`` block.
        """
        return {"raw_lag_penalties": {"lambda_group": 0.0, "lambda_smooth": 1000.0, "lambda_ridge": 0.01},
                "smoothness_order": 1, "negatives_per_positive": 10, "max_train_total": 6000,
                "solver": {"max_iter": 3000, "tol": 1e-6}, "subsampling_correction": "log_prior_offset",
                "calibration": {"scheme": "crossfit", "n_blocks": 5},
                "session_intercepts": {"compute": True, "indicator_scale": 10.0, "max_sweeps": 30, "tolerance": 1e-4,
                                       "newton_max_steps": 25, "newton_step_tolerance": 1e-10},
                "temporal_basis": cls._basis_settings(kind)}

    @classmethod
    def _session(cls, seed, n_frames=6000):
        """
        Description
        -----------
        Build a unit whose log-odds follow a smooth 0.4 s filter (summed weight 0.9) of one slow
        feature.

        Parameters
        ----------
        seed (int)
            Seed for ``np.random.default_rng``.
        n_frames (int)
            Session length in frames; defaults to 6000.

        Returns
        -------
        session (dict)
            ``feature_time_series``, ``n_frames``, ``spike_frames`` and ``quiet`` anchor frames.
        """
        generator = np.random.default_rng(seed)
        feature = np.cumsum(generator.normal(size=n_frames)) / 25.0
        lags = np.arange(cls.N_LAGS)
        planted = np.exp(-0.5 * ((lags - (cls.N_LAGS - 12)) / 6.0) ** 2)
        planted = 0.9 * planted / planted.sum()
        eta = np.convolve(feature, planted[::-1], mode="full")[: n_frames] - 2.2
        spikes = np.flatnonzero(generator.random(n_frames) < 1.0 / (1.0 + np.exp(-eta)))
        return {"feature_time_series": feature[:, None], "n_frames": n_frames, "spike_frames": spikes,
                "quiet": np.arange(cls.N_LAGS, n_frames - 5)}

    def test_an_unknown_basis_type_is_refused(self):
        """An unsupported basis type must raise rather than silently fall back to the raw path."""
        with pytest.raises(ValueError, match=r"temporal_basis\.type"):
            lag_basis(self.N_LAGS, self._basis_settings("fourier"))

    def test_none_keeps_one_weight_per_frame(self):
        """``"none"`` returns no basis, the signal for the raw one-weight-per-lag solver."""
        assert lag_basis(self.N_LAGS, self._basis_settings("none")) is None

    def test_the_basis_spans_the_window_and_is_scaled_to_lag_averages(self):
        """The basis has one row per lag and one column per spline, is finite and non-negative, and its
        columns average to unit mass, so spline weights read as lag-averaged filter values."""
        basis = lag_basis(self.N_LAGS, self._basis_settings())
        assert basis.shape == (self.N_LAGS, 8)
        assert np.all(np.isfinite(basis))
        assert np.all(basis >= 0.0)
        assert np.isclose(basis.sum(axis=0).mean(), 1.0)

    def test_an_unknown_spike_end_correction_is_refused(self):
        """An unsupported spike-end correction must raise rather than be ignored."""
        with pytest.raises(ValueError, match=r"spike_end_correction"):
            lag_basis(self.N_LAGS, self._basis_settings(correction="clamp"))

    def test_the_spike_end_correction_drops_the_outside_spline_and_cheapens_the_edge(self):
        """The outermost spline at the spike end (the last lag) is removed, and a spline partly outside
        the window is scaled up by the square root of its missing mass relative to an interior one."""
        plain = lag_basis(self.N_LAGS, self._basis_settings(alpha=0.0))
        corrected = lag_basis(self.N_LAGS, self._basis_settings(alpha=0.0, correction="drop_outside_and_rescale"))
        assert corrected.shape == (self.N_LAGS, 7)
        assert np.all(np.isfinite(corrected))
        assert np.isclose(corrected.sum(axis=0).mean(), 1.0)
        kept = plain[:, :-1].sum(axis=0)
        expected = kept * np.sqrt(kept.max() / kept)
        ratio = corrected.sum(axis=0) / corrected.sum(axis=0).max()
        np.testing.assert_allclose(ratio, expected / expected.max())
        assert ratio[-1] > (kept[-1] / kept.max())

    def test_the_lag_ridge_shrinks_the_oldest_splines_most(self):
        """Columns are scaled by 1/sqrt(1 + alpha lag^2), lag measured from the spike's frame, so under
        the ridge the oldest spline's column is the smallest relative to the newest."""
        flat = lag_basis(self.N_LAGS, self._basis_settings(alpha=0.0))
        ridged = lag_basis(self.N_LAGS, self._basis_settings(alpha=100.0))
        ratio = ridged.sum(axis=0) / flat.sum(axis=0)
        assert ratio[0] < ratio[-1]
        assert np.all(np.diff(ratio) >= -1e-12)

    def test_the_spline_fit_is_a_frame_filter_that_matches_the_raw_one(self):
        """Same data, same subsample: the spline fit must return a filter of n_lags weights per feature
        whose shape and held-out score agree with the raw solver's, and predict from a raw lagged design
        exactly as its coefficients say."""
        fit_session, test_session = self._session(seed=3), self._session(seed=4)
        raw, _ = fit_quiet_model({"a": fit_session}, ["a"], [0], self.N_LAGS, np.random.default_rng(0),
                                 self._encoding("none"), message_output=lambda *_a: None)
        spline, _ = fit_quiet_model({"a": fit_session}, ["a"], [0], self.N_LAGS, np.random.default_rng(0),
                                    self._encoding("bspline"), message_output=lambda *_a: None)
        assert isinstance(spline, FrameFilterModel)
        assert spline.coef_.shape == (self.N_LAGS,)
        assert spline.converged_
        assert np.corrcoef(raw.coef_, spline.coef_)[0, 1] > 0.8
        frames = test_session["quiet"]
        labels = spike_labels_at_frames(test_session["spike_frames"], frames, test_session["n_frames"])
        scores = {}
        for name, model in (("raw", raw), ("spline", spline)):
            eta = linear_predictor_at_frames(model, test_session["feature_time_series"], [0], frames,
                                             self.N_LAGS, 2000)
            scores[name], _slope = calibrated_explained_deviance(eta, labels, 30)
        assert scores["spline"] > 0.0
        assert abs(scores["spline"] - scores["raw"]) < 0.3 * max(scores["raw"], 1e-6)
        design = lagged_design(test_session["feature_time_series"][:, [0]], frames[:50], self.N_LAGS)
        np.testing.assert_allclose(spline.predict_eta(design), design @ spline.coef_ + spline.intercept_)

    @classmethod
    def _shifted_session(cls, seed, feature_shift, baseline_shift, n_frames=8000):
        """
        Description
        -----------
        Build the planted unit with the feature's mean and the session's firing level both moved.

        Parameters
        ----------
        seed (int)
            Seed for ``np.random.default_rng``.
        feature_shift (float)
            Mean the feature is re-centred on.
        baseline_shift (float)
            Amount added to the log-odds baseline.
        n_frames (int)
            Session length in frames; defaults to 8000.

        Returns
        -------
        session (dict)
            ``feature_time_series``, ``n_frames``, ``spike_frames`` and ``quiet`` anchor frames.
        """
        generator = np.random.default_rng(seed)
        feature = np.cumsum(generator.normal(size=n_frames)) / 25.0
        feature = feature - feature.mean() + feature_shift
        lags = np.arange(cls.N_LAGS)
        planted = np.exp(-0.5 * ((lags - (cls.N_LAGS - 12)) / 6.0) ** 2)
        planted = 0.9 * planted / planted.sum()
        eta = np.convolve(feature, planted[::-1], mode="full")[: n_frames] - 2.2 + baseline_shift
        spikes = np.flatnonzero(generator.random(n_frames) < 1.0 / (1.0 + np.exp(-eta)))
        return {"feature_time_series": feature[:, None], "n_frames": n_frames, "spike_frames": spikes,
                "quiet": np.arange(cls.N_LAGS, n_frames - 5)}

    def test_session_intercepts_keep_a_rate_difference_out_of_the_filter(self):
        """Two sessions share one filter (summed weight 0.9) but the second sits higher on the feature AND
        fires faster for reasons of its own. A single intercept can only explain the faster session through
        the feature, so its filter over-states the tuning; per-session intercepts recover it."""
        sessions = {"a": self._shifted_session(7, 0.0, 0.0), "b": self._shifted_session(8, 1.5, 1.0)}
        summed = {}
        for name, compute in (("shared", False), ("per_session", True)):
            encoding = self._encoding("bspline")
            encoding["session_intercepts"] = {"compute": compute, "indicator_scale": 10.0, "max_sweeps": 30,
                                          "tolerance": 1e-4, "newton_max_steps": 25,
                                          "newton_step_tolerance": 1e-10}
            model, _ = fit_quiet_model(sessions, ["a", "b"], [0], self.N_LAGS, np.random.default_rng(0),
                                       encoding, message_output=lambda *_a: None)
            summed[name] = float(model.coef_.sum())
        assert abs(summed["per_session"] - 0.9) < abs(summed["shared"] - 0.9)
        assert summed["shared"] > 0.9

    def test_the_raw_path_backfits_the_same_session_baselines(self):
        """The per-frame path cannot carry indicator columns (its smoothness penalty would couple them), so
        it backfits the baselines; it must correct the same confound the spline path's indicators do."""
        sessions = {"a": self._shifted_session(7, 0.0, 0.0), "b": self._shifted_session(8, 1.5, 1.0)}
        summed = {}
        for name, compute in (("shared", False), ("per_session", True)):
            encoding = self._encoding("none")
            encoding["session_intercepts"] = {"compute": compute, "indicator_scale": 10.0, "max_sweeps": 30,
                                              "tolerance": 1e-4, "newton_max_steps": 25,
                                              "newton_step_tolerance": 1e-10}
            model, _ = fit_quiet_model(sessions, ["a", "b"], [0], self.N_LAGS, np.random.default_rng(0),
                                       encoding, message_output=lambda *_a: None)
            summed[name] = float(np.asarray(model.coef_).sum())
        assert abs(summed["per_session"] - 0.9) < abs(summed["shared"] - 0.9)

    def test_the_shipped_default_is_the_spline_basis(self):
        """The shipped fit is parametrised in the spline basis, the raw per-frame path staying
        selectable as ``"none"``."""
        with (pathlib.Path(__file__).resolve().parents[2] / "src" / "usv_playpen"
              / "_parameter_settings" / "neural_modeling_settings.json").open() as handle:
            block = json.load(handle)["kinematic_encoding"]["temporal_basis"]
        assert block["type"] == "bspline"
        assert block["spike_end_correction"] == "drop_outside_and_rescale"
        assert set(block) == {"type", "n_splines", "spline_order", "l2_lag_alpha", "lambda_ridge", "lambda_smooth",
                              "lambda_group", "smoothness_order", "spike_end_correction"}

    def test_the_shipped_scoring_is_cross_fitted(self):
        """Every frozen-model score is calibrated on blocks that exclude the scored block."""
        with (pathlib.Path(__file__).resolve().parents[2] / "src" / "usv_playpen"
              / "_parameter_settings" / "neural_modeling_settings.json").open() as handle:
            block = json.load(handle)["kinematic_encoding"]["calibration"]
        assert block == {"scheme": "crossfit", "n_blocks": 5}

    def test_the_shipped_fit_has_per_session_intercepts(self):
        """Each training session gets its own unpenalised baseline."""
        with (pathlib.Path(__file__).resolve().parents[2] / "src" / "usv_playpen"
              / "_parameter_settings" / "neural_modeling_settings.json").open() as handle:
            block = json.load(handle)["kinematic_encoding"]["session_intercepts"]
        assert block == {"compute": True, "indicator_scale": 10.0, "max_sweeps": 30, "tolerance": 1e-4,
                                       "newton_max_steps": 25, "newton_step_tolerance": 1e-10}
