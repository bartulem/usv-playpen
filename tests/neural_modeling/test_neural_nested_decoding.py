"""
@author: bartulem
Coverage: claim 3's block-diagonal penalty and the behaviour control it is fitted against.

The penalty is the piece that makes a mixed design legal -- five behavioural features wanting 600 lags
of smoothing beside one neural column that must receive ridge only -- so these tests pin its SHAPE
(zeros off the behaviour blocks, zero on the neural diagonal) and its IDENTITY to the parent's
operator, since a second copy of the finite-difference operator could drift from the one P1 selected
its features under. They also pin the reduced model to P1's own result file rather than to a literal.
"""

from __future__ import annotations

import copy
import json
import pathlib
import pickle
import warnings
from typing import ClassVar

import numpy as np
import pytest

# Importing the estimator pulls in the JAX/Optax parent class, which sets a flag JAX v0.9 has
# deprecated; that one-time DeprecationWarning fires at import (collection) time, where the project's
# `filterwarnings = ["error"]` would turn it into a hard collection error. Suppressed just for this
# import, matching `tests/modeling/test_manifold_torus_regression.py` -- it is a third-party
# transition warning, not a defect under test.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from usv_playpen.modeling.manifold_torus_regression import (
        SmoothTorusManifoldRegression,
    )
    from usv_playpen.neural_modeling.neural_nested_decoding import (
        NestedTorusRegression,
        nested_design,
        nested_scores,
        reduced_model_features,
        reduced_predictions,
        shifted_neural_column,
    )

SETTINGS_PATH = (pathlib.Path(__file__).resolve().parents[2] / "src" / "usv_playpen"
                 / "_parameter_settings" / "neural_modeling_settings.json")


def _settings() -> dict:
    with SETTINGS_PATH.open() as handle:
        return json.load(handle)


class TestBlockDiagonalPenalty:
    """One temporal block per behavioural feature, and NOTHING on the neural column."""

    def test_the_neural_column_is_unsmoothed(self):
        est = NestedTorusRegression(n_behaviour_features=3, n_lags=10, n_neural_columns=1)
        penalty = est._smoothness_penalty()
        assert penalty.shape == (31, 31)
        assert penalty[30, 30] == 0.0
        assert np.abs(penalty[:30, 30]).max() == 0.0      # no cross terms either

    def test_features_do_not_smooth_into_each_other(self):
        """A single 600-lag block spanning two features would tie the end of one filter to the start
        of the next -- adjacent in the design, unrelated in time."""
        est = NestedTorusRegression(n_behaviour_features=2, n_lags=8, n_neural_columns=0)
        penalty = est._smoothness_penalty()
        assert penalty.shape == (16, 16)
        assert np.abs(penalty[:8, 8:]).max() == 0.0

    @pytest.mark.parametrize("order", [1, 2])
    def test_the_block_is_the_parents_operator_not_a_second_copy(self, order):
        """Including the reflective (Neumann) boundary rows the parent adds at order 2. A rebuilt
        operator could drift from the one the behaviour control was selected under."""
        parent = SmoothTorusManifoldRegression(n_features=1, n_time_bins=12,
                                               smoothness_derivative_order=order, metric="torus")
        nested = NestedTorusRegression(n_behaviour_features=1, n_lags=12, n_neural_columns=0,
                                       smoothness_derivative_order=order)
        np.testing.assert_allclose(nested._smoothness_penalty(), parent._smoothness_penalty())

    def test_the_reduced_model_penalty_has_no_neural_block(self):
        est = NestedTorusRegression(n_behaviour_features=5, n_lags=20, n_neural_columns=0)
        assert est._smoothness_penalty().shape == (100, 100)

    def test_the_shipped_shape_is_five_features_by_six_hundred_lags_plus_one(self):
        est = NestedTorusRegression(n_behaviour_features=5, n_lags=600, n_neural_columns=1)
        assert est._smoothness_penalty().shape == (3001, 3001)


class TestTheEstimatorFits:
    """The parent hard-checks column count against `n_features * n_time_bins`, so a mixed design only
    fits at all if this class reports its shape the way it does."""

    @staticmethod
    def _design(n=240, n_features=3, n_lags=8, seed=0, neural_signal=0.0):
        rng = np.random.default_rng(seed)
        behaviour = rng.normal(size=(n, n_features * n_lags))
        neural = rng.normal(size=(n, 1))
        angle = (0.6 * behaviour[:, 0] + 0.4 * behaviour[:, n_lags]
                 + neural_signal * neural[:, 0] + 0.05 * rng.normal(size=n))
        y = np.column_stack([angle % 1.0, (0.5 * angle) % 1.0])
        return behaviour, neural, y

    def test_a_mixed_design_fits(self):
        behaviour, neural, y = self._design()
        est = NestedTorusRegression(n_behaviour_features=3, n_lags=8, n_neural_columns=1)
        est.fit(np.hstack([behaviour, neural]), y)
        assert est.coef_.shape == (25, 4)          # 4-D torus embedding
        assert est.converged_

    def test_a_wrong_column_count_is_still_refused(self):
        behaviour, _neural, y = self._design()
        est = NestedTorusRegression(n_behaviour_features=3, n_lags=8, n_neural_columns=1)
        with pytest.raises(ValueError, match="columns"):
            est.fit(behaviour, y)                  # the neural column is missing

    def test_the_neural_column_still_receives_the_ridge(self):
        """Zero SMOOTHNESS must not mean zero regularisation -- `l2_reg` reaches every column through
        the parent's `+ l2_reg * I`, so a larger ridge must shrink the neural coefficient."""
        behaviour, neural, y = self._design(neural_signal=1.0)
        design = np.hstack([behaviour, neural])
        weak = NestedTorusRegression(n_behaviour_features=3, n_lags=8, n_neural_columns=1,
                                     l2_reg=1e-4).fit(design, y)
        strong = NestedTorusRegression(n_behaviour_features=3, n_lags=8, n_neural_columns=1,
                                       l2_reg=1e3).fit(design, y)
        assert np.abs(strong.coef_[-1]).sum() < np.abs(weak.coef_[-1]).sum()

    def test_the_full_model_tracks_a_neuron_the_behaviour_cannot_supply(self):
        """The synthetic gate for the nesting: when the target depends on a neural column that is
        independent of the behaviour block, the full model must fit better in-sample than the reduced
        one. (Out-of-sample scoring is the statistic; this only pins the mechanics.)"""
        behaviour, neural, y = self._design(neural_signal=2.0, seed=3)
        reduced = NestedTorusRegression(n_behaviour_features=3, n_lags=8,
                                        n_neural_columns=0).fit(behaviour, y)
        full = NestedTorusRegression(n_behaviour_features=3, n_lags=8,
                                     n_neural_columns=1).fit(np.hstack([behaviour, neural]), y)
        residual = lambda est, x: float(np.mean((est.predict(x, snap=False) - y) ** 2))  # noqa: E731
        assert residual(full, np.hstack([behaviour, neural])) < residual(reduced, behaviour)


class TestTheBehaviourControl:
    """Read P1's selection AND assert it against the frozen list -- reading alone lets a P1 re-run
    silently redefine 'beyond behaviour'; freezing alone lets the control drift from the selection."""

    FROZEN: ClassVar[list[str]] = ["self.neck_elevation", "nose-nose", "allo_yaw-nose",
                                   "self.back_pitch", "self.allo_roll"]

    @staticmethod
    def _fake_selection(tmp_path, features):
        path = tmp_path / "selection.pkl"
        with path.open("wb") as handle:
            pickle.dump({"steps": [{"final_model_features": list(features)}]}, handle)
        return path

    def test_the_shipped_settings_freeze_p1s_five_features(self):
        assert _settings()["nested_position_decoding"]["reduced_model_features"] == self.FROZEN

    def test_a_matching_file_returns_the_selection_order(self, tmp_path):
        settings = _settings()
        settings["data"]["behaviour_selection_result_path"] = str(
            self._fake_selection(tmp_path, self.FROZEN))
        assert reduced_model_features(settings) == self.FROZEN

    def test_drift_fails_loudly_rather_than_being_absorbed(self, tmp_path):
        settings = _settings()
        settings["data"]["behaviour_selection_result_path"] = str(
            self._fake_selection(tmp_path, [*self.FROZEN, "self.speed"]))
        with pytest.raises(ValueError, match="behaviour selection drift"):
            reduced_model_features(settings)

    def test_a_missing_file_raises_unless_explicitly_tolerated(self, tmp_path):
        settings = _settings()
        settings["data"]["behaviour_selection_result_path"] = str(tmp_path / "absent.pkl")
        with pytest.raises(FileNotFoundError, match="behaviour selection result not found"):
            reduced_model_features(settings)
        assert reduced_model_features(settings, require_selection_file=False) == self.FROZEN

    def test_a_present_file_is_checked_even_when_the_file_is_optional(self, tmp_path):
        """`require_selection_file=False` tolerates ABSENCE, never disagreement."""
        settings = _settings()
        settings["data"]["behaviour_selection_result_path"] = str(
            self._fake_selection(tmp_path, ["something.else"]))
        with pytest.raises(ValueError, match="behaviour selection drift"):
            reduced_model_features(settings, require_selection_file=False)

    def test_the_settings_do_not_mutate(self, tmp_path):
        settings = _settings()
        original = copy.deepcopy(settings["nested_position_decoding"]["reduced_model_features"])
        settings["data"]["behaviour_selection_result_path"] = str(
            self._fake_selection(tmp_path, self.FROZEN))
        reduced_model_features(settings)
        assert settings["nested_position_decoding"]["reduced_model_features"] == original


class TestNestedDesign:
    """Rows are CALLS: each behavioural feature contributes `n_lags` columns of history ending at the
    call's onset, and the neuron contributes exactly one z-scored count."""

    N_LAGS = 4

    @staticmethod
    def _events(n_per_session=6, n_sessions=2, first_frame=10):
        starts = np.concatenate([np.arange(n_per_session) * 1.0 + first_frame / 10.0
                                 for _ in range(n_sessions)])
        return {"call_start": starts,
                "session_index": np.repeat(np.arange(n_sessions), n_per_session),
                "counts": np.tile(np.arange(n_per_session, dtype=float), n_sessions),
                "positions": np.tile(np.linspace(0.0, 0.9, n_per_session)[:, None], (n_sessions, 2)),
                "region_labels": np.tile(np.arange(n_per_session, dtype=float), n_sessions),
                "session_ids": [f"s{i}" for i in range(n_sessions)]}

    @staticmethod
    def _per_session(session_ids, feature_names, n_frames=400, fps=10.0, shuffle_from=None):
        out = {}
        for i, sid in enumerate(session_ids):
            names = list(feature_names)
            if shuffle_from is not None and i >= shuffle_from:
                names = names[::-1]                       # a DIFFERENT column order in this session
            series = np.zeros((n_frames, len(names)))
            for column, name in enumerate(names):
                series[:, column] = feature_names.index(name) * 1000 + np.arange(n_frames)
            out[sid] = {"feature_time_series": series, "feature_names": names,
                        "fps": fps, "n_frames": n_frames}
        return out

    def test_shapes_are_features_by_lags_plus_one_neural_column(self):
        names = ["a", "b", "c"]
        events = self._events()
        design = nested_design(events, self._per_session(events["session_ids"], names),
                               names, self.N_LAGS)
        assert design["behaviour"].shape == (12, 3 * self.N_LAGS)
        assert design["neural"].shape == (12, 1)

    def test_features_are_selected_by_name_not_position(self):
        """Per-session column ORDER differs, so a positional read takes a different feature in
        different sessions -- the cache bug that once made cross-session LOSO look broken."""
        names = ["a", "b", "c"]
        events = self._events()
        per_session = self._per_session(events["session_ids"], names, shuffle_from=1)
        assert per_session["s0"]["feature_names"] != per_session["s1"]["feature_names"]
        design = nested_design(events, per_session, names, self.N_LAGS)
        # feature f's series is f*1000 + frame, so the block for f must sit in f's thousands band
        for f in range(3):
            block = design["behaviour"][:, f * self.N_LAGS:(f + 1) * self.N_LAGS]
            assert np.all((block >= f * 1000) & (block < (f + 1) * 1000))

    def test_the_history_ends_at_the_onset_frame(self):
        """`lagged_design` lays each window oldest-to-newest, so the LAST column of a feature's block
        is that feature's value AT the onset frame."""
        names = ["a"]
        events = self._events(n_sessions=1)
        per_session = self._per_session(events["session_ids"], names)
        design = nested_design(events, per_session, names, self.N_LAGS)
        frames = np.rint(events["call_start"] * 10.0).astype(int)
        np.testing.assert_allclose(design["behaviour"][:, self.N_LAGS - 1], frames)

    def test_events_without_a_full_history_are_dropped_and_counted(self):
        """A call in a session's first 4 s is usable for claim 2 and not for claim 3. Both counts are
        recorded so the difference is reported rather than discovered."""
        names = ["a"]
        events = self._events(n_sessions=1, first_frame=0)
        events["call_start"] = np.array([0.0, 0.1, 0.2, 1.0, 2.0, 3.0])   # first few lack history
        per_session = self._per_session(events["session_ids"], names)
        design = nested_design(events, per_session, names, self.N_LAGS)
        book = design["per_session"]["s0"]
        assert book["n_events_claim2"] == 6
        assert book["n_events_claim3"] == design["behaviour"].shape[0]
        assert book["n_dropped_short_history"] == 6 - book["n_events_claim3"]
        assert book["n_dropped_short_history"] > 0

    def test_kept_indexes_back_into_the_claim_two_event_set(self):
        names = ["a"]
        events = self._events(n_sessions=1, first_frame=0)
        events["call_start"] = np.array([0.0, 0.1, 1.0, 2.0, 3.0, 4.0])
        design = nested_design(events, self._per_session(events["session_ids"], names),
                               names, self.N_LAGS)
        np.testing.assert_allclose(design["positions"], events["positions"][design["kept"]])
        np.testing.assert_allclose(design["region_labels"], events["region_labels"][design["kept"]])

    def test_the_neural_column_is_z_scored_within_each_session(self):
        """Per session, not pooled: the same correction claim 2's WHEN axis applies, because a
        least-squares fit on a continuous predictor CAN standardise its input."""
        names = ["a"]
        events = self._events(n_sessions=2)
        events["counts"] = np.concatenate([np.array([0.0, 0, 0, 10, 10, 10]),
                                           np.array([100.0, 100, 100, 200, 200, 200])])
        design = nested_design(events, self._per_session(events["session_ids"], names),
                               names, self.N_LAGS)
        for slot in (0, 1):
            column = design["neural"][design["session_index"] == slot, 0]
            assert abs(column.mean()) < 1e-9        # each session centred on its own mean
            assert abs(column.std() - 1.0) < 1e-9

    def test_a_near_silent_session_cannot_produce_an_enormous_z(self):
        names = ["a"]
        events = self._events(n_sessions=1)
        events["counts"] = np.zeros(6)
        events["counts"][0] = 1e-4                  # essentially no spread
        design = nested_design(events, self._per_session(events["session_ids"], names),
                               names, self.N_LAGS, sigma_floor=0.05)
        assert np.abs(design["neural"]).max() < 10.0

    def test_a_feature_the_assembler_never_built_fails_loudly(self):
        names = ["a", "b"]
        events = self._events()
        with pytest.raises(ValueError, match="not in the assembled feature set"):
            nested_design(events, self._per_session(events["session_ids"], names),
                          ["a", "does.not.exist"], self.N_LAGS)

    def test_an_empty_feature_set_is_refused(self):
        events = self._events()
        with pytest.raises(ValueError, match="at least one behavioural feature"):
            nested_design(events, self._per_session(events["session_ids"], ["a"]), [], self.N_LAGS)


class TestNestedScores:
    """The statistic: pooled added `vm_logscore`, full over reduced, across leave-one-session-out."""

    SETTINGS: ClassVar[dict] = {"lambda_smooth": 1.0, "l2_reg": 0.01,
                                "smoothness_derivative_order": 1, "min_region_events": 2}

    @staticmethod
    def _design(n_per_session=90, n_sessions=3, n_lags=3, n_features=2, seed=0,
                neuron="informative"):
        """A target driven by behaviour, plus a neuron that either adds independent position
        information, merely RELAYS the behaviour, or carries nothing at all."""
        rng = np.random.default_rng(seed)
        n = n_per_session * n_sessions
        behaviour = rng.normal(size=(n, n_features * n_lags))
        behaviour_drive = behaviour[:, n_lags - 1] + 0.5 * behaviour[:, 2 * n_lags - 1]
        extra = rng.normal(size=n)
        if neuron == "informative":
            column, angle = extra, behaviour_drive + 1.5 * extra
        elif neuron == "relay":
            column, angle = behaviour_drive.copy(), behaviour_drive
        elif neuron == "relay_noisy":
            column, angle = behaviour_drive + 1.5 * rng.normal(size=n), behaviour_drive
        else:
            column, angle = rng.normal(size=n), behaviour_drive
        angle = angle + 0.05 * rng.normal(size=n)
        positions = np.column_stack([(0.15 * angle) % 1.0, (0.07 * angle) % 1.0])
        session_index = np.repeat(np.arange(n_sessions), n_per_session)
        regions = np.tile(np.arange(6, dtype=float), n // 6 + 1)[:n]
        return {"behaviour": behaviour,
                "neural": ((column - column.mean()) / column.std())[:, None],
                "positions": positions, "session_index": session_index,
                "region_labels": regions,
                "feature_names": [f"f{i}" for i in range(n_features)],
                "kept": np.arange(n), "per_session": {}}

    def test_it_returns_the_pieces_the_conjunction_needs(self):
        scores = nested_scores(self._design(), self.SETTINGS, n_lags=3)
        for key in ("added", "reduced", "full", "per_fold", "min_leave_one_fold_out",
                    "n_events", "n_regions_scored"):
            assert key in scores
        assert np.isfinite(scores["added"])
        assert scores["added"] == pytest.approx(scores["full"] - scores["reduced"])

    def test_a_neuron_carrying_position_beyond_behaviour_adds(self):
        scores = nested_scores(self._design(neuron="informative", seed=1), self.SETTINGS, n_lags=3)
        assert scores["added"] > 0

    def test_a_relays_contribution_falls_as_it_relays_more_noisily(self):
        """MEASURED, and it qualifies the plan's claim that "a pure relay adds ~0".

        A relay carries NO information the behaviour lacks, so in the infinite-data limit it adds
        nothing. At finite n it can still add, by VARIANCE REDUCTION: the behaviour block has to
        estimate its mapping from many penalised columns -- on real data the design is underdetermined
        (cl0401: 2,155 events against 3,000 columns) -- while a clean relay hands the model a
        low-variance, already-correctly-weighted summary of the same signal.

        So what is actually true, and what this pins, is that the contribution is governed by how
        NOISILY the neuron relays. A noiseless relay adds as much as a genuinely informative neuron; a
        realistically noisy one adds little. Measured on this synthetic, mean over five seeds:
        noise 0.0 -> +0.90, 0.25 -> +0.67, 0.5 -> +0.38, 1.0 -> +0.12, 2.0 -> +0.02.

        The circular-shift null does NOT catch this, because a relay's alignment with behaviour is
        genuinely real; the null only breaks the neuron's alignment in time."""
        clean = nested_scores(self._design(neuron="relay", seed=2), self.SETTINGS, n_lags=3)
        noisy = nested_scores(self._design(neuron="relay_noisy", seed=2), self.SETTINGS, n_lags=3)
        assert noisy["added"] < 0.25 * clean["added"]

    def test_a_pure_noise_column_does_not_add(self):
        """The score does not simply reward having one more column -- which is the first thing to
        check before reading anything into a positive added score."""
        scores = nested_scores(self._design(neuron="noise", seed=7), self.SETTINGS, n_lags=3)
        assert scores["added"] < 0.05

    def test_a_neuron_carrying_nothing_does_not_add(self):
        scores = nested_scores(self._design(neuron="noise", seed=4), self.SETTINGS, n_lags=3)
        assert scores["added"] < 0.05

    def test_the_neural_column_can_be_overridden_which_is_what_the_null_needs(self):
        """The null refits with a SHIFTED neural column against the same behaviour and target."""
        design = self._design(neuron="informative", seed=5)
        observed = nested_scores(design, self.SETTINGS, n_lags=3)["added"]
        rng = np.random.default_rng(0)
        scrambled = nested_scores(design, self.SETTINGS, n_lags=3,
                                  neural=rng.permutation(design["neural"]))["added"]
        assert observed > scrambled

    def test_the_leave_one_fold_out_minimum_is_reported(self):
        scores = nested_scores(self._design(neuron="informative", seed=6), self.SETTINGS, n_lags=3)
        assert np.isfinite(scores["min_leave_one_fold_out"])
        assert len(scores["per_fold"]) == 3

    def test_a_single_session_is_refused_rather_than_silently_scored(self):
        design = self._design(n_sessions=1)
        with pytest.raises(ValueError, match="at least two sessions"):
            nested_scores(design, self.SETTINGS, n_lags=3)


class TestTheNull:
    """Circular shift of the SPIKE TRAIN -- behaviour and position keep their real relationship, and
    only the neuron is decoupled in time."""

    SETTINGS: ClassVar[dict] = {"lambda_smooth": 1.0, "l2_reg": 0.01,
                                "smoothness_derivative_order": 1, "min_region_events": 2}

    def test_reusing_the_reduced_pass_is_exact_not_approximate(self):
        """The shift moves only the spike train, so the reduced fit CANNOT change between draws --
        recomputing it per draw would double the null's cost for a guaranteed-identical answer. If
        this ever stops being bit-exact, something has started leaking the neuron into the control."""
        design = TestNestedScores._design(neuron="informative", seed=8)
        fresh = nested_scores(design, self.SETTINGS, n_lags=3)
        reused = nested_scores(design, self.SETTINGS, n_lags=3,
                               reduced_predicted=reduced_predictions(design, self.SETTINGS, 3))
        assert reused["added"] == fresh["added"]
        assert reused["reduced"] == fresh["reduced"]

    @staticmethod
    def _train(n_sessions=2, rate=40.0, duration=600.0, seed=0):
        rng = np.random.default_rng(seed)
        return {s: np.sort(rng.uniform(0, duration, int(rate * duration / 10)))
                for s in range(n_sessions)}, dict.fromkeys(range(n_sessions), duration)

    def test_a_shifted_column_is_z_scored_per_session_like_the_observed_one(self):
        design = TestNestedScores._design(n_sessions=2, n_per_session=60, seed=9)
        spikes, durations = self._train()
        edges = np.linspace(30.0, 550.0, design["positions"].shape[0])
        column = shifted_neural_column(design, edges, spikes, durations, 0.05,
                                       np.random.default_rng(0), 20.0)
        assert column.shape == (design["positions"].shape[0], 1)
        for slot in (0, 1):
            block = column[design["session_index"] == slot, 0]
            assert abs(block.mean()) < 1e-9

    def test_different_draws_give_different_columns(self):
        """A null whose draws repeat is not a null."""
        design = TestNestedScores._design(n_sessions=2, n_per_session=60, seed=10)
        spikes, durations = self._train()
        edges = np.linspace(30.0, 550.0, design["positions"].shape[0])
        first = shifted_neural_column(design, edges, spikes, durations, 0.05,
                                      np.random.default_rng(1), 20.0)
        second = shifted_neural_column(design, edges, spikes, durations, 0.05,
                                       np.random.default_rng(2), 20.0)
        assert not np.allclose(first, second)

    def test_the_same_seed_reproduces_a_draw(self):
        design = TestNestedScores._design(n_sessions=2, n_per_session=60, seed=11)
        spikes, durations = self._train()
        edges = np.linspace(30.0, 550.0, design["positions"].shape[0])
        args = (design, edges, spikes, durations, 0.05)
        np.testing.assert_allclose(
            shifted_neural_column(*args, np.random.default_rng(7), 20.0),
            shifted_neural_column(*args, np.random.default_rng(7), 20.0))

    def test_counts_are_RECOUNTED_from_the_shifted_train_not_reordered(self):
        """A reordering would preserve the exact multiset of counts, testing something narrower than
        'this neuron, decoupled in time'. Recounting can change the multiset."""
        design = TestNestedScores._design(n_sessions=2, n_per_session=60, seed=12)
        spikes, durations = self._train(rate=8.0)
        edges = np.linspace(30.0, 550.0, design["positions"].shape[0])
        draws = [np.sort(shifted_neural_column(design, edges, spikes, durations, 0.5,
                                               np.random.default_rng(s), 20.0).ravel())
                 for s in range(6)]
        assert any(not np.allclose(draws[0], other) for other in draws[1:])
