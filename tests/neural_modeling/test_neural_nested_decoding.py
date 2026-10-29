"""
@author: bartulem
Unit tests for ``usv_playpen.neural_modeling.neural_nested_decoding``.

Coverage: the nested vocal-manifold position decoding's block-diagonal penalty and the behaviour control it is
fitted against; the nested design, the leave-one-session-out scores, the spike-train shift null and its
escalation, the settings guards, the no-behaviour baseline and the posterior (behaviour-as-prior) statistic.

The penalty is the piece that makes a mixed design legal -- behavioural features wanting 600 lags
of smoothing beside one neural column that must receive ridge only -- so these tests pin its SHAPE
(zeros off the behaviour blocks, zero on the neural diagonal) and its IDENTITY to the parent's
operator, since a second copy of the finite-difference operator could drift from the one the
behavioural model uses. They also pin the reduced model to the cohort behaviour-control selection,
checked against the frozen feature list.
"""

from __future__ import annotations

import copy
import inspect
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
    from usv_playpen.neural_modeling.neural_design_assembly import (
        onset_history_anchor_frames,
    )
    from usv_playpen.neural_modeling.neural_nested_decoding import (
        NestedTorusRegression,
        behaviour_prior_log_density,
        behaviour_priors,
        check_settings,
        nested_design,
        nested_scores,
        nested_settings,
        no_behaviour_predictions,
        pooled_macro_score,
        posterior_nested_gain,
        reduced_model_features,
        reduced_predictions,
    )
    from usv_playpen.neural_modeling.neural_significance import (
        within_session_permutation,
    )
    from usv_playpen.neural_modeling.neural_vocal_decoding import (
        decode_gain,
        decoding_context,
    )

SETTINGS_PATH = (pathlib.Path(__file__).resolve().parents[2] / "src" / "usv_playpen"
                 / "_parameter_settings" / "neural_modeling_settings.json")
with SETTINGS_PATH.open() as _settings_file:
    # The shipped torus period every analysis of the manifold reads.
    PERIOD = json.load(_settings_file)["vocalization_settings"]["torus_period"]


def _settings() -> dict:
    """
    Description
    -----------
    Read the shipped neural-modeling settings fresh from disk.

    Parameters
    ----------
    None

    Returns
    -------
    settings (dict)
        The parsed ``neural_modeling_settings.json``; a new dict on every read, so tests may mutate it.
    """
    with SETTINGS_PATH.open() as handle:
        return json.load(handle)


def _nested(**overrides) -> dict:
    """
    Description
    -----------
    Return the SHIPPED nested_vocal_decoding block, merged and validated exactly as a
    run does, with only tiny-synthetic knobs overridden. Built from the real file rather than a literal
    so that a key the code starts reading -- or stops honouring -- shows up here instead of only at run
    time.

    Parameters
    ----------
    **overrides (dict)
        Keys of the merged block to replace (e.g. ``min_region_events``).

    Returns
    -------
    block (dict)
        The merged, validated nested-decoding block with the overrides applied.
    """
    block = nested_settings(_settings())
    block.update(overrides)
    return block


class TestBlockDiagonalPenalty:
    """One temporal block per behavioural feature, and NOTHING on the neural column."""

    def test_the_neural_column_is_unsmoothed(self):
        """The neural column's diagonal entry and every cross term with it are zero, so the neuron gets
        ridge only and is never smoothed against a behavioural lag."""
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
        """With no neural column the penalty is exactly features x lags square, so the reduced model is
        the behaviour-only estimator."""
        est = NestedTorusRegression(n_behaviour_features=5, n_lags=20, n_neural_columns=0)
        assert est._smoothness_penalty().shape == (100, 100)

    def test_the_shipped_shape_is_five_features_by_six_hundred_lags_plus_one(self):
        """At the shipped design (5 features, 600 lags, one neural column) the penalty is 3001 x 3001."""
        est = NestedTorusRegression(n_behaviour_features=5, n_lags=600, n_neural_columns=1)
        assert est._smoothness_penalty().shape == (3001, 3001)


class TestTheEstimatorFits:
    """The parent hard-checks column count against `n_features * n_time_bins`, so a mixed design only
    fits at all if this class reports its shape the way it does."""

    @staticmethod
    def _design(n=240, n_features=3, n_lags=8, seed=0, neural_signal=0.0):
        """
        Description
        -----------
        Build a random behaviour block, one neural column, and a 2-D torus target driven by two
        behaviour columns plus, optionally, the neural column.

        Parameters
        ----------
        n (int)
            Number of rows; defaults to 240.
        n_features (int)
            Number of behavioural features; defaults to 3.
        n_lags (int)
            Lags per feature; defaults to 8.
        seed (int)
            Seed for ``np.random.default_rng``; defaults to 0.
        neural_signal (float)
            Weight of the neural column in the target angle; defaults to 0.0.

        Returns
        -------
        design (tuple[np.ndarray, np.ndarray, np.ndarray])
            ``(behaviour, neural, y)`` with ``y`` wrapped to [0, 1) on both axes.
        """
        rng = np.random.default_rng(seed)
        behaviour = rng.normal(size=(n, n_features * n_lags))
        neural = rng.normal(size=(n, 1))
        angle = (0.6 * behaviour[:, 0] + 0.4 * behaviour[:, n_lags]
                 + neural_signal * neural[:, 0] + 0.05 * rng.normal(size=n))
        y = np.column_stack([angle % 1.0, (0.5 * angle) % 1.0])
        return behaviour, neural, y

    def test_a_mixed_design_fits(self):
        """A behaviour-plus-neural design passes the parent's column check, converges, and returns one
        coefficient row per column for the 4-D torus embedding."""
        behaviour, neural, y = self._design()
        est = NestedTorusRegression(n_behaviour_features=3, n_lags=8, n_neural_columns=1)
        est.fit(np.hstack([behaviour, neural]), y)
        assert est.coef_.shape == (25, 4)          # 4-D torus embedding
        assert est.converged_

    def test_a_wrong_column_count_is_still_refused(self):
        """Omitting the declared neural column must still raise, so the shape override does not disable
        the parent's guard."""
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
    """Read the COHORT selection AND assert it against the frozen list -- reading alone lets a re-run
    silently redefine 'beyond behaviour'; freezing alone lets the control drift from the selection.

    A behavioural selection's FEATURE SET may be adopted; its COEFFICIENTS may not. The fitted map is
    what fails to transfer to the ephys preparation (-0.00113 over 41,087 USVs), while the same
    estimator refitted in domain predicts perfectly well -- so the names are read from either artifact
    shape and the control is always refitted in domain. An externally chosen set also carries NO
    selection optimism, since those features were never chosen on this preparation."""

    FROZEN: ClassVar[list[str]] = ["head-head"]

    @staticmethod
    def _fake_selection(tmp_path, features):
        """
        Description
        -----------
        Pickle a cohort-selection result carrying ``selected_features``.

        Parameters
        ----------
        tmp_path (pathlib.Path)
            Directory the pickle is written to.
        features (list[str])
            The selected feature names.

        Returns
        -------
        path (pathlib.Path)
            Path to ``selection.pkl``.
        """
        path = tmp_path / "selection.pkl"
        with path.open("wb") as handle:
            pickle.dump({"selected_features": list(features)}, handle)
        return path

    @staticmethod
    def _behavioural_selection(tmp_path, features):
        """
        Description
        -----------
        Pickle a behavioural-modelling selection, whose last forward step carries the chosen set.

        Parameters
        ----------
        tmp_path (pathlib.Path)
            Directory the pickle is written to.
        features (list[str])
            The final step's feature names; the first step holds only the first of them.

        Returns
        -------
        path (pathlib.Path)
            Path to ``behavioural_selection.pkl``.
        """
        path = tmp_path / "behavioural_selection.pkl"
        with path.open("wb") as handle:
            pickle.dump({"steps": [{"current_features": list(features[:1])},
                                   {"current_features": list(features)}]}, handle)
        return path

    def test_a_behavioural_selection_feature_set_is_adopted(self, tmp_path):
        """The last forward step's `current_features` defines the set; coefficients are never read."""
        settings = _settings()
        settings["data"]["behaviour_selection_result_path"]["manifold"] = str(
            self._behavioural_selection(tmp_path, self.FROZEN))
        assert reduced_model_features(settings, "manifold") == self.FROZEN

    def test_a_behavioural_selection_is_drift_checked_too(self, tmp_path):
        """Accepting the shape must not exempt it from the freeze."""
        settings = _settings()
        settings["data"]["behaviour_selection_result_path"]["manifold"] = str(
            self._behavioural_selection(tmp_path, [*self.FROZEN, "self.speed"]))
        with pytest.raises(ValueError, match="behaviour selection drift"):
            reduced_model_features(settings, "manifold")

    def test_an_unrecognised_artifact_is_refused(self, tmp_path):
        """Neither shape present would otherwise raise an unhelpful KeyError inside the reader."""
        path = tmp_path / "mystery.pkl"
        with path.open("wb") as handle:
            pickle.dump({"something_else": ["a"]}, handle)
        settings = _settings()
        settings["data"]["behaviour_selection_result_path"]["manifold"] = str(path)
        with pytest.raises(ValueError, match="neither a cohort selection result"):
            reduced_model_features(settings, "manifold")

    def test_the_shipped_settings_freeze_the_selected_features(self):
        """The shipped ``reduced_model_features`` equal the frozen list, so the control cannot change
        without this test changing too."""
        assert _settings()["nested_vocal_decoding"]["reduced_model_features"]["manifold"] == self.FROZEN

    def test_a_matching_file_returns_the_selection_order(self, tmp_path):
        """A selection file agreeing with the freeze returns the features in selection order."""
        settings = _settings()
        settings["data"]["behaviour_selection_result_path"]["manifold"] = str(
            self._fake_selection(tmp_path, self.FROZEN))
        assert reduced_model_features(settings, "manifold") == self.FROZEN

    def test_drift_fails_loudly_rather_than_being_absorbed(self, tmp_path):
        """A selection file disagreeing with the freeze must raise rather than silently redefine the
        control."""
        settings = _settings()
        settings["data"]["behaviour_selection_result_path"]["manifold"] = str(
            self._fake_selection(tmp_path, [*self.FROZEN, "self.speed"]))
        with pytest.raises(ValueError, match="behaviour selection drift"):
            reduced_model_features(settings, "manifold")

    def test_a_missing_file_raises_unless_explicitly_tolerated(self, tmp_path):
        """An absent selection file raises by default and falls back to the frozen list only when
        ``require_selection_file=False``."""
        settings = _settings()
        settings["data"]["behaviour_selection_result_path"]["manifold"] = str(tmp_path / "absent.pkl")
        with pytest.raises(FileNotFoundError, match="behaviour selection result not found"):
            reduced_model_features(settings, "manifold")
        assert reduced_model_features(settings, "manifold", require_selection_file=False) == self.FROZEN

    def test_a_present_file_is_checked_even_when_the_file_is_optional(self, tmp_path):
        """`require_selection_file=False` tolerates ABSENCE, never disagreement."""
        settings = _settings()
        settings["data"]["behaviour_selection_result_path"]["manifold"] = str(
            self._fake_selection(tmp_path, ["something.else"]))
        with pytest.raises(ValueError, match="behaviour selection drift"):
            reduced_model_features(settings, "manifold", require_selection_file=False)

    def test_the_settings_do_not_mutate(self, tmp_path):
        """Reading the control must not modify the frozen list in the caller's settings."""
        settings = _settings()
        original = copy.deepcopy(settings["nested_vocal_decoding"]["reduced_model_features"])
        settings["data"]["behaviour_selection_result_path"]["manifold"] = str(
            self._fake_selection(tmp_path, self.FROZEN))
        reduced_model_features(settings, "manifold")
        assert settings["nested_vocal_decoding"]["reduced_model_features"] == original


class TestNestedDesign:
    """Rows are USVs: each behavioural feature contributes `n_lags` columns of history ending at the
    USV's onset, and the neuron contributes exactly one z-scored count."""

    N_LAGS = 4

    @staticmethod
    def _events(n_per_session=6, n_sessions=2, first_frame=10):
        """
        Description
        -----------
        Build a vocal-decoding event set with one-second-spaced onsets per session.

        Parameters
        ----------
        n_per_session (int)
            Events per session; defaults to 6.
        n_sessions (int)
            Number of sessions; defaults to 2.
        first_frame (int)
            First onset in frames at 10 fps; defaults to 10.

        Returns
        -------
        events (dict)
            ``vocalization_start``, ``session_index``, ``counts``, ``positions``, ``region_labels`` and
            ``session_ids``.
        """
        starts = np.concatenate([np.arange(n_per_session) * 1.0 + first_frame / 10.0
                                 for _ in range(n_sessions)])
        return {"vocalization_start": starts,
                "session_index": np.repeat(np.arange(n_sessions), n_per_session),
                "counts": np.tile(np.arange(n_per_session, dtype=float), n_sessions),
                "positions": np.tile(np.linspace(0.0, 0.9, n_per_session)[:, None], (n_sessions, 2)),
                "region_labels": np.tile(np.arange(n_per_session, dtype=float), n_sessions),
                "session_ids": [f"s{i}" for i in range(n_sessions)]}

    @staticmethod
    def _per_session(session_ids, feature_names, n_frames=400, fps=10.0, shuffle_from=None):
        """
        Description
        -----------
        Build per-session feature records where feature ``f``'s series is ``f * 1000 + frame``, so a
        design column reveals which feature and frame it was read from.

        Parameters
        ----------
        session_ids (list[str])
            Session ids to build.
        feature_names (list[str])
            Canonical feature order.
        n_frames (int)
            Frames per session; defaults to 400.
        fps (float)
            Frame rate; defaults to 10.0.
        shuffle_from (int | None)
            Sessions from this index on store their columns in REVERSED order; ``None`` keeps every
            session canonical.

        Returns
        -------
        per_session (dict)
            ``{session_id: {"feature_time_series", "feature_names", "fps", "n_frames"}}``.
        """
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
        """The behaviour block is events x (features * lags) and the neural block one column."""
        names = ["a", "b", "c"]
        events = self._events()
        design = nested_design(events, self._per_session(events["session_ids"], names),
                               names, self.N_LAGS, 0.05)
        assert design["behaviour"].shape == (12, 3 * self.N_LAGS)
        assert design["neural"].shape == (12, 1)

    def test_features_are_selected_by_name_not_position(self):
        """Per-session column ORDER differs, so a positional read takes a different feature in
        different sessions -- the cache bug that once made cross-session LOSO look broken."""
        names = ["a", "b", "c"]
        events = self._events()
        per_session = self._per_session(events["session_ids"], names, shuffle_from=1)
        assert per_session["s0"]["feature_names"] != per_session["s1"]["feature_names"]
        design = nested_design(events, per_session, names, self.N_LAGS, 0.05)
        # feature f's series is f*1000 + frame, so the block for f must sit in f's thousands band
        for f in range(3):
            block = design["behaviour"][:, f * self.N_LAGS:(f + 1) * self.N_LAGS]
            assert np.all((block >= f * 1000) & (block < (f + 1) * 1000))

    def test_the_history_ends_at_the_frame_before_onset(self):
        """`lagged_design` lays each window oldest-to-newest, so the LAST column of a feature's block
        is that feature's value at the window's end -- the frame BEFORE the onset frame, the convention
        the cohort behaviour control is fitted at. It used to end AT the onset frame, one frame later
        than every history the control's coefficients were estimated on."""
        names = ["a"]
        events = self._events(n_sessions=1)
        per_session = self._per_session(events["session_ids"], names)
        design = nested_design(events, per_session, names, self.N_LAGS, 0.05)
        frames = np.rint(events["vocalization_start"] * 10.0).astype(int) - 1
        np.testing.assert_allclose(design["behaviour"][:, self.N_LAGS - 1], frames)
        np.testing.assert_array_equal(frames, onset_history_anchor_frames(events["vocalization_start"], 10.0))

    def test_events_without_a_full_history_are_dropped_and_counted(self):
        """A USV in a session's first 4 s is usable for the vocal decoding and not for the nested
        decoding. Both counts are
        recorded so the difference is reported rather than discovered."""
        names = ["a"]
        events = self._events(n_sessions=1, first_frame=0)
        events["vocalization_start"] = np.array([0.0, 0.1, 0.2, 1.0, 2.0, 3.0])   # first few lack history
        per_session = self._per_session(events["session_ids"], names)
        design = nested_design(events, per_session, names, self.N_LAGS, 0.05)
        book = design["per_session"]["s0"]
        assert book["n_events_vocal_decoding"] == 6
        assert book["n_events_with_full_history"] == design["behaviour"].shape[0]
        assert book["n_dropped_short_history"] == 6 - book["n_events_with_full_history"]
        assert book["n_dropped_short_history"] > 0

    def test_kept_indexes_back_into_the_vocal_decoding_event_set(self):
        """``kept`` maps design rows back to the vocal-decoding events, so positions and region labels
        stay paired with the right rows after short-history drops."""
        names = ["a"]
        events = self._events(n_sessions=1, first_frame=0)
        events["vocalization_start"] = np.array([0.0, 0.1, 1.0, 2.0, 3.0, 4.0])
        design = nested_design(events, self._per_session(events["session_ids"], names),
                               names, self.N_LAGS, 0.05)
        np.testing.assert_allclose(design["positions"], events["positions"][design["kept"]])
        np.testing.assert_allclose(design["region_labels"], events["region_labels"][design["kept"]])

    def test_the_neural_column_is_z_scored_within_each_session(self):
        """Per session, not pooled: the same correction the vocal-occurrence axis applies, because a
        least-squares fit on a continuous predictor CAN standardise its input."""
        names = ["a"]
        events = self._events(n_sessions=2)
        events["counts"] = np.concatenate([np.array([0.0, 0, 0, 10, 10, 10]),
                                           np.array([100.0, 100, 100, 200, 200, 200])])
        design = nested_design(events, self._per_session(events["session_ids"], names),
                               names, self.N_LAGS, 0.05)
        for slot in (0, 1):
            column = design["neural"][design["session_index"] == slot, 0]
            assert abs(column.mean()) < 1e-9        # each session centred on its own mean
            assert abs(column.std() - 1.0) < 1e-9

    def test_a_near_silent_session_cannot_produce_an_enormous_z(self):
        """The sigma floor keeps a session with almost no count spread from producing a huge z-score
        that would dominate the fit."""
        names = ["a"]
        events = self._events(n_sessions=1)
        events["counts"] = np.zeros(6)
        events["counts"][0] = 1e-4                  # essentially no spread
        design = nested_design(events, self._per_session(events["session_ids"], names),
                               names, self.N_LAGS, 0.05)
        assert np.abs(design["neural"]).max() < 10.0

    def test_a_feature_the_assembler_never_built_fails_loudly(self):
        """Requesting a feature absent from the assembled set must raise, never yield a zero block."""
        names = ["a", "b"]
        events = self._events()
        with pytest.raises(ValueError, match="not in the assembled feature set"):
            nested_design(events, self._per_session(events["session_ids"], names),
                          ["a", "does.not.exist"], self.N_LAGS, 0.05)

    def test_an_empty_feature_set_is_refused(self):
        """A nested design with no behaviour is meaningless and must raise."""
        events = self._events()
        with pytest.raises(ValueError, match="at least one behavioural feature"):
            nested_design(events, self._per_session(events["session_ids"], ["a"]), [], self.N_LAGS, 0.05)


class TestNestedScores:
    """The statistic: pooled added `vm_logscore`, full over reduced, across leave-one-session-out."""

    SETTINGS: ClassVar[dict] = _nested(min_region_events=2)

    @staticmethod
    def _design(n_per_session=90, n_sessions=3, n_lags=3, n_features=2, seed=0,
                neuron="informative"):
        """
        Description
        -----------
        Build a nested design whose target is driven by behaviour, plus a neuron that either adds
        independent position information, merely RELAYS the behaviour, or carries nothing at all.

        Parameters
        ----------
        n_per_session (int)
            Events per session; defaults to 90.
        n_sessions (int)
            Number of sessions; defaults to 3.
        n_lags (int)
            Lags per feature; defaults to 3.
        n_features (int)
            Number of behavioural features; defaults to 2.
        seed (int)
            Seed for ``np.random.default_rng``; defaults to 0.
        neuron (str)
            ``"informative"``, ``"relay"``, ``"relay_noisy"`` or anything else for pure noise.

        Returns
        -------
        design (dict)
            ``behaviour``, z-scored ``neural``, ``positions``, ``session_index``, ``region_labels``,
            ``feature_names``, ``kept`` and ``per_session``.
        """
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
        """The result carries every key the verdict reads, and ``added`` is exactly full minus reduced."""
        scores = nested_scores(self._design(), self.SETTINGS, n_lags=3)
        for key in ("added", "reduced", "full", "per_fold", "min_leave_one_fold_out",
                    "n_events", "n_regions_scored"):
            assert key in scores
        assert np.isfinite(scores["added"])
        assert scores["added"] == pytest.approx(scores["full"] - scores["reduced"])

    def test_a_neuron_carrying_position_beyond_behaviour_adds(self):
        """A neuron with independent position information yields a positive added score."""
        scores = nested_scores(self._design(neuron="informative", seed=1), self.SETTINGS, n_lags=3)
        assert scores["added"] > 0

    def test_a_relays_contribution_falls_as_it_relays_more_noisily(self):
        """A neuron that only relays behaviour can still add at finite n; how much depends on its noise.

        A relay carries NO information the behaviour lacks, so in the infinite-data limit it adds
        nothing. At finite n it can still add, by VARIANCE REDUCTION: the behaviour block has to
        estimate its mapping from many penalised columns -- on real data the design can have more
        columns than events -- while a clean relay hands the model a
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
        """A second noise seed: a neuron with no information stays below the small added-score bound."""
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
        """The leave-one-fold-out minimum is finite and there is one fold per session."""
        scores = nested_scores(self._design(neuron="informative", seed=6), self.SETTINGS, n_lags=3)
        assert np.isfinite(scores["min_leave_one_fold_out"])
        assert len(scores["per_fold"]) == 3

    def test_a_single_session_is_refused_rather_than_silently_scored(self):
        """Leave-one-session-out needs two sessions; one must raise rather than score in-sample."""
        design = self._design(n_sessions=1)
        with pytest.raises(ValueError, match="at least two sessions"):
            nested_scores(design, self.SETTINGS, n_lags=3)


class TestTheNull:
    """The null moves only the neural side, so the behaviour-only reduced fit is shared by every draw."""

    SETTINGS: ClassVar[dict] = _nested(min_region_events=2)

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


class TestSettingsAreNotHardCoded:
    """Every option key named `nested_vocal_decoding` described a real alternative and was read by
    NOTHING, so a run could declare one and silently get another."""

    def test_the_shipped_block_is_accepted(self):
        """The shipped block validates and carries the shared torus period of 1.0."""
        assert nested_settings(_settings())["torus_period"] == 1.0

    @pytest.mark.parametrize(("key", "bad"), [("vm_score_mode", "micro"),
                                              ("behaviour_control", "matched"),
                                              ("prevocal_window_n_bins", 3),
                                              ("rate_transform", "sqrt"),
                                              ("rate_basis", "bins")])
    def test_an_unimplemented_option_is_refused(self, key, bad):
        """Each named alternative that is not implemented must raise rather than run the default."""
        settings = _settings()
        settings["nested_vocal_decoding"][key] = bad
        with pytest.raises(ValueError, match="is not implemented"):
            nested_settings(settings)

    def test_the_torus_period_lives_in_the_shared_block_not_the_nested_decodings(self):
        """It is a property of the MANIFOLD, so every analysis sees one value; duplicating it into two
        blocks is exactly the drift this audit was about."""
        settings = _settings()
        assert "torus_period" in settings["vocalization_settings"]
        assert "torus_period" not in settings["nested_vocal_decoding"]

    def test_the_vocal_decoding_reads_the_period_it_is_given(self):
        """The vocal decoding takes the period as an argument (it used to keep a module constant), so
        a non-unit period reaches the grid, the stored folds and the decoded positions."""
        rng = np.random.default_rng(0)
        positions = rng.random((60, 2)) * 2.0
        sessions = np.repeat(np.arange(3), 20)
        decoding = _decoding()
        folds = decoding_context(positions, sessions, decoding, 2.0)
        grid_n = decoding["tuning_surface"]["grid_n"]
        assert all(fold["period"] == 2.0 for fold in folds)
        assert folds[0]["cell_area"] == pytest.approx((2.0 / grid_n) ** 2)
        assert folds[0]["grid_positions"].max() < 2.0
        assert folds[0]["grid_positions"].max() > 1.0

    def test_both_decodings_receive_the_shared_period_from_settings(self):
        """Every decoding_context call in the dispatcher hands it `vocalization_settings.torus_period`,
        the same key the nested decoding's block carries, so the two cannot disagree on the period."""
        source = (pathlib.Path(__file__).resolve().parents[2] / "src" / "usv_playpen" / "neural_modeling"
                  / "main_neural_vocal_dispatcher.py").read_text()
        n_contexts = source.count("decoding_context(")
        assert n_contexts >= 2
        assert source.count('settings["vocalization_settings"]["torus_period"]') >= n_contexts

    def test_the_sigma_floor_is_a_setting_not_a_default(self):
        """The vocal-occurrence axis takes it as a required parameter from settings; the nested decoding hard-coded the
        same constant as a default argument, which is how the two could have drifted apart."""
        parameter = inspect.signature(nested_design).parameters["sigma_floor"]
        assert parameter.default is inspect.Parameter.empty
        assert _settings()["nested_vocal_decoding"]["sigma_floor"] == 0.05

    def test_the_dead_bins_knob_is_gone(self):
        """`rate_basis_n_bins` only configured `rate_basis: "bins"`, which is now refused."""
        assert "rate_basis_n_bins" not in _settings()["nested_vocal_decoding"]



class TestNoBehaviourBaseline:
    """What is claimed is that the neuron beats the BEHAVIOUR MODEL, which is only worth claiming when that
    model predicts. The baseline it is judged against is the same estimator, folds and score given no behaviour."""

    SETTINGS: ClassVar[dict] = _nested(min_region_events=2)

    def test_it_can_only_learn_a_constant_per_held_out_session(self):
        """With no behaviour the estimator can only predict one position per held-out session."""
        design = TestNestedScores._design(seed=40)
        predicted = no_behaviour_predictions(design, self.SETTINGS, n_lags=3)
        for slot in np.unique(design["session_index"]):
            rows = predicted[design["session_index"] == slot]
            assert np.allclose(rows, rows[0])

    def test_behaviour_that_drives_position_beats_it(self):
        """When behaviour drives position the reduced model outscores the no-behaviour baseline, so the
        baseline is a real floor."""
        design = TestNestedScores._design(neuron="informative", seed=41)
        baseline = pooled_macro_score(no_behaviour_predictions(design, self.SETTINGS, n_lags=3),
                                      design["positions"], design["region_labels"], self.SETTINGS)
        assert nested_scores(design, self.SETTINGS, n_lags=3)["reduced"] > baseline


def _decoding(**surface_overrides) -> dict:
    """
    Description
    -----------
    Return the SHIPPED vocal_decoding block with only the tiny-run knobs overridden, as the identity
    tests build it: coarser grid, wider bandwidth, nothing structural touched.

    Parameters
    ----------
    **surface_overrides (dict)
        Extra ``tuning_surface`` keys to replace.

    Returns
    -------
    block (dict)
        A deep copy of ``vocal_decoding`` with ``grid_n=24`` and ``kde_bandwidth=0.10``.
    """
    block = copy.deepcopy(_settings()["vocal_decoding"])
    block["tuning_surface"].update({"grid_n": 24, "kde_bandwidth": 0.10, **surface_overrides})
    return block


class TestPosteriorNested:
    """The TESTED nested statistic: identity's decoder with behaviour entering as the prior.

    Synthetic events on the torus. The behaviour control predicts each position with wrapped noise --
    a weak or a strong control -- and the neuron either separates REGIONS with a count ladder (the
    case the point regression is structurally blind to), RELAYS what the control predicts, or
    carries nothing. The gate is the one a nested test must pass: region tuning adds beyond a weak
    control, a relay adds nothing beyond a strong one, and noise adds nothing beyond anything.
    """

    NESTED: ClassVar[dict] = nested_settings(_settings())
    DECODING: ClassVar[dict] = _decoding()

    @staticmethod
    def _unit(n_sessions=3, per_session=240, seed=0, neuron="region", control_noise=0.25):
        """
        Description
        -----------
        Build synthetic torus events with a behaviour-control prediction and Poisson counts from a
        region-tuned, relaying or null neuron.

        Parameters
        ----------
        n_sessions (int)
            Number of sessions; defaults to 3.
        per_session (int)
            Events per session; defaults to 240.
        seed (int)
            Seed for ``np.random.default_rng``; defaults to 0.
        neuron (str)
            ``"region"`` (count ladder over x-quadrants), ``"relay"`` (a function of the control's
            prediction) or anything else for a constant rate.
        control_noise (float)
            Gaussian noise added to the true position to form the control's prediction.

        Returns
        -------
        unit (tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray])
            ``(counts, positions, predicted, sessions)``.
        """
        rng = np.random.default_rng(seed)
        n = n_sessions * per_session
        positions = rng.random((n, 2))
        sessions = np.repeat(np.arange(n_sessions), per_session)
        predicted = (positions + control_noise * rng.normal(size=(n, 2))) % 1.0
        quadrant = np.floor(positions[:, 0] * 4.0)
        if neuron == "region":
            log_rate = np.log(2.0) + 0.45 * quadrant            # a 2 -> 7.7 ladder across q1
        elif neuron == "relay":
            log_rate = np.log(3.0) + np.cos(2.0 * np.pi * predicted[:, 0])   # a function of the CONTROL
        else:
            log_rate = np.full(n, np.log(3.0))
        counts = rng.poisson(np.exp(log_rate)).astype(float)
        return counts, positions, predicted, sessions

    def _run(self, neuron, control_noise, seed=0, permutation=None):
        """
        Description
        -----------
        Build a ``_unit``, fit the behaviour priors on its folds and compute the posterior nested gain.

        Parameters
        ----------
        neuron (str)
            Neuron kind passed to ``_unit``.
        control_noise (float)
            Control noise passed to ``_unit``.
        seed (int)
            Seed passed to ``_unit``; defaults to 0.
        permutation (np.ndarray | None)
            Optional within-session permutation for a null draw.

        Returns
        -------
        run (tuple)
            ``(result, kappas, (counts, positions, predicted, sessions, folds))``.
        """
        counts, positions, predicted, sessions = self._unit(seed=seed, neuron=neuron,
                                                            control_noise=control_noise)
        folds = decoding_context(positions, sessions, self.DECODING, PERIOD)
        kappas, priors = behaviour_priors(folds, predicted, positions, self.NESTED)
        result = posterior_nested_gain(counts, folds, self.DECODING, priors, kappas,
                                       permutation=permutation)
        return result, kappas, (counts, positions, predicted, sessions, folds)

    def test_a_region_tuned_neuron_adds_beyond_a_weak_control(self):
        """A region-tuned neuron adds beyond a weak control, the control still beats no behaviour, and
        the ladder's two steps sum to the full gain."""
        result, _kappas, _ = self._run("region", control_noise=0.25)
        assert result["added"] > 0.02
        assert result["behaviour_over_no_behaviour"] > 0.0     # the control is still worth something
        assert result["full_over_no_behaviour"] == pytest.approx(
            result["added"] + result["behaviour_over_no_behaviour"])

    def test_a_relay_adds_nothing_beyond_a_strong_control(self):
        """What the neuron knows is what behaviour predicts; conditioning on behaviour must remove it.
        The same neuron is informative UNCONDITIONED, which is what shows the prior did the work."""
        result, _kappas, (counts, _p, _pr, _s, folds) = self._run("relay", control_noise=0.05)
        unconditioned = decode_gain(counts, folds, self.DECODING)["gain"]
        assert unconditioned > 0.02
        assert abs(result["added"]) < 0.015

    def test_a_null_neuron_adds_nothing(self):
        """A constant-rate neuron's added gain stays within a small band around zero."""
        result, _kappas, _ = self._run("none", control_noise=0.25)
        assert abs(result["added"]) < 0.012

    def test_the_prior_is_a_density_over_the_masked_grid(self):
        """Each prior row integrates to 1 over the grid, places no mass outside the mask, and comes with
        a positive kappa."""
        _result, _kappas, (_c, positions, predicted, _s, folds) = self._run("none", 0.25)
        fold = folds[0]
        kappa, log_prior = behaviour_prior_log_density(fold, predicted, positions, PERIOD, self.NESTED)
        mass = np.exp(log_prior).sum(axis=1) * fold["cell_area"]
        assert np.allclose(mass, 1.0, rtol=1e-6)
        assert np.all(np.exp(log_prior)[:, ~fold["mask"]] <= 1e-299)
        assert kappa > 0.0

    def test_kappa_sharpens_with_the_control(self):
        """A more accurate control yields larger concentrations than a noisier one on every fold."""
        _r, sharp, _ = self._run("none", control_noise=0.05)
        _r, broad, _ = self._run("none", control_noise=0.30)
        assert min(sharp) > max(broad)

    def test_the_ladders_first_step_is_invariant_under_the_null_and_added_is_not(self):
        """A prior row travels with its position, so re-pairing positions against counts leaves what
        behaviour buys untouched and moves only what the neuron adds."""
        observed, _k, (_c, _p, _pr, sessions, _f) = self._run("region", 0.25)
        permutation = within_session_permutation(sessions, np.random.default_rng(7))
        drawn, _k, _ = self._run("region", 0.25, permutation=permutation)
        assert drawn["behaviour_over_no_behaviour"] == pytest.approx(observed["behaviour_over_no_behaviour"])
        assert drawn["added"] != pytest.approx(observed["added"])
        assert drawn["added"] < observed["added"]

    def test_the_same_permutation_reproduces_a_draw(self):
        """The same permutation gives the same added gain, so null draws are reproducible."""
        permutation = within_session_permutation(np.repeat(np.arange(3), 240),
                                                 np.random.default_rng(3))
        first, _k, _ = self._run("region", 0.25, permutation=permutation)
        second, _k, _ = self._run("region", 0.25, permutation=permutation)
        assert first["added"] == second["added"]

    def test_the_section_pieces_are_all_there(self):
        """The result carries every key the reporting reads, and every fold records its kappa and added
        gain."""
        result, _kappas, _ = self._run("region", 0.25)
        for key in ("added", "added_uncalibrated", "nats_per_spike", "behaviour_over_no_behaviour",
                    "full_over_no_behaviour", "n_masked", "n_folds_positive", "min_leave_one_fold_out",
                    "per_fold", "alphas", "ridges", "kappas"):
            assert key in result
        assert all("kappa" in fold and "added" in fold for fold in result["per_fold"])

    @staticmethod
    def _shrunk_control(n_sessions=3, per_session=240, seed=5):
        """
        Description
        -----------
        Build a CONCENTRATED occupancy (USVs cluster on the torus) with a SHRUNK control that returns
        the cluster's centre whatever the truth. It knows nothing beyond the base rate, so the right
        prior is the occupancy prior itself, kappa = 0 -- yet its residual spread is the truth's own
        spread, which is exactly what makes a residual-fitted kappa double-count the base rate.

        Parameters
        ----------
        n_sessions (int)
            Number of sessions; defaults to 3.
        per_session (int)
            Events per session; defaults to 240.
        seed (int)
            Seed for ``np.random.default_rng``; defaults to 5.

        Returns
        -------
        control (tuple[np.ndarray, np.ndarray, np.ndarray])
            ``(positions, predicted, sessions)``.
        """
        rng = np.random.default_rng(seed)
        n = n_sessions * per_session
        positions = (0.5 + rng.vonmises(0.0, 2.0, size=(n, 2)) / (2.0 * np.pi)) % 1.0
        predicted = (0.5 + 0.02 * rng.normal(size=(n, 2))) % 1.0
        sessions = np.repeat(np.arange(n_sessions), per_session)
        return positions, predicted, sessions

    @staticmethod
    def _over_occupancy(folds, priors):
        """
        Description
        -----------
        Compute the held-out log prior(truth) minus log occupancy(truth), pooled over the folds' test
        rows.

        Parameters
        ----------
        folds (list[dict])
            Decoding folds with ``truth_cell``, ``test_rows`` and the occupancy ``log_prior``.
        priors (list[np.ndarray])
            Per-fold behaviour log-prior arrays (events x grid cells).

        Returns
        -------
        gain (float)
            Mean nats per held-out event by which the behaviour prior beats occupancy (negative when
            it is worse).
        """
        values = []
        for fold, log_prior in zip(folds, priors, strict=True):
            cells = fold["truth_cell"][fold["test_rows"]]
            values.append(log_prior[fold["test_rows"], cells] - fold["log_prior"][cells])
        return float(np.nanmean(np.concatenate(values)))

    def test_a_residual_kappa_double_counts_the_base_rate_for_a_shrunk_control(self):
        """What this test pins is the MIS-SPECIFICATION: a shrunk predictor's residual spread is the
        truth's own spread, so the fitted kappa is well above zero and the prior it builds scores
        BELOW the occupancy prior at the truth."""
        positions, predicted, sessions = self._shrunk_control()
        folds = decoding_context(positions, sessions, self.DECODING, PERIOD)
        nested = copy.deepcopy(self.NESTED)
        nested["behaviour_prior"]["concentration"] = "train_residuals"
        kappas, priors = behaviour_priors(folds, predicted, positions, nested)
        assert min(kappas) > 0.5
        assert self._over_occupancy(folds, priors) < -0.02

    def test_an_inner_loso_kappa_never_leaves_the_behaviour_rung_below_occupancy(self):
        """The guarantee the tuned concentration gives: kappa = 0 is always on the grid, so the
        held-out prior cannot score below the occupancy prior by more than tuning noise, and the
        neuron is never credited for repairing a deficit the prior created. It is a NARROWER promise
        than kappa = 0 for a shrunk control -- on this fast grid the KDE over-smooths a tight cluster,
        so a tilt toward the centre re-sharpens the BASE RATE and legitimately earns a kappa of 0.5
        here."""
        positions, predicted, sessions = self._shrunk_control()
        folds = decoding_context(positions, sessions, self.DECODING, PERIOD)
        nested = copy.deepcopy(self.NESTED)
        nested["behaviour_prior"]["concentration"] = "train_tuned_inner_loso"
        tuned_kappas, tuned_priors = behaviour_priors(folds, predicted, positions, nested)
        nested["behaviour_prior"]["concentration"] = "train_residuals"
        residual_kappas, residual_priors = behaviour_priors(folds, predicted, positions, nested)
        tuned, residual = (self._over_occupancy(folds, tuned_priors),
                           self._over_occupancy(folds, residual_priors))
        assert max(tuned_kappas) < min(residual_kappas)
        assert tuned > residual
        assert tuned > -0.005

    def test_an_inner_loso_kappa_is_at_the_floor_for_a_know_nothing_control_on_a_broad_occupancy(self):
        """Where the occupancy prior is already well calibrated (a broad cluster the KDE does not
        over-smooth), a control unrelated to position has nothing to add and kappa sits at the
        bottom of the grid, to within the resolution the held-in criterion can tell apart."""
        rng = np.random.default_rng(11)
        n = 3 * 240
        positions = (0.5 + rng.vonmises(0.0, 0.3, size=(n, 2)) / (2.0 * np.pi)) % 1.0
        predicted = rng.random((n, 2))
        sessions = np.repeat(np.arange(3), 240)
        folds = decoding_context(positions, sessions, self.DECODING, PERIOD)
        nested = copy.deepcopy(self.NESTED)
        nested["behaviour_prior"]["concentration"] = "train_tuned_inner_loso"
        kappas, _priors = behaviour_priors(folds, predicted, positions, nested)
        assert max(kappas) <= nested["behaviour_prior"]["kappa_grid"][1]

    def test_an_inner_loso_kappa_is_positive_when_the_control_knows_something(self):
        """An accurate control earns a positive tuned kappa on every fold, so tuning does not collapse
        every prior to occupancy."""
        _counts, positions, predicted, sessions = self._unit(control_noise=0.05)
        folds = decoding_context(positions, sessions, self.DECODING, PERIOD)
        nested = copy.deepcopy(self.NESTED)
        nested["behaviour_prior"]["concentration"] = "train_tuned_inner_loso"
        kappas, _priors = behaviour_priors(folds, predicted, positions, nested)
        assert min(kappas) > 0.0

    @pytest.mark.parametrize(("path", "bad"), [(("statistic",), "point_regression"),
                                           (("behaviour_prior", "form"), "kde"),
                                           (("behaviour_prior", "concentration"), "cohort_fit"),
                                           (("discrimination_null", "method"), "circular_shift")])
    def test_unimplemented_posterior_options_are_refused(self, path, bad):
        """Each named posterior option that is not implemented must raise in ``check_settings``."""
        block = copy.deepcopy(self.NESTED)
        target = block
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = bad
        with pytest.raises(ValueError, match="not implemented"):
            check_settings(block)
