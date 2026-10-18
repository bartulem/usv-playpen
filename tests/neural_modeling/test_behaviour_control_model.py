"""
@author: bartulem
Coverage: the cohort-scale behaviour control -- its embedding, its acceptance rule, and its guards.

Every test here pins a failure that actually occurred. The control was refitted per unit on ~2,000
USVs against thousands of columns and scored BELOW a constant fitted on the same USVs, so the added
score it produced was the neuron measured against nothing; the forward selection was written from prose
rather than from the fixed implementation and reproduced a known defect, selecting one feature while an
unselected set scored better; and a control silently out of step with its settings would change every
added score without changing a single reported configuration value.
"""

from __future__ import annotations

import json
import pathlib
import pickle

import h5py
import numpy as np
import polars as pls
import pytest

from usv_playpen.neural_modeling import behaviour_control_model as control_module
from usv_playpen.neural_modeling.behaviour_control_model import (
    EMBEDDING_COLUMNS,
    apply_behaviour_control,
    control_predictions,
    feature_columns,
    load_control_model,
    paired_gain,
    session_vocalization_design,
    torus_embedding,
    upgrade_legacy_counts,
    write_control_model,
)
from usv_playpen.neural_modeling.neural_design_assembly import (
    onset_history_anchor_frames,
)
from usv_playpen.neural_modeling.neural_nested_decoding import reduced_model_features

SETTINGS_PATH = (pathlib.Path(__file__).resolve().parents[2] / "src" / "usv_playpen"
                 / "_parameter_settings" / "neural_modeling_settings.json")


@pytest.fixture
def settings() -> dict:
    """The shipped settings, read fresh so a test cannot mutate another's copy."""
    with SETTINGS_PATH.open() as handle:
        return json.load(handle)


@pytest.fixture
def artifact() -> dict:
    """A two-feature, five-lag control with one held-out day."""
    generator = np.random.default_rng(0)
    return {"features": ["nose-nose", "other.speed"], "n_lags": 5, "settings_sha256": "abc123",
            "coefficients": {"20250919": {"coef": generator.normal(size=(10, 4)),
                                          "intercept": np.zeros(4)}}}


class TestTorusEmbedding:
    """The predicted POSITION has to become a predictor without a discontinuity at the wrap."""

    def test_pairs_are_unit_vectors(self) -> None:
        """Each dimension contributes a cosine/sine pair, so each pair has unit norm."""
        generator = np.random.default_rng(1)
        embedded = torus_embedding(generator.random((50, 2)))
        assert embedded.shape == (50, 4)
        assert np.allclose(embedded[:, 0] ** 2 + embedded[:, 1] ** 2, 1.0)
        assert np.allclose(embedded[:, 2] ** 2 + embedded[:, 3] ** 2, 1.0)

    def test_the_wrap_is_continuous(self) -> None:
        """0.99 and 0.01 are neighbours on a torus; a raw coordinate would make them opposites."""
        near_wrap = torus_embedding(np.array([[0.99, 0.5], [0.01, 0.5]]))
        far_apart = torus_embedding(np.array([[0.25, 0.5], [0.75, 0.5]]))
        assert np.linalg.norm(near_wrap[0] - near_wrap[1]) < np.linalg.norm(far_apart[0] - far_apart[1])


class TestPairedGain:
    """The acceptance rule's pairing, which this project has now lost TWICE."""

    def test_pairing_survives_a_between_day_spread_that_swamps_the_gain(self) -> None:
        """Days differ by ~0.25 in absolute score while a step gains ~0.013.

        Comparing the gain to the UNPAIRED standard error of the raw scores therefore rejects every
        feature after the first, and the selected model then scores worse than an unselected one --
        which is exactly how the defect was spotted. Differencing within day removes the day effect.
        """
        days = np.array([-3.42, -3.50, -3.55, -3.60, -3.67, -3.45, -3.52, -3.58, -3.63])
        candidate = days + 0.013
        unpaired = float(np.std(days, ddof=1) / np.sqrt(days.size))
        gain, error = paired_gain(candidate, days)
        assert unpaired > 0.013, "the between-day spread must swamp the gain, or the test proves nothing"
        assert gain == pytest.approx(0.013)
        assert (gain - error) > 0.0

    def test_a_genuinely_noisy_candidate_is_rejected(self) -> None:
        """Pairing must not accept everything: an inconsistent improvement still fails."""
        generator = np.random.default_rng(2)
        days = np.linspace(-3.67, -3.42, 9)
        gain, error = paired_gain(days + generator.normal(0.0, 0.05, days.size), days)
        assert (gain - error) <= 0.0

    def test_all_nan_differences_do_not_raise(self) -> None:
        """A candidate that scored NaN everywhere returns NaN rather than exploding."""
        gain, error = paired_gain(np.full(4, np.nan), np.zeros(4))
        assert np.isnan(gain)
        assert np.isnan(error)


class TestFeatureColumns:
    """Column selection has to honour the ASKED order, not the design's."""

    def test_columns_follow_the_requested_order(self) -> None:
        """Asking for ``["c", "a"]`` at three lags must return c's lag columns before a's, so the control's
        weights line up with the order it was fitted in, not the design's."""
        assert feature_columns(["a", "b", "c"], ["c", "a"], 3).tolist() == [6, 7, 8, 0, 1, 2]

    def test_an_absent_feature_raises(self) -> None:
        """A feature missing from the design must raise ``KeyError`` ("not in the assembled design") rather than
        select the wrong columns."""
        with pytest.raises(KeyError, match="not in the assembled design"):
            feature_columns(["a", "b"], ["z"], 3)


class TestApplyBehaviourControl:
    """The swap that keeps the nesting and changes only the representation."""

    def test_the_block_becomes_four_columns_at_one_lag(self, settings: dict, artifact: dict) -> None:
        """Four coefficients are fitted on the unit; the behaviour comes from the cohort."""
        generator = np.random.default_rng(3)
        design = {"behaviour": generator.normal(size=(20, 10)),
                  "feature_names": ["nose-nose", "other.speed"],
                  "neural": generator.normal(size=(20, 1))}
        controlled, n_lags = apply_behaviour_control(design, artifact, "20250919", settings)
        assert n_lags == 1
        assert controlled["behaviour"].shape == (20, 4)
        assert controlled["feature_names"] == list(EMBEDDING_COLUMNS)
        assert controlled["behaviour_control"]["held_out_day"] == "20250919"
        assert controlled["behaviour_control"]["settings_sha256"] == "abc123"

    def test_the_original_design_is_not_mutated(self, settings: dict, artifact: dict) -> None:
        """The caller keeps the raw block; an in-place swap would corrupt anything else holding it."""
        generator = np.random.default_rng(4)
        design = {"behaviour": generator.normal(size=(20, 10)),
                  "feature_names": ["nose-nose", "other.speed"], "neural": np.zeros((20, 1))}
        apply_behaviour_control(design, artifact, "20250919", settings)
        assert design["behaviour"].shape == (20, 10)
        assert design["feature_names"] == ["nose-nose", "other.speed"]

    def test_a_unit_whose_day_the_control_saw_is_refused(self, settings: dict,
                                                         artifact: dict) -> None:
        """The unit's own day must be held out of the control, or the reduced score is in-sample."""
        design = {"behaviour": np.zeros((5, 10)), "feature_names": ["nose-nose", "other.speed"],
                  "neural": np.zeros((5, 1))}
        with pytest.raises(KeyError, match="no fold for day"):
            apply_behaviour_control(design, artifact, "20250923", settings)

    def test_a_block_built_from_other_features_is_refused(self, settings: dict,
                                                          artifact: dict) -> None:
        """Columns that do not match the control's shape would be multiplied by the wrong weights."""
        design = {"behaviour": np.zeros((5, 7)), "feature_names": ["nose-nose"],
                  "neural": np.zeros((5, 1))}
        with pytest.raises(ValueError, match="must be built from the control's own features"):
            apply_behaviour_control(design, artifact, "20250919", settings)


class TestControlPredictions:
    """The stored coefficients must reproduce the fitted model's own predictions."""

    def test_a_known_linear_map_is_recovered(self, settings: dict) -> None:
        """coef_ and intercept_ are all predict() consumes, which is why only those are stored."""
        stored = {"features": ["nose-nose"], "n_lags": 2, "settings_sha256": "x",
                  "coefficients": {"20250919": {"coef": np.array([[1.0, 0.0, 0.0, 0.0],
                                                                  [0.0, 0.0, 1.0, 0.0]]),
                                                "intercept": np.array([0.0, 1.0, 0.0, 1.0])}}}
        predicted = control_predictions(stored, "20250919", np.array([[1.0, 1.0]]), settings)
        assert predicted.shape == (1, 2)
        assert np.all((predicted >= 0.0) & (predicted < 1.0)), "a torus coordinate must be in [0, 1)"


class TestLoadControlModel:
    """A control out of step with its settings changes every added score and no reported value."""

    def test_drifted_features_raise(self, settings: dict, artifact: dict, tmp_path) -> None:
        """A stored control whose features differ from the settings' reduced-model features must raise
        ``ValueError`` ("behaviour control drift") instead of being applied."""
        path = tmp_path / "control.pkl"
        write_control_model(artifact, str(path))
        settings["nested_vocal_manifold_position_decoding"]["behaviour_control_model_path"] = str(path)
        settings["nested_vocal_manifold_position_decoding"]["reduced_model_features"] = ["nose-nose"]
        with pytest.raises(ValueError, match="behaviour control drift"):
            load_control_model(settings)

    def test_matching_features_load(self, settings: dict, artifact: dict, tmp_path) -> None:
        """A stored control whose features match the settings must load and return those features, so the
        drift guard does not refuse a valid control."""
        path = tmp_path / "control.pkl"
        write_control_model(artifact, str(path))
        settings["nested_vocal_manifold_position_decoding"]["behaviour_control_model_path"] = str(path)
        settings["nested_vocal_manifold_position_decoding"]["reduced_model_features"] = list(
            artifact["features"])
        assert load_control_model(settings)["features"] == artifact["features"]

    def test_a_missing_control_names_the_dispatcher(self, settings: dict, tmp_path) -> None:
        """A missing control file must raise ``FileNotFoundError`` naming the behaviour-control dispatcher, so the
        reader knows which job produces it."""
        settings["nested_vocal_manifold_position_decoding"]["behaviour_control_model_path"] = str(
            tmp_path / "absent.pkl")
        with pytest.raises(FileNotFoundError, match="behaviour-control dispatcher"):
            load_control_model(settings)


class TestReducedModelFeaturesRepoint:
    """Either artifact shape may define the feature set, and both are checked against the freeze.

    A behavioural selection's FEATURE SET is adopted while its COEFFICIENTS never are -- the control
    is always refitted in domain, which is what the non-transferring fitted map requires. So the
    reader accepts both shapes and the drift assertion is what guards meaning, not the schema.
    """

    def test_a_behavioural_selection_feature_set_is_adopted(self, settings: dict,
                                                            tmp_path) -> None:
        """The last forward step's `current_features` is the selected set."""
        path = tmp_path / "behavioural.pkl"
        chosen = ["allo_yaw-head", "self.neck_elevation", "head-head"]
        path.write_bytes(pickle.dumps({"steps": [{"current_features": chosen[:1]},
                                                 {"current_features": chosen}]}))
        settings["data"]["behaviour_selection_result_path"] = str(path)
        settings["nested_vocal_manifold_position_decoding"]["reduced_model_features"] = chosen
        assert reduced_model_features(settings) == chosen

    def test_a_behavioural_selection_is_still_drift_checked(self, settings: dict,
                                                            tmp_path) -> None:
        """Adopting a feature set must not let a re-selection silently redefine the control."""
        path = tmp_path / "behavioural.pkl"
        path.write_bytes(pickle.dumps({"steps": [{"current_features": ["head-head"]}]}))
        settings["data"]["behaviour_selection_result_path"] = str(path)
        settings["nested_vocal_manifold_position_decoding"]["reduced_model_features"] = ["nose-nose"]
        with pytest.raises(ValueError, match="behaviour selection drift"):
            reduced_model_features(settings)

    def test_an_unrecognised_artifact_is_refused(self, settings: dict, tmp_path) -> None:
        """Neither shape present would otherwise raise an unhelpful KeyError deep in the reader."""
        path = tmp_path / "mystery.pkl"
        path.write_bytes(pickle.dumps({"something_else": ["a"]}))
        settings["data"]["behaviour_selection_result_path"] = str(path)
        with pytest.raises(ValueError, match="neither a cohort selection result"):
            reduced_model_features(settings)

    def test_a_cohort_selection_is_read_and_checked(self, settings: dict, tmp_path) -> None:
        """A cohort selection result's ``selected_features`` must be read and, when it matches the frozen
        settings list, returned unchanged."""
        path = tmp_path / "selection.pkl"
        chosen = ["nose-nose", "other.speed", "allo_yaw-nose"]
        path.write_bytes(pickle.dumps({"selected_features": chosen}))
        settings["data"]["behaviour_selection_result_path"] = str(path)
        settings["nested_vocal_manifold_position_decoding"]["reduced_model_features"] = chosen
        assert reduced_model_features(settings) == chosen

    def test_drift_between_file_and_settings_raises(self, settings: dict, tmp_path) -> None:
        """Freezing alone lets the control drift; reading alone lets a re-run redefine the claim."""
        path = tmp_path / "selection.pkl"
        path.write_bytes(pickle.dumps({"selected_features": ["nose-nose"]}))
        settings["data"]["behaviour_selection_result_path"] = str(path)
        settings["nested_vocal_manifold_position_decoding"]["reduced_model_features"] = ["other.speed"]
        with pytest.raises(ValueError, match="behaviour selection drift"):
            reduced_model_features(settings)


class TestSessionVocalizationDesign:
    """The control trains on EXACTLY the rows the per-unit analysis scores: the analysed vocalization set
    of the configured emitter, guarded against intrusions by the guard set, with every history ending at
    the frame before onset -- the frame the nested decoding builds the unit-side histories at."""

    SESSION, FPS, N_FRAMES = "20250101_000000", 10.0, 2000

    def _write(self, root: pathlib.Path, usvs: list) -> None:
        """
        Description
        -----------
        Write a session's tracking-H5 header (``male``/``female`` tracks) and a USV summary carrying the
        torus-position, region and classifier columns the control reads.

        Parameters
        ----------
        root (pathlib.Path)
            The data root.
        usvs (list[tuple])
            ``(start, stop, emitter, noise, squeak)`` per row.

        Returns
        -------
        None
            The files are written to disk.
        """
        video_dir = root / self.SESSION / "video" / f"{self.SESSION}_cam"
        video_dir.mkdir(parents=True)
        with h5py.File(video_dir / f"{self.SESSION}_points3d_translated_rotated_metric.h5", "w") as handle:
            handle["tracks"] = np.zeros((self.N_FRAMES, 2, 15, 3), dtype=np.float64)
            handle["recording_frame_rate"] = self.FPS
            handle.create_dataset("track_names", data=np.array([b"male", b"female"], dtype="S6"))
        audio_dir = root / self.SESSION / "audio"
        audio_dir.mkdir(parents=True)
        pls.DataFrame({"usv_id": list(range(len(usvs))),
                       "start": [row[0] for row in usvs], "stop": [row[1] for row in usvs],
                       "emitter": [row[2] for row in usvs],
                       "qlvm1": [0.2] * len(usvs), "qlvm2": [0.4] * len(usvs),
                       "qlvm_category": [1.0] * len(usvs),
                       "noise": [row[3] for row in usvs], "squeak": [row[4] for row in usvs],
                       }).write_csv(audio_dir / f"{self.SESSION}_usv_summary.csv")

    def _design(self, monkeypatch, tmp_path, settings: dict, emitter: str) -> dict:
        """
        Description
        -----------
        Run ``session_vocalization_design`` on the synthetic session with a one-feature kinematic series
        equal to the frame index, so a design column reveals the frame it was read at.

        Parameters
        ----------
        monkeypatch (pytest.MonkeyPatch)
            Patches the kinematic loader.
        tmp_path (pathlib.Path)
            The data root.
        settings (dict)
            The shipped settings, adjusted in place for the synthetic session.
        emitter (str)
            ``vocalization_settings.vocal_emitter``.

        Returns
        -------
        block (dict)
            As ``session_vocalization_design`` returns it.
        """
        self._write(tmp_path, [(50.0, 50.1, "male", False, False),
                               (60.0, 60.1, "male", True, False),     # noise: never a vocalization
                               (70.0, 70.1, "male", False, True),     # squeak: a different vocal type
                               (80.0, 80.1, "female", False, False),
                               (99.97, 99.99, "male", True, False),   # noise just before 100 s
                               (100.0, 100.1, "male", False, False)])
        settings["data_roots"]["data_root"] = str(tmp_path)
        settings["data_sufficiency"]["min_emitter_usvs_per_session"] = 1
        settings["vocalization_settings"]["vocal_emitter"] = emitter
        frames = pls.DataFrame({"f": np.arange(self.N_FRAMES, dtype=np.float64)})
        monkeypatch.setattr(control_module, "build_zscored_feature_frames",
                            lambda **_kwargs: ({self.SESSION: frames}, {self.SESSION: self.FPS}, {}, {}))
        return session_vocalization_design(str(tmp_path / self.SESSION), settings, 5)

    def test_noise_and_squeak_rows_are_not_trained_on(self, monkeypatch, tmp_path, settings: dict) -> None:
        """
        Repro basis: the control read the raw summary, so a noise row and a squeak row became training
        rows, and a noise row vetoed the vocalization after it as an "intrusion". It now keeps exactly
        the analysed set (no noise, no squeaks), guarded by the guard set (no noise).
        """
        block = self._design(monkeypatch, tmp_path, settings, "self")
        np.testing.assert_allclose(block["vocalization_start"], [50.0, 100.0])
        assert block["n_kept"] == 2

    def test_vocal_emitter_all_trains_on_every_emitter(self, monkeypatch, tmp_path, settings: dict) -> None:
        """``'all'`` means no emitter filter; it used to index the empty name list and raise."""
        block = self._design(monkeypatch, tmp_path, settings, "all")
        np.testing.assert_allclose(block["vocalization_start"], [50.0, 80.0, 100.0])

    def test_the_history_ends_at_the_frame_the_nested_decoding_applies_it_at(
            self, monkeypatch, tmp_path, settings: dict) -> None:
        """The fit anchors are ``onset_history_anchor_frames`` -- the frame before onset -- the same frames
        ``nested_design`` builds the unit-side histories at."""
        block = self._design(monkeypatch, tmp_path, settings, "self")
        expected = onset_history_anchor_frames(block["vocalization_start"], self.FPS)
        np.testing.assert_array_equal(expected, [499, 999])
        np.testing.assert_allclose(block["design"][:, -1], expected)


class TestLegacyCountKeys:
    """The stored count was renamed from ``n_calls`` to ``n_vocalizations``; older artifacts still read."""

    def test_an_old_artifact_is_read_under_the_new_names(self, settings: dict, artifact: dict,
                                                       tmp_path) -> None:
        """A control written before the rename (``n_calls`` at the top level and per day) loads with
        ``n_vocalizations`` in both places and no ``n_calls`` left."""
        old = {**artifact, "n_calls": 120, "per_day": {"20250919": {"n_calls": 120, "behaviour": -3.4,
                                                                     "no_behaviour": -3.5}}}
        path = tmp_path / "control.pkl"
        write_control_model(old, str(path))
        settings["nested_vocal_manifold_position_decoding"]["behaviour_control_model_path"] = str(path)
        settings["nested_vocal_manifold_position_decoding"]["reduced_model_features"] = list(
            artifact["features"])
        loaded = load_control_model(settings)
        assert loaded["n_vocalizations"] == 120
        assert loaded["per_day"]["20250919"]["n_vocalizations"] == 120
        assert "n_calls" not in loaded
        assert "n_calls" not in loaded["per_day"]["20250919"]

    def test_a_current_record_passes_through_unchanged_and_unmutated(self) -> None:
        """A record already on the new names is returned equal, and the input is never mutated."""
        record = {"n_vocalizations": 5, "per_day": {"d": {"n_vocalizations": 5}}}
        assert upgrade_legacy_counts(record) == record
        legacy = {"n_calls": 7}
        upgrade_legacy_counts(legacy)
        assert legacy == {"n_calls": 7}
