"""
@author: bartulem
Coverage: the cohort-scale behaviour control -- its embedding, its acceptance rule, and its guards.

Every regression test here pins a failure that actually occurred. The control was refitted per unit on
~2,000 USVs against thousands of columns and scored BELOW a constant fitted on the same USVs, so the added
score it produced was the neuron measured against nothing; the forward selection was written from prose
rather than from the fixed implementation and reproduced a known defect, selecting one feature while an
unselected set scored better; and a control silently out of step with its settings would change every
added score without changing a single reported configuration value.

The cohort-scale stages are covered by KNOWN-ANSWER tests on a tiny synthetic cohort (four recording days,
forty USVs each, three features at two lags) in which one feature ("signal") drives the first torus
coordinate and the other two are pure noise: the day holdout must never let a day into its own training
set, the fitted control must recover the planted coefficient and beat the no-behaviour baseline, the
forward selection must pick the planted feature and nothing when nothing is planted, and the cohort
assembly must stack sessions in order, record each USV's day and refuse an empty or inconsistent cohort.
The process pools are replaced by an in-process serial executor so every worker body runs (and is
measured) inside the test process.
"""

from __future__ import annotations

import json
import pathlib
import pickle
import re
from typing import ClassVar

import h5py
import numpy as np
import polars as pls
import pytest

from usv_playpen.modeling.manifold_metric import inverse_region_frequency_weights
from usv_playpen.modeling.modeling_metadata import compute_settings_sha256
from usv_playpen.neural_modeling import behaviour_control_model as control_module
from usv_playpen.neural_modeling.behaviour_control_model import (
    EMBEDDING_COLUMNS,
    apply_behaviour_control,
    assemble_cohort,
    control_predictions,
    feature_columns,
    fit_control_models,
    held_out_by_day,
    load_control_model,
    paired_gain,
    pooled_macro,
    select_control_features,
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


SYNTHETIC_FEATURES = ["signal", "noise_a", "noise_b"]
SYNTHETIC_DAYS = ["20250101", "20250102", "20250103", "20250104"]
SYNTHETIC_N_LAGS = 2


class SerialExecutor:
    """
    Description
    -----------
    An in-process stand-in for ``concurrent.futures.ProcessPoolExecutor``.

    The cohort stages map their per-session and per-candidate work over a SPAWN-context process pool.
    Spawning re-imports the package in every child, which costs seconds per test and, worse, runs the
    worker bodies (``_assemble_one``, ``_score_candidate``) outside the measured process, so coverage
    never sees them. This executor runs ``map`` serially in the calling process, preserving the input
    order exactly as ``ProcessPoolExecutor.map`` does, and records the constructor arguments so a test
    can check that ``n_workers`` and the pool context were actually passed through.

    Parameters
    ----------
    max_workers (int)
        The worker count the caller asked for; recorded in ``SerialExecutor.requested_workers``.
    mp_context (object)
        The multiprocessing context the caller passed; recorded in ``SerialExecutor.contexts``.

    Returns
    -------
    executor (SerialExecutor)
        A context manager whose ``map`` returns a list of results in input order.
    """

    requested_workers: ClassVar[list] = []
    contexts: ClassVar[list] = []

    def __init__(self, max_workers: int, mp_context: object) -> None:
        """
        Description
        -----------
        Record the requested worker count and context on the class, so the test can read them after the
        executor has gone out of scope.

        Parameters
        ----------
        max_workers (int)
            The worker count the caller asked for.
        mp_context (object)
            The multiprocessing context the caller passed.

        Returns
        -------
        None
        """
        SerialExecutor.requested_workers.append(max_workers)
        SerialExecutor.contexts.append(mp_context)

    def __enter__(self) -> SerialExecutor:
        """
        Description
        -----------
        Enter the ``with`` block.

        Parameters
        ----------
        None

        Returns
        -------
        executor (SerialExecutor)
            This instance.
        """
        return self

    def __exit__(self, *_exc_info: object) -> bool:
        """
        Description
        -----------
        Leave the ``with`` block without suppressing any exception raised inside it.

        Parameters
        ----------
        *_exc_info (object)
            The exception type, value and traceback (all None on a clean exit); unused.

        Returns
        -------
        suppress (bool)
            Always False, so exceptions propagate exactly as they would from a real pool.
        """
        return False

    def map(self, function, iterable) -> list:
        """
        Description
        -----------
        Apply ``function`` to every item of ``iterable`` in order, in this process.

        Parameters
        ----------
        function (callable)
            The worker body.
        iterable (Iterable)
            The task arguments.

        Returns
        -------
        results (list)
            ``[function(item) for item in iterable]``.
        """
        return [function(item) for item in iterable]


@pytest.fixture
def serial_pools(monkeypatch) -> type:
    """
    Description
    -----------
    Replace the module's process pool with :class:`SerialExecutor` and its pool context with a sentinel
    string, clearing the executor's recorded arguments first so each test reads only its own pools.

    Parameters
    ----------
    monkeypatch (pytest.MonkeyPatch)
        Patches ``ProcessPoolExecutor`` and ``worker_process_context`` on the module under test.

    Returns
    -------
    executor_class (type)
        :class:`SerialExecutor`, whose ``requested_workers`` and ``contexts`` lists the test may inspect.
    """
    SerialExecutor.requested_workers = []
    SerialExecutor.contexts = []
    monkeypatch.setattr(control_module, "ProcessPoolExecutor", SerialExecutor)
    monkeypatch.setattr(control_module, "worker_process_context", lambda: "serial-context")
    return SerialExecutor


def synthetic_cohort(seed: int, planted: bool, per_day: int = 40) -> dict:
    """
    Description
    -----------
    A tiny cohort, in the exact shape ``assemble_cohort`` returns, with a KNOWN behaviour-to-position map.

    Each USV carries three features (``signal``, ``noise_a``, ``noise_b``) drawn uniformly on [-1, 1];
    each feature's value is repeated over ``SYNTHETIC_N_LAGS`` lag columns (feature-major, the layout
    ``feature_columns`` indexes) with small independent jitter (sd 0.05). When ``planted`` is True the
    first torus coordinate is ``0.25 + 0.15 * signal`` plus noise of sd 0.02, so ``signal`` drives
    position and the two noise features carry nothing; when False the first coordinate is ``0.25`` plus
    the same noise and no feature carries anything. The second coordinate is ``0.5`` plus noise of sd
    0.02 in both cases. All USVs share one acoustic region, and the days are ``SYNTHETIC_DAYS`` with
    ``per_day`` USVs each, enough to clear ``min_region_events`` (20) inside every day.

    Parameters
    ----------
    seed (int)
        Seed of the ``numpy`` generator, so every draw is reproducible.
    planted (bool)
        Whether ``signal`` drives the first torus coordinate.
    per_day (int)
        USVs per recording day.

    Returns
    -------
    cohort (dict)
        ``design`` (float32, ``(n, 3 * n_lags)``), ``positions`` (``(n, 2)`` in [0, 1)), ``regions``,
        ``days``, ``session_ids``, ``feature_names``, ``skipped`` (empty), plus ``signal`` (the
        undisturbed per-USV signal value, for building expected answers).
    """
    generator = np.random.default_rng(seed)
    n_rows = len(SYNTHETIC_DAYS) * per_day
    base = generator.uniform(-1.0, 1.0, (n_rows, len(SYNTHETIC_FEATURES)))
    design = np.repeat(base, SYNTHETIC_N_LAGS, axis=1) + generator.normal(
        0.0, 0.05, (n_rows, len(SYNTHETIC_FEATURES) * SYNTHETIC_N_LAGS)
    )
    drive = 0.15 * base[:, 0] if planted else 0.0
    first = (0.25 + drive + generator.normal(0.0, 0.02, n_rows)) % 1.0
    second = (0.5 + generator.normal(0.0, 0.02, n_rows)) % 1.0
    return {
        "design": design.astype(np.float32),
        "positions": np.column_stack([first, second]),
        "regions": np.ones(n_rows),
        "days": np.repeat(SYNTHETIC_DAYS, per_day),
        "session_ids": [f"{day}_120000" for day in SYNTHETIC_DAYS],
        "feature_names": list(SYNTHETIC_FEATURES),
        "skipped": {"20250105_120000": "3 tracked animals"},
        "signal": base[:, 0],
    }


def circular_error(predicted: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """
    Description
    -----------
    Wrap-aware absolute error per USV on the unit torus, taking the shorter way round on each axis and
    returning the larger of the two axis errors.

    Parameters
    ----------
    predicted (np.ndarray)
        ``(n, 2)`` predicted torus coordinates in [0, 1).
    truth (np.ndarray)
        ``(n, 2)`` true torus coordinates in [0, 1).

    Returns
    -------
    error (np.ndarray)
        ``(n,)`` per-USV error in torus units, in [0, 0.5].
    """
    difference = np.abs(predicted - truth) % 1.0
    return np.minimum(difference, 1.0 - difference).max(axis=1)



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

    def test_a_single_finite_day_returns_its_difference_with_zero_error(self) -> None:
        """
        Description
        -----------
        When only one day yields a finite difference, the gain is that one difference and the standard
        error is defined as 0.0 (a sample standard deviation of one value would be NaN under ``ddof=1``).
        Here three of four days are NaN and the surviving day improves by exactly 0.2.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        candidate = np.array([np.nan, -3.3, np.nan, np.nan])
        incumbent = np.array([-3.4, -3.5, np.nan, -3.6])
        gain, error = paired_gain(candidate, incumbent)
        assert gain == pytest.approx(0.2)
        assert error == 0.0


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

    def _write_tracking_only(self, root: pathlib.Path, track_names: list) -> None:
        """
        Description
        -----------
        Write ONLY the session's tracking-H5 header, with as many tracks as ``track_names`` names and no
        USV summary, for the skip paths that are decided before (or by) the summary read.

        Parameters
        ----------
        root (pathlib.Path)
            The data root.
        track_names (list[bytes])
            The ``track_names`` to store, one per tracked animal.

        Returns
        -------
        None
            The file is written to disk.
        """
        video_dir = root / self.SESSION / "video" / f"{self.SESSION}_cam"
        video_dir.mkdir(parents=True)
        with h5py.File(video_dir / f"{self.SESSION}_points3d_translated_rotated_metric.h5", "w") as handle:
            handle["tracks"] = np.zeros((self.N_FRAMES, len(track_names), 15, 3), dtype=np.float64)
            handle["recording_frame_rate"] = self.FPS
            handle.create_dataset("track_names", data=np.array(track_names, dtype="S6"))

    def test_a_session_without_exactly_two_animals_is_skipped(self, tmp_path, settings: dict) -> None:
        """
        Description
        -----------
        The emitter mapping needs a recorded animal and ONE partner, so a three-animal session must come
        back as a skip whose reason names the track count, before any summary is read (none exists here,
        so reaching the summary read would raise a different reason).

        Parameters
        ----------
        tmp_path (pathlib.Path)
            The data root.
        settings (dict)
            The shipped settings, pointed at the synthetic data root.

        Returns
        -------
        None
        """
        self._write_tracking_only(tmp_path, [b"male", b"female", b"third"])
        settings["data_roots"]["data_root"] = str(tmp_path)
        block = session_vocalization_design(str(tmp_path / self.SESSION), settings, 5)
        assert block == {"session_id": self.SESSION, "reason": "3 tracked animals"}

    def test_a_session_without_a_summary_is_skipped_not_raised(self, tmp_path, settings: dict) -> None:
        """
        Description
        -----------
        A session with tracking but no USV summary is a property of that one session, so it must be
        skipped with the loader's ``FileNotFoundError`` message as the reason rather than abort the
        cohort assembly.

        Parameters
        ----------
        tmp_path (pathlib.Path)
            The data root.
        settings (dict)
            The shipped settings, pointed at the synthetic data root.

        Returns
        -------
        None
        """
        self._write_tracking_only(tmp_path, [b"male", b"female"])
        settings["data_roots"]["data_root"] = str(tmp_path)
        block = session_vocalization_design(str(tmp_path / self.SESSION), settings, 5)
        assert set(block) == {"session_id", "reason"}
        assert block["reason"].startswith("usv_summary not found")

    def test_a_summary_missing_the_configured_columns_raises(self, tmp_path, settings: dict) -> None:
        """
        Description
        -----------
        A summary lacking a configured torus column hits every session alike, so it must RAISE a
        ``KeyError`` naming exactly the absent column and the two settings to reconcile, instead of being
        skipped (which is how this control once went silently unfitted).

        Parameters
        ----------
        tmp_path (pathlib.Path)
            The data root; the summary is rewritten without ``qlvm2``.
        settings (dict)
            The shipped settings, pointed at the synthetic data root.

        Returns
        -------
        None
        """
        self._write(tmp_path, [(50.0, 50.1, "male", False, False)])
        summary = tmp_path / self.SESSION / "audio" / f"{self.SESSION}_usv_summary.csv"
        pls.read_csv(summary).drop("qlvm2").write_csv(summary)
        settings["data_roots"]["data_root"] = str(tmp_path)
        with pytest.raises(KeyError, match=re.escape("usv_summary lacks ['qlvm2']")):
            session_vocalization_design(str(tmp_path / self.SESSION), settings, 5)

    def test_too_few_emitter_vocalizations_are_skipped_with_the_count(self, tmp_path, settings: dict) -> None:
        """
        Description
        -----------
        The session carries four male rows, of which one is noise and one a squeak, plus one female row,
        so exactly two analysed vocalizations from the recorded male remain. A per-session minimum of
        three must skip it with the reason ``"2 emitter vocalizations"`` -- the count AFTER the analysed-set
        and emitter filters, not the raw row count (5) -- and before any kinematics are loaded (none are
        patched here, so loading them would fail).

        Parameters
        ----------
        tmp_path (pathlib.Path)
            The data root.
        settings (dict)
            The shipped settings, pointed at the synthetic data root with a minimum of 3.

        Returns
        -------
        None
        """
        self._write(tmp_path, [(50.0, 50.1, "male", False, False),
                               (60.0, 60.1, "male", True, False),
                               (70.0, 70.1, "male", False, True),
                               (80.0, 80.1, "female", False, False),
                               (100.0, 100.1, "male", False, False)])
        settings["data_roots"]["data_root"] = str(tmp_path)
        settings["data_sufficiency"]["min_emitter_usvs_per_session"] = 3
        settings["vocalization_settings"]["vocal_emitter"] = "self"
        block = session_vocalization_design(str(tmp_path / self.SESSION), settings, 5)
        assert block == {"session_id": self.SESSION, "reason": "2 emitter vocalizations"}

    def test_a_history_longer_than_any_onset_leaves_nothing(
            self, monkeypatch, tmp_path, settings: dict) -> None:
        """
        Description
        -----------
        The surviving onsets sit at anchor frames 499 and 999; a 2,000-frame history therefore runs off the
        start of the session for both, so nothing survives and the session is skipped with that reason
        rather than ``lagged_design`` raising on a negative start index.

        Parameters
        ----------
        monkeypatch (pytest.MonkeyPatch)
            Patches the kinematic loader.
        tmp_path (pathlib.Path)
            The data root.
        settings (dict)
            The shipped settings, adjusted for the synthetic session.

        Returns
        -------
        None
        """
        self._write(tmp_path, [(50.0, 50.1, "male", False, False), (100.0, 100.1, "male", False, False)])
        settings["data_roots"]["data_root"] = str(tmp_path)
        settings["data_sufficiency"]["min_emitter_usvs_per_session"] = 1
        frames = pls.DataFrame({"f": np.arange(self.N_FRAMES, dtype=np.float64)})
        monkeypatch.setattr(control_module, "build_zscored_feature_frames",
                            lambda **_kwargs: ({self.SESSION: frames}, {self.SESSION: self.FPS}, {}, {}))
        block = session_vocalization_design(str(tmp_path / self.SESSION), settings, self.N_FRAMES)
        assert block == {"session_id": self.SESSION, "reason": "no vocalization survives the filters"}


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


class RecordingEstimator:
    """
    Description
    -----------
    A stand-in for the control estimator that RECORDS what each fold was fitted and asked to predict.

    The day holdout is the property under test, and it is invisible in a real estimator's output. This
    one stores, per instance, the rows it was trained on (read back from a design column that carries
    each USV's row index), the sample weights it received, and the rows it predicted; it predicts the
    plain mean of its training positions, so the expected held-out prediction for a day is computable by
    hand as the mean over the OTHER days.

    Parameters
    ----------
    n_features (int)
        Feature count the caller derived from the design; recorded for the test.
    n_lags (int)
        Lag count the caller passed; recorded for the test.
    registry (list)
        Shared list every new instance appends itself to.

    Returns
    -------
    estimator (RecordingEstimator)
        An unfitted recorder.
    """

    def __init__(self, n_features: int, n_lags: int, registry: list) -> None:
        """
        Description
        -----------
        Record the shape arguments and register the instance.

        Parameters
        ----------
        n_features (int)
            Feature count the caller derived from the design.
        n_lags (int)
            Lag count the caller passed.
        registry (list)
            Shared list this instance appends itself to.

        Returns
        -------
        None
        """
        self.shape = (n_features, n_lags)
        self.train_rows = None
        self.test_rows = None
        self.sample_weight = None
        self.mean_position = None
        self.snap = None
        registry.append(self)

    def fit(self, design: np.ndarray, positions: np.ndarray, sample_weight: np.ndarray) -> RecordingEstimator:
        """
        Description
        -----------
        Store the training rows (design column 0 holds each USV's row index), the weights, and the mean
        training position.

        Parameters
        ----------
        design (np.ndarray)
            Training design; column 0 is the row index.
        positions (np.ndarray)
            ``(n_train, 2)`` training positions.
        sample_weight (np.ndarray)
            Per-row training weights.

        Returns
        -------
        self (RecordingEstimator)
            The fitted recorder.
        """
        self.train_rows = design[:, 0].astype(np.int64)
        self.sample_weight = np.asarray(sample_weight, dtype=np.float64)
        self.mean_position = positions.mean(axis=0)
        return self

    def predict(self, design: np.ndarray, snap: bool) -> np.ndarray:
        """
        Description
        -----------
        Store the predicted rows and the ``snap`` flag, and predict the mean training position for every
        row.

        Parameters
        ----------
        design (np.ndarray)
            Test design; column 0 is the row index.
        snap (bool)
            The caller's snap flag; recorded so the test can confirm predictions are unsnapped.

        Returns
        -------
        predicted (np.ndarray)
            ``(n_test, 2)`` copies of the mean training position.
        """
        self.test_rows = design[:, 0].astype(np.int64)
        self.snap = snap
        return np.tile(self.mean_position, (design.shape[0], 1))


class TestHeldOutByDay:
    """Every day is predicted exactly once, by a model that never saw that day."""

    def test_no_day_is_ever_in_its_own_training_set(self, monkeypatch, settings: dict) -> None:
        """
        Description
        -----------
        Three days of unequal size (3, 4 and 5 USVs) and two regions of unequal size. A recording estimator
        replaces the real one, and the design's first column carries each row's index. The test checks,
        fold by fold, that: (i) one fold is fitted per unique day; (ii) its training rows are EXACTLY the
        rows of the other days and its test rows exactly the held-out day's, so train and test are disjoint
        and the test sets partition every row once; (iii) the weights it receives are the cohort-wide
        inverse-region-frequency weights restricted to the training rows (computed on the whole cohort,
        not refitted per fold); (iv) the feature count passed is ``n_columns // n_lags`` and predictions
        are requested unsnapped; and (v) each held-out row's prediction equals the hand-computed mean of
        the other days' positions.

        Parameters
        ----------
        monkeypatch (pytest.MonkeyPatch)
            Replaces ``control_estimator`` with the recorder factory.
        settings (dict)
            The shipped settings (unused by the recorder, passed through by the function).

        Returns
        -------
        None
        """
        days = np.array(["20250103"] * 3 + ["20250101"] * 4 + ["20250102"] * 5)
        n_rows = days.size
        design = np.column_stack([np.arange(n_rows, dtype=np.float64), np.zeros(n_rows),
                                  np.zeros(n_rows), np.zeros(n_rows)])
        positions = np.column_stack([np.linspace(0.05, 0.6, n_rows), np.linspace(0.3, 0.4, n_rows)])
        regions = np.array([1.0, 2.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0])
        registry: list = []
        monkeypatch.setattr(control_module, "control_estimator",
                            lambda n_features, n_lags, _settings: RecordingEstimator(n_features, n_lags,
                                                                                    registry))
        predicted = held_out_by_day(design, positions, regions, days, settings, n_lags=2)

        assert len(registry) == 3
        weights = inverse_region_frequency_weights(regions)
        seen_test_rows = []
        for fold in registry:
            held_out_days = set(days[fold.test_rows])
            assert len(held_out_days) == 1
            held_out = held_out_days.pop()
            np.testing.assert_array_equal(fold.test_rows, np.flatnonzero(days == held_out))
            np.testing.assert_array_equal(fold.train_rows, np.flatnonzero(days != held_out))
            assert held_out not in set(days[fold.train_rows])
            np.testing.assert_allclose(fold.sample_weight, weights[fold.train_rows])
            assert fold.shape == (2, 2)
            assert fold.snap is False
            expected = positions[days != held_out].mean(axis=0)
            np.testing.assert_allclose(predicted[days == held_out], np.tile(expected, ((days == held_out).sum(), 1)))
            seen_test_rows.extend(fold.test_rows.tolist())
        assert sorted(seen_test_rows) == list(range(n_rows))
        assert np.isfinite(predicted).all()

    def test_a_planted_feature_is_predicted_out_of_day(self, settings: dict) -> None:
        """
        Description
        -----------
        With the real estimator on the planted synthetic cohort, the held-out predictions from the full
        design must track the truth closely (median wrap-aware error under 0.03 torus units, against a
        noise sd of 0.02 and a signal range of 0.3), while predictions from an all-zero design -- the
        no-behaviour baseline -- can only recover the training mean and so miss by far more (median error
        above 0.05, the mean absolute value of ``0.15 * U(-1, 1)`` being 0.075). The held-out macro score
        must order the two the same way.

        Parameters
        ----------
        settings (dict)
            The shipped settings (closed-form ridge, so the fit is fast).

        Returns
        -------
        None
        """
        cohort = synthetic_cohort(seed=0, planted=True)
        predicted = held_out_by_day(cohort["design"], cohort["positions"], cohort["regions"], cohort["days"],
                                    settings, SYNTHETIC_N_LAGS)
        blank = held_out_by_day(np.zeros_like(cohort["design"][:, :SYNTHETIC_N_LAGS]), cohort["positions"],
                                cohort["regions"], cohort["days"], settings, SYNTHETIC_N_LAGS)
        assert np.median(circular_error(predicted, cohort["positions"])) < 0.03
        assert np.median(circular_error(blank, cohort["positions"])) > 0.05
        assert (pooled_macro(predicted, cohort["positions"], cohort["regions"], settings)
                > pooled_macro(blank, cohort["positions"], cohort["regions"], settings))


class TestFitControlModels:
    """The per-day fit must recover the planted map and report what it was fitted on."""

    def test_the_planted_coefficient_is_recovered_in_every_fold(self, settings: dict) -> None:
        """
        Description
        -----------
        The first torus coordinate is ``0.25 + 0.15 * signal``, so its cosine embedding is
        ``-sin(0.3 * pi * signal)``: decreasing in ``signal``. Each fold's coefficients for the cosine output
        on the two ``signal`` lag columns are first-order-smoothed and must therefore be near-equal, and
        their SUM must match the slope of an ordinary least-squares line of that cosine on the mean of the
        two lag columns, fitted on the same training days (within 5%: the ridge and smoothness penalties
        shrink it only slightly at this sample size). The noise feature's cosine coefficients must be an
        order of magnitude smaller. The features are requested as ``["noise_a", "signal"]``, so the
        signal's coefficients must sit in ROWS 2-3 -- the requested order, not the cohort's.

        Parameters
        ----------
        settings (dict)
            The shipped settings.

        Returns
        -------
        None
        """
        cohort = synthetic_cohort(seed=0, planted=True)
        artifact = fit_control_models(cohort, ["noise_a", "signal"], settings, SYNTHETIC_N_LAGS)
        assert sorted(artifact["coefficients"]) == SYNTHETIC_DAYS
        cosine = np.cos(2.0 * np.pi * cohort["positions"][:, 0])
        signal_columns = cohort["design"][:, 0:2].astype(np.float64).mean(axis=1)
        for day, fold in artifact["coefficients"].items():
            assert fold["coef"].shape == (4, 4)
            assert fold["intercept"].shape == (4,)
            train = cohort["days"] != day
            slope = np.polyfit(signal_columns[train], cosine[train], 1)[0]
            signal_coef = fold["coef"][2:4, 0]
            assert slope < -0.5
            assert signal_coef.sum() == pytest.approx(slope, rel=0.05)
            assert signal_coef[0] == pytest.approx(signal_coef[1], abs=0.05)
            assert np.abs(fold["coef"][0:2, 0]).max() < 0.1 * np.abs(signal_coef).max()

    def test_the_artifact_reports_scores_counts_and_provenance(self, settings: dict) -> None:
        """
        Description
        -----------
        The artifact must carry: the requested features and lag count; one per-day entry per day holding
        that day's USV count (40) and a behaviour score that BEATS that day's no-behaviour score (the
        planted feature is real); a pooled held-out score above the pooled no-behaviour score; the total
        USV count (160); the cohort's session ids and skip reasons copied through; and the SHA-256 of the
        exact settings it was fitted under, so a drifted control is detectable.

        Parameters
        ----------
        settings (dict)
            The shipped settings.

        Returns
        -------
        None
        """
        cohort = synthetic_cohort(seed=0, planted=True)
        artifact = fit_control_models(cohort, ["signal"], settings, SYNTHETIC_N_LAGS)
        assert artifact["features"] == ["signal"]
        assert artifact["n_lags"] == SYNTHETIC_N_LAGS
        assert sorted(artifact["per_day"]) == SYNTHETIC_DAYS
        for scores in artifact["per_day"].values():
            assert scores["n_vocalizations"] == 40
            assert scores["behaviour"] > scores["no_behaviour"]
        assert artifact["held_out_score"] > artifact["no_behaviour_score"]
        assert artifact["n_vocalizations"] == 160
        assert artifact["session_ids"] == cohort["session_ids"]
        assert artifact["skipped"] == cohort["skipped"]
        assert artifact["settings_sha256"] == compute_settings_sha256(settings)

    def test_the_stored_folds_reproduce_the_held_out_predictions(self, settings: dict) -> None:
        """
        Description
        -----------
        Only ``coef_`` and ``intercept_`` are stored, on the claim that they are all ``predict`` consumes.
        Re-predicting each day from its stored fold through ``control_predictions`` and re-scoring must
        therefore give back the artifact's own per-day behaviour score to numerical precision.

        Parameters
        ----------
        settings (dict)
            The shipped settings.

        Returns
        -------
        None
        """
        cohort = synthetic_cohort(seed=1, planted=True)
        artifact = fit_control_models(cohort, ["signal", "noise_b"], settings, SYNTHETIC_N_LAGS)
        columns = feature_columns(cohort["feature_names"], ["signal", "noise_b"], SYNTHETIC_N_LAGS)
        for day in SYNTHETIC_DAYS:
            rows = cohort["days"] == day
            predicted = control_predictions(artifact, day, cohort["design"][rows][:, columns], settings)
            rescored = pooled_macro(predicted, cohort["positions"][rows], cohort["regions"][rows], settings)
            assert rescored == pytest.approx(artifact["per_day"][day]["behaviour"], abs=1e-9)


class TestSelectControlFeatures:
    """Forward selection must find a planted feature and refuse to invent one from noise."""

    def _select(self, tmp_path: pathlib.Path, settings: dict, planted: bool) -> dict:
        """
        Description
        -----------
        Build the seed-0 synthetic cohort, store its design as the ``.npy`` the workers memory-map, and run
        ``select_control_features`` with ``noise_a`` as the external reference and two requested workers.

        Parameters
        ----------
        tmp_path (pathlib.Path)
            Where the design ``.npy`` is written.
        settings (dict)
            The shipped settings.
        planted (bool)
            Whether ``signal`` drives position in the synthetic cohort.

        Returns
        -------
        selection (dict)
            As ``select_control_features`` returns it.
        """
        cohort = synthetic_cohort(seed=0, planted=planted)
        design_path = tmp_path / "design.npy"
        np.save(design_path, cohort["design"])
        return select_control_features(cohort, settings, SYNTHETIC_N_LAGS, 2, str(design_path),
                                       external_reference=["noise_a"])

    def test_the_planted_feature_is_selected_and_the_noise_is_not(self, tmp_path, settings: dict,
                                                                 serial_pools: type) -> None:
        """
        Description
        -----------
        With ``signal`` driving position, the first step must accept ``signal`` with a paired gain that
        clears its own standard error, the second step must REJECT the better of the two noise features
        (gain less error not above zero) and stop, and the selection must be exactly ``["signal"]``. The
        selected model's pooled score must beat the no-behaviour score, the external reference (a noise
        feature chosen "elsewhere") must score below the selected model, and the bookkeeping (days, count,
        sessions, skips, settings hash) must describe the cohort it ran on. Each step opens one pool with
        the requested two workers and the module's pool context.

        Parameters
        ----------
        tmp_path (pathlib.Path)
            Where the design ``.npy`` is written.
        settings (dict)
            The shipped settings.
        serial_pools (type)
            The in-process executor class, whose recorded pool arguments are checked.

        Returns
        -------
        None
        """
        selection = self._select(tmp_path, settings, planted=True)
        assert selection["selected_features"] == ["signal"]
        assert [step["feature"] for step in selection["per_step"]] == ["signal"]
        first = selection["per_step"][0]
        assert first["paired_gain"] - first["standard_error"] > 0.0
        assert first["pooled"] == selection["pooled_score"]
        assert selection["rejected"]["feature"] in {"noise_a", "noise_b"}
        assert selection["rejected"]["paired_gain"] - selection["rejected"]["standard_error"] <= 0.0
        assert selection["pooled_score"] > selection["no_behaviour_score"]
        assert selection["external_reference"]["features"] == ["noise_a"]
        assert selection["external_reference"]["pooled"] < selection["pooled_score"]
        assert selection["days"] == SYNTHETIC_DAYS
        assert selection["n_vocalizations"] == 160
        assert selection["session_ids"] == [f"{day}_120000" for day in SYNTHETIC_DAYS]
        assert selection["skipped"] == {"20250105_120000": "3 tracked animals"}
        assert selection["settings_sha256"] == compute_settings_sha256(settings)
        assert serial_pools.requested_workers == [2, 2]
        assert serial_pools.contexts == ["serial-context", "serial-context"]

    def test_pure_noise_selects_nothing(self, tmp_path, settings: dict, serial_pools: type) -> None:
        """
        Description
        -----------
        With no feature carrying position, the very first step must be rejected, so the selection is EMPTY,
        no step is recorded, and the reported pooled score is the no-behaviour score itself (nothing was
        added to it). Only one pool is opened, because the loop stops at the first rejection.

        Parameters
        ----------
        tmp_path (pathlib.Path)
            Where the design ``.npy`` is written.
        settings (dict)
            The shipped settings.
        serial_pools (type)
            The in-process executor class, whose recorded pool arguments are checked.

        Returns
        -------
        None
        """
        selection = self._select(tmp_path, settings, planted=False)
        assert selection["selected_features"] == []
        assert selection["per_step"] == []
        assert selection["rejected"]["feature"] in set(SYNTHETIC_FEATURES)
        assert selection["rejected"]["paired_gain"] - selection["rejected"]["standard_error"] <= 0.0
        assert selection["pooled_score"] == selection["no_behaviour_score"]
        assert serial_pools.requested_workers == [2]

    def test_no_external_reference_records_none(self, tmp_path, settings: dict, serial_pools: type) -> None:
        """
        Description
        -----------
        Without an external reference the result must record ``None`` for it rather than score an empty
        feature set; the selection itself is unaffected (still exactly ``["signal"]``).

        Parameters
        ----------
        tmp_path (pathlib.Path)
            Where the design ``.npy`` is written.
        settings (dict)
            The shipped settings.
        serial_pools (type)
            The in-process executor class (patched in; not inspected here).

        Returns
        -------
        None
        """
        cohort = synthetic_cohort(seed=0, planted=True)
        design_path = tmp_path / "design.npy"
        np.save(design_path, cohort["design"])
        selection = select_control_features(cohort, settings, SYNTHETIC_N_LAGS, 1, str(design_path))
        assert selection["external_reference"] is None
        assert selection["selected_features"] == ["signal"]
        assert serial_pools.requested_workers == [1, 1]


    def test_accepting_every_feature_ends_the_loop_without_a_rejection(self, tmp_path, settings: dict,
                                                                     serial_pools: type) -> None:
        """
        Description
        -----------
        A cohort whose ONLY feature is the planted one accepts it at the first step and then runs out of
        candidates, so the loop must end on exhaustion: the selection is ``["signal"]``, ``rejected`` is
        ``None`` (nothing was turned down), and exactly one pool was opened.

        Parameters
        ----------
        tmp_path (pathlib.Path)
            Where the design ``.npy`` is written.
        settings (dict)
            The shipped settings.
        serial_pools (type)
            The in-process executor class, whose recorded pool arguments are checked.

        Returns
        -------
        None
        """
        cohort = synthetic_cohort(seed=0, planted=True)
        cohort["design"] = np.ascontiguousarray(cohort["design"][:, :SYNTHETIC_N_LAGS])
        cohort["feature_names"] = ["signal"]
        design_path = tmp_path / "design.npy"
        np.save(design_path, cohort["design"])
        selection = select_control_features(cohort, settings, SYNTHETIC_N_LAGS, 1, str(design_path))
        assert selection["selected_features"] == ["signal"]
        assert selection["rejected"] is None
        assert selection["pooled_score"] == selection["per_step"][0]["pooled"]
        assert serial_pools.requested_workers == [1]


class TestAssembleCohort:
    """Per-session blocks are stacked in order, each USV keeps its day, and a bad cohort is refused."""

    @staticmethod
    def _block(session_id: str, n_kept: int, value: float, names: list) -> dict:
        """
        Description
        -----------
        A per-session block in the shape ``session_vocalization_design`` returns, filled with a constant so
        the stacked rows reveal which session they came from.

        Parameters
        ----------
        session_id (str)
            The session directory basename; its first eight characters are the day.
        n_kept (int)
            USVs the session contributes.
        value (float)
            Constant filling the design, positions and regions.
        names (list)
            The resolved feature names.

        Returns
        -------
        block (dict)
            ``session_id``, ``design`` (``(n_kept, 2)``), ``positions``, ``regions``, ``feature_names``,
            ``n_kept`` and ``vocalization_start``.
        """
        return {"session_id": session_id, "design": np.full((n_kept, 2), value, dtype=np.float32),
                "positions": np.full((n_kept, 2), value / 10.0), "regions": np.full(n_kept, value),
                "feature_names": list(names), "n_kept": n_kept, "vocalization_start": np.arange(n_kept, dtype=float)}

    def _patch(self, monkeypatch, blocks: dict) -> None:
        """
        Description
        -----------
        Replace ``session_vocalization_design`` with a lookup into ``blocks`` keyed by session directory,
        checking along the way that the worker shim passes the settings and lag count through unchanged.

        Parameters
        ----------
        monkeypatch (pytest.MonkeyPatch)
            Patches the per-session assembler on the module under test.
        blocks (dict)
            ``{session_directory: block}``.

        Returns
        -------
        None
        """
        def lookup(session_directory: str, run_settings: dict, n_lags: int) -> dict:
            """
            Description
            -----------
            Return the prepared block for ``session_directory`` after checking the forwarded arguments.

            Parameters
            ----------
            session_directory (str)
                The session directory the shim forwarded.
            run_settings (dict)
                The settings the shim forwarded; must be the sentinel dict.
            n_lags (int)
                The lag count the shim forwarded; must be 7.

            Returns
            -------
            block (dict)
                ``blocks[session_directory]``.
            """
            assert run_settings == {"sentinel": True}
            assert n_lags == 7
            return blocks[session_directory]

        monkeypatch.setattr(control_module, "session_vocalization_design", lookup)

    def test_blocks_are_stacked_in_order_with_days_and_skips(self, monkeypatch, serial_pools: type) -> None:
        """
        Description
        -----------
        Three listed sessions: two contribute 2 and 3 USVs (two different days), one is skipped with a
        reason. The result must stack the two contributing blocks in LISTED order (2 rows of 1.0 then 3 of
        2.0), give every USV the first eight characters of its session id as its day, list only the
        contributing session ids, carry the skip reason under the skipped session's id, keep the shared
        feature names, and open one pool with the requested worker count.

        Parameters
        ----------
        monkeypatch (pytest.MonkeyPatch)
            Patches the per-session assembler.
        serial_pools (type)
            The in-process executor class, whose recorded pool arguments are checked.

        Returns
        -------
        None
        """
        names = ["self.speed", "head-head"]
        self._patch(monkeypatch, {
            "/data/20250101_100000": self._block("20250101_100000", 2, 1.0, names),
            "/data/20250102_100000": {"session_id": "20250102_100000", "reason": "3 tracked animals"},
            "/data/20250103_100000": self._block("20250103_100000", 3, 2.0, names)})
        cohort = assemble_cohort(["/data/20250101_100000", "/data/20250102_100000", "/data/20250103_100000"],
                                 {"sentinel": True}, 7, 3)
        np.testing.assert_array_equal(cohort["design"][:, 0], [1.0, 1.0, 2.0, 2.0, 2.0])
        np.testing.assert_allclose(cohort["positions"][:, 1], [0.1, 0.1, 0.2, 0.2, 0.2])
        np.testing.assert_array_equal(cohort["regions"], [1.0, 1.0, 2.0, 2.0, 2.0])
        assert cohort["days"].tolist() == ["20250101"] * 2 + ["20250103"] * 3
        assert cohort["session_ids"] == ["20250101_100000", "20250103_100000"]
        assert cohort["skipped"] == {"20250102_100000": "3 tracked animals"}
        assert cohort["feature_names"] == names
        assert serial_pools.requested_workers == [3]
        assert serial_pools.contexts == ["serial-context"]

    def test_a_cohort_with_no_contributing_session_is_refused(self, monkeypatch, serial_pools: type) -> None:
        """
        Description
        -----------
        When every listed session is skipped there is nothing to fit, so the assembly must raise
        ``ValueError`` saying no session contributed, rather than return an empty design for the fit to
        fail on later.

        Parameters
        ----------
        monkeypatch (pytest.MonkeyPatch)
            Patches the per-session assembler.
        serial_pools (type)
            The in-process executor class (patched in; not inspected here).

        Returns
        -------
        None
        """
        self._patch(monkeypatch, {"/data/20250101_100000": {"session_id": "20250101_100000",
                                                            "reason": "0 emitter vocalizations"}})
        with pytest.raises(ValueError, match="no session contributed USVs"):
            assemble_cohort(["/data/20250101_100000"], {"sentinel": True}, 7, 1)
        assert serial_pools.requested_workers == [1]

    def test_sessions_resolving_different_features_are_refused(self, monkeypatch, serial_pools: type) -> None:
        """
        Description
        -----------
        Two sessions resolving the same features in a DIFFERENT order would stack design columns that do
        not line up, so the assembly must raise ``ValueError`` naming the offending (second) session.

        Parameters
        ----------
        monkeypatch (pytest.MonkeyPatch)
            Patches the per-session assembler.
        serial_pools (type)
            The in-process executor class, whose recorded pool arguments are checked.

        Returns
        -------
        None
        """
        self._patch(monkeypatch, {
            "/data/20250101_100000": self._block("20250101_100000", 2, 1.0, ["a", "b"]),
            "/data/20250102_100000": self._block("20250102_100000", 2, 2.0, ["b", "a"])})
        with pytest.raises(ValueError, match="20250102_100000 resolved a different feature set"):
            assemble_cohort(["/data/20250101_100000", "/data/20250102_100000"], {"sentinel": True}, 7, 1)
        assert serial_pools.requested_workers == [1]
