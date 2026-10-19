"""
@author: bartulem
Unit tests for ``usv_playpen.neural_modeling.main_neural_encoding_dispatcher``.

Coverage: the cluster dispatcher of the kinematic encoding -- one outer fold per job (``run_fold``), the
step that pools every fold (``combine``), the select-once variant (``run_single``), and the command line
(``dispatch`` / ``main``). The dispatcher's own logic is WIRING: which session is held out, which settings
values reach the screen, the fit and the nulls, where each artifact lands and under which keys, and how
fold artifacts are merged into one result. The heavy fitters (screen, forward selection, the quiet fit,
the transfer pooling) are replaced by stubs that record their arguments and return small known objects,
so every path runs in milliseconds and every assertion is about what the dispatcher handed on. The cheap
pieces (``spike_labels_at_frames``, ``encoding_metrics``, ``escalated_empirical_pvalue``,
``filter_descriptors``, both artifact writers, ``representative_fold`` and ``per_fold_roster``) run for
real.
"""

from __future__ import annotations

import collections
import copy
import json
import pathlib
import sys
import types
from zlib import crc32

import numpy as np
import polars as pls
import pytest

from usv_playpen.neural_modeling import main_neural_encoding_dispatcher as enc
from usv_playpen.neural_modeling.neural_artifacts import read_unit_artifact

SETTINGS_PATH = (
    pathlib.Path(__file__).resolve().parents[2]
    / "src"
    / "usv_playpen"
    / "_parameter_settings"
    / "neural_modeling_settings.json"
)
FEATURES = ["f0", "f1", "f2"]
FPS = 10.0
N_LAGS = 40
N_FRAMES = 1000


def _settings() -> dict:
    """
    Description
    -----------
    Loads the shipped neural-modeling settings fresh from disk and deep-copies them, so a test that
    overrides a value cannot leak the change into another test.

    Parameters
    ----------
    None.

    Returns
    -------
    settings (dict)
        The parsed ``neural_modeling_settings.json``.
    """
    with SETTINGS_PATH.open() as handle:
        return copy.deepcopy(json.load(handle))


def _unit(sessions: list, vocal_sessions: list) -> dict:
    """
    Description
    -----------
    A unit record shaped exactly as ``build_unit_record`` returns it, for a synthetic unit of mouse
    ``m1`` recorded on 2025-01-01.

    Parameters
    ----------
    sessions (list)
        The unit's courtship sessions.
    vocal_sessions (list)
        The subset of ``sessions`` that passes the vocal gate.

    Returns
    -------
    unit (dict)
        ``unit_uid``, ``mouse_id``, ``rec_date``, ``unit_id``, ``courtship_sessions``,
        ``vocal_sessions`` and ``emitter_usvs_per_session``.
    """
    return {
        "unit_uid": "m1_20250101_imec0_cl0001_ch001_good",
        "mouse_id": "m1",
        "rec_date": 20250101,
        "unit_id": "imec0_cl0001_ch001_good",
        "courtship_sessions": list(sessions),
        "vocal_sessions": list(vocal_sessions),
        "emitter_usvs_per_session": dict.fromkeys(sessions, 50),
    }


def _per_session(quiet_sizes: dict, spike_counts: dict) -> dict:
    """
    Description
    -----------
    A synthetic ``assemble_unit_sessions`` output: every session runs at 10 fps (so a 4 s history is 40
    lags) over 1,000 frames with three kinematic features, its quiet anchors are the contiguous frames
    starting at frame 100, and its spikes are spread evenly over those anchors so both spike labels occur.

    Parameters
    ----------
    quiet_sizes (dict)
        ``{session_id: number of quiet anchors}``.
    spike_counts (dict)
        ``{session_id: number of spikes}``, each fewer than that session's quiet anchors.

    Returns
    -------
    per_session (dict)
        ``{session_id: session dict}`` with ``fps``, ``feature_names``, ``n_frames``, ``quiet``,
        ``spike_frames``, ``feature_time_series`` and ``track_names``.
    """
    per_session = {}
    for session_id, size in quiet_sizes.items():
        quiet = np.arange(100, 100 + size, dtype=np.int64)
        per_session[session_id] = {
            "fps": FPS,
            "feature_names": list(FEATURES),
            "n_frames": N_FRAMES,
            "quiet": quiet,
            "spike_frames": np.linspace(
                100, 100 + size - 1, spike_counts[session_id]
            ).astype(np.int64),
            "feature_time_series": np.zeros((N_FRAMES, len(FEATURES))),
            "track_names": ["m1", "f1"],
        }
    return per_session


def _recording_stub(calls: dict, name: str, result):
    """
    Description
    -----------
    A stand-in for a dispatcher dependency that appends every invocation to ``calls[name]`` as an
    ``(args, kwargs)`` pair and returns ``result``. When ``result`` is callable it is called with that
    pair, so a stub's return value can depend on what it was handed (e.g. an array sized to the frames it
    was given).

    Parameters
    ----------
    calls (dict)
        A ``collections.defaultdict(list)`` collecting every invocation by stub name.
    name (str)
        The key this stub records under.
    result (Any)
        A fixed return value, or a callable ``result((args, kwargs))`` computing one.

    Returns
    -------
    stub (Callable)
        The recording stand-in.
    """

    def stub(*args, **kwargs):
        """
        Description
        -----------
        Records the invocation and returns the configured result.

        Parameters
        ----------
        *args
            Positional arguments, recorded.
        **kwargs
            Keyword arguments, recorded.

        Returns
        -------
        result (Any)
            The configured result, or what the configured callable computed from the invocation.
        """
        calls[name].append((args, kwargs))
        return result((args, kwargs)) if callable(result) else result

    return stub


def _estimator(n_selected: int, converged: bool = True) -> types.SimpleNamespace:
    """
    Description
    -----------
    A fitted-estimator stand-in carrying exactly the attributes the dispatcher reads: a flat filter of
    ``n_selected * 40`` weights, an intercept, an iteration count and a convergence flag.

    Parameters
    ----------
    n_selected (int)
        Number of selected features, which sets the filter length.
    converged (bool)
        Whether the final fit reports convergence.

    Returns
    -------
    estimator (types.SimpleNamespace)
        ``coef_``, ``intercept_``, ``n_iter_``, ``converged_``.
    """
    return types.SimpleNamespace(
        coef_=np.linspace(-1.0, 1.0, n_selected * N_LAGS),
        intercept_=-2.5,
        n_iter_=17,
        converged_=converged,
    )


def _score_fold_result(call: tuple) -> dict:
    """
    Description
    -----------
    What the ``score_fold`` stub returns for one session: a per-frame linear predictor and labels sized
    to the vocal frames it was handed (argument 3), and fixed scalar scores.

    Parameters
    ----------
    call (tuple)
        The ``(args, kwargs)`` pair the stub received.

    Returns
    -------
    result (dict)
        ``eta``, ``labels``, ``fold_score``, ``fold_slope``, ``auroc``, ``spike_rate``, ``n_frames``.
    """
    frames = call[0][3]
    return {
        "eta": np.linspace(-1.0, 1.0, frames.size),
        "labels": (np.arange(frames.size) % 3 == 0).astype(np.float64),
        "fold_score": 0.02,
        "fold_slope": 0.5,
        "auroc": 0.6,
        "spike_rate": 0.1,
        "n_frames": int(frames.size),
    }


def _screen_rows(survived: tuple, scores: tuple) -> list:
    """
    Description
    -----------
    A screen table in the shape ``screen_features`` returns: one row per feature with its index, name,
    survival flag, score and p-value.

    Parameters
    ----------
    survived (tuple)
        Survival flag per feature.
    scores (tuple)
        Held-out score per feature (NaN for a feature that could not be scored).

    Returns
    -------
    rows (list)
        One dict per feature.
    """
    return [
        {
            "feature": index,
            "name": FEATURES[index],
            "survived": flag,
            "score": score,
            "p": 0.2,
        }
        for index, (flag, score) in enumerate(zip(survived, scores, strict=True))
    ]


def _install_fold_stubs(
    monkeypatch,
    per_session: dict,
    screen_rows: list,
    selected: list,
    blocks: dict | None = None,
    converged: bool = True,
) -> dict:
    """
    Description
    -----------
    Replaces every heavy dependency of ``run_fold`` / ``run_single`` / ``combine`` with a recording stub.
    The quiet score returned by ``frozen_model_score`` is the scored anchor count divided by 1,000, so
    sessions with different quiet sizes get different, known quiet scores; the quiet null is a ramp of
    small values that every such score beats, so the p-value sits at the floor of the draws taken.

    Parameters
    ----------
    monkeypatch (pytest.MonkeyPatch)
        The fixture used to install the stubs.
    per_session (dict)
        What ``assemble_unit_sessions`` returns.
    screen_rows (list)
        What ``screen_features`` returns.
    selected (list)
        The feature indices ``forward_select`` selects.
    blocks (dict)
        What ``quiet_block_sessions`` returns; ``None`` leaves it returning no blocks.
    converged (bool)
        Whether the final fit reports convergence.

    Returns
    -------
    calls (dict)
        ``{stub name: [(args, kwargs), ...]}``, filled as the dispatcher runs.
    """
    calls = collections.defaultdict(list)
    stubs = {
        "assemble_unit_sessions": per_session,
        "focal_vocal_frames": (
            np.arange(200, 260, dtype=np.int64),
            np.array([0, 30, 60], dtype=np.int64),
            np.array([1.5, 0.25]),
        ),
        "transfer_frames": lambda call: np.asarray(call[0][0]),
        "quiet_block_sessions": {} if blocks is None else blocks,
        "screen_features": screen_rows,
        "forward_select": (list(selected), [{"step": 1, "feature": selected[0]}]),
        "fit_quiet_model": (_estimator(len(selected), converged), 0.05),
        "linear_predictor_at_frames": lambda call: np.linspace(
            -1.0, 1.0, call[0][3].size
        ),
        "frozen_model_score": lambda call: (call[0][0].size / 1000.0, 1.2),
        "frozen_null_scores": lambda call: np.linspace(-0.01, 0.01, call[0][5]),
        "score_fold": _score_fold_result,
    }
    for name, result in stubs.items():
        monkeypatch.setattr(enc, name, _recording_stub(calls, name, result))
    return calls


def _combine_stubs(
    monkeypatch,
    calls: dict,
    per_session: dict,
    transfer_p: float,
    transfer_slope: float,
) -> None:
    """
    Description
    -----------
    Replaces the dependencies ``combine`` and ``run_single`` call after the folds exist -- the session
    assembly, the transfer pooling, the filter band and the spike-triggered average -- with recording
    stubs. ``combine_folds`` returns a transfer result carrying the given p-value and slope.

    Parameters
    ----------
    monkeypatch (pytest.MonkeyPatch)
        The fixture used to install the stubs.
    calls (dict)
        A ``collections.defaultdict(list)`` the stubs record into.
    per_session (dict)
        What ``assemble_unit_sessions`` returns.
    transfer_p (float)
        The pooled transfer p-value.
    transfer_slope (float)
        The pooled transfer calibration slope.

    Returns
    -------
    None
    """
    transfer = {
        "score": 0.013,
        "slope": transfer_slope,
        "p": transfer_p,
        "at_floor": False,
        "null": np.linspace(-0.01, 0.01, 7),
        "n_frames": 120,
        "n_bins": 12,
        "bin_width_frames": 10,
        "grain": "count",
        "universe": "bout_window",
        "leave_one_session_out": [0.01, 0.02],
        "min_leave_one_session_out": 0.01,
        "folds": [{"session": "a"}],
    }
    monkeypatch.setattr(
        enc,
        "assemble_unit_sessions",
        _recording_stub(calls, "assemble_unit_sessions", per_session),
    )
    monkeypatch.setattr(
        enc, "combine_folds", _recording_stub(calls, "combine_folds", transfer)
    )
    monkeypatch.setattr(
        enc,
        "filter_band",
        _recording_stub(calls, "filter_band", {"lower": np.zeros(3)}),
    )
    monkeypatch.setattr(
        enc,
        "spike_triggered_average",
        _recording_stub(calls, "spike_triggered_average", {"sta": np.ones(3)}),
    )


def _toy_fold_artifact(
    fold_index: int,
    test_id: str,
    quiet_score: float,
    quiet_p: float,
    n_vocal_frames: int,
    no_model: bool = False,
    best_screened: bool = False,
) -> dict:
    """
    Description
    -----------
    A fold artifact carrying every key ``run_fold`` writes on its model path, with known toy values, so
    ``combine`` can be tested on fold results whose merge is predictable. The fold selects ``f0`` and
    ``f2``, and its model is carried onto two vocal sessions, ``s0`` and ``s1``.

    Parameters
    ----------
    fold_index (int)
        Which fold this is.
    test_id (str)
        The held-out session.
    quiet_score (float)
        The fold's held-out quiet score.
    quiet_p (float)
        The fold's quiet p-value.
    n_vocal_frames (int)
        Size of the held-out session's vocal frames (the even-fold-count tie-breaker).
    no_model (bool)
        Whether the fold is flagged as having no selected model.
    best_screened (bool)
        Whether the fold's model is only the best screened feature.

    Returns
    -------
    artifact (dict)
        The fold artifact.
    """
    everywhere = [
        {
            "session_id": session_id,
            "frames": np.arange(10 * (slot + 1)),
            "eta": np.full(10 * (slot + 1), float(fold_index)),
            "labels": np.zeros(10 * (slot + 1)),
            "fold_score": 0.01 * (slot + 1),
            "pointer": np.array([0, 10 * (slot + 1)]),
            "gaps": np.array([1.0]),
        }
        for slot, session_id in enumerate(["s0", "s1"])
    ]
    return {
        "fold_index": fold_index,
        "test_id": test_id,
        "coefficients": np.full(2 * N_LAGS, float(fold_index + 1)),
        "intercept": -1.0 - fold_index,
        "n_iter": 10 + fold_index,
        "fit_seconds": 1.5 + fold_index,
        "quiet_metrics": {"deviance_explained": quiet_score, "auroc": 0.7},
        "selected": ["f0", "f2"],
        "selected_indices": [0, 2],
        "path": [{"step": 1}],
        "no_model": no_model,
        "best_screened": best_screened,
        "quiet_score": quiet_score,
        "quiet_slope": 0.9,
        "quiet_null": np.linspace(-0.01, 0.01, 5),
        "quiet_p": quiet_p,
        "quiet_at_floor": False,
        "eta_vocal": np.zeros(n_vocal_frames),
        "labels_vocal": np.zeros(n_vocal_frames),
        "pointer": np.array([0, n_vocal_frames]),
        "gaps": np.array([1.0]),
        "vocal_frames": np.arange(n_vocal_frames),
        "screen": [{"feature": 0, "survived": True}],
        "fold_vocal_score": 0.01,
        "fold_vocal_slope": 0.4,
        "auroc_vocal": 0.55,
        "spike_rate_vocal": 0.1,
        "final_fit_converged": True,
        "vocal_everywhere": everywhere,
    }


def _no_model_fold_artifact(fold_index: int, test_id: str) -> dict:
    """
    Description
    -----------
    A fold artifact in the shape ``run_fold`` writes when nothing survived the screen and no feature
    could be scored: no filter, no quiet metrics, quiet p 1.0, ``best_screened`` False.

    Parameters
    ----------
    fold_index (int)
        Which fold this is.
    test_id (str)
        The held-out session.

    Returns
    -------
    artifact (dict)
        The no-model fold artifact.
    """
    return {
        "fold_index": fold_index,
        "test_id": test_id,
        "selected": [],
        "path": [],
        "no_model": True,
        "quiet_score": 0.0,
        "quiet_slope": np.nan,
        "quiet_p": 1.0,
        "quiet_at_floor": False,
        "eta_vocal": np.zeros(0),
        "labels_vocal": np.zeros(0),
        "pointer": np.zeros(1, dtype=np.int64),
        "gaps": np.zeros(0),
        "screen": [{"feature": 1, "survived": False}],
        "best_screened": False,
    }


class TestFocalVocalFrames:
    """The focal USVs whose frames the transfer scores, resolved from ``vocalization_settings``."""

    @pytest.mark.parametrize(
        ("emitter", "frames", "pointer", "gaps"),
        [
            ("self", [100, 101, 102, 300, 301, 302], [0, 3, 6], [7.9, 19.75]),
            ("partner", [200, 201], [0, 2], [np.inf]),
            (
                "all",
                [100, 101, 102, 200, 201, 300, 301, 302],
                [0, 3, 5, 8],
                [7.9, 9.75, 9.8],
            ),
        ],
    )
    def test_the_emitter_setting_picks_whose_usvs_are_scored(
        self, monkeypatch, emitter, frames, pointer, gaps
    ):
        """
        Description
        -----------
        A session table holds USVs of the recorded animal ``m1`` (at 2.0 s, 10.0 s and 30.0 s, listed out
        of order), of its partner ``f1`` (20.0 s), and one noise and one squeak row of ``m1``.
        ``vocal_emitter`` must decide whose USVs are kept (``self`` -> ``m1``, ``partner`` -> ``f1``,
        ``all`` -> everyone), noise and squeaks must be dropped per the exclusion flags, USVs must be
        sorted by onset, the 2.0 s USV must be dropped because its 4 s history runs off the start
        (at 10 fps, 40 lags), and each kept USV must carry the silent gap from the previous one in the
        same focal set.

        Parameters
        ----------
        monkeypatch (pytest.MonkeyPatch)
            Replaces the USV table loader.
        emitter (str)
            ``vocalization_settings.vocal_emitter``.
        frames (list)
            Expected flattened vocal frames.
        pointer (list)
            Expected per-USV offsets into ``frames``.
        gaps (list)
            Expected silent gap before each kept USV.

        Returns
        -------
        None
        """
        table = pls.DataFrame(
            {
                "start": [30.0, 20.0, 10.0, 2.0, 40.0, 50.0],
                "stop": [30.3, 20.2, 10.25, 2.1, 40.1, 50.1],
                "emitter": ["m1", "f1", "m1", "m1", "m1", "m1"],
                "noise": [False, False, False, False, True, False],
                "squeak": [False, False, False, False, False, True],
            }
        )
        calls = collections.defaultdict(list)
        monkeypatch.setattr(
            enc, "load_session_usvs", _recording_stub(calls, "load_session_usvs", table)
        )
        vocal_settings = _settings()["vocalization_settings"]
        vocal_settings["vocal_emitter"] = emitter
        session = {"n_frames": N_FRAMES, "track_names": ["m1", "f1"]}

        got_frames, got_pointer, got_gaps = enc.focal_vocal_frames(
            session, "s0", "/data", "m1", FPS, N_LAGS, vocal_settings
        )

        assert calls["load_session_usvs"][0][0] == ("/data", "s0")
        np.testing.assert_array_equal(got_frames, frames)
        np.testing.assert_array_equal(got_pointer, pointer)
        np.testing.assert_allclose(got_gaps, gaps)

    def test_a_session_where_the_focal_animal_never_vocalized_contributes_nothing(
        self, monkeypatch
    ):
        """
        Description
        -----------
        When the focal animal has no USV in the session -- ordinary, in 8 of 77 courtship sessions -- the
        session contributes empty frames and gaps and a one-element pointer, instead of raising.

        Parameters
        ----------
        monkeypatch (pytest.MonkeyPatch)
            Replaces the USV table loader.

        Returns
        -------
        None
        """
        table = pls.DataFrame(
            {
                "start": [10.0],
                "stop": [10.2],
                "emitter": ["f1"],
                "noise": [False],
                "squeak": [False],
            }
        )
        monkeypatch.setattr(
            enc,
            "load_session_usvs",
            _recording_stub(collections.defaultdict(list), "load", table),
        )
        frames, pointer, gaps = enc.focal_vocal_frames(
            {"n_frames": N_FRAMES, "track_names": ["m1", "f1"]},
            "s0",
            "/data",
            "m1",
            FPS,
            N_LAGS,
            _settings()["vocalization_settings"],
        )
        assert frames.size == 0
        assert gaps.size == 0
        np.testing.assert_array_equal(pointer, [0])

    def test_no_representative_fold_exists_when_no_fold_has_a_model(self):
        """
        Description
        -----------
        ``representative_fold`` must return ``-1`` rather than an index when every fold is flagged
        ``no_model``, so a caller cannot mistake an absent model for fold 0.

        Parameters
        ----------
        None.

        Returns
        -------
        None
        """
        assert (
            enc.representative_fold(
                [{"no_model": True}, {"no_model": np.asarray(True)}]
            )
            == -1
        )


class TestRunFold:
    """One outer fold: the held-out session, the settings each stage receives, and the fold file it
    leaves for ``combine``."""

    def test_a_tuned_fold_reaches_every_stage_with_the_settings_values_and_writes_its_artifact(
        self, tmp_path, monkeypatch
    ):
        """
        Description
        -----------
        Fold 1 of a three-session unit holds out ``s1`` and screens, selects and fits on ``s0`` and
        ``s2``. Every stage must receive the settings values it is documented to read, the escalation
        ladder must top the quiet null up through the SAME draw closure (9 draws, then 10 more), the
        generator must be seeded from ``null.shuffle_seed`` and the held-out session id, the model must be
        carried onto exactly the vocal-eligible sessions, and the artifact must land at
        ``<output>/<unit_id>_fold1.npz`` with the documented keys.

        Parameters
        ----------
        tmp_path (pathlib.Path)
            Output directory for the fold file.
        monkeypatch (pytest.MonkeyPatch)
            Installs the stubs.

        Returns
        -------
        None
        """
        settings = _settings()
        settings["kinematic_encoding"]["clean_post_seconds"] = 1.5
        settings["null"]["n_shuffles"] = 9
        settings["null"]["shuffle_seed"] = 123
        settings["significance"]["escalation_ladder"] = [19]
        per_session = _per_session(
            {"s0": 300, "s1": 200, "s2": 250}, {"s0": 30, "s1": 20, "s2": 25}
        )
        calls = _install_fold_stubs(
            monkeypatch,
            per_session,
            _screen_rows((True, False, True), (0.02, 0.001, 0.03)),
            [0, 2],
        )
        unit = _unit(["s0", "s1", "s2"], ["s0", "s1"])

        artifact = enc.run_fold(
            unit, 1, settings, "/data", str(tmp_path), message_output=lambda _text: None
        )

        assemble_args = calls["assemble_unit_sessions"][0][0]
        assert assemble_args == (
            unit,
            "/data",
            settings["kinematic_features"],
            settings["kinematic_encoding"]["history_pre_seconds"],
            1.5,
            settings["vocalization_settings"]["clean_against"],
            settings["vocalization_settings"],
        )
        screen_args = calls["screen_features"][0][0]
        assert screen_args[1] == ["s0", "s2"]
        assert screen_args[2] == N_LAGS
        assert screen_args[4] is settings
        assert screen_args[5] == FEATURES
        expected_rng = np.random.default_rng(123 + crc32(b"s1") % 10_000)
        assert screen_args[3].bit_generator.state == expected_rng.bit_generator.state
        assert calls["forward_select"][0][0][2] == [0, 2]
        fit_args = calls["fit_quiet_model"][0][0]
        assert fit_args[1] == ["s0", "s2"]
        assert fit_args[2] == [0, 2]
        assert fit_args[5] is settings["kinematic_encoding"]

        null_calls = [call[0] for call in calls["frozen_null_scores"]]
        assert [args[5] for args in null_calls] == [9, 10]
        assert null_calls[0][7] == settings["null"]["shuffle_guard_seconds"]
        assert (
            null_calls[0][8]
            == settings["kinematic_encoding"]["solver"]["calibration_steps"]
        )
        assert (
            null_calls[0][9]
            == settings["kinematic_encoding"]["solver"]["null_calibration_bins"]
        )
        assert null_calls[0][10] == settings["kinematic_encoding"]["calibration"]
        assert (
            calls["frozen_model_score"][0][0][6]
            == settings["kinematic_encoding"]["calibration"]
        )

        # the held-out session first, then every VOCAL session the model is carried onto -- never s2
        assert [call[0][1] for call in calls["focal_vocal_frames"]] == [
            "s1",
            "s0",
            "s1",
        ]
        assert calls["focal_vocal_frames"][0][0][6] is settings["vocalization_settings"]
        assert all(
            call[0][6] is settings["kinematic_encoding"]["transfer"]
            for call in calls["transfer_frames"]
        )

        assert artifact["quiet_score"] == pytest.approx(0.2)
        assert artifact["quiet_null"].size == 19
        assert artifact["quiet_p"] == pytest.approx(1.0 / 20.0)
        assert artifact["quiet_at_floor"]
        assert artifact["selected"] == ["f0", "f2"]
        assert artifact["no_model"] is False
        assert artifact["best_screened"] is False
        assert artifact["final_fit_converged"] is True
        assert [entry["session_id"] for entry in artifact["vocal_everywhere"]] == [
            "s0",
            "s1",
        ]

        stored = dict(
            np.load(tmp_path / "imec0_cl0001_ch001_good_fold1.npz", allow_pickle=True)
        )
        assert set(stored) == set(artifact)
        assert str(stored["test_id"]) == "s1"
        assert [str(name) for name in stored["selected"]] == ["f0", "f2"]
        assert enc.unwrap_stored(stored["quiet_metrics"])["n_frames"] == 200
        np.testing.assert_array_equal(stored["coefficients"], artifact["coefficients"])

    def test_a_fold_whose_final_fit_did_not_converge_says_so(
        self, tmp_path, monkeypatch
    ):
        """
        Description
        -----------
        A final fit that did not converge must be flagged in the artifact AND in the log, since every
        number the fold reports rests on it.

        Parameters
        ----------
        tmp_path (pathlib.Path)
            Output directory for the fold file.
        monkeypatch (pytest.MonkeyPatch)
            Installs the stubs.

        Returns
        -------
        None
        """
        settings = _settings()
        settings["null"]["n_shuffles"] = 5
        per_session = _per_session(
            {"s0": 300, "s1": 200, "s2": 250}, {"s0": 30, "s1": 20, "s2": 25}
        )
        _install_fold_stubs(
            monkeypatch,
            per_session,
            _screen_rows((True, False, False), (0.02, 0.0, 0.0)),
            [0],
            converged=False,
        )
        messages = []
        artifact = enc.run_fold(
            _unit(["s0", "s1", "s2"], ["s0"]),
            0,
            settings,
            "/data",
            str(tmp_path),
            messages.append,
        )
        assert artifact["final_fit_converged"] is False
        assert any("DID NOT CONVERGE" in text for text in messages)

    def test_a_single_pool_session_without_two_quiet_blocks_writes_a_no_model_fold(
        self, tmp_path, monkeypatch
    ):
        """
        Description
        -----------
        A two-session unit leaves one pool session. Its inner split comes from
        ``data_sufficiency.single_session_inner_split_blocks`` contiguous blocks separated by half the
        history (``ceil(4 s * 10 fps / 2) = 20`` frames per end); with fewer than two usable blocks the
        fold has no model, and that fold must STILL be written -- ``combine`` demands a file for every
        fold -- without ever screening.

        Parameters
        ----------
        tmp_path (pathlib.Path)
            Output directory for the fold file.
        monkeypatch (pytest.MonkeyPatch)
            Installs the stubs.

        Returns
        -------
        None
        """
        settings = _settings()
        settings["data_sufficiency"]["single_session_inner_split_blocks"] = 3
        per_session = _per_session({"s0": 300, "s1": 200}, {"s0": 30, "s1": 20})
        calls = _install_fold_stubs(
            monkeypatch, per_session, [], [0], blocks={"s0#0": per_session["s0"]}
        )

        artifact = enc.run_fold(
            _unit(["s0", "s1"], ["s0", "s1"]),
            1,
            settings,
            "/data",
            str(tmp_path),
            lambda _text: None,
        )

        block_args = calls["quiet_block_sessions"][0][0]
        assert block_args[0] is per_session["s0"]
        assert block_args[1:] == ("s0", 3, 20)
        assert "screen_features" not in calls
        assert artifact["no_model"] is True
        assert artifact["quiet_p"] == 1.0
        stored = dict(
            np.load(tmp_path / "imec0_cl0001_ch001_good_fold1.npz", allow_pickle=True)
        )
        assert bool(stored["no_model"])
        assert list(stored["vocal_everywhere"]) == []

    def test_a_single_pool_session_with_blocks_screens_on_the_blocks_but_fits_on_the_session(
        self, tmp_path, monkeypatch
    ):
        """
        Description
        -----------
        With at least two quiet blocks the screen and forward selection run over the blocks as
        pseudo-sessions, while the final model is still fitted on the whole pool session.

        Parameters
        ----------
        tmp_path (pathlib.Path)
            Output directory for the fold file.
        monkeypatch (pytest.MonkeyPatch)
            Installs the stubs.

        Returns
        -------
        None
        """
        settings = _settings()
        settings["null"]["n_shuffles"] = 5
        per_session = _per_session({"s0": 300, "s1": 200}, {"s0": 30, "s1": 20})
        blocks = {"s0#0": per_session["s0"], "s0#1": per_session["s0"]}
        calls = _install_fold_stubs(
            monkeypatch,
            per_session,
            _screen_rows((True, False, False), (0.02, 0.0, 0.0)),
            [0],
            blocks=blocks,
        )
        enc.run_fold(
            _unit(["s0", "s1"], ["s0", "s1"]),
            1,
            settings,
            "/data",
            str(tmp_path),
            lambda _text: None,
        )
        screen_args = calls["screen_features"][0][0]
        assert screen_args[1] == ["s0#0", "s0#1"]
        assert set(screen_args[0]) == {"s0", "s1", "s0#0", "s0#1"}
        assert calls["forward_select"][0][0][1] == ["s0#0", "s0#1"]
        assert calls["fit_quiet_model"][0][0][1] == ["s0"]

    def test_with_no_survivor_the_best_screened_feature_is_fitted_and_flagged(
        self, tmp_path, monkeypatch
    ):
        """
        Description
        -----------
        When nothing survives the screen, the best finite-scoring feature (``f1`` here; ``f2`` scored
        NaN) is fitted and scored for description only: forward selection is skipped, and the artifact
        says both ``no_model`` and ``best_screened``.

        Parameters
        ----------
        tmp_path (pathlib.Path)
            Output directory for the fold file.
        monkeypatch (pytest.MonkeyPatch)
            Installs the stubs.

        Returns
        -------
        None
        """
        settings = _settings()
        settings["null"]["n_shuffles"] = 5
        per_session = _per_session(
            {"s0": 300, "s1": 200, "s2": 250}, {"s0": 30, "s1": 20, "s2": 25}
        )
        calls = _install_fold_stubs(
            monkeypatch,
            per_session,
            _screen_rows((False, False, False), (0.01, 0.03, np.nan)),
            [1],
        )
        artifact = enc.run_fold(
            _unit(["s0", "s1", "s2"], ["s0"]),
            2,
            settings,
            "/data",
            str(tmp_path),
            lambda _text: None,
        )
        assert "forward_select" not in calls
        assert calls["fit_quiet_model"][0][0][2] == [1]
        assert artifact["selected"] == ["f1"]
        assert artifact["path"] == []
        assert artifact["no_model"] is True
        assert artifact["best_screened"] is True

    def test_with_no_scorable_feature_the_fold_writes_its_screen_and_no_model(
        self, tmp_path, monkeypatch
    ):
        """
        Description
        -----------
        When no feature survives and none even has a finite score, nothing is fitted; the fold is written
        as a no-model fold that keeps its screen table, with ``best_screened`` False.

        Parameters
        ----------
        tmp_path (pathlib.Path)
            Output directory for the fold file.
        monkeypatch (pytest.MonkeyPatch)
            Installs the stubs.

        Returns
        -------
        None
        """
        per_session = _per_session(
            {"s0": 300, "s1": 200, "s2": 250}, {"s0": 30, "s1": 20, "s2": 25}
        )
        rows = _screen_rows((False, False, False), (np.nan, np.nan, np.nan))
        calls = _install_fold_stubs(monkeypatch, per_session, rows, [0])
        artifact = enc.run_fold(
            _unit(["s0", "s1", "s2"], ["s0"]),
            0,
            _settings(),
            "/data",
            str(tmp_path),
            lambda _text: None,
        )
        assert "fit_quiet_model" not in calls
        assert artifact["no_model"] is True
        assert artifact["best_screened"] is False
        assert artifact["screen"] == rows
        assert (tmp_path / "imec0_cl0001_ch001_good_fold0.npz").exists()


class TestCombine:
    """Pooling the folds: the representative model, the merged roster, and the result section."""

    def test_folds_written_by_run_fold_are_combined_around_the_median_fold(
        self, tmp_path, monkeypatch
    ):
        """
        Description
        -----------
        End to end on the dispatcher's own files: three folds are run with quiet scores 0.30 (``s0``),
        0.20 (``s1``) and 0.25 (``s2``), so the representative is fold 2. ``combine`` must take fold 2's
        quiet numbers, pool fold 2's model on every vocal session (``s0``, ``s1``) with ``session_id`` and
        ``frames`` stripped from each per-session result, compute the band and the spike-triggered
        average on the sessions other than ``s2`` with the settings values, and merge the result into
        ``<output>/<unit_uid>.pkl`` under ``kinematic_encoding`` with a per-fold roster of three.

        Parameters
        ----------
        tmp_path (pathlib.Path)
            Output directory for the fold files and the unit file.
        monkeypatch (pytest.MonkeyPatch)
            Installs the stubs.

        Returns
        -------
        None
        """
        settings = _settings()
        settings["null"]["n_shuffles"] = 5
        settings["kinematic_encoding"]["filter_band"]["n_resamples"] = 4
        per_session = _per_session(
            {"s0": 300, "s1": 200, "s2": 250}, {"s0": 30, "s1": 20, "s2": 25}
        )
        unit = _unit(["s0", "s1", "s2"], ["s0", "s1"])
        calls = _install_fold_stubs(
            monkeypatch,
            per_session,
            _screen_rows((True, False, True), (0.02, 0.0, 0.03)),
            [0, 2],
        )
        for fold_index in range(3):
            enc.run_fold(
                unit, fold_index, settings, "/data", str(tmp_path), lambda _text: None
            )
        _combine_stubs(
            monkeypatch, calls, per_session, transfer_p=0.5, transfer_slope=0.2
        )

        result = enc.combine(unit, settings, "/data", str(tmp_path), lambda _text: None)

        assert result["representative_fold"] == 2
        assert result["representative_session"] == "s2"
        assert result["quiet_score"] == pytest.approx(0.25)
        assert result["quiet_score_fold_average"] == pytest.approx(0.25)
        assert result["no_model"] is False
        fold_results, _per, scored_ids, frames_by_session, passed_settings = calls[
            "combine_folds"
        ][0][0][:5]
        assert scored_ids == ["s0", "s1"]
        assert set(frames_by_session) == {"s0", "s1"}
        assert passed_settings is settings
        assert all(
            "session_id" not in entry and "frames" not in entry
            for entry in fold_results
        )
        assert all("pointer" in entry and "gaps" in entry for entry in fold_results)
        band_args = calls["filter_band"][0][0]
        assert band_args[1] == ["s0", "s1"]
        assert band_args[2] == [0, 2]
        assert band_args[3] == N_LAGS
        assert band_args[5] is settings["kinematic_encoding"]
        assert band_args[6] == 4
        sta_args = calls["spike_triggered_average"][0][0]
        assert sta_args[1:] == (
            ["s0", "s1"],
            [0, 2],
            N_LAGS,
            settings["kinematic_encoding"]["chunk_rows"],
        )
        assert [entry["feature"] for entry in result["filter_descriptors"]] == [
            "f0",
            "f2",
        ]

        stored = read_unit_artifact(str(tmp_path), unit["unit_uid"])
        section = stored["kinematic_encoding"]
        assert stored["identity"] == unit
        assert section["transfer_p"] == 0.5
        assert section["transfer_session_ids"] == ["s0", "s1"]
        assert section["selected"] == ["f0", "f2"]
        assert len(section["filters_per_fold"]) == 3
        assert section["filters_per_fold"][1]["test_session"] == "s1"
        assert section["filters_per_fold"][1]["filter"].shape == (2, N_LAGS)
        assert section["quiet_score_per_fold"] == pytest.approx([0.3, 0.2, 0.25])

    def test_a_missing_fold_file_is_refused(self, tmp_path):
        """
        Description
        -----------
        ``combine`` needs every fold; a missing file must raise ``FileNotFoundError`` naming it rather
        than pool a partial rotation.

        Parameters
        ----------
        tmp_path (pathlib.Path)
            An output directory holding only fold 0.

        Returns
        -------
        None
        """
        unit = _unit(["s0", "s1"], ["s0", "s1"])
        enc.write_fold_artifact(
            _toy_fold_artifact(0, "s0", 0.1, 0.5, 20),
            unit,
            0,
            str(tmp_path),
            lambda _text: None,
        )
        with pytest.raises(FileNotFoundError, match="fold1"):
            enc.combine(unit, _settings(), "/data", str(tmp_path), lambda _text: None)

    def test_an_even_fold_count_with_a_no_model_fold_breaks_the_tie_on_vocal_frames(
        self, tmp_path, monkeypatch
    ):
        """
        Description
        -----------
        Folds 0 and 2 carry models, fold 1 has none. The two usable folds straddle the median, so the
        representative is the one with MORE held-out vocal frames (fold 2, 80 against 50). The roster
        must keep fold 1 in position, marked as absent, so later folds stay paired with their sessions.

        Parameters
        ----------
        tmp_path (pathlib.Path)
            Output directory for the fold files and the unit file.
        monkeypatch (pytest.MonkeyPatch)
            Installs the stubs.

        Returns
        -------
        None
        """
        settings = _settings()
        unit = _unit(["s0", "s1", "s2"], ["s0", "s1"])
        folds = [
            _toy_fold_artifact(0, "s0", 0.3, 0.001, 50),
            _no_model_fold_artifact(1, "s1"),
            _toy_fold_artifact(2, "s2", 0.1, 0.002, 80),
        ]
        for fold_index, artifact in enumerate(folds):
            enc.write_fold_artifact(
                artifact, unit, fold_index, str(tmp_path), lambda _text: None
            )
        calls = collections.defaultdict(list)
        _combine_stubs(
            monkeypatch, calls, _per_session({"s0": 300}, {"s0": 30}), 0.5, 0.2
        )

        result = enc.combine(unit, settings, "/data", str(tmp_path), lambda _text: None)

        assert result["representative_fold"] == 2
        assert result["quiet_p"] == pytest.approx(0.002)
        assert result["quiet_score_fold_average"] == pytest.approx(0.2)
        assert result["no_model_per_fold"] == [False, True, False]
        assert result["quiet_p_per_fold"] == [0.001, 1.0, 0.002]
        assert np.isnan(result["fit_seconds_per_fold"][1])
        assert result["filters_per_fold"][1]["filter"] is None
        assert result["n_iter_per_fold"] == [10, None, 12]
        # fold 2's model, carried onto the vocal sessions: its eta is filled with its own fold index
        assert all(
            np.all(entry["eta"] == 2.0) for entry in calls["combine_folds"][0][0][0]
        )

    def test_a_unit_where_no_fold_has_a_model_is_written_as_failing_the_screen(
        self, tmp_path, monkeypatch
    ):
        """
        Description
        -----------
        A unit whose every fold has no model must be WRITTEN (quiet p 1.0, transfer p NaN, ``no_model``)
        with its per-fold screens, so it never becomes a silent missing value in the cohort table; the
        transfer is never pooled.

        Parameters
        ----------
        tmp_path (pathlib.Path)
            Output directory for the fold files and the unit file.
        monkeypatch (pytest.MonkeyPatch)
            Installs the stubs.

        Returns
        -------
        None
        """
        unit = _unit(["s0", "s1"], ["s0", "s1"])
        for fold_index, session_id in enumerate(["s0", "s1"]):
            enc.write_fold_artifact(
                _no_model_fold_artifact(fold_index, session_id),
                unit,
                fold_index,
                str(tmp_path),
                lambda _text: None,
            )
        calls = collections.defaultdict(list)
        _combine_stubs(
            monkeypatch, calls, _per_session({"s0": 300}, {"s0": 30}), 0.5, 0.2
        )

        result = enc.combine(
            unit, _settings(), "/data", str(tmp_path), lambda _text: None
        )

        assert "combine_folds" not in calls
        section = read_unit_artifact(str(tmp_path), unit["unit_uid"])[
            "kinematic_encoding"
        ]
        assert section["no_model"] is True
        assert section["quiet_p"] == 1.0
        assert np.isnan(section["transfer_p"])
        assert section["n_lags"] == N_LAGS
        assert section["screen_per_fold"] == [[{"feature": 1, "survived": False}]] * 2
        assert result["no_model_per_fold"] == [True, True]

    def test_best_screened_folds_are_pooled_for_description_only(
        self, tmp_path, monkeypatch
    ):
        """
        Description
        -----------
        When every fold's model is only its best SCREENED feature, the folds are pooled exactly like
        selected models -- but the written result must still say ``no_model`` and
        ``best_screened_only``, and the printed reading must say the feature was rejected.

        Parameters
        ----------
        tmp_path (pathlib.Path)
            Output directory for the fold files and the unit file.
        monkeypatch (pytest.MonkeyPatch)
            Installs the stubs.

        Returns
        -------
        None
        """
        unit = _unit(["s0", "s1", "s2"], ["s0", "s1"])
        for fold_index, (session_id, score) in enumerate(
            zip(["s0", "s1", "s2"], [0.1, 0.3, 0.2], strict=True)
        ):
            artifact = _toy_fold_artifact(
                fold_index,
                session_id,
                score,
                0.001,
                30,
                no_model=True,
                best_screened=True,
            )
            enc.write_fold_artifact(
                artifact, unit, fold_index, str(tmp_path), lambda _text: None
            )
        calls = collections.defaultdict(list)
        _combine_stubs(
            monkeypatch, calls, _per_session({"s0": 300}, {"s0": 30}), 0.001, 0.2
        )
        messages = []

        result = enc.combine(unit, _settings(), "/data", str(tmp_path), messages.append)

        assert result["representative_fold"] == 2
        assert result["no_model"] is True
        assert result["best_screened_only"] is True
        assert result["no_model_per_fold"] == [False, False, False]
        assert any("NO FOLD SELECTED A MODEL" in text for text in messages)
        assert any("best SCREENED feature only" in text for text in messages)

    def test_the_filter_band_is_skipped_when_its_setting_is_off(
        self, tmp_path, monkeypatch
    ):
        """
        Description
        -----------
        ``kinematic_encoding.filter_band.compute = false`` must skip the band's refits entirely and store
        ``None`` in its place.

        Parameters
        ----------
        tmp_path (pathlib.Path)
            Output directory for the fold files and the unit file.
        monkeypatch (pytest.MonkeyPatch)
            Installs the stubs.

        Returns
        -------
        None
        """
        settings = _settings()
        settings["kinematic_encoding"]["filter_band"]["compute"] = False
        unit = _unit(["s0"], ["s0"])
        enc.write_fold_artifact(
            _toy_fold_artifact(0, "s0", 0.1, 0.5, 20),
            unit,
            0,
            str(tmp_path),
            lambda _text: None,
        )
        calls = collections.defaultdict(list)
        _combine_stubs(
            monkeypatch, calls, _per_session({"s0": 300}, {"s0": 30}), 0.5, 0.2
        )
        result = enc.combine(unit, settings, "/data", str(tmp_path), lambda _text: None)
        assert "filter_band" not in calls
        assert result["filter_band"] is None

    @pytest.mark.parametrize(
        ("quiet_p", "transfer_p", "transfer_slope", "reading"),
        [
            (0.001, 0.001, 0.3, "transformation candidate"),
            (0.001, 0.001, -0.3, "regime-switching"),
            (0.001, 0.5, 0.3, "purely kinematic"),
            (0.5, 0.001, 0.3, "not tuned on quiet"),
        ],
    )
    def test_the_printed_reading_follows_the_fdr_level_from_settings(
        self, tmp_path, monkeypatch, quiet_p, transfer_p, transfer_slope, reading
    ):
        """
        Description
        -----------
        The uncorrected reading printed for a single unit compares both p-values to
        ``significance.fdr_q`` (0.01) and the transfer slope to zero; each of its four outcomes must be
        reachable, and none of them may be persisted.

        Parameters
        ----------
        tmp_path (pathlib.Path)
            Output directory for the fold file and the unit file.
        monkeypatch (pytest.MonkeyPatch)
            Installs the stubs.
        quiet_p (float)
            The representative fold's quiet p-value.
        transfer_p (float)
            The pooled transfer p-value.
        transfer_slope (float)
            The pooled transfer slope.
        reading (str)
            The expected start of the printed reading.

        Returns
        -------
        None
        """
        unit = _unit(["s0"], ["s0"])
        enc.write_fold_artifact(
            _toy_fold_artifact(0, "s0", 0.1, quiet_p, 20),
            unit,
            0,
            str(tmp_path),
            lambda _text: None,
        )
        calls = collections.defaultdict(list)
        _combine_stubs(
            monkeypatch,
            calls,
            _per_session({"s0": 300}, {"s0": 30}),
            transfer_p,
            transfer_slope,
        )
        messages = []
        result = enc.combine(unit, _settings(), "/data", str(tmp_path), messages.append)
        assert any(
            f"UNCORRECTED READING (not persisted): {reading}" in text
            for text in messages
        )
        assert "reading" not in result


class TestRunSingle:
    """Select once, fit once: the held-out session, the inner split, and the written section."""

    @pytest.mark.parametrize(
        ("n_shuffles", "transfer_p", "transfer_slope", "reading"),
        [
            (199, 0.001, 0.3, "transformation candidate"),
            (199, 0.001, -0.3, "regime-switching"),
            (199, 0.5, 0.3, "purely kinematic"),
            (9, 0.5, 0.3, "not tuned on quiet"),
        ],
    )
    def test_the_session_with_most_anchors_other_than_the_sparsest_is_held_out(
        self, tmp_path, monkeypatch, n_shuffles, transfer_p, transfer_slope, reading
    ):
        """
        Description
        -----------
        ``s0`` has the most quiet anchors but the fewest spikes, so it is never the scoring session; of
        the rest, ``s2`` has the most anchors and is held out. Selection and the fit run on ``s0`` and
        ``s1``, the generator is seeded from the unit id, the quiet null takes ``null.n_shuffles`` draws,
        only the vocal-eligible sessions (``s0``, ``s2``) are scored for the transfer, and the section is
        merged into the unit file. The quiet p sits at the floor of the draws taken, so 199 draws make it
        significant at ``fdr_q`` and 9 do not, which reaches every printed reading.

        Parameters
        ----------
        tmp_path (pathlib.Path)
            Output directory for the unit file.
        monkeypatch (pytest.MonkeyPatch)
            Installs the stubs.
        n_shuffles (int)
            ``null.n_shuffles``.
        transfer_p (float)
            The pooled transfer p-value.
        transfer_slope (float)
            The pooled transfer slope.
        reading (str)
            The expected start of the printed reading.

        Returns
        -------
        None
        """
        settings = _settings()
        settings["null"]["n_shuffles"] = n_shuffles
        per_session = _per_session(
            {"s0": 400, "s1": 200, "s2": 300}, {"s0": 10, "s1": 40, "s2": 30}
        )
        calls = _install_fold_stubs(
            monkeypatch,
            per_session,
            _screen_rows((True, False, True), (0.02, 0.0, 0.03)),
            [0, 2],
        )
        _combine_stubs(monkeypatch, calls, per_session, transfer_p, transfer_slope)
        unit = _unit(["s0", "s1", "s2"], ["s0", "s2"])
        messages = []

        artifact = enc.run_single(
            unit, settings, "/data", str(tmp_path), messages.append
        )

        assert artifact["held_out"] == "s2"
        assert artifact["sparsest_session"] == "s0"
        assert artifact["fit_sessions"] == ["s0", "s1"]
        assert artifact["inner_split"] == "sessions"
        assert artifact["spike_counts"] == {"s0": 10, "s1": 40, "s2": 30}
        screen_args = calls["screen_features"][0][0]
        assert screen_args[1] == ["s0", "s1"]
        expected_rng = np.random.default_rng(
            crc32(unit["unit_id"].encode()) & 0xFFFFFFFF
        )
        assert screen_args[3].bit_generator.state == expected_rng.bit_generator.state
        assert calls["fit_quiet_model"][0][0][1] == ["s0", "s1"]
        assert calls["frozen_null_scores"][0][0][5] == n_shuffles
        assert [call[0][1] for call in calls["focal_vocal_frames"]] == ["s0", "s2"]
        vocal_results, _per, vocal_ids, frames_by_session, _settings_passed = calls[
            "combine_folds"
        ][0][0][:5]
        assert vocal_ids == ["s0", "s2"]
        assert set(frames_by_session) == {"s0", "s2"}
        assert all("pointer" in entry and "gaps" in entry for entry in vocal_results)
        assert artifact["quiet_score"] == pytest.approx(0.3)
        assert artifact["quiet_p"] == pytest.approx(1.0 / (n_shuffles + 1.0))
        assert artifact["transfer_p"] == transfer_p
        assert artifact["n_lags"] == N_LAGS
        assert artifact["fps"] == FPS
        assert artifact["filter"].shape == (2 * N_LAGS,)
        assert any(
            f"UNCORRECTED READING (not persisted): {reading}" in text
            for text in messages
        )
        section = read_unit_artifact(str(tmp_path), unit["unit_uid"])[
            "kinematic_encoding"
        ]
        assert section["held_out"] == "s2"
        assert section["selected"] == ["f0", "f2"]

    def test_one_fit_session_screens_on_quiet_blocks(self, tmp_path, monkeypatch):
        """
        Description
        -----------
        A two-session unit leaves one fit session, so the inner split is that session's quiet blocks,
        separated by ``ceil(4 s * 10 fps / 2) = 20`` frames at each end; the final fit still uses the
        whole session, and the non-converged fit is reported.

        Parameters
        ----------
        tmp_path (pathlib.Path)
            Output directory for the unit file.
        monkeypatch (pytest.MonkeyPatch)
            Installs the stubs.

        Returns
        -------
        None
        """
        settings = _settings()
        settings["null"]["n_shuffles"] = 5
        per_session = _per_session({"s0": 300, "s1": 200}, {"s0": 30, "s1": 20})
        blocks = {"s1#0": per_session["s1"], "s1#1": per_session["s1"]}
        calls = _install_fold_stubs(
            monkeypatch,
            per_session,
            _screen_rows((True, False, False), (0.02, 0.0, 0.0)),
            [0],
            blocks=blocks,
            converged=False,
        )
        _combine_stubs(monkeypatch, calls, per_session, 0.5, 0.2)
        messages = []
        artifact = enc.run_single(
            _unit(["s0", "s1"], ["s0", "s1"]),
            settings,
            "/data",
            str(tmp_path),
            messages.append,
        )
        # s1 has fewer spikes than s0, so s0 is held out and s1 is the one fit session
        assert artifact["held_out"] == "s0"
        assert artifact["inner_split"] == "blocks"
        assert calls["quiet_block_sessions"][0][0][1:] == (
            "s1",
            settings["data_sufficiency"]["single_session_inner_split_blocks"],
            20,
        )
        assert calls["screen_features"][0][0][1] == ["s1#0", "s1#1"]
        assert calls["fit_quiet_model"][0][0][1] == ["s1"]
        assert artifact["final_fit_converged"] is False
        assert any("DID NOT CONVERGE" in text for text in messages)

    def test_one_fit_session_without_two_blocks_has_no_model(
        self, tmp_path, monkeypatch
    ):
        """
        Description
        -----------
        With fewer than two usable quiet blocks there is no inner split, so the unit returns the
        documented no-model result (quiet p 1.0, transfer p NaN) without screening, and WRITES it to
        the unit file so the unit is not a silent gap in the cohort table.

        Parameters
        ----------
        tmp_path (pathlib.Path)
            Output directory for the unit file.
        monkeypatch (pytest.MonkeyPatch)
            Installs the stubs.

        Returns
        -------
        None
        """
        per_session = _per_session({"s0": 300, "s1": 200}, {"s0": 30, "s1": 20})
        calls = _install_fold_stubs(monkeypatch, per_session, [], [0])
        unit = _unit(["s0", "s1"], ["s0", "s1"])
        result = enc.run_single(
            unit, _settings(), "/data", str(tmp_path), lambda _text: None
        )
        assert result["no_model"] is True
        assert result["quiet_p"] == 1.0
        assert np.isnan(result["transfer_p"])
        assert result["unit_id"] == unit["unit_id"]
        assert "screen_features" not in calls
        section = read_unit_artifact(str(tmp_path), unit["unit_uid"])[
            "kinematic_encoding"
        ]
        assert section["no_model"] is True
        assert section["inner_split"] == "blocks"
        assert section["screen"] is None

    def test_no_survivor_returns_no_model(self, tmp_path, monkeypatch):
        """
        Description
        -----------
        With nothing surviving the screen, ``run_single`` returns the no-model result and never selects
        or fits.

        Parameters
        ----------
        tmp_path (pathlib.Path)
            Output directory for the unit file.
        monkeypatch (pytest.MonkeyPatch)
            Installs the stubs.

        Returns
        -------
        None
        """
        per_session = _per_session(
            {"s0": 300, "s1": 200, "s2": 250}, {"s0": 30, "s1": 20, "s2": 25}
        )
        calls = _install_fold_stubs(
            monkeypatch, per_session, _screen_rows((False,) * 3, (0.0,) * 3), [0]
        )
        result = enc.run_single(
            _unit(["s0", "s1", "s2"], ["s0"]),
            _settings(),
            "/data",
            str(tmp_path),
            lambda _text: None,
        )
        assert result["no_model"] is True
        assert "forward_select" not in calls
        assert "fit_quiet_model" not in calls

    def test_a_no_model_single_result_is_written_to_the_unit_file(
        self, tmp_path, monkeypatch
    ):
        """
        Description
        -----------
        ``combine`` writes a unit that fails at the screen (quiet p 1.0, transfer p NaN, ``no_model``) so
        the untuned units -- a large share of the cohort -- appear as results rather than as gaps.
        ``run_single`` must do the same on both of its no-model paths: here nothing survives the
        screen, and the written section carries the no-model fields and the screen table.

        Parameters
        ----------
        tmp_path (pathlib.Path)
            Output directory for the unit file.
        monkeypatch (pytest.MonkeyPatch)
            Installs the stubs.

        Returns
        -------
        None
        """
        per_session = _per_session(
            {"s0": 300, "s1": 200, "s2": 250}, {"s0": 30, "s1": 20, "s2": 25}
        )
        _install_fold_stubs(
            monkeypatch, per_session, _screen_rows((False,) * 3, (0.0,) * 3), [0]
        )
        unit = _unit(["s0", "s1", "s2"], ["s0"])
        result = enc.run_single(
            unit, _settings(), "/data", str(tmp_path), lambda _text: None
        )
        section = read_unit_artifact(str(tmp_path), unit["unit_uid"])[
            "kinematic_encoding"
        ]
        assert section["no_model"] is True
        assert section["quiet_p"] == 1.0
        assert np.isnan(section["transfer_p"])
        assert section["screen"] == result["screen"]
        assert len(section["screen"]) == 3
        assert section["held_out"] == result["held_out"]
        assert section["inner_split"] == "sessions"


class TestCommandLine:
    """``main`` parses the job's arguments and ``dispatch`` routes them to the right step."""

    def _patch(self, monkeypatch, tmp_path, threshold: int) -> tuple:
        """
        Description
        -----------
        Writes a settings file whose vocal gate threshold and vocal emitter differ from the shipped
        ones, stubs the per-session USV counter (``s0``: 25, ``s1``: 15) and the three run steps with
        recording stubs, and returns where they record.

        Parameters
        ----------
        monkeypatch (pytest.MonkeyPatch)
            Installs the stubs.
        tmp_path (pathlib.Path)
            Directory for the settings file.
        threshold (int)
            ``data_sufficiency.min_emitter_usvs_per_session`` written to the settings file.

        Returns
        -------
        settings_path, calls (tuple)
            Path of the written settings file, and the recording dict.
        """
        settings = _settings()
        settings["data_sufficiency"]["min_emitter_usvs_per_session"] = threshold
        settings["vocalization_settings"]["vocal_emitter"] = "partner"
        settings_path = tmp_path / "settings.json"
        settings_path.write_text(json.dumps(settings))
        calls = collections.defaultdict(list)
        monkeypatch.setattr(
            enc,
            "emitter_usv_counts",
            _recording_stub(calls, "emitter_usv_counts", {"s0": 25, "s1": 15}),
        )
        for name in ("run_fold", "combine", "run_single"):
            monkeypatch.setattr(enc, name, _recording_stub(calls, name, {}))
        return str(settings_path), calls

    def _argv(self, settings_path: str, *mode: str) -> list:
        """
        Description
        -----------
        A full command line for unit ``imec0_cl0001_ch001_good`` of mouse ``m1`` over sessions ``s0`` and
        ``s1``, followed by the given mode flags.

        Parameters
        ----------
        settings_path (str)
            Value of ``--settings-path``.
        *mode (str)
            The mode flags, e.g. ``"--fold", "1"``.

        Returns
        -------
        argv (list)
            ``sys.argv`` for the dispatcher.
        """
        return [
            "main_neural_encoding_dispatcher",
            "--unit-id",
            "imec0_cl0001_ch001_good",
            "--mouse-id",
            "m1",
            "--rec-date",
            "20250101",
            "--sessions",
            "s0",
            "s1",
            "--data-root",
            "/data",
            "--output-directory",
            "/out",
            "--settings-path",
            settings_path,
            *mode,
        ]

    def test_a_fold_job_builds_the_unit_record_from_the_settings_file_and_runs_that_fold(
        self, tmp_path, monkeypatch, capsys
    ):
        """
        Description
        -----------
        ``--fold 1`` must run fold 1 with the settings READ from ``--settings-path``: the vocal gate at
        10 USVs keeps both sessions (the shipped 20 would drop ``s1``), the emitter key reaches the USV
        counter, and the unit record carries the parsed identity. Success exits 0 and prints the
        duration.

        Parameters
        ----------
        tmp_path (pathlib.Path)
            Directory for the settings file.
        monkeypatch (pytest.MonkeyPatch)
            Installs the stubs and ``sys.argv``.
        capsys (pytest.CaptureFixture)
            Captures the exit line.

        Returns
        -------
        None
        """
        settings_path, calls = self._patch(monkeypatch, tmp_path, threshold=10)
        monkeypatch.setattr(sys, "argv", self._argv(settings_path, "--fold", "1"))
        assert enc.main() == 0
        unit, fold, settings, data_root, output_directory = calls["run_fold"][0][0]
        assert fold == 1
        assert (data_root, output_directory) == ("/data", "/out")
        assert settings["data_sufficiency"]["min_emitter_usvs_per_session"] == 10
        assert unit["vocal_sessions"] == ["s0", "s1"]
        assert unit["unit_uid"] == "m1_20250101_imec0_cl0001_ch001_good"
        assert unit["rec_date"] == 20250101
        counter_args = calls["emitter_usv_counts"][0][0]
        assert counter_args[:3] == (["s0", "s1"], "/data", {"s0": "m1", "s1": "m1"})
        assert counter_args[3] == "partner"
        assert "combine" not in calls
        assert "run_single" not in calls
        assert "finished in" in capsys.readouterr().out

    @pytest.mark.parametrize(
        ("flag", "step"), [("--combine", "combine"), ("--single", "run_single")]
    )
    def test_the_combine_and_single_flags_route_to_their_steps(
        self, tmp_path, monkeypatch, flag, step
    ):
        """
        Description
        -----------
        ``--combine`` and ``--single`` must each reach their own step, and only it; the vocal gate at the
        shipped-like 20 keeps only ``s0``.

        Parameters
        ----------
        tmp_path (pathlib.Path)
            Directory for the settings file.
        monkeypatch (pytest.MonkeyPatch)
            Installs the stubs and ``sys.argv``.
        flag (str)
            The mode flag.
        step (str)
            The step it must reach.

        Returns
        -------
        None
        """
        settings_path, calls = self._patch(monkeypatch, tmp_path, threshold=20)
        monkeypatch.setattr(sys, "argv", self._argv(settings_path, flag))
        assert enc.main() == 0
        assert set(calls) == {"emitter_usv_counts", step}
        unit, _settings_passed, data_root, output_directory = calls[step][0][0]
        assert unit["vocal_sessions"] == ["s0"]
        assert (data_root, output_directory) == ("/data", "/out")

    @pytest.mark.parametrize(
        "mode",
        [
            (),
            ("--fold", "1", "--combine"),
            ("--single", "--combine"),
            ("--fold", "one"),
        ],
    )
    def test_a_missing_or_conflicting_mode_is_refused(
        self, tmp_path, monkeypatch, mode
    ):
        """
        Description
        -----------
        Exactly one of ``--fold K`` / ``--combine`` / ``--single`` is required, and the fold must be an
        integer; anything else is a usage error (exit 2) that reaches no step.

        Parameters
        ----------
        tmp_path (pathlib.Path)
            Directory for the settings file.
        monkeypatch (pytest.MonkeyPatch)
            Installs the stubs and ``sys.argv``.
        mode (tuple)
            The mode flags.

        Returns
        -------
        None
        """
        settings_path, calls = self._patch(monkeypatch, tmp_path, threshold=20)
        monkeypatch.setattr(sys, "argv", self._argv(settings_path, *mode))
        with pytest.raises(SystemExit) as raised:
            enc.main()
        assert raised.value.code == 2
        assert not calls

    def test_a_missing_required_argument_is_refused(self, monkeypatch):
        """
        Description
        -----------
        The unit's identity is required: a command line without ``--unit-id`` is a usage error.

        Parameters
        ----------
        monkeypatch (pytest.MonkeyPatch)
            Installs ``sys.argv``.

        Returns
        -------
        None
        """
        argv = self._argv("unused", "--combine")
        del argv[1:3]
        monkeypatch.setattr(sys, "argv", argv)
        with pytest.raises(SystemExit) as raised:
            enc.main()
        assert raised.value.code == 2

    def test_a_failing_step_exits_one_with_its_traceback(
        self, tmp_path, monkeypatch, capsys
    ):
        """
        Description
        -----------
        An exception inside a step must not escape: ``dispatch`` prints the full traceback (so a cluster
        node leaves something diagnosable) and returns 1, without the success line.

        Parameters
        ----------
        tmp_path (pathlib.Path)
            Directory for the settings file.
        monkeypatch (pytest.MonkeyPatch)
            Installs the stubs and ``sys.argv``.
        capsys (pytest.CaptureFixture)
            Captures the traceback.

        Returns
        -------
        None
        """
        settings_path, _calls = self._patch(monkeypatch, tmp_path, threshold=20)

        def explode(*_args, **_kwargs):
            """
            Description
            -----------
            Stand-in for ``combine`` that fails as a starved cluster node would.

            Parameters
            ----------
            *_args
                Ignored positional arguments.
            **_kwargs
                Ignored keyword arguments.

            Returns
            -------
            Never returns; always raises ``RuntimeError``.
            """
            msg = "node lost its scratch disk"
            raise RuntimeError(msg)

        monkeypatch.setattr(enc, "combine", explode)
        monkeypatch.setattr(sys, "argv", self._argv(settings_path, "--combine"))
        assert enc.main() == 1
        captured = capsys.readouterr()
        assert "node lost its scratch disk" in captured.err
        assert "finished in" not in captured.out

    def test_the_shipped_settings_are_the_default(self):
        """
        Description
        -----------
        Without ``--settings-path`` the dispatcher must read the settings file shipped with the package.

        Parameters
        ----------
        None.

        Returns
        -------
        None
        """
        assert enc.load_settings(None) == _settings()
