"""
@author: bartulem
Unit tests for ``usv_playpen.neural_modeling.main_neural_vocal_dispatcher``.

Coverage: the job that runs vocal occurrence, vocalization identity, nested vocal-manifold position decoding
and vocal gating for one unit. Its own logic is thin, so these
tests pin the properties that would fail SILENTLY rather than loudly -- that nulls split across spawned
worker processes come back identical to the package's serial nulls, that escalation continues a draw
sequence instead of repeating it, that the written sections follow the shipped design (held-out
sessions by name, no gating decisions), and that vocal gating's on/off switch reads its threshold from settings
and skips an untuned unit without loading its kinematics.

Each analysis step is also run end to end IN PROCESS: ``spawn_pool`` is replaced by a pool that installs
the worker state through ``initialise_worker`` and maps the chunk functions in this process, the
kinematic and JAX-fitted pieces are replaced by recording stubs, and the tests assert the settings values
each null receives, that the null equals the package's serial null, and the section written to the unit
file. ``run_vocal``'s ordering and p-value hand-off and the command line (``dispatch`` / ``main``) are
covered with the steps stubbed.
"""

from __future__ import annotations

import collections
import copy
import json
import pathlib
import sys
import warnings

import numpy as np
import pytest

# The dispatcher imports the nested decoding's estimator, whose JAX/Optax parent sets a flag JAX has deprecated;
# that one-time DeprecationWarning fires at import, where `filterwarnings = ["error"]` would make it a
# collection error. Suppressed just for these imports, as in `test_neural_nested_decoding.py`.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from usv_playpen.neural_modeling import main_neural_encoding_dispatcher as enc
    from usv_playpen.neural_modeling import main_neural_vocal_dispatcher as vocal
    from usv_playpen.neural_modeling.neural_artifacts import (
        decision_key_paths,
        read_unit_artifact,
        write_unit_section,
    )
    from usv_playpen.neural_modeling.neural_category_decoding import (
        category_codes,
        category_context,
        category_gain,
    )
    from usv_playpen.neural_modeling.neural_nested_decoding import (
        behaviour_priors,
        nested_settings,
        posterior_nested_gain,
    )
    from usv_playpen.neural_modeling.neural_significance import (
        permutation_null,
        within_session_permutation,
    )
    from usv_playpen.neural_modeling.neural_vocal_decoding import (
        decode_gain,
        decoding_context,
    )
    from usv_playpen.neural_modeling.neural_vocal_occurrence import (
        counts_in_windows,
        occurrence_context,
        vocal_occurrence_shift_null,
    )

SETTINGS_PATH = (pathlib.Path(__file__).resolve().parents[2] / "src" / "usv_playpen"
                 / "_parameter_settings" / "neural_modeling_settings.json")
UNIT = {"unit_id": "imec1_cl0000_ch000_good", "unit_uid": "m_20250101_imec1_cl0000_ch000_good",
        "mouse_id": "m", "rec_date": 20250101, "courtship_sessions": ["s0", "s1"],
        "vocal_sessions": ["s0", "s1"]}


def _settings() -> dict:
    """
    Description
    -----------
    Loads the shipped neural-modeling settings fresh from disk, so a test that mutates the dictionary
    cannot leak the change into another test.

    Parameters
    ----------
    None.

    Returns
    -------
    settings (dict)
        The parsed ``neural_modeling_settings.json``.
    """
    with SETTINGS_PATH.open() as handle:
        return json.load(handle)


def _quiet(*_args, **_kwargs) -> None:
    """
    Description
    -----------
    Silent stand-in for the dispatcher's ``message_output`` printer, so the tests produce no output.

    Parameters
    ----------
    *_args
        Ignored positional arguments.
    **_kwargs
        Ignored keyword arguments.

    Returns
    -------
    None
    """


class TestParallelNullsAreExact:
    """A null split across workers must be the SAME null, value for value, or a cohort run would report
    p-values no serial run could reproduce."""

    def test_the_vocalization_identity_null_across_workers_equals_the_serial_permutation_null(self):
        """Five vocalization-identity null draws computed in two spawned workers must equal, value for value,
        the serial ``permutation_null`` at the same seed."""
        generator = np.random.default_rng(3)
        n_sessions, per_session = 3, 120
        positions = generator.random((n_sessions * per_session, 2))
        sessions = np.repeat(np.arange(n_sessions), per_session)
        shape = np.cos(2 * np.pi * positions[:, 0]) + np.cos(2 * np.pi * positions[:, 1])
        counts = generator.poisson(np.exp(1.4 * shape - 1.4 * shape.max() + np.log(3.0))).astype(float)
        decoding = copy.deepcopy(_settings()["vocal_decoding"])
        decoding["tuning_surface"].update({"grid_n": 24, "kde_bandwidth": 0.10})
        folds = decoding_context(positions, sessions, decoding, _settings()["vocalization_settings"]["torus_period"])

        serial = permutation_null(lambda permutation: decode_gain(counts, folds, decoding, permutation)["gain"],
                                  sessions, 5, 11)
        state = {"target": "manifold", "counts": counts, "folds": folds, "decoding": decoding,
                 "session_index": sessions, "seed": 11}
        with vocal.spawn_pool(2, state) as pool:
            parallel = vocal.parallel_draw_more(pool, vocal.vocalization_identity_null_chunk, 2)(5)
        np.testing.assert_array_equal(parallel, serial)


class TestTheWrittenSections:
    """The written sections name held-out sessions, carry the behaviour control's identity, and store no
    gating decisions."""

    def test_vocal_gating_writes_statistics_and_p_values_but_no_decisions(self, tmp_path):
        """Vocal gating's label, the borderline flag and the unit's best label are uncorrected by construction; the
        artifact writer refuses them, so the payload must not carry them at any depth."""
        draws = np.zeros(4)
        observed = {"f0": {"feature_main": -3.0, "vocal_main": 400.0, "interaction": 560.0,
                           "additive": 420.0, "baseline": 9000.0, "quiet_baseline": 700.0,
                           "interaction_minus_vocal": 160.0, "gating_sign": 1.0}}
        nulls = {"f0": {"interaction": draws, "vocal_main": draws, "feature_main": draws, "difference": draws}}
        verdicts = {"f0": {"label": "GATE", "p_interaction": 5e-4, "p_interaction_gt_vocal": 5e-4,
                           "p_feature_main_quiet": 0.4, "sigma_margin": 12.0, "borderline": False,
                           "bonferroni_bar": 5e-4}}
        unit_p = {"p_unit": 5e-4, "feature": "f0", "label": "GATE", "at_floor": True}
        payload = vocal.vocal_gating_payload(["f0", "f1"], observed, nulls, verdicts, unit_p,
                                       {"f1": "not_testable"}, 2000)
        assert decision_key_paths(payload) == []
        assert payload["per_feature"]["f1"] == {"tested": False, "reason": "not_testable"}
        assert payload["per_feature"]["f0"]["p_interaction"] == 5e-4
        assert payload["p_unit_feature"] == "f0"
        write_unit_section(str(tmp_path), UNIT, "vocal_gating", payload, _settings())

    def test_the_vocalization_identity_section_derives_its_counts_from_the_null_and_events(self):
        """The exceed count, mid-p, per-session event counts and event total must be derived from the null and
        the event arrays, so the stored section is internally consistent with the inputs."""
        identity = {"gain": 0.05}
        events = {"counts": np.array([1.0, 2.0, 3.0, 4.0]), "session_index": np.array([0, 0, 1, 1]),
                  "session_ids": ["a", "b"]}
        payload = vocal.vocalization_identity_payload(identity, np.array([0.0, 0.06, 0.01]), 0.5, False, events)
        assert payload["exceed_count"] == 1
        assert payload["mid_p"] == pytest.approx(1.5 / 4.0)
        assert payload["per_session_counts"] == {"a": 3, "b": 7}
        assert payload["n_events"] == 4


class TestTheGatingSwitch:
    """Vocal gating runs only on a unit whose vocal occurrence or vocalization identity shows tuning, at a
    threshold read from settings."""

    def test_the_threshold_is_a_setting(self):
        """Fixed at 0.01, the level every other per-unit threshold in vocal gating already uses."""
        assert _settings()["vocal_gating"]["vocal_tuning_alpha"] == 0.01

    def test_an_untuned_unit_is_skipped_without_loading_its_kinematics(self, tmp_path, monkeypatch):
        """A unit with no stored occurrence or identity tuning must be skipped with reason ``no_vocal_tuning``,
        written to its artifact, before its kinematics are ever loaded (which would be wasted work)."""
        def refuse(*_args, **_kwargs):
            """
            Description
            -----------
            Stand-in for ``assemble_unit_sessions`` that fails the test if the kinematics are loaded.

            Parameters
            ----------
            *_args
                Ignored positional arguments.
            **_kwargs
                Ignored keyword arguments.

            Returns
            -------
            Never returns; always raises ``AssertionError``.
            """
            msg = "an untuned unit must not load kinematics"
            raise AssertionError(msg)

        monkeypatch.setattr(vocal, "assemble_unit_sessions", refuse)
        payload = vocal.run_vocal_gating(UNIT, _settings(), "unused", str(tmp_path), 1, 0.2, 0.3, _quiet)
        assert payload == {"ran": False, "reason": "no_vocal_tuning"}
        assert read_unit_artifact(str(tmp_path), UNIT["unit_uid"])["vocal_gating"] == payload

    def test_the_threshold_is_read_not_assumed(self, tmp_path, monkeypatch):
        """At 0.5 the same p-values count as tuned, so the switch opens and reaches for the kinematics."""

        class Reached(Exception):
            """Sentinel raised when the dispatcher reaches the kinematics loader."""

        def reached(*_args, **_kwargs):
            """
            Description
            -----------
            Stand-in for ``assemble_unit_sessions`` that signals the switch opened by raising ``Reached``.

            Parameters
            ----------
            *_args
                Ignored positional arguments.
            **_kwargs
                Ignored keyword arguments.

            Returns
            -------
            Never returns; always raises ``Reached``.
            """
            raise Reached

        settings = _settings()
        settings["vocal_gating"]["vocal_tuning_alpha"] = 0.5
        monkeypatch.setattr(vocal, "assemble_unit_sessions", reached)
        with pytest.raises(Reached):
            vocal.run_vocal_gating(UNIT, settings, "unused", str(tmp_path), 1, 0.2, 0.3, _quiet)

    def test_vocal_gating_alone_refuses_without_occurrence_and_identity_results(self, tmp_path):
        """Requesting vocal gating alone on a unit with no occurrence or identity section must raise
        ``ValueError`` ("run vocal_occurrence first"), since the gating switch has nothing to read."""
        with pytest.raises(ValueError, match="run vocal_occurrence first"):
            vocal.run_vocal(UNIT, _settings(), "unused", str(tmp_path), 1, steps=("vocal_gating",),
                            message_output=_quiet)

    def test_a_threshold_too_few_draws_can_clear_is_refused(self, tmp_path):
        """Found on a real smoke run: at 20 draws every p-value floors at 1/21 = 0.048, above the 0.01
        threshold, so a strongly tuned unit read as untuned. Silently skipping every unit is refused."""
        write_unit_section(str(tmp_path), UNIT, "vocal_occurrence", {"p": 1 / 21, "n_shuffles": 20}, _settings())
        write_unit_section(str(tmp_path), UNIT, "vocalization_identity", {"p": 1 / 21, "n_permutations": 20},
                           _settings())
        with pytest.raises(ValueError, match="cannot fall below"):
            vocal.run_vocal(UNIT, _settings(), "unused", str(tmp_path), 1, steps=("vocal_gating",),
                            message_output=_quiet)

    def test_the_fewest_draws_that_can_clear_the_threshold_are_accepted(self):
        """1/(n+1) must be strictly below alpha: at 0.01 that is 100 draws, and 99 is refused."""
        vocal.check_tuning_threshold_is_reachable(0.01, {"vocal_occurrence": 100, "vocalization_identity": 1000})
        with pytest.raises(ValueError, match="at least 100 draws"):
            vocal.check_tuning_threshold_is_reachable(0.01, {"vocal_occurrence": 99})

    def test_an_unknown_step_is_refused(self, tmp_path):
        """An unrecognised step name must raise ``ValueError`` ("unknown vocal step") rather than be skipped,
        so a typo cannot silently run nothing."""
        with pytest.raises(ValueError, match="unknown vocal step"):
            vocal.run_vocal(UNIT, _settings(), "unused", str(tmp_path), 1, steps=("not_an_analysis",),
                            message_output=_quiet)


def _recording_stub(calls: dict, name: str, result):
    """
    Description
    -----------
    A stand-in for a dispatcher dependency that appends every invocation to ``calls[name]`` as an
    ``(args, kwargs)`` pair and returns ``result``. When ``result`` is callable it is called with that
    pair, so a stub's return value can depend on what it was handed.

    Parameters
    ----------
    calls (dict)
        A ``collections.defaultdict(list)`` collecting every invocation by stub name; its key order is
        the order in which the stubs were first reached.
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


class _InlinePool:
    """
    Description
    -----------
    An in-process stand-in for the spawned worker pool: entering it installs the worker state through
    the dispatcher's own ``initialise_worker``, and ``map`` runs the chunk function in this process, so
    the chunk functions are exercised (and measured) exactly as a worker would run them.

    Parameters
    ----------
    state (dict)
        The state the dispatcher handed to ``spawn_pool``.
    """

    def __init__(self, state: dict) -> None:
        """
        Description
        -----------
        Keeps the state to install on entry.

        Parameters
        ----------
        state (dict)
            The worker state.

        Returns
        -------
        None
        """
        self.state = state

    def __enter__(self) -> _InlinePool:
        """
        Description
        -----------
        Installs the state in the module-level worker dictionary, as the pool initializer would.

        Parameters
        ----------
        None.

        Returns
        -------
        pool (_InlinePool)
            This pool.
        """
        vocal.initialise_worker(self.state)
        return self

    def __exit__(self, *_exc_info) -> None:
        """
        Description
        -----------
        Leaves the context without suppressing exceptions.

        Parameters
        ----------
        *_exc_info
            The exception triple, ignored.

        Returns
        -------
        None
        """

    def map(self, function, *iterables) -> list:
        """
        Description
        -----------
        Runs ``function`` over the zipped iterables in this process, in order.

        Parameters
        ----------
        function (Callable)
            The chunk or feature function.
        *iterables
            Its argument sequences.

        Returns
        -------
        results (list)
            One result per argument tuple.
        """
        return list(map(function, *iterables))


def _inline_pools(monkeypatch) -> list:
    """
    Description
    -----------
    Replaces ``spawn_pool`` with a recorder that returns an ``_InlinePool``, and gives the module a fresh
    worker dictionary so no state leaks between tests.

    Parameters
    ----------
    monkeypatch (pytest.MonkeyPatch)
        Installs the replacement.

    Returns
    -------
    spawned (list)
        ``(n_workers, state)`` for every pool the dispatcher asked for.
    """
    spawned = []

    def fake_spawn(n_workers: int, state: dict) -> _InlinePool:
        """
        Description
        -----------
        Records the pool request and hands back an in-process pool.

        Parameters
        ----------
        n_workers (int)
            Worker count requested.
        state (dict)
            State to install.

        Returns
        -------
        pool (_InlinePool)
            The in-process pool.
        """
        spawned.append((n_workers, state))
        return _InlinePool(state)

    monkeypatch.setattr(vocal, "spawn_pool", fake_spawn)
    monkeypatch.setattr(vocal, "_WORKER", {})
    return spawned


def _vocal_events() -> dict:
    """
    Description
    -----------
    A synthetic ``assemble_unit_vocal_events`` output over three 300 s sessions (``sA``, ``sB``,
    ``sC``): 120 USVs per session with torus positions and spike counts tuned to position, and an
    occurrence window set of 40 prevocal windows and 80 quiet tiles per session, counted from a spike
    train that fires three extra spikes inside every prevocal window -- a strongly vocal-tuned unit.

    Parameters
    ----------
    None.

    Returns
    -------
    events (dict)
        ``counts``, ``positions``, ``session_index``, ``session_ids``, ``per_session`` (durations),
        ``occurrence_windows``, ``spike_seconds`` (the trains, keyed by session id, for the loader stub),
        ``vocalization_start`` and ``category_labels`` (four categories by the first torus coordinate).
    """
    generator = np.random.default_rng(3)
    n_sessions, per_session, width, duration = 3, 120, 0.1, 300.0
    positions = generator.random((n_sessions * per_session, 2))
    sessions = np.repeat(np.arange(n_sessions), per_session)
    shape = np.cos(2 * np.pi * positions[:, 0]) + np.cos(2 * np.pi * positions[:, 1])
    counts = generator.poisson(
        np.exp(1.4 * shape - 1.4 * shape.max() + np.log(3.0))
    ).astype(float)
    session_ids = ["sA", "sB", "sC"]
    window_counts, window_labels, window_sessions, window_edges, trains = (
        [],
        [],
        [],
        [],
        {},
    )
    for slot, session_id in enumerate(session_ids):
        vocal_edges = np.sort(generator.uniform(5.0, duration - 5.0, 40))
        quiet_edges = np.sort(generator.uniform(5.0, duration - 5.0, 80))
        bursts = (vocal_edges[:, None] + generator.uniform(0.0, width, (40, 3))).ravel()
        train = np.sort(np.concatenate([generator.uniform(0.0, duration, 300), bursts]))
        trains[session_id] = train
        edges = np.concatenate([vocal_edges, quiet_edges])
        window_counts.append(counts_in_windows(train, edges, width))
        window_labels.append(np.concatenate([np.ones(40), np.zeros(80)]))
        window_sessions.append(np.full(120, slot))
        window_edges.append(edges)
    return {
        "counts": counts,
        "positions": positions,
        "session_index": sessions,
        "session_ids": session_ids,
        "per_session": {
            session_id: {"duration_seconds": duration} for session_id in session_ids
        },
        "occurrence_windows": {
            "counts": np.concatenate(window_counts).astype(np.float64),
            "labels": np.concatenate(window_labels),
            "session_index": np.concatenate(window_sessions),
            "edges": np.concatenate(window_edges),
            "width": width,
        },
        "spike_seconds": trains,
        # Onsets in time order within each session, and an acoustic category per USV.
        "vocalization_start": np.concatenate([np.sort(generator.uniform(5.0, duration - 5.0, per_session))
                                              for _ in session_ids]),
        "category_labels": (np.floor(positions[:, 0] * 4.0) + 1.0).astype(np.float64),
    }


def _fast_decoding_settings() -> dict:
    """
    Description
    -----------
    The shipped settings with the decoder's grid coarsened (24 cells a side, bandwidth 0.10) so a decode
    takes a fraction of a second, and a seed that differs from the shipped one so a test can see it is
    read.

    Parameters
    ----------
    None.

    Returns
    -------
    settings (dict)
        The adjusted settings.
    """
    settings = _settings()
    settings["vocal_decoding"]["tuning_surface"].update(
        {"grid_n": 24, "kde_bandwidth": 0.10}
    )
    settings["null"]["shuffle_seed"] = 11
    return settings


class TestWorkerPlumbing:
    """The pieces that move work to the workers: state installation, chunking and draw continuation."""

    def test_installing_worker_state_replaces_what_was_there(self, monkeypatch):
        """
        Description
        -----------
        A reused worker must not keep a previous null's inputs: ``initialise_worker`` clears the module
        state before installing the new one.

        Parameters
        ----------
        monkeypatch (pytest.MonkeyPatch)
            Gives the module a fresh worker dictionary.

        Returns
        -------
        None
        """
        monkeypatch.setattr(vocal, "_WORKER", {"stale": 1})
        vocal.initialise_worker({"seed": 4})
        assert vocal._WORKER == {"seed": 4}

    def test_draws_are_handed_out_as_contiguous_global_ranges_that_continue(
        self, monkeypatch
    ):
        """
        Description
        -----------
        With one worker a 10-draw request is cut into chunks of ``ceil(10 / 4) = 3`` starting at global
        indices 0, 3, 6, 9; a request for zero draws returns an empty array WITHOUT advancing the counter;
        and the next request continues at index 10.

        Parameters
        ----------
        monkeypatch (pytest.MonkeyPatch)
            Gives the module a fresh worker dictionary.

        Returns
        -------
        None
        """
        monkeypatch.setattr(vocal, "_WORKER", {})
        requested = []

        def chunk(start: int, n_draws: int) -> np.ndarray:
            """
            Description
            -----------
            Records the chunk it was asked for and returns each draw's global index as its value.

            Parameters
            ----------
            start (int)
                Global index of the first draw.
            n_draws (int)
                Draws in the chunk.

            Returns
            -------
            values (np.ndarray)
                ``start, ..., start + n_draws - 1``.
            """
            requested.append((start, n_draws))
            return np.arange(start, start + n_draws, dtype=np.float64)

        with _InlinePool({}) as pool:
            draw_more = vocal.parallel_draw_more(pool, chunk, 1)
            first = draw_more(10)
            empty = draw_more(0)
            second = draw_more(5)
        assert requested[:4] == [(0, 3), (3, 3), (6, 3), (9, 1)]
        np.testing.assert_array_equal(first, np.arange(10))
        assert empty.size == 0
        np.testing.assert_array_equal(second, np.arange(10, 15))


class TestTheAnalysisSteps:
    """Each analysis end to end, in process: the state its null receives, the null itself, and the
    section it writes."""

    def test_vocal_occurrence_null_is_the_serial_shift_null_continued_by_escalation(
        self, tmp_path, monkeypatch
    ):
        """
        Description
        -----------
        The occurrence null must be built from the settings' seed, guard, occurrence block and
        cross-fitting blocks, with folds over the windows, spike trains loaded per session by name, and topped up by the escalation ladder from where it
        stopped -- so 6 draws plus 4 more equal the serial ``vocal_occurrence_shift_null`` over draws
        0..9. The section lands in the unit file with the sensitivity fractions computed from the windows.

        Parameters
        ----------
        tmp_path (pathlib.Path)
            Output directory for the unit file.
        monkeypatch (pytest.MonkeyPatch)
            Installs the in-process pool and the spike loader.

        Returns
        -------
        None
        """
        events = _vocal_events()
        settings = _settings()
        settings["null"].update(
            {"n_shuffles": 6, "shuffle_seed": 7, "shuffle_guard_seconds": 15.0}
        )
        settings["significance"]["escalation_ladder"] = [10]
        spawned = _inline_pools(monkeypatch)
        calls = collections.defaultdict(list)
        monkeypatch.setattr(
            vocal,
            "load_unit_spike_frames",
            _recording_stub(
                calls, "load", lambda call: (events["spike_seconds"][call[0][1]], None)
            ),
        )

        payload = vocal.run_vocal_occurrence(
            UNIT, events, settings, "/data", str(tmp_path), 3, _quiet
        )

        axis = settings["vocal_decoding"]["vocal_occurrence"]
        n_workers, state = spawned[0]
        assert n_workers == 3
        assert (state["seed"], state["guard_seconds"]) == (7, 15.0)
        assert state["axis"] is axis
        assert state["crossfit_blocks"] == settings["kinematic_encoding"]["calibration"]["n_blocks"]
        assert state["durations"] == {0: 300.0, 1: 300.0, 2: 300.0}
        assert [call[0] for call in calls["load"]] == [
            ("/data", session_id, UNIT["unit_id"])
            for session_id in events["session_ids"]
        ]
        trains = dict(
            enumerate(
                events["spike_seconds"][session_id]
                for session_id in events["session_ids"]
            )
        )
        windows = events["occurrence_windows"]
        folds = occurrence_context(windows["labels"], windows["session_index"], windows["edges"],
                                   settings["data_sufficiency"]["single_session_blocks"])
        serial = vocal_occurrence_shift_null(
            windows,
            trains,
            state["durations"],
            0,
            10,
            7,
            15.0,
            folds,
            axis,
            settings["kinematic_encoding"]["calibration"]["n_blocks"],
        )
        np.testing.assert_array_equal(payload["null"], serial)
        assert payload["n_shuffles"] == 10
        assert payload["at_floor"]
        assert payload["p"] == pytest.approx(1.0 / 11.0)
        windows = events["occurrence_windows"]
        vocal_rows = windows["labels"] > 0.5
        assert payload["fraction_vocal_windows_with_spike"] == pytest.approx(
            np.mean(windows["counts"][vocal_rows] > 0)
        )
        assert payload["fraction_quiet_windows_with_spike"] == pytest.approx(
            np.mean(windows["counts"][~vocal_rows] > 0)
        )
        stored = read_unit_artifact(str(tmp_path), UNIT["unit_uid"])
        assert stored["vocal_occurrence"]["n_shuffles"] == 10
        assert stored["identity"] == UNIT

    def test_vocalization_identity_null_is_the_serial_permutation_null(
        self, tmp_path, monkeypatch
    ):
        """
        Description
        -----------
        The identity null must take ``vocal_decoding.discrimination_null.n_permutations`` draws seeded from
        ``null.shuffle_seed`` and equal the serial ``permutation_null`` value for value; the section must
        carry the flagged descriptors the settings ask for and the per-session event counts, and land in
        the unit file.

        Parameters
        ----------
        tmp_path (pathlib.Path)
            Output directory for the unit file.
        monkeypatch (pytest.MonkeyPatch)
            Installs the in-process pool.

        Returns
        -------
        None
        """
        events = _vocal_events()
        settings = _fast_decoding_settings()
        settings["vocal_decoding"]["discrimination_null"]["n_permutations"] = 5
        spawned = _inline_pools(monkeypatch)

        payload = vocal.run_vocalization_identity(
            UNIT, events, settings, str(tmp_path), 2, _quiet
        )

        decoding = settings["vocal_decoding"]
        folds = decoding_context(
            events["positions"],
            events["session_index"],
            decoding,
            settings["vocalization_settings"]["torus_period"],
        )
        serial = permutation_null(
            lambda permutation: decode_gain(
                events["counts"], folds, decoding, permutation
            )["gain"],
            events["session_index"],
            5,
            11,
        )
        assert spawned[0][0] == 2
        assert spawned[0][1]["seed"] == 11
        np.testing.assert_array_equal(payload["null"], serial)
        assert payload["n_permutations"] == 5
        assert "bump_basis_gain" in payload
        assert payload["per_session_counts"] == {
            session_id: int(events["counts"][events["session_index"] == slot].sum())
            for slot, session_id in enumerate(events["session_ids"])
        }
        assert (
            read_unit_artifact(str(tmp_path), UNIT["unit_uid"])[
                "vocalization_identity"
            ]["gain"]
            == payload["gain"]
        )

    def test_the_multinomial_target_decodes_categories_against_their_own_null(self, tmp_path, monkeypatch):
        """
        Description
        -----------
        With ``vocal_target = "multinomial"`` identity decodes the acoustic category: the null takes
        ``n_permutations_category_grain`` draws, equals the serial permutation null of the categorical gain
        value for value, and the section records the target and the categories decoded.

        Parameters
        ----------
        tmp_path (pathlib.Path)
            Output directory for the unit file.
        monkeypatch (pytest.MonkeyPatch)
            Installs the in-process pool.

        Returns
        -------
        None
        """
        events = _vocal_events()
        settings = _fast_decoding_settings()
        settings["vocalization_settings"]["vocal_target"] = "multinomial"
        settings["vocal_decoding"]["discrimination_null"]["n_permutations_category_grain"] = 6
        _inline_pools(monkeypatch)

        payload = vocal.run_vocalization_identity(UNIT, events, settings, str(tmp_path), 2, _quiet)

        codes, kept, values = category_codes(events["category_labels"])
        folds = category_context(codes, events["session_index"][kept], events["vocalization_start"][kept],
                                 settings["data_sufficiency"]["single_session_blocks"])
        serial = permutation_null(
            lambda permutation: category_gain(events["counts"][kept], codes, folds, settings["vocal_decoding"],
                                              permutation)["gain"],
            events["session_index"][kept], 6, 11)
        np.testing.assert_array_equal(payload["null"], serial)
        assert payload["target"] == "multinomial"
        assert payload["categories"] == values.tolist()
        assert payload["n_permutations"] == 6
        assert "bump_basis_gain" not in payload

    def test_a_single_training_session_tunes_on_blocks(self, monkeypatch, tmp_path):
        """
        Description
        -----------
        A unit with two vocal sessions leaves each fold one training session; identity must hand the fold
        builder the USV onsets and the block count, so that session is cut into blocks for the amplitude
        search rather than left uncalibrated.

        Parameters
        ----------
        monkeypatch (pytest.MonkeyPatch)
            Installs the in-process pool and the recorder.
        tmp_path (pathlib.Path)
            Output directory for the unit file.

        Returns
        -------
        None
        """
        events = _vocal_events()
        two = events["session_index"] < 2
        events = {**events, "counts": events["counts"][two], "positions": events["positions"][two],
                  "session_index": events["session_index"][two], "session_ids": events["session_ids"][:2],
                  "vocalization_start": events["vocalization_start"][two],
                  "category_labels": events["category_labels"][two]}
        settings = _fast_decoding_settings()
        settings["vocal_decoding"]["discrimination_null"]["n_permutations"] = 2
        _inline_pools(monkeypatch)
        built = []
        original = vocal.decoding_context

        def recording(*args, **kwargs):
            """Record the folds the dispatcher builds."""
            folds = original(*args, **kwargs)
            built.append(folds)
            return folds

        monkeypatch.setattr(vocal, "decoding_context", recording)
        vocal.run_vocalization_identity(UNIT, events, settings, str(tmp_path), 1, _quiet)
        blocks = settings["data_sufficiency"]["single_session_blocks"]
        assert all(len(fold["inner_splits"]) == blocks for fold in built[0])

    @pytest.mark.parametrize("target", ["category", "MANIFOLD"])
    def test_an_unknown_target_is_refused(self, target, tmp_path):
        """A misspelled target must raise naming the key rather than fall back to the torus decoder."""
        settings = _fast_decoding_settings()
        settings["vocalization_settings"]["vocal_target"] = target
        with pytest.raises(ValueError, match="vocal_target"):
            vocal.run_vocalization_identity(UNIT, _vocal_events(), settings, str(tmp_path), 1, _quiet)

    def test_nested_decoding_carries_the_control_and_tests_the_posterior_statistic(
        self, tmp_path, monkeypatch
    ):
        """
        Description
        -----------
        The nested step must assemble the unit's kinematics with the encoding's history, build the design
        at ``floor(4 s * fps)`` lags with the nested floor, hand the cohort control the unit's recording day
        as an eight-character string, score the regression descriptor at the lag count the control returns
        (1) against the reduced predictions it computed, and test the POSTERIOR statistic on the kept
        events with a permutation null equal to the serial one. The section names held-out sessions and
        records the control's identity.

        Parameters
        ----------
        tmp_path (pathlib.Path)
            Output directory for the unit file.
        monkeypatch (pytest.MonkeyPatch)
            Installs the in-process pool and the stubs for the kinematic and JAX-fitted pieces.

        Returns
        -------
        None
        """
        events = _vocal_events()
        settings = _fast_decoding_settings()
        settings["nested_vocal_decoding"]["discrimination_null"][
            "n_permutations"
        ] = 4
        spawned = _inline_pools(monkeypatch)
        kept = np.arange(12, events["counts"].size)
        generator = np.random.default_rng(9)
        positions = events["positions"][kept]
        control_block = {
            "predicted_positions": (
                positions + 0.05 * generator.normal(size=positions.shape)
            )
            % 1.0,
            "features": ["head-head", "self.speed"],
            "held_out_day": "20250101",
            "settings_sha256": "abc123",
        }
        design = {
            "positions": positions,
            "session_index": events["session_index"][kept],
            "kept": kept,
            "region_labels": np.tile(np.arange(6.0), kept.size // 6 + 1)[: kept.size],
        }
        controlled = {**design, "behaviour_control": control_block}
        regression = {
            "added": 0.004,
            "reduced": -3.70,
            "full": -3.696,
            "n_events": int(kept.size),
            "n_regions_scored": 6,
            "min_leave_one_fold_out": 0.001,
            "per_fold": [{"held_out": 2, "n_events": 100, "added": 0.01}],
        }
        reduced = np.zeros((kept.size, 2))
        calls = collections.defaultdict(list)
        stubs = {
            "assemble_unit_sessions": {"s0": {"fps": 10.0}},
            "reduced_model_features": ["head-head", "self.speed"],
            "nested_design": design,
            "load_control_model": {"artifact": True},
            "apply_behaviour_control": (controlled, 1),
            "reduced_predictions": reduced,
            "nested_scores": regression,
            "no_behaviour_predictions": np.ones((kept.size, 2)),
            "pooled_macro_score": -3.74,
        }
        for name, result in stubs.items():
            monkeypatch.setattr(vocal, name, _recording_stub(calls, name, result))

        payload = vocal.run_nested_vocal_decoding(
            UNIT, events, settings, "/data", str(tmp_path), 2, _quiet
        )

        encoding = settings["kinematic_encoding"]
        assert calls["assemble_unit_sessions"][0][0][1:5] == (
            "/data",
            settings["kinematic_features"],
            encoding["history_pre_seconds"],
            encoding["clean_post_seconds"],
        )
        nested = nested_settings(settings)
        assert calls["nested_design"][0][0][2:] == (
            ["head-head", "self.speed"],
            40,
            nested["sigma_floor"],
        )
        assert calls["load_control_model"][0][0] == (settings,)
        assert calls["apply_behaviour_control"][0][0][1:] == (
            {"artifact": True},
            "20250101",
            settings,
        )
        assert calls["reduced_predictions"][0][0][2] == 1
        assert calls["nested_scores"][0][1]["reduced_predicted"] is reduced
        assert (
            calls["pooled_macro_score"][0][0][3]["torus_period"]
            == settings["vocalization_settings"]["torus_period"]
        )

        decoding = settings["vocal_decoding"]
        counts = events["counts"][kept]
        folds = decoding_context(
            positions,
            design["session_index"],
            decoding,
            settings["vocalization_settings"]["torus_period"],
        )
        kappas, priors = behaviour_priors(
            folds, control_block["predicted_positions"], positions, nested
        )
        serial = [
            posterior_nested_gain(
                counts,
                folds,
                decoding,
                priors,
                kappas,
                within_session_permutation(
                    design["session_index"], np.random.default_rng(11 + draw)
                ),
            )["added"]
            for draw in range(4)
        ]
        assert spawned[0][0] == 2
        np.testing.assert_array_equal(payload["null"], serial)
        assert payload["n_permutations"] == 4
        assert payload["n_events"] == kept.size
        assert payload["point_regression"]["no_behaviour"] == -3.74
        assert payload["point_regression"][
            "behaviour_over_no_behaviour"
        ] == pytest.approx(-3.70 + 3.74)
        assert payload["point_regression"]["per_fold"][0]["held_out_session"] == "sC"
        assert {fold["held_out_session"] for fold in payload["per_fold"]} == set(
            events["session_ids"]
        )
        assert payload["behaviour_control"] == {
            "features": ["head-head", "self.speed"],
            "held_out_day": "20250101",
            "settings_sha256": "abc123",
        }
        stored = read_unit_artifact(str(tmp_path), UNIT["unit_uid"])
        assert (
            stored["nested_vocal_decoding"]["added"]
            == payload["added"]
        )


def _gating_universe() -> dict:
    """
    Description
    -----------
    A synthetic ``gating_universe`` over 200 frames, the first 100 vocal: ``f0`` and ``f1`` vary during
    vocal frames, while ``f2`` is constant there and so has zero interquartile spread.

    Parameters
    ----------
    None.

    Returns
    -------
    universe (dict)
        Every key ``run_vocal_gating`` and the feature job read.
    """
    generator = np.random.default_rng(1)
    features = generator.normal(size=(200, 3))
    features[:100, 2] = 1.0
    return {
        "feature_names": ["f0", "f1", "f2"],
        "features": features,
        "features_quiet": features[100:],
        "vocal": np.concatenate([np.ones(100), np.zeros(100)]),
        "spikes": generator.poisson(1.0, 200),
        "spikes_quiet": np.ones(100),
        "session_index": np.zeros(200, dtype=int),
        "session_quiet": np.zeros(100, dtype=int),
        "fit_rows": None,
    }


def _gating_statistic(call: tuple) -> dict:
    """
    Description
    -----------
    What the ``feature_gating_statistic`` stub returns: a positive interaction and interaction-minus-vocal
    (so every all-zero null is beaten) and a negative quiet main effect (so it is never "present").

    Parameters
    ----------
    call (tuple)
        The ``(args, kwargs)`` pair the stub received; argument 0 is the feature's vocal-frame column.

    Returns
    -------
    statistic (dict)
        Every key the gating payload copies.
    """
    spread = float(np.std(call[0][0]))
    return {
        "feature_main": -1.0,
        "vocal_main": 10.0,
        "interaction": 100.0 * spread + 1.0,
        "additive": 50.0,
        "baseline": 900.0,
        "quiet_baseline": 70.0,
        "interaction_minus_vocal": 5.0 + spread,
        "gating_sign": 1.0,
    }


class TestVocalGatingRun:
    """Vocal gating past its switch: identifiability, the per-feature nulls, and the section it writes."""

    def _install(self, monkeypatch) -> dict:
        """
        Description
        -----------
        Stubs the kinematics loader, the universe builder, the per-feature statistic and its null with
        recorders, and installs the in-process pool; the verdicts and the unit p-value run for real.

        Parameters
        ----------
        monkeypatch (pytest.MonkeyPatch)
            Installs the stubs.

        Returns
        -------
        calls (dict)
            ``{stub name: [(args, kwargs), ...]}``.
        """
        calls = collections.defaultdict(list)
        _inline_pools(monkeypatch)
        monkeypatch.setattr(
            vocal,
            "assemble_unit_sessions",
            _recording_stub(calls, "assemble_unit_sessions", {"s0": {}}),
        )
        monkeypatch.setattr(
            vocal,
            "gating_universe",
            _recording_stub(calls, "gating_universe", _gating_universe()),
        )
        monkeypatch.setattr(
            vocal,
            "feature_gating_statistic",
            _recording_stub(calls, "statistic", _gating_statistic),
        )
        zeros = {
            key: np.zeros(2000)
            for key in ("interaction", "vocal_main", "feature_main", "difference")
        }
        monkeypatch.setattr(
            vocal, "gating_null", _recording_stub(calls, "gating_null", zeros)
        )
        return calls

    def test_a_tuned_unit_tests_every_identifiable_feature_and_stores_no_decisions(
        self, tmp_path, monkeypatch
    ):
        """
        Description
        -----------
        With the spread gate on, ``f2`` (no spread during USVs) is recorded as ``not_testable`` and the
        other two are tested, each with ``gating_screen_n_shuffles`` draws seeded
        ``seed + feature_index * n_draws`` and the shared guard. Every interaction beats its null, so each
        p sits at 1/2001 and the unit p is the two-feature Bonferroni of it. The written section carries
        no decision keys.

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
        settings["vocal_gating"]["min_feature_iqr_vocal"] = 0.5
        settings["null"]["shuffle_seed"] = 5
        calls = self._install(monkeypatch)

        payload = vocal.run_vocal_gating(
            UNIT, settings, "/data", str(tmp_path), 2, 0.001, 0.5, _quiet
        )

        universe_call = calls["gating_universe"][0]
        assert universe_call[0][3] is settings["vocal_gating"]
        assert universe_call[0][4] is settings["vocalization_settings"]
        assert universe_call[1] == {"seed": 5}
        null_calls = calls["gating_null"]
        assert [call[0][1] for call in null_calls] == [0, 1]
        assert [call[1]["seed"] for call in null_calls] == [5, 5 + 2000]
        assert all(
            call[0][3:5] == (settings["null"]["shuffle_guard_seconds"], 2000)
            for call in null_calls
        )
        assert payload["per_feature"]["f2"] == {
            "tested": False,
            "reason": "not_testable",
        }
        assert payload["per_feature"]["f0"]["p_interaction"] == pytest.approx(
            1.0 / 2001.0
        )
        assert payload["p_unit"] == pytest.approx(2.0 / 2001.0)
        assert payload["n_draws"] == 2000
        assert decision_key_paths(payload) == []
        assert (
            read_unit_artifact(str(tmp_path), UNIT["unit_uid"])["vocal_gating"]["ran"]
            is True
        )

    def test_an_untestable_and_untuned_unit_reports_the_stronger_fact(
        self, tmp_path, monkeypatch
    ):
        """
        Description
        -----------
        With the spike gate on and set above the unit's vocal spikes, an UNTUNED unit must still be
        reported ``not_testable`` -- identifiability is checked before tuning -- with every feature's
        reason recorded.

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
        settings["vocal_gating"]["min_vocal_spikes"] = 10_000
        calls = self._install(monkeypatch)
        payload = vocal.run_vocal_gating(
            UNIT, settings, "/data", str(tmp_path), 2, 0.5, 0.5, _quiet
        )
        assert "assemble_unit_sessions" in calls
        assert "gating_null" not in calls
        assert payload["ran"] is False
        assert payload["reason"] == "not_testable"
        assert set(payload["per_feature"]) == {"f0", "f1", "f2"}
        assert (
            read_unit_artifact(str(tmp_path), UNIT["unit_uid"])["vocal_gating"][
                "reason"
            ]
            == "not_testable"
        )

    def test_a_testable_but_untuned_unit_is_skipped_for_no_tuning(
        self, tmp_path, monkeypatch
    ):
        """
        Description
        -----------
        With a gate on but every feature identifiable, an untuned unit is skipped with
        ``no_vocal_tuning`` after the universe is built.

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
        settings["vocal_gating"]["min_vocal_spikes"] = 1
        self._install(monkeypatch)
        payload = vocal.run_vocal_gating(
            UNIT, settings, "/data", str(tmp_path), 2, 0.5, 0.5, _quiet
        )
        assert payload["reason"] == "no_vocal_tuning"


class TestRunVocalOrder:
    """``run_vocal`` assembles the events once, runs the steps in their fixed order, and feeds vocal gating
    the right p-values."""

    def _install(
        self, monkeypatch, occurrence_p: float = 0.001, identity_p: float = 0.2
    ) -> dict:
        """
        Description
        -----------
        Replaces the event assembler and the four analysis steps with recorders; occurrence and identity
        return the given p-values over 1,000 draws.

        Parameters
        ----------
        monkeypatch (pytest.MonkeyPatch)
            Installs the stubs.
        occurrence_p (float)
            Vocal-occurrence p-value the step returns.
        identity_p (float)
            Vocalization-identity p-value the step returns.

        Returns
        -------
        calls (dict)
            ``{stub name: [(args, kwargs), ...]}``, keyed in the order the stubs were reached.
        """
        calls = collections.defaultdict(list)
        events = {"counts": np.zeros(4), "session_ids": ["s0", "s1"]}
        stubs = {
            "assemble_unit_vocal_events": events,
            "run_vocal_occurrence": {"p": occurrence_p, "n_shuffles": 1000},
            "run_vocalization_identity": {"p": identity_p, "n_permutations": 1000},
            "run_nested_vocal_decoding": {},
            "run_vocal_gating": {},
        }
        for name, result in stubs.items():
            monkeypatch.setattr(vocal, name, _recording_stub(calls, name, result))
        return calls

    def test_every_step_runs_in_order_on_one_event_set(self, tmp_path, monkeypatch):
        """
        Description
        -----------
        Requested in reverse, the four steps must still run in ``VOCAL_STEPS`` order on ONE event set,
        assembled with the ``spike_window`` from settings, and vocal gating must receive the p-values this
        job computed.

        Parameters
        ----------
        tmp_path (pathlib.Path)
            Output directory.
        monkeypatch (pytest.MonkeyPatch)
            Installs the stubs.

        Returns
        -------
        None
        """
        settings = _settings()
        settings["vocalization_settings"]["spike_window"] = {
            "pre_offset_seconds": 0.07,
            "width_seconds": 0.2,
        }
        calls = self._install(monkeypatch)

        vocal.run_vocal(
            UNIT,
            settings,
            "/data",
            str(tmp_path),
            4,
            tuple(reversed(vocal.VOCAL_STEPS)),
            _quiet,
        )

        assert list(calls) == [
            "assemble_unit_vocal_events",
            *("run_" + step for step in vocal.VOCAL_STEPS),
        ]
        assert calls["assemble_unit_vocal_events"][0][0] == (
            UNIT,
            "/data",
            settings,
            0.07,
            0.2,
        )
        events = calls["run_vocal_occurrence"][0][0][1]
        assert calls["run_vocalization_identity"][0][0][1] is events
        assert calls["run_nested_vocal_decoding"][0][0][1] is events
        assert calls["run_vocal_occurrence"][0][0][5] == 4
        assert calls["run_vocal_gating"][0][0] == (
            UNIT,
            settings,
            "/data",
            str(tmp_path),
            4,
            0.001,
            0.2,
            _quiet,
        )

    def test_gating_reads_the_p_value_it_did_not_compute_from_the_unit_file(
        self, tmp_path, monkeypatch
    ):
        """
        Description
        -----------
        With occurrence computed in this job and identity only on disk, vocal gating must take the fresh
        occurrence p and the STORED identity p, without running identity or the nested step.

        Parameters
        ----------
        tmp_path (pathlib.Path)
            Output directory holding the stored identity section.
        monkeypatch (pytest.MonkeyPatch)
            Installs the stubs.

        Returns
        -------
        None
        """
        write_unit_section(
            str(tmp_path),
            UNIT,
            "vocalization_identity",
            {"p": 0.003, "n_permutations": 1000},
            _settings(),
        )
        write_unit_section(
            str(tmp_path),
            UNIT,
            "vocal_occurrence",
            {"p": 0.9, "n_shuffles": 1000},
            _settings(),
        )
        calls = self._install(monkeypatch, occurrence_p=0.002)
        vocal.run_vocal(
            UNIT,
            _settings(),
            "/data",
            str(tmp_path),
            1,
            ("vocal_gating", "vocal_occurrence"),
            _quiet,
        )
        assert list(calls) == [
            "assemble_unit_vocal_events",
            "run_vocal_occurrence",
            "run_vocal_gating",
        ]
        assert calls["run_vocal_gating"][0][0][5:7] == (0.002, 0.003)

    def test_gating_alone_never_assembles_events(self, tmp_path, monkeypatch):
        """
        Description
        -----------
        Vocal gating alone scores no USV-grain events, so the event set is never assembled and both
        p-values come from the unit file.

        Parameters
        ----------
        tmp_path (pathlib.Path)
            Output directory holding both stored sections.
        monkeypatch (pytest.MonkeyPatch)
            Installs the stubs.

        Returns
        -------
        None
        """
        write_unit_section(
            str(tmp_path),
            UNIT,
            "vocal_occurrence",
            {"p": 0.004, "n_shuffles": 1000},
            _settings(),
        )
        write_unit_section(
            str(tmp_path),
            UNIT,
            "vocalization_identity",
            {"p": 0.005, "n_permutations": 1000},
            _settings(),
        )
        calls = self._install(monkeypatch)
        vocal.run_vocal(
            UNIT, _settings(), "/data", str(tmp_path), 1, ("vocal_gating",), _quiet
        )
        assert list(calls) == ["run_vocal_gating"]
        assert calls["run_vocal_gating"][0][0][5:7] == (0.004, 0.005)

    def test_without_gating_nothing_reads_the_unit_file(self, tmp_path, monkeypatch):
        """
        Description
        -----------
        Running only the nested step must not reach vocal gating, nor demand stored occurrence or identity
        sections.

        Parameters
        ----------
        tmp_path (pathlib.Path)
            An empty output directory.
        monkeypatch (pytest.MonkeyPatch)
            Installs the stubs.

        Returns
        -------
        None
        """
        calls = self._install(monkeypatch)
        vocal.run_vocal(
            UNIT,
            _settings(),
            "/data",
            str(tmp_path),
            1,
            ("nested_vocal_decoding",),
            _quiet,
        )
        assert list(calls) == [
            "assemble_unit_vocal_events",
            "run_nested_vocal_decoding",
        ]


    def test_the_multinomial_target_refuses_the_nested_step_before_anything_runs(self, tmp_path, monkeypatch):
        """
        Description
        -----------
        The nested decoding has no multinomial behaviour control, so asking for it under the multinomial
        target must raise before the events are assembled or any section is written; without the nested
        step the other three run.

        Parameters
        ----------
        tmp_path (pathlib.Path)
            Output directory.
        monkeypatch (pytest.MonkeyPatch)
            Installs the stubs.

        Returns
        -------
        None
        """
        settings = _settings()
        settings["vocalization_settings"]["vocal_target"] = "multinomial"
        calls = self._install(monkeypatch)
        with pytest.raises(ValueError, match="multinomial behaviour control"):
            vocal.run_vocal(UNIT, settings, "/data", str(tmp_path), 1, vocal.VOCAL_STEPS, _quiet)
        assert not calls
        steps = tuple(step for step in vocal.VOCAL_STEPS if step != "nested_vocal_decoding")
        vocal.run_vocal(UNIT, settings, "/data", str(tmp_path), 1, steps, _quiet)
        assert "run_vocalization_identity" in calls


class TestNotVocalTestable:
    """A unit with fewer vocal sessions (or blocks) than ``min_vocal_sessions`` is recorded, not crashed."""

    def test_every_requested_section_is_written_as_not_run_and_nothing_is_fitted(self, tmp_path, monkeypatch):
        """
        Description
        -----------
        A one-session unit whose USVs all fall in one block has a single vocal block, so every vocal
        analysis would hold it out and train on nothing. ``run_vocal`` must fit nothing -- the event
        assembler and every step are recorders that must stay unreached -- and must write each requested
        section as ``{"ran": False, "reason": "not_vocal_testable", "n_vocal_sessions": 1}``.

        Parameters
        ----------
        tmp_path (pathlib.Path)
            Output directory.
        monkeypatch (pytest.MonkeyPatch)
            Installs the recorders.

        Returns
        -------
        None
        """
        calls = collections.defaultdict(list)
        for name in ("assemble_unit_vocal_events", "run_vocal_occurrence", "run_vocalization_identity",
                     "run_nested_vocal_decoding", "run_vocal_gating"):
            monkeypatch.setattr(vocal, name, _recording_stub(calls, name, {}))
        settings = _settings()
        settings["data_sufficiency"]["min_vocal_sessions"] = 2
        unit = {**UNIT, "courtship_sessions": [f"s0#block{i}" for i in range(5)], "vocal_sessions": ["s0#block0"]}

        vocal.run_vocal(unit, settings, "/data", str(tmp_path), 1, vocal.VOCAL_STEPS, _quiet)

        assert not calls
        stored = read_unit_artifact(str(tmp_path), unit["unit_uid"])
        for step in vocal.VOCAL_STEPS:
            assert stored[step] == {"ran": False, "reason": "not_vocal_testable", "n_vocal_sessions": 1}


class TestVocalCommandLine:
    """``main`` parses one unit's job and ``dispatch`` hands it to ``run_vocal``."""

    def _argv(self, *extra: str) -> list:
        """
        Description
        -----------
        A command line for unit ``imec1_cl0000_ch000_good`` of mouse ``m`` over sessions ``s0`` and ``s1``,
        followed by ``extra``.

        Parameters
        ----------
        *extra (str)
            Further arguments, e.g. ``"--n-workers", "3"``.

        Returns
        -------
        argv (list)
            ``sys.argv`` for the dispatcher.
        """
        return [
            "main_neural_vocal_dispatcher",
            "--unit-id",
            "imec1_cl0000_ch000_good",
            "--mouse-id",
            "m",
            "--rec-date",
            "20250101",
            "--sessions",
            "s0",
            "s1",
            "--data-root",
            "/data",
            "--output-directory",
            "/out",
            *extra,
        ]

    def _install(self, monkeypatch, run_vocal_result=None) -> dict:
        """
        Description
        -----------
        Stubs the per-session USV onset reader the unit record is built from (``s0``: 25 USVs, ``s1``: 5)
        and ``run_vocal`` with recorders.

        Parameters
        ----------
        monkeypatch (pytest.MonkeyPatch)
            Installs the stubs.
        run_vocal_result (Any)
            What the ``run_vocal`` stub returns, or a callable computing it (e.g. one that raises).

        Returns
        -------
        calls (dict)
            ``{stub name: [(args, kwargs), ...]}``.
        """
        calls = collections.defaultdict(list)
        monkeypatch.setattr(
            enc,
            "focal_usv_onsets",
            _recording_stub(calls, "focal_usv_onsets", {"s0": (np.arange(25.0), 150.0, 150_000),
                                                        "s1": (np.arange(5.0), 150.0, 150_000)}),
        )
        monkeypatch.setattr(
            vocal, "run_vocal", _recording_stub(calls, "run_vocal", run_vocal_result)
        )
        return calls

    def test_the_job_runs_every_step_with_the_shipped_settings_by_default(
        self, monkeypatch, capsys
    ):
        """
        Description
        -----------
        Without ``--only`` every step is requested, the shipped settings are read, the unit record applies
        the shipped 20-USV gate (keeping ``s0`` only), ``--n-workers`` arrives as an integer, and success
        exits 0 with the duration line.

        Parameters
        ----------
        monkeypatch (pytest.MonkeyPatch)
            Installs the stubs and ``sys.argv``.
        capsys (pytest.CaptureFixture)
            Captures the exit line.

        Returns
        -------
        None
        """
        calls = self._install(monkeypatch)
        monkeypatch.setattr(sys, "argv", self._argv("--n-workers", "3"))
        assert vocal.main() == 0
        unit, settings, data_root, output_directory, n_workers, steps = calls[
            "run_vocal"
        ][0][0]
        assert settings == _settings()
        assert unit["vocal_sessions"] == ["s0"]
        assert unit["unit_uid"] == "m_20250101_imec1_cl0000_ch000_good"
        assert (data_root, output_directory, n_workers) == ("/data", "/out", 3)
        assert steps == vocal.VOCAL_STEPS
        assert "finished in" in capsys.readouterr().out

    def test_only_restricts_the_steps_and_a_settings_path_is_read(
        self, tmp_path, monkeypatch
    ):
        """
        Description
        -----------
        ``--only`` must pass exactly the named steps on, and ``--settings-path`` must replace the shipped
        settings (a 3-USV gate keeps both sessions).

        Parameters
        ----------
        tmp_path (pathlib.Path)
            Directory for the settings file.
        monkeypatch (pytest.MonkeyPatch)
            Installs the stubs and ``sys.argv``.

        Returns
        -------
        None
        """
        settings = _settings()
        settings["data_sufficiency"]["min_emitter_usvs_per_session"] = 3
        settings_path = tmp_path / "settings.json"
        settings_path.write_text(json.dumps(settings))
        calls = self._install(monkeypatch)
        monkeypatch.setattr(
            sys,
            "argv",
            self._argv(
                "--n-workers",
                "1",
                "--settings-path",
                str(settings_path),
                "--only",
                "vocal_gating",
                "vocal_occurrence",
            ),
        )
        assert vocal.main() == 0
        unit, passed_settings, _root, _out, _workers, steps = calls["run_vocal"][0][0]
        assert steps == ("vocal_gating", "vocal_occurrence")
        assert passed_settings == settings
        assert unit["vocal_sessions"] == ["s0", "s1"]

    @pytest.mark.parametrize(
        "extra",
        [(), ("--n-workers", "two"), ("--n-workers", "1", "--only", "kinematics")],
    )
    def test_bad_arguments_are_refused(self, monkeypatch, extra):
        """
        Description
        -----------
        A missing or non-integer ``--n-workers`` and an ``--only`` step outside ``VOCAL_STEPS`` are usage
        errors (exit 2) that never reach ``run_vocal``.

        Parameters
        ----------
        monkeypatch (pytest.MonkeyPatch)
            Installs the stubs and ``sys.argv``.
        extra (tuple)
            The offending arguments.

        Returns
        -------
        None
        """
        calls = self._install(monkeypatch)
        monkeypatch.setattr(sys, "argv", self._argv(*extra))
        with pytest.raises(SystemExit) as raised:
            vocal.main()
        assert raised.value.code == 2
        assert "run_vocal" not in calls

    def test_a_failing_job_exits_one_with_its_traceback(self, monkeypatch, capsys):
        """
        Description
        -----------
        An exception inside the analyses must not escape: ``dispatch`` prints the traceback and returns 1,
        without the success line.

        Parameters
        ----------
        monkeypatch (pytest.MonkeyPatch)
            Installs the stubs and ``sys.argv``.
        capsys (pytest.CaptureFixture)
            Captures the traceback.

        Returns
        -------
        None
        """

        def explode(_call: tuple) -> None:
            """
            Description
            -----------
            Result function for the ``run_vocal`` stub that fails as a starved node would.

            Parameters
            ----------
            _call (tuple)
                The ignored ``(args, kwargs)`` pair.

            Returns
            -------
            Never returns; always raises ``RuntimeError``.
            """
            msg = "worker pool died"
            raise RuntimeError(msg)

        self._install(monkeypatch, explode)
        monkeypatch.setattr(sys, "argv", self._argv("--n-workers", "1"))
        assert vocal.main() == 1
        captured = capsys.readouterr()
        assert "worker pool died" in captured.err
        assert "finished in" not in captured.out
