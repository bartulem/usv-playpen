"""
@author: bartulem
Cohort runner for the cluster: plan the cohort once, then run one unit per array task, end to end.

``plan`` builds the cohort from the settings (``neural_cohort.build_cohort_manifest``), checks that the
inputs every unit needs exist, and writes ``cohort_manifest.json`` into the output directory: the
settings hash it was built under and one record per unit. The array size is the manifest's unit count,
printed with the ``sbatch`` range, so it is never typed by hand.

``run`` takes one manifest index (``--index``, or ``SLURM_ARRAY_TASK_ID`` plus ``--index-offset``) and runs
that unit's whole analysis into the shared output directory:

1. the kinematic encoding's outer folds, each in its own process (the existing
   ``main_neural_encoding_dispatcher --fold`` entry), up to ``--parallel-folds`` at a time;
2. their combination into the ``kinematic_encoding`` section;
3. the vocal steps in order -- occurrence, identity and the nested decoding on both targets, then gating
   (``main_neural_vocal_dispatcher.run_vocal``), with ``--n-workers`` processes for the nulls.

A run resumes: a fold file or a section already written under the same settings hash is kept, so a task
that timed out restarts where it stopped. The task log is a timestamped timeline (header, folds with a
heartbeat, one line per vocal step with its result, a closing summary with peak memory), and each unit's
stages are mirrored into ``<output_directory>/status/<unit_uid>.json``; ``status`` summarises those files
across the cohort. Once a vocal step is recomputed, every later step is too, since
gating reads the occurrence and identity results. A failure in the encoding does not stop the vocal steps
(they do not depend on it); the task exits non-zero if anything failed, and every stage's duration is
printed for sizing the cohort run.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import resource
import socket
import subprocess
import sys
import time
import traceback
from datetime import UTC, datetime

import numpy as np

from ..modeling.modeling_metadata import compute_settings_sha256
from ..os_utils import configure_path
from .main_neural_encoding_dispatcher import (
    archive_folds,
    build_unit_record,
    combine,
    fold_artifact_path,
    load_settings,
    read_fold_artifact,
)
from .main_neural_vocal_dispatcher import TARGET_STEPS, VOCAL_STEPS, run_vocal
from .neural_artifacts import build_provenance, read_unit_artifact, section_is_current
from .neural_cohort import build_cohort_manifest, summarize_cohort
from .neural_design_assembly import source_session
from .neural_nested_decoding import VOCAL_TARGETS, target_section

MANIFEST_NAME = "cohort_manifest.json"
# The environment variables that cap a numerical library's thread pool, set for each fold process so
# that parallel folds share the task's cores rather than each assuming all of them.
THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
# Per-unit status files, one JSON each, under the output directory.
STATUS_DIRECTORY = "status"
# How often the running folds are reported while the encoding stage runs, in seconds.
HEARTBEAT_SECONDS = 600.0
# Lines of a failed fold's log copied into the task log.
FAILED_FOLD_LOG_LINES = 30


def recorded_sessions(record: dict) -> list:
    """
    Description
    -----------
    The recorded sessions behind a cohort record: its courtship sessions with any block suffix removed
    (a one-session unit's sessions are blocks of one recording), sorted and unique.

    Parameters
    ----------
    record (dict)
        A cohort record from ``build_cohort_manifest``.

    Returns
    -------
    sessions (list)
        Recorded session identifiers.
    """

    return sorted({source_session(session_id) for session_id in record["courtship_sessions"]})


def check_inputs(settings: dict) -> list:
    """
    Description
    -----------
    The inputs every unit needs that a cohort run cannot recover from if missing, checked once before the
    array is launched: the unit catalog, the data root, and both targets' behaviour-control and selection
    files.

    Parameters
    ----------
    settings (dict)
        Full neural-modeling settings.

    Returns
    -------
    missing (list)
        Human-readable descriptions of the missing inputs; empty when all exist.
    """

    required = {"unit catalog (data_roots.catalog_path)": settings["data_roots"]["catalog_path"],
                "data root (data_roots.data_root)": settings["data_roots"]["data_root"]}
    for target in VOCAL_TARGETS:
        required[f"{target} behaviour control (nested_vocal_decoding.behaviour_control_model_path)"] = (
            settings["nested_vocal_decoding"]["behaviour_control_model_path"][target])
        required[f"{target} behaviour selection (data.behaviour_selection_result_path)"] = (
            settings["data"]["behaviour_selection_result_path"][target])
    return [f"{name}: {path}" for name, path in required.items()
            if not pathlib.Path(configure_path(path)).exists()]


def plan(output_directory: str, settings_path: str | None = None, message_output=print) -> pathlib.Path:
    """
    Description
    -----------
    Build the cohort, check its inputs, and write the manifest the array tasks index into.

    The manifest records the settings hash; ``run`` refuses a manifest built under other settings,
    because the cohort itself (which units, which sessions) depends on them.

    Parameters
    ----------
    output_directory (str)
        Where the manifest (and later every unit's results) is written.
    settings_path (str | None)
        Neural-modeling settings file; None uses the shipped one.
    message_output (Callable)
        Where the summary is reported.

    Returns
    -------
    manifest_path (pathlib.Path)
        The written manifest.
    """

    settings = load_settings(settings_path)
    missing = check_inputs(settings)
    if missing:
        msg = "inputs missing, nothing planned:\n  " + "\n  ".join(missing)
        raise FileNotFoundError(msg)
    cohort = build_cohort_manifest(settings)
    for record in cohort:
        record["recorded_sessions"] = recorded_sessions(record)
    manifest = {"settings_sha256": compute_settings_sha256(settings),
                "settings_path": settings_path,
                "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
                "n_units": len(cohort),
                "summary": summarize_cohort(cohort),
                "units": cohort}
    destination = pathlib.Path(output_directory) / MANIFEST_NAME
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=1, default=str))
    message_output(f"cohort: {len(cohort)} units -> {destination}")
    for key, value in manifest["summary"].items():
        message_output(f"  {key}: {value}")
    message_output(f"submit with: sbatch --array=0-{len(cohort) - 1} <script>  (add %N to cap concurrent tasks)")
    return destination


def load_manifest(manifest_path: str, settings: dict) -> dict:
    """
    Description
    -----------
    Read a manifest and check it was built under these settings.

    Parameters
    ----------
    manifest_path (str)
        The manifest written by :func:`plan`.
    settings (dict)
        Full neural-modeling settings the run uses.

    Returns
    -------
    manifest (dict)
        The manifest.
    """

    manifest = json.loads(pathlib.Path(configure_path(manifest_path)).read_text())
    if manifest["settings_sha256"] != compute_settings_sha256(settings):
        msg = (f"{manifest_path} was planned under other settings ({manifest['settings_sha256'][:12]}); "
               f"re-plan under the settings this run uses")
        raise ValueError(msg)
    return manifest


def step_sections(step: str) -> list:
    """
    Description
    -----------
    The result-file sections one vocal step writes: one per target for the per-target steps, else one.

    Parameters
    ----------
    step (str)
        One of ``VOCAL_STEPS``.

    Returns
    -------
    sections (list)
        Section names.
    """

    return [target_section(step, target) for target in VOCAL_TARGETS] if step in TARGET_STEPS else [step]


def pending_vocal_steps(artifact: dict, settings: dict) -> tuple:
    """
    Description
    -----------
    The vocal steps still to run: from the first whose sections are not all current under these
    settings, every step after it as well, since later steps read earlier ones' results.

    Parameters
    ----------
    artifact (dict)
        The unit's result file (empty when none exists).
    settings (dict)
        Full neural-modeling settings.

    Returns
    -------
    steps (tuple)
        Steps to run, in ``VOCAL_STEPS`` order; empty when everything is current.
    """

    for position, step in enumerate(VOCAL_STEPS):
        if not all(section_is_current(artifact, section, settings) for section in step_sections(step)):
            return VOCAL_STEPS[position:]
    return ()


def fold_is_current(output_directory: str, unit: dict, fold_index: int, settings: dict) -> bool:
    """
    Description
    -----------
    Whether a fold's artifact exists, as its ``.npz`` file or archived in the unit's result file, and
    was written under these settings.

    Parameters
    ----------
    output_directory (str)
        Where the fold files live.
    unit (dict)
        The unit record.
    fold_index (int)
        Which fold.
    settings (dict)
        Full neural-modeling settings.

    Returns
    -------
    current (bool)
        True when the fold need not be re-run.
    """

    stored = read_fold_artifact(output_directory, unit, fold_index)
    if stored is None:
        return False
    return "settings_sha256" in stored and str(stored["settings_sha256"]) == compute_settings_sha256(settings)


def slurm_variable(name: str) -> str | None:
    """
    Description
    -----------
    A SLURM environment variable, or None outside a SLURM job (a local run), for the task header.

    Parameters
    ----------
    name (str)
        The variable.

    Returns
    -------
    value (str | None)
        Its value, or None when it is not set.
    """

    return os.environ[name] if name in os.environ else None


def timestamped(message_output=print):
    """
    Description
    -----------
    Wrap a message function so every line carries the wall-clock time, which is what lets a task log be
    read as a timeline (how long a screen took, when the nulls started) without separate bookkeeping.

    Parameters
    ----------
    message_output (Callable)
        The underlying message function.

    Returns
    -------
    stamped (Callable)
        A message function prefixing ``[YYYY-MM-DD HH:MM:SS]``.
    """

    def stamped(message: str) -> None:
        message_output(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")

    return stamped


def peak_memory_gb() -> tuple:
    """
    Description
    -----------
    Peak resident memory of this process and of its finished child processes (the fold processes), for
    sizing the array's memory request. ``ru_maxrss`` is in kilobytes on Linux.

    Parameters
    ----------
    None

    Returns
    -------
    peaks (tuple)
        ``(self_gb, children_gb)``; the children figure is the largest single child, not their sum.
    """

    return (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2,
            resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024 ** 2)


def log_tail(path: pathlib.Path, n_lines: int) -> list:
    """
    Description
    -----------
    The last lines of a log file, for copying a failed fold's error into the task log and for the
    heartbeat's progress line.

    Parameters
    ----------
    path (pathlib.Path)
        The log file.
    n_lines (int)
        How many lines.

    Returns
    -------
    lines (list)
        The last non-empty lines (empty when the file does not exist yet).
    """

    if not path.exists():
        return []
    lines = [line.rstrip() for line in path.read_text(errors="replace").splitlines() if line.strip()]
    return lines[-n_lines:]


class UnitStatus:
    """
    Description
    -----------
    The unit's status file, ``<output_directory>/status/<unit_uid>.json``, rewritten at every stage
    change so the ``status`` command can summarise thousands of tasks without parsing their logs.

    Parameters
    ----------
    output_directory (str)
        Shared results directory.
    unit_uid (str)
        The unit.
    header (dict)
        Task facts recorded once: index, host, job and array ids, settings hash, commit.

    Returns
    -------
    None
    """

    def __init__(self, output_directory: str, unit_uid: str, header: dict) -> None:
        """
        Description
        -----------
        Start the record and write it.

        Parameters
        ----------
        output_directory (str)
            Shared results directory.
        unit_uid (str)
            The unit.
        header (dict)
            Task facts recorded once.

        Returns
        -------
        None
        """

        self.path = pathlib.Path(output_directory) / STATUS_DIRECTORY / f"{unit_uid}.json"
        self.record = {"unit_uid": unit_uid, **header, "state": "running",
                       "started": datetime.now().isoformat(timespec="seconds"), "finished": None, "stages": {}}
        self.write()

    def stage(self, name: str, state: str, minutes: float | None = None, detail: str | None = None) -> None:
        """
        Description
        -----------
        Record one stage's state (``running``, ``done``, ``kept``, ``failed``) and duration.

        Parameters
        ----------
        name (str)
            The stage.
        state (str)
            Its state.
        minutes (float | None)
            Its duration, once finished.
        detail (str | None)
            A short result or error line.

        Returns
        -------
        None
        """

        self.record["stages"][name] = {"state": state, "minutes": minutes, "detail": detail}
        self.write()

    def finish(self, succeeded: bool) -> None:
        """
        Description
        -----------
        Close the record with the task's outcome and peak memory.

        Parameters
        ----------
        succeeded (bool)
            Whether every stage succeeded.

        Returns
        -------
        None
        """

        own, children = peak_memory_gb()
        self.record.update({"state": "done" if succeeded else "failed",
                            "finished": datetime.now().isoformat(timespec="seconds"),
                            "peak_memory_gb": {"runner": round(own, 2), "largest_fold": round(children, 2)}})
        self.write()

    def write(self) -> None:
        """
        Description
        -----------
        Publish the record atomically, so the ``status`` command never reads half a file.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.record, indent=1))
        temporary.replace(self.path)


def run_folds(unit: dict, folds: list, output_directory: str, data_root: str, settings_path: str | None,
              parallel_folds: int, threads_per_fold: int, message_output=print) -> list:
    """
    Description
    -----------
    Run the given encoding folds, each as its own ``main_neural_encoding_dispatcher --fold`` process, at
    most ``parallel_folds`` at a time, each logging to ``<output_directory>/logs/<unit_uid>_fold<i>.log``.

    Separate processes rather than a pool: each fold builds a large design and runs its own JAX fits, and
    a process that exits returns its memory. The thread-pool variables cap each fold's numerical threads
    so parallel folds share the task's cores. Every ``HEARTBEAT_SECONDS`` the running folds are reported
    with their elapsed time and their log's latest line (the screen or selection step they are on); a
    failed fold's last ``FAILED_FOLD_LOG_LINES`` lines are copied into the task log.

    Parameters
    ----------
    unit (dict)
        The unit record.
    folds (list)
        Fold indices to run.
    output_directory (str)
        Where the fold files are written.
    data_root (str)
        The ``Data`` root.
    settings_path (str | None)
        Settings file handed to each fold; None uses the shipped one.
    parallel_folds (int)
        Folds run at once.
    threads_per_fold (int)
        Numerical threads per fold process.
    message_output (Callable)
        Where progress is reported.

    Returns
    -------
    failed (list)
        Fold indices whose process exited non-zero.
    """

    log_directory = pathlib.Path(output_directory) / "logs"
    log_directory.mkdir(parents=True, exist_ok=True)
    environment = {**os.environ, **dict.fromkeys(THREAD_VARIABLES, str(threads_per_fold))}
    base = [sys.executable, "-m", "usv_playpen.neural_modeling.main_neural_encoding_dispatcher",
            "--unit-id", unit["unit_id"], "--mouse-id", unit["mouse_id"], "--rec-date", str(unit["rec_date"]),
            "--sessions", *unit["recorded_sessions"], "--data-root", data_root,
            "--output-directory", output_directory]
    if settings_path:
        base += ["--settings-path", settings_path]
    waiting, running, failed = list(folds), {}, []
    last_heartbeat = time.time()
    while waiting or running:
        while waiting and len(running) < parallel_folds:
            fold_index = waiting.pop(0)
            log_path = log_directory / f"{unit['unit_uid']}_fold{fold_index}.log"
            log_handle = log_path.open("w")
            process = subprocess.Popen([*base, "--fold", str(fold_index)], stdout=log_handle,
                                       stderr=subprocess.STDOUT, env=environment)
            running[fold_index] = (process, log_handle, time.time(), log_path)
            message_output(f"  fold {fold_index} started ({threads_per_fold} threads, log {log_path.name})")
        for fold_index, (process, log_handle, started, log_path) in list(running.items()):
            if process.poll() is None:
                continue
            log_handle.close()
            del running[fold_index]
            minutes = (time.time() - started) / 60.0
            if process.returncode == 0:
                message_output(f"  fold {fold_index} done in {minutes:.1f} min")
            else:
                failed.append(fold_index)
                message_output(f"  fold {fold_index} FAILED (exit {process.returncode}) after {minutes:.1f} min; "
                               f"last lines of {log_path.name}:")
                for line in log_tail(log_path, FAILED_FOLD_LOG_LINES):
                    message_output(f"    | {line}")
        if running and time.time() - last_heartbeat >= HEARTBEAT_SECONDS:
            last_heartbeat = time.time()
            for fold_index, (_process, _handle, started, log_path) in sorted(running.items()):
                latest = log_tail(log_path, 1)
                message_output(f"  heartbeat: fold {fold_index} running {(time.time() - started) / 60.0:.0f} min"
                               f"{' | ' + latest[0].strip()[:140] if latest else ''}")
            if waiting:
                message_output(f"  heartbeat: folds waiting {waiting}")
        time.sleep(5.0)
    return sorted(failed)


def section_summary(artifact: dict, section: str) -> str:
    """
    Description
    -----------
    One line on a written section, for the task log: its p-value and effect, or why it did not run.

    Parameters
    ----------
    artifact (dict)
        The unit's result file.
    section (str)
        The section.

    Returns
    -------
    summary (str)
        For example ``p 1.0e-03 (floor) | gain +0.0518`` or ``not run: not_vocal_testable``.
    """

    if section not in artifact:
        return "missing"
    payload = artifact[section]
    if "ran" in payload and payload["ran"] is False:
        return f"not run: {payload['reason']}"
    parts = []
    for key in ("p", "quiet_p", "transfer_p", "p_unit"):
        if key in payload and payload[key] is not None:
            parts.append(f"{key} {float(payload[key]):.1e}")
    if "at_floor" in payload and bool(payload["at_floor"]):
        parts.append("(floor)")
    for key in ("gain", "added"):
        if key in payload and payload[key] is not None:
            parts.append(f"{key} {float(payload[key]):+.4f}")
    if "no_model" in payload and bool(payload["no_model"]):
        parts.append("no model")
    return " | ".join(parts) if parts else "written"


def run_unit(record: dict, settings: dict, settings_path: str | None, data_root: str, output_directory: str,
             n_workers: int, parallel_folds: int, task: dict | None = None, message_output=print) -> bool:
    """
    Description
    -----------
    Run one unit's whole analysis, keeping whatever is already current under these settings, and log it
    so the run can be followed while it happens.

    The unit record is rebuilt from the recorded sessions exactly as the per-analysis dispatchers build
    it, and must agree with the manifest's sessions; a disagreement means the data changed after
    planning, and the unit is refused rather than analysed on other sessions than the cohort lists.

    The log opens with a header (unit, area, sessions, settings hash, commit, host, job), carries a
    timestamp on every line (the dispatchers' own lines included), reports each fold, each vocal step with
    its duration and result line, and closes with a per-stage summary and the peak memory. The same
    stages are mirrored into the unit's status file (:class:`UnitStatus`).

    Parameters
    ----------
    record (dict)
        The unit's manifest record.
    settings (dict)
        Full neural-modeling settings.
    settings_path (str | None)
        The settings file, handed to the fold processes.
    data_root (str)
        The ``Data`` root.
    output_directory (str)
        Shared results directory.
    n_workers (int)
        Processes for the vocal nulls.
    parallel_folds (int)
        Encoding folds run at once.
    task (dict | None)
        Task facts for the header and status file (index, host, job and array ids); None records none.
    message_output (Callable)
        Where progress is reported (timestamps are added here).

    Returns
    -------
    succeeded (bool)
        False when any stage failed.
    """

    log = timestamped(message_output)
    unit = build_unit_record(record["unit_id"], record["mouse_id"], int(record["rec_date"]),
                             record["recorded_sessions"], data_root, settings)
    for key in ("courtship_sessions", "vocal_sessions"):
        if list(unit[key]) != list(record[key]):
            msg = (f"{record['unit_uid']}: {key} rebuilt as {unit[key]} but planned as {record[key]}; "
                   f"the data changed after planning -- re-plan")
            raise ValueError(msg)
    unit["recorded_sessions"] = list(record["recorded_sessions"])
    unit["brain_area"] = record["brain_area"]
    provenance = build_provenance(settings)
    header = {**(task or {}), "brain_area": unit["brain_area"], "settings_sha256": provenance["settings_sha256"],
              "git_commit": provenance["git_commit"], "n_workers": n_workers, "parallel_folds": parallel_folds}
    status = UnitStatus(output_directory, unit["unit_uid"], header)
    rule = "=" * 100
    log(rule)
    log(f"UNIT {unit['unit_uid']} | {unit['brain_area']}"
        f"{' | block split ' + str(record['block_split']) if record['block_split'] else ''}")
    log(f"  courtship sessions ({len(unit['courtship_sessions'])}): {', '.join(unit['courtship_sessions'])}")
    log(f"  vocal sessions ({len(unit['vocal_sessions'])}): {', '.join(unit['vocal_sessions'])}")
    log(f"  settings {provenance['settings_sha256'][:12]} | commit {str(provenance['git_commit'])[:12]}"
        f"{' (dirty)' if provenance['git_dirty'] else ''} | workers {n_workers} | parallel folds {parallel_folds}")
    if task:
        log("  task " + " | ".join(f"{key} {value}" for key, value in task.items()))
    log(rule)
    succeeded = True

    started = time.time()
    n_folds = len(unit["courtship_sessions"])
    stale = [fold for fold in range(n_folds) if not fold_is_current(output_directory, unit, fold, settings)]
    log(f"STAGE kinematic_encoding: {n_folds} folds, {n_folds - len(stale)} kept, {len(stale)} to run")
    status.stage("kinematic_encoding", "running")
    try:
        threads = max(1, n_workers // max(1, min(parallel_folds, len(stale)))) if stale else 1
        failed = run_folds(unit, stale, output_directory, data_root, settings_path, parallel_folds, threads, log)
        if failed:
            msg = f"encoding folds {failed} failed; see their logs"
            raise RuntimeError(msg)
        artifact = read_unit_artifact(output_directory, unit["unit_uid"])
        if stale or not section_is_current(artifact, "kinematic_encoding", settings):
            log("  combining folds")
            combine(unit, settings, data_root, output_directory, log)
            state = "done"
        else:
            state = "kept"
            # A task stopped between writing the section and archiving its folds leaves .npz files behind.
            leftover = [fold for fold in range(n_folds) if fold_artifact_path(output_directory, unit, fold).exists()]
            if leftover:
                archive_folds(unit, [read_fold_artifact(output_directory, unit, fold) for fold in range(n_folds)],
                              settings, output_directory, log)
        summary = section_summary(read_unit_artifact(output_directory, unit["unit_uid"]), "kinematic_encoding")
        minutes = (time.time() - started) / 60.0
        log(f"STAGE kinematic_encoding {state.upper()} in {minutes:.1f} min | {summary}")
        status.stage("kinematic_encoding", state, minutes, summary)
    except Exception as error:
        traceback.print_exc()
        minutes = (time.time() - started) / 60.0
        log(f"STAGE kinematic_encoding FAILED after {minutes:.1f} min: {error!r}")
        status.stage("kinematic_encoding", "failed", minutes, repr(error))
        succeeded = False

    steps = pending_vocal_steps(read_unit_artifact(output_directory, unit["unit_uid"]), settings)
    for step in VOCAL_STEPS:
        if step not in steps:
            log(f"STAGE {step} KEPT (current under these settings)")
            status.stage(step, "kept")
    for step in steps:
        started = time.time()
        log(f"STAGE {step}: started")
        status.stage(step, "running")
        try:
            run_vocal(unit, settings, data_root, output_directory, n_workers, (step,), log)
            artifact = read_unit_artifact(output_directory, unit["unit_uid"])
            summary = "; ".join(f"{section}: {section_summary(artifact, section)}" for section in step_sections(step))
            minutes = (time.time() - started) / 60.0
            log(f"STAGE {step} DONE in {minutes:.1f} min | {summary}")
            status.stage(step, "done", minutes, summary)
        except Exception as error:
            traceback.print_exc()
            minutes = (time.time() - started) / 60.0
            log(f"STAGE {step} FAILED after {minutes:.1f} min: {error!r}")
            status.stage(step, "failed", minutes, repr(error))
            succeeded = False

    own, children = peak_memory_gb()
    log(rule)
    log(f"SUMMARY {unit['unit_uid']}: {'OK' if succeeded else 'FAILED'}")
    for name, entry in status.record["stages"].items():
        minutes = "" if entry["minutes"] is None else f" {entry['minutes']:.1f} min"
        log(f"  {name:<26} {entry['state']:<7}{minutes}")
    log(f"  peak memory: runner {own:.1f} GB, largest fold process {children:.1f} GB")
    log(rule)
    status.finish(succeeded)
    return succeeded


def cohort_status(manifest_path: str, output_directory: str, message_output=print) -> dict:
    """
    Description
    -----------
    Summarise a cohort run from the units' status files: how many units are done, running, failed or not
    started, which failed and at which stage, and the median duration of each stage over finished units.

    A ``running`` record whose task was killed (time limit, node failure) stays ``running`` until the
    task is resubmitted; the record's ``started`` time tells a stuck task from a slow one.

    Parameters
    ----------
    manifest_path (str)
        The cohort manifest.
    output_directory (str)
        Shared results directory.
    message_output (Callable)
        Where the summary is reported.

    Returns
    -------
    summary (dict)
        ``counts`` per state, ``failed`` (uid to failed stages) and ``median_minutes`` per stage.
    """

    manifest = json.loads(pathlib.Path(configure_path(manifest_path)).read_text())
    status_directory = pathlib.Path(output_directory) / STATUS_DIRECTORY
    counts = {"done": 0, "running": 0, "failed": 0, "not_started": 0}
    failed, durations = {}, {}
    for record in manifest["units"]:
        path = status_directory / f"{record['unit_uid']}.json"
        if not path.exists():
            counts["not_started"] += 1
            continue
        entry = json.loads(path.read_text())
        counts[entry["state"]] += 1
        if entry["state"] == "failed":
            failed[record["unit_uid"]] = [name for name, stage in entry["stages"].items() if stage["state"] == "failed"]
        for name, stage in entry["stages"].items():
            if stage["state"] == "done" and stage["minutes"] is not None:
                durations.setdefault(name, []).append(stage["minutes"])
    median_minutes = {name: float(np.median(values)) for name, values in durations.items()}
    message_output(f"{manifest['n_units']} units: " + ", ".join(f"{state} {n}" for state, n in counts.items()))
    for name, minutes in median_minutes.items():
        message_output(f"  median {name}: {minutes:.1f} min over {len(durations[name])} units")
    for uid, stages in failed.items():
        message_output(f"  FAILED {uid}: {', '.join(stages) if stages else 'before any stage'}")
    return {"counts": counts, "failed": failed, "median_minutes": median_minutes}


def main() -> int:
    """
    Description
    -----------
    Command-line entry: ``plan``, ``run`` or ``status`` (see the module docstring).

    Parameters
    ----------
    None

    Returns
    -------
    status (int)
        Process exit status.
    """

    parser = argparse.ArgumentParser(description="Neural modeling cohort runner: plan once, then one unit per task.")
    commands = parser.add_subparsers(dest="command", required=True)
    plan_parser = commands.add_parser("plan", help="build the cohort manifest")
    plan_parser.add_argument("--output-directory", dest="output_directory", required=True)
    plan_parser.add_argument("--settings-path", dest="settings_path", default=None)
    run_parser = commands.add_parser("run", help="run one unit of the manifest")
    run_parser.add_argument("--manifest", required=True)
    run_parser.add_argument("--index", type=int, default=None,
                            help="manifest index; default is SLURM_ARRAY_TASK_ID + --index-offset")
    run_parser.add_argument("--index-offset", dest="index_offset", type=int, default=0)
    run_parser.add_argument("--output-directory", dest="output_directory", required=True)
    run_parser.add_argument("--data-root", dest="data_root", required=True)
    run_parser.add_argument("--n-workers", dest="n_workers", type=int, required=True)
    run_parser.add_argument("--parallel-folds", dest="parallel_folds", type=int, default=1)
    run_parser.add_argument("--settings-path", dest="settings_path", default=None)
    status_parser = commands.add_parser("status", help="summarise a cohort run from its status files")
    status_parser.add_argument("--manifest", required=True)
    status_parser.add_argument("--output-directory", dest="output_directory", required=True)
    args = parser.parse_args()

    if args.command == "plan":
        plan(args.output_directory, args.settings_path)
        return 0
    if args.command == "status":
        cohort_status(args.manifest, configure_path(args.output_directory))
        return 0
    settings = load_settings(args.settings_path)
    manifest = load_manifest(args.manifest, settings)
    index = args.index if args.index is not None else int(os.environ["SLURM_ARRAY_TASK_ID"]) + args.index_offset
    if not 0 <= index < manifest["n_units"]:
        print(f"index {index} is outside the manifest's {manifest['n_units']} units; nothing to do")
        return 0
    task = {"index": index, "host": socket.gethostname(),
            "slurm_job": slurm_variable("SLURM_ARRAY_JOB_ID") or slurm_variable("SLURM_JOB_ID"),
            "slurm_task": slurm_variable("SLURM_ARRAY_TASK_ID")}
    started = datetime.now()
    succeeded = run_unit(manifest["units"][index], settings, args.settings_path, configure_path(args.data_root),
                         configure_path(args.output_directory), args.n_workers, args.parallel_folds, task)
    print(f"unit {index} finished in {(datetime.now() - started).total_seconds() / 60:.1f} min"
          f"{'' if succeeded else ' WITH FAILURES'}")
    return 0 if succeeded else 1


if __name__ == "__main__":
    sys.exit(main())
