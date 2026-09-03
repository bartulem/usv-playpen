"""
@author: bartulem
Cluster dispatcher for the kinematic encoding analysis.

The outer folds of a unit share nothing -- separate pools, separate screens, separate selections -- so they
run as independent jobs and are combined afterwards. That is not an optimization. Screen nulls alone cost
about 195 minutes per outer fold on a six-session unit, so a serial run is a day and a half per unit and a
cohort is not reachable; six concurrent jobs bring the same unit to a few hours.

    --fold K     run one outer fold and write its artifacts
    --combine    pool every fold's held-out predictions, run the null, write the result

The split falls where the analysis already wanted it: pooling has to happen after all folds exist, because
the calibration is fitted once across every held-out prediction rather than per fold.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import traceback
from zlib import crc32
from datetime import datetime

import numpy as np

from .kinematic_encoding import (fit_quiet_model, forward_select, frozen_null_scores, inner_folds,
                                 linear_predictor_at_frames, screen_features)
from .neural_design_assembly import (assemble_unit_sessions, emitter_name, load_session_usvs,
                                     silent_gap_before_call, spike_labels_at_frames,
                                     vocal_span_frames)
from .deviance_metrics import calibrated_explained_deviance
from .quiet_to_vocal_transfer import combine_folds, score_fold
from .shift_null_inference import empirical_pvalue


def load_settings(settings_path: str = None) -> dict:
    """
    Description
    -----------
    Read the neural-modeling settings, defaulting to the shipped file next to the package.

    Parameters
    ----------
    settings_path (str)
        Explicit path, or None to use the shipped settings.

    Returns
    -------
    settings (dict)
        The parsed settings.
    """

    path = (pathlib.Path(settings_path) if settings_path
            else pathlib.Path(__file__).parent.parent / "_parameter_settings"
            / "neural_modeling_settings.json")
    with open(path, "r") as settings_file:
        return json.load(settings_file)


def focal_vocal_frames(session: dict, session_id: str, data_root: str, mouse_id: str, fps: float,
                       n_lags: int) -> tuple:
    """
    Description
    -----------
    The frames of the focal animal's own calls in one session, with the silent gap preceding each call.

    Every focal call is kept. The clean-pre-history rule that defines quiet anchors is deliberately not
    applied: it is the right rule for silence, but on a bout of calls it retains almost nothing and what it
    does retain is systematically bout-initial.

    Parameters
    ----------
    session (dict)
        The session's assembled data.
    session_id (str)
        Session directory basename.
    data_root (str)
        The ``Data`` root.
    mouse_id (str)
        Focal animal id.
    fps (float)
        Camera frame rate.
    n_lags (int)
        History length in frames.

    Returns
    -------
    frames (np.ndarray)
        Flattened vocal frame indices.
    pointer (np.ndarray)
        Per-call offsets into ``frames``.
    gaps (np.ndarray)
        Silent gap preceding each retained call.
    """

    usv_table = load_session_usvs(data_root, session_id)
    emitters = usv_table["emitter"].unique().to_list()
    focal = [name for name in emitters if mouse_id in str(name)][0]
    focal_calls = usv_table.filter(usv_table["emitter"] == focal)
    order = np.argsort(focal_calls["start"].to_numpy())
    starts = focal_calls["start"].to_numpy()[order]
    stops = focal_calls["stop"].to_numpy()[order]
    kept, frames, pointer = vocal_span_frames(starts, stops, fps, session["n_frames"], n_lags)
    return frames, pointer, silent_gap_before_call(starts, stops)[kept]


def run_fold(unit: dict, fold_index: int, settings: dict, data_root: str, output_directory: str,
             message_output=print) -> dict:
    """
    Description
    -----------
    Run one outer fold: screen and select inside the pool, then score the frozen model on the held-out
    session's quiet anchors and vocal frames.

    The held-out predictions are written out rather than reduced to a score, because the transfer is scored
    by pooling every fold's predictions and calibrating once. Reducing here would throw away exactly what
    the combine step needs.

    Parameters
    ----------
    unit (dict)
        ``unit_id``, ``mouse_id``, ``rec_date``, ``courtship_sessions``.
    fold_index (int)
        Which session is held out.
    settings (dict)
        Full neural-modeling settings.
    data_root (str)
        The ``Data`` root.
    output_directory (str)
        Where the fold artifact is written.
    message_output (Callable)
        Where progress is reported.

    Returns
    -------
    artifact (dict)
        Everything the combine step needs from this fold.
    """

    encoding = settings["kinematic_encoding"]
    sessions = unit["courtship_sessions"]
    test_id = sessions[fold_index]
    pool_ids = [session for session in sessions if session != test_id]
    message_output(f"[fold {fold_index}] test = {test_id} | pool = {pool_ids}")

    # model_predictor_mouse_index, NOT anchors.onset_emitter: the first says whose kinematics predict,
    # the second says whose calls define an onset. Wiring the second here silently swapped the predictor
    # animal and built a different feature set entirely.
    per_session = assemble_unit_sessions(unit, data_root, settings["kinematic_features"],
                                         encoding["model_predictor_mouse_index"],
                                         encoding["history_pre_seconds"], encoding["clean_post_seconds"])
    fps = per_session[sessions[0]]["fps"]
    n_lags = int(np.floor(encoding["history_pre_seconds"] * fps))
    feature_names = list(per_session[sessions[0]]["feature_names"])
    # Seed from the TEST SESSION id, not the fold index, so a fold is reproducible regardless of
    # which path ran it or how the sessions were ordered. crc32 rather than hash(): Python salts
    # string hashing per process, which made an earlier run irreproducible between invocations.
    rng = np.random.default_rng(settings["null"]["shuffle_seed"] + crc32(test_id.encode()) % 10_000)

    screen_rows = screen_features(per_session, pool_ids, n_lags, rng, settings, feature_names,
                                  message_output)
    survivors = [row["feature"] for row in screen_rows if row["survived"]]
    message_output(f"    survivors: {len(survivors)}/{len(screen_rows)}")

    session = per_session[test_id]
    vocal_frames, pointer, gaps = focal_vocal_frames(session, test_id, data_root, unit["mouse_id"],
                                                     fps, n_lags)
    if not survivors:
        message_output("    NO MODEL — this fold contributes nothing")
        return {"fold_index": fold_index, "test_id": test_id, "selected": [], "path": [],
                "no_model": True, "quiet_score": 0.0, "quiet_slope": np.nan,
                "eta_vocal": np.zeros(0), "labels_vocal": np.zeros(0), "pointer": pointer,
                "gaps": gaps, "screen": screen_rows}

    selected, path = forward_select(per_session, pool_ids, survivors, n_lags, rng, settings,
                                    feature_names, message_output)
    estimator, base_rate = fit_quiet_model(per_session, pool_ids, selected, n_lags, rng, encoding)

    eta_quiet = linear_predictor_at_frames(estimator, session["feature_time_series"], selected,
                                           session["quiet"], n_lags, encoding["chunk_rows"])
    labels_quiet = spike_labels_at_frames(session["spike_frames"], session["quiet"],
                                          session["n_frames"])
    quiet_score, quiet_slope = calibrated_explained_deviance(eta_quiet, labels_quiet,
                                                             encoding["solver"]["calibration_steps"])
    quiet_null = frozen_null_scores(eta_quiet, session["quiet"], session["spike_frames"],
                                    session["n_frames"], fps, settings["null"]["n_shuffles"], rng,
                                    settings["null"]["shuffle_guard_seconds"],
                                    encoding["solver"]["calibration_steps"])
    vocal = score_fold(estimator, session, selected, vocal_frames, n_lags, base_rate, encoding,
                       linear_predictor_at_frames)
    message_output(f"    TEST {test_id}: quiet {quiet_score:+.5f} (slope {quiet_slope:+.3f}) | "
                   f"vocal {vocal['fold_score']:+.5f} (slope {vocal['fold_slope']:+.3f}) | "
                   f"{vocal['n_frames']} vocal frames")

    artifact = {"fold_index": fold_index, "test_id": test_id,
                "selected": [feature_names[index] for index in selected],
                "selected_indices": selected, "path": path, "no_model": False,
                "quiet_score": quiet_score, "quiet_slope": quiet_slope, "quiet_null": quiet_null,
                "eta_vocal": vocal["eta"], "labels_vocal": vocal["labels"], "pointer": pointer,
                "gaps": gaps, "vocal_frames": vocal_frames, "screen": screen_rows,
                "fold_vocal_score": vocal["fold_score"], "fold_vocal_slope": vocal["fold_slope"],
                "auroc_vocal": vocal["auroc"], "spike_rate_vocal": vocal["spike_rate"]}
    destination = pathlib.Path(output_directory) / f"{unit['unit_id']}_fold{fold_index}.npz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez(destination, **{key: np.asarray(value, dtype=object) if isinstance(value, (list, dict))
                             else value for key, value in artifact.items()})
    message_output(f"    wrote {destination.name}")
    return artifact


def combine(unit: dict, settings: dict, data_root: str, output_directory: str,
            message_output=print) -> dict:
    """
    Description
    -----------
    Pool every fold's held-out predictions, run the shared-draw null, and report both halves.

    Parameters
    ----------
    unit (dict)
        ``unit_id``, ``mouse_id``, ``rec_date``, ``courtship_sessions``.
    settings (dict)
        Full neural-modeling settings.
    data_root (str)
        The ``Data`` root.
    output_directory (str)
        Where the fold artifacts live and the result is written.
    message_output (Callable)
        Where the result is reported.

    Returns
    -------
    result (dict)
        Both halves' scores, p-values and floor flags.
    """

    sessions = unit["courtship_sessions"]
    encoding = settings["kinematic_encoding"]
    artifacts = []
    for fold_index in range(len(sessions)):
        path = pathlib.Path(output_directory) / f"{unit['unit_id']}_fold{fold_index}.npz"
        if not path.exists():
            raise FileNotFoundError(f"fold artifact missing: {path}")
        artifacts.append(dict(np.load(path, allow_pickle=True)))

    # model_predictor_mouse_index, NOT anchors.onset_emitter: the first says whose kinematics predict,
    # the second says whose calls define an onset. Wiring the second here silently swapped the predictor
    # animal and built a different feature set entirely.
    per_session = assemble_unit_sessions(unit, data_root, settings["kinematic_features"],
                                         encoding["model_predictor_mouse_index"],
                                         encoding["history_pre_seconds"], encoding["clean_post_seconds"])
    fps = per_session[sessions[0]]["fps"]
    n_lags = int(np.floor(encoding["history_pre_seconds"] * fps))

    fold_results, vocal_frames_by_session, scored_ids = [], {}, []
    for artifact in artifacts:
        test_id = str(artifact["test_id"])
        if bool(artifact["no_model"]):
            continue
        fold_results.append({"eta": artifact["eta_vocal"], "labels": artifact["labels_vocal"],
                             "fold_score": float(artifact["fold_vocal_score"]),
                             "fold_slope": float(artifact["fold_vocal_slope"]),
                             "auroc": float(artifact["auroc_vocal"]),
                             "spike_rate": float(artifact["spike_rate_vocal"]),
                             "n_frames": int(artifact["labels_vocal"].size)})
        vocal_frames_by_session[test_id] = artifact["vocal_frames"]
        scored_ids.append(test_id)

    if not fold_results:
        message_output("no fold produced a model; unit fails at the screen")
        return {"unit_id": unit["unit_id"], "no_model": True, "quiet_p": 1.0, "transfer_p": np.nan}

    quiet_scores = [float(a["quiet_score"]) for a in artifacts if not bool(a["no_model"])]
    quiet_nulls = np.nanmean(np.vstack([a["quiet_null"] for a in artifacts
                                        if not bool(a["no_model"])]), axis=0)
    quiet_score = float(np.nanmean(quiet_scores))
    quiet_p, quiet_at_floor = empirical_pvalue(quiet_nulls, quiet_score)
    message_output(f"  QUIET  score {quiet_score:+.5f} | p {quiet_p:.4e}"
                   f"{' (at floor)' if quiet_at_floor else ''}")

    transfer = combine_folds(fold_results, per_session, scored_ids, vocal_frames_by_session, settings,
                             message_output)
    message_output(f"  VERDICT: quiet p {quiet_p:.4e} AND transfer p {transfer['p']:.4e} "
                   f"vs q = {settings['significance']['fdr_q']}")
    return {"unit_id": unit["unit_id"], "no_model": False, "quiet_score": quiet_score,
            "quiet_p": quiet_p, "quiet_at_floor": quiet_at_floor, "transfer_score": transfer["score"],
            "transfer_slope": transfer["slope"], "transfer_p": transfer["p"],
            "transfer_at_floor": transfer["at_floor"], "n_lags": n_lags, "fps": fps,
            "selected_per_fold": [list(a["selected"]) for a in artifacts]}


def dispatch(args: argparse.Namespace) -> int:
    """
    Description
    -----------
    Route one invocation to a fold job or to the combine step, reporting the full traceback on failure so a
    pre-empted or I/O-starved cluster node leaves something diagnosable behind.

    Parameters
    ----------
    args (argparse.Namespace)
        Parsed command-line arguments.

    Returns
    -------
    status (int)
        Process exit status.
    """

    settings = load_settings(args.settings_path)
    unit = {"unit_uid": f"{args.mouse_id}_{args.rec_date}_{args.unit_id}", "mouse_id": args.mouse_id,
            "rec_date": args.rec_date, "unit_id": args.unit_id, "courtship_sessions": args.sessions}
    started = datetime.now()
    try:
        if args.combine:
            combine(unit, settings, args.data_root, args.output_directory)
        else:
            run_fold(unit, args.fold, settings, args.data_root, args.output_directory)
    except Exception:
        traceback.print_exc()
        return 1
    print(f"finished in {(datetime.now() - started).total_seconds() / 60:.1f} min")
    return 0


def main() -> int:
    """Parse arguments and dispatch."""
    parser = argparse.ArgumentParser(description="Kinematic encoding analysis, one outer fold per job.")
    parser.add_argument("--unit-id", dest="unit_id", required=True)
    parser.add_argument("--mouse-id", dest="mouse_id", required=True)
    parser.add_argument("--rec-date", dest="rec_date", type=int, required=True)
    parser.add_argument("--sessions", nargs="+", required=True)
    parser.add_argument("--data-root", dest="data_root", required=True)
    parser.add_argument("--output-directory", dest="output_directory", required=True)
    parser.add_argument("--settings-path", dest="settings_path", default=None)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--fold", type=int)
    group.add_argument("--combine", action="store_true")
    return dispatch(parser.parse_args())


if __name__ == "__main__":
    sys.exit(main())
