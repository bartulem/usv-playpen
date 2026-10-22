"""
@author: bartulem
Cluster dispatcher for the kinematic encoding analysis.

The outer folds of a unit share nothing (separate pools, screens and selections), so each runs as an
independent job and the results are combined afterwards; pooling has to wait for every fold because the
transfer calibration is fitted once across all held-out predictions.

    --fold K     run one outer fold and write its artifact
    --combine    pick the representative fold, pool the transfer, compute the filter band and STA, and
                 write the unit's kinematic_encoding section
    --single     select once on all sessions but one, fit once, and write the same section

Exports: ``run_fold``, ``combine``, ``run_single``, ``representative_fold``, ``per_fold_roster``,
``focal_vocal_frames``, ``build_unit_record`` and the command-line entry point ``main``.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
import traceback
from datetime import datetime
from zlib import crc32

import numpy as np

from .deviance_metrics import encoding_metrics, frozen_model_score
from .kinematic_encoding import (
    filter_band,
    filter_descriptors,
    fit_quiet_model,
    forward_select,
    frozen_null_scores,
    linear_predictor_at_frames,
    quiet_block_sessions,
    screen_features,
    spike_triggered_average,
)
from .neural_artifacts import write_unit_section
from .neural_cohort import emitter_usv_counts
from .neural_design_assembly import (
    assemble_unit_sessions,
    emitter_names,
    filter_vocalizations,
    load_session_usvs,
    silent_gap_before_vocalization,
    spike_labels_at_frames,
    vocal_span_frames,
)
from .quiet_to_vocal_transfer import combine_folds, score_fold, transfer_frames
from .shift_null_inference import escalated_empirical_pvalue


def unwrap_stored(value):
    """
    Description
    -----------
    Recover a dict or list that ``np.savez`` wrapped in a 0-d object array.

    The per-fold artifacts are npz, which has no notion of a nested mapping: a dict goes in and a 0-d
    object array comes out.

    Parameters
    ----------
    value (Any)
        A value read back from an ``npz`` archive.

    Returns
    -------
    value (Any)
        The original object where one was boxed, otherwise the value unchanged.
    """

    return value.item() if isinstance(value, np.ndarray) and value.ndim == 0 else value


def load_settings(settings_path: str | None = None) -> dict:
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
    with pathlib.Path(path).open() as settings_file:
        return json.load(settings_file)


def focal_vocal_frames(session: dict, session_id: str, data_root: str, mouse_id: str, fps: float,
                       n_lags: int, vocal_settings: dict) -> tuple:
    """
    Description
    -----------
    The frames of the focal vocalizations in one session, with the silent gap preceding each.

    "Focal" is ``vocal_settings["vocal_emitter"]``, resolved against the session's track names exactly as
    the vocal-side assembler resolves it: ``'self'`` is the unit's own animal, ``'partner'`` the other one,
    and ``'all'`` applies no emitter filter, so the transfer frames and the vocal analyses' events always
    follow the same animal.

    Every focal vocalization is kept: the clean-history rule that defines quiet anchors would keep almost
    nothing inside a bout, and only bout-initial USVs.

    Parameters
    ----------
    session (dict)
        The session's assembled data; ``n_frames`` and ``track_names`` are read.
    session_id (str)
        Session directory basename.
    data_root (str)
        The ``Data`` root.
    mouse_id (str)
        The unit's (recorded) animal id.
    fps (float)
        Camera frame rate.
    n_lags (int)
        History length in frames.
    vocal_settings (dict)
        The ``vocalization_settings`` block: the exclusion flags (``'analysed'`` scope) and
        ``vocal_emitter``.

    Returns
    -------
    frames (np.ndarray)
        Flattened vocal frame indices.
    pointer (np.ndarray)
        Per-vocalization offsets into ``frames``.
    gaps (np.ndarray)
        Silent gap preceding each retained vocalization.
    """

    # Match emitter labels exactly, as the assembler does: a bare mouse id is a prefix of the same
    # animal's suffixed form, so substring matching is fragile.
    usv_table = filter_vocalizations(load_session_usvs(data_root, session_id), vocal_settings,
                                     "analysed")
    focal_names = emitter_names(session["track_names"], mouse_id, vocal_settings["vocal_emitter"])
    focal_vocalizations = (usv_table.filter(usv_table["emitter"].is_in(focal_names)) if focal_names
                           else usv_table)
    if focal_vocalizations.height == 0:
        # The focal animal did not vocalize in this session; it contributes no vocal frames.
        return (np.zeros(0, dtype=np.int64), np.zeros(1, dtype=np.int64), np.zeros(0, dtype=np.float64))
    order = np.argsort(focal_vocalizations["start"].to_numpy())
    starts = focal_vocalizations["start"].to_numpy()[order]
    stops = focal_vocalizations["stop"].to_numpy()[order]
    kept, frames, pointer = vocal_span_frames(starts, stops, fps, session["n_frames"], n_lags)
    return frames, pointer, silent_gap_before_vocalization(starts, stops)[kept]


def write_fold_artifact(artifact: dict, unit: dict, fold_index: int, output_directory: str,
                        message_output=print) -> dict:
    """
    Description
    -----------
    Save one fold's artifact where ``combine`` will look for it, and hand it back.

    Every fold goes through here, including folds that found no model, because ``combine`` requires a
    file for every fold and an untuned unit must appear as a "not tuned" result rather than as a failed
    job. Lists and dicts are boxed as object arrays; see :func:`unwrap_stored` for the reverse.

    Parameters
    ----------
    artifact (dict)
        The fold's results, model or no model.
    unit (dict)
        The unit record; ``unit_id`` names the file.
    fold_index (int)
        Which fold this is.
    output_directory (str)
        Where the fold files live.
    message_output (Callable)
        Where the write is reported.

    Returns
    -------
    artifact (dict)
        The same artifact, unchanged.
    """

    destination = pathlib.Path(output_directory) / f"{unit['unit_id']}_fold{fold_index}.npz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez(destination, **{key: np.asarray(value, dtype=object) if isinstance(value, (list, dict))
                             else value for key, value in artifact.items()})
    message_output(f"    wrote {destination.name}")
    return artifact


def run_fold(unit: dict, fold_index: int, settings: dict, data_root: str, output_directory: str,
             message_output=print) -> dict:
    """
    Description
    -----------
    Run one outer fold: screen and select inside the pool, then score the frozen model on the held-out
    session's quiet anchors and on every vocal session's transfer frames.

    The held-out predictions are written out rather than reduced to a score, because the transfer is
    scored by pooling predictions and calibrating once in ``combine``.

    Parameters
    ----------
    unit (dict)
        ``unit_id``, ``mouse_id``, ``rec_date``, ``courtship_sessions``, ``vocal_sessions``.
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

    # The `self.` role comes from the unit's own mouse_id; `vocal_emitter` (read by `focal_vocal_frames`)
    # decides whose vocalizations are scored.
    per_session = assemble_unit_sessions(unit, data_root, settings["kinematic_features"],
                                         encoding["history_pre_seconds"], encoding["clean_post_seconds"],
                                         settings["vocalization_settings"]["clean_against"],
                                         settings["vocalization_settings"])
    fps = per_session[sessions[0]]["fps"]
    # `history_pre_seconds` is the filter length, the quiet window's backward guard and the gap of the
    # single-session block split, so changing it redefines all three.
    n_lags = int(np.floor(encoding["history_pre_seconds"] * fps))
    feature_names = list(per_session[sessions[0]]["feature_names"])
    # Seeded from the test session id (crc32, not the per-process salted hash()), so a fold is
    # reproducible however it was run.
    rng = np.random.default_rng(settings["null"]["shuffle_seed"] + crc32(test_id.encode()) % 10_000)

    # The test session's transfer frames are computed before the pool split, since both no-model
    # returns below report them.
    session = per_session[test_id]
    transfer_settings = settings["kinematic_encoding"]["transfer"]
    vocal_frames, pointer, gaps = focal_vocal_frames(session, test_id, data_root, unit["mouse_id"],
                                                     fps, n_lags, settings["vocalization_settings"])
    vocal_frames = transfer_frames(vocal_frames, pointer, fps, session["n_frames"], n_lags, session["quiet"],
                                   transfer_settings)

    inner_sessions, inner_ids = per_session, pool_ids
    if len(pool_ids) == 1:
        # A pool of one session has no inner session to hold out, so the inner split uses contiguous
        # quiet-anchor blocks of that session, separated by the predictor history so no validation
        # frame's lags reach a training block. The final model is still fitted on the whole session.
        required_gap = float(encoding["history_pre_seconds"])
        blocks = quiet_block_sessions(per_session[pool_ids[0]], pool_ids[0],
                                      settings["data_sufficiency"]["single_session_inner_split_blocks"],
                                      int(np.ceil(required_gap * fps / 2.0)))
        if len(blocks) < 2:
            message_output(f"    NO MODEL: {pool_ids[0]} yields {len(blocks)} usable quiet blocks")
            return write_fold_artifact(
                {"fold_index": fold_index, "test_id": test_id, "selected": [], "path": [],
                 "no_model": True, "quiet_score": 0.0, "quiet_slope": np.nan,
                 "quiet_p": 1.0, "quiet_at_floor": False,
                 "eta_vocal": np.zeros(0), "labels_vocal": np.zeros(0), "pointer": pointer,
                 "gaps": gaps, "screen": [], "vocal_everywhere": []},
                unit, fold_index, output_directory, message_output)
        inner_sessions, inner_ids = {**per_session, **blocks}, list(blocks)
        message_output(f"    one pool session: inner split is {len(blocks)} contiguous quiet blocks "
                       f"with a {required_gap:g} s gap")

    screen_rows = screen_features(inner_sessions, inner_ids, n_lags, rng, settings, feature_names,
                                  message_output)
    survivors = [row["feature"] for row in screen_rows if row["survived"]]
    message_output(f"    survivors: {len(survivors)}/{len(screen_rows)}")

    # A fold with no survivor has no model. Its best screened feature is still fitted and scored, for
    # description only (`no_model` stays True): its p is a maximum over all screened features and is
    # never a test.
    best_screened = None
    if not survivors:
        scored_rows = [row for row in screen_rows if np.isfinite(row["score"])]
        if scored_rows:
            best_screened = max(scored_rows, key=lambda row: row["score"])
            message_output(f"    NO MODEL — best screened feature {best_screened['name']} "
                           f"(score {best_screened['score']:+.5f}, p {best_screened['p']:.2e}) is fitted for "
                           f"description only; it failed the {len(screen_rows)}-feature correction")
        else:
            message_output("    NO MODEL — this fold contributes nothing")

    selected = [best_screened["feature"]] if best_screened is not None else []
    path = []
    if survivors:
        selected, path = forward_select(inner_sessions, inner_ids, survivors, n_lags, rng, settings,
                                        feature_names, message_output)
    if not selected:
        return write_fold_artifact(
            {"fold_index": fold_index, "test_id": test_id, "selected": [], "path": [],
             "no_model": True, "quiet_score": 0.0, "quiet_slope": np.nan,
             "quiet_p": 1.0, "quiet_at_floor": False,
             "eta_vocal": np.zeros(0), "labels_vocal": np.zeros(0), "pointer": pointer,
             "gaps": gaps, "screen": screen_rows, "best_screened": False},
            unit, fold_index, output_directory, message_output)
    fit_started = time.time()
    estimator, base_rate = fit_quiet_model(per_session, pool_ids, selected, n_lags, rng, encoding,
                                           message_output)
    final_fit_seconds = time.time() - fit_started
    # Convergence of the final model is stored with the artifact, not only logged.
    final_converged = bool(estimator.converged_)

    eta_quiet = linear_predictor_at_frames(estimator, session["feature_time_series"], selected,
                                           session["quiet"], n_lags, encoding["chunk_rows"])
    labels_quiet = spike_labels_at_frames(session["spike_frames"], session["quiet"],
                                          session["n_frames"])
    quiet_score, quiet_slope = frozen_model_score(eta_quiet, labels_quiet,
                                                  np.zeros(labels_quiet.size, dtype=np.int64),
                                                  encoding["solver"]["calibration_steps"], "bernoulli", 1.0,
                                                  encoding["calibration"])
    def draw_quiet_null(count):
        """Fresh quiet-null draws for the escalation ladder; the generator carries on where it left off."""
        return frozen_null_scores(eta_quiet, session["quiet"], session["spike_frames"],
                                  session["n_frames"], fps, count, rng,
                                  settings["null"]["shuffle_guard_seconds"],
                                  encoding["solver"]["calibration_steps"],
                                  encoding["solver"]["null_calibration_bins"], encoding["calibration"],
                                  encoding["null_block_megabytes"])

    quiet_null = draw_quiet_null(settings["null"]["n_shuffles"])
    quiet_p, quiet_at_floor, quiet_null = escalated_empirical_pvalue(
        quiet_score, draw_quiet_null, settings["significance"]["escalation_ladder"], quiet_null,
        message_output)
    vocal = score_fold(estimator, session, selected, vocal_frames, n_lags, base_rate, encoding,
                       linear_predictor_at_frames, transfer_settings)

    # This fold's model is scored on every vocal session's transfer frames, not just its own, so the
    # transfer in `combine` pools ONE model (the representative fold's). No vocal frame was ever fitted,
    # so every session is held out for it. Only vocal sessions are scored.
    vocal_everywhere = []
    for other_id in [s for s in sessions if s in set(unit["vocal_sessions"])]:
        other_frames, other_pointer, other_gaps = focal_vocal_frames(
            per_session[other_id], other_id, data_root, unit["mouse_id"], fps, n_lags,
            settings["vocalization_settings"])
        other_frames = transfer_frames(other_frames, other_pointer, fps, per_session[other_id]["n_frames"],
                                       n_lags, per_session[other_id]["quiet"], transfer_settings)
        scored = score_fold(estimator, per_session[other_id], selected, other_frames, n_lags,
                            base_rate, encoding, linear_predictor_at_frames, transfer_settings)
        scored["session_id"] = other_id
        scored["pointer"], scored["gaps"] = other_pointer, other_gaps
        scored["frames"] = other_frames
        vocal_everywhere.append(scored)
    if not final_converged:
        message_output(f"    [!] fold {fold_index} FINAL MODEL DID NOT CONVERGE -- treat both halves "
                       f"of this fold as provisional")
    # The own-session vocal score is a diagnostic; the tested transfer is pooled in `combine`.
    message_output(f"    TEST {test_id}: quiet {quiet_score:+.5f} (slope {quiet_slope:+.3f}) "
                   f"p {quiet_p:.3e} | [diagnostic] own-session vocal {vocal['fold_score']:+.5f} "
                   f"(slope {vocal['fold_slope']:+.3f}) on {vocal['n_frames']} frames")

    artifact = {"fold_index": fold_index, "test_id": test_id,
                # the fitted filter itself, so shape descriptors and figures need no refit
                "coefficients": np.asarray(estimator.coef_), "intercept": float(estimator.intercept_),
                "n_iter": int(estimator.n_iter_), "fit_seconds": final_fit_seconds,
                "quiet_metrics": encoding_metrics(eta_quiet, labels_quiet,
                                                  encoding["solver"]["calibration_steps"]),
                "selected": [feature_names[index] for index in selected],
                "selected_indices": selected, "path": path, "no_model": best_screened is not None,
                "best_screened": best_screened is not None,
                "quiet_score": quiet_score, "quiet_slope": quiet_slope, "quiet_null": quiet_null,
                "quiet_p": quiet_p, "quiet_at_floor": quiet_at_floor,
                "eta_vocal": vocal["eta"], "labels_vocal": vocal["labels"], "pointer": pointer,
                "gaps": gaps, "vocal_frames": vocal_frames, "screen": screen_rows,
                "fold_vocal_score": vocal["fold_score"], "fold_vocal_slope": vocal["fold_slope"],
                "auroc_vocal": vocal["auroc"], "spike_rate_vocal": vocal["spike_rate"],
                "final_fit_converged": final_converged, "vocal_everywhere": vocal_everywhere}
    return write_fold_artifact(artifact, unit, fold_index, output_directory, message_output)


def representative_fold(artifacts: list) -> int:
    """
    Description
    -----------
    Pick the fold whose model represents the unit: the median by held-out quiet D2.

    With an even number of model folds, the two middle folds are equidistant from the median, so the one
    whose held-out session has more transfer frames is taken. The choice uses only quiet scores and frame
    counts, never a vocal score.

    Parameters
    ----------
    artifacts (list)
        Per-fold artifacts, including the ones flagged ``no_model``.

    Returns
    -------
    fold_index (int)
        Index into ``artifacts`` of the representative fold, or ``-1`` when none has a model.
    """

    usable = [index for index, a in enumerate(artifacts) if not bool(a["no_model"])]
    if not usable:
        return -1
    ordered = sorted(usable, key=lambda index: float(artifacts[index]["quiet_score"]))
    if len(ordered) % 2 == 1:
        return ordered[len(ordered) // 2]
    lower, upper = ordered[len(ordered) // 2 - 1], ordered[len(ordered) // 2]
    return max((lower, upper), key=lambda index: int(artifacts[index]["vocal_frames"].size))

def per_fold_roster(artifacts: list, n_lags: int) -> dict:
    """
    Description
    -----------
    Everything the rotation produced per fold, as parallel lists indexed by fold, ready to persist.

    Every fold is listed, including folds that found no model, so position ``i`` in every list is fold
    ``i``, which holds out ``courtship_sessions[i]``. A no-model fold records its quiet p as 1.0, its
    fit-only entries as ``None`` (``nan`` for the fit time), and keeps its screen table.

    Every fold's filter is kept, reshaped to ``(n_features, n_lags)`` with its feature names and held-out
    session, since folds select different features and filters cannot be recovered without refitting.

    Parameters
    ----------
    artifacts (list)
        Per-fold artifacts in fold order, as loaded from the fold files.
    n_lags (int)
        History length in frames.

    Returns
    -------
    roster (dict)
        ``no_model_per_fold``, ``selected_per_fold``, ``screen_per_fold``, ``forward_path_per_fold``,
        ``quiet_score_per_fold``, ``quiet_p_per_fold``, ``quiet_metrics_per_fold``,
        ``fit_seconds_per_fold``, ``n_iter_per_fold`` and ``filters_per_fold``, each a list with one
        entry per fold. A ``filters_per_fold`` entry carries ``fold_index``, ``test_session``,
        ``features``, ``feature_indices``, ``filter`` and ``intercept``.
    """

    roster = {key: [] for key in ("no_model_per_fold", "selected_per_fold", "screen_per_fold",
                                  "forward_path_per_fold", "quiet_score_per_fold", "quiet_p_per_fold",
                                  "quiet_metrics_per_fold", "fit_seconds_per_fold", "n_iter_per_fold",
                                  "filters_per_fold")}
    for artifact in artifacts:
        no_model = bool(artifact["no_model"])
        fold = {"fold_index": int(artifact["fold_index"]), "test_session": str(artifact["test_id"])}
        roster["no_model_per_fold"].append(no_model)
        roster["selected_per_fold"].append([str(name) for name in artifact["selected"]])
        roster["screen_per_fold"].append(list(artifact["screen"]))
        roster["forward_path_per_fold"].append(list(artifact["path"]))
        roster["quiet_score_per_fold"].append(float(artifact["quiet_score"]))
        if no_model:
            roster["quiet_p_per_fold"].append(1.0)
            roster["quiet_metrics_per_fold"].append(None)
            roster["fit_seconds_per_fold"].append(float("nan"))
            roster["n_iter_per_fold"].append(None)
            roster["filters_per_fold"].append({**fold, "features": [], "feature_indices": [],
                                               "filter": None, "intercept": None})
            continue
        indices = [int(index) for index in artifact["selected_indices"]]
        roster["quiet_p_per_fold"].append(float(artifact["quiet_p"]))
        roster["quiet_metrics_per_fold"].append(dict(unwrap_stored(artifact["quiet_metrics"])))
        roster["fit_seconds_per_fold"].append(float(artifact["fit_seconds"]))
        roster["n_iter_per_fold"].append(int(artifact["n_iter"]))
        roster["filters_per_fold"].append(
            {**fold, "features": [str(name) for name in artifact["selected"]], "feature_indices": indices,
             "filter": np.asarray(artifact["coefficients"], dtype=np.float64).reshape(len(indices), n_lags),
             "intercept": float(artifact["intercept"])})
    return roster


def run_single(unit: dict, settings: dict, data_root: str, output_directory: str,
               message_output=print) -> dict:
    """
    Description
    -----------
    Select once, fit once, and report ONE model for the unit.

    The single-split alternative to the per-fold rotation. Fitting and selection use quiet anchors
    only, so every vocal frame of the unit stays held out for the transfer and only the quiet half needs
    a reserved session:

        select + fit   quiet anchors of all sessions but one   -> one feature set, one weight vector
        score quiet    the held-out session's quiet anchors
        score vocal    every vocal session's transfer frames, pooled

    The held-out session is, excluding the session with the fewest spikes, the one with the most quiet
    anchors; both criteria are counts, never scores.

    Parameters
    ----------
    unit (dict)
        ``unit_id``, ``mouse_id``, ``rec_date``, ``courtship_sessions``, ``vocal_sessions``.
    settings (dict)
        Full neural-modeling settings.
    data_root (str)
        Root of the session data tree.
    output_directory (str)
        Where the artifact is written.
    message_output (Callable)
        Where progress is reported.

    Returns
    -------
    result (dict)
        The unit's single-model result, also written as the ``kinematic_encoding`` section of the
        unit file (``write_unit_section``), including when no model is found (``no_model`` True,
        quiet p 1.0, transfer p NaN).
    """

    encoding = settings["kinematic_encoding"]
    sessions = unit["courtship_sessions"]
    per_session = assemble_unit_sessions(unit, data_root, settings["kinematic_features"],
                                         encoding["history_pre_seconds"], encoding["clean_post_seconds"],
                                         settings["vocalization_settings"]["clean_against"],
                                         settings["vocalization_settings"])
    fps = per_session[sessions[0]]["fps"]
    n_lags = int(np.floor(encoding["history_pre_seconds"] * fps))
    feature_names = list(per_session[sessions[0]]["feature_names"])

    # Held-out session: never the sparsest train; among the rest, the most quiet anchors.
    spike_counts = {name: int(per_session[name]["spike_frames"].size) for name in sessions}
    sparsest = min(sessions, key=lambda name: spike_counts[name])
    candidates = [name for name in sessions if name != sparsest] or list(sessions)
    held_out = max(candidates, key=lambda name: per_session[name]["quiet"].size)
    fit_ids = [name for name in sessions if name != held_out]
    message_output(f"[single] quiet held-out = {held_out} ({per_session[held_out]['quiet'].size} "
                   f"anchors, {spike_counts[held_out]} spikes; sparsest is {sparsest} with "
                   f"{spike_counts[sparsest]}) | select+fit on {fit_ids}")

    # With a single session left to fit on there is no second session to hold out, so the inner split
    # comes from contiguous time blocks within it. The FINAL fit still uses the whole session.
    inner_sessions, inner_ids = per_session, fit_ids
    if len(fit_ids) == 1:
        # The gap equals the predictor history; each block is trimmed by half of it at both ends.
        required_gap = float(encoding["history_pre_seconds"])
        gap_frames = int(np.ceil(required_gap * fps / 2.0))
        blocks = quiet_block_sessions(per_session[fit_ids[0]], fit_ids[0],
                                      settings["data_sufficiency"]["single_session_inner_split_blocks"],
                                      gap_frames)
        if len(blocks) < 2:
            message_output(f"    NO MODEL: {fit_ids[0]} yields {len(blocks)} usable quiet blocks")
            # Written, not only returned, so an untuned unit appears with quiet p 1.0.
            result = {"unit_id": unit["unit_id"], "no_model": True, "quiet_p": 1.0, "transfer_p": np.nan,
                      "held_out": held_out, "fit_sessions": fit_ids, "spike_counts": spike_counts,
                      "sparsest_session": sparsest, "inner_split": "blocks", "screen": None,
                      "n_lags": n_lags, "fps": fps}
            written = write_unit_section(output_directory, unit, "kinematic_encoding", result, settings)
            message_output(f"    wrote {written.name} [kinematic_encoding, single-split, no model]")
            return result
        inner_sessions = {**per_session, **blocks}
        inner_ids = list(blocks)
        message_output(f"    one fit session: inner split is {len(blocks)} contiguous quiet blocks "
                       f"with a {required_gap:g} s gap (>= history {encoding['history_pre_seconds']:g} s)")

    rng = np.random.default_rng(crc32(unit["unit_id"].encode()) & 0xFFFFFFFF)
    screen_rows = screen_features(inner_sessions, inner_ids, n_lags, rng, settings, feature_names,
                                  message_output)
    survivors = [row["feature"] for row in screen_rows if row["survived"]]
    message_output(f"    survivors: {len(survivors)}/{len(feature_names)}")
    if not survivors:
        message_output("    NO MODEL: nothing survived the screen")
        # Written with the screen table, as in `combine`.
        result = {"unit_id": unit["unit_id"], "no_model": True, "quiet_p": 1.0, "transfer_p": np.nan,
                  "held_out": held_out, "fit_sessions": fit_ids, "spike_counts": spike_counts,
                  "sparsest_session": sparsest, "inner_split": "blocks" if len(fit_ids) == 1 else "sessions",
                  "screen": screen_rows, "n_lags": n_lags, "fps": fps}
        written = write_unit_section(output_directory, unit, "kinematic_encoding", result, settings)
        message_output(f"    wrote {written.name} [kinematic_encoding, single-split, no model]")
        return result

    selected, path = forward_select(inner_sessions, inner_ids, survivors, n_lags, rng, settings,
                                    feature_names, message_output)
    estimator, base_rate = fit_quiet_model(per_session, fit_ids, selected, n_lags, rng, encoding,
                                           message_output)
    final_converged = bool(estimator.converged_)
    message_output(f"    MODEL: {[feature_names[index] for index in selected]}")

    session = per_session[held_out]
    eta_quiet = linear_predictor_at_frames(estimator, session["feature_time_series"], selected,
                                           session["quiet"], n_lags, encoding["chunk_rows"])
    labels_quiet = spike_labels_at_frames(session["spike_frames"], session["quiet"],
                                          session["n_frames"])
    quiet_score, quiet_slope = frozen_model_score(eta_quiet, labels_quiet,
                                                  np.zeros(labels_quiet.size, dtype=np.int64),
                                                  encoding["solver"]["calibration_steps"], "bernoulli", 1.0,
                                                  encoding["calibration"])
    def draw_quiet_null(count):
        """Fresh quiet-null draws for the escalation ladder; the generator carries on where it left off."""
        return frozen_null_scores(eta_quiet, session["quiet"], session["spike_frames"],
                                  session["n_frames"], fps, count, rng,
                                  settings["null"]["shuffle_guard_seconds"],
                                  encoding["solver"]["calibration_steps"],
                                  encoding["solver"]["null_calibration_bins"], encoding["calibration"],
                                  encoding["null_block_megabytes"])

    quiet_null = draw_quiet_null(settings["null"]["n_shuffles"])
    quiet_p, quiet_at_floor, quiet_null = escalated_empirical_pvalue(
        quiet_score, draw_quiet_null, settings["significance"]["escalation_ladder"], quiet_null,
        message_output)
    message_output(f"  QUIET  {held_out}: score {quiet_score:+.5f} (slope {quiet_slope:+.3f}) "
                   f"| p {quiet_p:.4e}{' (at floor)' if quiet_at_floor else ''}")

    # Every vocal session's transfer frames are held out; only vocal sessions are scored.
    vocal_results, vocal_frames_by_session = [], {}
    vocal_ids = [s for s in sessions if s in set(unit["vocal_sessions"])]
    transfer_settings = settings["kinematic_encoding"]["transfer"]
    for session_id in vocal_ids:
        frames, pointer, gaps = focal_vocal_frames(per_session[session_id], session_id, data_root,
                                                   unit["mouse_id"], fps, n_lags,
                                                   settings["vocalization_settings"])
        frames = transfer_frames(frames, pointer, fps, per_session[session_id]["n_frames"], n_lags,
                                 per_session[session_id]["quiet"], transfer_settings)
        vocal_frames_by_session[session_id] = frames
        result = score_fold(estimator, per_session[session_id], selected, frames, n_lags, base_rate,
                            encoding, linear_predictor_at_frames, transfer_settings)
        result["pointer"], result["gaps"] = pointer, gaps
        vocal_results.append(result)
    transfer = combine_folds(vocal_results, per_session, vocal_ids, vocal_frames_by_session, settings,
                             message_output)

    if not final_converged:
        message_output("    [!] FINAL MODEL DID NOT CONVERGE -- treat every number here as provisional")
    # Printed, not stored: an uncorrected reading against a bare threshold (see `combine`).
    level = float(settings["significance"]["fdr_q"])
    quiet_significant = quiet_p < level
    transfer_significant = transfer["p"] < level
    if quiet_significant and transfer_significant and transfer["slope"] > 0:
        reading = "transformation candidate"
    elif quiet_significant and transfer_significant:
        reading = "regime-switching (tuned in both regimes, opposite sign)"
    elif quiet_significant:
        reading = "purely kinematic (no transfer)"
    else:
        reading = "not tuned on quiet"
    message_output(f"  UNCORRECTED READING (not persisted): {reading} | quiet p {quiet_p:.4e} "
                   f"| transfer p {transfer['p']:.4e} slope {transfer['slope']:+.3f} "
                   f"| bare threshold {level}, no FDR")

    artifact = {"unit_id": unit["unit_id"], "no_model": False, "held_out": held_out,
                "fit_sessions": fit_ids, "spike_counts": spike_counts, "sparsest_session": sparsest,
                "inner_split": "blocks" if len(fit_ids) == 1 else "sessions", "selected": [feature_names[index] for index in selected],
                "selected_indices": selected, "path": path, "screen": screen_rows,
                "quiet_score": quiet_score, "quiet_slope": quiet_slope, "quiet_p": quiet_p,
                "quiet_at_floor": quiet_at_floor, "quiet_null": quiet_null,
                "transfer_score": transfer["score"], "transfer_slope": transfer["slope"],
                "transfer_p": transfer["p"], "transfer_at_floor": transfer["at_floor"],
                "transfer_null": transfer["null"], "transfer_folds": transfer["folds"],
                "transfer_n_bins": transfer["n_bins"], "transfer_bin_width_frames": transfer["bin_width_frames"],
                "transfer_grain": transfer["grain"], "transfer_universe": transfer["universe"],
                "transfer_leave_one_session_out": transfer["leave_one_session_out"],
                "transfer_min_leave_one_session_out": transfer["min_leave_one_session_out"],
                "n_vocal_frames": transfer["n_frames"], "final_fit_converged": final_converged,
                # the fitted filter itself, feature-major with n_lags weights per feature, so a scorer
                # can carry this exact model to another universe without refitting it
                "filter": np.asarray(estimator.coef_, dtype=np.float64),
                "filter_intercept": float(estimator.intercept_),
                "n_lags": n_lags, "fps": fps}
    written = write_unit_section(output_directory, unit, "kinematic_encoding", artifact, settings)
    message_output(f"    wrote {written.name} [kinematic_encoding, single-split]")
    return artifact

def combine(unit: dict, settings: dict, data_root: str, output_directory: str,
            message_output=print) -> dict:
    """
    Description
    -----------
    Combine the fold artifacts into the unit's kinematic encoding result.

    The representative fold supplies the tested quiet score and the model that travels; that model's
    predictions on every vocal session are pooled into the transfer and tested. The
    filter band, shape descriptors and spike-triggered average are computed here and stored with the
    per-fold roster in the unit's ``kinematic_encoding`` section. A unit whose every fold failed the
    screen is written as ``no_model``.

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
        The written section: both halves' scores, p-values, nulls and floor flags, the transfer
        descriptors, the representative fold and its model, the filter band, descriptors and STA, and
        the per-fold roster of :func:`per_fold_roster`.
    """

    sessions = unit["courtship_sessions"]
    encoding = settings["kinematic_encoding"]
    artifacts = []
    for fold_index in range(len(sessions)):
        path = pathlib.Path(output_directory) / f"{unit['unit_id']}_fold{fold_index}.npz"
        if not path.exists():
            msg = f"fold artifact missing: {path}"
            raise FileNotFoundError(msg)
        artifacts.append(dict(np.load(path, allow_pickle=True)))

    # When every fold failed the screen but carries a best screened feature, those are pooled for
    # description only; the written section keeps `no_model` True.
    screened_only = bool(artifacts) and all(bool(artifact["no_model"]) for artifact in artifacts) and all(
        "best_screened" in artifact and bool(artifact["best_screened"]) for artifact in artifacts)
    if screened_only:
        message_output("  NO FOLD SELECTED A MODEL -- pooling each fold's best SCREENED feature, for "
                       "description only; these numbers are not a test")
        for artifact in artifacts:
            artifact["no_model"] = np.asarray(False)

    per_session = assemble_unit_sessions(unit, data_root, settings["kinematic_features"],
                                         encoding["history_pre_seconds"], encoding["clean_post_seconds"],
                                         settings["vocalization_settings"]["clean_against"],
                                         settings["vocalization_settings"])
    fps = per_session[sessions[0]]["fps"]
    n_lags = int(np.floor(encoding["history_pre_seconds"] * fps))
    feature_names = list(per_session[sessions[0]]["feature_names"])

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
        # Written, not only returned: quiet p 1.0, transfer p NaN and no_model, with the per-fold
        # screen tables, so an untuned unit is never a missing value in the cohort table.
        result = {"unit_id": unit["unit_id"], "no_model": True, "quiet_p": 1.0, "transfer_p": np.nan,
                  "n_lags": n_lags, "fps": fps, **per_fold_roster(artifacts, n_lags)}
        written = write_unit_section(output_directory, unit, "kinematic_encoding", result, settings)
        message_output(f"    wrote {written.name} [kinematic_encoding, no model]")
        return result

    quiet_scores = [float(a["quiet_score"]) for a in artifacts if not bool(a["no_model"])]
    # Both halves describe the representative fold's model; the fold-average quiet score is a
    # descriptor only.
    chosen = representative_fold(artifacts)
    chosen_artifact = artifacts[chosen]
    quiet_score = float(chosen_artifact["quiet_score"])
    quiet_p = float(chosen_artifact["quiet_p"])
    quiet_at_floor = bool(chosen_artifact["quiet_at_floor"])
    quiet_null = np.asarray(chosen_artifact["quiet_null"])
    quiet_mean_across_folds = float(np.nanmean(quiet_scores))
    message_output(f"  QUIET  score {quiet_score:+.5f} | p {quiet_p:.4e}"
                   f"{' (at floor)' if quiet_at_floor else ''} | {quiet_null.size:,} draws "
                   f"| fold-average {quiet_mean_across_folds:+.5f} (descriptor)")

    # ONE model travels: the representative fold's, scored on every session's vocal frames.
    everywhere = list(chosen_artifact["vocal_everywhere"])
    message_output(f"  REPRESENTATIVE fold {int(chosen_artifact['fold_index'])} "
                   f"({chosen_artifact['test_id']!s}): quiet {float(chosen_artifact['quiet_score']):+.5f} "
                   f"| model {list(chosen_artifact['selected'])}")
    scored_ids = [str(entry["session_id"]) for entry in everywhere]
    vocal_frames_by_session = {str(entry["session_id"]): entry["frames"] for entry in everywhere}
    fold_results = [{key: value for key, value in entry.items()
                     if key not in ("session_id", "frames")} for entry in everywhere]
    transfer = combine_folds(fold_results, per_session, scored_ids, vocal_frames_by_session, settings,
                             message_output)

    # The filter band needs refits of the representative model, so it is computed and stored here.
    band = None
    band_settings = encoding["filter_band"]
    representative_features = [int(f) for f in chosen_artifact["selected_indices"]]
    if band_settings["compute"] and representative_features:
        representative_id = str(chosen_artifact["test_id"])
        band = filter_band(per_session,
                           [s for s in sessions if s != representative_id],
                           representative_features, n_lags,
                           np.random.default_rng(settings["null"]["shuffle_seed"] + 977),
                           encoding, band_settings["n_resamples"], message_output)

    # A reading of these numbers is printed for a single unit and not stored: it compares raw p-values
    # to a bare threshold, and correction needs the cohort.
    level = float(settings["significance"]["fdr_q"])
    quiet_significant = quiet_p < level
    transfer_significant = transfer["p"] < level
    if quiet_significant and transfer_significant and transfer["slope"] > 0:
        reading = "transformation candidate"
    elif quiet_significant and transfer_significant:
        reading = "regime-switching (tuned in both regimes, opposite sign)"
    elif quiet_significant:
        reading = "purely kinematic (no transfer)"
    else:
        reading = "not tuned on quiet"
    if screened_only:
        reading = f"not tuned on quiet -- best SCREENED feature only, rejected by the screen ({reading})"
    message_output(f"  UNCORRECTED READING (not persisted): {reading} | quiet p {quiet_p:.4e} "
                   f"| transfer p {transfer['p']:.4e} slope {transfer['slope']:+.3f} "
                   f"| bare threshold {level}, no FDR")

    # Shape descriptors of the stored filter and the spike-triggered average.
    descriptors = filter_descriptors(np.asarray(chosen_artifact["coefficients"]),
                                     [feature_names[i] for i in representative_features],
                                     n_lags, fps) if representative_features else []
    sta = (spike_triggered_average(per_session,
                                   [s for s in sessions if s != str(chosen_artifact["test_id"])],
                                   representative_features, n_lags, encoding["chunk_rows"])
           if representative_features else None)

    result = {"unit_id": unit["unit_id"], "no_model": screened_only,
              "best_screened_only": screened_only,
              "quiet_score": quiet_score, "quiet_slope": float(chosen_artifact["quiet_slope"]),
              "quiet_p": quiet_p, "quiet_at_floor": quiet_at_floor, "quiet_null": quiet_null,
              "quiet_score_fold_average": quiet_mean_across_folds,
              "transfer_score": transfer["score"], "transfer_slope": transfer["slope"],
              "transfer_p": transfer["p"], "transfer_at_floor": transfer["at_floor"],
              "transfer_null": transfer["null"], "transfer_n_frames": transfer["n_frames"],
              "transfer_n_bins": transfer["n_bins"], "transfer_bin_width_frames": transfer["bin_width_frames"],
              "transfer_grain": transfer["grain"], "transfer_universe": transfer["universe"],
              "transfer_leave_one_session_out": transfer["leave_one_session_out"],
              "transfer_min_leave_one_session_out": transfer["min_leave_one_session_out"],
              "transfer_per_session": fold_results, "transfer_session_ids": scored_ids,
              "n_lags": n_lags, "fps": fps,
              "representative_fold": int(chosen_artifact["fold_index"]),
              "representative_session": str(chosen_artifact["test_id"]),
              "selected": [str(f) for f in chosen_artifact["selected"]],
              "selected_indices": representative_features,
              "filter_band": band,
              "filter": np.asarray(chosen_artifact["coefficients"]),
              "filter_intercept": float(chosen_artifact["intercept"]),
              "filter_descriptors": descriptors, "spike_triggered_average": sta,
              "quiet_metrics": dict(unwrap_stored(chosen_artifact["quiet_metrics"])),
              **per_fold_roster(artifacts, n_lags)}

    # Merged into the unit's own file, which the other analyses write their own sections into.
    written = write_unit_section(output_directory, unit, "kinematic_encoding", result, settings)
    message_output(f"    wrote {written.name} [kinematic_encoding]")
    return result


def build_unit_record(unit_id: str, mouse_id: str, rec_date: int, sessions: list, data_root: str,
                      settings: dict) -> dict:
    """
    Description
    -----------
    The unit's record as every dispatcher sees it: identity, both session sets, and the per-session USV
    counts the vocal gate was applied on.

    Every dispatcher uses this one builder, because each section write stamps the record into the
    unit's file as its ``identity`` block. The vocal session set is derived here with the cohort
    builder's gate, and governs only what is scored on vocalizations.

    Parameters
    ----------
    unit_id (str)
        Cluster id, e.g. ``imec1_cl0401_ch220_good``.
    mouse_id (str)
        The recorded animal.
    rec_date (int)
        Recording date, ``YYYYMMDD``.
    sessions (list)
        The unit's courtship sessions.
    data_root (str)
        The ``Data`` root.
    settings (dict)
        Full neural-modeling settings.

    Returns
    -------
    unit (dict)
        ``unit_uid``, ``mouse_id``, ``rec_date``, ``unit_id``, ``courtship_sessions``, ``vocal_sessions``
        and ``emitter_usvs_per_session``.
    """

    counts = emitter_usv_counts(sessions, data_root, dict.fromkeys(sessions, mouse_id),
                                settings["vocalization_settings"]["vocal_emitter"],
                                settings["vocalization_settings"])
    threshold = settings["data_sufficiency"]["min_emitter_usvs_per_session"]
    return {"unit_uid": f"{mouse_id}_{rec_date}_{unit_id}", "mouse_id": mouse_id,
            "rec_date": rec_date, "unit_id": unit_id, "courtship_sessions": list(sessions),
            "vocal_sessions": [s for s in sessions if counts[s] >= threshold],
            "emitter_usvs_per_session": counts}


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
    unit = build_unit_record(args.unit_id, args.mouse_id, args.rec_date, args.sessions, args.data_root,
                             settings)
    started = datetime.now()
    try:
        if args.single:
            run_single(unit, settings, args.data_root, args.output_directory)
        elif args.combine:
            combine(unit, settings, args.data_root, args.output_directory)
        else:
            run_fold(unit, args.fold, settings, args.data_root, args.output_directory)
    except Exception:
        traceback.print_exc()
        return 1
    # bare print rather than an injected `message_output`: this is the array task's own exit
    # line, written after the run object it would have been threaded through is gone.
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
    group.add_argument("--single", action="store_true")
    return dispatch(parser.parse_args())


if __name__ == "__main__":
    sys.exit(main())
