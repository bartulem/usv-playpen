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

from .kinematic_encoding import (fit_quiet_model, forward_select, frozen_null_scores,
                                 inner_folds, linear_predictor_at_frames,
                                 quiet_block_sessions, screen_features)
from .neural_design_assembly import (assemble_unit_sessions, emitter_name, load_session_usvs,
                                     silent_gap_before_call, spike_labels_at_frames,
                                     vocal_span_frames)
from .deviance_metrics import calibrated_explained_deviance
from .quiet_to_vocal_transfer import combine_folds, score_fold
from .shift_null_inference import empirical_pvalue, escalated_empirical_pvalue


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

    # A 2-session unit leaves a pool of ONE, and leave-one-pool-session-out then yields a fold with an
    # empty training set -- a crash, not a degradation. The inner split then comes from contiguous
    # quiet-anchor blocks within that single pool session, with a gap at least as long as the predictor
    # history so a validation frame's lags cannot reach into a training block. The model itself is still
    # fitted on the whole pool session.
    inner_sessions, inner_ids = per_session, pool_ids
    if len(pool_ids) == 1:
        required_gap = max(float(settings["cohort"]["block_cv_gap_seconds"]),
                           float(encoding["history_pre_seconds"]))
        blocks = quiet_block_sessions(per_session[pool_ids[0]], pool_ids[0],
                                      settings["cohort"]["block_cv_n_folds"],
                                      int(np.ceil(required_gap * fps / 2.0)))
        if len(blocks) < 2:
            message_output(f"    NO MODEL: {pool_ids[0]} yields {len(blocks)} usable quiet blocks")
            return {"fold_index": fold_index, "test_id": test_id, "selected": [], "path": [],
                    "no_model": True, "quiet_score": 0.0, "quiet_slope": np.nan,
                    "quiet_p": 1.0, "quiet_at_floor": False,
                    "eta_vocal": np.zeros(0), "labels_vocal": np.zeros(0), "pointer": pointer,
                    "gaps": gaps, "screen": [], "vocal_everywhere": []}
        inner_sessions, inner_ids = {**per_session, **blocks}, list(blocks)
        message_output(f"    one pool session: inner split is {len(blocks)} contiguous quiet blocks "
                       f"with a {required_gap:g} s gap")

    screen_rows = screen_features(inner_sessions, inner_ids, n_lags, rng, settings, feature_names,
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

    selected, path = forward_select(inner_sessions, inner_ids, survivors, n_lags, rng, settings,
                                    feature_names, message_output)
    estimator, base_rate = fit_quiet_model(per_session, pool_ids, selected, n_lags, rng, encoding,
                                           message_output)
    # The final model is what every reported number rests on, so its convergence travels with the
    # artifact rather than living only in a log line a cluster run may never have read back.
    final_converged = bool(estimator.converged_)

    eta_quiet = linear_predictor_at_frames(estimator, session["feature_time_series"], selected,
                                           session["quiet"], n_lags, encoding["chunk_rows"])
    labels_quiet = spike_labels_at_frames(session["spike_frames"], session["quiet"],
                                          session["n_frames"])
    quiet_score, quiet_slope = calibrated_explained_deviance(eta_quiet, labels_quiet,
                                                             encoding["solver"]["calibration_steps"])
    def draw_quiet_null(count):
        """Fresh quiet-null draws for the escalation ladder; the generator carries on where it left off."""
        return frozen_null_scores(eta_quiet, session["quiet"], session["spike_frames"],
                                  session["n_frames"], fps, count, rng,
                                  settings["null"]["shuffle_guard_seconds"],
                                  encoding["solver"]["calibration_steps"],
                                  encoding["solver"]["null_calibration_bins"])

    quiet_null = draw_quiet_null(settings["null"]["n_shuffles"])
    quiet_p, quiet_at_floor, quiet_null = escalated_empirical_pvalue(
        quiet_score, draw_quiet_null, settings["significance"]["escalation_ladder"], quiet_null,
        message_output)
    vocal = score_fold(estimator, session, selected, vocal_frames, n_lags, base_rate, encoding,
                       linear_predictor_at_frames)

    # This fold's model is also carried onto EVERY session's vocal frames, not just its own. The
    # transfer is a statement about ONE model, and pooling eta from a different model per fold would
    # mix six of them into a single calibration. Scoring every fold's model everywhere lets `combine`
    # take the representative fold's numbers, and lets the transfer conclusion be checked against the
    # other folds' models rather than resting on the choice. It is cheap -- no refit, just the linear
    # predictor and a two-parameter calibration -- and legitimate for every session, because quiet
    # anchors exclude vocal periods by construction, so the model has never seen a vocal frame.
    vocal_everywhere = []
    for other_id in sessions:
        other_frames, other_pointer, other_gaps = focal_vocal_frames(
            per_session[other_id], other_id, data_root, unit["mouse_id"], fps, n_lags)
        scored = score_fold(estimator, per_session[other_id], selected, other_frames, n_lags,
                            base_rate, encoding, linear_predictor_at_frames)
        scored["session_id"] = other_id
        scored["pointer"], scored["gaps"] = other_pointer, other_gaps
        scored["frames"] = other_frames
        vocal_everywhere.append(scored)
    if not final_converged:
        message_output(f"    [!] fold {fold_index} FINAL MODEL DID NOT CONVERGE -- treat both halves "
                       f"of this fold as provisional")
    # The quiet half of this line can become the tested number, if this fold is the representative.
    # The vocal half NEVER is: the tested transfer is the representative model pooled across every
    # session, computed in `combine`. Labelled so the two are not read as equivalent.
    message_output(f"    TEST {test_id}: quiet {quiet_score:+.5f} (slope {quiet_slope:+.3f}) "
                   f"p {quiet_p:.3e} | [diagnostic] own-session vocal {vocal['fold_score']:+.5f} "
                   f"(slope {vocal['fold_slope']:+.3f}) on {vocal['n_frames']} frames")

    artifact = {"fold_index": fold_index, "test_id": test_id,
                "selected": [feature_names[index] for index in selected],
                "selected_indices": selected, "path": path, "no_model": False,
                "quiet_score": quiet_score, "quiet_slope": quiet_slope, "quiet_null": quiet_null,
                "quiet_p": quiet_p, "quiet_at_floor": quiet_at_floor,
                "eta_vocal": vocal["eta"], "labels_vocal": vocal["labels"], "pointer": pointer,
                "gaps": gaps, "vocal_frames": vocal_frames, "screen": screen_rows,
                "fold_vocal_score": vocal["fold_score"], "fold_vocal_slope": vocal["fold_slope"],
                "auroc_vocal": vocal["auroc"], "spike_rate_vocal": vocal["spike_rate"],
                "final_fit_converged": final_converged, "vocal_everywhere": vocal_everywhere}
    destination = pathlib.Path(output_directory) / f"{unit['unit_id']}_fold{fold_index}.npz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez(destination, **{key: np.asarray(value, dtype=object) if isinstance(value, (list, dict))
                             else value for key, value in artifact.items()})
    message_output(f"    wrote {destination.name}")
    return artifact


def representative_fold(artifacts: list) -> int:
    """
    Description
    -----------
    Pick the fold whose model represents the unit: the MEDIAN by held-out quiet D2, tie-broken on vocal
    frame count.

    The transfer freezes one model and carries it to vocal periods, so exactly one of the folds' models
    has to be the one that travels. The median is chosen over the best because the best is a maximum of
    `n` draws and would need the null to take a maximum too; the median asks less of the null and is the
    more conservative summary of a spread that runs 2.3-2.4x across sessions on the pilot units.

    An even fold count has no middle element, and the two straddling folds are EQUIDISTANT from the median
    value by construction -- the median IS their midpoint -- so "closest to the median" can never separate
    them. The tie is therefore broken on the count of focal vocal frames in the fold's held-out session,
    taking the larger. That is outcome-blind, always resolves, and systematically avoids thin sessions
    rather than landing on one by chance: on the VTA pilot it takes the 10,695-frame fold over the
    4,745-frame fold, and the thin one was the single fold whose transfer disagreed with the other five.

    Choosing on the QUIET side keeps the transfer honest. Vocal frames play no part in the fit, in
    selection, or in this choice, so the chosen model's vocal score remains an ordinary held-out number.

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

def run_single(unit: dict, settings: dict, data_root: str, output_directory: str,
               message_output=print) -> dict:
    """
    Description
    -----------
    Select once, fit once, and report ONE model for the unit.

    The per-fold rotation gives an honest score but a different feature set per fold, because selection
    re-runs on a different pool each time. Correlated kinematic features then trade places between folds
    and the unit has no single interpretable model -- while the transfer, which freezes the SELECTED model
    and carries it to vocal periods, is left testing a different object on every fold.

    One set and honest scoring look mutually exclusive under pure rotation: to score session *i* the
    feature set must not depend on *i*, so a set common to all sessions can depend on none of them. The
    way out is that the two halves of claim 1 are scored on DISJOINT frames. A quiet anchor admits no USV
    from any emitter in ``[t - history_pre_seconds, t + clean_post_seconds]``, so its entire history is
    vocalization-free, while vocal frames lie inside focal calls. Fitting and selection touch quiet
    anchors only. **No vocal frame from any session enters the model at any stage**, so every vocal frame
    in the unit is held out for the transfer regardless of which sessions were fitted.

    So only the QUIET half needs a reserved session, and the transfer keeps every call the unit produced:

        select + fit   quiet anchors of all sessions but one   -> one feature set, one weight vector
        score quiet    the held-out session's quiet anchors
        score vocal    every session's vocal frames, pooled

    The held-out session is the one with the most quiet anchors, which is the most powerful place to
    measure the quiet statistic and depends only on frame counts -- never on spikes or on any score.

    Stated for the methods rather than glossed: a session contributing quiet anchors to the fit also
    contributes vocal frames to the transfer. That overlap is at the level of the session, not the frame,
    and the transfer calibration already carries per-session intercepts that absorb session-level rate
    differences.

    Parameters
    ----------
    unit (dict)
        ``unit_id``, ``mouse_id``, ``rec_date``, ``courtship_sessions``.
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
        The unit's single-model result, also written as ``<unit_id>_single.npz``.
    """

    encoding = settings["kinematic_encoding"]
    sessions = unit["courtship_sessions"]
    per_session = assemble_unit_sessions(unit, data_root, settings["kinematic_features"],
                                         encoding["model_predictor_mouse_index"],
                                         encoding["history_pre_seconds"], encoding["clean_post_seconds"])
    fps = per_session[sessions[0]]["fps"]
    n_lags = int(np.floor(encoding["history_pre_seconds"] * fps))
    feature_names = list(per_session[sessions[0]]["feature_names"])

    # The scoring session is never the one with the fewest spikes: the quiet statistic is measured
    # there, and the sparsest train is where it is least well estimated. Among the rest, take the one
    # with the most quiet anchors. Both criteria are counts, so neither looks at a score.
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
        # The gap must exceed the predictor history, or a validation frame's 4 s of lags reaches back
        # into a training block and the split leaks. Trimming happens at BOTH ends of every block, so
        # each end takes half the required separation and the realised gap is what was asked for --
        # previously the trim was applied whole at each end, making the gap silently twice the setting.
        required_gap = max(float(settings["cohort"]["block_cv_gap_seconds"]),
                           float(encoding["history_pre_seconds"]))
        gap_frames = int(np.ceil(required_gap * fps / 2.0))
        blocks = quiet_block_sessions(per_session[fit_ids[0]], fit_ids[0],
                                      settings["cohort"]["block_cv_n_folds"], gap_frames)
        if len(blocks) < 2:
            message_output(f"    NO MODEL: {fit_ids[0]} yields {len(blocks)} usable quiet blocks")
            return {"unit_id": unit["unit_id"], "no_model": True, "quiet_p": 1.0,
                    "transfer_p": np.nan}
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
        return {"unit_id": unit["unit_id"], "no_model": True, "quiet_p": 1.0, "transfer_p": np.nan}

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
    quiet_score, quiet_slope = calibrated_explained_deviance(eta_quiet, labels_quiet,
                                                             encoding["solver"]["calibration_steps"])
    def draw_quiet_null(count):
        """Fresh quiet-null draws for the escalation ladder; the generator carries on where it left off."""
        return frozen_null_scores(eta_quiet, session["quiet"], session["spike_frames"],
                                  session["n_frames"], fps, count, rng,
                                  settings["null"]["shuffle_guard_seconds"],
                                  encoding["solver"]["calibration_steps"],
                                  encoding["solver"]["null_calibration_bins"])

    quiet_null = draw_quiet_null(settings["null"]["n_shuffles"])
    quiet_p, quiet_at_floor, quiet_null = escalated_empirical_pvalue(
        quiet_score, draw_quiet_null, settings["significance"]["escalation_ladder"], quiet_null,
        message_output)
    message_output(f"  QUIET  {held_out}: score {quiet_score:+.5f} (slope {quiet_slope:+.3f}) "
                   f"| p {quiet_p:.4e}{' (at floor)' if quiet_at_floor else ''}")

    # Every session's vocal frames are held out -- the model was fitted on quiet anchors only.
    vocal_results, vocal_frames_by_session = [], {}
    for session_id in sessions:
        frames, pointer, gaps = focal_vocal_frames(per_session[session_id], session_id, data_root,
                                                   unit["mouse_id"], fps, n_lags)
        vocal_frames_by_session[session_id] = frames
        result = score_fold(estimator, per_session[session_id], selected, frames, n_lags, base_rate,
                            encoding, linear_predictor_at_frames)
        result["pointer"], result["gaps"] = pointer, gaps
        vocal_results.append(result)
    transfer = combine_folds(vocal_results, per_session, sessions, vocal_frames_by_session, settings,
                             message_output)

    if not final_converged:
        message_output("    [!] FINAL MODEL DID NOT CONVERGE -- treat every number here as provisional")
    # THREE-WAY verdict (RULED 2026-09-05, user). A transformation neuron does not switch tuning
    # between regimes, so a filter that is significantly ANTI-predictive during vocalization is not a
    # failed transformation neuron -- it is a different kind of neuron. Calibrated D2 grows with the
    # square of the calibration slope and is blind to sign, so the slope is what separates them:
    # without it cl0499 reads as a clean pass on p-values alone while being anti-predictive on all six
    # of its sessions (slopes -0.088 to -0.565, every AUROC below 0.5).
    level = float(settings["significance"]["fdr_q"])
    quiet_significant = quiet_p < level
    transfer_significant = transfer["p"] < level
    if quiet_significant and transfer_significant and transfer["slope"] > 0:
        verdict = "TRANSFORMATION CANDIDATE"
    elif quiet_significant and transfer_significant:
        verdict = "REGIME-SWITCHING (tuned in both regimes, opposite sign)"
    elif quiet_significant:
        verdict = "PURELY KINEMATIC (no transfer)"
    else:
        verdict = "NOT TUNED on quiet"
    message_output(f"  VERDICT: {verdict} | quiet p {quiet_p:.4e} | transfer p {transfer['p']:.4e} "
                   f"slope {transfer['slope']:+.3f} | q = {level}")

    artifact = {"unit_id": unit["unit_id"], "no_model": False, "held_out": held_out,
                "fit_sessions": fit_ids, "spike_counts": spike_counts, "sparsest_session": sparsest,
                "inner_split": "blocks" if len(fit_ids) == 1 else "sessions", "selected": [feature_names[index] for index in selected],
                "selected_indices": selected, "path": path, "screen": screen_rows,
                "quiet_score": quiet_score, "quiet_slope": quiet_slope, "quiet_p": quiet_p,
                "quiet_at_floor": quiet_at_floor, "quiet_null": quiet_null,
                "verdict": verdict,
                "transfer_score": transfer["score"], "transfer_slope": transfer["slope"],
                "transfer_p": transfer["p"], "transfer_at_floor": transfer["at_floor"],
                "transfer_null": transfer["null"], "transfer_folds": transfer["folds"],
                "n_vocal_frames": transfer["n_frames"], "final_fit_converged": final_converged,
                "n_lags": n_lags, "fps": fps}
    destination = pathlib.Path(output_directory) / f"{unit['unit_id']}_single.npz"
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
    # Both halves must describe the SAME model, or `max(p_quiet, p_transfer)` conjoins p-values about
    # two different objects. The transfer carries the representative fold's model, so the quiet
    # statistic is that fold's too. The fold-average survives as a DESCRIPTOR -- the honest summary of
    # a spread running 2.3-2.4x across sessions on the pilot units -- but it is not the tested number.
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
                   f"({str(chosen_artifact['test_id'])}): quiet {float(chosen_artifact['quiet_score']):+.5f} "
                   f"| model {list(chosen_artifact['selected'])}")
    scored_ids = [str(entry["session_id"]) for entry in everywhere]
    vocal_frames_by_session = {str(entry["session_id"]): entry["frames"] for entry in everywhere}
    fold_results = [{key: value for key, value in entry.items()
                     if key not in ("session_id", "frames")} for entry in everywhere]
    transfer = combine_folds(fold_results, per_session, scored_ids, vocal_frames_by_session, settings,
                             message_output)
    # THREE-WAY verdict (RULED 2026-09-05, user). A transformation neuron does not switch tuning
    # between regimes, so a filter that is significantly ANTI-predictive during vocalization is not a
    # failed transformation neuron -- it is a different kind of neuron. Calibrated D2 grows with the
    # square of the calibration slope and is blind to sign, so the slope is what separates them:
    # without it cl0499 reads as a clean pass on p-values alone while being anti-predictive on all six
    # of its sessions (slopes -0.088 to -0.565, every AUROC below 0.5).
    level = float(settings["significance"]["fdr_q"])
    quiet_significant = quiet_p < level
    transfer_significant = transfer["p"] < level
    if quiet_significant and transfer_significant and transfer["slope"] > 0:
        verdict = "TRANSFORMATION CANDIDATE"
    elif quiet_significant and transfer_significant:
        verdict = "REGIME-SWITCHING (tuned in both regimes, opposite sign)"
    elif quiet_significant:
        verdict = "PURELY KINEMATIC (no transfer)"
    else:
        verdict = "NOT TUNED on quiet"
    message_output(f"  VERDICT: {verdict} | quiet p {quiet_p:.4e} | transfer p {transfer['p']:.4e} "
                   f"slope {transfer['slope']:+.3f} | q = {level}")
    return {"unit_id": unit["unit_id"], "no_model": False, "quiet_score": quiet_score,
            "quiet_p": quiet_p, "quiet_at_floor": quiet_at_floor, "verdict": verdict,
            "transfer_score": transfer["score"],
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
        if args.single:
            run_single(unit, settings, args.data_root, args.output_directory)
        elif args.combine:
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
    group.add_argument("--single", action="store_true")
    return dispatch(parser.parse_args())


if __name__ == "__main__":
    sys.exit(main())
