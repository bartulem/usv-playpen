"""
@author: bartulem
Carrying a model fitted on silence into vocal periods, and asking whether it still predicts.

The model is frozen: no refitting, no re-selection. It was fitted on the pool's quiet anchors and is scored
on the held-out session's vocal frames, which it has never seen -- quiet anchors exclude every vocal period
by construction, so the model has in fact never seen a vocal frame from any session at all.

Scoring is POOLED across folds rather than averaged over them, and that is not a cosmetic choice. Averaging
per-fold scores lets an unequal fold dominate: on a three-session unit whose folds carried 1,045, 962 and
372 events, fitting two calibration parameters against a shuffled train on the smallest fold overfitted so
badly that its null mean exceeded its own observed value and its spread swamped the average. Concatenating
the held-out predictions and calibrating once dropped the null mean from +0.0081 to +0.0019 and the null
spread from 0.0086 to 0.0028, moving the same data from p = 0.065 to p = 9.9e-06. Nothing about the
predictions changed; only how they were counted.

Pooling is also what makes the null resolvable. The null statistic is smooth in the shift lag -- its
autocorrelation is 0.997 at a lag difference of 0.1 s and reaches 1/e only at 2.2 s -- so a single session
carries about 528 effectively independent shifts and a per-fold p-value floors near 1/529 no matter how
many draws are taken. Pooling shifts every session independently, giving a joint space of
528^n_sessions, against which far more draws are genuinely distinct.

Pooling does hide something, and that is what the leave-one-session-out descriptor is for: a pooled
number cannot say whether one session carries it. The per-session scores reported alongside do not answer
that either, because each fits its own two calibration parameters on one session's frames -- the fragile
construction pooling exists to avoid. Dropping a session and recalibrating on the rest asks the question in
the well-conditioned way, at the cost of one two-parameter refit per session, and is what the vocalization-identity
and nested decoding analyses already record. It is a descriptor and gates nothing; consistency is not a
criterion any test here applies.

The gate is discrimination, not level: whether the model tells a frame with a spike from one without,
scored against the vocal frames' own rate. Whether the unit simply fires faster during vocalization is
recorded as a descriptor and gates nothing, because a model can predict the rate change perfectly while
ordering frames at chance.
"""

from __future__ import annotations

from functools import partial

import numpy as np

from .deviance_metrics import (
    area_under_roc,
    encoding_metrics,
    explained_deviance_vs_reference_rate,
    frozen_model_score,
    pooled_calibrated_explained_deviance,
)
from .neural_design_assembly import spike_labels_at_frames
from .shift_null_inference import (
    escalated_empirical_pvalue,
    sample_circular_shift,
    shifted_spike_frames,
)

# Memory budget for one block of pooled null draws; see pooled_transfer_null.
_POOLED_BLOCK_BYTES = 64 * 1024 * 1024
TRANSFER_UNIVERSES = ("usv_span", "bout_window")
TRANSFER_GRAINS = ("frame", "count")


def transfer_frames(span_frames: np.ndarray, pointer: np.ndarray, fps: float, n_frames: int, n_lags: int,
                    quiet_frames: np.ndarray, transfer_settings: dict) -> np.ndarray:
    """
    Description
    -----------
    The frames the transfer is scored on: the USVs' own spans, or the BOUT WINDOW around them.

    ``"usv_span"`` is the original universe -- every frame inside one of the recorded animal's USVs,
    ~6 % of a session, where these units sit at 0.25-0.6 spikes per 6.7 ms frame and the partner is
    always close. ``"bout_window"`` (ruled 2026-10-14) takes every bout -- consecutive USVs closer than
    ``inter_bout_threshold_seconds``, single-USV bouts included -- and scores ``[bout start -
    pre_bout_seconds, bout end]``, the union over bouts. Two exclusions are load-bearing: frames without a
    full history (the session's first ``n_lags`` frames), and every QUIET ANCHOR, because a frame two to
    four seconds before an isolated bout is quiet by the quiet rule's two-second forward look and would
    otherwise be both a fitting frame and a scoring frame.

    Measured on the nine 20250919 units: four times the frames, base rates 0.14-0.24 instead of
    0.25-0.41, the two passers 2x stronger, two distance+speed units from nothing to the floor, the
    pure-speed units positive in every session, the known-flat unit still null; on 20250927, four of five
    span-failures to the floor and nothing backwards. The window's cost is definitional and is to be
    stated: "live during vocal episodes including their run-up", not "inside the sounds".

    Parameters
    ----------
    span_frames (np.ndarray)
        Concatenated frames of the kept USV spans, as :func:`neural_design_assembly.vocal_span_frames`
        returns them.
    pointer (np.ndarray)
        Offsets of length ``n_usvs + 1`` into ``span_frames``.
    fps (float)
        Camera frame rate.
    n_frames (int)
        Session frame count.
    n_lags (int)
        History length in frames.
    quiet_frames (np.ndarray)
        The session's quiet anchors.
    transfer_settings (dict)
        The ``kinematic_encoding.transfer`` settings block.

    Returns
    -------
    frames (np.ndarray)
        Sorted frame indices to score.
    """

    universe = transfer_settings["universe"]
    if universe not in TRANSFER_UNIVERSES:
        msg = f"transfer.universe must be one of {TRANSFER_UNIVERSES}; got {universe!r}."
        raise ValueError(msg)
    if universe == "usv_span" or span_frames.size == 0:
        return np.asarray(span_frames, dtype=np.int64)
    first = span_frames[pointer[:-1]]
    last = span_frames[pointer[1:] - 1]
    gaps = (first[1:] - last[:-1]) / fps
    starts = np.concatenate([[0], np.flatnonzero(gaps >= float(transfer_settings["inter_bout_threshold_seconds"])) + 1])
    ends = np.concatenate([starts[1:] - 1, [first.size - 1]])
    pre = int(np.round(float(transfer_settings["pre_bout_seconds"]) * fps))
    mask = np.zeros(n_frames, dtype=bool)
    for bout_start, bout_end in zip(starts, ends, strict=True):
        mask[max(int(first[bout_start]) - pre, 0):int(last[bout_end]) + 1] = True
    mask[:max(n_lags - 1, 0)] = False
    mask[np.asarray(quiet_frames, dtype=np.int64)] = False
    return np.flatnonzero(mask)


def bin_frame_matrix(frames: np.ndarray, width: int) -> np.ndarray:
    """
    Description
    -----------
    Cut a sorted frame set into full bins of ``width`` consecutive frames, run by run.

    Contiguous runs of the frame set are binned independently so no bin straddles a gap; a run's trailing
    partial bin is dropped rather than padded.

    Parameters
    ----------
    frames (np.ndarray)
        Sorted frame indices.
    width (int)
        Frames per bin.

    Returns
    -------
    bins (np.ndarray)
        ``(n_bins, width)`` frame indices.
    """

    frames = np.asarray(frames, dtype=np.int64)
    if frames.size == 0:
        return np.zeros((0, width), dtype=np.int64)
    breaks = np.flatnonzero(np.diff(frames) != 1) + 1
    pieces = []
    for run in np.split(frames, breaks):
        n_full = run.size // width
        if n_full:
            pieces.append(run[: n_full * width].reshape(n_full, width))
    return np.vstack(pieces) if pieces else np.zeros((0, width), dtype=np.int64)


def counts_per_frame(spike_frames: np.ndarray, n_frames: int) -> np.ndarray:
    """
    Description
    -----------
    Spikes in every frame of a session; a frame can hold more than one.

    Parameters
    ----------
    spike_frames (np.ndarray)
        One entry per spike, the frame it fell in.
    n_frames (int)
        Session frame count.

    Returns
    -------
    counts (np.ndarray)
        ``(n_frames,)`` integer counts.
    """

    valid = np.asarray(spike_frames, dtype=np.int64)
    valid = valid[(valid >= 0) & (valid < n_frames)]
    return np.bincount(valid, minlength=n_frames)


def bin_width_frames(transfer_settings: dict, fps: float) -> int:
    """
    Description
    -----------
    The count grain's bin width in frames.

    Parameters
    ----------
    transfer_settings (dict)
        The ``kinematic_encoding.transfer`` settings block.
    fps (float)
        Camera frame rate.

    Returns
    -------
    width (int)
        ``round(bin_seconds * fps)``, at least 1.
    """

    grain = transfer_settings["grain"]
    if grain not in TRANSFER_GRAINS:
        msg = f"transfer.grain must be one of {TRANSFER_GRAINS}; got {grain!r}."
        raise ValueError(msg)
    return max(int(np.round(float(transfer_settings["bin_seconds"]) * fps)), 1)


def score_fold(estimator, session: dict, feature_indices: list, vocal_frames: np.ndarray, n_lags: int,
               base_rate: float, encoding_settings: dict, linear_predictor_fn, transfer_settings: dict) -> dict:
    """
    Description
    -----------
    Score one fold's frozen model on its held-out session's vocal frames, returning the raw held-out
    predictions so they can be pooled later rather than collapsed now.

    The per-fold score is computed too, but only as a diagnostic. It is the quantity that misbehaves on
    small folds, and keeping it visible next to the pooled result is what makes that visible rather than
    mysterious.

    Parameters
    ----------
    estimator (object)
        The frozen model fitted on this fold's pool.
    session (dict)
        The held-out session's assembled data.
    feature_indices (list)
        Columns the model was fitted on.
    vocal_frames (np.ndarray)
        Frames of the focal animal's calls in this session.
    n_lags (int)
        History length in frames.
    base_rate (float)
        The fold's training (quiet) spike rate, for the level descriptor.
    encoding_settings (dict)
        The ``kinematic_encoding`` settings block.
    linear_predictor_fn (Callable)
        Function evaluating the model at frames; injected to keep this module independent of the fitting
        one.
    transfer_settings (dict)
        The ``kinematic_encoding.transfer`` settings block; its ``grain`` decides which score is the
        fold's own.

    Returns
    -------
    result (dict)
        ``eta``, ``labels``, ``fold_score``, ``fold_slope``, ``auroc``, ``spike_rate``, ``n_frames``, the
        level descriptors under ``level``, and -- under the count grain -- ``bin_eta`` (the filter's
        log-odds averaged per bin), ``bin_counts``, ``n_bins`` and ``bin_width_frames``, with
        ``fold_score`` / ``fold_slope`` then the Poisson pooled score on this session's bins. The
        per-frame descriptors are kept under either grain.
    """

    eta = linear_predictor_fn(estimator, session["feature_time_series"], feature_indices, vocal_frames,
                              n_lags, encoding_settings["chunk_rows"])
    labels = spike_labels_at_frames(session["spike_frames"], vocal_frames, session["n_frames"])
    calibration_steps = encoding_settings["solver"]["calibration_steps"]
    calibration = encoding_settings["calibration"]
    score, slope = frozen_model_score(eta, labels, np.zeros(labels.size, dtype=np.int64), calibration_steps,
                                      "bernoulli", 1.0, calibration)
    result = {"eta": eta, "labels": labels, "fold_score": score, "fold_slope": slope,
              "auroc": area_under_roc(eta, labels), "spike_rate": float(labels.mean()),
              "n_frames": int(labels.size),
              "metrics": encoding_metrics(eta, labels, calibration_steps),
              "level": explained_deviance_vs_reference_rate(eta, labels, base_rate)}
    if transfer_settings["grain"] == "count":
        width = bin_width_frames(transfer_settings, float(session["fps"]))
        bins = bin_frame_matrix(vocal_frames, width)
        # `vocal_frames` is sorted and unique, so each bin's frames locate their eta by search
        bin_eta = eta[np.searchsorted(vocal_frames, bins)].mean(axis=1) if bins.size else np.zeros(0)
        per_frame = counts_per_frame(session["spike_frames"], session["n_frames"])
        bin_counts = per_frame[bins].sum(axis=1) if bins.size else np.zeros(0, dtype=np.int64)
        count_score, count_slope = frozen_model_score(
            bin_eta, bin_counts, np.zeros(bin_counts.size, dtype=np.int64), calibration_steps, "poisson",
            float(width), calibration)
        result.update({"fold_score": count_score, "fold_slope": count_slope, "bin_eta": bin_eta,
                       "bin_counts": bin_counts, "n_bins": int(bin_counts.size), "bin_width_frames": width})
    return result


def pooled_transfer_null(fold_results: list, per_session: dict, session_ids: list,
                         vocal_frames_by_session: dict, n_shuffles: int, seed: int,
                         guard_seconds: float, calibration_steps: int,
                         transfer_settings: dict, calibration: dict) -> np.ndarray:
    """
    Description
    -----------
    Null distribution of the pooled transfer score.

    Each draw shifts every session's spike train independently, relabels that session's vocal frames, and
    re-scores the pooled, still-frozen predictions. Shifting the sessions independently is what gives the
    joint draw space its size; shifting them together would collapse it back to one session's worth of
    resolution.

    Parameters
    ----------
    fold_results (list)
        Per-fold outputs of :func:`score_fold`, in ``session_ids`` order.
    per_session (dict)
        Assembled session data.
    session_ids (list)
        Sessions in the order the folds were scored.
    vocal_frames_by_session (dict)
        ``{session_id: vocal frame indices}``.
    n_shuffles (int)
        Number of draws.
    seed (int)
        Base seed; each session's shifts are drawn from its own stream.
    guard_seconds (float)
        Excluded band at both ends of the circular wrap.
    calibration_steps (int)
        Newton iterations per calibration refit.
    transfer_settings (dict)
        The ``kinematic_encoding.transfer`` settings block; under the count grain each draw recounts
        the spikes per bin of the shifted train and re-scores the Poisson pooled calibration.

    Returns
    -------
    null (np.ndarray)
        ``n_shuffles`` pooled null scores.
    """

    if transfer_settings["grain"] == "count":
        width = int(fold_results[0]["bin_width_frames"])
        bin_eta = np.concatenate([result["bin_eta"] for result in fold_results])
        bin_index = np.concatenate([np.full(result["n_bins"], index) for index, result in enumerate(fold_results)])
        bins_by_session = {session_id: bin_frame_matrix(vocal_frames_by_session[session_id], width)
                           for session_id in session_ids}
        generators = [np.random.default_rng(seed + index) for index in range(len(session_ids))]
        null = np.empty(n_shuffles, dtype=np.float64)
        for draw in range(n_shuffles):
            drawn = []
            for index, session_id in enumerate(session_ids):
                session = per_session[session_id]
                shifted = shifted_spike_frames(
                    session["spike_frames"],
                    sample_circular_shift(generators[index], session["n_frames"], session["fps"], guard_seconds),
                    session["fps"], session["n_frames"])
                bins = bins_by_session[session_id]
                drawn.append(counts_per_frame(shifted, session["n_frames"])[bins].sum(axis=1) if bins.size
                             else np.zeros(0, dtype=np.int64))
            null[draw] = frozen_model_score(bin_eta, np.concatenate(drawn), bin_index, calibration_steps,
                                            "poisson", float(width), calibration)[0]
        return null

    eta = np.concatenate([result["eta"] for result in fold_results])
    session_index = np.concatenate([np.full(result["labels"].size, index)
                                    for index, result in enumerate(fold_results)])

    # Draws are processed in CHUNKS. The previous version built an (n_shuffles, n_frames) label matrix
    # per session and concatenated them, which is fine at the 1,000 draws it was written for and fatal
    # once the escalation ladder asks for more: at 69,174 pooled vocal frames that is 55 GB at 100,000
    # draws and 498 GB at 1,000,000. It exhausted host RAM and took the desktop down with it. The chunk
    # is sized by a memory budget, and each session keeps ONE generator across chunks so the sequence of
    # shifts is exactly what an unchunked run would have drawn.
    total_frames = max(int(eta.size), 1)
    chunk = int(max(1, min(n_shuffles, _POOLED_BLOCK_BYTES // (total_frames * 8))))
    generators = [np.random.default_rng(seed + index) for index in range(len(session_ids))]

    null = np.empty(n_shuffles, dtype=np.float64)
    for start in range(0, n_shuffles, chunk):
        stop = min(start + chunk, n_shuffles)
        pieces = []
        for index, session_id in enumerate(session_ids):
            session = per_session[session_id]
            frames = vocal_frames_by_session[session_id]
            generator = generators[index]
            block = np.empty((stop - start, frames.size), dtype=np.float64)
            for draw in range(stop - start):
                shifted = shifted_spike_frames(
                    session["spike_frames"],
                    sample_circular_shift(generator, session["n_frames"], session["fps"],
                                          guard_seconds),
                    session["fps"], session["n_frames"])
                block[draw] = spike_labels_at_frames(shifted, frames, session["n_frames"])
            pieces.append(block)
        stacked = np.concatenate(pieces, axis=1)
        for offset in range(stop - start):
            null[start + offset] = frozen_model_score(eta, stacked[offset], session_index, calibration_steps,
                                                      "bernoulli", 1.0, calibration)[0]
    return null


def leave_one_session_out_scores(eta: np.ndarray, labels: np.ndarray, session_index: np.ndarray,
                                 session_ids: list, calibration_steps: int,
                                 scorer=pooled_calibrated_explained_deviance) -> dict:
    """
    Description
    -----------
    Re-pool the transfer with each session dropped in turn, so it is visible whether any one of them
    carries the result.

    This RE-POOLS rather than reading the per-session scores reported beside it, and the distinction is
    the whole point. A per-session score fits its own slope and intercept on one session's vocal frames
    alone -- the fragile construction pooling exists to avoid, and the one that on a 372-call session
    produced a null whose mean (+0.0170) exceeded its own observed value (+0.0040). Dropping a session
    and recalibrating on the rest keeps the shared slope estimated on everything that remains, so a thin
    session's instability never becomes its own estimate.

    The slope travels with each drop because the transfer's sign condition is read off it: a drop that flips
    the sign says something a score alone cannot.

    Nothing is refitted -- the held-out predictions are already in hand, so each drop costs one
    two-parameter calibration, measured at about 10 ms on a six-session unit's 34,000 frames.

    This is the kinematic encoding's analogue of ``min_leave_one_fold_out`` on the decoding analyses, where the
    partition is folds rather than sessions. Like those, it is a DESCRIPTOR and gates nothing:
    consistency is deliberately not a criterion any test in this analysis applies.

    Parameters
    ----------
    eta (np.ndarray)
        Held-out linear predictor, concatenated across sessions.
    labels (np.ndarray)
        0/1 spike labels at the same frames.
    session_index (np.ndarray)
        Integer session index per frame, indexing into ``session_ids``.
    session_ids (list)
        The scored sessions, in the order their indices refer to.
    calibration_steps (int)
        Maximum Newton iterations for each recalibration.
    scorer (Callable)
        ``(x, y, session_index, calibration_steps) -> (score, slope)``: the per-frame pooled calibration
        by default, or the Poisson one (with its exposure bound) under the count grain.

    Returns
    -------
    scores (dict)
        ``{session_id: {'score', 'slope'}}`` for the pooled transfer WITHOUT that session. Empty when
        there is only one session, since dropping it leaves nothing to pool.
    """

    if len(session_ids) < 2:
        return {}
    scores = {}
    for index, session_id in enumerate(session_ids):
        keep = session_index != index
        score, slope = scorer(eta[keep], labels[keep], session_index[keep], calibration_steps)
        scores[session_id] = {"score": score, "slope": slope}
    return scores


def combine_folds(fold_results: list, per_session: dict, session_ids: list,
                  vocal_frames_by_session: dict, settings: dict, message_output=print) -> dict:
    """
    Description
    -----------
    Pool every fold's held-out predictions into one transfer result, with its null and p-value.

    One shared calibration slope is fitted across all folds, with a separate intercept per session, because
    the sessions are not on a common scale -- per-fold transfer slopes of +0.36, +0.59 and +0.16 on one unit
    -- while the slope, the quantity the test is about, is estimated on everything at once.

    When the p-value lands on the empirical floor the caller is told so, since resolving it further is a
    matter of more draws rather than a different statistic.

    Parameters
    ----------
    fold_results (list)
        Per-fold outputs of :func:`score_fold`, in ``session_ids`` order.
    per_session (dict)
        Assembled session data.
    session_ids (list)
        Sessions in the order the folds were scored.
    vocal_frames_by_session (dict)
        ``{session_id: vocal frame indices}``.
    settings (dict)
        Full neural-modeling settings.
    message_output (Callable)
        Where the result is reported.

    Returns
    -------
    result (dict)
        ``score``, ``slope``, ``p``, ``at_floor``, ``n_frames``, ``null``, the leave-one-session-out
        descriptors ``leave_one_session_out`` and ``min_leave_one_session_out``, and the per-fold
        diagnostics under ``folds``.
    """

    # Both of these name what this function DOES, and neither was read. A run could have declared
    # fold-averaged scoring in its settings, been reported as such, and pooled all along.
    transfer_settings = settings["kinematic_encoding"]["transfer"]
    if transfer_settings["calibration_grain"] != "pooled_across_folds":
        msg = (f"the transfer fits ONE calibration -- a shared slope with per-session intercepts -- "
               f"on every fold's held-out predictions at once, rather than one per fold; "
               f"calibration_grain is {transfer_settings['calibration_grain']!r}.")
        raise ValueError(msg)
    if transfer_settings["d2_reference_rate"] != "scored_frames":
        msg = (f"the transfer D2 is measured against an intercept-only model at the SCORED frames' "
               f"own spike rate, not the training rate; "
               f"d2_reference_rate is {transfer_settings['d2_reference_rate']!r}.")
        raise ValueError(msg)

    calibration_steps = settings["kinematic_encoding"]["solver"]["calibration_steps"]
    grain = transfer_settings["grain"]
    if grain not in TRANSFER_GRAINS:
        msg = f"transfer.grain must be one of {TRANSFER_GRAINS}; got {grain!r}."
        raise ValueError(msg)
    if grain == "count":
        # The tested quantity: spikes per bin against the frozen filter's log-odds averaged per bin,
        # Poisson, one shared gain, per-session intercepts, the bin width as exposure. The per-frame
        # arrays stay in the fold results as descriptors.
        width = int(fold_results[0]["bin_width_frames"])
        eta = np.concatenate([result["bin_eta"] for result in fold_results])
        labels = np.concatenate([result["bin_counts"] for result in fold_results])
        session_index = np.concatenate([np.full(result["n_bins"], index)
                                        for index, result in enumerate(fold_results)])
        scorer = partial(frozen_model_score, family="poisson", exposure=float(width),
                         calibration=settings["kinematic_encoding"]["calibration"])
    else:
        width = 1
        eta = np.concatenate([result["eta"] for result in fold_results])
        labels = np.concatenate([result["labels"] for result in fold_results])
        session_index = np.concatenate([np.full(result["labels"].size, index)
                                        for index, result in enumerate(fold_results)])
        scorer = partial(frozen_model_score, family="bernoulli", exposure=1.0,
                         calibration=settings["kinematic_encoding"]["calibration"])
    score, slope = scorer(eta, labels, session_index, calibration_steps)
    leave_one_session_out = leave_one_session_out_scores(eta, labels, session_index, session_ids,
                                                         calibration_steps, scorer)
    min_leave_one_session_out = (min(entry["score"] for entry in leave_one_session_out.values())
                                 if leave_one_session_out else float("nan"))
    escalation_round = 0

    def draw_transfer_null(count):
        """Fresh pooled-null draws; the seed advances so escalation never repeats a shift sequence."""
        nonlocal escalation_round
        seed = int(settings["null"]["shuffle_seed"]) + 1_000_003 * escalation_round
        escalation_round += 1
        return pooled_transfer_null(fold_results, per_session, session_ids, vocal_frames_by_session,
                                    count, seed, settings["null"]["shuffle_guard_seconds"],
                                    calibration_steps, transfer_settings,
                                    settings["kinematic_encoding"]["calibration"])

    null = draw_transfer_null(settings["null"]["n_shuffles"])
    p_value, at_floor, null = escalated_empirical_pvalue(
        score, draw_transfer_null, settings["significance"]["escalation_ladder"], null,
        message_output)
    n_frames_total = int(sum(result["n_frames"] for result in fold_results))
    unit_label = (f"{labels.size} bins of {width} frames over {n_frames_total} frames" if grain == "count"
                  else f"{labels.size} vocal frames")
    message_output(f"  TESTED TRANSFER (representative model, pooled, {transfer_settings['universe']}, "
                   f"{grain} grain)  score {score:+.5f} | slope {slope:+.3f} | p {p_value:.4e}"
                   f"{' (at floor, escalation would resolve further)' if at_floor else ''} "
                   f"| {unit_label}")
    if leave_one_session_out:
        message_output(f"    leave-one-session-out minimum {min_leave_one_session_out:+.5f} "
                       f"(descriptor, gates nothing)")
    for session_id, result in zip(session_ids, fold_results, strict=True):
        dropped = leave_one_session_out[session_id] if leave_one_session_out else None
        without = (f" | without it {dropped['score']:+.5f} slope {dropped['slope']:+.3f}"
                   if dropped is not None else "")
        message_output(f"    per-session {session_id}: score {result['fold_score']:+.5f} "
                       f"| slope {result['fold_slope']:+.3f} | AUROC {result['auroc']:.3f} "
                       f"| {result['n_frames']} frames at rate {result['spike_rate']:.4f}{without}")
    return {"score": score, "slope": slope, "p": p_value, "at_floor": at_floor,
            "n_frames": n_frames_total, "n_bins": int(labels.size) if grain == "count" else 0,
            "bin_width_frames": width, "grain": grain, "universe": transfer_settings["universe"],
            "null": null, "leave_one_session_out": leave_one_session_out,
            "min_leave_one_session_out": min_leave_one_session_out,
            "folds": [{key: value for key, value in result.items()
                       if key not in ("eta", "labels", "bin_eta", "bin_counts")}
                      for result in fold_results]}
