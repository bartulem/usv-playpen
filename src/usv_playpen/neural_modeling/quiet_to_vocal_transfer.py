"""
@author: bartulem
Quiet-to-vocal transfer: score a kinematic model fitted on silence on the vocal frames it never saw.

The selected model is frozen (no refitting, no re-selection) and evaluated on each held-out session's
transfer frames, either the USV spans or the bout window around them. Under the count grain the frames
are cut into fixed-width bins and scored as spike counts. Every fold's held-out predictions are pooled
into one calibration (a shared slope, per-session intercepts) and tested against a circular-shift null
in which each session is shifted independently. A leave-one-session-out minimum is reported as a
descriptor.

Exports: ``transfer_frames``, ``bin_frame_matrix``, ``counts_per_frame``, ``bin_width_frames``,
``score_fold``, ``pooled_transfer_null``, ``leave_one_session_out_scores`` and ``combine_folds``.
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
    The frames the transfer is scored on: the USVs' own spans, or the bout window around them.

    ``"usv_span"`` keeps every frame inside one of the recorded animal's USVs. ``"bout_window"`` groups
    the USVs into bouts (consecutive USVs closer than ``inter_bout_threshold_seconds``, single-USV bouts
    included) and keeps the union of ``[bout start - pre_bout_seconds, bout end]``. Frames without a full
    history and every quiet anchor are removed, so no frame is both fitted and scored.

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
    Score one fold's frozen model on its held-out session's transfer frames, returning the raw held-out
    predictions so they can be pooled later.

    The per-fold score is a diagnostic only; the tested score is the pooled one (:func:`combine_folds`).

    Parameters
    ----------
    estimator (object)
        The frozen model fitted on this fold's pool.
    session (dict)
        The held-out session's assembled data.
    feature_indices (list)
        Columns the model was fitted on.
    vocal_frames (np.ndarray)
        Transfer frames of this session, as :func:`transfer_frames` returns them.
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

    Each draw shifts every session's spike train independently, relabels that session's transfer frames
    (or recounts its bins), and re-scores the pooled, still-frozen predictions. Independent shifts give
    the joint draw space its size.

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
    calibration (dict)
        The ``kinematic_encoding.calibration`` settings block, passed to the scorer.

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

    # Draws are processed in chunks sized by a memory budget, since a full (n_shuffles, n_frames) label
    # matrix grows without bound with the draw count. Each session keeps one generator across chunks,
    # so the shifts are exactly those an unchunked run would draw.
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
    Re-pool the transfer with each session dropped in turn, to show whether any one session carries
    the result.

    Each drop recalibrates on the remaining sessions, keeping the shared slope estimated on everything
    that remains, rather than scoring sessions alone. Nothing is refitted. The slope is returned with
    each score because the transfer's sign condition is read off it. A descriptor; it gates nothing.

    Parameters
    ----------
    eta (np.ndarray)
        Held-out linear predictor, concatenated across sessions.
    labels (np.ndarray)
        0/1 spike labels at the same frames, or spike counts per bin under the count grain.
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

    One shared calibration slope is fitted across all folds, with a separate intercept per session, so
    the slope is estimated on every fold at once while sessions keep their own level.
    A p-value on the empirical floor is flagged as such.

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
        ``score``, ``slope``, ``p``, ``at_floor``, ``n_frames``, ``n_bins``, ``bin_width_frames``,
        ``grain``, ``universe``, ``null``, the leave-one-session-out descriptors
        ``leave_one_session_out`` and ``min_leave_one_session_out``, and the per-fold diagnostics under
        ``folds``.
    """

    # These settings name what this function does; reject any other value rather than ignore it.
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
