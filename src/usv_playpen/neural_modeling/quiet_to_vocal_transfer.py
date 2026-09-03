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

The gate is discrimination, not level: whether the model tells a frame with a spike from one without,
scored against the vocal frames' own rate. Whether the unit simply fires faster during vocalization is
recorded as a descriptor and gates nothing, because a model can predict the rate change perfectly while
ordering frames at chance.
"""

from __future__ import annotations

import numpy as np

from .deviance_metrics import (area_under_roc, calibrated_explained_deviance,
                               explained_deviance_vs_reference_rate,
                               pooled_calibrated_explained_deviance)
from .neural_design_assembly import spike_labels_at_frames
from .shift_null_inference import empirical_pvalue, sample_circular_shift, shifted_spike_frames


def score_fold(estimator, session: dict, feature_indices: list, vocal_frames: np.ndarray, n_lags: int,
               base_rate: float, encoding_settings: dict, linear_predictor_fn) -> dict:
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

    Returns
    -------
    result (dict)
        ``eta``, ``labels``, ``fold_score``, ``fold_slope``, ``auroc``, ``spike_rate``, ``n_frames``, and
        the level descriptors under ``level``.
    """

    eta = linear_predictor_fn(estimator, session["feature_time_series"], feature_indices, vocal_frames,
                              n_lags, encoding_settings["chunk_rows"])
    labels = spike_labels_at_frames(session["spike_frames"], vocal_frames, session["n_frames"])
    score, slope = calibrated_explained_deviance(eta, labels,
                                                 encoding_settings["solver"]["calibration_steps"])
    return {"eta": eta, "labels": labels, "fold_score": score, "fold_slope": slope,
            "auroc": area_under_roc(eta, labels), "spike_rate": float(labels.mean()),
            "n_frames": int(labels.size),
            "level": explained_deviance_vs_reference_rate(eta, labels, base_rate)}


def pooled_transfer_null(fold_results: list, per_session: dict, session_ids: list,
                         vocal_frames_by_session: dict, n_shuffles: int, seed: int,
                         guard_seconds: float, calibration_steps: int) -> np.ndarray:
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

    Returns
    -------
    null (np.ndarray)
        ``n_shuffles`` pooled null scores.
    """

    eta = np.concatenate([result["eta"] for result in fold_results])
    session_index = np.concatenate([np.full(result["labels"].size, index)
                                    for index, result in enumerate(fold_results)])
    shuffled_labels = []
    for index, session_id in enumerate(session_ids):
        session = per_session[session_id]
        frames = vocal_frames_by_session[session_id]
        rng = np.random.default_rng(seed + index)
        block = np.empty((n_shuffles, frames.size), dtype=np.float64)
        for draw in range(n_shuffles):
            shifted = shifted_spike_frames(
                session["spike_frames"],
                sample_circular_shift(rng, session["n_frames"], session["fps"], guard_seconds),
                session["fps"], session["n_frames"])
            block[draw] = spike_labels_at_frames(shifted, frames, session["n_frames"])
        shuffled_labels.append(block)

    stacked = np.concatenate(shuffled_labels, axis=1)
    return np.array([pooled_calibrated_explained_deviance(eta, stacked[draw], session_index,
                                                          calibration_steps)[0]
                     for draw in range(n_shuffles)], dtype=np.float64)


def combine_folds(fold_results: list, per_session: dict, session_ids: list,
                  vocal_frames_by_session: dict, settings: dict, message_output=print) -> dict:
    """
    Description
    -----------
    Pool every fold's held-out predictions into one transfer result, with its null and p-value.

    One shared calibration slope is fitted across all folds, with a separate intercept per session, because
    the sessions are not on a common scale -- per-fold transfer slopes of +0.36, +0.59 and +0.16 on one unit
    -- while the slope, the quantity the claim is about, is estimated on everything at once.

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
        ``score``, ``slope``, ``p``, ``at_floor``, ``n_frames``, ``null``, and the per-fold diagnostics
        under ``folds``.
    """

    eta = np.concatenate([result["eta"] for result in fold_results])
    labels = np.concatenate([result["labels"] for result in fold_results])
    session_index = np.concatenate([np.full(result["labels"].size, index)
                                    for index, result in enumerate(fold_results)])
    calibration_steps = settings["kinematic_encoding"]["solver"]["calibration_steps"]
    score, slope = pooled_calibrated_explained_deviance(eta, labels, session_index, calibration_steps)
    null = pooled_transfer_null(fold_results, per_session, session_ids, vocal_frames_by_session,
                                settings["null"]["n_shuffles"], settings["null"]["shuffle_seed"],
                                settings["null"]["shuffle_guard_seconds"], calibration_steps)
    p_value, at_floor = empirical_pvalue(null, score)
    message_output(f"  POOLED transfer  score {score:+.5f} | slope {slope:+.3f} | p {p_value:.4e}"
                   f"{' (at floor, escalation would resolve further)' if at_floor else ''} "
                   f"| {labels.size} vocal frames")
    for session_id, result in zip(session_ids, fold_results):
        message_output(f"    fold {session_id}: score {result['fold_score']:+.5f} "
                       f"| slope {result['fold_slope']:+.3f} | AUROC {result['auroc']:.3f} "
                       f"| {result['n_frames']} frames at rate {result['spike_rate']:.4f}")
    return {"score": score, "slope": slope, "p": p_value, "at_floor": at_floor,
            "n_frames": int(labels.size), "null": null,
            "folds": [{key: value for key, value in result.items() if key not in ("eta", "labels")}
                      for result in fold_results]}
