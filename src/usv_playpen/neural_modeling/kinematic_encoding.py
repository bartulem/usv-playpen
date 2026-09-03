"""
@author: bartulem
Fitting kinematic encoding models on quiet frames, and choosing which features enter them.

The procedure is nested. A day's sessions split into a POOL and a held-out TEST session; screening and
forward selection see only the pool, using an inner leave-one-pool-session-out split, and the test session
is untouched until the fitted model is scored. Selecting features by a score measured on the same session
that later reports the result would be circular, and nesting is what prevents it -- at the cost of one
session, which on a three-session day leaves only two inner folds.

Two things about the selection are easy to get wrong and were got wrong here first.

The screen judges each feature against ITS OWN null. Features differ in how easily they overfit: a slow,
smooth, heavily autocorrelated variable has a much wider null under a circular shift than a fast one, so a
pooled null would penalise the tight-null features for the loose ones.

Acceptance compares a PAIRED improvement against a PAIRED standard error. The obvious form -- accept when
the candidate's mean minus the candidate's own standard error beats the incumbent -- charges the
improvement for variation that both models share. Measured here: per-fold scores of [0.1941, 0.1418] for
one model and [0.2060, 0.1534] for the same model plus a feature give an improvement of [0.0118, 0.0116],
about as consistent as anything in the analysis, yet the unpaired standard error is 0.02629 against the
paired 0.00012 -- a factor of 219. A candidate improving by 97 of its own standard errors was rejected
three folds running. Session-to-session variation moves both models together and cancels in the
difference; only the paired form sees that.
"""

from __future__ import annotations

import numpy as np

from ..modeling.jax_group_elastic_net import GroupElasticNetGLM
from .deviance_metrics import (
    batched_calibrated_explained_deviance,
    binned_calibrated_explained_deviance,
    calibrated_explained_deviance,
)
from .neural_design_assembly import lagged_design, spike_labels_at_frames, subsample_quiet_anchors
from .shift_null_inference import empirical_pvalue, sample_circular_shift, shifted_spike_frames


# Memory budget for one batched null block. The Newton iteration holds several (n_frames, block)
# float64 arrays at once, so this caps the working set at a few hundred MB per block.
_NULL_BLOCK_BYTES = 32 * 1024 * 1024

def inner_folds(pool_session_ids: list) -> list:
    """
    Description
    -----------
    Leave-one-pool-session-out splits, as ``[(train_ids, validation_id), ...]``.

    Parameters
    ----------
    pool_session_ids (list)
        Sessions available for selection; never includes the test session.

    Returns
    -------
    folds (list)
        One (train, validation) pair per pool session.
    """

    return [([other for other in pool_session_ids if other != held_out], held_out)
            for held_out in pool_session_ids]


def fit_quiet_model(per_session: dict, train_session_ids: list, feature_indices: list, n_lags: int,
                    rng, encoding_settings: dict, message_output=print) -> tuple:
    """
    Description
    -----------
    Fit the encoding model on the quiet anchors of the given sessions, using only the requested feature
    columns.

    Single-feature screens and multi-feature selection candidates both come through here, so the two cannot
    drift apart. Anchors are class-balanced with the log-prior offset correction and the fit runs
    unweighted, which is the efficient case-control estimator. The per-session cap divides the global cap
    across training sessions, because the design is assembled from all of them at once and it is the total
    that has to stay in memory.

    Parameters
    ----------
    per_session (dict)
        ``{session_id: {feature_time_series, fps, n_frames, spike_frames, quiet}}``.
    train_session_ids (list)
        Sessions to fit on.
    feature_indices (list)
        Column indices of the features entering the model.
    n_lags (int)
        History length in frames.
    rng (np.random.Generator)
        Seeded generator for the subsampling.
    encoding_settings (dict)
        The ``kinematic_encoding`` settings block.

    Returns
    -------
    estimator (GroupElasticNetGLM)
        The fitted model.
    base_rate (float)
        Pooled training spike rate before subsampling, on the true-population scale.
    """

    penalties = encoding_settings["significance_model"]
    columns = list(feature_indices)
    design_blocks, label_blocks, offset_blocks = [], [], []
    n_positive_total, n_negative_total = 0, 0
    per_session_cap = max(int(encoding_settings["max_train_total"] / max(len(train_session_ids), 1)), 1)
    for session_id in train_session_ids:
        session = per_session[session_id]
        anchors, labels, offsets, n_positive, n_negative = subsample_quiet_anchors(
            session["quiet"], session["spike_frames"], session["n_frames"],
            encoding_settings["negatives_per_positive"], rng, per_session_cap)
        design_blocks.append(lagged_design(session["feature_time_series"][:, columns], anchors, n_lags))
        label_blocks.append(labels)
        offset_blocks.append(offsets)
        n_positive_total += n_positive
        n_negative_total += n_negative

    estimator = GroupElasticNetGLM(
        n_features=len(columns), n_time_bins=n_lags, family="bernoulli",
        lambda_group=penalties["lambda_group"], lambda_smooth=penalties["lambda_smooth"],
        lambda_ridge=penalties["lambda_ridge"], smoothness_order=encoding_settings["smoothness_order"],
        debias_refit=False, max_iter=encoding_settings["solver"]["max_iter"])
    estimator.fit(np.vstack(design_blocks), np.concatenate(label_blocks),
                  offset=np.concatenate(offset_blocks))

    # A FISTA run that exhausts max_iter returns whatever iterate it reached, and it is otherwise
    # indistinguishable from a converged fit -- same type, same attributes, a filter that looks
    # plausible. Real single-feature fits land at 2,300-5,000 iterations against a 5,000 cap, so this
    # is not a remote failure mode: it happens, and it decides whether a reported filter means
    # anything. Say so where it happens rather than leaving it in an attribute nobody reads.
    if not estimator.converged_:
        message_output(f"    [!] FIT DID NOT CONVERGE: {len(columns)} feature(s), "
                       f"{estimator.n_iter_} iterations at the max_iter cap of "
                       f"{encoding_settings['solver']['max_iter']} -- the filter is an unconverged "
                       f"iterate and every score derived from it is provisional")

    denominator = max(n_positive_total + n_negative_total, 1)
    return estimator, float(min(max(n_positive_total / denominator, 1e-6), 1.0 - 1e-6))


def linear_predictor_at_frames(estimator, feature_time_series: np.ndarray, feature_indices: list,
                               frames: np.ndarray, n_lags: int, chunk_rows: int) -> np.ndarray:
    """
    Description
    -----------
    The fitted model's linear predictor at the given frames, evaluated in chunks to bound peak memory.

    Computing this once per (model, frame set) is what makes the frozen nulls cheap: a shuffle changes only
    the LABELS, never the predictions, so a thousand draws cost a thousand relabellings rather than a
    thousand fits.

    Parameters
    ----------
    estimator (object)
        A fitted model exposing ``predict_eta``.
    feature_time_series (np.ndarray)
        ``(n_frames, n_features)`` predictor time series.
    feature_indices (list)
        Column indices the model was fitted on, in the same order.
    frames (np.ndarray)
        Frames to evaluate.
    n_lags (int)
        History length in frames.
    chunk_rows (int)
        Frames per chunk.

    Returns
    -------
    eta (np.ndarray)
        Linear predictor aligned with ``frames``.
    """

    columns = list(feature_indices)
    values = np.empty(frames.size, dtype=np.float64)
    for lo in range(0, frames.size, chunk_rows):
        block = frames[lo:lo + chunk_rows]
        values[lo:lo + block.size] = np.asarray(
            estimator.predict_eta(lagged_design(feature_time_series[:, columns], block, n_lags)))
    return values


def frozen_null_scores(eta: np.ndarray, frames: np.ndarray, spike_frames: np.ndarray, n_frames: int,
                       fps: float, n_shuffles: int, rng, guard_seconds: float,
                       calibration_steps: int, calibration_bins: int) -> np.ndarray:
    """
    Description
    -----------
    Null distribution of the calibrated score: circularly shift the scored session's spike train, relabel
    the same frames, and re-score the same frozen ``eta``. Nothing is refitted.

    Parameters
    ----------
    eta (np.ndarray)
        Frozen linear predictor at ``frames``.
    frames (np.ndarray)
        Frames being scored.
    spike_frames (np.ndarray)
        The scored session's spike-frame train.
    n_frames (int)
        Session frame count.
    fps (float)
        Camera frame rate.
    n_shuffles (int)
        Number of draws.
    rng (np.random.Generator)
        Seeded generator.
    guard_seconds (float)
        Excluded band at both ends of the circular wrap.
    calibration_steps (int)
        Newton iterations for each draw's calibration refit.
    calibration_bins (int)
        Equal-count predictor bins for the binned calibration. ``0`` runs the exact path instead. The
        responses are exact either way; see ``binned_calibrated_explained_deviance`` for the measured
        cost of the approximation.

    Returns
    -------
    null (np.ndarray)
        ``n_shuffles`` null scores.
    """

    # The shifts are drawn in one pass so the generator is consumed in exactly the order the
    # one-draw-at-a-time version consumed it, and the null stays reproducible against older runs.
    shifts = [sample_circular_shift(rng, n_frames, fps, guard_seconds) for _ in range(n_shuffles)]

    # eta is frozen across draws, so the calibration design is shared and the draws batch. The block is
    # sized by memory rather than by draw count: the Newton iteration holds several (n_frames, block)
    # float64 arrays at once, and a whole 10,000-draw block over ~120,000 frames would be ~10 GB.
    block = int(max(1, min(n_shuffles, _NULL_BLOCK_BYTES // max(1, frames.size * 8))))

    null = np.empty(n_shuffles, dtype=np.float64)
    for start in range(0, n_shuffles, block):
        stop = min(start + block, n_shuffles)
        labels = np.empty((frames.size, stop - start), dtype=np.float64)
        for column, draw in enumerate(range(start, stop)):
            shifted = shifted_spike_frames(spike_frames, shifts[draw], fps, n_frames)
            labels[:, column] = spike_labels_at_frames(shifted, frames, n_frames)
        if calibration_bins > 0:
            null[start:stop] = binned_calibrated_explained_deviance(
                eta, labels, calibration_steps, calibration_bins)
        else:
            null[start:stop] = batched_calibrated_explained_deviance(eta, labels, calibration_steps)
    return null


def screen_features(per_session: dict, pool_session_ids: list, n_lags: int, rng, settings: dict,
                    feature_names: list, message_output=print) -> list:
    """
    Description
    -----------
    Score every feature on its own and decide which may enter selection, entirely inside the pool.

    Each feature is fitted alone under the inner split and scored on the held-out POOL session. It survives
    when its fold-averaged score is positive, its calibration slope is positive, and its own frozen-filter
    null puts it past the Bonferroni-corrected threshold.

    The slope condition matters because the calibrated score grows with the square of the slope and is
    therefore blind to sign; a filter that is reliably BACKWARDS explains deviance as well as a right one.
    The slope relates the MODEL'S OUTPUT to the labels rather than the feature to the labels, so a
    negatively tuned unit still gives a positive slope -- the fit has already absorbed the sign into
    negative weights. A negative slope means the learned relationship flipped between the fitting sessions
    and the validation session, which is a failure to replicate, not negative tuning.

    The screen draws its own shuffle count, separate from the one behind the final p-values, because the two
    are set by different bars. A feature must clear a Bonferroni threshold of ``screen_alpha / n_features``,
    which for 19 features at 0.01 is 5.26e-4; an empirical p floors at ``1 / (n + 1)``, so a thousand draws
    floor at 9.99e-4 and NOTHING could ever pass. Ten thousand floors at 1.0e-4 and clears the bar with room
    to spare. This was masked for a long time by a parametric tail fit, which was quietly carrying the
    screen rather than only the final p-values.

    The test session is never touched: this is part of selection.

    Parameters
    ----------
    per_session (dict)
        Assembled session data.
    pool_session_ids (list)
        The pool sessions.
    n_lags (int)
        History length in frames.
    rng (np.random.Generator)
        Seeded generator.
    settings (dict)
        Full neural-modeling settings.
    feature_names (list)
        Feature names, indexed as the columns are.
    message_output (Callable)
        Where progress is reported.

    Returns
    -------
    rows (list)
        One dict per feature: ``feature``, ``name``, ``score``, ``slope``, ``p``, ``at_floor``,
        ``survived``.
    """

    encoding = settings["kinematic_encoding"]
    folds = inner_folds(pool_session_ids)
    n_features = per_session[pool_session_ids[0]]["feature_time_series"].shape[1]
    threshold = encoding["feature_selection"]["screen_alpha"] / n_features
    require_positive_slope = encoding["transfer"]["require_positive_slope"]
    rows = []
    for feature in range(n_features):
        fold_scores, fold_slopes, fold_nulls, fold_converged = [], [], [], []
        for train_ids, validation_id in folds:
            estimator, _base_rate = fit_quiet_model(per_session, train_ids, [feature], n_lags, rng,
                                                    encoding, message_output)
            fold_converged.append(bool(estimator.converged_))
            session = per_session[validation_id]
            eta = linear_predictor_at_frames(estimator, session["feature_time_series"], [feature],
                                             session["quiet"], n_lags, encoding["chunk_rows"])
            labels = spike_labels_at_frames(session["spike_frames"], session["quiet"],
                                            session["n_frames"])
            score, slope = calibrated_explained_deviance(eta, labels, encoding["solver"]["calibration_steps"])
            fold_scores.append(score)
            fold_slopes.append(slope)
            fold_nulls.append(frozen_null_scores(
                eta, session["quiet"], session["spike_frames"], session["n_frames"], session["fps"],
                settings["null"]["screen_n_shuffles"], rng, settings["null"]["shuffle_guard_seconds"],
                encoding["solver"]["calibration_steps"],
                encoding["solver"]["null_calibration_bins"]))
        score = float(np.nanmean(fold_scores))
        slope = float(np.nanmean(fold_slopes))
        p_value, at_floor = empirical_pvalue(np.nanmean(np.vstack(fold_nulls), axis=0), score)
        slope_ok = (not require_positive_slope) or (np.isfinite(slope) and slope > 0)
        survived = bool(score > 0 and slope_ok and np.isfinite(p_value) and p_value < threshold)
        rows.append({"feature": feature, "name": feature_names[feature], "score": score, "slope": slope,
                     "p": p_value, "at_floor": at_floor, "survived": survived,
                     "converged": bool(np.all(fold_converged))})
        message_output(f"    screen {feature_names[feature]:<32} score {score:+.5f} | slope {slope:+.3f} "
                       f"| p {p_value:.2e} | {'PASS' if survived else 'fail'}"
                       f"{'' if np.all(fold_converged) else ' | UNCONVERGED'}")
    return rows


def forward_select(per_session: dict, pool_session_ids: list, survivors: list, n_lags: int, rng,
                   settings: dict, feature_names: list, message_output=print) -> tuple:
    """
    Description
    -----------
    Greedy forward selection over the surviving features, on pool quiet anchors, stopping at the first
    rejection.

    A candidate is accepted when its mean paired per-fold improvement exceeds the standard error of those
    paired differences. See the module docstring for why the improvement must be paired: charging it the
    variation in the models' absolute scores made the rule about 219 times too conservative and stopped
    selection at the anchor every time.

    Parameters
    ----------
    per_session (dict)
        Assembled session data.
    pool_session_ids (list)
        The pool sessions.
    survivors (list)
        Feature indices that passed the screen.
    n_lags (int)
        History length in frames.
    rng (np.random.Generator)
        Seeded generator.
    settings (dict)
        Full neural-modeling settings.
    feature_names (list)
        Feature names, indexed as the columns are.
    message_output (Callable)
        Where progress is reported.

    Returns
    -------
    selected (list)
        Chosen feature indices, in the order accepted.
    path (list)
        One dict per step: ``step``, ``candidate``, ``mean``, ``improvement``, ``standard_error``,
        ``decision``.
    """

    encoding = settings["kinematic_encoding"]
    folds = inner_folds(pool_session_ids)

    def fold_scores(columns):
        """Per-inner-fold score vector for a candidate feature set, and whether every fit converged."""
        values = []
        converged = True
        for train_ids, validation_id in folds:
            estimator, _base_rate = fit_quiet_model(per_session, train_ids, columns, n_lags, rng, encoding,
                                                    message_output)
            converged = converged and bool(estimator.converged_)
            session = per_session[validation_id]
            eta = linear_predictor_at_frames(estimator, session["feature_time_series"], columns,
                                             session["quiet"], n_lags, encoding["chunk_rows"])
            labels = spike_labels_at_frames(session["spike_frames"], session["quiet"],
                                            session["n_frames"])
            values.append(calibrated_explained_deviance(
                eta, labels, encoding["solver"]["calibration_steps"])[0])
        return np.asarray(values, dtype=np.float64), converged

    def paired_improvement(candidate_scores, incumbent_scores):
        """Mean paired per-fold improvement and the standard error of those paired differences."""
        difference = candidate_scores - incumbent_scores
        finite = difference[np.isfinite(difference)]
        if finite.size == 0:
            return np.nan, np.nan
        error = (float(np.std(finite, ddof=1) / np.sqrt(finite.size)) if finite.size > 1 else 0.0)
        return float(finite.mean()), error

    scored = [(feature, *fold_scores([feature])) for feature in survivors]
    scored = [(feature, values, converged) for feature, values, converged in scored
              if np.isfinite(np.nanmean(values)) and np.nanmean(values) > 0]
    if not scored:
        return [], []
    anchor, incumbent_scores, anchor_converged = max(scored, key=lambda item: np.nanmean(item[1]))
    incumbent_mean = float(np.nanmean(incumbent_scores))
    selected = [anchor]
    path = [{"step": 0, "candidate": feature_names[anchor], "mean": incumbent_mean,
             "improvement": np.nan, "standard_error": np.nan, "decision": "ANCHOR",
             "converged": anchor_converged}]
    message_output(f"    select step 0  ANCHOR {feature_names[anchor]:<32} score {incumbent_mean:+.5f} "
                   f"| per fold {np.array2string(incumbent_scores, precision=4)}"
                   f"{'' if anchor_converged else ' | UNCONVERGED'}")

    remaining = [feature for feature in survivors if feature != anchor]
    step = 1
    while remaining:
        best_feature, best_mean, best_scores, best_converged = None, -np.inf, None, True
        for feature in remaining:
            values, converged = fold_scores(selected + [feature])
            mean = float(np.nanmean(values))
            if np.isfinite(mean) and mean > best_mean:
                best_feature, best_mean, best_scores, best_converged = feature, mean, values, converged
        if best_feature is None:
            break
        improvement, standard_error = paired_improvement(best_scores, incumbent_scores)
        accept = np.isfinite(improvement) and improvement > standard_error
        path.append({"step": step, "candidate": feature_names[best_feature], "mean": best_mean,
                     "improvement": improvement, "standard_error": standard_error,
                     "decision": "ACCEPT" if accept else "REJECT", "converged": best_converged})
        message_output(f"    select step {step}  {'ACCEPT' if accept else 'REJECT'} "
                       f"{feature_names[best_feature]:<32} score {best_mean:+.5f} | paired improvement "
                       f"{improvement:+.5f} vs its SE {standard_error:.5f}"
                       f"{'' if best_converged else ' | UNCONVERGED'}")
        if not accept:
            break
        selected.append(best_feature)
        remaining.remove(best_feature)
        incumbent_scores, incumbent_mean = best_scores, best_mean
        step += 1
    return selected, path
