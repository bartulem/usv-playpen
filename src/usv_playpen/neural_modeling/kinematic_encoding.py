"""
@author: bartulem
Fitting kinematic encoding models on quiet frames, and choosing which features enter them.

The procedure is nested: a unit's sessions split into a pool and a held-out test session, and screening and
forward selection see only the pool, through an inner leave-one-pool-session-out split. The model is a
Bernoulli GLM on lagged features, fitted on subsampled quiet anchors with the case-control log-prior
offset and per-session baselines, with each filter parametrised on a spline lag basis.

The module provides:
- the fit (:func:`fit_quiet_model`, :func:`lag_basis`, :class:`FrameFilterModel`) and its frozen linear
  predictor (:func:`linear_predictor_at_frames`);
- the per-feature screen against each feature's own circular-shift null (:func:`screen_features`,
  :func:`frozen_null_scores`);
- greedy forward selection with the paired one-standard-error rule (:func:`forward_select`);
- the inner splits (:func:`inner_folds`, and :func:`quiet_block_sessions` for a single-session pool);
- the stored descriptors: the block-resampled filter band, filter shape summaries and the spike-triggered
  average.
"""

from __future__ import annotations

import numpy as np

from ..modeling.jax_group_elastic_net import GroupElasticNetGLM
from ..modeling.modeling_bases_functions import gam_bspline_basis
from ..modeling.modeling_utils import paired_one_se_improvement
from .deviance_metrics import (
    finite_mean,
    frozen_model_null_scores,
    frozen_model_score,
)
from .neural_design_assembly import (
    lagged_design,
    spike_labels_at_frames,
    subsample_quiet_anchors,
)
from .shift_null_inference import (
    empirical_pvalue,
    sample_circular_shift,
    shifted_spike_frames,
)

TEMPORAL_BASIS_TYPES = ("none", "bspline")
SPIKE_END_CORRECTIONS = ("none", "drop_outside_and_rescale")


class FrameFilterModel:
    """
    Description
    -----------
    A fitted encoding model expressed as one weight per frame, whatever space it was fitted in.

    The spline path fits ``n_splines`` coefficients per feature and maps them back through the basis, so
    the object every consumer sees -- the screen, forward selection, the transfer, the band, the artifact
    -- carries the same ``coef_`` (feature-major, ``n_lags`` per feature), ``intercept_``, ``converged_``
    and ``n_iter_`` the raw solver carries, and predicts from a raw lagged design exactly as it does.

    Parameters
    ----------
    coef (np.ndarray)
        Frame-space weights, ``(n_features * n_lags,)``.
    intercept (float)
        Fitted intercept on the true-population scale.
    converged (bool)
        Whether the underlying solve reached its tolerance.
    n_iter (int)
        Iterations the underlying solve took.
    """

    def __init__(self, coef: np.ndarray, intercept: float, converged: bool, n_iter: int) -> None:
        self.coef_ = np.asarray(coef, dtype=np.float64)
        self.intercept_ = float(intercept)
        self.converged_ = bool(converged)
        self.n_iter_ = int(n_iter)

    def predict_eta(self, x_design: np.ndarray, offset: np.ndarray | None = None) -> np.ndarray:
        """
        Description
        -----------
        Linear predictor at the rows of a raw lagged design.

        Parameters
        ----------
        x_design (np.ndarray)
            ``(n_rows, n_features * n_lags)`` lagged design, the same layout the raw solver takes.
        offset (np.ndarray | None)
            Optional per-row offset added to the predictor.

        Returns
        -------
        eta (np.ndarray)
            ``(n_rows,)`` linear predictor.
        """

        eta = x_design @ self.coef_ + self.intercept_
        return eta if offset is None else eta + np.asarray(offset, dtype=np.float64)


def lag_basis(n_lags: int, temporal_basis: dict) -> np.ndarray | None:
    """
    Description
    -----------
    The lag-axis basis the quiet fit is parametrised in, or None for one free weight per frame.

    ``"bspline"`` (the shipped path) is the modeling pipeline's ``gam_bspline_basis``: ``n_splines``
    B-splines on knots spaced evenly over the history. Three adjustments follow:

    1. ``spike_end_correction == "drop_outside_and_rescale"`` drops the outermost spline at the spike end,
       which lies mostly outside the window, and scales each remaining spline by
       ``sqrt(largest in-window mass / its own in-window mass)``, so a uniform ridge does not pull the
       filter toward zero just before the spike. ``"none"`` keeps the uncorrected basis.
    2. Each column is scaled by ``1 / sqrt(1 + alpha * lag_fraction**2)``, where ``lag_fraction`` is the
       spline centre's distance from the spike frame as a fraction of the window, so a uniform ridge on
       the scaled coefficients grows with lag.
    3. The whole basis is divided by its mean column sum, which turns each projected column into a
       lag-average on the feature's own scale and keeps the solver's step size workable.

    Parameters
    ----------
    n_lags (int)
        History length in frames.
    temporal_basis (dict)
        The ``kinematic_encoding.temporal_basis`` settings block: ``type`` ("none" or "bspline"),
        ``n_splines``, ``spline_order``, ``l2_lag_alpha``, ``lambda_ridge``, ``lambda_smooth``,
        ``lambda_group``, ``smoothness_order`` (the last four read by the solver, not here),
        ``spike_end_correction`` ("none" or "drop_outside_and_rescale").

    Returns
    -------
    basis (np.ndarray | None)
        ``(n_lags, n_columns)`` scaled basis -- ``n_splines`` columns, one fewer under the spike-end
        correction -- or None when every frame keeps its own weight.
    """

    kind = temporal_basis["type"]
    if kind not in TEMPORAL_BASIS_TYPES:
        msg = f"temporal_basis.type must be one of {TEMPORAL_BASIS_TYPES}; got {kind!r}."
        raise ValueError(msg)
    if kind == "none":
        return None
    correction = temporal_basis["spike_end_correction"]
    if correction not in SPIKE_END_CORRECTIONS:
        msg = f"temporal_basis.spike_end_correction must be one of {SPIKE_END_CORRECTIONS}; got {correction!r}."
        raise ValueError(msg)
    basis = np.asarray(gam_bspline_basis(n_lags, temporal_basis["n_splines"], temporal_basis["spline_order"]),
                       dtype=np.float64)
    if correction == "drop_outside_and_rescale":
        # The spike's frame is the LAST lag, so the outermost spline at that end is the last column.
        basis = basis[:, :-1]
        mass = basis.sum(axis=0)
        basis = basis * np.sqrt(mass.max() / mass)[None, :]
    alpha = float(temporal_basis["l2_lag_alpha"])
    if alpha > 0.0:
        frames = np.arange(n_lags, dtype=np.float64)
        centres = (basis * frames[:, None]).sum(axis=0) / basis.sum(axis=0)
        lag_fraction = np.clip((n_lags - 1.0 - centres) / max(n_lags - 1.0, 1.0), 0.0, 1.0)
        basis = basis / np.sqrt(1.0 + alpha * lag_fraction ** 2)[None, :]
    return basis / float(basis.sum(axis=0).mean())

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


def quiet_block_sessions(session: dict, session_id: str, n_blocks: int, gap_frames: int) -> dict:
    """
    Description
    -----------
    Split one session's quiet anchors into contiguous time blocks presented as pseudo-sessions.

    The inner split for a pool of one session (``data_sufficiency.single_session_blocks``).
    The blocks look exactly like sessions, so ``inner_folds``, ``fit_quiet_model`` and the screen run over
    them unchanged. Blocks are contiguous because the kinematics are autocorrelated, and each is trimmed by
    ``gap_frames`` (derived from the history length) at both ends, so a validation frame's history cannot
    reach into a training block.

    Parameters
    ----------
    session (dict)
        The assembled session.
    session_id (str)
        Its identifier; block keys are derived from it.
    n_blocks (int)
        Number of contiguous blocks.
    gap_frames (int)
        Frames trimmed from each end of every block.

    Returns
    -------
    blocks (dict)
        ``{f"{session_id}#block{i}": session-like dict}``, skipping any block left empty by the trim.
    """

    quiet = np.sort(session["quiet"])
    edges = np.linspace(0, quiet.size, n_blocks + 1).astype(int)
    blocks = {}
    for index in range(n_blocks):
        piece = quiet[edges[index]:edges[index + 1]]
        if piece.size <= 2 * gap_frames:
            continue
        trimmed = piece[gap_frames:piece.size - gap_frames] if gap_frames > 0 else piece
        if trimmed.size == 0:
            continue
        blocks[f"{session_id}#block{index}"] = {**session, "quiet": trimmed}
    return blocks

def fit_quiet_model(per_session: dict, train_session_ids: list, feature_indices: list, n_lags: int,
                    rng, encoding_settings: dict, message_output=print) -> tuple:
    """
    Description
    -----------
    Fit the encoding model on the quiet anchors of the given sessions, using only the requested feature
    columns.

    Single-feature screens and multi-feature selection candidates both come through here. The majority
    class is subsampled with the log-prior offset correction and the fit is unweighted. The global
    ``max_train_total`` cap is divided across training sessions, since the design holds all of them at once.

    Each training session carries its own unpenalised baseline (``session_intercepts``), so the filters
    describe within-session modulation only.

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
    estimator (GroupElasticNetGLM | FrameFilterModel)
        The fitted model, always in frame space: the raw solver itself, or -- under
        ``temporal_basis.type == "bspline"`` -- the spline fit mapped back to one weight per frame. Its
        intercept is the training sessions' anchor-weighted mean baseline.
    base_rate (float)
        Pooled training spike rate before subsampling, on the true-population scale.
    """

    penalties = encoding_settings["raw_lag_penalties"]
    # The log-prior offset is the only implemented correction, so any other setting is refused.
    if encoding_settings["subsampling_correction"] != "log_prior_offset":
        msg = (f"the subsampling correction is the case-control log-prior offset; "
               f"subsampling_correction is {encoding_settings['subsampling_correction']!r}.")
        raise ValueError(msg)
    columns = list(feature_indices)
    design_blocks, label_blocks, offset_blocks, owner_blocks = [], [], [], []
    n_positive_total, n_negative_total = 0, 0
    per_session_cap = max(int(encoding_settings["max_train_total"] / max(len(train_session_ids), 1)), 1)
    for position, session_id in enumerate(train_session_ids):
        session = per_session[session_id]
        anchors, labels, offsets, n_positive, n_negative = subsample_quiet_anchors(
            session["quiet"], session["spike_frames"], session["n_frames"],
            encoding_settings["negatives_per_positive"], rng, per_session_cap)
        design_blocks.append(lagged_design(session["feature_time_series"][:, columns], anchors, n_lags))
        label_blocks.append(labels)
        offset_blocks.append(offsets)
        owner_blocks.append(np.full(anchors.size, position))
        n_positive_total += n_positive
        n_negative_total += n_negative

    design = np.vstack(design_blocks)
    labels_all, offsets_all = np.concatenate(label_blocks), np.concatenate(offset_blocks)
    owner = np.concatenate(owner_blocks)
    basis = lag_basis(n_lags, encoding_settings["temporal_basis"])
    if basis is not None:
        # Project each feature's history onto the basis; the fitted coefficients map back to one weight
        # per frame below. The reshape relies on the lagged design being feature-major.
        design = (design.reshape(design.shape[0], len(columns), n_lags) @ basis).reshape(design.shape[0], -1)

    def solve(session_offsets: np.ndarray, x: np.ndarray, n_groups: int) -> GroupElasticNetGLM:
        """One penalised fit of ``x`` (``n_groups`` column groups) with per-anchor session baselines added
        to the case-control offsets."""
        if basis is None:
            solver = GroupElasticNetGLM(
                n_features=n_groups, n_time_bins=n_lags, family="bernoulli",
                lambda_group=penalties["lambda_group"], lambda_smooth=penalties["lambda_smooth"],
                lambda_ridge=penalties["lambda_ridge"], smoothness_order=encoding_settings["smoothness_order"],
                debias_refit=False, max_iter=encoding_settings["solver"]["max_iter"],
                tol=encoding_settings["solver"]["tol"])
        else:
            basis_settings = encoding_settings["temporal_basis"]
            solver = GroupElasticNetGLM(
                n_features=n_groups, n_time_bins=basis.shape[1], family="bernoulli",
                lambda_group=basis_settings["lambda_group"],
                lambda_smooth=basis_settings["lambda_smooth"], lambda_ridge=basis_settings["lambda_ridge"],
                smoothness_order=basis_settings["smoothness_order"], debias_refit=False, max_iter=encoding_settings["solver"]["max_iter"],
                tol=encoding_settings["solver"]["tol"])
        solver.fit(x, labels_all, offset=offsets_all + session_offsets)
        return solver

    # Per-session baselines. The solver has one intercept and penalises every column, so:
    #   spline basis -- the k - 1 session indicators (relative to the first session) enter as one extra
    #     column group scaled by `indicator_scale`, which weakens the ridge on them by indicator_scale**2.
    #     This is safe only because the basis path has no smoothness penalty across columns.
    #   raw per-frame lags -- the smoothness penalty would couple indicator columns, so the baselines are
    #     backfitted: Newton on each baseline given the filters, refit, repeat to a fixed point.
    settings_block = encoding_settings["session_intercepts"]
    n_sessions = len(train_session_ids)
    weights = np.bincount(owner, minlength=n_sessions).astype(np.float64)
    deltas = np.zeros(n_sessions)
    use_sessions = settings_block["compute"] and n_sessions > 1
    if use_sessions and basis is not None:
        n_columns = basis.shape[1]
        if n_sessions - 1 > n_columns:
            msg = (f"{n_sessions} training sessions need {n_sessions - 1} indicator columns but one column "
                   f"group holds {n_columns}.")
            raise ValueError(msg)
        indicators = np.zeros((design.shape[0], n_columns))
        for k in range(1, n_sessions):
            indicators[owner == k, k - 1] = float(settings_block["indicator_scale"])
        solver = solve(np.zeros(owner.size), np.hstack([design, indicators]), len(columns) + 1)
        coefficients = np.asarray(solver.coef_, dtype=np.float64).reshape(len(columns) + 1, n_columns)
        deltas[1:] = coefficients[-1, :n_sessions - 1] * float(settings_block["indicator_scale"])
        filter_coefficients = coefficients[:len(columns)]
    else:
        solver = solve(deltas[owner], design, len(columns))
        if use_sessions:
            for _sweep in range(settings_block["max_sweeps"]):
                eta = design @ np.asarray(solver.coef_, dtype=np.float64) + float(solver.intercept_) + offsets_all
                previous = deltas.copy()
                for k in range(1, n_sessions):
                    rows = owner == k
                    for _step in range(settings_block["newton_max_steps"]):
                        p = 1.0 / (1.0 + np.exp(-(eta[rows] + deltas[k])))
                        hessian = float(np.sum(p * (1.0 - p)))
                        if hessian <= 0.0:
                            break
                        step = float(np.sum(labels_all[rows] - p)) / hessian
                        deltas[k] += step
                        if abs(step) < settings_block["newton_step_tolerance"]:
                            break
                solver = solve(deltas[owner], design, len(columns))
                if np.max(np.abs(deltas - previous)) < settings_block["tolerance"]:
                    break
        filter_coefficients = (np.asarray(solver.coef_, dtype=np.float64).reshape(len(columns), -1)
                               if basis is not None else None)
    # A new session's level is unknown to the model, so the returned intercept is the anchor-weighted mean
    # baseline; every consumer recalibrates it on the scored frames.
    intercept = float(solver.intercept_) + float(np.average(deltas, weights=weights))
    if basis is None:
        solver.intercept_ = intercept
        estimator = solver
    else:
        frame_weights = filter_coefficients @ basis.T
        estimator = FrameFilterModel(frame_weights.ravel(), intercept, bool(solver.converged_), int(solver.n_iter_))

    # A fit that exhausts max_iter otherwise looks exactly like a converged one, so report it here.
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

    Computed once per model and frame set: a null draw changes only the labels, never the predictions.

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
                       calibration_steps: int, calibration_bins: int, calibration: dict,
                       null_block_megabytes: float) -> np.ndarray:
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
        Equal-count predictor bins for the binned calibration; ``0`` runs the exact path. The responses
        are exact either way.
    calibration (dict)
        The ``kinematic_encoding.calibration`` block; every draw is scored by the same rule as the
        observed score (:func:`deviance_metrics.frozen_model_score`).
    null_block_megabytes (float)
        Memory budget for the label matrix of one block of draws (``kinematic_encoding.null_block_megabytes``).
        The Newton iteration holds several ``(n_frames, block)`` float64 arrays at once, so the working set
        is a few times this. It sets only how many draws run together; the scores do not depend on it.

    Returns
    -------
    null (np.ndarray)
        ``n_shuffles`` null scores.
    """

    # All shifts are drawn first, so the generator is consumed in a fixed order.
    shifts = [sample_circular_shift(rng, n_frames, fps, guard_seconds) for _ in range(n_shuffles)]

    # eta is frozen across draws, so the draws batch; the block is sized by memory, not by draw count.
    block_bytes = int(null_block_megabytes * 1024 * 1024)
    block = int(max(1, min(n_shuffles, block_bytes // max(1, frames.size * 8))))

    null = np.empty(n_shuffles, dtype=np.float64)
    for start in range(0, n_shuffles, block):
        stop = min(start + block, n_shuffles)
        labels = np.empty((frames.size, stop - start), dtype=np.float64)
        for column, draw in enumerate(range(start, stop)):
            shifted = shifted_spike_frames(spike_frames, shifts[draw], fps, n_frames)
            labels[:, column] = spike_labels_at_frames(shifted, frames, n_frames)
        null[start:stop] = frozen_model_null_scores(eta, labels, calibration_steps, calibration_bins,
                                                    calibration)
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

    The slope condition is needed because the calibrated score is blind to sign. The slope relates the
    model's output to the labels, so a negatively tuned unit still gives a positive slope; a negative one
    means the learned relationship flipped on the validation session.

    The screen uses its own shuffle count (``null.screen_n_shuffles``, 2,000), because an empirical p
    floors at ``1 / (n + 1)`` and must clear the Bonferroni bar ``screen_alpha / n_features`` (5.26e-4 for
    19 features at 0.01). The test session is never touched.

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
        ``survived``, ``converged``, and the per-inner-fold ``fold_scores`` and ``fold_slopes``.
    """

    encoding = settings["kinematic_encoding"]
    folds = inner_folds(pool_session_ids)
    n_features = per_session[pool_session_ids[0]]["feature_time_series"].shape[1]
    threshold = encoding["feature_selection"]["screen_alpha"] / n_features
    require_positive_slope = encoding["feature_selection"]["screen_require_positive_slope"]
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
            score, slope = frozen_model_score(eta, labels, np.zeros(labels.size, dtype=np.int64),
                                              encoding["solver"]["calibration_steps"], "bernoulli", 1.0,
                                              encoding["calibration"])
            fold_scores.append(score)
            fold_slopes.append(slope)
            fold_nulls.append(frozen_null_scores(
                eta, session["quiet"], session["spike_frames"], session["n_frames"], session["fps"],
                settings["null"]["screen_n_shuffles"], rng, settings["null"]["shuffle_guard_seconds"],
                encoding["solver"]["calibration_steps"],
                encoding["solver"]["null_calibration_bins"], encoding["calibration"],
                encoding["null_block_megabytes"]))
        score = finite_mean(np.asarray(fold_scores))
        slope = finite_mean(np.asarray(fold_slopes))
        p_value, at_floor = empirical_pvalue(np.vstack(fold_nulls).mean(axis=0), score)
        slope_ok = (not require_positive_slope) or (np.isfinite(slope) and slope > 0)
        survived = bool(score > 0 and slope_ok and np.isfinite(p_value) and p_value < threshold)
        rows.append({"feature": feature, "name": feature_names[feature], "score": score, "slope": slope,
                     "p": p_value, "at_floor": at_floor, "survived": survived,
                     "converged": bool(np.all(fold_converged)),
                     # Per-fold values show whether the folds agreed; the decision uses the mean.
                     "fold_scores": [float(value) for value in fold_scores],
                     "fold_slopes": [float(value) for value in fold_slopes]})
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
    paired differences. Pairing cancels the session-to-session variation both models share.

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
        One dict per step: ``step``, ``candidate``, ``mean``, ``fold_scores``, ``improvement``,
        ``standard_error``,
        ``decision``.
    """

    encoding = settings["kinematic_encoding"]
    # Only one method and one rule are implemented, so any other setting is refused.
    selection = encoding["feature_selection"]
    if selection["selection_method"] != "forward_d2":
        msg = (f"forward_select implements greedy forward selection by held-out D2 only; "
               f"selection_method is {selection['selection_method']!r}.")
        raise ValueError(msg)
    if selection["acceptance_rule"] != "paired_1se":
        msg = (f"the acceptance rule is the codebase's paired one-standard-error rule -- a candidate "
               f"must beat the incumbent by more than the SE of the PAIRED per-fold differences; "
               f"acceptance_rule is {selection['acceptance_rule']!r}.")
        raise ValueError(msg)
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
            values.append(frozen_model_score(eta, labels, np.zeros(labels.size, dtype=np.int64),
                                             encoding["solver"]["calibration_steps"], "bernoulli", 1.0,
                                             encoding["calibration"])[0])
        return np.asarray(values, dtype=np.float64), converged

    scored = [(feature, *fold_scores([feature])) for feature in survivors]
    scored = [(feature, values, converged) for feature, values, converged in scored
              if finite_mean(values) > 0]
    if not scored:
        return [], []
    anchor, incumbent_scores, anchor_converged = max(scored, key=lambda item: finite_mean(item[1]))
    incumbent_mean = finite_mean(incumbent_scores)
    selected = [anchor]
    # Each step keeps the per-inner-fold scores it was decided on.
    path = [{"step": 0, "candidate": feature_names[anchor], "mean": incumbent_mean,
             "improvement": np.nan, "standard_error": np.nan, "decision": "ANCHOR",
             "converged": anchor_converged,
             "fold_scores": [float(value) for value in incumbent_scores]}]
    message_output(f"    select step 0  ANCHOR {feature_names[anchor]:<32} score {incumbent_mean:+.5f} "
                   f"| per fold {np.array2string(incumbent_scores, precision=4)}"
                   f"{'' if anchor_converged else ' | UNCONVERGED'}")

    remaining = [feature for feature in survivors if feature != anchor]
    step = 1
    while remaining:
        best_feature, best_mean, best_scores, best_converged = None, -np.inf, None, True
        for feature in remaining:
            values, converged = fold_scores([*selected, feature])
            mean = finite_mean(values)
            if np.isfinite(mean) and mean > best_mean:
                best_feature, best_mean, best_scores, best_converged = feature, mean, values, converged
        if best_feature is None:
            break
        improvement, standard_error = paired_one_se_improvement(best_scores, incumbent_scores)
        accept = np.isfinite(improvement) and improvement > standard_error
        path.append({"step": step, "candidate": feature_names[best_feature], "mean": best_mean,
                     "improvement": improvement, "standard_error": standard_error,
                     "decision": "ACCEPT" if accept else "REJECT", "converged": best_converged,
                     "fold_scores": [float(value) for value in best_scores]})
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

def block_resample_anchors(anchors: np.ndarray, block_frames: int, rng) -> np.ndarray:
    """
    Description
    -----------
    Resample quiet anchors in contiguous BLOCKS, with replacement, to the original count.

    Blocks rather than frames, because adjacent quiet anchors share almost their entire history, and
    resampling frames independently would give a band far too narrow. Blocks span
    contiguous anchors, not wall-clock time, so across gaps in the quiet anchors a block covers more
    elapsed time, which errs toward a wider band.

    Parameters
    ----------
    anchors (np.ndarray)
        Quiet anchor frames, sorted.
    block_frames (int)
        Anchors per block; at least the predictor history, so a block is not shorter than the
        correlation it exists to respect.
    rng (np.random.Generator)
        Seeded generator.

    Returns
    -------
    resampled (np.ndarray)
        Sorted anchors, the same count as the input.
    """

    sorted_anchors = np.sort(anchors)
    if sorted_anchors.size == 0:
        return sorted_anchors
    width = max(int(block_frames), 1)
    n_blocks = int(np.ceil(sorted_anchors.size / width))
    starts = rng.integers(0, max(sorted_anchors.size - width, 0) + 1, size=n_blocks)
    picked = np.concatenate([sorted_anchors[start:start + width] for start in starts])
    return np.sort(picked[:sorted_anchors.size])


def filter_band(per_session: dict, train_session_ids: list, feature_indices: list, n_lags: int,
                rng, encoding_settings: dict, n_resamples: int, message_output=print) -> dict:
    """
    Description
    -----------
    The reported error band on a unit's temporal filter: refit the SELECTED model on block resamples
    of its quiet anchors and take the per-lag spread across the refits.

    The feature set is held fixed at the representative model's, so the band shows how much the
    coefficients move, not how much the selection moves. Computed during the run and stored, since
    recomputing it needs a full refit.

    Parameters
    ----------
    per_session (dict)
        Assembled session data.
    train_session_ids (list)
        The representative model's fitting sessions.
    feature_indices (list)
        The selected feature columns, frozen.
    n_lags (int)
        History length in frames.
    rng (np.random.Generator)
        Seeded generator.
    encoding_settings (dict)
        The ``kinematic_encoding`` block.
    n_resamples (int)
        Number of refits.
    message_output (Callable)
        Where progress is reported.

    Returns
    -------
    band (dict)
        ``filters`` ``(n_resamples, n_features, n_lags)``, the per-lag ``mean``, ``sd``, ``low`` and
        ``high`` (2.5th / 97.5th percentiles), ``intercepts``, and ``n_converged``.
    """

    block_frames = n_lags
    filters, intercepts, n_converged = [], [], 0
    for _draw in range(n_resamples):
        resampled = {}
        for session_id in train_session_ids:
            session = per_session[session_id]
            resampled[session_id] = {**session,
                                     "quiet": block_resample_anchors(session["quiet"],
                                                                     block_frames, rng)}
        estimator, _base_rate = fit_quiet_model(resampled, train_session_ids, feature_indices,
                                                n_lags, rng, encoding_settings, lambda *_a: None)
        filters.append(np.asarray(estimator.coef_).reshape(len(feature_indices), n_lags))
        intercepts.append(float(estimator.intercept_))
        n_converged += int(bool(estimator.converged_))

    stacked = np.stack(filters)
    message_output(f"  FILTER BAND  {n_resamples} block resamples of the representative model "
                   f"({len(feature_indices)} feature(s)) | {n_converged}/{n_resamples} converged")
    return {"filters": stacked,
            "mean": stacked.mean(axis=0), "sd": stacked.std(axis=0, ddof=1),
            "low": np.percentile(stacked, 2.5, axis=0),
            "high": np.percentile(stacked, 97.5, axis=0),
            "intercepts": np.asarray(intercepts), "n_converged": n_converged}


def filter_descriptors(coefficients: np.ndarray, feature_names: list, n_lags: int, fps: float) -> list:
    """
    Description
    -----------
    Per-feature shape summaries of a fitted temporal filter.

    Filters are compared by shape, not magnitude: ``norm`` is stored but is blind to shape and unreliable
    under collinearity, so ``roughness`` is the one to judge a filter by. ``peak_lag_seconds`` and
    ``centre_of_mass_seconds`` give where the filter is largest and how far back its weight sits on average.

    Parameters
    ----------
    coefficients (np.ndarray)
        Flat weight vector, ``n_features * n_lags``, lag-major within each feature.
    feature_names (list)
        Names in the same order as the feature blocks.
    n_lags (int)
        Lags per feature.
    fps (float)
        Frames per second, to report lags in seconds.

    Returns
    -------
    descriptors (list)
        One dict per feature: ``feature``, ``norm``, ``roughness``, ``peak_lag_seconds``,
        ``peak_weight``, ``centre_of_mass_seconds``.
    """

    weights = np.asarray(coefficients, dtype=np.float64).reshape(len(feature_names), n_lags)
    # the design puts lag 0 -- the anchor frame itself -- LAST, so lag age counts backwards from the end
    age_seconds = (n_lags - 1 - np.arange(n_lags)) / fps
    descriptors = []
    for index, name in enumerate(feature_names):
        row = weights[index]
        magnitude = np.abs(row)
        total = float(magnitude.sum())
        peak = int(np.argmax(magnitude))
        descriptors.append({
            "feature": name,
            "norm": float(np.linalg.norm(row)),
            "roughness": float(np.linalg.norm(np.diff(row))),
            "peak_lag_seconds": float(age_seconds[peak]),
            "peak_weight": float(row[peak]),
            "centre_of_mass_seconds": float(np.sum(magnitude * age_seconds) / total)
            if total > 0 else np.nan,
        })
    return descriptors


def spike_triggered_average(per_session: dict, session_ids: list, feature_indices: list,
                            n_lags: int, chunk_rows: int) -> dict:
    """
    Description
    -----------
    Spike-triggered average of the selected features over the history window, on quiet anchors.

    The STA is the robust readout; the fitted filter is roughly its deconvolution by the stimulus
    covariance, ``filter = C^-1 . STA``, which amplifies noise, so a clean STA beside a rougher filter is
    expected. Accumulated in chunks so the full lagged design is never held in memory.

    Parameters
    ----------
    per_session (dict)
        Assembled session data.
    session_ids (list)
        Sessions to pool over.
    feature_indices (list)
        The selected feature columns.
    n_lags (int)
        History length in frames.
    chunk_rows (int)
        Rows per chunk.

    Returns
    -------
    sta (dict)
        ``triggered`` and ``baseline``, each ``(n_features, n_lags)``, plus ``difference`` (the STA
        proper), ``n_spikes`` and ``n_anchors``.
    """

    columns = list(feature_indices)
    triggered = np.zeros((len(columns), n_lags), dtype=np.float64)
    baseline = np.zeros((len(columns), n_lags), dtype=np.float64)
    n_spikes = n_anchors = 0

    for session_id in session_ids:
        session = per_session[session_id]
        anchors = np.sort(session["quiet"])
        labels = spike_labels_at_frames(session["spike_frames"], anchors, session["n_frames"])
        for start in range(0, anchors.size, chunk_rows):
            block = anchors[start:start + chunk_rows]
            design = lagged_design(session["feature_time_series"][:, columns], block, n_lags)
            design = design.reshape(block.size, len(columns), n_lags)
            block_labels = labels[start:start + chunk_rows]
            triggered += design[block_labels > 0.5].sum(axis=0)
            baseline += design.sum(axis=0)
            n_spikes += int(block_labels.sum())
            n_anchors += int(block.size)

    triggered = triggered / max(n_spikes, 1)
    baseline = baseline / max(n_anchors, 1)
    return {"triggered": triggered, "baseline": baseline, "difference": triggered - baseline,
            "n_spikes": n_spikes, "n_anchors": n_anchors}
