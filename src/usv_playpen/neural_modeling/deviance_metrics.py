"""
@author: bartulem
Held-out scoring metrics for the neural encoding models.

Every statistic here is an explained deviance (D^2) of a frozen predictor against an intercept-only
reference, computed on frames the model was not fitted on. The reference is the scored frames' own rate,
and the predictor passes through a calibration (per-session intercepts and one shared scale) before it is
scored. The shipped scheme cross-fits that calibration over contiguous blocks, so no scored frame helps
estimate the numbers it is judged by.

The module provides:
- the scoring rule, :func:`frozen_model_score`, and its batched null-draw twin,
  :func:`frozen_model_null_scores`, which dispatch to the cross-fitted or in-sample calibrations;
- the in-sample calibrations (binary, pooled binary, pooled Poisson) and their batched and binned
  null-draw versions;
- the Newton cores for small logistic and Poisson fits;
- the descriptors stored alongside the score (:func:`encoding_metrics`,
  :func:`explained_deviance_vs_reference_rate`, :func:`area_under_roc`).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def finite_mean(values: np.ndarray) -> float:
    """
    Description
    -----------
    Mean of the finite entries, or NaN when there are none, without ``np.nanmean``'s all-NaN warning.
    A fold whose events are all masked comes back as NaN, visibly absent rather than quietly zero.

    Parameters
    ----------
    values (np.ndarray)
        Values that may contain NaN.

    Returns
    -------
    mean (float)
        Mean over the finite entries, or NaN.
    """

    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(finite.mean()) if finite.size else float("nan")


def finite_min(values) -> float:
    """
    Description
    -----------
    Minimum of the finite entries, or NaN when there are none (``np.nanmin`` without its all-NaN warning).
    The decoders' ``min_leave_one_fold_out`` goes through here, so a single NaN fold is skipped rather than
    turning the whole descriptor NaN.

    Parameters
    ----------
    values (array-like)
        Values that may contain NaN.

    Returns
    -------
    minimum (float)
        Minimum over the finite entries, or NaN.
    """

    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(finite.min()) if finite.size else float("nan")


def bernoulli_deviance(eta: np.ndarray, y: np.ndarray) -> float:
    """
    Description
    -----------
    Bernoulli deviance ``2 * sum(softplus(eta) - y * eta)``, computed via ``logaddexp`` so it stays stable
    for large ``|eta|`` where a naive ``log(1 + exp(eta))`` overflows.

    Parameters
    ----------
    eta (np.ndarray)
        Linear predictor.
    y (np.ndarray)
        0/1 labels.

    Returns
    -------
    deviance (float)
        Summed deviance.
    """

    return float(2.0 * np.sum(np.logaddexp(0.0, eta) - y * eta))



def area_under_roc(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    Description
    -----------
    Area under the ROC curve by rank sum. Scale-free, so it is stored as a descriptor next to the
    calibrated deviance but gates nothing.

    Ties take the average rank, counting a tied pair as half a win. This matters for the vocal-occurrence
    axis, whose predictor is a raw spike count taking a handful of integer values, so most pairs are tied.

    Parameters
    ----------
    scores (np.ndarray)
        Predictor values; higher should mean more likely positive.
    labels (np.ndarray)
        0/1 labels.

    Returns
    -------
    auroc (float)
        NaN when one class is absent.
    """

    positive = np.asarray(labels).astype(bool)
    if positive.all() or (~positive).all():
        return np.nan
    ranks = rankdata(np.asarray(scores, dtype=np.float64))
    n_positive, n_negative = float(positive.sum()), float((~positive).sum())
    return float((ranks[positive].sum() - n_positive * (n_positive + 1.0) / 2.0)
                 / (n_positive * n_negative))


def calibrate_intercept(offset: np.ndarray, y: np.ndarray, n_steps: int) -> float:
    """
    Description
    -----------
    Fit the intercept alone against a fixed offset: the level-only recalibration of a frozen model
    carried to a session it was not fitted on, which lets the baseline move while the slope stays fixed.

    Parameters
    ----------
    offset (np.ndarray)
        Fixed part of the linear predictor.
    y (np.ndarray)
        0/1 labels.
    n_steps (int)
        Maximum Newton iterations.

    Returns
    -------
    intercept (float)
        The fitted level.
    """

    return float(newton_logistic(np.ones((y.size, 1), dtype=np.float64), y,
                                 np.zeros(1, dtype=np.float64), n_steps, offset)[0])


def newton_logistic(x: np.ndarray, y: np.ndarray, ridge_vector: np.ndarray,
                    n_steps: int, offset: np.ndarray = None) -> np.ndarray:
    """
    Description
    -----------
    Ridge-penalized logistic regression by Newton steps, with an optional fixed offset. The shared core of
    every binary calibration fit in this module.

    Parameters
    ----------
    x (np.ndarray)
        Design matrix.
    y (np.ndarray)
        0/1 labels.
    ridge_vector (np.ndarray)
        Per-column ridge penalties; pass zeros for an unpenalized fit.
    n_steps (int)
        Maximum Newton iterations.
    offset (np.ndarray)
        Fixed additive term in the linear predictor, or None.

    Returns
    -------
    beta (np.ndarray)
        Fitted coefficients.
    """

    fixed = np.zeros(y.size, dtype=np.float64) if offset is None else offset
    beta = np.zeros(x.shape[1], dtype=np.float64)
    for _ in range(n_steps):
        eta = np.clip(x @ beta + fixed, -30.0, 30.0)
        mu = 1.0 / (1.0 + np.exp(-eta))
        weights = np.clip(mu * (1.0 - mu), 1e-8, None)
        gradient = x.T @ (y - mu) - ridge_vector * beta
        hessian = (x * weights[:, None]).T @ x + np.diag(ridge_vector)
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            break
        beta = beta + step
        if np.max(np.abs(step)) < 1e-9:
            break
    return beta


def newton_poisson(x: np.ndarray, counts: np.ndarray, ridge_vector: np.ndarray, n_steps: int,
                   offset: np.ndarray = None) -> np.ndarray:
    """
    Description
    -----------
    Ridge-penalized Poisson regression (log link) by damped Newton steps, with an optional fixed offset.

    The count-grain counterpart of :func:`newton_logistic`. Steps are damped to a largest coordinate change
    of 2, because a Poisson fit started from zero can overshoot on the first iteration when counts are large.

    Parameters
    ----------
    x (np.ndarray)
        Design matrix.
    counts (np.ndarray)
        Non-negative integer counts.
    ridge_vector (np.ndarray)
        Per-column ridge penalties; pass zeros for an unpenalized fit.
    n_steps (int)
        Maximum Newton iterations.
    offset (np.ndarray)
        Fixed additive term in the linear predictor (e.g. log exposure), or None.

    Returns
    -------
    beta (np.ndarray)
        Fitted coefficients.
    """

    fixed = np.zeros(counts.size, dtype=np.float64) if offset is None else offset
    beta = np.zeros(x.shape[1], dtype=np.float64)
    for _ in range(n_steps):
        mu = np.exp(np.clip(x @ beta + fixed, -30.0, 30.0))
        gradient = x.T @ (counts - mu) - ridge_vector * beta
        hessian = (x * mu[:, None]).T @ x + np.diag(ridge_vector) + 1e-12 * np.eye(beta.size)
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            break
        step = step / max(1.0, float(np.max(np.abs(step))) / 2.0)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-9:
            break
    return beta


def poisson_deviance(eta: np.ndarray, counts: np.ndarray) -> float:
    """
    Description
    -----------
    Poisson deviance of counts against the rates ``exp(eta)``.

    Parameters
    ----------
    eta (np.ndarray)
        Linear predictor, offset included.
    counts (np.ndarray)
        Observed counts.

    Returns
    -------
    deviance (float)
        ``2 * sum(counts * log(counts / mu) - (counts - mu))`` with the zero-count convention.
    """

    mu = np.exp(np.clip(eta, -30.0, 30.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(counts > 0, counts * np.log(counts / mu), 0.0)
    return float(2.0 * np.sum(term - (counts - mu)))


def pooled_poisson_explained_deviance(x: np.ndarray, counts: np.ndarray, session_index: np.ndarray,
                                      n_steps: int, exposure: float) -> tuple[float, float]:
    """
    Description
    -----------
    :func:`pooled_calibrated_explained_deviance` for COUNTS: one shared gain on the standardised predictor,
    one intercept per session, ``log(exposure)`` as a fixed offset, scored as Poisson deviance explained
    over the intercepts-only model.

    The in-sample transfer score at the count grain: the frozen quiet filter's mean log-odds per bin is the
    predictor and the spikes per bin the response. Counts keep the multi-spike frames that a binary frame
    label discards.

    Parameters
    ----------
    x (np.ndarray)
        Per-bin predictor, concatenated across sessions.
    counts (np.ndarray)
        Spikes per bin at the same bins.
    session_index (np.ndarray)
        Integer session index per bin.
    n_steps (int)
        Maximum Newton iterations for the calibration refit.
    exposure (float)
        Bin width in frames (or seconds -- any common unit), entered as a log offset.

    Returns
    -------
    explained_deviance (float)
        Poisson deviance explained over the per-session-intercept baseline.
    slope (float)
        The shared gain on standardized ``x``.
    """

    spread = float(np.std(x)) if np.size(x) else 0.0
    if spread < 1e-12 or counts.size == 0 or counts.sum() == 0:
        return np.nan, np.nan
    standardized = (x - np.mean(x)) / spread
    dummies = np.column_stack([(session_index == session).astype(np.float64)
                               for session in np.unique(session_index)])
    offset = np.full(counts.size, np.log(float(exposure)), dtype=np.float64)
    design = np.column_stack([standardized, dummies])
    beta = newton_poisson(design, counts, np.zeros(design.shape[1], dtype=np.float64), n_steps, offset)
    baseline_beta = newton_poisson(dummies, counts, np.zeros(dummies.shape[1], dtype=np.float64), n_steps,
                                   offset)
    baseline = poisson_deviance(dummies @ baseline_beta + offset, counts)
    if baseline <= 0:
        return np.nan, float(beta[0])
    return float(1.0 - poisson_deviance(design @ beta + offset, counts) / baseline), float(beta[0])


def intercept_only_deviance(y: np.ndarray, rate: float) -> float:
    """
    Description
    -----------
    Deviance of a constant model at the given probability, clipped away from 0 and 1.

    Parameters
    ----------
    y (np.ndarray)
        0/1 labels.
    rate (float)
        Constant predicted probability.

    Returns
    -------
    deviance (float)
        Summed deviance of the constant model.
    """

    probability = float(np.clip(rate, 1e-6, 1.0 - 1e-6))
    return bernoulli_deviance(np.full(y.size, np.log(probability / (1.0 - probability))), y)


def calibrated_explained_deviance(eta: np.ndarray, y: np.ndarray, n_steps: int) -> tuple[float, float]:
    """
    Description
    -----------
    Explained deviance after refitting ``y ~ a + b * z(eta)`` on the frames being scored (the in-sample
    scheme). The scale ``b`` cannot reorder frames, so it removes a frozen model's over- or under-stated
    magnitude and nothing else, and a predictor with useless ordering scores near 0 rather than far below
    it.

    Parameters
    ----------
    eta (np.ndarray)
        Frozen model's linear predictor at the scored frames.
    y (np.ndarray)
        0/1 labels at the same frames.
    n_steps (int)
        Maximum Newton iterations for the calibration refit.

    Returns
    -------
    explained_deviance (float)
        Explained deviance over an intercept-only model at the frames' own rate.
    slope (float)
        Fitted calibration slope on standardized ``eta``. Near 0 means the ordering is unusable here;
        negative means the relationship learned on training data points the wrong way on these frames.
    """

    spread = float(np.std(eta)) if np.size(eta) else 0.0
    if spread < 1e-12 or y.size == 0 or y.min() == y.max():
        return np.nan, np.nan
    standardized = (eta - np.mean(eta)) / spread
    design = np.column_stack([standardized, np.ones(standardized.size)])
    beta = newton_logistic(design, y, np.zeros(design.shape[1], dtype=np.float64), n_steps)
    baseline = intercept_only_deviance(y, float(y.mean()))
    if baseline <= 0:
        return np.nan, float(beta[0])
    return float(1.0 - bernoulli_deviance(design @ beta, y) / baseline), float(beta[0])


def batched_calibrated_explained_deviance(eta: np.ndarray, labels: np.ndarray,
                                          n_steps: int) -> np.ndarray:
    """
    Description
    -----------
    Calibrated explained deviance for many label vectors sharing one linear predictor.

    The in-sample null-draw inner loop. Every draw of a circular-shift null re-scores the same frozen
    ``eta`` against a different relabelling, so the design ``[z(eta), 1]`` is shared and only the response
    moves. The Newton iteration runs on an ``(n_frames, n_draws)`` block, and the per-draw 2x2 Hessian is
    inverted in closed form. The result matches :func:`calibrated_explained_deviance` to floating-point
    tolerance.

    Degenerate draws (all spikes or none) yield ``nan``; a draw whose Hessian goes singular keeps its last
    coefficients, as the scalar path does.

    Parameters
    ----------
    eta (np.ndarray)
        Linear predictor at the scored frames, shape ``(n_frames,)``. Frozen across draws.
    labels (np.ndarray)
        Binary responses, shape ``(n_frames, n_draws)`` -- one column per draw.
    n_steps (int)
        Maximum Newton iterations. Iteration stops early once every draw's step is below 1e-9.

    Returns
    -------
    scores (np.ndarray)
        Calibrated explained deviance per draw, shape ``(n_draws,)``, ``nan`` where undefined.
    """

    spread = float(np.std(eta)) if np.size(eta) else 0.0
    n_draws = labels.shape[1]
    if spread < 1e-12 or eta.size == 0 or n_draws == 0:
        return np.full(n_draws, np.nan, dtype=np.float64)

    standardized = ((eta - np.mean(eta)) / spread).astype(np.float64)
    response = labels.astype(np.float64, copy=False)

    # Degenerate draws carry no deviance to explain; the scalar path returns nan for them.
    rate = response.mean(axis=0)
    usable = (rate > 0.0) & (rate < 1.0)

    slope = np.zeros(n_draws, dtype=np.float64)
    intercept = np.zeros(n_draws, dtype=np.float64)
    squared = standardized * standardized
    active = usable.copy()

    for _ in range(n_steps):
        linear = np.clip(standardized[:, None] * slope[None, :] + intercept[None, :], -30.0, 30.0)
        mean = 1.0 / (1.0 + np.exp(-linear))
        weights = np.clip(mean * (1.0 - mean), 1e-8, None)
        residual = response - mean

        gradient_slope = standardized @ residual
        gradient_intercept = residual.sum(axis=0)

        # x.T W x for a two-column design, one 2x2 per draw, assembled as three vectors.
        hessian_ss = squared @ weights
        hessian_si = standardized @ weights
        hessian_ii = weights.sum(axis=0)

        determinant = hessian_ss * hessian_ii - hessian_si * hessian_si
        solvable = active & (np.abs(determinant) > 1e-30)
        if not np.any(solvable):
            break

        safe = np.where(solvable, determinant, 1.0)
        step_slope = np.where(solvable,
                              (hessian_ii * gradient_slope - hessian_si * gradient_intercept) / safe, 0.0)
        step_intercept = np.where(solvable,
                                  (hessian_ss * gradient_intercept - hessian_si * gradient_slope) / safe, 0.0)

        slope = slope + step_slope
        intercept = intercept + step_intercept

        active = solvable & (np.maximum(np.abs(step_slope), np.abs(step_intercept)) >= 1e-9)
        if not np.any(active):
            break

    linear = np.clip(standardized[:, None] * slope[None, :] + intercept[None, :], -30.0, 30.0)
    fitted = np.logaddexp(0.0, linear) - response * linear          # -log-likelihood per frame
    model_deviance = 2.0 * fitted.sum(axis=0)

    safe_rate = np.clip(rate, 1e-12, 1.0 - 1e-12)
    baseline = -2.0 * eta.size * (safe_rate * np.log(safe_rate)
                                  + (1.0 - safe_rate) * np.log1p(-safe_rate))

    scores = np.full(n_draws, np.nan, dtype=np.float64)
    valid = usable & (baseline > 0)
    scores[valid] = 1.0 - model_deviance[valid] / baseline[valid]
    return scores

def binned_calibrated_explained_deviance(eta: np.ndarray, labels: np.ndarray, n_steps: int,
                                         n_bins: int) -> np.ndarray:
    """
    Description
    -----------
    Calibrated explained deviance for many label vectors, with the frozen predictor summarised into bins.

    The fast in-sample null path. With the standardized predictor fixed, the responses enter the Newton
    iteration and the deviance only through two exact sums per draw, ``z @ y`` and ``sum(y)``. Only the
    predictor-side sums are approximated, by summarising ``z`` into equal-count bins, which follow the
    predictor's density rather than its range. The responses are never approximated.

    Parameters
    ----------
    eta (np.ndarray)
        Linear predictor at the scored frames, shape ``(n_frames,)``. Frozen across draws.
    labels (np.ndarray)
        Binary responses, shape ``(n_frames, n_draws)`` -- one column per draw.
    n_steps (int)
        Maximum Newton iterations, with the same early exit as the exact path.
    n_bins (int)
        Number of equal-count predictor bins.

    Returns
    -------
    scores (np.ndarray)
        Calibrated explained deviance per draw, shape ``(n_draws,)``, ``nan`` where undefined.
    """

    spread = float(np.std(eta)) if np.size(eta) else 0.0
    n_draws = labels.shape[1]
    if spread < 1e-12 or eta.size == 0 or n_draws == 0:
        return np.full(n_draws, np.nan, dtype=np.float64)

    standardized = ((eta - np.mean(eta)) / spread).astype(np.float64)
    response = labels.astype(np.float64, copy=False)

    # Exact, once: the only route by which the responses enter.
    cross = standardized @ response
    total = response.sum(axis=0)
    rate = total / standardized.size
    usable = (rate > 0.0) & (rate < 1.0)

    edges = np.quantile(standardized, np.linspace(0.0, 1.0, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    assignment = np.clip(np.searchsorted(edges, standardized, side="right") - 1, 0, n_bins - 1)
    count = np.bincount(assignment, minlength=n_bins).astype(np.float64)
    first = np.bincount(assignment, weights=standardized, minlength=n_bins)
    second = np.bincount(assignment, weights=standardized * standardized, minlength=n_bins)
    occupied = count > 0
    count, first, second = count[occupied], first[occupied], second[occupied]
    centre = first / count

    slope = np.zeros(n_draws, dtype=np.float64)
    intercept = np.zeros(n_draws, dtype=np.float64)
    active = usable.copy()

    for _ in range(n_steps):
        linear = np.clip(centre[:, None] * slope[None, :] + intercept[None, :], -30.0, 30.0)
        mean = 1.0 / (1.0 + np.exp(-linear))
        weights = np.clip(mean * (1.0 - mean), 1e-8, None)

        gradient_slope = cross - (first @ mean)
        gradient_intercept = total - (count @ mean)
        hessian_ss = second @ weights
        hessian_si = first @ weights
        hessian_ii = count @ weights

        determinant = hessian_ss * hessian_ii - hessian_si * hessian_si
        solvable = active & (np.abs(determinant) > 1e-30)
        if not np.any(solvable):
            break
        safe = np.where(solvable, determinant, 1.0)
        step_slope = np.where(solvable,
                              (hessian_ii * gradient_slope - hessian_si * gradient_intercept) / safe, 0.0)
        step_intercept = np.where(solvable,
                                  (hessian_ss * gradient_intercept - hessian_si * gradient_slope) / safe, 0.0)
        slope = slope + step_slope
        intercept = intercept + step_intercept
        active = solvable & (np.maximum(np.abs(step_slope), np.abs(step_intercept)) >= 1e-9)
        if not np.any(active):
            break

    linear = np.clip(centre[:, None] * slope[None, :] + intercept[None, :], -30.0, 30.0)
    model_deviance = 2.0 * (count @ np.logaddexp(0.0, linear) - slope * cross - intercept * total)

    safe_rate = np.clip(rate, 1e-12, 1.0 - 1e-12)
    baseline = -2.0 * standardized.size * (safe_rate * np.log(safe_rate)
                                           + (1.0 - safe_rate) * np.log1p(-safe_rate))
    scores = np.full(n_draws, np.nan, dtype=np.float64)
    valid = usable & (baseline > 0)
    scores[valid] = 1.0 - model_deviance[valid] / baseline[valid]
    return scores

def pooled_calibrated_explained_deviance(eta: np.ndarray, y: np.ndarray, session_index: np.ndarray,
                                         n_steps: int) -> tuple[float, float]:
    """
    Description
    -----------
    :func:`calibrated_explained_deviance` on held-out predictions pooled across cross-validation folds, with
    one shared slope and one intercept per session.

    Each fold's predictions are already held out, so concatenating them leaks nothing, and the calibration
    is then estimated once on every event rather than separately on small folds. Sessions
    keep their own intercepts; the slope is shared.

    Parameters
    ----------
    eta (np.ndarray)
        Held-out linear predictor, concatenated across folds.
    y (np.ndarray)
        0/1 labels at the same frames.
    session_index (np.ndarray)
        Integer session index per frame.
    n_steps (int)
        Maximum Newton iterations for the calibration refit.

    Returns
    -------
    explained_deviance (float)
        Explained deviance over a per-session-intercept baseline.
    slope (float)
        The shared calibration slope on standardized ``eta``.
    """

    spread = float(np.std(eta)) if np.size(eta) else 0.0
    if spread < 1e-12 or y.size == 0 or y.min() == y.max():
        return np.nan, np.nan
    standardized = (eta - np.mean(eta)) / spread
    dummies = np.column_stack([(session_index == session).astype(np.float64)
                               for session in np.unique(session_index)])
    design = np.column_stack([standardized, dummies])
    beta = newton_logistic(design, y, np.zeros(design.shape[1], dtype=np.float64), n_steps)
    baseline_beta = newton_logistic(dummies, y, np.zeros(dummies.shape[1], dtype=np.float64), n_steps)
    baseline = bernoulli_deviance(dummies @ baseline_beta, y)
    if baseline <= 0:
        return np.nan, float(beta[0])
    return float(1.0 - bernoulli_deviance(design @ beta, y) / baseline), float(beta[0])


def explained_deviance_vs_reference_rate(eta: np.ndarray, y: np.ndarray, reference_rate: float) -> dict:
    """
    Description
    -----------
    The uncalibrated level descriptor: how the frozen model scores against a constant at some external
    reference rate, typically the training epoch's. Reported, never gated on.

    It asks whether the model anticipates the unit's change in rate, not whether it knows which frames
    carry spikes, which is what the calibrated score asks.

    Parameters
    ----------
    eta (np.ndarray)
        Frozen model's linear predictor.
    y (np.ndarray)
        0/1 labels at the same frames.
    reference_rate (float)
        External reference probability, e.g. the training epoch's spike rate.

    Returns
    -------
    stats (dict)
        ``explained_deviance_vs_reference``, ``rate_ratio`` (scored rate over reference), ``log_loss`` and
        ``auroc``.
    """

    model_deviance = bernoulli_deviance(eta, y)
    reference_deviance = intercept_only_deviance(y, reference_rate)
    return {"explained_deviance_vs_reference": (1.0 - model_deviance / reference_deviance
                                                if reference_deviance > 0 else np.nan),
            "rate_ratio": float(y.mean() / np.clip(reference_rate, 1e-12, None)),
            "log_loss": model_deviance / (2.0 * y.size),
            "auroc": area_under_roc(eta, y)}


def encoding_metrics(eta: np.ndarray, y: np.ndarray, n_steps: int) -> dict:
    """
    Description
    -----------
    The full per-scoring-set metric roster for a frozen encoding model, from one calibration refit.

    The tested statistic is the calibrated explained deviance; the rest are descriptors that come with the
    same fit. ``llr_per_spike`` divides by the spike count, putting units of different
    firing rates on one scale. ``calibration_intercept`` is the ``a`` in ``y ~ a + b * z(eta)``, and says
    whether the model over- or under-predicts the rate. ``auroc`` is scale-free, so it is comparable
    between quiet and vocal frames.

    Parameters
    ----------
    eta (np.ndarray)
        Frozen model's linear predictor at the scored frames.
    y (np.ndarray)
        0/1 labels at the same frames.
    n_steps (int)
        Maximum Newton iterations for the calibration refit.

    Returns
    -------
    metrics (dict)
        ``deviance_explained``, ``calibration_slope``, ``calibration_intercept``, ``log_loss``,
        ``llr_per_spike``, ``brier``, ``auroc``, ``spike_rate``, ``n_frames``, ``n_spikes``.
    """

    labels = np.asarray(y, dtype=np.float64)
    n_spikes = float(labels.sum())
    spread = float(np.std(eta)) if np.size(eta) else 0.0
    if spread < 1e-12 or labels.size == 0 or labels.min() == labels.max():
        return {"deviance_explained": np.nan, "calibration_slope": np.nan,
                "calibration_intercept": np.nan, "log_loss": np.nan, "llr_per_spike": np.nan,
                "brier": np.nan, "auroc": np.nan, "spike_rate": float(labels.mean())
                if labels.size else np.nan, "n_frames": int(labels.size), "n_spikes": int(n_spikes)}

    standardized = (eta - np.mean(eta)) / spread
    design = np.column_stack([standardized, np.ones(standardized.size)])
    beta = newton_logistic(design, labels, np.zeros(2, dtype=np.float64), n_steps)
    fitted_eta = design @ beta
    probability = 1.0 / (1.0 + np.exp(-np.clip(fitted_eta, -30.0, 30.0)))

    baseline = intercept_only_deviance(labels, float(labels.mean()))
    model_deviance = bernoulli_deviance(fitted_eta, labels)
    return {"deviance_explained": float(1.0 - model_deviance / baseline) if baseline > 0 else np.nan,
            "calibration_slope": float(beta[0]), "calibration_intercept": float(beta[1]),
            "log_loss": float(model_deviance / (2.0 * labels.size)),
            "llr_per_spike": float((baseline - model_deviance) / (2.0 * n_spikes))
            if n_spikes > 0 else np.nan,
            "brier": float(np.mean((probability - labels) ** 2)),
            "auroc": area_under_roc(eta, labels),
            "spike_rate": float(labels.mean()), "n_frames": int(labels.size),
            "n_spikes": int(n_spikes)}


CALIBRATION_SCHEMES = ("in_sample", "crossfit")


def contiguous_blocks(group_index: np.ndarray, n_blocks: int) -> np.ndarray:
    """
    Description
    -----------
    Contiguous blocks within each group, numbered ``0 .. n_blocks - 1``, for cross-fitted scoring.

    Rows are assumed to be in time order within each group. Blocks are contiguous because neighbouring
    frames share their history, so an interleaved split would put near-copies of a scored frame into its
    own calibration data.

    Parameters
    ----------
    group_index (np.ndarray)
        Integer group (session) per row.
    n_blocks (int)
        Blocks per group.

    Returns
    -------
    blocks (np.ndarray)
        Block number per row.
    """

    blocks = np.zeros(group_index.size, dtype=np.int64)
    for group in np.unique(group_index):
        rows = np.flatnonzero(group_index == group)
        blocks[rows] = np.minimum((np.arange(rows.size) * n_blocks) // max(rows.size, 1), n_blocks - 1)
    return blocks


def _log_likelihood(eta: np.ndarray, y: np.ndarray, family: str) -> float:
    """Summed log-likelihood up to terms that cancel in a deviance ratio (Bernoulli logit or Poisson log)."""
    if family == "bernoulli":
        return float(np.sum(y * eta - np.logaddexp(0.0, eta)))
    return float(np.sum(y * eta - np.exp(np.clip(eta, -30.0, 30.0))))


def crossfitted_explained_deviance(x: np.ndarray, y: np.ndarray, group_index: np.ndarray, n_blocks: int,
                                   family: str, n_steps: int, exposure: float = 1.0) -> tuple[float, float]:
    """
    Description
    -----------
    Explained deviance of a frozen predictor where every number a scored row is judged by is estimated
    WITHOUT that row.

    The scored rows are cut into contiguous blocks within each group. For every block, the calibration (one
    intercept per group and one shared scale on the standardised predictor) is fitted on the other blocks
    and the block is scored with it. The reference is the cross-fitted per-group intercepts alone, so both
    sides are out of sample. The scale is fitted rather than fixed at 1, since at a fixed scale the null
    draws score far below zero and the test loses its resolution.

    Parameters
    ----------
    x (np.ndarray)
        Frozen predictor per row (per-frame log-odds, or a bin's mean log-odds).
    y (np.ndarray)
        Response per row: 0/1 labels (Bernoulli) or counts (Poisson).
    group_index (np.ndarray)
        Integer group per row; each group gets its own intercept (sessions).
    n_blocks (int)
        Contiguous blocks per group.
    family (str)
        ``"bernoulli"`` or ``"poisson"``.
    n_steps (int)
        Maximum Newton iterations per calibration fit.
    exposure (float)
        Poisson exposure (bin width), entered as ``log(exposure)``; ignored for Bernoulli.

    Returns
    -------
    explained_deviance (float)
        Cross-fitted deviance explained over the cross-fitted intercepts-only reference; NaN when the
        predictor is flat, the response degenerate, or a block's training rows leave a group without
        any events to estimate its intercept from.
    slope (float)
        Mean over blocks of the cross-fitted scale on the standardised predictor.
    """

    spread = float(np.std(x)) if np.size(x) else 0.0
    if spread < 1e-12 or y.size == 0 or float(np.sum(y)) == 0.0 or (family == "bernoulli" and y.min() == y.max()):
        return np.nan, np.nan
    standardized = (x - np.mean(x)) / spread
    groups = np.unique(group_index)
    position = np.searchsorted(groups, group_index)
    dummies = (position[:, None] == np.arange(groups.size)[None, :]).astype(np.float64)
    design = np.column_stack([standardized, dummies])
    offset = np.full(y.size, np.log(float(exposure)) if family == "poisson" else 0.0, dtype=np.float64)
    newton = newton_logistic if family == "bernoulli" else newton_poisson
    blocks = contiguous_blocks(position, n_blocks)
    ll_model, ll_reference, slopes = 0.0, 0.0, []
    for block in np.unique(blocks):
        train, test = blocks != block, blocks == block
        if np.any(np.bincount(position[train], weights=y[train], minlength=groups.size) <= 0.0):
            return np.nan, np.nan
        if family == "bernoulli" and np.any(np.bincount(position[train], weights=1.0 - y[train],
                                                        minlength=groups.size) <= 0.0):
            return np.nan, np.nan
        beta = newton(design[train], y[train], np.zeros(design.shape[1], dtype=np.float64), n_steps,
                      offset[train])
        beta_reference = newton(dummies[train], y[train], np.zeros(dummies.shape[1], dtype=np.float64),
                                n_steps, offset[train])
        ll_model += _log_likelihood(design[test] @ beta + offset[test], y[test], family)
        ll_reference += _log_likelihood(dummies[test] @ beta_reference + offset[test], y[test], family)
        slopes.append(float(beta[0]))
    if family == "bernoulli":
        ll_saturated = 0.0
    else:
        positive = y > 0
        ll_saturated = float(np.sum(y[positive] * np.log(y[positive]) - y[positive]))
    denominator = ll_saturated - ll_reference
    if denominator <= 0.0:
        return np.nan, float(np.mean(slopes))
    return float((ll_model - ll_reference) / denominator), float(np.mean(slopes))


def batched_crossfitted_explained_deviance(eta: np.ndarray, labels: np.ndarray, n_blocks: int, n_steps: int,
                                           n_bins: int) -> np.ndarray:
    """
    Description
    -----------
    :func:`crossfitted_explained_deviance` for many 0/1 label vectors sharing one frozen predictor and one
    group -- the inner loop of a quiet-frame circular-shift null.

    The same identity as :func:`binned_calibrated_explained_deviance` makes it fast: given the fixed
    standardised predictor, the responses enter a block's training fit only through ``z @ y`` and
    ``sum(y)`` over its training rows, and its test log-likelihood only through the same two sums over its
    test rows, all computed exactly. Only the predictor-side sums over ``log(1 + exp(a + b z))`` are taken
    over equal-count bins (``n_bins > 0``); ``n_bins == 0`` treats every row as its own bin, i.e. exact.
    The reference's cross-fitted intercept is closed-form (the training rows' rate).

    Parameters
    ----------
    eta (np.ndarray)
        Frozen predictor at the scored frames, shape ``(n_frames,)``, in time order.
    labels (np.ndarray)
        Binary responses, ``(n_frames, n_draws)``.
    n_blocks (int)
        Contiguous blocks.
    n_steps (int)
        Maximum Newton iterations.
    n_bins (int)
        Equal-count predictor bins, or 0 for the exact computation.

    Returns
    -------
    scores (np.ndarray)
        Cross-fitted explained deviance per draw, ``nan`` where a block's training rows are degenerate.
    """

    spread = float(np.std(eta)) if np.size(eta) else 0.0
    n_draws = labels.shape[1]
    if spread < 1e-12 or eta.size == 0 or n_draws == 0:
        return np.full(n_draws, np.nan, dtype=np.float64)
    standardized = ((eta - np.mean(eta)) / spread).astype(np.float64)
    response = labels.astype(np.float64, copy=False)
    blocks = contiguous_blocks(np.zeros(eta.size, dtype=np.int64), n_blocks)
    if n_bins > 0:
        edges = np.quantile(standardized, np.linspace(0.0, 1.0, n_bins + 1))
        edges[0], edges[-1] = -np.inf, np.inf
        assignment = np.clip(np.searchsorted(edges, standardized, side="right") - 1, 0, n_bins - 1)
        n_cells = n_bins
    else:
        assignment = np.arange(eta.size)
        n_cells = eta.size

    def cell_sums(rows):
        """Per-cell row count, sum of z and sum of z^2 over the given rows, occupied cells only."""
        count = np.bincount(assignment[rows], minlength=n_cells).astype(np.float64)
        first = np.bincount(assignment[rows], weights=standardized[rows], minlength=n_cells)
        second = np.bincount(assignment[rows], weights=standardized[rows] ** 2, minlength=n_cells)
        occupied = count > 0
        return count[occupied], first[occupied], second[occupied]

    ll_model = np.zeros(n_draws, dtype=np.float64)
    ll_reference = np.zeros(n_draws, dtype=np.float64)
    usable = np.ones(n_draws, dtype=bool)
    for block in np.unique(blocks):
        train, test = blocks != block, blocks == block
        cross_train = standardized[train] @ response[train]
        total_train = response[train].sum(axis=0)
        cross_test = standardized[test] @ response[test]
        total_test = response[test].sum(axis=0)
        n_train, n_test = float(train.sum()), float(test.sum())
        usable &= (total_train > 0.0) & (total_train < n_train)
        count, first, second = cell_sums(train)
        centre = first / count
        count_test, first_test, _ = cell_sums(test)
        centre_test = first_test / count_test
        slope = np.zeros(n_draws, dtype=np.float64)
        intercept = np.zeros(n_draws, dtype=np.float64)
        active = usable.copy()
        for _ in range(n_steps):
            linear = np.clip(centre[:, None] * slope[None, :] + intercept[None, :], -30.0, 30.0)
            mean = 1.0 / (1.0 + np.exp(-linear))
            weights = np.clip(mean * (1.0 - mean), 1e-8, None)
            gradient_slope = cross_train - (first @ mean)
            gradient_intercept = total_train - (count @ mean)
            hessian_ss = second @ weights
            hessian_si = first @ weights
            hessian_ii = count @ weights
            determinant = hessian_ss * hessian_ii - hessian_si * hessian_si
            solvable = active & (np.abs(determinant) > 1e-30)
            if not np.any(solvable):
                break
            safe = np.where(solvable, determinant, 1.0)
            step_slope = np.where(solvable, (hessian_ii * gradient_slope - hessian_si * gradient_intercept) / safe,
                                  0.0)
            step_intercept = np.where(solvable,
                                      (hessian_ss * gradient_intercept - hessian_si * gradient_slope) / safe, 0.0)
            slope = slope + step_slope
            intercept = intercept + step_intercept
            active = solvable & (np.maximum(np.abs(step_slope), np.abs(step_intercept)) >= 1e-9)
            if not np.any(active):
                break
        linear_test = np.clip(centre_test[:, None] * slope[None, :] + intercept[None, :], -30.0, 30.0)
        ll_model += slope * cross_test + intercept * total_test - count_test @ np.logaddexp(0.0, linear_test)
        rate = np.clip(total_train / n_train, 1e-12, 1.0 - 1e-12)
        reference_intercept = np.log(rate / (1.0 - rate))
        ll_reference += reference_intercept * total_test - n_test * np.logaddexp(0.0, reference_intercept)
    scores = np.full(n_draws, np.nan, dtype=np.float64)
    valid = usable & (ll_reference < 0.0)
    scores[valid] = (ll_model[valid] - ll_reference[valid]) / (-ll_reference[valid])
    return scores


def frozen_model_score(x: np.ndarray, y: np.ndarray, group_index: np.ndarray, n_steps: int, family: str,
                       exposure: float, calibration: dict) -> tuple[float, float]:
    """
    Description
    -----------
    The scoring rule for a frozen kinematic model, wherever it is scored: the per-feature screen, forward
    selection, the quiet score and the transfer, and every null draw of each.

    ``calibration["scheme"]`` selects the calibration: ``"crossfit"`` (the shipped scheme, see
    :func:`crossfitted_explained_deviance`) or ``"in_sample"`` (intercepts and scale refitted on the
    scored rows themselves).

    Parameters
    ----------
    x (np.ndarray)
        Frozen predictor per row.
    y (np.ndarray)
        Response per row (0/1 or counts).
    group_index (np.ndarray)
        Integer group (session) per row.
    n_steps (int)
        Maximum Newton iterations.
    family (str)
        ``"bernoulli"`` or ``"poisson"``.
    exposure (float)
        Poisson exposure; ignored for Bernoulli.
    calibration (dict)
        The ``kinematic_encoding.calibration`` block: ``scheme`` and ``n_blocks``.

    Returns
    -------
    explained_deviance (float)
        The score.
    slope (float)
        The scale on the standardised predictor (cross-fitted mean, or the in-sample fit).
    """

    scheme = calibration["scheme"]
    if scheme not in CALIBRATION_SCHEMES:
        msg = f"kinematic_encoding.calibration.scheme must be one of {CALIBRATION_SCHEMES}; got {scheme!r}."
        raise ValueError(msg)
    if scheme == "crossfit":
        return crossfitted_explained_deviance(x, y, group_index, int(calibration["n_blocks"]), family, n_steps,
                                              exposure)
    if family == "poisson":
        return pooled_poisson_explained_deviance(x, y, group_index, n_steps, exposure)
    if np.unique(group_index).size == 1:
        return calibrated_explained_deviance(x, y, n_steps)
    return pooled_calibrated_explained_deviance(x, y, group_index, n_steps)


def frozen_model_null_scores(eta: np.ndarray, labels: np.ndarray, n_steps: int, n_bins: int,
                             calibration: dict) -> np.ndarray:
    """
    Description
    -----------
    :func:`frozen_model_score` for many 0/1 label vectors on one group (a quiet-frame null block), batched.

    Parameters
    ----------
    eta (np.ndarray)
        Frozen predictor at the scored frames.
    labels (np.ndarray)
        ``(n_frames, n_draws)`` binary responses.
    n_steps (int)
        Maximum Newton iterations.
    n_bins (int)
        Equal-count predictor bins for the fast path; 0 for exact.
    calibration (dict)
        The ``kinematic_encoding.calibration`` block.

    Returns
    -------
    scores (np.ndarray)
        One score per draw.
    """

    scheme = calibration["scheme"]
    if scheme not in CALIBRATION_SCHEMES:
        msg = f"kinematic_encoding.calibration.scheme must be one of {CALIBRATION_SCHEMES}; got {scheme!r}."
        raise ValueError(msg)
    if scheme == "crossfit":
        return batched_crossfitted_explained_deviance(eta, labels, int(calibration["n_blocks"]), n_steps, n_bins)
    if n_bins > 0:
        return binned_calibrated_explained_deviance(eta, labels, n_steps, n_bins)
    return batched_calibrated_explained_deviance(eta, labels, n_steps)
