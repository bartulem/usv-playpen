"""
@author: bartulem
Held-out scoring metrics for the neural encoding models.

Every statistic here is an explained-deviance (D^2) against an intercept-only reference, computed on frames
the model was not fitted on. Two design decisions are baked in, and both were forced by measurement rather
than preference.

The reference rate is the scored frames' own spike rate, not the training rate. A unit can fire at a very
different rate during vocalization than during silence (0.031 against 0.291 per frame for one PAG unit, a
9.3x step). Scoring against the training rate then rewards the model for predicting the epoch's rate change
rather than for discriminating spike from no-spike, which is a different claim.
``explained_deviance_vs_reference_rate`` is retained as a descriptor because that level information is
interesting, but it never gates anything.

The calibration slope is refitted on the frames being scored. Raw D^2 grades two things at once, whether
the model orders frames correctly and whether its predictions are the right size, and the second makes it
unusable against a frozen null. Shuffling destroys the ordering while leaving the magnitude intact, so a
loud misaligned filter scores far below an intercept-only model, at a depth set by its magnitude rather
than its error. Measured on a real filter: observed +0.151 against a null mean of -0.534, which let all 19
features clear the bar including 17 that scored worse than predicting the average rate. Refitting one
scalar removes exactly that and nothing else, since a scalar cannot reorder frames.
"""

from __future__ import annotations

import numpy as np


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


def bernoulli_deviance_terms(eta: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Description
    -----------
    Per-observation Bernoulli deviance contributions, i.e. :func:`bernoulli_deviance` before summation.

    Needed wherever the score has to be re-weighted by something other than the observation, for example by
    call rather than by frame when checking whether a long vocalization should contribute proportionally
    more evidence than a short one.

    Parameters
    ----------
    eta (np.ndarray)
        Linear predictor.
    y (np.ndarray)
        0/1 labels.

    Returns
    -------
    terms (np.ndarray)
        Per-observation contributions, summing to the scalar deviance.
    """

    return 2.0 * (np.logaddexp(0.0, eta) - y * eta)


def area_under_roc(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    Description
    -----------
    Area under the ROC curve by rank sum. Scale-free, so it is unaffected by the calibration issues that
    motivate :func:`calibrated_explained_deviance`, which makes it a useful companion diagnostic though not
    the gate, since the rest of the project scores in deviance.

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
    ranks = np.argsort(np.argsort(np.asarray(scores, dtype=np.float64))) + 1.0
    n_positive, n_negative = float(positive.sum()), float((~positive).sum())
    return float((ranks[positive].sum() - n_positive * (n_positive + 1.0) / 2.0)
                 / (n_positive * n_negative))


def newton_logistic(x: np.ndarray, y: np.ndarray, ridge_vector: np.ndarray,
                    n_steps: int, offset: np.ndarray = None) -> np.ndarray:
    """
    Description
    -----------
    Ridge-penalized logistic regression by Newton steps, with an optional fixed offset.

    Shared numerical core for every calibration refit in this module, so the small logistic fits that turn
    a frozen model's output into a score all go through one implementation rather than drifting apart.

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
    Explained deviance after refitting the calibration slope on the frames being scored.

    A model carried to data it was not fitted on usually gets the order of observations roughly right while
    overstating how much they differ; coefficients fitted on one dataset are habitually too extreme for
    another, which is why the calibration slope is a routine diagnostic in prediction-model validation.
    Fitting ``y ~ a + b * z(eta)`` and scoring the fitted values removes that overstatement and nothing
    else, since ``b`` multiplies the whole coefficient vector and cannot reorder anything.

    A predictor whose ordering is useless fits ``b`` toward 0 and scores near 0, rather than the large
    negative value raw D^2 would assign it. That is what makes the statistic usable against a frozen null,
    where the null draws must land near zero for the comparison to mean anything.

    Two parameters are estimated on the scored frames, negligible against the tens of thousands typically
    present, and the null draws are scored the same way so the comparison stays exact. The slope cannot be
    frozen from training data: a frozen slope leaves the null's magnitude intact and reintroduces the
    problem.

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

    spread = float(np.std(eta))
    if spread < 1e-12 or y.size == 0 or y.min() == y.max():
        return np.nan, np.nan
    standardized = (eta - np.mean(eta)) / spread
    design = np.column_stack([standardized, np.ones(standardized.size)])
    beta = newton_logistic(design, y, np.zeros(design.shape[1], dtype=np.float64), n_steps)
    baseline = intercept_only_deviance(y, float(y.mean()))
    if baseline <= 0:
        return np.nan, float(beta[0])
    return float(1.0 - bernoulli_deviance(design @ beta, y) / baseline), float(beta[0])


def pooled_calibrated_explained_deviance(eta: np.ndarray, y: np.ndarray, session_index: np.ndarray,
                                         n_steps: int) -> tuple[float, float]:
    """
    Description
    -----------
    :func:`calibrated_explained_deviance` on held-out predictions pooled across cross-validation folds, with
    one shared slope and one intercept per session.

    Averaging a per-fold score across folds wastes the data when the folds are unequal. Each fold's
    predictions are already held-out, so concatenating them leaks nothing; what it buys is that the
    calibration is estimated once on every event instead of separately on each fold. Measured on a
    three-session unit whose folds held 1,045, 962 and 372 events: fitting two parameters against a shuffled
    train on the smallest fold overfitted so badly that its null mean exceeded its own observed value and
    its spread swamped the average. Pooling dropped the null mean from +0.0081 to +0.0019 and the null
    spread from 0.0086 to 0.0028, moving the result from p = 0.065 to p = 9.9e-06 on the same data.

    Sessions keep their own intercepts because they are not on a common scale, while the slope, the
    quantity the claim is about, is shared and estimated on everything.

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

    spread = float(np.std(eta))
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

    It answers a different question from :func:`calibrated_explained_deviance`, namely whether the model
    knows the unit will fire faster here, rather than whether it knows which frames carry spikes. A model
    that predicts the rate change perfectly but orders frames at chance scores highly here and at zero
    there. Both are worth knowing; only the second is discrimination.

    A bare rate change does not inflate this: a synthetic with no kinematic structure and a 9.4x rate step
    returns exactly 0.00000, because the model's own predictions did not move either.

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
