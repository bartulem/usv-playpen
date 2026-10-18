"""
@author: bartulem

Unit tests for ``usv_playpen.modeling.jax_group_elastic_net``.

Coverage: the fixed-step Bernoulli FISTA path of ``GroupElasticNetGLM``. Its step is ``1 / L`` with
``L`` built from ``_power_iter_curvature``, and that curvature must be the one of the design WITH its
intercept column, because FISTA steps the weights and the bias together. Bounding the design alone
under-estimated ``L`` by more than threefold on a small-scale design (no smoothness penalty, small
ridge, lag-averaged columns of small amplitude), and the fit then oscillated, never converged, and
returned an intercept and objective far from the optimum.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from usv_playpen.modeling.jax_group_elastic_net import (
    GroupElasticNetGLM,
    _power_iter_curvature,
)


def _small_scale_problem(scale: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """
    Description
    -----------
    A Bernoulli design whose columns have amplitude ``scale`` (the regime of lag-averaged spline
    columns) and roughly balanced classes, so the intercept column dominates the curvature.

    Parameters
    ----------
    scale (float)
        Standard deviation of every design column.
    seed (int)
        Seed for the design and the labels.

    Returns
    -------
    x_design (np.ndarray)
        ``(2000, 6)`` design (2 features x 3 time bins).
    y (np.ndarray)
        ``(2000,)`` binary labels.
    """
    rng = np.random.default_rng(seed)
    x_design = scale * rng.standard_normal((2000, 6))
    y = (rng.random(2000) < expit(x_design @ (np.full(6, 0.3) / scale))).astype(float)
    return x_design, y


def _reference_objective(x_design: np.ndarray, y: np.ndarray, ridge: float) -> tuple[float, float]:
    """
    Description
    -----------
    The exact optimum of the mean Bernoulli negative log-likelihood plus ``0.5 * ridge * ||w||^2``
    (intercept unpenalised), from a tightly converged L-BFGS-B fit.

    Parameters
    ----------
    x_design (np.ndarray)
        Design matrix.
    y (np.ndarray)
        Binary labels.
    ridge (float)
        Ridge strength on the weights.

    Returns
    -------
    objective (float)
        The optimal objective value.
    intercept (float)
        The optimal intercept.
    """
    def objective(params: np.ndarray) -> float:
        """Mean Bernoulli negative log-likelihood plus the weight ridge."""
        eta = x_design @ params[:-1] + params[-1]
        return float(np.mean(np.logaddexp(0.0, eta) - y * eta) + 0.5 * ridge * np.sum(params[:-1] ** 2))

    result = minimize(objective, np.zeros(x_design.shape[1] + 1), method='L-BFGS-B',
                      options={'gtol': 1e-12, 'maxiter': 10000})
    return float(result.fun), float(result.x[-1])


class TestFistaStepIncludesTheIntercept:
    """The fixed FISTA step bounds the curvature of the design together with its intercept column."""

    def test_the_curvature_bound_counts_the_intercept_column(self):
        """The power iteration returns the top eigenvalue of the Gram of ``[X, 1]``, not of ``X``."""
        x_design, _y = _small_scale_problem(0.05)
        weights = np.ones(x_design.shape[0])
        augmented = np.column_stack([x_design, np.ones(x_design.shape[0])])
        exact = float(np.linalg.eigvalsh(augmented.T @ augmented / x_design.shape[0]).max())
        assert abs(_power_iter_curvature(x_design, weights, n_iter=200) - exact) < 1e-6 * exact

    def test_a_small_scale_design_converges_to_the_true_optimum(self):
        """
        Repro basis: design scale 0.05, ridge 0.1, no smoothness penalty. With the intercept left out
        of the Lipschitz bound the fit never converged in 5,000 iterations and returned objective 0.93
        against the optimum 0.69 (intercept 1.47 against 0.04); with it, the fit converges to the
        optimum.
        """
        ridge = 0.1
        x_design, y = _small_scale_problem(0.05)
        model = GroupElasticNetGLM(n_features=2, n_time_bins=3, lambda_group=0.0, lambda_smooth=0.0,
                                   lambda_ridge=ridge, smoothness_order=1, debias_refit=False,
                                   max_iter=5000, tol=1e-9)
        model.fit(x_design, y)
        coef = np.asarray(model.coef_, dtype=np.float64).ravel()
        intercept = float(np.asarray(model.intercept_))
        eta = x_design @ coef + intercept
        fitted_objective = float(np.mean(np.logaddexp(0.0, eta) - y * eta) + 0.5 * ridge * np.sum(coef ** 2))
        reference_objective, reference_intercept = _reference_objective(x_design, y, ridge)
        assert model.converged_
        assert abs(fitted_objective - reference_objective) < 1e-4
        assert abs(intercept - reference_intercept) < 1e-2
