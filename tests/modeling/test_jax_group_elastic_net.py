"""
@author: bartulem

Unit tests for ``usv_playpen.modeling.jax_group_elastic_net``.

Coverage: the fixed-step Bernoulli FISTA path of ``GroupElasticNetGLM``. Its step is ``1 / L`` with
``L`` built from ``_power_iter_curvature``, and that curvature must be the one of the design WITH its
intercept column, because FISTA steps the weights and the bias together. Bounding the design alone
under-estimated ``L`` by more than threefold on a small-scale design (no smoothness penalty, small
ridge, lag-averaged columns of small amplitude), and the fit then oscillated, never converged, and
returned an intercept and objective far from the optimum.

Also covered, all against independent known answers: the ``double_precision`` scope (x64 on inside,
the prior setting restored after, including an already-enabled x64); the per-sample family NLL
against hand-computed Bernoulli and ``scipy.stats.poisson`` values; the constructor and design-shape
guards; the Poisson backtracking FISTA path against an L-BFGS-B optimum of the documented objective
(weights, offset, second-difference smoothness, ridge) and the scale-freedom of the sample weights;
exact group sparsity with the group-lasso KKT conditions, the relaxed refit against an unpenalised
fit of the selected columns (both families), and an all-zero solution at a huge penalty; and the
prediction and held-out log-loss methods against direct numpy computations.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import minimize
from scipy.special import expit, gammaln
from scipy.stats import poisson

from usv_playpen.modeling.jax_group_elastic_net import (
    GroupElasticNetGLM,
    _family_nll_per_sample,
    _power_iter_curvature,
    double_precision,
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


def _poisson_problem(seed: int = 0) -> tuple[np.ndarray, ...]:
    """
    Description
    -----------
    A Poisson design (2 features x 3 time bins) with non-uniform sample weights and a
    per-sample offset, the inputs the backtracking (Poisson) FISTA path is fed with in
    the count-grain models.

    Parameters
    ----------
    seed (int)
        Seed for the design, the offset, the weights and the counts.

    Returns
    -------
    x_design (np.ndarray)
        ``(1500, 6)`` design, columns of standard deviation 0.4.
    y (np.ndarray)
        ``(1500,)`` Poisson counts drawn at ``exp(0.3 + X @ w + offset)``.
    sample_weight (np.ndarray)
        ``(1500,)`` weights drawn uniformly in ``[0.5, 2]``.
    offset (np.ndarray)
        ``(1500,)`` offset of standard deviation 0.1.
    """
    rng = np.random.default_rng(seed)
    x_design = 0.4 * rng.standard_normal((1500, 6))
    offset = 0.1 * rng.standard_normal(1500)
    sample_weight = rng.uniform(0.5, 2.0, 1500)
    true_w = np.array([0.6, 0.5, 0.4, -0.3, -0.2, -0.1])
    y = rng.poisson(np.exp(0.3 + x_design @ true_w + offset)).astype(float)
    return x_design, y, sample_weight, offset


def _sparse_bernoulli_problem(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """
    Description
    -----------
    A Bernoulli design (2 features x 3 time bins) whose labels depend on feature 0 only;
    feature 1 is pure noise, so a group-lasso penalty above the noise gradient must zero
    feature 1's whole filter and keep feature 0's.

    Parameters
    ----------
    seed (int)
        Seed for the design and the labels.

    Returns
    -------
    x_design (np.ndarray)
        ``(4000, 6)`` standard-normal design.
    y (np.ndarray)
        ``(4000,)`` binary labels at ``sigmoid(X[:, :3] @ [0.8, 0.6, 0.4] - 0.5)``.
    """
    rng = np.random.default_rng(seed)
    x_design = rng.standard_normal((4000, 6))
    eta = x_design[:, :3] @ np.array([0.8, 0.6, 0.4]) - 0.5
    y = (rng.random(4000) < expit(eta)).astype(float)
    return x_design, y


def _poisson_reference(
    x_design: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
    offset: np.ndarray,
    lambda_smooth: float,
    lambda_ridge: float,
) -> np.ndarray:
    """
    Description
    -----------
    The exact optimum of the documented objective for ``lambda_group = 0`` and
    ``smoothness_order = 2`` on a 2-feature x 3-bin design: the sample-weighted mean
    Poisson NLL plus ``0.5 * lambda_smooth * ||D_2 W||^2`` plus
    ``0.5 * lambda_ridge * ||w||^2`` (intercept unpenalised), from a tightly converged
    L-BFGS-B fit.

    Parameters
    ----------
    x_design (np.ndarray)
        Design matrix ``(n_samples, 6)``.
    y (np.ndarray)
        Counts.
    sample_weight (np.ndarray)
        Per-sample weights.
    offset (np.ndarray)
        Per-sample offset added to the linear predictor.
    lambda_smooth (float)
        Second-difference smoothness strength.
    lambda_ridge (float)
        Ridge strength on the weights.

    Returns
    -------
    params (np.ndarray)
        ``(7,)`` optimum, the six weights followed by the intercept.
    """

    def objective(params: np.ndarray) -> float:
        """The documented smooth objective at ``params`` (weights, then intercept)."""
        w, b = params[:-1], params[-1]
        eta = x_design @ w + b + offset
        data = np.sum(sample_weight * (np.exp(eta) - y * eta)) / np.sum(sample_weight)
        smooth = np.sum(np.diff(w.reshape(2, 3), n=2, axis=1) ** 2)
        return float(
            data + 0.5 * lambda_smooth * smooth + 0.5 * lambda_ridge * np.sum(w**2)
        )

    result = minimize(
        objective,
        np.zeros(7),
        method="L-BFGS-B",
        options={"gtol": 1e-12, "ftol": 1e-15, "maxiter": 10000},
    )
    return result.x


class TestDoublePrecisionScope:
    """``double_precision`` turns x64 on for its block and restores the prior setting."""

    def test_x64_is_on_inside_and_restored_off_after(self):
        """
        Description
        -----------
        Starting from x64 off (the package-wide float32 default), the block must honour a
        float64 request and the flag must be off again afterwards, so importing or fitting
        this solver never leaks float64 into the float32 CNN code.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        previous = bool(jax.config.jax_enable_x64)
        try:
            jax.config.update("jax_enable_x64", False)
            with double_precision():
                assert jnp.zeros(2, dtype=jnp.float64).dtype == jnp.float64
            assert not jax.config.jax_enable_x64
        finally:
            jax.config.update("jax_enable_x64", previous)

    def test_an_already_enabled_x64_is_left_enabled(self):
        """
        Description
        -----------
        When x64 is already on before the block (enabled process-wide by the user code),
        the block must neither re-set nor switch it off on exit: restoring means restoring
        what was there, not forcing float32.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        previous = bool(jax.config.jax_enable_x64)
        try:
            jax.config.update("jax_enable_x64", True)
            with double_precision():
                assert jax.config.jax_enable_x64
            assert jax.config.jax_enable_x64
        finally:
            jax.config.update("jax_enable_x64", previous)


class TestFamilyNegativeLogLikelihood:
    """The per-sample NLL of each family, against hand-computed values."""

    def test_bernoulli_matches_the_logistic_nll(self):
        """
        Description
        -----------
        ``softplus(eta) - y * eta`` is ``-log Bernoulli(y | sigmoid(eta))``: at
        ``eta = 0, y = 1`` it is ``log 2``; at ``eta = 2, y = 0`` it is
        ``-log(1 - sigmoid(2)) = log(1 + e^2)``; at ``eta = -1, y = 1`` it is
        ``-log sigmoid(-1) = log(1 + e)``.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        with double_precision():
            eta = jnp.asarray([0.0, 2.0, -1.0], dtype=jnp.float64)
            y = jnp.asarray([1.0, 0.0, 1.0], dtype=jnp.float64)
            nll = np.asarray(_family_nll_per_sample(eta, y, "bernoulli"))
        expected = [np.log(2.0), np.log1p(np.exp(2.0)), np.log1p(np.exp(1.0))]
        np.testing.assert_allclose(nll, expected, rtol=1e-12)

    def test_poisson_matches_the_log_link_nll_without_the_factorial(self):
        """
        Description
        -----------
        ``exp(eta) - y * eta`` is ``-log Poisson(y | exp(eta)) - log(y!)``; checked
        against ``scipy.stats.poisson.logpmf`` with the ``log(y!)`` constant added back.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        eta_np = np.array([0.0, 1.5, -0.7, 2.2])
        y_np = np.array([0.0, 3.0, 1.0, 7.0])
        with double_precision():
            nll = np.asarray(
                _family_nll_per_sample(
                    jnp.asarray(eta_np, dtype=jnp.float64),
                    jnp.asarray(y_np, dtype=jnp.float64),
                    "poisson",
                )
            )
        expected = -poisson.logpmf(y_np, np.exp(eta_np)) - gammaln(y_np + 1.0)
        np.testing.assert_allclose(nll, expected, rtol=1e-12)

    def test_an_unknown_family_raises(self):
        """
        Description
        -----------
        Any family other than the two the solver implements must raise ``ValueError``
        rather than fall through to a silent default likelihood.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        with pytest.raises(ValueError, match="family must be"):
            _family_nll_per_sample(jnp.zeros(2), jnp.zeros(2), "gaussian")


class TestConstructorAndInputGuards:
    """Invalid configurations and mis-shaped designs are refused up front."""

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"family": "bernoulli", "smoothness_order": 3}, "smoothness_order"),
            ({"family": "gaussian", "smoothness_order": 1}, "family must be"),
        ],
    )
    def test_an_invalid_order_or_family_raises(self, kwargs, match):
        """
        Description
        -----------
        Only first- and second-difference smoothness and the Bernoulli / Poisson families
        are implemented; anything else must raise ``ValueError`` at construction.

        Parameters
        ----------
        kwargs (dict)
            The ``family`` and ``smoothness_order`` passed to the constructor.
        match (str)
            Regular expression the error message must match.

        Returns
        -------
        None
        """
        with pytest.raises(ValueError, match=match):
            GroupElasticNetGLM(
                n_features=2,
                n_time_bins=3,
                lambda_group=0.0,
                lambda_smooth=0.0,
                lambda_ridge=0.0,
                max_iter=10,
                tol=1e-6,
                **kwargs,
            )

    def test_a_design_with_the_wrong_column_count_raises(self):
        """
        Description
        -----------
        The design must have ``n_features * n_time_bins`` columns (the feature-major,
        lag-minor contract); five columns against a 2 x 3 model must raise ``ValueError``
        instead of being reshaped into the wrong groups.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        model = GroupElasticNetGLM(
            n_features=2,
            n_time_bins=3,
            lambda_group=0.0,
            lambda_smooth=0.0,
            lambda_ridge=0.0,
            smoothness_order=1,
            max_iter=10,
            tol=1e-6,
        )
        with pytest.raises(ValueError, match="columns"):
            model.fit(np.zeros((10, 5)), np.zeros(10))


class TestPoissonBacktrackingPath:
    """The Poisson host-loop FISTA with backtracking, against an independent optimum."""

    def test_zero_group_penalty_reaches_the_exact_optimum(self):
        """
        Description
        -----------
        With ``lambda_group = 0`` the objective is smooth and strictly convex, so the
        backtracking FISTA must converge to the L-BFGS-B optimum of the documented
        objective -- weighted Poisson NLL with offset, second-difference smoothness and
        ridge -- in weights and intercept alike.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        x_design, y, sample_weight, offset = _poisson_problem()
        model = GroupElasticNetGLM(
            n_features=2,
            n_time_bins=3,
            family="poisson",
            lambda_group=0.0,
            lambda_smooth=0.2,
            lambda_ridge=0.05,
            smoothness_order=2,
            debias_refit=False,
            max_iter=3000,
            tol=1e-10,
        )
        model.fit(x_design, y, sample_weight=sample_weight, offset=offset)
        reference = _poisson_reference(x_design, y, sample_weight, offset, 0.2, 0.05)
        assert model.converged_
        np.testing.assert_allclose(model.coef_, reference[:-1], atol=1e-6)
        assert model.intercept_ == pytest.approx(reference[-1], abs=1e-6)
        assert model.selected_features_.tolist() == [True, True]

    def test_sample_weights_are_scale_free(self):
        """
        Description
        -----------
        Weights are normalised to unit mean internally, so multiplying every weight by
        seven must leave the fitted weights and intercept unchanged to rounding.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        x_design, y, sample_weight, offset = _poisson_problem()
        fits = []
        for factor in (1.0, 7.0):
            model = GroupElasticNetGLM(
                n_features=2,
                n_time_bins=3,
                family="poisson",
                lambda_group=0.0,
                lambda_smooth=0.0,
                lambda_ridge=0.05,
                smoothness_order=1,
                debias_refit=False,
                max_iter=3000,
                tol=1e-10,
            )
            model.fit(x_design, y, sample_weight=factor * sample_weight, offset=offset)
            fits.append(np.r_[model.coef_, model.intercept_])
        np.testing.assert_allclose(fits[0], fits[1], atol=1e-9)

    def test_the_iteration_cap_reports_non_convergence(self):
        """
        Description
        -----------
        A fit stopped by ``max_iter`` before meeting the tolerance must say so:
        ``converged_`` False and ``n_iter_`` equal to the cap, so downstream code can tell an
        exhausted solve from a converged one.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        x_design, y, sample_weight, offset = _poisson_problem()
        model = GroupElasticNetGLM(
            n_features=2,
            n_time_bins=3,
            family="poisson",
            lambda_group=0.0,
            lambda_smooth=0.0,
            lambda_ridge=0.0,
            smoothness_order=1,
            debias_refit=False,
            max_iter=3,
            tol=1e-12,
        )
        model.fit(x_design, y, sample_weight=sample_weight, offset=offset)
        assert not model.converged_
        assert model.n_iter_ == 3

    def test_hitting_the_lipschitz_cap_never_accepts_an_ascent_step(self):
        """
        Description
        -----------
        One proximal-gradient step taken under a valid majorisation can only lower the
        objective. With ``backtrack_factor = 1e13`` the first rejected trial step pushes
        the Lipschitz estimate straight past the 1e12 cap; the step then taken must still
        not raise the objective above its warm-start value (zero weights, intercept at
        ``log(mean(y))``). The capped, rejected candidate is never applied: the fit stops
        at the warm start and reports ``converged_`` False (before the fix the rejected
        candidate was kept and the mean NLL jumped from 0.646 to about 9.4e10).

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        rng = np.random.default_rng(0)
        x_design = 3.0 * rng.standard_normal((500, 6))
        y = rng.poisson(np.exp(0.5 + x_design @ np.full(6, 0.1))).astype(float)
        model = GroupElasticNetGLM(
            n_features=2,
            n_time_bins=3,
            family="poisson",
            lambda_group=0.0,
            lambda_smooth=0.0,
            lambda_ridge=0.0,
            smoothness_order=1,
            debias_refit=False,
            max_iter=1,
            tol=1e-12,
            backtrack_factor=1e13,
        )
        model.fit(x_design, y)
        eta = x_design @ model.coef_ + model.intercept_
        start = np.log(np.mean(y))
        assert np.mean(np.exp(eta) - y * eta) <= np.mean(np.exp(start) - y * start)
        assert model.converged_ is False


class TestGroupSparsity:
    """The group-lasso prox zeroes whole feature filters, and the relaxed refit unshrinks the rest."""

    def test_the_noise_group_is_zeroed_exactly_and_the_kkt_conditions_hold(self):
        """
        Description
        -----------
        Without the refit, the returned point must satisfy the group-lasso optimality
        conditions: the noise feature's filter is EXACTLY zero and its smooth-gradient
        norm is at most ``lambda_group``; the kept feature's smooth gradient is cancelled
        by ``lambda_group * w_g / ||w_g||``.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        x_design, y = _sparse_bernoulli_problem()
        lambda_group, lambda_ridge = 0.05, 0.01
        model = GroupElasticNetGLM(
            n_features=2,
            n_time_bins=3,
            lambda_group=lambda_group,
            lambda_smooth=0.0,
            lambda_ridge=lambda_ridge,
            smoothness_order=1,
            debias_refit=False,
            max_iter=20000,
            tol=1e-12,
        )
        model.fit(x_design, y)
        assert model.selected_features_.tolist() == [True, False]
        assert np.all(model.coef_[3:] == 0.0)

        eta = x_design @ model.coef_ + model.intercept_
        gradient = x_design.T @ (expit(eta) - y) / y.size + lambda_ridge * model.coef_
        assert np.linalg.norm(gradient[3:]) <= lambda_group
        kept = model.coef_[:3]
        stationarity = gradient[:3] + lambda_group * kept / np.linalg.norm(kept)
        assert np.linalg.norm(stationarity) < 1e-5
        assert abs(np.mean(expit(eta) - y)) < 1e-6

    @pytest.mark.parametrize("family", ["bernoulli", "poisson"])
    def test_the_refit_equals_an_unpenalised_fit_of_the_selected_columns(self, family):
        """
        Description
        -----------
        The relaxed-lasso refit must give the selected feature exactly the coefficients
        and intercept a ``lambda_group = 0`` model fitted on its columns alone gives, and
        keep the unselected feature at exactly zero. Run for both families, so the
        Poisson backtracking path's soft-threshold branch is exercised too.

        Parameters
        ----------
        family (str)
            ``'bernoulli'`` or ``'poisson'``.

        Returns
        -------
        None
        """
        x_design, y_binary = _sparse_bernoulli_problem()
        y = y_binary
        if family == "poisson":
            rng = np.random.default_rng(3)
            eta = 0.4 * x_design[:, :3] @ np.array([0.8, 0.6, 0.4])
            y = rng.poisson(np.exp(eta)).astype(float)
        settings = {
            "family": family,
            "lambda_smooth": 0.1,
            "lambda_ridge": 0.01,
            "smoothness_order": 1,
            "max_iter": 5000,
            "tol": 1e-10,
        }
        model = GroupElasticNetGLM(
            n_features=2,
            n_time_bins=3,
            lambda_group=0.05,
            debias_refit=True,
            **settings,
        )
        model.fit(x_design, y)
        alone = GroupElasticNetGLM(
            n_features=1,
            n_time_bins=3,
            lambda_group=0.0,
            debias_refit=False,
            **settings,
        )
        alone.fit(x_design[:, :3], y)

        assert model.selected_features_.tolist() == [True, False]
        assert np.all(model.coef_[3:] == 0.0)
        np.testing.assert_allclose(model.coef_[:3], alone.coef_, rtol=1e-12)
        assert model.intercept_ == pytest.approx(alone.intercept_, rel=1e-12)

    def test_a_huge_penalty_zeroes_every_group_and_leaves_the_base_rate(self):
        """
        Description
        -----------
        A group penalty far above every gradient norm must zero all filters, select no
        feature (so no refit runs), and leave the intercept at the base-rate optimum
        ``logit(mean(y))``.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        x_design, y = _sparse_bernoulli_problem()
        model = GroupElasticNetGLM(
            n_features=2,
            n_time_bins=3,
            lambda_group=100.0,
            lambda_smooth=0.0,
            lambda_ridge=0.0,
            smoothness_order=1,
            debias_refit=True,
            max_iter=2000,
            tol=1e-12,
        )
        model.fit(x_design, y)
        rate = float(np.mean(y))
        assert not model.selected_features_.any()
        assert np.all(model.coef_ == 0.0)
        assert model.intercept_ == pytest.approx(np.log(rate / (1.0 - rate)), abs=1e-8)


def _fitted(family: str) -> tuple[GroupElasticNetGLM, np.ndarray, np.ndarray]:
    """
    Description
    -----------
    A small fitted model of the given family (2 features x 3 bins, ``lambda_group = 0``)
    together with fresh held-out data to score it on.

    Parameters
    ----------
    family (str)
        ``'bernoulli'`` or ``'poisson'``.

    Returns
    -------
    model (GroupElasticNetGLM)
        The fitted estimator.
    x_new (np.ndarray)
        ``(300, 6)`` held-out design.
    y_new (np.ndarray)
        ``(300,)`` held-out responses of the matching family.
    """
    rng = np.random.default_rng(11)
    x_design = 0.5 * rng.standard_normal((800, 6))
    x_new = 0.5 * rng.standard_normal((300, 6))
    w = np.array([0.5, 0.3, 0.1, -0.4, 0.0, 0.2])
    if family == "bernoulli":
        y = (rng.random(800) < expit(x_design @ w)).astype(float)
        y_new = (rng.random(300) < expit(x_new @ w)).astype(float)
    else:
        y = rng.poisson(np.exp(x_design @ w)).astype(float)
        y_new = rng.poisson(np.exp(x_new @ w)).astype(float)
    model = GroupElasticNetGLM(
        n_features=2,
        n_time_bins=3,
        family=family,
        lambda_group=0.0,
        lambda_smooth=0.0,
        lambda_ridge=0.01,
        smoothness_order=1,
        debias_refit=False,
        max_iter=2000,
        tol=1e-9,
    )
    model.fit(x_design, y)
    return model, x_new, y_new


class TestPredictionAndScoring:
    """The linear predictor, the rate and the held-out log-loss, against direct numpy."""

    @pytest.mark.parametrize("family", ["bernoulli", "poisson"])
    def test_eta_and_rate_follow_the_link(self, family):
        """
        Description
        -----------
        ``predict_eta`` is ``X @ coef_ + intercept_ (+ offset)`` and ``predict_rate`` its
        inverse link: ``sigmoid`` for Bernoulli, ``exp`` for Poisson, with the offset
        carried through.

        Parameters
        ----------
        family (str)
            ``'bernoulli'`` or ``'poisson'``.

        Returns
        -------
        None
        """
        model, x_new, _y_new = _fitted(family)
        offset = np.linspace(-0.5, 0.5, x_new.shape[0])
        expected_eta = x_new @ model.coef_ + model.intercept_ + offset
        np.testing.assert_allclose(
            model.predict_eta(x_new, offset=offset), expected_eta
        )
        np.testing.assert_allclose(
            model.predict_eta(x_new), x_new @ model.coef_ + model.intercept_
        )
        link = expit if family == "bernoulli" else np.exp
        np.testing.assert_allclose(
            model.predict_rate(x_new, offset=offset), link(expected_eta), rtol=1e-6
        )

    @pytest.mark.parametrize("family", ["bernoulli", "poisson"])
    def test_mean_log_loss_matches_a_direct_computation(self, family):
        """
        Description
        -----------
        Inside ``double_precision`` (the scope the solver runs in), ``mean_log_loss`` must
        equal the plain and the weighted mean of the family NLL computed directly in numpy.

        Parameters
        ----------
        family (str)
            ``'bernoulli'`` or ``'poisson'``.

        Returns
        -------
        None
        """
        model, x_new, y_new = _fitted(family)
        weights = np.linspace(0.2, 3.0, y_new.size)
        eta = x_new @ model.coef_ + model.intercept_
        if family == "bernoulli":
            nll = np.logaddexp(0.0, eta) - y_new * eta
        else:
            nll = np.exp(eta) - y_new * eta
        with double_precision():
            plain = model.mean_log_loss(x_new, y_new)
            weighted = model.mean_log_loss(x_new, y_new, sample_weight=weights)
        assert plain == pytest.approx(np.mean(nll), rel=1e-12)
        assert weighted == pytest.approx(
            np.sum(weights * nll) / np.sum(weights), rel=1e-12
        )

    def test_mean_log_loss_works_after_fit_without_a_precision_block(self):
        """
        Description
        -----------
        The natural use -- ``fit`` and then ``mean_log_loss`` on held-out data -- happens
        outside ``double_precision``, because ``fit`` only scopes x64 to its own body. The
        method must still score in float64 without warning (it scopes x64 itself, as
        ``fit`` does), matching a direct numpy computation to 1e-12.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        model, x_new, y_new = _fitted("bernoulli")
        eta = x_new @ model.coef_ + model.intercept_
        expected = np.mean(np.logaddexp(0.0, eta) - y_new * eta)
        assert model.mean_log_loss(x_new, y_new) == pytest.approx(expected, rel=1e-12)


def _poisson_objective(
    params: np.ndarray,
    x_design: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
    offset: np.ndarray,
    lambda_smooth: float,
    lambda_ridge: float,
    n_features: int,
    n_time_bins: int,
) -> float:
    """
    Description
    -----------
    The documented smooth objective for the Poisson family with first-difference
    smoothness: the sample-weighted mean Poisson NLL (no ``log(y!)``), plus
    ``0.5 * lambda_smooth * ||D_1 W||^2`` and ``0.5 * lambda_ridge * ||w||^2`` on the
    weights only.

    Parameters
    ----------
    params (np.ndarray)
        ``(n_features * n_time_bins + 1,)`` weights followed by the intercept.
    x_design (np.ndarray)
        Design matrix ``(n_samples, n_features * n_time_bins)``.
    y (np.ndarray)
        Counts ``(n_samples,)``.
    sample_weight (np.ndarray)
        Per-sample weights ``(n_samples,)``.
    offset (np.ndarray)
        Per-sample offset ``(n_samples,)`` added to the linear predictor.
    lambda_smooth (float)
        First-difference smoothness strength.
    lambda_ridge (float)
        Ridge strength on the weights.
    n_features (int)
        Number of feature groups.
    n_time_bins (int)
        Lags per feature.

    Returns
    -------
    objective (float)
        The smooth objective value at ``params``.
    """
    w, b = params[:-1], params[-1]
    eta = x_design @ w + b + offset
    data = np.sum(sample_weight * (np.exp(eta) - y * eta)) / np.sum(sample_weight)
    smooth = np.sum(np.diff(w.reshape(n_features, n_time_bins), n=1, axis=1) ** 2)
    return float(data + 0.5 * lambda_smooth * smooth + 0.5 * lambda_ridge * np.sum(w**2))


def _poisson_unpenalised_optimum(x_design: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Description
    -----------
    The exact optimum of the unweighted mean Poisson NLL (no penalty, no offset) on a
    2-feature x 3-bin design, from a tightly converged L-BFGS-B fit with the analytic
    gradient.

    Parameters
    ----------
    x_design (np.ndarray)
        Design matrix ``(n_samples, 6)``.
    y (np.ndarray)
        Counts ``(n_samples,)``.

    Returns
    -------
    params (np.ndarray)
        ``(7,)`` optimum, the six weights followed by the intercept.
    """
    augmented = np.column_stack([x_design, np.ones(x_design.shape[0])])

    def objective(params: np.ndarray) -> tuple[float, np.ndarray]:
        """Mean Poisson NLL and its gradient at ``params`` (weights, then intercept)."""
        eta = augmented @ params
        rate = np.exp(eta)
        return float(np.mean(rate - y * eta)), augmented.T @ (rate - y) / y.size

    result = minimize(
        objective,
        np.zeros(7),
        jac=True,
        method="L-BFGS-B",
        options={"gtol": 1e-13, "ftol": 1e-16, "maxiter": 20000},
    )
    return result.x


def _poisson_model(**overrides) -> GroupElasticNetGLM:
    """
    Description
    -----------
    A 2-feature x 3-bin Poisson estimator with no penalties, no refit, a 3000-iteration
    cap and ``tol = 1e-10``, with any keyword overridden.

    Parameters
    ----------
    **overrides
        Constructor keywords replacing the defaults above.

    Returns
    -------
    model (GroupElasticNetGLM)
        The unfitted estimator.
    """
    settings = {
        "n_features": 2,
        "n_time_bins": 3,
        "family": "poisson",
        "lambda_group": 0.0,
        "lambda_smooth": 0.0,
        "lambda_ridge": 0.0,
        "smoothness_order": 1,
        "debias_refit": False,
        "max_iter": 3000,
        "tol": 1e-10,
    }
    settings.update(overrides)
    return GroupElasticNetGLM(**settings)


class TestPowerIterationStart:
    """The power iteration starts from the unit-norm all-ones vector."""

    def test_one_iteration_returns_the_norm_of_the_gram_times_the_unit_all_ones_vector(self):
        """
        Description
        -----------
        After a single iteration the estimate is ``||G u||`` with ``G`` the weighted,
        mean-normalised Gram of ``[X, 1]`` and ``u`` the all-ones vector of UNIT norm, so it
        is a lower bound on the top eigenvalue at every iteration count. A start vector of
        any other norm scales the first estimate by that norm (and can overshoot the true
        eigenvalue, which would make the fixed FISTA step too small).

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        rng = np.random.default_rng(5)
        x_design = rng.standard_normal((400, 5)) + 0.3
        weights = rng.uniform(0.5, 2.0, 400)
        augmented = np.column_stack([x_design, np.ones(400)])
        gram = augmented.T @ (weights[:, None] * augmented) / np.sum(weights)
        unit = np.ones(6) / np.sqrt(6.0)
        exact = float(np.linalg.eigvalsh(gram).max())
        first = _power_iter_curvature(x_design, weights, n_iter=1)
        assert first == pytest.approx(float(np.linalg.norm(gram @ unit)), rel=1e-12)
        assert first <= exact


class TestBernoulliFixedStep:
    """The Bernoulli fit takes fixed steps of ``1 / L`` from the documented warm start."""

    def test_one_iteration_is_a_gradient_step_of_the_inverse_lipschitz_bound(self):
        """
        Description
        -----------
        With ``max_iter = 1`` and no group penalty, the Bernoulli fit must return exactly
        one gradient step from the warm start (zero weights, intercept
        ``logit(weighted mean y) - weighted mean offset``), with step ``1 / L`` and
        ``L = 0.25 * lambda_max(Gram of [X, 1]) + lambda_ridge + 4**order * lambda_smooth``.
        At zero weights the ridge and smoothness gradients vanish, so the step is
        ``-step * X^T W (sigmoid(eta0) - y) / sum(W)`` for the weights and the same
        residual mean for the intercept. The design is small-scale, so ``L`` is below 1
        (no clamp of the step), and the order is 2, so ``4**order`` differs from
        ``4 * order``.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        rng = np.random.default_rng(7)
        x_design = 0.05 * rng.standard_normal((1000, 6))
        offset = 0.3 * rng.standard_normal(1000)
        weights = rng.uniform(0.5, 2.0, 1000)
        y = (rng.random(1000) < expit(0.4 + offset)).astype(float)
        lambda_ridge, lambda_smooth = 0.02, 0.01
        model = GroupElasticNetGLM(
            n_features=2,
            n_time_bins=3,
            lambda_group=0.0,
            lambda_smooth=lambda_smooth,
            lambda_ridge=lambda_ridge,
            smoothness_order=2,
            debias_refit=False,
            max_iter=1,
            tol=1e-12,
        )
        model.fit(x_design, y, sample_weight=weights, offset=offset)

        normalised = weights / np.mean(weights)
        augmented = np.column_stack([x_design, np.ones(1000)])
        gram = augmented.T @ (normalised[:, None] * augmented) / np.sum(normalised)
        lipschitz = 0.25 * np.linalg.eigvalsh(gram).max() + lambda_ridge + 16.0 * lambda_smooth
        assert lipschitz < 1.0
        rate = np.sum(normalised * y) / np.sum(normalised)
        intercept0 = np.log(rate / (1.0 - rate)) - np.sum(normalised * offset) / np.sum(normalised)
        residual = normalised * (expit(intercept0 + offset) - y) / np.sum(normalised)
        np.testing.assert_allclose(model.coef_, -(x_design.T @ residual) / lipschitz, rtol=1e-8, atol=1e-14)
        assert model.intercept_ == pytest.approx(intercept0 - np.sum(residual) / lipschitz, rel=1e-10)
        assert model.n_iter_ == 1
        assert model.converged_ is False


class TestWarmStart:
    """With no iterations the fit returns the documented warm start."""

    @pytest.mark.parametrize(
        ("family", "labels"),
        [("bernoulli", "mixed"), ("bernoulli", "all_ones"), ("bernoulli", "all_zeros"), ("poisson", "mixed")],
    )
    def test_zero_iterations_return_zero_filters_and_the_base_rate_intercept(self, family, labels):
        """
        Description
        -----------
        At ``max_iter = 0`` the solver must hand back its warm start untouched: all-zero
        filters, ``n_iter_ = 0``, ``converged_`` False, and the intercept at the weighted
        base rate under the family link minus the weighted mean offset --
        ``logit(clip(y_bar, 1e-6, 1 - 1e-6))`` for Bernoulli (finite even when every label
        is 1 or every label is 0) and ``log(max(y_bar, 1e-6))`` for Poisson.

        Parameters
        ----------
        family (str)
            ``'bernoulli'`` or ``'poisson'``.
        labels (str)
            ``'mixed'`` (random responses), ``'all_ones'`` or ``'all_zeros'``.

        Returns
        -------
        None
        """
        rng = np.random.default_rng(9)
        x_design = rng.standard_normal((500, 6))
        offset = 0.4 * rng.standard_normal(500)
        weights = rng.uniform(0.5, 2.0, 500)
        if labels == "all_ones":
            y = np.ones(500)
        elif labels == "all_zeros":
            y = np.zeros(500)
        elif family == "bernoulli":
            y = (rng.random(500) < 0.3).astype(float)
        else:
            y = rng.poisson(0.6, 500).astype(float)
        model = GroupElasticNetGLM(
            n_features=2,
            n_time_bins=3,
            family=family,
            lambda_group=0.0,
            lambda_smooth=0.0,
            lambda_ridge=0.0,
            smoothness_order=1,
            debias_refit=False,
            max_iter=0,
            tol=1e-10,
        )
        model.fit(x_design, y, sample_weight=weights, offset=offset)
        rate = np.sum(weights * y) / np.sum(weights)
        mean_offset = np.sum(weights * offset) / np.sum(weights)
        if family == "bernoulli":
            rate = min(max(rate, 1e-6), 1.0 - 1e-6)
            expected = np.log(rate / (1.0 - rate)) - mean_offset
        else:
            expected = np.log(max(rate, 1e-6)) - mean_offset
        assert np.all(model.coef_ == 0.0)
        assert model.intercept_ == pytest.approx(expected, rel=1e-12)
        assert model.n_iter_ == 0
        assert model.converged_ is False


class TestAllZeroFeature:
    """A feature whose design columns are all zero stays exactly zero and finite."""

    def test_an_all_zero_feature_gets_a_zero_filter_and_leaves_the_rest_unchanged(self):
        """
        Description
        -----------
        A feature that never takes a non-zero value receives no data gradient, so with
        ridge it must be fitted as exactly zero -- the group prox (run at threshold 0 on
        the Bernoulli path) must map an all-zero filter to zero, not to ``0 / 0``. The
        other feature must then match a fit on its columns alone.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        rng = np.random.default_rng(13)
        x_design = np.zeros((1500, 6))
        x_design[:, :3] = rng.standard_normal((1500, 3))
        y = (rng.random(1500) < expit(x_design[:, :3] @ np.array([0.6, 0.3, -0.2]))).astype(float)
        settings = {
            "lambda_group": 0.0,
            "lambda_smooth": 0.0,
            "lambda_ridge": 0.01,
            "smoothness_order": 1,
            "debias_refit": False,
            "max_iter": 3000,
            "tol": 1e-11,
        }
        model = GroupElasticNetGLM(n_features=2, n_time_bins=3, **settings)
        model.fit(x_design, y)
        alone = GroupElasticNetGLM(n_features=1, n_time_bins=3, **settings)
        alone.fit(x_design[:, :3], y)
        assert np.all(model.coef_[3:] == 0.0)
        assert model.selected_features_.tolist() == [True, False]
        np.testing.assert_allclose(model.coef_[:3], alone.coef_, atol=1e-8)
        assert model.intercept_ == pytest.approx(alone.intercept_, abs=1e-8)


class TestPoissonConvergenceCheck:
    """The Poisson host loop stops on the true relative change, and only there."""

    def test_an_intercept_only_solution_converges_to_the_offset_corrected_base_rate(self):
        """
        Description
        -----------
        A group penalty far above every gradient keeps all filters at exactly zero, so only
        the intercept moves; with an offset its optimum is
        ``log(sum y / sum exp(offset))``, below the warm start
        ``log(mean y) - mean(offset)`` (Jensen), and the base rate is below one, so the
        intercept is negative and decreasing. The relative change must be measured as
        ``||(dw, db)|| / ||(w, b)||``: the fit must report convergence and land on the
        optimum to 1e-9.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        rng = np.random.default_rng(1)
        x_design = rng.standard_normal((1000, 6))
        offset = 0.5 * rng.standard_normal(1000)
        y = rng.poisson(0.3 * np.exp(offset)).astype(float)
        model = _poisson_model(lambda_group=1e3, max_iter=2000, tol=1e-12)
        model.fit(x_design, y, offset=offset)
        assert model.converged_
        assert np.all(model.coef_ == 0.0)
        assert model.intercept_ == pytest.approx(np.log(y.sum() / np.exp(offset).sum()), abs=1e-9)

    def test_negative_weights_converge_to_the_optimum(self):
        """
        Description
        -----------
        Every true weight is negative and the intercept is near zero, so any convergence
        scale built from signed sums or odd powers of the weights instead of their squared
        norm would go negative; the fit must still report convergence and match the
        L-BFGS-B optimum of the unpenalised Poisson NLL.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        rng = np.random.default_rng(2)
        x_design = 0.5 * rng.standard_normal((1500, 6))
        y = rng.poisson(np.exp(x_design @ np.full(6, -0.5))).astype(float)
        model = _poisson_model()
        model.fit(x_design, y)
        reference = _poisson_unpenalised_optimum(x_design, y)
        assert model.converged_
        np.testing.assert_allclose(np.r_[model.coef_, model.intercept_], reference, atol=1e-6)

    def test_a_zero_warm_start_and_a_small_solution_still_meet_the_relative_tolerance(self):
        """
        Description
        -----------
        With mean count exactly 1 the Poisson warm start is the origin (zero filters,
        intercept ``log 1 = 0``) and the optimum has norm about 0.1. The first iteration
        must not count as converged against the zero scale, and the stop must be on the
        RELATIVE change (not an absolute one, nor a change multiplied by the scale), so at
        ``tol = 1e-6`` the fit must match the L-BFGS-B optimum to 5e-7 (a criterion that is
        loose by the factor ``1 / ||(w, b)||`` stops near 1.3e-6).

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        rng = np.random.default_rng(3)
        x_design = rng.standard_normal((1000, 6))
        y = np.ones(1000)
        y[:300] = 0.0
        y[300:600] = 2.0
        rng.shuffle(y)
        model = _poisson_model(tol=1e-6)
        model.fit(x_design, y)
        reference = _poisson_unpenalised_optimum(x_design, y)
        assert model.converged_
        assert model.n_iter_ > 1
        np.testing.assert_allclose(np.r_[model.coef_, model.intercept_], reference, atol=5e-7)


class TestPoissonBacktrackingStep:
    """The Poisson backtracking line search and its group prox at a large Lipschitz constant."""

    @pytest.mark.parametrize(("scale", "base", "lambda_group", "min_doublings", "max_doublings"),
                             [(1.5, 1.2, 0.3, 3, 60), (0.1, -1.0, 0.01, 0, 0)])
    def test_one_iteration_matches_a_backtracked_proximal_gradient_step(self, scale, base, lambda_group,
                                                                        min_doublings, max_doublings):
        """
        Description
        -----------
        With ``max_iter = 1`` the Poisson fit must return exactly one backtracked proximal
        gradient step from the warm start: starting at the default ``L = lipschitz_init = 1``
        and multiplying by the default ``backtrack_factor = 2`` until the candidate
        ``prox(z - g / L)`` satisfies the majorisation
        ``F(cand) <= F(z) + <g, cand - z> + L/2 ||cand - z||^2`` (weights and intercept
        together). Recomputed here in numpy with the group soft-threshold at
        ``lambda_group / L``. The large-scale, high-rate design needs several doublings;
        the small-scale, low-rate one accepts the initial ``L = 1`` outright, so the
        starting estimate itself sets the step.

        Parameters
        ----------
        scale (float)
            Standard deviation of every design column.
        base (float)
            Log base rate of the simulated counts.
        lambda_group (float)
            Group-lasso strength.
        min_doublings (int)
            Fewest backtracking doublings the reference step must take.
        max_doublings (int)
            Most backtracking doublings the reference step may take.

        Returns
        -------
        None
        """
        rng = np.random.default_rng(17)
        x_design = scale * rng.standard_normal((800, 6))
        offset = 0.3 * rng.standard_normal(800)
        weights = rng.uniform(0.5, 2.0, 800)
        y = rng.poisson(np.exp(base + x_design @ np.array([0.2, 0.1, 0.0, 0.05, 0.0, 0.0]) + offset)).astype(float)
        lambda_smooth, lambda_ridge = 0.1, 0.01
        model = _poisson_model(lambda_group=lambda_group, lambda_smooth=lambda_smooth,
                               lambda_ridge=lambda_ridge, max_iter=1, tol=1e-12)
        model.fit(x_design, y, sample_weight=weights, offset=offset)

        normalised = weights / np.mean(weights)
        rate = np.sum(normalised * y) / np.sum(normalised)
        start = np.r_[np.zeros(6), np.log(rate) - np.sum(normalised * offset) / np.sum(normalised)]
        eta = x_design @ start[:-1] + start[-1] + offset
        residual = normalised * (np.exp(eta) - y) / np.sum(normalised)
        gradient = np.r_[x_design.T @ residual, np.sum(residual)]
        value = _poisson_objective(start, x_design, y, normalised, offset, lambda_smooth, lambda_ridge, 2, 3)
        lipschitz = 1.0
        n_doublings = 0
        while True:
            candidate = start - gradient / lipschitz
            filters = candidate[:-1].reshape(2, 3)
            norms = np.sqrt(np.sum(filters**2, axis=1) + 1e-12)
            candidate[:-1] = (filters * np.maximum(0.0, 1.0 - (lambda_group / lipschitz) / norms)[:, None]).ravel()
            step = candidate - start
            bound = value + gradient @ step + 0.5 * lipschitz * np.sum(step**2)
            cand_value = _poisson_objective(candidate, x_design, y, normalised, offset, lambda_smooth, lambda_ridge, 2, 3)
            if cand_value <= bound + 1e-12:
                break
            lipschitz *= 2.0
            n_doublings += 1
        assert min_doublings <= n_doublings <= max_doublings
        np.testing.assert_allclose(np.r_[model.coef_, model.intercept_], candidate, rtol=1e-9, atol=1e-12)

    def test_large_scale_group_lasso_meets_the_kkt_conditions(self):
        """
        Description
        -----------
        On a large-scale Poisson design (column standard deviation 2, so the accepted
        Lipschitz estimate is far above 1 and the prox threshold ``lambda_group / L``
        differs from ``lambda_group * L``), the converged group-lasso point must satisfy
        the optimality conditions: noise feature exactly zero with smooth-gradient norm at
        most ``lambda_group``, kept feature's gradient cancelled by
        ``lambda_group * w_g / ||w_g||``, and the intercept score at zero.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        rng = np.random.default_rng(4)
        x_design = 2.0 * rng.standard_normal((2000, 6))
        y = rng.poisson(np.exp(0.5 + x_design[:, :3] @ np.array([0.3, 0.2, 0.1]))).astype(float)
        lambda_group = 0.5
        model = _poisson_model(lambda_group=lambda_group, max_iter=5000, tol=1e-12)
        model.fit(x_design, y)
        eta = x_design @ model.coef_ + model.intercept_
        gradient = x_design.T @ (np.exp(eta) - y) / y.size
        kept = model.coef_[:3]
        assert model.converged_
        assert model.selected_features_.tolist() == [True, False]
        assert np.all(model.coef_[3:] == 0.0)
        assert np.linalg.norm(gradient[3:]) <= lambda_group
        assert np.linalg.norm(gradient[:3] + lambda_group * kept / np.linalg.norm(kept)) < 1e-8
        assert abs(np.mean(np.exp(eta) - y)) < 1e-8

    def test_an_ill_conditioned_design_converges_to_the_optimum(self):
        """
        Description
        -----------
        Strongly correlated columns (a shared factor of weight 0.9) make the problem
        ill-conditioned, so the accelerated iterates run far from the optimum before
        settling; the momentum must keep them where the objective is finite, so the fit
        must report convergence and match the L-BFGS-B optimum of the ridge-penalised
        Poisson NLL.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        rng = np.random.default_rng(0)
        shared = rng.standard_normal((1500, 1))
        x_design = 0.8 * (0.9 * shared + 0.3 * rng.standard_normal((1500, 6)))
        y = rng.poisson(np.exp(0.2 + x_design @ np.array([0.3, 0.2, 0.1, -0.1, -0.2, -0.3]))).astype(float)
        ones, zeros = np.ones(1500), np.zeros(1500)
        reference = minimize(
            _poisson_objective,
            np.zeros(7),
            args=(x_design, y, ones, zeros, 0.0, 1e-3, 2, 3),
            method="L-BFGS-B",
            options={"gtol": 1e-13, "ftol": 1e-16, "maxiter": 20000},
        ).x
        model = _poisson_model(lambda_ridge=1e-3, max_iter=5000)
        model.fit(x_design, y)
        assert model.converged_
        np.testing.assert_allclose(np.r_[model.coef_, model.intercept_], reference, atol=1e-5)


class TestRefitInputsAndBookkeeping:
    """The relaxed refit sees the caller's weights and offset; ``fit_time_`` is a real duration."""

    def test_the_refit_uses_the_sample_weights_and_the_offset(self):
        """
        Description
        -----------
        The relaxed refit must be fitted on the same sample weights and offset as the
        selection fit: its coefficients and intercept must equal those of a
        ``lambda_group = 0`` model fitted on the selected columns with those weights and
        that offset, and must differ from the same fit without them.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        x_design, y = _sparse_bernoulli_problem()
        rng = np.random.default_rng(21)
        weights = rng.uniform(0.2, 3.0, y.size)
        offset = 0.8 * rng.standard_normal(y.size)
        settings = {"lambda_smooth": 0.1, "lambda_ridge": 0.01, "smoothness_order": 1, "max_iter": 5000, "tol": 1e-10}
        model = GroupElasticNetGLM(n_features=2, n_time_bins=3, lambda_group=0.05, debias_refit=True, **settings)
        model.fit(x_design, y, sample_weight=weights, offset=offset)
        alone = GroupElasticNetGLM(n_features=1, n_time_bins=3, lambda_group=0.0, debias_refit=False, **settings)
        alone.fit(x_design[:, :3], y, sample_weight=weights, offset=offset)
        assert model.selected_features_.tolist() == [True, False]
        np.testing.assert_allclose(model.coef_[:3], alone.coef_, rtol=1e-12)
        assert model.intercept_ == pytest.approx(alone.intercept_, rel=1e-12)
        for kwargs in ({"sample_weight": weights}, {"offset": offset}):
            partial = GroupElasticNetGLM(n_features=1, n_time_bins=3, lambda_group=0.0, debias_refit=False, **settings)
            partial.fit(x_design[:, :3], y, **kwargs)
            assert np.max(np.abs(np.r_[partial.coef_, partial.intercept_] - np.r_[alone.coef_, alone.intercept_])) > 1e-3

    def test_fit_time_is_the_elapsed_duration_of_the_fit(self):
        """
        Description
        -----------
        ``fit_time_`` must be the wall-clock duration of the fit in seconds: a positive
        float no longer than the time measured around the ``fit`` call.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        x_design, y = _small_scale_problem(0.5)
        model = GroupElasticNetGLM(n_features=2, n_time_bins=3, lambda_group=0.0, lambda_smooth=0.0,
                                   lambda_ridge=0.01, smoothness_order=1, debias_refit=False,
                                   max_iter=50, tol=1e-9)
        before = time.perf_counter()
        model.fit(x_design, y)
        elapsed = time.perf_counter() - before
        assert isinstance(model.fit_time_, float)
        assert 0.0 < model.fit_time_ <= elapsed

    @pytest.mark.parametrize("family", ["bernoulli", "poisson"])
    def test_mean_log_loss_carries_the_offset(self, family):
        """
        Description
        -----------
        The held-out log-loss must score the linear predictor WITH the supplied offset:
        it must equal the direct numpy NLL of ``X @ coef_ + intercept_ + offset``.

        Parameters
        ----------
        family (str)
            ``'bernoulli'`` or ``'poisson'``.

        Returns
        -------
        None
        """
        model, x_new, y_new = _fitted(family)
        offset = np.linspace(-0.8, 0.8, y_new.size)
        eta = x_new @ model.coef_ + model.intercept_ + offset
        if family == "bernoulli":
            nll = np.logaddexp(0.0, eta) - y_new * eta
        else:
            nll = np.exp(eta) - y_new * eta
        assert model.mean_log_loss(x_new, y_new, offset=offset) == pytest.approx(np.mean(nll), rel=1e-12)
