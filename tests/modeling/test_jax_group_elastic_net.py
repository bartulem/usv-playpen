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
