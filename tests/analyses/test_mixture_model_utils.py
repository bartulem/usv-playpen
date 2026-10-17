"""
@author: bartulem
Targeted unit tests for the mixture_model_utils module.

These tests drive the branches of ``mixture_model_utils`` that the broader
analyses test-suite (``test_analyze.py``) does not exercise: the general-d
fixed-point mode finder in :func:`gmm_modes`, the 1D Newton gradient-fallback
step, the linear/negative-discriminant branches of
:func:`gmm_boundaries_logspace`, the degrees-of-freedom solver fallback in
:func:`_t_update_nu`, the empty-component skip in :func:`_sample_from_mixture`,
and the Student-t dispatch / progress-callback paths of :func:`bootstrap_lrt`.

Conventions mirror ``tests/analyses/test_analyze.py``: headless matplotlib,
seeded ``numpy`` RNG, sklearn ``GaussianMixture`` fixtures built from synthetic
log-space data, and direct dictionary access by key.
"""
from __future__ import annotations

import math
import struct
from unittest.mock import MagicMock

import matplotlib
import numpy as np
import polars as pls
import pytest
from numpy.linalg import cholesky, inv
from scipy import stats
from scipy.optimize import brentq
from scipy.special import digamma
from scipy.stats import invgauss
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold

# Headless matplotlib in case any imported helper touches a backend.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from matplotlib.patches import Polygon, Rectangle

from usv_playpen.analyses import mixture_model_utils
from usv_playpen.analyses.mixture_model_utils import (
    IGMixture,
    TMixture,
    _alpha,
    _bootstrap_lrt_replicate,
    _extract_params,
    _ig_logpdf_linear,
    _ig_mode,
    _log_gauss,
    _lr_statistic,
    _sample_from_mixture,
    _session_corrected_statistic,
    _session_design_effect,
    _splitmix64,
    _t_logpdf_1d,
    _t_update_nu,
    _tied_peak_lrt_replicate,
    bootstrap_lrt,
    fit_log_gmm,
    fit_log_ig_mixture,
    fit_log_t_mixture,
    fit_nested_by_splitting,
    fit_tied_nested_by_splitting,
    fit_tied_scale_t_mixture,
    gmm_boundaries_logspace,
    gmm_cdf_logspace,
    gmm_cv_neg_loglik,
    gmm_icl,
    gmm_modes,
    gmm_quantile_logspace,
    ig_mixture_cdf_logspace,
    ig_mixture_cv_neg_loglik,
    ig_mixture_icl,
    ig_mixture_quantile_logspace,
    plot_gmm_fit,
    qqplot_gmm,
    report_gmm_stats,
    report_ig_mixture_stats,
    report_t_mixture_stats,
    select_best_n_components,
    select_n_components_step_up_lrt,
    stable_subsample,
    summarize_best_gmm,
    summarize_best_ig_mixture,
    summarize_best_t_mixture,
    t_mixture_cdf_logspace,
    t_mixture_cv_neg_loglik,
    t_mixture_icl,
    t_mixture_modes,
    t_mixture_quantile_logspace,
    tied_peak_bootstrap_lrt,
    tied_scale_n_parameters,
)


def _fit_gmm(log_x, n_components, cov_type="full"):
    """
    Fits and returns a ``GaussianMixture`` on the supplied data.

    Parameters
    ----------
    log_x (np.ndarray)
        A (N, d) shape ndarray of training samples.
    n_components (int)
        Number of mixture components to fit.
    cov_type (str)
        sklearn covariance type; defaults to ``'full'``.

    Returns
    -------
    gmm (GaussianMixture)
        The fitted mixture.
    """
    return GaussianMixture(
        n_components=n_components,
        covariance_type=cov_type,
        random_state=0,
        n_init=2,
    ).fit(log_x)


def test_gmm_boundaries_logspace_linear_root_for_equal_variances():
    """
    Drives the near-equal-variance linear branch of
    :func:`gmm_boundaries_logspace` (source line 273).

    When two adjacent components share (almost) identical variances the
    quadratic coefficient ``a`` vanishes and the boundary solves the linear
    equation ``x = -c / b``. By constructing two unit-variance Gaussians with
    distinct, well-separated means and equal weights, the decision boundary at
    ``tau = 0.5`` must fall exactly at the midpoint of the two means.
    """
    gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=0)
    gmm.means_ = np.array([[-2.0], [2.0]])
    gmm.covariances_ = np.array([[[1.0]], [[1.0]]])
    gmm.weights_ = np.array([0.5, 0.5])
    gmm.precisions_cholesky_ = np.array([[[1.0]], [[1.0]]])

    log_b, sec_b = gmm_boundaries_logspace(gmm, tau=0.5)

    assert log_b.shape == (1,)
    assert sec_b.shape == (1,)
    # Equal variances + equal weights -> boundary is the mean midpoint (0.0).
    assert log_b[0] == pytest.approx(0.0, abs=1e-9)
    assert sec_b[0] == pytest.approx(np.exp(log_b[0]))


def test_gmm_boundaries_logspace_negative_discriminant_returns_nan():
    """
    Drives the negative-discriminant branch of
    :func:`gmm_boundaries_logspace` (source lines 277-278).

    With unequal variances the boundary equation is a genuine quadratic. A
    broad, dominant component sitting just left of a narrow, faint one can leave
    the quadratic with no real root (``disc < 0``); the function must emit
    ``NaN`` for that boundary rather than raising. A broad high-weight component
    at the lower mean paired with a tight low-weight component at the higher
    mean yields exactly this no-crossing configuration at ``tau = 0.5``.
    """
    gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=0)
    # Lower-mean component: broad, dominant. Higher-mean component: tight, faint.
    gmm.means_ = np.array([[0.0], [0.5]])
    gmm.covariances_ = np.array([[[1.0]], [[0.09]]])
    gmm.weights_ = np.array([0.9, 0.1])
    gmm.precisions_cholesky_ = np.array([[[1.0]], [[1.0 / 0.3]]])

    log_b, _ = gmm_boundaries_logspace(gmm, tau=0.5)

    assert log_b.shape == (1,)
    assert np.isnan(log_b[0])


def test_gmm_modes_1d_recovers_two_modes():
    """
    Exercises the 1D grid + Newton-polish path of :func:`gmm_modes`.

    A well-separated two-component log-Gaussian mixture has two density modes
    very close to the two component means. The returned modes must be sorted by
    descending density and lie near the generative means.
    """
    rng = np.random.default_rng(0)
    short = rng.normal(loc=-3.0, scale=0.2, size=400)
    long_ = rng.normal(loc=1.0, scale=0.2, size=400)
    log_x = np.concatenate([short, long_]).reshape(-1, 1)
    gmm = _fit_gmm(log_x, n_components=2)

    modes, dens = gmm_modes(gmm)

    assert modes.shape[1] == 1
    assert modes.shape[0] == 2
    assert dens.shape[0] == 2
    # Densities are returned in descending order.
    assert dens[0] >= dens[1]
    located = np.sort(modes.flatten())
    assert located[0] == pytest.approx(-3.0, abs=0.4)
    assert located[1] == pytest.approx(1.0, abs=0.4)


def test_gmm_modes_1d_gradient_fallback_on_nonnegative_hessian():
    """
    Drives the 1D Newton gradient-step fallback in :func:`gmm_modes`
    (source line 634).

    The Newton polish normally uses the analytic curvature (Hessian). When a
    grid-detected candidate sits where the curvature is non-negative or
    non-finite the function falls back to a small gradient step
    ``step = grad * 1e-3``. Capping ``max_iter`` at 1 forces at least one
    polish iteration to run for every detected peak; with a flat-topped
    near-uniform mixture (two close, broad components) at least one candidate
    triggers the fallback. The call must still return finite modes without
    raising.
    """
    rng = np.random.default_rng(3)
    a = rng.normal(loc=-0.05, scale=1.0, size=400)
    b = rng.normal(loc=0.05, scale=1.0, size=400)
    log_x = np.concatenate([a, b]).reshape(-1, 1)
    gmm = _fit_gmm(log_x, n_components=2)

    modes, dens = gmm_modes(gmm, max_iter=1)

    assert modes.shape[1] == 1
    assert np.isfinite(modes).all()
    assert np.isfinite(dens).all()


def test_gmm_modes_general_d_fixed_point_with_seeds():
    """
    Drives the general-d (d > 1) fixed-point branch of :func:`gmm_modes`
    (source lines 646-681), including the local-maximum verification.

    For a 2D two-component mixture the function dispatches to the
    Carreira-Perpinan fixed-point iteration seeded (by default) at the
    component means, then verifies each converged candidate is a local maximum
    along every axis before accepting it. A well-separated mixture must return
    finite, deduplicated 2D modes near the generative cluster centers.
    """
    rng = np.random.default_rng(1)
    c0 = rng.normal(loc=[-4.0, -4.0], scale=0.3, size=(300, 2))
    c1 = rng.normal(loc=[4.0, 4.0], scale=0.3, size=(300, 2))
    X = np.concatenate([c0, c1], axis=0)
    gmm = _fit_gmm(X, n_components=2)

    modes, dens = gmm_modes(gmm)

    assert modes.shape[1] == 2
    assert modes.shape[0] >= 1
    assert np.isfinite(modes).all()
    assert dens.shape[0] == modes.shape[0]
    # Recovered modes should sit near the two generative centers.
    located = modes[np.argsort(modes[:, 0])]
    assert located[0, 0] == pytest.approx(-4.0, abs=0.5)
    assert located[-1, 0] == pytest.approx(4.0, abs=0.5)


def test_gmm_modes_general_d_explicit_seeds_argument():
    """
    Drives the general-d branch of :func:`gmm_modes` with a caller-supplied
    ``seeds`` array (source lines 646-648), bypassing the default
    component-mean seeding.

    Passing explicit 2D seeds must reach the ``np.atleast_2d(seeds)`` coercion
    and still converge to the mixture's modes. The result must be finite and
    2D-shaped.
    """
    rng = np.random.default_rng(2)
    c0 = rng.normal(loc=[-3.0, 0.0], scale=0.3, size=(250, 2))
    c1 = rng.normal(loc=[3.0, 0.0], scale=0.3, size=(250, 2))
    X = np.concatenate([c0, c1], axis=0)
    gmm = _fit_gmm(X, n_components=2)

    seeds = np.array([[-3.0, 0.0], [3.0, 0.0]])
    modes, dens = gmm_modes(gmm, seeds=seeds)

    assert modes.shape[1] == 2
    assert np.isfinite(modes).all()
    assert dens.shape[0] == modes.shape[0]


def test_gmm_modes_general_d_rejects_saddle_seed():
    """
    Drives the saddle-point rejection branch of :func:`gmm_modes`
    (source lines 676-677: ``is_max = False`` / ``break``).

    Two identical-covariance, equal-weight Gaussians placed symmetrically at
    ``(-3, 0)`` and ``(3, 0)`` make the origin a saddle point of the mixture
    density (a minimum along the x-axis joining the two peaks, a maximum along
    y). Seeding the fixed-point iteration exactly at the origin keeps it pinned
    there by symmetry, so the converged candidate is a saddle. The local-maximum
    verification probes small coordinate perturbations, finds a higher density
    along x, flags the candidate as non-maximal, and rejects it — leaving an
    empty mode set.
    """
    gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=0)
    gmm.means_ = np.array([[-3.0, 0.0], [3.0, 0.0]])
    gmm.covariances_ = np.array([
        [[1.0, 0.0], [0.0, 1.0]],
        [[1.0, 0.0], [0.0, 1.0]],
    ])
    gmm.weights_ = np.array([0.5, 0.5])
    gmm.precisions_cholesky_ = np.array(
        [cholesky(inv(c)).T for c in gmm.covariances_]
    )

    modes, dens = gmm_modes(gmm, seeds=np.array([[0.0, 0.0]]), max_iter=5)

    assert modes.shape[0] == 0
    assert dens.shape[0] == 0


def test_t_update_nu_brentq_fallback_returns_50():
    """
    Drives the near-Gaussian side of the no-root case in :func:`_t_update_nu`.

    The degrees-of-freedom update solves a score equation via Brent's method
    on the bracket ``[2.001, 200.0]``. When the score is still positive at the
    upper end (the root lies above the bracket: effectively Gaussian) the
    function returns the near-Gaussian default of ``50.0``. A degenerate
    responsibility / latent-scale configuration (all-zero weights and unit
    latent scales) yields a score positive across the whole bracket.
    """
    n_components = 1
    z = np.zeros((n_components, 10))      # zero responsibilities
    u = np.ones((n_components, 10))       # unit latent scales
    nu = np.array([10.0])
    n_k = np.array([1e-10])               # near-zero effective count

    out = _t_update_nu(z[0], u[0], nu[0], n_k[0])

    assert out == 50.0


def test_t_update_nu_heavy_tails_clamp_to_the_lower_bound():
    """
    When the score is already negative at the lower end of the bracket (latent
    scales far from 1, i.e. tails heavier than ``nu = 2.001`` allows), the root
    lies below the bracket and the update returns the lower bound ``2.001``, not
    the near-Gaussian default. Inside the bracket the update is the score's root:
    the returned ``nu`` zeroes the score, and a larger mean of ``log(u) - u``
    (lighter tails) gives a larger ``nu``.
    """
    z = np.ones(50)
    heavy = _t_update_nu(z, np.full(50, 0.1), 10.0, 50.0)
    assert heavy == 2.001

    def score(nu, u, nu_old):
        psi_old = digamma((nu_old + 1.0) / 2.0) - np.log((nu_old + 1.0) / 2.0)
        return -digamma(nu / 2.0) + np.log(nu / 2.0) + 1.0 + psi_old + np.mean(np.log(u) - u)

    u_mid = np.full(50, 0.6)
    nu_mid = _t_update_nu(z, u_mid, 10.0, 50.0)
    assert 2.001 < nu_mid < 200.0
    assert score(nu_mid, u_mid, 10.0) == pytest.approx(0.0, abs=1e-9)
    u_light = np.full(50, 0.8)
    assert _t_update_nu(z, u_light, 10.0, 50.0) > nu_mid


def test_sample_from_mixture_gauss_skips_empty_component():
    """
    Drives the empty-component skip of the Gaussian branch in
    :func:`_sample_from_mixture` (source line 1807).

    With a two-component GMM whose first component carries (essentially) all
    the weight, the multinomial component assignment leaves the faint second
    component unsampled (``n_k == 0``) for a small draw, exercising the
    ``continue`` skip. The returned sample must have the requested length and
    be finite.
    """
    gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=0)
    gmm.means_ = np.array([[0.0], [50.0]])
    gmm.covariances_ = np.array([[[1.0]], [[1.0]]])
    gmm.weights_ = np.array([1.0 - 1e-9, 1e-9])
    gmm.precisions_cholesky_ = np.array([[[1.0]], [[1.0]]])

    rng = np.random.default_rng(0)
    out = _sample_from_mixture(gmm, N=20, rng=rng)

    assert out.shape == (20,)
    assert np.isfinite(out).all()
    # All draws should come from the dominant component near 0, not near 50.
    assert np.abs(out).max() < 10.0


def test_sample_from_mixture_t_branch_shape():
    """
    Exercises the ``TMixture`` branch of :func:`_sample_from_mixture`.

    Sampling from a Student-t mixture must dispatch through ``t_dist.rvs`` for
    each populated component and return a finite 1D draw of the requested size.
    """
    model = TMixture(
        weights=[0.5, 0.5],
        means=[-2.0, 2.0],
        covariances=[0.25, 0.25],
        nus=[8.0, 8.0],
    )
    rng = np.random.default_rng(0)
    out = _sample_from_mixture(model, N=100, rng=rng)

    assert out.shape == (100,)
    assert np.isfinite(out).all()


def test_bootstrap_lrt_rejects_k_alt_not_greater_than_k_null():
    """
    Confirms :func:`bootstrap_lrt` raises when ``K_alt <= K_null``.

    The test guards the precondition documented in the procedure and exercises
    the ``ValueError`` raised before any fitting occurs.
    """
    rng = np.random.default_rng(0)
    intervals = np.exp(rng.normal(loc=0.0, scale=0.5, size=50))
    with pytest.raises(ValueError, match="K_alt > K_null"):
        bootstrap_lrt(intervals, K_null=2, K_alt=1, B=2, model_class="gauss")


def test_bootstrap_lrt_rejects_unknown_model_class():
    """
    Confirms :func:`bootstrap_lrt` raises on an unsupported ``model_class``.

    Only ``'gauss'`` and ``'t'`` are valid; any other string must raise a
    ``ValueError`` from the dispatch branch.
    """
    rng = np.random.default_rng(0)
    intervals = np.exp(rng.normal(loc=0.0, scale=0.5, size=50))
    with pytest.raises(ValueError, match="model_class must be"):
        bootstrap_lrt(intervals, K_null=1, K_alt=2, B=2, model_class="laplace")


def test_bootstrap_lrt_t_dispatch_and_progress_callback():
    """
    Drives the Student-t dispatch (source line 1935) and the progress-callback
    branch (source line 1964) of :func:`bootstrap_lrt`.

    Running with ``model_class='t'`` selects ``fit_log_t_mixture`` as the fit
    function. With ``B = 10`` the ``(b + 1) % 10 == 0`` condition fires exactly
    once on the final replicate, so a mocked ``message_output`` callable must be
    invoked at least once. The returned result dictionary must carry the
    documented keys with the expected ``model_class`` and a valid p-value.
    """
    rng = np.random.default_rng(0)
    short = np.exp(rng.normal(loc=np.log(0.05), scale=0.1, size=120))
    long_ = np.exp(rng.normal(loc=np.log(2.0), scale=0.15, size=120))
    intervals = np.concatenate([short, long_])

    progress = MagicMock()
    result = bootstrap_lrt(
        intervals,
        K_null=1,
        K_alt=2,
        B=10,
        n_subsample=120,
        model_class="t",
        n_init_obs=1,
        n_init_boot=1,
        seed=0,
        message_output=progress,
    )

    assert result["model_class"] == "t"
    assert result["K_null"] == 1
    assert result["K_alt"] == 2
    assert result["B"] == 10
    assert result["lr_null"].shape == (10,)
    assert 0.0 <= result["p_value"] <= 1.0
    assert math.isfinite(result["lr_obs"])
    # The (b + 1) % 10 == 0 branch fires on the last replicate.
    progress.assert_called()


def test_bootstrap_lrt_gauss_dispatch_no_callback():
    """
    Exercises the Gaussian dispatch of :func:`bootstrap_lrt` with no progress
    callback (``message_output=None``) and a small ``B`` that never satisfies
    the ``% 10`` progress condition.

    This complements the Student-t test by covering the ``'gauss'`` fit-function
    selection and the silent-progress path. The result dictionary must report
    ``model_class == 'gauss'`` with a finite observed LR statistic.
    """
    rng = np.random.default_rng(1)
    short = np.exp(rng.normal(loc=np.log(0.05), scale=0.1, size=100))
    long_ = np.exp(rng.normal(loc=np.log(2.0), scale=0.15, size=100))
    intervals = np.concatenate([short, long_])

    result = bootstrap_lrt(
        intervals,
        K_null=1,
        K_alt=2,
        B=3,
        n_subsample=100,
        model_class="gauss",
        n_init_obs=1,
        n_init_boot=1,
        seed=0,
        message_output=None,
    )

    assert result["model_class"] == "gauss"
    assert result["lr_null"].shape == (3,)
    assert math.isfinite(result["lr_obs"])


def _synthetic_ig_intervals(seed=0, n_per=4000):
    """
    Draws a well-separated two-component inverse-Gaussian mixture sample
    (fast component IG(mu=0.07, lam=1.0), slow component IG(mu=3.0,
    lam=2.0)) for the IG fitter / interface tests.

    Parameters
    ----------
    seed (int)
        RNG seed.
    n_per (int)
        Samples per component.

    Returns
    -------
    x (np.ndarray)
        A (2 * n_per,) shape ndarray of strictly positive intervals.
    """
    rng = np.random.default_rng(seed)
    fast = invgauss.rvs(0.07 / 1.0, scale=1.0, size=n_per, random_state=rng)
    slow = invgauss.rvs(3.0 / 2.0, scale=2.0, size=n_per, random_state=rng)
    return np.concatenate([fast, slow])


def test_fit_log_ig_mixture_recovers_two_components():
    """The IG fitter must recover the means, shapes and weights of a
    well-separated synthetic two-component IG mixture."""
    x = _synthetic_ig_intervals()
    model, order = fit_log_ig_mixture(x, n_components=2, seed=0, n_init=3)
    mus = model.mus_[order]
    weights = model.weights_[order]
    assert abs(mus[0] - 0.07) < 0.01
    assert abs(mus[1] - 3.0) < 0.3
    assert abs(weights[0] - 0.5) < 0.05


def test_ig_mixture_log_density_integrates_to_one():
    """score_samples must be a proper density of log X: its exponential
    integrated over a wide log grid should be ~1 (validates the Jacobian
    convention that makes IG comparable with the log-space families)."""
    model = IGMixture(weights=[0.4, 0.6], mus=[0.08, 2.0], lambdas=[0.5, 3.0])
    grid = np.linspace(-12.0, 8.0, 20001)
    dens = np.exp(model.score_samples(grid))
    total = float(np.trapezoid(dens, grid))
    assert abs(total - 1.0) < 1e-3


def test_ig_mixture_interface_shapes_and_params():
    """The sklearn-compatible surface must match TMixture's conventions:
    shapes of score_samples / predict_proba, (K, 1) means_, and the
    3K - 1 parameter count feeding BIC/AIC."""
    model = IGMixture(weights=[0.5, 0.5], mus=[0.1, 1.0], lambdas=[1.0, 1.0])
    log_x = np.log(np.array([0.05, 0.1, 0.5, 1.0, 2.0]))
    assert model.score_samples(log_x).shape == (5,)
    z = model.predict_proba(log_x)
    assert z.shape == (5, 2)
    np.testing.assert_allclose(z.sum(axis=1), 1.0, atol=1e-12)
    assert model.means_.shape == (2, 1)
    assert model._n_params() == 5
    assert np.isfinite(model.bic(log_x))
    assert np.isfinite(model.aic(log_x))
    # slower component must dominate posterior at 2 s
    assert z[-1, 1] > 0.9


def test_sample_from_mixture_ig_branch_log_convention():
    """The bootstrap sampler's IG branch must return LOG-space samples whose
    per-component means match the model's mus."""
    model = IGMixture(weights=[0.5, 0.5], mus=[0.07, 3.0], lambdas=[1.0, 2.0])
    rng = np.random.default_rng(0)
    log_s = _sample_from_mixture(model, 20000, rng)
    assert log_s.shape == (20000,)
    x = np.exp(log_s)
    # overall mean is the weight-average of component means
    assert abs(float(np.mean(x)) - (0.5 * 0.07 + 0.5 * 3.0)) < 0.15


def test_bootstrap_lrt_ig_dispatch_smoke():
    """bootstrap_lrt must run end-to-end with model_class='ig' and return a
    finite observed LR statistic plus a B-length null sample."""
    x = _synthetic_ig_intervals(n_per=400)
    res = bootstrap_lrt(
        intervals_sec=x, K_null=1, K_alt=2, B=2, n_subsample=400,
        model_class="ig", n_init_obs=1, n_init_boot=1, seed=0,
    )
    assert res["model_class"] == "ig"
    assert np.isfinite(res["lr_obs"])
    assert res["lr_null"].shape == (2,)


def test_ig_mixture_cdf_quantile_roundtrip():
    """Quantile then CDF must round-trip across the mixture's support."""
    model = IGMixture(weights=[0.3, 0.7], mus=[0.08, 1.5], lambdas=[0.6, 2.5])
    qs = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
    log_q = ig_mixture_quantile_logspace(qs, model)
    back = ig_mixture_cdf_logspace(log_q, model)
    np.testing.assert_allclose(back, qs, atol=1e-6)


def test_bootstrap_lrt_parallel_deterministic_and_finite():
    """The n_jobs > 1 path must be reproducible run-to-run (per-replicate
    deterministic RNGs) and return finite statistics with the same shapes as
    the sequential path."""
    x = _synthetic_ig_intervals(n_per=300)
    kwargs = dict(intervals_sec=x, K_null=1, K_alt=2, B=4, n_subsample=300,
                  model_class="ig", n_init_obs=1, n_init_boot=1, seed=0)
    res1 = bootstrap_lrt(n_jobs=2, **kwargs)
    res2 = bootstrap_lrt(n_jobs=2, **kwargs)
    np.testing.assert_allclose(res1["lr_null"], res2["lr_null"])
    assert np.isfinite(res1["lr_obs"])
    assert res1["lr_null"].shape == (4,)
    # observed statistic does not depend on the bootstrap path
    res_seq = bootstrap_lrt(n_jobs=1, **kwargs)
    np.testing.assert_allclose(res_seq["lr_obs"], res1["lr_obs"])



def test_plot_gmm_fit_bin_means_are_below_the_point_wise_apex():
    """A bar is the density AVERAGED over its bin; the mixture curve is the
    density AT a point. Wherever a bin straddles the peak the average must be
    the smaller of the two, which is why a point-wise curve reads as an
    overshoot against a histogram even for a perfect fit -- and the gap widens
    with bin width, so it is a property of the drawing, not of the model."""
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 0.25, 20000)
    model = GaussianMixture(n_components=1, random_state=0).fit(x.reshape(-1, 1))
    apex = float(np.exp(model.score_samples(np.array([[0.0]])))[0])

    def peak_bin_mean(n_bins: int) -> float:
        """Model density averaged over whichever bin contains the mode."""
        edges = np.histogram_bin_edges(x, bins=n_bins)
        idx = int(np.searchsorted(edges, 0.0) - 1)
        fine = np.linspace(edges[idx], edges[idx + 1], 256)
        return float(np.trapezoid(np.exp(model.score_samples(fine.reshape(-1, 1))), fine)
                     / (edges[idx + 1] - edges[idx]))

    coarse, fine_bins = peak_bin_mean(8), peak_bin_mean(80)
    # the bin average never reaches the apex, and closes on it as bins narrow
    assert coarse < apex
    assert fine_bins < apex
    assert coarse < fine_bins

    # both draw modes render without disturbing the axes contract
    for as_bin_means in (False, True):
        fig, ax = plot_gmm_fit(model=model, x=x, bins=40,
                               density_as_bin_means=as_bin_means)
        assert ax.get_ylabel() == "Density"
        plt.close(fig)


def test_plot_gmm_fit_bin_means_conserve_probability_mass():
    """The per-bin heights must integrate to the model's mass in that range,
    so the step is a faithful redrawing of the density rather than a rescaling."""
    rng = np.random.default_rng(1)
    x = np.concatenate([rng.normal(-1.0, 0.3, 6000), rng.normal(1.0, 0.5, 4000)])
    model = GaussianMixture(n_components=2, random_state=0).fit(x.reshape(-1, 1))

    edges = np.histogram_bin_edges(x, bins=30)
    widths = np.diff(edges)
    bin_means = np.array([
        float(np.trapezoid(
            np.exp(model.score_samples(np.linspace(edges[i], edges[i + 1], 128).reshape(-1, 1))),
            np.linspace(edges[i], edges[i + 1], 128),
        ) / widths[i])
        for i in range(edges.size - 1)
    ])
    grid = np.linspace(edges[0], edges[-1], 4000)
    exact = float(np.trapezoid(np.exp(model.score_samples(grid.reshape(-1, 1))), grid))
    assert float(np.sum(bin_means * widths)) == pytest.approx(exact, rel=1e-3)

    fig, _ = plot_gmm_fit(model=model, x=x, bins=30, density_as_bin_means=True)
    plt.close(fig)


@pytest.mark.parametrize("fitter", [fit_log_gmm, fit_log_t_mixture])
def test_x_is_log_matches_the_seconds_route(fitter):
    """Passing log-space values with x_is_log=True is equivalent to passing the
    exponentiated seconds, because every fitter's first act is to take the log.
    The bootstrap relies on that equivalence to skip an exp/log round trip."""
    rng = np.random.default_rng(0)
    log_x = np.concatenate([rng.normal(-2.7, 0.25, 1500), rng.normal(-1.0, 0.5, 500)])

    via_seconds, _ = fitter(np.exp(log_x), 2, seed=0, n_init=2)
    via_log, _ = fitter(log_x, 2, seed=0, n_init=2, x_is_log=True)

    assert np.allclose(np.sort(np.ravel(via_seconds.means_)),
                       np.sort(np.ravel(via_log.means_)))


@pytest.mark.parametrize("fitter", [fit_log_gmm, fit_log_t_mixture])
def test_x_is_log_survives_a_draw_that_overflows_exp(fitter):
    """A heavy-tailed component can draw log_x > 709, whose exp is inf; routing
    through seconds then dies inside the fitter with "Input X contains
    infinity". The log-space route has nothing to overflow. This is the failure
    that killed the bootstrap LRT on a 172,331-interval pool."""
    rng = np.random.default_rng(1)
    log_x = np.concatenate([rng.normal(-2.7, 0.25, 800), [750.0]])
    # The suite escalates warnings to errors; exp() overflowing to inf is the
    # premise of this test, not the failure being asserted.
    with np.errstate(over="ignore"):
        seconds = np.exp(log_x)
    assert np.isinf(seconds).any()

    with pytest.raises(ValueError, match="infinity|too large"):
        fitter(seconds, 2, seed=0, n_init=1)

    model, _ = fitter(log_x, 2, seed=0, n_init=1, x_is_log=True)
    assert np.isfinite(np.ravel(model.means_)).all()


# stable_subsample — the value-keyed draw used by bootstrap_lrt


def _multiset_overlap(left, right):
    """How many points two draws share, counting repeats (interval values tie often)."""
    left_values, left_counts = np.unique(left, return_counts=True)
    right_values, right_counts = np.unique(right, return_counts=True)
    positions = np.searchsorted(right_values, left_values)
    positions[positions == right_values.size] = 0
    matched = right_values[positions] == left_values
    return int(np.minimum(left_counts[matched], right_counts[positions[matched]]).sum())


def test_stable_subsample_is_deterministic_and_seed_dependent():
    """The same pool and seed give the same draw every time, and a different seed gives a
    different one, so a run is reproducible without being locked to one arbitrary sample."""
    pool = np.random.default_rng(1).lognormal(-2.7, 1.0, 5000)
    assert np.array_equal(stable_subsample(pool, 1000, 0), stable_subsample(pool, 1000, 0))
    assert not np.array_equal(stable_subsample(pool, 1000, 0), stable_subsample(pool, 1000, 1))


def test_stable_subsample_returns_everything_when_pool_is_small():
    """A pool no larger than the requested size is returned whole, matching bootstrap_lrt's
    previous behaviour of skipping the draw entirely."""
    pool = np.random.default_rng(2).lognormal(-2.7, 1.0, 40)
    drawn = stable_subsample(pool, 1000, 0)
    assert drawn.size == 40
    assert _multiset_overlap(drawn, pool) == 40


def test_stable_subsample_survives_deletions_that_reroll_rng_choice():
    """The point of the value-keyed draw: deleting a small share of the pool removes about that
    share of the subsample, where rng.choice — which picks POSITIONS — re-rolls most of it because
    every element after the deletion shifts. This discontinuity is what let a 1.4% change in the
    interval pool flip the selected component count."""
    pool = np.round(np.random.default_rng(3).lognormal(-2.7, 1.0, 20000), 9)
    keep = np.ones(pool.size, dtype=bool)
    keep[np.random.default_rng(4).choice(pool.size, size=200, replace=False)] = False   # 1%

    keyed_before = stable_subsample(pool, 4000, 0)
    keyed_after = stable_subsample(pool[keep], 4000, 0)
    positional_before = np.random.default_rng(0).choice(pool, size=4000, replace=False)
    positional_after = np.random.default_rng(0).choice(pool[keep], size=4000, replace=False)

    assert _multiset_overlap(keyed_before, keyed_after) > 3800          # > 95% retained
    assert _multiset_overlap(positional_before, positional_after) < 2000  # < 50% retained


def test_stable_subsample_is_representative_of_the_pool():
    """The draw must stay a uniform subsample, not a stratified or biased one, or the bootstrap
    null it is compared against would no longer apply. A two-sample KS test against the full pool
    does not reject at any of several seeds."""
    pool = np.random.default_rng(5).lognormal(-2.7, 1.0, 40000)
    for seed in range(5):
        _, p_value = stats.ks_2samp(stable_subsample(pool, 5000, seed), pool)
        assert p_value > 0.01


def test_t_mixture_modes_recovers_two_separated_modes():
    """Well-separated components each own a mode, and the returned densities are sorted
    descending, matching :func:`gmm_modes`' contract."""
    model = TMixture(
        weights=np.array([0.6, 0.4]),
        means=np.array([-2.8, 0.5]),
        covariances=np.array([0.04, 0.04]),
        nus=np.array([8.0, 8.0]),
    )
    modes, densities = t_mixture_modes(model)

    assert modes.size == 2
    assert np.allclose(np.sort(modes), [-2.8, 0.5], atol=1e-3)
    assert densities[0] >= densities[1]


def test_t_mixture_modes_are_fewer_than_components_when_overlapping():
    """The reason this function exists: overlapping components tile one peak rather than marking
    several. Three components spaced well inside their own scale produce a single local maximum,
    so reporting component locations as modes would invent two peaks the density does not have."""
    model = TMixture(
        weights=np.array([0.5, 0.3, 0.2]),
        means=np.array([-2.80, -2.60, -2.40]),
        covariances=np.array([0.25, 0.25, 0.25]),
        nus=np.array([6.0, 6.0, 6.0]),
    )
    modes, _ = t_mixture_modes(model)

    assert modes.size == 1
    assert modes.size < model.means_.shape[0]


def test_t_mixture_modes_match_a_brute_force_grid():
    """The Newton polish must not move a peak off the density's actual maximum, so every mode is
    checked against a dense independent grid evaluated straight from the mixture."""
    model = TMixture(
        weights=np.array([0.70, 0.18, 0.12]),
        means=np.array([-2.77, -1.75, 2.30]),
        covariances=np.array([0.09, 0.05, 0.40]),
        nus=np.array([5.0, 9.0, 12.0]),
    )
    modes, _ = t_mixture_modes(model)

    grid = np.linspace(-8.0, 8.0, 400001)
    log_p = model.score_samples(grid)
    peaks = grid[np.flatnonzero((log_p[1:-1] > log_p[:-2]) & (log_p[1:-1] > log_p[2:])) + 1]

    assert modes.size == peaks.size
    assert np.allclose(np.sort(modes), np.sort(peaks), atol=1e-3)


def test_fit_tied_scale_t_mixture_ties_peak_scales_only():
    """Peak components must come back sharing one width while background components keep their
    own, which is the entire point of the constraint."""
    rng = np.random.default_rng(0)
    pool = np.concatenate([
        rng.normal(-2.77, 0.24, 6000),
        rng.normal(-1.72, 0.24, 800),
        rng.normal(-0.5, 1.2, 1500),
    ])
    model, _, shared = fit_tied_scale_t_mixture(pool, n_peak=2, n_background=1, n_init=2)
    scales = np.sqrt(np.asarray(model.covariances_, dtype=float).ravel())

    assert scales[0] == pytest.approx(scales[1], rel=1e-9)
    assert shared == pytest.approx(scales[0], rel=1e-9)
    assert scales[2] != pytest.approx(scales[0], rel=1e-3)


def test_fit_tied_scale_t_mixture_finds_a_far_secondary_peak_from_one_init():
    """The density-seeded start exists so a small, distant second peak is found without random
    luck. Seeding by quantile spread put both peaks on the dominant mode instead, 166 nats short
    on the real pool, so a single initialisation must recover both generative locations."""
    rng = np.random.default_rng(1)
    pool = np.concatenate([
        rng.normal(-2.77, 0.24, 8000),   # ~63 ms
        rng.normal(-1.72, 0.24, 700),    # ~179 ms, far out and lightly weighted
        rng.normal(-0.2, 1.3, 2000),
    ])
    model, _, _ = fit_tied_scale_t_mixture(pool, n_peak=2, n_background=1, n_init=1)
    peaks = np.sort(np.asarray(model.means_, dtype=float).ravel()[:2])

    assert peaks[0] == pytest.approx(-2.77, abs=0.15)
    assert peaks[1] == pytest.approx(-1.72, abs=0.25)


def test_fit_tied_nested_by_splitting_never_scores_below_its_base():
    """A peak split into two halves at the parent's location describes the same density, so the
    alternative starts at the null's likelihood and can only rise. Without this the peak-count
    null was contaminated by 17.1% negative replicates."""
    rng = np.random.default_rng(2)
    pool = np.concatenate([rng.normal(-2.8, 0.25, 4000), rng.normal(-0.4, 1.1, 1200)])
    base, base_ll, _ = fit_tied_scale_t_mixture(pool, n_peak=2, n_background=1, n_init=2)
    _, split_ll, _ = fit_tied_nested_by_splitting(pool, base, n_peak=2, n_background=1, n_init=1)

    assert split_ll >= base_ll - 1e-6


def test_tied_scale_n_parameters_charges_one_scale_for_all_peaks():
    """A tied peak costs two parameters where a free component costs three; BIC comparisons
    against an unconstrained mixture are only honest if that is counted."""
    # 3 peaks + 2 background: 5 weights - 1, 5 means, 5 nus, and 2 + 1 = 3 scales.
    assert tied_scale_n_parameters(3, 2) == 4 + 5 + 5 + 3
    assert tied_scale_n_parameters(1, 1) == 1 + 2 + 2 + 2


def test_fit_tied_scale_t_mixture_rejects_a_missing_role():
    """Both roles are required: with no background the tail has nowhere to go, and with no peak
    the constraint is meaningless."""
    pool = np.random.default_rng(3).normal(-2.0, 0.5, 500)
    with pytest.raises(ValueError, match="at least one peak"):
        fit_tied_scale_t_mixture(pool, n_peak=2, n_background=0, n_init=1)
    with pytest.raises(ValueError, match="at least one peak"):
        fit_tied_scale_t_mixture(pool, n_peak=0, n_background=2, n_init=1)


def test_stable_subsample_return_indices_aligns_with_values():
    """The indices exist so per-observation labels can follow a subsample into a test; they are
    only useful if ``values[indices]`` is exactly the subsample, in order."""
    pool = np.random.default_rng(6).lognormal(-2.7, 1.0, 5000)
    subsample, indices = stable_subsample(pool, 800, 0, return_indices=True)

    assert indices.shape == (800,)
    assert np.array_equal(pool[indices], subsample)
    assert np.array_equal(subsample, stable_subsample(pool, 800, 0))


def test_stable_subsample_return_indices_on_small_pool():
    """A pool no larger than the request is returned whole, and the indices must then be the
    identity so labels still line up."""
    pool = np.random.default_rng(7).lognormal(-2.7, 1.0, 300)
    subsample, indices = stable_subsample(pool, 800, 0, return_indices=True)

    assert subsample.size == 300
    assert np.array_equal(indices, np.arange(300))


def test_session_design_effect_is_one_for_independent_data():
    """With no within-session correlation, resampling sessions and resampling observations give the
    same variance, so the design effect must sit at one. This is the baseline the correction has to
    leave alone."""
    rng = np.random.default_rng(8)
    delta = rng.normal(0.0, 1.0, 12000)
    session_index = np.repeat(np.arange(120), 100)
    design_effect, _, _ = _session_design_effect(delta, session_index, n_bootstrap=1500, seed=0)

    assert design_effect == pytest.approx(1.0, abs=0.2)


def test_session_design_effect_is_one_for_independent_data_with_a_mean_and_unequal_sessions():
    """
    Independent observations with a non-zero mean, spread over sessions of unequal size, carry
    no clustering, so the design effect must still sit near one. A session resample holds a
    random number of observations; resampling uncentred session totals makes their spread grow
    with the session sizes times the mean (a design effect of ~9 here), so the totals have to
    be centred, ``sum_j - n_j * mean(delta)``, before they are resampled.
    """
    rng = np.random.default_rng(1)
    sizes = rng.geometric(1 / 85, size=118)
    session_index = np.repeat(np.arange(sizes.size), sizes)
    estimates = []
    for rep in range(5):
        delta = rng.normal(0.005, 0.02, size=session_index.size)
        design_effect, _, _ = _session_design_effect(delta, session_index, n_bootstrap=1500, seed=rep)
        estimates.append(design_effect)

    assert float(np.mean(estimates)) == pytest.approx(1.0, abs=0.25)


def test_session_design_effect_grows_with_session_level_shifts():
    """When every observation in a session shares a common offset, the statistic's variance under
    session resampling exceeds the iid one, and the design effect must say so. A shift with
    variance equal to the noise variance and 100 observations per session gives an intra-class
    correlation of 0.5 and a design effect near 1 + 99 * 0.5."""
    rng = np.random.default_rng(9)
    n_sessions, per_session = 120, 100
    offsets = rng.normal(0.0, 1.0, n_sessions)
    delta = np.repeat(offsets, per_session) + rng.normal(0.0, 1.0, n_sessions * per_session)
    session_index = np.repeat(np.arange(n_sessions), per_session)
    design_effect, clustered_se, independent_se = _session_design_effect(
        delta, session_index, n_bootstrap=1500, seed=0)

    assert clustered_se > independent_se
    assert 25.0 < design_effect < 80.0


def test_tied_peak_bootstrap_lrt_floors_design_effect_at_one():
    """With a handful of sessions the session bootstrap can return a design effect below one,
    which would inflate the statistic the correction exists to shrink. The reported effect must
    never fall below one, and the corrected statistic must never exceed the raw one."""
    rng = np.random.default_rng(10)
    pool = np.exp(np.concatenate([rng.normal(-2.77, 0.24, 1600), rng.normal(-1.7, 0.24, 200),
                                  rng.normal(-0.5, 1.2, 400)]))
    labels = np.repeat(np.arange(4), pool.size // 4 + 1)[:pool.size]
    result = tied_peak_bootstrap_lrt(pool, 1, 1, B=4, n_subsample=2000, n_init_obs=1,
                                     n_init_boot=1, seed=0, session_labels=labels,
                                     n_design_bootstrap=100)

    assert result["design_effect"] >= 1.0
    assert result["lr_corrected"] <= result["lr_obs"] + 1e-9


def test_bootstrap_lrt_session_correction_matches_its_own_statistic():
    """
    With session labels, the unconstrained test corrects its own statistic: the design effect is
    floored at one, the corrected statistic is exactly lr_obs / design_effect, the corrected
    p-value scores it against the same null, and the uncorrected fields are identical to a run
    without labels (up to floating-point reduction order). Misaligned labels raise instead of
    being silently truncated.
    """
    rng = np.random.default_rng(3)
    pool = np.exp(np.concatenate([rng.normal(-2.7, 0.25, 1500), rng.normal(0.0, 1.2, 700)]))
    labels = np.repeat(np.arange(20), pool.size // 20 + 1)[:pool.size]
    kwargs = dict(K_null=2, K_alt=3, B=4, n_subsample=1500, model_class="t", n_init_obs=1,
                  n_init_boot=1, seed=0)
    plain = bootstrap_lrt(pool, **kwargs)
    corrected = bootstrap_lrt(pool, **kwargs, session_labels=labels, n_design_bootstrap=100)

    assert "lr_corrected" not in plain
    # identical up to floating-point reduction order in the EM fits
    for key in ("lr_obs", "p_value", "null_mean", "null_p95", "null_max"):
        assert corrected[key] == pytest.approx(plain[key], rel=1e-9)
    np.testing.assert_allclose(corrected["lr_null"], plain["lr_null"], rtol=1e-9)
    assert corrected["design_effect"] >= 1.0
    assert corrected["lr_corrected"] == pytest.approx(corrected["lr_obs"] / corrected["design_effect"])
    assert corrected["p_value_corrected"] == pytest.approx(
        float((1 + np.sum(corrected["lr_null"] >= corrected["lr_corrected"])) / (1 + corrected["lr_null"].size)))
    assert corrected["p_value_corrected"] >= corrected["p_value"]

    with pytest.raises(ValueError, match="session_labels has"):
        bootstrap_lrt(pool, **kwargs, session_labels=labels[:-1], n_design_bootstrap=10)


def _t_model_two():
    """
    Builds a fixed, asymmetric two-component Student-t mixture for the closed-form tests.

    Parameters
    ----------

    Returns
    -------
    model (TMixture)
        Weights (0.7, 0.3), locations (-2.5, 0.4), squared scales (0.09, 0.36) and degrees of
        freedom (4, 9). The components differ in every parameter so a swapped or dropped index
        changes every downstream number.
    """
    return TMixture(weights=np.array([0.7, 0.3]), means=np.array([-2.5, 0.4]),
                    covariances=np.array([0.09, 0.36]), nus=np.array([4.0, 9.0]))


def _t_mixture_loglik(log_x, w, mu, sigma2, nu):
    """
    Total log-likelihood of a 1D Student-t mixture computed independently with scipy.

    Parameters
    ----------
    log_x (np.ndarray)
        A (N,) shape ndarray of evaluation points.
    w, mu, sigma2, nu (np.ndarray)
        A (K,) shape ndarray each of weights, locations, squared scales and degrees of freedom.

    Returns
    -------
    log_likelihood (float)
        ``sum_i log sum_k w_k t(x_i; nu_k, mu_k, sqrt(sigma2_k))``.
    """
    per_component = np.array([np.log(w[k]) + stats.t.logpdf(log_x, df=nu[k], loc=mu[k], scale=np.sqrt(sigma2[k]))
                              for k in range(len(w))])
    return float(np.sum(np.logaddexp.reduce(per_component, axis=0)))


def _t_kmeans_start(log_x, n_components, seed, reg_covar):
    """
    Rebuilds, by hand, the KMeans initialisation documented for :func:`fit_log_t_mixture`.

    Parameters
    ----------
    log_x (np.ndarray)
        A (N,) shape ndarray of log-space observations.
    n_components (int)
        Number of components.
    seed (int)
        KMeans random state.
    reg_covar (float)
        Variance floor.

    Returns
    -------
    start (tuple)
        ``(w, mu, sigma2, nu)``: KMeans centres, per-cluster variances floored at
        ``reg_covar``, ``nu = 10`` and cluster fractions floored at 1e-3 and renormalised.
    """
    km = KMeans(n_clusters=n_components, random_state=seed, n_init=10).fit(log_x.reshape(-1, 1))
    labels = km.labels_
    mu = km.cluster_centers_.flatten()
    sigma2 = np.array([max(np.var(log_x[labels == k]), reg_covar) for k in range(n_components)])
    w = np.array([np.mean(labels == k) for k in range(n_components)])
    w = np.where(w < 1e-3, 1e-3, w)
    return w / w.sum(), mu, sigma2, np.full(n_components, 10.0)


def _t_random_start(log_x, n_components, seed, restart, reg_covar):
    """
    Rebuilds, by hand, the seeded quantile initialisation used by every restart of
    :func:`fit_log_t_mixture` after the first.

    Parameters
    ----------
    log_x (np.ndarray)
        A (N,) shape ndarray of log-space observations.
    n_components (int)
        Number of components.
    seed (int)
        Base seed.
    restart (int)
        Restart index (``>= 1``); the generator is seeded with ``(seed, restart)``.
    reg_covar (float)
        Variance floor.

    Returns
    -------
    start (tuple)
        ``(w, mu, sigma2, nu)``: sorted uniform(0.02, 0.98) quantiles as centres, nearest-centre
        labels, per-label variances (the pooled variance for an empty label) floored at
        ``reg_covar``, ``nu = 10`` and label fractions floored at 1e-3 and renormalised.
    """
    rng = np.random.default_rng((seed, restart))
    qs = np.sort(rng.uniform(0.02, 0.98, size=n_components))
    mu = np.quantile(log_x, qs)
    labels = np.argmin(np.abs(log_x[:, None] - mu[None, :]), axis=1)
    sigma2 = np.array([max(np.var(log_x[labels == k]) if np.any(labels == k) else np.var(log_x), reg_covar)
                       for k in range(n_components)])
    w = np.array([np.mean(labels == k) for k in range(n_components)])
    w = np.where(w < 1e-3, 1e-3, w)
    return w / w.sum(), mu, sigma2, np.full(n_components, 10.0)


def _t_assert_model_equals(model, w, mu, sigma2, nu):
    """
    Asserts that a :class:`TMixture` holds exactly the given parameters.

    Parameters
    ----------
    model (TMixture)
        The model under test.
    w, mu, sigma2, nu (np.ndarray)
        Expected weights, locations, squared scales and degrees of freedom.

    Returns
    -------
    None
    """
    np.testing.assert_allclose(model.weights_, w, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(model.means_.ravel(), mu, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(model.covariances_.ravel(), sigma2, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(model.nus_, nu, rtol=1e-12, atol=0.0)


def _t_tied_density_start(log_x, n_peak, n_background, reg_covar):
    """
    Rebuilds, by hand, the deterministic density-seeded first restart of
    :func:`fit_tied_scale_t_mixture`.

    Parameters
    ----------
    log_x (np.ndarray)
        A (N,) shape ndarray of log-space observations.
    n_peak (int)
        Number of tied peak components.
    n_background (int)
        Number of free background components.
    reg_covar (float)
        Variance floor.

    Returns
    -------
    start (tuple)
        ``(w, mu, sigma2, nu, n_maxima)``: peaks at the highest maxima of the 512-bin histogram
        smoothed by a 17-tap Gaussian of 3-bin width (restored to ascending order, any shortfall
        filled from the 0.15-0.9 quantile ladder), background at the 0.4-0.97 quantile ladder,
        peak variance ``max(0.05 * var, reg_covar)``, background variance ``var``, ``nu`` 10 /
        5, uniform weights, and the number of strict maxima found.
    """
    seeds_log = np.quantile(log_x, np.linspace(0.15, 0.9, n_peak))
    counts, edges = np.histogram(log_x, bins=512)
    centers = 0.5 * (edges[:-1] + edges[1:])
    kernel = np.exp(-0.5 * (np.arange(-8, 9) / 3.0) ** 2)
    smoothed = np.convolve(counts.astype(float), kernel / kernel.sum(), mode="same")
    maxima = np.flatnonzero((smoothed[1:-1] > smoothed[:-2]) & (smoothed[1:-1] > smoothed[2:])) + 1
    chosen = maxima[np.argsort(smoothed[maxima])[::-1]][:n_peak]
    found = np.sort(centers[chosen])
    seeds_log[:found.size] = found
    mu = np.concatenate([seeds_log, np.quantile(log_x, np.linspace(0.4, 0.97, n_background))])
    total_variance = float(np.var(log_x))
    sigma2 = np.concatenate([np.full(n_peak, max(total_variance * 0.05, reg_covar)),
                             np.full(n_background, total_variance)])
    nu = np.concatenate([np.full(n_peak, 10.0), np.full(n_background, 5.0)])
    n_components = n_peak + n_background
    return np.full(n_components, 1.0 / n_components), mu, sigma2, nu, maxima.size


def _t_tied_random_start(log_x, n_peak, n_background, seed, restart, reg_covar):
    """
    Rebuilds, by hand, a seeded random restart (``restart >= 1``) of
    :func:`fit_tied_scale_t_mixture`.

    Parameters
    ----------
    log_x (np.ndarray)
        A (N,) shape ndarray of log-space observations.
    n_peak (int)
        Number of tied peak components.
    n_background (int)
        Number of free background components.
    seed (int)
        Base seed.
    restart (int)
        Restart index; the generator is seeded with ``(seed, restart)``.
    reg_covar (float)
        Variance floor.

    Returns
    -------
    start (tuple)
        ``(w, mu, sigma2, nu)``: peaks at sorted uniform(0.05, 0.95) quantiles, background at
        sorted uniform(0.35, 0.98) quantiles, peak variance ``max(0.05 * var, reg_covar)``,
        background variance ``var``, ``nu`` 10 / 5 and uniform weights.
    """
    rng = np.random.default_rng((seed, restart))
    peak_q = np.sort(rng.uniform(0.05, 0.95, size=n_peak))
    background_q = np.sort(rng.uniform(0.35, 0.98, size=n_background))
    mu = np.concatenate([np.quantile(log_x, peak_q), np.quantile(log_x, background_q)])
    total_variance = float(np.var(log_x))
    sigma2 = np.concatenate([np.full(n_peak, max(total_variance * 0.05, reg_covar)),
                             np.full(n_background, total_variance)])
    nu = np.concatenate([np.full(n_peak, 10.0), np.full(n_background, 5.0)])
    n_components = n_peak + n_background
    return np.full(n_components, 1.0 / n_components), mu, sigma2, nu


def _t_draw_t_mixture(seed, n, w, mu, sd, nu):
    """
    Draws a synthetic log-space sample from a Student-t mixture.

    Parameters
    ----------
    seed (int)
        Generator seed.
    n (int)
        Sample size.
    w, mu, sd, nu (sequence)
        Per-component weights, locations, scales and degrees of freedom.

    Returns
    -------
    log_x (np.ndarray)
        A (n,) shape ndarray of draws.
    """
    rng = np.random.default_rng(seed)
    comp = rng.choice(len(w), size=n, p=np.asarray(w) / np.sum(w))
    return np.asarray(mu)[comp] + np.asarray(sd)[comp] * rng.standard_t(np.asarray(nu)[comp])


def _t_heavy_tail_pool(seed):
    """
    The heavy-tailed pool on which a seeded quantile restart beats the KMeans start.

    Parameters
    ----------
    seed (int)
        Generator seed.

    Returns
    -------
    log_x (np.ndarray)
        A (440,) shape ndarray: 400 Student-t(2) draws scaled by 0.3 around -2 plus 40 narrow
        normal draws at 0.
    """
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.standard_t(2, 400) * 0.3 - 2.0, rng.normal(0.0, 0.05, 40)])


def _t_mixture_grad(x, model):
    """
    Analytic derivative of the mixture log-density, ``sum_k r_k(x) g_k(x)``.

    Parameters
    ----------
    x (float)
        Evaluation point.
    model (TMixture)
        The mixture.

    Returns
    -------
    grad (float)
        ``d/dx log p(x)`` with ``g_k = (nu_k + 1)(mu_k - x) / (nu_k sigma2_k + (x - mu_k)^2)``.
    """
    mu = model.means_.ravel()
    s2 = model.covariances_.ravel()
    nu = model.nus_
    logs = np.log(model.weights_) + stats.t.logpdf(x, df=nu, loc=mu, scale=np.sqrt(s2))
    r = np.exp(logs - np.logaddexp.reduce(logs))
    g = (nu + 1.0) * (mu - x) / (nu * s2 + (x - mu) ** 2)
    return float(np.sum(r * g))


def _t_numeric_derivatives(x, model, h=1e-4):
    """
    Central finite-difference first and second derivatives of the mixture log-density.

    Parameters
    ----------
    x (float)
        Evaluation point.
    model (TMixture)
        The mixture.
    h (float)
        Step; defaults to 1e-4.

    Returns
    -------
    derivatives (tuple)
        ``(d/dx log p, d^2/dx^2 log p)`` at ``x``.
    """
    lp = model.score_samples(np.array([x - h, x, x + h]))
    return (lp[2] - lp[0]) / (2.0 * h), (lp[2] - 2.0 * lp[1] + lp[0]) / h ** 2


class _TFakeScorer:
    """
    Minimal stand-in model whose ``score_samples`` returns a fixed per-observation array.
    """

    def __init__(self, values):
        """
        Stores the per-observation log-likelihoods to return.

        Parameters
        ----------
        values (np.ndarray)
            A (N,) shape ndarray returned by every ``score_samples`` call.

        Returns
        -------
        None
        """
        self.values = np.asarray(values, dtype=float)

    def score_samples(self, log_x):
        """
        Returns the stored array, checking that it lines up with the input.

        Parameters
        ----------
        log_x (np.ndarray)
            Evaluation points; only its length is used.

        Returns
        -------
        values (np.ndarray)
            The stored per-observation log-likelihoods.
        """
        assert np.asarray(log_x).size == self.values.size
        return self.values


def test_t_logpdf_1d_matches_scipy_student_t():
    """
    :func:`_t_logpdf_1d` is the log-density of a location-scale Student-t with squared scale
    ``sigma2``; it must agree with ``scipy.stats.t.logpdf(x, nu, mu, sqrt(sigma2))`` to machine
    precision for several degrees of freedom, locations and scales. This pins the
    ``(nu + 1) / 2`` gamma argument, the ``log(pi * nu * sigma2)`` normaliser and the
    ``(nu + 1) / 2`` tail exponent individually.
    """
    x = np.linspace(-6.0, 6.0, 41)
    for mu, sigma2, nu in ((0.0, 1.0, 3.0), (-1.3, 0.04, 2.5), (2.0, 4.0, 30.0)):
        np.testing.assert_allclose(_t_logpdf_1d(x, mu, sigma2, nu),
                                   stats.t.logpdf(x, df=nu, loc=mu, scale=np.sqrt(sigma2)),
                                   rtol=1e-12, atol=1e-12)


def _t_nu_score_offset(nu, nu_old):
    """
    The data-free part of the degrees-of-freedom score equation, evaluated in the same order
    as :func:`_t_update_nu` so the result is bit-identical.

    Parameters
    ----------
    nu (float)
        Candidate degrees of freedom.
    nu_old (float)
        Previous degrees of freedom.

    Returns
    -------
    offset (float)
        ``-digamma(nu / 2) + log(nu / 2) + 1 + psi_old``.
    """
    psi_old = digamma((nu_old + 1.0) / 2.0) - np.log((nu_old + 1.0) / 2.0)
    return -digamma(nu / 2.0) + np.log(nu / 2.0) + 1.0 + psi_old


def test_t_update_nu_score_exactly_zero_at_upper_end_returns_gaussian_default():
    """
    When the score is exactly zero at the upper bracket end ``nu = 200`` the update must return
    the near-Gaussian default ``50.0`` (the rule is ``f(nu_high) >= 0``), not run Brent's method,
    which would return the bracket end itself. A single observation with responsibility equal
    to the data-free offset and ``u = 1`` makes ``avg = -offset`` exactly, so ``f(200) == 0``.
    """
    offset = _t_nu_score_offset(200.0, 10.0)
    assert offset > 0.0
    out = _t_update_nu(np.array([offset]), np.array([1.0]), 10.0, 1.0)

    assert out == 50.0


def test_t_update_nu_root_just_above_200_still_returns_gaussian_default():
    """
    The bracket is ``[2.001, 200]``: a score root at ``nu = 200.5`` lies above it, so the score
    is still positive at 200 and the update returns ``50.0``. Widening the upper end even by
    one unit would put the root inside and return ~200.5 instead.
    """
    offset = _t_nu_score_offset(200.5, 10.0)
    out = _t_update_nu(np.array([offset]), np.array([1.0]), 10.0, 1.0)

    assert out == 50.0


def test_tmixture_constructor_casts_to_float_and_sklearn_shapes():
    """
    The constructor must store float arrays in sklearn's 1D shapes (``means_`` (K, 1),
    ``covariances_`` (K, 1, 1)) with ``covariance_type == 'full'`` even when handed integer
    lists, so in-place edits by downstream code keep fractional values instead of truncating.
    """
    model = TMixture(weights=[1, 1], means=[0, 2], covariances=[1, 4], nus=[5, 7])

    assert model.n_components == 2
    assert model.covariance_type == "full"
    assert model.means_.shape == (2, 1)
    assert model.covariances_.shape == (2, 1, 1)
    for attribute in (model.weights_, model.means_, model.covariances_, model.nus_):
        assert attribute.dtype == np.float64
    model.weights_[0] = 0.25
    model.means_[0, 0] = 0.5
    model.covariances_[0, 0, 0] = 0.75
    model.nus_[0] = 2.5
    assert (model.weights_[0], model.means_[0, 0], model.covariances_[0, 0, 0], model.nus_[0]) == \
        (0.25, 0.5, 0.75, 2.5)


def test_tmixture_scoring_methods_match_closed_forms():
    """
    ``score_samples`` is the per-point log of ``sum_k w_k t_k(x)`` (shape (N,)), ``score`` is its
    mean, ``predict_proba`` is the (N, K) posterior with rows summing to one, BIC is
    ``-2 LL + (4K - 1) log N`` and AIC is ``-2 LL + 2 (4K - 1)``, all computed here independently
    with scipy.
    """
    model = _t_model_two()
    x = np.linspace(-4.0, 2.0, 25)
    per_component = np.array([np.log(model.weights_[k]) + stats.t.logpdf(
        x, df=model.nus_[k], loc=model.means_[k, 0], scale=np.sqrt(model.covariances_[k, 0, 0]))
        for k in range(2)])
    expected_ll = np.logaddexp.reduce(per_component, axis=0)

    ll = model.score_samples(x.reshape(-1, 1))
    assert ll.shape == (25,)
    np.testing.assert_allclose(ll, expected_ll, rtol=1e-12)
    assert model.score(x) == pytest.approx(float(np.mean(expected_ll)), rel=1e-12)

    proba = model.predict_proba(x)
    assert proba.shape == (25, 2)
    np.testing.assert_allclose(proba, np.exp(per_component - expected_ll).T, rtol=1e-10)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-12)

    total = float(np.sum(expected_ll))
    assert model._n_params() == 7
    assert model.bic(x.reshape(-1, 1)) == pytest.approx(-2.0 * total + 7 * np.log(25), rel=1e-12)
    assert model.aic(x) == pytest.approx(-2.0 * total + 14.0, rel=1e-12)


def test_t_mixture_icl_is_bic_plus_twice_entropy_and_ignores_underflowed_posteriors():
    """
    ICL is ``BIC + 2 H(z)`` with ``H = -sum z log z``. One point sits between two overlapping
    components (non-trivial entropy) and the others sit on a component so remote from the
    second that its posterior underflows to exactly zero; such entries must contribute zero
    entropy (the ``1e-300`` clip), not ``0 * log 0 = nan``.
    """
    model = TMixture(weights=np.array([0.5, 0.5]), means=np.array([0.0, 100.0]),
                     covariances=np.array([1e-4, 1e-4]), nus=np.array([200.0, 200.0]))
    x = np.array([0.0, 0.01, -0.01])
    z = model.predict_proba(x)
    assert np.any(z == 0.0)
    expected = model.bic(x) + 2.0 * 0.0
    assert t_mixture_icl(model, x) == pytest.approx(expected, rel=1e-12)

    overlapping = TMixture(weights=np.array([0.6, 0.4]), means=np.array([0.0, 0.5]),
                           covariances=np.array([0.25, 0.25]), nus=np.array([5.0, 5.0]))
    y = np.array([-0.3, 0.1, 0.25, 0.7])
    zz = overlapping.predict_proba(y)
    entropy = float(-np.sum(zz * np.log(zz)))
    assert entropy > 0.5
    assert t_mixture_icl(overlapping, y) == pytest.approx(overlapping.bic(y) + 2.0 * entropy, rel=1e-12)


def test_report_and_summarize_t_mixture_apply_order_and_closed_forms():
    """
    :func:`report_t_mixture_stats` reorders log-means, log-scales (square roots of the
    variances), degrees of freedom and weights by ``order`` and evaluates the mixture density at
    each reordered log-mean; :func:`summarize_best_t_mixture` repackages those with
    ``medians_sec = exp(mu)``, ``means_sec = exp(mu + s^2 / 2)``, the component locations as
    modes, empty boundary arrays and ``tau = 0.5``.
    """
    model = _t_model_two()
    order = np.array([1, 0])
    logmeans, logscales, nus, weights, dens = report_t_mixture_stats(model, order)

    np.testing.assert_array_equal(logmeans, [0.4, -2.5])
    np.testing.assert_allclose(logscales, [0.6, 0.3], rtol=1e-12)
    np.testing.assert_array_equal(nus, [9.0, 4.0])
    np.testing.assert_array_equal(weights, [0.3, 0.7])
    np.testing.assert_allclose(dens, np.exp(model.score_samples(np.array([0.4, -2.5]))), rtol=1e-12)

    summary = summarize_best_t_mixture(model, order)
    np.testing.assert_array_equal(summary["logmeans"], [0.4, -2.5])
    np.testing.assert_allclose(summary["logsds"], [0.6, 0.3], rtol=1e-12)
    np.testing.assert_allclose(summary["medians_sec"], np.exp([0.4, -2.5]), rtol=1e-12)
    np.testing.assert_allclose(summary["means_sec"], np.exp([0.4 + 0.5 * 0.36, -2.5 + 0.5 * 0.09]),
                               rtol=1e-12)
    np.testing.assert_array_equal(summary["nus"], [9.0, 4.0])
    np.testing.assert_array_equal(summary["weights"], [0.3, 0.7])
    np.testing.assert_array_equal(summary["modes_log"], [0.4, -2.5])
    np.testing.assert_allclose(summary["modes_sec"], np.exp([0.4, -2.5]), rtol=1e-12)
    np.testing.assert_allclose(summary["mode_densities"], dens, rtol=1e-12)
    assert summary["boundaries_log"].shape == (0,)
    assert summary["boundaries_sec"].shape == (0,)
    assert summary["tau"] == 0.5


def test_t_mixture_cdf_is_the_weighted_sum_of_component_cdfs():
    """
    The analytic CDF is ``sum_k w_k T_{nu_k}((x - mu_k) / sigma_k)``; checked against scipy on
    an array (shape preserved) and on a scalar.
    """
    model = _t_model_two()
    x = np.array([[-3.0, -2.5], [0.0, 1.5]])
    expected = (0.7 * stats.t.cdf((x + 2.5) / 0.3, df=4.0)
                + 0.3 * stats.t.cdf((x - 0.4) / 0.6, df=9.0))
    out = t_mixture_cdf_logspace(x, model)

    assert out.shape == (2, 2)
    np.testing.assert_allclose(out, expected, rtol=1e-12)
    assert float(t_mixture_cdf_logspace(-1.0, model)) == pytest.approx(
        0.7 * stats.t.cdf(1.5 / 0.3, df=4.0) + 0.3 * stats.t.cdf(-1.4 / 0.6, df=9.0), rel=1e-12)


def test_t_mixture_quantile_inverts_the_cdf_including_far_tails():
    """
    The quantile function inverts the mixture CDF by Brent's method on a bracket of eight
    maximal scales past the extreme locations. Round trips must hold to 1e-10, including at
    1e-3 / 0.999 on a wide (scale 2), heavy-tailed (nu 5) component whose tail quantiles sit
    about six scales out, so a bracket narrower than documented cannot contain them.
    """
    model = _t_model_two()
    q = np.array([0.01, 0.2, 0.5, 0.71, 0.95])
    quantiles = t_mixture_quantile_logspace(q, model)
    np.testing.assert_allclose(t_mixture_cdf_logspace(quantiles, model), q, atol=1e-10)
    assert np.all(np.diff(quantiles) > 0)

    wide = TMixture(weights=np.array([0.5, 0.5]), means=np.array([0.0, 1.0]),
                    covariances=np.array([4.0, 4.0]), nus=np.array([5.0, 5.0]))
    q_tail = np.array([1e-3, 0.999])
    tails = t_mixture_quantile_logspace(q_tail, wide)
    np.testing.assert_allclose(t_mixture_cdf_logspace(tails, wide), q_tail, atol=1e-10)
    assert tails[0] < -10.0 and tails[1] > 11.0


def test_t_mixture_cv_neg_loglik_matches_a_manual_kfold(monkeypatch):
    """
    The cross-validated deviance is ``-2 * sum`` of held-out log-likelihoods over a shuffled,
    seeded KFold, each fold fitted by :func:`fit_log_t_mixture` on the training intervals (in
    seconds) with the caller's ``n_components``, ``seed``, ``n_init`` and ``reg_covar``. The fit
    is wrapped to record its arguments; the total is rebuilt from the recorded models.
    """
    pool = np.exp(_t_draw_t_mixture(4, 120, [0.6, 0.4], [-2.5, 0.0], [0.2, 0.3], [6.0, 6.0]))
    calls = []
    original = mixture_model_utils.fit_log_t_mixture

    def recording_fit(x, **kwargs):
        """Records the training data and keyword arguments, then fits for real."""
        result = original(x, **kwargs)
        calls.append((np.asarray(x).copy(), dict(kwargs), result[0]))
        return result

    monkeypatch.setattr(mixture_model_utils, "fit_log_t_mixture", recording_fit)
    out = t_mixture_cv_neg_loglik(pool, n_components=2, seed=7, n_folds=3, n_init=1, reg_covar=2e-3)

    log_x = np.log(pool)
    folds = list(KFold(n_splits=3, shuffle=True, random_state=7).split(log_x))
    assert len(calls) == 3
    expected = 0.0
    for (train_idx, test_idx), (x_train, kwargs, model) in zip(folds, calls):
        np.testing.assert_allclose(x_train, np.exp(log_x[train_idx]), rtol=1e-12)
        assert kwargs == {"n_components": 2, "seed": 7, "n_init": 1, "reg_covar": 2e-3}
        expected += float(np.sum(model.score_samples(log_x[test_idx])))
    assert out == pytest.approx(-2.0 * expected, rel=1e-12)


def test_t_mixture_cv_neg_loglik_size_guards():
    """
    Fewer samples than folds returns ``inf``; exactly as many samples as folds is allowed, and a
    training fold exactly as large as ``n_components`` is fitted rather than rejected (the guard
    is strictly ``size < n_components``), so the result is finite; one component more than
    the training fold holds returns ``inf``.
    """
    pool = np.exp(np.array([-2.0, -1.7, -1.1, -0.4, 0.3]))

    assert t_mixture_cv_neg_loglik(pool[:4], n_components=1, n_folds=5) == float("inf")
    assert np.isfinite(t_mixture_cv_neg_loglik(pool, n_components=1, n_folds=5, n_init=1))
    assert np.isfinite(t_mixture_cv_neg_loglik(pool, n_components=4, n_folds=5, n_init=1))
    assert t_mixture_cv_neg_loglik(pool, n_components=5, n_folds=5, n_init=1) == float("inf")


def test_fit_log_t_mixture_recovers_well_separated_components():
    """
    EM on 1500 draws from two well-separated Student-t components (weights 0.7 / 0.3, locations
    -2.5 / 0.5, scales 0.2 / 0.3, nu 6) must recover locations within 0.03, scales within 15%
    and weights within 0.03; ``order`` sorts components by location. Accepts seconds input.
    """
    log_x = _t_draw_t_mixture(11, 1500, [0.7, 0.3], [-2.5, 0.5], [0.2, 0.3], [6.0, 6.0])
    model, order = fit_log_t_mixture(np.exp(log_x), 2, seed=0, n_init=1)

    np.testing.assert_array_equal(order, np.argsort(model.means_.ravel()))
    mu = model.means_.ravel()[order]
    sd = np.sqrt(model.covariances_.ravel())[order]
    w = model.weights_[order]
    np.testing.assert_allclose(mu, [-2.5, 0.5], atol=0.03)
    np.testing.assert_allclose(sd, [0.2, 0.3], rtol=0.15)
    np.testing.assert_allclose(w, [0.7, 0.3], atol=0.03)
    assert w.sum() == pytest.approx(1.0, abs=1e-12)


def test_fit_log_t_mixture_tolerance_stops_after_the_second_sweep():
    """
    Convergence is ``|LL_t - LL_{t-1}| < tol``. With a tolerance above the first improvement but
    far below ``2 |LL|`` the loop must stop after the second sweep, so the fit equals a
    ``max_iter=2`` fit; it must not run on to ``max_iter``.
    """
    log_x = _t_draw_t_mixture(12, 400, [0.7, 0.3], [-2.5, 0.5], [0.2, 0.3], [6.0, 6.0])
    ll0 = fit_log_t_mixture(log_x, 2, n_init=1, max_iter=0, x_is_log=True)[0].score(log_x) * 400
    ll1 = fit_log_t_mixture(log_x, 2, n_init=1, max_iter=1, x_is_log=True)[0].score(log_x) * 400
    assert abs(ll1 - ll0) < 50.0 < abs(ll1 + ll0)

    stopped, _ = fit_log_t_mixture(log_x, 2, n_init=1, max_iter=60, tol=50.0, x_is_log=True)
    two, _ = fit_log_t_mixture(log_x, 2, n_init=1, max_iter=2, tol=0.0, x_is_log=True)
    many, _ = fit_log_t_mixture(log_x, 2, n_init=1, max_iter=60, tol=0.0, x_is_log=True)
    _t_assert_model_equals(stopped, two.weights_, two.means_.ravel(), two.covariances_.ravel(), two.nus_)
    assert not np.allclose(stopped.nus_, many.nus_, rtol=1e-6)


def test_fit_log_t_mixture_kmeans_start_is_returned_at_zero_iterations():
    """
    With ``max_iter=0`` and one restart the returned model IS the KMeans start: KMeans centres,
    per-cluster variances floored at ``reg_covar`` (a one-point cluster has zero variance),
    ``nu = 10``, and cluster fractions floored at 1e-3 and renormalised (the outlier's 1/2001
    fraction is raised to 1e-3).
    """
    rng = np.random.default_rng(13)
    log_x = np.concatenate([rng.normal(-2.0, 0.3, 2000), [15.0]])
    model, _ = fit_log_t_mixture(log_x, 2, seed=3, n_init=1, max_iter=0, x_is_log=True, reg_covar=1e-3)
    w, mu, sigma2, nu = _t_kmeans_start(log_x, 2, 3, 1e-3)

    assert np.min(w) == pytest.approx(1e-3 / (1.0 + 1e-3 - 1.0 / 2001), rel=1e-6)
    assert np.max(sigma2) > 1e-3 and np.min(sigma2) == 1e-3
    _t_assert_model_equals(model, w, mu, sigma2, nu)


def test_fit_log_t_mixture_quantile_restart_wins_when_it_scores_higher():
    """
    Restarts after the first draw sorted uniform(0.02, 0.98) quantiles from a ``(seed, ii)``
    generator. On this heavy-tailed pool that start scores above the KMeans start, so with
    ``max_iter=0`` and two restarts the returned model must be exactly the quantile start: the
    best restart is kept by its own log-likelihood.
    """
    log_x = _t_heavy_tail_pool(0)
    kmeans_start = _t_kmeans_start(log_x, 2, 0, 1e-4)
    random_start = _t_random_start(log_x, 2, 0, 1, 1e-4)
    assert _t_mixture_loglik(log_x, *random_start) > _t_mixture_loglik(log_x, *kmeans_start)

    model, _ = fit_log_t_mixture(log_x, 2, seed=0, n_init=2, max_iter=0, x_is_log=True)

    _t_assert_model_equals(model, *random_start)


def test_fit_log_t_mixture_bound_rejects_restarts_and_keeps_searching():
    """
    A restart with a component below ``min_component_log_mean`` is discarded and the search
    continues. Here the KMeans start owns the small low cluster (rejected) while the quantile
    restart places both centres on the dominant tied value 0 (the second label is empty, so its
    variance is the pooled one and its weight is floored at 1e-3); that restart is returned.
    The bound is set exactly at those centres: a mean equal to the bound is not below it.
    """
    rng = np.random.default_rng(14)
    log_x = np.concatenate([np.zeros(985), rng.normal(-3.0, 0.2, 15)])
    model, order = fit_log_t_mixture(log_x, 2, seed=0, n_init=2, max_iter=0, x_is_log=True,
                                     min_component_log_mean=0.0)
    start = _t_random_start(log_x, 2, 0, 1, 1e-4)

    np.testing.assert_array_equal(start[1], [0.0, 0.0])
    _t_assert_model_equals(model, *start)
    np.testing.assert_array_equal(order, np.argsort(model.means_.ravel()))


def test_fit_log_t_mixture_raises_when_every_restart_is_below_the_bound():
    """
    When every restart places a component below the bound the function raises a
    ``RuntimeError`` naming the bound rather than returning a model or relaxing it.
    """
    log_x = _t_draw_t_mixture(15, 200, [0.5, 0.5], [-2.0, 0.0], [0.2, 0.2], [6.0, 6.0])
    with pytest.raises(RuntimeError, match="every one of the 2 restarts placed a component below"):
        fit_log_t_mixture(log_x, 2, n_init=2, max_iter=0, x_is_log=True, min_component_log_mean=5.0)


def test_fit_log_t_mixture_explicit_start_runs_once_and_only_as_an_extra_restart():
    """
    ``init_params`` adds exactly one restart after the ``n_init`` random ones. With ``n_init=0``
    and ``max_iter=0`` the returned model is that start (variances floored at ``reg_covar``),
    even when it scores badly; with ``n_init=1`` the bad start loses to the KMeans start, so the
    result equals a fit without it. With no restarts at all the fit raises.
    """
    log_x = _t_draw_t_mixture(16, 300, [0.6, 0.4], [-2.0, 0.0], [0.2, 0.3], [6.0, 6.0])
    bad = (np.array([0.5, 0.5]), np.array([3.0, 4.0]), np.array([1e-6, 0.01]), np.array([3.0, 4.0]))

    only, _ = fit_log_t_mixture(log_x, 2, n_init=0, max_iter=0, x_is_log=True, init_params=bad)
    _t_assert_model_equals(only, bad[0], bad[1], np.array([1e-4, 0.01]), bad[3])

    with_kmeans, _ = fit_log_t_mixture(log_x, 2, n_init=1, max_iter=0, x_is_log=True, init_params=bad)
    plain, _ = fit_log_t_mixture(log_x, 2, n_init=1, max_iter=0, x_is_log=True)
    _t_assert_model_equals(with_kmeans, plain.weights_, plain.means_.ravel(),
                           plain.covariances_.ravel(), plain.nus_)

    with pytest.raises(RuntimeError, match="did not produce a valid fit"):
        fit_log_t_mixture(log_x, 2, n_init=0, x_is_log=True)


def test_fit_tied_scale_explicit_start_is_tied_and_normalised():
    """
    With ``n_init=0`` the explicit start is the only restart. Its variances are floored at
    ``reg_covar`` and the peak variances replaced by their mean before the first E-step, and its
    weights renormalised; at ``max_iter=0`` that is the returned model, the returned
    log-likelihood is that model's, and the shared scale is the square root of the tied value.
    With ``n_init=1`` a bad explicit start loses to the density-seeded start.
    """
    log_x = _t_draw_t_mixture(17, 600, [0.5, 0.2, 0.3], [-2.5, -1.5, 0.0], [0.2, 0.2, 1.0],
                              [8.0, 8.0, 5.0])
    start = (np.array([2.0, 1.0, 1.0]), np.array([-2.4, -1.6, 0.1]),
             np.array([1e-6, 0.08, 1.0]), np.array([9.0, 7.0, 4.0]))
    model, ll, shared = fit_tied_scale_t_mixture(log_x, 2, 1, n_init=0, max_iter=0,
                                                 reg_covar=1e-3, init_params=start)
    tied = (1e-3 + 0.08) / 2.0

    _t_assert_model_equals(model, [0.5, 0.25, 0.25], start[1], [tied, tied, 1.0], start[3])
    assert ll == pytest.approx(float(np.sum(model.score_samples(log_x))), rel=1e-12)
    assert shared == pytest.approx(np.sqrt(tied), rel=1e-12)

    bad = (np.array([1.0, 1.0, 1.0]), np.array([4.0, 5.0, 6.0]),
           np.array([0.01, 0.01, 0.01]), np.array([5.0, 5.0, 5.0]))
    with_bad, bad_ll, _ = fit_tied_scale_t_mixture(log_x, 2, 1, n_init=1, max_iter=0, init_params=bad)
    plain, plain_ll, _ = fit_tied_scale_t_mixture(log_x, 2, 1, n_init=1, max_iter=0)
    assert bad_ll == plain_ll
    _t_assert_model_equals(with_bad, plain.weights_, plain.means_.ravel(),
                           plain.covariances_.ravel(), plain.nus_)

    with pytest.raises(RuntimeError, match="no restart produced a model"):
        fit_tied_scale_t_mixture(log_x, 2, 1, n_init=0)


def test_fit_tied_scale_density_seeded_start_is_returned_at_zero_iterations():
    """
    The first restart seeds peaks at the strict maxima of a 512-bin histogram smoothed by a
    17-tap Gaussian of 3-bin width. On a smooth unimodal core plus a remote pair of points there
    are exactly two maxima (a point in the last bin is never an interior maximum), so with three peaks the third falls back to the quantile ladder; the
    long empty stretch between them smooths to exact zeros, which are not maxima. At
    ``max_iter=0`` the returned model is that start exactly.
    """
    core = stats.norm.ppf(np.linspace(0.0005, 0.9995, 20000), loc=-2.0, scale=0.3)
    log_x = np.concatenate([core, [6.0, 6.0, 8.0]])
    w, mu, sigma2, nu, n_maxima = _t_tied_density_start(log_x, 3, 1, 1e-4)
    assert n_maxima == 2

    model, ll, shared = fit_tied_scale_t_mixture(log_x, 3, 1, n_init=1, max_iter=0)

    _t_assert_model_equals(model, w, mu, sigma2, nu)
    assert ll == pytest.approx(_t_mixture_loglik(log_x, w, mu, sigma2, nu), rel=1e-10)
    assert shared == pytest.approx(np.sqrt(sigma2[0]), rel=1e-12)


def test_fit_tied_scale_density_start_on_a_bimodal_pool():
    """
    On a noisy bimodal pool the density-seeded start must place the two peaks exactly at the two
    highest smoothed-histogram maxima (in ascending order), with the background on the
    0.4-0.97 quantile ladder.
    """
    rng = np.random.default_rng(18)
    log_x = np.concatenate([rng.normal(-2.7, 0.2, 3000), rng.normal(-1.6, 0.2, 800),
                            rng.normal(0.0, 1.2, 600)])
    w, mu, sigma2, nu, _ = _t_tied_density_start(log_x, 2, 2, 1e-4)
    model, _, _ = fit_tied_scale_t_mixture(log_x, 2, 2, n_init=1, max_iter=0)

    _t_assert_model_equals(model, w, mu, sigma2, nu)


def test_fit_tied_scale_random_restart_wins_when_it_scores_higher():
    """
    Restarts after the first draw peak quantiles from uniform(0.05, 0.95) and background
    quantiles from uniform(0.35, 0.98) with a ``(seed, restart)`` generator. On this pool that
    start beats the density-seeded one, so at ``max_iter=0`` with two restarts it is returned
    exactly, together with its own log-likelihood.
    """
    rng = np.random.default_rng(1)
    log_x = np.concatenate([rng.normal(-2.5, 0.2, 300), rng.normal(-1.0, 0.2, 150),
                            rng.normal(0.0, 1.0, 100)])
    first = _t_tied_density_start(log_x, 2, 1, 1e-4)[:4]
    second = _t_tied_random_start(log_x, 2, 1, 1, 1, 1e-4)
    assert _t_mixture_loglik(log_x, *second) > _t_mixture_loglik(log_x, *first)

    model, ll, _ = fit_tied_scale_t_mixture(log_x, 2, 1, seed=1, n_init=2, max_iter=0)

    _t_assert_model_equals(model, *second)
    assert ll == pytest.approx(_t_mixture_loglik(log_x, *second), rel=1e-10)


def test_fit_tied_scale_em_recovers_peak_and_background_and_converges():
    """
    Full EM on a narrow peak plus a broad background (Student-t draws) must converge under a
    loose tolerance and recover the peak location within 0.03, the shared peak scale within
    15%, the background location within 0.15 and the weights within 0.04; the returned
    log-likelihood is the returned model's.
    """
    log_x = _t_draw_t_mixture(19, 2000, [0.6, 0.4], [-2.6, -0.5], [0.15, 1.0], [10.0, 5.0])
    model, ll, shared = fit_tied_scale_t_mixture(log_x, 1, 1, n_init=1, tol=1e-3)

    mu = model.means_.ravel()
    assert mu[0] == pytest.approx(-2.6, abs=0.03)
    assert shared == pytest.approx(0.15, rel=0.15)
    assert mu[1] == pytest.approx(-0.5, abs=0.15)
    np.testing.assert_allclose(model.weights_, [0.6, 0.4], atol=0.04)
    assert ll == pytest.approx(float(np.sum(model.score_samples(log_x))), rel=1e-12)


def test_fit_tied_scale_tolerance_stops_after_the_second_sweep():
    """
    As for the free mixture, a tolerance above the first improvement but far below ``2 |LL|``
    must stop the tied EM after its second sweep.
    """
    log_x = _t_draw_t_mixture(20, 400, [0.6, 0.4], [-2.6, -0.5], [0.15, 1.0], [10.0, 5.0])
    ll0 = fit_tied_scale_t_mixture(log_x, 1, 1, n_init=1, max_iter=0)[1]
    ll1 = fit_tied_scale_t_mixture(log_x, 1, 1, n_init=1, max_iter=1)[1]
    assert abs(ll1 - ll0) < 300.0 < abs(ll1 + ll0)

    stopped = fit_tied_scale_t_mixture(log_x, 1, 1, n_init=1, max_iter=60, tol=300.0)
    two = fit_tied_scale_t_mixture(log_x, 1, 1, n_init=1, max_iter=2, tol=0.0)
    many = fit_tied_scale_t_mixture(log_x, 1, 1, n_init=1, max_iter=60, tol=0.0)
    assert stopped[1] == two[1]
    assert stopped[1] != many[1]


def test_fit_tied_nested_by_splitting_builds_every_split_start_and_keeps_the_best(monkeypatch):
    """
    The nested fit runs one random-restart fit with ``n_peak + 1`` peaks and then, for every
    peak and for offsets 0 and ``split_delta`` (in units of the shared scale), one fit from the
    start that halves that peak's weight into two copies at ``mu -/+ offset * sigma`` (copy
    inserted right after the parent), duplicates its ``nu``, gives all peaks the mean peak
    variance and renormalises the weights (the base weights here sum to 0.9, so the
    renormalisation is visible). Each split fit uses ``n_init=0`` and the caller's
    seed and ``reg_covar``; the highest log-likelihood among all fits is returned. The inner
    fit is replaced by a recorder returning scripted log-likelihoods.
    """
    base = TMixture(weights=np.array([0.45, 0.18, 0.27]), means=np.array([-2.6, -1.6, 0.0]),
                    covariances=np.array([0.04, 0.05, 1.0]), nus=np.array([8.0, 6.0, 4.0]))
    scripted = [-50.0, -40.0, -30.0, -45.0, -35.0]
    calls = []

    def recording_fit(log_x, n_peak, n_background, **kwargs):
        """Records the call and returns a scripted (model, log-likelihood, scale) triple."""
        calls.append((n_peak, n_background, dict(kwargs)))
        return ("fit", len(calls) - 1), scripted[len(calls) - 1], 0.1

    monkeypatch.setattr(mixture_model_utils, "fit_tied_scale_t_mixture", recording_fit)
    best = fit_tied_nested_by_splitting(np.zeros(10), base, 2, 1, seed=3, n_init=2,
                                        reg_covar=2e-3, split_delta=0.5)

    assert best == (("fit", 2), -30.0, 0.1)
    assert len(calls) == 5
    assert calls[0] == (3, 1, {"seed": 3, "n_init": 2, "reg_covar": 2e-3})
    shared = 0.045
    sigma = np.sqrt(shared)
    expected = []
    for candidate, offset in ((0, 0.0), (0, 0.5), (1, 0.0), (1, 0.5)):
        w = [0.25, 0.25, 0.2, 0.3] if candidate == 0 else [0.5, 0.1, 0.1, 0.3]
        if candidate == 0:
            mu = [-2.6 - offset * sigma, -2.6 + offset * sigma, -1.6, 0.0]
            nu = [8.0, 8.0, 6.0, 4.0]
        else:
            mu = [-2.6, -1.6 - offset * sigma, -1.6 + offset * sigma, 0.0]
            nu = [8.0, 6.0, 6.0, 4.0]
        expected.append((w, mu, [shared, shared, shared, 1.0], nu))
    for (n_peak, n_background, kwargs), (w, mu, s2, nu) in zip(calls[1:], expected):
        assert (n_peak, n_background) == (3, 1)
        assert kwargs["seed"] == 3 and kwargs["n_init"] == 0 and kwargs["reg_covar"] == 2e-3
        got = kwargs["init_params"]
        np.testing.assert_allclose(got[0], w, rtol=1e-12)
        np.testing.assert_allclose(got[1], mu, rtol=1e-12)
        np.testing.assert_allclose(got[2], s2, rtol=1e-12)
        np.testing.assert_allclose(got[3], nu, rtol=1e-12)


def test_fit_tied_nested_by_splitting_keeps_the_random_fit_when_no_split_beats_it(monkeypatch):
    """
    When the random-restart fit scores highest, it is returned unchanged; a split start only
    replaces it by scoring strictly higher, so a split start that merely ties is not taken.
    """
    base = TMixture(weights=np.array([0.7, 0.3]), means=np.array([-2.6, 0.0]),
                    covariances=np.array([0.04, 1.0]), nus=np.array([8.0, 4.0]))
    scripted = [5.0, 5.0, 4.0]
    counter = []

    def recording_fit(log_x, n_peak, n_background, **kwargs):
        """Returns a scripted (model, log-likelihood, scale) triple."""
        counter.append(1)
        return ("fit", len(counter) - 1), scripted[len(counter) - 1], 9.0

    monkeypatch.setattr(mixture_model_utils, "fit_tied_scale_t_mixture", recording_fit)
    best = fit_tied_nested_by_splitting(np.zeros(10), base, 1, 1, n_init=1)

    assert best == (("fit", 0), 5.0, 9.0)


def test_tied_peak_lrt_replicate_draws_warm_starts_and_returns_twice_the_gain(monkeypatch):
    """
    One replicate draws ``n_draw`` points from the null model with ``default_rng(seed)``, fits
    the null warm-started at the generating parameters, fits the alternative by splitting the
    fitted null, passes ``seed``, ``n_init`` and ``reg_covar`` to both, and returns
    ``2 * (LL_alt - LL_null)``. Both fits are replaced by recorders.
    """
    model = TMixture(weights=np.array([0.6, 0.4]), means=np.array([-2.5, 0.0]),
                     covariances=np.array([0.04, 1.0]), nus=np.array([8.0, 4.0]))
    record = {}
    null_fit = object()

    def fake_tied(log_draw, n_peak, n_background, **kwargs):
        """Records the null fit's inputs and returns a scripted result."""
        record["null"] = (np.asarray(log_draw).copy(), n_peak, n_background, dict(kwargs))
        return null_fit, -100.0, 0.2

    def fake_nested(log_draw, null_model, n_peak, n_background, **kwargs):
        """Records the alternative fit's inputs and returns a scripted result."""
        record["alt"] = (np.asarray(log_draw).copy(), null_model, n_peak, n_background, dict(kwargs))
        return object(), -90.5, 0.3

    monkeypatch.setattr(mixture_model_utils, "fit_tied_scale_t_mixture", fake_tied)
    monkeypatch.setattr(mixture_model_utils, "fit_tied_nested_by_splitting", fake_nested)
    lr = _tied_peak_lrt_replicate(model, 50, 1, 1, 3, 2e-3, 17)

    assert lr == pytest.approx(19.0, abs=1e-12)
    expected_draw = _sample_from_mixture(model, 50, np.random.default_rng(17))
    draw, n_peak, n_background, kwargs = record["null"]
    np.testing.assert_array_equal(draw, expected_draw)
    assert (n_peak, n_background) == (1, 1)
    assert (kwargs["seed"], kwargs["n_init"], kwargs["reg_covar"]) == (17, 3, 2e-3)
    start = kwargs["init_params"]
    np.testing.assert_array_equal(start[0], [0.6, 0.4])
    np.testing.assert_array_equal(start[1], [-2.5, 0.0])
    np.testing.assert_array_equal(start[2], [0.04, 1.0])
    np.testing.assert_array_equal(start[3], [8.0, 4.0])
    alt_draw, alt_null, alt_peak, alt_background, alt_kwargs = record["alt"]
    np.testing.assert_array_equal(alt_draw, expected_draw)
    assert alt_null is null_fit
    assert (alt_peak, alt_background) == (1, 1)
    assert alt_kwargs == {"seed": 17, "n_init": 3, "reg_covar": 2e-3}


def _t_patch_tied_lrt(monkeypatch, record, lr_values, null_scores, alt_scores):
    """
    Replaces the fits and the replicate used by :func:`tied_peak_bootstrap_lrt` with recorders.

    Parameters
    ----------
    monkeypatch (pytest.MonkeyPatch)
        The fixture used to patch the module.
    record (dict)
        Receives ``'null'``, ``'alt'`` and ``'replicates'`` entries.
    lr_values (list)
        Replicate statistics, returned in order of the replicate seeds.
    null_scores (np.ndarray)
        Per-observation log-likelihoods of the stand-in null model.
    alt_scores (np.ndarray)
        Per-observation log-likelihoods of the stand-in alternative model.

    Returns
    -------
    models (tuple)
        The stand-in ``(null_model, alt_model)``.
    """
    null_model = _TFakeScorer(null_scores)
    alt_model = _TFakeScorer(alt_scores)
    record["replicates"] = []

    def fake_tied(log_sub, n_peak, n_background, **kwargs):
        """Records the observed null fit and returns a scripted result."""
        record["null"] = (np.asarray(log_sub).copy(), n_peak, n_background, dict(kwargs))
        return null_model, -120.0, 0.25

    def fake_nested(log_sub, base, n_peak, n_background, **kwargs):
        """Records the observed alternative fit and returns a scripted result."""
        record["alt"] = (base, n_peak, n_background, dict(kwargs))
        return alt_model, -118.0, 0.2

    def fake_replicate(model, n_draw, n_peak, n_background, n_init, reg_covar, seed):
        """Records one replicate's arguments and returns its scripted statistic."""
        record["replicates"].append((model, n_draw, n_peak, n_background, n_init, reg_covar, seed))
        return lr_values[len(record["replicates"]) - 1]

    monkeypatch.setattr(mixture_model_utils, "fit_tied_scale_t_mixture", fake_tied)
    monkeypatch.setattr(mixture_model_utils, "fit_tied_nested_by_splitting", fake_nested)
    monkeypatch.setattr(mixture_model_utils, "_tied_peak_lrt_replicate", fake_replicate)
    return null_model, alt_model


def test_tied_peak_bootstrap_lrt_plumbing_and_summary_without_sessions(monkeypatch):
    """
    Without session labels: the observed fits get the subsample's logs, the caller's seed,
    ``n_init_obs`` and ``reg_covar``; replicate ``b`` gets the fitted null, the subsample size,
    ``n_init_boot`` and seed ``seed + 1 + b``; ``lr_obs = 2 (LL_alt - LL_null)``; the p-value
    is (1 + replicates at least as large as ``lr_obs``, ties included) / (1 + B); the 95th percentile,
    the strictly-negative fraction and the shared null scale are reported; and no correction
    keys appear.
    """
    pool = np.exp(np.linspace(-3.0, 0.0, 40))
    record = {}
    lr_values = [4.0, -0.5, 0.0, 0.5, 10.0]
    null_model, _ = _t_patch_tied_lrt(monkeypatch, record, lr_values, np.zeros(40), np.ones(40))

    result = tied_peak_bootstrap_lrt(pool, 2, 1, B=5, n_subsample=1000, n_init_obs=3,
                                     n_init_boot=2, reg_covar=2e-3, seed=5)

    log_sub, n_peak, n_background, kwargs = record["null"]
    np.testing.assert_allclose(log_sub, np.log(pool), rtol=1e-12)
    assert (n_peak, n_background, kwargs) == (2, 1, {"seed": 5, "n_init": 3, "reg_covar": 2e-3})
    base, alt_peak, alt_background, alt_kwargs = record["alt"]
    assert base is null_model
    assert (alt_peak, alt_background, alt_kwargs) == (2, 1, {"seed": 5, "n_init": 3, "reg_covar": 2e-3})
    assert [r[6] for r in record["replicates"]] == [6, 7, 8, 9, 10]
    for model, n_draw, rp, rb, n_init, reg_covar, _ in record["replicates"]:
        assert model is null_model
        assert (n_draw, rp, rb, n_init, reg_covar) == (40, 2, 1, 2, 2e-3)

    assert result["n_peak_null"] == 2
    assert result["n_peak_alt"] == 3
    assert result["n_background"] == 1
    assert result["B"] == 5
    assert result["n_subsample"] == 40
    assert result["lr_obs"] == pytest.approx(4.0, abs=1e-12)
    np.testing.assert_array_equal(result["lr_null"], lr_values)
    assert result["lr_null"].dtype == np.float64
    assert result["p_value"] == pytest.approx(3.0 / 6.0)
    assert result["null_p95"] == pytest.approx(float(np.percentile(lr_values, 95)))
    assert result["null_p95"] != pytest.approx(float(np.percentile(lr_values, 96)))
    assert result["negative_fraction"] == pytest.approx(0.2)
    assert result["shared_scale_null"] == 0.25
    assert "lr_corrected" not in result
    assert "p_value_corrected" not in result


def test_tied_peak_bootstrap_lrt_null_start_joins_the_observed_null_fit(monkeypatch):
    """
    A supplied ``null_start`` (the previous rung's alternative) is handed to the observed null
    fit as its warm start, next to the caller's scratch restarts, so the fit keeps the better of
    the two; the returned null and alternative models are the observed fits, so a caller can pass
    this rung's alternative to the next rung.
    """
    pool = np.exp(np.linspace(-3.0, 0.0, 40))
    record = {}
    null_model, alt_model = _t_patch_tied_lrt(monkeypatch, record, [1.0, 2.0], np.zeros(40),
                                              np.ones(40))
    start = TMixture(weights=[0.5, 0.3, 0.2], means=[-2.5, -1.5, 0.0],
                     covariances=[0.04, 0.04, 1.0], nus=[8.0, 9.0, 4.0])

    result = tied_peak_bootstrap_lrt(pool, 2, 1, B=2, n_subsample=1000, n_init_obs=3, seed=5,
                                     null_start=start)

    _, n_peak, n_background, kwargs = record["null"]
    assert (n_peak, n_background) == (2, 1)
    assert (kwargs["seed"], kwargs["n_init"]) == (5, 3)
    np.testing.assert_array_equal(kwargs["init_params"][0], [0.5, 0.3, 0.2])
    np.testing.assert_array_equal(kwargs["init_params"][1], [-2.5, -1.5, 0.0])
    np.testing.assert_array_equal(kwargs["init_params"][2], [0.04, 0.04, 1.0])
    np.testing.assert_array_equal(kwargs["init_params"][3], [8.0, 9.0, 4.0])
    assert result["null_model"] is null_model
    assert result["alt_model"] is alt_model


def test_tied_peak_bootstrap_lrt_rejects_a_null_start_of_the_wrong_size():
    """A ``null_start`` whose component count is not ``n_peak_null + n_background`` cannot seed
    the null and must raise before any fitting."""
    pool = np.exp(np.linspace(-3.0, 0.0, 40))
    start = TMixture(weights=[0.5, 0.5], means=[-2.5, 0.0], covariances=[0.04, 1.0], nus=[8.0, 4.0])
    with pytest.raises(ValueError, match="null_start has 2 components"):
        tied_peak_bootstrap_lrt(pool, 2, 1, B=1, n_subsample=1000, null_start=start)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_tied_peak_bootstrap_lrt_laddered_null_is_never_worse_than_its_start():
    """
    With real fits, the observed null of rung ``n`` seeded by rung ``n - 1``'s alternative
    attains at least that alternative's log-likelihood on the same subsample (EM from a tied
    start cannot descend), so a scratch null that stops short can no longer inflate the
    statistic. One scratch restart is used so the warm start has to do the work.
    """
    rng = np.random.default_rng(3)
    pool = np.exp(np.concatenate([rng.normal(-2.77, 0.24, 1400), rng.normal(-1.7, 0.24, 300),
                                  rng.normal(-1.0, 0.24, 200), rng.normal(-0.5, 1.2, 400)]))
    first = tied_peak_bootstrap_lrt(pool, 1, 1, B=1, n_subsample=1500, n_init_obs=2,
                                    n_init_boot=1, seed=0)
    second = tied_peak_bootstrap_lrt(pool, 2, 1, B=1, n_subsample=1500, n_init_obs=1,
                                     n_init_boot=1, seed=0, null_start=first["alt_model"])
    log_sub = np.log(stable_subsample(pool, 1500, 0))

    ll_start = float(np.sum(first["alt_model"].score_samples(log_sub)))
    ll_null = float(np.sum(second["null_model"].score_samples(log_sub)))
    assert ll_null >= ll_start - 1e-6


def test_tied_peak_bootstrap_lrt_session_correction_uses_the_observed_delta(monkeypatch):
    """
    With session labels the correction is computed from ``delta = LL_alt_i - LL_null_i`` of the
    observed fits with the caller's seed, exactly as :func:`_session_corrected_statistic` does.
    Here every session holds a +1 / -1 pair, so the session totals vanish, the raw design effect
    is zero and the floored effect is one: the corrected statistic equals ``lr_obs`` and the
    corrected p-value counts the replicate tied with it: (1 + 2) / (1 + 5).
    """
    pool = np.exp(np.linspace(-3.0, 0.0, 40))
    labels = np.repeat(np.arange(20), 2)
    null_scores = np.full(40, 0.25)
    alt_scores = null_scores + np.tile([1.0, -1.0], 20)
    record = {}
    lr_values = [4.0, -0.5, 0.0, 0.5, 10.0]
    _t_patch_tied_lrt(monkeypatch, record, lr_values, null_scores, alt_scores)

    result = tied_peak_bootstrap_lrt(pool, 1, 1, B=5, n_subsample=1000, seed=5,
                                     session_labels=labels, n_design_bootstrap=50)
    expected = _session_corrected_statistic(4.0, alt_scores - null_scores, labels, 40,
                                            np.arange(40), 50, 5, "tied_peak_bootstrap_lrt")

    for key in ("design_effect_raw", "design_effect", "clustered_se", "independent_se",
                "effective_n", "lr_corrected"):
        assert result[key] == pytest.approx(expected[key], rel=1e-12, abs=1e-12)
    assert result["design_effect_raw"] == 0.0
    assert result["lr_corrected"] == 4.0
    assert result["p_value_corrected"] == pytest.approx(3.0 / 6.0)


def test_tied_peak_bootstrap_lrt_design_effect_follows_the_seed(monkeypatch):
    """
    The session bootstrap behind the design effect is seeded by the caller's seed, so a
    non-degenerate delta gives exactly the design effect :func:`_session_corrected_statistic`
    returns for that seed.
    """
    pool = np.exp(np.linspace(-3.0, 0.0, 60))
    labels = np.repeat(np.arange(12), 5)
    rng = np.random.default_rng(21)
    null_scores = rng.normal(0.0, 1.0, 60)
    alt_scores = null_scores + rng.normal(0.1, 1.0, 60) + np.repeat(rng.normal(0.0, 1.0, 12), 5)
    record = {}
    _t_patch_tied_lrt(monkeypatch, record, [1.0, 2.0], null_scores, alt_scores)

    result = tied_peak_bootstrap_lrt(pool, 1, 1, B=2, n_subsample=1000, seed=9,
                                     session_labels=labels, n_design_bootstrap=60)
    expected = _session_corrected_statistic(4.0, alt_scores - null_scores, labels, 60,
                                            np.arange(60), 60, 9, "tied_peak_bootstrap_lrt")

    assert result["design_effect_raw"] == pytest.approx(expected["design_effect_raw"], rel=1e-12)
    assert result["clustered_se"] == pytest.approx(expected["clustered_se"], rel=1e-12)


def test_t_mixture_modes_are_polished_to_the_exact_gradient_root():
    """
    Every reported mode must be a root of the analytic mixture score ``d/dx log p`` to 1e-9: the
    Newton polish (quadratic near a maximum) stops only when a step is below ``1e-7 (1 + |x|)``,
    so a wrong curvature, which slows it to linear convergence, leaves a measurable error. Two
    overlapping components and a narrow, remote third one exercise both the mixed-curvature
    term and a peak much narrower than the widest scale.
    """
    model = TMixture(weights=np.array([0.5, 0.3, 0.2]), means=np.array([-2.0, -1.4, 1.5]),
                     covariances=np.array([0.16, 0.36, 1e-4]), nus=np.array([3.0, 6.0, 5.0]))
    modes, densities = t_mixture_modes(model)

    assert modes.size == 2
    for mode in modes:
        root = brentq(_t_mixture_grad, mode - 0.005, mode + 0.005, args=(model,), xtol=1e-14)
        assert mode == pytest.approx(root, abs=1e-9)
    np.testing.assert_allclose(densities, np.exp(model.score_samples(modes)), rtol=1e-12)
    assert densities[0] >= densities[1]


def test_t_mixture_modes_resolve_narrow_peaks_on_the_grid():
    """
    The grid spans six of the widest scales past the extreme locations, so its spacing tracks
    the scales; two narrow (scale 0.01, nu 5) peaks one unit apart must both be found and
    polished onto their locations to 1e-6.
    """
    model = TMixture(weights=np.array([0.5, 0.5]), means=np.array([0.0, 1.0]),
                     covariances=np.array([1e-4, 1e-4]), nus=np.array([5.0, 5.0]))
    modes, _ = t_mixture_modes(model)

    np.testing.assert_allclose(np.sort(modes), [0.0, 1.0], atol=1e-6)


def test_t_mixture_modes_single_newton_step_from_a_concave_grid_point():
    """
    With a three-point grid the only candidate is the midpoint of the extreme locations. Where
    the log-density is concave there, one iteration is one Newton step ``x - g / h`` with ``g``
    and ``h`` the first two derivatives of ``log p``, compared here with finite differences.
    """
    model = TMixture(weights=np.array([0.7, 0.3]), means=np.array([0.0, 1.0]),
                     covariances=np.array([0.36, 0.36]), nus=np.array([5.0, 5.0]))
    g, h = _t_numeric_derivatives(0.5, model)
    assert h < 0.0 and abs(g) > 0.1

    modes, _ = t_mixture_modes(model, max_iter=1, grid_size=3)

    assert modes.size == 1
    assert modes[0] == pytest.approx(0.5 - g / h, abs=1e-5)


def test_t_mixture_modes_gradient_fallback_step_on_a_convex_grid_point():
    """
    Where the log-density is convex at the candidate (between two separated components) the
    Newton step would head for the minimum, so the polish takes the small gradient step
    ``x + 1e-3 * g`` instead; one iteration from the three-point grid's midpoint must move by
    exactly that.
    """
    model = TMixture(weights=np.array([0.6, 0.4]), means=np.array([-1.0, 1.0]),
                     covariances=np.array([0.09, 0.09]), nus=np.array([3.0, 3.0]))
    g, h = _t_numeric_derivatives(0.0, model)
    assert h > 1.0 and abs(g) > 0.1

    modes, _ = t_mixture_modes(model, max_iter=1, grid_size=3)

    assert modes.size == 1
    assert modes[0] == pytest.approx(1e-3 * g, rel=1e-5)


def test_t_mixture_modes_without_interior_grid_points_returns_empty_arrays():
    """
    A two-point grid has no interior point and so no candidate maximum; both returned arrays
    must be empty with shape ``(0,)``.
    """
    modes, densities = t_mixture_modes(_t_model_two(), grid_size=2)

    assert modes.shape == (0,)
    assert densities.shape == (0,)


def test_t_mixture_modes_gradient_fallback_also_applies_to_weak_convexity():
    """
    The fallback is taken whenever the curvature is non-negative, however small: with
    wide, well-separated components the midpoint's second derivative is positive but below one,
    and the single step must still be the gradient step ``x + 1e-3 * g``.
    """
    model = TMixture(weights=np.array([0.6, 0.4]), means=np.array([-5.0, 5.0]),
                     covariances=np.array([2.25, 2.25]), nus=np.array([3.0, 3.0]))
    g, h = _t_numeric_derivatives(0.0, model)
    assert 0.0 < h < 1.0

    modes, _ = t_mixture_modes(model, max_iter=1, grid_size=3)

    assert modes[0] == pytest.approx(1e-3 * g, rel=1e-5)


def _t_one_free_em_sweep(log_x, w, mu, sigma2, nu, reg_covar):
    """
    One Peel & McLachlan EM sweep for a free-scale Student-t mixture, written out by hand.

    Parameters
    ----------
    log_x (np.ndarray)
        A (N,) shape ndarray of observations.
    w, mu, sigma2, nu (np.ndarray)
        Current parameters, shape (K,) each.
    reg_covar (float)
        Variance floor.

    Returns
    -------
    sweep (tuple)
        ``(w, mu, sigma2, nu, n_k, scatter)`` after the sweep: responsibilities ``z`` from the
        current parameters, latent weights ``u = (nu + 1) / (nu + delta)``, ``mu`` the
        ``z u``-weighted mean (denominator floored at 1e-10), ``scatter`` the ``z u``-weighted
        squared deviation from the new means, ``sigma2 = scatter / max(n_k, 1e-10)`` floored at
        ``reg_covar``, ``w = n_k / N`` and ``nu`` from :func:`_t_update_nu`.
    """
    log_pdf = np.array([np.log(w[k]) + stats.t.logpdf(log_x, df=nu[k], loc=mu[k], scale=np.sqrt(sigma2[k]))
                        for k in range(len(w))])
    z = np.exp(log_pdf - np.logaddexp.reduce(log_pdf, axis=0))
    delta = (log_x[None, :] - mu[:, None]) ** 2 / sigma2[:, None]
    u = (nu[:, None] + 1.0) / (nu[:, None] + delta)
    n_k = z.sum(axis=1)
    zu = z * u
    mu_new = (zu * log_x).sum(axis=1) / np.maximum(zu.sum(axis=1), 1e-10)
    scatter = (zu * (log_x - mu_new[:, None]) ** 2).sum(axis=1)
    sigma2_new = np.maximum(scatter / np.maximum(n_k, 1e-10), reg_covar)
    nu_new = np.array([_t_update_nu(z[k], u[k], nu[k], max(n_k[k], 1e-10)) for k in range(len(w))])
    return n_k / log_x.size, mu_new, sigma2_new, nu_new, n_k, scatter


def _t_weak_component_pool():
    """
    Pool and three-component start in which the third component receives a total
    responsibility of only ~0.025, below one observation.

    Parameters
    ----------

    Returns
    -------
    pool (tuple)
        ``(log_x, start)`` with ``log_x`` a (300,) ndarray (two normal clusters at -2.5 and 0.5)
        and ``start`` the ``(w, mu, sigma2, nu)`` tuple.
    """
    rng = np.random.default_rng(22)
    log_x = np.concatenate([rng.normal(-2.5, 0.2, 200), rng.normal(0.5, 0.3, 100)])
    start = (np.array([0.6, 0.39, 0.01]), np.array([-2.5, 0.5, 1.6]),
             np.array([0.04, 0.09, 0.01]), np.array([6.0, 6.0, 10.0]))
    return log_x, start


def test_fit_log_t_mixture_one_em_sweep_matches_the_hand_update():
    """
    With the explicit start as the only restart and ``max_iter=1`` the returned model is exactly
    one EM sweep from that start. The start includes a component with total responsibility
    ~0.025, so its location divides by the true (sub-one) ``z u`` mass and its variance by the
    true (sub-one) ``n_k``: the 1e-10 floors only guard empty components and must not bind here.
    """
    log_x, start = _t_weak_component_pool()
    w, mu, sigma2, nu, n_k, _ = _t_one_free_em_sweep(log_x, *start, 1e-4)
    assert 1e-3 < n_k[2] < 0.1
    assert sigma2[2] > 2e-4

    model, _ = fit_log_t_mixture(log_x, 3, n_init=0, max_iter=1, x_is_log=True, init_params=start)

    np.testing.assert_allclose(model.weights_, w, rtol=1e-9)
    np.testing.assert_allclose(model.means_.ravel(), mu, rtol=1e-9)
    np.testing.assert_allclose(model.covariances_.ravel(), sigma2, rtol=1e-9)
    np.testing.assert_allclose(model.nus_, nu, rtol=1e-9)


def test_fit_tied_scale_one_em_sweep_pools_the_peak_scatter():
    """
    One tied EM sweep from an explicit start equals the free sweep except that the peak
    variance is the pooled ``sum_peaks scatter / sum_peaks n_k`` (floored at ``reg_covar``). The
    single peak here is the weak component (total responsibility ~0.025), so the pooled
    denominator is below one and the 1e-10 floor must not bind; the returned log-likelihood is
    that of the swept model.
    """
    log_x, start = _t_weak_component_pool()
    order = np.array([2, 0, 1])
    tied_start = tuple(p[order] for p in start)
    w, mu, sigma2, nu, n_k, scatter = _t_one_free_em_sweep(log_x, *tied_start, 1e-4)
    sigma2[0] = max(scatter[0] / n_k[0], 1e-4)
    assert n_k[0] < 0.1 and sigma2[0] > 2e-4

    model, ll, shared = fit_tied_scale_t_mixture(log_x, 1, 2, n_init=0, max_iter=1,
                                                 init_params=tied_start)

    np.testing.assert_allclose(model.weights_, w, rtol=1e-9)
    np.testing.assert_allclose(model.means_.ravel(), mu, rtol=1e-9)
    np.testing.assert_allclose(model.covariances_.ravel(), sigma2, rtol=1e-9)
    np.testing.assert_allclose(model.nus_, nu, rtol=1e-9)
    assert shared == pytest.approx(np.sqrt(sigma2[0]), rel=1e-9)
    assert ll == pytest.approx(float(np.sum(model.score_samples(log_x))), rel=1e-12)


def test_fit_log_t_mixture_ranks_restarts_by_total_log_likelihood():
    """
    Restarts are ranked by the total log-likelihood ``sum_i log sum_k w_k p_k(x_i)``. An
    explicit start that puts a needle component on three repeated values has by far the
    highest density at those points (and the highest pooled or per-component log-sum-exp) but
    a lower total log-likelihood than the KMeans start, so the KMeans start must be returned.
    """
    rng = np.random.default_rng(23)
    log_x = np.concatenate([rng.normal(-2.0, 0.4, 297), np.zeros(3)])
    needle = (np.array([0.5, 0.5]), np.array([0.0, -2.0]), np.array([1e-12, 0.16]),
              np.array([10.0, 10.0]))
    kmeans_start = _t_kmeans_start(log_x, 2, 0, 1e-14)
    assert _t_mixture_loglik(log_x, *needle) < _t_mixture_loglik(log_x, *kmeans_start)

    model, _ = fit_log_t_mixture(log_x, 2, n_init=1, max_iter=0, x_is_log=True, reg_covar=1e-14,
                                 init_params=needle)

    _t_assert_model_equals(model, *kmeans_start)


def test_fit_log_t_mixture_keeps_the_earlier_restart_on_an_exact_tie():
    """
    A later restart replaces the incumbent only by scoring strictly higher. The explicit start
    here is the KMeans start with its components swapped, which has a bit-identical
    log-likelihood, so the KMeans ordering must be kept.
    """
    log_x = _t_draw_t_mixture(24, 300, [0.6, 0.4], [-2.0, 0.0], [0.2, 0.3], [6.0, 6.0])
    w, mu, sigma2, nu = _t_kmeans_start(log_x, 2, 0, 1e-4)
    swapped = (w[::-1].copy(), mu[::-1].copy(), sigma2[::-1].copy(), nu[::-1].copy())

    model, _ = fit_log_t_mixture(log_x, 2, n_init=1, max_iter=0, x_is_log=True, init_params=swapped)

    _t_assert_model_equals(model, w, mu, sigma2, nu)


def test_fit_tied_scale_keeps_the_earlier_restart_on_an_exact_tie():
    """
    As for the free mixture, an explicit start that is the density-seeded start with its two
    peaks swapped ties it exactly and must not displace it.
    """
    rng = np.random.default_rng(18)
    log_x = np.concatenate([rng.normal(-2.7, 0.2, 3000), rng.normal(-1.6, 0.2, 800),
                            rng.normal(0.0, 1.2, 600)])
    w, mu, sigma2, nu, _ = _t_tied_density_start(log_x, 2, 1, 1e-4)
    swap = np.array([1, 0, 2])

    model, _, _ = fit_tied_scale_t_mixture(log_x, 2, 1, n_init=1, max_iter=0,
                                           init_params=(w[swap], mu[swap], sigma2[swap], nu[swap]))

    _t_assert_model_equals(model, w, mu, sigma2, nu)


def _g_make_gmm(means, variances, weights):
    """
    Builds a 1D full-covariance ``GaussianMixture`` from explicit parameters
    without running EM, so tests can place components exactly.

    Parameters
    ----------
    means (list)
        Per-component log-space means.
    variances (list)
        Per-component log-space variances.
    weights (list)
        Per-component mixing weights.

    Returns
    -------
    gmm (GaussianMixture)
        A GaussianMixture whose ``means_``, ``covariances_``, ``weights_`` and
        ``precisions_cholesky_`` are set so ``score_samples``,
        ``predict_proba`` and ``bic`` work as on a fitted model.
    """
    k = len(means)
    gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=0)
    gmm.means_ = np.asarray(means, dtype=float).reshape(k, 1)
    gmm.covariances_ = np.asarray(variances, dtype=float).reshape(k, 1, 1)
    gmm.weights_ = np.asarray(weights, dtype=float)
    gmm.precisions_cholesky_ = 1.0 / np.sqrt(gmm.covariances_)
    return gmm


def _g_make_gmm_2d(means, covariances, weights):
    """
    Builds a 2D full-covariance ``GaussianMixture`` from explicit parameters
    without running EM.

    Parameters
    ----------
    means (list)
        A (K, 2) nested list of component means.
    covariances (list)
        A (K, 2, 2) nested list of component covariance matrices.
    weights (list)
        Per-component mixing weights.

    Returns
    -------
    gmm (GaussianMixture)
        A GaussianMixture usable by ``score_samples`` and by the general-d
        branch of :func:`gmm_modes`.
    """
    means = np.asarray(means, dtype=float)
    covariances = np.asarray(covariances, dtype=float)
    gmm = GaussianMixture(n_components=means.shape[0], covariance_type="full", random_state=0)
    gmm.means_ = means
    gmm.covariances_ = covariances
    gmm.weights_ = np.asarray(weights, dtype=float)
    gmm.precisions_cholesky_ = np.array([np.linalg.cholesky(np.linalg.inv(c)) for c in covariances])
    return gmm


def _g_mix_derivatives(x, means, sds, weights):
    """
    Analytic first and second derivatives of the log of a 1D Gaussian mixture
    density, computed from scipy densities (independently of the module).

    Parameters
    ----------
    x (float or np.ndarray)
        Evaluation point(s).
    means (list)
        Component means.
    sds (list)
        Component standard deviations.
    weights (list)
        Component weights.

    Returns
    -------
    grad (float or np.ndarray)
        ``d/dx log p(x)``.
    hess (float or np.ndarray)
        ``d^2/dx^2 log p(x)``.
    """
    p = 0.0
    dp = 0.0
    d2p = 0.0
    for m, s, w in zip(means, sds, weights, strict=True):
        n = w * stats.norm.pdf(x, m, s)
        p += n
        dp += n * (m - x) / s ** 2
        d2p += n * (((m - x) / s ** 2) ** 2 - 1.0 / s ** 2)
    grad = dp / p
    return grad, d2p / p - grad ** 2


def _g_mix_logpdf(x, means, sds, weights):
    """
    Log-density of a 1D Gaussian mixture from scipy densities.

    Parameters
    ----------
    x (np.ndarray)
        Evaluation points.
    means (list)
        Component means.
    sds (list)
        Component standard deviations.
    weights (list)
        Component weights.

    Returns
    -------
    log_p (np.ndarray)
        Mixture log-density at ``x``.
    """
    x = np.asarray(x, dtype=float)
    return np.log(sum(w * stats.norm.pdf(x, m, s) for m, s, w in zip(means, sds, weights, strict=True)))


def _g_grid_peaks(means, sds, weights, grid_size):
    """
    Rebuilds the documented 1D mode-search grid of :func:`gmm_modes`
    (``[mu_min - 4 sigma_max, mu_max + 4 sigma_max]``, ``grid_size`` points)
    and returns the grid locations of its strict interior maxima.

    Parameters
    ----------
    means (list)
        Component means.
    sds (list)
        Component standard deviations.
    weights (list)
        Component weights.
    grid_size (int)
        Number of grid points.

    Returns
    -------
    starts (np.ndarray)
        Grid values of the strict interior local maxima of the log-density.
    """
    lo = min(means) - 4.0 * max(sds)
    hi = max(means) + 4.0 * max(sds)
    grid = np.linspace(lo, hi, grid_size)
    lp = _g_mix_logpdf(grid, means, sds, weights)
    idx = np.where((lp[1:-1] > lp[:-2]) & (lp[1:-1] > lp[2:]))[0] + 1
    return grid[idx]


def _g_true_modes_1d(means, sds, weights):
    """
    Locates every mode of a 1D Gaussian mixture to machine precision by
    bracketing sign changes of the analytic log-density gradient on a fine
    grid and polishing each with brentq.

    Parameters
    ----------
    means (list)
        Component means.
    sds (list)
        Component standard deviations.
    weights (list)
        Component weights.

    Returns
    -------
    modes (np.ndarray)
        Ascending mode locations.
    """
    grid = np.linspace(min(means) - 1.0, max(means) + 1.0, 20001)
    grads = _g_mix_derivatives(grid, means, sds, weights)[0]
    out = []
    for i in range(grid.size - 1):
        if grads[i] > 0.0 >= grads[i + 1]:
            out.append(brentq(lambda v: _g_mix_derivatives(v, means, sds, weights)[0],
                              grid[i], grid[i + 1], xtol=1e-14))
    return np.array(out)


def _g_boundary_roots(mu1, s1, w1, mu2, s2, w2, tau):
    """
    Finds every real crossing of the posterior-``tau`` boundary equation
    ``log(w1 N1(x)) - log(w2 N2(x)) = log(tau / (1 - tau))`` by brentq on a
    hand-written crossing function, scanning a wide window for sign changes.

    Parameters
    ----------
    mu1, s1, w1 (float)
        Left component mean, standard deviation and weight.
    mu2, s2, w2 (float)
        Right component mean, standard deviation and weight.
    tau (float)
        Posterior threshold of the left component.

    Returns
    -------
    roots (list)
        All real roots found in ``[mu1 - 20, mu2 + 20]``.
    """
    def f(v):
        """
        Crossing function whose zeros are the posterior-``tau`` boundaries.

        Parameters
        ----------
        v (float or np.ndarray)
            Evaluation point(s).

        Returns
        -------
        value (float or np.ndarray)
            Log-weighted-density difference minus the tau offset.
        """
        return (np.log(w1) + stats.norm.logpdf(v, mu1, s1)
                - np.log(w2) - stats.norm.logpdf(v, mu2, s2)
                - np.log(tau / (1.0 - tau)))

    grid = np.linspace(mu1 - 20.0, mu2 + 20.0, 40001)
    vals = f(grid)
    return [brentq(f, grid[i], grid[i + 1], xtol=1e-14)
            for i in range(grid.size - 1) if np.sign(vals[i]) != np.sign(vals[i + 1])]


def _g_ig_intervals(seed=0, n_per=150):
    """
    Draws a small, well-separated two-component inverse-Gaussian sample
    (IG(mu=0.07, lam=1.0) and IG(mu=1.5, lam=3.0)) for the IG EM tests.

    Parameters
    ----------
    seed (int)
        RNG seed.
    n_per (int)
        Samples per component.

    Returns
    -------
    x (np.ndarray)
        A (2 * n_per,) shape ndarray of strictly positive intervals.
    """
    rng = np.random.default_rng(seed)
    fast = stats.invgauss.rvs(0.07 / 1.0, scale=1.0, size=n_per, random_state=rng)
    slow = stats.invgauss.rvs(1.5 / 3.0, scale=3.0, size=n_per, random_state=rng)
    return np.concatenate([fast, slow])


def _g_ig_three_cluster_intervals():
    """
    Draws a fixed three-cluster inverse-Gaussian sample (120 per cluster)
    used to pin the restart initialisations of :func:`fit_log_ig_mixture`.

    Returns
    -------
    x (np.ndarray)
        A (360,) shape ndarray of strictly positive intervals.
    """
    rng = np.random.default_rng(0)
    return np.concatenate([
        stats.invgauss.rvs(0.05 / 0.5, scale=0.5, size=120, random_state=rng),
        stats.invgauss.rvs(0.4 / 2.0, scale=2.0, size=120, random_state=rng),
        stats.invgauss.rvs(3.0 / 6.0, scale=6.0, size=120, random_state=rng),
    ])


def _g_ig_init_from_labels(x, labels, k, reg_covar=1e-4):
    """
    Hand implementation of the documented IG initialisation from hard
    labels: per-cluster moment matching (``mu = mean``, ``lam = mean^3 /
    var`` capped at ``1 / reg_covar``, an empty cluster falling back to all
    data) and label-frequency weights floored at 1e-3 then renormalised.

    Parameters
    ----------
    x (np.ndarray)
        Linear-time intervals.
    labels (np.ndarray)
        Integer cluster label per interval.
    k (int)
        Number of components.
    reg_covar (float)
        The fitter's ``reg_covar``.

    Returns
    -------
    w (np.ndarray)
        Initial weights.
    mu (np.ndarray)
        Initial means.
    lam (np.ndarray)
        Initial shape parameters.
    """
    mu = np.empty(k)
    lam = np.empty(k)
    for j in range(k):
        xk = x[labels == j] if np.any(labels == j) else x
        m = float(np.mean(xk))
        v = float(np.var(xk)) + 1e-12
        mu[j] = m
        lam[j] = min(m ** 3 / v, 1.0 / reg_covar)
    w = np.array([np.mean(labels == j) for j in range(k)])
    w = np.where(w < 1e-3, 1e-3, w)
    return w / w.sum(), mu, lam


def _g_ig_linear_loglik(x, w, mu, lam):
    """
    Linear-time IG mixture log-likelihood from scipy's ``invgauss``.

    Parameters
    ----------
    x (np.ndarray)
        Linear-time intervals.
    w (np.ndarray)
        Weights.
    mu (np.ndarray)
        Means.
    lam (np.ndarray)
        Shape parameters.

    Returns
    -------
    ll (float)
        Total log-likelihood.
    """
    lp = np.array([np.log(w[j]) + stats.invgauss.logpdf(x, mu[j] / lam[j], scale=lam[j])
                   for j in range(len(w))])
    return float(np.sum(np.logaddexp.reduce(lp, axis=0)))


def test_fit_log_gmm_forwards_every_constructor_argument():
    """
    :func:`fit_log_gmm` must build its ``GaussianMixture`` with exactly the
    supplied ``n_components``, ``n_init``, ``reg_covar`` and ``random_state``
    (full covariance) and return the argsort of the fitted means.

    The fitted estimator keeps its constructor arguments as attributes, so a
    dropped or replaced keyword surfaces as a sklearn default (1 component,
    1 init, 1e-6 regularisation, ``None`` seed). Two fits with the same seed
    must agree exactly, and the returned order must sort the means ascending.
    """
    rng = np.random.default_rng(0)
    log_x = np.concatenate([rng.normal(-2.5, 0.2, 150), rng.normal(0.5, 0.3, 150)])
    gmm, order = fit_log_gmm(np.exp(log_x), 2, seed=3, n_init=2, reg_covar=1e-4)
    assert gmm.n_components == 2
    assert gmm.n_init == 2
    assert gmm.reg_covar == 1e-4
    assert gmm.random_state == 3
    assert gmm.covariance_type == "full"
    np.testing.assert_array_equal(order, np.argsort(gmm.means_.flatten()))
    assert gmm.means_.flatten()[order][0] == pytest.approx(-2.5, abs=0.1)
    gmm_again, _ = fit_log_gmm(np.exp(log_x), 2, seed=3, n_init=2, reg_covar=1e-4)
    np.testing.assert_array_equal(gmm.means_, gmm_again.means_)


@pytest.mark.parametrize(("offset", "flip"), [(0.0, 1), (1.0, 1), (0.75, -1)])
def test_gmm_boundaries_logspace_picks_the_root_nearest_the_midpoint(offset, flip):
    """
    When neither real root of the boundary quadratic lies between the two
    component means, :func:`gmm_boundaries_logspace` must return the root
    closest to the midpoint of the means.

    The configuration (broad heavy left component at ``0``, narrower lighter
    right component at ``0.5``, and its mirror image) has both crossings on
    the same side of the right mean, with a small positive discriminant. The
    expected value is the crossing nearest ``(mu1 + mu2) / 2`` among all real
    crossings of the hand-written posterior equation, found by brentq; three
    offsets / mirrorings make every alternative tie-break rule choose wrong in
    at least one case.
    """
    if flip == 1:
        params = [(0.0 + offset, 1.0, 0.63), (0.5 + offset, 0.7, 0.37)]
    else:
        params = [(-0.5 + offset, 0.7, 0.37), (0.0 + offset, 1.0, 0.63)]
    gmm = _g_make_gmm([p[0] for p in params], [p[1] ** 2 for p in params], [p[2] for p in params])
    (mu1, s1, w1), (mu2, s2, w2) = params
    roots = _g_boundary_roots(mu1, s1, w1, mu2, s2, w2, 0.5)
    assert len(roots) == 2
    assert not any(mu1 <= r <= mu2 for r in roots)
    mid = 0.5 * (mu1 + mu2)
    expected = min(roots, key=lambda r: abs(r - mid))

    log_b, sec_b = gmm_boundaries_logspace(gmm, tau=0.5)

    assert log_b[0] == pytest.approx(expected, abs=1e-9)
    assert sec_b[0] == pytest.approx(np.exp(expected), rel=1e-9)


@pytest.mark.parametrize("params", [
    [(-0.7, 1.97, 0.62), (2.29, 1.43, 0.38)],
    [(-1.75, 1.0, 0.63), (-1.25, 0.7, 0.37)],
    [(-0.75, 1.0, 0.64), (-0.25, 0.7, 0.36)],
])
def test_gmm_boundaries_logspace_matches_hand_crossings_for_unequal_pairs(params):
    """
    For a single pair of unequal-variance components the boundary must be
    the crossing of the posterior equation that lies between the means if
    one exists, and otherwise the crossing nearest the midpoint.

    The expected value is selected from all real crossings of the
    hand-written equation (found by brentq). The pairs cover a broad pair
    with a crossing between the means and two narrow/broad pairs whose
    crossings both lie outside the means, at locations where a mis-computed
    first quadratic root would be (wrongly) preferred.
    """
    (mu1, s1, w1), (mu2, s2, w2) = params
    gmm = _g_make_gmm([mu1, mu2], [s1 ** 2, s2 ** 2], [w1, w2])
    roots = _g_boundary_roots(mu1, s1, w1, mu2, s2, w2, 0.5)
    inner = [r for r in roots if mu1 <= r <= mu2]
    mid = 0.5 * (mu1 + mu2)
    expected = inner[0] if inner else min(roots, key=lambda r: abs(r - mid))

    log_b, _ = gmm_boundaries_logspace(gmm, tau=0.5)

    assert log_b[0] == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("tau", [0.5, 0.7, 0.3])
def test_gmm_boundaries_logspace_matches_hand_crossings_three_components(tau):
    """
    For three unequal-variance, unequal-weight components stored out of
    order, every adjacent boundary must equal the crossing of the
    posterior-``tau`` equation that lies between the two sorted means.

    The expected boundaries are found by brentq on
    ``log(w1 N1) - log(w2 N2) - log(tau / (1 - tau))`` between each pair of
    sorted means, so the quadratic coefficients, the tau term (sign and
    scale), the adjacent-component indexing and the root selection are all
    pinned. Increasing ``tau`` must move every boundary toward its left
    component, as documented.
    """
    means = [2.5, -2.0, 0.0]
    sds = [0.6, 0.5, 1.0]
    weights = [0.3, 0.3, 0.4]
    gmm = _g_make_gmm(means, [s ** 2 for s in sds], weights)
    order = np.argsort(means)
    m = np.array(means)[order]
    s = np.array(sds)[order]
    w = np.array(weights)[order]
    expected = []
    for k in range(2):
        roots = [r for r in _g_boundary_roots(m[k], s[k], w[k], m[k + 1], s[k + 1], w[k + 1], tau)
                 if m[k] <= r <= m[k + 1]]
        assert len(roots) == 1
        expected.append(roots[0])

    log_b, sec_b = gmm_boundaries_logspace(gmm, tau=tau)

    assert log_b.shape == (2,)
    np.testing.assert_allclose(log_b, expected, atol=1e-9)
    np.testing.assert_allclose(sec_b, np.exp(expected), rtol=1e-9)
    if tau > 0.5:
        base, _ = gmm_boundaries_logspace(gmm, tau=0.5)
        assert np.all(log_b < base)


def test_gmm_boundaries_logspace_linear_branch_unequal_weights_and_tau():
    """
    With equal variances the boundary is linear; with unequal weights and a
    non-default ``tau`` it must equal the closed form
    ``(mu1 + mu2) / 2 + s^2 (log(w1 / w2) - log(tau / (1 - tau))) / (mu2 - mu1)``.

    A first pair sharing both mean and variance has no boundary at all (the
    linear equation degenerates) and must yield NaN without stopping the
    loop, so the second, well-defined boundary is still returned.
    """
    tau = 0.7
    gmm = _g_make_gmm([0.0, 0.0, 3.0], [1.0, 1.0, 1.0], [0.2, 0.3, 0.5])

    log_b, sec_b = gmm_boundaries_logspace(gmm, tau=tau)

    assert log_b.shape == (2,)
    assert np.isnan(log_b[0])
    expected = 1.5 + (np.log(0.3 / 0.5) - np.log(tau / (1.0 - tau))) / 3.0
    assert log_b[1] == pytest.approx(expected, abs=1e-12)
    assert sec_b[1] == pytest.approx(np.exp(expected), rel=1e-12)
    assert np.isnan(sec_b[0])


def test_gmm_boundaries_logspace_nan_pair_does_not_stop_later_pairs():
    """
    A pair with no real crossing (negative discriminant) must contribute NaN
    and the loop must continue to the next pair.

    The first pair (broad dominant component at 0, faint narrow one at 0.5)
    cannot cross at ``tau = 0.5``; the second pair (that faint component and
    a well-separated one at 5) crosses between its means at the brentq root
    of the hand-written posterior equation.
    """
    means = [0.0, 0.5, 5.0]
    sds = [1.0, 0.3, 0.5]
    weights = [0.72, 0.08, 0.2]
    gmm = _g_make_gmm(means, [s ** 2 for s in sds], weights)
    roots = [r for r in _g_boundary_roots(0.5, 0.3, 0.08, 5.0, 0.5, 0.2, 0.5) if 0.5 <= r <= 5.0]

    log_b, _ = gmm_boundaries_logspace(gmm)

    assert log_b.shape == (2,)
    assert np.isnan(log_b[0])
    assert log_b[1] == pytest.approx(roots[0], abs=1e-9)


def test_gmm_cdf_logspace_matches_weighted_normal_cdfs():
    """
    :func:`gmm_cdf_logspace` must equal ``sum_k w_k Phi((x - mu_k) / s_k)``
    elementwise, for float arrays, integer arrays and 0-d scalars.

    Integer evaluation points must still produce float CDF values (the
    accumulator is float regardless of the input dtype).
    """
    means, sds, weights = [-1.0, 0.5, 2.0], [0.5, 1.2, 0.3], [0.2, 0.5, 0.3]
    gmm = _g_make_gmm(means, [s ** 2 for s in sds], weights)
    x = np.array([-2.0, -0.7, 0.0, 0.9, 2.1, 4.0])
    expected = sum(w * stats.norm.cdf(x, m, s) for m, s, w in zip(means, sds, weights, strict=True))

    np.testing.assert_allclose(gmm_cdf_logspace(x, gmm), expected, rtol=1e-13)

    x_int = np.array([-2, 0, 1, 3])
    out_int = gmm_cdf_logspace(x_int, gmm)
    assert out_int.dtype == np.float64
    np.testing.assert_allclose(out_int, sum(w * stats.norm.cdf(x_int, m, s)
                                            for m, s, w in zip(means, sds, weights, strict=True)), rtol=1e-13)
    scalar = gmm_cdf_logspace(0.9, gmm)
    assert float(scalar) == pytest.approx(float(expected[3]), rel=1e-13)


def test_gmm_quantile_logspace_inverts_the_cdf_and_covers_wide_tails():
    """
    :func:`gmm_quantile_logspace` must invert the mixture CDF exactly,
    including extreme quantiles of a broad (sd = 3) component, and return
    float64 output even for float32 probabilities.

    For a single component the quantile is ``mu + s * Phi^-1(q)``; for a
    mixture the round trip through the analytic CDF must return ``q``. The
    extreme quantiles (1e-3, 0.999) of the broad component sit more than
    ``8 / sd`` but less than ``8 * sd`` from the mean, so the root bracket
    must scale with (not inversely to) the widest standard deviation.
    """
    single = _g_make_gmm([1.0], [9.0], [1.0])
    q = np.array([1e-3, 0.1, 0.5, 0.9, 0.999])
    np.testing.assert_allclose(gmm_quantile_logspace(q, single),
                               1.0 + 3.0 * stats.norm.ppf(q), atol=1e-8)

    means, sds, weights = [-1.0, 2.0], [0.4, 3.0], [0.6, 0.4]
    gmm = _g_make_gmm(means, [s ** 2 for s in sds], weights)
    out = gmm_quantile_logspace(q, gmm)
    back = sum(w * stats.norm.cdf(out, m, s) for m, s, w in zip(means, sds, weights, strict=True))
    np.testing.assert_allclose(back, q, atol=1e-10)
    assert np.all(np.diff(out) > 0)

    out32 = gmm_quantile_logspace(q.astype(np.float32), gmm)
    assert out32.dtype == np.float64
    np.testing.assert_allclose(out32, gmm_quantile_logspace(q.astype(np.float32).astype(float), gmm),
                               rtol=0, atol=1e-12)


@pytest.mark.parametrize("cov_type", ["tied", "diag", "spherical"])
def test_extract_params_expands_every_covariance_type(cov_type):
    """
    :func:`_extract_params` must expand tied, diagonal and spherical
    covariances into a (K, d, d) stack of full matrices and return their
    inverses as precisions.

    Each case is built by hand in 2D with K = 2: a tied matrix is repeated
    once per component, a diagonal vector becomes ``diag(v)`` and a
    spherical variance becomes ``v * I``. Weights and means pass through.
    """
    gmm = GaussianMixture(n_components=2, covariance_type=cov_type)
    gmm.weights_ = np.array([0.4, 0.6])
    gmm.means_ = np.array([[0.0, 1.0], [2.0, -1.0]])
    if cov_type == "tied":
        gmm.covariances_ = np.array([[2.0, 0.3], [0.3, 0.5]])
        expected = np.array([gmm.covariances_, gmm.covariances_])
    elif cov_type == "diag":
        gmm.covariances_ = np.array([[0.5, 2.0], [3.0, 0.25]])
        expected = np.array([np.diag([0.5, 2.0]), np.diag([3.0, 0.25])])
    else:
        gmm.covariances_ = np.array([0.5, 4.0])
        expected = np.array([0.5 * np.eye(2), 4.0 * np.eye(2)])

    w, means, sig, prec = _extract_params(gmm)

    np.testing.assert_array_equal(w, [0.4, 0.6])
    np.testing.assert_array_equal(means, gmm.means_)
    assert sig.shape == (2, 2, 2)
    np.testing.assert_allclose(sig, expected, rtol=1e-15)
    np.testing.assert_allclose(prec, np.linalg.inv(expected), rtol=1e-12)


def test_extract_params_rejects_unknown_covariance_type():
    """
    An unrecognised ``covariance_type`` must raise ``ValueError`` naming the
    problem rather than returning garbage matrices.
    """
    gmm = GaussianMixture(n_components=1)
    gmm.weights_ = np.array([1.0])
    gmm.means_ = np.array([[0.0]])
    gmm.covariances_ = np.array([[[1.0]]])
    gmm.covariance_type = "bogus"
    with pytest.raises(ValueError, match="Unsupported covariance type"):
        _extract_params(gmm)


def test_log_gauss_matches_scipy_multivariate_normal():
    """
    :func:`_log_gauss` must equal scipy's multivariate normal log-density for
    a correlated 2D Gaussian with a non-zero mean, at several points.
    """
    mu = np.array([1.0, -2.0])
    sig = np.array([[2.0, 0.6], [0.6, 0.8]])
    x = np.array([[0.0, 0.0], [1.0, -2.0], [3.0, -1.5], [-1.0, -3.0]])
    expected = stats.multivariate_normal(mean=mu, cov=sig).logpdf(x)
    np.testing.assert_allclose(_log_gauss(x, mu, sig, np.linalg.inv(sig)), expected, rtol=1e-12)


def test_gmm_modes_1d_converges_to_the_analytic_modes():
    """
    The 1D Newton-polished modes of an overlapping, unequal-variance mixture
    must coincide with the zeros of the analytic log-density gradient to
    1e-9, and there must be exactly one mode per true mode.

    The components overlap enough that each mode is pulled away from its
    component mean, so the responsibilities, gradient and curvature all
    enter the answer. Densities must equal the mixture density at the modes,
    sorted descending.
    """
    means, sds, weights = [0.0, 1.5], [0.4, 0.9], [0.6, 0.4]
    gmm = _g_make_gmm(means, [s ** 2 for s in sds], weights)
    truth = _g_true_modes_1d(means, sds, weights)
    assert truth.size == 2

    modes, dens = gmm_modes(gmm)

    assert modes.shape == (2, 1)
    np.testing.assert_allclose(np.sort(modes.flatten()), truth, atol=1e-9)
    np.testing.assert_allclose(dens, np.exp(_g_mix_logpdf(modes.flatten(), means, sds, weights)), rtol=1e-10)
    assert dens[0] > dens[1]


def test_gmm_modes_1d_single_newton_step_from_the_grid_peak():
    """
    With ``max_iter=1`` each 1D mode must be exactly one Newton step
    ``x0 - g(x0) / h(x0)`` from its grid peak ``x0``, with the gradient and
    curvature of the log-density computed analytically.

    The grid is the documented ``[mu_min - 4 sd_max, mu_max + 4 sd_max]``
    linspace; the peak start points are its strict interior maxima. This pins
    the grid span, the peak indexing, and every term of the gradient and
    curvature (an error anywhere moves the one-step result well beyond the
    1e-10 tolerance).
    """
    means, sds, weights = [0.0, 1.5], [0.4, 0.9], [0.6, 0.4]
    gmm = _g_make_gmm(means, [s ** 2 for s in sds], weights)
    starts = _g_grid_peaks(means, sds, weights, 4096)
    assert starts.size == 2
    expected = []
    for x0 in starts:
        g, h = _g_mix_derivatives(x0, means, sds, weights)
        assert h < 0.0
        expected.append(x0 - g / h)

    modes, _ = gmm_modes(gmm, max_iter=1)

    np.testing.assert_allclose(np.sort(modes.flatten()), np.sort(expected), rtol=0, atol=1e-10)


def test_gmm_modes_1d_gradient_fallback_step_is_grad_times_1e_3():
    """
    When the grid peak sits where the log-density curvature is positive, the
    1D polish must take the gradient-ascent fallback step ``1e-3 * g(x0)``
    instead of a Newton step.

    Two narrow, slightly unequal components at +/-0.4 with a 3-point grid
    make the middle grid point (0) a grid maximum lying in the convex valley
    between the peaks. With ``max_iter=1`` the single returned "mode" must
    be exactly ``0 + 1e-3 * g(0)``.
    """
    means, sds, weights = [-0.4, 0.4], [0.2, 0.2], [0.45, 0.55]
    gmm = _g_make_gmm(means, [s ** 2 for s in sds], weights)
    starts = _g_grid_peaks(means, sds, weights, 3)
    np.testing.assert_allclose(starts, [0.0], atol=1e-15)
    g, h = _g_mix_derivatives(0.0, means, sds, weights)
    assert h > 0.0

    modes, dens = gmm_modes(gmm, max_iter=1, grid_size=3)

    assert modes.shape == (1, 1)
    assert modes[0, 0] == pytest.approx(1e-3 * g, rel=1e-10)
    assert dens.shape == (1,)


def test_gmm_modes_1d_coarse_grid_without_interior_peak_returns_empty():
    """
    If the (user-sized) grid contains no strict interior maximum, the 1D
    branch must return an empty (0, 1) mode array and an empty density array.

    Two far-apart components with ``grid_size=3`` leave only the two edge
    points and a valley point in the grid, so no peak is detected.
    """
    gmm = _g_make_gmm([0.0, 10.0], [1.0, 1.0], [0.5, 0.5])

    modes, dens = gmm_modes(gmm, grid_size=3)

    assert modes.shape == (0, 1)
    assert dens.shape == (0,)


def test_gmm_modes_1d_tolerance_controls_the_newton_stop():
    """
    The 1D polish must stop as soon as a step is smaller than
    ``tol * (1 + |x|)`` (``x`` = the point before the step), and not before.

    For a unimodal two-component mixture peaking near 1.43, ``tol`` is set
    just above (respectively below) the relative size of the first Newton
    step, with ``max_iter=2``. Above, the result is the one-step point;
    below, it is the two-step point. Both are computed with the analytic
    derivatives.
    """
    means, sds, weights = [1.2, 2.0], [0.5, 0.6], [0.5, 0.5]
    gmm = _g_make_gmm(means, [s ** 2 for s in sds], weights)
    starts = _g_grid_peaks(means, sds, weights, 4096)
    assert starts.size == 1
    x0 = starts[0]
    g0, h0 = _g_mix_derivatives(x0, means, sds, weights)
    x1 = x0 - g0 / h0
    g1, h1 = _g_mix_derivatives(x1, means, sds, weights)
    x2 = x1 - g1 / h1
    rel_step = abs(x1 - x0) / (1.0 + abs(x0))
    assert abs(x2 - x1) > 1e-14

    stopped, _ = gmm_modes(gmm, max_iter=2, tol=1.2 * rel_step)
    continued, _ = gmm_modes(gmm, max_iter=2, tol=0.8 * rel_step)

    assert stopped[0, 0] == pytest.approx(x1, abs=1e-13)
    assert continued[0, 0] == pytest.approx(x2, abs=1e-13)
    assert abs(stopped[0, 0] - continued[0, 0]) > 1e-14


def _g_cp_step(x, means, covs, weights):
    """
    One Carreira-Perpinan fixed-point update for a Gaussian mixture, from
    scipy densities: ``x_new = (sum a_k P_k)^-1 sum a_k P_k mu_k`` with
    ``a_k = w_k N(x | mu_k, S_k)`` and ``P_k = S_k^-1``.

    Parameters
    ----------
    x (np.ndarray)
        Current point, shape (d,).
    means (np.ndarray)
        Component means, shape (K, d).
    covs (np.ndarray)
        Component covariances, shape (K, d, d).
    weights (np.ndarray)
        Component weights, shape (K,).

    Returns
    -------
    x_new (np.ndarray)
        Updated point, shape (d,).
    """
    a_mat = np.zeros((x.size, x.size))
    b_vec = np.zeros(x.size)
    for m, c, w in zip(means, covs, weights, strict=True):
        a = w * stats.multivariate_normal(mean=m, cov=c).pdf(x)
        p = np.linalg.inv(c)
        a_mat += a * p
        b_vec += a * p @ m
    return np.linalg.solve(a_mat, b_vec)


def test_gmm_modes_general_d_keeps_a_minor_mode_next_to_a_major_one():
    """
    The general-d local-maximum check must probe only a tiny neighbourhood,
    so a genuine minor mode one unit from a heavier mode is kept.

    Two isotropic components (sd 0.3) at (0, 0) and (1, 0) with weights
    0.3 / 0.7 give two modes; a unit-sized probe from the minor mode would
    land on the major one and wrongly reject it. Both returned modes must
    match the fixed points of the independent Carreira-Perpinan iteration,
    sorted by descending density.
    """
    means = np.array([[0.0, 0.0], [1.0, 0.0]])
    covs = np.array([0.09 * np.eye(2), 0.09 * np.eye(2)])
    weights = np.array([0.3, 0.7])
    gmm = _g_make_gmm_2d(means, covs, weights)
    truth = []
    for start in means:
        x = start.copy()
        for _ in range(500):
            x = _g_cp_step(x, means, covs, weights)
        truth.append(x)

    modes, dens = gmm_modes(gmm)

    assert modes.shape == (2, 2)
    np.testing.assert_allclose(modes[0], truth[1], atol=1e-6)
    np.testing.assert_allclose(modes[1], truth[0], atol=1e-6)
    assert dens[0] > dens[1]


def test_gmm_modes_general_d_merges_seeds_that_reach_the_same_mode():
    """
    Seeds converging to the same mode must be deduplicated into one mode.

    A single 2D Gaussian at (2, 2) is its own fixed point, so two nearby
    seeds both land exactly on it; the result must be one mode at (2, 2).
    """
    gmm = _g_make_gmm_2d([[2.0, 2.0]], [[[0.5, 0.1], [0.1, 0.4]]], [1.0])

    modes, dens = gmm_modes(gmm, seeds=np.array([[2.0, 2.0], [2.3, 1.9]]))

    assert modes.shape == (1, 2)
    np.testing.assert_allclose(modes[0], [2.0, 2.0], atol=1e-12)
    assert dens.shape == (1,)


def test_gmm_modes_general_d_tolerance_controls_the_fixed_point_stop():
    """
    The general-d fixed-point iteration must stop as soon as
    ``||x_new - x|| < tol * (1 + ||x||)`` and not before.

    Starting a few 1e-5 away from the mode of an overlapping two-component
    2D mixture (mode norm > 1), ``tol`` is set just above / just below the
    relative first step with ``max_iter=2``. The returned mode must equal the
    one-step / two-step point of the independent Carreira-Perpinan update.
    """
    means = np.array([[1.2, 1.0], [2.0, 1.4]])
    covs = np.array([[[0.25, 0.05], [0.05, 0.3]], [[0.36, 0.0], [0.0, 0.2]]])
    weights = np.array([0.5, 0.5])
    gmm = _g_make_gmm_2d(means, covs, weights)
    mode = means[0].copy()
    for _ in range(2000):
        mode = _g_cp_step(mode, means, covs, weights)
    x0 = mode + np.array([3e-5, -2e-5])
    x1 = _g_cp_step(x0, means, covs, weights)
    x2 = _g_cp_step(x1, means, covs, weights)
    rel_step = np.linalg.norm(x1 - x0) / (1.0 + np.linalg.norm(x0))
    assert np.linalg.norm(x2 - x1) > 1e-9

    stopped, _ = gmm_modes(gmm, seeds=x0[None, :], max_iter=2, tol=1.2 * rel_step)
    continued, _ = gmm_modes(gmm, seeds=x0[None, :], max_iter=2, tol=0.8 * rel_step)

    np.testing.assert_allclose(stopped[0], x1, atol=1e-12)
    np.testing.assert_allclose(continued[0], x2, atol=1e-12)


def test_report_gmm_stats_orders_components_and_modes():
    """
    :func:`report_gmm_stats` must return component means / sds permuted by
    the given order, and the mixture modes sorted ascending with their own
    densities.

    The components are stored out of mean order; the modes of a bimodal and
    of a unimodal two-component model are checked against the analytic modes
    and mixture densities.
    """
    means, sds, weights = [1.5, 0.0], [0.9, 0.4], [0.4, 0.6]
    gmm = _g_make_gmm(means, [s ** 2 for s in sds], weights)
    order = np.array([1, 0])

    out_means, out_sds, modes, dens = report_gmm_stats(gmm, order)

    np.testing.assert_array_equal(out_means, [0.0, 1.5])
    np.testing.assert_allclose(out_sds, [0.4, 0.9], rtol=1e-15)
    np.testing.assert_allclose(modes, _g_true_modes_1d(means, sds, weights), atol=1e-9)
    np.testing.assert_allclose(dens, np.exp(_g_mix_logpdf(modes, means, sds, weights)), rtol=1e-10)

    u_means, u_sds, u_weights = [1.2, 2.0], [0.5, 0.6], [0.5, 0.5]
    unimodal = _g_make_gmm(u_means, [s ** 2 for s in u_sds], u_weights)
    u_truth = _g_true_modes_1d(u_means, u_sds, u_weights)
    assert u_truth.size == 1
    _, _, u_modes, u_dens = report_gmm_stats(unimodal, np.array([0, 1]))
    np.testing.assert_allclose(u_modes, u_truth, atol=1e-9)
    assert u_dens[0] == pytest.approx(float(np.exp(_g_mix_logpdf(u_truth, u_means, u_sds, u_weights))[0]), rel=1e-10)


def test_report_gmm_stats_handles_an_empty_mode_set(monkeypatch):
    """
    If the mode finder returns no modes, :func:`report_gmm_stats` must still
    return the component means / sds and two empty (0,) arrays.
    """
    gmm = _g_make_gmm([0.0, 1.0], [1.0, 1.0], [0.5, 0.5])
    monkeypatch.setattr(mixture_model_utils, "gmm_modes",
                        lambda _model: (np.empty((0, 1)), np.empty((0,))))

    out_means, out_sds, modes, dens = report_gmm_stats(gmm, np.array([0, 1]))

    np.testing.assert_array_equal(out_means, [0.0, 1.0])
    np.testing.assert_array_equal(out_sds, [1.0, 1.0])
    assert modes.shape == (0,)
    assert dens.shape == (0,)


def test_summarize_best_gmm_reports_every_field_by_hand():
    """
    :func:`summarize_best_gmm` must bundle ordered log-means / sds, medians
    ``exp(mu)``, log-normal means ``exp(mu + s^2 / 2)``, ordered weights,
    ascending modes (log and seconds) with their densities, and the
    boundaries at the requested ``tau``.

    The components are stored out of order so that the ordering argument
    matters, and a non-default ``tau`` is used so the boundaries must be
    computed at that threshold.
    """
    means, sds, weights = [2.5, -2.0, 0.0], [0.6, 0.5, 1.0], [0.3, 0.3, 0.4]
    gmm = _g_make_gmm(means, [s ** 2 for s in sds], weights)
    order = np.argsort(means)
    expected_b, expected_b_sec = gmm_boundaries_logspace(gmm, tau=0.7)

    summary = summarize_best_gmm(gmm, order, tau=0.7)

    mu = np.array(means)[order]
    s = np.array(sds)[order]
    np.testing.assert_array_equal(summary["logmeans"], mu)
    np.testing.assert_allclose(summary["logsds"], s, rtol=1e-15)
    np.testing.assert_allclose(summary["medians_sec"], np.exp(mu), rtol=1e-15)
    np.testing.assert_allclose(summary["means_sec"], np.exp(mu + 0.5 * s ** 2), rtol=1e-14)
    np.testing.assert_array_equal(summary["weights"], np.array(weights)[order])
    truth = _g_true_modes_1d(means, sds, weights)
    np.testing.assert_allclose(summary["modes_log"], truth, atol=1e-9)
    np.testing.assert_allclose(summary["modes_sec"], np.exp(summary["modes_log"]), rtol=1e-15)
    np.testing.assert_allclose(summary["mode_densities"],
                               np.exp(_g_mix_logpdf(summary["modes_log"], means, sds, weights)), rtol=1e-10)
    np.testing.assert_array_equal(summary["boundaries_log"], expected_b)
    np.testing.assert_array_equal(summary["boundaries_sec"], expected_b_sec)
    assert summary["tau"] == 0.7


def test_gmm_icl_equals_bic_plus_twice_the_assignment_entropy():
    """
    :func:`gmm_icl` must equal the hand-computed BIC
    (``-2 * loglik + (3K - 1) * log N``) plus twice the soft-assignment
    entropy, with posteriors computed from scipy densities.

    A second case with far-apart components makes some posteriors exactly
    zero; their contribution must be zero (not NaN), so ICL equals BIC there.
    """
    means, sds, weights = [0.0, 1.0], [0.8, 0.6], [0.55, 0.45]
    gmm = _g_make_gmm(means, [s ** 2 for s in sds], weights)
    log_x = np.linspace(-1.5, 2.5, 25).reshape(-1, 1)
    comp = np.array([w * stats.norm.pdf(log_x[:, 0], m, s) for m, s, w in zip(means, sds, weights, strict=True)])
    loglik = float(np.sum(np.log(comp.sum(axis=0))))
    bic = -2.0 * loglik + 5 * np.log(25)
    z = comp / comp.sum(axis=0)
    entropy = float(-np.sum(z * np.log(z)))
    assert entropy > 1.0

    assert gmm_icl(gmm, log_x) == pytest.approx(bic + 2.0 * entropy, rel=1e-10)

    far = _g_make_gmm([0.0, 100.0], [1.0, 1.0], [0.5, 0.5])
    pts = np.array([[-0.5], [0.0], [0.5], [99.5], [100.0], [100.5]])
    assert np.any(far.predict_proba(pts) == 0.0)
    assert gmm_icl(far, pts) == pytest.approx(float(far.bic(pts)), abs=1e-9)


def _g_spy_factory(target, recorded):
    """
    Wraps a function so every invocation's positional and keyword arguments
    are recorded before the real function runs.

    Parameters
    ----------
    target (function)
        The real function (class or function).
    recorded (list)
        List receiving one ``(args, kwargs)`` tuple per invocation.

    Returns
    -------
    spy (function)
        Recording wrapper with the same return value as ``target``.
    """
    def spy(*args, **kwargs):
        """
        Records the arguments and forwards them to the wrapped function.

        Parameters
        ----------
        *args
            Positional arguments.
        **kwargs
            Keyword arguments.

        Returns
        -------
        result
            Whatever the wrapped function returns.
        """
        recorded.append((args, kwargs))
        return target(*args, **kwargs)
    return spy


def test_gmm_cv_neg_loglik_matches_a_hand_built_shuffled_kfold(monkeypatch):
    """
    :func:`gmm_cv_neg_loglik` must equal ``-2 *`` the summed held-out
    log-likelihood over a shuffled, seeded KFold, with each fold fitted by a
    full-covariance GaussianMixture carrying the supplied arguments.

    The constructor is spied to pin every forwarded argument; the expected
    total is rebuilt fold by fold with sklearn directly.
    """
    rng = np.random.default_rng(4)
    x = np.exp(np.concatenate([rng.normal(-2.0, 0.3, 40), rng.normal(0.0, 0.4, 40)]))
    log_x = np.log(x).reshape(-1, 1)
    expected = 0.0
    for train_idx, test_idx in KFold(n_splits=4, shuffle=True, random_state=7).split(log_x):
        ref = GaussianMixture(n_components=2, covariance_type="full", random_state=7,
                              n_init=2, reg_covar=1e-3).fit(log_x[train_idx])
        expected += float(np.sum(ref.score_samples(log_x[test_idx])))
    recorded = []
    monkeypatch.setattr(mixture_model_utils, "GaussianMixture", _g_spy_factory(GaussianMixture, recorded))

    out = gmm_cv_neg_loglik(x, n_components=2, seed=7, n_folds=4, n_init=2, reg_covar=1e-3)

    assert out == pytest.approx(-2.0 * expected, rel=1e-10)
    assert len(recorded) == 4
    for args, kwargs in recorded:
        assert args == ()
        assert kwargs == {"n_components": 2, "covariance_type": "full", "random_state": 7,
                          "n_init": 2, "reg_covar": 1e-3}


def test_gmm_cv_neg_loglik_sample_count_guards():
    """
    The CV helper must return ``inf`` only when there are fewer samples than
    folds or fewer training samples than components, and a finite value at
    the boundaries (samples == folds; training size == components).
    """
    x = np.array([0.1, 0.2, 0.3, 0.5, 0.8, 1.3])
    assert gmm_cv_neg_loglik(x[:3], n_components=1, n_folds=4) == float("inf")
    assert np.isfinite(gmm_cv_neg_loglik(x[:3], n_components=1, n_folds=3, n_init=1))
    assert gmm_cv_neg_loglik(x, n_components=5, n_folds=3, n_init=1) == float("inf")
    assert np.isfinite(gmm_cv_neg_loglik(x, n_components=4, n_folds=3, n_init=1))


def test_ig_logpdf_and_mode_match_scipy_invgauss():
    """
    :func:`_ig_logpdf_linear` must equal scipy's ``invgauss`` log-density
    (shape ``mu / lam``, scale ``lam``), and :func:`_ig_mode` must equal the
    numerical argmax of that density.
    """
    x = np.array([0.01, 0.1, 0.5, 1.0, 3.0])
    for mu, lam in [(0.5, 2.0), (2.0, 0.7), (0.07, 1.0)]:
        np.testing.assert_allclose(_ig_logpdf_linear(x, mu, lam),
                                   stats.invgauss.logpdf(x, mu / lam, scale=lam), rtol=1e-12)
        grid = np.linspace(1e-4, 3.0 * mu, 200001)
        pdf = stats.invgauss.pdf(grid, mu / lam, scale=lam)
        assert _ig_mode(mu, lam) == pytest.approx(grid[np.argmax(pdf)], abs=3.0 * mu / 200000)


def test_ig_mixture_constructor_attributes():
    """
    The IG mixture constructor must expose ``means_`` as the log of each
    component's mode (K, 1), ``covariances_`` as the linear-time variances
    ``mu^3 / lam`` (K, 1, 1), and the full-covariance tag used by shared
    plotting code.
    """
    model = IGMixture(weights=[0.3, 0.7], mus=[0.08, 1.5], lambdas=[0.6, 2.5])
    assert model.covariance_type == "full"
    assert model.n_components == 2
    assert model.covariances_.shape == (2, 1, 1)
    np.testing.assert_allclose(model.covariances_.ravel(), [0.08 ** 3 / 0.6, 1.5 ** 3 / 2.5], rtol=1e-14)
    np.testing.assert_allclose(model.means_.ravel(),
                               np.log([_ig_mode(0.08, 0.6), _ig_mode(1.5, 2.5)]), rtol=1e-14)


def test_ig_mixture_score_bic_aic_by_hand():
    """
    ``score``, ``bic`` and ``aic`` of an IG mixture must match the
    log-axis log-likelihood built from scipy's ``invgauss`` plus the
    Jacobian ``log x``: mean, ``-2 LL + (3K - 1) log N`` and
    ``-2 LL + 2 (3K - 1)``.
    """
    model = IGMixture(weights=[0.3, 0.7], mus=[0.08, 1.5], lambdas=[0.6, 2.5])
    x = np.array([0.02, 0.05, 0.1, 0.4, 1.0, 2.0, 5.0])
    per = np.log(0.3 * stats.invgauss.pdf(x, 0.08 / 0.6, scale=0.6)
                 + 0.7 * stats.invgauss.pdf(x, 1.5 / 2.5, scale=2.5)) + np.log(x)
    log_x = np.log(x)
    assert model.score(log_x) == pytest.approx(float(np.mean(per)), rel=1e-12)
    assert model.bic(log_x) == pytest.approx(-2.0 * per.sum() + 5 * np.log(7), rel=1e-12)
    assert model.aic(log_x) == pytest.approx(-2.0 * per.sum() + 10.0, rel=1e-12)


def test_ig_mixture_icl_by_hand_and_with_exact_zero_posteriors():
    """
    :func:`ig_mixture_icl` must equal the IG BIC plus twice the hand-built
    assignment entropy; posteriors that underflow to exactly zero must add
    nothing (not NaN).
    """
    model = IGMixture(weights=[0.4, 0.6], mus=[0.3, 0.8], lambdas=[1.0, 2.0])
    x = np.array([0.1, 0.2, 0.4, 0.6, 1.0, 1.8])
    comp = np.array([0.4 * stats.invgauss.pdf(x, 0.3, scale=1.0),
                     0.6 * stats.invgauss.pdf(x, 0.4, scale=2.0)])
    z = comp / comp.sum(axis=0)
    entropy = float(-np.sum(z * np.log(z)))
    assert entropy > 1.0
    assert ig_mixture_icl(model, np.log(x)) == pytest.approx(model.bic(np.log(x)) + 2.0 * entropy, rel=1e-10)

    far = IGMixture(weights=[0.5, 0.5], mus=[0.01, 100.0], lambdas=[1.0, 1000.0])
    pts = np.log(np.array([0.008, 0.01, 0.012, 90.0, 100.0, 110.0]))
    assert np.any(far.predict_proba(pts) == 0.0)
    assert ig_mixture_icl(far, pts) == pytest.approx(far.bic(pts), abs=1e-9)


def test_ig_mixture_cdf_matches_scipy_and_quantile_inverts_it():
    """
    :func:`ig_mixture_cdf_logspace` must equal the weighted scipy
    ``invgauss`` CDFs at log-space points (array and scalar input), and
    :func:`ig_mixture_quantile_logspace` must return the log of the
    brentq-inverted scipy mixture CDF.
    """
    model = IGMixture(weights=[0.3, 0.7], mus=[0.08, 1.5], lambdas=[0.6, 2.5])
    t = np.array([0.01, 0.05, 0.08, 0.2, 1.0, 1.5, 4.0, 10.0])

    def ref_cdf(v):
        """
        Reference mixture CDF in linear time from scipy.

        Parameters
        ----------
        v (np.ndarray or float)
            Linear-time points.

        Returns
        -------
        cdf (np.ndarray or float)
            Mixture CDF.
        """
        return (0.3 * stats.invgauss.cdf(v, 0.08 / 0.6, scale=0.6)
                + 0.7 * stats.invgauss.cdf(v, 1.5 / 2.5, scale=2.5))

    np.testing.assert_allclose(ig_mixture_cdf_logspace(np.log(t), model), ref_cdf(t), rtol=1e-10, atol=1e-14)
    scalar = ig_mixture_cdf_logspace(np.log(0.2), model)
    assert scalar.shape == (1,)
    assert scalar[0] == pytest.approx(ref_cdf(0.2), rel=1e-10)

    qs = np.array([0.01, 0.3, 0.6, 0.99])
    expected = [np.log(brentq(lambda v, q=q: ref_cdf(v) - q, 1e-6, 100.0, xtol=1e-14)) for q in qs]
    np.testing.assert_allclose(ig_mixture_quantile_logspace(qs, model), expected, atol=1e-8)


def test_report_and_summarize_ig_mixture_by_hand():
    """
    :func:`report_ig_mixture_stats` and :func:`summarize_best_ig_mixture`
    must permute every per-component quantity by ``order`` and report
    ``log mu``, ``lam``, weights, log modes and the mixture log-axis density
    at each mode; the summary's ``logsds`` is ``sqrt(log1p(mu / lam))``,
    ``medians_sec`` carries ``mu`` and both boundary arrays are empty.
    """
    mus = np.array([1.5, 0.08])
    lams = np.array([2.5, 0.6])
    weights = np.array([0.7, 0.3])
    model = IGMixture(weights=weights, mus=mus, lambdas=lams)
    order = np.array([1, 0])
    modes_log = np.log([_ig_mode(0.08, 0.6), _ig_mode(1.5, 2.5)])
    t = np.exp(modes_log)
    dens = (0.3 * stats.invgauss.pdf(t, 0.08 / 0.6, scale=0.6)
            + 0.7 * stats.invgauss.pdf(t, 1.5 / 2.5, scale=2.5)) * t

    log_mus, out_lams, out_w, out_modes, out_dens = report_ig_mixture_stats(model, order)

    np.testing.assert_allclose(log_mus, np.log([0.08, 1.5]), rtol=1e-15)
    np.testing.assert_array_equal(out_lams, [0.6, 2.5])
    np.testing.assert_array_equal(out_w, [0.3, 0.7])
    np.testing.assert_allclose(out_modes, modes_log, rtol=1e-14)
    np.testing.assert_allclose(out_dens, dens, rtol=1e-10)

    summary = summarize_best_ig_mixture(model, order)

    np.testing.assert_allclose(summary["logmeans"], np.log([0.08, 1.5]), rtol=1e-15)
    np.testing.assert_allclose(summary["logsds"], np.sqrt(np.log1p(np.array([0.08 / 0.6, 1.5 / 2.5]))), rtol=1e-14)
    np.testing.assert_array_equal(summary["weights"], [0.3, 0.7])
    np.testing.assert_array_equal(summary["medians_sec"], [0.08, 1.5])
    np.testing.assert_allclose(summary["modes_log"], modes_log, rtol=1e-14)
    np.testing.assert_allclose(summary["densities"], dens, rtol=1e-10)
    assert summary["boundaries_log"].shape == (0,)
    assert summary["boundaries_sec"].shape == (0,)


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_fit_log_ig_mixture_constant_data_initialisation_and_one_em_step():
    """
    On constant data the IG initialisation is fully determined: KMeans finds
    one cluster, the empty second cluster falls back to all data, both
    shapes hit the ``1 / reg_covar`` ceiling, and the empty cluster's weight
    is floored at 1e-3 before renormalisation.

    With ``max_iter=0`` the returned model is the initialisation itself; with
    ``max_iter=1`` one EM step must leave it unchanged (identical component
    densities make responsibilities equal the weights, the residual ``1/x -
    1/mu`` is zero so the shape floor applies, and a component with total
    responsibility below one must not be inflated).
    """
    x = np.full(50, 0.5)
    w_expected = np.array([1.0, 1e-3]) / 1.001
    for max_iter in (0, 1):
        model, _ = fit_log_ig_mixture(x, 2, seed=0, n_init=1, max_iter=max_iter)
        np.testing.assert_allclose(model.weights_, w_expected, rtol=1e-12)
        np.testing.assert_allclose(model.mus_, [0.5, 0.5], rtol=1e-12)
        np.testing.assert_allclose(model.lambdas_, [1e4, 1e4], rtol=1e-12)


def test_fit_log_ig_mixture_kmeans_initialisation_is_moment_matched(monkeypatch):
    """
    With one restart and no EM iterations the fitted model must be exactly
    the documented KMeans initialisation: KMeans (``n_clusters=K``,
    ``random_state=seed``, ``n_init=10``) on ``log x``, then per-cluster
    ``mu = mean``, ``lam = mean^3 / var`` and label-frequency weights.

    KMeans is spied to pin its arguments and the single invocation.
    """
    x = _g_ig_three_cluster_intervals()
    km = KMeans(n_clusters=3, random_state=0, n_init=10).fit(np.log(x).reshape(-1, 1))
    w, mu, lam = _g_ig_init_from_labels(x, km.labels_, 3)
    recorded = []
    monkeypatch.setattr(mixture_model_utils, "KMeans", _g_spy_factory(KMeans, recorded))

    model, order = fit_log_ig_mixture(x, 3, seed=0, n_init=1, max_iter=0)

    assert len(recorded) == 1
    assert recorded[0][1] == {"n_clusters": 3, "random_state": 0, "n_init": 10}
    np.testing.assert_allclose(model.weights_, w, rtol=1e-12)
    np.testing.assert_allclose(model.mus_, mu, rtol=1e-12)
    np.testing.assert_allclose(model.lambdas_, lam, rtol=1e-12)
    np.testing.assert_array_equal(order, np.argsort(mu))


@pytest.mark.parametrize(("n_components", "seed"), [(2, 1), (2, 3), (3, 7)])
def test_fit_log_ig_mixture_returns_the_better_scoring_restart(monkeypatch, n_components, seed):
    """
    Restarts after the first must draw ``K`` sorted quantile levels from
    ``default_rng((seed, ii)).uniform(0.02, 0.98)``, assign each point to
    the nearest data quantile in log space, and the restart with the higher
    final log-likelihood must be returned.

    Both ``max_iter=0`` restarts are rebuilt by hand and scored with scipy
    log-likelihoods; the returned model must be exactly the better one. For
    (K=2, seed=1) and (K=2, seed=3) the quantile restart wins, for (K=3,
    seed=7) the KMeans restart wins (and a ranking on any other quantity
    than the total log-likelihood picks the wrong one). The generator seeds
    are spied.
    """
    x = _g_ig_three_cluster_intervals()
    lx = np.log(x)
    km = KMeans(n_clusters=n_components, random_state=seed, n_init=10).fit(lx.reshape(-1, 1))
    init_km = _g_ig_init_from_labels(x, km.labels_, n_components)
    qs = np.sort(np.random.default_rng((seed, 1)).uniform(0.02, 0.98, size=n_components))
    centers = np.quantile(lx, qs)
    labels = np.argmin(np.abs(lx[:, None] - centers[None, :]), axis=1)
    init_q = _g_ig_init_from_labels(x, labels, n_components)
    ll_q = _g_ig_linear_loglik(x, *init_q)
    ll_km = _g_ig_linear_loglik(x, *init_km)
    assert abs(ll_q - ll_km) > 1.0
    best = init_q if ll_q > ll_km else init_km
    seeds_seen = []
    real_rng = np.random.default_rng

    def rng_spy(seed=None):
        """
        Records the seed handed to ``default_rng`` and returns the real
        generator.

        Parameters
        ----------
        seed
            Seed argument.

        Returns
        -------
        rng (np.random.Generator)
            The real generator for that seed.
        """
        seeds_seen.append(seed)
        return real_rng(seed)

    monkeypatch.setattr(np.random, "default_rng", rng_spy)

    model, _ = fit_log_ig_mixture(x, n_components, seed=seed, n_init=2, max_iter=0)

    assert seeds_seen == [(seed, 1)]
    np.testing.assert_allclose(model.weights_, best[0], rtol=1e-12)
    np.testing.assert_allclose(model.mus_, best[1], rtol=1e-12)
    np.testing.assert_allclose(model.lambdas_, best[2], rtol=1e-12)


def test_fit_log_ig_mixture_converges_to_an_em_fixed_point():
    """
    A converged IG fit must satisfy the closed-form M-step equations at its
    own parameters: ``mu_k = sum z x / n_k``, ``1/lam_k = sum z (1/x) / n_k
    - 1/mu_k`` and ``w_k = n_k / N``, with responsibilities ``z`` computed
    from scipy's ``invgauss``; it must also recover the generating
    components of a well-separated sample.
    """
    x = _g_ig_intervals()
    model, order = fit_log_ig_mixture(x, 2, seed=0, n_init=1, tol=1e-12, max_iter=3000)
    w, mu, lam = model.weights_, model.mus_, model.lambdas_
    comp = np.array([w[k] * stats.invgauss.pdf(x, mu[k] / lam[k], scale=lam[k]) for k in range(2)])
    z = comp / comp.sum(axis=0)
    n_k = z.sum(axis=1)
    np.testing.assert_allclose(mu, (z * x).sum(axis=1) / n_k, rtol=1e-6)
    np.testing.assert_allclose(1.0 / lam, (z / x).sum(axis=1) / n_k - 1.0 / mu, rtol=1e-6)
    np.testing.assert_allclose(w, n_k / x.size, rtol=1e-6)
    assert mu[order][0] == pytest.approx(0.07, abs=0.01)
    assert mu[order][1] == pytest.approx(1.5, abs=0.3)
    assert w[order][0] == pytest.approx(0.5, abs=0.05)


def test_fit_log_ig_mixture_stops_when_the_loglik_change_is_below_tol():
    """
    The EM loop must stop right after the first iteration whose
    log-likelihood change ``|ll - prev_ll|`` falls below ``tol``.

    ``tol`` is chosen between the first log-likelihood change and the
    magnitude of the log-likelihood itself, so the fit must equal a fit
    capped at exactly two EM iterations.
    """
    x = _g_ig_intervals()
    jac = float(np.sum(np.log(x)))
    m0, _ = fit_log_ig_mixture(x, 2, seed=0, n_init=1, max_iter=0)
    m1, _ = fit_log_ig_mixture(x, 2, seed=0, n_init=1, max_iter=1)
    ll0 = float(np.sum(m0.score_samples(np.log(x)))) - jac
    ll1 = float(np.sum(m1.score_samples(np.log(x)))) - jac
    tol = 2.0 * abs(ll1 - ll0)
    assert tol < abs(ll1 + ll0)

    stopped, _ = fit_log_ig_mixture(x, 2, seed=0, n_init=1, max_iter=50, tol=tol)
    two, _ = fit_log_ig_mixture(x, 2, seed=0, n_init=1, max_iter=2)

    np.testing.assert_array_equal(stopped.mus_, two.mus_)
    np.testing.assert_array_equal(stopped.lambdas_, two.lambdas_)
    np.testing.assert_array_equal(stopped.weights_, two.weights_)


def test_fit_log_ig_mixture_without_restarts_raises():
    """
    With no restarts there is no fit to return, so the fitter must raise its
    documented ``RuntimeError``.
    """
    with pytest.raises(RuntimeError, match="EM did not produce a valid fit"):
        fit_log_ig_mixture(_g_ig_intervals(n_per=20), 2, n_init=0)


def test_ig_mixture_cv_neg_loglik_matches_a_hand_built_shuffled_kfold(monkeypatch):
    """
    :func:`ig_mixture_cv_neg_loglik` must fit each shuffled, seeded KFold
    training split with ``seed + fold_idx`` and the supplied ``n_init`` /
    ``reg_covar``, and return ``-2 *`` the summed held-out log-axis
    log-likelihood.

    The fitter is spied to pin the training data and arguments per fold; the
    expected total is rebuilt with direct fits.
    """
    x = _g_ig_intervals(n_per=60)
    splits = list(KFold(n_splits=3, shuffle=True, random_state=5).split(x))
    expected = 0.0
    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        ref, _ = fit_log_ig_mixture(x[train_idx], n_components=2, seed=5 + fold_idx, n_init=1, reg_covar=1e-3)
        expected += float(np.sum(ref.score_samples(np.log(x[test_idx]))))
    recorded = []
    monkeypatch.setattr(mixture_model_utils, "fit_log_ig_mixture", _g_spy_factory(fit_log_ig_mixture, recorded))

    out = ig_mixture_cv_neg_loglik(x, n_components=2, seed=5, n_folds=3, n_init=1, reg_covar=1e-3)

    assert out == pytest.approx(-2.0 * expected, rel=1e-10)
    assert len(recorded) == 3
    for fold_idx, ((args, kwargs), (train_idx, _)) in enumerate(zip(recorded, splits, strict=True)):
        np.testing.assert_array_equal(args[0], x[train_idx])
        assert kwargs == {"n_components": 2, "seed": 5 + fold_idx, "n_init": 1, "reg_covar": 1e-3}


_C_MISSING = "missing"
_C_GOLDEN = 0x9E3779B97F4A7C15
_C_MASK64 = (1 << 64) - 1


class _CFakeModel:
    """
    A minimal fitted-mixture stand-in whose log-likelihood is set by hand.

    It carries the attributes :func:`_sample_from_mixture` reads on its
    Gaussian branch (``n_components``, ``weights_``, ``means_``,
    ``covariances_``) plus ``nus_`` for the Student-t initialisation, and a
    ``score_samples`` that returns ``level + slope * x`` per point. The
    scorer insists on an ``(N, 1)`` column so any caller that forgets the
    column reshape fails loudly.

    Parameters
    ----------
    n_components (int)
        Number of components the fake claims to have.
    level (float)
        Constant per-point log-likelihood.
    slope (float)
        Per-point log-likelihood slope in the log-space value; defaults to 0.
    """

    def __init__(self, n_components, level, slope=0.0):
        self.n_components = int(n_components)
        self.weights_ = np.full(self.n_components, 1.0 / self.n_components)
        self.means_ = np.linspace(-1.0, 1.0, self.n_components).reshape(-1, 1)
        self.covariances_ = np.full((self.n_components, 1, 1), 0.25)
        self.nus_ = np.full(self.n_components, 8.0)
        self.level = float(level)
        self.slope = float(slope)

    def score_samples(self, X):
        """
        Returns the hand-set per-point log-likelihood of a column of points.

        Parameters
        ----------
        X (np.ndarray)
            An (N, 1) shape ndarray of log-space points.

        Returns
        -------
        log_likelihood (np.ndarray)
            An (N,) shape ndarray equal to ``level + slope * X[:, 0]``.
        """
        X = np.asarray(X)
        assert X.ndim == 2 and X.shape[1] == 1, f"score_samples got shape {X.shape}"
        return self.level + self.slope * X[:, 0]


class _CFakeFitter:
    """
    A recording stand-in for ``fit_log_gmm`` / ``fit_log_t_mixture``.

    Every invocation is stored (data, component count and each keyword, with
    an explicit ``'missing'`` sentinel for keywords the caller did not pass),
    and a :class:`_CFakeModel` is returned whose level comes from
    ``level_fn(n_components, seed, x_is_log)``. Because the lookup is keyed on
    the seed, a caller that passes the wrong seed raises immediately.

    Parameters
    ----------
    level_fn (callable)
        Maps ``(n_components, seed, x_is_log)`` to the fake's per-point level.
    slope_fn (callable or None)
        Optional map of the same arguments to the fake's slope; defaults to a
        zero slope.
    """

    def __init__(self, level_fn, slope_fn=None):
        self.level_fn = level_fn
        self.slope_fn = slope_fn
        self.requests = []

    def __call__(self, x, n_components, seed=_C_MISSING, n_init=_C_MISSING,
                 reg_covar=_C_MISSING, x_is_log=_C_MISSING, init_params=_C_MISSING):
        """
        Records the request and returns a fake fitted model.

        Parameters
        ----------
        x (np.ndarray)
            The data handed to the fitter.
        n_components (int)
            Requested component count.
        seed, n_init, reg_covar, x_is_log, init_params
            Fitter keywords; ``'missing'`` when not passed.

        Returns
        -------
        model (_CFakeModel)
            The fake fit.
        extra (None)
            Placeholder for the fitter's second return value.
        """
        slope = 0.0 if self.slope_fn is None else self.slope_fn(n_components, seed, x_is_log)
        model = _CFakeModel(n_components, self.level_fn(n_components, seed, x_is_log), slope)
        self.requests.append({
            "x": np.array(x, dtype=float, copy=True),
            "K": n_components,
            "seed": seed,
            "n_init": n_init,
            "reg_covar": reg_covar,
            "x_is_log": x_is_log,
            "init_params": init_params,
            "model": model,
        })
        return model, None


class _CFakeSplitter:
    """
    A recording stand-in for :func:`fit_nested_by_splitting`.

    Returns a :class:`_CFakeModel` with one more component than the base
    model, at the level given by ``level_fn(K + 1, seed)``, together with its
    summed log-likelihood on the supplied data.

    Parameters
    ----------
    level_fn (callable)
        Maps ``(n_components, seed)`` to the per-point level of the result.
    """

    def __init__(self, level_fn):
        self.level_fn = level_fn
        self.requests = []

    def __call__(self, log_x, base_model, n_init=_C_MISSING, seed=_C_MISSING,
                 reg_covar=_C_MISSING):
        """
        Records the request and returns the split model and its log-likelihood.

        Parameters
        ----------
        log_x (np.ndarray)
            Log-space data.
        base_model (_CFakeModel)
            The model being split.
        n_init, seed, reg_covar
            Keywords; ``'missing'`` when not passed.

        Returns
        -------
        model (_CFakeModel)
            The ``K + 1`` model.
        log_likelihood (float)
            Its summed log-likelihood on ``log_x``.
        """
        model = _CFakeModel(base_model.n_components + 1,
                            self.level_fn(base_model.n_components + 1, seed))
        self.requests.append({"log_x": np.array(log_x, dtype=float, copy=True),
                           "base": base_model, "n_init": n_init, "seed": seed,
                           "reg_covar": reg_covar, "model": model})
        return model, float(np.sum(model.score_samples(np.asarray(log_x).reshape(-1, 1))))


class _CParallelSpy:
    """
    A sequential, in-process stand-in for ``joblib.Parallel``.

    Records every ``n_jobs`` it is constructed with and evaluates the
    ``delayed`` tuples in order, so monkeypatched fitters stay visible to the
    replicates.
    """

    n_jobs_seen: list = []

    def __init__(self, n_jobs):
        type(self).n_jobs_seen.append(n_jobs)

    def __call__(self, tasks):
        """
        Runs each ``(function, args, kwargs)`` task in order.

        Parameters
        ----------
        tasks (iterable)
            ``joblib.delayed`` tuples.

        Returns
        -------
        results (list)
            The task results, in order.
        """
        return [function(*args, **kwargs) for function, args, kwargs in tasks]


def _c_pool(n=40, seed=0):
    """
    Returns a small, strictly positive interval pool in seconds.

    Parameters
    ----------
    n (int)
        Pool size; defaults to 40.
    seed (int)
        RNG seed; defaults to 0.

    Returns
    -------
    intervals (np.ndarray)
        An (n,) shape ndarray of log-normal intervals.
    """
    return np.exp(np.random.default_rng(seed).normal(-2.0, 0.5, n))


def _c_gmm(weights, means, variances):
    """
    Builds a 1D full-covariance ``GaussianMixture`` from explicit parameters.

    Parameters
    ----------
    weights (sequence)
        Component weights.
    means (sequence)
        Component means.
    variances (sequence)
        Component variances.

    Returns
    -------
    gmm (GaussianMixture)
        A mixture ready for ``score_samples`` / ``predict_proba``.
    """
    variances = np.asarray(variances, dtype=float)
    gmm = GaussianMixture(n_components=len(weights), covariance_type="full")
    gmm.weights_ = np.asarray(weights, dtype=float)
    gmm.means_ = np.asarray(means, dtype=float).reshape(-1, 1)
    gmm.covariances_ = variances.reshape(-1, 1, 1)
    gmm.precisions_cholesky_ = (1.0 / np.sqrt(variances)).reshape(-1, 1, 1)
    return gmm


def _c_splitmix64_int(value):
    """
    Pure-integer reference implementation of the splitmix64 finalizer.

    Parameters
    ----------
    value (int)
        A 64-bit state.

    Returns
    -------
    mixed (int)
        The mixed 64-bit value.
    """
    z = (value + _C_GOLDEN) & _C_MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _C_MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _C_MASK64
    return z ^ (z >> 31)


def test_splitmix64_matches_the_published_reference_sequence():
    """
    :func:`_splitmix64` is ``mix(state + golden)``, so feeding it ``0``,
    ``golden`` and ``2 * golden`` must reproduce the first three outputs of the
    reference splitmix64 generator seeded at zero (0xE220A8397B1DCDAF,
    0x6E789E6AA1B965F4, 0x06C45D188009454F). Further inputs, including the
    wrap-around value ``2**64 - 1``, must match a pure-integer reference
    implementation bit for bit, and the result must stay ``uint64``.
    """
    states = [0, _C_GOLDEN, (2 * _C_GOLDEN) & _C_MASK64, 1, 12345, _C_MASK64, 0xDEADBEEFCAFEBABE]
    mixed = _splitmix64(np.array(states, dtype=np.uint64))

    assert mixed.dtype == np.uint64
    assert [int(v) for v in mixed[:3]] == [0xE220A8397B1DCDAF, 0x6E789E6AA1B965F4, 0x06C45D188009454F]
    assert [int(v) for v in mixed] == [_c_splitmix64_int(s) for s in states]


def test_splitmix64_scalar_input_wraps_without_overflow_warning():
    """
    The 64-bit arithmetic is meant to wrap; :func:`_splitmix64` suppresses the
    overflow warning so a numpy ``uint64`` scalar (whose arithmetic, unlike an
    array's, does warn) is mixed silently. Under the suite's warnings-as-errors
    filter any leaked overflow warning fails this test.
    """
    mixed = _splitmix64(np.uint64(_C_MASK64))

    assert int(mixed) == _c_splitmix64_int(_C_MASK64)


def _c_reference_subsample_indices(values, n_subsample, seed):
    """
    Independent, loop-based reference for the positions :func:`stable_subsample`
    draws: each element's key is ``splitmix64(bits(value) ^ (occurrence *
    golden) ^ seed)``, where ``occurrence`` counts the exactly equal values
    that precede it in the ORIGINAL order, and the draw is the ``n_subsample``
    smallest keys.

    Parameters
    ----------
    values (np.ndarray)
        A (n,) shape ndarray of float values.
    n_subsample (int)
        Subsample size.
    seed (int)
        Seed mixed into every key.

    Returns
    -------
    indices (set)
        The drawn positions.
    """
    seen = {}
    keys = []
    for position, value in enumerate(values.tolist()):
        occurrence = seen.setdefault(value, 0)
        seen[value] = occurrence + 1
        bits = struct.unpack("<Q", struct.pack("<d", value))[0]
        state = bits ^ ((occurrence * _C_GOLDEN) & _C_MASK64) ^ seed
        keys.append((_c_splitmix64_int(state), position))
    keys.sort()
    return {position for _, position in keys[:n_subsample]}


def test_stable_subsample_matches_a_loop_reference_with_heavy_ties():
    """
    With many exactly tied values (including several copies of the minimum,
    which exercises the first-group bookkeeping) the occurrence index among
    equal values must follow the original order, the keys must use the
    golden-ratio constant and the seed, and the draw must be exactly the
    ``n_subsample`` smallest keys. The positions drawn are compared with an
    independent loop implementation for two seeds, and the values must be
    ``values[indices]``.
    """
    rng = np.random.default_rng(11)
    pool = np.round(rng.lognormal(-2.7, 1.0, 3000), 2)
    pool[rng.choice(pool.size, 40, replace=False)] = pool.min()
    for seed in (0, 97):
        subsample, indices = stable_subsample(pool, 700, seed, return_indices=True)

        assert indices.size == 700
        assert set(indices.tolist()) == _c_reference_subsample_indices(pool, 700, seed)
        assert np.array_equal(subsample, pool[indices])


@pytest.mark.parametrize("pool_seed, n_subsample, seed", [(101, 333, 32), (100, 500, 26)])
def test_stable_subsample_takes_exactly_the_smallest_keys(pool_seed, n_subsample, seed):
    """
    The draw must be exactly the ``n_subsample`` smallest keys, which needs
    the partition pivot at index ``n_subsample - 1``: a pivot one or two
    places off leaves the boundary unordered and, for these pools and seeds
    (found by search), swaps one of the smallest keys for a larger one.
    """
    pool = np.random.default_rng(pool_seed).lognormal(-2.7, 1.0, 1000)
    _, indices = stable_subsample(pool, n_subsample, seed, return_indices=True)

    assert set(indices.tolist()) == _c_reference_subsample_indices(pool, n_subsample, seed)


def test_stable_subsample_pool_of_exactly_the_requested_size_is_returned_in_order():
    """
    A pool holding exactly ``n_subsample`` values is "no more than" the
    request, so it is returned whole and in its original order, with identity
    indices -- not routed through the key draw, which would permute it.
    """
    pool = np.random.default_rng(12).lognormal(-2.7, 1.0, 50)
    subsample, indices = stable_subsample(pool, 50, 3, return_indices=True)

    assert np.array_equal(subsample, pool)
    assert np.array_equal(indices, np.arange(50))


def test_stable_subsample_casts_integer_pools_to_float64():
    """
    Keys are built from the float64 bit pattern, so an integer pool must be
    cast first: it draws exactly what its float64 copy draws and returns
    float64 values.
    """
    pool = np.random.default_rng(13).integers(1, 10_000, 600)
    drawn_int = stable_subsample(pool, 100, 5)
    drawn_float = stable_subsample(pool.astype(np.float64), 100, 5)

    assert drawn_int.dtype == np.float64
    assert np.array_equal(drawn_int, drawn_float)


def _c_expected_draws(model, n, seed, draw):
    """
    Re-derives the expected output of :func:`_sample_from_mixture`: normalise
    the weights, draw the categorical assignment, then fill each populated
    component, in component order, with ``draw(k, n_k, rng)``.

    Parameters
    ----------
    model
        The mixture being sampled.
    n (int)
        Number of samples.
    seed (int)
        RNG seed.
    draw (callable)
        ``draw(k, n_k, rng)`` returning the component-``k`` draws.

    Returns
    -------
    expected (np.ndarray)
        The (n,) expected samples.
    counts (np.ndarray)
        Per-component assignment counts.
    """
    rng = np.random.default_rng(seed)
    weights = np.asarray(model.weights_, dtype=float).ravel()
    comp = rng.choice(model.n_components, size=n, p=weights / weights.sum())
    expected = np.full(n, np.nan)
    for k in range(model.n_components):
        idx = comp == k
        if idx.sum() > 0:
            expected[idx] = draw(k, int(idx.sum()), rng)
    return expected, np.bincount(comp, minlength=model.n_components)


_C_SAMPLE_WEIGHTS = [2e-6, 0.04, 1.96]


def test_sample_from_mixture_gauss_branch_reproduces_the_seeded_stream():
    """
    The Gaussian branch must renormalise unnormalised weights (here summing to
    two), skip the empty first component without stopping, sample the
    single-draw second component, and draw each component at its own mean and
    standard deviation. Under seed 3 the assignment counts are (0, 1, 49), and
    the output must equal the hand-derived stream exactly.
    """
    gmm = _c_gmm(_C_SAMPLE_WEIGHTS, [-3.0, 0.5, 4.0], [0.04, 0.25, 1.0])
    out = _sample_from_mixture(gmm, 50, np.random.default_rng(3))
    expected, counts = _c_expected_draws(
        gmm, 50, 3, lambda k, n_k, rng: rng.normal(loc=gmm.means_[k, 0],
                                                   scale=np.sqrt(gmm.covariances_[k, 0, 0]), size=n_k))

    assert counts.tolist() == [0, 1, 49]
    np.testing.assert_array_equal(out, expected)


def test_sample_from_mixture_t_branch_reproduces_the_seeded_stream():
    """
    The Student-t branch must draw each populated component with its own
    degrees of freedom, location and scale (square root of the variance),
    skip the empty component without stopping and keep the single-draw one.
    The output must equal the hand-derived ``t.rvs`` stream exactly.
    """
    model = TMixture(weights=_C_SAMPLE_WEIGHTS, means=[-3.0, 0.5, 4.0],
                     covariances=[0.04, 0.25, 2.25], nus=[3.0, 6.0, 9.0])
    out = _sample_from_mixture(model, 50, np.random.default_rng(3))
    expected, counts = _c_expected_draws(
        model, 50, 3, lambda k, n_k, rng: stats.t.rvs(df=model.nus_[k], loc=model.means_[k, 0],
                                                      scale=np.sqrt(model.covariances_[k, 0, 0]),
                                                      size=n_k, random_state=rng))

    assert counts.tolist() == [0, 1, 49]
    np.testing.assert_array_equal(out, expected)


def test_sample_from_mixture_ig_branch_reproduces_the_seeded_stream():
    """
    The inverse-Gaussian branch must draw ``IG(mu, lambda)`` per populated
    component (scipy's ``invgauss(mu / lambda, scale=lambda)``), return the
    LOG of the draws, skip the empty component without stopping and keep the
    single-draw one. The output must equal the hand-derived stream exactly.
    """
    model = IGMixture(weights=_C_SAMPLE_WEIGHTS, mus=[0.05, 0.3, 2.0], lambdas=[1.0, 2.0, 3.0])
    out = _sample_from_mixture(model, 50, np.random.default_rng(3))
    expected, counts = _c_expected_draws(
        model, 50, 3, lambda k, n_k, rng: np.log(stats.invgauss.rvs(
            model.mus_[k] / model.lambdas_[k], scale=model.lambdas_[k], size=n_k, random_state=rng)))

    assert counts.tolist() == [0, 1, 49]
    np.testing.assert_array_equal(out, expected)


def test_lr_statistic_is_minus_two_times_the_log_likelihood_difference():
    """
    With a null scoring -1.5 nats per point and an alternative scoring -1.0
    on four points, ``LR = -2 * (-6 - (-4)) = 4`` exactly.
    """
    log_x = np.array([0.1, -0.2, 0.3, 0.4])
    lr = _lr_statistic(_CFakeModel(1, -1.5), _CFakeModel(2, -1.0), log_x)

    assert lr == 4.0


def _c_log_gauss_1d(x, mu, var):
    """
    1D Gaussian log-density written out by hand.

    Parameters
    ----------
    x (np.ndarray)
        Evaluation points.
    mu (float)
        Mean.
    var (float)
        Variance.

    Returns
    -------
    log_density (np.ndarray)
        ``-0.5 * ((x - mu)**2 / var + log(var) + log(2 pi))``.
    """
    return -0.5 * ((x - mu) ** 2 / var + np.log(var) + np.log(2.0 * np.pi))


def test_alpha_is_weighted_density_scaled_by_the_per_point_maximum():
    """
    ``_alpha`` returns ``w_k N(x | mu_k, var_k)`` divided by the largest of
    those over components, separately for every point. For a single point
    the largest entry is exactly one and the ratios match the hand-computed
    weighted densities; for a batch of points each column is scaled by its
    OWN maximum, not by a global one.
    """
    w = np.array([0.3, 0.7])
    M = np.array([[0.0], [2.0]])
    Sig = np.array([[[1.0]], [[0.5]]])
    Prec = np.array([[[1.0]], [[2.0]]])

    point = np.array([0.4])
    logs = np.log(w) + np.array([_c_log_gauss_1d(0.4, 0.0, 1.0), _c_log_gauss_1d(0.4, 2.0, 0.5)])
    a = _alpha(point, w, M, Sig, Prec)
    np.testing.assert_allclose(a, np.exp(logs - logs.max()), rtol=1e-12)

    batch = np.array([[-1.0], [0.4], [1.5], [3.0]])
    logs_b = np.log(w)[:, None] + np.vstack([_c_log_gauss_1d(batch[:, 0], 0.0, 1.0),
                                             _c_log_gauss_1d(batch[:, 0], 2.0, 0.5)])
    a_b = _alpha(batch, w, M, Sig, Prec)
    np.testing.assert_allclose(a_b, np.exp(logs_b - logs_b.max(axis=0)), rtol=1e-12)
    np.testing.assert_allclose(a_b.max(axis=0), 1.0)


def test_select_best_n_components_takes_the_row_minimising_the_ic():
    """
    For an information criterion other than CV, the winner per key is the
    single row with the smallest value of that column, and its rep is
    reported as-is (here the AIC winner's rep differs from the rep with the
    smallest BIC at that K, so using BIC would be visible). Keys must not
    leak into each other.
    """
    df = pls.DataFrame({
        "key": ["A"] * 4 + ["B"] * 4,
        "n_comp": [1, 2, 2, 3, 1, 2, 3, 3],
        "rep": [0, 0, 1, 0, 0, 0, 0, 1],
        "bic": [10.0, 3.0, 2.0, 1.0, 9.0, 8.0, 7.0, 0.5],
        "aic": [5.0, 4.0, 6.0, 7.0, 3.0, 2.5, 1.5, 2.0],
    })
    best = select_best_n_components(df, ic_col="aic")

    assert best == {"A": {"n_comp": 2, "rep": 0, "aic": 4.0},
                    "B": {"n_comp": 3, "rep": 0, "aic": 1.5}}


def test_select_best_n_components_cv_mode_picks_rep_by_bic_at_the_chosen_k():
    """
    In CV mode the component count comes from the smallest CV loss, but the
    rep is the best in-sample EM start at THAT count -- the smallest BIC among
    its rows -- even when another count holds a smaller BIC and even when the
    CV row that won is a different rep.
    """
    df = pls.DataFrame({
        "key": ["A"] * 6 + ["B"] * 3,
        "n_comp": [1, 1, 2, 2, 2, 3, 1, 2, 2],
        "rep": [0, 2, 0, 1, 2, 0, 0, 0, 3],
        "bic": [9.0, 0.1, 7.0, 5.0, 6.0, 4.0, 5.0, 8.0, 6.5],
        "cv_neg_loglik": [50.0, 50.0, 40.0, 40.1, 40.2, 45.0, 30.0, 20.0, 20.5],
    })
    best = select_best_n_components(df, ic_col="cv_neg_loglik")

    assert best == {"A": {"n_comp": 2, "rep": 1, "cv_neg_loglik": 40.0},
                    "B": {"n_comp": 2, "rep": 3, "cv_neg_loglik": 20.0}}


def test_step_up_lrt_stops_at_the_first_non_rejection_in_sorted_order():
    """
    Pairs are walked in ascending order regardless of dict order, and a
    p-value exactly equal to alpha counts as a failure to reject, so the walk
    stops there and returns that pair's K_null.
    """
    pairs = {(3, 4): {"p_value": 0.2}, (1, 2): {"p_value": 0.01}, (2, 3): {"p_value": 0.05}}

    assert select_n_components_step_up_lrt(pairs, alpha=0.05) == 2


def test_step_up_lrt_all_rejections_return_the_largest_alternative():
    """
    When every test rejects, the largest K_alt seen is selected; an empty
    result set raises with an explanatory message.
    """
    pairs = {(2, 3): {"p_value": 0.002}, (1, 2): {"p_value": 0.001}, (3, 4): {"p_value": 0.003}}

    assert select_n_components_step_up_lrt(pairs, alpha=0.05) == 4
    with pytest.raises(ValueError, match="no pair_results supplied"):
        select_n_components_step_up_lrt({})


def _c_design_effect_by_hand(delta, session_index, n_bootstrap, seed):
    """
    Loop-based re-derivation of :func:`_session_design_effect`: the differences
    centred on their mean, per-session sums of the centred values, ``n_bootstrap`` session resamples and ``min(n_bootstrap, 400)``
    observation resamples from one seeded stream, sample standard deviations
    (ddof 1) of twice the resampled sums, and their squared ratio.

    Parameters
    ----------
    delta (np.ndarray)
        Per-observation differences.
    session_index (np.ndarray)
        Integer session codes.
    n_bootstrap (int)
        Session-resampling replicates.
    seed (int)
        RNG seed.

    Returns
    -------
    result (tuple)
        ``(design_effect, clustered_se, independent_se)``.
    """
    n_sessions = int(max(session_index.tolist())) + 1
    mean_delta = sum(delta.tolist()) / delta.size
    delta = np.array([d - mean_delta for d in delta.tolist()])
    per_session = [sum(d for d, s in zip(delta.tolist(), session_index.tolist()) if s == j)
                   for j in range(n_sessions)]
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_sessions, size=(n_bootstrap, n_sessions))
    clustered = [2.0 * sum(per_session[j] for j in row) for row in draws.tolist()]
    interval_draws = rng.integers(0, delta.size, size=(min(n_bootstrap, 400), delta.size))
    independent = [2.0 * sum(delta[i] for i in row) for row in interval_draws.tolist()]
    clustered_se = float(np.std(clustered, ddof=1))
    independent_se = float(np.std(independent, ddof=1))
    return (clustered_se / independent_se) ** 2, clustered_se, independent_se


def test_session_design_effect_matches_a_hand_derivation():
    """
    On five uneven sessions with small differences (so the iid standard error
    is well below one) and ``n_bootstrap = 500`` (so the iid resample count is
    capped at 400), the design effect and both standard errors must match a
    loop-based derivation that uses the same seeded stream.
    """
    rng = np.random.default_rng(21)
    session_index = np.repeat(np.arange(5), [6, 14, 9, 20, 11])
    delta = rng.normal(0.0, 0.01, session_index.size) + 0.02 * session_index
    result = _session_design_effect(delta, session_index, 500, 4)
    expected = _c_design_effect_by_hand(delta, session_index, 500, 4)

    assert result[2] < 1.0
    np.testing.assert_allclose(result, expected, rtol=1e-9)


def test_session_design_effect_is_nan_when_differences_are_constant():
    """
    Constant per-observation differences give a zero iid standard error, so
    the ratio is undefined and must be reported as NaN rather than divided.
    """
    design_effect, _, independent_se = _session_design_effect(
        np.full(30, 0.5), np.repeat(np.arange(3), 10), 50, 0)

    assert independent_se == 0.0
    assert np.isnan(design_effect)


def _c_corrected_inputs(seed, n_sessions=12, per_session=10, extra=7):
    """
    Builds a session-structured ``delta`` plus a full-pool label array in which
    the subsample occupies shuffled positions, for the
    :func:`_session_corrected_statistic` tests.

    Parameters
    ----------
    seed (int)
        Seed of the session offsets and noise.
    n_sessions (int)
        Number of sessions in the subsample; defaults to 12.
    per_session (int)
        Observations per session; defaults to 10.
    extra (int)
        Pool positions left out of the subsample; defaults to 7.

    Returns
    -------
    delta (np.ndarray)
        Per-observation differences of the subsample.
    labels (np.ndarray)
        Session labels of the full pool.
    chosen (np.ndarray)
        Subsample positions in the pool.
    session_index (np.ndarray)
        Integer session codes of the subsample.
    """
    rng = np.random.default_rng(seed)
    offsets = rng.normal(0.0, 0.12, n_sessions)
    delta = np.repeat(offsets, per_session) + rng.normal(0.0, 1.0, n_sessions * per_session)
    session_index = np.repeat(np.arange(n_sessions), per_session)
    n_values = delta.size + extra
    chosen = np.random.default_rng(99).permutation(n_values)[:delta.size]
    labels = np.array(["s00"] * n_values, dtype=object)
    labels[chosen] = np.array([f"s{code:02d}" for code in session_index], dtype=object)
    return delta, labels, chosen, session_index


def test_session_corrected_statistic_divides_by_an_unfloored_design_effect():
    """
    When the raw design effect lies between one and two the floor does not
    bind: the reported effect is the raw one, computed from the subsample's
    sessions with the caller's seed, the effective sample size is
    ``n / design_effect`` and the corrected statistic ``lr / design_effect``.
    """
    delta, labels, chosen, session_index = _c_corrected_inputs(3)
    raw, clustered_se, independent_se = _session_design_effect(delta, session_index, 300, 6)
    out = _session_corrected_statistic(12.0, delta, labels, labels.size, chosen, 300, 6, "caller")

    assert 1.0 < raw < 2.0
    assert out["design_effect_raw"] == raw
    assert out["design_effect"] == raw
    assert out["clustered_se"] == clustered_se
    assert out["independent_se"] == independent_se
    assert out["effective_n"] == pytest.approx(delta.size / raw)
    assert out["lr_corrected"] == pytest.approx(12.0 / raw)


def test_session_corrected_statistic_floors_the_design_effect_at_one():
    """
    A raw design effect below one is floored at exactly one, which must leave
    the statistic and the effective sample size untouched (not NaN).
    """
    delta, labels, chosen, session_index = _c_corrected_inputs(8)
    out = _session_corrected_statistic(12.0, delta, labels, labels.size, chosen, 300, 0, "caller")

    assert out["design_effect_raw"] < 1.0
    assert out["design_effect"] == 1.0
    assert out["effective_n"] == float(delta.size)
    assert out["lr_corrected"] == 12.0


def _c_gauss_levels(gains):
    """
    Level map for a Gaussian-family fake: the null (one component) scores
    -1.0 per point, the alternative -1.0 + ``gains[seed]``. All values are
    dyadic, so every derived statistic is exact in floating point.

    Parameters
    ----------
    gains (dict)
        Seed -> per-point gain of the alternative.

    Returns
    -------
    level_fn (callable)
        ``level_fn(n_components, seed, x_is_log)``.
    """
    return lambda n_components, seed, x_is_log: -1.0 if n_components == 1 else -1.0 + gains[seed]


def test_bootstrap_lrt_gauss_statistic_p_value_and_fit_arguments(monkeypatch):
    """
    With fake Gaussian fits whose log-likelihoods are set per seed, the
    observed statistic is ``2 N gain(seed)`` and replicate ``b`` (refitted
    with seed ``seed + b``) gives ``2 N gain(seed + b)``. Replicate 0 shares
    the observed seed and therefore TIES the observed statistic, which the
    one-sided p-value ``(1 + #{LR_b >= LR_obs}) / (1 + B)`` must count: (1 + 3) / 6. The null
    summaries are checked by hand (95th percentile by linear interpolation).
    The observed fits receive the seconds-scale subsample with the caller's
    seed, restarts and variance floor; the replicate fits receive log-space
    draws, ``x_is_log=True``, seed ``seed + b`` and the bootstrap restarts.
    The sequential path must not touch joblib.
    """
    _CParallelSpy.n_jobs_seen = []
    monkeypatch.setattr(mixture_model_utils, "Parallel", _CParallelSpy)
    fitter = _CFakeFitter(_c_gauss_levels({3: 0.5, 4: 1.0, 5: 0.25, 6: 0.0, 7: 1.5}))
    monkeypatch.setattr(mixture_model_utils, "fit_log_gmm", fitter)
    pool = _c_pool()

    result = bootstrap_lrt(pool, K_null=1, K_alt=2, B=5, n_subsample=100, model_class="gauss",
                           n_init_obs=4, n_init_boot=2, reg_covar=1e-3, seed=3)

    assert result["lr_obs"] == 40.0
    np.testing.assert_array_equal(result["lr_null"], [40.0, 80.0, 20.0, 0.0, 120.0])
    assert result["p_value"] == pytest.approx(4.0 / 6.0)
    assert result["null_mean"] == 52.0
    assert result["null_p95"] == pytest.approx(112.0)
    assert result["null_max"] == 120.0
    assert result["n_subsample"] == 40
    assert _CParallelSpy.n_jobs_seen == []

    observed, replicates = fitter.requests[:2], fitter.requests[2:]
    assert [c["K"] for c in observed] == [1, 2]
    for request in observed:
        np.testing.assert_array_equal(request["x"], pool)
        assert (request["seed"], request["n_init"], request["reg_covar"], request["x_is_log"]) == (3, 4, 1e-3, _C_MISSING)
    assert [(c["K"], c["seed"]) for c in replicates] == [(K, 3 + b) for b in range(5) for K in (1, 2)]
    for request in replicates:
        assert (request["n_init"], request["reg_covar"], request["x_is_log"]) == (2, 1e-3, True)
        assert request["x"].shape == (40,)


def test_bootstrap_lrt_rejects_equal_component_counts():
    """
    ``K_alt`` must be strictly larger than ``K_null``; equal counts are not a
    nested test and must raise before any fitting.
    """
    with pytest.raises(ValueError, match="K_alt > K_null"):
        bootstrap_lrt(_c_pool(), K_null=2, K_alt=2, B=1, model_class="gauss")


@pytest.mark.parametrize(
    "null_level, alt_level, should_raise",
    [
        (0.0, 0.0, False),
        (-1.0, -1.0, False),
        (-1.0, -1.0 - 1e-7, False),
        (-1.0, -1.5, True),
    ],
)
def test_bootstrap_lrt_negative_observed_lr_tolerance(monkeypatch, null_level, alt_level, should_raise):
    """
    A materially negative observed LR (below ``-1e-6 * |logL_null|``) means
    the alternative's fit failed and must raise; an LR of exactly zero (also
    when the null log-likelihood itself is zero) or a negative LR inside the
    relative tolerance is accepted. Here N = 40, so the tolerance is
    ``-4e-5`` for a null at -1 nat per point: -8e-6 passes and -40 raises.
    """
    levels = {1: null_level, 2: alt_level}
    monkeypatch.setattr(mixture_model_utils, "fit_log_gmm",
                        _CFakeFitter(lambda n_components, seed, x_is_log: levels[n_components]))
    kwargs = dict(K_null=1, K_alt=2, B=1, n_subsample=100, model_class="gauss", seed=0)

    if should_raise:
        with pytest.raises(RuntimeError, match="observed LR is negative"):
            bootstrap_lrt(_c_pool(), **kwargs)
    else:
        result = bootstrap_lrt(_c_pool(), **kwargs)
        assert result["lr_obs"] == pytest.approx(2.0 * 40 * (alt_level - null_level), abs=1e-9)


def test_bootstrap_lrt_progress_messages_every_tenth_replicate(monkeypatch):
    """
    The sequential path reports progress after every tenth replicate only:
    with B = 12 exactly one message, ``'      bootstrap [10/12]'``, is
    emitted. Without a progress callable nothing is attempted.
    """
    monkeypatch.setattr(mixture_model_utils, "fit_log_gmm",
                        _CFakeFitter(_c_gauss_levels({s: 0.5 for s in range(12)})))
    messages = []
    bootstrap_lrt(_c_pool(), K_null=1, K_alt=2, B=12, n_subsample=100, model_class="gauss",
                  seed=0, message_output=messages.append)
    bootstrap_lrt(_c_pool(), K_null=1, K_alt=2, B=12, n_subsample=100, model_class="gauss",
                  seed=0, message_output=None)

    assert messages == ["      bootstrap [10/12]"]


def test_bootstrap_lrt_parallel_path_runs_replicates_through_joblib(monkeypatch):
    """
    With ``n_jobs > 1`` the replicates go through ``Parallel(n_jobs=n_jobs)``
    and ``_bootstrap_lrt_replicate``; their statistics fill ``lr_null`` as
    floats (here ``2 N gain(seed + b)`` exactly), and one message announces
    the worker count.
    """
    _CParallelSpy.n_jobs_seen = []
    monkeypatch.setattr(mixture_model_utils, "Parallel", _CParallelSpy)
    monkeypatch.setattr(mixture_model_utils, "fit_log_gmm",
                        _CFakeFitter(_c_gauss_levels({0: 0.5, 1: 0.25, 2: 1.0, 3: 0.0})))
    messages = []
    result = bootstrap_lrt(_c_pool(), K_null=1, K_alt=2, B=4, n_subsample=100, model_class="gauss",
                           n_jobs=3, seed=0, message_output=messages.append)

    assert _CParallelSpy.n_jobs_seen == [3]
    np.testing.assert_array_equal(result["lr_null"], [40.0, 20.0, 80.0, 0.0])
    assert messages == ["      bootstrap: 4 replicates across 3 workers"]


def test_bootstrap_lrt_session_correction_receives_its_own_fits(monkeypatch):
    """
    With session labels, the correction must be computed from the
    per-observation log-likelihood differences (alternative MINUS null) of the
    same observed fits on the log subsample, with the caller's seed and the
    caller's name, and its keys must be merged into the result. The corrected
    p-value counts replicates that meet or exceed the corrected statistic:
    with ``lr_corrected`` set equal to a replicate value (20), (1 + 4) / 6.
    """
    recorded = {}

    def _spy(lr_obs, delta, session_labels, n_values, chosen, n_design_bootstrap, seed, caller):
        """Records the correction arguments and returns a canned correction."""
        recorded.update(lr_obs=lr_obs, delta=np.array(delta), n_values=n_values,
                        chosen=np.array(chosen), n_design_bootstrap=n_design_bootstrap,
                        seed=seed, caller=caller)
        return {"design_effect": 2.0, "lr_corrected": 20.0}

    monkeypatch.setattr(mixture_model_utils, "_session_corrected_statistic", _spy)
    fitter = _CFakeFitter(_c_gauss_levels({5: 0.5, 6: 0.25, 7: 1.0, 8: 0.0, 9: 0.75}),
                          slope_fn=lambda n_components, seed, x_is_log:
                          0.125 if (n_components == 2 and x_is_log == _C_MISSING) else 0.0)
    monkeypatch.setattr(mixture_model_utils, "fit_log_gmm", fitter)
    pool = _c_pool()
    labels = np.repeat(np.arange(8), 5)

    result = bootstrap_lrt(pool, K_null=1, K_alt=2, B=5, n_subsample=100, model_class="gauss",
                           seed=5, session_labels=labels, n_design_bootstrap=33)

    log_x = np.log(pool)
    np.testing.assert_allclose(recorded["delta"], 0.5 + 0.125 * log_x, rtol=1e-12)
    assert recorded["lr_obs"] == result["lr_obs"]
    assert (recorded["n_values"], recorded["n_design_bootstrap"], recorded["seed"], recorded["caller"]) \
        == (40, 33, 5, "bootstrap_lrt")
    np.testing.assert_array_equal(recorded["chosen"], np.arange(40))
    np.testing.assert_array_equal(result["lr_null"], [40.0, 20.0, 80.0, 0.0, 60.0])
    assert result["design_effect"] == 2.0
    assert result["p_value_corrected"] == pytest.approx(5.0 / 6.0)


def test_bootstrap_lrt_rejects_misaligned_labels_naming_itself(monkeypatch):
    """
    Session labels that do not align with the pool raise a ``ValueError``
    whose message names ``bootstrap_lrt`` as the caller.
    """
    monkeypatch.setattr(mixture_model_utils, "fit_log_gmm", _CFakeFitter(_c_gauss_levels({0: 0.5})))
    with pytest.raises(ValueError, match="bootstrap_lrt: session_labels has 3 entries for 40"):
        bootstrap_lrt(_c_pool(), K_null=1, K_alt=2, B=1, n_subsample=100, model_class="gauss",
                      n_init_obs=1, session_labels=np.arange(3))


def test_bootstrap_lrt_t_family_builds_both_sides_by_splitting(monkeypatch):
    """
    For the Student-t family the observed null starts as a two-component
    random-restart fit and is grown by splitting ``K_null - 2`` times; the
    alternative is one more split of that null. With K_null = 3 there is one
    restart fit (K = 2, log data, caller seed / restarts / floor) and exactly
    two splits (K 2 -> 3 for the null, 3 -> 4 for the alternative), and the
    observed statistic is ``2 (logL_alt - logL_null) = 2 N (L4 - L3)``. The
    sequential replicates run the same procedure as the parallel ones: each
    refits the null (K = 3) from the observed null's parameters with
    ``x_is_log=True``, seed ``seed + b`` and the bootstrap restarts, and grows
    the alternative by one split of that refit.
    """
    level = {2: -1.0, 3: -0.75, 4: -0.5}
    fitter = _CFakeFitter(lambda n_components, seed, x_is_log: level[n_components] + 0.0 * seed)
    splitter = _CFakeSplitter(lambda n_components, seed: level[n_components] + 0.0 * seed)
    monkeypatch.setattr(mixture_model_utils, "fit_log_t_mixture", fitter)
    monkeypatch.setattr(mixture_model_utils, "fit_nested_by_splitting", splitter)
    pool = _c_pool()

    result = bootstrap_lrt(pool, K_null=3, K_alt=4, B=2, n_subsample=100, model_class="t",
                           n_init_obs=2, n_init_boot=1, reg_covar=1e-3, seed=7)

    assert result["lr_obs"] == 20.0
    first = fitter.requests[0]
    assert (first["K"], first["seed"], first["n_init"], first["reg_covar"], first["x_is_log"]) \
        == (2, 7, 2, 1e-3, True)
    np.testing.assert_allclose(first["x"], np.log(pool), rtol=1e-12)
    observed_splits, replicate_splits = splitter.requests[:2], splitter.requests[2:]
    assert [c["base"].n_components for c in observed_splits] == [2, 3]
    assert observed_splits[0]["base"] is first["model"]
    assert observed_splits[1]["base"] is observed_splits[0]["model"]
    for request in observed_splits:
        assert (request["n_init"], request["seed"], request["reg_covar"]) == (2, 7, 1e-3)
        np.testing.assert_allclose(request["log_x"], np.log(pool), rtol=1e-12)
    assert [(c["K"], c["seed"]) for c in fitter.requests[1:]] == [(3, 7), (3, 8)]
    for request, split in zip(fitter.requests[1:], replicate_splits, strict=True):
        assert (request["n_init"], request["reg_covar"], request["x_is_log"]) == (1, 1e-3, True)
        assert request["init_params"] is not _C_MISSING
        assert split["base"] is request["model"]
        assert (split["n_init"], split["seed"], split["reg_covar"]) == (1, request["seed"], 1e-3)
    np.testing.assert_array_equal(result["lr_null"], [20.0, 20.0])


def test_bootstrap_lrt_t_family_rejects_a_rung_wider_than_one_component():
    """
    The Student-t alternative is built by ONE split of the null, so it always has
    ``K_null + 1`` components; a request for ``K_alt > K_null + 1`` cannot be honoured and must
    raise before any fitting instead of silently testing ``K_null`` vs ``K_null + 1``.
    """
    with pytest.raises(ValueError, match="K_alt = K_null \\+ 1"):
        bootstrap_lrt(_c_pool(), K_null=2, K_alt=4, B=1, n_subsample=100, model_class="t")


def test_bootstrap_lrt_t_family_one_component_null_is_one_component(monkeypatch):
    """
    With ``K_null = 1`` the observed null must be a one-component fit and the alternative its
    single split; the ladder must not start at two components (which made the "null" the
    alternative and the alternative a three-component fit).
    """
    level = {1: -1.0, 2: -0.75, 3: -0.5}
    fitter = _CFakeFitter(lambda n_components, seed, x_is_log: level[n_components] + 0.0 * seed)
    splitter = _CFakeSplitter(lambda n_components, seed: level[n_components] + 0.0 * seed)
    monkeypatch.setattr(mixture_model_utils, "fit_log_t_mixture", fitter)
    monkeypatch.setattr(mixture_model_utils, "fit_nested_by_splitting", splitter)

    result = bootstrap_lrt(_c_pool(), K_null=1, K_alt=2, B=1, n_subsample=100, model_class="t",
                           n_init_obs=2, n_init_boot=1, seed=7)

    assert fitter.requests[0]["K"] == 1
    assert splitter.requests[0]["base"] is fitter.requests[0]["model"]
    assert splitter.requests[0]["model"].n_components == 2
    assert result["lr_obs"] == pytest.approx(2.0 * 40 * (-0.75 + 1.0))


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
@pytest.mark.parametrize("model_class", ["gauss", "t"])
def test_bootstrap_lrt_serial_and_parallel_paths_run_the_same_null(model_class):
    """
    ``n_jobs`` chooses how the replicates are scheduled, not which null procedure runs: the
    serial path must produce exactly the replicate statistics of the parallel one (per-replicate
    RNG seeded by ``(seed, b)``, and for the Student-t family the truth-started null refit and
    split alternative of :func:`_bootstrap_lrt_replicate`).
    """
    rng = np.random.default_rng(4)
    pool = np.exp(np.concatenate([rng.normal(-2.8, 0.25, 200), rng.normal(-0.5, 0.8, 100)]))
    kwargs = {"K_null": 1, "K_alt": 2, "B": 3, "n_subsample": 300, "model_class": model_class,
              "n_init_obs": 3, "n_init_boot": 1, "seed": 1}

    serial = bootstrap_lrt(pool, n_jobs=1, **kwargs)
    parallel = bootstrap_lrt(pool, n_jobs=2, **kwargs)

    np.testing.assert_allclose(serial["lr_null"], parallel["lr_null"], rtol=1e-10, atol=1e-8)


def test_bootstrap_lrt_replicate_gauss_path_arguments_and_statistic():
    """
    A non-t replicate draws ``N_sub`` log-space points from the observed null
    with an RNG seeded by ``(seed, b)``, refits null and alternative on those
    exact draws with seed ``seed + b``, the bootstrap restarts, the variance
    floor and ``x_is_log=True``, and returns their LR statistic
    (``2 * 20 * 0.5 = 20``).
    """
    fitter = _CFakeFitter(lambda n_components, seed, x_is_log: -1.0 if n_components == 1 else -0.5)
    model_null = _CFakeModel(1, -1.0)
    lr = _bootstrap_lrt_replicate(3, 5, model_null, fitter, 1, 2, 20, 2, 1e-3)
    expected_draws = _sample_from_mixture(model_null, 20, np.random.default_rng((5, 3)))

    assert lr == 20.0
    assert [c["K"] for c in fitter.requests] == [1, 2]
    for request in fitter.requests:
        np.testing.assert_array_equal(request["x"], expected_draws)
        assert (request["seed"], request["n_init"], request["reg_covar"], request["x_is_log"]) == (8, 2, 1e-3, True)


def test_bootstrap_lrt_replicate_t_path_starts_null_at_truth_and_splits(monkeypatch):
    """
    A Student-t replicate refits the null starting from the observed null's
    own parameters (weights, means, variances and degrees of freedom), then
    grows the alternative by splitting THAT refit, both with seed ``seed + b``,
    the bootstrap restarts and the variance floor, and returns
    ``2 (logL_alt - logL_null)``: ``2 (-10 - 20 * -1) = 20``.
    """
    fitter = _CFakeFitter(lambda n_components, seed, x_is_log: -1.0)
    splitter_requests = []

    def _splitter(log_x, base_model, n_init=_C_MISSING, seed=_C_MISSING, reg_covar=_C_MISSING):
        """Records the split request and returns a fixed log-likelihood of -10."""
        splitter_requests.append({"log_x": np.array(log_x), "base": base_model, "n_init": n_init,
                               "seed": seed, "reg_covar": reg_covar})
        return None, -10.0

    monkeypatch.setattr(mixture_model_utils, "fit_log_t_mixture", fitter)
    monkeypatch.setattr(mixture_model_utils, "fit_nested_by_splitting", _splitter)
    model_null = TMixture(weights=[0.4, 0.6], means=[-1.0, 1.0], covariances=[0.09, 0.25], nus=[4.0, 12.0])

    lr = _bootstrap_lrt_replicate(3, 5, model_null, mixture_model_utils.fit_log_t_mixture,
                                  2, 3, 20, 2, 1e-3)
    expected_draws = _sample_from_mixture(model_null, 20, np.random.default_rng((5, 3)))

    assert lr == 20.0
    (request,) = fitter.requests
    np.testing.assert_array_equal(request["x"], expected_draws)
    assert (request["K"], request["seed"], request["n_init"], request["reg_covar"], request["x_is_log"]) == (2, 8, 2, 1e-3, True)
    for got, want in zip(request["init_params"], (model_null.weights_, model_null.means_.ravel(),
                                               model_null.covariances_.ravel(), model_null.nus_)):
        np.testing.assert_array_equal(got, want)
    assert len(request["init_params"]) == 4
    (split,) = splitter_requests
    assert split["base"] is request["model"]
    np.testing.assert_array_equal(split["log_x"], expected_draws)
    assert (split["n_init"], split["seed"], split["reg_covar"]) == (2, 8, 1e-3)


def test_fit_nested_by_splitting_tries_every_split_and_keeps_the_first_best(monkeypatch):
    """
    The K+1 search runs one random-restart fit (caller seed, restarts and
    floor, log data) and then, for every component in order, two split starts
    with no restarts: an undisturbed duplicate and a pair pulled
    ``+-split_delta`` standard deviations apart, each copy carrying half the
    parent's weight and the parent's variance and degrees of freedom. A
    candidate replaces the incumbent only when STRICTLY better, so of two
    equally good candidates the first is kept.
    """
    lls = [-32.0, -32.0, -16.0, -16.0, -32.0]
    n_points = 16
    requests = []

    def _fitter(x, n_components, seed=_C_MISSING, n_init=_C_MISSING, reg_covar=_C_MISSING,
                x_is_log=_C_MISSING, init_params=_C_MISSING):
        """Returns a fake whose summed log-likelihood is the next scripted value."""
        model = _CFakeModel(n_components, lls[len(requests)] / n_points)
        requests.append({"x": np.array(x), "K": n_components, "seed": seed, "n_init": n_init,
                      "reg_covar": reg_covar, "x_is_log": x_is_log, "init_params": init_params,
                      "model": model})
        return model, None

    monkeypatch.setattr(mixture_model_utils, "fit_log_t_mixture", _fitter)
    base = TMixture(weights=[0.25, 0.75], means=[-1.0, 2.0], covariances=[0.25, 1.0], nus=[5.0, 10.0])
    log_x = np.linspace(-2.0, 3.0, n_points)

    model, ll = fit_nested_by_splitting(log_x, base, n_init=3, seed=4, reg_covar=1e-3, split_delta=0.5)

    assert model is requests[2]["model"]
    assert ll == -16.0
    assert len(requests) == 5
    restart = requests[0]
    assert (restart["K"], restart["seed"], restart["n_init"], restart["reg_covar"], restart["x_is_log"],
            restart["init_params"]) == (3, 4, 3, 1e-3, True, _C_MISSING)
    expected_starts = [
        ([0.75, 0.125, 0.125], [2.0, -1.0, -1.0], [1.0, 0.25, 0.25], [10.0, 5.0, 5.0]),
        ([0.75, 0.125, 0.125], [2.0, -1.25, -0.75], [1.0, 0.25, 0.25], [10.0, 5.0, 5.0]),
        ([0.25, 0.375, 0.375], [-1.0, 2.0, 2.0], [0.25, 1.0, 1.0], [5.0, 10.0, 10.0]),
        ([0.25, 0.375, 0.375], [-1.0, 1.5, 2.5], [0.25, 1.0, 1.0], [5.0, 10.0, 10.0]),
    ]
    for request, start in zip(requests[1:], expected_starts):
        assert (request["K"], request["seed"], request["n_init"], request["reg_covar"], request["x_is_log"]) \
            == (3, 4, 0, 1e-3, True)
        for got, want in zip(request["init_params"], start):
            np.testing.assert_allclose(got, want, rtol=1e-12)
        np.testing.assert_array_equal(request["x"], log_x)


def _c_plot_model():
    """
    Two-component 1D Gaussian mixture and a matching sample for the plotting
    tests.

    Returns
    -------
    gmm (GaussianMixture)
        Mixture with weights (0.4, 0.6), means (-1, 1), variances (0.09, 0.25).
    x (np.ndarray)
        A (3000,) sample drawn from it.
    """
    gmm = _c_gmm([0.4, 0.6], [-1.0, 1.0], [0.09, 0.25])
    rng = np.random.default_rng(31)
    x = np.concatenate([rng.normal(-1.0, 0.3, 1200), rng.normal(1.0, 0.5, 1800)])
    return gmm, x


def _c_mixture_pdf(grid):
    """
    Analytic density of :func:`_c_plot_model`'s mixture.

    Parameters
    ----------
    grid (np.ndarray)
        Evaluation points.

    Returns
    -------
    pdf (np.ndarray)
        Mixture density at ``grid``.
    """
    return 0.4 * stats.norm.pdf(grid, -1.0, 0.3) + 0.6 * stats.norm.pdf(grid, 1.0, 0.5)


def test_plot_gmm_fit_histogram_is_the_density_of_the_windowed_data():
    """
    With ``xlims`` the data are cut to the closed window (points ON either
    bound kept, points outside dropped) before binning, the histogram is the
    density histogram of exactly those points with the requested bin count,
    drawn with the requested face and edge colours, the figure has the
    requested size, and the axes are fixed to the window.
    """
    gmm, x = _c_plot_model()
    xlims = (-1.5, 1.8)
    x = np.concatenate([x, [xlims[0], xlims[1], -3.0, 4.0]])
    fig, ax = plot_gmm_fit(gmm, x, figsize=(4, 3), bins=25, xlims=xlims, color="#3366aa",
                           edge_color="#aa2200", histtype="bar")
    kept = x[(x >= xlims[0]) & (x <= xlims[1])]
    heights, edges = np.histogram(kept, bins=25, density=True)
    bars = [p for p in ax.patches if isinstance(p, Rectangle)]

    assert len(bars) == 25
    np.testing.assert_allclose([b.get_height() for b in bars], heights, rtol=1e-12)
    np.testing.assert_allclose([b.get_x() for b in bars], edges[:-1], rtol=1e-12, atol=1e-12)
    assert to_hex(bars[0].get_facecolor()) == "#3366aa"
    assert to_hex(bars[0].get_edgecolor()) == "#aa2200"
    np.testing.assert_allclose(fig.get_size_inches(), [4, 3])
    assert ax.get_xlim() == xlims
    plt.close(fig)


def test_plot_gmm_fit_default_histogram_is_one_filled_step_polygon():
    """
    The default ``histtype='stepfilled'`` draws the histogram as a single
    filled polygon spanning the data range, in the requested fill colour.
    """
    gmm, x = _c_plot_model()
    fig, ax = plot_gmm_fit(gmm, x, bins=20, color="#4682b4", legend=False)

    assert len(ax.patches) == 1
    assert isinstance(ax.patches[0], Polygon)
    assert to_hex(ax.patches[0].get_facecolor()) == "#4682b4"
    vertices_x = ax.patches[0].get_xy()[:, 0]
    assert vertices_x.min() == pytest.approx(x.min())
    assert vertices_x.max() == pytest.approx(x.max())
    plt.close(fig)


def test_plot_gmm_fit_point_wise_curve_and_components():
    """
    In point-wise mode the mixture is one solid black line of 500 points
    spanning the window, equal to the mixture density; with
    ``show_components`` each component is a dashed line of its
    posterior-weighted density ``w_k N_k(x)`` on the same grid. Axis labels
    are the log-interval / density labels and the legend names the mixture.
    """
    gmm, x = _c_plot_model()
    xlims = (-2.0, 2.5)
    fig, ax = plot_gmm_fit(gmm, x, bins=30, xlims=xlims, show_components=True)
    lines = ax.get_lines()
    grid = np.linspace(xlims[0], xlims[1], 500)

    assert len(lines) == 3
    mixture = lines[0]
    np.testing.assert_allclose(np.ravel(mixture.get_xdata()), grid, rtol=1e-12)
    np.testing.assert_allclose(np.ravel(mixture.get_ydata()), _c_mixture_pdf(grid), rtol=1e-9)
    assert to_hex(mixture.get_color()) == "#000000"
    assert mixture.get_linestyle() == "-"
    assert mixture.get_label() == "mixture"
    for line, (w, mu, sd) in zip(lines[1:], [(0.4, -1.0, 0.3), (0.6, 1.0, 0.5)]):
        assert line.get_linestyle() == "--"
        np.testing.assert_allclose(np.ravel(line.get_xdata()), grid, rtol=1e-12)
        np.testing.assert_allclose(np.ravel(line.get_ydata()), w * stats.norm.pdf(grid, mu, sd),
                                   rtol=1e-6, atol=1e-12)
    assert ax.get_xlabel() == r"$\mathrm{log}_{\mathrm{interval}}$ (s)"
    assert ax.get_ylabel() == "Density"
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["mixture"]
    plt.close(fig)


def test_plot_gmm_fit_bin_mean_curve_is_the_density_averaged_over_one_bin():
    """
    In bin-means mode every plotted point ``(x, y)`` must lie inside the
    window and ``y`` must equal the mixture density averaged over a bin-wide
    interval centred on ``x`` -- ``(F(x + w/2) - F(x - w/2)) / w`` with the
    analytic mixture CDF ``F`` and the histogram's bin width ``w`` -- up to
    the grid discretisation, including at the window edges. The curve is one
    solid black line with many points.
    """
    gmm, x = _c_plot_model()
    xlims = (-1.8, 2.2)
    bins = 8
    fig, ax = plot_gmm_fit(gmm, x, bins=bins, xlims=xlims, density_as_bin_means=True)
    line = ax.get_lines()[0]
    gx = np.ravel(line.get_xdata())
    gy = np.ravel(line.get_ydata())
    width = (xlims[1] - xlims[0]) / bins

    def _cdf(v):
        """Analytic mixture CDF."""
        return 0.4 * stats.norm.cdf(v, -1.0, 0.3) + 0.6 * stats.norm.cdf(v, 1.0, 0.5)

    expected = (_cdf(gx + width / 2) - _cdf(gx - width / 2)) / width

    assert gx.size > 1000
    assert gx.min() >= xlims[0] and gx.max() <= xlims[1]
    assert gx.min() - xlims[0] < 0.01 and xlims[1] - gx.max() < 0.01
    assert np.max(np.abs(gy - expected)) < 2e-3 * expected.max()
    assert to_hex(line.get_color()) == "#000000"
    assert line.get_linestyle() == "-"
    plt.close(fig)


def test_qqplot_gmm_points_and_reference_line():
    """
    The Q-Q figure plots, as unconnected dots, the empirical quantiles of the
    data in seconds against the analytic model quantiles in seconds at
    ``n_q`` probabilities evenly spaced over [0.01, 0.99]; for a
    one-component mixture the model quantile is ``exp(mu + sigma
    Phi^-1(q))``. A red dashed identity line spans the empirical range, both
    axes are logarithmic and the labels and title are set.
    """
    gmm = _c_gmm([1.0], [-1.0], [0.25])
    log_data = np.random.default_rng(41).normal(-1.0, 0.5, 400)
    fig = qqplot_gmm(log_data, gmm, n_q=60)
    ax = fig.axes[0]
    points, diagonal = ax.get_lines()
    qs = np.linspace(0.01, 0.99, 60)
    obs_q = np.quantile(np.exp(log_data), qs)

    np.testing.assert_allclose(points.get_xdata(), obs_q, rtol=1e-12)
    np.testing.assert_allclose(points.get_ydata(), np.exp(-1.0 + 0.5 * stats.norm.ppf(qs)), rtol=1e-6)
    assert points.get_marker() == "."
    assert points.get_linestyle() == "None"
    np.testing.assert_allclose(diagonal.get_xdata(), [obs_q.min(), obs_q.max()])
    np.testing.assert_allclose(diagonal.get_ydata(), [obs_q.min(), obs_q.max()])
    assert to_hex(diagonal.get_color()) == "#ff0000"
    assert diagonal.get_linestyle() == "--"
    assert ax.get_xscale() == "log" and ax.get_yscale() == "log"
    assert ax.get_xlabel() == "Observed quantiles (s)"
    assert ax.get_ylabel() == "Model quantiles (s)"
    assert ax.get_title() == "Q-Q (mixture model vs. data)"
    plt.close(fig)


def test_gmm_modes_single_component_with_a_grid_symmetric_about_the_mean():
    """
    Regression test: a one-component GMM must report its single mode even
    when the evaluation grid is symmetric about the component mean.

    :func:`gmm_modes` in 1D spans ``[mu - 4 sd, mu + 4 sd]`` with
    ``np.linspace``. For K=1 that interval is exactly symmetric about the
    mean, and at the default (even) ``grid_size=4096`` the two central grid
    points straddle the mean at equal distance, so their log-densities are
    equal up to floating-point rounding. A strict peak test
    (``>`` on both neighbours) then finds no interior local maximum and the
    function returns an empty mode set, although the unimodal density
    obviously has one mode. The fixed peak test uses ``>=`` against the
    left neighbour and ``>`` against the right, so the left point of the
    tied pair is accepted as the grid peak and the Newton polish lands on
    the analytic mode.

    The model is a K=1 fit (``n_init=1``) to 500 seeded log-normal draws
    (seed 0, log-mean -1, log-sd 0.5). For a single Gaussian the only mode
    is the fitted mean itself, so the test asserts exactly one mode equal to
    ``gmm.means_[0, 0]`` within an absolute tolerance of 1e-8.

    Parameters
    ----------

    Returns
    -------
    """
    rng = np.random.default_rng(0)
    g, _ = fit_log_gmm(np.exp(rng.normal(-1, 0.5, 500)), 1, n_init=1)

    modes, _ = gmm_modes(g)

    modes = np.asarray(modes).ravel()
    assert modes.size == 1
    np.testing.assert_allclose(modes[0], g.means_[0, 0], rtol=0.0, atol=1e-8)


def test_t_mixture_modes_single_component_with_a_grid_symmetric_about_the_location():
    """
    Regression test: a one-component Student-t mixture must report its
    location as its single mode even when the evaluation grid is symmetric
    about that location.

    :func:`t_mixture_modes` spans ``[mu - span_scales * s, mu + span_scales *
    s]`` with ``np.linspace``; for K=1 that interval is symmetric about the
    location ``mu``, and at the default even ``grid_size=16384`` the two
    central grid points have equal log-densities. A strict peak test on both
    neighbours rejected both of them and returned no mode at all. With the
    fixed test (``>=`` on the left neighbour, ``>`` on the right) the left
    point of the tie is kept and the Newton polish converges to the exact
    mode of a single Student-t component, which is its location.

    The model is built by hand (weight 1, location 0.3, squared scale 0.2,
    five degrees of freedom), so the expected answer is exactly ``[0.3]``.

    Parameters
    ----------

    Returns
    -------
    """
    model = TMixture(
        weights=np.array([1.0]),
        means=np.array([0.3]),
        covariances=np.array([0.2]),
        nus=np.array([5.0]),
    )

    modes, _ = t_mixture_modes(model)

    modes = np.asarray(modes).ravel()
    assert modes.size == 1
    np.testing.assert_allclose(modes, np.array([0.3]), rtol=0.0, atol=1e-8)
