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
from unittest.mock import MagicMock

import matplotlib
import numpy as np
import pytest
from numpy.linalg import cholesky, inv
from scipy import stats
from sklearn.mixture import GaussianMixture

# Headless matplotlib in case any imported helper touches a backend.
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from usv_playpen.analyses.mixture_model_utils import (
    IGMixture,
    TMixture,
    _sample_from_mixture,
    _session_design_effect,
    _t_update_nu,
    bootstrap_lrt,
    fit_log_gmm,
    fit_log_ig_mixture,
    fit_log_t_mixture,
    fit_tied_nested_by_splitting,
    fit_tied_scale_t_mixture,
    gmm_boundaries_logspace,
    gmm_modes,
    ig_mixture_cdf_logspace,
    ig_mixture_quantile_logspace,
    plot_gmm_fit,
    stable_subsample,
    t_mixture_modes,
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
    Drives the ``brentq`` ValueError fallback in :func:`_t_update_nu`
    (source lines 1109-1110).

    The degrees-of-freedom update solves a score equation via Brent's method
    on the bracket ``[2.001, 200.0]``. When the score function does not change
    sign across that bracket ``brentq`` raises ``ValueError`` and the function
    must return the conservative default of ``50.0``. A degenerate
    responsibility / latent-scale configuration (all-zero weights and unit
    latent scales) yields a monotone score with no sign change in the bracket.
    """
    n_components = 1
    z = np.zeros((n_components, 10))      # zero responsibilities
    u = np.ones((n_components, 10))       # unit latent scales
    nu = np.array([10.0])
    n_k = np.array([1e-10])               # near-zero effective count

    out = _t_update_nu(z[0], u[0], nu[0], n_k[0])

    assert out == 50.0


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
    from scipy.stats import invgauss
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


# ===========================================================================
# stable_subsample — the value-keyed draw used by bootstrap_lrt
# ===========================================================================


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
        float(np.mean(corrected["lr_null"] >= corrected["lr_corrected"])))
    assert corrected["p_value_corrected"] >= corrected["p_value"]

    with pytest.raises(ValueError, match="session_labels has"):
        bootstrap_lrt(pool, **kwargs, session_labels=labels[:-1], n_design_bootstrap=10)
