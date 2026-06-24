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
from sklearn.mixture import GaussianMixture

# Headless matplotlib in case any imported helper touches a backend.
matplotlib.use("Agg")

from usv_playpen.analyses.mixture_model_utils import (
    TMixture,
    _sample_from_mixture,
    _t_update_nu,
    bootstrap_lrt,
    gmm_boundaries_logspace,
    gmm_modes,
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
