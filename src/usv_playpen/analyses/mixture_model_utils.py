"""
@author: bartulem
Utilities for fitting and interpreting 1D mixture models on
log-transformed inter-vocalization intervals (inter-USV intervals).
Two mixture families are supported in this module:

* **Gaussian mixtures** (``GaussianMixture``-backed): see
  :func:`fit_log_gmm`, :func:`gmm_modes`, :func:`gmm_cdf_logspace`,
  :func:`gmm_quantile_logspace`, :func:`report_gmm_stats`,
  :func:`summarize_best_gmm`, :func:`gmm_icl`,
  :func:`gmm_cv_neg_loglik`, :func:`select_best_n_components`.
* **Student-t mixtures** (custom :class:`TMixture` + EM): see
  :func:`fit_log_t_mixture`, :func:`t_mixture_icl`,
  :func:`t_mixture_cv_neg_loglik`, :func:`report_t_mixture_stats`,
  :func:`summarize_best_t_mixture`, :func:`t_mixture_cdf_logspace`,
  :func:`t_mixture_quantile_logspace`.
* **Parametric bootstrap likelihood-ratio test (LRT)** for the number
  of components (family-agnostic, dispatches on ``model_class``): see
  :func:`bootstrap_lrt` and :func:`select_n_components_step_up_lrt`
  (with internal helpers :func:`_sample_from_mixture` and
  :func:`_lr_statistic`).

Functions prefixed ``gmm_`` are Gaussian-only; functions prefixed
``t_mixture_`` are Student-t-only. The two families share the same
EM-style scaffold (fit on ``log(x)``, predict modes, report quantiles)
but the implementation details differ enough that we keep them as
parallel surfaces rather than a generic abstraction.

Conventions
* All fits are performed on ``log(x)`` where ``x`` is in seconds.
* Per-component "mean_k" reported by the helpers below is the mean in
  log-space (``mu_k``). The corresponding *median* of that component in
  seconds is ``exp(mu_k)``; the *mean* of that component in seconds is
  ``exp(mu_k + sigma_k**2 / 2)`` (log-normal). The two should not be
  confused.
* Mixture *modes* are the local maxima of the mixture density itself —
  there is no 1:1 correspondence to components unless the components are
  well separated. Modes and component means are returned by
  :func:`report_gmm_stats` / :func:`report_t_mixture_stats` as
  independent objects.
"""

from __future__ import annotations

import numpy as np
import polars as pls
from scipy.optimize import brentq
from scipy.special import digamma, gammaln
from scipy.stats import invgauss, norm, t as t_dist
from joblib import Parallel, delayed
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold

import matplotlib.pyplot as plt


def fit_log_gmm(
    x: np.ndarray,
    n_components: int,
    seed: int = 0,
    n_init: int = 10,
    reg_covar: float = 1e-4,
) -> tuple[GaussianMixture, np.ndarray]:
    """
    Description
    -----------
    Fits a 1D Gaussian Mixture Model to ``log(x)`` with multi-start EM.

    Multi-start (``n_init=10``) initialisation substantially reduces the
    rate at which EM gets trapped in poor local optima for mixtures with
    >= 4 components on heavy-tailed log-inter-USV interval distributions; this gives
    cleaner BIC sweeps with fewer outer repeats. ``reg_covar`` is set
    above the sklearn default to keep small components from collapsing
    into near-singular covariances on log-space data.

    Parameters
    ----------
    x (np.ndarray)
        A (n_samples,) shape ndarray of strictly positive interval values
        (in seconds). Must contain no zero or negative entries.
    n_components (int)
        Number of mixture components to fit.
    seed (int)
        Random seed forwarded to ``GaussianMixture(random_state=...)``;
        defaults to 0.
    n_init (int)
        Number of EM restarts; the best-likelihood fit is kept. Defaults
        to 10.
    reg_covar (float)
        Regularisation added to component covariances; defaults to 1e-4.

    Returns
    -------
    gmm (GaussianMixture)
        The fitted sklearn ``GaussianMixture`` instance (full covariance,
        in log-space).
    mixture_model_order (np.ndarray)
        A (n_components,) shape ndarray of indices that sorts the
        components by their log-space means in ascending order.
    """

    log_x = np.log(x).reshape(-1, 1)
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type='full',
        random_state=seed,
        n_init=n_init,
        reg_covar=reg_covar,
    )
    gmm.fit(log_x)

    mixture_model_order = np.argsort(gmm.means_.flatten())
    return gmm, mixture_model_order


def plot_gmm_fit(
    model: GaussianMixture,
    x: np.ndarray,
    figsize: tuple = (5, 5),
    bins: int = 100,
    xlims: tuple | None = None,
    color: str = 'steelblue',
    edge_color: str = '#202020',
    histtype: str = 'stepfilled',
    show_components: bool = False,
    legend: bool = True,
    density_as_bin_means: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Description
    -----------
    Plots the histogram of ``x`` (assumed to already be in log-space)
    against the fitted GMM mixture density. Optionally overlays the
    posterior-weighted per-component densities. If ``xlims`` is given the
    histogram and the density grid are restricted to that window — both
    range and per-bin density are computed *after* the cut, so the
    rendered histogram always integrates to 1 within the visible range.

    The figure is returned without being saved; callers that want to
    write it to disk should use
    ``usv_playpen.visualizations.figure_io.save_figure``.

    Parameters
    ----------
    model (GaussianMixture)
        The fitted GMM (in log-space).
    x (np.ndarray)
        A (n_samples,) shape ndarray of log-space samples.
    figsize (tuple)
        Matplotlib figure size; defaults to (5, 5).
    bins (int)
        Number of histogram bins; defaults to 100.
    xlims (tuple)
        Optional (low, high) bounds in log-space for both the histogram
        cut and the density grid; defaults to None.
    color (str)
        Histogram fill colour; defaults to 'steelblue'.
    edge_color (str)
        Histogram edge colour; defaults to '#202020'.
    histtype (str)
        Matplotlib histogram type; defaults to 'stepfilled'.
    show_components (bool)
        If True, overlay each posterior-weighted component density;
        defaults to False.
    density_as_bin_means (bool)
        When True the mixture is drawn as the bar heights it predicts -- its
        density averaged over each histogram bin -- instead of the point-wise
        curve. Bars are bin averages, so this puts model and data on the same
        footing; the point-wise curve necessarily rises above the bars wherever
        a bin straddles a peak. Defaults to False (the point-wise curve).
    legend (bool)
        If True, render the matplotlib legend; defaults to True.

    Returns
    -------
    f (plt.Figure)
        The created figure.
    ax (plt.Axes)
        The axes containing the histogram and density curves.
    """

    if (xlims is not None):
        x = x[(x >= xlims[0]) & (x <= xlims[1])]
        val_min, val_max = xlims
    else:
        val_min, val_max = x.min(), x.max()

    xx = np.linspace(val_min, val_max, 500).reshape(-1, 1)
    logprob = model.score_samples(xx)
    pdf = np.exp(logprob)

    f, ax = plt.subplots(figsize=figsize)
    _, bin_edges, _ = ax.hist(x, bins=bins, density=True, alpha=1, color=color,
                              histtype=histtype, edgecolor=edge_color)
    if density_as_bin_means:
        # A histogram bar is the density AVERAGED over its bin, while a plain
        # mixture curve is the density AT a point. Where a bin straddles the
        # peak the average is necessarily the smaller of the two, so the
        # point-wise curve rides above the bars and reads as an overshoot the
        # fit does not actually have (on the male e2s K=3 fit the apex sat
        # +14% above the tallest bar but -0.1% once averaged over that bin).
        # Plotting the density smoothed by a moving average one bin wide keeps
        # a smooth curve while putting it on the bars' footing: its height at
        # x is what a bar centred on x would show.
        bin_width = float(bin_edges[1] - bin_edges[0])
        pad = bin_width / 2.0
        dense_grid = np.linspace(val_min - pad, val_max + pad, 4000)
        dense_pdf = np.exp(model.score_samples(dense_grid.reshape(-1, 1)))
        window = max(1, int(round(bin_width / (dense_grid[1] - dense_grid[0]))))
        kernel = np.ones(window, dtype=float) / window
        smoothed = np.convolve(dense_pdf, kernel, mode='same')
        inside = (dense_grid >= val_min) & (dense_grid <= val_max)
        ax.plot(dense_grid[inside], smoothed[inside], '-k', lw=2, label='mixture')
    else:
        ax.plot(xx, pdf, '-k', lw=2, label='mixture')

    if show_components:
        responsibilities = model.predict_proba(xx)
        pdf_individual = responsibilities * pdf[:, np.newaxis]
        ax.plot(xx, pdf_individual, '--', lw=1)

    if legend:
        ax.legend()
    ax.set_xlabel(r"$\mathrm{log}_{\mathrm{interval}}$ (s)")
    ax.set_ylabel('Density')

    if (xlims is not None):
        ax.set_xlim(xlims)

    return f, ax


# core helper: boundaries between adjacent components
def gmm_boundaries_logspace(gmm: GaussianMixture, tau: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """
    Description
    -----------
    Computes decision boundaries between adjacent components of a 1D GMM
    fit in log-space. Returns the boundaries in log-space and seconds.

    The boundary between adjacent components ``k`` (left) and ``k+1``
    (right) is the value of ``x`` where the posterior probability of the
    left component equals ``tau``. Setting ``tau = 0.5`` yields the
    standard Bayes boundary (equal posteriors). Increasing ``tau``
    requires *more* evidence before assigning ``x`` to the left
    component, which moves the boundary *toward* the left component and
    makes the left ("short") regime more conservative — i.e. smaller.

    The boundary equation
    ``log(w1 N1(x)) - log(w2 N2(x)) = log((1 - tau) / tau)``
    is solved by folding the ``tau``-dependent constant directly into
    the quadratic constant term ``c``; we do *not* rescale the weights
    (an earlier version did, but with the wrong sign — bumping ``tau``
    moved the boundary the opposite direction from what the docstring
    promised).

    If the discriminant of the quadratic is negative the two weighted
    Gaussians do not cross at this posterior level (one component
    fully dominates the other); the corresponding boundary is returned
    as ``np.nan`` so the caller can skip or warn rather than silently
    receiving a meaningless vertex value.

    Parameters
    ----------
    gmm (GaussianMixture)
        A fitted 1D GMM in log-space.
    tau (float)
        Posterior threshold for the LEFT component at the boundary.
        ``tau = 0.5`` gives the standard Bayes boundary. Defaults to
        0.5.

    Returns
    -------
    boundaries_log (np.ndarray)
        A (n_components - 1,) shape ndarray of boundaries in log-space
        (NaN where the boundary does not exist at the requested
        posterior level).
    boundaries_sec (np.ndarray)
        The same boundaries in seconds (i.e. ``exp(boundaries_log)``).
    """

    mu = gmm.means_.flatten()                    # means in log-space
    sd = np.sqrt(gmm.covariances_.flatten())     # stds in log-space
    w  = gmm.weights_.flatten()

    # sort by mean so "adjacent" makes sense
    order = np.argsort(mu)
    mu, sd, w = mu[order], sd[order], w[order]

    xs = []
    for k in range(len(mu) - 1):
        # Left = k, Right = k+1
        mu1, s1, w1 = mu[k],   sd[k],   w[k]
        mu2, s2, w2 = mu[k + 1], sd[k + 1], w[k + 1]

        a = 1.0 / s2**2 - 1.0 / s1**2
        b = -2.0 * mu2 / s2**2 + 2.0 * mu1 / s1**2
        c = (mu2**2) / s2**2 - (mu1**2) / s1**2 \
            - 2.0 * np.log((w2 * s1) / (w1 * s2)) \
            + 2.0 * np.log((1.0 - tau) / tau)

        if np.isclose(a, 0.0):  # near-equal variances -> linear
            if np.isclose(b, 0.0):
                xs.append(np.nan)
                continue
            x = -c / b
        else:
            disc = b * b - 4 * a * c
            if disc < 0.0:
                xs.append(np.nan)
                continue
            r1 = (-b + np.sqrt(disc)) / (2 * a)
            r2 = (-b - np.sqrt(disc)) / (2 * a)
            # choose the root between the two means (or closest to midpoint)
            mid = 0.5 * (mu1 + mu2)
            candidates = [r for r in (r1, r2) if min(mu1, mu2) <= r <= max(mu1, mu2)]
            x = candidates[0] if candidates else (r1 if abs(r1 - mid) < abs(r2 - mid) else r2)

        xs.append(x)

    xs = np.array(xs)
    return xs, np.exp(xs)  # (log-space, seconds)


# Analytic CDF of a 1D Gaussian Mixture (in log-space)
def gmm_cdf_logspace(x: np.ndarray | float, gmm: GaussianMixture) -> np.ndarray:
    """
    Description
    -----------
    Evaluates the analytic CDF of a fitted 1D GMM at ``x`` (in log-space).
    The CDF is the weighted sum of component Gaussian CDFs.

    Parameters
    ----------
    x (np.ndarray or float)
        Scalar or array of log-space values at which to evaluate the CDF.
    gmm (GaussianMixture)
        A fitted 1D GMM in log-space.

    Returns
    -------
    cdfs (np.ndarray)
        CDF evaluated elementwise at ``x``.
    """

    x = np.asarray(x)
    mu = gmm.means_.flatten()
    sd = np.sqrt(gmm.covariances_.flatten())
    w  = gmm.weights_.flatten()
    # weighted sum of Gaussian CDFs
    cdfs = np.zeros_like(x, dtype=float)
    for m, s, ww in zip(mu, sd, w):
        cdfs += ww * norm.cdf((x - m) / s)
    return cdfs


def gmm_quantile_logspace(q: np.ndarray, gmm: GaussianMixture) -> np.ndarray:
    """
    Description
    -----------
    Inverts the GMM CDF at probabilities ``q`` using Brent's method, so
    quantiles can be obtained without Monte-Carlo simulation. The
    bracket is chosen wide enough to contain all components' tails:
    ``[mu_min - 8 * sigma_max, mu_max + 8 * sigma_max]``.

    Parameters
    ----------
    q (np.ndarray)
        A (n_quantiles,) shape ndarray of probabilities in (0, 1).
    gmm (GaussianMixture)
        A fitted 1D GMM in log-space.

    Returns
    -------
    quantiles_log (np.ndarray)
        A (n_quantiles,) shape ndarray of log-space quantiles.
    """

    q = np.asarray(q, dtype=float).ravel()
    mu = gmm.means_.flatten()
    sd = np.sqrt(gmm.covariances_.flatten())
    w = gmm.weights_.flatten()
    lo = float(mu.min() - 8.0 * sd.max())
    hi = float(mu.max() + 8.0 * sd.max())

    # Pre-extract the per-component (mu, sd, w) arrays ONCE so each brentq
    # evaluation runs the closed-form weighted-Gaussian CDF over those fixed
    # arrays instead of re-calling ``gmm_cdf_logspace`` (which re-flattens
    # ``gmm.means_`` / ``gmm.covariances_`` / ``gmm.weights_`` on every probe).
    # brentq evaluates the objective dozens of times per quantile, so this
    # removes O(n_quantiles * n_brentq_evals) redundant attribute extractions.
    # The summed value is identical to the wrapper up to fp reassociation
    # (~1e-16).
    def _cdf_scalar(v: float) -> float:
        """
        Description
        -----------
        Closed-form mixture CDF at a single log-space point ``v``, evaluated
        over the pre-extracted ``mu`` / ``sd`` / ``w`` component arrays. Used
        as the brentq objective so root-finding never re-extracts the model
        parameters.

        Parameters
        ----------
        v (float)
            Scalar log-space evaluation point.

        Returns
        -------
        cdf (float)
            Weighted sum of component Gaussian CDFs at ``v``.
        """

        return float(np.sum(w * norm.cdf((v - mu) / sd)))

    out = np.empty_like(q)
    for i, qi in enumerate(q):
        # Bind ``qi`` as a default arg so the lambda doesn't capture
        # the loop variable by reference (functionally fine in this
        # synchronous call but ruff B023 flags the pattern as
        # fragile; the bind makes the intent explicit).
        out[i] = brentq(lambda v, qi=qi: _cdf_scalar(v) - qi, lo, hi)
    return out


# Q-Q plot (empirical vs model) using the analytic inverse CDF
def qqplot_gmm(
    log_data: np.ndarray,
    gmm: GaussianMixture,
    n_q: int = 200,
) -> plt.Figure:
    """
    Description
    -----------
    Build a Q-Q plot of empirical quantiles (in seconds) against model
    quantiles (in seconds) on log-log axes. Model quantiles are obtained
    by inverting the analytic GMM CDF, so the plot has no Monte-Carlo
    noise — important because the tails are exactly where one judges
    fit and where simulated empirical quantiles are noisiest.

    The figure is returned without being saved; callers that want to
    write it to disk should use
    ``usv_playpen.visualizations.figure_io.save_figure``.

    Parameters
    ----------
    log_data (np.ndarray)
        A (n_samples,) shape ndarray of log-space samples.
    gmm (GaussianMixture)
        A fitted 1D GMM in log-space.
    n_q (int)
        Number of evenly spaced quantile probabilities between 0.01 and
        0.99; defaults to 200.

    Returns
    -------
    fig (plt.Figure)
        The created Q-Q figure.
    """

    log_data = np.asarray(log_data).ravel()

    qs = np.linspace(0.01, 0.99, n_q)
    obs_q = np.quantile(np.exp(log_data), qs)        # seconds
    model_q = np.exp(gmm_quantile_logspace(qs, gmm)) # seconds

    fig = plt.figure(figsize=(5.5, 4))
    plt.plot(obs_q, model_q, '.', alpha=0.7)
    lo, hi = obs_q.min(), obs_q.max()
    plt.plot([lo, hi], [lo, hi], 'r--', lw=1)
    plt.xscale('log'); plt.yscale('log')
    plt.xlabel('Observed quantiles (s)')
    plt.ylabel('Model quantiles (s)')
    plt.title('Q-Q (mixture model vs. data)')
    plt.tight_layout()
    return fig


def _extract_params(gmm: GaussianMixture):
    """
    Description
    -----------
    Extracts weights, means, full covariance matrices and precision
    matrices from a fitted ``GaussianMixture`` regardless of its
    ``covariance_type`` setting.

    Parameters
    ----------
    gmm (GaussianMixture)
        A fitted GaussianMixture instance.

    Returns
    -------
    w (np.ndarray)
        A (K,) shape ndarray of mixture weights.
    M (np.ndarray)
        A (K, d) shape ndarray of component means.
    Sig (np.ndarray)
        A (K, d, d) shape ndarray of component covariances.
    Prec (np.ndarray)
        A (K, d, d) shape ndarray of component precisions
        (inverses of ``Sig``).
    """

    w = gmm.weights_
    M = gmm.means_
    cov_type = gmm.covariance_type
    if cov_type == "full":
        Sig = gmm.covariances_
    elif cov_type == "tied":
        Sig = np.repeat(gmm.covariances_[None, ...], len(w), axis=0)
    elif cov_type == "diag":
        Sig = np.array([np.diag(v) for v in gmm.covariances_])
    elif cov_type == "spherical":
        Sig = np.array([np.eye(M.shape[1]) * v for v in gmm.covariances_])
    else:
        raise ValueError("Unsupported covariance type")
    # precompute precisions
    Prec = np.array([np.linalg.inv(Sig[k]) for k in range(len(w))])
    return w, M, Sig, Prec


def _log_gauss(x: np.ndarray, mu: np.ndarray, Sig: np.ndarray, Prec: np.ndarray) -> np.ndarray:
    """
    Description
    -----------
    Evaluates the multivariate Gaussian log-density of ``x`` given mean
    ``mu``, covariance ``Sig`` and precision ``Prec``.

    Parameters
    ----------
    x (np.ndarray)
        A (..., d) shape ndarray of evaluation points.
    mu (np.ndarray)
        A (d,) shape ndarray of mean.
    Sig (np.ndarray)
        A (d, d) shape ndarray of covariance.
    Prec (np.ndarray)
        A (d, d) shape ndarray of precision (inverse covariance).

    Returns
    -------
    log_density (np.ndarray)
        A (...,) shape ndarray of log-densities.
    """

    d = x.shape[-1]
    xc = x - mu
    # full quadratic form with precision
    q = np.einsum('...i,ij,...j->...', xc, Prec, xc)
    log_det = np.linalg.slogdet(Sig)[1]
    return -0.5 * (q + log_det + d * np.log(2 * np.pi))


def _alpha(x: np.ndarray, w: np.ndarray, M: np.ndarray, Sig: np.ndarray, Prec: np.ndarray) -> np.ndarray:
    """
    Description
    -----------
    Computes the (unnormalised) responsibilities
    ``alpha_k(x) = w_k N(x | mu_k, Sig_k)`` for each mixture component
    at point ``x``. Numerical stability is preserved by subtracting the
    per-point log-max before exponentiating; the absolute scale does
    not matter for the fixed-point iteration.

    Parameters
    ----------
    x (np.ndarray)
        A (d,) shape ndarray (single point) at which to evaluate the
        responsibilities.
    w (np.ndarray)
        A (K,) shape ndarray of mixture weights.
    M (np.ndarray)
        A (K, d) shape ndarray of component means.
    Sig (np.ndarray)
        A (K, d, d) shape ndarray of component covariances.
    Prec (np.ndarray)
        A (K, d, d) shape ndarray of component precisions.

    Returns
    -------
    a (np.ndarray)
        A (K,) shape ndarray of unnormalised responsibilities.
    """

    logs = np.array([np.log(w[k]) + _log_gauss(x, M[k], Sig[k], Prec[k])
                     for k in range(len(w))])
    # subtract max for numerical stability, then exp
    m = logs.max(axis=0, keepdims=True)
    a = np.exp(logs - m)
    return a  # unnormalized is fine for the fixed-point update


def gmm_modes(
    gmm: GaussianMixture,
    seeds: np.ndarray | None = None,
    tol: float = 1e-7,
    max_iter: int = 300,
    dedup_eps: float = 1e-3,
    grid_size: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Description
    -----------
    Returns the unique modes of a fitted GMM and the mixture density
    evaluated at each mode, sorted by descending density.

    Two implementations are dispatched on dimensionality:

    * 1D (``gmm.means_.shape[1] == 1``): the mixture density is
      evaluated on a dense grid spanning
      ``[mu_min - 4 * sigma_max, mu_max + 4 * sigma_max]`` and local
      maxima are detected as strict interior peaks. Each candidate is
      then polished by one Newton step on the analytic
      ``d/dx log p(x)``. This is robust against saddle-point pitfalls
      that the general-d fixed-point iteration is known to have, and
      is the recommended path for inter-USV interval analysis.
    * General-d (d > 1): the standard Carreira-Perpinan (2000) mean-
      shift-like fixed-point iteration is used (kept verbatim from
      earlier revisions for non-1D callers). Each converged candidate
      is verified to be a local maximum by checking that the mixture
      log-density at the candidate exceeds the log-density at small
      coordinate-wise perturbations; non-maxima are rejected.

    ``dedup_eps`` is the Euclidean tolerance below which two modes are
    considered identical (default 1e-3 in log-space corresponds to
    ~0.1% in seconds, well below any meaningful biological scale).

    Parameters
    ----------
    gmm (GaussianMixture)
        A fitted GaussianMixture instance.
    seeds (np.ndarray)
        Optional (n_seeds, d) shape ndarray of starting points for the
        general-d branch. Ignored in the 1D path. Defaults to component
        means.
    tol (float)
        Convergence tolerance for the fixed-point / Newton update;
        defaults to 1e-7.
    max_iter (int)
        Maximum iterations for the fixed-point / Newton update;
        defaults to 300.
    dedup_eps (float)
        Euclidean distance below which two modes are merged; defaults
        to 1e-3.
    grid_size (int)
        Number of grid points used in the 1D branch; defaults to 4096.

    Returns
    -------
    unique_modes (np.ndarray)
        A (n_modes, d) shape ndarray of mode locations.
    densities (np.ndarray)
        A (n_modes,) shape ndarray of mixture densities at the modes,
        sorted descending.
    """

    w, M, Sig, Prec = _extract_params(gmm)
    K, d = M.shape

    if d == 1:
        # 1D grid + local-max detection + Newton polish
        mu = M.flatten()
        sd = np.sqrt(Sig.flatten())
        lo = float(mu.min() - 4.0 * sd.max())
        hi = float(mu.max() + 4.0 * sd.max())
        grid = np.linspace(lo, hi, grid_size).reshape(-1, 1)
        log_p = gmm.score_samples(grid)

        # strict interior local maxima
        peaks_idx = np.where(
            (log_p[1:-1] > log_p[:-2]) & (log_p[1:-1] > log_p[2:])
        )[0] + 1

        # Loop-invariant across every peak and every Newton iteration
        # (``Sig`` never changes), so compute the per-component inverse
        # variances once instead of re-deriving the identical array inside
        # the Newton loop; the value used downstream is byte-identical.
        inv_var = 1.0 / Sig[:, 0, 0]
        polished = []
        for i in peaks_idx:
            x = float(grid[i, 0])
            for _ in range(max_iter):
                # responsibilities at x (normalized)
                logs = np.array([
                    np.log(w[k])
                    - 0.5 * np.log(Sig[k, 0, 0])
                    - 0.5 * (x - mu[k]) ** 2 / Sig[k, 0, 0]
                    for k in range(K)
                ])
                m = logs.max()
                r = np.exp(logs - m)
                r = r / r.sum()
                # d/dx log p(x) = sum_k r_k * (mu_k - x) / sigma_k^2
                grad = float(np.sum(r * (mu - x) * inv_var))
                # d^2/dx^2 log p(x) (curvature) — Newton step
                term1 = -float(np.sum(r * inv_var))
                centered = (mu - x) * inv_var
                term2 = float(np.sum(r * centered ** 2) - (np.sum(r * centered)) ** 2)
                hess = term1 + term2
                if not np.isfinite(hess) or hess >= 0.0:
                    # not at a local max -> fall back to small gradient step
                    step = grad * 1e-3
                else:
                    step = -grad / hess
                x_new = x + step
                if abs(x_new - x) < tol * (1.0 + abs(x)):
                    x = x_new
                    break
                x = x_new
            polished.append([x])
        modes = np.array(polished) if polished else np.empty((0, 1))
    else:
        # default seeds: component means
        if seeds is None:
            seeds = M.copy()
        seeds = np.atleast_2d(seeds).astype(float)

        modes = []
        for x0 in seeds:
            x = x0.copy()
            for _ in range(max_iter):
                a = _alpha(x, w, M, Sig, Prec)  # shape (K,)
                A = np.zeros((d, d))
                b = np.zeros(d)
                for k in range(K):
                    Ak = a[k] * Prec[k]
                    A += Ak
                    b += Ak @ M[k]
                x_new = np.linalg.solve(A, b)
                if np.linalg.norm(x_new - x) < tol * (1.0 + np.linalg.norm(x)):
                    x = x_new
                    break
                x = x_new
            # verify x is a local max along each axis (reject saddle points)
            base = float(gmm.score_samples(x.reshape(1, -1))[0])
            eps = 1e-4
            is_max = True
            for j in range(d):
                e = np.zeros(d)
                e[j] = eps
                lp_plus = float(gmm.score_samples((x + e).reshape(1, -1))[0])
                lp_minus = float(gmm.score_samples((x - e).reshape(1, -1))[0])
                if lp_plus > base or lp_minus > base:
                    is_max = False
                    break
            if is_max:
                modes.append(x)

        modes = np.array(modes) if modes else np.empty((0, d))

    # deduplicate close modes (cluster by Euclidean distance)
    unique = []
    for m in modes:
        if not unique or min(np.linalg.norm(m - u) for u in unique) > dedup_eps:
            unique.append(m)
    unique = np.array(unique) if unique else np.empty((0, d))

    if unique.shape[0] == 0:
        return unique, np.empty((0,))

    # density at modes
    logp = gmm.score_samples(unique)
    p = np.exp(logp)

    # sort by density (high to low)
    order = np.argsort(-p)
    return unique[order], p[order]


def report_gmm_stats(
    gmm: GaussianMixture,
    mixture_model_order: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Description
    -----------
    Returns the per-component log-space means and standard deviations
    (aligned to ``mixture_model_order``, ascending in mean) and the mixture-level
    modes with their densities (independently sorted in ascending mode
    location).

    NOTE: The returned ``means`` and ``modes`` arrays are NOT
    component-aligned. ``means[i]`` refers to the ``i``-th component
    after sorting; ``modes[j]`` refers to the ``j``-th *mixture* mode
    after sorting. There is no 1:1 correspondence in general — a
    K-component GMM can have fewer than K mixture modes when components
    overlap. The two arrays are returned independently so that callers
    can interpret them correctly.

    Parameters
    ----------
    gmm (GaussianMixture)
        A fitted GMM in log-space.
    mixture_model_order (np.ndarray)
        Sort indices over components (typically from
        :func:`fit_log_gmm`).

    Returns
    -------
    means (np.ndarray)
        A (K,) shape ndarray of component log-space means, ascending.
    sds (np.ndarray)
        A (K,) shape ndarray of component log-space standard deviations,
        aligned to ``means``.
    modes (np.ndarray)
        A (n_modes,) shape ndarray of mixture modes (in log-space),
        ascending.
    densities (np.ndarray)
        A (n_modes,) shape ndarray of mixture densities at ``modes``.
    """

    means = gmm.means_.flatten()[mixture_model_order]
    sds   = np.sqrt(gmm.covariances_.flatten())[mixture_model_order]
    modes_arr, densities_arr = gmm_modes(gmm)
    if modes_arr.shape[0] == 0:
        return means, sds, np.empty((0,)), np.empty((0,))
    modes = modes_arr.flatten()
    mode_order = np.argsort(modes)
    modes = modes[mode_order]
    densities = densities_arr[mode_order]
    return means, sds, modes, densities


def summarize_best_gmm(
    gmm: GaussianMixture,
    mixture_model_order: np.ndarray,
    tau: float = 0.5,
) -> dict:
    """
    Description
    -----------
    Convenience wrapper that bundles per-component means/SDs (log and
    seconds), mixture modes (log and seconds), and inter-component
    boundaries (log and seconds) into a single dictionary, ready for
    tabular display in a notebook.

    The component-level "median in seconds" is reported as
    ``exp(mu_k)``. The component-level "mean in seconds" is the
    log-normal mean ``exp(mu_k + sigma_k**2 / 2)``. Both are reported
    so callers do not have to remember the conversion.

    Parameters
    ----------
    gmm (GaussianMixture)
        A fitted 1D GMM in log-space.
    mixture_model_order (np.ndarray)
        Sort indices over components.
    tau (float)
        Posterior threshold passed to
        :func:`gmm_boundaries_logspace`; defaults to 0.5.

    Returns
    -------
    summary (dict)
        Keys: ``'logmeans'``, ``'logsds'``, ``'medians_sec'``,
        ``'means_sec'``, ``'weights'``, ``'modes_log'``,
        ``'modes_sec'``, ``'mode_densities'``, ``'boundaries_log'``,
        ``'boundaries_sec'``, ``'tau'``.
    """

    logmeans, logsds, modes_log, mode_densities = report_gmm_stats(gmm, mixture_model_order)
    boundaries_log, boundaries_sec = gmm_boundaries_logspace(gmm, tau=tau)
    weights = gmm.weights_.flatten()[mixture_model_order]

    return {
        'logmeans': logmeans,
        'logsds': logsds,
        'medians_sec': np.exp(logmeans),
        'means_sec': np.exp(logmeans + 0.5 * logsds ** 2),
        'weights': weights,
        'modes_log': modes_log,
        'modes_sec': np.exp(modes_log) if modes_log.size else modes_log,
        'mode_densities': mode_densities,
        'boundaries_log': boundaries_log,
        'boundaries_sec': boundaries_sec,
        'tau': tau,
    }


def gmm_icl(gmm: GaussianMixture, log_x: np.ndarray) -> float:
    """
    Description
    -----------
    Integrated Classification Likelihood (ICL) of a fitted GMM at the
    log-space samples ``log_x``. ICL was introduced by Biernacki, Celeux
    & Govaert (2000) specifically because BIC selects the number of
    components that best fits the *density*, while clustering-style
    interpretation (which inter-USV interval bout-regime analysis is) wants components
    that are **well-separated** rather than collectively flexible.

    The standard soft-assignment formulation:

        ICL = BIC + 2 * H(z)
        H(z) = -sum_i sum_k z_ik * log(z_ik)

    where ``z_ik`` is the posterior responsibility of sample ``i`` for
    component ``k``. ``H(z) = 0`` when every sample is assigned crisply
    to a single component, and grows when components overlap. ICL is
    therefore always >= BIC, with the gap widening as separation
    decreases. Lower ICL = better, and ICL prefers fewer / cleaner
    components than BIC whenever the additional components only
    improve the density fit by overlapping existing ones.

    Parameters
    ----------
    gmm (GaussianMixture)
        A fitted GMM in log-space.
    log_x (np.ndarray)
        A (n_samples, d) shape ndarray of log-space samples (i.e.
        already reshaped to a 2D column vector for 1D fits).

    Returns
    -------
    icl (float)
        ICL value (lower is better).
    """

    bic = float(gmm.bic(log_x))
    z = gmm.predict_proba(log_x)
    # avoid log(0); 1e-300 keeps the contribution of a fully-confident
    # assignment numerically zero without producing -inf
    z_safe = np.clip(z, 1e-300, 1.0)
    entropy = float(-np.sum(z_safe * np.log(z_safe)))
    return bic + 2.0 * entropy


def gmm_cv_neg_loglik(
    intervals_sec: np.ndarray,
    n_components: int,
    seed: int = 0,
    n_folds: int = 5,
    n_init: int = 5,
    reg_covar: float = 1e-4,
) -> float:
    """
    Description
    -----------
    K-fold cross-validated negative log-likelihood of a 1D log-GMM at
    ``n_components``, returned in **deviance-like units** so it sits
    on the same scale as BIC / AIC / ICL (lower = better, threshold
    of 10 transfers naturally).

    For each fold the GMM is fitted on the training partition and the
    log-likelihood of the held-out partition is summed; this is
    accumulated across all folds so every sample contributes its
    log-likelihood exactly once. The total is multiplied by ``-2`` to
    convert to a deviance-like quantity.

    Held-out log-likelihood is the only criterion in the inter-USV interval selection
    pipeline that **directly measures generalisation** without
    imposing a clustering prior (BIC / AIC penalise complexity but
    don't enforce separability; ICL enforces separability and
    therefore under-fits when the underlying biology has overlapping
    modes — exactly the failure mode of inter-USV interval distributions, where
    intra-bout and inter-bout regimes shade into each other). CV-LL
    is the recommended default.

    KFold uses ``shuffle=True`` so the partitioning is independent of
    sample order, with ``random_state=seed`` for reproducibility.
    Each fold's fit uses ``n_init`` EM restarts; this is typically
    smaller than the in-sample ``n_init=10`` because folds already
    average out EM noise.

    Parameters
    ----------
    intervals_sec (np.ndarray)
        A (n_samples,) shape ndarray of strictly positive interval
        values (in seconds).
    n_components (int)
        Number of mixture components to fit.
    seed (int)
        Random seed for KFold splitting and EM initialisation;
        defaults to 0.
    n_folds (int)
        Number of CV folds; defaults to 5.
    n_init (int)
        Number of EM restarts per fold's fit; defaults to 5.
    reg_covar (float)
        Regularisation added to component covariances; defaults to
        1e-4.

    Returns
    -------
    cv_neg_loglik (float)
        Cross-validated deviance-equivalent: ``-2 * sum_i loglik(x_i)``
        evaluated on the fold in which sample ``i`` is held out.
        Lower is better. Returns ``np.inf`` if any fold has fewer
        training samples than ``n_components``. Also returns ``np.inf``
        when there are fewer samples than ``n_folds`` (KFold cannot be
        constructed).
    """

    log_x = np.log(intervals_sec).reshape(-1, 1)
    n_samples = log_x.shape[0]
    if n_samples < n_folds:
        return float("inf")

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    total_loglik = 0.0
    for train_idx, test_idx in kf.split(log_x):
        x_train = log_x[train_idx]
        x_test = log_x[test_idx]
        if x_train.shape[0] < n_components:
            return float("inf")
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type="full",
            random_state=seed,
            n_init=n_init,
            reg_covar=reg_covar,
        )
        gmm.fit(x_train)
        # gmm.score returns mean log-likelihood per sample on x_test;
        # accumulate the sum over all held-out samples so every sample
        # contributes once across the K folds.
        total_loglik += float(gmm.score(x_test)) * x_test.shape[0]

    return -2.0 * total_loglik


def select_best_n_components(df_results: pls.DataFrame, ic_col: str = 'bic') -> dict:
    """
    Description
    -----------
    Picks the best ``n_components`` per ``key`` group from a tidy BIC
    sweep DataFrame using the *minimum* information criterion across
    repeats — best init wins. Averaging over repeats blurs "this
    n_components is genuinely worse" with "EM got unlucky on these
    inits", which is why min-across-reps is the right reduction here.

    The DataFrame must have columns ``['key', 'n_comp', ic_col, 'rep']``.

    Parameters
    ----------
    df_results (pls.DataFrame)
        Tidy results from a BIC sweep.
    ic_col (str)
        Name of the information criterion column; defaults to 'bic'.

    Returns
    -------
    best (dict)
        Mapping ``key -> {'n_comp': int, 'rep': int, ic_col: float}``
        for the (n_comp, rep) pair with the lowest IC value within
        each key group.
    """

    cv_mode = (ic_col == 'cv_neg_loglik')

    best: dict = {}
    for key in df_results['key'].unique().to_list():
        sub = df_results.filter(pls.col('key') == key)
        # Pick K by argmin of `ic_col`. For CV the IC is constant per
        # (key, n_comp), so we pick the rep separately by argmin BIC
        # (best in-sample EM init at the chosen K). For BIC/AIC/ICL,
        # the rep is just the row that achieves the min.
        winner_by_ic = sub.sort(ic_col).head(1).row(0, named=True)
        chosen_n = int(winner_by_ic['n_comp'])
        chosen_ic = float(winner_by_ic[ic_col])
        if cv_mode:
            sub_n = sub.filter(pls.col('n_comp') == chosen_n)
            chosen_rep = int(sub_n.sort('bic').head(1).row(0, named=True)['rep'])
        else:
            chosen_rep = int(winner_by_ic['rep'])
        best[key] = {
            'n_comp': chosen_n,
            'rep': chosen_rep,
            ic_col: chosen_ic,
        }
    return best


# =====================================================================
# Student-t mixture model (1D, log-space) — sibling of the log-Gaussian
# code above. Same API surface where it makes sense (BIC / AIC / ICL /
# CV / score_samples / score / predict_proba) so downstream plotting
# and summary helpers can dispatch on the model class without having
# to special-case methods.
#
# The t-mixture is preferred over the log-Gaussian mixture for inter-USV interval
# bout-structure analysis because one heavy-tailed component can absorb
# the long-pause tail in a single component, which a log-Gaussian
# mixture must approximate by stacking several wide Gaussians. See the
# Peel & McLachlan (2000) EM algorithm for the derivation; the per-
# component degrees-of-freedom (``nu``) are updated by 1D root-finding
# each M-step.
# =====================================================================


def _t_logpdf_1d(x: np.ndarray, mu: float, sigma2: float, nu: float) -> np.ndarray:
    """
    Description
    -----------
    log-pdf of a 1D Student-t distribution with mean ``mu``, scale
    ``sigma2``, and degrees of freedom ``nu``, evaluated at ``x``.

    Parameters
    ----------
    x (np.ndarray)
        Evaluation points.
    mu (float)
        Location parameter.
    sigma2 (float)
        Scale parameter (variance for nu -> infinity).
    nu (float)
        Degrees of freedom (>= 2.001 in our EM).

    Returns
    -------
    log_pdf (np.ndarray)
        Elementwise log-density.
    """

    delta = (x - mu) ** 2 / sigma2
    return (
        gammaln((nu + 1.0) / 2.0)
        - gammaln(nu / 2.0)
        - 0.5 * (np.log(np.pi) + np.log(nu) + np.log(sigma2))
        - 0.5 * (nu + 1.0) * np.log1p(delta / nu)
    )


def _t_update_nu(z_k: np.ndarray, u_k: np.ndarray, nu_old: float, n_k: float) -> float:
    """
    Description
    -----------
    Solves the 1D Peel & McLachlan equation for the degrees-of-freedom
    update of one t-component, given current responsibilities and
    latent weights. Brent's method on a wide bracket; falls back to a
    near-Gaussian default of ``nu = 50.0`` when the equation does not
    change sign in the bracket (which happens when the component is
    effectively Gaussian and ``nu`` is unidentifiable above ~50).

    Parameters
    ----------
    z_k (np.ndarray)
        Responsibilities for component k, shape (N,).
    u_k (np.ndarray)
        Latent weights for component k, shape (N,).
    nu_old (float)
        Previous degrees-of-freedom estimate.
    n_k (float)
        Effective sample size for component k (sum of z_k).

    Returns
    -------
    nu_new (float)
        Updated degrees of freedom in [2.001, 200].
    """

    psi_old = digamma((nu_old + 1.0) / 2.0) - np.log((nu_old + 1.0) / 2.0)
    avg = np.sum(z_k * (np.log(u_k) - u_k)) / n_k

    def f(nu):
        """
        Description
        -----------
        Score equation for the EM update of a Student-t component's
        degrees-of-freedom (νₖ). Root corresponds to the ML estimate
        for the new ν given the current responsibilities and latent
        scale variables.

        Parameters
        ----------
        nu (float)
            Candidate degrees-of-freedom value.

        Returns
        -------
        score (float)
            Value of the score equation; root via Brent's method gives
            the updated νₖ.
        """
        return -digamma(nu / 2.0) + np.log(nu / 2.0) + 1.0 + psi_old + avg

    try:
        return brentq(f, 2.001, 200.0)
    except ValueError:
        return 50.0


class TMixture:
    """
    Description
    -----------
    Lightweight 1D Student-t mixture model with a sklearn-compatible
    method surface (``score_samples``, ``score``, ``predict_proba``,
    ``bic``, ``aic``, plus the public attributes ``means_``,
    ``covariances_``, ``weights_``, ``nus_``). Methods accept the same
    2D-column input ``log_x`` of shape ``(N, 1)`` that
    ``GaussianMixture`` does, so downstream code that does
    ``model.bic(log_x)`` etc. works identically.

    The class is fit via :func:`fit_log_t_mixture`; constructing it
    directly from parameter arrays is also supported for testing.
    """

    def __init__(
        self,
        weights: np.ndarray,
        means: np.ndarray,
        covariances: np.ndarray,
        nus: np.ndarray,
    ):
        """
        Description
        -----------
        Construct a `TMixture` directly from fitted parameter arrays.
        Shapes are reshaped to match `sklearn.mixture.GaussianMixture`'s
        1D conventions (`means_` (K, 1), `covariances_` (K, 1, 1)) so
        downstream BIC / AIC / scoring code can stay agnostic.

        Parameters
        ----------
        weights (np.ndarray, shape (K,))
            Component mixing weights; should sum to 1.
        means (np.ndarray, shape (K,) or (K, 1))
            Per-component means.
        covariances (np.ndarray, shape (K,) or (K, 1, 1))
            Per-component variances (1D Student-t scale parameters).
        nus (np.ndarray, shape (K,))
            Per-component degrees-of-freedom.

        Returns
        -------
        None
        """

        self.weights_ = np.asarray(weights, dtype=float).ravel()
        # Match GaussianMixture's shape conventions for 1D / d=1 fits:
        # means_ shape (K, 1); covariances_ shape (K, 1, 1).
        K = self.weights_.size
        self.means_ = np.asarray(means, dtype=float).reshape(K, 1)
        self.covariances_ = np.asarray(covariances, dtype=float).reshape(K, 1, 1)
        self.nus_ = np.asarray(nus, dtype=float).ravel()
        self.n_components = K
        self.covariance_type = "full"

    def _log_w_pdf(self, log_x: np.ndarray) -> np.ndarray:
        """
        Description
        -----------
        Per-component log(w_k) + log p_k(x), shape ``(K, N)``. Internal
        helper used by every other public method.

        Parameters
        ----------
        log_x (np.ndarray)
            Evaluation points, shape ``(N,)`` or ``(N, 1)``.

        Returns
        -------
        log_w_pdf (np.ndarray)
            A (K, N) shape array.
        """

        x = np.asarray(log_x).ravel()
        K = self.n_components
        return np.array([
            np.log(self.weights_[k]) + _t_logpdf_1d(
                x, self.means_[k, 0], self.covariances_[k, 0, 0], self.nus_[k]
            )
            for k in range(K)
        ])

    def score_samples(self, log_x: np.ndarray) -> np.ndarray:
        """
        Description
        -----------
        Per-sample log-likelihood under the mixture, matching
        ``sklearn.mixture.GaussianMixture.score_samples``.

        Parameters
        ----------
        log_x (np.ndarray)
            Evaluation points, shape ``(N,)`` or ``(N, 1)``.

        Returns
        -------
        log_lik (np.ndarray)
            Shape ``(N,)``.
        """

        return np.logaddexp.reduce(self._log_w_pdf(log_x), axis=0)

    def score(self, log_x: np.ndarray) -> float:
        """
        Description
        -----------
        Mean log-likelihood per sample, matching
        ``GaussianMixture.score``.

        Parameters
        ----------
        log_x (np.ndarray)
            Evaluation points.

        Returns
        -------
        mean_log_lik (float)
            Mean of ``score_samples(log_x)``.
        """

        return float(np.mean(self.score_samples(log_x)))

    def predict_proba(self, log_x: np.ndarray) -> np.ndarray:
        """
        Description
        -----------
        Posterior responsibilities ``z_ik = P(component k | x_i)``,
        shape ``(N, K)`` to match
        ``GaussianMixture.predict_proba``.

        Parameters
        ----------
        log_x (np.ndarray)
            Evaluation points.

        Returns
        -------
        z (np.ndarray)
            A (N, K) shape ndarray.
        """

        log_w_pdf = self._log_w_pdf(log_x)              # (K, N)
        log_norm = np.logaddexp.reduce(log_w_pdf, axis=0)  # (N,)
        return np.exp(log_w_pdf - log_norm).T            # (N, K)

    def _n_params(self) -> int:
        """
        Description
        -----------
        Number of free parameters: per component (mu, sigma2, nu, w),
        minus 1 because the weights sum to 1 (one weight is determined
        by the others). Used by :meth:`bic` and :meth:`aic`.

        Parameters
        ----------

        Returns
        -------
        n_params (int)
            ``4K - 1`` for a K-component 1D mixture.
        """

        return 4 * self.n_components - 1

    def bic(self, log_x: np.ndarray) -> float:
        """
        Description
        -----------
        Bayesian Information Criterion, matching
        ``GaussianMixture.bic``: ``-2 * loglik + n_params * log(N)``.

        Parameters
        ----------
        log_x (np.ndarray)
            Evaluation points.

        Returns
        -------
        bic (float)
            Lower is better.
        """

        x = np.asarray(log_x).ravel()
        N = x.size
        log_lik_total = float(np.sum(self.score_samples(x)))
        return -2.0 * log_lik_total + self._n_params() * np.log(N)

    def aic(self, log_x: np.ndarray) -> float:
        """
        Description
        -----------
        Akaike Information Criterion, matching
        ``GaussianMixture.aic``: ``-2 * loglik + 2 * n_params``.

        Parameters
        ----------
        log_x (np.ndarray)
            Evaluation points.

        Returns
        -------
        aic (float)
            Lower is better.
        """

        x = np.asarray(log_x).ravel()
        log_lik_total = float(np.sum(self.score_samples(x)))
        return -2.0 * log_lik_total + 2.0 * self._n_params()


def thin_seam_ladder_surplus(
    intervals_sec: np.ndarray,
    stride_samples: int,
    sampling_rate: int,
    half_width_ms: float,
    max_rung: int,
    seed: int = 0,
) -> np.ndarray:
    """
    Description
    -----------
    Removes the inter-USV intervals a non-overlapping DAS window tiling adds at
    exact multiples of its stride, returning a boolean keep-mask.

    A legacy DAS model that tiled its input without overlap judged each window
    without acoustic context from its neighbours, so the faint edges of calls
    flanking a real pause were clipped to the window seams. Every such pause is
    recorded with a width close to a whole number of strides, which piles a
    surplus of intervals onto a ladder of discrete values -- for the shipped
    model, multiples of 32.512 ms. The surplus is an instrument artifact, and a
    mixture model fitted to the raw intervals will spend a component describing
    it: at ``K = 4`` the male end-to-start distribution otherwise places a
    1.7%-weight component 3 ms from the main peak, on the ladder's second rung.

    Whole rung windows must NOT simply be deleted. The second rung falls inside
    the main peak of the distribution, so cutting a window out of it removes a
    slice of the very feature being measured and the fit responds by straddling
    the hole. Instead each rung window is thinned down to the density of the
    region either side of it: only the measured excess is dropped, at random,
    and the rest is kept, so the peak keeps its shape.

    Parameters
    ----------
    intervals_sec (np.ndarray)
        Strictly positive inter-USV intervals, in seconds.
    stride_samples (int)
        Window-stitching stride (samples) of the model that produced the
        annotations -- the same value the seam repair keys on.
    sampling_rate (int)
        Audio sampling rate (Hz) the stride refers to.
    half_width_ms (float)
        Half-width (ms) of the window taken around each rung.
    max_rung (int)
        Highest stride multiple corrected.
    seed (int)
        Seed for choosing which of a rung's intervals to drop. The components
        recovered are stable across seeds; the seed only fixes which individual
        intervals go.

    Returns
    -------
    keep (np.ndarray)
        Boolean mask over ``intervals_sec``; ``False`` marks an interval
        removed as ladder surplus.
    """

    milliseconds = np.asarray(intervals_sec, dtype=float) * 1000.0
    rung_spacing_ms = (stride_samples / sampling_rate) * 1000.0
    generator = np.random.default_rng(seed)
    keep = np.ones(milliseconds.size, dtype=bool)

    for multiple in range(1, max_rung + 1):
        rung = rung_spacing_ms * multiple
        in_window = np.flatnonzero((milliseconds >= rung - half_width_ms) &
                                   (milliseconds <= rung + half_width_ms))
        if in_window.size == 0:
            continue
        # Local background: equal-width windows offset either side of the rung,
        # far enough not to overlap it but close enough to share its density.
        background = float(np.mean([
            ((milliseconds >= rung + offset - half_width_ms) &
             (milliseconds <= rung + offset + half_width_ms)).sum()
            for offset in (-3.0 * half_width_ms, -2.0 * half_width_ms,
                           2.0 * half_width_ms, 3.0 * half_width_ms)
        ]))
        surplus = int(round(in_window.size - background))
        if surplus <= 0:
            continue
        keep[generator.choice(in_window, size=min(surplus, in_window.size),
                              replace=False)] = False

    return keep


def fit_log_t_mixture(
    x: np.ndarray,
    n_components: int,
    seed: int = 0,
    n_init: int = 5,
    max_iter: int = 300,
    tol: float = 1e-5,
    reg_covar: float = 1e-4,
) -> tuple[TMixture, np.ndarray]:
    """
    Description
    -----------
    Fits a 1D Student-t mixture to ``log(x)`` via the Peel & McLachlan
    (2000) EM algorithm. Best-of-``n_init`` random KMeans initialisations
    is kept (matches the multi-start convention of
    :func:`fit_log_gmm`). Each component's degrees-of-freedom ``nu`` is
    updated each M-step by 1D root-finding (Brent's method on the
    standard ν-update equation).

    Heavy tails are absorbed in single components when ``nu`` is small
    (~3-10); when ``nu`` is large (>= 50) the corresponding component
    is effectively Gaussian. This mixed regime is what lets the
    t-mixture model inter-USV interval distributions with one tail-component plus
    several near-Gaussian peak-components.

    Parameters
    ----------
    x (np.ndarray)
        A (n_samples,) shape ndarray of strictly positive interval
        values (in seconds).
    n_components (int)
        Number of t-components to fit.
    seed (int)
        Random seed forwarded to KMeans / RNG; defaults to 0.
    n_init (int)
        Number of random KMeans initialisations; defaults to 5
        (smaller than ``fit_log_gmm``'s 10 because t-mixture EM is
        more compute-intensive and better-conditioned at moderate K).
    max_iter (int)
        Maximum EM iterations per init; defaults to 300.
    tol (float)
        Convergence tolerance on log-likelihood; defaults to 1e-5.
    reg_covar (float)
        Lower bound on component variance to prevent singular fits;
        defaults to 1e-4 (matches :func:`fit_log_gmm`).

    Returns
    -------
    model (TMixture)
        The best-of-``n_init`` fitted t-mixture.
    order (np.ndarray)
        Indices that sort components by ascending log-mean (the
        analog of :func:`fit_log_gmm`'s ``mixture_model_order``).
    """

    log_x = np.log(np.asarray(x, dtype=float))
    N = log_x.size

    best_model: TMixture | None = None
    best_ll = -np.inf

    for ii in range(n_init):
        # Same restart-diversity scheme as fit_log_ig_mixture: KMeans once,
        # seeded quantile-drawn centers for the remaining restarts (KMeans is
        # seed-insensitive on large samples, collapsing multi-start to one).
        if ii == 0:
            km = KMeans(n_clusters=n_components, random_state=seed, n_init=10).fit(log_x.reshape(-1, 1))
            labels = km.labels_
            mu = km.cluster_centers_.flatten()
        else:
            rng_init = np.random.default_rng((seed, ii))
            qs = np.sort(rng_init.uniform(0.02, 0.98, size=n_components))
            mu = np.quantile(log_x, qs)
            labels = np.argmin(np.abs(log_x[:, None] - mu[None, :]), axis=1)
        sigma2 = np.array([max(np.var(log_x[labels == k]) if np.any(labels == k) else np.var(log_x), reg_covar) for k in range(n_components)])
        nu = np.full(n_components, 10.0)
        w = np.array([np.mean(labels == k) for k in range(n_components)])
        w = np.where(w < 1e-3, 1e-3, w)
        w = w / w.sum()

        prev_ll = -np.inf
        for it in range(max_iter):
            log_w_pdf = np.array([
                np.log(w[k]) + _t_logpdf_1d(log_x, mu[k], sigma2[k], nu[k])
                for k in range(n_components)
            ])
            log_norm = np.logaddexp.reduce(log_w_pdf, axis=0)
            ll = float(np.sum(log_norm))
            z = np.exp(log_w_pdf - log_norm)

            delta = np.array([(log_x - mu[k]) ** 2 / sigma2[k] for k in range(n_components)])
            u = (nu[:, None] + 1.0) / (nu[:, None] + delta)

            n_k = z.sum(axis=1)
            zu = z * u
            n_k_safe = np.maximum(n_k, 1e-10)

            mu_new = (zu * log_x).sum(axis=1) / np.maximum(zu.sum(axis=1), 1e-10)
            sigma2_new = (zu * (log_x - mu_new[:, None]) ** 2).sum(axis=1) / n_k_safe
            sigma2_new = np.maximum(sigma2_new, reg_covar)
            w_new = n_k / N
            nu_new = np.array([_t_update_nu(z[k], u[k], nu[k], n_k_safe[k]) for k in range(n_components)])

            mu, sigma2, nu, w = mu_new, sigma2_new, nu_new, w_new

            if abs(ll - prev_ll) < tol:
                break
            prev_ll = ll

        # ``ll`` above is computed from the parameters at the TOP of the last
        # iteration (i.e. one M-step stale relative to the final mu/sigma2/nu/w).
        # Recompute the log-likelihood of the final parameters so the restart
        # ranking and the stored ``best_ll`` correspond exactly to ``best_model``.
        final_log_w_pdf = np.array([
            np.log(w[k]) + _t_logpdf_1d(log_x, mu[k], sigma2[k], nu[k])
            for k in range(n_components)
        ])
        final_ll = float(np.sum(np.logaddexp.reduce(final_log_w_pdf, axis=0)))

        if final_ll > best_ll:
            best_ll = final_ll
            best_model = TMixture(weights=w, means=mu, covariances=sigma2, nus=nu)

    if best_model is None:
        # Should never happen because n_init >= 1, but be defensive.
        raise RuntimeError("fit_log_t_mixture: EM did not produce a valid fit.")

    order = np.argsort(best_model.means_.flatten())
    return best_model, order


def t_mixture_icl(model: TMixture, log_x: np.ndarray) -> float:
    """
    Description
    -----------
    Integrated Classification Likelihood for a fitted Student-t
    mixture, defined identically to :func:`gmm_icl`:
    ``ICL = BIC + 2 * H(z)`` where ``H(z) = -sum_i sum_k z_ik log z_ik``.
    Lower is better.

    The same caveat applies as for the Gaussian ICL: when the
    underlying biology has overlapping modes (as inter-USV interval distributions
    do), ICL tends to under-fit by penalising ambiguity that is real
    rather than spurious. Use as a diagnostic alongside CV-LL and the
    BIC elbow, not as the sole selector.

    Parameters
    ----------
    model (TMixture)
        A fitted t-mixture.
    log_x (np.ndarray)
        Evaluation points, shape ``(N,)`` or ``(N, 1)``.

    Returns
    -------
    icl (float)
        ICL value.
    """

    bic = model.bic(log_x)
    z = model.predict_proba(log_x)
    z_safe = np.clip(z, 1e-300, 1.0)
    entropy = float(-np.sum(z_safe * np.log(z_safe)))
    return bic + 2.0 * entropy


def t_mixture_cv_neg_loglik(
    intervals_sec: np.ndarray,
    n_components: int,
    seed: int = 0,
    n_folds: int = 5,
    n_init: int = 3,
    reg_covar: float = 1e-4,
) -> float:
    """
    Description
    -----------
    K-fold cross-validated negative log-likelihood for the Student-t
    mixture, deviance-scaled (``-2 * sum_held_out_loglik``) so it sits
    on the same scale as BIC / AIC / ICL: lower is better, the
    Δ-threshold of 10 transfers naturally.

    KFold uses ``shuffle=True`` so partitioning is independent of
    sample order, with ``random_state=seed`` for reproducibility.
    Each fold's fit uses ``n_init`` EM restarts; the in-sample fit
    uses ``fit_log_t_mixture``'s default of 5.

    See :func:`gmm_cv_neg_loglik` for the parallel Gaussian-mixture
    implementation.

    Parameters
    ----------
    intervals_sec (np.ndarray)
        A (n_samples,) shape ndarray of strictly positive intervals.
    n_components (int)
        Number of mixture components.
    seed (int)
        Random seed for KFold + EM init.
    n_folds (int)
        Number of CV folds; defaults to 5.
    n_init (int)
        Number of EM restarts per fold's fit; defaults to 3 (smaller
        than the in-sample default because folds already average out
        EM noise).
    reg_covar (float)
        Component variance floor, defaults to 1e-4.

    Returns
    -------
    cv_neg_loglik (float)
        ``-2 * sum_i loglik(x_i)`` evaluated on the fold in which
        ``x_i`` is held out. ``np.inf`` if any fold has fewer training
        samples than ``n_components``. Also returns ``np.inf`` when
        there are fewer samples than ``n_folds`` (KFold cannot be
        constructed).
    """

    log_x = np.log(np.asarray(intervals_sec, dtype=float))
    N = log_x.size
    if N < n_folds:
        return float("inf")

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    total_loglik = 0.0
    for train_idx, test_idx in kf.split(log_x):
        x_train = np.exp(log_x[train_idx])
        x_test = log_x[test_idx]
        if x_train.size < n_components:
            return float("inf")
        model, _ = fit_log_t_mixture(
            x_train,
            n_components=n_components,
            seed=seed,
            n_init=n_init,
            reg_covar=reg_covar,
        )
        total_loglik += float(np.sum(model.score_samples(x_test)))

    return -2.0 * total_loglik


def report_t_mixture_stats(
    model: TMixture,
    order: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Description
    -----------
    Per-component summary for a fitted Student-t mixture in
    log-space, ordered by ``order`` (typically ascending log-mean):
    log-means, log-scales (sqrt of sigma2), degrees of freedom,
    weights, and the per-component mixture density evaluated at each
    log-mean (the t-component's own mode in log-space).

    The last array is the natural analog of "mode densities" for
    triangle-marker plots: it tells the caller how high the mixture
    curve is at the location of each component's peak, useful for
    placing component-peak triangles on the rendered fit.

    Parameters
    ----------
    model (TMixture)
        A fitted t-mixture.
    order (np.ndarray)
        Indices that sort components in the desired order (typically
        ascending log-mean).

    Returns
    -------
    logmeans (np.ndarray)
        Per-component log-means, shape ``(K,)``.
    logscales (np.ndarray)
        Per-component log-space scales (sqrt of sigma2), shape ``(K,)``.
    nus (np.ndarray)
        Per-component degrees of freedom, shape ``(K,)``.
    weights (np.ndarray)
        Per-component weights, shape ``(K,)``.
    densities_at_means (np.ndarray)
        Mixture density at each component's log-mean, shape ``(K,)``.
    """

    logmeans = model.means_.flatten()[order]
    logscales = np.sqrt(model.covariances_.flatten())[order]
    nus = model.nus_[order]
    weights = model.weights_[order]
    densities_at_means = np.exp(model.score_samples(logmeans))
    return logmeans, logscales, nus, weights, densities_at_means


def t_mixture_cdf_logspace(x: np.ndarray | float, model: TMixture) -> np.ndarray:
    """
    Description
    -----------
    Analytic CDF of a fitted Student-t mixture, evaluated at ``x`` in
    log-space. The mixture CDF is the weighted sum of per-component
    Student-t CDFs.

    Parameters
    ----------
    x (np.ndarray or float)
        Log-space evaluation points.
    model (TMixture)
        Fitted t-mixture.

    Returns
    -------
    cdfs (np.ndarray)
        Elementwise CDF values.
    """

    x_arr = np.asarray(x, dtype=float)
    out = np.zeros_like(x_arr, dtype=float)
    for k in range(model.n_components):
        mu_k = float(model.means_[k, 0])
        sd_k = float(np.sqrt(model.covariances_[k, 0, 0]))
        nu_k = float(model.nus_[k])
        out += model.weights_[k] * t_dist.cdf((x_arr - mu_k) / sd_k, df=nu_k)
    return out


def t_mixture_quantile_logspace(q: np.ndarray, model: TMixture) -> np.ndarray:
    """
    Description
    -----------
    Inverts the t-mixture CDF at probabilities ``q`` via Brent's
    method. Bracket spans 8 sigma either side of the most extreme
    component mean (sigma being the per-component scale, since t
    quantiles match Gaussian quantiles to within ~10% over the bulk
    of probability mass for nu >= 5).

    Used by :func:`plot_qq`'s t-mixture path; analog of
    :func:`gmm_quantile_logspace` for the Gaussian case.

    Parameters
    ----------
    q (np.ndarray)
        Probabilities in (0, 1).
    model (TMixture)
        Fitted t-mixture.

    Returns
    -------
    quantiles_log (np.ndarray)
        Log-space quantiles.
    """

    q = np.asarray(q, dtype=float).ravel()
    mu = model.means_.flatten()
    sd = np.sqrt(model.covariances_.flatten())
    w = model.weights_.flatten()
    nu = model.nus_.flatten()
    lo = float(mu.min() - 8.0 * sd.max())
    hi = float(mu.max() + 8.0 * sd.max())

    # Pre-extract the per-component (mu, sd, w, nu) arrays ONCE so each brentq
    # evaluation runs the closed-form weighted-Student-t CDF over those fixed
    # arrays instead of re-calling ``t_mixture_cdf_logspace`` (which re-flattens
    # the model attributes and rebuilds per-component floats inside a Python loop
    # on every probe). brentq evaluates the objective dozens of times per
    # quantile, so this removes O(n_quantiles * n_brentq_evals) redundant
    # extractions; the summed value is identical to the wrapper up to fp
    # reassociation (~1e-16).
    def _cdf_scalar(v: float) -> float:
        """
        Description
        -----------
        Closed-form mixture CDF at a single log-space point ``v``, evaluated
        over the pre-extracted ``mu`` / ``sd`` / ``w`` / ``nu`` component
        arrays. Used as the brentq objective so root-finding never re-extracts
        the model parameters.

        Parameters
        ----------
        v (float)
            Scalar log-space evaluation point.

        Returns
        -------
        cdf (float)
            Weighted sum of component Student-t CDFs at ``v``.
        """

        return float(np.sum(w * t_dist.cdf((v - mu) / sd, df=nu)))

    out = np.empty_like(q)
    for i, qi in enumerate(q):
        # Bind ``qi`` as a default arg so the lambda doesn't capture
        # the loop variable by reference (see ``gmm_quantile_logspace``
        # for the same defensive pattern).
        out[i] = brentq(
            lambda v, qi=qi: _cdf_scalar(v) - qi, lo, hi
        )
    return out


def summarize_best_t_mixture(
    model: TMixture,
    order: np.ndarray,
) -> dict:
    """
    Description
    -----------
    Convenience wrapper around :func:`report_t_mixture_stats` that
    returns the per-component summary in a notebook-friendly dict.
    Mirrors :func:`summarize_best_gmm` so notebook code can dispatch
    on ``model_class`` without re-shaping the returned object.

    Note that t-component "modes" in log-space are simply the
    per-component means ``mu_k`` (the t-density is symmetric and
    unimodal about its location parameter), so we do not run the
    grid-based mixture-mode finder used for the Gaussian case. The
    triangle markers in the fit plot sit at ``(mu_k,
    mixture_pdf(mu_k))`` for each component.

    Parameters
    ----------
    model (TMixture)
        A fitted Student-t mixture.
    order (np.ndarray)
        Indices that sort components in ascending log-mean.

    Returns
    -------
    summary (dict)
        Keys: ``'logmeans'``, ``'logsds'`` (= log-space scales, named
        for parity with ``summarize_best_gmm``), ``'medians_sec'``
        (= ``exp(mu_k)``), ``'means_sec'`` (= log-normal-style mean,
        approximated as ``exp(mu_k + 0.5 * sigma2_k)`` — note this
        ignores the additional dispersion contributed by the
        degrees-of-freedom and is therefore approximate), ``'nus'``,
        ``'weights'``, ``'modes_log'`` (= ``logmeans``),
        ``'modes_sec'`` (= ``exp(logmeans)``),
        ``'mode_densities'`` (mixture density at each mode),
        ``'boundaries_log'`` (always empty array — t-mixture
        decision boundaries are not currently computed),
        ``'boundaries_sec'`` (always empty array), ``'tau'``
        (passthrough = 0.5 for parity).
    """

    logmeans, logscales, nus, weights, densities_at_means = report_t_mixture_stats(model, order)
    return {
        'logmeans': logmeans,
        'logsds': logscales,
        'medians_sec': np.exp(logmeans),
        'means_sec': np.exp(logmeans + 0.5 * logscales ** 2),
        'nus': nus,
        'weights': weights,
        'modes_log': logmeans,
        'modes_sec': np.exp(logmeans),
        'mode_densities': densities_at_means,
        'boundaries_log': np.empty((0,)),
        'boundaries_sec': np.empty((0,)),
        'tau': 0.5,
    }



def _ig_logpdf_linear(x: np.ndarray, mu: float, lam: float) -> np.ndarray:
    """
    Description
    -----------
    Log-density of the inverse-Gaussian (Wald) distribution
    ``IG(mu, lam)`` evaluated at linear-time points ``x``:

        log p(x) = 0.5 * [log(lam) - log(2 pi) - 3 log(x)]
                   - lam * (x - mu)^2 / (2 mu^2 x)

    The inverse-Gaussian is the first-passage-time distribution of a
    drifting diffusion to a fixed threshold.

    Parameters
    ----------
    x (np.ndarray)
        Strictly positive evaluation points (seconds), any shape.
    mu (float)
        Component mean (seconds).
    lam (float)
        Component shape parameter (seconds); larger = narrower.

    Returns
    -------
    log_pdf (np.ndarray)
        Log-density at ``x``, same shape as ``x``.
    """

    x = np.asarray(x, dtype=float)
    return 0.5 * (np.log(lam) - np.log(2.0 * np.pi) - 3.0 * np.log(x)) \
        - lam * (x - mu) ** 2 / (2.0 * mu ** 2 * x)


def _ig_mode(mu: float, lam: float) -> float:
    """
    Description
    -----------
    Closed-form mode of the inverse-Gaussian distribution:

        mode = mu * [ sqrt(1 + 9 mu^2 / (4 lam^2)) - 3 mu / (2 lam) ]

    Parameters
    ----------
    mu (float)
        Component mean (seconds).
    lam (float)
        Component shape parameter (seconds).

    Returns
    -------
    mode (float)
        The component's linear-time mode (seconds).
    """

    r = mu / lam
    return mu * (np.sqrt(1.0 + 2.25 * r ** 2) - 1.5 * r)


class IGMixture:
    """
    Description
    -----------
    Lightweight 1D inverse-Gaussian (Wald) mixture model with the same
    sklearn-compatible method surface as :class:`TMixture`
    (``score_samples``, ``score``, ``predict_proba``, ``bic``,
    ``aic``). CRITICAL calling convention: like the rest of this
    module, every public method takes **log-space** inputs ``log_x``
    and returns densities **of log X** (the linear-time IG density
    times the Jacobian ``exp(log_x)``), so LR statistics, BIC/AIC/ICL
    and cross-validated log-likelihoods are directly comparable with
    the Gaussian and Student-t families fitted in log-space.

    True parameters live in ``mus_`` (component means, seconds) and
    ``lambdas_`` (shape parameters, seconds). For the family-agnostic
    plot helpers, ``means_`` exposes the per-component LOG-SPACE MODE
    locations (shape ``(K, 1)``) -- the points where each component's
    log-axis density peaks -- and ``covariances_`` carries the
    linear-time component variances ``mu^3 / lam`` (shape
    ``(K, 1, 1)``; informational, not consumed by shared code paths).

    The class is fit via :func:`fit_log_ig_mixture`; constructing it
    directly from parameter arrays is also supported for testing and
    archive reconstruction.
    """

    def __init__(
        self,
        weights: np.ndarray,
        mus: np.ndarray,
        lambdas: np.ndarray,
    ):
        """
        Description
        -----------
        Construct an ``IGMixture`` directly from fitted parameter
        arrays.

        Parameters
        ----------
        weights (np.ndarray, shape (K,))
            Component mixing weights; should sum to 1.
        mus (np.ndarray, shape (K,))
            Per-component inverse-Gaussian means (seconds).
        lambdas (np.ndarray, shape (K,))
            Per-component inverse-Gaussian shape parameters (seconds).

        Returns
        -------
        None
        """

        self.weights_ = np.asarray(weights, dtype=float).ravel()
        self.mus_ = np.asarray(mus, dtype=float).ravel()
        self.lambdas_ = np.asarray(lambdas, dtype=float).ravel()
        K = self.weights_.size
        self.n_components = K
        self.covariance_type = "full"
        modes = np.array([_ig_mode(self.mus_[k], self.lambdas_[k]) for k in range(K)])
        # log-space peak locations for the family-agnostic plot helpers
        self.means_ = np.log(modes).reshape(K, 1)
        self.covariances_ = (self.mus_ ** 3 / self.lambdas_).reshape(K, 1, 1)

    def _log_w_pdf(self, log_x: np.ndarray) -> np.ndarray:
        """
        Description
        -----------
        Per-component ``log(w_k) + log p_k(log_x)`` where ``p_k`` is
        the density of ``log X`` under component k (linear-time IG
        density times the Jacobian ``exp(log_x)``). Shape ``(K, N)``.
        Internal helper used by every other public method.

        Parameters
        ----------
        log_x (np.ndarray)
            Log-space evaluation points, shape ``(N,)`` or ``(N, 1)``.

        Returns
        -------
        log_w_pdf (np.ndarray)
            A (K, N) shape array.
        """

        y = np.asarray(log_x, dtype=float).ravel()
        x = np.exp(y)
        K = self.n_components
        return np.array([
            np.log(self.weights_[k]) + _ig_logpdf_linear(x, self.mus_[k], self.lambdas_[k]) + y
            for k in range(K)
        ])

    def score_samples(self, log_x: np.ndarray) -> np.ndarray:
        """
        Description
        -----------
        Per-sample log-likelihood of ``log X`` under the mixture,
        matching ``sklearn.mixture.GaussianMixture.score_samples``.

        Parameters
        ----------
        log_x (np.ndarray)
            Log-space evaluation points, shape ``(N,)`` or ``(N, 1)``.

        Returns
        -------
        log_lik (np.ndarray)
            Shape ``(N,)``.
        """

        return np.logaddexp.reduce(self._log_w_pdf(log_x), axis=0)

    def score(self, log_x: np.ndarray) -> float:
        """
        Description
        -----------
        Mean log-likelihood per sample, matching
        ``GaussianMixture.score``.

        Parameters
        ----------
        log_x (np.ndarray)
            Log-space evaluation points.

        Returns
        -------
        mean_log_lik (float)
            Mean of ``score_samples(log_x)``.
        """

        return float(np.mean(self.score_samples(log_x)))

    def predict_proba(self, log_x: np.ndarray) -> np.ndarray:
        """
        Description
        -----------
        Posterior responsibilities ``z_ik = P(component k | x_i)``,
        shape ``(N, K)`` to match ``GaussianMixture.predict_proba``.
        Posterior odds are invariant to the (shared) Jacobian term, so
        these equal the linear-space responsibilities.

        Parameters
        ----------
        log_x (np.ndarray)
            Log-space evaluation points.

        Returns
        -------
        z (np.ndarray)
            A (N, K) shape ndarray.
        """

        log_w_pdf = self._log_w_pdf(log_x)
        log_norm = np.logaddexp.reduce(log_w_pdf, axis=0)
        return np.exp(log_w_pdf - log_norm).T

    def _n_params(self) -> int:
        """
        Description
        -----------
        Number of free parameters: per component (mu, lam, w), minus 1
        because the weights sum to 1. Used by :meth:`bic` and
        :meth:`aic`. One parameter FEWER per component than the
        Student-t (no degrees-of-freedom).

        Parameters
        ----------

        Returns
        -------
        n_params (int)
            ``3K - 1`` for a K-component 1D mixture.
        """

        return 3 * self.n_components - 1

    def bic(self, log_x: np.ndarray) -> float:
        """
        Description
        -----------
        Bayesian Information Criterion, matching
        ``GaussianMixture.bic``: ``-2 * loglik + n_params * log(N)``,
        computed in the shared log-space measure so values are
        directly comparable across the module's mixture families.

        Parameters
        ----------
        log_x (np.ndarray)
            Log-space evaluation points.

        Returns
        -------
        bic (float)
            Lower is better.
        """

        x = np.asarray(log_x).ravel()
        N = x.size
        log_lik_total = float(np.sum(self.score_samples(x)))
        return -2.0 * log_lik_total + self._n_params() * np.log(N)

    def aic(self, log_x: np.ndarray) -> float:
        """
        Description
        -----------
        Akaike Information Criterion, matching
        ``GaussianMixture.aic``: ``-2 * loglik + 2 * n_params``.

        Parameters
        ----------
        log_x (np.ndarray)
            Log-space evaluation points.

        Returns
        -------
        aic (float)
            Lower is better.
        """

        x = np.asarray(log_x).ravel()
        log_lik_total = float(np.sum(self.score_samples(x)))
        return -2.0 * log_lik_total + 2.0 * self._n_params()


def fit_log_ig_mixture(
    x: np.ndarray,
    n_components: int,
    seed: int = 0,
    n_init: int = 5,
    max_iter: int = 300,
    tol: float = 1e-5,
    reg_covar: float = 1e-4,
) -> tuple[IGMixture, np.ndarray]:
    """
    Description
    -----------
    Fits a 1D inverse-Gaussian (Wald) mixture to strictly positive
    interval data by EM in linear time. Best-of-``n_init`` KMeans
    (on ``log x``) initialisations is kept, matching the multi-start
    convention of :func:`fit_log_gmm` / :func:`fit_log_t_mixture`.
    Both M-step updates are closed-form (``mu_k`` = responsibility-
    weighted mean; ``1/lam_k`` = weighted mean of ``1/x - 1/mu_k``),
    which makes the IG sweep considerably faster than the Student-t's
    per-component ``nu`` root-finding.

    The signature and return contract mirror
    :func:`fit_log_t_mixture` exactly so the bootstrap-LRT machinery
    (:func:`bootstrap_lrt`, :func:`select_n_components_step_up_lrt`)
    consumes this fitter unchanged.

    Parameters
    ----------
    x (np.ndarray)
        A (n_samples,) shape ndarray of strictly positive interval
        values (in seconds).
    n_components (int)
        Number of IG components to fit.
    seed (int)
        Random seed forwarded to KMeans; defaults to 0.
    n_init (int)
        Number of random KMeans initialisations; defaults to 5.
    max_iter (int)
        Maximum EM iterations per init; defaults to 300.
    tol (float)
        Convergence tolerance on log-likelihood; defaults to 1e-5.
    reg_covar (float)
        Floor on the per-component mean-inverse residual
        ``1/lam`` (i.e. a ceiling ``lam <= 1/reg_covar``), preventing
        a component from collapsing onto a point mass; defaults to
        1e-4 (accepted for interface parity with the other fitters).

    Returns
    -------
    model (IGMixture)
        The best-of-``n_init`` fitted IG mixture.
    order (np.ndarray)
        Indices that sort components by ascending component mean
        ``mu`` (the analog of the other fitters'
        ``mixture_model_order``).
    """

    x = np.asarray(x, dtype=float).ravel()
    log_x = np.log(x)
    N = x.size

    best_model: IGMixture | None = None
    best_ll = -np.inf

    for ii in range(n_init):
        # Restart 0 uses KMeans; later restarts draw component centers from
        # seeded random data quantiles. KMeans on large samples converges to
        # the same centroids for every random_state, so KMeans-only restarts
        # are one restart in disguise -- quantile draws restore genuine
        # multi-start diversity (deterministic per (seed, ii)).
        if ii == 0:
            km = KMeans(n_clusters=n_components, random_state=seed, n_init=10).fit(log_x.reshape(-1, 1))
            labels = km.labels_
        else:
            rng_init = np.random.default_rng((seed, ii))
            qs = np.sort(rng_init.uniform(0.02, 0.98, size=n_components))
            centers = np.quantile(log_x, qs)
            labels = np.argmin(np.abs(log_x[:, None] - centers[None, :]), axis=1)
        mu = np.empty(n_components)
        lam = np.empty(n_components)
        for k in range(n_components):
            xk = x[labels == k] if np.any(labels == k) else x
            m = float(np.mean(xk))
            v = float(np.var(xk)) + 1e-12
            mu[k] = m
            lam[k] = min(m ** 3 / v, 1.0 / reg_covar)
        w = np.array([np.mean(labels == k) for k in range(n_components)])
        w = np.where(w < 1e-3, 1e-3, w)
        w = w / w.sum()

        prev_ll = -np.inf
        for it in range(max_iter):
            log_w_pdf = np.array([
                np.log(w[k]) + _ig_logpdf_linear(x, mu[k], lam[k])
                for k in range(n_components)
            ])
            log_norm = np.logaddexp.reduce(log_w_pdf, axis=0)
            ll = float(np.sum(log_norm))
            z = np.exp(log_w_pdf - log_norm)

            n_k = np.maximum(z.sum(axis=1), 1e-10)
            mu_new = (z * x).sum(axis=1) / n_k
            inv_resid = (z * (1.0 / x)).sum(axis=1) / n_k - 1.0 / np.maximum(mu_new, 1e-12)
            lam_new = 1.0 / np.maximum(inv_resid, reg_covar)
            w_new = n_k / N

            mu, lam, w = mu_new, lam_new, w_new

            if abs(ll - prev_ll) < tol:
                break
            prev_ll = ll

        # ``ll`` above is one M-step stale relative to the final parameters
        # (same convention as fit_log_t_mixture); recompute so restart ranking
        # matches the returned model exactly.
        final_log_w_pdf = np.array([
            np.log(w[k]) + _ig_logpdf_linear(x, mu[k], lam[k])
            for k in range(n_components)
        ])
        final_ll = float(np.sum(np.logaddexp.reduce(final_log_w_pdf, axis=0)))

        if final_ll > best_ll:
            best_ll = final_ll
            best_model = IGMixture(weights=w, mus=mu, lambdas=lam)

    if best_model is None:
        raise RuntimeError("fit_log_ig_mixture: EM did not produce a valid fit.")

    order = np.argsort(best_model.mus_)
    return best_model, order


def ig_mixture_icl(model: IGMixture, log_x: np.ndarray) -> float:
    """
    Description
    -----------
    Integrated Completed Likelihood for a fitted IG mixture:
    BIC plus twice the total assignment entropy, matching
    :func:`gmm_icl` / :func:`t_mixture_icl` so ICL values are
    comparable across the module's mixture families.

    Parameters
    ----------
    model (IGMixture)
        A fitted IG mixture.
    log_x (np.ndarray)
        Log-space sample points.

    Returns
    -------
    icl (float)
        ICL value; lower is better.
    """

    bic = model.bic(log_x)
    z = model.predict_proba(log_x)
    z_safe = np.clip(z, 1e-300, 1.0)
    entropy = float(-np.sum(z_safe * np.log(z_safe)))
    return bic + 2.0 * entropy


def ig_mixture_cv_neg_loglik(
    intervals_sec: np.ndarray,
    n_components: int,
    seed: int = 0,
    n_folds: int = 5,
    n_init: int = 3,
    reg_covar: float = 1e-4,
) -> float:
    """
    Description
    -----------
    K-fold cross-validated negative log-likelihood for the IG
    mixture, deviance-scaled (``-2 * sum_held_out_loglik`` in the
    shared log-space measure) so it sits on the same scale as
    BIC / AIC / ICL: lower is better. Mirrors
    :func:`t_mixture_cv_neg_loglik`.

    Parameters
    ----------
    intervals_sec (np.ndarray)
        Strictly positive interval values (seconds).
    n_components (int)
        Number of IG components.
    seed (int)
        Seed for the shuffled KFold split and per-fold fits.
    n_folds (int)
        Number of CV folds; defaults to 5.
    n_init (int)
        EM restarts per fold; defaults to 3.
    reg_covar (float)
        Forwarded to :func:`fit_log_ig_mixture`.

    Returns
    -------
    cv_neg_loglik (float)
        ``-2 *`` total held-out log-likelihood across folds.
    """

    x = np.asarray(intervals_sec, dtype=float).ravel()
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    total_heldout_ll = 0.0
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(x)):
        model, _ = fit_log_ig_mixture(
            x[train_idx], n_components=n_components,
            seed=seed + fold_idx, n_init=n_init, reg_covar=reg_covar,
        )
        total_heldout_ll += float(np.sum(model.score_samples(np.log(x[test_idx]))))
    return -2.0 * total_heldout_ll


def report_ig_mixture_stats(
    model: IGMixture,
    order: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Description
    -----------
    Per-component summary for a fitted IG mixture, ordered by
    ``order`` (typically ascending component mean): log of the
    component means, shape parameters ``lambda``, weights, log-space
    component MODE locations, and the mixture log-axis density at
    each mode (for placing component-peak triangles on rendered
    fits). The structural analog of :func:`report_t_mixture_stats`.

    Parameters
    ----------
    model (IGMixture)
        A fitted IG mixture.
    order (np.ndarray)
        Indices that sort components in the desired order (typically
        ascending ``mu``).

    Returns
    -------
    log_mus (np.ndarray)
        Log of per-component means ``mu`` (seconds), shape ``(K,)``.
    lambdas (np.ndarray)
        Per-component shape parameters (seconds), shape ``(K,)``.
    weights (np.ndarray)
        Per-component weights, shape ``(K,)``.
    modes_log (np.ndarray)
        Log-space component mode locations, shape ``(K,)``.
    densities_at_modes (np.ndarray)
        Mixture log-axis density at each component's mode, shape
        ``(K,)``.
    """

    log_mus = np.log(model.mus_[order])
    lambdas = model.lambdas_[order]
    weights = model.weights_[order]
    modes_log = model.means_.flatten()[order]
    densities_at_modes = np.exp(model.score_samples(modes_log))
    return log_mus, lambdas, weights, modes_log, densities_at_modes


def summarize_best_ig_mixture(
    model: IGMixture,
    order: np.ndarray,
) -> dict:
    """
    Description
    -----------
    Convenience wrapper around :func:`report_ig_mixture_stats` that
    returns the per-component summary in the notebook-friendly dict
    shape shared by :func:`summarize_best_gmm` /
    :func:`summarize_best_t_mixture`, so plotting code can dispatch
    on the model class without re-shaping the returned object.
    ``logmeans`` carries ``log(mu_k)`` -- the log-space component MEAN
    locations -- so the figure's triangle anchors and text legend state
    the same values as the archive columns (``logmean_k``) and any
    reported component means. The skewed IG component's mode lies below
    its mean; the mode locations remain available under ``modes_log``.
    Boundaries are left empty, matching the Student-t convention (no
    closed form; not rendered).

    Parameters
    ----------
    model (IGMixture)
        A fitted IG mixture.
    order (np.ndarray)
        Component ordering (typically ascending ``mu``).

    Returns
    -------
    summary (dict)
        Keys ``logmeans``, ``logsds``, ``weights``, ``medians_sec``,
        ``modes_log``, ``densities``, ``boundaries_log``,
        ``boundaries_sec``.
    """

    log_mus, lambdas, weights, modes_log, densities = report_ig_mixture_stats(model, order)
    return {
        "logmeans": log_mus,
        "logsds": np.sqrt(np.log1p(model.mus_[order] / lambdas)),
        "weights": weights,
        "medians_sec": model.mus_[order],
        "modes_log": modes_log,
        "densities": densities,
        "boundaries_log": np.array([]),
        "boundaries_sec": np.array([]),
    }


def ig_mixture_cdf_logspace(x: np.ndarray | float, model: IGMixture) -> np.ndarray:
    """
    Description
    -----------
    Mixture CDF evaluated at LOG-space points, matching the calling
    convention of :func:`gmm_cdf_logspace` /
    :func:`t_mixture_cdf_logspace`. Uses the closed-form IG CDF

        F(t) = Phi(sqrt(lam/t) (t/mu - 1))
             + exp(2 lam / mu) * Phi(-sqrt(lam/t) (t/mu + 1))

    with the second term evaluated in log-space
    (``exp(2 lam/mu + log Phi(.))``) so large ``lam/mu`` ratios do not
    overflow.

    Parameters
    ----------
    x (np.ndarray | float)
        Log-space evaluation points.
    model (IGMixture)
        A fitted IG mixture.

    Returns
    -------
    cdf (np.ndarray)
        Mixture CDF values in ``[0, 1]``, shape like ``x``.
    """

    y = np.atleast_1d(np.asarray(x, dtype=float))
    t = np.exp(y)
    out = np.zeros_like(t)
    for k in range(model.n_components):
        mu = model.mus_[k]
        lam = model.lambdas_[k]
        rt = np.sqrt(lam / t)
        term1 = norm.cdf(rt * (t / mu - 1.0))
        term2 = np.exp(2.0 * lam / mu + norm.logcdf(-rt * (t / mu + 1.0)))
        out += model.weights_[k] * (term1 + term2)
    return np.clip(out, 0.0, 1.0)


def ig_mixture_quantile_logspace(q: np.ndarray, model: IGMixture) -> np.ndarray:
    """
    Description
    -----------
    Quantile function (inverse CDF) of the IG mixture, returned in
    LOG-space to match :func:`gmm_quantile_logspace` /
    :func:`t_mixture_quantile_logspace`. Solved by bracketed
    root-finding (Brent) on :func:`ig_mixture_cdf_logspace`.

    Parameters
    ----------
    q (np.ndarray)
        Quantile levels in ``(0, 1)``.
    model (IGMixture)
        A fitted IG mixture.

    Returns
    -------
    log_q (np.ndarray)
        Log-space quantiles, shape like ``q``.
    """

    q = np.atleast_1d(np.asarray(q, dtype=float))
    lo = float(np.log(model.mus_.min()) - 20.0)
    hi = float(np.log(model.mus_.max()) + 20.0)

    def _cdf_scalar(v: float) -> float:
        return float(ig_mixture_cdf_logspace(v, model)[0])

    out = np.empty(q.size)
    for i, qi in enumerate(q):
        out[i] = brentq(lambda v: _cdf_scalar(v) - qi, lo, hi, xtol=1e-10)
    return out


# =====================================================================
# Parametric bootstrap likelihood-ratio test (LRT) for the number of
# mixture components.
#
# References:
#   - McLachlan (1987) "On bootstrapping the likelihood ratio test
#     statistic for the number of components in a normal mixture",
#     Applied Statistics 36(3), 318-324.
#   - McLachlan & Peel (2000), "Finite Mixture Models", Ch. 6.
#
# The standard chi-squared LRT does not apply to mixture-component
# testing because the null hypothesis "K = k" puts the extra
# component's weight on the boundary (w_{k+1} = 0), violating Wilks'
# theorem. The parametric bootstrap fixes this by simulating the null
# distribution of -2 log Lambda directly, then comparing the observed
# value to the simulated reference.
#
# For large datasets we subsample to ``n_subsample`` so the observed and
# bootstrap LR statistics are on the same N scale; this keeps the
# computation tractable while preserving validity (the LRT is asymptotic
# and converges for any sufficiently large ``n_subsample``).
# =====================================================================


def _sample_from_mixture(model, N: int, rng: np.random.Generator) -> np.ndarray:
    """
    Description
    -----------
    Draws ``N`` 1D samples from a fitted mixture in log-space.
    Dispatches on model type (``IGMixture`` / ``TMixture`` / sklearn's
    ``GaussianMixture``); all are sampled by first drawing a
    component assignment from the categorical(weights) distribution,
    then drawing from the chosen component's distribution.

    Sampling explicitly via ``rng`` (instead of the model's built-in
    sample method) keeps the random stream under the caller's control
    and allows reproducible bootstrap runs.

    Parameters
    ----------
    model
        A fitted ``TMixture`` or ``sklearn.mixture.GaussianMixture``.
    N (int)
        Number of samples.
    rng (np.random.Generator)
        Random number generator.

    Returns
    -------
    log_samples (np.ndarray)
        A (N,) shape ndarray of samples in log-space.
    """

    K = int(model.n_components)
    # Renormalize before sampling: TMixture weights are stored straight from
    # the final M-step (w = n_k / N) without an explicit renormalization, so
    # their sum is 1 only up to float accumulation. numpy's rng.choice rejects
    # p whose sum drifts by more than ~1.49e-8, which would otherwise abort a
    # bootstrap replicate; renormalizing makes the draw robust.
    weights = np.asarray(model.weights_).flatten()
    weights = weights / weights.sum()
    comp = rng.choice(K, size=N, p=weights)
    out = np.empty(N)

    if isinstance(model, IGMixture):
        for k in range(K):
            idx = comp == k
            n_k = int(idx.sum())
            if n_k == 0:
                continue
            # scipy's invgauss(mu_s, scale=s) has mean mu_s * s and shape s,
            # so (mu_s = mu/lam, scale = lam) yields IG(mu, lam) draws.
            draws = invgauss.rvs(
                float(model.mus_[k] / model.lambdas_[k]),
                scale=float(model.lambdas_[k]),
                size=n_k,
                random_state=rng,
            )
            out[idx] = np.log(draws)
    elif isinstance(model, TMixture):
        for k in range(K):
            idx = comp == k
            n_k = int(idx.sum())
            if n_k == 0:
                continue
            out[idx] = t_dist.rvs(
                df=float(model.nus_[k]),
                loc=float(model.means_[k, 0]),
                scale=float(np.sqrt(model.covariances_[k, 0, 0])),
                size=n_k,
                random_state=rng,
            )
    else:
        means = model.means_.flatten()
        sds = np.sqrt(model.covariances_.flatten())
        for k in range(K):
            idx = comp == k
            n_k = int(idx.sum())
            if n_k == 0:
                continue
            out[idx] = rng.normal(loc=float(means[k]), scale=float(sds[k]), size=n_k)

    return out


def _lr_statistic(model_null, model_alt, log_x: np.ndarray) -> float:
    """
    Description
    -----------
    Computes the likelihood-ratio test statistic
    ``LR = -2 * (logL_null - logL_alt)``.

    Both models must implement ``score_samples`` (the package's
    :class:`TMixture` and sklearn's ``GaussianMixture`` both do).
    The 2D-column reshape handles ``GaussianMixture``'s expectation
    of ``(N, 1)`` input while remaining compatible with TMixture's
    polymorphic ``score_samples``.

    Parameters
    ----------
    model_null
        Fitted mixture under H0.
    model_alt
        Fitted mixture under H1.
    log_x (np.ndarray)
        Log-space sample points.

    Returns
    -------
    lr (float)
        ``-2 * (logL_null - logL_alt)``. Higher = more evidence for
        H1.
    """

    log_x_2d = np.asarray(log_x).reshape(-1, 1)
    ll_null = float(np.sum(model_null.score_samples(log_x_2d)))
    ll_alt = float(np.sum(model_alt.score_samples(log_x_2d)))
    return -2.0 * (ll_null - ll_alt)



def _bootstrap_lrt_replicate(
    b: int,
    seed: int,
    model_null_obs,
    fit_fn,
    K_null: int,
    K_alt: int,
    N_sub: int,
    n_init_boot: int,
    reg_covar: float,
) -> float:
    """
    Description
    -----------
    One parametric-bootstrap LRT replicate, self-contained for parallel
    execution: draws a null sample of size ``N_sub`` from the fitted
    null model using a per-replicate deterministic RNG seeded by
    ``(seed, b)``, refits null and alternative models with fit seed
    ``seed + b`` (identical to the sequential path's fit seeds), and
    returns the replicate's LR statistic.

    Parameters
    ----------
    b (int)
        Replicate index.
    seed (int)
        Base seed of the enclosing :func:`bootstrap_lrt` call.
    model_null_obs
        The observed-data null-model fit to resample from.
    fit_fn (Callable)
        Family fitter (``fit_log_gmm`` / ``fit_log_t_mixture`` /
        ``fit_log_ig_mixture``).
    K_null (int)
        Null component count.
    K_alt (int)
        Alternative component count.
    N_sub (int)
        Sample size per replicate.
    n_init_boot (int)
        EM restarts per replicate fit.
    n_jobs (int)
        Number of parallel workers for the bootstrap replicates;
        defaults to 1 (the legacy sequential path, bit-identical to
        prior runs). With ``n_jobs > 1`` each replicate uses its own
        deterministic RNG seeded by ``(seed, b)``.
    reg_covar (float)
        Forwarded to the fitter.

    Returns
    -------
    lr (float)
        The replicate's ``-2 * (logL_null - logL_alt)`` statistic.
    """

    rng_b = np.random.default_rng((seed, b))
    log_x_b = _sample_from_mixture(model_null_obs, N_sub, rng_b)
    x_b = np.exp(log_x_b)
    m_null_b, _ = fit_fn(x_b, K_null, seed=seed + b, n_init=n_init_boot, reg_covar=reg_covar)
    m_alt_b, _ = fit_fn(x_b, K_alt, seed=seed + b, n_init=n_init_boot, reg_covar=reg_covar)
    return _lr_statistic(m_null_b, m_alt_b, log_x_b)


def bootstrap_lrt(
    intervals_sec: np.ndarray,
    K_null: int,
    K_alt: int,
    B: int = 50,
    n_subsample: int = 15000,
    model_class: str = "t",
    n_init_obs: int = 10,
    n_init_boot: int = 3,
    n_jobs: int = 1,
    reg_covar: float = 1e-4,
    seed: int = 0,
    message_output=None,
) -> dict:
    """
    Description
    -----------
    Parametric bootstrap likelihood-ratio test for ``H0: K = K_null``
    vs ``H1: K = K_alt`` (with ``K_alt > K_null``). Returns the
    observed LR statistic, the bootstrap null distribution, and the
    one-sided p-value.

    Procedure (McLachlan 1987; McLachlan & Peel 2000 Ch. 6):
      1. Subsample ``n_subsample`` points from ``intervals_sec`` so
         that observed and bootstrap statistics are on the same N
         scale. The test is asymptotically valid for any sufficiently
         large ``n_subsample``.
      2. Fit ``K_null`` and ``K_alt`` mixtures on the subsample.
         Compute LR_obs = -2 * (logL_null - logL_alt).
      3. For ``b = 1..B``: simulate a synthetic dataset of size
         ``n_subsample`` from the fitted ``K_null`` model
         (parametric bootstrap), refit both ``K_null`` and ``K_alt``,
         compute LR_b on the synthetic data.
      4. p_value = fraction of LR_b values that meet or exceed
         LR_obs. Small p_value rejects ``K_null`` in favor of
         ``K_alt``.

    The dispatch on ``model_class`` (``'gauss'``, ``'t'`` or ``'ig'``) selects
    the fit / sampling functions but the procedure is identical.

    Parameters
    ----------
    intervals_sec (np.ndarray)
        A (n_samples,) ndarray of strictly positive interval values
        in seconds.
    K_null (int)
        Number of components under H0.
    K_alt (int)
        Number of components under H1 (must be > ``K_null``).
    B (int)
        Number of bootstrap replicates; defaults to 50. Standard
        practice is 50-200; B=50 gives a p-value standard error of
        ~0.07.
    n_subsample (int)
        Subsample size used for both observed fit and each bootstrap
        replicate; defaults to 15000. If larger than
        ``intervals_sec.size`` the full data is used.
    model_class (str)
        ``'gauss'`` for log-Gaussian mixture, ``'t'`` for log-Student-t
        mixture. Defaults to ``'t'``.
    n_init_obs (int)
        Number of EM restarts for the observed fits; defaults to 10.
    n_init_boot (int)
        Number of EM restarts for each bootstrap fit; defaults to 3.
    reg_covar (float)
        Component variance floor.
    seed (int)
        Random seed for reproducibility.
    message_output (callable)
        Optional logging callable for progress messages; if None,
        progress is silent.

    Returns
    -------
    result (dict)
        Keys: ``'K_null'``, ``'K_alt'``, ``'B'``, ``'n_subsample'``,
        ``'lr_obs'``, ``'lr_null'`` (length-B array),
        ``'p_value'``, ``'null_mean'``, ``'null_p95'``,
        ``'null_max'``, ``'model_class'``.
    """

    if K_alt <= K_null:
        raise ValueError(
            f"bootstrap_lrt requires K_alt > K_null, got K_null={K_null}, K_alt={K_alt}."
        )
    if model_class == "gauss":
        fit_fn = fit_log_gmm
    elif model_class == "t":
        fit_fn = fit_log_t_mixture
    elif model_class == "ig":
        fit_fn = fit_log_ig_mixture
    else:
        raise ValueError(
            f"bootstrap_lrt: model_class must be 'gauss', 't' or 'ig', got {model_class!r}."
        )

    rng = np.random.default_rng(seed)
    intervals_sub = (
        rng.choice(intervals_sec, size=n_subsample, replace=False)
        if intervals_sec.size > n_subsample
        else np.asarray(intervals_sec, dtype=float)
    )
    log_x = np.log(intervals_sub)
    N_sub = log_x.size

    # Observed LR
    m_null_obs, _ = fit_fn(intervals_sub, K_null, seed=seed, n_init=n_init_obs, reg_covar=reg_covar)
    m_alt_obs, _ = fit_fn(intervals_sub, K_alt, seed=seed, n_init=n_init_obs, reg_covar=reg_covar)
    lr_obs = _lr_statistic(m_null_obs, m_alt_obs, log_x)
    # A materially negative observed LR is impossible for correctly optimized
    # nested mixtures (K_alt can always emulate K_null): it means the
    # alternative's EM landed in a worse local optimum, which would silently
    # bias the step-up selection toward K_null. Fail loud instead.
    if lr_obs < -1e-6 * abs(float(np.sum(m_null_obs.score_samples(log_x.reshape(-1, 1))))):
        msg = (
            f"bootstrap_lrt: observed LR is negative ({lr_obs:.2f}) for "
            f"K_null={K_null} vs K_alt={K_alt} -- the K_alt fit is worse than "
            "the nested K_null fit, indicating an EM optimization failure; "
            "increase n_init_obs or fix the initialization."
        )
        raise RuntimeError(msg)

    # Bootstrap null distribution. n_jobs == 1 preserves the legacy sequential
    # path bit-for-bit (one shared resampling RNG stream); n_jobs > 1 runs the
    # replicates in parallel with per-replicate deterministic RNGs seeded by
    # (seed, b) -- statistically equivalent and fully reproducible for a given
    # (seed, B), but not bit-identical to the sequential stream.
    lr_null = np.empty(B)
    if n_jobs == 1:
        for b in range(B):
            log_x_b = _sample_from_mixture(m_null_obs, N_sub, rng)
            x_b = np.exp(log_x_b)
            m_null_b, _ = fit_fn(x_b, K_null, seed=seed + b, n_init=n_init_boot, reg_covar=reg_covar)
            m_alt_b, _ = fit_fn(x_b, K_alt, seed=seed + b, n_init=n_init_boot, reg_covar=reg_covar)
            lr_null[b] = _lr_statistic(m_null_b, m_alt_b, log_x_b)
            if message_output is not None and (b + 1) % 10 == 0:
                message_output(f"      bootstrap [{b + 1}/{B}]")
    else:
        if message_output is not None:
            message_output(f"      bootstrap: {B} replicates across {n_jobs} workers")
        lr_list = Parallel(n_jobs=n_jobs)(
            delayed(_bootstrap_lrt_replicate)(
                b, seed, m_null_obs, fit_fn, K_null, K_alt, N_sub, n_init_boot, reg_covar,
            )
            for b in range(B)
        )
        lr_null[:] = np.asarray(lr_list, dtype=float)

    return {
        "K_null": int(K_null),
        "K_alt": int(K_alt),
        "B": int(B),
        "n_subsample": int(N_sub),
        "model_class": model_class,
        "lr_obs": float(lr_obs),
        "lr_null": lr_null,
        "p_value": float(np.mean(lr_null >= lr_obs)),
        "null_mean": float(lr_null.mean()),
        "null_p95": float(np.percentile(lr_null, 95)),
        "null_max": float(lr_null.max()),
    }


def select_n_components_step_up_lrt(
    pair_results: dict,
    alpha: float = 0.05,
) -> int:
    """
    Description
    -----------
    Step-up sequential selection from a dict of pairwise LRT
    results: walks from low K to high K, stopping at the first
    ``(K, K+1)`` test whose p-value fails to reject H0 (i.e. K+1
    does not significantly improve on K). The selected K is the
    smallest K for which the next step is non-significant.

    If every test rejects, the largest K_alt seen is returned.
    If no tests are present, raises ValueError.

    Parameters
    ----------
    pair_results (dict)
        Mapping ``(K_null, K_alt) -> result_dict``, where
        ``result_dict`` has key ``'p_value'``. Typically produced by
        running :func:`bootstrap_lrt` over a series of consecutive
        pairs.
    alpha (float)
        Significance level for rejecting H0; defaults to 0.05.

    Returns
    -------
    K_selected (int)
        The selected number of components.
    """

    if not pair_results:
        raise ValueError("select_n_components_step_up_lrt: no pair_results supplied.")
    pairs_sorted = sorted(pair_results.keys())
    for (K_null, K_alt) in pairs_sorted:
        if pair_results[(K_null, K_alt)]["p_value"] >= alpha:
            return int(K_null)
    # All tests rejected — go to the highest K_alt
    return int(max(K_alt for (_, K_alt) in pairs_sorted))
