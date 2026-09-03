"""
@author: bartulem
Group-elastic-net + temporal-smoothness GLM solver (Bernoulli / Poisson) via FISTA.

SCRATCHPAD PROTOTYPE for the P2 claim-1 backend (`neural_modeling.jax_group_elastic_net`).
Validated by the companion synthetic gate before promotion into the subpackage.

The estimator learns a weight vector `w` of shape `(n_features * n_time_bins,)` and a
scalar bias `b`; the linear predictor is `eta = X @ w + b + offset`, mapped to a rate by
the family link (Bernoulli/logistic or Poisson/log). The objective is

    min_{w, b}   (1/sum wgt) * sum_i wgt_i * NLL_family(eta_i, y_i)          # data term
               + 0.5 * lam_smooth * || D_order  reshape(w, n_feat, n_time) ||_2^2   # temporal smoothness
               + 0.5 * lam_ridge  * ||w||_2^2                                # ridge
               + lam_group * sum_f || w_f ||_2                               # group-lasso (feature sparsity)

where `w_f` is one feature's length-`n_time_bins` temporal filter (one group) and `D_order`
is the `order`-th finite-difference operator along the time axis. The first three terms are
smooth and convex; the group-lasso is non-smooth, so the solver is proximal-gradient
(accelerated / FISTA) with a per-group soft-threshold prox — NOT plain Adam, which cannot
produce exact group sparsity.

Design-matrix column contract
-----------------------------
Columns are FEATURE-major, lag-minor: `[feat0_lag0 .. feat0_lag(T-1), feat1_lag0 .., ...]`,
so `w.reshape(n_features, n_time_bins)` recovers per-feature temporal filters and each row is
one group. The caller (the lagged-design assembler) must emit columns in this order.

Subsampling correction
----------------------
Negative subsampling is handled at the GLM level by two standard, non-controversial hooks the
caller supplies: per-sample `sample_weight` (importance weights) and/or a per-sample `offset`
(added to `eta`). The exact subsampling-correction convention (importance weights vs a
log-prior intercept offset) is a PIPELINE decision, deferred to claim-1 wiring — the solver
just honours whatever weights/offset it is given.

Precision note (prototype): jax x64 is enabled here for convergence-test stability; the
promoted version must confirm the precision policy against the rest of `modeling/` (which
runs float32).
"""

from __future__ import annotations

import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

_JITTER = 1e-12


def _family_nll_per_sample(eta: jnp.ndarray, y: jnp.ndarray, family: str) -> jnp.ndarray:
    """
    Description
    -----------
    Per-sample negative log-likelihood of the linear predictor under the chosen GLM family,
    dropping additive constants that do not depend on the parameters (the Poisson ``log(y!)``
    term). ``family='bernoulli'`` uses the numerically-stable ``softplus`` form of the
    logistic NLL; ``family='poisson'`` uses the log-link NLL.

    Parameters
    ----------
    eta (jnp.ndarray)
        Linear predictor ``X @ w + b + offset``, shape ``(n_samples,)``.
    y (jnp.ndarray)
        Observed counts, shape ``(n_samples,)`` (0/1 for Bernoulli, non-negative ints for Poisson).
    family (str)
        ``'bernoulli'`` or ``'poisson'``.

    Returns
    -------
    nll (jnp.ndarray)
        Per-sample NLL, shape ``(n_samples,)``.
    """

    if family == "bernoulli":
        # softplus(eta) - y*eta  ==  -log Bernoulli(y | sigmoid(eta)); stable for large |eta|.
        return jax.nn.softplus(eta) - y * eta
    if family == "poisson":
        # exp(eta) - y*eta  ==  -log Poisson(y | exp(eta))  up to the constant log(y!).
        return jnp.exp(eta) - y * eta
    msg = f"family must be 'bernoulli' or 'poisson'; got {family!r}."
    raise ValueError(msg)


def _smooth_objective(
        params: tuple[jnp.ndarray, jnp.ndarray],
        x_design: jnp.ndarray,
        y: jnp.ndarray,
        sample_weight: jnp.ndarray,
        offset: jnp.ndarray,
        n_features: int,
        n_time: int,
        smoothness_order: int,
        lam_smooth: float,
        lam_ridge: float,
        family: str,
) -> jnp.ndarray:
    """
    Description
    -----------
    The SMOOTH part of the group-elastic-net objective: the sample-weighted mean family NLL
    plus the temporal-smoothness and ridge penalties. Excludes the non-smooth group-lasso term
    (handled by the prox). The smoothness penalty is the in-loss ``jnp.diff`` form, matching
    the modeling estimators; it penalises the ``smoothness_order``-th time-difference of each
    feature's filter. Ridge acts on ``w`` only, never the bias.

    Parameters
    ----------
    params (tuple[jnp.ndarray, jnp.ndarray])
        ``(w, b)`` with ``w`` shape ``(n_features*n_time,)`` and ``b`` scalar.
    x_design (jnp.ndarray)
        Design matrix, shape ``(n_samples, n_features*n_time)`` (feature-major, lag-minor).
    y (jnp.ndarray)
        Observed counts, shape ``(n_samples,)``.
    sample_weight (jnp.ndarray)
        Per-sample weights, shape ``(n_samples,)`` (normalised to unit mean by the caller).
    offset (jnp.ndarray)
        Per-sample offset added to ``eta``, shape ``(n_samples,)``.
    n_features (int)
        Number of feature groups.
    n_time (int)
        Number of temporal lags per feature.
    smoothness_order (int)
        Finite-difference order of the smoothness penalty (1 or 2).
    lam_smooth (float)
        Temporal-smoothness penalty strength.
    lam_ridge (float)
        Ridge penalty strength.
    family (str)
        GLM family (``'bernoulli'`` / ``'poisson'``).

    Returns
    -------
    objective (jnp.ndarray)
        Scalar smooth objective value.
    """

    w, b = params
    eta = x_design @ w + b + offset
    nll = _family_nll_per_sample(eta, y, family)
    data_term = jnp.sum(sample_weight * nll) / jnp.sum(sample_weight)

    w_reshaped = w.reshape(n_features, n_time)
    dw = jnp.diff(w_reshaped, n=smoothness_order, axis=1)
    smooth_term = 0.5 * lam_smooth * jnp.sum(dw ** 2)

    ridge_term = 0.5 * lam_ridge * jnp.sum(w ** 2)
    return data_term + smooth_term + ridge_term


_smooth_objective_jit = jax.jit(
    _smooth_objective,
    static_argnames=("n_features", "n_time", "smoothness_order", "family"),
)
_smooth_value_and_grad_jit = jax.jit(
    jax.value_and_grad(_smooth_objective, argnums=0),
    static_argnames=("n_features", "n_time", "smoothness_order", "family"),
)
_smooth_grad = jax.grad(_smooth_objective, argnums=0)


def _group_soft_threshold(w: jnp.ndarray, threshold: float, n_features: int, n_time: int) -> jnp.ndarray:
    """
    Description
    -----------
    Proximal operator of the group-lasso penalty: block soft-threshold each feature's temporal
    filter by ``threshold``. A feature whose filter L2 norm is below ``threshold`` is zeroed
    entirely (feature sparsity); otherwise the filter is scaled toward zero by
    ``1 - threshold/||w_f||``. This is the exact prox of ``threshold * sum_f ||w_f||_2``.

    Parameters
    ----------
    w (jnp.ndarray)
        Flat weight vector, shape ``(n_features*n_time,)``.
    threshold (float)
        Soft-threshold amount (``lam_group / L`` in FISTA).
    n_features (int)
        Number of feature groups.
    n_time (int)
        Lags per feature.

    Returns
    -------
    w_prox (jnp.ndarray)
        Thresholded weight vector, same shape as ``w``.
    """

    w_reshaped = w.reshape(n_features, n_time)
    norms = jnp.sqrt(jnp.sum(w_reshaped ** 2, axis=1) + _JITTER)
    scale = jnp.maximum(0.0, 1.0 - threshold / norms)
    return (w_reshaped * scale[:, None]).reshape(-1)


_group_soft_threshold_jit = jax.jit(_group_soft_threshold, static_argnames=("n_features", "n_time"))


def _power_iter_curvature(x_design: np.ndarray, sample_weight: np.ndarray, n_iter: int = 20) -> float:
    """
    Description
    -----------
    Power-iterate the largest eigenvalue of the sample-weighted, mean-normalised Gram matrix
    ``(1/sum_w) Xᵀ diag(w) X`` via matrix-vector products (no dense Gram formed). Multiplied by the
    family curvature bound this gives the data-term gradient's Lipschitz constant, so a fixed FISTA step
    ``1/L`` is provably safe (for Bernoulli, whose curvature is globally ≤ 0.25).

    Parameters
    ----------
    x_design (np.ndarray)
        Design matrix ``(n_samples, n_inputs)``.
    sample_weight (np.ndarray)
        Per-sample weights.
    n_iter (int)
        Power-iteration steps.

    Returns
    -------
    lambda_max (float)
        Estimated largest eigenvalue of the weighted mean-normalised Gram.
    """

    xs = np.asarray(x_design)
    ws = np.asarray(sample_weight)
    total = float(np.sum(ws))
    v = np.ones(xs.shape[1]) / np.sqrt(xs.shape[1])
    lambda_max = 0.0
    for _ in range(n_iter):
        mv = (xs.T @ (ws * (xs @ v))) / total
        lambda_max = float(np.linalg.norm(mv))
        v = mv / (lambda_max + 1e-12)
    return lambda_max


@partial(
    jax.jit,
    static_argnames=("n_features", "n_time", "smoothness_order", "family", "max_iter"),
)
def _fista_lax(
        x_design, y, sample_weight, offset, w_init, b_init,
        lam_group, lam_smooth, lam_ridge, step, tol, max_iter,
        n_features, n_time, smoothness_order, family,
):
    """
    Description
    -----------
    Fixed-step accelerated proximal gradient (FISTA) fused into a single ``jax.lax.while_loop`` — the whole
    descent runs on-device with NO per-iteration host syncs (the prototype's backtracking host loop was the
    bottleneck). ``step = 1/L`` is precomputed from the Lipschitz constant; the group-lasso prox is a
    threshold ``lam_group*step`` (0 => identity, so this also serves the relaxed-refit ``lam_group=0`` fit).

    Parameters
    ----------
    x_design, y, sample_weight, offset (jnp.ndarray)
        Design, labels, weights, per-sample offset.
    w_init, b_init (jnp.ndarray)
        Warm-start weights / bias.
    lam_group, lam_smooth, lam_ridge, step, tol (float)
        Penalties, fixed step, convergence tolerance.
    max_iter (int)
        Iteration cap.
    n_features, n_time, smoothness_order (int)
        Design/penalty structure.
    family (str)
        GLM family.

    Returns
    -------
    w, b, n_iter, converged (tuple)
        Fitted weights/bias, iterations run, convergence flag.
    """

    thresh = lam_group * step

    def cond_fn(state):
        i_, _xw, _xb, _yw, _yb, _t, conv = state
        return (i_ < max_iter) & (~conv)

    def body_fn(state):
        i_, xw, xb, yw, yb, t, _conv = state
        grads = _smooth_grad(
            (yw, yb), x_design, y, sample_weight, offset,
            n_features, n_time, smoothness_order, lam_smooth, lam_ridge, family,
        )
        gw, gb = grads
        nw = _group_soft_threshold(yw - step * gw, thresh, n_features, n_time)
        nb = yb - step * gb
        t_new = 0.5 * (1.0 + jnp.sqrt(1.0 + 4.0 * t * t))
        momentum = (t - 1.0) / t_new
        yw_new = nw + momentum * (nw - xw)
        yb_new = nb + momentum * (nb - xb)
        change = jnp.sqrt(jnp.sum((nw - xw) ** 2) + (nb - xb) ** 2)
        scale = jnp.sqrt(jnp.sum(xw ** 2) + xb ** 2) + 1e-12
        conv = (change / scale) < tol
        return (i_ + 1, nw, nb, yw_new, yb_new, t_new, conv)

    init = (jnp.asarray(0, dtype=jnp.int32), w_init, b_init, w_init, b_init,
            jnp.asarray(1.0, dtype=jnp.float64), jnp.asarray(False))
    i_final, xw_final, xb_final, _yw, _yb, _t, conv_final = jax.lax.while_loop(cond_fn, body_fn, init)
    return xw_final, xb_final, i_final, conv_final


class GroupElasticNetGLM:
    """
    Description
    -----------
    Family-agnostic (Bernoulli / Poisson) group-elastic-net + temporal-smoothness GLM, solved by
    accelerated proximal gradient (FISTA) with backtracking line search. The smooth part (family
    NLL + temporal-smoothness + ridge) is descended by gradient; the non-smooth group-lasso is
    applied by a per-group soft-threshold prox, giving exact feature sparsity. An optional
    relaxed-lasso refit re-fits the selected groups with the group-lasso switched off (smoothness
    + ridge only) for an unshrunk, honest fit.

    Penalties are FIXED (not tuned): ``lambda_group / lambda_smooth / lambda_ridge`` are set by the
    caller. Randomness is confined to nothing here — the solver is deterministic given its inputs.

    Parameters
    ----------
    n_features (int)
        Number of feature groups (each a length-``n_time_bins`` temporal filter).
    n_time_bins (int)
        Temporal lags per feature.
    family (str)
        ``'bernoulli'`` (default; per-frame 0/1 point process) or ``'poisson'`` (counts).
    lambda_group (float)
        Group-lasso strength (feature sparsity). ``0.0`` disables sparsity (pure smooth fit).
    lambda_smooth (float)
        Temporal-smoothness penalty strength.
    lambda_ridge (float)
        Ridge penalty strength.
    smoothness_order (int)
        Finite-difference order of the smoothness penalty (1 or 2). Default 2.
    debias_refit (bool)
        If True, relaxed-lasso refit of the selected groups with ``lambda_group=0``. Default True.
    max_iter (int)
        Maximum FISTA iterations. Default 5000.
    tol (float)
        Relative parameter-change convergence tolerance. Default 1e-6.
    backtrack_factor (float)
        Line-search step-shrink factor (>1). Default 2.0.
    lipschitz_init (float)
        Initial inverse step (Lipschitz estimate) for backtracking. Default 1.0.

    Returns
    -------
    None
    """

    def __init__(
            self,
            n_features: int,
            n_time_bins: int,
            family: str = "bernoulli",
            lambda_group: float = 1.0,
            lambda_smooth: float = 1.0,
            lambda_ridge: float = 0.01,
            smoothness_order: int = 2,
            debias_refit: bool = True,
            max_iter: int = 5000,
            tol: float = 1e-6,
            backtrack_factor: float = 2.0,
            lipschitz_init: float = 1.0,
    ) -> None:
        if smoothness_order not in (1, 2):
            msg = f"smoothness_order must be 1 or 2; got {smoothness_order}."
            raise ValueError(msg)
        if family not in ("bernoulli", "poisson"):
            msg = f"family must be 'bernoulli' or 'poisson'; got {family!r}."
            raise ValueError(msg)
        self.n_features = int(n_features)
        self.n_time_bins = int(n_time_bins)
        self.family = family
        self.lambda_group = float(lambda_group)
        self.lambda_smooth = float(lambda_smooth)
        self.lambda_ridge = float(lambda_ridge)
        self.smoothness_order = int(smoothness_order)
        self.debias_refit = bool(debias_refit)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.backtrack_factor = float(backtrack_factor)
        self.lipschitz_init = float(lipschitz_init)

    def _fista(
            self,
            x_design: jnp.ndarray,
            y: jnp.ndarray,
            sample_weight: jnp.ndarray,
            offset: jnp.ndarray,
            lam_group: float,
            w_init: jnp.ndarray,
            b_init: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, int, bool]:
        """
        Description
        -----------
        Accelerated proximal gradient (FISTA) with backtracking line search for the objective at a
        given ``lam_group`` (``0.0`` => plain smooth descent, prox is identity). Host-side Python
        loop calling jitted value/grad + prox — chosen for clarity/debuggability in the prototype;
        the promoted version may fuse into ``jax.lax.while_loop``.

        Parameters
        ----------
        x_design (jnp.ndarray)
            Design matrix ``(n_samples, n_features*n_time)``.
        y (jnp.ndarray)
            Observed counts ``(n_samples,)``.
        sample_weight (jnp.ndarray)
            Per-sample weights ``(n_samples,)`` (unit-mean normalised).
        offset (jnp.ndarray)
            Per-sample offset ``(n_samples,)``.
        lam_group (float)
            Group-lasso strength for this solve.
        w_init (jnp.ndarray)
            Warm-start weights ``(n_features*n_time,)``.
        b_init (jnp.ndarray)
            Warm-start bias (scalar array).

        Returns
        -------
        w, b, n_iter, converged (tuple[jnp.ndarray, jnp.ndarray, int, bool])
            Fitted weights/bias, iterations run, and whether the tolerance was met.
        """

        # Fast path: fixed-step FISTA fully inside `jax.lax.while_loop` (no per-iteration host syncs).
        # Bernoulli curvature is globally <= 0.25, so a Lipschitz-based fixed step is provably safe and
        # convergence is guaranteed. Poisson curvature is unbounded, so it keeps the backtracking host loop.
        if self.family == "bernoulli":
            curvature = _power_iter_curvature(np.asarray(x_design), np.asarray(sample_weight))
            lipschitz = 0.25 * curvature + self.lambda_ridge + 16.0 * self.lambda_smooth
            step = 1.0 / max(float(lipschitz), 1e-12)
            w_f, b_f, n_iter_f, conv_f = _fista_lax(
                x_design, y, sample_weight, offset, w_init, b_init,
                lam_group, self.lambda_smooth, self.lambda_ridge, step, self.tol, self.max_iter,
                self.n_features, self.n_time_bins, self.smoothness_order, self.family,
            )
            return w_f, b_f, int(n_iter_f), bool(conv_f)

        nf, nt, order, fam = self.n_features, self.n_time_bins, self.smoothness_order, self.family
        static = dict(n_features=nf, n_time=nt, smoothness_order=order, family=fam)

        def f_val(w: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
            return _smooth_objective_jit(
                (w, b), x_design, y, sample_weight, offset,
                lam_smooth=self.lambda_smooth, lam_ridge=self.lambda_ridge, **static,
            )

        def f_val_grad(w: jnp.ndarray, b: jnp.ndarray) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray]]:
            return _smooth_value_and_grad_jit(
                (w, b), x_design, y, sample_weight, offset,
                lam_smooth=self.lambda_smooth, lam_ridge=self.lambda_ridge, **static,
            )

        w_cur, b_cur = w_init, b_init
        w_prev, b_prev = w_cur, b_cur
        z_w, z_b = w_cur, b_cur
        t_cur = 1.0
        lipschitz = self.lipschitz_init
        converged = False
        n_iter = 0

        for k in range(self.max_iter):
            n_iter = k + 1
            fz, (gw, gz_b) = f_val_grad(z_w, z_b)

            # Backtracking line search on the Lipschitz constant (proximal-gradient majorisation).
            while True:
                step = 1.0 / lipschitz
                cand_w = _group_soft_threshold_jit(z_w - step * gw, lam_group * step, nf, nt) \
                    if lam_group > 0.0 else (z_w - step * gw)
                cand_b = z_b - step * gz_b
                f_cand = f_val(cand_w, cand_b)
                dw_vec = cand_w - z_w
                db_vec = cand_b - z_b
                inner = jnp.sum(gw * dw_vec) + gz_b * db_vec
                quad = 0.5 * lipschitz * (jnp.sum(dw_vec ** 2) + db_vec ** 2)
                if float(f_cand) <= float(fz + inner + quad) + 1e-12:
                    break
                lipschitz *= self.backtrack_factor
                if lipschitz > 1e12:
                    break

            w_new, b_new = cand_w, cand_b
            t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t_cur ** 2))
            momentum = (t_cur - 1.0) / t_new
            z_w = w_new + momentum * (w_new - w_cur)
            z_b = b_new + momentum * (b_new - b_cur)

            change = float(jnp.sqrt(jnp.sum((w_new - w_cur) ** 2) + (b_new - b_cur) ** 2))
            scale = float(jnp.sqrt(jnp.sum(w_cur ** 2) + b_cur ** 2)) + 1e-12
            w_prev, b_prev = w_cur, b_cur
            w_cur, b_cur = w_new, b_new
            t_cur = t_new
            if change / scale < self.tol:
                converged = True
                break

        return w_cur, b_cur, n_iter, converged

    def fit(
            self,
            x_design: np.ndarray,
            y: np.ndarray,
            sample_weight: np.ndarray | None = None,
            offset: np.ndarray | None = None,
    ) -> "GroupElasticNetGLM":
        """
        Description
        -----------
        Fit the group-elastic-net GLM by FISTA, then (if ``debias_refit``) relaxed-lasso refit the
        selected feature groups with the group-lasso disabled. Stores ``coef_`` (``w``),
        ``intercept_`` (``b``), ``selected_features_`` (boolean mask over the ``n_features`` groups),
        ``n_iter_``, ``converged_``, and ``fit_time_``.

        Parameters
        ----------
        x_design (np.ndarray)
            Design matrix ``(n_samples, n_features*n_time_bins)``, feature-major lag-minor.
        y (np.ndarray)
            Observed counts ``(n_samples,)``.
        sample_weight (np.ndarray | None)
            Per-sample weights; ``None`` => all ones. Normalised to unit mean internally.
        offset (np.ndarray | None)
            Per-sample offset added to ``eta``; ``None`` => zeros.

        Returns
        -------
        self (GroupElasticNetGLM)
            The fitted estimator.
        """

        t0 = time.perf_counter()
        n_inputs = self.n_features * self.n_time_bins
        if x_design.shape[1] != n_inputs:
            msg = (f"x_design has {x_design.shape[1]} columns but n_features*n_time_bins="
                   f"{n_inputs}.")
            raise ValueError(msg)

        x_j = jnp.asarray(x_design, dtype=jnp.float64)
        y_j = jnp.asarray(y, dtype=jnp.float64)
        n_samples = x_j.shape[0]
        if sample_weight is None:
            w_samp = jnp.ones(n_samples, dtype=jnp.float64)
        else:
            w_samp = jnp.asarray(sample_weight, dtype=jnp.float64)
            w_samp = w_samp * (n_samples / jnp.sum(w_samp))
        off_j = jnp.zeros(n_samples, dtype=jnp.float64) if offset is None \
            else jnp.asarray(offset, dtype=jnp.float64)

        # Warm start: zero filters, bias at the (weighted) family-appropriate base rate.
        w_init = jnp.zeros(n_inputs, dtype=jnp.float64)
        y_bar = float(jnp.sum(w_samp * y_j) / jnp.sum(w_samp))
        off_bar = float(jnp.sum(w_samp * off_j) / jnp.sum(w_samp))
        if self.family == "bernoulli":
            y_bar = min(max(y_bar, 1e-6), 1.0 - 1e-6)
            b_init = jnp.asarray(np.log(y_bar / (1.0 - y_bar)) - off_bar, dtype=jnp.float64)
        else:
            b_init = jnp.asarray(np.log(max(y_bar, 1e-6)) - off_bar, dtype=jnp.float64)

        w_fit, b_fit, n_iter, converged = self._fista(
            x_j, y_j, w_samp, off_j, self.lambda_group, w_init, b_init,
        )

        w_reshaped = np.asarray(w_fit).reshape(self.n_features, self.n_time_bins)
        selected = np.sqrt(np.sum(w_reshaped ** 2, axis=1)) > 1e-8

        if self.debias_refit and self.lambda_group > 0.0 and selected.any():
            # Relaxed lasso: refit selected groups with group-lasso OFF (smoothness + ridge only),
            # keeping unselected filters exactly zero.
            col_mask = np.repeat(selected, self.n_time_bins)
            x_sel = x_j[:, col_mask]
            sub = GroupElasticNetGLM(
                n_features=int(selected.sum()), n_time_bins=self.n_time_bins, family=self.family,
                lambda_group=0.0, lambda_smooth=self.lambda_smooth, lambda_ridge=self.lambda_ridge,
                smoothness_order=self.smoothness_order, debias_refit=False,
                max_iter=self.max_iter, tol=self.tol,
                backtrack_factor=self.backtrack_factor, lipschitz_init=self.lipschitz_init,
            )
            sub.fit(np.asarray(x_sel), y, sample_weight=sample_weight, offset=offset)
            w_full = np.zeros(n_inputs, dtype=np.float64)
            w_full[col_mask] = sub.coef_
            w_fit = jnp.asarray(w_full)
            b_fit = jnp.asarray(sub.intercept_)

        self.coef_ = np.asarray(w_fit, dtype=np.float64)
        self.intercept_ = float(np.asarray(b_fit))
        self.selected_features_ = selected
        self.n_iter_ = int(n_iter)
        self.converged_ = bool(converged)
        self.fit_time_ = time.perf_counter() - t0
        return self

    def predict_eta(self, x_design: np.ndarray, offset: np.ndarray | None = None) -> np.ndarray:
        """
        Description
        -----------
        Linear predictor ``eta = X @ coef_ + intercept_ + offset`` for new data.

        Parameters
        ----------
        x_design (np.ndarray)
            Design matrix ``(n_samples, n_features*n_time_bins)``.
        offset (np.ndarray | None)
            Per-sample offset; ``None`` => zeros.

        Returns
        -------
        eta (np.ndarray)
            Linear predictor, shape ``(n_samples,)``.
        """

        eta = x_design @ self.coef_ + self.intercept_
        if offset is not None:
            eta = eta + offset
        return eta

    def predict_rate(self, x_design: np.ndarray, offset: np.ndarray | None = None) -> np.ndarray:
        """
        Description
        -----------
        Predicted rate: ``sigmoid(eta)`` (Bernoulli probability) or ``exp(eta)`` (Poisson mean).

        Parameters
        ----------
        x_design (np.ndarray)
            Design matrix ``(n_samples, n_features*n_time_bins)``.
        offset (np.ndarray | None)
            Per-sample offset; ``None`` => zeros.

        Returns
        -------
        rate (np.ndarray)
            Predicted rate, shape ``(n_samples,)``.
        """

        eta = self.predict_eta(x_design, offset=offset)
        if self.family == "bernoulli":
            return np.asarray(jax.nn.sigmoid(jnp.asarray(eta)))
        return np.exp(eta)

    def mean_log_loss(
            self,
            x_design: np.ndarray,
            y: np.ndarray,
            sample_weight: np.ndarray | None = None,
            offset: np.ndarray | None = None,
    ) -> float:
        """
        Description
        -----------
        Sample-weighted mean negative log-likelihood (log-loss) on held-out data — the proper
        scoring rule used as the claim-1 metric (Bernoulli NLL at bin=1, Poisson NLL otherwise).

        Parameters
        ----------
        x_design (np.ndarray)
            Design matrix ``(n_samples, n_features*n_time_bins)``.
        y (np.ndarray)
            Observed counts ``(n_samples,)``.
        sample_weight (np.ndarray | None)
            Per-sample weights; ``None`` => all ones.
        offset (np.ndarray | None)
            Per-sample offset; ``None`` => zeros.

        Returns
        -------
        log_loss (float)
            Weighted mean per-sample NLL.
        """

        eta = jnp.asarray(self.predict_eta(x_design, offset=offset))
        nll = _family_nll_per_sample(eta, jnp.asarray(y, dtype=jnp.float64), self.family)
        if sample_weight is None:
            return float(jnp.mean(nll))
        w = jnp.asarray(sample_weight, dtype=jnp.float64)
        return float(jnp.sum(w * nll) / jnp.sum(w))
