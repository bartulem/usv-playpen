"""
@author: bartulem
JAX-based 2-D regression for behaviour-conditioned UMAP position prediction.

This module implements a scikit-learn-compatible estimator that maps a
high-dimensional behavioural-kinematic history `X` onto the 2-D coordinates
`(x, y)` of the UMAP acoustic manifold. The target is a deterministic point
in R^2 rather than a full probability density — we predict *where* a bout
will land, not *how uncertain* the prediction is.

Modelling choices (and why we replaced the bivariate-Gaussian density):
-----------------------------------------------------------------------
The previous `SmoothBivariateGaussianRegression` learned a linear map
`X -> (mu_x, mu_y)` alongside three *free-standing* global variance
parameters `(sigma_x, sigma_y, rho)` shared across every trial in a fold.
That covariance was not conditional on `X`, so the "calibration" metrics
(coverage at 68 %/95 %, Mahalanobis distance) were really measuring whether
the empirical residual distribution happened to match a single global
ellipse — a weaker claim than a probabilistic framing suggests. In addition,
UMAP coordinates are not a metric space in any principled sense (global
Euclidean distance over-weights embedding seams), and we never relied on
per-trial `(sigma_x, sigma_y, rho)` scientifically. Stripping the density
keeps every signal that actually mattered (the learned mean, the temporal
smoothness, the inverse-density sample weighting) and drops the machinery
whose guarantees we could not stand behind.

Loss
----
Density-weighted Huber regression on the 2-D residual:

    L(W, b) =  mean_i  w_i * huber_delta( ||y_true_i - (X_i W + b)||_2 )
             + 0.5 * lam_l2 * ||W||_2^2
             + 0.5 * lam_smooth * sum over the smoothness_derivative_order-th
                                  (1st- or 2nd-order) time-derivatives of W**2

- `w_i` is the inverse-density sample weight (KDE), normalised to unit mean
  so the loss scale is invariant to the caller's weighting convention and
  the regularisation hyper-parameters are decoupled from the weight scale.
- Huber is preferred over plain MSE so a handful of satellite vocalisations
  cannot dominate the gradient. The delta parameter is exposed via
  `huber_delta`; setting `huber_delta=np.inf` recovers pure squared loss.
- L2 constrains absolute weight magnitude; the temporal smoothness term
  penalises the discrete `smoothness_derivative_order`-th derivative (1st-
  or 2nd-order, configurable) of the weights along the time axis, just as
  in the old Gaussian model, so learned kinematic filters remain
  biologically plausible smooth curves.

Manifold-bounded predictions
----------------------------
At predict time, every raw linear output `X @ W + b` is projected onto
the nearest observed training UMAP point (`k=1` in Euclidean distance
over `Y_train_`, cached in a `scipy.spatial.cKDTree`). Training targets
and their kd-tree are stored on the fitted model, so `predict(snap=True)`
returns only coordinates that the training manifold actually contained.
This matters for the fairness of comparisons against the baselines:
`null_model_free` predicts a uniform draw from the training `Y` and the
candidate X-history shuffled `null` predicts on the same manifold
support, so the active model must not be allowed to gain apparent
accuracy by extrapolating off-manifold and then being penalised by a
metric the baselines structurally cannot incur. `predict(snap=False)`
is retained for diagnostics.

Outputs
-------
- `predict(X, snap=True)` -> (n_samples, 2) snapped UMAP coordinates, or
  the raw linear prediction when `snap=False`.
- `evaluate_metrics(X, Y_true, weights=None)` returns a metric bundle
  containing `r2_spatial` (the selection score on Euclidean / VAE / UMAP
  manifolds), `dcor_xy` (wrap-aware distance correlation between the
  decoded prediction and the truth -- the selection score on the TORUS
  manifold, `nan` on this Euclidean path), `euclidean_mae` /
  `euclidean_rmse` / `euclidean_mae_weighted` (native-scale
  interpretable errors on the *snapped* predictions), `euclidean_mae_raw`
  (unsnapped diagnostic), `mahalanobis_mae` (dimensionless error in the
  inverse-density-weighted training covariance), plus per-axis `mae_x`
  / `mae_y`, `spearman_x` / `spearman_y`, and `pearson_x` / `pearson_y`.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import time
from functools import partial
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from typing import Tuple, Any, Optional

from .manifold_metric import (
    signed_diff,
    signed_diff_jax,
    circular_mean,
    total_dispersion,
    torus_embed,
    dcor_prediction_truth,
    _validate_metric_period,
)


def _bivariate_loss_static(
        params,
        X,
        Y_true,
        weights,
        n_feats: int,
        n_time: int,
        lam_smooth,
        lam_l2,
        huber_delta,
        smoothness_derivative_order: int,
        metric: str,
        period: float,
):
    """
    Module-level mirror of `SmoothBivariateRegression._loss_fn`.

    Existing as a free function so the cache-friendly training-loop JIT
    (`_bivariate_train_loop_jit` below) can be defined at module scope
    without having to bind a method via `self`. Keeps the per-instance
    closure capture out of the JIT cache key — multiple estimator
    instances sharing the same input shapes hit the same compiled
    graph instead of forcing a re-trace per instance.

    The residual is computed via `signed_diff_jax` so torus targets are
    folded into the shortest-wrap representation before squaring; on
    `metric='euclidean'` this reduces to the plain `Y_true - Y_pred`
    difference.
    """

    W, b = params

    Y_pred = jnp.dot(X, W) + b
    residuals = signed_diff_jax(Y_true, Y_pred, metric=metric, period=period)
    norms = jnp.sqrt(jnp.sum(residuals ** 2, axis=1) + 1e-12)
    quad = jnp.minimum(norms, huber_delta)
    lin = norms - quad
    huber_per_sample = 0.5 * quad ** 2 + huber_delta * lin
    weighted_loss = jnp.mean(weights * huber_per_sample)

    l2_loss = 0.5 * lam_l2 * jnp.sum(W ** 2)

    W_reshaped = W.reshape(n_feats, n_time, 2)
    dW = jnp.diff(W_reshaped, n=smoothness_derivative_order, axis=1)
    smooth_loss = 0.5 * lam_smooth * jnp.sum(dW ** 2)

    return weighted_loss + l2_loss + smooth_loss


@partial(jax.jit, static_argnames=('n_feats', 'n_time', 'smoothness_derivative_order',
                                   'metric', 'max_iter'))
def _bivariate_train_loop_jit(
        params_init,
        opt_state_init,
        X,
        Y,
        w,
        lambda_smooth,
        l2_reg,
        huber_delta,
        learning_rate,
        grad_clip_norm,
        tol,
        max_iter: int,
        n_feats: int,
        n_time: int,
        smoothness_derivative_order: int,
        metric: str,
        period: float,
):
    """
    Full descent fused into a single `jax.lax.while_loop` inside one
    `@jax.jit` call — eliminates per-iteration Python dispatch on the
    training loop.

    Crucially this function is defined at **module scope** and takes
    every per-fit parameter as a JAX traced argument (or as one of the
    declared `static_argnames`). The JIT cache is therefore keyed on
    `(function id, static arg values, traced shape/dtype)` only, so
    every `SmoothBivariateRegression` instance with matching
    `(n_feats, n_time, smoothness_derivative_order)` and matching `X` /
    `Y` shapes hits the same compiled graph. This is essential for the
    joint inner-CV tuner, which constructs ~`|grid| * inner_cv_folds`
    estimators per outer fold; without shape-keyed caching, every one
    triggers a fresh recompile and the run stalls indefinitely on
    wide-feature graphs.

    The optimizer is rebuilt on each call. `optax.adam` is a thin
    pure-function wrapper, so this is cheap and the resulting Adam
    state pytree is identical to the one constructed externally by
    `fit()` (same shapes / dtypes), which lets the caller pass in a
    pre-initialised state without breaking JAX's structural typing.
    """

    # Cosine-decay the step to zero over `max_iter`, and clip the global norm of
    # the gradient before Adam consumes it. Both are required for the
    # `diff < tol` convergence test to be reachable at all: a *constant* Adam
    # step does not shrink as the gradient vanishes, so the parameter-change
    # norm plateaus at a floor set by `learning_rate` and the loop always runs
    # to `max_iter` reporting `converged=False`. The failure concentrates in the
    # stiff corner of the tuning grid -- a large `lambda_smooth` on the
    # 2nd-derivative penalty over hundreds of lags is badly ill-conditioned, and
    # a fixed step oscillates in those high-curvature directions rather than
    # settling. The objective is convex, so clipping never biases the solution;
    # it only bounds the per-step move. `max_iter` is a static argument because
    # the schedule needs it as a Python int at trace time. Mirrors the
    # multinomial estimator, which already carried both guards.
    scheduler = optax.cosine_decay_schedule(
        init_value=learning_rate, decay_steps=max_iter,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(grad_clip_norm),
        optax.adam(scheduler),
    )
    check_interval = 100

    def step(params, opt_state):
        grads = jax.grad(_bivariate_loss_static)(
            params, X, Y, w,
            n_feats, n_time,
            lambda_smooth, l2_reg, huber_delta,
            smoothness_derivative_order,
            metric, period,
        )
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state

    def cond_fn(state):
        i_, _params, _opt_state, _last, converged_flag = state
        return (i_ < max_iter) & (~converged_flag)

    def body_fn(state):
        i_, params_, opt_state_, last_, _converged = state
        params_next, opt_state_next = step(params_, opt_state_)
        i_next = i_ + 1

        is_check_step = (i_next > 1) & (i_next % check_interval == 0)
        w_diff = jnp.linalg.norm(params_next[0] - last_[0])
        b_diff = jnp.linalg.norm(params_next[1] - last_[1])
        diff = jnp.sqrt(w_diff ** 2 + b_diff ** 2)
        converged_next = is_check_step & (diff < tol)

        last_next = jax.tree_util.tree_map(
            lambda new, old: jnp.where(is_check_step, new, old),
            params_next, last_,
        )

        return (i_next, params_next, opt_state_next, last_next, converged_next)

    init_state = (
        jnp.asarray(0, dtype=jnp.int32),
        params_init,
        opt_state_init,
        params_init,
        jnp.asarray(False),
    )
    final_state = jax.lax.while_loop(cond_fn, body_fn, init_state)
    i_final, params_final, _, _, converged_final = final_state
    return params_final, i_final, converged_final


class SmoothBivariateRegression(BaseEstimator, RegressorMixin):
    """
    2-D regression with L2 and temporal-smoothness penalties (JAX implementation).

    This estimator learns a weight matrix `W` of shape
    `(n_features * n_time_bins, 2)` and a bias `b` of shape `(2,)`; predictions
    are the deterministic UMAP coordinates `X @ W + b`. The optimisation
    objective is a density-weighted Huber residual loss augmented by L2 and
    a temporal-smoothness penalty on the second time derivative of `W`.

    Parameters
    ----------
    n_features : int
        Number of distinct behavioural features (e.g., speed, distance).
        Crucial for reshaping the flat coefficient vector back into
        (features, time, 2) so the smoothness penalty is applied along the
        time axis of each feature only.
    n_time_bins : int
        Number of time steps per feature. The input `X` must have exactly
        `n_features * n_time_bins` columns.
    lambda_smooth : float, default=1e2
        Strength of the temporal smoothing penalty. Penalises the squared
        second derivative of the weights across the time axis of each
        (feature, output-coordinate) slice; larger values yield stiffer
        filters.
    l2_reg : float, default=0.1
        Strength of the standard L2 (ridge) penalty to constrain overall
        weight magnitude.
    smoothness_derivative_order : int, default=2
        Order of the finite-difference derivative used to build the
        temporal-smoothness penalty on `W` along the time axis. Both
        choices correspond to improper Gaussian-process priors and neither
        fixes a basis, but they push the filter toward different shape
        families:
          - `1` — squared first-difference penalty
            `sum((w_{t+1} - w_t)^2)`. Zero cost when the filter is flat;
            any change between adjacent time bins is penalised equally.
            Corresponds to a Brownian-motion / Ornstein-Uhlenbeck prior
            and biases the filter toward piecewise-constant / step-like
            shapes. This is the form used in the Paninski/Pillow
            GLM-HMM tradition (e.g., Calhoun et al. *Nature Neurosci*
            2019).
          - `2` — squared second-difference penalty
            `sum((w_{t+1} - 2 w_t + w_{t-1})^2)`. Zero cost when the
            filter is any straight line (constant or with non-zero
            slope); only curvature is penalised. Corresponds to a
            thin-plate-spline / smoothing-spline prior and biases the
            filter toward smooth curves. This is the classical GAM /
            `pyGAM`-style smoothness and is typically preferred when
            the scientific goal is to learn unbiased filter *shape*
            without a preference for piecewise-constant plateaux.
    huber_delta : float, default=1.0
        Transition point of the Huber loss in the native UMAP distance
        units. Residuals with `||r||_2 <= huber_delta` are penalised
        quadratically; larger residuals are penalised linearly. Set to
        `np.inf` to recover pure squared-error regression.
    learning_rate : float, default=1e-3
        Initial step size for the Adam optimiser. It is **cosine-decayed to
        zero over `max_iter` steps**; see `grad_clip_norm` for why.
    grad_clip_norm : float, default=1.0
        Global-norm gradient clip applied before Adam consumes the gradient.
        Together with the cosine-decayed learning rate this mirrors the
        multinomial estimator, and both exist for the same reason. Under a
        *constant* Adam step size the update is `lr * m_hat / (sqrt(v_hat) +
        eps)`, whose magnitude does **not** shrink to zero as the gradient
        does, so the parameter-change norm plateaus at a floor set by `lr` and
        the `diff < tol` convergence test can never fire — the loop simply runs
        to `max_iter` and reports `converged=False`. The problem is worst where
        the objective is stiffest: a large `lambda_smooth` on the 2nd-derivative
        penalty over hundreds of lags is a very ill-conditioned quadratic, and a
        fixed step oscillates in those high-curvature directions instead of
        settling. Empirically every non-converged fold in the 100-fold cluster
        output sat in the heavily regularised corner of the tuning grid
        (`lambda_smooth >= 10`, `l2_reg = 1`) while every converged fold sat at
        `lambda_smooth <= 1`. Decaying the step to zero makes convergence
        attainable; clipping bounds the stiff-direction move. The objective is
        convex, so clipping never biases the solution — it only bounds the
        per-step move.
    max_iter : int, default=5000
        Maximum number of optimisation steps (epochs) allowed before
        termination.
    tol : float, default=1e-4
        Convergence tolerance. Training stops when the L2 norm of the
        parameter change over the last 100 optimiser steps falls below
        this threshold.
    random_state : int, default=0
        Seed for the JAX PRNG so weight initialisation is reproducible.
    verbose : bool, default=False
        If True, prints the loss progress periodically during gradient
        descent.

    Attributes
    ----------
    coef_ : np.ndarray, shape (n_features * n_time_bins, 2)
        Learned weight matrix.
    intercept_ : np.ndarray, shape (2,)
        Learned bias vector.
    Y_train_ : np.ndarray, shape (n_samples, 2)
        Copy of the training UMAP targets. Used by `predict(snap=True)`
        to project raw linear predictions onto the nearest observed
        training point — a hard manifold constraint that keeps the model's
        output inside the same convex support as the null baselines so
        metric magnitudes remain directly comparable.
    train_mean_ : np.ndarray, shape (2,)
        Inverse-density-weighted training centroid (matches the weighting
        convention of the training loss).
    train_cov_inv_ : np.ndarray, shape (2, 2)
        Inverse of the inverse-density-weighted training covariance of
        `Y_train_`. Powers the `mahalanobis_mae` evaluation metric so a
        unit error on a tight UMAP axis is not silently treated the same
        as a unit error on a loose one.
    n_iter_ : int
        Number of optimiser steps actually taken (1-indexed). Equals
        `max_iter` when the tolerance check never fired.
    converged_ : bool
        True iff the tolerance check fired before `max_iter`. A
        `converged_=False` fold has hit the iteration cap without meeting
        the stopping criterion.
    fit_time_ : float
        Wall-clock fit time in seconds.
    is_fitted_ : bool
        Convenience flag consumed by `check_is_fitted`.
    """

    def __init__(
            self,
            n_features: int = 1,
            n_time_bins: int = 1,
            lambda_smooth: float = 1e2,
            l2_reg: float = 0.1,
            smoothness_derivative_order: int = 2,
            huber_delta: float = 1.0,
            learning_rate: float = 1e-3,
            grad_clip_norm: float = 1.0,
            max_iter: int = 5000,
            tol: float = 1e-4,
            random_state: int = 0,
            verbose: bool = False,
            _use_lax_loop: bool = False,
            metric: str = 'euclidean',
            period: float = 1.0,
    ):
        if smoothness_derivative_order not in (1, 2):
            raise ValueError(
                f"smoothness_derivative_order must be 1 or 2; got {smoothness_derivative_order}."
            )
        _validate_metric_period(metric, period)
        self.n_features = n_features
        self.n_time_bins = n_time_bins
        self.lambda_smooth = lambda_smooth
        self.l2_reg = l2_reg
        self.smoothness_derivative_order = int(smoothness_derivative_order)
        self.huber_delta = huber_delta
        self.learning_rate = learning_rate
        self.grad_clip_norm = grad_clip_norm
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose
        # Manifold-metric configuration. `metric='euclidean'` reproduces
        # the legacy flat-space behaviour (training loss is plain Huber
        # on `Y_true - Y_pred`, kdtree is built on `Y_train`, centroid
        # is the arithmetic mean). `metric='torus'` plumbs wrap-aware
        # signed differences through the loss, builds the kdtree on the
        # canonical 4-D torus embedding, and uses circular means /
        # wrap-aware covariances for `train_mean_` / `train_cov_inv_`.
        self.metric = metric
        self.period = float(period)
        # Opt-in fused training loop. When True, the full descent is
        # wrapped in a single `jax.lax.while_loop` inside a `@jax.jit`
        # call — eliminates per-iteration Python dispatch at the cost of
        # a large one-time compile. The fused path delegates to the
        # module-scope `_bivariate_train_loop_jit`, whose cache is keyed
        # on input shape + the static integers (`n_feats`, `n_time`,
        # `smoothness_derivative_order`), so estimators with matching
        # shapes share the same compiled graph. Even so the up-front
        # per-shape compile is large enough that any caller constructing
        # many short-lived estimators across varied shapes (notably the
        # joint inner-CV tuner, which builds ~175 per outer fold) can
        # stall for hours on wide-feature graphs (e.g. bin=1 with 600
        # time bins). Default is False: the standard Python for-loop path
        # has no compilation issues at any scale and is the correct
        # choice for the tuning path. Set to True only on the outer-fit
        # path when `max_iter` is very large (e.g. 10000+) on shapes that
        # compile quickly; the speedup is 1.3-1.8x on GPU in that regime.
        self._use_lax_loop = bool(_use_lax_loop)

    @staticmethod
    def _initialize_params(n_inputs: int, key: Any) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Initialises the linear-transformation weights `W` and bias `b`.

        Weights are initialised with Xavier/Glorot normal scaling so the
        variance of the gradients stays stable in the early Adam steps.
        Biases start at zero and are learned alongside the weights.

        Parameters
        ----------
        n_inputs : int
            Total number of input columns in the flattened design matrix
            (`n_features * n_time_bins`).
        key : jax.random.PRNGKey
            JAX random key for deterministic generation.

        Returns
        -------
        W : jnp.ndarray
            Initialised weight matrix of shape `(n_inputs, 2)`.
        b : jnp.ndarray
            Zero-initialised bias vector of shape `(2,)`.
        """

        n_outputs = 2
        scale = jnp.sqrt(2.0 / (n_inputs + n_outputs))
        W = jax.random.normal(key, (n_inputs, n_outputs)) * scale
        b = jnp.zeros((n_outputs,))
        return W, b

    @staticmethod
    @partial(jax.jit, static_argnums=(4, 5, 9, 10))
    def _loss_fn(
            params: Tuple[jnp.ndarray, jnp.ndarray],
            X: jnp.ndarray,
            Y_true: jnp.ndarray,
            weights: jnp.ndarray,
            n_feats: int,
            n_time: int,
            lam_smooth: float,
            lam_l2: float,
            huber_delta: float,
            smoothness_derivative_order: int,
            metric: str,
            period: float,
    ) -> jnp.ndarray:
        """
        Computes the total optimisation loss: density-weighted Huber + L2 +
        temporal smoothness.

        The Huber residual `||r||_2` is computed on the 2-D coordinate
        difference rather than per-axis so the loss treats a 1-unit miss
        along `x` identically to a 1-unit miss along `y`. The inverse-density
        weights `w_i` (normalised to unit mean by the caller in `fit`) scale
        every sample's contribution so rare satellite vocalisations receive
        proportionally more gradient pressure than dense-core bouts.

        Parameters
        ----------
        params : tuple
            `(W, b)` tuple of current model parameters.
        X : jnp.ndarray
            Input behavioural history matrix of shape
            `(n_samples, n_features * n_time)`.
        Y_true : jnp.ndarray
            True continuous acoustic targets of shape `(n_samples, 2)`.
        weights : jnp.ndarray
            Inverse-density sample weights of shape `(n_samples,)` —
            normalised to unit mean upstream.
        n_feats : int
            Number of distinct physical behavioural features. Static JIT
            argument — used to reshape `W` into
            `(n_features, n_time, 2)` so the smoothness penalty is applied
            along the time axis only.
        n_time : int
            Number of time bins per behavioural feature. Static JIT
            argument; same reshape purpose as `n_feats`.
        lam_smooth : float
            Temporal-smoothness penalty strength.
        lam_l2 : float
            L2 (Ridge) penalty strength.
        huber_delta : float
            Huber transition point in native UMAP distance units.
            Residuals with `||r||_2 <= huber_delta` are penalised
            quadratically; larger residuals are penalised linearly.
        smoothness_derivative_order : int
            Order of the finite-difference derivative (1 or 2) used to
            build the temporal-smoothness penalty on `W`. Static JIT
            argument. See the class docstring for the interpretability
            tradeoff between first- and second-difference penalties.
        metric : str
            Residual geometry selector, `'euclidean'` or `'torus'`. Static
            JIT argument. On `'euclidean'` the residual is the plain
            `Y_true - Y_pred` flat-space difference; on `'torus'` it is the
            wrap-aware shortest-path signed difference (`signed_diff_jax`),
            so points on opposite sides of the wrap boundary are scored by
            their `period - |raw_diff|` distance.
        period : float
            Per-axis wrap period. Used only on the `metric='torus'` path to
            fold each residual component into the shortest-wrap
            representation before squaring; ignored on the euclidean path.

        Returns
        -------
        loss : jnp.ndarray
            Scalar total loss: density-weighted Huber + 0.5 * `lam_l2`
            * ||W||^2 + 0.5 * `lam_smooth` * ||D_order W||^2 along the
            time axis.
        """

        W, b = params

        # Forward pass: linear projection directly to the 2-D output.
        Y_pred = jnp.dot(X, W) + b

        # Huber loss on the Euclidean residual (not per-axis): a 1-unit miss
        # along either coordinate is treated identically, and the quadratic
        # region is ||r|| <= huber_delta. On a torus manifold the residual
        # is wrapped into the shortest-path representation
        # (`signed_diff_jax`) before squaring, so points on opposite sides
        # of the wrap boundary are scored by their `period - |raw_diff|`
        # distance rather than the (much larger) `|raw_diff|`.
        residuals = signed_diff_jax(Y_true, Y_pred, metric=metric, period=period)
        norms = jnp.sqrt(jnp.sum(residuals ** 2, axis=1) + 1e-12)
        quad = jnp.minimum(norms, huber_delta)
        lin = norms - quad
        huber_per_sample = 0.5 * quad ** 2 + huber_delta * lin

        # Apply the inverse-density sample weights. The pipeline normalises
        # `weights` to unit mean before calling fit, so the multiplication
        # rescales the loss landscape without changing its global magnitude.
        weighted_loss = jnp.mean(weights * huber_per_sample)

        # L2 (Ridge) penalty on the linear weights.
        l2_loss = 0.5 * lam_l2 * jnp.sum(W ** 2)

        # Temporal smoothness: penalise the discrete n-th derivative of the
        # weights along the time axis of each (feature, output-dim) slice.
        # `smoothness_derivative_order=1` = first-difference penalty
        # (Brownian / piecewise-constant prior); `=2` = second-difference
        # penalty (thin-plate-spline / smooth-curve prior). See the class
        # docstring for the full tradeoff.
        W_reshaped = W.reshape(n_feats, n_time, 2)
        dW = jnp.diff(W_reshaped, n=smoothness_derivative_order, axis=1)
        smooth_loss = 0.5 * lam_smooth * jnp.sum(dW ** 2)

        return weighted_loss + l2_loss + smooth_loss

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> "SmoothBivariateRegression":
        """
        Fits the 2-D regression model using JAX/Optax optimisation.

        Parameters
        ----------
        X : np.ndarray
            Training vectors, shape `(n_samples, n_features * n_time_bins)`.
        y : np.ndarray
            Target UMAP coordinates, shape `(n_samples, 2)`.
        sample_weight : np.ndarray, optional
            Inverse-density weights of shape `(n_samples,)`. If `None`,
            defaults to uniform weights of 1.0 (standard weighted
            least-squares without spatial fairness). Normalised to unit mean
            internally so the loss scale stays decoupled from the caller's
            weight convention.

        Returns
        -------
        self : object
            The fitted estimator.
        """

        X, y = check_X_y(X, y, accept_sparse=False, multi_output=True)
        n_samples, n_inputs = X.shape

        expected_inputs = self.n_features * self.n_time_bins
        if n_inputs != expected_inputs:
            raise ValueError(
                f"Input X has {n_inputs} columns, but init parameters expect "
                f"n_features({self.n_features}) * n_time_bins({self.n_time_bins}) = {expected_inputs} columns."
            )

        if y.shape[1] != 2:
            raise ValueError(f"Target y must have exactly 2 columns (UMAP X and Y). Found shape {y.shape}.")

        if sample_weight is None:
            sample_weight = np.ones(n_samples)
        else:
            sample_weight = check_array(sample_weight, ensure_2d=False)
            if sample_weight.shape[0] != n_samples:
                raise ValueError(
                    f"sample_weight has length {sample_weight.shape[0]}, but X has "
                    f"{n_samples} samples; lengths must match."
                )

        # Guard against a degenerate (all-zero / non-positive-mean) weight
        # vector: dividing by `np.mean + 1e-12` would not raise but would
        # collapse every weight to ~0, so the model would train only on the
        # L2 + smoothness regularisers with no data signal. Mirror the
        # `circular_mean` / `total_dispersion` convention in `manifold_metric`,
        # which raise on non-positive weight sums.
        if np.mean(sample_weight) <= 0:
            raise ValueError(
                "sample_weight must have a positive mean; an all-zero / non-positive "
                "weight vector would silently zero out the data term of the loss."
            )

        # Normalise to unit mean so the loss magnitude is invariant to the
        # caller's weighting convention (raw KDE densities, counts, or
        # already-normalised arrays). Keeps `lambda_smooth` / `l2_reg`
        # decoupled from the absolute scale of `sample_weight`.
        sample_weight = sample_weight / (np.mean(sample_weight) + 1e-12)

        X_j = jnp.array(X)
        Y_j = jnp.array(y)
        w_j = jnp.array(sample_weight)

        rng = jax.random.PRNGKey(self.random_state)
        params = self._initialize_params(n_inputs, rng)

        # Must build the SAME optax chain as `_bivariate_train_loop_jit`: the
        # `opt_state` initialised here is handed to that JIT'd loop, so the two
        # optimiser pytrees have to be structurally identical (adding the clip
        # transform changes the state pytree).
        scheduler = optax.cosine_decay_schedule(
            init_value=self.learning_rate, decay_steps=self.max_iter,
        )
        optimizer = optax.chain(
            optax.clip_by_global_norm(self.grad_clip_norm),
            optax.adam(scheduler),
        )
        opt_state = optimizer.init(params)

        @partial(jax.jit, static_argnums=(5, 6))
        def step(params, opt_state, X_batch, Y_batch, w_batch, n_feats, n_time):
            grads = jax.grad(self._loss_fn)(
                params, X_batch, Y_batch, w_batch,
                n_feats, n_time,
                self.lambda_smooth, self.l2_reg, self.huber_delta,
                self.smoothness_derivative_order,
                self.metric, self.period,
            )
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state

        if self.verbose:
            print(f"Starting 2-D JAX regression for up to {self.max_iter} iterations...")

        fit_start = time.perf_counter()

        if not self._use_lax_loop:
            # Default path: Python for-loop driving a JIT-compiled
            # `step`. Compiles once for the first iteration and reuses
            # the cached graph for every subsequent step. Safe at any
            # problem size and any caller pattern (including the joint
            # inner-CV tuner that builds many short-lived estimators).
            last_check_params = params
            converged = False
            completed_iter = 0
            for i in range(self.max_iter):
                params, opt_state = step(params, opt_state, X_j, Y_j, w_j, self.n_features, self.n_time_bins)
                completed_iter = i + 1
                if i > 0 and i % 100 == 0:
                    w_diff = jnp.linalg.norm(params[0] - last_check_params[0])
                    b_diff = jnp.linalg.norm(params[1] - last_check_params[1])
                    diff = jnp.sqrt(w_diff ** 2 + b_diff ** 2)
                    if diff < self.tol:
                        if self.verbose:
                            print(f"Converged at iteration {i} with combined-update norm {diff:.2e}")
                        converged = True
                        break
                    last_check_params = params
        else:
            # Fused path: full descent runs inside a single JIT-compiled
            # `jax.lax.while_loop` defined at module scope. The cache is
            # keyed on shape + the static integers (`n_feats`, `n_time`,
            # `smoothness_derivative_order`); every per-fit scalar
            # (regularisation strengths, learning rate, tolerance,
            # iteration cap) is a traced argument. This means a tuner
            # that constructs many short-lived estimators with the same
            # input shapes hits the cached compile for every estimator
            # after the first.
            params, completed_iter_j, converged_j = _bivariate_train_loop_jit(
                params,
                opt_state,
                X_j, Y_j, w_j,
                jnp.asarray(self.lambda_smooth, dtype=jnp.float32),
                jnp.asarray(self.l2_reg, dtype=jnp.float32),
                jnp.asarray(self.huber_delta, dtype=jnp.float32),
                jnp.asarray(self.learning_rate, dtype=jnp.float32),
                jnp.asarray(self.grad_clip_norm, dtype=jnp.float32),
                jnp.asarray(self.tol, dtype=jnp.float32),
                int(self.max_iter),
                int(self.n_features),
                int(self.n_time_bins),
                int(self.smoothness_derivative_order),
                self.metric,
                jnp.asarray(self.period, dtype=jnp.float32),
            )
            completed_iter = int(completed_iter_j)
            converged = bool(converged_j)

        self.coef_ = np.array(params[0])
        self.intercept_ = np.array(params[1])
        self.n_iter_ = int(completed_iter)
        self.converged_ = bool(converged)
        self.fit_time_ = float(time.perf_counter() - fit_start)

        # Store the training targets + their precomputed spatial structures.
        # The kd-tree powers `predict(snap=True)`, projecting raw linear
        # predictions onto the nearest observed training UMAP point so the
        # model never extrapolates outside the convex support its metrics are
        # compared against. The inverse-density-weighted covariance of
        # `Y_train` powers the Mahalanobis metric — a standardized distance
        # that removes the axis-scale arbitrariness of the UMAP embedding.
        # On `metric='torus'` the kdtree is built on the canonical
        # 4-D torus embedding `(cos, sin, cos, sin)` so a 1-NN query
        # respects wraparound: a raw prediction at `(0.99, 0.5)` snaps
        # to a training point at `(0.01, 0.5)` rather than to a far-away
        # point that happens to be closer in flat-space Euclidean.
        Y_train_np = np.asarray(y, dtype=np.float64)
        self.Y_train_ = Y_train_np
        try:
            if self.metric == 'torus':
                self._train_kdtree = cKDTree(torus_embed(Y_train_np, self.period))
            else:
                self._train_kdtree = cKDTree(Y_train_np)
        except Exception:
            self._train_kdtree = None

        # Re-normalise the already-unit-mean sample weights to unit sum so
        # the weighted mean / covariance correspond to the density-weighted
        # empirical distribution of the training manifold.
        # On `metric='torus'` the centroid is the per-axis circular mean
        # and the covariance is built from the wrap-aware signed
        # residuals around that centroid; the resulting Mahalanobis
        # metric is the same dimensionless "spread-units off" measure
        # as in the flat-space case but computed under the torus
        # geometry so it does not blow up near the wrap boundary.
        w_cov = sample_weight / (np.sum(sample_weight) + 1e-12)
        mu_train = circular_mean(Y_train_np, metric=self.metric, period=self.period, weights=w_cov)
        diff = signed_diff(Y_train_np, mu_train[None, :], metric=self.metric, period=self.period)
        cov = (w_cov[:, None] * diff).T @ diff
        # Pinv guards against singular covariance in degenerate folds
        # (e.g. very small test splits on a rare satellite cluster).
        self.train_mean_ = mu_train.astype(np.float64)
        self.train_cov_inv_ = np.linalg.pinv(cov)

        self.is_fitted_ = True

        return self

    def predict(self, X: np.ndarray, snap: bool = True) -> np.ndarray:
        """
        Predicts the 2-D UMAP coordinates of upcoming vocalisations.

        Parameters
        ----------
        X : np.ndarray
            Input behavioural history matrix, shape
            `(n_samples, n_features * n_time_bins)`.
        snap : bool, default=True
            When True (the default), each raw prediction `X @ W + b` is
            projected onto the nearest training UMAP point (1-NN in
            Euclidean distance over `Y_train_`) so the model's output is
            constrained to the same convex support as the training
            manifold and the null baselines. When False, returns the raw
            linear-map prediction, which can land outside the manifold
            support for extreme kinematic inputs — useful as a diagnostic
            for how often the unconstrained model extrapolates.

        Returns
        -------
        Y_pred : np.ndarray
            Predicted `(x, y)` coordinates, shape `(n_samples, 2)`. Dtype
            is `float64` in both snap modes — `Y_train_` is stored as
            `float64` during `fit` (regardless of caller-supplied dtype),
            and `np.dot(X, self.coef_)` returns `float64` for the
            `float32` inputs that the runner passes in. Callers that need
            a `float32` array (e.g., to preserve the runner's disk format)
            should cast explicitly with `.astype(np.float32)`.
        """

        check_is_fitted(self, ['coef_', 'intercept_'])
        X = check_array(X)
        raw = np.dot(X, self.coef_) + self.intercept_

        if snap and getattr(self, '_train_kdtree', None) is not None:
            # On torus the kdtree was built on the 4-D embedding, so the
            # query point must be embedded before the 1-NN lookup. The
            # returned index points back into the original (un-embedded)
            # `Y_train_`, so the snapped prediction is automatically in
            # native UMAP coordinates.
            if self.metric == 'torus':
                query = torus_embed(raw, self.period)
            else:
                query = raw
            _, idx = self._train_kdtree.query(query, k=1)
            return self.Y_train_[idx]
        return raw

    def evaluate_metrics(self, X: np.ndarray, Y_true: np.ndarray, weights: Optional[np.ndarray] = None) -> dict:
        """
        Evaluates the fitted model on test data and returns a metric bundle
        aligned with the univariate runner and the forward-selection routine.

        All Euclidean- and correlation-based metrics are computed on the
        **snapped** predictions (raw linear output projected onto the
        nearest observed training UMAP point), so the regressor competes
        against the null baselines on the same manifold support.
        `euclidean_mae_raw` is preserved as a diagnostic to quantify how
        often the unconstrained linear map extrapolates off-manifold.

        Metrics glossary
        ----------------
        - `r2_spatial` : pooled-axis coefficient of determination against
          the test-fold marginal mean. **Selection score.** Higher is
          better; bounded above by 1 for well-specified models and can be
          slightly negative when the model under-performs a constant
          test-fold-mean predictor.
        - `euclidean_mae` : mean Euclidean distance between snapped
          predictions and truth, in native UMAP units. Interpretable
          headline error magnitude for reporting.
        - `euclidean_rmse` : root-mean-squared Euclidean distance on the
          snapped predictions. `RMSE / MAE` well above `sqrt(pi / 2)
          ≈ 1.25` signals heavy-tailed residuals.
        - `euclidean_mae_weighted` : MAE on the Euclidean residual weighted
          by the caller-supplied inverse-density weights so rare satellite
          vocalisations carry the same influence as dense-core bouts.
        - `euclidean_mae_raw` : **diagnostic only.** MAE on the *raw*
          unsnapped linear predictions. The gap `euclidean_mae_raw -
          euclidean_mae` quantifies how much the manifold projection
          improves the fair-comparison number.
        - `mahalanobis_mae` : mean Mahalanobis distance of the residual in
          the inverse-density-weighted training covariance of `Y_train`.
          Removes the axis-scale arbitrariness of the UMAP embedding so a
          unit error on a tight axis is not silently treated the same as
          a unit error on a loose one. Dimensionless; lower is better.
        - `mae_x`, `mae_y` : per-axis absolute error on the snapped
          predictions. Useful when one UMAP axis is systematically easier
          to predict than the other.
        - `spearman_x`, `spearman_y` : per-axis rank correlation between
          snapped predictions and truth.
        - `pearson_x`, `pearson_y` : per-axis linear correlation on the
          natural scale.
        - `dcor_xy` : wrap-aware **distance correlation** between the decoded
          prediction and the truth (subsampled; see
          `manifold_metric.dcor_prediction_truth`). **This is the
          feature-selection score on the TORUS (`metric='torus'`);
          `r2_spatial` remains the score on Euclidean (VAE/UMAP) manifolds, so
          it is computed only on the torus path and is `nan` otherwise.** The
          reason is geometric. The QLVM torus is a *near-uniform, fully-covered
          periodic* latent space (a Quasi-Monte-Carlo / Fibonacci-lattice code,
          not a UMAP scatter): one axis is essentially uniform, the other only
          mildly structured, with density modulations but no spatial gaps. On
          such a target the circular centroid is ill-defined and the total
          dispersion is near-maximal, so `r2_spatial` — a centroid-referenced
          *squared-geodesic* score — is structurally inverted: a weak predictor
          that commits to a coordinate is punished *harder* than the do-nothing
          centroid, so real signal reads more negative than junk. Distance
          correlation instead measures *dependence* between the prediction and
          truth (any linear or non-linear association), is robust to that
          squared-error pathology, and — because it is scale-invariant and the
          prediction is `atan2`-decoded — is insensitive to ridge magnitude, so
          no regularisation tuning is needed on this path. Validated on the real
          100-fold cluster output (neck, nose-nose, allo-yaw clear a
          within-session-shuffle null at p<1e-4 .. 3e-4 under Wilcoxon +
          Bonferroni; `usv_cat` controls fail) — the same predictions on which
          `r2_spatial` found nothing. The screen therefore compares the torus
          `dcor_xy` against the **`null` (within-session-shuffle)** strategy,
          not the `null_model_free` empirical-density draw (which is
          independent of the per-trial truth, so its `dcor_xy` is only the
          finite-sample chance floor, not a confound-controlled baseline).

        Parameters
        ----------
        X : np.ndarray
            Input behavioural history matrix, shape
            `(n_samples, n_features * n_time_bins)`.
        Y_true : np.ndarray
            True continuous acoustic targets, shape `(n_samples, 2)`.
        weights : np.ndarray, optional
            Inverse-density sample weights for `euclidean_mae_weighted`.
            Ignored by the unweighted metrics.

        Returns
        -------
        metrics : dict
            Dictionary keyed by metric name (see glossary above).
        """

        check_is_fitted(self, ['coef_', 'intercept_'])

        # Snapped predictions — constrained to the training manifold,
        # which is the fair-comparison mode for every metric below. The
        # raw linear prediction is recomputed separately so we can expose
        # `euclidean_mae_raw` as a diagnostic without paying two kd-tree
        # queries. On torus the kdtree was built on the 4-D embedding so
        # the raw prediction must be embedded before the lookup; the
        # returned index points back into the original `Y_train_`.
        Y_pred_raw = self.predict(X, snap=False)
        if getattr(self, '_train_kdtree', None) is not None:
            if self.metric == 'torus':
                _, nn_idx = self._train_kdtree.query(torus_embed(Y_pred_raw, self.period), k=1)
            else:
                _, nn_idx = self._train_kdtree.query(Y_pred_raw, k=1)
            Y_pred = self.Y_train_[nn_idx]
        else:
            Y_pred = Y_pred_raw

        if weights is None:
            weights = np.ones(Y_true.shape[0])
        else:
            # Validate the caller-supplied weights the same way `fit` does:
            # a wrong-length or 2-D array would otherwise broadcast silently
            # in `euclidean_mae_weighted` and corrupt only that one metric.
            weights = check_array(weights, ensure_2d=False)
            if weights.shape[0] != Y_true.shape[0]:
                raise ValueError(
                    f"weights has length {weights.shape[0]}, but Y_true has "
                    f"{Y_true.shape[0]} samples; lengths must match."
                )

        # Wrap-aware signed differences. On `metric='euclidean'` this is
        # `Y_true - Y_pred` element-wise; on torus each component is
        # folded into `(-period/2, period/2]` so an axis-distance of
        # `period - epsilon` is reported as `epsilon`. Every euclidean-
        # named metric below derives from these wrapped residuals — the
        # naming is preserved for backwards compatibility, the semantic
        # is the metric the manifold actually has.
        residual_xy = signed_diff(Y_true, Y_pred, metric=self.metric, period=self.period)
        dx = residual_xy[:, 0]
        dy = residual_xy[:, 1]

        euclidean_dist = np.sqrt(dx ** 2 + dy ** 2)
        euclidean_mae = float(np.mean(euclidean_dist))
        euclidean_rmse = float(np.sqrt(np.mean(euclidean_dist ** 2)))
        euclidean_mae_weighted = float(
            np.sum(weights * euclidean_dist) / (np.sum(weights) + 1e-12)
        )

        # Raw (unsnapped) MAE — how far off the manifold the
        # unconstrained linear map lands on average. Same wrap rule.
        residual_xy_raw = signed_diff(Y_true, Y_pred_raw, metric=self.metric, period=self.period)
        euclidean_mae_raw = float(np.mean(np.sqrt(np.sum(residual_xy_raw ** 2, axis=1))))

        # Mahalanobis MAE using the (metric-aware) inverse-density-weighted
        # training covariance. Interprets "how many spread-units off"
        # rather than "how many raw UMAP units off", removing the
        # arbitrariness of the UMAP embedding's axis scales. On torus
        # both the residual and the covariance were built from
        # wrap-aware signed differences (see `fit`), so the metric is
        # internally consistent with the rest of the bundle.
        if getattr(self, 'train_cov_inv_', None) is not None:
            quad = np.einsum('ij,jk,ik->i', residual_xy, self.train_cov_inv_, residual_xy)
            # Numerical noise in near-singular covariances can push tiny
            # residuals slightly negative; clip before the sqrt.
            mahalanobis_mae = float(np.mean(np.sqrt(np.maximum(quad, 0.0))))
        else:
            mahalanobis_mae = float('nan')

        mae_x = float(np.mean(np.abs(dx)))
        mae_y = float(np.mean(np.abs(dy)))

        # Pearson inlined as a centred-dot-product / norm-product so we
        # skip scipy's argument validation on every fold. NaN-safe when a
        # fold happens to produce a constant prediction (e.g., an exotic
        # feature with near-zero variance). On torus, per-axis correlation
        # is computed on the raw values (not unwrapped) — small bias near
        # the wrap boundary, but standard pearson/spearman have no
        # principled circular generalisation that returns a single
        # comparable scalar; the headline R^2 below is the principled
        # joint-axis figure.
        def _pearson_np(a, b):
            a_c = a - a.mean()
            b_c = b - b.mean()
            denom = np.sqrt(np.sum(a_c ** 2) * np.sum(b_c ** 2))
            if denom <= 0:
                return float('nan')
            return float(np.dot(a_c, b_c) / denom)

        def _spearman(a, b):
            try:
                value = spearmanr(a, b)[0]
            except ValueError:
                return float('nan')
            return float(value) if np.isfinite(value) else float('nan')

        pearson_x = _pearson_np(Y_true[:, 0], Y_pred[:, 0])
        pearson_y = _pearson_np(Y_true[:, 1], Y_pred[:, 1])
        spearman_x = _spearman(Y_true[:, 0], Y_pred[:, 0])
        spearman_y = _spearman(Y_true[:, 1], Y_pred[:, 1])

        # Wrap-aware distance correlation between the decoded prediction and
        # the truth — the torus-manifold selection score (see the metrics
        # glossary). Computed ONLY on the torus path (it is O(n^2) and
        # subsampled); on euclidean it is `nan` and `r2_spatial` is the score.
        # The `null_model_free` baseline is an empirical-density draw (not a
        # constant centroid), so its `dcor_xy` is computed the same way and
        # lands at the finite-sample chance floor.
        if self.metric == 'torus':
            dcor_xy = dcor_prediction_truth(
                Y_pred, Y_true, metric=self.metric, period=self.period,
                random_state=self.random_state,
            )
        else:
            dcor_xy = float('nan')

        # `r2_spatial` numerator: sum of squared wrap-aware residuals.
        # Denominator: total wrap-aware dispersion of `Y_true` around
        # its own (metric-aware) centroid. Computing both terms under
        # the same metric keeps R^2 interpretable on a torus —
        # using `np.var(Y_true)` against a wrap-aware SSE would give
        # nonsense near the boundary.
        ss_res = float(np.sum(dx ** 2 + dy ** 2))
        denom = total_dispersion(Y_true, metric=self.metric, period=self.period)
        r2_spatial = float(1.0 - (ss_res / denom)) if denom > 0 else 0.0

        return {
            'r2_spatial': r2_spatial,
            'euclidean_mae': euclidean_mae,
            'euclidean_rmse': euclidean_rmse,
            'euclidean_mae_weighted': euclidean_mae_weighted,
            'euclidean_mae_raw': euclidean_mae_raw,
            'mahalanobis_mae': mahalanobis_mae,
            'mae_x': mae_x,
            'mae_y': mae_y,
            'pearson_x': pearson_x,
            'pearson_y': pearson_y,
            'spearman_x': spearman_x,
            'spearman_y': spearman_y,
            'dcor_xy': dcor_xy,
        }
