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
             + 0.5 * lam_smooth * sum over second-time-derivatives of W**2

- `w_i` is the inverse-density sample weight (KDE), normalised to unit mean
  so the loss scale is invariant to the caller's weighting convention and
  the regularisation hyper-parameters are decoupled from the weight scale.
- Huber is preferred over plain MSE so a handful of satellite vocalisations
  cannot dominate the gradient. The delta parameter is exposed via
  `huber_delta`; setting `huber_delta=np.inf` recovers pure squared loss.
- L2 constrains absolute weight magnitude; the temporal smoothness term
  penalises the discrete second derivative of the weights along the time
  axis, just as in the old Gaussian model, so learned kinematic filters
  remain biologically plausible smooth curves.

Outputs
-------
- `predict(X)`  -> (n_samples, 2) predicted UMAP coordinates.
- `evaluate_metrics(X, Y_true, weights=None)` returns a metric dictionary
  with `euclidean_mae`, `euclidean_rmse`, per-axis `mae_x` / `mae_y`,
  `spearman_x` / `spearman_y`, `pearson_x` / `pearson_y`, and `r2_spatial`.
  Every metric is evaluated on the natural UMAP scale.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import time
from functools import partial
from scipy.stats import pearsonr, spearmanr
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from typing import Tuple, Any, Optional


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
    huber_delta : float, default=1.0
        Transition point of the Huber loss in the native UMAP distance
        units. Residuals with `||r||_2 <= huber_delta` are penalised
        quadratically; larger residuals are penalised linearly. Set to
        `np.inf` to recover pure squared-error regression.
    learning_rate : float, default=1e-3
        Step size for the Adam optimiser.
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
            huber_delta: float = 1.0,
            learning_rate: float = 1e-3,
            max_iter: int = 5000,
            tol: float = 1e-4,
            random_state: int = 0,
            verbose: bool = False
    ):
        self.n_features = n_features
        self.n_time_bins = n_time_bins
        self.lambda_smooth = lambda_smooth
        self.l2_reg = l2_reg
        self.huber_delta = huber_delta
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose

    @staticmethod
    def _initialize_params(n_inputs: int, key: Any) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Initialises the linear-transformation weights `W` and bias `b`.

        Weights are initialised with Xavier/Glorot normal scaling so variance
        gradients stay stable in the early Adam steps. Biases start at zero
        and are learned alongside the weights.

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
    @partial(jax.jit, static_argnums=(4, 5))
    def _loss_fn(
            params: Tuple[jnp.ndarray, jnp.ndarray],
            X: jnp.ndarray,
            Y_true: jnp.ndarray,
            weights: jnp.ndarray,
            n_feats: int,
            n_time: int,
            lam_smooth: float,
            lam_l2: float,
            huber_delta: float
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
            Number of distinct physical behavioural features.
        n_time : int
            Number of time bins per behavioural feature.
        lam_smooth : float
            Temporal-smoothness penalty strength.
        lam_l2 : float
            L2 (Ridge) penalty strength.
        huber_delta : float
            Huber transition point in native UMAP distance units.

        Returns
        -------
        loss : jnp.ndarray
            Scalar total loss.
        """

        W, b = params

        # Forward pass: linear projection directly to the 2-D output.
        Y_pred = jnp.dot(X, W) + b

        # Huber loss on the Euclidean residual (not per-axis): a 1-unit miss
        # along either coordinate is treated identically, and the quadratic
        # region is ||r|| <= huber_delta.
        residuals = Y_true - Y_pred
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

        # Temporal smoothness: penalise the discrete second derivative of the
        # weights along the time axis of each (feature, output-dim) slice.
        W_reshaped = W.reshape(n_feats, n_time, 2)
        d2w = jnp.diff(W_reshaped, n=2, axis=1)
        smooth_loss = 0.5 * lam_smooth * jnp.sum(d2w ** 2)

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

        optimizer = optax.adam(self.learning_rate)
        opt_state = optimizer.init(params)

        @partial(jax.jit, static_argnums=(5, 6))
        def step(params, opt_state, X_batch, Y_batch, w_batch, n_feats, n_time):
            grads = jax.grad(self._loss_fn)(
                params, X_batch, Y_batch, w_batch,
                n_feats, n_time,
                self.lambda_smooth, self.l2_reg, self.huber_delta
            )
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state

        if self.verbose:
            print(f"Starting 2-D JAX regression for up to {self.max_iter} iterations...")

        # Cache parameters at the last check-point so the convergence
        # diagnostic measures cumulative change over the previous 100 steps
        # (a single-step delta would collapse trivially near the optimum).
        last_check_params = params
        fit_start = time.perf_counter()
        converged = False
        completed_iter = 0

        for i in range(self.max_iter):
            params, opt_state = step(params, opt_state, X_j, Y_j, w_j, self.n_features, self.n_time_bins)
            completed_iter = i + 1

            if i > 0 and i % 100 == 0:
                diff = jnp.linalg.norm(params[0] - last_check_params[0])
                if diff < self.tol:
                    if self.verbose:
                        print(f"Converged at iteration {i} with weight-update norm {diff:.2e}")
                    converged = True
                    break
                last_check_params = params

        self.coef_ = np.array(params[0])
        self.intercept_ = np.array(params[1])
        self.n_iter_ = int(completed_iter)
        self.converged_ = bool(converged)
        self.fit_time_ = float(time.perf_counter() - fit_start)
        self.is_fitted_ = True

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the deterministic 2-D UMAP coordinates of upcoming vocalisations.

        Parameters
        ----------
        X : np.ndarray
            Input behavioural history matrix, shape
            `(n_samples, n_features * n_time_bins)`.

        Returns
        -------
        Y_pred : np.ndarray
            Predicted `(mu_x, mu_y)` coordinates, shape `(n_samples, 2)`.
        """

        check_is_fitted(self, ['coef_', 'intercept_'])
        X = check_array(X)
        return np.dot(X, self.coef_) + self.intercept_

    def evaluate_metrics(self, X: np.ndarray, Y_true: np.ndarray, weights: Optional[np.ndarray] = None) -> dict:
        """
        Evaluates the fitted model on test data and returns a metric bundle
        aligned with the expectations of the univariate runner and the
        forward-selection routine.

        Metrics glossary
        ----------------
        - `euclidean_mae` : Mean Euclidean distance between predicted and
          true UMAP coordinates. Interpretable error magnitude in native
          UMAP units (lower is better). This is the headline regression
          score.
        - `euclidean_rmse` : Root-mean-squared Euclidean distance.
          Complements MAE; an `RMSE / MAE` ratio substantially greater than
          `sqrt(pi / 2) ≈ 1.25` signals heavy-tailed residuals or outlier
          test folds.
        - `euclidean_mae_weighted` : MAE on the Euclidean residual weighted
          by the caller-supplied inverse-density weights so rare satellite
          vocalisations carry the same influence as dense-core bouts.
        - `mae_x`, `mae_y` : Per-axis Mean Absolute Error. Useful when one
          UMAP axis is systematically easier to predict than the other.
        - `spearman_x`, `spearman_y` : Per-axis rank correlation between
          predictions and truth (robust to monotonic transformations of the
          embedding).
        - `pearson_x`, `pearson_y` : Per-axis linear correlation on the
          natural scale. Complements Spearman by detecting magnitude
          compression / expansion.
        - `r2_spatial` : Fraction of total spatial variance (pooled across
          both axes) explained by the predictions. Bounded above by 1 for
          well-specified models.

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

        Y_pred = self.predict(X)

        if weights is None:
            weights = np.ones(Y_true.shape[0])

        dx = Y_true[:, 0] - Y_pred[:, 0]
        dy = Y_true[:, 1] - Y_pred[:, 1]

        euclidean_dist = np.sqrt(dx ** 2 + dy ** 2)
        euclidean_mae = float(np.mean(euclidean_dist))
        euclidean_rmse = float(np.sqrt(np.mean(euclidean_dist ** 2)))
        # Use the raw caller-supplied weights rather than internally
        # normalising so reviewers can directly compare this number with the
        # unweighted MAE for the same fold.
        euclidean_mae_weighted = float(
            np.sum(weights * euclidean_dist) / (np.sum(weights) + 1e-12)
        )

        mae_x = float(np.mean(np.abs(dx)))
        mae_y = float(np.mean(np.abs(dy)))

        # Correlation helpers are NaN-safe when a fold happens to produce a
        # constant prediction (e.g., an exotic feature with near-zero
        # variance).
        def _pearson(a, b):
            if np.std(a) == 0 or np.std(b) == 0:
                return float('nan')
            try:
                return float(pearsonr(a, b)[0])
            except ValueError:
                return float('nan')

        def _spearman(a, b):
            try:
                value = spearmanr(a, b)[0]
            except ValueError:
                return float('nan')
            return float(value) if np.isfinite(value) else float('nan')

        pearson_x = _pearson(Y_true[:, 0], Y_pred[:, 0])
        pearson_y = _pearson(Y_true[:, 1], Y_pred[:, 1])
        spearman_x = _spearman(Y_true[:, 0], Y_pred[:, 0])
        spearman_y = _spearman(Y_true[:, 1], Y_pred[:, 1])

        ss_res = np.sum(dx ** 2 + dy ** 2)
        ss_tot_x = np.sum((Y_true[:, 0] - np.mean(Y_true[:, 0])) ** 2)
        ss_tot_y = np.sum((Y_true[:, 1] - np.mean(Y_true[:, 1])) ** 2)
        denom = ss_tot_x + ss_tot_y
        r2_spatial = float(1.0 - (ss_res / denom)) if denom > 0 else 0.0

        return {
            'euclidean_mae': euclidean_mae,
            'euclidean_rmse': euclidean_rmse,
            'euclidean_mae_weighted': euclidean_mae_weighted,
            'mae_x': mae_x,
            'mae_y': mae_y,
            'pearson_x': pearson_x,
            'pearson_y': pearson_y,
            'spearman_x': spearman_x,
            'spearman_y': spearman_y,
            'r2_spatial': r2_spatial,
        }
