"""
@author: bartulem
JAX-based continuous probabilistic regression for modeling 2D acoustic manifolds.

This module implements a custom scikit-learn compatible estimator that maps
high-dimensional behavioral kinematics (X) to a continuous 2D acoustic space (Y).
Instead of forcing vocalizations into discrete categories, this model performs
density estimation by outputting the 5 parameters of a Bivariate Gaussian distribution
for every specific instance in time.

Mathematical Formulation (Homoscedastic Architecture):
------------------------------------------------------
To prevent massive gradient explosions (the heteroscedastic trap), this model
decouples the spatial prediction from the variance prediction:
1. The behavioral history matrix (X) linearly predicts ONLY the spatial center (mu_x, mu_y).
2. The structural uncertainty (sigma_x, sigma_y, rho) is learned as a set of global,
   freestanding parameters optimized for the overall fold, independent of individual
   kinematic fluctuations.

The network achieves this via specific activation functions:
- Identity function for means (unbounded space).
- Softplus function for standard deviations (strictly positive).
- Tanh function for correlation (bounded strictly between -1 and 1).

The optimization objective is the Density-Weighted Negative Log-Likelihood (NLL).
By weighting the NLL of each sample by its inverse spatial density, we mathematically
force the optimizer to care equally about rare "satellite" vocalizations and the
dense acoustic core.

Regularization:
---------------
1. L2 Regularization (Ridge): Penalizes absolute weight magnitude to prevent overfitting.
2. Temporal Smoothness: Penalizes the discrete second derivative of the weights across
   time bins, forcing the learned kinematic filters to take the shape of smooth,
   biologically plausible temporal curves rather than jagged noise.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
from functools import partial
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from typing import Tuple, Any, Optional


class SmoothBivariateGaussianRegression(BaseEstimator, RegressorMixin):
    """
    Continuous probabilistic regression with temporal smoothing (JAX implementation).

    This estimator learns a spatial weight matrix W of shape (n_features * n_time_bins, 2)
    and a freestanding global variance array of shape (3,). It minimizes the density-weighted
    Negative Log-Likelihood of a Bivariate Gaussian, augmented by L2 and temporal smoothing penalties.

    Parameters
    ----------
    n_features : int
        Number of distinct behavioral features (e.g., speed, distance).
        Crucial for reshaping the flat coefficient vector back into (features, time, 2)
        to calculate the discrete second derivative for temporal smoothing.
    n_time_bins : int
        Number of time steps per feature. The input X must have exactly
        `n_features * n_time_bins` columns.
    lambda_smooth : float, default=1e2
        Strength of the temporal smoothing penalty. Penalizes the squared second
        derivative of the weights across the time axis. Higher values yield stiffer curves.
    l2_reg : float, default=0.1
        Strength of the standard L2 (Ridge) penalty to constrain overall weight magnitude.
    learning_rate : float, default=1e-3
        Step size for the Adam optimizer.
    max_iter : int, default=5000
        Maximum number of optimization steps (epochs) allowed before termination.
    tol : float, default=1e-4
        Convergence tolerance. Training stops early if the L2 norm of the parameter
        updates falls below this threshold between evaluation checks.
    random_state : int, default=0
        Seed for the JAX PRNG key to ensure perfectly reproducible weight initialization.
    verbose : bool, default=False
        If True, prints the loss progress periodically during gradient descent.
    """

    def __init__(
            self,
            n_features: int = 1,
            n_time_bins: int = 1,
            lambda_smooth: float = 1e2,
            l2_reg: float = 0.1,
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
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose

    @staticmethod
    def _initialize_params(n_inputs: int, key: Any) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Initializes the linear transformation weights (W), biases (b), and global variance parameters.

        Weights are initialized using Xavier/Glorot normal distribution to ensure
        stable variance gradients in the early steps of Adam optimization.

        Parameters
        ----------
        n_inputs : int
            Total number of input features (columns in the flattened X matrix).
        key : jax.random.PRNGKey
            JAX random key for deterministic generation.

        Returns
        -------
        W_mu : jnp.ndarray
            Initialized weight matrix of shape (n_inputs, 2) for spatial coordinates.
        b_mu : jnp.ndarray
            Initialized bias vector of shape (2,) for spatial coordinates.
        global_var_params : jnp.ndarray
            Initialized freestanding variance parameters of shape (3,).
        """
        k1, k2 = jax.random.split(key)

        # 2 output nodes for the spatial coordinates (mu_x, mu_y)
        n_outputs = 2

        # Xavier/Glorot scaling factor for the spatial weights
        scale = jnp.sqrt(2.0 / (n_inputs + n_outputs))

        W_mu = jax.random.normal(k1, (n_inputs, n_outputs)) * scale
        b_mu = jnp.zeros((n_outputs,))

        # 3 freestanding global parameters for (sigma_x, sigma_y, rho).
        # Initialized at 0.0 -> Softplus(0) is approx 0.69 (a safe initial variance)
        global_var_params = jnp.zeros((3,))

        return W_mu, b_mu, global_var_params

    @staticmethod
    def _bivariate_gaussian_layer(mu_logits: jnp.ndarray, var_logits: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Applies mathematical constraints to the raw neural network logits to
        produce valid parameters for a Bivariate Gaussian distribution.

        Parameters
        ----------
        mu_logits : jnp.ndarray
            Raw linear outputs of shape (N, 2) from the equation XW + b.
        var_logits : jnp.ndarray
            Global variance parameters of shape (3,).

        Returns
        -------
        mu_x, mu_y : jnp.ndarray
            Spatial means. Identity activation (unbounded).
        sigma_x, sigma_y : jnp.ndarray
            Standard deviations. Softplus activation ensures strictly positive values.
            A tiny epsilon is added to prevent log(0) domain errors.
        rho : jnp.ndarray
            Correlation coefficient. Tanh activation bounds the value to (-1, 1).
            Scaled slightly to prevent mathematically unstable matrix singularities.
        """
        epsilon = 1e-6

        # Identity for spatial coordinates (they can exist anywhere on the UMAP)
        mu_x = mu_logits[:, 0]
        mu_y = mu_logits[:, 1]

        # Softplus for global variance (uncertainty must be a positive, non-zero number)
        sigma_x = jax.nn.softplus(var_logits[0]) + epsilon
        sigma_y = jax.nn.softplus(var_logits[1]) + epsilon

        # Tanh for global correlation (must be between -1 and 1)
        # Scaled by (1 - epsilon) to prevent division by zero in the NLL formula
        rho = jnp.tanh(var_logits[2]) * (1.0 - epsilon)

        return mu_x, mu_y, sigma_x, sigma_y, rho

    @staticmethod
    @partial(jax.jit, static_argnums=(4, 5))
    def _loss_fn(
            params: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
            X: jnp.ndarray,
            Y_true: jnp.ndarray,
            weights: jnp.ndarray,
            n_feats: int,
            n_time: int,
            lam_smooth: float,
            lam_l2: float
    ) -> jnp.ndarray:
        """
        Computes the total optimization loss: Density-Weighted NLL + L2 + Smoothness.

        This is the mathematical core of the probabilistic framework. It measures how
        well the predicted probability density contours encompass the true acoustic
        coordinates, applies the inverse-density spatial fairness weights, and adds
        the kinematic structural penalties.

        Parameters
        ----------
        params : tuple
            (W_mu, b_mu, global_var_params) tuple of current model parameters.
        X : jnp.ndarray
            Input behavioral history matrix of shape (Samples, Features * Time).
        Y_true : jnp.ndarray
            True continuous acoustic targets of shape (Samples, 2).
        weights : jnp.ndarray
            Inverse-density sample weights of shape (Samples,) computed via KDE.
        n_feats : int
            Number of distinct physical behavioral features.
        n_time : int
            Number of time bins per behavioral feature.
        lam_smooth : float
            Smoothing penalty strength.
        lam_l2 : float
            L2 (Ridge) penalty strength.

        Returns
        -------
        loss : jnp.ndarray
            Scalar representing the total computed loss for the batch.
        """
        W_mu, b_mu, global_var_params = params

        # Forward pass (linear projection predicts ONLY spatial location)
        mu_logits = jnp.dot(X, W_mu) + b_mu

        # Activation functions
        mu_x, mu_y, sigma_x, sigma_y, rho = SmoothBivariateGaussianRegression._bivariate_gaussian_layer(mu_logits, global_var_params)

        # Calculate Mahalanobis-like Z-score term for the Bivariate Gaussian
        dx = Y_true[:, 0] - mu_x
        dy = Y_true[:, 1] - mu_y

        z = ((dx / sigma_x) ** 2
             + (dy / sigma_y) ** 2
             - 2 * rho * (dx / sigma_x) * (dy / sigma_y))

        # Compute the Negative Log-Likelihood (NLL) for each sample
        # Formula: log(2 * pi * sigma_x * sigma_y * sqrt(1 - rho^2)) + z / (2 * (1 - rho^2))
        log_term = (jnp.log(2 * jnp.pi)
                    + jnp.log(sigma_x)
                    + jnp.log(sigma_y)
                    + 0.5 * jnp.log(1 - rho ** 2))

        nll_instance = log_term + (z / (2 * (1 - rho ** 2)))

        # Apply the geographic fairness (inverse-density) weights
        # Because the pipeline normalizes weights to have a mean of 1, multiplying and taking the mean
        # effectively scales the loss landscape without changing its global learning-rate magnitude.
        weighted_nll = jnp.mean(weights * nll_instance)

        # Apply L2 (Ridge) regularization to the spatial weights
        l2_loss = 0.5 * lam_l2 * jnp.sum(W_mu ** 2)

        # Apply temporal smoothness penalty (L2 on the second derivative)
        # Reshape flat W back into (features, time, 2 spatial parameters)
        W_reshaped = W_mu.reshape(n_feats, n_time, 2)

        # Calculate discrete acceleration (curvature) along the time axis (axis 1)
        d2w = jnp.diff(W_reshaped, n=2, axis=1)
        smooth_loss = 0.5 * lam_smooth * jnp.sum(d2w ** 2)

        # Total objective function
        return weighted_nll + l2_loss + smooth_loss

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> "SmoothBivariateGaussianRegression":
        """
        Fits the continuous probabilistic model using JAX/Optax optimization.

        Parameters
        ----------
        X : np.ndarray
            Training vectors, shape (n_samples, n_features * n_time_bins).
        y : np.ndarray
            Target UMAP coordinates, shape (n_samples, 2).
        sample_weight : np.ndarray, optional
            Inverse density weights, shape (n_samples,). If None, defaults to uniform
            weights of 1.0 (standard MLE without spatial fairness).

        Returns
        -------
        self : object
            The fitted estimator, holding the frozen weights and intercepts.
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

        X_j = jnp.array(X)
        Y_j = jnp.array(y)
        w_j = jnp.array(sample_weight)

        rng = jax.random.PRNGKey(self.random_state)
        params = self._initialize_params(n_inputs, rng)

        optimizer = optax.adam(self.learning_rate)
        opt_state = optimizer.init(params)

        @partial(jax.jit, static_argnums=(5, 6))
        def step(params, opt_state, X_batch, Y_batch, w_batch, n_feats, n_time):
            # Calculate gradients with respect to the loss function
            grads = jax.grad(self._loss_fn)(
                params, X_batch, Y_batch, w_batch,
                n_feats, n_time,
                self.lambda_smooth, self.l2_reg
            )
            # Compute and apply Adam updates
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state

        if self.verbose:
            print(f"Starting Homoscedastic JAX optimization for {self.max_iter} iterations...")

        for i in range(self.max_iter):
            old_params = params
            params, opt_state = step(params, opt_state, X_j, Y_j, w_j, self.n_features, self.n_time_bins)

            # Convergence check every 100 iterations
            if i > 0 and i % 100 == 0:
                diff = jnp.linalg.norm(params[0] - old_params[0])
                if diff < self.tol:
                    if self.verbose:
                        print(f"Converged at iteration {i} with weight update norm {diff:.2e}")
                    break

        # Store the learned parameters in standard scikit-learn format
        self.coef_ = np.array(params[0])
        self.intercept_ = np.array(params[1])
        self.global_var_params_ = np.array(params[2])
        self.is_fitted_ = True

        return self

    def predict_density(self, X: np.ndarray) -> np.ndarray:
        """
        Computes the complete continuous probability landscape for new data.

        This replaces the standard `predict_proba` method from classification.
        It returns the exact structural uncertainty and spatial mapping inferred
        by the kinematic history.

        Parameters
        ----------
        X : np.ndarray
            Input behavioral history matrix, shape (n_samples, n_features).

        Returns
        -------
        params_array : np.ndarray
            A highly interpretable matrix of shape (n_samples, 5) containing:
            Column 0: mu_x    (Predicted UMAP X center)
            Column 1: mu_y    (Predicted UMAP Y center)
            Column 2: sigma_x (Uncertainty footprint along X axis)
            Column 3: sigma_y (Uncertainty footprint along Y axis)
            Column 4: rho     (Topographical rotation/correlation)
        """
        check_is_fitted(self, ['coef_', 'intercept_', 'global_var_params_'])
        X = check_array(X)
        n_samples = X.shape[0]

        # Predict deterministic spatial locations
        mu_logits = jnp.dot(jnp.array(X), jnp.array(self.coef_)) + jnp.array(self.intercept_)

        # Extract scalar global variances
        mu_x, mu_y, sig_x_scalar, sig_y_scalar, rho_scalar = self._bivariate_gaussian_layer(
            mu_logits, jnp.array(self.global_var_params_)
        )

        # Broadcast the global scalar variances to shape (n_samples,)
        sigma_x = np.full(n_samples, float(sig_x_scalar))
        sigma_y = np.full(n_samples, float(sig_y_scalar))
        rho = np.full(n_samples, float(rho_scalar))

        return np.column_stack((mu_x, mu_y, sigma_x, sigma_y, rho))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the deterministic spatial center of the acoustic target.

        This serves as a standard regressor output, discarding the model's
        structural uncertainty and returning only the geographic spatial means.
        Used to calculate standard Euclidean distance or R-squared metrics.

        Parameters
        ----------
        X : np.ndarray
            Input behavioral history matrix, shape (n_samples, n_features).

        Returns
        -------
        means : np.ndarray
            The predicted (mu_x, mu_y) coordinates, shape (n_samples, 2).
        """
        density_params = self.predict_density(X)
        return density_params[:, :2]

    def evaluate_metrics(self, X: np.ndarray, Y_true: np.ndarray, weights: Optional[np.ndarray] = None) -> dict:
        """
        Evaluates the fitted model on test data and returns the comprehensive suite
        of probabilistic and physical performance metrics.

        Parameters
        ----------
        X : np.ndarray
            Input behavioral history matrix, shape (n_samples, n_features).
        Y_true : np.ndarray
            True continuous acoustic targets, shape (n_samples, 2).
        weights : np.ndarray, optional
            Inverse-density sample weights for the weighted NLL.

        Returns
        -------
        metrics : dict
            A dictionary containing: nll_raw, nll_weighted, euclidean_mae,
            mahalanobis_dist, and r2_spatial.
        """
        check_is_fitted(self, ['coef_', 'intercept_', 'global_var_params_'])

        if weights is None:
            weights = np.ones(Y_true.shape[0])

        density_params = self.predict_density(X)
        mu_x = density_params[:, 0]
        mu_y = density_params[:, 1]
        sigma_x = density_params[:, 2]
        sigma_y = density_params[:, 3]
        rho = density_params[:, 4]

        # Compute spatial error (Euclidean distances)
        dx = Y_true[:, 0] - mu_x
        dy = Y_true[:, 1] - mu_y
        euclidean_dist = np.sqrt(dx ** 2 + dy ** 2)
        euclidean_mae = float(np.mean(euclidean_dist))

        # Compute probabilistic error (Mahalanobis / z-score)
        z = ((dx / sigma_x) ** 2
             + (dy / sigma_y) ** 2
             - 2 * rho * (dx / sigma_x) * (dy / sigma_y))

        mahalanobis_dist = float(np.mean(np.sqrt(z / (1 - rho ** 2))))

        # Compute Negative Log-Likelihoods (raw and weighted)
        log_term = (np.log(2 * np.pi)
                    + np.log(sigma_x)
                    + np.log(sigma_y)
                    + 0.5 * np.log(1 - rho ** 2))

        nll_instance = log_term + (z / (2 * (1 - rho ** 2)))

        nll_raw = float(np.mean(nll_instance))
        nll_weighted = float(np.mean(weights * nll_instance))

        # Compute spatial R-squared (variance explained by the means)
        ss_res = np.sum(dx ** 2 + dy ** 2)
        ss_tot_x = np.sum((Y_true[:, 0] - np.mean(Y_true[:, 0])) ** 2)
        ss_tot_y = np.sum((Y_true[:, 1] - np.mean(Y_true[:, 1])) ** 2)
        r2_spatial = float(1 - (ss_res / (ss_tot_x + ss_tot_y)))

        return {
            'nll_raw': nll_raw,
            'nll_weighted': nll_weighted,
            'euclidean_mae': euclidean_mae,
            'mahalanobis_dist': mahalanobis_dist,
            'r2_spatial': r2_spatial
        }
