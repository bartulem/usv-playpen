"""
@author: bartulem
JAX-based multinomial logistic regression with temporal smoothing.

This module implements a custom scikit-learn compatible estimator that solves
multinomial logistic regression with two forms of regularization:
1. Standard L2 (Ridge) on the weights.
2. Temporal smoothing (L2 on the second derivative) to force weights to vary
   smoothly over time indices.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
from functools import partial
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.preprocessing import LabelBinarizer
from typing import Tuple, Any


class SmoothMultinomialLogisticRegression(BaseEstimator, ClassifierMixin):
    """
    Multinomial logistic regression with temporal smoothing (JAX implementation).

    This estimator learns a weight matrix W of shape (n_features * n_time_bins, n_classes).
    It minimizes an alpha-balanced focal loss augmented by:
    1. L1 regularization (lasso) which promotes sparsity in the weights.
    2. L2 regularization (ridge) which penalizes large weights.
    3. Temporal smoothing penalty which penalizes the second derivative of weights
       along the time axis, forcing the learned filters to be smooth curves. The
       per-class smoothing penalty is additionally scaled by the inverse-frequency
       class weight, so rare-class filters are regularized more strongly (stronger
       prior where data is thin).

    Parameters
    ----------
    n_features : int
        Number of distinct physical features (e.g., speed, distance).
        Crucial for reshaping the flat coefficient vector back into (features, time)
        to apply smoothing correctly.
    n_time_bins : int
        Number of time steps per feature.
        The input X must have `n_features * n_time_bins` columns.
    lambda_smooth : float, default=1
        Regularization strength for the temporal smoothing penalty (second derivative).
        Higher values force smoother (stiffer) curves.
    l1_reg : float, default=0.0
        Regularization strength for standard L1 (Lasso) penalty.
    l2_reg : float, default=0.1
        Regularization strength for standard L2 (Ridge) penalty.
    focal_gamma : float, default=2.0
        Focusing parameter of the focal loss: the `(1 - p_t) ** focal_gamma`
        modulator down-weights easy examples so gradient flow concentrates on
        hard / misclassified samples. Set to 0.0 to recover plain alpha-balanced
        cross-entropy.
    learning_rate : float, default=1e-3
        Step size for the Adam optimizer.
    max_iter : int, default=5000
        Maximum number of optimization steps (epochs).
    tol : float, default=1e-4
        Convergence tolerance. Training stops if the parameters change by less
        than this amount between checks.
    random_state : int, default=0
        Seed for JAX random number generation initialization.
    verbose : bool, default=False
        If True, prints loss progress during training.
    """

    def __init__(
            self,
            n_features: int = 1,
            n_time_bins: int = 1,
            lambda_smooth: float = 1,
            l1_reg: float = 0.0,
            l2_reg: float = 0.1,
            focal_gamma: float = 2.0,
            learning_rate: float = 1e-3,
            max_iter: int = 5000,
            tol: float = 1e-4,
            random_state: int = 0,
            verbose: bool = False
    ):
        self.n_features = n_features
        self.n_time_bins = n_time_bins
        self.lambda_smooth = lambda_smooth
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.focal_gamma = focal_gamma
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose

    @staticmethod
    def _initialize_params(n_inputs: int, n_classes: int, key: Any, log_priors: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Initializes weights (W) and bias (b) using Xavier/Glorot initialization.

        Parameters
        ----------
        n_inputs : int
            Total number of input features (cols in X).
        n_classes : int
            Number of target classes (cols in Y).
        key : jax.random.PRNGKey
            JAX random key.

        Returns
        -------
        W : jnp.ndarray
            Initialized weight matrix (n_inputs, n_classes).
        b : jnp.ndarray
            Initialized bias vector (n_classes,).
        """

        k1, k2 = jax.random.split(key)
        scale = jnp.sqrt(2.0 / (n_inputs + n_classes))
        W = jax.random.normal(k1, (n_inputs, n_classes)) * scale

        # Initialize bias with the natural log of class frequencies
        b = log_priors

        return W, b

    @staticmethod
    @partial(jax.jit, static_argnums=(3, 4))
    def _loss_fn(
            params: Tuple[jnp.ndarray, jnp.ndarray],
            X: jnp.ndarray,
            Y_onehot: jnp.ndarray,
            n_feats: int,
            n_time: int,
            lam_smooth: float,
            lam_l1: float,
            lam_l2: float,
            class_weights: jnp.ndarray,
            focal_gamma: float
    ) -> jnp.ndarray:

        W, b = params
        logits = jnp.dot(X, W) + b

        # 1. Get the Softmax probabilities for all classes
        probs = jax.nn.softmax(logits, axis=-1)

        # 2. Extract the predicted probability for the TRUE class (pt)
        # Y_onehot acts as a mask, zeroing out the wrong class probabilities
        pt = jnp.sum(probs * Y_onehot, axis=1)

        # 3. Calculate standard Cross Entropy
        # Add a tiny epsilon (1e-8) to prevent log(0) exploding to NaN
        ce_loss = -jnp.log(pt + 1e-8)

        # 4. Calculate the Focal Modulating Factor: (1 - pt)^gamma
        focal_modulator = (1.0 - pt) ** focal_gamma

        # 5. Extract the specific alpha (class weight) for each sample's true class
        alpha_t = jnp.sum(Y_onehot * class_weights, axis=1)

        # 6. Combine: Alpha-balanced Focal Loss
        focal_loss = alpha_t * focal_modulator * ce_loss

        # Take the mean across the batch
        mean_focal_loss = jnp.mean(focal_loss)

        # L2 Penalty
        l2_loss = 0.5 * lam_l2 * jnp.sum(W ** 2)

        # Class-Specific Temporal Smoothness
        W_reshaped = W.reshape(n_feats, n_time, -1)
        d2w = jnp.diff(W_reshaped, n=2, axis=1)

        # Sum the curvature across features (axis=0) and time (axis=1), leaving shape (n_classes,)
        class_smooth_penalties = jnp.sum(d2w ** 2, axis=(0, 1))

        # Scale the per-class smoothing penalty by the inverse-frequency class weight:
        # rare classes receive a larger weight, so their filters are regularized more
        # strongly (stronger prior where data is thin and noise-to-signal is highest).
        smooth_loss = 0.5 * lam_smooth * jnp.sum(class_smooth_penalties * class_weights)

        # L1 Loss (Sparsity - currently 0.0)
        l1_loss = lam_l1 * jnp.sum(jnp.abs(W))

        return mean_focal_loss + l1_loss + l2_loss + smooth_loss

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SmoothMultinomialLogisticRegression":
        """
        Fits the model to the training data using JAX/Optax optimization.

        Parameters
        ----------
        X : np.ndarray
            Training vectors, shape (n_samples, n_features * n_time_bins).
        y : np.ndarray
            Target values, shape (n_samples,).

        Returns
        -------
        self : object
            Fitted estimator.
        """

        X, y = check_X_y(X, y, accept_sparse=False)

        self.lb_ = LabelBinarizer()
        Y_onehot = self.lb_.fit_transform(y)

        if Y_onehot.shape[1] == 1:
            Y_onehot = np.hstack([1 - Y_onehot, Y_onehot])

        self.classes_ = self.lb_.classes_
        n_samples, n_inputs = X.shape
        n_classes = len(self.classes_)

        expected_inputs = self.n_features * self.n_time_bins
        if n_inputs != expected_inputs:
            raise ValueError(
                f"Input X has {n_inputs} columns, but init parameters expect "
                f"n_features({self.n_features}) * n_time_bins({self.n_time_bins}) = "
                f"{expected_inputs} columns."
            )

        X_j = jnp.array(X)
        Y_j = jnp.array(Y_onehot)

        class_counts = jnp.sum(Y_j, axis=0)
        """ total_samples = Y_j.shape[0]
        c_weights = total_samples / (n_classes * (class_counts + 1e-8)) """
        # Soften the extreme class imbalance by using the square root of the counts
        sqrt_counts = jnp.sqrt(class_counts + 1e-8)
        c_weights = jnp.sum(sqrt_counts) / (n_classes * sqrt_counts)

        # 1. Calculate log-priors for the intercepts
        total_samples = Y_j.shape[0]
        priors = class_counts / total_samples
        log_priors = jnp.log(priors + 1e-8)

        # Pass log_priors into the initialization
        rng = jax.random.PRNGKey(self.random_state)
        params = self._initialize_params(n_inputs, n_classes, rng, log_priors)

        # 2. Setup Cosine Decay Learning Rate
        scheduler = optax.cosine_decay_schedule(
            init_value=self.learning_rate,
            decay_steps=self.max_iter
        )
        optimizer = optax.adam(scheduler)
        opt_state = optimizer.init(params)

        @partial(jax.jit, static_argnums=(4, 5))
        def step(params, opt_state, X_batch, Y_batch, n_feats, n_time, w_batch): # <-- ADD w_batch
            grads = jax.grad(self._loss_fn)(
                params, X_batch, Y_batch,
                n_feats, n_time,
                self.lambda_smooth,
                self.l1_reg, self.l2_reg,
                w_batch, self.focal_gamma
            )
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state

        if self.verbose:
            print(f"Starting JAX optimization for {self.max_iter} iterations...")

        for i in range(self.max_iter):
            old_params = params
            params, opt_state = step(params, opt_state, X_j, Y_j, self.n_features, self.n_time_bins, c_weights)

            if i > 0 and i % 100 == 0:
                diff = jnp.linalg.norm(params[0] - old_params[0])
                if diff < self.tol:
                    if self.verbose:
                        print(f"Converged at iteration {i} with diff {diff:.2e}")
                    break

                if self.verbose and i % 500 == 0:
                    current_loss = self._loss_fn(
                        params, X_j, Y_j,
                        self.n_features, self.n_time_bins,
                        self.lambda_smooth,
                        self.l1_reg, self.l2_reg,
                        c_weights, self.focal_gamma
                    )
                    print(f"Iter {i}: Loss = {current_loss:.4f}")

        self.coef_ = np.array(params[0].T)
        self.intercept_ = np.array(params[1])
        self.log_priors_ = np.array(log_priors)
        self.is_fitted_ = True

        return self

    def predict_proba(self, X: np.ndarray, balanced: bool = False) -> np.ndarray:
        """
        Compute probability estimates for each class.

        Parameters
        ----------
        X : np.ndarray
            Input data, shape (n_samples, n_features).
        balanced : bool, default=False
            If True, subtracts the stored training log-priors (`self.log_priors_`)
            from the logits before the softmax. This cleanly neutralizes the
            base-rate contribution that the intercept absorbed during fitting,
            so the returned probabilities reflect the learned feature-to-class
            evidence as if every category were equally likely a priori. Zeroing
            the full intercept would be unsafe because, after training, it also
            carries residual weight adjustments beyond the pure log-prior.

        Returns
        -------
        probs : np.ndarray
            Returns the probability of the sample for each class in the model,
            shape (n_samples, n_classes). The probabilities across all classes
            will sum to 1 for each sample.
        """

        check_is_fitted(self, ['coef_', 'intercept_', 'log_priors_'])
        X = check_array(X)

        if balanced:
            logits = np.dot(X, self.coef_.T) + self.intercept_ - self.log_priors_
        else:
            logits = np.dot(X, self.coef_.T) + self.intercept_

        max_logits = np.max(logits, axis=1, keepdims=True)
        e_x = np.exp(logits - max_logits)
        probs = e_x / e_x.sum(axis=1, keepdims=True)

        return probs

    def predict(self, X: np.ndarray, balanced: bool = False) -> np.ndarray:
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        X : np.ndarray
            Input data, shape (n_samples, n_features).
        balanced : bool, default=False
            If True, makes predictions based purely on the behavioral feature evidence
            by neutralizing the class priors (intercepts) before calculating the argmax.
            This prevents the model from defaulting to the majority class (e.g., Category 5)
            when uncertain, allowing for the isolation and visualization of specific
            kinematic relationships to rare vocal categories.

        Returns
        -------
        labels : np.ndarray
            Predicted class labels, shape (n_samples,).
        """

        probs = self.predict_proba(X, balanced=balanced)
        indices = np.argmax(probs, axis=1)

        return self.classes_[indices]
