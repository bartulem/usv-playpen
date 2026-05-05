"""
@author: bartulem
JAX-based multinomial logistic regression with temporal smoothing.

This module implements a custom scikit-learn compatible estimator that solves
multinomial logistic regression with two forms of regularisation:
1. Standard L2 (Ridge) on the weights.
2. Temporal smoothing (penalising the n-th derivative of the weights along
   the time axis) to force filters to vary smoothly over time lags.

L1 (lasso) regularisation is intentionally absent from this estimator: an
earlier revision supported it but we removed it because the temporal
smoothness penalty (together with L2) already controls filter shape and
magnitude, and mixing L1 into the loss produced poorly-conditioned
gradients that interacted badly with the focal-loss alpha reweighting.
Feature selection is handled at a higher level by the forward-selection
routine in `model_selection.py`, not by the loss function.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import time
from functools import partial
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.preprocessing import LabelBinarizer
from typing import Tuple, Any


def _multinomial_loss_static(
        params,
        X,
        Y_onehot,
        n_feats: int,
        n_time: int,
        lam_smooth,
        lam_l2,
        class_weights,
        focal_gamma,
        smoothness_derivative_order: int,
):
    """
    Module-level mirror of `SmoothMultinomialLogisticRegression._loss_fn`.

    Existing as a free function so the cache-friendly training-loop JIT
    (`_multinomial_train_loop_jit` below) can be defined at module scope.
    Avoids per-instance closure capture of regularisation scalars and
    design-matrix arrays — the JIT cache is then keyed on shape +
    static integers only.
    """

    W, b = params
    logits = jnp.dot(X, W) + b

    probs = jax.nn.softmax(logits, axis=-1)
    pt = jnp.sum(probs * Y_onehot, axis=1)
    ce_loss = -jnp.log(pt + 1e-8)
    focal_modulator = (1.0 - pt) ** focal_gamma
    alpha_t = jnp.sum(Y_onehot * class_weights, axis=1)
    focal_loss = alpha_t * focal_modulator * ce_loss
    mean_focal_loss = jnp.mean(focal_loss)

    l2_loss = 0.5 * lam_l2 * jnp.sum(W ** 2)

    W_reshaped = W.reshape(n_feats, n_time, -1)
    dW = jnp.diff(W_reshaped, n=smoothness_derivative_order, axis=1)
    class_smooth_penalties = jnp.sum(dW ** 2, axis=(0, 1))
    smooth_loss = 0.5 * lam_smooth * jnp.sum(class_smooth_penalties * class_weights)

    return mean_focal_loss + l2_loss + smooth_loss


@partial(
    jax.jit,
    static_argnames=('n_feats', 'n_time', 'smoothness_derivative_order', 'max_iter'),
)
def _multinomial_train_loop_jit(
        params_init,
        opt_state_init,
        X,
        Y_onehot,
        class_weights,
        lambda_smooth,
        l2_reg,
        focal_gamma,
        learning_rate,
        tol,
        max_iter: int,
        n_feats: int,
        n_time: int,
        smoothness_derivative_order: int,
):
    """
    Full multinomial descent fused into a single
    `jax.lax.while_loop`-driven JIT call. Module scope + explicit
    arguments make the cache shape-keyed, so a tuner that constructs
    many short-lived estimators with matching input shapes hits the
    same compiled graph for all of them.

    `max_iter` is a static argument because the cosine-decay schedule
    needs it as a Python integer at trace time. In practice the only
    distinct values seen at runtime are the outer fit's `max_iter` and
    the inner-CV's `inner_max_iter`, so the cache holds at most two
    compiled variants per shape.
    """

    scheduler = optax.cosine_decay_schedule(
        init_value=learning_rate, decay_steps=max_iter,
    )
    optimizer = optax.adam(scheduler)
    check_interval = 100

    def step(params, opt_state):
        grads = jax.grad(_multinomial_loss_static)(
            params, X, Y_onehot,
            n_feats, n_time,
            lambda_smooth, l2_reg,
            class_weights, focal_gamma,
            smoothness_derivative_order,
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


class SmoothMultinomialLogisticRegression(BaseEstimator, ClassifierMixin):
    """
    Multinomial logistic regression with temporal smoothing (JAX implementation).

    This estimator learns a weight matrix W of shape (n_features * n_time_bins, n_classes).
    It minimises an alpha-balanced focal loss augmented by:
    1. L2 regularisation (ridge) which penalises large weights.
    2. Temporal smoothing penalty which penalises the discrete n-th derivative
       of the weights along the time axis (order 1 or 2, see
       `smoothness_derivative_order`), forcing the learned filters to be
       smooth. The per-class smoothing penalty is additionally scaled by the
       inverse-frequency class weight so rare-class filters are regularised
       more strongly (stronger prior where data is thin).

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
        Regularisation strength for the temporal smoothing penalty (see
        `smoothness_derivative_order` below for which derivative is
        penalised). Higher values force smoother (stiffer) curves.
    l2_reg : float, default=0.1
        Regularisation strength for standard L2 (Ridge) penalty.
    smoothness_derivative_order : int, default=2
        Order of the finite-difference derivative used to build the
        temporal-smoothness penalty on every class's filter along the
        time axis. Both choices correspond to improper Gaussian-process
        priors and neither fixes a basis, but they push the filters
        toward different shape families:
          - `1` — squared first-difference penalty
            `sum((w_{t+1} - w_t)^2)`, zero cost when the filter is flat.
            Biases the filter toward piecewise-constant / step-like
            shapes. Used in the Paninski/Pillow GLM-HMM literature
            (e.g., Calhoun et al. *Nature Neurosci* 2019 on fly song
            modes).
          - `2` — squared second-difference penalty
            `sum((w_{t+1} - 2 w_t + w_{t-1})^2)`, zero cost when the
            filter is any straight line. Biases the filter toward
            smooth curves; classical GAM / smoothing-spline choice.
            Recommended when the scientific goal is to learn unbiased
            filter *shape* without a piecewise-constant prior.
    focal_gamma : float, default=2.0
        Focusing parameter of the focal loss: the `(1 - p_t) ** focal_gamma`
        modulator down-weights easy examples so gradient flow concentrates on
        hard / misclassified samples. Set to 0.0 to recover plain alpha-balanced
        cross-entropy.
    uniform_class_weights : bool, default=False
        If True, replaces the softened inverse-frequency class weights used as the
        focal-loss alpha and as the class-specific smoothness scaler with uniform
        weights equal to 1 / n_classes. Intended to be combined with an externally
        pre-balanced training fold (e.g., via the runner's `balance_train_bool`
        option), where sample-level balancing has already equalized class exposure
        and any additional inverse-frequency reweighting would double-correct.
    learning_rate : float, default=1e-3
        Step size for the Adam optimizer.
    max_iter : int, default=5000
        Maximum number of optimization steps (epochs).
    tol : float, default=1e-4
        Convergence tolerance. Training stops if the L2 norm of the parameter
        update over the previous 100 optimizer steps falls below this value.
    random_state : int, default=0
        Seed for JAX random number generation initialization.
    verbose : bool, default=False
        If True, prints loss progress during training.

    Attributes
    ----------
    coef_ : np.ndarray, shape (n_classes, n_features * n_time_bins)
        Learned weight matrix (sklearn convention — transposed from the
        internal `(n_inputs, n_classes)` JAX layout).
    intercept_ : np.ndarray, shape (n_classes,)
        Learned per-class biases. Initialised from the log class prior.
    log_priors_ : np.ndarray, shape (n_classes,)
        `log(count_c / N + eps)` captured at fit time; used by
        `predict_proba(balanced=True)` to neutralise the base-rate
        contribution absorbed into the intercept during fitting.
    classes_ : np.ndarray
        Ordered class labels in the layout used by `coef_` columns.
    lb_ : sklearn.preprocessing.LabelBinarizer
        The binarizer fitted to the training targets.
    n_iter_ : int
        Number of optimizer steps actually taken (1-indexed). Equals
        `max_iter` when the tolerance check never fired.
    converged_ : bool
        True if the tolerance check fired before `max_iter`; False otherwise.
        Flags the main silent-failure mode of the estimator — a
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
            lambda_smooth: float = 1,
            l2_reg: float = 0.1,
            smoothness_derivative_order: int = 2,
            focal_gamma: float = 2.0,
            uniform_class_weights: bool = False,
            learning_rate: float = 1e-3,
            max_iter: int = 5000,
            tol: float = 1e-4,
            random_state: int = 0,
            verbose: bool = False,
            _use_lax_loop: bool = False
    ):
        if smoothness_derivative_order not in (1, 2):
            raise ValueError(
                f"smoothness_derivative_order must be 1 or 2; got {smoothness_derivative_order}."
            )
        self.n_features = n_features
        self.n_time_bins = n_time_bins
        self.lambda_smooth = lambda_smooth
        self.l2_reg = l2_reg
        self.smoothness_derivative_order = int(smoothness_derivative_order)
        self.focal_gamma = focal_gamma
        self.uniform_class_weights = uniform_class_weights
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose
        # Opt-in fused training loop. When True, the full descent is
        # wrapped in a single `jax.lax.while_loop` inside a `@jax.jit`
        # closure — eliminates per-iteration Python dispatch at the
        # cost of a large one-time compile. Because the jitted function
        # closes over `X_j`, `Y_j` and the regularisation scalars, the
        # cache is keyed per-instance rather than per-shape, so any
        # caller that constructs many short-lived estimators (notably
        # the joint inner-CV tuner, which builds ~175 per outer fold)
        # pays the full compile cost per instance and can stall for
        # hours on wide-feature graphs (e.g. bin=1 with 600 time bins
        # × 6 classes). Default is False: the standard Python for-loop
        # has no compilation issues at any scale and is the correct
        # choice for the tuning path. Set to True only on the outer-fit
        # path when `max_iter` is very large on shapes that compile
        # quickly; the speedup is 1.3-1.8x on GPU in that regime.
        self._use_lax_loop = bool(_use_lax_loop)

    @staticmethod
    def _initialize_params(n_inputs: int, n_classes: int, key: Any, log_priors: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Initializes weights (W) and bias (b) for the multinomial fit.

        Weights are drawn from a Xavier/Glorot normal distribution so early
        Adam updates stay numerically stable, while the bias is initialised
        from the empirical log class prior so the un-trained model already
        predicts the training-set marginal distribution (any subsequent
        gradient pressure on the intercept is then purely a refinement on
        top of the prior). This matches the `balanced` softmax path in
        `predict_proba`, which subtracts `log_priors_` to recover the
        pure feature-to-class evidence.

        Parameters
        ----------
        n_inputs : int
            Total number of input features (columns of `X`, equal to
            `n_features * n_time_bins`).
        n_classes : int
            Number of target classes (columns of the learned `W`).
        key : jax.random.PRNGKey
            JAX random key for deterministic weight initialisation.
        log_priors : jnp.ndarray
            Per-class `log(count_c / N + eps)` computed from the training
            one-hot target matrix. Used verbatim as the bias initialiser
            so that, at step 0, the model's softmax output is exactly the
            empirical class prior.

        Returns
        -------
        W : jnp.ndarray
            Initialised weight matrix of shape `(n_inputs, n_classes)`.
        b : jnp.ndarray
            Bias vector of shape `(n_classes,)`, initialised from
            `log_priors`.
        """

        scale = jnp.sqrt(2.0 / (n_inputs + n_classes))
        W = jax.random.normal(key, (n_inputs, n_classes)) * scale

        # Initialize bias with the natural log of class frequencies
        b = log_priors

        return W, b

    @staticmethod
    @partial(jax.jit, static_argnums=(3, 4, 9))
    def _loss_fn(
            params: Tuple[jnp.ndarray, jnp.ndarray],
            X: jnp.ndarray,
            Y_onehot: jnp.ndarray,
            n_feats: int,
            n_time: int,
            lam_smooth: float,
            lam_l2: float,
            class_weights: jnp.ndarray,
            focal_gamma: float,
            smoothness_derivative_order: int,
    ) -> jnp.ndarray:
        """
        Computes the total optimisation loss: alpha-balanced focal loss +
        L2 + class-specific temporal smoothness.

        The focal loss down-weights easy (high-confidence-correct) samples
        via the `(1 - p_t) ** focal_gamma` modulator so gradient flow
        concentrates on hard / misclassified examples; the per-class
        `alpha_t` factor further up-weights rare-class samples so the
        gradient is not swamped by majority-class residuals. The temporal
        smoothness term penalises the discrete n-th derivative of each
        class's filter along the time axis (order selectable via
        `smoothness_derivative_order`) and is additionally scaled by the
        per-class weight so rare classes — whose filters have the
        highest noise-to-signal ratio — get a stronger prior.

        Parameters
        ----------
        params : tuple
            `(W, b)` tuple of current model parameters, with
            `W` shape `(n_features * n_time, n_classes)` and
            `b` shape `(n_classes,)`.
        X : jnp.ndarray
            Input behavioural-history matrix of shape
            `(n_samples, n_features * n_time)`.
        Y_onehot : jnp.ndarray
            One-hot-encoded class labels of shape
            `(n_samples, n_classes)`; acts as the true-class mask that
            selects `p_t` from the full softmax probability row.
        n_feats : int
            Number of distinct physical behavioural features. Static
            JIT argument — used to reshape `W` back into
            `(n_features, n_time, n_classes)` so the smoothness penalty
            runs along the time axis only.
        n_time : int
            Number of time bins per feature. Static JIT argument, same
            reshape purpose as `n_feats`.
        lam_smooth : float
            Temporal-smoothness penalty strength (λ_smooth).
        lam_l2 : float
            L2 (Ridge) penalty strength.
        class_weights : jnp.ndarray
            Per-class weighting vector of shape `(n_classes,)`. Used both
            as the focal-loss alpha (`alpha_t = Y_onehot @ class_weights`)
            and as the per-class scaler on the smoothness penalty. In
            default mode these are softened inverse-frequency weights; in
            `uniform_class_weights=True` mode they are `1 / n_classes`
            because the training batch was externally pre-balanced.
        focal_gamma : float
            Focusing parameter of the focal loss. `0.0` recovers plain
            alpha-balanced cross-entropy.
        smoothness_derivative_order : int
            Order of the finite-difference derivative (1 or 2) used to
            build the temporal-smoothness penalty. Static JIT argument —
            see the class docstring for the interpretability tradeoff.

        Returns
        -------
        loss : jnp.ndarray
            Scalar total loss: mean focal cross-entropy + 0.5 *
            `lam_l2` * ||W||^2 + 0.5 * `lam_smooth` * sum_c
            class_weight_c * ||D_order W_c||^2.
        """

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

        # Class-Specific Temporal Smoothness — penalise the discrete
        # n-th derivative of each class's filter along the time axis.
        # `smoothness_derivative_order=1` biases toward piecewise-
        # constant filters; `=2` (default) biases toward smooth curves.
        W_reshaped = W.reshape(n_feats, n_time, -1)
        dW = jnp.diff(W_reshaped, n=smoothness_derivative_order, axis=1)

        # Sum the penalty across features (axis=0) and time (axis=1), leaving shape (n_classes,)
        class_smooth_penalties = jnp.sum(dW ** 2, axis=(0, 1))

        # Scale the per-class smoothing penalty by the inverse-frequency class weight:
        # rare classes receive a larger weight, so their filters are regularised more
        # strongly (stronger prior where data is thin and noise-to-signal is highest).
        smooth_loss = 0.5 * lam_smooth * jnp.sum(class_smooth_penalties * class_weights)

        return mean_focal_loss + l2_loss + smooth_loss

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
        if self.uniform_class_weights:
            # Training fold was already sample-level balanced upstream (e.g., the
            # runner's `balance_train_bool` path). Replace the softened
            # inverse-frequency weights with uniform weights so the focal-alpha
            # does not double-correct an already balanced batch.
            c_weights = jnp.ones(n_classes, dtype=Y_j.dtype)
        else:
            # Soften the extreme class imbalance by using the square root of the counts
            sqrt_counts = jnp.sqrt(class_counts + 1e-8)
            c_weights = jnp.sum(sqrt_counts) / (n_classes * sqrt_counts)

        # Normalise to unit mean in both branches so `lambda_smooth` has the
        # same effective meaning regardless of whether the uniform or the
        # softened-inverse-frequency path is active. `_loss_fn` uses
        # `class_weights` twice — once as the focal-loss alpha, once as the
        # per-class scaler on the smoothness penalty — and both uses are
        # scale-equivariant, so the mean-1 normalisation preserves the
        # intended *shape* of the reweighting while pinning the scale.
        c_weights = c_weights / (jnp.mean(c_weights) + 1e-12)

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
        def step(params, opt_state, X_batch, Y_batch, n_feats, n_time, w_batch):
            grads = jax.grad(self._loss_fn)(
                params, X_batch, Y_batch,
                n_feats, n_time,
                self.lambda_smooth, self.l2_reg,
                w_batch, self.focal_gamma,
                self.smoothness_derivative_order,
            )
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state

        if self.verbose:
            print(f"Starting JAX optimization for {self.max_iter} iterations...")

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
                params, opt_state = step(params, opt_state, X_j, Y_j, self.n_features, self.n_time_bins, c_weights)
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

                    if self.verbose and i % 500 == 0:
                        current_loss = self._loss_fn(
                            params, X_j, Y_j,
                            self.n_features, self.n_time_bins,
                            self.lambda_smooth, self.l2_reg,
                            c_weights, self.focal_gamma,
                            self.smoothness_derivative_order,
                        )
                        print(f"Iter {i}: Loss = {current_loss:.4f}")

                    last_check_params = params
        else:
            # Fused path: full descent runs inside a single JIT-compiled
            # `jax.lax.while_loop` defined at module scope. The cache is
            # keyed on shape + the static integers (`n_feats`, `n_time`,
            # `smoothness_derivative_order`, `max_iter`); every per-fit
            # scalar (regularisation strengths, learning rate,
            # tolerance, focal-gamma, class weights) is a traced
            # argument. Tuners that build many short-lived estimators
            # with matching input shapes hit the cached compile after
            # the first construction.
            params, completed_iter_j, converged_j = _multinomial_train_loop_jit(
                params,
                opt_state,
                X_j, Y_j, c_weights,
                jnp.asarray(self.lambda_smooth, dtype=jnp.float32),
                jnp.asarray(self.l2_reg, dtype=jnp.float32),
                jnp.asarray(self.focal_gamma, dtype=jnp.float32),
                jnp.asarray(self.learning_rate, dtype=jnp.float32),
                jnp.asarray(self.tol, dtype=jnp.float32),
                int(self.max_iter),
                int(self.n_features),
                int(self.n_time_bins),
                int(self.smoothness_derivative_order),
            )
            completed_iter = int(completed_iter_j)
            converged = bool(converged_j)

        self.coef_ = np.array(params[0].T)
        self.intercept_ = np.array(params[1])
        self.log_priors_ = np.array(log_priors)
        # Expose fit-time diagnostics so callers can persist per-fold convergence
        # evidence alongside the learned weights — the most common silent-failure
        # mode for this estimator is hitting `max_iter` without converging.
        self.n_iter_ = int(completed_iter)
        self.converged_ = bool(converged)
        self.fit_time_ = float(time.perf_counter() - fit_start)
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
