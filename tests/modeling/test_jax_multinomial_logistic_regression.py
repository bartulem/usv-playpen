"""
@author: bartulem
Unit tests for
``usv_playpen.modeling.jax_multinomial_logistic_regression`` — the
JAX/Optax smooth multinomial logistic classifier used for the vocal-
category analyses.

Coverage is end-to-end ("maximal"): weight/bias initialisation (the bias
is seeded from the empirical log-priors so the untrained softmax matches
the class marginal), then a full fit / predict / predict_proba cycle on a
small, well-separated 3-class synthetic problem where the model must reach
high accuracy. Reproducibility is pinned via ``random_state`` and the
constructor / fit validation paths are exercised.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# See the bivariate test for the rationale: importing the optax-backed
# estimator emits a one-time JAX DeprecationWarning at collection time
# that `filterwarnings = ["error"]` would otherwise promote to an error.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from usv_playpen.modeling.jax_multinomial_logistic_regression import (
        SmoothMultinomialLogisticRegression,
    )


def _make_separable_3class(n_per_class=80, n_inputs=4, seed=0):
    """Synthesise a well-separated 3-class problem so a correct classifier
    reaches near-perfect training accuracy."""

    rng = np.random.default_rng(seed)
    centres = np.array([
        [4.0, 0.0, 0.0, 0.0],
        [0.0, 4.0, 0.0, 0.0],
        [0.0, 0.0, 4.0, 0.0],
    ])
    X_chunks, y_chunks = [], []
    for cls, c in enumerate(centres):
        X_chunks.append(rng.normal(c, 0.4, size=(n_per_class, n_inputs)))
        y_chunks.append(np.full(n_per_class, cls))
    X = np.concatenate(X_chunks, axis=0).astype(np.float64)
    y = np.concatenate(y_chunks, axis=0)
    return X, y


def _fit_kwargs(n_inputs):
    """Clean-fit hyperparameters: no smoothness penalty, light L2, uniform
    class weights (the synthetic data is already balanced)."""

    return dict(
        n_features=n_inputs, n_time_bins=1,
        lambda_smooth=0.0, l2_reg=1e-3, focal_gamma=0.0,
        uniform_class_weights=True,
        learning_rate=0.05, max_iter=2000, tol=1e-7,
        random_state=0,
    )


def _make_imbalanced_3class(counts=(120, 60, 20), n_inputs=4, seed=2):
    """Synthesise an imbalanced, well-separated 3-class problem.

    The class counts are deliberately uneven so the default (non-uniform)
    softened inverse-frequency class-weighting branch in ``fit`` is
    exercised — the rare class receives a larger alpha / smoothness
    weight than the common class.

    Parameters
    ----------
    counts : tuple of int
        Per-class sample counts (length sets the number of classes).
    n_inputs : int
        Width of the design matrix.
    seed : int
        Seed for the NumPy default-RNG so the data is reproducible.

    Returns
    -------
    X : np.ndarray
        ``(sum(counts), n_inputs)`` design matrix, ``float64``.
    y : np.ndarray
        ``(sum(counts),)`` integer class labels.
    """

    rng = np.random.default_rng(seed)
    centres = np.array([
        [4.0, 0.0, 0.0, 0.0],
        [0.0, 4.0, 0.0, 0.0],
        [0.0, 0.0, 4.0, 0.0],
    ])
    X_chunks, y_chunks = [], []
    for cls, n in enumerate(counts):
        X_chunks.append(rng.normal(centres[cls], 0.4, size=(n, n_inputs)))
        y_chunks.append(np.full(n, cls))
    X = np.concatenate(X_chunks, axis=0).astype(np.float64)
    y = np.concatenate(y_chunks, axis=0)
    return X, y


def _make_binary(n_per_class=80, n_inputs=4, seed=3):
    """Synthesise a well-separated 2-class problem.

    A binary target makes ``LabelBinarizer`` emit a single column, which
    triggers the ``Y_onehot = np.hstack([1 - Y_onehot, Y_onehot])``
    two-column expansion branch in ``fit``.

    Parameters
    ----------
    n_per_class : int
        Samples per class.
    n_inputs : int
        Width of the design matrix.
    seed : int
        Seed for the NumPy default-RNG.

    Returns
    -------
    X : np.ndarray
        Design matrix, ``float64``.
    y : np.ndarray
        Integer ``{0, 1}`` labels.
    """

    rng = np.random.default_rng(seed)
    centres = np.array([
        [4.0, 0.0, 0.0, 0.0],
        [0.0, 4.0, 0.0, 0.0],
    ])
    X_chunks, y_chunks = [], []
    for cls, c in enumerate(centres):
        X_chunks.append(rng.normal(c, 0.4, size=(n_per_class, n_inputs)))
        y_chunks.append(np.full(n_per_class, cls))
    X = np.concatenate(X_chunks, axis=0).astype(np.float64)
    y = np.concatenate(y_chunks, axis=0)
    return X, y


class TestInitParams:

    def test_bias_equals_log_priors_and_weight_shape(self):
        """The bias is initialised verbatim from the supplied log-priors,
        and the weight matrix is ``(n_inputs, n_classes)``."""

        log_priors = jnp.log(jnp.array([0.2, 0.3, 0.5]))
        W, b = SmoothMultinomialLogisticRegression._initialize_params(
            n_inputs=4, n_classes=3, key=jax.random.PRNGKey(0), log_priors=log_priors,
        )
        assert W.shape == (4, 3)
        np.testing.assert_array_equal(np.asarray(b), np.asarray(log_priors))

    def test_log_prior_bias_reproduces_marginal_at_step_zero(self):
        """At step 0 (zero feature contribution) the softmax of the bias
        equals the empirical class prior."""

        priors = np.array([0.2, 0.3, 0.5])
        log_priors = jnp.log(jnp.array(priors))
        _, b = SmoothMultinomialLogisticRegression._initialize_params(
            4, 3, jax.random.PRNGKey(1), log_priors,
        )
        b = np.asarray(b)
        softmax = np.exp(b) / np.exp(b).sum()
        np.testing.assert_allclose(softmax, priors, atol=1e-6)


class TestConstructorValidation:

    def test_bad_smoothness_order_raises(self):
        """Only first/second-order smoothness penalties are supported."""

        with pytest.raises(ValueError):
            SmoothMultinomialLogisticRegression(smoothness_derivative_order=0)


class TestFitPredictProba:

    def test_high_accuracy_on_separable_data(self):
        """The classifier reaches near-perfect accuracy on a cleanly
        separable 3-class problem; coef/classes carry the expected
        shapes."""

        X, y = _make_separable_3class()
        model = SmoothMultinomialLogisticRegression(**_fit_kwargs(X.shape[1])).fit(X, y)
        assert model.coef_.shape == (3, X.shape[1])
        np.testing.assert_array_equal(model.classes_, np.array([0, 1, 2]))
        acc = float(np.mean(model.predict(X) == y))
        assert acc > 0.97

    def test_predict_proba_rows_sum_to_one(self):
        """Predicted probabilities form a valid distribution per sample."""

        X, y = _make_separable_3class()
        model = SmoothMultinomialLogisticRegression(**_fit_kwargs(X.shape[1])).fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (X.shape[0], 3)
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(X.shape[0]), atol=1e-6)
        assert np.all(proba >= 0.0)

    def test_predict_returns_class_labels(self):
        """``predict`` returns labels drawn from the trained class set."""

        X, y = _make_separable_3class()
        model = SmoothMultinomialLogisticRegression(**_fit_kwargs(X.shape[1])).fit(X, y)
        preds = model.predict(X)
        assert set(np.unique(preds)).issubset({0, 1, 2})

    def test_reproducible_under_random_state(self):
        """Two fits with the same ``random_state`` produce identical
        coefficients."""

        X, y = _make_separable_3class()
        m1 = SmoothMultinomialLogisticRegression(**_fit_kwargs(X.shape[1])).fit(X, y)
        m2 = SmoothMultinomialLogisticRegression(**_fit_kwargs(X.shape[1])).fit(X, y)
        np.testing.assert_array_equal(m1.coef_, m2.coef_)
        np.testing.assert_array_equal(m1.intercept_, m2.intercept_)

    def test_wrong_column_count_raises(self):
        """``fit`` rejects an X whose width disagrees with
        ``n_features * n_time_bins``."""

        X, y = _make_separable_3class(n_inputs=4)
        with pytest.raises(ValueError):
            SmoothMultinomialLogisticRegression(n_features=3, n_time_bins=1).fit(X, y)


class TestBalancedPrediction:

    def test_predict_proba_balanced_subtracts_log_priors(self):
        """``predict_proba(balanced=True)`` subtracts the stored log-priors
        from the logits, yielding a valid distribution that differs from the
        unbalanced probabilities on an imbalanced fit."""

        X, y = _make_imbalanced_3class()
        kwargs = _fit_kwargs(X.shape[1])
        kwargs.update(uniform_class_weights=False)
        model = SmoothMultinomialLogisticRegression(**kwargs).fit(X, y)
        proba_bal = model.predict_proba(X, balanced=True)
        proba_raw = model.predict_proba(X, balanced=False)
        np.testing.assert_allclose(
            proba_bal.sum(axis=1), np.ones(X.shape[0]), atol=1e-6,
        )
        # On an imbalanced fit the prior-neutralised probabilities differ
        # from the raw ones (the log-prior subtraction is non-trivial).
        assert not np.allclose(proba_bal, proba_raw)

    def test_predict_balanced_returns_valid_labels(self):
        """``predict(balanced=True)`` routes through the balanced softmax and
        returns labels from the trained class set."""

        X, y = _make_imbalanced_3class()
        kwargs = _fit_kwargs(X.shape[1])
        kwargs.update(uniform_class_weights=False)
        model = SmoothMultinomialLogisticRegression(**kwargs).fit(X, y)
        preds = model.predict(X, balanced=True)
        assert set(np.unique(preds)).issubset({0, 1, 2})


class TestClassWeightingBranches:

    def test_softened_inverse_frequency_weights(self):
        """With ``uniform_class_weights=False`` on imbalanced data the
        softened inverse-frequency class-weight branch in ``fit`` runs and
        the classifier still reaches high accuracy."""

        X, y = _make_imbalanced_3class()
        kwargs = _fit_kwargs(X.shape[1])
        kwargs.update(uniform_class_weights=False, focal_gamma=2.0)
        model = SmoothMultinomialLogisticRegression(**kwargs).fit(X, y)
        acc = float(np.mean(model.predict(X) == y))
        assert acc > 0.9

    def test_binary_target_expands_to_two_columns(self):
        """A binary target makes ``LabelBinarizer`` emit one column, which is
        expanded to two; the resulting model exposes two classes and
        classifies the separable problem accurately."""

        X, y = _make_binary()
        model = SmoothMultinomialLogisticRegression(**_fit_kwargs(X.shape[1])).fit(X, y)
        assert model.coef_.shape == (2, X.shape[1])
        np.testing.assert_array_equal(model.classes_, np.array([0, 1]))
        acc = float(np.mean(model.predict(X) == y))
        assert acc > 0.97


class TestVerboseAndConvergence:

    def test_verbose_prints_convergence(self, capsys):
        """``verbose=True`` with a loose ``tol`` prints the start banner and
        the convergence message — exercising the verbose convergence branch
        in the Python for-loop path."""

        X, y = _make_separable_3class()
        kwargs = _fit_kwargs(X.shape[1])
        kwargs.update(verbose=True, tol=1e9, max_iter=600, learning_rate=1e-4)
        model = SmoothMultinomialLogisticRegression(**kwargs).fit(X, y)
        captured = capsys.readouterr().out
        assert "Starting JAX optimization" in captured
        assert model.converged_ is True
        assert "Converged at iteration" in captured

    def test_verbose_prints_periodic_loss(self, capsys):
        """``verbose=True`` with a tight ``tol`` (so the loop does not
        converge before step 500) prints the periodic ``Iter ...: Loss``
        progress line — exercising the verbose loss-print branch."""

        X, y = _make_separable_3class()
        kwargs = _fit_kwargs(X.shape[1])
        kwargs.update(verbose=True, tol=1e-12, max_iter=600, learning_rate=1e-3)
        SmoothMultinomialLogisticRegression(**kwargs).fit(X, y)
        captured = capsys.readouterr().out
        assert "Starting JAX optimization" in captured
        assert "Loss =" in captured

    def test_tiny_max_iter_does_not_converge(self):
        """A tiny ``max_iter`` (below the first 100-step convergence check)
        leaves ``converged_=False`` and caps ``n_iter_``."""

        X, y = _make_separable_3class()
        kwargs = _fit_kwargs(X.shape[1])
        kwargs.update(max_iter=5)
        model = SmoothMultinomialLogisticRegression(**kwargs).fit(X, y)
        assert model.converged_ is False
        assert model.n_iter_ == 5


class TestLaxLoopPath:

    def test_lax_loop_classifies_separable_data(self):
        """The fused ``jax.lax.while_loop`` training path (``_use_lax_loop
        =True``) classifies the separable problem, exercising
        ``_multinomial_train_loop_jit`` and ``_multinomial_loss_static``."""

        X, y = _make_separable_3class()
        kwargs = _fit_kwargs(X.shape[1])
        kwargs.update(_use_lax_loop=True, max_iter=1500)
        model = SmoothMultinomialLogisticRegression(**kwargs).fit(X, y)
        acc = float(np.mean(model.predict(X) == y))
        assert acc > 0.9
        assert model.n_iter_ > 0

    def test_lax_loop_converges_with_loose_tol(self):
        """With a loose ``tol`` the fused loop's in-graph convergence check
        fires, setting ``converged_=True`` before the iteration cap."""

        X, y = _make_separable_3class()
        kwargs = _fit_kwargs(X.shape[1])
        kwargs.update(_use_lax_loop=True, tol=1e9, max_iter=600)
        model = SmoothMultinomialLogisticRegression(**kwargs).fit(X, y)
        assert model.converged_ is True
        assert model.n_iter_ < 600
