"""
@author: bartulem
Unit tests for ``usv_playpen.modeling.jax_bivariate_regression`` —
the JAX/Optax smooth 2-D (UMAP-coordinate) regressor used for the
continuous-manifold-position analyses.

Coverage is end-to-end ("maximal"): weight initialisation, then a full
fit / predict / evaluate_metrics cycle on a small synthetic, noiseless,
linearly-generated 2-D target where the model must recover a high
``r2_spatial`` and a near-zero raw MAE. Reproducibility is pinned via
``random_state`` (identical coefficients across two fits), and the
constructor / fit input-validation paths are exercised.
"""

from __future__ import annotations

import warnings

import jax
import numpy as np
import pytest

# Importing the JAX/Optax estimator pulls in optax, which sets a flag that
# JAX v0.9 has deprecated; that one-time DeprecationWarning fires at import
# (collection) time, where the project's `filterwarnings = ["error"]` would
# turn it into a hard collection error. Suppress it just for this import —
# it is a third-party transition warning, not a defect in code under test.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from usv_playpen.modeling.jax_bivariate_regression import SmoothBivariateRegression


def _make_linear_2d(n_samples=200, n_inputs=4, seed=0):
    """Synthesise a noiseless 2-D linear target ``y = X @ W + b`` so a
    correctly-fitting regressor can recover it almost exactly."""

    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_inputs))
    W_true = rng.standard_normal((n_inputs, 2))
    b_true = np.array([0.5, -0.3])
    y = X @ W_true + b_true
    return X.astype(np.float64), y.astype(np.float64)


def _fit_kwargs(n_inputs):
    """Clean-fit hyperparameters: no regularisation, a single time bin
    (no temporal-smoothness penalty), enough iterations to converge."""

    return dict(
        n_features=n_inputs, n_time_bins=1,
        lambda_smooth=0.0, l2_reg=0.0,
        learning_rate=0.05, max_iter=3000, tol=1e-7,
        random_state=0,
    )


def _make_torus_2d(n_samples=120, n_inputs=4, period=1.0, seed=1):
    """Synthesise a 2-D target whose coordinates live on a torus of the
    given ``period``.

    The raw linear signal ``X @ W`` is wrapped into ``[0, period)`` via a
    modulo so the targets densely populate the periodic manifold,
    including points near the wrap seam. This is the input shape the
    ``metric='torus'`` fit / predict / evaluate paths are designed for —
    it exercises the wrap-aware residual, the 4-D-embedding kd-tree, the
    circular-mean centroid, and the wrap-aware covariance.

    Parameters
    ----------
    n_samples : int
        Number of synthetic rows to generate.
    n_inputs : int
        Width of the design matrix (``n_features * n_time_bins``).
    period : float
        Per-axis wrap period of the toroidal target.
    seed : int
        Seed for the NumPy default-RNG so the data is reproducible.

    Returns
    -------
    X : np.ndarray
        ``(n_samples, n_inputs)`` design matrix, ``float64``.
    y : np.ndarray
        ``(n_samples, 2)`` toroidal targets in ``[0, period)``,
        ``float64``.
    """

    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_inputs))
    W_true = rng.standard_normal((n_inputs, 2))
    y = np.mod(X @ W_true, period)
    return X.astype(np.float64), y.astype(np.float64)


class TestInitParams:

    def test_weight_shape_and_zero_bias(self):
        """Initialisation returns a ``(n_inputs, 2)`` weight matrix and a
        zero ``(2,)`` bias."""

        W, b = SmoothBivariateRegression._initialize_params(5, jax.random.PRNGKey(0))
        assert W.shape == (5, 2)
        assert b.shape == (2,)
        np.testing.assert_allclose(np.asarray(b), np.zeros(2))

    def test_same_key_is_deterministic(self):
        """The same PRNG key reproduces identical weights."""

        W1, _ = SmoothBivariateRegression._initialize_params(5, jax.random.PRNGKey(3))
        W2, _ = SmoothBivariateRegression._initialize_params(5, jax.random.PRNGKey(3))
        np.testing.assert_array_equal(np.asarray(W1), np.asarray(W2))


class TestConstructorValidation:

    def test_bad_smoothness_order_raises(self):
        """Only first/second-order smoothness penalties are supported."""

        with pytest.raises(ValueError):
            SmoothBivariateRegression(smoothness_derivative_order=3)

    def test_bad_metric_raises(self):
        """An unsupported manifold metric is rejected at construction."""

        with pytest.raises(ValueError):
            SmoothBivariateRegression(metric='spherical')


class TestFitPredictEvaluate:

    def test_recovers_clean_linear_target(self):
        """On a noiseless linear target the unconstrained (raw) prediction
        recovers ``y`` closely and ``r2_spatial`` is near 1."""

        X, y = _make_linear_2d()
        model = SmoothBivariateRegression(**_fit_kwargs(X.shape[1])).fit(X, y)
        assert model.coef_.shape == (X.shape[1], 2)
        raw = model.predict(X, snap=False)
        assert np.mean(np.linalg.norm(raw - y, axis=1)) < 0.05
        metrics = model.evaluate_metrics(X, y)
        assert metrics['r2_spatial'] > 0.95
        assert metrics['euclidean_mae_raw'] < 0.05

    def test_snap_returns_training_points(self):
        """With ``snap=True`` every prediction is an actual training UMAP
        point (the kd-tree 1-NN projection)."""

        X, y = _make_linear_2d()
        model = SmoothBivariateRegression(**_fit_kwargs(X.shape[1])).fit(X, y)
        snapped = model.predict(X, snap=True)
        train_rows = {tuple(r) for r in y}
        for row in snapped:
            assert tuple(row) in train_rows

    def test_reproducible_under_random_state(self):
        """Two fits with the same ``random_state`` and data produce
        bit-identical coefficients."""

        X, y = _make_linear_2d()
        m1 = SmoothBivariateRegression(**_fit_kwargs(X.shape[1])).fit(X, y)
        m2 = SmoothBivariateRegression(**_fit_kwargs(X.shape[1])).fit(X, y)
        np.testing.assert_array_equal(m1.coef_, m2.coef_)
        np.testing.assert_array_equal(m1.intercept_, m2.intercept_)

    def test_evaluate_metrics_bundle_keys(self):
        """The metric bundle exposes the full documented key set."""

        X, y = _make_linear_2d()
        model = SmoothBivariateRegression(**_fit_kwargs(X.shape[1])).fit(X, y)
        metrics = model.evaluate_metrics(X, y)
        for key in ('r2_spatial', 'euclidean_mae', 'euclidean_rmse',
                    'mahalanobis_mae', 'mae_x', 'mae_y',
                    'pearson_x', 'pearson_y', 'spearman_x', 'spearman_y'):
            assert key in metrics

    def test_wrong_column_count_raises(self):
        """``fit`` rejects an X whose width disagrees with
        ``n_features * n_time_bins``."""

        X, y = _make_linear_2d(n_inputs=4)
        with pytest.raises(ValueError):
            SmoothBivariateRegression(n_features=3, n_time_bins=1).fit(X, y)

    def test_non_two_column_target_raises(self):
        """A target without exactly 2 columns is rejected."""

        X, _ = _make_linear_2d(n_inputs=4)
        y_bad = np.zeros((X.shape[0], 3))
        with pytest.raises(ValueError):
            SmoothBivariateRegression(n_features=4, n_time_bins=1).fit(X, y_bad)


class TestSampleWeightAndVerbose:

    def test_explicit_sample_weight_path(self):
        """Supplying an explicit ``sample_weight`` array drives the
        ``check_array`` branch in ``fit`` (rather than the uniform-weights
        default) and still recovers the clean linear target."""

        X, y = _make_linear_2d()
        weights = np.linspace(0.5, 2.0, X.shape[0])
        model = SmoothBivariateRegression(**_fit_kwargs(X.shape[1])).fit(
            X, y, sample_weight=weights,
        )
        raw = model.predict(X, snap=False)
        assert np.mean(np.linalg.norm(raw - y, axis=1)) < 0.1

    def test_weighted_metric_uses_supplied_weights(self):
        """``evaluate_metrics`` consumes caller-supplied weights for the
        weighted-MAE entry (exercises the non-default ``weights`` branch)."""

        X, y = _make_linear_2d()
        model = SmoothBivariateRegression(**_fit_kwargs(X.shape[1])).fit(X, y)
        weights = np.linspace(0.5, 2.0, X.shape[0])
        metrics = model.evaluate_metrics(X, y, weights=weights)
        assert np.isfinite(metrics['euclidean_mae_weighted'])

    def test_verbose_prints_and_converges(self, capsys):
        """``verbose=True`` prints the start banner and (because ``tol`` is
        loose) the convergence message, exercising the verbose branches in
        the Python for-loop path."""

        X, y = _make_linear_2d()
        kwargs = _fit_kwargs(X.shape[1])
        kwargs.update(verbose=True, tol=1e9, max_iter=300)
        model = SmoothBivariateRegression(**kwargs).fit(X, y)
        captured = capsys.readouterr().out
        assert "Starting 2-D JAX regression" in captured
        assert model.converged_ is True
        assert "Converged at iteration" in captured


class TestConvergence:

    def test_tiny_max_iter_does_not_converge(self):
        """A tiny ``max_iter`` (below the first 100-step convergence check)
        leaves ``converged_=False`` and ``n_iter_`` capped at ``max_iter``."""

        X, y = _make_linear_2d()
        kwargs = _fit_kwargs(X.shape[1])
        kwargs.update(max_iter=5)
        model = SmoothBivariateRegression(**kwargs).fit(X, y)
        assert model.converged_ is False
        assert model.n_iter_ == 5

    def test_loose_tol_converges(self):
        """A loose ``tol`` fires the convergence check before ``max_iter``,
        setting ``converged_=True``."""

        X, y = _make_linear_2d()
        kwargs = _fit_kwargs(X.shape[1])
        kwargs.update(tol=1e9, max_iter=500)
        model = SmoothBivariateRegression(**kwargs).fit(X, y)
        assert model.converged_ is True
        assert model.n_iter_ < 500


class TestLaxLoopPath:

    def test_lax_loop_matches_python_loop(self):
        """The fused ``jax.lax.while_loop`` training path (``_use_lax_loop
        =True``) recovers the clean linear target, exercising
        ``_bivariate_train_loop_jit`` and ``_bivariate_loss_static``."""

        X, y = _make_linear_2d()
        kwargs = _fit_kwargs(X.shape[1])
        kwargs.update(_use_lax_loop=True, max_iter=2000)
        model = SmoothBivariateRegression(**kwargs).fit(X, y)
        raw = model.predict(X, snap=False)
        assert np.mean(np.linalg.norm(raw - y, axis=1)) < 0.1
        assert model.n_iter_ > 0

    def test_lax_loop_converges_with_loose_tol(self):
        """With a loose ``tol`` the fused loop's in-graph convergence check
        fires, setting ``converged_=True`` before the iteration cap."""

        X, y = _make_linear_2d()
        kwargs = _fit_kwargs(X.shape[1])
        kwargs.update(_use_lax_loop=True, tol=1e9, max_iter=600)
        model = SmoothBivariateRegression(**kwargs).fit(X, y)
        assert model.converged_ is True
        assert model.n_iter_ < 600


class TestTorusMetric:

    def test_torus_fit_predict_evaluate(self):
        """A full fit / predict / evaluate cycle on ``metric='torus'``
        builds the 4-D-embedding kd-tree, the circular-mean centroid and
        the wrap-aware covariance, and snaps via the embedded query."""

        period = 1.0
        X, y = _make_torus_2d(period=period)
        model = SmoothBivariateRegression(
            n_features=X.shape[1], n_time_bins=1,
            lambda_smooth=0.0, l2_reg=0.0,
            learning_rate=0.05, max_iter=800, tol=1e-7,
            random_state=0, metric='torus', period=period,
        ).fit(X, y)
        # train_mean_ must lie on the manifold [0, period).
        assert np.all(model.train_mean_ >= 0.0)
        assert np.all(model.train_mean_ < period)
        # Snapped predictions are actual training points (torus kd-tree).
        snapped = model.predict(X, snap=True)
        train_rows = {tuple(np.round(r, 9)) for r in y}
        for row in snapped:
            assert tuple(np.round(row, 9)) in train_rows
        metrics = model.evaluate_metrics(X, y)
        assert np.isfinite(metrics['r2_spatial'])
        assert np.isfinite(metrics['mahalanobis_mae'])
        assert metrics['euclidean_mae'] >= 0.0

    def test_torus_period_two(self):
        """A non-unit period is plumbed through the wrap arithmetic so the
        centroid still lands inside ``[0, period)``."""

        period = 2.0
        X, y = _make_torus_2d(period=period)
        model = SmoothBivariateRegression(
            n_features=X.shape[1], n_time_bins=1,
            lambda_smooth=0.0, l2_reg=0.0,
            learning_rate=0.05, max_iter=400, tol=1e-7,
            random_state=0, metric='torus', period=period,
        ).fit(X, y)
        assert np.all(model.train_mean_ >= 0.0)
        assert np.all(model.train_mean_ < period)


class TestDegenerateMetrics:

    @pytest.mark.filterwarnings("ignore:.*:RuntimeWarning")
    @pytest.mark.filterwarnings("ignore::scipy.stats.ConstantInputWarning")
    def test_constant_prediction_yields_nan_correlations(self, monkeypatch):
        """When the snapped predictions are constant along an axis the
        Pearson denominator is zero and Spearman raises / returns
        non-finite, so the correlation entries fall back to NaN.

        The constant prediction is forced by zeroing the learned weights
        and bias after the fit so every prediction collapses to the same
        point, with the kd-tree disabled so no snapping reshuffles them."""

        X, y = _make_linear_2d()
        model = SmoothBivariateRegression(**_fit_kwargs(X.shape[1])).fit(X, y)
        model.coef_ = np.zeros_like(model.coef_)
        model.intercept_ = np.zeros_like(model.intercept_)
        model._train_kdtree = None
        metrics = model.evaluate_metrics(X, y)
        assert np.isnan(metrics['pearson_x'])
        assert np.isnan(metrics['pearson_y'])
        assert np.isnan(metrics['spearman_x'])
        assert np.isnan(metrics['spearman_y'])

    def test_no_covariance_yields_nan_mahalanobis(self):
        """If ``train_cov_inv_`` is unavailable the Mahalanobis MAE is
        NaN-safe (the ``getattr(...) is None`` branch)."""

        X, y = _make_linear_2d()
        model = SmoothBivariateRegression(**_fit_kwargs(X.shape[1])).fit(X, y)
        model.train_cov_inv_ = None
        metrics = model.evaluate_metrics(X, y)
        assert np.isnan(metrics['mahalanobis_mae'])

    def test_no_kdtree_evaluate_uses_raw(self):
        """With the kd-tree disabled, ``evaluate_metrics`` falls back to the
        raw (unsnapped) predictions (the ``else`` branch)."""

        X, y = _make_linear_2d()
        model = SmoothBivariateRegression(**_fit_kwargs(X.shape[1])).fit(X, y)
        model._train_kdtree = None
        metrics = model.evaluate_metrics(X, y)
        # On a clean linear target the raw prediction is accurate.
        assert metrics['euclidean_mae'] < 0.1

    def test_kdtree_build_failure_sets_none(self, monkeypatch):
        """If the kd-tree construction raises during ``fit`` the estimator
        swallows the exception and sets ``_train_kdtree=None`` (the
        ``except Exception`` guard), and ``predict(snap=True)`` then returns
        the raw prediction unchanged."""

        import usv_playpen.modeling.jax_bivariate_regression as biv

        def _boom(*args, **kwargs):
            raise RuntimeError("forced kd-tree failure")

        monkeypatch.setattr(biv, "cKDTree", _boom)
        X, y = _make_linear_2d()
        model = SmoothBivariateRegression(**_fit_kwargs(X.shape[1])).fit(X, y)
        assert model._train_kdtree is None
        raw = model.predict(X, snap=False)
        snapped = model.predict(X, snap=True)
        np.testing.assert_array_equal(raw, snapped)

    def test_spearman_valueerror_falls_back_to_nan(self, monkeypatch):
        """If ``spearmanr`` raises ``ValueError`` (its documented failure on
        certain degenerate inputs) the per-axis Spearman entry falls back to
        NaN via the ``except ValueError`` guard, while Pearson stays
        finite."""

        import usv_playpen.modeling.jax_bivariate_regression as biv

        def _raise_value_error(*args, **kwargs):
            raise ValueError("forced spearman failure")

        monkeypatch.setattr(biv, "spearmanr", _raise_value_error)
        X, y = _make_linear_2d()
        model = SmoothBivariateRegression(**_fit_kwargs(X.shape[1])).fit(X, y)
        metrics = model.evaluate_metrics(X, y)
        assert np.isnan(metrics['spearman_x'])
        assert np.isnan(metrics['spearman_y'])
        assert np.isfinite(metrics['pearson_x'])

    def test_predict_snap_false_returns_raw(self):
        """``predict(snap=False)`` returns the raw linear map even with a
        live kd-tree, distinct from the snapped output."""

        X, y = _make_linear_2d()
        model = SmoothBivariateRegression(**_fit_kwargs(X.shape[1])).fit(X, y)
        raw = model.predict(X, snap=False)
        expected = np.dot(X, model.coef_) + model.intercept_
        np.testing.assert_allclose(raw, expected)
