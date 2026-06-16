"""
@author: bartulem
Unit tests for ``usv_playpen.modeling.manifold_torus_regression`` — the
closed-form sin-cos embedding ridge used for ``metric='torus'`` continuous-
manifold-position analyses, plus the ``resolve_manifold_regressor_cls``
factory that wires it into the univariate runner and the model-selection
forward search.

Coverage is end-to-end: the factory dispatch (so euclidean runs keep the
unchanged coordinate model byte-for-byte and torus runs swap in the
embedding ridge), the closed-form solve validated bit-for-bit against an
independent normal-equation implementation (so the L2 + temporal-smoothness
generalised ridge is provably correct), the fit / predict / evaluate_metrics
contract and output shapes, the ``encode`` / ``decode`` round-trip, the
``metric='torus'`` guard, determinism, and — the scientific justification for
the class existing — the wound-recovery superiority over the coordinate model
on a target that winds around the torus.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy.linalg import block_diag

# Importing the estimator pulls in the JAX/Optax parent class, which sets a
# flag JAX v0.9 has deprecated; that one-time DeprecationWarning fires at
# import (collection) time, where the project's `filterwarnings = ["error"]`
# would turn it into a hard collection error. Suppress it just for this
# import — it is a third-party transition warning, not a defect under test.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from usv_playpen.modeling.jax_bivariate_regression import SmoothBivariateRegression
    from usv_playpen.modeling.manifold_torus_regression import (
        SmoothTorusManifoldRegression,
        resolve_manifold_regressor_cls,
    )


def _torus_kwargs(n_features, n_time_bins, l2_reg=0.01, lambda_smooth=1.0,
                  smoothness_derivative_order=2):
    """Reproduce the exact keyword set the wired call sites (the runner's
    outer fit and the model-selection candidate fits) pass to the resolved
    estimator class, including the iterative-optimiser arguments the torus
    estimator accepts-but-ignores."""

    return dict(
        n_features=n_features, n_time_bins=n_time_bins,
        lambda_smooth=lambda_smooth, l2_reg=l2_reg,
        smoothness_derivative_order=smoothness_derivative_order,
        huber_delta=1.0,
        learning_rate=0.005, max_iter=20000,
        tol=1e-4, random_state=0,
        _use_lax_loop=False,
        metric='torus', period=1.0,
    )


def _make_wrapped_torus_2d(n_samples=300, n_inputs=4, period=1.0, seed=3):
    """Synthesise a 2-D torus target as ``(X @ W) % period`` — coordinates
    densely populate the periodic manifold including the wrap seam. Used to
    exercise the closed-form solve and the fit / predict / evaluate paths."""

    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_inputs))
    W = rng.standard_normal((n_inputs, 2))
    y = (X @ W) % period
    return X.astype(np.float64), y.astype(np.float64)


def _make_wound_torus_2d(n_samples=600, n_inputs=4, period=1.0, seed=7):
    """Synthesise a target that *winds* around the torus: each coordinate is
    ``atan2(X @ w2, X @ w1)`` wrapped into ``[0, period)``. The 4-D embedding
    of this target is linear in ``X`` (up to the per-pair magnitude that
    ``atan2`` discards), so the embedding ridge recovers it almost exactly,
    while the wrapped ``atan2`` is strongly nonlinear in ``X`` so the
    coordinate model is structurally blind to it."""

    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_inputs))

    def _coord(g):
        w1, w2 = g.standard_normal(n_inputs), g.standard_normal(n_inputs)
        theta = np.arctan2(X @ w2, X @ w1)
        return (theta / (2.0 * np.pi) * period) % period

    yx = _coord(np.random.default_rng(seed + 1))
    yy = _coord(np.random.default_rng(seed + 2))
    return X.astype(np.float64), np.column_stack([yx, yy]).astype(np.float64)


def _manual_closed_form(X, Y, sample_weight, n_features, n_time_bins,
                        l2_reg, lambda_smooth, order, period):
    """Independent reference implementation of the weighted generalised-ridge
    normal-equation solve, used to validate the estimator's ``fit`` output
    bit-for-bit. Mirrors the documented objective without reusing any of the
    class's private helpers."""

    a = 2.0 * np.pi * Y / period
    emb = np.column_stack([
        np.cos(a[:, 0]), np.sin(a[:, 0]),
        np.cos(a[:, 1]), np.sin(a[:, 1]),
    ])
    w = sample_weight / (np.mean(sample_weight) + 1e-12)
    w_sum = float(np.sum(w))
    x_mean = (w[:, None] * X).sum(axis=0) / w_sum
    e_mean = (w[:, None] * emb).sum(axis=0) / w_sum
    x_c = X - x_mean
    e_c = emb - e_mean

    d_k = np.diff(np.eye(n_time_bins), n=order, axis=0)
    block = d_k.T @ d_k
    penalty_s = block if n_features == 1 else block_diag(*([block] * n_features))

    x_cw = x_c * w[:, None]
    a_mat = x_cw.T @ x_c + l2_reg * np.eye(X.shape[1]) + lambda_smooth * penalty_s
    coef = np.linalg.solve(a_mat, x_cw.T @ e_c)
    intercept = e_mean - x_mean @ coef
    return coef, intercept


def test_factory_dispatch():
    """euclidean -> unchanged coordinate model (byte-identical runs); torus
    -> the embedding ridge."""

    assert resolve_manifold_regressor_cls('euclidean') is SmoothBivariateRegression
    assert resolve_manifold_regressor_cls('torus') is SmoothTorusManifoldRegression
    # Any non-torus tag falls back to the coordinate model.
    assert resolve_manifold_regressor_cls('something_else') is SmoothBivariateRegression
    # The torus estimator is a strict subclass, so the inherited metric
    # bundle (`evaluate_metrics`) and persisted schema are shared.
    assert issubclass(SmoothTorusManifoldRegression, SmoothBivariateRegression)


def test_closed_form_matches_normal_equations():
    """The fitted ``coef_`` / ``intercept_`` match an independent
    normal-equation solve to machine precision — including the
    block-diagonal temporal-smoothness penalty (n_features > 1, n_time > 2)
    and a non-uniform sample weight."""

    n_features, n_time_bins = 2, 4
    n_inputs = n_features * n_time_bins
    rng = np.random.default_rng(11)
    X = rng.standard_normal((250, n_inputs))
    Y = (rng.random((250, 2))) % 1.0
    sw = rng.uniform(0.2, 3.0, size=250)

    kw = _torus_kwargs(n_features, n_time_bins, l2_reg=0.05, lambda_smooth=2.5,
                       smoothness_derivative_order=2)
    model = SmoothTorusManifoldRegression(**kw).fit(X, Y, sample_weight=sw)

    coef_ref, intercept_ref = _manual_closed_form(
        X, Y, sw, n_features, n_time_bins,
        l2_reg=0.05, lambda_smooth=2.5, order=2, period=1.0,
    )
    np.testing.assert_allclose(model.coef_, coef_ref, atol=1e-10, rtol=0)
    np.testing.assert_allclose(model.intercept_, intercept_ref, atol=1e-10, rtol=0)


def test_fit_contract_and_shapes():
    """The fit advertises a convex closed-form solve and the no-snap
    embedding geometry the inherited metric bundle relies on."""

    n_features, n_time_bins = 2, 5
    n_inputs = n_features * n_time_bins
    X, Y = _make_wrapped_torus_2d(n_samples=200, n_inputs=n_inputs)
    model = SmoothTorusManifoldRegression(**_torus_kwargs(n_features, n_time_bins))
    model.fit(X, Y)

    assert model.coef_.shape == (n_inputs, 4)
    assert model.intercept_.shape == (4,)
    assert model.n_iter_ == 1
    assert model.converged_ is True
    assert model.is_fitted_ is True
    assert model.fit_time_ >= 0.0
    # NO snap kd-tree: the decoded prediction is always a valid torus
    # coordinate, so snapping is deliberately disabled.
    assert model._train_kdtree is None
    assert model.train_mean_.shape == (2,)
    assert model.train_cov_inv_.shape == (2, 2)


def test_predict_contract():
    """``predict`` decodes the 4-D embedding to valid 2-D coordinates; with
    ``_train_kdtree=None`` the ``snap=True`` request is a no-op (raw decoded
    coordinates are returned)."""

    n_features, n_time_bins = 1, 6
    n_inputs = n_features * n_time_bins
    X, Y = _make_wrapped_torus_2d(n_samples=180, n_inputs=n_inputs)
    model = SmoothTorusManifoldRegression(**_torus_kwargs(n_features, n_time_bins)).fit(X, Y)

    pred_snap = model.predict(X, snap=True)
    pred_raw = model.predict(X, snap=False)
    assert pred_snap.shape == (X.shape[0], 2)
    assert np.all((pred_snap >= 0.0) & (pred_snap < 1.0))
    # snap is a no-op without a kd-tree.
    np.testing.assert_array_equal(pred_snap, pred_raw)


def test_encode_decode_round_trip():
    """``_decode(_encode(Y)) == Y`` to machine precision for coordinates in
    ``[0, period)`` (the embedding is a lossless re-parameterisation)."""

    model = SmoothTorusManifoldRegression(**_torus_kwargs(1, 1))
    rng = np.random.default_rng(5)
    Y = rng.random((100, 2)) % 1.0
    recovered = model._decode(model._encode(Y))
    np.testing.assert_allclose(recovered, Y, atol=1e-12, rtol=0)


def test_metric_guard_rejects_non_torus():
    """``fit`` refuses to run unless ``metric='torus'`` — the factory must be
    the only path that constructs this class, never a euclidean call site."""

    kw = _torus_kwargs(1, 1)
    kw['metric'] = 'euclidean'
    X, Y = _make_wrapped_torus_2d(n_samples=50, n_inputs=1)
    with pytest.raises(ValueError, match="requires metric='torus'"):
        SmoothTorusManifoldRegression(**kw).fit(X, Y)


def test_fit_is_deterministic():
    """The closed-form solve is exactly reproducible across two fits (no RNG,
    no iteration)."""

    n_features, n_time_bins = 2, 4
    n_inputs = n_features * n_time_bins
    X, Y = _make_wrapped_torus_2d(n_samples=160, n_inputs=n_inputs)
    a = SmoothTorusManifoldRegression(**_torus_kwargs(n_features, n_time_bins)).fit(X, Y)
    b = SmoothTorusManifoldRegression(**_torus_kwargs(n_features, n_time_bins)).fit(X, Y)
    np.testing.assert_array_equal(a.coef_, b.coef_)
    np.testing.assert_array_equal(a.intercept_, b.intercept_)


def test_evaluate_metrics_bundle_inherited():
    """The inherited ``evaluate_metrics`` produces the same wrap-aware metric
    schema as the coordinate model (so the persisted ledger is unchanged in
    shape)."""

    n_features, n_time_bins = 1, 4
    n_inputs = n_features * n_time_bins
    X, Y = _make_wrapped_torus_2d(n_samples=220, n_inputs=n_inputs)
    model = SmoothTorusManifoldRegression(**_torus_kwargs(n_features, n_time_bins)).fit(X, Y)
    metrics = model.evaluate_metrics(X, Y)
    for key in ('r2_spatial', 'euclidean_mae', 'mahalanobis_mae',
                'mae_x', 'mae_y', 'pearson_x', 'pearson_y',
                'spearman_x', 'spearman_y'):
        assert key in metrics
        assert np.isfinite(metrics[key])


def test_wound_recovery_beats_coordinate_model():
    """The scientific justification: on a target that winds around the torus
    the embedding ridge recovers a near-perfect held-out ``r2_spatial`` while
    the wound-blind coordinate model captures essentially none of the signal
    (it sits at the no-signal floor — at or below zero). Both are fit and
    scored under identical torus geometry, so the large gap is purely the
    estimator, not the metric."""

    X, Y = _make_wound_torus_2d(n_samples=600, n_inputs=4)
    tr, te = slice(0, 450), slice(450, 600)
    kw = _torus_kwargs(n_features=4, n_time_bins=1, l2_reg=1e-6, lambda_smooth=0.0)
    # The coordinate model needs enough iterations for a fair comparison.
    kw_coord = dict(kw, learning_rate=0.05, max_iter=4000, tol=1e-7)

    torus = SmoothTorusManifoldRegression(**kw).fit(X[tr], Y[tr])
    coord = SmoothBivariateRegression(**kw_coord).fit(X[tr], Y[tr])

    r2_torus = torus.evaluate_metrics(X[te], Y[te])['r2_spatial']
    r2_coord = coord.evaluate_metrics(X[te], Y[te])['r2_spatial']

    assert r2_torus > 0.85, f"embedding ridge should recover the wound target, got {r2_torus:.3f}"
    assert r2_coord < 0.2, f"coordinate model should be wound-blind (no signal), got {r2_coord:.3f}"
    assert r2_torus - r2_coord > 0.6
