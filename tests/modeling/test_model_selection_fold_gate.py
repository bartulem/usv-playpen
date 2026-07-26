"""
@author: bartulem
Unit tests for the fold-grain manifold selection gate helper in
``usv_playpen.modeling.model_selection``.

These drive the operative building block of the behaviour -> acoustic-manifold-
position feature-acceptance rule directly, with hand-built CV-fold dicts, so no
model fitting or on-disk artefacts are needed:

* ``_fold_paired_margin_bootstrap`` -- the per-fold paired score margin and fold
  bootstrap that both the screen (pooled von Mises, ``event_to_region=None``) and
  the forward-selection acceptance (macro von Mises, region-labelled) use: a
  predictor that tracks the truth must clear the gate (CI lower bound above 0,
  one-sided p ~ 0), a predictor independent of the truth must not (CI spanning 0,
  p not small), a degenerate single-fold input must return the non-significant
  sentinel rather than raise, the macro path must honour supplied region labels,
  and folds below ``min_fold_events`` must be dropped from the margin.
"""

import warnings

import numpy as np
import pytest

# The modeling import chain pulls optax -> a one-time JAX DeprecationWarning.
# Guard the top-level import so collection does not trip ``filterwarnings =
# ["error"]`` before any per-test marker can take effect (mirrors the guard in
# ``test_model_selection_tail``).
with warnings.catch_warnings():
    warnings.simplefilter('ignore', DeprecationWarning)
    from usv_playpen.modeling.model_selection import (
        _fold_paired_margin_bootstrap,
        _SELECTION_EFFECT_FLOOR,
        _SELECTION_N_BOOTSTRAP,
        _SELECTION_CI_LEVEL,
    )


def _folds_from_predictions(n_folds, n_per_fold, prediction_fn, seed):
    """
    Build ``n_folds`` CV folds of ``n_per_fold`` events each, contiguous in the
    global event index space, with ``prediction_fn(truth, rng)`` supplying each
    fold's predictions against a random torus truth.
    """
    rng = np.random.default_rng(seed)
    folds = {'test_indices': [], 'y_pred_xy': [], 'y_true': []}
    cursor = 0
    for _ in range(n_folds):
        idx = np.arange(cursor, cursor + n_per_fold)
        cursor += n_per_fold
        truth = rng.random((n_per_fold, 2))
        folds['test_indices'].append(idx)
        folds['y_true'].append(truth)
        folds['y_pred_xy'].append(prediction_fn(truth, rng))
    return folds


def _paired_null(actual, seed):
    """An independent-draw ``null`` block that shares `actual`'s truth/indices."""
    return {
        'test_indices': actual['test_indices'],
        'y_true': actual['y_true'],
        'y_pred_xy': [np.random.default_rng(seed + k).random(np.asarray(a).shape)
                      for k, a in enumerate(actual['y_pred_xy'])],
    }


def test_fold_bootstrap_accepts_a_real_predictor():
    """A prediction that tracks the truth clears the gate: CI lower bound > 0, p ~ 0."""
    # `actual` prediction = truth plus small wrapped noise (dependent); `null` = draw.
    actual = _folds_from_predictions(
        n_folds=8, n_per_fold=200,
        prediction_fn=lambda y, rng: np.mod(y + 0.05 * rng.standard_normal(y.shape), 1.0),
        seed=0,
    )
    null = _paired_null(actual, seed=100)
    result = _fold_paired_margin_bootstrap(
        actual, null,
        metric='torus', period=1.0, n_bootstrap=1000, ci_level=0.95, random_state=0,
    )
    assert result['n_folds'] == 8
    assert result['mean_margin'] > 0.0
    assert result['ci_low'] > 0.0
    assert result['p_value'] < 0.05


def test_fold_bootstrap_rejects_a_null_predictor():
    """A prediction independent of the truth fails: CI spans 0, p is not small."""
    actual = _folds_from_predictions(
        n_folds=8, n_per_fold=200,
        prediction_fn=lambda y, rng: rng.random(y.shape), seed=1)
    null = _paired_null(actual, seed=2)
    result = _fold_paired_margin_bootstrap(
        actual, null,
        metric='torus', period=1.0, n_bootstrap=1000, ci_level=0.95, random_state=0,
    )
    assert result['ci_low'] <= 0.0 <= result['ci_high']
    assert result['p_value'] > 0.05


def test_fold_bootstrap_degenerate_input_returns_non_significant():
    """Fewer than two usable folds -> NaN margin and p=1.0 (cannot pass), no raise."""
    actual = _folds_from_predictions(
        n_folds=1, n_per_fold=200,
        prediction_fn=lambda y, rng: y, seed=0)
    null = {'test_indices': actual['test_indices'], 'y_true': actual['y_true'],
            'y_pred_xy': actual['y_pred_xy']}
    result = _fold_paired_margin_bootstrap(
        actual, null,
        metric='torus', period=1.0, n_bootstrap=100, ci_level=0.95, random_state=0,
    )
    assert result['n_folds'] < 2
    assert np.isnan(result['mean_margin'])
    assert result['p_value'] == 1.0


def test_fold_bootstrap_macro_uses_region_labels():
    """
    The macro path (``event_to_region`` supplied) equal-weights regions and still
    accepts a truth-tracking predictor over an independent null.
    """
    n_folds, n_per_fold = 8, 200
    actual = _folds_from_predictions(
        n_folds=n_folds, n_per_fold=n_per_fold,
        prediction_fn=lambda y, rng: np.mod(y + 0.05 * rng.standard_normal(y.shape), 1.0),
        seed=3,
    )
    null = _paired_null(actual, seed=200)
    # Two acoustic regions, alternating over the global event index space so every
    # fold sees both regions well above the per-region floor.
    total_events = n_folds * n_per_fold
    event_to_region = np.tile([0.0, 1.0], total_events // 2)
    result = _fold_paired_margin_bootstrap(
        actual, null,
        metric='torus', period=1.0, n_bootstrap=1000, ci_level=0.95, random_state=0,
        event_to_region=event_to_region, min_region_events=20,
    )
    assert result['n_folds'] == n_folds
    assert result['ci_low'] > 0.0
    assert result['p_value'] < 0.05


def test_fold_bootstrap_skips_folds_below_min_events():
    """Folds contributing fewer than `min_fold_events` events drop out of the margin."""
    # Six folds of 200 events plus two tiny folds of 5 events; only the six large
    # folds should count once `min_fold_events` excludes the tiny ones.
    big = _folds_from_predictions(
        n_folds=6, n_per_fold=200,
        prediction_fn=lambda y, rng: np.mod(y + 0.05 * rng.standard_normal(y.shape), 1.0),
        seed=4,
    )
    small = _folds_from_predictions(
        n_folds=2, n_per_fold=5,
        prediction_fn=lambda y, rng: np.mod(y + 0.05 * rng.standard_normal(y.shape), 1.0),
        seed=5,
    )
    for key in ('test_indices', 'y_true', 'y_pred_xy'):
        big[key] = big[key] + small[key]
    null = _paired_null(big, seed=300)
    result = _fold_paired_margin_bootstrap(
        big, null,
        metric='torus', period=1.0, n_bootstrap=500, ci_level=0.95, random_state=0,
        min_fold_events=30,
    )
    assert result['n_folds'] == 6


def test_selection_gate_settings_are_present():
    """The three selection-gate settings load as sensible numbers."""
    assert 0.0 < _SELECTION_EFFECT_FLOOR < 1.0
    assert _SELECTION_N_BOOTSTRAP >= 100
    assert 0.0 < _SELECTION_CI_LEVEL < 1.0
