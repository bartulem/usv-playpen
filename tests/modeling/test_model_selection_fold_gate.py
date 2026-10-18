"""
@author: bartulem
Unit tests for the fold-grain manifold selection gate helper in
``usv_playpen.modeling.model_selection``.

These drive the operative building block of the behaviour -> acoustic-manifold-
position feature-acceptance rule directly, with hand-built CV-fold dicts, so no
model fitting or on-disk artefacts are needed.

Coverage:

* ``_fold_paired_margin_bootstrap`` -- the per-fold paired score margin and fold
  bootstrap that both the screen (pooled von Mises, ``event_to_region=None``) and
  the forward-selection acceptance (macro von Mises, region-labelled) use: a
  predictor that tracks the truth must clear the gate (CI lower bound above 0,
  one-sided p ~ 0), a predictor independent of the truth must not (CI spanning 0,
  p not small), a degenerate single-fold input must return the non-significant
  sentinel rather than raise, the macro path must honour supplied region labels,
  and folds below ``min_fold_events`` must be dropped from the margin.
* ``_frozen_selection_kappa`` -- the frozen concentration lets a fold predicted
  worse than the marginal score below the flat-torus floor.
* the three selection-gate settings and the gate's no-signal acceptance rate.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest

# The modeling import chain pulls optax -> a one-time JAX DeprecationWarning.
# Guard the top-level import so collection does not trip ``filterwarnings =
# ["error"]`` before any per-test marker can take effect (mirrors the guard in
# ``test_model_selection_tail``).
with warnings.catch_warnings():
    warnings.simplefilter('ignore', DeprecationWarning)
    from usv_playpen.modeling.manifold_metric import macro_von_mises_logscore
    from usv_playpen.modeling.model_selection import (
        _fold_paired_margin_bootstrap,
        _frozen_selection_kappa,
    )

# The selection-gate knobs as shipped (read from the settings file the run reads).
with (Path(__file__).resolve().parents[2] / 'src' / 'usv_playpen' / '_parameter_settings'
      / 'modeling_settings.json').open() as _settings_file:
    _SHIPPED_MODEL_PARAMS = json.load(_settings_file)['model_params']
_SELECTION_EFFECT_FLOOR = _SHIPPED_MODEL_PARAMS['selection_effect_floor']
_SELECTION_N_BOOTSTRAP = _SHIPPED_MODEL_PARAMS['selection_n_bootstrap']
_SELECTION_CI_LEVEL = _SHIPPED_MODEL_PARAMS['selection_ci_level']


def _folds_from_predictions(n_folds, n_per_fold, prediction_fn, seed):
    """
    Description
    -----------
    Build ``n_folds`` CV folds of ``n_per_fold`` events each, contiguous in the
    global event index space, with ``prediction_fn(truth, rng)`` supplying each
    fold's predictions against a random torus truth drawn uniformly on [0, 1)^2.

    Parameters
    ----------
    n_folds (int)
        Number of folds to build.
    n_per_fold (int)
        Number of events in every fold.
    prediction_fn (Callable[[np.ndarray, np.random.Generator], np.ndarray])
        Maps a fold's (n_per_fold, 2) truth and the shared generator to that
        fold's (n_per_fold, 2) predictions; invoked positionally.
    seed (int)
        Seed of the generator that draws the truths and feeds ``prediction_fn``.

    Returns
    -------
    folds (dict)
        ``{'test_indices': [...], 'y_pred_xy': [...], 'y_true': [...]}``, one
        list entry per fold, in the CV-fold dict shape the gate consumes.
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
    """
    Description
    -----------
    Build an independent-draw ``null`` fold block that shares ``actual``'s truth
    and test indices, so the gate sees a paired comparison against predictions
    that carry no information about the truth.

    Parameters
    ----------
    actual (dict)
        Fold dict from ``_folds_from_predictions``.
    seed (int)
        Base seed; fold ``k`` draws its uniform predictions with ``seed + k``.

    Returns
    -------
    null (dict)
        Fold dict with ``actual``'s ``test_indices`` / ``y_true`` and uniform
        random ``y_pred_xy`` of matching shapes.
    """
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
        metric='torus', period=1.0, n_bootstrap=1000, ci_level=0.95, random_state=0, min_region_events=1, min_fold_events=30, sign_flip_exact_max_folds=22, sign_flip_monte_carlo_draws=2**20, dcor_n_sub=2500, dcor_n_rep=1
    )
    assert result['n_folds'] == 8
    assert result['mean_margin'] > 0.0
    assert result['ci_low'] > 0.0
    assert result['p_value'] < 0.05


def test_fold_bootstrap_honors_frozen_kappa():
    """A supplied `kappa` is reused for every fold's torus score instead of the
    per-fold self-refit, so the mean margin equals the fold-mean of the paired
    pooled von Mises log-scores computed at that frozen kappa, and a different
    kappa moves the margin (it is genuinely used)."""
    actual = _folds_from_predictions(
        n_folds=8, n_per_fold=200,
        prediction_fn=lambda y, rng: np.mod(y + 0.05 * rng.standard_normal(y.shape), 1.0),
        seed=0,
    )
    null = _paired_null(actual, seed=100)
    k = 2.5
    res = _fold_paired_margin_bootstrap(
        actual, null, metric='torus', period=1.0,
        n_bootstrap=200, ci_level=0.95, random_state=0, kappa=k, min_region_events=1, min_fold_events=30, sign_flip_exact_max_folds=22, sign_flip_monte_carlo_draws=2**20, dcor_n_sub=2500, dcor_n_rep=1
    )
    # Manual per-fold paired margin at the SAME frozen kappa (pooled: no regions
    # passed, so `_fold_scores` scores `macro_von_mises_logscore(..., None)`).
    expected = float(np.mean([
        macro_von_mises_logscore(ap, at, None, metric='torus', period=1.0, kappa=k, min_region_events=1)
        - macro_von_mises_logscore(npred, at, None, metric='torus', period=1.0, kappa=k, min_region_events=1)
        for ap, npred, at in zip(actual['y_pred_xy'], null['y_pred_xy'], actual['y_true'], strict=True)
    ]))
    assert res['mean_margin'] == pytest.approx(expected)
    # A different frozen kappa moves the margin.
    res2 = _fold_paired_margin_bootstrap(
        actual, null, metric='torus', period=1.0,
        n_bootstrap=200, ci_level=0.95, random_state=0, kappa=0.5, min_region_events=1, min_fold_events=30, sign_flip_exact_max_folds=22, sign_flip_monte_carlo_draws=2**20, dcor_n_sub=2500, dcor_n_rep=1
    )
    assert res2['mean_margin'] != pytest.approx(res['mean_margin'])


def test_fold_bootstrap_rejects_a_null_predictor():
    """A prediction independent of the truth fails: CI spans 0, p is not small."""
    actual = _folds_from_predictions(
        n_folds=8, n_per_fold=200,
        prediction_fn=lambda y, rng: rng.random(y.shape), seed=1)
    null = _paired_null(actual, seed=2)
    result = _fold_paired_margin_bootstrap(
        actual, null,
        metric='torus', period=1.0, n_bootstrap=1000, ci_level=0.95, random_state=0, min_region_events=1, min_fold_events=30, sign_flip_exact_max_folds=22, sign_flip_monte_carlo_draws=2**20, dcor_n_sub=2500, dcor_n_rep=1
    )
    assert result['ci_low'] <= 0.0 <= result['ci_high']
    assert result['p_value'] > 0.05


def test_fold_bootstrap_degenerate_input_returns_non_significant():
    """Fewer than two usable folds -> NaN margin and p=1.0 (cannot pass), no raise."""
    actual = _folds_from_predictions(
        n_folds=1, n_per_fold=200,
        prediction_fn=lambda y, _rng: y, seed=0)
    null = {'test_indices': actual['test_indices'], 'y_true': actual['y_true'],
            'y_pred_xy': actual['y_pred_xy']}
    result = _fold_paired_margin_bootstrap(
        actual, null,
        metric='torus', period=1.0, n_bootstrap=100, ci_level=0.95, random_state=0, min_region_events=1, min_fold_events=30, sign_flip_exact_max_folds=22, sign_flip_monte_carlo_draws=2**20, dcor_n_sub=2500, dcor_n_rep=1
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
        event_to_region=event_to_region, min_region_events=20, min_fold_events=30, sign_flip_exact_max_folds=22, sign_flip_monte_carlo_draws=2**20, dcor_n_sub=2500, dcor_n_rep=1
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
        min_fold_events=30, min_region_events=1, sign_flip_exact_max_folds=22, sign_flip_monte_carlo_draws=2**20, dcor_n_sub=2500, dcor_n_rep=1
    )
    assert result['n_folds'] == 6


def test_selection_gate_settings_are_present():
    """The three selection-gate settings load as sensible numbers."""
    assert 0.0 < _SELECTION_EFFECT_FLOOR < 1.0
    assert _SELECTION_N_BOOTSTRAP >= 100
    assert 0.0 < _SELECTION_CI_LEVEL < 1.0


def test_frozen_kappa_scores_worse_than_marginal_below_the_flat_floor():
    """With the frozen concentration a fold predicted worse than the marginal scores
    below the flat-torus density -ln(4 pi^2) (a signed score), whereas a kappa refit on
    that fold's own residuals is clamped to 0 and returns exactly the floor -- the
    zero-margin floor that hid half the folds of the torus screen."""
    rng = np.random.default_rng(0)
    n = 4000
    y_true = (np.column_stack([rng.vonmises(0.0, 2.0, n), rng.vonmises(0.0, 2.0, n)]) / (2 * np.pi)) % 1.0
    kappa = _frozen_selection_kappa(y_true, np.ones(n), metric='torus', period=1.0)
    assert kappa > 0.5
    # Predictions half a period away from where the USVs are: worse than the marginal.
    y_bad = (y_true + 0.5 + rng.normal(0.0, 0.05, (n, 2))) % 1.0
    floor = -np.log(4.0 * np.pi ** 2)
    self_fit = macro_von_mises_logscore(y_bad, y_true, None, metric='torus', period=1.0, min_region_events=1)
    frozen = macro_von_mises_logscore(y_bad, y_true, None, metric='torus', period=1.0, kappa=kappa, min_region_events=1)
    assert abs(self_fit - floor) < 1e-9
    assert frozen < floor - 0.1


def test_fold_bootstrap_no_signal_acceptance_stays_at_its_measured_rate():
    """Calibration guard: with no true effect (candidate and incumbent both independent of the
    truth) the gate at the shipped CI level rarely accepts a single candidate (0.5% of 200
    replicates at K = 10 folds, measured 2026-10-16; bound 6%, while a 50% CI accepts 22.5%), the
    paired margin is unbiased, and the bootstrap p-value is not concentrated near 0. A change that
    loosens the gate (a sign flip, a one-sided CI on the wrong side, unpaired folds) fails here."""

    n_replicates = 200
    accepted, margins, p_values = 0, [], []
    for replicate in range(n_replicates):
        actual = _folds_from_predictions(
            n_folds=10, n_per_fold=60, prediction_fn=lambda y, rng: rng.random(y.shape), seed=10_000 + replicate)
        null = _paired_null(actual, seed=20_000 + 10 * replicate)
        result = _fold_paired_margin_bootstrap(
            actual, null, metric='torus', period=1.0, n_bootstrap=500,
            ci_level=_SELECTION_CI_LEVEL, random_state=replicate, kappa=1.0, min_region_events=1, min_fold_events=30, sign_flip_exact_max_folds=22, sign_flip_monte_carlo_draws=2**20, dcor_n_sub=2500, dcor_n_rep=1
        )
        accepted += int(result['ci_low'] > 0.0)
        margins.append(result['mean_margin'])
        p_values.append(result['p_value'])
    margins = np.asarray(margins)
    assert accepted / n_replicates <= 0.06
    assert abs(margins.mean()) < 3.0 * margins.std(ddof=1) / np.sqrt(n_replicates)
    assert np.mean(np.asarray(p_values) < 0.05) < 0.15
