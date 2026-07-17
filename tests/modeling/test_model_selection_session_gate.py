"""
@author: bartulem
Unit tests for the session-grain manifold selection gate helpers in
``usv_playpen.modeling.model_selection``.

These drive the two building blocks of the behaviour -> acoustic-manifold-position
feature-acceptance rule directly, with hand-built metadata and CV-fold dicts, so
no model fitting or on-disk artefacts are needed:

* ``_build_event_to_session`` -- the per-event -> session-index reconstruction
  from the univariate ``_input_metadata`` (both the ``{session: count}`` and the
  ``{session: {'usv': count}}`` count encodings), plus its coverage semantics;
* ``_session_paired_margin_bootstrap`` -- the per-session paired dcor margin and
  session bootstrap that both the screen and the forward-selection acceptance
  use: a predictor that tracks the truth must clear the gate (CI lower bound
  above 0, one-sided p ~ 0), a predictor independent of the truth must not (CI
  spanning 0, p ~ 0.5), and a degenerate single-session input must return the
  non-significant sentinel rather than raise.
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
        _build_event_to_session,
        _session_paired_margin_bootstrap,
        _SELECTION_EFFECT_FLOOR,
        _SELECTION_N_BOOTSTRAP,
        _SELECTION_CI_LEVEL,
    )


def _make_metadata(counts, nested):
    """Build an `_input_metadata`-shaped dict for `counts` events per session."""
    session_ids = [f"s{i}" for i in range(len(counts))]
    if nested:
        n_events = {s: {'usv': c} for s, c in zip(session_ids, counts)}
    else:
        n_events = {s: c for s, c in zip(session_ids, counts)}
    return {'session_ids': session_ids, 'n_events_per_session': n_events}


@pytest.mark.parametrize("nested", [False, True])
def test_build_event_to_session_maps_contiguous_blocks(nested):
    """Each session owns a contiguous, correctly ordered block of global indices."""
    counts = [3, 1, 4]
    event_to_session = _build_event_to_session(_make_metadata(counts, nested))
    assert event_to_session.tolist() == [0, 0, 0, 1, 2, 2, 2, 2]
    assert event_to_session.shape[0] == sum(counts)
    assert set(np.unique(event_to_session)) == set(range(len(counts)))


def _folds_from_predictions(event_to_session, prediction_fn, seed):
    """Leave-one-session-out folds; `prediction_fn(truth)` builds the predictions."""
    rng = np.random.default_rng(seed)
    n_sessions = int(event_to_session.max()) + 1
    truth = rng.random((event_to_session.shape[0], 2))
    folds = {'test_indices': [], 'y_pred_xy': [], 'y_true': []}
    for session_index in range(n_sessions):
        idx = np.where(event_to_session == session_index)[0]
        folds['test_indices'].append(idx)
        folds['y_true'].append(truth[idx])
        folds['y_pred_xy'].append(prediction_fn(truth[idx], rng))
    return folds, truth


def test_session_bootstrap_accepts_a_real_predictor():
    """A prediction that tracks the truth clears the gate: CI lower bound > 0, p ~ 0."""
    counts = [200] * 6
    event_to_session = _build_event_to_session(_make_metadata(counts, nested=False))
    # `actual` prediction = truth plus small noise (dependent); `null` = independent draw.
    actual, truth = _folds_from_predictions(
        event_to_session,
        lambda y, rng: np.mod(y + 0.05 * rng.standard_normal(y.shape), 1.0),
        seed=0,
    )
    null = {'test_indices': actual['test_indices'], 'y_true': actual['y_true'],
            'y_pred_xy': [np.random.default_rng(k).random(a.shape)
                          for k, a in enumerate(actual['y_pred_xy'])]}
    result = _session_paired_margin_bootstrap(
        actual, null, event_to_session,
        metric='torus', period=1.0, n_bootstrap=1000, ci_level=0.95, random_state=0,
    )
    assert result['n_sessions'] == len(counts)
    assert result['mean_margin'] > 0.0
    assert result['ci_low'] > 0.0
    assert result['p_value'] < 0.05


def test_session_bootstrap_rejects_a_null_predictor():
    """A prediction independent of the truth fails: CI spans 0, p is not small."""
    counts = [200] * 6
    event_to_session = _build_event_to_session(_make_metadata(counts, nested=False))
    actual, _ = _folds_from_predictions(
        event_to_session, lambda y, rng: rng.random(y.shape), seed=1)
    null, _ = _folds_from_predictions(
        event_to_session, lambda y, rng: rng.random(y.shape), seed=2)
    # Share the truth / test indices so the pairing is well defined per session.
    null['test_indices'] = actual['test_indices']
    null['y_true'] = actual['y_true']
    result = _session_paired_margin_bootstrap(
        actual, null, event_to_session,
        metric='torus', period=1.0, n_bootstrap=1000, ci_level=0.95, random_state=0,
    )
    assert result['ci_low'] <= 0.0 <= result['ci_high']
    assert result['p_value'] > 0.05


def test_session_bootstrap_degenerate_input_returns_non_significant():
    """Fewer than two usable sessions -> NaN margin and p=1.0 (cannot pass), no raise."""
    counts = [200]
    event_to_session = _build_event_to_session(_make_metadata(counts, nested=False))
    actual, _ = _folds_from_predictions(
        event_to_session, lambda y, rng: y, seed=0)
    null = {'test_indices': actual['test_indices'], 'y_true': actual['y_true'],
            'y_pred_xy': actual['y_pred_xy']}
    result = _session_paired_margin_bootstrap(
        actual, null, event_to_session,
        metric='torus', period=1.0, n_bootstrap=100, ci_level=0.95, random_state=0,
    )
    assert result['n_sessions'] < 2
    assert np.isnan(result['mean_margin'])
    assert result['p_value'] == 1.0


def test_selection_gate_settings_are_present():
    """The three new selection-gate settings load as sensible numbers."""
    assert 0.0 < _SELECTION_EFFECT_FLOOR < 1.0
    assert _SELECTION_N_BOOTSTRAP >= 100
    assert 0.0 < _SELECTION_CI_LEVEL < 1.0
