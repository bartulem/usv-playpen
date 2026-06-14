"""
Regression tests for the bout-onset selector's per-fold filter-shape
loop in ``model_selection.py``.

History: the bout-onset selector previously did a single global-
balanced GAM fit at the end and saved a one-entry ``filter_shapes``
list, which collapsed the downstream plotter's percentile band to
a zero-width line. The lost per-fold data was only spotted after a
6-day production run. The fix on commit ``56efb08`` restored the
per-fold loop. These tests pin the structural invariants the
plotter consumes so a future regression that collapses the layout
breaks at test time rather than after another long run.
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest


def _make_fake_gam_class(n_features: int):
    """
    Build a fake ``LogisticGAM`` class whose ``.fit(...).predict_mu(grid)``
    returns a deterministic function of the grid columns so the
    partial-dependence delta is non-zero per feature.
    """

    class _FakeGAM:
        def __init__(self, *_args, **_kwargs):
            self._fitted = False

        def fit(self, _X, _y):
            self._fitted = True
            return self

        def predict_mu(self, grid):
            mu = np.zeros(grid.shape[0], dtype=float)
            for k in range(n_features):
                mu += (k + 1) * grid[:, 2 * k]
            return mu

    return _FakeGAM


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_compute_filter_shapes_per_fold_returns_one_dict_per_fold(mocker):
    """
    Plotter contract: ``filter_shapes`` must be a list of
    ``len(cv_folds)`` dicts, each keyed by every
    ``current_model_features`` entry, each value a 1-D array of
    length ``history_frames`` with non-zero deltas. A regression
    that collapses to a single entry breaks this test.
    """

    history_frames = 8
    n_features = 3
    feature_names = [f"feat_{i}" for i in range(n_features)]
    n_folds = 4

    fake_gam_cls = _make_fake_gam_class(n_features)
    mocker.patch(
        'usv_playpen.modeling.model_selection.LogisticGAM',
        fake_gam_cls,
    )
    mocker.patch(
        'usv_playpen.modeling.model_selection.te',
        lambda *_a, **_kw: 1,
    )

    n_pos_total = 12
    n_neg_total = 12
    rng = np.random.default_rng(0)
    pooled_feature_cache = {
        fn: {
            'X_pos': rng.standard_normal((n_pos_total, history_frames)),
            'X_neg': rng.standard_normal((n_neg_total, history_frames)),
        }
        for fn in feature_names
    }
    for fn in feature_names:
        pooled_feature_cache[fn]['X_full'] = np.concatenate(
            (pooled_feature_cache[fn]['X_pos'], pooled_feature_cache[fn]['X_neg']),
            axis=0,
        )

    cv_folds = []
    all_idx = np.arange(n_pos_total + n_neg_total)
    for f in range(n_folds):
        test_ix = all_idx[f::n_folds]
        train_ix = np.array([i for i in all_idx if i not in test_ix])
        cv_folds.append({
            'type': 'mixed',
            'train_idx': train_ix,
            'test_idx': test_ix,
            'n_pos_total': n_pos_total,
            'n_neg_total': n_neg_total,
        })

    from usv_playpen.modeling.model_selection import (
        compute_filter_shapes_per_fold_bout_onset,
    )

    result = compute_filter_shapes_per_fold_bout_onset(
        cv_folds=cv_folds,
        current_model_features=feature_names,
        all_feature_data={},
        pooled_feature_cache=pooled_feature_cache,
        history_frames=history_frames,
        n_splines_value=4,
        n_splines_time=4,
        gam_kwargs={},
        random_seed=42,
        time_indices=np.arange(history_frames, dtype=float),
    )

    assert isinstance(result, list), "filter_shapes must be a list"
    assert len(result) == n_folds, (
        f"Expected one filter_shapes entry per CV fold "
        f"({n_folds}); got {len(result)}. The plotter's percentile "
        "band needs >= 2 folds to render."
    )
    for fold_idx, fold_res in enumerate(result):
        assert isinstance(fold_res, dict), \
            f"fold {fold_idx}: each entry must be a dict"
        assert set(fold_res.keys()) == set(feature_names), (
            f"fold {fold_idx}: keys {sorted(fold_res.keys())} "
            f"don't match expected {sorted(feature_names)}"
        )
        for fn, arr in fold_res.items():
            assert isinstance(arr, np.ndarray), \
                f"fold {fold_idx}: {fn} must be a numpy array"
            assert arr.shape == (history_frames,), (
                f"fold {fold_idx}: {fn} has shape {arr.shape}, "
                f"expected ({history_frames},)"
            )
            assert not np.allclose(arr, 0.0), (
                f"fold {fold_idx}: {fn} filter is all zero -- "
                "the test_grid manipulation likely no-ops."
            )


def test_load_selection_results_round_trips_multi_fold_filter_shapes(tmp_path):
    """
    The consolidated artifact must keep ``filter_shapes`` as a list
    of dicts (one per fold). The loader returns it unchanged; the
    plotter consumes that exact shape. This test pins the shape so
    a future load-side refactor cannot silently flatten it.
    """

    from usv_playpen.modeling.modeling_metadata import load_selection_results

    history_frames = 5
    feature_names = ['feat_a', 'feat_b']
    n_folds = 3
    fake_filter = [
        {fn: np.linspace(0.0, 1.0, history_frames) for fn in feature_names}
        for _ in range(n_folds)
    ]
    consolidated = {
        'steps': [
            {'step_idx': 0, 'selected_feature': 'feat_a',
             'current_features': ['feat_a'], 'candidates_summary': {}},
            {'step_idx': 1, 'selected_feature': 'feat_b',
             'current_features': ['feat_a', 'feat_b'],
             'candidates_summary': {},
             'filter_shapes': fake_filter,
             'final_model_features': ['feat_a', 'feat_b']},
        ],
        '_run_metadata': {},
        '_input_metadata': {},
        '_consolidation_metadata': {},
    }
    pkl = tmp_path / 'selection_synth.pkl'
    with pkl.open('wb') as fh:
        pickle.dump(consolidated, fh)

    steps, display_name, _md = load_selection_results(pkl)
    assert display_name == 'selection_synth.pkl'
    assert len(steps) == 2
    fs = steps[-1]['filter_shapes']
    assert isinstance(fs, list)
    assert len(fs) == n_folds
    for entry in fs:
        assert set(entry.keys()) == set(feature_names)
        for fn in feature_names:
            assert entry[fn].shape == (history_frames,)
