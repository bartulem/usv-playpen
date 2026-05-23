"""
Regression tests for the bout-onset selector's per-fold filter-shape
loop in ``model_selection.py`` and the legacy-pickle recovery
utility in ``recompute_filter_shapes.py``.

History: the bout-onset selector previously did a single global-
balanced GAM fit at the end and saved a one-entry ``filter_shapes``
list, which collapsed the downstream plotter's percentile band to
a zero-width line. The lost per-fold data was only spotted after a
6-day production run. The fix on commit ``56efb08`` restored the
per-fold loop; the recovery utility patches pre-fix pickles
in-place.

These tests:
1. Exercise ``compute_filter_shapes_per_fold_bout_onset`` in
   isolation with a mocked ``LogisticGAM`` and assert the
   structural invariants the plotter consumes
   (``filter_shapes`` is a list of ``n_folds`` dicts; each entry
   keyed by every selected feature; each value a 1-D array of
   length ``history_frames``).
2. Round-trip a synthetic consolidated artifact through
   ``load_selection_results`` and assert the multi-fold layout
   survives.
3. End-to-end smoke-test ``recompute_filter_shapes`` against a
   synthetic consolidated artifact + minimal feature-data
   sidecar, with ``pool_session_arrays`` and ``LogisticGAM``
   mocked. Verifies the backup is created, the in-place write
   carries ``len(cv_folds)`` filter-shape dicts, and a re-run
   refuses to overwrite the existing backup.
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest


def _make_fake_gam_class(history_frames: int, n_features: int):
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

    fake_gam_cls = _make_fake_gam_class(history_frames, n_features)
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


# Recovery-utility tests below.


def _make_pre_fix_consolidated(
        *,
        feature_names,
        n_sessions,
        history_frames,
        random_seed,
        n_splits,
        test_proportion,
        anchor_feature,
):
    """
    Build a synthetic consolidated bout-onset pickle that mimics the
    pre-fix layout: full ``_run_metadata`` + ``_input_metadata``
    blocks, a 2-step ``steps`` list, and a SINGLE-entry
    ``filter_shapes`` on the last step (the regression we're
    recovering from).
    """

    session_ids = [f"sess_{i:03d}" for i in range(n_sessions)]
    consolidated = {
        'steps': [
            {
                'step_idx': 0,
                'selected_feature': feature_names[0],
                'current_features': [feature_names[0]],
                'candidates_summary': {},
            },
            {
                'step_idx': 1,
                'selected_feature': None,  # rejection step
                'current_features': feature_names,
                'final_model_features': feature_names,
                'candidates_summary': {},
                'filter_shapes': [
                    {fn: np.zeros(history_frames) for fn in feature_names}
                ],
                'input_data_path': '/fake/input.pkl',
                'univariate_results_path': '/fake/univariate.pkl',
            },
        ],
        '_run_metadata': {
            'random_seed': random_seed,
            'split_strategy': 'mixed',
            'n_splits_selection': n_splits,
            'test_proportion': test_proportion,
            'anchor_feature': anchor_feature,
            'gam_kwargs': {'max_iter': 100, 'tol': 1e-4, 'lam': 0.6},
            'extra_knobs': {
                'n_splines_time': 8,
                'n_splines_value': 5,
            },
            'selection_function': 'bout_onset_model_selection',
        },
        '_input_metadata': {
            'filter_history_frames': history_frames,
            'session_ids': session_ids,
        },
        '_consolidation_metadata': {},
        '_univariate_metadata': {},
    }
    return consolidated


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_recompute_filter_shapes_writes_multi_fold_in_place(tmp_path, mocker):
    """
    End-to-end recovery: legacy single-entry filter_shapes ->
    multi-fold filter_shapes, original backed up.
    """

    history_frames = 6
    feature_names = ['feat_a', 'feat_b']
    n_features = len(feature_names)
    n_sessions = 12
    n_splits = 4
    test_proportion = 0.2
    random_seed = 0
    anchor_feature = feature_names[0]

    consolidated = _make_pre_fix_consolidated(
        feature_names=feature_names,
        n_sessions=n_sessions,
        history_frames=history_frames,
        random_seed=random_seed,
        n_splits=n_splits,
        test_proportion=test_proportion,
        anchor_feature=anchor_feature,
    )
    pkl_path = tmp_path / 'selection_synth_prefix.pkl'
    with pkl_path.open('wb') as fh:
        pickle.dump(consolidated, fh)

    # Stub out IO + heavy fitting:
    #   * configure_path is identity-mapping for the fake path so the
    #     resolver doesn't need a real filesystem.
    #   * load_pickle_modeling_data returns a per-feature dict of
    #     per-session arrays with positive + negative rows.
    #   * pool_session_arrays just concatenates the per-session
    #     positives / negatives.
    #   * LogisticGAM is the deterministic fake from above.
    def _identity_path(p):
        return p

    mocker.patch(
        'usv_playpen.modeling.recompute_filter_shapes.configure_path',
        side_effect=_identity_path,
    )
    mocker.patch(
        'usv_playpen.modeling.recompute_filter_shapes._resolve_local_path',
        side_effect=lambda p: pkl_path,  # any existing file works
    )

    rng = np.random.default_rng(42)
    per_feat_per_sess = {
        fn: {
            sess: {
                'usv_feature_arr': rng.standard_normal((5, history_frames)),
                'no_usv_feature_arr': rng.standard_normal((7, history_frames)),
            }
            for sess in [f"sess_{i:03d}" for i in range(n_sessions)]
        }
        for fn in feature_names
    }
    mocker.patch(
        'usv_playpen.modeling.recompute_filter_shapes.load_pickle_modeling_data',
        return_value=per_feat_per_sess,
    )

    def _fake_pool_session_arrays(per_sess_data, sessions, *, pos_key, neg_key, n_frames):
        Xp = np.concatenate(
            [per_sess_data[s][pos_key] for s in sessions if s in per_sess_data],
            axis=0,
        )
        Xn = np.concatenate(
            [per_sess_data[s][neg_key] for s in sessions if s in per_sess_data],
            axis=0,
        )
        return Xp, Xn

    mocker.patch(
        'usv_playpen.modeling.recompute_filter_shapes.pool_session_arrays',
        side_effect=_fake_pool_session_arrays,
    )

    fake_gam_cls = _make_fake_gam_class(history_frames, n_features)
    mocker.patch(
        'usv_playpen.modeling.model_selection.LogisticGAM',
        fake_gam_cls,
    )
    mocker.patch(
        'usv_playpen.modeling.model_selection.te',
        lambda *_a, **_kw: 1,
    )

    from usv_playpen.modeling.recompute_filter_shapes import (
        BACKUP_SUFFIX,
        recompute_filter_shapes,
    )

    out_path = recompute_filter_shapes(str(pkl_path))
    assert out_path == pkl_path.resolve()

    backup_path = pkl_path.with_suffix(pkl_path.suffix + BACKUP_SUFFIX)
    assert backup_path.exists(), "backup file should be created"

    with pkl_path.open('rb') as fh:
        patched = pickle.load(fh)
    fs = patched['steps'][-1]['filter_shapes']
    assert isinstance(fs, list)
    assert len(fs) == n_splits, (
        f"Expected {n_splits} folds after recovery; got {len(fs)}."
    )
    for entry in fs:
        assert set(entry.keys()) == set(feature_names)
        for fn in feature_names:
            assert entry[fn].shape == (history_frames,)

    # Backup is the original pre-fix pickle (single-entry list).
    with backup_path.open('rb') as fh:
        backed_up = pickle.load(fh)
    assert len(backed_up['steps'][-1]['filter_shapes']) == 1


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_recompute_filter_shapes_refuses_to_overwrite_existing_backup(
        tmp_path, mocker,
):
    """A second run on an already-recovered pickle must error out."""

    from usv_playpen.modeling.recompute_filter_shapes import (
        BACKUP_SUFFIX,
        recompute_filter_shapes,
    )

    pkl = tmp_path / 'selection_already_recovered.pkl'
    pkl.write_bytes(b'whatever')
    backup = pkl.with_suffix(pkl.suffix + BACKUP_SUFFIX)
    backup.write_bytes(b'old')

    with pytest.raises(FileExistsError):
        recompute_filter_shapes(str(pkl))


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_recompute_filter_shapes_validates_required_metadata(tmp_path):
    """
    Missing-metadata path must raise ValueError with a clear list
    of missing keys so the user can fix the pickle or accept that
    exact reproduction isn't possible.
    """

    from usv_playpen.modeling.recompute_filter_shapes import (
        recompute_filter_shapes,
    )

    bare = {
        'steps': [{
            'step_idx': 0,
            'final_model_features': ['feat_a'],
            'filter_shapes': [{'feat_a': np.zeros(5)}],
            'input_data_path': '/fake.pkl',
            'univariate_results_path': '/fake_u.pkl',
        }],
        '_run_metadata': {},  # everything missing
        '_input_metadata': {},
        '_consolidation_metadata': {},
    }
    pkl = tmp_path / 'selection_no_metadata.pkl'
    with pkl.open('wb') as fh:
        pickle.dump(bare, fh)

    with pytest.raises(ValueError, match=r"_run_metadata\.random_seed"):
        recompute_filter_shapes(str(pkl))
