"""
@author: bartulem
Tail-coverage tests for ``usv_playpen.modeling.model_selection``.

These tests deliberately do NOT re-walk the happy paths already covered by the
``test_pipeline_*`` smoke suites. Instead they drive the *uncovered* branches of
the forward-stepwise selectors:

* the standalone ``compute_filter_shapes_per_fold_bout_onset`` helper, exercised
  directly with hand-built CV-fold dicts so its unknown-fold-type skip, empty-
  class skip, and per-fold ``except`` accounting all execute;
* ``bout_onset_model_selection`` resume-from-checkpoint logic (a pre-existing
  Step pickle in the output directory makes the selector resume instead of
  recomputing), the unknown-``split_strategy`` ``raise``, the
  no-significant-candidates early-abort, the per-fold ``except`` (estimator
  forced to raise via ``mocker``) and the "all folds failed -> RuntimeError"
  final-refit guard;
* the ``vocal_category_model_selection`` and ``bout_parameter_model_selection``
  resume + per-fold-failure branches via the same monkeypatched-fit technique.

To keep the data tiny and the run fast, the univariate "ranking" pickle these
selectors screen is synthesized directly (a strong-signal ``actual`` log-loss
well below a chance-level ``null`` so the Bonferroni screen admits the
candidates) rather than computed by re-running the univariate dispatcher; the
modeling-input pickle reuses the ``_synth`` builders. Everything lives under
``tmp_path`` so the source-tree integrity guard never trips.

Warning policy
--------------
The project runs pytest with ``filterwarnings = ["error"]``. The modeling
import chain pulls ``optax`` -> a one-time JAX ``DeprecationWarning``, so the
top-level modeling imports are wrapped in a ``warnings.catch_warnings`` block
that ignores ``DeprecationWarning`` during import. ``pygam`` (Python 3.13)
emits ``DeprecationWarning: Bitwise inversion '~' on bool`` from inside its GAM
fit, demoted with a narrow per-test marker. ``matplotlib`` is forced onto the
headless ``Agg`` backend because the modeling import chain pulls ``pyplot``.
"""

from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use('Agg')

from tests.modeling._synth import (
    build_modeling_input_pickle,
    build_modeling_settings,
    build_session_tree,
    write_session_list_file,
)

# The modeling import chain pulls optax -> a one-time JAX DeprecationWarning.
# Guard the top-level imports so collection does not trip ``filterwarnings =
# ["error"]`` before any per-test marker can take effect.
with warnings.catch_warnings():
    warnings.simplefilter('ignore', DeprecationWarning)
    from usv_playpen.modeling import model_selection as ms
    from usv_playpen.modeling.model_selection import (
        bout_onset_model_selection,
        bout_parameter_model_selection,
        compute_filter_shapes_per_fold_bout_onset,
        continuous_vocal_manifold_model_selection,
        get_unrolled_X_for_multivariate,
        multinomial_vocal_category_model_selection,
        vocal_category_model_selection,
    )


# Tiny-data geometry. ``HISTORY_FRAMES`` is the column count of every per-event
# window; the synthetic input pickle must match the settings'
# ``floor(CAMERA_FPS * FILTER_HISTORY)``.
CAMERA_FPS = 60.0
FILTER_HISTORY = 0.5
HISTORY_FRAMES = int(np.floor(CAMERA_FPS * FILTER_HISTORY))
N_SESSIONS = 4


def _build_settings(tmp_path: Path, **overrides):
    """
    Description
    -----------
    Builds the synthetic session tree, the session-list file, and a trimmed
    ``modeling_settings`` dict rooted under ``tmp_path``. The session tree is
    only built so the settings' ``io`` paths point at real files; the selector
    tests inject their own input pickle and therefore never touch the on-disk
    extraction.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        The pytest-provided per-test scratch directory; every artifact lives
        below it so nothing is ever written into the package tree.
    overrides (dict)
        Extra keyword arguments forwarded to ``build_modeling_settings`` (e.g.
        ``model_engine``, ``split_strategy``, ``split_num``).

    Returns
    -------
    settings (dict)
        The ready-to-use ``modeling_settings`` dictionary.
    save_dir (pathlib.Path)
        The pipeline output directory (``tmp_path / 'out'``).
    """

    session_roots = build_session_tree(
        base_dir=tmp_path / 'sessions',
        n_sessions=N_SESSIONS,
        n_frames=1200,
        camera_fps=CAMERA_FPS,
        filter_history=FILTER_HISTORY,
        n_bouts=4,
        usv_per_bout=2,
    )
    list_file = write_session_list_file(session_roots, tmp_path / 'session_list.txt')
    save_dir = tmp_path / 'out'
    save_dir.mkdir(parents=True, exist_ok=True)

    settings = build_modeling_settings(
        session_list_file=list_file,
        save_directory=save_dir,
        camera_sampling_rate=CAMERA_FPS,
        filter_history=FILTER_HISTORY,
        **overrides,
    )
    settings['model_params']['usv_bout_time'] = FILTER_HISTORY
    return settings, save_dir


def _write_onset_univariate_ranking(
        save_path: Path,
        feature_names: list[str],
        n_folds: int = 4,
        significant: bool = True,
) -> Path:
    """
    Description
    -----------
    Synthesizes the consolidated univariate "ranking" pickle the bout-onset
    selector screens. Each feature carries an ``actual`` and a ``null`` log-loss
    array; when ``significant`` is True the ``actual`` mean log-loss is far below
    the chance-level ``null`` so the selector's Bonferroni screen admits the
    feature as a candidate. When False, ``actual`` equals ``null`` so no feature
    survives and the selector takes its "no significant features" abort branch.

    Parameters
    ----------
    save_path (pathlib.Path)
        Destination ``.pkl`` path (parent dirs created if absent).
    feature_names (list of str)
        Feature keys to emit, in the desired rank order (lower ``actual`` ll
        sorts first; the helper assigns ascending means in list order).
    n_folds (int)
        Number of per-fold log-loss entries per branch.
    significant (bool)
        Whether the synthesized ``actual`` log-loss clears the null screen.

    Returns
    -------
    save_path (pathlib.Path)
        The path the pickle was written to (absolute).
    """

    artifact: dict = {}
    for f_idx, feature in enumerate(feature_names):
        if significant:
            actual_ll = np.full(n_folds, 0.20 + 0.01 * f_idx, dtype=float)
        else:
            actual_ll = np.full(n_folds, 0.69, dtype=float)
        null_ll = np.full(n_folds, 0.69, dtype=float)
        artifact[feature] = {
            'actual': {'ll': actual_ll},
            'null': {'ll': null_ll},
        }

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open('wb') as fh:
        pickle.dump(artifact, fh)
    return save_path


def _make_onset_inputs(tmp_path: Path, feature_names, n_sessions=N_SESSIONS,
                       n_usv=24, n_no_usv=40):
    """
    Description
    -----------
    Builds the bout-onset modeling-input pickle (feature -> session ->
    ``usv_feature_arr`` / ``no_usv_feature_arr``) plus a matching strong-signal
    univariate ranking pickle, returning both paths and the session-id list.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Per-test scratch directory.
    feature_names (list of str)
        Behavioral-feature keys to populate.
    n_sessions (int)
        Number of synthetic sessions.
    n_usv (int)
        Positive (USV-bout) events per session.
    n_no_usv (int)
        Negative (No-USV) events per session.

    Returns
    -------
    input_pkl (str)
        Path to the modeling-input pickle.
    ranking_pkl (str)
        Path to the univariate ranking pickle.
    session_ids (list of str)
        The session identifiers the pickle was populated with.
    """

    session_ids = [f'session_{i}' for i in range(n_sessions)]
    input_pkl = build_modeling_input_pickle(
        save_path=tmp_path / 'modeling_input.pkl',
        feature_names=feature_names,
        session_ids=session_ids,
        history_frames=HISTORY_FRAMES,
        n_usv=n_usv,
        n_no_usv=n_no_usv,
        input_metadata={'analysis_tag': 'bout'},
    )
    ranking_pkl = _write_onset_univariate_ranking(
        save_path=tmp_path / 'univariate_combined.pkl',
        feature_names=feature_names,
    )
    return str(input_pkl), str(ranking_pkl), session_ids


def _settings_json(tmp_path: Path, settings: dict) -> str:
    """
    Description
    -----------
    Serializes a synthetic ``modeling_settings`` dict to ``tmp_path`` and returns
    its path, so the selectors load the shrunk settings rather than the package
    JSON.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Per-test scratch directory.
    settings (dict)
        The settings dictionary to serialize.

    Returns
    -------
    path (str)
        Absolute path to the written JSON file.
    """

    p = tmp_path / 'settings.json'
    p.write_text(json.dumps(settings))
    return str(p)


class TestComputeFilterShapesHelper:
    """Direct unit coverage of ``compute_filter_shapes_per_fold_bout_onset``."""

    def test_unknown_fold_type_is_skipped(self):
        """
        A fold dict whose ``type`` is neither ``'session'`` nor ``'mixed'`` hits
        the helper's unknown-fold-type branch: it prints a skip line and
        ``continue``s, contributing no entry to the returned list. With a single
        unknown fold the returned list is therefore empty.
        """

        out = compute_filter_shapes_per_fold_bout_onset(
            cv_folds=[{'type': 'bogus'}],
            current_model_features=['f0'],
            all_feature_data={},
            pooled_feature_cache={},
            history_frames=HISTORY_FRAMES,
            n_splines_value=4,
            n_splines_time=4,
            gam_kwargs={'max_iter': 5, 'tol': 1e-3, 'lam': 0.6},
            random_seed=0,
            time_indices=np.arange(HISTORY_FRAMES, dtype=float),
        )
        assert out == []

    def test_empty_class_fold_is_skipped(self):
        """
        A 'mixed' fold whose balanced training split lands an empty positive (or
        negative) class triggers the ``n_k == 0`` skip branch. Building a
        ``train_idx`` that selects only negative rows (all indices >=
        ``n_pos_total``) forces ``n_pos == 0`` so the fold is skipped and the
        returned list is empty.
        """

        n_pos_total, n_neg_total = 5, 5
        x_full = np.zeros((n_pos_total + n_neg_total, HISTORY_FRAMES), dtype=float)
        # train_idx selects only the negative half -> zero positive rows.
        neg_only_idx = np.arange(n_pos_total, n_pos_total + n_neg_total)
        out = compute_filter_shapes_per_fold_bout_onset(
            cv_folds=[{
                'type': 'mixed',
                'train_idx': neg_only_idx,
                'n_pos_total': n_pos_total,
                'n_neg_total': n_neg_total,
            }],
            current_model_features=['f0'],
            all_feature_data={},
            pooled_feature_cache={'f0': {'X_full': x_full}},
            history_frames=HISTORY_FRAMES,
            n_splines_value=4,
            n_splines_time=4,
            gam_kwargs={'max_iter': 5, 'tol': 1e-3, 'lam': 0.6},
            random_seed=0,
            time_indices=np.arange(HISTORY_FRAMES, dtype=float),
        )
        assert out == []

    def test_per_fold_exception_is_caught(self, monkeypatch):
        """
        Forcing ``LogisticGAM`` (as referenced inside ``model_selection``) to
        raise on ``.fit`` drives the helper's per-fold ``except`` branch: the
        error is printed and the fold is skipped, leaving an empty result list
        without propagating the exception.
        """

        class _BoomGAM:
            def __init__(self, *a, **k):
                pass

            def fit(self, *a, **k):
                raise RuntimeError("forced fit failure")

        monkeypatch.setattr(ms, 'LogisticGAM', _BoomGAM)

        n_pos_total, n_neg_total = 6, 6
        x_full = np.random.default_rng(0).standard_normal(
            (n_pos_total + n_neg_total, HISTORY_FRAMES)
        )
        out = compute_filter_shapes_per_fold_bout_onset(
            cv_folds=[{
                'type': 'mixed',
                'train_idx': np.arange(n_pos_total + n_neg_total),
                'n_pos_total': n_pos_total,
                'n_neg_total': n_neg_total,
            }],
            current_model_features=['f0'],
            all_feature_data={},
            pooled_feature_cache={'f0': {'X_full': x_full}},
            history_frames=HISTORY_FRAMES,
            n_splines_value=4,
            n_splines_time=4,
            gam_kwargs={'max_iter': 5, 'tol': 1e-3, 'lam': 0.6},
            random_seed=0,
            time_indices=np.arange(HISTORY_FRAMES, dtype=float),
        )
        assert out == []


class TestBoutOnsetSelectionBranches:
    """Uncovered control-flow branches of ``bout_onset_model_selection``."""

    def test_no_significant_features_aborts(self, tmp_path):
        """
        When no univariate feature clears the Bonferroni null screen (synthesized
        here with ``actual`` log-loss equal to ``null``), the selector prints
        "No significant features found." and returns early without writing any
        step pickle.
        """

        settings, _ = _build_settings(tmp_path, model_engine='sklearn',
                                      split_strategy='mixed', split_num=2)
        feature_names = ['self.speed', 'other.speed']
        session_ids = [f'session_{i}' for i in range(N_SESSIONS)]
        input_pkl = str(build_modeling_input_pickle(
            save_path=tmp_path / 'modeling_input.pkl',
            feature_names=feature_names,
            session_ids=session_ids,
            history_frames=HISTORY_FRAMES,
            n_usv=20, n_no_usv=30,
            input_metadata={'analysis_tag': 'bout'},
        ))
        ranking_pkl = str(_write_onset_univariate_ranking(
            save_path=tmp_path / 'univariate_combined.pkl',
            feature_names=feature_names,
            significant=False,
        ))

        ms_dir = tmp_path / 'model_selection'
        ms_dir.mkdir()
        bout_onset_model_selection(
            univariate_results_path=ranking_pkl,
            input_data_path=input_pkl,
            settings_path=_settings_json(tmp_path, settings),
            output_directory=str(ms_dir),
            use_top_rank_as_anchor=True,
            p_val=0.01,
        )
        assert list(ms_dir.glob('*.pkl')) == []

    def test_unknown_split_strategy_raises(self, tmp_path):
        """
        A ``split_strategy`` that is neither ``'session'`` nor ``'mixed'`` makes
        the selector raise ``ValueError`` while building the CV folds.
        """

        settings, _ = _build_settings(tmp_path, model_engine='sklearn',
                                      split_strategy='mixed', split_num=2)
        settings['model_params']['split_strategy'] = 'galaxy_brain'
        feature_names = ['self.speed', 'other.speed']
        input_pkl, ranking_pkl, _ = _make_onset_inputs(tmp_path, feature_names)

        ms_dir = tmp_path / 'model_selection'
        ms_dir.mkdir()
        with pytest.raises(ValueError, match="Unknown split_strategy"):
            bout_onset_model_selection(
                univariate_results_path=ranking_pkl,
                input_data_path=input_pkl,
                settings_path=_settings_json(tmp_path, settings),
                output_directory=str(ms_dir),
                use_top_rank_as_anchor=True,
                p_val=0.01,
            )

    @pytest.mark.filterwarnings("ignore:Bitwise inversion:DeprecationWarning")
    def test_resume_from_existing_step_pickle(self, tmp_path):
        """
        Dropping a pre-existing ``model_selection_..._step_0.pkl`` into the output
        directory drives the resume-from-checkpoint branch: the selector detects
        the prior run, loads the last step's ``current_features`` /
        ``baseline_score`` / ``candidates_summary``, and (because the saved
        candidate beats the baseline by more than its SE) promotes that candidate
        and continues from the next step. The run completes and at least one
        further step pickle is produced.
        """

        settings, _ = _build_settings(tmp_path, model_engine='sklearn',
                                      split_strategy='mixed', split_num=2)
        feature_names = ['self.speed', 'other.speed', 'self.neck_elevation']
        input_pkl, ranking_pkl, _ = _make_onset_inputs(tmp_path, feature_names)

        ms_dir = tmp_path / 'model_selection'
        ms_dir.mkdir()

        # The prefix the selector computes for an 'unknown' target condition.
        prefix = 'model_selection_unknown_bout_mixed_step_'
        # A checkpoint whose stored candidate ('self.speed') clears the
        # (baseline - score) > se promotion test, so resume appends it.
        checkpoint = {
            'step_idx': 0,
            'current_features': [],
            'baseline_score': float(np.log(2)),
            'selected_feature': None,
            'candidates_summary': {
                'self.speed': {'mean_ll': 0.10, 'se_ll': 0.01},
            },
        }
        with (ms_dir / f'{prefix}0.pkl').open('wb') as fh:
            pickle.dump(checkpoint, fh)

        bout_onset_model_selection(
            univariate_results_path=ranking_pkl,
            input_data_path=input_pkl,
            settings_path=_settings_json(tmp_path, settings),
            output_directory=str(ms_dir),
            use_top_rank_as_anchor=False,
            p_val=0.5,
        )

        step_pkls = sorted(ms_dir.glob(f'{prefix}*.pkl'))
        assert len(step_pkls) >= 2

    @pytest.mark.filterwarnings("ignore:Bitwise inversion:DeprecationWarning")
    def test_all_folds_fail_raises_runtimeerror(self, tmp_path, monkeypatch):
        """
        Forcing every ``LogisticGAM`` fit to raise makes every selection fold fail
        (so the anchored Step-0 records no valid log-loss and no step pickle is
        written), and the final filter-shape refit therefore produces an empty
        fold list. The caller-side guard raises ``RuntimeError`` rather than
        silently emitting an artifact with no usable per-fold filter data.
        """

        class _BoomGAM:
            def __init__(self, *a, **k):
                pass

            def fit(self, *a, **k):
                raise RuntimeError("forced GAM failure")

        monkeypatch.setattr(ms, 'LogisticGAM', _BoomGAM)

        settings, _ = _build_settings(tmp_path, model_engine='sklearn',
                                      split_strategy='mixed', split_num=2)
        feature_names = ['self.speed', 'other.speed']
        input_pkl, ranking_pkl, _ = _make_onset_inputs(tmp_path, feature_names)

        ms_dir = tmp_path / 'model_selection'
        ms_dir.mkdir()
        with pytest.raises(RuntimeError, match="fold"):
            bout_onset_model_selection(
                univariate_results_path=ranking_pkl,
                input_data_path=input_pkl,
                settings_path=_settings_json(tmp_path, settings),
                output_directory=str(ms_dir),
                use_top_rank_as_anchor=True,
                p_val=0.5,
            )


class TestPooledCacheMisalignment:
    """The pooled-feature-cache anchor-misalignment guard in bout-onset selection."""

    def test_feature_sample_count_mismatch_raises(self, tmp_path):
        """
        If a non-anchor feature has a different pooled (positive, negative) sample
        count than the anchor, the 'mixed' fold indices would silently misindex
        it; the selector raises ``ValueError`` at cache-build time. Built by
        making one feature carry fewer positive events per session than the
        anchor.
        """

        session_ids = [f'session_{i}' for i in range(N_SESSIONS)]
        # Anchor feature with the full positive count; a second feature with a
        # short positive count so the pooled (n_pos, n_neg) drifts.
        artifact: dict = {}
        rng = np.random.default_rng(0)
        for feat, n_usv in (('self.speed', 24), ('other.speed', 12)):
            artifact[feat] = {}
            for sess in session_ids:
                artifact[feat][sess] = {
                    'usv_feature_arr': rng.standard_normal((n_usv, HISTORY_FRAMES)),
                    'no_usv_feature_arr': rng.standard_normal((40, HISTORY_FRAMES)),
                }
        artifact['_input_metadata'] = {'analysis_tag': 'bout'}
        input_pkl = tmp_path / 'modeling_input.pkl'
        with input_pkl.open('wb') as fh:
            pickle.dump(artifact, fh)

        ranking_pkl = str(_write_onset_univariate_ranking(
            save_path=tmp_path / 'univariate_combined.pkl',
            feature_names=['self.speed', 'other.speed'],
        ))

        settings, _ = _build_settings(tmp_path, model_engine='sklearn',
                                      split_strategy='mixed', split_num=2)
        ms_dir = tmp_path / 'model_selection'
        ms_dir.mkdir()
        with pytest.raises(ValueError, match="misalignment"):
            bout_onset_model_selection(
                univariate_results_path=ranking_pkl,
                input_data_path=str(input_pkl),
                settings_path=_settings_json(tmp_path, settings),
                output_directory=str(ms_dir),
                use_top_rank_as_anchor=True,
                p_val=0.5,
            )


CATEGORY_COLUMN = 'vae_supercategory'
TARGET_CATEGORY = 1


def _category_input_metadata():
    """
    Description
    -----------
    Returns the minimal ``_input_metadata`` block ``vocal_category_model_selection``
    consults on a directly-injected ranking/input pickle: the ``analysis_tag`` and
    the ``analysis_specific.usv_category_column_name`` the selector reads to build
    the per-step filename prefix.

    Returns
    -------
    metadata (dict)
        The reserved-block dict to embed under ``_input_metadata``.
    """

    return {
        'analysis_tag': f"category_{CATEGORY_COLUMN}_{TARGET_CATEGORY}",
        'analysis_specific': {
            'target_category': TARGET_CATEGORY,
            'usv_category_column_name': CATEGORY_COLUMN,
        },
    }


def _write_category_input_pickle(save_path, feature_names, session_ids,
                                 n_target=24, n_other=48, seed=0):
    """
    Description
    -----------
    Serializes a one-vs-rest category modeling-input pickle (feature -> session ->
    ``target_feature_arr`` / ``other_feature_arr``) with a strong class-dependent
    mean offset so a logistic model finds above-chance structure. The reserved
    ``_input_metadata`` block is embedded so the selector can resolve the category
    column name.

    Parameters
    ----------
    save_path (pathlib.Path)
        Destination ``.pkl`` path (parent dirs created if absent).
    feature_names (list of str)
        Generic feature keys to populate.
    session_ids (list of str)
        Session identifiers to populate under every feature.
    n_target (int)
        Positive (target-category) events per session.
    n_other (int)
        Negative (other-category) events per session.
    seed (int)
        Base seed for the per-cell RNG.

    Returns
    -------
    save_path (pathlib.Path)
        The path the pickle was written to (absolute).
    """

    rng = np.random.default_rng(seed)
    artifact: dict = {}
    for f_idx, feature in enumerate(feature_names):
        artifact[feature] = {}
        for sess in session_ids:
            target_arr = (0.9 + 0.1 * f_idx
                          + rng.standard_normal((n_target, HISTORY_FRAMES))).astype(float)
            other_arr = (-0.9 - 0.1 * f_idx
                         + rng.standard_normal((n_other, HISTORY_FRAMES))).astype(float)
            artifact[feature][sess] = {
                'target_feature_arr': target_arr,
                'other_feature_arr': other_arr,
            }
    artifact['_input_metadata'] = _category_input_metadata()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open('wb') as fh:
        pickle.dump(artifact, fh)
    return save_path


def _write_category_ranking(save_path, feature_names, n_folds=4):
    """
    Description
    -----------
    Synthesizes the consolidated univariate ranking pickle the category selector
    screens, with a strong-signal ``actual`` log-loss far below a chance-level
    ``null`` so the Bonferroni screen admits every feature. Carries the reserved
    ``_input_metadata`` block (the selector prefers it over the input pickle's).

    Parameters
    ----------
    save_path (pathlib.Path)
        Destination ``.pkl`` path.
    feature_names (list of str)
        Feature keys to emit, in desired rank order.
    n_folds (int)
        Per-fold log-loss entry count per branch.

    Returns
    -------
    save_path (pathlib.Path)
        The path the pickle was written to (absolute).
    """

    artifact: dict = {}
    for f_idx, feature in enumerate(feature_names):
        artifact[feature] = {
            'actual': {'ll': np.full(n_folds, 0.20 + 0.01 * f_idx, dtype=float)},
            'null': {'ll': np.full(n_folds, 0.69, dtype=float)},
        }
    artifact['_input_metadata'] = _category_input_metadata()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open('wb') as fh:
        pickle.dump(artifact, fh)
    return save_path


class TestCategorySelectionBranches:
    """Uncovered control-flow branches of ``vocal_category_model_selection``."""

    def test_unknown_split_strategy_raises(self, tmp_path):
        """
        A ``split_strategy`` that is neither ``'session'`` nor ``'mixed'`` makes
        the category selector raise ``ValueError`` while building the CV folds.
        """

        settings, _ = _build_settings(tmp_path, model_engine='sklearn',
                                      split_strategy='mixed', split_num=2)
        settings['model_params']['model_type'] = 'sklearn'
        settings['model_params']['split_strategy'] = 'galaxy_brain'
        feature_names = ['self.speed', 'other.speed']
        session_ids = [f'session_{i}' for i in range(N_SESSIONS)]
        input_pkl = str(_write_category_input_pickle(
            tmp_path / 'modeling_input.pkl', feature_names, session_ids))
        ranking_pkl = str(_write_category_ranking(
            tmp_path / 'univariate_combined.pkl', feature_names))

        ms_dir = tmp_path / 'model_selection'
        ms_dir.mkdir()
        with pytest.raises(ValueError, match="split_strategy"):
            vocal_category_model_selection(
                univariate_results_path=ranking_pkl,
                input_data_path=input_pkl,
                settings_path=_settings_json(tmp_path, settings),
                output_directory=str(ms_dir),
                use_top_rank_as_anchor=True,
                p_val=0.5,
            )

    def test_unknown_engine_raises(self, tmp_path):
        """
        A ``model_engine`` that is neither ``'sklearn'`` nor ``'pygam'`` makes the
        category selector raise ``ValueError`` during engine initialization.
        """

        settings, _ = _build_settings(tmp_path, model_engine='sklearn',
                                      split_strategy='mixed', split_num=2)
        settings['model_params']['model_engine'] = 'tea_leaves'
        feature_names = ['self.speed', 'other.speed']
        session_ids = [f'session_{i}' for i in range(N_SESSIONS)]
        input_pkl = str(_write_category_input_pickle(
            tmp_path / 'modeling_input.pkl', feature_names, session_ids))
        ranking_pkl = str(_write_category_ranking(
            tmp_path / 'univariate_combined.pkl', feature_names))

        ms_dir = tmp_path / 'model_selection'
        ms_dir.mkdir()
        with pytest.raises(ValueError, match="model_engine"):
            vocal_category_model_selection(
                univariate_results_path=ranking_pkl,
                input_data_path=input_pkl,
                settings_path=_settings_json(tmp_path, settings),
                output_directory=str(ms_dir),
                use_top_rank_as_anchor=True,
                p_val=0.5,
            )

    def test_no_significant_features_aborts(self, tmp_path):
        """
        With ``actual`` log-loss equal to ``null`` no feature clears the screen, so
        the category selector prints "No significant features found." and returns
        early without writing any step pickle.
        """

        settings, _ = _build_settings(tmp_path, model_engine='sklearn',
                                      split_strategy='mixed', split_num=2)
        settings['model_params']['model_type'] = 'sklearn'
        feature_names = ['self.speed', 'other.speed']
        session_ids = [f'session_{i}' for i in range(N_SESSIONS)]
        input_pkl = str(_write_category_input_pickle(
            tmp_path / 'modeling_input.pkl', feature_names, session_ids))
        # actual == null -> nothing survives.
        artifact: dict = {}
        for feature in feature_names:
            artifact[feature] = {
                'actual': {'ll': np.full(4, 0.69)},
                'null': {'ll': np.full(4, 0.69)},
            }
        artifact['_input_metadata'] = _category_input_metadata()
        ranking_pkl = tmp_path / 'univariate_combined.pkl'
        with ranking_pkl.open('wb') as fh:
            pickle.dump(artifact, fh)

        ms_dir = tmp_path / 'model_selection'
        ms_dir.mkdir()
        vocal_category_model_selection(
            univariate_results_path=str(ranking_pkl),
            input_data_path=input_pkl,
            settings_path=_settings_json(tmp_path, settings),
            output_directory=str(ms_dir),
            use_top_rank_as_anchor=True,
            p_val=0.01,
        )
        assert list(ms_dir.glob('*.pkl')) == []

    @pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
    @pytest.mark.filterwarnings("ignore::sklearn.exceptions.UndefinedMetricWarning")
    def test_per_fold_failure_accounting(self, tmp_path, monkeypatch):
        """
        Forcing ``LogisticRegressionCV.fit`` to raise makes every category
        selection fold fail; the per-fold ``except`` appends NaN placeholders
        (and a NaN-filled (2, 2) confusion matrix), the anchored Step-0 records no
        valid log-loss, the forward loop's no-finite-folds skip fires, the REJECT
        branch persists the (empty-candidate) step pickle, and the final-refit
        ``try/except`` swallows the forced failure so the run still returns.
        """

        class _BoomLRCV:
            def __init__(self, *a, **k):
                pass

            def fit(self, *a, **k):
                raise RuntimeError("forced LRCV failure")

        monkeypatch.setattr(ms, 'LogisticRegressionCV', _BoomLRCV)

        settings, _ = _build_settings(tmp_path, model_engine='sklearn',
                                      split_strategy='mixed', split_num=2)
        settings['model_params']['model_type'] = 'sklearn'
        feature_names = ['self.speed', 'other.speed']
        session_ids = [f'session_{i}' for i in range(N_SESSIONS)]
        input_pkl = str(_write_category_input_pickle(
            tmp_path / 'modeling_input.pkl', feature_names, session_ids))
        ranking_pkl = str(_write_category_ranking(
            tmp_path / 'univariate_combined.pkl', feature_names))

        ms_dir = tmp_path / 'model_selection'
        ms_dir.mkdir()
        vocal_category_model_selection(
            univariate_results_path=ranking_pkl,
            input_data_path=input_pkl,
            settings_path=_settings_json(tmp_path, settings),
            output_directory=str(ms_dir),
            use_top_rank_as_anchor=True,
            p_val=0.5,
        )
        # Every fold failed -> the forward loop rejects immediately and the
        # REJECT branch still persists the (empty-candidate) step pickle.
        assert len(list(ms_dir.glob('model_selection_category_*_step_*.pkl'))) >= 1

    @pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
    @pytest.mark.filterwarnings("ignore::sklearn.exceptions.UndefinedMetricWarning")
    def test_resume_from_existing_step_pickle(self, tmp_path):
        """
        A pre-existing category Step pickle in the output directory drives the
        resume branch: the selector loads the last step's ``current_features`` /
        ``baseline_score`` / ``candidates_summary`` and (because the saved
        candidate beats the baseline by more than its SE) promotes it and
        continues from the next step. The run completes and a further step pickle
        is produced.
        """

        settings, _ = _build_settings(tmp_path, model_engine='sklearn',
                                      split_strategy='mixed', split_num=2)
        settings['model_params']['model_type'] = 'sklearn'
        feature_names = ['self.speed', 'other.speed', 'self.neck_elevation']
        session_ids = [f'session_{i}' for i in range(N_SESSIONS)]
        input_pkl = str(_write_category_input_pickle(
            tmp_path / 'modeling_input.pkl', feature_names, session_ids))
        ranking_pkl = str(_write_category_ranking(
            tmp_path / 'univariate_combined.pkl', feature_names))

        ms_dir = tmp_path / 'model_selection'
        ms_dir.mkdir()
        prefix = f'model_selection_category_{CATEGORY_COLUMN}_unknown_unknown_step_'
        checkpoint = {
            'step_idx': 0,
            'current_features': [],
            'baseline_score': float(np.log(2)),
            'selected_feature': None,
            'candidates_summary': {
                'self.speed': {'mean_ll': 0.10, 'se_ll': 0.01},
            },
        }
        with (ms_dir / f'{prefix}0.pkl').open('wb') as fh:
            pickle.dump(checkpoint, fh)

        vocal_category_model_selection(
            univariate_results_path=ranking_pkl,
            input_data_path=input_pkl,
            settings_path=_settings_json(tmp_path, settings),
            output_directory=str(ms_dir),
            use_top_rank_as_anchor=False,
            p_val=0.5,
        )
        assert len(sorted(ms_dir.glob(f'{prefix}*.pkl'))) >= 2


_TINY_RIDGE = {'alphas': [0.1, 1.0, 10.0], 'cv': 2, 'fit_intercept': True}


def _build_params_input_pickle(save_path, feature_names, session_ids,
                               n_per_session=20, seed=0):
    """
    Description
    -----------
    Serializes a continuous-regression modeling-input pickle (feature ->
    ``{'X', 'y', 'groups'}``). All features share the same strictly-positive
    Gamma ``y`` and ``groups`` arrays; the first feature carries ``log(y)``
    injected into its history so it screens as significant. Embeds the reserved
    ``_input_metadata`` block.

    Parameters
    ----------
    save_path (pathlib.Path)
        Destination ``.pkl`` path.
    feature_names (list of str)
        Generic feature keys to populate.
    session_ids (list of str)
        Session identifiers; ``n_per_session`` bouts emitted under each.
    n_per_session (int)
        Number of synthetic bouts per session.
    seed (int)
        Base RNG seed.

    Returns
    -------
    save_path (pathlib.Path)
        The path the pickle was written to (absolute).
    """

    rng = np.random.default_rng(seed)
    groups = np.array([sess for sess in session_ids for _ in range(n_per_session)])
    n_total = len(groups)
    y = np.abs(rng.gamma(shape=2.0, scale=0.05, size=n_total)) + 0.01

    artifact: dict = {}
    for f_idx, feature in enumerate(feature_names):
        X = rng.standard_normal((n_total, HISTORY_FRAMES))
        if f_idx == 0:
            X = X + np.log(y)[:, None] * 0.6
        artifact[feature] = {'X': X.astype(float), 'y': y.copy(), 'groups': groups.copy()}
    artifact['_input_metadata'] = {'analysis_tag': 'bout_durations'}
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open('wb') as fh:
        pickle.dump(artifact, fh)
    return save_path


def _write_params_ranking(save_path, feature_names, significant=True, n_folds=4):
    """
    Description
    -----------
    Synthesizes the consolidated univariate ranking pickle the bout-parameter
    selector screens (an ``explained_deviance`` array per branch). When
    ``significant`` is True the ``actual`` mean deviance is high and positive so
    the feature clears the screen; when False it equals the ``null`` so nothing
    survives.

    Parameters
    ----------
    save_path (pathlib.Path)
        Destination ``.pkl`` path.
    feature_names (list of str)
        Feature keys to emit.
    significant (bool)
        Whether the synthesized ``actual`` deviance clears the null screen.
    n_folds (int)
        Per-fold entry count per branch.

    Returns
    -------
    save_path (pathlib.Path)
        The path the pickle was written to (absolute).
    """

    artifact: dict = {}
    for f_idx, feature in enumerate(feature_names):
        actual = np.full(n_folds, (0.40 - 0.01 * f_idx) if significant else 0.01, dtype=float)
        artifact[feature] = {
            'actual': {'explained_deviance': actual},
            'null': {'explained_deviance': np.full(n_folds, 0.01, dtype=float)},
        }
    artifact['_input_metadata'] = {'analysis_tag': 'bout_durations'}
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open('wb') as fh:
        pickle.dump(artifact, fh)
    return save_path


def _params_settings(tmp_path, **overrides):
    """
    Description
    -----------
    Builds bout-parameter settings: the shared trimmed settings plus the
    regression-specific knobs the selector reads (``model_type`` mirrored to the
    engine, ``model_target_variable``, and a tiny ``ridge_regression`` block).

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Per-test scratch directory.
    overrides (dict)
        Forwarded to ``build_modeling_settings`` (e.g. ``model_engine``,
        ``split_strategy``, ``split_num``).

    Returns
    -------
    settings (dict)
        The ready-to-use ``modeling_settings`` dictionary.
    """

    settings, _ = _build_settings(tmp_path, **overrides)
    settings['model_params']['model_target_variable'] = 'bout_durations'
    settings['model_params']['model_type'] = settings['model_params']['model_engine']
    settings['hyperparameters']['classical']['ridge_regression'] = dict(_TINY_RIDGE)
    return settings


class TestBoutParameterSelectionBranches:
    """Uncovered control-flow branches of ``bout_parameter_model_selection``."""

    def test_no_significant_features_aborts(self, tmp_path):
        """
        With ``actual`` explained-deviance at the ``null`` level no feature clears
        the screen, so the selector prints "No significant features found." and
        returns early without writing any step pickle.
        """

        settings = _params_settings(tmp_path, model_engine='sklearn',
                                    split_strategy='mixed', split_num=2)
        feature_names = ['self.speed', 'other.speed']
        session_ids = [f'session_{i}' for i in range(N_SESSIONS)]
        input_pkl = str(_build_params_input_pickle(
            tmp_path / 'modeling_input.pkl', feature_names, session_ids))
        ranking_pkl = str(_write_params_ranking(
            tmp_path / 'univariate_combined.pkl', feature_names, significant=False))

        ms_dir = tmp_path / 'model_selection'
        ms_dir.mkdir()
        bout_parameter_model_selection(
            univariate_results_path=ranking_pkl,
            input_data_path=input_pkl,
            settings_path=_settings_json(tmp_path, settings),
            output_directory=str(ms_dir),
            target_variable='bout_durations',
            use_top_rank_as_anchor=True,
            p_val=0.01,
        )
        assert list(ms_dir.glob('*.pkl')) == []

    def test_unknown_engine_raises(self, tmp_path):
        """
        A ``model_engine`` that is neither ``'sklearn'`` nor ``'pygam'`` makes the
        bout-parameter selector raise ``ValueError`` during engine initialization.
        """

        settings = _params_settings(tmp_path, model_engine='sklearn',
                                    split_strategy='mixed', split_num=2)
        settings['model_params']['model_engine'] = 'abacus'
        feature_names = ['self.speed', 'other.speed']
        session_ids = [f'session_{i}' for i in range(N_SESSIONS)]
        input_pkl = str(_build_params_input_pickle(
            tmp_path / 'modeling_input.pkl', feature_names, session_ids))
        ranking_pkl = str(_write_params_ranking(
            tmp_path / 'univariate_combined.pkl', feature_names))

        ms_dir = tmp_path / 'model_selection'
        ms_dir.mkdir()
        with pytest.raises(ValueError, match="model_engine"):
            bout_parameter_model_selection(
                univariate_results_path=ranking_pkl,
                input_data_path=input_pkl,
                settings_path=_settings_json(tmp_path, settings),
                output_directory=str(ms_dir),
                target_variable='bout_durations',
                use_top_rank_as_anchor=True,
                p_val=0.5,
            )

    def test_unknown_split_strategy_raises(self, tmp_path):
        """
        A ``split_strategy`` that is neither ``'session'`` nor ``'mixed'`` makes
        the bout-parameter selector raise ``ValueError`` while building folds.
        """

        settings = _params_settings(tmp_path, model_engine='sklearn',
                                    split_strategy='mixed', split_num=2)
        settings['model_params']['split_strategy'] = 'origami'
        feature_names = ['self.speed', 'other.speed']
        session_ids = [f'session_{i}' for i in range(N_SESSIONS)]
        input_pkl = str(_build_params_input_pickle(
            tmp_path / 'modeling_input.pkl', feature_names, session_ids))
        ranking_pkl = str(_write_params_ranking(
            tmp_path / 'univariate_combined.pkl', feature_names))

        ms_dir = tmp_path / 'model_selection'
        ms_dir.mkdir()
        with pytest.raises(ValueError, match="split_strategy"):
            bout_parameter_model_selection(
                univariate_results_path=ranking_pkl,
                input_data_path=input_pkl,
                settings_path=_settings_json(tmp_path, settings),
                output_directory=str(ms_dir),
                target_variable='bout_durations',
                use_top_rank_as_anchor=True,
                p_val=0.5,
            )

    def test_resume_stale_checkpoint_restarts_fresh(self, tmp_path):
        """
        A pre-existing bout-parameter step pickle whose ``candidates_summary`` has
        no finite-scored candidate is a stale/broken checkpoint: the resume logic
        discards it, clears ``existing_steps``, and the run re-establishes the
        baseline and re-fires the auto-anchor, completing and writing at least one
        step pickle under the canonical prefix.
        """

        settings = _params_settings(tmp_path, model_engine='sklearn',
                                    split_strategy='mixed', split_num=2)
        feature_names = ['self.speed', 'other.speed']
        session_ids = [f'session_{i}' for i in range(N_SESSIONS)]
        input_pkl = str(_build_params_input_pickle(
            tmp_path / 'modeling_input.pkl', feature_names, session_ids))
        ranking_pkl = str(_write_params_ranking(
            tmp_path / 'univariate_combined.pkl', feature_names))

        ms_dir = tmp_path / 'model_selection'
        ms_dir.mkdir()
        prefix = 'model_selection_bout_durations_unknown_mixed_step_'
        # Stale checkpoint: a candidate with a non-finite mean deviance.
        stale = {
            'step_idx': 0,
            'current_features': [],
            'baseline_score': 0.0,
            'selected_feature': None,
            'candidates_summary': {
                'self.speed': {'mean_explained_deviance': float('nan'),
                               'se_explained_deviance': 0.0},
            },
        }
        with (ms_dir / f'{prefix}0.pkl').open('wb') as fh:
            pickle.dump(stale, fh)

        bout_parameter_model_selection(
            univariate_results_path=ranking_pkl,
            input_data_path=input_pkl,
            settings_path=_settings_json(tmp_path, settings),
            output_directory=str(ms_dir),
            target_variable='bout_durations',
            use_top_rank_as_anchor=True,
            p_val=0.5,
        )
        assert len(list(ms_dir.glob(f'{prefix}*.pkl'))) >= 1


class TestMultinomialAndManifoldAborts:
    """The early no-significant-candidates aborts of the multinomial / manifold selectors."""

    def test_multinomial_no_significant_features_aborts(self, tmp_path):
        """
        When every univariate ``actual`` macro-AUC sits at the ``null`` level the
        multinomial selector's screen admits no candidate, so it prints "No
        significant features found." and returns before loading the input data —
        no step pickle is written.
        """

        settings, _ = _build_settings(tmp_path, model_engine='sklearn',
                                      split_strategy='mixed', split_num=2)
        feature_names = ['self.speed', 'other.speed']
        ranking: dict = {}
        for feature in feature_names:
            # actual == null (~chance) so nothing clears the screen.
            ranking[feature] = {
                'actual': {'folds': {'metrics': {'auc': [0.50, 0.50]}}},
                'null': {'folds': {'metrics': {'auc': [0.50, 0.50]}}},
            }
        ranking_pkl = tmp_path / 'univariate_combined.pkl'
        with ranking_pkl.open('wb') as fh:
            pickle.dump(ranking, fh)

        ms_dir = tmp_path / 'model_selection'
        ms_dir.mkdir()
        multinomial_vocal_category_model_selection(
            univariate_results_path=str(ranking_pkl),
            input_data_path=str(tmp_path / 'does_not_need_to_exist.pkl'),
            settings_path=_settings_json(tmp_path, settings),
            output_directory=str(ms_dir),
            use_top_rank_as_anchor=True,
            p_val=0.01,
        )
        assert list(ms_dir.glob('*.pkl')) == []

    def test_manifold_no_significant_features_aborts(self, tmp_path):
        """
        When every univariate ``actual`` ``r2_spatial`` fails to beat the
        ``null_model_free`` centroid baseline (Gate 1: mean <= 0) the manifold
        selector's Wilcoxon screen admits no candidate, so it prints its abort
        line and returns before loading the input data — no step pickle written.
        """

        settings, _ = _build_settings(tmp_path, model_engine='sklearn',
                                      split_strategy='mixed', split_num=2)
        feature_names = ['self.speed', 'other.speed']
        ranking: dict = {}
        for feature in feature_names:
            # mean actual r2_spatial <= 0 -> Gate (1) drops every feature.
            ranking[feature] = {
                'actual': {'folds': {'metrics': {'r2_spatial': [-0.10, -0.20, -0.15, -0.05]}}},
                'null_model_free': {'folds': {'metrics': {'r2_spatial': [0.0, 0.0, 0.0, 0.0]}}},
            }
        ranking_pkl = tmp_path / 'univariate_combined.pkl'
        with ranking_pkl.open('wb') as fh:
            pickle.dump(ranking, fh)

        ms_dir = tmp_path / 'model_selection'
        ms_dir.mkdir()
        continuous_vocal_manifold_model_selection(
            univariate_results_path=str(ranking_pkl),
            input_data_path=str(tmp_path / 'does_not_need_to_exist.pkl'),
            settings_path=_settings_json(tmp_path, settings),
            output_directory=str(ms_dir),
            use_top_rank_as_anchor=True,
            p_val=0.05,
        )
        assert list(ms_dir.glob('*.pkl')) == []


@pytest.mark.parametrize('selector', [
    bout_onset_model_selection,
    vocal_category_model_selection,
    bout_parameter_model_selection,
    multinomial_vocal_category_model_selection,
    continuous_vocal_manifold_model_selection,
])
def test_missing_settings_file_raises(tmp_path, selector):
    """
    Every selector re-raises ``FileNotFoundError`` when the explicit
    ``settings_path`` does not exist, covering the shared settings-load guard in
    each function. The keyword-only ``p_val`` of the category selector is passed
    via ``**kwargs`` so the single parametrization covers all five signatures.
    """

    missing = str(tmp_path / 'no_such_settings.json')
    kwargs = dict(
        univariate_results_path=str(tmp_path / 'u.pkl'),
        input_data_path=str(tmp_path / 'i.pkl'),
        output_directory=str(tmp_path / 'out'),
        settings_path=missing,
        p_val=0.05,
    )
    with pytest.raises(FileNotFoundError):
        selector(**kwargs)


def test_get_unrolled_X_validation_raises():
    """
    The unrolled-X helper raises ``ValueError`` on an empty feature list and on a
    history-frame count that disagrees with the data's column count, covering its
    two input-validation guards.
    """

    with pytest.raises(ValueError, match="empty"):
        get_unrolled_X_for_multivariate(feature_data_dict_list=[], history_frames=HISTORY_FRAMES)

    bad = [np.zeros((5, HISTORY_FRAMES + 1))]
    with pytest.raises(ValueError, match="Frame mismatch"):
        get_unrolled_X_for_multivariate(feature_data_dict_list=bad, history_frames=HISTORY_FRAMES)
