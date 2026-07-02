"""
@author: bartulem
End-to-end smoke tests for the USV vocal-onset modeling pipeline and the two
HPC dispatchers, driven entirely on tiny synthetic data built by ``_synth``.

These tests deliberately walk the *production* code paths rather than testing
isolated helpers:

* ``TestOnsetInputExtraction`` runs the real
  ``VocalOnsetModelingPipeline.extract_and_save_modeling_input_data`` against a
  synthetic session tree, lighting up ``load_input_files``,
  ``modeling_utils`` (session prep, role resolution, kinematic-column
  selection, vocal-column building, harmonization, cross-session z-scoring,
  ``run_predictor_audits``), and ``modeling_metadata``.

* ``TestUnivariateDispatcher`` drives ``dispatch_univariate_job`` for the
  ``onset`` analysis with the fast ``sklearn`` engine, exercising the basis
  construction, ``_run_model_for_feature_sklearn``, run-metadata building, and
  the per-feature pickle serialization.

* ``TestModelSelection`` runs the real ``vocal_onset_model_selection`` (the bulk
  of the 2k-statement ``model_selection.py``) on a strong-signal synthetic
  input pickle plus a freshly-computed univariate ranking, asserting that the
  forward-selection step pickles are produced with the expected structure.

* ``TestModelSelectionDispatcher`` drives ``dispatch_model_selection`` so the
  dispatcher's path validation and routing are covered too.

Warning policy
--------------
The project runs pytest with ``filterwarnings = ["error"]``. The modeling
import chain pulls ``optax`` -> a one-time JAX ``DeprecationWarning``, so all
top-level modeling imports below are wrapped in a ``warnings.catch_warnings``
block that ignores ``DeprecationWarning`` during import. At run time, ``pygam``
(under Python 3.13) emits a ``DeprecationWarning: Bitwise inversion '~' on
bool`` from inside its GAM fit, and ``astropy``'s Gaussian smoothing emits an
``AstropyUserWarning`` for the tiny synthetic traces; both are demoted with
narrow per-test ``@pytest.mark.filterwarnings`` markers. ``matplotlib`` is
forced onto the headless ``Agg`` backend because the dispatcher imports
``pyplot`` for basis-verification plotting.
"""

from __future__ import annotations

import argparse
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
    import usv_playpen.modeling.main_model_selection_dispatcher as ms_dispatcher
    import usv_playpen.modeling.main_univariate_dispatcher as univ_dispatcher
    from usv_playpen.modeling.model_selection import vocal_onset_model_selection
    from usv_playpen.modeling.modeling_vocal_onsets import VocalOnsetModelingPipeline


# Tiny-data geometry shared across tests. These are chosen so the
# vocal-onset extraction yields a meaningful number of positive (USV bout) and
# negative (No-USV) events while keeping every array small. ``HISTORY_FRAMES``
# is the derived ``floor(CAMERA_FPS * FILTER_HISTORY)`` and is the column count
# of every per-event window; the synthetic input pickle must match it.
CAMERA_FPS = 60.0
FILTER_HISTORY = 0.5
HISTORY_FRAMES = int(np.floor(CAMERA_FPS * FILTER_HISTORY))
USV_BOUT_TIME = 0.5
N_FRAMES = 7200       # 120 s sessions -> plenty of clean silent epochs
N_SESSIONS = 4
N_BOUTS = 15
USV_PER_BOUT = 3


def _build_settings(tmp_path: Path, **overrides):
    """
    Description
    -----------
    Builds the synthetic session tree, the session-list file, and the trimmed
    ``modeling_settings`` dict for a smoke run, all rooted under ``tmp_path``.

    The behavioral / vocal data is sized by the module-level tiny-data
    constants; ``usv_bout_time`` is shrunk to ``USV_BOUT_TIME`` so the
    clean-epoch negative sampler is not starved. Any keyword in ``overrides`` is
    forwarded to ``build_modeling_settings``.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        The pytest-provided per-test scratch directory; every artifact lives
        below it so nothing is ever written into the package tree.
    overrides (dict)
        Extra keyword arguments passed through to ``build_modeling_settings``
        (e.g. ``model_engine``, ``usv_predictor_type``, ``split_strategy``).

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
        n_frames=N_FRAMES,
        camera_fps=CAMERA_FPS,
        filter_history=FILTER_HISTORY,
        n_bouts=N_BOUTS,
        usv_per_bout=USV_PER_BOUT,
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
    settings['model_params']['usv_bout_time'] = USV_BOUT_TIME
    return settings, save_dir


def _run_univariate_and_consolidate(settings, input_pkl, feature_names, out_dir, monkeypatch):
    """
    Description
    -----------
    Runs the univariate ``onset`` dispatcher once per feature against a given
    input pickle, then consolidates the per-feature result pickles into a single
    ``{feature: results}`` dict written to ``out_dir / 'univariate_combined.pkl'``
    (the single-file form ``vocal_onset_model_selection`` expects).

    The dispatcher hard-codes loading the *package* settings JSON; this helper
    monkeypatches ``json.load`` inside the dispatcher module so the synthetic,
    shrunk settings are used instead — without ever touching the package file.

    Parameters
    ----------
    settings (dict)
        The synthetic settings the dispatcher should consume.
    input_pkl (str)
        Path to the modeling input pickle (feature -> session -> arrays).
    feature_names (list of str)
        Behavioral-feature keys to run (one dispatcher invocation each).
    out_dir (pathlib.Path)
        Directory for the per-feature and consolidated univariate pickles.
    monkeypatch (pytest.MonkeyPatch)
        Used to redirect the dispatcher's settings load.

    Returns
    -------
    combined_path (pathlib.Path)
        Path to the consolidated univariate-results pickle.
    per_feature_paths (list of pathlib.Path)
        The individual per-feature univariate pickles that were produced.
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(univ_dispatcher.json, 'load', lambda _f: settings)

    for feature_idx in range(len(feature_names)):
        univ_dispatcher.dispatch_univariate_job(
            argparse.Namespace(
                analysis_type='onset',
                feature_idx=feature_idx,
                input_data=input_pkl,
                output_dir=str(out_dir),
            )
        )

    per_feature_paths = sorted(out_dir.glob('univariate_*.pkl'))
    combined = {}
    for p in per_feature_paths:
        with p.open('rb') as fh:
            payload = pickle.load(fh)
        for key, value in payload.items():
            if not key.startswith('_'):
                combined[key] = value

    combined_path = out_dir / 'univariate_combined.pkl'
    with combined_path.open('wb') as fh:
        pickle.dump(combined, fh)
    return combined_path, per_feature_paths


class TestOnsetInputExtraction:
    """End-to-end extraction of the modeling input pickle from a synthetic tree."""

    @pytest.mark.filterwarnings("ignore:Bitwise inversion:DeprecationWarning")
    @pytest.mark.filterwarnings("ignore::astropy.utils.exceptions.AstropyUserWarning")
    def test_extraction_produces_aligned_input_pickle(self, tmp_path):
        """
        The real ``extract_and_save_modeling_input_data`` writes a
        ``modeling_*.pkl`` whose structure matches the documented contract:
        a nested ``{generic_feature: {session: {usv_feature_arr,
        no_usv_feature_arr}}}`` dict carrying a reserved ``_input_metadata``
        block, with every per-event window ``HISTORY_FRAMES`` wide and the
        per-session positive/negative counts identical across features
        (the intra-session alignment invariant).
        """

        settings, save_dir = _build_settings(tmp_path, model_engine='sklearn')
        pipeline = VocalOnsetModelingPipeline(modeling_settings_dict=settings)
        pipeline.extract_and_save_modeling_input_data()

        pkls = list(save_dir.glob('modeling_*.pkl'))
        assert len(pkls) == 1, f"expected exactly one input pickle, got {pkls}"

        with pkls[0].open('rb') as fh:
            artifact = pickle.load(fh)

        assert '_input_metadata' in artifact
        feature_keys = sorted(k for k in artifact if not k.startswith('_'))
        # Egocentric ['speed', 'neck_elevation'] expand to self.* and other.*.
        assert feature_keys == [
            'other.neck_elevation', 'other.speed',
            'self.neck_elevation', 'self.speed',
        ]

        anchor = feature_keys[0]
        sessions = sorted(artifact[anchor].keys())
        assert len(sessions) >= 1

        total_usv = 0
        total_no_usv = 0
        for sess in sessions:
            usv = artifact[anchor][sess]['usv_feature_arr']
            no_usv = artifact[anchor][sess]['no_usv_feature_arr']
            assert usv.shape[1] == HISTORY_FRAMES
            assert no_usv.shape[1] == HISTORY_FRAMES
            assert np.isfinite(usv).all() and np.isfinite(no_usv).all()
            total_usv += usv.shape[0]
            total_no_usv += no_usv.shape[0]

            # Intra-session alignment: every feature shares this session's
            # positive / negative event counts.
            for feat in feature_keys[1:]:
                assert artifact[feat][sess]['usv_feature_arr'].shape[0] == usv.shape[0]
                assert artifact[feat][sess]['no_usv_feature_arr'].shape[0] == no_usv.shape[0]

        assert total_usv > 0
        assert total_no_usv > 0

        md = artifact['_input_metadata']
        assert md['analysis_type'] == 'onset'
        assert md['analysis_tag'] == 'bout'
        assert sorted(md['feature_zoo_kept']) == feature_keys

    @pytest.mark.filterwarnings("ignore:Bitwise inversion:DeprecationWarning")
    @pytest.mark.filterwarnings("ignore::astropy.utils.exceptions.AstropyUserWarning")
    def test_extraction_with_partner_vocal_predictors(self, tmp_path):
        """
        With ``usv_predictor_type='categories_rate'`` the extraction additionally
        materializes a partner per-category vocal predictor column
        (``other.usv_cat_1``), exercising ``build_vocal_signal_columns`` and the
        vocal-column harmonization branch.
        """

        settings, save_dir = _build_settings(
            tmp_path, model_engine='sklearn', usv_predictor_type='categories_rate'
        )
        pipeline = VocalOnsetModelingPipeline(modeling_settings_dict=settings)
        pipeline.extract_and_save_modeling_input_data()

        with next(save_dir.glob('modeling_*.pkl')).open('rb') as fh:
            artifact = pickle.load(fh)

        feature_keys = {k for k in artifact if not k.startswith('_')}
        assert 'other.usv_cat_1' in feature_keys


class TestUnivariateDispatcher:
    """The univariate onset dispatcher on a synthetic strong-signal input pickle."""

    @pytest.mark.filterwarnings("ignore:Bitwise inversion:DeprecationWarning")
    def test_dispatch_onset_sklearn_writes_per_feature_pickles(self, tmp_path, monkeypatch):
        """
        ``dispatch_univariate_job`` (analysis 'onset', sklearn engine) writes one
        per-feature pickle per feature index, each carrying ``_run_metadata`` and
        ``_input_metadata`` blocks plus the actual/null results branches with the
        full scalar-metric key set (``ll``, ``auc``, ``brier``, ...) and the
        per-fold ``filter_shapes`` array.
        """

        settings, _ = _build_settings(tmp_path, model_engine='sklearn')
        feature_names = ['self.speed', 'other.speed', 'self.neck_elevation']
        session_ids = [f'session_{i}' for i in range(N_SESSIONS)]
        input_pkl = str(build_modeling_input_pickle(
            save_path=tmp_path / 'modeling_input.pkl',
            feature_names=feature_names,
            session_ids=session_ids,
            history_frames=HISTORY_FRAMES,
            n_usv=40,
            n_no_usv=60,
            input_metadata={'analysis_tag': 'bout'},
        ))

        out_dir = tmp_path / 'univariate'
        out_dir.mkdir()
        monkeypatch.setattr(univ_dispatcher.json, 'load', lambda _f: settings)

        for feature_idx in range(len(feature_names)):
            univ_dispatcher.dispatch_univariate_job(
                argparse.Namespace(
                    analysis_type='onset',
                    feature_idx=feature_idx,
                    input_data=input_pkl,
                    output_dir=str(out_dir),
                )
            )

        per_feature = sorted(out_dir.glob('univariate_*.pkl'))
        assert len(per_feature) == len(feature_names)

        with per_feature[0].open('rb') as fh:
            payload = pickle.load(fh)
        assert '_run_metadata' in payload
        assert '_input_metadata' in payload

        feat_key = next(k for k in payload if not k.startswith('_'))
        branch = payload[feat_key]['actual']
        for metric in ('ll', 'auc', 'score', 'brier', 'ece', 'mcc'):
            assert metric in branch
            assert branch[metric].shape == (settings['model_params']['split_num'],)
        assert branch['filter_shapes'].shape == (
            settings['model_params']['split_num'], HISTORY_FRAMES,
        )
        # The strong-signal pickle yields at least one finite (fitted) fold.
        assert np.isfinite(branch['ll']).any()


class TestModelSelection:
    """The real forward-stepwise ``vocal_onset_model_selection`` orchestrator."""

    @pytest.mark.filterwarnings("ignore:Bitwise inversion:DeprecationWarning")
    def test_bout_onset_selection_writes_step_pickles(self, tmp_path, monkeypatch):
        """
        Running ``vocal_onset_model_selection`` on a strong-signal synthetic input
        pickle (with a matching freshly-computed univariate ranking) performs the
        greedy forward search and writes the per-step result pickles. Each step
        pickle carries the ``current_features`` / ``baseline_score`` /
        ``candidates_summary`` structure the consolidator expects, and the
        accepted feature set grows monotonically across steps.
        """

        settings, _ = _build_settings(
            tmp_path, model_engine='sklearn', split_strategy='mixed', split_num=2,
        )
        feature_names = ['self.speed', 'other.speed', 'self.neck_elevation']
        session_ids = [f'session_{i}' for i in range(N_SESSIONS)]
        input_pkl = str(build_modeling_input_pickle(
            save_path=tmp_path / 'modeling_input.pkl',
            feature_names=feature_names,
            session_ids=session_ids,
            history_frames=HISTORY_FRAMES,
            n_usv=40,
            n_no_usv=60,
            input_metadata={'analysis_tag': 'bout'},
        ))

        combined_path, per_feature = _run_univariate_and_consolidate(
            settings=settings,
            input_pkl=input_pkl,
            feature_names=feature_names,
            out_dir=tmp_path / 'univariate',
            monkeypatch=monkeypatch,
        )
        assert len(per_feature) == len(feature_names)

        settings_json = tmp_path / 'settings.json'
        settings_json.write_text(json.dumps(settings))

        ms_dir = tmp_path / 'model_selection'
        ms_dir.mkdir()
        vocal_onset_model_selection(
            univariate_results_path=str(combined_path),
            input_data_path=input_pkl,
            settings_path=str(settings_json),
            output_directory=str(ms_dir),
            use_top_rank_as_anchor=True,
            p_val=0.05,
        )

        step_pkls = sorted(ms_dir.glob('model_selection_*_step_*.pkl'))
        assert len(step_pkls) >= 1, "expected at least one forward-selection step pickle"

        accepted_counts = []
        for p in step_pkls:
            with p.open('rb') as fh:
                step = pickle.load(fh)
            assert 'current_features' in step
            assert 'baseline_score' in step
            assert 'candidates_summary' in step
            accepted_counts.append(len(step['current_features']))

        # The anchored search starts with one feature and never shrinks.
        assert min(accepted_counts) >= 1
        assert accepted_counts == sorted(accepted_counts)

    @pytest.mark.filterwarnings("ignore:Bitwise inversion:DeprecationWarning")
    def test_bout_onset_selection_session_strategy(self, tmp_path, monkeypatch):
        """
        Re-runs ``vocal_onset_model_selection`` with ``split_strategy='session'``
        and *without* the auto-anchor, exercising the session-split CV-fold
        construction and the non-anchored Step-0 candidate sweep — code paths
        distinct from the 'mixed'/anchored run above. Asserts the orchestrator
        completes and emits at least the Step-0 pickle.
        """

        settings, _ = _build_settings(
            tmp_path, model_engine='sklearn', split_strategy='session', split_num=2,
            test_proportion=0.5,
        )
        feature_names = ['self.speed', 'other.speed', 'self.neck_elevation']
        # Six sessions so the session-level ShuffleSplit has train/test room.
        session_ids = [f'session_{i}' for i in range(6)]
        input_pkl = str(build_modeling_input_pickle(
            save_path=tmp_path / 'modeling_input.pkl',
            feature_names=feature_names,
            session_ids=session_ids,
            history_frames=HISTORY_FRAMES,
            n_usv=20,
            n_no_usv=30,
            input_metadata={'analysis_tag': 'bout'},
        ))

        combined_path, _ = _run_univariate_and_consolidate(
            settings=settings,
            input_pkl=input_pkl,
            feature_names=feature_names,
            out_dir=tmp_path / 'univariate',
            monkeypatch=monkeypatch,
        )

        settings_json = tmp_path / 'settings.json'
        settings_json.write_text(json.dumps(settings))

        ms_dir = tmp_path / 'model_selection'
        ms_dir.mkdir()
        vocal_onset_model_selection(
            univariate_results_path=str(combined_path),
            input_data_path=input_pkl,
            settings_path=str(settings_json),
            output_directory=str(ms_dir),
            use_top_rank_as_anchor=False,
            p_val=0.05,
        )

        step_pkls = list(ms_dir.glob('model_selection_*_step_*.pkl'))
        assert len(step_pkls) >= 1


class TestModelSelectionDispatcher:
    """The model-selection dispatcher: path validation + routing."""

    @pytest.mark.filterwarnings("ignore:Bitwise inversion:DeprecationWarning")
    def test_dispatch_onset_runs_through_validation_and_routing(self, tmp_path, monkeypatch):
        """
        ``dispatch_model_selection`` validates the three required paths and routes
        the 'onset' task into ``vocal_onset_model_selection``. The dispatcher
        auto-resolves the package settings JSON, so the real selection function
        is wrapped to inject the synthetic settings path; this still exercises
        the dispatcher's own validation and routing statements end-to-end without
        raising.
        """

        settings, _ = _build_settings(
            tmp_path, model_engine='sklearn', split_strategy='mixed', split_num=2,
        )
        feature_names = ['self.speed', 'other.speed', 'self.neck_elevation']
        session_ids = [f'session_{i}' for i in range(N_SESSIONS)]
        input_pkl = str(build_modeling_input_pickle(
            save_path=tmp_path / 'modeling_input.pkl',
            feature_names=feature_names,
            session_ids=session_ids,
            history_frames=HISTORY_FRAMES,
            n_usv=40,
            n_no_usv=60,
            input_metadata={'analysis_tag': 'bout'},
        ))

        combined_path, _ = _run_univariate_and_consolidate(
            settings=settings,
            input_pkl=input_pkl,
            feature_names=feature_names,
            out_dir=tmp_path / 'univariate',
            monkeypatch=monkeypatch,
        )

        settings_json = tmp_path / 'settings.json'
        settings_json.write_text(json.dumps(settings))

        # The dispatcher resolves and passes the package settings path; redirect
        # the selection call to the synthetic settings instead of editing src/.
        real_selection = vocal_onset_model_selection

        def _wrapped(**kwargs):
            kwargs['settings_path'] = str(settings_json)
            return real_selection(**kwargs)

        monkeypatch.setattr(ms_dispatcher, 'vocal_onset_model_selection', _wrapped)

        ms_dir = tmp_path / 'model_selection'
        ms_dir.mkdir()
        ms_dispatcher.dispatch_model_selection(
            argparse.Namespace(
                analysis_type='onset',
                univariate_path=str(combined_path),
                input_path=input_pkl,
                output_dir=str(ms_dir),
                anchor=True,
                pval=0.05,
                target_variable='bout_durations',
            )
        )

        # The dispatcher swallows downstream exceptions and prints a traceback;
        # reaching here without an uncaught exception means validation + routing
        # executed. On this synthetic data the wrapped selection runs the screen
        # to completion, so at least the Step-0 pickle is written.
        step_pkls = list(ms_dir.glob('model_selection_*_step_*.pkl'))
        assert len(step_pkls) >= 1
