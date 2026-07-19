"""
@author: bartulem
End-to-end smoke tests for the continuous bout-PARAMETER (regression) modeling
pipeline and its forward-stepwise model-selection path, driven entirely on tiny
synthetic data.

These tests walk the *production* code paths for the continuous-regression
branch of the modeling stack (the ``'params'`` analysis route) rather than
exercising isolated helpers:

* ``TestBoutParameterExtraction`` runs the real
  ``BoutParameterPipeline.extract_and_save_modeling_input_data`` against a
  synthetic session tree, lighting up ``load_behavioral_feature_data``,
  ``find_variable_length_bouts`` (variable-length bout detection +
  mixture-model-IVI thresholding + continuous vocal-signal generation),
  ``modeling_utils`` (session prep, role resolution, kinematic-column
  selection, vocal-column building, harmonization, cross-session z-scoring,
  ``run_predictor_audits``), and ``modeling_metadata``. It asserts the
  regression input pickle obeys the ``{generic_feature: {X, y, groups}}``
  contract plus the reserved ``_input_metadata`` block, and that the
  intra-session bout alignment invariant holds across every feature.

* ``TestUnivariateParamsDispatcher`` drives ``dispatch_univariate_job`` for the
  ``'params'`` analysis with both the fast ``sklearn`` (Gamma-GLM) engine
  and the ``pygam`` (GammaGAM, log-link) engine, exercising
  ``BoutParameterPipeline._run_model_for_feature_sklearn`` /
  ``_run_model_for_feature_pygam``, the basis construction, the per-fold
  actual + null (label-permuted) branches, run-metadata building, and the
  per-feature pickle serialization.

* ``TestBoutParameterModelSelection`` runs the real
  ``bout_parameter_model_selection`` forward-stepwise orchestrator on a
  strong-signal synthetic input pickle (plus a matching freshly-computed
  univariate ranking) across the ``'mixed'``/anchored and ``'session'``/
  non-anchored strategies and both engines, asserting that per-step pickles are
  produced with the expected ``current_features`` / ``baseline_score`` /
  ``candidates_summary`` structure and the final CV-based ``filter_shapes``
  block.

* ``TestBoutParameterSelectionDispatcher`` drives
  ``dispatch_model_selection`` for the ``'params'`` task so the dispatcher's
  path validation and target-variable routing into
  ``bout_parameter_model_selection`` are covered too.

The continuous-target regression input contract differs from the binary
vocal-onset contract: every feature stores a 2-D history matrix ``X`` of shape
``(n_bouts, history_frames)``, a 1-D continuous target ``y`` of shape
``(n_bouts,)``, and a 1-D session-group label array ``groups`` of shape
``(n_bouts,)``. The reusable ``_synth`` builders manufacture the on-disk session
tree and the trimmed settings dict; the regression input pickle (which the
shared ``build_modeling_input_pickle`` does *not* emit, since it produces the
binary ``usv_feature_arr`` / ``no_usv_feature_arr`` schema) is built by the
LOCAL ``_build_params_input_pickle`` helper defined in this file.

Warning policy
--------------
The project runs pytest with ``filterwarnings = ["error"]``. The modeling import
chain pulls ``optax`` -> a one-time JAX ``DeprecationWarning``, so all top-level
modeling imports below are wrapped in a ``warnings.catch_warnings`` block that
ignores ``DeprecationWarning`` during import. At run time the ``pygam`` GAM fit
(under Python 3.13) emits a ``DeprecationWarning: Bitwise inversion '~' on
bool`` from inside the IRLS solver, and the tiny synthetic folds provoke benign
numpy / scipy / sklearn convergence and constant-input warnings; all are demoted
with narrow per-test ``@pytest.mark.filterwarnings`` markers. ``matplotlib`` is
forced onto the headless ``Agg`` backend because the univariate dispatcher
imports ``pyplot`` for basis-verification plotting.
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
    import usv_playpen.modeling.modeling_vocal_bout_parameters as bout_params_module
    from usv_playpen.modeling.model_selection import bout_parameter_model_selection
    from usv_playpen.modeling.modeling_vocal_bout_parameters import BoutParameterPipeline


# Tiny-data geometry shared across tests. ``HISTORY_FRAMES`` is the derived
# ``floor(CAMERA_FPS * FILTER_HISTORY)`` and is the column count of every
# per-bout history window; the synthetic regression input pickle must match it.
CAMERA_FPS = 60.0
FILTER_HISTORY = 0.5
HISTORY_FRAMES = int(np.floor(CAMERA_FPS * FILTER_HISTORY))
USV_BOUT_TIME = 0.5
N_FRAMES = 7200       # 120 s sessions -> plenty of room for spaced bouts
N_SESSIONS = 4
N_BOUTS = 15
USV_PER_BOUT = 3

# Per-test sklearn hyperparameters: a tiny alpha grid and 2-fold internal CV keep
# the ``GammaRegressor`` search fast and well-conditioned on the small synthetic
# design matrices. (The settings block is still named ``ridge_regression``.)
_TINY_RIDGE = {'alphas': [0.1, 1.0, 10.0], 'cv': 2, 'fit_intercept': True}


def _build_params_input_pickle(
        save_path: Path,
        feature_names: list[str],
        session_ids: list[str],
        history_frames: int,
        signal_feature_idx=0,
        n_per_session: int = 25,
        signal_strength: float = 0.6,
        seed: int = 0,
) -> Path:
    """
    Description
    -----------
    Serializes a controlled continuous-regression "modeling input pickle"
    directly, bypassing the on-disk extraction. The artifact matches the schema
    the ``'params'`` univariate dispatcher and ``bout_parameter_model_selection``
    consume:

        {
          '<generic_feature>': {
              'X':      np.ndarray (n_bouts, history_frames),  # history window
              'y':      np.ndarray (n_bouts,),                 # continuous target
              'groups': np.ndarray (n_bouts,),                 # session-id labels
          }, ...,
          '_input_metadata': {'analysis_tag': 'bout_durations'}  # reserved block
        }

    All features share the SAME ``y`` and ``groups`` arrays (the intra-session
    alignment invariant the real extractor enforces — every behavioral predictor
    is sampled at the identical set of bouts). The target ``y`` is drawn from a
    strictly positive, right-skewed Gamma distribution (the biology the
    Gamma-GAM / Gamma-GLM engines assume), and the ``signal_feature_idx`` feature
    has ``log(y)`` injected into its history so at least one predictor carries
    above-chance, screenable structure on the continuous contrast.

    Parameters
    ----------
    save_path (pathlib.Path)
        Destination ``.pkl`` path (parent dirs created if absent).
    feature_names (list of str)
        Generic feature keys (e.g. ``['self.speed', 'other.speed']``).
    session_ids (list of str)
        Session identifiers; ``n_per_session`` bouts are emitted under each.
    history_frames (int)
        Number of temporal lags (columns) in every per-bout window ``X``.
    signal_feature_idx (int or iterable of int)
        Index (or indices) into ``feature_names`` whose history receives the
        injected ``log(y)`` signal so it survives null-control screening.
        Passing several indices yields multiple significant candidates so the
        multivariate forward-selection candidate-fitting loop is exercised.
    n_per_session (int)
        Number of synthetic bouts per session.
    signal_strength (float)
        Multiplier on the injected ``log(y)`` signal (larger -> stronger,
        easier-to-detect predictive relationship).
    seed (int)
        Base seed for the RNG (reproducible design matrices and targets).

    Returns
    -------
    save_path (pathlib.Path)
        The path the pickle was written to (absolute).
    """

    rng = np.random.default_rng(seed)
    groups = np.array([sess for sess in session_ids for _ in range(n_per_session)])
    n_total = len(groups)

    # Strictly positive, right-skewed continuous target (Gamma) — the shape the
    # log-link / log-transform engines are built for; offset keeps it off zero.
    y = np.abs(rng.gamma(shape=2.0, scale=0.05, size=n_total)) + 0.01

    signal_indices = (
        {signal_feature_idx} if isinstance(signal_feature_idx, int)
        else set(signal_feature_idx)
    )

    artifact: dict = {}
    for f_idx, feature in enumerate(feature_names):
        X = rng.standard_normal((n_total, history_frames))
        if f_idx in signal_indices:
            X = X + np.log(y)[:, None] * signal_strength
        artifact[feature] = {
            'X': X.astype(float),
            'y': y.copy(),
            'groups': groups.copy(),
        }

    artifact['_input_metadata'] = {'analysis_tag': 'bout_durations'}

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open('wb') as fh:
        pickle.dump(artifact, fh)
    return save_path


def _build_settings(tmp_path: Path, **overrides):
    """
    Description
    -----------
    Builds the synthetic session tree, the session-list file, and the trimmed
    ``modeling_settings`` dict for a continuous bout-parameter smoke run, all
    rooted under ``tmp_path``.

    On top of ``_synth.build_modeling_settings`` this helper applies the
    regression-specific knobs the bout-parameter path reads but the shared
    builder does not set: ``model_target_variable`` (the continuous target),
    ``model_type`` (the regression engine selector, mirrored to ``model_engine``
    so the univariate dispatcher and the model-selection orchestrator agree),
    ``usv_bout_time``, and a tiny ``ridge_regression`` hyperparameter block.

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
    settings['model_params']['model_target_variable'] = 'bout_durations'
    # `bout_parameter_model_selection` reads `model_params['model_type']`
    # strictly; the univariate dispatcher reads `model_params['model_engine']`.
    # Keep both aligned to the requested engine so the two halves of the
    # pipeline pick the same estimator family.
    engine = settings['model_params']['model_engine']
    settings['model_params']['model_type'] = engine
    settings['hyperparameters']['classical']['ridge_regression'] = dict(_TINY_RIDGE)
    return settings, save_dir


def _run_univariate_and_consolidate(settings, input_pkl, feature_names, out_dir, monkeypatch):
    """
    Description
    -----------
    Runs the univariate ``'params'`` dispatcher once per feature against a given
    regression input pickle, then consolidates the per-feature result pickles
    into a single ``{feature: results}`` dict written to
    ``out_dir / 'univariate_combined.pkl'`` (the single-file form
    ``bout_parameter_model_selection`` ranks against).

    The dispatcher hard-codes loading the *package* settings JSON; this helper
    monkeypatches ``json.load`` inside the dispatcher module so the synthetic,
    shrunk settings are used instead — without ever touching the package file.

    Parameters
    ----------
    settings (dict)
        The synthetic settings the dispatcher should consume.
    input_pkl (str)
        Path to the regression input pickle (feature -> {X, y, groups}).
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
                analysis_type='params',
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


class TestBoutParameterExtraction:
    """End-to-end extraction of the regression input pickle from a synthetic tree."""

    @pytest.mark.filterwarnings("ignore::astropy.utils.exceptions.AstropyUserWarning")
    def test_extraction_produces_aligned_regression_pickle(self, tmp_path):
        """
        The real ``extract_and_save_modeling_input_data`` writes a
        ``modeling_bout_durations_*.pkl`` whose structure matches the regression
        contract: a nested ``{generic_feature: {X, y, groups}}`` dict carrying a
        reserved ``_input_metadata`` block, with every per-bout window
        ``HISTORY_FRAMES`` wide, every ``y`` strictly positive and finite, and
        the per-bout count / session-group ordering identical across features
        (the intra-session alignment invariant).
        """

        settings, save_dir = _build_settings(
            tmp_path, model_engine='sklearn', usv_predictor_type='categories_rate'
        )
        pipeline = BoutParameterPipeline(modeling_settings_dict=settings)
        pipeline.extract_and_save_modeling_input_data()

        pkls = list(save_dir.glob('modeling_bout_durations_*.pkl'))
        assert len(pkls) == 1, f"expected exactly one regression input pickle, got {pkls}"

        with pkls[0].open('rb') as fh:
            artifact = pickle.load(fh)

        assert '_input_metadata' in artifact
        feature_keys = sorted(k for k in artifact if not k.startswith('_'))
        # Egocentric ['speed', 'neck_elevation'] expand to self.* and other.*;
        # the partner categories_rate predictor adds 'other.usv_cat_1'.
        assert 'self.speed' in feature_keys
        assert 'other.speed' in feature_keys
        assert 'self.neck_elevation' in feature_keys
        assert 'other.usv_cat_1' in feature_keys

        anchor = feature_keys[0]
        ref_y = artifact[anchor]['y']
        ref_groups = artifact[anchor]['groups']
        n_bouts = len(ref_y)
        assert n_bouts > 0

        assert artifact[anchor]['X'].shape == (n_bouts, HISTORY_FRAMES)
        assert ref_groups.shape == (n_bouts,)
        assert np.isfinite(ref_y).all()
        assert (ref_y > 0).all()
        assert np.isfinite(artifact[anchor]['X']).all()

        # Intra-session alignment: every feature shares the bout count and the
        # exact session-group ordering of the anchor feature.
        for feat in feature_keys[1:]:
            assert artifact[feat]['X'].shape == (n_bouts, HISTORY_FRAMES)
            assert len(artifact[feat]['y']) == n_bouts
            assert np.array_equal(artifact[feat]['groups'], ref_groups)
            # The continuous target is identical across features (one target
            # per bout, broadcast to every predictor).
            assert np.array_equal(artifact[feat]['y'], ref_y)

        md = artifact['_input_metadata']
        assert md['analysis_type'] == 'params'
        assert md['analysis_tag'] == 'bout_durations'
        assert md['analysis_specific']['target_variable'] == 'bout_durations'
        # n_events_per_session is backfilled from the anchor feature's groups.
        assert sum(v['bout_onsets'] for v in md['n_events_per_session'].values()) == n_bouts

    def test_extraction_refuses_to_write_misaligned_artifact(self, tmp_path, monkeypatch):
        """If intra-session alignment fails (a feature's session groups diverge
        from the reference feature), extract_and_save raises RuntimeError and
        writes NO pickle -- rather than persisting a known-misaligned regression
        artifact that pairs predictors with the wrong USV targets."""
        settings, save_dir = _build_settings(
            tmp_path, model_engine='sklearn', usv_predictor_type='categories_rate'
        )
        pipeline = BoutParameterPipeline(modeling_settings_dict=settings)
        # Force the intra-session group-alignment check (the only np.array_equal in
        # this code path) to fail, simulating an upstream extraction misalignment.
        monkeypatch.setattr(bout_params_module.np, "array_equal", lambda *a, **k: False)
        with pytest.raises(RuntimeError, match="alignment FAILED"):
            pipeline.extract_and_save_modeling_input_data()
        assert list(save_dir.glob('modeling_bout_durations_*.pkl')) == []


class TestUnivariateParamsDispatcher:
    """The univariate 'params' dispatcher on a synthetic strong-signal pickle."""

    @pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
    @pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
    @pytest.mark.filterwarnings("ignore:An input array is constant:scipy.stats.ConstantInputWarning")
    def test_dispatch_params_sklearn_writes_per_feature_pickles(self, tmp_path, monkeypatch):
        """
        ``dispatch_univariate_job`` (analysis 'params', sklearn engine) writes one
        per-feature pickle per feature index, each carrying ``_run_metadata`` and
        ``_input_metadata`` blocks plus the actual/null results branches with the
        full regression-metric key set (``explained_deviance``, ``spearman_r``,
        ``pearson_r``, ``msle``, ``mae``, ``rmse``, ``residual_deviance``, ...).
        The signal-bearing feature yields at least one finite (fitted) fold.
        """

        settings, _ = _build_settings(tmp_path, model_engine='sklearn', split_num=2)
        feature_names = ['self.speed', 'other.speed', 'self.neck_elevation']
        session_ids = [f'session_{i}' for i in range(N_SESSIONS)]
        input_pkl = str(_build_params_input_pickle(
            save_path=tmp_path / 'params_input.pkl',
            feature_names=feature_names,
            session_ids=session_ids,
            history_frames=HISTORY_FRAMES,
            signal_feature_idx=0,
        ))

        out_dir = tmp_path / 'univariate'
        out_dir.mkdir()
        monkeypatch.setattr(univ_dispatcher.json, 'load', lambda _f: settings)

        for feature_idx in range(len(feature_names)):
            univ_dispatcher.dispatch_univariate_job(
                argparse.Namespace(
                    analysis_type='params',
                    feature_idx=feature_idx,
                    input_data=input_pkl,
                    output_dir=str(out_dir),
                )
            )

        per_feature = sorted(out_dir.glob('univariate_*.pkl'))
        assert len(per_feature) == len(feature_names)

        # Index 0 is the signal-bearing 'self.speed' feature.
        with per_feature[0].open('rb') as fh:
            payload = pickle.load(fh)
        assert '_run_metadata' in payload
        assert '_input_metadata' in payload

        feat_key = next(k for k in payload if not k.startswith('_'))
        for branch_name in ('actual', 'null'):
            branch = payload[feat_key][branch_name]
            for metric in (
                'explained_deviance', 'residual_deviance', 'spearman_r',
                'pearson_r', 'msle', 'mae', 'rmse',
            ):
                assert metric in branch
                assert branch[metric].shape == (settings['model_params']['split_num'],)
        # The strong-signal feature yields at least one finite (fitted) fold.
        assert np.isfinite(payload[feat_key]['actual']['explained_deviance']).any()

    @pytest.mark.filterwarnings("ignore:Bitwise inversion:DeprecationWarning")
    @pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
    @pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
    @pytest.mark.filterwarnings("ignore:An input array is constant:scipy.stats.ConstantInputWarning")
    def test_dispatch_params_pygam_writes_per_feature_pickles(self, tmp_path, monkeypatch):
        """
        ``dispatch_univariate_job`` (analysis 'params', pygam engine) routes into
        ``BoutParameterPipeline._run_model_for_feature_pygam``, fitting a
        Gamma-distributed GAM with a log-link per fold. The per-feature pickle
        carries the same regression-metric key set plus the pyGAM-only optimizer
        diagnostics (``n_iter``, ``converged``, ``fit_time``) and the per-fold
        ``filter_shapes`` (the actual branch reconstructs them; the null branch
        does not emit shapes by design).
        """

        # The 'session' split strategy routes the per-feature GAM fit through
        # the pipeline's own ``create_data_splits`` StratifiedGroupKFold branch
        # (distinct from the 'mixed' StratifiedShuffleSplit branch).
        settings, _ = _build_settings(
            tmp_path, model_engine='pygam', split_strategy='session', split_num=2,
            test_proportion=0.5,
        )
        feature_names = ['self.speed', 'other.speed']
        session_ids = [f'session_{i}' for i in range(6)]
        input_pkl = str(_build_params_input_pickle(
            save_path=tmp_path / 'params_input.pkl',
            feature_names=feature_names,
            session_ids=session_ids,
            history_frames=HISTORY_FRAMES,
            signal_feature_idx=0,
            n_per_session=20,
        ))

        out_dir = tmp_path / 'univariate'
        out_dir.mkdir()
        monkeypatch.setattr(univ_dispatcher.json, 'load', lambda _f: settings)

        for feature_idx in range(len(feature_names)):
            univ_dispatcher.dispatch_univariate_job(
                argparse.Namespace(
                    analysis_type='params',
                    feature_idx=feature_idx,
                    input_data=input_pkl,
                    output_dir=str(out_dir),
                )
            )

        per_feature = sorted(out_dir.glob('univariate_*.pkl'))
        assert len(per_feature) == len(feature_names)

        with per_feature[0].open('rb') as fh:
            payload = pickle.load(fh)
        feat_key = next(k for k in payload if not k.startswith('_'))
        actual = payload[feat_key]['actual']
        for metric in (
            'explained_deviance', 'spearman_r', 'msle', 'mae', 'rmse',
            'n_iter', 'converged', 'fit_time',
        ):
            assert metric in actual
            assert actual[metric].shape == (settings['model_params']['split_num'],)
        # The GAM actual branch stacks one filter-shape vector per fold.
        assert actual['filter_shapes'].shape == (
            settings['model_params']['split_num'], HISTORY_FRAMES,
        )
        assert np.isfinite(actual['explained_deviance']).any()
        # The null (shuffled-control) branch never reconstructs filter shapes,
        # so its arrayified ``filter_shapes`` stays empty by design.
        null = payload[feat_key]['null']
        assert null['filter_shapes'].size == 0


class TestBoutParameterModelSelection:
    """The real forward-stepwise ``bout_parameter_model_selection`` orchestrator."""

    @pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
    @pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
    @pytest.mark.filterwarnings("ignore:An input array is constant:scipy.stats.ConstantInputWarning")
    def test_selection_mixed_anchored_writes_step_pickles(self, tmp_path, monkeypatch):
        """
        Running ``bout_parameter_model_selection`` (sklearn engine, 'mixed'
        strategy, auto-anchored) on a strong-signal synthetic input pickle (with
        a matching freshly-computed univariate ranking) screens candidates,
        seeds Step-0 from the top-ranked feature, and runs the greedy forward
        search. Each step pickle carries the ``current_features`` /
        ``baseline_score`` / ``candidates_summary`` structure the consolidator
        expects, the accepted feature set never shrinks across steps, and the
        terminal step records the CV-based ``filter_shapes`` block.
        """

        settings, _ = _build_settings(
            tmp_path, model_engine='sklearn', split_strategy='mixed', split_num=2,
        )
        feature_names = ['self.speed', 'other.speed', 'self.neck_elevation']
        session_ids = [f'session_{i}' for i in range(N_SESSIONS)]
        input_pkl = str(_build_params_input_pickle(
            save_path=tmp_path / 'params_input.pkl',
            feature_names=feature_names,
            session_ids=session_ids,
            history_frames=HISTORY_FRAMES,
            # Two signal-bearing features so the multivariate forward-selection
            # candidate-fitting loop (basis-stacked Gamma GLM) is exercised.
            signal_feature_idx=(0, 1),
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
        bout_parameter_model_selection(
            univariate_results_path=str(combined_path),
            input_data_path=input_pkl,
            settings_path=str(settings_json),
            output_directory=str(ms_dir),
            target_variable='bout_durations',
            use_top_rank_as_anchor=True,
            p_val=0.5,
        )

        step_pkls = sorted(ms_dir.glob('model_selection_bout_durations_*_step_*.pkl'))
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

        # The terminal step pickle is augmented in-place with the CV-based
        # final-model filter shapes for visualization.
        with step_pkls[-1].open('rb') as fh:
            terminal = pickle.load(fh)
        assert 'final_model_features' in terminal
        assert 'filter_shapes' in terminal
        assert len(terminal['filter_shapes']) >= 1

    @pytest.mark.filterwarnings("ignore:Bitwise inversion:DeprecationWarning")
    @pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
    @pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
    @pytest.mark.filterwarnings("ignore:An input array is constant:scipy.stats.ConstantInputWarning")
    def test_selection_session_pygam_anchored(self, tmp_path, monkeypatch):
        """
        Re-runs ``bout_parameter_model_selection`` with the pyGAM engine, the
        'session' split strategy, and the auto-anchor enabled — exercising the
        ``StratifiedGroupKFold`` session-split construction, the pyGAM anchor
        fit (tensor-product GammaGAM, eta-scale aggregation), and the
        multivariate forward-selection candidate sweep, code paths distinct from
        the sklearn/'mixed' run above. Asserts the orchestrator completes and
        emits at least the Step-0 pickle.
        """

        settings, _ = _build_settings(
            tmp_path, model_engine='pygam', split_strategy='session', split_num=2,
            test_proportion=0.5,
        )
        feature_names = ['self.speed', 'other.speed']
        # Six sessions so the session-level StratifiedGroupKFold has room.
        session_ids = [f'session_{i}' for i in range(6)]
        input_pkl = str(_build_params_input_pickle(
            save_path=tmp_path / 'params_input.pkl',
            feature_names=feature_names,
            session_ids=session_ids,
            history_frames=HISTORY_FRAMES,
            # Both features carry signal so the multivariate tensor-product
            # GammaGAM forward-selection candidate fit is exercised.
            signal_feature_idx=(0, 1),
            n_per_session=20,
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
        bout_parameter_model_selection(
            univariate_results_path=str(combined_path),
            input_data_path=input_pkl,
            settings_path=str(settings_json),
            output_directory=str(ms_dir),
            target_variable='bout_durations',
            use_top_rank_as_anchor=True,
            p_val=0.5,
        )

        step_pkls = list(ms_dir.glob('model_selection_bout_durations_*_step_*.pkl'))
        assert len(step_pkls) >= 1


class TestBoutParameterSelectionDispatcher:
    """The model-selection dispatcher: 'params' path validation + routing."""

    @pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
    @pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
    @pytest.mark.filterwarnings("ignore:An input array is constant:scipy.stats.ConstantInputWarning")
    def test_dispatch_params_runs_through_validation_and_routing(self, tmp_path, monkeypatch):
        """
        ``dispatch_model_selection`` validates the three required paths and routes
        the 'params' task (with its ``target_variable``) into
        ``bout_parameter_model_selection``. The dispatcher auto-resolves the
        package settings JSON, so the real selection function is wrapped to inject
        the synthetic settings path; this still exercises the dispatcher's own
        validation and routing statements end-to-end without raising. The sklearn
        engine is run with the ``bspline`` basis so the alternate
        basis-construction branch (distinct from the default ``raised_cosine``)
        is exercised inside the selection orchestrator.
        """

        settings, _ = _build_settings(
            tmp_path, model_engine='sklearn', split_strategy='mixed', split_num=2,
        )
        settings['model_params']['model_basis_function'] = 'bspline'
        feature_names = ['self.speed', 'other.speed', 'self.neck_elevation']
        session_ids = [f'session_{i}' for i in range(N_SESSIONS)]
        input_pkl = str(_build_params_input_pickle(
            save_path=tmp_path / 'params_input.pkl',
            feature_names=feature_names,
            session_ids=session_ids,
            history_frames=HISTORY_FRAMES,
            signal_feature_idx=(0, 1),
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
        real_selection = bout_parameter_model_selection

        def _wrapped(**kwargs):
            kwargs['settings_path'] = str(settings_json)
            return real_selection(**kwargs)

        monkeypatch.setattr(ms_dispatcher, 'bout_parameter_model_selection', _wrapped)

        ms_dir = tmp_path / 'model_selection'
        ms_dir.mkdir()
        ms_dispatcher.dispatch_model_selection(
            argparse.Namespace(
                analysis_type='params',
                univariate_path=str(combined_path),
                input_path=input_pkl,
                output_dir=str(ms_dir),
                anchor=True,
                pval=0.5,
                target_variable='bout_durations',
            )
        )

        # The dispatcher swallows downstream exceptions and prints a traceback;
        # reaching here without an uncaught exception means validation + routing
        # executed. On this synthetic data the wrapped selection runs the screen
        # to completion, so at least the Step-0 pickle is written.
        step_pkls = list(ms_dir.glob('model_selection_bout_durations_*_step_*.pkl'))
        assert len(step_pkls) >= 1


def _minimal_bout_pipeline(
        split_strategy: str = 'mixed',
        split_num: int = 2,
        test_proportion: float = 0.3,
        random_seed=0,
        attach_feature_boundaries: bool = False,
):
    """
    Description
    -----------
    Builds a ``BoutParameterPipeline`` from a hand-rolled minimal settings dict —
    enough keys for ``create_data_splits`` and the two per-feature fitters
    (``_run_model_for_feature_pygam`` / ``_run_model_for_feature_sklearn``) to
    run, without manufacturing an on-disk session tree. The pipeline's
    ``__init__`` (inherited from ``VocalOnsetModelingPipeline``) derives
    ``history_frames`` from ``io.camera_sampling_rate`` and
    ``model_params.filter_history`` and reads ``feature_boundaries`` only when
    the key is present — so the ``attach_feature_boundaries`` toggle exercises
    the present / absent branches downstream.

    Parameters
    ----------
    split_strategy (str)
        ``'mixed'`` or ``'session'`` (or a deliberately-unknown value to drive
        the ``create_data_splits`` no-match ``raise`` branch).
    split_num (int)
        Number of Monte-Carlo split iterations the generator yields.
    test_proportion (float)
        Per-split test fraction; pass an out-of-range value to drive the
        guard-clause ``ValueError``.
    random_seed (int or None)
        Base RNG seed; ``None`` exercises the ``base_seed = 42`` fallback.
    attach_feature_boundaries (bool)
        When ``True`` a ``feature_boundaries`` block is injected so the
        ``hasattr(self, 'feature_boundaries')`` branch is taken.

    Returns
    -------
    pipeline (BoutParameterPipeline)
        A ready-to-use pipeline instance backed by the minimal settings dict.
    """

    settings = {
        'io': {'camera_sampling_rate': CAMERA_FPS, 'csv_separator': ','},
        'model_params': {
            'filter_history': FILTER_HISTORY,
            'split_strategy': split_strategy,
            'split_num': split_num,
            'test_proportion': test_proportion,
            'random_seed': random_seed,
            'model_engine': 'sklearn',
        },
        'hyperparameters': {
            'classical': {
                'pygam': {
                    'n_splines_value': 4, 'n_splines_time': 4,
                    'lam_penalty': 0.6, 'tol_val': 1e-4, 'max_iterations': 20,
                },
                'ridge_regression': dict(_TINY_RIDGE),
            }
        },
    }
    if attach_feature_boundaries:
        settings['feature_boundaries'] = {}
    return BoutParameterPipeline(modeling_settings_dict=settings)


def _bout_feature_data(history_frames: int, n_per_session: int = 25, n_sessions: int = 4, seed: int = 0):
    """
    Description
    -----------
    Manufactures a single-feature ``{'X', 'y', 'groups'}`` triple in the exact
    shape ``create_data_splits`` and the per-feature fitters consume: ``X`` of
    shape ``(n_bouts, history_frames)``, a strictly-positive continuous target
    ``y`` of shape ``(n_bouts,)`` and a session-group label array ``groups``.

    Parameters
    ----------
    history_frames (int)
        Number of temporal lags (columns) per window.
    n_per_session (int)
        Bouts emitted under each synthetic session id.
    n_sessions (int)
        Number of distinct session groups.
    seed (int)
        RNG seed (reproducible design matrices and targets).

    Returns
    -------
    feature_data (dict)
        The ``{'X', 'y', 'groups'}`` triple.
    """

    rng = np.random.default_rng(seed)
    session_ids = [f'session_{i}' for i in range(n_sessions)]
    groups = np.array([s for s in session_ids for _ in range(n_per_session)])
    n_total = len(groups)
    y = np.abs(rng.gamma(shape=2.0, scale=0.05, size=n_total)) + 0.01
    X = rng.standard_normal((n_total, history_frames)) + np.log(y)[:, None] * 0.6
    return {'X': X.astype(float), 'y': y, 'groups': groups}


class TestCreateDataSplitsBranches:
    """The ``create_data_splits`` guard / strategy / binning branches."""

    @pytest.mark.parametrize("split_strategy", ["mixed", "session"])
    def test_degenerate_target_disables_stratification_instead_of_raising(self, split_strategy):
        """A heavily-skewed target collapses the percentile bins into a singleton
        stratum; instead of the stratified splitter raising ('least populated
        class has only 1 member' / 'n_splits greater than members in each class')
        and aborting the feature, stratification is disabled (single bin) with a
        warning and the split still yields usable folds."""

        pipeline = _minimal_bout_pipeline(
            split_strategy=split_strategy, split_num=2, test_proportion=0.25,
        )
        n = 20
        rng = np.random.default_rng(0)
        # One low outlier among 19 identical values: the median percentile edge
        # isolates that single sample into its own stratum (count 1), which is too
        # small for StratifiedShuffleSplit (>=2) and StratifiedGroupKFold (>=n_folds).
        y = np.concatenate([[0.0], np.ones(n - 1)])
        feature_data = {
            'X': rng.standard_normal((n, pipeline.history_frames)),
            'y': y,
            'groups': np.repeat(np.arange(4), n // 4),
        }
        with pytest.warns(UserWarning, match="Stratification disabled"):
            splits = list(pipeline.create_data_splits(feature_data))
        assert len(splits) >= 1
        for x_train, _y_train, x_test, _y_test in splits:
            assert x_train.shape[0] > 0
            assert x_test.shape[0] > 0

    def test_test_proportion_out_of_range_raises(self):
        """
        A ``test_proportion`` outside the open interval ``(0, 1)`` trips the
        guard-clause ``ValueError`` before any splitter is constructed.
        """

        pipeline = _minimal_bout_pipeline(test_proportion=1.5)
        feature_data = _bout_feature_data(pipeline.history_frames)
        with pytest.raises(ValueError, match="test_proportion must be"):
            list(pipeline.create_data_splits(feature_data))

    def test_unknown_strategy_raises(self):
        """
        An unrecognized ``split_strategy`` falls through both the 'mixed' and
        'session' branches into the terminal ``ValueError``.
        """

        pipeline = _minimal_bout_pipeline(split_strategy='nonsense')
        feature_data = _bout_feature_data(pipeline.history_frames)
        with pytest.raises(ValueError, match="Unknown split strategy"):
            list(pipeline.create_data_splits(feature_data))

    @pytest.mark.filterwarnings("ignore::sklearn.exceptions.UndefinedMetricWarning")
    def test_none_seed_falls_back_on_session_strategy(self):
        """
        A ``None`` ``random_seed`` exercises the ``base_seed = 42`` fallback while
        the 'session' strategy routes through the per-iteration
        ``StratifiedGroupKFold`` construction and yields at least one
        train/test fold on a multi-session input.
        """

        pipeline = _minimal_bout_pipeline(
            split_strategy='session', random_seed=None, split_num=2,
            test_proportion=0.5,
        )
        feature_data = _bout_feature_data(
            pipeline.history_frames, n_per_session=20, n_sessions=6,
        )
        splits = list(pipeline.create_data_splits(feature_data))
        assert len(splits) >= 1
        X_tr, y_tr, _X_te, y_te = splits[0]
        assert X_tr.shape[1] == pipeline.history_frames
        assert len(y_tr) > 0 and len(y_te) > 0

    def test_binning_except_zeroes_y_binned(self, monkeypatch):
        """
        Forcing ``np.percentile`` to raise drives the quantile-binning ``try`` /
        broad ``except`` that collapses ``y_binned`` to an all-zeros vector; with
        a single stratification bin the 'mixed' ``StratifiedShuffleSplit`` still
        produces valid folds, proving the fallback keeps the generator alive.
        """

        pipeline = _minimal_bout_pipeline(split_strategy='mixed', split_num=2)
        feature_data = _bout_feature_data(pipeline.history_frames)

        real_percentile = bout_params_module.np.percentile

        def _boom(*a, **k):
            raise RuntimeError("forced percentile failure")

        monkeypatch.setattr(bout_params_module.np, 'percentile', _boom)
        try:
            splits = list(pipeline.create_data_splits(feature_data))
        finally:
            monkeypatch.setattr(bout_params_module.np, 'percentile', real_percentile)
        assert len(splits) == 2


class TestPerFeatureFitFailureHandlers:
    """Per-fold fit / metric ``except`` handlers in the two regression fitters."""

    @pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
    @pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
    @pytest.mark.filterwarnings("ignore:An input array is constant:scipy.stats.ConstantInputWarning")
    def test_sklearn_fit_failure_fills_nan(self, monkeypatch):
        """
        Monkeypatching ``GridSearchCV`` so every ``.fit`` raises drives BOTH the
        actual-branch and null-branch fit ``except`` handlers in
        ``_run_model_for_feature_sklearn``; each per-fold list is filled with a
        NaN placeholder (and a NaN-vector ``filter_shapes`` row for the actual
        branch) so the returned metric arrays are the right length and entirely
        non-finite.
        """

        pipeline = _minimal_bout_pipeline(split_strategy='mixed', split_num=2)
        feature_data = _bout_feature_data(pipeline.history_frames)
        basis = np.eye(pipeline.history_frames, 3, dtype=float)

        class _ExplodingSearch:
            def __init__(self, *a, **k):
                pass

            def fit(self, *a, **k):
                raise RuntimeError("forced sklearn fit failure")

        monkeypatch.setattr(bout_params_module, 'GridSearchCV', _ExplodingSearch)

        _fn, res = pipeline._run_model_for_feature_sklearn('self.speed', feature_data, basis)
        assert not np.isfinite(res['actual']['explained_deviance']).any()
        assert not np.isfinite(res['null']['explained_deviance']).any()
        assert res['actual']['filter_shapes'].shape == (2, pipeline.history_frames)
        assert np.all([not c for c in res['actual']['converged']])

    @pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
    @pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
    @pytest.mark.filterwarnings("ignore:An input array is constant:scipy.stats.ConstantInputWarning")
    def test_sklearn_metric_failure_sets_nan_metrics(self, monkeypatch):
        """
        Monkeypatching ``mean_gamma_deviance`` so it always raises lets the
        Gamma-GLM fits succeed but drives the inner metric-computation ``except``
        handlers (actual + null) that set ``d2`` / ``res_dev`` to NaN while the
        remaining per-fold metrics are still appended. The fold therefore lands
        in the ``DATA-PRESENT`` path (converged ``True``) yet reports a NaN
        explained deviance.
        """

        pipeline = _minimal_bout_pipeline(split_strategy='mixed', split_num=2)
        feature_data = _bout_feature_data(pipeline.history_frames)
        basis = np.eye(pipeline.history_frames, 3, dtype=float)

        def _boom(*a, **k):
            raise ValueError("forced gamma-deviance failure")

        monkeypatch.setattr(bout_params_module, 'mean_gamma_deviance', _boom)

        _fn, res = pipeline._run_model_for_feature_sklearn('self.speed', feature_data, basis)
        # The metric except fired -> explained_deviance is NaN, but the fit
        # itself succeeded (GammaRegressor converged within max_iter) so the fold is
        # marked converged.
        assert not np.isfinite(res['actual']['explained_deviance']).any()
        assert np.all([bool(c) for c in res['actual']['converged']])

    @pytest.mark.filterwarnings("ignore:Bitwise inversion:DeprecationWarning")
    @pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
    @pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
    @pytest.mark.filterwarnings("ignore:An input array is constant:scipy.stats.ConstantInputWarning")
    def test_pygam_fit_failure_fills_nan(self, monkeypatch):
        """
        Monkeypatching ``GAM`` so every constructed model's ``.fit`` raises drives
        BOTH the actual-branch and null-branch fit ``except`` handlers in
        ``_run_model_for_feature_pygam`` — the actual branch fills a NaN-vector
        ``filter_shapes`` row, the null branch leaves ``filter_shapes`` empty by
        design, and every other per-fold key is a NaN placeholder.
        """

        pipeline = _minimal_bout_pipeline(split_strategy='mixed', split_num=2)
        feature_data = _bout_feature_data(pipeline.history_frames)

        class _ExplodingGAM:
            def __init__(self, *a, **k):
                pass

            def fit(self, *a, **k):
                raise RuntimeError("forced GAM fit failure")

        monkeypatch.setattr(bout_params_module, 'GAM', _ExplodingGAM)

        _fn, res = pipeline._run_model_for_feature_pygam('self.speed', feature_data, None)
        assert not np.isfinite(res['actual']['explained_deviance']).any()
        assert not np.isfinite(res['null']['explained_deviance']).any()
        # Actual fit-failure path stacks a NaN filter-shape vector per fold.
        assert res['actual']['filter_shapes'].shape == (2, pipeline.history_frames)
        # Null branch never emits filter shapes.
        assert res['null']['filter_shapes'].size == 0

    @pytest.mark.filterwarnings("ignore:Bitwise inversion:DeprecationWarning")
    @pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
    @pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
    @pytest.mark.filterwarnings("ignore:An input array is constant:scipy.stats.ConstantInputWarning")
    def test_pygam_metric_failure_sets_nan_metrics(self, monkeypatch):
        """
        With the GAM fits succeeding, monkeypatching ``mean_gamma_deviance`` to
        raise drives the inner metric-computation ``except`` handlers (actual +
        null) in ``_run_model_for_feature_pygam`` that set ``d2`` / ``res_dev``
        to NaN while the surrounding spearman / pearson / msle / mae / rmse
        metrics are still appended and the fold is recorded as fitted.
        """

        pipeline = _minimal_bout_pipeline(split_strategy='mixed', split_num=2)
        feature_data = _bout_feature_data(pipeline.history_frames)

        def _boom(*a, **k):
            raise ValueError("forced gamma-deviance failure")

        monkeypatch.setattr(bout_params_module, 'mean_gamma_deviance', _boom)

        _fn, res = pipeline._run_model_for_feature_pygam('self.speed', feature_data, None)
        assert not np.isfinite(res['actual']['explained_deviance']).any()
        # The GAM fit succeeded, so the actual branch still recorded fit_time and
        # a per-fold filter-shape vector for each fold.
        assert res['actual']['filter_shapes'].shape == (2, pipeline.history_frames)
        assert np.isfinite(res['actual']['fit_time']).any()

    @pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
    @pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
    @pytest.mark.filterwarnings("ignore:An input array is constant:scipy.stats.ConstantInputWarning")
    def test_sklearn_zero_null_deviance_sets_d2_zero(self, monkeypatch):
        """
        Forcing ``mean_gamma_deviance`` to always return ``0.0`` makes the null
        deviance vanish, driving the ``if null_dev == 0: d2 = 0.0`` guard (which
        avoids a divide-by-zero) in both the actual and null branches of
        ``_run_model_for_feature_sklearn``. Every fitted fold therefore reports a
        finite explained deviance of exactly ``0.0``.
        """

        pipeline = _minimal_bout_pipeline(split_strategy='mixed', split_num=2)
        feature_data = _bout_feature_data(pipeline.history_frames)
        basis = np.eye(pipeline.history_frames, 3, dtype=float)

        monkeypatch.setattr(bout_params_module, 'mean_gamma_deviance', lambda *a, **k: 0.0)

        _fn, res = pipeline._run_model_for_feature_sklearn('self.speed', feature_data, basis)
        assert np.allclose(res['actual']['explained_deviance'], 0.0)
        assert np.allclose(res['null']['explained_deviance'], 0.0)

    @pytest.mark.filterwarnings("ignore:Bitwise inversion:DeprecationWarning")
    @pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
    @pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
    @pytest.mark.filterwarnings("ignore:An input array is constant:scipy.stats.ConstantInputWarning")
    def test_pygam_zero_null_deviance_sets_d2_zero(self, monkeypatch):
        """
        Forcing ``mean_gamma_deviance`` to always return ``0.0`` drives the
        ``if null_dev == 0: d2 = 0.0`` guard in both branches of
        ``_run_model_for_feature_pygam`` so each fitted fold reports a finite
        explained deviance of exactly ``0.0`` rather than a divide-by-zero.
        """

        pipeline = _minimal_bout_pipeline(split_strategy='mixed', split_num=2)
        feature_data = _bout_feature_data(pipeline.history_frames)

        monkeypatch.setattr(bout_params_module, 'mean_gamma_deviance', lambda *a, **k: 0.0)

        _fn, res = pipeline._run_model_for_feature_pygam('self.speed', feature_data, None)
        assert np.allclose(res['actual']['explained_deviance'], 0.0)
        assert np.allclose(res['null']['explained_deviance'], 0.0)
