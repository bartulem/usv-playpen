"""
@author: bartulem
End-to-end smoke tests for the continuous USV-MANIFOLD-POSITION (2-D UMAP)
modeling pipeline and its JAX `SmoothBivariateRegression` model-selection path,
driven entirely on tiny synthetic data.

These tests deliberately walk the *production* code paths rather than testing
isolated helpers:

* ``TestContinuousInputExtraction`` runs the real
  ``ContinuousModelingPipeline.extract_and_save_continuous_data`` against a
  synthetic session tree, lighting up the continuous-target extraction,
  the inverse-density KDE weighting (``compute_inverse_density_weights``),
  the kinematic-column selection / harmonization / cross-session z-scoring
  chain, the predictor-audit wrapper, and ``modeling_metadata``. Both the
  ``euclidean`` and the ``torus`` manifold metrics are exercised (the torus
  path additionally drives the 3x3 lattice-replication KDE).

* ``TestContinuousModelRunner`` drives
  ``ContinuousModelRunner.run_univariate_training`` on the extracted pickle
  with tiny JAX knobs (small ``max_iter``, ``bin_resizing_factor`` downsampled,
  Python-loop solver). This exercises ``load_univariate_data_blocks``, the
  spatial K-Means cross-validation splitter, the ``actual`` / ``null`` /
  ``null_model_free`` strategies, and the per-fold metric / diagnostic storage.

* ``TestManifoldModelSelection`` runs the real
  ``continuous_vocal_manifold_model_selection`` (the manifold branch of the
  multi-thousand-line ``model_selection.py``) on a strong-signal synthetic
  input pickle plus a freshly-computed univariate ranking. The signal feature
  drives the 2-D target linearly so it clears the ``r2_spatial`` screening
  gate, the Step-0 empirical-density baseline is established, the auto-anchor
  fires, and the forward-selection loop runs to its finalization step. The
  per-step pickles are asserted to carry the documented structure.

Warning policy
--------------
The project runs pytest with ``filterwarnings = ["error"]``. The modeling
import chain pulls ``optax`` -> a one-time JAX ``DeprecationWarning``, so all
top-level modeling imports below are wrapped in a ``warnings.catch_warnings``
block that ignores ``DeprecationWarning`` during import. At run time the JAX /
SciPy KDE and correlation helpers can emit ``RuntimeWarning`` on the tiny
synthetic traces (degenerate variance, all-NaN slices); these are demoted with
narrow per-test ``@pytest.mark.filterwarnings`` markers. ``matplotlib`` is
forced onto the headless ``Agg`` backend in case any imported module pulls
``pyplot`` at import time.
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
    build_modeling_settings,
    build_session_tree,
    write_session_list_file,
)

# The modeling import chain pulls optax -> a one-time JAX DeprecationWarning.
# Guard the top-level imports so collection does not trip ``filterwarnings =
# ["error"]`` before any per-test marker can take effect.
with warnings.catch_warnings():
    warnings.simplefilter('ignore', DeprecationWarning)
    from usv_playpen.modeling.modeling_usv_manifold_position import (
        ContinuousModelingPipeline,
        ContinuousModelRunner,
        compute_inverse_density_weights,
        get_stratified_spatial_splits_stable,
        _log_spaced_grid,
        _tune_manifold_regularization,
    )
    from usv_playpen.modeling.jax_bivariate_regression import (
        SmoothBivariateRegression,
    )
    from usv_playpen.modeling.model_selection import (
        continuous_vocal_manifold_model_selection,
    )
    from usv_playpen.modeling.manifold_metric import (
        circular_mean,
        signed_diff,
        total_dispersion,
    )


# Tiny-data geometry shared across the extraction tests. The continuous
# pipeline extracts every non-noise USV after the silent ``filter_history``
# warm-up as a target (not just bouts), so a generous USV count per session is
# obtained cheaply. ``HISTORY_FRAMES`` is the derived
# ``floor(CAMERA_FPS * FILTER_HISTORY)`` column count of every per-event window.
CAMERA_FPS = 60.0
FILTER_HISTORY = 0.5
HISTORY_FRAMES = int(np.floor(CAMERA_FPS * FILTER_HISTORY))
N_FRAMES = 7200       # 120 s sessions -> plenty of post-warm-up vocal targets
N_SESSIONS = 4
N_BOUTS = 15
USV_PER_BOUT = 3
SPATIAL_CLUSTER_NUM = 3   # small so K-Means always populates every cluster slot


def _build_manifold_settings(tmp_path: Path, **overrides):
    """
    Description
    -----------
    Builds the synthetic session tree, the session-list file, and a trimmed,
    JAX-shrunk ``modeling_settings`` dict for a tiny continuous-manifold smoke
    run, all rooted under ``tmp_path``.

    The behavioural / vocal data is sized by the module-level tiny-data
    constants. On top of the shared ``build_modeling_settings`` trimming this
    helper additionally shrinks the manifold-specific knobs: the spatial
    K-Means cluster count is dropped to ``SPATIAL_CLUSTER_NUM`` (so the
    cluster-coverage invariant is satisfiable on the small synthetic target
    cloud), and the JAX ``SmoothBivariateRegression`` hyperparameters are
    forced tiny (``bin_resizing_factor`` downsampling, low ``max_iter``,
    regularisation tuning off, Python-loop solver) so every fit is fast.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        The pytest-provided per-test scratch directory; every artifact lives
        below it so nothing is ever written into the package tree.
    overrides (dict)
        Extra keyword arguments forwarded to ``build_modeling_settings`` (e.g.
        ``split_strategy``, ``split_num``, ``usv_predictor_type``).

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

    # The continuous pipeline reads ``usv_predictor_type`` to optionally build
    # vocal-signal predictor columns; default to ``None`` so the design matrix
    # stays minimal (the continuous-target extraction is independent of it).
    overrides.setdefault('usv_predictor_type', None)

    settings = build_modeling_settings(
        session_list_file=list_file,
        save_directory=save_dir,
        camera_sampling_rate=CAMERA_FPS,
        filter_history=FILTER_HISTORY,
        **overrides,
    )
    settings['model_params']['usv_bout_time'] = 0.5
    settings['model_params']['spatial_cluster_num'] = SPATIAL_CLUSTER_NUM

    hp = settings['hyperparameters']['jax_linear']['bivariate']
    hp['bin_resizing_factor'] = 10        # 30 -> 3 temporal predictors
    hp['max_iter'] = 60
    hp['tune_regularization_bool'] = False
    hp['use_lax_loop'] = False
    hp['learning_rate'] = 0.05

    return settings, save_dir


def _build_signal_continuous_pickle(
        save_path: Path,
        feature_names: list[str],
        session_ids: list[str],
        history_frames: int,
        input_metadata: dict,
        n_per_session: int = 60,
        seed: int = 0,
        target_kind: str = 'linear',
) -> Path:
    """
    Description
    -----------
    Serializes a controlled continuous-manifold input pickle directly, with a
    deliberate linear ``X -> Y`` signal on the *first* feature so the manifold
    model-selection screening gate (mean ``r2_spatial`` > 0 and a per-fold
    Wilcoxon ``actual > null``) can be cleared and the full
    Step-0 / anchor / forward-selection machinery is exercised.

    The artifact matches the schema the continuous runner and
    ``continuous_vocal_manifold_model_selection`` consume:

        {
          '<feature>': {
              '<session_id>': {
                  'X': np.ndarray (n_per_session, history_frames),
                  'Y': np.ndarray (n_per_session, 2),
                  'w': np.ndarray (n_per_session,),
              }, ...
          }, ...,
          '_input_metadata': {...}
        }

    The selection routine builds its global ``Y`` / ``w`` / ``groups`` arrays
    from the *first ranked* feature, so within every session a single shared
    ``Y`` (a low-noise linear function of the signal feature's per-trial
    history mean) and unit weights are written across all features. The
    non-signal features carry independent random histories, so they fail the
    screen while the signal feature passes.

    Parameters
    ----------
    save_path (pathlib.Path)
        Destination ``.pkl`` path (parent dirs created if absent).
    feature_names (list of str)
        Feature keys; ``feature_names[0]`` is the signal-bearing feature.
    session_ids (list of str)
        Session identifiers populated under every feature.
    history_frames (int)
        Number of temporal lags (columns) per event window.
    input_metadata (dict)
        The reserved ``_input_metadata`` block. Must carry
        ``analysis_specific.usv_category_column_name`` so the selector can
        build its per-step filename prefix.
    n_per_session (int)
        Number of vocal events per session.
    seed (int)
        Base seed for the per-cell RNG.
    target_kind (str)
        ``'linear'`` (default) writes the original ``Y = [2*base, -1.5*base]``
        coordinate signal the euclidean coordinate model recovers. ``'wound_torus'``
        writes a target that *winds* around the torus: each axis is the wrapped
        ``atan2`` of two fixed (cross-session) linear projections of the signal
        feature's history, so the 4-D embedding is linear in ``X`` and the convex
        embedding ridge recovers it while the coordinate model cannot. The
        winding construction is exact (noise-free) so it clears the torus
        screening gate; it is meaningful only when the caller disables temporal
        binning (``bin_resizing_factor=1``) so the full-width projection survives.

    Returns
    -------
    save_path (pathlib.Path)
        The path the pickle was written to (absolute).
    """

    rng = np.random.default_rng(seed)
    artifact: dict = {}

    for feature in feature_names:
        artifact[feature] = {}
        for sess in session_ids:
            X = rng.standard_normal((n_per_session, history_frames)).astype(np.float32)
            artifact[feature][sess] = {'X': X, 'Y': None, 'w': None}

    target_rng = np.random.default_rng(seed + 99)
    # Fixed (cross-session) linear projections for the wound-torus target so the
    # same X -> Y relationship holds globally and the embedding ridge can learn
    # it; for the linear target these are unused.
    proj_x1, proj_x2 = target_rng.standard_normal(history_frames), target_rng.standard_normal(history_frames)
    proj_y1, proj_y2 = target_rng.standard_normal(history_frames), target_rng.standard_normal(history_frames)
    for sess in session_ids:
        signal_X = artifact[feature_names[0]][sess]['X']
        if target_kind == 'wound_torus':
            theta_x = np.arctan2(signal_X @ proj_x2, signal_X @ proj_x1)
            theta_y = np.arctan2(signal_X @ proj_y2, signal_X @ proj_y1)
            Y = np.stack(
                [(theta_x / (2.0 * np.pi)) % 1.0, (theta_y / (2.0 * np.pi)) % 1.0],
                axis=1,
            ).astype(np.float32)
        else:
            base = signal_X.mean(axis=1)
            Y = np.stack(
                [2.0 * base, -1.5 * base], axis=1
            ).astype(np.float32) + 0.05 * target_rng.standard_normal(
                (len(base), 2)
            ).astype(np.float32)
        w = np.ones(len(Y), dtype=np.float32)
        for feature in feature_names:
            artifact[feature][sess]['Y'] = Y
            artifact[feature][sess]['w'] = w

    artifact['_input_metadata'] = input_metadata

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open('wb') as fh:
        pickle.dump(artifact, fh)
    return save_path


class TestContinuousInputExtraction:
    """End-to-end extraction of the continuous-manifold input pickle from a tree."""

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_extraction_produces_continuous_xyw_pickle(self, tmp_path):
        """
        The real ``extract_and_save_continuous_data`` writes a
        ``modeling_manifold_*.pkl`` whose structure matches the documented
        continuous contract: a nested ``{feature: {session: {X, Y, w}}}`` dict
        carrying a reserved ``_input_metadata`` block. Every per-event window
        is ``HISTORY_FRAMES`` wide, every target is the 2-D UMAP coordinate
        pair, the inverse-density weights are finite with unit global mean, and
        the per-session event counts are identical across features (the
        intra-session alignment invariant).
        """

        settings, save_dir = _build_manifold_settings(tmp_path)
        pipeline = ContinuousModelingPipeline(modeling_settings_dict=settings)
        pipeline.extract_and_save_continuous_data()

        pkls = list(save_dir.glob('modeling_manifold_*.pkl'))
        assert len(pkls) == 1, f"expected exactly one manifold pickle, got {pkls}"

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
        all_w = []
        for sess in sessions:
            entry = artifact[anchor][sess]
            X = entry['X']
            Y = entry['Y']
            w = entry['w']
            assert X.shape[1] == HISTORY_FRAMES
            assert Y.shape[1] == 2
            assert X.shape[0] == Y.shape[0] == w.shape[0]
            assert np.isfinite(X).all() and np.isfinite(Y).all() and np.isfinite(w).all()
            total_usv += X.shape[0]
            all_w.append(w)

            # Intra-session alignment: every feature shares this session's
            # event count, target matrix, and weight vector.
            for feat in feature_keys[1:]:
                other = artifact[feat][sess]
                assert other['X'].shape[0] == X.shape[0]
                np.testing.assert_array_equal(other['Y'], Y)
                np.testing.assert_array_equal(other['w'], w)

        assert total_usv > 0
        # Global KDE weights are normalised to unit mean by construction.
        pooled_w = np.concatenate(all_w)
        assert pytest.approx(1.0, abs=1e-4) == float(np.mean(pooled_w))
        assert (pooled_w > 0).all()

        md = artifact['_input_metadata']
        assert md['analysis_type'] == 'continuous'
        assert md['analysis_tag'] == 'manifold_vae_supercategory'
        spec = md['analysis_specific']
        assert spec['manifold_metric'] == 'euclidean'
        assert spec['usv_category_column_name'] == 'vae_supercategory'
        assert list(spec['usv_manifold_column_names']) == ['vae_umap1', 'vae_umap2']

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_extraction_torus_metric(self, tmp_path):
        """
        With ``vocal_features.usv_manifold_metric='torus'`` the extraction
        drives the 3x3 lattice-replication KDE inside
        ``compute_inverse_density_weights`` instead of the flat-space KDE.
        The output schema is identical to the euclidean run and the metadata
        records the torus metric, so a single settings flip is the only
        observable difference.
        """

        settings, save_dir = _build_manifold_settings(tmp_path)
        settings['vocal_features']['usv_manifold_metric'] = 'torus'
        settings['vocal_features']['usv_manifold_period'] = 10.0

        pipeline = ContinuousModelingPipeline(modeling_settings_dict=settings)
        pipeline.extract_and_save_continuous_data()

        with next(save_dir.glob('modeling_manifold_*.pkl')).open('rb') as fh:
            artifact = pickle.load(fh)

        feature_keys = [k for k in artifact if not k.startswith('_')]
        anchor = sorted(feature_keys)[0]
        pooled_w = np.concatenate([artifact[anchor][s]['w'] for s in artifact[anchor]])
        assert np.isfinite(pooled_w).all()
        assert pytest.approx(1.0, abs=1e-4) == float(np.mean(pooled_w))
        assert artifact['_input_metadata']['analysis_specific']['manifold_metric'] == 'torus'

    def test_too_few_manifold_columns_raises(self, tmp_path):
        """
        A ``usv_manifold_column_names`` list with fewer than two entries trips
        the first dimensionality guard in ``extract_and_save_continuous_data``.
        """

        settings, _ = _build_manifold_settings(tmp_path)
        settings['vocal_features']['usv_manifold_column_names'] = ['vae_umap1']
        pipeline = ContinuousModelingPipeline(modeling_settings_dict=settings)
        with pytest.raises(ValueError, match="at least"):
            pipeline.extract_and_save_continuous_data()

    def test_more_than_two_manifold_columns_raises(self, tmp_path):
        """
        A 3-D ``usv_manifold_column_names`` list trips the second guard: the
        continuous pipeline currently assumes a strictly 2-D UMAP target.
        """

        settings, _ = _build_manifold_settings(tmp_path)
        settings['vocal_features']['usv_manifold_column_names'] = [
            'vae_umap1', 'vae_umap2', 'vae_umap3',
        ]
        pipeline = ContinuousModelingPipeline(modeling_settings_dict=settings)
        with pytest.raises(ValueError, match="2-D target"):
            pipeline.extract_and_save_continuous_data()


class TestContinuousModelRunner:
    """The continuous univariate JAX runner on the freshly-extracted pickle."""

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_run_univariate_training_emits_three_strategies(self, tmp_path):
        """
        ``ContinuousModelRunner.run_univariate_training`` fits the
        ``SmoothBivariateRegression`` estimator per spatial-CV fold and returns
        a nested ``{strategy: {'folds': {...}}}`` dict for the three strategies
        (``actual``, ``null``, ``null_model_free``). Each strategy carries the
        full ``r2_spatial`` / ``euclidean_mae`` / ``mahalanobis_mae`` metric
        bundle plus per-fold optimiser diagnostics, with one entry per split.
        """

        settings, save_dir = _build_manifold_settings(
            tmp_path, split_strategy='mixed', split_num=2, test_proportion=0.3,
        )
        pipeline = ContinuousModelingPipeline(modeling_settings_dict=settings)
        pipeline.extract_and_save_continuous_data()
        input_pkl = str(next(save_dir.glob('modeling_manifold_*.pkl')))

        runner = ContinuousModelRunner(pipeline)
        results = runner.run_univariate_training(input_pkl, 'self.speed')

        assert set(results.keys()) == {'actual', 'null', 'null_model_free'}

        n_splits = settings['model_params']['split_num']
        for strategy in ('actual', 'null', 'null_model_free'):
            folds = results[strategy]['folds']
            metrics = folds['metrics']
            for key in ('r2_spatial', 'euclidean_mae', 'mahalanobis_mae'):
                assert key in metrics
                assert len(metrics[key]) == n_splits
            assert len(folds['y_pred_xy']) == n_splits
            assert len(folds['test_indices']) == n_splits
            # Every fold's prediction is the (n_test, 2) UMAP coordinate pair.
            for pred in folds['y_pred_xy']:
                assert pred.ndim == 2 and pred.shape[1] == 2

        # The modelled strategies expose per-fold optimiser diagnostics; the
        # empirical-density-draw baseline records a zero-iteration "fit".
        assert len(results['actual']['folds']['converged']) == n_splits
        assert all(it == 0 for it in results['null_model_free']['folds']['n_iter'])

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_run_univariate_training_with_regularization_tuning(self, tmp_path):
        """
        With ``tune_regularization_bool=True`` (and tiny inner-CV grids) each
        outer fold runs ``_tune_manifold_regularization`` — a joint inner cross-
        validation over the log-spaced ``(lambda_smooth, l2_reg)`` grids — before
        the outer fit. The runner persists the selected per-fold hyperparameters,
        a ``hyperparams_tuned`` flag, and the full inner-CV ``grid_audit`` payload
        (``grid_scores`` / ``grid_ses`` / ``argmax_pair`` / ``one_se_applied`` /
        ``one_se_threshold``) for the modelled strategies.
        """

        settings, save_dir = _build_manifold_settings(
            tmp_path, split_strategy='mixed', split_num=2, test_proportion=0.3,
        )
        # Tiny tuning grids so the inner CV is cheap: a 3-point lambda_smooth
        # grid, a 1-point l2 grid, two inner folds, very few inner iterations.
        hp = settings['hyperparameters']['jax_linear']['bivariate']
        hp['tune_regularization_bool'] = True
        tune_params = hp['tune_regularization_params']
        tune_params['lambda_smooth_decades_each_side'] = 1
        tune_params['l2_reg_decades_each_side'] = 0
        tune_params['inner_cv_folds'] = 2
        tune_params['inner_max_iter'] = 20

        pipeline = ContinuousModelingPipeline(modeling_settings_dict=settings)
        pipeline.extract_and_save_continuous_data()
        input_pkl = str(next(save_dir.glob('modeling_manifold_*.pkl')))

        runner = ContinuousModelRunner(pipeline)
        results = runner.run_univariate_training(input_pkl, 'self.speed')

        n_splits = settings['model_params']['split_num']
        actual_folds = results['actual']['folds']
        assert actual_folds['hyperparams_tuned'] == [True] * n_splits
        assert len(actual_folds['selected_lambda_smooth']) == n_splits
        assert all(np.isfinite(actual_folds['selected_lambda_smooth']))

        audit = actual_folds['hyperparam_grid_audit'][0]
        assert set(audit.keys()) == {
            'grid_scores', 'grid_ses', 'argmax_pair',
            'one_se_applied', 'one_se_threshold',
        }
        # The 3-point lambda_smooth x 1-point l2 grid yields three scored pairs.
        assert len(audit['grid_scores']) == 3

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_run_univariate_training_torus_metric(self, tmp_path):
        """
        With ``vocal_features.usv_manifold_metric='torus'`` the runner resolves
        the convex closed-form ``SmoothTorusManifoldRegression`` via the
        ``resolve_manifold_regressor_cls`` factory and fits it per outer fold
        (and inside the inner-CV tuner). The full per-fold loop completes for all
        three strategies; the two modelled strategies record the 4-D sin-cos
        embedding filter (``coef_`` width 4, not the coordinate model's 2) and a
        single closed-form iteration that is always converged, while the decoded
        predictions remain valid 2-D torus coordinates. This is the end-to-end
        proof that the univariate runner is fully functional on a torus run.
        """

        settings, save_dir = _build_manifold_settings(
            tmp_path, split_strategy='mixed', split_num=2, test_proportion=0.3,
        )
        settings['vocal_features']['usv_manifold_metric'] = 'torus'
        pipeline = ContinuousModelingPipeline(modeling_settings_dict=settings)
        pipeline.extract_and_save_continuous_data()
        input_pkl = str(next(save_dir.glob('modeling_manifold_*.pkl')))

        runner = ContinuousModelRunner(pipeline)
        results = runner.run_univariate_training(input_pkl, 'self.speed')

        assert set(results.keys()) == {'actual', 'null', 'null_model_free'}
        n_splits = settings['model_params']['split_num']
        for strategy in ('actual', 'null'):
            folds = results[strategy]['folds']
            assert len(folds['weights']) == n_splits
            for w in folds['weights']:
                # Torus path: the filter targets the 4-D sin-cos embedding,
                # unlike the coordinate model's 2-wide (x, y) filter.
                assert w.ndim == 2 and w.shape[1] == 4
            # Closed-form solve -> exactly one "iteration", always converged.
            assert all(it == 1 for it in folds['n_iter'])
            assert all(folds['converged'])
            for pred in folds['y_pred_xy']:
                assert pred.ndim == 2 and pred.shape[1] == 2
        # The model-free density-draw baseline is unaffected (still a 0-iter "fit").
        assert all(it == 0 for it in results['null_model_free']['folds']['n_iter'])


class TestManifoldModelSelection:
    """The real forward-stepwise ``continuous_vocal_manifold_model_selection``."""

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_selection_writes_step_pickles_on_signal_data(self, tmp_path):
        """
        Running ``continuous_vocal_manifold_model_selection`` on a strong-signal
        synthetic input pickle (with a matching freshly-computed univariate
        ranking) clears the ``r2_spatial`` screening gate on the signal feature,
        establishes the Step-0 empirical-density baseline, fires the auto-anchor,
        and runs the forward-selection loop to its finalization step. The
        per-step pickles carry the ``current_features`` / ``baseline_score`` /
        ``candidates_summary`` / ``selected_feature`` structure the consolidator
        expects, the accepted feature set never shrinks, and the Step-0 pickle
        records the ``null_model_free`` empirical-density baseline.
        """

        # The session-grain screen bootstraps SESSIONS, so its statistical power
        # comes from the session count (not the fold count). Use a larger session
        # panel under session-holdout -- the split the gate is designed for -- so
        # the strong-signal feature's per-session margin clears the bootstrap CI.
        gate_n_sessions = 25
        settings, _save_dir = _build_manifold_settings(
            tmp_path, split_strategy='session', split_num=10, test_proportion=0.3,
        )
        history_frames = HISTORY_FRAMES
        feature_names = ['self.speed', 'other.speed', 'self.neck_elevation']
        session_ids = [f'session_{i}' for i in range(gate_n_sessions)]

        input_md = {
            'analysis_type': 'continuous',
            'analysis_tag': 'manifold_vae_supercategory',
            'session_ids': session_ids,
            'n_events_per_session': {sess_id: 60 for sess_id in session_ids},
            'analysis_specific': {
                'usv_category_column_name': 'vae_supercategory',
                'manifold_metric': 'euclidean',
                'manifold_period': 1.0,
            },
        }
        input_pkl = str(_build_signal_continuous_pickle(
            save_path=tmp_path / 'manifold_input.pkl',
            feature_names=feature_names,
            session_ids=session_ids,
            history_frames=history_frames,
            input_metadata=input_md,
        ))

        # Build the univariate ranking by running the real runner per feature
        # against the same signal pickle, then consolidate into the single-file
        # ``{feature: results}`` form the selector loads.
        runner = ContinuousModelRunner(
            ContinuousModelingPipeline(modeling_settings_dict=settings)
        )
        combined = {}
        for feature in feature_names:
            combined[feature] = runner.run_univariate_training(input_pkl, feature)
        combined['_input_metadata'] = input_md
        combined_path = tmp_path / 'univariate_combined.pkl'
        with combined_path.open('wb') as fh:
            pickle.dump(combined, fh)

        settings_json = tmp_path / 'settings.json'
        settings_json.write_text(json.dumps(settings))

        ms_dir = tmp_path / 'model_selection'
        ms_dir.mkdir()
        continuous_vocal_manifold_model_selection(
            univariate_results_path=str(combined_path),
            input_data_path=input_pkl,
            output_directory=str(ms_dir),
            settings_path=str(settings_json),
            use_top_rank_as_anchor=True,
            # Widened so the Bonferroni-corrected screening gate
            # (alpha / n_features) clears on the tiny synthetic target cloud.
            p_val=0.5,
        )

        step_pkls = sorted(ms_dir.glob('model_selection_continuous_manifold_*_step_*.pkl'))
        assert len(step_pkls) >= 2, "expected the Step-0 baseline plus at least one forward step"

        accepted_counts = []
        saw_null_model_free = False
        for p in step_pkls:
            with p.open('rb') as fh:
                step = pickle.load(fh)
            assert 'current_features' in step
            assert 'baseline_score' in step
            assert 'candidates_summary' in step
            assert 'selected_feature' in step
            accepted_counts.append(len(step['current_features']))
            if step['selected_feature'] == 'null_model_free':
                saw_null_model_free = True

        # The Step-0 empirical-density baseline must have been established.
        assert saw_null_model_free
        # The anchored search never shrinks the accepted feature set.
        assert accepted_counts == sorted(accepted_counts)
        # The signal feature must have been picked up as the anchor.
        with step_pkls[-1].open('rb') as fh:
            final_step = pickle.load(fh)
        assert 'self.speed' in final_step['current_features']
        # dcor_xy is the selection score on BOTH geometries now.
        assert final_step['_run_metadata']['selection_metric'] == 'dcor_xy'

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_selection_torus_metric_runs_forward_search(self, tmp_path):
        """
        The same end-to-end forward search on a ``metric='torus'`` run: with the
        signal target generated as a wound torus coordinate (``atan2`` of two
        linear projections of the signal feature, wrapped into ``[0, 1)``), the
        convex embedding ridge recovers it, clears the screening gate, and the
        selector runs Step-0 / anchor / forward-selection to finalization exactly
        as in the euclidean case. This is the end-to-end proof that the
        model-selection forward search is fully functional on a torus run; both
        the univariate ranking and every candidate fit construct the torus
        estimator via the factory and pass ``metric``/``period``.
        """

        # Session-holdout with a large session panel: the session-grain gate
        # bootstraps SESSIONS, so its power scales with the session count, and it
        # assumes whole-session holdout (a per-session margin is only leak-free
        # when the session's events are all in the test fold).
        gate_n_sessions = 25
        settings, _save_dir = _build_manifold_settings(
            tmp_path, split_strategy='session', split_num=10, test_proportion=0.3,
        )
        settings['vocal_features']['usv_manifold_metric'] = 'torus'
        # No temporal binning: the wound-torus target is built from the full-width
        # projection, so the model must see the full-width design matrix to
        # recover it. The closed-form embedding ridge is fast at full width.
        settings['hyperparameters']['jax_linear']['bivariate']['bin_resizing_factor'] = 1
        history_frames = HISTORY_FRAMES
        feature_names = ['self.speed', 'other.speed', 'self.neck_elevation']
        session_ids = [f'session_{i}' for i in range(gate_n_sessions)]

        input_md = {
            'analysis_type': 'continuous',
            'analysis_tag': 'manifold_qlvm_supercategory',
            'session_ids': session_ids,
            'n_events_per_session': {sess_id: 60 for sess_id in session_ids},
            'analysis_specific': {
                'usv_category_column_name': 'qlvm_supercategory',
                'manifold_metric': 'torus',
                'manifold_period': 1.0,
            },
        }
        input_pkl = str(_build_signal_continuous_pickle(
            save_path=tmp_path / 'manifold_input.pkl',
            feature_names=feature_names,
            session_ids=session_ids,
            history_frames=history_frames,
            input_metadata=input_md,
            target_kind='wound_torus',
        ))

        runner = ContinuousModelRunner(
            ContinuousModelingPipeline(modeling_settings_dict=settings)
        )
        combined = {}
        for feature in feature_names:
            combined[feature] = runner.run_univariate_training(input_pkl, feature)
        combined['_input_metadata'] = input_md
        combined_path = tmp_path / 'univariate_combined.pkl'
        with combined_path.open('wb') as fh:
            pickle.dump(combined, fh)

        settings_json = tmp_path / 'settings.json'
        settings_json.write_text(json.dumps(settings))

        ms_dir = tmp_path / 'model_selection'
        ms_dir.mkdir()
        continuous_vocal_manifold_model_selection(
            univariate_results_path=str(combined_path),
            input_data_path=input_pkl,
            output_directory=str(ms_dir),
            settings_path=str(settings_json),
            use_top_rank_as_anchor=True,
            p_val=0.5,
        )

        step_pkls = sorted(ms_dir.glob('model_selection_continuous_manifold_*_step_*.pkl'))
        assert len(step_pkls) >= 2, "expected the Step-0 baseline plus at least one forward step"

        saw_null_model_free = False
        for p in step_pkls:
            with p.open('rb') as fh:
                step = pickle.load(fh)
            for key in ('current_features', 'baseline_score',
                        'candidates_summary', 'selected_feature'):
                assert key in step
            if step['selected_feature'] == 'null_model_free':
                saw_null_model_free = True
        assert saw_null_model_free
        with step_pkls[-1].open('rb') as fh:
            final_step = pickle.load(fh)
        assert 'self.speed' in final_step['current_features']
        # The torus path screens/scores on dcor_xy, not r2_spatial.
        assert final_step['_run_metadata']['selection_metric'] == 'dcor_xy'

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_selection_torus_stale_ranking_raises(self, tmp_path):
        """
        On a torus run the selection screens on the session-grain paired-dcor
        margin, which needs `session_ids` / `n_events_per_session` in the
        univariate `_input_metadata` and per-fold predictions in each result. A
        stale ranking that predates that schema (only `r2_spatial` fold metrics,
        no session metadata) must raise a clear, actionable error rather than
        silently skipping every feature and mis-reporting the schema mismatch as
        "no significant features found" (a misleading false null).
        """

        settings, _ = _build_manifold_settings(
            tmp_path, split_strategy='mixed', split_num=2,
        )
        settings['vocal_features']['usv_manifold_metric'] = 'torus'
        # A pre-`dcor_xy` ranking: r2_spatial only.
        ranking = {
            'self.speed': {
                'actual': {'folds': {'metrics': {'r2_spatial': [0.1, 0.2]}}},
                'null_model_free': {'folds': {'metrics': {'r2_spatial': [0.0, 0.0]}}},
            },
            '_input_metadata': {'analysis_specific': {'manifold_metric': 'torus'}},
        }
        ranking_pkl = tmp_path / 'univariate_combined.pkl'
        with ranking_pkl.open('wb') as fh:
            pickle.dump(ranking, fh)
        settings_json = tmp_path / 'settings.json'
        settings_json.write_text(json.dumps(settings))
        ms_dir = tmp_path / 'model_selection'
        ms_dir.mkdir()

        with pytest.raises(ValueError, match="session_ids"):
            continuous_vocal_manifold_model_selection(
                univariate_results_path=str(ranking_pkl),
                input_data_path=str(tmp_path / 'does_not_need_to_exist.pkl'),
                output_directory=str(ms_dir),
                settings_path=str(settings_json),
                use_top_rank_as_anchor=True,
                p_val=0.05,
            )

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_selection_torus_resume_metric_mismatch_restarts_fresh(self, tmp_path, capsys):
        """
        Resuming a checkpoint scored on a different selection metric (the
        manifold metric was flipped between runs, so r2_spatial and dcor_xy
        live on different scales) must discard it and restart fresh rather than
        compare incompatible scores. Run the torus selection, tamper the latest
        checkpoint's recorded selection_metric to 'r2_spatial', re-run, and
        assert the resume announces the mismatch and finalises afresh on
        dcor_xy.
        """

        gate_n_sessions = 25
        settings, _ = _build_manifold_settings(
            tmp_path, split_strategy='session', split_num=10, test_proportion=0.3,
        )
        settings['vocal_features']['usv_manifold_metric'] = 'torus'
        settings['hyperparameters']['jax_linear']['bivariate']['bin_resizing_factor'] = 1
        feature_names = ['self.speed', 'other.speed', 'self.neck_elevation']
        session_ids = [f'session_{i}' for i in range(gate_n_sessions)]
        input_md = {
            'analysis_type': 'continuous',
            'analysis_tag': 'manifold_qlvm_supercategory',
            'session_ids': session_ids,
            'n_events_per_session': {sess_id: 60 for sess_id in session_ids},
            'analysis_specific': {
                'usv_category_column_name': 'qlvm_supercategory',
                'manifold_metric': 'torus', 'manifold_period': 1.0,
            },
        }
        input_pkl = str(_build_signal_continuous_pickle(
            save_path=tmp_path / 'manifold_input.pkl',
            feature_names=feature_names, session_ids=session_ids,
            history_frames=HISTORY_FRAMES, input_metadata=input_md,
            target_kind='wound_torus',
        ))
        runner = ContinuousModelRunner(
            ContinuousModelingPipeline(modeling_settings_dict=settings)
        )
        combined = {f: runner.run_univariate_training(input_pkl, f) for f in feature_names}
        combined['_input_metadata'] = input_md
        combined_path = tmp_path / 'univariate_combined.pkl'
        with combined_path.open('wb') as fh:
            pickle.dump(combined, fh)
        settings_json = tmp_path / 'settings.json'
        settings_json.write_text(json.dumps(settings))
        ms_dir = tmp_path / 'model_selection'
        ms_dir.mkdir()

        def _run():
            continuous_vocal_manifold_model_selection(
                univariate_results_path=str(combined_path), input_data_path=input_pkl,
                output_directory=str(ms_dir), settings_path=str(settings_json),
                use_top_rank_as_anchor=True, p_val=0.5,
            )

        _run()
        steps = sorted(ms_dir.glob('*_step_*.pkl'))
        assert steps
        # Tamper the latest checkpoint to look like a euclidean (r2_spatial) run.
        with steps[-1].open('rb') as fh:
            payload = pickle.load(fh)
        payload['_run_metadata']['selection_metric'] = 'r2_spatial'
        with steps[-1].open('wb') as fh:
            pickle.dump(payload, fh)

        capsys.readouterr()  # clear captured output
        _run()
        out = capsys.readouterr().out
        assert "different scales" in out and "starting fresh" in out
        final = sorted(ms_dir.glob('*_step_*.pkl'))[-1]
        with final.open('rb') as fh:
            assert pickle.load(fh)['_run_metadata']['selection_metric'] == 'dcor_xy'

    def test_metric_aware_centroid_and_r2_math(self):
        """
        Contract guard for the **metric-aware centroid / dispersion math**
        (``circular_mean`` / ``signed_diff`` / ``total_dispersion``) that
        underpins ``r2_spatial`` (its centroid-referenced denominator and the
        active / ``null`` strategies' residuals) and the ``null_model_free``
        mahalanobis covariance, all parameterized by the run's manifold metric
        so the score is computed on the SAME metric as the active models.

        (The Step-0 ``null_model_free`` *prediction* is now an empirical-density
        draw, not the centroid — see ``manifold_prediction_metrics``; this test
        guards the centroid math that the draw's mahalanobis term and r2's
        denominator still rely on.)

        This reproduces the metric-aware centroid r2 formula and checks two
        things on a cluster that straddles the wrap boundary:
        (1) on ``euclidean`` it equals the original flat computation (so
        euclidean runs are unchanged — the end-to-end euclidean path is
        additionally covered by ``test_selection_writes_step_pickles_on_signal_data``);
        (2) on ``torus`` the circular centroid lands inside the dense cluster
        near the wrap boundary instead of the flat arithmetic mean in the
        empty middle of the circle, giving a materially different (correct)
        baseline score. A regression to flat math would collapse the two.
        """

        period = 1.0
        # Train cluster straddling the 0/period seam: 0.95 and 0.05 are 0.1
        # apart on the torus but ~0.9 apart on the flat line.
        Y_tr = np.array(
            [[0.95, 0.95], [0.05, 0.05], [0.00, 0.00], [0.90, 0.90]],
            dtype=np.float64,
        )
        Y_te = np.array([[0.98, 0.98], [0.02, 0.02]], dtype=np.float64)
        w_tr = np.ones(len(Y_tr), dtype=np.float64)

        def _baseline(metric):
            mu = circular_mean(Y_tr, metric=metric, period=period, weights=w_tr)
            resid = signed_diff(Y_te, mu[None, :], metric=metric, period=period)
            sse = float(np.sum(resid ** 2))
            denom = total_dispersion(Y_te, metric=metric, period=period)
            r2 = (1.0 - sse / denom) if denom > 0 else 0.0
            return r2, mu

        # Reference: the original FLAT baseline formula (arithmetic centroid,
        # flat residuals, flat per-axis SST) the code used before the fix.
        mu_flat = np.average(Y_tr, axis=0, weights=w_tr)
        dx = Y_te[:, 0] - mu_flat[0]
        dy = Y_te[:, 1] - mu_flat[1]
        sse_flat = np.sum(dx ** 2 + dy ** 2)
        denom_flat = (
            np.sum((Y_te[:, 0] - np.mean(Y_te[:, 0])) ** 2)
            + np.sum((Y_te[:, 1] - np.mean(Y_te[:, 1])) ** 2)
        )
        r2_flat = (1.0 - sse_flat / denom_flat) if denom_flat > 0 else 0.0

        r2_eucl, mu_eucl = _baseline('euclidean')
        r2_torus, mu_torus = _baseline('torus')

        # (1) Euclidean baseline is byte-equivalent to the old flat formula.
        assert np.allclose(mu_eucl, mu_flat)
        assert np.isclose(r2_eucl, r2_flat)

        # (2) Torus centroid sits in the dense cluster near the seam (~0 or
        # ~period), not at the flat arithmetic mean (~0.475).
        assert mu_torus[0] < 0.1 or mu_torus[0] > period - 0.1
        assert 0.4 < mu_eucl[0] < 0.55
        # The two metrics therefore give materially different baseline scores;
        # a revert to flat math would make them identical.
        assert not np.isclose(r2_torus, r2_eucl)


def _spatial_blob(n_clusters: int, per_cluster: int, seed: int = 0):
    """
    Description
    -----------
    Builds a tiny, well-separated 2-D point cloud of ``n_clusters`` Gaussian
    blobs (``per_cluster`` points each) so K-Means inside the spatial splitter
    deterministically recovers exactly ``n_clusters`` non-empty clusters. Used
    by the pure-NumPy splitter tests below, which must satisfy the splitter's
    cohort-wide cluster-coverage invariant without any on-disk extraction.

    Parameters
    ----------
    n_clusters (int)
        Number of well-separated blobs (= the splitter's ``n_clusters``).
    per_cluster (int)
        Points generated per blob.
    seed (int)
        RNG seed for the within-blob jitter.

    Returns
    -------
    Y (np.ndarray)
        ``(n_clusters * per_cluster, 2)`` coordinate cloud.
    cluster_id (np.ndarray)
        Ground-truth blob id per row (handy for building session groups).
    """

    rng = np.random.default_rng(seed)
    centres = np.array([[10.0 * c, 0.0] for c in range(n_clusters)], dtype=np.float64)
    pts = []
    ids = []
    for c in range(n_clusters):
        pts.append(centres[c] + 0.05 * rng.standard_normal((per_cluster, 2)))
        ids.append(np.full(per_cluster, c))
    return np.vstack(pts).astype(np.float32), np.concatenate(ids)


class TestSpatialSplitter:
    """Pure-NumPy / K-Means branches of ``get_stratified_spatial_splits_stable``."""

    def test_invalid_strategy_raises(self):
        """An unsupported ``split_strategy`` raises ``ValueError`` immediately."""

        Y, _ = _spatial_blob(n_clusters=3, per_cluster=10)
        with pytest.raises(ValueError, match="Invalid split_strategy"):
            get_stratified_spatial_splits_stable(
                groups=np.zeros(len(Y), dtype=int), Y=Y, n_clusters=3,
                split_strategy='bogus', test_prop=0.3, n_splits=1, random_seed=0,
            )

    @pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
    def test_degenerate_Y_fewer_clusters_raises(self):
        """
        A near-constant ``Y`` collapses several K-Means centres, so fewer than
        ``n_clusters`` non-empty clusters are produced and the cohort-wide
        cluster-coverage invariant raises ``ValueError``.
        """

        Y = np.zeros((20, 2), dtype=np.float32)   # all identical points
        with pytest.raises(ValueError, match="non-empty clusters"):
            get_stratified_spatial_splits_stable(
                groups=np.zeros(20, dtype=int), Y=Y, n_clusters=4,
                split_strategy='mixed', test_prop=0.3, n_splits=1, random_seed=0,
            )

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_mixed_strategy_torus_metric_folds(self):
        """
        On ``metric='torus'`` the K-Means proxy runs on the 4-D torus
        embedding rather than the raw coordinates (the torus splitter arm).
        The mixed strategy then returns ``n_splits`` disjoint folds that each
        cover every cluster on both sides.
        """

        Y, _ = _spatial_blob(n_clusters=3, per_cluster=20, seed=1)
        groups = np.arange(len(Y)) % 4
        folds = get_stratified_spatial_splits_stable(
            groups=groups, Y=Y, n_clusters=3, split_strategy='mixed',
            test_prop=0.3, n_splits=2, random_seed=0,
            metric='torus', period=100.0,
        )
        assert len(folds) == 2
        for tr_idx, te_idx in folds:
            assert set(tr_idx.tolist()).isdisjoint(te_idx.tolist())

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_session_strategy_keeps_sessions_disjoint(self):
        """
        The 'session' rejection-sampler branch treats whole sessions as the
        atomic unit. With one well-separated cluster per session replicated
        across enough sessions, it accepts folds whose test sessions never
        appear in train, covering every cluster on both sides.
        """

        # Six sessions, each carrying all three spatial clusters, so the
        # session-level sampler can find folds covering every cluster.
        Y_list, groups_list = [], []
        for sess in range(6):
            Yc, _ = _spatial_blob(n_clusters=3, per_cluster=8, seed=sess)
            Y_list.append(Yc)
            groups_list.append(np.full(len(Yc), sess))
        Y = np.vstack(Y_list)
        groups = np.concatenate(groups_list)

        folds = get_stratified_spatial_splits_stable(
            groups=groups, Y=Y, n_clusters=3, split_strategy='session',
            test_prop=0.5, n_splits=2, random_seed=0, tolerance=0.5,
        )
        assert len(folds) == 2
        for tr_idx, te_idx in folds:
            tr_sessions = set(groups[tr_idx].tolist())
            te_sessions = set(groups[te_idx].tolist())
            assert tr_sessions.isdisjoint(te_sessions)

    def test_session_strategy_raises_when_unsatisfiable(self):
        """
        When a spatial cluster lives in a single session, no session split can
        keep all clusters on both sides; the rejection sampler exhausts
        ``max_total_attempts`` (driving the tolerance-widening arm) and raises
        ``RuntimeError``.
        """

        # Two sessions share clusters 0 and 1; cluster 2 lives only in
        # session 2, so any split isolating session 2 strips cluster 2 from
        # one side.
        Y0, _ = _spatial_blob(n_clusters=2, per_cluster=10, seed=0)
        Y1, _ = _spatial_blob(n_clusters=2, per_cluster=10, seed=1)
        Y2, _ = _spatial_blob(n_clusters=3, per_cluster=10, seed=2)
        Y = np.vstack([Y0, Y1, Y2])
        groups = np.concatenate([
            np.full(len(Y0), 0), np.full(len(Y1), 1), np.full(len(Y2), 2),
        ])
        with pytest.raises(RuntimeError, match="valid spatial splits"):
            get_stratified_spatial_splits_stable(
                groups=groups, Y=Y, n_clusters=3, split_strategy='session',
                test_prop=1.0 / 3.0, n_splits=2, random_seed=0,
                tolerance=0.0, max_total_attempts=1500, widen_every=1000,
                widen_step=0.0,
            )


class TestManifoldLogGrid:
    """The standalone manifold ``_log_spaced_grid`` helper guards."""

    def test_grid_shapes_and_invalid_inputs(self):
        """
        Returns a sorted log-spaced grid of length ``2*decades+1`` centred on
        the value, collapses to ``[center]`` at ``decades=0``, and raises on a
        negative half-width or non-positive centre.
        """

        np.testing.assert_allclose(
            _log_spaced_grid(center=1.0, decades_each_side=2),
            [1e-2, 1e-1, 1e0, 1e1, 1e2],
        )
        np.testing.assert_allclose(_log_spaced_grid(center=0.7, decades_each_side=0), [0.7])
        with pytest.raises(ValueError, match='decades_each_side must be >= 0'):
            _log_spaced_grid(center=1.0, decades_each_side=-1)
        with pytest.raises(ValueError, match='center must be positive'):
            _log_spaced_grid(center=0.0, decades_each_side=1)


class TestManifoldTuner:
    """Branches of ``_tune_manifold_regularization`` the runner test skips."""

    def test_degenerate_single_session_returns_grid_centre(self):
        """
        A single-session (or too-few-sample) training fold cannot be inner-
        split, so the tuner short-circuits to the median grid value of each
        grid and an empty audit payload without touching the estimator.
        """

        Y, _ = _spatial_blob(n_clusters=2, per_cluster=4, seed=0)
        X = np.zeros((len(Y), 1), dtype=np.float32)
        w = np.ones(len(Y), dtype=np.float32)
        groups = np.zeros(len(Y), dtype=int)   # single session -> early bail
        lambda_grid = np.array([0.1, 1.0, 10.0])
        l2_grid = np.array([0.001, 0.01, 0.1])

        best_lam, best_l2, audit = _tune_manifold_regularization(
            X, Y, w, groups,
            lambda_smooth_grid=lambda_grid,
            l2_reg_grid=l2_grid,
            inner_cv_folds=2,
            inner_cv_scoring_metric='r2_spatial',
            inner_cv_use_one_se_rule=False,
            n_features=1,
            n_time_bins=1,
            spatial_cluster_num=2,
            smoothness_derivative_order=1,
            huber_delta=1.0,
            learning_rate=0.05,
            inner_max_iter=5,
            tol=0.01,
            random_state=0,
            verbose=False,
            use_lax_loop=False,
            regressor_cls=SmoothBivariateRegression,
            metric='euclidean',
            period=1.0,
        )
        assert best_lam == 1.0
        assert best_l2 == 0.01
        assert audit == {
            'grid_scores': {}, 'grid_ses': {},
            'argmax_pair': None, 'one_se_applied': False, 'one_se_threshold': None,
        }

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_all_nan_grid_falls_back_with_verbose(self):
        """
        When every inner fit raises (stub regressor), all grid pairs score
        NaN, the per-fold ``verbose`` exception-print arm runs, and the tuner
        returns the median grid values with a populated NaN-scored audit
        instead of crashing.
        """

        class _AlwaysRaises:
            """Stub regressor whose ``fit`` always raises to force the NaN path."""

            def __init__(self, **kwargs):
                pass

            def fit(self, X, Y, sample_weight=None):
                raise RuntimeError("forced inner-fit failure")

            def evaluate_metrics(self, X, Y, weights=None):
                return {}

        # Two sessions, each covering both inner clusters, so the inner
        # 'mixed' splitter is satisfiable and the grid loop actually runs.
        Y_list, groups_list = [], []
        for sess in range(2):
            Yc, _ = _spatial_blob(n_clusters=2, per_cluster=10, seed=sess)
            Y_list.append(Yc)
            groups_list.append(np.full(len(Yc), sess))
        Y = np.vstack(Y_list)
        groups = np.concatenate(groups_list)
        X = np.zeros((len(Y), 1), dtype=np.float32)
        w = np.ones(len(Y), dtype=np.float32)
        lambda_grid = np.array([0.1, 1.0, 10.0])
        l2_grid = np.array([0.01, 0.1])

        best_lam, best_l2, audit = _tune_manifold_regularization(
            X, Y, w, groups,
            lambda_smooth_grid=lambda_grid,
            l2_reg_grid=l2_grid,
            inner_cv_folds=2,
            inner_cv_scoring_metric='r2_spatial',
            inner_cv_use_one_se_rule=True,
            n_features=1,
            n_time_bins=1,
            spatial_cluster_num=2,
            smoothness_derivative_order=1,
            huber_delta=1.0,
            learning_rate=0.05,
            inner_max_iter=5,
            tol=0.01,
            random_state=0,
            verbose=True,
            use_lax_loop=False,
            regressor_cls=_AlwaysRaises,
            metric='euclidean',
            period=1.0,
        )
        assert best_lam == 1.0
        assert best_l2 == 0.1
        assert len(audit['grid_scores']) == lambda_grid.size * l2_grid.size
        assert all(not np.isfinite(v) for v in audit['grid_scores'].values())
        assert audit['argmax_pair'] is None

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_one_se_rule_selects_smoothest_in_band(self):
        """
        With a deterministic stub regressor that returns fixed per-pair scores
        (so the inner CV is exact), the 1-SE rule fires: it computes the
        threshold, collects the in-band pairs, and returns the smoothest
        (largest ``lambda_smooth``) among them. This drives the higher-is-
        better one-SE branch end-to-end.
        """

        # Per-fold scores keyed by lambda_smooth. With two inner folds the
        # argmax pair (lambda=1.0, mean 0.80) carries a non-zero SE (fold
        # scores 0.70 / 0.90), so its 1-SE band [0.80 - 0.10, ...] = >= 0.70
        # admits the smoother lambda=10.0 (mean 0.78). The 1-SE rule therefore
        # returns the smoother (larger-lambda) in-band pair, not the argmax.
        score_table = {
            0.1: [0.10, 0.12],     # clearly worst, never in-band
            1.0: [0.70, 0.90],     # mean 0.80 (argmax), SE = 0.10
            10.0: [0.76, 0.80],    # mean 0.78, inside the 0.70 threshold band
        }

        class _FixedScore:
            """Stub returning per-fold ``r2_spatial`` keyed by ``lambda_smooth``."""

            _call_counts: dict = {}

            def __init__(self, **kwargs):
                self._lam = round(kwargs['lambda_smooth'], 6)

            def fit(self, X, Y, sample_weight=None):
                return self

            def evaluate_metrics(self, X, Y, weights=None):
                idx = _FixedScore._call_counts.get(self._lam, 0)
                _FixedScore._call_counts[self._lam] = idx + 1
                return {'r2_spatial': score_table[self._lam][idx % 2]}

        _FixedScore._call_counts = {}

        Y_list, groups_list = [], []
        for sess in range(2):
            Yc, _ = _spatial_blob(n_clusters=2, per_cluster=10, seed=sess)
            Y_list.append(Yc)
            groups_list.append(np.full(len(Yc), sess))
        Y = np.vstack(Y_list)
        groups = np.concatenate(groups_list)
        X = np.zeros((len(Y), 1), dtype=np.float32)
        w = np.ones(len(Y), dtype=np.float32)

        best_lam, _best_l2, audit = _tune_manifold_regularization(
            X, Y, w, groups,
            lambda_smooth_grid=np.array([0.1, 1.0, 10.0]),
            l2_reg_grid=np.array([0.01]),
            inner_cv_folds=2,
            inner_cv_scoring_metric='r2_spatial',
            inner_cv_use_one_se_rule=True,
            n_features=1,
            n_time_bins=1,
            spatial_cluster_num=2,
            smoothness_derivative_order=1,
            huber_delta=1.0,
            learning_rate=0.05,
            inner_max_iter=5,
            tol=0.01,
            random_state=0,
            verbose=False,
            use_lax_loop=False,
            regressor_cls=_FixedScore,
            metric='euclidean',
            period=1.0,
        )
        assert best_lam == 10.0          # smoothest in-band pair
        assert audit['one_se_applied'] is True
        assert audit['argmax_pair'][0] == 1.0   # raw argmax was the wigglier one
        assert audit['one_se_threshold'] is not None

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_argmax_returned_when_one_se_rule_off(self):
        """
        With ``inner_cv_use_one_se_rule=False`` and a valid (finite-scoring)
        grid the tuner returns the raw performance-argmax pair without
        entering the 1-SE band logic. Driven on a real (tiny) estimator so
        the full grid loop, argmax selection, and the rule-off early return
        are all exercised.
        """

        score_table = {0.1: [0.20, 0.30], 1.0: [0.80, 0.85]}

        class _FixedScore:
            """Deterministic per-fold stub keyed by ``lambda_smooth``."""

            _calls: dict = {}

            def __init__(self, **kwargs):
                self._lam = round(kwargs['lambda_smooth'], 6)

            def fit(self, X, Y, sample_weight=None):
                return self

            def evaluate_metrics(self, X, Y, weights=None):
                idx = _FixedScore._calls.get(self._lam, 0)
                _FixedScore._calls[self._lam] = idx + 1
                return {'r2_spatial': score_table[self._lam][idx % 2]}

        _FixedScore._calls = {}

        Y_list, groups_list = [], []
        for sess in range(2):
            Yc, _ = _spatial_blob(n_clusters=2, per_cluster=10, seed=sess)
            Y_list.append(Yc)
            groups_list.append(np.full(len(Yc), sess))
        Y = np.vstack(Y_list)
        groups = np.concatenate(groups_list)
        X = np.zeros((len(Y), 1), dtype=np.float32)
        w = np.ones(len(Y), dtype=np.float32)

        best_lam, _best_l2, audit = _tune_manifold_regularization(
            X, Y, w, groups,
            lambda_smooth_grid=np.array([0.1, 1.0]),
            l2_reg_grid=np.array([0.01]),
            inner_cv_folds=2,
            inner_cv_scoring_metric='r2_spatial',
            inner_cv_use_one_se_rule=False,
            n_features=1, n_time_bins=1, spatial_cluster_num=2,
            smoothness_derivative_order=1, huber_delta=1.0, learning_rate=0.05,
            inner_max_iter=5, tol=0.01, random_state=0,
            verbose=False, use_lax_loop=False, regressor_cls=_FixedScore,
            metric='euclidean', period=1.0,
        )
        assert best_lam == 1.0           # raw argmax
        assert audit['one_se_applied'] is False

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_one_se_rule_lower_is_better_metric(self):
        """
        For a lower-is-better scoring metric (``euclidean_mae``) the 1-SE rule
        flips its comparison: the threshold becomes ``argmax + SE`` and the
        in-band test keeps pairs scoring *below* it. Drives the lower-is-
        better one-SE branch with a deterministic stub.
        """

        # Lower is better: lambda=1.0 best (mean 0.20, SE non-zero), lambda=10
        # within band (mean 0.24), lambda=0.1 far worse.
        score_table = {
            0.1: [0.90, 0.92],
            1.0: [0.10, 0.30],     # mean 0.20, SE = 0.10 -> band <= 0.30
            10.0: [0.22, 0.26],    # mean 0.24, in-band
        }

        class _FixedScore:
            _calls: dict = {}

            def __init__(self, **kwargs):
                self._lam = round(kwargs['lambda_smooth'], 6)

            def fit(self, X, Y, sample_weight=None):
                return self

            def evaluate_metrics(self, X, Y, weights=None):
                idx = _FixedScore._calls.get(self._lam, 0)
                _FixedScore._calls[self._lam] = idx + 1
                return {'euclidean_mae': score_table[self._lam][idx % 2]}

        _FixedScore._calls = {}

        Y_list, groups_list = [], []
        for sess in range(2):
            Yc, _ = _spatial_blob(n_clusters=2, per_cluster=10, seed=sess)
            Y_list.append(Yc)
            groups_list.append(np.full(len(Yc), sess))
        Y = np.vstack(Y_list)
        groups = np.concatenate(groups_list)
        X = np.zeros((len(Y), 1), dtype=np.float32)
        w = np.ones(len(Y), dtype=np.float32)

        best_lam, _, audit = _tune_manifold_regularization(
            X, Y, w, groups,
            lambda_smooth_grid=np.array([0.1, 1.0, 10.0]),
            l2_reg_grid=np.array([0.01]),
            inner_cv_folds=2,
            inner_cv_scoring_metric='euclidean_mae',
            inner_cv_use_one_se_rule=True,
            n_features=1, n_time_bins=1, spatial_cluster_num=2,
            smoothness_derivative_order=1, huber_delta=1.0, learning_rate=0.05,
            inner_max_iter=5, tol=0.01, random_state=0,
            verbose=False, use_lax_loop=False, regressor_cls=_FixedScore,
            metric='euclidean', period=1.0,
        )
        assert best_lam == 10.0          # smoothest in the lower-is-better band
        assert audit['one_se_applied'] is True


class TestInverseDensityWeights:
    """Direct coverage of the KDE inverse-density weighting helper."""

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_euclidean_and_torus_unit_mean(self):
        """
        Both the flat-space (euclidean) KDE and the 3x3 lattice-replication
        (torus) KDE return finite, strictly-positive weights normalised to
        exactly unit mean.
        """

        rng = np.random.default_rng(0)
        Y = rng.standard_normal((60, 2))
        for metric in ('euclidean', 'torus'):
            w = compute_inverse_density_weights(Y, metric=metric, period=8.0)
            assert w.shape == (60,)
            assert np.isfinite(w).all()
            assert (w > 0).all()
            assert pytest.approx(1.0, abs=1e-6) == float(np.mean(w))


class TestContinuousPipelineInit:
    """The ``__init__`` configuration-loading branches of the pipeline."""

    def test_init_loads_default_settings_when_none(self):
        """
        Constructing with ``modeling_settings_dict=None`` loads the shipped
        package JSON and derives ``history_frames`` from its camera-rate /
        filter-history block.
        """

        pipeline = ContinuousModelingPipeline(modeling_settings_dict=None)
        assert isinstance(pipeline.modeling_settings, dict)
        assert isinstance(pipeline.history_frames, int)

    def test_init_caches_feature_boundaries_and_kwargs(self):
        """
        A settings dict carrying ``feature_boundaries`` caches it on the
        instance, and extra keyword arguments are attached verbatim.
        """

        settings = {
            'io': {'camera_sampling_rate': 60.0},
            'model_params': {'filter_history': 0.5},
            'feature_boundaries': {'speed': [0.0, 1.0]},
        }
        pipeline = ContinuousModelingPipeline(
            modeling_settings_dict=settings, custom_marker='x'
        )
        assert pipeline.feature_boundaries == {'speed': [0.0, 1.0]}
        assert pipeline.custom_marker == 'x'
        assert pipeline.history_frames == 30

    def test_init_missing_history_keys_raises(self):
        """Missing camera-rate / filter-history keys raise ``KeyError``."""

        with pytest.raises(KeyError, match="Critical setting missing"):
            ContinuousModelingPipeline(modeling_settings_dict={'io': {}, 'model_params': {}})


class TestContinuousRunnerExtra:
    """Runner guards / branches the strong-signal tri-strategy test skips."""

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_run_univariate_training_unknown_feature_raises(self, tmp_path):
        """
        Requesting a feature absent from the loaded pickle raises ``KeyError``
        — the runner's not-found guard before any fold loop executes.
        """

        settings, save_dir = _build_manifold_settings(
            tmp_path, split_strategy='mixed', split_num=2, test_proportion=0.3,
        )
        pipeline = ContinuousModelingPipeline(modeling_settings_dict=settings)
        pipeline.extract_and_save_continuous_data()
        input_pkl = str(next(save_dir.glob('modeling_manifold_*.pkl')))

        runner = ContinuousModelRunner(pipeline)
        with pytest.raises(KeyError, match="not found"):
            runner.run_univariate_training(input_pkl, 'does.not.exist')

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_load_blocks_bins_all_features_when_no_filter(self, tmp_path):
        """
        ``load_univariate_data_blocks`` with ``feature_filter=None`` bins and
        returns every feature in the pickle (the legacy whole-pickle arm),
        stripping the reserved ``_input_metadata`` key.
        """

        settings, save_dir = _build_manifold_settings(tmp_path)
        pipeline = ContinuousModelingPipeline(modeling_settings_dict=settings)
        pipeline.extract_and_save_continuous_data()
        input_pkl = str(next(save_dir.glob('modeling_manifold_*.pkl')))

        blocks = ContinuousModelRunner.load_univariate_data_blocks(
            input_pkl, bin_size=10, feature_filter=None,
        )
        assert '_input_metadata' not in blocks
        assert len(blocks) >= 1
        anchor = sorted(blocks.keys())[0]
        assert blocks[anchor]['Y'].shape[1] == 2
        assert blocks[anchor]['X'].shape[0] == blocks[anchor]['Y'].shape[0]
