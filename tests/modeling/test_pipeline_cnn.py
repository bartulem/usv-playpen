"""
@author: bartulem
End-to-end smoke tests for the continuous USV-MANIFOLD-POSITION 1-D CNN
training runner (``NeuralContinuousCNNRunner`` in
``usv_playpen.modeling.jax_neural_network_cnn``), driven entirely on tiny
synthetic data rooted under ``tmp_path``.

These tests deliberately walk the *production* deep-learning execution paths
rather than testing isolated helpers. The runner is invoked exactly the way
the analysis notebook drives it:

    runner = NeuralContinuousCNNRunner(modeling_settings=<dict>)
    data_blocks = runner.load_multivariate_data_blocks(pkl_path=<manifold.pkl>)
    runner.run_cnn_training(data_blocks=data_blocks)

Coverage targets inside the runner:

* ``load_multivariate_data_blocks`` — reads the session-nested
  ``{feature: {session: {X, Y, w, supercategory, category}}}`` manifold
  pickle, strips the reserved ``_input_metadata`` block, truncates every
  per-event window to ``history_frames``, stacks the per-feature 2-D
  matrices into the ``(N, F, T)`` tensor, and surfaces the per-USV cluster
  labels when every session carries them.

* ``run_cnn_training`` — the full tri-strategy training loop
  (``null_model_free`` empirical-density draw, ``null`` session-isolated
  target shuffle, ``actual`` kinematic fit), the spatial K-Means CV
  splitter, Phase-1 per-fold checkpointing, Phase-2 post-hoc permutation
  feature importance, and Phase-3 contrastive centroid-gradient saliency
  (which transitively exercises ``compute_centroid_saliency``). Every knob
  is shrunk to the minimum that still executes the code: ``epochs=1``,
  ``n_folds=2``, ``permutation_iterations=1``, a tiny ``hidden_dim`` /
  ``batch_size``, inception kernels off, KDE-weighted sampling on (the
  simplest epoch sampler).

* ``compute_centroid_saliency`` — additionally driven directly with the
  trained best-fold weights so the saliency batching / padding /
  input*gradient / contrastive-subtraction path is covered even when the
  Phase-3 dual filter keeps zero true positives.

* ``restrict_to_fold_indices`` — the Phase-3 recovery knob that skips
  non-target folds and emits ``+inf`` placeholder errors.

Warning policy
--------------
The project runs pytest with ``filterwarnings = ["error"]``. The modeling
import chain pulls ``optax`` -> a one-time JAX ``DeprecationWarning``, so the
top-level runner import below is wrapped in a ``warnings.catch_warnings``
block that ignores ``DeprecationWarning`` during import. At run time the JAX
training / saliency helpers can emit ``RuntimeWarning`` on the tiny synthetic
traces (degenerate variance, empty mean/std slices in the permutation SNR);
these are demoted with narrow per-test ``@pytest.mark.filterwarnings``
markers. ``matplotlib`` is forced onto the headless ``Agg`` backend in case
any imported module pulls ``pyplot`` at import time.
"""

from __future__ import annotations

import copy
import json
import pickle
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use('Agg')

# The CNN modeling import chain pulls optax -> a one-time JAX
# DeprecationWarning. Guard the top-level import so collection does not trip
# ``filterwarnings = ["error"]`` before any per-test marker can take effect.
with warnings.catch_warnings():
    warnings.simplefilter('ignore', DeprecationWarning)
    import jax
    import jax.numpy as jnp
    from usv_playpen.modeling.jax_neural_network_cnn import (
        NeuralContinuousCNNRunner,
        init_cnn_params_and_state,
    )

# Canonical location of the shipped settings JSON, resolved relative to the
# installed package so the synthetic settings inherit every real key (no
# schema drift against the production runner).
_SETTINGS_JSON = (
    Path(__file__).resolve().parents[2]
    / 'src' / 'usv_playpen' / '_parameter_settings' / 'modeling_settings.json'
)

# Tiny-data geometry shared across the runner tests. ``HISTORY_FRAMES`` is the
# derived ``floor(CAMERA_FPS * FILTER_HISTORY)`` column count of every
# per-event window; kept small so each (N, F, T) batch is cheap to convolve.
# The ResNet-1D stacks four residual-pooling blocks, each halving the temporal
# dimension; the window must therefore survive four floor-halvings without
# collapsing to zero (which would make the SE-attention ``jnp.max(..., axis=2)``
# reduce a zero-size array). 36 frames -> 18 -> 9 -> 4 -> 2 clears all four.
CAMERA_FPS = 60.0
FILTER_HISTORY = 0.6
HISTORY_FRAMES = int(np.floor(CAMERA_FPS * FILTER_HISTORY))  # 36
N_SESSIONS = 2
N_PER_SESSION = 24
FEATURE_NAMES = ['self.speed', 'other.speed']
# Two non-noise supercategory labels (plus the noise label 0) so the saliency
# pre-flight resolves >= 2 cluster centres and the alpha-gap radius rule fires.
SUPERCATEGORIES = (1, 2)


def _build_cnn_settings(save_dir: Path, source_pkl_path: Path, **mp_overrides) -> dict:
    """
    Description
    -----------
    Loads the canonical ``modeling_settings.json`` shipped with the package
    and returns a deep-copied, hyperparameter-shrunk variant suitable for a
    tiny end-to-end CNN smoke run. The real JSON is the source of truth for
    every key (so no schema drift), and only the IO paths, the CV split
    knobs, and the ``deep_learning.cnn_continuous`` deep-learning block are
    overridden down to the minimum that still executes every runner path.

    The shrunk CNN knobs are:
      - ``epochs=1`` / ``n_folds=2`` / ``permutation_iterations=1`` so the
        tri-strategy loop, the CV splitter, and Phase-2 all run once.
      - ``batch_size=4`` / ``hidden_dim=4`` so the convolution tensors and
        the dense head are tiny.
      - ``use_inception_kernels=False`` / ``use_hybrid_flatten=False`` /
        ``use_scheduler=False`` / ``use_kinematic_masking=False`` to keep the
        forward/update graph minimal and fast to compile.
      - ``use_kde_weights=True`` so the simplest (full-epoch weighted draw)
        sampler is used instead of the grid-balanced sampler.
      - patience set high enough that the single epoch is never short-
        circuited before a "best" snapshot is taken.

    Parameters
    ----------
    save_dir (pathlib.Path)
        Directory where ``run_cnn_training`` writes its deep-storage pickle
        (must live under ``tmp_path``; never the package tree).
    source_pkl_path (pathlib.Path)
        Path to the synthetic manifold input pickle; only its basename is
        consulted by the runner (for the sex-modifier filename tag) but it is
        threaded through for provenance fidelity.
    mp_overrides (dict)
        Extra ``model_params`` overrides (e.g. ``split_strategy``,
        ``test_proportion``, ``spatial_cluster_num``).

    Returns
    -------
    settings (dict)
        A ready-to-use ``modeling_settings`` dictionary.
    """

    with open(_SETTINGS_JSON, 'r') as fh:
        settings = copy.deepcopy(json.load(fh))

    settings['io']['save_directory'] = str(save_dir)
    settings['io']['camera_sampling_rate'] = CAMERA_FPS

    mp = settings['model_params']
    mp['filter_history'] = FILTER_HISTORY
    mp['split_strategy'] = 'mixed'
    mp['random_seed'] = 0
    mp['spatial_cluster_num'] = 2
    mp['test_proportion'] = 0.4
    mp.update(mp_overrides)

    hp = settings['hyperparameters']['deep_learning']['cnn_continuous']
    hp['epochs'] = 1
    hp['n_folds'] = 2
    hp['permutation_iterations'] = 1
    hp['batch_size'] = 4
    hp['hidden_dim'] = 4
    hp['se_reduction'] = 2
    hp['kernel_size'] = 3
    hp['use_inception_kernels'] = False
    hp['use_hybrid_flatten'] = False
    hp['use_scheduler'] = False
    hp['use_kinematic_masking'] = False
    hp['use_kde_weights'] = True
    hp['warp_range'] = 0.05
    hp['patience'] = 99
    hp['null_patience'] = 99
    hp['dropout_rate'] = 0.0
    hp['cnn_torus_output_encoding'] = 'sin_cos'

    return settings


def _build_cnn_input_pickle(
        save_path: Path,
        feature_names: list[str],
        session_ids: list[str],
        history_frames: int,
        n_per_session: int,
        with_labels: bool = True,
        seed: int = 0,
) -> Path:
    """
    Description
    -----------
    Serializes a controlled continuous-manifold input pickle directly, in the
    exact schema ``NeuralContinuousCNNRunner.load_multivariate_data_blocks``
    consumes:

        {
          '<feature>': {
              '<session_id>': {
                  'X': np.ndarray (n_per_session, history_frames),
                  'Y': np.ndarray (n_per_session, 2),
                  'w': np.ndarray (n_per_session,),
                  'supercategory': np.ndarray (n_per_session,)  # optional
                  'category':      np.ndarray (n_per_session,)  # optional
              }, ...
          }, ...,
          '_input_metadata': {...}
        }

    A deliberate (weak) linear ``X -> Y`` signal is baked into the first
    feature so the ``actual`` fit has something to learn, but the run is not
    asserted to beat the controls (one epoch on a tiny cloud is not expected
    to). Within a session the same ``Y`` / ``w`` are shared across features
    (the intra-session alignment invariant the loader relies on). The
    per-USV ``supercategory`` / ``category`` labels are split across two
    non-noise classes plus the noise label ``0`` so the saliency pre-flight
    finds >= 2 cluster centres; at least one USV per non-noise class is
    placed near that class's manifold centroid so the Phase-3 dual filter can
    keep a true positive.

    Parameters
    ----------
    save_path (pathlib.Path)
        Destination ``.pkl`` path (parent dirs created if absent).
    feature_names (list of str)
        Feature keys; ``feature_names[0]`` carries the weak signal.
    session_ids (list of str)
        Session identifiers populated under every feature.
    history_frames (int)
        Number of temporal lags (columns) per event window.
    n_per_session (int)
        Number of vocal events per session.
    with_labels (bool)
        When True, attach ``supercategory`` / ``category`` arrays so the
        saliency phase can run; when False omit them (legacy-pickle path).
    seed (int)
        Base seed for the per-cell RNG.

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
    for sess in session_ids:
        signal_X = artifact[feature_names[0]][sess]['X']
        base = signal_X.mean(axis=1)
        # Two well-separated manifold lobes (one per non-noise supercategory)
        # so derive_cluster_centers_empirically resolves two distinct centres.
        labels = np.where(
            np.arange(n_per_session) % 2 == 0, SUPERCATEGORIES[0], SUPERCATEGORIES[1]
        ).astype(np.int64)
        centre_x = np.where(labels == SUPERCATEGORIES[0], -3.0, 3.0)
        centre_y = np.where(labels == SUPERCATEGORIES[0], 2.0, -2.0)
        Y = np.stack(
            [centre_x + 0.3 * base, centre_y - 0.2 * base], axis=1
        ).astype(np.float32) + 0.05 * target_rng.standard_normal(
            (n_per_session, 2)
        ).astype(np.float32)
        w = np.ones(n_per_session, dtype=np.float32)

        for feature in feature_names:
            artifact[feature][sess]['Y'] = Y
            artifact[feature][sess]['w'] = w
            if with_labels:
                artifact[feature][sess]['supercategory'] = labels.copy()
                artifact[feature][sess]['category'] = labels.copy()

    artifact['_input_metadata'] = {
        'analysis_type': 'continuous',
        'analysis_tag': 'manifold_vae_supercategory',
        'analysis_specific': {
            'usv_category_column_name': 'vae_supercategory',
            'manifold_metric': 'euclidean',
            'manifold_period': 1.0,
        },
    }

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open('wb') as fh:
        pickle.dump(artifact, fh)
    return save_path


class TestRunnerConstruction:
    """Settings resolution and derived-attribute wiring in ``__init__``."""

    def test_init_from_none_loads_shipped_json(self):
        """
        ``NeuralContinuousCNNRunner(modeling_settings=None)`` falls back to the
        shipped ``modeling_settings.json``, derives ``history_frames`` from
        ``filter_history * camera_sampling_rate``, resolves the manifold
        metric/period, and promotes them into the ``hp`` HashableDict (so the
        JIT cache key for the forward / update functions flips on the metric).
        """

        runner = NeuralContinuousCNNRunner(modeling_settings=None)
        assert runner.history_frames > 0
        assert runner.manifold_metric in ('euclidean', 'torus')
        assert runner.hp['manifold_metric'] == runner.manifold_metric
        assert float(runner.hp['manifold_period']) == float(runner.manifold_period)
        assert runner.restrict_to_fold_indices is None


class TestLoadMultivariateDataBlocks:
    """The session-nested manifold pickle -> (N, F, T) tensor fusion path."""

    def test_load_stacks_tensor_and_surfaces_labels(self, tmp_path):
        """
        ``load_multivariate_data_blocks`` strips the reserved
        ``_input_metadata`` key, sorts the feature list, truncates every
        per-event window to ``history_frames``, stacks the per-feature 2-D
        matrices into a single ``(N, F, T)`` float32 tensor, and surfaces the
        per-USV ``supercategory`` / ``category`` cluster labels (present on
        every session) aligned to the pooled ``Y``.
        """

        save_dir = tmp_path / 'out'
        input_pkl = _build_cnn_input_pickle(
            save_path=tmp_path / 'manifold_input.pkl',
            feature_names=FEATURE_NAMES,
            session_ids=[f'session_{i}' for i in range(N_SESSIONS)],
            history_frames=HISTORY_FRAMES,
            n_per_session=N_PER_SESSION,
        )
        settings = _build_cnn_settings(save_dir, input_pkl)
        runner = NeuralContinuousCNNRunner(modeling_settings=settings)
        block = runner.load_multivariate_data_blocks(pkl_path=str(input_pkl))

        n_total = N_SESSIONS * N_PER_SESSION
        assert block['features'] == sorted(FEATURE_NAMES)
        assert block['num_bins'] == HISTORY_FRAMES
        assert block['X_seq'].shape == (n_total, len(FEATURE_NAMES), HISTORY_FRAMES)
        assert block['X_seq'].dtype == np.float32
        assert block['Y'].shape == (n_total, 2)
        assert block['w'].shape == (n_total,)
        assert block['groups'].shape == (n_total,)
        assert len(np.unique(block['groups'])) == N_SESSIONS
        # Per-USV labels surfaced because every session carried them.
        assert block['supercategory'].shape == (n_total,)
        assert block['category'].shape == (n_total,)
        assert str(block['source_pkl_path']) == str(input_pkl)

    def test_load_omits_labels_for_legacy_pickle(self, tmp_path):
        """
        A pickle built without per-USV ``supercategory`` / ``category``
        arrays (the legacy contract) loads cleanly; the loader simply omits
        those keys from the returned block so the saliency phase can later
        raise its explicit "re-extract" guidance.
        """

        save_dir = tmp_path / 'out'
        input_pkl = _build_cnn_input_pickle(
            save_path=tmp_path / 'legacy_input.pkl',
            feature_names=FEATURE_NAMES,
            session_ids=[f'session_{i}' for i in range(N_SESSIONS)],
            history_frames=HISTORY_FRAMES,
            n_per_session=N_PER_SESSION,
            with_labels=False,
        )
        settings = _build_cnn_settings(save_dir, input_pkl)
        runner = NeuralContinuousCNNRunner(modeling_settings=settings)
        block = runner.load_multivariate_data_blocks(pkl_path=str(input_pkl))

        assert 'supercategory' not in block
        assert 'category' not in block


class TestRunCnnTrainingFull:
    """The full tri-strategy + permutation + saliency orchestrator."""

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_run_cnn_training_writes_deep_storage(self, tmp_path):
        """
        ``run_cnn_training`` runs the three strategies per spatial-CV fold,
        the Phase-2 permutation feature importance, and the Phase-3
        contrastive centroid-gradient saliency, then serialises a single
        ``cnn_manifold_integrated_predictions_*.pkl`` deep-storage file. The
        saved structure must carry the metadata block, one cross-validation
        entry per fold (each holding the per-strategy ``Y_pred_*`` /
        ``error_*`` plus the persisted ``actual`` weights), the feature-
        importance bundle (means / stds / snrs / rankings), and the
        per-cluster saliency maps.
        """

        save_dir = tmp_path / 'out'
        input_pkl = _build_cnn_input_pickle(
            save_path=tmp_path / 'manifold_input.pkl',
            feature_names=FEATURE_NAMES,
            session_ids=[f'session_{i}' for i in range(N_SESSIONS)],
            history_frames=HISTORY_FRAMES,
            n_per_session=N_PER_SESSION,
        )
        settings = _build_cnn_settings(save_dir, input_pkl)
        runner = NeuralContinuousCNNRunner(modeling_settings=settings)
        data_blocks = runner.load_multivariate_data_blocks(pkl_path=str(input_pkl))
        runner.run_cnn_training(data_blocks=data_blocks)

        out_pkls = list(save_dir.glob('cnn_manifold_integrated_predictions_*.pkl'))
        assert len(out_pkls) == 1, f"expected one deep-storage pickle, got {out_pkls}"
        with out_pkls[0].open('rb') as fh:
            deep = pickle.load(fh)

        n_folds = settings['hyperparameters']['deep_learning']['cnn_continuous']['n_folds']

        # Metadata provenance.
        md = deep['metadata']
        assert md['features_list'] == sorted(FEATURE_NAMES)
        assert md['n_time_bins'] == HISTORY_FRAMES
        assert md['split_strategy'] == 'mixed'
        assert md['manifold_metric'] == 'euclidean'
        assert md['output_encoding'] in ('raw', 'sin_cos')

        # One cross-validation record per fold; each carries the tri-strategy
        # predictions / errors and the persisted "actual" weights.
        cv = deep['cross_validation']
        assert len(cv) == n_folds
        for fold_res in cv:
            assert fold_res['Y_pred_null_model_free'].shape[1] == 2
            assert fold_res['Y_pred_null'].shape[1] == 2
            assert fold_res['Y_pred_actual'].shape[1] == 2
            assert np.isfinite(fold_res['error_null_model_free'])
            assert 'params_actual' in fold_res
            assert 'state_actual' in fold_res

        # Phase-2 permutation importance bundle.
        fi = deep['feature_importance']
        assert set(fi['means'].keys()) == set(FEATURE_NAMES)
        assert set(fi['snrs'].keys()) == set(FEATURE_NAMES)
        assert sorted(fi['ranked_features']) == sorted(FEATURE_NAMES)
        assert 0 <= fi['best_fold_idx'] < n_folds

        # Phase-3 saliency maps (one per resolved non-noise cluster centre).
        assert 'saliency_maps' in deep
        assert len(deep['saliency_maps']) >= 1
        for cname, sal in deep['saliency_maps'].items():
            assert cname.startswith('supercategory_')
            assert sal['contrastive_saliency'].ndim == 3
            assert 'centroid' in sal and 'radius' in sal
        assert 'cluster_geometry' in deep

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_run_cnn_training_saliency_disabled(self, tmp_path):
        """
        With ``saliency.enable=False`` the Phase-3 block is skipped: the
        deep-storage pickle still carries the tri-strategy cross-validation
        and the Phase-2 permutation importance, but ``saliency_maps`` is left
        empty and no ``cluster_geometry`` is written. This also exercises the
        legacy-pickle (no per-USV labels) load path through training, since
        the saliency pre-flight that would demand labels is bypassed.
        """

        save_dir = tmp_path / 'out'
        input_pkl = _build_cnn_input_pickle(
            save_path=tmp_path / 'manifold_input.pkl',
            feature_names=FEATURE_NAMES,
            session_ids=[f'session_{i}' for i in range(N_SESSIONS)],
            history_frames=HISTORY_FRAMES,
            n_per_session=N_PER_SESSION,
            with_labels=False,
        )
        settings = _build_cnn_settings(save_dir, input_pkl)
        settings['hyperparameters']['deep_learning']['cnn_continuous']['saliency']['enable'] = False
        runner = NeuralContinuousCNNRunner(modeling_settings=settings)
        data_blocks = runner.load_multivariate_data_blocks(pkl_path=str(input_pkl))
        runner.run_cnn_training(data_blocks=data_blocks)

        with next(save_dir.glob('cnn_manifold_integrated_predictions_*.pkl')).open('rb') as fh:
            deep = pickle.load(fh)

        assert deep['saliency_maps'] == {}
        assert 'cluster_geometry' not in deep
        assert len(deep['cross_validation']) == 2
        assert set(deep['feature_importance']['means'].keys()) == set(FEATURE_NAMES)

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_run_cnn_training_restrict_to_fold(self, tmp_path):
        """
        With ``restrict_to_fold_indices`` set, only the listed fold is
        actually trained; the other folds are emitted as ``+inf``-error
        skipped placeholders (carrying only ``fold_idx`` / ``test_indices`` /
        ``Y_true`` / ``skipped``). The best-fold selection then naturally
        resolves to the single trained fold for the Phase-2 permutation pass.
        """

        save_dir = tmp_path / 'out'
        input_pkl = _build_cnn_input_pickle(
            save_path=tmp_path / 'manifold_input.pkl',
            feature_names=FEATURE_NAMES,
            session_ids=[f'session_{i}' for i in range(N_SESSIONS)],
            history_frames=HISTORY_FRAMES,
            n_per_session=N_PER_SESSION,
        )
        settings = _build_cnn_settings(save_dir, input_pkl)
        settings['hyperparameters']['deep_learning']['cnn_continuous']['saliency']['enable'] = False
        runner = NeuralContinuousCNNRunner(modeling_settings=settings)
        runner.restrict_to_fold_indices = [1]
        data_blocks = runner.load_multivariate_data_blocks(pkl_path=str(input_pkl))
        runner.run_cnn_training(data_blocks=data_blocks)

        with next(save_dir.glob('cnn_manifold_integrated_predictions_*.pkl')).open('rb') as fh:
            deep = pickle.load(fh)

        cv = deep['cross_validation']
        assert len(cv) == 2
        skipped = [f for f in cv if f.get('skipped')]
        trained = [f for f in cv if not f.get('skipped')]
        assert len(skipped) == 1
        assert len(trained) == 1
        assert deep['feature_importance']['best_fold_idx'] == 1


class TestRunCnnTrainingConfigVariants:
    """Alternate hyperparameter branches of the training loop / saliency."""

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_grid_sampler_masking_mse_and_category_saliency(self, tmp_path):
        """
        Flips every "other branch" hyperparameter so the alternate code paths
        run in a single fold:
          - ``use_kde_weights=False`` -> grid-balanced epoch sampler and the
            grid-balanced ``steps_per_epoch`` count.
          - ``use_kinematic_masking=True`` -> the kinematic-masking augmenter.
          - ``loss_function='mse'`` -> the standard-MSE residual branch.
          - ``weight_decay_exclude_output_head=False`` -> the un-masked AdamW.
          - ``saliency.segmentation='category'`` -> the category-label saliency
            branch (instead of supercategory).
          - a source-pickle basename containing ``female`` -> the ``female``
            sex-modifier filename tag.
        The run still produces a complete deep-storage pickle.
        """

        save_dir = tmp_path / 'out'
        input_pkl = _build_cnn_input_pickle(
            save_path=tmp_path / 'manifold_female_input.pkl',
            feature_names=FEATURE_NAMES,
            session_ids=[f'session_{i}' for i in range(N_SESSIONS)],
            history_frames=HISTORY_FRAMES,
            n_per_session=N_PER_SESSION,
        )
        settings = _build_cnn_settings(save_dir, input_pkl, spatial_cluster_num=2)
        hp = settings['hyperparameters']['deep_learning']['cnn_continuous']
        hp['use_kde_weights'] = False
        hp['use_kinematic_masking'] = True
        hp['masking_prob'] = 0.5
        hp['masking_length_frames'] = 3
        hp['loss_function'] = 'mse'
        hp['weight_decay_exclude_output_head'] = False
        hp['grid_size'] = 3
        hp['samples_per_cell'] = 8
        hp['saliency']['segmentation'] = 'category'

        runner = NeuralContinuousCNNRunner(modeling_settings=settings)
        data_blocks = runner.load_multivariate_data_blocks(pkl_path=str(input_pkl))
        runner.run_cnn_training(data_blocks=data_blocks)

        out_pkls = list(save_dir.glob('cnn_manifold_integrated_predictions_female_*.pkl'))
        assert len(out_pkls) == 1, f"expected one female-tagged pickle, got {out_pkls}"
        with out_pkls[0].open('rb') as fh:
            deep = pickle.load(fh)
        assert len(deep['cross_validation']) == 2
        for cname in deep['saliency_maps']:
            assert cname.startswith('category_')


class TestRunCnnTrainingTorus:
    """The torus ``sin_cos`` output-head / wrap-aware loss + eval branches."""

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_torus_sin_cos_run(self, tmp_path):
        """
        With ``vocal_features.usv_manifold_metric='torus'`` and the default
        ``cnn_torus_output_encoding='sin_cos'`` the runner emits the raw 4-D
        per-axis ``(sin, cos)`` head, trains with the ``sin_cos`` MSE residual
        against the encoded target, and decodes predictions back to a 2-D
        angle via ``angle_decode_jax`` in every eval / permutation / saliency
        consumer. The deep-storage metadata records ``output_encoding`` ==
        ``'sin_cos'`` and the saved ``Y_pred_*`` arrays are 2-D. The
        ``male_mute_partner`` source-basename also drives that sex-modifier
        filename tag.
        """

        save_dir = tmp_path / 'out'
        input_pkl = _build_cnn_input_pickle(
            save_path=tmp_path / 'manifold_male_mute_partner_input.pkl',
            feature_names=FEATURE_NAMES,
            session_ids=[f'session_{i}' for i in range(N_SESSIONS)],
            history_frames=HISTORY_FRAMES,
            n_per_session=N_PER_SESSION,
        )
        # Re-write Y / labels onto a periodic [0, period) torus so the
        # wrap-aware geometry is well-defined for the saliency centres.
        period = 10.0
        with input_pkl.open('rb') as fh:
            artifact = pickle.load(fh)
        rng = np.random.default_rng(7)
        for sess in artifact[FEATURE_NAMES[0]]:
            labels = artifact[FEATURE_NAMES[0]][sess]['supercategory']
            base_xy = np.where(labels[:, None] == SUPERCATEGORIES[0], 2.0, 7.0)
            Y = (base_xy + 0.3 * rng.standard_normal((len(labels), 2))) % period
            for feature in FEATURE_NAMES:
                artifact[feature][sess]['Y'] = Y.astype(np.float32)
        with input_pkl.open('wb') as fh:
            pickle.dump(artifact, fh)

        settings = _build_cnn_settings(save_dir, input_pkl)
        settings['vocal_features']['usv_manifold_metric'] = 'torus'
        settings['vocal_features']['usv_manifold_period'] = period
        hp = settings['hyperparameters']['deep_learning']['cnn_continuous']
        hp['cnn_torus_output_encoding'] = 'sin_cos'

        runner = NeuralContinuousCNNRunner(modeling_settings=settings)
        data_blocks = runner.load_multivariate_data_blocks(pkl_path=str(input_pkl))
        runner.run_cnn_training(data_blocks=data_blocks)

        out_pkls = list(save_dir.glob('cnn_manifold_integrated_predictions_male_mute_partner_*.pkl'))
        assert len(out_pkls) == 1, f"expected one male_mute_partner pickle, got {out_pkls}"
        with out_pkls[0].open('rb') as fh:
            deep = pickle.load(fh)

        assert deep['metadata']['manifold_metric'] == 'torus'
        assert deep['metadata']['output_encoding'] == 'sin_cos'
        for fold_res in deep['cross_validation']:
            # Decoded back to the 2-D angle representation for every consumer.
            assert fold_res['Y_pred_actual'].shape[1] == 2


class TestSaliencyPreflightGuards:
    """The Phase-1 pre-flight that fails fast before any fold is trained."""

    def test_preflight_invalid_segmentation(self, tmp_path):
        """An unknown ``saliency.segmentation`` value raises ``ValueError``
        before Phase 1 starts (no folds trained)."""

        save_dir = tmp_path / 'out'
        input_pkl = _build_cnn_input_pickle(
            save_path=tmp_path / 'manifold_input.pkl',
            feature_names=FEATURE_NAMES,
            session_ids=[f'session_{i}' for i in range(N_SESSIONS)],
            history_frames=HISTORY_FRAMES,
            n_per_session=N_PER_SESSION,
        )
        settings = _build_cnn_settings(save_dir, input_pkl)
        settings['hyperparameters']['deep_learning']['cnn_continuous']['saliency']['segmentation'] = 'bogus'
        runner = NeuralContinuousCNNRunner(modeling_settings=settings)
        data_blocks = runner.load_multivariate_data_blocks(pkl_path=str(input_pkl))
        with pytest.raises(ValueError, match="saliency.segmentation"):
            runner.run_cnn_training(data_blocks=data_blocks)

    def test_preflight_missing_labels(self, tmp_path):
        """``saliency.enable=True`` on a legacy pickle that does not carry the
        requested per-USV labels raises ``RuntimeError`` before Phase 1."""

        save_dir = tmp_path / 'out'
        input_pkl = _build_cnn_input_pickle(
            save_path=tmp_path / 'manifold_input.pkl',
            feature_names=FEATURE_NAMES,
            session_ids=[f'session_{i}' for i in range(N_SESSIONS)],
            history_frames=HISTORY_FRAMES,
            n_per_session=N_PER_SESSION,
            with_labels=False,
        )
        settings = _build_cnn_settings(save_dir, input_pkl)
        runner = NeuralContinuousCNNRunner(modeling_settings=settings)
        data_blocks = runner.load_multivariate_data_blocks(pkl_path=str(input_pkl))
        with pytest.raises(RuntimeError, match="does not carry"):
            runner.run_cnn_training(data_blocks=data_blocks)

    def test_preflight_single_cluster_centre(self, tmp_path):
        """When only one non-noise cluster centre can be resolved (every
        labelled USV shares one class), the pre-flight raises ``RuntimeError``
        about the alpha-gap radius rule needing >= 2 centres."""

        save_dir = tmp_path / 'out'
        save_path = tmp_path / 'manifold_input.pkl'
        # Build the labelled pickle, then overwrite every label with the same
        # single non-noise class so only one centre survives.
        _build_cnn_input_pickle(
            save_path=save_path,
            feature_names=FEATURE_NAMES,
            session_ids=[f'session_{i}' for i in range(N_SESSIONS)],
            history_frames=HISTORY_FRAMES,
            n_per_session=N_PER_SESSION,
        )
        with save_path.open('rb') as fh:
            artifact = pickle.load(fh)
        for feature in FEATURE_NAMES:
            for sess in artifact[feature]:
                n = artifact[feature][sess]['supercategory'].shape[0]
                artifact[feature][sess]['supercategory'] = np.full(n, SUPERCATEGORIES[0], dtype=np.int64)
                artifact[feature][sess]['category'] = np.full(n, SUPERCATEGORIES[0], dtype=np.int64)
        with save_path.open('wb') as fh:
            pickle.dump(artifact, fh)

        settings = _build_cnn_settings(save_dir, save_path)
        runner = NeuralContinuousCNNRunner(modeling_settings=settings)
        data_blocks = runner.load_multivariate_data_blocks(pkl_path=str(save_path))
        with pytest.raises(RuntimeError, match="cluster centre"):
            runner.run_cnn_training(data_blocks=data_blocks)


class TestComputeCentroidSaliencyDirect:
    """Drive the contrastive centroid-gradient saliency helper directly."""

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_centroid_saliency_shapes_and_finiteness(self, tmp_path):
        """
        ``compute_centroid_saliency`` returns a single ``contrastive_saliency``
        tensor shaped exactly like the test input ``(N, F, T)``. Calling it
        directly with freshly-initialised network weights exercises the
        mini-batch padding path (a remainder batch smaller than
        ``batch_size``), the region/global gradient vmaps, the input*gradient
        scaling, and the contrastive trial-template subtraction without
        needing a full training run.
        """

        save_dir = tmp_path / 'out'
        input_pkl = _build_cnn_input_pickle(
            save_path=tmp_path / 'manifold_input.pkl',
            feature_names=FEATURE_NAMES,
            session_ids=[f'session_{i}' for i in range(N_SESSIONS)],
            history_frames=HISTORY_FRAMES,
            n_per_session=N_PER_SESSION,
        )
        settings = _build_cnn_settings(save_dir, input_pkl)
        runner = NeuralContinuousCNNRunner(modeling_settings=settings)
        block = runner.load_multivariate_data_blocks(pkl_path=str(input_pkl))

        # Initialise a minimal network exactly as run_cnn_training does.
        n_feats = len(block['features'])
        n_bins = block['num_bins']
        Y = block['Y']
        Y_center = jnp.array((np.max(Y, 0) + np.min(Y, 0)) / 2.0)
        Y_scale = jnp.array((np.max(Y, 0) - np.min(Y, 0)) / 2.0 * 1.1)
        params, state = init_cnn_params_and_state(
            jax.random.PRNGKey(0), n_feats, n_bins, runner.hp
        )

        # Pick a test slice whose size is NOT a multiple of batch_size so the
        # remainder-batch padding branch is exercised.
        batch_size = runner.hp['batch_size']
        n_take = batch_size + 1
        X_te = jnp.array(block['X_seq'][:n_take])

        # The global saliency baseline is now computed once (cluster-invariant) and
        # passed in; mirror the production call path here.
        global_template = runner._compute_global_saliency_template(
            params, state, X_te, Y_center, Y_scale,
        )
        out = runner.compute_centroid_saliency(
            params, state, X_te, Y_center, Y_scale,
            polygon_centroid=(float(Y_center[0]), float(Y_center[1])),
            global_template=global_template,
        )

        sal = out['contrastive_saliency']
        assert sal.shape == (n_take, n_feats, n_bins)
        assert sal.dtype == np.float32
        assert np.isfinite(sal).all()
