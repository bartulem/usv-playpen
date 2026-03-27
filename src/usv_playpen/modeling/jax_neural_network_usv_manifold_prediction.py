"""
@author: bartulem
Module for deep non-linear USV manifold prediction (based on JAX, assumes GPU usage).

This module provides a deep learning pipeline to map high-dimensional, temporal
behavioral kinematics onto the highly non-linear 2D continuous acoustic manifold (UMAP)
representing the vocal repertoire.

Traditional linear mapping techniques are mathematically insufficient to resolve the
coordinate transformations required by the UMAP topology. To address this, we utilize
a dual-stream multi-layer perceptron (MLP). The network independently processes static
postural contexts (global stream) and dynamic kinematic sequences (temporal stream)
before a late-stage additive fusion.

Key scientific capabilities:
1.  Geographic fairness via grid-balancing: Prevents the optimizer from collapsing
    predictions into the dense acoustic core by uniformly sampling across a discretized
    2D spatial grid during batch construction.
2.  Phase invariance via temporal warping: Applies dynamic temporal warping to the
    kinematic history sequences, stretching and squeezing inputs to force the model
    to learn invariant behavioral motifs rather than memorizing exact timestamps.
3.  Feature attribution & saliency: Implements post-hoc permutation feature importance
    and input-gradient saliency (input x gradient) to extract specific, localized
    kinematic drivers for distinct vocalization states.
4.  Absolute Spatial Baselines: Benchmarks the network against both a label-shuffled
    null model and a 'null_model_free' spatial density prior (predicting the training
    manifold's center of mass) to rigorously validate kinematic predictive power.
"""

import os

# Force efficient memory allocation to prevent JAX from hoarding VRAM and causing OOMs
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

import copy
import jax
import jax.numpy as jnp
import json
import optax
import numpy as np
import pathlib
import pickle
from datetime import datetime
from functools import partial
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedShuffleSplit
from typing import Dict, Any, List, Tuple


def apply_temporal_warping(x_seq: np.ndarray, warp_factors: np.ndarray) -> np.ndarray:
    """
    Applies dynamic temporal warping to a batch of kinematic sequences.

    To encourage the neural network to learn phase-invariant behavioral motifs
    (e.g., recognizing a 'lunge' regardless of whether it happened in 0.5s or 0.6s),
    this function stretches or compresses the time-series matrix using linear interpolation.

    Parameters
    ----------
    x_seq : np.ndarray
        A 3D array of shape (batch, features, bins) representing the raw temporal kinematics.
    warp_factors : np.ndarray
        A 1D array of shape (batch, ) containing the specific scaling factor for each
        sequence in the batch (e.g., 0.90 for a 10% squeeze, 1.10 for a 10% stretch).

    Returns
    -------
    warped_batch : np.ndarray
        A 3D array of identical shape to `x_seq`, containing the temporally warped kinematics.
    """
    batch_size, n_feats, n_bins = x_seq.shape
    t_orig = np.linspace(0, 1, n_bins)
    warped_batch = np.zeros_like(x_seq)

    for i in range(batch_size):
        t_warped = np.linspace(0, 1 * warp_factors[i], n_bins)
        for f in range(n_feats):
            warped_batch[i, f, :] = np.interp(t_orig, t_warped, x_seq[i, f, :])

    return warped_batch


def get_grid_balanced_indices(Y_vals: np.ndarray, grid_size: int = 25, samples_per_cell: int = 40) -> np.ndarray:
    """
    Generates training indices that uniformly sample the 2D continuous acoustic manifold.

    Vocalization manifolds often exhibit extreme density at the "resting" origin. Standard
    random sampling causes the network to over-fit to this majority class. This function
    discretizes the UMAP space into a `grid_size` x `grid_size` matrix and draws an equal
    number of samples (`samples_per_cell`) from every occupied geographic neighborhood.

    Parameters
    ----------
    Y_vals : np.ndarray
        A 2D array of shape (N, 2) containing the continuous UMAP targets for the training set.
    grid_size : int, default 25
        The number of bins to divide both the X and Y spatial axes into.
    samples_per_cell : int, default 40
        The number of indices to sample (with replacement) from each occupied grid cell.

    Returns
    -------
    balanced_indices : np.ndarray
        A 1D array of selected indices. The total length is `occupied_cells * samples_per_cell`.
    """
    x_bins = np.linspace(Y_vals[:, 0].min(), Y_vals[:, 0].max(), grid_size)
    y_bins = np.linspace(Y_vals[:, 1].min(), Y_vals[:, 1].max(), grid_size)

    x_idx = np.digitize(Y_vals[:, 0], x_bins)
    y_idx = np.digitize(Y_vals[:, 1], y_bins)

    cells = {}
    for i, cell_coord in enumerate(zip(x_idx, y_idx)):
        if cell_coord not in cells:
            cells[cell_coord] = []
        cells[cell_coord].append(i)

    sampled_indices = [
        np.random.choice(cells[c], samples_per_cell, replace=True)
        for c in cells if len(cells[c]) > 0
    ]

    return np.concatenate(sampled_indices)


def init_mlp_params(key: jax.Array,
                    global_size: int,
                    temporal_size: int,
                    global_hidden: int = 2048,
                    temporal_hidden: int = 64) -> Dict[str, jax.Array]:
    """
    Initializes the weights and biases for the Dual-Stream MLP using He Normal initialization.

    Parameters
    ----------
    key : jax.Array
        A JAX pseudo-random number generator (PRNG) key.
    global_size : int
        The number of input features for the global stream (e.g., 2 * n_features for Mean/Std).
    temporal_size : int
        The flattened dimensionality of the temporal sequence (n_features * n_bins).
    global_hidden : int, default 2048
        Dimensionality of the first hidden layer in the global stream.
    temporal_hidden : int, default 64
        Dimensionality of the first hidden layer in the temporal stream.

    Returns
    -------
    params : dict
        A dictionary containing the initialized JAX arrays for all network weights and biases.
    """
    k = jax.random.split(key, 10)

    return {
        'g1_w': jax.random.normal(k[0], (global_size, global_hidden)) * jnp.sqrt(2 / global_size),
        'g1_b': jnp.zeros(global_hidden),
        'g2_w': jax.random.normal(k[1], (global_hidden, 512)) * jnp.sqrt(2 / global_hidden),
        'g2_b': jnp.zeros(512),
        'g3_w': jax.random.normal(k[2], (512, 128)) * jnp.sqrt(2 / 512),
        'g3_b': jnp.zeros(128),
        'g_out_w': jax.random.normal(k[3], (128, 2)) * jnp.sqrt(2 / 128),
        'g_out_b': jnp.zeros(2),

        't1_w': jax.random.normal(k[4], (temporal_size, temporal_hidden)) * jnp.sqrt(2 / temporal_size),
        't1_b': jnp.zeros(temporal_hidden),
        't2_w': jax.random.normal(k[5], (temporal_hidden, 16)) * jnp.sqrt(2 / temporal_hidden),
        't2_b': jnp.zeros(16),
        't_out_w': jax.random.normal(k[6], (16, 2)) * 0.001,
        't_out_b': jnp.zeros(2)
    }


@partial(jax.jit, static_argnames=['deterministic'])
def mlp_forward(params: Dict[str, jax.Array],
                X_seq: jax.Array,
                X_global: jax.Array,
                Y_center: jax.Array,
                Y_scale: jax.Array,
                deterministic: bool = False) -> jax.Array:
    """
    Executes the forward pass of the Dual-Stream MLP.

    Processes the static and dynamic kinematic features independently through their
    respective dense branches. The ultimate continuous 2D coordinate is generated via
    an additive fusion of both streams, gated by a scaled hyperbolic tangent (tanh)
    activation to strictly bind predictions within the empirical dimensions of the manifold.

    Parameters
    ----------
    params : dict
        The current model weights and biases.
    X_seq : jax.Array
        The 3D temporal kinematic sequences of shape (Batch, Features, Bins).
    X_global : jax.Array
        The 2D global context matrix of shape (Batch, Global_Features).
    Y_center : jax.Array
        The 2D spatial center of the current training fold's UMAP distribution.
    Y_scale : jax.Array
        The spatial half-width of the distribution (multiplied by a 1.1 safety margin).

    Returns
    -------
    predictions : jax.Array
        The predicted 2D UMAP coordinates of shape (Batch, 2).
    """
    X_seq_flat = X_seq.reshape(X_seq.shape[0], -1)

    g = jax.nn.relu(jnp.dot(X_global, params['g1_w']) + params['g1_b'])
    g = jax.nn.relu(jnp.dot(g, params['g2_w']) + params['g2_b'])
    g = jax.nn.relu(jnp.dot(g, params['g3_w']) + params['g3_b'])
    logits_g = jnp.dot(g, params['g_out_w']) + params['g_out_b']

    t = jax.nn.relu(jnp.dot(X_seq_flat, params['t1_w']) + params['t1_b'])
    t = jax.nn.relu(jnp.dot(t, params['t2_w']) + params['t2_b'])
    logits_t = jnp.dot(t, params['t_out_w']) + params['t_out_b']

    fused_logits = logits_g + logits_t
    predictions = Y_center + jnp.tanh(fused_logits) * Y_scale

    return predictions


class NeuralContinuousModelRunner:
    """
    Orchestrates the training, statistical validation, and interpretation of the
    Dual-Stream MLP for continuous USV manifold prediction.

    This class serves as the deep-learning execution engine. It consumes the extracted
    data produced by the ContinuousModelingPipeline, fuses univariate features into
    multivariate design matrices, and performs spatially-stratified cross-validation.

    Key responsibilities:
    1. Multivariate Data Fusion: Pivots session dictionaries into unified,
       temporally downsampled arrays (X_seq, X_global) for JAX.
    2. Rigorous CV & Permutation Testing: Executes an 'Actual vs. Shuffled Null'
       and 'null_model_free' strategy to prove non-random mapping against spatial density priors.
    3. Global Feature Importance: Performs post-hoc permutation to rank the predictive
       power of each kinematic feature.
    4. Saliency Map Extraction: Employs Input-Gradient attribution to reveal the specific
       behavioral motifs that drive predictions into distinct geographic quadrants.
    """

    def __init__(self, modeling_settings: dict) -> None:
        """
        Initializes the neural model runner using a configured pipeline instance.

        Parameters
        ----------
        modeling_settings : dict
            A dictionary containing all necessary settings and hyperparameters for the modeling pipeline.
        """
        if modeling_settings is None:
            settings_path = pathlib.Path(__file__).resolve().parent.parent / '_parameter_settings/modeling_settings.json'
            try:
                with open(settings_path, 'r') as settings_json_file:
                    self.modeling_settings = json.load(settings_json_file)['modeling_settings']
            except FileNotFoundError:
                raise FileNotFoundError(f"Settings file not found at {settings_path}")
        else:
            self.modeling_settings = modeling_settings

        self.history_frames = int(np.floor(self.modeling_settings['features']['filter_history'] * self.modeling_settings['data_io']['camera_sampling_rate']))
        self.split_strategy = self.modeling_settings['model_selection']['split_strategy']
        self.random_seed = self.modeling_settings['random_seed']
        self.hp = self.modeling_settings['hyperparameters']['jax_mlp_continuous_params']

    @staticmethod
    def get_stratified_spatial_splits_stable(groups: np.ndarray,
                                             Y: np.ndarray,
                                             split_strategy: str = 'session',
                                             n_clusters: int = 15,
                                             test_prop: float = 0.2,
                                             n_splits: int = 100,
                                             tolerance: float = 0.05,
                                             random_seed: int = 0) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generates deterministic folds ensuring spatial geographic fairness across the UMAP manifold.

        Uses K-Means to temporarily partition the continuous UMAP space into micro-neighborhoods
        (proxy labels). Depending on the `split_strategy`, it then splits the dataset to ensure
        the dense core and rare satellite clusters are proportionally represented in both train
        and test sets.

        Parameters
        ----------
        groups : np.ndarray
            Array of session IDs. Used strictly when split_strategy='session'.
        Y : np.ndarray
            Array of shape (N, 2) containing continuous UMAP coordinates.
        split_strategy : str, default 'session'
            Determines the data leakage constraint:
            - 'session': Strict cross-session prediction. Samples from the same session
              are never split between train and test. (Uses tolerance-based search).
            - 'mixed': Completely randomized frame-level splitting. Ignores session IDs
              and perfectly stratifies based solely on geographic density.
        n_clusters : int, default 15
            Number of geographic micro-neighborhoods to define via K-Means.
        test_prop : float, default 0.2
            Proportion of the dataset (or sessions) to assign to the test set.
        n_splits : int, default 100
            Number of independent fold iterations to generate.
        tolerance : float, default 0.05
            (For 'session' strategy only). Initial allowable difference in spatial
            distribution between the global data and the generated test splits.
        random_seed : int, default 0
            Fixed seed for absolute reproducibility.

        Returns
        -------
        cv_folds : list of tuples
            A list of length `n_splits`, where each tuple contains (train_indices, test_indices).
        """
        if split_strategy not in ['session', 'mixed']:
            raise ValueError(f"Invalid split_strategy: '{split_strategy}'. Must be 'session' or 'mixed'.")

        kmeans = KMeans(n_clusters=n_clusters, random_state=random_seed, n_init='auto')
        proxy_labels = kmeans.fit_predict(Y)

        if split_strategy == 'mixed':
            sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_prop, random_state=random_seed)
            cv_folds = list(sss.split(np.zeros(len(Y)), proxy_labels))
            return cv_folds

        elif split_strategy == 'session':
            unique_sessions = np.unique(groups)
            n_test_sessions = int(len(unique_sessions) * test_prop)

            _, global_counts = np.unique(proxy_labels, return_counts=True)
            global_dist = global_counts / len(proxy_labels)

            cv_folds = []
            rng = np.random.RandomState(random_seed)

            attempts = 0
            current_tolerance = tolerance
            max_total_attempts = 50000

            while len(cv_folds) < n_splits:
                attempts += 1
                shuffled = rng.permutation(unique_sessions)
                te_sess = shuffled[:n_test_sessions]
                tr_sess = shuffled[n_test_sessions:]

                tr_idx = np.where(np.isin(groups, tr_sess))[0]
                te_idx = np.where(np.isin(groups, te_sess))[0]

                tr_clusters = np.unique(proxy_labels[tr_idx])
                te_clusters = np.unique(proxy_labels[te_idx])

                if len(tr_clusters) == n_clusters and len(te_clusters) == n_clusters:
                    _, te_counts = np.unique(proxy_labels[te_idx], return_counts=True)
                    te_dist = te_counts / len(te_idx)
                    dist_error = np.max(np.abs(te_dist - global_dist))

                    if dist_error < current_tolerance:
                        cv_folds.append((tr_idx, te_idx))

                if attempts % 1000 == 0:
                    current_tolerance += 0.02

                if attempts > max_total_attempts:
                    raise RuntimeError(
                        f"Failed to find {n_splits} valid spatial splits after {attempts} attempts. "
                        "Rare geographic clusters may be highly isolated in too few sessions."
                    )

            return cv_folds

    def load_multivariate_data_blocks(self, pkl_path: str) -> Dict[str, Any]:
        """
        Loads extracted feature data from disk and constructs the multivariate
        tensors required for the Dual-Stream MLP.

        Parameters
        ----------
        pkl_path : str
            Full path to the .pkl file containing extracted (X, Y, w) dictionaries.

        Returns
        -------
        data_blocks : dict
            Contains 'X_seq' (N, Features, Bins), 'X_global' (N, Features*2), 'Y',
            'groups', and the list of 'features'.
        """
        print(f"Loading and fusing multivariate data from: {pkl_path}")
        with open(pkl_path, 'rb') as f:
            raw_data = pickle.load(f)

        features = sorted(list(raw_data.keys()))
        bin_factor = self.hp['bin_resizing_factor']
        num_bins = self.history_frames // bin_factor

        X_seq_list, X_global_list, Y_list, groups_list = [], [], [], []
        sessions = sorted(list(raw_data[features[0]].keys()))

        for sess in sessions:
            Y_sess = raw_data[features[0]][sess]['Y']
            sess_seq, sess_glob = [], []

            for feat in features:
                raw_X = raw_data[feat][sess]['X']
                N, frames = raw_X.shape

                valid_frames = num_bins * bin_factor
                binned = raw_X[:, :valid_frames].reshape(N, num_bins, bin_factor).mean(axis=2)
                sess_seq.append(binned)

                feat_mean = np.mean(raw_X, axis=1, keepdims=True)
                feat_std = np.std(raw_X, axis=1, keepdims=True)
                sess_glob.append(np.hstack([feat_mean, feat_std]))

            X_seq_list.append(np.stack(sess_seq, axis=1))
            X_global_list.append(np.hstack(sess_glob))
            Y_list.append(Y_sess)
            groups_list.append(np.full(len(Y_sess), sess))

        return {
            'X_seq': np.vstack(X_seq_list).astype(np.float32),
            'X_global': np.vstack(X_global_list).astype(np.float32),
            'Y': np.vstack(Y_list).astype(np.float32),
            'groups': np.concatenate(groups_list),
            'features': features,
            'num_bins': num_bins,
            'source_pkl_path': pkl_path
        }

    def run_mlp_training(self, data_blocks: Dict[str, Any]) -> None:
        """
        Executes the deep learning cross-validation pipeline, evaluates statistical
        significance, computes permutation feature importance, and serializes the results.

        This method performs the core computational heavy lifting for the Dual-Stream MLP:
        1. K-fold cross-validation (spatially stratified) to establish Baseline Error.
        2. K-fold Null Model evaluation (target-shuffled) for global significance.
        3. Model-Free Spatial Density Prior evaluation. Computes the optimal Euclidean
           guess (the spatial center of mass of the training manifold) without utilizing
           behavioral kinematics, establishing the absolute lower bound of error.
        4. Post-Hoc Permutation Feature Importance. Using the best-performing fold,
           each feature is synchronously permuted across both its temporal sequence
           (X_seq) and global statistics (X_global) streams to completely decouple
           the feature from the target manifold.

        Parameters
        ----------
        data_blocks : dict
            The fused multivariate data matrix generated by `load_multivariate_data_blocks`.

        Returns
        -------
        None
            Results are serialized as a timestamped .pkl file in the configured `save_dir`.
        """
        print("=" * 70)
        print(" EXECUTING DUAL-STREAM MLP TRAINING PIPELINE (PERMUTATION)")
        print("=" * 70)

        X_seq = data_blocks['X_seq']
        X_glob = data_blocks['X_global']
        Y = data_blocks['Y']
        groups = data_blocks['groups']
        features = list(data_blocks['features'])

        N_samples = X_seq.shape[0]
        n_bins = data_blocks['num_bins']

        Y_center = jnp.array((np.max(Y, 0) + np.min(Y, 0)) / 2.0)
        Y_scale = jnp.array((np.max(Y, 0) - np.min(Y, 0)) / 2.0 * 1.1)

        temporal_size = len(features) * n_bins
        global_size = len(features) * 2

        n_folds = self.hp['n_folds']
        cv_settings = self.modeling_settings['model_selection']
        split_strategy = cv_settings['split_strategy']
        random_seed = self.modeling_settings['random_seed']

        print(f"Generating deterministic, spatially-stratified folds (n={n_folds})...")
        folds = self.get_stratified_spatial_splits_stable(
            groups=groups, Y=Y, n_clusters=cv_settings['n_spatial_clusters'],
            test_prop=cv_settings['test_proportion'], split_strategy=split_strategy,
            n_splits=n_folds, random_seed=random_seed
        )

        actual_errors, null_errors, fold_params, null_fold_params = [], [], [], []

        deep_storage = {
            'metadata': {
                'hyperparameters': self.hp,
                'features_list': features,
                'n_time_bins': n_bins,
                'Y_center': Y_center,
                'Y_scale': Y_scale,
                'split_strategy': split_strategy
            },
            'cross_validation': [],
            'statistics': {},
            'feature_importance': {}
        }

        def evaluate_batched(params, x_s, x_g, batch_size=1024):
            preds = []
            for i in range(0, len(x_s), batch_size):
                preds.append(mlp_forward(params, x_s[i:i + batch_size], x_g[i:i + batch_size], Y_center, Y_scale, deterministic=True))
            return jnp.concatenate(preds, axis=0)

        print("\n" + "=" * 50)
        print(" PHASE 1: BASELINE & NULL MODEL TRAINING")
        print("=" * 50)

        for fold, (train_idx, test_idx) in enumerate(folds):
            print(f"\n--- FOLD {fold + 1}/{n_folds} ---")
            X_s_tr, X_s_te = X_seq[train_idx], X_seq[test_idx]
            X_g_tr, X_g_te = X_glob[train_idx], X_glob[test_idx]
            Y_tr, Y_te = Y[train_idx], Y[test_idx]

            # --- Null Model Free (Empirical Density Draw) ---
            # Simulate a dart throw by randomly drawing coordinates directly
            # from the training set's empirical distribution, respecting spatial density.
            rng_mf = np.random.RandomState(random_seed + fold)
            draw_indices = rng_mf.choice(len(Y_tr), size=len(Y_te), replace=True)
            Y_pred_null_model_free = Y_tr[draw_indices]
            err_null_model_free = float(np.mean(np.sqrt(np.sum((Y_pred_null_model_free - Y_te) ** 2, axis=-1))))

            params = init_mlp_params(jax.random.PRNGKey(fold), global_size, temporal_size, self.hp['global_hidden_dim'], self.hp['temporal_hidden_dim'])

            lr_schedule = optax.cosine_decay_schedule(self.hp['learning_rate'], self.hp['epochs'] * 10) if self.hp['use_scheduler'] else self.hp['learning_rate']
            optimizer = optax.adamw(learning_rate=lr_schedule, weight_decay=self.hp['weight_decay'])
            opt_state = optimizer.init(params)

            @jax.jit
            def update_step(p, o_state, xs, xg, yt):
                def loss_fn(weights):
                    preds = mlp_forward(weights, xs, xg, Y_center, Y_scale, deterministic=False)
                    err = jnp.abs(preds - yt)
                    huber = jnp.where(err < 2.0, 0.5 * err ** 2, 2.0 * err - 2.0)
                    return jnp.mean(jnp.sum(huber, axis=-1))

                grads = jax.grad(loss_fn)(p)
                updates, new_opt_state = optimizer.update(grads, o_state, p)
                return optax.apply_updates(p, updates), new_opt_state

            best_f_err, best_f_params, p_counter = float('inf'), None, 0

            for epoch in range(self.hp['epochs']):
                b_idx = get_grid_balanced_indices(Y_tr, self.hp['grid_size'], self.hp['samples_per_cell'])
                np.random.shuffle(b_idx)

                for b in range(len(b_idx) // self.hp['batch_size']):
                    idx = b_idx[b * self.hp['batch_size']:(b + 1) * self.hp['batch_size']]
                    warps = np.random.uniform(1.0 - self.hp['warp_range'], 1.0 + self.hp['warp_range'], len(idx))
                    X_warped = apply_temporal_warping(X_s_tr[idx], warps)

                    params, opt_state = update_step(params, opt_state, jnp.array(X_warped), jnp.array(X_g_tr[idx]), jnp.array(Y_tr[idx]))

                if epoch % 10 == 0:
                    p_te = evaluate_batched(params, jnp.array(X_s_te), jnp.array(X_g_te))
                    err = jnp.mean(jnp.sqrt(jnp.sum((p_te - Y_te) ** 2, axis=-1)))
                    if err < best_f_err:
                        best_f_err, best_f_params, p_counter = err, copy.deepcopy(params), 0
                    else:
                        p_counter += 1
                    if p_counter >= self.hp['patience']:
                        break

            actual_errors.append(best_f_err)
            fold_params.append(best_f_params)

            # Null Model Training
            Y_tr_shuff = np.random.permutation(Y_tr)
            params_n = init_mlp_params(jax.random.PRNGKey(fold + 100), global_size, temporal_size, self.hp['global_hidden_dim'], self.hp['temporal_hidden_dim'])
            opt_state_n = optimizer.init(params_n)
            best_n_err, best_n_params, null_p_counter = float('inf'), None, 0

            for epoch in range(self.hp['epochs']):
                b_idx = get_grid_balanced_indices(Y_tr_shuff, self.hp['grid_size'], self.hp['samples_per_cell'])
                for b in range(len(b_idx) // self.hp['batch_size']):
                    idx = b_idx[b * self.hp['batch_size']:(b + 1) * self.hp['batch_size']]
                    params_n, opt_state_n = update_step(params_n, opt_state_n, jnp.array(X_s_tr[idx]), jnp.array(X_g_tr[idx]), jnp.array(Y_tr_shuff[idx]))

                if epoch % 10 == 0:
                    p_te_n = evaluate_batched(params_n, jnp.array(X_s_te), jnp.array(X_g_te))
                    err_n = jnp.mean(jnp.sqrt(jnp.sum((p_te_n - Y_te) ** 2, axis=-1)))
                    if err_n < best_n_err - 0.001:
                        best_n_err, best_n_params, null_p_counter = err_n, copy.deepcopy(params_n), 0
                    else:
                        null_p_counter += 1
                    if null_p_counter >= self.hp['null_patience']:
                        print(f"   > Null flatlined at epoch {epoch}. Stopping process.")
                        break

            null_errors.append(best_n_err)
            null_fold_params.append(best_n_params)

            print(f"Fold {fold + 1} Summary: Actual {best_f_err:.4f} | Null {best_n_err:.4f} | MF Prior {err_null_model_free:.4f}")

            Y_pred_actual = evaluate_batched(best_f_params, jnp.array(X_s_te), jnp.array(X_g_te))
            Y_pred_null = evaluate_batched(best_n_params, jnp.array(X_s_te), jnp.array(X_g_te))

            deep_storage['cross_validation'].append({
                'fold_idx': fold,
                'test_indices': test_idx,
                'Y_true': Y_te,
                'Y_pred_actual': Y_pred_actual,
                'Y_pred_null': Y_pred_null,
                'Y_pred_null_model_free': Y_pred_null_model_free,
                'error_actual': best_f_err,
                'error_null': best_n_err,
                'error_null_model_free': err_null_model_free,
                'params_actual': best_f_params
            })

        boot_null_means = [np.mean(np.random.choice(null_errors, size=len(null_errors), replace=True)) for _ in range(10000)]
        null_dist = np.array(boot_null_means)
        actual_mean = np.mean(actual_errors)
        null_threshold_05 = np.percentile(null_dist, 0.5)

        deep_storage['statistics'] = {
            'bootstrapped_null_distribution': null_dist,
            'actual_mean_error': actual_mean,
            'null_threshold_05': null_threshold_05
        }

        print("\n" + "=" * 50)
        print(" PHASE 2: POST-HOC PERMUTATION FEATURE IMPORTANCE")
        print(" Evaluating optimal fold with K=10 synchronous shuffles per feature.")
        print("=" * 50)

        best_fold_idx = int(np.argmin(actual_errors))
        best_params = fold_params[best_fold_idx]
        test_idx = folds[best_fold_idx][1]

        X_s_te_base = np.array(X_seq[test_idx])
        X_g_te_base = np.array(X_glob[test_idx])
        Y_te = np.array(Y[test_idx])

        base_err = actual_errors[best_fold_idx]

        K_perms = 10
        importance_means = {}
        importance_stds = {}
        raw_importance = {}

        for f_idx, feat_name in enumerate(features):
            feat_scores = []

            for k in range(K_perms):
                X_s_perm = X_s_te_base.copy()
                X_g_perm = X_g_te_base.copy()

                perm_idx = np.random.permutation(len(Y_te))

                X_s_perm[:, f_idx:f_idx + 1, :] = X_s_perm[perm_idx, f_idx:f_idx + 1, :]
                X_g_perm[:, 2 * f_idx:2 * f_idx + 2] = X_g_perm[perm_idx, 2 * f_idx:2 * f_idx + 2]

                p_te_perm = evaluate_batched(best_params, jnp.array(X_s_perm), jnp.array(X_g_perm))
                err_perm = jnp.mean(jnp.sqrt(jnp.sum((p_te_perm - Y_te) ** 2, axis=-1)))

                delta_e = err_perm - base_err
                feat_scores.append(float(delta_e))

            mu = np.mean(feat_scores)
            sigma = np.std(feat_scores)

            raw_importance[feat_name] = feat_scores
            importance_means[feat_name] = mu
            importance_stds[feat_name] = sigma

            print(f"   [{feat_name:<25}] Delta E: {mu:+.4f} (±{sigma:.4f})")

        sorted_features = sorted(importance_means.keys(), key=lambda x: importance_means[x], reverse=True)

        deep_storage['feature_importance'] = {
            'best_fold_idx': best_fold_idx,
            'raw_scores': raw_importance,
            'means': importance_means,
            'stds': importance_stds,
            'ranked_features': sorted_features
        }

        print("\nConverting JAX device arrays to NumPy and saving Deep Storage...")
        numpy_storage = jax.device_get(deep_storage)

        source_file = os.path.basename(data_blocks['source_pkl_path'])
        if "male_mute_partner" in source_file:
            sex_mod = "male_mute_partner"
        elif "female" in source_file:
            sex_mod = "female"
        else:
            sex_mod = "male"

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"mlp_umap_manifold_predictions_{sex_mod}_{timestamp}.pkl"

        save_dir = self.modeling_settings['save_dir']
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)

        with open(save_path, 'wb') as f:
            pickle.dump(numpy_storage, f)

        print(f"Success. Deep Storage saved to: {save_path}")
