"""
@author: bartulem
Module for continuous probabilistic USV modeling (based on JAX, assumes GPU usage).

This module provides a specialized pipeline for mapping behavioral kinematics to the
continuous topological space of the vocal repertoire. Rather than imposing arbitrary
discrete boundaries, it predicts the full conditional probability density (modeled
as a Bivariate Gaussian) of an upcoming vocalization's (x, y) coordinates on a
2D acoustic manifold (UMAP).

Key scientific capabilities:
1.  Probabilistic target extraction: Extracts continuous spatial coordinate pairs
    for every valid bout, enabling the model to learn a continuous mapping from
    behavioral history to acoustic outcomes alongside structural uncertainty.
2.  Geographic fairness (Inverse Density Weighting): Computes a Gaussian Kernel
    Density Estimate (KDE) over the acoustic space to apply inverse density sample
    weights. This prevents the model from ignoring rare acoustic "satellite islands"
    in favor of the dense manifold core.
3.  Spatial cross-validation: Implements a deterministic K-Means spatial proxy
    strategy for Stratified Group K-Fold splitting. This strictly prevents session
    leakage while guaranteeing that both rare and common acoustic regions are
    proportionally represented in all training and testing folds.
4.  Rigorous Null Baselines: Compares model performance against both a within-session
    shuffled control ('null') and a purely spatial density-based prior ('null_model_free')
    to strictly validate behavioral predictive power.
"""

import json
import numpy as np
import os
import pathlib
import pickle
import re
from datetime import datetime
from scipy.stats import gaussian_kde, wilcoxon
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedShuffleSplit
from typing import Any, List, Tuple
from tqdm import tqdm

from .load_input_files import load_behavioral_feature_data, find_usv_categories
from .modeling_utils import (
    prepare_modeling_sessions,
    resolve_mouse_roles,
    select_kinematic_columns,
    build_vocal_signal_columns,
    harmonize_session_columns,
    zscore_features_across_sessions,
)
from .jax_bivariate_gaussian_regression import SmoothBivariateGaussianRegression
from ..analyses.compute_behavioral_features import FeatureZoo


def compute_inverse_density_weights(Y: np.ndarray,
                                    clip_percentile: float = 95.0,
                                    epsilon: float = 1e-5) -> np.ndarray:
    """
    Computes inverse density sample weights using Gaussian Kernel Density Estimation (KDE).

    This neutralizes the topographical bias of the UMAP space. Rare "satellite"
    vocalizations receive mathematically higher weights than syllables in the
    dense manifold core, ensuring the optimizer treats all geographic regions equally.

    Parameters
    ----------
    Y : np.ndarray
        Array of shape (N, 2) containing continuous UMAP coordinates.
    clip_percentile : float, default 95.0
        The percentile at which to cap maximum weights, preventing single extreme
        outliers from mathematically dominating the loss landscape.
    epsilon : float, default 1e-5
        Small constant added to the denominator to prevent division by zero.

    Returns
    -------
    weights : np.ndarray
        Array of shape (N,) containing normalized sample weights. The mean
        of the weights is exactly 1.0, preserving the global scale of the loss.
    """

    kde = gaussian_kde(Y.T)
    densities = kde(Y.T)

    raw_weights = 1.0 / (densities + epsilon)

    max_allowed_weight = np.percentile(raw_weights, clip_percentile)
    clipped_weights = np.clip(raw_weights, a_min=None, a_max=max_allowed_weight)

    normalized_weights = clipped_weights / np.mean(clipped_weights)

    return normalized_weights


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


class ContinuousModelingPipeline(FeatureZoo):

    def __init__(self, modeling_settings_dict: dict = None, **kwargs):
        """
        Initializes the ContinuousModelingPipeline class.

        This constructor establishes the configuration environment required for extracting
        continuous probabilistic modeling data. It strictly loads the modeling settings
        (either via a provided dictionary or the default JSON file), validates the presence
        of required nested parameters, and computes the fixed-length kinematic history
        window (in frames) based on the exact camera sampling rate and physical time constraints.
        It also inherits from `FeatureZoo` to ensure access to standardized behavioral
        feature computations.

        Parameters
        ----------
        modeling_settings_dict : dict, optional
            A nested dictionary containing the complete modeling configuration (e.g., IO paths,
            hyperparameters, continuous feature bounds). If None, defaults to loading from
            `_parameter_settings/modeling_settings.json`.
        **kwargs : dict
            Additional keyword arguments to dynamically set as instance attributes.
        """

        if modeling_settings_dict is None:
            settings_path = pathlib.Path(__file__).resolve().parent.parent / '_parameter_settings/modeling_settings.json'
            try:
                with open(settings_path, 'r') as f:
                    self.modeling_settings = json.load(f)
            except FileNotFoundError:
                raise FileNotFoundError(f"Settings file not found at {settings_path}")
        else:
            self.modeling_settings = modeling_settings_dict

        if 'feature_boundaries' in self.modeling_settings:
            self.feature_boundaries = self.modeling_settings['feature_boundaries']

        try:
            cam_rate = self.modeling_settings['io']['camera_sampling_rate']
            hist_sec = self.modeling_settings['model_params']['filter_history']
            self.history_frames = int(np.floor(cam_rate * hist_sec))
            print(f"Continuous Pipeline Init: History frames calculated: {self.history_frames} (for {hist_sec}s at {cam_rate}fps)")
        except KeyError as e:
            raise KeyError(f"Critical setting missing for history calculation: {e}")

        FeatureZoo.__init__(self)

        for kw_arg, kw_val in kwargs.items():
            self.__dict__[kw_arg] = kw_val

    def extract_and_save_continuous_data(self) -> None:
        """
        Extracts, processes, and saves (X, Y, w) data matrices for continuous probabilistic regression.

        This pipeline orchestrates the end-to-end data engineering required to map behavioral
        kinematics to a continuous acoustic manifold. Unlike categorical classification, this
        method preserves the true topological reality of the vocal repertoire.

        Process Outline:
        1. Target extraction: Identifies all valid USVs across specified sessions, verifies they
           meet the historical time constraints, and extracts their continuous 2D UMAP coordinates (Y).
        2. Geographic fairness (KDE weights): Computes a global Gaussian Kernel Density Estimate
           across the universal Y manifold to generate normalized inverse-density sample weights (w).
           This mathematically neutralizes the topographical bias of the dense acoustic core.
        3. Feature standardization: Filters the behavioral dataset for specific self/partner
           predictors, injects continuous vocal history, and standardizes (z-scores) all features
           globally across the pooled sessions.
        4. Epoch slicing: Slices the z-scored behavioral traces into fixed-length arrays (X)
           representing the precise temporal kinematics preceding every USV event.

        Returns
        -------
        None
            Saves a highly structured `.pkl` file to the configured `save_dir`. The output is a
            nested dictionary explicitly organized for the downstream JAX engine:
            `data[feature_name][session_id] = {'X': array, 'Y': array, 'w': array}`

            - 'X': Predictor history matrix of shape (n_samples, history_frames).
            - 'Y': Target spatial matrix of shape (n_samples, 2) containing (umap_x, umap_y).
            - 'w': Inverse-density sample weights of shape (n_samples,).
        """

        txt_sessions = prepare_modeling_sessions(self.modeling_settings)

        print("Loading behavioral feature data...")
        beh_data_dict, cam_fps_dict, mouse_names_dict = load_behavioral_feature_data(
            behavior_file_paths=txt_sessions,
            csv_sep=self.modeling_settings['io']['csv_separator']
        )

        voc_settings = self.modeling_settings['vocal_features']
        kin_settings = self.modeling_settings['kinematic_features']

        voc_mode = voc_settings['usv_predictor_type']
        smooth_sd = voc_settings['usv_predictor_smoothing_sd']
        column_name_cats = voc_settings['usv_category_column_name']
        noise_cats = voc_settings['usv_noise_categories']

        filter_hist = self.modeling_settings['model_params']['filter_history']
        pred_idx = self.modeling_settings['model_params']['model_predictor_mouse_index']
        targ_idx = abs(pred_idx - 1)

        usv_data_dict = find_usv_categories(
            root_directories=txt_sessions,
            mouse_ids_dict=mouse_names_dict,
            camera_fps_dict=cam_fps_dict,
            features_dict=beh_data_dict,
            csv_sep=self.modeling_settings['io']['csv_separator'],
            target_category=None,
            category_column=column_name_cats,
            filter_history=filter_hist,
            vocal_output_type=voc_mode,
            proportion_smoothing_sd=smooth_sd,
            noise_vocal_categories=noise_cats
        )

        print("Extracting continuous targets and computing Global Inverse Density Weights...")
        continuous_targets_dict = {}
        all_valid_Y_list = []

        for sess_id in list(beh_data_dict.keys()):
            if sess_id not in mouse_names_dict:
                continue

            targ_name = mouse_names_dict[sess_id][targ_idx]

            if sess_id not in usv_data_dict or targ_name not in usv_data_dict[sess_id]:
                continue

            if 'continuous_onsets' not in usv_data_dict[sess_id][targ_name] or 'continuous_targets' not in usv_data_dict[sess_id][targ_name]:
                continue

            onsets = usv_data_dict[sess_id][targ_name]['continuous_onsets']
            targets = usv_data_dict[sess_id][targ_name]['continuous_targets']

            if onsets is None or targets is None:
                continue

            fps = cam_fps_dict[sess_id]
            max_frame_idx = beh_data_dict[sess_id].height - 1

            valid_onsets = []
            valid_targets = []

            frame_indices = np.round(onsets * fps).astype(int)
            for i, f_idx in enumerate(frame_indices):
                if self.history_frames <= f_idx <= max_frame_idx:
                    valid_onsets.append(f_idx)
                    valid_targets.append(targets[i])

            if valid_onsets:
                valid_onsets_arr = np.array(valid_onsets, dtype=int)
                valid_targets_arr = np.array(valid_targets, dtype=np.float32)

                continuous_targets_dict[sess_id] = {
                    'onsets': valid_onsets_arr,
                    'targets': valid_targets_arr
                }
                all_valid_Y_list.append(valid_targets_arr)

        if not all_valid_Y_list:
            raise ValueError("No valid continuous targets extracted. Check UMAP data.")

        global_Y_matrix = np.vstack(all_valid_Y_list)
        global_weights = compute_inverse_density_weights(global_Y_matrix)

        ptr = 0
        for sess_id in continuous_targets_dict.keys():
            n_sess_usvs = len(continuous_targets_dict[sess_id]['targets'])
            continuous_targets_dict[sess_id]['weights'] = global_weights[ptr: ptr + n_sess_usvs]
            ptr += n_sess_usvs

        print("Filtering features and injecting vocal history...")
        processed_beh_data = {}

        for sess_id in list(beh_data_dict.keys()):
            if sess_id not in mouse_names_dict or len(mouse_names_dict[sess_id]) < 2:
                continue

            current_df = beh_data_dict[sess_id]
            _, _, pred_name, targ_name = resolve_mouse_roles(
                modeling_settings=self.modeling_settings,
                mouse_names_dict=mouse_names_dict,
                session_id=sess_id
            )

            cols_to_keep = select_kinematic_columns(
                session_df_columns=current_df.columns,
                target_name=targ_name,
                predictor_name=pred_name,
                kin_settings=kin_settings,
                predictor_idx=pred_idx
            )

            new_voc_cols, _ = build_vocal_signal_columns(
                usv_data_dict=usv_data_dict,
                session_id=sess_id,
                target_name=targ_name,
                predictor_name=pred_name,
                voc_settings=voc_settings
            )

            current_df = current_df.select(sorted(list(set(cols_to_keep))))

            if new_voc_cols:
                current_df = current_df.with_columns(new_voc_cols)

            processed_beh_data[sess_id] = current_df

        print("Standardizing columns and z-scoring...")
        processed_beh_data, revised_predictors = harmonize_session_columns(
            processed_beh_dict=processed_beh_data,
            mouse_names_dict=mouse_names_dict,
            target_idx=targ_idx,
            predictor_idx=pred_idx
        )

        feature_bounds = self.feature_boundaries if hasattr(self, 'feature_boundaries') else {}

        processed_beh_data = zscore_features_across_sessions(
            processed_beh_dict=processed_beh_data,
            suffixes=revised_predictors,
            feature_bounds=feature_bounds
        )

        print("Extracting epochs and saving to disk...")
        final_data = {}

        for sess_id, data_packet in tqdm(continuous_targets_dict.items(), desc='Slicing Epochs'):
            if sess_id not in processed_beh_data:
                continue

            sess_df = processed_beh_data[sess_id]
            t_name = mouse_names_dict[sess_id][targ_idx]
            p_name = mouse_names_dict[sess_id][pred_idx]

            onsets = data_packet['onsets']
            Y_targets = data_packet['targets']
            weights = data_packet['weights']

            slice_starts = onsets - self.history_frames
            slice_ends = onsets
            num_samples = len(onsets)

            for col_name in sess_df.columns:
                suffix = col_name.split('.')[-1]
                if col_name.startswith(f"{t_name}."):
                    feat_key = f"self.{suffix}"
                elif col_name.startswith(f"{p_name}."):
                    feat_key = f"other.{suffix}"
                else:
                    feat_key = col_name

                if feat_key not in final_data:
                    final_data[feat_key] = {}

                X_arr = np.empty((num_samples, self.history_frames), dtype=np.float32)
                col_values = sess_df[col_name].to_numpy()

                for i in range(num_samples):
                    s, e = slice_starts[i], slice_ends[i]
                    chunk = col_values[s:e]
                    if np.isnan(chunk).any():
                        chunk = np.nan_to_num(chunk, nan=0.0)
                    X_arr[i, :] = chunk

                final_data[feat_key][sess_id] = {
                    'X': X_arr,
                    'Y': Y_targets,
                    'w': weights
                }

        if not final_data:
            print("No valid data extracted. Aborting save.")
            return

        feature_names = sorted(list(final_data.keys()))
        n_feats = len(feature_names)
        sample_feat = feature_names[0] if n_feats > 0 else None

        if sample_feat:
            n_sessions = len(final_data[sample_feat])
            all_Y = np.vstack([final_data[sample_feat][s]['Y'] for s in final_data[sample_feat]])
            all_w = np.concatenate([final_data[sample_feat][s]['w'] for s in final_data[sample_feat]])
            total_samples = len(all_Y)

            print("\n" + "=" * 60)
            print(f"CONTINUOUS PROBABILISTIC EXTRACTION SUMMARY")
            print("=" * 60)
            print(f"Total Unique Features:  {n_feats}")
            print(f"Total Valid Sessions:   {n_sessions}")
            print(f"Total USVs included (N): {total_samples}")
            print(f"Global Weights Mean:    {np.mean(all_w):.4f}")
            print(f"Global Weights Max:     {np.max(all_w):.4f}")
            print("-" * 60)

            print("EXTRACTED FEATURES:")
            for i in range(0, len(feature_names), 4):
                print("  " + "".join(f"{name: <25}" for name in feature_names[i:i + 4]))
            print("=" * 60 + "\n")

        target_mouse_sex = 'male' if targ_idx == 0 else 'female'
        fname = f"modeling_UMAP_manifold_position_{target_mouse_sex}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_hist{filter_hist}s.pkl"
        save_dir = self.modeling_settings['io']['save_directory']
        save_path = os.path.join(save_dir, fname)

        os.makedirs(save_dir, exist_ok=True)

        print(f"Saving continuous extraction results to:\n{save_path}")
        with open(save_path, 'wb') as f:
            pickle.dump(final_data, f)
        print("[+] Save Complete.")


class ContinuousModelRunner:
    """
    Orchestrates the training and statistical evaluation of continuous probabilistic
    USV models.

    This class serves as the execution engine for the continuous modeling phase. It
    consumes the extracted (X, Y, w) data produced by the `ContinuousModelingPipeline`,
    transforms it into binned univariate design matrices, and performs rigorous
    cross-validation using the JAX-accelerated Bivariate Gaussian estimator.

    Key responsibilities:
    1. Data transformation: Pivots nested session dictionaries into unified,
       temporally downsampled (binned) arrays for efficient gradient descent.
    2. Experimental control: Evaluates features against an 'actual' framework, a
       shuffled 'null' framework, and a 'null_model_free' spatial density prior.
    3. Spatial Validation: Utilizes deterministic K-Means geographic clustering
       to ensure spatially fair cross-validation folds.
    4. Deep metadata storage: Persists fold-level probabilistic predictions (mu, sigma, rho),
       performance metrics (NLL, Mahalanobis, Euclidean), and exact test indices.
    """

    def __init__(self, pipeline_instance: Any) -> None:
        """
        Initializes the model runner using a configured pipeline instance.

        Parameters
        ----------
        pipeline_instance : ContinuousModelingPipeline
            An instance of the extraction class which holds the
            'modeling_settings' dictionary and calculated attributes.
        """
        self.modeling_settings = pipeline_instance.modeling_settings

        if hasattr(pipeline_instance, 'feature_boundaries'):
            self.feature_boundaries = pipeline_instance.feature_boundaries

    @staticmethod
    def load_univariate_data_blocks(pkl_path: str, bin_size: int = 10) -> dict:
        """
        Loads extracted feature data from disk and applies temporal downsampling (binning).

        This method aggregates session-specific behavioral windows into a unified
        design matrix for each feature. Crucially, it applies temporal binning to
        reduce the dimensionality of the predictor space, acting as a low-pass filter
        on the behavioral time-series.

        By averaging frames into bins (e.g., bin_size=10), the model focuses on slower,
        behaviorally meaningful kinematic envelopes rather than high-frequency jitter.
        This drastically reduces the number of parameters the JAX solver must optimize.

        Parameters
        ----------
        pkl_path : str
            The full path to the .pkl file containing the extracted (X, Y, w)
            dictionaries from the ContinuousModelingPipeline.
        bin_size : int, optional
            The resizing factor for temporal downsampling. A value of 10
            means every 10 frames are averaged into 1 bin. Default is 10.

        Returns
        -------
        data_blocks : dict
            A dictionary keyed by feature name (e.g., 'self.speed'). Each value is
            another dictionary containing:
            - 'X' : np.ndarray (n_samples, n_binned_time)
                The flattened and binned behavioral history matrix.
            - 'Y' : np.ndarray (n_samples, 2)
                The continuous (x, y) UMAP targets.
            - 'w' : np.ndarray (n_samples,)
                The inverse-density sample weights.
            - 'groups' : np.ndarray (n_samples,)
                String IDs used for session-aware splitting and null shuffling.
            - 'n_time_bins' : int
                The final count of temporal predictors after binning.
        """

        print(f"Loading and binning continuous data (bin_size={bin_size}) from: {pkl_path}")

        with open(pkl_path, 'rb') as f:
            raw_data = pickle.load(f)

        sorted_features = sorted(list(raw_data.keys()))
        data_blocks = {}

        for feat in sorted_features:
            X_list, Y_list, w_list, groups_list = [], [], [], []
            sessions = sorted(list(raw_data[feat].keys()))

            for sess_id in sessions:
                X_sess = raw_data[feat][sess_id]['X']
                Y_sess = raw_data[feat][sess_id]['Y']
                w_sess = raw_data[feat][sess_id]['w']

                N, T = X_sess.shape
                if bin_size > 1:
                    new_T = T // bin_size
                    X_sess = X_sess[:, :new_T * bin_size].reshape(N, new_T, bin_size).mean(axis=2)

                X_list.append(X_sess)
                Y_list.append(Y_sess)
                w_list.append(w_sess)
                groups_list.append(np.full(len(Y_sess), sess_id))

            data_blocks[feat] = {
                'X': np.vstack(X_list).astype(np.float32),
                'Y': np.vstack(Y_list).astype(np.float32),
                'w': np.concatenate(w_list).astype(np.float32),
                'groups': np.concatenate(groups_list),
                'n_time_bins': X_list[0].shape[1]
            }

        return data_blocks

    def run_univariate_training(self, data_blocks: dict, feat_name: str) -> dict:
        """
        Executes the cross-validation and statistical evaluation loop for a single feature.

        This method applies the Bivariate Gaussian JAX model to the temporal kinematics
        (X) to predict the acoustic topography (Y). It rigidly evaluates performance across
        three distinct strategies:

        1. 'actual': Fits the true kinematic-to-acoustic mapping.
        2. 'null': Shuffles acoustic labels within-session. This destroys the temporal
           mapping but maintains session-level biases.
        3. 'null_model_free': Abandons modeling entirely and computes the literal global
           Marginal Prior (density of the point space). It calculates the closed-form Bivariate
           Gaussian of the training set and uses it as the static prediction for the test set.

        Data Persistence:
        -----------------
        Saves full-resolution tracking data (`test_indices`, `y_true`, `y_pred_params`) for
        ALL three strategies to enable downstream comparative scatter plotting and manifold
        visualization.

        Parameters
        ----------
        data_blocks : dict
            The full dictionary of loaded and binned data returned by `load_univariate_data_blocks`.
        feat_name : str
            The specific behavioral feature (e.g., 'self.speed') to evaluate.

        Returns
        -------
        results : dict
            The comprehensive Deep Storage dictionary containing global metrics,
            frozen model parameters, raw test data, and full probability predictions
            for all strategies.
        """

        print(f"--- Starting Univariate Training: {feat_name} ---")

        feat_data = data_blocks[feat_name]
        X = feat_data['X']
        Y = feat_data['Y']
        w = feat_data['w']
        groups = feat_data['groups']
        n_time_bins = feat_data['n_time_bins']

        hp = self.modeling_settings['hyperparameters']['jax_linear']['bivariate_gaussian']
        lam_smooth = hp['lambda_smooth']
        lam_l2 = hp['l2_reg']
        lr = hp['learning_rate']
        max_iter = hp['max_iter']
        tol = hp['tol']
        random_seed = hp['random_state']
        verbose = hp['verbose']

        cv_settings = self.modeling_settings['model_params']
        n_clusters = cv_settings['spatial_cluster_num']
        test_prop = cv_settings['test_proportion']
        n_splits = cv_settings['split_num']
        split_strategy = cv_settings['split_strategy']

        print("Generating deterministic, spatially-stratified folds...")
        folds = get_stratified_spatial_splits_stable(
            groups=groups,
            Y=Y,
            n_clusters=n_clusters,
            test_prop=test_prop,
            n_splits=n_splits,
            split_strategy=split_strategy,
            random_seed=random_seed
        )

        results = {}
        strategies = ['actual', 'null', 'null_model_free']
        rng = np.random.RandomState(random_seed)

        for strategy in strategies:
            print(f"  Executing Strategy: [{strategy.upper()}]")

            results[strategy] = {
                'folds': {
                    'metrics': {
                        'nll_weighted': [],
                        'nll_raw': [],
                        'euclidean_mae': [],
                        'mahalanobis_dist': [],
                        'r2_spatial': []
                    },
                    'weights': [],
                    'intercepts': [],
                    'test_indices': [],
                    'y_true': [],
                    'w_test': [],
                    'y_pred_params': []
                }
            }

            for fold_idx, (train_idx, test_idx) in enumerate(folds):

                Y_active = Y.copy()
                w_active = w.copy()

                if strategy == 'null':
                    unique_groups = np.unique(groups)
                    for g in unique_groups:
                        g_idx = np.where(groups == g)[0]
                        shuffled_idx = rng.permutation(g_idx)
                        Y_active[g_idx] = Y_active[shuffled_idx]
                        w_active[g_idx] = w_active[shuffled_idx]

                X_train, X_test = X[train_idx], X[test_idx]
                Y_train, Y_test = Y_active[train_idx], Y_active[test_idx]
                w_train, w_test = w_active[train_idx], w_active[test_idx]

                y_pred_params = None

                if strategy == 'null_model_free':
                    # Bypass the JAX model and manually compute the Marginal Prior (Spatial Density)
                    mu = np.average(Y_train, axis=0, weights=w_train)
                    diff = Y_train - mu
                    cov = np.average(diff[:, :, None] * diff[:, None, :], axis=0, weights=w_train)

                    sig_x = np.sqrt(cov[0, 0])
                    sig_y = np.sqrt(cov[1, 1])
                    rho = np.clip(cov[0, 1] / (sig_x * sig_y), -0.99, 0.99)

                    # Compute closed-form metrics for the test set against this fixed prior
                    dx = Y_test[:, 0] - mu[0]
                    dy = Y_test[:, 1] - mu[1]

                    z = (dx / sig_x) ** 2 - 2 * rho * (dx / sig_x) * (dy / sig_y) + (dy / sig_y) ** 2
                    det = sig_x * sig_y * np.sqrt(1 - rho ** 2)
                    log_pdf = -np.log(2 * np.pi * det) - z / (2 * (1 - rho ** 2))

                    sse = np.sum((Y_test - mu) ** 2)
                    sst = np.sum((Y_test - np.mean(Y_test, axis=0)) ** 2)

                    metrics = {
                        'nll_raw': float(-np.mean(log_pdf)),
                        'nll_weighted': float(-np.average(log_pdf, weights=w_test)),
                        'euclidean_mae': float(np.mean(np.sqrt(dx ** 2 + dy ** 2))),
                        'mahalanobis_dist': float(np.mean(np.sqrt(z / (1 - rho ** 2)))),
                        'r2_spatial': float(1.0 - (sse / sst)) if sst > 0 else 0.0
                    }

                    # Construct the static (N, 5) prediction array for downstream plotting
                    static_params = np.array([mu[0], mu[1], sig_x ** 2, sig_y ** 2, rho], dtype=np.float32)
                    y_pred_params = np.tile(static_params, (len(Y_test), 1))

                    # Model-free has no trainable weights
                    fold_weights, fold_intercepts = None, None

                else:
                    # Execute actual modeling using JAX
                    model = SmoothBivariateGaussianRegression(
                        n_features=1,
                        n_time_bins=n_time_bins,
                        lambda_smooth=lam_smooth,
                        l2_reg=lam_l2,
                        learning_rate=lr,
                        max_iter=max_iter,
                        tol=tol,
                        random_state=random_seed + fold_idx,
                        verbose=verbose
                    )
                    model.fit(X_train, Y_train, sample_weight=w_train)
                    metrics = model.evaluate_metrics(X_test, Y_test, weights=w_test)

                    # Extract full continuous density predictions
                    y_pred_params = model.predict_density(X_test)
                    fold_weights, fold_intercepts = model.coef_, model.intercept_

                print(f"      Fold {fold_idx + 1:03d}/{n_splits} | Weighted NLL: {metrics['nll_weighted']:.4f} | Euclidean Err: {metrics['euclidean_mae']:.4f}", flush=True)

                for m_key, m_val in metrics.items():
                    results[strategy]['folds']['metrics'][m_key].append(m_val)

                # Save deep storage for ALL strategies to enable comparative visualization
                results[strategy]['folds']['weights'].append(fold_weights)
                results[strategy]['folds']['intercepts'].append(fold_intercepts)
                results[strategy]['folds']['test_indices'].append(test_idx)
                results[strategy]['folds']['y_true'].append(Y_test)
                results[strategy]['folds']['w_test'].append(w_test)
                results[strategy]['folds']['y_pred_params'].append(y_pred_params)

        metrics_to_test = ['nll_weighted', 'euclidean_mae', 'mahalanobis_dist', 'r2_spatial']

        print(f"\n" + "=" * 90)
        print(f"FINAL STATISTICAL SUMMARY: {feat_name}")
        print("=" * 90)

        for metric in metrics_to_test:
            act_vals = results['actual']['folds']['metrics'][metric]
            null_vals = results['null']['folds']['metrics'][metric]
            mf_vals = results['null_model_free']['folds']['metrics'][metric]

            act_m = np.mean(act_vals)
            null_m = np.mean(null_vals)
            mf_m = np.mean(mf_vals)

            alt = 'greater' if metric == 'r2_spatial' else 'less'

            try:
                _, p_null = wilcoxon(act_vals, null_vals, alternative=alt)
                _, p_mf = wilcoxon(act_vals, mf_vals, alternative=alt)
            except ValueError:
                p_null, p_mf = 1.0, 1.0

            sig_null = "***" if p_null < 0.01 else "   "
            sig_mf = "***" if p_mf < 0.01 else "   "

            print(f"{metric:<16} | Act: {act_m:>7.4f} | Null: {null_m:>7.4f} (p={p_null:>7.1e}) {sig_null} | MF: {mf_m:>7.4f} (p={p_mf:>7.1e}) {sig_mf}")

        print("=" * 90 + "\n")

        return results
