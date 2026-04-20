"""
@author: bartulem
Module for multinomial USV category modeling (based on JAX, assumes GPU usage).

This module provides a specialized pipeline for predicting the specific semantic
category of a USV bout (e.g., category 0 vs category 1 vs ... category K) using a
multinomial (softmax) framework. It extracts behavioral and vocal history preceding a
vocalization and classifies the integer category label of that vocalization.

Key scientific capabilities:
1.  Single-pass extraction: unlike target-vs-rest approaches, this pipeline
    extracts a single target vector 'y' containing integer class labels for
    every valid bout, enabling simultaneous multi-class probability estimation.
2.  Strict syntax guardrails: incorporates prior vocal history from both the
    subject and partner while strictly excluding current-frame density metrics
    to prevent trivial self-prediction.
3.  Project wide standardization: ensures the feature matrix 'X' has consistent
    columns across all sessions, even if specific vocal categories are missing
    in individual recordings.
"""

import json
import numpy as np
import os
import pathlib
import pickle
from datetime import datetime
from sklearn.metrics import (
    balanced_accuracy_score, log_loss, f1_score,
    precision_score, recall_score, roc_auc_score,
    precision_recall_curve, auc
)
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm
from typing import Any

from .load_input_files import load_behavioral_feature_data, find_usv_categories
from .modeling_utils import (
    prepare_modeling_sessions,
    resolve_mouse_roles,
    select_kinematic_columns,
    build_vocal_signal_columns,
    harmonize_session_columns,
    zscore_features_across_sessions,
)
from .jax_multinomial_logistic_regression import SmoothMultinomialLogisticRegression
from ..analyses.compute_behavioral_features import FeatureZoo


def get_stratified_group_splits_stable(
        groups: np.ndarray,
        y: np.ndarray,
        split_strategy: str = 'session',
        test_prop: float = 0.2,
        n_splits: int = 100,
        tolerance: float = 0.05,
        random_seed: int = 0,
        n_categories: int = 6
) -> list:
    """
    Generates 100 independent 80/20 session-based splits that are
    statistically verified for category representation.

    Parameters:
    -----------
    groups : np.ndarray
        Array of session IDs (ensure samples from the same session stay together).
    y : np.ndarray
        Array of USV category labels (1-6).
    test_prop : float
        Proportion of sessions to assign to the test set (e.g., 0.2).
    n_splits : int
        Number of independent iterations (100).
    tolerance : float
        Initial allowable difference in category distribution between
        the global data and the generated splits.
    """

    if split_strategy not in ['session', 'mixed']:
        raise ValueError("split_strategy must be 'session' or 'mixed'.")

    # MIXED STRATEGY: Ignores sessions, perfectly stratifies the 6 categories
    if split_strategy == 'mixed':
        sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_prop, random_state=random_seed)
        return list(sss.split(np.zeros(len(y)), y))

    # SESSION STRATEGY: Strict cross-session prediction (your original logic)
    unique_sessions = np.unique(groups)
    n_test_sessions = int(len(unique_sessions) * test_prop)

    # Calculate global distribution for fairness check
    _, global_counts = np.unique(y, return_counts=True)
    global_dist = global_counts / len(y)

    cv_folds = []
    rng = np.random.RandomState(random_seed)

    attempts = 0
    current_tolerance = tolerance
    max_total_attempts = 50000

    while len(cv_folds) < n_splits:
        attempts += 1

        # partition sessions randomly
        shuffled = rng.permutation(unique_sessions)
        te_sess = shuffled[:n_test_sessions]
        tr_sess = shuffled[n_test_sessions:]

        tr_idx = np.where(np.isin(groups, tr_sess))[0]
        te_idx = np.where(np.isin(groups, te_sess))[0]

        # all classes must be in both sets
        tr_classes = np.unique(y[tr_idx])
        te_classes = np.unique(y[te_idx])

        if len(tr_classes) == n_categories and len(te_classes) == n_categories:
            # check distribution deviation
            _, te_counts = np.unique(y[te_idx], return_counts=True)
            te_dist = te_counts / len(te_idx)
            dist_error = np.max(np.abs(te_dist - global_dist))

            if dist_error < current_tolerance:
                cv_folds.append((tr_idx, te_idx))

        # widen the net every 1000 failures
        if attempts % 1000 == 0:
            current_tolerance += 0.02

        # prevent infinite loops
        if attempts > max_total_attempts:
            raise RuntimeError(
                f"Failed to find {n_splits} valid splits after {attempts} attempts. "
                "The rare categories may be concentrated in too few sessions."
            )

    return cv_folds

class MultinomialModelingPipeline(FeatureZoo):

    def __init__(self, modeling_settings_dict: dict = None, **kwargs):
        """
        Initializes the MultinomialModelingPipeline class.

        This constructor loads the configuration settings strictly (no flattening),
        validates the presence of required nested keys, and calculates the
        fixed-length history window size (in frames) derived from the camera
        sampling rate.

        Parameters
        ----------
        modeling_settings_dict : dict, optional
            A nested dictionary containing the modeling configuration.
            If None, the settings are loaded from the default JSON file located at
            `_parameter_settings/modeling_settings.json`.
        **kwargs : dict
            Additional keyword arguments to set as instance attributes.
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

        json_bounds = self.modeling_settings.get('feature_boundaries')
        if json_bounds:
            self.feature_boundaries = json_bounds

        try:
            cam_rate = self.modeling_settings['io']['camera_sampling_rate']
            hist_sec = self.modeling_settings['model_params']['filter_history']
            self.history_frames = int(np.floor(cam_rate * hist_sec))
            print(f"Multinomial Pipeline Init: History frames calculated: {self.history_frames} (for {hist_sec}s at {cam_rate}fps)")
        except KeyError as e:
            raise KeyError(f"Critical setting missing for history calculation: {e}")

        FeatureZoo.__init__(self)

        for kw_arg, kw_val in kwargs.items():
            self.__dict__[kw_arg] = kw_val

    def extract_and_save_multinomial_input_data(self) -> None:
        """
        Extracts, processes, and saves (X, y) data pairs for multinomial (USV category) classification.

        This pipeline generates a dataset where the target 'y' is the specific integer category
        label of a USV bout, and 'X' is the history of behavioral and vocal predictors preceding it.

        Parameters
        ----------
        This method operates entirely on the instance's `self.modeling_settings`.
        Key configurations include:
        - 'session_list_file': Path to the list of sessions to process.
        - 'random_seed': Seed for reproducibility.
        - 'vocal_features': Settings for vocal extraction (e.g., 'category_column_name').
        - 'features': Settings for behavioral history (e.g., 'filter_history').

        Returns
        -------
        Saves a pickle file to `self.modeling_settings['save_dir']` containing a nested dictionary:
        `data[feature_name][session_id] = {'X': np.array, 'y': np.array}`.
        - 'X': Predictor matrix of shape (n_samples, history_frames).
        - 'y': Target vector of shape (n_samples,) containing integer class labels.
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

        # detect all USV categories project-wide
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

        # extract multinomial targets (integer USV category labels) across sessions
        print(f"Extracting multinomial targets (all USV categories)...")
        multinomial_targets = {}

        for sess_id in list(beh_data_dict.keys()):
            if sess_id not in mouse_names_dict:
                continue

            targ_name = mouse_names_dict[sess_id][targ_idx]

            if sess_id not in usv_data_dict or targ_name not in usv_data_dict[sess_id]:
                continue

            events_by_category_dict = usv_data_dict[sess_id][targ_name]['events_by_category']

            fps = cam_fps_dict[sess_id]
            max_frame_idx = beh_data_dict[sess_id].height - 1

            temp_events = []
            for cat_id, start_times in events_by_category_dict.items():
                frame_indices = np.round(start_times * fps).astype(int)

                for f_idx in frame_indices:
                    if self.history_frames <= f_idx <= max_frame_idx:
                        temp_events.append((f_idx, int(cat_id)))

            if temp_events:
                temp_events.sort(key=lambda x: x[0])
                sorted_onsets, sorted_labels = zip(*temp_events)

                multinomial_targets[sess_id] = {
                    'onsets': np.array(sorted_onsets, dtype=int),
                    'labels': np.array(sorted_labels, dtype=int)
                }

        # select appropriate features planned for use as predictors
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

        # rename features with self/other and standardize columns across sessions
        print("Standardizing columns and z-scoring...")
        processed_beh_data, revised_predictors = harmonize_session_columns(
            processed_beh_dict=processed_beh_data,
            mouse_names_dict=mouse_names_dict,
            target_idx=targ_idx,
            predictor_idx=pred_idx
        )

        processed_beh_data = zscore_features_across_sessions(
            processed_beh_dict=processed_beh_data,
            suffixes=revised_predictors,
            feature_bounds=getattr(self, 'feature_boundaries', {}),
            abs_features=['allo_roll', 'allo_yaw-nose', 'nose-allo_yaw', 'allo_yaw-TTI', 'TTI-allo_yaw']
        )

        # slice epochs and save to disk
        print("Extracting epochs and saving to disk...")
        final_data = {}

        for sess_id, targets in tqdm(multinomial_targets.items(), desc='Slicing Epochs'):
            if sess_id not in processed_beh_data:
                continue

            sess_df = processed_beh_data[sess_id]
            t_name = mouse_names_dict[sess_id][targ_idx]
            p_name = mouse_names_dict[sess_id][pred_idx]

            onsets = targets['onsets']
            labels = targets['labels']

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
                    'y': labels
                }

        if not final_data:
            print("No valid data extracted. Aborting save.")
            return

        feature_names = sorted(list(final_data.keys()))
        n_feats = len(feature_names)
        sample_feat = feature_names[0] if n_feats > 0 else None

        if sample_feat:
            n_sessions = len(final_data[sample_feat])
            all_y = np.concatenate([final_data[sample_feat][s]['y'] for s in final_data[sample_feat]])
            total_samples = len(all_y)
            unique_cats, cat_counts = np.unique(all_y, return_counts=True)

            print("\n" + "=" * 60)
            print(f"MULTINOMIAL EXTRACTION SUMMARY")
            print("=" * 60)
            print(f"Total Unique Features:  {n_feats}")
            print(f"Total Valid Sessions:   {n_sessions}")
            print(f"Total USVs included (N): {total_samples}")
            print(f"Total USV Categories:   {len(unique_cats)}")
            print("-" * 60)

            print("EXTRACTED FEATURES:")
            # Print features in a clean, wrapped format (4 per line)
            for i in range(0, len(feature_names), 4):
                print("  " + "".join(f"{name: <25}" for name in feature_names[i:i + 4]))

            print("-" * 60)
            print("CATEGORY DISTRIBUTION:")
            for cat, count in zip(unique_cats, cat_counts):
                percentage = (count / total_samples) * 100
                print(f"  Category {cat: <2}: {count: >6} USVs ({percentage: >5.2f}%)")
            print("=" * 60 + "\n")

        target_mouse_sex = 'male' if targ_idx == 0 else 'female'
        fname = f"modeling_multinomial_category_{target_mouse_sex}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_hist{filter_hist}s.pkl"
        save_dir = self.modeling_settings['io']['save_directory']
        save_path = os.path.join(save_dir, fname)

        os.makedirs(save_dir, exist_ok=True)

        print(f"Saving extraction results to:\n{save_path} ...")
        with open(save_path, 'wb') as f:
            pickle.dump(final_data, f)
        print("[+] Save Complete.")


class MultinomialModelRunner:
    """
    Orchestrates the training and statistical validation of multinomial USV
    category models.

    This class serves as the execution engine for the modeling phase of the
    pipeline. It consumes the extracted data produced by the
    `MultinomialModelingPipeline`, transforms it into binned univariate
    design matrices, and performs rigorous cross-validation using JAX-accelerated
    optimization.

    Key responsibilities:
    ---------------------
    1. Data transformation: Pivots nested session-based dictionaries into
       unified, temporally downsampled (binned) arrays for efficient modeling.
    2. Experimental control: Implements a dual-pass 'Actual vs. Null' strategy
       to ensure that behavioral predictors are statistically meaningful.
    3. Session-aware validation: Utilizes `StratifiedGroupKFold` to prevent
       data leakage by ensuring all samples from a single recording session
       remain within either the training or testing set for a given fold.
    4. Deep metadata storage: Persists raw fold-level data (predictions, labels,
       and weights) rather than just summary statistics, enabling downstream
       confusion matrix analysis and significance testing.

    The runner relies strictly on the 'hyperparameters' and 'model_selection'
    blocks within the provided modeling settings to ensure reproducibility.
    """

    def __init__(self, pipeline_instance: Any) -> None:
        """
        Initializes the model runner using a pipeline instance.

        Parameters
        ----------
        pipeline_instance : MultinomialModelingPipeline
            An instance of the extraction class which holds the
            'modeling_settings' dictionary and calculated attributes.
        """

        self.modeling_settings = pipeline_instance.modeling_settings

        if hasattr(pipeline_instance, 'feature_boundaries'):
            self.feature_boundaries = pipeline_instance.feature_boundaries

    @staticmethod
    def load_univariate_data_blocks(pkl_path: str, bin_size: int = 10) -> dict:
        """
        Loads extracted feature data from disk and applies temporal downsampling.

        This method acts as a data-pivoting layer. It aggregates session-specific
        behavioral windows into a unified design matrix for each feature. Crucially,
        it applies temporal binning (resizing) to reduce the dimensionality of
        the predictor space, effectively acting as a low-pass filter on the
        behavioral time-series.

        Logic and Purpose:
        ------------------
        Raw behavioral data (e.g., 150 fps) contains high-frequency noise that can
        destabilize multinomial logistic regression. By averaging frames into
        bins (e.g., bin_size=10), the model focuses on slower, behaviorally
        meaningful trends in the history window. This also reduces the number of
        parameters the JAX solver must optimize, improving convergence and filter
        interpretability.

        Parameters
        ----------
        pkl_path : str
            The full path to the .pkl file containing the extracted (X, y)
            dictionaries from the MultinomialModelingPipeline.
        bin_size : int, optional
            The resizing factor for temporal downsampling. A value of 10
            means every 10 frames are averaged into 1 bin. Default is 10.

        Returns
        -------
        data_blocks : dict
            A dictionary keyed by feature name (e.g., 'self.speed'). Each value is
            another dictionary containing:
            - 'X' : np.ndarray (n_samples, n_binned_time)
                The flattened and binned behavioral history.
            - 'y' : np.ndarray (n_samples,)
                The integer labels for USV categories.
            - 'groups' : np.ndarray (n_samples,)
                String or integer IDs used for session-aware splitting.
            - 'n_time_bins' : int
                The final count of temporal predictors after binning.
        """

        print(f"Loading and binning data (bin_size={bin_size}) from: {pkl_path}")

        with open(pkl_path, 'rb') as f:
            raw_data = pickle.load(f)

        sorted_features = sorted(list(raw_data.keys()))
        data_blocks = {}

        for feat in sorted_features:
            X_list, y_list, groups_list = [], [], []
            sessions = sorted(list(raw_data[feat].keys()))

            for sess_id in sessions:
                X_sess = raw_data[feat][sess_id]['X']
                y_sess = raw_data[feat][sess_id]['y']

                N, T = X_sess.shape
                if bin_size > 1:
                    new_T = T // bin_size
                    X_sess = X_sess[:, :new_T * bin_size].reshape(N, new_T, bin_size).mean(axis=2)

                X_list.append(X_sess)
                y_list.append(y_sess)
                groups_list.append(np.full(len(y_sess), sess_id))

            data_blocks[feat] = {
                'X': np.vstack(X_list).astype(np.float32),
                'y': np.concatenate(y_list).astype(np.int32),
                'groups': np.concatenate(groups_list),
                'n_time_bins': X_list[0].shape[1]
            }

        return data_blocks

    def run_univariate_training(self, pkl_path: str, feat_name: str) -> tuple:
        """
        Executes an independent experimental pass for a single behavioral feature.

        This function evaluates the predictive capacity of a behavioral feature
        using a JAX-accelerated smooth multinomial logistic regression. It
        calculates performance against ground truth ('actual'), a statistical
        baseline ('null'), and an absolute density baseline ('null_model_free')
        to rigorously determine the significance of the findings.

        Experimental Logic:
        -------------------
        The function determines if the temporal "shape" of a behavior (e.g.,
        an increase in speed or a decrease in elevation) reliably precedes
        specific USV categories.

        To validate results, it employs a tri-strategy design:
        1. 'actual': Trains the model on the real temporal association
           between behavior (X) and vocal category (y).
        2. 'null': Shuffles the vocal labels (y) strictly WITHIN each
           session. This breaks the temporal link while preserving the session-specific
           vocal repertoire and behavioral variance.
        3. 'null_model_free': The absolute marginal prior. Instead of modeling, it
           calculates the empirical class distribution (the frequency of each USV
           category) from the training set, and predicts that exact static probability
           distribution for every sample in the test set. If the model cannot beat
           this score, it is merely guessing the majority class.

        Data Persistence (Deep Storage):
        -------------------------------
        Unlike standard training functions, this method saves all raw fold outputs,
        including true labels, class probabilities, and learned weights. This
        enables:
        - Post-hoc generation of multi-class Confusion Matrices.
        - Construction of ROC and Precision-Recall curves.
        - Statistical paired tests between Actual, Null, and Model-Free distributions.

        Parameters
        ----------
        pkl_path : str
            The path to the source data pickle file.
        feat_name : str
            The specific behavioral feature to extract and train (e.g., 'ego_yaw').

        Returns
        -------
        feat_name : str
            The name of the feature that was processed.
        combined_results : dict
            A nested dictionary containing 'actual', 'null', and 'null_model_free' strategies.
            Each strategy contains a 'folds' key with list-based storage for:
            - 'metrics' : Dict of performance scores per cross-validation fold.
            - 'weights' : The learned coefficient matrices per fold (None for model-free).
            - 'y_true'  : The actual USV labels per fold.
            - 'y_pred'  : The model's hard-choice predictions per fold.
            - 'y_probs' : The softmax probabilities for all classes per fold.
        """

        # Strict dictionary lookups (No .get() allowed)
        hp = self.modeling_settings['hyperparameters']['jax_linear']['multinomial_logistic']
        n_splits = self.modeling_settings['model_params']['split_num']
        split_strategy = self.modeling_settings['model_params']['split_strategy']
        test_prop = self.modeling_settings['model_params']['test_proportion']
        bin_size = hp['bin_resizing_factor']

        all_blocks = self.load_univariate_data_blocks(pkl_path, bin_size=bin_size)
        if feat_name not in all_blocks:
            raise KeyError(f"Feature '{feat_name}' not found. Available: {list(all_blocks.keys())}")

        feat_data = all_blocks[feat_name]
        X = feat_data['X']
        groups = feat_data['groups']
        n_time = feat_data['n_time_bins']

        cv_folds = get_stratified_group_splits_stable(
            groups=groups,
            y=feat_data['y'],
            split_strategy=split_strategy,
            test_prop=test_prop,
            n_splits=n_splits,
            random_seed=self.modeling_settings['model_params']['random_seed'],
            n_categories=self.modeling_settings['vocal_features']['usv_category_number']
        )

        strategies = ['actual', 'null', 'null_model_free']
        combined_results = {}

        for strategy in strategies:
            print(f"\n" + "=" * 60)
            print(f"FEATURE: {feat_name} | STRATEGY: {strategy.upper()}")
            print("=" * 60)

            y = feat_data['y'].copy()
            if strategy == 'null':
                for sess_id in np.unique(groups):
                    sess_mask = (groups == sess_id)
                    sess_labels = y[sess_mask]
                    np.random.shuffle(sess_labels)
                    y[sess_mask] = sess_labels

            strategy_data = {
                'folds': {
                    'metrics': {m: [] for m in ['auc', 'score', 'precision', 'recall', 'f1', 'll']},
                    'weights': [],
                    'intercepts': [],
                    'y_true': [],
                    'y_pred': [],
                    'y_probs': [],
                    'test_indices': []
                },
                'classes': None
            }

            for fold, (train_idx, test_idx) in enumerate(cv_folds):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                fold_weights = None
                fold_intercepts = None

                if strategy == 'null_model_free':
                    # Compute empirical class priors from the training set
                    unique_classes, counts = np.unique(y_train, return_counts=True)
                    class_priors = counts / float(np.sum(counts))

                    # Predict this exact same probability distribution for every test sample
                    probabilities = np.tile(class_priors, (len(y_test), 1))

                    # Hard prediction is simply the majority class
                    majority_class = unique_classes[np.argmax(class_priors)]
                    predictions = np.full(len(y_test), majority_class)

                    model_classes = unique_classes
                else:
                    model = SmoothMultinomialLogisticRegression(
                        n_features=1,
                        n_time_bins=n_time,
                        lambda_smooth=hp['lambda_smooth'],
                        l1_reg=hp['l1_reg'],
                        l2_reg=hp['l2_reg'],
                        learning_rate=hp['learning_rate'],
                        max_iter=hp['max_iter'],
                        tol=hp['tol'],
                        random_state=hp['random_state'] + fold,
                        verbose=hp['verbose']
                    )

                    model.fit(X_train, y_train)

                    probabilities = model.predict_proba(X_test, balanced=hp['balance_predictions_bool'])
                    predictions = model.predict(X_test, balanced=hp['balance_predictions_bool'])

                    model_classes = model.classes_
                    fold_weights = model.coef_
                    fold_intercepts = model.intercept_

                # Apply numerical clipping to prevent log-loss infinity issues
                eps = 1e-15
                probabilities_clipped = np.clip(probabilities, eps, 1 - eps)

                f_score = balanced_accuracy_score(y_test, predictions)

                try:
                    # model_classes guarantees perfect multi-class alignment
                    f_ll = log_loss(y_test, probabilities_clipped, labels=model_classes)
                except ValueError:
                    f_ll = np.nan

                try:
                    f_auc = roc_auc_score(y_test, probabilities, multi_class='ovr', average='macro', labels=model_classes)
                except ValueError:
                    f_auc = np.nan

                f_met = strategy_data['folds']['metrics']
                f_met['score'].append(f_score)
                f_met['ll'].append(f_ll)
                f_met['auc'].append(f_auc)
                f_met['precision'].append(precision_score(y_test, predictions, average='macro', zero_division=0))
                f_met['recall'].append(recall_score(y_test, predictions, average='macro', zero_division=0))
                f_met['f1'].append(f1_score(y_test, predictions, average='macro', zero_division=0))

                print(f"Fold {fold + 1:02d}/{n_splits:02d} | Score: {f_score:.3f} | AUC: {f_auc:.3f} | LL: {f_ll:.3f}")

                # Persist deep storage matrices
                strategy_data['folds']['weights'].append(fold_weights)
                strategy_data['folds']['intercepts'].append(fold_intercepts)
                strategy_data['folds']['y_true'].append(y_test)
                strategy_data['folds']['y_pred'].append(predictions)
                strategy_data['folds']['y_probs'].append(probabilities)
                strategy_data['folds']['test_indices'].append(test_idx)

                if strategy_data['classes'] is None:
                    strategy_data['classes'] = model_classes

            m = {k: np.nanmean(v) for k, v in strategy_data['folds']['metrics'].items()}
            print(f"\nFINISHED {strategy.upper()} | Avg Score: {m['score']:.3f} | Avg AUC: {m['auc']:.3f}")

            combined_results[strategy] = strategy_data

        return feat_name, combined_results
