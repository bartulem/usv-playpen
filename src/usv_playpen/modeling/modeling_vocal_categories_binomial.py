"""
@author: bartulem
Module for target-vs-rest USV category modeling.

This module provides an advanced modeling pipeline designed to identify behavioral
and vocal predictors of specific ultrasonic vocalization (USV) categories.
Beyond standard behavioral features, it supports "Full Syntax" models where
the prior vocalizations of both the subject (self) and the partner (other)
can serve as predictors for upcoming target categories.

Key scientific capabilities:
1.  Extracts and balances epochs to distinguish a
    specific "target" USV category from a pooled "other" class (all other valid
    vocalizations), enabling high-resolution feature importance ranking.
2.  Incorporates partner vocal traces and subject-own
    vocal history. Implements an "identity guard" to exclude the target category
    itself from the predictor set, ensuring results capture true state
    transitions rather than mathematical circularity.
3.  Manages complex dyadic features (e.g., nose-TTI, nose-allo_yaw)
    by automatically excluding perspective-inconsistent predictors based on the
    assigned role of the predictor mouse.
4.  Implements mixed or session-based K-fold splitting with
    strict sample-size matching between actual and null-other strategies to
    eliminate sample-size bias in significance testing.
"""

from datetime import datetime
import json
import numpy as np
import os
import pathlib
import pickle
from pygam import LogisticGAM, te
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, log_loss, f1_score, precision_score, recall_score, accuracy_score
from tqdm import tqdm

from .load_input_files import load_behavioral_feature_data, find_usv_categories
from .modeling_utils import (
    prepare_modeling_sessions,
    resolve_mouse_roles,
    select_kinematic_columns,
    build_vocal_signal_columns,
    harmonize_session_columns,
    zscore_features_across_sessions,
    pool_session_arrays,
    balance_two_class_arrays,
    unroll_history_matrix,
)
from ..analyses.compute_behavioral_features import FeatureZoo


class VocalCategoryModelingPipeline(FeatureZoo):

    def __init__(self, modeling_settings_dict: dict = None, **kwargs):
        """
        Initializes the VocalCategoryModelingPipeline class.

        This class orchestrates the extraction, preprocessing, and modeling of behavioral and
        vocalization data for a specific "target-vs-rest" modeling analysis (e.g. predicting Category 3).
        It manages settings, calculates necessary constants (like history frames), and prepares
        the environment for both data extraction and parallel model fitting.

        Parameters
        ----------
        modeling_settings_dict : dict, optional
            A dictionary containing the modeling settings. If None, settings are loaded
            from the default JSON file located at `_parameter_settings/modeling_settings.json`.
            The dictionary structure can be nested (e.g., grouped by 'features' or 'data_io').
        **kwargs : dict
            Additional keyword arguments. These are set as instance attributes on the
            pipeline object, allowing for dynamic overrides of settings or injection
            of additional parameters.
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
            print(f"History frames calculated: {self.history_frames} (for {hist_sec}s at {cam_rate}fps)")
        except KeyError as e:
            raise KeyError(f"Critical setting missing for history calculation: {e}")

        FeatureZoo.__init__(self)

        for kw_arg, kw_val in kwargs.items():
            self.__dict__[kw_arg] = kw_val

    def extract_and_save_category_input_data(self, target_category: int) -> None:
        """
        Extracts, processes, and saves data for target-vs-rest classification of a specific USV category.

        This method executes the full data preparation pipeline:
        1.  Loads behavioral tracking data and USV summaries for all sessions.
        2.  Calls `find_usv_categories` to separate the `target_category`
            from others and generates continuous vocal predictors (binary or smoothed).
        3.  Feature Engineering:
            - Filters standard behavioral features (self, other, dyadic) based on settings.
            - Applies refined exclusion logic for directional dyadic features
              (e.g., 'nose-allo_yaw', 'nose-TTI') based on which mouse is the current predictor.
            - Incorporates pre-generated vocal syntax traces directly
              from the loading module.
              * **Partner Mouse:** All vocal traces are ingested as predictors.
                  This is true, unless the partner mouse never vocalized in any of the sessions.
              * **Subject Mouse (Self):** Ingests vocal categories to capture syntax transitions
                  (e.g., Cat 1 predicting Cat 3). To ensure scientific validity, it strictly
                  excludes the 'target_category' itself to avoid self-prediction, and excludes
                  density traces (proportion/event) to prevent trivial autocorrelation.
        4.  Z-scores all features across sessions, respecting physical boundaries
            defined in `feature_boundaries` while bypassing USV traces.
        5.  Slices continuous data into fixed-length history windows using
            precise rounding to align behavioral frames with USV onset timestamps.
        6.  Dumps processed data into a `.pkl` file structured by feature -> session.

        Parameters
        ----------
        target_category : int
            The integer ID of the USV category to predict (e.g., 3). This defines the positive class.
            All other valid USV categories (excluding noise) form the negative class.

        Returns
        -------
        None
            Saves a pickle file containing 'target_feature_arr' and 'other_feature_arr' for each feature.
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

        print(f"Loading USV data and generating predictors for Category {target_category}...")
        usv_data_dict = find_usv_categories(
            root_directories=txt_sessions,
            mouse_ids_dict=mouse_names_dict,
            camera_fps_dict=cam_fps_dict,
            features_dict=beh_data_dict,
            csv_sep=self.modeling_settings['io']['csv_separator'],
            target_category=target_category,
            category_column=column_name_cats,
            filter_history=filter_hist,
            vocal_output_type=voc_mode,
            proportion_smoothing_sd=smooth_sd,
            noise_vocal_categories=noise_cats
        )

        processed_beh_data = {}
        pred_idx = self.modeling_settings['model_params']['model_predictor_mouse_index']
        targ_idx = abs(pred_idx - 1)
        category_self_exclude = ('usv_rate', 'usv_event', f"usv_cat_{target_category}")

        for sess_id in list(beh_data_dict.keys()):
            if sess_id not in mouse_names_dict or len(mouse_names_dict[sess_id]) < 2:
                continue

            current_df = beh_data_dict[sess_id]

            (pred_idx,
             targ_idx,
             pred_name,
             targ_name) = resolve_mouse_roles(
                modeling_settings=self.modeling_settings,
                mouse_names_dict=mouse_names_dict,
                session_id=sess_id
            )

            sess_cols = current_df.columns

            cols_to_keep = select_kinematic_columns(
                session_df_columns=sess_cols,
                target_name=targ_name,
                predictor_name=pred_name,
                kin_settings=kin_settings,
                predictor_idx=pred_idx
            )

            new_voc_cols, new_voc_col_names = build_vocal_signal_columns(
                usv_data_dict=usv_data_dict,
                session_id=sess_id,
                target_name=targ_name,
                predictor_name=pred_name,
                voc_settings=voc_settings,
                usv_self_exclude=category_self_exclude
            )

            cols_to_keep = sorted(set(cols_to_keep) | set(new_voc_col_names))

            if new_voc_cols:
                current_df = current_df.with_columns(new_voc_cols)
            processed_beh_data[sess_id] = current_df.select([c for c in cols_to_keep if c in current_df.columns])

        print("Standardizing columns (Project-Wide Consistency)...")
        processed_beh_data, revised_predictors = harmonize_session_columns(
            processed_beh_dict=processed_beh_data,
            mouse_names_dict=mouse_names_dict,
            target_idx=targ_idx,
            predictor_idx=pred_idx
        )

        print("Z-scoring features across sessions...")
        processed_beh_data = zscore_features_across_sessions(
            processed_beh_dict=processed_beh_data,
            suffixes=revised_predictors,
            feature_bounds=getattr(self, 'feature_boundaries', {})
        )

        print("Extracting epochs...")
        final_data = {}
        for sess_id in tqdm(processed_beh_data.keys(), desc='Sessions'):
            sess_df = processed_beh_data[sess_id]
            t_name = mouse_names_dict[sess_id][targ_idx]
            p_name = mouse_names_dict[sess_id][pred_idx]
            try:
                target_times = usv_data_dict[sess_id][t_name]['target_events']
                other_times = usv_data_dict[sess_id][t_name]['other_events']
            except KeyError:
                continue

            fps = cam_fps_dict[sess_id]
            hist_frames = self.history_frames

            for col in sess_df.columns:
                suffix = col.split('.')[-1]
                if '-' in suffix:
                    gen_key = suffix
                elif col.startswith(f"{t_name}."):
                    gen_key = f"self.{suffix}"
                elif col.startswith(f"{p_name}."):
                    gen_key = f"other.{suffix}"
                else:
                    gen_key = suffix

                if gen_key not in final_data:
                    final_data[gen_key] = {}
                if sess_id not in final_data[gen_key]:
                    final_data[gen_key][sess_id] = {}

                col_data = sess_df[col].to_numpy()
                max_idx = len(col_data)
                t_arr = np.full((target_times.size, hist_frames), np.nan)
                o_arr = np.full((other_times.size, hist_frames), np.nan)

                def fill_epochs(times, arr):
                    ends = np.round(times * fps).astype(int)
                    starts = ends - hist_frames
                    for j in range(times.size):
                        s, e = starts[j], ends[j]
                        if s >= 0 and e <= max_idx:
                            chunk = col_data[s:e].copy()
                            chunk[np.isnan(chunk)] = 0.0
                            arr[j, :] = chunk

                fill_epochs(target_times, t_arr)
                fill_epochs(other_times, o_arr)
                final_data[gen_key][sess_id]['target_feature_arr'] = t_arr
                final_data[gen_key][sess_id]['other_feature_arr'] = o_arr

        final_features = sorted(list(final_data.keys()))
        if not final_features: return

        first_feat = final_features[0]
        all_sessions = sorted(list(final_data[first_feat].keys()))

        total_target = sum(final_data[first_feat][s]['target_feature_arr'].shape[0] for s in all_sessions)
        total_rest = sum(final_data[first_feat][s]['other_feature_arr'].shape[0] for s in all_sessions)

        print("\n" + "=" * 105)
        print(f"CATEGORY MODELING SUMMARY: Category {target_category}")
        print("=" * 105)
        print(f"{'#':<4} {'Feature Name':<45} | {'Sess':<6} | {'Target':<10} | {'Rest':<10} | {'Total N'}")
        print("-" * 105)
        for i, feat in enumerate(final_features, 1):
            feat_t = sum(final_data[feat][s]['target_feature_arr'].shape[0] for s in final_data[feat])
            feat_o = sum(final_data[feat][s]['other_feature_arr'].shape[0] for s in final_data[feat])
            print(f"{i:3}. {feat:<45} | {len(final_data[feat]):<6} | {feat_t:<10} | {feat_o:<10} | {feat_t + feat_o:<8}")
        print("-" * 105)
        print(f"PROJECT-WIDE CATEGORY TALLY:")
        print(f"  > Total unique features:        {len(final_features)}")
        print(f"  > Total sessions:               {len(all_sessions)}")
        print(f"  > Total target USVs ({target_category}):        {total_target}")
        print(f"  > Total other USVs:             {total_rest}")
        print(f"  > Grand total USVs (N):         {total_target + total_rest}")
        print("=" * 105 + "\n")

        target_mouse_sex = 'male' if targ_idx == 0 else 'female'
        fname = f"modeling_category_{target_category}_{target_mouse_sex}_{voc_mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_hist{filter_hist}s.pkl"
        save_path = os.path.join(self.modeling_settings['io']['save_directory'], fname)
        os.makedirs(self.modeling_settings['io']['save_directory'], exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(final_data, f)
        print(f"\n[+] Successfully saved category input data to:\n    {save_path}")

    def create_category_splits(self, feature_data: dict, strategy: str = 'actual'):
        """
        Generator yielding train/test splits for K-Fold validation.

        This function orchestrates the data splitting for model validation, dynamically
        handling both 'session' and 'mixed' cross-validation strategies based on the
        project configuration. It ensures rigorous comparability between experimental
        strategies by strictly enforcing identical sample sizes and class balances.

        Splitting strategies:
        ---------------------
        1. 'session': Evaluates model generalizability across independent recording sessions.
           Whole sessions are held out for testing. (e.g., Train on Sessions A and B;
           Test on Session C).
        2. 'mixed': Evaluates model performance on a completely randomized pool of data.
           Epochs from all sessions are combined and then randomized into train/test sets,
           ensuring a purely proportional split of the total project-wide sample size.
           This utilizes StratifiedShuffleSplit to ensure class ratios remain identical
           across all folds.

        Balancing Process:
        ------------------
        1.  Identifies data containing both target (positive) and other (negative) classes.
        2.  Executes the selected split strategy ('session' or 'mixed') to generate base
            training and testing data pools.
        3.  Calculates the maximum possible sample size based on the minority class
            (usually target USVs) in the resulting pools to prevent frequency bias.
        4.  Execution strategy:
            - 'actual': Balances Target vs. Other data 50/50 using the calculated limit.
              (e.g., 100 Target vs 100 Other).
            - 'null_other': Uses ONLY 'other' data but splits it into two pseudo-classes.
              Crucially, it downsamples this data to MATCH the 'actual' size exactly.
              (e.g., 100 other (A) vs 100 other (B)). This prevents the null model from
              performing artificially well simply due to having a larger training volume.

        Parameters
        ----------
        feature_data : dict
            The data dictionary for a single feature, containing 'target_feature_arr'
            and 'other_feature_arr' for each session.
        strategy : str, optional
            The experimental condition strategy to use:
            - 'actual': Target (1) vs. Other (0).
            - 'null_other': Other (Pseudo-1) vs. Other (Pseudo-0).
            Default is 'actual'.

        Yields
        ------
        tuple
            A tuple of (X_train, y_train, X_test, y_test) for one validation fold
            represented as NumPy arrays.
        """

        all_sessions = list(feature_data.keys())
        valid_sessions = []
        for sess in all_sessions:
            try:
                if (feature_data[sess]['target_feature_arr'].shape[0] > 0 and
                        feature_data[sess]['other_feature_arr'].shape[0] > 0):
                    valid_sessions.append(sess)
            except (KeyError, AttributeError):
                continue

        valid_sessions = np.array(valid_sessions)
        if len(valid_sessions) < 2:
            return

        # Strict lookup, no .get()
        model_ops = self.modeling_settings['model_params']
        n_splits = model_ops['split_num']
        test_prop = model_ops['test_proportion']
        split_strategy = model_ops['split_strategy']
        rand_seed = self.modeling_settings['model_params']['random_seed']

        splits_data = []

        if split_strategy == 'session':
            ss = ShuffleSplit(n_splits=n_splits, test_size=test_prop, random_state=rand_seed)
            for train_idx, test_idx in ss.split(valid_sessions):
                train_sessions = valid_sessions[train_idx]
                test_sessions = valid_sessions[test_idx]

                X_tr_targ, X_tr_other = pool_session_arrays(feature_data, train_sessions, pos_key="target_feature_arr", neg_key="other_feature_arr", n_frames=self.history_frames)
                X_te_targ, X_te_other = pool_session_arrays(feature_data, test_sessions, pos_key="target_feature_arr", neg_key="other_feature_arr", n_frames=self.history_frames)
                splits_data.append((X_tr_targ, X_tr_other, X_te_targ, X_te_other))

        elif split_strategy == 'mixed':
            X_all_targ, X_all_other = pool_session_arrays(feature_data, valid_sessions, pos_key="target_feature_arr", neg_key="other_feature_arr", n_frames=self.history_frames)

            if len(X_all_targ) > 0 and len(X_all_other) > 0:
                # Combine entirely to run a true stratified split
                X_pooled = np.vstack([X_all_targ, X_all_other])
                y_pooled = np.hstack([np.ones(len(X_all_targ)), np.zeros(len(X_all_other))])

                sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_prop, random_state=rand_seed)

                for tr_idx, te_idx in sss.split(X_pooled, y_pooled):
                    X_tr = X_pooled[tr_idx]
                    y_tr = y_pooled[tr_idx]
                    X_te = X_pooled[te_idx]
                    y_te = y_pooled[te_idx]

                    # Separate back into target/other arrays for the Phase 2 balancing step
                    X_tr_targ = X_tr[y_tr == 1]
                    X_tr_other = X_tr[y_tr == 0]
                    X_te_targ = X_te[y_te == 1]
                    X_te_other = X_te[y_te == 0]

                    splits_data.append((X_tr_targ, X_tr_other, X_te_targ, X_te_other))
        else:
            raise ValueError(f"Unknown split_strategy '{split_strategy}'. Must be 'session' or 'mixed'.")

        # Phase 2: Ensure train/test data are rigorously balanced across strategies
        for X_tr_targ, X_tr_other, X_te_targ, X_te_other in splits_data:

            n_tr_limit = min(X_tr_targ.shape[0], X_tr_other.shape[0])
            n_te_limit = min(X_te_targ.shape[0], X_te_other.shape[0])

            if n_tr_limit == 0 or n_te_limit == 0:
                continue

            if strategy == 'actual':
                X_tr_A, X_tr_B = balance_two_class_arrays(X_tr_targ, X_tr_other)
                X_te_A, X_te_B = balance_two_class_arrays(X_te_targ, X_te_other)

            elif strategy == 'null_other':
                def balance_pseudo(X, limit):
                    needed = limit * 2
                    if len(X) < needed:
                        return None, None
                    X_sub = X[np.random.choice(len(X), needed, replace=False)]
                    return X_sub[:limit], X_sub[limit:]

                X_tr_A, X_tr_B = balance_pseudo(X_tr_other, n_tr_limit)
                X_te_A, X_te_B = balance_pseudo(X_te_other, n_te_limit)

                if X_tr_A is None or X_te_A is None:
                    continue

            X_train = np.vstack([X_tr_A, X_tr_B])
            y_train = np.hstack([np.ones(len(X_tr_A)), np.zeros(len(X_tr_B))])

            X_test = np.vstack([X_te_A, X_te_B])
            y_test = np.hstack([np.ones(len(X_te_A)), np.zeros(len(X_te_B))])

            p_train = np.random.permutation(len(y_train))
            p_test = np.random.permutation(len(y_test))

            yield X_train[p_train], y_train[p_train], X_test[p_test], y_test[p_test]

    def _run_modeling_category(self, feature_name: str, feature_data: dict, basis_matrix: np.ndarray | None) -> tuple[str, dict]:
        """
        Executes a one-vs-rest univariate classification analysis for a specific USV category.

        This method acts as the computational core for category-specific modeling. It evaluates
        the predictive power of a behavioral or vocal feature by attempting to distinguish
        the 'target' category (positive class) from a pooled 'other' category (negative class).
        The analysis implements rigorous statistical controls, including size-matched null
        distributions and cross-validated regularized regression.

        The computational workflow includes:
        1.  Dual-strategy splitting:
            - 'actual': Trains on true target vs. other labels to identify behavioral predictors.
            - 'null': Trains on pseudo-classes derived solely from 'other' epochs. Crucially,
               this null distribution is forced to match the exact sample size and class
               ratio of the actual data to eliminate sample-size bias in performance metrics.
        2. Linear basis projection (sklearn engine):
            - Reduces high-dimensional temporal history (N_lags) into a compressed basis
              representation (N_bases).
            - Fits a `LogisticRegressionCV` to determine the optimal L1/L2 penalty (C).
            - Reconstructs linear filter shapes via back-projection (coefs · basis_matrix.T).
        3.  Non-linear spline fitting (pyGAM engine):
            - Unrolls temporal windows into long-form matrices for tensor product spline (te)
              fitting.
            - Estimates non-linear log-odds surfaces across feature values and time lags.
            - Extracts linear filter approximations via partial dependence calculations.
        4.  Metric aggregation: Computes fold-wise classification statistics (AUC, F1,
            Log-Loss, Accuracy) for both actual and null models to enable significance testing.

        Parameters
        ----------
        feature_name : str
            The identifier for the feature being analyzed (e.g., 'self.speed', 'other.usv_cat_2').
        feature_data : dict
            A nested dictionary containing session-specific arrays for the positive
            ('target_feature_arr') and negative ('other_feature_arr') classes.
        basis_matrix : np.ndarray | None
            The projection matrix [lags, bases] used for dimensionality reduction in Sklearn.
            Must be None if `model_type` is 'pygam'.

        Returns
        -------
        tuple[str, dict]
            A tuple containing the feature name and a nested results dictionary with 'actual'
            and 'null' sub-keys. Sub-keys contain NumPy arrays for computed metrics,
            reconstructed filter shapes, and (if sklearn) optimized hyperparameters across
            all validation folds.
        """

        # Strict lookup, no .get()
        model_type = self.modeling_settings['model_params']['model_engine']
        n_splits = self.modeling_settings['model_params']['split_num']

        metrics = ['auc', 'score', 'precision', 'recall', 'f1', 'll']
        results = {
            'actual': {m: np.full(n_splits, np.nan) for m in metrics},
            'null': {m: np.full(n_splits, np.nan) for m in metrics}
        }

        results['actual']['filter_shapes'] = np.full((n_splits, self.history_frames), np.nan)

        n_bases = basis_matrix.shape[1] if basis_matrix is not None else 0
        if model_type == 'sklearn':
            results['actual']['coefs_projected'] = np.full((n_splits, n_bases), np.nan)
            results['actual']['optimal_C'] = np.full(n_splits, np.nan)

        strategies = ['actual', 'null_other']
        time_indices = np.arange(self.history_frames, dtype=np.float32)

        for strat in strategies:
            splitter = self.create_category_splits(feature_data, strategy=strat)
            key = 'actual' if strat == 'actual' else 'null'

            for split_idx, (X_tr, y_tr, X_te, y_te) in enumerate(splitter):
                if split_idx >= n_splits: break

                print(f"    Processing {strat.upper()} split {split_idx + 1}/{n_splits}...")

                try:
                    y_prob = None
                    y_pred = None

                    if model_type == 'sklearn':
                        X_tr_p = np.dot(X_tr, basis_matrix)
                        X_te_p = np.dot(X_te, basis_matrix)

                        lr_params = self.modeling_settings['hyperparameters']['classical']['logistic_regression']

                        lr_actual = LogisticRegressionCV(
                            penalty=lr_params['penalty'],
                            Cs=lr_params['cs'],
                            cv=lr_params['cv'],
                            class_weight='balanced',
                            solver=lr_params['solver'],
                            max_iter=lr_params['max_iter'],
                            random_state=self.modeling_settings['model_params']['random_seed']
                        )

                        lr_actual.fit(X_tr_p, y_tr)

                        y_prob = lr_actual.predict_proba(X_te_p)[:, 1]
                        y_pred = lr_actual.predict(X_te_p)

                        if strat == 'actual':
                            results['actual']['coefs_projected'][split_idx, :] = lr_actual.coef_.flatten()
                            results['actual']['optimal_C'][split_idx] = lr_actual.C_[0]

                            # Back-project filter shape
                            filter_shape_actual = np.dot(lr_actual.coef_, basis_matrix.T).ravel()
                            results['actual']['filter_shapes'][split_idx, :] = filter_shape_actual

                    elif model_type == 'pygam':
                        X_tr_gam = unroll_history_matrix(X_tr)
                        X_te_gam = unroll_history_matrix(X_te)
                        y_tr_gam = np.repeat(y_tr, self.history_frames).astype(int)

                        pg_params = self.modeling_settings['hyperparameters']['classical']['pygam']
                        n_splines_val = pg_params['n_splines_value']
                        n_splines_time = pg_params['n_splines_time']
                        gam_args = {
                            'lam': pg_params['lam_penalty'],
                            'max_iter': pg_params['max_iterations'],
                            'tol': pg_params['tol_val']
                        }

                        gam = LogisticGAM(
                            te(0, 1, n_splines=[n_splines_val, n_splines_time]),
                            **gam_args
                        ).fit(X_tr_gam, y_tr_gam)

                        diffs = gam.logs_['diffs']
                        print(f"      Completed in {len(diffs)} iters | "
                              f"Final Δ: {diffs[-1] if diffs else 0.0:.2e} (Tol: {gam_args['tol']:.2e}) | "
                              f"Deviance: {gam.logs_['deviance'][-1]:.2f}")

                        y_prob_frame = gam.predict_proba(X_te_gam)
                        y_prob = np.mean(y_prob_frame.reshape(X_te.shape), axis=1)
                        y_pred = (y_prob > 0.5).astype(int)

                        if strat == 'actual':
                            grid_0 = np.column_stack([np.zeros(self.history_frames), time_indices])
                            grid_1 = np.column_stack([np.ones(self.history_frames), time_indices])
                            shape = gam.predict_mu(grid_1) - gam.predict_mu(grid_0)
                            results['actual']['filter_shapes'][split_idx] = shape

                    if y_prob is not None and y_pred is not None:
                        results[key]['auc'][split_idx] = roc_auc_score(y_te, y_prob)
                        results[key]['ll'][split_idx] = log_loss(y_te, np.clip(y_prob, 1e-15, 1 - 1e-15))

                        results[key]['score'][split_idx] = accuracy_score(y_te, y_pred)
                        results[key]['precision'][split_idx] = precision_score(y_te, y_pred, zero_division=0.0)
                        results[key]['recall'][split_idx] = recall_score(y_te, y_pred, zero_division=0.0)
                        results[key]['f1'][split_idx] = f1_score(y_te, y_pred, average='micro', zero_division=0.0)

                        print(f"    > {strat.capitalize()} Fold {split_idx} (Train N={len(y_tr)}, Test N={len(y_te)}): "
                              f"AUC={results[key]['auc'][split_idx]:.3f}, "
                              f"LL={results[key]['ll'][split_idx]:.3f}, "
                              f"Acc={results[key]['score'][split_idx]:.3f}")

                except Exception as e:
                    print(f"Fit error {feature_name} ({strat}), fold {split_idx}: {e}")

        mean_auc_act = np.nanmean(results['actual']['auc'])
        mean_auc_null = np.nanmean(results['null']['auc'])
        print(f"  Feature {feature_name}: Mean AUC={mean_auc_act:.3f} (Null={mean_auc_null:.3f})")

        return feature_name, results
