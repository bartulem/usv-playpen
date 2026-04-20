"""
@author: bartulem
Module for modeling USV (bout) onsets.

This module provides the core pipeline for modeling mouse vocal onset probability
using 3D-extracted kinematic features. It handles the orchestration of
data preparation, feature engineering (egocentric/dyadic), and statistical
classification using both Linear (sklearn) and Non-linear (PyGAM) engines.
"""

from datetime import datetime
import json
import numpy as np
import os
import pathlib
import polars as pls
from pygam import LogisticGAM, te
import pickle
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, log_loss, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
from tqdm import tqdm

from .load_input_files import load_behavioral_feature_data, find_bout_epochs
from .modeling_utils import (
    prepare_modeling_sessions,
    resolve_mouse_roles,
    select_kinematic_columns,
    build_vocal_signal_columns,
    identify_empty_event_sessions,
    collect_predictor_suffixes,
    zero_fill_missing_feature_columns,
    zscore_features_across_sessions,
    pool_session_arrays,
    balance_two_class_arrays,
    unroll_history_matrix,
    concat_two_class_with_labels,
    shuffle_train_test_arrays,
    bounded_test_proportion,
)
from ..analyses.compute_behavioral_features import FeatureZoo


class VocalOnsetModelingPipeline(FeatureZoo):

    def __init__(self, modeling_settings_dict, **kwargs):
        """
        Initializes the VocalOnsetModelingPipeline class.

        Loads modeling settings from a dictionary or JSON file and initializes
        the parent FeatureZoo class. Calculates and stores `history_frames`.

        Parameters
        ----------
        modeling_settings_dict : dict or None
            Dictionary containing modeling settings. If None, settings are loaded
            from '_parameter_settings/modeling_settings.json'.
        **kwargs : dict
            Additional keyword arguments to set as attributes.
        """

        if modeling_settings_dict is None:
            settings_path = pathlib.Path(__file__).resolve().parent.parent / '_parameter_settings/modeling_settings.json'
            try:
                with open(settings_path, 'r') as settings_json_file:
                    self.modeling_settings = json.load(settings_json_file)

            except FileNotFoundError:
                raise FileNotFoundError(f"Settings file not found at {settings_path}")
        else:
            self.modeling_settings = modeling_settings_dict

        json_bounds = self.modeling_settings.get('feature_boundaries')

        if json_bounds:
            self.feature_boundaries = json_bounds

        try:
            camera_rate = self.modeling_settings['io']['camera_sampling_rate']
            filter_history_sec = self.modeling_settings['model_params']['filter_history']
            self.history_frames = int(np.floor(camera_rate * filter_history_sec))
            print(f"History frames calculated: {self.history_frames} (for {filter_history_sec}s at {camera_rate}fps)")
        except KeyError as e:
            raise KeyError(f"Setting missing for history_frames calculation: {e}")

        FeatureZoo.__init__(self)

        for kw_arg, kw_val in kwargs.items():
            self.__dict__[kw_arg] = kw_val

    def extract_and_save_modeling_input_data_(self) -> None:
        """
        Extracts, processes, and saves behavioral and vocalization data PER SESSION for modeling analysis.

        This method orchestrates the data preparation pipeline for each session, specifically designed
        for classification tasks (USV vs. No-USV epochs). It incorporates advanced bout detection
        using Gaussian Mixture Models (GMM) to define dynamic inter-bout intervals (IBI).

        Pipeline Steps:
        1.  Calls `find_bout_epochs` to detect vocal bouts.
            - Uses `gmm_component_index` and `gmm_z_score` to calculate a dynamic IBI threshold per mouse.
            - Filters out specific noise categories (e.g., [0, 19]) via `noise_vocal_categories`.
            - Enforces `min_usv_per_bout` constraints.
            - Returns positive (`positive_events`) and negative (`negative_events`) event times.
        2. Removes sessions where the target mouse has 0 valid bouts.
        3. Selects relevant behavioral features per session via the three-bucket
            kinematic schema in `modeling_settings['kinematic_features']`:
            - `egocentric`: per-mouse scalar features kept for both target
              (self) and predictor (other) mice.
            - `dyadic_pose`: directional two-mouse pose features; when
              `dyadic_pose_symmetric` is False, a directional allo_yaw/TTI rule
              drops one symmetric half based on the predictor-mouse convention.
            - `dyadic_engagement`: interaction features such as Social
              Engagement Index (SEI); the directional rule is never applied here.
            - Adds 1st/2nd derivatives if configured.
        4.  Ingests pre-generated vocal signals (proportion/categories)
            from `find_bout_epochs`.
            - Partner (other) mouse: Ingests all biological vocal signals.
            - Subject (self) mouse: Ingests ONLY categories (syntax), never proportion/event,
              to prevent self-autocorrelation from drowning out behavioral predictors.
            - Controlled by `vocal_output_partner_only` toggle.
        5.  Normalizes all features (including vocal signals) across all sessions
            using global statistics to ensure comparability.
        6.  Slices the behavioral history (defined by `filter_history`)
            preceding every valid USV and No-USV event.
        7.  Renames features to generic 'self.*' / 'other.*' prefixes
            and organizes data into a nested dictionary: `Feature -> Session -> {usv_arr, no_usv_arr}`.
        8.  Serializes the final dictionary to a .pkl file for downstream modeling.

        Settings are controlled via the `self.modeling_settings` attribute.

        Outputs
        -------
        .pkl file
            A pickle file containing a nested dictionary. Structure:
            `{'generic_feature_name': {'session_id1': {'usv_feature_arr': ..., 'no_usv_feature_arr': ...},
                                      'session_id2': {...}, ...},
             'another_feature': {...}, ...}`
        """

        txt_modeling_sessions = prepare_modeling_sessions(self.modeling_settings)

        print("Loading behavioral feature data...")
        beh_feature_data_dict, camera_fr_dict, mouse_track_names_dict = load_behavioral_feature_data(
            behavior_file_paths=txt_modeling_sessions,
            csv_sep=self.modeling_settings['io']['csv_separator']
        )
        print("Loading USV data and selecting epochs...")

        usv_data_dict = find_bout_epochs(
            root_directories=txt_modeling_sessions,
            mouse_ids_dict=mouse_track_names_dict,
            camera_fps_dict=camera_fr_dict,
            features_dict=beh_feature_data_dict,
            csv_sep=self.modeling_settings['io']['csv_separator'],
            gmm_component_index=self.modeling_settings['model_params']['gmm_component_index'],
            gmm_z_score=self.modeling_settings['model_params']['gmm_z_score'],
            gmm_params=self.modeling_settings['gmm_params'],
            noise_vocal_categories=self.modeling_settings['vocal_features']['usv_noise_categories'],
            proportion_smoothing_sd=self.modeling_settings['vocal_features']['usv_predictor_smoothing_sd'],
            vocal_output_type=self.modeling_settings['vocal_features']['usv_predictor_type'],
            filter_history=self.modeling_settings['model_params']['filter_history'],
            prediction_mode=self.modeling_settings['model_params']['model_target_vocal_type'],
            usv_bout_time=self.modeling_settings['model_params']['usv_bout_time'],
            min_usv_per_bout=self.modeling_settings['model_params']['usv_per_bout_floor']
        )

        predictor_mouse_idx = self.modeling_settings['model_params']['model_predictor_mouse_index']
        target_mouse_idx = abs(predictor_mouse_idx - 1)

        sessions_to_remove = identify_empty_event_sessions(
            usv_data_dict=usv_data_dict,
            mouse_names_dict=mouse_track_names_dict,
            target_idx=target_mouse_idx,
            event_key='positive_events',
            warn_label='session'
        )
        for sess in sessions_to_remove:
            if sess in beh_feature_data_dict:
                del beh_feature_data_dict[sess]

        print(f"Proceeding with {len(beh_feature_data_dict)} sessions after filtering empty ones.")

        processed_beh_feature_data_dict = {}
        kin_settings = self.modeling_settings['kinematic_features']
        voc_settings = self.modeling_settings['vocal_features']

        for behavioral_session in list(beh_feature_data_dict.keys()):
            (predictor_mouse_idx,
             target_mouse_idx,
             predictor_mouse_name,
             target_mouse_name) = resolve_mouse_roles(
                modeling_settings=self.modeling_settings,
                mouse_names_dict=mouse_track_names_dict,
                session_id=behavioral_session
            )

            session_df_columns = beh_feature_data_dict[behavioral_session].columns

            columns_to_keep_session = select_kinematic_columns(
                session_df_columns=session_df_columns,
                target_name=target_mouse_name,
                predictor_name=predictor_mouse_name,
                kin_settings=kin_settings,
                predictor_idx=predictor_mouse_idx
            )

            new_voc_cols, new_voc_col_names = build_vocal_signal_columns(
                usv_data_dict=usv_data_dict,
                session_id=behavioral_session,
                target_name=target_mouse_name,
                predictor_name=predictor_mouse_name,
                voc_settings=voc_settings
            )

            columns_to_keep_session = sorted(set(columns_to_keep_session) | set(new_voc_col_names))
            existing_cols_to_select = [c for c in columns_to_keep_session if c in session_df_columns]
            current_df = beh_feature_data_dict[behavioral_session].select(existing_cols_to_select)

            if new_voc_cols:
                current_df = current_df.with_columns(new_voc_cols)

            processed_beh_feature_data_dict[behavioral_session] = current_df

        del beh_feature_data_dict
        print("Feature filtering complete.")

        if not processed_beh_feature_data_dict:
            print("Error: No sessions remaining after filtering.")
            return

        revised_behavioral_predictors = collect_predictor_suffixes(processed_beh_feature_data_dict)
        if not revised_behavioral_predictors:
            raise ValueError("No features selected.")
        print(f"Final feature suffixes selected: {revised_behavioral_predictors}")

        print("Standardizing columns across sessions...")
        processed_beh_feature_data_dict = zero_fill_missing_feature_columns(
            processed_beh_dict=processed_beh_feature_data_dict,
            mouse_names_dict=mouse_track_names_dict,
            target_idx=target_mouse_idx,
            predictor_idx=predictor_mouse_idx,
            suffixes=revised_behavioral_predictors,
            voc_settings=voc_settings,
            session_list_file=self.modeling_settings['io']['session_list_file'],
            skip_dyadic_suffixes=True
        )

        print("Z-scoring features across sessions...")
        processed_beh_feature_data_dict = zscore_features_across_sessions(
            processed_beh_dict=processed_beh_feature_data_dict,
            suffixes=revised_behavioral_predictors,
            feature_bounds=getattr(self, 'feature_boundaries', {}),
            abs_features=['allo_roll', 'allo_yaw-nose', 'nose-allo_yaw', 'allo_yaw-TTI', 'TTI-allo_yaw']
        )
        print("Z-scoring complete.")

        print("Extracting epochs per session and renaming features...")
        modeling_final_data_dict = {}

        for session_idx, beh_session_id in enumerate(tqdm(processed_beh_feature_data_dict.keys(), desc='Extracting Epochs')):
            session_df = processed_beh_feature_data_dict[beh_session_id]
            predictor_mouse_name = mouse_track_names_dict[beh_session_id][predictor_mouse_idx]
            target_mouse_name = mouse_track_names_dict[beh_session_id][target_mouse_idx]

            try:
                usv_event_times = usv_data_dict[beh_session_id][target_mouse_name]['positive_events']
                no_usv_event_times = usv_data_dict[beh_session_id][target_mouse_name]['negative_events']
            except KeyError:
                continue

            history_frames = self.history_frames
            session_sampling_rate = camera_fr_dict[beh_session_id]

            for full_column_name in session_df.columns:

                base_feature = full_column_name.split('.')[-1]
                if base_feature.isdigit(): continue

                # Determine generic key based on prefix
                if full_column_name.startswith(f"{target_mouse_name}."):
                    generic_key = f"self.{base_feature}"
                elif full_column_name.startswith(f"{predictor_mouse_name}."):
                    generic_key = f"other.{base_feature}"
                else:
                    generic_key = base_feature

                if generic_key not in modeling_final_data_dict: modeling_final_data_dict[generic_key] = {}
                if beh_session_id not in modeling_final_data_dict[generic_key]:
                    modeling_final_data_dict[generic_key][beh_session_id] = {}

                # Extract Epochs
                usv_feature_arr = np.full((usv_event_times.size, history_frames), np.nan)
                no_usv_feature_arr = np.full((no_usv_event_times.size, history_frames), np.nan)
                column_data_np = session_df[full_column_name].to_numpy()
                max_frame_idx = len(column_data_np)

                def fill_arr(times, arr):
                    ends = np.round(times * session_sampling_rate).astype(int)
                    starts = ends - history_frames
                    for i in range(times.size):
                        s, e = starts[i], ends[i]
                        if s >= 0 and e <= max_frame_idx:
                            chunk = column_data_np[s:e].copy()
                            chunk[np.isnan(chunk)] = 0.0
                            arr[i, :] = chunk

                fill_arr(usv_event_times, usv_feature_arr)
                fill_arr(no_usv_event_times, no_usv_feature_arr)

                modeling_final_data_dict[generic_key][beh_session_id]['usv_feature_arr'] = usv_feature_arr
                modeling_final_data_dict[generic_key][beh_session_id]['no_usv_feature_arr'] = no_usv_feature_arr
        print("Epoch extraction and renaming complete.")

        if not modeling_final_data_dict:
            print("Warning: Final data dictionary is empty.")
            return

        final_covariate_names = sorted(list(modeling_final_data_dict.keys()))
        total_covariates = len(final_covariate_names)
        first_feat = final_covariate_names[0]
        all_sessions = sorted(list(modeling_final_data_dict[first_feat].keys()))

        alignment_passed = True
        mismatched_sessions = []

        for sess_id in all_sessions:
            # Reference counts for this specific session
            ref_usv_n = modeling_final_data_dict[first_feat][sess_id]['usv_feature_arr'].shape[0]
            ref_none_n = modeling_final_data_dict[first_feat][sess_id]['no_usv_feature_arr'].shape[0]

            for feat in final_covariate_names[1:]:
                feat_usv_n = modeling_final_data_dict[feat][sess_id]['usv_feature_arr'].shape[0]
                feat_none_n = modeling_final_data_dict[feat][sess_id]['no_usv_feature_arr'].shape[0]

                if feat_usv_n != ref_usv_n or feat_none_n != ref_none_n:
                    alignment_passed = False
                    mismatched_sessions.append(sess_id)
                    break

        grand_total_sessions = len(all_sessions)
        grand_total_usv = sum(modeling_final_data_dict[first_feat][s]['usv_feature_arr'].shape[0] for s in all_sessions)
        grand_total_none = sum(modeling_final_data_dict[first_feat][s]['no_usv_feature_arr'].shape[0] for s in all_sessions)

        print("\n" + "=" * 105)
        print(f"MODELING INPUT: GLOBAL AGGREGATE SUMMARY")
        print("=" * 105)
        print(f"{'#':<4} {'Feature Name':<40} | {'Sess':<6} | {'USV Bouts':<12} | {'No-USV Bouts':<12} | {'Total N'}")
        print("-" * 105)

        for i, feat in enumerate(final_covariate_names, 1):
            feat_usv_bouts = sum(modeling_final_data_dict[feat][s]['usv_feature_arr'].shape[0] for s in modeling_final_data_dict[feat])
            feat_none_bouts = sum(modeling_final_data_dict[feat][s]['no_usv_feature_arr'].shape[0] for s in modeling_final_data_dict[feat])

            print(f"{i:3}. {feat:<40} | {len(modeling_final_data_dict[feat]):<6} | {feat_usv_bouts:<12} | {feat_none_bouts:<12} | {feat_usv_bouts + feat_none_bouts:<8}")

        print("-" * 105)
        print(f"PROJECT-WIDE TOTAL TALLY:")
        print(f"  > Total Unique Features:        {total_covariates}")
        print(f"  > Total Sessions:               {grand_total_sessions}")
        print(f"  > Total USV Bouts:              {grand_total_usv}")
        print(f"  > Total No-USV Bouts:           {grand_total_none}")
        print(f"  > Grand Total Epochs (N):       {grand_total_usv + grand_total_none}")
        print(f"  > INTRA-SESSION ALIGNMENT:      {'PASSED (True)' if alignment_passed else 'FAILED (False)'}")

        if not alignment_passed:
            print(f"  [!] ALERT: Dimensional mismatch in sessions: {set(mismatched_sessions)}")
        print("=" * 105 + "\n")

        target_mouse_sex = 'male' if target_mouse_idx == 0 else 'female'
        max_hist_sec = self.modeling_settings['model_params']['filter_history']
        file_name_ = f"modeling_{self.modeling_settings['model_params']['model_target_vocal_type']}_gmm{self.modeling_settings['model_params']['gmm_component_index']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(txt_modeling_sessions)}sess_hist{max_hist_sec}s.pkl"
        save_path = os.path.join(self.modeling_settings['io']['save_directory'], file_name_)

        try:
            os.makedirs(self.modeling_settings['io']['save_directory'], exist_ok=True)
            with open(save_path, 'wb') as f:
                pickle.dump(modeling_final_data_dict, f)
            print(f"\n[+] Successfully saved PER-SESSION renamed modeling input data to:\n    {save_path}")
        except Exception as e:
            print(f"\n[!] Error saving final pickle file: {e}")

    def create_data_splits(self, feature_data: dict, strategy_override: str = None):
        """
        A generator that yields train/test splits based on the 'split_strategy'.

        This function reads 'self.modeling_settings' for 'split_strategy',
        'num_splits', and 'test_proportion'. It then yields balanced and
        shuffled train/test sets according to the chosen strategy.

        Strategies:
        - 'mixed': Pools all sessions, balances the total data (Bout vs. No-Bout),
          then creates 'n_splits' using StratifiedShuffleSplit based on 'test_proportion'.
        - 'session': Splits the *list* of sessions into train/test sets 'n_splits'
          times using ShuffleSplit. The training data (pooled from
          training sessions) is balanced per-split. The test data remains unbalanced.
        - 'null_control': (Like 'mixed') Pools all *No-Bout* epochs, creates two
          fake balanced classes, and runs StratifiedShuffleSplit.
        - 'session_null_control': (Like 'session') Splits sessions into train/test.
          Creates a fake balanced training set from the *training sessions'* No-Bout
          data that **matches the exact size** of the *actual* balanced training set.
          Creates a fake unbalanced test set from the *test sessions'* No-Bout
          data that **matches the exact size and class ratio** of the *actual* test set.

        Parameters
        ----------
        feature_data : dict
            The data dictionary for a *single feature* (e.g.,
            `modeling_feature_arr_dict['self.speed']`), where keys are session IDs and
            values contain 'usv_feature_arr' and 'no_usv_feature_arr'.
        strategy_override : str, optional
            If provided, this strategy will be used instead of the one in settings.

        Yields
        ------
        tuple
            A tuple of (X_train, y_train, X_test, y_test) for each split.
        """

        split_strategy = self.modeling_settings['model_params']['split_strategy']
        if strategy_override:
            split_strategy = strategy_override

        n_splits = self.modeling_settings['model_params']['split_num']
        test_proportion = self.modeling_settings['model_params']['test_proportion']
        random_state = self.modeling_settings['model_params']['random_seed']

        all_sessions = list(feature_data.keys())
        X_pos_all, X_neg_all = pool_session_arrays(feature_data, all_sessions, pos_key="usv_feature_arr", neg_key="no_usv_feature_arr", n_frames=self.history_frames)
        n_pos_total = X_pos_all.shape[0]
        n_neg_total = X_neg_all.shape[0]

        ### Strategy 1: 'mixed' (all sessions together)
        if split_strategy == 'mixed':
            X_pos, X_neg = balance_two_class_arrays(X_pos_all, X_neg_all)

            if X_pos.shape[0] == 0:
                print(f"Warning: No balanced data for feature. Skipping splits.")
                return

            X, y = concat_two_class_with_labels(X_pos, X_neg)

            print(f"--- 'mixed' strategy: Created balanced dataset of {X.shape[0]} samples.")

            sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_proportion, random_state=random_state)

            for train_idx, test_idx in sss.split(X, y):
                yield X[train_idx], y[train_idx], X[test_idx], y[test_idx]

        ### Strategy 2: 'session' (some sessions go to train, some to test)
        elif split_strategy == 'session':
            all_sessions_array = np.array(all_sessions)
            n_sessions = len(all_sessions_array)

            actual_test_proportion = bounded_test_proportion(test_proportion, n_sessions)
            if n_sessions * (1 - actual_test_proportion) < 1:
                print(f"Warning: test_proportion ({test_proportion}) too high for {n_sessions} sessions. Skipping.")
                return

            ss = ShuffleSplit(n_splits=n_splits, test_size=actual_test_proportion, random_state=random_state)

            split_num = 0
            for train_session_idx, test_session_idx in ss.split(all_sessions_array):
                split_num += 1
                train_session_list = all_sessions_array[train_session_idx]
                test_session_list = all_sessions_array[test_session_idx]

                X_pos_train, X_neg_train = pool_session_arrays(feature_data, train_session_list, pos_key="usv_feature_arr", neg_key="no_usv_feature_arr", n_frames=self.history_frames)
                X_pos_test, X_neg_test = pool_session_arrays(feature_data, test_session_list, pos_key="usv_feature_arr", neg_key="no_usv_feature_arr", n_frames=self.history_frames)

                # Balance the *training set ONLY
                X_pos_train_bal, X_neg_train_bal = balance_two_class_arrays(X_pos_train, X_neg_train)

                if X_pos_train_bal.shape[0] == 0:
                    print(f"Warning: No balanced training data for split {split_num}. Skipping.")
                    continue

                # Create final train arrays (balanced)
                X_train, y_train = concat_two_class_with_labels(X_pos_train_bal, X_neg_train_bal)

                # Create final test arrays (NB: unbalanced!)
                X_test, y_test = concat_two_class_with_labels(X_pos_test, X_neg_test)

                yield shuffle_train_test_arrays(X_train, y_train, X_test, y_test)

        ### Strategy 3: 'null_control' (pooled no-bout, matching 'mixed' size)
        elif split_strategy == 'null_control':
            print("--- Using 'null_control' (pooled) strategy (No-Bout vs. No-Bout) ---")

            n_balanced_samples = n_pos_total
            print(f"  Targeting {n_balanced_samples} samples per class to match 'mixed' strategy.")

            if n_neg_total < n_balanced_samples * 2:
                print(f"Warning: Not enough No-Bout samples ({n_neg_total}) to create null control of size {n_balanced_samples * 2}. Sampling with replacement.")
                shuffled_indices = np.random.choice(n_neg_total, size=n_balanced_samples * 2, replace=True)
            else:
                shuffled_indices = np.random.permutation(n_neg_total)

            X_fake_pos = X_neg_all[shuffled_indices[:n_balanced_samples]]
            X_fake_neg = X_neg_all[shuffled_indices[n_balanced_samples: n_balanced_samples * 2]]

            X, y = concat_two_class_with_labels(X_fake_pos, X_fake_neg)

            print(f"  Created null dataset of {X.shape[0]} samples ({X_fake_pos.shape[0]} fake_pos, {X_fake_neg.shape[0]} fake_neg) ---")

            sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_proportion, random_state=random_state)

            for train_idx, test_idx in sss.split(X, y):
                yield X[train_idx], y[train_idx], X[test_idx], y[test_idx]

        ### Strategy 4: 'session_null_control' (session-split no-bout, matching 'session' size)
        elif split_strategy == 'session_null_control':
            print("--- Using 'session_null_control' (session-split) strategy ---")
            all_sessions_array = np.array(all_sessions)
            n_sessions = len(all_sessions_array)

            actual_test_proportion = bounded_test_proportion(test_proportion, n_sessions)
            if n_sessions * (1 - actual_test_proportion) < 1:
                print(f"Warning: test_proportion ({test_proportion}) too high for {n_sessions} sessions. Skipping.")
                return

            ss = ShuffleSplit(n_splits=n_splits, test_size=actual_test_proportion, random_state=random_state)

            split_num = 0
            for train_session_idx, test_session_idx in ss.split(all_sessions_array):
                split_num += 1
                train_session_list = all_sessions_array[train_session_idx]
                test_session_list = all_sessions_array[test_session_idx]

                # Get actual data counts for this split to match them
                X_pos_train_actual, X_neg_train_actual = pool_session_arrays(feature_data, train_session_list, pos_key="usv_feature_arr", neg_key="no_usv_feature_arr", n_frames=self.history_frames)
                X_pos_test_actual, X_neg_test_actual = pool_session_arrays(feature_data, test_session_list, pos_key="usv_feature_arr", neg_key="no_usv_feature_arr", n_frames=self.history_frames)

                # Find the target sizes from the actual data
                X_pos_train_bal, X_neg_train_bal = balance_two_class_arrays(X_pos_train_actual, X_neg_train_actual)

                n_balanced_train_half = X_pos_train_bal.shape[0]
                n_total_train_needed = n_balanced_train_half * 2

                n_test_pos_target = X_pos_test_actual.shape[0]
                n_test_neg_target = X_neg_test_actual.shape[0]

                if n_balanced_train_half == 0:
                    print(f"Warning: No *actual* balanced training data for split {split_num}. Cannot match size. Skipping.")
                    continue

                # Pool ONLY No-Bout data from selected sessions
                _, X_neg_train_all = pool_session_arrays(feature_data, train_session_list, pos_key="usv_feature_arr", neg_key="no_usv_feature_arr", n_frames=self.history_frames)
                _, X_neg_test_all = pool_session_arrays(feature_data, test_session_list, pos_key="usv_feature_arr", neg_key="no_usv_feature_arr", n_frames=self.history_frames)

                # Create fake balanced training set, matching 'session' size
                n_train_neg_available = X_neg_train_all.shape[0]

                if n_train_neg_available < n_total_train_needed:
                    print(f"Warning: Not enough No-Bout samples ({n_train_neg_available}) in train sessions to create null control of size {n_total_train_needed}. Sampling with replacement.")
                    train_neg_indices = np.random.choice(n_train_neg_available, size=n_total_train_needed, replace=True)
                else:
                    train_neg_indices = np.random.permutation(n_train_neg_available)

                X_fake_pos_train = X_neg_train_all[train_neg_indices[:n_balanced_train_half]]
                X_fake_neg_train = X_neg_train_all[train_neg_indices[n_balanced_train_half: n_total_train_needed]]

                X_train, y_train = concat_two_class_with_labels(X_fake_pos_train, X_fake_neg_train)

                # Create fake UNBALANCED test set, matching 'session' size AND ratio
                n_test_neg_available = X_neg_test_all.shape[0]

                if n_test_neg_available == 0:
                    print(f"Warning: No No-Bout test samples available for split {split_num}. Skipping.")
                    continue

                fake_pos_indices = np.random.choice(n_test_neg_available, size=n_test_pos_target, replace=True)
                fake_neg_indices = np.random.choice(n_test_neg_available, size=n_test_neg_target, replace=True)

                X_fake_pos_test = X_neg_test_all[fake_pos_indices]
                X_fake_neg_test = X_neg_test_all[fake_neg_indices]

                X_test, y_test = concat_two_class_with_labels(X_fake_pos_test, X_fake_neg_test)

                # Shuffle the final arrays before yielding
                yield shuffle_train_test_arrays(X_train, y_train, X_test, y_test)

        else:
            raise ValueError(f"Unknown split_strategy: {split_strategy}. Must be 'mixed', 'session', 'null_control', or 'session_null_control'.")

    def _run_model_for_feature_sklearn(self,
                                     feature_name: str,
                                     feature_data: dict,
                                     basis_matrix: np.ndarray
                                     ) -> tuple[str, dict]:
        """
        Executes a univariate Logistic Regression analysis for a single behavioral or vocal feature.

        This method serves as the core computational engine for the modeling pipeline. It processes
        high-dimensional temporal history data by projecting it onto a reduced basis set to
        extract smooth kernels (filters). It supports parallel execution and implements
        rigorous statistical validation through balanced training and shuffled-label
        null-hypothesis testing.

        The computational workflow includes:
        1.  Projects raw temporal history ([N_events x N_lags])
            onto the provided basis matrix ([N_lags x N_bases]) to obtain compressed feature
            representations ([N_events x N_bases]).
        2.  Utilizes a generator-based splitting strategy (mixed
            sample-pooling or session-exclusive splits) to create independent training
            and testing sets.
        3.  Fits a `LogisticRegressionCV` model with
            integrated Cross-Validation to determine the optimal L1/L2 regularization
            strength (C) for the projected feature set.
        4.  Maps the resulting coefficients back into the
            temporal domain ([coefs x basis.T]) to recover the underlying linear
            filter shape.
        5.  Optionally repeats the fit on shuffled labels to
            establish a baseline performance metric (AUC/Log-Loss) for significance testing.

        Parameters
        ----------
        feature_name : str
            The identifier for the feature being modeled (e.g., 'self.speed', 'other.usv_cat_1').
        feature_data : dict
            A nested dictionary containing session-specific NumPy arrays for positive
            ('usv_feature_arr') and negative ('no_usv_feature_arr') class samples.
        basis_matrix : np.ndarray
            The projection matrix [frames, bases] used to constrain the temporal
            filter complexity.

        Returns
        -------
        tuple[str, dict]
            A tuple containing the feature name and a results' dictionary. The dictionary
            contains 'actual' and 'shuffled' sub-structures holding arrays for:
            - filter_shapes: The reconstructed temporal filters across splits.
            - coefs_projected: The raw weights in the basis space.
            - auc, score, log_loss: Standard performance metrics for binary classification.
            - split_sizes: Metadata tracking the N for each training/test fold.
        """

        # Initialize results structure with 'actual' and 'shuffled' keys
        n_splits = self.modeling_settings['model_params']['split_num']
        n_bases = basis_matrix.shape[1]
        history_frames = basis_matrix.shape[0]

        results = {
            'actual': {
                'filter_shapes': np.full((n_splits, history_frames), np.nan),
                'coefs_projected': np.full((n_splits, n_bases), np.nan),
                'optimal_C': np.full(n_splits, np.nan), 'score': np.full(n_splits, np.nan),
                'precision': np.full(n_splits, np.nan), 'recall': np.full(n_splits, np.nan),
                'f1': np.full(n_splits, np.nan), 'auc': np.full(n_splits, np.nan),
                'll': np.full(n_splits, np.nan)
            },
            'shuffled': {
                'filter_shapes': np.full((n_splits, history_frames), np.nan),
                'coefs_projected': np.full((n_splits, n_bases), np.nan),
                'optimal_C': np.full(n_splits, np.nan), 'score': np.full(n_splits, np.nan),
                'precision': np.full(n_splits, np.nan), 'recall': np.full(n_splits, np.nan),
                'f1': np.full(n_splits, np.nan), 'auc': np.full(n_splits, np.nan),
                'll': np.full(n_splits, np.nan)
            },
            'split_sizes': {'train': [], 'test': []}
        }

        data_splitter = self.create_data_splits(feature_data)
        split_has_data = False

        for split_idx, (X_train, y_train, X_test, y_test) in enumerate(data_splitter):

            if X_train.shape[0] == 0 or X_test.shape[0] == 0:
                continue
            split_has_data = True
            print(f"  Feature '{feature_name}', Split {split_idx}: Train={X_train.shape}, Test={X_test.shape}")

            results['split_sizes']['train'].append(X_train.shape[0])
            results['split_sizes']['test'].append(X_test.shape[0])

            X_train_proj = np.dot(X_train, basis_matrix)
            X_test_proj = np.dot(X_test, basis_matrix)

            try:
                lr_actual = LogisticRegressionCV(
                    penalty=self.modeling_settings['hyperparameters']['classical']['logistic_regression']['penalty'],
                    Cs=self.modeling_settings['hyperparameters']['classical']['logistic_regression']['cs'],
                    cv=self.modeling_settings['hyperparameters']['classical']['logistic_regression']['cv'],
                    class_weight='balanced',
                    solver=self.modeling_settings['hyperparameters']['classical']['logistic_regression']['solver'],
                    max_iter=self.modeling_settings['hyperparameters']['classical']['logistic_regression']['max_iter'],
                    random_state=self.modeling_settings['model_params']['random_seed']
                ).fit(X_train_proj, y_train)

                y_pred_actual = lr_actual.predict(X_test_proj)
                y_proba_actual = lr_actual.predict_proba(X_test_proj)[:, 1]

                results['actual']['coefs_projected'][split_idx, :] = lr_actual.coef_.flatten()
                results['actual']['optimal_C'][split_idx] = lr_actual.C_[0]
                filter_shape_actual = np.dot(lr_actual.coef_, basis_matrix.T).ravel()
                results['actual']['filter_shapes'][split_idx, :] = filter_shape_actual
                results['actual']['score'][split_idx] = lr_actual.score(X_test_proj, y_test)
                results['actual']['precision'][split_idx] = precision_score(y_test, y_pred_actual, zero_division=0.0)
                results['actual']['recall'][split_idx] = recall_score(y_test, y_pred_actual, zero_division=0.0)
                results['actual']['f1'][split_idx] = f1_score(y_test, y_pred_actual, average='binary', zero_division=0.0)

                if len(np.unique(y_test)) > 1:
                    results['actual']['auc'][split_idx] = roc_auc_score(y_test, y_proba_actual)
                    epsilon = 1e-15
                    y_proba_actual_clipped = np.clip(y_proba_actual, epsilon, 1 - epsilon)
                    results['actual']['ll'][split_idx] = log_loss(y_test, y_proba_actual_clipped)

            except Exception as e:
                print(f"  ERROR during ACTUAL model fit/predict for {feature_name}, split {split_idx}: {e}")

            try:
                shuffle_seed = self.modeling_settings['model_params']['random_seed']
                if shuffle_seed is not None: np.random.seed(shuffle_seed + split_idx + 1)  # Offset seed
                y_train_shuffled = np.random.permutation(y_train)

                lr_shuffled = LogisticRegressionCV(
                    penalty=self.modeling_settings['hyperparameters']['classical']['logistic_regression']['penalty'],
                    Cs=self.modeling_settings['hyperparameters']['classical']['logistic_regression']['cs'],
                    cv=self.modeling_settings['hyperparameters']['classical']['logistic_regression']['cv'],
                    class_weight='balanced',
                    solver=self.modeling_settings['hyperparameters']['classical']['logistic_regression']['solver'],
                    max_iter=self.modeling_settings['hyperparameters']['classical']['logistic_regression']['max_iter'],
                    random_state=self.modeling_settings['model_params']['random_seed']
                ).fit(X_train_proj, y_train_shuffled)

                y_pred_shuffled = lr_shuffled.predict(X_test_proj)
                y_proba_shuffled = lr_shuffled.predict_proba(X_test_proj)[:, 1]

                results['shuffled']['coefs_projected'][split_idx, :] = lr_shuffled.coef_.flatten()
                results['shuffled']['optimal_C'][split_idx] = lr_shuffled.C_[0]
                filter_shape_shuffled = np.dot(lr_shuffled.coef_, basis_matrix.T).ravel()
                results['shuffled']['filter_shapes'][split_idx, :] = filter_shape_shuffled
                results['shuffled']['score'][split_idx] = lr_shuffled.score(X_test_proj, y_test)
                results['shuffled']['precision'][split_idx] = precision_score(y_test, y_pred_shuffled, zero_division=0.0)
                results['shuffled']['recall'][split_idx] = recall_score(y_test, y_pred_shuffled, zero_division=0.0)
                results['shuffled']['f1'][split_idx] = f1_score(y_test, y_pred_shuffled, average='binary', zero_division=0.0)

                if len(np.unique(y_test)) > 1:
                    results['shuffled']['auc'][split_idx] = roc_auc_score(y_test, y_proba_shuffled)
                    epsilon = 1e-15
                    y_proba_shuffled_clipped = np.clip(y_proba_shuffled, epsilon, 1 - epsilon)
                    results['shuffled']['ll'][split_idx] = log_loss(y_test, y_proba_shuffled_clipped)

            except Exception as e:
                print(f"  ERROR during SHUFFLED model fit/predict for {feature_name}, split {split_idx}: {e}")

        results['split_sizes']['train'] = np.array(results['split_sizes']['train'])
        results['split_sizes']['test'] = np.array(results['split_sizes']['test'])

        if split_has_data:
            mean_auc_actual = np.nanmean(results['actual']['auc'])
            print_msg = f"  --- Finished {feature_name}. Mean Actual AUC: {mean_auc_actual:.4f}"
            mean_auc_shuffled = np.nanmean(results['shuffled']['auc'])
            print_msg += f", Mean Shuffled AUC: {mean_auc_shuffled:.4f}"
            print(print_msg)
        else:
            print(f"  --- No valid splits processed for feature: {feature_name} ---")

        return feature_name, results

    def _run_model_for_feature_pygam(self,
                                   feature_name: str,
                                   feature_data: dict,
                                   basis_matrix: np.ndarray  # NOT USED
                                   ) -> tuple[str, dict]:
        """
        Runs modeling analysis for a single feature using pygam's LogisticGAM
        with a 2D tensor product spline (te) to find a smooth filter.

        This function runs two models in parallel within a single loop for each split:
        1.  Fits a model using the `split_strategy` from settings
            (e.g., 'session') on the real "Bout vs. No-Bout" data.
        2.  Fits a *null control* model by forcing the
            `create_data_splits` function to use a null strategy
            (e.g., 'session_null_control') on "No-Bout vs. No-Bout" data.

        Tho code covers the following steps:
        1.  It fits a `pygam.LogisticGAM` model. The core of
            this model is a `te(0, 1, ...)` term (tensor product spline). This
            fits a single, smooth 2D *surface* that represents the log-odds
            contribution based on the *interaction* between the feature's value
            (axis 0) and the time lag (axis 1). This is a non-linear filter.
        2.  The `pygam` library automatically applies a
            smoothness penalty (penalizing the "wiggleness" of the 2D surface)
            and finds the optimal penalty strength using Generalized Cross-Validation (GCV).
        3.  It calculates metrics for both the 'actual' data
            and a 'shuffled' (null) model. Note: For evaluation, the tiled
            probabilities from the test set are averaged per-epoch before
            calculating scores like AUC, F1, etc., against the original y_test labels.
        4.  It extracts the 1D *linear filter shape* by
            calculating the partial dependence (the difference in log-odds) for
            a one-unit change in the feature value (from 0 to 1) at each time point.
        5.  The function returns the feature name and a nested dictionary
            containing all results.

        Parameters
        ----------
        feature_name : str
            The name of the behavioral feature (e.g., 'self.speed').
        feature_data : dict
            The data dictionary for this single feature, containing session IDs as
            keys, which in turn contain 'usv_feature_arr' and 'no_usv_feature_arr'.
        basis_matrix : np.ndarray
            This argument is **ignored** by the `pygam` method but is included
            for API compatibility with the `run_model_analysis_parallel` caller.

        Returns
        -------
        tuple[str, dict]
            A tuple containing:
            1. feature_name (str): The name of the feature analyzed.
            2. results (dict): A nested dictionary containing 'actual' and 'shuffled'
               keys, each holding arrays of metrics (e.g., 'filter_shapes', 'auc',
               'score', 'precision', 'recall', 'f1', 'll'). Also contains a
               'split_sizes' key. 'shuffled' metrics are NaN if not calculated.
        """

        print(f"--- Running [pygam] analysis for feature: {feature_name} ---")

        n_splits = self.modeling_settings['model_params']['split_num']
        history_frames = int(np.floor(self.modeling_settings['io']['camera_sampling_rate'] * self.modeling_settings['model_params']['filter_history']))

        try:
            pygam_params = self.modeling_settings['hyperparameters']['classical']['pygam']
            n_splines_time = pygam_params['n_splines_time']
            n_splines_value = pygam_params['n_splines_value']
            lam_penalty = pygam_params['lam_penalty']
            max_iterations = pygam_params['max_iterations']
            tol_val = pygam_params['tol_val']
        except KeyError:
            n_splines_time, n_splines_value, lam_penalty, max_iterations, tol_val = 8, 5, 0.6, 100, 1e-4

        gam_kwargs_actual = {
            'max_iter': max_iterations,
            'tol': tol_val
        }

        if lam_penalty is not None:
            gam_kwargs_actual['lam'] = lam_penalty
            print(f"  Using FIXED smoothness penalty: lam={lam_penalty}")
        else:
            print("  Using GCV to find optimal smoothness (lam=None)")

        gam_kwargs_shuffled = {
            'max_iter': max_iterations,
            'tol': tol_val,
            'lam': lam_penalty
        }

        results = {
            'actual': {
                'filter_shapes': np.full((n_splits, history_frames), np.nan),
                'coefs_projected': np.empty((n_splits, 0)),
                'optimal_C': np.empty((n_splits, 0)),
                'score': np.full(n_splits, np.nan), 'precision': np.full(n_splits, np.nan),
                'recall': np.full(n_splits, np.nan), 'f1': np.full(n_splits, np.nan),
                'auc': np.full(n_splits, np.nan), 'll': np.full(n_splits, np.nan)
            },
            'shuffled': {
                'filter_shapes': np.full((n_splits, history_frames), np.nan),
                'coefs_projected': np.empty((n_splits, 0)), 'optimal_C': np.empty((n_splits, 0)),
                'score': np.full(n_splits, np.nan), 'precision': np.full(n_splits, np.nan),
                'recall': np.full(n_splits, np.nan), 'f1': np.full(n_splits, np.nan),
                'auc': np.full(n_splits, np.nan), 'll': np.full(n_splits, np.nan)
            },
            'split_sizes': {'train': [], 'test': []}
        }

        time_indices = np.arange(history_frames, dtype=np.float32)

        def unroll_data_for_gam(X):
            return unroll_history_matrix(X, time_indices=time_indices)

        actual_data_splitter = self.create_data_splits(feature_data, strategy_override=None)

        current_strategy = self.modeling_settings['model_params']['split_strategy']
        null_strategy = 'session_null_control' if current_strategy == 'session' else 'null_control'
        shuffled_data_splitter = self.create_data_splits(feature_data, strategy_override=null_strategy)

        split_has_data_actual = False

        for split_idx, (actual_split, shuffled_split) in enumerate(zip(actual_data_splitter, shuffled_data_splitter)):

            if split_idx >= n_splits:
                break

            if actual_split and actual_split[0].shape[0] > 0 and actual_split[2].shape[0] > 0:
                (X_train, y_train, X_test, y_test) = actual_split
                split_has_data_actual = True
                print(f"  ACTUAL Split {split_idx}: Train={X_train.shape}, Test={X_test.shape}")

                results['split_sizes']['train'].append(X_train.shape[0])
                results['split_sizes']['test'].append(X_test.shape[0])

                y_train_tiled = np.repeat(y_train.astype(np.float32), history_frames)
                y_test_int = y_test.astype(int)
                X_train_gam = unroll_data_for_gam(X_train.astype(np.float32))
                X_test_gam = unroll_data_for_gam(X_test.astype(np.float32))

                try:
                    gam_actual = LogisticGAM(
                        te(0, 1, n_splines=[n_splines_value, n_splines_time]), **gam_kwargs_actual
                    ).fit(X_train_gam, y_train_tiled)

                    diffs = gam_actual.logs_.get('diffs', [])
                    print(f"      Completed in {len(diffs)} iters | "
                          f"Final Δ: {diffs[-1] if diffs else 0.0:.2e} (Tol: {tol_val:.2e}) | "
                          f"Deviance: {gam_actual.logs_.get('deviance', [0.0])[-1]:.2f}")

                    y_proba_tiled = gam_actual.predict_proba(X_test_gam)
                    y_proba_mean_epoch = np.mean(y_proba_tiled.reshape(X_test.shape), axis=1)
                    y_pred_mean_epoch = (y_proba_mean_epoch > 0.5).astype(int)

                    grid_X_0 = np.stack([np.zeros(history_frames, dtype=np.float32), time_indices], axis=1)
                    grid_X_1 = np.stack([np.ones(history_frames, dtype=np.float32), time_indices], axis=1)
                    # predict_mu returns the Bernoulli mean (probability), not log-odds.
                    prob_0 = gam_actual.predict_mu(grid_X_0).astype(np.float32)
                    prob_1 = gam_actual.predict_mu(grid_X_1).astype(np.float32)

                    results['actual']['filter_shapes'][split_idx, :] = (prob_1 - prob_0).flatten()
                    results['actual']['score'][split_idx] = accuracy_score(y_test_int, y_pred_mean_epoch)
                    results['actual']['precision'][split_idx] = precision_score(y_test_int, y_pred_mean_epoch, zero_division=0.0)
                    results['actual']['recall'][split_idx] = recall_score(y_test_int, y_pred_mean_epoch, zero_division=0.0)
                    results['actual']['f1'][split_idx] = f1_score(y_test_int, y_pred_mean_epoch, average='binary', zero_division=0.0)
                    if len(np.unique(y_test_int)) > 1:
                        results['actual']['auc'][split_idx] = roc_auc_score(y_test_int, y_proba_mean_epoch)
                        results['actual']['ll'][split_idx] = log_loss(y_test_int, np.clip(y_proba_mean_epoch, 1e-15, 1 - 1e-15))

                    print(f"    > ACTUAL Fold {split_idx} (Train N={len(y_train)}, Test N={len(y_test)}): "
                          f"AUC={results['actual']['auc'][split_idx]:.3f}, "
                          f"LL={results['actual']['ll'][split_idx]:.3f}, "
                          f"Acc={results['actual']['score'][split_idx]:.3f}")

                except Exception as e:
                    print(f"  ERROR during ACTUAL [pygam] fit/predict for {feature_name}, split {split_idx}: {e}")

            if shuffled_split and shuffled_split[0].shape[0] > 0 and shuffled_split[2].shape[0] > 0:
                (X_train_null, y_train_null, X_test_null, y_test_null) = shuffled_split
                print(f"  SHUFFLED Split {split_idx}: Train={X_train_null.shape}, Test={X_test_null.shape}")

                y_train_tiled_null = np.repeat(y_train_null.astype(np.float32), history_frames)
                y_test_int_null = y_test_null.astype(int)
                X_train_gam_null = unroll_data_for_gam(X_train_null.astype(np.float32))
                X_test_gam_null = unroll_data_for_gam(X_test_null.astype(np.float32))

                try:
                    gam_shuffled = LogisticGAM(
                        te(0, 1, n_splines=[n_splines_value, n_splines_time]), **gam_kwargs_shuffled
                    ).fit(X_train_gam_null, y_train_tiled_null)

                    y_proba_shuffled_tiled = gam_shuffled.predict_proba(X_test_gam_null)
                    y_proba_shuffled_mean = np.mean(y_proba_shuffled_tiled.reshape(X_test_null.shape), axis=1)
                    y_pred_shuffled_mean = (y_proba_shuffled_mean > 0.5).astype(int)

                    grid_X_0_null = np.stack([np.zeros(history_frames, dtype=np.float32), time_indices], axis=1)
                    grid_X_1_null = np.stack([np.ones(history_frames, dtype=np.float32), time_indices], axis=1)
                    # predict_mu returns the Bernoulli mean (probability), not log-odds.
                    prob_0_null = gam_shuffled.predict_mu(grid_X_0_null).astype(np.float32)
                    prob_1_null = gam_shuffled.predict_mu(grid_X_1_null).astype(np.float32)
                    filter_shape_null = (prob_1_null - prob_0_null).flatten()
                    results['shuffled']['filter_shapes'][split_idx, :] = filter_shape_null

                    results['shuffled']['score'][split_idx] = accuracy_score(y_test_int_null, y_pred_shuffled_mean)
                    results['shuffled']['precision'][split_idx] = precision_score(y_test_int_null, y_pred_shuffled_mean, zero_division=0.0)
                    results['shuffled']['recall'][split_idx] = recall_score(y_test_int_null, y_pred_shuffled_mean, zero_division=0.0)
                    results['shuffled']['f1'][split_idx] = f1_score(y_test_int_null, y_pred_shuffled_mean, average='binary', zero_division=0.0)
                    if len(np.unique(y_test_int_null)) > 1:
                        results['shuffled']['auc'][split_idx] = roc_auc_score(y_test_int_null, y_proba_shuffled_mean)
                        results['shuffled']['ll'][split_idx] = log_loss(y_test_int_null, np.clip(y_proba_shuffled_mean, 1e-15, 1 - 1e-15))

                except Exception as e:
                    print(f"  ERROR during SHUFFLED [pygam] fit/predict for {feature_name}, split {split_idx}: {e}")

        results['split_sizes']['train'] = np.array(results['split_sizes']['train'])
        results['split_sizes']['test'] = np.array(results['split_sizes']['test'])

        if split_has_data_actual:
            mean_auc_actual = np.nanmean(results['actual']['auc'])
            print_msg = f"  --- Finished {feature_name} [pygam]. Mean Actual AUC: {mean_auc_actual:.4f}"
            mean_auc_shuffled = np.nanmean(results['shuffled']['auc'])
            print_msg += f", Mean Shuffled (Null Control) AUC: {mean_auc_shuffled:.4f}"
            print(print_msg)
        else:
            print(f"  --- No valid splits processed for feature: {feature_name} [pygam] ---")

        return feature_name, results
