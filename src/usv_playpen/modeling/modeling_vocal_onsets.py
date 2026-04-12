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
from sklearn.metrics import log_loss, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
from tqdm import tqdm

from .modeling_cross_session_normalization import zscore_different_sessions_together
from .load_input_files import load_behavioral_feature_data, find_bout_epochs
from ..analyses.compute_behavioral_features import FeatureZoo
from ..os_utils import configure_path


class GeneralizedLinearModelPipeline(FeatureZoo):

    def __init__(self, modeling_settings_dict, **kwargs):
        """
        Initializes the GeneralizedLinearModelPipeline class.

        Loads GLM settings from a dictionary or JSON file and initializes
        the parent FeatureZoo class. Calculates and stores `history_frames`.

        Parameters
        ----------
        modeling_settings_dict : dict or None
            Dictionary containing GLM settings. If None, settings are loaded
            from '_parameter_settings/modeling_settings.json'.
        **kwargs : dict
            Additional keyword arguments to set as attributes.
        """

        if modeling_settings_dict is None:
            settings_path = pathlib.Path(__file__).resolve().parent.parent / '_parameter_settings/modeling_settings.json'
            try:
                with open(settings_path, 'r') as settings_json_file:
                    self.modeling_settings_dict = json.load(settings_json_file)['modeling_settings']

            except FileNotFoundError:
                raise FileNotFoundError(f"Settings file not found at {settings_path}")
        else:
            self.modeling_settings_dict = modeling_settings_dict

        json_bounds = self.modeling_settings_dict.get('feature_boundaries')

        if json_bounds:
            self.feature_boundaries = json_bounds

        try:
            camera_rate = self.modeling_settings_dict['data_io']['camera_sampling_rate']
            filter_history_sec = self.modeling_settings_dict['features']['filter_history']
            self.history_frames = int(np.floor(camera_rate * filter_history_sec))
            print(f"History frames calculated: {self.history_frames} (for {filter_history_sec}s at {camera_rate}fps)")
        except KeyError as e:
            raise KeyError(f"Setting missing for history_frames calculation: {e}")

        FeatureZoo.__init__(self)

        for kw_arg, kw_val in kwargs.items():
            self.__dict__[kw_arg] = kw_val

    def extract_and_save_modeling_input_data_(self) -> None:
        """
        Extracts, processes, and saves behavioral and vocalization data PER SESSION for GLM analysis.

        This method orchestrates the data preparation pipeline for each session, specifically designed
        for classification tasks (USV vs. No-USV epochs). It incorporates advanced bout detection
        using Gaussian Mixture Models (GMM) to define dynamic inter-bout intervals (IBI).

        Pipeline Steps:
        1.  Calls `find_bout_epochs` to detect vocal bouts.
            - Uses `gmm_component_index` and `gmm_z_score` to calculate a dynamic IBI threshold per mouse.
            - Filters out specific noise categories (e.g., [0, 19]) via `noise_vocal_categories`.
            - Enforces `min_usv_per_bout` constraints.
            - Returns positive (`glm_usv`) and negative (`glm_none`) event times.
        2. Removes sessions where the target mouse has 0 valid bouts.
        3. Selects relevant behavioral features per session:
            - Keeps 'self' (target) and 'other' (predictor) egocentric features.
            - Selectively filters dyadic features based on the predictor mouse identity.
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

        Settings are controlled via the `self.modeling_settings_dict` attribute.

        Outputs
        -------
        .pkl file
            A pickle file containing a nested dictionary. Structure:
            `{'generic_feature_name': {'session_id1': {'usv_feature_arr': ..., 'no_usv_feature_arr': ...},
                                      'session_id2': {...}, ...},
             'another_feature': {...}, ...}`
        """

        if self.modeling_settings_dict.get('random_seed') is not None:
            np.random.seed(self.modeling_settings_dict['random_seed'])
            print(f"Random seed set to: {self.modeling_settings_dict['random_seed']}")
        else:
            np.random.seed(None)
            print("Random seed not set.")

        txt_glm_sessions = []
        try:
            sessions_file = self.modeling_settings_dict['session_list_file']
            with open(configure_path(sessions_file)) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        txt_glm_sessions.append(configure_path(line))
            if not txt_glm_sessions:
                raise ValueError("No valid session paths found.")
            print(f"Loaded {len(txt_glm_sessions)} session paths.")
        except FileNotFoundError:
            raise FileNotFoundError(f"Sessions list file not found: {sessions_file}")
        except Exception as e:
            raise RuntimeError(f"Error reading session paths: {e}")

        print("Loading behavioral feature data...")
        beh_feature_data_dict, camera_fr_dict, mouse_track_names_dict = load_behavioral_feature_data(
            behavior_file_paths=txt_glm_sessions,
            csv_sep=self.modeling_settings_dict['data_io']['csv_sheet_delimiter']
        )
        print("Loading USV data and selecting epochs...")

        usv_data_dict = find_bout_epochs(
            root_directories=txt_glm_sessions,
            mouse_ids_dict=mouse_track_names_dict,
            camera_fps_dict=camera_fr_dict,
            features_dict=beh_feature_data_dict,
            csv_sep=self.modeling_settings_dict['data_io']['csv_sheet_delimiter'],
            gmm_component_index=self.modeling_settings_dict['task_definition']['gmm_component_index'],
            gmm_z_score=self.modeling_settings_dict['task_definition']['gmm_z_score'],
            noise_vocal_categories=self.modeling_settings_dict['vocal_features']['noise_vocal_categories'],
            proportion_smoothing_sd=self.modeling_settings_dict['vocal_features']['vocal_proportion_smoothing_sd'],
            vocal_output_type=self.modeling_settings_dict['vocal_features']['vocal_output_type'],
            filter_history=self.modeling_settings_dict['features']['filter_history'],
            prediction_mode=self.modeling_settings_dict['task_definition']['prediction_mode'],
            usv_bout_time=self.modeling_settings_dict['task_definition']['usv_bout_time'],
            min_usv_per_bout=self.modeling_settings_dict['task_definition']['min_usv_per_bout']
        )

        predictor_mouse_idx = self.modeling_settings_dict['features']['predictor_mouse']
        target_mouse_idx = abs(predictor_mouse_idx - 1)

        sessions_to_remove = []
        for session_id in list(beh_feature_data_dict.keys()):
            if session_id not in mouse_track_names_dict:
                sessions_to_remove.append(session_id)
                continue

            targ_name = mouse_track_names_dict[session_id][target_mouse_idx]

            # Check for valid bouts using
            if targ_name in usv_data_dict.get(session_id, {}):
                bouts = usv_data_dict[session_id][targ_name].get('glm_usv', [])
                if len(bouts) == 0:
                    print(f"Skipping session {session_id}: 0 valid bouts found for {targ_name}.")
                    sessions_to_remove.append(session_id)
            else:
                sessions_to_remove.append(session_id)

        for sess in sessions_to_remove:
            if sess in beh_feature_data_dict:
                del beh_feature_data_dict[sess]

        print(f"Proceeding with {len(beh_feature_data_dict)} sessions after filtering empty ones.")

        processed_beh_feature_data_dict = {}

        for behavioral_session in list(beh_feature_data_dict.keys()):
            predictor_mouse_idx = self.modeling_settings_dict['features']['predictor_mouse']
            target_mouse_idx = abs(predictor_mouse_idx - 1)
            predictor_mouse_name = mouse_track_names_dict[behavioral_session][predictor_mouse_idx]  # 'other'
            target_mouse_name = mouse_track_names_dict[behavioral_session][target_mouse_idx]  # 'self'

            columns_to_keep_session = []
            session_df_columns = beh_feature_data_dict[behavioral_session].columns

            for base_feature in self.modeling_settings_dict['features']['behavioral_predictors']:
                session_cols = [col for col in session_df_columns if col.split('.')[-1] == base_feature]
                for feature in session_cols:
                    is_self_ego = feature.startswith(f"{target_mouse_name}.")
                    is_other_ego = feature.startswith(f"{predictor_mouse_name}.")
                    is_dyadic = '-' in feature.split('.')[0]
                    is_diff = 'diff' in base_feature

                    is_excluded = False
                    if is_dyadic and not is_diff:
                        try:
                            feature_parts = feature.split('.')[-1].split('-')
                            if len(feature_parts) == 2:
                                if predictor_mouse_idx == 0:
                                    if feature_parts[0] == 'allo_yaw' or feature_parts[1] == 'TTI': is_excluded = True
                                else:
                                    if feature_parts[1] == 'allo_yaw' or feature_parts[0] == 'TTI': is_excluded = True
                        except (IndexError, AttributeError):
                            pass

                    if is_self_ego or is_other_ego or is_diff or (is_dyadic and not is_excluded):
                        columns_to_keep_session.append(feature)
                        if base_feature not in ('speed', 'acceleration'):
                            der_1st, der_2nd = f'{feature}_1st_der', f'{feature}_2nd_der'
                            if self.modeling_settings_dict['features']['include_1st_der_features_bool'] and der_1st in session_df_columns:
                                columns_to_keep_session.append(der_1st)
                            if self.modeling_settings_dict['features']['include_2nd_der_features_bool'] and der_2nd in session_df_columns:
                                columns_to_keep_session.append(der_2nd)

            voc_out_type = self.modeling_settings_dict['vocal_features']['vocal_output_type']
            partner_only = self.modeling_settings_dict['vocal_features']['vocal_output_partner_only']
            new_voc_cols = []

            if voc_out_type:
                # Loop through both mice to find signals
                for m_name in [target_mouse_name, predictor_mouse_name]:
                    is_target = (m_name == target_mouse_name)

                    # If partner_only is True, skip target mouse vocalizations entirely
                    if partner_only and is_target:
                        continue

                    if m_name in usv_data_dict[behavioral_session]:
                        vocal_signals = usv_data_dict[behavioral_session][m_name].get('continuous_vocal_signals', {})

                        for sig_key, sig_arr in vocal_signals.items():
                            # SCIENTIFIC CONSTRAINT: Never include 'proportion' or 'event' for the self mouse
                            # as it creates trivial autocorrelation that masks behavioral effects.
                            if is_target and any(k in sig_key for k in ['proportion', 'event', 'count']):
                                continue

                            col_name = f"{m_name}.{sig_key}"
                            new_voc_cols.append(pls.Series(col_name, sig_arr))
                            columns_to_keep_session.append(col_name)

            # Filter DataFrame and Add Vocalization Data
            columns_to_keep_session = sorted(list(set(columns_to_keep_session)))
            existing_cols_to_select = [c for c in columns_to_keep_session if c in session_df_columns]
            current_df = beh_feature_data_dict[behavioral_session].select(existing_cols_to_select)

            if new_voc_cols:
                current_df = current_df.with_columns(new_voc_cols)

            processed_beh_feature_data_dict[behavioral_session] = current_df

        del beh_feature_data_dict
        print("Feature filtering complete.")

        final_suffixes = set()
        if not processed_beh_feature_data_dict:
            print("Error: No sessions remaining after filtering.")
            return

        for sess_id, sess_df in processed_beh_feature_data_dict.items():
            for col in sess_df.columns:
                suffix = col.split('.')[-1]
                if suffix.isdigit(): continue
                final_suffixes.add(suffix)

        revised_behavioral_predictors = sorted(list(final_suffixes))
        if not revised_behavioral_predictors: raise ValueError("No features selected.")
        print(f"Final feature suffixes selected: {revised_behavioral_predictors}")

        print("Standardizing columns across sessions...")
        for sess_id in processed_beh_feature_data_dict:
            df = processed_beh_feature_data_dict[sess_id]
            existing_cols = set(df.columns)
            new_zeros = []

            t_name = mouse_track_names_dict[sess_id][target_mouse_idx]
            p_name = mouse_track_names_dict[sess_id][predictor_mouse_idx]

            for pred_suffix in revised_behavioral_predictors:
                for prefix, m_name in [('self', t_name), ('other', p_name)]:
                    expected_col = f"{m_name}.{pred_suffix}"

                    if expected_col not in existing_cols:
                        is_vocal = 'usv_' in pred_suffix
                        if is_vocal:
                            if m_name == t_name and (partner_only or any(k in pred_suffix for k in ['proportion', 'event', 'count'])):
                                continue

                        new_zeros.append(pls.Series(expected_col, np.zeros(df.height, dtype=np.float32)))

            if new_zeros:
                processed_beh_feature_data_dict[sess_id] = df.with_columns(new_zeros)

        print("Z-scoring features across sessions...")
        processed_beh_feature_data_dict = zscore_different_sessions_together(
            data_dict=processed_beh_feature_data_dict,
            feature_lst=revised_behavioral_predictors,
            feature_bounds=getattr(self, 'feature_boundaries', {})
        )
        print("Z-scoring complete.")

        print("Extracting epochs per session and renaming features...")
        glm_final_data_dict = {}

        for session_idx, beh_session_id in enumerate(tqdm(processed_beh_feature_data_dict.keys(), desc='Extracting Epochs')):
            session_df = processed_beh_feature_data_dict[beh_session_id]
            predictor_mouse_name = mouse_track_names_dict[beh_session_id][predictor_mouse_idx]
            target_mouse_name = mouse_track_names_dict[beh_session_id][target_mouse_idx]

            try:
                usv_event_times = usv_data_dict[beh_session_id][target_mouse_name]['glm_usv']
                no_usv_event_times = usv_data_dict[beh_session_id][target_mouse_name]['glm_none']
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

                if generic_key not in glm_final_data_dict: glm_final_data_dict[generic_key] = {}
                if beh_session_id not in glm_final_data_dict[generic_key]:
                    glm_final_data_dict[generic_key][beh_session_id] = {}

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

                glm_final_data_dict[generic_key][beh_session_id]['usv_feature_arr'] = usv_feature_arr
                glm_final_data_dict[generic_key][beh_session_id]['no_usv_feature_arr'] = no_usv_feature_arr
        print("Epoch extraction and renaming complete.")

        if not glm_final_data_dict:
            print("Warning: Final data dictionary is empty.")
            return

        final_covariate_names = sorted(list(glm_final_data_dict.keys()))
        total_covariates = len(final_covariate_names)
        first_feat = final_covariate_names[0]
        all_sessions = sorted(list(glm_final_data_dict[first_feat].keys()))

        alignment_passed = True
        mismatched_sessions = []

        for sess_id in all_sessions:
            # Reference counts for this specific session
            ref_usv_n = glm_final_data_dict[first_feat][sess_id]['usv_feature_arr'].shape[0]
            ref_none_n = glm_final_data_dict[first_feat][sess_id]['no_usv_feature_arr'].shape[0]

            for feat in final_covariate_names[1:]:
                feat_usv_n = glm_final_data_dict[feat][sess_id]['usv_feature_arr'].shape[0]
                feat_none_n = glm_final_data_dict[feat][sess_id]['no_usv_feature_arr'].shape[0]

                if feat_usv_n != ref_usv_n or feat_none_n != ref_none_n:
                    alignment_passed = False
                    mismatched_sessions.append(sess_id)
                    break

        grand_total_sessions = len(all_sessions)
        grand_total_usv = sum(glm_final_data_dict[first_feat][s]['usv_feature_arr'].shape[0] for s in all_sessions)
        grand_total_none = sum(glm_final_data_dict[first_feat][s]['no_usv_feature_arr'].shape[0] for s in all_sessions)

        print("\n" + "=" * 105)
        print(f"GLM MODELING INPUT: GLOBAL AGGREGATE SUMMARY")
        print("=" * 105)
        print(f"{'#':<4} {'Feature Name':<40} | {'Sess':<6} | {'USV Bouts':<12} | {'No-USV Bouts':<12} | {'Total N'}")
        print("-" * 105)

        for i, feat in enumerate(final_covariate_names, 1):
            feat_usv_bouts = sum(glm_final_data_dict[feat][s]['usv_feature_arr'].shape[0] for s in glm_final_data_dict[feat])
            feat_none_bouts = sum(glm_final_data_dict[feat][s]['no_usv_feature_arr'].shape[0] for s in glm_final_data_dict[feat])

            print(f"{i:3}. {feat:<40} | {len(glm_final_data_dict[feat]):<6} | {feat_usv_bouts:<12} | {feat_none_bouts:<12} | {feat_usv_bouts + feat_none_bouts:<8}")

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
        max_hist_sec = self.modeling_settings_dict['features']['filter_history']
        file_name_ = f"modeling_{self.modeling_settings_dict['task_definition']['prediction_mode']}_gmm{self.modeling_settings_dict['task_definition']['gmm_component_index']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(txt_glm_sessions)}sess_hist{max_hist_sec}s.pkl"
        save_path = os.path.join(self.modeling_settings_dict['save_dir'], file_name_)

        try:
            os.makedirs(self.modeling_settings_dict['save_dir'], exist_ok=True)
            with open(save_path, 'wb') as f:
                pickle.dump(glm_final_data_dict, f)
            print(f"\n[+] Successfully saved PER-SESSION renamed GLM input data to:\n    {save_path}")
        except Exception as e:
            print(f"\n[!] Error saving final pickle file: {e}")

    def _pool_data_from_sessions(self,
                                 feature_data: dict,
                                 session_list: list) -> tuple[np.ndarray, np.ndarray]:
        """
        Pools all USV and no-USV data from a specific list of sessions.

        This helper method iterates through a given list of session IDs,
        accesses the corresponding data in the feature dictionary, and
        concatenates all 'usv_feature_arr' and 'no_usv_feature_arr' arrays.

        Parameters
        ----------
        feature_data : dict
            The data dictionary for a *single feature* (e.g.,
            `glm_feature_arr_dict['speed']`), where keys are session IDs.
        session_list : list
            A list of session_id strings (e.g., ['20230119_172410', ...])
            to pool data from.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            A tuple containing two arrays:
            1. X_pos: A single NumPy array of all 'usv_feature_arr' data
               concatenated (n_events, n_frames).
            2. X_neg: A single NumPy array of all 'no_usv_feature_arr' data
               concatenated (n_events, n_frames).
        """

        usv_data_list = []
        no_usv_data_list = []

        for session_id in session_list:
            if session_id in feature_data:
                usv_data_list.append(feature_data[session_id]['usv_feature_arr'])
                no_usv_data_list.append(feature_data[session_id]['no_usv_feature_arr'])

        X_pos = np.concatenate(usv_data_list, axis=0) if usv_data_list else np.empty((0, self.modeling_settings_dict['features']['filter_history']))
        X_neg = np.concatenate(no_usv_data_list, axis=0) if no_usv_data_list else np.empty((0, self.modeling_settings_dict['features']['filter_history']))

        return X_pos, X_neg

    def _balance_data(self,
                      X_pos: np.ndarray,
                      X_neg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Balances positive and negative samples by down-sampling the majority class.

        This function takes the two arrays of positive (Y=1) and negative (Y=0)
        samples, finds the minimum count, and randomly subsamples the
        larger array to match the smaller one.

        Parameters
        ----------
        X_pos : np.ndarray
            Array of positive samples (n_samples, n_features).
        X_neg : np.ndarray
            Array of negative samples (n_samples, n_features).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            A tuple containing the balanced arrays:
            1. X_pos_balanced: The (potentially) down-sampled positive data.
            2. X_neg_balanced: The (potentially) down-sampled negative data.
            Both arrays will have the same n_samples.
        """

        n_pos = X_pos.shape[0]
        n_neg = X_neg.shape[0]

        if n_pos == 0 or n_neg == 0:
            # return empty arrays if one class has no data
            return np.empty((0, X_pos.shape[1])), np.empty((0, X_neg.shape[1]))

        n_samples = min(n_pos, n_neg)

        if n_pos > n_samples:
            pos_indices = np.random.choice(n_pos, n_samples, replace=False)
            X_pos = X_pos[pos_indices]

        if n_neg > n_samples:
            neg_indices = np.random.choice(n_neg, n_samples, replace=False)
            X_neg = X_neg[neg_indices]

        return X_pos, X_neg

    def create_data_splits(self, feature_data: dict, strategy_override: str = None):
        """
        A generator that yields train/test splits based on the 'split_strategy'.

        This function reads 'self.modeling_settings_dict' for 'split_strategy',
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
            `glm_feature_arr_dict['self.speed']`), where keys are session IDs and
            values contain 'usv_feature_arr' and 'no_usv_feature_arr'.
        strategy_override : str, optional
            If provided, this strategy will be used instead of the one in settings.

        Yields
        ------
        tuple
            A tuple of (X_train, y_train, X_test, y_test) for each split.
        """

        split_strategy = self.modeling_settings_dict['model_selection']['split_strategy']
        if strategy_override:
            split_strategy = strategy_override

        n_splits = self.modeling_settings_dict['model_selection']['num_splits']
        test_proportion = self.modeling_settings_dict['model_selection']['test_proportion']
        random_state = self.modeling_settings_dict['random_seed']

        all_sessions = list(feature_data.keys())
        X_pos_all, X_neg_all = self._pool_data_from_sessions(feature_data, all_sessions)
        n_pos_total = X_pos_all.shape[0]
        n_neg_total = X_neg_all.shape[0]

        ### Strategy 1: 'mixed' (all sessions together)
        if split_strategy == 'mixed':
            X_pos, X_neg = self._balance_data(X_pos_all, X_neg_all)

            if X_pos.shape[0] == 0:
                print(f"Warning: No balanced data for feature. Skipping splits.")
                return

            y_pos = np.ones(X_pos.shape[0])
            y_neg = np.zeros(X_neg.shape[0])
            X = np.concatenate((X_pos, X_neg), axis=0)
            y = np.concatenate((y_pos, y_neg), axis=0)

            print(f"--- 'mixed' strategy: Created balanced dataset of {X.shape[0]} samples.")

            sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_proportion, random_state=random_state)

            for train_idx, test_idx in sss.split(X, y):
                yield X[train_idx], y[train_idx], X[test_idx], y[test_idx]

        ### Strategy 2: 'session' (some sessions go to train, some to test)
        elif split_strategy == 'session':
            all_sessions_array = np.array(all_sessions)
            n_sessions = len(all_sessions_array)

            min_test_sessions = 1
            actual_test_proportion = max(test_proportion, min_test_sessions / n_sessions if n_sessions > 0 else 0)
            if n_sessions * (1 - actual_test_proportion) < 1:
                print(f"Warning: test_proportion ({test_proportion}) too high for {n_sessions} sessions. Skipping.")
                return

            ss = ShuffleSplit(n_splits=n_splits, test_size=actual_test_proportion, random_state=random_state)

            split_num = 0
            for train_session_idx, test_session_idx in ss.split(all_sessions_array):
                split_num += 1
                train_session_list = all_sessions_array[train_session_idx]
                test_session_list = all_sessions_array[test_session_idx]

                X_pos_train, X_neg_train = self._pool_data_from_sessions(feature_data, train_session_list)
                X_pos_test, X_neg_test = self._pool_data_from_sessions(feature_data, test_session_list)

                # Balance the *training set ONLY
                X_pos_train_bal, X_neg_train_bal = self._balance_data(X_pos_train, X_neg_train)

                if X_pos_train_bal.shape[0] == 0:
                    print(f"Warning: No balanced training data for split {split_num}. Skipping.")
                    continue

                # Create final train arrays (balanced)
                y_pos_train = np.ones(X_pos_train_bal.shape[0])
                y_neg_train = np.zeros(X_neg_train_bal.shape[0])
                X_train = np.concatenate((X_pos_train_bal, X_neg_train_bal), axis=0)
                y_train = np.concatenate((y_pos_train, y_neg_train), axis=0)

                # Create final test arrays (NB: unbalanced!)
                y_pos_test = np.ones(X_pos_test.shape[0])
                y_neg_test = np.zeros(X_neg_test.shape[0])
                X_test = np.concatenate((X_pos_test, X_neg_test), axis=0)
                y_test = np.concatenate((y_pos_test, y_neg_test), axis=0)

                train_shuffle_idx = np.random.permutation(X_train.shape[0])
                test_shuffle_idx = np.random.permutation(X_test.shape[0])

                yield X_train[train_shuffle_idx], y_train[train_shuffle_idx], X_test[test_shuffle_idx], y_test[test_shuffle_idx]

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

            y_fake_pos = np.ones(X_fake_pos.shape[0])
            y_fake_neg = np.zeros(X_fake_neg.shape[0])
            X = np.concatenate((X_fake_pos, X_fake_neg), axis=0)
            y = np.concatenate((y_fake_pos, y_fake_neg), axis=0)

            print(f"  Created null dataset of {X.shape[0]} samples ({X_fake_pos.shape[0]} fake_pos, {X_fake_neg.shape[0]} fake_neg) ---")

            sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_proportion, random_state=random_state)

            for train_idx, test_idx in sss.split(X, y):
                yield X[train_idx], y[train_idx], X[test_idx], y[test_idx]

        ### Strategy 4: 'session_null_control' (session-split no-bout, matching 'session' size)
        elif split_strategy == 'session_null_control':
            print("--- Using 'session_null_control' (session-split) strategy ---")
            all_sessions_array = np.array(all_sessions)
            n_sessions = len(all_sessions_array)

            min_test_sessions = 1
            actual_test_proportion = max(test_proportion, min_test_sessions / n_sessions if n_sessions > 0 else 0)
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
                X_pos_train_actual, X_neg_train_actual = self._pool_data_from_sessions(feature_data, train_session_list)
                X_pos_test_actual, X_neg_test_actual = self._pool_data_from_sessions(feature_data, test_session_list)

                # Find the target sizes from the actual data
                X_pos_train_bal, X_neg_train_bal = self._balance_data(X_pos_train_actual, X_neg_train_actual)

                n_balanced_train_half = X_pos_train_bal.shape[0]
                n_total_train_needed = n_balanced_train_half * 2

                n_test_pos_target = X_pos_test_actual.shape[0]
                n_test_neg_target = X_neg_test_actual.shape[0]

                if n_balanced_train_half == 0:
                    print(f"Warning: No *actual* balanced training data for split {split_num}. Cannot match size. Skipping.")
                    continue

                # Pool ONLY No-Bout data from selected sessions
                _, X_neg_train_all = self._pool_data_from_sessions(feature_data, train_session_list)
                _, X_neg_test_all = self._pool_data_from_sessions(feature_data, test_session_list)

                # Create fake balanced training set, matching 'session' size
                n_train_neg_available = X_neg_train_all.shape[0]

                if n_train_neg_available < n_total_train_needed:
                    print(f"Warning: Not enough No-Bout samples ({n_train_neg_available}) in train sessions to create null control of size {n_total_train_needed}. Sampling with replacement.")
                    train_neg_indices = np.random.choice(n_train_neg_available, size=n_total_train_needed, replace=True)
                else:
                    train_neg_indices = np.random.permutation(n_train_neg_available)

                X_fake_pos_train = X_neg_train_all[train_neg_indices[:n_balanced_train_half]]
                X_fake_neg_train = X_neg_train_all[train_neg_indices[n_balanced_train_half: n_total_train_needed]]

                y_pos_train = np.ones(X_fake_pos_train.shape[0])
                y_neg_train = np.zeros(X_fake_neg_train.shape[0])
                X_train = np.concatenate((X_fake_pos_train, X_fake_neg_train), axis=0)
                y_train = np.concatenate((y_pos_train, y_neg_train), axis=0)

                # Create fake UNBALANCED test set, matching 'session' size AND ratio
                n_test_neg_available = X_neg_test_all.shape[0]

                if n_test_neg_available == 0:
                    print(f"Warning: No No-Bout test samples available for split {split_num}. Skipping.")
                    continue

                fake_pos_indices = np.random.choice(n_test_neg_available, size=n_test_pos_target, replace=True)
                fake_neg_indices = np.random.choice(n_test_neg_available, size=n_test_neg_target, replace=True)

                X_fake_pos_test = X_neg_test_all[fake_pos_indices]
                X_fake_neg_test = X_neg_test_all[fake_neg_indices]

                y_pos_test = np.ones(X_fake_pos_test.shape[0])
                y_neg_test = np.zeros(X_fake_neg_test.shape[0])
                X_test = np.concatenate((X_fake_pos_test, X_fake_neg_test), axis=0)
                y_test = np.concatenate((y_pos_test, y_neg_test), axis=0)

                # Shuffle the final arrays before yielding
                train_shuffle_idx = np.random.permutation(X_train.shape[0])
                test_shuffle_idx = np.random.permutation(X_test.shape[0])

                yield X_train[train_shuffle_idx], y_train[train_shuffle_idx], X_test[test_shuffle_idx], y_test[test_shuffle_idx]

        else:
            raise ValueError(f"Unknown split_strategy: {split_strategy}. Must be 'mixed', 'session', 'null_control', or 'session_null_control'.")

    def _run_glm_for_feature_sklearn(self,
                                     feature_name: str,
                                     feature_data: dict,
                                     basis_matrix: np.ndarray
                                     ) -> tuple[str, dict]:
        """
        Executes a univariate Logistic Regression analysis for a single behavioral or vocal feature.

        This method serves as the core computational engine for the GLM pipeline. It processes
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
        n_splits = self.modeling_settings_dict['model_selection']['num_splits']
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
                    penalty=self.modeling_settings_dict['hyperparameters']['logistic_regression_params']['penalty'],
                    Cs=self.modeling_settings_dict['hyperparameters']['logistic_regression_params']['cs'],
                    cv=self.modeling_settings_dict['hyperparameters']['logistic_regression_params']['cv'],
                    class_weight='balanced',
                    solver=self.modeling_settings_dict['hyperparameters']['logistic_regression_params']['solver'],
                    max_iter=self.modeling_settings_dict['hyperparameters']['logistic_regression_params']['max_iter'],
                    random_state=self.modeling_settings_dict['random_seed']
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
                results['actual']['f1'][split_idx] = f1_score(y_test, y_pred_actual, average='micro', zero_division=0.0)

                if len(np.unique(y_test)) > 1:
                    results['actual']['auc'][split_idx] = roc_auc_score(y_test, y_proba_actual)
                    epsilon = 1e-15
                    y_proba_actual_clipped = np.clip(y_proba_actual, epsilon, 1 - epsilon)
                    results['actual']['ll'][split_idx] = log_loss(y_test, y_proba_actual_clipped)

            except Exception as e:
                print(f"  ERROR during ACTUAL GLM fit/predict for {feature_name}, split {split_idx}: {e}")

            try:
                shuffle_seed = self.modeling_settings_dict.get('random_seed')
                if shuffle_seed is not None: np.random.seed(shuffle_seed + split_idx + 1)  # Offset seed
                y_train_shuffled = np.random.permutation(y_train)

                lr_shuffled = LogisticRegressionCV(
                    penalty=self.modeling_settings_dict['logistic_regression_params']['penalty'],
                    Cs=self.modeling_settings_dict['logistic_regression_params']['cs'],
                    cv=self.modeling_settings_dict['logistic_regression_params']['cv'],
                    class_weight='balanced',
                    solver=self.modeling_settings_dict['logistic_regression_params']['solver'],
                    max_iter=self.modeling_settings_dict['logistic_regression_params']['max_iter'],
                    random_state=self.modeling_settings_dict.get('random_seed')
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
                results['shuffled']['f1'][split_idx] = f1_score(y_test, y_pred_shuffled, average='micro', zero_division=0.0)

                if len(np.unique(y_test)) > 1:
                    results['shuffled']['auc'][split_idx] = roc_auc_score(y_test, y_proba_shuffled)
                    epsilon = 1e-15
                    y_proba_shuffled_clipped = np.clip(y_proba_shuffled, epsilon, 1 - epsilon)
                    results['shuffled']['ll'][split_idx] = log_loss(y_test, y_proba_shuffled_clipped)

            except Exception as e:
                print(f"  ERROR during SHUFFLED GLM fit/predict for {feature_name}, split {split_idx}: {e}")

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

    def _run_glm_for_feature_pygam(self,
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
            for API compatibility with the `run_glm_analysis_parallel` caller.

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

        n_splits = self.modeling_settings_dict['model_selection']['num_splits']
        history_frames = int(np.floor(self.modeling_settings_dict['data_io']['camera_sampling_rate'] * self.modeling_settings_dict['features']['filter_history']))

        try:
            pygam_params = self.modeling_settings_dict['hyperparameters']['pygam_params']
            n_splines_time = pygam_params.get('n_splines_time', 8)
            n_splines_value = pygam_params.get('n_splines_value', 5)
            lam_penalty = pygam_params.get('lam_penalty', None)
            max_iterations = pygam_params.get('max_iterations', 100)
            tol_val = pygam_params.get('tol_val', 1e-4)
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
            n_samples, n_frames = X.shape
            X_unrolled = np.zeros((n_samples * n_frames, 2), dtype=np.float32)
            X_unrolled[:, 0] = X.ravel()
            X_unrolled[:, 1] = np.tile(time_indices, n_samples)
            return X_unrolled

        actual_data_splitter = self.create_data_splits(feature_data, strategy_override=None)

        current_strategy = self.modeling_settings_dict['model_selection']['split_strategy']
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
                    log_odds_0 = gam_actual.predict_mu(grid_X_0).astype(np.float32)
                    log_odds_1 = gam_actual.predict_mu(grid_X_1).astype(np.float32)

                    results['actual']['filter_shapes'][split_idx, :] = (log_odds_1 - log_odds_0).flatten()
                    results['actual']['score'][split_idx] = f1_score(y_test_int, y_pred_mean_epoch, average='micro', zero_division=0.0)
                    results['actual']['precision'][split_idx] = precision_score(y_test_int, y_pred_mean_epoch, zero_division=0.0)
                    results['actual']['recall'][split_idx] = recall_score(y_test_int, y_pred_mean_epoch, zero_division=0.0)
                    results['actual']['f1'][split_idx] = f1_score(y_test_int, y_pred_mean_epoch, average='micro', zero_division=0.0)
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
                    log_odds_0_null = gam_shuffled.predict_mu(grid_X_0_null).astype(np.float32)
                    log_odds_1_null = gam_shuffled.predict_mu(grid_X_1_null).astype(np.float32)
                    filter_shape_null = (log_odds_1_null - log_odds_0_null).flatten()
                    results['shuffled']['filter_shapes'][split_idx, :] = filter_shape_null

                    results['shuffled']['score'][split_idx] = f1_score(y_test_int_null, y_pred_shuffled_mean, average='micro', zero_division=0.0)
                    results['shuffled']['precision'][split_idx] = precision_score(y_test_int_null, y_pred_shuffled_mean, zero_division=0.0)
                    results['shuffled']['recall'][split_idx] = recall_score(y_test_int_null, y_pred_shuffled_mean, zero_division=0.0)
                    results['shuffled']['f1'][split_idx] = f1_score(y_test_int_null, y_pred_shuffled_mean, average='micro', zero_division=0.0)
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
