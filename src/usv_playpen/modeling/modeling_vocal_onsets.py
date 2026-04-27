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
from pathlib import Path
import polars as pls
from pygam import LogisticGAM, te
import pickle
import time
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import balanced_accuracy_score, log_loss, f1_score, recall_score, roc_auc_score, brier_score_loss
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
from tqdm import tqdm

from .load_input_files import load_behavioral_feature_data, find_bout_epochs
from .modeling_metadata import (
    build_input_metadata, derive_experimental_condition,
    derive_feature_zoo_full, derive_camera_fps_field, inject_metadata,
)
from .load_input_files import _calculate_ibi_threshold
from .modeling_utils import (
    prepare_modeling_sessions,
    resolve_mouse_roles,
    select_kinematic_columns,
    build_vocal_signal_columns,
    identify_empty_event_sessions,
    harmonize_session_columns,
    zscore_features_across_sessions,
    pool_session_arrays,
    balance_two_class_arrays,
    unroll_history_matrix,
    concat_two_class_with_labels,
    shuffle_train_test_arrays,
    bounded_test_proportion,
    expected_calibration_error,
    safe_matthews_corrcoef,
    safe_confusion_matrix,
    run_predictor_audits,
)
from ..analyses.compute_behavioral_features import FeatureZoo


class VocalOnsetModelingPipeline(FeatureZoo):
    """
    End-to-end pipeline for modeling the onset of USV bouts from behavioural kinematics.

    The pipeline has three responsibilities:

    1. **Data preparation**: ingests raw behavioural time-series (polars DataFrames
       produced by the upstream extraction), detects vocal bouts via GMM-driven
       inter-bout-interval thresholds, assembles per-session history windows
       around USV and No-USV event timestamps, applies cross-session z-scoring,
       and serializes the resulting `{feature: {session: {usv_feature_arr,
       no_usv_feature_arr}}}` dictionary to disk.
    2. **Cross-validation splitting**: the `create_data_splits` generator
       implements four strategies ('mixed', 'session', 'null_control',
       'session_null_control') with the canonical "balanced train / natural-rate
       test" invariant — the training fold is always down-sampled to 50/50 so
       gradient updates see both classes, while the test fold preserves the
       natural base rate so reported metrics reflect real imbalance.
    3. **Univariate model fitting**: `_run_model_for_feature_sklearn` fits a
       basis-projected `LogisticRegressionCV` per feature; `_run_model_for_feature_pygam`
       fits a tensor-product-spline `LogisticGAM`. Both return a
       `{'actual', 'null'}` results dict with calibration (Brier, ECE),
       chance-corrected (MCC), and optimizer diagnostic fields per split.

    Subclasses (e.g. `BoutParameterPipeline`) reuse the extraction / splitting
    infrastructure and override the per-feature modelling methods to handle
    continuous regression targets instead of classification.
    """

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
            settings_path = Path(__file__).resolve().parent.parent / '_parameter_settings/modeling_settings.json'
            try:
                with open(settings_path, 'r') as settings_json_file:
                    self.modeling_settings = json.load(settings_json_file)

            except FileNotFoundError:
                raise FileNotFoundError(f"Settings file not found at {settings_path}")
        else:
            self.modeling_settings = modeling_settings_dict

        # Strict membership-then-index access (matches the other pipelines)
        # rather than `.get()`. `feature_boundaries` is an optional top-level
        # JSON key; when absent the downstream `getattr` fallback kicks in.
        if 'feature_boundaries' in self.modeling_settings:
            self.feature_boundaries = self.modeling_settings['feature_boundaries']

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

    def extract_and_save_modeling_input_data(self) -> None:
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
              `dyadic_pose_symmetric` is False, a directional
              allo_yaw/allo_pitch/TTI rule drops one symmetric half based on
              the predictor-mouse convention.
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
            min_usv_per_bout=self.modeling_settings['model_params']['usv_per_bout_floor'],
            category_column=self.modeling_settings['vocal_features']['usv_category_column_name'],
            noise_column=self.modeling_settings['vocal_features']['usv_noise_column'],
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

        print("Standardizing columns across sessions...")
        # `harmonize_session_columns` performs the dyad-rename
        # (`{m1-m2}.{suffix}` -> `{suffix}`) in addition to zero-filling
        # missing ego/dyadic/USV columns with a project-wide existence
        # gate. The dyad-rename is required *before* the audit so that
        # dyadic columns share a stable cross-session key — otherwise
        # every dyadic feature looks unique-to-one-session and gets
        # dropped by the audit's "feature must contribute a block in
        # every contributing session" filter.
        processed_beh_feature_data_dict, revised_behavioral_predictors = harmonize_session_columns(
            processed_beh_dict=processed_beh_feature_data_dict,
            mouse_names_dict=mouse_track_names_dict,
            target_idx=target_mouse_idx,
            predictor_idx=predictor_mouse_idx,
        )
        if not revised_behavioral_predictors:
            raise ValueError("No features selected.")
        print(f"Final feature suffixes selected: {revised_behavioral_predictors}")

        print("Z-scoring features across sessions...")
        processed_beh_feature_data_dict = zscore_features_across_sessions(
            processed_beh_dict=processed_beh_feature_data_dict,
            suffixes=revised_behavioral_predictors,
            feature_bounds=getattr(self, 'feature_boundaries', {}),
            abs_features=['allo_roll', 'allo_yaw-nose', 'nose-allo_yaw', 'allo_yaw-TTI', 'TTI-allo_yaw']
        )
        print("Z-scoring complete.")

        max_hist_sec = self.modeling_settings['model_params']['filter_history']
        target_vocal_type = self.modeling_settings['model_params']['model_target_vocal_type']
        gmm_idx = self.modeling_settings['model_params']['gmm_component_index']

        # Build the unified Level-1 filename — the analysis_tag identifies
        # the pipeline, the experimental_condition identifies the cohort,
        # and the rest of the historical fields (n_sessions, hist, gmm
        # idx, target_vocal_type) live in `_input_metadata` rather than
        # in the filename.
        cohort_condition = derive_experimental_condition(self.modeling_settings)
        analysis_tag = f"onsets-{target_vocal_type}"
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_name_ = f"modeling_{analysis_tag}_{cohort_condition}_{ts}.pkl"

        # Build the `_input_metadata` block once. The collinearity /
        # timescale audits embed it verbatim into their artifacts, the
        # input pickle embeds it under the reserved key `_input_metadata`,
        # and the dispatcher copies it through into every per-feature
        # univariate pickle so each downstream artifact is independently
        # provenance-complete.
        ibi_thresholds_md = {}
        gmm_params = self.modeling_settings['gmm_params']
        for sex in ('male', 'female'):
            params = gmm_params[sex]
            if gmm_idx < len(params['means']):
                ibi_thresholds_md[sex] = float(_calculate_ibi_threshold(
                    params['means'][gmm_idx], params['sds'][gmm_idx],
                    self.modeling_settings['model_params']['gmm_z_score'],
                ))
            else:
                ibi_thresholds_md[sex] = float('nan')

        # Pool the kept generic feature suffixes that survived
        # `harmonize_session_columns` from one representative session;
        # `revised_behavioral_predictors` only knows the bare suffix list
        # (per-mouse expansion happens column-by-column at pool time).
        first_sess_id = next(iter(processed_beh_feature_data_dict))
        kept_columns_first_sess = list(processed_beh_feature_data_dict[first_sess_id].columns)
        feature_zoo_kept_md = sorted({
            c.split('.', 1)[-1] if '.' in c and not c.split('.')[-1].isdigit() else c
            for c in kept_columns_first_sess
        })

        input_metadata = build_input_metadata(
            modeling_settings=self.modeling_settings,
            analysis_type='onset',
            analysis_tag=analysis_tag,
            pipeline_class=type(self).__name__,
            target_idx=target_mouse_idx,
            predictor_idx=predictor_mouse_idx,
            n_sessions_used=len(processed_beh_feature_data_dict),
            session_ids=sorted(processed_beh_feature_data_dict.keys()),
            n_events_per_session={},  # populated below after epoch slicing
            feature_zoo_full=derive_feature_zoo_full(self.modeling_settings),
            feature_zoo_kept=feature_zoo_kept_md,
            dyadic_engagement_features_used=list(kin_settings['dyadic_engagement']),
            dyadic_pose_symmetric_features_used=kin_settings['dyadic_pose_symmetric'],
            noise_vocal_categories_excluded=list(voc_settings['usv_noise_categories']),
            vocal_signal_columns_added=sorted({
                c for c in kept_columns_first_sess
                if c.split('.', 1)[-1] in (voc_settings['usv_predictor_type'] or '')
                or any(tok in c for tok in ('usv_rate', 'usv_cat_'))
            }),
            filter_history_seconds=float(max_hist_sec),
            filter_history_frames=int(self.history_frames),
            camera_sampling_rate_hz=derive_camera_fps_field(camera_fr_dict),
            ibi_thresholds=ibi_thresholds_md,
            analysis_specific={
                'model_target_vocal_type': target_vocal_type,
                'usv_bout_time': self.modeling_settings['model_params']['usv_bout_time'],
                'usv_per_bout_floor': self.modeling_settings['model_params']['usv_per_bout_floor'],
            },
        )

        # Predictor diagnostics audit (collinearity + timescales). Runs on
        # the harmonized, z-scored, but not-yet-sliced feature dict so the
        # ACF traces and the per-event summary statistics are computed on
        # the same processed signal the model will see. Diagnostic-only:
        # any failure inside the wrapper warns and continues.
        run_predictor_audits(
            processed_beh_dict=processed_beh_feature_data_dict,
            usv_data_dict=usv_data_dict,
            mouse_names_dict=mouse_track_names_dict,
            camera_fps_dict=camera_fr_dict,
            target_idx=target_mouse_idx,
            predictor_idx=predictor_mouse_idx,
            history_frames=self.history_frames,
            event_keys=['positive_events', 'negative_events'],
            settings=self.modeling_settings,
            save_dir=self.modeling_settings['io']['save_directory'],
            pickle_basename=file_name_,
            input_metadata=input_metadata,
        )

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

                # Extract epochs. Events whose full history window does not fit
                # inside the session recording (s < 0 or e > max_frame_idx) are
                # dropped at this stage rather than carried forward as NaN rows:
                # leaving them in would silently produce NaN-valued basis
                # projections downstream (np.dot propagates NaN), which the
                # splitting / balancing helpers do not filter out.
                column_data_np = session_df[full_column_name].to_numpy()
                max_frame_idx = len(column_data_np)

                def _collect_valid_windows(times):
                    ends = np.round(times * session_sampling_rate).astype(int)
                    starts = ends - history_frames
                    valid_mask = (starts >= 0) & (ends <= max_frame_idx)
                    if not np.any(valid_mask):
                        return np.empty((0, history_frames), dtype=column_data_np.dtype)
                    valid_starts = starts[valid_mask]
                    valid_ends = ends[valid_mask]
                    out = np.empty((valid_starts.size, history_frames), dtype=column_data_np.dtype)
                    for row_idx, (s, e) in enumerate(zip(valid_starts, valid_ends)):
                        chunk = column_data_np[s:e].copy()
                        chunk[np.isnan(chunk)] = 0.0
                        out[row_idx, :] = chunk
                    return out

                usv_feature_arr = _collect_valid_windows(usv_event_times)
                no_usv_feature_arr = _collect_valid_windows(no_usv_event_times)

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

        save_dir = Path(self.modeling_settings['io']['save_directory'])
        save_path = save_dir / file_name_

        # Backfill the per-session event counts that the audit could not
        # see (it ran on the un-sliced feature dict). For each session we
        # report `{'usv': n_usv, 'no_usv': n_no_usv}` for the anchor
        # feature, which is invariant across features by the alignment
        # check above.
        n_events_per_session = {}
        if final_covariate_names:
            anchor_feat = final_covariate_names[0]
            for sess_id in modeling_final_data_dict[anchor_feat]:
                n_events_per_session[sess_id] = {
                    'usv': int(modeling_final_data_dict[anchor_feat][sess_id]['usv_feature_arr'].shape[0]),
                    'no_usv': int(modeling_final_data_dict[anchor_feat][sess_id]['no_usv_feature_arr'].shape[0]),
                }
        input_metadata['n_events_per_session'] = n_events_per_session
        input_metadata['feature_zoo_kept'] = sorted(modeling_final_data_dict.keys())

        # Wrap the final data dict in the metadata-injecting helper so
        # the on-disk artifact carries `_input_metadata` alongside the
        # feature data without colliding with any feature key.
        artifact = inject_metadata(modeling_final_data_dict, _input_metadata=input_metadata)

        try:
            save_dir.mkdir(parents=True, exist_ok=True)
            with save_path.open('wb') as f:
                pickle.dump(artifact, f)
            print(f"\n[+] Successfully saved PER-SESSION renamed modeling input data to:\n    {save_path}")
        except Exception as e:
            print(f"\n[!] Error saving final pickle file: {e}")

    def create_data_splits(self, feature_data: dict, strategy_override: str = None):
        """
        A generator that yields train/test splits based on the 'split_strategy'.

        This function reads 'self.modeling_settings' for 'split_strategy',
        'num_splits', and 'test_proportion'. Across all strategies, the training
        fold is down-sampled to a 50/50 class balance (Bout vs. No-Bout), while
        the test fold is kept at the natural class prior. This keeps gradient
        updates informative for both classes during training, while evaluation
        reflects the true imbalance of the task. Metrics should therefore be
        imbalance-robust (e.g., balanced_accuracy, AUC, log-loss).

        Strategies:
        - 'mixed': Pools all sessions into a single sample-level dataset. Uses
          `StratifiedShuffleSplit` on the *unbalanced* pool so each split
          preserves the natural class ratio in both halves. The training fold
          is then down-sampled to 50/50 via `balance_two_class_arrays`; the
          test fold is left at the natural base rate.
        - 'session': Splits the *list* of sessions into train/test sets
          `n_splits` times using `ShuffleSplit`. The training data (pooled from
          training sessions) is balanced per split to 50/50; the test data
          (pooled from test sessions) retains the natural class prior.
        - 'null_control': Pools all *No-Bout* epochs and runs `StratifiedShuffleSplit`
          on fake labels matched to the per-split train/test sizes and class
          ratios produced by 'mixed'. The training fold is balanced 50/50; the
          test fold mirrors 'mixed's natural-rate test (same N and same ratio).
        - 'session_null_control': (Like 'session') Splits sessions into
          train/test. Builds a fake balanced training set from the *training
          sessions'* No-Bout data matching the exact size of the *actual*
          balanced training set, and a fake unbalanced test set from the *test
          sessions'* No-Bout data matching the exact size and class ratio of
          the *actual* test set.

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
            X_train/y_train are class-balanced; X_test/y_test preserve the
            natural class prior of the source data.
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
            if n_pos_total == 0 or n_neg_total == 0:
                print(f"Warning: No data for one of the classes (pos={n_pos_total}, neg={n_neg_total}). Skipping splits.")
                return

            # Pool all epochs at the natural class prior, then stratified-split.
            X, y = concat_two_class_with_labels(X_pos_all, X_neg_all)

            print(f"--- 'mixed' strategy: pooled {X.shape[0]} samples "
                  f"(pos={n_pos_total}, neg={n_neg_total}); splitting at natural rate, balancing train only.")

            sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_proportion, random_state=random_state)

            for train_idx, test_idx in sss.split(X, y):
                X_train_all, y_train_all = X[train_idx], y[train_idx]
                X_test, y_test = X[test_idx], y[test_idx]

                # Split the training fold back into classes and balance 50/50.
                X_pos_train = X_train_all[y_train_all == 1]
                X_neg_train = X_train_all[y_train_all == 0]
                X_pos_train_bal, X_neg_train_bal = balance_two_class_arrays(X_pos_train, X_neg_train)

                if X_pos_train_bal.shape[0] == 0:
                    print(f"Warning: No balanced training data for a 'mixed' split. Skipping.")
                    continue

                X_train, y_train = concat_two_class_with_labels(X_pos_train_bal, X_neg_train_bal)
                yield shuffle_train_test_arrays(X_train, y_train, X_test, y_test)

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

        ### Strategy 3: 'null_control' (pooled no-bout, matching new 'mixed' shape per split)
        elif split_strategy == 'null_control':
            print("--- Using 'null_control' (pooled) strategy (No-Bout vs. No-Bout) ---")

            if n_pos_total == 0 or n_neg_total == 0:
                print(f"Warning: Missing class data (pos={n_pos_total}, neg={n_neg_total}). Skipping null control.")
                return

            # Mirror 'mixed': stratified split on the natural-rate pool to discover the
            # per-split train/test sizes and class ratios, then build fake arrays from
            # No-Bout data that match those sizes/ratios. Training is balanced 50/50,
            # test preserves the natural class prior.
            X_real, y_real = concat_two_class_with_labels(X_pos_all, X_neg_all)
            sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_proportion, random_state=random_state)

            for split_num, (train_idx, test_idx) in enumerate(sss.split(X_real, y_real)):
                y_train_real = y_real[train_idx]
                y_test_real = y_real[test_idx]

                n_pos_train_nat = int(np.sum(y_train_real == 1))
                n_neg_train_nat = int(np.sum(y_train_real == 0))
                n_balanced_train_half = min(n_pos_train_nat, n_neg_train_nat)

                n_test_pos_target = int(np.sum(y_test_real == 1))
                n_test_neg_target = int(np.sum(y_test_real == 0))

                if n_balanced_train_half == 0:
                    print("Warning: No balanced training data available for a 'null_control' split. Skipping.")
                    continue

                n_total_needed = (2 * n_balanced_train_half) + n_test_pos_target + n_test_neg_target

                # Seeded per split so the null-control draw is reproducible and
                # does not inherit whatever state prior code left on the global
                # NumPy RNG.
                null_rng = np.random.default_rng(random_state + split_num)
                if n_neg_total < n_total_needed:
                    print(f"Warning: Not enough No-Bout samples ({n_neg_total}) for null control of size "
                          f"{n_total_needed}. Sampling with replacement.")
                    draw = null_rng.choice(n_neg_total, size=n_total_needed, replace=True)
                else:
                    draw = null_rng.permutation(n_neg_total)[:n_total_needed]

                c1 = n_balanced_train_half
                c2 = c1 + n_balanced_train_half
                c3 = c2 + n_test_pos_target
                c4 = c3 + n_test_neg_target

                X_fake_pos_train = X_neg_all[draw[:c1]]
                X_fake_neg_train = X_neg_all[draw[c1:c2]]
                X_fake_pos_test = X_neg_all[draw[c2:c3]]
                X_fake_neg_test = X_neg_all[draw[c3:c4]]

                X_train, y_train = concat_two_class_with_labels(X_fake_pos_train, X_fake_neg_train)
                X_test, y_test = concat_two_class_with_labels(X_fake_pos_test, X_fake_neg_test)

                yield shuffle_train_test_arrays(X_train, y_train, X_test, y_test)

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

                # Pool actual bout / no-bout data once per session list — the
                # "pool only No-Bout" duplicate call below is redundant since
                # `pool_session_arrays` already returns both classes.
                X_pos_train_actual, X_neg_train_all = pool_session_arrays(feature_data, train_session_list, pos_key="usv_feature_arr", neg_key="no_usv_feature_arr", n_frames=self.history_frames)
                X_pos_test_actual, X_neg_test_all = pool_session_arrays(feature_data, test_session_list, pos_key="usv_feature_arr", neg_key="no_usv_feature_arr", n_frames=self.history_frames)

                # Find the target sizes from the actual data
                X_pos_train_bal, X_neg_train_bal = balance_two_class_arrays(X_pos_train_actual, X_neg_train_all)

                n_balanced_train_half = X_pos_train_bal.shape[0]
                n_total_train_needed = n_balanced_train_half * 2

                n_test_pos_target = X_pos_test_actual.shape[0]
                n_test_neg_target = X_neg_test_all.shape[0]

                if n_balanced_train_half == 0:
                    print(f"Warning: No *actual* balanced training data for split {split_num}. Cannot match size. Skipping.")
                    continue

                # Create fake balanced training set, matching 'session' size.
                # The per-split `default_rng` keeps every draw reproducible and
                # independent of whatever global RNG state prior calls left.
                null_rng = np.random.default_rng(random_state + split_num)
                n_train_neg_available = X_neg_train_all.shape[0]

                if n_train_neg_available < n_total_train_needed:
                    print(f"Warning: Not enough No-Bout samples ({n_train_neg_available}) in train sessions to create null control of size {n_total_train_needed}. Sampling with replacement.")
                    train_neg_indices = null_rng.choice(n_train_neg_available, size=n_total_train_needed, replace=True)
                else:
                    train_neg_indices = null_rng.permutation(n_train_neg_available)

                X_fake_pos_train = X_neg_train_all[train_neg_indices[:n_balanced_train_half]]
                X_fake_neg_train = X_neg_train_all[train_neg_indices[n_balanced_train_half: n_total_train_needed]]

                X_train, y_train = concat_two_class_with_labels(X_fake_pos_train, X_fake_neg_train)

                # Create fake UNBALANCED test set, matching 'session' size AND ratio
                n_test_neg_available = X_neg_test_all.shape[0]

                if n_test_neg_available == 0:
                    print(f"Warning: No No-Bout test samples available for split {split_num}. Skipping.")
                    continue

                fake_pos_indices = null_rng.choice(n_test_neg_available, size=n_test_pos_target, replace=True)
                fake_neg_indices = null_rng.choice(n_test_neg_available, size=n_test_neg_target, replace=True)

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
        5.  Re-fits the same estimator on a reproducibly label-shuffled copy
            of the training target (seeded per split from `random_seed`) to
            establish a null performance distribution (AUC / log-loss / Brier
            / ECE) for significance testing against the `actual` branch.

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
            contains 'actual' and 'null' sub-structures (the 'null' branch holds
            the label-shuffled control — same estimator fit to a permuted target)
            with arrays for:
            - `filter_shapes` : reconstructed temporal filters across splits.
            - `coefs_projected` : raw weights in the basis space.
            - `optimal_C` : regularisation strength selected by `LogisticRegressionCV`.
            - `auc` : threshold-free ranking quality (macro one-vs-rest ROC-AUC);
              imbalance-robust but blind to calibration.
            - `score` : balanced accuracy (mean per-class recall).
            - `recall` : positive-class recall; kept because macro-F1 already
              summarizes the precision/recall trade-off.
            - `f1` : binary F1 (harmonic mean of precision and recall).
            - `ll` : log-loss; strictly proper probabilistic score that punishes
              confident wrong predictions.
            - `brier` : Brier score (mean squared error of the positive-class
              probability). Quadratic counterpart to log-loss, more robust to
              occasional overconfidence.
            - `ece` : top-label Expected Calibration Error (10 bins). Measures
              whether "N%-confident" predictions are right N% of the time.
            - `mcc` : Matthews correlation coefficient; chance-corrected,
              imbalance-robust summary in [-1, +1].
            - `confusion_matrix` : (2, 2) per-split matrix with `[0, 1]` label
              ordering. Cheap to persist, lets downstream code derive precision
              and sensitivity / specificity on demand.
            - `n_iter`, `converged`, `fit_time` : optimizer diagnostics exposing
              folds that terminated at `max_iter` without converging.
            - `split_sizes` : metadata tracking the N for each training/test fold.
        """

        # Initialize results structure with 'actual' and 'shuffled' keys
        n_splits = self.modeling_settings['model_params']['split_num']
        n_bases = basis_matrix.shape[1]
        history_frames = basis_matrix.shape[0]

        # Scalar per-split metrics. `precision` is dropped (recoverable from
        # the saved confusion matrix), Brier/ECE/MCC are added as calibration
        # and chance-corrected summary scores, and `n_iter`/`converged`/
        # `fit_time` expose silent-failure modes of the underlying solver.
        def _scalar():
            return np.full(n_splits, np.nan)

        def _make_branch():
            return {
                'filter_shapes': np.full((n_splits, history_frames), np.nan),
                'coefs_projected': np.full((n_splits, n_bases), np.nan),
                'optimal_C': _scalar(),
                'score': _scalar(), 'recall': _scalar(),
                'f1': _scalar(), 'auc': _scalar(),
                'll': _scalar(),
                'brier': _scalar(), 'ece': _scalar(), 'mcc': _scalar(),
                'confusion_matrix': np.full((n_splits, 2, 2), np.nan),
                'n_iter': _scalar(), 'converged': _scalar(), 'fit_time': _scalar()
            }

        # Renamed "shuffled" -> "null" to match the modeling vocabulary used
        # by the multinomial / manifold pipelines (all three mean "same
        # training procedure, label-destroyed control").
        results = {
            'actual': _make_branch(),
            'null': _make_branch(),
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
                lr_max_iter = self.modeling_settings['hyperparameters']['classical']['logistic_regression']['max_iter']
                fit_start = time.perf_counter()
                lr_actual = LogisticRegressionCV(
                    penalty=self.modeling_settings['hyperparameters']['classical']['logistic_regression']['penalty'],
                    Cs=self.modeling_settings['hyperparameters']['classical']['logistic_regression']['cs'],
                    cv=self.modeling_settings['hyperparameters']['classical']['logistic_regression']['cv'],
                    class_weight='balanced',
                    solver=self.modeling_settings['hyperparameters']['classical']['logistic_regression']['solver'],
                    max_iter=lr_max_iter,
                    random_state=self.modeling_settings['model_params']['random_seed']
                ).fit(X_train_proj, y_train)
                fit_time = float(time.perf_counter() - fit_start)

                y_pred_actual = lr_actual.predict(X_test_proj)
                y_proba_actual = lr_actual.predict_proba(X_test_proj)[:, 1]

                results['actual']['coefs_projected'][split_idx, :] = lr_actual.coef_.flatten()
                results['actual']['optimal_C'][split_idx] = lr_actual.C_[0]
                filter_shape_actual = np.dot(lr_actual.coef_, basis_matrix.T).ravel()
                results['actual']['filter_shapes'][split_idx, :] = filter_shape_actual
                results['actual']['score'][split_idx] = balanced_accuracy_score(y_test, y_pred_actual)
                results['actual']['recall'][split_idx] = recall_score(y_test, y_pred_actual, zero_division=0.0)
                results['actual']['f1'][split_idx] = f1_score(y_test, y_pred_actual, average='binary', zero_division=0.0)
                results['actual']['mcc'][split_idx] = safe_matthews_corrcoef(y_test, y_pred_actual)
                results['actual']['confusion_matrix'][split_idx] = safe_confusion_matrix(
                    y_test, y_pred_actual, labels=np.array([0, 1])
                )
                try:
                    results['actual']['n_iter'][split_idx] = float(np.max(lr_actual.n_iter_))
                    results['actual']['converged'][split_idx] = float(results['actual']['n_iter'][split_idx] < lr_max_iter)
                except Exception:
                    pass
                results['actual']['fit_time'][split_idx] = fit_time

                if len(np.unique(y_test)) > 1:
                    results['actual']['auc'][split_idx] = roc_auc_score(y_test, y_proba_actual)
                    epsilon = 1e-15
                    y_proba_actual_clipped = np.clip(y_proba_actual, epsilon, 1 - epsilon)
                    results['actual']['ll'][split_idx] = log_loss(y_test, y_proba_actual_clipped)
                    results['actual']['brier'][split_idx] = float(brier_score_loss(y_test, y_proba_actual))
                    try:
                        y_proba_2d = np.column_stack([1.0 - y_proba_actual, y_proba_actual])
                        results['actual']['ece'][split_idx] = expected_calibration_error(y_test, y_pred_actual, y_proba_2d, n_bins=10)
                    except Exception:
                        pass

            except Exception as e:
                print(f"  ERROR during ACTUAL model fit/predict for {feature_name}, split {split_idx}: {e}")

            try:
                # Seeded per-split Generator so the label shuffle is reproducible
                # and independent of any ambient global RNG state.
                shuffle_rng = np.random.default_rng(
                    self.modeling_settings['model_params']['random_seed'] + split_idx + 1
                )
                y_train_shuffled = shuffle_rng.permutation(y_train)

                fit_start = time.perf_counter()
                lr_shuffled = LogisticRegressionCV(
                    penalty=self.modeling_settings['hyperparameters']['classical']['logistic_regression']['penalty'],
                    Cs=self.modeling_settings['hyperparameters']['classical']['logistic_regression']['cs'],
                    cv=self.modeling_settings['hyperparameters']['classical']['logistic_regression']['cv'],
                    class_weight='balanced',
                    solver=self.modeling_settings['hyperparameters']['classical']['logistic_regression']['solver'],
                    max_iter=lr_max_iter,
                    random_state=self.modeling_settings['model_params']['random_seed']
                ).fit(X_train_proj, y_train_shuffled)
                fit_time = float(time.perf_counter() - fit_start)

                y_pred_shuffled = lr_shuffled.predict(X_test_proj)
                y_proba_shuffled = lr_shuffled.predict_proba(X_test_proj)[:, 1]

                results['null']['coefs_projected'][split_idx, :] = lr_shuffled.coef_.flatten()
                results['null']['optimal_C'][split_idx] = lr_shuffled.C_[0]
                filter_shape_shuffled = np.dot(lr_shuffled.coef_, basis_matrix.T).ravel()
                results['null']['filter_shapes'][split_idx, :] = filter_shape_shuffled
                results['null']['score'][split_idx] = balanced_accuracy_score(y_test, y_pred_shuffled)
                results['null']['recall'][split_idx] = recall_score(y_test, y_pred_shuffled, zero_division=0.0)
                results['null']['f1'][split_idx] = f1_score(y_test, y_pred_shuffled, average='binary', zero_division=0.0)
                results['null']['mcc'][split_idx] = safe_matthews_corrcoef(y_test, y_pred_shuffled)
                results['null']['confusion_matrix'][split_idx] = safe_confusion_matrix(
                    y_test, y_pred_shuffled, labels=np.array([0, 1])
                )
                try:
                    results['null']['n_iter'][split_idx] = float(np.max(lr_shuffled.n_iter_))
                    results['null']['converged'][split_idx] = float(results['null']['n_iter'][split_idx] < lr_max_iter)
                except Exception:
                    pass
                results['null']['fit_time'][split_idx] = fit_time

                if len(np.unique(y_test)) > 1:
                    results['null']['auc'][split_idx] = roc_auc_score(y_test, y_proba_shuffled)
                    epsilon = 1e-15
                    y_proba_shuffled_clipped = np.clip(y_proba_shuffled, epsilon, 1 - epsilon)
                    results['null']['ll'][split_idx] = log_loss(y_test, y_proba_shuffled_clipped)
                    results['null']['brier'][split_idx] = float(brier_score_loss(y_test, y_proba_shuffled))
                    try:
                        y_proba_2d = np.column_stack([1.0 - y_proba_shuffled, y_proba_shuffled])
                        results['null']['ece'][split_idx] = expected_calibration_error(y_test, y_pred_shuffled, y_proba_2d, n_bins=10)
                    except Exception:
                        pass

            except Exception as e:
                print(f"  ERROR during NULL model fit/predict for {feature_name}, split {split_idx}: {e}")

        results['split_sizes']['train'] = np.array(results['split_sizes']['train'])
        results['split_sizes']['test'] = np.array(results['split_sizes']['test'])

        if split_has_data:
            mean_auc_actual = np.nanmean(results['actual']['auc'])
            print_msg = f"  --- Finished {feature_name}. Mean Actual AUC: {mean_auc_actual:.4f}"
            mean_auc_null = np.nanmean(results['null']['auc'])
            print_msg += f", Mean Null AUC: {mean_auc_null:.4f}"
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
            and a 'null' (label-shuffled) model. Note: For evaluation, the tiled
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
            2. results (dict): A nested dictionary containing 'actual' and 'null'
               keys (the 'null' branch holds the label-shuffled control), each
               holding arrays of metrics:
               - `filter_shapes` : 1D partial-dependence filter per split.
               - `auc` : macro one-vs-rest ROC-AUC (threshold-free ranking).
               - `score` : balanced accuracy.
               - `recall`, `f1` : standard imbalance-robust summaries.
               - `ll` : log-loss (strictly proper probabilistic score).
               - `brier` : mean squared error on predicted probabilities.
               - `ece` : top-label calibration error (10 equal-width bins).
               - `mcc` : Matthews correlation coefficient.
               - `confusion_matrix` : (2, 2) per-split confusion matrix.
               - `n_iter`, `converged`, `fit_time` : optimizer diagnostics.
               Also contains a 'split_sizes' key. 'null' metrics are NaN if not
               calculated.
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

        def _scalar():
            return np.full(n_splits, np.nan)

        def _make_pygam_branch():
            return {
                'filter_shapes': np.full((n_splits, history_frames), np.nan),
                'coefs_projected': np.empty((n_splits, 0)),
                'optimal_C': np.empty((n_splits, 0)),
                'score': _scalar(), 'recall': _scalar(),
                'f1': _scalar(), 'auc': _scalar(), 'll': _scalar(),
                'brier': _scalar(), 'ece': _scalar(), 'mcc': _scalar(),
                'confusion_matrix': np.full((n_splits, 2, 2), np.nan),
                'n_iter': _scalar(), 'converged': _scalar(), 'fit_time': _scalar()
            }

        results = {
            'actual': _make_pygam_branch(),
            'null': _make_pygam_branch(),
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
                    fit_start = time.perf_counter()
                    gam_actual = LogisticGAM(
                        te(0, 1, n_splines=[n_splines_value, n_splines_time]), **gam_kwargs_actual
                    ).fit(X_train_gam, y_train_tiled)
                    fit_time = float(time.perf_counter() - fit_start)

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
                    results['actual']['score'][split_idx] = balanced_accuracy_score(y_test_int, y_pred_mean_epoch)
                    results['actual']['recall'][split_idx] = recall_score(y_test_int, y_pred_mean_epoch, zero_division=0.0)
                    results['actual']['f1'][split_idx] = f1_score(y_test_int, y_pred_mean_epoch, average='binary', zero_division=0.0)
                    results['actual']['mcc'][split_idx] = safe_matthews_corrcoef(y_test_int, y_pred_mean_epoch)
                    results['actual']['confusion_matrix'][split_idx] = safe_confusion_matrix(
                        y_test_int, y_pred_mean_epoch, labels=np.array([0, 1])
                    )
                    results['actual']['n_iter'][split_idx] = float(len(diffs))
                    results['actual']['converged'][split_idx] = float(bool(diffs and diffs[-1] < tol_val))
                    results['actual']['fit_time'][split_idx] = fit_time

                    if len(np.unique(y_test_int)) > 1:
                        results['actual']['auc'][split_idx] = roc_auc_score(y_test_int, y_proba_mean_epoch)
                        results['actual']['ll'][split_idx] = log_loss(y_test_int, np.clip(y_proba_mean_epoch, 1e-15, 1 - 1e-15))
                        results['actual']['brier'][split_idx] = float(brier_score_loss(y_test_int, y_proba_mean_epoch))
                        try:
                            y_proba_2d = np.column_stack([1.0 - y_proba_mean_epoch, y_proba_mean_epoch])
                            results['actual']['ece'][split_idx] = expected_calibration_error(y_test_int, y_pred_mean_epoch, y_proba_2d, n_bins=10)
                        except Exception:
                            pass

                    print(f"    > ACTUAL Fold {split_idx} (Train N={len(y_train)}, Test N={len(y_test)}): "
                          f"AUC={results['actual']['auc'][split_idx]:.3f}, "
                          f"LL={results['actual']['ll'][split_idx]:.3f}, "
                          f"Brier={results['actual']['brier'][split_idx]:.3f}")

                except Exception as e:
                    print(f"  ERROR during ACTUAL [pygam] fit/predict for {feature_name}, split {split_idx}: {e}")

            if shuffled_split and shuffled_split[0].shape[0] > 0 and shuffled_split[2].shape[0] > 0:
                (X_train_null, y_train_null, X_test_null, y_test_null) = shuffled_split
                print(f"  NULL Split {split_idx}: Train={X_train_null.shape}, Test={X_test_null.shape}")

                y_train_tiled_null = np.repeat(y_train_null.astype(np.float32), history_frames)
                y_test_int_null = y_test_null.astype(int)
                X_train_gam_null = unroll_data_for_gam(X_train_null.astype(np.float32))
                X_test_gam_null = unroll_data_for_gam(X_test_null.astype(np.float32))

                try:
                    fit_start = time.perf_counter()
                    gam_shuffled = LogisticGAM(
                        te(0, 1, n_splines=[n_splines_value, n_splines_time]), **gam_kwargs_shuffled
                    ).fit(X_train_gam_null, y_train_tiled_null)
                    fit_time = float(time.perf_counter() - fit_start)

                    y_proba_shuffled_tiled = gam_shuffled.predict_proba(X_test_gam_null)
                    y_proba_shuffled_mean = np.mean(y_proba_shuffled_tiled.reshape(X_test_null.shape), axis=1)
                    y_pred_shuffled_mean = (y_proba_shuffled_mean > 0.5).astype(int)

                    grid_X_0_null = np.stack([np.zeros(history_frames, dtype=np.float32), time_indices], axis=1)
                    grid_X_1_null = np.stack([np.ones(history_frames, dtype=np.float32), time_indices], axis=1)
                    # predict_mu returns the Bernoulli mean (probability), not log-odds.
                    prob_0_null = gam_shuffled.predict_mu(grid_X_0_null).astype(np.float32)
                    prob_1_null = gam_shuffled.predict_mu(grid_X_1_null).astype(np.float32)
                    filter_shape_null = (prob_1_null - prob_0_null).flatten()
                    results['null']['filter_shapes'][split_idx, :] = filter_shape_null

                    results['null']['score'][split_idx] = balanced_accuracy_score(y_test_int_null, y_pred_shuffled_mean)
                    results['null']['recall'][split_idx] = recall_score(y_test_int_null, y_pred_shuffled_mean, zero_division=0.0)
                    results['null']['f1'][split_idx] = f1_score(y_test_int_null, y_pred_shuffled_mean, average='binary', zero_division=0.0)
                    results['null']['mcc'][split_idx] = safe_matthews_corrcoef(y_test_int_null, y_pred_shuffled_mean)
                    results['null']['confusion_matrix'][split_idx] = safe_confusion_matrix(
                        y_test_int_null, y_pred_shuffled_mean, labels=np.array([0, 1])
                    )
                    null_diffs = gam_shuffled.logs_.get('diffs', [])
                    results['null']['n_iter'][split_idx] = float(len(null_diffs))
                    results['null']['converged'][split_idx] = float(bool(null_diffs and null_diffs[-1] < tol_val))
                    results['null']['fit_time'][split_idx] = fit_time

                    if len(np.unique(y_test_int_null)) > 1:
                        results['null']['auc'][split_idx] = roc_auc_score(y_test_int_null, y_proba_shuffled_mean)
                        results['null']['ll'][split_idx] = log_loss(y_test_int_null, np.clip(y_proba_shuffled_mean, 1e-15, 1 - 1e-15))
                        results['null']['brier'][split_idx] = float(brier_score_loss(y_test_int_null, y_proba_shuffled_mean))
                        try:
                            y_proba_2d = np.column_stack([1.0 - y_proba_shuffled_mean, y_proba_shuffled_mean])
                            results['null']['ece'][split_idx] = expected_calibration_error(y_test_int_null, y_pred_shuffled_mean, y_proba_2d, n_bins=10)
                        except Exception:
                            pass

                except Exception as e:
                    print(f"  ERROR during NULL [pygam] fit/predict for {feature_name}, split {split_idx}: {e}")

        results['split_sizes']['train'] = np.array(results['split_sizes']['train'])
        results['split_sizes']['test'] = np.array(results['split_sizes']['test'])

        if split_has_data_actual:
            mean_auc_actual = np.nanmean(results['actual']['auc'])
            print_msg = f"  --- Finished {feature_name} [pygam]. Mean Actual AUC: {mean_auc_actual:.4f}"
            mean_auc_null = np.nanmean(results['null']['auc'])
            print_msg += f", Mean Null (Null Control) AUC: {mean_auc_null:.4f}"
            print(print_msg)
        else:
            print(f"  --- No valid splits processed for feature: {feature_name} [pygam] ---")

        return feature_name, results
