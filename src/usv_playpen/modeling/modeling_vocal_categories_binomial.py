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
from pathlib import Path
import pickle
import time
from pygam import LogisticGAM, te
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, log_loss, f1_score, recall_score, balanced_accuracy_score, brier_score_loss
from tqdm import tqdm

from .load_input_files import load_behavioral_feature_data, find_usv_categories
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
    harmonize_session_columns,
    zscore_features_across_sessions,
    run_predictor_audits,
    pool_session_arrays,
    balance_two_class_arrays,
    unroll_history_matrix,
    concat_two_class_with_labels,
    shuffle_train_test_arrays,
    expected_calibration_error,
    safe_matthews_corrcoef,
    safe_confusion_matrix,
)
from ..analyses.compute_behavioral_features import FeatureZoo


def _collect_category_windows(times: np.ndarray,
                              column_data: np.ndarray,
                              sampling_rate: float,
                              history_frames: int,
                              max_frame_idx: int) -> np.ndarray:
    """
    Slice history windows around event timestamps, skipping out-of-bounds rows.

    For each timestamp `t` in `times`, the window `[end - history_frames, end)`
    with `end = round(t * sampling_rate)` is extracted from `column_data`.
    Events whose window falls off the start or end of the recording are
    silently skipped (rather than being stored as a NaN row), so the returned
    array only contains fully-populated history windows. Any within-window
    NaN samples are replaced with `0.0` — that is the intended treatment of
    transient tracking dropouts inside an otherwise valid window.

    Parameters
    ----------
    times : np.ndarray
        Event timestamps in seconds, shape `(n_events,)`.
    column_data : np.ndarray
        The 1-D behavioural trace sampled at `sampling_rate`.
    sampling_rate : float
        Frames per second of `column_data`.
    history_frames : int
        Length of the slice preceding each event, in frames.
    max_frame_idx : int
        Length of `column_data`; `end` must not exceed this value.

    Returns
    -------
    np.ndarray
        Dense `(n_valid_events, history_frames)` matrix whose rows are the
        fully-populated history windows (in `column_data`'s dtype).
    """

    ends = np.round(times * sampling_rate).astype(int)
    starts = ends - history_frames
    valid_mask = (starts >= 0) & (ends <= max_frame_idx)
    if not np.any(valid_mask):
        return np.empty((0, history_frames), dtype=column_data.dtype)
    valid_starts = starts[valid_mask]
    valid_ends = ends[valid_mask]
    out = np.empty((valid_starts.size, history_frames), dtype=column_data.dtype)
    for row_idx, (s, e) in enumerate(zip(valid_starts, valid_ends)):
        chunk = column_data[s:e].copy()
        chunk[np.isnan(chunk)] = 0.0
        out[row_idx, :] = chunk
    return out


class VocalCategoryModelingPipeline(FeatureZoo):
    """
    End-to-end pipeline for one-vs-rest USV-category modelling from behavioural kinematics.

    The pipeline has three responsibilities:

    1. **Data preparation**: loads behavioural time-series and USV category
       assignments, separates the `target_category` bouts from the pooled
       "other" bouts, injects vocal-syntax predictors (with identity guards
       to prevent trivial self-prediction), applies cross-session z-scoring,
       and serializes the resulting `{feature: {session: {target_feature_arr,
       other_feature_arr}}}` dictionary to disk.
    2. **Cross-validation splitting**: the `create_category_splits` generator
       implements two strategies ('session' and 'mixed') plus a size- and
       ratio-matched `'null_other'` condition that draws pseudo-classes from
       the negative pool. All strategies honour the canonical "balanced train
       / natural-rate test" invariant — training is down-sampled to 50/50
       so gradient updates see both classes, while the test fold preserves
       the natural base rate so reported metrics reflect real imbalance.
    3. **Univariate model fitting**: `_run_modeling_category` fits either a
       basis-projected `LogisticRegressionCV` (sklearn engine) or a
       tensor-product-spline `LogisticGAM` (pygam engine) per feature and
       returns an `{'actual', 'null'}` results dict with calibration (Brier,
       ECE), chance-corrected (MCC), confusion-matrix, and optimizer-
       diagnostic fields per split.
    """

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
            settings_path = Path(__file__).resolve().parent.parent / '_parameter_settings/modeling_settings.json'
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
              from the loading module (gated by
              `vocal_features.usv_predictor_partner_only`).
              * **Partner Mouse:** all vocal traces are ingested as predictors
                whenever the partner vocalized in any session; under the
                `partner_only=True` default this is the only source of vocal
                predictors.
              * **Subject Mouse (Self):** when `partner_only=False`, ingests
                subject-side vocal categories to capture syntax transitions
                (e.g., Cat 1 predicting Cat 3). To ensure scientific validity,
                it strictly excludes the 'target_category' itself to avoid
                self-prediction, and excludes density traces (proportion /
                event) to prevent trivial autocorrelation.
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
            noise_vocal_categories=noise_cats,
            noise_column=voc_settings['usv_noise_column'],
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
            feature_bounds=getattr(self, 'feature_boundaries', {}),
            abs_features=['allo_roll', 'allo_yaw-nose', 'nose-allo_yaw', 'allo_yaw-TTI', 'TTI-allo_yaw']
        )

        # Build the unified Level-1 filename — the analysis_tag includes
        # the category being modeled, the experimental_condition pins
        # the cohort, and the rest of the historical fields (voc_mode,
        # hist) live in `_input_metadata`.
        cohort_condition = derive_experimental_condition(self.modeling_settings)
        analysis_tag = f"category-{target_category}"
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        fname = f"modeling_{analysis_tag}_{cohort_condition}_{ts}.pkl"

        # Build `_input_metadata` once. Per-session event counts are
        # backfilled at the dump site (after epoch slicing).
        gmm_idx = self.modeling_settings['model_params']['gmm_component_index']
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

        first_sess_id = next(iter(processed_beh_data))
        kept_columns_first_sess = list(processed_beh_data[first_sess_id].columns)
        feature_zoo_kept_md = sorted({
            c.split('.', 1)[-1] if '.' in c and not c.split('.')[-1].isdigit() else c
            for c in kept_columns_first_sess
        })
        vocal_columns_md = sorted({
            c for c in kept_columns_first_sess
            if any(tok in c for tok in ('usv_rate', 'usv_cat_', 'usv_event'))
        })

        input_metadata = build_input_metadata(
            modeling_settings=self.modeling_settings,
            analysis_type='category',
            analysis_tag=analysis_tag,
            pipeline_class=type(self).__name__,
            target_idx=targ_idx,
            predictor_idx=pred_idx,
            n_sessions_used=len(processed_beh_data),
            session_ids=sorted(processed_beh_data.keys()),
            n_events_per_session={},
            feature_zoo_full=derive_feature_zoo_full(self.modeling_settings),
            feature_zoo_kept=feature_zoo_kept_md,
            dyadic_engagement_features_used=list(kin_settings['dyadic_engagement']),
            dyadic_pose_symmetric_features_used=kin_settings['dyadic_pose_symmetric'],
            noise_vocal_categories_excluded=list(noise_cats),
            vocal_signal_columns_added=vocal_columns_md,
            filter_history_seconds=float(filter_hist),
            filter_history_frames=int(self.history_frames),
            camera_sampling_rate_hz=derive_camera_fps_field(cam_fps_dict),
            ibi_thresholds=ibi_thresholds_md,
            analysis_specific={
                'target_category': int(target_category),
                'category_self_exclude': list(category_self_exclude),
            },
        )

        # Predictor diagnostics audit (collinearity + timescales). Runs on
        # the harmonized, z-scored, but not-yet-sliced feature dict.
        # Diagnostic-only: any failure inside the wrapper warns and
        # continues.
        run_predictor_audits(
            processed_beh_dict=processed_beh_data,
            usv_data_dict=usv_data_dict,
            mouse_names_dict=mouse_names_dict,
            camera_fps_dict=cam_fps_dict,
            target_idx=targ_idx,
            predictor_idx=pred_idx,
            history_frames=self.history_frames,
            event_keys=['target_events', 'other_events'],
            settings=self.modeling_settings,
            save_dir=self.modeling_settings['io']['save_directory'],
            pickle_basename=fname,
            input_metadata=input_metadata,
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

                # Events whose full history window does not fit inside the
                # session recording are dropped at extraction time instead of
                # being carried forward as NaN rows. Downstream pooling /
                # balancing helpers do not filter NaN, so leaving them in
                # would silently corrupt basis projections.
                t_arr = _collect_category_windows(target_times, col_data, fps, hist_frames, max_idx)
                o_arr = _collect_category_windows(other_times, col_data, fps, hist_frames, max_idx)
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

        save_dir = Path(self.modeling_settings['io']['save_directory'])
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / fname

        # Backfill per-session counts from the anchor feature.
        n_events_per_session = {}
        if final_features:
            anchor_feat = final_features[0]
            for sess_id in final_data[anchor_feat]:
                n_events_per_session[sess_id] = {
                    'target': int(final_data[anchor_feat][sess_id]['target_feature_arr'].shape[0]),
                    'other': int(final_data[anchor_feat][sess_id]['other_feature_arr'].shape[0]),
                }
        input_metadata['n_events_per_session'] = n_events_per_session
        input_metadata['feature_zoo_kept'] = sorted(final_data.keys())

        artifact = inject_metadata(final_data, _input_metadata=input_metadata)
        with save_path.open('wb') as f:
            pickle.dump(artifact, f)
        print(f"\n[+] Successfully saved category input data to:\n    {save_path}")

    def create_category_splits(self, feature_data: dict, strategy: str = 'actual'):
        """
        Generator yielding train/test splits for K-Fold validation.

        This function orchestrates the data splitting for model validation, dynamically
        handling both 'session' and 'mixed' cross-validation strategies based on the
        project configuration. Across all strategies, the training fold is down-sampled
        to a 50/50 class balance (Target vs. Other), while the test fold is kept at
        the natural class prior. This keeps gradient updates informative for both
        classes during training, while evaluation reflects the true imbalance of the
        task. Reported metrics should therefore be imbalance-robust
        (e.g., balanced_accuracy, AUC, log-loss).

        Splitting strategies:
        ---------------------
        1. 'session': Evaluates model generalizability across independent recording sessions.
           Whole sessions are held out for testing (e.g., Train on Sessions A and B;
           Test on Session C). Training is balanced 50/50 per split; test retains the
           natural class prior of the held-out sessions.
        2. 'mixed': Pools epochs from all sessions and uses `StratifiedShuffleSplit`
           on the *unbalanced* pool so each split preserves the natural class ratio
           in both halves. The training fold is then balanced 50/50; the test fold
           is left at the natural base rate.

        Balancing Process:
        ------------------
        1.  Identifies sessions containing both target (positive) and other (negative) classes.
        2.  Executes the selected split strategy ('session' or 'mixed') to generate base
            training and testing data pools at the natural class prior.
        3.  Executes the selected condition ('actual' or 'null_other'):
            - 'actual': Balances the training Target vs. Other data 50/50; test
              data is passed through at the natural class prior.
            - 'null_other': Uses ONLY 'other' data to construct pseudo-classes.
              The pseudo-training set is balanced 50/50 and sized to the 'actual'
              balanced training size for this split. The pseudo-test set is
              size- and ratio-matched to the 'actual' natural-rate test set,
              preventing sample-size or prior mismatches between real and null
              comparisons.

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
            represented as NumPy arrays. X_train/y_train are class-balanced;
            X_test/y_test preserve the natural class prior of the source data.
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
                X_pooled, y_pooled = concat_two_class_with_labels(X_all_targ, X_all_other)

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

        # Phase 2: Balance training folds only; keep test folds at the natural class prior.
        for split_num, (X_tr_targ, X_tr_other, X_te_targ, X_te_other) in enumerate(splits_data):

            n_tr_limit = min(X_tr_targ.shape[0], X_tr_other.shape[0])
            n_te_target = X_te_targ.shape[0]
            n_te_other = X_te_other.shape[0]

            if n_tr_limit == 0 or (n_te_target + n_te_other) == 0:
                continue

            if strategy == 'actual':
                # Balance training 50/50; test passes through at natural rate.
                X_tr_A, X_tr_B = balance_two_class_arrays(X_tr_targ, X_tr_other)
                X_te_A = X_te_targ
                X_te_B = X_te_other

            elif strategy == 'null_other':
                # Pseudo-train: balanced 50/50, matching 'actual' train size (n_tr_limit per half).
                # Pseudo-test: size- and ratio-matched to 'actual' test (n_te_target + n_te_other
                # drawn from Other pool, split according to the natural-rate target/other shares).
                # Seeded per-split Generator so pseudo-class draws are
                # reproducible and do not leak ambient global RNG state.
                null_rng = np.random.default_rng(rand_seed + split_num)

                def draw_pseudo_train(X, limit):
                    needed = limit * 2
                    if len(X) < needed:
                        return None, None
                    X_sub = X[null_rng.choice(len(X), needed, replace=False)]
                    return X_sub[:limit], X_sub[limit:]

                def draw_pseudo_test(X, n_pseudo_pos, n_pseudo_neg):
                    needed = n_pseudo_pos + n_pseudo_neg
                    if needed == 0 or len(X) == 0:
                        return np.empty((0,) + X.shape[1:]), np.empty((0,) + X.shape[1:])
                    replace = len(X) < needed
                    X_sub = X[null_rng.choice(len(X), needed, replace=replace)]
                    return X_sub[:n_pseudo_pos], X_sub[n_pseudo_pos:]

                X_tr_A, X_tr_B = draw_pseudo_train(X_tr_other, n_tr_limit)
                X_te_A, X_te_B = draw_pseudo_test(X_te_other, n_te_target, n_te_other)

                if X_tr_A is None:
                    continue

            X_train, y_train = concat_two_class_with_labels(X_tr_A, X_tr_B)
            X_test, y_test = concat_two_class_with_labels(X_te_A, X_te_B)

            yield shuffle_train_test_arrays(X_train, y_train, X_test, y_test)

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
        4.  Metric aggregation: Computes fold-wise classification statistics for both
            actual and null models to enable significance testing.

        Metrics saved per split (see the `results[key][...]` layout):
        - `auc` : threshold-free ranking quality (ROC-AUC).
        - `score` : balanced accuracy; imbalance-robust hard-label accuracy.
        - `recall` : positive-class recall (precision is derivable from the
          saved confusion matrix on demand).
        - `f1` : binary F1 (harmonic mean of precision and recall).
        - `ll` : log-loss; strictly proper probabilistic score.
        - `brier` : Brier score on the positive-class probability; quadratic
          counterpart to log-loss, robust to occasional overconfidence.
        - `ece` : top-label Expected Calibration Error (10 equal-width bins);
          measures whether predicted confidences match empirical accuracies.
        - `mcc` : Matthews correlation coefficient; chance-corrected
          imbalance-robust summary in [-1, +1].
        - `confusion_matrix` : (2, 2) per-split matrix with `[0, 1]` labels.
        - `n_iter`, `converged`, `fit_time` : optimizer diagnostics that flag
          folds terminating at `max_iter` without meeting the tolerance.

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

        # Scalar per-split metrics. `precision` is dropped because it is
        # derivable on demand from the saved confusion matrices and macro-F1
        # already summarizes the precision / recall trade-off.
        metrics = ['auc', 'score', 'recall', 'f1', 'll', 'brier', 'ece', 'mcc']
        results = {
            'actual': {m: np.full(n_splits, np.nan) for m in metrics},
            'null': {m: np.full(n_splits, np.nan) for m in metrics}
        }

        # Per-split (2, 2) confusion matrix with canonical [0, 1] label
        # ordering and per-split optimizer diagnostics for silent-failure
        # detection (`converged=False` flags folds that hit `max_iter`).
        for key in ('actual', 'null'):
            results[key]['confusion_matrix'] = np.full((n_splits, 2, 2), np.nan)
            results[key]['n_iter'] = np.full(n_splits, np.nan)
            results[key]['converged'] = np.full(n_splits, np.nan)
            results[key]['fit_time'] = np.full(n_splits, np.nan)

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
                    fit_start = time.perf_counter()
                    fold_n_iter = np.nan
                    # `None` is an unambiguous "no diagnostic available" sentinel:
                    # `np.nan is not np.nan` is only True by CPython's singleton
                    # reuse, so the previous sentinel was fragile across interpreters.
                    fold_converged = None

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

                        # Sklearn exposes `n_iter_` per class / per C-grid cell;
                        # summarize as the max across folds of the regularisation
                        # CV grid so a non-converged fit is always visible.
                        try:
                            fold_n_iter = float(np.max(lr_actual.n_iter_))
                            fold_converged = bool(fold_n_iter < lr_params['max_iter'])
                        except Exception:
                            fold_n_iter, fold_converged = np.nan, None

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
                        fold_n_iter = float(len(diffs))
                        fold_converged = bool(diffs and diffs[-1] < gam_args['tol'])
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

                    fit_time = float(time.perf_counter() - fit_start)

                    if y_prob is not None and y_pred is not None:
                        y_prob_clipped = np.clip(y_prob, 1e-15, 1 - 1e-15)
                        y_proba_2d = np.column_stack([1.0 - y_prob, y_prob])

                        results[key]['auc'][split_idx] = roc_auc_score(y_te, y_prob)
                        results[key]['ll'][split_idx] = log_loss(y_te, y_prob_clipped)

                        results[key]['score'][split_idx] = balanced_accuracy_score(y_te, y_pred)
                        results[key]['recall'][split_idx] = recall_score(y_te, y_pred, zero_division=0.0)
                        results[key]['f1'][split_idx] = f1_score(y_te, y_pred, average='binary', zero_division=0.0)
                        results[key]['brier'][split_idx] = float(brier_score_loss(y_te, y_prob))
                        try:
                            results[key]['ece'][split_idx] = expected_calibration_error(y_te, y_pred, y_proba_2d, n_bins=10)
                        except Exception:
                            pass
                        results[key]['mcc'][split_idx] = safe_matthews_corrcoef(y_te, y_pred)
                        results[key]['confusion_matrix'][split_idx] = safe_confusion_matrix(
                            y_te, y_pred, labels=np.array([0, 1])
                        )
                        results[key]['n_iter'][split_idx] = fold_n_iter
                        results[key]['converged'][split_idx] = float(fold_converged) if fold_converged is not None else np.nan
                        results[key]['fit_time'][split_idx] = fit_time

                        print(f"    > {strat.capitalize()} Fold {split_idx} (Train N={len(y_tr)}, Test N={len(y_te)}): "
                              f"AUC={results[key]['auc'][split_idx]:.3f}, "
                              f"LL={results[key]['ll'][split_idx]:.3f}, "
                              f"Brier={results[key]['brier'][split_idx]:.3f}, "
                              f"MCC={results[key]['mcc'][split_idx]:.3f}")

                except Exception as e:
                    print(f"Fit error {feature_name} ({strat}), fold {split_idx}: {e}")

        mean_auc_act = np.nanmean(results['actual']['auc'])
        mean_auc_null = np.nanmean(results['null']['auc'])
        print(f"  Feature {feature_name}: Mean AUC={mean_auc_act:.3f} (Null={mean_auc_null:.3f})")

        return feature_name, results
