"""
@author: bartulem
Module for modeling continuous USV bout parameters.

This module provides a specialized pipeline for predicting continuous vocal
features — specifically "bout duration" and "bout complexity" — using behavioral and
vocal predictors. By inheriting from the VocalOnsetModelingPipeline, it
leverages standardized preprocessing and cross-session normalization while
implementing specialized logic for regression on strictly positive, skewed data.

Key Scientific and Computational Components:
1.  Focuses exclusively on valid USV bouts
    identified via GMM-based inter-vocal interval (IVI) clustering. Unlike
    onset modeling, this pipeline performs regression on the properties of
    the vocalization events themselves.
2.  Tailored for modeling non-negative, heavy-tailed
    distributions. Implements Gamma-distributed GAMs (via PyGAM) with Log-links
    and Ridge regression on log-transformed targets (via Sklearn) to ensure
    mathematical alignment with the biology of USV timing.
3.  Ingests continuous vocal traces from both mice.
    Maintains a "scientific guardrail" that allows prior vocal categories to
    act as predictors while excluding self-density signals (proportion/count)
    to prevent trivial autocorrelation.
4.  Employs a Repeated Monte Carlo splitting
    strategy (Stratified Group K-Fold) that uses quantile binning of the
    continuous target variable. This ensures that the training and testing
    sets represent the full range of bout durations and complexities.
5.  Evaluates models using distribution-appropriate
    metrics, including Explained Deviance (D^2), Gamma Deviance, and
    Mean Absolute Error (MAE), providing a more robust assessment than
    standard R^2 for skewed biological data.
"""

from datetime import datetime
import numpy as np
import os
import pickle
import gc
import time
from pygam import GAM, te
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_gamma_deviance, mean_squared_log_error
from tqdm import tqdm

from .modeling_vocal_onsets import VocalOnsetModelingPipeline
from .load_input_files import load_behavioral_feature_data, find_variable_length_bouts
from .modeling_utils import (
    prepare_modeling_sessions,
    resolve_mouse_roles,
    select_kinematic_columns,
    build_vocal_signal_columns,
    collect_predictor_suffixes,
    identify_empty_event_sessions,
    zero_fill_missing_feature_columns,
    zscore_features_across_sessions,
    run_predictor_audits,
    unroll_history_matrix,
    pearson_r_safe,
    root_mean_squared_error,
    mean_absolute_error_1d,
)


class BoutParameterPipeline(VocalOnsetModelingPipeline):
    """
    Pipeline for predicting continuous vocal bout parameters (Duration, Complexity).

    This class inherits from VocalOnsetModelingPipeline to reuse initialization,
    settings management, and parallelization structures. It overrides specific methods
    to handle the Regression task (Continuous Y) instead of Classification (Binary Y).

    Key differences from the base class:
    1.  **Data Extraction**: Extracts continuous targets (e.g., 'bout_durations') for
        valid bouts only. No "No-USV" epochs are created. It also dynamically
        injects smoothed vocal density signals (Proportion / Categories) into the
        predictor set.
    2.  **Splitting Strategy**: Implements 'Stratified Group K-Fold' (for 'session' strategy)
        and 'Stratified K-Fold' (for 'mixed' strategy) using quantile binning of the
        continuous target variable.
    3.  **Modeling**: Uses GammaGAM (pyGAM) and RidgeCV on Log(y) (sklearn) to model
        strictly positive, skewed distributions typical of duration/complexity data.
    """

    def __init__(self, modeling_settings_dict=None):
        """
        Initialize the pipeline.

        Passes the settings dictionary to the parent VocalOnsetModelingPipeline,
        which handles loading defaults (if None), flattening the dictionary structure,
        and initializing the FeatureZoo attributes (like feature_boundaries).

        Parameters
        ----------
        modeling_settings_dict : dict, optional
            The dictionary containing configuration settings. If None, the parent
            class loads '_parameter_settings/modeling_settings.json'.
        """

        super().__init__(modeling_settings_dict=modeling_settings_dict)

    def extract_and_save_modeling_input_data(self) -> None:
        """
        Extracts, processes, and saves (X, y, group) triples for regression analysis of bout parameters.

        This method orchestrates the preparation of behavioral and vocal predictors to model
        continuous bout features (e.g., duration, complexity).

        Key Logical & Scientific Configurations:
        1. Vocal feature configuration (vocal_output_type & vocal_output_partner_only):
           - If 'vocal_output_partner_only' is True: Only the partner's USV signals are ingested.
           - If 'vocal_output_partner_only' is False (Full syntax mode): Both mice's USV signals
             are ingested. This allows the model to determine if the subject's own prior vocal
             patterns (syntax) influence the parameters of the current bout.
        2. Scientific guardrail (self-predictor filtering):
           - When 'partner_only' is False, the script allows USV category/syntax traces from the
             subject (self) to be used as predictors.
           - However, it strictly excludes density-based signals (e.g., 'usv_rate',
             'usv_event') for the subject mouse. This prevents the model from trivially predicting
             bout duration based on the fact that the mouse is currently vocalizing.
        3. Generic renaming (self vs other):
           - Harmonizes egocentric (subject) and allocentric (partner) signals into a
             standardized 'self.*' and 'other.*' naming convention, regardless of the
             actual mouse IDs in the session.
        4. Target variable selection (target_variable):
           - Determines the continuous biological metric used as the regression target (y).
           - Supported values include:
             * 'bout_durations': The total time elapsed (seconds) from the first to the
               last syllable in a clustered sequence.
             * 'bout_syllable_counts': The discrete number of individual USVs contained
               within the bout (a proxy for vocal intensity).
             * 'mean_mask_complexity': The average number of distinct spectrotemporal
               components (masks) per syllable within the bout.
             * 'total_mask_complexity': The cumulative sum of all syllable masks across
               the bout (representing total information content).
        """

        target_variable = self.modeling_settings['model_params']['model_target_variable']
        print(f"--- Extracting Data for Regression Target: {target_variable} ---")

        txt_modeling_sessions = prepare_modeling_sessions(self.modeling_settings)

        gmm_idx = self.modeling_settings['model_params']['gmm_component_index']
        gmm_z = self.modeling_settings['model_params']['gmm_z_score']
        min_usv = self.modeling_settings['model_params']['usv_per_bout_floor']

        voc_type = self.modeling_settings['vocal_features']['usv_predictor_type']
        partner_only = self.modeling_settings['vocal_features']['usv_predictor_partner_only']
        smooth_sd = self.modeling_settings['vocal_features']['usv_predictor_smoothing_sd']
        noise_cats = self.modeling_settings['vocal_features']['usv_noise_categories']

        print("Loading behavioral feature data...")
        beh_feature_data_dict, camera_fr_dict, mouse_track_names_dict = load_behavioral_feature_data(
            behavior_file_paths=txt_modeling_sessions,
            csv_sep=self.modeling_settings['io']['csv_separator']
        )

        print(f"Identifying Bouts & Generating Vocal Signals (Type: {voc_type}, Partner Only: {partner_only})...")
        bout_data_dict = find_variable_length_bouts(
            root_directories=txt_modeling_sessions,
            mouse_ids_dict=mouse_track_names_dict,
            camera_fps_dict=camera_fr_dict,
            features_dict=beh_feature_data_dict,
            csv_sep=self.modeling_settings['io']['csv_separator'],
            gmm_component_index=gmm_idx,
            gmm_z_score=gmm_z,
            gmm_params=self.modeling_settings['gmm_params'],
            min_vocalizations=min_usv,
            filter_history=self.modeling_settings['model_params']['filter_history'],
            proportion_smoothing_sd=smooth_sd,
            vocal_output_type=voc_type,
            noise_vocal_categories=noise_cats
        )

        processed_beh_feature_data_dict = {}
        kin_settings = self.modeling_settings['kinematic_features']
        voc_settings = self.modeling_settings['vocal_features']
        predictor_mouse_idx = self.modeling_settings['model_params']['model_predictor_mouse_index']
        target_mouse_idx = abs(predictor_mouse_idx - 1)

        sessions_to_remove = identify_empty_event_sessions(
            usv_data_dict=bout_data_dict,
            mouse_names_dict=mouse_track_names_dict,
            target_idx=target_mouse_idx,
            event_key='bout_onsets',
            warn_label='session'
        )
        for sess in sessions_to_remove:
            if sess in beh_feature_data_dict:
                del beh_feature_data_dict[sess]

        print(f"Proceeding with {len(beh_feature_data_dict)} sessions after filtering empty ones.")

        for sess_id, session_df in beh_feature_data_dict.items():
            if sess_id not in mouse_track_names_dict:
                continue

            (predictor_mouse_idx,
             target_mouse_idx,
             p_name,
             t_name) = resolve_mouse_roles(
                modeling_settings=self.modeling_settings,
                mouse_names_dict=mouse_track_names_dict,
                session_id=sess_id
            )

            session_df_cols = session_df.columns

            columns_to_keep_session = select_kinematic_columns(
                session_df_columns=session_df_cols,
                target_name=t_name,
                predictor_name=p_name,
                kin_settings=kin_settings,
                predictor_idx=predictor_mouse_idx
            )

            new_voc_cols, new_voc_col_names = build_vocal_signal_columns(
                usv_data_dict=bout_data_dict,
                session_id=sess_id,
                target_name=t_name,
                predictor_name=p_name,
                voc_settings=voc_settings
            )

            columns_to_keep_session = sorted(set(columns_to_keep_session) | set(new_voc_col_names))
            existing_cols = [c for c in columns_to_keep_session if c in session_df_cols]
            current_df = session_df.select(existing_cols)
            if new_voc_cols:
                current_df = current_df.with_columns(new_voc_cols)

            processed_beh_feature_data_dict[sess_id] = current_df

        revised_behavioral_predictors = collect_predictor_suffixes(processed_beh_feature_data_dict)

        print("Standardizing columns ...")
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

        # Explicit lookup — feature_boundaries is an optional attribute set by
        # the parent __init__ when the setting is present in the JSON.
        if hasattr(self, 'feature_boundaries'):
            feature_bounds = self.feature_boundaries
        else:
            feature_bounds = {}

        processed_beh_feature_data_dict = zscore_features_across_sessions(
            processed_beh_dict=processed_beh_feature_data_dict,
            suffixes=revised_behavioral_predictors,
            feature_bounds=feature_bounds,
            abs_features=['allo_roll', 'allo_yaw-nose', 'nose-allo_yaw', 'allo_yaw-TTI', 'TTI-allo_yaw']
        )

        t_sex = 'male' if target_mouse_idx == 0 else 'female'
        fname = f"modeling_bout_param_{target_variable}_{t_sex}_gmm{gmm_idx}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_hist{self.modeling_settings['model_params']['filter_history']}.pkl"

        # Predictor diagnostics audit (collinearity + timescales). Diagnostic-
        # only: any failure inside the wrapper warns and continues.
        run_predictor_audits(
            processed_beh_dict=processed_beh_feature_data_dict,
            usv_data_dict=bout_data_dict,
            mouse_names_dict=mouse_track_names_dict,
            camera_fps_dict=camera_fr_dict,
            target_idx=target_mouse_idx,
            predictor_idx=predictor_mouse_idx,
            history_frames=self.history_frames,
            event_keys=['bout_onsets'],
            settings=self.modeling_settings,
            save_dir=self.modeling_settings['io']['save_directory'],
            pickle_basename=fname,
        )

        final_data_dict = {}
        for sess_id, df in tqdm(processed_beh_feature_data_dict.items(), desc="Extracting Epochs"):
            t_name = mouse_track_names_dict[sess_id][target_mouse_idx]
            p_name = mouse_track_names_dict[sess_id][predictor_mouse_idx]

            if t_name not in bout_data_dict[sess_id]: continue
            onsets = bout_data_dict[sess_id][t_name]['bout_onsets']
            targets = bout_data_dict[sess_id][t_name][target_variable]
            onset_frames = np.round(onsets * camera_fr_dict[sess_id]).astype(int)

            for col in df.columns:
                base_feature = col.split('.')[-1]
                if base_feature.isdigit(): continue

                if '-' in base_feature:
                    generic_key = base_feature

                elif col.startswith(f"{t_name}."):
                    generic_key = f"self.{base_feature}"
                elif col.startswith(f"{p_name}."):
                    generic_key = f"other.{base_feature}"
                else:
                    generic_key = base_feature

                if generic_key not in final_data_dict:
                    final_data_dict[generic_key] = {'X': [], 'y': [], 'groups': []}

                col_data = df[col].to_numpy()
                for i, frame_idx in enumerate(onset_frames):
                    start = frame_idx - self.history_frames
                    if start >= 0 and frame_idx <= len(col_data):
                        chunk = col_data[start:frame_idx].copy()
                        chunk[np.isnan(chunk)] = 0.0
                        final_data_dict[generic_key]['X'].append(chunk)
                        final_data_dict[generic_key]['y'].append(targets[i])
                        final_data_dict[generic_key]['groups'].append(sess_id)

        for feat in final_data_dict:
            for k in ['X', 'y', 'groups']:
                final_data_dict[feat][k] = np.array(final_data_dict[feat][k])

        final_features = sorted(list(final_data_dict.keys()))
        total_covariates = len(final_features)

        alignment_passed = True
        mismatched_features = []

        if total_covariates > 0:
            first_feat = final_features[0]
            ref_y = final_data_dict[first_feat]['y']
            ref_groups = final_data_dict[first_feat]['groups']

            for feat in final_features[1:]:
                # Check if total number of bouts matches
                if len(final_data_dict[feat]['y']) != len(ref_y):
                    alignment_passed = False
                    mismatched_features.append(feat)
                    continue

                # Check if the session groups are identical and in the same order
                if not np.array_equal(final_data_dict[feat]['groups'], ref_groups):
                    alignment_passed = False
                    mismatched_features.append(feat)

        if total_covariates > 0:
            total_n = len(final_data_dict[first_feat]['y'])
            unique_sess_count = len(np.unique(final_data_dict[first_feat]['groups']))
            hist_len = final_data_dict[first_feat]['X'].shape[1]
        else:
            total_n = unique_sess_count = hist_len = 0

        print("\n" + "=" * 105)
        print(f"REGRESSION DATA SUMMARY: {target_variable.upper()}")
        print("=" * 105)
        print(f"{'#':<4} {'Generic Feature Name':<45} | {'Bouts (N)':<10} | {'Sess Count':<10} | {'Status'}")
        print("-" * 105)

        for i, feat in enumerate(final_features, 1):
            feat_n = len(final_data_dict[feat]['y'])
            feat_sess = len(np.unique(final_data_dict[feat]['groups']))

            is_zero = np.all(final_data_dict[feat]['X'] == 0)
            status = "ZERO-FILLED" if is_zero else "DATA-PRESENT"

            guard = " [PROTECTED]" if feat.startswith("self.") and any(k in feat for k in ('usv_rate', 'usv_event')) else ""

            print(f"{i:3}. {feat:<45} | {feat_n:<10} | {feat_sess:<10} | {status}{guard}")

        print("-" * 105)
        print(f"PROJECT-WIDE REGRESSION TALLY:")
        print(f"  > Total Unique Covariates:      {total_covariates}")
        print(f"  > Total Sessions Included:      {unique_sess_count}")
        print(f"  > Total Bouts Across Project:   {total_n}")
        print(f"  > History Window Length:        {hist_len} frames")
        print(f"  > INTRA-SESSION ALIGNMENT:      {'PASSED (True)' if alignment_passed else 'FAILED (False)'}")

        if not alignment_passed:
            print(f"  [!] ALERT: Dimensional or Grouping mismatch in: {mismatched_features}")
            print(f"      (This will cause the model to misalign behavioral predictors with USV targets!)")
        print("=" * 105 + "\n")

        save_path = os.path.join(self.modeling_settings['io']['save_directory'], fname)
        os.makedirs(self.modeling_settings['io']['save_directory'], exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(final_data_dict, f)
        print(f"[+] Saved: {save_path}")

    def create_data_splits(self, feature_data: dict, strategy_override: str = None):
        """
        Generates stratified splits for continuous regression targets using a
        Repeated Monte Carlo approach.

        Strategies:
        1. 'mixed' (StratifiedShuffleSplit):
           - Ideal for pooling data.
           - Guarantees that the Test set has the same distribution of 'y' as the Train set.
           - Uses standard Monte Carlo resampling.

        2. 'session' (Repeated StratifiedGroupKFold):
           - Preserves session boundaries (Groups).
           - Balances 'y' distribution across Train/Test (Stratification).
           - Since 'StratifiedGroupShuffleSplit' does not exist in sklearn, we approximate
             it by re-initializing a StratifiedGroupKFold with a new random seed
             for every iteration and taking a single fold.

        Parameters
        ----------
        feature_data : dict
            Contains 'X', 'y', 'groups'.
        strategy_override : str, optional
            Force 'session' or 'mixed' strategy.

        Yields
        ------
        tuple
            (X_train, y_train, X_test, y_test)
        """

        X = feature_data['X']
        y = feature_data['y']
        groups = feature_data['groups']
        n_samples = len(y)

        # 1. Configuration
        model_selection = self.modeling_settings['model_params']
        split_strategy = strategy_override or model_selection['split_strategy']

        num_iterations = model_selection['split_num']
        test_prop = model_selection['test_proportion']
        base_seed = self.modeling_settings['model_params']['random_seed']
        if base_seed is None:
            base_seed = 42

        # 2. Safety checks
        if test_prop <= 0 or test_prop >= 1:
            raise ValueError(f"test_proportion must be (0, 1). Got {test_prop}")

        # 3. Stratification binning (discretize continuous Y)
        try:
            n_bins = max(2, min(10, n_samples // 20))
            bins = np.percentile(y, np.linspace(0, 100, n_bins + 1))
            y_binned = np.digitize(y, bins[1:-1])
        except Exception:
            y_binned = np.zeros(len(y))

        # 4. Splitting Execution
        if split_strategy == 'mixed':

            sss = StratifiedShuffleSplit(
                n_splits=num_iterations,
                test_size=test_prop,
                random_state=base_seed
            )

            for train_idx, test_idx in sss.split(X, y_binned):
                yield X[train_idx], y[train_idx], X[test_idx], y[test_idx]

        elif split_strategy == 'session':

            n_folds = int(np.floor(1.0 / test_prop))
            n_folds = max(2, n_folds)

            for i in range(num_iterations):
                current_seed = base_seed + i

                # We re-initialize the splitter every iteration with a new seed.
                sgkf = StratifiedGroupKFold(
                    n_splits=n_folds,
                    shuffle=True,
                    random_state=current_seed
                )

                # Take the first valid split for this seed
                try:
                    train_idx, test_idx = next(sgkf.split(X, y_binned, groups=groups))
                    yield X[train_idx], y[train_idx], X[test_idx], y[test_idx]
                except StopIteration:
                    continue
        else:
            raise ValueError(f"Unknown split strategy: {split_strategy}")

    def _run_model_for_feature_pygam(self, feature_name, feature_data, basis_matrix):
        r"""
        Runs a univariate GammaGAM regression and permutation test for a single feature.

        This method evaluates the predictive relationship between a behavioral/vocal feature
        and a continuous bout parameter (duration or complexity). It utilizes a Generalized
        Additive Model (GAM) tailored for biological timing data, which is typically
        strictly positive and right-skewed.

        Statistical Framework:
        1. Gamma distribution & log link: Fits the model assuming y ~ Gamma. The log-link
           function (ln(mu) = X/beta) ensures that predicted bout parameters remain
           strictly positive ($exp(X/beta)) and accounts for the heteroscedasticity
           where variance increases with the mean.
        2. Permutation test (shuffled control): To establish a baseline for null
           predictability, the method executes a parallel "shuffled" analysis. In this
           branch, the target labels (y) are randomly permuted while preserving the
           temporal structure of the features (X).
           - Actual model: Captures the true relationship between behavior and bout metrics.
           - Shuffled model: Captures "chance" performance and overfitting tendencies
             given the specific distribution and sample size of the dataset.

        Computational Optimization:
        - Matrix reuse: The high-dimensional "unrolled" feature matrices (X_tr and X_te)
          are calculated once per fold. Both the actual and shuffled models are fit using
          these shared matrices to minimize CPU tiling overhead and memory fragmentation.
        - Memory management: Employs float32 precision for unrolled tensors and explicit
          garbage collection after each model fit to maintain stability on HPC nodes.

        Metrics Calculated:
        - `explained_deviance` (D^2): `1 - (Residual Deviance / Null Deviance)`.
          Proportion of deviance explained by the model; the Gamma-equivalent of
          R^2. Higher is better; bounded above by 1 for well-specified models.
        - `residual_deviance`: raw Gamma deviance on the test set. Interpretable
          in the native Gamma-GLM scoring scale (lower is better).
        - `spearman_r`: rank-correlation between predicted and true values;
          robust to outliers and insensitive to the magnitude of the log-link.
        - `pearson_r`: linear-scale correlation on the back-transformed
          predictions; complements Spearman by detecting compression/expansion
          of magnitudes.
        - `msle` (Mean Squared Logarithmic Error): penalizes relative errors
          equally across scales — aligned with the log-link assumption.
        - `mae` (Mean Absolute Error): interpretable magnitude of per-trial
          miss in native units (e.g., seconds). Robust to outlier trials.
        - `rmse` (Root Mean Squared Error): heavy-tail sensitive counterpart
          to MAE; an `RMSE / MAE` ratio well above `sqrt(pi / 2) ≈ 1.25`
          flags outlier-dominated folds.
        - `n_iter`, `converged`, `fit_time`: per-fold optimizer diagnostics.
          `converged=False` surfaces folds that hit `max_iter` before the
          tolerance check fired.

        Parameters
        ----------
        feature_name : str
            Name of the feature being analyzed (e.g., 'self.speed').
        feature_data : dict
            Dictionary containing 'X' (history windows), 'y' (bout targets), and
            'groups' (session IDs).
        basis_matrix : np.ndarray | None
            Required for API parity with the Sklearn engine; however, PyGAM
            generates its own internal tensor product splines (te).

        Returns
        -------
        tuple[str, dict]
            A tuple of (feature_name, results_dict) where results contains 'actual'
            and 'null' metrics across all cross-validation splits (the 'null'
            branch holds the permutation control — same estimator re-fit on a
            shuffled target).
        """

        print(f"--- Running [GammaGAM] for: {feature_name} ---")

        hist_frames = self.history_frames

        # Metric bundle saved per split for each strategy. The 'null' key holds
        # the label-shuffled control (renamed from the legacy 'shuffled' to
        # match the vocabulary used by the classifier / manifold pipelines).
        def _make_bout_branch():
            return {
                'explained_deviance': [], 'spearman_r': [], 'pearson_r': [],
                'msle': [], 'mae': [], 'rmse': [], 'residual_deviance': [],
                'filter_shapes': [],
                'n_iter': [], 'converged': [], 'fit_time': []
            }

        results = {
            'actual': _make_bout_branch(),
            'null': _make_bout_branch()
        }

        # Strict dictionary lookups (No .get())
        pygam_params = self.modeling_settings['hyperparameters']['classical']['pygam']
        n_val = pygam_params['n_splines_value']
        n_time = pygam_params['n_splines_time']
        lam = pygam_params['lam_penalty']
        tol_val = pygam_params['tol_val']
        max_iterations = pygam_params['max_iterations']

        time_indices = np.arange(hist_frames)

        splitter = self.create_data_splits(feature_data)

        # Seed for per-split null-label permutation Generators so the shuffled
        # control is reproducible and does not inherit ambient global NumPy
        # RNG state from prior calls.
        base_seed = self.modeling_settings['model_params']['random_seed']

        for split_idx, (X_tr, y_tr, X_te, y_te) in enumerate(splitter):
            print(f"  > Processing Split {split_idx + 1}...")

            X_tr_gam = unroll_history_matrix(X_tr, time_indices=time_indices).astype(np.float32)
            X_te_gam = unroll_history_matrix(X_te, time_indices=time_indices).astype(np.float32)

            try:
                y_tr_tiled = np.repeat(y_tr + 1e-6, hist_frames).astype(np.float32)

                gam = GAM(
                    te(0, 1, n_splines=[n_val, n_time]),
                    distribution='gamma',
                    link='log',
                    max_iter=max_iterations,
                    tol=tol_val,
                    lam=lam
                )
                fit_start = time.perf_counter()
                gam.fit(X_tr_gam, y_tr_tiled)
                fit_time = float(time.perf_counter() - fit_start)

                gam_diffs = gam.logs_.get('diffs', [])
                n_iter_actual = int(len(gam_diffs))
                converged_actual = bool(gam_diffs and gam_diffs[-1] < tol_val)

                # Prediction
                y_pred_tiled = gam.predict(X_te_gam)
                y_pred = np.mean(y_pred_tiled.reshape(len(y_te), hist_frames), axis=1)

                try:
                    # Sanitize inputs (Gamma crashes on 0.0)
                    y_te_safe = np.maximum(y_te, 1e-6)
                    y_pred_safe = np.maximum(y_pred, 1e-6)
                    y_null_pred = np.full_like(y_te_safe, np.mean(y_te_safe))

                    # Calculate deviances
                    res_dev = mean_gamma_deviance(y_te_safe, y_pred_safe)
                    null_dev = mean_gamma_deviance(y_te_safe, y_null_pred)

                    # Calculate D2
                    if null_dev == 0:
                        d2 = 0.0
                    else:
                        d2 = 1 - (res_dev / null_dev)

                except Exception as e:
                    print(f"      [!] Actual Metric Failed: {e}")
                    d2 = np.nan
                    res_dev = np.nan

                # Store results
                results['actual']['explained_deviance'].append(d2)
                results['actual']['residual_deviance'].append(res_dev)
                results['actual']['spearman_r'].append(spearmanr(y_te, y_pred)[0])
                results['actual']['pearson_r'].append(pearson_r_safe(y_te, y_pred))
                results['actual']['msle'].append(mean_squared_log_error(y_te, y_pred))
                results['actual']['mae'].append(mean_absolute_error_1d(y_te, y_pred))
                results['actual']['rmse'].append(root_mean_squared_error(y_te, y_pred))
                results['actual']['n_iter'].append(n_iter_actual)
                results['actual']['converged'].append(converged_actual)
                results['actual']['fit_time'].append(fit_time)

                # ExtractsShapes
                grid_0 = np.stack([np.zeros(hist_frames), time_indices], axis=1)
                grid_1 = np.stack([np.ones(hist_frames), time_indices], axis=1)
                shape = (gam.predict(grid_1) - gam.predict(grid_0)).flatten()
                results['actual']['filter_shapes'].append(shape)

                del gam, y_tr_tiled
                gc.collect()

            except Exception as e:
                print(f"Fit failed for Actual Split {split_idx + 1}: {e}")
                # Keep every per-fold list aligned on failure so downstream
                # consumers can rely on matching lengths across keys. Fill
                # `filter_shapes` with a NaN vector of the correct width
                # rather than a scalar so the list can later stack cleanly.
                results['actual']['explained_deviance'].append(np.nan)
                results['actual']['residual_deviance'].append(np.nan)
                results['actual']['spearman_r'].append(np.nan)
                results['actual']['pearson_r'].append(np.nan)
                results['actual']['msle'].append(np.nan)
                results['actual']['mae'].append(np.nan)
                results['actual']['rmse'].append(np.nan)
                results['actual']['filter_shapes'].append(np.full(hist_frames, np.nan))
                results['actual']['n_iter'].append(np.nan)
                results['actual']['converged'].append(False)
                results['actual']['fit_time'].append(np.nan)

            try:
                null_rng = np.random.default_rng(base_seed + split_idx + 1)
                y_tr_shuff_tiled = np.repeat(null_rng.permutation(y_tr) + 1e-6, hist_frames).astype(np.float32)

                gam_null = GAM(
                    te(0, 1, n_splines=[n_val, n_time]),
                    distribution='gamma',
                    link='log',
                    max_iter=max_iterations,
                    tol=tol_val,
                    lam=lam
                )

                fit_start = time.perf_counter()
                gam_null.fit(X_tr_gam, y_tr_shuff_tiled)
                fit_time_null = float(time.perf_counter() - fit_start)

                null_diffs = gam_null.logs_.get('diffs', [])
                n_iter_null = int(len(null_diffs))
                converged_null = bool(null_diffs and null_diffs[-1] < tol_val)

                y_pred_shuff_tiled = gam_null.predict(X_te_gam)
                y_pred_shuff = np.mean(y_pred_shuff_tiled.reshape(len(y_te), hist_frames), axis=1)

                # NOTE: We compare the null-model prediction against the *REAL* y_te
                try:
                    # Sanitize inputs
                    y_te_safe = np.maximum(y_te, 1e-6)
                    y_pred_shuff_safe = np.maximum(y_pred_shuff, 1e-6)
                    y_null_pred = np.full_like(y_te_safe, np.mean(y_te_safe))

                    # Calculate deviances
                    res_dev_null = mean_gamma_deviance(y_te_safe, y_pred_shuff_safe)
                    null_dev = mean_gamma_deviance(y_te_safe, y_null_pred)

                    # Calculate D2
                    if null_dev == 0:
                        d2_null = 0.0
                    else:
                        d2_null = 1 - (res_dev_null / null_dev)

                except Exception as e:
                    print(f"      [!] Null Metric Failed: {e}")
                    d2_null = np.nan
                    res_dev_null = np.nan

                results['null']['explained_deviance'].append(d2_null)
                results['null']['residual_deviance'].append(res_dev_null)
                results['null']['spearman_r'].append(spearmanr(y_te, y_pred_shuff)[0])
                results['null']['pearson_r'].append(pearson_r_safe(y_te, y_pred_shuff))
                results['null']['msle'].append(mean_squared_log_error(y_te, y_pred_shuff))
                results['null']['mae'].append(mean_absolute_error_1d(y_te, y_pred_shuff))
                results['null']['rmse'].append(root_mean_squared_error(y_te, y_pred_shuff))
                results['null']['n_iter'].append(n_iter_null)
                results['null']['converged'].append(converged_null)
                results['null']['fit_time'].append(fit_time_null)

                del gam_null, y_tr_shuff_tiled
                gc.collect()

            except Exception as e:
                print(f"Fit failed for Null Split {split_idx + 1}: {e}")
                # The pyGAM null branch never writes `filter_shapes`, so that
                # list stays empty by design; every other per-fold key gets a
                # NaN placeholder to keep lengths aligned across folds.
                results['null']['explained_deviance'].append(np.nan)
                results['null']['residual_deviance'].append(np.nan)
                results['null']['spearman_r'].append(np.nan)
                results['null']['pearson_r'].append(np.nan)
                results['null']['msle'].append(np.nan)
                results['null']['mae'].append(np.nan)
                results['null']['rmse'].append(np.nan)
                results['null']['n_iter'].append(np.nan)
                results['null']['converged'].append(False)
                results['null']['fit_time'].append(np.nan)

            del X_tr_gam, X_te_gam
            gc.collect()

        # Safely convert lists to arrays
        for k in results:
            for m in results[k]:
                results[k][m] = np.array(results[k][m])

        return feature_name, results

    def _run_model_for_feature_sklearn(self, feature_name, feature_data, basis_matrix):
        r"""
        Runs a univariate RidgeCV regression on log-transformed bout parameters.

        This method provides a linear alternative to the GAM approach. It models the relationship
        between behavioral history and bout parameters by transforming the strictly positive,
        skewed target variable into a Gaussian-like distribution in log-space.

        Statistical and mathematical framework:
        1. Log-transformation: The target variable (y) is transformed via ln(y + epsilon).
           Since bout duration and complexity are strictly positive and typically log-normally
           distributed, this mapping satisfies the homoscedasticity and normality assumptions
           of linear regression.
        2. Basis projection: The high-dimensional feature history (X) is projected onto
           the provided `basis_matrix` (e.g., raised cosine or B-splines). This reduces the
           parameter space from N_lags to N_bases, preventing overfitting and
           ensuring the resulting temporal filters (kernels) are smooth and interpretable.
        3. L2 regularization (RidgeCV): Fits the model ln(y) = Phi\beta + \alpha||\beta||_2,
           where Phi is the basis-projected feature matrix. The optimal regularization
           strength (alpha) is determined via internal leave-one-out cross-validation (LOOCV).

        Evaluation and permutation control:
        - Original scale metrics: Predictions are back-transformed via exp(\hat{y}_{log}).
          Performance is evaluated using Spearman's Rank Correlation (to assess monotonic
          ranking accuracy independent of scale) and Mean Squared Logarithmic Error
          (MSLE, to assess magnitude accuracy in the log-space relevant to the distribution),
          alongside R-squared (R^2).
        - Permutation test: Parallel to the actual model, a "shuffled" control is executed
          by permuting the target labels. This provides a null distribution for these
          metrics to verify that the behavioral history contains more predictive information
          than random chance.

        Parameters
        ----------
        feature_name : str
            Name of the feature being analyzed (e.g., 'other.distance').
        feature_data : dict
            Dictionary containing 'X' (predictor epochs), 'y' (bout targets), and
            'groups' (session IDs for validation).
        basis_matrix : np.ndarray
            The [lags x bases] matrix used to transform the raw history into the
            reduced feature space.

        Returns
        -------
        tuple[str, dict]
            A tuple of (feature_name, results_dict) containing cross-validated performance
            metrics, optimized hyperparameters, and reconstructed filter shapes for both
            the 'actual' model and the 'null' (label-permuted) control.
        """

        print(f"--- Running [RidgeCV] for: {feature_name} ---")

        def _make_ridge_branch():
            return {
                'explained_deviance': [], 'spearman_r': [], 'pearson_r': [],
                'msle': [], 'mae': [], 'rmse': [], 'residual_deviance': [],
                'filter_shapes': [],
                'n_iter': [], 'converged': [], 'fit_time': []
            }

        results = {
            'actual': _make_ridge_branch(),
            'null': _make_ridge_branch()
        }

        splitter = self.create_data_splits(feature_data)

        # Strict dictionary lookups
        ridge_params = self.modeling_settings['hyperparameters']['classical']['ridge_regression']
        alphas = ridge_params['alphas']
        cv = ridge_params['cv']

        # Seed for per-split null-label permutation Generators so the shuffled
        # control is reproducible and independent of ambient global RNG state.
        base_seed = self.modeling_settings['model_params']['random_seed']

        for split_idx, (X_tr, y_tr, X_te, y_te) in enumerate(splitter):
            print(f"  > Processing Split {split_idx + 1}...")

            X_tr_proj = np.dot(X_tr, basis_matrix)
            X_te_proj = np.dot(X_te, basis_matrix)

            try:
                # Log-transform target for ridge (homoscedasticity assumption)
                y_tr_log = np.log(y_tr + 1e-6)

                fit_start = time.perf_counter()
                model = RidgeCV(alphas=alphas, cv=cv).fit(X_tr_proj, y_tr_log)
                fit_time = float(time.perf_counter() - fit_start)

                # Predict and back-transform
                y_pred_log = model.predict(X_te_proj)
                y_pred = np.exp(y_pred_log)

                try:
                    # Sanitize inputs (Gamma metrics crash on 0.0)
                    y_te_safe = np.maximum(y_te, 1e-6)
                    y_pred_safe = np.maximum(y_pred, 1e-6)
                    y_null_pred = np.full_like(y_te_safe, np.mean(y_te_safe))

                    # Calculate deviances
                    res_dev = mean_gamma_deviance(y_te_safe, y_pred_safe)
                    null_dev = mean_gamma_deviance(y_te_safe, y_null_pred)

                    # Calculate Explained Deviance (D2)
                    if null_dev == 0:
                        d2 = 0.0
                    else:
                        d2 = 1 - (res_dev / null_dev)

                except Exception as e:
                    print(f"      [!] Actual Metric Failed: {e}")
                    d2 = np.nan
                    res_dev = np.nan

                # Store results
                results['actual']['explained_deviance'].append(d2)
                results['actual']['residual_deviance'].append(res_dev)
                results['actual']['spearman_r'].append(spearmanr(y_te, y_pred)[0])
                results['actual']['pearson_r'].append(pearson_r_safe(y_te, y_pred))
                results['actual']['msle'].append(mean_squared_log_error(y_te, y_pred))
                results['actual']['mae'].append(mean_absolute_error_1d(y_te, y_pred))
                results['actual']['rmse'].append(root_mean_squared_error(y_te, y_pred))
                # RidgeCV has no iterative optimizer to report; store NaN for
                # `n_iter` and `True` for converged (closed-form solve).
                results['actual']['n_iter'].append(np.nan)
                results['actual']['converged'].append(True)
                results['actual']['fit_time'].append(fit_time)

                # Store filter shape (in log domain)
                shape_log = np.dot(model.coef_, basis_matrix.T)
                results['actual']['filter_shapes'].append(shape_log)

            except Exception as e:
                print(f"Ridge Fit failed for Actual Split {split_idx + 1}: {e}")
                results['actual']['explained_deviance'].append(np.nan)
                results['actual']['residual_deviance'].append(np.nan)
                results['actual']['spearman_r'].append(np.nan)
                results['actual']['pearson_r'].append(np.nan)
                results['actual']['msle'].append(np.nan)
                results['actual']['mae'].append(np.nan)
                results['actual']['rmse'].append(np.nan)
                # The back-projected filter is `history_frames` wide (coef @
                # basis_matrix.T); use a NaN vector of the same width so the
                # list stays stackable across folds.
                results['actual']['filter_shapes'].append(np.full(self.history_frames, np.nan))
                results['actual']['n_iter'].append(np.nan)
                results['actual']['converged'].append(False)
                results['actual']['fit_time'].append(np.nan)

            try:
                null_rng = np.random.default_rng(base_seed + split_idx + 1)
                y_tr_shuff_log = np.log(null_rng.permutation(y_tr) + 1e-6)

                fit_start = time.perf_counter()
                model_null = RidgeCV(alphas=alphas, cv=cv).fit(X_tr_proj, y_tr_shuff_log)
                fit_time_null = float(time.perf_counter() - fit_start)

                y_pred_null_log = model_null.predict(X_te_proj)
                y_pred_null = np.exp(y_pred_null_log)

                try:
                    # Sanitize inputs
                    y_te_safe = np.maximum(y_te, 1e-6)
                    y_pred_shuff_safe = np.maximum(y_pred_null, 1e-6)
                    y_null_pred = np.full_like(y_te_safe, np.mean(y_te_safe))

                    # Calculate deviances
                    res_dev_null = mean_gamma_deviance(y_te_safe, y_pred_shuff_safe)
                    null_dev = mean_gamma_deviance(y_te_safe, y_null_pred)

                    # Calculate explained deviance (D2)
                    if null_dev == 0:
                        d2_null = 0.0
                    else:
                        d2_null = 1 - (res_dev_null / null_dev)

                except Exception as e:
                    print(f"      [!] Null Metric Failed: {e}")
                    d2_null = np.nan
                    res_dev_null = np.nan

                results['null']['explained_deviance'].append(d2_null)
                results['null']['residual_deviance'].append(res_dev_null)
                results['null']['spearman_r'].append(spearmanr(y_te, y_pred_null)[0])
                results['null']['pearson_r'].append(pearson_r_safe(y_te, y_pred_null))
                results['null']['msle'].append(mean_squared_log_error(y_te, y_pred_null))
                results['null']['mae'].append(mean_absolute_error_1d(y_te, y_pred_null))
                results['null']['rmse'].append(root_mean_squared_error(y_te, y_pred_null))
                results['null']['n_iter'].append(np.nan)
                results['null']['converged'].append(True)
                results['null']['fit_time'].append(fit_time_null)

            except (ValueError, ZeroDivisionError, Exception) as e:
                print(f"Fit failed for Null Split {split_idx + 1}: {e}")
                # Ridge null branch does not emit filter shapes; every other
                # per-fold key gets a NaN placeholder to keep lengths aligned.
                results['null']['explained_deviance'].append(np.nan)
                results['null']['residual_deviance'].append(np.nan)
                results['null']['spearman_r'].append(np.nan)
                results['null']['pearson_r'].append(np.nan)
                results['null']['msle'].append(np.nan)
                results['null']['mae'].append(np.nan)
                results['null']['rmse'].append(np.nan)
                results['null']['n_iter'].append(np.nan)
                results['null']['converged'].append(False)
                results['null']['fit_time'].append(np.nan)

        # Safely convert lists to arrays
        for k in results:
            for m in results[k]:
                results[k][m] = np.array(results[k][m])

        return feature_name, results
