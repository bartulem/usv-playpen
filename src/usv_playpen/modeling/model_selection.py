"""
@author: bartulem
Forward stepwise model selection for predicting bout and USV dynamics.
"""

import os
import numpy as np
import pickle
import json
import pathlib
import re
import gc
import time
from pygam import LogisticGAM, GAM, te
from scipy.stats import spearmanr, wilcoxon
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit, ShuffleSplit
from sklearn.metrics import (log_loss, roc_auc_score, f1_score, recall_score,
                             accuracy_score, balanced_accuracy_score, mean_squared_log_error,
                             mean_gamma_deviance, precision_recall_curve, auc, brier_score_loss)
from .load_input_files import load_pickle_modeling_data
from .modeling_bases_functions import _normalizecols, bsplines, identity, laplacian_pyramid, raised_cosine
from .modeling_utils import (
    pool_session_arrays,
    brier_score_multi,
    expected_calibration_error,
    safe_matthews_corrcoef,
    safe_confusion_matrix,
    pearson_r_safe,
    root_mean_squared_error,
    mean_absolute_error_1d,
)
from .modeling_vocal_onsets import VocalOnsetModelingPipeline
from .modeling_vocal_categories_multinomial import (
    get_stratified_group_splits_stable,
    _balance_multinomial_train_indices,
    MultinomialModelingPipeline,
    MultinomialModelRunner,
)
from .modeling_usv_manifold_position import (
    get_stratified_spatial_splits_stable,
    _log_spaced_grid,
    _tune_manifold_regularization,
)
from .jax_multinomial_logistic_regression import SmoothMultinomialLogisticRegression
from .jax_bivariate_regression import SmoothBivariateRegression


def get_unrolled_X_for_multivariate(feature_data_dict_list: list = None,
                                    history_frames: int = None) -> np.ndarray:
    """
    Prepares the 'unrolled' X matrix for a multivariate pyGAM by stacking
    multiple features side-by-side in the specific (Value, Time) format required
    for tensor product splines.

    For a single feature, pyGAM's `te(feature, time)` term expects an input with
    two columns: [Feature_Value, Time_Index].

    For a multivariate model with N features (e.g., `te(A, time) + te(B, time)`),
    we need to construct a matrix where each feature gets its own pair of columns.
    This function transforms a list of (n_samples, history_frames) matrices into
    a single tall, wide matrix suitable for fitting.

    Parameters
    ----------
    feature_data_dict_list : list of np.ndarray
        A list of numpy arrays, where each array contains the history data for
        one feature.
        - Each array must have shape (n_samples, history_frames).
        - The order of arrays in this list determines the column order in the output.
        - Example: [X_nose_nose, X_speed, X_yaw]
    history_frames : int
        The number of time lags (columns) in each input array. Used for validation
        and generating the time index column.

    Returns
    -------
    X_unrolled : np.ndarray
        A 2D numpy array of shape (n_samples * history_frames, 2 * n_features).

        The columns are organized as pairs for each feature:
        - Col 0: Feature 1 Value
        - Col 1: Feature 1 Time Index (0, 1, ... history_frames-1)
        - Col 2: Feature 2 Value
        - Col 3: Feature 2 Time Index
        - ... and so on.

        This format allows constructing a GAM model with terms like:
        `te(0, 1) + te(2, 3) + ...`
    """

    if not feature_data_dict_list:
        raise ValueError("feature_data_dict_list is empty.")

    n_samples, n_frames = feature_data_dict_list[0].shape
    if n_frames != history_frames:
        raise ValueError(f"Frame mismatch: Data has {n_frames}, expected {history_frames}")

    n_features = len(feature_data_dict_list)
    n_total_rows = n_samples * n_frames

    X_unrolled = np.empty((n_total_rows, 2 * n_features), dtype=np.float32)

    time_indices = np.arange(history_frames, dtype=np.float32)
    tiled_time = np.tile(time_indices, n_samples)

    for i, X_feat in enumerate(feature_data_dict_list):
        if X_feat.shape[0] != n_samples:
            raise ValueError(f"Sample count mismatch at feature index {i}. Expected {n_samples}, got {X_feat.shape[0]}")

        col_val_idx = i * 2
        col_time_idx = i * 2 + 1

        X_unrolled[:, col_val_idx] = X_feat.ravel()
        X_unrolled[:, col_time_idx] = tiled_time

    return X_unrolled


def bout_onset_model_selection(univariate_results_path: str,
                               input_data_path: str,
                               output_directory: str,
                               settings_path: str = None,
                               use_top_rank_as_anchor: bool = False,
                               p_val: float = 0.01) -> None:
    """
    Performs Forward Stepwise Selection to identify the "minimal sufficient set"
    of behavioral features that predict vocal (bout) onset.

    This function implements a rigorous, greedy search strategy using Generalized
    Additive Models (GAMs) with tensor product interactions. To ensure memory stability,
    filter shapes are NOT calculated during the selection loop. Instead, a single
    refit is performed on the final winning model to extract shapes for visualization,
    which are then appended to the results file of the last *successful* step.

    Splitting & balancing invariant:
    --------------------------------
    Across both 'session' and 'mixed' strategies, the training fold is down-sampled
    to a 50/50 class balance (Bout vs. No-Bout), while the test fold preserves the
    natural class prior of the source data. The 'mixed' strategy achieves this by
    running `StratifiedShuffleSplit` on the *unbalanced* pool and balancing the
    training half inside the per-fold loop. Reported metrics should therefore be
    imbalance-robust: the `score` field stores `balanced_accuracy`, paired with
    AUC and log-loss.

    Crucially, this version preserves full-resolution metric data. For every
    candidate tested at every step, the raw per-fold results are saved for:
    Log-Likelihood (LL), AUC, Recall, F1, Balanced Accuracy (`score`), Brier
    score, Expected Calibration Error (ECE), Matthews correlation coefficient
    (MCC), and the (2, 2) confusion matrix. Per-fold optimizer diagnostics —
    `n_iter`, `converged`, `fit_time` — are stored alongside so folds that
    terminated at `max_iter` without meeting the tolerance can be audited
    after the fact. Precision is intentionally not stored: it is recoverable
    from the saved confusion matrices and macro-F1 already summarizes the
    precision / recall trade-off. This allows plotting scripts to
    reconstruct model selection trajectories with accurate error bars and
    individual fold data points.

    Stability features:
    - Safe cleanup: Prevents `UnboundLocalError` during garbage collection if a
      fold fails early.
    - Crash reporting: Logs detailed error messages for failed fits.
    - Resource management: Explicit garbage collection triggers to prevent
      memory fragmentation.
    """

    print("--- Starting Model Selection ---")
    chance_ll = np.log(2)

    if settings_path is None:
        settings_path_obj = pathlib.Path(__file__).resolve().parent.parent / '_parameter_settings/modeling_settings.json'
        settings_path = str(settings_path_obj)

    try:
        with open(settings_path, 'r') as f:
            settings = json.load(f)
        print(f"Loaded settings from: {settings_path}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Settings file not found at {settings_path}")

    model_selection_dir = output_directory
    os.makedirs(model_selection_dir, exist_ok=True)

    print(f"Loading univariate results from: {univariate_results_path}")
    print(f"Loading raw data from: {input_data_path}")
    with open(univariate_results_path, 'rb') as f:
        univariate_data = pickle.load(f)

    candidates = []
    for feat_name, results in univariate_data.items():
        if 'actual' not in results or 'll' not in results['actual']:
            continue
        actual_ll = results['actual']['ll']
        null_ll = results['null']['ll']
        valid_actual = actual_ll[~np.isnan(actual_ll)]
        valid_null = null_ll[~np.isnan(null_ll)]
        if len(valid_actual) == 0 or len(valid_null) == 0:
            continue
        mean_actual_ll = np.mean(valid_actual)
        null_threshold = np.percentile(valid_null, q=(p_val / len(univariate_data)) * 100)
        if mean_actual_ll < null_threshold:
            candidates.append({'feature': feat_name, 'mean_ll': mean_actual_ll})

    candidates.sort(key=lambda x: x['mean_ll'])
    ranked_features = [x['feature'] for x in candidates]
    if not ranked_features:
        print("No significant features found. Aborting.")
        return

    all_feature_data = load_pickle_modeling_data(input_data_path)
    pipeline = VocalOnsetModelingPipeline(modeling_settings_dict=settings)
    if not hasattr(pipeline, 'history_frames'):
        pipeline.history_frames = int(np.floor(settings['io']['camera_sampling_rate'] * settings['model_params']['filter_history']))
    history_frames = pipeline.history_frames

    common_sessions = set(all_feature_data[ranked_features[0]].keys())
    for feat in ranked_features[1:]:
        common_sessions = common_sessions.intersection(set(all_feature_data[feat].keys()))
    all_sessions = sorted(list(common_sessions))

    pygam_params = settings['hyperparameters']['classical']['pygam']
    n_splines_time = pygam_params['n_splines_time']
    n_splines_value = pygam_params['n_splines_value']
    lam_penalty = pygam_params['lam_penalty']

    gam_kwargs = {
        'max_iter': int(pygam_params['max_iterations']),
        'tol': float(pygam_params['tol_val']),
        'lam': lam_penalty
    }

    model_ops = settings['model_params']
    split_strategy = model_ops['split_strategy']
    n_splits_selection = model_ops['split_num']
    test_prop = model_ops['test_proportion']

    random_seed = settings['model_params']['random_seed']
    print(f"Random Seed: {random_seed} | Split Strategy: {split_strategy} | Num Splits: {n_splits_selection}")
    anchor_feature = ranked_features[0]

    cv_folds = []
    if split_strategy == 'session':
        all_sessions_arr = np.array(all_sessions)
        ss = ShuffleSplit(n_splits=n_splits_selection, test_size=max(test_prop, 1.0 / len(all_sessions)), random_state=random_seed)
        for train_idx, test_idx in ss.split(all_sessions_arr):
            cv_folds.append({'train_sessions': all_sessions_arr[train_idx], 'test_sessions': all_sessions_arr[test_idx], 'type': 'session'})
    elif split_strategy == 'mixed':
        # Stratified split over the *unbalanced* pool: preserves the natural class
        # ratio in both train and test indices. Per-fold, training will be balanced
        # 50/50 inside the loop; test retains the natural class prior.
        X_p_all, X_n_all = pool_session_arrays(all_feature_data[anchor_feature], all_sessions, pos_key="usv_feature_arr", neg_key="no_usv_feature_arr", n_frames=pipeline.history_frames)
        n_pos_total = X_p_all.shape[0]
        n_neg_total = X_n_all.shape[0]
        y_full = np.concatenate((np.ones(n_pos_total), np.zeros(n_neg_total)))
        sss = StratifiedShuffleSplit(n_splits=n_splits_selection, test_size=test_prop, random_state=random_seed)
        for train_ix, test_ix in sss.split(np.zeros(len(y_full)), y_full):
            cv_folds.append({'train_idx': train_ix, 'test_idx': test_ix, 'n_pos_total': n_pos_total, 'n_neg_total': n_neg_total, 'type': 'mixed'})
    else:
        raise ValueError(
            f"Unknown split_strategy '{split_strategy}' for bout-onset model selection. "
            f"Supported strategies: 'session', 'mixed'."
        )

    # Pre-pool each candidate feature's full (positive, negative) design matrix
    # once up front. The forward-selection loop re-evaluates every feature for
    # every fold and every accepted step; without this cache the inner body
    # would otherwise call `pool_session_arrays` O(n_steps * n_features * n_folds)
    # times, re-concatenating the same per-session arrays at every iteration.
    pooled_feature_cache = {}
    for _feat_name in ranked_features:
        _X_p, _X_n = pool_session_arrays(
            all_feature_data[_feat_name],
            all_sessions,
            pos_key="usv_feature_arr",
            neg_key="no_usv_feature_arr",
            n_frames=pipeline.history_frames,
        )
        pooled_feature_cache[_feat_name] = {
            'X_pos': _X_p,
            'X_neg': _X_n,
            'X_full': np.concatenate((_X_p, _X_n), axis=0),
        }

    current_model_features = []
    best_current_score = chance_ll
    best_current_se = 0.0
    time_indices = np.arange(history_frames, dtype=float)
    step_counter = 0

    fname = os.path.basename(univariate_results_path)
    cond_match = re.search(r'((?:male|female).*?)(?=_splits|_lam|_gmm|\.pkl)', fname)
    target_condition = cond_match.group(1) if cond_match else "unknown"
    prediction_mode = settings['model_params']['model_target_vocal_type']

    prefix = f"model_selection_{target_condition}_{prediction_mode}_{split_strategy}_step_"

    existing_steps = []
    if os.path.exists(model_selection_dir):
        for f_name in os.listdir(model_selection_dir):
            if f_name.startswith(prefix) and f_name.endswith(".pkl"):
                try:
                    existing_steps.append(int(f_name.replace(prefix, "").replace(".pkl", "")))
                except ValueError:
                    pass

    if existing_steps:
        last_step = max(existing_steps)
        print(f"\n[RESUME] Detected previous run. Checking Step {last_step}...")
        try:
            with open(os.path.join(model_selection_dir, f"{prefix}{last_step}.pkl"), 'rb') as f:
                last_results = pickle.load(f)
            current_model_features = last_results['current_features']
            best_current_score = last_results['baseline_score']
            cand_dict = last_results['candidates_summary']
            if cand_dict:
                cand_stats = []
                for feat, res in cand_dict.items():
                    m = res['mean_ll']
                    s = res['se_ll']
                    cand_stats.append((feat, m, s))
                if cand_stats:
                    cand_stats.sort(key=lambda x: x[1])
                    b_name, b_score, b_se = cand_stats[0]
                    if (best_current_score - b_score) > b_se:
                        current_model_features.append(b_name)
                        best_current_score, best_current_se, step_counter = b_score, b_se, last_step + 1
        except Exception as e:
            print(f"Resume failed: {e}")

    if use_top_rank_as_anchor and step_counter == 0:
        anchor_to_force = ranked_features[0]
        print(f"\n*** AUTO-ANCHOR: Starting with top ranked feature '{anchor_to_force}' ***")
        current_model_features = [anchor_to_force]
        gam_terms = te(0, 1, n_splines=[n_splines_value, n_splines_time])
        # Per-fold scalar metrics for the anchor. See the function docstring for
        # the definition of each key. `precision` is not stored (derivable from
        # the confusion matrix); `brier` / `ece` / `mcc` are added as
        # calibration and chance-corrected summary scores.
        metrics = {'ll': [], 'auc': [], 'score': [], 'f1': [], 'recall': [],
                   'brier': [], 'ece': [], 'mcc': [],
                   'confusion_matrix': [], 'n_iter': [], 'converged': [], 'fit_time': []}

        for fold_i, fold_info in enumerate(cv_folds):
            try:
                X_train_list, X_test_list = [], []
                if fold_info['type'] == 'session':
                    train_sess, test_sess = fold_info['train_sessions'], fold_info['test_sessions']
                    anc_data = all_feature_data[anchor_feature]
                    X_p_tr, X_n_tr = pool_session_arrays(anc_data, train_sess, pos_key="usv_feature_arr", neg_key="no_usv_feature_arr", n_frames=pipeline.history_frames)
                    n_k = min(X_p_tr.shape[0], X_n_tr.shape[0])
                    anc_rng = np.random.default_rng(random_seed + fold_i)
                    idx_p = anc_rng.choice(X_p_tr.shape[0], n_k, replace=False)
                    idx_n = anc_rng.choice(X_n_tr.shape[0], n_k, replace=False)
                    y_tr_fold = np.concatenate((np.ones(n_k), np.zeros(n_k)))
                    X_p_te, X_n_te = pool_session_arrays(anc_data, test_sess, pos_key="usv_feature_arr", neg_key="no_usv_feature_arr", n_frames=pipeline.history_frames)
                    y_te_fold = np.concatenate((np.ones(X_p_te.shape[0]), np.zeros(X_n_te.shape[0])))
                    X_train_list.append(np.concatenate((X_p_tr[idx_p], X_n_tr[idx_n])))
                    X_test_list.append(np.concatenate((X_p_te, X_n_te)))
                elif fold_info['type'] == 'mixed':
                    train_ix, test_ix = fold_info['train_idx'], fold_info['test_idx']
                    n_pos_total = fold_info['n_pos_total']
                    n_neg_total = fold_info['n_neg_total']
                    y_full = np.concatenate((np.ones(n_pos_total), np.zeros(n_neg_total)))
                    # Pull the already-pooled full design matrix from the cache
                    # instead of rebuilding it on every fold.
                    X_full = pooled_feature_cache[anchor_to_force]['X_full']

                    X_tr_all = X_full[train_ix]
                    y_tr_all = y_full[train_ix]
                    X_te = X_full[test_ix]
                    y_te_fold = y_full[test_ix]

                    # Balance the training fold 50/50; test preserves natural rate.
                    X_tr_pos = X_tr_all[y_tr_all == 1]
                    X_tr_neg = X_tr_all[y_tr_all == 0]
                    n_tr_keep = min(X_tr_pos.shape[0], X_tr_neg.shape[0])
                    if n_tr_keep == 0:
                        raise ValueError("Training fold has no samples in one class after split.")
                    anc_rng = np.random.default_rng(random_seed + fold_i)
                    idx_p = anc_rng.choice(X_tr_pos.shape[0], n_tr_keep, replace=False)
                    idx_n = anc_rng.choice(X_tr_neg.shape[0], n_tr_keep, replace=False)
                    X_tr_bal = np.concatenate((X_tr_pos[idx_p], X_tr_neg[idx_n]))
                    y_tr_fold = np.concatenate((np.ones(n_tr_keep), np.zeros(n_tr_keep)))

                    X_train_list.append(X_tr_bal)
                    X_test_list.append(X_te)

                X_tr_gam = get_unrolled_X_for_multivariate(X_train_list, history_frames)
                X_te_gam = get_unrolled_X_for_multivariate(X_test_list, history_frames)
                fit_start = time.perf_counter()
                gam = LogisticGAM(gam_terms, **gam_kwargs).fit(X_tr_gam, np.repeat(y_tr_fold.astype(float), history_frames))
                fit_time = float(time.perf_counter() - fit_start)

                y_proba_tiled = gam.predict_proba(X_te_gam)
                y_proba_mean = np.mean(y_proba_tiled.reshape(len(y_te_fold), history_frames), axis=1)
                metrics['ll'].append(log_loss(y_te_fold.astype(int), np.clip(y_proba_mean, 1e-15, 1 - 1e-15)))

                y_pred_mean = (y_proba_mean > 0.5).astype(int)
                metrics['score'].append(balanced_accuracy_score(y_te_fold, y_pred_mean))
                metrics['f1'].append(f1_score(y_te_fold, y_pred_mean, zero_division=0))
                metrics['recall'].append(recall_score(y_te_fold, y_pred_mean, zero_division=0))
                metrics['auc'].append(roc_auc_score(y_te_fold, y_proba_mean) if len(np.unique(y_te_fold)) > 1 else np.nan)
                metrics['brier'].append(float(brier_score_loss(y_te_fold.astype(int), y_proba_mean)))
                try:
                    y_proba_2d = np.column_stack([1.0 - y_proba_mean, y_proba_mean])
                    metrics['ece'].append(expected_calibration_error(y_te_fold.astype(int), y_pred_mean, y_proba_2d, n_bins=10))
                except Exception:
                    metrics['ece'].append(np.nan)
                metrics['mcc'].append(safe_matthews_corrcoef(y_te_fold.astype(int), y_pred_mean))
                metrics['confusion_matrix'].append(safe_confusion_matrix(
                    y_te_fold.astype(int), y_pred_mean, labels=np.array([0, 1])
                ))
                gam_diffs = gam.logs_.get('diffs', [])
                metrics['n_iter'].append(float(len(gam_diffs)))
                metrics['converged'].append(bool(gam_diffs and gam_diffs[-1] < gam_kwargs['tol']))
                metrics['fit_time'].append(fit_time)

                del gam, X_tr_gam, X_te_gam
                gc.collect()
            except Exception as e:
                print(f"    [!] Error fitting {anchor_to_force} (Fold {fold_i}): {e}")
                for _k in metrics:
                    # Keep the per-fold list lengths aligned across every key.
                    # `confusion_matrix` stores (2, 2) arrays, so on failure we
                    # append a NaN-filled (2, 2) placeholder rather than a
                    # scalar NaN; otherwise np.stack(...) downstream would fail
                    # on a heterogeneous list.
                    if _k == 'confusion_matrix':
                        metrics[_k].append(np.full((2, 2), np.nan))
                    else:
                        metrics[_k].append(np.nan)

        valid_ll = [x for x in metrics['ll'] if np.isfinite(x)]
        if valid_ll:
            best_current_score = np.mean(valid_ll)
            best_current_se = np.std(valid_ll, ddof=1) / np.sqrt(len(valid_ll))

            # Step 0 Save
            step_0_metadata = {
                'step_idx': 0, 'current_features': [anchor_to_force],
                'baseline_score': chance_ll, 'selected_feature': anchor_to_force,
                'candidates_summary': {
                    anchor_to_force: {
                        'll': metrics['ll'], 'auc': metrics['auc'], 'score': metrics['score'],
                        'f1': metrics['f1'], 'recall': metrics['recall'],
                        'brier': metrics['brier'], 'ece': metrics['ece'], 'mcc': metrics['mcc'],
                        'confusion_matrix': metrics['confusion_matrix'],
                        'n_iter': metrics['n_iter'], 'converged': metrics['converged'],
                        'fit_time': metrics['fit_time'],
                        'mean_ll': best_current_score, 'se_ll': best_current_se
                    }
                }
            }
            # Reuse the canonical `prefix` derived above so Step 0 and all
            # subsequent steps share a single filename scheme — the legacy
            # substring-based `target_sex` path disagreed with the regex-based
            # `target_condition` and could orphan the Step 0 file from the rest
            # of the run on selection resume.
            s0_name = f"{prefix}0.pkl"
            with open(os.path.join(model_selection_dir, s0_name), 'wb') as f:
                pickle.dump(step_0_metadata, f)
            step_counter = 1

    while True:
        print(f"\n=== Step {step_counter} === Best LL: {best_current_score:.5f}")
        step_results_metadata = {
            'step_idx': step_counter, 'current_features': list(current_model_features),
            'baseline_score': best_current_score, 'candidates_summary': {},
            'selected_feature': None
        }

        best_candidate, best_candidate_score, best_candidate_se = None, float('inf'), 0.0

        for i_feat, feat in enumerate(ranked_features):
            if feat in current_model_features: continue
            gc.collect()
            trial_features = current_model_features + [feat]
            print(f"  [{i_feat}/{len(ranked_features)}] Testing +{feat}...", end="", flush=True)

            gam_terms = te(0, 1, n_splines=[n_splines_value, n_splines_time])
            for i in range(1, len(trial_features)):
                gam_terms += te(i * 2, i * 2 + 1, n_splines=[n_splines_value, n_splines_time])

            metrics = {'ll': [], 'auc': [], 'score': [], 'f1': [], 'recall': [],
                       'brier': [], 'ece': [], 'mcc': [],
                       'confusion_matrix': [], 'n_iter': [], 'converged': [], 'fit_time': []}
            for fold_i, fold_info in enumerate(cv_folds):
                try:
                    X_train_list, X_test_list = [], []
                    if fold_info['type'] == 'session':
                        train_sess, test_sess = fold_info['train_sessions'], fold_info['test_sessions']
                        anc_data = all_feature_data[anchor_feature]
                        X_p_tr, X_n_tr = pool_session_arrays(anc_data, train_sess, pos_key="usv_feature_arr", neg_key="no_usv_feature_arr", n_frames=pipeline.history_frames)
                        trial_rng = np.random.default_rng(random_seed + fold_i)
                        n_k = min(X_p_tr.shape[0], X_n_tr.shape[0])
                        idx_p = trial_rng.choice(X_p_tr.shape[0], n_k, replace=False)
                        idx_n = trial_rng.choice(X_n_tr.shape[0], n_k, replace=False)
                        y_tr_fold = np.concatenate((np.ones(n_k), np.zeros(n_k)))

                        X_p_te_anc, X_n_te_anc = pool_session_arrays(anc_data, test_sess, pos_key="usv_feature_arr", neg_key="no_usv_feature_arr", n_frames=pipeline.history_frames)
                        y_te_fold = np.concatenate((np.ones(X_p_te_anc.shape[0]), np.zeros(X_n_te_anc.shape[0])))

                        for f_name in trial_features:
                            f_p_tr, f_n_tr = pool_session_arrays(all_feature_data[f_name], train_sess, pos_key="usv_feature_arr", neg_key="no_usv_feature_arr", n_frames=pipeline.history_frames)
                            f_p_te, f_n_te = pool_session_arrays(all_feature_data[f_name], test_sess, pos_key="usv_feature_arr", neg_key="no_usv_feature_arr", n_frames=pipeline.history_frames)
                            X_train_list.append(np.concatenate((f_p_tr[idx_p], f_n_tr[idx_n])))
                            X_test_list.append(np.concatenate((f_p_te, f_n_te)))
                    elif fold_info['type'] == 'mixed':
                        train_ix, test_ix = fold_info['train_idx'], fold_info['test_idx']
                        n_pos_total = fold_info['n_pos_total']
                        n_neg_total = fold_info['n_neg_total']
                        y_full = np.concatenate((np.ones(n_pos_total), np.zeros(n_neg_total)))
                        y_tr_all = y_full[train_ix]
                        y_te_fold = y_full[test_ix]

                        # Determine balanced-train indices once (shared across all trial features).
                        pos_mask = (y_tr_all == 1)
                        neg_mask = (y_tr_all == 0)
                        pos_positions = np.where(pos_mask)[0]
                        neg_positions = np.where(neg_mask)[0]
                        n_tr_keep = min(pos_positions.size, neg_positions.size)
                        if n_tr_keep == 0:
                            raise ValueError("Training fold has no samples in one class after split.")
                        fs_rng = np.random.default_rng(random_seed + fold_i)
                        sel_pos = fs_rng.choice(pos_positions.size, n_tr_keep, replace=False)
                        sel_neg = fs_rng.choice(neg_positions.size, n_tr_keep, replace=False)
                        bal_train_local = np.concatenate((pos_positions[sel_pos], neg_positions[sel_neg]))
                        y_tr_fold = np.concatenate((np.ones(n_tr_keep), np.zeros(n_tr_keep)))

                        for f_name in trial_features:
                            # Use the pre-pooled cache rather than re-running
                            # `pool_session_arrays` on every fold for every
                            # trial feature (the dominant cost of the forward
                            # loop under 'mixed' strategy).
                            X_full = pooled_feature_cache[f_name]['X_full']
                            X_tr_all = X_full[train_ix]
                            X_train_list.append(X_tr_all[bal_train_local])
                            X_test_list.append(X_full[test_ix])

                    X_tr_gam = get_unrolled_X_for_multivariate(X_train_list, history_frames)
                    X_te_gam = get_unrolled_X_for_multivariate(X_test_list, history_frames)
                    fit_start = time.perf_counter()
                    gam = LogisticGAM(gam_terms, **gam_kwargs).fit(X_tr_gam, np.repeat(y_tr_fold.astype(float), history_frames))
                    fit_time = float(time.perf_counter() - fit_start)

                    y_proba_tiled = gam.predict_proba(X_te_gam)
                    y_proba_mean = np.mean(y_proba_tiled.reshape(len(y_te_fold), history_frames), axis=1)
                    metrics['ll'].append(log_loss(y_te_fold.astype(int), np.clip(y_proba_mean, 1e-15, 1 - 1e-15)))

                    y_pred_mean = (y_proba_mean > 0.5).astype(int)
                    metrics['score'].append(balanced_accuracy_score(y_te_fold, y_pred_mean))
                    metrics['f1'].append(f1_score(y_te_fold, y_pred_mean, zero_division=0))
                    metrics['recall'].append(recall_score(y_te_fold, y_pred_mean, zero_division=0))
                    metrics['auc'].append(roc_auc_score(y_te_fold, y_proba_mean) if len(np.unique(y_te_fold)) > 1 else np.nan)
                    metrics['brier'].append(float(brier_score_loss(y_te_fold.astype(int), y_proba_mean)))
                    try:
                        y_proba_2d = np.column_stack([1.0 - y_proba_mean, y_proba_mean])
                        metrics['ece'].append(expected_calibration_error(y_te_fold.astype(int), y_pred_mean, y_proba_2d, n_bins=10))
                    except Exception:
                        metrics['ece'].append(np.nan)
                    metrics['mcc'].append(safe_matthews_corrcoef(y_te_fold.astype(int), y_pred_mean))
                    metrics['confusion_matrix'].append(safe_confusion_matrix(
                        y_te_fold.astype(int), y_pred_mean, labels=np.array([0, 1])
                    ))
                    gam_diffs = gam.logs_.get('diffs', [])
                    metrics['n_iter'].append(float(len(gam_diffs)))
                    metrics['converged'].append(bool(gam_diffs and gam_diffs[-1] < gam_kwargs['tol']))
                    metrics['fit_time'].append(fit_time)

                    del gam, X_tr_gam, X_te_gam
                    gc.collect()
                except Exception as e:
                    print(f"    [!] Error fitting {feat} (Fold {fold_i}): {e}")
                    for _k in metrics:
                        # `confusion_matrix` holds (2, 2) arrays, so on failure
                        # append a NaN-filled matrix instead of a scalar NaN to
                        # keep the per-fold list homogeneous for downstream
                        # np.stack / indexing.
                        if _k == 'confusion_matrix':
                            metrics[_k].append(np.full((2, 2), np.nan))
                        else:
                            metrics[_k].append(np.nan)

            valid = [x for x in metrics['ll'] if np.isfinite(x)]

            if valid:
                mean_ll, se_ll = np.mean(valid), np.std(valid, ddof=1) / np.sqrt(len(valid))
                print(f" LL: {mean_ll:.4f} (range: {min(valid):.4f}-{max(valid):.4f})")

                step_results_metadata['candidates_summary'][feat] = {
                    'll': metrics['ll'], 'auc': metrics['auc'], 'score': metrics['score'],
                    'f1': metrics['f1'], 'recall': metrics['recall'],
                    'brier': metrics['brier'], 'ece': metrics['ece'], 'mcc': metrics['mcc'],
                    'confusion_matrix': metrics['confusion_matrix'],
                    'n_iter': metrics['n_iter'], 'converged': metrics['converged'],
                    'fit_time': metrics['fit_time'],
                    'mean_ll': mean_ll, 'se_ll': se_ll
                }

                if mean_ll < best_candidate_score:
                    best_candidate_score, best_candidate_se, best_candidate = mean_ll, se_ll, feat
            else:
                print(f" Failed (No valid numeric scores). Metrics: {metrics['ll']}")

        if (best_current_score - best_candidate_score) > best_candidate_se:
            print(f"  ACCEPT {best_candidate}")
            step_results_metadata['selected_feature'] = best_candidate
            current_model_features.append(best_candidate)

            fname = f"{prefix}{step_counter}.pkl"
            with open(os.path.join(model_selection_dir, fname), 'wb') as f:
                pickle.dump(step_results_metadata, f)

            best_current_score, best_current_se, step_counter = best_candidate_score, best_candidate_se, step_counter + 1
        else:
            print("  REJECT. Selection Finished.")
            step_results_metadata['selected_feature'] = None
            fname = f"{prefix}{step_counter}.pkl"
            with open(os.path.join(model_selection_dir, fname), 'wb') as f:
                pickle.dump(step_results_metadata, f)
            break

        if len(current_model_features) == len(ranked_features):
            break

    print("\n--- Final Model Fit for Visualization (CV-based) ---")
    try:
        # Use the pre-pooled anchor arrays rather than re-running
        # pool_session_arrays here.
        X_p = pooled_feature_cache[anchor_feature]['X_pos']
        X_n = pooled_feature_cache[anchor_feature]['X_neg']

        final_rng = np.random.default_rng(random_seed)
        n_k = min(X_p.shape[0], X_n.shape[0])
        idx_p = final_rng.choice(X_p.shape[0], n_k, replace=False)
        idx_n = final_rng.choice(X_n.shape[0], n_k, replace=False)

        y_final = np.concatenate((np.ones(n_k), np.zeros(n_k)))

        X_list_final = []
        for f in current_model_features:
            f_p = pooled_feature_cache[f]['X_pos']
            f_n = pooled_feature_cache[f]['X_neg']
            X_list_final.append(np.concatenate((f_p[idx_p], f_n[idx_n])))

        last_file = os.path.join(model_selection_dir, fname)
        final_fold_shapes = []

        print(f"  Calculating filter shapes across {len(cv_folds)} folds...")

        for fold_idx, (tr_idx, te_idx) in enumerate(cv_folds):

            X_list_tr = [x[tr_idx] for x in X_list_final]
            y_tr = y_final[tr_idx]

            X_gam_tr = get_unrolled_X_for_multivariate(X_list_tr, history_frames)
            y_gam_tr = np.repeat(y_tr.astype(float), history_frames)

            gam_terms = te(0, 1, n_splines=[n_splines_value, n_splines_time])
            for i in range(1, len(current_model_features)):
                gam_terms += te(i * 2, i * 2 + 1, n_splines=[n_splines_value, n_splines_time])

            gam_fold = LogisticGAM(gam_terms, **gam_kwargs).fit(X_gam_tr, y_gam_tr)

            base_grid = np.zeros((history_frames, 2 * len(current_model_features)))

            for k in range(len(current_model_features)):
                base_grid[:, k * 2 + 1] = time_indices

            base_prob = gam_fold.predict_mu(base_grid)

            fold_res = {}
            for k, f_name in enumerate(current_model_features):
                test_grid = base_grid.copy()
                test_grid[:, k * 2] = 1.0

                pred_feat = gam_fold.predict_mu(test_grid)
                fold_res[f_name] = (pred_feat - base_prob).flatten()

            final_fold_shapes.append(fold_res)

            del gam_fold, X_gam_tr, y_gam_tr
            gc.collect()

        with open(last_file, 'rb') as f:
            data = pickle.load(f)

        data.update({
            'final_model_features': current_model_features,
            'filter_shapes': final_fold_shapes,
            'univariate_results_path': univariate_results_path,
            'input_data_path': input_data_path
        })

        with open(last_file, 'wb') as f:
            pickle.dump(data, f)
        print("Final model filters saved (CV-based).")

    except Exception as e:
        print(f"Final fit failed: {e}")


def _pool_category_features(
        all_feature_data: dict,
        features_to_load: list,
        session_list: np.ndarray,
        history_frames: int
) -> tuple[list, list]:
    """
    Extracts and concatenates Target and Other arrays for a given list of features
    across specified sessions.

    Unlike older implementations, this function strictly handles pooling without
    enforcing balancing. This separation of concerns allows the upstream
    cross-validation logic to seamlessly support both 'session' and 'mixed'
    splitting strategies before the final downsampling occurs.

    Parameters
    ----------
    all_feature_data : dict
        The nested dictionary containing raw data for all features.
        Structure: `dict[feature_name][session_id]['target_feature_arr' | 'other_feature_arr']`.
    features_to_load : list of str
        The list of feature names (keys in `all_feature_data`) to extract.
    session_list : np.ndarray
        Array of session IDs (strings) to pool data from.
    history_frames : int
        The number of time columns in the feature matrices.

    Returns
    -------
    tuple[list, list]
        - X_target_list: List of pooled (n_samples, history_frames) arrays for the Target class.
        - X_other_list: List of pooled (n_samples, history_frames) arrays for the Other class.
    """

    raw_target = {feat: [] for feat in features_to_load}
    raw_other = {feat: [] for feat in features_to_load}

    for sess in session_list:
        for feat in features_to_load:
            if sess in all_feature_data[feat]:
                raw_target[feat].append(all_feature_data[feat][sess]['target_feature_arr'])
                raw_other[feat].append(all_feature_data[feat][sess]['other_feature_arr'])

    X_target_list = [np.concatenate(raw_target[f], axis=0) if raw_target[f] else np.empty((0, history_frames)) for f in features_to_load]
    X_other_list = [np.concatenate(raw_other[f], axis=0) if raw_other[f] else np.empty((0, history_frames)) for f in features_to_load]

    return X_target_list, X_other_list


def _balance_multivariate_arrays(
        X_targ_list: list,
        X_other_list: list,
        random_seed: int
) -> tuple[list, list, np.ndarray, np.ndarray]:
    """
    Downsamples the majority class to enforce a strict 50/50 balance across all
    feature arrays simultaneously.

    This ensures that row-alignment is maintained across the multivariate feature
    lists, preventing misalignment between predictors during classification.

    Parameters
    ----------
    X_targ_list : list of np.ndarray
        The aligned list of Target class arrays.
    X_other_list : list of np.ndarray
        The aligned list of Other class arrays.
    random_seed : int
        Seed for the random number generator used for downsampling indices.

    Returns
    -------
    tuple
        1. X_tr_t_bal (list of np.ndarray): Balanced Target arrays.
        2. X_tr_o_bal (list of np.ndarray): Balanced Other arrays.
        3. y_target (np.ndarray): Array of ones.
        4. y_other (np.ndarray): Array of zeros.
        Returns ([], [], None, None) if data is insufficient.
    """

    n_target = X_targ_list[0].shape[0]
    n_other = X_other_list[0].shape[0]

    if n_target == 0 or n_other == 0:
        return [], [], None, None

    limit = min(n_target, n_other)
    rng = np.random.default_rng(random_seed)
    idx_target = rng.choice(n_target, limit, replace=False)
    idx_other = rng.choice(n_other, limit, replace=False)

    X_tr_t_bal = [x[idx_target] for x in X_targ_list]
    X_tr_o_bal = [x[idx_other] for x in X_other_list]

    y_target = np.ones(limit)
    y_other = np.zeros(limit)

    return X_tr_t_bal, X_tr_o_bal, y_target, y_other


def vocal_category_model_selection(
        univariate_results_path: str,
        input_data_path: str,
        output_directory: str,
        settings_path: str = None,
        use_top_rank_as_anchor: bool = False,
        p_val: float = 0.025
) -> None:
    """
    Performs Forward Stepwise Selection for Vocal Category prediction using strict
    mixed or session-based splitting, class balancing, and engine toggling (sklearn/pygam).

    This function identifies the optimal subset of behavioral features that predict
    a specific USV category (one-vs-rest). It adapts the standard forward selection
    algorithm to handle the specific data structure of vocal categories ('target' vs 'other')
    and enforces a consistent splitting & balancing invariant (see below) across
    all cross-validation folds.

    Splitting & balancing invariant:
    --------------------------------
    Across both 'session' and 'mixed' strategies, the training fold is down-sampled
    to a 50/50 class balance (Target vs. Other), while the test fold preserves the
    natural class prior of the source data. The 'mixed' strategy achieves this by
    running `StratifiedShuffleSplit` on the *unbalanced* pool and balancing the
    training half inside the per-fold loop. Reported metrics should therefore be
    imbalance-robust: the `score` field stores `balanced_accuracy`, paired with
    AUC and log-loss.

    Engine Flexibility:
    -------------------
    Reads the user-defined `model_type` from the JSON settings to dynamically build models:
    - 'sklearn': Utilizes `LogisticRegressionCV` combined with linear basis projection
                 (e.g., raised_cosines, b-splines) to rapidly isolate temporal predictors.
    - 'pygam': Utilizes `LogisticGAM` with unrolled tensor product splines to capture
               non-linear log-odds surfaces.

    The process follows these steps:
    1.  Extracts the target category and sex directly from the univariate path.
    2.  Loads univariate results and ranks candidate features by Log-Likelihood (LL).
    3.  Evaluates 'split_strategy'. If 'session', isolates whole recording days.
        If 'mixed', pools all data and uses StratifiedShuffleSplit to ensure proportional
        epoch-level splitting across all folds.
    4.  Forward selection loop:
        - Starts with an empty model (or the top-ranked anchor).
        - Iteratively tests adding every remaining candidate feature.
        - Calculates and saves raw lists of metrics (Log-Loss, AUC, F1,
          Balanced Accuracy, Precision, Recall) for every fold for every candidate.
    5.  Decision rule: Adopts the one-standard-error (1SE) rule.
    6.  Final Refit: Computes filter shapes for visualization depending on the chosen
        engine (back-projection for sklearn, partial dependence for pygam).

    Parameters
    ----------
    univariate_results_path : str
        The absolute path to the .pkl file containing univariate modeling results.
    input_data_path : str
        The absolute path to the .pkl file containing the raw feature data.
    output_directory : str
        The specific directory path where the resulting step files (.pkl) will be saved.
    settings_path : str, optional
        The path to the 'modeling_settings.json' configuration file.
    use_top_rank_as_anchor : bool, default False
        If True, the selection process initializes with the highest-ranked univariate feature.
    p_val : float, default 0.025
        The alpha level used to determine significance against shuffled null.

    Returns
    -------
    None
    """

    print("--- Starting Vocal Category Model Selection ---")
    chance_ll = np.log(2)

    if settings_path is None:
        settings_path_obj = pathlib.Path(__file__).resolve().parent.parent / '_parameter_settings/modeling_settings.json'
        settings_path = str(settings_path_obj)

    try:
        with open(settings_path, 'r') as f:
            settings = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Settings file not found at {settings_path}")

    model_selection_dir = output_directory
    os.makedirs(model_selection_dir, exist_ok=True)
    print(f"Results will be saved to: {model_selection_dir}")
    print(f"Loading univariate results from: {univariate_results_path}")
    print(f"Loading raw data from: {input_data_path}")

    # Extract target metadata safely
    fname = os.path.basename(univariate_results_path)

    cat_match = re.search(r'category_(\d+)', fname)
    target_category = f"category_{cat_match.group(1)}" if cat_match else "category_unknown"
    cond_match = re.search(r'((?:male|female).*?)(?=_splits|_lam|_gmm|\.pkl)', fname)
    target_condition = cond_match.group(1) if cond_match else "unknown"
    print(f"Target: {target_category} ({target_condition})")

    with open(univariate_results_path, 'rb') as f:
        univariate_data = pickle.load(f)

    # Filter candidates safely
    candidates = []
    for feat_name, payload in univariate_data.items():
        if isinstance(payload, tuple) and len(payload) == 2:
            _, results = payload
        else:
            results = payload

        if 'actual' not in results or 'll' not in results['actual']:
            continue

        actual_ll = results['actual']['ll']
        null_ll = results['null']['ll']

        valid_actual = actual_ll[~np.isnan(actual_ll)]
        valid_null = null_ll[~np.isnan(null_ll)]

        if len(valid_actual) == 0 or len(valid_null) == 0:
            continue

        mean_actual_ll = np.mean(valid_actual)
        null_threshold = np.percentile(valid_null, q=(p_val / len(univariate_data)) * 100)

        if mean_actual_ll < null_threshold:
            candidates.append({'feature': feat_name, 'mean_ll': mean_actual_ll})

    candidates.sort(key=lambda x: x['mean_ll'])
    ranked_features = [x['feature'] for x in candidates]
    if not ranked_features:
        print("No significant features found. Aborting.")
        return
    print(f"Identified {len(ranked_features)} significant candidates. Top: {ranked_features[0]}")

    print("Loading raw input data...")
    all_feature_data = load_pickle_modeling_data(input_data_path)

    common_sessions = set(all_feature_data[ranked_features[0]].keys())
    for feat in ranked_features[1:]:
        common_sessions = common_sessions.intersection(set(all_feature_data[feat].keys()))

    valid_sessions = []
    for sess in common_sessions:
        try:
            if (all_feature_data[ranked_features[0]][sess]['target_feature_arr'].shape[0] > 0 and
                    all_feature_data[ranked_features[0]][sess]['other_feature_arr'].shape[0] > 0):
                valid_sessions.append(sess)
        except KeyError:
            continue

    all_sessions_arr = np.array(sorted(valid_sessions))
    if len(all_sessions_arr) < 2:
        raise ValueError("Not enough common sessions for cross-validation.")

    # Strict Dictionary Access Only (No .get())
    camera_rate = settings['io']['camera_sampling_rate']
    filter_history_sec = settings['model_params']['filter_history']
    history_frames = int(np.floor(camera_rate * filter_history_sec))

    model_ops = settings['model_params']
    model_type = model_ops['model_type']
    n_splits = model_ops['split_num']
    test_prop = model_ops['test_proportion']
    split_strategy = model_ops['split_strategy']
    random_seed = settings['model_params']['random_seed']

    basis_matrix = None
    gam_kwargs = {}
    n_splines_time = 0
    n_splines_value = 0
    lr_params = {}

    # Initialize Engine Logic
    if model_type == 'sklearn':
        basis_type = model_ops['model_basis_function']
        if basis_type == 'raised_cosine':
            p = settings['hyperparameters']['basis_functions']['raised_cosine']
            kp = int(np.floor(history_frames * p['kpeaks_proportion']))
            basis_matrix = raised_cosine(neye=p['neye'], ncos=p['ncos'], kpeaks=[0, kp], b=p['b'], w=history_frames)
        elif basis_type == 'bspline':
            p = settings['hyperparameters']['basis_functions']['bspline']
            deg = p['degree']
            max_k = max(0, history_frames - deg)
            knots = np.linspace(0, max_k, p['n_splines'] - deg + 1).astype(int)
            basis_matrix = _normalizecols(bsplines(width=history_frames, positions=knots, degree=deg))
        elif basis_type == 'laplacian_pyramid':
            p = settings['hyperparameters']['basis_functions']['laplacian_pyramid']
            basis_matrix = _normalizecols(laplacian_pyramid(width=history_frames, levels=p['levels'], fwhm=p['fwhm']))
        elif basis_type == 'identity':
            basis_matrix = identity(width=history_frames)

        lr_params = settings['hyperparameters']['classical']['logistic_regression']
        print(f"Engine: Sklearn LogisticRegressionCV with '{basis_type}' projection.")

    elif model_type == 'pygam':
        pygam_params = settings['hyperparameters']['classical']['pygam']
        n_splines_time = pygam_params['n_splines_time']
        n_splines_value = pygam_params['n_splines_value']
        gam_kwargs = {
            'max_iter': int(pygam_params['max_iterations']),
            'tol': float(pygam_params['tol_val']),
            'lam': pygam_params['lam_penalty']
        }
        print(f"Engine: PyGAM LogisticGAM with Tensor Splines.")
    else:
        raise ValueError("model_type in settings must be either 'sklearn' or 'pygam'.")

    print(f"Random Seed: {random_seed} | Num Splits: {n_splits} | Strategy: {split_strategy}")

    # Establish CV indices globally to maintain fold consistency across features
    cv_folds = []
    n_targ_total, n_other_total = 0, 0

    # Pre-pool each candidate feature's full (target, other) arrays once for
    # the 'mixed' strategy. Because the pooled result depends only on the
    # feature (session list is always `all_sessions_arr`), the anchor and
    # forward loops read from this cache instead of re-running
    # `_pool_category_features` per fold per trial feature — the dominant cost
    # of the inner loop at ~ (n_steps * n_features * n_folds) pool calls.
    pooled_category_cache = {}
    if split_strategy == 'session':
        ss = ShuffleSplit(n_splits=n_splits, test_size=test_prop, random_state=random_seed)
        cv_folds = list(ss.split(all_sessions_arr))
    elif split_strategy == 'mixed':
        for _feat_name in ranked_features:
            _t_list, _o_list = _pool_category_features(
                all_feature_data, [_feat_name], all_sessions_arr, history_frames
            )
            pooled_category_cache[_feat_name] = {
                'target': _t_list[0],
                'other': _o_list[0],
            }
        ref_targ = pooled_category_cache[ranked_features[0]]['target']
        ref_other = pooled_category_cache[ranked_features[0]]['other']
        n_targ_total = ref_targ.shape[0]
        n_other_total = ref_other.shape[0]

        y_dummy = np.hstack([np.ones(n_targ_total), np.zeros(n_other_total)])
        sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_prop, random_state=random_seed)
        cv_folds = list(sss.split(np.zeros(len(y_dummy)), y_dummy))
    else:
        raise ValueError("split_strategy in settings must be either 'session' or 'mixed'.")

    prefix = f"model_selection_{target_condition}_{target_category}_step_"
    existing_steps = []
    if os.path.exists(model_selection_dir):
        for f_name in os.listdir(model_selection_dir):
            if f_name.startswith(prefix) and f_name.endswith(".pkl"):
                try:
                    existing_steps.append(int(f_name.replace(prefix, "").replace(".pkl", "")))
                except ValueError:
                    pass

    step_counter, current_model_features = 0, []
    best_current_score, best_current_se = chance_ll, 0.0

    if existing_steps:
        last_step = max(existing_steps)
        print(f"\n[RESUME] Detected previous run. Checking Step {last_step}...")
        try:
            with open(os.path.join(model_selection_dir, f"{prefix}{last_step}.pkl"), 'rb') as f:
                last_results = pickle.load(f)
            current_model_features = last_results['current_features']
            best_current_score = last_results['baseline_score']
            cand_dict = last_results['candidates_summary'] if 'candidates_summary' in last_results else {}

            if cand_dict:
                cand_stats = []
                for feat, res in cand_dict.items():
                    m = res['mean_ll'] if 'mean_ll' in res else chance_ll
                    s = res['se_ll'] if 'se_ll' in res else 0.0
                    cand_stats.append((feat, m, s))
                if cand_stats:
                    cand_stats.sort(key=lambda x: x[1])
                    b_name, b_score, b_se = cand_stats[0]
                    if (best_current_score - b_score) > b_se:
                        current_model_features.append(b_name)
                        best_current_score, best_current_se, step_counter = b_score, b_se, last_step + 1
        except Exception as e:
            print(f"Resume failed: {e}")

    if use_top_rank_as_anchor and step_counter == 0:
        anchor = ranked_features[0]
        print(f"\n*** AUTO-ANCHOR: Starting with {anchor} ***")
        current_model_features = [anchor]
        metrics = {'ll': [], 'auc': [], 'score': [], 'f1': [], 'recall': [],
                   'brier': [], 'ece': [], 'mcc': [],
                   'confusion_matrix': [], 'n_iter': [], 'converged': [], 'fit_time': []}

        if model_type == 'pygam':
            gam_terms = te(0, 1, n_splines=[n_splines_value, n_splines_time])

        for fold_i, (tr_idx, te_idx) in enumerate(cv_folds):
            try:
                if split_strategy == 'session':
                    X_tr_t_raw, X_tr_o_raw = _pool_category_features(all_feature_data, [anchor], all_sessions_arr[tr_idx], history_frames)
                    X_te_t_raw, X_te_o_raw = _pool_category_features(all_feature_data, [anchor], all_sessions_arr[te_idx], history_frames)
                else:
                    # Pull the already-pooled target / other arrays from the
                    # cache instead of rebuilding them on every fold.
                    X_all_t_raw = [pooled_category_cache[anchor]['target']]
                    X_all_o_raw = [pooled_category_cache[anchor]['other']]

                    tr_targ_idx = tr_idx[tr_idx < n_targ_total]
                    tr_other_idx = tr_idx[tr_idx >= n_targ_total] - n_targ_total
                    te_targ_idx = te_idx[te_idx < n_targ_total]
                    te_other_idx = te_idx[te_idx >= n_targ_total] - n_targ_total

                    X_tr_t_raw = [x[tr_targ_idx] for x in X_all_t_raw]
                    X_tr_o_raw = [x[tr_other_idx] for x in X_all_o_raw]
                    X_te_t_raw = [x[te_targ_idx] for x in X_all_t_raw]
                    X_te_o_raw = [x[te_other_idx] for x in X_all_o_raw]

                X_tr_t, X_tr_o, y_tr_t, y_tr_o = _balance_multivariate_arrays(X_tr_t_raw, X_tr_o_raw, random_seed + fold_i)

                # Test set: natural class prior (no balancing).
                if not X_tr_t or len(X_te_t_raw) == 0:
                    continue
                n_te_targ = X_te_t_raw[0].shape[0]
                n_te_other = X_te_o_raw[0].shape[0]
                if (n_te_targ + n_te_other) == 0:
                    continue

                X_tr_list = [np.concatenate([t, o], axis=0) for t, o in zip(X_tr_t, X_tr_o)]
                y_tr = np.concatenate([y_tr_t, y_tr_o])
                perm = np.random.default_rng(random_seed + fold_i).permutation(len(y_tr))

                X_te_list = [np.concatenate([t, o], axis=0) for t, o in zip(X_te_t_raw, X_te_o_raw)]
                y_te = np.concatenate([np.ones(n_te_targ), np.zeros(n_te_other)]).astype(int)

                if model_type == 'sklearn':
                    X_tr_stacked = np.hstack([np.dot(x[perm], basis_matrix) for x in X_tr_list])
                    X_te_stacked = np.hstack([np.dot(x, basis_matrix) for x in X_te_list])

                    model = LogisticRegressionCV(
                        penalty=lr_params['penalty'],
                        Cs=lr_params['cs'],
                        cv=lr_params['cv'],
                        class_weight='balanced',
                        solver=lr_params['solver'],
                        max_iter=lr_params['max_iter'],
                        random_state=random_seed
                    )
                    fit_start = time.perf_counter()
                    model.fit(X_tr_stacked, y_tr[perm])
                    fit_time = float(time.perf_counter() - fit_start)
                    y_proba = model.predict_proba(X_te_stacked)[:, 1]
                    y_pred = model.predict(X_te_stacked)
                    try:
                        fold_n_iter = float(np.max(model.n_iter_))
                        fold_converged = bool(fold_n_iter < lr_params['max_iter'])
                    except Exception:
                        fold_n_iter, fold_converged = np.nan, False
                else:
                    X_tr_gam = get_unrolled_X_for_multivariate([x[perm] for x in X_tr_list], history_frames)
                    y_tr_gam = np.repeat(y_tr[perm].astype(float), history_frames)
                    fit_start = time.perf_counter()
                    gam = LogisticGAM(gam_terms, **gam_kwargs).fit(X_tr_gam, y_tr_gam)
                    fit_time = float(time.perf_counter() - fit_start)

                    X_te_gam = get_unrolled_X_for_multivariate(X_te_list, history_frames)
                    y_proba_tiled = gam.predict_proba(X_te_gam)
                    y_proba = np.mean(y_proba_tiled.reshape(len(y_te), history_frames), axis=1)
                    y_pred = (y_proba >= 0.5).astype(int)
                    gam_diffs = gam.logs_.get('diffs', [])
                    fold_n_iter = float(len(gam_diffs))
                    fold_converged = bool(gam_diffs and gam_diffs[-1] < gam_kwargs['tol'])

                metrics['ll'].append(log_loss(y_te, np.clip(y_proba, 1e-15, 1 - 1e-15)))
                metrics['auc'].append(roc_auc_score(y_te, y_proba))
                metrics['f1'].append(f1_score(y_te, y_pred))
                metrics['recall'].append(recall_score(y_te, y_pred, zero_division=0))
                metrics['score'].append(balanced_accuracy_score(y_te, y_pred))
                metrics['brier'].append(float(brier_score_loss(y_te, y_proba)))
                try:
                    y_proba_2d = np.column_stack([1.0 - y_proba, y_proba])
                    metrics['ece'].append(expected_calibration_error(y_te, y_pred, y_proba_2d, n_bins=10))
                except Exception:
                    metrics['ece'].append(np.nan)
                metrics['mcc'].append(safe_matthews_corrcoef(y_te, y_pred))
                metrics['confusion_matrix'].append(safe_confusion_matrix(y_te, y_pred, labels=np.array([0, 1])))
                metrics['n_iter'].append(fold_n_iter)
                metrics['converged'].append(fold_converged)
                metrics['fit_time'].append(fit_time)
                gc.collect()
            except Exception as e:
                print(f"    [!] Error fitting {anchor} (Fold {fold_i}): {e}")
                for _k in metrics:
                    # `confusion_matrix` holds (2, 2) arrays, so on failure we
                    # append a NaN-filled matrix instead of a scalar NaN to
                    # keep the per-fold list homogeneous for downstream stacking.
                    if _k == 'confusion_matrix':
                        metrics[_k].append(np.full((2, 2), np.nan))
                    else:
                        metrics[_k].append(np.nan)

        valid = [s for s in metrics['ll'] if np.isfinite(s)]
        if valid:
            best_current_score = np.mean(valid)
            best_current_se = np.std(valid, ddof=1) / np.sqrt(len(valid))
            step_results = {
                'step_idx': 0, 'current_features': list(current_model_features),
                'baseline_score': best_current_score,
                'selected_feature': anchor,
                'candidates_summary': {
                    anchor: {
                        'll': metrics['ll'], 'auc': metrics['auc'],
                        'f1': metrics['f1'], 'recall': metrics['recall'],
                        'score': metrics['score'],
                        'brier': metrics['brier'], 'ece': metrics['ece'], 'mcc': metrics['mcc'],
                        'confusion_matrix': metrics['confusion_matrix'],
                        'n_iter': metrics['n_iter'], 'converged': metrics['converged'],
                        'fit_time': metrics['fit_time'],
                        'mean_ll': best_current_score, 'se_ll': best_current_se
                    }
                }
            }
            with open(os.path.join(model_selection_dir, f"{prefix}0.pkl"), 'wb') as f:
                pickle.dump(step_results, f)
            step_counter = 1

    print("\n--- Starting Forward Selection ---")
    while True:
        print(f"\n=== Step {step_counter} === Best LL: {best_current_score:.5f}")
        step_results_metadata = {
            'step_idx': step_counter, 'current_features': list(current_model_features),
            'baseline_score': best_current_score, 'candidates_summary': {}
        }
        best_cand_name, best_cand_score, best_cand_se = None, float('inf'), 0.0

        for i_feat, feat in enumerate(ranked_features):
            if feat in current_model_features: continue
            gc.collect()
            trial_feats = current_model_features + [feat]
            print(f"  [{i_feat + 1}/{len(ranked_features)}] Testing +{feat}...", end="", flush=True)

            if model_type == 'pygam':
                gam_terms = te(0, 1, n_splines=[n_splines_value, n_splines_time])
                for i in range(1, len(trial_feats)):
                    gam_terms += te(i * 2, i * 2 + 1, n_splines=[n_splines_value, n_splines_time])

            metrics = {'ll': [], 'auc': [], 'score': [], 'f1': [], 'recall': [],
                       'brier': [], 'ece': [], 'mcc': [],
                       'confusion_matrix': [], 'n_iter': [], 'converged': [], 'fit_time': []}

            for fold_i, (tr_idx, te_idx) in enumerate(cv_folds):
                try:
                    if split_strategy == 'session':
                        X_tr_t_raw, X_tr_o_raw = _pool_category_features(all_feature_data, trial_feats, all_sessions_arr[tr_idx], history_frames)
                        X_te_t_raw, X_te_o_raw = _pool_category_features(all_feature_data, trial_feats, all_sessions_arr[te_idx], history_frames)
                    else:
                        # Reassemble the per-trial feature list from the
                        # pre-pooled cache so every fold reuses the same dense
                        # (target, other) arrays without re-concatenating per
                        # session on each iteration.
                        X_all_t_raw = [pooled_category_cache[f]['target'] for f in trial_feats]
                        X_all_o_raw = [pooled_category_cache[f]['other'] for f in trial_feats]

                        tr_targ_idx = tr_idx[tr_idx < n_targ_total]
                        tr_other_idx = tr_idx[tr_idx >= n_targ_total] - n_targ_total
                        te_targ_idx = te_idx[te_idx < n_targ_total]
                        te_other_idx = te_idx[te_idx >= n_targ_total] - n_targ_total

                        X_tr_t_raw = [x[tr_targ_idx] for x in X_all_t_raw]
                        X_tr_o_raw = [x[tr_other_idx] for x in X_all_o_raw]
                        X_te_t_raw = [x[te_targ_idx] for x in X_all_t_raw]
                        X_te_o_raw = [x[te_other_idx] for x in X_all_o_raw]

                    X_tr_t, X_tr_o, y_tr_t, y_tr_o = _balance_multivariate_arrays(X_tr_t_raw, X_tr_o_raw, random_seed + fold_i)

                    # Test set: natural class prior (no balancing).
                    if not X_tr_t or len(X_te_t_raw) == 0:
                        continue
                    n_te_targ = X_te_t_raw[0].shape[0]
                    n_te_other = X_te_o_raw[0].shape[0]
                    if (n_te_targ + n_te_other) == 0:
                        continue

                    X_tr_list = [np.concatenate([t, o], axis=0) for t, o in zip(X_tr_t, X_tr_o)]
                    y_tr = np.concatenate([y_tr_t, y_tr_o])
                    perm = np.random.default_rng(random_seed + fold_i).permutation(len(y_tr))

                    X_te_list = [np.concatenate([t, o], axis=0) for t, o in zip(X_te_t_raw, X_te_o_raw)]
                    y_te = np.concatenate([np.ones(n_te_targ), np.zeros(n_te_other)]).astype(int)

                    if model_type == 'sklearn':
                        X_tr_stacked = np.hstack([np.dot(x[perm], basis_matrix) for x in X_tr_list])
                        X_te_stacked = np.hstack([np.dot(x, basis_matrix) for x in X_te_list])

                        model = LogisticRegressionCV(
                            penalty=lr_params['penalty'],
                            Cs=lr_params['cs'],
                            cv=lr_params['cv'],
                            class_weight='balanced',
                            solver=lr_params['solver'],
                            max_iter=lr_params['max_iter'],
                            random_state=random_seed
                        )
                        fit_start = time.perf_counter()
                        model.fit(X_tr_stacked, y_tr[perm])
                        fit_time = float(time.perf_counter() - fit_start)
                        y_proba = model.predict_proba(X_te_stacked)[:, 1]
                        y_pred = model.predict(X_te_stacked)
                        try:
                            fold_n_iter = float(np.max(model.n_iter_))
                            fold_converged = bool(fold_n_iter < lr_params['max_iter'])
                        except Exception:
                            fold_n_iter, fold_converged = np.nan, False
                    else:
                        X_tr_gam = get_unrolled_X_for_multivariate([x[perm] for x in X_tr_list], history_frames)
                        y_tr_gam = np.repeat(y_tr[perm].astype(float), history_frames)
                        fit_start = time.perf_counter()
                        gam = LogisticGAM(gam_terms, **gam_kwargs).fit(X_tr_gam, y_tr_gam)
                        fit_time = float(time.perf_counter() - fit_start)

                        X_te_gam = get_unrolled_X_for_multivariate(X_te_list, history_frames)
                        y_proba_tiled = gam.predict_proba(X_te_gam)
                        y_proba = np.mean(y_proba_tiled.reshape(len(y_te), history_frames), axis=1)
                        y_pred = (y_proba >= 0.5).astype(int)
                        gam_diffs = gam.logs_.get('diffs', [])
                        fold_n_iter = float(len(gam_diffs))
                        fold_converged = bool(gam_diffs and gam_diffs[-1] < gam_kwargs['tol'])

                    metrics['ll'].append(log_loss(y_te, np.clip(y_proba, 1e-15, 1 - 1e-15)))
                    metrics['auc'].append(roc_auc_score(y_te, y_proba))
                    metrics['f1'].append(f1_score(y_te, y_pred))
                    metrics['recall'].append(recall_score(y_te, y_pred, zero_division=0))
                    metrics['score'].append(balanced_accuracy_score(y_te, y_pred))
                    metrics['brier'].append(float(brier_score_loss(y_te, y_proba)))
                    try:
                        y_proba_2d = np.column_stack([1.0 - y_proba, y_proba])
                        metrics['ece'].append(expected_calibration_error(y_te, y_pred, y_proba_2d, n_bins=10))
                    except Exception:
                        metrics['ece'].append(np.nan)
                    metrics['mcc'].append(safe_matthews_corrcoef(y_te, y_pred))
                    metrics['confusion_matrix'].append(safe_confusion_matrix(y_te, y_pred, labels=np.array([0, 1])))
                    metrics['n_iter'].append(fold_n_iter)
                    metrics['converged'].append(fold_converged)
                    metrics['fit_time'].append(fit_time)
                    gc.collect()
                except Exception as e:
                    print(f"    [!] Error fitting {feat} (Fold {fold_i}): {e}")
                    for _k in metrics:
                        # `confusion_matrix` holds (2, 2) arrays, so on failure
                        # append a NaN-filled matrix instead of a scalar NaN to
                        # keep the per-fold list homogeneous for downstream
                        # np.stack / indexing.
                        if _k == 'confusion_matrix':
                            metrics[_k].append(np.full((2, 2), np.nan))
                        else:
                            metrics[_k].append(np.nan)

            valid = [x for x in metrics['ll'] if np.isfinite(x)]
            if not valid:
                print(" Failed (no finite folds).")
                continue

            mean_ll = np.mean(valid)
            se_ll = np.std(valid, ddof=1) / np.sqrt(len(valid))
            print(f" LL: {mean_ll:.4f} | AUC: {np.nanmean(metrics['auc']):.3f}")

            step_results_metadata['candidates_summary'][feat] = {
                'll': metrics['ll'], 'auc': metrics['auc'],
                'f1': metrics['f1'], 'recall': metrics['recall'],
                'score': metrics['score'],
                'brier': metrics['brier'], 'ece': metrics['ece'], 'mcc': metrics['mcc'],
                'confusion_matrix': metrics['confusion_matrix'],
                'n_iter': metrics['n_iter'], 'converged': metrics['converged'],
                'fit_time': metrics['fit_time'],
                'mean_ll': mean_ll, 'se_ll': se_ll
            }
            if mean_ll < best_cand_score:
                best_cand_score, best_cand_se, best_cand_name = mean_ll, se_ll, feat

        if best_cand_name and (best_current_score - best_cand_score) > best_cand_se:
            print(f"  ACCEPT {best_cand_name}")
            current_model_features.append(best_cand_name)
            step_results_metadata['selected_feature'] = best_cand_name
            with open(os.path.join(model_selection_dir, f"{prefix}{step_counter}.pkl"), 'wb') as f:
                pickle.dump(step_results_metadata, f)
            # Track `best_current_se` alongside the score so the 1SE acceptance
            # test always compares against the *current* accepted model's
            # sampling variability rather than a stale Step-0 value.
            best_current_score, best_current_se, step_counter = best_cand_score, best_cand_se, step_counter + 1
            if len(current_model_features) == len(ranked_features): break
        else:
            print("  REJECT. Stopping.")
            step_results_metadata['selected_feature'] = None
            with open(os.path.join(model_selection_dir, f"{prefix}{step_counter}.pkl"), 'wb') as f:
                pickle.dump(step_results_metadata, f)
            break

    print("Final Feature Set:", current_model_features)

    print("\n--- Final Refit for Visualization (CV-based) ---")
    try:
        last_file = os.path.join(model_selection_dir, f"{prefix}{step_counter}.pkl")
        if not os.path.exists(last_file):
            last_file = os.path.join(model_selection_dir, f"{prefix}{step_counter - 1}.pkl")

        final_fold_shapes = []
        print(f"  Calculating filter shapes across {len(cv_folds)} folds...")

        for fold_idx, (tr_idx, te_idx) in enumerate(cv_folds):

            if split_strategy == 'session':
                X_tr_t_raw, X_tr_o_raw = _pool_category_features(all_feature_data, current_model_features, all_sessions_arr[tr_idx], history_frames)
            else:
                # Reuse the pre-pooled target / other arrays for the final
                # refit so the visualisation loop doesn't re-pool the full
                # dataset per fold.
                X_all_t_raw = [pooled_category_cache[f]['target'] for f in current_model_features]
                X_all_o_raw = [pooled_category_cache[f]['other'] for f in current_model_features]

                tr_targ_idx = tr_idx[tr_idx < n_targ_total]
                tr_other_idx = tr_idx[tr_idx >= n_targ_total] - n_targ_total

                X_tr_t_raw = [x[tr_targ_idx] for x in X_all_t_raw]
                X_tr_o_raw = [x[tr_other_idx] for x in X_all_o_raw]

            X_tr_t, X_tr_o, y_tr_t, y_tr_o = _balance_multivariate_arrays(X_tr_t_raw, X_tr_o_raw, random_seed + fold_idx)

            if not X_tr_t:
                continue

            X_tr_list = [np.concatenate([t, o], axis=0) for t, o in zip(X_tr_t, X_tr_o)]
            y_tr = np.concatenate([y_tr_t, y_tr_o])

            if model_type == 'sklearn':
                X_tr_stacked = np.hstack([np.dot(x, basis_matrix) for x in X_tr_list])

                model = LogisticRegressionCV(
                    penalty=lr_params['penalty'],
                    Cs=lr_params['cs'],
                    cv=lr_params['cv'],
                    class_weight='balanced',
                    solver=lr_params['solver'],
                    max_iter=lr_params['max_iter'],
                    random_state=random_seed
                )
                model.fit(X_tr_stacked, y_tr)

                coefs = model.coef_.flatten()
                n_bases = basis_matrix.shape[1]

                fold_res = {}
                for k, f_name in enumerate(current_model_features):
                    feat_coefs = coefs[k * n_bases: (k + 1) * n_bases]
                    fold_res[f_name] = np.dot(feat_coefs, basis_matrix.T).flatten()

                final_fold_shapes.append(fold_res)

            else:
                X_gam_tr = get_unrolled_X_for_multivariate(X_tr_list, history_frames)
                y_gam_tr = np.repeat(y_tr.astype(float), history_frames)

                gam_terms = te(0, 1, n_splines=[n_splines_value, n_splines_time])
                for i in range(1, len(current_model_features)):
                    gam_terms += te(i * 2, i * 2 + 1, n_splines=[n_splines_value, n_splines_time])

                gam_fold = LogisticGAM(gam_terms, **gam_kwargs).fit(X_gam_tr, y_gam_tr)

                time_idx = np.arange(history_frames)
                base_grid = np.zeros((history_frames, 2 * len(current_model_features)))
                for k in range(len(current_model_features)):
                    base_grid[:, k * 2 + 1] = time_idx

                base_prob = gam_fold.predict_mu(base_grid)

                fold_res = {}
                for k, f_name in enumerate(current_model_features):
                    test_grid = base_grid.copy()
                    test_grid[:, k * 2] = 1.0
                    fold_res[f_name] = (gam_fold.predict_mu(test_grid) - base_prob).flatten()

                final_fold_shapes.append(fold_res)

                del gam_fold, X_gam_tr, y_gam_tr

            gc.collect()

        final_res = {
            'final_model_features': current_model_features,
            'filter_shapes': final_fold_shapes,
            'univariate_results_path': univariate_results_path,
            'input_data_path': input_data_path
        }

        with open(last_file, 'rb') as f:
            data = pickle.load(f)

        data.update(final_res)

        with open(last_file, 'wb') as f:
            pickle.dump(data, f)

        print("Model selection complete. CV-based shapes saved.")

    except Exception as e:
        print(f"Final fit failed: {e}")


def bout_parameter_model_selection(
        univariate_results_path: str,
        input_data_path: str,
        output_directory: str,
        settings_path: str = None,
        target_variable: str = 'bout_durations',
        use_top_rank_as_anchor: bool = False,
        p_val: float = 0.01
) -> None:
    """
    Performs forward stepwise selection for continuous bout parameter regression.

    Identifies the minimal sufficient set of behavioral and vocal syntax features
    that predict a continuous target (e.g., duration, complexity). The framework
    accommodates strictly positive, right-skewed biological data.

    Engine Flexibility:
    -------------------
    Reads `model_type` from JSON settings to dynamically construct models:
    - 'sklearn': Utilizes `RidgeCV` combined with linear basis projection. To satisfy
                 normality and homoscedasticity assumptions, the target variable (y)
                 is log-transformed prior to fitting, and predictions are back-transformed.
                 Back-transformation applies a Jensen-inequality bias correction
                 (`y_pred = exp(X_te @ beta + sigma^2 / 2)`, with `sigma^2` estimated from the
                 training residuals of `log(y_tr + 1e-6)`): without the `+ sigma^2 / 2` shift,
                 naive exponentiation of the linear-predictor targets the conditional median
                 rather than the conditional mean under a lognormal model, which systematically
                 biases Gamma deviance and MSLE against the sklearn branch.
    - 'pygam': Utilizes `GAM` with a Gamma distribution and Log-link function. High-dimensional
               features are unrolled into tensor product splines (te) to capture non-linear surfaces.
               For each test trial the `H` per-frame predictions produced by the tensor-product
               unroll are aggregated on the linear-predictor (eta = log mu) scale before applying
               the inverse link (i.e. `y_pred = exp(mean(eta))`, not `mean(exp(eta))`), which
               avoids the Jensen-inequality bias that natural-scale averaging would introduce
               whenever the per-frame eta have any spread.

    Known caveat (tile-and-repeat unroll):
    --------------------------------------
    The tensor-product unroll duplicates each trial's scalar target `y_tr` across `H` history
    frames via `np.repeat(y_tr, H)` and fits as if those `N * H` rows were independent
    observations. This inflates the effective sample size seen by pyGAM's penalty selection
    (GCV/REML), nudging it toward under-smoothing relative to a truly i.i.d. fit on `N`
    observations. Test-time aggregation is performed per-trial so held-out metrics remain on
    the correct `N` scale, and cross-validation is the practical safeguard against the
    resulting optimism; the bias is shared by both engines' multivariate paths and so does
    not confound the sklearn-vs-pyGAM comparison.

    Splitting Strategies:
    ---------------------
    Continuous target variables are discretized into quantile bins (`y_binned`) to enable stratification.
    - 'session': Uses `StratifiedGroupKFold` (re-seeded per fold) to isolate whole recording days
                 while preserving the target distribution.
    - 'mixed': Uses `StratifiedShuffleSplit` to pool all data and split at the epoch level,
               guaranteeing identical continuous distributions across train and test sets.

    Selection Metric:
    -----------------
    Explained Deviance (D^2). We select features using the one-standard-error (1SE) rule:
    a feature is accepted only if it improves the D^2 by more than the standard error (SE)
    of the candidate model's cross-validated performance.

    Data Persistence:
    -----------------
    Preserves full-resolution metric data. For every candidate tested at
    every step, the raw per-fold results are saved to match the metric set
    emitted by the univariate runner: Explained Deviance (D^2), Residual
    Gamma Deviance, Spearman's rho, Pearson's r, Mean Squared Logarithmic
    Error (MSLE), Mean Absolute Error (MAE), Root Mean Squared Error (RMSE),
    the raw test-fold `y_true` / `y_pred` arrays, and per-fold optimizer
    diagnostics (`n_iter`, `converged`, `fit_time`). `converged=False`
    flags folds that terminated at `max_iter` without meeting the tolerance,
    surfacing the main silent-failure mode of the pyGAM path.

    Parameters
    ----------
    univariate_results_path : str
        Path to the univariate regression results pickle file.
    input_data_path : str
        Path to the raw extracted feature data .pkl file containing (X, y, groups).
    output_directory : str
        The directory path where the resulting step files (.pkl) will be saved.
    settings_path : str, optional
        Path to the 'modeling_settings.json' configuration file.
    target_variable : str, default 'bout_durations'
        The name of the target variable to pull from the input data.
    use_top_rank_as_anchor : bool, default False
        If True, starts the search with the #1 ranked univariate feature.
    p_val : float, default 0.01
        The alpha level used for significance screening against shuffled controls.

    Returns
    -------
    None
        Saves step-wise results and final CV-based filter shapes to disk.
    """

    print(f"--- Starting Regression Model Selection for {target_variable} ---")

    if settings_path is None:
        settings_path_obj = pathlib.Path(__file__).resolve().parent.parent / '_parameter_settings/modeling_settings.json'
        settings_path = str(settings_path_obj)

    try:
        with open(settings_path, 'r') as f:
            settings = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Settings file not found at {settings_path}")

    model_selection_dir = output_directory
    os.makedirs(model_selection_dir, exist_ok=True)
    print(f"Results will be saved to: {model_selection_dir}")

    print(f"Loading univariate results from: {univariate_results_path}")
    print(f"Loading raw data from: {input_data_path}")
    with open(univariate_results_path, 'rb') as f:
        univariate_data = pickle.load(f)

    # 1. Feature Screening & Ranking
    candidates = []
    num_features = len(univariate_data)

    for feat_name, payload in univariate_data.items():
        if isinstance(payload, tuple) and len(payload) == 2:
            _, results = payload
        else:
            results = payload

        if 'actual' not in results or 'explained_deviance' not in results['actual']:
            continue
        if 'null' not in results or 'explained_deviance' not in results['null']:
            continue

        actual_deviance = results['actual']['explained_deviance']
        null_deviance = results['null']['explained_deviance']

        valid_actual = actual_deviance[~np.isnan(actual_deviance)]
        valid_null = null_deviance[~np.isnan(null_deviance)]

        if len(valid_actual) == 0 or len(valid_null) == 0:
            continue

        mean_actual_deviance = np.mean(valid_actual)
        null_threshold = np.percentile(valid_null, q=100 - (p_val / num_features) * 100)

        if mean_actual_deviance > null_threshold and mean_actual_deviance > 0:
            candidates.append({'feature': feat_name, 'mean_explained_deviance': mean_actual_deviance})
        else:
            reason = "Negative D^2" if mean_actual_deviance <= 0 else "Not Significant"
            print(f"  Dropping {feat_name}: {reason} (Mean Dev {mean_actual_deviance:.4f} vs Null {null_threshold:.4f})")

    candidates.sort(key=lambda x: x['mean_explained_deviance'], reverse=True)
    ranked_features = [x['feature'] for x in candidates]

    if not ranked_features:
        print("No significant features found. Aborting.")
        return

    print(f"Identified {len(ranked_features)} significant candidates. Top: {ranked_features[0]}")

    # 2. Raw Data Preparation
    print("Loading raw input data...")
    all_feature_data = load_pickle_modeling_data(input_data_path)
    y_global = all_feature_data[ranked_features[0]]['y']
    groups_global = all_feature_data[ranked_features[0]]['groups']

    camera_rate = settings['io']['camera_sampling_rate']
    filter_history_sec = settings['model_params']['filter_history']
    history_frames = int(np.floor(camera_rate * filter_history_sec))

    # Strict Dictionary Lookup
    model_ops = settings['model_params']
    model_type = model_ops['model_type']
    split_strategy = model_ops['split_strategy']
    n_splits = model_ops['split_num']
    test_prop = model_ops['test_proportion']
    random_seed = settings['model_params']['random_seed']

    print(f"Engine: {model_type.upper()} | Random Seed: {random_seed} | Strategy: {split_strategy} | Num Splits: {n_splits}")

    # Engine-Specific Setup
    basis_matrix = None
    gam_kwargs = {}
    n_splines_time = 0
    n_splines_value = 0
    lr_params = {}

    if model_type == 'sklearn':
        basis_type = model_ops['model_basis_function']
        if basis_type == 'raised_cosine':
            p = settings['hyperparameters']['basis_functions']['raised_cosine']
            kp = int(np.floor(history_frames * p['kpeaks_proportion']))
            basis_matrix = raised_cosine(neye=p['neye'], ncos=p['ncos'], kpeaks=[0, kp], b=p['b'], w=history_frames)
        elif basis_type == 'bspline':
            p = settings['hyperparameters']['basis_functions']['bspline']
            deg = p['degree']
            max_k = max(0, history_frames - deg)
            knots = np.linspace(0, max_k, p['n_splines'] - deg + 1).astype(int)
            basis_matrix = _normalizecols(bsplines(width=history_frames, positions=knots, degree=deg))
        elif basis_type == 'laplacian_pyramid':
            p = settings['hyperparameters']['basis_functions']['laplacian_pyramid']
            basis_matrix = _normalizecols(laplacian_pyramid(width=history_frames, levels=p['levels'], fwhm=p['fwhm']))
        elif basis_type == 'identity':
            basis_matrix = identity(width=history_frames)

        lr_params = settings['hyperparameters']['classical']['ridge_regression']

    elif model_type == 'pygam':
        pygam_params = settings['hyperparameters']['classical']['pygam']
        n_splines_time = pygam_params['n_splines_time']
        n_splines_value = pygam_params['n_splines_value']
        gam_kwargs = {
            'max_iter': int(pygam_params['max_iterations']),
            'tol': float(pygam_params['tol_val']),
            'lam': pygam_params['lam_penalty']
        }
    else:
        raise ValueError("model_type in settings must be either 'sklearn' or 'pygam'.")

    # 3. Target Quantile Binning for Stratification
    n_bins = max(2, min(10, len(y_global) // n_splits))
    bins = np.unique(np.percentile(y_global, np.linspace(0, 100, n_bins + 1)))
    if len(bins) < 2:
        y_binned = np.zeros(len(y_global))
    else:
        y_binned = np.digitize(y_global, bins[1:-1])

    # 4. Generate Splits
    cv_folds = []
    n_folds_mc = max(2, int(round(1.0 / test_prop)))

    if split_strategy == 'session':
        print(f"  > Generating {n_splits} Monte Carlo Session Splits (Test Prop: {test_prop})...")
        for i in range(n_splits):
            current_seed = random_seed + i
            stratified_group_k_fold = StratifiedGroupKFold(n_splits=n_folds_mc, shuffle=True, random_state=current_seed)
            try:
                train_idx, test_idx = next(stratified_group_k_fold.split(np.zeros(len(y_global)), y_binned, groups=groups_global))
                cv_folds.append((train_idx, test_idx))
            except StopIteration:
                pass
    elif split_strategy == 'mixed':
        print(f"  > Generating {n_splits} Monte Carlo Mixed Splits (Test Prop: {test_prop})...")
        sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_prop, random_state=random_seed)
        for train_idx, test_idx in sss.split(np.zeros(len(y_global)), y_binned):
            cv_folds.append((train_idx, test_idx))
    else:
        raise ValueError("split_strategy in settings must be either 'session' or 'mixed'.")

    # Resume Logic
    current_model_features = []
    best_current_score = 0.0
    best_current_se = 0.0
    step_counter = 0

    fname = os.path.basename(univariate_results_path)
    cond_match = re.search(r'((?:male|female).*?)(?=_splits|_lam|_gmm|\.pkl)', fname)
    target_condition = cond_match.group(1) if cond_match else "unknown"

    prefix = f"model_selection_{target_variable}_{target_condition}_{split_strategy}_step_"

    existing_steps = []
    if os.path.exists(model_selection_dir):
        for f_name in os.listdir(model_selection_dir):
            if f_name.startswith(prefix) and f_name.endswith(".pkl"):
                try:
                    existing_steps.append(int(f_name.replace(prefix, "").replace(".pkl", "")))
                except ValueError:
                    pass

    if existing_steps:
        last_step = max(existing_steps)
        print(f"[RESUME] Restoring from Step {last_step}...")
        try:
            with open(os.path.join(model_selection_dir, f"{prefix}{last_step}.pkl"), 'rb') as f:
                last_res = pickle.load(f)
            current_model_features = last_res['current_features']
            best_current_score = last_res['baseline_score']

            if 'candidates_summary' in last_res and last_res['candidates_summary']:
                cand_dict = last_res['candidates_summary']
                best_cand_in_file = max(cand_dict.items(), key=lambda x: x[1]['mean_explained_deviance'])
                name, stats = best_cand_in_file

                if (stats['mean_explained_deviance'] - stats['se_explained_deviance']) > best_current_score:
                    if name not in current_model_features:
                        current_model_features.append(name)
                    best_current_score = stats['mean_explained_deviance']
                    best_current_se = stats['se_explained_deviance']
                    step_counter = last_step + 1
                else:
                    print("[RESUME] Selection already converged. Stopping loop.")
                    step_counter = last_step
        except Exception as e:
            print(f"Resume failed: {e}. Starting fresh.")

    # 5. Auto-Anchor Logic
    if use_top_rank_as_anchor and step_counter == 0:
        anchor = ranked_features[0]
        print(f"\n*** AUTO-ANCHOR: Initializing with {anchor} ***")
        # Metric set is aligned with the univariate runner so downstream
        # plotting / aggregation sees the same keys whether it reads a
        # univariate or a selection-step pickle.
        metrics = {
            'explained_deviance': [], 'residual_deviance': [],
            'spearman_r': [], 'pearson_r': [],
            'msle': [], 'mae': [], 'rmse': [],
            'y_true': [], 'y_pred': [],
            'n_iter': [], 'converged': [], 'fit_time': []
        }

        for tr_idx, te_idx in cv_folds:
            try:
                X_tr, y_tr = all_feature_data[anchor]['X'][tr_idx], y_global[tr_idx]
                X_te, y_te = all_feature_data[anchor]['X'][te_idx], y_global[te_idx]

                if model_type == 'sklearn':
                    X_tr_proj = np.dot(X_tr, basis_matrix)
                    X_te_proj = np.dot(X_te, basis_matrix)

                    y_tr_log = np.log(y_tr + 1e-6)
                    fit_start = time.perf_counter()
                    model = RidgeCV(alphas=lr_params['alphas'], cv=lr_params['cv']).fit(X_tr_proj, y_tr_log)
                    fit_time = float(time.perf_counter() - fit_start)

                    # Jensen bias correction: fitting log(y) with OLS/Ridge targets the median on the
                    # natural scale; for a (conditionally) lognormal model, E[y|X] = exp(Xb + sigma^2/2).
                    # Without the +sigma^2/2 shift, exp(predict(...)) systematically under-estimates the
                    # mean, biasing gamma-deviance / MSLE against the sklearn branch.
                    resid = y_tr_log - model.predict(X_tr_proj)
                    sigma2 = float(np.var(resid, ddof=1))
                    y_pred = np.exp(model.predict(X_te_proj) + 0.5 * sigma2)
                    # RidgeCV has no iterative optimizer to report; the closed-form
                    # solve is trivially converged.
                    fold_n_iter = np.nan
                    fold_converged = True
                else:
                    X_tr_unrolled = np.empty((len(X_tr) * history_frames, 2))
                    X_tr_unrolled[:, 0] = X_tr.ravel()
                    X_tr_unrolled[:, 1] = np.tile(np.arange(history_frames), len(X_tr))

                    gam = GAM(te(0, 1, n_splines=[n_splines_value, n_splines_time]), distribution='gamma', link='log', **gam_kwargs)
                    fit_start = time.perf_counter()
                    gam.fit(X_tr_unrolled, np.repeat(y_tr + 1e-6, history_frames))
                    fit_time = float(time.perf_counter() - fit_start)

                    X_te_unrolled = np.empty((len(X_te) * history_frames, 2))
                    X_te_unrolled[:, 0] = X_te.ravel()
                    X_te_unrolled[:, 1] = np.tile(np.arange(history_frames), len(X_te))

                    # Aggregate the H per-frame predictions on the linear-predictor (eta = log mu) scale
                    # before applying the inverse link: exp(mean(eta)) rather than mean(exp(eta)). This
                    # avoids the Jensen-inequality bias introduced by averaging on the natural (mu) scale,
                    # which would otherwise over-estimate E[y|X] whenever the per-frame eta have any spread.
                    eta_te = np.log(gam.predict_mu(X_te_unrolled)).reshape(len(y_te), history_frames)
                    y_pred = np.exp(np.mean(eta_te, axis=1))
                    gam_diffs = gam.logs_.get('diffs', [])
                    fold_n_iter = float(len(gam_diffs))
                    fold_converged = bool(gam_diffs and gam_diffs[-1] < gam_kwargs['tol'])

                # Robust Metric Calculation
                y_te_safe = np.maximum(y_te, 1e-6)
                y_pred_safe = np.maximum(y_pred, 1e-6)
                y_null_pred = np.full_like(y_te_safe, np.mean(y_te_safe))

                res_dev = mean_gamma_deviance(y_te_safe, y_pred_safe)
                null_dev = mean_gamma_deviance(y_te_safe, y_null_pred)

                d2 = 0.0 if null_dev == 0 else 1 - (res_dev / null_dev)

                metrics['residual_deviance'].append(res_dev)
                metrics['explained_deviance'].append(d2)
                metrics['spearman_r'].append(spearmanr(y_te, y_pred)[0])
                metrics['pearson_r'].append(pearson_r_safe(y_te, y_pred))
                metrics['msle'].append(mean_squared_log_error(y_te_safe, y_pred_safe))
                metrics['mae'].append(mean_absolute_error_1d(y_te, y_pred))
                metrics['rmse'].append(root_mean_squared_error(y_te, y_pred))
                metrics['y_true'].append(y_te_safe)
                metrics['y_pred'].append(y_pred_safe)
                metrics['n_iter'].append(fold_n_iter)
                metrics['converged'].append(fold_converged)
                metrics['fit_time'].append(fit_time)
                gc.collect()

            except Exception as e:
                print(f"    [!] Error fitting: {e}")
                # Append shape-preserving placeholders so every per-fold list
                # stays the same length. y_true / y_pred normally hold test-
                # fold arrays, so on failure we use empty arrays rather than a
                # scalar NaN to keep the list homogeneous for stacking.
                metrics['explained_deviance'].append(np.nan)
                metrics['residual_deviance'].append(np.nan)
                metrics['spearman_r'].append(np.nan)
                metrics['pearson_r'].append(np.nan)
                metrics['msle'].append(np.nan)
                metrics['mae'].append(np.nan)
                metrics['rmse'].append(np.nan)
                metrics['y_true'].append(np.empty((0,), dtype=np.float64))
                metrics['y_pred'].append(np.empty((0,), dtype=np.float64))
                metrics['n_iter'].append(np.nan)
                metrics['converged'].append(False)
                metrics['fit_time'].append(np.nan)

        valid_dev = [m for m in metrics['explained_deviance'] if np.isfinite(m)]
        if valid_dev:
            mean_anchor_score = np.mean(valid_dev)
            if mean_anchor_score > 0:
                best_current_score = mean_anchor_score
                best_current_se = np.std(valid_dev, ddof=1) / np.sqrt(len(valid_dev))
                current_model_features = [anchor]

                step_0_res = {
                    'step_idx': 0, 'current_features': [anchor],
                    'baseline_score': 0.0,
                    'selected_feature': anchor,
                    'candidates_summary': {
                        anchor: {
                            **{k: metrics[k] for k in metrics},
                            'mean_explained_deviance': mean_anchor_score,
                            'se_explained_deviance': best_current_se
                        }
                    }
                }
                with open(os.path.join(model_selection_dir, f"{prefix}0.pkl"), 'wb') as f:
                    pickle.dump(step_0_res, f)
                step_counter = 1
            else:
                print(f"*** ANCHOR FAILED: {anchor} D^2={mean_anchor_score:.4f} <= 0. Reverting to empty start. ***")

    # 6. Forward Selection Loop
    print("\n--- Starting Forward Selection ---")
    while True:
        fold_base_X_tr, fold_base_X_te = [], []
        for tr_idx, te_idx in cv_folds:
            fold_base_X_tr.append([all_feature_data[f]['X'][tr_idx] for f in current_model_features])
            fold_base_X_te.append([all_feature_data[f]['X'][te_idx] for f in current_model_features])

        best_cand, best_cand_score, best_cand_se = None, -float('inf'), 0.0
        step_results = {
            'step_idx': step_counter, 'current_features': list(current_model_features),
            'baseline_score': best_current_score, 'candidates_summary': {},
            'selected_feature': None
        }

        for feat in ranked_features:
            if feat in current_model_features: continue
            gc.collect()
            print(f" Testing +{feat}...", end="", flush=True)

            metrics = {
                'explained_deviance': [], 'residual_deviance': [],
                'spearman_r': [], 'pearson_r': [],
                'msle': [], 'mae': [], 'rmse': [],
                'y_true': [], 'y_pred': [],
                'n_iter': [], 'converged': [], 'fit_time': []
            }

            if model_type == 'pygam':
                gam_terms = te(0, 1, n_splines=[n_splines_value, n_splines_time])
                for i in range(1, len(current_model_features) + 1):
                    gam_terms += te(i * 2, i * 2 + 1, n_splines=[n_splines_value, n_splines_time])

            for fold_idx, (tr_idx, te_idx) in enumerate(cv_folds):
                try:
                    trial_tr = fold_base_X_tr[fold_idx] + [all_feature_data[feat]['X'][tr_idx]]
                    trial_te = fold_base_X_te[fold_idx] + [all_feature_data[feat]['X'][te_idx]]
                    y_tr, y_te = y_global[tr_idx], y_global[te_idx]

                    if model_type == 'sklearn':
                        X_tr_stacked = np.hstack([np.dot(x, basis_matrix) for x in trial_tr])
                        X_te_stacked = np.hstack([np.dot(x, basis_matrix) for x in trial_te])

                        y_tr_log = np.log(y_tr + 1e-6)
                        fit_start = time.perf_counter()
                        model = RidgeCV(alphas=lr_params['alphas'], cv=lr_params['cv']).fit(X_tr_stacked, y_tr_log)
                        fit_time = float(time.perf_counter() - fit_start)

                        # Jensen bias correction on the natural scale (see anchor fit for rationale).
                        resid = y_tr_log - model.predict(X_tr_stacked)
                        sigma2 = float(np.var(resid, ddof=1))
                        y_pred = np.exp(model.predict(X_te_stacked) + 0.5 * sigma2)
                        fold_n_iter = np.nan
                        fold_converged = True
                    else:
                        X_tr_gam = get_unrolled_X_for_multivariate(trial_tr, history_frames)
                        X_te_gam = get_unrolled_X_for_multivariate(trial_te, history_frames)

                        fit_start = time.perf_counter()
                        gam = GAM(gam_terms, distribution='gamma', link='log', **gam_kwargs).fit(X_tr_gam, np.repeat(y_tr + 1e-6, history_frames))
                        fit_time = float(time.perf_counter() - fit_start)

                        # Aggregate the H per-frame predictions on the linear-predictor scale (see anchor fit).
                        eta_te = np.log(gam.predict_mu(X_te_gam)).reshape(len(y_te), history_frames)
                        y_pred = np.exp(np.mean(eta_te, axis=1))
                        gam_diffs = gam.logs_.get('diffs', [])
                        fold_n_iter = float(len(gam_diffs))
                        fold_converged = bool(gam_diffs and gam_diffs[-1] < gam_kwargs['tol'])

                    y_te_safe = np.maximum(y_te, 1e-6)
                    y_pred_safe = np.maximum(y_pred, 1e-6)
                    y_null_pred = np.full_like(y_te_safe, np.mean(y_te_safe))

                    res_dev = mean_gamma_deviance(y_te_safe, y_pred_safe)
                    null_dev = mean_gamma_deviance(y_te_safe, y_null_pred)

                    d2 = 0.0 if null_dev == 0 else 1 - (res_dev / null_dev)

                    metrics['residual_deviance'].append(res_dev)
                    metrics['explained_deviance'].append(d2)
                    metrics['spearman_r'].append(spearmanr(y_te, y_pred)[0])
                    metrics['pearson_r'].append(pearson_r_safe(y_te, y_pred))
                    metrics['msle'].append(mean_squared_log_error(y_te_safe, y_pred_safe))
                    metrics['mae'].append(mean_absolute_error_1d(y_te, y_pred))
                    metrics['rmse'].append(root_mean_squared_error(y_te, y_pred))
                    metrics['y_true'].append(y_te_safe)
                    metrics['y_pred'].append(y_pred_safe)
                    metrics['n_iter'].append(fold_n_iter)
                    metrics['converged'].append(fold_converged)
                    metrics['fit_time'].append(fit_time)

                    gc.collect()
                except Exception as e:
                    print(f"    [!] Error fitting {feat} (Fold {fold_idx}): {e}")
                    metrics['explained_deviance'].append(np.nan)
                    metrics['residual_deviance'].append(np.nan)
                    metrics['spearman_r'].append(np.nan)
                    metrics['pearson_r'].append(np.nan)
                    metrics['msle'].append(np.nan)
                    metrics['mae'].append(np.nan)
                    metrics['rmse'].append(np.nan)
                    # Empty arrays rather than scalar NaN so `y_true` /
                    # `y_pred` stay a list of ndarrays (stackable downstream).
                    metrics['y_true'].append(np.empty((0,), dtype=np.float64))
                    metrics['y_pred'].append(np.empty((0,), dtype=np.float64))
                    metrics['n_iter'].append(np.nan)
                    metrics['converged'].append(False)
                    metrics['fit_time'].append(np.nan)

            valid = [m for m in metrics['explained_deviance'] if np.isfinite(m)]
            if not valid:
                print(" Failed (no finite folds).")
                continue

            m_dev, s_dev = np.mean(valid), np.std(valid, ddof=1) / np.sqrt(len(valid))
            print(f" D^2: {m_dev:.4f}")

            step_results['candidates_summary'][feat] = {
                **{k: metrics[k] for k in metrics},
                'mean_explained_deviance': m_dev,
                'se_explained_deviance': s_dev
            }

            if m_dev > best_cand_score:
                best_cand_score, best_cand_se, best_cand = m_dev, s_dev, feat

        if (best_cand_score - best_cand_se) > best_current_score:
            print(f"  ACCEPT {best_cand}")
            step_results['selected_feature'] = best_cand
            current_model_features.append(best_cand)

            with open(os.path.join(model_selection_dir, f"{prefix}{step_counter}.pkl"), 'wb') as f:
                pickle.dump(step_results, f)

            best_current_score = best_cand_score
            best_current_se = best_cand_se
            step_counter += 1
        else:
            print("  REJECT. Selection Finished.")
            step_results['selected_feature'] = None
            with open(os.path.join(model_selection_dir, f"{prefix}{step_counter}.pkl"), 'wb') as f:
                pickle.dump(step_results, f)
            break

        if len(current_model_features) == len(ranked_features): break

    # 7. Final Model Fit for Visualization
    print("\n--- Final Model Fit for Visualization (CV-based) ---")
    try:
        last_file = os.path.join(model_selection_dir, f"{prefix}{step_counter}.pkl")
        if not os.path.exists(last_file): last_file = os.path.join(model_selection_dir, f"{prefix}{step_counter - 1}.pkl")

        final_fold_shapes = []
        print(f"  Calculating filter shapes across {len(cv_folds)} folds...")

        for fold_idx, (tr_idx, te_idx) in enumerate(cv_folds):

            X_list_tr = [all_feature_data[f]['X'][tr_idx] for f in current_model_features]
            y_tr = y_global[tr_idx]

            fold_res = {}
            if model_type == 'sklearn':
                X_tr_stacked = np.hstack([np.dot(x, basis_matrix) for x in X_list_tr])
                y_tr_log = np.log(y_tr + 1e-6)

                model = RidgeCV(alphas=lr_params['alphas'], cv=lr_params['cv']).fit(X_tr_stacked, y_tr_log)
                coefs = model.coef_.flatten()
                n_bases = basis_matrix.shape[1]

                for k, f_name in enumerate(current_model_features):
                    feat_coefs = coefs[k * n_bases: (k + 1) * n_bases]
                    fold_res[f_name] = np.dot(feat_coefs, basis_matrix.T).flatten()
            else:
                X_tr_gam = get_unrolled_X_for_multivariate(X_list_tr, history_frames)
                y_tr_gam = np.repeat(y_tr + 1e-6, history_frames)

                gam_terms = te(0, 1, n_splines=[n_splines_value, n_splines_time])
                for i in range(1, len(current_model_features)):
                    gam_terms += te(i * 2, i * 2 + 1, n_splines=[n_splines_value, n_splines_time])

                gam_fold = GAM(gam_terms, distribution='gamma', link='log', **gam_kwargs).fit(X_tr_gam, y_tr_gam)

                time_idx = np.arange(history_frames)
                base_grid = np.zeros((history_frames, 2 * len(current_model_features)))
                for k in range(len(current_model_features)): base_grid[:, k * 2 + 1] = time_idx

                pred_base = gam_fold.predict(base_grid)

                for k, feat in enumerate(current_model_features):
                    grid = base_grid.copy()
                    grid[:, k * 2] = 1.0
                    pred_feat = gam_fold.predict(grid)
                    fold_res[feat] = (pred_feat - pred_base).flatten()

                del gam_fold, X_tr_gam, y_tr_gam

            final_fold_shapes.append(fold_res)
            gc.collect()

        final_res = {
            'final_model_features': current_model_features,
            'filter_shapes': final_fold_shapes,
            'univariate_results_path': univariate_results_path,
            'input_data_path': input_data_path
        }

        with open(last_file, 'rb') as open_pkl_file:
            data = pickle.load(open_pkl_file)

        data.update(final_res)

        with open(last_file, 'wb') as save_pkl_file:
            pickle.dump(data, save_pkl_file)

        print("Model selection complete. CV-based shapes saved.")

    except Exception as e:
        print(f"Final fit failed: {e}")


def multinomial_vocal_category_model_selection(
        univariate_results_path: str,
        input_data_path: str,
        output_directory: str,
        settings_path: str = None,
        use_top_rank_as_anchor: bool = False,
        p_val: float = 0.01
) -> None:
    """
    Performs forward stepwise selection for multinomial USV category prediction.

    This function identifies the optimal subset of behavioral features that
    predict the full repertoire of USV categories simultaneously. It uses the
    JAX-accelerated SmoothMultinomialLogisticRegression model.

    Splitting & balancing invariant:
    --------------------------------
    The test fold always preserves the natural class prior of the source data
    (stratified in 'mixed' mode, within-tolerance of global in 'session' mode).
    The training fold follows one of two paths, selected by
    `hyperparameters.jax_linear.multinomial_logistic.balance_train_bool`:

    - `false` (default): the training fold retains the natural class prior,
      and class imbalance is handled inside the JAX loss through softened
      inverse-frequency alpha weights combined with focal-gamma modulation
      (`focal_loss_gamma`).
    - `true`: every learned-model training fold (anchor + forward-selection
      trials) is sample-level down-sampled so each class contributes
      `min(class_count)` rows. The JAX fit is then invoked with
      `focal_gamma=0` and uniform `1 / n_classes` class weights to avoid
      double-correcting an already balanced batch. The `null_model_free`
      baseline keeps the natural-rate training distribution so its empirical
      prior floor remains an informative reference.

    Reported metrics are imbalance-robust (balanced accuracy, log-loss,
    macro OvR AUC).

    Algorithm adaptations for multinomial multi-feature data:
    1.  Data stacking: Instead of unrolling into (Value, Time) columns for pyGAM,
        features are horizontally concatenated (hstack) into a wide matrix of
        shape (N_samples, N_features * N_time_bins).
    2.  Dynamic architecture: The JAX model is re-instantiated at each step with
        `n_features = len(trial_features)` to ensure the temporal smoothing
        penalty (2nd derivative) is applied strictly within each feature's bins.
    3.  Direct filter extraction: Because the model is linear in logit space,
        filter shapes do not require synthetic grid predictions. The learned
        `coef_` matrix is sliced directly to yield the exact temporal filters
        for every USV category.
    4.  Model-Free Baseline (Step 0): Computes the Marginal Class Prior (the
        empirical frequency of each USV category in the training set) as the
        absolute baseline. Features must prove they offer predictive power
        above and beyond simply guessing the majority class.

    Data Persistence:
    -----------------
    At each step of the forward sweep, a .pkl file is saved containing the complete
    experimental state. This deep storage preserves all raw fold-level data to enable
    post-hoc statistical testing and confusion matrix generation without refitting.

    The saved dictionary at each step contains:
    - 'step_idx' (int): The current iteration number.
    - 'current_features' (list): Features selected prior to this step.
    - 'selected_feature' (str or None): The winning feature added in this step.
    - 'baseline_score' (float): The AUC of the current_features model.
    - 'candidates_summary' (dict): Maps each tested feature to its detailed results:
        - 'mean_auc', 'se_auc': Aggregated AUC metrics used for the 1SE rule.
        - 'classes': Array of original USV category string labels.
        - 'folds': A nested dictionary containing lists (length = n_splits) of:
            * 'metrics': dict of the following per-fold scalars —
                - `ll` : multinomial log-loss; strictly proper probabilistic score.
                - `auc` : macro one-vs-rest ROC-AUC; threshold-free ranking quality.
                - `f1` : macro F1; imbalance-robust harmonic mean of precision / recall.
                - `recall` : macro recall (precision is derivable from the saved
                  confusion matrix, so it is intentionally not stored).
                - `score` : balanced accuracy (mean of per-class recall).
                - `brier` : multiclass Brier score; quadratic counterpart to log-loss.
                - `ece` : top-label expected calibration error (10 bins); gap
                  between predicted confidence and empirical accuracy.
                - `mcc` : Matthews correlation coefficient; chance-corrected
                  [-1, +1] summary of the confusion matrix.
            * 'weights': The learned JAX coefficient matrix.
            * 'intercepts': The learned JAX biases.
            * 'y_true', 'y_pred', 'y_probs': Ground truth, hard predictions, and softmax arrays.
            * 'test_indices': The specific dataset rows used in the validation fold.
            * 'confusion_matrix': (K, K) matrix with canonical class ordering.
            * 'p_train', 'p_test': Realized per-class proportions in the fold.
            * 'n_iter', 'converged', 'fit_time': JAX optimizer diagnostics.

    *Note: The final step's file is additionally appended with 'final_model_features'
    and 'weights_reshaped' (the extracted temporal kernels for all included features).*

    Parameters
    ----------
    univariate_results_path : str
        Path to the univariate regression results pickle file.
    input_data_path : str
        Path to the raw extracted feature data .pkl file.
    output_directory : str
        Directory to save the step-wise state dictionaries.
    settings_path : str, optional
        Path to the JSON settings file.
    use_top_rank_as_anchor : bool, default False
        If True, initializes the search by forcing the highest-ranked univariate
        feature as the baseline model.
    p_val : float, default 0.01
        The alpha level used for significance screening against shuffled controls.
    """

    print("--- Starting Multinomial Vocal Category Model Selection ---")

    if settings_path is None:
        settings_path_obj = pathlib.Path(__file__).resolve().parent.parent / '_parameter_settings/modeling_settings.json'
        settings_path = str(settings_path_obj)

    try:
        with open(settings_path, 'r') as f:
            settings = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Settings file not found at {settings_path}")

    model_selection_dir = output_directory
    os.makedirs(model_selection_dir, exist_ok=True)

    with open(univariate_results_path, 'rb') as f:
        univariate_data = pickle.load(f)

    candidates = []
    num_features = len(univariate_data)

    for feat_name, payload in univariate_data.items():
        if isinstance(payload, tuple) and len(payload) == 2:
            _, results = payload
        else:
            results = payload

        if ('actual' not in results or
                'auc' not in results['actual']['folds']['metrics']):
            continue

        actual_auc = np.array(results['actual']['folds']['metrics']['auc'])
        null_auc = np.array(results['null']['folds']['metrics']['auc'])

        valid_actual = actual_auc[~np.isnan(actual_auc)]
        valid_null = null_auc[~np.isnan(null_auc)]

        if len(valid_actual) == 0 or len(valid_null) == 0:
            continue

        mean_actual_auc = np.mean(valid_actual)
        null_threshold = np.percentile(valid_null, q=100 - ((p_val / num_features) * 100))

        if mean_actual_auc > null_threshold:
            candidates.append({'feature': feat_name, 'mean_auc': mean_actual_auc})

    candidates.sort(key=lambda x: x['mean_auc'], reverse=True)
    ranked_features = [x['feature'] for x in candidates]

    if not ranked_features:
        print("No significant features found. Aborting.")
        return

    print("Loading and binning raw input data...")
    with open(input_data_path, 'rb') as f:
        raw_data = pickle.load(f)

    hp = settings['hyperparameters']['jax_linear']['multinomial_logistic']
    bin_size = hp['bin_resizing_factor']

    binned_data = {}
    y_global, groups_global = None, None
    n_time_bins = None

    for feat in ranked_features:
        X_list, y_list, g_list = [], [], []
        sessions = sorted(list(raw_data[feat].keys()))
        for sess_id in sessions:
            X_s = raw_data[feat][sess_id]['X']
            y_s = raw_data[feat][sess_id]['y']

            N, T = X_s.shape
            if bin_size > 1:
                new_T = T // bin_size
                X_s = X_s[:, :new_T * bin_size].reshape(N, new_T, bin_size).mean(axis=2)

            X_list.append(X_s)
            y_list.append(y_s)
            g_list.append(np.full(len(y_s), sess_id))

        binned_data[feat] = np.vstack(X_list).astype(np.float32)
        if y_global is None:
            y_global = np.concatenate(y_list).astype(np.int32)
            groups_global = np.concatenate(g_list)
            n_time_bins = binned_data[feat].shape[1]

    del raw_data
    gc.collect()

    model_ops = settings['model_params']
    split_strategy = model_ops['split_strategy']
    n_splits = model_ops['split_num']
    test_prop = model_ops['test_proportion']
    random_seed = settings['model_params']['random_seed']
    balance_train = hp['balance_train_bool']

    if split_strategy == 'session':
        cv_folds, _ = get_stratified_group_splits_stable(
            groups=groups_global,
            y=y_global,
            test_prop=test_prop,
            n_splits=n_splits,
            random_seed=random_seed,
            max_total_attempts=model_ops['session_split_max_attempts'],
            widen_step=model_ops['session_split_widen_step'],
            widen_every=model_ops['session_split_widen_every']
        )
    elif split_strategy == 'mixed':
        sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_prop, random_state=random_seed)
        cv_folds = list(sss.split(binned_data[ranked_features[0]], y_global))
    else:
        raise ValueError("split_strategy in settings must be either 'session' or 'mixed'.")

    # When balance_train_bool is active the learned-model paths (anchor,
    # forward-selection trials) down-sample each training fold per-class and
    # the JAX fit is switched to focal_gamma=0 + uniform class weights so the
    # already balanced batch is not double-corrected. The null_model_free
    # baseline intentionally keeps the natural-rate training distribution so
    # its empirical-prior floor remains an informative baseline.
    def _fold_train_indices_for_model(tr_idx: np.ndarray, fold_idx: int) -> np.ndarray:
        if not balance_train:
            return tr_idx
        fold_rng = np.random.default_rng(random_seed + fold_idx + 1)
        return _balance_multinomial_train_indices(tr_idx, y_global, fold_rng)

    if balance_train:
        model_focal_gamma = 0.0
        model_uniform_weights = True
    else:
        model_focal_gamma = hp['focal_loss_gamma']
        model_uniform_weights = False

    # Project-wide canonical class order so every per-fold p_train / p_test
    # vector has identical length and index-to-class mapping, enabling direct
    # cross-fold comparison even if a fold happens to miss a rare class.
    canonical_classes = np.unique(y_global)

    def _class_proportions(labels: np.ndarray) -> np.ndarray:
        if len(labels) == 0:
            return np.zeros(len(canonical_classes), dtype=np.float32)
        counts = np.array([(labels == c).sum() for c in canonical_classes], dtype=np.float64)
        return (counts / counts.sum()).astype(np.float32)

    current_model_features = []
    best_current_score = 0.0
    best_current_se = 0.0
    step_counter = 0

    fname = os.path.basename(univariate_results_path)
    # Match the binomial / bout-onset regex grouping: `(?:...)` non-capture on
    # the alternation so conditions like `male_mute_partner` are captured in
    # full. The older `(male|female.*?)` form captured only the literal `male`
    # for any `male_...` filename and truncated the prefix.
    cond_match = re.search(r'((?:male|female).*?)(?=_splits|_lam|_gmm|\.pkl)', fname)
    target_condition = cond_match.group(1) if cond_match else "unknown"
    prefix = f"model_selection_multinomial_vocal_category_{target_condition}_{split_strategy}_step_"

    existing_steps = []
    if os.path.exists(model_selection_dir):
        for f_name in os.listdir(model_selection_dir):
            if f_name.startswith(prefix) and f_name.endswith(".pkl"):
                try:
                    existing_steps.append(int(f_name.replace(prefix, "").replace(".pkl", "")))
                except ValueError:
                    pass

    if existing_steps:
        last_step = max(existing_steps)
        print(f"[RESUME] Restoring from Step {last_step}...")
        try:
            with open(os.path.join(model_selection_dir, f"{prefix}{last_step}.pkl"), 'rb') as f:
                last_res = pickle.load(f)
            current_model_features = last_res['current_features']
            best_current_score = last_res['baseline_score']
            cand_dict = last_res['candidates_summary'] if 'candidates_summary' in last_res else {}

            if cand_dict:
                best_cand_in_file = max(cand_dict.items(), key=lambda x: x[1]['mean_auc'])
                name, stats = best_cand_in_file

                if (stats['mean_auc'] - stats['se_auc']) > best_current_score:
                    if name not in current_model_features:
                        current_model_features.append(name)
                    best_current_score = stats['mean_auc']
                    best_current_se = stats['se_auc']
                    step_counter = last_step + 1
                else:
                    print("[RESUME] Selection already converged. Stopping loop.")
                    step_counter = last_step
        except Exception as e:
            print(f"Resume failed: {e}. Starting fresh.")
            existing_steps = []

    # Calculate Model-Free Baseline (Step 0) if starting fresh
    if not existing_steps:
        print("\n--- Establishing Absolute Baseline (Model-Free Marginal Prior) ---")

        baseline_data = {
            'folds': {
                # `precision` is not stored (derivable from the saved
                # confusion matrix); Brier / ECE / MCC are added as
                # calibration and chance-corrected summary scores.
                'metrics': {m: [] for m in
                            ['auc', 'score', 'recall', 'f1', 'll',
                             'brier', 'ece', 'mcc']},
                'y_true': [], 'y_pred': [], 'y_probs': [], 'test_indices': [],
                'confusion_matrix': []
            },
            'classes': None
        }

        for tr_idx, te_idx in cv_folds:
            y_tr, y_te = y_global[tr_idx], y_global[te_idx]

            unique_classes, counts = np.unique(y_tr, return_counts=True)
            class_priors = counts / float(np.sum(counts))

            y_proba = np.tile(class_priors, (len(y_te), 1))
            majority_class = unique_classes[np.argmax(class_priors)]
            y_pred = np.full(len(y_te), majority_class)

            eps = 1e-15
            y_proba_clipped = np.clip(y_proba, eps, 1 - eps)

            f_met = baseline_data['folds']['metrics']

            try:
                f_auc = roc_auc_score(y_te, y_proba, multi_class='ovr', average='macro', labels=unique_classes)
            except ValueError:
                f_auc = np.nan

            f_met['auc'].append(f_auc)
            f_met['score'].append(balanced_accuracy_score(y_te, y_pred))
            f_met['f1'].append(f1_score(y_te, y_pred, average='macro', zero_division=0))
            f_met['recall'].append(recall_score(y_te, y_pred, average='macro', zero_division=0))

            try:
                f_ll = log_loss(y_te, y_proba_clipped, labels=unique_classes)
            except ValueError:
                f_ll = np.nan
            f_met['ll'].append(f_ll)

            # Canonical-ordered probability matrix for Brier / ECE.
            probs_canonical = np.zeros((len(y_te), len(canonical_classes)), dtype=y_proba.dtype)
            for col_idx, cls in enumerate(unique_classes):
                target_col = int(np.where(canonical_classes == cls)[0][0])
                probs_canonical[:, target_col] = y_proba[:, col_idx]
            try:
                f_met['brier'].append(brier_score_multi(y_te, probs_canonical, canonical_classes))
            except Exception:
                f_met['brier'].append(np.nan)
            try:
                f_met['ece'].append(expected_calibration_error(y_te, y_pred, probs_canonical, n_bins=10))
            except Exception:
                f_met['ece'].append(np.nan)
            f_met['mcc'].append(safe_matthews_corrcoef(y_te, y_pred))

            baseline_data['folds']['y_true'].append(y_te)
            baseline_data['folds']['y_pred'].append(y_pred)
            baseline_data['folds']['y_probs'].append(y_proba)
            baseline_data['folds']['test_indices'].append(te_idx)
            baseline_data['folds']['confusion_matrix'].append(
                safe_confusion_matrix(y_te, y_pred, labels=canonical_classes)
            )
            if baseline_data['classes'] is None:
                baseline_data['classes'] = unique_classes

        valid_auc = [m for m in baseline_data['folds']['metrics']['auc'] if not np.isnan(m)]
        best_current_score = np.mean(valid_auc)
        best_current_se = np.std(valid_auc, ddof=1) / np.sqrt(len(valid_auc))

        print(f"  Baseline Global AUC established at: {best_current_score:.4f}")

        baseline_data['mean_auc'] = best_current_score
        baseline_data['se_auc'] = best_current_se

        step_0_res = {
            'step_idx': 0,
            'current_features': [],
            'baseline_score': best_current_score,
            'selected_feature': 'null_model_free',
            'candidates_summary': {'null_model_free': baseline_data}
        }

        with open(os.path.join(model_selection_dir, f"{prefix}0.pkl"), 'wb') as f:
            pickle.dump(step_0_res, f)

        step_counter = 1

    # Auto-anchor logic
    if use_top_rank_as_anchor and step_counter == 1:
        anchor = ranked_features[0]
        print(f"\n*** AUTO-ANCHOR: Testing {anchor} against Baseline ***")

        cand_data = {
            'folds': {
                'metrics': {m: [] for m in
                            ['auc', 'score', 'recall', 'f1', 'll',
                             'brier', 'ece', 'mcc']},
                'weights': [], 'intercepts': [], 'y_true': [], 'y_pred': [], 'y_probs': [], 'test_indices': [],
                'p_train': [], 'p_test': [],
                'confusion_matrix': [],
                'n_iter': [], 'converged': [], 'fit_time': [],
                'balanced_train': bool(balance_train)
            },
            'classes': None,
            'canonical_classes': canonical_classes
        }

        for fold_idx, (tr_idx, te_idx) in enumerate(cv_folds):
            try:
                tr_idx_model = _fold_train_indices_for_model(tr_idx, fold_idx)
                X_tr, X_te = binned_data[anchor][tr_idx_model], binned_data[anchor][te_idx]
                y_tr, y_te = y_global[tr_idx_model], y_global[te_idx]

                model = SmoothMultinomialLogisticRegression(
                    n_features=1, n_time_bins=n_time_bins,
                    lambda_smooth=hp['lambda_smooth'],
                    l1_reg=hp['l1_reg'],
                    l2_reg=hp['l2_reg'],
                    focal_gamma=model_focal_gamma,
                    uniform_class_weights=model_uniform_weights,
                    learning_rate=hp['learning_rate'],
                    max_iter=hp['max_iter'],
                    tol=hp['tol'],
                    random_state=hp['random_state'] + fold_idx
                )
                model.fit(X_tr, y_tr)

                y_proba = model.predict_proba(X_te, balanced=hp['balance_predictions_bool'])
                y_pred = model.predict(X_te, balanced=hp['balance_predictions_bool'])

                eps = 1e-15
                y_proba_clipped = np.clip(y_proba, eps, 1 - eps)

                f_met = cand_data['folds']['metrics']
                f_met['auc'].append(roc_auc_score(y_te, y_proba, multi_class='ovr', average='macro', labels=model.classes_))
                f_met['score'].append(balanced_accuracy_score(y_te, y_pred))
                f_met['f1'].append(f1_score(y_te, y_pred, average='macro', zero_division=0))
                f_met['recall'].append(recall_score(y_te, y_pred, average='macro', zero_division=0))
                f_met['ll'].append(log_loss(y_te, y_proba_clipped, labels=model.classes_))

                # Canonical-ordered probability matrix for Brier / ECE so
                # missing classes don't misalign columns against `canonical_classes`.
                probs_canonical = np.zeros((len(y_te), len(canonical_classes)), dtype=y_proba.dtype)
                for col_idx, cls in enumerate(model.classes_):
                    target_col = int(np.where(canonical_classes == cls)[0][0])
                    probs_canonical[:, target_col] = y_proba[:, col_idx]
                try:
                    f_met['brier'].append(brier_score_multi(y_te, probs_canonical, canonical_classes))
                except Exception:
                    f_met['brier'].append(np.nan)
                try:
                    f_met['ece'].append(expected_calibration_error(y_te, y_pred, probs_canonical, n_bins=10))
                except Exception:
                    f_met['ece'].append(np.nan)
                f_met['mcc'].append(safe_matthews_corrcoef(y_te, y_pred))

                cand_data['folds']['weights'].append(model.coef_)
                cand_data['folds']['intercepts'].append(model.intercept_)
                cand_data['folds']['y_true'].append(y_te)
                cand_data['folds']['y_pred'].append(y_pred)
                cand_data['folds']['y_probs'].append(y_proba)
                cand_data['folds']['test_indices'].append(te_idx)
                cand_data['folds']['p_train'].append(_class_proportions(y_tr))
                cand_data['folds']['p_test'].append(_class_proportions(y_te))
                cand_data['folds']['confusion_matrix'].append(
                    safe_confusion_matrix(y_te, y_pred, labels=canonical_classes)
                )
                cand_data['folds']['n_iter'].append(int(model.n_iter_))
                cand_data['folds']['converged'].append(bool(model.converged_))
                cand_data['folds']['fit_time'].append(float(model.fit_time_))
                if cand_data['classes'] is None:
                    cand_data['classes'] = model.classes_
            except Exception as e:
                print(f"    [!] Error fitting anchor: {e}")
                # Keep every per-fold list aligned on failure so downstream
                # consumers can safely zip / stack across keys. Metrics get
                # scalar NaNs; confusion_matrix needs a (K, K) NaN placeholder
                # (to stay stackable); weights / intercepts become None; and
                # the fold-level data arrays become empty-but-well-shaped.
                K = len(canonical_classes)
                for _k, _v in cand_data['folds']['metrics'].items():
                    _v.append(np.nan)
                cand_data['folds']['weights'].append(None)
                cand_data['folds']['intercepts'].append(None)
                cand_data['folds']['y_true'].append(np.empty((0,), dtype=np.int32))
                cand_data['folds']['y_pred'].append(np.empty((0,), dtype=np.int32))
                cand_data['folds']['y_probs'].append(np.empty((0, K), dtype=np.float32))
                cand_data['folds']['test_indices'].append(np.empty((0,), dtype=np.int64))
                cand_data['folds']['p_train'].append(np.full(K, np.nan, dtype=np.float32))
                cand_data['folds']['p_test'].append(np.full(K, np.nan, dtype=np.float32))
                cand_data['folds']['confusion_matrix'].append(np.full((K, K), np.nan))
                cand_data['folds']['n_iter'].append(np.nan)
                cand_data['folds']['converged'].append(False)
                cand_data['folds']['fit_time'].append(np.nan)

        valid_auc = [m for m in cand_data['folds']['metrics']['auc'] if np.isfinite(m)]
        if valid_auc:
            mean_anc_auc = np.mean(valid_auc)
            se_anc_auc = np.std(valid_auc, ddof=1) / np.sqrt(len(valid_auc))

            cand_data['mean_auc'] = mean_anc_auc
            cand_data['se_auc'] = se_anc_auc

            if (mean_anc_auc - se_anc_auc) > best_current_score:
                print(f"  *** ANCHOR ACCEPTED: AUC improved to {mean_anc_auc:.4f} ***")
                best_current_score = mean_anc_auc
                best_current_se = se_anc_auc
                current_model_features = [anchor]

                step_1_res = {
                    'step_idx': 1,
                    'current_features': [anchor],
                    'baseline_score': best_current_score,
                    'selected_feature': anchor,
                    'candidates_summary': {anchor: cand_data}
                }
                with open(os.path.join(model_selection_dir, f"{prefix}1.pkl"), 'wb') as f:
                    pickle.dump(step_1_res, f)
                step_counter = 2
            else:
                print(f"  *** ANCHOR REJECTED: Failed to beat spatial baseline. Continuing from Empty Model. ***")

    # Main Forward Selection Loop
    while True:
        print(f"\n=== Step {step_counter} === Best AUC: {best_current_score:.5f}")
        step_results = {
            'step_idx': step_counter,
            'current_features': list(current_model_features),
            'baseline_score': best_current_score,
            'candidates_summary': {},
            'selected_feature': None
        }
        best_cand, best_cand_score, best_cand_se = None, 0.0, 0.0

        for i_feat, feat in enumerate(ranked_features):
            if feat in current_model_features:
                continue
            gc.collect()
            print(f"  [{i_feat + 1}/{len(ranked_features)}] Testing +{feat}...", end="", flush=True)

            trial_feats = current_model_features + [feat]
            n_trial_feats = len(trial_feats)
            cand_data = {
                'folds': {
                    'metrics': {m: [] for m in
                                ['auc', 'score', 'recall', 'f1', 'll',
                                 'brier', 'ece', 'mcc']},
                    'weights': [], 'intercepts': [], 'y_true': [], 'y_pred': [], 'y_probs': [], 'test_indices': [],
                    'p_train': [], 'p_test': [],
                    'confusion_matrix': [],
                    'n_iter': [], 'converged': [], 'fit_time': [],
                    'balanced_train': bool(balance_train)
                },
                'classes': None,
                'canonical_classes': canonical_classes
            }

            for fold_idx, (tr_idx, te_idx) in enumerate(cv_folds):
                try:
                    tr_idx_model = _fold_train_indices_for_model(tr_idx, fold_idx)
                    X_tr_stacked = np.hstack([binned_data[f][tr_idx_model] for f in trial_feats])
                    X_te_stacked = np.hstack([binned_data[f][te_idx] for f in trial_feats])
                    y_tr, y_te = y_global[tr_idx_model], y_global[te_idx]

                    model = SmoothMultinomialLogisticRegression(
                        n_features=n_trial_feats, n_time_bins=n_time_bins,
                        lambda_smooth=hp['lambda_smooth'],
                        l1_reg=hp['l1_reg'],
                        # Divide L2 by sqrt(n_trial_feats): the feature trial stacks
                        # n_trial_feats columns of `n_time_bins` into a wide design
                        # matrix, so without this rescale the summed L2 penalty on
                        # `W ** 2` would grow ~linearly with feature count and silently
                        # over-regularize larger models. Scaling by 1/sqrt(n_feats)
                        # keeps the per-feature penalty magnitude roughly constant as
                        # the selector adds features.
                        l2_reg=hp['l2_reg'] / np.sqrt(n_trial_feats),
                        focal_gamma=model_focal_gamma,
                        uniform_class_weights=model_uniform_weights,
                        learning_rate=hp['learning_rate'],
                        max_iter=hp['max_iter'],
                        tol=hp['tol'],
                        random_state=hp['random_state'] + fold_idx
                    )
                    model.fit(X_tr_stacked, y_tr)

                    y_proba = model.predict_proba(X_te_stacked, balanced=hp['balance_predictions_bool'])
                    y_pred = model.predict(X_te_stacked, balanced=hp['balance_predictions_bool'])

                    eps = 1e-15
                    y_proba_clipped = np.clip(y_proba, eps, 1 - eps)

                    f_met = cand_data['folds']['metrics']
                    f_met['auc'].append(roc_auc_score(y_te, y_proba, multi_class='ovr', average='macro', labels=model.classes_))
                    f_met['score'].append(balanced_accuracy_score(y_te, y_pred))
                    f_met['f1'].append(f1_score(y_te, y_pred, average='macro', zero_division=0))
                    f_met['recall'].append(recall_score(y_te, y_pred, average='macro', zero_division=0))
                    f_met['ll'].append(log_loss(y_te, y_proba_clipped, labels=model.classes_))

                    # Canonical-ordered probability matrix so Brier / ECE are
                    # computed against a stable column ordering across folds
                    # even when a rare class is absent from `model.classes_`.
                    probs_canonical = np.zeros((len(y_te), len(canonical_classes)), dtype=y_proba.dtype)
                    for col_idx, cls in enumerate(model.classes_):
                        target_col = int(np.where(canonical_classes == cls)[0][0])
                        probs_canonical[:, target_col] = y_proba[:, col_idx]
                    try:
                        f_met['brier'].append(brier_score_multi(y_te, probs_canonical, canonical_classes))
                    except Exception:
                        f_met['brier'].append(np.nan)
                    try:
                        f_met['ece'].append(expected_calibration_error(y_te, y_pred, probs_canonical, n_bins=10))
                    except Exception:
                        f_met['ece'].append(np.nan)
                    f_met['mcc'].append(safe_matthews_corrcoef(y_te, y_pred))

                    cand_data['folds']['weights'].append(model.coef_)
                    cand_data['folds']['intercepts'].append(model.intercept_)
                    cand_data['folds']['y_true'].append(y_te)
                    cand_data['folds']['y_pred'].append(y_pred)
                    cand_data['folds']['y_probs'].append(y_proba)
                    cand_data['folds']['test_indices'].append(te_idx)
                    cand_data['folds']['p_train'].append(_class_proportions(y_tr))
                    cand_data['folds']['p_test'].append(_class_proportions(y_te))
                    cand_data['folds']['confusion_matrix'].append(
                        safe_confusion_matrix(y_te, y_pred, labels=canonical_classes)
                    )
                    cand_data['folds']['n_iter'].append(int(model.n_iter_))
                    cand_data['folds']['converged'].append(bool(model.converged_))
                    cand_data['folds']['fit_time'].append(float(model.fit_time_))
                    if cand_data['classes'] is None:
                        cand_data['classes'] = model.classes_
                except Exception:
                    # Append placeholders to every sibling list so per-fold
                    # lengths stay aligned across `metrics`, the fold-level
                    # data arrays, and the optimizer-diagnostic flags.
                    K = len(canonical_classes)
                    for _k, _v in cand_data['folds']['metrics'].items():
                        _v.append(np.nan)
                    cand_data['folds']['weights'].append(None)
                    cand_data['folds']['intercepts'].append(None)
                    cand_data['folds']['y_true'].append(np.empty((0,), dtype=np.int32))
                    cand_data['folds']['y_pred'].append(np.empty((0,), dtype=np.int32))
                    cand_data['folds']['y_probs'].append(np.empty((0, K), dtype=np.float32))
                    cand_data['folds']['test_indices'].append(np.empty((0,), dtype=np.int64))
                    cand_data['folds']['p_train'].append(np.full(K, np.nan, dtype=np.float32))
                    cand_data['folds']['p_test'].append(np.full(K, np.nan, dtype=np.float32))
                    cand_data['folds']['confusion_matrix'].append(np.full((K, K), np.nan))
                    cand_data['folds']['n_iter'].append(np.nan)
                    cand_data['folds']['converged'].append(False)
                    cand_data['folds']['fit_time'].append(np.nan)

            valid_auc = [x for x in cand_data['folds']['metrics']['auc'] if np.isfinite(x)]
            if not valid_auc:
                print(" Failed (no finite folds).")
                continue

            mean_auc = np.mean(valid_auc)
            se_auc = np.std(valid_auc, ddof=1) / np.sqrt(len(valid_auc))
            print(f" AUC: {mean_auc:.4f}")
            cand_data['mean_auc'], cand_data['se_auc'] = mean_auc, se_auc
            step_results['candidates_summary'][feat] = cand_data
            if mean_auc > best_cand_score:
                best_cand_score, best_cand_se, best_cand = mean_auc, se_auc, feat

        if (best_cand and (best_cand_score - best_current_score) > best_cand_se):
            print(f"  ACCEPT {best_cand}")
            step_results['selected_feature'] = best_cand
            current_model_features.append(best_cand)
            with open(os.path.join(model_selection_dir, f"{prefix}{step_counter}.pkl"), 'wb') as f:
                pickle.dump(step_results, f)
            best_current_score, best_current_se, step_counter = best_cand_score, best_cand_se, step_counter + 1
        else:
            print("  REJECT. Selection Finished.")
            step_results['selected_feature'] = None
            with open(os.path.join(model_selection_dir, f"{prefix}{step_counter}.pkl"), 'wb') as f:
                pickle.dump(step_results, f)
            break

        if len(current_model_features) == len(ranked_features):
            break

    # 7. Final Data Promotion for Visualization
    print("\n--- Finalizing Results for Visualization ---")
    try:
        last_file_path = os.path.join(model_selection_dir, f"{prefix}{max(0, step_counter - 1)}.pkl")
        with open(last_file_path, 'rb') as f:
            step_data = pickle.load(f)

        winner = step_data['selected_feature'] or current_model_features[-1]
        winning_cand_data = step_data['candidates_summary'][winner]

        # Only process reshaping if the winning candidate was a real model with weights
        if winner != 'null_model_free':
            raw_weights = np.array(winning_cand_data['folds']['weights'])
            n_folds, n_classes, _ = raw_weights.shape
            n_final_feats = len(current_model_features)
            reshaped_weights = raw_weights.reshape(n_folds, n_classes, n_final_feats, n_time_bins)
        else:
            reshaped_weights = None

        step_data.update({
            'final_model_features': current_model_features,
            'weights_reshaped': reshaped_weights,
            'classes': winning_cand_data['classes']
        })
        with open(last_file_path, 'wb') as f:
            pickle.dump(step_data, f)

        print("Final model configuration successfully saved.")
    except Exception as e:
        print(f"Final promotion failed: {e}")

def continuous_vocal_manifold_model_selection(
        univariate_results_path: str,
        input_data_path: str,
        output_directory: str,
        settings_path: str = None,
        use_top_rank_as_anchor: bool = False,
        p_val: float = 0.05
) -> None:
    """
    Performs forward stepwise selection for continuous UMAP-position
    prediction using the `SmoothBivariateRegression` estimator.

    The selector identifies the minimal set of behavioural features that
    jointly predict the `(x, y)` UMAP coordinates of upcoming USVs.
    Candidates are ranked by `r2_spatial` (the coefficient of
    determination pooled across UMAP axes) — higher is better,
    interpretable as the fraction of test-fold spatial variance the
    model explains above the test-fold marginal mean.

    Key algorithmic choices
    -----------------------
    1. Manifold-bounded predictions. Every candidate prediction is
       snapped to the nearest observed training UMAP point before
       metrics are computed, so the active model and the baselines
       compete on the same support and the metric magnitudes are
       directly comparable (see `SmoothBivariateRegression.predict`).
    2. Wilcoxon screening (higher-is-better). Candidates are ranked by a
       Bonferroni-corrected one-sided Wilcoxon signed-rank test on the
       paired per-fold `r2_spatial` of `actual` vs. `null` (the within-
       session X-history shuffle).
    3. Spatial stratification. Folds come from
       `get_stratified_spatial_splits_stable`, which uses deterministic
       K-Means geographic clustering so rare acoustic satellites are
       proportionally represented across train / test halves under both
       `session` and `mixed` strategies.
    4. Inverse-density weighting. The pre-computed KDE spatial weights
       `w` are forwarded to the JAX optimiser so gradient updates give
       satellite vocalisations the same pull as dense-core bouts.
    5. Spatial-centroid baseline (Step 0). `null_model_free` predicts
       the KDE-weighted training centroid for every test trial — the
       absolute floor any model with real signal must clear.
    6. 1SE rule on R². Because R² is higher-is-better, a feature is
       added only when `(best_cand_score - best_current_score) >
       best_cand_se`, i.e. the candidate's mean score beats the current
       best by more than one SE of its own per-fold distribution.

    Joint per-fold hyperparameter tuning
    ------------------------------------
    When `hyperparameters.jax_linear.bivariate.tune_regularization_bool`
    is `true`, every candidate fold (anchor + every forward-selection
    trial) runs its own joint inner CV over the log-spaced
    `(lambda_smooth, l2_reg)` grids before the outer fit. The l2 grid is
    rescaled by `1 / sqrt(n_trial_features)` so the search window tracks
    the same effective regularisation strength as the fixed fallback
    (which is rescaled the same way). The winning pair, the full grid of
    inner-CV scores, and a `hyperparams_tuned` flag are persisted per
    candidate per fold alongside the filter weights. Setting the flag to
    `false` preserves the legacy single-fixed-centre behaviour and
    writes the same schema.

    Deep storage for post-hoc visualisation
    ---------------------------------------
    Every step persists, for each candidate and fold: the full
    `evaluate_metrics` bundle, the learned `coef_` / `intercept_`
    matrices, the test coordinates `y_true`, the manifold-snapped
    predictions `y_pred_xy`, the test-fold weights `w_test`, the per-
    fold JAX optimiser diagnostics (`n_iter`, `converged`, `fit_time`),
    and the hyperparameter audit trail (`selected_lambda_smooth`,
    `selected_l2_reg`, `hyperparam_grid_scores`, `hyperparams_tuned`).

    Parameters
    ----------
    univariate_results_path : str
        Path to the univariate regression results pickle file containing the
        paired actual / null per-fold metric arrays.
    input_data_path : str
        Path to the extracted UMAP data containing X (history), Y (UMAP),
        and w (KDE spatial weights).
    output_directory : str
        Directory to save the step-wise state dictionaries.
    settings_path : str, optional
        Path to the JSON settings file.
    use_top_rank_as_anchor : bool, default False
        If True, initialises the search by forcing the single highest-ranked
        Wilcoxon feature as Step 1.
    p_val : float, default 0.05
        The overall alpha level, Bonferroni-corrected by dividing it by the
        number of evaluated features.
    """

    print("--- Starting Continuous Vocal Manifold Model Selection ---")

    if settings_path is None:
        settings_path_obj = pathlib.Path(__file__).resolve().parent.parent / '_parameter_settings/modeling_settings.json'
        settings_path = str(settings_path_obj)

    try:
        with open(settings_path, 'r') as f:
            settings = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Settings file not found at {settings_path}")

    model_selection_dir = output_directory
    os.makedirs(model_selection_dir, exist_ok=True)
    print(f"Results will be saved to: {model_selection_dir}")

    print(f"Loading univariate results from: {univariate_results_path}")
    with open(univariate_results_path, 'rb') as f:
        univariate_data = pickle.load(f)

    num_features = len(univariate_data)
    alpha_corrected = p_val / num_features
    candidates = []

    # `r2_spatial` is the headline screening score: it's directly
    # comparable across features / sex groups, bounded above by 1, and
    # universally interpretable as "fraction of test-fold spatial
    # variance explained above the marginal mean". Higher is better, so
    # the one-sided Wilcoxon alternative is `greater` (actual beats the
    # within-session X-history shuffled null).
    SCREENING_METRIC = 'r2_spatial'

    print(f"Screening {num_features} features (Bonferroni alpha = {alpha_corrected:.2e})...")
    for feat_name, payload in univariate_data.items():
        if isinstance(payload, tuple) and len(payload) == 2:
            _, results = payload
        else:
            results = payload

        if 'actual' not in results:
            continue

        actual_metrics = results['actual']['folds']['metrics']
        null_metrics = results['null']['folds']['metrics']
        if SCREENING_METRIC not in actual_metrics or SCREENING_METRIC not in null_metrics:
            continue

        actual_score = np.array(actual_metrics[SCREENING_METRIC])
        null_score = np.array(null_metrics[SCREENING_METRIC])

        valid_actual = actual_score[~np.isnan(actual_score)]
        valid_null = null_score[~np.isnan(null_score)]

        if len(valid_actual) == 0 or len(valid_null) == 0:
            continue

        try:
            _, p_value = wilcoxon(valid_actual, valid_null, alternative='greater')
        except ValueError:
            p_value = 1.0

        if p_value < alpha_corrected:
            mean_score = float(np.mean(valid_actual))
            candidates.append({
                'feature': feat_name,
                'mean_r2': mean_score,
                'p_val': p_value,
            })

    # Rank by descending R^2 (best features first).
    candidates.sort(key=lambda x: x['mean_r2'], reverse=True)
    ranked_features = [x['feature'] for x in candidates]

    if not ranked_features:
        print(f"No significant features found (p < {alpha_corrected:.2e}). Aborting.")
        return

    print(f"Identified {len(ranked_features)} significant candidates. Top: {ranked_features[0]}")

    print("Loading and binning raw continuous input data...")
    with open(input_data_path, 'rb') as f:
        raw_data = pickle.load(f)

    hp = settings['hyperparameters']['jax_linear']['bivariate']
    bin_size = hp['bin_resizing_factor']

    binned_data = {}
    y_global, w_global, groups_global = None, None, None
    n_time_bins = None

    for feat in ranked_features:
        X_list, y_list, w_list, g_list = [], [], [], []
        sessions = sorted(list(raw_data[feat].keys()))
        for sess_id in sessions:
            X_s = raw_data[feat][sess_id]['X']
            y_s = raw_data[feat][sess_id]['Y']
            w_s = raw_data[feat][sess_id]['w']

            N, T = X_s.shape
            if bin_size > 1:
                new_T = T // bin_size
                X_s = X_s[:, :new_T * bin_size].reshape(N, new_T, bin_size).mean(axis=2)

            X_list.append(X_s)
            y_list.append(y_s)
            w_list.append(w_s)
            g_list.append(np.full(len(y_s), sess_id))

        binned_data[feat] = np.vstack(X_list).astype(np.float32)
        if y_global is None:
            y_global = np.vstack(y_list).astype(np.float32)
            w_global = np.concatenate(w_list).astype(np.float32)
            groups_global = np.concatenate(g_list)
            n_time_bins = binned_data[feat].shape[1]

    del raw_data
    gc.collect()

    model_ops = settings['model_params']
    n_splits = model_ops['split_num']
    test_prop = model_ops['test_proportion']
    n_clusters = model_ops['spatial_cluster_num']
    split_strategy = model_ops['split_strategy']
    random_seed = settings['model_params']['random_seed']

    # Joint-tuning configuration. Mirrors the runner: fixed fallback values
    # are the grid centres; the `tune_regularization_bool` flag flips
    # whether each outer fold / candidate runs an inner CV to pick the
    # `(λ_smooth, l2_reg)` pair. Keeping both regularisers in the same
    # search guarantees the selector behaves consistently — the current
    # active model, the anchor, and every forward-selection trial all use
    # hyperparameters chosen by their own inner CV on their own training
    # fold.
    tune_regularization_bool = hp['tune_regularization_bool']
    tune_params = hp['tune_regularization_params']
    lambda_smooth_grid = _log_spaced_grid(
        center=hp['lambda_smooth_fixed'],
        decades_each_side=tune_params['lambda_smooth_decades_each_side'],
    )
    l2_reg_grid = _log_spaced_grid(
        center=hp['l2_reg_fixed'],
        decades_each_side=tune_params['l2_reg_decades_each_side'],
    )
    inner_cv_folds = tune_params['inner_cv_folds']
    inner_cv_scoring_metric = tune_params['inner_cv_scoring_metric']

    print(f"Random Seed: {random_seed} | Num Splits: {n_splits} | Split Strategy: Spatial Proxy ({split_strategy.upper()})")
    if tune_regularization_bool:
        print(
            f"  Joint per-fold tuning ENABLED: |λ_smooth grid|={len(lambda_smooth_grid)}, "
            f"|l2_reg grid|={len(l2_reg_grid)}, inner CV folds={inner_cv_folds}, "
            f"scoring={inner_cv_scoring_metric}"
        )
    else:
        print(
            f"  Joint per-fold tuning DISABLED (fixed λ_smooth={hp['lambda_smooth_fixed']}, "
            f"fixed l2_reg={hp['l2_reg_fixed']})"
        )

    cv_folds = get_stratified_spatial_splits_stable(
        groups=groups_global,
        Y=y_global,
        n_clusters=n_clusters,
        test_prop=test_prop,
        n_splits=n_splits,
        split_strategy=split_strategy,
        random_seed=random_seed
    )

    current_model_features = []
    # R^2 is higher-is-better, so the pre-Step-0 "best" is `-inf` (any
    # finite baseline beats it); 1SE acceptance becomes
    # `(best_cand_score - best_current_score) > best_cand_se`.
    best_current_score = float('-inf')
    best_current_se = 0.0
    step_counter = 0

    fname = os.path.basename(univariate_results_path)
    # Use a non-capturing alternation so conditions like `male_mute_partner`
    # are captured in full rather than truncated to `male`.
    cond_match = re.search(r'((?:male|female).*?)(?=_splits|_lam|_gmm|\.pkl)', fname)
    target_condition = cond_match.group(1) if cond_match else "unknown"

    prefix = f"model_selection_continuous_manifold_{target_condition}_{split_strategy}_step_"

    existing_steps = []
    if os.path.exists(model_selection_dir):
        for f_name in os.listdir(model_selection_dir):
            if f_name.startswith(prefix) and f_name.endswith(".pkl"):
                try:
                    existing_steps.append(int(f_name.replace(prefix, "").replace(".pkl", "")))
                except ValueError:
                    pass

    if existing_steps:
        last_step = max(existing_steps)
        print(f"[RESUME] Restoring from Step {last_step}...")
        try:
            with open(os.path.join(model_selection_dir, f"{prefix}{last_step}.pkl"), 'rb') as f:
                last_res = pickle.load(f)
            current_model_features = last_res['current_features']
            best_current_score = last_res['baseline_score']

            if 'candidates_summary' in last_res and last_res['candidates_summary']:
                cand_dict = last_res['candidates_summary']
                # Best-of-file is the candidate with highest mean R^2.
                best_cand_in_file = max(
                    cand_dict.items(),
                    key=lambda x: x[1]['mean_r2'],
                )
                name, stats = best_cand_in_file

                if (stats['mean_r2'] - best_current_score) > stats['se_r2']:
                    if name not in current_model_features:
                        current_model_features.append(name)
                    best_current_score = stats['mean_r2']
                    best_current_se = stats['se_r2']
                    step_counter = last_step + 1
                else:
                    print("[RESUME] Selection already converged. Stopping loop.")
                    step_counter = last_step
        except Exception as e:
            print(f"Resume failed: {e}. Starting fresh.")
            existing_steps = []

    # Canonical ordering of the metric keys so the dict initialiser and the
    # downstream aggregation stay in lockstep with the estimator output.
    # `r2_spatial` is first because it is the selection score.
    MANIFOLD_METRIC_KEYS = [
        'r2_spatial',
        'euclidean_mae',
        'euclidean_rmse',
        'euclidean_mae_weighted',
        'euclidean_mae_raw',
        'mahalanobis_mae',
        'mae_x',
        'mae_y',
        'pearson_x',
        'pearson_y',
        'spearman_x',
        'spearman_y',
    ]

    # Calculate Model-Free Baseline (Step 0) if starting fresh.
    if not existing_steps:
        print("\n--- Establishing Absolute Baseline (Spatial-Centroid Prior) ---")

        baseline_data = {
            'folds': {
                'metrics': {m: [] for m in MANIFOLD_METRIC_KEYS},
                'test_indices': [],
                'y_true': [],
                'w_test': [],
                # `(x, y)` predictions per fold: the KDE-weighted centroid of
                # the training set tiled across the test fold.
                'y_pred_xy': [],
                # Hyperparameter audit fields — the centroid baseline has
                # no model, so these are all NaN / empty / False. Kept so
                # the saved schema is uniform across strategies.
                'selected_lambda_smooth': [],
                'selected_l2_reg': [],
                'hyperparam_grid_scores': [],
                'hyperparams_tuned': [],
            }
        }

        for fold_idx, (tr_idx, te_idx) in enumerate(cv_folds):
            Y_tr, Y_te = y_global[tr_idx], y_global[te_idx]
            w_tr, w_te = w_global[tr_idx], w_global[te_idx]

            mu = np.average(Y_tr, axis=0, weights=w_tr)

            dx = Y_te[:, 0] - mu[0]
            dy = Y_te[:, 1] - mu[1]
            euclidean_dist = np.sqrt(dx ** 2 + dy ** 2)
            sse = np.sum(dx ** 2 + dy ** 2)
            ss_tot_x = np.sum((Y_te[:, 0] - np.mean(Y_te[:, 0])) ** 2)
            ss_tot_y = np.sum((Y_te[:, 1] - np.mean(Y_te[:, 1])) ** 2)
            denom = ss_tot_x + ss_tot_y

            # Mahalanobis MAE under the KDE-weighted training covariance
            # of Y_tr — matches the regressor's convention so the baseline
            # and the active models compete on the same distance metric.
            w_cov = w_tr / (np.sum(w_tr) + 1e-12)
            diff_tr = Y_tr - mu
            cov_tr = (w_cov[:, None] * diff_tr).T @ diff_tr
            cov_inv = np.linalg.pinv(cov_tr)
            residual = np.stack([dx, dy], axis=1)
            quad = np.einsum('ij,jk,ik->i', residual, cov_inv, residual)
            mahalanobis_mae_val = float(np.mean(np.sqrt(np.maximum(quad, 0.0))))

            mae_val = float(np.mean(euclidean_dist))
            f_met = baseline_data['folds']['metrics']
            f_met['r2_spatial'].append(float(1.0 - (sse / denom)) if denom > 0 else 0.0)
            f_met['euclidean_mae'].append(mae_val)
            f_met['euclidean_rmse'].append(float(np.sqrt(np.mean(euclidean_dist ** 2))))
            f_met['euclidean_mae_weighted'].append(
                float(np.sum(w_te * euclidean_dist) / (np.sum(w_te) + 1e-12))
            )
            # Centroid prediction is constant and already on-manifold, so
            # raw and snapped MAE coincide.
            f_met['euclidean_mae_raw'].append(mae_val)
            f_met['mahalanobis_mae'].append(mahalanobis_mae_val)
            f_met['mae_x'].append(float(np.mean(np.abs(dx))))
            f_met['mae_y'].append(float(np.mean(np.abs(dy))))
            # Constant predictions make per-axis correlations undefined;
            # emit NaN to match the regressor's NaN-safe convention.
            f_met['pearson_x'].append(float('nan'))
            f_met['pearson_y'].append(float('nan'))
            f_met['spearman_x'].append(float('nan'))
            f_met['spearman_y'].append(float('nan'))

            y_pred_xy = np.tile(mu.astype(np.float32), (len(Y_te), 1))

            baseline_data['folds']['test_indices'].append(te_idx)
            baseline_data['folds']['y_true'].append(Y_te)
            baseline_data['folds']['w_test'].append(w_te)
            baseline_data['folds']['y_pred_xy'].append(y_pred_xy)
            baseline_data['folds']['selected_lambda_smooth'].append(float('nan'))
            baseline_data['folds']['selected_l2_reg'].append(float('nan'))
            baseline_data['folds']['hyperparam_grid_scores'].append({})
            baseline_data['folds']['hyperparams_tuned'].append(False)

        baseline_scores = np.asarray(
            baseline_data['folds']['metrics']['r2_spatial'], dtype=float
        )
        valid_scores = baseline_scores[~np.isnan(baseline_scores)]
        best_current_score = (
            float(np.mean(valid_scores)) if valid_scores.size else float('-inf')
        )
        best_current_se = (
            float(np.std(valid_scores, ddof=1) / np.sqrt(valid_scores.size))
            if valid_scores.size > 1 else 0.0
        )

        print(f"  Baseline R^2 (centroid) established at: {best_current_score:.4f}")

        baseline_data['mean_r2'] = best_current_score
        baseline_data['se_r2'] = best_current_se

        step_0_res = {
            'step_idx': 0,
            'current_features': [],
            'baseline_score': best_current_score,
            'selected_feature': 'null_model_free',
            'candidates_summary': {'null_model_free': baseline_data}
        }

        with open(os.path.join(model_selection_dir, f"{prefix}0.pkl"), 'wb') as f:
            pickle.dump(step_0_res, f)

        step_counter = 1

    def _make_manifold_cand_data():
        return {
            'folds': {
                'metrics': {m: [] for m in MANIFOLD_METRIC_KEYS},
                'weights': [],
                'intercepts': [],
                'test_indices': [],
                'y_true': [],
                'w_test': [],
                'y_pred_xy': [],
                'n_iter': [],
                'converged': [],
                'fit_time': [],
                # Mirror of the runner's schema so the selection-step pickles
                # and the univariate pickles carry the same hyperparameter
                # audit trail. Both tuned and fixed runs populate these
                # keys; `null_model_free` writes NaN.
                'selected_lambda_smooth': [],
                'selected_l2_reg': [],
                'hyperparam_grid_scores': [],
                'hyperparams_tuned': [],
            }
        }

    def _append_failed_fold(cand_data_):
        """Append shape-preserving placeholders so per-fold lists stay aligned."""
        for _mk in cand_data_['folds']['metrics']:
            cand_data_['folds']['metrics'][_mk].append(np.nan)
        cand_data_['folds']['weights'].append(None)
        cand_data_['folds']['intercepts'].append(None)
        cand_data_['folds']['test_indices'].append(np.empty((0,), dtype=np.int64))
        cand_data_['folds']['y_true'].append(np.empty((0, 2), dtype=np.float32))
        cand_data_['folds']['w_test'].append(np.empty((0,), dtype=np.float32))
        cand_data_['folds']['y_pred_xy'].append(np.empty((0, 2), dtype=np.float32))
        cand_data_['folds']['n_iter'].append(np.nan)
        cand_data_['folds']['converged'].append(False)
        cand_data_['folds']['fit_time'].append(np.nan)
        cand_data_['folds']['selected_lambda_smooth'].append(float('nan'))
        cand_data_['folds']['selected_l2_reg'].append(float('nan'))
        cand_data_['folds']['hyperparam_grid_scores'].append({})
        cand_data_['folds']['hyperparams_tuned'].append(False)

    def _pick_fold_hyperparams(X_tr_, Y_tr_, w_tr_, groups_tr_, n_feats_, fold_idx_):
        """
        Returns `(lambda_smooth, l2_reg, grid_scores, tuned_flag)` for this
        fold and trial feature set.

        When `tune_regularization_bool` is False the fixed centres are used
        directly (with the standard `1 / sqrt(n_features)` rescale on the
        l2 centre so larger trial models don't silently over-regularise).
        When tuning is on, the inner-CV routine searches the Cartesian
        product of the settings-level grids, with the l2 grid *also*
        rescaled by `1 / sqrt(n_features)` so the search window tracks
        the same effective regularisation strength as the fixed fallback.
        """
        if not tune_regularization_bool:
            lam_sm = float(hp['lambda_smooth_fixed'])
            lam_l2 = float(hp['l2_reg_fixed'] / np.sqrt(n_feats_))
            return lam_sm, lam_l2, {}, False

        rescaled_l2_grid = l2_reg_grid / np.sqrt(n_feats_)
        lam_sm_win, lam_l2_win, grid_scores_ = _tune_manifold_regularization(
            X_train=X_tr_,
            Y_train=Y_tr_,
            w_train=w_tr_,
            groups_train=groups_tr_,
            lambda_smooth_grid=lambda_smooth_grid,
            l2_reg_grid=rescaled_l2_grid,
            inner_cv_folds=inner_cv_folds,
            inner_cv_scoring_metric=inner_cv_scoring_metric,
            n_features=n_feats_,
            n_time_bins=n_time_bins,
            spatial_cluster_num=n_clusters,
            huber_delta=hp['huber_delta'],
            learning_rate=hp['learning_rate'],
            max_iter=hp['max_iter'],
            tol=hp['tol'],
            random_state=hp['random_state'] + fold_idx_,
            verbose=False,
            regressor_cls=SmoothBivariateRegression,
        )
        return lam_sm_win, lam_l2_win, grid_scores_, True

    # Auto-anchor logic
    if use_top_rank_as_anchor and step_counter == 1:
        anchor = ranked_features[0]
        print(f"\n*** AUTO-ANCHOR: Testing {anchor} against Baseline ***")

        cand_data = _make_manifold_cand_data()

        for fold_idx, (tr_idx, te_idx) in enumerate(cv_folds):
            try:
                X_tr, X_te = binned_data[anchor][tr_idx], binned_data[anchor][te_idx]
                Y_tr, Y_te = y_global[tr_idx], y_global[te_idx]
                w_tr, w_te = w_global[tr_idx], w_global[te_idx]
                groups_tr = groups_global[tr_idx]

                fold_lambda_smooth, fold_l2_reg, fold_grid_scores, fold_tuned_flag = (
                    _pick_fold_hyperparams(
                        X_tr, Y_tr, w_tr, groups_tr,
                        n_feats_=1,
                        fold_idx_=fold_idx,
                    )
                )

                model = SmoothBivariateRegression(
                    n_features=1, n_time_bins=n_time_bins,
                    lambda_smooth=fold_lambda_smooth, l2_reg=fold_l2_reg,
                    huber_delta=hp['huber_delta'],
                    learning_rate=hp['learning_rate'], max_iter=hp['max_iter'],
                    tol=hp['tol'], random_state=hp['random_state'] + fold_idx
                )
                model.fit(X_tr, Y_tr, sample_weight=w_tr)

                metrics = model.evaluate_metrics(X_te, Y_te, weights=w_te)
                y_pred_xy = model.predict(X_te, snap=True).astype(np.float32)

                f_met = cand_data['folds']['metrics']
                for _mk in f_met:
                    f_met[_mk].append(metrics[_mk])

                cand_data['folds']['weights'].append(model.coef_)
                cand_data['folds']['intercepts'].append(model.intercept_)
                cand_data['folds']['test_indices'].append(te_idx)
                cand_data['folds']['y_true'].append(Y_te)
                cand_data['folds']['w_test'].append(w_te)
                cand_data['folds']['y_pred_xy'].append(y_pred_xy)
                cand_data['folds']['n_iter'].append(int(model.n_iter_))
                cand_data['folds']['converged'].append(bool(model.converged_))
                cand_data['folds']['fit_time'].append(float(model.fit_time_))
                cand_data['folds']['selected_lambda_smooth'].append(fold_lambda_smooth)
                cand_data['folds']['selected_l2_reg'].append(fold_l2_reg)
                cand_data['folds']['hyperparam_grid_scores'].append(fold_grid_scores)
                cand_data['folds']['hyperparams_tuned'].append(fold_tuned_flag)

                del model
                gc.collect()
            except Exception as e:
                print(f"    [!] Error fitting anchor (Fold {fold_idx}): {e}")
                _append_failed_fold(cand_data)

        valid_scores = np.asarray(
            cand_data['folds']['metrics']['r2_spatial'], dtype=float
        )
        valid_scores = valid_scores[np.isfinite(valid_scores)]
        if valid_scores.size:
            mean_anc = float(np.mean(valid_scores))
            se_anc = (
                float(np.std(valid_scores, ddof=1) / np.sqrt(valid_scores.size))
                if valid_scores.size > 1 else 0.0
            )

            cand_data['mean_r2'] = mean_anc
            cand_data['se_r2'] = se_anc

            # 1SE rule — R^2 is higher-is-better, so the candidate is
            # accepted only when its mean score beats the current best by
            # more than one SE of its own per-fold distribution.
            if (mean_anc - best_current_score) > se_anc:
                print(f"  *** ANCHOR ACCEPTED: R^2 rose to {mean_anc:.4f} ***")
                best_current_score = mean_anc
                best_current_se = se_anc
                current_model_features = [anchor]

                step_1_res = {
                    'step_idx': 1, 'current_features': [anchor],
                    'baseline_score': best_current_score, 'selected_feature': anchor,
                    'candidates_summary': {anchor: cand_data}
                }
                with open(os.path.join(model_selection_dir, f"{prefix}1.pkl"), 'wb') as f:
                    pickle.dump(step_1_res, f)
                step_counter = 2
            else:
                print(f"  *** ANCHOR REJECTED: Failed to beat spatial baseline. Continuing from Empty Model. ***")

    # Forward stepwise selection loop
    print("\n--- Starting Forward Selection ---")
    while True:
        print(f"\n=== Step {step_counter} === Best R^2: {best_current_score:.5f}")
        step_results = {
            'step_idx': step_counter, 'current_features': list(current_model_features),
            'baseline_score': best_current_score, 'candidates_summary': {},
            'selected_feature': None
        }

        # R^2 is higher-is-better; initialise the per-step best score at
        # `-inf` so any finite candidate score is an improvement.
        best_cand, best_cand_score, best_cand_se = None, float('-inf'), 0.0

        for i_feat, feat in enumerate(ranked_features):
            if feat in current_model_features: continue
            gc.collect()
            print(f"  [{i_feat + 1}/{len(ranked_features)}] Testing +{feat}...", end="", flush=True)

            trial_feats = current_model_features + [feat]
            n_trial_feats = len(trial_feats)

            cand_data = _make_manifold_cand_data()

            for fold_idx, (tr_idx, te_idx) in enumerate(cv_folds):
                try:
                    X_tr_stacked = np.hstack([binned_data[f][tr_idx] for f in trial_feats])
                    X_te_stacked = np.hstack([binned_data[f][te_idx] for f in trial_feats])
                    Y_tr, Y_te = y_global[tr_idx], y_global[te_idx]
                    w_tr, w_te = w_global[tr_idx], w_global[te_idx]
                    groups_tr = groups_global[tr_idx]

                    # Per-fold hyperparameters. When tuning is off,
                    # `_pick_fold_hyperparams` returns the fixed centres
                    # with the standard `1 / sqrt(n_trial_feats)` rescale
                    # on l2 so larger trial models don't silently get
                    # over-regularised as the selector grows. When tuning
                    # is on, the inner CV searches a rescaled grid with
                    # the same centre structure.
                    fold_lambda_smooth, fold_l2_reg, fold_grid_scores, fold_tuned_flag = (
                        _pick_fold_hyperparams(
                            X_tr_stacked, Y_tr, w_tr, groups_tr,
                            n_feats_=n_trial_feats,
                            fold_idx_=fold_idx,
                        )
                    )

                    model = SmoothBivariateRegression(
                        n_features=n_trial_feats, n_time_bins=n_time_bins,
                        lambda_smooth=fold_lambda_smooth, l2_reg=fold_l2_reg,
                        huber_delta=hp['huber_delta'],
                        learning_rate=hp['learning_rate'], max_iter=hp['max_iter'],
                        tol=hp['tol'], random_state=hp['random_state'] + fold_idx
                    )
                    model.fit(X_tr_stacked, Y_tr, sample_weight=w_tr)

                    metrics = model.evaluate_metrics(X_te_stacked, Y_te, weights=w_te)
                    y_pred_xy = model.predict(X_te_stacked, snap=True).astype(np.float32)

                    f_met = cand_data['folds']['metrics']
                    for _mk in f_met:
                        f_met[_mk].append(metrics[_mk])

                    cand_data['folds']['weights'].append(model.coef_)
                    cand_data['folds']['intercepts'].append(model.intercept_)
                    cand_data['folds']['test_indices'].append(te_idx)
                    cand_data['folds']['y_true'].append(Y_te)
                    cand_data['folds']['w_test'].append(w_te)
                    cand_data['folds']['y_pred_xy'].append(y_pred_xy)
                    cand_data['folds']['n_iter'].append(int(model.n_iter_))
                    cand_data['folds']['converged'].append(bool(model.converged_))
                    cand_data['folds']['fit_time'].append(float(model.fit_time_))
                    cand_data['folds']['selected_lambda_smooth'].append(fold_lambda_smooth)
                    cand_data['folds']['selected_l2_reg'].append(fold_l2_reg)
                    cand_data['folds']['hyperparam_grid_scores'].append(fold_grid_scores)
                    cand_data['folds']['hyperparams_tuned'].append(fold_tuned_flag)

                    del model, X_tr_stacked, X_te_stacked
                    gc.collect()
                except Exception as e:
                    print(f"    [!] Error fitting {feat} (Fold {fold_idx}): {e}")
                    _append_failed_fold(cand_data)

            valid_scores = np.asarray(
                cand_data['folds']['metrics']['r2_spatial'], dtype=float
            )
            valid_scores = valid_scores[np.isfinite(valid_scores)]
            if not valid_scores.size:
                print(" Failed (no finite folds).")
                continue

            mean_score = float(np.mean(valid_scores))
            se_score = (
                float(np.std(valid_scores, ddof=1) / np.sqrt(valid_scores.size))
                if valid_scores.size > 1 else 0.0
            )
            mean_mae = float(np.nanmean(cand_data['folds']['metrics']['euclidean_mae']))
            print(f" R^2: {mean_score:.4f} | MAE: {mean_mae:.4f}")

            cand_data['mean_r2'] = mean_score
            cand_data['se_r2'] = se_score
            step_results['candidates_summary'][feat] = cand_data

            if mean_score > best_cand_score:
                best_cand_score, best_cand_se, best_cand = mean_score, se_score, feat

        # 1SE rule on R^2 (higher is better): accept only when the best
        # candidate beats the current best by more than its own SE.
        if best_cand and (best_cand_score - best_current_score) > best_cand_se:
            print(f"  ACCEPT {best_cand}")
            step_results['selected_feature'] = best_cand
            current_model_features.append(best_cand)

            with open(os.path.join(model_selection_dir, f"{prefix}{step_counter}.pkl"), 'wb') as f:
                pickle.dump(step_results, f)

            best_current_score = best_cand_score
            best_current_se = best_cand_se
            step_counter += 1
        else:
            print("  REJECT. Selection Finished.")
            step_results['selected_feature'] = None
            with open(os.path.join(model_selection_dir, f"{prefix}{step_counter}.pkl"), 'wb') as f:
                pickle.dump(step_results, f)
            break

        if len(current_model_features) == len(ranked_features):
            break

    print("\n--- Finalizing Results ---")
    try:
        last_file_path = os.path.join(model_selection_dir, f"{prefix}{step_counter}.pkl")
        if not os.path.exists(last_file_path):
            last_file_path = os.path.join(model_selection_dir, f"{prefix}{step_counter - 1}.pkl")

        with open(last_file_path, 'rb') as f:
            step_data = pickle.load(f)

        final_res = {
            'final_model_features': current_model_features,
            'univariate_results_path': univariate_results_path,
            'input_data_path': input_data_path
        }

        step_data.update(final_res)
        with open(last_file_path, 'wb') as f:
            pickle.dump(step_data, f)

        print(f"Success. Final model configuration saved to {os.path.basename(last_file_path)}")

    except Exception as e:
        print(f"Final promotion failed: {e}")
