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
from pygam import LogisticGAM, GAM, te
from scipy.stats import spearmanr, wilcoxon
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit, ShuffleSplit
from sklearn.metrics import (log_loss, roc_auc_score, f1_score, precision_score, recall_score,
                             accuracy_score, balanced_accuracy_score, mean_squared_log_error,
                             mean_gamma_deviance, precision_recall_curve, auc)
from .load_input_files import load_pickle_modeling_data
from .modeling_bases_functions import _normalizecols, bsplines, identity, laplacian_pyramid, raised_cosine
from .modeling_vocal_onsets import GeneralizedLinearModelPipeline
from .modeling_vocal_categories_multinomial import get_stratified_group_splits_stable, MultinomialModelingPipeline, MultinomialModelRunner
from .jax_multinomial_logistic_regression import SmoothMultinomialLogisticRegression
from .jax_bivariate_gaussian_regression import SmoothBivariateGaussianRegression


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

    Crucially, this version preserves full-resolution metric data. For every candidate
    tested at every step, the raw results of all 10 cross-validation folds are saved
    for Log-Likelihood (LL), AUC, Precision, Recall, F1, and Accuracy (score). This
    allows plotting scripts to reconstruct model selection trajectories with
    accurate error bars and individual fold data points.

    Stability features:
    - Safe cleanup: Prevents `UnboundLocalError` during garbage collection if a
      fold fails early.
    - Crash reporting: Logs detailed error messages for failed fits.
    - Resource management: Explicit garbage collection triggers to prevent
      memory fragmentation.
    """

    print("--- Starting Model Selection ---")
    chance_ll = 0.693147

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
        shuffled_ll = results['shuffled']['ll']
        valid_actual = actual_ll[~np.isnan(actual_ll)]
        valid_shuffled = shuffled_ll[~np.isnan(shuffled_ll)]
        if len(valid_actual) == 0 or len(valid_shuffled) == 0:
            continue
        mean_actual_ll = np.mean(valid_actual)
        shuffled_threshold = np.percentile(valid_shuffled, q=(p_val / len(univariate_data)) * 100)
        if mean_actual_ll < shuffled_threshold:
            candidates.append({'feature': feat_name, 'mean_ll': mean_actual_ll})

    candidates.sort(key=lambda x: x['mean_ll'])
    ranked_features = [x['feature'] for x in candidates]
    if not ranked_features:
        print("No significant features found. Aborting.")
        return

    all_feature_data = load_pickle_modeling_data(input_data_path)
    pipeline = GeneralizedLinearModelPipeline(modeling_settings_dict=settings)
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
        X_p_all, X_n_all = pipeline._pool_data_from_sessions(all_feature_data[anchor_feature], all_sessions)
        n_keep = min(X_p_all.shape[0], X_n_all.shape[0])
        pos_indices = np.random.choice(X_p_all.shape[0], n_keep, replace=False)
        neg_indices = np.random.choice(X_n_all.shape[0], n_keep, replace=False)
        y_balanced = np.concatenate((np.ones(n_keep), np.zeros(n_keep)))
        sss = StratifiedShuffleSplit(n_splits=n_splits_selection, test_size=test_prop, random_state=random_seed)
        for train_ix, test_ix in sss.split(np.zeros(len(y_balanced)), y_balanced):
            cv_folds.append({'train_idx': train_ix, 'test_idx': test_ix, 'pos_indices': pos_indices, 'neg_indices': neg_indices, 'type': 'mixed'})

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
            cand_dict = last_results.get('candidates_summary', {})
            if cand_dict:
                cand_stats = []
                for feat, res in cand_dict.items():
                    m = res.get('mean_ll', 0.7031)
                    s = res.get('se_ll', 0.0)
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
        metrics = {'ll': [], 'auc': [], 'score': [], 'f1': [], 'precision': [], 'recall': []}

        for fold_i, fold_info in enumerate(cv_folds):
            try:
                X_train_list, X_test_list = [], []
                if fold_info['type'] == 'session':
                    train_sess, test_sess = fold_info['train_sessions'], fold_info['test_sessions']
                    anc_data = all_feature_data[anchor_feature]
                    X_p_tr, X_n_tr = pipeline._pool_data_from_sessions(anc_data, train_sess)
                    n_k = min(X_p_tr.shape[0], X_n_tr.shape[0])
                    idx_p = np.random.choice(X_p_tr.shape[0], n_k, replace=False)
                    idx_n = np.random.choice(X_n_tr.shape[0], n_k, replace=False)
                    y_tr_fold = np.concatenate((np.ones(n_k), np.zeros(n_k)))
                    X_p_te, X_n_te = pipeline._pool_data_from_sessions(anc_data, test_sess)
                    y_te_fold = np.concatenate((np.ones(X_p_te.shape[0]), np.zeros(X_n_te.shape[0])))
                    X_train_list.append(np.concatenate((X_p_tr[idx_p], X_n_tr[idx_n])))
                    X_test_list.append(np.concatenate((X_p_te, X_n_te)))
                elif fold_info['type'] == 'mixed':
                    train_ix, test_ix = fold_info['train_idx'], fold_info['test_idx']
                    pos_ix, neg_ix = fold_info['pos_indices'], fold_info['neg_indices']
                    y_bal = np.concatenate((np.ones(len(pos_ix)), np.zeros(len(neg_ix))))
                    y_tr_fold, y_te_fold = y_bal[train_ix], y_bal[test_ix]
                    X_p, X_n = pipeline._pool_data_from_sessions(all_feature_data[anchor_to_force], all_sessions)
                    X_bal = np.concatenate((X_p[pos_ix], X_n[neg_ix]))
                    X_train_list.append(X_bal[train_ix])
                    X_test_list.append(X_bal[test_ix])

                X_tr_gam = get_unrolled_X_for_multivariate(X_train_list, history_frames)
                X_te_gam = get_unrolled_X_for_multivariate(X_test_list, history_frames)
                gam = LogisticGAM(gam_terms, **gam_kwargs).fit(X_tr_gam, np.repeat(y_tr_fold.astype(float), history_frames))

                y_proba_tiled = gam.predict_proba(X_te_gam)
                y_proba_mean = np.mean(y_proba_tiled.reshape(len(y_te_fold), history_frames), axis=1)
                metrics['ll'].append(log_loss(y_te_fold.astype(int), np.clip(y_proba_mean, 1e-15, 1 - 1e-15)))

                y_pred_mean = (y_proba_mean > 0.5).astype(int)
                metrics['score'].append(accuracy_score(y_te_fold, y_pred_mean))
                metrics['f1'].append(f1_score(y_te_fold, y_pred_mean, zero_division=0))
                metrics['precision'].append(precision_score(y_te_fold, y_pred_mean, zero_division=0))
                metrics['recall'].append(recall_score(y_te_fold, y_pred_mean, zero_division=0))
                metrics['auc'].append(roc_auc_score(y_te_fold, y_proba_mean) if len(np.unique(y_te_fold)) > 1 else np.nan)

                del gam, X_tr_gam, X_te_gam
                gc.collect()
            except Exception as e:
                print(f"    [!] Error fitting {feat} (Fold {fold_i}): {e}")
                metrics['ll'].append(chance_ll)

        valid_ll = [x for x in metrics['ll'] if x < chance_ll + 0.05]
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
                        'f1': metrics['f1'], 'precision': metrics['precision'], 'recall': metrics['recall'],
                        'mean_ll': best_current_score, 'se_ll': best_current_se
                    }
                }
            }
            target_sex = 'female' if 'female' in univariate_results_path else 'male'
            s0_name = f"model_selection_{target_sex}_{settings['model_params']['model_target_vocal_type']}_{split_strategy}_step_0.pkl"
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

            metrics = {'ll': [], 'auc': [], 'score': [], 'f1': [], 'precision': [], 'recall': []}
            for fold_i, fold_info in enumerate(cv_folds):
                try:
                    X_train_list, X_test_list = [], []
                    if fold_info['type'] == 'session':
                        train_sess, test_sess = fold_info['train_sessions'], fold_info['test_sessions']
                        anc_data = all_feature_data[anchor_feature]
                        X_p_tr, X_n_tr = pipeline._pool_data_from_sessions(anc_data, train_sess)
                        np.random.seed(random_seed + fold_i)
                        n_k = min(X_p_tr.shape[0], X_n_tr.shape[0])
                        idx_p = np.random.choice(X_p_tr.shape[0], n_k, replace=False)
                        idx_n = np.random.choice(X_n_tr.shape[0], n_k, replace=False)
                        y_tr_fold = np.concatenate((np.ones(n_k), np.zeros(n_k)))

                        X_p_te_anc, X_n_te_anc = pipeline._pool_data_from_sessions(anc_data, test_sess)
                        y_te_fold = np.concatenate((np.ones(X_p_te_anc.shape[0]), np.zeros(X_n_te_anc.shape[0])))

                        for f_name in trial_features:
                            f_p_tr, f_n_tr = pipeline._pool_data_from_sessions(all_feature_data[f_name], train_sess)
                            f_p_te, f_n_te = pipeline._pool_data_from_sessions(all_feature_data[f_name], test_sess)
                            X_train_list.append(np.concatenate((f_p_tr[idx_p], f_n_tr[idx_n])))
                            X_test_list.append(np.concatenate((f_p_te, f_n_te)))
                    elif fold_info['type'] == 'mixed':
                        train_ix, test_ix = fold_info['train_idx'], fold_info['test_idx']
                        pos_ix, neg_ix = fold_info['pos_indices'], fold_info['neg_indices']
                        y_tr_fold = np.concatenate((np.ones(len(pos_ix)), np.zeros(len(neg_ix))))[train_ix]
                        y_te_fold = np.concatenate((np.ones(len(pos_ix)), np.zeros(len(neg_ix))))[test_ix]
                        for f_name in trial_features:
                            X_p, X_n = pipeline._pool_data_from_sessions(all_feature_data[f_name], all_sessions)
                            X_bal = np.concatenate((X_p[pos_ix], X_n[neg_ix]))
                            X_train_list.append(X_bal[train_ix])
                            X_test_list.append(X_bal[test_ix])

                    X_tr_gam = get_unrolled_X_for_multivariate(X_train_list, history_frames)
                    X_te_gam = get_unrolled_X_for_multivariate(X_test_list, history_frames)
                    gam = LogisticGAM(gam_terms, **gam_kwargs).fit(X_tr_gam, np.repeat(y_tr_fold.astype(float), history_frames))

                    y_proba_tiled = gam.predict_proba(X_te_gam)
                    y_proba_mean = np.mean(y_proba_tiled.reshape(len(y_te_fold), history_frames), axis=1)
                    metrics['ll'].append(log_loss(y_te_fold.astype(int), np.clip(y_proba_mean, 1e-15, 1 - 1e-15)))

                    y_pred_mean = (y_proba_mean > 0.5).astype(int)
                    metrics['score'].append(accuracy_score(y_te_fold, y_pred_mean))
                    metrics['f1'].append(f1_score(y_te_fold, y_pred_mean, zero_division=0))
                    metrics['precision'].append(precision_score(y_te_fold, y_pred_mean, zero_division=0))
                    metrics['recall'].append(recall_score(y_te_fold, y_pred_mean, zero_division=0))
                    metrics['auc'].append(roc_auc_score(y_te_fold, y_proba_mean) if len(np.unique(y_te_fold)) > 1 else np.nan)

                    del gam, X_tr_gam, X_te_gam
                    gc.collect()
                except Exception as e:
                    print(f"    [!] Error fitting {feat} (Fold {fold_i}): {e}")
                    metrics['ll'].append(chance_ll)

            valid = [x for x in metrics['ll'] if np.isfinite(x)]

            if valid:
                mean_ll, se_ll = np.mean(valid), np.std(valid, ddof=1) / np.sqrt(len(valid))
                print(f" LL: {mean_ll:.4f} (range: {min(valid):.4f}-{max(valid):.4f})")

                step_results_metadata['candidates_summary'][feat] = {
                    'll': metrics['ll'], 'auc': metrics['auc'], 'score': metrics['score'],
                    'f1': metrics['f1'], 'precision': metrics['precision'], 'recall': metrics['recall'],
                    'mean_ll': mean_ll, 'se_ll': se_ll
                }

                if mean_ll < best_candidate_score:
                    best_candidate_score, best_candidate_se, best_candidate = mean_ll, se_ll, feat
            else:
                print(f" Failed (No valid numeric scores). Metrics: {metrics['ll']}")

        target_sex = 'female' if 'female' in univariate_results_path else 'male'

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
        X_p, X_n = pipeline._pool_data_from_sessions(all_feature_data[anchor_feature], all_sessions)

        np.random.seed(random_seed)
        n_k = min(X_p.shape[0], X_n.shape[0])
        idx_p = np.random.choice(X_p.shape[0], n_k, replace=False)
        idx_n = np.random.choice(X_n.shape[0], n_k, replace=False)

        y_final = np.concatenate((np.ones(n_k), np.zeros(n_k)))

        X_list_final = []
        for f in current_model_features:
            f_p, f_n = pipeline._pool_data_from_sessions(all_feature_data[f], all_sessions)
            X_list_final.append(np.concatenate((f_p[idx_p], f_n[idx_n])))

        last_file = os.path.join(model_selection_dir, fname)
        final_fold_shapes = []

        print(f"  Calculating filter shapes across {len(cv_folds)} folds...")

        for fold_idx, (tr_idx, te_idx) in enumerate(cv_folds):

            X_list_tr = [x[tr_idx] for x in X_list_final]
            y_tr = y_final[tr_idx]

            X_gam_tr = get_unrolled_X_for_multivariate(X_list_tr, history_frames)
            y_gam_tr = np.repeat(y_tr.astype(float) + 1e-6, history_frames)

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
    and ensures rigorous validation by enforcing 50/50 class balance within every
    cross-validation fold.

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
        - Calculates and saves raw lists of metrics (Log-Loss, AUC, F1, Accuracy,
          Precision, Recall) for every fold for every candidate.
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
    chance_ll = 0.693147

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

    # Safely extracting using assumed re logic without inline imports
    import re
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
        null_key = 'null' if 'null' in results else 'shuffled'
        shuffled_ll = results[null_key]['ll']

        valid_actual = actual_ll[~np.isnan(actual_ll)]
        valid_shuffled = shuffled_ll[~np.isnan(shuffled_ll)]

        if len(valid_actual) == 0 or len(valid_shuffled) == 0:
            continue

        mean_actual_ll = np.mean(valid_actual)
        shuffled_threshold = np.percentile(valid_shuffled, q=(p_val / len(univariate_data)) * 100)

        if mean_actual_ll < shuffled_threshold:
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

    if split_strategy == 'session':
        ss = ShuffleSplit(n_splits=n_splits, test_size=test_prop, random_state=random_seed)
        cv_folds = list(ss.split(all_sessions_arr))
    elif split_strategy == 'mixed':
        ref_targ, ref_other = _pool_category_features(all_feature_data, [ranked_features[0]], all_sessions_arr, history_frames)
        n_targ_total = ref_targ[0].shape[0]
        n_other_total = ref_other[0].shape[0]

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
        metrics = {'ll': [], 'auc': [], 'score': [], 'f1': [], 'precision': [], 'recall': []}

        if model_type == 'pygam':
            gam_terms = te(0, 1, n_splines=[n_splines_value, n_splines_time])

        for fold_i, (tr_idx, te_idx) in enumerate(cv_folds):
            try:
                if split_strategy == 'session':
                    X_tr_t_raw, X_tr_o_raw = _pool_category_features(all_feature_data, [anchor], all_sessions_arr[tr_idx], history_frames)
                    X_te_t_raw, X_te_o_raw = _pool_category_features(all_feature_data, [anchor], all_sessions_arr[te_idx], history_frames)
                else:
                    X_all_t_raw, X_all_o_raw = _pool_category_features(all_feature_data, [anchor], all_sessions_arr, history_frames)

                    tr_targ_idx = tr_idx[tr_idx < n_targ_total]
                    tr_other_idx = tr_idx[tr_idx >= n_targ_total] - n_targ_total
                    te_targ_idx = te_idx[te_idx < n_targ_total]
                    te_other_idx = te_idx[te_idx >= n_targ_total] - n_targ_total

                    X_tr_t_raw = [x[tr_targ_idx] for x in X_all_t_raw]
                    X_tr_o_raw = [x[tr_other_idx] for x in X_all_o_raw]
                    X_te_t_raw = [x[te_targ_idx] for x in X_all_t_raw]
                    X_te_o_raw = [x[te_other_idx] for x in X_all_o_raw]

                X_tr_t, X_tr_o, y_tr_t, y_tr_o = _balance_multivariate_arrays(X_tr_t_raw, X_tr_o_raw, random_seed + fold_i)
                X_te_t, X_te_o, y_te_t, y_te_o = _balance_multivariate_arrays(X_te_t_raw, X_te_o_raw, random_seed + fold_i + 100)

                if not X_tr_t or not X_te_t: continue

                X_tr_list = [np.concatenate([t, o], axis=0) for t, o in zip(X_tr_t, X_tr_o)]
                y_tr = np.concatenate([y_tr_t, y_tr_o])
                perm = np.random.permutation(len(y_tr))

                X_te_list = [np.concatenate([t, o], axis=0) for t, o in zip(X_te_t, X_te_o)]
                y_te = np.concatenate([y_te_t, y_te_o]).astype(int)

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
                    model.fit(X_tr_stacked, y_tr[perm])
                    y_proba = model.predict_proba(X_te_stacked)[:, 1]
                    y_pred = model.predict(X_te_stacked)
                else:
                    X_tr_gam = get_unrolled_X_for_multivariate([x[perm] for x in X_tr_list], history_frames)
                    y_tr_gam = np.repeat(y_tr[perm].astype(float), history_frames)
                    gam = LogisticGAM(gam_terms, **gam_kwargs).fit(X_tr_gam, y_tr_gam)

                    X_te_gam = get_unrolled_X_for_multivariate(X_te_list, history_frames)
                    y_proba_tiled = gam.predict_proba(X_te_gam)
                    y_proba = np.mean(y_proba_tiled.reshape(len(y_te), history_frames), axis=1)
                    y_pred = (y_proba >= 0.5).astype(int)

                metrics['ll'].append(log_loss(y_te, np.clip(y_proba, 1e-15, 1 - 1e-15)))
                metrics['auc'].append(roc_auc_score(y_te, y_proba))
                metrics['f1'].append(f1_score(y_te, y_pred))
                metrics['precision'].append(precision_score(y_te, y_pred, zero_division=0))
                metrics['recall'].append(recall_score(y_te, y_pred, zero_division=0))
                metrics['score'].append(accuracy_score(y_te, y_pred))
                gc.collect()
            except Exception as e:
                print(f"    [!] Error fitting {anchor} (Fold {fold_i}): {e}")
                metrics['ll'].append(chance_ll)

        valid = [s for s in metrics['ll'] if s < chance_ll + 0.05]
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
                        'f1': metrics['f1'], 'precision': metrics['precision'],
                        'recall': metrics['recall'], 'score': metrics['score'],
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

            metrics = {'ll': [], 'auc': [], 'score': [], 'f1': [], 'precision': [], 'recall': []}

            for fold_i, (tr_idx, te_idx) in enumerate(cv_folds):
                try:
                    if split_strategy == 'session':
                        X_tr_t_raw, X_tr_o_raw = _pool_category_features(all_feature_data, trial_feats, all_sessions_arr[tr_idx], history_frames)
                        X_te_t_raw, X_te_o_raw = _pool_category_features(all_feature_data, trial_feats, all_sessions_arr[te_idx], history_frames)
                    else:
                        X_all_t_raw, X_all_o_raw = _pool_category_features(all_feature_data, trial_feats, all_sessions_arr, history_frames)

                        tr_targ_idx = tr_idx[tr_idx < n_targ_total]
                        tr_other_idx = tr_idx[tr_idx >= n_targ_total] - n_targ_total
                        te_targ_idx = te_idx[te_idx < n_targ_total]
                        te_other_idx = te_idx[te_idx >= n_targ_total] - n_targ_total

                        X_tr_t_raw = [x[tr_targ_idx] for x in X_all_t_raw]
                        X_tr_o_raw = [x[tr_other_idx] for x in X_all_o_raw]
                        X_te_t_raw = [x[te_targ_idx] for x in X_all_t_raw]
                        X_te_o_raw = [x[te_other_idx] for x in X_all_o_raw]

                    X_tr_t, X_tr_o, y_tr_t, y_tr_o = _balance_multivariate_arrays(X_tr_t_raw, X_tr_o_raw, random_seed + fold_i)
                    X_te_t, X_te_o, y_te_t, y_te_o = _balance_multivariate_arrays(X_te_t_raw, X_te_o_raw, random_seed + fold_i + 100)

                    if not X_tr_t or not X_te_t: continue

                    X_tr_list = [np.concatenate([t, o], axis=0) for t, o in zip(X_tr_t, X_tr_o)]
                    y_tr = np.concatenate([y_tr_t, y_tr_o])
                    perm = np.random.permutation(len(y_tr))

                    X_te_list = [np.concatenate([t, o], axis=0) for t, o in zip(X_te_t, X_te_o)]
                    y_te = np.concatenate([y_te_t, y_te_o]).astype(int)

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
                        model.fit(X_tr_stacked, y_tr[perm])
                        y_proba = model.predict_proba(X_te_stacked)[:, 1]
                        y_pred = model.predict(X_te_stacked)
                    else:
                        X_tr_gam = get_unrolled_X_for_multivariate([x[perm] for x in X_tr_list], history_frames)
                        y_tr_gam = np.repeat(y_tr[perm].astype(float), history_frames)
                        gam = LogisticGAM(gam_terms, **gam_kwargs).fit(X_tr_gam, y_tr_gam)

                        X_te_gam = get_unrolled_X_for_multivariate(X_te_list, history_frames)
                        y_proba_tiled = gam.predict_proba(X_te_gam)
                        y_proba = np.mean(y_proba_tiled.reshape(len(y_te), history_frames), axis=1)
                        y_pred = (y_proba >= 0.5).astype(int)

                    metrics['ll'].append(log_loss(y_te, np.clip(y_proba, 1e-15, 1 - 1e-15)))
                    metrics['auc'].append(roc_auc_score(y_te, y_proba))
                    metrics['f1'].append(f1_score(y_te, y_pred))
                    metrics['precision'].append(precision_score(y_te, y_pred, zero_division=0))
                    metrics['recall'].append(recall_score(y_te, y_pred, zero_division=0))
                    metrics['score'].append(accuracy_score(y_te, y_pred))
                    gc.collect()
                except Exception as e:
                    print(f"    [!] Error fitting {feat} (Fold {fold_i}): {e}")
                    metrics['ll'].append(chance_ll)

            valid = [x for x in metrics['ll'] if x < chance_ll + 0.05]
            mean_ll = np.mean(valid) if valid else chance_ll
            se_ll = (np.std(valid, ddof=1) / np.sqrt(len(valid))) if valid else 0.0
            print(f" LL: {mean_ll:.4f} | AUC: {np.nanmean(metrics['auc']):.3f}")

            step_results_metadata['candidates_summary'][feat] = {
                'll': metrics['ll'], 'auc': metrics['auc'],
                'f1': metrics['f1'], 'precision': metrics['precision'],
                'recall': metrics['recall'], 'score': metrics['score'],
                'mean_ll': mean_ll, 'se_ll': se_ll
            }
            if mean_ll < best_cand_score:
                best_cand_score, best_cand_se, best_cand_name = mean_ll, se_ll, feat

        if best_cand_name and (best_current_score - best_cand_score) > best_cand_se:
            print(f"  ACCEPT {best_cand_name}")
            current_model_features.append(best_cand_name)
            best_current_score, step_counter = best_cand_score, step_counter + 1
            step_results_metadata['selected_feature'] = best_cand_name
            with open(os.path.join(model_selection_dir, f"{prefix}{step_counter - 1}.pkl"), 'wb') as f:
                pickle.dump(step_results_metadata, f)
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
                X_all_t_raw, X_all_o_raw = _pool_category_features(all_feature_data, current_model_features, all_sessions_arr, history_frames)

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
    - 'pygam': Utilizes `GAM` with a Gamma distribution and Log-link function. High-dimensional
               features are unrolled into tensor product splines (te) to capture non-linear surfaces.

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
    Preserves full-resolution metric data. For every candidate tested at every step,
    the raw results of all cross-validation folds are saved for Explained Deviance (D2),
    Spearman's Rho, Mean Squared Logarithmic Error (MSLE), and Residual Deviance.

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

        actual_deviance = results['actual']['explained_deviance']
        null_key = 'null' if 'null' in results else 'shuffled'

        if null_key not in results or 'explained_deviance' not in results[null_key]:
            continue

        shuffled_deviance = results[null_key]['explained_deviance']

        valid_actual = actual_deviance[~np.isnan(actual_deviance)]
        valid_shuffled = shuffled_deviance[~np.isnan(shuffled_deviance)]

        if len(valid_actual) == 0 or len(valid_shuffled) == 0:
            continue

        mean_actual_deviance = np.mean(valid_actual)
        shuffled_threshold = np.percentile(valid_shuffled, q=100 - (p_val / num_features) * 100)

        if mean_actual_deviance > shuffled_threshold and mean_actual_deviance > 0:
            candidates.append({'feature': feat_name, 'mean_explained_deviance': mean_actual_deviance})
        else:
            reason = "Negative D^2" if mean_actual_deviance <= 0 else "Not Significant"
            print(f"  Dropping {feat_name}: {reason} (Mean Dev {mean_actual_deviance:.4f} vs Shuffled {shuffled_threshold:.4f})")

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
    n_folds_mc = int(np.floor(1.0 / test_prop))
    n_folds_mc = max(2, n_folds_mc)

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
        metrics = {
            'explained_deviance': [], 'residual_deviance': [],
            'spearman_r': [], 'msle': [],
            'y_true': [], 'y_pred': []
        }

        for tr_idx, te_idx in cv_folds:
            try:
                X_tr, y_tr = all_feature_data[anchor]['X'][tr_idx], y_global[tr_idx]
                X_te, y_te = all_feature_data[anchor]['X'][te_idx], y_global[te_idx]

                if model_type == 'sklearn':
                    X_tr_proj = np.dot(X_tr, basis_matrix)
                    X_te_proj = np.dot(X_te, basis_matrix)

                    y_tr_log = np.log(y_tr + 1e-6)
                    model = RidgeCV(alphas=lr_params['alphas'], cv=lr_params['cv']).fit(X_tr_proj, y_tr_log)

                    y_pred = np.exp(model.predict(X_te_proj))
                else:
                    X_tr_unrolled = np.empty((len(X_tr) * history_frames, 2))
                    X_tr_unrolled[:, 0] = X_tr.ravel()
                    X_tr_unrolled[:, 1] = np.tile(np.arange(history_frames), len(X_tr))

                    gam = GAM(te(0, 1, n_splines=[n_splines_value, n_splines_time]), distribution='gamma', link='log', **gam_kwargs)
                    gam.fit(X_tr_unrolled, np.repeat(y_tr + 1e-6, history_frames))

                    X_te_unrolled = np.empty((len(X_te) * history_frames, 2))
                    X_te_unrolled[:, 0] = X_te.ravel()
                    X_te_unrolled[:, 1] = np.tile(np.arange(history_frames), len(X_te))

                    y_pred = np.mean(gam.predict(X_te_unrolled).reshape(len(y_te), history_frames), axis=1)

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
                metrics['msle'].append(mean_squared_log_error(y_te_safe, y_pred_safe))
                metrics['y_true'].append(y_te_safe)
                metrics['y_pred'].append(y_pred_safe)
                gc.collect()

            except Exception as e:
                print(f"    [!] Error fitting: {e}")
                metrics['explained_deviance'].append(np.nan)
                metrics['residual_deviance'].append(np.nan)
                metrics['spearman_r'].append(np.nan)
                metrics['msle'].append(np.nan)

        valid_dev = [m for m in metrics['explained_deviance'] if not np.isnan(m)]
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
                            'explained_deviance': metrics['explained_deviance'],
                            'residual_deviance': metrics['residual_deviance'],
                            'spearman_r': metrics['spearman_r'],
                            'msle': metrics['msle'],
                            'y_true': metrics['y_true'],
                            'y_pred': metrics['y_pred'],
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
                'spearman_r': [], 'msle': [],
                'y_true': [], 'y_pred': []
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
                        model = RidgeCV(alphas=lr_params['alphas'], cv=lr_params['cv']).fit(X_tr_stacked, y_tr_log)
                        y_pred = np.exp(model.predict(X_te_stacked))
                    else:
                        X_tr_gam = get_unrolled_X_for_multivariate(trial_tr, history_frames)
                        X_te_gam = get_unrolled_X_for_multivariate(trial_te, history_frames)

                        gam = GAM(gam_terms, distribution='gamma', link='log', **gam_kwargs).fit(X_tr_gam, np.repeat(y_tr + 1e-6, history_frames))
                        y_pred = np.mean(gam.predict(X_te_gam).reshape(len(y_te), history_frames), axis=1)

                    y_te_safe = np.maximum(y_te, 1e-6)
                    y_pred_safe = np.maximum(y_pred, 1e-6)
                    y_null_pred = np.full_like(y_te_safe, np.mean(y_te_safe))

                    res_dev = mean_gamma_deviance(y_te_safe, y_pred_safe)
                    null_dev = mean_gamma_deviance(y_te_safe, y_null_pred)

                    d2 = 0.0 if null_dev == 0 else 1 - (res_dev / null_dev)

                    metrics['residual_deviance'].append(res_dev)
                    metrics['explained_deviance'].append(d2)
                    metrics['spearman_r'].append(spearmanr(y_te, y_pred)[0])
                    metrics['msle'].append(mean_squared_log_error(y_te_safe, y_pred_safe))
                    metrics['y_true'].append(y_te_safe)
                    metrics['y_pred'].append(y_pred_safe)

                    gc.collect()
                except Exception as e:
                    print(f"    [!] Error fitting {feat} (Fold {fold_idx}): {e}")
                    metrics['explained_deviance'].append(np.nan)
                    metrics['residual_deviance'].append(np.nan)
                    metrics['spearman_r'].append(np.nan)
                    metrics['msle'].append(np.nan)

            valid = [m for m in metrics['explained_deviance'] if not np.isnan(m)]
            if valid:
                m_dev, s_dev = np.mean(valid), np.std(valid, ddof=1) / np.sqrt(len(valid))
                print(f" D^2: {m_dev:.4f}")

                step_results['candidates_summary'][feat] = {
                    'explained_deviance': metrics['explained_deviance'],
                    'residual_deviance': metrics['residual_deviance'],
                    'spearman_r': metrics['spearman_r'],
                    'msle': metrics['msle'],
                    'y_true': metrics['y_true'],
                    'y_pred': metrics['y_pred'],
                    'mean_explained_deviance': m_dev,
                    'se_explained_deviance': s_dev
                }

                if m_dev > best_cand_score:
                    best_cand_score, best_cand_se, best_cand = m_dev, s_dev, feat
            else:
                print(" Failed.")

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
            * 'metrics': dict of 'll', 'auc', 'f1', 'precision', 'recall', 'score'.
            * 'weights': The learned JAX coefficient matrix.
            * 'intercepts': The learned JAX biases.
            * 'y_true', 'y_pred', 'y_probs': Ground truth, hard predictions, and softmax arrays.
            * 'test_indices': The specific dataset rows used in the validation fold.

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

        null_key = 'null' if 'null' in results else 'shuffled'
        null_auc = np.array(results[null_key]['folds']['metrics']['auc'])

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

    if split_strategy == 'session':
        cv_folds = get_stratified_group_splits_stable(
            groups=groups_global,
            y=y_global,
            test_prop=test_prop,
            n_splits=n_splits,
            random_seed=random_seed
        )
    elif split_strategy == 'mixed':
        sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_prop, random_state=random_seed)
        cv_folds = list(sss.split(binned_data[ranked_features[0]], y_global))
    else:
        raise ValueError("split_strategy in settings must be either 'session' or 'mixed'.")

    current_model_features = []
    best_current_score = 0.0
    best_current_se = 0.0
    step_counter = 0

    fname = os.path.basename(univariate_results_path)
    cond_match = re.search(r'(male|female.*?)(?=_splits|_lam|_gmm|\.pkl)', fname)
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
                'metrics': {m: [] for m in ['auc', 'score', 'precision', 'recall', 'f1', 'll']},
                'y_true': [], 'y_pred': [], 'y_probs': [], 'test_indices': []
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
            f_met['precision'].append(precision_score(y_te, y_pred, average='macro', zero_division=0))
            f_met['recall'].append(recall_score(y_te, y_pred, average='macro', zero_division=0))

            try:
                f_ll = log_loss(y_te, y_proba_clipped, labels=unique_classes)
            except ValueError:
                f_ll = np.nan
            f_met['ll'].append(f_ll)

            baseline_data['folds']['y_true'].append(y_te)
            baseline_data['folds']['y_pred'].append(y_pred)
            baseline_data['folds']['y_probs'].append(y_proba)
            baseline_data['folds']['test_indices'].append(te_idx)
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
                'metrics': {m: [] for m in ['auc', 'score', 'precision', 'recall', 'f1', 'll']},
                'weights': [], 'intercepts': [], 'y_true': [], 'y_pred': [], 'y_probs': [], 'test_indices': []
            },
            'classes': None
        }

        for tr_idx, te_idx in cv_folds:
            try:
                X_tr, X_te = binned_data[anchor][tr_idx], binned_data[anchor][te_idx]
                y_tr, y_te = y_global[tr_idx], y_global[te_idx]

                model = SmoothMultinomialLogisticRegression(
                    n_features=1, n_time_bins=n_time_bins,
                    lambda_smooth=hp['lambda_smooth'],
                    l1_reg=hp['l1_reg'],
                    l2_reg=hp['l2_reg'],
                    learning_rate=hp['learning_rate'],
                    max_iter=hp['max_iter'],
                    tol=hp['tol'],
                    random_state=hp['random_state']
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
                f_met['precision'].append(precision_score(y_te, y_pred, average='macro', zero_division=0))
                f_met['recall'].append(recall_score(y_te, y_pred, average='macro', zero_division=0))
                f_met['ll'].append(log_loss(y_te, y_proba_clipped, labels=model.classes_))

                cand_data['folds']['weights'].append(model.coef_)
                cand_data['folds']['intercepts'].append(model.intercept_)
                cand_data['folds']['y_true'].append(y_te)
                cand_data['folds']['y_pred'].append(y_pred)
                cand_data['folds']['y_probs'].append(y_proba)
                cand_data['folds']['test_indices'].append(te_idx)
                if cand_data['classes'] is None:
                    cand_data['classes'] = model.classes_
            except Exception as e:
                print(f"    [!] Error fitting anchor: {e}")
                cand_data['folds']['metrics']['auc'].append(np.nan)

        valid_auc = [m for m in cand_data['folds']['metrics']['auc'] if not np.isnan(m)]
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
                    'metrics': {m: [] for m in ['auc', 'score', 'precision', 'recall', 'f1', 'll']},
                    'weights': [], 'intercepts': [], 'y_true': [], 'y_pred': [], 'y_probs': [], 'test_indices': []
                },
                'classes': None
            }

            for fold_idx, (tr_idx, te_idx) in enumerate(cv_folds):
                try:
                    X_tr_stacked = np.hstack([binned_data[f][tr_idx] for f in trial_feats])
                    X_te_stacked = np.hstack([binned_data[f][te_idx] for f in trial_feats])
                    y_tr, y_te = y_global[tr_idx], y_global[te_idx]

                    model = SmoothMultinomialLogisticRegression(
                        n_features=n_trial_feats, n_time_bins=n_time_bins,
                        lambda_smooth=hp['lambda_smooth'],
                        l1_reg=hp['l1_reg'],
                        l2_reg=hp['l2_reg'] / np.sqrt(n_trial_feats),
                        learning_rate=hp['learning_rate'],
                        max_iter=hp['max_iter'],
                        tol=hp['tol'],
                        random_state=hp['random_state']
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
                    f_met['precision'].append(precision_score(y_te, y_pred, average='macro', zero_division=0))
                    f_met['recall'].append(recall_score(y_te, y_pred, average='macro', zero_division=0))
                    f_met['ll'].append(log_loss(y_te, y_proba_clipped, labels=model.classes_))

                    cand_data['folds']['weights'].append(model.coef_)
                    cand_data['folds']['intercepts'].append(model.intercept_)
                    cand_data['folds']['y_true'].append(y_te)
                    cand_data['folds']['y_pred'].append(y_pred)
                    cand_data['folds']['y_probs'].append(y_proba)
                    cand_data['folds']['test_indices'].append(te_idx)
                    if cand_data['classes'] is None:
                        cand_data['classes'] = model.classes_
                except Exception:
                    cand_data['folds']['metrics']['auc'].append(np.nan)

            valid_auc = [x for x in cand_data['folds']['metrics']['auc'] if not np.isnan(x)]
            if valid_auc:
                mean_auc = np.mean(valid_auc)
                se_auc = np.std(valid_auc, ddof=1) / np.sqrt(len(valid_auc))
                print(f" AUC: {mean_auc:.4f}")
                cand_data['mean_auc'], cand_data['se_auc'] = mean_auc, se_auc
                step_results['candidates_summary'][feat] = cand_data
                if mean_auc > best_cand_score:
                    best_cand_score, best_cand_se, best_cand = mean_auc, se_auc, feat
            else:
                print(" Failed.")

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
    Performs forward stepwise selection for continuous UMAP vocal manifold prediction.

    This function identifies the optimal subset of behavioral features that jointly
    predict the continuous (x,y) topography of upcoming USVs. It uses the stable,
    Homoscedastic Bivariate Gaussian JAX architecture to model both the deterministic
    spatial center and the structural variance of the acoustic space.

    Key algorithmic adaptations for continuous topographic modeling:
    1. Wilcoxon screening: Instead of comparing means to a shuffled percentile,
       this algorithm uses a Bonferroni-corrected Wilcoxon signed-rank test
       on perfectly paired, spatially identical cross-validation folds to
       identify significant univariate candidates.
    2. Spatial stratification: To prevent over-indexing on the dense manifold core,
       test splits are generated using deterministic K-Means geographic clustering.
       It dynamically supports 'session' (isolating days) or 'mixed' (proportional
       epoch splitting) while ensuring rare acoustic satellites are proportionally
       represented across all validation folds.
    3. Inverse-density weighting: Passes the pre-computed KDE spatial weights (w)
       to the JAX optimizer to enforce geographic fairness during gradient descent.
    4. Model-Free Baseline (Step 0): Computes the global Marginal Prior (the spatial
       density of the training manifold) as the absolute baseline. Features must prove
       they offer temporal predictive power above and beyond simply guessing the
       center of the acoustic space.
    5. 1SE Rule (Error Minimization): A feature is added only if it reduces the
       Mean Weighted Negative Log-Likelihood (NLL) by an amount greater than the
       candidate's Standard Error across the validation folds.

    Deep Storage for Post-Hoc Visualization:
    ----------------------------------------
    At every step, the function saves a deep dictionary containing not just the global
    metrics, but the fold-level JAX weights (`coef_`), biases (`intercept_`), global
    variance parameters (`global_vars_`), the actual test coordinates (`y_true`), and
    the full predicted probability footprints (`y_pred_params`).

    Parameters
    ----------
    univariate_results_path : str
        Path to the univariate regression results pickle file containing the
        paired Actual/Null NLL fold arrays.
    input_data_path : str
        Path to the extracted UMAP data containing X (history), Y (UMAP),
        and w (KDE spatial weights).
    output_directory : str
        Directory to save the step-wise state dictionaries.
    settings_path : str, optional
        Path to the JSON settings file.
    use_top_rank_as_anchor : bool, default False
        If True, initializes the search by forcing the single highest-ranked
        Wilcoxon feature as Step 1.
    p_val : float, default 0.05
        The overall alpha level, strictly Bonferroni-corrected by dividing
        it by the number of evaluated features.
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

    print(f"Screening {num_features} features (Bonferroni alpha = {alpha_corrected:.2e})...")
    for feat_name, payload in univariate_data.items():
        if isinstance(payload, tuple) and len(payload) == 2:
            _, results = payload
        else:
            results = payload

        if 'actual' not in results:
            continue

        if 'nll_weighted' not in results['actual']['folds']['metrics']:
            continue

        actual_nll = np.array(results['actual']['folds']['metrics']['nll_weighted'])
        null_nll = np.array(results['null']['folds']['metrics']['nll_weighted'])

        valid_actual = actual_nll[~np.isnan(actual_nll)]
        valid_null = null_nll[~np.isnan(null_nll)]

        if len(valid_actual) == 0 or len(valid_null) == 0:
            continue

        try:
            _, p_value = wilcoxon(valid_actual, valid_null, alternative='less')
        except ValueError:
            p_value = 1.0

        if p_value < alpha_corrected:
            mean_nll = np.mean(valid_actual)
            candidates.append({'feature': feat_name, 'mean_nll': mean_nll, 'p_val': p_value})

    candidates.sort(key=lambda x: x['mean_nll'])
    ranked_features = [x['feature'] for x in candidates]

    if not ranked_features:
        print(f"No significant features found (p < {alpha_corrected:.2e}). Aborting.")
        return

    print(f"Identified {len(ranked_features)} significant candidates. Top: {ranked_features[0]}")

    print("Loading and binning raw continuous input data...")
    with open(input_data_path, 'rb') as f:
        raw_data = pickle.load(f)

    hp = settings['hyperparameters']['jax_linear']['bivariate_gaussian']
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

    print(f"Random Seed: {random_seed} | Num Splits: {n_splits} | Split Strategy: Spatial Proxy ({split_strategy.upper()})")

    cv_folds = NeuralContinuousModelRunner.get_stratified_spatial_splits_stable(
        groups=groups_global,
        Y=y_global,
        n_clusters=n_clusters,
        test_prop=test_prop,
        n_splits=n_splits,
        split_strategy=split_strategy,
        random_seed=random_seed
    )

    current_model_features = []
    best_current_score = float('inf')
    best_current_se = 0.0
    step_counter = 0

    fname = os.path.basename(univariate_results_path)
    cond_match = re.search(r'(male|female.*?)(?=_splits|_lam|_gmm|\.pkl)', fname)
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
                best_cand_in_file = min(cand_dict.items(), key=lambda x: x[1]['mean_nll'])
                name, stats = best_cand_in_file

                if (best_current_score - stats['mean_nll']) > stats['se_nll']:
                    if name not in current_model_features:
                        current_model_features.append(name)
                    best_current_score = stats['mean_nll']
                    best_current_se = stats['se_nll']
                    step_counter = last_step + 1
                else:
                    print("[RESUME] Selection already converged. Stopping loop.")
                    step_counter = last_step
        except Exception as e:
            print(f"Resume failed: {e}. Starting fresh.")
            existing_steps = []

    # Calculate Model-Free Baseline (Step 0) if starting fresh
    if not existing_steps:
        print("\n--- Establishing Absolute Baseline (Model-Free Spatial Density Prior) ---")

        baseline_metrics = {
            'nll_weighted': [], 'nll_raw': [], 'euclidean_mae': [],
            'mahalanobis_dist': [], 'r2_spatial': []
        }

        baseline_data = {
            'folds': {
                'metrics': baseline_metrics,
                'test_indices': [],
                'y_true': [],
                'w_test': [],
                'y_pred_params': []
            }
        }

        for fold_idx, (tr_idx, te_idx) in enumerate(cv_folds):
            Y_tr, Y_te = y_global[tr_idx], y_global[te_idx]
            w_tr, w_te = w_global[tr_idx], w_global[te_idx]

            mu = np.average(Y_tr, axis=0, weights=w_tr)
            diff = Y_tr - mu
            cov = np.average(diff[:, :, None] * diff[:, None, :], axis=0, weights=w_tr)

            sig_x = np.sqrt(cov[0, 0])
            sig_y = np.sqrt(cov[1, 1])
            rho = np.clip(cov[0, 1] / (sig_x * sig_y), -0.99, 0.99)

            dx = Y_te[:, 0] - mu[0]
            dy = Y_te[:, 1] - mu[1]
            z = (dx / sig_x) ** 2 - 2 * rho * (dx / sig_x) * (dy / sig_y) + (dy / sig_y) ** 2
            det = sig_x * sig_y * np.sqrt(1 - rho ** 2)
            log_pdf = -np.log(2 * np.pi * det) - z / (2 * (1 - rho ** 2))

            sse = np.sum((Y_te - mu) ** 2)
            sst = np.sum((Y_te - np.mean(Y_te, axis=0)) ** 2)

            f_met = baseline_data['folds']['metrics']
            f_met['nll_raw'].append(float(-np.mean(log_pdf)))
            f_met['nll_weighted'].append(float(-np.average(log_pdf, weights=w_te)))
            f_met['euclidean_mae'].append(float(np.mean(np.sqrt(dx ** 2 + dy ** 2))))
            f_met['mahalanobis_dist'].append(float(np.mean(np.sqrt(z / (1 - rho ** 2)))))
            f_met['r2_spatial'].append(float(1.0 - (sse / sst)) if sst > 0 else 0.0)

            static_params = np.array([mu[0], mu[1], sig_x ** 2, sig_y ** 2, rho], dtype=np.float32)
            y_pred_params = np.tile(static_params, (len(Y_te), 1))

            baseline_data['folds']['test_indices'].append(te_idx)
            baseline_data['folds']['y_true'].append(Y_te)
            baseline_data['folds']['w_test'].append(w_te)
            baseline_data['folds']['y_pred_params'].append(y_pred_params)

        valid_baseline = [m for m in baseline_data['folds']['metrics']['nll_weighted'] if not np.isnan(m)]
        best_current_score = np.mean(valid_baseline)
        best_current_se = np.std(valid_baseline, ddof=1) / np.sqrt(len(valid_baseline))

        print(f"  Baseline Global NLL established at: {best_current_score:.4f}")

        baseline_data['mean_nll'] = best_current_score
        baseline_data['se_nll'] = best_current_se

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
                'metrics': {m: [] for m in ['nll_weighted', 'nll_raw', 'euclidean_mae', 'mahalanobis_dist', 'r2_spatial']},
                'weights': [], 'intercepts': [], 'global_vars': [],
                'test_indices': [], 'y_true': [], 'w_test': [], 'y_pred_params': []
            }
        }

        for fold_idx, (tr_idx, te_idx) in enumerate(cv_folds):
            try:
                X_tr, X_te = binned_data[anchor][tr_idx], binned_data[anchor][te_idx]
                Y_tr, Y_te = y_global[tr_idx], y_global[te_idx]
                w_tr, w_te = w_global[tr_idx], w_global[te_idx]

                model = SmoothBivariateGaussianRegression(
                    n_features=1, n_time_bins=n_time_bins,
                    lambda_smooth=hp['lambda_smooth'], l2_reg=hp['l2_reg'],
                    learning_rate=hp['learning_rate'], max_iter=hp['max_iter'],
                    tol=hp['tol'], random_state=hp['random_state'] + fold_idx
                )
                model.fit(X_tr, Y_tr, sample_weight=w_tr)

                metrics = model.evaluate_metrics(X_te, Y_te, weights=w_te)
                y_pred_params = model.predict_density(X_te)

                f_met = cand_data['folds']['metrics']
                f_met['nll_weighted'].append(metrics['nll_weighted'])
                f_met['nll_raw'].append(metrics['nll_raw'])
                f_met['euclidean_mae'].append(metrics['euclidean_mae'])
                f_met['mahalanobis_dist'].append(metrics['mahalanobis_dist'])
                f_met['r2_spatial'].append(metrics['r2_spatial'])

                cand_data['folds']['weights'].append(model.coef_)
                cand_data['folds']['intercepts'].append(model.intercept_)
                cand_data['folds']['global_vars'].append(model.global_var_params_)
                cand_data['folds']['test_indices'].append(te_idx)
                cand_data['folds']['y_true'].append(Y_te)
                cand_data['folds']['w_test'].append(w_te)
                cand_data['folds']['y_pred_params'].append(y_pred_params)

                del model
                gc.collect()
            except Exception as e:
                print(f"    [!] Error fitting anchor (Fold {fold_idx}): {e}")
                cand_data['folds']['metrics']['nll_weighted'].append(np.nan)

        valid_nll = [m for m in cand_data['folds']['metrics']['nll_weighted'] if not np.isnan(m)]
        if valid_nll:
            mean_anc_nll = np.mean(valid_nll)
            se_anc_nll = np.std(valid_nll, ddof=1) / np.sqrt(len(valid_nll))

            cand_data['mean_nll'] = mean_anc_nll
            cand_data['se_nll'] = se_anc_nll

            # Test Anchor against Baseline (1SE Rule)
            if (best_current_score - mean_anc_nll) > se_anc_nll:
                print(f"  *** ANCHOR ACCEPTED: NLL dropped to {mean_anc_nll:.4f} ***")
                best_current_score = mean_anc_nll
                best_current_se = se_anc_nll
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
        print(f"\n=== Step {step_counter} === Best NLL: {best_current_score:.5f}")
        step_results = {
            'step_idx': step_counter, 'current_features': list(current_model_features),
            'baseline_score': best_current_score, 'candidates_summary': {},
            'selected_feature': None
        }

        best_cand, best_cand_score, best_cand_se = None, float('inf'), 0.0

        for i_feat, feat in enumerate(ranked_features):
            if feat in current_model_features: continue
            gc.collect()
            print(f"  [{i_feat + 1}/{len(ranked_features)}] Testing +{feat}...", end="", flush=True)

            trial_feats = current_model_features + [feat]
            n_trial_feats = len(trial_feats)

            cand_data = {
                'folds': {
                    'metrics': {m: [] for m in ['nll_weighted', 'nll_raw', 'euclidean_mae', 'mahalanobis_dist', 'r2_spatial']},
                    'weights': [], 'intercepts': [], 'global_vars': [],
                    'test_indices': [], 'y_true': [], 'w_test': [], 'y_pred_params': []
                }
            }

            for fold_idx, (tr_idx, te_idx) in enumerate(cv_folds):
                try:
                    X_tr_stacked = np.hstack([binned_data[f][tr_idx] for f in trial_feats])
                    X_te_stacked = np.hstack([binned_data[f][te_idx] for f in trial_feats])
                    Y_tr, Y_te = y_global[tr_idx], y_global[te_idx]
                    w_tr, w_te = w_global[tr_idx], w_global[te_idx]

                    adjusted_l2 = hp['l2_reg'] / np.sqrt(n_trial_feats)

                    model = SmoothBivariateGaussianRegression(
                        n_features=n_trial_feats, n_time_bins=n_time_bins,
                        lambda_smooth=hp['lambda_smooth'], l2_reg=adjusted_l2,
                        learning_rate=hp['learning_rate'], max_iter=hp['max_iter'],
                        tol=hp['tol'], random_state=hp['random_state'] + fold_idx
                    )
                    model.fit(X_tr_stacked, Y_tr, sample_weight=w_tr)

                    metrics = model.evaluate_metrics(X_te_stacked, Y_te, weights=w_te)
                    y_pred_params = model.predict_density(X_te_stacked)

                    f_met = cand_data['folds']['metrics']
                    f_met['nll_weighted'].append(metrics['nll_weighted'])
                    f_met['nll_raw'].append(metrics['nll_raw'])
                    f_met['euclidean_mae'].append(metrics['euclidean_mae'])
                    f_met['mahalanobis_dist'].append(metrics['mahalanobis_dist'])
                    f_met['r2_spatial'].append(metrics['r2_spatial'])

                    cand_data['folds']['weights'].append(model.coef_)
                    cand_data['folds']['intercepts'].append(model.intercept_)
                    cand_data['folds']['global_vars'].append(model.global_var_params_)
                    cand_data['folds']['test_indices'].append(te_idx)
                    cand_data['folds']['y_true'].append(Y_te)
                    cand_data['folds']['w_test'].append(w_te)
                    cand_data['folds']['y_pred_params'].append(y_pred_params)

                    del model, X_tr_stacked, X_te_stacked
                    gc.collect()
                except Exception as e:
                    print(f"    [!] Error fitting {feat} (Fold {fold_idx}): {e}")
                    cand_data['folds']['metrics']['nll_weighted'].append(np.nan)

            valid_nll = [x for x in cand_data['folds']['metrics']['nll_weighted'] if not np.isnan(x)]
            if valid_nll:
                mean_nll, se_nll = np.mean(valid_nll), np.std(valid_nll, ddof=1) / np.sqrt(len(valid_nll))
                mean_r2 = np.nanmean(cand_data['folds']['metrics']['r2_spatial'])
                print(f" NLL: {mean_nll:.4f} | R2: {mean_r2:.4f}")

                cand_data['mean_nll'] = mean_nll
                cand_data['se_nll'] = se_nll
                step_results['candidates_summary'][feat] = cand_data

                if mean_nll < best_cand_score:
                    best_cand_score, best_cand_se, best_cand = mean_nll, se_nll, feat
            else:
                print(" Failed.")

        if best_cand and (best_current_score - best_cand_score) > best_cand_se:
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

def coarse_binary_model_selection(
        univariate_results_path: str,
        input_data_path: str,
        output_directory: str,
        settings_path: str = None,
        use_top_rank_as_anchor: bool = False,
        p_val: float = 0.01
) -> None:
    """
    Performs forward stepwise selection for the Hierarchical Stage 1 "Gatekeeper" model.

    Scientific and Computational Purpose:
    ------------------------------------
    This function identifies the optimal combination of behavioral features that
    discriminates between 'Simple' (categories 3, 4, 5) and 'Complex' (categories 1, 2, 6)
    USV families. By isolating family-level kinematic signatures, it reduces the
    label entropy for subsequent within-family classification stages.

    The selection process utilizes a JAX-accelerated smooth logistic regression
    framework. Unlike the flat multinomial model, this stage optimizes for family
    separation, using ROC-AUC as the primary decision metric to handle the frequency
    imbalance between simple chirps and complex trills.

    Algorithmic Logic:
    ------------------
    1.  Binary Screening: Filters univariate features based on an 'Actual vs. Null'
        distribution test. A feature is accepted as a candidate only if its mean
        Actual ROC-AUC exceeds the 1-p_val percentile of its session-shuffled
        null distribution.
    2.  Forward Stepwise Selection: Implements a greedy search algorithm. At each
        iteration, every remaining candidate is added to the current model
        features. The feature providing the highest improvement is accepted
        only if it satisfies the One-Standard-Error (1SE) rule.
    3.  Full Metric Integration: For every fold of every candidate tested, the
        function calculates and persists:
        - AUC (ROC): Global separability of Simple vs. Complex.
        - AUC (PR): Precision-Recall balance, critical for rare complex categories.
        - Score: Balanced Accuracy, neutralizing the bias toward majority classes.
        - Log-Loss (LL): The probabilistic cross-entropy used by the optimizer.

    Data Persistence & Hierarchical Routing (Update):
    -------------------------------------------------
    Each step generates a '.pkl' file containing the complete experimental state,
    including fold-level weights, intercepts, true labels, and softmax probabilities.
    Crucially, it now explicitly retains the 'y_original' multiclass labels (1-6)
    and 'test_indices' for every fold. This allows Stage 2 scripts to perform
    "Data Purification" (filtering true positives/negatives into sub-category models)
    directly from the selection outputs without needing to reload the raw data.

    Parameters
    ----------
    univariate_results_path : str
        Path to the results of 'run_coarse_simple_complex_univariate_training'.
    input_data_path : str
        Path to the raw extracted feature data .pkl file.
    output_directory : str
        Directory to save step-wise selection checkpoints.
    settings_path : str, optional
        Path to modeling_settings.json. Defaults to the project standard path.
    use_top_rank_as_anchor : bool, default=False
        If True, initializes Step 0 with the highest-ranked significant feature
        from the univariate pass.
    p_val : float, default=0.01
        Significance threshold for the initial univariate screening pass
        (Bonferroni-corrected internally).

    Returns
    -------
    None
        Results are saved to disk as a series of structured pickle files.
    """

    print("--- Starting Hierarchical Stage 1 (Simple vs Complex) Model Selection ---")

    # 1. Environment and Settings Setup
    if settings_path is None:
        settings_path_obj = pathlib.Path(__file__).resolve().parent.parent / '_parameter_settings/modeling_settings.json'
        settings_path = str(settings_path_obj)

    try:
        with open(settings_path, 'r') as f:
            settings = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Settings file not found at {settings_path}")

    os.makedirs(output_directory, exist_ok=True)

    # 2. Univariate Screening (Actual vs. Null)
    with open(univariate_results_path, 'rb') as f:
        univariate_data = pickle.load(f)

    candidates = []
    num_total_features = len(univariate_data)

    for feat_name, payload in univariate_data.items():
        if isinstance(payload, tuple) and len(payload) == 2:
            _, results = payload
        else:
            results = payload

        if 'actual' not in results:
            continue

        actual_auc = np.array(results['actual']['folds']['metrics']['auc'])
        null_auc = np.array(results['null']['folds']['metrics']['auc'])

        valid_actual = actual_auc[~np.isnan(actual_auc)]
        valid_null = null_auc[~np.isnan(null_auc)]

        if (len(valid_actual) == 0 or len(valid_null) == 0):
            continue

        mean_actual_auc = np.mean(valid_actual)
        null_threshold = np.percentile(valid_null, q=100 - ((p_val / num_total_features) * 100))

        if mean_actual_auc > null_threshold:
            candidates.append({'feature': feat_name, 'mean_auc': mean_actual_auc})

    candidates.sort(key=lambda x: x['mean_auc'], reverse=True)
    ranked_features = [x['feature'] for x in candidates]

    if not ranked_features:
        print("No significant features found. Selection aborted.")
        return

    print(f"Significant candidates: {len(ranked_features)}. Top: {ranked_features[0]}")

    # 3. Data Preparation and Binary Mapping
    hp = settings['hyperparameters']['jax_linear']['multinomial_logistic']
    voc_settings = settings['vocal_features']
    complex_cats = voc_settings['usv_complex_categories']
    bin_size = hp['bin_resizing_factor']

    print("\n--- Initializing Pipeline to load identically scaled data ---")

    # Utilizing the global classes to ensure Z-scoring/scaling identically matches the univariate run
    pipeline = MultinomialModelingPipeline(modeling_settings_dict=settings)
    runner = MultinomialModelRunner(pipeline_instance=pipeline)

    all_blocks = runner.load_univariate_data_blocks(input_data_path, bin_size=bin_size)

    binned_data = {}
    y_binary_global = None
    y_original_global = None
    groups_global = None
    n_time_bins = None

    for feat in ranked_features:
        if feat not in all_blocks:
            continue

        feat_data = all_blocks[feat]
        X_s = feat_data['X']
        y_s = feat_data['y']
        g_s = feat_data['groups']

        y_bin = np.zeros_like(y_s)
        y_bin[np.isin(y_s, complex_cats)] = 1

        binned_data[feat] = X_s.astype(np.float32)

        if y_binary_global is None:
            y_binary_global = y_bin.astype(np.int32)
            y_original_global = y_s
            groups_global = g_s
            n_time_bins = X_s.shape[1]

    del all_blocks
    gc.collect()

    # 4. Cross-Validation Split Strategy (Restored session/mixed logic)
    model_ops = settings['model_params']
    split_strategy = model_ops['split_strategy']
    n_splits = model_ops['split_num']
    test_prop = model_ops['test_proportion']
    random_seed = settings['model_params']['random_seed']

    if split_strategy == 'session':
        cv_folds = get_stratified_group_splits_stable(
            groups=groups_global,
            y=y_binary_global,
            test_prop=test_prop,
            n_splits=n_splits,
            random_seed=random_seed
        )
    else:
        sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_prop, random_state=random_seed)
        cv_folds = list(sss.split(binned_data[ranked_features[0]], y_binary_global))

    current_model_features = []
    best_current_score = 0.5
    best_current_se = 0.0
    step_counter = 0
    prefix = f"model_selection_hierarchical_stage1_{split_strategy}_step_"

    # 5. Auto-Anchor Logic
    if use_top_rank_as_anchor:
        anchor = ranked_features[0]
        print(f"\n*** AUTO-ANCHOR: Initializing Stage 1 with {anchor} ***")

        anchor_data = {
            'folds': {
                'metrics': {m: [] for m in ['auc', 'auc_pr', 'score', 'll']},
                'weights': [], 'intercepts': [], 'y_true': [], 'y_probs': [],
                'y_original': [], 'test_indices': []
            },
            'classes': [0, 1]
        }

        for tr_idx, te_idx in cv_folds:
            X_tr, X_te = binned_data[anchor][tr_idx], binned_data[anchor][te_idx]
            y_tr, y_te = y_binary_global[tr_idx], y_binary_global[te_idx]
            y_orig_te = y_original_global[te_idx]

            model = SmoothMultinomialLogisticRegression(
                n_features=1, n_time_bins=n_time_bins,
                lambda_smooth=hp['lambda_smooth'], l1_reg=hp['l1_reg'], l2_reg=hp['l2_reg'],
                learning_rate=hp['learning_rate'], max_iter=hp['max_iter'],
                random_state=hp['random_state']
            )
            model.fit(X_tr, y_tr)

            probs = model.predict_proba(X_te, balanced=hp['balance_predictions_bool'])[:, 1]
            preds = (probs >= 0.5).astype(int)

            eps = 1e-15
            probs_clipped = np.clip(probs, eps, 1 - eps)
            prob_matrix = np.column_stack([1 - probs_clipped, probs_clipped])

            try:
                anchor_data['folds']['metrics']['auc'].append(roc_auc_score(y_te, probs))
                prec, rec, _ = precision_recall_curve(y_te, probs)
                anchor_data['folds']['metrics']['auc_pr'].append(auc(rec, prec))
                anchor_data['folds']['metrics']['score'].append(balanced_accuracy_score(y_te, preds))
                anchor_data['folds']['metrics']['ll'].append(log_loss(y_te, prob_matrix, labels=[0, 1]))
            except ValueError:
                for m in ['auc', 'auc_pr', 'score', 'll']:
                    anchor_data['folds']['metrics'][m].append(np.nan)

            anchor_data['folds']['y_true'].append(y_te)
            anchor_data['folds']['y_probs'].append(probs)
            anchor_data['folds']['y_original'].append(y_orig_te)
            anchor_data['folds']['test_indices'].append(te_idx)
            anchor_data['folds']['weights'].append(model.coef_)
            anchor_data['folds']['intercepts'].append(model.intercept_)

        valid_auc = [v for v in anchor_data['folds']['metrics']['auc'] if not np.isnan(v)]
        if np.mean(valid_auc) > 0.5:
            best_current_score = np.mean(valid_auc)
            best_current_se = np.std(valid_auc, ddof=1) / np.sqrt(len(valid_auc))
            current_model_features = [anchor]

            step_0_res = {
                'step_idx': 0,
                'current_features': [anchor],
                'baseline_score': best_current_score,
                'selected_feature': anchor,
                'candidates_summary': {anchor: anchor_data}
            }
            with open(os.path.join(output_directory, f"{prefix}0.pkl"), 'wb') as f:
                pickle.dump(step_0_res, f)
            step_counter = 1

    # 6. Main Forward Selection Loop
    while True:
        print(f"\n=== Step {step_counter} === Best ROC-AUC: {best_current_score:.5f}")
        step_results = {
            'step_idx': step_counter,
            'current_features': list(current_model_features),
            'baseline_score': best_current_score,
            'candidates_summary': {},
            'selected_feature': None
        }
        best_cand, best_cand_score, best_cand_se = None, 0.0, 0.0

        for feat in ranked_features:
            if feat in current_model_features:
                continue
            gc.collect()
            print(f"  Testing +{feat}...", end="", flush=True)

            trial_feats = current_model_features + [feat]
            n_trial_feats = len(trial_feats)
            cand_data = {
                'folds': {
                    'metrics': {m: [] for m in ['auc', 'auc_pr', 'score', 'll']},
                    'weights': [], 'intercepts': [], 'y_true': [], 'y_probs': [],
                    'y_original': [], 'test_indices': []
                }
            }

            for tr_idx, te_idx in cv_folds:
                try:
                    X_tr_stacked = np.hstack([binned_data[f][tr_idx] for f in trial_feats])
                    X_te_stacked = np.hstack([binned_data[f][te_idx] for f in trial_feats])
                    y_tr, y_te = y_binary_global[tr_idx], y_binary_global[te_idx]
                    y_orig_te = y_original_global[te_idx]

                    model = SmoothMultinomialLogisticRegression(
                        n_features=n_trial_feats, n_time_bins=n_time_bins,
                        lambda_smooth=hp['lambda_smooth'],
                        l2_reg=hp['l2_reg'] / np.sqrt(n_trial_feats),
                        learning_rate=hp['learning_rate'], max_iter=hp['max_iter'],
                        random_state=hp['random_state']
                    )
                    model.fit(X_tr_stacked, y_tr)

                    probs = model.predict_proba(X_te_stacked, balanced=hp['balance_predictions_bool'])[:, 1]
                    preds = (probs >= 0.5).astype(int)

                    eps = 1e-15
                    probs_clipped = np.clip(probs, eps, 1 - eps)
                    prob_matrix = np.column_stack([1 - probs_clipped, probs_clipped])

                    cand_data['folds']['metrics']['ll'].append(log_loss(y_te, prob_matrix, labels=[0, 1]))
                    cand_data['folds']['metrics']['auc'].append(roc_auc_score(y_te, probs))
                    prec, rec, _ = precision_recall_curve(y_te, probs)
                    cand_data['folds']['metrics']['auc_pr'].append(auc(rec, prec))
                    cand_data['folds']['metrics']['score'].append(balanced_accuracy_score(y_te, preds))

                    cand_data['folds']['weights'].append(model.coef_)
                    cand_data['folds']['intercepts'].append(model.intercept_)
                    cand_data['folds']['y_true'].append(y_te)
                    cand_data['folds']['y_probs'].append(probs)
                    cand_data['folds']['y_original'].append(y_orig_te)
                    cand_data['folds']['test_indices'].append(te_idx)
                except Exception:
                    cand_data['folds']['metrics']['auc'].append(np.nan)

            valid_auc = [v for v in cand_data['folds']['metrics']['auc'] if not np.isnan(v)]
            if valid_auc:
                mean_auc = np.mean(valid_auc)
                se_auc = np.std(valid_auc, ddof=1) / np.sqrt(len(valid_auc))
                print(f" AUC: {mean_auc:.4f}")
                cand_data['mean_auc'], cand_data['se_auc'] = mean_auc, se_auc
                step_results['candidates_summary'][feat] = cand_data
                if mean_auc > best_cand_score:
                    best_cand_score, best_cand_se, best_cand = mean_auc, se_auc, feat
            else:
                print(" Failed.")

        if best_cand and (best_cand_score - best_current_score) > best_cand_se:
            print(f"  ACCEPT {best_cand}")
            step_results['selected_feature'] = best_cand
            current_model_features.append(best_cand)
            with open(os.path.join(output_directory, f"{prefix}{step_counter}.pkl"), 'wb') as f:
                pickle.dump(step_results, f)
            best_current_score, best_current_se, step_counter = best_cand_score, best_cand_se, step_counter + 1
        else:
            print("  REJECT. Stage 1 Selection Converged.")
            step_results['selected_feature'] = None
            with open(os.path.join(output_directory, f"{prefix}{step_counter}.pkl"), 'wb') as f:
                pickle.dump(step_results, f)
            break

        if len(current_model_features) == len(ranked_features):
            break

    # 7. Final visualization data promotion
    print("\n--- Finalizing Stage 1 Selection Results ---")
    try:
        last_step_idx = max(0, step_counter - 1)
        last_file = os.path.join(output_directory, f"{prefix}{last_step_idx}.pkl")
        with open(last_file, 'rb') as f:
            final_step = pickle.load(f)

        winner = final_step['selected_feature'] or current_model_features[-1]
        raw_weights = np.array(final_step['candidates_summary'][winner]['folds']['weights'])
        n_features_final, n_classes = len(current_model_features), raw_weights.shape[1]

        final_step['weights_reshaped'] = raw_weights.reshape(n_splits, n_classes, n_features_final, n_time_bins)
        final_step['final_model_features'] = current_model_features
        with open(last_file, 'wb') as f:
            pickle.dump(final_step, f)
        print(f"Final promotion successful for step {last_step_idx}.")
    except Exception as e:
        print(f"Final promotion failed: {e}")
