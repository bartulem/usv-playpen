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
    recall_score, roc_auc_score,
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
    brier_score_multi,
    expected_calibration_error,
    safe_matthews_corrcoef,
    safe_confusion_matrix,
    align_probs_to_canonical,
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
        n_categories: int = 6,
        max_total_attempts: int = 50000,
        widen_step: float = 0.02,
        widen_every: int = 1000
) -> tuple:
    """
    Generates `n_splits` independent train/test index pairs and records the
    splitter metadata needed to audit the realized class distribution of each
    fold.

    Two strategies are supported:

    - 'mixed':  sessions are ignored; `StratifiedShuffleSplit` is used, which
                perfectly stratifies the `n_categories` labels at the configured
                test proportion. Both train and test folds therefore preserve
                the natural class prior.
    - 'session': whole sessions are sampled into test / train. A fold is only
                accepted if every class is represented in both halves and the
                maximum absolute deviation between the test-set empirical class
                distribution and the global class distribution is below the
                current `tolerance`. If the sampler cannot find enough valid
                folds, `tolerance` is widened by `widen_step` every
                `widen_every` failed attempts (up to `max_total_attempts`).
                The `tolerance` that was in force at the moment each fold was
                accepted is recorded per fold so downstream analyses can
                audit / filter lenient folds.

    Parameters
    ----------
    groups : np.ndarray
        Array of session IDs (ensures samples from the same session stay together
        for 'session' strategy).
    y : np.ndarray
        Array of USV category labels.
    split_strategy : {'session', 'mixed'}, default='session'
        Splitting rule (see above).
    test_prop : float, default=0.2
        Proportion of sessions ('session') or samples ('mixed') routed to test.
    n_splits : int, default=100
        Number of independent folds to return.
    tolerance : float, default=0.05
        Initial allowable max per-class distribution deviation between the
        test fold and the global data (only used by 'session').
    random_seed : int, default=0
        Seed for reproducibility.
    n_categories : int, default=6
        Expected number of USV categories.
    max_total_attempts : int, default=50000
        Hard ceiling on rejection-sampling attempts before raising.
    widen_step : float, default=0.02
        Amount by which `tolerance` is increased each time the sampler fails
        to accept a fold for `widen_every` consecutive attempts.
    widen_every : int, default=1000
        Number of failed attempts between successive tolerance widenings.

    Returns
    -------
    cv_folds : list of tuple[np.ndarray, np.ndarray]
        List of `(train_idx, test_idx)` pairs.
    fold_tolerances : list of float
        Length `n_splits`. For 'session', the `tolerance` in force when each
        fold was accepted. For 'mixed', always `0.0` (perfectly stratified).
    """

    if split_strategy not in ['session', 'mixed']:
        raise ValueError("split_strategy must be 'session' or 'mixed'.")

    # MIXED STRATEGY: Ignores sessions, perfectly stratifies the 6 categories
    if split_strategy == 'mixed':
        sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_prop, random_state=random_seed)
        cv_folds = list(sss.split(np.zeros(len(y)), y))
        fold_tolerances = [0.0] * len(cv_folds)
        return cv_folds, fold_tolerances

    # SESSION STRATEGY: Strict cross-session prediction (your original logic)
    unique_sessions = np.unique(groups)
    n_test_sessions = int(len(unique_sessions) * test_prop)

    # Calculate global distribution for fairness check
    _, global_counts = np.unique(y, return_counts=True)
    global_dist = global_counts / len(y)

    cv_folds = []
    fold_tolerances = []
    rng = np.random.RandomState(random_seed)

    attempts = 0
    current_tolerance = tolerance

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
                fold_tolerances.append(float(current_tolerance))

        # widen the net every `widen_every` failures
        if attempts % widen_every == 0:
            current_tolerance += widen_step
            print(
                f"[session-splits] widening tolerance -> {current_tolerance:.3f} "
                f"after {attempts} attempts ({len(cv_folds)}/{n_splits} folds accepted)."
            )

        # prevent infinite loops
        if attempts > max_total_attempts:
            raise RuntimeError(
                f"Failed to find {n_splits} valid splits after {attempts} attempts. "
                "The rare categories may be concentrated in too few sessions."
            )

    return cv_folds, fold_tolerances


def _balance_multinomial_train_indices(
        train_idx: np.ndarray,
        y: np.ndarray,
        rng: np.random.Generator
) -> np.ndarray:
    """
    Down-samples `train_idx` so every USV class contributes the same number of
    rows (`min(class_count)`). Index order is shuffled so downstream consumers
    can slice the balanced subset directly.

    Parameters
    ----------
    train_idx : np.ndarray
        Absolute row indices (into the full design matrix) that belong to the
        training fold.
    y : np.ndarray
        Integer class labels aligned to the full design matrix.
    rng : np.random.Generator
        Seeded generator used for per-class sampling (and final shuffling).

    Returns
    -------
    balanced_idx : np.ndarray
        Subset of `train_idx` with every class represented `min_count` times.
    """

    y_train = y[train_idx]
    classes, counts = np.unique(y_train, return_counts=True)
    min_count = int(counts.min())

    picked = []
    for cls in classes:
        cls_positions = train_idx[y_train == cls]
        if len(cls_positions) == min_count:
            picked.append(cls_positions)
        else:
            picked.append(rng.choice(cls_positions, size=min_count, replace=False))

    balanced_idx = np.concatenate(picked)
    rng.shuffle(balanced_idx)
    return balanced_idx


def _log_spaced_grid_multinomial(center: float, decades_each_side: int) -> np.ndarray:
    """
    Returns a log-spaced grid of candidate regularisation strengths centred
    on `center`, spanning `decades_each_side` orders of magnitude on each
    side.

    The multinomial tuner consumes the Cartesian product of two such
    grids (one for `lambda_smooth`, one for `l2_reg`), so the centre is
    always the user-supplied fixed value from the settings block and the
    half-width controls how aggressively the inner CV is allowed to
    stray from it.

    Example
    -------
    `_log_spaced_grid_multinomial(center=1.0, decades_each_side=3)`
    returns `[1e-3, 1e-2, 1e-1, 1e+0, 1e+1, 1e+2, 1e+3]`.
    `decades_each_side=0` returns `[center]` — the "no tuning" degenerate
    case that collapses the grid to the fixed value.

    Parameters
    ----------
    center : float
        Centre of the grid. This is the fixed "best guess" value from the
        settings block (`lambda_smooth_fixed` or `l2_reg_fixed`); the grid
        searches a symmetric window of orders of magnitude around it.
        Must be strictly positive.
    decades_each_side : int
        Half-width of the search window in decades. Must be non-negative;
        `0` collapses the grid to the single value `[center]`.

    Returns
    -------
    grid : np.ndarray
        Sorted 1-D array of length `2 * decades_each_side + 1`.
    """

    if decades_each_side < 0:
        raise ValueError(f"decades_each_side must be >= 0, got {decades_each_side}.")
    if center <= 0:
        raise ValueError(f"center must be positive, got {center}.")
    offsets = np.arange(-decades_each_side, decades_each_side + 1, dtype=np.float64)
    return float(center) * (10.0 ** offsets)


def _tune_multinomial_regularization(X_train: np.ndarray,
                                     y_train: np.ndarray,
                                     *,
                                     lambda_smooth_grid: np.ndarray,
                                     l2_reg_grid: np.ndarray,
                                     inner_cv_folds: int,
                                     inner_cv_scoring_metric: str,
                                     inner_cv_use_one_se_rule: bool,
                                     n_features: int,
                                     n_time_bins: int,
                                     smoothness_derivative_order: int,
                                     focal_gamma: float,
                                     uniform_class_weights: bool,
                                     learning_rate: float,
                                     max_iter: int,
                                     inner_max_iter: int,
                                     tol: float,
                                     random_state: int,
                                     verbose: bool,
                                     use_lax_loop: bool,
                                     regressor_cls) -> tuple:
    """
    Selects `(lambda_smooth, l2_reg)` jointly for a multinomial logistic
    fit by stratified inner cross-validation on the supplied training
    fold.

    `focal_gamma` and `uniform_class_weights` are intentionally **not**
    tuned — they are treated as structural loss-modelling choices (how
    imbalanced is the dataset, how aggressively should hard samples
    drive the gradient) rather than regularisation dials. They are
    passed through verbatim on every inner fit.

    Strategy mirrors `_tune_manifold_regularization` in the manifold
    runner:
    1. Partition the training fold with `StratifiedShuffleSplit` on the
       class labels (`inner_cv_folds` splits, test_prop = 1 /
       inner_cv_folds). Session-mixing inside the outer training fold
       is acceptable so the inner splitter doesn't need a second layer
       of session-holdout logic.
    2. For every `(lambda_smooth, l2_reg)` pair, fit the regressor on
       each inner-training sub-fold and score it on the inner-validation
       sub-fold using `inner_cv_scoring_metric`. Aggregate per-pair
       across inner folds to a mean score and a standard error.
    3. Identify the argmax pair. If `inner_cv_use_one_se_rule=True`,
       return the smoothest pair whose mean score is within one SE of
       the argmax (tiebreak: smallest `l2_reg`). This biases the choice
       toward filter interpretability by refusing to chase wiggly
       filters whose apparent R^2 / AUC gain is statistically
       indistinguishable from noise.

    Supported scoring metrics
    -------------------------
    - higher-is-better: `auc` (macro OvR), `score` (balanced accuracy),
      `f1` (macro), `recall` (macro), `mcc` (Matthews correlation).
    - lower-is-better: `ll` (log-loss), `brier` (multiclass Brier),
      `ece` (expected calibration error).

    Parameters mirror the manifold tuner; the `regressor_cls` injection
    keeps the function unit-testable without importing JAX at module
    scope. `inner_max_iter` caps iterations on every inner fit — kept
    smaller than the outer `max_iter` so the tuner produces usable
    ranking scores without paying full-convergence wall time per pair.

    Returns
    -------
    best_lambda_smooth : float
    best_l2_reg : float
    grid_audit : dict
        Keys `grid_scores`, `grid_ses`, `argmax_pair`, `one_se_applied`,
        `one_se_threshold`. See `_tune_manifold_regularization` in the
        manifold pipeline for identical semantics.
    """

    higher_is_better_metrics = {'auc', 'score', 'f1', 'recall', 'mcc'}
    lower_is_better_metrics = {'ll', 'brier', 'ece'}
    if inner_cv_scoring_metric in higher_is_better_metrics:
        higher_is_better = True
    elif inner_cv_scoring_metric in lower_is_better_metrics:
        higher_is_better = False
    else:
        raise ValueError(
            f"Unsupported inner_cv_scoring_metric '{inner_cv_scoring_metric}' for "
            f"the multinomial tuner. Supported: "
            f"{sorted(higher_is_better_metrics | lower_is_better_metrics)}."
        )

    def _empty_audit():
        return {
            'grid_scores': {},
            'grid_ses': {},
            'argmax_pair': None,
            'one_se_applied': False,
            'one_se_threshold': None,
        }

    unique_y = np.unique(y_train)
    n_unique = int(unique_y.size)
    if len(y_train) < inner_cv_folds * 2 or n_unique < 2:
        return (
            float(lambda_smooth_grid[len(lambda_smooth_grid) // 2]),
            float(l2_reg_grid[len(l2_reg_grid) // 2]),
            _empty_audit(),
        )

    sss = StratifiedShuffleSplit(
        n_splits=inner_cv_folds,
        test_size=1.0 / inner_cv_folds,
        random_state=random_state + 7919,
    )
    inner_folds = list(sss.split(np.zeros(len(y_train)), y_train))

    def _compute_score(y_true_, y_pred_, y_proba_, model_classes_):
        try:
            if inner_cv_scoring_metric == 'auc':
                return float(roc_auc_score(
                    y_true_, y_proba_, multi_class='ovr', average='macro',
                    labels=model_classes_,
                ))
            if inner_cv_scoring_metric == 'score':
                return float(balanced_accuracy_score(y_true_, y_pred_))
            if inner_cv_scoring_metric == 'f1':
                return float(f1_score(y_true_, y_pred_, average='macro', zero_division=0))
            if inner_cv_scoring_metric == 'recall':
                return float(recall_score(y_true_, y_pred_, average='macro', zero_division=0))
            if inner_cv_scoring_metric == 'mcc':
                return float(safe_matthews_corrcoef(y_true_, y_pred_))
            if inner_cv_scoring_metric == 'll':
                return float(log_loss(
                    y_true_, np.clip(y_proba_, 1e-15, 1 - 1e-15),
                    labels=model_classes_,
                ))
            if inner_cv_scoring_metric == 'brier':
                return float(brier_score_multi(y_true_, y_proba_, model_classes_))
            if inner_cv_scoring_metric == 'ece':
                return float(expected_calibration_error(y_true_, y_pred_, y_proba_, n_bins=10))
        except (ValueError, RuntimeError):
            return float('nan')
        return float('nan')

    grid_scores = {}
    grid_ses = {}
    for lam_sm in lambda_smooth_grid:
        for lam_l2 in l2_reg_grid:
            fold_scores = []
            for inner_idx, (in_tr, in_va) in enumerate(inner_folds):
                try:
                    model = regressor_cls(
                        n_features=n_features,
                        n_time_bins=n_time_bins,
                        lambda_smooth=float(lam_sm),
                        l2_reg=float(lam_l2),
                        smoothness_derivative_order=smoothness_derivative_order,
                        focal_gamma=focal_gamma,
                        uniform_class_weights=uniform_class_weights,
                        learning_rate=learning_rate,
                        max_iter=inner_max_iter,
                        tol=tol,
                        random_state=random_state + inner_idx,
                        verbose=False,
                        _use_lax_loop=use_lax_loop,
                    )
                    model.fit(X_train[in_tr], y_train[in_tr])
                    y_pred = model.predict(X_train[in_va], balanced=False)
                    y_proba = model.predict_proba(X_train[in_va], balanced=False)
                    fold_scores.append(_compute_score(
                        y_train[in_va], y_pred, y_proba, model.classes_,
                    ))
                except Exception as exc:
                    if verbose:
                        print(
                            f"        [inner-cv] λ_smooth={lam_sm:.3g}, l2={lam_l2:.3g}, "
                            f"fold {inner_idx}: {exc}"
                        )
                    fold_scores.append(np.nan)

            finite = np.asarray([s for s in fold_scores if np.isfinite(s)], dtype=float)
            pair_key = (float(lam_sm), float(lam_l2))
            if finite.size:
                grid_scores[pair_key] = float(np.mean(finite))
                grid_ses[pair_key] = (
                    float(np.std(finite, ddof=1) / np.sqrt(finite.size))
                    if finite.size > 1 else 0.0
                )
            else:
                grid_scores[pair_key] = float('nan')
                grid_ses[pair_key] = float('nan')

    valid_pairs = [(pair, score) for pair, score in grid_scores.items() if np.isfinite(score)]
    if not valid_pairs:
        audit = _empty_audit()
        audit['grid_scores'] = grid_scores
        audit['grid_ses'] = grid_ses
        return (
            float(lambda_smooth_grid[len(lambda_smooth_grid) // 2]),
            float(l2_reg_grid[len(l2_reg_grid) // 2]),
            audit,
        )

    key_fn = (lambda kv: kv[1]) if higher_is_better else (lambda kv: -kv[1])
    argmax_pair, argmax_score = max(valid_pairs, key=key_fn)
    argmax_se = grid_ses.get(argmax_pair, 0.0)
    if not np.isfinite(argmax_se):
        argmax_se = 0.0

    audit = {
        'grid_scores': grid_scores,
        'grid_ses': grid_ses,
        'argmax_pair': (float(argmax_pair[0]), float(argmax_pair[1])),
        'one_se_applied': False,
        'one_se_threshold': None,
    }

    if not inner_cv_use_one_se_rule:
        return float(argmax_pair[0]), float(argmax_pair[1]), audit

    if higher_is_better:
        threshold = argmax_score - argmax_se

        def _within(score_: float) -> bool:
            return score_ >= threshold
    else:
        threshold = argmax_score + argmax_se

        def _within(score_: float) -> bool:
            return score_ <= threshold

    in_band = [pair for pair, score in valid_pairs if _within(score)]
    if not in_band:
        return float(argmax_pair[0]), float(argmax_pair[1]), audit

    # Tiebreak: largest lambda_smooth (smoothness preference), then smallest l2_reg.
    winner = max(in_band, key=lambda p: (p[0], -p[1]))
    audit['one_se_applied'] = True
    audit['one_se_threshold'] = float(threshold)
    return float(winner[0]), float(winner[1]), audit


class MultinomialModelingPipeline(FeatureZoo):
    """
    End-to-end extraction pipeline for multinomial USV-category modelling.

    The pipeline has two responsibilities:

    1. **Data preparation**: loads behavioural time-series and USV category
       assignments, assembles per-session history windows `[onset -
       history_frames, onset)` preceding every valid bout (onsets that would
       require samples before frame 0 are filtered out at collection time so
       no NaN rows leak into the design matrix), applies cross-session z-
       scoring, and serializes the resulting `{feature: {session: {X, y}}}`
       dictionary to disk. The class of each bout (integer category label) is
       retained alongside the design matrix so downstream fitting can solve
       the multinomial problem in a single pass rather than target-vs-rest.
    2. **Identity-guarded vocal predictors**: injects partner-side and,
       when `usv_predictor_partner_only=False`, subject-side vocal syntax
       traces as predictors while strictly excluding the target-category
       density channels to prevent self-prediction / trivial autocorrelation.

    Cross-validation and model fitting live in the separate
    `MultinomialModelRunner` class below, which consumes the pickle produced
    here and drives the JAX-accelerated
    `SmoothMultinomialLogisticRegression` estimator with the "balanced train /
    natural-rate test" invariant.
    """

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

        if 'feature_boundaries' in self.modeling_settings:
            self.feature_boundaries = self.modeling_settings['feature_boundaries']

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
    2. Experimental control: Implements a tri-strategy 'Actual vs. Null vs.
       Null-Model-Free' design so behavioral predictive power is benchmarked
       against both a within-session shuffled null and a class-prior floor.
    3. Cross-validation: Delegates to `get_stratified_group_splits_stable`,
       which dispatches on `split_strategy`:
         - 'mixed':  `StratifiedShuffleSplit` on pooled samples — perfectly
                     stratified across classes, sessions are ignored.
         - 'session': custom rejection sampler that partitions whole sessions
                     and accepts folds whose test-set class distribution stays
                     within a widening tolerance of the global class prior,
                     ensuring no session leaks between train and test.
    4. Optional train-fold balancing: when
       `hyperparameters.jax_linear.multinomial_logistic.balance_train_bool`
       is `true`, each training fold is per-class down-sampled to the minimum
       class count, and the JAX fit is switched to `focal_gamma=0` with
       uniform class weights so focal-alpha does not double-correct an already
       balanced batch. When `false` (default), the natural training-rate path
       is kept and rebalancing happens implicitly inside the loss (softened
       inverse-frequency alpha + focal modulation).
    5. Deep metadata storage: persists raw fold-level data (predictions,
       labels, probabilities, learned weights, realized p_train / p_test,
       per-fold accepted tolerance) rather than just summary statistics,
       enabling downstream confusion matrix analysis, distribution audits,
       and significance testing.

    Metrics saved per fold (stored under `folds.metrics`):
    ------------------------------------------------------
    - `auc` : Macro-averaged one-vs-rest ROC-AUC. Threshold-free ranking
      quality; insensitive to class imbalance but agnostic to calibration.
    - `score` : Balanced accuracy (mean of per-class recall). Hard-label
      accuracy that doesn't reward the model for predicting the majority
      class.
    - `recall` : Macro-averaged recall. Kept; precision is derivable from
      the confusion matrix on demand.
    - `f1` : Macro-averaged F1 (harmonic mean of precision and recall).
    - `ll` : Log-loss (multinomial cross-entropy). Strictly proper
      probabilistic score; penalizes confident wrong predictions harshly.
    - `brier` : Multiclass Brier score (mean squared error between
      predicted probabilities and one-hot truth). Complements log-loss
      with a quadratic (rather than logarithmic) penalty, which is more
      robust to occasional overconfidence.
    - `ece` : Expected Calibration Error (top-label, 10 equal-width bins).
      Reports the average gap between predicted confidence and empirical
      accuracy; near-zero means "70%-confident predictions are correct
      70% of the time."
    - `mcc` : Matthews correlation coefficient. Chance-corrected,
      imbalance-robust single-number summary of the confusion matrix in
      [-1, +1].

    Additional deep-storage fields per fold:
    - `confusion_matrix` : (K, K) matrix with canonical class ordering.
      Cheap to persist, lets downstream code derive precision, class-pair
      confusions, sensitivity, and specificity on demand.
    - `n_iter`, `converged`, `fit_time` : JAX optimizer diagnostics —
      `converged=False` flags folds that terminated at `max_iter` without
      meeting the tolerance (the main silent-failure mode for this
      estimator).

    The runner relies strictly on the 'hyperparameters' and 'model_params'
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
    def load_univariate_data_blocks(pkl_path: str,
                                    bin_size: int = 10,
                                    feature_filter=None) -> dict:
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
        feature_filter : iterable of str or str, optional
            If provided, only bin and return the listed features. The HPC
            dispatcher runs one feature per process, so paying the binning
            cost for every feature in the pickle on every call is wasteful;
            pass the feature(s) actually needed to skip the rest. The default
            (`None`) retains the legacy behaviour of binning every feature in
            the pickle.

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

        if feature_filter is None:
            sorted_features = sorted(list(raw_data.keys()))
        else:
            wanted = {feature_filter} if isinstance(feature_filter, str) else set(feature_filter)
            sorted_features = sorted(feat for feat in raw_data.keys() if feat in wanted)

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

        Splitting & balancing invariant:
        --------------------------------
        The test fold always preserves the natural class prior of the source
        data (stratified in 'mixed' mode, natural-within-tolerance in 'session'
        mode). The training fold follows one of two paths, selected by
        `hyperparameters.jax_linear.multinomial_logistic.balance_train_bool`:

        - `false` (default): the training fold retains the natural class prior,
          and class imbalance is handled inside the JAX loss through softened
          inverse-frequency alpha weights combined with focal-gamma modulation
          (`focal_loss_gamma` in settings).
        - `true`: the training fold is sample-level down-sampled so every class
          contributes `min(class_count)` rows. The JAX fit is then invoked with
          `focal_gamma=0` and uniform `1 / n_classes` class weights to avoid
          double-correcting an already balanced batch. This mirrors the binary
          pipeline's "balanced train / natural-rate test" invariant.

        Reported metrics are imbalance-robust (balanced accuracy, log-loss,
        macro OvR AUC).

        Data Persistence (Deep Storage):
        -------------------------------
        Unlike standard training functions, this method saves all raw fold outputs,
        including true labels, class probabilities, learned weights, realized
        per-fold class proportions (`p_train`, `p_test`), and the split tolerance
        in force when each 'session' fold was accepted. This enables:
        - Post-hoc generation of multi-class Confusion Matrices.
        - Construction of ROC and Precision-Recall curves.
        - Audits of realized train/test class balance per fold.
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
            - 'metrics'       : Dict of performance scores per cross-validation fold.
            - 'weights'       : Learned coefficient matrices per fold (None for model-free).
            - 'intercepts'    : Learned intercept vectors per fold.
            - 'y_true'        : Actual USV labels per fold.
            - 'y_pred'        : Model's hard-choice predictions per fold.
            - 'y_probs'       : Softmax probabilities for all classes per fold.
            - 'test_indices'  : Absolute row indices defining each test fold.
            - 'p_train'       : Realized per-class proportions of the training fold
                                (aligned to the canonical project-wide class order).
            - 'p_test'        : Realized per-class proportions of the test fold.
            - 'tolerance'     : Per-fold accepted distribution-error tolerance
                                (only meaningful for 'session'; 0.0 for 'mixed').
            - 'balanced_train': Whether `balance_train_bool` was active this run.
            Plus top-level 'classes' (training class order for this strategy)
            and 'canonical_classes' (project-wide class ordering for `p_*`).
        """

        # Strict dictionary lookups (No .get() allowed)
        hp = self.modeling_settings['hyperparameters']['jax_linear']['multinomial_logistic']
        n_splits = self.modeling_settings['model_params']['split_num']
        split_strategy = self.modeling_settings['model_params']['split_strategy']
        test_prop = self.modeling_settings['model_params']['test_proportion']
        bin_size = hp['bin_resizing_factor']
        balance_train = hp['balance_train_bool']
        base_seed = self.modeling_settings['model_params']['random_seed']
        n_categories_total = self.modeling_settings['vocal_features']['usv_category_number']

        # Joint-tuning configuration. Fixed-fallback `lambda_smooth_fixed`
        # and `l2_reg_fixed` are used when the toggle is off; when on,
        # every outer fold runs an inner CV over log-spaced grids centred
        # on those fixed values and picks the winning pair subject to the
        # 1-SE interpretability rule if enabled. `focal_loss_gamma` stays
        # structural (data-modelling choice), not part of the grid.
        lam_smooth_fixed = hp['lambda_smooth_fixed']
        lam_l2_fixed = hp['l2_reg_fixed']
        smoothness_order = hp['smoothness_derivative_order']
        tune_regularization_bool = hp['tune_regularization_bool']
        tune_params = hp['tune_regularization_params']
        lambda_smooth_grid = _log_spaced_grid_multinomial(
            center=lam_smooth_fixed,
            decades_each_side=tune_params['lambda_smooth_decades_each_side'],
        )
        l2_reg_grid = _log_spaced_grid_multinomial(
            center=lam_l2_fixed,
            decades_each_side=tune_params['l2_reg_decades_each_side'],
        )
        inner_cv_folds = tune_params['inner_cv_folds']
        inner_cv_scoring_metric = tune_params['inner_cv_scoring_metric']
        inner_cv_use_one_se_rule = tune_params['inner_cv_use_one_se_rule']
        inner_max_iter = tune_params['inner_max_iter']
        use_lax_loop = hp['use_lax_loop']

        # Only bin the feature we are about to train. The HPC dispatcher
        # invokes this method once per feature per process, so there is no
        # reason to pay the binning cost for every other feature in the
        # pickle on every invocation.
        all_blocks = self.load_univariate_data_blocks(
            pkl_path, bin_size=bin_size, feature_filter=feat_name
        )
        if feat_name not in all_blocks:
            raise KeyError(f"Feature '{feat_name}' not found in {pkl_path}.")

        feat_data = all_blocks[feat_name]
        X = feat_data['X']
        groups = feat_data['groups']
        n_time = feat_data['n_time_bins']

        mp = self.modeling_settings['model_params']
        cv_folds, fold_tolerances = get_stratified_group_splits_stable(
            groups=groups,
            y=feat_data['y'],
            split_strategy=split_strategy,
            test_prop=test_prop,
            n_splits=n_splits,
            random_seed=base_seed,
            n_categories=n_categories_total,
            max_total_attempts=mp['session_split_max_attempts'],
            widen_step=mp['session_split_widen_step'],
            widen_every=mp['session_split_widen_every']
        )

        # Project-wide canonical class ordering so per-fold proportion arrays are
        # directly comparable across folds and strategies even when a fold happens
        # to be missing a rare class.
        canonical_classes = np.unique(feat_data['y'])

        def _class_proportions(labels: np.ndarray) -> np.ndarray:
            if len(labels) == 0:
                return np.zeros(len(canonical_classes), dtype=np.float32)
            counts = np.array([(labels == c).sum() for c in canonical_classes], dtype=np.float64)
            return (counts / counts.sum()).astype(np.float32)

        strategies = ['actual', 'null', 'null_model_free']
        combined_results = {}

        # Reproducible RNG for the 'null' strategy: derived from base_seed, no
        # reliance on ambient NumPy global state.
        null_rng = np.random.default_rng(base_seed + 9_973)

        for strategy in strategies:
            print(f"\n" + "=" * 60)
            print(f"FEATURE: {feat_name} | STRATEGY: {strategy.upper()}")
            print("=" * 60)

            y = feat_data['y'].copy()
            if strategy == 'null':
                for sess_id in np.unique(groups):
                    sess_mask = (groups == sess_id)
                    sess_labels = y[sess_mask].copy()
                    null_rng.shuffle(sess_labels)
                    y[sess_mask] = sess_labels

            strategy_data = {
                'folds': {
                    # Per-fold scalar metrics (see class docstring for what
                    # each one measures). `precision` is not stored: it is
                    # fully recoverable from the confusion matrix on demand,
                    # and macro-F1 already summarizes the precision/recall
                    # trade-off.
                    'metrics': {m: [] for m in
                                ['auc', 'score', 'recall', 'f1', 'll',
                                 'brier', 'ece', 'mcc']},
                    'weights': [],
                    'intercepts': [],
                    'y_true': [],
                    'y_pred': [],
                    'y_probs': [],
                    'test_indices': [],
                    'p_train': [],
                    'p_test': [],
                    # Per-fold (K, K) confusion matrix with canonical class
                    # ordering. Cheap to store, enables reviewers to derive
                    # precision, recall, and failure-mode diagnostics
                    # downstream without re-fitting the model.
                    'confusion_matrix': [],
                    # Per-fold optimizer diagnostics exposed by the JAX
                    # estimator — flags folds that terminated at `max_iter`
                    # without converging.
                    'n_iter': [],
                    'converged': [],
                    'fit_time': [],
                    # Hyperparameter audit trail. When the tuning toggle is
                    # on, every outer fold's `(lambda_smooth, l2_reg)` is
                    # chosen by an inner CV; the winning pair, the full
                    # grid scores / SEs, the raw argmax pair, and whether
                    # the 1-SE rule softened the choice are persisted
                    # here so the published filters can always be paired
                    # with the hyperparameters that produced them.
                    # `null_model_free` writes NaN / empty placeholders so
                    # the schema is uniform across strategies.
                    'selected_lambda_smooth': [],
                    'selected_l2_reg': [],
                    'hyperparam_grid_audit': [],
                    'hyperparams_tuned': [],
                    'tolerance': list(fold_tolerances),
                    'balanced_train': bool(balance_train)
                },
                'classes': None,
                'canonical_classes': canonical_classes
            }

            for fold, (train_idx_raw, test_idx) in enumerate(cv_folds):
                # Optional sample-level balancing of the training fold. The
                # natural-rate training distribution is intentionally preserved
                # for 'null_model_free' so the empirical-prior floor remains a
                # meaningful baseline (uniform priors would collapse it).
                if balance_train and strategy != 'null_model_free':
                    fold_rng = np.random.default_rng(base_seed + fold + 1)
                    train_idx = _balance_multinomial_train_indices(train_idx_raw, y, fold_rng)
                else:
                    train_idx = train_idx_raw

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
                    fold_n_iter = 0
                    fold_converged = True
                    fold_fit_time = 0.0
                    # `null_model_free` has no model to tune.
                    fold_lambda_smooth = float('nan')
                    fold_l2_reg = float('nan')
                    fold_grid_audit = {
                        'grid_scores': {}, 'grid_ses': {},
                        'argmax_pair': None,
                        'one_se_applied': False, 'one_se_threshold': None,
                    }
                    fold_tuned_flag = False
                else:
                    # When sample-level balancing is active, the softened
                    # inverse-frequency focal-alpha would double-correct an
                    # already balanced batch, so we zero the focal-gamma and use
                    # uniform class weights inside the JAX fit.
                    if balance_train:
                        effective_focal_gamma = 0.0
                        use_uniform_weights = True
                    else:
                        effective_focal_gamma = hp['focal_loss_gamma']
                        use_uniform_weights = False

                    # Per-fold hyperparameters. When tuning is on we run an
                    # inner CV (same focal_gamma / uniform_class_weights as
                    # the outer fit) to pick (lambda_smooth, l2_reg); when
                    # off we use the settings-level fixed centres. Both the
                    # `actual` and within-session X-shuffled `null`
                    # strategies are tuned identically so the permutation
                    # test compares like-against-like hyperparameters.
                    if tune_regularization_bool:
                        fold_lambda_smooth, fold_l2_reg, fold_grid_audit = _tune_multinomial_regularization(
                            X_train=X_train,
                            y_train=y_train,
                            lambda_smooth_grid=lambda_smooth_grid,
                            l2_reg_grid=l2_reg_grid,
                            inner_cv_folds=inner_cv_folds,
                            inner_cv_scoring_metric=inner_cv_scoring_metric,
                            inner_cv_use_one_se_rule=inner_cv_use_one_se_rule,
                            n_features=1,
                            n_time_bins=n_time,
                            smoothness_derivative_order=smoothness_order,
                            focal_gamma=effective_focal_gamma,
                            uniform_class_weights=use_uniform_weights,
                            learning_rate=hp['learning_rate'],
                            max_iter=hp['max_iter'],
                            inner_max_iter=inner_max_iter,
                            tol=hp['tol'],
                            random_state=hp['random_state'] + fold,
                            verbose=hp['verbose'],
                            use_lax_loop=use_lax_loop,
                            regressor_cls=SmoothMultinomialLogisticRegression,
                        )
                        fold_tuned_flag = True
                    else:
                        fold_lambda_smooth = float(lam_smooth_fixed)
                        fold_l2_reg = float(lam_l2_fixed)
                        fold_grid_audit = {
                            'grid_scores': {}, 'grid_ses': {},
                            'argmax_pair': None,
                            'one_se_applied': False, 'one_se_threshold': None,
                        }
                        fold_tuned_flag = False

                    model = SmoothMultinomialLogisticRegression(
                        n_features=1,
                        n_time_bins=n_time,
                        lambda_smooth=fold_lambda_smooth,
                        l2_reg=fold_l2_reg,
                        smoothness_derivative_order=smoothness_order,
                        focal_gamma=effective_focal_gamma,
                        uniform_class_weights=use_uniform_weights,
                        learning_rate=hp['learning_rate'],
                        max_iter=hp['max_iter'],
                        tol=hp['tol'],
                        random_state=hp['random_state'] + fold,
                        verbose=hp['verbose'],
                        _use_lax_loop=use_lax_loop,
                    )

                    model.fit(X_train, y_train)

                    probabilities = model.predict_proba(X_test, balanced=hp['balance_predictions_bool'])
                    predictions = model.predict(X_test, balanced=hp['balance_predictions_bool'])

                    model_classes = model.classes_
                    fold_weights = model.coef_
                    fold_intercepts = model.intercept_
                    fold_n_iter = int(model.n_iter_)
                    fold_converged = bool(model.converged_)
                    fold_fit_time = float(model.fit_time_)

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

                # Align the probability matrix to canonical class ordering so
                # Brier / ECE are computed against a column ordering that does
                # not silently shift when a rare class is absent from a fold.
                probs_canonical = align_probs_to_canonical(
                    probabilities, model_classes, canonical_classes
                )

                try:
                    f_brier = brier_score_multi(y_test, probs_canonical, canonical_classes)
                except Exception:
                    f_brier = np.nan
                try:
                    f_ece = expected_calibration_error(y_test, predictions, probs_canonical, n_bins=10)
                except Exception:
                    f_ece = np.nan
                f_mcc = safe_matthews_corrcoef(y_test, predictions)

                f_met = strategy_data['folds']['metrics']
                f_met['score'].append(f_score)
                f_met['ll'].append(f_ll)
                f_met['auc'].append(f_auc)
                f_met['recall'].append(recall_score(y_test, predictions, average='macro', zero_division=0))
                f_met['f1'].append(f1_score(y_test, predictions, average='macro', zero_division=0))
                f_met['brier'].append(f_brier)
                f_met['ece'].append(f_ece)
                f_met['mcc'].append(f_mcc)

                print(f"Fold {fold + 1:02d}/{n_splits:02d} | Score: {f_score:.3f} | AUC: {f_auc:.3f} | LL: {f_ll:.3f} | Brier: {f_brier:.3f} | ECE: {f_ece:.3f} | MCC: {f_mcc:.3f}")

                # Persist deep storage matrices
                strategy_data['folds']['weights'].append(fold_weights)
                strategy_data['folds']['intercepts'].append(fold_intercepts)
                strategy_data['folds']['y_true'].append(y_test)
                strategy_data['folds']['y_pred'].append(predictions)
                strategy_data['folds']['y_probs'].append(probabilities)
                strategy_data['folds']['test_indices'].append(test_idx)
                strategy_data['folds']['p_train'].append(_class_proportions(y_train))
                strategy_data['folds']['p_test'].append(_class_proportions(y_test))
                strategy_data['folds']['confusion_matrix'].append(
                    safe_confusion_matrix(y_test, predictions, labels=canonical_classes)
                )
                strategy_data['folds']['n_iter'].append(fold_n_iter)
                strategy_data['folds']['converged'].append(fold_converged)
                strategy_data['folds']['fit_time'].append(fold_fit_time)
                strategy_data['folds']['selected_lambda_smooth'].append(fold_lambda_smooth)
                strategy_data['folds']['selected_l2_reg'].append(fold_l2_reg)
                strategy_data['folds']['hyperparam_grid_audit'].append(fold_grid_audit)
                strategy_data['folds']['hyperparams_tuned'].append(fold_tuned_flag)

                if strategy_data['classes'] is None:
                    strategy_data['classes'] = model_classes

            m = {k: np.nanmean(v) for k, v in strategy_data['folds']['metrics'].items()}
            print(f"\nFINISHED {strategy.upper()} | Avg Score: {m['score']:.3f} | Avg AUC: {m['auc']:.3f}")

            combined_results[strategy] = strategy_data

        # Cross-fold hyperparameter report. If tuning was on, the spread
        # of the selected (λ_smooth, l2_reg) across folds tells us
        # whether the data supports a well-defined regularisation level
        # (tight concentration → yes; wide scatter → no). Mirrors the
        # manifold-runner summary so users get the same diagnostic for
        # both pipelines.
        if tune_regularization_bool:
            print("-" * 60)
            print("HYPERPARAMETER SELECTION SUMMARY (log10 units)")
            for strat in ('actual', 'null'):
                if strat not in combined_results:
                    continue
                sm_vals = np.asarray(
                    combined_results[strat]['folds']['selected_lambda_smooth'], dtype=float
                )
                l2_vals = np.asarray(
                    combined_results[strat]['folds']['selected_l2_reg'], dtype=float
                )
                grid_audits = combined_results[strat]['folds']['hyperparam_grid_audit']
                one_se_fired = [bool(a.get('one_se_applied')) for a in grid_audits]
                one_se_count = int(sum(one_se_fired))
                one_se_total = len(one_se_fired)
                valid_sm = sm_vals[np.isfinite(sm_vals) & (sm_vals > 0)]
                valid_l2 = l2_vals[np.isfinite(l2_vals) & (l2_vals > 0)]
                if valid_sm.size == 0 or valid_l2.size == 0:
                    continue
                log_sm = np.log10(valid_sm)
                log_l2 = np.log10(valid_l2)
                print(
                    f"  {strat.upper():<6} | "
                    f"λ_smooth: mean(log10)={np.mean(log_sm):+.2f}, "
                    f"SD(log10)={np.std(log_sm, ddof=1):.2f} | "
                    f"l2: mean(log10)={np.mean(log_l2):+.2f}, "
                    f"SD(log10)={np.std(log_l2, ddof=1):.2f} | "
                    f"1SE-softened: {one_se_count}/{one_se_total}"
                )
            print("-" * 60 + "\n")

        return feat_name, combined_results
