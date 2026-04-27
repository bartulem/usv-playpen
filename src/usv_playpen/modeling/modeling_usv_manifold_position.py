"""
@author: bartulem
Module for continuous UMAP-position USV modelling (JAX, assumes GPU usage).

This module provides a pipeline for mapping behavioural kinematics onto the
continuous 2-D UMAP manifold of the vocal repertoire. It predicts the
deterministic `(x, y)` location of an upcoming vocalisation from a short
history window of behavioural features. Earlier revisions fitted a full
bivariate-Gaussian density with globally shared variance parameters; we
replaced that formulation with plain 2-D regression because (a) the
variance parameters were not conditional on the behavioural history, so
the "calibration" metrics (coverage at 68 %/95 %, Mahalanobis distance)
only tested whether the empirical residual distribution happened to match
a single global ellipse, and (b) UMAP coordinates are not a metric space
in any principled sense, which makes density-level claims fragile anyway.
The regression retains everything that carried scientific weight — the
learned linear map, the temporal smoothness penalty, and the inverse-
density sample weighting — and drops only the density head.

Key scientific capabilities:
1.  Continuous target extraction: extracts `(x, y)` UMAP coordinate pairs
    for every valid bout, enabling the model to learn a continuous mapping
    from behavioural history to acoustic outcomes.
2.  Geographic fairness (inverse density weighting): computes a Gaussian
    Kernel Density Estimate (KDE) over the acoustic space to apply inverse
    density sample weights. This prevents the model from ignoring rare
    acoustic "satellite islands" in favour of the dense manifold core.
3.  Spatial cross-validation: implements a deterministic K-Means spatial-
    proxy strategy for Stratified Group K-Fold splitting, strictly
    preventing session leakage while guaranteeing that rare and common
    acoustic regions are proportionally represented in every train / test
    fold.
4.  Rigorous null baselines: compares model performance against a
    within-session **X-history shuffle** (`null`) and the KDE-weighted
    training-set centroid (`null_model_free`). The X-history shuffle is
    the canonical permutation test for regression: trial `i`'s Y stays
    where it is, but its kinematic history is replaced with the history
    of another trial from the same session. This preserves session-level
    biases and within-session vocal-repertoire autocorrelation — so a
    predictor that just memorises "this session tends to vocalise at
    region R" cannot beat the null by accident — and isolates the
    kinematics-to-manifold mapping as the only signal the actual model
    can exploit.
"""

import json
import numpy as np
from pathlib import Path
import pickle
import re
from datetime import datetime
from scipy.stats import gaussian_kde, wilcoxon
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedShuffleSplit
from typing import Any, List, Tuple
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
)
from .jax_bivariate_regression import SmoothBivariateRegression
from ..analyses.compute_behavioral_features import FeatureZoo


def compute_inverse_density_weights(Y: np.ndarray,
                                    clip_percentile: float = 95.0,
                                    epsilon: float = 1e-5) -> np.ndarray:
    """
    Computes inverse density sample weights using Gaussian Kernel Density Estimation (KDE).

    This neutralizes the topographical bias of the UMAP space. Rare "satellite"
    vocalizations receive mathematically higher weights than syllables in the
    dense manifold core, ensuring the optimizer treats all geographic regions equally.

    Parameters
    ----------
    Y : np.ndarray
        Array of shape (N, 2) containing continuous UMAP coordinates.
    clip_percentile : float, default 95.0
        The percentile at which to cap maximum weights, preventing single extreme
        outliers from mathematically dominating the loss landscape.
    epsilon : float, default 1e-5
        Small constant added to the denominator to prevent division by zero.

    Returns
    -------
    weights : np.ndarray
        Array of shape (N,) containing normalized sample weights. The mean
        of the weights is exactly 1.0, preserving the global scale of the loss.
    """

    kde = gaussian_kde(Y.T)
    densities = kde(Y.T)

    raw_weights = 1.0 / (densities + epsilon)

    max_allowed_weight = np.percentile(raw_weights, clip_percentile)
    clipped_weights = np.clip(raw_weights, a_min=None, a_max=max_allowed_weight)

    normalized_weights = clipped_weights / np.mean(clipped_weights)

    return normalized_weights


def get_stratified_spatial_splits_stable(groups: np.ndarray,
                                         Y: np.ndarray,
                                         split_strategy: str = 'session',
                                         n_clusters: int = 15,
                                         test_prop: float = 0.2,
                                         n_splits: int = 100,
                                         tolerance: float = 0.05,
                                         random_seed: int = 0,
                                         max_total_attempts: int = 50000,
                                         widen_step: float = 0.02,
                                         widen_every: int = 1000) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generates deterministic folds ensuring spatial geographic fairness across the UMAP manifold.

    Uses K-Means to temporarily partition the continuous UMAP space into micro-neighborhoods
    (proxy labels). Depending on the `split_strategy`, it then splits the dataset to ensure
    the dense core and rare satellite clusters are proportionally represented in both train
    and test sets.

    Parameters
    ----------
    groups : np.ndarray
        Array of session IDs. Used strictly when split_strategy='session'.
    Y : np.ndarray
        Array of shape (N, 2) containing continuous UMAP coordinates.
    split_strategy : str, default 'session'
        Determines the data leakage constraint:
        - 'session': Strict cross-session prediction. Samples from the same session
          are never split between train and test. (Uses tolerance-based search).
        - 'mixed': Completely randomized frame-level splitting. Ignores session IDs
          and perfectly stratifies based solely on geographic density.
    n_clusters : int, default 15
        Number of geographic micro-neighborhoods to define via K-Means.
    test_prop : float, default 0.2
        Proportion of the dataset (or sessions) to assign to the test set.
    n_splits : int, default 100
        Number of independent fold iterations to generate.
    tolerance : float, default 0.05
        (For 'session' strategy only). Initial allowable difference in spatial
        distribution between the global data and the generated test splits.
    random_seed : int, default 0
        Fixed seed for absolute reproducibility.
    max_total_attempts : int, default 50000
        (For 'session' strategy only). Hard ceiling on rejection-sampling
        attempts before raising `RuntimeError`.
    widen_step : float, default 0.02
        (For 'session' strategy only). Amount by which `tolerance` is
        increased each time the sampler fails to accept a fold for
        `widen_every` consecutive attempts.
    widen_every : int, default 1000
        (For 'session' strategy only). Number of failed attempts between
        successive tolerance widenings.

    Returns
    -------
    cv_folds : list of tuples
        A list of length `n_splits`, where each tuple contains (train_indices, test_indices).
    """
    if split_strategy not in ['session', 'mixed']:
        raise ValueError(f"Invalid split_strategy: '{split_strategy}'. Must be 'session' or 'mixed'.")

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_seed, n_init='auto')
    proxy_labels = kmeans.fit_predict(Y)

    if split_strategy == 'mixed':
        sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_prop, random_state=random_seed)
        cv_folds = list(sss.split(np.zeros(len(Y)), proxy_labels))
        return cv_folds

    elif split_strategy == 'session':
        unique_sessions = np.unique(groups)
        n_test_sessions = int(len(unique_sessions) * test_prop)

        _, global_counts = np.unique(proxy_labels, return_counts=True)
        global_dist = global_counts / len(proxy_labels)

        # Precompute a session-ID -> row-indices map once so each rejection
        # iteration becomes a cheap `np.concatenate([session_rows[s] for s
        # in te_sess])` instead of a linear `np.isin` scan over every row
        # of the groups array. With `max_total_attempts=50000` this is a
        # measurable saving.
        session_to_rows = {
            sess: np.where(groups == sess)[0]
            for sess in unique_sessions
        }

        cv_folds = []
        rng = np.random.RandomState(random_seed)

        attempts = 0
        current_tolerance = tolerance

        while len(cv_folds) < n_splits:
            attempts += 1
            shuffled = rng.permutation(unique_sessions)
            te_sess = shuffled[:n_test_sessions]
            tr_sess = shuffled[n_test_sessions:]

            te_idx = np.concatenate([session_to_rows[s] for s in te_sess])
            tr_idx = np.concatenate([session_to_rows[s] for s in tr_sess])

            tr_clusters = np.unique(proxy_labels[tr_idx])
            te_clusters = np.unique(proxy_labels[te_idx])

            if len(tr_clusters) == n_clusters and len(te_clusters) == n_clusters:
                _, te_counts = np.unique(proxy_labels[te_idx], return_counts=True)
                te_dist = te_counts / len(te_idx)
                dist_error = np.max(np.abs(te_dist - global_dist))

                if dist_error < current_tolerance:
                    cv_folds.append((tr_idx, te_idx))

            if attempts % widen_every == 0:
                current_tolerance += widen_step

            if attempts > max_total_attempts:
                raise RuntimeError(
                    f"Failed to find {n_splits} valid spatial splits after {attempts} attempts. "
                    "Rare geographic clusters may be highly isolated in too few sessions."
                )

        return cv_folds


def _log_spaced_grid(center: float, decades_each_side: int) -> np.ndarray:
    """
    Returns a log-spaced grid of candidate regularisation strengths centred
    on `center`, spanning `decades_each_side` orders of magnitude on each
    side.

    Example
    -------
    `_log_spaced_grid(center=1.0, decades_each_side=3)` returns
    `[1e-3, 1e-2, 1e-1, 1e+0, 1e+1, 1e+2, 1e+3]`. `decades_each_side=0`
    returns `[center]` — the "no tuning" degenerate case.

    Parameters
    ----------
    center : float
        Centre of the grid. This is the "best guess" fixed value from the
        settings block; the grid searches a symmetric window of orders of
        magnitude around it.
    decades_each_side : int
        Half-width of the search window in decades. Must be non-negative.

    Returns
    -------
    np.ndarray
        Sorted 1-D array of length `2 * decades_each_side + 1`.
    """

    if decades_each_side < 0:
        raise ValueError(f"decades_each_side must be >= 0, got {decades_each_side}.")
    if center <= 0:
        raise ValueError(f"center must be positive, got {center}.")
    offsets = np.arange(-decades_each_side, decades_each_side + 1, dtype=np.float64)
    return float(center) * (10.0 ** offsets)


def _tune_manifold_regularization(X_train: np.ndarray,
                                  Y_train: np.ndarray,
                                  w_train: np.ndarray,
                                  groups_train: np.ndarray,
                                  *,
                                  lambda_smooth_grid: np.ndarray,
                                  l2_reg_grid: np.ndarray,
                                  inner_cv_folds: int,
                                  inner_cv_scoring_metric: str,
                                  inner_cv_use_one_se_rule: bool,
                                  n_features: int,
                                  n_time_bins: int,
                                  spatial_cluster_num: int,
                                  smoothness_derivative_order: int,
                                  huber_delta: float,
                                  learning_rate: float,
                                  max_iter: int,
                                  inner_max_iter: int,
                                  tol: float,
                                  random_state: int,
                                  verbose: bool,
                                  use_lax_loop: bool,
                                  regressor_cls) -> tuple:
    """
    Selects `(lambda_smooth, l2_reg)` jointly by inner cross-validation on
    the supplied training fold, with an optional 1-SE rule biased toward
    the smoothest regulariser that is statistically indistinguishable from
    the performance-argmax.

    Strategy
    --------
    1. Partition the training fold into `inner_cv_folds` spatially-
       stratified sub-folds using the same K-means proxy used by the outer
       splitter (`get_stratified_spatial_splits_stable`), with a reduced
       cluster count so rare clusters still populate every inner split.
    2. For every `(lambda_smooth, l2_reg)` pair on the Cartesian product of
       the two grids, fit the regressor on each inner-training sub-fold
       and score it on the inner-validation sub-fold using
       `inner_cv_scoring_metric`. Aggregate per-pair across the inner folds
       to a mean score and a standard error of the mean (SE = std / sqrt(n)
       over finite fold scores).
    3. Identify the argmax pair — the pure performance-maximising winner.
    4. If `inner_cv_use_one_se_rule=False`, return the argmax winner.
       Otherwise apply the canonical 1-SE rule biased toward filter
       interpretability: find every pair whose mean score is within
       `SE(argmax)` of the argmax (i.e. statistically indistinguishable
       from the best), and among those pick the one with the **largest
       `lambda_smooth`** (tiebreak: smallest `l2_reg`). Intuition: if two
       smoothness levels score within noise of each other there is no
       data-supported reason to prefer the wigglier filter, so we pick
       the smoother one. `l2_reg` does not shape the temporal structure
       of the filter, only its magnitude, so it is not subject to the
       same interpretability preference.

    Returned hyperparameters are **always** the selected winner (argmax
    when the flag is off, 1-SE winner when it is on). The audit payload
    includes the full grid of mean scores *and* the argmax pair so a
    reader can see how much the 1-SE rule softened the choice.

    The scoring direction is inferred from the metric name: `r2_spatial`,
    `pearson_*`, and `spearman_*` are higher-is-better; everything else is
    treated as lower-is-better (MAE / RMSE / Mahalanobis variants).

    Parameters
    ----------
    X_train, Y_train, w_train, groups_train : np.ndarray
        Training-fold design matrix, UMAP targets, inverse-density weights,
        and session IDs.
    lambda_smooth_grid, l2_reg_grid : np.ndarray
        1-D candidate grids, typically log-spaced.
    inner_cv_folds : int
        Number of inner sub-folds.
    inner_cv_scoring_metric : str
        Key returned by `SmoothBivariateRegression.evaluate_metrics` used
        to select the winning pair.
    inner_cv_use_one_se_rule : bool
        When True, apply the 1-SE interpretability rule described above.
        When False, return the performance-argmax pair.
    n_features, n_time_bins : int
        Regressor shape arguments.
    spatial_cluster_num : int
        Outer `spatial_cluster_num`. The inner splitter uses
        `max(2, spatial_cluster_num // 2)` so it still partitions the
        manifold at a reasonable granularity without requiring as many
        points per cluster as the outer run.
    huber_delta, learning_rate, tol, random_state, verbose :
        Passed through to `regressor_cls` on every inner fit.
    max_iter : int
        Iteration cap for the *outer* fit (unused here — kept in the
        signature so callers can share a single settings block across
        both the outer and inner paths).
    inner_max_iter : int
        Iteration cap used for every inner-CV fit. Typically smaller
        than `max_iter` (e.g., 2500 vs. 10000) — the inner CV only
        needs a usable score to rank regularisation pairs, not the
        final fully-converged weights. Reduces tuning wall time by
        `max_iter / inner_max_iter` per pair per fold.
    regressor_cls : callable
        The estimator class (injected so the function is unit-testable
        without importing the JAX estimator at module scope).

    Returns
    -------
    best_lambda_smooth : float
        Winning smoothness penalty strength.
    best_l2_reg : float
        Winning L2 penalty strength.
    grid_audit : dict
        Audit payload with keys:

        - `'grid_scores'` : `{(lambda_smooth, l2_reg): mean_inner_score}`
          for every pair on the grid. Useful for persistence / plotting.
        - `'grid_ses'`    : `{(lambda_smooth, l2_reg): se_inner_score}`.
        - `'argmax_pair'` : tuple `(lambda_smooth, l2_reg)` the pair that
          maximises (or minimises) the mean inner score without the 1-SE
          adjustment.
        - `'one_se_applied'` : bool. True when the returned pair was
          selected by the 1-SE rule rather than the raw argmax.
        - `'one_se_threshold'` : float or None. The score threshold used
          when the rule fired (argmax_score - SE(argmax) for
          higher-is-better metrics; argmax_score + SE(argmax) for
          lower-is-better).
    """

    higher_is_better_metrics = {
        'r2_spatial',
        'pearson_x', 'pearson_y',
        'spearman_x', 'spearman_y',
    }
    higher_is_better = inner_cv_scoring_metric in higher_is_better_metrics

    def _empty_audit():
        return {
            'grid_scores': {},
            'grid_ses': {},
            'argmax_pair': None,
            'one_se_applied': False,
            'one_se_threshold': None,
        }

    # Degenerate training folds (single session, single class) cannot be
    # inner-split; fall back to the grid centre (= the user's fixed value)
    # and a single-pair "grid" so the caller still receives a well-formed
    # diagnostic payload.
    if len(np.unique(groups_train)) < 2 or len(Y_train) < inner_cv_folds * 2:
        return (
            float(lambda_smooth_grid[len(lambda_smooth_grid) // 2]),
            float(l2_reg_grid[len(l2_reg_grid) // 2]),
            _empty_audit(),
        )

    inner_cluster_num = max(2, spatial_cluster_num // 2)
    inner_folds = get_stratified_spatial_splits_stable(
        groups=groups_train,
        Y=Y_train,
        n_clusters=inner_cluster_num,
        test_prop=1.0 / inner_cv_folds,
        n_splits=inner_cv_folds,
        # Inner CV lives inside the outer training fold, where session-mixing
        # is already acceptable; use 'mixed' so inner splits are quick and
        # cluster-balanced without a second layer of session-holdout logic.
        split_strategy='mixed',
        random_seed=random_state + 7919,
    )

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
                        huber_delta=huber_delta,
                        learning_rate=learning_rate,
                        max_iter=inner_max_iter,
                        tol=tol,
                        random_state=random_state + inner_idx,
                        verbose=False,
                        _use_lax_loop=use_lax_loop,
                    )
                    model.fit(X_train[in_tr], Y_train[in_tr], sample_weight=w_train[in_tr])
                    metrics = model.evaluate_metrics(
                        X_train[in_va], Y_train[in_va], weights=w_train[in_va]
                    )
                    fold_scores.append(metrics[inner_cv_scoring_metric])
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
                # Preserve a NaN entry so the grid shape is recoverable; it
                # just can't win the selection.
                grid_scores[pair_key] = float('nan')
                grid_ses[pair_key] = float('nan')

    # Argmax winner (pure performance).
    valid_pairs = [(pair, score) for pair, score in grid_scores.items() if np.isfinite(score)]
    if not valid_pairs:
        # Every pair failed — fall back to the grid centres so the outer
        # fit still runs with a reasonable default instead of crashing.
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

    # 1-SE rule: "smoothest pair whose mean score is within one SE of the
    # argmax". Biases the tuner toward interpretable filters when two
    # regularisation levels are statistically indistinguishable.
    #
    # `threshold` is the worst score a pair can have and still be
    # considered "within noise" of the argmax. For higher-is-better
    # metrics this is `argmax_score - SE(argmax)`; for lower-is-better
    # it flips to `argmax_score + SE(argmax)`.
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
        # Pathological: the argmax pair itself should always be in-band,
        # so this path is unreachable under normal numerics. Defensive
        # fallback keeps the function total.
        return float(argmax_pair[0]), float(argmax_pair[1]), audit

    # Tiebreak: largest lambda_smooth (smoothness preference), then
    # smallest l2_reg (minimise variance-shrinkage when smoothness ties).
    winner = max(in_band, key=lambda p: (p[0], -p[1]))
    audit['one_se_applied'] = True
    audit['one_se_threshold'] = float(threshold)
    return float(winner[0]), float(winner[1]), audit


class ContinuousModelingPipeline(FeatureZoo):

    def __init__(self, modeling_settings_dict: dict = None, **kwargs):
        """
        Initializes the ContinuousModelingPipeline class.

        This constructor establishes the configuration environment required for extracting
        continuous probabilistic modeling data. It strictly loads the modeling settings
        (either via a provided dictionary or the default JSON file), validates the presence
        of required nested parameters, and computes the fixed-length kinematic history
        window (in frames) based on the exact camera sampling rate and physical time constraints.
        It also inherits from `FeatureZoo` to ensure access to standardized behavioral
        feature computations.

        Parameters
        ----------
        modeling_settings_dict : dict, optional
            A nested dictionary containing the complete modeling configuration (e.g., IO paths,
            hyperparameters, continuous feature bounds). If None, defaults to loading from
            `_parameter_settings/modeling_settings.json`.
        **kwargs : dict
            Additional keyword arguments to dynamically set as instance attributes.
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
            print(f"Continuous Pipeline Init: History frames calculated: {self.history_frames} (for {hist_sec}s at {cam_rate}fps)")
        except KeyError as e:
            raise KeyError(f"Critical setting missing for history calculation: {e}")

        FeatureZoo.__init__(self)

        for kw_arg, kw_val in kwargs.items():
            self.__dict__[kw_arg] = kw_val

    def extract_and_save_continuous_data(self) -> None:
        """
        Extracts, processes, and saves (X, Y, w) data matrices for continuous probabilistic regression.

        This pipeline orchestrates the end-to-end data engineering required to map behavioral
        kinematics to a continuous acoustic manifold. Unlike categorical classification, this
        method preserves the true topological reality of the vocal repertoire.

        Process Outline:
        1. Target extraction: Identifies all valid USVs across specified sessions, verifies they
           meet the historical time constraints, and extracts their continuous 2D UMAP coordinates (Y).
        2. Geographic fairness (KDE weights): Computes a global Gaussian Kernel Density Estimate
           across the universal Y manifold to generate normalized inverse-density sample weights (w).
           This mathematically neutralizes the topographical bias of the dense acoustic core.
        3. Feature standardization: Filters the behavioral dataset for specific self/partner
           predictors, injects continuous vocal history, and standardizes (z-scores) all features
           globally across the pooled sessions.
        4. Epoch slicing: Slices the z-scored behavioral traces into fixed-length arrays (X)
           representing the precise temporal kinematics preceding every USV event.

        Returns
        -------
        None
            Saves a highly structured `.pkl` file to the configured `save_dir`. The output is a
            nested dictionary explicitly organized for the downstream JAX engine:
            `data[feature_name][session_id] = {'X': array, 'Y': array, 'w': array}`

            - 'X': Predictor history matrix of shape (n_samples, history_frames).
            - 'Y': Target spatial matrix of shape (n_samples, 2) containing (umap_x, umap_y).
            - 'w': Inverse-density sample weights of shape (n_samples,).
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
        manifold_cols = voc_settings['usv_manifold_column_names']

        if not isinstance(manifold_cols, (list, tuple)) or len(manifold_cols) < 2:
            raise ValueError(
                "'vocal_features.usv_manifold_column_names' must be a list with at least "
                f"two entries; got {manifold_cols!r}."
            )
        if len(manifold_cols) != 2:
            raise ValueError(
                "The continuous manifold pipeline currently assumes a 2-D target (bivariate "
                f"Gaussian / CNN regression). Received {len(manifold_cols)} columns: {manifold_cols}."
            )

        filter_hist = self.modeling_settings['model_params']['filter_history']
        pred_idx = self.modeling_settings['model_params']['model_predictor_mouse_index']
        targ_idx = abs(pred_idx - 1)

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
            noise_vocal_categories=noise_cats,
            manifold_column_names=manifold_cols
        )

        print("Extracting continuous targets and computing Global Inverse Density Weights...")
        continuous_targets_dict = {}
        all_valid_Y_list = []

        for sess_id in list(beh_data_dict.keys()):
            if sess_id not in mouse_names_dict:
                continue

            targ_name = mouse_names_dict[sess_id][targ_idx]

            if sess_id not in usv_data_dict or targ_name not in usv_data_dict[sess_id]:
                continue

            if 'continuous_onsets' not in usv_data_dict[sess_id][targ_name] or 'continuous_targets' not in usv_data_dict[sess_id][targ_name]:
                continue

            onsets = usv_data_dict[sess_id][targ_name]['continuous_onsets']
            targets = usv_data_dict[sess_id][targ_name]['continuous_targets']

            if onsets is None or targets is None:
                continue

            fps = cam_fps_dict[sess_id]
            max_frame_idx = beh_data_dict[sess_id].height - 1

            valid_onsets = []
            valid_targets = []

            frame_indices = np.round(onsets * fps).astype(int)
            for i, f_idx in enumerate(frame_indices):
                if self.history_frames <= f_idx <= max_frame_idx:
                    valid_onsets.append(f_idx)
                    valid_targets.append(targets[i])

            if valid_onsets:
                valid_onsets_arr = np.array(valid_onsets, dtype=int)
                valid_targets_arr = np.array(valid_targets, dtype=np.float32)

                continuous_targets_dict[sess_id] = {
                    'onsets': valid_onsets_arr,
                    'targets': valid_targets_arr
                }
                all_valid_Y_list.append(valid_targets_arr)

        if not all_valid_Y_list:
            raise ValueError("No valid continuous targets extracted. Check UMAP data.")

        global_Y_matrix = np.vstack(all_valid_Y_list)
        global_weights = compute_inverse_density_weights(global_Y_matrix)

        ptr = 0
        for sess_id in continuous_targets_dict.keys():
            n_sess_usvs = len(continuous_targets_dict[sess_id]['targets'])
            continuous_targets_dict[sess_id]['weights'] = global_weights[ptr: ptr + n_sess_usvs]
            ptr += n_sess_usvs

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

        print("Standardizing columns and z-scoring...")
        processed_beh_data, revised_predictors = harmonize_session_columns(
            processed_beh_dict=processed_beh_data,
            mouse_names_dict=mouse_names_dict,
            target_idx=targ_idx,
            predictor_idx=pred_idx
        )

        feature_bounds = self.feature_boundaries if hasattr(self, 'feature_boundaries') else {}

        processed_beh_data = zscore_features_across_sessions(
            processed_beh_dict=processed_beh_data,
            suffixes=revised_predictors,
            feature_bounds=feature_bounds,
            abs_features=['allo_roll', 'allo_yaw-nose', 'nose-allo_yaw', 'allo_yaw-TTI', 'TTI-allo_yaw']
        )

        cohort_condition = derive_experimental_condition(self.modeling_settings)
        analysis_tag = "manifold"
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        fname = f"modeling_{analysis_tag}_{cohort_condition}_{ts}.pkl"

        # Predictor diagnostics audit. The continuous pipeline stores
        # onsets as frame indices in `continuous_targets_dict[sess]`;
        # convert to seconds-per-session for the audit wrapper.
        # Diagnostic-only — failures warn and continue.
        precomputed_events = {}
        for sess_id, data_packet in continuous_targets_dict.items():
            fps_local = cam_fps_dict[sess_id]
            precomputed_events[sess_id] = (data_packet['onsets'].astype(np.float64) / fps_local)

        gmm_idx_md = self.modeling_settings['model_params']['gmm_component_index']
        ibi_thresholds_md = {}
        gmm_params_md = self.modeling_settings['gmm_params']
        for sex in ('male', 'female'):
            params = gmm_params_md[sex]
            if gmm_idx_md < len(params['means']):
                ibi_thresholds_md[sex] = float(_calculate_ibi_threshold(
                    params['means'][gmm_idx_md], params['sds'][gmm_idx_md],
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

        n_events_per_session_md = {
            sid: {'usv': int(packet['onsets'].size)}
            for sid, packet in continuous_targets_dict.items()
        }

        input_metadata = build_input_metadata(
            modeling_settings=self.modeling_settings,
            analysis_type='continuous',
            analysis_tag=analysis_tag,
            pipeline_class=type(self).__name__,
            target_idx=targ_idx,
            predictor_idx=pred_idx,
            n_sessions_used=len(processed_beh_data),
            session_ids=sorted(processed_beh_data.keys()),
            n_events_per_session=n_events_per_session_md,
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
                'usv_manifold_column_names': list(manifold_cols),
            },
        )

        run_predictor_audits(
            processed_beh_dict=processed_beh_data,
            usv_data_dict=usv_data_dict,
            mouse_names_dict=mouse_names_dict,
            camera_fps_dict=cam_fps_dict,
            target_idx=targ_idx,
            predictor_idx=pred_idx,
            history_frames=self.history_frames,
            event_keys=[],
            settings=self.modeling_settings,
            save_dir=self.modeling_settings['io']['save_directory'],
            pickle_basename=fname,
            precomputed_event_times=precomputed_events,
            input_metadata=input_metadata,
        )

        print("Extracting epochs and saving to disk...")
        final_data = {}

        for sess_id, data_packet in tqdm(continuous_targets_dict.items(), desc='Slicing Epochs'):
            if sess_id not in processed_beh_data:
                continue

            sess_df = processed_beh_data[sess_id]
            t_name = mouse_names_dict[sess_id][targ_idx]
            p_name = mouse_names_dict[sess_id][pred_idx]

            onsets = data_packet['onsets']
            Y_targets = data_packet['targets']
            weights = data_packet['weights']

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
                    'Y': Y_targets,
                    'w': weights
                }

        if not final_data:
            print("No valid data extracted. Aborting save.")
            return

        feature_names = sorted(list(final_data.keys()))
        n_feats = len(feature_names)
        sample_feat = feature_names[0] if n_feats > 0 else None

        if sample_feat:
            n_sessions = len(final_data[sample_feat])
            all_Y = np.vstack([final_data[sample_feat][s]['Y'] for s in final_data[sample_feat]])
            all_w = np.concatenate([final_data[sample_feat][s]['w'] for s in final_data[sample_feat]])
            total_samples = len(all_Y)

            print("\n" + "=" * 60)
            print(f"CONTINUOUS PROBABILISTIC EXTRACTION SUMMARY")
            print("=" * 60)
            print(f"Total Unique Features:  {n_feats}")
            print(f"Total Valid Sessions:   {n_sessions}")
            print(f"Total USVs included (N): {total_samples}")
            print(f"Global Weights Mean:    {np.mean(all_w):.4f}")
            print(f"Global Weights Max:     {np.max(all_w):.4f}")
            print("-" * 60)

            print("EXTRACTED FEATURES:")
            for i in range(0, len(feature_names), 4):
                print("  " + "".join(f"{name: <25}" for name in feature_names[i:i + 4]))
            print("=" * 60 + "\n")

        save_dir = Path(self.modeling_settings['io']['save_directory'])
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / fname

        input_metadata['feature_zoo_kept'] = sorted(final_data.keys())
        artifact = inject_metadata(final_data, _input_metadata=input_metadata)

        print(f"Saving continuous extraction results to:\n{save_path}")
        with save_path.open('wb') as f:
            pickle.dump(artifact, f)
        print("[+] Save Complete.")


class ContinuousModelRunner:
    """
    Orchestrates the training and statistical evaluation of continuous UMAP-
    position USV regression models.

    This class serves as the execution engine for the continuous modelling
    phase. It consumes the extracted `(X, Y, w)` data produced by the
    `ContinuousModelingPipeline`, transforms it into binned univariate
    design matrices, and performs cross-validation using the JAX-accelerated
    `SmoothBivariateRegression` estimator.

    Key responsibilities:
    1. Data transformation: pivots nested session dictionaries into unified,
       temporally downsampled (binned) arrays for efficient gradient
       descent.
    2. Experimental control: evaluates features against an `actual`
       framework, a within-session shuffled `null` framework, and a
       `null_model_free` spatial-centroid baseline (weighted training-set
       mean, matching the marginal prior used by the KDE weighting).
    3. Spatial validation: uses deterministic K-Means geographic clustering
       to build spatially fair cross-validation folds.
    4. Deep metadata storage: persists fold-level predictions `(y_pred_xy)`,
       learned weights and intercepts, and per-fold optimizer diagnostics.

    Metrics saved per fold (see `SmoothBivariateRegression.evaluate_metrics`
    for full definitions). Every fit uses the manifold-snapped predictions
    so the active model and both baselines are evaluated on the same
    support:
    - `r2_spatial` — pooled spatial variance explained by the predictions;
      bounded above by 1. **Selection score** (higher is better).
    - `euclidean_mae` — mean Euclidean distance between snapped predictions
      and truth, in native UMAP units. Interpretable headline error.
    - `euclidean_rmse` — root-mean-squared Euclidean distance; a large
      `RMSE / MAE` ratio flags heavy-tailed outlier folds.
    - `euclidean_mae_weighted` — MAE on the Euclidean residual weighted by
      the inverse-density KDE weights so satellite vocalisations count as
      much as dense-core bouts.
    - `euclidean_mae_raw` — diagnostic only; MAE on the *unsnapped* raw
      linear predictions. The gap `euclidean_mae_raw - euclidean_mae`
      quantifies how often the unconstrained model extrapolates
      off-manifold.
    - `mahalanobis_mae` — mean standardized residual distance using the
      inverse-density-weighted training covariance of `Y_train`. Removes
      the UMAP axis-scale arbitrariness; dimensionless, lower is better.
    - `mae_x`, `mae_y` — per-axis absolute error on snapped predictions.
    - `pearson_x`, `pearson_y`, `spearman_x`, `spearman_y` — per-axis
      linear and rank correlations between predictions and truth.
    - `n_iter`, `converged`, `fit_time` — per-fold JAX optimizer
      diagnostics. `converged=False` flags folds that terminated at
      `max_iter` without meeting the tolerance.
    - `selected_lambda_smooth`, `selected_l2_reg` — the regularisation
      strengths actually used for each outer fold's fit. Equal to the
      fixed settings centres when `tune_regularization_bool=False`, or
      the winners of per-fold joint inner CV when `True`. Always
      persisted so the storage schema is fixed and the published filter
      shapes can be paired with the hyperparameters that produced them.
    - `hyperparam_grid_audit` — per-fold dict with keys `grid_scores`
      (`{(λ_smooth, l2_reg) -> mean inner score}`), `grid_ses` (the
      matching per-pair standard errors across inner folds), the
      performance `argmax_pair`, and flags `one_se_applied` /
      `one_se_threshold` recording whether the 1-SE interpretability
      rule softened the final choice and what the resulting score
      threshold was. Empty / `False` when tuning is off. Diagnostic
      only.
    - `hyperparams_tuned` — bool flag per fold recording whether the
      hyperparameters were tuned or fixed.
    """

    def __init__(self, pipeline_instance: Any) -> None:
        """
        Initializes the model runner using a configured pipeline instance.

        Parameters
        ----------
        pipeline_instance : ContinuousModelingPipeline
            An instance of the extraction class which holds the
            'modeling_settings' dictionary and calculated attributes.
        """
        self.modeling_settings = pipeline_instance.modeling_settings

        if hasattr(pipeline_instance, 'feature_boundaries'):
            self.feature_boundaries = pipeline_instance.feature_boundaries

    @staticmethod
    def load_univariate_data_blocks(pkl_path: str,
                                    bin_size: int = 1,
                                    feature_filter=None) -> dict:
        """
        Loads extracted feature data from disk and applies temporal downsampling (binning).

        This method aggregates session-specific behavioral windows into a unified
        design matrix for each feature. Crucially, it applies temporal binning to
        reduce the dimensionality of the predictor space, acting as a low-pass filter
        on the behavioral time-series.

        By averaging frames into bins (e.g., bin_size=10), the model focuses on slower,
        behaviorally meaningful kinematic envelopes rather than high-frequency jitter.
        This drastically reduces the number of parameters the JAX solver must optimize.

        Parameters
        ----------
        pkl_path : str
            The full path to the .pkl file containing the extracted (X, Y, w)
            dictionaries from the ContinuousModelingPipeline.
        bin_size : int, optional
            The resizing factor for temporal downsampling. A value of 10
            means every 10 frames are averaged into 1 bin. Default is 1,
            matching `hyperparameters.jax_linear.bivariate.bin_resizing_factor`
            in the settings file — any downstream caller should normally
            pass the settings value through rather than rely on this
            default.
        feature_filter : iterable of str or str, optional
            If provided, only bin and return the listed features. The HPC
            dispatcher runs one feature per process, so paying the binning
            cost for every feature in the pickle on every call is wasteful;
            pass the feature(s) actually needed to skip the rest. The
            default (`None`) retains the behaviour of binning every
            feature in the pickle.

        Returns
        -------
        data_blocks : dict
            A dictionary keyed by feature name (e.g., 'self.speed'). Each value is
            another dictionary containing:
            - 'X' : np.ndarray (n_samples, n_binned_time)
                The flattened and binned behavioral history matrix.
            - 'Y' : np.ndarray (n_samples, 2)
                The continuous (x, y) UMAP targets.
            - 'w' : np.ndarray (n_samples,)
                The inverse-density sample weights.
            - 'groups' : np.ndarray (n_samples,)
                String IDs used for session-aware splitting and null shuffling.
            - 'n_time_bins' : int
                The final count of temporal predictors after binning.
        """

        print(f"Loading and binning continuous data (bin_size={bin_size}) from: {pkl_path}")

        with open(pkl_path, 'rb') as f:
            raw_data = pickle.load(f)

        # Strip metadata blocks before iterating features. Without this
        # filter, the underscore-prefixed reserved keys
        # (`_input_metadata` etc.) would be treated as feature names and
        # the per-session array indexing below would raise.
        from .modeling_metadata import RESERVED_METADATA_KEYS as _RES_KEYS
        feature_keys = [k for k in raw_data.keys() if k not in _RES_KEYS]

        if feature_filter is None:
            sorted_features = sorted(feature_keys)
        else:
            wanted = {feature_filter} if isinstance(feature_filter, str) else set(feature_filter)
            sorted_features = sorted(feat for feat in feature_keys if feat in wanted)

        data_blocks = {}

        for feat in sorted_features:
            X_list, Y_list, w_list, groups_list = [], [], [], []
            sessions = sorted(list(raw_data[feat].keys()))

            for sess_id in sessions:
                X_sess = raw_data[feat][sess_id]['X']
                Y_sess = raw_data[feat][sess_id]['Y']
                w_sess = raw_data[feat][sess_id]['w']

                N, T = X_sess.shape
                if bin_size > 1:
                    new_T = T // bin_size
                    X_sess = X_sess[:, :new_T * bin_size].reshape(N, new_T, bin_size).mean(axis=2)

                X_list.append(X_sess)
                Y_list.append(Y_sess)
                w_list.append(w_sess)
                groups_list.append(np.full(len(Y_sess), sess_id))

            data_blocks[feat] = {
                'X': np.vstack(X_list).astype(np.float32),
                'Y': np.vstack(Y_list).astype(np.float32),
                'w': np.concatenate(w_list).astype(np.float32),
                'groups': np.concatenate(groups_list),
                'n_time_bins': X_list[0].shape[1]
            }

        return data_blocks

    def run_univariate_training(self, pkl_path: str, feat_name: str) -> dict:
        """
        Executes the cross-validation and statistical evaluation loop for a
        single feature.

        This method applies `SmoothBivariateRegression` to the temporal
        kinematics `X` to predict the UMAP position `Y`. Performance is
        evaluated across three strategies:

        1. `actual` — fits the true kinematic-to-acoustic mapping.
        2. `null` — within-session X-history shuffle. For the training
           fold, every trial's kinematic history is replaced with the
           history of another trial from the same session (seeded per
           fold via `np.random.default_rng`); `Y_train` and `w_train`
           stay in place. This is the classical permutation test for
           regression and, crucially, preserves session-level vocal-
           repertoire autocorrelation — so a predictor that exploits
           "this session tends to vocalise around region R" cannot beat
           the null accidentally. The test fold is left untouched so the
           null estimator is evaluated on the same real `(X_test,
           Y_test)` pairing as the actual model.
        3. `null_model_free` — bypasses modelling entirely and predicts the
           KDE-weighted training centroid for every test trial. This is
           the "no-kinematics, no-session-structure" floor that any model
           with genuine signal must beat.

        Selection score
        ---------------
        `r2_spatial` (pooled-axis coefficient of determination against the
        test-fold marginal mean) is the headline score — directly
        comparable across features and sex groups. All other metrics are
        reported as diagnostics (see the class docstring).

        Hyperparameter tuning
        ---------------------
        Reads `hyperparameters.jax_linear.bivariate.tune_regularization_bool`:

        - `false` (default): every outer fold uses the fixed settings-
          level `lambda_smooth_fixed` and `l2_reg_fixed` values.
        - `true`: each outer fold runs a joint inner CV over the log-
          spaced `(lambda_smooth, l2_reg)` grids (centred on the fixed
          values, half-width controlled by
          `tune_regularization_params.{lambda_smooth, l2_reg}_decades_each_side`)
          and picks the pair that maximises `r2_spatial` (or whatever
          `inner_cv_scoring_metric` is set to) on held-out inner folds.
          The `null` strategy is tuned the same way so the permutation
          test compares like-against-like hyperparameters rather than
          penalising the null by forcing it to use the actual model's
          settings.

          When `inner_cv_use_one_se_rule=True` (default) the tuner then
          applies the canonical 1-SE rule biased toward filter
          interpretability: the returned pair is the smoothest one whose
          mean inner score is within one SE of the performance argmax,
          with `l2_reg` broken as a secondary tiebreak. This prevents
          the tuner from chasing wiggly filters whose R² gains are
          statistically indistinguishable from noise. Set the flag to
          `False` to recover the raw performance-argmax behaviour.

        The winning pair per outer fold is persisted alongside the filter
        weights (see `selected_lambda_smooth`, `selected_l2_reg`,
        `hyperparam_grid_audit`, `hyperparams_tuned` in the class
        docstring). The `hyperparam_grid_audit` entry also records the
        raw argmax pair and whether the 1-SE rule fired, so downstream
        code can see how much the rule softened the choice.

        Data persistence
        -----------------
        Saves full-resolution tracking data (`test_indices`, `y_true`,
        `y_pred_xy` — manifold-snapped, `weights`, `intercepts`,
        convergence diagnostics, `w_test`) for every strategy so
        downstream comparative scatter plotting and manifold
        visualisation can be regenerated without re-training.

        Parameters
        ----------
        pkl_path : str
            Path to the source data pickle produced by
            `ContinuousModelingPipeline.extract_and_save_continuous_data`.
            Only the single feature named by `feat_name` is loaded and
            binned — matches the multinomial runner's per-feature HPC
            dispatch convention.
        feat_name : str
            The specific behavioural feature (e.g., `'self.speed'`) to
            evaluate.

        Returns
        -------
        results : dict
            Nested dictionary keyed by strategy. Each strategy exposes a
            `folds` dict whose `metrics` sub-dict mirrors the output of
            `SmoothBivariateRegression.evaluate_metrics` plus the per-fold
            optimiser diagnostics documented in the class docstring.
        """

        print(f"--- Starting Univariate Training: {feat_name} ---")

        hp = self.modeling_settings['hyperparameters']['jax_linear']['bivariate']
        bin_size = hp['bin_resizing_factor']

        # Only bin the feature we're about to train. Mirrors the multinomial
        # runner's per-feature dispatch so HPC one-feature-per-process
        # invocations don't pay the binning cost for every other feature
        # in the pickle.
        data_blocks = self.load_univariate_data_blocks(
            pkl_path, bin_size=bin_size, feature_filter=feat_name,
        )
        if feat_name not in data_blocks:
            raise KeyError(f"Feature '{feat_name}' not found in {pkl_path}.")

        feat_data = data_blocks[feat_name]
        X = feat_data['X']
        Y = feat_data['Y']
        w = feat_data['w']
        groups = feat_data['groups']
        n_time_bins = feat_data['n_time_bins']

        lam_smooth_fixed = hp['lambda_smooth_fixed']
        lam_l2_fixed = hp['l2_reg_fixed']
        smoothness_order = hp['smoothness_derivative_order']
        huber_delta = hp['huber_delta']
        lr = hp['learning_rate']
        max_iter = hp['max_iter']
        tol = hp['tol']
        random_seed = hp['random_state']
        verbose = hp['verbose']
        use_lax_loop = hp['use_lax_loop']

        tune_regularization_bool = hp['tune_regularization_bool']
        tune_params = hp['tune_regularization_params']
        # Grids are reconstructed once up front from the settings-level
        # centre + half-width so the user interacts with a compact numeric
        # spec rather than explicit lists.
        lambda_smooth_grid = _log_spaced_grid(
            center=lam_smooth_fixed,
            decades_each_side=tune_params['lambda_smooth_decades_each_side'],
        )
        l2_reg_grid = _log_spaced_grid(
            center=lam_l2_fixed,
            decades_each_side=tune_params['l2_reg_decades_each_side'],
        )
        inner_cv_folds = tune_params['inner_cv_folds']
        inner_cv_scoring_metric = tune_params['inner_cv_scoring_metric']
        inner_cv_use_one_se_rule = tune_params['inner_cv_use_one_se_rule']
        inner_max_iter = tune_params['inner_max_iter']

        cv_settings = self.modeling_settings['model_params']
        n_clusters = cv_settings['spatial_cluster_num']
        test_prop = cv_settings['test_proportion']
        n_splits = cv_settings['split_num']
        split_strategy = cv_settings['split_strategy']

        print("Generating deterministic, spatially-stratified folds...")
        folds = get_stratified_spatial_splits_stable(
            groups=groups,
            Y=Y,
            n_clusters=n_clusters,
            test_prop=test_prop,
            n_splits=n_splits,
            split_strategy=split_strategy,
            random_seed=random_seed,
            max_total_attempts=cv_settings['session_split_max_attempts'],
            widen_step=cv_settings['session_split_widen_step'],
            widen_every=cv_settings['session_split_widen_every'],
        )

        # Canonical set of metric keys emitted by `evaluate_metrics`. Used
        # both to initialise the per-fold metric dict and to summarise at
        # the end so the two stay in lockstep. `r2_spatial` is first
        # because it is the selection score.
        metric_keys = [
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

        results = {}
        strategies = ['actual', 'null', 'null_model_free']

        def _shuffle_X_within_groups(X_block: np.ndarray,
                                     groups_block: np.ndarray,
                                     rng_: np.random.Generator) -> np.ndarray:
            """
            Returns `X_block` with its rows permuted inside each session
            block, preserving cross-session structure.

            This implements the "within-session X-history shuffle" that
            underpins the `null` permutation test: every trial's kinematic
            history is re-paired with another trial's history from the
            same session, while `Y_train` and `w_train` stay aligned to
            the original row order at the call site. The permutation is
            drawn independently per session so trials never cross session
            boundaries (session-level vocal-repertoire autocorrelation is
            preserved under the null) and single-trial sessions are left
            untouched (a 1-element permutation is a no-op).

            Parameters
            ----------
            X_block : np.ndarray
                Flattened behavioural-history design matrix of shape
                `(n_samples, n_features * n_time_bins)`. Row `i`
                corresponds to the kinematic history preceding trial `i`.
            groups_block : np.ndarray
                Session-ID label for each row of `X_block`, shape
                `(n_samples,)`. Determines which rows may be shuffled
                together.
            rng_ : np.random.Generator
                Seeded NumPy generator used to draw the per-session
                permutations. Passed in from the caller so the null
                strategy is deterministic per fold.

            Returns
            -------
            X_shuffled : np.ndarray
                Copy of `X_block` with rows permuted inside each session
                block. Same shape and dtype as the input.
            """
            perm = np.arange(len(X_block))
            for sess_id in np.unique(groups_block):
                sess_positions = np.where(groups_block == sess_id)[0]
                shuffled = sess_positions.copy()
                rng_.shuffle(shuffled)
                perm[sess_positions] = shuffled
            return X_block[perm]

        for strategy in strategies:
            print(f"  Executing Strategy: [{strategy.upper()}]")

            results[strategy] = {
                'folds': {
                    'metrics': {m: [] for m in metric_keys},
                    # Learned linear map and bias (None for `null_model_free`,
                    # which has no trainable parameters).
                    'weights': [],
                    'intercepts': [],
                    'test_indices': [],
                    'y_true': [],
                    'w_test': [],
                    # Deterministic (x, y) predictions for every test trial —
                    # shape `(n_test, 2)`. Predictions are manifold-snapped
                    # to the nearest training UMAP point for the active and
                    # `null` strategies; `null_model_free` predicts the
                    # training centroid (already inside the manifold hull).
                    'y_pred_xy': [],
                    # Per-fold optimiser diagnostics. `converged=False`
                    # flags folds that terminated at `max_iter` without
                    # meeting the tolerance — the main silent-failure
                    # mode of the JAX estimator.
                    'n_iter': [],
                    'converged': [],
                    'fit_time': [],
                    # Hyperparameters actually used for the outer fit (either
                    # the user-supplied fixed values or the winners of the
                    # per-fold inner-CV tuning). Always persisted so the
                    # storage schema is the same whether tuning was on or
                    # off. `null_model_free` records NaN since there is no
                    # model to tune.
                    'selected_lambda_smooth': [],
                    'selected_l2_reg': [],
                    # Full inner-CV audit payload per fold. Dict with
                    # keys `grid_scores`, `grid_ses`, `argmax_pair`,
                    # `one_se_applied`, `one_se_threshold` — see
                    # `_tune_manifold_regularization`. Empty audit when
                    # tuning is disabled or when the inner splitter
                    # degenerated. Kept so the reader can see how peaked
                    # or flat the tuning landscape was, how much the
                    # 1-SE rule softened the choice, and what the raw
                    # performance-argmax would have been.
                    'hyperparam_grid_audit': [],
                    # Booleans declaring whether this fold's hyperparameters
                    # were tuned or fixed. Handy for downstream plots that
                    # want to overlay the tuned / fixed runs on the same
                    # axes.
                    'hyperparams_tuned': [],
                }
            }

            for fold_idx, (train_idx, test_idx) in enumerate(folds):

                X_train, X_test = X[train_idx], X[test_idx]
                Y_train, Y_test = Y[train_idx], Y[test_idx]
                w_train, w_test = w[train_idx], w[test_idx]
                groups_train = groups[train_idx]

                if strategy == 'null':
                    # Within-session X-history shuffle: each training
                    # trial's kinematic history is swapped with another
                    # trial's history from the same session. Session-level
                    # vocal-repertoire structure is preserved, so the null
                    # is strictly about "does the specific kinematic
                    # signal matter?"
                    shuffle_rng = np.random.default_rng(random_seed + fold_idx + 1)
                    X_train = _shuffle_X_within_groups(X_train, groups_train, shuffle_rng)

                if strategy == 'null_model_free':
                    # Spatial-centroid baseline: predict the KDE-weighted
                    # training centroid for every test trial — the
                    # absolute floor before any kinematic signal enters.
                    mu = np.average(Y_train, axis=0, weights=w_train)
                    y_pred_xy = np.tile(mu.astype(np.float32), (len(Y_test), 1))

                    dx = Y_test[:, 0] - mu[0]
                    dy = Y_test[:, 1] - mu[1]
                    euclidean_dist = np.sqrt(dx ** 2 + dy ** 2)
                    sse = np.sum(dx ** 2 + dy ** 2)
                    ss_tot_x = np.sum((Y_test[:, 0] - np.mean(Y_test[:, 0])) ** 2)
                    ss_tot_y = np.sum((Y_test[:, 1] - np.mean(Y_test[:, 1])) ** 2)
                    denom = ss_tot_x + ss_tot_y

                    # Mahalanobis MAE using the KDE-weighted training
                    # covariance, matching the regressor's convention.
                    w_cov = w_train / (np.sum(w_train) + 1e-12)
                    diff_tr = Y_train - mu
                    cov_tr = (w_cov[:, None] * diff_tr).T @ diff_tr
                    cov_inv = np.linalg.pinv(cov_tr)
                    residual = np.stack([dx, dy], axis=1)
                    quad = np.einsum('ij,jk,ik->i', residual, cov_inv, residual)
                    mahalanobis_mae = float(np.mean(np.sqrt(np.maximum(quad, 0.0))))

                    # Constant predictions: per-axis correlations are
                    # mathematically undefined → NaN, matching the
                    # estimator's NaN-safe convention. The centroid is
                    # already on-manifold so the raw / snapped MAE are
                    # identical.
                    mae_val = float(np.mean(euclidean_dist))
                    metrics = {
                        'r2_spatial': float(1.0 - (sse / denom)) if denom > 0 else 0.0,
                        'euclidean_mae': mae_val,
                        'euclidean_rmse': float(np.sqrt(np.mean(euclidean_dist ** 2))),
                        'euclidean_mae_weighted': float(
                            np.sum(w_test * euclidean_dist) / (np.sum(w_test) + 1e-12)
                        ),
                        'euclidean_mae_raw': mae_val,
                        'mahalanobis_mae': mahalanobis_mae,
                        'mae_x': float(np.mean(np.abs(dx))),
                        'mae_y': float(np.mean(np.abs(dy))),
                        'pearson_x': float('nan'),
                        'pearson_y': float('nan'),
                        'spearman_x': float('nan'),
                        'spearman_y': float('nan'),
                    }

                    fold_weights, fold_intercepts = None, None
                    # The centroid "fit" is closed-form and instantaneous.
                    fold_n_iter = 0
                    fold_converged = True
                    fold_fit_time = 0.0
                    # `null_model_free` has no hyperparameters to tune.
                    fold_lambda_smooth = float('nan')
                    fold_l2_reg = float('nan')
                    fold_grid_audit = {
                        'grid_scores': {},
                        'grid_ses': {},
                        'argmax_pair': None,
                        'one_se_applied': False,
                        'one_se_threshold': None,
                    }
                    fold_tuned_flag = False
                else:
                    # Pick the regularisation strengths for this outer fold.
                    # Tuning is strategy-agnostic: the within-session
                    # X-history shuffled `null` is also tuned, so the
                    # permutation test compares like-against-like
                    # hyperparameters rather than penalising the null by
                    # forcing it to use the actual model's fixed values.
                    if tune_regularization_bool:
                        fold_lambda_smooth, fold_l2_reg, fold_grid_audit = _tune_manifold_regularization(
                            X_train=X_train,
                            Y_train=Y_train,
                            w_train=w_train,
                            groups_train=groups_train,
                            lambda_smooth_grid=lambda_smooth_grid,
                            l2_reg_grid=l2_reg_grid,
                            inner_cv_folds=inner_cv_folds,
                            inner_cv_scoring_metric=inner_cv_scoring_metric,
                            inner_cv_use_one_se_rule=inner_cv_use_one_se_rule,
                            n_features=1,
                            n_time_bins=n_time_bins,
                            spatial_cluster_num=n_clusters,
                            smoothness_derivative_order=smoothness_order,
                            huber_delta=huber_delta,
                            learning_rate=lr,
                            max_iter=max_iter,
                            inner_max_iter=inner_max_iter,
                            tol=tol,
                            random_state=random_seed + fold_idx,
                            verbose=verbose,
                            use_lax_loop=use_lax_loop,
                            regressor_cls=SmoothBivariateRegression,
                        )
                        fold_tuned_flag = True
                    else:
                        fold_lambda_smooth = float(lam_smooth_fixed)
                        fold_l2_reg = float(lam_l2_fixed)
                        fold_grid_audit = {
                            'grid_scores': {},
                            'grid_ses': {},
                            'argmax_pair': None,
                            'one_se_applied': False,
                            'one_se_threshold': None,
                        }
                        fold_tuned_flag = False

                    model = SmoothBivariateRegression(
                        n_features=1,
                        n_time_bins=n_time_bins,
                        lambda_smooth=fold_lambda_smooth,
                        l2_reg=fold_l2_reg,
                        smoothness_derivative_order=smoothness_order,
                        huber_delta=huber_delta,
                        learning_rate=lr,
                        max_iter=max_iter,
                        tol=tol,
                        random_state=random_seed + fold_idx,
                        verbose=verbose,
                        _use_lax_loop=use_lax_loop,
                    )
                    model.fit(X_train, Y_train, sample_weight=w_train)
                    metrics = model.evaluate_metrics(X_test, Y_test, weights=w_test)

                    y_pred_xy = model.predict(X_test, snap=True).astype(np.float32)
                    fold_weights = model.coef_
                    fold_intercepts = model.intercept_
                    fold_n_iter = int(model.n_iter_)
                    fold_converged = bool(model.converged_)
                    fold_fit_time = float(model.fit_time_)

                print(
                    f"      Fold {fold_idx + 1:03d}/{n_splits} | "
                    f"R^2: {metrics['r2_spatial']:.3f} | MAE: {metrics['euclidean_mae']:.4f} "
                    f"| Mahal: {metrics['mahalanobis_mae']:.4f} | "
                    f"λ_sm={fold_lambda_smooth:.3g} l2={fold_l2_reg:.3g} | converged: {fold_converged}",
                    flush=True,
                )

                for m_key, m_val in metrics.items():
                    results[strategy]['folds']['metrics'][m_key].append(m_val)

                results[strategy]['folds']['weights'].append(fold_weights)
                results[strategy]['folds']['intercepts'].append(fold_intercepts)
                results[strategy]['folds']['test_indices'].append(test_idx)
                results[strategy]['folds']['y_true'].append(Y_test)
                results[strategy]['folds']['w_test'].append(w_test)
                results[strategy]['folds']['y_pred_xy'].append(y_pred_xy)
                results[strategy]['folds']['n_iter'].append(fold_n_iter)
                results[strategy]['folds']['converged'].append(fold_converged)
                results[strategy]['folds']['fit_time'].append(fold_fit_time)
                results[strategy]['folds']['selected_lambda_smooth'].append(fold_lambda_smooth)
                results[strategy]['folds']['selected_l2_reg'].append(fold_l2_reg)
                results[strategy]['folds']['hyperparam_grid_audit'].append(fold_grid_audit)
                results[strategy]['folds']['hyperparams_tuned'].append(fold_tuned_flag)

        # Summary Wilcoxon tests against the two null baselines. Error
        # metrics are "lower is better"; `r2_spatial` and the correlations
        # are "higher is better against the null", so the alternative flips
        # accordingly.
        higher_is_better = {
            'r2_spatial', 'pearson_x', 'pearson_y', 'spearman_x', 'spearman_y',
        }

        print(f"\n" + "=" * 90)
        print(f"FINAL STATISTICAL SUMMARY: {feat_name}")
        print("=" * 90)

        for metric in metric_keys:
            act_vals = np.asarray(results['actual']['folds']['metrics'][metric], dtype=float)
            null_vals = np.asarray(results['null']['folds']['metrics'][metric], dtype=float)
            mf_vals = np.asarray(results['null_model_free']['folds']['metrics'][metric], dtype=float)

            act_m = float(np.nanmean(act_vals)) if act_vals.size else float('nan')
            null_m = float(np.nanmean(null_vals)) if null_vals.size else float('nan')
            mf_m = float(np.nanmean(mf_vals)) if mf_vals.size else float('nan')

            alt = 'greater' if metric in higher_is_better else 'less'

            try:
                _, p_null = wilcoxon(act_vals, null_vals, alternative=alt)
            except ValueError:
                p_null = 1.0
            try:
                _, p_mf = wilcoxon(act_vals, mf_vals, alternative=alt)
            except ValueError:
                p_mf = 1.0

            sig_null = "***" if p_null < 0.01 else "   "
            sig_mf = "***" if p_mf < 0.01 else "   "

            print(
                f"{metric:<22} | Act: {act_m:>7.4f} | Null: {null_m:>7.4f} "
                f"(p={p_null:>7.1e}) {sig_null} | MF: {mf_m:>7.4f} "
                f"(p={p_mf:>7.1e}) {sig_mf}"
            )

        # Cross-fold hyperparameter report. If tuning was on, the spread of
        # the selected `(λ_smooth, l2_reg)` across folds is a direct
        # diagnostic: tight concentration (small SD of log10) means the
        # data pick the same regularisation strength regardless of which
        # folds they see; wide spread means the "right" amount of
        # regularisation is itself fold-dependent and any single number
        # would be a poor summary.
        if tune_regularization_bool:
            print("-" * 90)
            print("HYPERPARAMETER SELECTION SUMMARY (log10 units)")
            for strategy in ('actual', 'null'):
                lam_sm_vals = np.asarray(
                    results[strategy]['folds']['selected_lambda_smooth'], dtype=float
                )
                lam_l2_vals = np.asarray(
                    results[strategy]['folds']['selected_l2_reg'], dtype=float
                )
                grid_audits = results[strategy]['folds']['hyperparam_grid_audit']
                one_se_fired = [bool(a.get('one_se_applied')) for a in grid_audits]
                one_se_count = int(sum(one_se_fired))
                one_se_total = len(one_se_fired)
                valid_sm = lam_sm_vals[np.isfinite(lam_sm_vals) & (lam_sm_vals > 0)]
                valid_l2 = lam_l2_vals[np.isfinite(lam_l2_vals) & (lam_l2_vals > 0)]
                if valid_sm.size == 0 or valid_l2.size == 0:
                    continue
                log_sm = np.log10(valid_sm)
                log_l2 = np.log10(valid_l2)
                print(
                    f"  {strategy.upper():<6} | "
                    f"λ_smooth: mean(log10)={np.mean(log_sm):+.2f}, SD(log10)={np.std(log_sm, ddof=1):.2f} | "
                    f"l2: mean(log10)={np.mean(log_l2):+.2f}, SD(log10)={np.std(log_l2, ddof=1):.2f} | "
                    f"1SE-softened: {one_se_count}/{one_se_total}"
                )

        print("=" * 90 + "\n")

        return results
