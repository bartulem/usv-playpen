"""
@author: bartulem
Module for visualizing modeling results and raw behavioral features.

This module provides a comprehensive suite of plotting functions to interpret
vocal/behavioral modeling outcomes across two complementary model families:
the linear/GAM-based feature ranking pipeline and the deep CNN-based manifold
mapping pipeline.

Linear / GAM-based visualizations
---------------------------------
1.  Feature importance ranking: Comparison of actual model performance
    (LL/AUC/etc.) against null distributions with Bonferroni-corrected
    significance testing.
2.  Temporal filter analysis: Visualizing learned kernels (filter shapes) with
    99% confidence intervals for significantly predictive features.
3.  Model selection trajectories: Tracking the improvement of model fit and
    classification metrics during forward sequential feature selection.
4.  Raw data validation: Generating heatmaps and bootstrap-averaged profiles
    to compare raw behavioral feature intensities between conditions
    (e.g., USV vs. No-USV).
5.  Multi-subject comparison: Consistent color-coding and labeling for
    Male (self/other) and Female (self/other) subjects across dyads.

Deep non-linear USV manifold visualizations
-------------------------------------------
A specialized interpretation suite for Dual-Stream MLP/CNN models that
quantitatively and qualitatively assess how behavioral kinematics map onto
the continuous acoustic UMAP manifold.

1.  Statistical validation: Bootstrapped permutation testing against null
    models.
2.  Global importance: SNR-weighted feature ranking to identify primary
    drivers.
3.  Spatial calibration: Tiled density grids with Euclidean bias
    (Delta d) calculation.
4.  Local saliency: Contrastive gradient attribution to identify region-
    specific motifs.
"""

import cmasher as cmr
import json
import math
import warnings
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.transforms as mtransforms
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D
import numpy as np
import os
import pathlib
import pickle
import re
import seaborn as sns
from scipy.stats import gaussian_kde
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.cluster import KMeans
from typing import Optional
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

from ..modeling.modeling_metadata import RESERVED_METADATA_KEYS, load_selection_results
from ..modeling.manifold_metric import pairwise_distance
from ..analyses.compute_behavioral_features import FeatureZoo
from ..os_utils import configure_path
from .plot_style import apply_plot_style


apply_plot_style()

# Load `visualizations_settings.json` at module import time to resolve
# the canonical sex / social / cmap palette. The module-level constants
# below are derived from this single source; in particular both the
# multinomial-filter `DYADIC_COLOR` and the timescale-audit
# `TIMESCALE_SOCIAL_COLOR` now share the `social_colors[0]` value, so
# the project's "social-feature" colour is in one place.
_PKG_ROOT = pathlib.Path(__file__).parent.parent
with (_PKG_ROOT / "_parameter_settings" / "visualizations_settings.json").open() as _vf:
    _VIZ_SETTINGS = json.load(_vf)

# Global color definitions
male_color = _VIZ_SETTINGS["male_colors"][0]
female_color = _VIZ_SETTINGS["female_colors"][0]
DYADIC_COLOR = _VIZ_SETTINGS["social_colors"][0]
NEUTRAL_COLOR = "#D3D3D3"
MEAN_LINE_COLOR = '#DCB400'
TEXT_COLOR = '#202020'
REFERENCE_LINE_COLOR = "#808080"    # dashed zero / chance reference lines (was 'gray')

# Timescale-audit palette overrides: zero / axis lines stay black, so
# social/dyadic gets the canonical social colour (distinct from the
# axis lines and from the male/female sex colors).
TIMESCALE_SOCIAL_COLOR = DYADIC_COLOR
TIMESCALE_AXIS_COLOR = "#000000"   # zero / spine reference lines (was 'black')
TIMESCALE_NULL_COLOR = "#808080"   # circular-shift null fill / envelope (was 'gray')

# Global default colormap — shared with `figures.cmap` so the
# multinomial / continuous heatmap defaults match the rest of the repo.
_GLOBAL_CMAP = _VIZ_SETTINGS["figures"]["cmap"]

def plot_feature_ranking(
        results_file_loc: str,
        p_val: float = 0.01,
        evaluation_metric: str = 'explained_deviance',
        evaluation_metric_name: str = 'Explained Deviance (D²)',
        secondary_metric: str = 'spearman_r',
        secondary_metric_name: str = "Spearman's Rho",
        ignore_features: list | None = None,
        save_plot: bool = False,
        output_dir: str = None
) -> None:
    """
    Generates ranking plots for univariate modeling feature importance.

    Produces two plots:
    1. Evaluation Metric: Sorted by value, colored by its own significance.
    2. Secondary Metric: Sorted by value, but colored by the significance of the Evaluation Metric.

    Parameters
    ----------
    results_file_loc : str
        Path to the .pkl results file.
    p_val : float
        The alpha level for significance testing (default 0.01).
        Bonferroni-corrected internally.
    evaluation_metric : str
        The key for the primary metric used to determine significance (color).
    evaluation_metric_name : str
        Label for the Y-axis of the evaluation metric plot.
    secondary_metric : str
        The key for the secondary metric (plotted with colors derived from eval metric).
    secondary_metric_name : str
        Label for the Y-axis of the secondary metric plot.
    ignore_features : list, optional
        List of feature names to exclude from the plot.
    save_plot : bool
        If True, saves figures to output_dir.
    output_dir : str, optional
        Directory to save the figure.
    """

    results_file_loc = configure_path(str(results_file_loc))
    if output_dir is not None:
        output_dir = configure_path(str(output_dir))

    if ignore_features is None:
        ignore_features = []

    lower_is_better = ['ll', 'residual_deviance', 'msle']

    results_path = pathlib.Path(results_file_loc)
    with open(results_path, 'rb') as f:
        modeling_data = pickle.load(f)

    fname_str = results_path.name
    if '_male_' in fname_str:
        self_color = male_color
        other_color = female_color
        self_label = "Male (self)"
        other_label = "Female (other)"
    elif '_female_' in fname_str:
        self_color = female_color
        other_color = male_color
        self_label = "Female (self)"
        other_label = "Male (other)"
    else:
        self_color = male_color
        other_color = female_color
        self_label = "Self"
        other_label = "Other"

    # Strip reserved metadata blocks (`_input_metadata`, `_run_metadata`,
    # `_univariate_metadata`, `_consolidation_metadata`) before scanning
    # for the null-distribution key or iterating feature entries — they
    # are top-level dict keys in the consolidated artifact but are not
    # behavioral features.
    feature_keys = [k for k in modeling_data.keys() if k not in RESERVED_METADATA_KEYS]
    first_feat = modeling_data[feature_keys[0]]
    if 'shuffled' in first_feat:
        null_key = 'shuffled'
        print("Detected 'shuffled' key (Bout Analysis mode)")
    elif 'null' in first_feat:
        null_key = 'null'
        print("Detected 'null' key (Category Analysis mode)")
    else:
        raise KeyError("Could not find 'shuffled' or 'null' key in results dictionary.")

    print(f"Calculating significance based on: {evaluation_metric_name}...")

    valid_features = [k for k in feature_keys if k not in ignore_features]
    n_features = len(valid_features)
    corrected_p_top = (1 - (p_val / n_features)) * 100
    corrected_p_bottom = (p_val / n_features) * 100

    significance_map = {}

    for feature in valid_features:
        actual_data = modeling_data[feature]['actual'][evaluation_metric]
        null_data = modeling_data[feature][null_key][evaluation_metric]

        valid_actual = actual_data[~np.isnan(actual_data)]
        valid_null = null_data[~np.isnan(null_data)]

        if len(valid_actual) == 0:
            significance_map[feature] = False
            continue

        actual_mean = np.mean(valid_actual)

        is_sig = False
        if evaluation_metric in lower_is_better:
            threshold = np.percentile(valid_null, q=corrected_p_bottom)
            if actual_mean < threshold:
                is_sig = True
        else:
            threshold = np.percentile(valid_null, q=corrected_p_top)
            if actual_mean > threshold:
                is_sig = True

        significance_map[feature] = is_sig

    plot_configs = [
        (evaluation_metric, evaluation_metric_name),
        (secondary_metric, secondary_metric_name)
    ]

    for metric_key, metric_label in plot_configs:
        print(f"Plotting: {metric_label}...")

        kept_features = []
        means_list = []

        for feature in valid_features:
            raw_vals = modeling_data[feature]['actual'][metric_key]
            valid_vals = raw_vals[~np.isnan(raw_vals)]

            if len(valid_vals) == 0:
                continue

            kept_features.append(feature)
            means_list.append(np.mean(valid_vals))

        feat_arr = np.array(kept_features)
        mean_arr = np.array(means_list)

        if metric_key in lower_is_better:
            sorted_indices = np.argsort(mean_arr)
        else:
            sorted_indices = np.argsort(mean_arr)[::-1]

        feats_sorted = feat_arr[sorted_indices]
        means_sorted = mean_arr[sorted_indices]

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 5), dpi=300, tight_layout=True)
        fig.patch.set_facecolor('#FFFFFF')
        ax.set_facecolor('#FFFFFF')

        for spine in ax.spines.values():
            spine.set_edgecolor(TEXT_COLOR)

        ax.tick_params(axis='x', colors=TEXT_COLOR)
        ax.tick_params(axis='y', colors=TEXT_COLOR)
        ax.xaxis.label.set_color(TEXT_COLOR)
        ax.yaxis.label.set_color(TEXT_COLOR)

        for i, feature_name in enumerate(feats_sorted):

            is_significant = significance_map[feature_name]

            if is_significant:
                dyadic_keywords = ["nose-nose", "nose-TTI", "TTI-nose", "allo_yaw-nose",
                                   "nose-allo_yaw", "allo_yaw-TTI", "TTI-allo_yaw",
                                   "allo_pitch-nose", "nose-allo_pitch",
                                   "allo_pitch-TTI", "TTI-allo_pitch"]
                if any(x in feature_name for x in dyadic_keywords):
                    feat_color = DYADIC_COLOR
                elif '-sei' in feature_name:
                    # Per the column-selection rule in `modeling_utils.select_kinematic_columns`,
                    # SEI columns are kept in the `{target}-{predictor}.{feature}` orientation,
                    # so the surviving SEI signal is the target's attention to the partner.
                    # The target is "self" → self_color, matching `_classify_predictor_feature`.
                    feat_color = self_color
                elif 'self' in feature_name:
                    feat_color = self_color
                else:
                    feat_color = other_color
            else:
                feat_color = NEUTRAL_COLOR

            raw_values = modeling_data[feature_name]['actual'][metric_key]
            raw_values = raw_values[~np.isnan(raw_values)]

            mean_pos = 0.25 + (0.25 * i)

            x_positions = np.random.normal(loc=mean_pos, scale=0.005, size=len(raw_values))

            ax.scatter(x_positions, raw_values, color=feat_color, s=10, alpha=0.15, edgecolors='none')

            actual_mean = means_sorted[i]
            ax.hlines(y=actual_mean, xmin=mean_pos - 0.08, xmax=mean_pos + 0.08, lw=1.5, color=MEAN_LINE_COLOR)

        ax.set_xlabel('Behavioral Feature', fontsize=12, color=TEXT_COLOR)
        ax.set_xticks(np.arange(0.25, (0.25 * len(feats_sorted)) + 0.25, 0.25))
        ax.set_xticklabels(feats_sorted, rotation=60, ha='right', fontsize=6, color=TEXT_COLOR)

        ax.set_ylabel(f'{metric_label} (Held-out Data)', fontsize=12, color=TEXT_COLOR)
        ax.tick_params(axis='both', colors=TEXT_COLOR)
        ax.minorticks_off()

        if metric_key == 'auc':
            ax.axhline(0.5, ls='--', lw=0.5, color='#202020', zorder=0)
        elif metric_key == 'explained_deviance' or metric_key == 'd2':
            ax.axhline(0.0, ls='--', lw=0.5, color='#202020', zorder=0)

        legend_elements = [
            Patch(facecolor=self_color, alpha=0.7, label=self_label),
            Patch(facecolor=other_color, alpha=0.7, label=other_label),
            Patch(facecolor=DYADIC_COLOR, alpha=0.7, label='Social features'),
            Patch(facecolor=NEUTRAL_COLOR, alpha=0.7, label='Not significant\n(in Eval Metric)'),
            Line2D([0], [0], color=MEAN_LINE_COLOR, lw=1.5, label='Mean')
        ]

        leg = ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1),
                        fontsize=8, frameon=False)
        for text in leg.get_texts():
            text.set_color(TEXT_COLOR)

        if save_plot:
            if output_dir is None:
                output_dir = results_path.parent
            out_name = pathlib.Path(output_dir) / f"{results_path.stem}_{metric_key}_ranking.svg"
            fig.savefig(out_name, bbox_inches='tight', dpi=300)
            print(f"Plot saved to {out_name}")

        plt.show()

def plot_significant_filters(
        results_file_loc: str,
        metric: str = 'auc',
        ignore_features: list | None = None,
        p_val: float = 0.01,
        save_plot: bool = False,
        output_dir: str = None
) -> None:
    """
    Plots the temporal filters (kernels) for significantly predictive features.

    For every feature whose actual performance exceeds the Bonferroni-corrected
    null percentile on the selected metric, the function draws the mean filter
    shape and a two-sided confidence interval (derived from the per-fold filter
    ensemble) against time-before-vocalization. Self/other/dyadic features are
    color-coded from the filename and a dashed zero-reference line is shown.

    Parameters
    ----------
    results_file_loc : str
        Absolute path to the .pkl results file containing per-feature
        'actual' and 'shuffled'/'null' fold-level metrics and filter shapes.
    metric : str, default 'auc'
        Metric used to determine significance (e.g., 'auc' or 'll'). The
        lower-is-better metric 'll' has its comparison direction inverted
        (significant when the actual mean falls below the lower null
        threshold); all other metrics are treated as higher-is-better.
    ignore_features : list of str, optional
        Feature names to skip. If None, no features are excluded.
    p_val : float, default 0.01
        Family-wise alpha level for significance testing. Internally
        Bonferroni-corrected by the number of valid features.
    save_plot : bool, default False
        If True, saves each per-feature plot as an SVG to disk.
    output_dir : str, optional
        Directory to save plots. If None, defaults to the parent directory
        of `results_file_loc`.
    """

    results_file_loc = configure_path(str(results_file_loc))
    if output_dir is not None:
        output_dir = configure_path(str(output_dir))

    if ignore_features is None:
        ignore_features = []

    results_path = pathlib.Path(results_file_loc)
    with open(results_path, 'rb') as f:
        modeling_data = pickle.load(f)

    fname_str = results_path.name
    if '_male_' in fname_str:
        self_color, other_color = male_color, female_color
    elif '_female_' in fname_str:
        self_color, other_color = female_color, male_color
    else:
        self_color, other_color = male_color, female_color

    # Strip reserved metadata blocks before iterating feature entries.
    feature_keys = [k for k in modeling_data.keys() if k not in RESERVED_METADATA_KEYS]
    first_feat = modeling_data[feature_keys[0]]
    if 'shuffled' in first_feat:
        null_key = 'shuffled'
    elif 'null' in first_feat:
        null_key = 'null'
    else:
        raise KeyError("Missing 'shuffled' or 'null' key")

    valid_features = [k for k in feature_keys if k not in ignore_features]
    n_feats = len(valid_features)

    # Bonferroni correction for CI
    p_lower = (p_val / n_feats / 2) * 100
    p_upper = (1 - (p_val / n_feats / 2)) * 100

    for feature in valid_features:
        actual_vals = modeling_data[feature]['actual'][metric]
        null_vals = modeling_data[feature][null_key][metric]

        valid_actual = actual_vals[~np.isnan(actual_vals)]
        valid_null = null_vals[~np.isnan(null_vals)]

        if len(valid_actual) == 0 or len(valid_null) == 0: continue

        mean_val = np.mean(valid_actual)
        null_lower_thresh = np.percentile(valid_null, p_lower)
        null_upper_thresh = np.percentile(valid_null, p_upper)

        is_significant = False
        if metric == 'll':
            if mean_val < null_lower_thresh: is_significant = True
        else:
            if mean_val > null_upper_thresh: is_significant = True

        if not is_significant:
            continue

        try:
            filter_data = modeling_data[feature]['actual']['filter_shapes']

            if np.all(np.isnan(filter_data)): continue

            mean_filter = np.nanmean(filter_data, axis=0)

            p_low = np.nanpercentile(filter_data, q=0.5, axis=0)
            p_high = np.nanpercentile(filter_data, q=99.5, axis=0)

        except KeyError:
            print(f"Filter shapes missing for {feature}")
            continue

        dyadic_keywords = ["nose-nose", "nose-TTI", "TTI-nose", "allo_yaw-nose",
                           "nose-allo_yaw", "allo_yaw-TTI", "TTI-allo_yaw",
                           "allo_pitch-nose", "nose-allo_pitch",
                           "allo_pitch-TTI", "TTI-allo_pitch"]
        if any(x in feature for x in dyadic_keywords):
            feat_color = DYADIC_COLOR
        elif '-sei' in feature:
            # SEI signals are target-attending-to-predictor (see column-selection rule);
            # target is "self" → self_color.
            feat_color = self_color
        elif 'self' in feature:
            feat_color = self_color
        else:
            feat_color = other_color

        fig, ax = plt.subplots(figsize=(3, 2.5), dpi=300, facecolor='#FFFFFF', tight_layout=True)
        ax.set_facecolor('#FFFFFF')

        filter_size_vector = np.arange(mean_filter.size)

        ax.plot(filter_size_vector, mean_filter, color=feat_color, lw=1.5)
        ax.fill_between(filter_size_vector, p_low, p_high, color=feat_color, alpha=0.3, edgecolor='none')

        ax.set_xlabel(xlabel='Time prior to event (s)', fontsize=10, color=TEXT_COLOR)
        ax.set_ylabel(ylabel='Filter Amplitude', fontsize=10, color=TEXT_COLOR)
        ax.set_title(label=f'{feature}', fontsize=10, color=TEXT_COLOR)

        ax.tick_params(axis='both', colors=TEXT_COLOR, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(TEXT_COLOR)

        tick_positions = np.arange(0, 650, 75)

        tick_labels = np.round(np.arange(-4, 0.5, 0.5), decimals=2)

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)

        ax.minorticks_off()
        ax.axhline(0, color=NEUTRAL_COLOR, ls='--', lw=0.5, zorder=0)

        if save_plot:
            if output_dir is None: output_dir = results_path.parent
            safe_name = feature.replace('.', '_')
            out_name = pathlib.Path(output_dir) / f"{results_path.stem}_filter_{safe_name}.svg"
            fig.savefig(out_name, bbox_inches='tight', dpi=300, facecolor='#FFFFFF', transparent=False)
            print(f"Saved: {out_name.name}")

        plt.show()
        # Close the figure unconditionally so per-feature figures don't
        # accumulate (and trip matplotlib's >20-open-figures warning)
        # when save_plot is False.
        plt.close(fig)


def plot_significant_filters_grid(
        results_file_loc: str,
        ignore_features: list | None = None,
        metric: str = 'auc',
        p_val_threshold: float = 0.01,
        save_plot: bool = False,
        output_dir: str = None,
) -> None:
    """
    Loads results, calculates significance, baseline-corrects, and plots filter shapes.

    This function loads a pickle file containing modeling results, identifies
    significantly predictive features using a Bonferroni-corrected threshold
    on a specified metric, and plots their temporal filters (kernels) with
    99% confidence intervals in a small-multiples grid. Each filter is
    baseline-corrected by subtracting its value at the leftmost time bin so
    all kernels start at zero and can be visually compared on a shared
    Y-axis. The filter history length (seconds) is inferred from the filename
    by matching `histNs` and defaults to 4.0 seconds if not present.

    Parameters
    ----------
    results_file_loc : str
        The absolute path to the .pkl results file.
    ignore_features : list of str, optional
        A list of feature names to exclude from analysis. Default is None.
    metric : str, default 'auc'
        The performance metric used to assess significance (e.g., 'auc', 'll').
        For lower-is-better metrics (ll, nll, rmse, mse, loss) the direction
        of the significance test is automatically inverted.
    p_val_threshold : float, default 0.01
        The family-wise alpha level for significance testing. This value is
        Bonferroni-corrected by the number of valid features found in the
        file.
    save_plot : bool, default False
        Whether to save the generated SVG plot to disk.
    output_dir : str, optional
        The directory where the SVG plot will be saved. If None, defaults to
        the parent directory of `results_file_loc`.

    Returns
    -------
    None
        The function displays the plot and saves it to disk; it does not return objects.
    """

    results_file_loc = configure_path(str(results_file_loc))
    if output_dir is not None:
        output_dir = configure_path(str(output_dir))

    if ignore_features is None:
        ignore_features = []

    print(f"Loading results from: {results_file_loc}")
    try:
        with open(results_file_loc, 'rb') as f:
            modeling_data = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {results_file_loc}")
        return

    dyadic_color = DYADIC_COLOR
    fname_str = pathlib.Path(results_file_loc).name
    if '_male_' in fname_str:
        self_color, other_color = male_color, female_color
    elif '_female_' in fname_str:
        self_color, other_color = female_color, male_color
    else:
        self_color, other_color = male_color, female_color

    kept_features = []
    means_list = []
    all_shuffled_values_list = []
    all_filter_shapes = {}

    print(f"--- Processing features for metric: {metric} ---")

    for feature in modeling_data.keys():
        if feature in RESERVED_METADATA_KEYS:
            continue
        if feature in ignore_features:
            continue

        if 'actual' not in modeling_data[feature]:
            continue
        if metric not in modeling_data[feature]['actual']:
            continue
        if 'filter_shapes' not in modeling_data[feature]['actual']:
            continue

        try:
            actual_metric_data = modeling_data[feature]['actual'][metric]
            if 'shuffled' in modeling_data[feature]:
                shuffled_metric_data = modeling_data[feature]['shuffled'][metric]
            elif 'null' in modeling_data[feature]:
                shuffled_metric_data = modeling_data[feature]['null'][metric]
            else:
                continue
        except KeyError:
            continue

        valid_actual = actual_metric_data[~np.isnan(actual_metric_data)]
        valid_shuffled = shuffled_metric_data[~np.isnan(shuffled_metric_data)]

        if valid_actual.size > 0 and valid_shuffled.size > 0:
            kept_features.append(feature)
            means_list.append(np.mean(valid_actual))
            all_shuffled_values_list.append(valid_shuffled)
            all_filter_shapes[feature] = modeling_data[feature]['actual']['filter_shapes']

    if not kept_features:
        print("No valid features found. Exiting.")
        return

    behavioral_features = np.array(kept_features)
    behavioral_feature_means = np.array(means_list)

    n_features = len(behavioral_features)
    p_thresh_corrected = p_val_threshold / n_features

    lower_is_better = metric in ['ll', 'nll', 'rmse', 'mse', 'loss']

    lower_q = p_thresh_corrected * 100
    upper_q = (1 - p_thresh_corrected) * 100

    is_significant_list = []
    for i in range(n_features):
        actual_mean = behavioral_feature_means[i]
        shuffled_dist = all_shuffled_values_list[i]

        if lower_is_better:
            thresh = np.percentile(shuffled_dist, lower_q)
            is_significant_list.append(actual_mean < thresh)
        else:
            thresh = np.percentile(shuffled_dist, upper_q)
            is_significant_list.append(actual_mean > thresh)

    sig_indices = [i for i, x in enumerate(is_significant_list) if x]

    if not sig_indices:
        print(f"Found 0 significant features (Corrected p < {p_thresh_corrected:.5f}).")
        return

    significant_features = behavioral_features[sig_indices]
    significant_feature_means = behavioral_feature_means[sig_indices]

    if lower_is_better:
        sort_idx = np.argsort(significant_feature_means)
    else:
        sort_idx = np.argsort(significant_feature_means)[::-1]

    significant_features = significant_features[sort_idx]
    significant_feature_means = significant_feature_means[sort_idx]

    n_significant = len(significant_features)
    print(f"Found {n_significant} significant features.")

    plot_data = []
    all_y_values = []

    hist_sec_match = re.search(r"hist(\d+\.?\d*)s", results_file_loc)
    filter_history_sec = float(hist_sec_match.group(1)) if hist_sec_match else 4.0

    for i, beh_feature in enumerate(significant_features):
        filter_data = all_filter_shapes[beh_feature]

        regression_filter_mean = np.nanmean(filter_data, axis=0)
        filter_ci_lower = np.nanpercentile(filter_data, q=0.5, axis=0)  # 99% CI
        filter_ci_upper = np.nanpercentile(filter_data, q=99.5, axis=0)

        baseline_offset = regression_filter_mean[0]

        filter_mean_corrected = regression_filter_mean - baseline_offset
        ci_lower_corrected = filter_ci_lower - baseline_offset
        ci_upper_corrected = filter_ci_upper - baseline_offset

        all_y_values.extend(ci_lower_corrected)
        all_y_values.extend(ci_upper_corrected)

        if any(x in beh_feature for x in ["nose-nose", "nose-TTI", "TTI-nose", "neck_elevation_diff",
                                          "allo_yaw-nose", "nose-allo_yaw", "allo_yaw-TTI", "TTI-allo_yaw",
                                          "allo_pitch-nose", "nose-allo_pitch",
                                          "allo_pitch-TTI", "TTI-allo_pitch"]):
            c = dyadic_color
        elif '-sei' in beh_feature:
            # SEI signals are target-attending-to-predictor; target is "self" → self_color.
            c = self_color
        elif 'self' in beh_feature:
            c = self_color
        else:
            c = other_color

        plot_data.append({
            'name': beh_feature,
            'mean': filter_mean_corrected,
            'lower': ci_lower_corrected,
            'upper': ci_upper_corrected,
            'metric_val': significant_feature_means[i],  # Renamed key
            'color': c,
            'n_frames': regression_filter_mean.shape[0]
        })

    ncols = 4
    nrows = math.ceil(n_significant / ncols)

    fig_grid, axes_grid = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(ncols * 2.5, nrows * 2.0),
        dpi=300,
        sharex=True,
        sharey=True
    )

    fig_grid.patch.set_facecolor('#FFFFFF')

    if nrows == 1 and ncols == 1:
        axes_grid = np.array([axes_grid])
    axes_grid = axes_grid.flatten()

    global_ymin = np.nanmin(all_y_values)
    global_ymax = np.nanmax(all_y_values)
    y_range = global_ymax - global_ymin
    global_ymin -= y_range * 0.1
    global_ymax += y_range * 0.1

    axes_grid[0].set_ylim(global_ymin, global_ymax)

    for i, data in enumerate(plot_data):
        ax = axes_grid[i]

        ax.set_facecolor('#FFFFFF')
        ax.tick_params(axis='x', colors='#000000')
        ax.tick_params(axis='y', colors='#000000')
        for spine in ax.spines.values():
            spine.set_edgecolor('#000000')

        time_axis_sec = np.linspace(-filter_history_sec, 0, data['n_frames'])

        ax.plot(time_axis_sec, data['mean'], color=data['color'], lw=1.5, alpha=1.0)
        ax.fill_between(time_axis_sec, data['lower'], data['upper'], color=data['color'], alpha=0.3, lw=0)

        ax.axhline(0, color=REFERENCE_LINE_COLOR, linestyle='--', lw=0.7)
        ax.set_title(f"{data['name']}\n{metric.upper()}: {data['metric_val']:.2f}", fontsize=8, color='#000000')

        tick_locs = [-filter_history_sec, -filter_history_sec / 2, 0.0]
        ax.set_xticks(tick_locs)
        ax.set_xticklabels([f"{t:.1f}" for t in tick_locs], fontsize=7)
        ax.tick_params(axis='y', labelsize=7)
        ax.minorticks_off()

    for i in range(n_significant, len(axes_grid)):
        fig_grid.delaxes(axes_grid[i])

    for i in range(n_significant):
        ax = axes_grid[i]

        row = i // ncols
        col = i % ncols

        idx_below = (row + 1) * ncols + col
        is_bottom = (row == nrows - 1) or (idx_below >= n_significant)

        if is_bottom:
            ax.tick_params(labelbottom=True)

    if save_plot:
        if output_dir is None: output_dir = pathlib.Path(results_file_loc).parent
        out_name = pathlib.Path(output_dir) / f"{pathlib.Path(results_file_loc).stem}_filter_grid.svg"
        fig_grid.savefig(out_name, bbox_inches='tight', dpi=300, facecolor='#FFFFFF', transparent=False)
        print(f"Saved: {out_name.name}")
        plt.close(fig_grid)

    plt.tight_layout()
    plt.show()


def plot_raw_feature_difference(
        pickle_file_path: str,
        feature_key: str,
        feature_color: str = '#202020',
        subset_fraction: float = 0.05,
        n_bootstraps: int = 1000,
        save_plots: bool = False,
        output_dir: str = None,
        value_sanity_cap: float | None = 90.0
) -> None:
    """
    Visualizes the raw difference of a specific feature between two conditions.

    This function loads modeling data from a pickle file, automatically detects the
    data structure (either "Bout Onset" or "Vocal Category"), and generates two
    complementary visualizations:
    1.  **Bootstrap Average Plot:** A line plot comparing the mean feature value
        of the target condition vs. the contrast condition (e.g., USV vs. No-USV).
        Shaded regions represent the 99% Confidence Interval calculated via
        bootstrapping on a random subset of the data.
    2.  **Raw Heatmaps:** Two stacked heatmaps showing the raw feature intensity
        for every single epoch in the dataset, sorted by their initial value to
        reveal structure.

    Parameters
    ----------
    pickle_file_path : str
        The absolute path to the .pkl file containing the modeling input data.
    feature_key : str
        The dictionary key representing the specific feature to analyze
        (e.g., 'nose-nose_1st_der').
    feature_color : str, default '#202020'
        The hex color code used to plot the primary ("Target" or "USV") condition.
    subset_fraction : float, default 0.05
        The fraction of total epochs to randomly sample for the bootstrap
        calculation. Using a subset speeds up processing for large datasets.
    n_bootstraps : int, default 1000
        The number of bootstrap iterations to perform for confidence interval estimation.
    save_plots : bool, default False
        If True, saves the resulting average plot (.svg) and heatmap (.png) to disk.
    output_dir : str, optional
        The directory where plots will be saved. If None, defaults to the
        parent directory of the input pickle file.
    value_sanity_cap : float or None, default 90.0
        Per-frame values that exceed this cap are replaced with NaN before
        the mean/CI so extreme outliers don't bias the estimate. The default
        of 90 suits angular features measured in
        degrees; pass a feature-appropriate cap for non-angular features, or
        ``None`` to disable the filter entirely.

    Returns
    -------
    None
        Displays the plots and optionally saves them to disk.
    """

    pickle_file_path = configure_path(str(pickle_file_path))
    if output_dir is not None:
        output_dir = configure_path(str(output_dir))

    print(f"Loading data from: {pickle_file_path}")
    with open(pickle_file_path, 'rb') as pickle_file:
        modeling_input_data = pickle.load(pickle_file)

    if feature_key not in modeling_input_data:
        print(f"Error: Feature '{feature_key}' not found in pickle file.")
        return

    first_session = next(iter(modeling_input_data[feature_key].values()))

    if 'usv_feature_arr' in first_session:
        key_target = 'usv_feature_arr'
        key_other = 'no_usv_feature_arr'
        label_target = 'USV Bout Trials'
        label_other = 'No-USV Trials'
        figure_name_label = 'bout_onset'
        print("Detected mode: Bout Onset (USV vs No-USV)")

    elif 'target_feature_arr' in first_session:
        key_target = 'target_feature_arr'
        key_other = 'other_feature_arr'
        label_target = 'Target Category'
        label_other = 'Other Categories'
        figure_name_label = 'vocal_categories'
        print("Detected mode: Vocal Categories (Target vs Other)")

    else:
        print("Error: Could not detect standard array keys (usv_feature_arr or target_feature_arr).")
        return

    target_epochs_list = []
    other_epochs_list = []

    for session_key in modeling_input_data[feature_key].keys():
        sess_data = modeling_input_data[feature_key][session_key]

        if key_target in sess_data and key_other in sess_data:
            target_epochs_list.append(sess_data[key_target])
            other_epochs_list.append(sess_data[key_other])

    all_target_epochs = np.concatenate(target_epochs_list, axis=0) if target_epochs_list else np.empty((0, 650))
    all_other_epochs = np.concatenate(other_epochs_list, axis=0) if other_epochs_list else np.empty((0, 650))

    n_time_points = all_target_epochs.shape[1] if all_target_epochs.shape[0] > 0 else 650
    time_in_seconds = np.linspace(-4, 0, n_time_points)

    print(f"Total Epochs -> {label_target}: {all_target_epochs.shape[0]}, {label_other}: {all_other_epochs.shape[0]}")
    print(f"Data Range Check -> Min: {np.nanmin(all_target_epochs):.2f}, Max: {np.nanmax(all_target_epochs):.2f}")

    target_subset_size = max(1, int(all_target_epochs.shape[0] * subset_fraction))
    other_subset_size = max(1, int(all_other_epochs.shape[0] * subset_fraction))

    target_idx = np.random.choice(all_target_epochs.shape[0], size=target_subset_size, replace=False)
    other_idx = np.random.choice(all_other_epochs.shape[0], size=other_subset_size, replace=False)

    target_subset = all_target_epochs[target_idx, :]
    other_subset = all_other_epochs[other_idx, :]

    target_subset = target_subset[~np.all(np.isnan(target_subset), axis=1)]
    other_subset = other_subset[~np.all(np.isnan(other_subset), axis=1)]

    # Sanity filter: values above ``value_sanity_cap`` (extreme outliers, or
    # for angular features in degrees anything beyond the meaningful range)
    # are replaced with NaN so they don't bias the per-frame mean/CI. The cap
    # is caller-configurable (default 90, suited to angular features); pass
    # ``None`` to disable it for features where large values are legitimate.
    if value_sanity_cap is not None:
        target_subset[target_subset > value_sanity_cap] = np.nan
        other_subset[other_subset > value_sanity_cap] = np.nan

    boot_target_means = np.zeros((n_bootstraps, n_time_points))
    boot_other_means = np.zeros((n_bootstraps, n_time_points))

    for i in range(n_bootstraps):
        resample_target = np.random.choice(target_subset.shape[0], size=target_subset.shape[0], replace=True)
        resample_other = np.random.choice(other_subset.shape[0], size=other_subset.shape[0], replace=True)

        boot_target_means[i, :] = np.nanmean(target_subset[resample_target, :], axis=0)
        boot_other_means[i, :] = np.nanmean(other_subset[resample_other, :], axis=0)

    target_mean = np.nanmean(target_subset, axis=0)
    other_mean = np.nanmean(other_subset, axis=0)

    target_ci_lower = np.nanpercentile(boot_target_means, 0.5, axis=0)
    target_ci_upper = np.nanpercentile(boot_target_means, 99.5, axis=0)
    other_ci_lower = np.nanpercentile(boot_other_means, 0.5, axis=0)
    other_ci_upper = np.nanpercentile(boot_other_means, 99.5, axis=0)

    fig_avg, ax_avg = plt.subplots(figsize=(3, 2), dpi=300, tight_layout=True)

    fig_avg.patch.set_facecolor('#FFFFFF')
    ax_avg.set_facecolor('#FFFFFF')

    ax_avg.plot(time_in_seconds, target_mean, color=feature_color, lw=1.0, label=label_target)
    ax_avg.fill_between(time_in_seconds, target_ci_lower, target_ci_upper, color=feature_color, alpha=0.3)

    ax_avg.plot(time_in_seconds, other_mean, color='#D3D3D3', lw=1.0, label=label_other)
    ax_avg.fill_between(time_in_seconds, other_ci_lower, other_ci_upper, color='#D3D3D3', alpha=0.3)

    ax_avg.set_xticks(np.arange(-4, 0.5, 0.5))
    ax_avg.set_xticklabels([f"{sec:.1f}" for sec in np.arange(-4, 0.5, 0.5)])
    ax_avg.set_xlabel('Time prior to event (s)', fontsize=8)
    ax_avg.set_ylabel(f'{feature_key} (z-scored)', fontsize=8)
    ax_avg.legend(fontsize=6)

    ax_avg.tick_params(axis='x', colors='#000000', labelsize=8)
    ax_avg.tick_params(axis='y', colors='#000000', labelsize=8)
    ax_avg.yaxis.label.set_color('#000000')
    ax_avg.xaxis.label.set_color('#000000')
    for spine in ax_avg.spines.values():
        spine.set_edgecolor('#000000')

    sorted_target = all_target_epochs[np.argsort(all_target_epochs[:, 0])[::-1], :]
    sorted_other = all_other_epochs[np.argsort(all_other_epochs[:, 0])[::-1], :]

    fig_heat, axes_heat = plt.subplots(nrows=2, ncols=1, figsize=(2, 4), dpi=300, sharex=True)

    fig_heat.patch.set_facecolor('#FFFFFF')

    data_min = np.nanpercentile(np.concatenate([sorted_target, sorted_other]), 1)
    data_max = np.nanpercentile(np.concatenate([sorted_target, sorted_other]), 99)

    axes_heat[0].imshow(
        sorted_target, aspect='auto', cmap='binary',
        vmin=data_min, vmax=data_max,
        interpolation="gaussian",
        extent=[-4, 0, sorted_target.shape[0], 0]
    )
    axes_heat[0].set_ylabel(label_target, fontsize=8)

    im2 = axes_heat[1].imshow(
        sorted_other, aspect='auto', cmap='binary',
        vmin=data_min, vmax=data_max,
        interpolation="gaussian",
        extent=[-4, 0, sorted_other.shape[0], 0]
    )
    axes_heat[1].set_ylabel(label_other, fontsize=8)

    for ax in axes_heat:
        ax.set_facecolor('#FFFFFF')
        ax.tick_params(axis='x', colors='#000000', labelsize=6)
        ax.tick_params(axis='y', colors='#000000', labelsize=6)
        ax.yaxis.label.set_color('#000000')
        ax.xaxis.label.set_color('#000000')
        for spine in ax.spines.values():
            spine.set_edgecolor('#000000')

    axes_heat[1].set_xticks(np.arange(-4, 0.5, 0.5))
    axes_heat[1].set_xticklabels([f"{x:.1f}" for x in np.arange(-4, 0.5, 0.5)], rotation=0)
    axes_heat[1].set_xlabel('Time prior to event onset (s)', fontsize=8)

    cbar = fig_heat.colorbar(
        im2, ax=axes_heat.ravel().tolist(), shrink=0.6, pad=0.15,
        orientation='horizontal', fraction=0.05, location='top'
    )

    cbar.set_ticks([data_min, (data_min + data_max) / 2, data_max])
    cbar.set_ticklabels([f'{data_min:.1f}', f'{(data_min + data_max) / 2:.1f}', f'{data_max:.1f}'])

    cbar.ax.xaxis.set_ticks_position('bottom')
    cbar.ax.xaxis.set_label_position('bottom')
    cbar.set_label(label='Value', rotation=0, labelpad=2, fontsize=7)

    cbar.ax.xaxis.set_tick_params(color='#000000', labelcolor='#000000')
    cbar.outline.set_edgecolor('#000000')
    cbar.ax.xaxis.label.set_color('#000000')

    plt.show()

    if save_plots:
        if output_dir is None:
            output_dir = pathlib.Path(pickle_file_path).parent
        else:
            output_dir = pathlib.Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        avg_path = output_dir / f"{figure_name_label}_{feature_key}_zscored_data_avg_bootstrap.svg"
        fig_avg.savefig(avg_path, dpi=300)

        heat_path = output_dir / f"{figure_name_label}_{feature_key}_zscored_data_heatmap.svg"
        fig_heat.savefig(heat_path, bbox_inches='tight', dpi=300)


def _resolve_cohort_sexes(selection_metadata: dict | None) -> tuple:
    """
    Returns the ``(target_mouse_sex, predictor_mouse_sex)`` pair from a
    loaded selection artifact's metadata blocks, or ``(None, None)`` when
    the ``_input_metadata`` block (or its sex fields) is absent — e.g. a
    legacy artifact. The sexes drive the per-cohort fill of the canonical
    feature labels (``FeatureZoo.resolve_feature_label``).

    Parameters
    ----------
    selection_metadata : dict or None
        The metadata-blocks dict returned by ``load_selection_results``
        (its third element).

    Returns
    -------
    tuple
        ``(self_sex, other_sex)``; either element may be ``None``.
    """

    input_metadata = (selection_metadata or {}).get('_input_metadata') or {}
    return input_metadata.get('target_mouse_sex'), input_metadata.get('predictor_mouse_sex')


def _make_feature_pretty(feature_label_overrides: dict | None,
                         selection_metadata: dict | None):
    """
    Builds the feature-name -> display-label function used by the
    selection plotters. A per-call ``feature_label_overrides`` entry wins;
    otherwise the label comes from the single source of truth,
    ``FeatureZoo.resolve_feature_label``, filled with the cohort sexes read
    from the artifact's ``_input_metadata`` (so labels are consistent and
    cohort-correct — male- vs female-target — without per-figure dicts).

    Parameters
    ----------
    feature_label_overrides : dict or None
        Optional ``{generic_key: label}`` overrides; take priority.
    selection_metadata : dict or None
        Metadata blocks from ``load_selection_results`` (for the sexes).

    Returns
    -------
    callable
        ``_pretty(fname: str) -> str``.
    """

    overrides = feature_label_overrides if feature_label_overrides is not None else {}
    self_sex, other_sex = _resolve_cohort_sexes(selection_metadata)

    def _pretty(fname: str) -> str:
        if fname in overrides:
            return overrides[fname]
        return FeatureZoo.resolve_feature_label(fname, self_sex, other_sex)

    return _pretty


def plot_model_selection_results(
        selection_results_path: str,
        metric_secondary: str = 'auc',
        save_plots: bool = False,
        output_dir: str = None,
        feature_label_overrides: dict = None,
) -> None:
    """
    Plot a Forward Sequential Feature Selection result set and the
    temporal filters of the final accepted model.

    Two figures are produced:

    1. A two-panel summary figure.

       * **Left panel** -- per-step horizontal bars showing the
         cumulative trajectory of the primary metric (Negative Log-
         Likelihood) one row per accepted step, top-to-bottom in
         selection order. Each bar is split into a lighter base
         (the previous cumulative NLL) and a darker tip (THIS step's
         NLL reduction). Bars are coloured by the same
         self / other / dyadic palette used in the filter grid. A
         rejected final step, if any, is drawn below a thin
         separator in desaturated grey.
       * **Right panel** -- two vertical bars on a balanced-
         classification axis (chance = 0.5):
         best univariate single-feature model and final accepted
         multivariate model. A rejected final step, if any, appears
         as a grey row in the left panel, not as a right-panel bar.
         The features composing each model are listed above the
         corresponding bar.

    2. A grid of per-feature partial-dependence (filter) plots for
       the final accepted model, one panel per feature, plotting
       mean +/- SEM across CV folds.

    Parameters
    ----------
    selection_results_path : str
        Path to the consolidated ``selection_*.pkl`` artifact produced
        by ``consolidate_model_selection_results``. May be either the
        file itself or a directory containing one (the latest by mtime
        wins when multiple are present). The legacy per-step layout is
        no longer supported by the loader.
    metric_secondary : str, default 'auc'
        The key for the secondary metric to plot in the right panel of
        the summary figure (typical values: ``'auc'``, ``'score'``,
        ``'f1'``).
    save_plots : bool, default False
        Whether to save the figures to disk.
    output_dir : str, optional
        Directory to save the plots. Defaults to the parent dir of
        ``selection_results_path`` (or to the path itself if a
        directory was supplied).
    feature_label_overrides : dict, optional
        Mapping from raw feature names (as stored in the pickle) to
        presentation-friendly labels used for every y-tick / annotation
        in the summary figure. Features not present in the map are
        rendered with their raw name. The filter grid keeps the raw
        feature names so it can be cross-referenced with the pickle.
    """

    BG_COLOR = '#FFFFFF'

    # Route input + output paths through ``configure_path`` so the
    # caller can hand in any of the equivalent SMB-mount conventions
    # (``/mnt/falkner/...`` on Linux, ``/Volumes/falkner/...`` on
    # macOS, ``F:\...`` on Windows) and have it resolved for the
    # current OS without per-machine edits to the notebook.
    selection_results_path = configure_path(str(selection_results_path))
    if output_dir is not None:
        output_dir = configure_path(str(output_dir))

    # Load steps via the metadata-aware helper: prefers a consolidated
    # `selection_*.pkl` artifact in the directory, falls back to legacy
    # `*_step_*.pkl` glob. `display_name` keeps the substring-based sex
    # inference below working in both modes.
    selection_steps, display_name, selection_metadata = load_selection_results(selection_results_path)

    if not selection_steps:
        print(f"No step data found in {selection_results_path}")
        return

    if '_male_' in display_name:
        self_color, other_color = male_color, female_color
    elif '_female_' in display_name:
        self_color, other_color = female_color, male_color
    else:
        self_color, other_color = male_color, female_color

    # 1. Determine Valid Steps for Trajectory Plot
    valid_steps_for_plot = list(selection_steps)
    if len(selection_steps) > 1:
        last_data = selection_steps[-1]
        if last_data['selected_feature'] is None:
            print("Last step was a rejection. Excluding it from trajectory plots.")
            valid_steps_for_plot = selection_steps[:-1]

    # 2. Process Trajectory Data
    steps_data = []
    print(f"Processing {len(valid_steps_for_plot)} steps for trajectory...")

    first_data = valid_steps_for_plot[0]
    first_cands = first_data.get('candidates', first_data.get('candidates_summary', {}))

    primary_metric = 'll'
    is_minimization = True
    metric_label = 'Negative Log-Likelihood'

    if first_cands:
        first_key = next(iter(first_cands))
        if 'explained_deviance' in first_cands[first_key]:
            primary_metric = 'explained_deviance'
            is_minimization = False
            metric_label = 'Explained Deviance ($D^2$)'
        elif 'll' in first_cands[first_key]:
            primary_metric = 'll'
            is_minimization = True
            metric_label = 'Negative Log-Likelihood'

    print(f"Primary Metric Detected: {metric_label}")

    for i, data in enumerate(valid_steps_for_plot):
        candidates = data.get('candidates', data.get('candidates_summary', {}))
        if not candidates: continue

        best_feat = None
        best_prim_mean = np.inf if is_minimization else -np.inf

        for feat, metrics in candidates.items():
            if primary_metric not in metrics: continue

            valid_vals = np.array(metrics[primary_metric])
            valid_vals = valid_vals[~np.isnan(valid_vals)]
            if len(valid_vals) == 0: continue

            current_mean = np.mean(valid_vals)

            if is_minimization:
                if current_mean < best_prim_mean:
                    best_prim_mean = current_mean
                    best_feat = feat
            else:
                if current_mean > best_prim_mean:
                    best_prim_mean = current_mean
                    best_feat = feat

        if best_feat is None: continue

        winner_data = candidates[best_feat]

        steps_data.append({
            'step_idx': i,
            'feature_name': best_feat,
            'prim_scores': np.array(winner_data[primary_metric]),
            'prim_mean': best_prim_mean,
            'sec_scores': np.array(winner_data.get(metric_secondary, [np.nan])),
            'sec_mean': np.mean(winner_data.get(metric_secondary, [np.nan]))
        })

    n_steps = len(steps_data)
    if n_steps == 0:
        print("No valid step data extracted.")
        return

    # 3. Summary figure: NLL trajectory (left) + score bars (right)
    # Left panel: horizontal bars per accepted step, growing LEFTWARD
    # from the chance NLL baseline (right side of the axis) toward 0
    # as features are added. Light base = previous cumulative NLL,
    # dark tip = THIS step's NLL reduction. Coloured by self / other
    # / dyadic palette. Optional rejected last step rendered grey
    # below a small visual gap.
    #
    # Right panel: three vertical bars on the secondary-metric axis
    # (e.g., balanced accuracy) with a 0.5 floor:
    #   * best univariate (best secondary-metric mean across step 0
    #     candidates);
    #   * final accepted multivariate model;
    #   * final + the rejected step's best secondary-metric candidate,
    #     drawn grey.
    # Above each bar, a vertical stack of constituent features
    # (renamed via ``feature_label_overrides``) makes the composition
    # of each model self-explanatory without needing a legend.

    dyadic_keywords = ["nose-nose", "nose-TTI", "TTI-nose", "allo_yaw-nose",
                       "nose-allo_yaw", "allo_yaw-TTI", "TTI-allo_yaw",
                       "allo_pitch-nose", "nose-allo_pitch",
                       "allo_pitch-TTI", "TTI-allo_pitch"]

    def _category_color(fname: str) -> str:
        """Return the self / other / dyadic hex colour for a feature."""
        if any(x in fname for x in dyadic_keywords):
            return DYADIC_COLOR
        if '-sei' in fname:
            # SEI signals are target-attending-to-predictor; target is
            # "self" -> self_color.
            return self_color
        if 'self' in fname:
            return self_color
        return other_color

    def _lighten(hex_color: str, factor: float = 0.65) -> str:
        """
        Linearly interpolate a hex colour toward white.

        Parameters
        ----------
        hex_color : str
            ``#RRGGBB`` source colour.
        factor : float
            0.0 returns the source colour unchanged; 1.0 returns
            white. Used to derive the "previous cumulative" shade
            from a feature's category colour.
        """

        h = hex_color.lstrip('#')
        r = int(h[0:2], 16)
        g = int(h[2:4], 16)
        b = int(h[4:6], 16)
        r = int(round(r + (255 - r) * factor))
        g = int(round(g + (255 - g) * factor))
        b = int(round(b + (255 - b) * factor))
        return f"#{r:02x}{g:02x}{b:02x}"

    _pretty = _make_feature_pretty(feature_label_overrides, selection_metadata)

    # Left-panel data: NLL trajectory
    # The selector's primary metric is NLL (minimization). The chance
    # baseline is the null-model NLL stored in step 0's
    # ``baseline_score``; for a balanced binary classifier that is
    # ln(2) per sample. Fall back to ln(2) if the field is missing.
    if 'baseline_score' in selection_steps[0]:
        chance_nll = float(selection_steps[0]['baseline_score'])
    else:
        chance_nll = float(np.log(2.0))
    cum_nlls = [float(d['prim_mean']) for d in steps_data]

    # Right-panel data: secondary-metric bars
    chance_secondary = 0.5
    first_step = valid_steps_for_plot[0]
    if 'candidates_summary' in first_step:
        first_cands_all = first_step['candidates_summary']
    elif 'candidates' in first_step:
        first_cands_all = first_step['candidates']
    else:
        first_cands_all = {}
    best_univariate_value = np.nan
    best_univariate_feat = None
    for _f, _m in first_cands_all.items():
        if metric_secondary not in _m:
            continue
        _v = np.array(_m[metric_secondary], dtype=float)
        _v = _v[~np.isnan(_v)]
        if _v.size == 0:
            continue
        _mu = float(np.mean(_v))
        if np.isnan(best_univariate_value) or _mu > best_univariate_value:
            best_univariate_value = _mu
            best_univariate_feat = _f

    final_score = float(steps_data[-1]['sec_mean'])

    # Rejected-final-step lookup (by secondary metric, so the rejected
    # bar in the right panel and the rejected row in the left panel
    # refer to the same candidate). We also pull the same candidate's
    # mean NLL so the left-panel row has a real value to plot.
    rejection_row = None
    if len(selection_steps) > len(valid_steps_for_plot):
        rej_step = selection_steps[-1]
        if 'candidates_summary' in rej_step:
            rej_cands = rej_step['candidates_summary']
        elif 'candidates' in rej_step:
            rej_cands = rej_step['candidates']
        else:
            rej_cands = {}
        if rej_cands:
            best_rej_feat = None
            best_rej_mean = -np.inf
            for _f, _m in rej_cands.items():
                if metric_secondary not in _m:
                    continue
                _v = np.array(_m[metric_secondary], dtype=float)
                _v = _v[~np.isnan(_v)]
                if _v.size == 0:
                    continue
                _mu = float(np.mean(_v))
                if _mu > best_rej_mean:
                    best_rej_mean = _mu
                    best_rej_feat = _f
            if best_rej_feat is not None:
                _ll_v = np.array(
                    rej_cands[best_rej_feat]['ll'], dtype=float
                )
                _ll_v = _ll_v[~np.isnan(_ll_v)]
                rej_ll_mean = float(np.mean(_ll_v)) if _ll_v.size > 0 else float('nan')
                rejection_row = {
                    'feature_name': best_rej_feat,
                    'score_mean': best_rej_mean,
                    'll_mean': rej_ll_mean,
                }

    # Figure layout
    n_rows_total = len(steps_data) + (1 if rejection_row is not None else 0)
    fig_height = max(3.0, 0.32 * n_rows_total + 1.8)
    fig_traj, (ax_traj, ax_bars) = plt.subplots(
        nrows=1, ncols=2,
        figsize=(10.5, fig_height), dpi=300,
        gridspec_kw={'width_ratios': [2.2, 1.0]},
    )
    fig_traj.patch.set_facecolor(BG_COLOR)
    ax_traj.set_facecolor(BG_COLOR)
    ax_bars.set_facecolor(BG_COLOR)

    # Left panel: NLL trajectory bars (grow leftward)
    bar_height = 0.88
    y_positions = list(range(len(steps_data)))
    rej_y = len(steps_data) + 0.4 if rejection_row is not None else None

    for row_idx, d in enumerate(steps_data):
        y = y_positions[row_idx]
        base_color = _category_color(d['feature_name'])
        light_color = _lighten(base_color, factor=0.65)
        prev_nll = cum_nlls[row_idx - 1] if row_idx > 0 else chance_nll
        cur_nll = cum_nlls[row_idx]
        delta = prev_nll - cur_nll  # positive NLL reduction

        # Light segment: from prev_nll RIGHTWARD to chance_nll
        # (represents the cumulative improvement BEFORE this feature).
        if chance_nll > prev_nll:
            ax_traj.barh(y, chance_nll - prev_nll, left=prev_nll,
                         height=bar_height, color=light_color,
                         edgecolor='none')
        # Dark tip: from cur_nll RIGHTWARD to prev_nll (THIS step's
        # contribution; further leftward extension of the bar).
        if prev_nll > cur_nll:
            ax_traj.barh(y, prev_nll - cur_nll, left=cur_nll,
                         height=bar_height, color=base_color,
                         edgecolor='none')

        ax_traj.text(cur_nll - 0.003, y,
                     f"{cur_nll:.3f}  (Δ -{delta:.3f})",
                     ha='left', va='center', fontsize=7,
                     color=TEXT_COLOR)

    if rejection_row is not None and not np.isnan(rejection_row['ll_mean']):
        rejected_light = '#D7D7D7'
        rejected_dark = '#9A9A9A'
        prev_nll = cum_nlls[-1]
        cur_nll = rejection_row['ll_mean']
        if chance_nll > prev_nll:
            ax_traj.barh(rej_y, chance_nll - prev_nll, left=prev_nll,
                         height=bar_height, color=rejected_light,
                         edgecolor='none')
        if prev_nll > cur_nll:
            ax_traj.barh(rej_y, prev_nll - cur_nll, left=cur_nll,
                         height=bar_height, color=rejected_dark,
                         edgecolor='none')
        ax_traj.text(cur_nll - 0.003, rej_y,
                     f"{cur_nll:.3f}  (Δ -{prev_nll - cur_nll:.3f}, ns)",
                     ha='left', va='center', fontsize=7,
                     color=NEUTRAL_COLOR, style='italic')

    ytick_positions = list(y_positions)
    ytick_labels = [f"+{_pretty(d['feature_name'])}" for d in steps_data]
    if rejection_row is not None:
        ytick_positions.append(rej_y)
        ytick_labels.append(f"+{_pretty(rejection_row['feature_name'])}")
    ax_traj.set_yticks(ytick_positions)
    ax_traj.set_yticklabels(ytick_labels, fontsize=10, color=TEXT_COLOR)
    # Drop the y-axis tick MARKS (the small lines) while keeping the
    # feature-name labels — cleaner look for a categorical axis.
    ax_traj.tick_params(axis='y', length=0)
    ax_traj.invert_yaxis()

    if rejection_row is not None:
        sep_y = (len(steps_data) - 1) + 0.5 + 0.10
        ax_traj.axhline(sep_y, color=NEUTRAL_COLOR, linestyle='-',
                        lw=0.4, alpha=0.5, zorder=0)

    all_left_edges = list(cum_nlls)
    if rejection_row is not None and not np.isnan(rejection_row['ll_mean']):
        all_left_edges.append(rejection_row['ll_mean'])
    nll_span = chance_nll - min(all_left_edges)
    x_left = min(all_left_edges) - 0.30 * nll_span
    x_right = chance_nll + 0.015
    ax_traj.set_xlim(x_left, x_right)
    # Invert the x-axis so chance NLL is on the LEFT (start) and lower
    # (better) NLL is on the RIGHT (end). Matches the right panel's
    # "further from chance = better" reading direction; the cost is
    # numeric x-tick labels decreasing left-to-right, which is the
    # expected convention for a "lower is better" metric.
    ax_traj.invert_xaxis()
    ax_traj.set_xlabel("Negative log-likelihood (held-out data)",
                       fontsize=9, color=TEXT_COLOR)
    ax_traj.spines['top'].set_visible(False)
    ax_traj.spines['right'].set_visible(False)
    ax_traj.tick_params(axis='both', colors=TEXT_COLOR, labelsize=8)
    for spine in ax_traj.spines.values():
        spine.set_edgecolor(TEXT_COLOR)

    # Right panel: 3 vertical bars on the held-out accuracy axis
    # Bar 1 (best univariate) is a single solid bar in that feature's
    # category colour. Bars 2 and 3 are stacked: each segment is the
    # marginal accuracy contribution of an accepted feature, coloured
    # by the same self / other / dyadic palette as the trajectory and
    # the filter grid. Bar 3 has the rejected step's best secondary-
    # metric candidate stacked grey on top. Note that per-step marginal
    # accuracy is not guaranteed monotonic (the selector optimises
    # NLL, not accuracy), so a small negative segment is possible and
    # is rendered honestly as a thin downward step.
    bar_width = 0.6
    bar_x_positions = [0, 1]
    bar_group_labels = ['best univariate', 'final model']

    cum_scores = [float(d['sec_mean']) for d in steps_data]
    score_marginals = []
    _prev_score = chance_secondary
    for _v in cum_scores:
        score_marginals.append(_v - _prev_score)
        _prev_score = _v

    # Bar 1: best univariate (single solid bar).
    bar1_color = (_category_color(best_univariate_feat)
                  if best_univariate_feat is not None else NEUTRAL_COLOR)
    ax_bars.bar(0, best_univariate_value - chance_secondary,
                bottom=chance_secondary, width=bar_width,
                color=bar1_color, edgecolor='none')

    # Bar 2: stacked accepted features.
    _bottom = chance_secondary
    for d, marginal in zip(steps_data, score_marginals):
        seg_color = _category_color(d['feature_name'])
        ax_bars.bar(1, marginal, bottom=_bottom, width=bar_width,
                    color=seg_color, edgecolor='none')
        _bottom += marginal

    # Y-axis: 0.5 floor; visible labels up to 0.95; ceiling extended
    # if necessary to give room for the feature-label stack above the
    # tallest bar.
    bar_tops = [best_univariate_value, final_score]
    y_data_max = max(bar_tops)
    label_line_spacing = 0.018
    label_y_start_offset = 0.010
    label_fontsize = 8
    max_label_lines = len(steps_data)
    label_stack_top = (y_data_max + label_y_start_offset
                       + (max_label_lines + 1) * label_line_spacing)
    y_top = max(0.97, label_stack_top)
    ax_bars.set_ylim(chance_secondary, y_top)
    visible_ticks = np.arange(0.5, 0.951, 0.05)
    ax_bars.set_yticks(visible_ticks)
    ax_bars.set_yticklabels([f"{t:.2f}" for t in visible_ticks],
                            fontsize=8, color=TEXT_COLOR)

    ax_bars.set_xticks(bar_x_positions)
    ax_bars.set_xticklabels(bar_group_labels, fontsize=7, color=TEXT_COLOR)
    ax_bars.set_xlim(-0.6, len(bar_group_labels) - 0.4)
    ax_bars.set_ylabel("Accuracy (held-out data)",
                       fontsize=9, color=TEXT_COLOR)

    final_feat_labels = [f"+{_pretty(d['feature_name'])}" for d in steps_data]

    # Bar 1 label stack: just the best-univariate feature name.
    if best_univariate_feat is not None:
        ax_bars.text(0, best_univariate_value + label_y_start_offset,
                     _pretty(best_univariate_feat),
                     ha='center', va='bottom',
                     fontsize=label_fontsize, color=TEXT_COLOR)

    # Bar 2 label stack: all accepted features in selection order
    # (bottom of stack -> first added, top -> last added).
    for j, lab in enumerate(final_feat_labels):
        ax_bars.text(1, final_score + label_y_start_offset + j * label_line_spacing,
                     lab, ha='center', va='bottom',
                     fontsize=label_fontsize, color=TEXT_COLOR)

    ax_bars.spines['top'].set_visible(False)
    ax_bars.spines['right'].set_visible(False)
    ax_bars.tick_params(axis='both', colors=TEXT_COLOR, labelsize=8)
    for spine in ax_bars.spines.values():
        spine.set_edgecolor(TEXT_COLOR)

    fig_traj.subplots_adjust(left=0.18, right=0.97, top=0.95,
                             bottom=0.12, wspace=0.12)

    if save_plots:
        if output_dir is None:
            _fallback = pathlib.Path(selection_results_path)
            _out_dir = _fallback.parent if _fallback.is_file() else _fallback
        else:
            _out_dir = pathlib.Path(output_dir)
        out_traj = _out_dir / "model_selection_trajectory.svg"
        fig_traj.savefig(out_traj, bbox_inches='tight', dpi=300,
                         facecolor=BG_COLOR, transparent=False)
        print(f"Saved trajectory figure to: {out_traj}")

    plt.show()

    # 4. Filter Grid Visualization
    # Always inspect the LAST step (even if it was a rejection) to try and
    # find filter shapes; fall back to the second-to-last step when the
    # rejection step lacks a fitted final model.
    final_data = selection_steps[-1]
    raw_filter_data = final_data.get('filter_shapes', None)

    if not raw_filter_data:
        print(f"DEBUG: 'filter_shapes' not found in last step (idx {len(selection_steps) - 1})")
        if len(selection_steps) > 1:
            prev_data = selection_steps[-2]
            if 'filter_shapes' in prev_data:
                print(f"DEBUG: Found 'filter_shapes' in previous step (idx {len(selection_steps) - 2})")
                raw_filter_data = prev_data['filter_shapes']

    if not raw_filter_data:
        print("Could not extract filter_shapes from final steps. Skipping filter grid.")
        return

    # Determine feature keys and structure type
    feature_keys = []
    is_cv_list = False

    if isinstance(raw_filter_data, list) and len(raw_filter_data) > 0:
        # Case: List of dictionaries (CV folds)
        is_cv_list = True
        first_fold = raw_filter_data[0]
        if isinstance(first_fold, dict):
            feature_keys = list(first_fold.keys())
    elif isinstance(raw_filter_data, dict):
        # Case: Single dictionary (Legacy or Single fit)
        is_cv_list = False
        feature_keys = list(raw_filter_data.keys())

    print(f"Found {len(feature_keys)} features in final model: {feature_keys}")

    n_feats = len(feature_keys)
    ncols = 4
    nrows = math.ceil(n_feats / ncols)

    fig_grid, axes_grid = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(ncols * 2.5, nrows * 2.0),
        dpi=300, constrained_layout=True
    )
    fig_grid.patch.set_facecolor(BG_COLOR)

    if nrows == 1 and ncols == 1: axes_grid = np.array([axes_grid])
    axes_grid = axes_grid.flatten()

    dyadic_keywords = ["nose-nose", "nose-TTI", "TTI-nose", "allo_yaw-nose",
                       "nose-allo_yaw", "allo_yaw-TTI", "TTI-allo_yaw",
                       "allo_pitch-nose", "nose-allo_pitch",
                       "allo_pitch-TTI", "TTI-allo_pitch"]

    for i, feature in enumerate(feature_keys):
        ax = axes_grid[i]
        ax.set_facecolor(BG_COLOR)

        try:
            mean_filter = None
            p_low = None
            p_high = None

            if is_cv_list:
                # 1. Stack all fold arrays for this feature
                feat_folds = [fold[feature] for fold in raw_filter_data if fold is not None and feature in fold]

                if feat_folds:
                    feat_matrix = np.vstack(feat_folds)  # Shape: (n_folds, n_timepoints)

                    if not np.all(np.isnan(feat_matrix)):
                        mean_filter = np.nanmean(feat_matrix, axis=0)
                        # Per-timepoint mean +/- SEM across CV folds.
                        # SEM = sample-std / sqrt(n_valid_folds);
                        # nan folds (failed fits) are excluded from
                        # both the count and the moments.
                        n_valid = np.sum(~np.isnan(feat_matrix), axis=0)
                        fold_std = np.nanstd(feat_matrix, axis=0, ddof=1)
                        fold_sem = fold_std / np.sqrt(np.maximum(n_valid, 1))
                        p_low = mean_filter - fold_sem
                        p_high = mean_filter + fold_sem

            else:
                # 2. Legacy/Single Fit
                feat_data = raw_filter_data[feature]
                if not np.all(np.isnan(feat_data)):
                    mean_filter = feat_data
                    p_low = None  # No shading for single fit

            if mean_filter is not None:
                filter_size_vector = np.arange(mean_filter.size)

                if any(x in feature for x in dyadic_keywords):
                    c = DYADIC_COLOR
                elif '-sei' in feature:
                    # SEI signals are target-attending-to-predictor; target is "self" → self_color.
                    c = self_color
                elif 'self' in feature:
                    c = self_color
                else:
                    c = other_color

                # Plot Shading first (so it is behind the line)
                if p_low is not None and p_high is not None:
                    ax.fill_between(filter_size_vector, p_low, p_high, color=c, alpha=0.3, edgecolor='none')

                # Plot Mean Line
                ax.plot(filter_size_vector, mean_filter, color=c, lw=1.5)

                tick_positions = np.arange(0, 650, 75)
                tick_labels = np.round(np.arange(-4, 0.5, 0.5), decimals=2)
                ax.set_xticks(tick_positions)
                ax.set_xticklabels(tick_labels, fontsize=7, color=TEXT_COLOR)
                ax.set_xlabel("Time relative to bout onset (s)", fontsize=8, color=TEXT_COLOR)
                ax.set_ylabel("Partial dependence (ΔP)", fontsize=8, color=TEXT_COLOR)

                ax.axhline(0, color=NEUTRAL_COLOR, ls='--', lw=0.5, zorder=0)
                ax.set_title(_pretty(feature), fontsize=9, color=TEXT_COLOR)
                ax.tick_params(axis='both', colors=TEXT_COLOR, labelsize=7)
                for spine in ax.spines.values():
                    spine.set_edgecolor(TEXT_COLOR)


        except Exception as e:
            print(f"Error plotting feature {feature}: {e}")

    for i in range(n_feats, len(axes_grid)):
        fig_grid.delaxes(axes_grid[i])

    if save_plots:
        if output_dir is None:
            # ``selection_results_path`` is polymorphic (file or
            # directory); fall back to the file's parent dir so the
            # saved figure lands alongside the artifact rather than
            # at the file path itself.
            _fallback = pathlib.Path(selection_results_path)
            output_dir = _fallback.parent if _fallback.is_file() else _fallback
        out_name = pathlib.Path(output_dir) / "model_selection_final_model_filters.svg"
        fig_grid.savefig(out_name, bbox_inches='tight', dpi=300, facecolor=BG_COLOR, transparent=False)
        print(f"Saved final model filter grid to: {out_name}")

    plt.show()


def plot_univariate_multinomial_performance(
        results_file_loc: str,
        evaluation_metric: str = 'll',
        evaluation_metric_name: str = 'Negative Log-Likelihood',
        secondary_metric: str = 'score',
        secondary_metric_name: str = 'Balanced Accuracy',
        p_val_threshold: float = 0.01,
        base_cmap: str = 'mako',
        diff_cmap: str = 'RdBu_r',
        save_plot: bool = False,
        output_dir: str = None
) -> None:
    """
    Evaluates and ranks feature importance while diagnosing categorical
    prediction errors across the USV repertoire.

    This function acts as the primary diagnostic suite for multinomial behavioral
    models, generating two distinct visualizations:

    1. The Feature Leaderboard (ranking plots):
       Evaluates the predictive power of all behavioral features.
       - Significance testing: A feature is considered significant only if its
         mean performance on the `evaluation_metric` outperforms the Null
         distribution at a Bonferroni-corrected alpha level.
       - Visualization: Displays the cross-validation variance by plotting the
         metric score for every fold, overlaid with the mean. Significant features
         are colored by their social identity (Male, Female, Dyadic).

    2. The Confusion Trio (categorical diagnosis):
       For the single most predictive significant feature, this generates a 1x3
       grid of row-normalized (recall-based) confusion matrices:
       - Actual: The true behavioral model's categorical predictions.
       - Null: The baseline predictions (driven purely by class imbalance).
       - Subtracted (Actual - Null): Isolates the specific "information gain"
         provided by the behavior, highlighting which specific USV category
         confusions were resolved by incorporating behavioral history.

    Parameters
    ----------
    results_file_loc : str
        Path to the stored .pkl results file containing fold-level metrics,
        predictions, and true labels.
    evaluation_metric : str, default 'll'
        The dictionary key for the primary metric used to sort features and
        determine statistical significance.
    evaluation_metric_name : str, default 'Negative Log-Likelihood'
        The human-readable label for the y-axis of the primary ranking plot.
    secondary_metric : str, default 'score'
        The dictionary key for the secondary metric (plotted alongside the primary).
    secondary_metric_name : str, default 'Balanced Accuracy'
        The human-readable label for the y-axis of the secondary ranking plot.
    p_val_threshold : float, default 0.01
        The family-wise error rate. Internally Bonferroni-corrected by the total
        number of tested features.
    base_cmap : str, default 'mako'
        The sequential colormap used for the Actual and Null proportion matrices.
    diff_cmap : str, default 'RdBu_r'
        The diverging colormap used for the Subtracted (Information Gain) matrix.
    save_plot : bool, default False
        If True, saves the generated SVG plots to disk.
    output_dir : str, optional
        Target directory for saved plots. Defaults to the same directory as the
        results file if None.
    """

    results_file_loc = configure_path(str(results_file_loc))
    if output_dir is not None:
        output_dir = configure_path(str(output_dir))

    # Local override: MEAN_LINE_COLOR differs from the module-level default
    # ('#DCB400') because the ranking plot reads better with a neutral grey.
    MEAN_LINE_COLOR = "#202020"

    results_path = pathlib.Path(results_file_loc)
    with open(results_path, 'rb') as f:
        modeling_data = pickle.load(f)

    if '_male_' in results_path.name:
        self_color, other_color, self_label, other_label = male_color, female_color, "Male (self)", "Female (other)"
    elif '_female_' in results_path.name:
        self_color, other_color, self_label, other_label = female_color, male_color, "Female (self)", "Male (other)"
    else:
        self_color, other_color, self_label, other_label = male_color, female_color, "Self", "Other"

    # Strip reserved metadata blocks before iterating feature entries.
    valid_features = [k for k in modeling_data.keys() if k not in RESERVED_METADATA_KEYS]
    n_features = len(valid_features)

    lower_is_better = ['ll', 'nll', 'log_loss', 'loss', 'mse', 'rmse']
    eval_is_lower_better = evaluation_metric in lower_is_better
    corrected_p_top = (1 - (p_val_threshold / n_features)) * 100
    corrected_p_bottom = (p_val_threshold / n_features) * 100

    feature_stats = []
    for feat in valid_features:
        try:
            actual_eval = np.array(modeling_data[feat]['actual']['folds']['metrics'][evaluation_metric])
            null_eval = np.array(modeling_data[feat]['null']['folds']['metrics'][evaluation_metric])
            actual_sec = np.array(modeling_data[feat]['actual']['folds']['metrics'][secondary_metric])
        except KeyError as e:
            print(f"Skipping {feat} - Missing metric data: {e}")
            continue

        actual_eval = actual_eval[~np.isnan(actual_eval)]
        null_eval = null_eval[~np.isnan(null_eval)]
        if len(actual_eval) == 0 or len(null_eval) == 0: continue

        actual_mean = np.mean(actual_eval)

        if eval_is_lower_better:
            thresh = np.percentile(null_eval, corrected_p_bottom)
            is_sig = actual_mean < thresh
        else:
            thresh = np.percentile(null_eval, corrected_p_top)
            is_sig = actual_mean > thresh

        if is_sig:
            if any(x in feat for x in ["nose", "TTI", "allo_yaw", "neck_elevation_diff"]):
                color = DYADIC_COLOR
            elif '-sei' in feat:
                # SEI signals are target-attending-to-predictor; target is "self" → self_color.
                color = self_color
            elif 'self' in feat:
                color = self_color
            else:
                color = other_color
        else:
            color = NEUTRAL_COLOR

        feature_stats.append({
            'name': feat,
            'actual_eval': actual_eval,
            'actual_sec': actual_sec,
            'is_sig': is_sig,
            'color': color,
            'mean_eval': actual_mean
        })

    if not feature_stats:
        print("No valid features remaining after parsing.")
        return

    # 1. Prepare the two different sort orders
    # Order for the Evaluation Metric (LL)
    primary_stats = sorted(feature_stats, key=lambda x: x['mean_eval'], reverse=not eval_is_lower_better)

    # Order for the Secondary Metric (Balanced Accuracy) - Always Higher is Better
    secondary_stats = sorted(feature_stats, key=lambda x: np.mean(x['actual_sec']), reverse=True)

    orders = [primary_stats, secondary_stats]

    # Ranking Plots
    fig_ranking, axes_ranking = plt.subplots(nrows=1, ncols=2, figsize=(14, 6), dpi=300)
    fig_ranking.patch.set_facecolor('#FFFFFF')
    configs = [(evaluation_metric_name, 'actual_eval'), (secondary_metric_name, 'actual_sec')]

    for i, (label, key) in enumerate(configs):
        ax = axes_ranking[i]
        ax.set_facecolor('#FFFFFF')
        ax.grid(False)

        current_order = orders[i]

        for idx, stat in enumerate(current_order):
            vals = stat[key]
            x_pos = np.random.normal(idx, 0.005, size=len(vals))
            ax.scatter(x_pos, vals, color=stat['color'], alpha=0.4, s=20, edgecolors='none')
            ax.hlines(np.mean(vals), idx - 0.3, idx + 0.3, color=MEAN_LINE_COLOR, lw=2)

        ax.set_xticks(range(len(current_order)))
        ax.set_xticklabels([s['name'] for s in current_order], rotation=45, ha='right', fontsize=8, color=TEXT_COLOR)
        ax.set_ylabel(f"{label} (Held-out data)", color=TEXT_COLOR)
        ax.tick_params(axis='both', colors=TEXT_COLOR)

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(TEXT_COLOR)

        if 'score' in key.lower() or 'accuracy' in label.lower():
            n_classes = len(modeling_data[valid_features[0]]['actual']['classes'])
            ax.axhline(1 / n_classes, ls='--', color=REFERENCE_LINE_COLOR, alpha=0.5, label='Chance', zorder=0)

    # Define Legend Elements (Fixes NameError)
    legend_elements = [
        Patch(facecolor=self_color, alpha=0.7, label=self_label),
        Patch(facecolor=other_color, alpha=0.7, label=other_label),
        Patch(facecolor=DYADIC_COLOR, alpha=0.7, label='Social features'),
        Patch(facecolor=NEUTRAL_COLOR, alpha=0.7, label='Not significant'),
        Line2D([0], [0], color=MEAN_LINE_COLOR, lw=2, label='Mean Score')
    ]

    leg = axes_ranking[1].legend(
        handles=legend_elements,
        loc='upper left',
        bbox_to_anchor=(1.05, 1),
        frameon=False,
        labelcolor=TEXT_COLOR,
        fontsize=9
    )

    plt.tight_layout(rect=[0, 0, 0.9, 1])

    if save_plot:
        out_dir = pathlib.Path(output_dir) if output_dir else results_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        ranking_out = out_dir / f"{results_path.stem}_{evaluation_metric}_ranking.svg"
        fig_ranking.savefig(
            ranking_out,
            bbox_inches='tight',
            bbox_extra_artists=(leg,),
            facecolor='#FFFFFF'
        )
        print(f"Ranking plot saved to: {ranking_out.name}")

    plt.show()

    # Confusion Trio Plot
    top_sig_feat = next((s for s in primary_stats if s['is_sig']), None)

    if top_sig_feat:
        print(f"\nGenerating Confusion Trio for top significant feature: {top_sig_feat['name']}")
        fig_trio, axes_trio = plt.subplots(1, 3, figsize=(18, 5), dpi=300)
        fig_trio.patch.set_facecolor('#FFFFFF')
        class_names = modeling_data[top_sig_feat['name']]['actual']['classes']

        mats = []
        for strategy in ['actual', 'null']:
            y_t = np.concatenate(modeling_data[top_sig_feat['name']][strategy]['folds']['y_true'])
            y_p = np.concatenate(modeling_data[top_sig_feat['name']][strategy]['folds']['y_pred'])
            # Pass the canonical class order so both the 'actual' and 'null'
            # matrices are K x K with identical row/column ordering: the null
            # model often predicts only the majority class, so without labels=
            # sklearn would infer a smaller label set for null_cm and the
            # actual_cm - null_cm subtraction below would broadcast-error or
            # silently subtract mismatched categories.
            mats.append(confusion_matrix(y_t, y_p, labels=class_names, normalize='true'))

        actual_cm, null_cm = mats[0], mats[1]
        diff_cm = actual_cm - null_cm

        data_list = [actual_cm, null_cm, diff_cm]
        titles = ["Actual Model (Row-Norm)", "Null Baseline (Row-Norm)", "Information Gain (Actual - Null)"]
        cmaps = [base_cmap, base_cmap, diff_cmap]

        for j in range(3):
            ax = axes_trio[j]

            if j == 2:
                v_limit = np.max(np.abs(diff_cm))
                vmin, vmax = -v_limit, v_limit
                cbar_bool = True
            else:
                vmin, vmax = 0, 1
                cbar_bool = (j == 1)

            sns.heatmap(data_list[j], ax=ax, annot=True, fmt=".3f", cmap=cmaps[j],
                        xticklabels=class_names, yticklabels=class_names,
                        vmin=vmin, vmax=vmax,
                        cbar=cbar_bool, cbar_kws={'shrink': 0.8} if cbar_bool else None)

            if cbar_bool:
                cbar = ax.collections[0].colorbar
                if j == 1:
                    cbar.set_ticks([0, 1])
                    cbar.set_ticklabels(["0%", "100%"])
                elif j == 2:
                    cbar.set_ticks([-v_limit, 0, v_limit])
                    cbar.set_ticklabels(["Worse", "0", "Better"])

            ax.set_title(titles[j], color=TEXT_COLOR, pad=10)
            ax.set_xlabel('Predicted USV Category', color=TEXT_COLOR)
            if j == 0: ax.set_ylabel('True USV Category', color=TEXT_COLOR)
            ax.tick_params(axis='both', colors=TEXT_COLOR)

        plt.suptitle(f"Categorical Diagnosis: {top_sig_feat['name']}", fontsize=16, y=1.05, color=TEXT_COLOR)

        if save_plot:
            trio_out = out_dir / f"{results_path.stem}_{top_sig_feat['name'].replace('.', '_')}_confusion_trio.svg"
            fig_trio.savefig(trio_out, bbox_inches='tight', facecolor='#FFFFFF')
            print(f"Confusion Trio plot saved to: {trio_out.name}")

        plt.show()
    else:
        print("\nNo significant features found. Skipping Confusion Trio.")


def plot_univariate_multinomial_filters_grid(
        results_file_loc: str,
        evaluation_metric: str = 'll',
        p_val_threshold: float = 0.01,
        history_window_sec: float = 4.0,
        cmap: str = 'RdBu_r',
        save_plot: bool = False,
        output_dir: str = None
) -> None:
    """
    Plots the temporal behavioral filters for all significant features in a
    small-multiples grid of heatmaps.

    This function isolates features that significantly predict USV categories
    and visualizes their learned multinomial weight matrices.

    Visualization logic:
    --------------------
    - Each subplot is a single behavioral feature.
    - Rows in the heatmap represent the USV categories.
    - The X-axis represents the time history leading up to the vocalization.
    - Colors represent the impact of the behavior on the log-odds of the USV:
      * Red (Positive): The behavior promotes the USV category.
      * Blue (Negative): The behavior suppresses the USV category.
    - To prevent washout and avoid clutter, each subplot is symmetrically scaled
      to its own maximum absolute amplitude, and colorbars are omitted in favor
      of a text annotation in the title.

    Parameters
    ----------
    results_file_loc : str
        Path to the stored .pkl results file.
    evaluation_metric : str, default 'll'
        The metric used to determine statistical significance (e.g., 'll' or 'score').
    p_val_threshold : float, default 0.01
        The family-wise error rate, internally Bonferroni-corrected.
    history_window_sec : float, default 4.0
        The total length of the behavioral history window in seconds. Used to
        correctly format the X-axis time labels.
    cmap : str, default 'RdBu_r'
        The diverging colormap used to represent negative/positive filter weights.
    save_plot : bool, default False
        If True, saves the figure as an SVG.
    output_dir : str, optional
        Target directory for saved plots.
    """

    results_file_loc = configure_path(str(results_file_loc))
    if output_dir is not None:
        output_dir = configure_path(str(output_dir))

    results_path = pathlib.Path(results_file_loc)
    with open(results_path, 'rb') as f:
        modeling_data = pickle.load(f)

    if '_male_' in results_path.name:
        self_color, other_color = male_color, female_color
    elif '_female_' in results_path.name:
        self_color, other_color = female_color, male_color
    else:
        self_color, other_color = male_color, female_color

    # Strip reserved metadata blocks before iterating feature entries.
    valid_features = [k for k in modeling_data.keys() if k not in RESERVED_METADATA_KEYS]
    n_features = len(valid_features)

    lower_is_better = ['ll', 'nll', 'log_loss', 'loss', 'mse', 'rmse']
    eval_is_lower_better = evaluation_metric in lower_is_better
    corrected_p_top = (1 - (p_val_threshold / n_features)) * 100
    corrected_p_bottom = (p_val_threshold / n_features) * 100

    plot_data = []

    for feat in valid_features:
        try:
            actual_eval = np.array(modeling_data[feat]['actual']['folds']['metrics'][evaluation_metric])
            null_eval = np.array(modeling_data[feat]['null']['folds']['metrics'][evaluation_metric])
        except KeyError:
            continue

        actual_eval = actual_eval[~np.isnan(actual_eval)]
        null_eval = null_eval[~np.isnan(null_eval)]
        if len(actual_eval) == 0 or len(null_eval) == 0: continue

        actual_mean = np.mean(actual_eval)

        if eval_is_lower_better:
            is_sig = actual_mean < np.percentile(null_eval, corrected_p_bottom)
        else:
            is_sig = actual_mean > np.percentile(null_eval, corrected_p_top)

        if not is_sig:
            continue

        if any(x in feat for x in ["nose", "TTI", "allo_yaw", "neck_elevation_diff"]):
            feat_color = DYADIC_COLOR
        elif '-sei' in feat:
            # SEI signals are target-attending-to-predictor; target is "self" → self_color.
            feat_color = self_color
        elif 'self' in feat:
            feat_color = self_color
        else:
            feat_color = other_color

        # Average the weights across folds
        # shape of folds['weights']: (n_folds, n_classes, n_time_bins)
        raw_weights = np.array(modeling_data[feat]['actual']['folds']['weights'])
        mean_weights = np.mean(raw_weights, axis=0)

        class_names = modeling_data[feat]['actual']['classes']

        plot_data.append({
            'name': feat,
            'mean_eval': actual_mean,
            'weights': mean_weights,
            'classes': class_names,
            'color': feat_color
        })

    if not plot_data:
        print(f"No significant features found for metric '{evaluation_metric}'. Exiting.")
        return

    # sort by metric performance (the best comes first)
    plot_data.sort(key=lambda x: x['mean_eval'], reverse=not eval_is_lower_better)
    n_significant = len(plot_data)
    print(f"Found {n_significant} significant features. Plotting grid...")

    ncols = 4
    nrows = math.ceil(n_significant / ncols)

    n_classes = len(plot_data[0]['classes'])
    fig_height = max(4.0, nrows * (n_classes * 0.4))

    fig, axes_grid = plt.subplots(nrows=nrows, ncols=ncols,
                                  figsize=(ncols * 3.5, fig_height),
                                  dpi=300, sharex=True, sharey=True)
    fig.patch.set_facecolor('#FFFFFF')

    if nrows == 1 and ncols == 1: axes_grid = np.array([axes_grid])
    axes_grid = axes_grid.flatten()

    for i, data in enumerate(plot_data):
        ax = axes_grid[i]
        weights = data['weights']

        max_amp = np.max(np.abs(weights))

        ax.imshow(weights, aspect='auto', cmap=cmap, vmin=-max_amp, vmax=max_amp, interpolation='nearest')

        n_time_bins = weights.shape[1]
        tick_times = np.arange(-int(history_window_sec), 1)
        tick_locs = [(t + history_window_sec) / history_window_sec * (n_time_bins - 1) for t in tick_times]

        ax.set_xticks(tick_locs)
        ax.set_xticklabels([f"{t}s" for t in tick_times], fontsize=8, color=TEXT_COLOR)

        ax.set_yticks(np.arange(len(data['classes'])))
        if i % ncols == 0:
            ax.set_yticklabels(data['classes'], fontsize=8, color=TEXT_COLOR)
        else:
            ax.set_yticklabels([])

        ax.tick_params(axis='both', colors=TEXT_COLOR, length=3)

        title_text = f"{data['name']}\n{evaluation_metric.upper()}: {data['mean_eval']:.3f} | Max Amp: ±{max_amp:.2f}"
        ax.set_title(title_text, fontsize=9, color=data['color'], fontweight='bold', pad=8)

        for spine in ax.spines.values():
            spine.set_edgecolor(data['color'])
            spine.set_linewidth(1.5)

    for i in range(n_significant, len(axes_grid)):
        fig.delaxes(axes_grid[i])

    for i in range(n_significant):
        row = i // ncols
        col = i % ncols
        idx_below = (row + 1) * ncols + col
        is_bottom = (row == nrows - 1) or (idx_below >= n_significant)
        if is_bottom:
            axes_grid[i].tick_params(labelbottom=True)

    plt.tight_layout()

    if save_plot:
        out_dir = pathlib.Path(output_dir) if output_dir else results_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        out_name = out_dir / f"{results_path.stem}_multinomial_filters_grid.svg"
        fig.savefig(out_name, bbox_inches='tight', facecolor='#FFFFFF')
        print(f"Filter grid saved to: {out_name.name}")

    plt.show()


def plot_multinomial_selection_trajectory(
        selection_results_path: str,
        metric_primary: str = 'auc',
        primary_metric_name: str = "Area Under the ROC Curve",
        metric_secondary: str = 'score',
        secondary_metric_name: str = "Balanced Accuracy",
        save_plot: bool = False,
        output_dir: str = None,
        feature_label_overrides: dict = None,
        secondary_ylim_max: float = None,
) -> None:
    """
    Plot the multinomial forward-selection trajectory as a compact
    two-panel summary, mirroring the design of
    ``plot_model_selection_results`` (the binary bout-onset analogue).

    Layout
    ------
    * **Left panel** -- one horizontal bar per accepted step, top-to-
      bottom in selection order. Each bar is split into a lighter
      base (the previous cumulative value of ``metric_primary``) and
      a darker tip (THIS step's marginal contribution). Bars are
      coloured by the same self / other / dyadic palette used in the
      filter grid. A rejected final step, if any, is drawn below a
      thin separator in grey. For maximisation metrics (AUC, score,
      etc.) bars grow rightward from the chance baseline on the left;
      for minimisation metrics (log-loss, brier) the x-axis is
      inverted and bars grow rightward from chance on the right.
    * **Right panel** -- two vertical bars on the secondary-metric
      axis: best univariate (single solid bar in the winning
      feature's category colour) and final accepted model (stacked
      bar, one segment per accepted feature, segment height = that
      feature's marginal contribution to the secondary metric).
      Feature labels stack above each bar. The y-axis floor is the
      chance baseline computed for the number of USV categories in
      the pickle (``1 / K`` for balanced accuracy / recall, ``0.5``
      for macro-AUC).

    Robustness
    ----------
    All per-fold means use ``np.nanmean`` so the silent NaN-fold
    failure mode in the multinomial selector (now patched at the
    source, but still present in pre-existing pickles) does not
    poison the trajectory.

    Parameters
    ----------
    selection_results_path : str
        Path to the consolidated ``selection_*.pkl`` artifact
        produced by ``consolidate_model_selection_results``. May be
        either the file itself or a directory containing one
        (latest mtime wins).
    metric_primary : str, default ``'auc'``
        Key for the primary per-fold metric. Used for the left-panel
        trajectory.
    primary_metric_name : str, default ``'Area Under the ROC Curve'``
        Display name for ``metric_primary``; used as the left-panel
        x-axis label (with `` (held-out data)`` appended).
    metric_secondary : str, default ``'score'``
        Key for the secondary per-fold metric. Used for the right-
        panel bars.
    secondary_metric_name : str, default ``'Balanced Accuracy'``
        Display name for ``metric_secondary``; used as the right-
        panel y-axis label (with `` (held-out data)`` appended).
    save_plot : bool, default False
        Whether to save the figure to disk.
    output_dir : str, optional
        Directory to save the plot. Defaults to the parent dir of
        ``selection_results_path`` (or to the path itself if a
        directory was supplied).
    feature_label_overrides : dict, optional
        Mapping from raw feature names (as stored in the pickle) to
        presentation-friendly labels used for every annotation. Raw
        names not in the map render unchanged.
    secondary_ylim_max : float, optional
        Hard upper bound for the right-panel y-axis (the secondary-
        metric bars). When supplied, overrides the default rate-metric
        cap (0.31) used for AUC / balanced-accuracy / recall / etc.
        The visible tick range stops at this value; the label-stack
        above each bar still extends past it if needed so feature
        annotations are not clipped.
    """

    selection_results_path = configure_path(str(selection_results_path))
    if output_dir is not None:
        output_dir = configure_path(str(output_dir))

    BG_COLOR = '#FFFFFF'

    selection_steps, display_name, selection_metadata = load_selection_results(selection_results_path)

    if not selection_steps:
        print(f"No multinomial step data found in {selection_results_path}")
        return

    # Sex-aware self/other palette inferred from the filename, matching
    # the binary selector's convention.
    if '_male_' in display_name:
        self_col, other_col = male_color, female_color
    elif '_female_' in display_name:
        self_col, other_col = female_color, male_color
    else:
        self_col, other_col = male_color, female_color

    lower_is_better = {'ll', 'nll', 'log_loss', 'loss', 'mse', 'rmse', 'brier'}
    is_minimization = metric_primary in lower_is_better

    # Number of USV categories in this run -- pulled from any candidate
    # that has a stored ``classes`` array. Drives the secondary-metric
    # chance baseline (1/K for balanced accuracy etc.).
    n_classes = None
    for _s in selection_steps:
        _cs = _s['candidates_summary']
        for _c in _cs.values():
            _cl = _c.get('classes') if isinstance(_c, dict) else None
            if _cl is not None and len(_cl) > 0:
                n_classes = len(_cl)
                break
        if n_classes is not None:
            break
    if n_classes is None or n_classes < 2:
        n_classes = 2

    # Chance baselines.
    rate_metrics = {'score', 'recall', 'accuracy', 'f1', 'mcc'}
    if metric_primary == 'auc':
        chance_primary = 0.5
    elif metric_primary in rate_metrics:
        chance_primary = 1.0 / n_classes
    elif metric_primary in {'ll', 'log_loss', 'nll'}:
        chance_primary = float(np.log(n_classes))
    else:
        chance_primary = 0.0

    if metric_secondary == 'auc':
        chance_secondary = 0.5
    elif metric_secondary in rate_metrics:
        chance_secondary = 1.0 / n_classes
    elif metric_secondary in {'ll', 'log_loss', 'nll'}:
        chance_secondary = float(np.log(n_classes))
    else:
        chance_secondary = 0.0

    # Accepted steps = those with a real selected_feature that is not
    # the multinomial selector's null baseline marker. Order preserved.
    accepted_steps = [
        s for s in selection_steps
        if s['selected_feature'] not in (None, 'null_model_free')
    ]
    if not accepted_steps:
        print(f"No accepted feature steps in {selection_results_path}")
        return

    def _fold_mean(folds_metrics: dict, key: str) -> float:
        if key not in folds_metrics:
            return float('nan')
        arr = np.array(folds_metrics[key], dtype=float)
        arr = arr[np.isfinite(arr)]
        return float(np.mean(arr)) if arr.size else float('nan')

    steps_data = []
    for s in accepted_steps:
        winner = s['selected_feature']
        wd = s['candidates_summary'][winner]
        fm = wd['folds']['metrics']
        sec_vals = np.array(fm.get(metric_secondary, []), dtype=float)
        sec_vals = sec_vals[np.isfinite(sec_vals)]
        steps_data.append({
            'feature_name': winner,
            'prim_mean': _fold_mean(fm, metric_primary),
            'sec_mean': _fold_mean(fm, metric_secondary),
            'sec_vals': sec_vals,
        })

    # First accepted step is the multinomial anchor selection: every
    # candidate evaluated here is a single-feature model, so its
    # candidates_summary is the right place to pull best-univariate
    # numbers from.
    anchor_step = accepted_steps[0]
    best_univariate_value = float('nan')
    best_univariate_feat = None
    for f, cdata in anchor_step['candidates_summary'].items():
        fm = cdata['folds']['metrics']
        v = np.array(fm.get(metric_secondary, []), dtype=float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        mu = float(np.mean(v))
        if np.isnan(best_univariate_value) or mu > best_univariate_value:
            best_univariate_value = mu
            best_univariate_feat = f
    if np.isnan(best_univariate_value) and steps_data:
        # Fall back to the anchor winner's score if the candidate scan
        # produced no finite means (extreme NaN-fold case).
        best_univariate_value = steps_data[0]['sec_mean']
        best_univariate_feat = steps_data[0]['feature_name']

    final_score = float(steps_data[-1]['sec_mean'])

    # Rejected final step lookup (by secondary metric, like binary case).
    rejection_row = None
    if selection_steps[-1]['selected_feature'] is None:
        rej_step = selection_steps[-1]
        rej_cs = rej_step['candidates_summary']
        if rej_cs:
            best_rej_feat = None
            best_rej_sec = -np.inf
            best_rej_pri = float('nan')
            for f, cdata in rej_cs.items():
                fm = cdata['folds']['metrics']
                v = np.array(fm.get(metric_secondary, []), dtype=float)
                v = v[np.isfinite(v)]
                if v.size == 0:
                    continue
                mu = float(np.mean(v))
                if mu > best_rej_sec:
                    best_rej_sec = mu
                    best_rej_feat = f
                    pv = np.array(fm.get(metric_primary, []), dtype=float)
                    pv = pv[np.isfinite(pv)]
                    best_rej_pri = float(np.mean(pv)) if pv.size else float('nan')
            if best_rej_feat is not None:
                rejection_row = {
                    'feature_name': best_rej_feat,
                    'sec_mean': best_rej_sec,
                    'prim_mean': best_rej_pri,
                }

    # Feature-category colour map (mirrors the bout-onset selector
    # plotter so the same feature always gets the same colour across
    # figures).
    dyadic_keywords = ["nose-nose", "nose-TTI", "TTI-nose", "allo_yaw-nose",
                       "nose-allo_yaw", "allo_yaw-TTI", "TTI-allo_yaw",
                       "allo_pitch-nose", "nose-allo_pitch",
                       "allo_pitch-TTI", "TTI-allo_pitch"]

    def _category_color(fname: str) -> str:
        if any(x in fname for x in dyadic_keywords):
            return DYADIC_COLOR
        if '-sei' in fname:
            return self_col
        if 'self' in fname:
            return self_col
        return other_col

    def _lighten(hex_color: str, factor: float = 0.65) -> str:
        h = hex_color.lstrip('#')
        r = int(h[0:2], 16)
        g = int(h[2:4], 16)
        b = int(h[4:6], 16)
        r = int(round(r + (255 - r) * factor))
        g = int(round(g + (255 - g) * factor))
        b = int(round(b + (255 - b) * factor))
        return f"#{r:02x}{g:02x}{b:02x}"

    _pretty = _make_feature_pretty(feature_label_overrides, selection_metadata)

    cum_prim = [d['prim_mean'] for d in steps_data]
    cum_sec = [d['sec_mean'] for d in steps_data]
    sec_marginals = [cum_sec[0] - chance_secondary]
    for i in range(1, len(cum_sec)):
        sec_marginals.append(cum_sec[i] - cum_sec[i - 1])

    # Figure layout
    n_rows_total = len(steps_data) + (1 if rejection_row is not None else 0)
    fig_height = max(3.0, 0.32 * n_rows_total + 1.8)
    fig_traj, (ax_traj, ax_bars) = plt.subplots(
        nrows=1, ncols=2,
        figsize=(10.5, fig_height), dpi=300,
        gridspec_kw={'width_ratios': [2.2, 1.0]},
    )
    fig_traj.patch.set_facecolor(BG_COLOR)
    ax_traj.set_facecolor(BG_COLOR)
    ax_bars.set_facecolor(BG_COLOR)

    bar_height = 0.88
    y_positions = list(range(len(steps_data)))
    rej_y = len(steps_data) + 0.4 if rejection_row is not None else None

    for row_idx, d in enumerate(steps_data):
        y = y_positions[row_idx]
        base_color = _category_color(d['feature_name'])
        light_color = _lighten(base_color, factor=0.65)
        prev_p = cum_prim[row_idx - 1] if row_idx > 0 else chance_primary
        cur_p = cum_prim[row_idx]

        if is_minimization:
            # Bars grow leftward (lower = better). Light = prev->chance,
            # dark tip = cur->prev. x-axis inverted below.
            if chance_primary > prev_p:
                ax_traj.barh(y, chance_primary - prev_p, left=prev_p,
                             height=bar_height, color=light_color,
                             edgecolor='none')
            if prev_p > cur_p:
                ax_traj.barh(y, prev_p - cur_p, left=cur_p,
                             height=bar_height, color=base_color,
                             edgecolor='none')
            delta = prev_p - cur_p
            ax_traj.text(cur_p - 0.003 * (chance_primary - cur_p + 1e-9), y,
                         f"{cur_p:.3f}  (Δ -{delta:.3f})",
                         ha='left', va='center', fontsize=7,
                         color=TEXT_COLOR)
        else:
            # Bars grow rightward (higher = better). Light = chance->prev,
            # dark tip = prev->cur.
            if prev_p > chance_primary:
                ax_traj.barh(y, prev_p - chance_primary, left=chance_primary,
                             height=bar_height, color=light_color,
                             edgecolor='none')
            if cur_p > prev_p:
                ax_traj.barh(y, cur_p - prev_p, left=prev_p,
                             height=bar_height, color=base_color,
                             edgecolor='none')
            delta = cur_p - prev_p
            ax_traj.text(cur_p + 0.003, y,
                         f"{cur_p:.3f}  (Δ +{delta:.3f})",
                         ha='left', va='center', fontsize=7,
                         color=TEXT_COLOR)

    if rejection_row is not None and not np.isnan(rejection_row['prim_mean']):
        rejected_light = '#D7D7D7'
        rejected_dark = '#9A9A9A'
        prev_p = cum_prim[-1]
        cur_p = rejection_row['prim_mean']
        if is_minimization:
            if chance_primary > prev_p:
                ax_traj.barh(rej_y, chance_primary - prev_p, left=prev_p,
                             height=bar_height, color=rejected_light,
                             edgecolor='none')
            if prev_p > cur_p:
                ax_traj.barh(rej_y, prev_p - cur_p, left=cur_p,
                             height=bar_height, color=rejected_dark,
                             edgecolor='none')
            ax_traj.text(cur_p - 0.003, rej_y,
                         f"{cur_p:.3f}  (Δ -{prev_p - cur_p:.3f}, ns)",
                         ha='left', va='center', fontsize=7,
                         color=NEUTRAL_COLOR, style='italic')
        else:
            if prev_p > chance_primary:
                ax_traj.barh(rej_y, prev_p - chance_primary, left=chance_primary,
                             height=bar_height, color=rejected_light,
                             edgecolor='none')
            if cur_p > prev_p:
                ax_traj.barh(rej_y, cur_p - prev_p, left=prev_p,
                             height=bar_height, color=rejected_dark,
                             edgecolor='none')
            ax_traj.text(cur_p + 0.003, rej_y,
                         f"{cur_p:.3f}  (Δ +{cur_p - prev_p:.3f}, ns)",
                         ha='left', va='center', fontsize=7,
                         color=NEUTRAL_COLOR, style='italic')

    ytick_positions = list(y_positions)
    ytick_labels = [f"+{_pretty(d['feature_name'])}" for d in steps_data]
    if rejection_row is not None:
        ytick_positions.append(rej_y)
        ytick_labels.append(f"+{_pretty(rejection_row['feature_name'])}")
    ax_traj.set_yticks(ytick_positions)
    ax_traj.set_yticklabels(ytick_labels, fontsize=10, color=TEXT_COLOR)
    ax_traj.tick_params(axis='y', length=0)
    ax_traj.invert_yaxis()

    if rejection_row is not None:
        sep_y = (len(steps_data) - 1) + 0.5 + 0.10
        ax_traj.axhline(sep_y, color=NEUTRAL_COLOR, linestyle='-',
                        lw=0.4, alpha=0.5, zorder=0)

    # X-axis range / orientation depending on direction.
    all_endpoints = list(cum_prim)
    if rejection_row is not None and not np.isnan(rejection_row['prim_mean']):
        all_endpoints.append(rejection_row['prim_mean'])
    if is_minimization:
        span = chance_primary - min(all_endpoints)
        span = span if span > 1e-9 else 1.0
        x_left_lim = min(all_endpoints) - 0.30 * span
        x_right_lim = chance_primary + 0.015 * abs(chance_primary if chance_primary > 0 else 1)
        ax_traj.set_xlim(x_left_lim, x_right_lim)
        ax_traj.invert_xaxis()
    else:
        span = max(all_endpoints) - chance_primary
        span = span if span > 1e-9 else 1.0
        x_left_lim = chance_primary - 0.015
        x_right_lim = max(all_endpoints) + 0.30 * span
        ax_traj.set_xlim(x_left_lim, x_right_lim)

    ax_traj.set_xlabel(f"{primary_metric_name} (held-out data)",
                       fontsize=10, color=TEXT_COLOR)
    ax_traj.spines['top'].set_visible(False)
    ax_traj.spines['right'].set_visible(False)
    ax_traj.tick_params(axis='x', colors=TEXT_COLOR)
    for spine in ax_traj.spines.values():
        spine.set_edgecolor(TEXT_COLOR)

    # Right panel: 2 vertical bars (best univariate, final)
    bar_width = 0.6
    bar_x_positions = [0, 1]
    bar_group_labels = ['best univariate', 'final model']

    bar1_color = (_category_color(best_univariate_feat)
                  if best_univariate_feat is not None else NEUTRAL_COLOR)
    ax_bars.bar(0, best_univariate_value - chance_secondary,
                bottom=chance_secondary, width=bar_width,
                color=bar1_color, edgecolor='none')

    _bottom = chance_secondary
    for d, marginal in zip(steps_data, sec_marginals):
        seg_color = _category_color(d['feature_name'])
        ax_bars.bar(1, marginal, bottom=_bottom, width=bar_width,
                    color=seg_color, edgecolor='none')
        _bottom += marginal

    bar_tops = [best_univariate_value, final_score]
    y_data_max = float(np.nanmax(bar_tops))
    label_line_spacing = 0.006
    label_y_start_offset = 0.005
    label_fontsize = 8
    max_label_lines = len(steps_data)
    label_stack_top = (y_data_max + label_y_start_offset
                       + (max_label_lines + 1) * label_line_spacing)

    # Cap visible y-axis at 0.4 -- multinomial balanced accuracy with
    # K=6 classes tops out around 0.3 in practice, so a higher cap
    # leaves the bars looking stubby. y_top extends past the cap when
    # the feature-label stack above the bars would otherwise clip;
    # the visible tick range still stops at the cap so the axis reads
    # cleanly. First tick is rounded up to the nearest 0.05 above
    # the chance floor (e.g., 0.20 when chance = 1/6 ~= 0.167).
    # ``secondary_ylim_max`` overrides the default rate-metric cap
    # when the caller wants tighter framing (e.g., 0.26 for K=7).
    if metric_secondary in rate_metrics or metric_secondary == 'auc':
        cap = float(secondary_ylim_max) if secondary_ylim_max is not None else 0.31
        y_top = max(cap, label_stack_top + 0.01)
        first_tick = float(np.ceil(chance_secondary * 20) / 20)
        visible_ticks = np.arange(first_tick, cap + 1e-9, 0.05)
        if visible_ticks.size == 0:
            visible_ticks = np.linspace(first_tick, cap, 3)
        ax_bars.set_ylim(chance_secondary, y_top)
        ax_bars.set_yticks(visible_ticks)
        ax_bars.set_yticklabels([f"{t:.2f}" for t in visible_ticks],
                                fontsize=8, color=TEXT_COLOR)
    else:
        if secondary_ylim_max is not None:
            y_top = max(float(secondary_ylim_max), label_stack_top + 0.01)
        else:
            y_top = max(y_data_max + 0.10, label_stack_top)
        ax_bars.set_ylim(chance_secondary, y_top)

    ax_bars.set_xticks(bar_x_positions)
    ax_bars.set_xticklabels(bar_group_labels, fontsize=7, color=TEXT_COLOR)
    ax_bars.set_xlim(-0.6, len(bar_group_labels) - 0.4)
    ax_bars.set_ylabel(f"{secondary_metric_name} (held-out data)",
                       fontsize=10, color=TEXT_COLOR)

    final_feat_labels = [f"+{_pretty(d['feature_name'])}" for d in steps_data]

    if best_univariate_feat is not None:
        ax_bars.text(0, best_univariate_value + label_y_start_offset,
                     _pretty(best_univariate_feat),
                     ha='center', va='bottom',
                     fontsize=label_fontsize, color=TEXT_COLOR)

    for j, lab in enumerate(final_feat_labels):
        ax_bars.text(1, final_score + label_y_start_offset
                     + j * label_line_spacing,
                     lab, ha='center', va='bottom',
                     fontsize=label_fontsize, color=TEXT_COLOR)

    ax_bars.spines['top'].set_visible(False)
    ax_bars.spines['right'].set_visible(False)
    ax_bars.tick_params(axis='both', colors=TEXT_COLOR)
    for spine in ax_bars.spines.values():
        spine.set_edgecolor(TEXT_COLOR)

    fig_traj.subplots_adjust(left=0.18, right=0.97, top=0.95,
                             bottom=0.12, wspace=0.12)

    if save_plot:
        if output_dir is None:
            _fallback = pathlib.Path(selection_results_path)
            _out_dir = _fallback.parent if _fallback.is_file() else _fallback
        else:
            _out_dir = pathlib.Path(output_dir)
        _out_dir.mkdir(parents=True, exist_ok=True)
        path_str = str(selection_results_path).lower()
        if 'male_mute_partner' in path_str:
            condition = 'male_mute_partner'
        elif 'female' in path_str:
            condition = 'female'
        elif 'male' in path_str:
            condition = 'male'
        else:
            condition = 'unknown'
        fname = (f"multinomial_selection_trajectory_{condition}_"
                 f"{metric_primary}.svg")
        save_path = _out_dir / fname
        fig_traj.savefig(save_path, bbox_inches='tight', dpi=300,
                         facecolor=BG_COLOR, transparent=False)
        print(f"Trajectory plot saved to: {save_path}")

    plt.show()


def plot_multinomial_multivariate_filters(
        selection_results_path: str,
        history_window_sec: float = 4.0,
        cmap: str = 'RdBu_r',
        save_plot: bool = False,
        output_dir: str = None
) -> None:
    """
    Visualizes the converged multivariate multinomial model as a comprehensive
    atlas of temporal-categorical behavioral influence.

    Purpose and Statistical Framework:
    ---------------------------------
    The primary goal of this visualization is to move beyond 'importance ranking'
    and into 'mechanistic interpretation.' In a multinomial logistic framework,
    the model learns a weight matrix W for each behavioral feature that maps
    behavioral history to the log-odds of a specific USV category.

    Unlike univariate filters, which capture the total correlation between a
    behavior and a vocalization, these multivariate filters represent 'partial
    effects.' They isolate the unique contribution of each feature *after* accounting
    for the predictive variance captured by all other features in
    the optimal set. This is critical for disambiguating co-linear behaviors
    (e.g., distinguishing whether 'self-speed' or 'inter-animal distance' is
    the primary driver of a specific vocal transition).

    Biophysical Interpretation:
    --------------------------
    1. Feature-Category Mapping: Each subplot represents a behavior. Within
       the subplot, a row represents a USV category. If a row is predominantly
       red (positive) at a specific time lag, that behavioral state 'promotes'
       the likelihood of that USV category relative to the baseline.
    2. Temporal Dynamics: The X-axis (time) reveals the 'integration window.'
       Some behaviors may influence vocalizations instantaneously (near 0s),
       while others may reflect a slower 'behavioral state' (e.g., high speed
       sustained over 4 seconds) that sets the stage for specific vocal outputs.
    3. Categorical Contrast: By sharing the Y-axis (USV categories) across
       subplots, the user can perform a vertical 'cross-behavioral sweep' to
       see how different features combine to form the predictive signature of
       a single USV type (e.g., 'Trills' might be driven by high speed AND
       proximity).

    Data Structure & Scaling:
    ------------------------
    - Input: Targets the 'weights_reshaped' 4D tensor (Folds x Classes x Features x Time).
    - Aggregation: Weights are averaged across cross-validation folds to
      represent the model's consensus.
    - Normalization: To preserve the 'signature' of features with different
      units or variances, each subplot is symmetrically scaled to its own
      maximum absolute amplitude (± max|W|). This ensures that subtle,
      temporally-precise features remain visible alongside high-amplitude
      dominant drivers.

    Parameters
    ----------
    selection_results_path : str
        Path to the consolidated ``selection_*.pkl`` artifact produced
        by ``consolidate_model_selection_results``. May be either the
        file itself or a directory containing one (the latest by
        mtime wins when multiple are present). The function extracts
        data from the final accepted model state.
    history_window_sec : float, default=4.0
        The duration of behavioral history analyzed. Used to convert internal
        indices into a human-readable time axis.
    cmap : str, default='RdBu_r'
        Diverging colormap; Red indicates promotion of a category, Blue
        indicates suppression.
    save_plot : bool, default=False
        If True, exports the grid as an SVG file for publication-quality editing.
    output_dir : str, optional
        Directory for saving the figure. Defaults to the parent dir
        of ``selection_results_path`` (or to the path itself if a
        directory was supplied).

    Returns
    -------
    None
        Displays the high-resolution Matplotlib grid.
    """

    selection_results_path = configure_path(str(selection_results_path))
    if output_dir is not None:
        output_dir = configure_path(str(output_dir))

    # 1. Local overrides — restored at function exit so they don't leak
    #    into any plot rendered after this function returns.
    TEXT_COLOR = '#000000'
    _rcp_override = {
        'axes.grid': False,
        'xtick.minor.visible': False,
        'ytick.minor.visible': False
    }
    _saved_rcp = {k: plt.rcParams[k] for k in _rcp_override}
    plt.rcParams.update(_rcp_override)

    selection_steps, _, _ = load_selection_results(selection_results_path)

    if not selection_steps:
        print(f"No step data found in {selection_results_path}")
        return

    # Find the last step that actually selected a feature (ignores the final rejection step)
    valid_data = None
    for data in reversed(selection_steps):
        if data['selected_feature'] is not None:
            valid_data = data
            break

    if valid_data is None:
        print("No valid features were selected in this run.")
        return

    # Self-Healing Logic: Reconstruct weights if the training script failed to finalize
    if 'weights_reshaped' in valid_data:
        weights = valid_data['weights_reshaped']
        features = valid_data['final_model_features']
        classes = valid_data['classes']
    else:
        print("Finalized matrix not found. Reconstructing weights on the fly...")
        winner = valid_data['selected_feature']
        features = valid_data['current_features'] + [winner]
        winner_data = valid_data['candidates_summary'][winner]
        classes = winner_data['classes']

        # Pull raw JAX coefficients and reshape them manually
        raw_weights = np.array(winner_data['folds']['weights'])
        n_folds, n_classes, n_total_inputs = raw_weights.shape
        n_features = len(features)
        n_time_bins = n_total_inputs // n_features
        weights = raw_weights.reshape(n_folds, n_classes, n_features, n_time_bins)

    # ``nanmean`` so folds that failed to converge (stored as all-NaN
    # weight tensors by the CNN trainer) are skipped at each position
    # rather than poisoning every cell of the averaged weight matrix.
    # ``np.mean`` here would propagate any NaN into the cell, which
    # turns the whole heatmap into NaN as soon as a single fold is
    # bad. ``catch_warnings`` silences the "All-NaN slice" emission
    # that fires only for positions where every fold is NaN -- those
    # cells stay NaN and the plotting code below handles them.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_weights = np.nanmean(weights, axis=0)

    n_feats = len(features)
    ncols = 3
    nrows = math.ceil(n_feats / ncols)

    # MANUAL LAYOUT (Absolute Figure Coordinates)
    fig = plt.figure(figsize=(18, 5 * nrows), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')

    LM, BM = 0.15, 0.12
    W_GAP, H_GAP = 0.08, 0.18
    SUB_W = (1.0 - LM - 0.1) / ncols - W_GAP
    SUB_H = (1.0 - BM - 0.1) / nrows - H_GAP

    for i in range(n_feats):
        row = i // ncols
        col = i % ncols

        ax_x = LM + col * (SUB_W + W_GAP)
        ax_y = (1.0 - 0.08 - SUB_H) - row * (SUB_H + H_GAP)

        ax = fig.add_axes([ax_x, ax_y, SUB_W, SUB_H])
        ax.set_facecolor('#FFFFFF')

        feat_slice = mean_weights[:, i, :]
        # ``nanmax`` returns NaN when every cell of ``feat_slice`` is
        # NaN (would happen only if every fold is NaN at every
        # position for this feature -- still possible if the feature
        # was unlucky enough to land entirely in failed folds).
        # NaN is truthy in Python so ``nanmax(...) or 1.0`` would
        # leave NaN as the colormap range -- be explicit instead.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            peak = np.nanmax(np.abs(feat_slice))
        max_amp = float(peak) if np.isfinite(peak) and peak > 0 else 1.0

        ax.imshow(feat_slice, aspect='auto', cmap=cmap,
                  vmin=-max_amp, vmax=max_amp, interpolation='nearest')

        # X-AXIS
        n_bins = feat_slice.shape[1]
        tick_times = np.arange(-int(history_window_sec), 1)
        tick_locs = [(t + history_window_sec) / history_window_sec * (n_bins - 1) for t in tick_times]
        ax.set_xticks(tick_locs)
        ax.set_xticklabels([f"{t}s" for t in tick_times], color=TEXT_COLOR, fontsize=10)
        ax.set_xlabel("Time prior to USV (s)", color=TEXT_COLOR, fontsize=11, labelpad=12)

        # Y-AXIS
        ax.set_yticks(np.arange(len(classes)))
        ax.set_yticklabels(classes, color=TEXT_COLOR, fontsize=10)
        ax.set_ylabel("USV Category", color=TEXT_COLOR, fontsize=12, fontweight='bold', labelpad=20)

        # TITLES & SPINES
        ax.set_title(f"{features[i]}\nInfluence: ±{max_amp:.3f}",
                     color=TEXT_COLOR, fontsize=12, fontweight='bold', pad=25)

        for side in ['top', 'right', 'bottom', 'left']:
            ax.spines[side].set_visible(True)
            ax.spines[side].set_edgecolor('#000000')
            ax.spines[side].set_linewidth(1.5)

        ax.tick_params(axis='both', which='both', bottom=True, left=True,
                       labelbottom=True, color=TEXT_COLOR, length=5)

    cbar_ax = fig.add_axes([0.92, 0.3, 0.012, 0.4])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=-1, vmax=1))
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Predictive Weight (Blue: Suppress | Red: Promote)', color=TEXT_COLOR, fontsize=11, labelpad=15)
    cbar.ax.yaxis.set_tick_params(color=TEXT_COLOR, labelcolor=TEXT_COLOR)
    cbar.set_ticks([-1, 0, 1])
    cbar.ax.set_yticklabels(['-', '0', '+'], color=TEXT_COLOR)

    if save_plot:
        path_str = str(selection_results_path).lower()
        condition = 'male_mute_partner' if 'male_mute_partner' in path_str else \
            ('female' if 'female' in path_str else 'male')
        # ``selection_results_path`` is polymorphic; if a file was
        # passed, anchor the output dir to its parent.
        _fallback = pathlib.Path(selection_results_path)
        if _fallback.is_file():
            _fallback = _fallback.parent
        out_dir = pathlib.Path(output_dir) if output_dir else _fallback
        fname = f"model_selection_multinomial_usv_category_{condition}_filters_final.svg"
        fig.savefig(out_dir / fname, facecolor='#FFFFFF', bbox_inches=None)

    plt.show()
    plt.rcParams.update(_saved_rcp)


def plot_multinomial_selection_diagnosis(
        selection_results_path: str,
        save_plot: bool = False,
        output_dir: str = None,
        feature_label_overrides: dict = None,
) -> None:
    """
    Two-figure post-hoc audit of the final multinomial selection model
    -- replaces the prior 3-heatmap + slope-chart layout, which was
    built around the multinomial selector's null-model baseline at
    step 0 (always predicts the majority class) and therefore made
    the univariate-vs-multivariate comparison degenerate.

    The two figures rendered here are computed entirely from the held-
    out predictions and confusion matrices already stored in the final
    accepted step's ``folds`` dict; failed folds (silent NaN
    placeholders the multinomial selector used to write) are skipped
    by ``nanmean`` / finiteness checks throughout.

    Figure 1 -- pairwise binary AUC matrix
        K x K heatmap where cell (i, j) is the binary AUC of "class i
        vs class j" computed on samples whose true label is i or j,
        using the model's ``p_i / (p_i + p_j)`` as the score. Reveals
        which class pairs are behaviourally separable; the full
        multinomial accuracy can be modest while many pairs are
        well-separated. Only the lower triangle + diagonal is drawn
        because the matrix is symmetric.

    Figure 2 -- per-class recall, log y-axis
        One bar per USV category showing the final multivariate
        model's recall (= correctly classified / total per class).
        Chance baseline (1/K) plotted as a dashed reference line.
        Log scale because recall ranges span ~50x across categories
        when one class is heavily favoured by the model.

    Parameters
    ----------
    selection_results_path : str
        Consolidated ``selection_*.pkl`` produced by
        ``consolidate_model_selection_results`` (file or containing
        dir). Routed through ``configure_path`` for cross-OS mounts.
    save_plot : bool, default False
        Whether to save the two figures to disk.
    output_dir : str, optional
        Output directory. Defaults to the parent dir of
        ``selection_results_path``.
    feature_label_overrides : dict, optional
        Mapping from raw modeling feature names to presentation-
        friendly labels; used in the figure title that names the
        final-model feature set.
    """

    selection_results_path = configure_path(str(selection_results_path))
    if output_dir is not None:
        output_dir = configure_path(str(output_dir))
    BG_COLOR = '#FFFFFF'

    selection_steps, _, selection_metadata = load_selection_results(selection_results_path)
    if not selection_steps:
        print(f"No multinomial step data found in {selection_results_path}")
        return
    _pretty = _make_feature_pretty(feature_label_overrides, selection_metadata)

    # Final accepted step = last step with a real (non-null) winner;
    # skip a rejected last step if present.
    final_step = None
    for s in reversed(selection_steps):
        sel = s['selected_feature']
        if sel and sel != 'null_model_free':
            final_step = s
            break
    if final_step is None:
        print(f"No accepted multivariate step in {selection_results_path}")
        return

    winner = final_step['selected_feature']
    cdata = final_step['candidates_summary'][winner]
    folds = cdata['folds']
    classes_raw = cdata.get('canonical_classes')
    if classes_raw is None:
        classes_raw = cdata.get('classes')
    classes = np.asarray(classes_raw).ravel()
    K = len(classes)
    final_features = list(final_step['final_model_features'] or [])

    # Pool every valid fold's predictions for the AUC matrix
    def _concat_finite(key: str) -> Optional[np.ndarray]:
        parts = []
        for v in folds[key]:
            if v is None:
                continue
            arr = np.asarray(v)
            if arr.size == 0:
                continue
            if np.issubdtype(arr.dtype, np.floating) and not np.all(np.isfinite(arr)):
                continue
            parts.append(arr)
        if not parts:
            return None
        return np.concatenate(parts, axis=0)

    y_true_pool = _concat_finite('y_true')
    y_probs_pool = _concat_finite('y_probs')
    if y_true_pool is None or y_probs_pool is None:
        print(
            "No valid (y_true, y_probs) pairs across any fold of "
            f"the final step in {selection_results_path}"
        )
        return

    auc_matrix = np.full((K, K), np.nan, dtype=float)
    for i in range(K):
        c_i = classes[i]
        for j in range(K):
            if i == j:
                continue
            c_j = classes[j]
            mask = (y_true_pool == c_i) | (y_true_pool == c_j)
            if not np.any(mask):
                continue
            yt_bin = (y_true_pool[mask] == c_i).astype(int)
            if yt_bin.sum() in (0, len(yt_bin)):
                continue
            p_i = y_probs_pool[mask, i]
            p_j = y_probs_pool[mask, j]
            denom = p_i + p_j
            score = np.where(denom > 0, p_i / denom, 0.5)
            try:
                auc_matrix[i, j] = roc_auc_score(yt_bin, score)
            except ValueError:
                continue

    # Per-class recall from pooled confusion matrix
    summed_cms = []
    for cm in folds['confusion_matrix']:
        if cm is None:
            continue
        arr = np.asarray(cm, dtype=float)
        if arr.size == 0 or not np.all(np.isfinite(arr)):
            continue
        summed_cms.append(arr)
    if not summed_cms:
        print(
            "No valid confusion matrices in the final step of "
            f"{selection_results_path}"
        )
        return
    summed_cm = np.sum(summed_cms, axis=0)
    row_sums = summed_cm.sum(axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        recall_per_class = np.where(
            row_sums > 0,
            np.diag(summed_cm) / row_sums,
            0.0,
        )
    chance = 1.0 / K

    # Save-path resolver shared by both figures
    def _resolve_out_dir() -> pathlib.Path:
        if output_dir is None:
            _fallback = pathlib.Path(selection_results_path)
            return _fallback.parent if _fallback.is_file() else _fallback
        return pathlib.Path(output_dir)

    path_str = str(selection_results_path).lower()
    if 'male_mute_partner' in path_str:
        condition = 'male_mute_partner'
    elif 'female' in path_str:
        condition = 'female'
    elif 'male' in path_str:
        condition = 'male'
    else:
        condition = 'unknown'

    # Figure 1 -- pairwise binary AUC, lower triangle + diagonal
    fig_auc, ax = plt.subplots(figsize=(4.0, 3.6), dpi=300)
    fig_auc.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    upper_tri_mask = np.triu(np.ones_like(auc_matrix, dtype=bool), k=1)
    cmap_auc = plt.get_cmap('cividis').copy()
    cmap_auc.set_bad(color=BG_COLOR)
    masked = np.ma.array(
        auc_matrix,
        mask=upper_tri_mask | ~np.isfinite(auc_matrix),
    )
    im = ax.imshow(masked, aspect='equal', cmap=cmap_auc,
                   interpolation='nearest', vmin=0.5, vmax=1.0)
    for i in range(K):
        for j in range(K):
            if j > i:
                continue
            v = auc_matrix[i, j]
            if not np.isfinite(v):
                ax.text(j, i, '–', ha='center', va='center',
                        fontsize=6, color='#909090')
                continue
            txt_color = '#FFFFFF' if v < 0.72 else '#202020'
            ax.text(j, i, f"{v:.2f}", ha='center', va='center',
                    fontsize=6, color=txt_color)
    ax.set_xticks(np.arange(K))
    ax.set_yticks(np.arange(K))
    ax.set_xticklabels([str(c) for c in classes], fontsize=7)
    ax.set_yticklabels([str(c) for c in classes], fontsize=7)
    ax.tick_params(axis='both', length=0, pad=2)
    ax.set_xlabel(r'USV category $j$', fontsize=8, labelpad=4)
    ax.set_ylabel(r'USV category $i$', fontsize=8, labelpad=4)
    cb = fig_auc.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    cb.set_label(r'binary AUC ($i$ vs $j$)', fontsize=7, labelpad=2)
    cb.set_ticks([0.5, 0.7, 1.0])
    cb.ax.tick_params(labelsize=6, length=1.5)
    cb.outline.set_edgecolor('#B0B0B0')
    cb.outline.set_linewidth(0.5)
    # Triangular outline: keep left + bottom spines, hide top + right,
    # and draw a hypotenuse from the top-left corner of cell (0,0) to
    # the bottom-right corner of cell (K-1,K-1).
    for side in ('left', 'bottom'):
        ax.spines[side].set_edgecolor('#404040')
        ax.spines[side].set_linewidth(0.6)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.plot([-0.5, K - 0.5], [-0.5, K - 0.5],
            color='#404040', lw=0.6, clip_on=False, zorder=4)
    fig_auc.subplots_adjust(left=0.16, right=0.95, top=0.94, bottom=0.16)
    if save_plot:
        out_dir = _resolve_out_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = f"multinomial_pairwise_auc_{condition}.svg"
        fig_auc.savefig(out_dir / fname, bbox_inches='tight', dpi=300,
                        facecolor=BG_COLOR, transparent=False)
        print(f"Pairwise AUC figure saved to: {out_dir / fname}")

    # Figure 2 -- per-class recall (log y-axis, bars touch)
    fig_rec, ax_r = plt.subplots(figsize=(3.6, 2.2), dpi=300)
    fig_rec.patch.set_facecolor(BG_COLOR)
    ax_r.set_facecolor(BG_COLOR)

    x = np.arange(K)
    ax_r.bar(x, recall_per_class, width=1.0, color='#404040',
             edgecolor='none',
             label=f"final multivariate ({len(final_features)} features)")
    ax_r.axhline(chance, color='#909090', ls='--', lw=0.6, zorder=0,
                 label=f"chance = 1/{K} = {chance:.2f}")
    ax_r.set_yscale('log')
    finite_positive = recall_per_class[recall_per_class > 0]
    y_lo = float(finite_positive.min()) * 0.5 if finite_positive.size else 1e-3
    ax_r.set_ylim(y_lo, 1.0)
    ax_r.set_xticks(x)
    ax_r.set_xticklabels([str(c) for c in classes], fontsize=7)
    ax_r.set_xlabel('USV category', fontsize=8, labelpad=4)
    ax_r.set_ylabel(
        r'recall = correctly classified / total$_\mathrm{class}$',
        fontsize=8, labelpad=4,
    )
    ax_r.tick_params(axis='x', length=0, pad=2)
    ax_r.tick_params(axis='y', length=2, pad=2, labelsize=6)
    for side in ('top', 'right'):
        ax_r.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax_r.spines[side].set_edgecolor('#404040')
        ax_r.spines[side].set_linewidth(0.6)
    ax_r.legend(frameon=False, fontsize=6, loc='upper right',
                handlelength=1.5, handletextpad=0.4, labelspacing=0.3)
    fig_rec.subplots_adjust(left=0.14, right=0.97, top=0.94, bottom=0.18)
    if save_plot:
        out_dir = _resolve_out_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = f"multinomial_per_class_recall_{condition}.svg"
        fig_rec.savefig(out_dir / fname, bbox_inches='tight', dpi=300,
                        facecolor=BG_COLOR, transparent=False)
        print(f"Per-class recall figure saved to: {out_dir / fname}")

    # Surface which feature set this audit was computed on, so a reader
    # who sees the two figures can connect them back to a specific
    # selection run.
    pretty_feats = [_pretty(f) for f in final_features]
    print(
        f"Multinomial audit: final-model features = {pretty_feats}; "
        f"K = {K}; condition = {condition}"
    )

    plt.show()


def plot_manifold_selection_trajectory(
        selection_results_path: str,
        metric_primary: str = 'r2_spatial',
        primary_metric_name: str = "R² (spatial, KDE-weighted)",
        metric_secondary: str = 'pearson_y',
        secondary_metric_name: str = "Pearson r (manifold y)",
        save_plot: bool = False,
        output_dir: str = None,
        feature_label_overrides: dict = None,
) -> None:
    """
    Plot the continuous-vocal-manifold forward-selection trajectory as
    a compact two-panel summary, mirroring
    ``plot_multinomial_selection_trajectory`` but adapted to the 2-D
    regression metrics emitted by
    ``continuous_vocal_manifold_model_selection``.

    Layout
    ------
    * **Left panel** -- one horizontal bar per accepted step, top-to-
      bottom in selection order. Each bar is split into a lighter base
      (the previous cumulative value of ``metric_primary``) and a
      darker tip (this step's marginal contribution). Bars are
      coloured by the self / other / dyadic palette so the same
      feature gets the same colour across figures. A rejected final
      step, if any, is drawn below a thin separator in grey.
      Higher-is-better metrics (``r2_spatial``, ``pearson_x/y``,
      ``spearman_x/y``) grow rightward from the chance baseline on
      the left; lower-is-better error metrics (``euclidean_mae``,
      ``euclidean_rmse``, ``euclidean_mae_weighted``, ``mahalanobis_mae``,
      ``mae_x``, ``mae_y``) flip the x-axis and grow rightward from the
      step-0 (null-model / intercept-only) baseline on the right.
    * **Right panel** -- two vertical bars on the secondary-metric
      axis: best univariate (= the anchor, since the manifold selector
      restricts the first step to a single anchor feature) and final
      accepted model (stacked bar, one segment per accepted feature,
      segment height = that feature's marginal contribution to the
      secondary metric). Feature labels stack above each bar.

    Chance baselines
    ----------------
    * Higher-is-better metrics: ``0.0`` (r2 / correlation chance).
    * Lower-is-better metrics: read from the step-0
      ``baseline_score`` when present; otherwise pulled from the
      same metric on the null-model fold dict; otherwise falls back
      to the worst (max) across accepted-step means.

    Robustness
    ----------
    All per-fold means use ``np.nanmean`` so NaN folds from the JAX
    bivariate trainer's hyperparameter-grid failures do not poison
    the trajectory.

    Parameters
    ----------
    selection_results_path : str
        Path to the consolidated ``model_selection_final_*.pkl``
        artifact produced by ``continuous_vocal_manifold_model_selection``.
        May be either the file itself or a directory containing one
        (latest mtime wins). Routed through ``configure_path``.
    metric_primary : str, default ``'r2_spatial'``
        Key for the primary per-fold metric. Used for the left-panel
        trajectory.
    primary_metric_name : str, default ``'R² (spatial, KDE-weighted)'``
        Display name for ``metric_primary``; used as the left-panel
        x-axis label (with `` (held-out data)`` appended).
    metric_secondary : str, default ``'pearson_y'``
        Key for the secondary per-fold metric. Used for the right-
        panel bars.
    secondary_metric_name : str, default ``'Pearson r (manifold y)'``
        Display name for ``metric_secondary``; used as the right-
        panel y-axis label (with `` (held-out data)`` appended).
    save_plot : bool, default False
        Whether to save the figure to disk.
    output_dir : str, optional
        Directory to save the plot. Defaults to the parent dir of
        ``selection_results_path`` (or to the path itself if a
        directory was supplied).
    feature_label_overrides : dict, optional
        Mapping from raw feature names (as stored in the pickle) to
        presentation-friendly labels used for every annotation. Raw
        names not in the map render unchanged.
    """

    selection_results_path = configure_path(str(selection_results_path))
    if output_dir is not None:
        output_dir = configure_path(str(output_dir))

    BG_COLOR = '#FFFFFF'

    selection_steps, display_name, selection_metadata = load_selection_results(selection_results_path)

    if not selection_steps:
        print(f"No manifold selection step data found in {selection_results_path}")
        return

    # Sex-aware palette inferred from the filename, matching the
    # bout-onset / multinomial selectors' convention.
    if '_male_' in display_name:
        self_col, other_col = male_color, female_color
    elif '_female_' in display_name:
        self_col, other_col = female_color, male_color
    else:
        self_col, other_col = male_color, female_color

    lower_is_better_set = {
        'euclidean_mae', 'euclidean_rmse', 'euclidean_mae_weighted',
        'euclidean_mae_raw', 'mahalanobis_mae', 'mae_x', 'mae_y',
        'mse', 'rmse',
    }
    is_minimization_primary = metric_primary in lower_is_better_set
    is_minimization_secondary = metric_secondary in lower_is_better_set

    def _fold_mean(folds_metrics: dict, key: str) -> float:
        if not folds_metrics or key not in folds_metrics:
            return float('nan')
        arr = np.array(folds_metrics[key], dtype=float)
        arr = arr[np.isfinite(arr)]
        return float(np.mean(arr)) if arr.size else float('nan')

    def _chance_value(metric_name: str, minimisation: bool) -> float:
        """Higher-is-better metrics anchor at 0; for error metrics we
        try to pull the null-model baseline from step 0, otherwise use
        the worst observed mean across accepted steps."""
        if not minimisation:
            return 0.0
        null_step = selection_steps[0] if selection_steps else None
        if null_step is not None:
            null_cs = null_step['candidates_summary']
            for cd in null_cs.values():
                fm = cd['folds']['metrics']
                val = _fold_mean(fm, metric_name)
                if np.isfinite(val):
                    return val
        return float('nan')

    # Accepted steps = those with a real selected_feature that is not
    # the manifold selector's null baseline marker.
    accepted_steps = [
        s for s in selection_steps
        if s['selected_feature'] not in (None, 'null_model_free')
    ]
    if not accepted_steps:
        print(f"No accepted feature steps in {selection_results_path}")
        return

    chance_primary = _chance_value(metric_primary, is_minimization_primary)
    chance_secondary = _chance_value(metric_secondary, is_minimization_secondary)
    if not np.isfinite(chance_primary):
        chance_primary = 0.0
    if not np.isfinite(chance_secondary):
        chance_secondary = 0.0

    steps_data = []
    for s in accepted_steps:
        winner = s['selected_feature']
        wd = s['candidates_summary'][winner]
        fm = wd['folds']['metrics']
        steps_data.append({
            'feature_name': winner,
            'prim_mean': _fold_mean(fm, metric_primary),
            'sec_mean': _fold_mean(fm, metric_secondary),
        })

    # Best univariate = the anchor step (the manifold selector locks
    # the first accepted step to a single anchor feature, so there is
    # no broader single-feature screen here).
    anchor_step = accepted_steps[0]
    anchor_winner = anchor_step['selected_feature']
    anchor_fm = (
        anchor_step['candidates_summary'][anchor_winner]['folds']['metrics']
    )
    best_univariate_value = _fold_mean(anchor_fm, metric_secondary)
    best_univariate_feat = anchor_winner

    final_score = float(steps_data[-1]['sec_mean'])

    # Rejected final step lookup -- mirrored from the multinomial
    # trajectory plotter so a rejection row is rendered consistently.
    rejection_row = None
    if selection_steps[-1]['selected_feature'] is None:
        rej_step = selection_steps[-1]
        rej_cs = rej_step['candidates_summary']
        if rej_cs:
            best_rej_feat = None
            best_rej_pri = float('nan')
            comp = (lambda a, b: a < b) if is_minimization_primary else (lambda a, b: a > b)
            best_pri_so_far = np.inf if is_minimization_primary else -np.inf
            for f, cdata in rej_cs.items():
                fm = cdata['folds']['metrics']
                pv = _fold_mean(fm, metric_primary)
                if not np.isfinite(pv):
                    continue
                if comp(pv, best_pri_so_far):
                    best_pri_so_far = pv
                    best_rej_feat = f
                    best_rej_pri = pv
            if best_rej_feat is not None:
                rejection_row = {
                    'feature_name': best_rej_feat,
                    'prim_mean': best_rej_pri,
                }

    # Feature-category colour map: dyadic features get the social
    # colour; self-prefixed and sex-emitted-from-self features get
    # self_col; everything else (other.*, partner orofacial, etc.)
    # gets other_col.
    dyadic_keywords = [
        "nose-nose", "nose-TTI", "TTI-nose",
        "allo_yaw-nose", "nose-allo_yaw",
        "allo_yaw-TTI", "TTI-allo_yaw",
        "allo_pitch-nose", "nose-allo_pitch",
        "allo_pitch-TTI", "TTI-allo_pitch",
    ]

    def _category_color(fname: str) -> str:
        if any(x in fname for x in dyadic_keywords):
            return DYADIC_COLOR
        if '-sei' in fname:
            return self_col
        if fname.startswith('self.') or 'self' in fname:
            return self_col
        return other_col

    def _lighten(hex_color: str, factor: float = 0.65) -> str:
        h = hex_color.lstrip('#')
        r = int(h[0:2], 16); g = int(h[2:4], 16); b = int(h[4:6], 16)
        r = int(round(r + (255 - r) * factor))
        g = int(round(g + (255 - g) * factor))
        b = int(round(b + (255 - b) * factor))
        return f"#{r:02x}{g:02x}{b:02x}"

    _pretty = _make_feature_pretty(feature_label_overrides, selection_metadata)

    cum_prim = [d['prim_mean'] for d in steps_data]
    cum_sec = [d['sec_mean'] for d in steps_data]
    sec_marginals = [cum_sec[0] - chance_secondary]
    for i in range(1, len(cum_sec)):
        sec_marginals.append(cum_sec[i] - cum_sec[i - 1])

    # Figure layout
    n_rows_total = len(steps_data) + (1 if rejection_row is not None else 0)
    fig_height = max(3.0, 0.32 * n_rows_total + 1.8)
    fig_traj, (ax_traj, ax_bars) = plt.subplots(
        nrows=1, ncols=2,
        figsize=(10.5, fig_height), dpi=300,
        gridspec_kw={'width_ratios': [2.2, 1.0]},
    )
    fig_traj.patch.set_facecolor(BG_COLOR)
    ax_traj.set_facecolor(BG_COLOR)
    ax_bars.set_facecolor(BG_COLOR)

    bar_height = 0.88
    y_positions = list(range(len(steps_data)))
    rej_y = len(steps_data) + 0.4 if rejection_row is not None else None

    for row_idx, d in enumerate(steps_data):
        y = y_positions[row_idx]
        base_color = _category_color(d['feature_name'])
        light_color = _lighten(base_color, factor=0.65)
        prev_p = cum_prim[row_idx - 1] if row_idx > 0 else chance_primary
        cur_p = cum_prim[row_idx]

        if is_minimization_primary:
            if chance_primary > prev_p:
                ax_traj.barh(y, chance_primary - prev_p, left=prev_p,
                             height=bar_height, color=light_color,
                             edgecolor='none')
            if prev_p > cur_p:
                ax_traj.barh(y, prev_p - cur_p, left=cur_p,
                             height=bar_height, color=base_color,
                             edgecolor='none')
            delta = prev_p - cur_p
            ax_traj.text(cur_p - 0.003 * (chance_primary - cur_p + 1e-9), y,
                         f"{cur_p:.3f}  (Δ -{delta:.3f})",
                         ha='left', va='center', fontsize=7,
                         color=TEXT_COLOR)
        else:
            if prev_p > chance_primary:
                ax_traj.barh(y, prev_p - chance_primary, left=chance_primary,
                             height=bar_height, color=light_color,
                             edgecolor='none')
            if cur_p > prev_p:
                ax_traj.barh(y, cur_p - prev_p, left=prev_p,
                             height=bar_height, color=base_color,
                             edgecolor='none')
            delta = cur_p - prev_p
            ax_traj.text(cur_p + 0.003, y,
                         f"{cur_p:.3f}  (Δ +{delta:.3f})",
                         ha='left', va='center', fontsize=7,
                         color=TEXT_COLOR)

    if rejection_row is not None and not np.isnan(rejection_row['prim_mean']):
        rejected_light = '#D7D7D7'
        rejected_dark = '#9A9A9A'
        prev_p = cum_prim[-1]
        cur_p = rejection_row['prim_mean']
        if is_minimization_primary:
            if chance_primary > prev_p:
                ax_traj.barh(rej_y, chance_primary - prev_p, left=prev_p,
                             height=bar_height, color=rejected_light,
                             edgecolor='none')
            if prev_p > cur_p:
                ax_traj.barh(rej_y, prev_p - cur_p, left=cur_p,
                             height=bar_height, color=rejected_dark,
                             edgecolor='none')
            ax_traj.text(cur_p - 0.003, rej_y,
                         f"{cur_p:.3f}  (Δ -{prev_p - cur_p:.3f}, ns)",
                         ha='left', va='center', fontsize=7,
                         color=NEUTRAL_COLOR, style='italic')
        else:
            if prev_p > chance_primary:
                ax_traj.barh(rej_y, prev_p - chance_primary, left=chance_primary,
                             height=bar_height, color=rejected_light,
                             edgecolor='none')
            if cur_p > prev_p:
                ax_traj.barh(rej_y, cur_p - prev_p, left=prev_p,
                             height=bar_height, color=rejected_dark,
                             edgecolor='none')
            ax_traj.text(cur_p + 0.003, rej_y,
                         f"{cur_p:.3f}  (Δ +{cur_p - prev_p:.3f}, ns)",
                         ha='left', va='center', fontsize=7,
                         color=NEUTRAL_COLOR, style='italic')

    ytick_positions = list(y_positions)
    ytick_labels = [f"+{_pretty(d['feature_name'])}" for d in steps_data]
    if rejection_row is not None:
        ytick_positions.append(rej_y)
        ytick_labels.append(f"+{_pretty(rejection_row['feature_name'])}")
    ax_traj.set_yticks(ytick_positions)
    ax_traj.set_yticklabels(ytick_labels, fontsize=10, color=TEXT_COLOR)
    ax_traj.tick_params(axis='y', length=0)
    ax_traj.invert_yaxis()

    if rejection_row is not None:
        sep_y = (len(steps_data) - 1) + 0.5 + 0.10
        ax_traj.axhline(sep_y, color=NEUTRAL_COLOR, linestyle='-',
                        lw=0.4, alpha=0.5, zorder=0)

    # X-axis range / orientation depending on direction.
    all_endpoints = list(cum_prim)
    if rejection_row is not None and not np.isnan(rejection_row['prim_mean']):
        all_endpoints.append(rejection_row['prim_mean'])
    if is_minimization_primary:
        span = chance_primary - min(all_endpoints)
        span = span if span > 1e-9 else 1.0
        x_left_lim = min(all_endpoints) - 0.30 * span
        x_right_lim = chance_primary + 0.015 * abs(chance_primary if chance_primary > 0 else 1)
        ax_traj.set_xlim(x_left_lim, x_right_lim)
        ax_traj.invert_xaxis()
    else:
        span = max(all_endpoints) - chance_primary
        span = span if span > 1e-9 else 1.0
        x_left_lim = chance_primary - 0.015 * (1.0 if span < 0.1 else span)
        x_right_lim = max(all_endpoints) + 0.30 * span
        ax_traj.set_xlim(x_left_lim, x_right_lim)

    ax_traj.set_xlabel(f"{primary_metric_name} (held-out data)",
                       fontsize=10, color=TEXT_COLOR)
    ax_traj.spines['top'].set_visible(False)
    ax_traj.spines['right'].set_visible(False)
    ax_traj.tick_params(axis='x', colors=TEXT_COLOR)
    for spine in ax_traj.spines.values():
        spine.set_edgecolor(TEXT_COLOR)

    # Right panel: 2 vertical bars (best univariate, final)
    bar_width = 0.6
    bar_x_positions = [0, 1]
    bar_group_labels = ['best univariate', 'final model']

    bar1_color = _category_color(best_univariate_feat) if best_univariate_feat else NEUTRAL_COLOR
    ax_bars.bar(0, best_univariate_value - chance_secondary,
                bottom=chance_secondary, width=bar_width,
                color=bar1_color, edgecolor='none')

    _bottom = chance_secondary
    for d, marginal in zip(steps_data, sec_marginals):
        seg_color = _category_color(d['feature_name'])
        ax_bars.bar(1, marginal, bottom=_bottom, width=bar_width,
                    color=seg_color, edgecolor='none')
        _bottom += marginal

    bar_tops = [best_univariate_value, final_score]
    y_data_max = float(np.nanmax(bar_tops))
    y_data_min = float(np.nanmin(bar_tops + [chance_secondary]))
    label_line_spacing_data = max(0.006, 0.012 * abs(y_data_max - chance_secondary + 1e-9))
    label_y_start_offset = max(0.005, 0.01 * abs(y_data_max - chance_secondary + 1e-9))
    label_fontsize = 8
    max_label_lines = len(steps_data)
    label_stack_top = (y_data_max + label_y_start_offset
                       + (max_label_lines + 1) * label_line_spacing_data)

    # Generic axis sizing: pad ~25% above the highest bar (or label
    # stack, whichever is taller) and ~5% below the chance baseline.
    y_top = max(y_data_max + 0.25 * abs(y_data_max - chance_secondary + 1e-9),
                label_stack_top)
    y_bot = min(chance_secondary - 0.05 * abs(y_data_max - chance_secondary + 1e-9),
                y_data_min - 0.05 * abs(y_data_max - chance_secondary + 1e-9))
    ax_bars.set_ylim(y_bot, y_top)

    ax_bars.set_xticks(bar_x_positions)
    ax_bars.set_xticklabels(bar_group_labels, fontsize=7, color=TEXT_COLOR)
    ax_bars.set_xlim(-0.6, len(bar_group_labels) - 0.4)
    ax_bars.set_ylabel(f"{secondary_metric_name} (held-out data)",
                       fontsize=10, color=TEXT_COLOR)

    final_feat_labels = [f"+{_pretty(d['feature_name'])}" for d in steps_data]

    if best_univariate_feat is not None:
        ax_bars.text(0, best_univariate_value + label_y_start_offset,
                     _pretty(best_univariate_feat),
                     ha='center', va='bottom',
                     fontsize=label_fontsize, color=TEXT_COLOR)

    for j, lab in enumerate(final_feat_labels):
        ax_bars.text(1, final_score + label_y_start_offset
                     + j * label_line_spacing_data,
                     lab, ha='center', va='bottom',
                     fontsize=label_fontsize, color=TEXT_COLOR)

    ax_bars.spines['top'].set_visible(False)
    ax_bars.spines['right'].set_visible(False)
    ax_bars.tick_params(axis='both', colors=TEXT_COLOR)
    for spine in ax_bars.spines.values():
        spine.set_edgecolor(TEXT_COLOR)

    fig_traj.subplots_adjust(left=0.18, right=0.97, top=0.95,
                             bottom=0.12, wspace=0.12)

    if save_plot:
        if output_dir is None:
            _fallback = pathlib.Path(selection_results_path)
            _out_dir = _fallback.parent if _fallback.is_file() else _fallback
        else:
            _out_dir = pathlib.Path(output_dir)
        _out_dir.mkdir(parents=True, exist_ok=True)
        path_str = str(selection_results_path).lower()
        if 'male_mute_partner' in path_str:
            condition = 'male_mute_partner'
        elif 'female' in path_str:
            condition = 'female'
        elif 'male' in path_str:
            condition = 'male'
        else:
            condition = 'unknown'
        fname = (f"manifold_selection_trajectory_{condition}_"
                 f"{metric_primary}.svg")
        save_path = _out_dir / fname
        fig_traj.savefig(save_path, bbox_inches='tight', dpi=300,
                         facecolor=BG_COLOR, transparent=False)
        print(f"Manifold trajectory plot saved to: {save_path}")

    plt.show()


def plot_manifold_multivariate_filters(
        selection_results_path: str,
        history_window_sec: float = 4.0,
        cmap: str = 'RdBu_r',
        save_plot: bool = False,
        output_dir: str = None,
) -> None:
    """
    Visualize the converged multivariate manifold (bivariate-regression)
    model as a per-feature atlas of temporal influence on the 2-D
    acoustic manifold output.

    Purpose
    -------
    Mirror of ``plot_multinomial_multivariate_filters`` but for the 2-D
    output case: each behavioral feature gets a single subplot, and
    within that subplot the two rows correspond to the manifold
    coordinates (row 0 = manifold-x partial filter, row 1 = manifold-y
    partial filter). The x-axis is time prior to USV onset; the cell
    colour is the signed coefficient (red = pushes the predicted USV
    location toward +axis, blue = pushes toward -axis), symmetrically
    scaled to that feature's own ± max|W| so subtle filters remain
    visible alongside dominant drivers.

    Data layout
    -----------
    The bivariate regressor stores per-fold weights as
    ``(n_features * n_time_bins, 2)`` -- the 2 here is the manifold
    output dim. We reshape to
    ``(n_folds, n_features, n_time_bins, 2)``, ``nanmean`` across folds
    (so a single failed-fold NaN doesn't poison the average), and
    visualise the ``(2, n_time_bins)`` slice per feature.

    Parameters
    ----------
    selection_results_path : str
        Path to the consolidated ``model_selection_final_*.pkl``
        artifact produced by ``continuous_vocal_manifold_model_selection``.
        May be either the file itself or a directory containing one
        (latest mtime wins). Routed through ``configure_path``.
    history_window_sec : float, default 4.0
        The duration of behavioral history analyzed. Used to convert
        the internal time-bin index into a human-readable axis.
    cmap : str, default ``'RdBu_r'``
        Diverging colormap. Red = positive coefficient (pushes the
        predicted USV location toward the +axis); blue = negative.
    save_plot : bool, default False
        If True, exports the grid as an SVG for publication-quality
        editing.
    output_dir : str, optional
        Directory for saving the figure. Defaults to the parent dir
        of ``selection_results_path`` (or to the path itself if a
        directory was supplied).

    Returns
    -------
    None
        Displays the high-resolution Matplotlib grid.
    """

    selection_results_path = configure_path(str(selection_results_path))
    if output_dir is not None:
        output_dir = configure_path(str(output_dir))

    TEXT_COLOR_LOCAL = '#000000'
    _rcp_override = {
        'axes.grid': False,
        'xtick.minor.visible': False,
        'ytick.minor.visible': False,
    }
    _saved_rcp = {k: plt.rcParams[k] for k in _rcp_override}
    plt.rcParams.update(_rcp_override)

    try:
        selection_steps, _, _ = load_selection_results(selection_results_path)
        if not selection_steps:
            print(f"No manifold selection step data found in {selection_results_path}")
            return

        # Last step with an actual selected feature -- skips the
        # rejection / null marker steps so we always render the final
        # accepted multivariate model.
        valid_data = None
        for data in reversed(selection_steps):
            sel = data['selected_feature']
            if sel is not None and sel != 'null_model_free':
                valid_data = data
                break
        if valid_data is None:
            print("No accepted feature steps in this run.")
            return

        winner = valid_data['selected_feature']
        features = list(valid_data['current_features']) + [winner]
        if features and features[0] == winner and len(features) > 1 and features.count(winner) > 1:
            # Anchor step writes the same feature into both
            # ``current_features`` and ``selected_feature``; drop the
            # duplicate so the filter atlas isn't doubled.
            features = list(valid_data['current_features'])
        winner_data = valid_data['candidates_summary'][winner]

        raw_weights = np.array(winner_data['folds']['weights'])
        if raw_weights.ndim != 3 or raw_weights.shape[-1] != 2:
            print(
                f"Unexpected weight shape {raw_weights.shape}; expected "
                f"(n_folds, n_features*n_time_bins, 2). Aborting."
            )
            return
        n_folds, n_total_inputs, _ = raw_weights.shape
        n_features = len(features)
        if n_features == 0 or n_total_inputs % n_features != 0:
            print(
                f"Cannot reshape weights: {n_total_inputs} columns is not "
                f"divisible by {n_features} features. Aborting."
            )
            return
        n_time_bins = n_total_inputs // n_features
        weights = raw_weights.reshape(n_folds, n_features, n_time_bins, 2)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean_weights = np.nanmean(weights, axis=0)

        ncols = 3
        nrows = math.ceil(n_features / ncols)

        fig = plt.figure(figsize=(18, 3.6 * nrows), dpi=300)
        fig.patch.set_facecolor('#FFFFFF')

        LM, BM = 0.10, 0.14
        W_GAP, H_GAP = 0.08, 0.32
        SUB_W = (1.0 - LM - 0.1) / ncols - W_GAP
        SUB_H = (1.0 - BM - 0.12) / nrows - H_GAP

        manifold_axis_labels = ['manifold x', 'manifold y']

        for i in range(n_features):
            row = i // ncols
            col = i % ncols
            ax_x = LM + col * (SUB_W + W_GAP)
            ax_y = (1.0 - 0.10 - SUB_H) - row * (SUB_H + H_GAP)
            ax = fig.add_axes([ax_x, ax_y, SUB_W, SUB_H])
            ax.set_facecolor('#FFFFFF')

            feat_slice = mean_weights[i, :, :].T  # shape (2, n_time_bins)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                peak = np.nanmax(np.abs(feat_slice))
            max_amp = float(peak) if np.isfinite(peak) and peak > 0 else 1.0

            ax.imshow(feat_slice, aspect='auto', cmap=cmap,
                      vmin=-max_amp, vmax=max_amp, interpolation='nearest')

            tick_times = np.arange(-int(history_window_sec), 1)
            tick_locs = [
                (t + history_window_sec) / history_window_sec * (n_time_bins - 1)
                for t in tick_times
            ]
            ax.set_xticks(tick_locs)
            ax.set_xticklabels([f"{t}s" for t in tick_times],
                               color=TEXT_COLOR_LOCAL, fontsize=10)
            ax.set_xlabel("Time prior to USV (s)", color=TEXT_COLOR_LOCAL,
                          fontsize=11, labelpad=10)

            ax.set_yticks([0, 1])
            ax.set_yticklabels(manifold_axis_labels,
                               color=TEXT_COLOR_LOCAL, fontsize=10)
            ax.set_ylabel("Output dim", color=TEXT_COLOR_LOCAL,
                          fontsize=11, fontweight='bold', labelpad=12)

            ax.set_title(f"{features[i]}\nInfluence: ±{max_amp:.3f}",
                         color=TEXT_COLOR_LOCAL, fontsize=11,
                         fontweight='bold', pad=14)

            for side in ('top', 'right', 'bottom', 'left'):
                ax.spines[side].set_visible(True)
                ax.spines[side].set_edgecolor('#000000')
                ax.spines[side].set_linewidth(1.2)

            ax.tick_params(axis='both', which='both', bottom=True, left=True,
                           labelbottom=True, color=TEXT_COLOR_LOCAL, length=4)

        cbar_ax = fig.add_axes([0.92, 0.3, 0.012, 0.4])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=-1, vmax=1))
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label(
            'Per-feature partial weight (blue: -axis | red: +axis)',
            color=TEXT_COLOR_LOCAL, fontsize=11, labelpad=12,
        )
        cbar.ax.yaxis.set_tick_params(color=TEXT_COLOR_LOCAL,
                                      labelcolor=TEXT_COLOR_LOCAL)
        cbar.set_ticks([-1, 0, 1])
        cbar.ax.set_yticklabels(['-', '0', '+'], color=TEXT_COLOR_LOCAL)

        if save_plot:
            path_str = str(selection_results_path).lower()
            condition = 'male_mute_partner' if 'male_mute_partner' in path_str else \
                ('female' if 'female' in path_str else 'male')
            _fallback = pathlib.Path(selection_results_path)
            if _fallback.is_file():
                _fallback = _fallback.parent
            out_dir = pathlib.Path(output_dir) if output_dir else _fallback
            out_dir.mkdir(parents=True, exist_ok=True)
            fname = f"model_selection_manifold_{condition}_filters_final.svg"
            fig.savefig(out_dir / fname, facecolor='#FFFFFF', bbox_inches=None)
            print(f"Manifold filters plot saved to: {out_dir / fname}")

        plt.show()
    finally:
        plt.rcParams.update(_saved_rcp)


class DeepResultsVisualizer:
    """
    Interpretation engine for CNN-based USV manifold models.

    This class provides a high-level API to visualize the statistical validity,
    global feature reliance, spatial calibration, and localized behavioral
    saliency of the deep kinematic-to-acoustic mapping. It handles data
    aggregation across cross-validation folds and manages high-resolution
    vector exports.

    Attributes
    ----------
    data : dict
        The loaded Deep Storage dictionary from the training pkl.
    metadata : dict
        Model hyperparameters and feature lists.
    features : list
        List of kinematic feature names.
    n_bins : int
        Number of temporal bins in the history window.
    save_dir : str
        Default directory for saving generated plots.
    default_color : str
        The primary color (male_color or female_color) detected from the filename.
    """

    def __init__(self, results_pkl_path: str,
                 modeling_settings: dict | None,
                 visualization_settings: dict | None,):
        """
        Initializes the visualizer, loads results, and detects subject sex for coloring.

        Parameters
        ----------
        results_pkl_path : str
            Full path to the .pkl file generated by run_cnn_training.
        modeling_settings : dict
            A nested dictionary containing IO paths, filter histories, split strategies,
            and JAX/Optax hyperparameters. If None is provided, it attempts to load
            from the default JSON configuration file.
        visualization_settings : dict
            A nested dictionary containing visualization settings, including
            animal colors, etc. If None is provided, it loads from the default
            visualizations_settings.json configuration file.
        """

        results_pkl_path = configure_path(str(results_pkl_path))
        if not os.path.exists(results_pkl_path):
            raise FileNotFoundError(f"Results file not found: {results_pkl_path}")

        self.results_pkl_path = results_pkl_path

        with open(results_pkl_path, 'rb') as f:
            self.data = pickle.load(f)

        self.metadata = self.data['metadata']
        self.features = self.metadata['features_list']
        self.n_bins = self.metadata['n_time_bins']
        self.save_dir = self.metadata.get('save_dir', './plots')

        # Manifold-metric provenance. Recent CNN runs save the
        # `(metric, period)` pair into `data['metadata']`; legacy
        # pickles predate this and default to flat-space euclidean
        # so the visualisations remain back-compatible without any
        # caller-side awareness. The plots that compute distances
        # between predictions and ground truth (`spatial_precision_grid`,
        # `error_landscape`, `regional_saliency_inset`) read these
        # via `signed_diff` so torus pickles render with wrap-aware
        # error magnitudes.
        self.manifold_metric = self.metadata.get('manifold_metric', 'euclidean')
        self.manifold_period = float(self.metadata.get('manifold_period', 1.0))

        if modeling_settings is None:
            settings_path = pathlib.Path(__file__).resolve().parent.parent / '_parameter_settings/modeling_settings.json'
            try:
                with open(settings_path, 'r') as settings_json_file:
                    self.modeling_settings = json.load(settings_json_file)
            except FileNotFoundError:
                raise FileNotFoundError(f"Settings file not found at {settings_path}")
        else:
            self.modeling_settings = modeling_settings

        if visualization_settings is None:
            viz_settings_path = pathlib.Path(__file__).resolve().parent.parent / '_parameter_settings/visualizations_settings.json'
            try:
                with open(viz_settings_path, 'r') as viz_settings_json_file:
                    self.visualization_settings = json.load(viz_settings_json_file)
            except FileNotFoundError:
                raise FileNotFoundError(f"Settings file not found at {viz_settings_path}")
        else:
            self.visualization_settings = visualization_settings

        # Logic to set the default subject color based on filename
        fname = os.path.basename(results_pkl_path).lower()
        if 'female' in fname:
            self.default_color = female_color
        elif 'male' in fname:
            self.default_color = male_color
        else:
            self.default_color = '#8CA252'

    def _handle_save(self, fig: plt.Figure, name: str, save_plot: bool,
                     output_dir: Optional[str], file_format: str) -> None:
        """
        Internal helper to manage figure exports with SVG default and white facecolor.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            The figure object to save.
        name : str
            Base filename for the export.
        save_plot : bool
            Whether to actually save the file.
        output_dir : str, optional
            Override for the default save directory.
        file_format : str
            The file extension/format (defaults to svg).
        """
        if not save_plot:
            return

        if output_dir is not None:
            output_dir = configure_path(str(output_dir))
        target_dir = pathlib.Path(output_dir) if output_dir else pathlib.Path(self.save_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        ext = file_format.strip('.') if file_format else 'svg'
        save_path = target_dir / f"{name}.{ext}"

        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
        print(f"[+] Figure saved: {save_path}")

    def plot_permutation_test(self,
                              actual_color: str = None,
                              null_color: str = '#D3D3D3',
                              threshold_color: str = '#000000',
                              title_fontsize: int = 10,
                              label_fontsize: int = 8,
                              figsize: tuple = (3, 4),
                              n_bootstraps: int = 1000,
                              save_plot: bool = False,
                              output_dir: str = None,
                              file_format: str = 'svg') -> None:
        """
        Plots the bootstrapped permutation test histograms to prove model validity.

        This visualization is a two-panel figure:
        1. (Top) Compares the mean Euclidean error of the actual model against the
           empirical density draw (the 'null_model_free' strategy) using a broken x-axis.
        2. (Bottom) Compares the Error Reduction Skill Score of the actual model
           against the null distribution using a broken x-axis.

        The Skill Score is defined as:
        $$Skill = 1 - \\frac{Error_{Actual}}{Error_{Null}}$$
        where 0.00 represents performance equal to the null baseline and 1.00
        represents perfect prediction.

        Grids are explicitly disabled for a clean aesthetic.

        Parameters
        ----------
        actual_color : str, optional
            Color of the vertical line for actual model. Defaults to self.default_color.
        null_color : str, default '#D3D3D3'
            Color of the bootstrapped null distribution histogram.
        threshold_color : str, default '#000000'
            Color of the 0.5% significance threshold line.
        title_fontsize : int, default 10
            Font size for the plot title.
        label_fontsize : int, default 8
            Font size for the axis labels.
        figsize : tuple, default (3, 4)
            Figure dimensions in inches.
        n_bootstraps: int, default 1000
            The number of resampling iterations to generate the null distribution.
        save_plot : bool, default False
            If True, exports the figure to disk.
        output_dir : str, optional
            Path to save the figure.
        file_format : str, default 'svg'
            Format for the saved file.
        """

        c_act = actual_color if actual_color is not None else getattr(self, 'default_color', '#1f77b4')
        text_color = '#000000'

        # 1. Data Extraction
        # Filter out placeholder folds emitted by the runner's
        # `restrict_to_fold_indices` recovery path — they carry only
        # `Y_true` / `test_indices` and would `KeyError` on
        # `Y_pred_actual` / `error_*` lookups below.
        cv_folds = [f for f in self.data['cross_validation'] if not f.get('skipped')]
        actual_errors = np.array([fold['error_actual'] for fold in cv_folds])
        null_free_errors = np.array([fold['error_null_model_free'] for fold in cv_folds])

        # Compute Actual Skill Score: 1 - (Actual / Null)
        actual_skill = 1.0 - (actual_errors / null_free_errors)

        # COMPUTE PERMUTED NULL SKILL: 1 - (Null_shuffled / Null)
        # This gives the Null "width" so it doesn't break the Y-axis scaling
        shuffled_null_errors = np.random.permutation(null_free_errors)
        null_skill_dist = 1.0 - (shuffled_null_errors / null_free_errors)

        actual_mean = np.mean(actual_errors)
        actual_mean_skill = np.mean(actual_skill)

        # 2. Bootstrap the distributions
        np.random.seed(42)

        # Euclidean Error Bootstrapping (Panel A)
        null_dist = [np.mean(np.random.choice(null_free_errors, size=len(null_free_errors), replace=True))
                     for _ in range(n_bootstraps)]
        actual_dist = [np.mean(np.random.choice(actual_errors, size=len(actual_errors), replace=True))
                       for _ in range(n_bootstraps)]
        threshold = np.percentile(null_dist, 0.5)

        # Skill Score Bootstrapping (Panel B)
        actual_dist_skill = [np.mean(np.random.choice(actual_skill, size=len(actual_skill), replace=True))
                             for _ in range(n_bootstraps)]
        null_dist_skill = [np.mean(np.random.choice(null_skill_dist, size=len(null_skill_dist), replace=True))
                           for _ in range(n_bootstraps)]

        # 3. Plotting Physics: GridSpec for 4 subplots (2 rows of broken axes)
        fig = plt.figure(figsize=figsize, facecolor='#FFFFFF')
        gs = fig.add_gridspec(2, 2, width_ratios=[1, 2], height_ratios=[1, 1], wspace=0.05, hspace=0.6)

        # Top Row: Euclidean Error (Broken X-Axis)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)

        # Bottom Row: Skill Score (Broken X-Axis)
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1], sharey=ax3)

        # Set background, remove grids, and hardcode spine visibility for all 4 axes
        for ax in (ax1, ax2, ax3, ax4):
            ax.set_facecolor('#FFFFFF')
            ax.grid(False)
            ax.tick_params(colors=text_color, which='both', labelsize=label_fontsize - 1)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(True)
            ax.spines['bottom'].set_color(text_color)
            ax.spines['bottom'].set_linewidth(1)

        # Spine handling for broken axes logic
        for left_ax, right_ax in [(ax1, ax2), (ax3, ax4)]:
            left_ax.spines['left'].set_visible(True)
            left_ax.spines['left'].set_color(text_color)
            left_ax.spines['left'].set_linewidth(1)
            right_ax.spines['left'].set_visible(False)
            right_ax.tick_params(axis='y', which='both', left=False, labelleft=False)

        # PANEL A: EUCLIDEAN ERROR (TOP ROW)
        # Actual Model (ax1)
        ax1.hist(actual_dist, bins=30, histtype='step', fill=True, color=c_act, alpha=0.5, edgecolor=c_act, linewidth=1.5)
        ax1.axvline(actual_mean, color=c_act, linewidth=1.0, linestyle='-')
        ax1.text(actual_mean, 1.02, f'{actual_mean:.2f}', color=c_act,
                 transform=ax1.get_xaxis_transform(), ha='center', va='bottom',
                 fontsize=label_fontsize, fontweight='bold', clip_on=False)

        # Null Model (ax2)
        ax2.hist(null_dist, bins=30, histtype='step', fill=True, color=null_color, alpha=0.5, edgecolor=null_color, linewidth=1.5)
        ax2.axvline(threshold, color=threshold_color, linewidth=1.0, linestyle='--')
        ax2.text(threshold, 1.02, f'{threshold:.2f}', color=null_color,
                 transform=ax2.get_xaxis_transform(), ha='center', va='bottom',
                 fontsize=label_fontsize, fontweight='bold', clip_on=False)

        # Legend Proxy Artists (Required for labels)
        ax1.plot([], [], color=c_act, linewidth=1.5, linestyle='-', label='Model')
        ax2.plot([], [], color=null_color, linewidth=1.5, linestyle='-', label='Null')

        # Set strict axis limits for A
        pad_actual = (max(actual_dist) - min(actual_dist)) * 0.5
        if pad_actual == 0: pad_actual = 0.05
        ax1.set_xlim(min(actual_dist) - pad_actual, max(actual_dist) + pad_actual)

        pad_null = (max(null_dist) - min(null_dist)) * 0.5
        ax2.set_xlim(min(null_dist) - pad_null, max(null_dist) + pad_null)

        # PANEL B: SKILL SCORE (BOTTOM ROW)
        ax3.hist(null_dist_skill, bins=50, histtype='step', fill=True, color=null_color, alpha=0.5, edgecolor=null_color, linewidth=1.5, range=(-0.02, 0.02))
        ax3.axvline(np.percentile(null_dist_skill, q=99.5), color=threshold_color, linewidth=1.0, linestyle='--')
        ax3.text(0.0, 1.02, '0.00', color=null_color,
                 transform=ax3.get_xaxis_transform(), ha='center', va='bottom',
                 fontsize=label_fontsize, fontweight='bold', clip_on=False)

        # Actual Model Skill Score (ax4)
        ax4.hist(actual_dist_skill, bins=30, histtype='step', fill=True, color=c_act, alpha=0.5, edgecolor=c_act, linewidth=1.5)
        ax4.axvline(actual_mean_skill, color=c_act, linewidth=1.0, linestyle='-')
        ax4.text(actual_mean_skill, 1.02, f'{actual_mean_skill:.2f}', color=c_act,
                 transform=ax4.get_xaxis_transform(), ha='center', va='bottom',
                 fontsize=label_fontsize, fontweight='bold', clip_on=False)

        # Limits for Skill Score
        ax3.set_xlim(-0.03, 0.03)
        pad_skill = (max(actual_dist_skill) - min(actual_dist_skill)) * 0.5
        if pad_skill == 0: pad_skill = 0.05
        ax4.set_xlim(min(actual_dist_skill) - pad_skill, max(actual_dist_skill) + pad_skill)

        # SHARED DECORATION & WRAP-UP
        # Diagonal break marks (//) for both rows
        d_x1, d_x2, d_y = 0.02, 0.01, 0.03
        kwargs = dict(color=text_color, clip_on=False, lw=1.5)

        for l_ax, r_ax in [(ax1, ax2), (ax3, ax4)]:
            l_ax.plot((1 - d_x1, 1 + d_x1), (-d_y, +d_y), transform=l_ax.transAxes, **kwargs)
            r_ax.plot((-d_x2, +d_x2), (-d_y, +d_y), transform=r_ax.transAxes, **kwargs)
            l_ax.tick_params(axis='x', which='both', top=False, bottom=True)
            r_ax.tick_params(axis='x', which='both', top=False, bottom=True)

        # Axes labels
        ax1.set_ylabel('Bootstrapped count', fontsize=label_fontsize, color=text_color)
        ax3.set_ylabel('Bootstrapped count', fontsize=label_fontsize, color=text_color)

        # Perfectly center the X-labels under the broken axes pairs
        ax1.text(1.5, -0.25, 'Euclidean Error (UMAP Units)', transform=ax1.transAxes,
                 ha='center', va='top', fontsize=label_fontsize, color=text_color)
        ax3.text(1.5, -0.25, 'Error Reduction Skill Score', transform=ax3.transAxes,
                 ha='center', va='top', fontsize=label_fontsize, color=text_color)

        # Final Layout & Export
        fig.suptitle('Model Performance', fontsize=title_fontsize, color=text_color, y=0.98)

        # Combine legends from top row proxy artists
        lines_labels = [ax.get_legend_handles_labels() for ax in [ax1, ax2]]
        lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        leg = ax2.legend(lines, labels, frameon=False, loc='best', fontsize=label_fontsize - 1)
        for text in leg.get_texts():
            text.set_color(text_color)

        plt.subplots_adjust(bottom=0.15, left=0.15, right=0.95, top=0.88)

        # Dynamic filename based on path metadata
        pkl_path = getattr(self, 'results_pkl_path', '')
        sex_mod = "male_mute_partner" if "male_mute_partner" in pkl_path else ("female" if "female" in pkl_path else "male")
        save_filename = f"cnn_permutation_test_{sex_mod}"

        if hasattr(self, '_handle_save'):
            self._handle_save(fig, save_filename, save_plot, output_dir, file_format)

        plt.show()

    def plot_feature_importance(self,
                                snr_threshold: float = 3.0,
                                cmap: str = _GLOBAL_CMAP,
                                error_bar_color: str = '#000000',
                                title_fontsize: int = 10,
                                label_fontsize: int = 8,
                                figsize: tuple = (4, 6),
                                save_plot: bool = False,
                                output_dir: Optional[str] = None,
                                file_format: str = 'svg') -> None:
        r"""
        Visualizes Post-Hoc Permutation feature importance with dynamic Signal-to-Noise Ratio (SNR) thresholding.

        This method adapts to the raw fold-wise error data from the CNN ablation study. It calculates
        the Mean Increase in Euclidean Error (Delta E) and the variance across cross-validation
        folds to determine feature significance relative to stochastic noise.

        SNR is defined as the mean performance drop divided by the standard deviation of that drop
        across folds ($SNR = \mu_{\Delta E} / \sigma_{\Delta E}$). Features failing the `snr_threshold`
        are visually de-emphasized by mapping them to the lowest value of the selected colormap.

        Parameters
        ----------
        snr_threshold : float, default 3.0
            The minimum Signal-to-Noise Ratio required for a feature to be considered
            statistically significant relative to permutation noise.
        cmap : str, default `figures.cmap` (currently 'inferno')
            Colormap applied to the bars of statistically significant features.
        error_bar_color : str, default '#000000'
            Color for the standard deviation error bars representing variance across folds.
        title_fontsize : int, default 10
            Font size for the plot title.
        label_fontsize : int, default 8
            Font size for the kinematic feature names on the Y-axis.
        figsize : tuple, default (4, 6)
            Figure dimensions in inches.
        save_plot : bool, default False
            If True, exports the figure to disk.
        output_dir : str, optional
            Path to save the figure.
        file_format : str, default 'svg'
            Format for the saved file.

        Returns
        -------
        None
            Displays the generated matplotlib figure and optionally saves it to disk.
        """

        # Access the nested CNN importance structure
        imp_data = self.data['feature_importance']

        # 1. Safely extract features using the CNN-generated ranking
        # These features WILL have the 'self.' prefix as intended
        feats = [f for f in imp_data['ranked_features'] if f != 'knockoff_probe']

        # 2. Extract means and stds from their respective sub-dictionaries
        # Pull per-feature means and stds from their dedicated 'means'/'stds' keys.
        actual_means = np.array([imp_data['means'][f] for f in feats])
        actual_stds = np.array([imp_data['stds'][f] for f in feats])

        # Dynamic SNR Calculation & Thresholding
        snrs = np.where(actual_stds > 1e-8, actual_means / actual_stds, 0.0)
        significant_mask = snrs > snr_threshold

        # Color Mapping Logic
        norm = plt.Normalize(vmin=actual_means.min(), vmax=actual_means.max())
        mapper = plt.get_cmap(cmap)
        lowest_color = mapper(0.0)

        # Significant features get the map; non-significant get the lowest 'glow'
        bar_colors = [mapper(norm(val)) if sig else lowest_color
                      for val, sig in zip(actual_means, significant_mask)]

        # Visualization
        fig, ax = plt.subplots(figsize=figsize, facecolor='#FFFFFF')
        ax.set_facecolor('#FFFFFF')
        ax.grid(False)

        y_pos = np.arange(len(feats))

        # Render the horizontal importance bars
        ax.barh(y_pos, actual_means, xerr=actual_stds, color=bar_colors,
                edgecolor='#000000', ecolor=error_bar_color, capsize=3)

        # Reference line at zero
        ax.axvline(0, color='#000000', linewidth=1, linestyle='-', alpha=0.5)

        ax.set_yticks(y_pos)
        ax.invert_yaxis()

        # Update labels (preserving 'self.' prefixes) and de-emphasize non-significant ones
        yticklabels = ax.set_yticklabels(feats, fontsize=label_fontsize)
        for idx, label in enumerate(yticklabels):
            label.set_color("#202020" if significant_mask[idx] else "#808080")

        ax.set_title('Global Permutation Kinematic Importance', fontsize=title_fontsize, color='#000000', pad=20)
        ax.set_xlabel(r'Mean Increase in Euclidean Error ($\Delta E$)', fontsize=label_fontsize + 1, color="#202020")

        # Spine and Tick styling
        ax.tick_params(colors="#202020", which='both')
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['left', 'bottom']:
            ax.spines[spine].set_color('#000000')

        plt.tight_layout()

        # Save Logic (with sex-specific naming)
        pkl_path = getattr(self, 'results_pkl_path', '')
        if "male_mute_partner" in pkl_path:
            sex_mod = "male_mute_partner"
        elif "female" in pkl_path:
            sex_mod = "female"
        else:
            sex_mod = "male"

        save_filename = f"cnn_feature_importance_{sex_mod}"

        if hasattr(self, '_handle_save'):
            self._handle_save(fig, save_filename, save_plot, output_dir, file_format)

        plt.show()

    def plot_spatial_precision_grid(self,
                                    plot_type: str = 'density',
                                    n_patches: int = 20,
                                    patch_size: Optional[float] = None,
                                    min_samples: int = 50,
                                    bg_pt_color: str = '#E0E0E0',
                                    peak_pt_color: str = '#00FFFF',
                                    square_edge_color: str = '#000000',
                                    panel_fontsize: int = 9,
                                    figsize_unit: float = 3.0,
                                    grid_shape: Optional[tuple] = None,
                                    save_plot: bool = False,
                                    output_dir: Optional[str] = None,
                                    file_format: str = 'svg') -> None:
        """
        Generates a tiled grid of 'Hero Shot' panels focusing on representative regions across the manifold.

        This version uses K-Means clustering to identify patch centers. Unlike a pure density
        search, K-Means ensures that isolated clusters (like small UMAP islands) are
        guaranteed a representative panel, as centroids are distributed to minimize
        global spatial variance regardless of local point density.

        Each panel visualizes the relationship between the true geometric center of a
        sampled behavioral neighborhood and the peak (mode) of the model's predicted
        coordinates, calculating the Euclidean bias (Delta d).

        Parameters
        ----------
        plot_type : str, default 'density'
            The visualization style for the prediction distribution ('density' or 'contour').
        n_patches : int, default 20
            Euclidean-manifold only: total number of K-means-derived
            patch panels. Ignored on the torus branch, which uses a
            uniform ``grid_shape`` instead.
        patch_size : float, optional
            Side length of the square sampling window in manifold
            units. If ``None``, defaults to ``2.5`` on Euclidean
            manifolds (unbounded UMAP plane) and to
            ``0.20 * manifold_period`` on a torus manifold (so the
            patch is ~20% of the unit cell on each side).
        min_samples : int, default 50
            Minimum number of data points required inside a patch.
        bg_pt_color : str, default '#E0E0E0'
            Hex code for the background global UMAP coordinates.
        peak_pt_color : str, default '#00FFFF'
            Color of the crosshair ('+') marking the peak density of predictions.
        square_edge_color : str, default '#000000'
            Color of the border representing the true spatial bin.
        panel_fontsize : int, default 9
            Font size for the Bias score in the subplot titles.
        figsize_unit : float, default 3.0
            Inches per subplot.
        grid_shape : tuple of (n_cols, n_rows), optional
            Torus-manifold only: shape of the uniform patch grid
            covering the unit cell. Defaults to ``(4, 4)`` when on
            torus and not supplied. Ignored on Euclidean manifolds
            (which use K-means selection from ``n_patches``).
        save_plot : bool, default False
            If True, exports the generated figure to disk.
        output_dir : str, optional
            Directory for exported files.
        file_format : str, default 'svg'
            Image format for saving (e.g., 'svg', 'png', 'pdf').

        Returns
        -------
        None
            Displays the generated matplotlib figure and optionally saves it to disk.
        """

        if self.manifold_metric == 'torus':
            self._plot_spatial_precision_grid_torus(
                plot_type=plot_type,
                grid_shape=grid_shape,
                patch_size=patch_size,
                min_samples=min_samples,
                bg_pt_color=bg_pt_color,
                peak_pt_color=peak_pt_color,
                square_edge_color=square_edge_color,
                panel_fontsize=panel_fontsize,
                figsize_unit=figsize_unit,
                save_plot=save_plot,
                output_dir=output_dir,
                file_format=file_format,
            )
            return
        if patch_size is None:
            patch_size = 2.5

        # 1. Data Preparation (Aggregating Across CNN Folds)
        # Filter out placeholder folds emitted by the runner's
        # `restrict_to_fold_indices` recovery path — they carry only
        # `Y_true` / `test_indices` and would `KeyError` on
        # `Y_pred_actual` / `error_*` lookups below.
        cv_folds = [f for f in self.data['cross_validation'] if not f.get('skipped')]
        # Explicitly wrapping in np.array to handle potential JAX DeviceArrays
        Y_true = np.vstack([np.array(f['Y_true']) for f in cv_folds])
        Y_pred = np.vstack([np.array(f['Y_pred_actual']) for f in cv_folds])

        # 2. Custom White-Base derived from the global cmap
        base_cmap = plt.cm.get_cmap(_GLOBAL_CMAP)
        cmap_colors = base_cmap(np.linspace(0, 1, 256))
        white = np.array([1, 1, 1, 1])
        cmap_colors[:25, :] = np.linspace(white, cmap_colors[25, :], 25)
        white_inferno = ListedColormap(cmap_colors)

        # 3. K-Means Guided Search (Ensures Island Capture)
        print(f"Identifying {n_patches} representative neighborhoods using K-Means...")
        kmeans = KMeans(n_clusters=n_patches, n_init=10, random_state=42)
        kmeans.fit(Y_true)
        centers = kmeans.cluster_centers_

        selected_patches = []
        half_s = patch_size / 2.0

        for cx, cy in centers:
            mask = (Y_true[:, 0] >= cx - half_s) & (Y_true[:, 0] <= cx + half_s) & \
                   (Y_true[:, 1] >= cy - half_s) & (Y_true[:, 1] <= cy + half_s)

            indices = np.where(mask)[0]
            density = len(indices)

            if density >= min_samples:
                selected_patches.append({'center': (cx, cy), 'density': density, 'indices': indices})
            else:
                # Wrap-aware nearest-neighbour to the K-means centre.
                # On torus this picks up neighbours from across the
                # wrap boundary so a centre near the period edge isn't
                # silently snapped to a far-side point.
                center_arr = np.asarray([[cx, cy]], dtype=Y_true.dtype)
                dist_to_center = pairwise_distance(
                    Y_true, center_arr,
                    metric=self.manifold_metric, period=self.manifold_period,
                )
                nearest_idx = np.argmin(dist_to_center)
                nx, ny = Y_true[nearest_idx]

                new_mask = (Y_true[:, 0] >= nx - half_s) & (Y_true[:, 0] <= nx + half_s) & \
                           (Y_true[:, 1] >= ny - half_s) & (Y_true[:, 1] <= ny + half_s)

                selected_patches.append({
                    'center': (nx, ny),
                    'density': np.sum(new_mask),
                    'indices': np.where(new_mask)[0]
                })

        selected_patches = sorted(selected_patches, key=lambda x: x['density'], reverse=True)

        # 4. Tiled Visualization
        n = len(selected_patches)
        cols = 5
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * figsize_unit, rows * figsize_unit), facecolor='#FFFFFF')
        axes = axes.flatten()

        global_x_min, global_x_max = Y_true[:, 0].min() - 1, Y_true[:, 0].max() + 1
        global_y_min, global_y_max = Y_true[:, 1].min() - 1, Y_true[:, 1].max() + 1

        for idx, patch in enumerate(selected_patches):
            ax = axes[idx]
            ax.set_facecolor('#FFFFFF')
            ax.grid(False)
            cx, cy = patch['center']

            # Background Global Points
            ax.scatter(Y_true[:, 0], Y_true[:, 1], s=0.5, c=bg_pt_color, alpha=0.2, zorder=1)

            # True Spatial Bin Border
            rect = Rectangle((cx - half_s, cy - half_s), patch_size, patch_size,
                             linewidth=1.5, edgecolor=square_edge_color, facecolor='none',
                             linestyle='--', zorder=10)
            ax.add_patch(rect)

            # Local Prediction Density
            p_subset = Y_pred[patch['indices']]
            # gaussian_kde needs >= 2 points with a non-degenerate covariance;
            # a single-point or all-identical patch raises "singular data
            # covariance matrix" and would crash the whole figure. Skip the
            # density overlay (the scatter + bin border are already drawn) for
            # such degenerate patches.
            try:
                kde = gaussian_kde(p_subset.T)
            except (np.linalg.LinAlgError, ValueError):
                ax.set_title(f"n={p_subset.shape[0]} (no KDE)",
                             fontsize=panel_fontsize, color='#000000')
                ax.set_axis_off()
                ax.set_xlim(global_x_min, global_x_max)
                ax.set_ylim(global_y_min, global_y_max)
                continue

            # Alignment Grid
            xi_grid, yi_grid = np.mgrid[global_x_min:global_x_max:100j, global_y_min:global_y_max:100j]
            zi_grid = kde(np.vstack([xi_grid.flatten(), yi_grid.flatten()])).reshape(xi_grid.shape)

            peak_idx = np.unravel_index(np.argmax(zi_grid), zi_grid.shape)
            peak_coord = (xi_grid[peak_idx], yi_grid[peak_idx])
            # Wrap-aware bias score so panels near the wrap boundary
            # report the actual on-manifold distance rather than the
            # `period - epsilon` long-way-around alternative.
            dist = float(pairwise_distance(
                np.asarray([peak_coord]), np.asarray([[cx, cy]]),
                metric=self.manifold_metric, period=self.manifold_period,
            )[0])

            if plot_type.lower() == 'contour':
                z_flat_sorted = np.sort(zi_grid.flatten())[::-1]
                z_cumsum = np.cumsum(z_flat_sorted) / np.sum(z_flat_sorted)
                # De-duplicate so the levels are strictly increasing: a
                # degenerate (near-flat) density maps several percentiles to the
                # same z value, which makes `ax.contour` raise "Contour levels
                # must be increasing".
                levels = sorted({z_flat_sorted[np.searchsorted(z_cumsum, p)] for p in [0.50, 0.75, 0.90]})
                if levels:
                    ax.contour(xi_grid, yi_grid, zi_grid, levels=levels, cmap=white_inferno, linewidths=1.2, zorder=5)

            elif plot_type.lower() == 'density':
                zi_norm = (zi_grid - zi_grid.min()) / (zi_grid.max() - zi_grid.min() + 1e-10)
                ax.imshow(zi_norm.T, cmap=white_inferno, interpolation='bilinear', origin='lower',
                          extent=[global_x_min, global_x_max, global_y_min, global_y_max],
                          aspect='auto', alpha=0.7, zorder=2)

            # Predicted Peak Marker
            ax.scatter(peak_coord[0], peak_coord[1], marker='+', c=peak_pt_color, s=60, linewidth=2, zorder=15)

            # Metadata Display
            ax.set_title(rf"Bias $\Delta d$: {dist:.3f}", fontsize=panel_fontsize, color='#000000')

            ax.set_axis_off()
            ax.set_xlim(global_x_min, global_x_max)
            ax.set_ylim(global_y_min, global_y_max)

        for i in range(n, len(axes)): fig.delaxes(axes[i])
        plt.tight_layout()

        # 5. Updated Save Logic
        pkl_path = getattr(self, 'results_pkl_path', '')
        if "male_mute_partner" in pkl_path:
            sex_mod = "male_mute_partner"
        elif "female" in pkl_path:
            sex_mod = "female"
        else:
            sex_mod = "male"

        save_filename = f"cnn_precision_grid_{sex_mod}_{plot_type}"

        if hasattr(self, '_handle_save'):
            self._handle_save(fig, save_filename, save_plot, output_dir, file_format)

        plt.show()

    def _plot_spatial_precision_grid_torus(
            self,
            plot_type: str,
            grid_shape: Optional[tuple],
            patch_size: Optional[float],
            min_samples: int,
            bg_pt_color: str,
            peak_pt_color: str,
            square_edge_color: str,
            panel_fontsize: int,
            figsize_unit: float,
            save_plot: bool,
            output_dir: Optional[str],
            file_format: str,
    ) -> None:
        """
        Torus-manifold counterpart to ``plot_spatial_precision_grid``.

        Renders a uniform ``grid_shape`` grid of square patches on the
        unit cell. Each panel uses:
          * a wrap-aware rectangular sample mask (torus distance
            ``(p - c + period/2) % period - period/2`` per axis);
          * a 3x3-tiled ``gaussian_kde`` so the periodic boundary of
            the predicted-density estimate is honoured;
          * a 9-tiled dashed square outline so a patch crossing the
            wrap boundary shows its wrapped half on the opposite
            edge;
          * a wrap-aware bias distance to the predicted-density peak
            (already in the project as
            ``manifold_metric.pairwise_distance(..., metric='torus',
            period=...)``).

        Display axes are fixed to ``[0, period]^2`` with no padding;
        rows are reversed so the top of the figure is the top of the
        torus (matching the convention readers expect from a 2D
        coordinate plane).
        """

        period = float(self.manifold_period)

        # defaults specific to the torus branch
        if patch_size is None:
            patch_size = 0.20 * period
        if grid_shape is None:
            grid_shape = (4, 4)
        nx, ny = int(grid_shape[0]), int(grid_shape[1])
        half_s = patch_size / 2.0

        # pool predictions across successfully-fit folds
        cv_folds = [f for f in self.data['cross_validation']
                    if not f.get('skipped')]
        Y_true = np.vstack([np.asarray(f['Y_true']) for f in cv_folds])
        Y_pred = np.vstack([np.asarray(f['Y_pred_actual']) for f in cv_folds])

        # white-base inferno cmap (matches Euclidean branch)
        base_cmap = plt.cm.get_cmap(_GLOBAL_CMAP)
        cmap_colors = base_cmap(np.linspace(0, 1, 256))
        white = np.array([1, 1, 1, 1])
        cmap_colors[:25, :] = np.linspace(white, cmap_colors[25, :], 25)
        white_inferno = ListedColormap(cmap_colors)

        # uniform grid centres on the unit cell
        x_centres = (np.arange(nx) + 0.5) / nx * period
        y_centres = (np.arange(ny) + 0.5) / ny * period
        # Row-major top-to-bottom: reverse y so figure-top = torus-top.
        centres = [(cx, cy) for cy in y_centres for cx in x_centres]

        # figure
        # No explicit ``dpi=`` here so the inline-notebook display
        # matches the Euclidean branch's call (which also omits dpi
        # and inherits the inline backend's default ~100). Setting
        # dpi=300 here caused the inline preview to render at 3x
        # the Euclidean equivalent's pixel count for the same
        # ``figsize_unit``. SVG / PNG saves still honour
        # ``savefig.dpi`` from the project style.
        fig, axes = plt.subplots(
            ny, nx,
            figsize=(figsize_unit * nx, figsize_unit * ny + 0.3),
            facecolor='#FFFFFF',
        )
        axes = np.atleast_2d(axes)

        # Single KDE evaluation grid over the unit cell.
        grid_n = 120
        xi, yi = np.mgrid[0:period:complex(0, grid_n),
                          0:period:complex(0, grid_n)]
        grid_pts = np.vstack([xi.flatten(), yi.flatten()])

        # 3x3 tile offsets used both by the KDE and by the wrap-
        # spanning patch-outline renderer.
        tile_offsets = [(dx * period, dy * period)
                        for dx in (-1, 0, 1) for dy in (-1, 0, 1)]

        for idx, (cx, cy) in enumerate(centres):
            col_idx = idx % nx
            # Reverse rows so figure-top is torus-top.
            row_idx = (ny - 1) - (idx // nx)
            ax = axes[row_idx, col_idx]
            ax.set_facecolor('#FFFFFF')

            # Wrap-aware rectangular mask.
            dx = (Y_true[:, 0] - cx + period / 2.0) % period - period / 2.0
            dy = (Y_true[:, 1] - cy + period / 2.0) % period - period / 2.0
            mask = (np.abs(dx) <= half_s) & (np.abs(dy) <= half_s)
            n_in = int(mask.sum())

            # ``rasterized=True`` embeds this scatter as a single
            # raster image inside the SVG instead of one <circle>
            # element per point. With ~80k points x ~25 panels in
            # the default 5x5 grid that's the difference between a
            # ~3 MB SVG and a 300+ MB one. Saved-resolution is still
            # ``savefig.dpi`` from the project mplstyle (300), so
            # the rasterized layer stays crisp on export.
            ax.scatter(Y_true[:, 0], Y_true[:, 1], s=0.3,
                       c=bg_pt_color, alpha=0.20, zorder=1,
                       rasterized=True)

            # Always draw the 9-tiled patch outline so wrap pieces show.
            for ox, oy in tile_offsets:
                ax.add_patch(Rectangle(
                    (cx - half_s + ox, cy - half_s + oy),
                    patch_size, patch_size,
                    fill=False, edgecolor=square_edge_color,
                    linestyle='--', lw=0.8, zorder=10,
                ))

            if n_in < min_samples:
                ax.set_title(f"only {n_in} pts", fontsize=panel_fontsize - 2)
                ax.set_xlim(0, period); ax.set_ylim(0, period)
                ax.set_aspect('equal')
                ax.set_xticks([]); ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_edgecolor('#B0B0B0'); s.set_linewidth(0.4)
                continue

            # 3x3 tiled KDE on the prediction subset.
            p_subset = Y_pred[mask]
            tiled = np.vstack([p_subset + np.array(off, dtype=float)
                               for off in tile_offsets])
            # Identical / collinear predictions give a singular covariance and
            # crash gaussian_kde; skip the density overlay for such patches
            # (tiling a single point 9x is still rank-deficient, so the
            # n_in >= min_samples guard above does not catch this case).
            try:
                kde = gaussian_kde(tiled.T)
            except (np.linalg.LinAlgError, ValueError):
                ax.set_title(f"n={n_in} (no KDE)", fontsize=panel_fontsize - 2)
                ax.set_xlim(0, period); ax.set_ylim(0, period)
                ax.set_aspect('equal')
                ax.set_xticks([]); ax.set_yticks([])
                continue
            zi = kde(grid_pts).reshape(xi.shape)
            zi_norm = (zi - zi.min()) / (zi.max() - zi.min() + 1e-12)

            if plot_type.lower() == 'contour':
                z_flat_sorted = np.sort(zi.flatten())[::-1]
                z_cumsum = np.cumsum(z_flat_sorted) / np.sum(z_flat_sorted)
                # De-duplicate so levels are strictly increasing (see the
                # euclidean precision grid for the rationale).
                levels = sorted({z_flat_sorted[np.searchsorted(z_cumsum, p)]
                                 for p in [0.50, 0.75, 0.90]})
                if levels:
                    ax.contour(xi, yi, zi, levels=levels, cmap=white_inferno,
                               linewidths=1.0, zorder=5)
            else:
                ax.imshow(zi_norm.T, cmap=white_inferno,
                          interpolation='bilinear', origin='lower',
                          extent=(0, period, 0, period),
                          aspect='equal', alpha=0.78, zorder=2)

            # Wrap-aware peak coord + bias distance.
            peak_idx = np.unravel_index(np.argmax(zi), zi.shape)
            peak_xy = np.asarray([[xi[peak_idx], yi[peak_idx]]],
                                 dtype=float)
            bias = float(pairwise_distance(
                peak_xy, np.asarray([[cx, cy]], dtype=float),
                metric=self.manifold_metric, period=period,
            )[0])
            ax.scatter(peak_xy[0, 0], peak_xy[0, 1], marker='+',
                       c=peak_pt_color, s=35, lw=1.5, zorder=15)

            ax.set_title(rf"$\Delta d$={bias:.2f} ({n_in})",
                         fontsize=panel_fontsize - 2, pad=2)
            ax.set_xlim(0, period); ax.set_ylim(0, period)
            ax.set_aspect('equal')
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_edgecolor('#B0B0B0'); s.set_linewidth(0.4)

        fig.suptitle(
            f"CNN torus precision grid ({nx}x{ny} patches, "
            f"side={patch_size:.2f} of period={period:.2f})",
            fontsize=panel_fontsize - 1, y=0.995,
        )
        fig.subplots_adjust(left=0.02, right=0.99, top=0.93,
                            bottom=0.02, wspace=0.08, hspace=0.18)

        pkl_path = getattr(self, 'results_pkl_path', '')
        if "male_mute_partner" in pkl_path:
            sex_mod = "male_mute_partner"
        elif "female" in pkl_path:
            sex_mod = "female"
        else:
            sex_mod = "male"
        save_filename = f"cnn_precision_grid_{sex_mod}_torus_{plot_type}"

        if hasattr(self, '_handle_save'):
            self._handle_save(fig, save_filename, save_plot, output_dir,
                              file_format)

        plt.show()

    def plot_error_landscape(self,
                             gridsize: int = 30,
                             cmap: str = _GLOBAL_CMAP,
                             diff_cmap: str = 'RdBu_r',  # Reverting to RdBu_r for standard diverging look
                             vmax_percentile: float = 95.0,
                             title_fontsize: int = 10,
                             label_fontsize: int = 8,
                             figsize: tuple = (12, 5),
                             save_plot: bool = False,
                             output_dir: Optional[str] = None,
                             file_format: str = 'svg') -> None:
        """
        Visualizes the spatial distribution of prediction errors and error reduction for the CNN.

        This method generates a two-panel figure using the cross-validation results:
        1. Actual Error Landscape: A hexbin plot where colors represent the mean
           Euclidean prediction error in that region.
        2. Error Reduction: A diverging hexbin plot showing the difference
           between the Null (Model-Free Prior) error and the Actual Model error.
           Positive values indicate regions where behavioral features significantly
           reduced the error compared to random density draws.

        Parameters
        ----------
        gridsize : int, default 30
            The number of hexagons in the x-direction. Controls spatial resolution.
        cmap : str, default `figures.cmap` (currently 'inferno')
            Colormap for the absolute error landscape.
        diff_cmap : str, default 'RdBu_r'
            Diverging colormap for the Error Reduction panel.
        vmax_percentile : float, default 95.0
            The percentile at which to cap the maximum color value for visual clarity.
        title_fontsize : int, default 10
            Font size for the plot titles.
        label_fontsize : int, default 8
            Font size for the axis labels.
        figsize : tuple, default (12, 5)
            Figure dimensions in inches.
        save_plot : bool, default False
            If True, exports the figure to disk.
        output_dir : str, optional
            Path to save the figure.
        file_format : str, default 'svg'
            Format for the saved file.
        """

        # 1. Data Preparation (Aggregating Across CNN Folds)
        # Filter out placeholder folds emitted by the runner's
        # `restrict_to_fold_indices` recovery path — they carry only
        # `Y_true` / `test_indices` and would `KeyError` on
        # `Y_pred_actual` / `error_*` lookups below.
        cv_folds = [f for f in self.data['cross_validation'] if not f.get('skipped')]

        # Pull true coordinates and predictions from both Actual and Model-Free Null
        Y_true = np.vstack([np.array(f['Y_true']) for f in cv_folds])
        Y_pred_act = np.vstack([np.array(f['Y_pred_actual']) for f in cv_folds])
        Y_pred_null = np.vstack([np.array(f['Y_pred_null_model_free']) for f in cv_folds])

        # Calculate Euclidean errors point-by-point
        # Wrap-aware per-sample error magnitudes — euclidean on flat
        # manifolds, shortest-wrap on torus.
        errors_act = pairwise_distance(
            Y_true, Y_pred_act,
            metric=self.manifold_metric, period=self.manifold_period,
        )
        errors_null = pairwise_distance(
            Y_true, Y_pred_null,
            metric=self.manifold_metric, period=self.manifold_period,
        )

        # Delta E: Positive means the Model is better (lower error) than the Null
        error_diff = errors_null - errors_act

        # 2. Visualization
        fig, axes = plt.subplots(1, 2, figsize=figsize, facecolor='#FFFFFF')

        # Panel 1: Absolute Error
        ax1 = axes[0]
        vmax_act = np.percentile(errors_act, vmax_percentile)
        hb1 = ax1.hexbin(Y_true[:, 0], Y_true[:, 1], C=errors_act, reduce_C_function=np.mean,
                         gridsize=gridsize, cmap=cmap, edgecolors='none', vmin=0, vmax=vmax_act)
        ax1.set_title('Global Error Landscape (Actual Model)', fontsize=title_fontsize, color='#000000', pad=15)

        cbar1 = fig.colorbar(hb1, ax=ax1, pad=0.02, shrink=0.8)
        cbar1.set_label('Mean Euclidean Error', color='#202020', rotation=270, labelpad=20, fontsize=label_fontsize)

        # Panel 2: Error Reduction (Contrast)
        ax2 = axes[1]
        vmax_diff = np.percentile(np.abs(error_diff), vmax_percentile)
        hb2 = ax2.hexbin(Y_true[:, 0], Y_true[:, 1], C=error_diff, reduce_C_function=np.mean,
                         gridsize=gridsize, cmap=diff_cmap, edgecolors='none', vmin=-vmax_diff, vmax=vmax_diff)
        ax2.set_title('Error Reduction vs. Null Model', fontsize=title_fontsize, color='#000000', pad=15)

        cbar2 = fig.colorbar(hb2, ax=ax2, pad=0.02, shrink=0.8)
        cbar2.set_label(r'Error Reduction ($\Delta E$)', color='#202020', rotation=270, labelpad=20, fontsize=label_fontsize)

        # Axis-label prefix reflects the upstream latent space:
        # ``torus`` -> the QLVM latent; otherwise the VAE / UMAP plane.
        dim_prefix = 'QLVM' if self.manifold_metric == 'torus' else 'UMAP'

        # Formatting all axes
        for ax in axes:
            ax.set_facecolor('#FFFFFF')
            ax.grid(False)
            ax.set_aspect('equal')

            ax.set_xlabel(f'{dim_prefix} Dimension 1', fontsize=label_fontsize, color='#202020')
            ax.set_ylabel(f'{dim_prefix} Dimension 2', fontsize=label_fontsize, color='#202020')
            ax.tick_params(colors='#202020', which='both', labelsize=label_fontsize - 1)

            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)
            for spine in ['left', 'bottom']:
                ax.spines[spine].set_color('#000000')

        for cb in [cbar1, cbar2]:
            cb.ax.yaxis.set_tick_params(color='#202020', labelcolor='#202020', labelsize=label_fontsize - 1)
            cb.outline.set_edgecolor('#000000')

        plt.tight_layout()

        # 3. Save Logic
        pkl_path = getattr(self, 'results_pkl_path', '')
        if "male_mute_partner" in pkl_path:
            sex_mod = "male_mute_partner"
        elif "female" in pkl_path:
            sex_mod = "female"
        else:
            sex_mod = "male"

        save_filename = f"cnn_error_landscape_{sex_mod}"

        if hasattr(self, '_handle_save'):
            self._handle_save(fig, save_filename, save_plot, output_dir, file_format)

        plt.show()

    def plot_regional_saliency_inset(self,
                                     region_key: str,
                                     category_name: Optional[str] = None,
                                     prediction_plot_type: str = 'contour',
                                     highlight_color: Optional[str] = None,
                                     null_color: str = '#D3D3D3',
                                     cmap: str = cmr.fusion_r,
                                     figsize: tuple = (18, 9),
                                     radius: Optional[float] = None,
                                     smoothing_sigma: float = 60.0,
                                     save_plot: bool = False,
                                     output_dir: Optional[str] = None,
                                     file_format: str = 'svg') -> None:
        """
        Visualizes regional manifold dynamics for one pre-computed
        saliency region. The region is defined by the (centroid,
        radius) pair stored alongside the saliency map -- the same
        circular region the saliency runner used at training time --
        so the spatial filter, the contrastive map, and the displayed
        region outline all refer to identically the same set of
        samples.

        Polygon-based region definitions were dropped: the saliency
        runner now emits circular regions exclusively, and the
        polygon-vs-circle mismatch between the function signature
        and the data caused silent misalignment between the filtered
        samples and the saliency map. Pass a region_key that already
        exists in self.data['saliency_maps']; centroid + radius are
        read straight from that entry.

        Parameters
        ----------
        region_key : str
            The internal identifier used to look up the pre-computed
            saliency entry stored in ``self.data['saliency_maps']``.
            The entry must carry ``centroid`` (shape (2,)) and
            ``radius`` (float) alongside ``contrastive_saliency``.
            Also used as the display title if ``category_name`` is
            None.
        category_name : str, optional
            Human-readable plot title (e.g., 'Category 3: Complex').
            If None, ``region_key`` is used.
        prediction_plot_type : str, default 'contour'
            Visualization style for the predicted UMAP coordinates.
            Options: ['contour', 'density', 'hexbin', 'scatter'].
        highlight_color : str, optional
            Color for the region border, the peak density marker,
            and the model's error distribution. If None, falls back
            to ``self.default_color`` (set in __init__).
        null_color : str, default '#D3D3D3'
            Color for the model-free null distribution in the error
            inset.
        cmap : str, default ``cmr.fusion_r``
            Diverging colormap for the contrastive saliency heatmap.
        figsize : tuple, default (18, 9)
            Figure dimensions in inches.
        radius : float, optional
            Override the region radius (in manifold units) used for
            both the wrap-aware sample selection AND the displayed
            circle outline. Defaults to ``None``, in which case the
            radius stored with the saliency entry (the one the
            saliency runner actually used) is read directly. The
            stored saliency map itself was computed at the *stored*
            radius, so overriding here changes which samples are
            included in the per-trial scatter / KDE / error inset
            but does not retrain the saliency map; expect a console
            note when an override is in effect.
        smoothing_sigma : float, default 60.0
            Gaussian smoothing standard deviation, in time bins,
            applied to the contrastive saliency map along the time
            axis before cubic-spline interpolation for display. At
            the project's 150 fps camera rate, the default ~= 400 ms.
            Set to ``0`` to disable smoothing entirely (the heatmap
            renders the raw per-bin contrastive values, which can be
            visibly noisy). Smaller values preserve more high-
            frequency structure; larger values further integrate
            bin-to-bin variation.
        save_plot : bool, default False
            If True, saves the figure to ``output_dir``.
        output_dir : str, optional
            Export directory. Defaults to the visualizer's save_dir.
        file_format : str, default 'svg'
            File format for the export.

        Returns
        -------
        None
        """

        # 0. COLOR MANAGEMENT
        # Reliably pull the color index 0 mapped in __init__
        if highlight_color is None:
            highlight_color = getattr(self, 'default_color', '#9AC0CD')

        # 1. DISPLAY TITLE
        display_title = category_name if category_name is not None else region_key

        # 2. DATA AGGREGATION
        imp_data = self.data['feature_importance']
        best_fold_idx = imp_data['best_fold_idx']
        fold_res = self.data['cross_validation'][best_fold_idx]

        Y_te = np.array(fold_res['Y_true'])
        Y_pred = np.array(fold_res['Y_pred_actual'])
        Y_pred_null = np.array(fold_res['Y_pred_null_model_free'])

        features_list = self.metadata['features_list']
        num_bins = self.metadata['n_time_bins']
        num_features = len(features_list)

        # 3. SALIENCY EXTRACTION + REGION DEFINITION
        # Region geometry now comes from the stored saliency entry --
        # the same circle the saliency runner used to assemble the
        # contrastive map.
        if 'saliency_maps' not in self.data or region_key not in self.data['saliency_maps']:
            raise NotImplementedError(
                f"No pre-computed saliency map stored for region '{region_key}'. "
                f"Re-compute saliency during training and rerun, or pass a "
                f"region_key already present in self.data['saliency_maps'] "
                f"(available: {list(self.data.get('saliency_maps', {}).keys())})."
            )
        sal_entry = self.data['saliency_maps'][region_key]
        centroid = np.asarray(sal_entry['centroid'], dtype=float).reshape(2)
        stored_radius = float(sal_entry['radius'])
        raw_saliency = np.asarray(sal_entry['contrastive_saliency'])
        contrastive_map = np.mean(raw_saliency, axis=0)
        if radius is None:
            effective_radius = stored_radius
            print(
                f"   > Extracting pre-computed saliency for {region_key} "
                f"(centroid={centroid.tolist()}, radius={effective_radius:.3f})"
            )
        else:
            effective_radius = float(radius)
            print(
                f"   > Extracting pre-computed saliency for {region_key} "
                f"(centroid={centroid.tolist()}); USING radius override "
                f"{effective_radius:.3f} for sample selection + display "
                f"(stored radius was {stored_radius:.3f}; the saliency "
                f"map itself was computed at the stored radius)."
            )
        radius = effective_radius

        # 4. SPATIAL FILTERING (circle, wrap-aware on torus)
        # Distances are computed with the same manifold metric the
        # CNN used at training so torus wrap is honoured automatically
        # when applicable.
        distances = pairwise_distance(
            Y_te, centroid.reshape(1, 2),
            metric=self.manifold_metric, period=self.manifold_period,
        ).ravel()
        r_mask = distances <= radius
        r_idx = np.where(r_mask)[0]

        if len(r_idx) < 3:
            print(f"Warning: Region '{region_key}' contains only {len(r_idx)} samples. Skipping.")
            return

        # Wrap-aware per-trial error magnitudes for the in-region
        # subset; matches the metric used by the model's loss / R^2.
        err_actual = pairwise_distance(
            Y_te[r_idx], Y_pred[r_idx],
            metric=self.manifold_metric, period=self.manifold_period,
        )
        err_null = pairwise_distance(
            Y_te[r_idx], Y_pred_null[r_idx],
            metric=self.manifold_metric, period=self.manifold_period,
        )

        # 5. VISUALIZATION PIPELINE
        text_color = '#000000'
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, facecolor='#FFFFFF',
                                       gridspec_kw={'width_ratios': [1, 1.2]})

        # BRUTE-FORCE Axes Visibility (Overrides global style stripping)
        for ax in (ax1, ax2):
            ax.set_axis_on()
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor(text_color)
                spine.set_linewidth(1.0)

        base_cmap = plt.cm.get_cmap(_GLOBAL_CMAP)
        cmap_colors = base_cmap(np.linspace(0, 1, 256))
        cmap_colors[:25, :] = np.linspace(np.array([1, 1, 1, 1]), cmap_colors[25, :], 25)
        white_inferno = ListedColormap(cmap_colors)

        # PANEL 1: Manifold Context
        ax1.set_facecolor('#FFFFFF')
        ax1.set_title(f"UMAP Context: {display_title}", fontsize=14, color=text_color, pad=15)

        # Background scatter (both the out-of-region grey dots and
        # the in-region highlighted dots) intentionally omitted --
        # the density layer + dashed circle outline carry the
        # spatial story on their own and the per-sample scatter
        # blew the SVG up to tens of MB without adding information.
        y_p_region = Y_pred[r_idx]

        if prediction_plot_type.lower() == 'contour':
            try:
                kde = gaussian_kde(y_p_region.T)
                xx, yy = np.mgrid[Y_te[:, 0].min() - 2:Y_te[:, 0].max() + 2:100j,
                Y_te[:, 1].min() - 2:Y_te[:, 1].max() + 2:100j]
                zz = kde(np.vstack([xx.flatten(), yy.flatten()])).reshape(xx.shape)
                ax1.contour(xx, yy, zz, levels=3, cmap=white_inferno, linewidths=2.0, zorder=3)
                peak_idx = np.unravel_index(np.argmax(zz), zz.shape)
                ax1.scatter(xx[peak_idx], yy[peak_idx], marker='x', color=highlight_color, s=120, linewidths=3, zorder=4)
            except Exception:
                ax1.scatter(y_p_region[:, 0], y_p_region[:, 1], c=highlight_color, s=15, alpha=0.6, edgecolors='none', zorder=2)

        elif prediction_plot_type.lower() == 'density':
            try:
                kde = gaussian_kde(y_p_region.T)
                N_GRID = 300
                x_min_plot, x_max_plot = Y_te[:, 0].min() - 2, Y_te[:, 0].max() + 2
                y_min_plot, y_max_plot = Y_te[:, 1].min() - 2, Y_te[:, 1].max() + 2
                xi, yi = np.linspace(x_min_plot, x_max_plot, N_GRID), np.linspace(y_min_plot, y_max_plot, N_GRID)
                xx, yy = np.meshgrid(xi, yi)
                zz = kde(np.vstack([xx.flatten(), yy.flatten()])).reshape(xx.shape)
                zz_norm = (zz - zz.min()) / (zz.max() - zz.min())

                max_idx = np.unravel_index(np.argmax(zz), zz.shape)
                peak_x, peak_y = xx[max_idx], yy[max_idx]

                ax1.imshow(zz_norm, cmap=white_inferno, interpolation='bilinear', origin='lower',
                           extent=[x_min_plot, x_max_plot, y_min_plot, y_max_plot], aspect='auto', zorder=0)
                ax1.scatter([peak_x], [peak_y], marker='x', color='#00FFFF', s=120, linewidths=3, zorder=4)
            except Exception:
                ax1.scatter(y_p_region[:, 0], y_p_region[:, 1], c=highlight_color, s=15, alpha=0.7, edgecolors='none', zorder=2)

        elif prediction_plot_type.lower() == 'hexbin':
            ax1.hexbin(y_p_region[:, 0], y_p_region[:, 1], gridsize=40, cmap=white_inferno, mincnt=1, zorder=2)

        else:
            ax1.scatter(y_p_region[:, 0], y_p_region[:, 1], c=highlight_color, s=15, alpha=0.6, edgecolors='none', zorder=2)

        # Region circle outline. On torus the circle is drawn 9-tiled
        # so the wrapped half of a boundary-crossing circle appears on
        # the opposite edge of the unit cell, matching the wrap-aware
        # spatial-filter mask above.
        if self.manifold_metric == 'torus':
            period = float(self.manifold_period)
            tile_offsets = [(dx * period, dy * period)
                            for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
        else:
            tile_offsets = [(0.0, 0.0)]
        for ox, oy in tile_offsets:
            ax1.add_patch(plt.Circle(
                (centroid[0] + ox, centroid[1] + oy), radius,
                fill=False, edgecolor='#000000', lw=1.5,
                linestyle='--', zorder=5,
            ))

        # Axes extent: torus -> exactly the unit cell, no padding;
        # Euclidean -> the original padded view that leaves the
        # bottom-left corner free for the inset.
        if self.manifold_metric == 'torus':
            period = float(self.manifold_period)
            ax1.set_xlim(0, period)
            ax1.set_ylim(0, period)
            ax1.set_aspect('equal')
        else:
            # Dynamic Padding: Shift data up and right to give the inset an empty bottom-left corner
            x_min, x_max = Y_te[:, 0].min(), Y_te[:, 0].max()
            y_min, y_max = Y_te[:, 1].min(), Y_te[:, 1].max()
            ax1.set_xlim(x_min - (x_max - x_min) * 0.25, x_max + (x_max - x_min) * 0.05)
            ax1.set_ylim(y_min - (y_max - y_min) * 0.25, y_max + (y_max - y_min) * 0.05)

        # Axis-label prefix reflects the upstream latent space:
        # ``torus`` -> the QLVM latent (named QLVM Dimension N);
        # otherwise the VAE / UMAP-style continuous plane.
        dim_prefix = 'QLVM' if self.manifold_metric == 'torus' else 'UMAP'
        ax1.set_xlabel(f'{dim_prefix} Dimension 1', fontsize=12, color=text_color)
        ax1.set_ylabel(f'{dim_prefix} Dimension 2', fontsize=12, color=text_color)
        ax1.tick_params(axis='both', colors=text_color, labelsize=10,
                        bottom=True, left=True, labelbottom=True, labelleft=True)

        # Error Comparison Inset (Positioned explicitly in the new padded corner)
        ax_ins = ax1.inset_axes([0.07, 0.07, 0.22, 0.16])
        ax_ins.set_facecolor('none')
        ax_ins.grid(False)
        # Darken the highlight (male) colour 50% toward black for the
        # CNN-histogram edges; null histogram gets a plain black edge.
        _h = highlight_color.lstrip('#')
        _r, _g, _b = int(_h[0:2], 16), int(_h[2:4], 16), int(_h[4:6], 16)
        darker_highlight = (
            f"#{int(round(_r * 0.5)):02x}"
            f"{int(round(_g * 0.5)):02x}"
            f"{int(round(_b * 0.5)):02x}"
        )
        # ``histtype='stepfilled'`` draws a single filled outline
        # rather than one rectangle per bin, so the edgecolor is just
        # the outer step contour -- no edges visible between bins.
        ax_ins.hist(err_null, bins=15, density=True, color=null_color,
                    alpha=0.6, edgecolor='#000000', linewidth=0.7,
                    histtype='stepfilled', label='Null', zorder=1)
        ax_ins.hist(err_actual, bins=15, density=True, color=highlight_color,
                    alpha=0.6, edgecolor=darker_highlight, linewidth=0.7,
                    histtype='stepfilled', label='CNN', zorder=2)
        ax_ins.set_title("Prediction Error", fontsize=9, color=text_color)
        ax_ins.set_xlabel("Euclidean Distance", fontsize=8, color=text_color)
        ax_ins.set_ylabel("Density", fontsize=8, color=text_color)
        ax_ins.tick_params(axis='both', labelsize=7, colors=text_color, bottom=True, left=True, labelbottom=True, labelleft=True)
        leg = ax_ins.legend(fontsize=7, frameon=False, loc='best')
        for text in leg.get_texts():
            text.set_color(text_color)
        for spine in ax_ins.spines.values():
            spine.set_edgecolor(text_color)
        ax_ins.spines['top'].set_visible(False)
        ax_ins.spines['right'].set_visible(False)

        # PANEL 2: Contrastive Drivers Heatmap
        ax2.set_facecolor('#FFFFFF')

        # Gaussian temporal smoothing of the contrastive saliency map
        # along the time axis. Default sigma ~= 60 frames = 400 ms at
        # 150 fps -- matches the treatment applied in
        # ``plot_multinomial_multivariate_filters``. ``mode='reflect'``
        # keeps the boundaries well-behaved. Caller can override via
        # ``smoothing_sigma`` (set to 0 to disable smoothing entirely
        # and render the raw per-bin contrastive map).
        if smoothing_sigma > 0:
            smoothed_contrastive = gaussian_filter1d(
                contrastive_map, sigma=float(smoothing_sigma),
                axis=-1, mode='reflect',
            )
        else:
            smoothed_contrastive = contrastive_map

        v_lim = float(np.max(np.abs(smoothed_contrastive)))

        original_time = np.linspace(-4, 0, num_bins)
        smooth_time = np.linspace(-4, 0, 500)
        smooth_map = interp1d(
            original_time, smoothed_contrastive,
            kind='cubic', axis=1,
        )(smooth_time)

        im = ax2.imshow(smooth_map, aspect='auto', cmap=cmap,
                        extent=[-4, 0, num_features, 0], vmin=-v_lim, vmax=v_lim)

        ax2.set_xticks([-4, -3, -2, -1, 0])
        ax2.set_xlabel('Time before vocalization (s)', fontsize=12, color=text_color)
        ax2.set_yticks(np.arange(num_features) + 0.5)
        ax2.set_yticklabels(features_list, fontsize=8, color=text_color)

        # BRUTE-FORCE Axes labels to ensure they render over global styles
        ax2.tick_params(axis='x', colors=text_color, labelsize=10, bottom=True, labelbottom=True)
        ax2.tick_params(axis='y', colors=text_color, labelsize=8, left=True, labelleft=True)
        for label in ax2.get_xticklabels() + ax2.get_yticklabels():
            label.set_visible(True)

        ax2.set_title(f"Contrastive Drivers: {display_title}", fontsize=14, color=text_color, pad=15)

        # Colorbar
        cbar = fig.colorbar(im, ax=ax2, pad=0.02)
        cbar.set_label('Rel. Saliency (A.U.)', rotation=270, labelpad=15, color=text_color)
        cbar.ax.yaxis.set_tick_params(color=text_color, labelcolor=text_color, right=True, labelright=True)
        cbar.outline.set_edgecolor('#000000')
        for label in cbar.ax.get_yticklabels():
            label.set_visible(True)

        plt.tight_layout()

        # Filename Logic
        pkl_path = getattr(self, 'results_pkl_path', '')
        sex_mod = "male_mute_partner" if "male_mute_partner" in pkl_path else ("female" if "female" in pkl_path else "male")

        self._handle_save(fig, f"cnn_regional_saliency_{region_key}_{sex_mod}",
                          save_plot, output_dir, file_format)
        plt.show()


# Predictor diagnostics: plotting for the audits in
# `modeling.modeling_collinearity_audit`.

def _classify_predictor_feature(fname: str) -> int:
    """
    Classify a generic-keyed predictor feature into one of four
    presentation groups for the timescale-audit plots.

    Returns
    -------
    int
        0 → self.* (target-mouse egocentric features),
        1 → SEI features (`orofacial-sei`, `anogenital-sei`, including
            their `_1st_der` / `_2nd_der` derivatives),
        2 → other.* (predictor-mouse egocentric features),
        3 → social/dyadic kinematics (e.g. nose-nose, allo_yaw-nose).
    """

    if fname.startswith('self.'):
        return 0
    # SEI base columns end with `-sei`; derivative columns append
    # `_1st_der` or `_2nd_der`. Strip those tails before testing so
    # that e.g. `orofacial-sei_1st_der` lands in the SEI group rather
    # than the social/dyadic catch-all.
    base = fname
    for der_suffix in ('_1st_der', '_2nd_der'):
        if base.endswith(der_suffix):
            base = base[:-len(der_suffix)]
            break
    if base.endswith('-sei'):
        return 1
    if fname.startswith('other.'):
        return 2
    if '-' in fname:
        return 3
    # Default catch-all (e.g. an unprefixed pooled vocal signal):
    # treat as a social/dyadic feature for plot purposes.
    return 3


def _order_and_color_predictor_features(feature_names, source_pickle: str):
    """
    Stable-sort `feature_names` into the canonical timescale-audit
    group order (self → SEI → other → social/dyadic) and emit a
    parallel list of per-feature colours.

    Colour scheme
    -------------
    - self.*       → male/female colour of the focal/target mouse.
    - SEI          → same colour as `self.*` (full saturation). The
                     audit's column-selection rule
                     (`select_kinematic_columns` in
                     `modeling.modeling_utils`) keeps the
                     `{target}-{predictor}.{feature}` orientation of
                     each SEI column, so the kept SEI is the focal
                     mouse observing the partner — i.e. it *is* the
                     focal's engagement signal. Shared colour with the
                     self group; SEI block sits immediately after
                     self in the row order, and each row carries its
                     own y-axis label, so the groups are
                     distinguishable without a hue / saturation
                     difference.
    - other.*      → male/female colour of the partner mouse.
    - social/dyadic → `TIMESCALE_SOCIAL_COLOR` (a neutral slate that is
                      distinct from the axis lines, which stay black).

    The target/focal sex is inferred from the source-pickle filename
    (`_male_` / `_female_` token), mirroring the convention used
    elsewhere in this module.

    Parameters
    ----------
    feature_names : list of str
        Generic-keyed predictor feature names.
    source_pickle : str
        The artifact's `source_pickle` field (modeling pickle basename).

    Returns
    -------
    tuple
        `(order, ordered_colors)` where `order` is a list of indices
        into `feature_names` giving the new presentation order, and
        `ordered_colors` is a list of hex colour strings of the same
        length as `feature_names`, indexed in the new order.
    """

    # `_male_` / `_female_` token in the cohort filename identifies
    # the *target / focal / self* sex, matching the convention used
    # elsewhere in this module (see `plot_feature_ranking`). With the
    # `select_kinematic_columns` rule keeping the
    # `{target}-{predictor}.{feature}` SEI orientation, the focal is
    # the observer in the kept SEI columns, so SEI features are drawn
    # in the same target colour as `self.*` at full saturation. The
    # two groups are distinguished by their position in the row order
    # and by the per-row y-axis labels, not by hue or saturation.
    fname_low = (source_pickle or '').lower()
    if '_female_' in fname_low:
        # target / focal = female; SEI observer = female.
        self_col, other_col = female_color, male_color
    else:
        # `_male_` token (or no token, default): target / focal = male;
        # SEI observer = male.
        self_col, other_col = male_color, female_color
    sei_col = self_col
    color_by_group = {
        0: self_col,
        1: sei_col,
        2: other_col,
        3: TIMESCALE_SOCIAL_COLOR,
    }

    groups = [_classify_predictor_feature(f) for f in feature_names]
    order = sorted(range(len(feature_names)),
                   key=lambda i: (groups[i], feature_names[i]))
    ordered_colors = [color_by_group[groups[i]] for i in order]
    return order, ordered_colors


def _rolling_mean_1d(arr, window: int):
    """
    Centred rolling-average smoothing of a 1-D array.

    Uses reflection at the boundaries (no zero-padding bias) so the
    edge bins remain comparable to interior bins. Returns the input
    unchanged when `window <= 1`.

    Parameters
    ----------
    arr : array-like
        1-D array to smooth. NaN values propagate within the window.
    window : int
        Window size in bins. Even windows are bumped up to the next
        odd integer so the smoothing remains centred.

    Returns
    -------
    np.ndarray
        Smoothed array, same length as input.
    """

    arr = np.asarray(arr, dtype=np.float64)
    if window is None or window <= 1 or arr.size == 0:
        return arr.astype(np.float32, copy=False)
    # Round odd-up so the window is symmetric around each output bin.
    w = int(window) | 1
    if w >= arr.size:
        return np.full(arr.shape, np.nanmean(arr), dtype=np.float32)
    half = w // 2
    # Reflect-pad to avoid zero-bias at the boundaries.
    pad = np.concatenate([arr[half:0:-1], arr, arr[-2:-half - 2:-1]])
    kernel = np.ones(w, dtype=np.float64) / w
    smoothed = np.convolve(pad, kernel, mode='valid')
    return smoothed[:arr.size].astype(np.float32, copy=False)


def _require_timescale_payload(payload, timescale_pkl_path: str) -> None:
    """
    Validate that the pickle at `timescale_pkl_path` is a timescale-
    audit artifact (as produced by
    `audit_predictor_timescales`). Raises a clear `ValueError` if it
    is missing the canonical fields, hinting that the user might have
    pointed at the modeling input pickle instead of the
    `_timescales.pkl` companion artifact.
    """

    required = ('features', 'acf_lags_seconds', 'acf_median', 'rho_signal',
                'signal_lags_seconds', 'acf_null_mean', 'acf_null_p99_5',
                'rho_signal_null_mean', 'rho_signal_null_p99_5')
    missing = [k for k in required if k not in payload]
    if missing:
        raise ValueError(
            f"`{timescale_pkl_path}` does not look like a timescale-audit "
            f"pickle (missing keys: {missing}). The timescale audit writes "
            f"its companion artifact alongside the modeling input pickle "
            f"with the suffix `_timescales.pkl` — make sure the path you "
            f"passed ends with that suffix and not, for example, the "
            f"modeling input pickle itself."
        )


def _last_bin_of_consecutive_run(above_mask, run_length: int):
    """
    Locate the *latest* index `k` such that all of
    `above_mask[k - run_length + 1 : k + 1]` are True.

    Used by the timescale-audit plot to mark the lag at which the
    feature's ACF leaves a sustained run above the shuffled-feature
    null band — i.e. the last lag of the longest-still-present block of
    `run_length` consecutive bins above the band.

    Parameters
    ----------
    above_mask : array-like of bool
        Boolean mask, e.g. `acf_median[f] > acf_null_p99_5[f]`.
    run_length : int
        Required run length in bins (e.g. 15 bins ≈ 100 ms at 150 fps).

    Returns
    -------
    int or None
        The latest valid run-end index, or `None` when no qualifying run
        exists.
    """

    arr = np.asarray(above_mask, dtype=bool)
    n = arr.size
    if run_length <= 0 or n < run_length:
        return None
    # Scan from the back so the first hit is the latest run-end.
    for k in range(n - 1, run_length - 2, -1):
        if arr[k - run_length + 1:k + 1].all():
            return int(k)
    return None


def _signal_outer_run_marker(rho,
                             null_lo,
                             null_hi,
                             min_run_bins: int,
                             idx_floor: int,
                             idx_max: int):
    """
    Locate the cross-correlation right-side significance marker.

    Algorithm: **first** (earliest-starting) sign-consistent
    outside-null run on the symmetric lag axis whose end falls
    within the positive-lag search range, anchored at the run's
    largest-lag end.

      1. Identify every contiguous run of bins on the lag axis where
         `rho` is sign-consistently outside the null band — either
         all entries are strictly above `null_hi` (positive run) or
         all are strictly below `null_lo` (negative run). Mixed
         (above/below alternating) regions do not form a single
         run; they break into multiple separate same-sign runs.
      2. Discard runs whose last bin is below `idx_floor` (so
         sub-floor-only runs cannot anchor a marker — the floor
         exists precisely to suppress very-near-zero noise on the
         right side).
      3. Discard runs shorter than `min_run_bins` (the noise-
         fragment threshold; e.g. ~30 bins ≈ 200 ms at 150 fps).
      4. Among the surviving runs, pick the run with the **smallest
         start index** (earliest lag at which the run begins). Runs
         may start at negative lags; what counts is the order on the
         lag axis, not whether the run sits in positive territory.
         Tie-break: smaller end index.
      5. The marker sits at the **end** (largest lag) of that run.
         The marker's sign is the run's direction (+1 above-null,
         −1 below-null).
      6. If no surviving run exists, return `None` (no marker drawn
         on the panel).
      7. If the picked run extends to the right edge of the search
         window (`end == idx_max`), `exceeds_window` is set True so
         the plot can annotate the lag with a `+` to indicate the
         horizon was not yet reached at `+max_lag`.

    Why "first run" rather than "longest run":
    on cohort-mean ρ curves with a strong central peak the symmetric
    lag axis often contains two qualifying runs — an early "main"
    run that begins on the negative-lag side and ends just past the
    central peak (the visible cross-correlation feature), and a
    secondary late re-emergence on the right edge of the window.
    Length-based tiebreaks flip between these two whenever the late
    re-emergence is even marginally longer than the main run,
    producing markers that pin to `+max_lag` for one feature and to
    the visible crossing for another. Anchoring on the earliest-
    starting run pins the marker to the main run consistently:
    the sustained departure that begins first on the lag axis is
    what a reader's eye reads as "the" cross-correlation event,
    regardless of whether a later, only slightly longer fragment
    happens to extend to the window edge.

    Parameters
    ----------
    rho : array-like
        Per-lag actual correlation curve (1-D).
    null_lo, null_hi : array-like
        Per-lag lower / upper null envelope, same length as `rho`.
    min_run_bins : int
        Minimum sustained-run length in bins. Runs shorter than
        this are dropped before the first-run pick. Set above
        the typical noise-fragment scale; converted from
        `signal_min_run_seconds` × fps at the call site.
    idx_floor : int
        Smallest lag index that may host the run's end. Runs whose
        last bin is below `idx_floor` are dropped (so very-near-zero
        excursions cannot anchor a marker). Runs may **start**
        below `idx_floor` provided they extend at or above it —
        the floor only constrains the right edge.
    idx_max : int
        Largest lag index considered (typically `n_lags − 1`,
        corresponding to `+max_lag_seconds`).

    Returns
    -------
    tuple or None
        `(marker_idx, sign, exceeds_window)` where
        `sign ∈ {+1, −1}` and `exceeds_window` is True when the
        run extends to `idx_max`. `None` when no qualifying run
        exists in the search range.
    """

    rho = np.asarray(rho)
    null_lo = np.asarray(null_lo)
    null_hi = np.asarray(null_hi)

    if min_run_bins <= 0:
        return None
    if idx_floor < 0 or idx_max >= rho.size or idx_max < idx_floor:
        return None

    above = rho > null_hi
    below = rho < null_lo

    # Find all contiguous True-runs in `mask` via padded-diff trick.
    # Returns list of `(start_idx, end_idx_inclusive, length)`.
    def _runs(mask):
        m = np.asarray(mask, dtype=bool)
        if not m.any():
            return []
        padded = np.concatenate([[False], m, [False]])
        d = np.diff(padded.astype(np.int8))
        starts = np.where(d == 1)[0]
        ends = np.where(d == -1)[0] - 1  # inclusive
        return list(zip(starts.tolist(), ends.tolist(),
                        (ends - starts + 1).tolist()))

    candidates = []
    for sign_val, mask in ((+1, above), (-1, below)):
        for s, e, L in _runs(mask):
            if e < idx_floor:
                continue  # entire run sub-floor
            if e > idx_max:
                continue  # past search window (defensive)
            if L < min_run_bins:
                continue
            candidates.append((s, e, sign_val))

    if not candidates:
        return None

    # Earliest-starting run wins. Tie-break: smaller end index.
    candidates.sort(key=lambda c: (c[0], c[1]))
    _, end_idx, sign_val = candidates[0]
    return (int(end_idx), int(sign_val), bool(end_idx == idx_max))


def _save_audit_figure(fig, out_dir: str, basename: str, file_format: str = 'svg') -> str:
    """
    Writes an audit figure to disk in the requested format and returns the
    absolute path. Defaults to SVG (vector, lossless) so the diagnostic
    plots remain publication-ready by default; pass `file_format='png'` /
    `'pdf'` to override.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save.
    out_dir : str
        Output directory; created if missing.
    basename : str
        Filename stem (no extension).
    file_format : str, default 'svg'
        Matplotlib `savefig` format.

    Returns
    -------
    str
        Absolute path of the written file.
    """

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{basename}.{file_format}")
    fig.savefig(out_path, format=file_format, bbox_inches='tight')
    return out_path


def plot_collinearity_audit(audit_pkl_path: str,
                            save_dir: str = None,
                            save_plot_bool: bool = True,
                            plot_format: str = 'svg',
                            cmap: str = 'RdBu_r',
                            outline_threshold: float = 0.7,
                            outline_color: str = '#000000') -> dict:
    """
    Renders the collinearity audit produced by
    `modeling.modeling_collinearity_audit.audit_predictor_collinearity`
    as a two-panel diagnostic figure that mirrors the timescale-audit
    plots in feature ordering and colour coding.

    The figure tells the reader, at a glance, whether the kept
    predictor set is sufficiently decorrelated for stable forward
    stepwise selection. It is meant to be inspected side-by-side with
    `plot_timescale_audit_per_feature` and `plot_timescale_audit`:
    the row order, the per-feature group colour, and the
    single-legend-bottom-right convention are deliberately aligned
    across all three so that a feature can be cross-referenced by
    position and hue without re-reading axis labels.

    Layout
    ------
    Left panel — Spearman ρ heatmap (group-ordered)
        Rows / columns ordered by feature group (self → SEI →
        other → social/dyadic) using
        `_order_and_color_predictor_features`, the same routine
        the timescale plots use; within each group, features are
        alphabetised. Hairline black separator lines on both axes
        delimit group boundaries so the within-group vs. cross-
        group block structure is explicit. Tick labels are bold
        and coloured per feature group (matching the per-feature
        timescale panels); the axis tick *marks* are suppressed
        because the colour-coded labels are sufficient. Diverging
        symmetric colormap (default `'RdBu_r'`, exposed via the
        `cmap` parameter so the caller can plug in any other
        diverging colormap — sequential maps would misrepresent
        the sign of ρ and should not be used). Off-diagonal cells
        whose `|ρ|` exceeds `outline_threshold` receive a thick
        outline in `outline_color` and are annotated with their
        signed ρ value. The diagonal is unannotated (ρ_ii = 1 by
        construction).

    Right panel — VIF horizontal bars (group-coloured)
        Per-feature variance inflation factors, sorted descending,
        with bar fill and y-tick label colours matching the
        feature's group colour from the heatmap. Mean (solid) and
        median (dashed) reference lines, with a ▲ pointer + bold
        numeric annotation at the cohort mean — the same
        convention used by `plot_timescale_audit` so the two
        cohort summaries read identically. The x-axis upper edge
        is the next decade (multiple of 10) above the largest
        finite VIF, so the bars use the available width without
        a fixed cap. Only the first (`0`) and last
        (`{cap_x:g}`) x-tick labels are drawn — the intermediate
        decade tick marks remain visible but unlabelled. Infinite
        VIFs (perfect linear dependence) are drawn at the right
        edge and annotated as ``inf``.

    Parameters
    ----------
    audit_pkl_path : str
        Path to the `_collinearity.pkl` artifact produced at
        extraction time (see `audit_predictor_collinearity`).
    save_dir : str, optional
        Output directory for the figure. Defaults to the directory
        containing the audit pickle. Only consulted when
        `save_plot_bool` is True.
    save_plot_bool : bool, default True
        When True, the figure is written to disk via
        `_save_audit_figure` and closed. When False, the figure is
        neither saved nor closed (suitable for inline display in
        a notebook), matching the timescale-plot convention.
    plot_format : str, default 'svg'
        Matplotlib `savefig` format. SVG is the project default
        for publication-quality output. Only consulted when
        `save_plot_bool` is True.
    cmap : str, default 'RdBu_r'
        Matplotlib colormap name for the Spearman ρ heatmap. Must
        be a *diverging* colormap because the colour scale is
        symmetric on `[-1, 1]` — sequential maps would conflate
        sign information.
    outline_threshold : float, default 0.7
        Minimum `|ρ|` for an off-diagonal cell to receive a thick
        outline + numeric annotation. The artifact's own
        `concern_threshold` / `exclude_threshold` are not
        consulted here; this single, plot-level threshold keeps
        the heatmap readable when it is the diagnostic the
        figure is meant to flag.
    outline_color : str, default '#000000'
        Hex colour used both for the cell outline and for the
        in-cell numeric ρ annotation. Pick a colour that contrasts
        with the chosen `cmap` so the outline reads against the
        cell fill.

    Returns
    -------
    dict
        `{'figure_path': str, 'n_features': int, 'n_flagged': int,
        'condition_number': float, 'mean_vif': float,
        'median_vif': float}`. `figure_path` is `''` when
        `save_plot_bool` is False; `mean_vif` / `median_vif` are
        computed over the finite VIF values only and are NaN if
        no feature has a finite VIF.
    """

    audit_pkl_path = configure_path(str(audit_pkl_path))
    if save_dir is not None:
        save_dir = configure_path(str(save_dir))

    with open(audit_pkl_path, 'rb') as fh:
        payload = pickle.load(fh)

    feature_names = list(payload['features'])
    rho = np.asarray(payload['spearman_rho'])
    vif = np.asarray(payload['vif'])
    flagged = payload['flagged_pairs']
    n_events = int(payload['n_events'])
    cond_num = float(payload['condition_number'])
    source = payload['source_pickle']

    n_features = len(feature_names)
    if n_features == 0:
        print(f"[plot] collinearity audit at {audit_pkl_path} has no features — skipping.")
        return {
            'figure_path': '',
            'n_features': 0,
            'n_flagged': 0,
            'condition_number': cond_num,
            'mean_vif': float('nan'),
            'median_vif': float('nan'),
        }

    # Group-based ordering: self → SEI → other → social-dyadic. Same
    # routine and palette as the per-feature timescale plot, so the
    # heatmap row order and the timescale panel row order match
    # one-for-one.
    order, ordered_colors = _order_and_color_predictor_features(
        feature_names, source
    )
    names_ord = [feature_names[i] for i in order]
    colors_ord = list(ordered_colors)
    rho_ord = rho[np.ix_(order, order)]

    # Group-boundary indices: where the classifier flips from one
    # group to the next in `names_ord`. Used to draw hairline
    # separator lines between groups on both heatmap axes.
    groups_ord = [_classify_predictor_feature(n) for n in names_ord]
    group_breaks = [k for k in range(1, n_features)
                    if groups_ord[k] != groups_ord[k - 1]]

    fig_height = max(0.32 * n_features + 2.4, 4.5)
    fig = plt.figure(figsize=(13.0, fig_height))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.4, 1.0], wspace=0.45,
                           top=0.94, bottom=0.18, left=0.18, right=0.97)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    # Left panel: ρ heatmap
    im = ax1.imshow(rho_ord, cmap=cmap, vmin=-1.0, vmax=1.0, aspect='equal')
    ax1.set_xticks(np.arange(n_features))
    ax1.set_yticks(np.arange(n_features))
    # Bold tick labels — same fontsize as before, but bold weight so
    # the per-group colour reads cleanly even on the lighter SEI /
    # social hues.
    ax1.set_xticklabels(names_ord, rotation=90, fontsize=7,
                        fontweight='bold')
    ax1.set_yticklabels(names_ord, fontsize=7, fontweight='bold')
    for tick_label, c in zip(ax1.get_xticklabels(), colors_ord):
        tick_label.set_color(c)
    for tick_label, c in zip(ax1.get_yticklabels(), colors_ord):
        tick_label.set_color(c)
    ax1.tick_params(axis='both', which='both', length=0)

    # Group separator lines: a hairline black line between each pair
    # of groups on both axes. Positioned at `b - 0.5` so the line
    # falls exactly between the last cell of the previous group and
    # the first cell of the next group.
    for b in group_breaks:
        ax1.axvline(b - 0.5, color=TIMESCALE_AXIS_COLOR,
                    linewidth=0.3, zorder=4)
        ax1.axhline(b - 0.5, color=TIMESCALE_AXIS_COLOR,
                    linewidth=0.3, zorder=4)

    # Single-tier outline + numeric annotation on cells with
    # |ρ| > outline_threshold. The diagonal is skipped (ρ_ii = 1 by
    # construction).
    for i in range(n_features):
        for j in range(n_features):
            if i == j:
                continue
            r = rho_ord[i, j]
            if not np.isfinite(r):
                continue
            absr = abs(r)
            if absr >= outline_threshold:
                ax1.add_patch(Rectangle(
                    (j - 0.5, i - 0.5), 1.0, 1.0,
                    fill=False, edgecolor=outline_color,
                    linewidth=1.5, zorder=5,
                ))
                ax1.text(j, i, f"{r:+.2f}", ha='center', va='center',
                         fontsize=5, color=outline_color, zorder=6)

    cb = fig.colorbar(im, ax=ax1, fraction=0.04, pad=0.02)
    cb.set_label('Spearman ρ', fontsize=9)
    # Major ticks at canonical [-1, -0.5, 0, 0.5, 1]; minor ticks
    # disabled — the colorbar is purely a legend, not a precise
    # scale, so intermediate gradations only add visual noise.
    cb.set_ticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    cb.minorticks_off()
    cb.ax.tick_params(labelsize=7)
    ax1.set_title(
        f"Pairwise correlations  ({n_events} events × {n_features} features)",
        fontsize=10,
    )

    # Right panel: VIF horizontal bars
    color_by_name = {name: c for name, c in zip(names_ord, colors_ord)}
    finite_mask = np.isfinite(vif)
    finite_vif = np.where(finite_mask, vif, np.nan)
    # Sort descending by VIF; +inf gets pushed to the top via the
    # masking trick (`-inf` for the negative-sort, sorted ascending).
    sort_idx = np.argsort(-np.where(finite_mask, vif, -np.inf))
    vif_sorted = vif[sort_idx]
    names_sorted = [feature_names[i] for i in sort_idx]
    colors_sorted = [color_by_name[n] for n in names_sorted]

    finite_max = float(np.nanmax(finite_vif)) if finite_mask.any() else 5.0
    # X-axis upper edge: the next decade (multiple of 10) above the
    # largest finite VIF, with a floor of 10 so even very-low-VIF
    # cohorts get a visible scale. No artificial cap — if every
    # feature is finite and small, the chart is tight; if a
    # feature is enormous, the decade-rounding grows to fit.
    cap_x = max(10.0, float(np.ceil(finite_max / 10.0) * 10.0))
    vif_display = np.where(finite_mask[sort_idx],
                           np.minimum(vif_sorted, cap_x),
                           cap_x)

    y_pos = np.arange(n_features)[::-1]
    ax2.barh(y_pos, vif_display, color=colors_sorted,
             edgecolor=TIMESCALE_AXIS_COLOR, linewidth=0.5,
             height=1.0, zorder=2)
    # Annotate `inf` bars at their (capped) right edge so the
    # singular-design feature stays visually distinct from a
    # merely-large-but-finite VIF at the same display length.
    for k, v in enumerate(vif_sorted):
        if not np.isfinite(v):
            ax2.text(cap_x * 0.99, y_pos[k], 'inf',
                     va='center', ha='right', fontsize=7,
                     color=TIMESCALE_AXIS_COLOR, fontweight='bold',
                     zorder=4)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(names_sorted, fontsize=8, fontweight='bold')
    for tick_label, c in zip(ax2.get_yticklabels(), colors_sorted):
        tick_label.set_color(c)
    ax2.tick_params(axis='y', length=0)

    if finite_mask.any():
        mean_vif = float(np.nanmean(finite_vif))
        median_vif = float(np.nanmedian(finite_vif))
    else:
        mean_vif = float('nan')
        median_vif = float('nan')

    # Cohort mean / median lines (matching the timescale-cohort plot
    # style: solid mean, dashed median, both in axis colour). No
    # canonical-threshold reference lines — the cohort references
    # are the only ones drawn on this panel.
    if np.isfinite(mean_vif):
        ax2.axvline(mean_vif, color=TIMESCALE_AXIS_COLOR,
                    linestyle='-', linewidth=1.0, zorder=4)
    if np.isfinite(median_vif):
        ax2.axvline(median_vif, color=TIMESCALE_AXIS_COLOR,
                    linestyle='--', linewidth=1.0, zorder=4)

    # ▲ pointer + bold mean numeric annotation just below the x-axis
    # spine, exactly the convention `plot_timescale_audit` uses for
    # its cohort mean.
    x_data_y_axes = mtransforms.blended_transform_factory(
        ax2.transData, ax2.transAxes
    )
    if np.isfinite(mean_vif):
        ax2.plot(mean_vif, 0, marker='^', color=TIMESCALE_AXIS_COLOR,
                 markersize=7, linestyle='None', zorder=6,
                 clip_on=False, transform=x_data_y_axes)
        ax2.text(mean_vif, -0.015, f'{mean_vif:.2f}',
                 ha='center', va='top', fontweight='bold',
                 fontsize=8, color=TIMESCALE_AXIS_COLOR,
                 transform=x_data_y_axes, clip_on=False)

    ax2.set_xlim(0.0, cap_x)
    ax2.set_ylim(-0.5, n_features - 0.5)
    # Tick marks at every decade from 0 to `cap_x`; only the first
    # (`0`) and last (`cap_x`) tick labels are drawn — the
    # intermediate ticks remain as visual reference but uncluttered
    # by labels (mirroring the timescale-cohort plot's tick style).
    xticks = list(range(0, int(cap_x) + 1, 10))
    xticklabels = ['' for _ in xticks]
    xticklabels[0] = '0'
    xticklabels[-1] = str(int(cap_x))
    ax2.set_xticks(xticks)
    ax2.set_xticklabels(xticklabels)
    # With most x-tick labels blank, the x-axis label can sit close
    # to the spine (matplotlib default labelpad). The bold mean
    # numeric annotation already lives just below the spine; the
    # default ~4-pt labelpad places the axis label below the
    # annotation without colliding with it.
    ax2.set_xlabel('Variance Inflation Factor (VIF)')
    ax2.set_title(
        f'Variance Inflation\n'
        f'cond(X̃) = {cond_num:.1f}',
        fontsize=10,
    )
    ax2.tick_params(axis='x', labelsize=8)

    # Single mean / median legend in the lower-right of the right
    # panel — same placement and style as the cohort timescale plot
    # so the two figures read consistently when stacked.
    legend_handles = [
        Line2D([0], [0], color=TIMESCALE_AXIS_COLOR, linestyle='-',
               linewidth=1.0, label='mean'),
        Line2D([0], [0], color=TIMESCALE_AXIS_COLOR, linestyle='--',
               linewidth=1.0, label='median'),
    ]
    ax2.legend(handles=legend_handles, loc='lower right',
               ncol=1, fontsize=9, frameon=False)

    # No figure-level suptitle: the `source_pickle` filename — which
    # already carries the cohort + timestamp — is the figure's
    # provenance, and the saved figure inherits that filename via
    # `_save_audit_figure`.

    if save_plot_bool:
        if save_dir is None:
            save_dir = os.path.dirname(audit_pkl_path)
        base = os.path.splitext(os.path.basename(audit_pkl_path))[0]
        out_path = _save_audit_figure(fig, save_dir, base,
                                      file_format=plot_format)
        plt.close(fig)
        print(f"[plot] collinearity figure written: {out_path}")
    else:
        out_path = ''

    return {
        'figure_path': out_path,
        'n_features': n_features,
        'n_flagged': len(flagged),
        'condition_number': cond_num,
        'mean_vif': mean_vif,
        'median_vif': median_vif,
    }


def _compute_timescale_horizons(payload: dict,
                                signal_smooth_window: int = 0,
                                acf_run_length: int = 15) -> tuple:
    """
    Per-feature ACF and cross-correlation horizons (single-number
    summaries derived from the per-feature triangle markers).

    For each feature, replicates the marker-finding logic used by
    `plot_timescale_audit_per_feature`:

    - **ACF horizon** = the latest lag at which `acf_run_length`
      consecutive bins are above the upper ACF null envelope
      (`acf_null_p99_5`). Computed via
      `_last_bin_of_consecutive_run`.
    - **XC horizon** = the lag at the largest-lag end of the
      earliest-starting sustained outside-null run on the symmetric
      lag axis, after applying the lag floor `signal_floor_seconds`.
      Computed via `_signal_outer_run_marker`. The exceeds-window
      flag is preserved so the caller can render `+max_lag` markers
      distinctively if desired.

    Features without a qualifying marker are excluded from the
    returned mapping (the caller decides whether to skip them or
    represent them otherwise).

    Returns
    -------
    tuple
        `(acf_horizons, xc_horizons, xc_exceeds)` where each is a
        `{feature_name: ...}` dict: `acf_horizons` and `xc_horizons`
        map to the marker lag in seconds, while `xc_exceeds` maps to
        the boolean exceeds-window flag (True when the XC run reaches
        `+max_lag`).
    """

    feature_names = list(payload['features'])
    acf_lags_seconds = np.asarray(payload['acf_lags_seconds'])
    acf_med = np.asarray(payload['acf_median'])
    acf_null_hi = np.asarray(payload['acf_null_p99_5'])
    signal_lags_seconds = np.asarray(payload['signal_lags_seconds'])
    rho_signal = np.asarray(payload['rho_signal'])
    rho_signal_null_lo = np.asarray(payload['rho_signal_null_p0_5'])
    rho_signal_null_hi = np.asarray(payload['rho_signal_null_p99_5'])
    signal_floor_seconds = (
        float(payload['signal_floor_seconds'])
        if 'signal_floor_seconds' in payload else 0.5
    )
    # Minimum sustained-run length (seconds) for the cross-correlation
    # marker. Falls back to 0.2 s for older artifacts that pre-date
    # the field (matches the ~30-bin default at 150 fps that
    # suppresses noise-fragment cohorts while preserving real
    # multi-hundred-bin runs).
    signal_min_run_seconds = (
        float(payload['signal_min_run_seconds'])
        if 'signal_min_run_seconds' in payload else 0.2
    )
    # Convert seconds → bins via the lag-axis spacing. `np.ceil` so a
    # threshold of e.g. 0.2 s on a 150-fps grid gives 30 bins
    # (slightly above 0.2 s = 30/150 = 0.2 s exactly), never less.
    if signal_lags_seconds.size > 1:
        delta_t = float(signal_lags_seconds[1] - signal_lags_seconds[0])
        sig_min_run_bins = int(np.ceil(signal_min_run_seconds / delta_t))
    else:
        sig_min_run_bins = 1

    acf_horizons = {}
    xc_horizons = {}
    xc_exceeds = {}

    n_features = len(feature_names)
    for i in range(n_features):
        fname = feature_names[i]

        # ACF marker (still uses the immediately-preceding-run rule
        # — the ACF panel's marker semantics are unchanged).
        if acf_med.size and i < acf_med.shape[0]:
            above_mask = acf_med[i] > acf_null_hi[i]
            mark_idx = _last_bin_of_consecutive_run(above_mask, acf_run_length)
            if mark_idx is not None:
                acf_horizons[fname] = float(acf_lags_seconds[mark_idx])

        # XC marker (earliest sign-consistent run ≥ `sig_min_run_bins`).
        if signal_lags_seconds.size and i < rho_signal.shape[0]:
            mean_curve = _rolling_mean_1d(rho_signal[i], signal_smooth_window)
            sig_idx_floor = int(np.searchsorted(
                signal_lags_seconds, signal_floor_seconds, side='left'
            ))
            sig_idx_max = signal_lags_seconds.size - 1
            sig_hit = _signal_outer_run_marker(
                mean_curve,
                rho_signal_null_lo[i],
                rho_signal_null_hi[i],
                sig_min_run_bins,
                sig_idx_floor,
                sig_idx_max,
            )
            if sig_hit is not None:
                sig_idx, _sig_sign, sig_exceeds_flag = sig_hit
                xc_horizons[fname] = float(signal_lags_seconds[sig_idx])
                xc_exceeds[fname] = bool(sig_exceeds_flag)

    return acf_horizons, xc_horizons, xc_exceeds


def plot_timescale_audit(timescale_pkl_path: str,
                         save_dir: str = None,
                         save_plot_bool: bool = True,
                         plot_format: str = 'svg',
                         signal_smooth_window: int = 0) -> dict:
    """
    Cohort-level summary companion to `plot_timescale_audit_per_feature`.

    Two horizontal-bar panels, one per measure, summarising the
    triangle-marker positions from the per-feature plot as a single
    number per feature:

    - **Left panel ("Auto-correlation")**: bar length = the lag at
      which the ACF leaves a sustained run above its upper null.
      One bar per feature with an ACF marker.
    - **Right panel ("Cross-correlation")**: bar length = the
      positive-side significance horizon (largest-lag end of the
      pre-cross outside-null run, with the lag-floor
      `signal_floor_seconds` applied). One bar per feature with a
      qualifying XC marker.

    Each panel is sorted independently descending by value (longest
    horizon at top) so the cohort distribution is legible at a
    glance. Bars touch (no inter-bar gap), each with a thin black
    outline. Bar colour is the per-feature group colour
    (`_order_and_color_predictor_features`), matching the per-feature
    panel; rows can be cross-referenced by colour. The y-tick
    labels are coloured to match each row's bar, and the y-tick
    *marks* are suppressed (the colour-coded labels are sufficient).

    Both panels share an x-axis range `[0, ceil(max_horizon)]`
    so they're directly comparable; tick marks are placed every 1 s,
    but only the first and last ticks receive labels — leaving room
    below the axis for the bold mean annotation and the x-axis
    label.

    Cohort references on each panel:

    - Solid black vertical line at the **mean** horizon.
    - Dashed black vertical line at the **median** horizon.
    - Just below the x-axis spine at the mean's x-position: a small
      `▲` pointer plus the bold mean value (`{value} s`) so the
      cohort mean is explicitly numeric on the axis.
    - The x-axis label `Lag (s)` sits further below, beneath the
      mean annotation (via `labelpad`).

    A single legend in the lower-right corner of the right panel
    identifies the mean (solid) and median (dashed) line styles for
    both panels — the styles are shared, so one legend suffices.

    Features without a qualifying marker for a panel are **skipped**
    (no row drawn, not counted toward mean / median). This is
    deliberate: a "no horizon" entry doesn't belong in a horizon
    distribution. Features in the cross-correlation panel whose
    significance horizon exceeds the lag window are clipped at
    `+max_lag`; the cohort under the configurations we ship doesn't
    hit this case, but it is handled correctly if it does.

    The sign of the cross-correlation marker (▽ vs. △) is
    intentionally not encoded here — this figure is a *time*
    summary, not a strength-or-direction summary.

    Parameters
    ----------
    timescale_pkl_path : str
        Path to the `_timescales.pkl` artifact.
    save_dir : str, optional
        Output directory. Defaults to the directory containing the
        timescale pickle. Only consulted when `save_plot_bool` is
        True.
    save_plot_bool : bool, default True
        When True (default), the figure is written to disk via
        `_save_audit_figure` and closed. When False, the figure is
        neither saved nor closed.
    plot_format : str, default 'svg'
        Matplotlib `savefig` format. Only consulted when
        `save_plot_bool` is True.
    signal_smooth_window : int, default 0
        Rolling-average window size (in lag bins) applied to the
        cross-correlation mean curve before computing the XC marker.
        Forwarded to `_compute_timescale_horizons`. Same default as
        the per-feature plot.

    Returns
    -------
    dict
        `{'figure_path': str, 'n_features_acf': int, 'n_features_xc': int,
        'mean_acf_horizon_s': float, 'median_acf_horizon_s': float,
        'mean_xc_horizon_s': float, 'median_xc_horizon_s': float,
        'configured_filter_history': float}`.
        Mean / median entries are NaN when the corresponding panel is
        empty. `figure_path` is `''` when `save_plot_bool` is False.
    """

    with open(timescale_pkl_path, 'rb') as fh:
        payload = pickle.load(fh)
    _require_timescale_payload(payload, timescale_pkl_path)

    feature_names = list(payload['features'])
    cfg_hist = float(payload['configured_filter_history'])
    source = payload['source_pickle']
    n_features_total = len(feature_names)

    acf_horizons, xc_horizons, xc_exceeds = _compute_timescale_horizons(
        payload, signal_smooth_window=signal_smooth_window
    )

    # Per-feature group colour, matching the per-feature panel exactly.
    order, ordered_colors = _order_and_color_predictor_features(
        feature_names, source
    )
    color_by_name = {}
    for new_i, orig_i in enumerate(order):
        color_by_name[feature_names[orig_i]] = ordered_colors[new_i]

    # Sort each panel's features descending by horizon (longest at the
    # top of the bar plot — `barh` with descending y-positions).
    acf_items = sorted(acf_horizons.items(), key=lambda kv: -kv[1])
    xc_items = sorted(xc_horizons.items(), key=lambda kv: -kv[1])

    n_acf = len(acf_items)
    n_xc = len(xc_items)

    if n_acf == 0 and n_xc == 0:
        print(f"[plot] timescale audit at {timescale_pkl_path}: no features "
              f"with markers in either panel — skipping.")
        return {
            'figure_path': '',
            'n_features_acf': 0,
            'n_features_xc': 0,
            'mean_acf_horizon_s': float('nan'),
            'median_acf_horizon_s': float('nan'),
            'mean_xc_horizon_s': float('nan'),
            'median_xc_horizon_s': float('nan'),
            'configured_filter_history': cfg_hist,
        }

    # Shared x-axis range across both panels so the two are directly
    # comparable. The right edge is `int(np.ceil(joint_max))` — no
    # additional rounding. Tick marks are placed at every whole
    # number from 0 to `x_max_int` inclusive; only the first ("0")
    # and last (right edge) tick labels are drawn, the others are
    # tick marks without labels — leaving room below the axis for
    # the bold mean annotation and for the x-axis label further
    # down.
    all_horizon_vals = (
        [v for _, v in acf_items] + [v for _, v in xc_items]
    )
    if all_horizon_vals:
        x_max_data = float(max(all_horizon_vals))
    else:
        x_max_data = 1.0
    x_max_int = max(int(np.ceil(x_max_data)), 1)
    shared_xticks = list(range(0, x_max_int + 1))
    shared_xticklabels = ['' for _ in shared_xticks]
    shared_xticklabels[0] = '0'
    shared_xticklabels[-1] = str(x_max_int)

    # Figure size: vertical scaling proportional to the larger of the
    # two row counts, with a minimum so very small cohorts still look
    # reasonable. Width split evenly between the two panels.
    n_rows_max = max(n_acf, n_xc, 1)
    fig_height = max(0.32 * n_rows_max + 1.8, 3.5)
    fig = plt.figure(figsize=(11.0, fig_height))
    gs = gridspec.GridSpec(1, 2, wspace=0.55,
                           top=0.93, bottom=0.18, left=0.18, right=0.97)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])

    def _render_panel(ax, items, panel_title, x_axis_label):
        if not items:
            ax.set_title(panel_title, fontsize=11)
            ax.set_xlabel(x_axis_label, labelpad=18)
            ax.text(0.5, 0.5, 'no features with marker',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=9, color=TIMESCALE_NULL_COLOR)
            ax.set_yticks([])
            ax.set_xlim(0, x_max_int)
            ax.set_xticks(shared_xticks)
            ax.set_xticklabels(shared_xticklabels)
            return float('nan'), float('nan')

        names = [n for n, _ in items]
        vals = np.asarray([v for _, v in items], dtype=np.float64)
        cols = [color_by_name[n] for n in names]
        # Top-of-axis = largest. With `barh(y, width)` and y_pos
        # descending from len-1 at top to 0 at bottom, names[0]
        # (largest) sits at top.
        y_pos = np.arange(len(items))[::-1]

        # `height=1.0` so adjacent bars touch with no inter-bar gap.
        ax.barh(y_pos, vals, color=cols,
                edgecolor=TIMESCALE_AXIS_COLOR, linewidth=0.5,
                height=1.0, zorder=2)
        # Y-axis: feature names coloured to match each row's bar
        # colour (so the row → feature mapping is doubly encoded by
        # the bar fill and the label hue), and tick marks suppressed
        # because the colored labels alone are sufficient.
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=9, fontweight='bold')
        for tick_label, fname in zip(ax.get_yticklabels(), names):
            tick_label.set_color(color_by_name[fname])
        ax.tick_params(axis='y', length=0)

        mean_val = float(np.mean(vals))
        median_val = float(np.median(vals))

        # Solid line at mean, dashed line at median.
        ax.axvline(mean_val, color=TIMESCALE_AXIS_COLOR,
                   linestyle='-', linewidth=1.0, zorder=4)
        ax.axvline(median_val, color=TIMESCALE_AXIS_COLOR,
                   linestyle='--', linewidth=1.0, zorder=4)

        # ▲ pointer + bold value text on the x-axis at the mean.
        # Blended transform so x is in data coords (lag seconds) and
        # y is in axes fraction (pinned to the axis line). `clip_on`
        # is set False so the marker / text can extend slightly
        # below the panel.
        x_data_y_axes = mtransforms.blended_transform_factory(
            ax.transData, ax.transAxes
        )
        ax.plot(mean_val, 0, marker='^', color=TIMESCALE_AXIS_COLOR,
                markersize=7, linestyle='None', zorder=6,
                clip_on=False, transform=x_data_y_axes)
        # Mean numeric annotation sits just under the axis spine
        # (most tick labels are blank, so vertical space is tight
        # but uncluttered). The x-axis label goes further down via
        # `labelpad`.
        ax.text(mean_val, -0.015, f'{mean_val:.2f} s',
                ha='center', va='top', fontweight='bold',
                fontsize=8, color=TIMESCALE_AXIS_COLOR,
                transform=x_data_y_axes, clip_on=False)

        # Shared x-range and sparse tick labelling (only first and
        # last tick get a label).
        ax.set_xlim(0, x_max_int)
        ax.set_xticks(shared_xticks)
        ax.set_xticklabels(shared_xticklabels)
        ax.set_ylim(-0.5, len(items) - 0.5)
        # `labelpad` pushes the x-axis label below the bold mean
        # annotation drawn above it.
        ax.set_xlabel(x_axis_label)
        ax.set_title(panel_title, fontsize=11)
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', which='both', length=0)

        return mean_val, median_val

    mean_acf, median_acf = _render_panel(
        axA, acf_items,
        panel_title='Auto-correlation horizon',
        x_axis_label='Lag (s)',
    )
    mean_xc, median_xc = _render_panel(
        axB, xc_items,
        panel_title='Cross-correlation horizon (behavior leads)',
        x_axis_label='Lag (s)',
    )

    # Mean / median legend, drawn on the right subplot in the lower-
    # right corner with one entry per row (vertical stack). Single
    # legend serves both panels because the line styles are shared.
    legend_handles = [
        Line2D([0], [0], color=TIMESCALE_AXIS_COLOR, linestyle='-',
               linewidth=1.0, label='mean'),
        Line2D([0], [0], color=TIMESCALE_AXIS_COLOR, linestyle='--',
               linewidth=1.0, label='median'),
    ]
    axB.legend(handles=legend_handles, loc='lower right',
               ncol=1, fontsize=9, frameon=False)

    if save_plot_bool:
        if save_dir is None:
            save_dir = os.path.dirname(timescale_pkl_path)
        base = os.path.splitext(os.path.basename(timescale_pkl_path))[0]
        out_path = _save_audit_figure(fig, save_dir, base, file_format=plot_format)
        plt.close(fig)
        print(f"[plot] timescale figure written: {out_path}")
    else:
        out_path = ''
    return {
        'figure_path': out_path,
        'n_features_acf': n_acf,
        'n_features_xc': n_xc,
        'n_features_total': n_features_total,
        'mean_acf_horizon_s': mean_acf,
        'median_acf_horizon_s': median_acf,
        'mean_xc_horizon_s': mean_xc,
        'median_xc_horizon_s': median_xc,
        'configured_filter_history': cfg_hist,
    }


def plot_timescale_audit_per_feature(timescale_pkl_path: str,
                                     save_dir: str = None,
                                     save_plot_bool: bool = True,
                                     plot_format: str = 'svg',
                                     signal_smooth_window: int = 0) -> dict:
    """
    Renders the timescale audit as a small-multiples grid: one row per
    feature, two columns (ACF on the left, signal correlation on the
    right). Complements `plot_timescale_audit` (which overlays every
    feature on a single pair of axes) by giving each feature its own
    panel so the per-feature shape, peak location, and null margin are
    legible without colour discrimination across 20+ overlapping curves.

    Layout
    ------
    - One row per feature, two columns. Each subplot is titled with
      the feature name in the feature's group colour.
    - Left column: ACF (median ± IQR across sessions, positive lags).
      A grey band shows the shuffled-feature null at 0.5 / 99.5
      percentiles, with a thin dashed line on the upper boundary so
      the threshold is visible even when the band collapses to
      near-zero. A downward triangle marks the latest lag at which
      15 consecutive bins (≈ 100 ms at 150 fps) are still above the
      upper null band, with the lag time annotated above (bold,
      "{value} s"). Y-limits are `[0, 1.05]`.
    - Right column: symmetric signal-correlation curve (ρ vs. binary
      bout-onset indicator). The actual curve is the per-session
      mean (negative lag = bout leads feature; positive lag =
      feature leads bout); the SEM-across-sessions is drawn as a
      filled band around it. The circular-shift null is on the
      cohort-mean scale (shuffles paired by index across sessions,
      cohort-mean per shuffle, 0.5/99.5 percentiles across the
      `n_shuffles` cohort-mean curves) and is shown as a grey filled
      band plus a pair of thin dashed grey lines at the upper /
      lower envelope. The dashed lines guarantee the null threshold
      stays visible even when the band collapses below pixel
      resolution because the actual peak is large (e.g. nose-nose
      at ρ ≈ 0.3 with cohort-mean null at ±0.001). A triangle
      marker on the *positive lag axis* indicates the cross-
      correlation right-side significance horizon, defined as the
      end (largest lag) of the **earliest-starting sign-consistent
      outside-null run** on the positive lag axis. Above-null and
      below-null runs are tracked separately; the earliest-starting
      qualifying run wins (ties broken toward the smaller end index)
      and its direction sets the marker shape (▽
      for above-null, △ for below-null). Two filters apply before
      the earliest-run pick: runs whose last bin is below
      `signal_floor_seconds` are excluded (so very-near-zero noise
      cannot anchor a marker), and runs shorter than
      `signal_min_run_seconds` (in seconds, converted to bins via
      the lag-axis spacing) are excluded as scattered fragments.
      Reading: "the first sustained excursion of the curve away
      from the shuffled distribution ends at this lag." When the
      earliest qualifying run extends to `+max_lag`, the marker
      lands at the right edge of the panel and the lag annotation
      gets a trailing `+`. Same min-x text clamp as the ACF marker
      prevents the label from overrunning the y-axis when the
      marker is close to lag 0; right-edge alignment when the
      marker is close to `+max_lag`. Lag annotation is bold,
      formatted as "{value} s" (or "{value} s+" in the
      exceeds-window case). Y-limits are auto-fit to whatever
      artists are drawn — the actual cohort-mean curve, the SEM
      band, and the null envelope — so each feature's panel uses
      its natural asymmetric range rather than
      mirroring around zero.

    Parameters
    ----------
    timescale_pkl_path : str
        Path to the `_timescales.pkl` artifact produced by
        `audit_predictor_timescales`.
    save_dir : str, optional
        Output directory. Defaults to the directory containing the
        timescale pickle. Only consulted when `save_plot_bool` is True.
    save_plot_bool : bool, default True
        When True (default), the figure is written to disk via
        `_save_audit_figure` and closed. When False, the figure is
        neither saved nor closed — the caller can display it inline
        (notebook) or further customise it. `figure_path` in the
        returned dict is `''` in that case.
    plot_format : str, default 'svg'
        Matplotlib `savefig` format. Only consulted when
        `save_plot_bool` is True.
    signal_smooth_window : int, default 0
        Rolling-average window size (in lag bins) applied to the
        signal-correlation mean curve and SEM band at plot time only —
        the artifact remains raw. `0` or `1` disables smoothing.
        `10` ≈ 67 ms at 150 fps. Even windows are bumped odd-up so
        the smoothing stays centred.

    Returns
    -------
    dict
        `{'figure_path': str, 'n_features': int, 'configured_filter_history': float}`.
        `figure_path` is `''` when `save_plot_bool` is False.
    """

    with open(timescale_pkl_path, 'rb') as fh:
        payload = pickle.load(fh)
    _require_timescale_payload(payload, timescale_pkl_path)

    feature_names = payload['features']
    acf_lags_seconds = np.asarray(payload['acf_lags_seconds'])
    acf_med = np.asarray(payload['acf_median'])
    acf_p25 = np.asarray(payload['acf_p25'])
    acf_p75 = np.asarray(payload['acf_p75'])
    signal_lags_seconds = np.asarray(payload['signal_lags_seconds'])
    rho_signal = np.asarray(payload['rho_signal'])
    cfg_hist = float(payload['configured_filter_history'])
    # Lag floor (in seconds) for the cross-correlation right-side
    # significance marker — runs whose last bin is below this floor
    # are excluded. Falls back to a 0.5 s default when older
    # artifacts (pre-floor) are read.
    signal_floor_seconds = float(payload['signal_floor_seconds']) \
        if 'signal_floor_seconds' in payload else 0.5
    # Minimum sustained-run length (seconds) for the cross-correlation
    # marker. The marker is placed at the end of the longest
    # sign-consistent outside-null run on the positive lag axis whose
    # length is at least this many seconds. Falls back to 0.2 s for
    # older artifacts that pre-date the field.
    signal_min_run_seconds = float(payload['signal_min_run_seconds']) \
        if 'signal_min_run_seconds' in payload else 0.2
    source = payload['source_pickle']

    # ACF circular-shift null (per-feature, per-lag): 0.5/99.5 percentile
    # band, computed by random circular shifts in
    # `[shuffle_min_seconds, shuffle_max_seconds]` per session. Only the band
    # is drawn (no null-mean curve), so the mean is not loaded here.
    acf_null_lo = np.asarray(payload['acf_null_p0_5'])
    acf_null_hi = np.asarray(payload['acf_null_p99_5'])

    # Per-session SEM around the actual cross-correlation curve.
    rho_signal_sem = np.asarray(payload['rho_signal_per_session_sem'])

    # Cross-correlation circular-shift null (per-feature, per-lag): 0.5/99.5
    # percentile band only (no null-mean curve is drawn, so it is not loaded).
    rho_signal_null_lo = np.asarray(payload['rho_signal_null_p0_5'])
    rho_signal_null_hi = np.asarray(payload['rho_signal_null_p99_5'])

    n_features = len(feature_names)
    if n_features == 0:
        print(f"[plot] timescale audit at {timescale_pkl_path} has no features — skipping.")
        return {'figure_path': '', 'n_features': 0, 'configured_filter_history': cfg_hist}

    # Group + colour features: self → SEI → other → social-dyadic.
    order, ordered_colors = _order_and_color_predictor_features(
        feature_names, source
    )

    # Run-length used for the ACF "leaves the band" marker. 15 bins
    # ≈ 100 ms at 150 fps — request from the user.
    acf_run_length = 15

    # Compact layout: each subplot is near-square with the horizontal
    # dimension slightly larger than the vertical. Two columns ×
    # ~2.1 in wide each gives ~4.2 in total figure width; row height
    # 1.85 in.
    fig, axes = plt.subplots(
        n_features, 2,
        figsize=(4.2, 1.85 * n_features),
        sharex=False, squeeze=False,
    )

    # Marker size (in matplotlib points) used by both the ACF and
    # cross-correlation triangle markers. Hoisted out of the
    # per-feature loop so the cross-correlation block can reuse it
    # even when the ACF run-out branch did not draw a marker for
    # this feature.
    marker_size = 7

    for new_i, orig_i in enumerate(order):
        fname = feature_names[orig_i]
        col = ordered_colors[new_i]

        # Column 1 — ACF
        axA = axes[new_i, 0]
        # Circular-shift ACF null: shaded band between 0.5 and 99.5
        # percentiles. No mean line — it sits very close to zero and
        # is visually redundant with the band centre.
        axA.fill_between(acf_lags_seconds,
                         acf_null_lo[orig_i], acf_null_hi[orig_i],
                         color=TIMESCALE_NULL_COLOR, alpha=0.30, linewidth=0)
        # Upper-boundary dashed line for the ACF null, mirroring the
        # cross-correlation null envelope. The lower boundary sits
        # very close to zero (ACF is non-negative in practice) so a
        # second line there is visually redundant.
        axA.plot(acf_lags_seconds, acf_null_hi[orig_i],
                 color=TIMESCALE_NULL_COLOR, linewidth=0.6, linestyle='--', alpha=0.85)
        axA.fill_between(acf_lags_seconds, acf_p25[orig_i], acf_p75[orig_i],
                         color=col, alpha=0.20, linewidth=0)
        axA.plot(acf_lags_seconds, acf_med[orig_i], color=col, linewidth=1.5)

        # Mark the latest lag at which `acf_run_length` consecutive
        # bins are still above the upper null band — the practical
        # decorrelation horizon. Annotate the lag in seconds above
        # the marker. The down-triangle is offset upward by half its
        # height so the apex (bottom tip) lands on the ACF curve
        # rather than the centroid passing through the curve. The
        # text annotation has a minimum y in data units so it
        # doesn't collide with the x-axis when the curve is near
        # zero (e.g. fast-decorrelating features like
        # `other.usv_rate`), and a minimum x (as a fraction of the
        # x-axis span) so the centred label doesn't overrun the
        # y-axis when the marker lag is small (e.g. ~0.2 s on a
        # 10 s axis).
        above_mask = acf_med[orig_i] > acf_null_hi[orig_i]
        mark_idx = _last_bin_of_consecutive_run(above_mask, acf_run_length)
        if mark_idx is not None:
            lag_s = float(acf_lags_seconds[mark_idx])
            y_at = float(acf_med[orig_i, mark_idx])
            # Shift the down-triangle up by half its height (in points)
            # so its apex lands on the curve rather than the centroid.
            tip_transform = mtransforms.offset_copy(
                axA.transData, fig=fig, x=0, y=marker_size / 2.0, units='points'
            )
            # Black outline (`#000000`) on the filled triangle so the
            # marker reads cleanly against any per-feature fill colour.
            axA.plot(lag_s, y_at, marker='v', color=col,
                     markersize=marker_size, linestyle='None', zorder=5,
                     markeredgecolor='#000000', markeredgewidth=0.6,
                     transform=tip_transform)
            # Text anchor: hold a minimum data-y so the label clears
            # the x-axis when y_at is small. To keep the label clear
            # of the y-axis when lag_s is small, switch the
            # horizontal alignment from `center` to `left` once the
            # marker enters the leftmost band of the panel — and
            # additionally clamp the text x to a minimum data-x so
            # the left edge of the label sits a comfortable distance
            # right of the y-axis spine. The marker itself still
            # draws at the actual (lag_s, y_at).
            text_y_anchor = max(y_at, 0.08)
            x_axis_max = float(acf_lags_seconds[-1]) if acf_lags_seconds.size else 1.0
            min_text_x = 0.04 * x_axis_max
            left_align_threshold = 0.18 * x_axis_max
            if lag_s < left_align_threshold:
                text_ha = 'left'
                text_x_anchor = max(lag_s, min_text_x)
            else:
                text_ha = 'center'
                text_x_anchor = lag_s
            axA.annotate(f'{lag_s:.2f} s',
                         xy=(text_x_anchor, text_y_anchor),
                         xytext=(0, 12), textcoords='offset points',
                         ha=text_ha, va='bottom',
                         color=col, fontsize=7, fontweight='bold',
                         annotation_clip=False)

        axA.set_xlim(0, acf_lags_seconds[-1] if acf_lags_seconds.size else 1.0)
        axA.set_ylim(0.0, 1.05)
        axA.set_ylabel('autocorrelation (ρ)', fontsize=8)
        axA.set_title(fname, fontsize=9, color=col, pad=3, fontweight='bold')
        axA.tick_params(labelsize=7)
        if new_i == n_features - 1:
            axA.set_xlabel('Lag (s)')
        else:
            axA.set_xticklabels([])

        # Column 2 — Signal correlation. Per-session mean (line) ±
        # SEM (filled band), with the circular-shift null mean +
        # 0.5/99.5 percentile band underneath. Y-limits are per-plot
        # so each feature's full peak structure is visible regardless
        # of cross-feature magnitude differences.
        axB = axes[new_i, 1]
        # Null band first, so the actual curve and SEM draw over it.
        # The cohort-mean null is `~σ_session/√n_sessions` wide — for
        # high-ρ features (e.g. nose-nose with peak ρ ≈ 0.3) this is
        # ~0.3% of the y-axis and the fill collapses below pixel
        # resolution. Overlay thin dashed lines at the upper / lower
        # null envelope so the threshold is always visible regardless
        # of the per-feature y-scale; the fill carries the per-lag
        # shape when the band is comparable scale to the actual curve.
        axB.fill_between(signal_lags_seconds,
                         rho_signal_null_lo[orig_i], rho_signal_null_hi[orig_i],
                         color=TIMESCALE_NULL_COLOR, alpha=0.25, linewidth=0)
        axB.plot(signal_lags_seconds, rho_signal_null_hi[orig_i],
                 color=TIMESCALE_NULL_COLOR, linewidth=0.6, linestyle='--', alpha=0.85)
        axB.plot(signal_lags_seconds, rho_signal_null_lo[orig_i],
                 color=TIMESCALE_NULL_COLOR, linewidth=0.6, linestyle='--', alpha=0.85)
        # Optional rolling-average smoothing for plotting only — the
        # artifact stays raw. Applied symmetrically to mean and SEM so
        # the band stays consistent with the (smoothed) curve.
        mean_curve_plot = _rolling_mean_1d(rho_signal[orig_i], signal_smooth_window)
        sem_curve_plot = _rolling_mean_1d(rho_signal_sem[orig_i], signal_smooth_window)
        axB.fill_between(signal_lags_seconds,
                         mean_curve_plot - sem_curve_plot,
                         mean_curve_plot + sem_curve_plot,
                         color=col, alpha=0.25, linewidth=0)
        axB.plot(signal_lags_seconds, mean_curve_plot, color=col, linewidth=1.5)
        axB.axhline(0, color=TIMESCALE_AXIS_COLOR, linewidth=0.5)
        axB.axvline(0, color=TIMESCALE_AXIS_COLOR, linewidth=0.6)
        if signal_lags_seconds.size:
            axB.set_xlim(signal_lags_seconds[0], signal_lags_seconds[-1])

        # Cross-correlation right-side significance marker.
        #
        # Algorithm: earliest-starting sign-consistent outside-null
        # run on the symmetric lag axis (above-null and below-null
        # are tracked as separate runs). Runs whose last bin is
        # below `signal_floor_seconds` are excluded; runs shorter
        # than `signal_min_run_seconds` × fps are excluded as
        # noise-fragments. Among the remainder, the run with the
        # smallest start lag is selected (ties broken toward smaller
        # end index). The marker sits at the **end** (largest lag)
        # of that run, with sign matching the run direction (▽ for
        # above-null, △ for below-null).
        #
        # Reading: "the first sustained departure of the curve
        # from the shuffled distribution ends — and the curve
        # re-enters the null envelope — at this lag."
        #
        # `signal_floor_seconds` suppresses very-near-zero noise;
        # `signal_min_run_seconds` suppresses scattered short
        # excursions (e.g. ~28-bin fragments on `other.usv_rate`)
        # while preserving multi-hundred-bin real runs.
        #
        # Edge case: when the picked run extends all the way to
        # `+max_lag`, the marker is drawn at the right edge of the
        # panel and the lag annotation gets a trailing `+` to
        # indicate the horizon exceeds the configured window.
        if signal_lags_seconds.size > 0:
            sig_max_lag_s = float(signal_lags_seconds[-1])
            sig_idx_floor = int(np.searchsorted(
                signal_lags_seconds, signal_floor_seconds, side='left'
            ))
            sig_idx_max = signal_lags_seconds.size - 1
            # Convert the seconds-valued threshold to bins via the
            # lag-axis spacing. `np.ceil` so 0.2 s on a 150-fps grid
            # gives 30 bins (= 0.2 s exactly), never less.
            if signal_lags_seconds.size > 1:
                _delta_t = float(signal_lags_seconds[1] - signal_lags_seconds[0])
                sig_min_run_bins = int(np.ceil(signal_min_run_seconds / _delta_t))
            else:
                sig_min_run_bins = 1
            sig_hit = _signal_outer_run_marker(
                mean_curve_plot,
                rho_signal_null_lo[orig_i],
                rho_signal_null_hi[orig_i],
                sig_min_run_bins,
                sig_idx_floor,
                sig_idx_max,
            )
        else:
            sig_hit = None
            sig_max_lag_s = 1.0

        if sig_hit is not None:
            sig_idx, sig_sign, sig_exceeds = sig_hit
            sig_lag_s = float(signal_lags_seconds[sig_idx])
            sig_y_at = float(mean_curve_plot[sig_idx])
            sig_marker = 'v' if sig_sign > 0 else '^'
            # Tip-offset trick (analogous to the ACF marker): for a
            # ▽ apex pointing down to land on the curve we shift the
            # marker UP by half its height; for △ apex pointing up,
            # shift DOWN.
            sig_offset_pts = (marker_size / 2.0) * (1.0 if sig_sign > 0 else -1.0)
            sig_tip_transform = mtransforms.offset_copy(
                axB.transData, fig=fig, x=0, y=sig_offset_pts, units='points'
            )
            # Black outline (`#000000`) on the filled triangle so the
            # marker reads cleanly against any per-feature fill colour
            # and the apex is visually clear against the curve.
            axB.plot(sig_lag_s, sig_y_at, marker=sig_marker, color=col,
                     markersize=marker_size, linestyle='None', zorder=5,
                     markeredgecolor='#000000', markeredgewidth=0.6,
                     transform=sig_tip_transform)
            # Text x-anchor: same min-x clamp + left/center alignment
            # switch as the ACF marker so the centred label doesn't
            # overrun the y-axis when sig_lag_s is small. When the
            # marker sits at +max_lag (exceeds-window case) we right-
            # align the label so it doesn't run off the right edge.
            sig_min_text_x = 0.04 * sig_max_lag_s
            sig_left_align_threshold = 0.18 * sig_max_lag_s
            sig_right_align_threshold = 0.82 * sig_max_lag_s
            if sig_lag_s < sig_left_align_threshold:
                sig_text_ha = 'left'
                sig_text_x_anchor = max(sig_lag_s, sig_min_text_x)
            elif sig_lag_s > sig_right_align_threshold:
                sig_text_ha = 'right'
                sig_text_x_anchor = sig_lag_s
            else:
                sig_text_ha = 'center'
                sig_text_x_anchor = sig_lag_s
            # Text vertical placement: above the marker for positive
            # separation (▽ sits above the curve), below for negative
            # separation (△ sits below). `va` matches so the text edge
            # nearest the curve lines up against the marker body.
            sig_text_y_offset = 12 if sig_sign > 0 else -12
            sig_text_va = 'bottom' if sig_sign > 0 else 'top'
            sig_label = f'{sig_lag_s:.2f} s+' if sig_exceeds else f'{sig_lag_s:.2f} s'
            axB.annotate(sig_label,
                         xy=(sig_text_x_anchor, sig_y_at),
                         xytext=(0, sig_text_y_offset),
                         textcoords='offset points',
                         ha=sig_text_ha, va=sig_text_va,
                         color=col, fontsize=7, fontweight='bold',
                         annotation_clip=False)

        # No explicit y-limit — matplotlib auto-fits to the union of
        # all artists drawn (mean curve, SEM band, null fill, null
        # envelope dashed lines). Lets each feature's curve be shown
        # at its natural asymmetric range rather than padding the
        # opposite side to a symmetric mirror of the peak. (Was
        # previously `set_ylim(-feat_ymax, feat_ymax)`.)
        axB.tick_params(labelsize=7)
        # Push the right-column y-tick labels to the outer (right)
        # edge so they don't overlap the left column's plot area in
        # the narrower figure.
        axB.yaxis.tick_right()
        axB.yaxis.set_label_position('right')
        axB.set_ylabel('cross-correlation (ρ)', fontsize=8)
        axB.set_title(fname, fontsize=9, color=col, pad=3, fontweight='bold')
        if new_i == n_features - 1:
            axB.set_xlabel('bout leads --- Lag (s) --- feature leads')
        else:
            axB.set_xticklabels([])

    # Tighter left margin (vertical y-labels), a touch of right margin
    # for the moved-to-the-right column-2 y-ticks, and a bit of extra
    # vertical space between rows so per-row titles don't collide
    # with the row above. No figure-level suptitle — the per-row
    # bold feature-name titles carry the identity, and the source
    # filename is on the saved file path.
    fig.subplots_adjust(left=0.08, right=0.92, top=0.98,
                        bottom=0.04, hspace=0.45, wspace=0.22)

    if save_plot_bool:
        if save_dir is None:
            save_dir = os.path.dirname(timescale_pkl_path)
        base = os.path.splitext(os.path.basename(timescale_pkl_path))[0]
        out_path = _save_audit_figure(fig, save_dir, f"{base}_per-feature",
                                      file_format=plot_format)
        plt.close(fig)
        print(f"[plot] per-feature timescale figure written: {out_path}")
    else:
        out_path = ''
    return {
        'figure_path': out_path,
        'n_features': n_features,
        'configured_filter_history': cfg_hist,
    }
