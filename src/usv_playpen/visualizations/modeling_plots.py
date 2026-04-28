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
import glob
import json
import math
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.mlab as mlab
from matplotlib.colors import ListedColormap
from matplotlib.path import Path
from matplotlib.patches import Patch, Rectangle, ConnectionPatch
from matplotlib.lines import Line2D
import numpy as np
import os
import pathlib
import pickle
import re
import seaborn as sns
from scipy.stats import gaussian_kde
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from typing import Optional
import h5py
import polars as pls
from tqdm import tqdm
from scipy.interpolate import griddata
from scipy.interpolate import interp1d

from .auxiliary_plot_functions import create_colormap
from ..modeling.modeling_metadata import load_selection_results


_PKG_ROOT = pathlib.Path(__file__).resolve().parent.parent
fm.fontManager.addfont(str(_PKG_ROOT / 'fonts' / 'Helvetica.ttf'))
plt.style.use(str(_PKG_ROOT / '_config' / 'usv_playpen.mplstyle'))

# Global color definitions
male_color = "#9AC0CD"
female_color = "#FF6347"
DYADIC_COLOR = "#000000"
NEUTRAL_COLOR = "#D3D3D3"
MEAN_LINE_COLOR = '#DCB400'
TEXT_COLOR = '#202020'

# Initialize custom colormaps (for males and females)
female_cmap = create_colormap(input_parameter_dict={
    "cm_length": 255,
    "cm_name": "female_cm",
    "cm_type": "sequential",
    "cm_start": (
        int(female_color[1:3], 16),
        int(female_color[3:5], 16),
        int(female_color[5:7], 16),
    ),
    "cm_end": (255, 255, 255),
    "equalize_luminance": True,
    "match_luminance_by": "max",
    "change_saturation": 0.5,
    "cm_opacity": 1,
})

male_cmap = create_colormap(input_parameter_dict={
    "cm_length": 255,
    "cm_name": "male_cm",
    "cm_type": "sequential",
    "cm_start": (
        int(male_color[1:3], 16),
        int(male_color[3:5], 16),
        int(male_color[5:7], 16),
    ),
    "cm_end": (255, 255, 255),
    "equalize_luminance": True,
    "match_luminance_by": "max",
    "change_saturation": 0.5,
    "cm_opacity": 1,
})


def plot_vocalization_embedding_space(
        umap_position_file_path: str,
        cluster_category_file_path: str,
        x_range: tuple = None,
        y_range: tuple = None,
        target_subdirs: list = None,
        csv_category_column_id: str = 'usv_supercategory',
        grid_res: int = 600,
        cmap_name: str = 'tab20',
        spec_cmap_name: str = 'magma',
        border_color: str = '#000000',
        point_size: float = 0.5,
        point_alpha: float = 0.6,
        grid_dims: tuple = (4, 4),
        figsize: tuple = (20, 10),
        save_plot: bool = False,
        output_dir: str = None
) -> None:
    """
    Extracts, aligns, and plots the global USV UMAP manifold colored by supercategory,
    with calculated boundary lines separating distinct vocalization territories.
    Optionally draws an expanded spectrogram inset for a target region.

    This function serves a dual purpose for deep vocal analysis:

    1. Global Manifold Mapping (Default):
       Cross-references HDF5 coordinate/label data with CSV metadata to filter out
       noise (category 0) and perfectly align the valid USVs. It uses a nearest-neighbor
       grid interpolation to mathematically define the boundaries between behavioral
       categories and draws sharp 'cracks' where different behavioral states collide.

    2. Regional Audio Inspection (If x_range and y_range are provided):
       Calculates an evenly spaced grid across the specified UMAP bounding box. For each
       ideal grid intersection, it finds the nearest real USV to sample the topology evenly.
       It locates the raw .mmap 24-channel audio file for each USV, calculates the exact
       sample indices using the sampling rate embedded in the filename, and extracts
       the sequence. It calculates the variance across all 24 channels to weight and
       average them into a single high-SNR audio array.

       Each spectrogram is independently auto-scaled (vmin = local min, vmax = local max)
       to ensure maximum contrast regardless of the specific USV's absolute volume. The grid
       of spectrograms has custom frequency y-limits of 30-125 kHz with explicit, minimal kHz labels.
       Because USV durations vary, every spectrogram maintains its own independent Time (x) axis.

    Parameters
    ----------
    umap_position_file_path : str
        Path to the .h5 file containing the UMAP coordinates for each session.
    cluster_category_file_path : str
        Path to the .h5 file containing the cluster labels for each session.
    x_range : tuple, optional
        The (min, max) boundaries for the target UMAP X-axis region. Use None for open bounds.
    y_range : tuple, optional
        The (min, max) boundaries for the target UMAP Y-axis region. Use None for open bounds.
    target_subdirs : list of str, optional
        Directories to search for the metadata CSV files and .mmap audio files.
    csv_category_column_id : str, default 'usv_supercategory'
        The specific column in the CSV file to use for category labels.
    grid_res : int, default 600
        The resolution of the invisible meshgrid used to calculate the territorial boundaries.
    cmap_name : str, default 'tab20'
        The matplotlib colormap used to assign colors to the distinct categories on the UMAP.
    spec_cmap_name : str, default 'magma'
        The matplotlib colormap used to plot the spectrograms.
    border_color : str, default '#000000'
        The color of the boundary lines drawn between the UMAP supercategories.
    point_size : float, default 0.5
        The size of the individual scatter points on the UMAP.
    point_alpha : float, default 0.6
        The opacity of the scatter points.
    grid_dims : tuple, default (4, 4)
        The (rows, columns) shape of the sampled spectrogram grid.
    figsize : tuple, default (20, 10)
        The dimensions of the generated matplotlib figure.
    save_plot : bool, default False
        If True, saves the figure to disk.
    output_dir : str, optional
        The directory where the plot will be saved.

    Returns
    -------
    None
        Displays the generated Matplotlib figure.
    """
    if target_subdirs is None:
        target_subdirs = ["Liza/data", "Jinrun/Data", "Bartul/Data"]

    all_coordinates = []
    all_categories = []
    candidate_usvs = []

    print("Extracting coordinates, metadata, and matching supercategories...")

    # --- 1. Data Extraction & Alignment ---
    with h5py.File(umap_position_file_path, mode='r') as umap_position_file, \
            h5py.File(cluster_category_file_path, mode='r') as super_category_h5_file:

        for session_key in tqdm(umap_position_file.keys(), desc="Scanning Sessions"):
            if session_key not in super_category_h5_file:
                continue

            session_coords = umap_position_file[session_key][:]
            session_labels = super_category_h5_file[session_key][:].flatten()

            search_term = session_key[:15]
            usv_csv_file = None

            for subdir in target_subdirs:
                current_search_root = pathlib.Path('/mnt/falkner') / subdir
                if current_search_root.exists():
                    for path in current_search_root.glob(f"*{search_term}"):
                        if path.is_dir():
                            usv_csv_file = next(path.glob(f"**{os.sep}*_usv_summary.csv"), None)
                            if usv_csv_file:
                                break
                if usv_csv_file:
                    break

            if usv_csv_file and session_coords.size > 0 and len(session_coords) == len(session_labels):
                try:
                    usv_summary_df = pls.read_csv(usv_csv_file)

                    # Dynamically pulling the requested category column
                    session_supercats = usv_summary_df[csv_category_column_id].to_numpy()

                    if len(session_coords) == len(session_supercats):
                        valid_mask = (session_labels != 0)
                        filtered_coords = session_coords[valid_mask]
                        filtered_cats = session_supercats[valid_mask]

                        all_coordinates.append(filtered_coords)
                        all_categories.append(filtered_cats)

                        if x_range is not None and y_range is not None:
                            starts = usv_summary_df['start'].to_numpy()[valid_mask]
                            stops = usv_summary_df['stop'].to_numpy()[valid_mask]

                            x_min_bound = x_range[0] if x_range[0] is not None else -np.inf
                            x_max_bound = x_range[1] if x_range[1] is not None else np.inf
                            y_min_bound = y_range[0] if y_range[0] is not None else -np.inf
                            y_max_bound = y_range[1] if y_range[1] is not None else np.inf

                            in_box = (filtered_coords[:, 0] >= x_min_bound) & (filtered_coords[:, 0] <= x_max_bound) & \
                                     (filtered_coords[:, 1] >= y_min_bound) & (filtered_coords[:, 1] <= y_max_bound)

                            valid_indices = np.where(in_box)[0]
                            for idx in valid_indices:
                                candidate_usvs.append({
                                    'session_key': session_key,
                                    'start': starts[idx],
                                    'stop': stops[idx],
                                    'x': filtered_coords[idx, 0],
                                    'y': filtered_coords[idx, 1],
                                    'csv_dir': usv_csv_file.parent
                                })
                except Exception:
                    pass

    # --- 2. Aggregation & Boundary Math ---

    if len(all_coordinates) == 0:
        print("No valid coordinates extracted. Exiting plot generation.")
        return

    all_coordinates = np.vstack(all_coordinates)
    all_categories = np.concatenate(all_categories)
    print(f"Total valid USVs extracted: {all_coordinates.shape[0]}")

    print(f"Computing regional boundary lines on a {grid_res}x{grid_res} grid...")
    unique_cats, cat_indices = np.unique(all_categories, return_inverse=True)

    x_min, x_max = all_coordinates[:, 0].min() - 1, all_coordinates[:, 0].max() + 1
    y_min, y_max = all_coordinates[:, 1].min() - 1, all_coordinates[:, 1].max() + 1

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_res),
                         np.linspace(y_min, y_max, grid_res))

    Z = griddata(all_coordinates, cat_indices, (xx, yy), method='nearest')

    # --- 3. Figure Setup ---
    TEXT_COLOR = '#000000'
    BG_COLOR = '#FFFFFF'
    HIGHLIGHT_COLOR = '#000000'

    is_dual_panel = (x_range is not None and y_range is not None and len(candidate_usvs) > 0)

    if is_dual_panel:
        fig = plt.figure(figsize=figsize, facecolor=BG_COLOR, dpi=200)
        fig.patch.set_alpha(1.0)
        gs_main = gridspec.GridSpec(1, 2, width_ratios=[1, 1.5], figure=fig, wspace=0.1)
        ax_umap = fig.add_subplot(gs_main[0])
    else:
        single_figsize = (14, 12) if figsize == (20, 10) else figsize
        fig, ax_umap = plt.subplots(figsize=single_figsize, facecolor=BG_COLOR, dpi=300)
        fig.patch.set_alpha(1.0)

    ax_umap.set_facecolor(BG_COLOR)
    umap_cmap = plt.get_cmap(cmap_name)

    for idx, cat in enumerate(unique_cats):
        cat_mask = (all_categories == cat)
        ax_umap.scatter(
            all_coordinates[cat_mask, 0],
            all_coordinates[cat_mask, 1],
            s=point_size,
            alpha=point_alpha,
            label=str(cat),
            color=umap_cmap(idx % len(unique_cats)),
            edgecolors='none',
            zorder=1
        )

    ax_umap.contour(xx, yy, Z,
                    levels=np.arange(len(unique_cats) + 1) - 0.5,
                    colors=border_color,
                    linewidths=1.5,
                    zorder=2)

    ax_umap.set_title("USV UMAP manifold by category", fontsize=16, color=TEXT_COLOR, pad=15)
    ax_umap.set_xlabel("UMAP 1", fontsize=14, color=TEXT_COLOR)
    ax_umap.set_ylabel("UMAP 2", fontsize=14, color=TEXT_COLOR)
    ax_umap.set_aspect('equal', adjustable='datalim')

    ax_umap.tick_params(colors=TEXT_COLOR, labelsize=10)
    for spine in ax_umap.spines.values():
        spine.set_edgecolor(TEXT_COLOR)
        spine.set_visible(True)
        spine.set_linewidth(1.5)

    leg_fontsize = 10 if is_dual_panel else 12
    leg = ax_umap.legend(title=csv_category_column_id.replace('_', ' ').title(),
                         loc='lower right',
                         markerscale=20, frameon=True, fontsize=leg_fontsize,
                         facecolor=BG_COLOR, framealpha=0.9)

    for lh in leg.legend_handles:
        lh.set_alpha(1.0)
    leg.get_title().set_color(TEXT_COLOR)
    for text in leg.get_texts():
        text.set_color(TEXT_COLOR)

    # --- 4. Spectrogram Inset Processing ---
    if is_dual_panel:
        rows, cols = grid_dims

        # Compute physical bounds for the dashed box
        rect_x_min = x_range[0] if x_range[0] is not None else all_coordinates[:, 0].min()
        rect_x_max = x_range[1] if x_range[1] is not None else all_coordinates[:, 0].max()
        rect_y_min = y_range[0] if y_range[0] is not None else all_coordinates[:, 1].min()
        rect_y_max = y_range[1] if y_range[1] is not None else all_coordinates[:, 1].max()

        # --- Grid Mapping Nearest Neighbor Algorithm ---
        ideal_x = np.linspace(rect_x_min, rect_x_max, cols)
        ideal_y = np.linspace(rect_y_max, rect_y_min, rows)  # Top to bottom for natural mapping

        sampled_usvs = []
        candidate_coords = np.array([[u['x'], u['y']] for u in candidate_usvs])
        available_mask = np.ones(len(candidate_usvs), dtype=bool)

        for r in range(rows):
            for c in range(cols):
                if not np.any(available_mask):
                    sampled_usvs.append(None)  # Out of candidates
                    continue

                target_x = ideal_x[c]
                target_y = ideal_y[r]

                # Compute Euclidean distance from ideal grid point to all available USVs
                dx = candidate_coords[available_mask, 0] - target_x
                dy = candidate_coords[available_mask, 1] - target_y
                sq_dists = dx ** 2 + dy ** 2

                # Pick the closest available USV
                available_indices = np.where(available_mask)[0]
                best_idx = available_indices[np.argmin(sq_dists)]

                sampled_usvs.append(candidate_usvs[best_idx])
                available_mask[best_idx] = False  # Sample without replacement

        # Highlight ONLY the successfully sampled USVs on the UMAP
        sx = [u['x'] for u in sampled_usvs if u is not None]
        sy = [u['y'] for u in sampled_usvs if u is not None]
        ax_umap.scatter(sx, sy, facecolors='none', edgecolors=HIGHLIGHT_COLOR, s=60, linewidth=2.0, zorder=10)

        rect = patches.Rectangle((rect_x_min, rect_y_min), rect_x_max - rect_x_min, rect_y_max - rect_y_min,
                                 linewidth=2.5, edgecolor=HIGHLIGHT_COLOR, facecolor='none', linestyle='--', zorder=10)
        ax_umap.add_patch(rect)

        gs_specs = gs_main[1].subgridspec(rows, cols, wspace=0.05, hspace=0.45)
        spec_axes = []
        for i in range(rows * cols):
            r = i // cols
            c = i % cols
            ax = fig.add_subplot(gs_specs[r, c])
            ax.set_facecolor(BG_COLOR)
            spec_axes.append(ax)

        mmap_cache = {}

        for idx, usv in enumerate(tqdm(sampled_usvs, desc="Processing Audio")):
            ax = spec_axes[idx]

            if usv is None:
                ax.axis('off')
                continue

            sess = usv['session_key']

            if sess not in mmap_cache:
                mmap_path = next(usv['csv_dir'].glob(f"**{os.sep}*_24_int16.mmap"), None)
                mmap_cache[sess] = mmap_path

            audio_loc = mmap_cache[sess]
            if audio_loc is None or not audio_loc.exists():
                ax.set_title("Audio Missing", fontsize=8, color='red')
                ax.axis('off')
                continue

            parts = audio_loc.name.replace('.mmap', '').split('_')
            try:
                channel_num = int(parts[-2])
                sample_num = int(parts[-3])
                sample_rate = int(parts[-4])
            except (ValueError, IndexError):
                ax.axis('off')
                continue

            try:
                start_sample = int(np.floor(usv['start'] * sample_rate))
                stop_sample = int(np.floor(usv['stop'] * sample_rate))
                stop_sample = min(stop_sample, sample_num)

                audio_data = np.memmap(filename=audio_loc, dtype=np.int16, mode='r',
                                       shape=(sample_num, channel_num), order='C')
                raw_chunk = audio_data[start_sample:stop_sample, :]

                channel_vars = np.var(raw_chunk, axis=0)
                sum_var = np.sum(channel_vars)
                weights = channel_vars / sum_var if sum_var > 0 else np.ones(channel_num) / channel_num
                weighted_audio = np.sum(raw_chunk * weights, axis=1)

                Pxx, freqs, bins = mlab.specgram(weighted_audio, NFFT=1024, Fs=sample_rate, noverlap=800)
                Pxx_db = 10 * np.log10(Pxx + 1e-10)

                freq_mask = (freqs >= 30000) & (freqs <= 125000)
                visible_Pxx_db = Pxx_db[freq_mask, :]
                local_vmax = np.max(visible_Pxx_db)
                local_vmin = np.percentile(visible_Pxx_db, 5)

                ax.imshow(Pxx_db, aspect='auto', origin='lower',
                          extent=[bins[0], bins[-1], freqs[0], freqs[-1]],
                          cmap=spec_cmap_name, vmin=local_vmin, vmax=local_vmax)

                ax.grid(False)

                ax.set_ylim(30000, 125000)
                ax.set_yticks([30000, 75000, 125000])
                ax.set_yticklabels(['30', '75', '125'])

                ax.set_title(f"X:{usv['x']:.1f} Y:{usv['y']:.1f}", fontsize=9, color=TEXT_COLOR)
                ax.tick_params(colors=TEXT_COLOR, labelsize=8)
                for spine in ax.spines.values():
                    spine.set_edgecolor(TEXT_COLOR)

                if (idx % cols) != 0:
                    ax.set_yticklabels([])
                else:
                    ax.set_ylabel("Frequency (kHz)", fontsize=9, color=TEXT_COLOR)

                ax.set_xlabel("Time (s)", fontsize=8, color=TEXT_COLOR, labelpad=2)
                ax.tick_params(axis='x', colors=TEXT_COLOR, labelsize=7)
                ax.tick_params(axis='y', colors=TEXT_COLOR, labelsize=8)

            except Exception:
                ax.axis('off')

        con1 = ConnectionPatch(xyA=(rect_x_max, rect_y_max), xyB=(0, 1),
                               coordsA="data", coordsB="axes fraction",
                               axesA=ax_umap, axesB=spec_axes[0],
                               color=HIGHLIGHT_COLOR, linestyle=":", linewidth=2.0, zorder=20)

        bottom_left_idx = (rows - 1) * cols
        if bottom_left_idx < len(spec_axes):
            target_ax = spec_axes[bottom_left_idx]
            con2 = ConnectionPatch(xyA=(rect_x_max, rect_y_min), xyB=(0, 0),
                                   coordsA="data", coordsB="axes fraction",
                                   axesA=ax_umap, axesB=target_ax,
                                   color=HIGHLIGHT_COLOR, linestyle=":", linewidth=2.0, zorder=20)
            ax_umap.add_artist(con2)

        ax_umap.add_artist(con1)

    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message="This figure includes Axes that are not compatible with tight_layout")
        plt.tight_layout()

    if save_plot:
        out_dir = pathlib.Path(output_dir) if output_dir else pathlib.Path(umap_position_file_path).parent
        out_dir.mkdir(parents=True, exist_ok=True)

        if is_dual_panel:
            safe_name = f"vocalization_embedding_spectrograms_inset_X{rect_x_min:.1f}-{rect_x_max:.1f}_Y{rect_y_min:.1f}-{rect_y_max:.1f}.svg"
        else:
            safe_name = "global_vocalization_embedding_space.svg"

        save_path = out_dir / safe_name
        fig.savefig(save_path, bbox_inches='tight', facecolor=BG_COLOR, dpi=300)
        print(f"Plot saved to: {save_path}")

    plt.show()


def gauss(mu=0, sigma=1) -> float:
    """Returns a single random number from Gaussian distribution for plotting jitter."""
    x = np.random.normal(mu, sigma, 1)
    return x[0]


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

    first_feat = next(iter(modeling_data.values()))
    if 'shuffled' in first_feat:
        null_key = 'shuffled'
        print("Detected 'shuffled' key (Bout Analysis mode)")
    elif 'null' in first_feat:
        null_key = 'null'
        print("Detected 'null' key (Category Analysis mode)")
    else:
        raise KeyError("Could not find 'shuffled' or 'null' key in results dictionary.")

    print(f"Calculating significance based on: {evaluation_metric_name}...")

    valid_features = [k for k in modeling_data.keys() if k not in ignore_features]
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
        TEXT_COLOR = '#202020'

        for spine in ax.spines.values():
            spine.set_edgecolor(TEXT_COLOR)

        ax.tick_params(axis='x', colors=TEXT_COLOR)
        ax.tick_params(axis='y', colors=TEXT_COLOR)
        ax.xaxis.label.set_color(TEXT_COLOR)
        ax.yaxis.label.set_color(TEXT_COLOR)

        for i, feature_name in enumerate(feats_sorted):

            is_significant = significance_map.get(feature_name, False)

            if is_significant:
                dyadic_keywords = ["nose-nose", "nose-TTI", "TTI-nose", "allo_yaw-nose",
                                   "nose-allo_yaw", "allo_yaw-TTI", "TTI-allo_yaw",
                                   "allo_pitch-nose", "nose-allo_pitch",
                                   "allo_pitch-TTI", "TTI-allo_pitch"]
                if any(x in feature_name for x in dyadic_keywords):
                    feat_color = DYADIC_COLOR
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
        Metric used to determine significance (e.g., 'auc' or 'll'). For
        lower-is-better metrics (ll, nll, rmse, mse, loss) the comparison
        direction is automatically inverted.
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

    first_feat = next(iter(modeling_data.values()))
    if 'shuffled' in first_feat:
        null_key = 'shuffled'
    elif 'null' in first_feat:
        null_key = 'null'
    else:
        raise KeyError("Missing 'shuffled' or 'null' key")

    valid_features = [k for k in modeling_data.keys() if k not in ignore_features]
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
            plt.close(fig)

        plt.show()


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

        ax.axhline(0, color='gray', linestyle='--', lw=0.7)
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
        output_dir: str = None
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

    Returns
    -------
    None
        Displays the plots and optionally saves them to disk.
    """

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

    # Sanity filter: values above 90 (typically angular features in degrees
    # or extreme outliers) are replaced with NaN so they don't bias the
    # per-frame mean/CI. TODO: make this threshold a parameter or feature-
    # dependent; the current cap implicitly assumes an angular feature.
    target_subset[target_subset > 90] = np.nan
    other_subset[other_subset > 90] = np.nan

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

    im1 = axes_heat[0].imshow(
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


def plot_model_selection_results(
        selection_results_dir: str,
        metric_secondary: str = 'auc',
        save_plots: bool = False,
        output_dir: str = None
) -> None:
    """
    Plots the trajectory of a Forward Sequential Feature Selection process and
    visualizes the temporal filters of the final accepted model.

    Visualizes the improvement in the Primary Metric (LL or Explained Deviance)
    and a secondary metric (e.g., AUC) as features are added to the model step-by-step.
    It also generates a grid plot showing the temporal filter shapes for ALL features
    in the final model.

    Parameters
    ----------
    selection_results_dir : str
        Directory containing the '_step_X.pkl' files from the model selection process.
    metric_secondary : str, default 'auc'
        The key for the secondary metric to plot in the right subplot.
    save_plots : bool, default False
        Whether to save the plot to disk.
    output_dir : str, optional
        Directory to save the plot. Defaults to 'selection_results_dir'.
    """

    BG_COLOR = '#FFFFFF'
    COLOR_PRIM_DOT = "#000000"
    COLOR_SEC_DOT = MEAN_LINE_COLOR

    # Load steps via the metadata-aware helper: prefers a consolidated
    # `selection_*.pkl` artifact in the directory, falls back to legacy
    # `*_step_*.pkl` glob. `display_name` keeps the substring-based sex
    # inference below working in both modes.
    selection_steps, display_name, _ = load_selection_results(selection_results_dir)

    if not selection_steps:
        print(f"No step data found in {selection_results_dir}")
        return

    if '_male_' in display_name:
        self_color, other_color = male_color, female_color
    elif '_female_' in display_name:
        self_color, other_color = female_color, male_color
    else:
        self_color, other_color = male_color, female_color

    # --- 1. Determine Valid Steps for Trajectory Plot ---
    valid_steps_for_plot = list(selection_steps)
    if len(selection_steps) > 1:
        last_data = selection_steps[-1]
        if last_data.get('selected_feature') is None:
            print(f"Last step was a rejection. Excluding it from trajectory plots.")
            valid_steps_for_plot = selection_steps[:-1]

    # --- 2. Process Trajectory Data ---
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

    # --- 3. Plot Trajectories ---
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 3.5), dpi=300, constrained_layout=True)
    fig.patch.set_facecolor(BG_COLOR)

    step_indices = [d['step_idx'] for d in steps_data]

    # Plot Primary
    ax_prim = axes[0]
    ax_prim.set_facecolor(BG_COLOR)
    prim_means = [d['prim_mean'] for d in steps_data]
    ax_prim.plot(step_indices, prim_means, color=TEXT_COLOR, alpha=0.6, lw=1.5, zorder=1)

    for i, d in enumerate(steps_data):
        x = d['step_idx']
        y_vals = d['prim_scores']
        ax_prim.scatter([x] * len(y_vals), y_vals, color=COLOR_PRIM_DOT, s=10, alpha=0.6, zorder=2, edgecolors='none')

        y_max, y_min = np.max(y_vals), np.min(y_vals)
        text_y = y_max + ((y_max - y_min) * 0.1 if (y_max - y_min) > 0 else 0.005)
        label_text = d['feature_name'] if i == 0 else f"+{d['feature_name']}"
        ha_align = 'left' if i == 0 else 'center'
        ax_prim.text(x, text_y, label_text, fontsize=6, ha=ha_align, va='bottom', color=TEXT_COLOR)

    ax_prim.set_title(f'Model Fit ({metric_label})', fontsize=10, color=TEXT_COLOR)
    ax_prim.set_ylabel(metric_label, fontsize=9, color=TEXT_COLOR)
    ax_prim.set_xlabel('Model Step', fontsize=9, color=TEXT_COLOR)

    ax_sec = axes[1]
    ax_sec.set_facecolor(BG_COLOR)
    sec_means = [d['sec_mean'] for d in steps_data]
    ax_sec.plot(step_indices, sec_means, color=MEAN_LINE_COLOR, alpha=0.6, lw=1.5, zorder=1)

    for i, d in enumerate(steps_data):
        x = d['step_idx']
        y_vals = d['sec_scores']
        ax_sec.scatter([x] * len(y_vals), y_vals, color=COLOR_SEC_DOT, s=10, alpha=0.4, zorder=2, edgecolors='none')
        y_max, y_min = np.max(y_vals), np.min(y_vals)
        text_y = y_max + ((y_max - y_min) * 0.1 if (y_max - y_min) > 0 else 0.005)
        label_text = d['feature_name'] if i == 0 else f"+{d['feature_name']}"
        ha_align = 'left' if i == 0 else 'center'
        ax_sec.text(x, text_y, label_text, fontsize=6, ha=ha_align, va='bottom', color=TEXT_COLOR)

    ax_sec.set_title(f'Performance ({metric_secondary.upper()})', fontsize=10, color=TEXT_COLOR)
    ax_sec.set_ylabel(metric_secondary.upper(), fontsize=9, color=TEXT_COLOR)
    ax_sec.set_xlabel('Model Step', fontsize=9, color=TEXT_COLOR)

    for ax in axes:
        ax.set_xticks(step_indices)
        ax.set_xticklabels([f"Step {i}" for i in step_indices], fontsize=8, color=TEXT_COLOR)
        ax.grid(axis='y', linestyle='--', alpha=0.3, color=NEUTRAL_COLOR)
        ax.tick_params(axis='both', colors=TEXT_COLOR, labelsize=8)
        for spine in ax.spines.values(): spine.set_edgecolor(TEXT_COLOR)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        y_l, y_h = ax.get_ylim()
        ax.set_ylim(y_l - (y_h - y_l) * 0.1, y_h + (y_h - y_l) * 0.1)

    plt.show()

    # --- 4. Filter Grid Visualization ---
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
                        # Percentiles: 0.5 and 99.0
                        p_low = np.nanpercentile(feat_matrix, q=0.5, axis=0)
                        p_high = np.nanpercentile(feat_matrix, q=99.0, axis=0)

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
                ax.set_ylabel("Filter amplitude (a.u.)", fontsize=8, color=TEXT_COLOR)

                ax.axhline(0, color=NEUTRAL_COLOR, ls='--', lw=0.5, zorder=0)
                ax.set_title(feature, fontsize=9, color=TEXT_COLOR)
                ax.tick_params(axis='both', colors=TEXT_COLOR, labelsize=7)
                for spine in ax.spines.values():
                    spine.set_edgecolor(TEXT_COLOR)


        except Exception as e:
            print(f"Error plotting feature {feature}: {e}")

    for i in range(n_feats, len(axes_grid)):
        fig_grid.delaxes(axes_grid[i])

    if save_plots:
        if output_dir is None: output_dir = selection_results_dir
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

    valid_features = list(modeling_data.keys())
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
            ax.axhline(1 / n_classes, ls='--', color='gray', alpha=0.5, label='Chance', zorder=0)

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
            mats.append(confusion_matrix(y_t, y_p, normalize='true'))

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

    results_path = pathlib.Path(results_file_loc)
    with open(results_path, 'rb') as f:
        modeling_data = pickle.load(f)

    if '_male_' in results_path.name:
        self_color, other_color = male_color, female_color
    elif '_female_' in results_path.name:
        self_color, other_color = female_color, male_color
    else:
        self_color, other_color = male_color, female_color

    valid_features = list(modeling_data.keys())
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

        im = ax.imshow(weights, aspect='auto', cmap=cmap, vmin=-max_amp, vmax=max_amp, interpolation='nearest')

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
        selection_results_dir: str,
        metric_primary: str = 'auc',
        primary_metric_name: str = "Area Under ROC",
        metric_secondary: str = 'score',
        secondary_metric_name: str = "Balanced Accuracy",
        save_plot: bool = False,
        output_dir: str = None
) -> None:
    """
    Visualizes the stepwise progression and convergence of the forward
    feature selection process for multinomial USV category prediction.

    This function parses the sequential '.pkl' output files generated during
    multivariate modeling to reconstruct the "search path" of the greedy
    algorithm. It provides a dual-panel diagnostic view to evaluate whether
    the inclusion of additional behavioral features provided statistically
    significant improvements in vocal category classification.

    Panel 1: Selection Criterion (Area Under ROC)
    -------------------------------------------
    Displays the Area Under the Receiver Operating Characteristic (AUC) across steps.
    This panel tracks the primary metric used to rank and select features during
    the forward sweep. The plot includes:
    * Fold-level variance: Individual cross-validation folds are plotted as
      jittered points to reveal performance stability across session splits.
    * 1-Standard Error (1SE) Rule: Error bars represent the SE across folds.
      Forward selection logic aims to find the simplest model that performs
      within 1SE of the numerical maximum to prevent over-parameterization.
    * Feature Annotations: Each step is labeled with the specific feature
      incorporated into the model (e.g., "+ self.usv_cat_5"), visualizing
      the incremental information gain.
    * Chance Baseline: A red dashed line at 0.5 marks the performance of
      a random classifier (Step 0).

    Panel 2: Biological Performance (Secondary Metric)
    -------------------------------------------------
    Displays an auxiliary metric, defined by secondary_metric_name (typically
    Balanced Accuracy). This panel demonstrates how the model's categorical
    recall—its ability to correctly identify specific USV types—improves
    as multivariate behavioral context is added.

    Parameters
    ----------
    selection_results_dir : str
        Path to the directory containing the stepwise results (e.g.,
        'step_0.pkl', 'step_1.pkl', etc.).
    metric_primary : str, default='auc'
        The key for the primary performance metric used for selection.
    primary_metric_name : str, default="Area Under ROC"
        The human-readable label for the Y-axis of the primary metric plot.
    metric_secondary : str, default='score'
        The key for the secondary performance metric to be plotted.
    secondary_metric_name : str, default="Balanced Accuracy"
        The human-readable label for the Y-axis of the secondary metric plot.
    save_plot : bool, default=False
        If True, saves the figure to the specified output directory.
    output_dir : str, optional
        The directory where the plot will be saved. Defaults to the
        selection_results_dir if None.

    Returns
    -------
    None
        Displays the generated Matplotlib figure.
    """

    # 1. Local visibility overrides — scoped below so we don't leak into
    #    any plot rendered after this function returns.
    TEXT_COLOR = '#202020'
    _rcp_override = {
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'axes.edgecolor': TEXT_COLOR,
        'axes.labelcolor': TEXT_COLOR,
        'xtick.color': TEXT_COLOR,
        'ytick.color': TEXT_COLOR,
        'text.color': TEXT_COLOR
    }
    _saved_rcp = {k: plt.rcParams[k] for k in _rcp_override}
    plt.rcParams.update(_rcp_override)

    selection_steps, _, _ = load_selection_results(selection_results_dir)

    if not selection_steps:
        print(f"No multinomial step data found in {selection_results_dir}")
        return

    steps_data = []
    for data in selection_steps:
        if data.get('selected_feature') is None: continue

        winner = data['selected_feature']
        winner_data = data['candidates_summary'][winner]

        pri_vals = np.array(winner_data['folds']['metrics'][metric_primary])
        pri_vals = pri_vals[~np.isnan(pri_vals)]
        sec_vals = np.array(winner_data['folds']['metrics'].get(metric_secondary, [np.nan]))
        sec_vals = sec_vals[~np.isnan(sec_vals)]

        steps_data.append({
            'step_idx': data['step_idx'],
            'feature': winner,
            'pri_vals': pri_vals,
            'pri_mean': np.mean(pri_vals) if len(pri_vals) > 0 else 0.5,
            'pri_se': np.std(pri_vals, ddof=1) / np.sqrt(len(pri_vals)) if len(pri_vals) > 1 else 0,
            'sec_vals': sec_vals,
            'sec_mean': np.mean(sec_vals) if len(sec_vals) > 0 else 0
        })

    if not steps_data: return

    # 2. Use a specific constrained_layout alternative: plt.subplot2grid or manual rect
    fig = plt.figure(figsize=(16, 9), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')

    # Create axes manually with specific rects to ensure they never move/clip
    # [left, bottom, width, height]
    ax0 = fig.add_axes([0.1, 0.2, 0.35, 0.55])
    ax1 = fig.add_axes([0.55, 0.2, 0.35, 0.55])
    axes = [ax0, ax1]

    x_steps = [d['step_idx'] for d in steps_data]

    # Plot Panel 0
    pri_means = [d['pri_mean'] for d in steps_data]
    ax0.plot(x_steps, pri_means, color=TEXT_COLOR, lw=2, alpha=0.3)
    for d in steps_data:
        x_jitter = np.random.normal(d['step_idx'], 0.05, size=len(d['pri_vals']))
        ax0.scatter(x_jitter, d['pri_vals'], color='black', s=20, alpha=0.2, edgecolors='none')
        ax0.errorbar(d['step_idx'], d['pri_mean'], yerr=d['pri_se'], fmt='o', color='black', capsize=5)

        label = d['feature'] if d['step_idx'] == 0 else f"+ {d['feature']}"
        ax0.text(d['step_idx'], d['pri_mean'] + d['pri_se'] + 0.01, f" {label}",
                 fontsize=10, rotation=40, va='bottom', ha='left', fontweight='bold')

    ax0.set_title(primary_metric_name, pad=40)
    ax0.set_ylabel(primary_metric_name)
    ax0.axhline(0.5, ls='--', color='red', alpha=0.5)

    # Plot Panel 1
    sec_means = [d['sec_mean'] for d in steps_data]
    ax1.plot(x_steps, sec_means, color=TEXT_COLOR, lw=2, alpha=0.3)
    step0_sec_baseline = steps_data[0]['sec_mean']
    ax1.axhline(step0_sec_baseline, ls=':', color='red', lw=2, label='Step 0 Baseline')

    for d in steps_data:
        x_jitter = np.random.normal(d['step_idx'], 0.05, size=len(d['sec_vals']))
        ax1.scatter(x_jitter, d['sec_vals'], color=TEXT_COLOR, s=20, alpha=0.2, edgecolors='none')
        ax1.scatter(d['step_idx'], d['sec_mean'], color='white', s=70, edgecolors=TEXT_COLOR, lw=2, zorder=5)

    ax1.set_title(secondary_metric_name, pad=40)
    ax1.set_ylabel(secondary_metric_name)
    ax1.legend(frameon=False, loc='lower right')

    # 3. FORCE label visibility by not using any "tight" or "auto" layout
    for ax in axes:
        ax.set_facecolor('#FFFFFF')
        ax.set_xlabel("Selection Step", labelpad=15)
        ax.set_xticks(x_steps)
        ax.set_xticklabels([f"Step {i}" for i in x_steps])

        # Force spines
        for side in ['top', 'right', 'bottom', 'left']:
            ax.spines[side].set_visible(side in ['left', 'bottom'])
            ax.spines[side].set_color(TEXT_COLOR)

        ax.tick_params(axis='both', which='both', bottom=True, left=True, labelbottom=True, labelleft=True)

    if save_plot:
        out_dir = pathlib.Path(output_dir) if output_dir else pathlib.Path(selection_results_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1. Identify condition from the path (male, female, male_mute_partner)
        path_str = str(selection_results_dir).lower()
        if 'male_mute_partner' in path_str:
            condition = 'male_mute_partner'
        elif 'female' in path_str:
            condition = 'female'
        elif 'male' in path_str:
            condition = 'male'
        else:
            condition = 'unknown'

        # 2. Construct the specific filename
        base_name = "multinomial_usv_category_model_selection_trajectory"
        fname = f"{base_name}_{condition}_{metric_primary}.svg"

        # 3. Save with high DPI and explicit white background
        save_path = out_dir / fname
        fig.savefig(save_path, facecolor='#FFFFFF', bbox_inches=None)
        print(f"Trajectory plot saved to: {save_path.name}")

    plt.show()
    plt.rcParams.update(_saved_rcp)


def plot_multinomial_multivariate_filters(
        selection_results_dir: str,
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
    selection_results_dir : str
        Path to the directory containing the stepwise results (e.g., 'step_0.pkl').
        The function extracts data from the final accepted model state.
    history_window_sec : float, default=4.0
        The duration of behavioral history analyzed. Used to convert internal
        indices into a human-readable time axis.
    cmap : str, default='RdBu_r'
        Diverging colormap; Red indicates promotion of a category, Blue
        indicates suppression.
    save_plot : bool, default=False
        If True, exports the grid as an SVG file for publication-quality editing.
    output_dir : str, optional
        Directory for saving the figure. Defaults to the source directory.

    Returns
    -------
    None
        Displays the high-resolution Matplotlib grid.
    """

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

    selection_steps, _, _ = load_selection_results(selection_results_dir)

    if not selection_steps:
        print(f"No step data found in {selection_results_dir}")
        return

    # --- BULLETPROOF DATA EXTRACTION ---
    # Find the last step that actually selected a feature (ignores the final rejection step)
    valid_data = None
    for data in reversed(selection_steps):
        if data.get('selected_feature') is not None:
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

    mean_weights = np.mean(weights, axis=0)

    n_feats = len(features)
    ncols = 3
    nrows = math.ceil(n_feats / ncols)

    # --- MANUAL LAYOUT (Absolute Figure Coordinates) ---
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
        max_amp = np.nanmax(np.abs(feat_slice)) or 1.0

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
        path_str = str(selection_results_dir).lower()
        condition = 'male_mute_partner' if 'male_mute_partner' in path_str else \
            ('female' if 'female' in path_str else 'male')
        out_dir = pathlib.Path(output_dir) if output_dir else pathlib.Path(selection_results_dir)
        fname = f"model_selection_multinomial_usv_category_{condition}_filters_final.svg"
        fig.savefig(out_dir / fname, facecolor='#FFFFFF', bbox_inches=None)

    plt.show()
    plt.rcParams.update(_saved_rcp)


def plot_multinomial_selection_diagnosis(
        selection_results_dir: str,
        cmap_base: str = 'mako',
        cmap_diff: str = 'RdBu_r',
        save_plot: bool = False,
        output_dir: str = None
) -> None:
    """
    Evaluates categorical Information Gain and fold-level reliability between univariate and multivariate models.

    Purpose and Diagnostic Value:
    -----------------------------
    This function provides a comprehensive post-hoc audit of the model selection process. It specifically
    addresses whether the addition of multivariate behavioral features actually improves the classification
    of difficult or rare vocal categories.

    The diagnostic is split into two primary visual tiers:
    1. The Top Tier (Heatmaps): Compares the row-normalized recall (sensitivity) of the best univariate
       anchor (Step 0) against the final multivariate ensemble. The middle plot's colorbar is scaled
       to its own local maximum to highlight classification nuances. The third plot (Information Gain)
       isolates the delta (Final - Univariate).
    2. The Bottom Tier (Slope Charts): Visualizes the "Correct vs. Incorrect" proportions across all
       cross-validation folds. Each fold is represented as a dot pair connected by a line, revealing
       the stability of the model's performance and the variance across different experimental sessions.

    Parameters
    ----------
    selection_results_dir : str
        Directory containing the stepwise selection '.pkl' files. The function automatically
        identifies Step 0 and the final successful step for comparison.
    cmap_base : str, default='mako'
        Colormap for the Baseline and Multivariate matrices.
    cmap_diff : str, default='RdBu_r'
        Diverging colormap for the Information Gain matrix.
    save_plot : bool, default=False
        If True, saves the diagnostic figure as an SVG file.
    output_dir : str, optional
        Target directory for saved figures. Defaults to selection_results_dir.

    Returns
    -------
    None
        Displays the multi-panel diagnostic figure.
    """

    # 1. Aesthetics and local styling overrides — restored at function exit.
    TEXT_COLOR = '#000000'
    BG_COLOR = '#FFFFFF'

    _rcp_override = {
        'text.color': TEXT_COLOR,
        'axes.labelcolor': TEXT_COLOR,
        'xtick.color': TEXT_COLOR,
        'ytick.color': TEXT_COLOR,
        'axes.edgecolor': TEXT_COLOR,
        'figure.facecolor': BG_COLOR,
        'axes.facecolor': BG_COLOR,
        'font.family': 'sans-serif'
    }
    _saved_rcp = {k: plt.rcParams[k] for k in _rcp_override}
    plt.rcParams.update(_rcp_override)

    # 2. Step Discovery and Selection
    selection_steps, _, _ = load_selection_results(selection_results_dir)

    if len(selection_steps) < 2:
        print("Diagnosis requires at least two successful steps (Univariate vs Multivariate).")
        return

    # Load Step 0 (Univariate anchor)
    step0_data = selection_steps[0]

    # Identify the final successful multivariate step
    final_idx = len(selection_steps) - 1
    final_data = selection_steps[final_idx]

    if final_data.get('selected_feature') is None:
        final_idx -= 1
        final_data = selection_steps[final_idx]

    # 3. Data Extraction Helper
    def get_tier_data(step_dict):
        winner = step_dict['selected_feature']
        cand_summary = step_dict['candidates_summary'][winner]
        folds = cand_summary['folds']
        classes = cand_summary['classes']

        # Initialize storage for performance slopes
        fold_correct_prop = []
        fold_incorrect_prop = []
        cat_fold_correct_prop = {c: [] for c in classes}
        cat_fold_incorrect_prop = {c: [] for c in classes}

        # Calculate proportions per fold
        for y_t, y_p in zip(folds['y_true'], folds['y_pred']):

            # Global fold proportions
            total_fold_samples = len(y_t)
            correct_count = np.sum(y_t == y_p)
            fold_correct_prop.append(correct_count / total_fold_samples)
            fold_incorrect_prop.append((total_fold_samples - correct_count) / total_fold_samples)

            # Per category fold proportions
            for c in classes:
                cat_mask = (y_t == c)
                n_cat_samples = np.sum(cat_mask)

                if n_cat_samples > 0:
                    correct_cat = np.sum((y_t == c) & (y_p == c))
                    cat_fold_correct_prop[c].append(correct_cat / n_cat_samples)
                    cat_fold_incorrect_prop[c].append((n_cat_samples - correct_cat) / n_cat_samples)
                else:
                    cat_fold_correct_prop[c].append(np.nan)
                    cat_fold_incorrect_prop[c].append(np.nan)

        # Calculate Global Confusion Matrix (Row-Normalized for Recall)
        y_true_all = np.concatenate(folds['y_true'])
        y_pred_all = np.concatenate(folds['y_pred'])
        cm = confusion_matrix(y_true_all, y_pred_all, normalize='true')

        return cm, classes, fold_correct_prop, fold_incorrect_prop, cat_fold_correct_prop, cat_fold_incorrect_prop

    # Process Univariate and Multivariate datasets
    cm_uni, classes, _, _, _, _ = get_tier_data(step0_data)
    cm_multi, _, f_corr_p, f_inc_p, c_f_corr_p, c_f_inc_p = get_tier_data(final_data)
    cm_gain = cm_multi - cm_uni

    # 4. Figure Construction
    n_cats = len(classes)
    fig = plt.figure(figsize=(22, 12), dpi=300)
    gs = gridspec.GridSpec(2, n_cats + 1, height_ratios=[1, 0.8], hspace=0.3)

    # 5. Top Tier: Heatmaps (Baseline, Final, Gain)
    axes_hm = [
        fig.add_subplot(gs[0, 0:2]),
        fig.add_subplot(gs[0, 2:4]),
        fig.add_subplot(gs[0, 4:])
    ]

    titles_hm = [
        f"Univariate Baseline\n({step0_data['selected_feature']})",
        f"Final Multivariate Model\n({len(final_data['current_features']) + 1} Features)",
        "Information Gain\n(Multi - Uni)"
    ]

    data_hm = [cm_uni, cm_multi, cm_gain]

    for i, (ax, data, title) in enumerate(zip(axes_hm, data_hm, titles_hm)):

        if i == 2:
            v_lim = np.max(np.abs(data))
            vmin, vmax, cmap = -v_lim, v_lim, cmap_diff
        else:
            vmin, vmax, cmap = 0, np.max(data), cmap_base

        sns.heatmap(
            data, ax=ax, annot=True, fmt=".2f", cmap=cmap, vmin=vmin, vmax=vmax,
            xticklabels=classes, yticklabels=classes if i == 0 else [],
            cbar_kws={'shrink': 0.8}, annot_kws={"size": 9}
        )
        ax.set_title(title, fontsize=12, fontweight='bold', pad=15)

    # 6. Bottom Tier: Performance Slope Charts (Proportions)
    def plot_performance_slopes(ax, corr_list, inc_list, title):

        labels = ['Match', 'Mismatch']
        # Use nanmean to handle folds where a specific category might be absent
        means = [np.nanmean(corr_list), np.nanmean(inc_list)]

        # Plot background bars for average proportions
        ax.bar(labels, means, color=['#2ecc71', '#e74c3c'], alpha=0.2)

        # Plot individual fold dots and connection lines
        for c, inc in zip(corr_list, inc_list):
            if not np.isnan(c):
                ax.plot(
                    labels, [c, inc],
                    color='black', alpha=0.35, linewidth=0.7,
                    marker='o', markersize=3.5
                )

        ax.set_title(title, fontsize=10, fontweight='bold', pad=8)
        ax.set_ylim(0, 1.05)  # Fixed scale for proportions
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=9)

    # Global Performance Slope
    ax_global = fig.add_subplot(gs[1, 0])
    plot_performance_slopes(ax_global, f_corr_p, f_inc_p, "Overall (All Folds)")
    ax_global.set_ylabel("Proportion of Samples", fontweight='bold', labelpad=10)

    # Per-Category Performance Slopes
    for i, cat in enumerate(classes):

        ax_cat = fig.add_subplot(gs[1, i + 1])
        plot_performance_slopes(ax_cat, c_f_corr_p[cat], c_f_inc_p[cat], f"Category: {cat}")

        if i > 0:
            ax_cat.set_yticklabels([])

    # 7. Final Polish and Save
    plt.suptitle(
        f"Multinomial Model Selection Audit: {pathlib.Path(selection_results_dir).stem}",
        fontsize=16, fontweight='bold', y=0.98
    )

    if save_plot:
        out_path = pathlib.Path(output_dir or selection_results_dir)
        fname = "multinomial_selection_diagnostic_audit_proportions.svg"
        fig.savefig(out_path / fname, facecolor=BG_COLOR, bbox_inches='tight')

    plt.show()
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
            animal colors, etc.
        """

        if not os.path.exists(results_pkl_path):
            raise FileNotFoundError(f"Results file not found: {results_pkl_path}")

        self.results_pkl_path = results_pkl_path

        with open(results_pkl_path, 'rb') as f:
            self.data = pickle.load(f)

        self.metadata = self.data['metadata']
        self.features = self.metadata['features_list']
        self.n_bins = self.metadata['n_time_bins']
        self.save_dir = self.metadata.get('save_dir', './plots')

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
        cv_folds = self.data['cross_validation']
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
        threshold_skill = np.percentile(null_dist_skill, 99.5)

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

        # ==========================================
        # PANEL A: EUCLIDEAN ERROR (TOP ROW)
        # ==========================================
        # --- EXACTLY AS IN SUBPLOT 1 ---
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

        # ==========================================
        # PANEL B: SKILL SCORE (BOTTOM ROW)
        # ==========================================
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

        # ==========================================
        # SHARED DECORATION & WRAP-UP
        # ==========================================
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
                                cmap: str = 'inferno',
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
        cmap : str, default 'inferno'
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
        # This is where I messed up before by ignoring the 'means'/'stds' keys
        actual_means = np.array([imp_data['means'][f] for f in feats])
        actual_stds = np.array([imp_data['stds'][f] for f in feats])

        # --- Dynamic SNR Calculation & Thresholding ---
        snrs = np.where(actual_stds > 1e-8, actual_means / actual_stds, 0.0)
        significant_mask = snrs > snr_threshold

        # --- Color Mapping Logic ---
        norm = plt.Normalize(vmin=actual_means.min(), vmax=actual_means.max())
        mapper = plt.get_cmap(cmap)
        lowest_color = mapper(0.0)

        # Significant features get the map; non-significant get the lowest 'glow'
        bar_colors = [mapper(norm(val)) if sig else lowest_color
                      for val, sig in zip(actual_means, significant_mask)]

        # --- Visualization ---
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

        # --- Save Logic (with sex-specific naming) ---
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
                                    patch_size: float = 2.5,
                                    min_samples: int = 50,
                                    bg_pt_color: str = '#E0E0E0',
                                    peak_pt_color: str = 'cyan',
                                    square_edge_color: str = '#000000',
                                    panel_fontsize: int = 9,
                                    figsize_unit: float = 3.0,
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
            The total number of subplot panels to generate.
        patch_size : float, default 2.5
            The side length of the square sampling window in UMAP units.
        min_samples : int, default 50
            The minimum number of data points required within a patch.
        bg_pt_color : str, default '#E0E0E0'
            Hex code for the background global UMAP coordinates.
        peak_pt_color : str, default 'cyan'
            Color of the crosshair ('+') marking the peak density of predictions.
        square_edge_color : str, default '#000000'
            Color of the border representing the true spatial bin.
        panel_fontsize : int, default 9
            Font size for the Bias score in the subplot titles.
        figsize_unit : float, default 3.0
            Inches per subplot.
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

        # --- 1. Data Preparation (Aggregating Across CNN Folds) ---
        cv_folds = self.data['cross_validation']
        # Explicitly wrapping in np.array to handle potential JAX DeviceArrays
        Y_true = np.vstack([np.array(f['Y_true']) for f in cv_folds])
        Y_pred = np.vstack([np.array(f['Y_pred_actual']) for f in cv_folds])

        # --- 2. Custom White-Base Inferno ---
        base_cmap = plt.cm.get_cmap('inferno')
        cmap_colors = base_cmap(np.linspace(0, 1, 256))
        white = np.array([1, 1, 1, 1])
        cmap_colors[:25, :] = np.linspace(white, cmap_colors[25, :], 25)
        white_inferno = ListedColormap(cmap_colors)

        # --- 3. K-Means Guided Search (Ensures Island Capture) ---
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
                dist_to_center = np.sqrt(np.sum((Y_true - [cx, cy]) ** 2, axis=1))
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

        # --- 4. Tiled Visualization ---
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
            kde = gaussian_kde(p_subset.T)

            # Alignment Grid
            xi_grid, yi_grid = np.mgrid[global_x_min:global_x_max:100j, global_y_min:global_y_max:100j]
            zi_grid = kde(np.vstack([xi_grid.flatten(), yi_grid.flatten()])).reshape(xi_grid.shape)

            peak_idx = np.unravel_index(np.argmax(zi_grid), zi_grid.shape)
            peak_coord = (xi_grid[peak_idx], yi_grid[peak_idx])
            dist = np.sqrt((peak_coord[0] - cx) ** 2 + (peak_coord[1] - cy) ** 2)

            if plot_type.lower() == 'contour':
                z_flat_sorted = np.sort(zi_grid.flatten())[::-1]
                z_cumsum = np.cumsum(z_flat_sorted) / np.sum(z_flat_sorted)
                levels = sorted([z_flat_sorted[np.searchsorted(z_cumsum, p)] for p in [0.50, 0.75, 0.90]])
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

        # --- 5. Updated Save Logic ---
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

    def plot_error_landscape(self,
                             gridsize: int = 30,
                             cmap: str = 'inferno',
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
        cmap : str, default 'inferno'
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

        # --- 1. Data Preparation (Aggregating Across CNN Folds) ---
        cv_folds = self.data['cross_validation']

        # Pull true coordinates and predictions from both Actual and Model-Free Null
        Y_true = np.vstack([np.array(f['Y_true']) for f in cv_folds])
        Y_pred_act = np.vstack([np.array(f['Y_pred_actual']) for f in cv_folds])
        Y_pred_null = np.vstack([np.array(f['Y_pred_null_model_free']) for f in cv_folds])

        # Calculate Euclidean errors point-by-point
        errors_act = np.sqrt(np.sum((Y_true - Y_pred_act) ** 2, axis=1))
        errors_null = np.sqrt(np.sum((Y_true - Y_pred_null) ** 2, axis=1))

        # Delta E: Positive means the Model is better (lower error) than the Null
        error_diff = errors_null - errors_act

        # --- 2. Visualization ---
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

        # Formatting all axes
        for ax in axes:
            ax.set_facecolor('#FFFFFF')
            ax.grid(False)
            ax.set_aspect('equal')

            ax.set_xlabel('UMAP Dimension 1', fontsize=label_fontsize, color='#202020')
            ax.set_ylabel('UMAP Dimension 2', fontsize=label_fontsize, color='#202020')
            ax.tick_params(colors='#202020', which='both', labelsize=label_fontsize - 1)

            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)
            for spine in ['left', 'bottom']:
                ax.spines[spine].set_color('#000000')

        for cb in [cbar1, cbar2]:
            cb.ax.yaxis.set_tick_params(color='#202020', labelcolor='#202020', labelsize=label_fontsize - 1)
            cb.outline.set_edgecolor('#000000')

        plt.tight_layout()

        # --- 3. Save Logic ---
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
                                     source_data_path: str,
                                     region_key: str,
                                     polygon_vertices: list,
                                     polygon_centroid: list,
                                     category_name: Optional[str] = None,
                                     prediction_plot_type: str = 'contour',
                                     highlight_color: Optional[str] = None,
                                     null_color: str = '#D3D3D3',
                                     cmap: str = cmr.fusion_r,
                                     figsize: tuple = (18, 9),
                                     save_plot: bool = False,
                                     output_dir: Optional[str] = None,
                                     file_format: str = 'svg') -> None:
        """
        Visualizes regional manifold dynamics by mapping spatial polygon annotations
        to kinematic saliency on the UMAP manifold.

        This method identifies the specific behavioral motifs that causally drive
        the network's predictions into localized acoustic clusters. It leverages
        Contrastive Centroid-Gradient Saliency to isolate region-specific drivers
        from the global postural baseline.

        The method operates in retrieval mode only: it expects `region_key`
        to exist in the pre-computed saliency dictionary
        (`self.data['saliency_maps']`) that was populated during CNN training.
        If the key is absent, a NotImplementedError is raised — on-the-fly
        saliency recomputation is not supported in this plotting helper and
        must be done in the training module.

        Parameters
        ----------
        source_data_path : str
            Full path to the source .pkl file containing the raw temporal
            kinematics. Currently unused (reserved for future on-the-fly
            saliency recomputation); kept in the signature for API stability.
        region_key : str
            The internal identifier used to look up pre-computed saliency maps
            stored in self.data['saliency_maps']. Also used as the display title
            if category_name is None.
        polygon_vertices : list
            A list of (x, y) coordinate pairs defining the boundary polygon
            for the target UMAP region. Previously stored in modeling_settings
            under spatial_annotations; now passed directly by the caller.
        polygon_centroid : list
            A two-element [x, y] list specifying the centroid of the target
            region. Previously stored in modeling_settings under
            spatial_annotations; now passed directly by the caller. Currently
            used only for diagnostic printing.
        category_name : str, optional
            The human-readable title for the plot (e.g., 'Category 3: Complex').
            If None, the 'region_key' is used for the display title.
        prediction_plot_type : str, default 'contour'
            Visualization style for the predicted UMAP coordinates.
            Options: ['contour', 'density', 'hexbin', 'scatter'].
        highlight_color : str, optional
            Color for the target polygon border, the peak density marker,
            and the model's error distribution. If None, it uses the default
            animal color assigned during visualizer initialization.
        null_color : str, default '#D3D3D3'
            Color for the model-free null distribution in the error inset.
        cmap : str, default 'cmr.fusion_r'
            The diverging colormap applied to the contrastive saliency heatmap.
            Values represent relative importance compared to the global mean.
        figsize : tuple, default (18, 9)
            Dimensions of the final figure in inches.
        save_plot : bool, default False
            If True, saves the figure to the specified output directory.
        output_dir : str, optional
            Path to the export directory. Defaults to the visualizer's save_dir.
        file_format : str, default 'svg'
            Format for the exported file (e.g., 'png', 'pdf', 'svg').

        Returns
        -------
        None
            Generates a two-panel matplotlib figure and optionally saves to disk.
        """

        # --- 0. COLOR MANAGEMENT ---
        # Reliably pull the color index 0 mapped in __init__
        if highlight_color is None:
            highlight_color = getattr(self, 'default_color', '#9AC0CD')

        # --- 1. DISPLAY TITLE ---
        display_title = category_name if category_name is not None else region_key

        # --- 2. DATA AGGREGATION ---
        imp_data = self.data['feature_importance']
        best_fold_idx = imp_data['best_fold_idx']
        fold_res = self.data['cross_validation'][best_fold_idx]

        Y_te = np.array(fold_res['Y_true'])
        Y_pred = np.array(fold_res['Y_pred_actual'])
        Y_pred_null = np.array(fold_res['Y_pred_null_model_free'])

        features_list = self.metadata['features_list']
        num_bins = self.metadata['n_time_bins']
        num_features = len(features_list)

        # --- 3. SPATIAL FILTERING ---
        poly_path = Path(polygon_vertices)
        r_mask = poly_path.contains_points(Y_te)
        r_idx = np.where(r_mask)[0]

        if len(r_idx) < 3:
            print(f"Warning: Region '{region_key}' contains only {len(r_idx)} samples. Skipping.")
            return

        err_actual = np.linalg.norm(Y_te[r_idx] - Y_pred[r_idx], axis=1)
        err_null = np.linalg.norm(Y_te[r_idx] - Y_pred_null[r_idx], axis=1)

        # --- 4. SALIENCY EXTRACTION ---
        if 'saliency_maps' in self.data and region_key in self.data['saliency_maps']:
            print(f"   > Extracting pre-computed saliency for {region_key}...")
            raw_saliency = self.data['saliency_maps'][region_key]['contrastive_saliency']
            contrastive_map = np.mean(raw_saliency, axis=0)
        else:
            # On-the-fly saliency recomputation is not implemented in this
            # visualizer. It would require re-loading raw temporal blocks
            # and rerunning a centroid-gradient attribution pass — both of
            # which belong in the training module, not a plotting helper.
            raise NotImplementedError(
                f"No pre-computed saliency map stored for region '{region_key}'. "
                f"Re-compute saliency during training and rerun, or pass a "
                f"region_key already present in self.data['saliency_maps'] "
                f"(available: {list(self.data.get('saliency_maps', {}).keys())})."
            )

        # --- 5. VISUALIZATION PIPELINE ---
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

        base_cmap = plt.cm.get_cmap('inferno')
        cmap_colors = base_cmap(np.linspace(0, 1, 256))
        cmap_colors[:25, :] = np.linspace(np.array([1, 1, 1, 1]), cmap_colors[25, :], 25)
        white_inferno = ListedColormap(cmap_colors)

        # PANEL 1: Manifold Context
        ax1.set_facecolor('#FFFFFF')
        ax1.set_title(f"UMAP Context: {display_title}", fontsize=14, color=text_color, pad=15)

        # Plot the dots inside the polygon with the highlight color
        ax1.scatter(Y_te[~r_mask, 0], Y_te[~r_mask, 1],
                    c='#B0B0B0', s=5, alpha=0.3, edgecolors='none', zorder=1)

        # Plot the SELECTED points (larger, highlighted color, black edges, top layer)
        ax1.scatter(Y_te[r_mask, 0], Y_te[r_mask, 1],
                    c=highlight_color, s=25, alpha=0.3, edgecolors='#000000', linewidths=0.5, zorder=2)

        y_p_region = Y_pred[r_idx]

        # --- FULLY RESTORED PREDICTION PLOT TYPE LOGIC ---
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

        # Dynamic Padding: Shift data up and right to give the inset an empty bottom-left corner
        x_min, x_max = Y_te[:, 0].min(), Y_te[:, 0].max()
        y_min, y_max = Y_te[:, 1].min(), Y_te[:, 1].max()
        ax1.set_xlim(x_min - (x_max - x_min) * 0.25, x_max + (x_max - x_min) * 0.05)
        ax1.set_ylim(y_min - (y_max - y_min) * 0.25, y_max + (y_max - y_min) * 0.05)

        ax1.set_xlabel('UMAP Dimension 1', fontsize=12, color=text_color)
        ax1.set_ylabel('UMAP Dimension 2', fontsize=12, color=text_color)
        ax1.tick_params(axis='both', colors=text_color, labelsize=10,
                        bottom=True, left=True, labelbottom=True, labelleft=True)

        # Error Comparison Inset (Positioned explicitly in the new padded corner)
        ax_ins = ax1.inset_axes([0.07, 0.07, 0.22, 0.16])
        ax_ins.set_facecolor('none')
        ax_ins.grid(False)
        ax_ins.hist(err_null, bins=15, density=True, color=null_color, alpha=0.6, label='Null', zorder=1)
        ax_ins.hist(err_actual, bins=15, density=True, color=highlight_color, alpha=0.6, label='CNN', zorder=2)
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
        v_lim = np.max(np.abs(contrastive_map))

        original_time = np.linspace(-4, 0, num_bins)
        smooth_time = np.linspace(-4, 0, 500)
        smooth_map = interp1d(original_time, contrastive_map, kind='cubic', axis=1)(smooth_time)

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


# ---------------------------------------------------------------------------
# Predictor diagnostics: plotting for the audits in
# `modeling.modeling_collinearity_audit`.
# ---------------------------------------------------------------------------

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
                            file_format: str = 'svg',
                            top_n_pairs: int = 25) -> dict:
    """
    Renders the collinearity audit produced by
    `modeling.modeling_collinearity_audit.audit_predictor_collinearity` as
    a single three-panel diagnostic figure.

    The figure summarises whether the kept predictor set is sufficiently
    decorrelated for stable forward stepwise selection. It is intended to
    accompany the modeling input pickle and to be visually inspected
    before committing to a particular feature shortlist.

    Panels
    ------
    Left
        Spearman ρ heatmap, with rows / columns reordered by hierarchical
        clustering on the absolute correlation distance so that blocks of
        mutually correlated features cluster visually. Cells whose
        absolute value exceeds the artifact's `concern_threshold` are
        annotated with their numeric value. Diverging blue/red colormap,
        symmetric around zero.
    Middle
        VIF bar chart, sorted descending, with horizontal reference lines
        at VIF = 5 (concern) and VIF = 10 (serious). Bars exceeding 10
        are highlighted in red.
    Right
        Top-N flagged pairs (where N = `top_n_pairs`) as horizontal bars
        of `|ρ|`, color-coded by tier (`exclude` vs. `concern`). Each bar
        labelled with the two feature names and the signed ρ value.

    The panel widths are scaled to keep a square heatmap on the left, a
    proportional VIF chart in the middle, and a long-and-narrow pair list
    on the right.

    Parameters
    ----------
    audit_pkl_path : str
        Path to the `_collinearity.pkl` artifact produced at extraction
        time. The artifact's `source_pickle` field is used in the figure
        title for provenance.
    save_dir : str, optional
        Output directory for the figure. Defaults to the directory
        containing the audit pickle.
    file_format : str, default 'svg'
        Matplotlib `savefig` format. SVG is the project default for
        publication-quality output.
    top_n_pairs : int, default 25
        Maximum number of flagged pairs to render in the right-hand
        panel. The artifact's `flagged_pairs` list is already sorted by
        descending |ρ|.

    Returns
    -------
    dict
        `{'figure_path': str, 'n_features': int, 'n_flagged': int}` —
        useful for callers that want to reference the artifact
        downstream.
    """

    with open(audit_pkl_path, 'rb') as fh:
        payload = pickle.load(fh)

    feature_names = payload['features']
    rho = np.asarray(payload['spearman_rho'])
    vif = np.asarray(payload['vif'])
    flagged = payload['flagged_pairs']
    concern_thr = payload['concern_threshold']
    exclude_thr = payload['exclude_threshold']
    n_events = payload['n_events']
    cond_num = payload['condition_number']
    source = payload['source_pickle']

    n_features = len(feature_names)
    if n_features == 0:
        print(f"[plot] collinearity audit at {audit_pkl_path} has no features — skipping.")
        return {'figure_path': '', 'n_features': 0, 'n_flagged': 0}

    # Hierarchical-clustering reorder via SciPy linkage on |ρ|-distance.
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform
    dist = 1.0 - np.abs(rho)
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0
    try:
        order = leaves_list(linkage(squareform(dist, checks=False), method='average'))
    except ValueError:
        order = np.arange(n_features)
    rho_ord = rho[np.ix_(order, order)]
    names_ord = [feature_names[i] for i in order]

    fig = plt.figure(figsize=(20, max(8, 0.25 * n_features + 4)))
    gs = gridspec.GridSpec(1, 3, width_ratios=[2.4, 1.2, 1.4], wspace=0.35)

    # Panel 1: Spearman ρ heatmap
    ax1 = fig.add_subplot(gs[0, 0])
    im = ax1.imshow(rho_ord, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax1.set_xticks(np.arange(n_features))
    ax1.set_yticks(np.arange(n_features))
    ax1.set_xticklabels(names_ord, rotation=90, fontsize=7)
    ax1.set_yticklabels(names_ord, fontsize=7)
    ax1.set_title(f"Spearman ρ (clustered)\n{n_events} events × {n_features} features", fontsize=11)
    cb = fig.colorbar(im, ax=ax1, fraction=0.04, pad=0.02)
    cb.set_label('ρ', rotation=0, labelpad=8)
    # Annotate cells above concern threshold
    for i in range(n_features):
        for j in range(n_features):
            if i == j:
                continue
            r = rho_ord[i, j]
            if abs(r) >= concern_thr:
                ax1.text(j, i, f"{r:.2f}", ha='center', va='center',
                         fontsize=5,
                         color='white' if abs(r) > 0.6 else 'black')

    # Panel 2: VIF bar chart
    ax2 = fig.add_subplot(gs[0, 1])
    sort_idx = np.argsort(-np.where(np.isfinite(vif), vif, -np.inf))
    vif_sorted = vif[sort_idx]
    names_sorted_v = [feature_names[i] for i in sort_idx]
    colors = ['#cc3333' if (np.isfinite(v) and v > 10)
              else '#dd9933' if (np.isfinite(v) and v > 5)
              else '#3377bb' for v in vif_sorted]
    # Cap displayed VIF for readability; annotate inf separately.
    vif_display = np.where(np.isfinite(vif_sorted),
                           np.minimum(vif_sorted, 50.0),
                           50.0)
    y_pos = np.arange(n_features)[::-1]
    ax2.barh(y_pos, vif_display, color=colors, edgecolor='none')
    for k, v in enumerate(vif_sorted):
        if not np.isfinite(v):
            ax2.text(50.0, y_pos[k], '  inf', va='center', fontsize=6, color='#cc3333')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(names_sorted_v, fontsize=7)
    ax2.axvline(5, color='#dd9933', linestyle=':', linewidth=1, label='VIF = 5')
    ax2.axvline(10, color='#cc3333', linestyle=':', linewidth=1, label='VIF = 10')
    ax2.set_xlabel('VIF (capped at 50)', fontsize=9)
    ax2.set_title(f"Variance Inflation\ncond(X) = {cond_num:.1f}", fontsize=11)
    ax2.legend(fontsize=7, loc='lower right')

    # Panel 3: Flagged pairs
    ax3 = fig.add_subplot(gs[0, 2])
    pairs = flagged[:top_n_pairs]
    if pairs:
        labels = [f"{f1}  ↔  {f2}" for f1, f2, _, _ in pairs]
        rhos = [r for *_, r, _ in pairs]
        tiers = [t for *_, t in pairs]
        bar_colors = ['#cc3333' if t == 'exclude' else '#dd9933' for t in tiers]
        y_pos = np.arange(len(pairs))[::-1]
        ax3.barh(y_pos, [abs(r) for r in rhos], color=bar_colors, edgecolor='none')
        for k, (label, r) in enumerate(zip(labels, rhos)):
            ax3.text(abs(r) + 0.01, y_pos[k], f"{r:+.2f}", va='center', fontsize=6)
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(labels, fontsize=6)
        ax3.set_xlim(0, 1.05)
        ax3.axvline(concern_thr, color='#dd9933', linestyle=':', linewidth=1)
        ax3.axvline(exclude_thr, color='#cc3333', linestyle=':', linewidth=1)
        ax3.set_xlabel('|ρ|', fontsize=9)
        ax3.set_title(
            f"Flagged pairs (top {len(pairs)} of {len(flagged)})\n"
            f"red: |ρ| > {exclude_thr}, orange: > {concern_thr}",
            fontsize=11,
        )
    else:
        ax3.text(0.5, 0.5, "No pairs above\nconcern threshold",
                 ha='center', va='center', fontsize=12, transform=ax3.transAxes)
        ax3.set_xticks([]); ax3.set_yticks([])
        ax3.set_title("Flagged pairs", fontsize=11)

    fig.suptitle(f"Collinearity audit  —  source: {source}", fontsize=13, y=1.02)

    if save_dir is None:
        save_dir = os.path.dirname(audit_pkl_path)
    base = os.path.splitext(os.path.basename(audit_pkl_path))[0]
    out_path = _save_audit_figure(fig, save_dir, base, file_format=file_format)
    plt.close(fig)
    print(f"[plot] collinearity figure written: {out_path}")
    return {'figure_path': out_path, 'n_features': n_features, 'n_flagged': len(flagged)}


def plot_timescale_audit(timescale_pkl_path: str,
                         save_dir: str = None,
                         save_plot_bool: bool = True,
                         plot_format: str = 'svg') -> dict:
    """
    Renders the timescale audit produced by
    `modeling.modeling_collinearity_audit.audit_predictor_timescales` as
    a two-panel diagnostic figure (ACF + signal correlation).

    Panels
    ------
    Left (ACF)
        Per-feature autocorrelation curves: the median ACF across
        sessions (solid) with the inter-quartile range as a translucent
        envelope. The configured `filter_history` is drawn as a heavy
        vertical reference line. Captures how long each feature holds
        memory of itself.

    Right (Signal correlation)
        Per-feature Spearman ρ vs. lag between every predictor and the
        binary USV indicator, evaluated symmetrically over
        `[-max_lag, +max_lag]`. Negative lags ⇒ USV precedes feature;
        positive lags ⇒ feature precedes USV. The within-session
        circular-shift 95th-percentile null envelope is shaded for
        reference.

    Parameters
    ----------
    timescale_pkl_path : str
        Path to the `_timescales.pkl` artifact.
    save_dir : str, optional
        Output directory for the figure. Defaults to the directory
        containing the timescale pickle. Only consulted when
        `save_plot_bool` is True.
    save_plot_bool : bool, default True
        When True (default), the figure is written to disk via
        `_save_audit_figure` and closed. When False, the figure is
        neither saved nor closed — the caller can display it inline
        (notebook) or further customise it. `figure_path` in the
        returned dict is `''` in that case.
    plot_format : str, default 'svg'
        Matplotlib `savefig` format. Only consulted when
        `save_plot_bool` is True.

    Returns
    -------
    dict
        `{'figure_path': str, 'n_features': int, 'configured_filter_history': float}`.
        `figure_path` is `''` when `save_plot_bool` is False.
    """

    with open(timescale_pkl_path, 'rb') as fh:
        payload = pickle.load(fh)

    feature_names = payload['features']
    acf_lags_seconds = np.asarray(payload['acf_lags_seconds'])
    acf_med = np.asarray(payload['acf_median'])
    acf_p25 = np.asarray(payload['acf_p25'])
    acf_p75 = np.asarray(payload['acf_p75'])
    signal_lags_seconds = np.asarray(payload['signal_lags_seconds'])
    rho_signal = np.asarray(payload['rho_signal'])
    rho_signal_null = np.asarray(payload['rho_signal_null_p95'])
    cfg_hist = float(payload['configured_filter_history'])
    source = payload['source_pickle']

    n_features = len(feature_names)
    if n_features == 0:
        print(f"[plot] timescale audit at {timescale_pkl_path} has no features — skipping.")
        return {'figure_path': '', 'n_features': 0, 'configured_filter_history': cfg_hist}

    fig = plt.figure(figsize=(20, 8))
    gs = gridspec.GridSpec(1, 2, wspace=0.22)

    # Stable per-feature colour mapping shared across panels.
    cmap = plt.get_cmap('tab20', max(n_features, 20))
    colors = [cmap(i % cmap.N) for i in range(n_features)]

    # Panel A: ACF (positive lags only)
    axA = fig.add_subplot(gs[0, 0])
    for i, fname in enumerate(feature_names):
        axA.fill_between(acf_lags_seconds, acf_p25[i], acf_p75[i],
                         color=colors[i], alpha=0.12, linewidth=0)
        axA.plot(acf_lags_seconds, acf_med[i], color=colors[i],
                 linewidth=0.9, alpha=0.75, label=fname)
    # Median across features — the unifying envelope.
    if acf_med.size:
        acf_aggregate = np.nanmedian(acf_med, axis=0)
        axA.plot(acf_lags_seconds, acf_aggregate, color='black',
                 linewidth=2.4, label='median across features', zorder=10)
    axA.axhline(0, color='black', linewidth=0.5)
    axA.axhline(1.0 / np.e, color='gray', linestyle=':', linewidth=0.7, label='1/e')
    axA.axvline(cfg_hist, color='black', linestyle='--', linewidth=1.0,
                label=f'filter_history = {cfg_hist:.1f}s')
    axA.set_xlim(0, acf_lags_seconds[-1] if acf_lags_seconds.size else 1.0)
    axA.set_ylim(-0.3, 1.05)
    axA.set_xlabel('Lag (s)')
    axA.set_ylabel('ACF (median ± IQR)')
    axA.set_title('Predictor ACF', fontsize=11)
    axA.legend(fontsize=6, loc='upper right', ncol=2)

    # Panel B: Signal correlation (symmetric lags)
    axB = fig.add_subplot(gs[0, 1])
    for i, fname in enumerate(feature_names):
        axB.fill_between(signal_lags_seconds, -rho_signal_null[i], rho_signal_null[i],
                         color=colors[i], alpha=0.08, linewidth=0)
        axB.plot(signal_lags_seconds, rho_signal[i], color=colors[i],
                 linewidth=0.9, alpha=0.75, label=fname)
    # Median across features — the unifying envelope.
    if rho_signal.size:
        sig_aggregate = np.nanmedian(rho_signal, axis=0)
        axB.plot(signal_lags_seconds, sig_aggregate, color='black',
                 linewidth=2.4, label='median across features', zorder=10)
    axB.axhline(0, color='black', linewidth=0.5)
    axB.axvline(0, color='black', linewidth=0.6, linestyle='-')
    if signal_lags_seconds.size:
        axB.set_xlim(signal_lags_seconds[0], signal_lags_seconds[-1])
    axB.set_xlabel('Lag (s)   (negative: USV leads feature   |   positive: feature leads USV)')
    axB.set_ylabel('Spearman ρ vs. binary USV indicator   (shaded: |null| 95th pct)')
    axB.set_title('Signal correlation', fontsize=11)
    axB.legend(fontsize=6, loc='upper right', ncol=2)

    fig.suptitle(f"Timescale audit  —  source: {source}", fontsize=13, y=0.995)

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
        'n_features': n_features,
        'configured_filter_history': cfg_hist,
    }


def plot_timescale_audit_per_feature(timescale_pkl_path: str,
                                     save_dir: str = None,
                                     save_plot_bool: bool = True,
                                     plot_format: str = 'svg') -> dict:
    """
    Renders the timescale audit as a small-multiples grid: one row per
    feature, two columns (ACF on the left, signal correlation on the
    right). Complements `plot_timescale_audit` (which overlays every
    feature on a single pair of axes) by giving each feature its own
    panel so the per-feature shape, peak location, and null margin are
    legible without colour discrimination across 20+ overlapping curves.

    Layout
    ------
    - One row per feature; left column shows ACF (median ± IQR across
      sessions, positive lags only); right column shows the symmetric
      signal correlation curve (ρ vs. binary USV indicator) with the
      per-lag null 95th-percentile envelope shaded in grey.
    - Y-axis scale of the signal-correlation column is shared across
      all features so peak magnitudes are directly comparable. The ACF
      column uses the standard `[-0.3, 1.05]` range used by
      `plot_timescale_audit`.

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

    Returns
    -------
    dict
        `{'figure_path': str, 'n_features': int, 'configured_filter_history': float}`.
        `figure_path` is `''` when `save_plot_bool` is False.
    """

    with open(timescale_pkl_path, 'rb') as fh:
        payload = pickle.load(fh)

    feature_names = payload['features']
    acf_lags_seconds = np.asarray(payload['acf_lags_seconds'])
    acf_med = np.asarray(payload['acf_median'])
    acf_p25 = np.asarray(payload['acf_p25'])
    acf_p75 = np.asarray(payload['acf_p75'])
    signal_lags_seconds = np.asarray(payload['signal_lags_seconds'])
    rho_signal = np.asarray(payload['rho_signal'])
    rho_signal_null = np.asarray(payload['rho_signal_null_p95'])
    cfg_hist = float(payload['configured_filter_history'])
    source = payload['source_pickle']

    n_features = len(feature_names)
    if n_features == 0:
        print(f"[plot] timescale audit at {timescale_pkl_path} has no features — skipping.")
        return {'figure_path': '', 'n_features': 0, 'configured_filter_history': cfg_hist}

    cmap = plt.get_cmap('tab20', max(n_features, 20))
    colors = [cmap(i % cmap.N) for i in range(n_features)]

    # Shared y-limits for the signal-correlation column so per-feature
    # peak magnitudes are directly comparable across panels.
    sig_max_abs = float(np.nanmax(np.abs(rho_signal))) if rho_signal.size else 0.005
    sig_ymax = max(sig_max_abs * 1.10, 1e-3)

    fig, axes = plt.subplots(
        n_features, 2,
        figsize=(14, 1.6 * n_features),
        sharex=False, squeeze=False,
    )

    for i, fname in enumerate(feature_names):
        col = colors[i]

        # Column 1 — ACF
        axA = axes[i, 0]
        axA.fill_between(acf_lags_seconds, acf_p25[i], acf_p75[i],
                         color=col, alpha=0.20, linewidth=0)
        axA.plot(acf_lags_seconds, acf_med[i], color=col, linewidth=1.5)
        axA.axhline(0, color='black', linewidth=0.5)
        axA.axhline(1.0 / np.e, color='gray', linestyle=':', linewidth=0.6)
        axA.axvline(cfg_hist, color='black', linestyle='--', linewidth=0.8)
        axA.set_xlim(0, acf_lags_seconds[-1] if acf_lags_seconds.size else 1.0)
        axA.set_ylim(-0.3, 1.05)
        axA.set_ylabel(fname, fontsize=9, rotation=0, ha='right', va='center', labelpad=4)
        axA.tick_params(labelsize=7)
        if i == 0:
            axA.set_title('ACF (median ± IQR)', fontsize=11)
        if i == n_features - 1:
            axA.set_xlabel('Lag (s)')
        else:
            axA.set_xticklabels([])

        # Column 2 — Signal correlation
        axB = axes[i, 1]
        axB.fill_between(signal_lags_seconds, -rho_signal_null[i], rho_signal_null[i],
                         color='gray', alpha=0.25, linewidth=0)
        axB.plot(signal_lags_seconds, rho_signal[i], color=col, linewidth=1.5)
        axB.axhline(0, color='black', linewidth=0.5)
        axB.axvline(0, color='black', linewidth=0.6)
        if signal_lags_seconds.size:
            axB.set_xlim(signal_lags_seconds[0], signal_lags_seconds[-1])
        axB.set_ylim(-sig_ymax, sig_ymax)
        axB.tick_params(labelsize=7)
        if i == 0:
            axB.set_title('Signal correlation (ρ vs. binary USV — shaded: |null| 95th pct)',
                          fontsize=11)
        if i == n_features - 1:
            axB.set_xlabel('Lag (s)   (neg: USV leads feature   |   pos: feature leads USV)')
        else:
            axB.set_xticklabels([])

    fig.suptitle(f"Timescale audit — per-feature panels — source: {source}",
                 fontsize=12, y=0.999)
    fig.subplots_adjust(left=0.18, right=0.98,
                        top=1.0 - 0.6 / max(n_features, 1),
                        bottom=0.04, hspace=0.10, wspace=0.18)

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
