from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import polars as pls
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter1d
from scipy.stats import pearsonr, gaussian_kde, sem, t
import seaborn as sns

import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd


def extract_session_metadata(session_root: str) -> dict[str, Any]:
    """
    Description
    ----------
    This method extracts core experimental metadata from a session directory, including
    mouse track names, recording frame rate, and experimental codes.

    It searches for the metric H5 tracking file within the provided
    directory and extracts identity strings for the animals involved. It is
    specifically designed for social interaction sessions (male-female).
    ----------

    Parameters
    ----------
    session_root (str)
        The absolute path to the session directory containing the .h5 tracking files.
    ----------

    Returns
    ----------
    metadata (dict)
        Contains 'male_id', 'female_id', 'frame_rate', 'experiment_code', and 'tracking_file'.
    ----------
    """

    session_path = Path(session_root)
    tracking_file = next(session_path.glob('**/*_points3d_translated_rotated_metric.h5'), None)

    if tracking_file is None:
        msg = f"No tracking file found in {session_root}"
        raise FileNotFoundError(msg)

    with h5py.File(name=str(tracking_file), mode='r') as h5_file:
        track_names = [item.decode('utf-8') for item in list(h5_file['track_names'])]
        if len(track_names) < 2:
            msg = f"Session {session_root} does not contain two animal tracks."
            raise IndexError(msg)

        return {
            'male_id': track_names[0],
            'female_id': track_names[1],
            'frame_rate': float(h5_file['recording_frame_rate'][()]),
            'experiment_code': h5_file['experimental_code'][()].decode("utf-8"),
            'tracking_file': tracking_file
        }

def load_and_filter_usv_data(
    session_root: str,
    frame_rate: float,
    noise_col_id: str,
    noise_categories: list[int]
) -> pls.DataFrame:
    """
    Description
    ----------
    This method loads USV summary CSV data using Polars and appends calculated frame
    indices based on the provided recording frame rate.

    The function filters the entire dataset to remove noise based on the provided
    noise column and a list of noise categories. The remaining valid vocalizations
    (male, female, and unassigned) are retained and returned.
    ----------

    Parameters
    ----------
    session_root (str)
        The absolute path to the session directory.
    frame_rate (float)
        The sampling rate of the video recording used to synchronize USVs with behavioral frames.
    noise_col_id (str)
        The name of the column in the CSV that dictates the noise classification.
    noise_categories (list[int])
        A list of specific integer values in the noise column that identify a row as noise to be excluded.
    ----------

    Returns
    ----------
    usv_info (pls.DataFrame)
        Contains USV starts, durations, emitters, and a newly calculated 'frame_index',
        with all identified noise removed.
    ----------
    """

    session_path = Path(session_root)
    usv_file = next(session_path.glob('**/*_usv_summary.csv'), None)

    if usv_file is None:
        msg = f"USV summary file missing in {session_root}"
        raise FileNotFoundError(msg)

    usv_info = pls.read_csv(str(usv_file))

    # Remove noise across all categories provided in the list
    usv_info_clean = usv_info.filter(
        ~pls.col(noise_col_id).is_in(noise_categories)
    )

    return usv_info_clean.with_columns(
        (pls.col("start") * frame_rate).floor().cast(pls.UInt32).alias("frame_index")
    )

def extract_category_embedding_data(
    session_roots: list[str],
    noise_col_id: str,
    noise_categories: list[int],
    usv_category_col: str,
    usv_continuous_cols: tuple[str, str]
) -> pls.DataFrame:
    """
    Description
    ----------
    Extracts category labels and continuous embedding coordinates (e.g., UMAP)
    for all non-noise vocalizations across multiple sessions.

    This function first filters out any noise rows, ensures the target category
    and embedding columns exist in the CSV, and then categorizes each valid
    USV as 'male', 'female', or 'unassigned' based on the session's metadata.
    Rows with missing coordinate data are dropped to ensure clean downstream plotting.

    Parameters
    ----------
    session_roots : list[str]
        A list of absolute paths pointing to the session directories to be analyzed.
    noise_col_id : str
        The name of the column in the CSV that dictates the noise classification.
    noise_categories : list[int]
        A list of specific integer values in the noise column that identify a row as
        noise to be excluded.
    usv_category_col : str
        The name of the column containing the integer category/cluster ID
        (e.g., 'usv_supercategory').
    usv_continuous_cols : tuple[str, str]
        A tuple of two strings specifying the column names for the 2D embedding
        coordinates (e.g., ('umap_x', 'umap_y')).

    Returns
    ----------
    pls.DataFrame
        A concatenated Polars DataFrame containing the columns: 'sex', 'category',
        'dim1' (x-coordinate), and 'dim2' (y-coordinate). Returns an empty
        DataFrame if no valid data is found.
    """

    all_data = []

    for session_root in session_roots:
        try:
            metadata = extract_session_metadata(session_root)
            male_id = metadata['male_id']
            female_id = metadata['female_id']

            # Filter noise
            usv_info = load_and_filter_usv_data(
                session_root=session_root,
                frame_rate=metadata['frame_rate'],
                noise_col_id=noise_col_id,
                noise_categories=noise_categories
            )

            # Ensure the required columns actually exist in this session's CSV
            req_cols = [usv_category_col, usv_continuous_cols[0], usv_continuous_cols[1]]
            if not all(col in usv_info.columns for col in req_cols):
                continue

            # Map sex and select target columns
            usv_processed = usv_info.with_columns([
                pls.when(pls.col("emitter") == male_id).then(pls.lit("male"))
                .when(pls.col("emitter") == female_id).then(pls.lit("female"))
                .otherwise(pls.lit("unassigned"))
                .alias("sex")
            ]).select([
                "sex",
                pls.col(usv_category_col).alias("category"),
                pls.col(usv_continuous_cols[0]).alias("dim1"),
                pls.col(usv_continuous_cols[1]).alias("dim2")
            ]).drop_nulls() # Drop rows missing coordinate data

            all_data.append(usv_processed)

        except (FileNotFoundError, IndexError):
            continue

    return pls.concat(all_data) if all_data else pls.DataFrame()

def get_session_behavioral_features(session_root: str) -> pls.DataFrame:
    """
    Description
    ----------
    This method loads the behavioral features CSV (e.g., distances, angles) for a
    specific session.

    This function specifically targets the file containing calculated
    metrics like 'nose-nose' distance and 'allo_yaw' angles. It adds
    a row index to serve as a 'frame_index' for joining with USV data.
    ----------

    Parameters
    ----------
    session_root (str)
        The absolute path to the session directory.
    ----------

    Returns
    ----------
    behavioral_features (pls.DataFrame)
        Contains all frame-by-frame behavioral metrics.
    ----------
    """

    session_path = Path(session_root)
    features_file = next(session_path.glob('**/*_behavioral_features.csv'), None)

    if features_file is None:
        msg = f"Behavioral features file missing in {session_root}"
        raise FileNotFoundError(msg)

    return pls.read_csv(str(features_file)).with_row_index("frame_index")

def merge_usv_and_behavioral_features(
    usv_info: pls.DataFrame,
    behavioral_features: pls.DataFrame,
    nose_distance_col: str,
    mf_angle_col: str,
    fm_angle_col: str,
    usv_category_col: str
) -> pls.DataFrame:
    """
    Description
    ----------
    This method merges ultrasonic vocalization (USV) timing, assignment, and category data
    with frame-by-frame continuous behavioral features.

    It extracts the specified distance and angle metrics from the broad behavioral
    dataset, renames them for clarity, and performs an inner join with the USV
    dataset using the shared 'frame_index'. This creates a unified dataset where
    every individual vocalization is annotated with the physical distance,
    relative angles of the mice, and its acoustic category at the exact moment
    the call was emitted.
    ----------

    Parameters
    ----------
    usv_info (pls.DataFrame)
        A Polars DataFrame containing USV data, which must include 'frame_index',
        'emitter', 'duration', and the specified category columns.
    behavioral_features (pls.DataFrame)
        A Polars DataFrame containing continuous tracking features, which must
        include a 'frame_index' column alongside the metric columns.
    nose_distance_col (str)
        The exact column name in behavioral_features representing the distance
        between the animals' noses.
    mf_angle_col (str)
        The exact column name in behavioral_features representing the male-to-female
        allocentric yaw angle.
    fm_angle_col (str)
        The exact column name in behavioral_features representing the female-to-male
        allocentric yaw angle.
    usv_category_col (str)
        The name of the column containing the integer category/cluster ID.
    ----------

    Returns
    ----------
    usv_behavior (pls.DataFrame)
        A combined Polars DataFrame containing 'frame_index', 'emitter',
        'category', 'usv_duration', 'distance', 'mf_angle', and 'fm_angle'
        for every matched vocalization.
    ----------
    """

    behavioral_subset = behavioral_features.select(
        pls.col("frame_index"),
        pls.col(nose_distance_col).alias("distance"),
        pls.col(mf_angle_col).alias("mf_angle"),
        pls.col(fm_angle_col).alias("fm_angle")
    )

    usv_subset = usv_info.select(
        pls.col("frame_index"),
        pls.col("emitter"),
        pls.col(usv_category_col).alias("category"),
        pls.col("duration").alias("usv_duration")
    )

    return usv_subset.join(behavioral_subset, on="frame_index", how="inner")

def build_master_usv_dataframe(
    session_roots: list[str],
    noise_col_id: str,
    noise_categories: list[int],
    usv_category_col: str,
    distance_suffix: str,
    mf_angle_suffix: str,
    fm_angle_suffix: str
) -> tuple[pls.DataFrame, pls.DataFrame, int]:
    """
    Description
    ----------
    This method is the primary data extraction and aggregation entry point for the
    USV summary statistics pipeline. It replaces the multiple per-analysis session
    loops previously scattered across the analysis notebook with a single,
    comprehensive pass over all session directories.

    For each session it reads metadata from the H5 tracking file, loads and
    noise-filters the USV summary CSV, maps emitters to a 'sex' column, and
    attempts to join in frame-by-frame spatial behavioral features (nose-nose
    distance and relative angles). All data is returned as two tidy Polars
    DataFrames that can be filtered, grouped, and aggregated for any downstream
    analysis without requiring additional per-session loops.

    The returned 'usv_df' has one row per vocalization. Animal IDs are stripped
    of null bytes and leading/trailing whitespace to ensure consistent identity
    matching across sessions.

    The returned 'background_df' has one row per video frame for sessions that
    have a behavioral features file, and provides the spatial occupancy baseline
    required for occupancy-normalised polar KDE plots.
    ----------

    Parameters
    ----------
    session_roots (list[str])
        A list of absolute paths pointing to the session directories to be analyzed.
    noise_col_id (str)
        The name of the column in the CSV that dictates the noise classification.
    noise_categories (list[int])
        A list of specific integer values in the noise column that identify a row
        as noise to be excluded.
    usv_category_col (str)
        The name of the column containing the integer category/cluster ID.
    distance_suffix (str)
        The string suffix used to identify the nose-to-nose distance column in the
        behavioral features CSV (e.g., 'nose-nose').
    mf_angle_suffix (str)
        The string suffix used to identify the male-to-female angle column in the
        behavioral features CSV (e.g., 'allo_yaw-nose').
    fm_angle_suffix (str)
        The string suffix used to identify the female-to-male angle column in the
        behavioral features CSV (e.g., 'nose-allo_yaw').
    ----------

    Returns
    ----------
    usv_df (pls.DataFrame)
        A tidy Polars DataFrame with one row per non-noise vocalization across all
        sessions. Columns: 'session_id', 'date', 'hour', 'male_id', 'female_id',
        'experiment_code', 'emitter', 'sex', 'category', 'start', 'duration',
        'frame_index', 'distance', 'mf_angle', 'fm_angle'. The last three columns
        are null for sessions missing a behavioral features file.
    background_df (pls.DataFrame)
        A tidy Polars DataFrame with one row per video frame for each session that
        has a behavioral features file. Columns: 'session_id', 'distance',
        'mf_angle', 'fm_angle'. Used as the occupancy baseline in polar KDE plots.
    total_noise_filtered (int)
        The total number of rows removed across all sessions based on noise_categories.
    ----------
    """

    all_usv_rows: list[pls.DataFrame] = []
    all_bg_rows: list[pls.DataFrame] = []
    total_noise_filtered = 0

    for session_root in session_roots:
        session_path = Path(session_root)

        try:
            metadata = extract_session_metadata(session_root)
        except (FileNotFoundError, IndexError):
            continue

        raw_male_id = metadata['male_id']
        raw_female_id = metadata['female_id']
        male_id = str(raw_male_id).strip('\x00').strip()
        female_id = str(raw_female_id).strip('\x00').strip()
        frame_rate = metadata['frame_rate']
        experiment_code = metadata['experiment_code']

        usv_file = next(session_path.glob('**/*_usv_summary.csv'), None)
        if usv_file is None:
            continue

        raw_data = pls.read_csv(str(usv_file))
        total_noise_filtered += raw_data.filter(pls.col(noise_col_id).is_in(noise_categories)).height

        session_id = usv_file.stem.replace('_usv_summary', '')
        date_str = session_id.split('_')[0]
        hour_int = int(session_id.split('_')[1][0:2])

        try:
            usv_info = load_and_filter_usv_data(
                session_root=session_root,
                frame_rate=frame_rate,
                noise_col_id=noise_col_id,
                noise_categories=noise_categories
            )
        except FileNotFoundError:
            continue

        if usv_category_col not in usv_info.columns:
            continue

        usv_processed = usv_info.with_columns([
            pls.when(pls.col('emitter') == raw_male_id).then(pls.lit('male'))
            .when(pls.col('emitter') == raw_female_id).then(pls.lit('female'))
            .otherwise(pls.lit('unassigned'))
            .alias('sex'),
            pls.lit(session_id).alias('session_id'),
            pls.lit(date_str).alias('date'),
            pls.lit(hour_int).cast(pls.Int32).alias('hour'),
            pls.lit(male_id).alias('male_id'),
            pls.lit(female_id).alias('female_id'),
            pls.lit(experiment_code).alias('experiment_code'),
            pls.col(usv_category_col).alias('category')
        ]).select([
            'session_id', 'date', 'hour', 'male_id', 'female_id', 'experiment_code',
            'emitter', 'sex', 'category', 'start', 'duration', 'frame_index'
        ])

        has_behavioral = False
        try:
            behavioral_features = get_session_behavioral_features(session_root)
            dist_col = next((c for c in behavioral_features.columns if c.endswith(distance_suffix)), None)
            mf_col = next((c for c in behavioral_features.columns if c.endswith(mf_angle_suffix)), None)
            fm_col = next((c for c in behavioral_features.columns if c.endswith(fm_angle_suffix)), None)

            if dist_col and mf_col and fm_col:
                has_behavioral = True

                all_bg_rows.append(behavioral_features.select([
                    pls.lit(session_id).alias('session_id'),
                    pls.col(dist_col).alias('distance'),
                    pls.col(mf_col).alias('mf_angle'),
                    pls.col(fm_col).alias('fm_angle')
                ]))

                usv_processed = usv_processed.join(
                    behavioral_features.select([
                        'frame_index',
                        pls.col(dist_col).alias('distance'),
                        pls.col(mf_col).alias('mf_angle'),
                        pls.col(fm_col).alias('fm_angle')
                    ]),
                    on='frame_index',
                    how='left'
                )

        except (FileNotFoundError, StopIteration):
            pass

        if not has_behavioral:
            usv_processed = usv_processed.with_columns([
                pls.lit(None).cast(pls.Float64).alias('distance'),
                pls.lit(None).cast(pls.Float64).alias('mf_angle'),
                pls.lit(None).cast(pls.Float64).alias('fm_angle')
            ])

        all_usv_rows.append(usv_processed)

    usv_df = pls.concat(all_usv_rows) if all_usv_rows else pls.DataFrame()
    background_df = pls.concat(all_bg_rows) if all_bg_rows else pls.DataFrame()

    return usv_df, background_df, total_noise_filtered


def plot_assignment_stacked_bars(
    assignment_df: pls.DataFrame,
    plot_proportions: bool,
    male_color: str,
    female_color: str,
    unassigned_color: str
) -> tuple[plt.Figure, plt.Axes, dict[str, Any]]:
    """
    Description
    ----------
    This method generates a horizontal stacked bar chart showing either the total
    number of USVs or the relative proportion of USVs assigned to males, females,
    and the unassigned category per recording session.

    The sessions are automatically sorted on the y-axis by the total number of
    vocalizations to make distributions easier to read.
    ----------

    Parameters
    ----------
    assignment_df (pls.DataFrame)
        A Polars DataFrame containing the columns 'session', 'male', 'female',
        and 'unassigned', representing USV counts per category.
    plot_proportions (bool)
        If True, the data will be normalized to proportions (0 to 1) before plotting.
        If False, raw USV counts will be plotted.
    male_color (str)
        Hex code for the male category color.
    female_color (str)
        Hex code for the female category color.
    unassigned_color (str)
        Hex code for the unassigned category color.
    ----------

    Returns
    ----------
    fig (plt.Figure)
        The matplotlib Figure object.
    ax (plt.Axes)
        The matplotlib Axes object containing the stacked bars.
    stats_dict (dict)
        Contains 'total_sessions', 'max_session_total', and 'mean_session_total'.
    ----------
    """

    df_processed = assignment_df.with_columns(
        (pls.col('male') + pls.col('female') + pls.col('unassigned')).alias('total')
    ).sort('total')

    stats_dict = {
        'total_sessions': df_processed.height,
        'max_session_total': df_processed['total'].max(),
        'mean_session_total': df_processed['total'].mean()
    }

    if plot_proportions:
        df_processed = df_processed.with_columns([
            (pls.col('male') / pls.col('total')).alias('male'),
            (pls.col('female') / pls.col('total')).alias('female'),
            (pls.col('unassigned') / pls.col('total')).alias('unassigned')
        ])
        x_label = 'Proportion of Vocalizations'
    else:
        x_label = 'Number of Vocalizations'

    sessions = df_processed['session'].to_numpy()
    male_vals = df_processed['male'].to_numpy()
    female_vals = df_processed['female'].to_numpy()
    unassigned_vals = df_processed['unassigned'].to_numpy()

    fig, ax = plt.subplots(figsize=(6, 10))

    ax.barh(sessions, male_vals, label='Male', color=male_color)
    ax.barh(sessions, female_vals, left=male_vals, label='Female', color=female_color)
    ax.barh(sessions, unassigned_vals, left=male_vals + female_vals, label='Unassigned', color=unassigned_color)

    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel('Recording Session', fontsize=14)
    if not plot_proportions:
        ax.legend(loc='lower right', fontsize=14)

    if plot_proportions:
        ax.set_xlim(0, 1)

    ax.minorticks_off()
    ax.tick_params(axis='x', labelsize=12)
    ax.margins(y=0.001)

    fig.tight_layout()

    return fig, ax, stats_dict


def plot_assignment_summary_panel(
    assignment_df: pls.DataFrame,
    male_color: str,
    female_color: str,
    unassigned_color: str,
    jitter_strength: float
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes, plt.Axes], dict[str, Any]]:
    """
    Description
    ----------
    This method generates a comprehensive 3-panel figure summarizing USV assignments.

    Panel 1: A scatter plot showing variability in USV counts on a log scale.
    Panel 2: A horizontal Seaborn violin plot showing distributions and IQRs.
    Panel 3: An aggregated stacked bar chart showing the grand total and global proportions.
    ----------

    Parameters
    ----------
    assignment_df (pls.DataFrame)
        A Polars DataFrame containing the columns 'session', 'male', 'female',
        and 'unassigned', representing USV counts per category.
    male_color (str)
        Hex code for the male category color.
    female_color (str)
        Hex code for the female category color.
    unassigned_color (str)
        Hex code for the unassigned category color.
    jitter_strength (float)
        The standard deviation of the normal distribution used to jitter points on the x-axis.
    ----------

    Returns
    ----------
    fig (plt.Figure)
        The matplotlib Figure object containing all three panels.
    axes (tuple)
        A tuple of the three matplotlib Axes objects (ax_scatter, ax_violin, ax_bar).
    stats_dict (dict)
        Contains global medians, IQRs, grand totals, and global proportions for each category.
    ----------
    """

    colors = {'male': male_color, 'female': female_color, 'unassigned': unassigned_color}
    x_positions = {'male': 1, 'female': 2, 'unassigned': 3}

    df_long = assignment_df.unpivot(
        index='session',
        on=['male', 'female', 'unassigned'],
        variable_name='category',
        value_name='count'
    )

    fig, (ax_scatter, ax_violin, ax_bar) = plt.subplots(nrows=3, ncols=1, figsize=(6, 9))

    # Panel 1: Scatter plot
    for category, pos in x_positions.items():
        subset = df_long.filter(pls.col('category') == category)
        counts = subset['count'].to_numpy()
        jittered_x = np.random.normal(loc=pos, scale=jitter_strength, size=len(counts))
        ax_scatter.scatter(jittered_x, counts, color=colors[category], label=category, alpha=0.7, s=50)

    ax_scatter.set_xticks(list(x_positions.values()))
    ax_scatter.set_xticklabels(list(x_positions.keys()))
    ax_scatter.xaxis.set_major_locator(plt.NullLocator())
    ax_scatter.set_xlabel('Category')
    ax_scatter.set_ylabel('Number of Vocalizations')
    ax_scatter.grid(axis='y', linestyle='--', alpha=0.4)
    ax_scatter.set_yscale('log')
    ax_scatter.set_title('Variability in USV counts per assignment category')

    # Panel 2: Violin plot
    df_long_pd = df_long.to_pandas()
    sns.violinplot(
        data=df_long_pd,
        x='count',
        y='category',
        hue='category',
        palette=colors,
        inner='box',
        legend=False,
        ax=ax_violin
    )
    ax_violin.set_yticklabels([])
    ax_violin.set_title('Distribution of USV Counts per assignment category')
    ax_violin.set_xlabel('Number of Vocalizations')
    ax_violin.set_ylabel('Category')
    ax_violin.grid(axis='x', linestyle='--', alpha=0.4)

    stats_dict = {}
    stats_text_lines = []

    for cat in ['male', 'female', 'unassigned']:
        data = df_long.filter(pls.col('category') == cat)['count'].to_numpy()
        q1, median, q3 = np.percentile(data, [25, 50, 75])
        iqr = q3 - q1

        stats_dict[f'{cat}_median'] = float(median)
        stats_dict[f'{cat}_iqr'] = float(iqr)
        stats_text_lines.append(f"{cat.capitalize()}: {median:.2f} ± {iqr:.2f}")

    bbox_props = dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.6, edgecolor='grey')
    ax_violin.text(0.95, 0.05, "\n".join(stats_text_lines),
                   transform=ax_violin.transAxes, fontsize=9,
                   verticalalignment='bottom', horizontalalignment='right',
                   bbox=bbox_props, color='white')

    # Panel 3: Stacked proportions bar
    totals_df = df_long.group_by('category').agg(pls.sum('count').alias('total_count'))
    grand_total = df_long['count'].sum()
    totals_df = totals_df.with_columns((pls.col('total_count') / grand_total).alias('proportion'))
    totals_pd = totals_df.to_pandas().set_index('category')

    male_total = totals_pd.loc['male', 'total_count']
    female_total = totals_pd.loc['female', 'total_count']
    unassigned_total = totals_pd.loc['unassigned', 'total_count']

    male_prop = totals_pd.loc['male', 'proportion']
    female_prop = totals_pd.loc['female', 'proportion']
    unassigned_prop = totals_pd.loc['unassigned', 'proportion']

    stats_dict.update({
        'grand_total': float(grand_total),
        'male_total': float(male_total),
        'female_total': float(female_total),
        'unassigned_total': float(unassigned_total),
        'male_proportion': float(male_prop),
        'female_proportion': float(female_prop),
        'unassigned_proportion': float(unassigned_prop)
    })

    ax_bar.barh(0, male_total, color=colors['male'], edgecolor='white', height=0.4)
    ax_bar.barh(0, female_total, left=male_total, color=colors['female'], edgecolor='white', height=0.4)
    ax_bar.barh(0, unassigned_total, left=male_total + female_total, color=colors['unassigned'], edgecolor='white', height=0.4)

    text_y_offset = 0.1
    ax_bar.text(male_total / 2, text_y_offset, f'{male_total}', ha='center', va='center', fontsize=9)
    ax_bar.text(male_total + (female_total / 2), text_y_offset, f'{female_total}', ha='center', va='center', fontsize=9)
    ax_bar.text(male_total + female_total + (unassigned_total / 2), text_y_offset, f'{unassigned_total}', ha='center', va='center', fontsize=9)

    ax_bar.text(male_total / 2, -text_y_offset, f'{male_prop:.2f}', ha='center', va='center', fontsize=9)
    ax_bar.text(male_total + (female_total / 2), -text_y_offset, f'{female_prop:.2f}', ha='center', va='center', fontsize=9)
    ax_bar.text(male_total + female_total + (unassigned_total / 2), -text_y_offset, f'{unassigned_prop:.2f}', ha='center', va='center', fontsize=9)

    ax_bar.set_title('USV assignment summary', fontsize=14)
    ax_bar.set_xlabel('Total Number of Vocalizations', fontsize=14)
    ax_bar.get_yaxis().set_visible(False)
    ax_bar.spines[['left', 'top', 'right']].set_visible(False)
    ax_bar.set_ylim(-1, 1)
    ax_bar.grid(False)
    ax_bar.set_box_aspect(0.3)
    ax_bar.tick_params(axis='x', labelsize=12)

    fig.tight_layout(pad=2.0)

    return fig, (ax_scatter, ax_violin, ax_bar), stats_dict


def plot_animal_participation_stats(
    animal_stats: dict[str, dict[str, int]],
    sex_label: str,
    bar_color: str,
    text_color: str
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes], dict[str, Any]]:
    """
    Description
    ----------
    This method visualizes individual animal participation by generating two side-by-side
    horizontal bar charts. The first panel displays the number of sessions each animal
    participated in, and the second displays their average USV vocal rate per session.
    ----------

    Parameters
    ----------
    animal_stats (dict)
        A nested dictionary containing animal IDs as keys and sub-dictionaries
        with 'session_count' and 'total_usvs' as values.
    sex_label (str)
        A string identifier used for the plot title (e.g., 'Male' or 'Female').
    bar_color (str)
        Hex code for the bar color.
    text_color (str)
        Color for the bar label annotations.
    ----------

    Returns
    ----------
    fig (plt.Figure)
        The matplotlib Figure object.
    axes (tuple)
        A tuple of the two matplotlib Axes objects (ax_sessions, ax_rate).
    stats_dict (dict)
        Contains summary statistics regarding animal session counts and vocal rates.
    ----------
    """

    df = pd.DataFrame.from_dict(animal_stats, orient='index')

    if not df.empty:
        df['vocal_rate'] = (df['total_usvs'] / df['session_count']).fillna(0)
    else:
        df['vocal_rate'] = pd.Series(dtype='float64')

    df_sorted = df.sort_values('session_count', ascending=True)

    stats_dict = {
        'total_animals': len(df),
        'mean_session_count': df['session_count'].mean() if not df.empty else 0,
        'max_session_count': df['session_count'].max() if not df.empty else 0,
        'mean_vocal_rate': df['vocal_rate'].mean() if not df.empty else 0,
        'max_vocal_rate': df['vocal_rate'].max() if not df.empty else 0
    }

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5), sharey=True)
    fig.suptitle(f'{sex_label} Animal N={len(animal_stats.keys())} Statistics', y=1.02, fontsize=16)

    # Subplot 1: Session count
    ax1 = axes[0]
    bars1 = ax1.barh(df_sorted.index, df_sorted['session_count'], color=bar_color)
    ax1.set_title('Session Appearances', fontsize=14)
    ax1.set_xlabel('Number of Sessions', fontsize=14)
    ax1.bar_label(bars1, fmt='%d', padding=3, color=text_color, fontsize=9)
    ax1.grid(axis='x', linestyle='--', alpha=0.4)
    ax1.spines[['top', 'right']].set_visible(False)
    ax1.set_xlim(right=ax1.get_xlim()[1] * 1.1)
    ax1.tick_params(axis='x', labelsize=12)
    ax1.minorticks_off()

    # Subplot 2: Vocal rate
    ax2 = axes[1]
    bars2 = ax2.barh(df_sorted.index, df_sorted['vocal_rate'], color=bar_color)
    ax2.set_title('Vocal Rate (USVs/Session)', fontsize=14)
    ax2.set_xlabel('Average USVs per Session', fontsize=14)
    ax2.bar_label(bars2, fmt='%.1f', padding=3, color=text_color, fontsize=9)
    ax2.grid(axis='x', linestyle='--', alpha=0.4)
    ax2.spines[['top', 'right']].set_visible(False)
    ax2.set_xlim(right=ax2.get_xlim()[1] * 1.1)
    ax2.tick_params(axis='x', labelsize=12)
    ax2.minorticks_off()

    fig.tight_layout(pad=1.0)

    return fig, (ax1, ax2), stats_dict

def plot_polar_kde_distance_angle(
    usv_distances: np.ndarray,
    usv_angles_deg: np.ndarray,
    all_distances: np.ndarray,
    all_angles_deg: np.ndarray,
    max_distance: float,
    colormap: mcolors.Colormap | str,
    ylabel: str,
    occupancy_threshold: float,
    max_kde_points: int = 50000
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes], dict[str, Any]]:
    """
    Description
    ----------
    Generates a two-panel half-circle polar contour plot comparing raw USV spatial
    density to occupancy-normalized vocalization likelihood.

    The first panel (Raw Density) visualizes the distribution of nose-to-nose
    distances and relative angles at the moment of USV emission. The second panel
    (Normalized Likelihood) divides the USV density by the overall spatial
    occupancy of the animals across all recorded frames, highlighting areas where
    vocalizations occur more frequently than expected by chance.

    To ensure high visual quality and computational efficiency, the function:
    1. Subsamples background data to prevent KDE processing overhead.
    2. Applies 98th percentile clipping to both plots to prevent outliers from
       washing out the color scale or creating jagged contour artifacts.
    3. Utilizes a high number of contour levels (100) for smooth gradients.

    Parameters
    ----------
    usv_distances : np.ndarray
        1D array of nose-to-nose distances (cm) during USV emission.
    usv_angles_deg : np.ndarray
        1D array of relative angles (degrees) during USV emission.
    all_distances : np.ndarray
        1D array of nose-to-nose distances (cm) across all video frames.
    all_angles_deg : np.ndarray
        1D array of relative angles (degrees) across all video frames.
    max_distance : float
        Maximum radial distance (cm) to include in the KDE and plot.
    colormap : mcolors.Colormap | str
        Matplotlib colormap for the filled contours.
    ylabel : str
        Label for the radial (distance) axes.
    occupancy_threshold : float
        Minimum occupancy density value required to calculate likelihood.
        Areas below this threshold are masked to prevent division-by-zero
        artifacts in the normalized plot.
    max_kde_points : int, default 50000
        Maximum number of points used for KDE calculation. Data exceeding this
        is randomly subsampled to optimize performance.

    Returns
    ----------
    fig : plt.Figure
        The generated matplotlib figure.
    axes : tuple[plt.Axes, plt.Axes]
        The (ax_raw, ax_norm) polar axes objects.
    stats_dict : dict[str, Any]
        Dictionary containing 'n_usv_points' and 'n_all_points' reflecting
        original counts before subsampling.
    """

    # 1. Filter and prepare data
    valid_usv = ~np.isnan(usv_distances) & ~np.isnan(usv_angles_deg) & (usv_distances <= max_distance)
    filt_usv_dist = usv_distances[valid_usv]
    abs_usv_rad = np.deg2rad(np.abs(usv_angles_deg[valid_usv]))

    valid_all = ~np.isnan(all_distances) & ~np.isnan(all_angles_deg) & (all_distances <= max_distance)
    filt_all_dist = all_distances[valid_all]
    abs_all_rad = np.deg2rad(np.abs(all_angles_deg[valid_all]))

    stats_dict = {'n_usv_points': len(filt_usv_dist), 'n_all_points': len(filt_all_dist)}

    # 2. Subsample for performance
    if len(filt_usv_dist) > max_kde_points:
        idx = np.random.choice(len(filt_usv_dist), max_kde_points, replace=False)
        filt_usv_dist, abs_usv_rad = filt_usv_dist[idx], abs_usv_rad[idx]
    if len(filt_all_dist) > max_kde_points:
        idx = np.random.choice(len(filt_all_dist), max_kde_points, replace=False)
        filt_all_dist, abs_all_rad = filt_all_dist[idx], abs_all_rad[idx]

    fig = plt.figure(figsize=(10, 4))
    ax_raw = fig.add_subplot(121, projection='polar')
    ax_norm = fig.add_subplot(122, projection='polar')

    if len(abs_usv_rad) < 2 or len(abs_all_rad) < 2:
        ax_raw.set_title("Insufficient Data for KDE")
        ax_norm.set_title("Insufficient Data for KDE")
        return fig, (ax_raw, ax_norm), stats_dict

    try:
        # 3. Compute KDEs
        kde_usv = gaussian_kde(np.vstack([abs_usv_rad, filt_usv_dist]))
        kde_all = gaussian_kde(np.vstack([abs_all_rad, filt_all_dist]))

        n_grid = 200
        ag, rg = np.linspace(0, np.pi, n_grid), np.linspace(0, max_distance, n_grid)
        mesh = np.stack(np.meshgrid(ag, rg), axis=0)

        dens_usv = kde_usv(mesh.reshape(2, -1)).reshape(n_grid, n_grid)
        dens_all = kde_all(mesh.reshape(2, -1)).reshape(n_grid, n_grid)

        # 4. Normalize and mask
        norm_dens = np.zeros_like(dens_usv)
        valid_mask = dens_all > occupancy_threshold
        norm_dens[valid_mask] = dens_usv[valid_mask] / dens_all[valid_mask]

        # 5. Raw plot (left) - Robust scaling with 98th percentile clipping
        vmax_raw = np.percentile(dens_usv, 98)
        plot_raw = np.clip(dens_usv, 0, vmax_raw)

        contour_raw = ax_raw.contourf(ag, rg, plot_raw, cmap=colormap, levels=100)
        cb_raw = fig.colorbar(contour_raw, ax=ax_raw, shrink=0.7, pad=0.1)
        cb_raw.set_label('Raw Density Estimate', fontsize=12)
        cb_raw.locator = plt.MaxNLocator(nbins=5)
        cb_raw.update_ticks()

        # 6. Normalized plot (right) - Robust scaling with 98th percentile clipping
        if np.any(valid_mask):
            vmax_norm = np.percentile(norm_dens[valid_mask], 98)
            plot_norm = np.clip(norm_dens, 0, vmax_norm)
        else:
            plot_norm = norm_dens
            vmax_norm = 1.0

        contour_norm = ax_norm.contourf(ag, rg, plot_norm, cmap=colormap, levels=100)
        cb_norm = fig.colorbar(contour_norm, ax=ax_norm, shrink=0.7, pad=0.1)
        cb_norm.set_label('Normalized Likelihood', fontsize=12)
        cb_norm.locator = plt.MaxNLocator(nbins=5)
        cb_norm.update_ticks()

        # 7. Common polar formatting
        for ax in (ax_raw, ax_norm):
            ax.set_thetamin(0)
            ax.set_thetamax(180)
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            ax.set_ylabel(ylabel, fontsize=12, labelpad=5)
            ax.set_rticks(np.linspace(0, max_distance, 5))
            ax.set_rlabel_position(90)
            ax.tick_params(axis='both', labelsize=10)

        ax_raw.set_title('USV Count Density', fontsize=14, pad=15)
        ax_norm.set_title('Occupancy-Normalized Likelihood', fontsize=14, pad=15)

    except np.linalg.LinAlgError:
        ax_raw.set_title("KDE Computation Failed (Singular Matrix)")
        ax_norm.set_title("KDE Computation Failed (Singular Matrix)")

    fig.tight_layout()
    return fig, (ax_raw, ax_norm), stats_dict

def plot_behavior_duration_regressions(
    male_df: pd.DataFrame,
    female_df: pd.DataFrame,
    male_color: str,
    female_color: str,
    line_color: str
) -> tuple[plt.Figure, np.ndarray, dict[str, Any]]:
    """
    Description
    ----------
    This method generates a 2x2 grid of Seaborn regression plots assessing the
    relationship between behavioral features (distance, angle) and USV duration
    for both sexes.

    It calculates Pearson's r and p-values for each relationship, embedding the
    results directly onto the plot axes and returning them in the statistics dictionary.
    ----------

    Parameters
    ----------
    male_df (pd.DataFrame)
        A Pandas DataFrame containing 'distance', 'angle', and 'usv_duration' for males.
    female_df (pd.DataFrame)
        A Pandas DataFrame containing 'distance', 'angle', and 'usv_duration' for females.
    male_color (str)
        Hex code for the male scatter points.
    female_color (str)
        Hex code for the female scatter points.
    line_color (str)
        Hex code for the regression lines.
    ----------

    Returns
    ----------
    fig (plt.Figure)
        The matplotlib Figure object.
    axes (np.ndarray)
        The 2x2 numpy array of matplotlib Axes objects.
    stats_dict (dict)
        Contains Pearson r and p-values for all four statistical comparisons.
    ----------
    """

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
    stats_dict = {}

    def plot_and_annotate(data: pd.DataFrame, x_col: str, y_col: str, ax: plt.Axes, color: str, prefix: str):
        sns.regplot(data=data, x=x_col, y=y_col, ax=ax,
                    scatter_kws={'color': color, 'alpha': 0.5, 's': 15},
                    line_kws={'color': line_color})

        r, p = pearsonr(data[x_col], data[y_col])
        stats_dict[f'{prefix}_r'] = float(r)
        stats_dict[f'{prefix}_p'] = float(p)

        p_text = f'p = {p:.3f}' if p >= 0.001 else 'p < .001'
        stat_text = f'r = {r:.2f}, {p_text}'

        x_pos = ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.01
        y_pos = ax.get_ylim()[1] - (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.01
        ax.text(x_pos, y_pos, stat_text, fontsize=9, verticalalignment='top')

        ax.grid(False)
        ax.minorticks_off()
        ax.tick_params(axis='both', labelsize=10)

    plot_and_annotate(male_df, 'distance', 'usv_duration', axes[0, 0], male_color, 'male_distance')
    axes[0, 0].set_xlabel("Nose-Nose Distance (cm)", fontsize=14)
    axes[0, 0].set_ylabel("USV duration (s)", fontsize=14)

    plot_and_annotate(male_df, 'angle', 'usv_duration', axes[0, 1], male_color, 'male_angle')
    axes[0, 1].set_xlabel("MF Angle (°)", fontsize=14)
    axes[0, 1].set_ylabel("USV duration (s)", fontsize=14)

    plot_and_annotate(female_df, 'distance', 'usv_duration', axes[1, 0], female_color, 'female_distance')
    axes[1, 0].set_xlabel("Nose-Nose Distance (cm)", fontsize=14)
    axes[1, 0].set_ylabel("USV duration (s)", fontsize=14)

    plot_and_annotate(female_df, 'angle', 'usv_duration', axes[1, 1], female_color, 'female_angle')
    axes[1, 1].set_xlabel("FM Angle (°)", fontsize=14)
    axes[1, 1].set_ylabel("USV duration (s)", fontsize=14)

    fig.tight_layout()
    return fig, axes, stats_dict


def plot_distance_by_assignment_kde_anova(
    df_plot: pd.DataFrame,
    min_samples_anova: int,
    male_color: str,
    female_color: str,
    unassigned_color: str
) -> tuple[plt.Figure, plt.Axes, dict[str, Any]]:
    """
    Description
    ----------
    This method generates an overlaid KDE plot comparing nose-to-nose distances
    between male, female, and unassigned USV categories.

    It computes a One-Way ANOVA, effect size (Omega-Squared), and Tukey's HSD
    post-hoc tests across the assignment categories, displaying a comprehensive
    statistical summary text box directly on the plot.
    ----------

    Parameters
    ----------
    df_plot (pd.DataFrame)
        A Pandas DataFrame containing the columns 'distance' and 'category'
        ('male', 'female', 'unassigned').
    min_samples_anova (int)
        The minimum number of samples required per group to run the ANOVA.
    male_color (str)
        Hex code for the male KDE density.
    female_color (str)
        Hex code for the female KDE density.
    unassigned_color (str)
        Hex code for the unassigned KDE density.
    ----------

    Returns
    ----------
    fig (plt.Figure)
        The matplotlib Figure object.
    ax (plt.Axes)
        The matplotlib Axes object.
    stats_dict (dict)
        Contains ANOVA F-statistic, p-value, df, Omega-Squared, means, SEMs,
        and significant Tukey HSD pairs.
    ----------
    """

    colors = {'male': male_color, 'female': female_color, 'unassigned': unassigned_color}
    stats_dict: dict[str, Any] = {}

    male_dist = df_plot[df_plot['category'] == 'male']['distance'].to_numpy()
    female_dist = df_plot[df_plot['category'] == 'female']['distance'].to_numpy()
    unassigned_dist = df_plot[df_plot['category'] == 'unassigned']['distance'].to_numpy()

    anova_result_text = "ANOVA: Not enough samples"
    omega_squared_text = "Omega-Squared: N/A"
    tukey_summary_for_plot = "Post-hoc (Tukey): N/A"

    if (len(male_dist) >= min_samples_anova and
        len(female_dist) >= min_samples_anova and
        len(unassigned_dist) >= min_samples_anova):

        model = ols('distance ~ C(category)', data=df_plot).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)

        f_value = anova_table['F'].iloc[0]
        p_value_anova = anova_table['PR(>F)'].iloc[0]
        df_between = anova_table['df'].iloc[0]
        df_within = anova_table['df'].iloc[1]
        ss_between = anova_table['sum_sq'].iloc[0]
        ss_within = anova_table['sum_sq'].iloc[1]
        ms_within = ss_within / df_within

        stats_dict['anova'] = {'F': float(f_value), 'p': float(p_value_anova), 'df_b': float(df_between), 'df_w': float(df_within)}

        p_text_anova = f'p = {p_value_anova:.3f}' if p_value_anova >= 0.001 else 'p < 0.001'
        anova_result_text = f"ANOVA: F({int(df_between)}, {int(df_within)}) = {f_value:.2f}, {p_text_anova}"

        omega_sq = (ss_between - (df_between * ms_within)) / (ss_between + ss_within + ms_within)
        stats_dict['omega_squared'] = float(omega_sq)
        omega_squared_text = f"Omega-Squared (ω²): {omega_sq:.3f} \n"

        if p_value_anova < 0.05:
            tukey = pairwise_tukeyhsd(endog=df_plot['distance'], groups=df_plot['category'], alpha=0.05)
            tukey_df = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])
            significant_pairs = tukey_df[tukey_df['reject'] == True]

            stats_dict['tukey_significant'] = significant_pairs.to_dict('records')

            if not significant_pairs.empty:
                pairs_text_list = []
                for _, row in significant_pairs.iterrows():
                     p_adj = row['p-adj']
                     p_adj_text = f'p={p_adj:.3f}' if p_adj >= 0.001 else 'p<.001'
                     pairs_text_list.append(f"{row['group1']} vs {row['group2']} ({p_adj_text})")
                tukey_summary_for_plot = "\n ".join(pairs_text_list)
            else:
                tukey_summary_for_plot = "Post-hoc (Tukey): No sig. pairs"
        else:
            tukey_summary_for_plot = "Post-hoc (Tukey): ANOVA not significant"

    stats_lines = []
    stats_dict['descriptive'] = {}
    for cat, data in [('Male', male_dist), ('Female', female_dist), ('Unassigned', unassigned_dist)]:
        if len(data) > 0:
            mean_val, sem_val = np.mean(data), sem(data)
            stats_dict['descriptive'][cat] = {'mean': float(mean_val), 'sem': float(sem_val)}
            stats_lines.append(f"{cat}: {mean_val:.2f} ± {sem_val:.2f}")
        else:
            stats_lines.append(f"{cat}: N/A")

    full_stats_text = (f"{anova_result_text}\n{omega_squared_text}" + "\n".join(stats_lines) + f"\n{tukey_summary_for_plot}")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.kdeplot(data=df_plot, x='distance', hue='category', palette=colors, fill=True, alpha=0.5, common_norm=False, bw_adjust=1.5, ax=ax)

    bbox_props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7, edgecolor='grey')
    ax.text(0.95, 0.98, full_stats_text, transform=ax.transAxes, fontsize=12, verticalalignment='top', horizontalalignment='right', bbox=bbox_props, color='#202020')

    ax.set_xlabel('Nose-Nose distance (cm)', fontsize=14)
    ax.set_ylabel('Density', fontsize=14)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xlim(-10, 40)
    ax.minorticks_off()
    ax.tick_params(axis='both', labelsize=12)

    fig.tight_layout()
    return fig, ax, stats_dict


def plot_unassigned_proportion_vs_distance_jointplot(
    df_combined: pd.DataFrame,
    scatter_color: str,
    line_color: str,
    hist_color: str
) -> tuple[sns.JointGrid, dict[str, Any]]:
    """
    Description
    ----------
    This method generates a Seaborn JointGrid mapping the correlation between
    the median nose-to-nose distance in a session and the overall proportion
    of unassigned USVs in that session.

    It computes Pearson's r and p-values and renders the statistics directly
    onto the central regression panel.
    ----------

    Parameters
    ----------
    df_combined (pd.DataFrame)
        A Pandas DataFrame containing 'median_distance' and 'unassigned_prop'
        for each session.
    scatter_color (str)
        Hex code for the joint plot scatter points.
    line_color (str)
        Hex code for the regression line and statistics text.
    hist_color (str)
        Hex code for the marginal histograms.
    ----------

    Returns
    ----------
    g (sns.JointGrid)
        The Seaborn JointGrid object containing the figure and axes.
    stats_dict (dict)
        Contains 'pearson_r' and 'pearson_p'.
    ----------
    """

    stats_dict = {}

    g = sns.jointplot(
        data=df_combined,
        x='median_distance',
        y='unassigned_prop',
        kind='reg',
        height=7,
        joint_kws={'scatter_kws': {'alpha': 0.7, 's': 50, 'color': scatter_color}},
        line_kws={'color': line_color},
        marginal_kws={'color': hist_color, 'kde': False}
    )

    g.ax_joint.grid(False)

    if len(df_combined) >= 2:
        r, p = pearsonr(df_combined['median_distance'], df_combined['unassigned_prop'])
        stats_dict['pearson_r'] = float(r)
        stats_dict['pearson_p'] = float(p)

        p_text = f'p = {p:.3f}' if p >= 0.001 else 'p < .001'
        stat_text = f'r = {r:.2f}, {p_text}'

        g.ax_joint.text(4.0, 0.12, stat_text, fontsize=12, color=line_color)

    g.set_axis_labels('Median Nose-Nose Distance', 'Proportion of Unassigned Calls', fontsize=12)
    g.figure.suptitle('Correlation between Distance and Unassigned Calls', y=1.02, fontsize=14)

    g.figure.tight_layout()
    return g, stats_dict

def plot_duration_histograms_by_sex(
    plot_data: pd.DataFrame,
    bin_width_ms: float,
    max_duration_ms: float,
    male_color: str,
    female_color: str
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes], dict[str, Any]]:
    """
    Description
    ----------
    This method generates stacked histograms displaying the distribution of
    individual USV durations for males and females.

    It computes the mean and median durations for each sex, plots them as
    vertical dashed and dotted lines over the histograms, and compiles these
    statistics into a dictionary for downstream analysis.
    ----------

    Parameters
    ----------
    plot_data (pd.DataFrame)
        A Pandas DataFrame containing at least two columns: 'sex' ('male'/'female')
        and 'duration_ms' (duration of the vocalization in milliseconds).
    bin_width_ms (float)
        The width of each histogram bin in milliseconds.
    max_duration_ms (float)
        The maximum duration to include in the histogram x-axis limit.
    male_color (str)
        Hex code for the male histogram bars.
    female_color (str)
        Hex code for the female histogram bars.
    ----------

    Returns
    ----------
    fig (plt.Figure)
        The matplotlib Figure object.
    axes (tuple)
        A tuple of two matplotlib Axes objects (ax_male, ax_female).
    stats_dict (dict)
        Contains the calculated mean and median durations for males and females.
    ----------
    """

    male_durations = plot_data[plot_data['sex'] == 'male']['duration_ms']
    female_durations = plot_data[plot_data['sex'] == 'female']['duration_ms']

    bins = np.arange(0, max_duration_ms + bin_width_ms, bin_width_ms)

    stats_dict = {}

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 8), sharex=True)

    # Male
    if not male_durations.empty:
        male_mean, male_median = male_durations.mean(), male_durations.median()
        stats_dict['male_mean'] = float(male_mean)
        stats_dict['male_median'] = float(male_median)

        axes[0].hist(male_durations, bins=bins, color=male_color, alpha=0.8, edgecolor='white', linewidth=0.5)
        axes[0].axvline(male_mean, color='#202020', linestyle='--', linewidth=1.5, label=f'Mean ({male_mean:.1f} ms)')
        axes[0].axvline(male_median, color='#202020', linestyle=':', linewidth=1.5, label=f'Median ({male_median:.1f} ms)')
        axes[0].legend(fontsize=12)

    axes[0].set_ylabel("Number of USVs", fontsize=14)
    axes[0].grid(axis='y', linestyle='--', alpha=0.4)
    axes[0].set_xlim(0, max_duration_ms)
    axes[0].minorticks_off()
    axes[0].tick_params(axis='both', labelsize=12)

    # Female
    if not female_durations.empty:
        female_mean, female_median = female_durations.mean(), female_durations.median()
        stats_dict['female_mean'] = float(female_mean)
        stats_dict['female_median'] = float(female_median)

        axes[1].hist(female_durations, bins=bins, color=female_color, alpha=0.8, edgecolor='white', linewidth=0.5)
        axes[1].axvline(female_mean, color='#202020', linestyle='--', linewidth=1.5, label=f'Mean ({female_mean:.1f} ms)')
        axes[1].axvline(female_median, color='#202020', linestyle=':', linewidth=1.5, label=f'Median ({female_median:.1f} ms)')
        axes[1].legend(fontsize=12)

    axes[1].set_xlabel("USV Duration (ms)", fontsize=14)
    axes[1].set_ylabel("Number of USVs", fontsize=14)
    axes[1].grid(axis='y', linestyle='--', alpha=0.4)
    axes[1].minorticks_off()
    axes[1].tick_params(axis='both', labelsize=12)

    fig.tight_layout()
    return fig, (axes[0], axes[1]), stats_dict


def plot_hourly_regressions(
    df_raw: pd.DataFrame,
    y_col: str,
    y_label: str,
    male_color: str,
    female_color: str,
    line_color: str
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes], dict[str, Any]]:
    """
    Description
    ----------
    This method generates a two-panel Seaborn regression plot (male top, female bottom)
    mapping the hour of the day against a specified continuous USV metric (e.g.,
    USV count per session or individual USV duration) to assess global vocal fatigue.

    It computes Pearson's r and p-values for both sexes to quantify the temporal
    trend, rendering the statistics in text boxes on the respective plots.
    ----------

    Parameters
    ----------
    df_raw (pd.DataFrame)
        A Pandas DataFrame containing at least 'hour', 'sex', and the target y_col.
    y_col (str)
        The name of the column containing the dependent variable (e.g., 'usv_count'
        or 'duration').
    y_label (str)
        The text label for the Y-axis.
    male_color (str)
        Hex code for the male scatter points.
    female_color (str)
        Hex code for the female scatter points.
    line_color (str)
        Hex code for the regression line and text.
    ----------

    Returns
    ----------
    fig (plt.Figure)
        The matplotlib Figure object.
    axes (tuple)
        A tuple of two matplotlib Axes objects (ax_male, ax_female).
    stats_dict (dict)
        Contains Pearson r and p-values for male and female temporal regressions.
    ----------
    """

    male_df = df_raw[df_raw['sex'] == 'male']
    female_df = df_raw[df_raw['sex'] == 'female']

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(4, 5), sharex=True)
    stats_dict = {}

    def annotate_stats(ax: plt.Axes, x_data: np.ndarray, y_data: np.ndarray, prefix: str):
        valid_idx = ~np.isnan(x_data) & ~np.isnan(y_data)
        if valid_idx.sum() >= 2:
            r, p = pearsonr(x_data[valid_idx], y_data[valid_idx])
            stats_dict[f'{prefix}_r'] = float(r)
            stats_dict[f'{prefix}_p'] = float(p)

            p_text = f'p = {p:.3f}' if p >= 0.001 else 'p < .001'
            stat_text = f'r = {r:.2f}, {p_text}'

            x_pos = ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.01
            y_pos = ax.get_ylim()[1] - (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02
            ax.text(x_pos, y_pos, stat_text, fontsize=9, verticalalignment='top')

    # Male
    if not male_df.empty:
        sns.regplot(data=male_df, x='hour', y=y_col, ax=axes[0],
                    scatter_kws={'color': male_color, 'alpha': 0.3, 's': 15},
                    line_kws={'color': line_color})
        annotate_stats(axes[0], male_df['hour'].values, male_df[y_col].values, 'male')

    axes[0].set_ylabel(y_label, fontsize=14)
    axes[0].set_xlabel('')
    axes[0].grid(False)
    axes[0].minorticks_off()
    axes[0].tick_params(axis='both', labelsize=12)

    # Female
    if not female_df.empty:
        sns.regplot(data=female_df, x='hour', y=y_col, ax=axes[1],
                    scatter_kws={'color': female_color, 'alpha': 0.3, 's': 15},
                    line_kws={'color': line_color})
        annotate_stats(axes[1], female_df['hour'].values, female_df[y_col].values, 'female')

    axes[1].set_ylabel(y_label, fontsize=14)
    axes[1].set_xlabel("Hour of Day (24h)", fontsize=14)
    axes[1].set_xticks(range(0, 24, 2))
    axes[1].grid(False)
    axes[1].minorticks_off()
    axes[1].tick_params(axis='both', labelsize=12)

    axes[0].set_xlim(11, 23)
    axes[1].set_xlim(11, 23)
    axes[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    fig.tight_layout()
    return fig, (axes[0], axes[1]), stats_dict

def plot_local_fatigue_binned_trends(
    binned_df: pd.DataFrame,
    y_mean_col: str,
    y_sem_col: str,
    y_label: str,
    bin_width_seconds: int,
    n_bins: int,
    male_color: str,
    female_color: str,
    use_log_scale: bool
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes], dict[str, Any]]:
    """
    Description
    ----------
    This method visualizes within-session temporal dynamics (local vocal fatigue)
    by plotting aggregated metrics across discrete time bins.

    It generates two vertically stacked line plots (male and female) displaying
    the mean values with shaded SEM bounds. The x-axis labels are dynamically
    calculated from the bin width to represent session minutes.
    ----------

    Parameters
    ----------
    binned_df (pd.DataFrame)
        A Pandas DataFrame containing aggregated bin data with columns: 'sex',
        'time_bin', and the specified mean and sem target columns.
    y_mean_col (str)
        The column name containing the calculated means (e.g., 'mean_usv').
    y_sem_col (str)
        The column name containing the standard error of the mean (e.g., 'sem_usv').
    y_label (str)
        The text label for the Y-axis.
    bin_width_seconds (int)
        The duration of each temporal bin in seconds, used to generate minute labels.
    n_bins (int)
        The total number of sequential bins.
    male_color (str)
        Hex code for the male lines and shaded regions.
    female_color (str)
        Hex code for the female lines and shaded regions.
    use_log_scale (bool)
        If True, applies a base-10 logarithmic scale to the Y-axes.
    ----------

    Returns
    ----------
    fig (plt.Figure)
        The matplotlib Figure object.
    axes (tuple)
        A tuple of two matplotlib Axes objects (ax_male, ax_female).
    stats_dict (dict)
        Contains global min/max boundaries for plotting or scaling purposes.
    ----------
    """

    male_df = binned_df[binned_df['sex'] == 'male']
    female_df = binned_df[binned_df['sex'] == 'female']

    stats_dict = {
        'global_max': float(binned_df[y_mean_col].max() if not binned_df.empty else 0),
        'global_min': float(binned_df[y_mean_col].min() if not binned_df.empty else 0)
    }

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(5, 5), sharex=True)

    # Male
    if not male_df.empty:
        axes[0].plot(male_df['time_bin'], male_df[y_mean_col], marker='o', linestyle='-', color=male_color)
        axes[0].fill_between(
            male_df['time_bin'],
            male_df[y_mean_col] - male_df[y_sem_col],
            male_df[y_mean_col] + male_df[y_sem_col],
            alpha=0.3, color=male_color
        )

    axes[0].set_ylabel(y_label, fontsize=14)
    axes[0].grid(axis='y', linestyle='--', alpha=0.4)
    axes[0].minorticks_off()
    axes[0].tick_params(axis='y', labelsize=12)
    if use_log_scale:
        axes[0].set_yscale('log')

    # Female
    if not female_df.empty:
        axes[1].plot(female_df['time_bin'], female_df[y_mean_col], marker='o', linestyle='-', color=female_color)
        axes[1].fill_between(
            female_df['time_bin'],
            female_df[y_mean_col] - female_df[y_sem_col],
            female_df[y_mean_col] + female_df[y_sem_col],
            alpha=0.3, color=female_color
        )

    axes[1].set_ylabel(y_label, fontsize=14)
    axes[1].grid(axis='y', linestyle='--', alpha=0.4)
    axes[1].minorticks_off()
    axes[1].tick_params(axis='y', labelsize=12)
    if use_log_scale:
        axes[1].set_yscale('log')

    bin_width_min = bin_width_seconds // 60
    bin_labels = [f"{i*bin_width_min}-{(i+1)*bin_width_min}" for i in range(n_bins)]

    axes[1].set_xlabel("Time in Session (minutes)", fontsize=14)
    axes[1].set_xticks(range(n_bins))
    axes[1].set_xticklabels(bin_labels)
    axes[0].tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)

    fig.tight_layout()
    return fig, (axes[0], axes[1]), stats_dict

def plot_category_local_fatigue_heatmap(
    binned_df: pd.DataFrame,
    bin_width_seconds: int,
    n_bins: int,
    smoothing_sigma: float = 0.75,
    colormap: str = 'inferno',
    facet_figsize: tuple[int, int] = (12, 10)
) -> tuple[plt.Figure, np.ndarray, dict[str, Any]]:
    """
    Description
    ----------
    Generates a two-panel figure (Male and Female) containing smoothed heatmaps
    visualizing the temporal decay (local fatigue) of specific USV categories
    over the course of a session.

    The function applies a 1D Gaussian filter along the time axis to reduce
    stochastic noise in sparse categories, followed by row-wise normalization
    to isolate the fatigue 'shape' from absolute counts. Inferno is used as
    the default colormap for high-contrast visualization of density.

    Parameters
    ----------
    binned_df : pd.DataFrame
        A Pandas DataFrame containing the columns 'session_id', 'sex',
        'category', 'time_bin', and 'usv_count'. This should be the
        concatenated result of multiple sessions.
    bin_width_seconds : int
        The duration of each time bin in seconds (used for axis labeling).
    n_bins : int
        The total number of sequential time bins to display on the x-axis.
    smoothing_sigma : float, default 0.75
        The standard deviation for the Gaussian kernel applied along the
        time axis. Set to 0 to disable smoothing.
    colormap : str, default 'inferno'
        The matplotlib colormap used for the heatmap intensity.
    facet_figsize : tuple[int, int], default (12, 10)
        The total size of the generated figure.

    Returns
    ----------
    fig : plt.Figure
        The generated matplotlib figure.
    axes : np.ndarray
        A 1D array containing the two Axes objects (ax_male, ax_female).
    stats_dict : dict
        A dictionary containing the peak raw vocal rates for each category
        before normalization, used for referencing absolute magnitude.
    """

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=facet_figsize, sharex=True)
    ax_male, ax_female = axes

    stats_dict = {'male_peaks': {}, 'female_peaks': {}}
    bin_labels = [f"{i*(bin_width_seconds//60)}-{(i+1)*(bin_width_seconds//60)}" for i in range(n_bins)]

    # Sex-specific titles and mapping
    plot_configs = [
        ('male', ax_male, 'Male Local Fatigue (Smoothed)'),
        ('female', ax_female, 'Female Local Fatigue (Smoothed)')
    ]

    for sex_key, ax, title in plot_configs:
        # 1. Isolate sex and aggregate across sessions to get the mean count per bin
        subset = binned_df[binned_df['sex'] == sex_key]

        if subset.empty:
            ax.text(0.5, 0.5, f"No {sex_key} data available", ha='center', va='center')
            continue

        # Aggregate: Mean count per category per bin across all sessions
        agg_bin = subset.groupby(['category', 'time_bin'])['usv_count'].mean().reset_index()

        # 2. Pivot to Heatmap format (Rows = Category, Columns = Time Bin)
        pivot_df = agg_bin.pivot(index='category', columns='time_bin', values='usv_count')
        pivot_df = pivot_df.reindex(columns=range(n_bins), fill_value=0.0).fillna(0.0)

        # 3. Apply Gaussian Smoothing (along axis 1: time bins)
        if smoothing_sigma > 0:
            smoothed_values = gaussian_filter1d(pivot_df.values, sigma=smoothing_sigma, axis=1)
            data_to_norm = pd.DataFrame(smoothed_values, index=pivot_df.index, columns=pivot_df.columns)
        else:
            data_to_norm = pivot_df

        # 4. Row-wise Normalization
        row_maxes = data_to_norm.max(axis=1)
        stats_dict[f'{sex_key}_peaks'] = row_maxes.to_dict()

        # Avoid division by zero for silent categories
        norm_df = data_to_norm.div(row_maxes.replace(0, 1), axis=0)

        # 5. Plotting
        sns.heatmap(
            norm_df,
            ax=ax,
            cmap=colormap,
            cbar_kws={'label': 'Normalized Intensity'},
            linewidths=0.1,
            linecolor='#222222',
            vmin=0,
            vmax=1
        )

        ax.set_title(title, fontsize=14, pad=10)
        ax.set_ylabel("USV Category ID", fontsize=12)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

        # CLEAR AUTO-GENERATED LABEL: This prevents 'time_bin' from appearing under the top plot
        ax.set_xlabel('')

        # Explicitly turn off minor ticks
        ax.minorticks_off()

    # Global X-axis formatting (Only on the bottom plot)
    ax_female.set_xlabel("Session Time (minutes)", fontsize=12)
    ax_female.set_xticks(np.arange(len(bin_labels)) + 0.5)
    ax_female.set_xticklabels(bin_labels, rotation=45)

    plt.tight_layout()
    return fig, axes, stats_dict


def plot_estrous_ratio_scatter(
    ratio_dict: dict[str, list[float]],
    category_order: list[str],
    category_labels: list[str],
    scatter_colors: list[str],
    line_color: str,
    text_color: str,
    confidence_level: float = 0.99,
    use_log_scale: bool = False
) -> tuple[plt.Figure, plt.Axes, dict[str, dict[str, float]]]:
    """
    Description
    ----------
    Generates a jittered scatter plot of the male-to-female USV ratio across
    estrous stages.

    The function applies a standard logarithmic scale if requested, allowing
    for the visualization of large dynamic ranges in vocal behavior. It includes
    robust NaN filtering to ensure that statistical summaries (mean/SEM) are
    calculated only from valid sessions.
    ----------

    Parameters
    ----------
    ratio_dict (dict)
        Dictionary mapping estrous stage characters to lists of session ratios.
    category_order (list[str])
        The order of categories on the x-axis (e.g., ['p', 'e', 'm', 'd']).
    category_labels (list[str])
        The full names for the x-axis labels.
    scatter_colors (list[str])
        List of colors for each estrous stage.
    line_color (str)
        Color for the mean and error bars.
    text_color (str)
        Color for the statistical annotations.
    confidence_level (float), default 0.99
        Confidence level for CI bounds.
    use_log_scale (bool), default False
        If True, applies a standard base-10 logarithmic scale to the y-axis.
    ----------

    Returns
    ----------
    fig (plt.Figure), ax (plt.Axes), stats_dict (dict)
    """

    fig, ax = plt.subplots(figsize=(4, 3))
    stats_dict: dict[str, dict[str, float]] = {}

    mean_line_width = 0.03
    jitter_width = 0.15

    # Track global bounds for intelligent axis scaling
    global_min = float('inf')
    global_max = float('-inf')

    for idx, category in enumerate(category_order):
        # Extract and filter NaNs
        raw_values = np.array(ratio_dict.get(category, []))
        values = raw_values[~np.isnan(raw_values)]

        n = len(values)

        if n > 0:
            global_min = min(global_min, values.min())
            global_max = max(global_max, values.max())

        if n == 0:
            stats_dict[category] = {'n': 0, 'mean': float('nan'), 'sem': float('nan')}
            continue

        # Descriptive Statistics
        mean_val = float(np.mean(values))
        if n > 1:
            sem_val = float(sem(values))
            t_crit = t.ppf((1 + confidence_level) / 2, n - 1)
            ci_margin = t_crit * sem_val
            ci_lower, ci_upper = mean_val - ci_margin, mean_val + ci_margin
            stats_text = f"{mean_val:.2f} ± {sem_val:.2f}"
        else:
            sem_val, ci_lower, ci_upper = float('nan'), float('nan'), float('nan')
            stats_text = f"Mean: {mean_val:.2f}"

        stats_dict[category] = {
            'n': n, 'mean': mean_val, 'sem': sem_val,
            'ci_lower': ci_lower, 'ci_upper': ci_upper
        }

        # Jitter and Plot Points
        cloud_center_x = idx
        x_positions = np.random.uniform(cloud_center_x - jitter_width, cloud_center_x + jitter_width, size=n)

        ax.scatter(x_positions, values, color=scatter_colors[idx], alpha=0.4, s=15, zorder=5)

        # Plot Mean and SEM
        ax.hlines(y=mean_val, xmin=cloud_center_x - mean_line_width, xmax=cloud_center_x + mean_line_width,
                  color=line_color, linewidth=2.5, zorder=10)

        if not np.isnan(sem_val):
            ax.errorbar(cloud_center_x, mean_val, yerr=sem_val, fmt='none',
                        color=line_color, elinewidth=1.5, capsize=0, zorder=9)

        # Statistical Text Labels
        ax.text(cloud_center_x, -0.15, stats_text, transform=ax.get_xaxis_transform(),
                ha='center', va='top', fontsize=7, color=text_color, linespacing=1.3)

    # Axis Formatting
    ax.set_xticks(np.arange(len(category_order)))
    ax.set_xticklabels(category_labels, fontsize=14)
    ax.set_ylabel('Male to female USV ratio')
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.minorticks_off()

    if use_log_scale:
        ax.set_yscale('log')
        if global_min != float('inf'):
            # Provide breathing room at the bottom and top
            ax.set_ylim(bottom=global_min * 0.8, top=global_max * 1.5)
    else:
        if global_min != float('inf'):
            padding = (global_max - global_min) * 0.05
            ax.set_ylim(bottom=max(0, global_min - padding), top=global_max + padding)

    fig.tight_layout()
    return fig, ax, stats_dict

def plot_estrous_usv_rates(
    session_counts: dict[str, int],
    male_usv_counts: dict[str, int],
    female_usv_counts: dict[str, int],
    category_order: list[str],
    category_labels: list[str],
    male_color: str,
    female_color: str,
    text_color: str
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes], dict[str, dict[str, float]]]:
    """
    Description
    ----------
    This method generates a side-by-side bar chart visualizing the average number
    of USVs emitted per session, split by sex and categorized by the female's
    estrous stage.

    It computes the mean USV rate by dividing total vocalizations by the total
    number of sessions recorded for each estrous stage.
    ----------

    Parameters
    ----------
    session_counts (dict)
        A dictionary mapping estrous stage characters to total session counts.
    male_usv_counts (dict)
        A dictionary mapping estrous stage characters to total male USV counts.
    female_usv_counts (dict)
        A dictionary mapping estrous stage characters to total female USV counts.
    category_order (list[str])
        The specific order in which the categories should appear on the x-axis.
    category_labels (list[str])
        The formatted strings to display on the x-axis for each category.
    male_color (str)
        Hex code for the male bar color.
    female_color (str)
        Hex code for the female bar color.
    text_color (str)
        Hex code for the bar label annotations.
    ----------

    Returns
    ----------
    fig (plt.Figure)
        The matplotlib Figure object.
    axes (tuple)
        A tuple containing two matplotlib Axes objects (ax_female, ax_male).
    stats_dict (dict)
        A dictionary detailing 'male_rate' and 'female_rate' per estrous stage.
    ----------
    """

    male_rates = []
    female_rates = []
    stats_dict: dict[str, dict[str, float]] = {}

    for stage in category_order:
        s_count = session_counts.get(stage, 0)
        if s_count > 0:
            m_rate = male_usv_counts.get(stage, 0) / s_count
            f_rate = female_usv_counts.get(stage, 0) / s_count
        else:
            m_rate, f_rate = 0.0, 0.0

        male_rates.append(m_rate)
        female_rates.append(f_rate)
        stats_dict[stage] = {'male_rate': float(m_rate), 'female_rate': float(f_rate)}

    x_pos = np.arange(len(category_labels))
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5), sharey=False)

    # Female
    bars_female = axes[0].bar(x_pos, female_rates, color=female_color, alpha=0.8)
    axes[0].set_ylabel('Average USVs/Session', fontsize=14)
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(category_labels, fontsize=14)
    axes[0].tick_params(axis='y', labelsize=12)
    axes[0].bar_label(bars_female, fmt='%.1f', padding=3, color=text_color, fontsize=12)
    axes[0].grid(axis='y', linestyle='--', alpha=0.4)
    axes[0].spines[['top', 'right']].set_visible(False)
    axes[0].set_ylim(bottom=0)

    # Male
    bars_male = axes[1].bar(x_pos, male_rates, color=male_color, alpha=0.8)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(category_labels, fontsize=14)
    axes[1].tick_params(axis='y', labelsize=12)
    axes[1].bar_label(bars_male, fmt='%.1f', padding=3, color=text_color, fontsize=12)
    axes[1].grid(axis='y', linestyle='--', alpha=0.4)
    axes[1].spines[['top', 'right']].set_visible(False)
    axes[1].minorticks_off()

    fig.tight_layout(pad=2.0)
    return fig, (axes[0], axes[1]), stats_dict


def plot_estrous_stage_pie_chart(
    session_counts: dict[str, int],
    label_map: dict[str, str],
    slice_colors: list[str]
) -> tuple[plt.Figure, plt.Axes, dict[str, float]]:
    """
    Description
    ----------
    Generates a donut-style pie chart of recording sessions across estrous stages.
    Slices are forced into a specific biological order (P -> E -> M -> D) and
    plotted clockwise starting from the top.
    ----------

    Parameters
    ----------
    session_counts (dict)
        Mapping of estrous stage characters to session counts.
    label_map (dict)
        Mapping of characters to full strings (e.g., {'p': 'Proestrus'}).
    slice_colors (list[str])
        Hex colors matching the biological order ['p', 'e', 'm', 'd'].
    ----------

    Returns
    ----------
    fig, ax, stats_dict
    """

    # Force the biological order (DO NOT sort by frequency/count)
    category_order = ['p', 'e', 'm', 'd']

    # Filter for stages that actually have data to avoid empty slices
    plot_stages = [s for s in category_order if s in session_counts and session_counts[s] > 0]
    counts = [session_counts[s] for s in plot_stages]
    labels_full = [label_map.get(s, s) for s in plot_stages]

    # Map colors to the specific stages present in the plot
    # (Matches the color to the stage regardless of if some stages are missing)
    color_map = dict(zip(category_order, slice_colors))
    current_colors = [color_map[s] for s in plot_stages]

    total_sessions = sum(counts)
    stats_dict = {
        label_map.get(k, k): float((v / total_sessions) * 100) if total_sessions > 0 else 0.0
        for k, v in session_counts.items()
    }

    fig, ax = plt.subplots(figsize=(7, 7))

    # startangle=90 puts the first slice at the very top (12 o'clock)
    # counterclock=False ensures subsequent slices move to the right
    wedges, texts, autotexts = ax.pie(
        counts,
        labels=labels_full,
        autopct='%1.1f%%',
        startangle=90,
        counterclock=False,
        colors=current_colors,
        pctdistance=0.85,
        textprops={'fontsize': 12}
    )

    # Add center circle for donut effect
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    ax.add_artist(centre_circle)

    ax.axis('equal')
    fig.tight_layout()

    return fig, ax, stats_dict

def plot_category_prevalence_and_embedding(
    df_embedding: pls.DataFrame,
    male_color: str,
    female_color: str,
    unassigned_color: str,
    plot_type: str = 'density',
    boundary_color: str = '#00FF00',
    log_scale_bars: bool = False,
    grid_res: int = 300
):
    """
    Description
    ----------
    Generates a 4x2 grid of subplots visualizing the acoustic repertoire and spatial
    embedding of USVs broken down by assignment (Male, Female, Unassigned), plus a
    global summary row.

    The left column displays bar charts showing the count (and final proportion) of
    each USV category. The right column visualizes the 2D embedding space using either
    a density heatmap ('imshow' with white_inferno) OR a scatter plot.

    Global territorial boundaries are calculated using nearest-neighbor interpolation
    across ALL valid USVs and are overlaid prominently on top of every embedding plot.

    Parameters
    ----------
    df_embedding : pls.DataFrame
        The Polars DataFrame output from `extract_category_embedding_data`.
    male_color : str
        Hex color string used for the male bar chart and scatter plot.
    female_color : str
        Hex color string used for the female bar chart and scatter plot.
    unassigned_color : str
        Hex color string used for the unassigned bar chart and scatter plot.
    plot_type : str, default 'density'
        Visual style for the embedding space. Must be 'density' or 'scatter'.
    boundary_color : str, default '#00FF00'
        Hex color for the territorial boundary lines overlaid on the embedding.
    log_scale_bars : bool, default True
        If True, applies a base-10 logarithmic scale to the y-axis of the raw count bar charts.
    grid_res : int, default 300
        The resolution of the internal meshgrid. Higher values produce smoother
        global boundary lines and KDE maps, but increase computation time.

    Returns
    ----------
    fig : matplotlib.figure.Figure
        The generated 4x2 matplotlib figure.
    axes : numpy.ndarray
        A 2D array of the matplotlib axes objects.
    """

    if plot_type not in ['density', 'scatter']:
        error_msg = "plot_type must be either 'density' or 'scatter'."
        raise ValueError(error_msg)

    df_pd = df_embedding.to_pandas()

    # 1. Prepare global boundaries (griddata over all valid USVs)
    print("Computing global acoustic territorial boundaries...")
    dim1_all = df_pd['dim1'].to_numpy()
    dim2_all = df_pd['dim2'].to_numpy()
    cats_all = df_pd['category'].to_numpy()

    x_min, x_max = dim1_all.min(), dim1_all.max()
    y_min, y_max = dim2_all.min(), dim2_all.max()

    # Add 5% padding
    x_pad, y_pad = (x_max - x_min) * 0.05, (y_max - y_min) * 0.05
    x_min, x_max = x_min - x_pad, x_max + x_pad
    y_min, y_max = y_min - y_pad, y_max + y_pad

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_res),
        np.linspace(y_min, y_max, grid_res)
    )

    # Interpolate boundaries based on nearest neighbor category
    Z = griddata((dim1_all, dim2_all), cats_all, (xx, yy), method='nearest')
    unique_cats = np.unique(cats_all)

    # Build the custom white-inferno colormap
    base_cmap = plt.cm.get_cmap('inferno')
    cmap_colors = base_cmap(np.linspace(0, 1, 256))
    white = np.array([1, 1, 1, 1])
    cmap_colors[:25, :] = np.linspace(white, cmap_colors[25, :], 25)
    white_inferno = mcolors.ListedColormap(cmap_colors)

    # 3. Figure setup
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(12, 18), facecolor='#FFFFFF')

    plot_configs = [
        ('male', 'Male', male_color, axes[0, :]),
        ('female', 'Female', female_color, axes[1, :]),
        ('unassigned', 'Unassigned', unassigned_color, axes[2, :])
    ]

    ### Individual assignments
    for sex_key, title_prefix, color, row_axes in plot_configs:
        ax_bar, ax_emb = row_axes
        subset = df_pd[df_pd['sex'] == sex_key]

        # Column 1: Bar Chart
        if not subset.empty:
            cat_counts = subset['category'].value_counts().sort_index()
            ax_bar.bar(cat_counts.index.astype(str), cat_counts.values, color=color, edgecolor='#000000')

        ax_bar.set_title(f"{title_prefix} Category Prevalence", fontsize=12)
        ax_bar.set_ylabel("USV Count", fontsize=10)
        ax_bar.set_xlabel("Category ID", fontsize=10)
        ax_bar.grid(axis='y', linestyle='--', alpha=0.4)
        ax_bar.spines[['top', 'right']].set_visible(False)
        if log_scale_bars:
            ax_bar.set_yscale('log')

        # Column 2: Embedding space (density OR scatter)
        ax_emb.set_facecolor('#FFFFFF')
        ax_emb.grid(False)

        if len(subset) > 3:
            if plot_type == 'density':
                kde = gaussian_kde(np.vstack([subset['dim1'], subset['dim2']]))
                zz = kde(np.vstack([xx.flatten(), yy.flatten()])).reshape(xx.shape)
                zz_norm = (zz - zz.min()) / (zz.max() - zz.min() + 1e-10)

                ax_emb.imshow(
                    zz_norm, cmap=white_inferno, interpolation='bilinear', origin='lower',
                    extent=[x_min, x_max, y_min, y_max], aspect='auto', zorder=2
                )
            elif plot_type == 'scatter':
                ax_emb.scatter(
                    subset['dim1'], subset['dim2'], c=color, s=15,
                    alpha=0.7, edgecolors='none', zorder=2
                )

        # Draw global background boundaries ON TOP
        ax_emb.contour(
            xx, yy, Z, levels=np.arange(len(unique_cats)+1)-0.5,
            colors=boundary_color, linewidths=2.5, zorder=10
        )

        ax_emb.set_title(f"{title_prefix} Embedding Space ({plot_type.capitalize()})", fontsize=12)
        ax_emb.set_xlim(x_min, x_max)
        ax_emb.set_ylim(y_min, y_max)
        ax_emb.set_xticks([])
        ax_emb.set_yticks([])
        for spine in ax_emb.spines.values():
            spine.set_edgecolor('#000000')
            spine.set_linewidth(1.0)


    ### Global summary
    ax_sum_bar = axes[3, 0]
    ax_sum_emb = axes[3, 1]

    # Column 1: 100% Stacked Proportions Bar Chart
    # Group by category and sex to get counts, fill missing combos with 0
    cat_sex_counts = df_pd.groupby(['category', 'sex']).size().unstack(fill_value=0)

    # Ensure all columns exist to prevent KeyErrors if a category is missing entirely
    for col in ['male', 'female', 'unassigned']:
        if col not in cat_sex_counts.columns:
            cat_sex_counts[col] = 0

    # Calculate proportions (divide each row by its sum)
    cat_totals = cat_sex_counts.sum(axis=1)
    cat_sex_props = cat_sex_counts.div(cat_totals.replace(0, 1), axis=0) # replace 0 with 1 to avoid div by zero

    categories_str = cat_sex_props.index.astype(str)
    m_props = cat_sex_props['male'].values
    f_props = cat_sex_props['female'].values
    u_props = cat_sex_props['unassigned'].values

    ax_sum_bar.bar(categories_str, m_props, color=male_color, edgecolor='#000000')
    ax_sum_bar.bar(categories_str, f_props, bottom=m_props, color=female_color, edgecolor='#000000')
    ax_sum_bar.bar(categories_str, u_props, bottom=m_props + f_props, color=unassigned_color, edgecolor='#000000')

    ax_sum_bar.set_title("100% Proportions by Assignment", fontsize=12)
    ax_sum_bar.set_ylabel("Proportion", fontsize=10)
    ax_sum_bar.set_xlabel("Category ID", fontsize=10)
    ax_sum_bar.set_ylim(0, 1.0)
    ax_sum_bar.grid(axis='y', linestyle='--', alpha=0.4)
    ax_sum_bar.spines[['top', 'right']].set_visible(False)

    # Column 2: Global embedding
    ax_sum_emb.set_facecolor('#FFFFFF')
    ax_sum_emb.grid(False)

    if plot_type == 'density':
        # Single global density of ALL calls
        kde_global = gaussian_kde(np.vstack([dim1_all, dim2_all]))
        zz_global = kde_global(np.vstack([xx.flatten(), yy.flatten()])).reshape(xx.shape)
        zz_norm_global = (zz_global - zz_global.min()) / (zz_global.max() - zz_global.min() + 1e-10)

        ax_sum_emb.imshow(
            zz_norm_global, cmap=white_inferno, interpolation='bilinear', origin='lower',
            extent=[x_min, x_max, y_min, y_max], aspect='auto', zorder=2
        )
    elif plot_type == 'scatter':
        # Scatter all calls, colored by assignment
        for sex_key, color in [('male', male_color), ('female', female_color), ('unassigned', unassigned_color)]:
            subset = df_pd[df_pd['sex'] == sex_key]
            ax_sum_emb.scatter(
                subset['dim1'], subset['dim2'], c=color, s=15,
                alpha=0.7, edgecolors='none', zorder=2
            )

    # Global boundaries on top
    ax_sum_emb.contour(
        xx, yy, Z, levels=np.arange(len(unique_cats)+1)-0.5,
        colors=boundary_color, linewidths=2.5, zorder=10
    )

    ax_sum_emb.set_title(f"Global Summary Embedding Space ({plot_type.capitalize()})", fontsize=12)
    ax_sum_emb.set_xlim(x_min, x_max)
    ax_sum_emb.set_ylim(y_min, y_max)
    ax_sum_emb.set_xticks([])
    ax_sum_emb.set_yticks([])
    for spine in ax_sum_emb.spines.values():
        spine.set_edgecolor('#000000')
        spine.set_linewidth(1.0)

    plt.tight_layout()
    return fig, axes

def plot_category_global_fatigue_heatmap(
    global_usv_df: pls.DataFrame,
    smoothing_sigma: float,
    colormap: str
) -> tuple[plt.Figure, np.ndarray, dict[str, Any]]:
    """
    Description
    ----------
    Generates a two-panel vertical grid (Top: Male, Bottom: Female) of heatmaps
    visualizing the temporal fatigue of USV categories using 2-hour bins.

    The function bins the 24-hour cycle into 12 two-hour blocks, calculates
    the mean vocal rate per category across sessions, and truncates leading
    empty time blocks to focus on the active recording period. It applies 1D
    Gaussian smoothing and row-wise normalization ($0.0$ to $1.0$) to highlight
    fatigue trends regardless of absolute vocalization counts.
    ----------

    Parameters
    ----------
    global_usv_df : pls.DataFrame
        Consolidated Polars DataFrame containing 'session_id', 'sex', 'category',
        'hour', and 'len' (vocal count).
    smoothing_sigma : float
        Standard deviation for the Gaussian kernel applied along the 2-hour
        time bins (X-axis).
    colormap : str
        Matplotlib colormap used to represent normalized vocal intensity.
    ----------

    Returns
    ----------
    fig : plt.Figure
        The matplotlib Figure object containing the two cleaned heatmap panels.
    axes : np.ndarray
        A 1D array containing the two Axes objects (ax_male, ax_female).
    stats_dict : dict
        Contains the processed, truncated, and smoothed pivot tables for
        both sexes.
    ----------
    """

    # 1. 2-Hour Binning and Aggregation
    # Create 2-hour blocks: 0-2, 2-4 ... 22-24
    binned_data = global_usv_df.with_columns(
        (pls.col("hour") // 2).alias("bi_hour")
    ).group_by(["session_id", "sex", "category", "bi_hour"]).len()

    df_pd = binned_data.to_pandas()

    # 2. Determine global start point (first non-zero hour bin)
    # We find the first bin that has any vocalization across the entire dataset
    global_pivot = df_pd.pivot_table(
        index='category',
        columns='bi_hour',
        values='len',
        aggfunc='mean'
    ).reindex(columns=range(12), fill_value=0.0)

    first_active_bin = int((global_pivot.sum(axis=0) > 0).idxmax())

    # Prepare figure
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(14, 10), sharex=True)
    stats_dict = {}
    sexes = [('male', axes[0]), ('female', axes[1])]

    for i, (sex_key, ax) in enumerate(sexes):
        subset = df_pd[df_pd['sex'] == sex_key]

        # Pivot and pad to ensure all 12 bins are represented
        pivot_df = subset.pivot_table(
            index='category',
            columns='bi_hour',
            values='len',
            aggfunc='mean'
        ).reindex(columns=range(12), fill_value=0.0).fillna(0.0)

        # Truncate data prior to the first non-zero cell found globally
        pivot_df = pivot_df.iloc[:, first_active_bin:]

        # Temporal Smoothing
        if smoothing_sigma > 0:
            smoothed_vals = gaussian_filter1d(pivot_df.values, sigma=smoothing_sigma, axis=1)
            data_to_norm = pd.DataFrame(smoothed_vals, index=pivot_df.index, columns=pivot_df.columns)
        else:
            data_to_norm = pivot_df

        # Row-Wise Normalization
        row_max = data_to_norm.max(axis=1)
        norm_df = data_to_norm.div(row_max.replace(0, 1), axis=0)

        # Plotting
        sns.heatmap(
            norm_df,
            ax=ax,
            cmap=colormap,
            cbar_kws={'label': 'Normalized Intensity'},
            vmin=0,
            vmax=1
        )

        # Cleanup: Remove minorticks and specific labels
        ax.minorticks_off()
        ax.set_title(f"Global {sex_key.capitalize()} Fatigue Heatmap (2h Bins)", fontsize=14)
        ax.set_ylabel("USV Category ID", fontsize=12)

        # Remove X-label from the top subplot specifically
        if i == 0:
            ax.set_xlabel('')
        else:
            # Format X-axis labels for the bottom subplot (e.g., 12-14, 14-16)
            active_bins = range(first_active_bin, 12)
            bin_labels = [f"{b*2}-{(b+1)*2}" for b in active_bins]
            ax.set_xticklabels(bin_labels, rotation=0)
            ax.set_xlabel("Hour of Day (2h Blocks)", fontsize=12)

        stats_dict[sex_key] = norm_df

    plt.tight_layout()
    return fig, axes, stats_dict

def plot_category_estrous_rates_grid(
    estrous_data: dict[int, dict[str, Any]],
    valid_stages: list[str],
    male_color: str,
    female_color: str
) -> tuple[plt.Figure, np.ndarray, dict[str, Any]]:
    """
    Description
    ----------
    Generates a facet grid of subplots visualizing average USV rates across
    full estrous stage names using dual Y-axes to account for sexual dimorphism
    in vocalization volume.

    Each subplot utilizes a twin axis system: the left Y-axis (Male) and
    right Y-axis (Female) scale independently to their respective local maxima.
    This allows for a direct comparison of the "behavioral shape" across
    estrous stages despite significant differences in absolute vocalization
    counts. Category IDs are displayed as integers, and X-axis labels use
    full biological stage names.
    ----------

    Parameters
    ----------
    estrous_data : dict
        A nested dictionary keyed by category ID containing session counts
        and raw USV counts per stage and sex.
    valid_stages : list[str]
        A list of characters representing the estrous stages in biological order.
    male_color : str
        Hex color code for the male vocal rate bars (left axis).
    female_color : str
        Hex color code for the female vocal rate bars (right axis).
    ----------

    Returns
    ----------
    fig : plt.Figure
        The matplotlib Figure object containing the facet grid with dual Y-axes.
    axes : np.ndarray
        An array of the primary (left) Axes objects.
    stats_dict : dict
        A dictionary mapping category IDs to their respective mean rates
        for males and females across all stages.
    ----------
    """

    stage_label_map = {
        'p': 'Proestrus',
        'e': 'Estrus',
        'm': 'Metestrus',
        'd': 'Diestrus'
    }

    categories = sorted(estrous_data.keys())
    n_cats = len(categories)
    cols = 3
    rows = int(np.ceil(n_cats / cols))

    # Label configuration
    full_stage_names = [stage_label_map.get(s.lower(), s) for s in valid_stages]
    x_indices = np.arange(len(full_stage_names))
    bar_width = 0.35

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4.5))
    axes_flat = axes.flatten()
    stats_dict = {}

    for i, cat in enumerate(categories):
        ax_male = axes_flat[i]
        # Create the twin axis for females
        ax_female = ax_male.twinx()

        m_rates = []
        f_rates = []

        for stage in valid_stages:
            s_count = estrous_data[cat]['session_counts'].get(stage, 0)
            m_r = (estrous_data[cat]['male_usv_counts'].get(stage, 0) / s_count) if s_count > 0 else 0.0
            f_r = (estrous_data[cat]['female_usv_counts'].get(stage, 0) / s_count) if s_count > 0 else 0.0
            m_rates.append(m_r)
            f_rates.append(f_r)

        # Plot male bars (shifted left)
        ax_male.bar(x_indices - bar_width/2, m_rates, width=bar_width,
                    color=male_color, label='Male', alpha=0.8)

        # Plot female bars (shifted right)
        ax_female.bar(x_indices + bar_width/2, f_rates, width=bar_width,
                      color=female_color, label='Female', alpha=0.8)

        ax_male.set_title(f"Category {int(cat)}", fontsize=15, fontweight='bold', pad=15)

        # Left axis (male)
        m_max = max(m_rates) if max(m_rates) > 0 else 10
        ax_male.set_ylim(0, m_max * 1.3)
        ax_male.set_ylabel("Male USVs/Session", color=male_color, fontsize=10, fontweight='bold')
        ax_male.tick_params(axis='y', labelcolor=male_color)

        # Right axis (female)
        f_max = max(f_rates) if max(f_rates) > 0 else 1
        ax_female.set_ylim(0, f_max * 1.3)
        ax_female.set_ylabel("Female USVs/Session", color=female_color, fontsize=10, fontweight='bold')
        ax_female.tick_params(axis='y', labelcolor=female_color)

        # X-Axis formatting
        ax_male.set_xticks(x_indices)
        ax_male.set_xticklabels(full_stage_names, fontsize=10)

        # Clean up
        ax_male.minorticks_off()
        ax_female.minorticks_off()
        ax_male.grid(axis='y', linestyle='--', alpha=0.3)

        stats_dict[cat] = {
            'stages': full_stage_names,
            'male_rates': m_rates,
            'female_rates': f_rates
        }

    # Hide unused subplots
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis('off')

    plt.tight_layout()
    return fig, axes, stats_dict


def plot_category_estrous_ratio_grid(
    estrous_data: dict[int, dict[str, Any]],
    valid_stages: list[str],
    scatter_colors: list[str]
) -> tuple[tuple[plt.Figure, plt.Figure], tuple[np.ndarray, np.ndarray], dict[str, Any]]:
    """
    Description
    ----------
    Generates two distinct figure panels visualizing Male-to-Female USV ratios
    across estrous stages.

    Figure 1: A facet grid where each subplot is a USV Category, showing
    ratios across the 4 biological stages.
    Figure 2: A 2x2 facet grid where each subplot is an Estrous Stage,
    showing the ratios for all USV Categories side-by-side.

    Both figures utilize a log-scale Y-axis, stage-specific point coloring,
    and a high-contrast black statistical overlay (Mean + SEM). Sample size
    labels are removed for clarity.
    ----------

    Parameters
    ----------
    estrous_data : dict
        A nested dictionary keyed by category ID containing lists of
        session-wise ratios per stage.
    valid_stages : list[str]
        A list of characters representing the estrous stages in biological order.
    scatter_colors : list[str]
        A list of 4 hex color codes corresponding to the stages in valid_stages.
    ----------

    Returns
    ----------
    figs : tuple[plt.Figure, plt.Figure]
        A tuple containing (fig_categories, fig_stages).
    axes : tuple[np.ndarray, np.ndarray]
        A tuple containing the axes arrays for both figures.
    stats_dict : dict
        A nested dictionary containing Mean and SEM for every category-stage pair.
    ----------
    """

    stage_label_map = {
        'p': 'Proestrus',
        'e': 'Estrus',
        'm': 'Metestrus',
        'd': 'Diestrus'
    }

    categories = sorted(estrous_data.keys())
    full_stage_names = [stage_label_map.get(s.lower(), s) for s in valid_stages]
    mean_line_width = 0.2
    stats_dict = {}

    # Figure 1: Category facets
    n_cats = len(categories)
    cols = 3
    rows = int(np.ceil(n_cats / cols))
    fig_cats, axes_cats = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    axes_cats_flat = axes_cats.flatten()

    for i, cat in enumerate(categories):
        ax = axes_cats_flat[i]
        cat_stats = {}

        for idx, stage in enumerate(valid_stages):
            ratios = np.array(estrous_data[cat]['male_female_ratios'].get(stage, []))
            ratios = ratios[np.isfinite(ratios)]
            n = len(ratios)

            if n > 0:
                # Jittered scatter
                rng = np.random.default_rng()
                x_jit = rng.normal(idx, 0.08, size=n)
                ax.scatter(x_jit, ratios, color=scatter_colors[idx], alpha=0.5, s=20, zorder=5)

                # Statistics
                m = np.mean(ratios)
                s = sem(ratios) if n > 1 else 0

                # Black statistical overlay
                ax.hlines(y=m, xmin=idx - mean_line_width, xmax=idx + mean_line_width,
                          color='#000000', linewidth=2.0, zorder=10)
                ax.errorbar(idx, m, yerr=s, fmt='none', color='#000000',
                            elinewidth=0.5, capsize=0, zorder=9)

                cat_stats[stage] = {'mean': float(m), 'sem': float(s), 'n': n}

        ax.set_yscale('log')
        ax.axhline(1.0, color='#202020', linestyle='--', alpha=0.3, linewidth=1)
        ax.set_xticks(range(len(valid_stages)))
        ax.set_xticklabels(full_stage_names, fontsize=10)
        ax.set_title(f"Category {int(cat)} USV ratio", fontsize=13, fontweight='bold', pad=12)
        ax.set_ylabel("Male to Female Ratio", fontsize=11)
        ax.minorticks_off()
        ax.grid(axis='y', linestyle=':', alpha=0.4)
        stats_dict[cat] = cat_stats

    for j in range(i + 1, len(axes_cats_flat)):
        axes_cats_flat[j].axis('off')
    fig_cats.tight_layout()

    # Figure 2: Eestus stage facets
    fig_stages, axes_stages = plt.subplots(2, 2, figsize=(12, 12), sharey=True)
    axes_stages_flat = axes_stages.flatten()

    for idx, stage_char in enumerate(valid_stages):
        ax = axes_stages_flat[idx]
        stage_name = stage_label_map.get(stage_char.lower(), stage_char)

        for cat_idx, cat_id in enumerate(categories):
            ratios = np.array(estrous_data[cat_id]['male_female_ratios'].get(stage_char, []))
            ratios = ratios[np.isfinite(ratios)]
            n = len(ratios)

            if n > 0:
                # Jittered scatter
                rng = np.random.default_rng()
                x_jit = rng.normal(cat_idx, 0.08, size=n)
                ax.scatter(x_jit, ratios, color=scatter_colors[idx], alpha=0.5, s=20, zorder=5)

                # Use pre-calculated stats from Figure 1 loop
                m = stats_dict[cat_id][stage_char]['mean']
                s = stats_dict[cat_id][stage_char]['sem']

                # Black Statistical Overlay
                ax.hlines(y=m, xmin=cat_idx - mean_line_width, xmax=cat_idx + mean_line_width,
                          color='#000000', linewidth=2.0, zorder=10)
                ax.errorbar(cat_idx, m, yerr=s, fmt='none', color='#000000',
                            elinewidth=1.5, capsize=0, zorder=9)

        ax.set_yscale('log')
        ax.axhline(1.0, color='#202020', linestyle='--', alpha=0.3, linewidth=1)
        ax.set_title(f"{stage_name} Stage: All Categories", fontsize=14, fontweight='bold', pad=15)
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels([f"Cat {int(c)}" for c in categories], fontsize=10)

        if idx % 2 == 0:
            ax.set_ylabel("Male to Female Ratio", fontsize=11)

        ax.minorticks_off()
        ax.grid(axis='y', linestyle=':', alpha=0.4)

    fig_stages.tight_layout()

    return (fig_cats, fig_stages), (axes_cats, axes_stages), stats_dict


def plot_category_polar_kde_grid(
    global_behavior_metrics: dict[Any, Any],
    sex_key: str,
    max_distance: float,
    threshold: int,
    colormap: str
) -> tuple[plt.Figure, np.ndarray, dict[str, Any]]:
    """
    Description
    ----------
    Generates a grid of small-multiple half-circle polar plots showing the
    Occupancy-Normalized Likelihood of vocalizing at specific spatial
    locations for a specific sex, broken down by USV category.

    The function dynamically selects the correct behavioral keys (e.g.,
    'mf_angle' for males and 'fm_angle' for females) and performs
    occupancy-normalization by dividing USV density by the background
    spatial probability of the selected sex. It includes a guardrail to
    prevent crashes if no categories are found and a sparse data filter
    to handle rare vocalization types. Heat scaling is unified across the
    grid using 98th percentile clipping.

    Parameters
    ----------
    global_behavior_metrics : dict
        Nested dictionary containing 'all_frames' tracking data and
        per-category USV coordinates.
    sex_key : str
        The target sex for plotting. Must be either 'male' or 'female'.
    max_distance : float
        The maximum radial distance (cm) to include in the polar plots.
    threshold : int
        The minimum number of USV points required to generate a valid density map.
    colormap : str
        The matplotlib colormap for spatial density.

    Returns
    ----------
    fig : plt.Figure
        The matplotlib Figure object containing the polar grid.
    axes : np.ndarray
        An array of polar Axes objects (one per category).
    stats_dict : dict
        Contains the global_vmax scaling factor and point counts per category.
    ----------
    """

    # Parameter validation
    valid_sexes = ['male', 'female']
    if sex_key.lower() not in valid_sexes:
        msg = f"sex_key must be one of {valid_sexes}"
        raise ValueError(msg)

    angle_key = 'mf_angle' if sex_key.lower() == 'male' else 'fm_angle'

    # Identify categories
    categories = sorted([c for c in global_behavior_metrics.keys() if c != 'all_frames'])
    n_cats = len(categories)

    if n_cats == 0:
        msg = "No USV categories found in the global_behavior_metrics dictionary."
        raise ValueError(msg)

    cols = 3
    rows = int(np.ceil(n_cats / cols))

    # Background occupancy
    all_dist = np.array(global_behavior_metrics['all_frames']['distance'])
    all_angle_rad = np.deg2rad(np.abs(global_behavior_metrics['all_frames'][angle_key]))
    valid_all = ~np.isnan(all_dist) & ~np.isnan(all_angle_rad) & (all_dist <= max_distance)

    if valid_all.sum() < 10:
        msg = f"Insufficient background tracking data for {sex_key} to compute occupancy."
        raise ValueError(msg)

    kde_all = gaussian_kde(np.vstack([all_angle_rad[valid_all], all_dist[valid_all]]))

    n_grid = 100
    ag, rg = np.linspace(0, np.pi, n_grid), np.linspace(0, max_distance, n_grid)
    mesh = np.stack(np.meshgrid(ag, rg), axis=0)
    dens_all = kde_all(mesh.reshape(2, -1)).reshape(n_grid, n_grid)

    category_densities = {}
    all_norm_values = []

    # Process categories
    for cat in categories:
        u_dist = np.array(global_behavior_metrics[cat][sex_key]['distance'])
        u_angle_rad = np.deg2rad(np.abs(global_behavior_metrics[cat][sex_key][angle_key]))

        valid_u = ~np.isnan(u_dist) & ~np.isnan(u_angle_rad) & (u_dist <= max_distance)

        if valid_u.sum() >= threshold:
            try:
                kde_u = gaussian_kde(np.vstack([u_angle_rad[valid_u], u_dist[valid_u]]))
                dens_u = kde_u(mesh.reshape(2, -1)).reshape(n_grid, n_grid)

                norm_dens = np.zeros_like(dens_u)
                occ_mask = dens_all > (dens_all.max() * 0.01)
                norm_dens[occ_mask] = dens_u[occ_mask] / dens_all[occ_mask]

                category_densities[cat] = norm_dens
                all_norm_values.extend(norm_dens[occ_mask].flatten())
            except np.linalg.LinAlgError:
                continue

    # Unified heat scaling
    global_vmax = np.percentile(all_norm_values, 98) if all_norm_values else 1.0

    # Plotting
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4),
                             subplot_kw={'projection': 'polar'})

    axes_flat = axes.flatten() if n_cats > 1 else [axes]
    stats_dict = {'global_vmax': global_vmax, 'sex_plotted': sex_key}

    for i, cat in enumerate(categories):
        ax = axes_flat[i]

        # Half-circle formatting
        ax.set_thetamin(0)
        ax.set_thetamax(180)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)

        ax.set_rticks([0, 5, 10, 15, 20])
        ax.set_rlabel_position(90)

        if cat in category_densities:
            ax.contourf(ag, rg, np.clip(category_densities[cat], 0, global_vmax),
                        cmap=colormap, levels=50)

            try:
                cat_label = int(float(cat))
            except (ValueError, TypeError):
                cat_label = cat
            ax.set_title(f"Category {cat_label} {sex_key.capitalize()}", fontsize=11, pad=15)
        else:
            ax.text(np.pi/2, max_distance/2, "Insufficient\nData", ha='center', va='center')
            ax.set_title(f"Category {cat}", fontsize=11, pad=15)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis('off')

    plt.tight_layout()
    return fig, axes, stats_dict

def plot_estrous_category_kde_grid(
    usv_pls: pls.DataFrame,
    bg_pls: pls.DataFrame,
    sex_key: str,
    valid_stages: list[str],
    stage_label_map: dict[str, str],
    max_distance: float,
    occupancy_threshold: float,
    colormap: mcolors.Colormap | str,
    threshold: int = 30,
    max_kde_points: int = 50000
) -> tuple[plt.Figure, np.ndarray, dict[str, Any]]:
    """
    Description
    ----------
    Generates a 2D facet grid of occupancy-normalised half-circle polar KDE plots
    showing spatial vocalization likelihood broken down simultaneously by USV
    acoustic category and female estrous stage.

    Rows correspond to USV categories (sorted numerically) and columns correspond
    to estrous stages in the order supplied by 'valid_stages'. Each subplot uses
    the same shared background occupancy KDE (computed once from 'bg_pls') and a
    unified colour scale (98th percentile of all valid normalised densities) so
    that intensity is directly comparable across the entire grid.

    Subplots with fewer USVs than 'threshold' display "N/A" instead of a KDE to
    avoid fitting degenerate distributions on sparse data.

    The 'estrous_stage' column must already be present in 'usv_pls'. Add it with:
        usv_pls = usv_pls.with_columns(
            pls.col('experiment_code').str.slice(-1).alias('estrous_stage')
        )
    ----------

    Parameters
    ----------
    usv_pls (pls.DataFrame)
        The master USV DataFrame returned by build_master_usv_dataframe, with an
        'estrous_stage' column already attached.
    bg_pls (pls.DataFrame)
        The background occupancy DataFrame returned by build_master_usv_dataframe,
        containing 'distance', 'mf_angle', and 'fm_angle' columns.
    sex_key (str)
        The target sex for plotting. Must be 'male' or 'female'.
    valid_stages (list[str])
        The estrous stage characters in the order they should appear as columns
        (e.g., ['p', 'e', 'm', 'd']).
    stage_label_map (dict[str, str])
        Mapping from stage characters to full display names used as column titles
        (e.g., {'p': 'Proestrus', 'e': 'Estrus', 'm': 'Metestrus', 'd': 'Diestrus'}).
    max_distance (float)
        Maximum radial distance (cm) to include in the KDE and plot axes.
    occupancy_threshold (float)
        Relative fraction of the background KDE peak below which occupancy is
        considered too low to normalise (e.g., 0.01 = 1 % of peak). Cells below
        this threshold are masked to zero to prevent division-by-noise artefacts.
    colormap (mcolors.Colormap | str)
        Matplotlib colormap for the filled contours.
    threshold (int), default 30
        Minimum number of USV points required per (category, stage) cell to
        attempt KDE fitting. Cells below this are displayed as "N/A".
    max_kde_points (int), default 50000
        Maximum number of background frame points used for KDE calculation.
        Data exceeding this is randomly subsampled for performance.
    ----------

    Returns
    ----------
    fig (plt.Figure)
        The matplotlib Figure object containing the full grid.
    axes (np.ndarray)
        2D array of polar Axes objects with shape (n_categories, n_stages).
    stats_dict (dict)
        Contains 'global_vmax', 'sex_plotted', and 'n_points' — a nested dict
        mapping (category, stage) pairs to the number of USV points used.
    ----------
    """

    valid_sexes = ['male', 'female']
    if sex_key.lower() not in valid_sexes:
        msg = f"sex_key must be one of {valid_sexes}"
        raise ValueError(msg)

    if 'estrous_stage' not in usv_pls.columns:
        msg = (
            "usv_pls must contain an 'estrous_stage' column. "
            "Add it with: usv_pls = usv_pls.with_columns("
            "pls.col('experiment_code').str.slice(-1).alias('estrous_stage'))"
        )
        raise ValueError(msg)

    angle_key = 'mf_angle' if sex_key.lower() == 'male' else 'fm_angle'

    categories = sorted(usv_pls['category'].drop_nulls().unique().to_list())
    n_cats = len(categories)
    n_stages = len(valid_stages)

    if n_cats == 0:
        msg = "No USV categories found in usv_pls."
        raise ValueError(msg)

    # Compute background KDE once — shared across all subplots
    bg_dist = bg_pls['distance'].drop_nulls().to_numpy()
    bg_angle_rad = np.deg2rad(np.abs(bg_pls[angle_key].drop_nulls().to_numpy()))

    valid_bg = ~np.isnan(bg_dist) & ~np.isnan(bg_angle_rad) & (bg_dist <= max_distance)
    if valid_bg.sum() < 10:
        msg = f"Insufficient background tracking data for sex_key='{sex_key}'."
        raise ValueError(msg)

    bg_dist_f = bg_dist[valid_bg]
    bg_angle_f = bg_angle_rad[valid_bg]

    if len(bg_dist_f) > max_kde_points:
        rng = np.random.default_rng()
        idx = rng.choice(len(bg_dist_f), max_kde_points, replace=False)
        bg_dist_f, bg_angle_f = bg_dist_f[idx], bg_angle_f[idx]

    kde_bg = gaussian_kde(np.vstack([bg_angle_f, bg_dist_f]))

    n_grid = 100
    ag = np.linspace(0, np.pi, n_grid)
    rg = np.linspace(0, max_distance, n_grid)
    mesh = np.stack(np.meshgrid(ag, rg), axis=0)
    dens_bg = kde_bg(mesh.reshape(2, -1)).reshape(n_grid, n_grid)
    occ_mask = dens_bg > (dens_bg.max() * occupancy_threshold)

    # Pre-compute all (category, stage) normalised densities
    cell_densities: dict[tuple, np.ndarray | None] = {}
    n_points: dict[tuple, int] = {}
    all_norm_values: list[float] = []

    for cat in categories:
        for stage in valid_stages:
            subset = (
                usv_pls
                .filter(pls.col('category') == cat)
                .filter(pls.col('estrous_stage') == stage)
                .filter(pls.col('sex') == sex_key)
                .select(['distance', angle_key])
                .drop_nulls()
            )

            u_dist = subset['distance'].to_numpy()
            u_angle_rad = np.deg2rad(np.abs(subset[angle_key].to_numpy()))
            valid_u = ~np.isnan(u_dist) & ~np.isnan(u_angle_rad) & (u_dist <= max_distance)
            n_valid = int(valid_u.sum())
            n_points[(cat, stage)] = n_valid

            if n_valid < threshold:
                cell_densities[(cat, stage)] = None
                continue

            try:
                kde_u = gaussian_kde(np.vstack([u_angle_rad[valid_u], u_dist[valid_u]]))
                dens_u = kde_u(mesh.reshape(2, -1)).reshape(n_grid, n_grid)

                norm_dens = np.zeros_like(dens_u)
                norm_dens[occ_mask] = dens_u[occ_mask] / dens_bg[occ_mask]

                cell_densities[(cat, stage)] = norm_dens
                all_norm_values.extend(norm_dens[occ_mask].flatten())

            except np.linalg.LinAlgError:
                cell_densities[(cat, stage)] = None

    global_vmax = np.percentile(all_norm_values, 98) if all_norm_values else 1.0

    # Build figure grid
    fig, axes = plt.subplots(
        n_cats, n_stages,
        figsize=(n_stages * 4, n_cats * 4),
        subplot_kw={'projection': 'polar'}
    )

    # Normalise axes to 2D array regardless of grid shape
    if n_cats == 1 and n_stages == 1:
        axes = np.array([[axes]])
    elif n_cats == 1:
        axes = axes[np.newaxis, :]
    elif n_stages == 1:
        axes = axes[:, np.newaxis]

    for r, cat in enumerate(categories):
        for c, stage in enumerate(valid_stages):
            ax = axes[r, c]

            ax.set_thetamin(0)
            ax.set_thetamax(180)
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            ax.set_rticks(np.linspace(0, max_distance, 5))
            ax.set_rlabel_position(90)
            ax.tick_params(axis='both', labelsize=7)

            norm_dens = cell_densities.get((cat, stage))
            if norm_dens is not None:
                ax.contourf(ag, rg, np.clip(norm_dens, 0, global_vmax), cmap=colormap, levels=50)
            else:
                ax.text(np.pi / 2, max_distance / 2, "N/A", ha='center', va='center', fontsize=9)

            if r == 0:
                ax.set_title(stage_label_map.get(stage, stage), fontsize=12, pad=15)

            if c == 0:
                ax.set_ylabel(f"Cat {int(cat)}", fontsize=10, labelpad=30)

    fig.suptitle(
        f'{sex_key.capitalize()} Spatial Likelihood: Category × Estrous Stage',
        fontsize=14, y=1.01
    )
    fig.tight_layout()

    return fig, axes, {'global_vmax': global_vmax, 'sex_plotted': sex_key, 'n_points': n_points}
