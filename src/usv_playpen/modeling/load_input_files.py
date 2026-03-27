"""
@author: bartulem
Module for loading raw data and orchestrating the data preparation pipeline for GLM
and regression analysis.

Key Capabilities:
1.  Data ingestion: Loading 3D behavioral features (CSV), track metadata (H5),
    and USV summaries.
2.  Epoch sampling: Identifying USV and No-USV event times using dynamic GMM
    clustering (bout mode), individual syllable onsets, or state-based sampling.
3.  Category classification: Organizing USVs into target vs. other categories
    to enable one-vs-rest syntax models.
4.  Bout parameter extraction: Calculating continuous bout properties, including
    duration, syllable count, and mask complexity (complexity of vocal patterns).
5.  Signal processing: Generating Gaussian-smoothed continuous vocal density
    traces and binarized activity traces for mice and categories.
6.  Data cleaning: Applying category-based noise filtering and clean-history
    constraints to ensure biological accuracy.
"""

import h5py
import numpy as np
import os
import pathlib
import pickle
import polars as pls
from astropy.convolution import convolve
from astropy.convolution import Gaussian1DKernel


def load_behavioral_feature_data(behavior_file_paths: list = None,
                                 csv_sep: str = ',') -> tuple:
    """
    Loads behavior data from a 3D behavioral features .csv file.

    Parameters
    ----------
    behavior_file_paths : list
        Paths to the sessions containing behavioral feature data.
    csv_sep : str, optional
        Separator used in the .csv file.

    Returns
    -------
    behavior_data : tuple (dict. dict, dict)
        Behavior, camera frame rate and track name data (keys are file names and values pd.DataFrames, float, list).
    """

    beh_feature_data_dict = {}
    camera_fr_dict = {}
    mouse_track_names_dict = {}
    for behavior_file_path in behavior_file_paths:
        features_csv_file_path = next(pathlib.Path(f"{behavior_file_path}{os.sep}video{os.sep}").glob(f"**{os.sep}*_points3d_translated_rotated_metric_behavioral_features.csv"), None)
        track_file_path = next(pathlib.Path(f"{behavior_file_path}{os.sep}video{os.sep}").glob(f"**{os.sep}[!speaker]*_points3d_translated_rotated_metric.h5"), None)
        with h5py.File(name=track_file_path, mode='r') as h5_file_mouse_obj:
            camera_fr_dict[behavior_file_path.split(os.sep)[-1]] = float(h5_file_mouse_obj['recording_frame_rate'][()])
            mouse_track_names_dict[behavior_file_path.split(os.sep)[-1]] = [item.decode('utf-8') for item in list(h5_file_mouse_obj['track_names'])]
        beh_feature_data_dict[behavior_file_path.split(os.sep)[-1]] = pls.read_csv(source=features_csv_file_path, separator=csv_sep)

    return beh_feature_data_dict, camera_fr_dict, mouse_track_names_dict

def _get_clean_tiled_epochs(usv_starts_all: np.ndarray,
                            usv_stops_all: np.ndarray,
                            filter_history: float,
                            session_duration_sec: float) -> np.ndarray:
    """
    Finds all valid 'clean' (no-USV) epochs using the "forbidden zone" tiling method.

    This method works by:
    1. Defining a "forbidden zone" around every USV as [start_time, stop_time + filter_history].
    2. Merging all overlapping forbidden zones.
    3. Finding the "clean" gaps *between* these merged zones.
    4. Tiling these clean gaps with non-overlapping windows of length 'filter_history'.

    Parameters
    ----------
    usv_starts_all : np.ndarray
        Array of all USV start times (in seconds), including uncategorized.
    usv_stops_all : np.ndarray
        Array of all USV stop times (in seconds), including uncategorized.
    filter_history : float
        The duration (in seconds) of the pre-event window. This defines the
        minimum size of a "clean" gap and the size of the non-overlapping tiles.
    session_duration_sec : float
        The total duration of the session in seconds.

    Returns
    -------
    np.ndarray
        An array of valid, non-overlapping "no-USV" event times (window end times)
        in seconds.
    """

    # Define forbidden intervals: (start, stop + filter_history)
    forbidden_starts = usv_starts_all
    forbidden_ends = usv_stops_all + filter_history

    # Merge overlapping forbidden intervals
    if forbidden_starts.size > 0:
        indices = np.argsort(forbidden_starts)
        sorted_starts = forbidden_starts[indices]
        sorted_ends = forbidden_ends[indices]

        merged_starts = [sorted_starts[0]]
        merged_ends = [sorted_ends[0]]

        for i in range(1, sorted_starts.size):
            if sorted_starts[i] <= merged_ends[-1]:
                # Overlap: extend the end of the last merged interval
                merged_ends[-1] = max(merged_ends[-1], sorted_ends[i])
            else:
                # No overlap: start a new interval
                merged_starts.append(sorted_starts[i])
                merged_ends.append(sorted_ends[i])

        merged_starts = np.array(merged_starts)
        merged_ends = np.array(merged_ends)
    else:
        merged_starts = np.array([])
        merged_ends = np.array([])

    # Invert to get "clean" zones (cannot sample before filter_history)
    session_start = filter_history
    clean_starts = []
    clean_ends = []

    if merged_starts.size == 0:
        # No USVs at all, entire session is clean
        clean_starts.append(session_start)
        clean_ends.append(session_duration_sec)
    else:
        # First clean zone: from session_start to the first forbidden start
        if merged_starts[0] > session_start:
            clean_starts.append(session_start)
            clean_ends.append(merged_starts[0])

        # Middle clean zones: gaps between forbidden zones
        for i in range(merged_ends.size - 1):
            clean_starts.append(merged_ends[i])
            clean_ends.append(merged_starts[i + 1])

        # Last clean zone: from last forbidden end to session_end
        if merged_ends[-1] < session_duration_sec:
            clean_starts.append(merged_ends[-1])
            clean_ends.append(session_duration_sec)

    # "Tile" the clean zones and get all valid t's (in seconds)
    all_valid_t = []
    for start, end in zip(clean_starts, clean_ends):
        duration = end - start
        if duration >= filter_history:
            # Use np.arange to find all possible end points (in seconds)
            possible_ts = np.arange(start + filter_history, end + 1e-9, filter_history)
            all_valid_t.extend(possible_ts)

    return np.array(all_valid_t)

def _generate_vocal_trace(event_starts: np.ndarray,
                          event_stops: np.ndarray,
                          duration_frames: int,
                          fps: float,
                          smooth_sd: float = None) -> np.ndarray:
    """
    Core utility to convert USV timestamps into a continuous temporal trace.

    This function handles the conversion of start/stop times into a binary
    occupancy array and optionally applies Gaussian smoothing.

    Parameters
    ----------
    event_starts : np.ndarray
        Array of start times in seconds.
    event_stops : np.ndarray
        Array of stop times in seconds.
    duration_frames : int
        Total number of frames in the session (to define array length).
    fps : float
        Camera frames per second.
    smooth_sd : float, optional
        Standard deviation for Gaussian smoothing. If None or 0,
        returns the raw binary trace (0/1).

    Returns
    -------
    trace : np.ndarray
        The generated temporal trace (either binary or smoothed density).
    """
    trace = np.zeros(duration_frames, dtype=float)

    start_indices = np.floor(event_starts * fps).astype(int)
    stop_indices = np.ceil(event_stops * fps).astype(int)

    # Clip indices to ensure they stay within session boundaries
    start_indices = np.clip(start_indices, 0, duration_frames)
    stop_indices = np.clip(stop_indices, 0, duration_frames)

    for s, e in zip(start_indices, stop_indices):
        if e > s:
            trace[s:e] = 1.0
        elif s < duration_frames:
            # Handle edge case where USV duration is sub-frame
            trace[s] = 1.0

    if smooth_sd is not None and smooth_sd > 0:
        kernel = Gaussian1DKernel(stddev=smooth_sd)
        trace = convolve(trace, kernel, boundary='extend',
                         nan_treatment='interpolate', preserve_nan=True)

    return trace


def find_bout_epochs(root_directories: list = None,
                     mouse_ids_dict: dict = None,
                     camera_fps_dict: dict = None,
                     features_dict: dict = None,
                     csv_sep: str = ',',
                     proportion_smoothing_sd: int | float = None,
                     filter_history: int | float = None,
                     prediction_mode: str = 'bout',
                     usv_bout_time: int | float = None,
                     min_usv_per_bout: int = None,
                     gmm_component_index: int = 0,
                     gmm_z_score: float = 2.58,
                     vocal_output_type: str = None,
                     noise_vocal_categories: list = None) -> dict:
    """
    Loads USV information data from a .csv file and samples epochs based on prediction mode.
    (See 'find_usv_categories' for category-based sampling).

    Parameters
    ----------
    root_directories : list
        Root directories of the input sessions.
    mouse_ids_dict : dict
        Sessions with mouse ID lists.
    camera_fps_dict : dict
        Sessions with camera frame rates.
    features_dict : dict
        Sessions with behavioral feature data.
    csv_sep : str, optional
        Separator used in the .csv file.
    proportion_smoothing_sd : int / float
        Smoothing sigma (in bins) for USV proportion.
    filter_history : int / float
        Amount of time (in s) preceding each event.
    prediction_mode : str, optional
        Controls sampling logic:
        - 'bout': Clean USV bout onsets (clean history + future bout)
                  vs.
                  Clean silent epochs (clean history + future silence).
        - 'individual': All valid USV onsets vs. Clean silent epochs.
        - 'state': Vocalizing state vs. Non-vocalizing state.
    usv_bout_time : int / float
        Duration of the "post-onset" window (in s). Used in 'bout' mode logic for NEGATIVE events.
    min_usv_per_bout : int
        Min USVs for a positive 'bout' event. Used in 'bout' mode.
    gmm_component_index : int
        GMM component index for IBI threshold calculation (default 0).
    gmm_z_score : float
        Z-score for IBI threshold calculation (default 2.58).
    vocal_output_type : str, optional
        Controls the type of vocal predictors generated in 'continuous_vocal_signals':
        - 'binary_joined': Aggregate binary trace (0/1) of all biological USVs ('usv_event').
        - 'presence_joined': Aggregate smoothed density of all biological USVs ('usv_proportion').
        - 'presence_categories': Individual smoothed density per category ('usv_cat_X').
        - 'presence_all': Both 'usv_proportion' and individual 'usv_cat_X' signals.
    noise_vocal_categories : list, optional
        List of USV categories to ignore (e.g., [0, 19] for noise/background).

    Returns
    -------
    usv_data_dict : dict
        Nested dictionary: session - mouseID - data.
        Includes unbalanced 'glm_usv' and 'glm_none' event time arrays.
    """

    # GMM parameters (modeling inter-USV interval distributions)
    male_gmm_params = {
        'means': [-2.78176965, -1.61892112, -0.62569187],
        'sds': [0.26162863, 0.77768956, 2.2298624]
    }
    female_gmm_params = {
        'means': [-2.76859759, -1.64223541, 1.88505038],
        'sds': [0.26499761, 1.07984569, 1.36805932]
    }

    usv_data_dict = {}
    for one_root_directory in root_directories:
        session_id = one_root_directory.split(os.sep)[-1]
        usv_data_dict[session_id] = {}
        usv_summary_data = pls.read_csv(source=next(pathlib.Path(f"{one_root_directory}{os.sep}audio{os.sep}").glob(f"**{os.sep}*_usv_summary.csv"), None),
                                        separator=csv_sep)

        has_category = 'usv_category' in usv_summary_data.columns
        if noise_vocal_categories and has_category:
            usv_summary_data = usv_summary_data.filter(~pls.col('usv_category').is_in(list(noise_vocal_categories)))

        mouse_track_names = mouse_ids_dict[session_id]

        for i, mouse_name in enumerate(mouse_track_names):
            usv_data_dict[session_id][mouse_name] = {'continuous_vocal_signals': {}}

            # Find inter-bout interval threshold based on GMM
            if i == 0:
                params = male_gmm_params
                sex_label = 'male'
            else:
                params = female_gmm_params
                sex_label = 'female'

            try:
                comp_mean = params['means'][gmm_component_index]
                comp_sd = params['sds'][gmm_component_index]
            except IndexError:
                raise ValueError(f"Invalid gmm_component_index {gmm_component_index} for {sex_label}.")

            ibi_threshold = _calculate_ibi_threshold(comp_mean, comp_sd, gmm_z_score)

            # Finds start and stop times of USVs for this particular mouse
            mouse_usvs_df = usv_summary_data.filter(pls.col('emitter') == mouse_name)
            usv_data_dict[session_id][mouse_name]['start'] = np.array(mouse_usvs_df['start'])
            usv_data_dict[session_id][mouse_name]['stop'] = np.array(mouse_usvs_df['stop'])

            # Get all USVs (this mouse + uncategorized) - this is important for clean epoch sampling
            all_usvs_df = usv_summary_data.filter((pls.col('emitter').is_null()) | (pls.col('emitter') == mouse_name))
            usv_start_mouse_and_uncategorized = np.array(all_usvs_df['start'])
            usv_stop_mouse_and_uncategorized = np.array(all_usvs_df['stop'])

            # Binarize the USV data
            session_fps = camera_fps_dict[session_id]
            session_duration_frames = features_dict[session_id].shape[0]

            # Compute local proportion of time spent vocalizing (Legacy/State Mode Support)
            usv_frame_events = _generate_vocal_trace(usv_data_dict[session_id][mouse_name]['start'],
                                                     usv_data_dict[session_id][mouse_name]['stop'],
                                                     session_duration_frames, session_fps, smooth_sd=None)

            usv_frame_proportion = _generate_vocal_trace(usv_data_dict[session_id][mouse_name]['start'],
                                                         usv_data_dict[session_id][mouse_name]['stop'],
                                                         session_duration_frames, session_fps, smooth_sd=proportion_smoothing_sd)

            usv_data_dict[session_id][mouse_name]['usv_count'] = usv_frame_events
            usv_data_dict[session_id][mouse_name]['usv_proportion'] = usv_frame_proportion

            # Generates continuous vocal signals based on specified output type
            if vocal_output_type in ['binary_joined', 'presence_joined', 'presence_categories', 'presence_all']:

                # A. Joined Aggregate logic
                if vocal_output_type in ['binary_joined', 'presence_joined', 'presence_all'] and mouse_usvs_df.height > 0:
                    if vocal_output_type == 'binary_joined':
                        usv_data_dict[session_id][mouse_name]['continuous_vocal_signals']['usv_event'] = usv_frame_events
                    else:
                        usv_data_dict[session_id][mouse_name]['continuous_vocal_signals']['usv_proportion'] = usv_frame_proportion

                # B. Per-category logic
                if vocal_output_type in ['presence_categories', 'presence_all'] and has_category and mouse_usvs_df.height > 0:
                    unique_cats = mouse_usvs_df['usv_category'].unique().to_list()
                    for cat_id in unique_cats:
                        try:
                            cat_int = int(cat_id)
                        except (ValueError, TypeError):
                            continue

                        cat_df = mouse_usvs_df.filter(pls.col('usv_category') == cat_id)
                        usv_data_dict[session_id][mouse_name]['continuous_vocal_signals'][f'usv_cat_{cat_int}'] = _generate_vocal_trace(
                            cat_df['start'].to_numpy(), cat_df['stop'].to_numpy(), session_duration_frames, session_fps, smooth_sd=proportion_smoothing_sd
                        )

            session_duration_sec = session_duration_frames / session_fps

            ### Mode 1: 'bout' (both USV and no-USV pre-bout periods must be clean)
            if prediction_mode == 'bout':

                # Get USV events (positive class)
                starts = usv_data_dict[session_id][mouse_name]['start']
                stops = usv_data_dict[session_id][mouse_name]['stop']

                valid_bouts = []

                if len(starts) > 0:
                    # Logic: filter using IBI threshold
                    if len(starts) > 1:
                        gaps = starts[1:] - stops[:-1]
                        break_indices = np.where(gaps >= ibi_threshold)[0]
                        bout_start_indices = np.concatenate(([0], break_indices + 1))
                        bout_end_indices = np.concatenate((break_indices, [len(starts) - 1]))
                    else:
                        bout_start_indices = np.array([0])
                        bout_end_indices = np.array([0])

                    for j in range(len(bout_start_indices)):
                        idx_start = bout_start_indices[j]
                        idx_end = bout_end_indices[j]

                        # Check size constraint
                        count = idx_end - idx_start + 1
                        if count < min_usv_per_bout: continue

                        # Check clean history constraint
                        bout_start_time = starts[idx_start]

                        # A. Must be far enough into session
                        if bout_start_time <= filter_history: continue

                        # B. Previous USV must be > filter_history away
                        if idx_start > 0:
                            prev_usv_end = stops[idx_start - 1]
                            if (bout_start_time - prev_usv_end) <= filter_history:
                                continue

                        valid_bouts.append(bout_start_time)

                usv_events_positive = np.array(valid_bouts)

                # Get no-USV events (negative class)
                # Get all possible tiled clean onsets
                all_clean_onsets = _get_clean_tiled_epochs(usv_start_mouse_and_uncategorized,
                                                           usv_stop_mouse_and_uncategorized,
                                                           filter_history,
                                                           session_duration_sec)

                # Get this mouse's USV starts
                mouse_usv_starts = usv_data_dict[session_id][mouse_name]['start']

                neg_list = []
                for t_onset in all_clean_onsets:
                    t_future_end = t_onset + usv_bout_time

                    # Count USVs in the "future" window
                    usvs_in_future = np.sum(
                        (mouse_usv_starts >= t_onset) &
                        (mouse_usv_starts < t_future_end)
                    )

                    # Only add if there are zero USVs in the future window
                    if usvs_in_future == 0:
                        neg_list.append(t_onset)

                usv_events_negative = np.array(neg_list)

            ### Mode 2: 'individual' (USV pre-vocalization periods can be "dirty" and no-USV must be clean)
            elif prediction_mode == 'individual':
                usv_starts_filter_bool = (usv_data_dict[session_id][mouse_name]['start'] > filter_history)
                usv_events_positive = usv_data_dict[session_id][mouse_name]['start'][usv_starts_filter_bool]

                usv_events_negative = _get_clean_tiled_epochs(usv_start_mouse_and_uncategorized,
                                                              usv_stop_mouse_and_uncategorized,
                                                              filter_history,
                                                              session_duration_sec)

            ### Mode 3: 'state' (both USV and no-USV pre-vocalization periods can be "dirty")
            elif prediction_mode == 'state':
                all_t = np.arange(filter_history, session_duration_sec + 1e-9, filter_history)
                frame_indices = np.floor(all_t * session_fps).astype(int)
                last_valid_index = session_duration_frames - 1
                valid_mask = frame_indices <= last_valid_index

                sample_t = all_t[valid_mask]
                sample_frames = frame_indices[valid_mask]

                if sample_frames.size > 0:
                    labels = usv_frame_events[sample_frames]
                    usv_events_positive = sample_t[labels == 1]
                    usv_events_negative = sample_t[labels == 0]
                else:
                    print(f"Warning: No valid 'state' samples found for {session_id}, {mouse_name}.")
                    usv_events_positive = np.array([])
                    usv_events_negative = np.array([])

            else:
                raise ValueError(f"Unknown prediction_mode: {prediction_mode}. Must be 'bout', 'individual', or 'state'.")

            usv_data_dict[session_id][mouse_name]['glm_usv'] = np.sort(usv_events_positive)
            usv_data_dict[session_id][mouse_name]['glm_none'] = np.sort(usv_events_negative)

    return usv_data_dict


def find_usv_categories(root_directories: list = None,
                        mouse_ids_dict: dict = None,
                        camera_fps_dict: dict = None,
                        features_dict: dict = None,
                        csv_sep: str = ',',
                        target_category: int = None,
                        category_column: str = 'usv_category',
                        filter_history: int | float = 0.0,
                        vocal_output_type: str = None,
                        proportion_smoothing_sd: float = 1.0,
                        noise_vocal_categories: list = None) -> dict:
    """
    Parses USV data for either one-vs-rest (binary) or multinomial (all-category) analysis,
    as well as extracting continuous spatial targets (UMAP coordinates) for probabilistic modeling.

    This function applies a consistent "Single Pipeline" filter to the raw data:
    1. Filters by mouse.
    2. Removes specified noise categories globally.
    3. Removes "history" (period of filter duration at session start) to ensure model stability.

    All outputs (modeling events, continuous signals, category streams, and continuous targets)
    are derived strictly from this filtered dataset to ensure mathematical consistency.

    Parameters
    ----------
    root_directories : list
        Root directories of the input sessions.
    mouse_ids_dict : dict
        Dictionary mapping session_id -> list of mouse names.
    camera_fps_dict : dict
        Mapping session_id -> frames per second.
    features_dict : dict
        Mapping session_id -> behavioral dataframe (used for session duration).
    csv_sep : str, optional
        Separator used in the .csv file.
    target_category : int, optional
        The integer ID of the USV category to predict (Positive Class).
        If None, the function runs in Multinomial mode and populates 'glm_events' with all categories.
    category_column : str, default 'usv_category'
        The name of the column in the CSV containing the category labels.
    filter_history : float, optional
        Minimum time (seconds) from the start of the session. Discards USVs before this.
    vocal_output_type : str, optional, default=None
        Controls the type of vocal predictors generated in 'continuous_vocal_signals':
        - 'binary_joined': Aggregate binary trace (0/1) of all biological USVs ('usv_event').
        - 'presence_joined': Aggregate smoothed density of all biological USVs ('usv_proportion').
        - 'presence_categories': Individual smoothed density per category ('usv_cat_X').
        - 'presence_all': Both 'usv_proportion' and individual 'usv_cat_X' signals.
    proportion_smoothing_sd : float, default 1.0
        Standard deviation for Gaussian smoothing (in frames).
    noise_vocal_categories : list, optional
        List of category IDs to exclude from continuous signals, models, and streams.

    Returns
    -------
    dict
        Nested dictionary: session -> mouseID -> data.
        Keys:
            'glm_events': Dict {cat_id: start_times_array} (Primary for Multinomial mode).
            'glm_target': Start times of target category (Only if target_category is set).
            'glm_other': Start times of all other USVs (Only if target_category is set).
            'continuous_vocal_signals': Continuous arrays for X variables (smoothed/binary).
            'category_streams': Dict {cat_id: {'start': np.array, 'stop': np.array}} (Filtered).
            'continuous_onsets': np.array of start times for valid USVs (used for continuous models).
            'continuous_targets': np.array of shape (N, 2) containing (umap1, umap2) coordinates.
    """

    usv_data_dict = {}

    for one_root_directory in root_directories:
        session_id = one_root_directory.split(os.sep)[-1]
        usv_data_dict[session_id] = {}

        # Locate USV Summary CSV
        csv_path = next(pathlib.Path(f"{one_root_directory}{os.sep}audio{os.sep}").glob(f"**{os.sep}*_usv_summary.csv"), None)
        if csv_path is None:
            print(f"Warning: No USV summary found for {session_id}. Skipping.")
            continue

        usv_summary_data = pls.read_csv(source=csv_path, separator=csv_sep)

        if category_column not in usv_summary_data.columns:
            raise ValueError(f"Column '{category_column}' missing in {csv_path}.")

        mouse_track_names = mouse_ids_dict.get(session_id, [])
        session_fps = camera_fps_dict[session_id]

        if session_id not in features_dict:
            print(f"Warning: No feature data for {session_id} to determine duration. Skipping.")
            continue

        session_duration_frames = features_dict[session_id].shape[0]

        for mouse_name in mouse_track_names:
            usv_data_dict[session_id][mouse_name] = {
                'continuous_vocal_signals': {},
                'category_streams': {},
                'glm_events': {},
                'glm_target': None,
                'glm_other': None,
                'continuous_onsets': None,
                'continuous_targets': None
            }

            # Filter by mouse
            mouse_usvs = usv_summary_data.filter(pls.col('emitter') == mouse_name).sort('start')

            # Filter noise categories (global removal)
            if noise_vocal_categories:
                mouse_usvs = mouse_usvs.filter(~pls.col(category_column).is_in(list(noise_vocal_categories)))

            # Filter history period (at start of session)
            mouse_usvs = mouse_usvs.filter(pls.col('start') > filter_history)

            if mouse_usvs.height == 0:
                continue

            # Get data is target-vs-other structure
            if target_category is not None:
                target_usvs = mouse_usvs.filter(pls.col(category_column) == target_category)
                other_usvs = mouse_usvs.filter(pls.col(category_column) != target_category)

                usv_data_dict[session_id][mouse_name]['glm_target'] = np.sort(target_usvs['start'].to_numpy())
                usv_data_dict[session_id][mouse_name]['glm_other'] = np.sort(other_usvs['start'].to_numpy())

            # Get data for all categories separately
            unique_cats = mouse_usvs[category_column].unique().to_list()

            for cat_id in unique_cats:
                try:
                    cat_int = int(cat_id)
                except (ValueError, TypeError):
                    continue

                cat_df = mouse_usvs.filter(pls.col(category_column) == cat_id)
                usv_data_dict[session_id][mouse_name]['glm_events'][cat_int] = np.sort(cat_df['start'].to_numpy())

            # Extract continuous vocal signals based on specified output type
            if vocal_output_type in ['binary_joined', 'presence_joined', 'presence_categories', 'presence_all']:

                # A. Aggregate (all calls combined)
                if vocal_output_type in ['binary_joined', 'presence_joined', 'presence_all']:
                    starts_all = mouse_usvs['start'].to_numpy()
                    stops_all = mouse_usvs['stop'].to_numpy()

                    if vocal_output_type == 'binary_joined':
                        usv_data_dict[session_id][mouse_name]['continuous_vocal_signals']['usv_event'] = _generate_vocal_trace(
                            starts_all, stops_all, session_duration_frames, session_fps, smooth_sd=None
                        )
                    else:
                        usv_data_dict[session_id][mouse_name]['continuous_vocal_signals']['usv_proportion'] = _generate_vocal_trace(
                            starts_all, stops_all, session_duration_frames, session_fps, smooth_sd=proportion_smoothing_sd
                        )

                # Per-category density
                if vocal_output_type in ['presence_categories', 'presence_all']:
                    for cat_id in unique_cats:
                        try:
                            cat_int = int(cat_id)
                        except (ValueError, TypeError):
                            continue

                        cat_df = mouse_usvs.filter(pls.col(category_column) == cat_id)
                        usv_data_dict[session_id][mouse_name]['continuous_vocal_signals'][f'usv_cat_{cat_int}'] = _generate_vocal_trace(
                            cat_df['start'].to_numpy(), cat_df['stop'].to_numpy(), session_duration_frames, session_fps, smooth_sd=proportion_smoothing_sd
                        )

            # Save raw category streams (for potential future use, e.g., custom signal generation or validation)
            unique_cats_raw = mouse_usvs[category_column].unique().to_list()
            for cat_id in unique_cats_raw:
                cat_df = mouse_usvs.filter(pls.col(category_column) == cat_id)
                if cat_df.height > 0:
                    usv_data_dict[session_id][mouse_name]['category_streams'][cat_id] = {
                        'start': np.sort(cat_df['start'].to_numpy()),
                        'stop': np.sort(cat_df['stop'].to_numpy())
                    }

            # Extract continuous targets (UMAP Coordinates)
            if 'usv_umap1' in mouse_usvs.columns and 'usv_umap2' in mouse_usvs.columns:
                usv_data_dict[session_id][mouse_name]['continuous_onsets'] = mouse_usvs['start'].to_numpy()

                umap_x = mouse_usvs['usv_umap1'].to_numpy()
                umap_y = mouse_usvs['usv_umap2'].to_numpy()
                usv_data_dict[session_id][mouse_name]['continuous_targets'] = np.column_stack((umap_x, umap_y))

    return usv_data_dict


def _calculate_ibi_threshold(log_mean: float, log_sd: float, z_score: float) -> float:
    """
    Calculates the Inter-Bout Interval (IBI) threshold based on GMM statistics.
    Typically, uses the log-normal properties of the first component (respiratory rhythm).

    IBI = exp( mu_log + (Z * sigma_log) )

    Parameters
    ----------
    log_mean : float
        The mean of the log-transformed inter-syllable intervals (Component 1).
    log_sd : float
        The standard deviation of the log-transformed intervals (Component 1).
    z_score : float
        The statistical cutoff (e.g., 2.58 for 99.5%).

    Returns
    -------
    float
        The calculated time threshold in seconds.
    """
    log_cutoff = log_mean + (z_score * log_sd)
    return np.exp(log_cutoff)


def find_variable_length_bouts(root_directories: list = None,
                               mouse_ids_dict: dict = None,
                               camera_fps_dict: dict = None,
                               features_dict: dict = None,
                               csv_sep: str = ',',
                               gmm_component_index: int = 0,
                               gmm_z_score: float = 2.58,
                               min_vocalizations: int = 2,
                               filter_history: float = 4.0,
                               proportion_smoothing_sd: float = 1.0,
                               vocal_output_type: str = None,
                               noise_vocal_categories: list = None) -> dict:
    """
    Identifies variable-length vocal bouts and generates continuous vocal density signals
    for regression analysis.

    This function processes USV data to define bouts (clusters of USVs) and
    "signals" (continuous density traces). It applies a strict filtering pipeline to
    ensure mechanical noise does not artificially bridge gaps between biological syllables.

    Process Outline:
    1.  Noise Filtering: Immediately removes rows where `usv_category` matches
        any integer in `noise_vocal_categories`. This prevents noise from acting as a "bridge"
        that merges distinct bouts and ensures continuous signals represent only biological audio.
    2.  GMM Thresholding: Selects sex-specific hardcoded Gaussian Mixture Model (GMM) parameters.
        Calculates a dynamic inter-bout interval (IBI) threshold using the log-mean and log-sd
        of the specified component (usually respiratory rhythm) plus a Z-score buffer.
    3.  Continuous Signal Generation:
        - Based on `vocal_output_type`, generates aggregate or category-specific signals.
        - Supports binary traces (0/1) or smoothed density traces via Gaussian convolution.
    4.  Bout Definition:
        - Calculates gaps between remaining valid syllables.
        - Clusters syllables into bouts wherever the gap is smaller than the calculated IBI.
    5.  Bout Validation:
        - Discards bouts with fewer than `min_vocalizations`.
        - Discards bouts starting before `filter_history` (insufficient pre-bout data).
    6.  Metric Calculation: Computes duration, USV count, and mask complexity for each valid bout.

    Parameters
    ----------
    root_directories : list
        Root directories of the input sessions.
    mouse_ids_dict : dict
        Dictionary mapping session_id -> list of mouse names [Male, Female].
    camera_fps_dict : dict
        Mapping session_id -> frames per second (needed for continuous signals).
    features_dict : dict
        Mapping session_id -> behavioral dataframe (needed for session duration).
    csv_sep : str, optional
        Separator for the CSV files.
    gmm_component_index : int, default 0
        GMM component index to use for IBI threshold calculation.
    gmm_z_score : float, default 2.58
        Z-score to apply to the GMM component statistics.
    min_vocalizations : int, default 2
        Minimum number of syllables required to form a valid bout.
    filter_history : float, default 4.0
        Time in seconds. Bouts starting before this time are discarded.
    proportion_smoothing_sd : float, default 1.0
        Standard deviation of the Gaussian kernel (in frames) used to smooth continuous signals.
    vocal_output_type : str, optional
        Controls the type of vocal predictors generated:
        - 'binary_joined': Aggregate binary trace (0/1) of all biological USVs ('usv_event').
        - 'presence_joined': Aggregate smoothed density of all biological USVs ('usv_proportion').
        - 'presence_categories': Individual smoothed density per category ('usv_cat_X').
        - 'presence_all': Both 'usv_proportion' and individual 'usv_cat_X' signals.
    noise_vocal_categories : list, optional
        List of USV category integers to exclude (e.g., [0, 19]). Defaults to [0, 19] if None.

    Returns
    -------
    dict
        Nested dictionary: session -> mouse -> data.
        Keys include:
            'bout_onsets': np.array of start times (seconds).
            'bout_durations': np.array of bout durations (seconds).
            'continuous_vocal_signals': dict containing generated arrays (e.g., 'usv_proportion').
    """

    # GMM parameters (for modeling inter-USV interval distributions)
    male_gmm_params = {
        'means': [-2.78176965, -1.61892112, -0.62569187],
        'sds': [0.26162863, 0.77768956, 2.2298624]
    }
    female_gmm_params = {
        'means': [-2.76859759, -1.64223541, 1.88505038],
        'sds': [0.26499761, 1.07984569, 1.36805932]
    }

    usv_data_dict = {}

    for one_root_directory in root_directories:
        session_id = one_root_directory.split(os.sep)[-1]
        usv_data_dict[session_id] = {}

        csv_path = next(pathlib.Path(f"{one_root_directory}{os.sep}audio{os.sep}").glob(f"**{os.sep}*_usv_summary.csv"), None)
        if csv_path is None:
            print(f"Warning: No USV summary found for {session_id}. Skipping.")
            continue

        usv_summary_data = pls.read_csv(source=csv_path, separator=csv_sep)

        has_mask = 'mask_number' in usv_summary_data.columns
        has_category = 'usv_category' in usv_summary_data.columns
        if not has_mask:
            print(f"Warning: 'mask_number' missing in {session_id}. Complexity = 0.")

        mouse_track_names = mouse_ids_dict.get(session_id, [])
        session_fps = camera_fps_dict[session_id]
        session_duration_frames = features_dict[session_id].shape[0]

        for i, mouse_name in enumerate(mouse_track_names):
            if i == 0:
                params = male_gmm_params
                sex_label = 'male'
            else:
                params = female_gmm_params
                sex_label = 'female'

            try:
                comp_mean = params['means'][gmm_component_index]
                comp_sd = params['sds'][gmm_component_index]
            except IndexError:
                raise ValueError(f"Invalid gmm_component_index {gmm_component_index} for {sex_label}.")

            ibi_threshold = _calculate_ibi_threshold(comp_mean, comp_sd, gmm_z_score)

            usv_data_dict[session_id][mouse_name] = {
                'bout_onsets': [],
                'bout_durations': [],
                'bout_syllable_counts': [],
                'mean_mask_complexity': [],
                'total_mask_complexity': [],
                'ibi_threshold_used': ibi_threshold,
                'continuous_vocal_signals': {}
            }

            # Filter for mouse and sort by start time
            mouse_usvs = usv_summary_data.filter(pls.col('emitter') == mouse_name).sort('start')

            # Remove noise categories
            if noise_vocal_categories and has_category:
                mouse_usvs = mouse_usvs.filter(~pls.col('usv_category').is_in(list(noise_vocal_categories)))

            # Generate continuous vocal signals based on specified output type
            if vocal_output_type in ['binary_joined', 'presence_joined', 'presence_categories', 'presence_all']:

                # A. Joined Aggregate logic
                if vocal_output_type in ['binary_joined', 'presence_joined', 'presence_all'] and mouse_usvs.height > 0:
                    starts_all = mouse_usvs['start'].to_numpy()
                    stops_all = mouse_usvs['stop'].to_numpy()

                    if vocal_output_type == 'binary_joined':
                        usv_data_dict[session_id][mouse_name]['continuous_vocal_signals']['usv_event'] = _generate_vocal_trace(
                            starts_all, stops_all, session_duration_frames, session_fps, smooth_sd=None
                        )
                    else:
                        usv_data_dict[session_id][mouse_name]['continuous_vocal_signals']['usv_proportion'] = _generate_vocal_trace(
                            starts_all, stops_all, session_duration_frames, session_fps, smooth_sd=proportion_smoothing_sd
                        )

                # B. Per-category logic
                if vocal_output_type in ['presence_categories', 'presence_all'] and has_category and mouse_usvs.height > 0:
                    unique_cats = mouse_usvs['usv_category'].unique().to_list()
                    for cat_id in unique_cats:
                        try:
                            cat_int = int(cat_id)
                        except (ValueError, TypeError):
                            continue

                        cat_df = mouse_usvs.filter(pls.col('usv_category') == cat_id)
                        usv_data_dict[session_id][mouse_name]['continuous_vocal_signals'][f'usv_cat_{cat_int}'] = _generate_vocal_trace(
                            cat_df['start'].to_numpy(), cat_df['stop'].to_numpy(), session_duration_frames, session_fps, smooth_sd=proportion_smoothing_sd
                        )

            if mouse_usvs.height == 0:
                continue

            starts = mouse_usvs['start'].to_numpy()
            stops = mouse_usvs['stop'].to_numpy()
            masks = mouse_usvs['mask_number'].to_numpy() if has_mask else np.ones(len(starts))

            if len(starts) > 1:
                gaps = starts[1:] - stops[:-1]
                break_indices = np.where(gaps >= ibi_threshold)[0]
                bout_start_indices = np.concatenate(([0], break_indices + 1))
                bout_end_indices = np.concatenate((break_indices, [len(starts) - 1]))
            else:
                bout_start_indices = np.array([0])
                bout_end_indices = np.array([0])

            for j in range(len(bout_start_indices)):
                idx_start = bout_start_indices[j]
                idx_end = bout_end_indices[j]

                # Metric: USV count (within bout)
                count = idx_end - idx_start + 1
                if count < min_vocalizations: continue

                # Metric: start time
                bout_start_time = starts[idx_start]
                if bout_start_time <= filter_history: continue

                # Metric: duration & complexity
                bout_end_time = stops[idx_end]
                duration = bout_end_time - bout_start_time
                bout_masks = masks[idx_start: idx_end + 1]
                total_complexity = np.sum(bout_masks)
                mean_complexity = np.mean(bout_masks)

                usv_data_dict[session_id][mouse_name]['bout_onsets'].append(bout_start_time)
                usv_data_dict[session_id][mouse_name]['bout_durations'].append(duration)
                usv_data_dict[session_id][mouse_name]['bout_syllable_counts'].append(count)
                usv_data_dict[session_id][mouse_name]['mean_mask_complexity'].append(mean_complexity)
                usv_data_dict[session_id][mouse_name]['total_mask_complexity'].append(total_complexity)

            for k in usv_data_dict[session_id][mouse_name]:
                if k != 'continuous_vocal_signals' and isinstance(usv_data_dict[session_id][mouse_name][k], list):
                    usv_data_dict[session_id][mouse_name][k] = np.array(usv_data_dict[session_id][mouse_name][k])

    return usv_data_dict

def load_pickle_modeling_data(pickle_file_path: str = None) -> dict:
    """
    Loads data from a .pickle file.

    Parameters
    ----------
    pickle_file_path : str
        Path to the .pickle file.

    Returns
    -------
    modeling_data : dict
        Modeling data.
    """

    with open(pickle_file_path, 'rb') as pickle_file:
        modeling_data = pickle.load(pickle_file)

    return modeling_data
