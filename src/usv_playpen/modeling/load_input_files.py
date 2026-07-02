"""
@author: bartulem
Module for loading raw data and orchestrating the data preparation pipeline for modeling
and regression analysis.

Key Capabilities:
1.  Data ingestion: Loading 3D behavioral features (CSV), track metadata (H5),
    and USV summaries.
2.  Epoch sampling: Identifying USV and No-USV event times using dynamic mixture-model
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
from pathlib import Path
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
    behavior_data : tuple (dict, dict, dict)
        Behavior, camera frame rate and track name data (keys are file names and values polars.DataFrames, float, list).
    """

    beh_feature_data_dict = {}
    camera_fr_dict = {}
    mouse_track_names_dict = {}
    for behavior_file_path in behavior_file_paths:
        beh_root = Path(behavior_file_path)
        sess_id = beh_root.name
        features_csv_file_path = next((beh_root / 'video').glob('**/*_points3d_translated_rotated_metric_behavioral_features.csv'), None)
        track_file_path = next((beh_root / 'video').glob('**/[!speaker]*_points3d_translated_rotated_metric.h5'), None)
        # Guard against missing input files (glob found no match) before passing
        # the path to h5py/polars, mirroring the `csv_path is None` skip in the
        # sibling USV loaders. Passing None to h5py.File/pls.read_csv would raise
        # an opaque low-level TypeError/OSError instead of a clear warning.
        if features_csv_file_path is None or track_file_path is None:
            print(f"Warning: Missing behavioral feature/track file for {sess_id}. Skipping.")
            continue
        with h5py.File(name=track_file_path, mode='r') as h5_file_mouse_obj:
            camera_fr_dict[sess_id] = float(h5_file_mouse_obj['recording_frame_rate'][()])
            mouse_track_names_dict[sess_id] = [item.decode('utf-8') for item in list(h5_file_mouse_obj['track_names'])]
        beh_feature_data_dict[sess_id] = pls.read_csv(source=features_csv_file_path, separator=csv_sep)

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

def _build_binary_occupancy(event_starts: np.ndarray,
                            event_stops: np.ndarray,
                            duration_frames: int,
                            fps: float) -> np.ndarray:
    """
    Convert USV start/stop timestamps into a binary occupancy trace.

    Each event spans the frames `[floor(start * fps), ceil(stop * fps))`,
    clipped to the session bounds, and contributes `1.0` to those frames; a
    sub-frame event (`stop` index not past `start` index) marks the single
    `start` frame. This is the raw occupancy array that
    :func:`_generate_vocal_trace` optionally smooths, factored out so a caller
    that needs *both* the binary and the smoothed trace for the same event set
    can build the occupancy once and reuse it (see :func:`_smooth_occupancy`)
    rather than rasterising the same timestamps twice.

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

    Returns
    -------
    trace : np.ndarray
        Binary occupancy trace (0/1) of shape `(duration_frames,)`, dtype
        `float`.
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

    return trace


def _smooth_occupancy(binary_trace: np.ndarray, smooth_sd: float) -> np.ndarray:
    """
    Gaussian-smooth a precomputed binary occupancy trace.

    Applies the same `Gaussian1DKernel` convolution that
    :func:`_generate_vocal_trace` uses, factored out so a caller holding an
    already-built occupancy trace (from :func:`_build_binary_occupancy`) can
    smooth it directly without rebuilding the binary trace from the raw
    timestamps. Producing the smoothed trace this way is bit-identical to
    calling `_generate_vocal_trace(..., smooth_sd=smooth_sd)`, because the
    convolution input (the binary occupancy) is the same array.

    Parameters
    ----------
    binary_trace : np.ndarray
        Binary occupancy trace as returned by :func:`_build_binary_occupancy`.
    smooth_sd : float
        Standard deviation (in frames) for the Gaussian smoothing kernel.

    Returns
    -------
    trace : np.ndarray
        Gaussian-smoothed density trace, same shape as `binary_trace`.
    """
    kernel = Gaussian1DKernel(stddev=smooth_sd)
    return convolve(binary_trace, kernel, boundary='extend',
                    nan_treatment='interpolate', preserve_nan=True)


def _generate_vocal_trace(event_starts: np.ndarray,
                          event_stops: np.ndarray,
                          duration_frames: int,
                          fps: float,
                          smooth_sd: float = None) -> np.ndarray:
    """
    Core utility to convert USV timestamps into a continuous temporal trace.

    This function handles the conversion of start/stop times into a binary
    occupancy array and optionally applies Gaussian smoothing. The binary
    rasterisation and the smoothing are factored into
    :func:`_build_binary_occupancy` and :func:`_smooth_occupancy`; callers that
    need both the binary and the smoothed trace for the same event set should
    call those two helpers directly so the occupancy is built only once.

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
    trace = _build_binary_occupancy(event_starts, event_stops, duration_frames, fps)

    if smooth_sd is not None and smooth_sd > 0:
        trace = _smooth_occupancy(trace, smooth_sd)

    return trace


def find_onset_epochs(root_directories: list = None,
                     mouse_ids_dict: dict = None,
                     camera_fps_dict: dict = None,
                     features_dict: dict = None,
                     csv_sep: str = ',',
                     proportion_smoothing_sd: int | float = None,
                     filter_history: int | float = None,
                     prediction_mode: str = 'bout',
                     usv_bout_time: int | float = None,
                     min_usv_per_bout: int = None,
                     mixture_model_component_index: int = 0,
                     mixture_model_z_score: float = 2.58,
                     mixture_model_params: dict = None,
                     vocal_output_type: str = None,
                     noise_vocal_categories: list = None,
                     category_column: str = 'usv_category',
                     target_category: int = None,
                     noise_column: str = 'usv_supercategory') -> dict:
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
        Smoothing sigma (in frames) for USV proportion.
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
    mixture_model_component_index : int
        mixture-model component index for IBI threshold calculation (default 0).
    mixture_model_z_score : float
        Z-score for IBI threshold calculation (default 2.58).
    mixture_model_params : dict
        A dict with 'male' and 'female' keys, each containing 'means' and 'sds' lists
        for the sex-specific IBI mixture-model components. Required: it is dereferenced
        unconditionally (mixture_model_params['male'] / mixture_model_params['female']), so passing
        None raises TypeError. Typically loaded from modeling_settings['mixture_model_params'].
    vocal_output_type : str, optional
        Controls the type of vocal predictors generated in 'continuous_vocal_signals':
        - 'pooled_binary': Aggregate binary trace (0/1) of all biological USVs ('usv_event').
        - 'pooled_rate': Aggregate smoothed density of all biological USVs ('usv_rate').
        - 'categories_rate': Individual smoothed density per category ('usv_cat_X').
        - 'all_rate': Both 'usv_rate' and individual 'usv_cat_X' signals.
    noise_vocal_categories : list, optional
        List of USV categories to ignore (e.g., [0, 19] for noise/background).
    category_column : str, optional
        Name of the per-USV category column in the summary .csv (e.g.
        'vae_supercategory', 'qlvm_supercategory', 'vae_category',
        'qlvm_category'). Used both for the per-category continuous predictor
        signals and, when `target_category` is set, for the onset-target filter.
    target_category : int, optional
        If set (and `prediction_mode == 'individual'`), restricts the POSITIVE
        onset events to USVs whose `category_column` value equals this category
        (e.g. broadband vocalizations = `vae_supercategory` 6). The predictor
        vocal traces ('usv_rate'/'usv_count'/'usv_cat_X') and the silent-epoch
        (negative) reference are still computed over ALL of the mouse's USVs, so
        the category choice changes only which onsets count as positive events.
        Ignored in 'bout' and 'state' modes, because the mixture-model inter-syllable-
        interval threshold used for bout grouping is calibrated on the all-USV
        interval distribution and would mis-group a category-sparsified
        sequence; in those modes all categories are pooled as before. If None
        (default), all USV categories are pooled (original behavior).
    noise_column : str, optional
        Name of the supercategory column used for global noise filtering. Kept
        separate from `category_column` so noise removal stays cohort-stable
        regardless of which experimental category column is chosen.

    Returns
    -------
    usv_data_dict : dict
        Nested dictionary: session - mouseID - data.
        Per-mouse keys exported:
            'start': np.array of positive-source USV start times (category-filtered
                only in 'individual' mode with a target_category, else all USVs).
            'stop': np.array of positive-source USV stop times.
            'continuous_vocal_signals': dict of continuous predictor traces keyed by
                vocal_output_type ('usv_event'/'usv_rate'/'usv_cat_X').
            'positive_events': unbalanced array of POSITIVE event times (seconds).
            'negative_events': unbalanced array of NEGATIVE (no-USV) event times (seconds).
            'usv_count': raw binary occupancy trace (0/1) over the full per-mouse USV set.
            'usv_rate': Gaussian-smoothed density trace over the full per-mouse USV set.
    """

    # mixture-model parameters (modeling inter-USV interval distributions)
    male_mixture_model_params = mixture_model_params['male']
    female_mixture_model_params = mixture_model_params['female']

    usv_data_dict = {}
    for one_root_directory in root_directories:
        sess_root = Path(one_root_directory)
        session_id = sess_root.name
        usv_data_dict[session_id] = {}

        csv_path = next((sess_root / 'audio').glob('**/*_usv_summary.csv'), None)
        if csv_path is None:
            print(f"Warning: No USV summary found for {session_id}. Skipping.")
            continue

        usv_summary_data = pls.read_csv(source=csv_path, separator=csv_sep)

        has_category = category_column in usv_summary_data.columns
        has_noise_col = noise_column in usv_summary_data.columns
        if noise_vocal_categories and has_noise_col:
            usv_summary_data = usv_summary_data.filter(~pls.col(noise_column).is_in(list(noise_vocal_categories)))

        if session_id not in mouse_ids_dict:
            print(f"Warning: No mouse names registered for {session_id}. Skipping.")
            continue
        mouse_track_names = mouse_ids_dict[session_id]

        for i, mouse_name in enumerate(mouse_track_names):
            usv_data_dict[session_id][mouse_name] = {'continuous_vocal_signals': {}}

            # Find inter-bout interval threshold based on mixture model
            if i == 0:
                params = male_mixture_model_params
                sex_label = 'male'
            else:
                params = female_mixture_model_params
                sex_label = 'female'

            try:
                comp_mean = params['means'][mixture_model_component_index]
                comp_sd = params['sds'][mixture_model_component_index]
            except IndexError:
                raise ValueError(f"Invalid mixture_model_component_index {mixture_model_component_index} for {sex_label}.")

            ibi_threshold = _calculate_ibi_threshold(comp_mean, comp_sd, mixture_model_z_score)

            # Finds start and stop times of USVs for this particular mouse.
            # Sort by `start` so downstream IBI-gap and bout-indexing logic
            # (which assumes monotonic starts/stops) is correct regardless of
            # the upstream CSV row order — matches `find_usv_categories` and
            # `find_variable_length_bouts`.
            mouse_usvs_df = usv_summary_data.filter(pls.col('emitter') == mouse_name).sort('start')

            # Positive-event source. When a single target USV category is
            # requested ('individual' mode only), restrict the USVs that become
            # POSITIVE onsets to that category. The full `mouse_usvs_df` still
            # drives the predictor vocal traces below, and the all-USV frame
            # still drives the silent-epoch (negative) reference, so neither the
            # predictors nor the negatives are affected by the category choice.
            if target_category is not None and prediction_mode == 'individual':
                if has_category:
                    positive_source_df = mouse_usvs_df.filter(pls.col(category_column) == target_category)
                else:
                    print(f"Warning: category column '{category_column}' absent for {session_id}; "
                          f"cannot restrict onsets to category {target_category}. Using all USVs.")
                    positive_source_df = mouse_usvs_df
            else:
                positive_source_df = mouse_usvs_df

            usv_data_dict[session_id][mouse_name]['start'] = np.array(positive_source_df['start'])
            usv_data_dict[session_id][mouse_name]['stop'] = np.array(positive_source_df['stop'])

            # Get all USVs (this mouse + uncategorized) - this is important for clean epoch sampling
            all_usvs_df = usv_summary_data.filter((pls.col('emitter').is_null()) | (pls.col('emitter') == mouse_name)).sort('start')
            usv_start_mouse_and_uncategorized = np.array(all_usvs_df['start'])
            usv_stop_mouse_and_uncategorized = np.array(all_usvs_df['stop'])

            # Binarize the USV data
            session_fps = camera_fps_dict[session_id]
            session_duration_frames = features_dict[session_id].shape[0]

            # Compute local proportion of time spent vocalizing (Legacy/State Mode Support).
            # These predictor traces are always derived from the full per-mouse
            # USV set (`mouse_usvs_df`), never the category-filtered positive
            # source, so an onset category filter cannot leak into the
            # 'usv_rate'/'usv_count' predictors or the 'state'-mode labels.
            # Build the binary occupancy once and reuse it for both the binary
            # `usv_count` trace and the smoothed `usv_rate` trace. The previous
            # code rasterised the same start/stop timestamps twice (once with
            # `smooth_sd=None`, once with the smoothing sd), rebuilding an
            # identical binary occupancy inside the second call before smoothing
            # it. Smoothing the shared occupancy here is bit-identical to that
            # second `_generate_vocal_trace` call. The `else` branch preserves
            # the original behaviour when `proportion_smoothing_sd` is None or 0
            # (in which case `_generate_vocal_trace` returned the raw binary
            # trace).
            usv_frame_events = _build_binary_occupancy(mouse_usvs_df['start'].to_numpy(),
                                                       mouse_usvs_df['stop'].to_numpy(),
                                                       session_duration_frames, session_fps)

            if proportion_smoothing_sd is not None and proportion_smoothing_sd > 0:
                usv_frame_rate = _smooth_occupancy(usv_frame_events, proportion_smoothing_sd)
            else:
                usv_frame_rate = usv_frame_events

            usv_data_dict[session_id][mouse_name]['usv_count'] = usv_frame_events
            usv_data_dict[session_id][mouse_name]['usv_rate'] = usv_frame_rate

            # Generates continuous vocal signals based on specified output type
            if vocal_output_type in ['pooled_binary', 'pooled_rate', 'categories_rate', 'all_rate']:

                # A. Joined Aggregate logic
                if vocal_output_type in ['pooled_binary', 'pooled_rate', 'all_rate'] and mouse_usvs_df.height > 0:
                    if vocal_output_type == 'pooled_binary':
                        usv_data_dict[session_id][mouse_name]['continuous_vocal_signals']['usv_event'] = usv_frame_events
                    else:
                        usv_data_dict[session_id][mouse_name]['continuous_vocal_signals']['usv_rate'] = usv_frame_rate

                # B. Per-category logic
                if vocal_output_type in ['categories_rate', 'all_rate'] and has_category and mouse_usvs_df.height > 0:
                    unique_cats = mouse_usvs_df[category_column].unique().to_list()
                    for cat_id in unique_cats:
                        try:
                            cat_int = int(cat_id)
                        except (ValueError, TypeError):
                            continue

                        cat_df = mouse_usvs_df.filter(pls.col(category_column) == cat_id)
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

                # No-USV epochs must be completely silent: include this
                # mouse's USVs *and* uncategorized USVs in the future-window
                # check. The previous version only counted the predictor
                # mouse's USVs, which let partner / uncategorized
                # vocalisations leak into the No-Bout class.
                all_usv_starts = usv_start_mouse_and_uncategorized

                # Vectorized equivalent of the per-onset future-window count: keep each
                # clean onset only if zero USVs (any source) fall in
                # [t_onset, t_onset + usv_bout_time). all_usv_starts is sorted, so two
                # side='left' searchsorted calls give every window's count at once --
                # `lo` counts USVs strictly before the onset (those >= t_onset are
                # kept) and `hi` counts USVs strictly before the window end (< the
                # future end), so `hi - lo` is exactly the old boolean-AND reduction.
                # Byte-identical, including the all_clean_onsets ordering of kept events.
                lo = np.searchsorted(all_usv_starts, all_clean_onsets, side='left')
                hi = np.searchsorted(all_usv_starts, all_clean_onsets + usv_bout_time, side='left')
                usv_events_negative = all_clean_onsets[(hi - lo) == 0]

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

            usv_data_dict[session_id][mouse_name]['positive_events'] = np.sort(usv_events_positive)
            usv_data_dict[session_id][mouse_name]['negative_events'] = np.sort(usv_events_negative)

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
                        noise_vocal_categories: list = None,
                        manifold_column_names: list = None,
                        noise_column: str = 'usv_supercategory') -> dict:
    """
    Parses USV data for either one-vs-rest (binary) or multinomial (all-category) analysis,
    as well as extracting continuous spatial targets (acoustic manifold coordinates) for
    probabilistic modeling.

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
        If None, the function runs in Multinomial mode and populates 'events_by_category' with all categories.
    category_column : str, default 'usv_category'
        The name of the column in the CSV containing the category labels.
    filter_history : float, optional
        Minimum time (seconds) from the start of the session. Discards USVs before this.
    vocal_output_type : str, optional, default=None
        Controls the type of vocal predictors generated in 'continuous_vocal_signals':
        - 'pooled_binary': Aggregate binary trace (0/1) of all biological USVs ('usv_event').
        - 'pooled_rate': Aggregate smoothed density of all biological USVs ('usv_rate').
        - 'categories_rate': Individual smoothed density per category ('usv_cat_X').
        - 'all_rate': Both 'usv_rate' and individual 'usv_cat_X' signals.
    proportion_smoothing_sd : float, default 1.0
        Standard deviation for Gaussian smoothing (in frames).
    noise_vocal_categories : list, optional
        List of category IDs to exclude from continuous signals, models, and streams.
    manifold_column_names : list, optional
        Ordered list of column names in the USV summary CSV that encode each USV's
        coordinates on the continuous acoustic manifold. If any of the configured
        columns is missing from the CSV for a given session/mouse, no continuous
        targets are written for that mouse. When None or empty, continuous target
        extraction is skipped entirely.
    noise_column : str, default 'usv_supercategory'
        Name of the supercategory column used for global noise filtering (removing
        the categories in `noise_vocal_categories`). Kept separate from
        `category_column` so the cohort-stable noise scheme stays fixed regardless
        of which experimental-category column the caller varies.

    Returns
    -------
    dict
        Nested dictionary: session -> mouseID -> data.
        Keys:
            'events_by_category': Dict {cat_id: start_times_array} (Primary for Multinomial mode).
            'target_events': Start times of target category (Only if target_category is set).
            'other_events': Start times of all other USVs (Only if target_category is set).
            'continuous_vocal_signals': Continuous arrays for X variables (smoothed/binary).
            'category_streams': Dict {cat_id: {'start': np.array, 'stop': np.array}} (Filtered).
            'continuous_onsets': np.array of start times for valid USVs (used for continuous models).
            'continuous_targets': np.array of shape (N, D) stacking the configured
                manifold columns in the order given by `manifold_column_names`.
            'continuous_supercategory': np.array of per-USV supercategory labels,
                aligned 1:1 with 'continuous_onsets'. Present only when the
                '<manifold_prefix>_supercategory' column exists in the source CSV.
            'continuous_category': np.array of per-USV category labels, aligned 1:1
                with 'continuous_onsets'. Present only when the
                '<manifold_prefix>_category' column exists in the source CSV.
    """

    usv_data_dict = {}

    for one_root_directory in root_directories:
        sess_root = Path(one_root_directory)
        session_id = sess_root.name
        usv_data_dict[session_id] = {}

        # Locate USV Summary CSV
        csv_path = next((sess_root / 'audio').glob('**/*_usv_summary.csv'), None)
        if csv_path is None:
            print(f"Warning: No USV summary found for {session_id}. Skipping.")
            continue

        usv_summary_data = pls.read_csv(source=csv_path, separator=csv_sep)

        if category_column not in usv_summary_data.columns:
            raise ValueError(f"Column '{category_column}' missing in {csv_path}.")

        # Strict membership check + direct lookup (no `.get()`
        # default). A session listed in the input directory but not
        # registered in `mouse_ids_dict` is a project-config bug
        # rather than a recoverable runtime case — skip with a
        # warning so it surfaces.
        if session_id not in mouse_ids_dict:
            print(f"Warning: No mouse names registered for {session_id}. Skipping.")
            continue
        mouse_track_names = mouse_ids_dict[session_id]
        session_fps = camera_fps_dict[session_id]

        if session_id not in features_dict:
            print(f"Warning: No feature data for {session_id} to determine duration. Skipping.")
            continue

        session_duration_frames = features_dict[session_id].shape[0]

        for mouse_name in mouse_track_names:
            usv_data_dict[session_id][mouse_name] = {
                'continuous_vocal_signals': {},
                'category_streams': {},
                'events_by_category': {},
                'target_events': None,
                'other_events': None,
                'continuous_onsets': None,
                'continuous_targets': None
            }

            # Filter by mouse
            mouse_usvs = usv_summary_data.filter(pls.col('emitter') == mouse_name).sort('start')

            # Filter noise categories (global removal). The noise filter
            # uses `noise_column` rather than `category_column` so the
            # cohort-stable noise scheme (typically `usv_supercategory`)
            # can be combined with any experimental-category column
            # (`category_column`) the caller wants to vary independently.
            if noise_vocal_categories and noise_column in mouse_usvs.columns:
                mouse_usvs = mouse_usvs.filter(~pls.col(noise_column).is_in(list(noise_vocal_categories)))

            # Filter history period (at start of session)
            mouse_usvs = mouse_usvs.filter(pls.col('start') > filter_history)

            if mouse_usvs.height == 0:
                continue

            # Get data in target-vs-other structure
            if target_category is not None:
                target_usvs = mouse_usvs.filter(pls.col(category_column) == target_category)
                other_usvs = mouse_usvs.filter(pls.col(category_column) != target_category)

                usv_data_dict[session_id][mouse_name]['target_events'] = np.sort(target_usvs['start'].to_numpy())
                usv_data_dict[session_id][mouse_name]['other_events'] = np.sort(other_usvs['start'].to_numpy())

            # Get data for all categories separately
            unique_cats = mouse_usvs[category_column].unique().to_list()

            for cat_id in unique_cats:
                try:
                    cat_int = int(cat_id)
                except (ValueError, TypeError):
                    continue

                cat_df = mouse_usvs.filter(pls.col(category_column) == cat_id)
                usv_data_dict[session_id][mouse_name]['events_by_category'][cat_int] = np.sort(cat_df['start'].to_numpy())

            # Extract continuous vocal signals based on specified output type
            if vocal_output_type in ['pooled_binary', 'pooled_rate', 'categories_rate', 'all_rate']:

                # A. Aggregate (all calls combined)
                if vocal_output_type in ['pooled_binary', 'pooled_rate', 'all_rate']:
                    starts_all = mouse_usvs['start'].to_numpy()
                    stops_all = mouse_usvs['stop'].to_numpy()

                    if vocal_output_type == 'pooled_binary':
                        usv_data_dict[session_id][mouse_name]['continuous_vocal_signals']['usv_event'] = _generate_vocal_trace(
                            starts_all, stops_all, session_duration_frames, session_fps, smooth_sd=None
                        )
                    else:
                        usv_data_dict[session_id][mouse_name]['continuous_vocal_signals']['usv_rate'] = _generate_vocal_trace(
                            starts_all, stops_all, session_duration_frames, session_fps, smooth_sd=proportion_smoothing_sd
                        )

                # Per-category density
                if vocal_output_type in ['categories_rate', 'all_rate']:
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
            for cat_id in unique_cats:
                cat_df = mouse_usvs.filter(pls.col(category_column) == cat_id)
                if cat_df.height > 0:
                    usv_data_dict[session_id][mouse_name]['category_streams'][cat_id] = {
                        'start': np.sort(cat_df['start'].to_numpy()),
                        'stop': np.sort(cat_df['stop'].to_numpy())
                    }

            # Extract continuous targets (user-configured acoustic manifold coordinates)
            if manifold_column_names:
                if all(col in mouse_usvs.columns for col in manifold_column_names):
                    usv_data_dict[session_id][mouse_name]['continuous_onsets'] = mouse_usvs['start'].to_numpy()

                    manifold_arrays = [mouse_usvs[col].to_numpy() for col in manifold_column_names]
                    usv_data_dict[session_id][mouse_name]['continuous_targets'] = np.column_stack(manifold_arrays)

                    # Per-USV supercategory and category labels. Used by
                    # downstream region-conditioned analyses (CNN saliency,
                    # cluster-circle membership). Derived from the manifold
                    # prefix: e.g., 'vae_umap1' -> 'vae' -> 'vae_supercategory',
                    # 'vae_category'. Stored as plain numpy arrays aligned
                    # 1:1 with continuous_onsets / continuous_targets above.
                    # Stored only when the columns are present in the source
                    # CSV; absent label arrays signal "this USV summary
                    # predates supercategory/category labelling."
                    manifold_prefix = manifold_column_names[0].rsplit('_', 1)[0]
                    super_col = f"{manifold_prefix}_supercategory"
                    cat_col = f"{manifold_prefix}_category"
                    if super_col in mouse_usvs.columns:
                        usv_data_dict[session_id][mouse_name]['continuous_supercategory'] = (
                            mouse_usvs[super_col].to_numpy()
                        )
                    if cat_col in mouse_usvs.columns:
                        usv_data_dict[session_id][mouse_name]['continuous_category'] = (
                            mouse_usvs[cat_col].to_numpy()
                        )

    return usv_data_dict


def _calculate_ibi_threshold(log_mean: float, log_sd: float, z_score: float) -> float:
    """
    Calculates the Inter-Bout Interval (IBI) threshold based on mixture-model statistics.
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
                               mixture_model_component_index: int = 0,
                               mixture_model_z_score: float = 2.58,
                               mixture_model_params: dict = None,
                               min_vocalizations: int = 2,
                               filter_history: float = 4.0,
                               proportion_smoothing_sd: float = 1.0,
                               vocal_output_type: str = None,
                               noise_vocal_categories: list = None,
                               category_column: str = 'usv_category',
                               noise_column: str = 'usv_supercategory') -> dict:
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
    2.  Mixture-model Thresholding: Selects sex-specific mixture-model parameters (from `mixture_model_params`).
        Calculates a dynamic inter-bout interval (IBI) threshold using the log-mean
        and log-sd of the specified component (usually respiratory rhythm) plus a Z-score buffer.
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
    mixture_model_component_index : int, default 0
        mixture-model component index to use for IBI threshold calculation.
    mixture_model_z_score : float, default 2.58
        Z-score to apply to the mixture-model component statistics.
    mixture_model_params : dict
        A dict with 'male' and 'female' keys, each containing 'means' and 'sds' lists
        for the sex-specific IBI mixture-model components. Required: it is dereferenced
        unconditionally (mixture_model_params['male'] / mixture_model_params['female']), so passing
        None raises TypeError. Typically loaded from modeling_settings['mixture_model_params'].
    min_vocalizations : int, default 2
        Minimum number of syllables required to form a valid bout.
    filter_history : float, default 4.0
        Time in seconds. Bouts starting before this time are discarded.
    proportion_smoothing_sd : float, default 1.0
        Standard deviation of the Gaussian kernel (in frames) used to smooth continuous signals.
    vocal_output_type : str, optional
        Controls the type of vocal predictors generated:
        - 'pooled_binary': Aggregate binary trace (0/1) of all biological USVs ('usv_event').
        - 'pooled_rate': Aggregate smoothed density of all biological USVs ('usv_rate').
        - 'categories_rate': Individual smoothed density per category ('usv_cat_X').
        - 'all_rate': Both 'usv_rate' and individual 'usv_cat_X' signals.
    noise_vocal_categories : list, optional
        List of USV category integers to exclude (e.g., [0, 19]). When `None`,
        no category-based noise filtering is applied — pass an explicit list
        if you want noise rows dropped before bout detection.
    category_column : str, default 'usv_category'
        Name of the per-USV experimental-category column in the summary .csv,
        used for the per-category continuous predictor signals ('usv_cat_X')
        when `vocal_output_type` requests them. May vary independently between
        runs.
    noise_column : str, default 'usv_supercategory'
        Name of the supercategory column used for global noise filtering
        (removing the categories in `noise_vocal_categories`). Kept separate from
        `category_column` so the cohort-stable noise scheme stays fixed regardless
        of which experimental-category column the caller varies.

    Returns
    -------
    dict
        Nested dictionary: session -> mouse -> data.
        Keys include:
            'bout_onsets': np.array of start times (seconds).
            'bout_durations': np.array of bout durations (seconds).
            'continuous_vocal_signals': dict containing generated arrays (e.g., 'usv_rate').
    """

    # mixture-model parameters (for modeling inter-USV interval distributions)
    male_mixture_model_params = mixture_model_params['male']
    female_mixture_model_params = mixture_model_params['female']

    usv_data_dict = {}

    for one_root_directory in root_directories:
        sess_root = Path(one_root_directory)
        session_id = sess_root.name
        usv_data_dict[session_id] = {}

        csv_path = next((sess_root / 'audio').glob('**/*_usv_summary.csv'), None)
        if csv_path is None:
            print(f"Warning: No USV summary found for {session_id}. Skipping.")
            continue

        usv_summary_data = pls.read_csv(source=csv_path, separator=csv_sep)

        has_mask = 'mask_number' in usv_summary_data.columns
        has_category = category_column in usv_summary_data.columns
        has_noise_col = noise_column in usv_summary_data.columns
        if not has_mask:
            print(f"Warning: 'mask_number' missing in {session_id}. "
                  f"Complexity defaults to the per-bout syllable count (mask = 1 per USV).")

        # Strict membership check + direct lookup (no `.get()`
        # default). A session listed in the input directory but not
        # registered in `mouse_ids_dict` is a project-config bug
        # rather than a recoverable runtime case — skip with a
        # warning so it surfaces.
        if session_id not in mouse_ids_dict:
            print(f"Warning: No mouse names registered for {session_id}. Skipping.")
            continue
        mouse_track_names = mouse_ids_dict[session_id]
        session_fps = camera_fps_dict[session_id]
        session_duration_frames = features_dict[session_id].shape[0]

        for i, mouse_name in enumerate(mouse_track_names):
            if i == 0:
                params = male_mixture_model_params
                sex_label = 'male'
            else:
                params = female_mixture_model_params
                sex_label = 'female'

            try:
                comp_mean = params['means'][mixture_model_component_index]
                comp_sd = params['sds'][mixture_model_component_index]
            except IndexError:
                raise ValueError(f"Invalid mixture_model_component_index {mixture_model_component_index} for {sex_label}.")

            ibi_threshold = _calculate_ibi_threshold(comp_mean, comp_sd, mixture_model_z_score)

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

            # Remove noise categories using `noise_column` (cohort-stable),
            # not `category_column` (experimental, may change between runs).
            if noise_vocal_categories and has_noise_col:
                mouse_usvs = mouse_usvs.filter(~pls.col(noise_column).is_in(list(noise_vocal_categories)))

            # Generate continuous vocal signals based on specified output type
            if vocal_output_type in ['pooled_binary', 'pooled_rate', 'categories_rate', 'all_rate']:

                # A. Joined Aggregate logic
                if vocal_output_type in ['pooled_binary', 'pooled_rate', 'all_rate'] and mouse_usvs.height > 0:
                    starts_all = mouse_usvs['start'].to_numpy()
                    stops_all = mouse_usvs['stop'].to_numpy()

                    if vocal_output_type == 'pooled_binary':
                        usv_data_dict[session_id][mouse_name]['continuous_vocal_signals']['usv_event'] = _generate_vocal_trace(
                            starts_all, stops_all, session_duration_frames, session_fps, smooth_sd=None
                        )
                    else:
                        usv_data_dict[session_id][mouse_name]['continuous_vocal_signals']['usv_rate'] = _generate_vocal_trace(
                            starts_all, stops_all, session_duration_frames, session_fps, smooth_sd=proportion_smoothing_sd
                        )

                # B. Per-category logic
                if vocal_output_type in ['categories_rate', 'all_rate'] and has_category and mouse_usvs.height > 0:
                    unique_cats = mouse_usvs[category_column].unique().to_list()
                    for cat_id in unique_cats:
                        try:
                            cat_int = int(cat_id)
                        except (ValueError, TypeError):
                            continue

                        cat_df = mouse_usvs.filter(pls.col(category_column) == cat_id)
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
