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
from __future__ import annotations

import pickle
import re
from pathlib import Path

import h5py
import numpy as np
import polars as pls
from astropy.convolution import Gaussian1DKernel, convolve
from scipy.stats import invgauss, norm

from ..os_utils import VOCAL_FLAG_COLUMNS, call_class_mask, drop_noise_usvs


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


# Vocal-predictor modes that build one trace per USV category and therefore need a
# per-call label column; 'pooled_rate' / 'pooled_binary' never read one.
CATEGORY_PREDICTOR_TYPES = ('categories_rate', 'all_rate')


def require_usv_category_column(category_column: str | None,
                                purpose: str,
                                summary_columns: list | None = None,
                                source: str | None = None) -> None:
    """
    Description
    -----------
    Fails clearly when a category-dependent analysis has no USV category label
    column to read. The usv_summary.csv files written by ``infer-qlvm-latents``
    carry the QLVM category of the regular model (``qlvm_category``, written by
    ``assign-qlvm-categories``; the shipped ``vocal_features.usv_category_column_name``),
    but a summary whose categories were not assigned yet, or a setting of ``null``,
    leaves a path without them. Every
    path that needs labels (per-category vocal predictors, the multinomial and
    binomial category models, the single-category onset target) calls this
    first, so the run stops with a message naming the setting instead of
    crashing obscurely on a missing column or silently building nothing.

    Called with ``summary_columns`` = None it checks only the setting (the early,
    before-anything-is-loaded check); with the columns of one summary it also
    checks that the configured column exists in that file.

    Parameters
    ----------
    category_column (str | None)
        The configured ``vocal_features.usv_category_column_name``.
    purpose (str)
        What needs the labels (named in the error), e.g.
        ``"usv_predictor_type 'categories_rate'"``.
    summary_columns (list | None)
        Columns of one usv_summary.csv; None skips the per-file check.
    source (str | None)
        The summary the columns came from (named in the error).

    Returns
    -------
    None
    """

    if category_column is None or category_column == '':
        error_message = (
            f"QLVM category labels are not available; set vocal_features.usv_category_column_name to an "
            f"existing label column. {purpose} needs a per-USV category label, and the setting is "
            f"{category_column!r}. Point the setting at a label column the usv_summary.csv files carry "
            f"(e.g. 'qlvm_category', written by assign-qlvm-categories) or use a "
            f"label-free alternative (e.g. usv_predictor_type 'pooled_rate')."
        )
        raise ValueError(error_message)
    if summary_columns is not None and category_column not in summary_columns:
        error_message = (
            f"QLVM category labels are not available; set vocal_features.usv_category_column_name to an "
            f"existing label column. {purpose} needs the column '{category_column}', which is absent from "
            f"{source}."
        )
        raise ValueError(error_message)


def require_labels_for_vocal_predictors(voc_settings: dict) -> None:
    """
    Description
    -----------
    The early, settings-only form of the vocal-predictor label check: a pipeline
    calls it before loading any session, so a ``usv_predictor_type`` of
    ``'categories_rate'`` / ``'all_rate'`` with no category label column
    (``usv_category_column_name`` null) stops at once instead of after the
    behavioral features of every session were read. The per-summary check (the
    column exists in each file) runs later, inside the loaders.

    Parameters
    ----------
    voc_settings (dict)
        The ``vocal_features`` block of the modeling settings; must contain
        ``usv_predictor_type`` and ``usv_category_column_name``.

    Returns
    -------
    None
    """

    if voc_settings['usv_predictor_type'] in CATEGORY_PREDICTOR_TYPES:
        require_usv_category_column(
            voc_settings['usv_category_column_name'],
            f"vocal_features.usv_predictor_type '{voc_settings['usv_predictor_type']}'",
        )


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


def _group_calls_into_bouts(starts: np.ndarray,
                            stops: np.ndarray,
                            ibi_threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Splits one mouse's time-sorted calls into bouts at every inter-call gap
    of at least ``ibi_threshold`` seconds.

    Parameters
    ----------
    starts : np.ndarray
        Call start times (s), sorted.
    stops : np.ndarray
        Call stop times (s), same order.
    ibi_threshold : float
        Gap (next start minus previous stop) at or above which a new bout begins.

    Returns
    -------
    bout_start_indices, bout_end_indices : tuple of np.ndarray
        Index of the first and last call of every bout; empty arrays when there
        are no calls.
    """

    if len(starts) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    if len(starts) > 1:
        gaps = starts[1:] - stops[:-1]
        break_indices = np.where(gaps >= ibi_threshold)[0]
        bout_start_indices = np.concatenate(([0], break_indices + 1))
        bout_end_indices = np.concatenate((break_indices, [len(starts) - 1]))
    else:
        bout_start_indices = np.array([0])
        bout_end_indices = np.array([0])
    return bout_start_indices, bout_end_indices


def _bout_offset_events(starts: np.ndarray,
                        stops: np.ndarray,
                        bout_start_indices: np.ndarray,
                        bout_end_indices: np.ndarray,
                        min_usv_per_bout: int,
                        negative_scheme: str,
                        min_singing_after_negative: float,
                        time_since_bout_onset_tolerance: float,
                        max_negatives_per_bout: int | None) -> tuple[np.ndarray, np.ndarray]:
    """
    Positive and negative event times for ``'bout_offset'`` mode.

    A positive is the offset of a bout's last call. A negative is an interior
    call's offset, and the question the two classes pose is: given that he is
    singing, why does he stop here rather than continue. ``'cross_bout'`` draws
    the negative from another bout of the same session at the same time since
    bout onset, so position in the bout is held fixed and the bouts that
    continue are, by definition, the longer ones. ``'within_bout'`` draws it
    from the same bout, so bout length and context are shared and only the
    approach to the end differs, at the price of admitting only bouts long
    enough to hold such a call.

    Parameters
    ----------
    starts : np.ndarray
        Call start times (s), sorted.
    stops : np.ndarray
        Call stop times (s), same order.
    bout_start_indices : np.ndarray
        First-call index of every bout (see ``_group_calls_into_bouts``).
    bout_end_indices : np.ndarray
        Last-call index of every bout.
    min_usv_per_bout : int
        Bouts with fewer calls contribute neither positives nor negatives.
    negative_scheme : str
        ``'cross_bout'`` or ``'within_bout'``.
    min_singing_after_negative : float
        Seconds the bout must keep singing after a negative's call.
    time_since_bout_onset_tolerance : float
        ``'cross_bout'`` only: maximal position mismatch (s) between a positive
        and its negative.
    max_negatives_per_bout : int or None
        ``'cross_bout'`` only: cap on negatives drawn from one bout.

    Returns
    -------
    positives, negatives : tuple of np.ndarray
        Event times (s); equal in length under both schemes.
    """

    if negative_scheme not in ('cross_bout', 'within_bout'):
        raise ValueError(f"Unknown negative_scheme: {negative_scheme}. Must be 'cross_bout' or 'within_bout'.")
    bouts = [(int(i0), int(i1)) for i0, i1 in zip(bout_start_indices, bout_end_indices)
             if (i1 - i0 + 1) >= min_usv_per_bout]
    positives: list[float] = []
    negatives: list[float] = []
    if negative_scheme == 'within_bout':
        for i0, i1 in bouts:
            end_time = stops[i1]
            eligible = [stops[k] for k in range(i0, i1) if (end_time - stops[k]) >= min_singing_after_negative]
            if not eligible:
                continue
            positives.append(float(end_time))
            negatives.append(float(max(eligible)))
        return np.array(positives), np.array(negatives)

    # cross_bout: every interior call with enough singing left is a candidate,
    # tagged with its position (time since its bout's onset) and its bout.
    candidate_position: list[float] = []
    candidate_time: list[float] = []
    candidate_bout: list[int] = []
    for b, (i0, i1) in enumerate(bouts):
        onset = starts[i0]
        end_time = stops[i1]
        for k in range(i0, i1):
            if (end_time - stops[k]) >= min_singing_after_negative:
                candidate_position.append(float(stops[k] - onset))
                candidate_time.append(float(stops[k]))
                candidate_bout.append(b)
    cand_pos = np.array(candidate_position)
    cand_time = np.array(candidate_time)
    cand_bout = np.array(candidate_bout, dtype=int)
    used = np.zeros(cand_pos.size, dtype=bool)
    per_bout_use = np.zeros(len(bouts), dtype=int)
    # Positives in order of position so the nearest-candidate rule is deterministic.
    ordered = sorted(((float(stops[i1] - starts[i0]), float(stops[i1]), b) for b, (i0, i1) in enumerate(bouts)))
    for position, end_time, b in ordered:
        if cand_pos.size == 0:
            break
        allowed = (~used) & (cand_bout != b) & (np.abs(cand_pos - position) <= time_since_bout_onset_tolerance)
        if max_negatives_per_bout is not None:
            allowed &= per_bout_use[cand_bout] < max_negatives_per_bout
        options = np.where(allowed)[0]
        if options.size == 0:
            continue
        j = options[np.argmin(np.abs(cand_pos[options] - position))]
        used[j] = True
        per_bout_use[cand_bout[j]] += 1
        positives.append(end_time)
        negatives.append(float(cand_time[j]))
    return np.array(positives), np.array(negatives)


def find_onset_epochs(root_directories: list = None,
                     mouse_ids_dict: dict = None,
                     camera_fps_dict: dict = None,
                     features_dict: dict = None,
                     csv_sep: str = ',',
                     proportion_smoothing_sd: int | float = None,
                     filter_history: int | float = None,
                     prediction_mode: str = 'bout_onset',
                     usv_bout_time: int | float = None,
                     min_usv_per_bout: int = None,
                     mixture_model_component_index: int = 0,
                     mixture_model_z_score: float = 2.58,
                     mixture_model_params: dict = None,
                     vocal_output_type: str = None,
                     exclude_noise_usvs: bool = True,
                     category_column: str = 'usv_category',
                     target_category: int = None,
                     target_type: str = 'usv',
                     negative_scheme: str = None,
                     min_singing_after_negative: int | float = None,
                     time_since_bout_onset_tolerance: int | float = None,
                     max_negatives_per_bout: int = None) -> dict:
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
        - 'bout_onset': Clean USV bout onsets (clean history + future bout)
                  vs.
                  Clean silent epochs (clean history + future silence).
        - 'individual': All valid USV onsets vs. Clean silent epochs.
        - 'state': Vocalizing state vs. Non-vocalizing state.
    usv_bout_time : int / float
        Duration of the "post-onset" window (in s). Used in 'bout_onset' mode logic for NEGATIVE events.
    min_usv_per_bout : int
        Min USVs for a positive 'bout_onset' event. Used in 'bout_onset' mode.
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
    exclude_noise_usvs : bool, optional
        Whether to drop the segments ``detect_usv_noise`` flagged as holding no
        vocalization (default True). A summary without the ``noise`` column raises.
    category_column : str, optional
        Name of the per-USV category column in the summary .csv (e.g.
        'qlvm_category'). Used both for the per-category continuous predictor
        signals and, when `target_category` is set, for the onset-target filter.
        May be None when neither of those needs it; a `vocal_output_type` of 'categories_rate' /
        'all_rate', or a `target_category` in 'individual' mode, with a None
        column or one absent from a session's summary raises ValueError
        (see `require_usv_category_column`) before any trace is built.
    target_category : int, optional
        If set (and `prediction_mode == 'individual'`), restricts the POSITIVE
        onset events to USVs whose `category_column` value equals this category
        (e.g. `qlvm_category` 3). The predictor
        vocal traces ('usv_rate'/'usv_count'/'usv_cat_X') and the silent-epoch
        (negative) reference are still computed over ALL of the mouse's USVs, so
        the category choice changes only which onsets count as positive events.
        Ignored in 'bout_onset' and 'state' modes, because the mixture-model inter-syllable-
        interval threshold used for bout grouping is calibrated on the all-USV
        interval distribution and would mis-group a category-sparsified
        sequence; in those modes all categories are pooled as before. If None
        (default), all USV categories are pooled (original behavior).
    target_type : str, optional
        Which calls are the POSITIVE onset source, read from the summary's
        ``usv`` / ``squeak`` booleans (written by the call classifier, ``detect-usv-squeaks``):
        ``'usv'`` (default) keeps pure USVs only (``usv & ~squeak``), ``'squeak'``
        pure squeaks only (``squeak & ~usv``), ``'all'`` every row. A segment holding
        both (a squeak and an ultrasonic call together) and null booleans (a segment
        too short to score) are in neither ``'usv'`` nor
        ``'squeak'``, so they are removed from both positive sequences. Applied before
        `target_category`, in every mode whose positives are call times ('bout_onset',
        'individual', 'bout_offset'), so in 'usv' mode bouts are grouped from pure
        ultrasonic calls alone -- a squeak or a "both" segment between two calls no
        longer joins or splits a bout -- and squeak onsets are no longer counted as USV
        onsets. As with `target_category`, the predictor vocal traces and the
        silent-epoch (negative) reference still use ALL of the mouse's calls, so a
        negative window is silent of squeaks too. ``'squeak'`` is accepted in
        'individual' mode only: bout grouping needs an inter-bout threshold, and
        the per-sex thresholds are calibrated on ultrasonic-call intervals, not on
        squeaks. A summary without the ``usv`` / ``squeak`` columns raises unless ``'all'``.
    negative_scheme : str, optional
        ``'bout_offset'`` mode only. ``'cross_bout'``: each positive (the last
        call's offset of a bout) is paired with an interior call's offset from
        ANOTHER bout of the same session, at the same time since that bout's
        onset (within ``time_since_bout_onset_tolerance``), in a bout that kept
        singing for at least ``min_singing_after_negative`` seconds after it;
        one negative per positive, nearest in position, each candidate used
        once, at most ``max_negatives_per_bout`` from any one bout; positives
        without a partner are dropped so the two groups share the same
        distribution of time since bout onset. ``'within_bout'``: the negative
        is the latest interior call of the SAME bout that ended at least
        ``min_singing_after_negative`` seconds before the bout's end; bouts too
        short to hold one contribute nothing. Either way the pairing is only
        the sampling recipe -- the returned groups are pooled downstream.
    min_singing_after_negative : int or float, optional
        ``'bout_offset'`` mode only: seconds the bout must keep singing after a
        negative's call, so the negative sits outside the ending itself.
    time_since_bout_onset_tolerance : int or float, optional
        ``'bout_offset'`` / ``'cross_bout'`` only: how far apart, in seconds, the
        time since bout onset of a positive and its negative may be.
    max_negatives_per_bout : int or None, optional
        ``'bout_offset'`` / ``'cross_bout'`` only: cap on negatives drawn from
        one bout; ``None`` means no cap.
        ``'bout_offset'`` targets the END of a bout: positives are the offsets
        of the last call of every bout with at least ``min_usv_per_bout``
        calls, with no clean-history or clean-future requirement (the
        inter-call threshold already defines the end), and negatives are
        interior-call offsets chosen by ``negative_scheme``.

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

    if target_type not in ('usv', 'squeak', 'all'):
        raise ValueError(f"Unknown target_type: {target_type!r}. Must be 'usv', 'squeak' or 'all'.")
    if target_type == 'squeak' and prediction_mode != 'individual':
        raise ValueError(
            f"target_type 'squeak' is supported in 'individual' mode only, not {prediction_mode!r}: "
            "bout grouping needs an inter-bout threshold, and the per-sex thresholds come from "
            "ultrasonic-call intervals."
        )

    # Labels are needed only by the per-category predictor traces and by the
    # single-category onset target ('individual' mode); either one without a label
    # column stops here, before any session is read, instead of silently building
    # no category traces or falling back to all calls.
    category_purposes = []
    if vocal_output_type in CATEGORY_PREDICTOR_TYPES:
        category_purposes.append(f"vocal_features.usv_predictor_type '{vocal_output_type}'")
    if target_category is not None and prediction_mode == 'individual':
        category_purposes.append(f"model_params.onset_target_category {target_category}")
    for category_purpose in category_purposes:
        require_usv_category_column(category_column, category_purpose)

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

        for category_purpose in category_purposes:
            require_usv_category_column(category_column, category_purpose,
                                        summary_columns=usv_summary_data.columns, source=str(csv_path))
        if exclude_noise_usvs:
            usv_summary_data = drop_noise_usvs(usv_summary_data, Path(csv_path).name)[0]
        if target_type != 'all' and any(column not in usv_summary_data.columns for column in VOCAL_FLAG_COLUMNS):
            raise ValueError(
                f"{Path(csv_path).name} has no {list(VOCAL_FLAG_COLUMNS)} columns, so target_type {target_type!r} "
                "cannot be applied; run detect-usv-squeaks on the session, or use 'all'."
            )

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

            # Legacy blocks carry log-space lognormal 'means'/'sds'; an
            # inverse-Gaussian block declares itself via 'model_class': 'ig'
            # and carries linear-time 'mus'/'lambdas' instead.
            try:
                if 'model_class' in params and params['model_class'] == 'ig':
                    comp_mu = params['mus'][mixture_model_component_index]
                    comp_lam = params['lambdas'][mixture_model_component_index]
                else:
                    comp_mean = params['means'][mixture_model_component_index]
                    comp_sd = params['sds'][mixture_model_component_index]
            except IndexError:
                raise ValueError(f"Invalid mixture_model_component_index {mixture_model_component_index} for {sex_label}.")

            if 'model_class' in params and params['model_class'] == 'ig':
                ibi_threshold = _calculate_ibi_threshold_ig(comp_mu, comp_lam, mixture_model_z_score)
            else:
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
            # The target type is applied first: pure ultrasonic calls, pure squeaks, or
            # every call. A segment holding both and null booleans match neither single type.
            if target_type in ('usv', 'squeak'):
                typed_source_df = mouse_usvs_df.filter(
                    call_class_mask(mouse_usvs_df, (target_type,), Path(csv_path).name)
                )
            else:
                typed_source_df = mouse_usvs_df
            if target_category is not None and prediction_mode == 'individual':
                # The column's presence was checked above, when the summary was read.
                positive_source_df = typed_source_df.filter(pls.col(category_column) == target_category)
            else:
                positive_source_df = typed_source_df

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
                if vocal_output_type in CATEGORY_PREDICTOR_TYPES and mouse_usvs_df.height > 0:
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

            ### Mode 1: 'bout_onset' (both USV and no-USV pre-bout periods must be clean)
            if prediction_mode == 'bout_onset':

                # Get USV events (positive class)
                starts = usv_data_dict[session_id][mouse_name]['start']
                stops = usv_data_dict[session_id][mouse_name]['stop']

                valid_bouts = []

                if len(starts) > 0:
                    # Logic: filter using IBI threshold
                    bout_start_indices, bout_end_indices = _group_calls_into_bouts(starts, stops, ibi_threshold)

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

            ### Mode 4: 'bout_offset' (the END of a bout against an interior call, see _bout_offset_events)
            elif prediction_mode == 'bout_offset':
                starts = usv_data_dict[session_id][mouse_name]['start']
                stops = usv_data_dict[session_id][mouse_name]['stop']
                bout_start_indices, bout_end_indices = _group_calls_into_bouts(starts, stops, ibi_threshold)
                usv_events_positive, usv_events_negative = _bout_offset_events(
                    starts=starts, stops=stops,
                    bout_start_indices=bout_start_indices, bout_end_indices=bout_end_indices,
                    min_usv_per_bout=min_usv_per_bout, negative_scheme=negative_scheme,
                    min_singing_after_negative=min_singing_after_negative,
                    time_since_bout_onset_tolerance=time_since_bout_onset_tolerance,
                    max_negatives_per_bout=max_negatives_per_bout)

            else:
                raise ValueError(f"Unknown prediction_mode: {prediction_mode}. Must be 'bout_onset', 'individual', 'state' or 'bout_offset'.")

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
                        exclude_noise_usvs: bool = True,
                        manifold_column_names: list = None) -> dict:
    """
    Parses USV data for either one-vs-rest (binary) or multinomial (all-category) analysis,
    as well as extracting continuous spatial targets (acoustic manifold coordinates) for
    probabilistic modeling.

    This function applies a consistent "Single Pipeline" filter to the raw data:
    1. Filters by mouse.
    2. Removes the segments the noise classifier flagged (``noise`` column), when
       ``exclude_noise_usvs`` is True.
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
    category_column : str | None, default 'usv_category'
        The name of the column in the CSV containing the category labels. It is
        required (a None / empty value, or a column absent from a session's
        summary, raises ValueError via `require_usv_category_column`) on the
        categorical paths: no `manifold_column_names` (the multinomial / binomial
        category models), a `target_category`, or a `vocal_output_type` of
        'categories_rate' / 'all_rate'. On the continuous manifold path it is
        optional: None returns the manifold targets with empty
        'events_by_category' / 'category_streams'; a column that is set must
        still exist in every summary.
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
    exclude_noise_usvs : bool, optional
        Whether to drop the segments ``detect_usv_noise`` flagged as holding no
        vocalization (default True). A summary without the ``noise`` column raises.
    manifold_column_names : list, optional
        Ordered list of column names in the USV summary CSV that encode each USV's
        coordinates on the continuous acoustic manifold. If any of the configured
        columns is missing from the CSV for a given session/mouse, no continuous
        targets are written for that mouse. When None or empty, continuous target
        extraction is skipped entirely. Calls whose coordinates are null / NaN in
        any configured column (calls the embedding could not place) are dropped
        from 'continuous_onsets', 'continuous_targets' and the label arrays, with
        the number dropped printed per session-mouse pair and in total.

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
            'continuous_category': np.array of per-USV category labels, aligned 1:1
                with 'continuous_onsets'. Present only when the
                '<manifold_prefix>_category' column exists in the source CSV.
    """

    # Per-call labels are needed by the categorical paths only: the multinomial /
    # binomial category models (no manifold columns), a `target_category`, and the
    # per-category predictor traces. The continuous manifold path reads the torus
    # coordinates alone, so there a null `category_column` simply means "no
    # category packets". A categorical path without labels stops here, before any
    # session is read.
    category_purposes = []
    if not manifold_column_names:
        category_purposes.append("The USV category models (multinomial / binomial)")
    if target_category is not None:
        category_purposes.append(f"The binomial target category {target_category}")
    if vocal_output_type in CATEGORY_PREDICTOR_TYPES:
        category_purposes.append(f"vocal_features.usv_predictor_type '{vocal_output_type}'")
    for category_purpose in category_purposes:
        require_usv_category_column(category_column, category_purpose)
    use_categories = category_column is not None and category_column != ''

    usv_data_dict = {}
    n_unplaced_total = 0
    n_unplaced_sessions = 0

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

        # A configured column must exist (an explicit setting that names a missing
        # column is a configuration error on every path, the manifold one included).
        if use_categories:
            require_usv_category_column(category_column, "vocal_features.usv_category_column_name",
                                        summary_columns=usv_summary_data.columns, source=str(csv_path))

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

            # Drop the segments holding no vocalization (global removal). This is independent of
            # `category_column`: the noise verdict comes from the classifier, so the experimental
            # category the caller models can vary without changing which rows are real calls.
            if exclude_noise_usvs:
                mouse_usvs = drop_noise_usvs(mouse_usvs, f"{session_id} ({mouse_name})")[0]

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

            # Get data for all categories separately (none without a label column)
            unique_cats = mouse_usvs[category_column].unique().to_list() if use_categories else []

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
                if vocal_output_type in CATEGORY_PREDICTOR_TYPES:
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
                    # A call the embedding could not place (outside the model's
                    # duration window, no SAM mask, no condition value) has null
                    # coordinates; it has no manifold target, so it is dropped here,
                    # together with its onset and labels, before anything downstream
                    # (the inverse-density KDE, the regressions, the GLM-HMM) sees a NaN.
                    # A column CSV inference read as text (all-null) casts to NaN too.
                    manifold_arrays = [
                        mouse_usvs[col].cast(pls.Float64, strict=False).fill_null(np.nan).to_numpy()
                        for col in manifold_column_names
                    ]
                    manifold_targets = np.column_stack(manifold_arrays)
                    placed = np.isfinite(manifold_targets).all(axis=1)
                    n_unplaced = int(np.count_nonzero(~placed))
                    if n_unplaced > 0:
                        print(f"  {session_id} ({mouse_name}): dropped {n_unplaced} of {placed.size} calls with "
                              f"null/NaN manifold coordinates ({', '.join(manifold_column_names)}).")
                        n_unplaced_total += n_unplaced
                        n_unplaced_sessions += 1
                    usv_data_dict[session_id][mouse_name]['continuous_onsets'] = mouse_usvs['start'].to_numpy()[placed]
                    usv_data_dict[session_id][mouse_name]['continuous_targets'] = manifold_targets[placed]

                    # Per-USV category labels, the acoustic regions of
                    # downstream region-conditioned analyses (torus macro
                    # score, CNN saliency, cluster-circle membership). Derived
                    # from the manifold prefix: e.g., 'qlvm1' -> 'qlvm' ->
                    # 'qlvm_category' (R-1..R-k). Only the regular map carries
                    # a category column, so a conditional map ('qlvm_dur',
                    # 'qlvm_ent') gets none. Stored as a plain numpy array
                    # aligned 1:1 with continuous_onsets / continuous_targets
                    # above, only when the column is present in the source CSV;
                    # an absent label array signals "this USV summary carries
                    # no category for this map."
                    manifold_prefix = re.sub(r'\d+$', '', manifold_column_names[0])
                    cat_col = f"{manifold_prefix}_category"
                    if cat_col in mouse_usvs.columns:
                        usv_data_dict[session_id][mouse_name]['continuous_category'] = (
                            mouse_usvs[cat_col].to_numpy()[placed]
                        )

    if manifold_column_names:
        print(f"Manifold targets: dropped {n_unplaced_total} calls with null/NaN coordinates in total "
              f"({n_unplaced_sessions} session-mouse pairs affected).")

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


def _calculate_ibi_threshold_ig(mu: float, lam: float, z_score: float) -> float:
    """
    Calculates the Inter-Bout Interval (IBI) threshold from an inverse-Gaussian
    first component. The z-score cutoff convention is preserved by mapping z to
    its Gaussian upper-tail probability (e.g. 2.58 -> 99.5%) and taking that
    quantile of the IG component:

    IBI = F_IG^{-1}( Phi(z); mu, lambda )

    Parameters
    ----------
    mu : float
        The inverse-Gaussian mean of the first component (seconds).
    lam : float
        The inverse-Gaussian shape parameter of the first component (seconds).
    z_score : float
        The statistical cutoff (e.g., 2.58 for 99.5%), mapped through the
        standard-normal CDF to a quantile level.

    Returns
    -------
    float
        The calculated time threshold in seconds.
    """
    q = float(norm.cdf(z_score))
    # scipy's invgauss(mu_s, scale=s) has mean mu_s * s and shape s, so
    # (mu_s = mu/lam, scale = lam) is IG(mu, lam).
    return float(invgauss.ppf(q, mu / lam, scale=lam))


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
                               exclude_noise_usvs: bool = True,
                               category_column: str = 'usv_category') -> dict:
    """
    Identifies variable-length vocal bouts and generates continuous vocal density signals
    for regression analysis.

    This function processes USV data to define bouts (clusters of USVs) and
    "signals" (continuous density traces). It applies a strict filtering pipeline to
    ensure mechanical noise does not artificially bridge gaps between biological syllables.

    Process Outline:
    1.  Noise Filtering: Immediately removes the segments `detect_usv_noise` flagged as holding
        no vocalization. This prevents noise from acting as a "bridge" that merges distinct bouts
        and ensures continuous signals represent only biological audio.
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
    exclude_noise_usvs : bool, optional
        Whether to drop the segments ``detect_usv_noise`` flagged as holding no
        vocalization (default True). A summary without the ``noise`` column raises.
    category_column : str | None, default 'usv_category'
        Name of the per-USV experimental-category column in the summary .csv,
        used for the per-category continuous predictor signals ('usv_cat_X')
        when `vocal_output_type` requests them. May vary independently between
        runs. Only read by 'categories_rate' / 'all_rate', which raise ValueError
        (via `require_usv_category_column`) when it is None or absent from a
        session's summary; the pooled modes accept None.

    Returns
    -------
    dict
        Nested dictionary: session -> mouse -> data.
        Keys include:
            'bout_onsets': np.array of start times (seconds).
            'bout_durations': np.array of bout durations (seconds).
            'continuous_vocal_signals': dict containing generated arrays (e.g., 'usv_rate').
    """

    # Per-category traces need a label column; without one they stop here, before
    # any session is read, instead of silently building no category predictors.
    category_traces = vocal_output_type in CATEGORY_PREDICTOR_TYPES
    category_purpose = f"vocal_features.usv_predictor_type '{vocal_output_type}'"
    if category_traces:
        require_usv_category_column(category_column, category_purpose)

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
        if category_traces:
            require_usv_category_column(category_column, category_purpose,
                                        summary_columns=usv_summary_data.columns, source=str(csv_path))
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

            # Legacy blocks carry log-space lognormal 'means'/'sds'; an
            # inverse-Gaussian block declares itself via 'model_class': 'ig'
            # and carries linear-time 'mus'/'lambdas' instead.
            try:
                if 'model_class' in params and params['model_class'] == 'ig':
                    comp_mu = params['mus'][mixture_model_component_index]
                    comp_lam = params['lambdas'][mixture_model_component_index]
                else:
                    comp_mean = params['means'][mixture_model_component_index]
                    comp_sd = params['sds'][mixture_model_component_index]
            except IndexError:
                raise ValueError(f"Invalid mixture_model_component_index {mixture_model_component_index} for {sex_label}.")

            if 'model_class' in params and params['model_class'] == 'ig':
                ibi_threshold = _calculate_ibi_threshold_ig(comp_mu, comp_lam, mixture_model_z_score)
            else:
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

            # Drop the segments holding no vocalization; independent of `category_column`, which
            # is experimental and may change between runs.
            if exclude_noise_usvs:
                mouse_usvs = drop_noise_usvs(mouse_usvs, f"{session_id} ({mouse_name})")[0]

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
                if category_traces and mouse_usvs.height > 0:
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
