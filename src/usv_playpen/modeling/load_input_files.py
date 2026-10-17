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
from collections.abc import Iterable
from pathlib import Path

import h5py
import numpy as np
import polars as pls
from astropy.convolution import Gaussian1DKernel, convolve
from scipy.stats import invgauss, norm

from ..os_utils import (
    QLVM_CATEGORY_COLUMN,
    QLVM_SQUEAK_MAP,
    VOCAL_CLASSES,
    VOCAL_FLAG_COLUMNS,
    drop_noise_usvs,
    vocal_class_mask,
    vocal_event_times,
    vocal_occupancy_times,
)


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
# per-USV label column; 'pooled_rate' / 'pooled_binary' never read one.
CATEGORY_PREDICTOR_TYPES = ('categories_rate', 'all_rate')

# Vocal classes a bout-grouping prediction mode accepts as its target
# (`model_params.onset_target_classes`): bouts are grouped with the per-sex
# inter-bout threshold, which is calibrated on USV intervals, and squeaks do not
# come in bouts, so a target holding pure squeaks has no bouts to group.
# 'variable_length_bouts' labels `find_variable_length_bouts` (the bout-parameter
# and behavioral-response pipelines), which groups bouts from the same classes.
BOUT_PREDICTION_MODES = ('bout_onset', 'bout_offset', 'variable_length_bouts')
BOUT_TARGET_CLASSES = ('usv', 'overlap')

# The vocal-signal names a predictor trace can carry (see `vocal_trace_name`), and
# the suffixes the trace builders append: '_rate' (smoothed density), '_event'
# (binary occupancy) and '_cat_<k>' (smoothed density of one QLVM category).
VOCAL_TRACE_NAMES = ('usv', 'squeak', 'overlap', 'all_vocalizations')
POOLED_VOCAL_SUFFIXES = ('_rate', '_event')
CATEGORY_VOCAL_INFIX = '_cat_'


def vocal_trace_name(classes: Iterable[str]) -> str:
    """
    Description
    -----------
    Names the trace built from one list of vocal classes by what it holds. An
    overlap segment holds a USV and a squeak, so it adds nothing to the name of a
    list that already holds one of them:

    * ``["usv"]`` or ``["usv", "overlap"]`` -> ``"usv"``;
    * ``["squeak"]`` or ``["squeak", "overlap"]`` -> ``"squeak"``;
    * ``["overlap"]`` -> ``"overlap"``;
    * ``["usv", "squeak"]`` or ``["usv", "squeak", "overlap"]`` -> ``"all_vocalizations"``.

    The exact list a trace came from is recorded in the run metadata
    (``vocal_predictor_classes``), so ``["usv"]`` and ``["usv", "overlap"]`` runs
    stay distinguishable although their traces share a name.

    Parameters
    ----------
    classes (Iterable[str])
        Vocal classes, each one of ``os_utils.VOCAL_CLASSES``.

    Returns
    -------
    name (str)
        One of ``VOCAL_TRACE_NAMES``.
    """

    class_set = set(classes)
    if 'usv' in class_set and 'squeak' in class_set:
        return 'all_vocalizations'
    if 'usv' in class_set:
        return 'usv'
    if 'squeak' in class_set:
        return 'squeak'
    return 'overlap'


def onset_target_tag(target_classes: list) -> str:
    """
    Description
    -----------
    The filename segment of an onset target (``model_params.onset_target_classes``):
    empty for the default ``["usv"]``, otherwise ``"_"`` plus the classes in
    ``os_utils.VOCAL_CLASSES`` order joined by ``"+"`` (e.g. ``["overlap", "usv"]``
    -> ``"_usv+overlap"``). The Level-1 onset pipeline and the model selection
    both build their file names with it, so runs of different target classes never
    share an artifact name and the selection finds its own step files.

    Parameters
    ----------
    target_classes (list)
        The configured target classes.

    Returns
    -------
    tag (str)
        ``""`` or ``"_<class>+<class>..."``.
    """

    ordered = [value for value in VOCAL_CLASSES if value in target_classes]
    return "" if ordered == ['usv'] else "_" + "+".join(ordered)


def validate_vocal_class_settings(target_classes: list | None,
                                  predictor_classes: list | None,
                                  prediction_mode: str | None) -> None:
    """
    Description
    -----------
    Checks the two vocal-class settings before any session is read, so a
    misconfigured run stops at once with a message naming the setting:

    * ``model_params.onset_target_classes`` (``target_classes``): a non-empty list
      of distinct ``os_utils.VOCAL_CLASSES``; in a bout-grouping mode
      (``BOUT_PREDICTION_MODES``) every class must be one of
      ``BOUT_TARGET_CLASSES``, because squeaks do not come in bouts and the
      inter-bout threshold is calibrated on USV intervals;
    * ``vocal_features.vocal_predictor_classes`` (``predictor_classes``): a list of
      lists, each a non-empty list of distinct ``os_utils.VOCAL_CLASSES`` that
      becomes one trace (``vocal_trace_name``). Lists may share classes (an
      overlap segment holds both a USV and a squeak, so it belongs in a USV trace
      and in a squeak trace alike), but two lists that give the same trace name
      would give two columns of one name and are refused.

    Either argument may be None to skip its check (a pipeline without an onset
    target, or without vocal predictors).

    Parameters
    ----------
    target_classes (list | tuple | None)
        The configured ``model_params.onset_target_classes``.
    predictor_classes (list | tuple | None)
        The configured ``vocal_features.vocal_predictor_classes`` (lists or tuples).
    prediction_mode (str | None)
        The configured ``model_params.model_target_vocal_type``; None skips the
        bout-mode restriction.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        Any of the rules above is broken.
    """

    if target_classes is not None:
        if not isinstance(target_classes, (list, tuple)) or len(target_classes) == 0:
            raise ValueError(f"model_params.onset_target_classes must be a non-empty list of {list(VOCAL_CLASSES)}; got {target_classes!r}.")
        unknown = [value for value in target_classes if value not in VOCAL_CLASSES]
        if unknown or len(set(target_classes)) != len(target_classes):
            raise ValueError(f"model_params.onset_target_classes {target_classes!r} must hold distinct classes from {list(VOCAL_CLASSES)}.")
        if prediction_mode in BOUT_PREDICTION_MODES and not set(target_classes) <= set(BOUT_TARGET_CLASSES):
            raise ValueError(
                f"model_params.onset_target_classes {target_classes!r} is not valid for {prediction_mode!r}: bouts are "
                f"grouped from {list(BOUT_TARGET_CLASSES)} only, because squeaks do not come in bouts and the inter-bout "
                "threshold is calibrated on USV intervals. Use 'individual' mode for squeak onsets."
            )
    if predictor_classes is not None:
        if not isinstance(predictor_classes, (list, tuple)) or len(predictor_classes) == 0:
            raise ValueError(f"vocal_features.vocal_predictor_classes must be a non-empty list of class lists, e.g. [['usv']]; got {predictor_classes!r}.")
        trace_names = []
        for group in predictor_classes:
            if not isinstance(group, (list, tuple)) or len(group) == 0:
                raise ValueError(f"vocal_features.vocal_predictor_classes: every entry must be a non-empty list of classes, e.g. ['usv', 'overlap']; got {group!r}.")
            unknown = [value for value in group if value not in VOCAL_CLASSES]
            if unknown or len(set(group)) != len(group):
                raise ValueError(f"vocal_features.vocal_predictor_classes entry {group!r} must hold distinct classes from {list(VOCAL_CLASSES)}.")
            trace_names.append(vocal_trace_name(group))
        duplicated = sorted({name for name in trace_names if trace_names.count(name) > 1})
        if duplicated:
            clashing = [group for group in predictor_classes if vocal_trace_name(group) in duplicated]
            raise ValueError(
                f"vocal_features.vocal_predictor_classes entries {clashing} all give the trace name(s) {duplicated}; "
                "each entry must give a different trace (see vocal_trace_name)."
            )


def is_vocal_signal(signal_key: str) -> bool:
    """
    Description
    -----------
    Whether a column (``"<animal>.<signal>"``) or a bare signal key is a vocal
    predictor trace built by the loaders: ``<name>_rate``, ``<name>_event`` or
    ``<name>_cat_<k>`` with ``<name>`` one of ``VOCAL_TRACE_NAMES``. Every
    pipeline uses this to tell its vocal columns from its kinematic ones.

    Parameters
    ----------
    signal_key (str)
        A column name or a bare ``continuous_vocal_signals`` key.

    Returns
    -------
    is_vocal (bool)
        True for a vocal predictor trace.
    """

    bare = signal_key.split('.', 1)[-1]
    for name in VOCAL_TRACE_NAMES:
        if bare in tuple(f"{name}{suffix}" for suffix in POOLED_VOCAL_SUFFIXES):
            return True
        category_prefix = f"{name}{CATEGORY_VOCAL_INFIX}"
        if bare.startswith(category_prefix) and bare[len(category_prefix):].isdigit():
            return True
    return False


def is_pooled_vocal_signal(signal_key: str) -> bool:
    """
    Description
    -----------
    Whether a column or bare signal key is a POOLED vocal trace (``<name>_rate`` /
    ``<name>_event``), as opposed to a per-category one. The target animal's own
    pooled traces are left out of its predictors (``build_vocal_signal_columns``):
    its own vocal density right before its own onsets would predict them
    trivially, while its per-category traces carry which categories it emitted.

    Parameters
    ----------
    signal_key (str)
        A column name or a bare ``continuous_vocal_signals`` key.

    Returns
    -------
    is_pooled (bool)
        True for ``<name>_rate`` / ``<name>_event`` with ``<name>`` in ``VOCAL_TRACE_NAMES``.
    """

    bare = signal_key.split('.', 1)[-1]
    return any(bare == f"{name}{suffix}" for name in VOCAL_TRACE_NAMES for suffix in POOLED_VOCAL_SUFFIXES)


def build_vocal_predictor_signals(mouse_usvs: pls.DataFrame,
                                  predictor_classes: list,
                                  vocal_output_type: str | None,
                                  category_column: str | None,
                                  duration_frames: int,
                                  fps: float,
                                  smooth_sd: float | None,
                                  source: str) -> dict:
    """
    Description
    -----------
    Builds one animal's vocal predictor traces: one trace per entry of
    ``predictor_classes``, from the rows of ``mouse_usvs`` whose vocal class
    (``os_utils.vocal_class_mask``, read from the ``usv`` / ``squeak`` booleans) is
    in that entry, named by ``vocal_trace_name``. Each row is timed as the event
    the entry asks about (``os_utils.vocal_event_times``): a pure squeak by its
    squeak extent ``squeak_start`` / ``squeak_end``, a pure USV by its segment
    ``start`` / ``stop``, and an overlap segment by the squeak extent in a
    squeak-only entry, by the segment in a USV-only entry, and by the union of the
    two otherwise; an entry that can time a row by the squeak extent raises when
    the summary has no ``squeak_start`` / ``squeak_end``. ``vocal_output_type``
    decides how each trace is built:

    * ``'pooled_binary'``: binary occupancy, ``<name>_event``;
    * ``'pooled_rate'``: Gaussian-smoothed occupancy, ``<name>_rate``;
    * ``'categories_rate'``: one smoothed trace per QLVM category,
      ``<name>_cat_<k>``, for an entry without pure squeaks; an entry holding
      pure squeaks gets one ``<name>_rate`` instead, because pure squeaks carry no
      QLVM category (the QLVM map embeds USVs, and overlap segments, only);
    * ``'all_rate'``: ``<name>_rate`` plus, for an entry without pure squeaks,
      the per-category traces;
    * None or ``''``: no traces.

    A pooled trace is built even when the entry matches no row (an all-zero
    trace: the animal emitted none of those classes), so every session carries
    the same pooled columns; per-category traces exist for the categories the
    animal emitted, as before.

    Parameters
    ----------
    mouse_usvs (polars.DataFrame)
        One animal's summary rows (noise already dropped when configured),
        sorted by ``start``.
    predictor_classes (list | tuple)
        The configured ``vocal_features.vocal_predictor_classes`` (validated by
        ``validate_vocal_class_settings``).
    vocal_output_type (str | None)
        The configured ``vocal_features.vocal_predictor_type``.
    category_column (str | None)
        The per-USV QLVM category column; needed only by ``'categories_rate'`` /
        ``'all_rate'`` (checked by the caller).
    duration_frames (int)
        Session length in camera frames.
    fps (float)
        Camera frame rate.
    smooth_sd (float | None)
        Gaussian smoothing sd in frames for the ``_rate`` / ``_cat_`` traces.
    source (str)
        What the rows came from, named in the error when the booleans are missing.

    Returns
    -------
    signals (dict)
        Signal key -> trace of shape ``(duration_frames,)``.
    """

    signals = {}
    if not vocal_output_type:
        return signals
    if vocal_output_type not in ('pooled_binary', 'pooled_rate', 'categories_rate', 'all_rate'):
        raise ValueError(
            f"Unknown vocal_features.vocal_predictor_type {vocal_output_type!r}; choose 'pooled_binary', "
            "'pooled_rate', 'categories_rate', 'all_rate' or null."
        )
    for group in predictor_classes:
        name = vocal_trace_name(group)
        group_usvs = vocal_event_times(mouse_usvs.filter(vocal_class_mask(mouse_usvs, group, source)), group, source)
        starts = group_usvs['start'].to_numpy()
        stops = group_usvs['stop'].to_numpy()
        split_by_category = vocal_output_type in CATEGORY_PREDICTOR_TYPES and 'squeak' not in group
        if vocal_output_type == 'pooled_binary':
            signals[f"{name}_event"] = _generate_vocal_trace(starts, stops, duration_frames, fps, smooth_sd=None)
        elif vocal_output_type in ('pooled_rate', 'all_rate') or not split_by_category:
            signals[f"{name}_rate"] = _generate_vocal_trace(starts, stops, duration_frames, fps, smooth_sd=smooth_sd)
        if split_by_category and group_usvs.height > 0:
            for cat_id in group_usvs[category_column].unique().to_list():
                try:
                    cat_int = int(cat_id)
                except (ValueError, TypeError):
                    continue
                cat_usvs = group_usvs.filter(pls.col(category_column) == cat_id)
                signals[f"{name}{CATEGORY_VOCAL_INFIX}{cat_int}"] = _generate_vocal_trace(
                    cat_usvs['start'].to_numpy(), cat_usvs['stop'].to_numpy(), duration_frames, fps, smooth_sd=smooth_sd
                )
    return signals


def read_usv_summary_table(csv_path: str | Path, csv_sep: str) -> pls.DataFrame:
    """
    Description
    -----------
    Reads one session's ``*_usv_summary.csv`` for the modeling loaders.

    Every column's type is inferred from ALL rows (``infer_schema_length=None``):
    with the default 100-row inference a column empty in the first 100 rows (e.g.
    ``qlvm_category`` or ``mask_number`` of a session that opens with noise or
    unlabelled rows) is read as String, and the category filters
    (``pls.col(category_column) == k``) and mask sums then fail. ``usv_id`` stays
    a zero-padded string.

    The per-mouse selections filter on ``emitter``, so a session whose
    ``emitter`` column is entirely empty (``vcl-assign`` has not run since the
    last ``das-summarize``, which rewrites the summary without it) gives every
    mouse zero vocalizations; a warning names that cause here, before the downstream
    "no events" errors that would otherwise hide it.

    Parameters
    ----------
    csv_path (str | pathlib.Path)
        Path of the session's USV summary.
    csv_sep (str)
        Column separator of the summary.

    Returns
    -------
    usv_summary_data (polars.DataFrame)
        The summary table.
    """

    usv_summary_data = pls.read_csv(
        source=csv_path, separator=csv_sep, infer_schema_length=None, schema_overrides={'usv_id': pls.String}
    )
    if 'emitter' in usv_summary_data.columns and usv_summary_data.height and usv_summary_data['emitter'].null_count() == usv_summary_data.height:
        print(
            f"Warning: {Path(csv_path).name} has no assigned emitter (the emitter column is empty), so no vocalization "
            f"is attributed to either mouse; run vcl-assign on the session (after das-summarize)."
        )
    return usv_summary_data


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
        ``"vocal_predictor_type 'categories_rate'"``.
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
            f"label-free alternative (e.g. vocal_predictor_type 'pooled_rate')."
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
    calls it before loading any session, so a ``vocal_predictor_type`` of
    ``'categories_rate'`` / ``'all_rate'`` with no category label column
    (``usv_category_column_name`` null) stops at once instead of after the
    behavioral features of every session were read. The per-summary check (the
    column exists in each file) runs later, inside the loaders. With vocal
    predictors on, it also checks ``vocal_predictor_classes``
    (``validate_vocal_class_settings``).

    Parameters
    ----------
    voc_settings (dict)
        The ``vocal_features`` block of the modeling settings; must contain
        ``vocal_predictor_type``, ``vocal_predictor_classes`` and
        ``usv_category_column_name``.

    Returns
    -------
    None
    """

    if voc_settings['vocal_predictor_type']:
        validate_vocal_class_settings(None, voc_settings['vocal_predictor_classes'], None)
    if voc_settings['vocal_predictor_type'] in CATEGORY_PREDICTOR_TYPES:
        require_usv_category_column(
            voc_settings['usv_category_column_name'],
            f"vocal_features.vocal_predictor_type '{voc_settings['vocal_predictor_type']}'",
        )


# The torus coordinates of the squeak map (``os_utils.QLVM_SQUEAK_MAP``): manifold targets read
# from these columns are timed by the squeak extent (``find_usv_categories``).
SQUEAK_MAP_COLUMNS = (f"{QLVM_SQUEAK_MAP}1", f"{QLVM_SQUEAK_MAP}2")


def _get_clean_tiled_epochs(usv_starts_all: np.ndarray,
                            usv_stops_all: np.ndarray,
                            filter_history: float,
                            session_duration_sec: float) -> np.ndarray:
    """
    Finds all valid 'clean' (no-USV) epochs using the "forbidden zone" tiling method.

    A negative epoch is identified by its window END ``t`` (the pre-event window
    is ``[t - filter_history, t]``), and it needs exactly the clean history a
    positive onset needs under :func:`_history_is_dirty`: no vocalization of the
    silence reference may start before ``t`` and end after
    ``t - filter_history``. Nothing more is required, so a negative may end as
    early after a vocalization as a clean positive may begin.

    This method works by:
    1. Defining a "forbidden zone" of window ends around every vocalization as the
       open interval (start_time, stop_time + filter_history): exactly the ends
       whose window a vocalization would touch.
    2. Merging all overlapping forbidden zones.
    3. Finding the "clean" zones *between* these merged zones (their boundary
       points are clean: a window ending at ``stop + filter_history`` starts
       exactly where the vocalization stopped, and one ending at a vocalization's
       start ends exactly where it begins). The first zone begins at
       ``filter_history``, the earliest end whose window fits in the session.
    4. Tiling the window ENDS of each clean zone from its start, every
       ``filter_history`` seconds, so successive windows abut without
       overlapping and the first window of a zone ends at the zone's start.

    Parameters
    ----------
    usv_starts_all : np.ndarray
        Array of all USV start times (in seconds), including uncategorized.
    usv_stops_all : np.ndarray
        Array of all USV stop times (in seconds), including uncategorized.
    filter_history : float
        The duration (in seconds) of the pre-event window. This defines the
        length of the history that must be silent and the spacing of the
        non-overlapping tiles.
    session_duration_sec : float
        The total duration of the session in seconds.

    Returns
    -------
    np.ndarray
        An array of valid, non-overlapping "no-USV" event times (window end times)
        in seconds, ascending.
    """

    # Forbidden window ends: the open interval (start, stop + filter_history)
    forbidden_starts = np.asarray(usv_starts_all, dtype=float)
    forbidden_ends = np.asarray(usv_stops_all, dtype=float) + filter_history

    # Merge overlapping forbidden intervals; open intervals that only touch leave
    # their shared boundary point clean, so they are not merged.
    if forbidden_starts.size > 0:
        indices = np.argsort(forbidden_starts)
        sorted_starts = forbidden_starts[indices]
        sorted_ends = forbidden_ends[indices]

        merged_starts = [sorted_starts[0]]
        merged_ends = [sorted_ends[0]]

        for i in range(1, sorted_starts.size):
            if sorted_starts[i] < merged_ends[-1]:
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

    # Invert to get "clean" zones of window ends (no window may begin before the
    # session does, so no end before filter_history)
    session_start = filter_history
    clean_starts = []
    clean_ends = []

    if merged_starts.size == 0:
        # No USVs at all, entire session is clean
        clean_starts.append(session_start)
        clean_ends.append(session_duration_sec)
    else:
        # First clean zone: from session_start to the first forbidden start
        if merged_starts[0] >= session_start:
            clean_starts.append(session_start)
            clean_ends.append(merged_starts[0])

        # Middle clean zones: gaps between forbidden zones
        for i in range(merged_ends.size - 1):
            clean_starts.append(max(merged_ends[i], session_start))
            clean_ends.append(merged_starts[i + 1])

        # Last clean zone: from last forbidden end to session_end
        if merged_ends[-1] <= session_duration_sec:
            clean_starts.append(max(merged_ends[-1], session_start))
            clean_ends.append(session_duration_sec)

    # "Tile" the window ends of each clean zone from its start (in seconds)
    all_valid_t = []
    for start, end in zip(clean_starts, clean_ends, strict=True):
        if end >= start:
            possible_ts = np.arange(start, end + 1e-9, filter_history)
            all_valid_t.extend(possible_ts)

    return np.array(all_valid_t)


def _future_window_is_silent(candidate_onsets: np.ndarray,
                             reference_starts: np.ndarray,
                             future_seconds: float) -> np.ndarray:
    """
    Description
    -----------
    Flags the candidate negative (no-USV) onsets whose future window
    ``[t, t + future_seconds)`` holds no start of the silence reference (the
    target mouse's vocalizations of every class plus those assigned to no
    animal). ``reference_starts`` is sorted, so two ``side='left'``
    ``searchsorted`` calls give every window's count at once -- ``lo`` counts
    starts strictly before the onset (those ``>= t`` are kept) and ``hi`` counts
    starts strictly before the window end, so ``hi - lo`` is the number of
    vocalizations starting inside the window. Without this check a tiled
    negative with a clean history could end milliseconds before an onset.

    Parameters
    ----------
    candidate_onsets (np.ndarray)
        Candidate negative onsets (s), e.g. from ``_get_clean_tiled_epochs``.
    reference_starts (np.ndarray)
        Sorted start times (s) of the silence reference.
    future_seconds (float)
        Length (s) of the window that must stay silent after each onset
        (``model_params.usv_bout_time``).

    Returns
    -------
    silent (np.ndarray)
        Boolean mask over ``candidate_onsets``, True where the window is silent.
    """

    lo = np.searchsorted(reference_starts, candidate_onsets, side='left')
    hi = np.searchsorted(reference_starts, candidate_onsets + future_seconds, side='left')
    return (hi - lo) == 0


def _history_is_dirty(onset_time: float,
                      reference_starts: np.ndarray,
                      reference_stops: np.ndarray,
                      filter_history: float) -> bool:
    """
    Description
    -----------
    Whether a positive onset's pre-onset window holds a vocalization of the
    silence reference (the target mouse's vocalizations of every class plus those
    assigned to no animal): one that starts before the onset and ends after
    ``onset_time - filter_history``. That is exactly the rule the negative epochs
    obey (``_get_clean_tiled_epochs`` forbids the window ends in the open interval
    ``(start, stop + filter_history)`` around every reference vocalization and
    tiles each clean zone from its start), so a clean positive and a negative need
    equally silent histories, and nothing more: a vocalization that ends exactly
    where the window begins does not touch it, for either class. The onset's own
    vocalization starts AT ``onset_time`` and is therefore not counted.

    Parameters
    ----------
    onset_time (float)
        The candidate onset, in seconds.
    reference_starts (np.ndarray)
        Start times (s) of the silence reference.
    reference_stops (np.ndarray)
        Stop times (s), same order.
    filter_history (float)
        Length (s) of the pre-onset window.

    Returns
    -------
    dirty (bool)
        True when the window holds a reference vocalization.
    """

    return bool(np.any((reference_starts < onset_time) & (reference_stops > onset_time - filter_history)))


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


def _group_vocalizations_into_bouts(starts: np.ndarray,
                            stops: np.ndarray,
                            ibi_threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Splits one mouse's time-sorted vocalizations into bouts at every inter-vocalization gap
    of at least ``ibi_threshold`` seconds.

    Parameters
    ----------
    starts : np.ndarray
        Vocalization start times (s), sorted.
    stops : np.ndarray
        Vocalization stop times (s), same order.
    ibi_threshold : float
        Gap (next start minus previous stop) at or above which a new bout begins.

    Returns
    -------
    bout_start_indices, bout_end_indices : tuple of np.ndarray
        Index of the first and last vocalization of every bout; empty arrays when there
        are no vocalizations.
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

    A positive is the offset of a bout's last vocalization. A negative is an interior
    vocalization's offset, and the question the two classes pose is: given that he is
    singing, why does he stop here rather than continue. ``'cross_bout'`` draws
    the negative from another bout of the same session at the same time since
    bout onset, so position in the bout is held fixed and the bouts that
    continue are, by definition, the longer ones. ``'within_bout'`` draws it
    from the same bout, so bout length and context are shared and only the
    approach to the end differs, at the price of admitting only bouts long
    enough to hold such a vocalization.

    Parameters
    ----------
    starts : np.ndarray
        Vocalization start times (s), sorted.
    stops : np.ndarray
        Vocalization stop times (s), same order.
    bout_start_indices : np.ndarray
        First-vocalization index of every bout (see ``_group_vocalizations_into_bouts``).
    bout_end_indices : np.ndarray
        Last-vocalization index of every bout.
    min_usv_per_bout : int
        Bouts with fewer vocalizations contribute neither positives nor negatives.
    negative_scheme : str
        ``'cross_bout'`` or ``'within_bout'``.
    min_singing_after_negative : float
        Seconds the bout must keep singing after a negative's vocalization.
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

    # cross_bout: every interior vocalization with enough singing left is a candidate,
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
                     target_classes: list | tuple = ('usv',),
                     predictor_classes: list | tuple = (('usv',),),
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
        - 'individual': All valid USV onsets vs. Clean silent epochs (clean history
                  + future silence, the same negatives rule as 'bout_onset').
        - 'state': Vocalizing state vs. Non-vocalizing state.
    usv_bout_time : int / float
        Duration of the "post-onset" window (in s). Used in 'bout_onset' and 'individual'
        mode logic for NEGATIVE events: a negative onset is kept only if no vocalization
        of the silence reference starts within this window after it.
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
        one trace per entry of `predictor_classes`, named ``<name>`` by
        ``vocal_trace_name`` (see ``build_vocal_predictor_signals``):
        - 'pooled_binary': binary occupancy (0/1), '<name>_event'.
        - 'pooled_rate': smoothed density, '<name>_rate'.
        - 'categories_rate': smoothed density per QLVM category, '<name>_cat_X'
          (an entry holding pure squeaks, which carry no category: '<name>_rate').
        - 'all_rate': '<name>_rate' plus the per-category '<name>_cat_X' traces.
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
        vocal traces and the silent-epoch
        (negative) reference are still computed over ALL of the mouse's USVs, so
        the category choice changes only which onsets count as positive events.
        Ignored in 'bout_onset' and 'state' modes, because the mixture-model inter-syllable-
        interval threshold used for bout grouping is calibrated on the all-USV
        interval distribution and would mis-group a category-sparsified
        sequence; in those modes all categories are pooled as before. If None
        (default), all USV categories are pooled (original behavior).
    target_classes : list or tuple, default ('usv',)
        Which vocal classes are the POSITIVE source (``model_params.onset_target_classes``),
        read from the summary's ``usv`` / ``squeak`` booleans (written by ``detect-usv-squeaks``,
        ``os_utils.vocal_class_mask``): any non-empty list of ``"usv"`` (pure USV),
        ``"squeak"`` (pure squeak) and ``"overlap"`` (a USV and a squeak in one segment),
        e.g. ``["usv"]`` (pure USVs, the default) or ``["usv", "overlap"]`` (every
        segment holding a USV). Null booleans (a segment too short to score) are in no
        class. Applied before `target_category`, in every mode: the positives of
        'bout_onset', 'individual' and 'bout_offset' are the target classes' onsets
        (bouts are grouped from them alone, so a vocalization of another class between
        two target ones neither joins nor splits a bout), and the 'state' labels are
        the target classes' occupancy. The bout modes accept ``"usv"`` and
        ``"overlap"`` only (``BOUT_TARGET_CLASSES``): squeaks do not come in bouts.
        Each positive row is timed as the event the target classes ask about
        (``os_utils.vocal_event_times``): a pure squeak by its squeak extent
        ``squeak_start`` / ``squeak_end``, a pure USV by its segment ``start`` / ``stop``,
        and an overlap segment by the squeak extent when the classes name ``"squeak"``
        but not ``"usv"``, by the segment when they name ``"usv"`` but not ``"squeak"``,
        and by the union of the two otherwise; target classes that can time a row by
        the squeak extent raise when a summary has no ``squeak_start`` / ``squeak_end``.
        Silence is NOT class-specific: a negative epoch's window and a 'bout_onset'
        positive's clean history must both be free of every vocalization of the target
        animal (any class) and of every vocalization assigned to no animal, so the two
        groups are built under the same rule whatever the target classes; a
        squeak-bearing row blocks the union of its segment and its squeak extent
        (``os_utils.vocal_occupancy_times``), so every summary needs ``squeak_start`` /
        ``squeak_end``.
    predictor_classes : list or tuple, default (('usv',),)
        ``vocal_features.vocal_predictor_classes``: a list of class lists, each one
        predictor trace (``build_vocal_predictor_signals``, named by ``vocal_trace_name``),
        e.g. ``[["usv"]]`` or ``[["usv", "overlap"], ["squeak"]]``. Needed when
        `vocal_output_type` is set.
    negative_scheme : str, optional
        ``'bout_offset'`` mode only. ``'cross_bout'``: each positive (the last
        vocalization's offset of a bout) is paired with an interior vocalization's offset from
        ANOTHER bout of the same session, at the same time since that bout's
        onset (within ``time_since_bout_onset_tolerance``), in a bout that kept
        singing for at least ``min_singing_after_negative`` seconds after it;
        one negative per positive, nearest in position, each candidate used
        once, at most ``max_negatives_per_bout`` from any one bout; positives
        without a partner are dropped so the two groups share the same
        distribution of time since bout onset. ``'within_bout'``: the negative
        is the latest interior vocalization of the SAME bout that ended at least
        ``min_singing_after_negative`` seconds before the bout's end; bouts too
        short to hold one contribute nothing. Either way the pairing is only
        the sampling recipe -- the returned groups are pooled downstream.
    min_singing_after_negative : int or float, optional
        ``'bout_offset'`` mode only: seconds the bout must keep singing after a
        negative's vocalization, so the negative sits outside the ending itself.
    time_since_bout_onset_tolerance : int or float, optional
        ``'bout_offset'`` / ``'cross_bout'`` only: how far apart, in seconds, the
        time since bout onset of a positive and its negative may be.
    max_negatives_per_bout : int or None, optional
        ``'bout_offset'`` / ``'cross_bout'`` only: cap on negatives drawn from
        one bout; ``None`` means no cap.
        ``'bout_offset'`` targets the END of a bout: positives are the offsets
        of the last vocalization of every bout with at least ``min_usv_per_bout``
        vocalizations, with no clean-history or clean-future requirement (the
        inter-vocalization threshold already defines the end), and negatives are
        interior-vocalization offsets chosen by ``negative_scheme``.

    Returns
    -------
    usv_data_dict : dict
        Nested dictionary: session - mouseID - data.
        Per-mouse keys exported:
            'start': np.array of positive-source USV start times (category-filtered
                only in 'individual' mode with a target_category, else all USVs).
            'stop': np.array of positive-source USV stop times.
            'continuous_vocal_signals': dict of continuous predictor traces, one per
                entry of `predictor_classes` ('<name>_event'/'<name>_rate'/'<name>_cat_X').
            'positive_events': unbalanced array of POSITIVE event times (seconds).
            'negative_events': unbalanced array of NEGATIVE (no-USV) event times (seconds).
            'usv_count': raw binary occupancy trace (0/1) of the target classes ('state' labels).
            'usv_rate': Gaussian-smoothed density trace of the target classes.
    """

    validate_vocal_class_settings(target_classes, predictor_classes if vocal_output_type else None, prediction_mode)

    # Labels are needed only by the per-category predictor traces and by the
    # single-category onset target ('individual' mode); either one without a label
    # column stops here, before any session is read, instead of silently building
    # no category traces or falling back to all USVs.
    category_purposes = []
    if vocal_output_type in CATEGORY_PREDICTOR_TYPES:
        category_purposes.append(f"vocal_features.vocal_predictor_type '{vocal_output_type}'")
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

        usv_summary_data = read_usv_summary_table(csv_path, csv_sep)

        for category_purpose in category_purposes:
            require_usv_category_column(category_column, category_purpose,
                                        summary_columns=usv_summary_data.columns, source=str(csv_path))
        if exclude_noise_usvs:
            usv_summary_data = drop_noise_usvs(usv_summary_data, Path(csv_path).name)[0]
        if any(column not in usv_summary_data.columns for column in VOCAL_FLAG_COLUMNS):
            raise ValueError(
                f"{Path(csv_path).name} has no {list(VOCAL_FLAG_COLUMNS)} columns, so its USVs cannot be told from its "
                "squeaks for onset_target_classes / vocal_predictor_classes; run detect-usv-squeaks on the session."
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
            # A negative index would silently pick a component from the END of the
            # mixture (Python wrap-around) and set the wrong inter-bout threshold.
            if mixture_model_component_index < 0:
                msg = f"Invalid mixture_model_component_index {mixture_model_component_index} for {sex_label}: must be >= 0."
                raise ValueError(msg)
            try:
                if 'model_class' in params and params['model_class'] == 'ig':
                    comp_mu = params['mus'][mixture_model_component_index]
                    comp_lam = params['lambdas'][mixture_model_component_index]
                else:
                    comp_mean = params['means'][mixture_model_component_index]
                    comp_sd = params['sds'][mixture_model_component_index]
            except IndexError as error:
                msg = f"Invalid mixture_model_component_index {mixture_model_component_index} for {sex_label}."
                raise ValueError(msg) from error

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

            # Positive-event source: the rows of the target classes
            # (`target_classes`, read from the usv / squeak booleans; null booleans
            # match no class) and, when a single target USV category is requested
            # ('individual' mode only), of that category. The predictor traces are
            # built from their own class lists (`predictor_classes`) and the
            # silence reference from every class, so neither depends on the target.
            # Each row is timed as the event the target classes ask about
            # (`vocal_event_times`): a pure squeak by its squeak extent, an overlap
            # segment by the squeak extent, the segment or their union as the classes
            # ask about squeaks, USVs or both; re-sorted by that onset.
            typed_source_df = vocal_event_times(
                mouse_usvs_df.filter(vocal_class_mask(mouse_usvs_df, target_classes, Path(csv_path).name)),
                target_classes, Path(csv_path).name,
            )
            if target_category is not None and prediction_mode == 'individual':
                # The column's presence was checked above, when the summary was read.
                positive_source_df = typed_source_df.filter(pls.col(category_column) == target_category)
            else:
                positive_source_df = typed_source_df

            usv_data_dict[session_id][mouse_name]['start'] = np.array(positive_source_df['start'])
            usv_data_dict[session_id][mouse_name]['stop'] = np.array(positive_source_df['stop'])

            # The silence reference: every vocalization of this mouse (any class) and
            # every one assigned to no animal. Negative epochs must be free of all of
            # them, and so must a 'bout_onset' positive's clean history. A squeak-bearing
            # row spans the union of its segment and its squeak extent
            # (`vocal_occupancy_times`), which may reach past the segment.
            all_usvs_df = vocal_occupancy_times(
                usv_summary_data.filter((pls.col('emitter').is_null()) | (pls.col('emitter') == mouse_name)),
                Path(csv_path).name,
            )
            usv_start_mouse_and_uncategorized = np.array(all_usvs_df['start'])
            usv_stop_mouse_and_uncategorized = np.array(all_usvs_df['stop'])

            session_fps = camera_fps_dict[session_id]
            session_duration_frames = features_dict[session_id].shape[0]

            # Occupancy of the target classes ('state'-mode labels): the binary
            # trace `usv_count` and its Gaussian-smoothed `usv_rate`, built once from
            # the class-filtered rows (`typed_source_df`, never the category-filtered
            # positive source, so an onset category filter cannot leak into the
            # 'state' labels). Smoothing the shared occupancy is bit-identical to a
            # second `_generate_vocal_trace` call with the smoothing sd.
            usv_frame_events = _build_binary_occupancy(typed_source_df['start'].to_numpy(),
                                                       typed_source_df['stop'].to_numpy(),
                                                       session_duration_frames, session_fps)

            if proportion_smoothing_sd is not None and proportion_smoothing_sd > 0:
                usv_frame_rate = _smooth_occupancy(usv_frame_events, proportion_smoothing_sd)
            else:
                usv_frame_rate = usv_frame_events

            usv_data_dict[session_id][mouse_name]['usv_count'] = usv_frame_events
            usv_data_dict[session_id][mouse_name]['usv_rate'] = usv_frame_rate

            # Predictor traces, one per entry of `predictor_classes`.
            usv_data_dict[session_id][mouse_name]['continuous_vocal_signals'] = build_vocal_predictor_signals(
                mouse_usvs=mouse_usvs_df, predictor_classes=predictor_classes, vocal_output_type=vocal_output_type,
                category_column=category_column, duration_frames=session_duration_frames, fps=session_fps,
                smooth_sd=proportion_smoothing_sd, source=Path(csv_path).name)

            session_duration_sec = session_duration_frames / session_fps

            ### Mode 1: 'bout_onset' (both USV and no-USV pre-bout periods must be clean)
            if prediction_mode == 'bout_onset':

                # Get USV events (positive class)
                starts = usv_data_dict[session_id][mouse_name]['start']
                stops = usv_data_dict[session_id][mouse_name]['stop']

                valid_bouts = []

                if len(starts) > 0:
                    # Logic: filter using IBI threshold
                    bout_start_indices, bout_end_indices = _group_vocalizations_into_bouts(starts, stops, ibi_threshold)

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

                        # B. Clean history under the negatives' rule: no vocalization of
                        # this mouse (any class, not only the target classes) and none
                        # assigned to no animal may end within filter_history before the
                        # onset or start before it and still be running. Checking only
                        # the target classes would let, e.g., a squeak sit in a positive's
                        # history while every negative is free of squeaks.
                        if _history_is_dirty(bout_start_time, usv_start_mouse_and_uncategorized,
                                             usv_stop_mouse_and_uncategorized, filter_history):
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

                # Keep each clean onset only if zero USVs (any source) fall in
                # [t_onset, t_onset + usv_bout_time) (`_future_window_is_silent`).
                usv_events_negative = all_clean_onsets[
                    _future_window_is_silent(all_clean_onsets, all_usv_starts, usv_bout_time)]

            ### Mode 2: 'individual' (USV pre-vocalization periods can be "dirty" and no-USV must be clean)
            elif prediction_mode == 'individual':
                usv_starts_filter_bool = (usv_data_dict[session_id][mouse_name]['start'] > filter_history)
                usv_events_positive = usv_data_dict[session_id][mouse_name]['start'][usv_starts_filter_bool]

                all_clean_onsets = _get_clean_tiled_epochs(usv_start_mouse_and_uncategorized,
                                                           usv_stop_mouse_and_uncategorized,
                                                           filter_history,
                                                           session_duration_sec)

                # The same future-window rule as 'bout_onset': a negative must be
                # followed by `usv_bout_time` of silence (any source), or a tile with
                # a clean history could end milliseconds before a positive onset.
                usv_events_negative = all_clean_onsets[
                    _future_window_is_silent(all_clean_onsets, usv_start_mouse_and_uncategorized, usv_bout_time)]

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

            ### Mode 4: 'bout_offset' (the END of a bout against an interior vocalization, see _bout_offset_events)
            elif prediction_mode == 'bout_offset':
                starts = usv_data_dict[session_id][mouse_name]['start']
                stops = usv_data_dict[session_id][mouse_name]['stop']
                bout_start_indices, bout_end_indices = _group_vocalizations_into_bouts(starts, stops, ibi_threshold)
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
                        predictor_classes: list | tuple = (('usv',),),
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
        one trace per entry of `predictor_classes`, named ``<name>`` by
        ``vocal_trace_name`` (see ``build_vocal_predictor_signals``):
        - 'pooled_binary': binary occupancy (0/1), '<name>_event'.
        - 'pooled_rate': smoothed density, '<name>_rate'.
        - 'categories_rate': smoothed density per QLVM category, '<name>_cat_X'
          (an entry holding pure squeaks, which carry no category: '<name>_rate').
        - 'all_rate': '<name>_rate' plus the per-category '<name>_cat_X' traces.
    proportion_smoothing_sd : float, default 1.0
        Standard deviation for Gaussian smoothing (in frames).
    predictor_classes : list or tuple, default (('usv',),)
        ``vocal_features.vocal_predictor_classes``: a list of class lists, each one
        predictor trace (``build_vocal_predictor_signals``), e.g. ``[["usv"]]``.
        Needed when `vocal_output_type` is set.
    exclude_noise_usvs : bool, optional
        Whether to drop the segments ``detect_usv_noise`` flagged as holding no
        vocalization (default True). A summary without the ``noise`` column raises.
    manifold_column_names : list, optional
        Ordered list of column names in the USV summary CSV that encode each USV's
        coordinates on the continuous acoustic manifold. If any of the configured
        columns is missing from the CSV for a given session/mouse, no continuous
        targets are written for that mouse. When None or empty, continuous target
        extraction is skipped entirely. Vocalizations whose coordinates are null / NaN in
        any configured column (vocalizations the embedding could not place) are dropped
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
            'continuous_onsets': np.array of start times for valid USVs (used for continuous models);
                    for the squeak map (``SQUEAK_MAP_COLUMNS``) the squeaks' own onsets
                    (``squeak_start``, ``os_utils.vocal_event_times``).
            'continuous_targets': np.array of shape (N, D) stacking the configured
                manifold columns in the order given by `manifold_column_names`.
            'continuous_category': np.array of per-USV category labels, aligned 1:1
                with 'continuous_onsets': the regular map's QLVM category
                (os_utils.QLVM_CATEGORY_COLUMN, 'qlvm_category') whatever map the
                manifold columns come from, because the categories are defined on
                the regular map only and label each vocalization, not a map. Present only
                when that column exists in the source CSV (a summary whose
                categories assign-qlvm-categories has not written yet has none).
    """

    # Per-USV labels are needed by the categorical paths only: the multinomial /
    # binomial category models (no manifold columns), a `target_category`, and the
    # per-category predictor traces. The continuous manifold path reads the torus
    # coordinates alone, so there a null `category_column` simply means "no
    # category packets". A categorical path without labels stops here, before any
    # session is read.
    validate_vocal_class_settings(None, predictor_classes if vocal_output_type else None, None)
    category_purposes = []
    if not manifold_column_names:
        category_purposes.append("The USV category models (multinomial / binomial)")
    if target_category is not None:
        category_purposes.append(f"The binomial target category {target_category}")
    if vocal_output_type in CATEGORY_PREDICTOR_TYPES:
        category_purposes.append(f"vocal_features.vocal_predictor_type '{vocal_output_type}'")
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

        usv_summary_data = read_usv_summary_table(csv_path, csv_sep)

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
            # category the caller models can vary without changing which rows are real vocalizations.
            if exclude_noise_usvs:
                mouse_usvs = drop_noise_usvs(mouse_usvs, f"{session_id} ({mouse_name})")[0]

            # The predictor traces cover the whole session, history period included, so
            # an event shortly after `filter_history` still sees the vocalizations in its
            # history window (the onset and bout loaders build theirs the same way).
            mouse_usvs_all = mouse_usvs
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
            unique_cats = mouse_usvs[category_column].drop_nulls().unique().to_list() if use_categories else []

            for cat_id in unique_cats:
                try:
                    cat_int = int(cat_id)
                except (ValueError, TypeError):
                    continue

                cat_df = mouse_usvs.filter(pls.col(category_column) == cat_id)
                usv_data_dict[session_id][mouse_name]['events_by_category'][cat_int] = np.sort(cat_df['start'].to_numpy())

            # Predictor traces, one per entry of `predictor_classes`.
            usv_data_dict[session_id][mouse_name]['continuous_vocal_signals'] = build_vocal_predictor_signals(
                mouse_usvs=mouse_usvs_all, predictor_classes=predictor_classes, vocal_output_type=vocal_output_type,
                category_column=category_column, duration_frames=session_duration_frames, fps=session_fps,
                smooth_sd=proportion_smoothing_sd, source=f"{session_id} ({mouse_name})")

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
                    # The squeak map places the squeak of every squeak-bearing segment
                    # (pure squeaks and overlap segments), so its onsets are the squeaks'
                    # own (`vocal_event_times` with the class group ("squeak",): every
                    # squeak-bearing row by its squeak extent, never by the overlap
                    # segment's USV onset), re-sorted and re-filtered against the history
                    # period on that onset; every other map places USVs and keeps the
                    # segment onsets.
                    if set(manifold_column_names) <= set(SQUEAK_MAP_COLUMNS):
                        manifold_usvs = vocal_event_times(mouse_usvs, ("squeak",), f"{session_id} ({mouse_name})")
                        manifold_usvs = manifold_usvs.filter(pls.col('start') > filter_history)
                    else:
                        manifold_usvs = mouse_usvs
                    # A vocalization the embedding could not place (outside the model's
                    # duration window, no SAM mask, no condition value) has null
                    # coordinates; it has no manifold target, so it is dropped here,
                    # together with its onset and labels, before anything downstream
                    # (the inverse-density KDE, the regressions, the GLM-HMM) sees a NaN.
                    # A column CSV inference read as text (all-null) casts to NaN too.
                    manifold_arrays = [
                        manifold_usvs[col].cast(pls.Float64, strict=False).fill_null(np.nan).to_numpy()
                        for col in manifold_column_names
                    ]
                    manifold_targets = np.column_stack(manifold_arrays)
                    placed = np.isfinite(manifold_targets).all(axis=1)
                    n_unplaced = int(np.count_nonzero(~placed))
                    if n_unplaced > 0:
                        print(f"  {session_id} ({mouse_name}): dropped {n_unplaced} of {placed.size} vocalizations with "
                              f"null/NaN manifold coordinates ({', '.join(manifold_column_names)}).")
                        n_unplaced_total += n_unplaced
                        n_unplaced_sessions += 1
                    usv_data_dict[session_id][mouse_name]['continuous_onsets'] = manifold_usvs['start'].to_numpy()[placed]
                    usv_data_dict[session_id][mouse_name]['continuous_targets'] = manifold_targets[placed]

                    # Per-USV category labels, the acoustic regions of
                    # downstream region-conditioned analyses (torus macro
                    # score, equal-region reweighting, model selection, CNN
                    # saliency, GLM-HMM targets). The label is the regular
                    # map's qlvm_category (R-1..R-k) for EVERY map -- qlvm and
                    # the four conditional maps alike: the categories are defined
                    # on the regular map only (os_utils.QLVM_CATEGORY_MAP) and
                    # are a property of the vocalization, so a conditional map's vocalizations
                    # carry the same labels. Stored as a plain numpy array
                    # aligned 1:1 with continuous_onsets / continuous_targets
                    # above, only when the column is present in the source CSV;
                    # an absent label array signals "this USV summary carries
                    # no categories yet" (assign-qlvm-categories not run).
                    if QLVM_CATEGORY_COLUMN in manifold_usvs.columns:
                        usv_data_dict[session_id][mouse_name]['continuous_category'] = (
                            manifold_usvs[QLVM_CATEGORY_COLUMN].to_numpy()[placed]
                        )

    if manifold_column_names:
        print(f"Manifold targets: dropped {n_unplaced_total} vocalizations with null/NaN coordinates in total "
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
                               predictor_classes: list | tuple = (('usv',),),
                               target_classes: list | tuple = ('usv',),
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
        - Keeps the vocalizations of the target classes (`target_classes`,
          ``model_params.onset_target_classes``), so a bout here is the same as in the
          onset pipeline: a vocalization of another class (e.g. a squeak between two
          USVs) neither joins nor splits a bout.
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
        one trace per entry of `predictor_classes`, named ``<name>`` by
        ``vocal_trace_name`` (see ``build_vocal_predictor_signals``):
        - 'pooled_binary': binary occupancy (0/1), '<name>_event'.
        - 'pooled_rate': smoothed density, '<name>_rate'.
        - 'categories_rate': smoothed density per QLVM category, '<name>_cat_X'
          (an entry holding pure squeaks, which carry no category: '<name>_rate').
        - 'all_rate': '<name>_rate' plus the per-category '<name>_cat_X' traces.
    predictor_classes : list or tuple, default (('usv',),)
        ``vocal_features.vocal_predictor_classes``: a list of class lists, each one
        predictor trace (``build_vocal_predictor_signals``), e.g. ``[["usv"]]``.
        Needed when `vocal_output_type` is set.
    target_classes : list or tuple, default ('usv',)
        ``model_params.onset_target_classes``: the vocal classes bouts are grouped
        from (``"usv"`` and ``"overlap"`` only, ``BOUT_TARGET_CLASSES``), each row timed
        as the event the classes ask about (``os_utils.vocal_event_times``: with
        ``"usv"`` named, an overlap segment keeps its segment times; ``["overlap"]``
        alone takes the union of the segment and its squeak extent). The predictor
        traces are built from their own lists (`predictor_classes`), not from these.
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
            'bout_offsets': np.array of stop times (seconds), the last USV's stored stop
                bit for bit; ``onset + duration`` can fall one ulp before it.
            'continuous_vocal_signals': dict containing generated arrays (e.g., 'usv_rate').
    """

    # Per-category traces need a label column; without one they stop here, before
    # any session is read, instead of silently building no category predictors.
    validate_vocal_class_settings(target_classes, predictor_classes if vocal_output_type else None, 'variable_length_bouts')
    category_traces = vocal_output_type in CATEGORY_PREDICTOR_TYPES
    category_purpose = f"vocal_features.vocal_predictor_type '{vocal_output_type}'"
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

        usv_summary_data = read_usv_summary_table(csv_path, csv_sep)

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
            # A negative index would silently pick a component from the END of the
            # mixture (Python wrap-around) and set the wrong inter-bout threshold.
            if mixture_model_component_index < 0:
                msg = f"Invalid mixture_model_component_index {mixture_model_component_index} for {sex_label}: must be >= 0."
                raise ValueError(msg)
            try:
                if 'model_class' in params and params['model_class'] == 'ig':
                    comp_mu = params['mus'][mixture_model_component_index]
                    comp_lam = params['lambdas'][mixture_model_component_index]
                else:
                    comp_mean = params['means'][mixture_model_component_index]
                    comp_sd = params['sds'][mixture_model_component_index]
            except IndexError as error:
                msg = f"Invalid mixture_model_component_index {mixture_model_component_index} for {sex_label}."
                raise ValueError(msg) from error

            if 'model_class' in params and params['model_class'] == 'ig':
                ibi_threshold = _calculate_ibi_threshold_ig(comp_mu, comp_lam, mixture_model_z_score)
            else:
                ibi_threshold = _calculate_ibi_threshold(comp_mean, comp_sd, mixture_model_z_score)

            usv_data_dict[session_id][mouse_name] = {
                'bout_onsets': [],
                'bout_durations': [],
                'bout_offsets': [],
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

            # Predictor traces, one per entry of `predictor_classes`.
            usv_data_dict[session_id][mouse_name]['continuous_vocal_signals'] = build_vocal_predictor_signals(
                mouse_usvs=mouse_usvs, predictor_classes=predictor_classes, vocal_output_type=vocal_output_type,
                category_column=category_column, duration_frames=session_duration_frames, fps=session_fps,
                smooth_sd=proportion_smoothing_sd, source=f"{session_id} ({mouse_name})")

            # Bouts are grouped from the target classes only (null booleans match no class),
            # each row timed as the event the classes ask about (`vocal_event_times`) and
            # re-sorted by that onset, so a squeak bout's gaps run between the squeaks.
            mouse_usvs = vocal_event_times(
                mouse_usvs.filter(vocal_class_mask(mouse_usvs, target_classes, f"{session_id} ({mouse_name})")),
                target_classes, f"{session_id} ({mouse_name})",
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
                usv_data_dict[session_id][mouse_name]['bout_offsets'].append(bout_end_time)
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
