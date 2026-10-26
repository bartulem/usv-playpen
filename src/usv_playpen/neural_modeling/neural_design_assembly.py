"""
@author: bartulem
Turning a unit's sessions into the per-frame design the encoding models are fitted on.

Loads a unit's sessions, builds the pooled z-scored predictor time series with the behavioural
pipeline's own loaders and column utilities, and does the frame bookkeeping the models are fitted on:

- quiet anchors     — frames with no vocalization from any emitter (per ``clean_against``) from
                      ``history_pre_seconds`` before to ``clean_post_seconds`` after;
- vocal frames      — frames inside the recorded animal's own USVs, with no cleanliness condition;
- per-USV events    — one row per analysed USV with its torus position, acoustic region, silent gap
                      and the unit's spike count in the peri-onset window, plus quiet tiles of the same
                      width as the negative class;
- blocks            — a unit recorded in one session is tested on contiguous time blocks of it, each
                      standing in for a session (``<session>#block<i>``); every assembler clips what it
                      returns to the block's window.

The modelling itself lives elsewhere; this module only decides what gets modelled.
"""

from __future__ import annotations

import multiprocessing
import os
from pathlib import Path

import h5py
import numpy as np
import polars as pls
from numpy.lib.stride_tricks import sliding_window_view

from ..analyses.compute_behavioral_features import FeatureZoo
from ..analyses.unit_triage_aggregator import _parse_unit_id
from ..modeling.load_input_files import load_behavioral_feature_data
from ..modeling.modeling_utils import (
    harmonize_session_columns,
    resolve_mouse_roles,
    select_kinematic_columns,
    zscore_features_across_sessions,
)
from ..os_utils import configure_path
from .neural_vocal_occurrence import counts_in_windows, quiet_tile_edges

#: The environment variables through which the common BLAS and OpenMP runtimes take their thread count.
BLAS_THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                         "NUMEXPR_NUM_THREADS")


def worker_process_context() -> multiprocessing.context.BaseContext:
    """
    Description
    -----------
    The multiprocessing context for a pool of numerical workers, with every worker's BLAS held to ONE
    thread -- so that ``n_workers`` bounds the cores a pool takes, which on its own it does not.

    Each worker that imports numpy otherwise receives a BLAS pool sized to the whole machine, so the
    pool oversubscribes by a factor of ``n_workers``. The variables are set with ``setdefault`` in this
    process's environment, so an explicit shell value wins; the context is spawn rather than fork
    because a forked child inherits the parent's already-initialised BLAS pool.

    Parameters
    ----------
    None

    Returns
    -------
    context (multiprocessing.context.BaseContext)
        The spawn context, to pass as ``mp_context`` to a ``ProcessPoolExecutor``.
    """

    for name in BLAS_THREAD_VARIABLES:
        os.environ.setdefault(name, "1")
    return multiprocessing.get_context("spawn")


def quiet_anchor_frames(usv_starts_seconds: np.ndarray, usv_stops_seconds: np.ndarray, n_frames: int,
                        fps: float, history_pre_seconds: float,
                        clean_post_seconds: float) -> np.ndarray:
    """
    Description
    -----------
    Frames whose window ``[t/fps - history_pre, t/fps + clean_post]`` contains no vocalization at all.

    Built by rasterizing each USV's forbidden span into a boolean mask and keeping what falls outside every
    one of them. A USV forbids ``[start - clean_post, stop + history_pre]``, which is the same statement
    read from the USV's side rather than the frame's. Frames without a full pre-history or post-buffer
    inside the recording are dropped, since their window is not fully observed.

    Pass starts and stops for EVERY emitter, not just the focal animal: a quiet anchor is meant to be a
    moment when nothing was audible.

    Parameters
    ----------
    usv_starts_seconds (np.ndarray)
        USV start times in seconds, all emitters.
    usv_stops_seconds (np.ndarray)
        USV stop times in seconds, all emitters.
    n_frames (int)
        Session frame count.
    fps (float)
        Camera frame rate.
    history_pre_seconds (float)
        Silence required before the frame, and the length of the predictor's history window.
    clean_post_seconds (float)
        Silence required after the frame.

    Returns
    -------
    frames (np.ndarray)
        Sorted integer quiet-anchor frame indices.
    """

    forbidden = np.zeros(n_frames, dtype=bool)
    for start, stop in zip(usv_starts_seconds, usv_stops_seconds, strict=True):
        lo = max(int(np.floor((start - clean_post_seconds) * fps)), 0)
        hi = min(int(np.ceil((stop + history_pre_seconds) * fps)), n_frames)
        if hi > lo:
            forbidden[lo:hi] = True

    history_frames = int(np.floor(history_pre_seconds * fps))
    post_frames = int(np.ceil(clean_post_seconds * fps))
    eligible = ~forbidden
    eligible[:history_frames] = False
    if post_frames > 0:
        eligible[n_frames - post_frames:] = False
    return np.flatnonzero(eligible)


def vocal_span_frames(starts_seconds: np.ndarray, stops_seconds: np.ndarray, fps: float, n_frames: int,
                      n_lags: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Description
    -----------
    Expand each focal USV into the frames of its ``[start, stop)`` span, keeping only USVs whose frames
    all carry a full in-bounds predictor history. Spans are ragged, so they come back flattened with an
    offset pointer rather than padded.

    The rounding carries a 1e-9 tolerance, so a floating-point overshoot such as
    ``220.00000000000003`` does not add a frame the USV never occupied.

    Parameters
    ----------
    starts_seconds (np.ndarray)
        Focal USV start times in seconds.
    stops_seconds (np.ndarray)
        Focal USV stop times in seconds.
    fps (float)
        Camera frame rate.
    n_frames (int)
        Session frame count.
    n_lags (int)
        Predictor history length in frames; a frame needs this much history in bounds.

    Returns
    -------
    kept (np.ndarray)
        Boolean mask over the input USVs, marking those retained.
    flat_frames (np.ndarray)
        Concatenated span frame indices of the kept USVs.
    pointer (np.ndarray)
        Offsets of length ``n_kept + 1``; USV ``i`` owns ``flat_frames[pointer[i]:pointer[i + 1]]``.
    """

    starts = np.asarray(starts_seconds, dtype=np.float64)
    stops = np.asarray(stops_seconds, dtype=np.float64)
    tolerance = 1e-9
    lo = np.floor(starts * fps + tolerance).astype(np.int64)
    hi = np.maximum(np.ceil(stops * fps - tolerance).astype(np.int64), lo + 1)
    kept = (lo >= n_lags - 1) & (hi <= n_frames)
    blocks, offsets = [], [0]
    for span_start, span_stop in zip(lo[kept], hi[kept], strict=True):
        blocks.append(np.arange(span_start, span_stop, dtype=np.int64))
        offsets.append(offsets[-1] + int(span_stop - span_start))
    flat = np.concatenate(blocks) if blocks else np.zeros(0, dtype=np.int64)
    return kept, flat, np.asarray(offsets, dtype=np.int64)


def silent_gap_before_vocalization(starts_seconds: np.ndarray, stops_seconds: np.ndarray) -> np.ndarray:
    """
    Description
    -----------
    Silent gap in seconds from the previous focal vocalization's offset to each vocalization's onset, with
    infinity for the first one.

    Kept continuous rather than thresholded into bout-initial and mid-bout, so bout structure stays
    checkable after the fact.

    Parameters
    ----------
    starts_seconds (np.ndarray)
        Focal vocalization start times in seconds, sorted.
    stops_seconds (np.ndarray)
        Focal vocalization stop times in seconds, sorted.

    Returns
    -------
    gaps (np.ndarray)
        Silent gap preceding each vocalization.
    """

    gaps = np.full(starts_seconds.size, np.inf, dtype=np.float64)
    if starts_seconds.size > 1:
        gaps[1:] = starts_seconds[1:] - stops_seconds[:-1]
    return gaps


def onset_history_anchor_frames(starts_seconds: np.ndarray, fps: float) -> np.ndarray:
    """
    Description
    -----------
    The frame each vocalization's behaviour history ENDS at: the frame before the onset frame,
    ``rint(start * fps) - 1``, so the lagged window built at it is strictly pre-onset.

    The same convention is used where the cohort behaviour control is fitted and where its
    prediction is applied by the nested decoding.

    Parameters
    ----------
    starts_seconds (np.ndarray)
        Vocalization onset times in seconds, on the tracking clock.
    fps (float)
        Camera frame rate.

    Returns
    -------
    anchors (np.ndarray)
        Integer anchor frame per vocalization (may be negative or past the session for onsets near its
        edges; the caller drops those that lack a full in-bounds history).
    """

    return np.rint(np.asarray(starts_seconds, dtype=np.float64) * fps).astype(np.int64) - 1


def spike_labels_at_frames(spike_frames: np.ndarray, anchor_frames: np.ndarray,
                           n_frames: int) -> np.ndarray:
    """
    Description
    -----------
    Binary spike labels at the requested frames.

    At the tracking rate a frame lasts about 6.7 ms, so the refractory period caps it at one spike and
    the binary label loses nothing. Wider bins are modelled as counts instead.

    Parameters
    ----------
    spike_frames (np.ndarray)
        Integer spike-frame indices.
    anchor_frames (np.ndarray)
        Frames to label.
    n_frames (int)
        Session frame count.

    Returns
    -------
    labels (np.ndarray)
        0/1 labels aligned to ``anchor_frames``.
    """

    occupancy = np.zeros(n_frames, dtype=np.int8)
    valid = (spike_frames >= 0) & (spike_frames < n_frames)
    occupancy[spike_frames[valid]] = 1
    return occupancy[anchor_frames].astype(np.float64)


def lagged_design(feature_time_series: np.ndarray, anchor_frames: np.ndarray,
                  n_lags: int) -> np.ndarray:
    """
    Description
    -----------
    The lagged predictor at each anchor frame: every feature's preceding ``n_lags`` samples, laid out as one
    row per anchor.

    Built with a sliding window view, so no per-lag copy is made. The window ending at frame ``t``
    begins at index ``t - n_lags + 1``.

    Parameters
    ----------
    feature_time_series (np.ndarray)
        ``(n_frames, n_features)`` predictor time series.
    anchor_frames (np.ndarray)
        Frames to build rows for; every one needs ``n_lags - 1`` frames of history behind it.
    n_lags (int)
        History length in frames.

    Returns
    -------
    design (np.ndarray)
        ``(n_anchors, n_features * n_lags)`` design matrix.
    """

    if np.any(anchor_frames < n_lags - 1):
        msg = (f"all anchor frames must be >= n_lags-1 ({n_lags - 1}); "
               f"got min {int(anchor_frames.min())}.")
        raise ValueError(msg)
    start_index = anchor_frames - (n_lags - 1)
    return np.hstack([sliding_window_view(feature_time_series[:, feature], n_lags)[start_index]
                      for feature in range(feature_time_series.shape[1])])


def subsample_quiet_anchors(quiet_frames: np.ndarray, spike_frames: np.ndarray, n_frames: int,
                            negatives_per_positive: int, rng, max_total: int | None = None) -> tuple:
    """
    Description
    -----------
    Class-balanced, memory-bounded subsampling of quiet anchors, with the log-prior offset correction that
    keeps the fit calibrated afterwards.

    The minority class is kept whole and the majority subsampled to ``negatives_per_positive`` times
    its size, so a high-firing unit, where the zeros are scarcer, is handled correctly too. Subsampling
    shifts the log-odds by the log ratio of the two keep-fractions; that constant is returned as a
    per-sample offset to add during the fit, so the intercept stays on the population scale
   . ``max_total`` scales both classes down together to bound memory, and the offset
    absorbs that too.

    Parameters
    ----------
    quiet_frames (np.ndarray)
        Candidate quiet anchor frames.
    spike_frames (np.ndarray)
        Integer spike-frame indices.
    n_frames (int)
        Session frame count.
    negatives_per_positive (int)
        Majority-to-minority ratio to subsample down to.
    rng (np.random.Generator)
        Seeded generator.
    max_total (int)
        Absolute cap on retained anchors, or None.

    Returns
    -------
    anchors (np.ndarray)
        Retained anchor frames.
    labels (np.ndarray)
        Their 0/1 spike labels.
    offsets (np.ndarray)
        Per-sample log-prior offset to add to the linear predictor during fitting.
    n_positive_total (int)
        Positives available before subsampling.
    n_negative_total (int)
        Negatives available before subsampling.
    """

    labels_full = spike_labels_at_frames(spike_frames, quiet_frames, n_frames)
    positive = quiet_frames[labels_full > 0.5]
    negative = quiet_frames[labels_full <= 0.5]
    keep_positive, keep_negative = positive.size, negative.size
    if positive.size >= negative.size:
        keep_positive = (min(positive.size, negatives_per_positive * negative.size)
                         if negative.size else positive.size)
    else:
        keep_negative = (min(negative.size, negatives_per_positive * positive.size)
                         if positive.size else negative.size)

    if max_total is not None and (keep_positive + keep_negative) > max_total:
        scale = max_total / (keep_positive + keep_negative)
        keep_positive = max(1, int(keep_positive * scale))
        keep_negative = max(1, int(keep_negative * scale))

    selected_positive = (rng.choice(positive, keep_positive, replace=False)
                         if keep_positive < positive.size else positive)
    selected_negative = (rng.choice(negative, keep_negative, replace=False)
                         if keep_negative < negative.size else negative)
    fraction_positive = selected_positive.size / positive.size if positive.size else 1.0
    fraction_negative = selected_negative.size / negative.size if negative.size else 1.0
    log_prior_offset = (float(np.log(fraction_positive / fraction_negative))
                        if (fraction_positive > 0 and fraction_negative > 0) else 0.0)

    anchors = np.concatenate([selected_positive, selected_negative])
    labels = np.concatenate([np.ones(selected_positive.size), np.zeros(selected_negative.size)])
    offsets = np.full(anchors.size, log_prior_offset)
    return anchors, labels, offsets, int(positive.size), int(negative.size)


def load_unit_spike_frames(data_root: str, session_id: str, unit_id: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Description
    -----------
    Load a unit's spike train for one session from its ``cluster_data`` ``.npy`` (``unit_id`` is the file
    stem; the probe folder ``imec{i}`` is parsed from the id). Returns spike times in seconds (row 0) and
    integer spike-frame indices (row 1, ``np.round`` -> int), matching the two rows used by the vocal
    decoding (seconds, ``searchsorted``) and the kinematic encoding (frames, per-frame binning).

    Parameters
    ----------
    data_root (str)
        The ``Data`` root (e.g. ``/mnt/falkner/Bartul/Data``), resolved via ``configure_path``.
    session_id (str)
        Session directory basename (e.g. ``'20241107_135544'``).
    unit_id (str)
        The unit id = the ``.npy`` file stem (e.g. ``'imec0_cl0001_ch019_good'``).

    Returns
    -------
    spike_seconds, spike_frames (tuple[np.ndarray, np.ndarray])
        Spike times in seconds and integer spike-frame indices.
    """

    imec, _cluster_num, _peak_channel, _kslabel = _parse_unit_id(unit_id)
    npy_path = Path(configure_path(data_root)) / session_id / "ephys" / f"imec{imec}" / "cluster_data" / f"{unit_id}.npy"
    if not npy_path.exists():
        msg = f"spike file not found: {npy_path}"
        raise FileNotFoundError(msg)
    arr = np.load(npy_path)
    spike_seconds = np.asarray(arr[0, :], dtype=float)
    spike_frames = np.round(arr[1, :]).astype(np.int64)
    return spike_seconds, spike_frames


def load_session_usvs(data_root: str, session_id: str, csv_sep: str = ",") -> pls.DataFrame:
    """
    Description
    -----------
    Read a session's ``<session>_usv_summary.csv`` (carries ``start``/``stop``/``emitter``, the QLVM
    columns named by ``vocal_decoding.usv_manifold_column_names``, and raw acoustics).

    Parameters
    ----------
    data_root (str)
        The ``Data`` root.
    session_id (str)
        Session directory basename.
    csv_sep (str)
        CSV separator.

    Returns
    -------
    usv_df (pls.DataFrame)
        The USV summary table.
    """

    usv_path = Path(configure_path(data_root)) / session_id / "audio" / f"{session_id}_usv_summary.csv"
    if not usv_path.exists():
        msg = f"usv_summary not found: {usv_path}"
        raise FileNotFoundError(msg)
    return pls.read_csv(usv_path, separator=csv_sep, infer_schema_length=5000)


VOCALIZATION_SCOPES = ("analysed", "guard")


def filter_vocalizations(usv_df: pls.DataFrame, vocal_settings: dict, scope: str) -> pls.DataFrame:
    """
    Description
    -----------
    Drop the rows of a USV summary that the settings say are not vocalizations of the analysed kind.

    ``noise`` is not a vocalization, so it is removed at both scopes, including from the guard that
    decides which frames count as quiet. ``squeak`` is a real sound of a different type, outside the
    analysed USV set; whether it still makes surrounding frames non-quiet is set by
    ``squeaks_block_quiet`` (shipped true, so "quiet" means no sound of any kind).

    Parameters
    ----------
    usv_df (pls.DataFrame)
        A session's USV summary, as ``load_session_usvs`` returns it.
    vocal_settings (dict)
        The ``vocalization_settings`` block, carrying the three exclusion flags.
    scope (str)
        ``'analysed'`` for the USVs an analysis scores, ``'guard'`` for the set that decides
        cleanliness.

    Returns
    -------
    filtered (pls.DataFrame)
        The rows that survive, in their original order.
    """

    if scope not in VOCALIZATION_SCOPES:
        msg = f"scope must be one of {VOCALIZATION_SCOPES}, got {scope!r}"
        raise ValueError(msg)

    drop_noise = vocal_settings["exclude_noise"]
    drop_squeaks = vocal_settings["exclude_squeaks"]
    if scope == "guard" and vocal_settings["squeaks_block_quiet"]:
        drop_squeaks = False

    wanted = ([("noise", drop_noise)] if drop_noise else []) + ([("squeak", drop_squeaks)]
                                                                if drop_squeaks else [])
    for column, _flag in wanted:
        # Fail loudly: a summary predating the classifier would otherwise pass unfiltered.
        if column not in usv_df.columns:
            msg = (f"vocalization_settings asks to exclude '{column}' but the usv_summary has no such "
                   f"column. Available: {usv_df.columns}.")
            raise KeyError(msg)

    filtered = usv_df
    for column, _flag in wanted:
        filtered = filtered.filter(~pls.col(column).cast(pls.Boolean))
    return filtered


def emitter_names(track_names: list, recorded_mouse_id: str, spec: str) -> list:
    """
    Description
    -----------
    Resolve an emitter SPEC to the mouse-id strings it names.

    ``'self'`` is the recorded, probe-bearing animal, resolved by name from the unit's ``mouse_id``;
    ``'partner'`` is every other track name; ``'all'`` returns an empty list meaning no filter, so USVs
    with no assigned emitter are included too.

    Parameters
    ----------
    track_names (list)
        The session's ``track_names``.
    recorded_mouse_id (str)
        The unit's mouse; must appear in ``track_names``.
    spec (str)
        ``'self'`` | ``'partner'`` | ``'all'``.

    Returns
    -------
    names (list)
        The mouse ids the spec names, or an EMPTY list for ``'all'`` (no filter).
    """

    if spec == "all":
        return []
    if recorded_mouse_id not in track_names:
        msg = f"recorded mouse {recorded_mouse_id!r} is not in track_names {track_names}."
        raise ValueError(msg)
    if spec == "self":
        return [recorded_mouse_id]
    if spec == "partner":
        return [name for name in track_names if name != recorded_mouse_id]
    msg = f"emitter spec must be 'self' | 'partner' | 'all'; got {spec!r}."
    raise ValueError(msg)


def build_zscored_feature_frames(
        session_dirs: list[str],
        kinematic_features: dict,
        recorded_mouse_id: str,
        csv_sep: str = ",",
) -> tuple[dict, dict, dict, list]:
    """
    Description
    -----------
    Build the z-scored behavioural feature time series for the given sessions with the behavioural
    modeling chain: ``load_behavioral_feature_data`` -> per-session ``select_kinematic_columns`` ->
    ``harmonize_session_columns`` -> ``zscore_features_across_sessions``. No vocal-signal columns are
    added. The z-scoring pools all supplied sessions, and columns are renamed to ``self.``/``other.``
    in one canonical order so position-indexed access is consistent across sessions.

    Parameters
    ----------
    session_dirs (list[str])
        Session directory paths (``Data/<session_id>``), resolved via ``configure_path`` by the loader.
    kinematic_features (dict)
        The kinematic-features schema (egocentric / dyadic_pose / abs_features / ...).
    recorded_mouse_id (str)
        The unit's own animal. It becomes ``self.``; the remaining track name becomes ``other.``.
        Resolved by name and exclusion, never by slot; each session must hold exactly two tracks.
    csv_sep (str)
        CSV separator.

    Returns
    -------
    feature_frames, camera_fr_dict, mouse_names_dict, suffixes (tuple[dict, dict, dict, list])
        Per-session z-scored feature DataFrames (keyed by session_id), camera fps, mouse track names, and
        the ordered feature suffix list (the design's feature order).
    """

    resolved = [configure_path(p) for p in session_dirs]
    beh_dict, camera_fr_dict, mouse_names_dict = load_behavioral_feature_data(
        behavior_file_paths=resolved, csv_sep=csv_sep,
    )

    processed: dict = {}
    predictor_idx = target_idx = None
    for sess_id, session_df in beh_dict.items():
        if sess_id not in mouse_names_dict:
            continue
        # `self.` is the recorded animal by name and `other.` its partner by exclusion. The downstream
        # `resolve_mouse_roles` derives the partner as `abs(idx - 1)`, which holds only in a dyad, so
        # the dyad is asserted here.
        names = mouse_names_dict[sess_id]
        partner_names = emitter_names(names, recorded_mouse_id, "partner")
        if len(partner_names) != 1:
            msg = (f"{sess_id}: expected exactly one partner for recorded mouse {recorded_mouse_id!r}, "
                   f"got {partner_names} from track_names {names}; the self/other feature split "
                   f"assumes a dyad.")
            raise ValueError(msg)
        session_settings = {
            "model_params": {"model_predictor_mouse_index": names.index(partner_names[0])},
            "kinematic_features": kinematic_features,
        }
        session_predictor, session_target, p_name, t_name = resolve_mouse_roles(
            modeling_settings=session_settings, mouse_names_dict=mouse_names_dict, session_id=sess_id,
        )
        # The column harmonisation takes one pair of indices, so the animal must sit in the same slot
        # in every session of the unit.
        if predictor_idx is not None and (session_predictor, session_target) != (predictor_idx, target_idx):
            msg = (f"{sess_id}: {recorded_mouse_id!r} sits in a different track slot than in the "
                   f"unit's other sessions; the column harmonisation cannot mix slots.")
            raise ValueError(msg)
        predictor_idx, target_idx = session_predictor, session_target
        keep_cols = select_kinematic_columns(
            session_df_columns=session_df.columns, target_name=t_name, predictor_name=p_name,
            kin_settings=kinematic_features, predictor_idx=predictor_idx,
        )
        existing = [c for c in keep_cols if c in session_df.columns]
        processed[sess_id] = session_df.select(existing)

    processed, suffixes = harmonize_session_columns(
        processed_beh_dict=processed, mouse_names_dict=mouse_names_dict,
        target_idx=target_idx, predictor_idx=predictor_idx,
    )
    # Clip to the same physical ranges as the behavioural models before pooled z-scoring.
    processed = zscore_features_across_sessions(
        processed_beh_dict=processed, suffixes=suffixes,
        feature_bounds=FeatureZoo.feature_boundaries,
        abs_features=kinematic_features["abs_features"],
        smooth_abs_features=kinematic_features["smooth_abs_features"],
    )

    # Rename ego columns from raw mouse ids to `self.`/`other.` and reindex every session to one sorted
    # column order: the partner id differs between sessions, so without this the same array column
    # would point at different features in different sessions.
    canonical: dict = {}
    canonical_names: list[str] | None = None
    for sess_id, df in processed.items():
        # The dyad guard above has run for every session here, so the single partner index is safe.
        names = mouse_names_dict[sess_id]
        t_name = recorded_mouse_id
        p_name = emitter_names(names, recorded_mouse_id, "partner")[0]
        rename_map: dict = {}
        for col in df.columns:
            if col.startswith(f"{t_name}."):
                rename_map[col] = f"self.{col.split('.', 1)[1]}"
            elif col.startswith(f"{p_name}."):
                rename_map[col] = f"other.{col.split('.', 1)[1]}"
        renamed = df.rename(rename_map)
        ordered = sorted(renamed.columns)
        canonical[sess_id] = renamed.select(ordered)
        if canonical_names is None:
            canonical_names = ordered
        elif ordered != canonical_names:
            msg = (
                f"canonical column set for session {sess_id} differs from the reference: "
                f"only-in-{sess_id}={sorted(set(ordered) - set(canonical_names))}, "
                f"only-in-reference={sorted(set(canonical_names) - set(ordered))}"
            )
            raise ValueError(msg)
    return canonical, camera_fr_dict, mouse_names_dict, suffixes


#: Separator between a session id and a block index in a block identifier, ``<session>#block<i>``.
BLOCK_TAG = "#block"


def source_session(session_id: str) -> str:
    """
    Description
    -----------
    The recording session a session or block identifier refers to: the identifier itself for a
    session, the part before ``BLOCK_TAG`` for a block. Every load from disk goes through this.

    Parameters
    ----------
    session_id (str)
        A session directory basename or a block identifier.

    Returns
    -------
    source (str)
        The session directory basename.
    """

    return session_id.split(BLOCK_TAG, maxsplit=1)[0]


def session_block_ids(session_id: str, n_blocks: int) -> list[str]:
    """
    Description
    -----------
    The block identifiers a single recording session is split into, in time order.

    Parameters
    ----------
    session_id (str)
        Session directory basename.
    n_blocks (int)
        Number of blocks.

    Returns
    -------
    block_ids (list[str])
        ``[f"{session_id}#block0", ..., f"{session_id}#block{n_blocks - 1}"]``.
    """

    return [f"{session_id}{BLOCK_TAG}{index}" for index in range(n_blocks)]


def block_window_frames(n_frames: int, fps: float, n_blocks: int, gap_seconds: float) -> list[tuple[int, int]]:
    """
    Description
    -----------
    The frame windows of a session's blocks: ``n_blocks`` stretches of equal duration, with
    ``gap_seconds`` removed around every boundary between two blocks (half on each side), so nothing
    within ``gap_seconds`` of a neighbouring block is kept. The session's own start and end are not
    trimmed. Windows are half-open, ``[low, high)``.

    Parameters
    ----------
    n_frames (int)
        Session frame count.
    fps (float)
        Camera frame rate.
    n_blocks (int)
        Number of blocks.
    gap_seconds (float)
        Separation kept between neighbouring blocks.

    Returns
    -------
    windows (list[tuple[int, int]])
        ``(low, high)`` frame bounds per block, in time order.
    """

    edges = np.round(np.linspace(0, n_frames, n_blocks + 1)).astype(np.int64)
    half = int(np.ceil(gap_seconds * fps / 2.0))
    windows = []
    for index in range(n_blocks):
        low = int(edges[index]) + (half if index > 0 else 0)
        high = int(edges[index + 1]) - (half if index < n_blocks - 1 else 0)
        windows.append((low, max(high, low)))
    return windows


def session_window(unit: dict, session_id: str, n_frames: int, fps: float, gap_seconds: float) -> tuple[int, int]:
    """
    Description
    -----------
    The frame window a session or block identifier covers: the whole session for a session, the
    block's window (:func:`block_window_frames`) for a block of a unit carrying a ``block_split``.

    Parameters
    ----------
    unit (dict)
        The unit record; ``block_split`` is read (``None`` for a unit tested across sessions).
    session_id (str)
        A session directory basename or a block identifier.
    n_frames (int)
        Frame count of the source session.
    fps (float)
        Camera frame rate.
    gap_seconds (float)
        Separation between neighbouring blocks (``kinematic_encoding.history_pre_seconds``).

    Returns
    -------
    window (tuple[int, int])
        ``(low, high)``, half-open frame bounds.
    """

    if BLOCK_TAG not in session_id:
        return 0, int(n_frames)
    if unit["block_split"] is None:
        msg = f"{session_id} is a block identifier, but the unit record carries no block split."
        raise ValueError(msg)
    index = int(session_id.split(BLOCK_TAG)[1])
    return block_window_frames(n_frames, fps, unit["block_split"]["n_blocks"], gap_seconds)[index]


def frames_in_window(frames: np.ndarray, window: tuple[int, int]) -> np.ndarray:
    """
    Description
    -----------
    Keep the frames inside a half-open window.

    Parameters
    ----------
    frames (np.ndarray)
        Frame indices.
    window (tuple[int, int])
        ``(low, high)``.

    Returns
    -------
    kept (np.ndarray)
        The frames with ``low <= frame < high``, in their original order.
    """

    frames = np.asarray(frames, dtype=np.int64)
    return frames[(frames >= window[0]) & (frames < window[1])]


def assemble_unit_sessions(
        unit: dict,
        data_root: str,
        kinematic_features: dict,
        history_pre_seconds: float,
        clean_post_seconds: float,
        clean_against,
        vocal_settings: dict,
) -> dict:
    """
    Description
    -----------
    Assemble, for every session of a cohort unit, the arrays the encoder needs: the pooled-z-scored
    per-frame feature time series (NaNs imputed to 0), the integer spike-frame train, the quiet anchor
    frames and the session's track names (which the transfer resolves its emitter against). Features
    are z-scored across all the unit's sessions together; they are same-day blocks with the same mice,
    so feature columns align across them.

    For a unit tested on blocks of one session, every block shares that session's arrays and differs
    only in its ``window`` and in its quiet anchors, which are clipped to the window. The spike train and
    feature series stay whole, because a history reaches back past the window's start and a circular
    shift wraps over the whole session.

    Parameters
    ----------
    unit (dict)
        A cohort record (``unit_id``, ``courtship_sessions``, ``block_split``, ...).
    data_root (str)
        The ``Data`` root.
    kinematic_features (dict)
        The kinematic-features schema.
    history_pre_seconds (float)
        Kinematic-history / pre-window length, and the separation between neighbouring blocks.
    clean_post_seconds (float)
        Quiet-anchor post-buffer.
    clean_against (str)
        Whose USVs make a frame unclean: ``'all'`` | ``'self'`` | ``'partner'``. Must match what the
        vocal-side assembler is given, so one quiet definition serves the kinematic encoding and the
        vocal-decoding negatives.
    vocal_settings (dict)
        The ``vocalization_settings`` block; its exclusion flags decide which summary rows the quiet
        guard counts (``filter_vocalizations`` at ``'guard'`` scope).

    Returns
    -------
    per_session (dict)
        ``{session_id: {feature_time_series, feature_names, fps, n_frames, spike_frames, quiet,
        track_names, window}}``, keyed by the unit's session or block identifiers.
    """

    # Blocks of one session load that session once.
    source_ids = list(dict.fromkeys(source_session(s) for s in unit["courtship_sessions"]))
    session_dirs = [f"{data_root}/{s}" for s in source_ids]
    feats, cam_fr, names, _suffixes = build_zscored_feature_frames(
        session_dirs=session_dirs, kinematic_features=kinematic_features,
        recorded_mouse_id=unit["mouse_id"],
    )

    per_source: dict = {}
    for session_id in source_ids:
        if session_id not in feats:
            continue
        feat_df = feats[session_id]
        feature_names = feat_df.columns
        feature_ts = np.nan_to_num(feat_df.to_numpy().astype(np.float64), nan=0.0)
        fps = cam_fr[session_id]
        n_frames = feature_ts.shape[0]

        _spk_sec, spike_frames = load_unit_spike_frames(data_root, session_id, unit["unit_id"])
        # The guard keeps squeaks when `squeaks_block_quiet` is set; noise is always dropped.
        usv = filter_vocalizations(load_session_usvs(data_root, session_id), vocal_settings, "guard")
        starts = usv["start"].to_numpy()
        stops = usv["stop"].to_numpy()

        # `clean_against` decides whose USVs make a frame unclean; by default every emitter, so an
        # auditory response is not fitted as kinematic tuning.
        guard_starts, guard_stops = starts, stops
        guard_names = emitter_names(names[session_id], unit["mouse_id"], clean_against)
        if guard_names:
            guard = usv.filter(usv["emitter"].is_in(guard_names))
            guard_starts = guard["start"].to_numpy()
            guard_stops = guard["stop"].to_numpy()
        quiet = quiet_anchor_frames(guard_starts, guard_stops, n_frames, fps,
                                    history_pre_seconds, clean_post_seconds)

        per_source[session_id] = {
            "feature_time_series": feature_ts, "feature_names": feature_names, "fps": fps, "n_frames": n_frames,
            "spike_frames": spike_frames, "quiet": quiet, "track_names": list(names[session_id]),
        }

    per_session: dict = {}
    for session_id in unit["courtship_sessions"]:
        source_id = source_session(session_id)
        if source_id not in per_source:
            continue
        entry = per_source[source_id]
        window = session_window(unit, session_id, entry["n_frames"], entry["fps"], history_pre_seconds)
        per_session[session_id] = {**entry, "quiet": frames_in_window(entry["quiet"], window),
                                   "window": window}
    return per_session


def session_timebase(data_root: str, session_id: str) -> tuple[list[str], float, int]:
    """
    Description
    -----------
    Read a session's tracking-H5 HEADER only: the mouse track names, the camera frame rate and the frame
    count.

    The decoding analyses need the track names and the session duration but no kinematics, so only
    three small datasets are read. Callers resolve the recorded animal in ``track_names`` by name
    (``emitter_names``), not by slot.

    Parameters
    ----------
    data_root (str)
        The ``Data`` root.
    session_id (str)
        Session directory basename.

    Returns
    -------
    track_names, frame_rate, n_frames (tuple[list[str], float, int])
        The session's ``track_names``, its camera frame rate, and the number of tracked frames.
    """

    video_dir = Path(configure_path(data_root)) / session_id / "video"
    track_path = next(video_dir.glob("**/[0-9]*_points3d_translated_rotated_metric.h5"), None)
    if track_path is None:
        msg = f"tracking file not found under {video_dir}"
        raise FileNotFoundError(msg)
    with h5py.File(name=track_path, mode="r") as h5_obj:
        track_names = [item.decode("utf-8") for item in list(h5_obj["track_names"])]
        frame_rate = float(h5_obj["recording_frame_rate"][()])
        n_frames = int(h5_obj["tracks"].shape[0])
    return track_names, frame_rate, n_frames


def clean_window_mask(onsets: np.ndarray, all_starts: np.ndarray, all_stops: np.ndarray,
                      pre_offset: float) -> np.ndarray:
    """
    Description
    -----------
    Mark the focal USVs whose pre-onset window holds no part of any other vocalization.

    A USV is kept when no span from any emitter in the guard set overlaps
    ``(onset - pre_offset, onset)``. Its own span begins exactly at ``onset``, so a USV never excludes
    itself. Short gaps are common, so the window is kept narrow.

    Parameters
    ----------
    onsets (np.ndarray)
        Focal USV onsets in seconds.
    all_starts (np.ndarray)
        USV onsets in seconds, ALL emitters.
    all_stops (np.ndarray)
        USV offsets in seconds, ALL emitters.
    pre_offset (float)
        How far before onset the window reaches.

    Returns
    -------
    keep (np.ndarray)
        Boolean mask over ``onsets``, True where the pre-onset window is clean.
    """

    onset_times = np.asarray(onsets, dtype=np.float64)
    starts = np.asarray(all_starts, dtype=np.float64)
    stops = np.asarray(all_stops, dtype=np.float64)
    if onset_times.size == 0:
        return np.zeros(0, dtype=bool)
    if starts.size == 0:
        return np.ones(onset_times.size, dtype=bool)

    overlaps = ((starts[None, :] < onset_times[:, None])
                & (stops[None, :] > onset_times[:, None] - pre_offset))
    return ~overlaps.any(axis=1)


def assemble_unit_vocal_events(unit: dict, data_root: str, settings: dict, pre_offset: float,
                               width: float) -> dict:
    """
    Description
    -----------
    Assemble, across a unit's sessions, the USV-grain arrays both vocal-decoding axes -- occurrence and identity -- are computed from.

    Two event sets come back. The positives are the unit's analysed emitter USVs (optionally
    clean-prevocal filtered), each with its torus position and the unit's spike count in
    ``[onset - pre_offset, onset - pre_offset + width)``, a peri-onset window under the shipped
    settings. The negatives are disjoint quiet tiles of the same width under the quiet-anchor
    definition, capped per session at ``max_quiet_tiles_per_session``. Whose USVs count as
    contamination is ``vocalization_settings.clean_against`` for both. Both decoding axes receive the
    same positive events. USVs with a missing torus position are dropped and counted.

    For a unit tested on blocks, each block is a session here: its events are the USVs whose onset lies
    in the block's window and its tiles those lying wholly inside it, so nothing within
    ``history_pre_seconds`` of a neighbouring block is kept. The circular shift still wraps over the
    whole session, so ``duration_seconds`` is the session's.

    Parameters
    ----------
    unit (dict)
        A cohort record; ``unit_id``, ``mouse_id``, ``vocal_sessions`` and ``block_split`` are read.
    data_root (str)
        The ``Data`` root.
    settings (dict)
        The whole neural-modeling settings dict; ``vocal_decoding``, ``kinematic_encoding``,
        ``vocalization_settings`` and ``null`` blocks are read.
    pre_offset (float)
        How far before onset the spike window starts, in seconds.
    width (float)
        Spike-window width in seconds, also the quiet-tile width.

    Returns
    -------
    events (dict)
        ``positions``, ``session_index``, ``counts``, ``region_labels`` (the acoustic region per
        event, NaN where the summary carries no such column), ``silent_gap``, ``vocalization_start``,
        ``vocalization_stop`` at vocalization grain; ``occurrence_windows`` (``counts``, ``labels``, ``session_index``, plus the window ``edges`` and
        their ``width``, which the circular-shift null needs to recount both classes from a shifted
        train) over positives and tiles
        together; ``session_ids``; and ``per_session`` bookkeeping (durations, the window in seconds,
        event and tile counts, how many USVs were dropped by each filter).
    """

    decoding = settings["vocal_decoding"]
    encoding = settings["kinematic_encoding"]
    position_columns = decoding["usv_manifold_column_names"]
    tiles_cap = decoding["max_quiet_tiles_per_session"]
    history_pre_seconds = encoding["history_pre_seconds"]
    clean_post_seconds = encoding["clean_post_seconds"]
    region_column = settings["nested_vocal_manifold_position_decoding"]["region_label_column"]
    vocal_emitter = settings["vocalization_settings"]["vocal_emitter"]
    clean_against = settings["vocalization_settings"]["clean_against"]
    rng = np.random.default_rng(settings["null"]["shuffle_seed"])

    session_ids: list[str] = []
    positions_parts, counts_parts, gaps_parts, region_parts = [], [], [], []
    starts_parts, stops_parts, index_parts, window_edge_parts = [], [], [], []
    tile_counts_parts, tile_edge_parts, tile_index_parts = [], [], []
    per_session: dict = {}

    # Vocal sessions only. Blocks load their source session from disk.
    for slot, session_id in enumerate(unit["vocal_sessions"]):
        source_id = source_session(session_id)
        track_names, frame_rate, n_frames = session_timebase(data_root, source_id)
        duration = n_frames / frame_rate
        window = session_window(unit, session_id, n_frames, frame_rate, history_pre_seconds)
        window_low, window_high = window[0] / frame_rate, window[1] / frame_rate
        # Resolved by name from the unit's `mouse_id`; an empty list for 'all' means no filter.
        focal_names = emitter_names(track_names, unit["mouse_id"], vocal_emitter)

        # `counts_in_windows` bisects, so the train must be sorted.
        spike_seconds, _spike_frames = load_unit_spike_frames(data_root, source_id, unit["unit_id"])
        spike_seconds = np.sort(spike_seconds)

        usv = load_session_usvs(data_root, source_id)
        missing = [column for column in position_columns if column not in usv.columns]
        if missing:
            msg = (f"{session_id}: usv_summary is missing the torus position column(s) {missing}. "
                   f"Available columns: {usv.columns}.")
            raise KeyError(msg)

        # `clean_against` governs the tile guard and the prevocal filter alike.
        guard_names = emitter_names(track_names, unit["mouse_id"], clean_against)
        guard_table = filter_vocalizations(usv, settings["vocalization_settings"], "guard")
        guard = guard_table if not guard_names else guard_table.filter(
            guard_table["emitter"].is_in(guard_names))
        all_starts = guard["start"].to_numpy().astype(np.float64)
        all_stops = guard["stop"].to_numpy().astype(np.float64)
        analysed = filter_vocalizations(usv, settings["vocalization_settings"], "analysed")
        focal = analysed.filter(analysed["emitter"].is_in(focal_names)) if focal_names else analysed
        focal_starts = focal["start"].to_numpy().astype(np.float64)
        focal_stops = focal["stop"].to_numpy().astype(np.float64)
        focal_positions = focal.select(position_columns).to_numpy().astype(np.float64)

        # The region label is carried with the events so the nested decoding's macro score uses
        # exactly the events scored here; NaN where the summary has no such column.
        focal_regions = (focal[region_column].to_numpy().astype(np.float64)
                         if region_column in focal.columns
                         else np.full(focal_starts.size, np.nan))

        # The gap covariate `start[i] - stop[i-1]` needs rows in temporal order.
        if focal_starts.size > 1 and np.any(np.diff(focal_starts) < 0):
            msg = f"{session_id}: usv_summary rows are not ordered by start time."
            raise ValueError(msg)

        finite = np.isfinite(focal_positions).all(axis=1)
        clean = (clean_window_mask(focal_starts, all_starts, all_stops, pre_offset)
                 if decoding["require_clean_prevocal_bool"]
                 else np.ones(focal_starts.size, dtype=bool))
        # A block keeps the USVs whose onset lies in its window; a session keeps every USV.
        is_block = BLOCK_TAG in session_id
        inside = ((focal_starts >= window_low) & (focal_starts < window_high) if is_block
                  else np.ones(focal_starts.size, dtype=bool))
        keep = finite & clean & inside

        gaps = silent_gap_before_vocalization(focal_starts, focal_stops)
        window_edges = focal_starts[keep] - pre_offset
        event_counts = counts_in_windows(spike_seconds, window_edges, width)

        # Disjoint tiles at the analysis window's own width, so tile and event counts are comparable.
        tile_edges = quiet_tile_edges(all_starts, all_stops, duration, width,
                                      history_pre_seconds, clean_post_seconds)
        if is_block:
            tile_edges = tile_edges[(tile_edges >= window_low) & (tile_edges + width <= window_high)]
        if tiles_cap is not None and tile_edges.size > tiles_cap:
            tile_edges = np.sort(rng.choice(tile_edges, tiles_cap, replace=False))
        tile_counts = counts_in_windows(spike_seconds, tile_edges, width)

        session_ids.append(session_id)
        positions_parts.append(focal_positions[keep])
        region_parts.append(focal_regions[keep])
        counts_parts.append(event_counts)
        window_edge_parts.append(window_edges)
        gaps_parts.append(gaps[keep])
        starts_parts.append(focal_starts[keep])
        stops_parts.append(focal_stops[keep])
        index_parts.append(np.full(int(keep.sum()), slot, dtype=np.int64))
        tile_counts_parts.append(tile_counts)
        tile_edge_parts.append(tile_edges)
        tile_index_parts.append(np.full(tile_edges.size, slot, dtype=np.int64))

        per_session[session_id] = {
            "slot": slot, "duration_seconds": duration, "frame_rate": frame_rate,
            "window_seconds": (window_low, window_high),
            "n_focal_vocalizations": int(inside.sum()), "n_events": int(keep.sum()),
            "n_dropped_unclean": int((~clean & inside).sum()),
            "n_dropped_missing_position": int((~finite & inside).sum()),
            "n_baseline_tiles": int(tile_edges.size), "n_spikes": int(spike_seconds.size),
        }

    positions = np.vstack(positions_parts) if positions_parts else np.empty((0, 2), dtype=np.float64)
    counts = np.concatenate(counts_parts) if counts_parts else np.empty(0, dtype=np.float64)
    session_index = np.concatenate(index_parts) if index_parts else np.empty(0, dtype=np.int64)
    tile_counts = np.concatenate(tile_counts_parts) if tile_counts_parts else np.empty(0, dtype=np.float64)
    tile_index = np.concatenate(tile_index_parts) if tile_index_parts else np.empty(0, dtype=np.int64)
    tile_edges = np.concatenate(tile_edge_parts) if tile_edge_parts else np.empty(0, dtype=np.float64)
    window_edges = np.concatenate(window_edge_parts) if window_edge_parts else np.empty(0, dtype=np.float64)

    return {
        "positions": positions,
        "session_index": session_index,
        "counts": counts,
        "region_labels": (np.concatenate(region_parts) if region_parts
                          else np.empty(0, dtype=np.float64)),
        "silent_gap": np.concatenate(gaps_parts) if gaps_parts else np.empty(0, dtype=np.float64),
        "vocalization_start": (np.concatenate(starts_parts) if starts_parts
                               else np.empty(0, dtype=np.float64)),
        "vocalization_stop": (np.concatenate(stops_parts) if stops_parts
                              else np.empty(0, dtype=np.float64)),
        "occurrence_windows": {
            "counts": np.concatenate([counts, tile_counts]),
            "labels": np.concatenate([np.ones(counts.size), np.zeros(tile_counts.size)]),
            "session_index": np.concatenate([session_index, tile_index]),
            # The circular-shift null recounts both classes from the shifted train, so it needs the
            # window edges, aligned with `counts`.
            "edges": np.concatenate([window_edges, tile_edges]),
            "width": width,
        },
        "session_ids": session_ids,
        "per_session": per_session,
    }
