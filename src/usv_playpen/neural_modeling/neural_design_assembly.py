"""
@author: bartulem
Turning a unit's sessions into the per-frame design the encoding models are fitted on.

The module has two halves. The first loads a unit's sessions and builds the pooled, z-scored predictor
time series, leaning on the behavioural pipeline's own loaders and column utilities rather than
reimplementing them, so the neural features are the same features P1 uses. The second is deterministic
frame bookkeeping: which frames count as silence, which frames a vocalization occupies, what the lagged
predictor looks like at a frame, and whether the unit spiked there. The modelling itself lives elsewhere;
this module only decides what gets modelled.

Two definitions carry most of the weight and are worth stating precisely, because the asymmetry between
them is deliberate rather than accidental.

A QUIET anchor is a frame with no vocalization from ANY emitter in the window running from
``history_pre_seconds`` before it to ``clean_post_seconds`` after it. Cleanliness is judged against every
animal, not just the focal one, because the point is that nothing was audible.

A VOCAL frame is a frame inside a FOCAL animal's own call, and no cleanliness condition is applied to it at
all. Applying the quiet rule to vocal events would be the wrong instrument: it is the right rule for
silence, where quiet IS the definition, but on a bout of calls it retained 77 of 2,437 focal USVs and every
survivor was bout-initial, which is a crippling loss of power and a systematic selection bias at the same
time. Every focal call is kept instead, with the silent gap to the previous one recorded as a covariate so
bout structure stays checkable after the fact rather than being filtered for up front.

The consequence is that a session divides into three parts, not two: quiet anchors, focal vocal frames, and
a substantial remainder belonging to neither -- the guard bands, and every frame inside a non-focal
animal's call. On one three-session day that split was 63.1% quiet, 6.4% vocal, 30.5% neither.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from numpy.lib.stride_tricks import sliding_window_view

from ..analyses.compute_behavioral_features import FeatureZoo
from ..analyses.unit_triage_aggregator import _parse_unit_id
from ..modeling.load_input_files import load_behavioral_feature_data
from ..modeling.modeling_utils import (harmonize_session_columns, resolve_mouse_roles,
                                       select_kinematic_columns, zscore_features_across_sessions)
from ..os_utils import configure_path


def quiet_anchor_frames(usv_starts_seconds: np.ndarray, usv_stops_seconds: np.ndarray, n_frames: int,
                        fps: float, history_pre_seconds: float,
                        clean_post_seconds: float) -> np.ndarray:
    """
    Description
    -----------
    Frames whose window ``[t/fps - history_pre, t/fps + clean_post]`` contains no vocalization at all.

    Built by rasterizing each call's forbidden span into a boolean mask and keeping what falls outside every
    one of them. A call forbids ``[start - clean_post, stop + history_pre]``, which is the same statement
    read from the call's side rather than the frame's. Frames without a full pre-history or post-buffer
    inside the recording are dropped, since their window is not fully observed.

    Pass starts and stops for EVERY emitter, not just the focal animal: a quiet anchor is meant to be a
    moment when nothing was audible.

    Parameters
    ----------
    usv_starts_seconds (np.ndarray)
        Call start times in seconds, all emitters.
    usv_stops_seconds (np.ndarray)
        Call stop times in seconds, all emitters.
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
    for start, stop in zip(usv_starts_seconds, usv_stops_seconds):
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
    Expand each focal call into the frames of its ``[start, stop)`` span, keeping only calls whose frames
    all carry a full in-bounds predictor history. Spans are ragged, so they come back flattened with an
    offset pointer rather than padded.

    The rounding is nudged before it is applied. ``stop * fps`` lands on values like 220.00000000000003, and
    a bare ceiling then adds a frame the call never occupied -- which inflates every span whose edge is
    representable-adjacent, and inflates it worst for the longest calls. A tolerance of one part in a
    billion removes that without affecting any genuine boundary.

    Parameters
    ----------
    starts_seconds (np.ndarray)
        Focal call start times in seconds.
    stops_seconds (np.ndarray)
        Focal call stop times in seconds.
    fps (float)
        Camera frame rate.
    n_frames (int)
        Session frame count.
    n_lags (int)
        Predictor history length in frames; a frame needs this much history in bounds.

    Returns
    -------
    kept (np.ndarray)
        Boolean mask over the input calls, marking those retained.
    flat_frames (np.ndarray)
        Concatenated span frame indices of the kept calls.
    pointer (np.ndarray)
        Offsets of length ``n_kept + 1``; call ``i`` owns ``flat_frames[pointer[i]:pointer[i + 1]]``.
    """

    starts = np.asarray(starts_seconds, dtype=np.float64)
    stops = np.asarray(stops_seconds, dtype=np.float64)
    tolerance = 1e-9
    lo = np.floor(starts * fps + tolerance).astype(np.int64)
    hi = np.maximum(np.ceil(stops * fps - tolerance).astype(np.int64), lo + 1)
    kept = (lo >= n_lags - 1) & (hi <= n_frames)
    blocks, offsets = [], [0]
    for span_start, span_stop in zip(lo[kept], hi[kept]):
        blocks.append(np.arange(span_start, span_stop, dtype=np.int64))
        offsets.append(offsets[-1] + int(span_stop - span_start))
    flat = np.concatenate(blocks) if blocks else np.zeros(0, dtype=np.int64)
    return kept, flat, np.asarray(offsets, dtype=np.int64)


def silent_gap_before_call(starts_seconds: np.ndarray, stops_seconds: np.ndarray) -> np.ndarray:
    """
    Description
    -----------
    Silent gap in seconds from the previous focal call's offset to each call's onset, with infinity for the
    first call.

    Recorded per event so that bout structure remains checkable after the fact. It is deliberately left
    CONTINUOUS rather than thresholded into bout-initial and mid-bout, since any threshold would be an
    invented constant; a binary can be derived later from the gaps if one is ever justified.

    Parameters
    ----------
    starts_seconds (np.ndarray)
        Focal call start times in seconds, sorted.
    stops_seconds (np.ndarray)
        Focal call stop times in seconds, sorted.

    Returns
    -------
    gaps (np.ndarray)
        Silent gap preceding each call.
    """

    gaps = np.full(starts_seconds.size, np.inf, dtype=np.float64)
    if starts_seconds.size > 1:
        gaps[1:] = starts_seconds[1:] - stops_seconds[:-1]
    return gaps


def spike_labels_at_frames(spike_frames: np.ndarray, anchor_frames: np.ndarray,
                           n_frames: int) -> np.ndarray:
    """
    Description
    -----------
    Binary spike labels at the requested frames.

    Binarization is exact rather than approximate at this bin width: the tracking rate gives frames of about
    6.7 ms, where the refractory period caps a frame at one spike, so Bernoulli is the true likelihood and
    nothing is discarded. That stops being true the moment the bin is widened -- over a 200 ms window a fast
    unit emits a dozen spikes and collapsing them to a single bit is data destruction, not a modelling
    convenience.

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

    Built with a sliding window view rather than an explicit loop over lags, so the whole history is a
    stride trick over the original array and no per-lag copy is made. ``sliding_window_view(column,
    n_lags)[i]`` is ``column[i : i + n_lags]``, so the window ENDING at frame ``t`` begins at index
    ``t - n_lags + 1``.

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
        raise ValueError(f"all anchor frames must be >= n_lags-1 ({n_lags - 1}); "
                         f"got min {int(anchor_frames.min())}.")
    start_index = anchor_frames - (n_lags - 1)
    return np.hstack([sliding_window_view(feature_time_series[:, feature], n_lags)[start_index]
                      for feature in range(feature_time_series.shape[1])])


def subsample_quiet_anchors(quiet_frames: np.ndarray, spike_frames: np.ndarray, n_frames: int,
                            negatives_per_positive: int, rng, max_total: int = None) -> tuple:
    """
    Description
    -----------
    Class-balanced, memory-bounded subsampling of quiet anchors, with the log-prior offset correction that
    keeps the fit calibrated afterwards.

    All of the minority class is kept and the majority is subsampled to ``negatives_per_positive`` times its
    size. For the usual rare-firing unit that reduces to keeping every spike frame and thinning the zeros,
    but stating it by minority class rather than by label means a high-firing unit, where the zeros are
    scarcer, is handled correctly instead of backwards.

    Subsampling shifts the log-odds by the log ratio of the two keep-fractions. That constant comes back as
    a per-sample OFFSET to be added during the fit, so the learned intercept is already on the
    true-population scale and prediction on un-subsampled data, where the offset is zero, is calibrated. The
    fit itself then runs unweighted, which is the efficient case-control estimator.

    ``max_total`` scales both classes down together when even the balanced design would be too large; the
    offset absorbs that too. Without it a six-session unit at a high firing rate builds a design of tens of
    gigabytes and is killed.

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
    integer spike-frame indices (row 1, ``np.round`` -> int), matching the two rows used by claims 2/3
    (seconds, ``searchsorted``) and claim 1 (frames, per-frame binning).

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


def load_session_usvs(data_root: str, session_id: str, csv_sep: str = ",") -> pl.DataFrame:
    """
    Description
    -----------
    Read a session's ``<session>_usv_summary.csv`` (carries ``start``/``stop``/``emitter``, the QLVM
    columns ``qlvm_dim1``/``qlvm_dim2``, and raw acoustics).

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
    usv_df (pl.DataFrame)
        The USV summary table.
    """

    usv_path = Path(configure_path(data_root)) / session_id / "audio" / f"{session_id}_usv_summary.csv"
    if not usv_path.exists():
        msg = f"usv_summary not found: {usv_path}"
        raise FileNotFoundError(msg)
    return pl.read_csv(usv_path, separator=csv_sep, infer_schema_length=5000)


def emitter_name(mouse_names: list[str], emitter_index: int) -> str:
    """
    Description
    -----------
    Resolve an emitter index to its mouse-id string via the per-session ``track_names`` list (convention
    index 0 = male / recorded animal, 1 = female).

    Parameters
    ----------
    mouse_names (list[str])
        The session's ``track_names``.
    emitter_index (int)
        The emitter slot (e.g. 0 = recorded animal).

    Returns
    -------
    name (str)
        The mouse-id string.
    """

    if emitter_index < 0 or emitter_index >= len(mouse_names):
        msg = f"emitter_index {emitter_index} out of range for track_names {mouse_names}."
        raise ValueError(msg)
    return mouse_names[emitter_index]


def build_zscored_feature_frames(
        session_dirs: list[str],
        kinematic_features: dict,
        model_predictor_mouse_index: int,
        csv_sep: str = ",",
) -> tuple[dict, dict, dict, list]:
    """
    Description
    -----------
    Build the P1-identical z-scored behavioural feature time series for the given sessions by reusing the
    modeling chain: ``load_behavioral_feature_data`` -> per-session ``select_kinematic_columns`` ->
    ``harmonize_session_columns`` -> ``zscore_features_across_sessions``. NO vocal-signal columns are added
    (claim 1 is kinematic-only). The z-scoring pools across all supplied sessions (the all-frames ruler).

    Parameters
    ----------
    session_dirs (list[str])
        Session directory paths (``Data/<session_id>``), resolved via ``configure_path`` by the loader.
    kinematic_features (dict)
        The kinematic-features schema (egocentric / dyadic_pose / abs_features / ...).
    model_predictor_mouse_index (int)
        The predictor mouse slot (P1 convention), used by ``resolve_mouse_roles`` for the self/other split.
    csv_sep (str)
        CSV separator.

    Returns
    -------
    feature_frames, camera_fr_dict, mouse_names_dict, suffixes (tuple[dict, dict, dict, list])
        Per-session z-scored feature DataFrames (keyed by session_id), camera fps, mouse track names, and
        the ordered feature suffix list (the design's feature order).
    """

    settings = {
        "model_params": {"model_predictor_mouse_index": model_predictor_mouse_index},
        "kinematic_features": kinematic_features,
    }
    resolved = [configure_path(p) for p in session_dirs]
    beh_dict, camera_fr_dict, mouse_names_dict = load_behavioral_feature_data(
        behavior_file_paths=resolved, csv_sep=csv_sep,
    )

    processed: dict = {}
    predictor_idx = model_predictor_mouse_index
    target_idx = abs(predictor_idx - 1)
    for sess_id, session_df in beh_dict.items():
        if sess_id not in mouse_names_dict:
            continue
        predictor_idx, target_idx, p_name, t_name = resolve_mouse_roles(
            modeling_settings=settings, mouse_names_dict=mouse_names_dict, session_id=sess_id,
        )
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
    # `FeatureZoo.feature_boundaries` is the canonical physical-range dict P1's pipelines use to clip
    # out-of-range values before pooled z-scoring — reuse it so the neural features match P1 exactly.
    processed = zscore_features_across_sessions(
        processed_beh_dict=processed, suffixes=suffixes,
        feature_bounds=FeatureZoo.feature_boundaries,
        abs_features=kinematic_features["abs_features"],
        smooth_abs_features=kinematic_features["smooth_abs_features"],
    )

    # Canonicalise columns to ROLE-based names in a single fixed order across sessions.
    # `harmonize_session_columns` leaves ego columns prefixed with the raw mouse id (e.g.
    # `181316_0.speed` for self, `{partner_id}.speed` for the other animal). Because the partner id
    # differs between sessions AND the column order is not enforced, `df.to_numpy()` is NOT
    # position-consistent across sessions: the same array column index points at a different feature in
    # different sessions. Any cross-session position-indexed use (design assembly, LOSO) would then read
    # the wrong feature -- a silent, catastrophic bug. Here each session's ego columns are re-prefixed to
    # the generic `self.`/`other.` roles (dyadic suffix-only columns are left unchanged), then every
    # session is reindexed to one canonical, sorted column order. After this, a single `names` list is
    # valid for every session and by-name and by-position access agree.
    canonical: dict = {}
    canonical_names: list[str] | None = None
    for sess_id, df in processed.items():
        pred_i, tgt_i, p_name, t_name = resolve_mouse_roles(
            modeling_settings=settings, mouse_names_dict=mouse_names_dict, session_id=sess_id,
        )
        rename_map: dict = {}
        for col in df.columns:
            if col.startswith(f"{t_name}."):
                rename_map[col] = f"self.{col.split('.', 1)[1]}"
            elif col.startswith(f"{p_name}."):
                rename_map[col] = f"other.{col.split('.', 1)[1]}"
        df = df.rename(rename_map)
        ordered = sorted(df.columns)
        df = df.select(ordered)
        canonical[sess_id] = df
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


def onset_anchor_frames(
        focal_starts_sec: np.ndarray,
        all_starts_sec: np.ndarray,
        all_stops_sec: np.ndarray,
        n_frames: int,
        fps: float,
        history_pre_seconds: float,
) -> np.ndarray:
    """
    Description
    -----------
    Frames at focal-USV onsets that have a clean pre-history (no USV overlapping ``[t/fps - history_pre,
    t/fps)``), the claim-1 transfer / claim-3-seed anchors. A frame needs its full pre-history in-bounds.

    Parameters
    ----------
    focal_starts_sec (np.ndarray)
        Onset times (seconds) of the focal emitter's USVs.
    all_starts_sec (np.ndarray)
        Start times of ALL USVs (for the clean-pre-history test).
    all_stops_sec (np.ndarray)
        Stop times of all USVs.
    n_frames (int)
        Session frame count.
    fps (float)
        Camera frame rate.
    history_pre_seconds (float)
        Pre-history window length.

    Returns
    -------
    frames (np.ndarray)
        Sorted integer onset-anchor frame indices with clean pre-history.
    """

    history_frames = int(np.floor(history_pre_seconds * fps))
    kept: list[int] = []
    starts = np.asarray(all_starts_sec)
    stops = np.asarray(all_stops_sec)
    for onset in focal_starts_sec:
        frame = int(np.floor(onset * fps))
        if frame < history_frames or frame >= n_frames:
            continue
        window_lo = onset - history_pre_seconds
        # any OTHER usv overlapping [window_lo, onset)?
        overlaps = (starts < onset) & (stops > window_lo) & (starts > window_lo - 1e-9)
        if not np.any(overlaps):
            kept.append(frame)
    return np.array(sorted(set(kept)), dtype=np.int64)


def assemble_unit_sessions(
        unit: dict,
        data_root: str,
        kinematic_features: dict,
        model_predictor_mouse_index: int,
        history_pre_seconds: float,
        clean_post_seconds: float,
) -> dict:
    """
    Description
    -----------
    Assemble, for every session of a cohort unit, the arrays the encoder needs: the pooled-z-scored
    per-frame feature time series (NaNs imputed to 0), the integer spike-frame train, and the quiet /
    USV-onset anchor frames. Features are z-scored across ALL the unit's sessions together (one ruler); the
    unit's sessions are same-day blocks with the same mice, so feature columns align across them.

    Parameters
    ----------
    unit (dict)
        A cohort record (``unit_id``, ``courtship_sessions``, ...).
    data_root (str)
        The ``Data`` root.
    kinematic_features (dict)
        The kinematic-features schema.
    model_predictor_mouse_index (int)
        Predictor mouse slot (P1 convention).
    history_pre_seconds (float)
        Kinematic-history / pre-window length.
    clean_post_seconds (float)
        Quiet-anchor post-buffer.

    Returns
    -------
    per_session (dict)
        ``{session_id: {feature_ts, feature_names, fps, n_frames, spike_frames, quiet, onset}}``.
    """

    session_dirs = [f"{data_root}/{s}" for s in unit["courtship_sessions"]]
    feats, cam_fr, names, _suffixes = build_zscored_feature_frames(
        session_dirs=session_dirs, kinematic_features=kinematic_features,
        model_predictor_mouse_index=model_predictor_mouse_index,
    )

    per_session: dict = {}
    for session_id in unit["courtship_sessions"]:
        if session_id not in feats:
            continue
        feat_df = feats[session_id]
        feature_names = feat_df.columns
        feature_ts = np.nan_to_num(feat_df.to_numpy().astype(np.float64), nan=0.0)
        fps = cam_fr[session_id]
        n_frames = feature_ts.shape[0]

        _spk_sec, spike_frames = load_unit_spike_frames(data_root, session_id, unit["unit_id"])
        usv = load_session_usvs(data_root, session_id)
        starts = usv["start"].to_numpy()
        stops = usv["stop"].to_numpy()
        focal = emitter_name(names[session_id], 0)
        focal_starts = usv.filter(usv["emitter"] == focal)["start"].to_numpy() if usv.height else np.array([])

        quiet = quiet_anchor_frames(starts, stops, n_frames, fps, history_pre_seconds, clean_post_seconds)
        onset = onset_anchor_frames(focal_starts, starts, stops, n_frames, fps, history_pre_seconds)

        per_session[session_id] = {
            "feature_time_series": feature_ts, "feature_names": feature_names, "fps": fps, "n_frames": n_frames,
            "spike_frames": spike_frames, "quiet": quiet, "onset": onset,
        }
    return per_session
