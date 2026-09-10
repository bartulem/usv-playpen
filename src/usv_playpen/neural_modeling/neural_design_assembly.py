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

The decoding claims work at a third grain, which the last section of this module supplies. There the
observation is a CALL rather than a frame: one row per focal vocalization, carrying its position on the
acoustic torus and the unit's spike count in a short window before onset. The quiet definition is reused
unchanged for the negative class, tiled at the window's own width so the two counts are comparable, which
is what lets one guard band serve the encoding claim, the timing axis and the content baselines alike.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import polars as pl
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
from .neural_when import counts_in_windows, quiet_tile_edges


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
    for span_start, span_stop in zip(lo[kept], hi[kept], strict=True):
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
    usv_df (pl.DataFrame)
        The USV summary table.
    """

    usv_path = Path(configure_path(data_root)) / session_id / "audio" / f"{session_id}_usv_summary.csv"
    if not usv_path.exists():
        msg = f"usv_summary not found: {usv_path}"
        raise FileNotFoundError(msg)
    return pl.read_csv(usv_path, separator=csv_sep, infer_schema_length=5000)


def emitter_names(track_names: list, recorded_mouse_id: str, spec: str) -> list:
    """
    Description
    -----------
    Resolve an emitter SPEC to the mouse-id strings it names.

    The vocabulary is semantic rather than positional -- ``'self'`` (the recorded, probe-bearing
    animal), ``'partner'`` (the other one), ``'all'`` (every emitter). A slot index would say
    "whatever happens to be in position 0", which is a different claim and one that needs a separate
    assertion to be safe; resolving the recorded animal by LOOKUP from the unit's own ``mouse_id`` is
    the same principle that removed the configured predictor-slot index. ``'self'`` and ``'partner'``
    also match the feature naming this assembler produces (``self.speed``, ``other.speed``), so one
    vocabulary describes both the kinematics and the calls.

    ``'all'`` returns an empty list meaning NO FILTER, deliberately: a session's USV table can carry
    rows whose emitter was never assigned, and a call nobody was credited with is still a call the
    animal heard. Restricting "all" to the two track names would silently let those through.

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
    recorded_mouse_id (str)
        The unit's own animal. It becomes ``self.``; the remaining track name becomes ``other.``. Resolved
        by NAME and EXCLUSION, never by slot -- see the dyad assertion in the body.
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
        # The `self.` role is not a setting: we model the RECORDED animal's spikes, so `self.` must be
        # its kinematics, and which track slot it occupies is a lookup in that session's own
        # `track_names`. A configured index could disagree with the data -- pointing it at the partner
        # would silently analyse the wrong animal and build an entirely different feature set -- and it
        # would assume the slot is identical in every session, which nothing guarantees.
        names = mouse_names_dict[sess_id]
        # `self.` is resolved by NAME and `other.` by EXCLUSION -- the same rule `emitter_names` applies
        # on the vocal side, so ONE definition of self/partner serves both the kinematics and the calls.
        # This matters because the P1 utility downstream, `resolve_mouse_roles`, takes an INDEX and
        # derives the other animal as `abs(idx - 1)`: a two-slot flip that is the partner only in a
        # DYAD. A third track would send it to an arbitrary animal with no error at all, so the dyad it
        # assumes is ASSERTED here rather than trusted. Measured over the cohort's 73 courtship
        # sessions: `track_names` has length 2 in 73 of 73, recorded animal at slot 0 in 73 of 73 --
        # the assumption holds, which is exactly why it would go unnoticed if it ever stopped holding.
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
        # The column harmonisation downstream takes ONE pair of indices, so a unit whose animal sits in
        # different slots across its sessions cannot be handled by it. Measured 73/73 at slot 0 in this
        # cohort, so this is an assumption that holds -- and it fails loudly rather than silently using
        # whichever session happened to be last if it ever stops holding.
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
        # Same derivation as above: the recorded animal is `self.`, its partner `other.` -- by name and
        # by exclusion, not by slot. The dyad guard in the first loop has already run for every session
        # that reached here, so the single-element index is safe.
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
        history_pre_seconds: float,
        clean_post_seconds: float,
        clean_against,
        vocal_emitter: str,
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
    history_pre_seconds (float)
        Kinematic-history / pre-window length.
    clean_post_seconds (float)
        Quiet-anchor post-buffer.
    clean_against (str)
        Whose calls make a frame unclean: ``'all'`` | ``'self'`` | ``'partner'``. MUST match what the vocal-side
        assembler is given -- the plan requires ONE quiet definition across claim 1, the WHEN
        negatives and the claim-2 baselines, and two code paths that can disagree is how that breaks.
    vocal_emitter (str)
        Whose calls seed the USV-onset anchors: ``'self'`` | ``'partner'`` | ``'all'``. The SAME key the
        vocal-side assembler reads, for the same reason ``clean_against`` is shared -- pointing it at
        the partner must move claim 1's onsets and claim 2's events together, or the two claims of one
        conjunction would be built on different animals' calls.

    Returns
    -------
    per_session (dict)
        ``{session_id: {feature_ts, feature_names, fps, n_frames, spike_frames, quiet, onset}}``.
    """

    session_dirs = [f"{data_root}/{s}" for s in unit["courtship_sessions"]]
    feats, cam_fr, names, _suffixes = build_zscored_feature_frames(
        session_dirs=session_dirs, kinematic_features=kinematic_features,
        recorded_mouse_id=unit["mouse_id"],
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
        onset_names = emitter_names(names[session_id], unit["mouse_id"], vocal_emitter)
        onset_table = usv.filter(usv["emitter"].is_in(onset_names)) if onset_names else usv
        onset_starts = onset_table["start"].to_numpy() if usv.height else np.array([])

        # `clean_against` decides WHOSE calls make a frame unclean. Under the default every emitter
        # counts, so a quiet frame is one in which the animal was not even HEARING a call -- PAG sits
        # in a vocal-auditory circuit, and an auditory response folded into a "quiet" frame would be
        # fitted as kinematic tuning. Measured cost of that strictness: 0-10% of quiet frames, and
        # exactly zero in sessions where the partner never vocalized.
        guard_starts, guard_stops = starts, stops
        guard_names = emitter_names(names[session_id], unit["mouse_id"], clean_against)
        if guard_names:
            guard = usv.filter(usv["emitter"].is_in(guard_names))
            guard_starts = guard["start"].to_numpy()
            guard_stops = guard["stop"].to_numpy()
        quiet = quiet_anchor_frames(guard_starts, guard_stops, n_frames, fps,
                                    history_pre_seconds, clean_post_seconds)
        onset = onset_anchor_frames(onset_starts, starts, stops, n_frames, fps, history_pre_seconds)

        per_session[session_id] = {
            "feature_time_series": feature_ts, "feature_names": feature_names, "fps": fps, "n_frames": n_frames,
            "spike_frames": spike_frames, "quiet": quiet, "onset": onset,
        }
    return per_session


def session_timebase(data_root: str, session_id: str) -> tuple[list[str], float, int]:
    """
    Description
    -----------
    Read a session's tracking-H5 HEADER only: the mouse track names, the camera frame rate and the frame
    count.

    The decoding claims need the emitter mapping and the session duration but no kinematics at all, while
    the behavioural loader that normally supplies them reads a whole feature table per session on the way.
    Three small dataset reads cost a file handle instead, which is the difference between a cheap event
    assembler and one that pays the encoding pipeline's price for information it discards.

    Track-name order is the convention the whole project uses: index 0 is the recorded, probe-bearing
    animal, index 1 its partner.

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
    track_path = next(video_dir.glob("**/[!speaker]*_points3d_translated_rotated_metric.h5"), None)
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
    Mark the focal calls whose pre-onset window holds no part of any other vocalization.

    A call is kept when no span from ANY emitter overlaps ``(onset - pre_offset, onset)``. Its own span
    begins exactly at ``onset``, so a call never excludes itself, and a partner's call counts against it
    just as a preceding focal call does -- the predictor is meant to measure this call's preamble, not the
    tail of whatever came before it.

    The filter is nearly free at the shipped 50 ms offset and ruinous much beyond it. Measured across all
    73 courtship sessions (56,170 intervals), the silent gap to the previous call's OFFSET is under 50 ms
    for 3.7% of calls but under 100 ms for 55.5%, so widening the window does not buy power, it spends
    most of the events and biases the survivors toward bout-initial calls.

    Parameters
    ----------
    onsets (np.ndarray)
        Focal call onsets in seconds.
    all_starts (np.ndarray)
        Call onsets in seconds, ALL emitters.
    all_stops (np.ndarray)
        Call offsets in seconds, ALL emitters.
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
    Assemble, across a unit's sessions, the CALL-grain arrays both claim-2 axes are computed from.

    Two event sets come back, and they are deliberately built from one definition each rather than two.

    Whose calls count as contamination is ``vocalization_settings.clean_against``: every emitter by default, so that a
    quiet window means the animal could not even hear its partner, or one named mouse. The same answer
    governs the quiet tiles and the prevocal filter, because they are asking the same question.
    The positives are the unit's focal calls -- optionally clean-prevocal filtered -- each carrying its
    torus position and the unit's spike count in ``[onset - pre_offset, onset - pre_offset + width)``.
    The negatives are claim-1 quiet tiles: stretches with no call from any emitter in the guard band,
    tiled at the window's own width so a tile count and a prevocal count measure the same thing. Sharing
    the quiet definition with the encoding claim is what keeps one guard band serving three purposes, and
    its four-second FORWARD arm is what makes the timing contrast "a call in 50 ms" against "no call for
    seconds" rather than merely "not during a call".

    Both axes are handed the SAME positive events. That is not tidiness: the WHEN and WHAT verdicts cross
    into a 2x2, and a cell of that table means nothing if the two axes were scored on different calls.

    Tiles are laid at the baseline width and then re-centred on the analysis window, so a wider window
    keeps its tiles concentric with the narrow ones instead of walking forward through the gap. The shift
    is at most tens of milliseconds against a four-second guard, so a re-centred tile is still deep inside
    silence.

    Calls whose torus position is missing are dropped and COUNTED rather than passed on as NaN, since a
    NaN position would silently poison a prior, a basis and a truth cell alike.

    Parameters
    ----------
    unit (dict)
        A cohort record; ``unit_id``, ``mouse_id`` and ``vocal_sessions`` are read.
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
        event, NaN where the summary carries no such column), ``silent_gap``, ``call_start``,
        ``call_stop`` at
        call grain; ``when`` (``counts``, ``labels``, ``session_index``, plus the window ``edges`` and
        their ``width``, which the circular-shift null needs to recount both classes from a shifted
        train) over positives and tiles
        together; ``session_ids``; and ``per_session`` bookkeeping (durations, event and tile counts,
        how many calls were dropped by each filter).
    """

    decoding = settings["vocal_decoding"]
    encoding = settings["kinematic_encoding"]
    position_columns = decoding["usv_manifold_column_names"]
    tiles_cap = decoding["max_quiet_tiles_per_session"]
    history_pre_seconds = encoding["history_pre_seconds"]
    clean_post_seconds = encoding["clean_post_seconds"]
    region_column = settings["nested_position_decoding"]["region_label_column"]
    vocal_emitter = settings["vocalization_settings"]["vocal_emitter"]
    clean_against = settings["vocalization_settings"]["clean_against"]
    rng = np.random.default_rng(settings["null"]["shuffle_seed"])

    session_ids: list[str] = []
    positions_parts, counts_parts, gaps_parts, region_parts = [], [], [], []
    starts_parts, stops_parts, index_parts, window_edge_parts = [], [], [], []
    tile_counts_parts, tile_edge_parts, tile_index_parts = [], [], []
    per_session: dict = {}

    # VOCAL sessions, not every courtship session: both claim-2 axes score on the calls themselves,
    # and a session with a handful of them contributes a leave-one-session-out fold whose statistic
    # rests on those few events. The encoding claim keeps such a session, since it needs silence.
    for slot, session_id in enumerate(unit["vocal_sessions"]):
        track_names, frame_rate, n_frames = session_timebase(data_root, session_id)
        duration = n_frames / frame_rate
        # The emitter is resolved by NAME from the unit's own `mouse_id`, never by track slot, so a
        # session whose track order differs cannot silently hand back the partner's calls;
        # `emitter_names` raises if the unit's animal is not in this session at all.
        focal_name = emitter_names(track_names, unit["mouse_id"], vocal_emitter)[0]

        # `counts_in_windows` bisects, so an unsorted train would return wrong counts with no error.
        # The files on disk are ordered, but that is a property of the writer, not a guarantee here.
        spike_seconds, _spike_frames = load_unit_spike_frames(data_root, session_id, unit["unit_id"])
        spike_seconds = np.sort(spike_seconds)

        usv = load_session_usvs(data_root, session_id)
        missing = [column for column in position_columns if column not in usv.columns]
        if missing:
            msg = (f"{session_id}: usv_summary is missing the torus position column(s) {missing}. "
                   f"Available columns: {usv.columns}.")
            raise KeyError(msg)

        # `clean_against` decides WHOSE calls make a window unclean: every emitter (the default, so
        # that "quiet" means the animal could not even hear a partner) or one named mouse. It governs
        # the tile guard and the prevocal filter alike, so one definition serves both.
        guard_names = emitter_names(track_names, unit["mouse_id"], clean_against)
        guard = usv if not guard_names else usv.filter(usv["emitter"].is_in(guard_names))
        all_starts = guard["start"].to_numpy().astype(np.float64)
        all_stops = guard["stop"].to_numpy().astype(np.float64)
        focal = usv.filter(usv["emitter"] == focal_name)
        focal_starts = focal["start"].to_numpy().astype(np.float64)
        focal_stops = focal["stop"].to_numpy().astype(np.float64)
        focal_positions = focal.select(position_columns).to_numpy().astype(np.float64)

        # The acoustic-region label rides along with the events rather than being re-read later,
        # because claim 3's macro score averages within region and MUST do so over exactly the events
        # claim 2 scored -- a second read could filter differently and silently compare two event
        # sets. NaN where the summary carries no such column: claim 2 never needs it, and claim 3
        # raises for itself rather than making a summary without supercategories unusable here.
        focal_regions = (focal[region_column].to_numpy().astype(np.float64)
                         if region_column in focal.columns
                         else np.full(focal_starts.size, np.nan))

        # The gap covariate is `start[i] - stop[i-1]`, so it means nothing unless the rows are in
        # temporal order. Every summary on disk is, but an out-of-order one would produce negative
        # gaps rather than an error, and the covariate would quietly stop describing bout structure.
        if focal_starts.size > 1 and np.any(np.diff(focal_starts) < 0):
            msg = f"{session_id}: usv_summary rows are not ordered by start time."
            raise ValueError(msg)

        finite = np.isfinite(focal_positions).all(axis=1)
        clean = (clean_window_mask(focal_starts, all_starts, all_stops, pre_offset)
                 if decoding["require_clean_prevocal_bool"]
                 else np.ones(focal_starts.size, dtype=bool))
        keep = finite & clean

        gaps = silent_gap_before_call(focal_starts, focal_stops)
        window_edges = focal_starts[keep] - pre_offset
        event_counts = counts_in_windows(spike_seconds, window_edges, width)

        # Tiles are laid at the ANALYSIS window's own width -- a tile count and a spike-window count
        # must be the same width or they are not comparable. One window is analysed per run
        # (`vocalization_settings.spike_window`), so there is no second width to re-centre a lattice against, and
        # the tiles are disjoint rather than an overlapping lattice.
        tile_edges = quiet_tile_edges(all_starts, all_stops, duration, width,
                                      history_pre_seconds, clean_post_seconds)
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
            "n_focal_calls": int(focal_starts.size), "n_events": int(keep.sum()),
            "n_dropped_unclean": int((~clean).sum()), "n_dropped_missing_position": int((~finite).sum()),
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
        "call_start": np.concatenate(starts_parts) if starts_parts else np.empty(0, dtype=np.float64),
        "call_stop": np.concatenate(stops_parts) if stops_parts else np.empty(0, dtype=np.float64),
        "when": {
            "counts": np.concatenate([counts, tile_counts]),
            "labels": np.concatenate([np.ones(counts.size), np.zeros(tile_counts.size)]),
            "session_index": np.concatenate([session_index, tile_index]),
            # The circular-shift null recounts BOTH classes from the shifted train, so it needs the
            # window EDGES, not merely the counts taken from the real one. Returning them aligned
            # with `counts` makes the null a per-session recount rather than an event-set rebuild.
            "edges": np.concatenate([window_edges, tile_edges]),
            "width": width,
        },
        "session_ids": session_ids,
        "per_session": per_session,
    }
