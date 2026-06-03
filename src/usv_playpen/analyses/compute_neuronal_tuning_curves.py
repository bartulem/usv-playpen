"""
@author: bartulem
Compute per-cluster neuronal tuning curves: behavioral and vocal.

One class, one entry point, one output pkl per cluster. Each side of the
pipeline (behavioral and vocal) is internally gated on the presence of
its own required inputs. Missing-input is a graceful skip (logged); only
the absence of spike `.npy` files is a hard error, since "no clusters"
means there is nothing to tune at all.

Behavioral path (driven by `*_behavioral_features.csv` + tracking H5):
  - per-cluster, per-temporal-offset 1D feature ratemaps with shuffled
    nulls (`generate_ratemaps`)
  - per-cluster, per-animal 2D spatial ratemap

Vocal path (driven by `*_usv_summary.csv` + tracking H5 + audio sync):
  `usv_peth`              pooled pre-USV PETH per emitter side
                          (default [-2, 0] s window, 50 ms bins)
  `usv_property_tuning`   within-USV firing rate vs each continuous
                          acoustic property (duration, mean / peak
                          frequency, bandwidth, amplitude, spectral
                          entropy, mask number)
  `usv_category_tuning`   per-category within-USV firing rate (VAE /
                          QLVM `category` and `supercategory`)
  `usv_category_peth`     per-category time-resolved peri-USV PETH

Both paths write into the same per-cluster pickle:
  ephys/tuning_curves/{cluster_id}_tuning_curves_data.pkl
Behavioral keys are flat at the top level (`beh_offset=*s`); vocal keys
sit alongside (`usv_peth`, `usv_property_tuning`, `usv_category_tuning`,
`usv_category_peth`, `usv_metadata`). Each path uses a
load-existing-or-create helper so running them out of order, in either
order, never clobbers the other's output.
"""

from __future__ import annotations

import json
import pathlib
import pickle
from collections import OrderedDict
from datetime import datetime
from typing import Any

import h5py
import numpy as np
import polars as pls
from scipy import ndimage, stats
from tqdm import tqdm

from ..os_utils import first_match_or_raise
from ..time_utils import is_gui_context, smart_wait
from .compute_behavioral_features import FeatureZoo


CONTINUOUS_PROPERTIES = (
    "duration",
    "mean_freq_hz",
    "peak_freq_hz",
    "freq_bandwidth_hz",
    "mean_amplitude",
    "max_amplitude",
    "spectral_entropy",
    "mask_number",
)

CATEGORICAL_FEATURES = (
    "vae_category",
    "vae_supercategory",
    "qlvm_category",
    "qlvm_supercategory",
)

# `mask_number` is integer-valued; the boundary range [0.5, 12.5] paired
# with 12 bins yields one bin per integer in 1..12. All other continuous
# properties use `total_bin_num` from the settings dict.
MASK_NUMBER_BIN_COUNT = 12


# behavioral-path free helpers


def generate_ratemaps(
    feature_arr: np.ndarray,
    spike_arr: np.ndarray,
    shuffled_spike_arr: np.ndarray,
    min_val: int,
    max_val: int,
    num_bins: int,
    camera_fr: int | float,
    space_bool: bool = False,
) -> (
    tuple[np.ndarray, np.ndarray, np.ndarray]
    | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
):
    """
    Description
    -----------
    Compute occupancy stats and spike counts for a given feature array.

    Parameters
    ----------
    feature_arr (np.ndarray)
        A (n_frames) shape ndarray containing behavioral feature data.
    spike_arr (np.ndarray)
        A (n_spikes) shape ndarray containing spike event frames.
    shuffled_spike_arr (np.ndarray)
        A (n_shuffles, n_frames) shape ndarray containing shuffled spike event frames.
    min_val (int)
        Minimum possible value feature could attain.
    max_val (int)
        Maximum possible value feature could attain.
    num_bins (int)
        Number of bins to divide features in.
    camera_fr (int / float)
        Camera frame rate.
    space_bool (bool)
        Boolean indicating if feature is spatial.

    Returns
    -------
    ratemap (np.ndarray)
        A (n_bins, 2) shape ndarray containing spike counts and occupancy (in seconds)
        for each feature bin (first column spike counts, second column occ).
    sh_counts (np.ndarray)
        A (n_bins) shape ndarray containing shuffled spike counts.
    bin_centers (np.ndarray)
        A (n_bins) shape ndarray containing bin centers for given feature.
    bin_edges (np.ndarray)
        A (n_bins) shape ndarray containing bin edges for given feature.
    """

    if space_bool:
        bins_in_one_dir = int(np.ceil(np.sqrt(num_bins)))
        bin_edges = np.linspace(min_val, max_val, bins_in_one_dir + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        spike_events_x = np.take(a=feature_arr[:, 0], indices=spike_arr)
        spike_events_y = np.take(a=feature_arr[:, 1], indices=spike_arr)
        # `(left, right]` semantics: val exactly at edge[i] (i >= 1) belongs
        # to bin i-1; val exactly at min_val is excluded. searchsorted with
        # side='left' followed by `-1` gives precisely that mapping.
        sx_idx = np.searchsorted(bin_edges, spike_events_x, side="left") - 1
        sy_idx = np.searchsorted(bin_edges, spike_events_y, side="left") - 1
        ox_idx = np.searchsorted(bin_edges, feature_arr[:, 0], side="left") - 1
        oy_idx = np.searchsorted(bin_edges, feature_arr[:, 1], side="left") - 1
        sv = (sx_idx >= 0) & (sx_idx < bins_in_one_dir) & (sy_idx >= 0) & (sy_idx < bins_in_one_dir)
        ov = (ox_idx >= 0) & (ox_idx < bins_in_one_dir) & (oy_idx >= 0) & (oy_idx < bins_in_one_dir)
        spike_flat = sx_idx[sv] * bins_in_one_dir + sy_idx[sv]
        occ_flat = ox_idx[ov] * bins_in_one_dir + oy_idx[ov]
        spike_2d = np.bincount(spike_flat, minlength=bins_in_one_dir * bins_in_one_dir).reshape(bins_in_one_dir, bins_in_one_dir)
        occ_2d = np.bincount(occ_flat, minlength=bins_in_one_dir * bins_in_one_dir).reshape(bins_in_one_dir, bins_in_one_dir)
        ratemap = np.zeros((bins_in_one_dir, bins_in_one_dir, 2))
        ratemap[:, :, 0] = spike_2d.astype(float)
        ratemap[:, :, 1] = occ_2d / camera_fr

        return ratemap, bin_centers, bin_edges

    bin_edges = np.linspace(min_val, max_val, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # observed spike + occupancy counts via searchsorted -> bincount.
    # `(left, right]` semantics matches the legacy implementation
    # bit-exactly: spike at val=bin_edges[i] (i>=1) goes to bin i-1;
    # spike at val=min_val is excluded.
    spike_event_features = np.take(a=feature_arr, indices=spike_arr)
    spike_idx = np.searchsorted(bin_edges, spike_event_features, side="left") - 1
    occ_idx = np.searchsorted(bin_edges, feature_arr, side="left") - 1
    spike_v = (spike_idx >= 0) & (spike_idx < num_bins)
    occ_v = (occ_idx >= 0) & (occ_idx < num_bins)
    spike_counts = np.bincount(spike_idx[spike_v], minlength=num_bins)
    occ_counts = np.bincount(occ_idx[occ_v], minlength=num_bins)
    ratemap = np.column_stack([spike_counts.astype(float), occ_counts / camera_fr])

    # shuffled spike counts: vectorized across (n_shuffles, n_spikes) via
    # one bincount over flattened (shuffle, bin) indices.
    n_shuffles = shuffled_spike_arr.shape[0]
    n_spikes = shuffled_spike_arr.shape[1]
    sh_features = feature_arr[shuffled_spike_arr]                 # (n_shuffles, n_spikes)
    sh_idx_flat = np.searchsorted(bin_edges, sh_features.ravel(), side="left") - 1
    sh_valid = (sh_idx_flat >= 0) & (sh_idx_flat < num_bins)
    shuffle_axis = np.repeat(np.arange(n_shuffles, dtype=np.int64), n_spikes)
    flat = shuffle_axis[sh_valid] * num_bins + sh_idx_flat[sh_valid]
    sh_counts = (
        np.bincount(flat, minlength=n_shuffles * num_bins)
        .reshape(n_shuffles, num_bins)
        .astype(float)
    )

    return ratemap, sh_counts, bin_centers, bin_edges


def shuffle_spikes(
    spike_array: np.ndarray,
    total_fr_num: int,
    shuffle_min_fr: int,
    shuffle_max_fr: int,
    n_shuffles: int,
    seed: int | None = None,
) -> np.ndarray:
    """
    Description
    -----------
    Shuffle a 1D array of spike-frame indices by a random integer offset
    drawn uniformly from [shuffle_min_fr, shuffle_max_fr) and wrap any
    spike that falls past `total_fr_num` back to the beginning of the
    recording.

    Parameters
    ----------
    spike_array (np.ndarray)
         A (n_spikes,) array of spike event frames.
    total_fr_num (int)
        Total number of frames in the recording.
    shuffle_min_fr (int)
        Minimum number of frames to shuffle spikes by.
    shuffle_max_fr (int)
        Maximum number of frames to shuffle spikes by.
    n_shuffles (int)
        Number of shuffles to perform.
    seed (int | None)
        Optional seed for reproducibility; `None` draws from a fresh
        default-RNG state (non-reproducible). Passing a fixed seed makes the
        null distribution -- and therefore the significance verdict -- exactly
        reproducible across runs.

    Returns
    -------
    shuffled_spike_arr (np.ndarray)
        A (n_shuffles, n_spikes) array of shifted-and-sorted spike event
        frames.
    """

    rng = np.random.default_rng(seed)
    shuffled_amounts = rng.integers(
        low=shuffle_min_fr, high=shuffle_max_fr, size=n_shuffles, dtype=np.int32
    )
    shuffled = np.tile(spike_array, reps=(n_shuffles, 1)) + shuffled_amounts[:, np.newaxis]
    # modulo wrap so the shift is correct even when it can exceed
    # `total_fr_num` (single-subtraction approach silently broke on very
    # short recordings).
    return np.sort(np.mod(shuffled, total_fr_num), axis=1)


# vocal-path free helpers


def _circular_shift_spike_times(
    spike_times: np.ndarray,
    duration_seconds: float,
    shift_seconds: float,
) -> np.ndarray:
    """
    Description
    -----------
    Apply one circular shift to a sorted spike-time array. Times that
    exceed `duration_seconds` after shifting are wrapped back to the
    beginning of the recording. The returned array is re-sorted so that
    downstream `np.searchsorted` calls remain valid.

    Parameters
    ----------
    spike_times (np.ndarray)
        Sorted (1D) array of spike times in seconds.
    duration_seconds (float)
        Recording duration in seconds.
    shift_seconds (float)
        Amount to shift (positive scalar).

    Returns
    -------
    shifted (np.ndarray)
        Sorted (1D) array of shifted spike times.
    """

    shifted = spike_times + shift_seconds
    over_mask = shifted >= duration_seconds
    shifted[over_mask] = shifted[over_mask] - duration_seconds
    shifted.sort()
    return shifted


def _latest_other_stop_before_anchor(
    anchor_idx: np.ndarray,
    starts: np.ndarray,
    stops: np.ndarray,
) -> np.ndarray:
    """
    Description
    -----------
    For each anchor, returns the most recent stop time among USVs that
    started strictly before the anchor's start time. Used by the
    post-anchor cleanliness check: the [anchor + Δt, anchor] segment is
    clean iff `anchor + Δt > latest_other_stop_before_anchor`.

    Parameters
    ----------
    anchor_idx (np.ndarray)
        Indices (into `starts`/`stops`) of the USVs treated as anchors.
    starts (np.ndarray)
        Sorted-by-start array of all USV start times (seconds).
    stops (np.ndarray)
        Paired stop times (seconds).

    Returns
    -------
    latest_other_stop (np.ndarray)
        Same length as `anchor_idx`. -inf where no qualifying other USV
        exists.
    """

    out = np.full(anchor_idx.size, -np.inf, dtype=float)
    for k, ai in enumerate(anchor_idx):
        anchor_start = starts[ai]
        end = int(np.searchsorted(starts, anchor_start, side="left"))
        if end == 0:
            continue
        candidate_stops = stops[:end]
        # if anchor itself is in [:end] (start equal to anchor_start and at index ai < end),
        # exclude it
        if ai < end:
            candidate_stops = np.concatenate(
                (candidate_stops[:ai], candidate_stops[ai + 1:])
            )
        if candidate_stops.size > 0:
            out[k] = float(candidate_stops.max())
    return out


def _within_usv_validity(
    anchor_idx: np.ndarray,
    starts: np.ndarray,
    stops: np.ndarray,
) -> np.ndarray:
    """
    Description
    -----------
    Per-anchor boolean: the anchor's own [start, stop] window is clean of
    any other USV.

    Parameters
    ----------
    anchor_idx (np.ndarray)
        Indices (into `starts`/`stops`) of the USVs treated as anchors.
    starts (np.ndarray)
        Sorted-by-start array of all USV start times.
    stops (np.ndarray)
        Paired stop times.

    Returns
    -------
    valid (np.ndarray)
        Boolean array, same length as `anchor_idx`.
    """

    valid = np.ones(anchor_idx.size, dtype=bool)
    for k, ai in enumerate(anchor_idx):
        a_start = starts[ai]
        a_stop = stops[ai]
        cand_end = int(np.searchsorted(starts, a_stop, side="left"))
        idx = np.arange(cand_end)
        idx = idx[idx != ai]
        if idx.size == 0:
            continue
        if (stops[idx] > a_start).any():
            valid[k] = False
    return valid


def _anchor_bin_validity_grid(
    anchor_idx: np.ndarray,
    starts: np.ndarray,
    stops: np.ndarray,
    duration_seconds: float,
    rel_bin_lo: np.ndarray,
    rel_bin_hi: np.ndarray,
    require_clean_post: bool,
    require_clean_prior: bool,
) -> np.ndarray:
    """
    Description
    -----------
    Per-(anchor, bin) boolean validity grid for a PETH spanning a fixed
    set of pre-anchor bins.

    A bin at relative offset Δt is valid for an anchor iff:
      (i) its absolute window lies within [0, duration_seconds];
      (ii) if `require_clean_post`: no other USV intersects
           [anchor + Δt_lo, anchor];
      (iii) if `require_clean_prior`: no other USV intersects
           [anchor + Δt_min, anchor + Δt_lo].

    Parameters
    ----------
    anchor_idx (np.ndarray)
        Indices of anchor USVs in `starts`/`stops`.
    starts (np.ndarray)
        Sorted-by-start start times (seconds) of ALL USVs.
    stops (np.ndarray)
        Paired stop times (seconds).
    duration_seconds (float)
        Recording duration in seconds.
    rel_bin_lo (np.ndarray)
        Lower edge of each bin relative to anchor onset (1D, n_bins).
    rel_bin_hi (np.ndarray)
        Upper edge of each bin relative to anchor onset (1D, n_bins).
    require_clean_post (bool)
        Enforce clause (ii).
    require_clean_prior (bool)
        Enforce clause (iii).

    Returns
    -------
    validity (np.ndarray)
        Boolean grid of shape (n_anchors, n_bins).
    """

    n_anchors = anchor_idx.size
    n_bins = rel_bin_lo.size
    validity = np.ones((n_anchors, n_bins), dtype=bool)

    anchor_starts = starts[anchor_idx]
    bin_lo_abs = anchor_starts[:, None] + rel_bin_lo[None, :]
    bin_hi_abs = anchor_starts[:, None] + rel_bin_hi[None, :]

    in_recording = (bin_lo_abs >= 0.0) & (bin_hi_abs <= duration_seconds)
    validity &= in_recording

    if not (require_clean_post or require_clean_prior):
        return validity

    latest_stop = _latest_other_stop_before_anchor(anchor_idx, starts, stops)

    if require_clean_post:
        clean_post = bin_lo_abs > latest_stop[:, None]
        validity &= clean_post

    if require_clean_prior:
        rel_min = float(rel_bin_lo.min())
        for k, ai in enumerate(anchor_idx):
            a_start = anchor_starts[k]
            window_lo = a_start + rel_min
            lo_idx = int(np.searchsorted(starts, window_lo, side="left"))
            hi_idx = int(np.searchsorted(starts, a_start, side="left"))
            cand = np.arange(lo_idx, hi_idx)
            cand = cand[cand != ai]
            if cand.size == 0:
                continue
            cand_stops = stops[cand]
            for i in range(n_bins):
                if (cand_stops > bin_lo_abs[k, i]).any():
                    validity[k, i] = False

    return validity


def _generate_shuffle_offsets(
    n_shuffles: int,
    shuffle_min_seconds: float,
    shuffle_max_seconds: float,
    seed: int | None = None,
) -> np.ndarray:
    """
    Description
    -----------
    Sample `n_shuffles` random offsets uniformly from
    `[shuffle_min_seconds, shuffle_max_seconds)`. Each offset is later
    used as a circular shift on the spike train to build the null
    distribution for the tuning curves.

    Parameters
    ----------
    n_shuffles (int)
        Number of shuffle offsets to draw.
    shuffle_min_seconds (float)
        Lower bound of the offset distribution (in s).
    shuffle_max_seconds (float)
        Upper bound (exclusive) of the offset distribution (in s).
    seed (int | None)
        Optional seed for reproducibility; `None` draws from a fresh
        default-RNG state.

    Returns
    -------
    offsets (np.ndarray, shape (n_shuffles,))
        Float array of shuffle offsets in seconds.
    """

    rng = np.random.default_rng(seed)
    return rng.uniform(
        low=shuffle_min_seconds, high=shuffle_max_seconds, size=n_shuffles
    ).astype(float)


def _bin_property_indices(
    values: np.ndarray, bin_range: list, n_bins: int
) -> np.ndarray:
    """
    Description
    -----------
    Return the bin index (0..n_bins-1) of each value, clipped into the
    valid range. NaN values map to -1 (sentinel; downstream callers skip).

    Parameters
    ----------
    values (np.ndarray)
        1D array of property values.
    bin_range (list)
        [low, high].
    n_bins (int)
        Number of bins; bins are linearly spaced edges
        np.linspace(low, high, n_bins+1).

    Returns
    -------
    bin_idx (np.ndarray)
        int64 array of bin indices; -1 for NaN.
    """

    low, high = float(bin_range[0]), float(bin_range[1])
    out = np.full(values.size, -1, dtype=np.int64)
    finite = np.isfinite(values)
    if not finite.any():
        return out
    width = (high - low) / n_bins
    if width <= 0:
        return out
    raw = np.floor((values[finite] - low) / width).astype(np.int64)
    raw = np.clip(raw, 0, n_bins - 1)
    out[finite] = raw
    return out


def _bin_count_for_property(prop: str, default_n_bins: int) -> int:
    """
    Description
    -----------
    Return the number of bins to use for a vocal continuous property.
    Most properties use the global `total_bin_num` setting; the
    integer-valued `mask_number` property always uses
    `MASK_NUMBER_BIN_COUNT` so each integer occupies one bin.

    Parameters
    ----------
    prop (str)
        Property name (one of `CONTINUOUS_PROPERTIES`).
    default_n_bins (int)
        Default bin count from the settings file.

    Returns
    -------
    n_bins (int)
        Bin count for this property.
    """

    if prop == "mask_number":
        return MASK_NUMBER_BIN_COUNT
    return int(default_n_bins)


def _percentiles_block(arr: np.ndarray) -> dict:
    """
    Description
    -----------
    Compute per-cell shuffle statistics over axis 0 (the shuffle
    dimension). Returns the keys expected by the per-feature payload:
    `null_mean`, `null_std`, `null_p0_5`, `null_p2_5`, `null_p97_5`,
    `null_p99_5`. NaN-aware along axis 0.

    Parameters
    ----------
    arr (np.ndarray, shape (n_shuffles, ...))
        Shuffle array; the leading axis is the shuffle dimension.

    Returns
    -------
    dict
        Mapping of `null_*` keys to per-cell aggregate arrays (each
        with one fewer dimension than `arr`).
    """
    return {
        "null_mean": np.nanmean(arr, axis=0),
        "null_std": np.nanstd(arr, axis=0),
        "null_p0_5": np.nanpercentile(arr, 0.5, axis=0),
        "null_p2_5": np.nanpercentile(arr, 2.5, axis=0),
        "null_p97_5": np.nanpercentile(arr, 97.5, axis=0),
        "null_p99_5": np.nanpercentile(arr, 99.5, axis=0),
    }


def _gaussian_smooth_1d(
    data: np.ndarray, sigma: float, axis: int = -1, preserve_nan: bool = True
) -> np.ndarray:
    """
    Description
    -----------
    NaN-aware 1D Gaussian smoothing along `axis`. Implements
    `astropy.convolution.convolve(... nan_treatment='interpolate',
    boundary='extend')` semantics via a weighted convolution: NaN
    positions are excluded from the weighted average and (when
    `preserve_nan=True`) restored to NaN at their original positions
    after smoothing. Vectorized across all leading axes via
    `scipy.ndimage.gaussian_filter1d`, so a 2D `(n_shuffles, n_bins)`
    array is smoothed in one call.

    Parameters
    ----------
    data (np.ndarray)
        Array to smooth. Can be 1D or N-D; smoothing is applied along
        `axis`.
    sigma (float)
        Gaussian sigma in bin units. `sigma <= 0` returns the input
        unchanged.
    axis (int)
        Axis along which to smooth.
    preserve_nan (bool)
        If True, NaN positions in the input remain NaN in the output;
        only valid samples participate in the weighted convolution.
        If False (used for the 2D spatial map historically), NaN
        positions are filled in by the weighted Gaussian.

    Returns
    -------
    smoothed (np.ndarray)
        Same shape as `data`.
    """

    if sigma <= 0:
        return data
    nan_mask = np.isnan(data)
    if not nan_mask.any():
        return ndimage.gaussian_filter1d(data, sigma=sigma, axis=axis, mode="nearest")
    weights = (~nan_mask).astype(float)
    data_filled = np.where(nan_mask, 0.0, data)
    smoothed_data = ndimage.gaussian_filter1d(
        data_filled, sigma=sigma, axis=axis, mode="nearest"
    )
    smoothed_weights = ndimage.gaussian_filter1d(
        weights, sigma=sigma, axis=axis, mode="nearest"
    )
    out = np.full_like(data, np.nan, dtype=float)
    valid = smoothed_weights > 0
    out[valid] = smoothed_data[valid] / smoothed_weights[valid]
    if preserve_nan:
        out[nan_mask] = np.nan
    return out


def _gaussian_smooth_2d(
    data: np.ndarray, sigma: float, preserve_nan: bool = False
) -> np.ndarray:
    """
    Description
    -----------
    NaN-aware 2D Gaussian smoothing on a 2D array. Same weighted-
    convolution trick as `_gaussian_smooth_1d`. Defaults to
    `preserve_nan=False` to match the historical behavioral 2D
    spatial-tuning behavior (NaN bins get filled by neighbors at
    plot-render).

    Parameters
    ----------
    data (np.ndarray)
        2D array to smooth.
    sigma (float)
        Gaussian sigma in bin units along both axes. `sigma <= 0`
        returns the input unchanged.
    preserve_nan (bool)
        See `_gaussian_smooth_1d`.

    Returns
    -------
    smoothed (np.ndarray)
        Same shape as `data`.
    """

    if sigma <= 0:
        return data
    nan_mask = np.isnan(data)
    if not nan_mask.any():
        return ndimage.gaussian_filter(data, sigma=sigma, mode="nearest")
    weights = (~nan_mask).astype(float)
    data_filled = np.where(nan_mask, 0.0, data)
    smoothed_data = ndimage.gaussian_filter(data_filled, sigma=sigma, mode="nearest")
    smoothed_weights = ndimage.gaussian_filter(weights, sigma=sigma, mode="nearest")
    out = np.full_like(data, np.nan, dtype=float)
    valid = smoothed_weights > 0
    out[valid] = smoothed_data[valid] / smoothed_weights[valid]
    if preserve_nan:
        out[nan_mask] = np.nan
    return out


# triage-stat helpers (per-cluster scalar summaries written into
# `triage_stats` at compute time, consumed downstream by
# `unit_triage_aggregator`).


def _longest_run(mask: np.ndarray, *, circular: bool = False) -> tuple[int, int, int]:
    """
    Description
    -----------
    Find the longest contiguous run of True in a 1D bool array.

    Parameters
    ----------
    mask (np.ndarray of bool, shape (n_bins,))
    circular (bool)
        If True, a run that ends at the last bin and continues into the
        first bin is treated as one contiguous run. The reported
        end_idx may then be < start_idx (the run wraps).

    Returns
    -------
    start_idx, end_idx, length
        Indices of the longest run (inclusive). When circular and the
        run wraps, end_idx < start_idx and length counts both segments.
        Returns (-1, -1, 0) if `mask` has no True bins.
    """

    if mask.size == 0 or not mask.any():
        return -1, -1, 0
    n = int(mask.size)
    if circular:
        # Doubled-mask trick: a wrap-around run becomes contiguous in
        # the concatenation. Cap the length at n so an all-True input
        # doesn't double-count.
        m = np.concatenate([mask, mask])
    else:
        m = mask
    ext = np.concatenate([[False], m, [False]]).astype(int)
    diffs = np.diff(ext)
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0] - 1
    if starts.size == 0:
        return -1, -1, 0
    lengths = ends - starts + 1
    if circular:
        lengths = np.minimum(lengths, n)
    best = int(np.argmax(lengths))
    length = int(lengths[best])
    s = int(starts[best]) % n
    e = int(starts[best] + length - 1) % n
    return s, e, length


def _peak_z_info(
    rate: np.ndarray, null_mean: np.ndarray, null_std: np.ndarray
) -> tuple[float, int, float]:
    """
    Description
    -----------
    Locate the bin with maximum |z|, where z = (rate − null_mean) / null_std.

    Parameters
    ----------
    rate, null_mean, null_std (np.ndarray, same shape)

    Returns
    -------
    peak_abs_z (float)
    peak_idx (int)        Flat index into the input arrays.
    peak_signed_z (float)
        The signed z at the peak (positive = excitation,
        negative = suppression). NaN if no valid bins.
    """

    rate = np.asarray(rate, dtype=float).ravel()
    null_mean = np.asarray(null_mean, dtype=float).ravel()
    null_std = np.asarray(null_std, dtype=float).ravel()
    valid = (
        np.isfinite(rate)
        & np.isfinite(null_mean)
        & np.isfinite(null_std)
        & (null_std > 0)
    )
    if not valid.any():
        return float("nan"), -1, float("nan")
    z = np.full_like(rate, np.nan)
    z[valid] = (rate[valid] - null_mean[valid]) / null_std[valid]
    abs_z = np.abs(z)
    peak_idx = int(np.nanargmax(abs_z))
    return float(abs_z[peak_idx]), peak_idx, float(z[peak_idx])


def _run_analysis(
    rate: np.ndarray,
    null_p_low: np.ndarray,
    null_p_high: np.ndarray,
    null_mean: np.ndarray,
    null_std: np.ndarray,
    *,
    circular: bool = False,
) -> dict:
    """
    Description
    -----------
    Per-direction divergence-segment analysis on a 1D ratemap. For each
    of {excitation, suppression}, where excitation = `rate > null_p_high`
    and suppression = `rate < null_p_low`, return:
      * `n_bins` — total bins on that side anywhere in the array
      * `max_run` — length of the longest contiguous run on that side
      * `run_start_idx`, `run_end_idx` — endpoints (inclusive) of that run
      * `peak_idx` — index of the most-extreme z within that run
      * `peak_z` — the signed z value at `peak_idx`

    For `circular=True` the run search wraps around (relevant for
    behavioral features like head direction).

    Parameters
    ----------

    Returns
    -------
    {"excit": {...}, "suppress": {...}}
    """

    rate = np.asarray(rate, dtype=float).ravel()
    null_p_low = np.asarray(null_p_low, dtype=float).ravel()
    null_p_high = np.asarray(null_p_high, dtype=float).ravel()
    null_mean = np.asarray(null_mean, dtype=float).ravel()
    null_std = np.asarray(null_std, dtype=float).ravel()
    n = rate.size

    valid = (
        np.isfinite(rate) & np.isfinite(null_p_low) & np.isfinite(null_p_high)
    )
    excit_mask = valid & (rate > null_p_high)
    suppress_mask = valid & (rate < null_p_low)

    z = np.full_like(rate, np.nan)
    z_valid = (
        valid
        & np.isfinite(null_mean)
        & np.isfinite(null_std)
        & (null_std > 0)
    )
    z[z_valid] = (rate[z_valid] - null_mean[z_valid]) / null_std[z_valid]

    out: dict = {}
    for direction, mask in (("excit", excit_mask), ("suppress", suppress_mask)):
        n_bins = int(mask.sum())
        s_idx, e_idx, length = _longest_run(mask, circular=circular)
        if length > 0:
            if circular and e_idx < s_idx:
                run_indices = np.concatenate(
                    [np.arange(s_idx, n), np.arange(0, e_idx + 1)]
                )
            else:
                run_indices = np.arange(s_idx, e_idx + 1)
            run_z = z[run_indices]
            if not np.any(np.isfinite(run_z)):
                peak_idx = int(run_indices[0])
            elif direction == "excit":
                pi_local = int(np.nanargmax(run_z))
                peak_idx = int(run_indices[pi_local])
            else:
                pi_local = int(np.nanargmin(run_z))
                peak_idx = int(run_indices[pi_local])
            peak_z = float(z[peak_idx]) if np.isfinite(z[peak_idx]) else float("nan")
        else:
            peak_idx = -1
            peak_z = float("nan")
        out[direction] = {
            "n_bins": n_bins,
            "max_run": int(length),
            "run_start_idx": int(s_idx),
            "run_end_idx": int(e_idx),
            "peak_idx": int(peak_idx),
            "peak_z": peak_z,
        }
    return out


def _selectivity_index(rate: np.ndarray) -> float:
    """
    Description
    -----------
    `(max - min) / (max + min)` over the finite values of `rate`.
    Higher = more selective. NaN if undefined.

    Parameters
    ----------
    rate (np.ndarray)
        1D ratemap.

    Returns
    -------
    si (float)
        Selectivity index in [0, 1], or NaN if all-NaN / max+min == 0.
    """

    finite = np.asarray(rate, dtype=float).ravel()
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    rmax = float(finite.max())
    rmin = float(finite.min())
    s = rmax + rmin
    if s == 0:
        return float("nan")
    return (rmax - rmin) / s


def _monotonicity_spearman(rate: np.ndarray) -> float:
    """
    Description
    -----------
    Spearman ρ between bin index and rate, ignoring NaN bins. Captures
    "rate increases monotonically with the property axis". NaN if
    fewer than 3 finite bins.

    Parameters
    ----------
    rate (np.ndarray)
        1D ratemap.

    Returns
    -------
    rho (float)
        Spearman rank correlation in [-1, 1], or NaN if undefined.
    """

    rate = np.asarray(rate, dtype=float).ravel()
    valid = np.isfinite(rate)
    if int(valid.sum()) < 3:
        return float("nan")
    idx = np.arange(rate.size)[valid]
    try:
        r, _ = stats.spearmanr(idx, rate[valid])
    except (ValueError, RuntimeError):
        return float("nan")
    return float(r) if np.isfinite(r) else float("nan")


def _skaggs_info_rate_bps(
    rate: np.ndarray, occupancy_seconds: np.ndarray
) -> float:
    """
    Description
    -----------
    Skaggs information rate (bits/spike) for a 1D or 2D ratemap.

      I = sum_i p_i * (r_i / R) * log2(r_i / R)

    where p_i = occupancy fraction, r_i = rate at bin i, and
    R = sum_i p_i * r_i (occupancy-weighted mean rate). Bins with
    occupancy 0 or rate 0 are excluded.

    Parameters
    ----------
    rate (np.ndarray)
        1D or 2D ratemap (raveled internally).
    occupancy_seconds (np.ndarray)
        Occupancy time per bin (s), same shape as `rate`.

    Returns
    -------
    info_bps (float)
        Information rate in bits/spike, or NaN if undefined.
    """

    rate = np.asarray(rate, dtype=float).ravel()
    occ = np.asarray(occupancy_seconds, dtype=float).ravel()
    # `valid_occ` covers ALL bins with non-zero occupancy (including
    # bins where the cell happens to fire zero spikes). Those bins still
    # contribute time to the occupancy-weighted mean rate `R`, so they
    # must be in the p / R normalization. The Skaggs sum itself is
    # restricted to r_i > 0 because the term `p_i * (r_i/R) * log2(r_i/R)`
    # has the limit 0 as r_i -> 0 but numerically blows up there.
    valid_occ = np.isfinite(rate) & np.isfinite(occ) & (occ > 0)
    if not valid_occ.any():
        return float("nan")
    p = occ[valid_occ] / occ[valid_occ].sum()
    r = rate[valid_occ]
    R = float((p * r).sum())
    if R <= 0:
        return float("nan")
    nz = r > 0
    if not nz.any():
        return float("nan")
    return float((p[nz] * (r[nz] / R) * np.log2(r[nz] / R)).sum())


def _skaggs_sparsity(
    rate: np.ndarray, occupancy_seconds: np.ndarray
) -> float:
    """
    Description
    -----------
    Skaggs sparsity: `(sum_i p_i r_i)^2 / sum_i p_i r_i^2`. Smaller =
    sparser (more peaked). 1 = uniform.

    Parameters
    ----------
    rate (np.ndarray)
        1D or 2D ratemap.
    occupancy_seconds (np.ndarray)
        Occupancy time per bin (s), same shape as `rate`.

    Returns
    -------
    sparsity (float)
        Sparsity in (0, 1], or NaN if undefined.
    """

    rate = np.asarray(rate, dtype=float).ravel()
    occ = np.asarray(occupancy_seconds, dtype=float).ravel()
    valid = np.isfinite(rate) & np.isfinite(occ) & (occ > 0)
    if not valid.any():
        return float("nan")
    p = occ[valid] / occ[valid].sum()
    r = rate[valid]
    num = float((p * r).sum()) ** 2
    denom = float((p * r ** 2).sum())
    if denom <= 0:
        return float("nan")
    return float(num / denom)


def _spatial_coherence(rate_2d: np.ndarray) -> float:
    """
    Description
    -----------
    Pearson correlation between each bin's rate and the mean of its
    8-neighborhood (3x3 box minus center). Standard "spatial coherence"
    diagnostic for 2D ratemaps. NaN bins are skipped pairwise. Returns
    NaN if fewer than 3 valid pixel-pairs survive.

    Parameters
    ----------
    rate_2d (np.ndarray, 2D)
        Spatial firing-rate map.

    Returns
    -------
    coherence (float)
        Pearson r in [-1, 1], or NaN if undefined.
    """

    a = np.asarray(rate_2d, dtype=float)
    if a.ndim != 2:
        return float("nan")
    kernel = np.array([[1.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
    valid_mask = np.isfinite(a).astype(float)
    a_filled = np.where(np.isfinite(a), a, 0.0)
    neighbor_sum = ndimage.convolve(a_filled, kernel, mode="constant", cval=0.0)
    neighbor_count = ndimage.convolve(valid_mask, kernel, mode="constant", cval=0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        neighbor_mean = neighbor_sum / neighbor_count
    valid = (
        np.isfinite(a)
        & np.isfinite(neighbor_mean)
        & (neighbor_count > 0)
    )
    if int(valid.sum()) < 3:
        return float("nan")
    try:
        r = float(np.corrcoef(a[valid], neighbor_mean[valid])[0, 1])
    except (ValueError, FloatingPointError):
        return float("nan")
    return r if np.isfinite(r) else float("nan")


def _ramp_index(rate: np.ndarray, bin_centers_s: np.ndarray) -> float:
    """
    Description
    -----------
    Two-point ramp shape descriptor for a pre-USV PETH:

      `ramp = (rate(t≈-0.1) - rate(t≈-1.5)) / (rate(t≈-0.1) + rate(t≈-1.5))`

    Returns NaN if either anchor bin is non-finite or the sum is 0.
    Independent of any shuffle test — purely a shape metric.

    Parameters
    ----------
    rate (np.ndarray, shape (n_bins,))
        PETH rate (smoothed if available).
    bin_centers_s (np.ndarray, shape (n_bins,))
        Bin centers relative to USV onset (s); negative values
        precede the anchor.

    Returns
    -------
    ramp (float)
        Two-point ramp index in [-1, 1], or NaN if either anchor bin
        is non-finite or `rate(-0.1) + rate(-1.5) == 0`.
    """

    rate = np.asarray(rate, dtype=float).ravel()
    centers = np.asarray(bin_centers_s, dtype=float).ravel()
    if rate.size != centers.size or rate.size == 0:
        return float("nan")
    near_idx = int(np.argmin(np.abs(centers - (-0.1))))
    far_idx = int(np.argmin(np.abs(centers - (-1.5))))
    rn = rate[near_idx]
    rf = rate[far_idx]
    if not (np.isfinite(rn) and np.isfinite(rf)):
        return float("nan")
    s = float(rn + rf)
    if s == 0:
        return float("nan")
    return float((rn - rf) / s)


# unified class


class NeuronalTuning(FeatureZoo):
    """
    Description
    -----------
    Per-cluster neuronal tuning compute: behavioral and vocal in a single
    pass over the cluster .npy files. Each side is internally gated on
    the presence of its own required inputs; missing-input is a graceful
    skip; only the absence of spike .npy files is a hard error.

    Parameters
    ----------
    root_directory (str)
        Session root directory.
    tuning_parameters_dict (dict)
        Settings dictionary; keys read by behavioral path:
        `temporal_offsets`, `n_shuffles`, `total_bin_num`,
        `n_spatial_bins`, `spatial_scale_cm`. Keys read by vocal path:
        `n_shuffles`, `total_bin_num`, `shuffle_seconds_range`,
        `peth_window_seconds`, `peth_bin_seconds`, `bout_quiet_seconds`,
        `vocal_require_clean_post_anchor`,
        `vocal_require_clean_prior_anchor`, `n_usv_min_self`,
        `n_usv_min_partner`, `n_usv_min_category`,
        `usv_property_min_occupancy_seconds`,
        `include_partner_vocalization_tuning_bool`,
        `shuffle_chunk_size`.
    message_output (Callable)
        Logger; defaults to print.
    """

    def __init__(self, **kwargs):
        """
        Description
        -----------
        Initialize the unified per-cluster compute. Loads `FeatureZoo`
        feature definitions (boundaries / labels / vocal segmentation
        metadata), stashes any keyword arguments as attributes (notably
        `root_directory`, `tuning_parameters_dict`, `message_output`),
        records GUI-vs-CLI execution context, and pins the path of the
        bundled UMAP segmentation file used by the categorical vocal
        compute.

        Parameters
        ----------
        **kwargs
            Forwarded as-is to `self.__dict__`. Expected keys include
            `root_directory`, `tuning_parameters_dict`, `message_output`.

        Returns
        -------
        None
        """

        FeatureZoo.__init__(self)
        for kw_arg, kw_val in kwargs.items():
            self.__dict__[kw_arg] = kw_val
        self.app_context_bool = is_gui_context()
        self._segmentation_path = (
            pathlib.Path(__file__).parent.parent
            / "_config"
            / "vocal_umap_segmentation.npz"
        )

    # pkl merge helper (used by both paths)

    def _save_partial_to_cluster_pkl(
        self, cluster_id: str, partial_payload: dict
    ) -> None:
        """
        Description
        -----------
        Load `<cluster_id>_tuning_curves_data.pkl` if it exists, merge
        the supplied partial payload into it (top-level keys only), and
        write back. Creates the file if absent. The behavioral path
        contributes flat `beh_offset=*s` keys; the vocal path contributes
        `usv_peth`, `usv_property_tuning`, `usv_category_tuning`, `usv_category_peth`, and
        `usv_metadata`. Order of the two paths therefore does not
        matter.

        Parameters
        ----------
        cluster_id (str)
            Cluster identifier (file stem of the spike .npy).
        partial_payload (dict)
            Top-level keys to merge into the per-cluster pkl.

        Returns
        -------
        None
        """

        out_dir = pathlib.Path(self.root_directory) / "ephys" / "tuning_curves"
        out_dir.mkdir(parents=True, exist_ok=True)
        pkl_path = out_dir / f"{cluster_id}_tuning_curves_data.pkl"
        if pkl_path.exists():
            with pkl_path.open("rb") as fh:
                existing = pickle.load(fh)
        else:
            existing = {}
        # `triage_stats` is the one top-level key that *both* compute
        # paths write into (behavioral writes its modality, vocal writes
        # vmi + its modalities). A naive wholesale `update` would
        # clobber whichever path wrote first; instead one-level-deep
        # merge that key so e.g. existing["triage_stats"]["vmi"] survives
        # when behavioral later adds existing["triage_stats"]["behavioral"].
        for top_key, new_value in partial_payload.items():
            if (
                top_key == "triage_stats"
                and isinstance(new_value, dict)
                and isinstance(existing.get(top_key), dict)
            ):
                existing[top_key] = {**existing[top_key], **new_value}
            else:
                existing[top_key] = new_value
        with pkl_path.open("wb") as fh:
            pickle.dump(existing, fh)

    # input loaders (each is graceful — returns None on missing)

    def _load_behavioral_inputs(self) -> dict | None:
        """
        Description
        -----------
        Locate `*_behavioral_features.csv` and the tracking H5; load
        behavioral features into a Polars DataFrame and read the camera
        frame rate + animal IDs from the H5. Returns None if either file
        is missing.

        Parameters
        ----------

        Returns
        -------
        bundle (dict | None)
            None if any required input is missing; otherwise dict with
            keys `behavioral_data` (pls.DataFrame), `animal_ids`
            (list[str]), `empirical_camera_sr` (float).
        """

        root = pathlib.Path(self.root_directory)
        try:
            behavioral_data_file = first_match_or_raise(
                root=root,
                pattern="*_behavioral_features.csv*",
                recursive=True,
                label="behavioral features CSV",
            )
        except (StopIteration, FileNotFoundError):
            return None
        try:
            mouse_data_h5 = first_match_or_raise(
                root=root,
                pattern="[!speaker]*_points3d_translated_rotated_metric.h5*",
                recursive=True,
                label="translated/rotated mouse points3d .h5",
            )
        except (StopIteration, FileNotFoundError):
            return None

        behavioral_data = pls.read_csv(str(behavioral_data_file))
        with h5py.File(mouse_data_h5, mode="r") as tracking_data_3d:
            animal_ids = [
                t.decode("utf-8").strip() for t in tracking_data_3d["track_names"]
            ]
            empirical_camera_sr = float(tracking_data_3d["recording_frame_rate"][()])

        return {
            "behavioral_data": behavioral_data,
            "animal_ids": animal_ids,
            "empirical_camera_sr": empirical_camera_sr,
        }

    def _load_vocal_inputs(self) -> dict | None:
        """
        Description
        -----------
        Locate `*_usv_summary.csv` and the tracking H5; filter the USV
        summary to non-noise rows, read sex assignment from h5
        `track_names`, and derive session duration from the H5 as
        `tracks.shape[0] / recording_frame_rate` (same time base the
        spikes are aligned to). Returns None if either required file
        is missing or the filtered USV table is empty.

        Parameters
        ----------

        Returns
        -------
        bundle (dict | None)
            None if any required input is missing; otherwise a dict with
            keys: `usv_df` (filtered pls.DataFrame), `track_names`,
            `male`, `female`, `duration_seconds`, `starts`, `stops`,
            `emitters`.
        """

        root = pathlib.Path(self.root_directory)
        try:
            usv_csv = first_match_or_raise(
                root=root,
                pattern="*_usv_summary.csv",
                recursive=True,
                label="USV summary CSV",
            )
        except (StopIteration, FileNotFoundError):
            return None

        df = pls.read_csv(str(usv_csv))
        if "vae_supercategory" not in df.columns:
            return None
        df = df.filter(pls.col("vae_supercategory") != 0)
        if df.shape[0] == 0:
            return None

        emitters = [
            (e.strip() if e is not None else None) for e in df["emitter"].to_list()
        ]
        starts = df["start"].to_numpy()
        stops = df["stop"].to_numpy()

        try:
            h5_path = first_match_or_raise(
                root=root,
                pattern="[!speaker]*_points3d_translated_rotated_metric.h5*",
                recursive=True,
                label="translated/rotated mouse points3d .h5",
            )
        except (StopIteration, FileNotFoundError):
            return None
        # Session duration is derived from the tracking H5: the leading
        # axis of `tracks` is the frame count, divided by the camera
        # frame rate gives seconds. This is the SAME time base the
        # spike data is aligned to (and that the behavioral compute
        # uses), so circular shuffles wrap modulo the right interval.
        with h5py.File(h5_path, mode="r") as f:
            track_names = [t.decode("utf-8").strip() for t in f["track_names"]]
            n_frames = int(f["tracks"].shape[0])
            recording_fr = float(f["recording_frame_rate"][()])
        duration_seconds = float(n_frames) / recording_fr
        male = track_names[0] if len(track_names) >= 1 else None
        female = track_names[1] if len(track_names) >= 2 else None

        return {
            "usv_df": df,
            "track_names": track_names,
            "male": male,
            "female": female,
            "duration_seconds": duration_seconds,
            "starts": starts,
            "stops": stops,
            "emitters": np.array(emitters, dtype=object),
        }

    # main entry point

    def calculate_neuronal_tuning_curves(self) -> None:
        """
        Description
        -----------
        Run both compute paths in sequence over the same set of cluster
        .npy files. Cluster discovery is shared. Behavioral runs first
        if its inputs are present; vocal runs second if its inputs are
        present. Each per-cluster output is merged into a single pkl per
        cluster.

        Raises FileNotFoundError if no spike .npy files are found under
        `<root>/ephys/**/cluster_data/` (no-cluster sessions are not a
        valid input). Missing behavioral CSV or vocal CSV are logged and
        the corresponding side is skipped.

        Parameters
        ----------

        Returns
        -------
        None
        """

        message_output = (
            self.message_output if hasattr(self, "message_output") else print
        )
        message_output(
            f"Computing neuronal tuning curves started at: "
            f"{datetime.now().hour:02d}:{datetime.now().minute:02d}:{datetime.now().second:02d}"
        )
        smart_wait(app_context_bool=self.app_context_bool, seconds=1)

        root = pathlib.Path(self.root_directory)
        cluster_files = sorted(
            (root / "ephys").rglob("cluster_data/*.npy"),
            key=lambda p: p.name,
        )
        if not cluster_files:
            err_msg = (
                f"No spike .npy files under {root}/ephys/**/cluster_data/. "
                "Cannot compute tuning curves without cluster data."
            )
            raise FileNotFoundError(err_msg)

        # behavioral side
        beh_inputs = self._load_behavioral_inputs()
        if beh_inputs is None:
            message_output(
                "  behavioral skipped: missing *_behavioral_features.csv "
                "or tracking .h5 in this session."
            )
        else:
            message_output("  computing behavioral tuning curves ...")
            for cluster_file in tqdm(
                cluster_files, desc="behavioral tuning per cluster"
            ):
                beh_partial = self._compute_one_cluster_behavioral(
                    cluster_file=cluster_file,
                    beh_inputs=beh_inputs,
                )
                self._save_partial_to_cluster_pkl(cluster_file.stem, beh_partial)

        # vocal side
        voc_inputs = self._load_vocal_inputs()
        if voc_inputs is None:
            message_output(
                "  vocal skipped: missing *_usv_summary.csv or tracking .h5 "
                "in this session."
            )
            return

        side_precompute = self._build_vocal_side_precompute(voc_inputs)
        if side_precompute is None:
            message_output(
                "  vocal skipped: no emitter side passes n_usv_min_self "
                f"({int(self.tuning_parameters_dict['n_usv_min_self'])})."
            )
            return

        message_output("  computing vocal tuning curves ...")
        for cluster_file in tqdm(cluster_files, desc="vocal tuning per cluster"):
            try:
                voc_partial = self._compute_one_cluster_vocal(
                    cluster_file=cluster_file,
                    voc_inputs=voc_inputs,
                    side_precompute=side_precompute,
                )
            except Exception as exc:
                message_output(
                    f"    vocal: cluster {cluster_file.name} failed: {exc}"
                )
                continue
            self._save_partial_to_cluster_pkl(cluster_file.stem, voc_partial)

    # behavioral per-cluster compute

    def _compute_one_cluster_behavioral(
        self,
        cluster_file: pathlib.Path,
        beh_inputs: dict,
    ) -> dict:
        """
        Description
        -----------
        Compute behavioral tuning for one cluster across every value in
        `temporal_offsets`. Returns a dict whose top-level keys are
        `f"beh_offset={offset}s"` plus a single `behavioral_metadata`
        block; each offset block maps to a per-feature dict with shape:

        1D feature columns
            rate                (n_bins,)        — observed firing rate (sp/s)
            occupancy_seconds   (n_bins,)        — per-bin total time (s)
            null_mean / null_std / null_p0_5 / null_p2_5 / null_p97_5 /
            null_p99_5          (n_bins,)        — shuffle-null rate stats
            bin_centers         (n_bins,)
            bin_edges           (n_bins+1,)

        2D per-animal `<animal_id>.space` (no shuffles in current impl)
            rate                (b, b)
            occupancy_seconds   (b, b)
            bin_centers         (b,)
            bin_edges           (b+1,)

        Storage convention matches the vocal-side pkl: rates and per-cell
        shuffle summary stats only — the full (n_shuffles, n_bins)
        shuffle array is reduced before saving so per-cluster pkls stay
        small.

        Parameters
        ----------
        cluster_file (pathlib.Path)
            Path to cluster .npy (row 1 = spike-frame indices).
        beh_inputs (dict)
            Output of `_load_behavioral_inputs`.

        Returns
        -------
        partial (dict)
            Top-level keys merge directly into the cluster pkl.
        """

        params = self.tuning_parameters_dict
        behavioral_data = beh_inputs["behavioral_data"]
        animal_ids = beh_inputs["animal_ids"]
        empirical_camera_sr = beh_inputs["empirical_camera_sr"]
        smoothing_sd = float(params.get("smoothing_sd", 0.0))

        cluster_data_frames_original = np.load(file=cluster_file)[1, :]
        partial: dict = {}

        for one_offset in params["temporal_offsets"]:
            cluster_data_frames = cluster_data_frames_original.astype(np.int32) + int(
                np.floor(one_offset * empirical_camera_sr)
            )
            cluster_data_frames = cluster_data_frames[
                (cluster_data_frames >= 0)
                & (cluster_data_frames < behavioral_data.shape[0])
            ]
            file_name_addendum_offset = f"beh_offset={one_offset}s"
            partial[file_name_addendum_offset] = {}

            cluster_data_shuffled = shuffle_spikes(
                spike_array=cluster_data_frames,
                total_fr_num=behavioral_data.shape[0],
                shuffle_min_fr=int(np.floor(20 * empirical_camera_sr)),
                shuffle_max_fr=int(np.floor(60 * empirical_camera_sr)),
                n_shuffles=params["n_shuffles"],
                seed=params["shuffle_seed"],
            )

            # 1D feature ratemaps for every non-spatial column
            for column in behavioral_data.columns:
                if column.split(".")[-1] in ("spaceX", "spaceY", "spaceZ"):
                    continue
                ratemap_counts, sh_counts, bin_centers, bin_edges = generate_ratemaps(
                    feature_arr=np.array(behavioral_data[column]),
                    spike_arr=cluster_data_frames,
                    shuffled_spike_arr=cluster_data_shuffled,
                    min_val=self.feature_boundaries[column.split(".")[-1]][0],
                    max_val=self.feature_boundaries[column.split(".")[-1]][1],
                    num_bins=params["total_bin_num"],
                    camera_fr=empirical_camera_sr,
                    space_bool=False,
                )
                spike_count = ratemap_counts[:, 0]
                occupancy_s = ratemap_counts[:, 1]
                ok = occupancy_s > 0
                rate = np.full(spike_count.shape, np.nan, dtype=float)
                rate[ok] = spike_count[ok] / occupancy_s[ok]
                # convert shuffle counts to shuffle rates by dividing by occupancy
                # (broadcast across the shuffle axis); bins with zero occupancy stay NaN
                sh_rates = np.full(sh_counts.shape, np.nan, dtype=float)
                sh_rates[:, ok] = sh_counts[:, ok] / occupancy_s[ok]
                stats = _percentiles_block(sh_rates)
                feature_payload = {
                    "rate": rate,
                    "occupancy_seconds": occupancy_s,
                    "bin_centers": bin_centers,
                    "bin_edges": bin_edges,
                    **stats,
                }
                if smoothing_sd > 0:
                    # smooth observed and EACH per-shuffle rate map first,
                    # THEN reduce shuffles via percentiles. This is the
                    # statistically correct order: it tightens the null
                    # distribution coherently rather than blurring an
                    # already-reduced percentile array.
                    # Plain D NaN policy: BOTH observed rate AND each
                    # shuffle's rate map are smoothed with
                    # `preserve_nan=False` so the weighted Gaussian
                    # interpolates over low-occupancy gaps. This keeps
                    # the smoothed rate line and the smoothed shuffle
                    # bands at the same x-extent in plots; the raw
                    # arrays still record NaN at sparse bins for any
                    # downstream analysis that needs the explicit
                    # sparsity signal.
                    rate_smoothed = _gaussian_smooth_1d(
                        rate, smoothing_sd, axis=-1, preserve_nan=False
                    )
                    sh_rates_smoothed = _gaussian_smooth_1d(
                        sh_rates, smoothing_sd, axis=-1, preserve_nan=False
                    )
                    stats_smoothed = _percentiles_block(sh_rates_smoothed)
                    feature_payload["rate_smoothed"] = rate_smoothed
                    feature_payload.update(
                        {f"{k}_smoothed": v for k, v in stats_smoothed.items()}
                    )
                partial[file_name_addendum_offset][column] = feature_payload

            # one 2D spatial ratemap per animal that has both spaceX and spaceY columns
            for animal_id in animal_ids:
                spaceX_col = f"{animal_id}.spaceX"
                spaceY_col = f"{animal_id}.spaceY"
                if (
                    spaceX_col not in behavioral_data.columns
                    or spaceY_col not in behavioral_data.columns
                ):
                    continue
                space_key = f"{animal_id}.space"
                ratemap_2d, sp_bin_centers, sp_bin_edges = generate_ratemaps(
                    feature_arr=np.stack(
                        arrays=(
                            np.array(behavioral_data[spaceX_col]),
                            np.array(behavioral_data[spaceY_col]),
                        ),
                        axis=1,
                    ),
                    spike_arr=cluster_data_frames,
                    shuffled_spike_arr=cluster_data_shuffled,
                    min_val=-params["spatial_scale_cm"],
                    max_val=params["spatial_scale_cm"],
                    num_bins=params["n_spatial_bins"],
                    camera_fr=empirical_camera_sr,
                    space_bool=True,
                )
                spike_count_2d = ratemap_2d[:, :, 0]
                occupancy_s_2d = ratemap_2d[:, :, 1]
                ok_2d = occupancy_s_2d > 0
                rate_2d = np.full(spike_count_2d.shape, np.nan, dtype=float)
                rate_2d[ok_2d] = spike_count_2d[ok_2d] / occupancy_s_2d[ok_2d]
                space_payload = {
                    "rate": rate_2d,
                    "occupancy_seconds": occupancy_s_2d,
                    "bin_centers": sp_bin_centers,
                    "bin_edges": sp_bin_edges,
                }
                if smoothing_sd > 0:
                    space_payload["rate_smoothed"] = _gaussian_smooth_2d(
                        rate_2d, smoothing_sd
                    )
                partial[file_name_addendum_offset][space_key] = space_payload

        partial["behavioral_metadata"] = {
            "cluster_id": cluster_file.stem,
            "session_root": str(self.root_directory),
            "n_shuffles": int(params["n_shuffles"]),
            "temporal_offsets": list(params["temporal_offsets"]),
            "total_bin_num": int(params["total_bin_num"]),
            "n_spatial_bins": int(params["n_spatial_bins"]),
            "spatial_scale_cm": params["spatial_scale_cm"],
            "empirical_camera_sr": float(empirical_camera_sr),
            "smoothing_sd": float(smoothing_sd),
            "behavioral_min_occupancy_seconds": float(
                params["behavioral_min_occupancy_seconds"]
            ),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        }
        self._attach_behavioral_triage_stats(partial, params)
        return partial

    # vocal session-level pre-compute

    def _build_vocal_side_precompute(self, voc_inputs: dict) -> dict | None:
        """
        Description
        -----------
        For each emitter side that passes the inclusion gates, build
        per-anchor bookkeeping that is independent of any spike train
        and can therefore be reused across observed + n_shuffles
        iterations and across clusters within a session.

        Side selection
        --------------
        - `male` = `track_names[0]`, `female` = `track_names[1]` (locked
          convention).
        - Sort sides by USV count; the more-vocal side is `self`,
          the less-vocal side is `partner`.
        - `self` included iff its USV count >= `n_usv_min_self`.
        - `partner` included iff
          `include_partner_vocalization_tuning_bool` AND its USV count
          >= `n_usv_min_partner`.

        Parameters
        ----------
        voc_inputs (dict)
            Output of `_load_vocal_inputs`.

        Returns
        -------
        precompute (dict | None)
            None if no eligible side exists; otherwise a dict whose keys
            are role labels (`"self"` and optionally `"partner"`), each
            mapping to per-side bookkeeping consumed by
            `_compute_one_cluster_vocal`. Also adds `_grid` carrying the
            PETH bin geometry shared across sides.
        """

        params = self.tuning_parameters_dict
        n_bins_default = int(params["total_bin_num"])
        peth_lo, peth_hi = (
            float(params["peth_window_seconds"][0]),
            float(params["peth_window_seconds"][1]),
        )
        peth_bin_s = float(params["peth_bin_seconds"])
        require_post = bool(params["vocal_require_clean_post_anchor"])
        require_prior = bool(params["vocal_require_clean_prior_anchor"])
        n_min_self = int(params["n_usv_min_self"])
        n_min_partner = int(params["n_usv_min_partner"])
        include_partner = bool(params["include_partner_vocalization_tuning_bool"])

        starts = voc_inputs["starts"]
        stops = voc_inputs["stops"]
        emitters = voc_inputs["emitters"]
        male = voc_inputs["male"]
        female = voc_inputs["female"]
        duration_seconds = voc_inputs["duration_seconds"]
        usv_df = voc_inputs["usv_df"]

        n_peth_bins = round((peth_hi - peth_lo) / peth_bin_s)
        rel_edges = peth_lo + np.arange(n_peth_bins + 1) * peth_bin_s
        rel_bin_lo = rel_edges[:-1]
        rel_bin_hi = rel_edges[1:]
        rel_bin_centers = 0.5 * (rel_bin_lo + rel_bin_hi)

        sides_to_run: list[dict] = []
        for emitter_str, sex_label in ((male, "male"), (female, "female")):
            if emitter_str is None:
                continue
            mask = np.array([e == emitter_str for e in emitters])
            n = int(mask.sum())
            if n == 0:
                continue
            sides_to_run.append(
                {"emitter": emitter_str, "sex": sex_label, "mask": mask, "n": n}
            )
        if not sides_to_run:
            return None

        sides_to_run.sort(key=lambda d: -d["n"])
        self_side = sides_to_run[0]
        partner_side = sides_to_run[1] if len(sides_to_run) > 1 else None

        sides_payload: list[dict] = []
        if self_side["n"] >= n_min_self:
            sides_payload.append({**self_side, "role": "self"})
        else:
            return None
        if (
            include_partner
            and partner_side is not None
            and partner_side["n"] >= n_min_partner
        ):
            sides_payload.append({**partner_side, "role": "partner"})

        side_precompute: dict[str, dict] = {}
        for side in sides_payload:
            anchor_idx = np.where(side["mask"])[0]
            anchor_starts = starts[anchor_idx]
            anchor_stops = stops[anchor_idx]
            anchor_durations = anchor_stops - anchor_starts

            within_valid = _within_usv_validity(anchor_idx, starts, stops)

            peth_valid = _anchor_bin_validity_grid(
                anchor_idx=anchor_idx,
                starts=starts,
                stops=stops,
                duration_seconds=duration_seconds,
                rel_bin_lo=rel_bin_lo,
                rel_bin_hi=rel_bin_hi,
                require_clean_post=require_post,
                require_clean_prior=require_prior,
            )
            peth_denom_seconds = peth_valid.sum(axis=0).astype(float) * peth_bin_s

            anchor_property_bin_idx: dict[str, np.ndarray] = {}
            anchor_property_occ_seconds: dict[str, np.ndarray] = {}
            for prop in CONTINUOUS_PROPERTIES:
                values = usv_df[prop].to_numpy()[anchor_idx]
                bin_range = self.vocal_boundaries[prop]
                n_bins_prop = _bin_count_for_property(prop, n_bins_default)
                idx = _bin_property_indices(values, bin_range, n_bins_prop)
                anchor_property_bin_idx[prop] = idx
                occ = np.zeros(n_bins_prop, dtype=float)
                w = np.where(within_valid & (idx >= 0), anchor_durations, 0.0)
                np.add.at(
                    occ,
                    np.clip(idx, 0, n_bins_prop - 1),
                    np.where(idx >= 0, w, 0.0),
                )
                anchor_property_occ_seconds[prop] = occ

            anchor_categorical: dict[str, dict] = {}
            for cat_feat in CATEGORICAL_FEATURES:
                cat_values = usv_df[cat_feat].to_numpy()[anchor_idx]
                non_null = (
                    cat_values[cat_values >= 0]
                    if cat_values.dtype.kind in "iu"
                    else cat_values
                )
                unique_cats = np.array(sorted(set(non_null.tolist())))
                cat_to_idx = {int(c): i for i, c in enumerate(unique_cats)}

                def _safe_idx(c, mapping=cat_to_idx):
                    """
                    Description
                    -----------
                    Map a category label to its dense index, returning
                    -1 for unassigned / NaN / non-numeric labels.

                    Parameters
                    ----------
                    c
                        Category label from the USV summary CSV.
                    mapping (dict[int, int])
                        `category_id -> dense_index` map (default-bound
                        to the enclosing scope's `cat_to_idx`).

                    Returns
                    -------
                    idx (int)
                        Dense index, or -1 if `c` is unassigned.
                    """
                    # int(NaN) raises ValueError; non-numeric raises TypeError;
                    # None signals an unassigned label. All three map to -1
                    # (the sentinel downstream callers skip).
                    if c is None:
                        return -1
                    try:
                        return mapping.get(int(c), -1)
                    except (ValueError, TypeError):
                        return -1

                anchor_cat_idx_dense = np.array(
                    [_safe_idx(c) for c in cat_values], dtype=np.int64
                )
                n_cats = unique_cats.size
                count_per_cat = np.zeros(n_cats, dtype=np.int64)
                occ_seconds_per_cat = np.zeros(n_cats, dtype=float)
                for i_c in cat_to_idx.values():
                    member = anchor_cat_idx_dense == i_c
                    count_per_cat[i_c] = int(member.sum())
                    occ_seconds_per_cat[i_c] = float(
                        anchor_durations[member & within_valid].sum()
                    )
                peth_denom_per_cat_bin = np.zeros(
                    (n_cats, n_peth_bins), dtype=float
                )
                for i_c in range(n_cats):
                    member = anchor_cat_idx_dense == i_c
                    if member.any():
                        peth_denom_per_cat_bin[i_c, :] = (
                            peth_valid[member].sum(axis=0).astype(float) * peth_bin_s
                        )
                anchor_categorical[cat_feat] = {
                    "unique_cats": unique_cats,
                    "anchor_cat_idx_dense": anchor_cat_idx_dense,
                    "count_per_cat": count_per_cat,
                    "occ_seconds_per_cat": occ_seconds_per_cat,
                    "peth_denom_per_cat_bin": peth_denom_per_cat_bin,
                }

            side_precompute[side["role"]] = {
                "side": side,
                "anchor_idx": anchor_idx,
                "anchor_starts": anchor_starts,
                "anchor_stops": anchor_stops,
                "within_valid": within_valid,
                "peth_valid": peth_valid,
                "peth_denom_seconds": peth_denom_seconds,
                "anchor_property_bin_idx": anchor_property_bin_idx,
                "anchor_property_occ_seconds": anchor_property_occ_seconds,
                "anchor_categorical": anchor_categorical,
            }

        side_precompute["_grid"] = {
            "rel_bin_lo": rel_bin_lo,
            "rel_bin_hi": rel_bin_hi,
            "rel_bin_centers": rel_bin_centers,
        }
        return side_precompute

    # triage-stat attachers (compute scalar summaries per modality
    # from the rate / null arrays already produced by the main compute,
    # and write them into `partial["triage_stats"][...]`).

    def _attach_behavioral_triage_stats(
        self, partial: dict, params: dict
    ) -> None:
        """
        Description
        -----------
        Walk every `beh_offset=*s` block in `partial` and add per-feature
        triage stats. 1D features land in `triage_stats["behavioral"]`;
        2D spatial features (any feature whose name component contains
        "space") land in `triage_stats["spatial"]` with Skaggs info /
        sparsity / coherence in addition to peak-z. Circular axes
        (configured in `params["circular_features"]`) get wrap-around
        run detection.

        Parameters
        ----------
        partial (dict)
            The behavioral compute payload (keys `beh_offset=*s`,
            `behavioral_metadata`, possibly `triage_stats`).
        params (dict)
            `calculate_neuronal_tuning_curves` settings block. Reads
            `circular_features`.

        Returns
        -------
        None
        """

        circular_set = set(params.get("circular_features", []))
        partial.setdefault("triage_stats", {})
        partial["triage_stats"].setdefault("behavioral", {})
        partial["triage_stats"].setdefault("spatial", {})

        beh_offset_keys = [k for k in partial if k.startswith("beh_offset=")]
        for offset_key in beh_offset_keys:
            beh_block: dict = {}
            spatial_block: dict = {}
            for feature_key, payload in partial[offset_key].items():
                feat_name = feature_key.split(".")[-1]
                rate = payload.get("rate_smoothed", payload["rate"])
                occ = payload["occupancy_seconds"]

                if "space" in feat_name:
                    # 2D spatial maps are computed without shuffle nulls,
                    # so peak_z is not defined. Report the unshuffled
                    # peak rate + its location plus Skaggs info /
                    # sparsity / coherence (the standard place-cell
                    # diagnostics).
                    rate_arr = np.asarray(rate, dtype=float)
                    peak_rate = float("nan")
                    peak_row = peak_col = -1
                    finite_2d = np.isfinite(rate_arr)
                    if finite_2d.any():
                        flat_idx = int(
                            np.nanargmax(np.where(finite_2d, rate_arr, -np.inf))
                        )
                        pr, pc = (
                            int(x)
                            for x in np.unravel_index(flat_idx, rate_arr.shape)
                        )
                        peak_row, peak_col = pr, pc
                        peak_rate = float(rate_arr[pr, pc])
                    spatial_block[feature_key] = {
                        "peak_rate_sps": peak_rate,
                        "peak_row": peak_row,
                        "peak_col": peak_col,
                        "info_rate_bps": _skaggs_info_rate_bps(rate, occ),
                        "sparsity": _skaggs_sparsity(rate, occ),
                        "coherence": _spatial_coherence(rate),
                    }
                    continue

                # 1D feature
                bin_centers = payload["bin_centers"]
                null_mean = payload.get(
                    "null_mean_smoothed", payload["null_mean"]
                )
                null_std = payload.get(
                    "null_std_smoothed", payload["null_std"]
                )
                null_lo = payload.get(
                    "null_p0_5_smoothed", payload["null_p0_5"]
                )
                null_hi = payload.get(
                    "null_p99_5_smoothed", payload["null_p99_5"]
                )
                is_circular = feat_name in circular_set
                runs = _run_analysis(
                    rate, null_lo, null_hi, null_mean, null_std,
                    circular=is_circular,
                )
                # decorate runs with axis-value bounds for human consumption
                for direction in ("excit", "suppress"):
                    info = runs[direction]
                    if info["max_run"] > 0:
                        s_idx, e_idx = info["run_start_idx"], info["run_end_idx"]
                        info["range_low"] = float(bin_centers[s_idx])
                        info["range_high"] = float(bin_centers[e_idx])
                    else:
                        info["range_low"] = float("nan")
                        info["range_high"] = float("nan")
                    info["peak_bin_value"] = (
                        float(bin_centers[info["peak_idx"]])
                        if info["peak_idx"] >= 0 else float("nan")
                    )

                peak_abs_z, peak_idx, peak_signed = _peak_z_info(
                    rate, null_mean, null_std
                )
                beh_block[feature_key] = {
                    "peak_abs_z": peak_abs_z,
                    "peak_signed_z": peak_signed,
                    "peak_idx": int(peak_idx),
                    "peak_bin_value": (
                        float(bin_centers[peak_idx])
                        if peak_idx >= 0 else float("nan")
                    ),
                    "selectivity": _selectivity_index(rate),
                    "monotonicity": _monotonicity_spearman(rate),
                    "is_circular": bool(is_circular),
                    "excit": runs["excit"],
                    "suppress": runs["suppress"],
                }
            partial["triage_stats"]["behavioral"][offset_key] = beh_block
            partial["triage_stats"]["spatial"][offset_key] = spatial_block

    def _attach_vocal_triage_stats(
        self, partial: dict, params: dict
    ) -> None:
        """
        Description
        -----------
        Walk the vocal payloads (`usv_peth`, `usv_property_tuning`,
        `usv_category_tuning`, `usv_category_peth`) and add per-emitter /
        per-property / per-category triage stats. None of the vocal
        axes are circular.

        Parameters
        ----------
        partial (dict)
            The vocal compute payload. Must already contain the four
            vocal blocks plus `triage_stats["vmi"]` (added by the VMI
            block above).
        params (dict)
            `calculate_neuronal_tuning_curves` settings block (currently
            unused here; reserved for future thresholds).

        Returns
        -------
        None
        """

        del params  # unused (reserved)
        partial.setdefault("triage_stats", {})
        ts = partial["triage_stats"]
        ts.setdefault("usv_peth", {})
        ts.setdefault("usv_property_tuning", {})
        ts.setdefault("usv_category_tuning", {})
        ts.setdefault("usv_category_peth", {})

        # usv_peth: 1D PETH per emitter
        for emitter, payload in partial.get("usv_peth", {}).items():
            rate = payload.get("rate_smoothed", payload["rate"])
            null_mean = payload.get("null_mean_smoothed", payload["null_mean"])
            null_std = payload.get("null_std_smoothed", payload["null_std"])
            null_lo = payload.get("null_p0_5_smoothed", payload["null_p0_5"])
            null_hi = payload.get("null_p99_5_smoothed", payload["null_p99_5"])
            bin_centers_s = payload["bin_centers_s"]

            runs = _run_analysis(
                rate, null_lo, null_hi, null_mean, null_std, circular=False
            )
            for direction in ("excit", "suppress"):
                info = runs[direction]
                if info["max_run"] > 0:
                    info["run_t_start"] = float(bin_centers_s[info["run_start_idx"]])
                    info["run_t_end"] = float(bin_centers_s[info["run_end_idx"]])
                else:
                    info["run_t_start"] = float("nan")
                    info["run_t_end"] = float("nan")
                info["peak_t"] = (
                    float(bin_centers_s[info["peak_idx"]])
                    if info["peak_idx"] >= 0 else float("nan")
                )

            peak_abs_z, peak_idx, peak_signed = _peak_z_info(
                rate, null_mean, null_std
            )
            ts["usv_peth"][emitter] = {
                "peak_abs_z": peak_abs_z,
                "peak_signed_z": peak_signed,
                "peak_idx": int(peak_idx),
                "peak_t": (
                    float(bin_centers_s[peak_idx])
                    if peak_idx >= 0 else float("nan")
                ),
                "ramp_index": _ramp_index(rate, bin_centers_s),
                "excit": runs["excit"],
                "suppress": runs["suppress"],
            }

        # usv_property_tuning: 1D feature per (emitter, property)
        for emitter, props in partial.get("usv_property_tuning", {}).items():
            ts["usv_property_tuning"][emitter] = {}
            for prop, payload in props.items():
                rate = payload.get("rate_smoothed", payload["rate"])
                null_mean = payload.get("null_mean_smoothed", payload["null_mean"])
                null_std = payload.get("null_std_smoothed", payload["null_std"])
                null_lo = payload.get("null_p0_5_smoothed", payload["null_p0_5"])
                null_hi = payload.get("null_p99_5_smoothed", payload["null_p99_5"])
                bin_centers = payload["bin_centers"]

                runs = _run_analysis(
                    rate, null_lo, null_hi, null_mean, null_std, circular=False
                )
                for direction in ("excit", "suppress"):
                    info = runs[direction]
                    if info["max_run"] > 0:
                        info["range_low"] = float(bin_centers[info["run_start_idx"]])
                        info["range_high"] = float(bin_centers[info["run_end_idx"]])
                    else:
                        info["range_low"] = float("nan")
                        info["range_high"] = float("nan")
                    info["peak_bin_value"] = (
                        float(bin_centers[info["peak_idx"]])
                        if info["peak_idx"] >= 0 else float("nan")
                    )

                peak_abs_z, peak_idx, peak_signed = _peak_z_info(
                    rate, null_mean, null_std
                )
                ts["usv_property_tuning"][emitter][prop] = {
                    "peak_abs_z": peak_abs_z,
                    "peak_signed_z": peak_signed,
                    "peak_idx": int(peak_idx),
                    "peak_bin_value": (
                        float(bin_centers[peak_idx])
                        if peak_idx >= 0 else float("nan")
                    ),
                    "selectivity": _selectivity_index(rate),
                    "monotonicity": _monotonicity_spearman(rate),
                    "excit": runs["excit"],
                    "suppress": runs["suppress"],
                }

        # usv_category_tuning: 1D over unordered categories — no run analysis.
        for emitter, cat_feats in partial.get("usv_category_tuning", {}).items():
            ts["usv_category_tuning"][emitter] = {}
            for cat_feat, payload in cat_feats.items():
                rate = np.asarray(payload["rate"], dtype=float)
                null_mean = np.asarray(payload["null_mean"], dtype=float)
                null_std = np.asarray(payload["null_std"], dtype=float)
                null_lo = np.asarray(payload["null_p0_5"], dtype=float)
                null_hi = np.asarray(payload["null_p99_5"], dtype=float)
                cats = np.asarray(payload["categories"])

                peak_abs_z, peak_idx, peak_signed = _peak_z_info(
                    rate, null_mean, null_std
                )
                # significant-category count: # of categories where the
                # observed rate sits outside the (p0.5, p99.5) band.
                valid = (
                    np.isfinite(rate) & np.isfinite(null_lo) & np.isfinite(null_hi)
                )
                sig_mask = valid & ((rate > null_hi) | (rate < null_lo))

                ts["usv_category_tuning"][emitter][cat_feat] = {
                    "peak_abs_z": peak_abs_z,
                    "peak_signed_z": peak_signed,
                    "best_cat": (
                        int(cats[peak_idx]) if peak_idx >= 0 else -1
                    ),
                    "n_sig_categories": int(sig_mask.sum()),
                    "selectivity": _selectivity_index(rate),
                }

        # usv_category_peth: 2D (n_categories x n_time_bins) per (emitter, cat_feat)
        for emitter, cat_feats in partial.get("usv_category_peth", {}).items():
            ts["usv_category_peth"][emitter] = {}
            for cat_feat, payload in cat_feats.items():
                rate_2d = payload.get("rate_smoothed", payload["rate"])
                null_mean_2d = payload.get(
                    "null_mean_smoothed", payload["null_mean"]
                )
                null_std_2d = payload.get(
                    "null_std_smoothed", payload["null_std"]
                )
                null_lo_2d = payload.get(
                    "null_p0_5_smoothed", payload["null_p0_5"]
                )
                null_hi_2d = payload.get(
                    "null_p99_5_smoothed", payload["null_p99_5"]
                )
                bin_centers_s = payload["bin_centers_s"]
                cats = np.asarray(payload["categories"])

                # per-category: run analysis along time axis only
                per_category = {}
                best_abs_z = -1.0
                best_cat_id = -1
                best_t_idx = -1
                best_z = float("nan")
                best_excit = None
                best_suppress = None

                for ci in range(cats.size):
                    rate_c = rate_2d[ci, :]
                    nm_c = null_mean_2d[ci, :]
                    ns_c = null_std_2d[ci, :]
                    nlo_c = null_lo_2d[ci, :]
                    nhi_c = null_hi_2d[ci, :]
                    runs_c = _run_analysis(
                        rate_c, nlo_c, nhi_c, nm_c, ns_c, circular=False
                    )
                    for direction in ("excit", "suppress"):
                        info = runs_c[direction]
                        if info["max_run"] > 0:
                            info["run_t_start"] = float(
                                bin_centers_s[info["run_start_idx"]]
                            )
                            info["run_t_end"] = float(
                                bin_centers_s[info["run_end_idx"]]
                            )
                        else:
                            info["run_t_start"] = float("nan")
                            info["run_t_end"] = float("nan")
                        info["peak_t"] = (
                            float(bin_centers_s[info["peak_idx"]])
                            if info["peak_idx"] >= 0 else float("nan")
                        )
                    pz, pi, ps = _peak_z_info(rate_c, nm_c, ns_c)
                    per_category[int(cats[ci])] = {
                        "peak_abs_z": pz,
                        "peak_signed_z": ps,
                        "peak_t_idx": int(pi),
                        "peak_t": (
                            float(bin_centers_s[pi])
                            if pi >= 0 else float("nan")
                        ),
                        "excit": runs_c["excit"],
                        "suppress": runs_c["suppress"],
                    }
                    if np.isfinite(pz) and pz > best_abs_z:
                        best_abs_z = pz
                        best_cat_id = int(cats[ci])
                        best_t_idx = int(pi)
                        best_z = ps
                        best_excit = runs_c["excit"]
                        best_suppress = runs_c["suppress"]

                ts["usv_category_peth"][emitter][cat_feat] = {
                    "best_cat": best_cat_id,
                    "best_abs_z": (
                        float(best_abs_z) if best_abs_z >= 0 else float("nan")
                    ),
                    "best_signed_z": best_z,
                    "best_t_idx": best_t_idx,
                    "best_t": (
                        float(bin_centers_s[best_t_idx])
                        if best_t_idx >= 0 else float("nan")
                    ),
                    "best_excit": best_excit,
                    "best_suppress": best_suppress,
                    "per_category": per_category,
                }

    # VMI helpers (module-private but bound here for grouping with vocal compute)

    @staticmethod
    def _detect_bouts(
        starts: np.ndarray,
        stops: np.ndarray,
        quiet_seconds: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Description
        -----------
        Group sorted same-emitter USV intervals into bouts. A new bout
        begins whenever the gap from the previous USV's stop to this
        USV's start exceeds `quiet_seconds`, OR for the very first USV.
        A bout ends at the last USV before the next gap exceeds
        `quiet_seconds` (or at the final USV).

        Parameters
        ----------
        starts (np.ndarray, shape (n_usvs,))
            USV start times in seconds, sorted ascending.
        stops (np.ndarray, shape (n_usvs,))
            USV stop times in seconds, paired with `starts`.
        quiet_seconds (float)
            Inter-USV silence (s) required to begin a new bout.

        Returns
        -------
        bout_starts (np.ndarray, shape (n_bouts,))
            Start time of the first USV in each bout (seconds).
        bout_stops (np.ndarray, shape (n_bouts,))
            Stop time of the last USV in each bout (seconds).
        bout_idx (np.ndarray of int, shape (n_usvs,))
            Bout index assigned to each input USV (in [0, n_bouts)).
        """

        if starts.size == 0:
            return (
                np.empty(0, dtype=float),
                np.empty(0, dtype=float),
                np.empty(0, dtype=int),
            )
        is_bout_start = np.concatenate(
            [[True], (starts[1:] - stops[:-1]) > quiet_seconds]
        )
        is_bout_stop = np.concatenate(
            [(starts[1:] - stops[:-1]) > quiet_seconds, [True]]
        )
        bout_starts = starts[is_bout_start]
        bout_stops = stops[is_bout_stop]
        bout_idx = np.cumsum(is_bout_start.astype(int)) - 1
        return bout_starts, bout_stops, bout_idx

    @staticmethod
    def _compute_vmi_for_emitter(
        spike_times: np.ndarray,
        em_starts: np.ndarray,
        em_stops: np.ndarray,
        em_durations: np.ndarray,
        bout_quiet_s: float,
    ) -> dict:
        """
        Description
        -----------
        Compute the Vocalization Modulation Index (VMI) for one emitter
        side of a single cluster, following Mimica et al. After bout
        segmentation (`_detect_bouts` with `quiet_seconds=bout_quiet_s`),
        for each bout we compute:

          FR_baseline_per_bout =
              (#spikes in [bout_start − bout_quiet_s, bout_start])
              / bout_quiet_s
          FR_usv_per_bout =
              mean over USVs in the bout of
              ((#spikes in [usv_start, usv_stop]) / usv_duration)

          VMI = (mean(FR_usv) − mean(FR_baseline))
              / (mean(FR_usv) + mean(FR_baseline))

        Bouts whose 2-s baseline window starts before t = 0 (i.e.
        `bout_start < bout_quiet_s`) get NaN baseline (they cannot
        deliver an honest pre-bout estimate).

        A paired Wilcoxon signed-rank test on the per-bout
        (FR_baseline, FR_usv) pairs gates significance downstream.
        Returned p-values are NaN if there are no valid pairs (or all
        pairs are equal — Wilcoxon is undefined).

        Parameters
        ----------
        spike_times (np.ndarray)
            Sorted spike times of this cluster in seconds.
        em_starts, em_stops (np.ndarray, shape (n_emitter_usvs,))
            Emitter-filtered USV start / stop times in seconds, sorted
            ascending by start.
        em_durations (np.ndarray, shape (n_emitter_usvs,))
            `em_stops − em_starts`. Passed in rather than recomputed so
            we honor the CSV's stored duration when present.
        bout_quiet_s (float)
            Silence threshold for both bout segmentation AND baseline
            window length (one parameter for both, since the 2-s clean
            baseline pre-bout is guaranteed by construction once a new
            bout starts).

        Returns
        -------
        dict
            Keys: `vmi`, `fr_baseline`, `fr_usv`, `wilcoxon_statistic`,
            `wilcoxon_pvalue`, `n_bouts`, `fr_baseline_per_bout`,
            `fr_usv_per_bout`. Scalars are NaN when the per-bout arrays
            have no finite entries.
        """

        if em_starts.size == 0:
            return {
                "vmi": np.nan,
                "fr_baseline": np.nan,
                "fr_usv": np.nan,
                "wilcoxon_statistic": np.nan,
                "wilcoxon_pvalue": np.nan,
                "n_bouts": 0,
                "fr_baseline_per_bout": np.empty(0, dtype=float),
                "fr_usv_per_bout": np.empty(0, dtype=float),
            }

        bout_starts, _bout_stops, bout_idx = NeuronalTuning._detect_bouts(
            em_starts, em_stops, bout_quiet_s
        )
        n_bouts = int(bout_starts.size)

        fr_baseline_per_bout = np.full(n_bouts, np.nan, dtype=float)
        fr_usv_per_bout = np.full(n_bouts, np.nan, dtype=float)

        for b in range(n_bouts):
            bs = float(bout_starts[b])
            if bs >= bout_quiet_s:
                lo_idx = int(np.searchsorted(spike_times, bs - bout_quiet_s, side="left"))
                hi_idx = int(np.searchsorted(spike_times, bs, side="left"))
                fr_baseline_per_bout[b] = (hi_idx - lo_idx) / bout_quiet_s
            # else: leave NaN (pre-recording)

            usv_indices = np.where(bout_idx == b)[0]
            usv_rates = []
            for i_usv in usv_indices:
                us = float(em_starts[i_usv])
                ue = float(em_stops[i_usv])
                ud = float(em_durations[i_usv])
                if ud <= 0:
                    continue
                lo_idx = int(np.searchsorted(spike_times, us, side="left"))
                hi_idx = int(np.searchsorted(spike_times, ue, side="right"))
                usv_rates.append((hi_idx - lo_idx) / ud)
            if usv_rates:
                fr_usv_per_bout[b] = float(np.mean(usv_rates))

        if np.isnan(fr_baseline_per_bout).all() or np.isnan(fr_usv_per_bout).all():
            fr_baseline = np.nan
            fr_usv = np.nan
            vmi = np.nan
        else:
            fr_baseline = float(np.nanmean(fr_baseline_per_bout))
            fr_usv = float(np.nanmean(fr_usv_per_bout))
            denom = fr_usv + fr_baseline
            vmi = ((fr_usv - fr_baseline) / denom) if denom > 0 else np.nan

        valid = np.isfinite(fr_baseline_per_bout) & np.isfinite(fr_usv_per_bout)
        n_valid = int(valid.sum())
        if n_valid >= 1 and not np.array_equal(
            fr_baseline_per_bout[valid], fr_usv_per_bout[valid]
        ):
            try:
                method = "exact" if n_valid <= 25 else "approx"
                wtest = stats.wilcoxon(
                    fr_baseline_per_bout[valid],
                    fr_usv_per_bout[valid],
                    zero_method="zsplit",
                    correction=True,
                    alternative="two-sided",
                    method=method,
                )
                w_stat = float(wtest.statistic)
                w_p = float(wtest.pvalue)
            except (ValueError, RuntimeError):
                w_stat = np.nan
                w_p = np.nan
        else:
            w_stat = np.nan
            w_p = np.nan

        return {
            "vmi": vmi,
            "fr_baseline": fr_baseline,
            "fr_usv": fr_usv,
            "wilcoxon_statistic": w_stat,
            "wilcoxon_pvalue": w_p,
            "n_bouts": n_bouts,
            "fr_baseline_per_bout": fr_baseline_per_bout,
            "fr_usv_per_bout": fr_usv_per_bout,
        }

    # vocal per-cluster compute

    def _compute_one_cluster_vocal(
        self,
        cluster_file: pathlib.Path,
        voc_inputs: dict,
        side_precompute: dict,
    ) -> dict:
        """
        Description
        -----------
        Compute usv_peth, usv_property_tuning, usv_category_tuning and
        usv_category_peth for one cluster across all
        emitter sides queued by `_build_vocal_side_precompute`. Each
        observed+n_shuffles iteration shares the validity grids and per-
        anchor bookkeeping; only the spike train changes between
        iterations.

        Parameters
        ----------
        cluster_file (pathlib.Path)
            Path to the cluster's `*.npy` (row 0 = spike times in seconds).
        voc_inputs (dict)
            Output of `_load_vocal_inputs`.
        side_precompute (dict)
            Output of `_build_vocal_side_precompute`.

        Returns
        -------
        partial (dict)
            Top-level keys (`usv_peth`, `usv_property_tuning`, `usv_category_tuning`,
            `usv_category_peth`, `usv_metadata`) ready to merge into the
            cluster pkl.
        """

        params = self.tuning_parameters_dict
        n_shuffles = int(params["n_shuffles"])
        shuffle_min_s, shuffle_max_s = (
            float(params["shuffle_seconds_range"][0]),
            float(params["shuffle_seconds_range"][1]),
        )
        n_bins_default = int(params["total_bin_num"])
        prop_min_occ_s = float(params["usv_property_min_occupancy_seconds"])
        n_min_category = int(params["n_usv_min_category"])
        bout_quiet_s = float(params["bout_quiet_seconds"])
        smoothing_sd = float(params.get("smoothing_sd", 0.0))
        require_post = bool(params["vocal_require_clean_post_anchor"])
        require_prior = bool(params["vocal_require_clean_prior_anchor"])
        duration_seconds = voc_inputs["duration_seconds"]

        rel_bin_lo = side_precompute["_grid"]["rel_bin_lo"]
        rel_bin_hi = side_precompute["_grid"]["rel_bin_hi"]
        rel_bin_centers = side_precompute["_grid"]["rel_bin_centers"]

        cluster_data = np.load(cluster_file)
        spike_times_observed = np.sort(np.asarray(cluster_data[0, :], dtype=float))

        shuffle_offsets = _generate_shuffle_offsets(
            n_shuffles=n_shuffles,
            shuffle_min_seconds=shuffle_min_s,
            shuffle_max_seconds=shuffle_max_s,
            seed=params["shuffle_seed"],
        )

        per_side_acc: dict[str, dict] = {}
        for role, ps in side_precompute.items():
            if role == "_grid":
                continue
            n_peth_bins = rel_bin_lo.size

            peth_rates = np.full((n_shuffles + 1, n_peth_bins), np.nan, dtype=float)

            prop_rates: dict[str, np.ndarray] = {}
            for prop in CONTINUOUS_PROPERTIES:
                n_bins_prop = _bin_count_for_property(prop, n_bins_default)
                prop_rates[prop] = np.full(
                    (n_shuffles + 1, n_bins_prop), np.nan, dtype=float
                )

            q3w_rates: dict[str, np.ndarray] = {}
            for cat_feat in CATEGORICAL_FEATURES:
                n_cats = ps["anchor_categorical"][cat_feat]["unique_cats"].size
                q3w_rates[cat_feat] = np.full(
                    (n_shuffles + 1, n_cats), np.nan, dtype=float
                )

            q3p_rates: dict[str, np.ndarray] = {}
            for cat_feat in CATEGORICAL_FEATURES:
                n_cats = ps["anchor_categorical"][cat_feat]["unique_cats"].size
                q3p_rates[cat_feat] = np.full(
                    (n_shuffles + 1, n_cats, n_peth_bins), np.nan, dtype=float
                )

            per_side_acc[role] = {
                "peth_rates": peth_rates,
                "prop_rates": prop_rates,
                "q3w_rates": q3w_rates,
                "q3p_rates": q3p_rates,
            }

        # iterate observed (k=0) + n_shuffles iterations
        for k in range(n_shuffles + 1):
            if k == 0:
                spike_times = spike_times_observed
            else:
                spike_times = _circular_shift_spike_times(
                    spike_times_observed.copy(),
                    duration_seconds,
                    shuffle_offsets[k - 1],
                )

            for role, ps in side_precompute.items():
                if role == "_grid":
                    continue
                anchor_starts = ps["anchor_starts"]
                anchor_stops = ps["anchor_stops"]
                within_valid = ps["within_valid"]
                peth_valid = ps["peth_valid"]
                peth_denom_seconds = ps["peth_denom_seconds"]
                anchor_property_bin_idx = ps["anchor_property_bin_idx"]
                anchor_property_occ_seconds = ps["anchor_property_occ_seconds"]
                anchor_categorical = ps["anchor_categorical"]

                # spike counts per (anchor, bin)
                bin_lo_abs = anchor_starts[:, None] + rel_bin_lo[None, :]
                bin_hi_abs = anchor_starts[:, None] + rel_bin_hi[None, :]
                cnt_lo = np.searchsorted(
                    spike_times, bin_lo_abs.ravel(), side="left"
                ).reshape(bin_lo_abs.shape)
                cnt_hi = np.searchsorted(
                    spike_times, bin_hi_abs.ravel(), side="left"
                ).reshape(bin_hi_abs.shape)
                per_anchor_bin_count = (cnt_hi - cnt_lo).astype(np.int64)
                per_anchor_bin_count[~peth_valid] = 0

                # within-USV per-anchor spike count
                wcnt_lo = np.searchsorted(spike_times, anchor_starts, side="left")
                wcnt_hi = np.searchsorted(spike_times, anchor_stops, side="left")
                per_anchor_within_count = (wcnt_hi - wcnt_lo).astype(np.int64)
                per_anchor_within_count[~within_valid] = 0

                # usv_peth
                num = per_anchor_bin_count.sum(axis=0).astype(float)
                rate_peth = np.full_like(num, np.nan, dtype=float)
                ok = peth_denom_seconds > 0
                rate_peth[ok] = num[ok] / peth_denom_seconds[ok]
                per_side_acc[role]["peth_rates"][k, :] = rate_peth

                # usv_property_tuning
                for prop in CONTINUOUS_PROPERTIES:
                    n_bins_prop = _bin_count_for_property(prop, n_bins_default)
                    bin_idx = anchor_property_bin_idx[prop]
                    occ_seconds = anchor_property_occ_seconds[prop]
                    valid_anchor = within_valid & (bin_idx >= 0)
                    counts_per_bin = np.zeros(n_bins_prop, dtype=float)
                    contributions = np.where(
                        valid_anchor, per_anchor_within_count, 0
                    ).astype(float)
                    if valid_anchor.any():
                        np.add.at(
                            counts_per_bin,
                            np.clip(bin_idx, 0, n_bins_prop - 1),
                            contributions,
                        )
                    rate_prop = np.full(n_bins_prop, np.nan, dtype=float)
                    ok = occ_seconds >= prop_min_occ_s
                    rate_prop[ok] = counts_per_bin[ok] / occ_seconds[ok]
                    per_side_acc[role]["prop_rates"][prop][k, :] = rate_prop

                # usv_category_tuning & usv_category_peth
                for cat_feat in CATEGORICAL_FEATURES:
                    cat_info = anchor_categorical[cat_feat]
                    cat_idx = cat_info["anchor_cat_idx_dense"]
                    n_cats = cat_info["unique_cats"].size
                    occ_per_cat = cat_info["occ_seconds_per_cat"]
                    count_per_cat = cat_info["count_per_cat"]
                    peth_denom_per_cat_bin = cat_info["peth_denom_per_cat_bin"]

                    counts_per_cat = np.zeros(n_cats, dtype=float)
                    contrib = np.where(
                        within_valid & (cat_idx >= 0),
                        per_anchor_within_count,
                        0,
                    ).astype(float)
                    valid_cat_idx = np.where(cat_idx >= 0, cat_idx, 0)
                    np.add.at(counts_per_cat, valid_cat_idx, contrib)
                    rate_q3w = np.full(n_cats, np.nan, dtype=float)
                    enough = (count_per_cat >= n_min_category) & (occ_per_cat > 0)
                    rate_q3w[enough] = counts_per_cat[enough] / occ_per_cat[enough]
                    per_side_acc[role]["q3w_rates"][cat_feat][k, :] = rate_q3w

                    rate_q3p = np.full((n_cats, rel_bin_lo.size), np.nan, dtype=float)
                    for i_c in range(n_cats):
                        if count_per_cat[i_c] < n_min_category:
                            continue
                        member = cat_idx == i_c
                        if not member.any():
                            continue
                        num_per_bin = (
                            per_anchor_bin_count[member].sum(axis=0).astype(float)
                        )
                        denom = peth_denom_per_cat_bin[i_c]
                        ok = denom > 0
                        rate_row = np.full(rel_bin_lo.size, np.nan, dtype=float)
                        rate_row[ok] = num_per_bin[ok] / denom[ok]
                        rate_q3p[i_c] = rate_row
                    per_side_acc[role]["q3p_rates"][cat_feat][k, :, :] = rate_q3p

        # assemble payload
        partial: dict[str, Any] = {
            "usv_peth": OrderedDict(),
            "usv_property_tuning": OrderedDict(),
            "usv_category_tuning": OrderedDict(),
            "usv_category_peth": OrderedDict(),
            "triage_stats": {"vmi": OrderedDict()},
        }

        # VMI per emitter side. Bouts are detected on the emitter's own
        # USVs (gap > bout_quiet_s starts a new bout); the same threshold
        # doubles as the baseline window length, since the silence-before-
        # the-bout is guaranteed to be at least bout_quiet_s by
        # construction. VMI is independent of the n_shuffles loop above —
        # it uses the observed spike train only.
        all_emitters = voc_inputs["emitters"]
        all_starts = voc_inputs["starts"]
        all_stops = voc_inputs["stops"]
        usv_durations_full = (
            voc_inputs["usv_df"]["duration"].to_numpy()
            if "duration" in voc_inputs["usv_df"].columns
            else (all_stops - all_starts)
        )
        for role, ps in side_precompute.items():
            if role == "_grid":
                continue
            emitter_str = ps["side"]["emitter"]
            sex_label = ps["side"]["sex"]
            em_mask = all_emitters == emitter_str
            em_starts = all_starts[em_mask]
            em_stops = all_stops[em_mask]
            em_durations = usv_durations_full[em_mask]
            vmi_payload = NeuronalTuning._compute_vmi_for_emitter(
                spike_times=spike_times_observed,
                em_starts=em_starts,
                em_stops=em_stops,
                em_durations=em_durations,
                bout_quiet_s=bout_quiet_s,
            )
            vmi_payload["role"] = role
            vmi_payload["sex"] = sex_label
            vmi_payload["emitter"] = emitter_str
            partial["triage_stats"]["vmi"][emitter_str] = vmi_payload

        for role, ps in side_precompute.items():
            if role == "_grid":
                continue
            side = ps["side"]
            emitter_str = side["emitter"]
            acc = per_side_acc[role]

            # usv_peth: smooth observed and per-shuffle PETHs along the time-bin
            # axis, then percentile across smoothed shuffles.
            peth = acc["peth_rates"]
            peth_observed = peth[0, :]
            peth_null = peth[1:, :]
            peth_stats = _percentiles_block(peth_null)
            peth_payload = {
                "rate": peth_observed,
                "denom_seconds": ps["peth_denom_seconds"],
                "bin_centers_s": rel_bin_centers,
                "n_anchors": int(ps["anchor_idx"].size),
                "role": role,
                "sex": side["sex"],
                **peth_stats,
            }
            if smoothing_sd > 0:
                # plain D NaN policy (see _compute_one_cluster_behavioral):
                # observed and per-shuffle rate maps both fill sparse-bin
                # gaps via the weighted Gaussian, so smoothed rate and
                # smoothed null bands cover the same x-extent.
                peth_obs_sm = _gaussian_smooth_1d(
                    peth_observed, smoothing_sd, axis=-1, preserve_nan=False
                )
                peth_null_sm = _gaussian_smooth_1d(
                    peth_null, smoothing_sd, axis=-1, preserve_nan=False
                )
                peth_stats_sm = _percentiles_block(peth_null_sm)
                peth_payload["rate_smoothed"] = peth_obs_sm
                peth_payload.update({f"{k}_smoothed": v for k, v in peth_stats_sm.items()})
            partial["usv_peth"][emitter_str] = peth_payload

            partial["usv_property_tuning"][emitter_str] = OrderedDict()
            for prop in CONTINUOUS_PROPERTIES:
                arr = acc["prop_rates"][prop]
                obs = arr[0, :]
                null = arr[1:, :]
                stats = _percentiles_block(null)
                low, high = self.vocal_boundaries[prop]
                n_bins_prop = _bin_count_for_property(prop, n_bins_default)
                edges = np.linspace(low, high, n_bins_prop + 1)
                centers = 0.5 * (edges[:-1] + edges[1:])
                prop_payload = {
                    "rate": obs,
                    "occupancy_seconds": ps["anchor_property_occ_seconds"][prop],
                    "bin_centers": centers,
                    "bin_edges": edges,
                    "role": role,
                    "sex": side["sex"],
                    **stats,
                }
                if smoothing_sd > 0:
                    obs_sm = _gaussian_smooth_1d(
                        obs, smoothing_sd, axis=-1, preserve_nan=False
                    )
                    null_sm = _gaussian_smooth_1d(
                        null, smoothing_sd, axis=-1, preserve_nan=False
                    )
                    stats_sm = _percentiles_block(null_sm)
                    prop_payload["rate_smoothed"] = obs_sm
                    prop_payload.update({f"{k}_smoothed": v for k, v in stats_sm.items()})
                partial["usv_property_tuning"][emitter_str][prop] = prop_payload

            # usv_category_tuning: per-category rate at offset 0. Categorical x-axis,
            # so smoothing across categories is meaningless and skipped.
            partial["usv_category_tuning"][emitter_str] = OrderedDict()
            for cat_feat in CATEGORICAL_FEATURES:
                arr = acc["q3w_rates"][cat_feat]
                obs = arr[0, :]
                null = arr[1:, :]
                stats = _percentiles_block(null)
                cat_info = ps["anchor_categorical"][cat_feat]
                partial["usv_category_tuning"][emitter_str][cat_feat] = {
                    "categories": cat_info["unique_cats"],
                    "rate": obs,
                    "occupancy_seconds": cat_info["occ_seconds_per_cat"],
                    "occupancy_count": cat_info["count_per_cat"],
                    "role": role,
                    "sex": side["sex"],
                    **stats,
                }

            # usv_category_peth: per-category time-resolved PETH. Smooth along the
            # time-bin axis (axis=-1 of the 3D (n_shuffles+1, n_cats,
            # n_bins) array), then percentile across smoothed shuffles.
            partial["usv_category_peth"][emitter_str] = OrderedDict()
            for cat_feat in CATEGORICAL_FEATURES:
                arr = acc["q3p_rates"][cat_feat]
                obs = arr[0, :, :]
                null = arr[1:, :, :]
                stats = _percentiles_block(null)
                cat_info = ps["anchor_categorical"][cat_feat]
                cat_payload = {
                    "categories": cat_info["unique_cats"],
                    "rate": obs,
                    "denom_seconds": cat_info["peth_denom_per_cat_bin"],
                    "bin_centers_s": rel_bin_centers,
                    "role": role,
                    "sex": side["sex"],
                    **stats,
                }
                if smoothing_sd > 0:
                    obs_sm = _gaussian_smooth_1d(
                        obs, smoothing_sd, axis=-1, preserve_nan=False
                    )
                    null_sm = _gaussian_smooth_1d(
                        null, smoothing_sd, axis=-1, preserve_nan=False
                    )
                    stats_sm = _percentiles_block(null_sm)
                    cat_payload["rate_smoothed"] = obs_sm
                    cat_payload.update({f"{k}_smoothed": v for k, v in stats_sm.items()})
                partial["usv_category_peth"][emitter_str][cat_feat] = cat_payload

        partial["usv_metadata"] = {
            "cluster_id": cluster_file.stem,
            "session_root": str(self.root_directory),
            "n_shuffles": int(n_shuffles),
            "shuffle_seconds_range": [shuffle_min_s, shuffle_max_s],
            "peth_window_seconds": [float(rel_bin_lo[0]), float(rel_bin_hi[-1])],
            "peth_bin_seconds": float(rel_bin_hi[0] - rel_bin_lo[0]),
            "vocal_require_clean_post_anchor": require_post,
            "vocal_require_clean_prior_anchor": require_prior,
            "n_usv_min_category": n_min_category,
            "usv_property_min_occupancy_seconds": prop_min_occ_s,
            "bout_quiet_seconds": bout_quiet_s,
            "smoothing_sd": float(smoothing_sd),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "duration_seconds": float(duration_seconds),
        }
        self._attach_vocal_triage_stats(partial, params)
        return partial
