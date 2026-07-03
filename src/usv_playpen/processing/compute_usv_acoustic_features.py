"""
@author: bartulem
Compute per-USV acoustic features from generated spectrograms and merge them
into the session's ``*_usv_summary.csv``.

Reads the per-session ``audio/spectrograms/<session>_spectrograms.h5`` produced
by :mod:`generate_spectrograms` (native-resolution specs; NOT the curated
training ``.npz``), computes interpretable spectral/amplitude descriptors for
each USV, and writes them back into the matching rows of the USV summary CSV
(``mean_freq_hz``, ``peak_freq_hz``, ``freq_bandwidth_hz``, ``mean_amplitude``,
``max_amplitude``, ``spectral_entropy``) — exactly the columns the downstream
visualizations/tuning code already consumes.

This is kept separate from spectrogram generation so feature definitions can be
revised and re-run cheaply, without regenerating the (slow) spectrograms.

Mask region
-----------
Like the original pipeline, features are computed over the **SAM-masked region**
of each spectrogram so background does not dilute the statistics. When the
per-session H5 carries a ``mask/<session>`` group (written by
:mod:`generate_masks`), each USV's 2D ``(freq, time)`` region is the union
(``np.any``) of its instance segmentations and features are restricted to those
true pixels. When no mask group is present -- or for an individual valid USV the
box detector produced no mask -- the region falls back to the USV's signal
**time-window** ``[0, duration)`` (all frequencies, zero-padded frames dropped),
which is the mask-free approximation.
"""

from __future__ import annotations

import pathlib
from collections.abc import Callable
from datetime import datetime

import click
import h5py
import numpy as np
import polars as pls
from click.core import ParameterSource

from ..cli_utils import modify_settings_json_for_cli
from ..os_utils import first_match_or_raise
from ..time_utils import is_gui_context, smart_wait

# Numerical floor that keeps divisions and logs finite on empty/degenerate regions.
_EPS = 1e-8

# Feature columns written into the USV summary CSV, in output order. The six
# continuous spectral/amplitude descriptors mirror the downstream consumers'
# `_CONTINUOUS_ACOUSTIC_FEATURES`; `mask_number` — the integer count of SAM masks
# per USV — is appended after them (it is integer-valued, so the continuous-stats
# list omits it, but the neuronal-tuning consumers bin it as a discrete property).
FEATURE_COLUMNS = (
    "mean_freq_hz",
    "peak_freq_hz",
    "freq_bandwidth_hz",
    "mean_amplitude",
    "max_amplitude",
    "spectral_entropy",
    "mask_number",
)


def build_time_window_masks(durations: np.ndarray, n_time_bins: int) -> np.ndarray:
    """
    Description
    -----------
    Builds per-spectrogram binary region masks that are 1 over the USV's signal
    time-window ``[0, min(duration, n_time_bins))`` (all frequencies) and 0 over
    the zero-padded tail. This is the fallback region used when no SAM mask group
    is present, or for an individual valid USV the detector produced no mask (see
    the module header's "Mask region" note).

    Parameters
    ----------
    durations (np.ndarray)
        Native (pre-pad) spectrogram time-bin counts, shape ``(N,)``.
    n_time_bins (int)
        Padded spectrogram time-bin count (``T``).

    Returns
    -------
    region_masks (np.ndarray)
        A ``(N, T)`` float32 array; broadcast over the frequency axis by callers.
    """

    n = durations.shape[0]
    region = np.zeros((n, n_time_bins), dtype=np.float32)
    for i in range(n):
        signal_len = int(min(max(int(durations[i]), 1), n_time_bins))
        region[i, :signal_len] = 1.0
    return region


def build_mask_region_masks(
    durations: np.ndarray,
    usv_indices: np.ndarray,
    segmentations: np.ndarray,
    spectrogram_index: np.ndarray,
    n_freq: int,
    n_time: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Description
    -----------
    Builds per-USV 2D ``(freq, time)`` region masks from a ``mask/<session>``
    group, aligned 1:1 with the valid (``duration > 0``) USVs the caller has
    selected. For each valid USV at original summary row ``usv_indices[i]``, the
    region is the boolean union (``np.any``) of every ``segmentations`` row whose
    ``spectrogram_index`` equals that summary row. When a valid USV has NO mask
    rows (the box detector produced no mask for that call), the region falls back
    to that USV's signal time-window ``[0, min(duration, n_time))`` so its
    features stay meaningful rather than degenerating on an all-false region.

    Parameters
    ----------
    durations (np.ndarray)
        Native time-bin counts of the valid USVs, shape ``(N_valid,)`` (already
        filtered and aligned to ``usv_indices``).
    usv_indices (np.ndarray)
        Original usv_summary.csv row index of each valid USV, shape
        ``(N_valid,)``; matched against ``spectrogram_index``.
    segmentations (np.ndarray)
        All instance masks for the session, shape ``(M, n_freq, n_time)`` boolean.
    spectrogram_index (np.ndarray)
        The owning usv_summary row of each mask, shape ``(M,)``.
    n_freq (int)
        Frequency-bin count ``F`` of the region (mask height).
    n_time (int)
        Time-bin count ``T`` of the region (mask width).

    Returns
    -------
    region_masks (np.ndarray)
        A ``(N_valid, F, T)`` float32 array (1.0 inside the region, 0.0 outside).
    mask_counts (np.ndarray)
        Per-USV count of instance masks owning each valid USV (0 when the box
        detector produced none), shape ``(N_valid,)`` int64 — the ``mask_number``
        feature.
    fallback_count (int)
        Number of valid USVs that had no mask and fell back to the time-window.
    """

    n_valid = usv_indices.shape[0]
    region_masks = np.zeros((n_valid, n_freq, n_time), dtype=np.float32)
    mask_counts = np.zeros(n_valid, dtype=np.int64)
    fallback_count = 0
    for i in range(n_valid):
        summary_row = int(usv_indices[i])
        mask_rows = np.flatnonzero(spectrogram_index == summary_row)
        mask_counts[i] = mask_rows.size
        union_mask = (
            np.any(segmentations[mask_rows], axis=0) if mask_rows.size > 0
            else None
        )
        # Use the SAM mask only when it actually selects at least one pixel. A
        # mask row that exists but is entirely empty (all-False union) would
        # otherwise leave region_masks[i] all-zero, which makes every masked
        # feature downstream degenerate (NaN / undefined). Treat that case
        # exactly like "no mask row" and fall back to the time-window region.
        if union_mask is not None and union_mask.any():
            region_masks[i] = union_mask.astype(np.float32)
        else:
            signal_len = int(min(max(int(durations[i]), 1), n_time))
            region_masks[i, :, :signal_len] = 1.0
            fallback_count += 1
    return region_masks, mask_counts, fallback_count


def compute_acoustic_features(
    specs: np.ndarray,
    durations: np.ndarray,
    freq_axis: np.ndarray,
    low_energy_frac: float,
    high_energy_frac: float,
    region_masks: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """
    Description
    -----------
    Computes the per-USV acoustic features over each spectrogram's region of
    interest. Shared intermediates (region mask, masked spectrogram, per-bin
    power) are computed once and reused across features. Ported from the external
    ``generate_spec_features`` numpy kernels.

    The region is either the true SAM-masked 2D ``(freq, time)`` region (when
    ``region_masks`` is supplied) or, by default, the USV's signal time-window
    ``[0, min(duration, T))`` over all frequencies. All reductions are written
    against the materialized/broadcast region so both cases share one code path:
    the per-frequency-bin power is each row's masked power summed over time and
    divided by that row's UNMASKED-column count -- constant across frequencies for
    the time-window region, variable for a true 2D mask.

    Parameters
    ----------
    specs (np.ndarray)
        Spectrograms, shape ``(N, F, T)``. Must be NON-NEGATIVE (a normalized
        ``[0, 1]`` or linear-power spectrogram): the reductions mask the
        background to ``0`` and treat in-region values as amplitudes / probability
        weights, so a dB-scaled spectrogram (negative values, i.e.
        ``normalize=False`` in :func:`generate_spectrograms`) would make ``0`` the
        per-pixel maximum and yield ``0`` / ``NaN`` / garbage features.
    durations (np.ndarray)
        Native time-bin counts, shape ``(N,)``.
    freq_axis (np.ndarray)
        Center frequency (Hz) of each spectrogram row, shape ``(F,)``.
    low_energy_frac (float)
        Lower edge of the bandwidth energy band (e.g. 0.05).
    high_energy_frac (float)
        Upper edge of the bandwidth energy band (e.g. 0.95).
    region_masks (np.ndarray | None)
        Optional per-USV 2D region masks, shape ``(N, F, T)`` (nonzero inside the
        region). When None, the time-window region from :func:`build_time_window_masks`
        is used (broadcast over frequency).

    Returns
    -------
    features (dict[str, np.ndarray])
        Mapping of each spectral / amplitude descriptor name to a length-``N``
        array (the :data:`FEATURE_COLUMNS` entries except ``mask_number``, which
        the caller injects from the per-USV mask counts).

    Raises
    ------
    ValueError
        If ``specs`` contains negative values (a dB-scaled / un-normalized
        spectrogram), for which the masked reductions are undefined.
    """

    _n, n_freq, n_time = specs.shape

    # The masked reductions below zero the background and treat in-region values
    # as amplitudes / probability weights, which requires non-negative input. A
    # dB-scaled spectrogram (normalize=False) has negative values, so 0 would
    # become the per-pixel maximum (max_amplitude -> 0, peak_freq on a background
    # pixel) and the power normalization / entropy would emit garbage / NaN. Fail
    # loudly instead of silently corrupting the features.
    if specs.size and specs.min() < 0:
        raise ValueError(
            "compute_acoustic_features requires non-negative spectrograms (a "
            "normalized [0, 1] or linear-power spectrogram); got specs.min() = "
            f"{specs.min():.4g}. Run generate_spectrograms with normalize=True."
        )
    if region_masks is None:
        region_time = build_time_window_masks(durations, n_time)    # [N, T]
        region = region_time[:, None, :]                            # [N, 1, T] broadcasts over freq
    else:
        region = region_masks.astype(specs.dtype, copy=False)       # [N, F, T] true 2D mask
    region_full = np.broadcast_to(region, specs.shape)              # [N, F, T] view (no copy when broadcast)
    masked_specs = specs * region                                  # background zeroed

    # Time-averaged power per frequency bin (over unmasked columns of that row
    # only), then normalized to a per-row probability distribution over frequency.
    region_counts = region_full.sum(axis=2)                          # [N, F] unmasked cols per freq row
    # n_region (total unmasked pixels per spectrogram) is the row-sum of the
    # per-frequency unmasked-column counts, so derive it from region_counts
    # instead of a second full (1, 2)-axis reduction over region_full.
    n_region = region_counts.sum(axis=1)                             # [N] unmasked-pixel count
    sum_region = masked_specs.sum(axis=(1, 2))                       # [N]

    freq_power = masked_specs.sum(axis=2) / (region_counts + _EPS)   # [N, F]
    freq_power_norm = freq_power / (freq_power.sum(axis=1, keepdims=True) + _EPS)

    # mean frequency (amplitude-weighted)
    freq_grid = freq_axis[:, None]                                   # [F, 1]
    mean_freq = (masked_specs * freq_grid).sum(axis=(1, 2)) / (sum_region + _EPS)

    # peak frequency (row of the loudest in-window pixel)
    flat_argmax = masked_specs.reshape(_n, -1).argmax(axis=1)
    # `// n_time` maps the flat (F*T) argmax index back to its frequency-row index.
    peak_freq = freq_axis[flat_argmax // n_time]

    # frequency bandwidth: span between cumulative-energy crossings
    cumsum = freq_power_norm.cumsum(axis=1)
    # The count of bins below the threshold == index of the first bin whose
    # cumulative energy crosses it (clamped to the last bin).
    low_bin = np.minimum((cumsum < low_energy_frac).sum(axis=1), n_freq - 1)
    high_bin = np.minimum((cumsum < high_energy_frac).sum(axis=1), n_freq - 1)
    bandwidth = freq_axis[high_bin] - freq_axis[low_bin]

    # spectral entropy (nats) of the normalized frequency power profile
    prob = freq_power_norm + _EPS
    entropy = -(prob * np.log(prob)).sum(axis=1)

    return {
        "mean_freq_hz": mean_freq,
        "peak_freq_hz": peak_freq,
        "freq_bandwidth_hz": bandwidth,
        "mean_amplitude": sum_region / (n_region + _EPS),
        "max_amplitude": masked_specs.max(axis=(1, 2)),
        "spectral_entropy": entropy,
    }


class USVAcousticFeatureExtractor:
    """
    Description
    -----------
    Computes per-USV acoustic features from a session's spectrogram H5 and
    merges them into the session's ``*_usv_summary.csv`` (one row per USV).
    """

    def __init__(
        self,
        root_directory: str | None = None,
        input_parameter_dict: dict | None = None,
        message_output: Callable | None = None,
    ) -> None:
        """
        Description
        -----------
        Initializes the USVAcousticFeatureExtractor.

        Parameters
        ----------
        root_directory (str)
            Session root directory (contains the ``audio`` tree).
        input_parameter_dict (dict)
            Processing settings; the ``compute_usv_acoustic_features`` block
            supplies the bandwidth energy-band edges.
        message_output (Callable)
            Logging callback; defaults to ``print``.

        Returns
        -------
        None
        """

        self.root_directory = root_directory
        self.input_parameter_dict = input_parameter_dict if input_parameter_dict is not None else {}
        self.message_output = message_output if message_output is not None else print
        self.app_context_bool = is_gui_context()

    def merge_features_into_summary(self) -> None:
        """
        Description
        -----------
        Reads the per-session spectrogram H5 and the USV summary CSV, computes
        the acoustic features, and writes them into the matching summary rows
        (joined on ``_usv_row``, the positional index of each USV row in the
        summary CSV, which is 1:1 with the spectrogram rows). USVs absent from
        the H5 (e.g. skipped during generation) get null features. Any
        pre-existing feature columns are replaced.

        Parameters
        ----------

        Returns
        -------
        Updated ``*_usv_summary.csv`` with the acoustic-feature columns.
        """

        self.message_output(
            f"USV acoustic-feature extraction started at: {datetime.now().hour:02d}:{datetime.now().minute:02d}:{datetime.now().second:02d}."
        )
        smart_wait(app_context_bool=self.app_context_bool, seconds=1)

        cfg = self.input_parameter_dict['compute_usv_acoustic_features']
        low_energy_frac = cfg['low_energy_frac']
        high_energy_frac = cfg['high_energy_frac']

        root = pathlib.Path(self.root_directory)
        h5_loc = first_match_or_raise(
            root=root / "audio" / "spectrograms",
            pattern="*_spectrograms.h5",
            label="per-session spectrogram H5",
        )
        mask_group_key = f"mask/{root.name}"
        with h5py.File(h5_loc, "r") as h5_file:
            session_group = h5_file[f"spectrogram/{root.name}"]
            specs = session_group["spectrograms"][:]
            durations = session_group["durations"][:]
            freq_axis = h5_file["frequency_bins"][:]
            has_masks = mask_group_key in h5_file
            if has_masks:
                segmentations = h5_file[mask_group_key]["segmentations"][:]
                mask_spec_index = h5_file[mask_group_key]["spectrogram_index"][:]
        # spectrogram rows are 1:1 with usv_summary.csv; keep only the real
        # (duration > 0) USVs and remember their row positions for the merge.
        usv_indices = np.flatnonzero(durations > 0).astype(np.uint32)
        specs = specs[usv_indices]
        durations = durations[usv_indices]

        # Restrict features to the true SAM-masked region when masks exist, else to
        # the signal time-window (and per-USV the same fallback when a valid call
        # has no detected mask).
        region_masks = None
        if has_masks:
            region_masks, mask_counts, fallback_count = build_mask_region_masks(
                durations=durations,
                usv_indices=usv_indices,
                segmentations=segmentations,
                spectrogram_index=mask_spec_index,
                n_freq=specs.shape[1],
                n_time=specs.shape[2],
            )
            region_masks = region_masks.astype(np.float64)
            self.message_output(
                f"Using SAM mask regions for {len(usv_indices) - fallback_count}/{len(usv_indices)} USVs "
                f"({fallback_count} with no detected mask fell back to the signal time-window)."
            )
        else:
            mask_counts = np.zeros(len(usv_indices), dtype=np.int64)
            self.message_output(
                "No mask group in the spectrogram H5; computing features over the signal time-window."
            )

        features = compute_acoustic_features(
            specs=specs.astype(np.float64),
            durations=durations,
            freq_axis=freq_axis.astype(np.float64),
            low_energy_frac=low_energy_frac,
            high_energy_frac=high_energy_frac,
            region_masks=region_masks,
        )
        # `mask_number` is the per-USV SAM mask count -- not a spectral descriptor,
        # so it is injected here rather than computed inside
        # `compute_acoustic_features`. It is 0 for USVs with no detected mask (and
        # for every USV when the session has no mask group at all).
        features["mask_number"] = mask_counts

        features_df = pls.DataFrame(
            {"_usv_row": usv_indices, **{name: features[name].astype(np.float64) for name in FEATURE_COLUMNS}}
        )

        usv_summary_loc = first_match_or_raise(
            root=root / "audio",
            pattern="*_usv_summary.csv",
            recursive=True,
            label="USV summary CSV",
        )
        usv_df = pls.read_csv(source=str(usv_summary_loc))
        usv_df = usv_df.drop([c for c in FEATURE_COLUMNS if c in usv_df.columns])
        usv_df = usv_df.with_row_index(name="_usv_row")
        merged = usv_df.join(features_df, on="_usv_row", how="left").drop("_usv_row")
        merged.write_csv(file=str(usv_summary_loc))

        self.message_output(
            f"Merged acoustic features for {len(usv_indices)} USVs into {usv_summary_loc.name}."
        )
        self.message_output(
            f"USV acoustic-feature extraction ended at: {datetime.now().hour:02d}:{datetime.now().minute:02d}:{datetime.now().second:02d}."
        )


@click.command(name="generate-usv-acoustic-features")
@click.option('--root-directory', type=click.Path(exists=True, file_okay=False, dir_okay=True), required=True, help='Session root directory path.')
@click.option('--low-energy-frac', 'low_energy_frac', type=float, default=None, required=False, help='Lower edge of the bandwidth energy band.')
@click.option('--high-energy-frac', 'high_energy_frac', type=float, default=None, required=False, help='Upper edge of the bandwidth energy band.')
@click.pass_context
def compute_usv_acoustic_features_cli(ctx, root_directory, **kwargs) -> None:
    """
    Description
    -----------
    A command-line tool to compute per-USV acoustic features and merge them into
    the session's USV summary CSV.

    Parameters
    ----------

    Returns
    -------
    None
    """

    provided_params = [key for key in kwargs if ctx.get_parameter_source(key) == ParameterSource.COMMANDLINE]

    processing_settings_dict = modify_settings_json_for_cli(
        ctx=ctx,
        provided_params=provided_params,
        settings_dict='processing_settings',
    )

    USVAcousticFeatureExtractor(
        root_directory=root_directory,
        input_parameter_dict=processing_settings_dict,
        message_output=print,
    ).merge_features_into_summary()
