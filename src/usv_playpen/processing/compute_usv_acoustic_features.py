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

TEMPORARY / UPGRADE NEEDED
--------------------------
The original pipeline computed these features over the **SAM-masked region** of
each spectrogram so background did not dilute the statistics. This in-house
version is currently **mask-free** and restricts each feature to the USV's
signal **time-window** ``[0, duration)`` (dropping the zero-padded frames) as a
stop-gap. Once the downstream SAM box-detector masks are available, this module
MUST be upgraded to compute features over the true masked region; until then the
values are an approximation and should be treated as provisional.
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

# Feature columns written into the USV summary CSV, in output order. Mirrors the
# downstream consumers' `_CONTINUOUS_ACOUSTIC_FEATURES`.
FEATURE_COLUMNS = (
    "mean_freq_hz",
    "peak_freq_hz",
    "freq_bandwidth_hz",
    "mean_amplitude",
    "max_amplitude",
    "spectral_entropy",
)


def build_time_window_masks(durations: np.ndarray, n_time_bins: int) -> np.ndarray:
    """
    Description
    -----------
    Builds per-spectrogram binary region masks that are 1 over the USV's signal
    time-window ``[0, min(duration, n_time_bins))`` (all frequencies) and 0 over
    the zero-padded tail. This is the mask-free stand-in for the SAM mask (see
    the module's UPGRADE NEEDED note).

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


def compute_acoustic_features(
    specs: np.ndarray,
    durations: np.ndarray,
    freq_axis: np.ndarray,
    low_energy_frac: float,
    high_energy_frac: float,
) -> dict[str, np.ndarray]:
    """
    Description
    -----------
    Computes the per-USV acoustic features over each spectrogram's signal
    time-window. Shared intermediates (region mask, masked spectrogram, per-bin
    power) are computed once and reused across features. Ported from the
    external ``generate_spec_features`` numpy kernels, with the SAM mask replaced
    by the time-window region (see UPGRADE NEEDED).

    Parameters
    ----------
    specs (np.ndarray)
        Spectrograms, shape ``(N, F, T)``.
    durations (np.ndarray)
        Native time-bin counts, shape ``(N,)``.
    freq_axis (np.ndarray)
        Center frequency (Hz) of each spectrogram row, shape ``(F,)``.
    low_energy_frac (float)
        Lower edge of the bandwidth energy band (e.g. 0.05).
    high_energy_frac (float)
        Upper edge of the bandwidth energy band (e.g. 0.95).

    Returns
    -------
    features (dict[str, np.ndarray])
        Mapping of each name in :data:`FEATURE_COLUMNS` to a length-``N`` array.
    """

    _n, n_freq, n_time = specs.shape
    region_time = build_time_window_masks(durations, n_time)        # [N, T]
    region = region_time[:, None, :]                                # [N, 1, T] broadcasts over freq
    masked_specs = specs * region                                  # background zeroed

    n_region = np.broadcast_to(region, specs.shape).sum(axis=(1, 2))  # [N]
    sum_region = masked_specs.sum(axis=(1, 2))                        # [N]

    # Time-averaged power per frequency bin (over in-window columns only), then
    # normalized to a per-row probability distribution over frequency.
    region_counts = region_time.sum(axis=1)                          # [N] in-window columns
    freq_power = masked_specs.sum(axis=2) / (region_counts[:, None] + _EPS)  # [N, F]
    freq_power_norm = freq_power / (freq_power.sum(axis=1, keepdims=True) + _EPS)

    # mean frequency (amplitude-weighted)
    freq_grid = freq_axis[:, None]                                   # [F, 1]
    mean_freq = (masked_specs * freq_grid).sum(axis=(1, 2)) / (sum_region + _EPS)

    # peak frequency (row of the loudest in-window pixel)
    flat_argmax = masked_specs.reshape(_n, -1).argmax(axis=1)
    peak_freq = freq_axis[flat_argmax // n_time]

    # frequency bandwidth: span between cumulative-energy crossings
    cumsum = freq_power_norm.cumsum(axis=1)
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
        (joined on the per-USV index encoded in each ``spec_id``). USVs absent
        from the H5 (e.g. skipped during generation) get null features. Any
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
        with h5py.File(h5_loc, "r") as h5_file:
            session_group = h5_file[f"spectrogram/{root.name}"]
            specs = session_group["spectrograms"][:]
            durations = session_group["durations"][:]
            freq_axis = h5_file["frequency_bins"][:]
        # spectrogram rows are 1:1 with usv_summary.csv; keep only the real
        # (duration > 0) USVs and remember their row positions for the merge.
        usv_indices = np.flatnonzero(durations > 0).astype(np.uint32)
        specs = specs[usv_indices]
        durations = durations[usv_indices]

        features = compute_acoustic_features(
            specs=specs.astype(np.float64),
            durations=durations,
            freq_axis=freq_axis.astype(np.float64),
            low_energy_frac=low_energy_frac,
            high_energy_frac=high_energy_frac,
        )

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
