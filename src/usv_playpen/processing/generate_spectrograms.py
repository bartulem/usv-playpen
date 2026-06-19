"""
@author: bartulem
Generate per-USV spectrograms for the QLVM vocalization model.

For one session this reads the concatenated HPSS-filtered audio memmap and the
DAS ``*_usv_summary.csv``, computes a spectrogram for every USV segment, and
writes them to ``audio/spectrograms/<session>_spectrograms.h5``. Each USV's
spectrogram is the **variance-weighted average across all audio channels** (so
the channels carrying the most signal energy dominate), matching the
representation the QLVM was trained on.

This module is the in-house, JAX-native (torch-free) port of the external
``generate_spectrograms.py`` + ``spec_func.get_spec_librosa``; the spectrogram
math is plain ``librosa`` and the memmap is parsed exactly as
``das_inference.summarize_das_findings`` parses it (sampling rate, sample count,
channel count and dtype encoded in the ``*.mmap`` filename).

IMPORTANT: the spectrogram parameters in the ``generate_spectrograms`` settings
block MUST match the parameters the QLVM model was trained with -- otherwise new
spectrograms are out-of-distribution for the decoder and their latent
embeddings are meaningless. Validate against the reference ``specgen_out`` regen
before trusting downstream embeddings.
"""

from __future__ import annotations

import pathlib
from collections.abc import Callable
from datetime import datetime

import click
import h5py
import librosa
import numpy as np
import polars as pls
from click.core import ParameterSource

from ..cli_utils import modify_settings_json_for_cli
from ..os_utils import first_match_or_raise
from ..time_utils import is_gui_context, smart_wait

# Numerical floor reused from the original spectrogram code for normalization.
_NORMALIZE_EPS = 1e-6


def compute_usv_spectrogram(
    audio_segment_channels: np.ndarray,
    sampling_rate: int,
    spec_params: dict,
    normalize: bool = True,
) -> tuple[np.ndarray | None, int]:
    """
    Description
    -----------
    Computes the variance-weighted, multi-channel average spectrogram of a
    single (already-sliced) USV audio segment.

    For every channel the power STFT is computed (``librosa.stft`` magnitude
    squared), band-limited to ``[min_freq, max_freq]``, converted to dB
    (``power_to_db`` with ``ref=max``), resampled along frequency to
    ``num_freq_bins`` and fixed along time to ``num_time_bins``. The per-channel
    spectrograms are then averaged with weights equal to each channel's audio
    variance (louder/cleaner channels dominate); if every channel has zero
    variance the weights fall back to uniform. The averaged spectrogram is
    optionally min-max normalized to ``[0, 1]``.

    Parameters
    ----------
    audio_segment_channels (np.ndarray)
        The ``(n_samples, n_channels)`` slice of the audio memmap covering the
        segment (already sliced by the caller for memory efficiency).
    sampling_rate (int)
        Audio sampling rate in Hz.
    spec_params (dict)
        Spectrogram parameters: ``num_freq_bins``, ``num_time_bins``,
        ``nperseg``, ``noverlap``, ``min_freq``, ``max_freq``, ``hop_length``,
        ``window``.
    normalize (bool)
        Whether to min-max normalize the averaged spectrogram. Defaults to True.

    Returns
    -------
    avg_spectrogram (np.ndarray | None)
        A ``(num_freq_bins, num_time_bins)`` array, or None if no channel
        produced a valid spectrogram.
    original_time_bins (int)
        The native (pre-``fix_length``) STFT time-bin count for the segment;
        this is the USV's ``duration`` in spectrogram frames.
    """

    num_freq_bins = spec_params['num_freq_bins']
    num_time_bins = spec_params['num_time_bins']
    nperseg = spec_params['nperseg']
    min_freq = spec_params['min_freq']
    max_freq = spec_params['max_freq']
    window = spec_params['window']
    hop_length = spec_params['hop_length'] if spec_params['hop_length'] is not None else nperseg // 4

    n_channels = audio_segment_channels.shape[1]
    per_channel_specs: list[np.ndarray] = []
    per_channel_vars: list[float] = []
    original_time_bins = 0

    for ch_idx in range(n_channels):
        audio_segment = audio_segment_channels[:, ch_idx].astype(np.float64)
        if audio_segment.shape[0] < nperseg:
            continue
        audio_segment = audio_segment - np.mean(audio_segment)

        power_spec = (
            np.abs(
                librosa.stft(
                    audio_segment,
                    n_fft=nperseg,
                    hop_length=hop_length,
                    win_length=nperseg,
                    window=window,
                    center=True,
                )
            )
            ** 2
        )

        freqs = librosa.fft_frequencies(sr=sampling_rate, n_fft=nperseg)
        freq_mask = (freqs >= min_freq) & (freqs <= max_freq)
        power_spec = power_spec[freq_mask]
        spec_db = librosa.power_to_db(power_spec, ref=np.max)

        # Resample along the frequency axis to the target bin count.
        if spec_db.shape[0] != num_freq_bins:
            freq_interp = np.linspace(0, 1, num_freq_bins)
            freq_orig = np.linspace(0, 1, spec_db.shape[0])
            spec_db = np.stack(
                [np.interp(freq_interp, freq_orig, spec_slice) for spec_slice in spec_db.T]
            ).T

        original_time_bins = spec_db.shape[1]
        if spec_db.shape[1] != num_time_bins:
            spec_db = librosa.util.fix_length(spec_db, size=num_time_bins, axis=1)

        per_channel_specs.append(spec_db)
        per_channel_vars.append(float(np.var(audio_segment)))

    if not per_channel_specs:
        return None, 0

    weights = np.asarray(per_channel_vars, dtype=np.float64)
    if weights.sum() == 0:
        weights = np.ones_like(weights)
    avg_spectrogram = np.average(np.asarray(per_channel_specs), axis=0, weights=weights)

    if normalize:
        avg_spectrogram = avg_spectrogram - avg_spectrogram.min()
        avg_spectrogram = avg_spectrogram / (avg_spectrogram.max() + _NORMALIZE_EPS)

    return avg_spectrogram, original_time_bins


class SpectrogramGenerator:
    """
    Description
    -----------
    Generates per-USV spectrograms for one session and writes them to an HDF5
    file consumed by the QLVM training-set builder and the acoustic-feature /
    latent-inference steps.
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
        Initializes the SpectrogramGenerator.

        Parameters
        ----------
        root_directory (str)
            Session root directory (contains the ``audio`` tree).
        input_parameter_dict (dict)
            Processing settings; the ``generate_spectrograms`` block supplies
            the spectrogram parameters.
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

    def generate_session_spectrograms(self) -> None:
        """
        Description
        -----------
        Reads the session's HPSS-filtered audio memmap and ``*_usv_summary.csv``,
        computes the variance-weighted average spectrogram of every USV, and
        writes ``audio/spectrograms/<session>_spectrograms.h5`` containing the
        ``spectrograms`` (N, F, T), ``durations`` (N,), ``freq_bins`` (F,) and
        ``spectrogram_ids`` (N,) datasets, where ``spectrogram_ids`` are the
        per-USV ROW INDICES into ``usv_summary.csv``. Session provenance is the
        file-level ``session_id`` attribute (one session per file; also
        ``created`` and ``total_spectrograms``).

        Parameters
        ----------

        Returns
        -------
        .h5 spectrogram file
            One HDF5 file per session under ``audio/spectrograms/``.
        """

        self.message_output(
            f"Spectrogram generation started at: {datetime.now().hour:02d}:{datetime.now().minute:02d}:{datetime.now().second:02d}."
        )
        smart_wait(app_context_bool=self.app_context_bool, seconds=1)

        spec_params = self.input_parameter_dict['generate_spectrograms']
        normalize = spec_params['normalize']
        offset = spec_params['offset']

        root = pathlib.Path(self.root_directory)
        session_id = root.name

        usv_summary_loc = first_match_or_raise(
            root=root / "audio",
            pattern="*_usv_summary.csv",
            recursive=True,
            label="USV summary CSV",
        )
        usv_summary_df = pls.read_csv(source=str(usv_summary_loc))

        audio_file_loc = first_match_or_raise(
            root=root / "audio" / "hpss_filtered",
            pattern="*.mmap",
            label="concatenated audio mmap",
        )
        audio_file_name = audio_file_loc.name
        data_type, channel_num, sample_num, audio_sampling_rate = (
            audio_file_name.split("_")[-1][:-5],
            int(audio_file_name.split("_")[-2]),
            int(audio_file_name.split("_")[-3]),
            int(audio_file_name.split("_")[-4]),
        )
        audio_file_data = np.memmap(
            filename=audio_file_loc,
            mode="r",
            dtype=data_type,
            shape=(sample_num, channel_num),
        )

        all_specs: list[np.ndarray] = []
        all_durations: list[int] = []
        # spectrogram_ids are the per-USV ROW INDICES into this session's
        # usv_summary.csv (option A): the session itself is stored once as the
        # H5 ``session_id`` attribute, so the global id is composed downstream as
        # f"{session_id}_{index}" rather than duplicated into every row.
        all_usv_indices: list[int] = []

        starts = usv_summary_df["start"].to_numpy()
        stops = usv_summary_df["stop"].to_numpy()
        for usv_idx in range(usv_summary_df.height):
            t0 = float(starts[usv_idx]) - offset
            t1 = float(stops[usv_idx]) + offset
            s0 = max(0, round(t0 * audio_sampling_rate))
            s1 = min(sample_num, round(t1 * audio_sampling_rate))
            if s1 <= s0:
                continue
            segment = np.asarray(audio_file_data[s0:s1, :])
            spectrogram, original_time_bins = compute_usv_spectrogram(
                audio_segment_channels=segment,
                sampling_rate=audio_sampling_rate,
                spec_params=spec_params,
                normalize=normalize,
            )
            if spectrogram is None:
                continue
            all_specs.append(spectrogram.astype(np.float32))
            all_durations.append(int(original_time_bins))
            all_usv_indices.append(usv_idx)

        if not all_specs:
            self.message_output(
                f"No spectrograms generated for '{self.root_directory}' (no valid USV segments)."
            )
            return

        # Clean linspace axis (30000..120000); the ~0.1% offset from the true
        # band-limited STFT bin frequencies is sub-bin (rows span ~703 Hz) and
        # matches the feature-extraction axis, so round numbers are kept.
        freq_bins = np.linspace(spec_params['min_freq'], spec_params['max_freq'], spec_params['num_freq_bins'])

        spectrograms_dir = root / "audio" / "spectrograms"
        spectrograms_dir.mkdir(parents=True, exist_ok=True)
        h5_file_path = spectrograms_dir / f"{session_id}_spectrograms.h5"
        with h5py.File(h5_file_path, "w") as h5_file:
            # Session provenance lives once, as file-level attributes (one
            # session per file), so it is not duplicated into every row.
            h5_file.attrs["session_id"] = session_id
            h5_file.attrs["created"] = "generate_spectrograms"
            h5_file.attrs["total_spectrograms"] = len(all_specs)
            h5_file.create_dataset("spectrograms", data=np.asarray(all_specs), compression="gzip", compression_opts=6)
            h5_file.create_dataset("durations", data=np.asarray(all_durations, dtype=np.int64), compression="gzip", compression_opts=6)
            h5_file.create_dataset("freq_bins", data=freq_bins, compression="gzip", compression_opts=6)
            # Per-USV row indices into usv_summary.csv (NOT "{session}_{idx}" strings).
            h5_file.create_dataset("spectrogram_ids", data=np.asarray(all_usv_indices, dtype=np.int64), compression="gzip", compression_opts=6)

        self.message_output(
            f"Generated {len(all_specs)} spectrograms for session {session_id} -> {h5_file_path}."
        )
        self.message_output(
            f"Spectrogram generation ended at: {datetime.now().hour:02d}:{datetime.now().minute:02d}:{datetime.now().second:02d}."
        )


@click.command(name="generate-spectrograms")
@click.option('--root-directory', type=click.Path(exists=True, file_okay=False, dir_okay=True), required=True, help='Session root directory path.')
@click.option('--num-freq-bins', 'num_freq_bins', type=int, default=None, required=False, help='Number of spectrogram frequency bins.')
@click.option('--num-time-bins', 'num_time_bins', type=int, default=None, required=False, help='Number of spectrogram time bins.')
@click.option('--nperseg', 'nperseg', type=int, default=None, required=False, help='STFT window length (n_fft).')
@click.option('--min-freq', 'min_freq', type=float, default=None, required=False, help='Lower frequency cutoff (Hz).')
@click.option('--max-freq', 'max_freq', type=float, default=None, required=False, help='Upper frequency cutoff (Hz).')
@click.pass_context
def generate_spectrograms_cli(ctx, root_directory, **kwargs) -> None:
    """
    Description
    -----------
    A command-line tool to generate per-USV spectrograms for one session.

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

    SpectrogramGenerator(
        root_directory=root_directory,
        input_parameter_dict=processing_settings_dict,
        message_output=print,
    ).generate_session_spectrograms()
