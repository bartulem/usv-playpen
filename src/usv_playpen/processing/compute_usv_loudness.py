"""
@author: bartulem
Absolute per-USV loudness: the masked, image-level level in dB that a QLVM
loudness conditional (model package v3, phase 11) is trained and decoded on.

No stored array carries how loud a call was: :func:`compute_usv_spectrogram`
converts every microphone to dB relative to that microphone's own peak for that
call (``power_to_db(ref=np.max)``) and min-max normalizes the average, so every
stored spectrogram spans the same [0, 1], and the ``usv_summary.csv`` amplitude
columns are computed on those images. This module re-reads the call from the
HPSS-filtered audio with the generator's own slice, STFT, band and frequency
resampling, but keeps the power ABSOLUTE (int16 counts squared). Per eligible
microphone ``c`` the call's level is the mean over its SAM mask region ``M`` of
the pixel dB,

    ``L_c = mean_{p in M} 10 * log10(P_c(p) + 1e-12)``,

and the microphones are combined with the generator's own weights, each
segment's variance ``w_c``:

    ``image_level_db = sum_c w_c * L_c / sum_c w_c``,

i.e. the level of the recording the stored spectrogram is made of. This is a
port of MMMmB ``under_development/qlvm_cond_candidates/loudness/
extract_call_loudness.py`` (``image_level_db``; the definition is in MMMmB
``docs/qlvm_playpen_runs/dataset/CALL_LOUDNESS_EXTRACTION.md`` section 3), the
extractor that produced the package's corpus values, so it keeps that script's
order of operations: the power (not the dB) is resampled along frequency, and
calls longer than ``num_time_bins`` frames keep their first ones, as the stored
spectrogram does.
"""

from __future__ import annotations

import pathlib
from collections.abc import Callable

import librosa
import numpy as np

from .generate_spectrograms import excluded_audio_columns, open_hpss_audio

# Added to the power before log10 so an all-zero pixel stays finite.
_POWER_FLOOR = 1e-12


def call_image_level_db(
    audio_segment_channels: np.ndarray,
    sampling_rate: int,
    spec_params: dict,
    region: np.ndarray,
) -> float:
    """
    Description
    -----------
    The image-level loudness of one call: per channel, the mean over the call's
    mask region of ``10 * log10`` of the absolute band power (resampled to
    ``num_freq_bins`` rows, first ``num_time_bins`` frames), combined across
    channels weighted by each de-meaned segment's variance (see the module
    docstring).

    Parameters
    ----------
    audio_segment_channels (np.ndarray)
        The ``(n_samples, n_channels)`` audio slice of the call, eligible channels only.
    sampling_rate (int)
        Audio sampling rate in Hz.
    spec_params (dict)
        The ``generate_spectrograms`` settings the session's spectrograms were
        made with: ``num_freq_bins``, ``num_time_bins``, ``nperseg``,
        ``min_freq``, ``max_freq``, ``hop_length``, ``window``.
    region (np.ndarray)
        ``(num_freq_bins, num_time_bins)`` boolean mask region of the call on the
        stored spectrogram's grid.

    Returns
    -------
    image_level_db (float)
        Loudness in dB re 1 int16 count squared; NaN when the segment is shorter
        than ``nperseg``, the region is empty over the call's frames, or every
        channel has zero variance.
    """

    nperseg = spec_params['nperseg']
    hop_length = spec_params['hop_length'] if spec_params['hop_length'] is not None else nperseg // 4
    if audio_segment_channels.shape[0] < nperseg:
        return float("nan")
    freqs = librosa.fft_frequencies(sr=sampling_rate, n_fft=nperseg)
    band = (freqs >= spec_params['min_freq']) & (freqs <= spec_params['max_freq'])
    rows_out = np.linspace(0, 1, spec_params['num_freq_bins'])
    rows_in = np.linspace(0, 1, int(band.sum()))

    levels, weights = [], []
    for ch_idx in range(audio_segment_channels.shape[1]):
        segment = audio_segment_channels[:, ch_idx].astype(np.float64)
        segment = segment - np.mean(segment)
        power = np.abs(
            librosa.stft(
                segment, n_fft=nperseg, hop_length=hop_length, win_length=nperseg,
                window=spec_params['window'], center=True,
            )
        ) ** 2
        power = np.stack([np.interp(rows_out, rows_in, column) for column in power[band].T]).T
        power = power[:, :spec_params['num_time_bins']]
        call_region = np.asarray(region, dtype=bool)[:, :power.shape[1]]
        if not call_region.any():
            return float("nan")
        levels.append(np.mean(10.0 * np.log10(power[call_region] + _POWER_FLOOR)))
        weights.append(float(np.var(segment)))

    # The extractor stored each channel's level and weight as float32 before
    # combining them in float64; rounding the same way reproduces its values exactly.
    weights = np.asarray(weights, dtype=np.float32).astype(np.float64)
    levels = np.asarray(levels, dtype=np.float32).astype(np.float64)
    if not levels.size or weights.sum() <= 0:
        return float("nan")
    return float((weights * levels).sum() / weights.sum())


def session_image_level_db(
    root_directory: str,
    starts: np.ndarray,
    stops: np.ndarray,
    regions: np.ndarray,
    spec_params: dict,
    message_output: Callable | None = None,
) -> np.ndarray:
    """
    Description
    -----------
    The image-level loudness (:func:`call_image_level_db`) of a session's calls,
    sliced from its HPSS-filtered audio memmap exactly as
    :class:`generate_spectrograms.SpectrogramGenerator` slices them (floor of
    ``start - offset``, ceil of ``stop + offset``) over the channels the session
    metadata does not exclude.

    Parameters
    ----------
    root_directory (str)
        Session root directory (contains ``audio/hpss_filtered``).
    starts (np.ndarray)
        ``(N,)`` call starts in seconds (``usv_summary.csv`` ``start``).
    stops (np.ndarray)
        ``(N,)`` call stops in seconds (``usv_summary.csv`` ``stop``).
    regions (np.ndarray)
        ``(N, num_freq_bins, num_time_bins)`` boolean SAM mask regions, on the
        stored spectrograms' grid.
    spec_params (dict)
        The ``generate_spectrograms`` settings block (also supplies ``offset``).
    message_output (Callable | None)
        Logging callback; defaults to ``print``.

    Returns
    -------
    image_level_db (np.ndarray)
        ``(N,)`` float32 loudness in dB (NaN where :func:`call_image_level_db` is).
    """

    message_output = message_output if message_output is not None else print
    audio, sampling_rate = open_hpss_audio(pathlib.Path(root_directory))
    sample_num, channel_num = audio.shape
    _excluded, excluded_columns = excluded_audio_columns(root_directory, logger=message_output)
    eligible_columns = [ch for ch in range(channel_num) if ch not in excluded_columns]
    offset = spec_params['offset']

    loudness = np.full(len(starts), np.nan, dtype=np.float32)
    for call_idx, (start, stop) in enumerate(zip(starts, stops, strict=True)):
        s0 = max(0, int(np.floor((float(start) - offset) * sampling_rate)))
        s1 = min(sample_num, int(np.ceil((float(stop) + offset) * sampling_rate)))
        if s1 <= s0:
            continue
        segment = np.asarray(audio[s0:s1, :])[:, eligible_columns]
        loudness[call_idx] = call_image_level_db(segment, sampling_rate, spec_params, regions[call_idx])
    return loudness
