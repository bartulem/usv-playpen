"""
@author: bartulem
Different functions for modifying files:
(1a) break from multi to single channel
(1b) perform harmonic-percussive source separation
(1c) perform band-pass filtering
(1d) concatenate single channel audio (e.g., wav) files
(1e) broadband filter: remove line-noise tones, high-pass at 2 kHz and
     write the channels straight into one broadband memmap
(2a) concatenate video (e.g., mp4) files
(2b) change video (e.g., mp4) sampling rate (fps)
(3a) concatenate e-phys binary files
(3b) split manually curated clusters into sessions
"""

from __future__ import annotations

import concurrent.futures
import configparser
import csv
import json
import multiprocessing
import os
import pathlib
import re
import shutil
import subprocess
import time
from collections.abc import Callable
from datetime import datetime
from importlib import metadata

import librosa
import numpy as np
import polars as pls
import soundfile as sf
from imgstore import new_for_filename
from scipy import signal
from scipy.ndimage import median_filter
from scipy.io import wavfile
from spikeinterface.curation.curation_tools import find_duplicated_spikes
from tqdm import tqdm

from ..os_utils import (
    AUDIO_MMAP_BAND_FOLDERS,
    atomic_output_path,
    audio_mmap_name_regex,
    configure_path,
    ephys_base_for_data_root,
    first_match_or_raise,
    wait_for_subprocesses,
)
from ..time_utils import is_gui_context, smart_wait
from ..yaml_utils import load_session_metadata, save_session_metadata
from .load_audio_files import DataLoader

# Name of the per-session report the broadband filter writes next to its memmap.
BROADBAND_REPORT_NAME = "line_noise.json"

# Settings of the broadband filter that change WHAT is written (and so decide
# whether an existing output is still valid); the remaining keys (chunk length,
# thread count) only change how fast it is written. The removed band
# ``filter_freq_bounds`` enters as its upper edge, ``highpass_cutoff_hz`` (see
# broadband_output_settings).
BROADBAND_OUTPUT_SETTINGS = (
    "source_dir",
    "source_glob",
    "transition_width_hz",
    "stopband_attenuation_db",
    "line_noise_search_bands_hz",
    "line_noise_min_height_db",
    "line_noise_estimation_windows",
    "line_noise_estimation_window_s",
    "line_noise_max_tones_per_band",
    "line_noise_min_separation_hz",
    "line_noise_floor_window_hz",
    "line_noise_block_s",
    "line_noise_smoothing_blocks",
)

# Frequencies (Hz) at which the high-pass response is evaluated and recorded in
# the report (the design targets: stopband <= 1.5 kHz, -6 dB at 2 kHz, flat from
# 2.5 kHz, plus the two line-noise frequencies and the USV band).
BROADBAND_RESPONSE_PROBES_HZ = (1000, 1500, 1652.5, 2000, 2122.5, 2500, 3000, 8000, 30000)

# Columns of the per-session report CSV the batch runner appends to.
BROADBAND_BATCH_REPORT_COLUMNS = (
    "session_root",
    "status",
    "runtime_s",
    "n_channels",
    "n_samples",
    "output_bytes",
    "tones_kept",
    "error",
    "finished_at",
)


def design_broadband_highpass(sampling_rate: int,
                              cutoff_hz: float,
                              transition_width_hz: float,
                              stopband_attenuation_db: float) -> tuple[np.ndarray, float]:
    """
    Description
    -----------
    Designs the linear-phase high-pass FIR of the broadband filter: a
    Kaiser-windowed sinc (``scipy.signal.firwin``) whose length and Kaiser beta
    come from ``scipy.signal.kaiserord`` for the requested stopband attenuation
    and transition width, the same construction sox's ``sinc -t <width> <cutoff>``
    effect uses. With the defaults (2 kHz cutoff, 1 kHz transition, 120 dB) at
    250 kHz the filter has 1953 taps and measures -6.02 dB at 2 kHz, below
    -120 dB at 1.5 kHz and below, -0.32 dB at 2.25 kHz and 0.00 dB from 2.5 kHz
    up (sox's own ``sinc -t 1000 2k``, measured with an impulse: 1861 taps,
    -6.02 dB at 2 kHz, -154.6 dB at 1.5 kHz, -0.31 dB at 2.25 kHz, 0.00 dB from
    2.5 kHz).

    The filter is applied centred (zero delay), so the output sample ``n`` lines
    up with the input sample ``n``.

    Parameters
    ----------
    sampling_rate (int)
        Audio sampling rate (Hz).
    cutoff_hz (float)
        -6 dB point of the high-pass (Hz); the transition band is centred on it.
    transition_width_hz (float)
        Width of the transition band (Hz), from the stopband edge
        ``cutoff_hz - transition_width_hz / 2`` to the passband edge
        ``cutoff_hz + transition_width_hz / 2``.
    stopband_attenuation_db (float)
        Minimum stopband attenuation (dB) the Kaiser design targets.

    Returns
    -------
    taps (np.ndarray)
        Odd-length, symmetric float64 FIR coefficients.
    kaiser_beta (float)
        The Kaiser window beta used.
    """

    numtaps, kaiser_beta = signal.kaiserord(stopband_attenuation_db, transition_width_hz / (sampling_rate / 2))
    # a high-pass FIR must have an odd number of taps (a type-I filter); this
    # also gives an integer group delay, so the output can be centred exactly
    numtaps = int(numtaps) | 1
    taps = signal.firwin(numtaps, cutoff_hz, window=("kaiser", kaiser_beta), pass_zero=False, fs=sampling_rate)
    return taps, float(kaiser_beta)


def highpass_response_db(taps: np.ndarray, sampling_rate: int, frequencies_hz) -> dict:
    """
    Description
    -----------
    Magnitude response (dB) of an FIR at a few frequencies, for the report.

    Parameters
    ----------
    taps (np.ndarray)
        FIR coefficients.
    sampling_rate (int)
        Audio sampling rate (Hz).
    frequencies_hz (iterable of float)
        Frequencies (Hz) at which to evaluate the response.

    Returns
    -------
    response (dict)
        ``{str(frequency): gain in dB rounded to 3 decimals}``.
    """

    frequencies = np.asarray(list(frequencies_hz), dtype=np.float64)
    _, response = signal.freqz(taps, worN=frequencies, fs=sampling_rate)
    gains = 20 * np.log10(np.maximum(np.abs(response), 1e-300))
    return {f"{frequency:g}": round(float(gain), 3) for frequency, gain in zip(frequencies, gains, strict=True)}


def unit_phasor(frequency_hz: float, sampling_rate: int, start: int, length: int, sign: int = 1) -> np.ndarray:
    """
    Description
    -----------
    Returns ``exp(sign * 1j * 2 * pi * frequency_hz * n / sampling_rate)`` for
    the ABSOLUTE sample indices ``n = start, ..., start + length - 1``.

    The phase is referenced to sample 0 of the recording, so phasors computed
    for different chunks of one recording join without a phase jump. To keep
    the phase exact at sample indices of ~3e8 (a 20 min recording at 250 kHz)
    the phase of ``start`` is reduced modulo one turn, and the remaining offsets
    ``m = a * R + b`` are built as the outer product of two short tables
    ``exp(i w a R)`` and ``exp(i w b)``, which is also much cheaper than one
    complex exponential per sample.

    Parameters
    ----------
    frequency_hz (float)
        Tone frequency (Hz).
    sampling_rate (int)
        Audio sampling rate (Hz).
    start (int)
        Absolute index of the first sample.
    length (int)
        Number of samples.
    sign (int)
        +1 for ``exp(+i w n)``, -1 for ``exp(-i w n)``.

    Returns
    -------
    phasor (np.ndarray)
        complex128 array of shape ``(length,)``.
    """

    if length <= 0:
        return np.zeros(0, dtype=np.complex128)
    cycles_per_sample = frequency_hz / sampling_rate
    row_length = 1024
    n_rows = -(-length // row_length)
    start_turns = np.mod(cycles_per_sample * float(start), 1.0)
    row_turns = np.mod(cycles_per_sample * row_length * np.arange(n_rows, dtype=np.float64), 1.0)
    column_turns = cycles_per_sample * np.arange(row_length, dtype=np.float64)
    rows = np.exp(sign * 2j * np.pi * (row_turns + start_turns))
    columns = np.exp(sign * 2j * np.pi * column_turns)
    return np.outer(rows, columns).ravel()[:length]


def block_mean_phasors(samples: np.ndarray, frequency_hz: float, sampling_rate: int, start: int, block_length: int) -> np.ndarray:
    """
    Description
    -----------
    Complex demodulation of a stretch of audio at one frequency: the mean of
    ``x[n] * exp(-i w n)`` over consecutive blocks of ``block_length`` samples
    (the last block may be shorter), with ``n`` the ABSOLUTE sample index. For a
    tone ``A cos(w n + phi)`` the block mean is ``(A / 2) exp(i phi)``; everything
    more than ~``1 / block duration`` Hz away from ``frequency_hz`` averages out.

    Each block is computed as ``exp(-i w k L) * (x_block @ exp(-i w j)) / L`` (block
    start ``k L``, offsets ``j < L``), a matrix-vector product instead of a
    complex exponential per sample.

    Parameters
    ----------
    samples (np.ndarray)
        Real samples, the first one at absolute index ``start``.
    frequency_hz (float)
        Demodulation frequency (Hz).
    sampling_rate (int)
        Audio sampling rate (Hz).
    start (int)
        Absolute index of ``samples[0]``.
    block_length (int)
        Block length in samples.

    Returns
    -------
    phasors (np.ndarray)
        complex128 array, one value per block.
    """

    samples = np.asarray(samples, dtype=np.float64)
    n_full = samples.shape[0] // block_length
    offsets = unit_phasor(frequency_hz, sampling_rate, 0, block_length, sign=-1)
    phasors = []
    if n_full > 0:
        blocks = samples[:n_full * block_length].reshape(n_full, block_length)
        sums = blocks @ offsets.real + 1j * (blocks @ offsets.imag)
        block_starts = start + block_length * np.arange(n_full, dtype=np.float64)
        rotation = np.exp(-2j * np.pi * np.mod(frequency_hz / sampling_rate * block_starts, 1.0))
        phasors.append(rotation * sums / block_length)
    remainder = samples[n_full * block_length:]
    if remainder.shape[0] > 0:
        tail_start = start + n_full * block_length
        tail = remainder @ unit_phasor(frequency_hz, sampling_rate, tail_start, remainder.shape[0], sign=-1)
        phasors.append(np.array([tail / remainder.shape[0]]))
    if not phasors:
        return np.zeros(0, dtype=np.complex128)
    return np.concatenate(phasors)


def estimate_line_noise(wav_path: str | pathlib.Path,
                        search_bands_hz: list,
                        min_height_db: float,
                        n_windows: int,
                        window_s: float,
                        max_tones_per_band: int,
                        min_separation_hz: float,
                        floor_window_hz: float) -> list[dict]:
    """
    Description
    -----------
    Finds the narrow line-noise tones of one channel. For every search band
    ``[lo, hi]`` the channel is read in ``n_windows`` windows of ``window_s``
    seconds spread evenly over the recording; each window is demodulated at the
    band centre and decimated to ~1 kHz (block means of ``sampling_rate // 1000``
    samples, flat to within 0.04 dB over +-50 Hz), Hann-windowed and Fourier
    transformed with 8x zero padding, and the power spectra of the windows are
    averaged (frequency grid ``1 / (8 window_s)`` Hz, resolution ``~2 / window_s``
    Hz).

    The averaged spectrum is whitened by its running median over
    ``floor_window_hz`` (the local floor), so a sloping background (the HPSS
    spectrum rises steeply between 1.5 and 2.6 kHz) neither inflates the height
    of a band edge nor hides a line next to a stronger background. Within the
    band, peaks of the whitened spectrum are taken greedily, highest first, while
    they stand at least ``min_height_db`` above the local floor, are local maxima
    of the power, and lie at least ``max(min_separation_hz, 2 / window_s)`` Hz
    (so outside the Hann main lobe) from every peak already taken, up to
    ``max_tones_per_band`` peaks; these are KEPT (later subtracted). Several
    peaks per band are needed because the low lines come as combs and close
    doublets (e.g. 2090.05 / 2090.31, 2120.06 / 2120.42 and 2150.05 Hz on the slave
    channels of one session). Each frequency is refined by parabolic
    interpolation of the log power. The frequencies are estimated per channel
    because the lines follow each recording device's clock (e.g. 8000.17 Hz on the
    master and 8000.67 Hz on the slave device of one session). A band without any
    kept peak reports its highest whitened peak with ``kept`` False.

    Parameters
    ----------
    wav_path (str | pathlib.Path)
        Single-channel int16 wav.
    search_bands_hz (list)
        ``[[lo, hi], ...]`` search bands (Hz), each at most 100 Hz wide.
    min_height_db (float)
        Minimum peak height above the local floor (dB) to keep a tone.
    n_windows (int)
        Number of analysis windows.
    window_s (float)
        Length of each window (s).
    max_tones_per_band (int)
        Maximum number of tones kept per band.
    min_separation_hz (float)
        Minimum distance (Hz) between two kept tones of a band.
    floor_window_hz (float)
        Width (Hz) of the running median that estimates the local floor.

    Returns
    -------
    tones (list of dict)
        Per search band, its kept tones (or its best candidate when none is
        kept), each with ``search_band_hz`` ([lo, hi]), ``frequency_hz`` (float),
        ``height_db`` (float) and ``kept`` (bool).
    """

    info = sf.info(str(wav_path))
    sampling_rate = int(info.samplerate)
    n_frames = int(info.frames)
    decimation = max(1, sampling_rate // 1000)
    window_length = min(n_frames, int(round(window_s * sampling_rate)))
    window_length -= window_length % decimation
    decimated_rate = sampling_rate / decimation
    n_decimated = window_length // decimation
    n_fft = 8 * n_decimated
    hann = np.hanning(n_decimated)
    frequency_offsets = np.fft.fftshift(np.fft.fftfreq(n_fft, d=1.0 / decimated_rate))
    bin_hz = frequency_offsets[1] - frequency_offsets[0]
    window_starts = np.unique(np.linspace(0, n_frames - window_length, max(1, n_windows)).astype(np.int64))
    spectra = [np.zeros(n_fft, dtype=np.float64) for _ in search_bands_hz]
    with sf.SoundFile(str(wav_path)) as sound_file:
        for window_start in window_starts:
            sound_file.seek(int(window_start))
            window = sound_file.read(window_length, dtype="int16").astype(np.float64)
            for band_index, (low_hz, high_hz) in enumerate(search_bands_hz):
                centre_hz = 0.5 * (low_hz + high_hz)
                decimated = block_mean_phasors(window, centre_hz, sampling_rate, int(window_start), decimation)
                spectrum = np.fft.fftshift(np.fft.fft(decimated * hann, n=n_fft))
                spectra[band_index] += np.abs(spectrum) ** 2
    main_lobe_hz = 2.0 * decimated_rate / n_decimated
    separation_bins = int(np.ceil(max(min_separation_hz, main_lobe_hz) / bin_hz))
    lobe_bins = max(1, int(np.ceil(main_lobe_hz / bin_hz)))
    floor_bins = max(3, int(round(floor_window_hz / bin_hz)) | 1)
    tones = []
    for band_index, (low_hz, high_hz) in enumerate(search_bands_hz):
        centre_hz = 0.5 * (low_hz + high_hz)
        power = spectra[band_index] / len(window_starts)
        in_band = np.flatnonzero((frequency_offsets >= low_hz - centre_hz) & (frequency_offsets <= high_hz - centre_hz))
        context_low = max(0, int(in_band[0]) - floor_bins)
        context_high = min(n_fft, int(in_band[-1]) + floor_bins + 1)
        floor = median_filter(power[context_low:context_high], size=floor_bins, mode='nearest')[in_band - context_low]
        whitened = power[in_band] / np.maximum(floor, np.finfo(np.float64).tiny)
        candidates = []
        for position in np.argsort(whitened)[::-1]:
            peak = int(in_band[position])
            height_db = float(10 * np.log10(max(whitened[position], np.finfo(np.float64).tiny)))
            n_kept = sum(candidate['kept'] for candidate in candidates)
            if candidates and (height_db < min_height_db or n_kept >= max_tones_per_band):
                break
            if power[peak] < power[max(0, peak - lobe_bins):peak + lobe_bins + 1].max():
                continue
            if any(abs(peak - candidate['bin']) < separation_bins for candidate in candidates):
                continue
            frequency_hz = centre_hz + frequency_offsets[peak]
            if 0 < peak < n_fft - 1 and np.all(power[peak - 1:peak + 2] > 0):
                left, middle, right = np.log(power[peak - 1:peak + 2])
                curvature = left - 2 * middle + right
                if curvature < 0:
                    frequency_hz += 0.5 * (left - right) / curvature * bin_hz
            candidates.append({'bin': peak, 'kept': bool(height_db >= min_height_db),
                               'frequency_hz': round(float(frequency_hz), 4), 'height_db': round(height_db, 2)})
            if not candidates[-1]['kept']:
                break
        kept = [candidate for candidate in candidates if candidate['kept']]
        for candidate in (kept if kept else candidates[:1]):
            tones.append({
                "search_band_hz": [float(low_hz), float(high_hz)],
                "frequency_hz": candidate['frequency_hz'],
                "height_db": candidate['height_db'],
                "kept": candidate['kept'],
            })
    return tones


class _BroadbandChannelStream:
    """
    Description
    -----------
    Streams one single-channel wav through the broadband filter, chunk by chunk
    in time order, reading every sample exactly once.

    For every kept line-noise tone it keeps the complex demodulation of each
    ``block_length`` block (absolute block grid ``k * block_length``), smooths it
    with a running median over ``2 * half_window + 1`` blocks (real and imaginary
    parts separately, the window truncated at the recording edges; the median
    ignores short transients such as a call crossing the tone frequency),
    interpolates the smoothed complex envelope linearly between block centres
    and subtracts the tone ``2 Re(envelope(n) exp(i w n))``. The cleaned signal is
    then convolved with the centred high-pass FIR (zero padding beyond the
    recording edges) and rounded to int16 (round half to even, clipped, no
    dither). Because the block grid, the median windows and the FIR are all
    defined on absolute sample indices, the output does not depend on the chunk
    length (up to float rounding of the FFT convolution).
    """

    def __init__(self, wav_path: pathlib.Path, n_samples: int, sampling_rate: int,
                 taps: np.ndarray, tone_frequencies_hz: list,
                 block_length: int, half_window: int) -> None:
        """
        Description
        -----------
        Opens the wav and prepares the per-tone state.

        Parameters
        ----------
        wav_path (pathlib.Path)
            Single-channel int16 wav.
        n_samples (int)
            Number of samples in the wav.
        sampling_rate (int)
            Audio sampling rate (Hz).
        taps (np.ndarray)
            Odd-length high-pass FIR, applied centred.
        tone_frequencies_hz (list of float)
            Frequencies (Hz) of the tones to subtract (may be empty).
        block_length (int)
            Demodulation block length (samples).
        half_window (int)
            Half-width (blocks) of the running median.

        Returns
        -------
        None
        """

        self.sound_file = sf.SoundFile(str(wav_path))
        self.n_samples = n_samples
        self.sampling_rate = sampling_rate
        self.taps = taps
        self.half_taps = (taps.shape[0] - 1) // 2
        self.tone_frequencies_hz = list(tone_frequencies_hz)
        self.block_length = block_length
        self.half_window = half_window
        self.n_blocks = -(-n_samples // block_length)
        block_starts = block_length * np.arange(self.n_blocks, dtype=np.float64)
        block_lengths = np.minimum(block_starts + block_length, n_samples) - block_starts
        self.block_centres = block_starts + (block_lengths - 1) / 2
        self.raw_phasors = [np.zeros(self.n_blocks, dtype=np.complex128) for _ in self.tone_frequencies_hz]
        self.blocks_done = 0
        self.buffer = np.zeros(0, dtype=np.int16)
        self.buffer_start = 0
        self.read_upto = 0

    def close(self) -> None:
        """
        Description
        -----------
        Closes the wav.

        Returns
        -------
        None
        """

        self.sound_file.close()

    def _read_until(self, end_sample: int) -> None:
        """
        Description
        -----------
        Extends the sample buffer up to ``end_sample`` (exclusive, clipped to the
        recording) and demodulates every block that became complete.

        Parameters
        ----------
        end_sample (int)
            Absolute sample index to read up to.

        Returns
        -------
        None
        """

        end_sample = min(end_sample, self.n_samples)
        if end_sample > self.read_upto:
            self.sound_file.seek(self.read_upto)
            new_samples = self.sound_file.read(end_sample - self.read_upto, dtype="int16")
            if new_samples.shape[0] != end_sample - self.read_upto:
                raise OSError(f"Short read from '{self.sound_file.name}': expected {end_sample - self.read_upto} samples, got {new_samples.shape[0]}.")
            self.buffer = np.concatenate([self.buffer, new_samples])
            self.read_upto = end_sample
        if not self.tone_frequencies_hz:
            return
        first_block = self.blocks_done
        last_block = self.n_blocks if self.read_upto >= self.n_samples else self.read_upto // self.block_length
        if last_block > first_block:
            block_start = first_block * self.block_length
            block_end = min(last_block * self.block_length, self.n_samples)
            stretch = self.buffer[block_start - self.buffer_start:block_end - self.buffer_start]
            for tone_index, frequency_hz in enumerate(self.tone_frequencies_hz):
                self.raw_phasors[tone_index][first_block:last_block] = block_mean_phasors(
                    stretch, frequency_hz, self.sampling_rate, block_start, self.block_length)
            self.blocks_done = last_block

    def smoothed_phasors(self, tone_index: int, first_block: int, last_block: int) -> np.ndarray:
        """
        Description
        -----------
        Running-median smoothed envelope of one tone for blocks
        ``first_block .. last_block`` (inclusive); the median window is
        ``[k - half_window, k + half_window]`` truncated to the recording.

        Parameters
        ----------
        tone_index (int)
            Index of the tone.
        first_block, last_block (int)
            Block range (inclusive).

        Returns
        -------
        envelope (np.ndarray)
            complex128 array of length ``last_block - first_block + 1``.
        """

        raw = self.raw_phasors[tone_index]
        envelope = np.empty(last_block - first_block + 1, dtype=np.complex128)
        for out_index, block in enumerate(range(first_block, last_block + 1)):
            window = raw[max(0, block - self.half_window):min(self.n_blocks, block + self.half_window + 1)]
            envelope[out_index] = np.median(window.real) + 1j * np.median(window.imag)
        return envelope

    def process(self, start: int, end: int) -> np.ndarray:
        """
        Description
        -----------
        Returns the filtered int16 output for samples ``[start, end)``; must be
        called with consecutive, non-overlapping ranges in time order.

        Parameters
        ----------
        start, end (int)
            Absolute output sample range.

        Returns
        -------
        output (np.ndarray)
            int16 array of length ``end - start``.
        """

        need_low = max(0, start - self.half_taps)
        need_high = min(self.n_samples, end + self.half_taps)
        offset = (self.block_length - 1) / 2
        first_block = max(0, int(np.floor((need_low - offset) / self.block_length)))
        last_block = min(self.n_blocks - 1, int(np.floor((need_high - 1 - offset) / self.block_length)) + 1)
        if self.tone_frequencies_hz:
            last_needed_block = min(self.n_blocks - 1, last_block + self.half_window)
            self._read_until((last_needed_block + 1) * self.block_length)
        else:
            self._read_until(need_high)

        segment = self.buffer[need_low - self.buffer_start:need_high - self.buffer_start].astype(np.float64)
        if self.tone_frequencies_hz:
            sample_index = np.arange(need_low, need_high, dtype=np.float64)
            centres = self.block_centres[first_block:last_block + 1]
            for tone_index, frequency_hz in enumerate(self.tone_frequencies_hz):
                envelope = self.smoothed_phasors(tone_index, first_block, last_block)
                envelope_at_samples = np.interp(sample_index, centres, envelope.real) + 1j * np.interp(sample_index, centres, envelope.imag)
                carrier = unit_phasor(frequency_hz, self.sampling_rate, need_low, need_high - need_low, sign=1)
                segment -= 2.0 * (envelope_at_samples * carrier).real

        padded = np.zeros(end - start + 2 * self.half_taps, dtype=np.float64)
        pad_left = need_low - (start - self.half_taps)
        padded[pad_left:pad_left + segment.shape[0]] = segment
        filtered = signal.oaconvolve(padded, self.taps, mode="valid")
        output = np.clip(np.rint(filtered), -32768, 32767).astype(np.int16)

        # keep only what the next chunk can still need (its FIR margin and the
        # blocks not yet demodulated)
        keep_from = max(0, end - self.half_taps)
        if self.tone_frequencies_hz:
            keep_from = min(keep_from, self.blocks_done * self.block_length)
        keep_from = max(keep_from, self.buffer_start)
        self.buffer = self.buffer[keep_from - self.buffer_start:]
        self.buffer_start = keep_from
        return output


def broadband_source_files(root_directory: str | pathlib.Path, settings: dict) -> list[pathlib.Path]:
    """
    Description
    -----------
    The single-channel source wavs of the broadband filter, sorted by name:
    ``<root>/audio/<source_dir>/<source_glob>`` (by default the 24 full-band
    HPSS wavs ``audio/hpss/*_cropped_to_video_hpss.wav``). The sorted order is
    the memmap column order, the same order ``concatenate_audio_files`` uses for
    the ``hpss_filtered`` memmap (master channels 1-12, then slave 1-12). Stray
    files in the folder (e.g. an empty ``output.wav``) do not match the glob.

    Parameters
    ----------
    root_directory (str | pathlib.Path)
        Session root directory.
    settings (dict)
        The ``broadband_filter_audio`` settings block.

    Returns
    -------
    wav_paths (list of pathlib.Path)
        Sorted source wavs.
    """

    source_dir = pathlib.Path(root_directory) / "audio" / settings["source_dir"]
    return sorted(source_dir.glob(settings["source_glob"]), key=lambda path: path.name)


def broadband_highpass_cutoff(settings: dict) -> float:
    """
    Description
    -----------
    The high-pass cutoff of the broadband filter, read from its removed band
    ``filter_freq_bounds`` = ``[low, high]`` (Hz), the same convention as the
    ``filter_audio_files`` block: the band between ``low`` and ``high`` is
    filtered out, so ``[0, 30000]`` there is the 30 kHz high-pass of the USV
    wavs and ``[0, 2000]`` here is the 2 kHz high-pass of the broadband memmap.
    The broadband filter is a high-pass only, so ``low`` must be 0; ``high`` is
    the -6 dB point of the filter.

    Parameters
    ----------
    settings (dict)
        The ``broadband_filter_audio`` settings block.

    Returns
    -------
    cutoff_hz (float)
        The -6 dB point of the high-pass (Hz), ``filter_freq_bounds[1]``.

    Raises
    ------
    ValueError
        ``filter_freq_bounds`` is not two numbers ``[0, high]`` with ``high > 0``.
    """

    bounds = settings['filter_freq_bounds']
    if len(bounds) != 2 or bounds[0] != 0 or not bounds[1] > 0:
        error_message = f"broadband filter_freq_bounds must be [0, high] with high > 0 (the filter is a high-pass), got {bounds}."
        raise ValueError(error_message)
    return float(bounds[1])


def broadband_output_settings(settings: dict) -> dict:
    """
    Description
    -----------
    The subset of the broadband settings that decides the written output
    (``BROADBAND_OUTPUT_SETTINGS``, plus the high-pass cutoff as
    ``highpass_cutoff_hz``, the upper edge of ``filter_freq_bounds``, see
    :func:`broadband_highpass_cutoff`), JSON-normalised (tuples become lists) so
    it compares equal to the copy stored in ``line_noise.json``. The cutoff is
    recorded as one number under ``highpass_cutoff_hz`` because that is the
    form every report written so far holds, so a report stays current as long
    as the filter it describes is unchanged.

    Parameters
    ----------
    settings (dict)
        The ``broadband_filter_audio`` settings block.

    Returns
    -------
    output_settings (dict)
        The output-defining settings.
    """

    broadband_highpass_cutoff(settings)
    output_settings = {key: settings[key] for key in BROADBAND_OUTPUT_SETTINGS}
    output_settings['highpass_cutoff_hz'] = settings['filter_freq_bounds'][1]
    return json.loads(json.dumps(output_settings))


def validate_broadband_output(root_directory: str | pathlib.Path, settings: dict) -> tuple[bool, str]:
    """
    Description
    -----------
    Checks whether a session already holds a complete, current broadband
    memmap: the expected memmap (name from the sources' recording id, sampling
    rate, sample count and channel count) exists with the exact byte size, it is
    the only broadband memmap in the folder, and ``line_noise.json`` exists,
    is marked complete, names that memmap, lists the same source files with the
    same byte sizes, and records the same output-defining settings.

    Parameters
    ----------
    root_directory (str | pathlib.Path)
        Session root directory.
    settings (dict)
        The ``broadband_filter_audio`` settings block.

    Returns
    -------
    valid (bool)
        True if the existing output can be kept as is.
    reason (str)
        Why it is (not) valid.
    """

    wav_paths = broadband_source_files(root_directory, settings)
    if not wav_paths:
        return False, "no source wavs"
    infos = [sf.info(str(path)) for path in wav_paths]
    output_dir = pathlib.Path(root_directory) / "audio" / AUDIO_MMAP_BAND_FOLDERS["broadband"]
    expected_name = (f"{wav_paths[0].name.split('_')[1]}_concatenated_audio_{AUDIO_MMAP_BAND_FOLDERS['broadband']}_"
                     f"{int(infos[0].samplerate)}_{int(infos[0].frames)}_{len(wav_paths)}_int16.mmap")
    mmap_path = output_dir / expected_name
    report_path = output_dir / BROADBAND_REPORT_NAME
    if not mmap_path.is_file():
        return False, f"missing {expected_name}"
    if mmap_path.stat().st_size != int(infos[0].frames) * len(wav_paths) * 2:
        return False, f"{expected_name} has {mmap_path.stat().st_size} bytes, expected {int(infos[0].frames) * len(wav_paths) * 2}"
    regex = audio_mmap_name_regex("broadband")
    others = [path.name for path in output_dir.iterdir() if regex.match(path.name) and path.name != expected_name]
    if others:
        return False, f"other broadband memmaps present: {others}"
    if not report_path.is_file():
        return False, f"missing {BROADBAND_REPORT_NAME}"
    try:
        with open(report_path, encoding="utf-8") as report_file:
            report = json.load(report_file)
    except (OSError, ValueError) as report_error:
        return False, f"unreadable {BROADBAND_REPORT_NAME}: {report_error}"
    current_sources = [{"column": column, "file": path.name, "bytes": path.stat().st_size} for column, path in enumerate(wav_paths)]
    try:
        if report["complete"] is not True:
            return False, "report not marked complete"
        if report["output"]["file"] != expected_name:
            return False, "report names another memmap"
        if report["sources"] != current_sources:
            return False, "source files changed"
        if report["settings"] != broadband_output_settings(settings):
            return False, "settings changed"
    except (KeyError, TypeError) as report_error:
        return False, f"malformed {BROADBAND_REPORT_NAME}: missing {report_error}"
    return True, "complete and current"


class Operator:

    def __init__(self, root_directory: str | list[str] = None,
                 input_parameter_dict: dict = None,
                 message_output: Callable | None = None):
        """
        Description
        -----------
        Initializes the Operator class.

        Parameters
        ----------
        root_directory (str / list of str)
            Root directory for data; defaults to None.
        input_parameter_dict (dict)
            Processing parameters; defaults to None.
        message_output (function)
            Defines output messages; defaults to None.

        Returns
        -------
        None
        """

        if input_parameter_dict is None or root_directory is None:
            with open(pathlib.Path(__file__).parent.parent / '_parameter_settings/processing_settings.json') as json_file:
                _settings = json.load(json_file)

        if input_parameter_dict is not None:
            self.input_parameter_dict = input_parameter_dict['modify_files']['Operator']
            self.input_parameter_dict_2 = input_parameter_dict['synchronize_files']['Synchronizer']
        else:
            self.input_parameter_dict = _settings['modify_files']['Operator']
            self.input_parameter_dict_2 = _settings['synchronize_files']['Synchronizer']

        self.root_directory = root_directory if root_directory is not None else _settings['modify_files']['root_directory']
        self.message_output = message_output if message_output is not None else print

        self.app_context_bool = is_gui_context()

    def split_clusters_to_sessions(self) -> None:
        """
        Description
        -----------
        This method converts every spike sample time into seconds,
        relative to tracking start and splits spikes back into
        individual sessions (if binary files were concatenated).

        NB: If you have recorded multiple sessions in one day,
        it is sufficient to put only one root directory for that day,
        e.g., the first one. The script will find EPHYS root directory,
        and split spikes from all probes into sessions based on the
        inputs in the changepoints JSON file.

        Parameters
        ----------

        Returns
        -------
         spike times (np.ndarray)
            Arrays that contain spike times: seconds (row 0) and frames (row 1);
            saved as .npy files in a separate directory.
        """

        self.message_output(f"Splitting clusters to sessions started at: {datetime.now().hour:02d}:{datetime.now().minute:02d}:{datetime.now().second:02d}")
        smart_wait(app_context_bool=self.app_context_bool, seconds=1)

        # read headstage sampling rates
        calibrated_sr_config = configparser.ConfigParser()
        calibrated_sr_config.read(pathlib.Path(__file__).parent.parent / '_config/calibrated_sample_rates_imec.ini')

        # optional near-coincident duplicate-spike removal (e.g. from Phy merges that
        # combine two templates' detections of the same physical spike); reuses the
        # same SpikeInterface routine as the amplitude-CV quality metric
        remove_duplicate_spikes = bool(self.input_parameter_dict['get_spike_times']['remove_duplicate_spikes'])
        duplicate_censored_period_ms = float(self.input_parameter_dict['get_spike_times']['duplicate_censored_period_ms'])

        for one_root_dir in self.root_directory:
            _ephys_base = ephys_base_for_data_root(one_root_dir) / pathlib.Path(one_root_dir).name.split('_')[0]
            for ephys_dir in sorted(_ephys_base.parent.glob(f"{_ephys_base.name}_imec*")):

                probe_id = re.search(r'imec\d', ephys_dir.name).group()

                self.message_output(f"Working on getting spike times from clusters in: {ephys_dir}, started at {datetime.now()}.")

                # load the changepoint .json file
                with open(
                    first_match_or_raise(
                        root=ephys_dir,
                        pattern='changepoints_info_*.json',
                        label="ephys changepoints_info JSON",
                    ),
                    'r',
                ) as binary_info_input_file:
                    binary_files_info = json.load(binary_info_input_file)

                    for session_key in binary_files_info.keys():
                        binary_files_info[session_key]['root_directory'] = configure_path(pa=binary_files_info[session_key]['root_directory'])

                # get info about session start
                se_dict = {}
                esr_dict = {}
                frame_least_dict = {}
                root_dict = {}
                unit_count_dict = {'noise': 0, 'unsorted': 0}
                for session_key in binary_files_info.keys():

                    unit_count_dict[session_key] = {'good': 0, 'mua': 0}

                    # load info from camera_frame_count_dict
                    with open(
                        first_match_or_raise(
                            root=pathlib.Path(binary_files_info[session_key]['root_directory']),
                            pattern='video/*_camera_frame_count_dict.json',
                            label=f"camera frame count JSON for session '{session_key}'",
                        ),
                        'r',
                    ) as frame_count_infile:
                        camera_frame_info = json.load(frame_count_infile)
                        esr_dict[session_key] = camera_frame_info['median_empirical_camera_sr']
                        frame_least_dict[session_key] = camera_frame_info['total_frame_number_least']
                        root_dict[session_key] = binary_files_info[session_key]['root_directory']

                    if any(np.isnan(value) for value in binary_files_info[session_key]['tracking_start_end']):
                        se_dict[session_key] = binary_files_info[session_key]['session_start_end']
                    else:
                        se_dict[session_key] = binary_files_info[session_key]['tracking_start_end']

                    (pathlib.Path(root_dict[session_key]) / 'ephys' / probe_id / 'cluster_data').mkdir(parents=True, exist_ok=True)

                # duplicate-removal censored period in samples, at this probe's
                # calibrated headstage rate (shared across the probe's sessions)
                if remove_duplicate_spikes:
                    _dup_headstage_sn = binary_files_info[next(iter(binary_files_info))]['headstage_sn']
                    _dup_sr = float(calibrated_sr_config['CalibratedHeadStages'][_dup_headstage_sn])
                    duplicate_censored_period_samples = round(duplicate_censored_period_ms * 1e-3 * _dup_sr)

                # load the Kilosort output files
                ks_dir = ephys_dir / f"kilosort{self.input_parameter_dict['get_spike_times']['kilosort_version']}"
                phy_curation_bool = (ks_dir / 'cluster_info.tsv').is_file()
                spike_clusters = np.load(ks_dir / 'spike_clusters.npy')
                spike_times = np.load(ks_dir / 'spike_times.npy')

                if phy_curation_bool:
                    cluster_info = pls.read_csv(source=str(ks_dir / 'cluster_info.tsv'),
                                                separator='\t')
                else:
                    self.message_output("Phy2 curation has not been done for this session, no cluster_info.tsv file exists.")
                    continue

                smart_wait(app_context_bool=self.app_context_bool, seconds=1)

                # Group spike indices by cluster id in a single argsort pass instead
                # of re-scanning the whole spike_clusters array (millions of spikes)
                # once per cluster: O(n_spikes log n_spikes) once vs O(n_clusters *
                # n_spikes). A STABLE argsort keeps each cluster's indices ascending,
                # so cluster_order[run] equals the old np.sort(np.where(== cid)[0])
                # exactly (verified byte-identical against real Kilosort data);
                # ravel() makes it correct for both (N,) and (N, 1) spike_clusters.
                flat_spike_clusters = np.asarray(spike_clusters).ravel()
                cluster_order = np.argsort(flat_spike_clusters, kind='stable')
                sorted_cluster_ids = flat_spike_clusters[cluster_order]
                unique_cluster_ids, cluster_run_starts = np.unique(sorted_cluster_ids, return_index=True)
                cluster_run_bounds = np.append(cluster_run_starts, sorted_cluster_ids.shape[0])
                cluster_indices_by_id = {
                    int(unique_cluster_ids[ci]): cluster_order[cluster_run_bounds[ci]:cluster_run_bounds[ci + 1]]
                    for ci in range(unique_cluster_ids.shape[0])
                }

                for idx in tqdm(range(cluster_info.shape[0])):
                    if cluster_info[idx, 'group'] in ('good', 'mua'):

                        # collect all spikes for any given cluster (from the precomputed group)
                        cluster_id_val = int(cluster_info[idx, 'cluster_id'])
                        cluster_indices = (
                            cluster_indices_by_id[cluster_id_val]
                            if cluster_id_val in cluster_indices_by_id
                            else np.empty(0, dtype=cluster_order.dtype)
                        )
                        spike_events = np.take(spike_times, cluster_indices)

                        # drop near-coincident duplicate spikes for this unit before
                        # splitting into sessions (keeps the first of each violating pair)
                        if remove_duplicate_spikes and spike_events.shape[0] > 1:
                            duplicate_indices = find_duplicated_spikes(spike_events, duplicate_censored_period_samples,
                                                                       method="keep_first_iterative")
                            spike_events = np.delete(spike_events, duplicate_indices)

                        # filter spikes for each session
                        for session_key in binary_files_info.keys():
                            session_spikes_sec = ((spike_events[(spike_events >= se_dict[session_key][0]) & (spike_events < se_dict[session_key][1])] - se_dict[session_key][0]) /
                                                  float(calibrated_sr_config['CalibratedHeadStages'][binary_files_info[session_key]['headstage_sn']]))

                            session_spikes_fps = np.round(session_spikes_sec * esr_dict[session_key])
                            # clamp any frame index at or beyond the least total frame count to the last valid frame,
                            # so row 1 of session_spikes never indexes out of bounds into an array of length frame_least
                            session_spikes_fps[session_spikes_fps >= frame_least_dict[session_key]] = frame_least_dict[session_key]-1

                            session_spikes = np.vstack((session_spikes_sec, session_spikes_fps))

                            # save spiking data
                            if session_spikes_sec.shape[0] > self.input_parameter_dict['get_spike_times']['min_spike_num']:
                                cluster_id = f"{probe_id}_cl{cluster_info[idx, 'cluster_id']:04d}_ch{cluster_info[idx, 'ch']:03d}_{cluster_info[idx, 'group']}"
                                np.save(file=pathlib.Path(root_dict[session_key]) / 'ephys' / probe_id / 'cluster_data' / cluster_id, arr=session_spikes)

                                unit_count_dict[session_key][cluster_info[idx, 'group']] += 1

                    elif cluster_info[idx, 'group'] == 'noise':
                        unit_count_dict['noise'] += 1

                    else:
                        unit_count_dict['unsorted'] += 1

                self.message_output(f"For {ephys_dir}, there were {unit_count_dict['noise']} noise clusters and {unit_count_dict['unsorted']} unsorted clusters.")
                for session_key in binary_files_info.keys():
                    self.message_output(f"For {root_dict[session_key]} probe {probe_id}, there were {unit_count_dict[session_key]['good']} good and {unit_count_dict[session_key]['mua']} MUA clusters.")

    def concatenate_binary_files(self) -> None:
        """
        Description
        -----------
        This method concatenates binary files from Neuropixels recordings into one
        .bin file (can be used from "ap" or "lf" files). It goes through all root
        directories and concatenates all binary files for any given probe, say "imec0",
        into one binary file.

        NB: If you have recorded multiple sessions in one day,
        it is necessary to list all of their root directories
        to conduct concatenation. The script operates by first
        locating all available probes in the root directories,
        and then concatenates all binary files for each probe.

        Parameters
        ----------

        Returns
        -------
        binary_files_info (.json)
            Dictionary w/ information about changepoints and binary file lengths.
        concatenated (.bin)
           Concatenated binary file.
        """

        self.message_output(f"E-phys file concatenation started at: {datetime.now().hour:02d}:{datetime.now().minute:02d}:{datetime.now().second:02d}. "
                            f"Please be patient - this could take >1 hour.")
        smart_wait(app_context_bool=self.app_context_bool, seconds=1)

        # read headstage sampling rates
        calibrated_sr_config = configparser.ConfigParser()
        calibrated_sr_config.read(pathlib.Path(__file__).parent.parent / '_config/calibrated_sample_rates_imec.ini')

        # create list of directories to save concatenated files in
        concat_save_dir = []
        available_probes = []
        for one_root_dir in self.root_directory:
            ephys_save_dir_base = str(ephys_base_for_data_root(one_root_dir) / pathlib.Path(one_root_dir).name.split('_')[0])
            for one_probe_dir in sorted((pathlib.Path(one_root_dir) / 'ephys').iterdir()):
                if one_probe_dir.is_dir() and 'imec' in one_probe_dir.name:
                    if one_probe_dir.name not in available_probes:
                        available_probes.append(one_probe_dir.name)
                    if not any(one_probe_dir.name in one_concat_dir for one_concat_dir in concat_save_dir):
                        concat_save_dir.append(f'{ephys_save_dir_base}_{one_probe_dir.name}')

        npx_file_type = self.input_parameter_dict_2['validate_ephys_video_sync']['npx_file_type']
        # create dictionary to store information about binary files and generate stitching command
        for probe_idx, probe_id in enumerate(available_probes):
            binary_files_info = {}
            changepoints = [0]
            concatenation_command = 'copy /b ' if os.name == 'nt' else 'cat '
            total_size_bytes = 0
            total_time_secs = 0.0
            for one_root_dir in self.root_directory:
                ephys_probe_dir = pathlib.Path(one_root_dir) / 'ephys' / probe_id
                if ephys_probe_dir.is_dir():
                    for one_file in sorted(ephys_probe_dir.glob(f"*{npx_file_type}.bin*")):
                        # Derive each .meta from its own .bin (same stem, .meta
                        # extension) instead of zipping two independent sorted
                        # globs: the two globs can have different lengths / sort
                        # orders (e.g. a stray .bin without a sibling .meta), and
                        # zip would then silently pair a .bin with the WRONG
                        # session's .meta. `.bin` is always the final suffix here,
                        # so swapping it for `.meta` is exact.
                        one_meta = one_file.with_name(one_file.name[:-len(".bin")] + ".meta") if one_file.name.endswith(".bin") else one_file.with_suffix(".meta")
                        if one_file.is_file() and one_meta.is_file():

                            # parse metadata file for channel and headstage information.
                            # Initialise every parsed variable up front so a .meta
                            # missing one of these keys fails loudly below instead
                            # of silently reusing the previous file's value.
                            total_num_channels = None
                            headstage_sn = None
                            spike_glx_sr = None
                            imec_probe_sn = None
                            with open(one_meta) as meta_data_file:
                                for line in meta_data_file:
                                    key, value = line.strip().split("=")
                                    if key == 'acqApLfSy':
                                        total_num_channels = int(value.split(',')[0]) + int(value.split(',')[-1])
                                    elif key == 'imDatHs_sn':
                                        headstage_sn = value
                                        spike_glx_sr = float(calibrated_sr_config['CalibratedHeadStages'][headstage_sn])
                                    elif key == 'imDatPrb_sn':
                                        imec_probe_sn = value
                                    elif key == 'fileSizeBytes':
                                        total_size_bytes += int(value)
                                    elif key == 'fileTimeSecs':
                                        total_time_secs += float(value)

                            # Validate that the required keys were present; any
                            # of these being None means the .meta is malformed
                            # and the downstream channel/sample arithmetic would
                            # otherwise crash cryptically (or reuse stale state).
                            missing_meta_keys = [
                                meta_key
                                for meta_key, meta_val in (
                                    ('acqApLfSy', total_num_channels),
                                    ('imDatHs_sn', headstage_sn),
                                    ('imDatPrb_sn', imec_probe_sn),
                                )
                                if meta_val is None
                            ]
                            if missing_meta_keys:
                                raise KeyError(
                                    f"{one_meta} is missing required meta key(s): "
                                    f"{', '.join(missing_meta_keys)}."
                                )

                            binary_file_info_id = one_file.name[:-7]
                            binary_files_info[binary_file_info_id] = {'session_start_end': [np.nan, np.nan],
                                                                      'tracking_start_end': [np.nan, np.nan],
                                                                      'largest_camera_break_duration': np.nan,
                                                                      'file_duration_samples': np.nan,
                                                                      'root_directory': str(pathlib.Path(one_root_dir)),
                                                                      'total_num_channels': total_num_channels,
                                                                      'headstage_sn': headstage_sn,
                                                                      'imec_probe_sn': imec_probe_sn}

                            # Only the total sample count is needed from the
                            # binary; read its length, then release the memmap
                            # immediately so its file handle / mapping is not held
                            # open across the whole (potentially many-file) loop.
                            one_recording = np.memmap(filename=one_file, mode='r', dtype='int16', order='C')
                            one_recording_length = int(one_recording.shape[0])
                            del one_recording

                            self.message_output(f"File {pathlib.Path(one_file).name}, recorded with hs #{headstage_sn} & probe #{imec_probe_sn} has total length {one_recording_length}, or {one_recording_length // total_num_channels} "
                                                f"samples on {total_num_channels} channels, totaling {round((one_recording_length // total_num_channels) / (spike_glx_sr * 60), 2)} minutes of recording.")

                            binary_files_info[binary_file_info_id]['file_duration_samples'] = int(one_recording_length // total_num_channels)

                            if len(changepoints) == 1:
                                binary_files_info[binary_file_info_id]['session_start_end'][0] = 0
                                binary_files_info[binary_file_info_id]['session_start_end'][1] = int(one_recording_length // total_num_channels)
                                changepoints.append(int(one_recording_length // total_num_channels))
                                concatenation_command += '{} '.format(one_file)
                            else:
                                binary_files_info[binary_file_info_id]['session_start_end'][0] = changepoints[-1]
                                binary_files_info[binary_file_info_id]['session_start_end'][1] = int(one_recording_length // total_num_channels) + changepoints[-1]
                                changepoints.append(int(one_recording_length // total_num_channels) + changepoints[-1])
                                if os.name == 'nt':
                                    concatenation_command += '+ {} '.format(one_file)
                                else:
                                    concatenation_command += '{} '.format(one_file)

            # create save directory if one doesn't exist already
            concat_save_path = pathlib.Path(concat_save_dir[probe_idx])
            concat_save_path.mkdir(parents=True, exist_ok=True)

            # save concatenated META file
            concatenated_meta_file = concat_save_path / f"concatenated_{concat_save_path.name}.{npx_file_type}.meta"
            if not concatenated_meta_file.is_file():
                src_meta_file = first_match_or_raise(
                    root=ephys_probe_dir,
                    pattern=f"*{npx_file_type}.meta*",
                    label=f"source .{npx_file_type}.meta for concatenation",
                )
                with open(src_meta_file, 'r', encoding='utf-8') as f_in, \
                        open(concatenated_meta_file, 'w', encoding='utf-8') as f_out:
                    for line in f_in:
                        if line.strip().startswith('fileSizeBytes='):
                            f_out.write(f"fileSizeBytes={total_size_bytes}\n")
                        elif line.strip().startswith('fileTimeSecs='):
                            f_out.write(f"fileTimeSecs={total_time_secs}\n")
                        else:
                            f_out.write(line)

            if os.name == 'nt':
                concatenation_command += f'"{concat_save_path / f"concatenated_{concat_save_path.name}.{npx_file_type}.bin"}"'
            else:
                concatenation_command += f'> {concat_save_path / f"concatenated_{concat_save_path.name}.{npx_file_type}.bin"}'

            # save changepoint information in JSON file
            changepoints_json = concat_save_path / f'changepoints_info_{concat_save_path.name}.json'
            if not changepoints_json.exists():
                with open(changepoints_json, 'w') as binary_info_output_file:
                    json.dump(binary_files_info, binary_info_output_file, indent=4)
            else:
                with open(changepoints_json, 'r') as existing_json_file:
                    changepoint_info_data = json.load(existing_json_file)

                for file_key in binary_files_info.keys():
                    if file_key not in changepoint_info_data:
                        changepoint_info_data[file_key] = binary_files_info[file_key]
                    else:
                        for component_key in changepoint_info_data[file_key].keys():
                            if component_key != 'tracking_start_end' and component_key != 'largest_camera_break_duration' and component_key != 'root_directory' and changepoint_info_data[file_key][component_key] != binary_files_info[file_key][component_key]:
                                changepoint_info_data[file_key][component_key] = binary_files_info[file_key][component_key]
                            elif component_key == 'tracking_start_end' and changepoint_info_data[file_key][component_key] != [np.nan, np.nan]:
                                changepoint_info_data[file_key][component_key][0] = changepoint_info_data[file_key][component_key][0] + binary_files_info[file_key]['session_start_end'][0]
                                changepoint_info_data[file_key][component_key][1] = changepoint_info_data[file_key][component_key][1] + binary_files_info[file_key]['session_start_end'][0]

                with open(changepoints_json, 'w') as binary_info_output_file:
                    json.dump(changepoint_info_data, binary_info_output_file, indent=4)

            smart_wait(app_context_bool=self.app_context_bool, seconds=2)

            # run command in shell
            #
            # shell=True is required here: the command relies on shell I/O
            # redirection ('cat ... > out.bin' on POSIX) and the 'copy /b'
            # cmd.exe builtin on Windows, neither of which works with a plain
            # argv list. The arguments are the session's own .bin file paths,
            # not externally supplied input.
            #
            # No timeout is imposed on .wait(): concatenating ~10 multi-hour
            # e-phys recordings can legitimately run for hours or overnight,
            # and terminating a live 'cat'/'copy' would leave a truncated,
            # silently-corrupt .bin. We do, however, inspect the exit code so a
            # failed concatenation (e.g. full disk, unreadable source) is
            # surfaced loudly instead of being mistaken for a clean run by the
            # downstream stages that read this binary.
            concat_process = subprocess.Popen(args=concatenation_command,
                                              shell=True,
                                              stdout=subprocess.DEVNULL,
                                              stderr=subprocess.STDOUT,
                                              cwd=concat_save_dir[probe_idx])
            concat_return_code = concat_process.wait()
            if concat_return_code != 0:
                self.message_output(f"WARNING: binary concatenation in {concat_save_dir[probe_idx]} exited with "
                                    f"non-zero status {concat_return_code}; the concatenated .bin may be "
                                    f"incomplete or corrupt and should not be trusted downstream.")

    def multichannel_to_channel_audio(self) -> None:
        """
        Description
        -----------
        This method splits multichannel audio file into single channel files and
        concatenates single channel files via Sox, since multichannel files were
        split due to a size limitation.

        Parameters
        ----------

        Returns
        -------
        wave_file (.wav files)
            Concatenated single channel wave files.
        """

        self.message_output(f"Multichannel to single channel audio conversion started at: {datetime.now().hour:02d}:{datetime.now().minute:02d}:{datetime.now().second:02d}")
        smart_wait(app_context_bool=self.app_context_bool, seconds=1)

        (pathlib.Path(self.root_directory) / 'audio' / 'temp').mkdir(parents=True, exist_ok=True)

        # separate each channel of every multichannel audio file
        mc_audio_files = sorted((pathlib.Path(self.root_directory) / 'audio' / 'original_mc').glob('*.wav'))
        for mc_audio_file in mc_audio_files:
            separate_ch_subprocesses = []
            for ch in range(1, 13):
                output_file = str(pathlib.Path(self.root_directory) / 'audio' / 'temp' / f'{mc_audio_file.stem}_ch{ch:02d}.wav')
                sep_ch_subp = subprocess.Popen(args=["static_sox", mc_audio_file.name, output_file, "remix", str(ch)],
                                               cwd=pathlib.Path(self.root_directory) / 'audio' / 'original_mc',
                                               stdout=subprocess.DEVNULL,
                                               stderr=subprocess.STDOUT,
                                               shell=False)

                separate_ch_subprocesses.append(sep_ch_subp)

            wait_for_subprocesses(
                subps=separate_ch_subprocesses,
                max_seconds=2 * 60 * 60,
                label="multichannel channel-separation",
                poll_interval_s=5,
                message_output=self.message_output,
                raise_on_nonzero=False,
                raise_on_timeout=False,
            )

        smart_wait(app_context_bool=self.app_context_bool, seconds=2)

        # derive name_origin from the master MC file stem (strip the 'm_' device
        # prefix). stable regardless of (1) how many channels are in audio/temp,
        # (2) glob ordering, and (3) whether the avisoft filename contains
        # internal underscores. the previous approach — split('_')[2] on a
        # per-channel temp file — was an off-by-one that landed on the channel
        # suffix ('chNN.wav') whenever the MC stem had only two underscore-
        # separated tokens (the normal 'm_<datetime>' case), corrupting every
        # downstream filename (cropped_to_video, hpss, filtered, concatenated).
        master_mc_files = sorted(
            (pathlib.Path(self.root_directory) / 'audio' / 'original_mc').glob('m_*.wav')
        )
        if not master_mc_files:
            raise FileNotFoundError(
                f"master multichannel .wav for naming: no 'm_*.wav' files found under "
                f"'{pathlib.Path(self.root_directory) / 'audio' / 'original_mc'}'."
            )
        name_origin = master_mc_files[0].stem[2:]

        # concatenate single channel files for master/slave
        separation_subprocesses = []
        for device_id in ['m', 's']:
            for ch in range(1, 13):
                partial_input_files = [p.name for p in sorted((pathlib.Path(self.root_directory) / 'audio' / 'temp').glob(f'{device_id}_*_ch{ch:02d}.wav'))]
                output_file = str(pathlib.Path(self.root_directory) / 'audio' / 'original' / f'{device_id}_{name_origin}_ch{ch:02d}.wav')
                command_args = ['static_sox', *partial_input_files, '-q', output_file]
                mc_to_sc_subp = subprocess.Popen(args=command_args,
                                                 cwd=pathlib.Path(self.root_directory) / 'audio' / 'temp',
                                                 stdout=subprocess.DEVNULL,
                                                 stderr=subprocess.STDOUT,
                                                 shell=False)

                separation_subprocesses.append(mc_to_sc_subp)

        wait_for_subprocesses(
            subps=separation_subprocesses,
            max_seconds=2 * 60 * 60,
            label="master/slave single-channel concatenation",
            poll_interval_s=5,
            message_output=self.message_output,
            raise_on_nonzero=False,
            raise_on_timeout=False,
        )

        # delete temp directory (w/ all files in it)
        shutil.rmtree(pathlib.Path(self.root_directory) / 'audio' / 'temp')

    def hpss_audio(self) -> None:
        """
        Description
        -----------
        This function performs the harmonic/percussive source separation (HPSS)
        on the provided audio (WAV) files. The harmonic component is then converted
        back to the time domain and saved as a new WAV file.

        Parameters
        ----------

        Returns
        -------
        harmonic_data_clipped (.wav file)
            Output audio file w/ only the harmonics component.
        """

        self.message_output(f"Harmonic-percussive source separation started at: {datetime.now().hour:02d}:{datetime.now().minute:02d}:{datetime.now().second:02d}")
        smart_wait(app_context_bool=self.app_context_bool, seconds=1)

        wav_file_lst = sorted((pathlib.Path(self.root_directory) / 'audio' / 'cropped_to_video').glob('*.wav'))

        for one_wav_file in wav_file_lst:
            self.message_output(f"Working on file: {one_wav_file}")
            smart_wait(app_context_bool=self.app_context_bool, seconds=1)

            # read the audio file (use Scipy, not Librosa because Librosa performs scaling)
            sampling_rate_audio, audio_data = wavfile.read(one_wav_file)

            # convert to float32 because librosa.stft() requires float32
            audio_data = np.array(audio_data, dtype='float32')

            # perform Short-Time Fourier Transform (STFT) on the audio data
            spectrogram_data = librosa.stft(y=audio_data,
                                            n_fft=self.input_parameter_dict['hpss_audio']['stft_window_length_hop_size'][0],
                                            hop_length=self.input_parameter_dict['hpss_audio']['stft_window_length_hop_size'][1])

            # perform HPSS on the spectrogram data
            D_harmonic, _ = librosa.decompose.hpss(S=spectrogram_data,
                                                              kernel_size=self.input_parameter_dict['hpss_audio']['kernel_size'],
                                                              power=self.input_parameter_dict['hpss_audio']['hpss_power'],
                                                              mask=False,
                                                              margin=self.input_parameter_dict['hpss_audio']['margin'])

            # convert the harmonic component back to the time domain
            harmonic_data = librosa.istft(stft_matrix=D_harmonic,
                                          length=audio_data.shape[0],
                                          win_length=self.input_parameter_dict['hpss_audio']['stft_window_length_hop_size'][0],
                                          hop_length=self.input_parameter_dict['hpss_audio']['stft_window_length_hop_size'][1])

            # ensure the float values are within the range of 16-bit integers
            # clip values outside the range to the minimum and maximum representable values
            harmonic_data_clipped = np.clip(a=harmonic_data,
                                            a_min=-32768,
                                            a_max=32767).astype('int16')

            # save the harmonic component as a new WAV file
            hpss_dir = pathlib.Path(self.root_directory) / 'audio' / 'hpss'
            hpss_dir.mkdir(parents=True, exist_ok=True)

            wavfile.write(filename=hpss_dir / f'{one_wav_file.stem}_hpss.wav',
                          rate=sampling_rate_audio,
                          data=harmonic_data_clipped)

    def filter_audio_files(self) -> None:
        """
        Description
        -----------
        This method filters audio files via Sox.

        It applies a sinc kaiser-windowed low-pass, high-pass, band-pass, or band-reject filter
        to the signal. The freqHP and freqLP parameters give the frequencies of the 6dB points
        of a high-pass and low-pass filter that may be invoked individually, or together. If
        both are given, then freqHP less than freqLP creates a band-pass filter, freqHP greater
        than freqLP creates a band-reject filter. For example, the invocations:
           sinc 3k
           sinc -4k
           sinc 3k-4k
           sinc 4k-3k
        create a high-pass, low-pass, band-pass, and band-reject filter respectively.

        Parameters
        ----------

        Returns
        -------
        wave_file (.wav file)
            Filtered wave file.
        """

        freq_lp = self.input_parameter_dict['filter_audio_files']['filter_freq_bounds'][0]
        freq_hp = self.input_parameter_dict['filter_audio_files']['filter_freq_bounds'][1]

        self.message_output(f"Filtering out signal between {freq_lp} and {freq_hp} Hz in audio files started at: "
                            f"{datetime.now().hour:02d}:{datetime.now().minute:02d}:{datetime.now().second:02d}")
        smart_wait(app_context_bool=self.app_context_bool, seconds=1)

        for one_dir in self.input_parameter_dict['filter_audio_files']['filter_dirs']:

            filter_output_dir = pathlib.Path(self.root_directory) / 'audio' / f'{one_dir}_filtered'
            filter_output_dir.mkdir(parents=True, exist_ok=True)

            filter_subprocesses = []
            all_audio_files = sorted((pathlib.Path(self.root_directory) / 'audio' / one_dir).glob(f"*.{self.input_parameter_dict['filter_audio_files']['filter_audio_format']}"))

            if len(all_audio_files) > 0:
                for one_file in all_audio_files:
                    filter_subp = subprocess.Popen(args=["static_sox", "--ignore-length", one_file.name, str(filter_output_dir / f'{one_file.stem}_filtered.wav'), "sinc", f"{freq_hp}-{freq_lp}"],
                                                   cwd=pathlib.Path(self.root_directory) / 'audio' / one_dir,
                                                   stdout=subprocess.DEVNULL,
                                                   stderr=subprocess.STDOUT,
                                                   shell=False)

                    filter_subprocesses.append(filter_subp)

            wait_for_subprocesses(
                subps=filter_subprocesses,
                max_seconds=2 * 60 * 60,
                label="audio band-pass filtering",
                poll_interval_s=5,
                message_output=self.message_output,
                raise_on_nonzero=False,
                raise_on_timeout=False,
            )

    def concatenate_audio_files(self) -> None:
        """
        Description
        -----------
        This method concatenates audio files into a memmap array.

        Parameters
        ----------

        Returns
        -------
        memmap file
            Concatenated wave file (shape: n_samples X n_channels).
        """

        self.message_output(f"Audio concatenation started at: {datetime.now().hour:02d}:{datetime.now().minute:02d}:{datetime.now().second:02d}")
        smart_wait(app_context_bool=self.app_context_bool, seconds=1)

        for audio_file_type in self.input_parameter_dict['concatenate_audio_files']['concat_dirs']:

            audio_type_dir = pathlib.Path(self.root_directory) / 'audio' / audio_file_type
            all_audio_files = sorted(audio_type_dir.glob(f"*.{self.input_parameter_dict['concatenate_audio_files']['concatenate_audio_format']}"))

            if len(all_audio_files) > 1:

                data_dict = DataLoader(input_parameter_dict={'wave_data_loc': [str(audio_type_dir)],
                                                             'load_wavefile_data': {'library': 'scipy', 'conditional_arg': []}}).load_wavefile_data()

                first_key = next(iter(data_dict.keys()))
                name_origin = first_key.split('_')[1]
                dim_1 = data_dict[first_key]['wav_data'].shape[0]
                dim_2 = len(data_dict.keys())
                sr = data_dict[first_key]['sampling_rate']
                complete_mm_file_name = str(audio_type_dir / f"{name_origin}_concatenated_audio_{audio_file_type}_{sr}_{dim_1}_{dim_2}_int16.mmap")

                audio_mm_arr = np.memmap(filename=complete_mm_file_name,
                                         dtype='int16',
                                         mode='w+',
                                         shape=(dim_1, dim_2))

                for file_idx, one_file in enumerate(data_dict.keys()):
                    audio_mm_arr[:, file_idx] = data_dict[one_file]['wav_data']

                audio_mm_arr.flush()

            else:
                self.message_output(f"There are <2 audio files per provided directory: '{pathlib.Path(self.root_directory) / 'audio' / 'cropped_to_video'}', "
                                    f"so concatenation impossible.")

    def broadband_filter_audio(self) -> dict:
        """
        Description
        -----------
        Writes the session's BROADBAND audio memmap
        ``audio/broadband_filtered/<id>_concatenated_audio_broadband_filtered_<sr>_<n>_<ch>_int16.mmap``
        straight from the full-band HPSS wavs (``audio/hpss``), with no
        per-channel wavs stored. It is the 2 kHz+ counterpart of the
        ``hpss_filtered`` (30 kHz+) memmap the USV pipeline reads; readers ask
        for it explicitly with ``os_utils.find_audio_mmap(root, 'broadband')``,
        so no USV reader can pick it up by accident.

        Per channel (memmap column = position of the wav in the sorted wav
        names, the same order ``concatenate_audio_files`` uses):
        1. line-noise tones are estimated (``estimate_line_noise``) in the
           ``line_noise_search_bands_hz`` bands (by default around 8 kHz, its
           16 kHz harmonic, the line near 1.65 kHz and the comb of lines ~30 Hz
           apart around 2.09-2.15 kHz), up to ``line_noise_max_tones_per_band``
           per band, and kept only where they stand at least
           ``line_noise_min_height_db`` above the running-median spectral
           floor; the frequencies are estimated per channel because they follow
           each recording device's clock;
        2. each kept tone is subtracted as a sinusoid with a slowly varying
           amplitude and phase (complex demodulation over
           ``line_noise_block_s`` blocks, running median over
           ``line_noise_smoothing_blocks`` blocks, linear interpolation);
        3. the result is high-passed with a linear-phase Kaiser FIR equivalent to
           sox ``sinc -t 1000 2k`` (``filter_freq_bounds`` = ``[0, 2000]`` is
           the band removed, the ``filter_audio_files`` convention; -6 dB at its
           upper edge, the cutoff, stopband below ``cutoff - transition_width_hz / 2``,
           flat above ``cutoff + transition_width_hz / 2``), in float, without
           dither, then rounded to int16 (round half to even) and clipped.

        The work runs in time chunks of ``chunk_s`` seconds across all channels
        (each wav sample is read once; only ~``chunk_s`` plus a few seconds of
        lookahead of every channel is ever held in memory), with the channels of
        a chunk spread over ``n_threads`` threads. The memmap is written to a
        hidden temporary sibling and renamed into place when complete; then
        ``audio/broadband_filtered/line_noise.json`` is written (per channel: the
        tones searched, found and removed, with frequency, height and amplitude;
        the filter specification and its measured response; the source files and
        their sizes; the output-defining settings; the code version). A session
        whose memmap and report already exist and validate
        (``validate_broadband_output``) is skipped, so the step is idempotent and
        a batch can be resumed.

        Parameters
        ----------

        Returns
        -------
        summary (dict)
            ``status`` ('written' or 'skipped'), ``reason``, ``mmap_path``,
            ``runtime_s``, ``n_channels``, ``n_samples``, ``output_bytes`` and
            ``tones_kept`` (total number of tones subtracted over all channels).
        """

        settings = self.input_parameter_dict['broadband_filter_audio']
        root = pathlib.Path(self.root_directory)
        started = time.perf_counter()
        self.message_output(f"Broadband filtering of {root} started at: {datetime.now().hour:02d}:{datetime.now().minute:02d}:{datetime.now().second:02d}")

        wav_paths = broadband_source_files(root, settings)
        if not wav_paths:
            raise FileNotFoundError(f"No '{settings['source_glob']}' files in '{root / 'audio' / settings['source_dir']}'.")
        infos = [sf.info(str(path)) for path in wav_paths]
        sampling_rate = int(infos[0].samplerate)
        n_samples = int(infos[0].frames)
        n_channels = len(wav_paths)
        for path, info in zip(wav_paths, infos, strict=True):
            if int(info.samplerate) != sampling_rate or int(info.frames) != n_samples or int(info.channels) != 1 or info.subtype != 'PCM_16':
                raise ValueError(f"'{path.name}' is {info.channels} ch {info.subtype} at {info.samplerate} Hz with {info.frames} samples; "
                                 f"every source must be 1 ch PCM_16 at {sampling_rate} Hz with {n_samples} samples.")

        output_dir = root / 'audio' / AUDIO_MMAP_BAND_FOLDERS['broadband']
        mmap_name = f"{wav_paths[0].name.split('_')[1]}_concatenated_audio_{AUDIO_MMAP_BAND_FOLDERS['broadband']}_{sampling_rate}_{n_samples}_{n_channels}_int16.mmap"
        mmap_path = output_dir / mmap_name
        output_bytes = n_samples * n_channels * 2

        valid, reason = validate_broadband_output(root, settings)
        if valid:
            self.message_output(f"Broadband memmap of {root} is {reason}; skipping.")
            return {'status': 'skipped', 'reason': reason, 'mmap_path': str(mmap_path), 'runtime_s': round(time.perf_counter() - started, 2),
                    'n_channels': n_channels, 'n_samples': n_samples, 'output_bytes': output_bytes, 'tones_kept': -1}
        self.message_output(f"Broadband memmap of {root} will be (re)written ({reason}).")

        output_dir.mkdir(parents=True, exist_ok=True)
        # temporaries of an interrupted earlier run (atomic_output_path names them '.<final>.tmp-<pid>')
        for stale_temporary in output_dir.glob('.*.tmp-*'):
            stale_temporary.unlink()

        taps, kaiser_beta = design_broadband_highpass(sampling_rate=sampling_rate,
                                                      cutoff_hz=broadband_highpass_cutoff(settings),
                                                      transition_width_hz=settings['transition_width_hz'],
                                                      stopband_attenuation_db=settings['stopband_attenuation_db'])
        block_length = int(round(settings['line_noise_block_s'] * sampling_rate))
        if settings['line_noise_smoothing_blocks'] < 1 or settings['line_noise_smoothing_blocks'] % 2 != 1:
            raise ValueError(f"line_noise_smoothing_blocks must be a positive odd integer, got {settings['line_noise_smoothing_blocks']}.")
        half_window = (settings['line_noise_smoothing_blocks'] - 1) // 2
        chunk_length = max(1, int(round(settings['chunk_s'] * sampling_rate)))

        with concurrent.futures.ThreadPoolExecutor(max_workers=settings['n_threads']) as executor:
            channel_tones = list(executor.map(
                lambda path: estimate_line_noise(wav_path=path,
                                                 search_bands_hz=settings['line_noise_search_bands_hz'],
                                                 min_height_db=settings['line_noise_min_height_db'],
                                                 n_windows=settings['line_noise_estimation_windows'],
                                                 window_s=settings['line_noise_estimation_window_s'],
                                                 max_tones_per_band=settings['line_noise_max_tones_per_band'],
                                                 min_separation_hz=settings['line_noise_min_separation_hz'],
                                                 floor_window_hz=settings['line_noise_floor_window_hz']),
                wav_paths))
            tones_kept = sum(tone['kept'] for tones in channel_tones for tone in tones)
            self.message_output(f"Line-noise tones kept for subtraction: {tones_kept} over {n_channels} channels.")

            streams = [_BroadbandChannelStream(wav_path=path, n_samples=n_samples, sampling_rate=sampling_rate, taps=taps,
                                               tone_frequencies_hz=[tone['frequency_hz'] for tone in tones if tone['kept']],
                                               block_length=block_length, half_window=half_window)
                       for path, tones in zip(wav_paths, channel_tones, strict=True)]
            try:
                with atomic_output_path(mmap_path) as temporary_path:
                    with open(temporary_path, 'wb') as output_file:
                        next_report = 0.1
                        for chunk_start in range(0, n_samples, chunk_length):
                            chunk_end = min(n_samples, chunk_start + chunk_length)
                            columns = list(executor.map(lambda stream, s=chunk_start, e=chunk_end: stream.process(s, e), streams))
                            output_file.write(np.ascontiguousarray(np.stack(columns, axis=1)).tobytes())
                            if chunk_end / n_samples >= next_report:
                                self.message_output(f"Broadband filtering: {100 * chunk_end / n_samples:.0f}% done ({time.perf_counter() - started:.0f} s).")
                                next_report += 0.1
                        output_file.flush()
                        os.fsync(output_file.fileno())
                    if temporary_path.stat().st_size != output_bytes:
                        raise OSError(f"Broadband memmap has {temporary_path.stat().st_size} bytes, expected {output_bytes}.")
                    # exactly one broadband memmap may exist: drop any older one with another name
                    regex = audio_mmap_name_regex('broadband')
                    for older in output_dir.iterdir():
                        if regex.match(older.name) and older.name != mmap_name:
                            older.unlink()
                channel_reports = []
                for column, (path, tones, stream) in enumerate(zip(wav_paths, channel_tones, streams, strict=True)):
                    kept_index = 0
                    for tone in tones:
                        if tone['kept']:
                            amplitude = 2 * np.abs(stream.smoothed_phasors(kept_index, 0, stream.n_blocks - 1))
                            tone['amplitude_median_lsb'] = round(float(np.median(amplitude)), 4)
                            tone['amplitude_p05_lsb'] = round(float(np.percentile(amplitude, 5)), 4)
                            tone['amplitude_p95_lsb'] = round(float(np.percentile(amplitude, 95)), 4)
                            kept_index += 1
                    channel_reports.append({'column': column, 'file': path.name, 'tones': tones})
            finally:
                for stream in streams:
                    stream.close()

        runtime_s = round(time.perf_counter() - started, 2)
        try:
            code_version = metadata.version('usv-playpen')
        except metadata.PackageNotFoundError:
            code_version = 'unknown'
        report = {
            'complete': True,
            'code_version': code_version,
            'created': datetime.now().isoformat(timespec='seconds'),
            'runtime_s': runtime_s,
            'output': {'file': mmap_name, 'folder': f"audio/{AUDIO_MMAP_BAND_FOLDERS['broadband']}", 'dtype': 'int16',
                       'sampling_rate': sampling_rate, 'n_samples': n_samples, 'n_channels': n_channels, 'bytes': output_bytes,
                       'column_order': 'sorted source wav names'},
            'sources': [{'column': column, 'file': path.name, 'bytes': path.stat().st_size} for column, path in enumerate(wav_paths)],
            'filter': {'type': 'linear-phase Kaiser-windowed sinc FIR high-pass (scipy.signal.kaiserord + firwin), applied centred (zero delay)',
                       'equivalent_of': f"sox sinc -t {settings['transition_width_hz']:g} {broadband_highpass_cutoff(settings):g}",
                       'cutoff_hz': broadband_highpass_cutoff(settings),
                       'transition_width_hz': settings['transition_width_hz'],
                       'stopband_attenuation_db': settings['stopband_attenuation_db'],
                       'numtaps': int(taps.shape[0]),
                       'kaiser_beta': round(kaiser_beta, 4),
                       'edges': 'zero padding beyond the recording',
                       'dither': 'none',
                       'rounding': 'round half to even, clipped to int16',
                       'response_db': highpass_response_db(taps, sampling_rate, BROADBAND_RESPONSE_PROBES_HZ)},
            'line_noise': {'method': ('per channel: tone frequency and height from the averaged zero-padded spectrum of '
                                      f"{settings['line_noise_estimation_windows']} windows of {settings['line_noise_estimation_window_s']} s "
                                      'demodulated at each search-band centre, whitened by its running median over '
                                      f"{settings['line_noise_floor_window_hz']} Hz, up to {settings['line_noise_max_tones_per_band']} peaks per band "
                                      f"at least {settings['line_noise_min_separation_hz']} Hz apart; a kept tone is subtracted as 2 Re(a(n) exp(i w n)) with a(n) the "
                                      f"{settings['line_noise_block_s']} s block complex demodulation, running median over "
                                      f"{settings['line_noise_smoothing_blocks']} blocks, linearly interpolated; amplitudes in int16 LSB (peak)"),
                           'min_height_db': settings['line_noise_min_height_db'],
                           'channels': channel_reports},
            'settings': broadband_output_settings(settings),
        }
        with atomic_output_path(output_dir / BROADBAND_REPORT_NAME) as temporary_report:
            with open(temporary_report, 'w', encoding='utf-8') as report_file:
                json.dump(report, report_file, indent=2)

        self.message_output(f"Broadband filtering of {root} finished in {runtime_s:.0f} s: {mmap_path}")
        return {'status': 'written', 'reason': reason, 'mmap_path': str(mmap_path), 'runtime_s': runtime_s,
                'n_channels': n_channels, 'n_samples': n_samples, 'output_bytes': output_bytes, 'tones_kept': int(tones_kept)}

    def concatenate_video_files(self) -> None:
        """
        Description
        -----------
        This method concatenates video files via ffmpeg.

        Parameters
        ----------

        Returns
        -------
        concatenated_temp (.mp4 file)
            Concatenated video file.
        """

        self.message_output(f"Video concatenation started at: {datetime.now().hour:02d}:{datetime.now().minute:02d}:{datetime.now().second:02d}")
        smart_wait(app_context_bool=self.app_context_bool, seconds=2)

        subprocesses = []

        for sub_directory in (pathlib.Path(self.root_directory) / 'video').iterdir():
            if 'calibration' not in sub_directory.name \
                    and sub_directory.name.split('.')[-1] in self.input_parameter_dict['concatenate_video_files']['concatenate_camera_serial_num']:

                current_working_dir = sub_directory

                vid_name = f"{self.input_parameter_dict['concatenate_video_files']['concatenated_video_name']}_{sub_directory.name.split('.')[-1]}"
                vid_extension = self.input_parameter_dict['concatenate_video_files']['concatenate_video_extension']
                all_video_files = sorted(current_working_dir.glob(f'*.{vid_extension}'))

                if len(all_video_files) > 1:

                    # create .txt file with video files to concatenate
                    with open(current_working_dir / f"file_concatenation_list_{sub_directory.name.split('.')[-1]}.txt", 'w', encoding="utf-8") as concat_txt_file:
                        for file_path in all_video_files:
                            concat_txt_file.write(f"file '{file_path.name}'\n")

                    # concatenate videos
                    one_subprocess = subprocess.Popen(args=["ffmpeg", "-loglevel", "warning", "-f", "concat", "-i", f"file_concatenation_list_{sub_directory.name.split('.')[-1]}.txt", "-c", "copy", f"{vid_name}.{vid_extension}"],
                                                      stdout=subprocess.DEVNULL,
                                                      stderr=subprocess.STDOUT,
                                                      cwd=current_working_dir,
                                                      shell=False)

                    subprocesses.append(one_subprocess)

        wait_for_subprocesses(
            subps=subprocesses,
            max_seconds=3 * 60 * 60,
            label="video concatenation",
            poll_interval_s=5,
            message_output=self.message_output,
            raise_on_nonzero=False,
            raise_on_timeout=False,
        )

        #  copy files over to video directory
        for sub_directory in (pathlib.Path(self.root_directory) / 'video').iterdir():
            if 'calibration' not in sub_directory.name \
                    and sub_directory.name.split('.')[-1] in self.input_parameter_dict['concatenate_video_files']['concatenate_camera_serial_num']:
                current_working_dir = sub_directory
                cam_serial = sub_directory.name.split('.')[-1]
                concat_name = self.input_parameter_dict['concatenate_video_files']['concatenated_video_name']
                vid_ext = self.input_parameter_dict['concatenate_video_files']['concatenate_video_extension']

                (current_working_dir / f"file_concatenation_list_{cam_serial}.txt").unlink()
                shutil.move(src=current_working_dir / f"{concat_name}_{cam_serial}.{vid_ext}",
                            dst=pathlib.Path(self.root_directory) / 'video' / f"{concat_name}_{cam_serial}.{vid_ext}")

    def rectify_video_fps(self, conduct_concat: bool = True) -> None:
        """
        Description
        -----------
        This method changes video sampling rate via ffmpeg.

        Parameters
        ----------
        conduct_concat (bool)
            If True, concatenation was conducted prior to running this.

        Returns
        -------
        fps_corrected_video (.mp4 file)
            FPS modified video file.
        camera_frame_count_dict (.json file)
            Dictionary with camera frame counts,
            empirical capture rates and total video time.
        """

        self.message_output(f"Video re-encoding started at: {datetime.now().hour:02d}:{datetime.now().minute:02d}:{datetime.now().second:02d}")
        smart_wait(app_context_bool=self.app_context_bool, seconds=2)

        video_dir = pathlib.Path(self.root_directory) / 'video'
        non_hidden_files = [p.name for p in video_dir.iterdir() if p.is_file() and not p.name.startswith('.')]

        if not conduct_concat and len(non_hidden_files) == 0:
            for sub_directory in video_dir.iterdir():
                if 'calibration' not in sub_directory.name \
                        and sub_directory.name.split('.')[-1] in self.input_parameter_dict['rectify_video_fps']['encode_camera_serial_num']:
                    cam_serial = sub_directory.name.split('.')[-1]
                    target_file = self.input_parameter_dict['rectify_video_fps']['conversion_target_file']
                    vid_ext = self.input_parameter_dict['rectify_video_fps']['encode_video_extension']

                    shutil.copy(src=sub_directory / f"{target_file}.{vid_ext}",
                                dst=video_dir / f"{target_file}_{cam_serial}.{vid_ext}")

        # load metadata
        metadata, metadata_path = load_session_metadata(
            root_directory=self.root_directory,
            logger=self.message_output
        )

        date_joint = ''
        total_frame_number = None
        total_video_time = None
        camera_frame_count_dict = {}
        empirical_camera_sr = np.zeros(len(self.input_parameter_dict['rectify_video_fps']['encode_camera_serial_num']))
        empirical_camera_sr[:] = np.nan
        camera_idx = 0

        fsp_subprocesses = []
        for sub_directory in sorted(video_dir.iterdir()):
            if (sub_directory.is_dir()
                    and '.' in sub_directory.name
                    and sub_directory.name.split('.')[-1] in self.input_parameter_dict['rectify_video_fps']['encode_camera_serial_num']):

                cam_serial = sub_directory.name.split('.')[-1]
                if camera_idx == 0:
                    date_joint = sub_directory.name.split('.')[0].split('_')[-2] + sub_directory.name.split('.')[0].split('_')[-1]

                # get frame count and empirical sampling rate; the store is closed as soon as
                # its values are read so it never keeps the backing video chunk open -- a
                # lingering handle blocks any later move/delete of that file on Windows
                img_store = new_for_filename(str(sub_directory / 'metadata.yaml'))
                try:
                    total_frame_num = img_store.frame_count
                    last_frame_num = img_store.frame_max
                    frame_times = img_store.get_frame_metadata()['frame_time']
                finally:
                    img_store.close()
                video_duration = frame_times[-1] - frame_times[0]
                esr = round(number=total_frame_num / video_duration, ndigits=4)
                if 'calibration' not in sub_directory.name:
                    empirical_camera_sr[camera_idx] = esr
                    camera_frame_count_dict[cam_serial] = (total_frame_num, esr)
                    if total_frame_num == last_frame_num:
                        self.message_output(f"Camera {cam_serial} has {total_frame_num} total frames, no dropped frames, "
                                            f"video duration of {video_duration:.4f} seconds, and sampling rate of {esr} fps.")
                        if total_frame_number is None or total_frame_num < total_frame_number:
                            total_frame_number = total_frame_num
                        if total_video_time is None or video_duration < total_video_time:
                            total_video_time = video_duration

                        if metadata is not None:
                            metadata['Session']['session_duration'] = round(video_duration, 3)
                            save_session_metadata(data=metadata, filepath=metadata_path, logger=self.message_output)
                    else:
                        self.message_output(f"WARNING: The last frame on camera {cam_serial} is {last_frame_num}, which is more than {total_frame_num} in total, "
                                            f"suggesting dropped frames. The video duration is {video_duration:.4f} seconds")
                    camera_idx += 1

                crf = self.input_parameter_dict['rectify_video_fps']['constant_rate_factor']
                enc_preset = self.input_parameter_dict['rectify_video_fps']['encoding_preset']
                vid_ext = self.input_parameter_dict['rectify_video_fps']['encode_video_extension']
                conv_target = self.input_parameter_dict['rectify_video_fps']['conversion_target_file']

                current_working_dir = video_dir
                if 'calibration' not in sub_directory.name:
                    target_file = f"{conv_target}_{cam_serial}.{vid_ext}"
                    new_file = f"{cam_serial}-{date_joint}.{vid_ext}"
                else:
                    current_working_dir = sub_directory
                    target_file = f"000000.{vid_ext}"
                    new_file = f"{cam_serial}-{date_joint}-calibration.{vid_ext}"

                # change video sampling rate
                fps_subp = subprocess.Popen(args=["ffmpeg", "-loglevel", "warning", "-y", "-r", str(esr), "-i", target_file, "-fps_mode", "passthrough", "-crf", str(crf), "-preset", enc_preset, new_file],
                                            stdout=subprocess.DEVNULL,
                                            stderr=subprocess.STDOUT,
                                            cwd=current_working_dir,
                                            shell=False)

                fsp_subprocesses.append(fps_subp)

        wait_for_subprocesses(
            subps=fsp_subprocesses,
            max_seconds=6 * 60 * 60,
            label="video re-encoding (fps/crf)",
            poll_interval_s=5,
            message_output=self.message_output,
            raise_on_nonzero=False,
            raise_on_timeout=False,
        )

        # move files to special directory
        for sub_directory in sorted(video_dir.iterdir()):
            if (sub_directory.is_dir()
                    and '.' in sub_directory.name
                    and sub_directory.name.split('.')[-1] in self.input_parameter_dict['rectify_video_fps']['encode_camera_serial_num']):

                cam_serial = sub_directory.name.split('.')[-1]
                vid_ext = self.input_parameter_dict['rectify_video_fps']['encode_video_extension']
                conv_target = self.input_parameter_dict['rectify_video_fps']['conversion_target_file']
                dest_base = video_dir / date_joint / cam_serial
                (dest_base / 'calibration_images').mkdir(parents=True, exist_ok=True)

                current_working_dir = video_dir
                if 'calibration' not in sub_directory.name:
                    target_file = f"{conv_target}_{cam_serial}.{vid_ext}"
                    new_file = f"{cam_serial}-{date_joint}.{vid_ext}"

                    shutil.move(src=current_working_dir / new_file,
                                dst=dest_base / new_file)
                else:
                    current_working_dir = sub_directory
                    target_file = f"000000.{vid_ext}"
                    new_file = f"{cam_serial}-{date_joint}-calibration.{vid_ext}"

                    shutil.move(src=current_working_dir / new_file,
                                dst=dest_base / 'calibration_images' / new_file)

                # Clean up the disposable intermediate ONLY.
                #
                # In a calibration sub-directory `target_file` is '000000.<ext>' -- the RAW
                # loopbio recording, not a throwaway artefact -- and the re-encoded copy has
                # already been moved into 'calibration_images'. Deleting it would irreversibly
                # destroy original footage, so calibration directories are never cleaned.
                # Non-calibration directories delete '<conv_target>_<serial>.<ext>', which is
                # the temporary concatenation product and safe to remove.
                if (self.input_parameter_dict['rectify_video_fps']['delete_old_file']
                        and 'calibration' not in sub_directory.name
                        and (current_working_dir / target_file).is_file()):
                    (current_working_dir / target_file).unlink()

        # save camera_frame_count_dict to a file
        # If no camera produced a clean (no-dropped-frames) recording, there is
        # no trustworthy "least" frame count / duration. Downstream stages
        # (anipose_operations, synchronize_files) read these strictly as numbers
        # via int()/arithmetic, so writing None would crash them. Warn loudly
        # and fall back to the historical numeric sentinel so the failure is
        # visible instead of silently propagating a fabricated count.
        if total_frame_number is None:
            self.message_output("WARNING: no camera produced a clean (no-dropped-frames) recording for this session; "
                                "total_frame_number_least/total_video_time_least fall back to a sentinel and should "
                                "not be trusted downstream.")
            total_frame_number = int(1e9)
            total_video_time = 1e9
        camera_frame_count_dict['total_frame_number_least'] = total_frame_number
        camera_frame_count_dict['total_video_time_least'] = total_video_time
        # Use nanmedian so a single missing/failed camera (whose pre-allocated
        # slot stays NaN) does not collapse the whole session's median to NaN
        # and poison the frame rate that every downstream stage
        # (assign_vocalizations, anipose_operations, synchronize_files) reads.
        # Only the pathological all-cameras-missing case stays NaN, which we
        # surface explicitly rather than silently writing a misleading number.
        if np.all(np.isnan(empirical_camera_sr)):
            self.message_output("WARNING: no empirical camera sampling rates were recorded for this session; "
                                "median_empirical_camera_sr written as NaN.")
            camera_frame_count_dict['median_empirical_camera_sr'] = float('nan')
        else:
            camera_frame_count_dict['median_empirical_camera_sr'] = round(number=np.nanmedian(empirical_camera_sr), ndigits=4)
        with open(video_dir / f'{date_joint}_camera_frame_count_dict.json', 'w') as frame_count_outfile:
            json.dump(camera_frame_count_dict, frame_count_outfile, indent=4)


def read_broadband_session_list(sessions_file: str | None = None,
                                usv_counts_csv: str | None = None,
                                tag: str = 'ok') -> list[str]:
    """
    Description
    -----------
    Builds the list of session roots for a broadband backfill, from either a
    plain sessions file (one session root per line; blank lines and lines
    starting with '#' are ignored) or a session table with ``dir`` and ``tag``
    columns (e.g. ``noise_labelling/session_usv_counts.csv``), keeping the rows
    whose ``tag`` equals ``tag``. Paths are passed through ``configure_path`` so
    the same list works on every OS. Order is preserved; duplicates are dropped.

    Parameters
    ----------
    sessions_file (str | None)
        Plain sessions file.
    usv_counts_csv (str | None)
        Session table with ``dir`` and ``tag`` columns.
    tag (str)
        The ``tag`` value to keep from ``usv_counts_csv``.

    Returns
    -------
    session_roots (list of str)
        Session root directories.
    """

    if (sessions_file is None) == (usv_counts_csv is None):
        raise ValueError("Give exactly one of sessions_file and usv_counts_csv.")
    if sessions_file is not None:
        with open(sessions_file, encoding='utf-8') as session_list:
            entries = [line.strip() for line in session_list if line.strip() and not line.strip().startswith('#')]
    else:
        table = pls.read_csv(usv_counts_csv)
        entries = table.filter(pls.col('tag') == tag)['dir'].to_list()
    session_roots = []
    for entry in entries:
        session_root = configure_path(entry)
        if session_root not in session_roots:
            session_roots.append(session_root)
    return session_roots


def _broadband_filter_session_worker(session_root: str, processing_settings: dict, log_path: str) -> dict:
    """
    Description
    -----------
    Runs ``Operator.broadband_filter_audio`` on one session inside a batch
    worker process, appending its progress messages (timestamped, prefixed with
    the session name) to the shared batch log, and turns any failure into a
    ``failed`` result row instead of an exception so one bad session does not
    stop the batch.

    Parameters
    ----------
    session_root (str)
        Session root directory.
    processing_settings (dict)
        Full processing settings (``modify_files`` and ``synchronize_files``
        blocks are read by ``Operator``).
    log_path (str)
        Batch log file (appended to).

    Returns
    -------
    row (dict)
        One report row (keys ``BROADBAND_BATCH_REPORT_COLUMNS``).
    """

    session_name = pathlib.Path(session_root).name

    def log_message(message: str) -> None:
        """
        Description
        -----------
        Appends one timestamped line to the batch log.

        Parameters
        ----------
        message (str)
            Message text.

        Returns
        -------
        None
        """

        with open(log_path, 'a', encoding='utf-8') as log_file:
            log_file.write(f"{datetime.now().isoformat(timespec='seconds')} [{session_name}] {message}\n")

    started = time.perf_counter()
    try:
        summary = Operator(root_directory=session_root, input_parameter_dict=processing_settings,
                           message_output=log_message).broadband_filter_audio()
        row = {'session_root': session_root, 'status': summary['status'], 'runtime_s': summary['runtime_s'],
               'n_channels': summary['n_channels'], 'n_samples': summary['n_samples'], 'output_bytes': summary['output_bytes'],
               'tones_kept': summary['tones_kept'], 'error': ''}
    except Exception as session_error:  # batch robustness: record the failure and continue with the next session
        log_message(f"FAILED: {type(session_error).__name__}: {session_error}")
        row = {'session_root': session_root, 'status': 'failed', 'runtime_s': round(time.perf_counter() - started, 2),
               'n_channels': '', 'n_samples': '', 'output_bytes': '', 'tones_kept': '', 'error': f"{type(session_error).__name__}: {session_error}"}
    row['finished_at'] = datetime.now().isoformat(timespec='seconds')
    return row


def broadband_filter_sessions(session_roots: list[str],
                              processing_settings: dict,
                              n_workers: int,
                              log_path: str,
                              report_csv_path: str,
                              message_output: Callable | None = None) -> list[dict]:
    """
    Description
    -----------
    Batch runner for the broadband backfill: runs
    ``Operator.broadband_filter_audio`` on many sessions, ``n_workers`` sessions
    at a time in separate processes (each session additionally spreads its
    channels over the ``n_threads`` threads of its settings block). It is
    resumable: sessions whose broadband memmap and report already validate are
    skipped by the step itself (status ``skipped``), and an interrupted session
    leaves only a hidden temporary file that the next run removes. Every
    finished session appends one row to ``report_csv_path`` (columns
    ``BROADBAND_BATCH_REPORT_COLUMNS``; the header is written when the file is
    new) and its messages go to ``log_path``.

    Parameters
    ----------
    session_roots (list of str)
        Session root directories.
    processing_settings (dict)
        Full processing settings (the ``modify_files.Operator.broadband_filter_audio``
        block configures the step).
    n_workers (int)
        Number of sessions processed in parallel.
    log_path (str)
        Batch log file (appended to; created if absent).
    report_csv_path (str)
        Per-session report CSV (appended to; created with a header if absent).
    message_output (Callable | None)
        Progress messages; defaults to print.

    Returns
    -------
    rows (list of dict)
        The report rows of this run, in completion order.
    """

    message_output = message_output if message_output is not None else print
    pathlib.Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(report_csv_path).parent.mkdir(parents=True, exist_ok=True)
    message_output(f"Broadband backfill: {len(session_roots)} sessions, {n_workers} parallel, "
                   f"{processing_settings['modify_files']['Operator']['broadband_filter_audio']['n_threads']} threads each; "
                   f"log '{log_path}', report '{report_csv_path}'.")
    rows = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = {pool.submit(_broadband_filter_session_worker, session_root, processing_settings, log_path): session_root
                   for session_root in session_roots}
        for done_index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            row = future.result()
            write_header = not pathlib.Path(report_csv_path).is_file()
            with open(report_csv_path, 'a', newline='', encoding='utf-8') as report_file:
                writer = csv.DictWriter(report_file, fieldnames=list(BROADBAND_BATCH_REPORT_COLUMNS))
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
            rows.append(row)
            message_output(f"[{done_index}/{len(session_roots)}] {row['session_root']}: {row['status']} ({row['runtime_s']} s){' ' + row['error'] if row['error'] else ''}")
    return rows
