"""
@author: bartulem

Build the squeak spectrogram store: a 2-125 kHz, log-frequency spectrogram of
every squeak that the squeak QLVM map can hold, written once so the embedding
explorer (``notebooks/usv_embedding_explorer.py``) can show squeak thumbnails
without touching session audio while brushing.

The consolidated store (``consolidate-spectrogram-store``) holds the ultrasonic
spectrograms (30-125 kHz, linear frequency, min-max normalized per call) the
USV maps are built from. A squeak is a broadband harmonic stack whose
fundamental sits around 3-8 kHz, so those thumbnails show only its ultrasonic
tail. This store covers the whole band instead, on a log frequency axis that
gives the 2-10 kHz harmonics about as much height as the ultrasonic part.

Why a module of its own: it is a cohort step that writes one multi-session H5
(like :mod:`consolidate_spectrogram_store`), but it reads session AUDIO through
the squeak front end of :mod:`detect_usv_squeaks` (wav channel selection, row
selection, crop frames), not per-session spectrogram H5 files and QLVM package
tables, so neither of those modules is a natural home; adding it to
:mod:`detect_usv_squeaks` (per-session summary writers) would mix a cohort
store into a per-session step.

Rows. Per session, the rows of ``*_usv_summary.csv`` with ``squeak`` true (pure
squeaks and segments holding both a squeak and a USV) and ``noise`` not true
(:func:`detect_usv_squeaks.squeak_qlvm_rows`), the rows
``infer-qlvm-squeak-latents`` considers; every row that carries
``qlvm_squeak1`` / ``qlvm_squeak2`` is among them. The store holds every call
class the squeak map can show, so the explorer's squeak-class filter
(squeak / both / squeak + both) needs no rebuild.

Spectrogram. The audio of a row is its squeak crop window
(:func:`detect_usv_squeaks.squeak_crop_window`: the segment, widened where needed
to hold the squeak envelope ``squeak_start`` .. ``squeak_end`` plus the
embedding's two context frames, because a squeak often runs past the segment the
ultrasonic segmenter cut; the segment alone when the row has no envelope), from
``round(window_start * 250000)`` to ``round(window_stop * 250000)`` samples on every unfiltered HPSS channel
(``audio/hpss/*_cropped_to_video_hpss.wav``), dropping the channels the session
metadata marks as excluded when ``exclude_metadata_audio_channels`` is on
(:func:`detect_usv_squeaks.squeak_wav_channels`, the same selection
``detect-usv-squeaks`` averages over). Each channel is mean-removed and turned
into a power STFT with the squeak classifier's own front end: Blackman-Harris
window, ``nperseg`` 2048 (8.19 ms, 122 Hz bin spacing), hop 512 (2.048 ms per
frame), centred (``detect_usv_squeaks.SQUEAK_SPEC_PARAMS``). That window is
long enough to separate harmonics 3 kHz apart at the low end (main lobe about
1 kHz wide) and short enough to follow a 20-50 ms squeak, and it keeps frame
``t`` centred at ``window_start + t * 0.002048`` s, so the frames line up with
the squeak envelope and with the embedding's crops. The linear
bins are mapped onto ``n_frequency_bins`` log-spaced bins between ``min_freq``
and ``max_freq`` (geometric edges; the power of a log bin is the mean power of
the linear bins whose centres fall inside it, or, where a log bin is narrower
than the 122 Hz linear spacing -- below about 7.6 kHz with 128 bins -- the
power linearly interpolated at its geometric centre; :func:`log_frequency_weights`),
converted to absolute dB (``ref`` 1.0, no ``top_db`` clamp, librosa's
``amin`` 1e-10 puts silence at -100 dB) and averaged across channels with
weights equal to each channel's audio variance (the rule of
``generate_spectrograms.compute_usv_spectrogram``).

Time extent. As in the consolidated store, every call sits in a fixed window
(``window_frames``, 256 frames = 524 ms by default) from column 0 with its
native frame count in ``durations``, so a thumbnail padded to the window shows
the call's true duration. An audio window longer than the display window keeps
the ``window_frames`` frames centred on the squeak's crop (the frames inside its
envelope plus the embedding's two context frames,
:func:`detect_usv_squeaks.squeak_crop_frames`), clipped to the audio window;
``window_first`` records the first kept frame and ``audio_start_s`` the session
time of the audio window's frame 0.

Storage. dB values are clipped to ``[db_floor, db_ceil]`` and quantized to
uint8 (0 = ``db_floor``, 255 = ``db_ceil``; :func:`quantize_db` /
:func:`dequantize_db`), gzip-compressed in one chunk per call so the explorer
reads a call without decompressing its neighbours. Layout:

* ``frequency_bins`` -- ``(F,)`` float64 geometric centres (Hz) of the log bins;
  ``frequency_bin_edges`` -- ``(F + 1,)`` their edges;
* ``spectrogram/<session>/spectrograms`` -- ``(n, F, window_frames)`` uint8;
* ``spectrogram/<session>/row_index`` -- ``(n,)`` int64 0-based summary rows,
  ascending (the explorer's ``row_index``);
* ``spectrogram/<session>/durations`` -- ``(n,)`` int32 frames of the call in
  the window (0 when the segment is shorter than one STFT window);
* ``spectrogram/<session>/n_frames`` -- ``(n,)`` int32 native frames of the
  whole audio window; ``window_first`` -- ``(n,)`` int32 first kept frame;
  ``audio_start_s`` -- ``(n,)`` float64 session time (s) of the audio window's
  frame 0;
* ``sessions`` -- one row per session: id, root directory, summary SHA-256,
  summary rows and stored squeaks.

Root attrs record the settings, the STFT, the quantization range and the
provenance. The store is written to ``spectrograms_root`` as
``squeak_spectrograms_<F>logbins_<S>sessions_<N>squeaks_<UTC timestamp>.h5``;
``os_utils.resolve_squeak_spectrogram_store_path`` picks the newest one (the
name does not match the consolidated store's ``spectrograms_*.h5`` pattern).
Sessions are processed in parallel worker processes (``n_workers``); a session
that fails is reported and left out, never half-written.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import pathlib
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime

import click
import h5py
import librosa
import numpy as np
import polars as pls
import soundfile as sf
from click.core import ParameterSource

from ..analyses.usv_interval_archive import _polars_to_h5, git_sha_for_provenance
from ..cli_utils import modify_settings_json_for_cli
from ..os_utils import atomic_output_path, configure_path, first_match_or_raise
from ..time_utils import is_gui_context, smart_wait
from .build_qlvm_training_set import file_sha256
from .detect_usv_squeaks import (
    FRAME_DT_S,
    SQUEAK_SAMPLING_RATE,
    SQUEAK_SPEC_PARAMS,
    squeak_crop_frames,
    squeak_crop_window,
    squeak_qlvm_rows,
    squeak_wav_channels,
)

# File-name stem (and layout tag) of the stores this module writes.
STORE_NAME_PREFIX = "squeak_spectrograms"

# Highest uint8 code of the dB quantization (0 = db_floor, 255 = db_ceil).
QUANTIZATION_MAX = 255

# Thread-pool environment variables pinned to 1 in the spawned workers: each
# worker's small per-frame matrix products would otherwise start one BLAS / OpenMP
# thread per core in every process, and n_workers processes oversubscribe the CPU
# (measured: 6 workers ran at ~400 % CPU each and about 5x slower per call).
WORKER_THREAD_ENV_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")

# Absolute-dB reference of the stored spectrograms (power relative to 1.0 on
# float audio in [-1, 1], as the squeak classifier's front end).
SQUEAK_STORE_DB_REF = 1.0


def log_frequency_bins(min_freq: float, max_freq: float, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Description
    -----------
    Geometric (log-spaced) frequency bins between ``min_freq`` and ``max_freq``:
    ``n_bins + 1`` edges from ``np.geomspace`` and, per bin, the geometric mean
    of its two edges as its centre.

    Parameters
    ----------
    min_freq (float)
        Lower edge of the first bin in Hz (> 0).
    max_freq (float)
        Upper edge of the last bin in Hz (> ``min_freq``).
    n_bins (int)
        Number of bins (>= 1).

    Returns
    -------
    edges (np.ndarray)
        ``(n_bins + 1,)`` float64 bin edges in Hz.
    centres (np.ndarray)
        ``(n_bins,)`` float64 geometric bin centres in Hz.

    Raises
    ------
    ValueError
        The band or the bin count is invalid.
    """

    if not 0 < min_freq < max_freq or n_bins < 1:
        error_message = f"log_frequency_bins needs 0 < min_freq < max_freq and n_bins >= 1, got {min_freq}, {max_freq}, {n_bins}."
        raise ValueError(error_message)
    edges = np.geomspace(float(min_freq), float(max_freq), int(n_bins) + 1)
    centres = np.sqrt(edges[:-1] * edges[1:])
    return edges, centres


def log_frequency_weights(linear_freqs: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """
    Description
    -----------
    The linear map from linear STFT bins to log-spaced bins, applied to POWER.
    A log bin whose ``[lower edge, upper edge)`` interval (the last one closed on
    the right) contains the centres of one or more linear bins averages their
    power with equal weights. A log bin narrower than the linear spacing that
    contains no linear centre takes the power linearly interpolated between the
    two linear bins around its geometric centre. Every row sums to 1, so a flat
    spectrum stays flat.

    Parameters
    ----------
    linear_freqs (np.ndarray)
        ``(K,)`` ascending centre frequencies (Hz) of the linear STFT bins
        (``librosa.fft_frequencies``).
    edges (np.ndarray)
        ``(F + 1,)`` ascending log-bin edges (Hz) inside
        ``[linear_freqs[0], linear_freqs[-1]]``.

    Returns
    -------
    weights (np.ndarray)
        ``(F, K)`` float64 weights; ``weights @ power`` maps a ``(K, T)`` power
        spectrogram to ``(F, T)``.

    Raises
    ------
    ValueError
        The edges fall outside the linear frequency range.
    """

    linear_freqs = np.asarray(linear_freqs, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.float64)
    if edges[0] < linear_freqs[0] or edges[-1] > linear_freqs[-1]:
        error_message = (
            f"log_frequency_weights: the log bins span {edges[0]:.1f}-{edges[-1]:.1f} Hz, outside the linear "
            f"STFT range {linear_freqs[0]:.1f}-{linear_freqs[-1]:.1f} Hz."
        )
        raise ValueError(error_message)
    n_bins = edges.size - 1
    weights = np.zeros((n_bins, linear_freqs.size), dtype=np.float64)
    for bin_index in range(n_bins):
        lower, upper = edges[bin_index], edges[bin_index + 1]
        if bin_index == n_bins - 1:
            inside = np.flatnonzero((linear_freqs >= lower) & (linear_freqs <= upper))
        else:
            inside = np.flatnonzero((linear_freqs >= lower) & (linear_freqs < upper))
        if inside.size:
            weights[bin_index, inside] = 1.0 / inside.size
            continue
        centre = np.sqrt(lower * upper)
        right = int(np.searchsorted(linear_freqs, centre, side="right"))
        left = right - 1
        fraction = (centre - linear_freqs[left]) / (linear_freqs[right] - linear_freqs[left])
        weights[bin_index, left] = 1.0 - fraction
        weights[bin_index, right] = fraction
    return weights


def log_frequency_spectrogram(
    audio_segment_channels: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray | None, int]:
    """
    Description
    -----------
    The variance-weighted, multi-channel average log-frequency spectrogram of
    one audio segment, in absolute dB. Per channel: mean removal, the squeak
    classifier's power STFT (``SQUEAK_SPEC_PARAMS``: Blackman-Harris,
    ``nperseg`` 2048, hop 512, centred), the log-bin map ``weights`` on power
    (:func:`log_frequency_weights`), and ``librosa.power_to_db`` with ``ref``
    1.0 and no ``top_db`` clamp. The channel spectrograms are averaged with
    weights equal to each channel's audio variance (uniform when every channel
    is silent), as ``generate_spectrograms.compute_usv_spectrogram`` does.
    Channels shorter than one STFT window are skipped.

    Parameters
    ----------
    audio_segment_channels (np.ndarray)
        ``(n_samples, n_channels)`` float audio of the segment (float64 from
        soundfile, int16 / 32768).
    weights (np.ndarray)
        ``(F, 1 + nperseg // 2)`` log-bin weights.

    Returns
    -------
    spectrogram_db (np.ndarray | None)
        ``(F, n_frames)`` float32 absolute-dB spectrogram, or None when no
        channel is long enough for one STFT frame.
    n_frames (int)
        Native STFT frame count (0 when there is no spectrogram).
    """

    nperseg = SQUEAK_SPEC_PARAMS["nperseg"]
    per_channel_specs: list[np.ndarray] = []
    per_channel_vars: list[float] = []
    for channel_index in range(audio_segment_channels.shape[1]):
        audio_segment = audio_segment_channels[:, channel_index].astype(np.float64)
        if audio_segment.shape[0] < nperseg:
            continue
        audio_segment = audio_segment - np.mean(audio_segment)
        power_spec = np.abs(
            librosa.stft(
                audio_segment,
                n_fft=nperseg,
                hop_length=SQUEAK_SPEC_PARAMS["hop_length"],
                win_length=nperseg,
                window=SQUEAK_SPEC_PARAMS["window"],
                center=True,
            )
        ) ** 2
        per_channel_specs.append(librosa.power_to_db(weights @ power_spec, ref=SQUEAK_STORE_DB_REF, top_db=None))
        per_channel_vars.append(float(np.var(audio_segment)))
    if not per_channel_specs:
        return None, 0
    channel_weights = np.asarray(per_channel_vars, dtype=np.float64)
    if channel_weights.sum() == 0:
        channel_weights = np.ones_like(channel_weights)
    average = np.average(np.asarray(per_channel_specs), axis=0, weights=channel_weights)
    return average.astype(np.float32), int(average.shape[1])


def display_window_first(
    n_frames: int,
    crop_first: int | None,
    crop_last: int | None,
    window_frames: int,
) -> int:
    """
    Description
    -----------
    The first frame of the stored window of one segment. A segment that fits
    in the window starts at frame 0. A longer one keeps the ``window_frames``
    frames centred on its squeak crop (``(crop_first + crop_last) / 2``),
    clipped so the window stays inside the segment; without a crop (no squeak
    extent) it keeps the first ``window_frames`` frames.

    Parameters
    ----------
    n_frames (int)
        Native frame count of the segment.
    crop_first (int | None)
        First frame of the squeak crop, or None.
    crop_last (int | None)
        Last frame of the squeak crop (inclusive), or None.
    window_frames (int)
        Width of the stored window in frames.

    Returns
    -------
    window_first (int)
        First kept frame, in ``0 .. max(n_frames - window_frames, 0)``.
    """

    if n_frames <= window_frames or crop_first is None or crop_last is None:
        return 0
    centre = 0.5 * (crop_first + crop_last)
    return int(np.clip(round(centre - 0.5 * window_frames), 0, n_frames - window_frames))


def quantize_db(spectrogram_db: np.ndarray, db_floor: float, db_ceil: float) -> np.ndarray:
    """
    Description
    -----------
    Clips absolute dB to ``[db_floor, db_ceil]`` and maps it linearly onto the
    uint8 codes ``0 .. 255`` (rounded), so code 0 is ``db_floor`` (and the
    zero padding of the stored window reads as the floor).

    Parameters
    ----------
    spectrogram_db (np.ndarray)
        Absolute-dB values of any shape.
    db_floor (float)
        dB value of code 0.
    db_ceil (float)
        dB value of code 255.

    Returns
    -------
    codes (np.ndarray)
        uint8 array of the same shape.
    """

    scaled = (np.clip(spectrogram_db, db_floor, db_ceil) - db_floor) / (db_ceil - db_floor)
    return np.round(scaled * QUANTIZATION_MAX).astype(np.uint8)


def dequantize_db(codes: np.ndarray, db_floor: float, db_ceil: float) -> np.ndarray:
    """
    Description
    -----------
    Inverse of :func:`quantize_db`: uint8 codes back to dB, with a rounding
    error of at most half a step, ``(db_ceil - db_floor) / 510`` dB.

    Parameters
    ----------
    codes (np.ndarray)
        uint8 codes of any shape.
    db_floor (float)
        dB value of code 0.
    db_ceil (float)
        dB value of code 255.

    Returns
    -------
    spectrogram_db (np.ndarray)
        float32 dB values of the same shape.
    """

    return (db_floor + codes.astype(np.float32) * np.float32((db_ceil - db_floor) / QUANTIZATION_MAX)).astype(np.float32)


def squeak_store_thumbnail(
    session_group: h5py.Group,
    row_index: int,
    db_floor: float,
    db_ceil: float,
    dynamic_range_db: float,
) -> np.ndarray | None:
    """
    Description
    -----------
    One call's display tile from a session group of the squeak store: its
    stored frames (``spectrograms[position, :, :durations[position]]``),
    dequantized to dB and mapped to ``[0, 1]`` over the top
    ``dynamic_range_db`` dB of the call (``(x - (max - range)) / range``,
    clipped), so the harmonic stack and the ultrasonic part share one contrast
    and every call is shown relative to its own peak, as the min-max-normalized
    USV thumbnails are. The row is found by binary search on the ascending
    ``row_index`` dataset.

    Parameters
    ----------
    session_group (h5py.Group)
        ``spectrogram/<session>`` of the squeak store.
    row_index (int)
        0-based USV summary row.
    db_floor (float)
        The store's ``db_floor`` attr.
    db_ceil (float)
        The store's ``db_ceil`` attr.
    dynamic_range_db (float)
        dB range below the call's peak mapped onto ``[0, 1]``.

    Returns
    -------
    tile (np.ndarray | None)
        ``(F, durations[position])`` float32 tile in ``[0, 1]``, or None when
        the row is not in the store or has no spectrogram.
    """

    rows = session_group["row_index"][:]
    position = int(np.searchsorted(rows, row_index))
    if position >= rows.size or int(rows[position]) != int(row_index):
        return None
    duration = int(session_group["durations"][position])
    if duration <= 0:
        return None
    tile_db = dequantize_db(session_group["spectrograms"][position, :, :duration], db_floor, db_ceil)
    low = float(tile_db.max()) - float(dynamic_range_db)
    return np.clip((tile_db - low) / float(dynamic_range_db), 0.0, 1.0).astype(np.float32)


def read_session_lists(session_list_paths: list[str]) -> list[str]:
    """
    Description
    -----------
    The session roots of one or more session-list text files (one root per
    line; blank lines and lines starting with ``#`` skipped), each run through
    ``configure_path``, de-duplicated in first-seen order.

    Parameters
    ----------
    session_list_paths (list[str])
        Paths of the ``*.txt`` session lists.

    Returns
    -------
    root_directories (list[str])
        Session root directories.

    Raises
    ------
    FileNotFoundError
        A list file does not exist.
    """

    seen: set[str] = set()
    root_directories: list[str] = []
    for list_path in session_list_paths:
        resolved_list = pathlib.Path(configure_path(str(list_path)))
        if not resolved_list.is_file():
            error_message = f"Session list {resolved_list} does not exist."
            raise FileNotFoundError(error_message)
        for line in resolved_list.read_text().splitlines():
            session_root = line.strip()
            if not session_root or session_root.startswith("#"):
                continue
            resolved_root = configure_path(session_root)
            if resolved_root not in seen:
                seen.add(resolved_root)
                root_directories.append(resolved_root)
    return root_directories


def session_squeak_spectrograms(session_root: str, cfg: dict) -> dict:
    """
    Description
    -----------
    Builds the store entries of one session (runs inside a worker process): the
    ``squeak`` / ``both`` rows that are not noise
    (:func:`detect_usv_squeaks.squeak_qlvm_rows`), each rebuilt from the session's
    HPSS wavs over its squeak crop window
    (:func:`detect_usv_squeaks.squeak_crop_window`) as a log-frequency
    absolute-dB spectrogram (:func:`log_frequency_spectrogram`), cut to the
    display window (:func:`display_window_first`) and quantized
    (:func:`quantize_db`).

    Parameters
    ----------
    session_root (str)
        Session root directory.
    cfg (dict)
        The ``build_squeak_spectrogram_store`` settings block.

    Returns
    -------
    entry (dict)
        ``session_id``, ``root_directory``, ``usv_summary_sha256``,
        ``n_summary_rows``, ``row_index`` (``(n,)`` int64), ``spectrograms``
        (``(n, F, window_frames)`` uint8), ``durations`` / ``n_frames`` /
        ``window_first`` (``(n,)`` int32), ``audio_start_s`` (``(n,)`` float64)
        and ``messages`` (log lines).
    """

    root = pathlib.Path(session_root)
    messages: list[str] = []
    usv_summary_loc = first_match_or_raise(
        root=root / "audio",
        pattern="*_usv_summary.csv",
        recursive=True,
        label="USV summary CSV",
    )
    usv_summary = pls.read_csv(source=str(usv_summary_loc), schema_overrides={"usv_id": pls.String})
    rows = squeak_qlvm_rows(usv_summary)
    window_frames = int(cfg['window_frames'])
    edges, _ = log_frequency_bins(cfg['min_freq'], cfg['max_freq'], cfg['n_frequency_bins'])
    weights = log_frequency_weights(
        librosa.fft_frequencies(sr=SQUEAK_SAMPLING_RATE, n_fft=SQUEAK_SPEC_PARAMS["nperseg"]), edges
    )

    spectrograms = np.zeros((rows.size, int(cfg['n_frequency_bins']), window_frames), dtype=np.uint8)
    durations = np.zeros(rows.size, dtype=np.int32)
    n_frames_all = np.zeros(rows.size, dtype=np.int32)
    window_first_all = np.zeros(rows.size, dtype=np.int32)
    audio_start_all = np.zeros(rows.size, dtype=np.float64)
    if rows.size:
        wav_paths = squeak_wav_channels(root, cfg['exclude_metadata_audio_channels'], messages.append)
        starts = usv_summary["start"].to_numpy()
        stops = usv_summary["stop"].to_numpy()
        squeak_start = usv_summary["squeak_start"].cast(pls.Float64).fill_null(np.nan).to_numpy()
        squeak_end = usv_summary["squeak_end"].cast(pls.Float64).fill_null(np.nan).to_numpy()
        handles = [sf.SoundFile(str(wav_path), mode="r") for wav_path in wav_paths]
        try:
            for position, row_index in enumerate(rows):
                has_extent = bool(np.isfinite(squeak_start[row_index]) and np.isfinite(squeak_end[row_index]))
                audio_start, audio_stop = float(starts[row_index]), float(stops[row_index])
                if has_extent:
                    window_start, window_stop = squeak_crop_window(
                        np.array([starts[row_index]]), np.array([stops[row_index]]),
                        np.array([squeak_start[row_index]]), np.array([squeak_end[row_index]]),
                    )
                    audio_start, audio_stop = float(window_start[0]), float(window_stop[0])
                first_sample = round(audio_start * SQUEAK_SAMPLING_RATE)
                last_sample = round(audio_stop * SQUEAK_SAMPLING_RATE)
                channel_audio = []
                for handle in handles:
                    handle.seek(first_sample)
                    channel_audio.append(handle.read(frames=max(0, last_sample - first_sample), dtype="float64", always_2d=False))
                spectrogram_db, n_frames = log_frequency_spectrogram(np.stack(channel_audio, axis=1), weights)
                if spectrogram_db is None:
                    continue
                crop_first, crop_last = None, None
                if has_extent:
                    first, last = squeak_crop_frames(
                        window_start_s=np.array([audio_start]),
                        squeak_start_s=np.array([squeak_start[row_index]]),
                        squeak_end_s=np.array([squeak_end[row_index]]),
                        n_frames=np.array([n_frames]),
                    )
                    crop_first, crop_last = int(first[0]), int(last[0])
                window_first = display_window_first(n_frames, crop_first, crop_last, window_frames)
                kept = spectrogram_db[:, window_first:window_first + window_frames]
                spectrograms[position, :, :kept.shape[1]] = quantize_db(kept, cfg['db_floor'], cfg['db_ceil'])
                durations[position] = kept.shape[1]
                n_frames_all[position] = n_frames
                window_first_all[position] = window_first
                audio_start_all[position] = audio_start
        finally:
            for handle in handles:
                handle.close()
    return {
        "session_id": root.name,
        "root_directory": str(root),
        "usv_summary_sha256": file_sha256(usv_summary_loc),
        "n_summary_rows": int(usv_summary.height),
        "row_index": rows.astype(np.int64),
        "spectrograms": spectrograms,
        "durations": durations,
        "n_frames": n_frames_all,
        "window_first": window_first_all,
        "audio_start_s": audio_start_all,
        "messages": messages,
    }


def _session_worker(session_root: str, cfg: dict) -> dict:
    """
    Description
    -----------
    Worker-process wrapper of :func:`session_squeak_spectrograms` that turns an
    exception into an error entry, so one broken session does not stop the
    pool.

    Parameters
    ----------
    session_root (str)
        Session root directory.
    cfg (dict)
        The ``build_squeak_spectrogram_store`` settings block.

    Returns
    -------
    entry (dict)
        The session entry, or ``{"root_directory", "error"}`` on failure.
    """

    try:
        return session_squeak_spectrograms(session_root, cfg)
    except Exception as exc:  # any failure is reported per session by the builder
        return {"root_directory": str(session_root), "error": f"{type(exc).__name__}: {exc}"}


class SqueakSpectrogramStoreBuilder:
    """
    Description
    -----------
    Builds the squeak spectrogram store (2-125 kHz, log frequency) of a list
    of sessions and writes it to ``spectrograms_root``.
    """

    def __init__(
        self,
        root_directories: list[str] | None = None,
        input_parameter_dict: dict | None = None,
        message_output: Callable | None = None,
    ) -> None:
        """
        Description
        -----------
        Initializes the SqueakSpectrogramStoreBuilder.

        Parameters
        ----------
        root_directories (list[str])
            Session root directories.
        input_parameter_dict (dict)
            Processing settings; the ``build_squeak_spectrogram_store`` block
            supplies the band, bins, window, dB range, channel exclusion and
            worker count, and the top-level ``spectrograms_root`` the output
            directory.
        message_output (Callable)
            Logging callback; defaults to ``print``.

        Returns
        -------
        None
        """

        self.root_directories = root_directories if root_directories is not None else []
        self.input_parameter_dict = input_parameter_dict if input_parameter_dict is not None else {}
        self.message_output = message_output if message_output is not None else print
        self.app_context_bool = is_gui_context()

    def build(self) -> pathlib.Path:
        """
        Description
        -----------
        Processes every session (in ``n_workers`` spawned worker processes when
        ``n_workers`` > 1, in this process otherwise), then writes the store
        atomically, sessions in sorted id order. Sessions that fail are
        reported and left out; sessions without squeaks are listed in the
        ``sessions`` table with no spectrogram group.

        Parameters
        ----------

        Returns
        -------
        store_path (pathlib.Path)
            Path of the written store.

        Raises
        ------
        ValueError
            No session list was given, or no session could be processed.
        """

        start_time = datetime.now()
        self.message_output(f"Squeak spectrogram store build started at: {start_time:%H:%M:%S}.")
        smart_wait(app_context_bool=self.app_context_bool, seconds=1)

        cfg = self.input_parameter_dict['build_squeak_spectrogram_store']
        if not self.root_directories:
            error_message = "No session root directories were given to build the squeak spectrogram store from."
            raise ValueError(error_message)
        spectrograms_root = pathlib.Path(configure_path(self.input_parameter_dict['spectrograms_root']))
        edges, centres = log_frequency_bins(cfg['min_freq'], cfg['max_freq'], cfg['n_frequency_bins'])

        entries: list[dict] = []
        failures: list[dict] = []
        n_workers = int(cfg['n_workers'])
        n_sessions = len(self.root_directories)

        def _collect(entry: dict) -> None:
            if "error" in entry:
                failures.append(entry)
                self.message_output(f"FAILED {entry['root_directory']}: {entry['error']}")
                return
            for message in entry['messages']:
                self.message_output(f"{entry['session_id']}: {message}")
            entries.append(entry)
            self.message_output(
                f"[{len(entries) + len(failures)}/{n_sessions}] {entry['session_id']}: "
                f"{entry['row_index'].size} squeak(s) of {entry['n_summary_rows']} rows."
            )

        if n_workers > 1:
            # Spawned workers read the thread-pool variables when they import
            # numpy, so they are set here, before the pool starts, and restored after.
            saved_env = {name: os.environ[name] for name in WORKER_THREAD_ENV_VARS if name in os.environ}
            os.environ.update(dict.fromkeys(WORKER_THREAD_ENV_VARS, "1"))
            try:
                with ProcessPoolExecutor(max_workers=n_workers, mp_context=multiprocessing.get_context("spawn")) as pool:
                    futures = [pool.submit(_session_worker, root, cfg) for root in self.root_directories]
                    for future in as_completed(futures):
                        _collect(future.result())
            finally:
                for name in WORKER_THREAD_ENV_VARS:
                    if name in saved_env:
                        os.environ[name] = saved_env[name]
                    else:
                        del os.environ[name]
        else:
            for root in self.root_directories:
                _collect(_session_worker(root, cfg))

        if not entries:
            error_message = f"None of the {n_sessions} sessions could be processed; nothing was written."
            raise ValueError(error_message)
        entries.sort(key=lambda entry: entry['session_id'])
        n_squeaks = int(sum(entry['row_index'].size for entry in entries))
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%SZ")
        store_path = spectrograms_root / (
            f"{STORE_NAME_PREFIX}_{int(cfg['n_frequency_bins'])}logbins_{len(entries)}sessions_{n_squeaks}squeaks_{timestamp}.h5"
        )

        with atomic_output_path(store_path) as tmp_store_path, h5py.File(tmp_store_path, mode="w") as store_h5:
            store_h5.attrs["created_by"] = "build-squeak-spectrogram-store"
            store_h5.attrs["created_date"] = datetime.now(UTC).isoformat()
            store_h5.attrs["git_commit"] = git_sha_for_provenance(pathlib.Path(__file__).resolve().parent)
            store_h5.attrs["store_layout"] = STORE_NAME_PREFIX
            store_h5.attrs["n_sessions"] = len(entries)
            store_h5.attrs["n_squeaks"] = n_squeaks
            store_h5.attrs["settings"] = json.dumps(cfg, sort_keys=True)
            store_h5.attrs["stft"] = json.dumps({
                "sampling_rate": SQUEAK_SAMPLING_RATE,
                "nperseg": SQUEAK_SPEC_PARAMS["nperseg"],
                "hop_length": SQUEAK_SPEC_PARAMS["hop_length"],
                "window": SQUEAK_SPEC_PARAMS["window"],
                "center": True,
                "db_ref": SQUEAK_STORE_DB_REF,
                "top_db": None,
                "frequency_scale": "log (geometric edges; mean power of the linear bins inside a log bin, "
                                   "linear interpolation at the geometric centre where none falls inside)",
                "channel_average": "audio-variance weighted, in dB",
            }, sort_keys=True)
            store_h5.attrs["frame_dt_s"] = FRAME_DT_S
            store_h5.attrs["window_frames"] = int(cfg['window_frames'])
            store_h5.attrs["db_floor"] = float(cfg['db_floor'])
            store_h5.attrs["db_ceil"] = float(cfg['db_ceil'])
            store_h5.attrs["quantization"] = f"uint8 code = round((clip(dB, db_floor, db_ceil) - db_floor) / (db_ceil - db_floor) * {QUANTIZATION_MAX})"
            store_h5.attrs["failed_sessions"] = json.dumps({entry['root_directory']: entry['error'] for entry in failures})
            store_h5.create_dataset("frequency_bins", data=centres)
            store_h5.create_dataset("frequency_bin_edges", data=edges)
            for entry in entries:
                if entry['row_index'].size == 0:
                    continue
                group = store_h5.create_group(f"spectrogram/{entry['session_id']}")
                group.create_dataset(
                    "spectrograms",
                    data=entry['spectrograms'],
                    chunks=(1, *entry['spectrograms'].shape[1:]),
                    compression="gzip",
                    compression_opts=4,
                )
                for name in ("row_index", "durations", "n_frames", "window_first", "audio_start_s"):
                    group.create_dataset(name, data=entry[name])
            _polars_to_h5(store_h5, "sessions", pls.DataFrame({
                "session_id": [entry['session_id'] for entry in entries],
                "root_directory": [entry['root_directory'] for entry in entries],
                "usv_summary_sha256": [entry['usv_summary_sha256'] for entry in entries],
                "n_summary_rows": [entry['n_summary_rows'] for entry in entries],
                "n_squeaks": [int(entry['row_index'].size) for entry in entries],
            }, schema={
                "session_id": pls.String, "root_directory": pls.String, "usv_summary_sha256": pls.String,
                "n_summary_rows": pls.Int64, "n_squeaks": pls.Int64,
            }))

        end_time = datetime.now()
        self.message_output(
            f"Wrote {n_squeaks} squeak spectrograms of {len(entries)} sessions ({len(failures)} failed) -> {store_path} "
            f"({store_path.stat().st_size / 1e6:.1f} MB) in {(end_time - start_time).total_seconds() / 60:.1f} min."
        )
        self.message_output(f"Squeak spectrogram store build ended at: {end_time:%H:%M:%S}.")
        return store_path


@click.command(name="build-squeak-spectrogram-store")
@click.option('--root-directories', type=str, default=None, required=False, help='Comma-separated string of session root directory paths. Give this and/or --session-lists.')
@click.option('--session-lists', 'session_lists', type=str, default=None, required=False, help='Comma-separated string of session-list .txt files (one session root per line). Give this and/or --root-directories.')
@click.option('--spectrograms-root', 'spectrograms_root', type=click.Path(exists=True, file_okay=False, dir_okay=True), default=None, required=False, help='Output directory the store is written to (the spectrograms_root setting when not given).')
@click.option('--min-freq', 'min_freq', type=float, default=None, required=False, help='Lower edge (Hz) of the lowest log-frequency bin.')
@click.option('--max-freq', 'max_freq', type=float, default=None, required=False, help='Upper edge (Hz) of the highest log-frequency bin (at most 125000, the Nyquist frequency).')
@click.option('--n-frequency-bins', 'n_frequency_bins', type=int, default=None, required=False, help='Number of log-spaced frequency bins.')
@click.option('--window-frames', 'window_frames', type=int, default=None, required=False, help='Width (2.048 ms frames) of the fixed window every call is stored in.')
@click.option('--db-floor', 'db_floor', type=float, default=None, required=False, help='Absolute dB mapped to uint8 code 0 (values below are clipped).')
@click.option('--db-ceil', 'db_ceil', type=float, default=None, required=False, help='Absolute dB mapped to uint8 code 255 (values above are clipped).')
@click.option('--exclude-metadata-audio-channels/--no-exclude-metadata-audio-channels', 'exclude_metadata_audio_channels', default=None, required=False, help='Drop channels the session metadata marks as excluded from the spectrogram average (keep it equal to the detect-usv-squeaks run).')
@click.option('--n-workers', 'n_workers', type=int, default=None, required=False, help='Sessions processed in parallel worker processes (1: in this process).')
@click.pass_context
def build_squeak_spectrogram_store_cli(ctx, root_directories, session_lists, spectrograms_root, **kwargs) -> None:
    """
    Description
    -----------
    A command-line tool to build the squeak spectrogram store (2-125 kHz, log
    frequency) the embedding explorer shows on its Squeaks map, from session
    root directories and/or session-list files.

    Parameters
    ----------

    Returns
    -------
    None
    """

    if not root_directories and not session_lists:
        usage_message = "Give --root-directories and/or --session-lists."
        raise click.UsageError(usage_message)

    provided_params = [key for key in kwargs if ctx.get_parameter_source(key) == ParameterSource.COMMANDLINE]

    processing_settings_dict = modify_settings_json_for_cli(
        ctx=ctx,
        provided_params=provided_params,
        settings_dict='processing_settings',
        block='build_squeak_spectrogram_store',
    )
    if spectrograms_root is not None:
        processing_settings_dict['spectrograms_root'] = spectrograms_root

    all_paths: list[str] = []
    if session_lists:
        all_paths.extend(read_session_lists([one_list.strip() for one_list in session_lists.split(',') if one_list.strip()]))
    if root_directories:
        for one_dir in root_directories.split(','):
            resolved = configure_path(one_dir.strip()) if one_dir.strip() else ""
            if resolved and resolved not in all_paths:
                all_paths.append(resolved)

    SqueakSpectrogramStoreBuilder(
        root_directories=all_paths,
        input_parameter_dict=processing_settings_dict,
        message_output=print,
    ).build()
