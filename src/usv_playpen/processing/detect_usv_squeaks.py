"""
@author: bartulem
Detect squeaks (broadband vocalizations) among a session's USV segments and
merge the result into its ``*_usv_summary.csv``.

A squeak is a broadband harmonic stack (fundamental around 3-8 kHz) that the
ultrasonic DAS segmenter picks up as part of a USV segment. This step scores
every segment with the v3 time-resolved multiple-instance classifier
(``TimeMIL``, 250,118 parameters, decision threshold 0.385), ported here from
``/mnt/falkner/Dexter/vocal_beh/models/bbv_classifier/code/bbv_infer.py`` with
the same architecture, input transform and frame mask, but without that file's
import-time process-global torch settings.

Input contract (fixed by the trained model, therefore not exposed as settings):
the spectrogram of each segment is rebuilt from the UNFILTERED per-channel
``audio/hpss/*_cropped_to_video_hpss.wav`` files (``audio/hpss_filtered`` is
high-passed above 30 kHz and carries no squeak energy), read through soundfile
as float64 (int16 / 32768), with a Blackman-Harris STFT (nperseg 2048, hop 512,
centred), 3-30 kHz, 128 linear frequency bins, absolute dB (``ref=1.0``, no
``top_db`` clamp) and a variance-weighted average across channels. This front
end reproduces the reference ``_sonic_wav_`` spectrogram store bit-exactly
(0.000 dB, verified 2026-09-14). The model sees a fixed 128-frame window
(262 ms): a shorter segment is padded at -100 dB and the padding masked out; a
longer one is truncated to its first 128 frames.

Columns written into ``usv_summary.csv`` (any pre-existing ones are replaced):

* ``squeak`` -- True / False in every row (``p >= squeak_threshold`` on the
  first 128 frames, the reference segment-level method);
* ``squeak_probability`` -- that probability, on squeak rows only (empty
  otherwise);
* ``squeak_start`` / ``squeak_end`` -- session-clock seconds (the same clock as
  ``start`` / ``stop``) of the centres of the first and last frames whose
  per-frame probability reaches the threshold, on squeak rows only. These come
  from ONE FULL-LENGTH pass over the whole segment, because the 128-frame
  window cuts the squeak end off at frame 127 for about a third of long squeak
  segments (median 60 ms early). Frame ``t`` is centred at
  ``start + t * 0.002048`` s. Every squeak row has a start and an end: the
  segment logit is an attention-weighted average of frame logits, so the
  segment cannot reach the threshold unless some frame does.
* ``squeak_frame_runs`` -- integer in every row (0 on unscorable rows): the
  number of runs of at least 3 consecutive frames whose per-frame probability
  reaches the threshold, counted over the frames of the SAME 128-frame window
  the segment call is made on (the first ``min(n_frames, 128)`` frames, padding
  excluded), not over the full-length pass. This is the ``n_bouts_min3`` column
  of the reference squeak index, whose strict squeak rule (``squeak`` true OR
  ``squeak_frame_runs >= 1``) left broadband calls out of the reference USV
  training sets; the window matches the reference so that rule can be
  evaluated from the summary (``build-qlvm-training-set
  --strict-squeak-exclusion``).

Squeak QLVM embedding (``infer-qlvm-squeak-latents``, :class:`USVSqueakQLVMEmbedder`):
a second step places every squeak row that is not noise on the torus of one of
the phase 3 broadband-vocalization QLVM cells
(``qlvm_models_latest/phase3_BBVs_qlvm/<cell>``: the
``infer_qlvm_squeak_latents.model_cell_directory`` setting, filled with the
production ``natural_session_N11000_nomask`` cell when empty) and writes the two
float columns ``qlvm_squeak1`` / ``qlvm_squeak2`` (torus coordinates in
``[0, 1)``, null on every other row). Its input is the same sonic spectrogram,
cropped to the squeak's own extent the way the reference builder
``build_bbv_dataset.py`` built the cells' training sets: frames
``squeak_start`` .. ``squeak_end`` plus two frames of context either side
(clipped to the segment), min-max normalized per crop, zero-padded to 128
frames, centred by ``stretch_specs`` and min-max normalized once more as the
decoder's data loader did (:func:`squeak_qlvm_inputs`).
Crops narrower than 8 frames (the training set's minimum) or wider than 128
frames (the decoder's frame; the training set never compressed a crop in time)
get nulls. Unlike the training set, the crop is taken from the FULL-LENGTH
spectrogram with the full-length ``squeak_start`` / ``squeak_end`` this step
writes, so a squeak that runs past frame 127 of a longer segment is embedded
with its measured extent (the training set, built from a 128-frame store,
dropped such right-censored squeaks instead). The embedding is the posterior
mean over the cell's Fibonacci lattice (``lattice_m`` of its ``manifest.json``)
built in float32 as the torch driver built it (:func:`qlvm_model.gen_fib_basis_float32`).

Note on training sessions: the deployed checkpoint was fitted on labelled
segments from 15 sessions (20230124_172125, 20250211_165612, 20250403_205653,
20250418_184440, 20250424_175844, 20250506_155030, 20250923_203320,
20250927_160820, 20250928_185641, 20250928_212046, 20251118_103002,
20251221_111055, 20251221_114053, 20251221_124144, 20251221_142801). Its
output there is a fit, not a prediction, so leave those sessions out of any
cohort squeak-rate claim.
"""

from __future__ import annotations

import json
import pathlib
from collections.abc import Callable
from datetime import datetime

import click
import jax.numpy as jnp
import numpy as np
import polars as pls
import soundfile as sf
import torch
from click.core import ParameterSource
from torch import nn

from ..cli_utils import modify_settings_json_for_cli
from ..os_utils import (
    atomic_output_path,
    configure_path,
    derive_spectrogram_model_paths,
    first_match_or_raise,
    order_usv_summary_columns,
)
from ..time_utils import is_gui_context, smart_wait
from ..yaml_utils import read_excluded_audio_channels
from .build_qlvm_training_set import stretch_specs
from .generate_spectrograms import compute_usv_spectrogram
from .qlvm_latents import cell_file, load_decoder_params, normalize_model_inputs
from .qlvm_model import decoder_head, embed_data, gen_fib_basis_float32

# Columns written into the USV summary CSV.
SQUEAK_COLUMNS = ("squeak", "squeak_probability", "squeak_start", "squeak_end")

# Column holding the number of above-threshold frame runs of at least
# SQUEAK_FRAME_RUN_MIN_FRAMES frames in the 128-frame model window (the reference
# squeak index's n_bouts_min3); kept out of SQUEAK_COLUMNS so the steps that
# require the four squeak columns do not require it on older summaries.
SQUEAK_FRAME_RUNS_COLUMN = "squeak_frame_runs"
SQUEAK_FRAME_RUN_MIN_FRAMES = 3

# Input contract of the trained squeak model; changing any of these makes the
# checkpoint's output meaningless, so they are constants rather than settings.
SQUEAK_WAV_GLOB = "*_cropped_to_video_hpss.wav"
SQUEAK_SAMPLING_RATE = 250000
SQUEAK_SPEC_PARAMS = {
    "num_freq_bins": 128,
    "num_time_bins": None,
    "nperseg": 2048,
    "min_freq": 3000.0,
    "max_freq": 30000.0,
    "hop_length": 512,
    "window": "blackmanharris",
}
SQUEAK_DB_REF = 1.0
MODEL_WINDOW_FRAMES = 128
PAD_DB = -100.0
DB_FLOOR = -100.0
DB_CEIL = 50.0
DB_CENTER = -25.0
DB_HALF = 75.0
FRAME_DT_S = SQUEAK_SPEC_PARAMS["hop_length"] / SQUEAK_SAMPLING_RATE
CHECKPOINT_NORM_KIND = "fixed_affine_absolute_db"
CHECKPOINT_MASK_RULE = "arange(128) < min(n_frames,128)"

# Columns the squeak QLVM embedding writes into the USV summary CSV.
SQUEAK_QLVM_COLUMNS = ("qlvm_squeak1", "qlvm_squeak2")

# Input contract of the phase 3 squeak (BBV) QLVM cells, fixed by the way
# scripts/dataset_construct/build_bbv_dataset.py built their training sets (the
# cells record them only in their model cards), therefore constants, not settings:
# two context frames either side of the squeak extent, crops of at least 8 frames,
# per-crop min-max with epsilon 1e-6, a 128 x 128 frame the crop is centred in
# without time stretching, and the data loader's second per-spectrogram min-max
# with epsilon 1e-8 (qmc_deep_gen data/mouse_data.py).
SQUEAK_QLVM_CONTEXT_FRAMES = 2
SQUEAK_QLVM_MIN_CROP_FRAMES = 8
SQUEAK_QLVM_CROP_EPSILON = 1e-6
SQUEAK_QLVM_TARGET_SHAPE = (128, 128)
SQUEAK_QLVM_INPUT_CONTRACT = {"input_normalization": "minmax", "normalization_epsilon": 1e-8, "floor": None}


class TimeMIL(nn.Module):
    """
    Description
    -----------
    Time-resolved multiple-instance squeak classifier (the reference ``TimeMIL``,
    verbatim architecture). Four Conv2d-BatchNorm-ReLU blocks pool only along
    frequency, so every time column keeps its own representation; a 1x1 frame
    head emits a per-frame logit and a 1x1 attention head weights the frames.
    The segment logit is the attention-weighted sum of the frame logits over the
    valid (unmasked) frames. Attribute names match the checkpoint's
    ``state_dict`` keys and must not change.
    """

    def __init__(self, ch: int = 32, pool: str = "attn") -> None:
        """
        Description
        -----------
        Builds the network.

        Parameters
        ----------
        ch (int)
            Base channel width; the last block has ``4 * ch`` channels.
        pool (str)
            ``"attn"`` (deployed) for attention pooling, or ``"max"`` for the
            max over valid frame logits (present in the source, never deployed).

        Returns
        -------
        None
        """

        super().__init__()
        self.pool = pool
        c = ch
        self.body = nn.Sequential(
            nn.Conv2d(1, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(c, c * 2, 3, padding=1), nn.BatchNorm2d(c * 2), nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(c * 2, c * 4, 3, padding=1), nn.BatchNorm2d(c * 4), nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(c * 4, c * 4, 3, padding=1), nn.BatchNorm2d(c * 4), nn.ReLU(),
            nn.AdaptiveMaxPool2d((1, None)),
        )
        self.frame = nn.Conv1d(c * 4, 1, 1)
        self.attn = nn.Sequential(nn.Conv1d(c * 4, 64, 1), nn.Tanh(), nn.Conv1d(64, 1, 1))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Description
        -----------
        Runs the network on a batch of normalized spectrograms.

        Parameters
        ----------
        x (torch.Tensor)
            Normalized spectrograms, shape ``(B, 1, 128, T)``; ``T`` may be any
            length (the network is fully convolutional along time).
        mask (torch.Tensor)
            Boolean validity mask, shape ``(B, T)``; False frames are ignored by
            the pooling.

        Returns
        -------
        segment_logit (torch.Tensor)
            Segment logit, shape ``(B,)``.
        frame_logit (torch.Tensor)
            Per-frame logit, shape ``(B, T)`` (NOT masked).
        """

        h = self.body(x).squeeze(2)
        frame_logit = self.frame(h).squeeze(1)
        neg = torch.finfo(frame_logit.dtype).min
        if self.pool == "max":
            segment_logit = frame_logit.masked_fill(~mask, neg).max(dim=1).values
        else:
            attention = self.attn(h).squeeze(1).masked_fill(~mask, neg)
            weights = torch.softmax(attention, dim=1)
            segment_logit = (weights * frame_logit.masked_fill(~mask, 0.0)).sum(1)
        return segment_logit, frame_logit


def load_squeak_model(checkpoint_path: str, device: torch.device) -> TimeMIL:
    """
    Description
    -----------
    Loads the squeak classifier checkpoint and asserts that its recorded input
    normalization and frame-mask rule match the constants this module applies,
    so a checkpoint trained under a different input contract is refused rather
    than silently producing meaningless probabilities.

    Parameters
    ----------
    checkpoint_path (str)
        Path to the ``.pt`` checkpoint (a dict with ``state_dict``, ``norm``,
        ``ch``, ``pool`` and, for the deployed model, ``mask``).
    device (torch.device)
        Device to place the model on.

    Returns
    -------
    model (TimeMIL)
        The model in eval mode.
    """

    checkpoint = torch.load(configure_path(checkpoint_path), map_location="cpu", weights_only=True)
    norm = checkpoint["norm"]
    expected_norm = (CHECKPOINT_NORM_KIND, DB_FLOOR, DB_CEIL, DB_CENTER, DB_HALF)
    found_norm = (norm["kind"], norm["db_floor"], norm["db_ceil"], norm["center"], norm["half"])
    if found_norm != expected_norm:
        error_message = f"Squeak checkpoint normalization {found_norm} does not match the expected {expected_norm}."
        raise ValueError(error_message)
    if "mask" in checkpoint and checkpoint["mask"] != CHECKPOINT_MASK_RULE:
        error_message = f"Squeak checkpoint mask rule {checkpoint['mask']!r} does not match {CHECKPOINT_MASK_RULE!r}."
        raise ValueError(error_message)
    model = TimeMIL(ch=checkpoint["ch"], pool=checkpoint["pool"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def normalize_absolute_db(spectrogram_db: np.ndarray) -> np.ndarray:
    """
    Description
    -----------
    Applies the model's single fixed affine input transform,
    ``(clip(x_dB, -100, 50) - (-25)) / 75``, mapping absolute dB to ``[-1, 1]``
    with padding at -1. No per-segment statistics are used, so a threshold
    means the same thing in every recording.

    Parameters
    ----------
    spectrogram_db (np.ndarray)
        Absolute-dB spectrogram(s) of any shape.

    Returns
    -------
    normalized (np.ndarray)
        Float32 array of the same shape.
    """

    return (np.clip(spectrogram_db, DB_FLOOR, DB_CEIL).astype(np.float32) - DB_CENTER) / DB_HALF


def model_window(spectrogram_db: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Description
    -----------
    Builds the fixed 128-frame model window of one segment exactly as the
    training store did: the first 128 frames of a longer segment, or a shorter
    segment padded to 128 frames at -100 dB, together with the validity mask
    ``arange(128) < min(n_frames, 128)``.

    Parameters
    ----------
    spectrogram_db (np.ndarray)
        Full-length absolute-dB spectrogram, shape ``(128, n_frames)``.

    Returns
    -------
    window (np.ndarray)
        Float32 window, shape ``(128, 128)``.
    valid (np.ndarray)
        Boolean mask, shape ``(128,)``.
    """

    n_frames = spectrogram_db.shape[1]
    if n_frames >= MODEL_WINDOW_FRAMES:
        window = spectrogram_db[:, :MODEL_WINDOW_FRAMES]
    else:
        padding = np.full(
            (spectrogram_db.shape[0], MODEL_WINDOW_FRAMES - n_frames), PAD_DB, dtype=spectrogram_db.dtype
        )
        window = np.concatenate([spectrogram_db, padding], axis=1)
    valid = np.arange(MODEL_WINDOW_FRAMES) < min(n_frames, MODEL_WINDOW_FRAMES)
    return window.astype(np.float32), valid


def squeak_extent_seconds(
    frame_probability: np.ndarray,
    segment_start_s: float,
    threshold: float,
) -> tuple[float | None, float | None]:
    """
    Description
    -----------
    Converts a segment's per-frame squeak probabilities into the session-clock
    times of the first and last frames at or above the threshold, using the
    centred-STFT frame convention ``start + t * 0.002048`` s.

    Parameters
    ----------
    frame_probability (np.ndarray)
        Per-frame probabilities over the segment's valid frames, shape ``(n,)``.
    segment_start_s (float)
        The segment's ``start`` in session seconds.
    threshold (float)
        Frame decision threshold.

    Returns
    -------
    squeak_start (float | None)
        Centre of the first above-threshold frame, or None if no frame reaches it.
    squeak_end (float | None)
        Centre of the last above-threshold frame, or None if no frame reaches it.
    """

    above = np.flatnonzero(frame_probability >= threshold)
    if above.size == 0:
        return None, None
    return (
        float(segment_start_s + above[0] * FRAME_DT_S),
        float(segment_start_s + above[-1] * FRAME_DT_S),
    )


def squeak_frame_run_count(
    frame_probability: np.ndarray,
    threshold: float,
    min_run_frames: int = SQUEAK_FRAME_RUN_MIN_FRAMES,
) -> int:
    """
    Description
    -----------
    Counts the maximal runs of consecutive frames whose per-frame squeak
    probability reaches the threshold (``p >= threshold``) and that span at
    least ``min_run_frames`` frames. With the default of 3 frames and the
    frames of the 128-frame model window this is the ``n_bouts_min3`` of the
    reference squeak index (its ``bout_stats``: runs of the boolean frame
    presence ``(p_frame >= 0.385) & valid`` over the first ``n_valid_frames``
    frames, a run counted when its length is ``>= 3``).

    Parameters
    ----------
    frame_probability (np.ndarray)
        Per-frame probabilities over the frames to count, shape ``(n,)``
        (padding must already be excluded).
    threshold (float)
        Frame decision threshold.
    min_run_frames (int)
        Shortest run that is counted, in frames.

    Returns
    -------
    n_runs (int)
        Number of above-threshold runs of at least ``min_run_frames`` frames
        (0 for an empty input).
    """

    above = np.asarray(frame_probability) >= threshold
    if not above.any():
        return 0
    edges = np.diff(np.concatenate(([0], above.astype(np.int8), [0])))
    run_lengths = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
    return int(np.count_nonzero(run_lengths >= min_run_frames))


def squeak_wav_channels(
    session_root: pathlib.Path,
    exclude_metadata_audio_channels: bool,
    message_output: Callable,
) -> list[pathlib.Path]:
    """
    Description
    -----------
    Lists the session's unfiltered per-channel HPSS wavs, optionally dropping
    the channels the session metadata marks as hardware-excluded
    (``Equipment -> audio_Avisoft -> excluded_channels``, names such as
    ``m_ch02`` / ``s_ch11``), and checks that every remaining channel shares the
    model's sampling rate and one common length.

    Parameters
    ----------
    session_root (pathlib.Path)
        Session root directory.
    exclude_metadata_audio_channels (bool)
        Whether to drop metadata-excluded channels from the average.
    message_output (Callable)
        Logging callback.

    Returns
    -------
    wav_paths (list[pathlib.Path])
        Sorted wav paths that enter the average.
    """

    wav_paths = sorted((session_root / "audio" / "hpss").glob(SQUEAK_WAV_GLOB))
    if not wav_paths:
        error_message = f"No {SQUEAK_WAV_GLOB} files under {session_root / 'audio' / 'hpss'}."
        raise FileNotFoundError(error_message)
    if exclude_metadata_audio_channels:
        excluded_channels = set(read_excluded_audio_channels(str(session_root), logger=message_output))
        if excluded_channels:
            message_output(f"Excluding audio channel(s) {sorted(excluded_channels)} from the squeak spectrogram average per session metadata.")
        wav_paths = [
            wav_path for wav_path in wav_paths
            if f"{wav_path.name.split('_')[0]}_{wav_path.name.split('_')[2]}" not in excluded_channels
        ]
    infos = [sf.info(str(wav_path)) for wav_path in wav_paths]
    sampling_rates = {info.samplerate for info in infos}
    lengths = {info.frames for info in infos}
    if sampling_rates != {SQUEAK_SAMPLING_RATE} or len(lengths) != 1:
        error_message = f"Squeak wav channels disagree or have the wrong rate in {session_root}: rates={sampling_rates}, lengths={lengths}."
        raise ValueError(error_message)
    return wav_paths


def squeak_segment_spectrograms(
    session_root: pathlib.Path,
    usv_summary: pls.DataFrame,
    row_indices: np.ndarray,
    exclude_metadata_audio_channels: bool,
    message_output: Callable,
) -> list[np.ndarray | None]:
    """
    Description
    -----------
    Rebuilds the full-length, absolute-dB sonic spectrogram of each requested
    USV summary row from the session's unfiltered HPSS wavs, with the squeak
    model's input contract (``SQUEAK_SPEC_PARAMS``: Blackman-Harris STFT,
    nperseg 2048, hop 512, centred, 3-30 kHz, 128 linear frequency bins,
    ``ref=1.0``, no ``top_db`` clamp, variance-weighted channel average; the
    front end that reproduces the reference ``_sonic_wav_`` store bit-exactly). The
    audio of a row spans ``round(start * 250000)`` to ``round(stop * 250000)``
    samples on every channel. Frame ``t`` of a spectrogram is centred at
    ``start + t * FRAME_DT_S`` s. Both the squeak classifier
    (:func:`score_squeak_rows`) and the squeak QLVM embedding
    (:func:`squeak_qlvm_inputs`) read their spectrograms here.

    Parameters
    ----------
    session_root (pathlib.Path)
        Session root directory.
    usv_summary (pls.DataFrame)
        The session's USV summary (must hold ``start`` and ``stop`` in seconds).
    row_indices (np.ndarray)
        0-based summary row indices to rebuild.
    exclude_metadata_audio_channels (bool)
        Whether to drop metadata-excluded channels from the average.
    message_output (Callable)
        Logging callback.

    Returns
    -------
    spectrograms (list[np.ndarray | None])
        One entry per requested row, in order: the float32 ``(128, n_frames)``
        absolute-dB spectrogram, or None when the segment is too short for a
        single STFT frame on any channel.
    """

    wav_paths = squeak_wav_channels(session_root, exclude_metadata_audio_channels, message_output)
    starts = usv_summary["start"].to_numpy()
    stops = usv_summary["stop"].to_numpy()

    spectrograms: list[np.ndarray | None] = []
    handles = [sf.SoundFile(str(wav_path), mode="r") for wav_path in wav_paths]
    try:
        for row_index in row_indices:
            first_sample = round(float(starts[row_index]) * SQUEAK_SAMPLING_RATE)
            last_sample = round(float(stops[row_index]) * SQUEAK_SAMPLING_RATE)
            channel_audio = []
            for handle in handles:
                handle.seek(first_sample)
                channel_audio.append(handle.read(frames=max(0, last_sample - first_sample), dtype="float64", always_2d=False))
            spectrogram, n_frames = compute_usv_spectrogram(
                audio_segment_channels=np.stack(channel_audio, axis=1),
                sampling_rate=SQUEAK_SAMPLING_RATE,
                spec_params=SQUEAK_SPEC_PARAMS,
                normalize=False,
                db_ref=SQUEAK_DB_REF,
                top_db=None,
            )
            spectrograms.append(None if spectrogram is None or n_frames == 0 else spectrogram.astype(np.float32))
    finally:
        for handle in handles:
            handle.close()
    return spectrograms


def score_squeak_rows(
    session_root: pathlib.Path,
    usv_summary: pls.DataFrame,
    row_indices: np.ndarray,
    model: TimeMIL,
    device: torch.device,
    threshold: float,
    exclude_metadata_audio_channels: bool,
    batch_size: int,
    message_output: Callable,
) -> pls.DataFrame:
    """
    Description
    -----------
    Scores the requested USV summary rows of one session. For each row it
    rebuilds the full-length absolute-dB spectrogram from the HPSS wavs, scores
    the fixed 128-frame window in batches (segment probability, the reference
    method), and takes squeak onset / offset from a full-length pass (the
    128-frame window's own frames when the segment fits inside it, a separate
    forward over all frames when it does not). The number of above-threshold
    frame runs of at least ``SQUEAK_FRAME_RUN_MIN_FRAMES`` frames
    (:func:`squeak_frame_run_count`) is counted on every scorable row over the
    valid frames of the 128-frame window, the frames the reference squeak index
    counted its ``n_bouts_min3`` on. Rows whose segment is too short for a
    single STFT frame on any channel are returned with no probability,
    ``squeak`` False and no frame runs.

    cuDNN is put in deterministic mode for the duration of the call and restored
    afterwards, and autocast is disabled locally, so a mixed-precision context
    left open by an earlier step in the same process (the mask step enters one)
    cannot change the probabilities.

    Parameters
    ----------
    session_root (pathlib.Path)
        Session root directory.
    usv_summary (pls.DataFrame)
        The session's USV summary (must hold ``start`` and ``stop`` in seconds).
    row_indices (np.ndarray)
        0-based summary row indices to score.
    model (TimeMIL)
        Loaded squeak model.
    device (torch.device)
        Device the model lives on.
    threshold (float)
        Segment and frame decision threshold.
    exclude_metadata_audio_channels (bool)
        Whether to drop metadata-excluded channels from the average.
    batch_size (int)
        Number of 128-frame windows per forward pass.
    message_output (Callable)
        Logging callback.

    Returns
    -------
    scores (pls.DataFrame)
        One row per requested index: ``row_index``, ``n_frames`` (native STFT
        frames, 0 when unscorable), ``raw_probability`` (the segment probability
        for every scorable row), the four ``SQUEAK_COLUMNS`` and
        ``SQUEAK_FRAME_RUNS_COLUMN`` (int64 in every row, 0 when unscorable).
    """

    starts = usv_summary["start"].to_numpy()
    spectrograms = squeak_segment_spectrograms(
        session_root=session_root,
        usv_summary=usv_summary,
        row_indices=row_indices,
        exclude_metadata_audio_channels=exclude_metadata_audio_channels,
        message_output=message_output,
    )

    n_rows = len(row_indices)
    raw_probability = np.full(n_rows, np.nan)
    full_pass_frames: list[np.ndarray | None] = [None] * n_rows
    frame_runs = np.zeros(n_rows, dtype=np.int64)
    scorable = [position for position in range(n_rows) if spectrograms[position] is not None]

    previous_deterministic = torch.backends.cudnn.deterministic
    previous_benchmark = torch.backends.cudnn.benchmark
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
            for batch_start in range(0, len(scorable), batch_size):
                batch_positions = scorable[batch_start:batch_start + batch_size]
                windows, valids = zip(*(model_window(spectrograms[position]) for position in batch_positions), strict=True)
                x = torch.from_numpy(normalize_absolute_db(np.stack(windows))).unsqueeze(1).to(device)
                mask = torch.from_numpy(np.stack(valids)).to(device)
                segment_logit, frame_logit = model(x, mask)
                segment_probability = torch.sigmoid(segment_logit).float().cpu().numpy()
                frame_probability = torch.sigmoid(frame_logit).float().cpu().numpy()
                for offset, position in enumerate(batch_positions):
                    raw_probability[position] = float(segment_probability[offset])
                    n_frames = spectrograms[position].shape[1]
                    frame_runs[position] = squeak_frame_run_count(
                        frame_probability[offset, :min(n_frames, MODEL_WINDOW_FRAMES)], threshold
                    )
                    if n_frames <= MODEL_WINDOW_FRAMES:
                        full_pass_frames[position] = frame_probability[offset, :n_frames]

            for position in scorable:
                spectrogram = spectrograms[position]
                if spectrogram.shape[1] > MODEL_WINDOW_FRAMES and raw_probability[position] >= threshold:
                    x = torch.from_numpy(normalize_absolute_db(spectrogram)).unsqueeze(0).unsqueeze(0).to(device)
                    mask = torch.ones((1, spectrogram.shape[1]), dtype=torch.bool, device=device)
                    _, frame_logit = model(x, mask)
                    full_pass_frames[position] = torch.sigmoid(frame_logit).float().cpu().numpy()[0]
    finally:
        torch.backends.cudnn.deterministic = previous_deterministic
        torch.backends.cudnn.benchmark = previous_benchmark

    squeak = np.zeros(n_rows, dtype=bool)
    squeak_probability: list[float | None] = [None] * n_rows
    squeak_start: list[float | None] = [None] * n_rows
    squeak_end: list[float | None] = [None] * n_rows
    for position in scorable:
        if raw_probability[position] >= threshold:
            squeak[position] = True
            squeak_probability[position] = float(raw_probability[position])
            squeak_start[position], squeak_end[position] = squeak_extent_seconds(
                frame_probability=full_pass_frames[position],
                segment_start_s=float(starts[row_indices[position]]),
                threshold=threshold,
            )

    return pls.DataFrame(
        {
            "row_index": np.asarray(row_indices, dtype=np.int64),
            "n_frames": [0 if spectrogram is None else int(spectrogram.shape[1]) for spectrogram in spectrograms],
            "raw_probability": raw_probability,
            "squeak": squeak,
            "squeak_probability": squeak_probability,
            "squeak_start": squeak_start,
            "squeak_end": squeak_end,
            SQUEAK_FRAME_RUNS_COLUMN: frame_runs,
        },
        schema={
            "row_index": pls.Int64,
            "n_frames": pls.Int64,
            "raw_probability": pls.Float64,
            "squeak": pls.Boolean,
            "squeak_probability": pls.Float64,
            "squeak_start": pls.Float64,
            "squeak_end": pls.Float64,
            SQUEAK_FRAME_RUNS_COLUMN: pls.Int64,
        },
    )


class USVSqueakDetector:
    """
    Description
    -----------
    Scores every USV segment of one session for a squeak and merges the four
    squeak columns and ``squeak_frame_runs`` into its ``*_usv_summary.csv``.
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
        Initializes the USVSqueakDetector.

        Parameters
        ----------
        root_directory (str)
            Session root directory (contains the ``audio`` tree).
        input_parameter_dict (dict)
            Processing settings; the ``detect_usv_squeaks`` block supplies the
            checkpoint path, threshold, channel-exclusion switch and batch size.
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

    def detect_and_merge(self) -> None:
        """
        Description
        -----------
        Loads the squeak model, scores every row of the session's USV summary
        (see :func:`score_squeak_rows`), and writes ``squeak``,
        ``squeak_probability``, ``squeak_start``, ``squeak_end`` and
        ``squeak_frame_runs`` into the summary, replacing any existing ones. Run it after
        ``das_summarize``: re-summarizing rewrites the CSV with its base columns
        only and would remove these.

        Parameters
        ----------

        Returns
        -------
        Updated ``*_usv_summary.csv`` with the four squeak columns and
        ``squeak_frame_runs``.
        """

        self.message_output(
            f"USV squeak detection started at: {datetime.now().hour:02d}:{datetime.now().minute:02d}:{datetime.now().second:02d}."
        )
        smart_wait(app_context_bool=self.app_context_bool, seconds=1)

        derive_spectrogram_model_paths(self.input_parameter_dict)
        cfg = self.input_parameter_dict['detect_usv_squeaks']
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_squeak_model(cfg['squeak_model_path'], device)

        root = pathlib.Path(self.root_directory)
        usv_summary_loc = first_match_or_raise(
            root=root / "audio",
            pattern="*_usv_summary.csv",
            recursive=True,
            label="USV summary CSV",
        )
        usv_df = pls.read_csv(source=str(usv_summary_loc), schema_overrides={"usv_id": pls.String})
        usv_df = usv_df.drop([column for column in (*SQUEAK_COLUMNS, SQUEAK_FRAME_RUNS_COLUMN) if column in usv_df.columns])

        scores = score_squeak_rows(
            session_root=root,
            usv_summary=usv_df,
            row_indices=np.arange(usv_df.height),
            model=model,
            device=device,
            threshold=cfg['squeak_threshold'],
            exclude_metadata_audio_channels=cfg['exclude_metadata_audio_channels'],
            batch_size=cfg['batch_size'],
            message_output=self.message_output,
        )
        merged = order_usv_summary_columns(pls.concat([usv_df, scores.select(*SQUEAK_COLUMNS, SQUEAK_FRAME_RUNS_COLUMN)], how="horizontal"))
        merged.write_csv(file=str(usv_summary_loc))

        self.message_output(
            f"Merged squeak calls into {usv_summary_loc.name}: {int(scores['squeak'].sum())} squeak(s) among {usv_df.height} USVs."
        )
        self.message_output(
            f"USV squeak detection ended at: {datetime.now().hour:02d}:{datetime.now().minute:02d}:{datetime.now().second:02d}."
        )


def squeak_crop_frames(
    segment_start_s: np.ndarray,
    squeak_start_s: np.ndarray,
    squeak_end_s: np.ndarray,
    n_frames: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Description
    -----------
    The first and last spectrogram frame of each squeak's QLVM crop: the frames
    whose centres are ``squeak_start`` and ``squeak_end``
    (``round((t - start) / FRAME_DT_S)``, the inverse of
    :func:`squeak_extent_seconds`), widened by ``SQUEAK_QLVM_CONTEXT_FRAMES``
    either side and clipped to the segment's frames ``0 .. n_frames - 1``. This
    is the rule of the reference builder ``build_bbv_dataset.py``, except that the segment
    there was clipped to its first 128 frames (``n_valid_frames``) because the
    store it cropped from held no more.

    Parameters
    ----------
    segment_start_s (np.ndarray)
        ``(N,)`` segment ``start`` in session seconds.
    squeak_start_s (np.ndarray)
        ``(N,)`` ``squeak_start`` in session seconds.
    squeak_end_s (np.ndarray)
        ``(N,)`` ``squeak_end`` in session seconds.
    n_frames (np.ndarray)
        ``(N,)`` number of frames of each segment's full-length spectrogram.

    Returns
    -------
    first (np.ndarray)
        ``(N,)`` int64 first frame of each crop.
    last (np.ndarray)
        ``(N,)`` int64 last frame of each crop (inclusive).
    """

    segment_start_s = np.asarray(segment_start_s, dtype=np.float64)
    first = np.round((np.asarray(squeak_start_s, dtype=np.float64) - segment_start_s) / FRAME_DT_S).astype(np.int64)
    last = np.round((np.asarray(squeak_end_s, dtype=np.float64) - segment_start_s) / FRAME_DT_S).astype(np.int64)
    first = np.maximum(first - SQUEAK_QLVM_CONTEXT_FRAMES, 0)
    last = np.minimum(last + SQUEAK_QLVM_CONTEXT_FRAMES, np.asarray(n_frames, dtype=np.int64) - 1)
    return first, last


def squeak_crop_inputs(
    spectrograms_db: list[np.ndarray],
    first: np.ndarray,
    last: np.ndarray,
) -> np.ndarray:
    """
    Description
    -----------
    Turns squeak crops into squeak QLVM decoder inputs, step for step as
    the reference builder ``build_bbv_dataset.py`` (``--normalization per-crop``) and the
    decoder's data loader did: each crop ``spectrogram[:, first:last + 1]``
    (float32) is min-max normalized on its own,
    ``(x - min) / (max - min + 1e-6)``, written into a 128-frame zero frame from
    column 0, centred with ``stretch_specs(..., time_stretch=False)`` (the
    128 x 128 input is not resized, only the crop's columns are moved to the
    middle), and min-max normalized once more with epsilon 1e-8
    (:func:`qlvm_latents.normalize_model_inputs`).

    Parameters
    ----------
    spectrograms_db (list[np.ndarray])
        ``N`` full-length absolute-dB spectrograms, each ``(128, n_frames)``.
    first (np.ndarray)
        ``(N,)`` first crop frame of each (:func:`squeak_crop_frames`).
    last (np.ndarray)
        ``(N,)`` last crop frame of each, inclusive.

    Returns
    -------
    inputs (np.ndarray)
        ``(N, 128, 128)`` float32 decoder inputs in ``[0, 1]``.

    Raises
    ------
    ValueError
        A crop is empty or wider than the 128-frame decoder frame.
    """

    n_freq, n_time = SQUEAK_QLVM_TARGET_SHAPE
    specs = np.zeros((len(spectrograms_db), n_freq, n_time), dtype=np.float32)
    widths = np.empty(len(spectrograms_db), dtype=np.int64)
    for position, spectrogram_db in enumerate(spectrograms_db):
        crop = spectrogram_db[:, int(first[position]):int(last[position]) + 1].astype(np.float32)
        width = crop.shape[1]
        if not 1 <= width <= n_time or crop.shape[0] != n_freq:
            error_message = (
                f"squeak_crop_inputs: crop {position} is {crop.shape[0]} x {width}; the decoder frame is "
                f"{n_freq} x {n_time} and a crop must span 1 to {n_time} frames."
            )
            raise ValueError(error_message)
        low, high = float(crop.min()), float(crop.max())
        specs[position, :, :width] = (crop - low) / (high - low + SQUEAK_QLVM_CROP_EPSILON)
        widths[position] = width
    resized = stretch_specs(specs, widths, SQUEAK_QLVM_TARGET_SHAPE, False)
    return normalize_model_inputs(resized, SQUEAK_QLVM_INPUT_CONTRACT)


def squeak_qlvm_rows(usv_summary: pls.DataFrame) -> np.ndarray:
    """
    Description
    -----------
    The USV summary rows the squeak QLVM embedding considers: ``squeak`` true
    and not noise. A null ``noise`` (a segment the noise model could not score)
    counts as not noise, the single definition of noise the analyses share
    (:func:`os_utils.drop_noise_usvs`); a null ``squeak`` counts as not a squeak.

    Parameters
    ----------
    usv_summary (pls.DataFrame)
        The session's USV summary; must hold ``squeak`` and ``noise``.

    Returns
    -------
    rows (np.ndarray)
        Ascending int64 row indices.

    Raises
    ------
    ValueError
        The summary lacks ``noise`` or any squeak column.
    """

    missing = [column for column in ("noise", *SQUEAK_COLUMNS) if column not in usv_summary.columns]
    if missing:
        error_message = (
            f"The USV summary has no {missing} column(s); run detect-usv-noise and detect-usv-squeaks "
            f"on the session before embedding its squeaks."
        )
        raise ValueError(error_message)
    squeak = usv_summary["squeak"].cast(pls.Boolean).fill_null(False).to_numpy()
    noise = usv_summary["noise"].cast(pls.Boolean).fill_null(False).to_numpy()
    return np.flatnonzero(squeak & ~noise).astype(np.int64)


def squeak_qlvm_inputs(
    session_root: pathlib.Path,
    usv_summary: pls.DataFrame,
    exclude_metadata_audio_channels: bool,
    message_output: Callable,
) -> dict:
    """
    Description
    -----------
    Builds the squeak QLVM decoder inputs of one session: selects the squeak
    rows that are not noise (:func:`squeak_qlvm_rows`), rebuilds their
    full-length sonic spectrograms (:func:`squeak_segment_spectrograms`), crops
    each to its squeak extent plus context (:func:`squeak_crop_frames`), leaves
    out crops narrower than ``SQUEAK_QLVM_MIN_CROP_FRAMES`` (the training set's
    minimum) or wider than the 128-frame decoder frame (never trained on:
    the decoder frame has no room for them and the training crops were never
    compressed in time), and normalizes the rest (:func:`squeak_crop_inputs`).

    Parameters
    ----------
    session_root (pathlib.Path)
        Session root directory.
    usv_summary (pls.DataFrame)
        The session's USV summary (``start``, ``stop``, ``noise`` and the four
        squeak columns).
    exclude_metadata_audio_channels (bool)
        Whether to drop metadata-excluded channels from the spectrogram average.
    message_output (Callable)
        Logging callback.

    Returns
    -------
    squeak_inputs (dict)
        ``row_index`` (``(M,)`` int64 summary rows embedded), ``first`` /
        ``last`` (``(M,)`` int64 crop frames), ``inputs`` (``(M, 128, 128)``
        float32 decoder inputs), ``n_candidates`` (squeak rows that are not
        noise) and ``excluded`` (reason -> number of candidate rows left out:
        ``"no spectrogram"``, ``"no squeak extent"``, ``"crop < 8 frames"``,
        ``"crop > 128 frames"``).
    """

    candidates = squeak_qlvm_rows(usv_summary)
    empty = {
        "row_index": np.empty(0, dtype=np.int64),
        "first": np.empty(0, dtype=np.int64),
        "last": np.empty(0, dtype=np.int64),
        "inputs": np.empty((0, *SQUEAK_QLVM_TARGET_SHAPE), dtype=np.float32),
        "n_candidates": int(candidates.size),
    }
    excluded = {"no spectrogram": 0, "no squeak extent": 0, "crop < 8 frames": 0, "crop > 128 frames": 0}
    if candidates.size == 0:
        return {**empty, "excluded": excluded}

    spectrograms = squeak_segment_spectrograms(
        session_root=session_root,
        usv_summary=usv_summary,
        row_indices=candidates,
        exclude_metadata_audio_channels=exclude_metadata_audio_channels,
        message_output=message_output,
    )
    has_spectrogram = np.array([spectrogram is not None for spectrogram in spectrograms], dtype=bool)
    squeak_start = usv_summary["squeak_start"].cast(pls.Float64).fill_null(np.nan).to_numpy()[candidates]
    squeak_end = usv_summary["squeak_end"].cast(pls.Float64).fill_null(np.nan).to_numpy()[candidates]
    has_extent = np.isfinite(squeak_start) & np.isfinite(squeak_end)
    excluded["no spectrogram"] = int(np.count_nonzero(~has_spectrogram))
    excluded["no squeak extent"] = int(np.count_nonzero(has_spectrogram & ~has_extent))

    usable = np.flatnonzero(has_spectrogram & has_extent)
    n_frames = np.array([spectrograms[position].shape[1] for position in usable], dtype=np.int64)
    first, last = squeak_crop_frames(
        segment_start_s=usv_summary["start"].to_numpy()[candidates[usable]],
        squeak_start_s=squeak_start[usable],
        squeak_end_s=squeak_end[usable],
        n_frames=n_frames,
    )
    width = last - first + 1
    too_narrow = width < SQUEAK_QLVM_MIN_CROP_FRAMES
    too_wide = width > SQUEAK_QLVM_TARGET_SHAPE[1]
    excluded["crop < 8 frames"] = int(np.count_nonzero(too_narrow))
    excluded["crop > 128 frames"] = int(np.count_nonzero(too_wide))
    keep = ~too_narrow & ~too_wide
    if not np.any(keep):
        return {**empty, "excluded": excluded}
    return {
        "row_index": candidates[usable[keep]],
        "first": first[keep],
        "last": last[keep],
        "inputs": squeak_crop_inputs([spectrograms[position] for position in usable[keep]], first[keep], last[keep]),
        "n_candidates": int(candidates.size),
        "excluded": excluded,
    }


def load_squeak_qlvm_cell(model_cell_directory: str) -> dict:
    """
    Description
    -----------
    Loads one of the phase 3 squeak (BBV) QLVM cells
    (``qlvm_models_latest/phase3_BBVs_qlvm/<cell>``). These cells keep the OLD
    package layout: every file at the cell root, no ``config/`` or
    ``inference/`` folder and no ``training_contract.json``. What inference
    needs is read from the files they do ship instead, and checked:

    * ``checkpoint.tar`` -- the decoder weights, read without torch
      (:func:`qlvm_latents.load_decoder_params`); they must be an unconditional
      2-D decoder (first layer input width 4) of a known head;
    * ``run_config.json`` -- ``latent_dim`` must be 2, ``mask_tag``
      ``"nomask"`` and the training ``dataset`` a BBV set (name starting
      ``"bbv-"``);
    * ``manifest.json`` (the corpus embedding's record) -- its ``dataset`` must
      be unmasked (``masking_type`` ``"none"``, ``apply_mask`` false),
      ``target_shape`` ``[128, 128]`` and ``time_stretch`` false, and
      ``analysis.lattice_m`` gives the Fibonacci lattice the corpus was embedded
      on, rebuilt in float32 as the torch driver built it
      (:func:`qlvm_model.gen_fib_basis_float32`).

    Parameters
    ----------
    model_cell_directory (str)
        Path to the cell, e.g.
        ``/mnt/falkner/Dexter/vocal_beh/models/qlvm_models/qlvm_models_latest/phase3_BBVs_qlvm/natural_lumped_N11000_nomask``.

    Returns
    -------
    model (dict)
        ``params`` (decoder weights), ``lattice`` (``(fib(lattice_m), 2)``
        float32), ``lattice_m`` (int), ``head`` (``"legacy"`` or ``"relu"``) and
        ``model_id`` (``<phase>/<cell>``, the last two path components).

    Raises
    ------
    ValueError
        Any check fails; every failure is reported together.
    """

    cell = pathlib.Path(configure_path(model_cell_directory))
    with cell_file(cell, "run_config.json").open() as run_config_file:
        run_config = json.load(run_config_file)
    with cell_file(cell, "manifest.json").open() as manifest_file:
        manifest = json.load(manifest_file)
    params = load_decoder_params(str(cell / "checkpoint.tar"))
    dataset = manifest["dataset"]
    problems = []
    if run_config["latent_dim"] != 2:
        problems.append(f"run_config.json latent_dim is {run_config['latent_dim']!r}, expected 2")
    if run_config["mask_tag"] != "nomask":
        problems.append(f"run_config.json mask_tag is {run_config['mask_tag']!r}, expected 'nomask'")
    if not str(run_config["dataset"]).startswith("bbv-"):
        problems.append(f"run_config.json dataset {run_config['dataset']!r} is not a BBV (squeak) training set")
    if dataset["masking_type"] != "none" or dataset["apply_mask"]:
        problems.append(f"manifest.json dataset masking_type {dataset['masking_type']!r} / apply_mask {dataset['apply_mask']!r}, expected 'none' / false")
    if [int(value) for value in dataset["target_shape"]] != list(SQUEAK_QLVM_TARGET_SHAPE):
        problems.append(f"manifest.json dataset target_shape {dataset['target_shape']!r}, expected {list(SQUEAK_QLVM_TARGET_SHAPE)}")
    if dataset["time_stretch"]:
        problems.append("manifest.json dataset time_stretch is true, expected false")
    if int(params["0.weight"].shape[1]) != 4:
        problems.append(f"the decoder's first layer takes {int(params['0.weight'].shape[1])} inputs, expected 4 (an unconditional 2-D torus)")
    if problems:
        error_message = f"{cell} is not a squeak QLVM cell this step can embed with:\n  " + "\n  ".join(problems)
        raise ValueError(error_message)
    lattice_m = int(manifest["analysis"]["lattice_m"])
    return {
        "params": params,
        "lattice": gen_fib_basis_float32(lattice_m),
        "lattice_m": lattice_m,
        "head": decoder_head(params),
        "model_id": "/".join(cell.parts[-2:]),
    }


class USVSqueakQLVMEmbedder:
    """
    Description
    -----------
    Places every squeak of one session that is not noise on the torus of a
    squeak (BBV) QLVM cell and merges ``qlvm_squeak1`` / ``qlvm_squeak2`` into
    its ``*_usv_summary.csv``.
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
        Initializes the USVSqueakQLVMEmbedder.

        Parameters
        ----------
        root_directory (str)
            Session root directory (contains the ``audio`` tree).
        input_parameter_dict (dict)
            Processing settings; the ``infer_qlvm_squeak_latents`` block
            supplies the model cell, the channel-exclusion switch and the batch
            sizes.
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

    def embed_and_merge(self) -> None:
        """
        Description
        -----------
        Loads the cell of ``infer_qlvm_squeak_latents.model_cell_directory``
        (:func:`load_squeak_qlvm_cell`; an empty setting is filled with the
        production cell by ``os_utils.derive_spectrogram_model_paths`` and raises
        only when there is no ``spectrograms_root`` to derive it from), builds the decoder inputs of the
        session's squeaks that are not noise (:func:`squeak_qlvm_inputs`),
        embeds them as the posterior mean over the cell's lattice
        (:func:`qlvm_model.embed_data`) and writes ``qlvm_squeak1`` /
        ``qlvm_squeak2`` (torus coordinates in ``[0, 1)``) on those rows and
        nulls on every other row, replacing any earlier squeak coordinates. Run
        it after ``detect-usv-noise`` and ``detect-usv-squeaks``. The summary is
        rewritten atomically.

        Parameters
        ----------

        Returns
        -------
        Updated ``*_usv_summary.csv`` with the two squeak torus columns.
        """

        self.message_output(
            f"Squeak QLVM embedding started at: {datetime.now().hour:02d}:{datetime.now().minute:02d}:{datetime.now().second:02d}."
        )
        smart_wait(app_context_bool=self.app_context_bool, seconds=1)

        derive_spectrogram_model_paths(self.input_parameter_dict)
        cfg = self.input_parameter_dict['infer_qlvm_squeak_latents']
        if not cfg['model_cell_directory']:
            error_message = (
                "infer_qlvm_squeak_latents.model_cell_directory is empty and spectrograms_root is not set, so the "
                "production squeak QLVM cell is not filled in. Set spectrograms_root or name a "
                "qlvm_models_latest/phase3_BBVs_qlvm cell (e.g. via --model-cell-directory)."
            )
            raise ValueError(error_message)
        model = load_squeak_qlvm_cell(cfg['model_cell_directory'])
        self.message_output(
            f"{'/'.join(SQUEAK_QLVM_COLUMNS)}: squeak QLVM cell {model['model_id']} ({model['head']} head, "
            f"{model['lattice'].shape[0]}-point float32 Fibonacci lattice, m = {model['lattice_m']})."
        )

        root = pathlib.Path(self.root_directory)
        usv_summary_loc = first_match_or_raise(
            root=root / "audio",
            pattern="*_usv_summary.csv",
            recursive=True,
            label="USV summary CSV",
        )
        usv_df = pls.read_csv(source=str(usv_summary_loc), schema_overrides={"usv_id": pls.String})
        usv_df = usv_df.drop([column for column in SQUEAK_QLVM_COLUMNS if column in usv_df.columns])

        squeak_inputs = squeak_qlvm_inputs(
            session_root=root,
            usv_summary=usv_df,
            exclude_metadata_audio_channels=cfg['exclude_metadata_audio_channels'],
            message_output=self.message_output,
        )
        left_out = ", ".join(f"{count} {reason}" for reason, count in squeak_inputs['excluded'].items() if count)
        null_note = f"; null {'/'.join(SQUEAK_QLVM_COLUMNS)} for {left_out}" if left_out else ""
        self.message_output(
            f"{squeak_inputs['n_candidates']} squeaks that are not noise, {squeak_inputs['row_index'].size} embedded{null_note}."
        )

        coords = np.empty((0, 2), dtype=np.float64)
        if squeak_inputs['row_index'].size:
            coords = np.asarray(embed_data(
                model['lattice'],
                jnp.asarray(squeak_inputs['inputs'][:, None, :, :]),
                model['params'],
                cfg['lattice_batch_size'],
                cfg['data_batch_size'],
            ), dtype=np.float64).reshape(-1, 2)
        coordinate_frame = pls.DataFrame(
            {
                "_usv_row": squeak_inputs['row_index'].astype(np.uint32),
                SQUEAK_QLVM_COLUMNS[0]: coords[:, 0],
                SQUEAK_QLVM_COLUMNS[1]: coords[:, 1],
            },
            schema={"_usv_row": pls.UInt32, SQUEAK_QLVM_COLUMNS[0]: pls.Float64, SQUEAK_QLVM_COLUMNS[1]: pls.Float64},
        )
        merged = usv_df.with_row_index(name="_usv_row").join(coordinate_frame, on="_usv_row", how="left")
        merged = order_usv_summary_columns(merged.drop("_usv_row"))
        with atomic_output_path(usv_summary_loc) as tmp_summary_path:
            merged.write_csv(file=str(tmp_summary_path))

        self.message_output(
            f"Merged the squeak torus coordinates of {squeak_inputs['row_index'].size} squeaks into {usv_summary_loc.name}."
        )
        self.message_output(
            f"Squeak QLVM embedding ended at: {datetime.now().hour:02d}:{datetime.now().minute:02d}:{datetime.now().second:02d}."
        )


@click.command(name="detect-usv-squeaks")
@click.option('--root-directory', type=click.Path(exists=True, file_okay=False, dir_okay=True), required=True, help='Session root directory path.')
@click.option('--squeak-model-path', 'squeak_model_path', type=str, default=None, required=False, help='Path to the squeak classifier checkpoint (.pt); derived from spectrograms_root when empty.')
@click.option('--squeak-threshold', 'squeak_threshold', type=float, default=None, required=False, help='Decision threshold on the squeak probability, used for both the segment call and the onset / offset frames.')
@click.option('--exclude-metadata-audio-channels/--no-exclude-metadata-audio-channels', 'exclude_metadata_audio_channels', default=None, required=False, help='Drop channels the session metadata marks as excluded from the spectrogram average.')
@click.option('--batch-size', 'batch_size', type=int, default=None, required=False, help='Number of 128-frame windows scored per forward pass.')
@click.pass_context
def detect_usv_squeaks_cli(ctx, root_directory, **kwargs) -> None:
    """
    Description
    -----------
    A command-line tool to detect squeaks among a session's USV segments and
    merge the squeak columns into its USV summary CSV.

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
        block='detect_usv_squeaks',
    )

    USVSqueakDetector(
        root_directory=root_directory,
        input_parameter_dict=processing_settings_dict,
        message_output=print,
    ).detect_and_merge()


@click.command(name="infer-qlvm-squeak-latents")
@click.option('--root-directory', type=click.Path(exists=True, file_okay=False, dir_okay=True), required=True, help='Session root directory path.')
@click.option('--model-cell-directory', 'model_cell_directory', type=str, default=None, required=False, help='A squeak (BBV) QLVM cell; filled with the production phase3_BBVs_qlvm/natural_session_N11000_nomask cell when empty and spectrograms_root is set.')
@click.option('--exclude-metadata-audio-channels/--no-exclude-metadata-audio-channels', 'exclude_metadata_audio_channels', default=None, required=False, help='Drop channels the session metadata marks as excluded from the spectrogram average (keep it equal to the detect-usv-squeaks run).')
@click.option('--lattice-batch-size', 'lattice_batch_size', type=int, default=None, required=False, help='Lattice points decoded and scored per block; lower it to cut memory.')
@click.option('--data-batch-size', 'data_batch_size', type=int, default=None, required=False, help='Squeaks whose lattice posteriors are computed together; memory grows with this times the lattice size.')
@click.pass_context
def infer_qlvm_squeak_latents_cli(ctx, root_directory, **kwargs) -> None:
    """
    Description
    -----------
    A command-line tool to place a session's squeaks that are not noise on the
    torus of a squeak (BBV) QLVM cell and merge ``qlvm_squeak1`` /
    ``qlvm_squeak2`` into its USV summary CSV.

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
        block='infer_qlvm_squeak_latents',
    )

    USVSqueakQLVMEmbedder(
        root_directory=root_directory,
        input_parameter_dict=processing_settings_dict,
        message_output=print,
    ).embed_and_merge()
