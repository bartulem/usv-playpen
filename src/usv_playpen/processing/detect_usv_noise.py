"""
@author: bartulem
Flag the USV segments that hold no vocalization at all and merge the result into a session's
``*_usv_summary.csv``.

The DAS segmenter keeps every interval a channel fired on, so a session's summary also holds segments
with no call in them: electrical clicks, cage knocks, broadband transients and faint smears. This step
scores every segment with an ensemble of five time-resolved multiple-instance classifiers and writes two
columns:

* ``noise`` -- True / False in every row: True EXCLUDES the segment, because the ensemble is not
  confident it holds a vocalization (``noise_probability`` at or above the bundle's
  ``decision['exclude_at_or_above']``);
* ``noise_probability`` -- the ensemble's probability in every row, so an analysis can separate confident
  noise (at or above ``decision['noise_at_or_above']``) from the uncertain band, or re-threshold, without
  re-running this step.

A segment is NOISE only when it contains no vocalization at all -- neither a USV nor a squeak -- which is
also the rule its training labels follow, so a faint call on one channel, or a call mixed with noise, is
a vocalization, not noise.

The decision is fixed by the model bundle, not by a setting. The ensemble's probabilities split the
segments three ways -- confident vocalization, uncertain, confident noise -- and ``noise`` is True for
the last two: uncertain segments are excluded with the noise rather than guessed. The two cut-offs were
chosen as the narrowest uncertain band whose confident decisions reach precision and recall of 0.95 on
consensus-labelled segments drawn at random from the whole cohort, and the same rule applied held out
(by session) gave precision 0.944 and recall 0.955 with 0.9% of segments uncertain. The bundle's
``decision`` block records the cut-offs, those held-out numbers and the cost of the exclusion (about 71
real calls per 10,000 segments, most of them in the uncertain band); this step prints them on every run.

Input contract (fixed by the trained models, therefore constants rather than settings): for every segment
the two-band absolute-dB spectrogram (30-120 kHz and 3-30 kHz, 128 linear bins each) is rebuilt from the
UNFILTERED per-channel ``audio/hpss/*_cropped_to_video_hpss.wav`` files, read through soundfile as float64
(int16 / 32768), with a Blackman-Harris STFT (nperseg 2048, hop 512, centred), a variance-weighted
average across channels and no ``top_db`` clamp (``ref=1.0``). The audio window extends
``context_frames`` hops either side of the segment so the channel weights and the STFT edges match how
the training inputs were built, and the spectrogram is then cropped back to the segment's own frames: the
model sees the segment only. Values are mapped by the fixed affine ``(clip(x, -100, 50) + 25) / 75``.

Run it after ``das_summarize`` (re-summarizing rewrites the CSV with its base columns only) and, by
convention, before ``detect_usv_squeaks``, so the summary reads ``emitter``, the two noise columns, then
the squeak block.

The module also trains the ensemble (``train-noise-model``, :class:`USVNoiseModelTrainer`): from a labels
CSV it builds every labelled segment's input with the same function scoring uses
(:func:`window_segment_input`), trains one :class:`NoiseTimeMIL` per seed with the production recipe
(40 epochs, batch 32, Adam 1e-3 / weight decay 1e-4 under a cosine schedule, label smoothing 0.05,
gain / frequency-roll / SpecAugment augmentation) and writes a bundle this step loads unchanged, carrying
the calibration table and decision block of a supplied JSON. Training lives here rather than in its own
module so the network, the input construction and the padding exist once, shared by both directions.
"""

from __future__ import annotations

import json
import math
import pathlib
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import click
import numpy as np
import polars as pls
import soundfile as sf
import torch
from click.core import ParameterSource
from torch import nn

from ..cli_utils import modify_settings_json_for_cli
from ..os_utils import (
    configure_path,
    derive_spectrogram_model_paths,
    first_match_or_raise,
    order_usv_summary_columns,
)
from ..time_utils import is_gui_context, smart_wait

# The per-channel hpss wav lister (metadata-excluded channels dropped, rate and length checked) is
# shared with the squeak step, which introduced it; both steps read exactly the same files.
from .detect_usv_squeaks import squeak_wav_channels
from .generate_spectrograms import compute_usv_spectrogram

# Columns written into the USV summary CSV.
NOISE_COLUMNS = ("noise", "noise_probability")

# Input contract of the trained noise models.
NOISE_SAMPLING_RATE = 250000
NOISE_SPEC_BASE = {
    "num_freq_bins": 128,
    "num_time_bins": None,
    "nperseg": 2048,
    "hop_length": 512,
    "window": "blackmanharris",
}
NOISE_BANDS_HZ = ((30000.0, 120000.0), (3000.0, 30000.0))
NOISE_DB_REF = 1.0
HOP_SAMPLES = NOISE_SPEC_BASE["hop_length"]
# Frame count a `batch_size` of segments is budgeted at (the median call is ~16 frames; this is the
# model's own window), so a batch's padded size stays bounded however long the calls are.
TYPICAL_SEGMENT_FRAMES = 128

# Input transform and window every trained bundle records: dB clip range, the affine map onto [-1, 1], the
# frame cap and the context hops read either side of a segment. ``train-noise-model`` builds its inputs
# with these and writes them into the bundle; ``detect-usv-noise`` reads them back from the bundle.
NOISE_DB_FLOOR = -100.0
NOISE_DB_CEIL = 50.0
NOISE_DB_CENTER = -25.0
NOISE_DB_HALF = 75.0
NOISE_MAX_FRAMES = 512
NOISE_CONTEXT_FRAMES = 49
NOISE_INPUT_CONTRACT = {
    "bands_hz": NOISE_BANDS_HZ,
    "db_floor": NOISE_DB_FLOOR,
    "db_ceil": NOISE_DB_CEIL,
    "db_center": NOISE_DB_CENTER,
    "db_half": NOISE_DB_HALF,
    "max_frames": NOISE_MAX_FRAMES,
    "context_frames": NOISE_CONTEXT_FRAMES,
}
NOISE_SCALAR_NAMES = ("log_chs_count", "log_duration_s")

# Training augmentation of the production recipe, applied to the spectrogram channels of every training
# batch (never to the segment indicator or the padding): a per-segment gain jitter of up to +-5 dB, a
# frequency roll of up to +-2 rows, and with probability 0.5 each a time mask (up to 15% of the frames)
# and a frequency mask (1-10 rows) filled with the padding floor.
NOISE_AUG_GAIN_DB = 5.0
NOISE_AUG_FREQ_ROLL = 2
NOISE_AUG_TIME_MASK_FRACTION = 0.15
NOISE_AUG_FREQ_MASK_ROWS = 10

# Columns a training labels CSV must hold (one row per labelled segment; ``noise`` 1 = no vocalization).
NOISE_LABEL_COLUMNS = ("sample_id", "session_dir", "start", "stop", "chs_count", "noise")
# Keys the detector reads from a bundle's ``decision`` block, so a trained bundle must carry them.
NOISE_DECISION_KEYS = ("exclude_at_or_above", "noise_at_or_above", "held_out_precision", "held_out_recall", "real_calls_excluded_per_10000")


def _conv_block(in_channels: int, out_channels: int, pool: tuple[int, int] | None) -> nn.Sequential:
    """
    Description
    -----------
    One convolution block of the trained body: Conv2d 3x3 -> BatchNorm -> ReLU, optionally followed by a
    MaxPool that halves the frequency axis only (time is never pooled, so every frame keeps its own
    representation).

    Parameters
    ----------
    in_channels (int)
        Input channels.
    out_channels (int)
        Output channels.
    pool (tuple[int, int] | None)
        MaxPool kernel, or None for the last block.

    Returns
    -------
    block (nn.Sequential)
        The block, with the layer order the checkpoint's keys assume.
    """

    layers = [nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU()]
    if pool is not None:
        layers.append(nn.MaxPool2d(pool))
    return nn.Sequential(*layers)


class NoiseTimeMIL(nn.Module):
    """
    Description
    -----------
    The trained noise classifier. Four Conv2d-BatchNorm-ReLU blocks pool only along frequency, so every
    time column keeps its own representation; five residual dilated Conv1d blocks (dilations 1 to 16)
    then give each frame a receptive field of about 61 frames, so a frame is judged in the context of the
    whole call. A 1x1 frame head emits a per-frame logit, a 1x1 attention head weights the frames, and
    the segment logit is their attention-weighted sum plus a linear term in the two scalars (log channel
    count, log duration). Attribute names match the checkpoint's ``state_dict`` keys and must not change.
    """

    def __init__(self, in_channels: int, n_scalars: int, ch: int = 32) -> None:
        """
        Description
        -----------
        Builds the network.

        Parameters
        ----------
        in_channels (int)
            Input channels: the two spectrogram bands plus the segment-frame indicator.
        n_scalars (int)
            Number of per-segment scalar inputs.
        ch (int)
            Base channel width; the last block has ``4 * ch`` channels.

        Returns
        -------
        None
        """

        super().__init__()
        # One nn.Sequential per convolution block, as the training code built them: the checkpoint's keys
        # are "body.<block>.<layer>", so flattening these into one Sequential would not load.
        self.body = nn.Sequential(
            _conv_block(in_channels, ch, (2, 1)),
            _conv_block(ch, ch * 2, (2, 1)),
            _conv_block(ch * 2, ch * 4, (2, 1)),
            _conv_block(ch * 4, ch * 4, None),
            nn.AdaptiveMaxPool2d((1, None)),
        )
        self.tcn = nn.ModuleList([
            nn.Sequential(nn.Conv1d(ch * 4, ch * 4, 3, padding=dilation, dilation=dilation), nn.BatchNorm1d(ch * 4), nn.ReLU())
            for dilation in (1, 2, 4, 8, 16)
        ])
        self.head = NoiseMILHead(ch * 4, n_scalars)

    def forward(self, x: torch.Tensor, valid: torch.Tensor, segment: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        """
        Description
        -----------
        Computes the segment logit.

        Parameters
        ----------
        x (torch.Tensor)
            Normalized inputs, ``(B, C, 128, T)``.
        valid (torch.Tensor)
            Boolean frame validity (padding excluded), ``(B, T)``.
        segment (torch.Tensor)
            Boolean frames belonging to the segment itself, ``(B, T)``; pooling runs over these.
        scalars (torch.Tensor)
            Standardized scalar inputs, ``(B, n_scalars)``.

        Returns
        -------
        segment_logit (torch.Tensor)
            ``(B,)``.
        """

        h = self.body(x).squeeze(2) * valid.unsqueeze(1)
        for block in self.tcn:
            h = (h + block(h)) * valid.unsqueeze(1)
        return self.head(h, segment, scalars)


class NoiseMILHead(nn.Module):
    """
    Description
    -----------
    Per-frame logits, attention pooling over the segment's frames, and a linear scalar term added to the
    pooled logit. Attribute names match the checkpoint's ``state_dict`` keys.
    """

    def __init__(self, features: int, n_scalars: int) -> None:
        """
        Description
        -----------
        Builds the head.

        Parameters
        ----------
        features (int)
            Channel count of the frame representation.
        n_scalars (int)
            Number of per-segment scalar inputs.

        Returns
        -------
        None
        """

        super().__init__()
        self.frame = nn.Conv1d(features, 1, 1)
        self.attn = nn.Sequential(nn.Conv1d(features, 64, 1), nn.Tanh(), nn.Conv1d(64, 1, 1))
        self.scalar = nn.Linear(n_scalars, 1)

    def forward(self, h: torch.Tensor, mask: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        """
        Description
        -----------
        Pools the frame logits over the masked frames and adds the scalar term.

        Parameters
        ----------
        h (torch.Tensor)
            Frame representation, ``(B, features, T)``.
        mask (torch.Tensor)
            Frames to pool over, ``(B, T)``.
        scalars (torch.Tensor)
            Standardized scalar inputs, ``(B, n_scalars)``.

        Returns
        -------
        segment_logit (torch.Tensor)
            ``(B,)``.
        """

        frame_logit = self.frame(h).squeeze(1)
        attention = self.attn(h).squeeze(1).masked_fill(~mask, torch.finfo(frame_logit.dtype).min)
        return (torch.softmax(attention, dim=1) * frame_logit.masked_fill(~mask, 0.0)).sum(1) + self.scalar(scalars).squeeze(1)


def load_noise_model(noise_model_path: str, device: torch.device) -> dict:
    """
    Description
    -----------
    Loads the noise model bundle: the five seeds' weights in eval mode on ``device``, the scalar
    standardization and dB constants they were trained with, and the calibration table.

    Parameters
    ----------
    noise_model_path (str)
        Path to the bundle (``.pt``); derived from the spectrograms root when the setting is empty.
    device (torch.device)
        Device to load the models onto.

    Returns
    -------
    bundle (dict)
        ``models`` (list[NoiseTimeMIL]), ``scalar_mean``, ``scalar_std``, ``db_floor``, ``db_ceil``,
        ``db_center``, ``db_half``, ``max_frames``, ``context_frames``, ``bands_hz`` and ``calibration``.
    """

    path = pathlib.Path(configure_path(noise_model_path))
    if not path.is_file():
        error_message = (
            f"Noise model bundle not found: {path}. Set processing_settings['detect_usv_noise']['noise_model_path'] "
            f"or leave it empty to derive it from shared_resources['spectrograms_root']."
        )
        raise FileNotFoundError(error_message)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if [tuple(band) for band in checkpoint["bands_hz"]] != [tuple(band) for band in NOISE_BANDS_HZ]:
        error_message = f"{path} was trained on bands {checkpoint['bands_hz']}, but this module builds {list(NOISE_BANDS_HZ)}."
        raise ValueError(error_message)
    if "decision" not in checkpoint:
        error_message = (
            f"{path} carries no 'decision' block (exclusion cut-offs with their held-out precision and recall); "
            "it predates the validated decision rule. Use noise_timemil_ens5_n4680_20260926.pt or later."
        )
        raise ValueError(error_message)
    models = []
    for state_dict in checkpoint["state_dicts"]:
        model = NoiseTimeMIL(in_channels=checkpoint["in_channels"], n_scalars=checkpoint["n_scalars"]).to(device)
        model.load_state_dict(state_dict)
        model.eval()
        models.append(model)
    return {
        "models": models,
        "scalar_mean": np.asarray(checkpoint["scalar_mean"], dtype=np.float32),
        "scalar_std": np.asarray(checkpoint["scalar_std"], dtype=np.float32),
        "db_floor": float(checkpoint["db_floor"]), "db_ceil": float(checkpoint["db_ceil"]),
        "db_center": float(checkpoint["db_center"]), "db_half": float(checkpoint["db_half"]),
        "max_frames": int(checkpoint["max_frames"]), "context_frames": int(checkpoint["context_frames"]),
        "bands_hz": NOISE_BANDS_HZ, "calibration": checkpoint["calibration"],
        "decision": checkpoint["decision"],
    }


def segment_input(
    audio_window: np.ndarray,
    first_frame: int,
    n_frames: int,
    bundle: dict,
) -> tuple[np.ndarray, int] | tuple[None, int]:
    """
    Description
    -----------
    Builds one segment's model input: the two-band absolute-dB spectrogram of the context window
    (variance-weighted across channels), cropped to the segment's own frames, mapped through the fixed
    affine transform, with the segment-indicator channel appended.

    Parameters
    ----------
    audio_window (np.ndarray)
        ``(samples, channels)`` audio of the context window.
    first_frame (int)
        Index of the segment's first frame within the window's spectrogram.
    n_frames (int)
        The segment's own frame count (its stored ``duration`` in frames).
    bundle (dict)
        Loaded model bundle (dB constants and the frame cap).

    Returns
    -------
    x (np.ndarray | None)
        ``(C, 128, T)`` float32 input, or None when the window is too short for one STFT window.
    n_frames_used (int)
        Frames kept (after the cap), 0 when unscorable.
    """

    bands = []
    for low, high in bundle["bands_hz"]:
        band, _ = compute_usv_spectrogram(
            audio_window, NOISE_SAMPLING_RATE, {**NOISE_SPEC_BASE, "min_freq": low, "max_freq": high},
            normalize=False, db_ref=NOISE_DB_REF, top_db=None,
        )
        if band is None:
            return None, 0
        bands.append(band)
    spectrogram = np.stack(bands).astype(np.float32)
    n_used = min(n_frames, spectrogram.shape[2] - first_frame, bundle["max_frames"])
    if n_used < 1:
        return None, 0
    crop = spectrogram[:, :, first_frame:first_frame + n_used]
    normalized = (np.clip(crop, bundle["db_floor"], bundle["db_ceil"]) - bundle["db_center"]) / bundle["db_half"]
    indicator = np.ones((1, normalized.shape[1], normalized.shape[2]), dtype=np.float32)
    return np.concatenate([normalized, indicator], axis=0).astype(np.float32), n_used


def window_segment_input(
    handles: list[sf.SoundFile],
    n_file: int,
    start: float,
    stop: float,
    bundle: dict,
) -> tuple[np.ndarray, int] | tuple[None, int]:
    """
    Description
    -----------
    Reads one segment's audio window from the open per-channel wavs and builds its model input. The
    window starts exactly ``context_frames`` hops before the segment's first sample (fewer at the start of
    a recording) and ends ``context_frames`` hops after its last sample (or at the end of the file), so
    with the centred STFT the segment occupies frames ``first_frame .. first_frame + n_frames - 1`` of the
    window's spectrogram, ``n_frames`` being ``1 + (ceil(stop * fs) - floor(start * fs)) // hop`` (the
    stored-duration convention). Scoring (``detect-usv-noise``) and training (``train-noise-model``) both
    build their inputs here, so a trained bundle sees exactly the input it is later scored on.

    Parameters
    ----------
    handles (list[sf.SoundFile])
        Open per-channel wavs (one per averaged channel, all of one length).
    n_file (int)
        Frame count (samples) of the wavs.
    start (float)
        Segment start (s).
    stop (float)
        Segment stop (s).
    bundle (dict)
        Loaded model bundle, or ``NOISE_INPUT_CONTRACT`` when training (``bands_hz``, the dB constants,
        ``max_frames`` and ``context_frames`` are read).

    Returns
    -------
    x (np.ndarray | None)
        ``(C, 128, T)`` float32 input, or None when the window is too short for one STFT window.
    n_frames_used (int)
        Frames kept (after the cap), 0 when unscorable.
    """

    context = bundle["context_frames"]
    first_sample = math.floor(start * NOISE_SAMPLING_RATE)
    last_sample = math.ceil(stop * NOISE_SAMPLING_RATE)
    first_frame = min(context, first_sample // HOP_SAMPLES)
    read_start = first_sample - first_frame * HOP_SAMPLES
    read_stop = min(n_file, last_sample + context * HOP_SAMPLES)
    channels = []
    for handle in handles:
        handle.seek(read_start)
        channels.append(handle.read(frames=read_stop - read_start, dtype="float64", always_2d=False))
    window = np.stack(channels, axis=1)
    return segment_input(window, first_frame, 1 + (last_sample - first_sample) // HOP_SAMPLES, bundle)


def noise_scalars(chs_count: float, start: float, stop: float) -> np.ndarray:
    """
    Description
    -----------
    Builds one segment's raw scalar inputs, in ``NOISE_SCALAR_NAMES`` order: the log of the number of
    channels DAS detected the segment on, and the log of its duration in seconds. They are standardized
    with the bundle's ``scalar_mean`` / ``scalar_std`` before entering the model.

    Parameters
    ----------
    chs_count (float)
        The summary's ``chs_count`` (at least 1).
    start (float)
        Segment start (s).
    stop (float)
        Segment stop (s).

    Returns
    -------
    scalars (np.ndarray)
        ``(2,)`` float32 ``[log(chs_count), log(stop - start)]``.
    """

    return np.array([np.log(chs_count), np.log(stop - start)], dtype=np.float32)


def pad_noise_batch(chunk: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """
    Description
    -----------
    Pads a batch of segment inputs to its longest segment: spectrogram channels are padded with -1 (the
    normalized floor), the segment-indicator channel with 0, and a frame-validity mask marks the real
    frames. Scoring and training pad identically.

    Parameters
    ----------
    chunk (list[np.ndarray])
        Per-segment ``(C, 128, T)`` inputs.

    Returns
    -------
    x (np.ndarray)
        ``(B, C, 128, T_max)`` float32 batch.
    valid (np.ndarray)
        ``(B, T_max)`` boolean frame validity.
    """

    width = max(item.shape[2] for item in chunk)
    x = np.full((len(chunk), chunk[0].shape[0], chunk[0].shape[1], width), -1.0, dtype=np.float32)
    x[:, -1] = 0.0
    valid = np.zeros((len(chunk), width), dtype=bool)
    for position, item in enumerate(chunk):
        x[position, :, :, :item.shape[2]] = item
        valid[position, :item.shape[2]] = True
    return x, valid


def ensemble_noise_probability(
    models: list[NoiseTimeMIL],
    inputs: list[np.ndarray],
    standardized_scalars: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """
    Description
    -----------
    Scores segment inputs with every model of an ensemble and returns the mean of their probabilities.
    Segments are grouped into frame-budgeted batches (see :func:`frame_budget_batches`) and the model
    pools over every valid frame, which is the segment itself.

    Parameters
    ----------
    models (list[NoiseTimeMIL])
        Ensemble members, in eval mode on ``device``.
    inputs (list[np.ndarray])
        Per-segment ``(C, 128, T)`` inputs.
    standardized_scalars (np.ndarray)
        ``(N, n_scalars)`` float32 scalars, already standardized with the bundle's mean and std.
    batch_size (int)
        Segments per forward pass at the typical call length.
    device (torch.device)
        Scoring device.

    Returns
    -------
    probability (np.ndarray)
        ``(N,)`` ensemble-mean noise probability.
    """

    runs = []
    for model in models:
        out = []
        with torch.no_grad():
            for start_index, stop_index in frame_budget_batches(inputs, batch_size):
                x, valid = pad_noise_batch(inputs[start_index:stop_index])
                x_tensor = torch.from_numpy(x).to(device)
                valid_tensor = torch.from_numpy(valid).to(device)
                scalar_tensor = torch.tensor(np.asarray(standardized_scalars[start_index:stop_index], dtype=np.float32), device=device)
                out.append(torch.sigmoid(model(x_tensor, valid_tensor, valid_tensor, scalar_tensor)).cpu().numpy())
        runs.append(np.concatenate(out))
    return np.mean(runs, axis=0)


def frame_budget_batches(inputs: list[np.ndarray], batch_size: int) -> list[tuple[int, int]]:
    """
    Description
    -----------
    Splits the session's segments into consecutive batches that hold at most ``batch_size *
    TYPICAL_SEGMENT_FRAMES`` frame slots, a batch always taking at least one segment. Every batch is
    padded to its longest segment, so counting segments alone is not enough: one 512-frame call among
    255 short ones would inflate the batch 30-fold and exhaust the GPU (measured: a cohort run with
    eight sessions in parallel failed on batches asking for 1.8 GiB at ``batch_size`` 256). Budgeting
    frames keeps a batch's memory roughly constant whatever the call lengths are.

    Parameters
    ----------
    inputs (list[np.ndarray])
        Per-segment ``(C, 128, T)`` inputs, in summary order.
    batch_size (int)
        Segments per forward pass at the typical call length; the frame budget is this times
        ``TYPICAL_SEGMENT_FRAMES``.

    Returns
    -------
    batches (list[tuple[int, int]])
        ``(start, stop)`` index pairs covering ``inputs`` in order.
    """

    budget = max(1, batch_size) * TYPICAL_SEGMENT_FRAMES
    batches: list[tuple[int, int]] = []
    start = 0
    width = 0
    for position, item in enumerate(inputs):
        candidate_width = max(width, item.shape[2])
        if position > start and (position - start + 1) * candidate_width > budget:
            batches.append((start, position))
            start, width = position, item.shape[2]
        else:
            width = candidate_width
    batches.append((start, len(inputs)))
    return batches


def score_noise_rows(
    session_root: pathlib.Path,
    usv_summary: pls.DataFrame,
    bundle: dict,
    device: torch.device,
    threshold: float,
    exclude_metadata_audio_channels: bool,
    batch_size: int,
    message_output: Callable,
) -> pls.DataFrame:
    """
    Description
    -----------
    Scores every row of one session's USV summary with the ensemble and returns the two noise columns.
    Segments too short for one STFT window get a null probability and ``noise`` False.

    Parameters
    ----------
    session_root (pathlib.Path)
        Session root directory.
    usv_summary (pls.DataFrame)
        The session's USV summary (``start``, ``stop``, ``chs_count`` are read).
    bundle (dict)
        Loaded model bundle.
    device (torch.device)
        Scoring device.
    threshold (float)
        Decision threshold on the ensemble probability.
    exclude_metadata_audio_channels (bool)
        Drop channels the session metadata marks as excluded from the average.
    batch_size (int)
        Segments per forward pass at the typical call length (see :func:`frame_budget_batches`).
    message_output (Callable)
        Logging callback.

    Returns
    -------
    scores (pls.DataFrame)
        ``noise`` (bool) and ``noise_probability`` (float, null when unscorable), one row per summary row.
    """

    wav_paths = squeak_wav_channels(session_root, exclude_metadata_audio_channels, message_output)
    handles = [sf.SoundFile(str(path), mode="r") for path in wav_paths]
    n_file = handles[0].frames
    inputs: list[np.ndarray] = []
    scalars: list[np.ndarray] = []
    scored_rows: list[int] = []
    try:
        for row_index in range(usv_summary.height):
            start = float(usv_summary["start"][row_index])
            stop = float(usv_summary["stop"][row_index])
            x, _n_used = window_segment_input(handles, n_file, start, stop, bundle)
            if x is None:
                continue
            inputs.append(x)
            scalars.append(noise_scalars(float(usv_summary["chs_count"][row_index]), start, stop))
            scored_rows.append(row_index)
    finally:
        for handle in handles:
            handle.close()

    probability = np.full(usv_summary.height, np.nan, dtype=np.float64)
    if scored_rows:
        standardized = (np.stack(scalars) - bundle["scalar_mean"]) / bundle["scalar_std"]
        probability[np.asarray(scored_rows)] = ensemble_noise_probability(bundle["models"], inputs, standardized, batch_size, device)
    unscorable = usv_summary.height - len(scored_rows)
    if unscorable:
        message_output(f"{unscorable} USV(s) are too short for one STFT window and get an empty noise probability.")
    return pls.DataFrame({
        "noise": np.nan_to_num(probability, nan=0.0) >= threshold,
        "noise_probability": probability,
    }, schema={"noise": pls.Boolean, "noise_probability": pls.Float64})


def read_noise_labels(labels_csv_path: str, message_output: Callable) -> pls.DataFrame:
    """
    Description
    -----------
    Reads and checks a training labels CSV: one row per training example with ``sample_id`` (the
    segment's name), ``session_dir`` (session root), ``start`` / ``stop`` (s, as in the session's USV
    summary), ``chs_count`` (the summary's channel count) and ``noise`` (1 = the segment holds no
    vocalization at all, neither a USV nor a squeak; 0 = it holds one). The segment is defined by these
    columns rather than by a summary row, so a later re-summarization cannot silently change what a label
    refers to. Unsure answers must be dropped (or resolved) before training; any ``noise`` other than 0
    or 1 stops the run. A ``sample_id`` may repeat (a segment labelled in two rounds is then trained on
    twice, as the production ensemble was: 43 of its 4,680 rows repeat a segment, two of them with the
    opposite label); repeats and conflicting repeats are counted and reported, not removed.

    Parameters
    ----------
    labels_csv_path (str)
        Path to the labels CSV.
    message_output (Callable)
        Logging callback.

    Returns
    -------
    labels (pls.DataFrame)
        The six columns, typed (``noise`` as Int64), in file order; row order is training order.
    """

    path = pathlib.Path(configure_path(labels_csv_path))
    if not path.is_file():
        error_message = f"Noise training labels not found: {path}."
        raise FileNotFoundError(error_message)
    labels = pls.read_csv(str(path), schema_overrides={"sample_id": pls.String, "session_dir": pls.String}, infer_schema_length=None)
    missing = [column for column in NOISE_LABEL_COLUMNS if column not in labels.columns]
    if missing:
        error_message = f"{path} lacks the column(s) {missing}; a labels CSV holds {list(NOISE_LABEL_COLUMNS)}."
        raise ValueError(error_message)
    labels = labels.select(
        pls.col("sample_id"), pls.col("session_dir"),
        pls.col("start").cast(pls.Float64), pls.col("stop").cast(pls.Float64),
        pls.col("chs_count").cast(pls.Float64), pls.col("noise").cast(pls.Int64),
    )
    if labels.height == 0:
        error_message = f"{path} holds no labelled segments."
        raise ValueError(error_message)
    if labels.null_count().sum_horizontal()[0] > 0:
        error_message = f"{path} has empty cells in {list(NOISE_LABEL_COLUMNS)}; drop unsure or incomplete rows first."
        raise ValueError(error_message)
    if labels["sample_id"].n_unique() != labels.height:
        conflicting = labels.group_by("sample_id").agg(pls.col("noise").n_unique().alias("n_labels")).filter(pls.col("n_labels") > 1).height
        message_output(
            f"{labels.height - labels['sample_id'].n_unique()} row(s) repeat an earlier sample_id ({conflicting} segment(s) with "
            f"conflicting labels); every row is trained on as given."
        )
    if not labels["noise"].is_in([0, 1]).all():
        error_message = f"{path}: noise must be 0 (vocalization) or 1 (noise); drop unsure answers (e.g. 2) first."
        raise ValueError(error_message)
    if labels.filter(pls.col("noise") == 1).height == 0 or labels.filter(pls.col("noise") == 0).height == 0:
        error_message = f"{path} needs both noise (1) and vocalization (0) labels."
        raise ValueError(error_message)
    if not ((labels["stop"] > labels["start"]).all() and (labels["start"] >= 0).all() and (labels["chs_count"] >= 1).all()):
        error_message = f"{path}: every row needs 0 <= start < stop and chs_count >= 1."
        raise ValueError(error_message)
    return labels


def read_noise_calibration(calibration_path: str) -> dict:
    """
    Description
    -----------
    Reads the calibration a trained bundle carries and checks it holds what ``detect-usv-noise`` reads.
    The JSON holds ``calibration`` (a list of rows, each with at least ``threshold``, ``precision`` and
    ``recall``: the cross-fitted precision / recall of this training procedure at every threshold) and
    ``decision`` (the exclusion cut-offs ``exclude_at_or_above`` <= ``noise_at_or_above`` plus their
    held-out ``held_out_precision``, ``held_out_recall`` and ``real_calls_excluded_per_10000``, and any
    further provenance keys). ``calibration_source`` and ``calibration_population`` are copied into the
    bundle when present. Training does not measure these: they come from the session-grouped cross-fit
    and the cut-off selection on consensus-labelled segments, so they describe the bundle only when it
    is trained with the same labels and recipe they were measured for.

    Parameters
    ----------
    calibration_path (str)
        Path to the calibration JSON.

    Returns
    -------
    calibration (dict)
        ``calibration``, ``decision`` and, when present, ``calibration_source`` / ``calibration_population``.
    """

    path = pathlib.Path(configure_path(calibration_path))
    if not path.is_file():
        error_message = (
            f"Noise calibration JSON not found: {path}. Set processing_settings['train_noise_model']['calibration_path'] "
            f"to a JSON holding the 'calibration' table and the 'decision' block the bundle will carry."
        )
        raise FileNotFoundError(error_message)
    content = json.loads(path.read_text(encoding="utf-8"))
    for key in ("calibration", "decision"):
        if key not in content:
            error_message = f"{path} lacks '{key}'; the detector reads both the calibration table and the decision block."
            raise ValueError(error_message)
    if not content["calibration"] or any(key not in row for row in content["calibration"] for key in ("threshold", "precision", "recall")):
        error_message = f"{path}: 'calibration' must be a non-empty list of rows with threshold, precision and recall."
        raise ValueError(error_message)
    missing = [key for key in NOISE_DECISION_KEYS if key not in content["decision"]]
    if missing:
        error_message = f"{path}: the 'decision' block lacks {missing}, which detect-usv-noise reads."
        raise ValueError(error_message)
    decision = content["decision"]
    if not 0.0 < decision["exclude_at_or_above"] <= decision["noise_at_or_above"] <= 1.0:
        error_message = f"{path}: the decision needs 0 < exclude_at_or_above <= noise_at_or_above <= 1."
        raise ValueError(error_message)
    calibration = {"calibration": content["calibration"], "decision": decision}
    for key in ("calibration_source", "calibration_population"):
        if key in content:
            calibration[key] = content[key]
    return calibration


def session_noise_training_inputs(
    session_dir: str,
    segments: list[tuple[int, float, float, float]],
    exclude_metadata_audio_channels: bool,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """
    Description
    -----------
    Builds the model inputs and raw scalars of one session's labelled segments, opening the session's
    per-channel wavs once. Run in a worker thread by :func:`build_noise_training_inputs`.

    Parameters
    ----------
    session_dir (str)
        Session root directory.
    segments (list[tuple[int, float, float, float]])
        ``(label row, start, stop, chs_count)`` of every labelled segment of the session.
    exclude_metadata_audio_channels (bool)
        Drop channels the session metadata marks as excluded from the average.

    Returns
    -------
    built (dict[int, tuple[np.ndarray, np.ndarray]])
        Label row -> ``(input, raw scalars)``; segments too short for one STFT window are absent.
    """

    wav_paths = squeak_wav_channels(pathlib.Path(configure_path(session_dir)), exclude_metadata_audio_channels, lambda *_args, **_kwargs: None)
    handles = [sf.SoundFile(str(path), mode="r") for path in wav_paths]
    built = {}
    try:
        for row, start, stop, chs_count in segments:
            x, _n_used = window_segment_input(handles, handles[0].frames, start, stop, NOISE_INPUT_CONTRACT)
            if x is not None:
                built[row] = (x, noise_scalars(chs_count, start, stop))
    finally:
        for handle in handles:
            handle.close()
    return built


def build_noise_training_inputs(
    labels: pls.DataFrame,
    exclude_metadata_audio_channels: bool,
    n_workers: int,
    message_output: Callable,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """
    Description
    -----------
    Builds the model input of every labelled segment exactly as ``detect-usv-noise`` builds it at
    inference (:func:`window_segment_input` with ``NOISE_INPUT_CONTRACT``, over the same per-channel
    ``audio/hpss`` wavs and channel exclusion), plus its raw scalars. Sessions are processed in
    ``n_workers`` threads, each session's wavs opened once: the work is dominated by the latency of
    reading short windows from 24 wavs on a network share (measured ~44 ms per read, ~14 s per session
    serially), which threads overlap. Results are collected by label row, so the output order does not
    depend on the thread schedule. Segments too short for one STFT window are left out and reported.

    Parameters
    ----------
    labels (pls.DataFrame)
        Checked labels (:func:`read_noise_labels`).
    exclude_metadata_audio_channels (bool)
        Drop channels the session metadata marks as excluded from the average (as the detector does).
    n_workers (int)
        Sessions read concurrently (threads).
    message_output (Callable)
        Logging callback.

    Returns
    -------
    inputs (list[np.ndarray])
        Per kept segment, the ``(3, 128, T)`` float32 input, in label order.
    raw_scalars (np.ndarray)
        ``(N_kept, 2)`` float32 unstandardized scalars.
    kept_rows (np.ndarray)
        Indices into ``labels`` of the kept segments.
    """

    segments_by_session: dict[str, list[tuple[int, float, float, float]]] = {}
    for row, (session_dir, start, stop, chs_count) in enumerate(labels.select("session_dir", "start", "stop", "chs_count").iter_rows()):
        segments_by_session.setdefault(session_dir, []).append((row, float(start), float(stop), float(chs_count)))
    built: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    with ThreadPoolExecutor(max_workers=max(1, n_workers)) as pool:
        futures = [pool.submit(session_noise_training_inputs, session_dir, segments, exclude_metadata_audio_channels)
                   for session_dir, segments in segments_by_session.items()]
        for session_number, future in enumerate(as_completed(futures), start=1):
            built.update(future.result())
            if session_number % 50 == 0 or session_number == len(futures):
                message_output(f"Built noise training inputs for {session_number}/{len(futures)} session(s), {len(built)} segment(s).")
    inputs_by_row = {row: item[0] for row, item in built.items()}
    scalars_by_row = {row: item[1] for row, item in built.items()}
    kept_rows = np.array(sorted(inputs_by_row), dtype=np.int64)
    if kept_rows.size < labels.height:
        dropped = sorted(set(range(labels.height)) - set(kept_rows.tolist()))
        message_output(f"{len(dropped)} labelled segment(s) are too short for one STFT window and are left out: {[labels['sample_id'][row] for row in dropped]}.")
    return [inputs_by_row[row] for row in kept_rows], np.stack([scalars_by_row[row] for row in kept_rows]), kept_rows


def augment_noise_batch(x: torch.Tensor, valid: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """
    Description
    -----------
    Augments one training batch with the production recipe, on the spectrogram channels only (the last,
    segment-indicator channel is passed through): a per-segment gain jitter of up to
    ``NOISE_AUG_GAIN_DB`` (clamped to [-1, 1], valid frames only), a frequency roll of up to
    ``NOISE_AUG_FREQ_ROLL`` rows, and with probability 0.5 each a time mask of up to
    ``NOISE_AUG_TIME_MASK_FRACTION`` of the segment's frames (segments longer than four frames) and a
    frequency mask of 1 to ``NOISE_AUG_FREQ_MASK_ROWS`` rows over the valid frames, both filled with the
    padding floor (-1). Every random draw comes from ``generator`` (on the CPU), in a fixed order, so a
    seed reproduces the draws.

    Parameters
    ----------
    x (torch.Tensor)
        ``(B, C, 128, T)`` normalized batch; the last channel is the segment indicator.
    valid (torch.Tensor)
        ``(B, T)`` boolean frame validity.
    generator (torch.Generator)
        Seeded CPU generator.

    Returns
    -------
    x (torch.Tensor)
        The augmented batch (a new tensor).
    """

    spectra = x[:, :-1].clone()
    batch, _channels, rows, _width = spectra.shape
    gain = ((torch.rand(batch, generator=generator) * 2 - 1) * NOISE_AUG_GAIN_DB / NOISE_DB_HALF).to(x.device)
    spectra = torch.where(valid[:, None, None, :], (spectra + gain[:, None, None, None]).clamp(-1.0, 1.0), spectra)
    for i in range(batch):
        shift = int(torch.randint(-NOISE_AUG_FREQ_ROLL, NOISE_AUG_FREQ_ROLL + 1, (1,), generator=generator))
        if shift:
            spectra[i] = torch.roll(spectra[i], shifts=shift, dims=1)
        n_frames = int(valid[i].sum())
        if float(torch.rand(1, generator=generator)) < 0.5 and n_frames > 4:
            length = max(1, int(n_frames * NOISE_AUG_TIME_MASK_FRACTION * float(torch.rand(1, generator=generator))))
            start = int(torch.randint(0, n_frames - length + 1, (1,), generator=generator))
            spectra[i, :, :, start:start + length] = -1.0
        if float(torch.rand(1, generator=generator)) < 0.5:
            length = int(torch.randint(1, NOISE_AUG_FREQ_MASK_ROWS + 1, (1,), generator=generator))
            start = int(torch.randint(0, rows - length + 1, (1,), generator=generator))
            spectra[i, :, start:start + length, :n_frames] = -1.0
    return torch.cat([spectra, x[:, -1:]], dim=1)


def train_noise_seed(
    inputs: list[np.ndarray],
    standardized_scalars: np.ndarray,
    y: np.ndarray,
    seed: int,
    recipe: dict,
    device: torch.device,
    message_output: Callable,
) -> NoiseTimeMIL:
    """
    Description
    -----------
    Trains one ensemble member with the production recipe: the network is initialized after
    ``torch.manual_seed(seed)``; every epoch visits the segments in a ``numpy.random.default_rng(seed)``
    permutation in padded batches of ``recipe['batch_size']``, augmented by :func:`augment_noise_batch`
    (a CPU generator seeded with ``seed``); targets are label-smoothed towards 0.5 by
    ``recipe['label_smoothing']``; the loss is binary cross-entropy on the segment logit (pooled over the
    segment's frames), minimized by Adam (``learning_rate``, ``weight_decay``) under a cosine learning-rate
    schedule over all ``recipe['epochs']`` epochs, stepped once per batch.

    Parameters
    ----------
    inputs (list[np.ndarray])
        Per-segment ``(3, 128, T)`` inputs.
    standardized_scalars (np.ndarray)
        ``(N, 2)`` float32 scalars standardized over the training set.
    y (np.ndarray)
        ``(N,)`` float32 labels (1 = noise).
    seed (int)
        Seed of this member.
    recipe (dict)
        ``epochs``, ``batch_size``, ``learning_rate``, ``weight_decay`` and ``label_smoothing``.
    device (torch.device)
        Training device.
    message_output (Callable)
        Logging callback.

    Returns
    -------
    model (NoiseTimeMIL)
        The trained model, in eval mode on ``device``.
    """

    torch.manual_seed(seed)
    model = NoiseTimeMIL(in_channels=inputs[0].shape[0], n_scalars=standardized_scalars.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=recipe['learning_rate'], weight_decay=recipe['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=recipe['epochs'] * math.ceil(len(inputs) / recipe['batch_size']))
    loss_fn = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(seed)
    generator = torch.Generator().manual_seed(seed)
    smoothing = recipe['label_smoothing']
    for epoch in range(recipe['epochs']):
        model.train()
        order = rng.permutation(len(inputs))
        losses = []
        for start in range(0, len(order), recipe['batch_size']):
            chunk = order[start:start + recipe['batch_size']]
            x, valid = pad_noise_batch([inputs[i] for i in chunk])
            x_tensor = torch.from_numpy(x).to(device)
            valid_tensor = torch.from_numpy(valid).to(device)
            scalar_tensor = torch.tensor(standardized_scalars[chunk], device=device)
            target = torch.tensor(y[chunk], dtype=torch.float32, device=device) * (1 - smoothing) + 0.5 * smoothing
            x_tensor = augment_noise_batch(x_tensor, valid_tensor, generator)
            optimizer.zero_grad()
            loss = loss_fn(model(x_tensor, valid_tensor, valid_tensor, scalar_tensor), target)
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach()))
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == recipe['epochs']:
            message_output(f"  seed {seed}: epoch {epoch + 1}/{recipe['epochs']}, mean training loss {np.mean(losses):.4f}.")
    model.eval()
    return model


class USVNoiseDetector:
    """
    Description
    -----------
    Scores every USV segment of one session for noise and merges the two noise columns into its
    ``*_usv_summary.csv``.
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
        Initializes the USVNoiseDetector.

        Parameters
        ----------
        root_directory (str)
            Session root directory (contains the ``audio`` tree).
        input_parameter_dict (dict)
            Processing settings; the ``detect_usv_noise`` block supplies the model path, the minimum
            precision, the channel-exclusion switch and the batch size.
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
        Loads the model bundle, takes the exclusion threshold from its ``decision`` block, scores every
        row of the session's USV summary and writes ``noise`` and ``noise_probability`` into it,
        replacing any existing noise columns. Run it after ``das_summarize``: re-summarizing rewrites the CSV with its
        base columns only and would remove these.

        Parameters
        ----------

        Returns
        -------
        Updated ``*_usv_summary.csv`` with the two noise columns.
        """

        self.message_output(
            f"USV noise detection started at: {datetime.now().hour:02d}:{datetime.now().minute:02d}:{datetime.now().second:02d}."
        )
        smart_wait(app_context_bool=self.app_context_bool, seconds=1)

        derive_spectrogram_model_paths(self.input_parameter_dict)
        cfg = self.input_parameter_dict['detect_usv_noise']
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bundle = load_noise_model(cfg['noise_model_path'], device)
        decision = bundle['decision']
        self.message_output(
            f"noise = p >= {decision['exclude_at_or_above']:.2f}: confident noise (p >= {decision['noise_at_or_above']:.2f}) "
            f"plus the uncertain band, which is excluded rather than guessed. Held out, the confident decisions "
            f"reach precision {decision['held_out_precision']:.3f} and recall {decision['held_out_recall']:.3f}; "
            f"the exclusion costs about {decision['real_calls_excluded_per_10000']:.0f} real calls per 10,000 segments."
        )

        root = pathlib.Path(self.root_directory)
        usv_summary_loc = first_match_or_raise(
            root=root / "audio",
            pattern="*_usv_summary.csv",
            recursive=True,
            label="USV summary CSV",
        )
        usv_df = pls.read_csv(source=str(usv_summary_loc), schema_overrides={"usv_id": pls.String})
        usv_df = usv_df.drop([column for column in NOISE_COLUMNS if column in usv_df.columns])

        scores = score_noise_rows(
            session_root=root,
            usv_summary=usv_df,
            bundle=bundle,
            device=device,
            threshold=decision['exclude_at_or_above'],
            exclude_metadata_audio_channels=cfg['exclude_metadata_audio_channels'],
            batch_size=cfg['batch_size'],
            message_output=self.message_output,
        )
        merged = order_usv_summary_columns(pls.concat([usv_df, scores], how="horizontal"))
        merged.write_csv(file=str(usv_summary_loc))

        self.message_output(
            f"Merged noise calls into {usv_summary_loc.name}: {int(scores['noise'].sum())} noise segment(s) among {usv_df.height} USVs."
        )
        self.message_output(
            f"USV noise detection ended at: {datetime.now().hour:02d}:{datetime.now().minute:02d}:{datetime.now().second:02d}."
        )


class USVNoiseModelTrainer:
    """
    Description
    -----------
    Trains the noise-model ensemble on a labels CSV and writes a bundle ``detect-usv-noise`` loads
    unchanged.
    """

    def __init__(
        self,
        labels_csv_path: str | None = None,
        bundle_path: str | None = None,
        input_parameter_dict: dict | None = None,
        message_output: Callable | None = None,
    ) -> None:
        """
        Description
        -----------
        Initializes the USVNoiseModelTrainer.

        Parameters
        ----------
        labels_csv_path (str)
            Labels CSV (see :func:`read_noise_labels`).
        bundle_path (str)
            Output bundle (``.pt``); must not exist yet.
        input_parameter_dict (dict)
            Processing settings; the ``train_noise_model`` block supplies the calibration JSON, the
            channel-exclusion switch, the seeds and the recipe.
        message_output (Callable)
            Logging callback; defaults to ``print``.

        Returns
        -------
        None
        """

        self.labels_csv_path = labels_csv_path
        self.bundle_path = bundle_path
        self.input_parameter_dict = input_parameter_dict if input_parameter_dict is not None else {}
        self.message_output = message_output if message_output is not None else print

    def train(self) -> pathlib.Path:
        """
        Description
        -----------
        Reads and checks the labels and the calibration JSON (before any work, so a bad file fails
        fast), builds every labelled segment's input with the detector's own extraction, standardizes
        the scalars over the training set, trains one model per seed with the production recipe (cuDNN
        in deterministic mode for the run; a GPU is still not bit-reproducible, so a retrained ensemble
        matches an earlier one in its decisions, not in its weights), writes the bundle with the input
        contract, the calibration, the decision and the provenance, and loads it back through
        :func:`load_noise_model` to prove ``detect-usv-noise`` accepts it.

        Parameters
        ----------

        Returns
        -------
        bundle_path (pathlib.Path)
            The written bundle.
        """

        start_time = datetime.now()
        self.message_output(f"Noise model training started at: {start_time.hour:02d}:{start_time.minute:02d}:{start_time.second:02d}.")
        cfg = self.input_parameter_dict['train_noise_model']
        bundle_path = pathlib.Path(configure_path(self.bundle_path))
        if bundle_path.exists():
            error_message = f"{bundle_path} already exists; choose a new bundle path (a trained bundle is never overwritten)."
            raise FileExistsError(error_message)
        if not cfg['seeds'] or len(set(cfg['seeds'])) != len(cfg['seeds']):
            error_message = f"train_noise_model['seeds'] must list distinct seeds, one per ensemble member; got {cfg['seeds']}."
            raise ValueError(error_message)
        labels = read_noise_labels(self.labels_csv_path, self.message_output)
        calibration = read_noise_calibration(cfg['calibration_path'])
        self.message_output(
            f"{labels.height} labelled segment(s) ({int(labels['noise'].sum())} noise) from {labels['session_dir'].n_unique()} session(s)."
        )

        inputs, raw_scalars, kept_rows = build_noise_training_inputs(labels, cfg['exclude_metadata_audio_channels'], cfg['n_workers'], self.message_output)
        y = labels["noise"].to_numpy()[kept_rows].astype(np.float32)
        scalar_mean = raw_scalars.mean(0)
        scalar_std = raw_scalars.std(0) + 1e-6
        standardized = (raw_scalars - scalar_mean) / scalar_std
        recipe = {
            "epochs": cfg['epochs'], "batch_size": cfg['batch_size'], "learning_rate": cfg['learning_rate'],
            "weight_decay": cfg['weight_decay'], "label_smoothing": cfg['label_smoothing'],
        }

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.message_output(f"Training {len(cfg['seeds'])} model(s) on {len(inputs)} segment(s) ({int(y.sum())} noise) on {device}.")
        deterministic, benchmark = torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        state_dicts = []
        try:
            for seed in cfg['seeds']:
                model = train_noise_seed(inputs, standardized, y, int(seed), recipe, device, self.message_output)
                state_dicts.append({key: value.detach().cpu() for key, value in model.state_dict().items()})
        finally:
            torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = deterministic, benchmark

        bundle = {
            "model": "noise_timemil_ensemble",
            "state_dicts": state_dicts,
            "in_channels": int(inputs[0].shape[0]), "n_scalars": int(raw_scalars.shape[1]),
            "scalar_names": list(NOISE_SCALAR_NAMES),
            "scalar_mean": scalar_mean.tolist(), "scalar_std": scalar_std.tolist(),
            "db_floor": NOISE_DB_FLOOR, "db_ceil": NOISE_DB_CEIL, "db_center": NOISE_DB_CENTER, "db_half": NOISE_DB_HALF,
            "max_frames": NOISE_MAX_FRAMES, "bands_hz": [list(band) for band in NOISE_BANDS_HZ],
            "context_frames": NOISE_CONTEXT_FRAMES,
            "input": "two-band absolute-dB spectrogram of the segment, variance-weighted average over the "
                     "non-excluded channels of a window that extends context_frames hops either side, cropped "
                     "back to the segment's own frames",
            **calibration,
            "n_labels": len(inputs), "n_noise": int(y.sum()),
            "labels_csv": str(pathlib.Path(configure_path(self.labels_csv_path)).resolve()),
            "recipe": {
                **recipe, "seeds": [int(seed) for seed in cfg['seeds']],
                "augmentation": {
                    "gain_db": NOISE_AUG_GAIN_DB, "freq_roll_rows": NOISE_AUG_FREQ_ROLL,
                    "time_mask_fraction": NOISE_AUG_TIME_MASK_FRACTION, "freq_mask_rows": NOISE_AUG_FREQ_MASK_ROWS,
                },
                "exclude_metadata_audio_channels": bool(cfg['exclude_metadata_audio_channels']),
            },
            "built_by": "usv_playpen train-noise-model",
            "created": datetime.now().isoformat(timespec="seconds"),
        }
        bundle_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(bundle, bundle_path)
        load_noise_model(str(bundle_path), torch.device("cpu"))

        elapsed_minutes = (datetime.now() - start_time).total_seconds() / 60
        self.message_output(f"Wrote {bundle_path} ({bundle_path.stat().st_size / 1e6:.1f} MB) in {elapsed_minutes:.1f} min; detect-usv-noise loads it.")
        return bundle_path


@click.command(name="detect-usv-noise")
@click.option('--root-directory', type=click.Path(exists=True, file_okay=False, dir_okay=True), required=True, help='Session root directory path.')
@click.option('--noise-model-path', 'noise_model_path', type=str, default=None, required=False, help='Path to the noise model bundle (.pt); derived from spectrograms_root when empty.')
@click.option('--exclude-metadata-audio-channels/--no-exclude-metadata-audio-channels', 'exclude_metadata_audio_channels', default=None, required=False, help='Drop channels the session metadata marks as excluded from the spectrogram average.')
@click.option('--batch-size', 'batch_size', type=int, default=None, required=False, help='Segments per forward pass at the typical call length; a batch is budgeted at batch-size x 128 frame slots, so one long call never inflates it.')
@click.pass_context
def detect_usv_noise_cli(ctx, root_directory, **kwargs) -> None:
    """
    Description
    -----------
    A command-line tool to flag the USV segments that hold no vocalization and merge the noise columns
    into a session's USV summary CSV.

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
        block='detect_usv_noise',
    )

    USVNoiseDetector(
        root_directory=root_directory,
        input_parameter_dict=processing_settings_dict,
        message_output=print,
    ).detect_and_merge()


@click.command(name="train-noise-model")
@click.option('--labels-csv', 'labels_csv_path', type=click.Path(exists=True, file_okay=True, dir_okay=False), required=True, help='Labels CSV: sample_id, session_dir, start, stop, chs_count, noise (1 = no vocalization, 0 = vocalization).')
@click.option('--bundle-path', 'bundle_path', type=click.Path(file_okay=True, dir_okay=False), required=True, help='Output bundle (.pt); must not exist yet.')
@click.option('--calibration-path', 'calibration_path', type=str, default=None, required=False, help='JSON with the calibration table and the decision block the bundle carries.')
@click.option('--exclude-metadata-audio-channels/--no-exclude-metadata-audio-channels', 'exclude_metadata_audio_channels', default=None, required=False, help='Drop channels the session metadata marks as excluded from the spectrogram average (keep it as detect-usv-noise runs).')
@click.option('--n-workers', 'n_workers', type=int, default=None, required=False, help='Sessions whose audio is read concurrently (threads) while the training inputs are built.')
@click.option('--seed', 'seeds', type=int, multiple=True, default=None, required=False, help='Seed of one ensemble member; repeat once per member (replaces the seeds setting).')
@click.option('--epochs', 'epochs', type=int, default=None, required=False, help='Training epochs per member.')
@click.option('--batch-size', 'batch_size', type=int, default=None, required=False, help='Segments per training batch.')
@click.option('--learning-rate', 'learning_rate', type=float, default=None, required=False, help='Adam learning rate (cosine-annealed over the run).')
@click.option('--weight-decay', 'weight_decay', type=float, default=None, required=False, help='Adam weight decay.')
@click.option('--label-smoothing', 'label_smoothing', type=float, default=None, required=False, help='Label smoothing towards 0.5.')
@click.pass_context
def train_noise_model_cli(ctx, labels_csv_path, bundle_path, **kwargs) -> None:
    """
    Description
    -----------
    A command-line tool to train the noise-model ensemble on a labels CSV and write a bundle
    detect-usv-noise loads unchanged.

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
        parameters_lists=['seeds'],
        settings_dict='processing_settings',
        block='train_noise_model',
    )

    USVNoiseModelTrainer(
        labels_csv_path=labels_csv_path,
        bundle_path=bundle_path,
        input_parameter_dict=processing_settings_dict,
        message_output=print,
    ).train()
