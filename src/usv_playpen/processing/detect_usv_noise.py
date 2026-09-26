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
"""

from __future__ import annotations

import math
import pathlib
from collections.abc import Callable
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
    context = bundle["context_frames"]
    inputs: list[np.ndarray] = []
    scalars: list[np.ndarray] = []
    scored_rows: list[int] = []
    try:
        for row_index in range(usv_summary.height):
            start = float(usv_summary["start"][row_index])
            stop = float(usv_summary["stop"][row_index])
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
            x, _n_used = segment_input(window, first_frame, 1 + (last_sample - first_sample) // HOP_SAMPLES, bundle)
            if x is None:
                continue
            inputs.append(x)
            scalars.append(np.array([np.log(float(usv_summary["chs_count"][row_index])), np.log(stop - start)], dtype=np.float32))
            scored_rows.append(row_index)
    finally:
        for handle in handles:
            handle.close()

    probability = np.full(usv_summary.height, np.nan, dtype=np.float64)
    if scored_rows:
        standardized = [(scalar - bundle["scalar_mean"]) / bundle["scalar_std"] for scalar in scalars]
        runs = []
        for model in bundle["models"]:
            out = []
            with torch.no_grad():
                for start_index, stop_index in frame_budget_batches(inputs, batch_size):
                    chunk = inputs[start_index:stop_index]
                    width = max(item.shape[2] for item in chunk)
                    x = np.full((len(chunk), chunk[0].shape[0], 128, width), -1.0, dtype=np.float32)
                    x[:, -1] = 0.0
                    valid = np.zeros((len(chunk), width), dtype=bool)
                    for position, item in enumerate(chunk):
                        x[position, :, :, :item.shape[2]] = item
                        valid[position, :item.shape[2]] = True
                    x_tensor = torch.from_numpy(x).to(device)
                    valid_tensor = torch.from_numpy(valid).to(device)
                    scalar_tensor = torch.tensor(np.stack(standardized[start_index:stop_index]), device=device)
                    out.append(torch.sigmoid(model(x_tensor, valid_tensor, valid_tensor, scalar_tensor)).cpu().numpy())
            runs.append(np.concatenate(out))
        probability[np.asarray(scored_rows)] = np.mean(runs, axis=0)
    unscorable = usv_summary.height - len(scored_rows)
    if unscorable:
        message_output(f"{unscorable} USV(s) are too short for one STFT window and get an empty noise probability.")
    return pls.DataFrame({
        "noise": np.nan_to_num(probability, nan=0.0) >= threshold,
        "noise_probability": probability,
    }, schema={"noise": pls.Boolean, "noise_probability": pls.Float64})


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
