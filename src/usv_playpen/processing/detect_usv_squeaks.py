"""
@author: bartulem
Detect squeaks (broadband vocalizations) among a session's USV segments and
merge the result into its ``*_usv_summary.csv``.

A squeak is a broadband harmonic stack (fundamental around 3-8 kHz) that the
ultrasonic DAS segmenter picks up as part of a USV segment. This step scores
every segment with Dexter's v3 time-resolved multiple-instance classifier
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
end reproduces Dexter's ``_sonic_wav_`` spectrogram store bit-exactly
(0.000 dB, verified 2026-09-14). The model sees a fixed 128-frame window
(262 ms): a shorter segment is padded at -100 dB and the padding masked out; a
longer one is truncated to its first 128 frames.

Columns written into ``usv_summary.csv`` (any pre-existing ones are replaced):

* ``squeak`` -- True / False in every row (``p >= squeak_threshold`` on the
  first 128 frames, Dexter's segment-level method);
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

Note on training sessions: the deployed checkpoint was fitted on labelled
segments from 15 sessions (20230124_172125, 20250211_165612, 20250403_205653,
20250418_184440, 20250424_175844, 20250506_155030, 20250923_203320,
20250927_160820, 20250928_185641, 20250928_212046, 20251118_103002,
20251221_111055, 20251221_114053, 20251221_124144, 20251221_142801). Its
output there is a fit, not a prediction, so leave those sessions out of any
cohort squeak-rate claim.
"""

from __future__ import annotations

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
from ..yaml_utils import read_excluded_audio_channels
from .generate_spectrograms import compute_usv_spectrogram

# Columns written into the USV summary CSV.
SQUEAK_COLUMNS = ("squeak", "squeak_probability", "squeak_start", "squeak_end")

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


class TimeMIL(nn.Module):
    """
    Description
    -----------
    Time-resolved multiple-instance squeak classifier (Dexter's ``TimeMIL``,
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
    the fixed 128-frame window in batches (segment probability, Dexter's
    method), and takes squeak onset / offset from a full-length pass (the
    128-frame window's own frames when the segment fits inside it, a separate
    forward over all frames when it does not). Rows whose segment is too short
    for a single STFT frame on any channel are returned with no probability and
    ``squeak`` False.

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
        for every scorable row), and the four ``SQUEAK_COLUMNS``.
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

    n_rows = len(row_indices)
    raw_probability = np.full(n_rows, np.nan)
    full_pass_frames: list[np.ndarray | None] = [None] * n_rows
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
        },
        schema={
            "row_index": pls.Int64,
            "n_frames": pls.Int64,
            "raw_probability": pls.Float64,
            "squeak": pls.Boolean,
            "squeak_probability": pls.Float64,
            "squeak_start": pls.Float64,
            "squeak_end": pls.Float64,
        },
    )


class USVSqueakDetector:
    """
    Description
    -----------
    Scores every USV segment of one session for a squeak and merges the four
    squeak columns into its ``*_usv_summary.csv``.
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
        ``squeak_probability``, ``squeak_start`` and ``squeak_end`` into the
        summary, replacing any existing squeak columns. Run it after
        ``das_summarize``: re-summarizing rewrites the CSV with its base columns
        only and would remove these.

        Parameters
        ----------

        Returns
        -------
        Updated ``*_usv_summary.csv`` with the four squeak columns.
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
        usv_df = usv_df.drop([column for column in SQUEAK_COLUMNS if column in usv_df.columns])

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
        merged = order_usv_summary_columns(pls.concat([usv_df, scores.select(SQUEAK_COLUMNS)], how="horizontal"))
        merged.write_csv(file=str(usv_summary_loc))

        self.message_output(
            f"Merged squeak calls into {usv_summary_loc.name}: {int(scores['squeak'].sum())} squeak(s) among {usv_df.height} USVs."
        )
        self.message_output(
            f"USV squeak detection ended at: {datetime.now().hour:02d}:{datetime.now().minute:02d}:{datetime.now().second:02d}."
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
