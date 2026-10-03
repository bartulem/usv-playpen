"""
@author: bartulem
Classify every USV segment that is not noise as an ultrasonic call (``usv``), a
squeak (``squeak``) or both in one segment (``both``), locate each squeak in time,
and merge the result into the session's ``*_usv_summary.csv``.

A squeak is a broadband harmonic stack (fundamental around 3-8 kHz) that the
ultrasonic DAS segmenter picks up as part of a USV segment; a segment can hold a
squeak alone, an ultrasonic call alone, or both. This step scores every segment
the noise step did not flag (``noise`` not true) with an ensemble of
time-resolved multiple-instance classifiers that have two heads on one trunk: a
three-class head that decides the segment's class, and a per-frame squeak head
that marks where every squeak in view sits.

Input contract (fixed by the trained bundle, therefore read from it rather than
exposed as settings): the noise model's two-band input (:mod:`detect_usv_noise`):
two absolute-dB bands (30-120 kHz and 3-30 kHz, 128 linear rows each) of the
variance-weighted average of the UNFILTERED per-channel
``audio/hpss/*_cropped_to_video_hpss.wav`` files (metadata-excluded channels
dropped when ``exclude_metadata_audio_channels`` is on), Blackman-Harris STFT
(nperseg 2048, hop 512 = 2.048 ms, centred), mapped by
``(clip(x, -100, 50) + 25) / 75``. The audio window is the noise model's
(``context_frames`` = 49 hops, 100.4 ms, before the segment's first sample and
after its last, fewer at the file edges), but, unlike the noise model, the
spectrogram is NOT cropped back to the segment: the classifier sees the context
too, because squeaks often run past the segment the ultrasonic segmenter cut.
The third input channel is the segment indicator (1 on the segment's own frames,
0 on context frames), so the network knows which frames the class decision is
about. Frame ``k`` of a window is centred at ``read_start + k * 0.002048`` s,
``read_start`` being the window's first sample in session seconds. A window
longer than the bundle's ``max_frames`` (1024 frames, 2.1 s) keeps its first
``max_frames`` frames.

Network (:class:`USVSqueakTimeMIL`): the noise model's trunk (four
frequency-pooling Conv2d blocks and five residual dilated Conv1d blocks, about
61 frames of receptive field per frame) with a multiple-instance class head
(per-frame three-class logits attention-pooled over the SEGMENT's frames, plus a
linear term in the noise model's two scalars, log channel count and log
duration) and a per-frame squeak head over every valid frame of the window. The
bundle holds several members; their class probabilities and frame squeak
probabilities are averaged.

Columns written into ``usv_summary.csv`` (any pre-existing ones are replaced, and
the columns of earlier encodings -- ``squeak_probability``, ``squeak_frame_runs``,
``call_class``, ``squeak_spans``, ``n_squeaks`` -- are removed):

* ``usv`` / ``squeak`` -- two booleans from the class of highest ensemble-mean
  probability: a pure USV is ``(true, false)``, a pure squeak ``(false, true)``
  and a segment holding both ``(true, true)``; both null on noise rows and on
  rows too short for one STFT window. ``usv`` alone does NOT mean "USV only":
  a pure-USV filter is ``usv & ~squeak`` (:func:`os_utils.pure_usv_mask`);
* ``p_usv`` / ``p_squeak`` / ``p_both`` -- the ensemble-mean class
  probabilities (they sum to 1); null where the booleans are null;
* ``squeak_start`` / ``squeak_end`` -- ONE squeak extent per segment, in
  session seconds, on rows with ``squeak`` true only (null otherwise): the
  envelope of every frame whose ensemble-mean squeak probability exceeds the
  bundle's threshold (``extent_rule['threshold']``, 0.6), counting only the
  above-threshold runs that touch the segment's own frames (a run lying wholly
  in the context belongs to a neighbouring segment) and with no minimum run
  length, from half a hop before the first such frame's centre to half a hop
  after the last (:func:`squeak_envelope_frames`). The extent may reach into
  the context. A ``squeak`` row with no above-threshold frame touching the
  segment falls back to the segment frame of highest squeak probability (a
  one-frame extent); the step reports how many rows needed it.

The threshold was chosen on session-grouped cross-fitted predictions of the
same recipe; the bundle records it, its class names, its
input constants and scalar standardization, its training recipe and the label
files it was trained on, and this step reads every one of them from the bundle.

Squeak QLVM embedding (``infer-qlvm-squeak-latents``,
:class:`USVSqueakQLVMEmbedder`): a second step places every row of call class
``squeak`` true (pure squeaks and "both") that is not noise on the torus of one of the phase 3
broadband-vocalization QLVM cells (``qlvm_models_latest/phase3_BBVs_qlvm/<cell>``:
the ``infer_qlvm_squeak_latents.model_cell_directory`` setting, filled with the
production ``natural_session_N11000_nomask`` cell when empty) and writes the two
float columns ``qlvm_squeak1`` / ``qlvm_squeak2`` (torus coordinates in
``[0, 1)``, null on every other row). Its input is the sonic spectrogram
(3-30 kHz, 128 linear bins, absolute dB, the cells' front end) cropped to the
squeak extent the way the reference builder ``build_bbv_dataset.py`` built the
cells' training sets: the frames whose centres lie inside
``[squeak_start, squeak_end]`` plus two frames of context either side, min-max
normalized per crop, zero-padded to 128 frames, centred by ``stretch_specs`` and
min-max normalized once more as the decoder's data loader did
(:func:`squeak_qlvm_inputs`). One row has one squeak extent (the envelope of all its
above-threshold frames), so a row with several squeaks is embedded ONCE, over
that envelope (gaps included).
Because a squeak may extend past its segment, the audio the spectrogram is built
from covers the segment and the envelope plus its context
(:func:`squeak_crop_window`), so a crop is never cut off at the segment
boundary. Crops narrower than 8 frames (the training set's minimum) or wider
than 128 frames (the decoder's frame; the training set never compressed a crop
in time) get nulls. The embedding is the posterior mean over the cell's
Fibonacci lattice (``lattice_m`` of its ``manifest.json``) built in float32 as
the torch driver built it (:func:`qlvm_model.gen_fib_basis_float32`).

Training (``train-usv-squeak-model``, :class:`USVSqueakModelTrainer`): from the
labelling tool's files -- one or more label sets, each a labels CSV
(``panel_id``, ``label`` 0 usv / 1 squeak / 2 both / 3 unsure,
``squeak_extents_s`` JSON spans) and its sample CSV (``panel_id``,
``session_dir``, ``row_index``, ``start``, ``stop``), plus optional review CSVs
that override a set's labels by panel -- it builds every labelled segment's
input with the same function scoring uses (:func:`usv_squeak_window_input`),
trains one :class:`USVSqueakTimeMIL` per seed (trunk initialized from the noise
ensemble, the noise model's augmentation, joint class cross-entropy + frame
binary cross-entropy) and writes a bundle this step loads unchanged. Training
lives in this module, not in its own, for the reason the noise model's does: the
network, the input construction and the label-to-frame mapping must exist once,
shared by both directions, so a trained bundle sees exactly the input it is
later scored on. The labelled spans (several per segment where the labeller
marked several squeaks) remain the frame head's training targets; the envelope
rule is applied at inference only. Session-grouped cross-validation and the
choice of the threshold are an evaluation step outside the package; the trainer
takes the threshold from its settings and records it in the bundle.
"""

from __future__ import annotations

import json
import math
import pathlib
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    CALL_CLASSES,
    SQUEAK_FLAG_COLUMN,
    USV_FLAG_COLUMN,
    atomic_output_path,
    configure_path,
    derive_spectrogram_model_paths,
    first_match_or_raise,
    order_usv_summary_columns,
    squeak_bearing_mask,
)
from ..time_utils import is_gui_context, smart_wait
from .build_qlvm_training_set import stretch_specs
from .detect_usv_noise import (
    HOP_SAMPLES,
    NOISE_AUG_FREQ_MASK_ROWS,
    NOISE_AUG_FREQ_ROLL,
    NOISE_AUG_GAIN_DB,
    NOISE_AUG_TIME_MASK_FRACTION,
    NOISE_BANDS_HZ,
    NOISE_INPUT_CONTRACT,
    NOISE_SAMPLING_RATE,
    NOISE_SCALAR_NAMES,
    NoiseTimeMIL,
    augment_noise_batch,
    frame_budget_batches,
    load_noise_model,
    noise_scalars,
    pad_noise_batch,
    segment_input,
    squeak_wav_channels,
)
from .generate_spectrograms import compute_usv_spectrogram
from .qlvm_latents import cell_file, load_decoder_params, normalize_model_inputs
from .qlvm_model import decoder_head, embed_data, gen_fib_basis_float32

# Columns written into the USV summary CSV, in canonical order.
VOCAL_CLASS_COLUMNS = (USV_FLAG_COLUMN, SQUEAK_FLAG_COLUMN, "p_usv", "p_squeak", "p_both", "squeak_start", "squeak_end")

# The two booleans each call class is written as: (usv, squeak).
CALL_CLASS_FLAGS = {"usv": (True, False), "squeak": (False, True), "both": (True, True)}

# Columns of earlier encodings (the retired binary squeak detector's and the call-class / span
# encoding's), dropped whenever a summary is rewritten here.
RETIRED_SQUEAK_COLUMNS = ("squeak_probability", "squeak_frame_runs", "call_class", "squeak_spans", "n_squeaks")

# Name the bundle records for its network, checked on load.
USV_SQUEAK_MODEL_NAME = "usv_squeak_timemil_ensemble"

# Input contract of the call-class model: the noise model's, with the frame cap raised so a whole
# window (the longest labelled segment is 0.81 s = 396 frames, plus 2 x 49 context frames) is never
# truncated. Training writes these into the bundle; scoring reads them back from it.
USV_SQUEAK_INPUT_CONTRACT = {**NOISE_INPUT_CONTRACT, "max_frames": 1024}

# Labelling-tool answers: 0 usv, 1 squeak, 2 both (the class index), 3 unsure (never trained on).
USV_SQUEAK_UNSURE_LABEL = 3

# Columns a labels CSV, a sample CSV and an override CSV of the labelling tool must hold.
USV_SQUEAK_LABEL_COLUMNS = ("panel_id", "label", "squeak_extents_s")
USV_SQUEAK_SAMPLE_COLUMNS = ("panel_id", "session_dir", "row_index", "start", "stop")

# Largest |summary start - sample start| (s) accepted when a labelled segment is looked up in its
# session's USV summary: a larger gap means the summary was rewritten since the sample was drawn.
USV_SQUEAK_START_TOLERANCE_S = 1e-5

# Decimals of the session-second times written into squeak_start / squeak_end (1 us; the audio
# sample is 4 us).
SQUEAK_TIME_DECIMALS = 6

# Front end of the squeak QLVM cells (the sonic band of the retired squeak classifier, which their
# training sets were cropped from): 3-30 kHz, 128 linear bins, Blackman-Harris STFT (nperseg 2048, hop
# 512, centred), absolute dB (ref 1.0, no top_db clamp), variance-weighted channel average. Constants,
# because the cells' inputs depend on them.
SQUEAK_SAMPLING_RATE = NOISE_SAMPLING_RATE
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
FRAME_DT_S = SQUEAK_SPEC_PARAMS["hop_length"] / SQUEAK_SAMPLING_RATE

# Width (frames) of the reference 128-frame squeak spectrogram store the phase 3 squeak QLVM training
# sets were cropped from (the window of the retired squeak classifier).
SQUEAK_REFERENCE_WINDOW_FRAMES = 128

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


class USVSqueakTimeMIL(NoiseTimeMIL):
    """
    Description
    -----------
    The call-class network: the noise model's trunk (:class:`detect_usv_noise.NoiseTimeMIL`'s four
    frequency-pooling Conv2d blocks and five residual dilated Conv1d blocks, so every frame keeps its own
    128-channel representation with about 61 frames of receptive field) with a three-class
    multiple-instance head and a per-frame squeak head. The noise model's binary head is removed.
    Trunk attribute names are the noise model's, so a noise checkpoint's ``body.*`` / ``tcn.*`` weights
    load into it unchanged (the pretrained initialization), and head attribute names match the
    call-class bundle's ``state_dict`` keys and must not change.

    The noise model's per-frame instance logits cannot double as the squeak track: the class decision
    "squeak" vs "both" hinges on whether an ultrasonic call is ALSO present, not on where the squeak is,
    so the frame head for the squeak itself is a separate target with its own supervision.
    """

    def __init__(self, in_channels: int, n_scalars: int, n_classes: int = 3, ch: int = 32) -> None:
        """
        Description
        -----------
        Builds the trunk through :class:`NoiseTimeMIL` and replaces its binary head with the class
        head (``class_frame`` per-frame logits, ``class_attn`` attention, ``class_scalar`` scalar term)
        and the squeak head (``squeak_frame``, 1x1 convolutions 128 -> 64 -> 1).

        Parameters
        ----------
        in_channels (int)
            Input channels: the two spectrogram bands plus the segment indicator.
        n_scalars (int)
            Number of per-segment scalar inputs.
        n_classes (int)
            Number of segment classes.
        ch (int)
            Base channel width; the trunk ends at ``4 * ch`` channels.

        Returns
        -------
        None
        """

        super().__init__(in_channels=in_channels, n_scalars=n_scalars, ch=ch)
        features = ch * 4
        self.head = None
        self.class_frame = nn.Conv1d(features, n_classes, 1)
        self.class_attn = nn.Sequential(nn.Conv1d(features, 64, 1), nn.Tanh(), nn.Conv1d(64, 1, 1))
        self.class_scalar = nn.Linear(n_scalars, n_classes)
        self.squeak_frame = nn.Sequential(nn.Conv1d(features, 64, 1), nn.ReLU(), nn.Conv1d(64, 1, 1))

    def forward(self, x: torch.Tensor, valid: torch.Tensor, segment: torch.Tensor, scalars: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Description
        -----------
        Computes the segment's class logits (per-frame class logits attention-pooled over the segment's
        frames, plus the linear scalar term) and the per-frame squeak logits.

        Parameters
        ----------
        x (torch.Tensor)
            ``(B, 3, 128, T)`` normalized inputs (last channel the segment indicator).
        valid (torch.Tensor)
            ``(B, T)`` boolean frame validity (padding excluded).
        segment (torch.Tensor)
            ``(B, T)`` boolean segment frames; the class attention pools over these.
        scalars (torch.Tensor)
            ``(B, n_scalars)`` standardized scalars.

        Returns
        -------
        class_logits (torch.Tensor)
            ``(B, n_classes)``.
        frame_logits (torch.Tensor)
            ``(B, T)`` squeak logits (meaningful on valid frames only).
        """

        h = self.body(x).squeeze(2) * valid.unsqueeze(1)
        for block in self.tcn:
            h = (h + block(h)) * valid.unsqueeze(1)
        frame_class = self.class_frame(h)
        attention = self.class_attn(h).squeeze(1).masked_fill(~segment, torch.finfo(h.dtype).min)
        weights = torch.softmax(attention, dim=1).unsqueeze(1)
        class_logits = (weights * frame_class.masked_fill(~segment.unsqueeze(1), 0.0)).sum(2) + self.class_scalar(scalars)
        return class_logits, self.squeak_frame(h).squeeze(1)


def load_usv_squeak_model(model_path: str, device: torch.device) -> dict:
    """
    Description
    -----------
    Loads the call-class bundle: every member's weights in eval mode on ``device``, the input
    constants, the scalar standardization, the class names and the squeak-extent threshold. A bundle whose model
    name, class names or bands differ from what this module builds is refused rather than scored with.

    Parameters
    ----------
    model_path (str)
        Path to the bundle (``.pt``); derived from the spectrograms root when the setting is empty.
    device (torch.device)
        Device to load the members onto.

    Returns
    -------
    bundle (dict)
        ``models`` (list[USVSqueakTimeMIL]), ``class_names`` (tuple), ``scalar_mean`` / ``scalar_std``
        (float32 arrays), ``bands_hz``, ``db_floor``, ``db_ceil``, ``db_center``, ``db_half``,
        ``max_frames``, ``context_frames``, ``span_threshold`` (float, the squeak-frame threshold of
        the extent rule) and ``labels`` (the label files it was trained on).

    Raises
    ------
    FileNotFoundError
        The bundle does not exist.
    ValueError
        The bundle is not a call-class bundle this module can score with.
    """

    path = pathlib.Path(configure_path(model_path))
    if not path.is_file():
        error_message = (
            f"Call-class model bundle not found: {path}. Set processing_settings['detect_usv_squeaks']['squeak_model_path'] "
            f"or leave it empty to derive it from shared_resources['spectrograms_root']."
        )
        raise FileNotFoundError(error_message)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    problems = []
    if checkpoint["model"] != USV_SQUEAK_MODEL_NAME:
        problems.append(f"model is {checkpoint['model']!r}, expected {USV_SQUEAK_MODEL_NAME!r} (the retired binary squeak checkpoint cannot be scored here)")
    if tuple(checkpoint["class_names"]) != CALL_CLASSES:
        problems.append(f"class_names are {checkpoint['class_names']}, expected {list(CALL_CLASSES)}")
    if [tuple(band) for band in checkpoint["bands_hz"]] != [tuple(band) for band in NOISE_BANDS_HZ]:
        problems.append(f"bands_hz are {checkpoint['bands_hz']}, this module builds {list(NOISE_BANDS_HZ)}")
    if int(checkpoint["sampling_rate"]) != NOISE_SAMPLING_RATE or int(checkpoint["hop_samples"]) != HOP_SAMPLES:
        problems.append(f"sampling_rate / hop_samples are {checkpoint['sampling_rate']} / {checkpoint['hop_samples']}, expected {NOISE_SAMPLING_RATE} / {HOP_SAMPLES}")
    if problems:
        error_message = f"{path} is not a call-class bundle detect-usv-squeaks can score with:\n  " + "\n  ".join(problems)
        raise ValueError(error_message)
    models = []
    for state_dict in checkpoint["state_dicts"]:
        model = USVSqueakTimeMIL(in_channels=checkpoint["in_channels"], n_scalars=checkpoint["n_scalars"], n_classes=len(CALL_CLASSES)).to(device)
        model.load_state_dict(state_dict)
        model.eval()
        models.append(model)
    return {
        "models": models,
        "class_names": tuple(checkpoint["class_names"]),
        "scalar_mean": np.asarray(checkpoint["scalar_mean"], dtype=np.float32),
        "scalar_std": np.asarray(checkpoint["scalar_std"], dtype=np.float32),
        "bands_hz": NOISE_BANDS_HZ,
        "db_floor": float(checkpoint["db_floor"]), "db_ceil": float(checkpoint["db_ceil"]),
        "db_center": float(checkpoint["db_center"]), "db_half": float(checkpoint["db_half"]),
        "max_frames": int(checkpoint["max_frames"]), "context_frames": int(checkpoint["context_frames"]),
        "span_threshold": float(checkpoint["extent_rule"]["threshold"]),
        "labels": list(checkpoint["labels"]),
    }


def usv_squeak_window_input(
    handles: list[sf.SoundFile],
    n_file: int,
    start: float,
    stop: float,
    contract: dict,
) -> dict | None:
    """
    Description
    -----------
    Reads one segment's audio window from the open per-channel wavs and builds its call-class input.
    The window is the noise model's (:func:`detect_usv_noise.window_segment_input`): it starts exactly
    ``context_frames`` hops before the segment's first sample ``floor(start * fs)`` (fewer at the start
    of a recording) and ends ``context_frames`` hops after its last sample ``ceil(stop * fs)`` (or at the
    end of the file). :func:`detect_usv_noise.segment_input` builds the two-band spectrogram of the WHOLE
    window (first frame 0, capped at ``max_frames``), and its all-ones third channel is replaced by the
    segment indicator: 1 on the segment's own ``1 + (last - first) // hop`` frames, which start at frame
    ``first_frame`` (the hops of context actually read), 0 elsewhere. Scoring (``detect-usv-squeaks``)
    and training (``train-usv-squeak-model``) both build their inputs here.

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
    contract (dict)
        Loaded bundle, or ``USV_SQUEAK_INPUT_CONTRACT`` when training (``bands_hz``, the dB constants,
        ``max_frames`` and ``context_frames`` are read).

    Returns
    -------
    window (dict | None)
        ``x`` (``(3, 128, T)`` float32 input), ``read_start_s`` (session time of frame 0's centre),
        ``first_frame`` (first segment frame) and ``n_segment_frames`` (segment frames kept after the
        cap); None when the window is too short for one STFT window or the cap leaves no segment frame.
    """

    context = contract["context_frames"]
    first_sample = math.floor(start * NOISE_SAMPLING_RATE)
    last_sample = math.ceil(stop * NOISE_SAMPLING_RATE)
    first_frame = min(context, first_sample // HOP_SAMPLES)
    read_start = first_sample - first_frame * HOP_SAMPLES
    read_stop = min(n_file, last_sample + context * HOP_SAMPLES)
    channels = []
    for handle in handles:
        handle.seek(read_start)
        channels.append(handle.read(frames=read_stop - read_start, dtype="float64", always_2d=False))
    x, n_used = segment_input(np.stack(channels, axis=1), 0, 10 ** 9, contract)
    if x is None:
        return None
    n_segment = min(1 + (last_sample - first_sample) // HOP_SAMPLES, n_used - first_frame)
    if n_segment < 1:
        return None
    indicator = np.zeros(n_used, dtype=np.float32)
    indicator[first_frame:first_frame + n_segment] = 1.0
    x[-1] = indicator[None, :]
    return {"x": x, "read_start_s": read_start / NOISE_SAMPLING_RATE, "first_frame": int(first_frame), "n_segment_frames": int(n_segment)}


def ensemble_usv_squeak_predict(
    models: list[USVSqueakTimeMIL],
    inputs: list[np.ndarray],
    standardized_scalars: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Description
    -----------
    Scores call-class inputs with every member of an ensemble and returns the member-mean class
    probabilities (softmax) and frame squeak probabilities (sigmoid). Inputs are grouped into
    frame-budgeted batches (:func:`detect_usv_noise.frame_budget_batches`) and padded as in training
    (:func:`detect_usv_noise.pad_noise_batch`); the class head pools over each input's segment frames
    (indicator channel above 0.5), the squeak head runs over every valid frame.

    Parameters
    ----------
    models (list[USVSqueakTimeMIL])
        Ensemble members, in eval mode on ``device``.
    inputs (list[np.ndarray])
        Per-segment ``(3, 128, T)`` inputs.
    standardized_scalars (np.ndarray)
        ``(N, n_scalars)`` float32 scalars, standardized with the bundle's mean and std.
    batch_size (int)
        Inputs per forward pass at the typical length (the frame budget is this times 128 frames).
    device (torch.device)
        Scoring device.

    Returns
    -------
    class_probability (np.ndarray)
        ``(N, 3)`` member-mean class probabilities, in ``CALL_CLASSES`` order.
    frame_probability (list[np.ndarray])
        Per input, the ``(T_i,)`` member-mean frame squeak probability.
    """

    class_sum = np.zeros((len(inputs), len(CALL_CLASSES)), dtype=np.float64)
    frame_sum = [np.zeros(item.shape[2], dtype=np.float64) for item in inputs]
    batches = frame_budget_batches(inputs, batch_size)
    with torch.no_grad():
        for model in models:
            for start_index, stop_index in batches:
                chunk = inputs[start_index:stop_index]
                x, valid = pad_noise_batch(chunk)
                x_tensor = torch.from_numpy(x).to(device)
                valid_tensor = torch.from_numpy(valid).to(device)
                scalar_tensor = torch.tensor(np.asarray(standardized_scalars[start_index:stop_index], dtype=np.float32), device=device)
                class_logits, frame_logits = model(x_tensor, valid_tensor, x_tensor[:, -1, 0, :] > 0.5, scalar_tensor)
                class_sum[start_index:stop_index] += torch.softmax(class_logits.float(), dim=1).cpu().numpy()
                frame_np = torch.sigmoid(frame_logits.float()).cpu().numpy()
                for offset, item in enumerate(chunk):
                    frame_sum[start_index + offset] += frame_np[offset, :item.shape[2]]
    return class_sum / len(models), [frames / len(models) for frames in frame_sum]


def squeak_envelope_frames(
    frame_probability: np.ndarray,
    threshold: float,
    segment_first: int,
    segment_last: int,
) -> tuple[int, int, bool]:
    """
    Description
    -----------
    The single squeak extent of one window, in frames: every maximal run of consecutive frames whose
    probability EXCEEDS ``threshold`` (strictly; no minimum run length) is kept when it touches the
    segment's own frames ``segment_first .. segment_last`` (a run lying wholly in the context belongs
    to a neighbouring segment), and the extent runs from the first kept run's first frame to the last
    kept run's last frame (so it may reach into the context). When no frame above the threshold
    touches the segment, the extent falls back to the one segment frame of highest probability.

    Parameters
    ----------
    frame_probability (np.ndarray)
        ``(T,)`` frame squeak probabilities of one window.
    threshold (float)
        Frame threshold (the bundle's ``extent_rule['threshold']``).
    segment_first (int)
        First segment frame of the window.
    segment_last (int)
        Last segment frame of the window (inclusive).

    Returns
    -------
    first (int)
        First frame of the extent.
    last (int)
        Last frame of the extent (inclusive).
    fallback (bool)
        True when no above-threshold frame touched the segment and the highest-scoring segment frame
        was used.
    """

    probability = np.asarray(frame_probability)
    above = np.concatenate([[False], probability > threshold, [False]])
    edges = np.flatnonzero(np.diff(above.astype(np.int8)))
    kept = [(int(first), int(stop) - 1) for first, stop in zip(edges[::2], edges[1::2], strict=True)
            if first <= segment_last and stop - 1 >= segment_first]
    if kept:
        return kept[0][0], kept[-1][1], False
    best = segment_first + int(np.argmax(probability[segment_first:segment_last + 1]))
    return best, best, True


def frames_to_session_seconds(first: int, last: int, read_start_s: float) -> tuple[float, float]:
    """
    Description
    -----------
    Converts an inclusive frame range of a window into session seconds: from half a hop before the
    first frame's centre to half a hop after the last frame's centre (frame ``k`` is centred at
    ``read_start_s + k * hop``), rounded to ``SQUEAK_TIME_DECIMALS``.

    Parameters
    ----------
    first (int)
        First frame.
    last (int)
        Last frame (inclusive).
    read_start_s (float)
        Session time of frame 0's centre.

    Returns
    -------
    start_s (float)
        Extent start (s).
    end_s (float)
        Extent end (s).
    """

    hop_s = HOP_SAMPLES / NOISE_SAMPLING_RATE
    return (round(read_start_s + (first - 0.5) * hop_s, SQUEAK_TIME_DECIMALS),
            round(read_start_s + (last + 0.5) * hop_s, SQUEAK_TIME_DECIMALS))


def classify_usv_squeak_rows(
    session_root: pathlib.Path,
    usv_summary: pls.DataFrame,
    bundle: dict,
    device: torch.device,
    exclude_metadata_audio_channels: bool,
    batch_size: int,
    message_output: Callable,
) -> pls.DataFrame:
    """
    Description
    -----------
    Classifies every row of one session's USV summary that is not noise and returns the vocal-class
    columns. For each such row the call-class input is built (:func:`usv_squeak_window_input`) with
    the noise model's scalars (:func:`detect_usv_noise.noise_scalars`, standardized with the bundle's
    mean and std), the ensemble scores every input (:func:`ensemble_usv_squeak_predict`), the class is
    the arg-max of the mean probabilities, written as the ``usv`` / ``squeak`` booleans
    (``CALL_CLASS_FLAGS``), and on rows with ``squeak`` true the squeak extent is the envelope of the
    mean frame track's above-threshold frames touching the segment (:func:`squeak_envelope_frames`,
    with its fallback; :func:`frames_to_session_seconds`). Rows flagged as noise (``noise`` true; a
    null ``noise`` counts as not noise, :func:`os_utils.drop_noise_usvs`) are not read at all and get
    nulls in every column, as do rows too short for one STFT window.

    cuDNN is put in deterministic mode for the duration of the call and restored afterwards, and
    autocast is disabled locally, so a mixed-precision context left open by an earlier step in the same
    process (the mask step enters one) cannot change the probabilities.

    Parameters
    ----------
    session_root (pathlib.Path)
        Session root directory.
    usv_summary (pls.DataFrame)
        The session's USV summary (``start``, ``stop``, ``chs_count`` and ``noise`` are read).
    bundle (dict)
        Loaded call-class bundle (:func:`load_usv_squeak_model`).
    device (torch.device)
        Scoring device.
    exclude_metadata_audio_channels (bool)
        Drop channels the session metadata marks as excluded from the average.
    batch_size (int)
        Inputs per forward pass at the typical length (:func:`detect_usv_noise.frame_budget_batches`).
    message_output (Callable)
        Logging callback.

    Returns
    -------
    scores (pls.DataFrame)
        One row per summary row, the ``VOCAL_CLASS_COLUMNS``: ``usv`` / ``squeak`` (Boolean),
        ``p_usv`` / ``p_squeak`` / ``p_both`` (Float64) and ``squeak_start`` / ``squeak_end``
        (Float64). The number of squeak rows whose extent fell back to one frame is logged.

    Raises
    ------
    ValueError
        The summary has no ``noise`` column (run ``detect-usv-noise`` first).
    """

    if "noise" not in usv_summary.columns:
        error_message = "The USV summary has no 'noise' column; run detect-usv-noise on the session before detect-usv-squeaks."
        raise ValueError(error_message)
    n_rows = usv_summary.height
    is_noise = usv_summary["noise"].cast(pls.Boolean).fill_null(False).to_numpy()
    candidates = np.flatnonzero(~is_noise)
    starts = usv_summary["start"].cast(pls.Float64).to_numpy()
    stops = usv_summary["stop"].cast(pls.Float64).to_numpy()
    chs_counts = usv_summary["chs_count"].cast(pls.Float64).to_numpy()

    windows: list[dict] = []
    scored_rows: list[int] = []
    if candidates.size:
        wav_paths = squeak_wav_channels(session_root, exclude_metadata_audio_channels, message_output)
        handles = [sf.SoundFile(str(path), mode="r") for path in wav_paths]
        try:
            n_file = handles[0].frames
            for row_index in candidates:
                window = usv_squeak_window_input(handles, n_file, float(starts[row_index]), float(stops[row_index]), bundle)
                if window is None:
                    continue
                windows.append(window)
                scored_rows.append(int(row_index))
        finally:
            for handle in handles:
                handle.close()

    usv_flag: list[bool | None] = [None] * n_rows
    squeak_flag: list[bool | None] = [None] * n_rows
    probability = np.full((n_rows, len(CALL_CLASSES)), np.nan, dtype=np.float64)
    squeak_start: list[float | None] = [None] * n_rows
    squeak_end: list[float | None] = [None] * n_rows
    n_fallback = 0
    if scored_rows:
        raw_scalars = np.stack([noise_scalars(chs_counts[row], starts[row], stops[row]) for row in scored_rows])
        standardized = ((raw_scalars - bundle["scalar_mean"]) / bundle["scalar_std"]).astype(np.float32)
        previous_deterministic = torch.backends.cudnn.deterministic
        previous_benchmark = torch.backends.cudnn.benchmark
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            with torch.autocast(device_type=device.type, enabled=False):
                class_probability, frame_probability = ensemble_usv_squeak_predict(
                    bundle["models"], [window["x"] for window in windows], standardized, batch_size, device,
                )
        finally:
            torch.backends.cudnn.deterministic = previous_deterministic
            torch.backends.cudnn.benchmark = previous_benchmark
        for position, row in enumerate(scored_rows):
            probability[row] = class_probability[position]
            label = CALL_CLASSES[int(np.argmax(class_probability[position]))]
            usv_flag[row], squeak_flag[row] = CALL_CLASS_FLAGS[label]
            if squeak_flag[row]:
                window = windows[position]
                first, last, fallback = squeak_envelope_frames(
                    frame_probability[position], bundle["span_threshold"],
                    window["first_frame"], window["first_frame"] + window["n_segment_frames"] - 1,
                )
                n_fallback += fallback
                squeak_start[row], squeak_end[row] = frames_to_session_seconds(first, last, window["read_start_s"])
    unscorable = candidates.size - len(scored_rows)
    if unscorable:
        message_output(f"{unscorable} non-noise USV(s) are too short for one STFT window and get empty vocal-class columns.")
    n_squeak = sum(1 for row in scored_rows if squeak_flag[row])
    message_output(
        f"Squeak extents: {n_fallback} of {n_squeak} squeak-bearing segment(s) had no frame above {bundle['span_threshold']} "
        f"touching the segment and took the highest-scoring segment frame instead."
    )
    return pls.DataFrame(
        {
            USV_FLAG_COLUMN: usv_flag,
            SQUEAK_FLAG_COLUMN: squeak_flag,
            "p_usv": probability[:, 0],
            "p_squeak": probability[:, 1],
            "p_both": probability[:, 2],
            "squeak_start": squeak_start,
            "squeak_end": squeak_end,
        },
        schema={
            USV_FLAG_COLUMN: pls.Boolean,
            SQUEAK_FLAG_COLUMN: pls.Boolean,
            "p_usv": pls.Float64,
            "p_squeak": pls.Float64,
            "p_both": pls.Float64,
            "squeak_start": pls.Float64,
            "squeak_end": pls.Float64,
        },
        nan_to_null=True,
    )


class USVSqueakDetector:
    """
    Description
    -----------
    Classifies every USV segment of one session that is not noise as a pure USV, a pure squeak or both,
    writes the class as the ``usv`` / ``squeak`` booleans with the squeak extent, and merges the
    vocal-class columns into its ``*_usv_summary.csv``.
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
            Processing settings; the ``detect_usv_squeaks`` block supplies the bundle path, the
            channel-exclusion switch and the batch size.
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
        Loads the call-class bundle, classifies every non-noise row of the session's USV summary (see
        :func:`classify_usv_squeak_rows`), and writes the ``VOCAL_CLASS_COLUMNS`` into the summary,
        replacing any existing ones (an old ``squeak`` column included) and removing the columns of
        earlier encodings (``RETIRED_SQUEAK_COLUMNS``). Every other column is left as it was. Run it
        after ``das_summarize`` (re-summarizing rewrites the CSV with its base columns only) and after
        ``detect-usv-noise`` (whose ``noise`` column it reads). The summary is rewritten atomically.

        Parameters
        ----------

        Returns
        -------
        Updated ``*_usv_summary.csv`` with the vocal-class columns.
        """

        self.message_output(
            f"USV call-class detection started at: {datetime.now().hour:02d}:{datetime.now().minute:02d}:{datetime.now().second:02d}."
        )
        smart_wait(app_context_bool=self.app_context_bool, seconds=1)

        derive_spectrogram_model_paths(self.input_parameter_dict)
        cfg = self.input_parameter_dict['detect_usv_squeaks']
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bundle = load_usv_squeak_model(cfg['squeak_model_path'], device)
        self.message_output(
            f"Call classes {'/'.join(bundle['class_names'])} from a {len(bundle['models'])}-member ensemble; the squeak extent is the "
            f"envelope of frames with squeak probability > {bundle['span_threshold']} touching the segment."
        )

        root = pathlib.Path(self.root_directory)
        usv_summary_loc = first_match_or_raise(
            root=root / "audio",
            pattern="*_usv_summary.csv",
            recursive=True,
            label="USV summary CSV",
        )
        usv_df = pls.read_csv(source=str(usv_summary_loc), schema_overrides={"usv_id": pls.String})
        usv_df = usv_df.drop([column for column in (*VOCAL_CLASS_COLUMNS, *RETIRED_SQUEAK_COLUMNS) if column in usv_df.columns])

        scores = classify_usv_squeak_rows(
            session_root=root,
            usv_summary=usv_df,
            bundle=bundle,
            device=device,
            exclude_metadata_audio_channels=cfg['exclude_metadata_audio_channels'],
            batch_size=cfg['batch_size'],
            message_output=self.message_output,
        )
        merged = order_usv_summary_columns(pls.concat([usv_df, scores], how="horizontal"))
        with atomic_output_path(usv_summary_loc) as tmp_summary_path:
            merged.write_csv(file=str(tmp_summary_path))

        usv_flag = scores[USV_FLAG_COLUMN].fill_null(False)
        squeak_flag = scores[SQUEAK_FLAG_COLUMN].fill_null(False)
        self.message_output(
            f"Merged vocal classes into {usv_summary_loc.name}: {int((usv_flag & ~squeak_flag).sum())} pure usv, "
            f"{int((squeak_flag & ~usv_flag).sum())} pure squeak, {int((usv_flag & squeak_flag).sum())} both among {usv_df.height} "
            f"segments ({int(scores[USV_FLAG_COLUMN].null_count())} without a class: noise or too short)."
        )
        self.message_output(
            f"USV call-class detection ended at: {datetime.now().hour:02d}:{datetime.now().minute:02d}:{datetime.now().second:02d}."
        )


def read_usv_squeak_labels(label_sets: list[dict], label_overrides: list[dict], message_output: Callable) -> pls.DataFrame:
    """
    Description
    -----------
    Reads the labelling tool's files into one training table. Each label set is a labels CSV
    (``panel_id``, ``label``, ``squeak_extents_s``) and its sample CSV (``panel_id``, ``session_dir``,
    ``row_index``, ``start``, ``stop``), joined on ``panel_id`` (every labelled panel must have a sample
    row); ``sample_id`` is ``<set name>_<panel_id>``, so panel ids may repeat across sets. Each override
    (a review CSV in the labels format) replaces the label and spans of the panels it lists in the named
    set; the set's own CSV is never edited, and an override panel that the set does not hold stops the
    run. Unsure answers (``label`` 3) are dropped and counted; any other label outside 0 / 1 / 2 stops the
    run, as does a ``usv`` label with spans or a ``squeak`` / ``both`` label without any. Row order is
    the sets' order, then each labels CSV's order.

    Parameters
    ----------
    label_sets (list[dict])
        One dict per set: ``name``, ``labels_csv`` and ``sample_csv``.
    label_overrides (list[dict])
        One dict per override, applied in order: ``name`` (the set it overrides) and ``override_csv``.
    message_output (Callable)
        Logging callback.

    Returns
    -------
    labels (pls.DataFrame)
        ``sample_id``, ``set``, ``session_dir``, ``row_index`` (Int64), ``start`` / ``stop`` (Float64),
        ``label`` (Int64, 0 / 1 / 2), ``squeak_extents_s`` (JSON text) and ``reviewed`` (Boolean).

    Raises
    ------
    FileNotFoundError
        A listed CSV does not exist.
    ValueError
        A file lacks a column, a panel lacks a sample row, an override names an unknown set or panel,
        set names repeat, or a label / span combination is invalid.
    """

    if not label_sets:
        error_message = "train_usv_squeak_model['label_sets'] is empty; list at least one labels CSV with its sample CSV."
        raise ValueError(error_message)
    names = [label_set['name'] for label_set in label_sets]
    if len(set(names)) != len(names):
        error_message = f"Label set names must be distinct (they prefix the sample ids); got {names}."
        raise ValueError(error_message)

    def _read(csv_path: str, required: tuple[str, ...]) -> pls.DataFrame:
        # Every column is read as text (types are cast once, after the sets are joined), so an id
        # that looks numeric keeps its leading zeros and an all-empty column never changes type.
        path = pathlib.Path(configure_path(csv_path))
        if not path.is_file():
            error_message = f"Call-class training file not found: {path}."
            raise FileNotFoundError(error_message)
        table = pls.read_csv(str(path), infer_schema_length=0)
        missing = [column for column in required if column not in table.columns]
        if missing:
            error_message = f"{path} lacks the column(s) {missing}; it must hold {list(required)}."
            raise ValueError(error_message)
        return table.select(list(required))

    frames = []
    for label_set in label_sets:
        labels = _read(label_set['labels_csv'], USV_SQUEAK_LABEL_COLUMNS)
        sample = _read(label_set['sample_csv'], USV_SQUEAK_SAMPLE_COLUMNS)
        joined = labels.join(sample, on="panel_id", how="inner", maintain_order="left")
        if joined.height != labels.height:
            error_message = f"Label set {label_set['name']!r}: {labels.height - joined.height} labelled panel(s) have no row in {label_set['sample_csv']}."
            raise ValueError(error_message)
        joined = joined.with_columns(pls.lit(False).alias("reviewed"))
        for override in label_overrides:
            if override['name'] != label_set['name']:
                continue
            review = _read(override['override_csv'], USV_SQUEAK_LABEL_COLUMNS)
            unknown = sorted(set(review["panel_id"]) - set(joined["panel_id"]))
            if unknown:
                error_message = f"{override['override_csv']}: panel(s) {unknown} are not in label set {label_set['name']!r}."
                raise ValueError(error_message)
            joined = (
                joined.join(review.rename({"label": "review_label", "squeak_extents_s": "review_extents"}), on="panel_id", how="left", maintain_order="left")
                .with_columns(
                    pls.coalesce("review_label", "label").alias("label"),
                    pls.coalesce("review_extents", "squeak_extents_s").alias("squeak_extents_s"),
                    (pls.col("reviewed") | pls.col("review_label").is_not_null()).alias("reviewed"),
                )
                .drop("review_label", "review_extents")
            )
        frames.append(joined.with_columns(
            (pls.lit(f"{label_set['name']}_") + pls.col("panel_id")).alias("sample_id"),
            pls.lit(label_set['name']).alias("set"),
        ))
    unknown_sets = sorted({override['name'] for override in label_overrides} - set(names))
    if unknown_sets:
        error_message = f"train_usv_squeak_model['label_overrides'] names unknown label set(s) {unknown_sets}."
        raise ValueError(error_message)

    table = pls.concat(frames, how="vertical").select(
        "sample_id", "set", "session_dir",
        pls.col("row_index").cast(pls.Int64), pls.col("start").cast(pls.Float64), pls.col("stop").cast(pls.Float64),
        pls.col("label").cast(pls.Int64), pls.col("squeak_extents_s").fill_null("[]"), "reviewed",
    )
    if table.select(pls.col("sample_id", "session_dir", "row_index", "start", "stop", "label").null_count()).sum_horizontal()[0] > 0:
        error_message = "Call-class training labels have empty cells in sample_id / session_dir / row_index / start / stop / label."
        raise ValueError(error_message)
    n_unsure = int((table["label"] == USV_SQUEAK_UNSURE_LABEL).sum())
    table = table.filter(pls.col("label") != USV_SQUEAK_UNSURE_LABEL)
    if not table["label"].is_in([0, 1, 2]).all():
        error_message = "Call-class labels must be 0 (usv), 1 (squeak), 2 (both) or 3 (unsure)."
        raise ValueError(error_message)
    n_spans = np.array([len(json.loads(text)) for text in table["squeak_extents_s"]], dtype=np.int64)
    labels_array = table["label"].to_numpy()
    if np.any((labels_array == 0) & (n_spans > 0)) or np.any((labels_array > 0) & (n_spans == 0)):
        error_message = "Every usv label must have no squeak span and every squeak / both label at least one."
        raise ValueError(error_message)
    counts = {name: int((labels_array == index).sum()) for index, name in enumerate(CALL_CLASSES)}
    message_output(
        f"{table.height} labelled segment(s) from {table['session_dir'].n_unique()} session(s) in {len(label_sets)} set(s) "
        f"({counts}; {int(table['reviewed'].sum())} reviewed); {n_unsure} unsure answer(s) dropped."
    )
    return table


def session_usv_squeak_training_inputs(
    session_dir: str,
    segments: list[tuple[int, int, float, float]],
    exclude_metadata_audio_channels: bool,
) -> dict[int, dict]:
    """
    Description
    -----------
    Builds the call-class inputs of one session's labelled segments, opening the session's wavs and USV
    summary once, and reads each segment's ``chs_count`` from the summary row the sample names (checking
    that the row still starts where the sample says). Run in a worker thread by
    :func:`build_usv_squeak_training_inputs`.

    Parameters
    ----------
    session_dir (str)
        Session root directory.
    segments (list[tuple[int, int, float, float]])
        ``(label row, summary row_index, start, stop)`` of every labelled segment of the session.
    exclude_metadata_audio_channels (bool)
        Drop channels the session metadata marks as excluded from the average.

    Returns
    -------
    built (dict[int, dict])
        Label row -> the :func:`usv_squeak_window_input` dict plus ``chs_count``; segments too short for
        one STFT window are absent.

    Raises
    ------
    ValueError
        A summary row index is out of range or the row's start moved since the sample was drawn.
    """

    root = pathlib.Path(configure_path(session_dir))
    usv_summary_loc = first_match_or_raise(root=root / "audio", pattern="*_usv_summary.csv", recursive=True, label="USV summary CSV")
    summary = pls.read_csv(str(usv_summary_loc), columns=["start", "chs_count"], infer_schema_length=None)
    summary_start = summary["start"].cast(pls.Float64).to_numpy()
    summary_chs = summary["chs_count"].cast(pls.Float64).to_numpy()
    wav_paths = squeak_wav_channels(root, exclude_metadata_audio_channels, lambda *_args, **_kwargs: None)
    handles = [sf.SoundFile(str(path), mode="r") for path in wav_paths]
    built = {}
    try:
        for row, row_index, start, stop in segments:
            if not 0 <= row_index < summary.height or abs(summary_start[row_index] - start) > USV_SQUEAK_START_TOLERANCE_S:
                error_message = f"{usv_summary_loc}: row {row_index} does not start at the labelled {start} s; the summary changed since sampling."
                raise ValueError(error_message)
            window = usv_squeak_window_input(handles, handles[0].frames, start, stop, USV_SQUEAK_INPUT_CONTRACT)
            if window is not None:
                built[row] = {**window, "chs_count": float(summary_chs[row_index])}
    finally:
        for handle in handles:
            handle.close()
    return built


def build_usv_squeak_training_inputs(
    labels: pls.DataFrame,
    exclude_metadata_audio_channels: bool,
    n_workers: int,
    message_output: Callable,
) -> tuple[list[dict], np.ndarray]:
    """
    Description
    -----------
    Builds every labelled segment's call-class input exactly as ``detect-usv-squeaks`` builds it at
    inference (:func:`usv_squeak_window_input` with ``USV_SQUEAK_INPUT_CONTRACT``, over the same
    per-channel ``audio/hpss`` wavs and channel exclusion). Sessions are processed in ``n_workers``
    threads (the work is the latency of reading short windows from many wavs on a network share), each
    session's wavs opened once; results are collected by label row, so the order does not depend on the
    thread schedule. Segments too short for one STFT window are left out and reported.

    Parameters
    ----------
    labels (pls.DataFrame)
        Training labels (:func:`read_usv_squeak_labels`).
    exclude_metadata_audio_channels (bool)
        Drop channels the session metadata marks as excluded from the average (as the detector does).
    n_workers (int)
        Sessions read concurrently (threads).
    message_output (Callable)
        Logging callback.

    Returns
    -------
    windows (list[dict])
        Per kept segment, in label order, the window dict (``x``, ``read_start_s``, ``first_frame``,
        ``n_segment_frames``, ``chs_count``).
    kept_rows (np.ndarray)
        Indices into ``labels`` of the kept segments.
    """

    by_session: dict[str, list[tuple[int, int, float, float]]] = {}
    for row, (session_dir, row_index, start, stop) in enumerate(labels.select("session_dir", "row_index", "start", "stop").iter_rows()):
        by_session.setdefault(session_dir, []).append((row, int(row_index), float(start), float(stop)))
    built: dict[int, dict] = {}
    with ThreadPoolExecutor(max_workers=max(1, n_workers)) as pool:
        futures = [pool.submit(session_usv_squeak_training_inputs, session_dir, segments, exclude_metadata_audio_channels)
                   for session_dir, segments in by_session.items()]
        for session_number, future in enumerate(as_completed(futures), start=1):
            built.update(future.result())
            if session_number % 50 == 0 or session_number == len(futures):
                message_output(f"Built call-class training inputs for {session_number}/{len(futures)} session(s), {len(built)} segment(s).")
    kept_rows = np.array(sorted(built), dtype=np.int64)
    if kept_rows.size < labels.height:
        dropped = sorted(set(range(labels.height)) - set(kept_rows.tolist()))
        message_output(f"{len(dropped)} labelled segment(s) are too short for one STFT window and are left out: {[labels['sample_id'][row] for row in dropped]}.")
    return [built[row] for row in kept_rows], kept_rows


def usv_squeak_frame_targets(windows: list[dict], squeak_extents: list[str]) -> list[np.ndarray]:
    """
    Description
    -----------
    Builds each segment's frame target: 1 on every frame whose centre (``read_start_s + k * hop``)
    lies inside any labelled span (closed interval), 0 elsewhere. The labeller marked every squeak in
    view, the context included, so a target can be 1 outside the segment.

    Parameters
    ----------
    windows (list[dict])
        Window dicts (``x``, ``read_start_s``) of the kept segments.
    squeak_extents (list[str])
        The same segments' ``squeak_extents_s`` JSON texts.

    Returns
    -------
    targets (list[np.ndarray])
        Per segment, the ``(T,)`` float32 frame target.
    """

    hop_s = HOP_SAMPLES / NOISE_SAMPLING_RATE
    targets = []
    for window, extents in zip(windows, squeak_extents, strict=True):
        centres = window["read_start_s"] + np.arange(window["x"].shape[2]) * hop_s
        target = np.zeros(window["x"].shape[2], dtype=np.float32)
        for span_start, span_end in json.loads(extents):
            target[(centres >= span_start) & (centres <= span_end)] = 1.0
        targets.append(target)
    return targets


def pad_frame_targets(targets: list[np.ndarray], width: int) -> np.ndarray:
    """
    Description
    -----------
    Pads per-segment frame targets to a batch width with zeros (padded frames are never supervised).

    Parameters
    ----------
    targets (list[np.ndarray])
        ``(T_i,)`` targets.
    width (int)
        Batch width.

    Returns
    -------
    padded (np.ndarray)
        ``(B, width)`` float32.
    """

    padded = np.zeros((len(targets), width), dtype=np.float32)
    for position, target in enumerate(targets):
        padded[position, :target.size] = target
    return padded


def train_usv_squeak_seed(
    inputs: list[np.ndarray],
    standardized_scalars: np.ndarray,
    y: np.ndarray,
    frame_targets: list[np.ndarray],
    seed: int,
    recipe: dict,
    trunk_state_dict: dict | None,
    device: torch.device,
    message_output: Callable,
) -> USVSqueakTimeMIL:
    """
    Description
    -----------
    Trains one ensemble member with the noise model's recipe on the joint loss. The network is
    initialized after ``torch.manual_seed(seed)`` and, when ``trunk_state_dict`` is given, its trunk
    (``body.*`` / ``tcn.*``) is loaded from a noise-model member (the pretrained initialization). Every
    epoch visits the segments in a ``numpy.random.default_rng(seed)`` permutation in padded batches of
    ``recipe['batch_size']``, augmented by :func:`detect_usv_noise.augment_noise_batch` (a CPU generator
    seeded with ``seed``; the indicator channel is never augmented). The loss is the class
    cross-entropy (label smoothing ``recipe['label_smoothing']``; with ``recipe['class_weighted']``
    weighted by inverse training class frequency) plus ``recipe['frame_loss_weight']`` times the frame
    binary cross-entropy, averaged over the supervised frames of the batch: every valid frame of a
    squeak / both segment (the labeller marked every squeak in view), but only the segment's own frames
    of a usv segment (the label says the segment holds no squeak, not that its context holds none).
    Adam (``learning_rate``, ``weight_decay``) under a cosine schedule over all epochs, stepped per batch.

    Parameters
    ----------
    inputs (list[np.ndarray])
        Per-segment ``(3, 128, T)`` inputs.
    standardized_scalars (np.ndarray)
        ``(N, 2)`` float32 scalars standardized over the training set.
    y (np.ndarray)
        ``(N,)`` int64 class indices.
    frame_targets (list[np.ndarray])
        Per-segment ``(T,)`` frame targets (:func:`usv_squeak_frame_targets`).
    seed (int)
        Seed of this member.
    recipe (dict)
        ``epochs``, ``batch_size``, ``learning_rate``, ``weight_decay``, ``label_smoothing``,
        ``frame_loss_weight`` and ``class_weighted``.
    trunk_state_dict (dict | None)
        Noise-model member weights whose trunk initializes this member, or None to train from scratch.
    device (torch.device)
        Training device.
    message_output (Callable)
        Logging callback.

    Returns
    -------
    model (USVSqueakTimeMIL)
        The trained member, in eval mode on ``device``.

    Raises
    ------
    ValueError
        The trunk weights hold keys the network does not have.
    """

    torch.manual_seed(seed)
    model = USVSqueakTimeMIL(in_channels=inputs[0].shape[0], n_scalars=standardized_scalars.shape[1], n_classes=len(CALL_CLASSES)).to(device)
    if trunk_state_dict is not None:
        trunk = {key: value for key, value in trunk_state_dict.items() if key.startswith(("body.", "tcn."))}
        loaded = model.load_state_dict(trunk, strict=False)
        if loaded.unexpected_keys:
            error_message = f"The noise trunk holds keys the call-class network lacks: {loaded.unexpected_keys}."
            raise ValueError(error_message)
    counts = np.bincount(y, minlength=len(CALL_CLASSES)).astype(np.float64)
    class_weight = (torch.tensor(counts.sum() / (len(CALL_CLASSES) * counts), dtype=torch.float32, device=device)
                    if recipe['class_weighted'] else None)
    class_loss = nn.CrossEntropyLoss(weight=class_weight, label_smoothing=recipe['label_smoothing'])
    optimizer = torch.optim.Adam(model.parameters(), lr=recipe['learning_rate'], weight_decay=recipe['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=recipe['epochs'] * math.ceil(len(inputs) / recipe['batch_size']))
    rng = np.random.default_rng(seed)
    generator = torch.Generator().manual_seed(seed)
    for epoch in range(recipe['epochs']):
        model.train()
        order = rng.permutation(len(inputs))
        losses = []
        for start in range(0, len(order), recipe['batch_size']):
            chunk = order[start:start + recipe['batch_size']]
            x, valid = pad_noise_batch([inputs[i] for i in chunk])
            x_tensor = torch.from_numpy(x).to(device)
            valid_tensor = torch.from_numpy(valid).to(device)
            segment_tensor = x_tensor[:, -1, 0, :] > 0.5
            frame_target = torch.from_numpy(pad_frame_targets([frame_targets[i] for i in chunk], x.shape[3])).to(device)
            has_squeak = torch.from_numpy(y[chunk] > 0).to(device)
            supervised = torch.where(has_squeak[:, None], valid_tensor, segment_tensor)
            x_tensor = augment_noise_batch(x_tensor, valid_tensor, generator)
            optimizer.zero_grad()
            class_logits, frame_logits = model(x_tensor, valid_tensor, segment_tensor, torch.from_numpy(standardized_scalars[chunk]).to(device))
            frame_loss = nn.functional.binary_cross_entropy_with_logits(frame_logits[supervised], frame_target[supervised])
            loss = class_loss(class_logits, torch.from_numpy(y[chunk]).to(device)) + recipe['frame_loss_weight'] * frame_loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach()))
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == recipe['epochs']:
            message_output(f"  seed {seed}: epoch {epoch + 1}/{recipe['epochs']}, mean training loss {np.mean(losses):.4f}.")
    model.eval()
    return model


class USVSqueakModelTrainer:
    """
    Description
    -----------
    Trains the call-class ensemble on the labelling tool's files and writes a bundle
    ``detect-usv-squeaks`` loads unchanged.
    """

    def __init__(
        self,
        bundle_path: str | None = None,
        input_parameter_dict: dict | None = None,
        message_output: Callable | None = None,
    ) -> None:
        """
        Description
        -----------
        Initializes the USVSqueakModelTrainer.

        Parameters
        ----------
        bundle_path (str)
            Output bundle (``.pt``); must not exist yet.
        input_parameter_dict (dict)
            Processing settings; the ``train_usv_squeak_model`` block supplies the label sets and
            overrides, the initialization, the squeak-extent threshold, the channel-exclusion switch, the seeds and the
            recipe (and ``detect_usv_noise.noise_model_path`` the pretrained trunk).
        message_output (Callable)
            Logging callback; defaults to ``print``.

        Returns
        -------
        None
        """

        self.bundle_path = bundle_path
        self.input_parameter_dict = input_parameter_dict if input_parameter_dict is not None else {}
        self.message_output = message_output if message_output is not None else print

    def train(self) -> pathlib.Path:
        """
        Description
        -----------
        Reads and checks the labels (before any audio is read, so a bad file fails fast), builds every
        labelled segment's input with the detector's own extraction, standardizes the scalars over the
        training set, trains one member per seed (cuDNN in deterministic mode for the run; a GPU is still
        not bit-reproducible, so a retrained ensemble matches an earlier one in its decisions, not its
        weights), writes the bundle with the input contract, the scalar standardization, the class
        names, the squeak-extent threshold, the recipe and the provenance, and loads it back through
        :func:`load_usv_squeak_model` to prove ``detect-usv-squeaks`` accepts it. With ``pretrained``
        the trunk of member ``i`` starts from member ``seed_i mod n`` of the noise ensemble of
        ``detect_usv_noise.noise_model_path`` (derived from ``spectrograms_root`` when empty).

        Parameters
        ----------

        Returns
        -------
        bundle_path (pathlib.Path)
            The written bundle.
        """

        start_time = datetime.now()
        self.message_output(f"Call-class model training started at: {start_time.hour:02d}:{start_time.minute:02d}:{start_time.second:02d}.")
        derive_spectrogram_model_paths(self.input_parameter_dict)
        cfg = self.input_parameter_dict['train_usv_squeak_model']
        bundle_path = pathlib.Path(configure_path(self.bundle_path))
        if bundle_path.exists():
            error_message = f"{bundle_path} already exists; choose a new bundle path (a trained bundle is never overwritten)."
            raise FileExistsError(error_message)
        if not cfg['seeds'] or len(set(cfg['seeds'])) != len(cfg['seeds']):
            error_message = f"train_usv_squeak_model['seeds'] must list distinct seeds, one per ensemble member; got {cfg['seeds']}."
            raise ValueError(error_message)
        if not 0.0 < cfg['span_threshold'] < 1.0:
            error_message = "train_usv_squeak_model needs 0 < span_threshold < 1."
            raise ValueError(error_message)
        labels = read_usv_squeak_labels(cfg['label_sets'], cfg['label_overrides'], self.message_output)
        noise_state_dicts = None
        if cfg['pretrained']:
            noise_model_path = self.input_parameter_dict['detect_usv_noise']['noise_model_path']
            load_noise_model(noise_model_path, torch.device("cpu"))
            noise_state_dicts = torch.load(pathlib.Path(configure_path(noise_model_path)), map_location="cpu", weights_only=True)["state_dicts"]
            self.message_output(f"Trunks start from the {len(noise_state_dicts)} members of {noise_model_path}.")

        windows, kept_rows = build_usv_squeak_training_inputs(labels, cfg['exclude_metadata_audio_channels'], cfg['n_workers'], self.message_output)
        kept = labels[kept_rows.tolist()]
        inputs = [window["x"] for window in windows]
        y = kept["label"].to_numpy().astype(np.int64)
        frame_targets = usv_squeak_frame_targets(windows, kept["squeak_extents_s"].to_list())
        raw_scalars = np.stack([noise_scalars(window["chs_count"], start, stop)
                                for window, start, stop in zip(windows, kept["start"].to_list(), kept["stop"].to_list(), strict=True)])
        scalar_mean = raw_scalars.mean(0)
        scalar_std = raw_scalars.std(0) + 1e-6
        standardized = ((raw_scalars - scalar_mean) / scalar_std).astype(np.float32)
        recipe = {
            "epochs": cfg['epochs'], "batch_size": cfg['batch_size'], "learning_rate": cfg['learning_rate'],
            "weight_decay": cfg['weight_decay'], "label_smoothing": cfg['label_smoothing'],
            "frame_loss_weight": cfg['frame_loss_weight'], "class_weighted": bool(cfg['class_weighted']),
        }

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.message_output(f"Training {len(cfg['seeds'])} member(s) on {len(inputs)} segment(s) on {device}.")
        deterministic, benchmark = torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        state_dicts = []
        try:
            for seed in cfg['seeds']:
                trunk = None if noise_state_dicts is None else noise_state_dicts[int(seed) % len(noise_state_dicts)]
                model = train_usv_squeak_seed(inputs, standardized, y, frame_targets, int(seed), recipe, trunk, device, self.message_output)
                state_dicts.append({key: value.detach().cpu() for key, value in model.state_dict().items()})
        finally:
            torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = deterministic, benchmark

        bundle = {
            "model": USV_SQUEAK_MODEL_NAME,
            "state_dicts": state_dicts,
            "class_names": list(CALL_CLASSES),
            "in_channels": int(inputs[0].shape[0]), "n_scalars": int(raw_scalars.shape[1]),
            "scalar_names": list(NOISE_SCALAR_NAMES),
            "scalar_mean": scalar_mean.tolist(), "scalar_std": scalar_std.tolist(),
            **{key: ([list(band) for band in value] if key == "bands_hz" else value) for key, value in USV_SQUEAK_INPUT_CONTRACT.items()},
            "sampling_rate": NOISE_SAMPLING_RATE, "hop_samples": HOP_SAMPLES,
            "input": "two-band absolute-dB spectrogram of the whole window (context_frames hops either side of the segment, read as "
                     "window_segment_input reads it), NOT cropped; channel 3 = 1 on the segment's frames, 0 on context frames",
            "extent_rule": {
                "threshold": float(cfg['span_threshold']),
                "rule": "one extent per segment: envelope of the frames above threshold in runs touching the segment, "
                        "no minimum run; the highest-scoring segment frame when none",
            },
            "recipe": {
                **recipe, "pretrained": bool(cfg['pretrained']), "seeds": [int(seed) for seed in cfg['seeds']],
                "augmentation": {
                    "gain_db": NOISE_AUG_GAIN_DB, "freq_roll_rows": NOISE_AUG_FREQ_ROLL,
                    "time_mask_fraction": NOISE_AUG_TIME_MASK_FRACTION, "freq_mask_rows": NOISE_AUG_FREQ_MASK_ROWS,
                },
                "exclude_metadata_audio_channels": bool(cfg['exclude_metadata_audio_channels']),
            },
            "labels": [str(label_set['labels_csv']) for label_set in cfg['label_sets']],
            "label_overrides": [[override['name'], str(override['override_csv'])] for override in cfg['label_overrides']],
            "n_labels": {name: int((y == index).sum()) for index, name in enumerate(CALL_CLASSES)},
            "built_by": "usv_playpen train-usv-squeak-model",
            "created": datetime.now().isoformat(timespec="seconds"),
        }
        bundle_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(bundle, bundle_path)
        load_usv_squeak_model(str(bundle_path), torch.device("cpu"))

        elapsed_minutes = (datetime.now() - start_time).total_seconds() / 60
        self.message_output(f"Wrote {bundle_path} ({bundle_path.stat().st_size / 1e6:.1f} MB) in {elapsed_minutes:.1f} min; detect-usv-squeaks loads it.")
        return bundle_path


def squeak_crop_window(
    segment_start_s: np.ndarray,
    segment_stop_s: np.ndarray,
    squeak_start_s: np.ndarray,
    squeak_end_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Description
    -----------
    The audio window a squeak's sonic spectrogram is built from: the segment ``[start, stop]`` widened,
    where needed, to hold the squeak envelope ``[squeak_start, squeak_end]`` plus
    ``SQUEAK_QLVM_CONTEXT_FRAMES`` hops either side, starting no earlier than 0 s. When the envelope and
    its context lie inside the segment the window IS the segment, which is how the squeak QLVM cells'
    training crops were built; a squeak that extends past its segment (common: the ultrasonic segmenter
    cuts the segment, the squeak track does not) is then cropped whole instead of cut at the boundary.
    The window also sets the channel weights of the variance-weighted average.

    Parameters
    ----------
    segment_start_s (np.ndarray)
        ``(N,)`` segment ``start`` (s).
    segment_stop_s (np.ndarray)
        ``(N,)`` segment ``stop`` (s).
    squeak_start_s (np.ndarray)
        ``(N,)`` ``squeak_start`` (s).
    squeak_end_s (np.ndarray)
        ``(N,)`` ``squeak_end`` (s).

    Returns
    -------
    window_start_s (np.ndarray)
        ``(N,)`` float64 window starts (s); frame ``t`` of the window's spectrogram is centred at
        ``window_start_s + t * FRAME_DT_S``.
    window_stop_s (np.ndarray)
        ``(N,)`` float64 window stops (s).
    """

    context_s = SQUEAK_QLVM_CONTEXT_FRAMES * FRAME_DT_S
    window_start = np.maximum(np.minimum(np.asarray(segment_start_s, dtype=np.float64), np.asarray(squeak_start_s, dtype=np.float64) - context_s), 0.0)
    window_stop = np.maximum(np.asarray(segment_stop_s, dtype=np.float64), np.asarray(squeak_end_s, dtype=np.float64) + context_s)
    return window_start, window_stop


def squeak_window_n_frames(window_start_s: np.ndarray, window_stop_s: np.ndarray) -> np.ndarray:
    """
    Description
    -----------
    Number of frames of each window's sonic spectrogram (:func:`squeak_window_spectrograms`), without
    reading audio: the window spans ``round(stop * 250000) - round(start * 250000)`` samples, and the
    centred STFT (``nperseg`` 2048, hop 512) of ``n`` samples has ``1 + n // 512`` frames; a window
    shorter than one STFT window has none (0). A window running past the end of the recording has fewer
    frames than this when it is read.

    Parameters
    ----------
    window_start_s (np.ndarray)
        ``(N,)`` window starts (s).
    window_stop_s (np.ndarray)
        ``(N,)`` window stops (s).

    Returns
    -------
    n_frames (np.ndarray)
        ``(N,)`` int64 frame counts.
    """

    n_samples = (np.round(np.asarray(window_stop_s, dtype=np.float64) * SQUEAK_SAMPLING_RATE).astype(np.int64)
                 - np.round(np.asarray(window_start_s, dtype=np.float64) * SQUEAK_SAMPLING_RATE).astype(np.int64))
    return np.where(n_samples >= SQUEAK_SPEC_PARAMS["nperseg"], 1 + n_samples // SQUEAK_SPEC_PARAMS["hop_length"], 0).astype(np.int64)


def squeak_window_spectrograms(
    session_root: pathlib.Path,
    window_start_s: np.ndarray,
    window_stop_s: np.ndarray,
    exclude_metadata_audio_channels: bool,
    message_output: Callable,
) -> list[np.ndarray | None]:
    """
    Description
    -----------
    Rebuilds the full-length, absolute-dB sonic spectrogram of each requested audio window from the
    session's unfiltered HPSS wavs, with the squeak QLVM cells' front end (``SQUEAK_SPEC_PARAMS``:
    Blackman-Harris STFT, nperseg 2048, hop 512, centred, 3-30 kHz, 128 linear frequency bins,
    ``ref=1.0``, no ``top_db`` clamp, variance-weighted channel average; the front end that reproduces
    the reference ``_sonic_wav_`` store bit-exactly). A window spans ``round(start * 250000)`` to
    ``round(stop * 250000)`` samples on every channel, and frame ``t`` of its spectrogram is centred at
    ``start + t * FRAME_DT_S`` s. The squeak QLVM embedding (:func:`squeak_qlvm_inputs`) and the squeak
    training-set builder read their spectrograms here.

    Parameters
    ----------
    session_root (pathlib.Path)
        Session root directory.
    window_start_s (np.ndarray)
        ``(N,)`` window starts (s), e.g. from :func:`squeak_crop_window`.
    window_stop_s (np.ndarray)
        ``(N,)`` window stops (s).
    exclude_metadata_audio_channels (bool)
        Whether to drop metadata-excluded channels from the average.
    message_output (Callable)
        Logging callback.

    Returns
    -------
    spectrograms (list[np.ndarray | None])
        One entry per window, in order: the float32 ``(128, n_frames)`` absolute-dB spectrogram, or None
        when the window is too short for a single STFT frame.
    """

    wav_paths = squeak_wav_channels(session_root, exclude_metadata_audio_channels, message_output)
    spectrograms: list[np.ndarray | None] = []
    handles = [sf.SoundFile(str(wav_path), mode="r") for wav_path in wav_paths]
    try:
        for start_s, stop_s in zip(np.asarray(window_start_s, dtype=np.float64), np.asarray(window_stop_s, dtype=np.float64), strict=True):
            first_sample = round(float(start_s) * SQUEAK_SAMPLING_RATE)
            last_sample = round(float(stop_s) * SQUEAK_SAMPLING_RATE)
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


def squeak_extent_frames(
    window_start_s: np.ndarray,
    squeak_start_s: np.ndarray,
    squeak_end_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Description
    -----------
    The first and last frame of a window's spectrogram whose centres lie inside the squeak extent
    ``[squeak_start, squeak_end]`` (closed): ``ceil((squeak_start - window_start) / FRAME_DT_S)`` and
    ``floor((squeak_end - window_start) / FRAME_DT_S)``, with a 1e-9-frame tolerance against float
    error. This is the frame rule the call-class model's frame targets use, so a squeak's crop holds
    exactly the frames its span was labelled on.

    Parameters
    ----------
    window_start_s (np.ndarray)
        ``(N,)`` window starts (s; frame 0's centre).
    squeak_start_s (np.ndarray)
        ``(N,)`` ``squeak_start`` (s).
    squeak_end_s (np.ndarray)
        ``(N,)`` ``squeak_end`` (s).

    Returns
    -------
    first (np.ndarray)
        ``(N,)`` int64 first frame inside the extent.
    last (np.ndarray)
        ``(N,)`` int64 last frame inside the extent (inclusive).
    """

    window_start_s = np.asarray(window_start_s, dtype=np.float64)
    first = np.ceil((np.asarray(squeak_start_s, dtype=np.float64) - window_start_s) / FRAME_DT_S - 1e-9).astype(np.int64)
    last = np.floor((np.asarray(squeak_end_s, dtype=np.float64) - window_start_s) / FRAME_DT_S + 1e-9).astype(np.int64)
    return first, last


def squeak_crop_frames(
    window_start_s: np.ndarray,
    squeak_start_s: np.ndarray,
    squeak_end_s: np.ndarray,
    n_frames: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Description
    -----------
    The first and last spectrogram frame of each squeak's QLVM crop: the frames inside the squeak extent
    (:func:`squeak_extent_frames`), widened by ``SQUEAK_QLVM_CONTEXT_FRAMES`` either side and clipped to
    the window's frames ``0 .. n_frames - 1``. This is the rule of the reference builder
    ``build_bbv_dataset.py``, except that its window was the segment's first 128 frames (the store it
    cropped from held no more) and its extent frames were the frames of the retired classifier.

    Parameters
    ----------
    window_start_s (np.ndarray)
        ``(N,)`` window start (s) of each spectrogram (:func:`squeak_crop_window`).
    squeak_start_s (np.ndarray)
        ``(N,)`` ``squeak_start`` (s).
    squeak_end_s (np.ndarray)
        ``(N,)`` ``squeak_end`` (s).
    n_frames (np.ndarray)
        ``(N,)`` number of frames of each window's spectrogram.

    Returns
    -------
    first (np.ndarray)
        ``(N,)`` int64 first frame of each crop.
    last (np.ndarray)
        ``(N,)`` int64 last frame of each crop (inclusive).
    """

    first, last = squeak_extent_frames(window_start_s, squeak_start_s, squeak_end_s)
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
    The USV summary rows the squeak QLVM embedding (and the squeak spectrogram store) considers:
    ``squeak`` true (pure squeaks and "both", :func:`os_utils.squeak_bearing_mask`) and not noise. A
    noise row has null booleans, so the noise test is a guard; a null ``noise`` (a segment the noise model could not score)
    counts as not noise, the single definition of noise the analyses share
    (:func:`os_utils.drop_noise_usvs`).

    Parameters
    ----------
    usv_summary (pls.DataFrame)
        The session's USV summary; must hold ``usv``, ``squeak`` and ``noise``.

    Returns
    -------
    rows (np.ndarray)
        Ascending int64 row indices.

    Raises
    ------
    ValueError
        The summary lacks ``noise``, ``usv``, ``squeak`` or the squeak extent columns.
    """

    missing = [column for column in ("noise", USV_FLAG_COLUMN, SQUEAK_FLAG_COLUMN, "squeak_start", "squeak_end") if column not in usv_summary.columns]
    if missing:
        error_message = (
            f"The USV summary has no {missing} column(s); run detect-usv-noise and detect-usv-squeaks "
            f"on the session before embedding its squeaks."
        )
        raise ValueError(error_message)
    squeaky = squeak_bearing_mask(usv_summary, "the USV summary").to_numpy()
    noise = usv_summary["noise"].cast(pls.Boolean).fill_null(False).to_numpy()
    return np.flatnonzero(squeaky & ~noise).astype(np.int64)


def squeak_qlvm_inputs(
    session_root: pathlib.Path,
    usv_summary: pls.DataFrame,
    exclude_metadata_audio_channels: bool,
    message_output: Callable,
) -> dict:
    """
    Description
    -----------
    Builds the squeak QLVM decoder inputs of one session: selects the rows with ``squeak`` true (pure
    squeaks and "both") that are not noise (:func:`squeak_qlvm_rows`), leaves out rows without a
    squeak extent (only possible in a hand-edited summary: every squeak row gets one), rebuilds each remaining row's sonic spectrogram over its crop window (the segment
    widened to hold the envelope plus its context, :func:`squeak_crop_window`,
    :func:`squeak_window_spectrograms`), crops each to its envelope plus context
    (:func:`squeak_crop_frames`), leaves out crops narrower than ``SQUEAK_QLVM_MIN_CROP_FRAMES`` (the
    training set's minimum) or wider than the 128-frame decoder frame (never trained on: the decoder
    frame has no room for them and the training crops were never compressed in time), and normalizes
    the rest (:func:`squeak_crop_inputs`). The extent is already the envelope of all the row's
    above-threshold squeak frames, so a row with several squeaks is cropped to their envelope (one row
    holds one pair of coordinates). A fallback extent (one frame) gives a crop of 5 frames, below the
    8-frame minimum, so those rows get nulls.

    Parameters
    ----------
    session_root (pathlib.Path)
        Session root directory.
    usv_summary (pls.DataFrame)
        The session's USV summary (``start``, ``stop``, ``noise``, ``usv``, ``squeak``, ``squeak_start``
        and ``squeak_end``).
    exclude_metadata_audio_channels (bool)
        Whether to drop metadata-excluded channels from the spectrogram average.
    message_output (Callable)
        Logging callback.

    Returns
    -------
    squeak_inputs (dict)
        ``row_index`` (``(M,)`` int64 summary rows embedded), ``window_start_s`` (``(M,)`` float64),
        ``first`` / ``last`` (``(M,)`` int64 crop frames of the window), ``inputs`` (``(M, 128, 128)``
        float32 decoder inputs), ``n_candidates`` (squeak / both rows that are not noise) and
        ``excluded`` (reason -> number of candidate rows left out: ``"no squeak extent"``,
        ``"no spectrogram"``, ``"crop < 8 frames"``, ``"crop > 128 frames"``).
    """

    candidates = squeak_qlvm_rows(usv_summary)
    empty = {
        "row_index": np.empty(0, dtype=np.int64),
        "window_start_s": np.empty(0, dtype=np.float64),
        "first": np.empty(0, dtype=np.int64),
        "last": np.empty(0, dtype=np.int64),
        "inputs": np.empty((0, *SQUEAK_QLVM_TARGET_SHAPE), dtype=np.float32),
        "n_candidates": int(candidates.size),
    }
    excluded = {"no squeak extent": 0, "no spectrogram": 0, "crop < 8 frames": 0, "crop > 128 frames": 0}
    if candidates.size == 0:
        return {**empty, "excluded": excluded}

    squeak_start = usv_summary["squeak_start"].cast(pls.Float64).fill_null(np.nan).to_numpy()[candidates]
    squeak_end = usv_summary["squeak_end"].cast(pls.Float64).fill_null(np.nan).to_numpy()[candidates]
    has_extent = np.isfinite(squeak_start) & np.isfinite(squeak_end)
    excluded["no squeak extent"] = int(np.count_nonzero(~has_extent))
    with_extent = np.flatnonzero(has_extent)
    if with_extent.size == 0:
        return {**empty, "excluded": excluded}
    window_start, window_stop = squeak_crop_window(
        usv_summary["start"].cast(pls.Float64).to_numpy()[candidates[with_extent]],
        usv_summary["stop"].cast(pls.Float64).to_numpy()[candidates[with_extent]],
        squeak_start[with_extent],
        squeak_end[with_extent],
    )
    spectrograms = squeak_window_spectrograms(
        session_root=session_root,
        window_start_s=window_start,
        window_stop_s=window_stop,
        exclude_metadata_audio_channels=exclude_metadata_audio_channels,
        message_output=message_output,
    )
    has_spectrogram = np.array([spectrogram is not None for spectrogram in spectrograms], dtype=bool)
    excluded["no spectrogram"] = int(np.count_nonzero(~has_spectrogram))
    usable = np.flatnonzero(has_spectrogram)
    n_frames = np.array([spectrograms[position].shape[1] for position in usable], dtype=np.int64)
    first, last = squeak_crop_frames(
        window_start_s=window_start[usable],
        squeak_start_s=squeak_start[with_extent][usable],
        squeak_end_s=squeak_end[with_extent][usable],
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
        "row_index": candidates[with_extent[usable[keep]]],
        "window_start_s": window_start[usable[keep]],
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
    Places every ``squeak`` / ``both`` row of one session that is not noise on the torus of a squeak
    (BBV) QLVM cell and merges ``qlvm_squeak1`` / ``qlvm_squeak2`` into its ``*_usv_summary.csv``.
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
        session's ``squeak`` / ``both`` rows that are not noise (:func:`squeak_qlvm_inputs`, one crop
        per row over its squeak envelope), embeds them as the posterior mean over the cell's lattice
        (:func:`qlvm_model.embed_data`) and writes ``qlvm_squeak1`` / ``qlvm_squeak2`` (torus
        coordinates in ``[0, 1)``) on those rows and nulls on every other row, replacing any earlier
        squeak coordinates. Run it after ``detect-usv-noise`` and ``detect-usv-squeaks``. The summary
        is rewritten atomically.

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
            f"{squeak_inputs['n_candidates']} squeak / both segments that are not noise, {squeak_inputs['row_index'].size} embedded{null_note}."
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
            f"Merged the squeak torus coordinates of {squeak_inputs['row_index'].size} segments into {usv_summary_loc.name}."
        )
        self.message_output(
            f"Squeak QLVM embedding ended at: {datetime.now().hour:02d}:{datetime.now().minute:02d}:{datetime.now().second:02d}."
        )


@click.command(name="detect-usv-squeaks")
@click.option('--root-directory', type=click.Path(exists=True, file_okay=False, dir_okay=True), required=True, help='Session root directory path.')
@click.option('--squeak-model-path', 'squeak_model_path', type=str, default=None, required=False, help='Path to the call-class (usv / squeak / both) model bundle (.pt); derived from spectrograms_root when empty.')
@click.option('--exclude-metadata-audio-channels/--no-exclude-metadata-audio-channels', 'exclude_metadata_audio_channels', default=None, required=False, help='Drop channels the session metadata marks as excluded from the spectrogram average.')
@click.option('--batch-size', 'batch_size', type=int, default=None, required=False, help='Segments per forward pass at the typical window length; a batch is budgeted at batch-size x 128 frame slots, so one long window never inflates it.')
@click.pass_context
def detect_usv_squeaks_cli(ctx, root_directory, **kwargs) -> None:
    """
    Description
    -----------
    A command-line tool to classify every non-noise USV segment of a session as usv / squeak / both,
    locate its squeaks, and merge the call-class columns into its USV summary CSV.

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


@click.command(name="train-usv-squeak-model")
@click.option('--bundle-path', 'bundle_path', type=click.Path(file_okay=True, dir_okay=False), required=True, help='Output bundle (.pt); must not exist yet.')
@click.option('--label-set', 'label_set', type=(str, str, str), multiple=True, default=None, required=False, help='NAME LABELS_CSV SAMPLE_CSV of one label set (labelling-tool labels and their sample); repeat per set (replaces the label_sets setting).')
@click.option('--label-override', 'label_override', type=(str, str), multiple=True, default=None, required=False, help='NAME OVERRIDE_CSV: a review CSV (labels format) whose panels replace those of label set NAME; repeat per file (replaces the label_overrides setting).')
@click.option('--pretrained/--no-pretrained', 'pretrained', default=None, required=False, help='Initialize every member\'s trunk from the noise ensemble (detect_usv_noise.noise_model_path).')
@click.option('--span-threshold', 'span_threshold', type=float, default=None, required=False, help='Frame squeak-probability threshold of the squeak-extent rule the bundle carries.')
@click.option('--exclude-metadata-audio-channels/--no-exclude-metadata-audio-channels', 'exclude_metadata_audio_channels', default=None, required=False, help='Drop channels the session metadata marks as excluded from the spectrogram average (keep it as detect-usv-squeaks runs).')
@click.option('--n-workers', 'n_workers', type=int, default=None, required=False, help='Sessions whose audio is read concurrently (threads) while the training inputs are built.')
@click.option('--seed', 'seeds', type=int, multiple=True, default=None, required=False, help='Seed of one ensemble member; repeat once per member (replaces the seeds setting).')
@click.option('--epochs', 'epochs', type=int, default=None, required=False, help='Training epochs per member.')
@click.option('--batch-size', 'batch_size', type=int, default=None, required=False, help='Segments per training batch.')
@click.option('--learning-rate', 'learning_rate', type=float, default=None, required=False, help='Adam learning rate (cosine-annealed over the run).')
@click.option('--weight-decay', 'weight_decay', type=float, default=None, required=False, help='Adam weight decay.')
@click.option('--label-smoothing', 'label_smoothing', type=float, default=None, required=False, help='Label smoothing of the class cross-entropy.')
@click.option('--frame-loss-weight', 'frame_loss_weight', type=float, default=None, required=False, help='Weight of the frame squeak loss relative to the class loss.')
@click.option('--class-weighted/--no-class-weighted', 'class_weighted', default=None, required=False, help='Weight the class loss by inverse training class frequency.')
@click.pass_context
def train_usv_squeak_model_cli(ctx, bundle_path, label_set, label_override, **kwargs) -> None:
    """
    Description
    -----------
    A command-line tool to train the call-class (usv / squeak / both) ensemble on the labelling tool's
    files and write a bundle detect-usv-squeaks loads unchanged.

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
        block='train_usv_squeak_model',
    )
    if label_set:
        processing_settings_dict['train_usv_squeak_model']['label_sets'] = [
            {"name": name, "labels_csv": labels_csv, "sample_csv": sample_csv} for name, labels_csv, sample_csv in label_set
        ]
    if label_override:
        processing_settings_dict['train_usv_squeak_model']['label_overrides'] = [
            {"name": name, "override_csv": override_csv} for name, override_csv in label_override
        ]

    USVSqueakModelTrainer(
        bundle_path=bundle_path,
        input_parameter_dict=processing_settings_dict,
        message_output=print,
    ).train()


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
    A command-line tool to place a session's squeak / both segments that are not noise on the torus of a
    squeak (BBV) QLVM cell and merge ``qlvm_squeak1`` / ``qlvm_squeak2`` into its USV summary CSV.

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
