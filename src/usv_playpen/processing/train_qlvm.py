"""
@author: bartulem
Train a QLVM (QMC latent-variable model) decoder on a prebuilt USV (or BBV)
spectrogram training set and write it as a QLVM model package cell.

This is the in-house JAX port of the recipe the shipped models were trained with
(qmc_deep_gen's ``bartul_mouse.run_mouse_experiments``, run under the "shipped"
protocol of the v3 package: constant learning rate, last-epoch checkpoint, and
the ReLU head for phases 6 and 9 or the legacy head for phase 3). It trains in the
same framework, and with the same decoder, likelihood and lattice code, that
``infer-qlvm-latents`` embeds with (:mod:`qlvm_model`); torch is used only to
write the checkpoint in the format every QLVM model package cell ships.

The recipe, step by step:

* **Data.** A dataset directory in the format ``build-qlvm-training-set`` and the
  model-package builders write: ``train_data.npz`` (or ``full_data.npz``), an
  optional ``val_data.npz`` and a ``metadata.npz``. Each split holds
  ``spectrograms`` ``(N, 128, 128)``, ``masks``, ``masks_len`` and ``durations``,
  and may declare ``apply_mask``. Every spectrogram is min-max normalized on its
  own, ``(x - min) / (max - min + 1e-8)`` in float32, and then multiplied by its
  binarized mask (``mask > 0.5``) when the set applies masks: a split's own
  ``apply_mask`` decides, and a split that does not declare one applies masks when
  ``metadata.npz`` says ``masking_type`` ``"sam"`` (sets built in-house, whose
  spectrograms are already masked, so the product changes nothing). A loudness
  floor, when the set has one (phase 6), is already baked into the stored
  spectrograms; it is recorded in the training contract, not re-applied.
* **Model.** A fixed 2-D Fibonacci lattice of ``fib(training_fib_m)`` points on
  the torus (610 at ``m = 15``) and the QLVM ConvTranspose decoder
  (:func:`qlvm_model.decoder_forward`) with the ``"relu"`` or ``"legacy"`` head,
  initialized as torch initializes it (:func:`qlvm_model.init_decoder_params`).
* **Objective.** For each batch, one ``U[0, 1)^2`` shift of the whole lattice, and
  the negative mean QMC log evidence of the batch under the decoded lattice
  (:func:`qlvm_model.binary_evidence`).
* **Optimization.** Adam (``betas = (0.9, 0.999)``, ``eps = 1e-8``) at a constant
  ``learning_rate``, ``batch_size`` spectrograms per step, the training set
  reshuffled every epoch, ``n_epochs`` epochs; the weights of the last epoch are
  the ones saved.
* **Validation.** Every ``val_freq`` epochs and after the last one, the same
  objective on a fixed validation subset -- at most ``val_samples_per_mask_count``
  spectrograms of each ``masks_len`` value, drawn once with ``seed`` exactly as
  the reference trainer draws them -- one spectrogram at a time, each against its
  own random shift of the ``fib(validation_fib_m)``-point lattice.

The output directory is laid out like a v3 model package cell:
``checkpoint.tar`` (torch zip: the decoder ``state_dict`` under
``"model"`` with ``decoder.`` keys, a torch ``Adam`` ``state_dict`` under
``"optimizer"``, and every batch's training loss under ``"run info"``),
``config/training_contract.json`` (what ``infer-qlvm-latents`` checks its
settings against), ``config/run_config.json`` (the run's settings and data) and
``metrics/val_diagnostics.npz`` (per-epoch training loss, the validation losses
and the validation subset). Clustering the trained torus into
``inference/clusters_<level>/label_grid.npy`` is a separate, later step; until
it exists the cell cannot be passed to ``infer-qlvm-latents``.

With ``conditional`` null the decoder takes no conditioning input (``c_dim``
0). With ``conditional`` one of ``duration``, ``mean_freq``, ``bandwidth`` or
``loudness`` it trains a conditional decoder (``c_dim`` 1) with the recipe of the
v3 package's phase 11 cells (qmc_deep_gen ``bartul_mouse_cond.py`` with
``bin_scheme="quantile_capped"``, ``scale_loss_by_batch=True`` and a per-call
table; ``data/conditionals.py``, ``data/mouse_data.py`` and ``train/train.py``):

* **The value, per call.** ``duration``: ``(d - d_min) / (d_max - d_min + 1e-8)``
  of the pre-resize duration in time bins, ``d_min`` / ``d_max`` the training
  split's (8 and 127 in every shipped cell). ``mean_freq``:
  ``(f - 30 kHz) * 127 / (128 * 90 kHz)`` of the call's mean frequency ``f`` over
  its SAM-masked region (the USV summary's ``mean_freq_hz``), which equals the
  energy-weighted row centroid of the resized SAM-masked spectrogram
  ``infer-qlvm-latents`` computes to 1.5e-7. ``bandwidth``:
  ``clip(freq_bandwidth_hz / 90 kHz, 0, 1)``. ``loudness``:
  ``clip((dB - 28.83) / (95.85 - 28.83), 0, 1)`` of the call's absolute
  image-level loudness over its SAM mask (the summary's ``loudness_db``,
  :func:`compute_usv_loudness.session_image_level_db`; 28.83-95.85 dB is the
  0.1-99.9 percentile range of the package corpus). The raw values come from the
  split's ``mean_freq_hz`` / ``freq_bandwidth_hz`` / ``loudness_db`` columns
  (``build-qlvm-training-set`` copies them from the summaries), or, with
  ``condition_table``, from a per-call ``.npz`` keyed by ``spec_id`` (the package's
  ``corpus/cond_table.npz``, for sets built outside the repository). A row without
  a finite value stops the run.
* **Bins.** The training split's values are cut into ``condition_n_bins`` (32)
  quantile bins; with ``condition_bin_scheme`` ``"quantile_capped"`` every bin
  wider than the widest inner bin is then split into equal-width pieces no wider
  than it (in practice only the two end bins split), and the pieces are kept
  however few rows they hold.
* **Batches.** Every batch is drawn from one bin, because the decoder decodes the
  whole lattice at one conditioning value per batch -- the batch mean of its rows'
  values (:func:`qlvm_model.decode_shifted_lattice`). Each epoch ``e`` (from 0),
  with numpy's ``default_rng(seed + 2 * e + 1)``, the rows of every bin are
  shuffled and chunked into batches of at most ``batch_size`` and the batches of
  all bins are shuffled together: the reference ``ConditionGroupedBatchSampler``
  draw for draw, as the shipped runs used it. That sampler draws epoch ``k`` of its
  own count from ``default_rng(seed + k)``, and torch's multi-worker ``DataLoader``
  (the shipped runs had 16 workers) starts two sampler iterations per epoch (one in
  ``_BaseDataLoaderIter.__init__``, one in ``_MultiProcessingDataLoaderIter._reset``)
  and trains on the second; the shipped phase 11 checkpoints' per-batch losses
  line up with this order and not with ``seed + e``.
* **Loss.** With ``condition_scale_loss_by_batch`` each step's loss is multiplied
  by ``rows / batch_size``, so a 5-row tail batch steps with 5/512 of a full step
  instead of a full one; the recorded loss is the unscaled batch mean. Short
  batches are padded to ``batch_size`` with zero-weight rows, which changes
  neither the loss nor its gradient.
* **Validation.** One call at a time at its own value.
* **Output.** The cell's contract names the condition (a ``condition`` block in
  the form of the shipped phase 11 contracts: the value's definition, its
  constants and its decode rule) and ``config/condition_bins.npz`` records the
  bins (``edges``, ``group_ids``, ``group_sizes``, ``group_means``), the training
  range of the value (``train_c_min``, ``train_c_max``) and the decode grid
  (``decode_grid``, ``condition_decode_grid_step`` apart from ``train_c_min``
  past ``train_c_max``). ``infer-qlvm-latents`` decodes a new call as the package
  decoded its corpus: at its own value clamped to the training range for
  duration and bandwidth (``decode`` ``"exact"``: few distinct values, 120
  durations and 128 bandwidth steps) and at the nearest grid point for mean
  frequency and loudness (``decode`` ``"grid"``).
"""

from __future__ import annotations

import collections
import itertools
import json
import math
import pathlib
import time
from collections.abc import Callable
from datetime import datetime

import click
import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
from click.core import ParameterSource

from ..cli_utils import modify_settings_json_for_cli
from ..os_utils import atomic_output_path
from ..time_utils import is_gui_context, smart_wait
from .build_qlvm_training_set import SUMMARY_CONDITION_COLUMNS
from .qlvm_latents import (
    CONDITION_NAMES,
    compute_condition_values,
    minmax_per_spectrogram,
)
from .qlvm_model import (
    binary_evidence,
    decode_shifted_lattice,
    decoder_parameter_shapes,
    gen_fib_basis,
    init_decoder_params,
)

# The Fibonacci lattice is two-dimensional, so every decoder trained here has a 2-D torus.
LATENT_DIM = 2

# Decoder heads: "relu" (phases 6 and 9) puts a ReLU between the two Linear layers,
# "legacy" (phase 3 and earlier) has none; see qlvm_model.decoder_parameter_shapes.
DECODER_HEADS = ("relu", "legacy")

# Added to each spectrogram's range by the per-spectrogram min-max.
NORMALIZATION_EPSILON = 1e-8

# Adam's constants, torch's defaults (the reference trainer did not change them).
_ADAM_BETAS = (0.9, 0.999)
_ADAM_EPSILON = 1e-8

# Output layout: a v3 model package cell.
CHECKPOINT_NAME = "checkpoint.tar"
CONFIG_DIRECTORY = "config"
METRICS_DIRECTORY = "metrics"

# Rows min-max normalized (and masked) per chunk, which bounds the temporary copies.
_NORMALIZATION_CHUNK = 4096

# Conditional decoders. The split column (a build-qlvm-training-set USV summary
# column, SUMMARY_CONDITION_COLUMNS) each condition's raw per-call value is read
# from; duration uses the split's own durations.
CONDITION_RAW_COLUMNS = {"mean_freq": "mean_freq_hz", "bandwidth": "freq_bandwidth_hz", "loudness": "loudness_db"}

# The v3 package's per-call table (corpus/cond_table.npz) names the loudness column
# image_level_db; a condition_table may use either name.
_CONDITION_TABLE_ALIASES = {"loudness_db": "image_level_db"}

# How infer-qlvm-latents decodes a call, per condition, as the v3 package decoded
# its corpus (qlvm_latents.frozen_condition_values): "exact" at the call's own
# value clamped to the training range, "grid" at the nearest decode_grid point.
CONDITION_DECODE = {"duration": "exact", "mean_freq": "grid", "bandwidth": "exact", "loudness": "grid"}

# Bin schemes of the conditioning value (qmc_deep_gen data/conditionals.py bin_edges).
CONDITION_BIN_SCHEMES = ("quantile", "quantile_capped")

# Frozen maps from raw units to the conditioning value (qmc_deep_gen
# data/conditionals.py CONDITIONAL_REGISTRY), constants rather than per-dataset
# ranges, so a training split and every later call map one call to one value:
# the spectrograms' 30-120 kHz band over 128 frequency rows (mean frequency and
# bandwidth) and the 0.1-99.9 percentile loudness range of the package corpus.
MEAN_FREQ_LOW_HZ = 30000.0
FREQUENCY_SPAN_HZ = 90000.0
FREQUENCY_ROWS = 128
LOUDNESS_DB_RANGE = (28.83, 95.85)

# Epsilon of the duration min-max and of the mean-frequency centroid's energy clamp
# (qmc_deep_gen data/mouse_data.py; recorded in the condition block).
CONDITION_EPSILON = 1e-8


def prepare_split(spectrograms: np.ndarray, masks: np.ndarray | None) -> np.ndarray:
    """
    Description
    -----------
    Turns one split's stored spectrograms into decoder training inputs, in place
    and a chunk of rows at a time: each spectrogram is min-max normalized on its
    own (:func:`qlvm_latents.minmax_per_spectrogram` with ``epsilon = 1e-8``) and,
    when ``masks`` is given, multiplied by its binarized mask (``mask > 0.5``) --
    the order the reference trainer's ``mouse_data.__getitem__`` uses.

    Parameters
    ----------
    spectrograms (np.ndarray)
        ``(N, F, T)`` float32 spectrograms; overwritten with the inputs.
    masks (np.ndarray | None)
        ``(N, F, T)`` masks, or ``None`` for a set whose masks are not applied.

    Returns
    -------
    inputs (np.ndarray)
        The same ``(N, F, T)`` float32 array, now holding values in ``[0, 1)``.
    """
    for start in range(0, spectrograms.shape[0], _NORMALIZATION_CHUNK):
        rows = slice(start, start + _NORMALIZATION_CHUNK)
        chunk = minmax_per_spectrogram(spectrograms[rows], np.float32(NORMALIZATION_EPSILON))
        if masks is not None:
            chunk = chunk * (masks[rows] > 0.5).astype(np.float32)
        spectrograms[rows] = chunk
    return spectrograms


def load_split(npz_path: pathlib.Path, metadata_masking_type: str) -> dict:
    """
    Description
    -----------
    Reads one ``.npz`` split and prepares its decoder inputs (:func:`prepare_split`).
    Whether the split's masks are applied is its own ``apply_mask`` scalar when it
    declares one (every model-package set does); otherwise the set's
    ``masking_type``: ``"sam"`` sets (built in-house, spectrograms already
    masked) apply them, ``"none"`` sets (all-zero placeholder masks) do not.
    Applying all-zero masks would zero every spectrogram, so a split whose first
    1,024 masks are all zero while masks are applied is refused.

    Parameters
    ----------
    npz_path (pathlib.Path)
        ``train_data.npz`` / ``val_data.npz`` / ``full_data.npz``.
    metadata_masking_type (str)
        The ``masking_type`` recorded in the set's ``metadata.npz``.

    Returns
    -------
    split (dict)
        ``inputs`` ``(N, F, T)`` float32 decoder inputs, ``masks_len`` ``(N,)`` and
        ``durations`` ``(N,)`` int64, ``apply_mask`` (bool), ``spec_id`` (``(N,)``
        str, or None when the split has none) and ``condition_columns`` (the raw
        conditioning columns of ``SUMMARY_CONDITION_COLUMNS`` the split holds,
        name -> ``(N,)`` float64; empty for a set built without them).
    """
    with np.load(npz_path, allow_pickle=False) as split:
        apply_mask = bool(split["apply_mask"]) if "apply_mask" in split.files else metadata_masking_type == "sam"
        spectrograms = np.asarray(split["spectrograms"], dtype=np.float32)
        masks = split["masks"] if apply_mask else None
        masks_len = np.asarray(split["masks_len"], dtype=np.int64)
        durations = np.asarray(split["durations"], dtype=np.int64)
        spec_id = split["spec_id"].astype(str) if "spec_id" in split.files else None
        condition_columns = {
            column: np.asarray(split[column], dtype=np.float64) for column in SUMMARY_CONDITION_COLUMNS if column in split.files
        }
    if masks is not None and masks.shape[0] and not np.any(masks[:1024]):
        error_message = (
            f"{npz_path} applies its masks (apply_mask), but its first 1,024 masks are all zero, so masking would "
            f"zero every spectrogram. A masking_type 'none' set must declare apply_mask False."
        )
        raise ValueError(error_message)
    return {
        "inputs": prepare_split(spectrograms, masks),
        "masks_len": masks_len,
        "durations": durations,
        "apply_mask": apply_mask,
        "spec_id": spec_id,
        "condition_columns": condition_columns,
    }


def validation_subset_indices(masks_len: np.ndarray, samples_per_mask_count: int, seed: int) -> np.ndarray:
    """
    Description
    -----------
    The fixed validation subset the validation loss is computed on, drawn as the
    reference trainer draws its "val diagnostic" set: for each ``masks_len`` value
    in increasing order, every row when the value has at most
    ``samples_per_mask_count`` of them, otherwise that many rows drawn without
    replacement from numpy's legacy generator seeded with ``seed`` (the one
    ``np.random.seed`` sets); the union is returned sorted. With the reference
    trainer's seed this reproduces its subset row for row.

    Parameters
    ----------
    masks_len (np.ndarray)
        ``(N,)`` mask counts of the validation split.
    samples_per_mask_count (int)
        Cap on rows per ``masks_len`` value.
    seed (int)
        Seed of the draw.

    Returns
    -------
    indices (np.ndarray)
        Sorted ``(M,)`` int64 row indices into the validation split.
    """
    generator = np.random.RandomState(seed)
    chosen: list[int] = []
    for value in np.unique(masks_len):
        rows = np.where(masks_len == value)[0]
        picked = rows if len(rows) <= samples_per_mask_count else generator.choice(rows, samples_per_mask_count, replace=False)
        chosen.extend(picked.tolist())
    return np.array(sorted(chosen), dtype=np.int64)


def condition_bin_edges(values: np.ndarray, n_bins: int, scheme: str) -> np.ndarray:
    """
    Description
    -----------
    The bin edges a conditional run groups its training rows by (qmc_deep_gen
    ``data/conditionals.py:bin_edges``). ``"quantile"``: the ``n_bins + 1``
    quantiles of ``values`` (``np.quantile``'s linear rule), duplicates collapsed
    (a spike in the distribution leaves fewer bins). ``"quantile_capped"``: the same
    edges, then every bin wider than the widest INNER bin (all but the first and
    the last) is split into ``ceil(width / cap - 1e-9)`` equal-width pieces, so no
    piece is wider than the widest inner bin; in practice only the two end bins,
    which the tails stretch to 20-40 % of the range, are split. The pieces are
    kept however few rows they hold (the tails are where the value needs the
    resolution; the loss scaling compensates for their small batches). With fewer
    than four edges the quantile edges are returned unchanged.

    Parameters
    ----------
    values (np.ndarray)
        ``(N,)`` conditioning values of the training split.
    n_bins (int)
        Number of quantile bins.
    scheme (str)
        ``"quantile"`` or ``"quantile_capped"``.

    Returns
    -------
    edges (np.ndarray)
        Increasing float64 bin edges, the first the smallest value and the last the
        largest.
    """
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1)))
    if scheme == "quantile" or edges.size < 4:
        return edges
    cap = float(np.diff(edges)[1:-1].max())
    capped = [edges[0]]
    for low, high in itertools.pairwise(edges):
        pieces = int(np.ceil((high - low) / cap - 1e-9))
        capped.extend(np.linspace(low, high, pieces + 1)[1:].tolist())
    return np.asarray(capped)


def condition_group_ids(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """
    Description
    -----------
    The bin each conditioning value falls in (qmc_deep_gen
    ``data/conditionals.py:ids_from_edges``): ``digitize(value, edges[1:-1])``, so a
    bin holds ``[edge_i, edge_i+1)`` and values beyond the edges land in the end
    bins. Fewer than two edges put every value in bin 0.

    Parameters
    ----------
    values (np.ndarray)
        ``(N,)`` conditioning values.
    edges (np.ndarray)
        Bin edges (:func:`condition_bin_edges`).

    Returns
    -------
    group_ids (np.ndarray)
        ``(N,)`` int64 bin index per value.
    """
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if np.asarray(edges).size < 2:
        return np.zeros(values.size, dtype=np.int64)
    return np.digitize(values, np.asarray(edges)[1:-1], right=False).astype(np.int64)


def condition_decode_grid(c_min: float, c_max: float, step: float) -> np.ndarray:
    """
    Description
    -----------
    The fixed conditioning values ``infer-qlvm-latents`` decodes calls at under a
    ``"grid"`` decode rule (qmc_deep_gen ``data/conditionals.py:decode_grid``):
    ``c_min + step * k`` for ``k = 0 .. ceil((c_max - c_min) / step - 1e-9)``, so the
    grid starts at the training minimum and its last point is at or just past the
    training maximum. A call inside the training range is then decoded at most
    ``step / 2`` from its own value.

    Parameters
    ----------
    c_min (float)
        Smallest training value.
    c_max (float)
        Largest training value.
    step (float)
        Grid spacing (> 0).

    Returns
    -------
    grid (np.ndarray)
        Increasing float64 grid.
    """
    n_points = int(np.ceil((c_max - c_min) / step - 1e-9)) + 1
    return c_min + step * np.arange(n_points)


def grouped_batches(group_ids: np.ndarray, batch_size: int, generator: np.random.Generator) -> list[np.ndarray]:
    """
    Description
    -----------
    One epoch of conditioning-homogeneous batches (one epoch of qmc_deep_gen's
    ``ConditionGroupedBatchSampler`` with ``shuffle=True``, ``drop_last=False``):
    for every group in increasing id order, its rows are permuted with
    ``generator`` and chunked into batches of at most ``batch_size`` (a group's
    last batch may be short), and the batches of all groups are then permuted
    together with the same generator. Every row appears exactly once. Given the
    generator of the reference epoch (``default_rng(seed + 2 * epoch + 1)`` for the
    shipped multi-worker runs, module docstring) this is its batch order, draw for
    draw.

    Parameters
    ----------
    group_ids (np.ndarray)
        ``(N,)`` group id of every training row (:func:`condition_group_ids`).
    batch_size (int)
        Largest batch.
    generator (np.random.Generator)
        The epoch's generator.

    Returns
    -------
    batches (list[np.ndarray])
        Row-index arrays, in the order they are trained on.
    """
    batches: list[np.ndarray] = []
    for group in np.unique(group_ids):
        rows = generator.permutation(np.flatnonzero(group_ids == group))
        batches.extend(rows[start:start + batch_size] for start in range(0, rows.size, batch_size))
    return [batches[position] for position in generator.permutation(len(batches))]


def condition_contract_block(
    conditional: str,
    duration_range: tuple[int, int],
    bin_scheme: str,
    n_bins: int,
    scale_loss_by_batch: bool,
    batch_size: int,
    decode_grid_step: float,
) -> dict:
    """
    Description
    -----------
    The ``condition`` block of a conditional cell's training contract, in the form
    of the shipped phase 11 contracts: ``name``, the constants
    ``infer-qlvm-latents`` computes a call's value with
    (:func:`qlvm_latents.compute_condition_values`: ``duration_min`` /
    ``duration_max`` / ``epsilon`` for duration, ``spectrogram`` / ``epsilon`` for
    mean frequency, ``db_range`` for loudness), the value's ``source`` and ``c``
    formula, its ``decode`` rule (``CONDITION_DECODE``) and ``decode_rule`` text,
    the ``training_bins`` recipe and ``condition_bins_format``.

    Parameters
    ----------
    conditional (str)
        ``"duration"``, ``"mean_freq"``, ``"bandwidth"`` or ``"loudness"``.
    duration_range (tuple[int, int])
        The training split's shortest and longest duration in time bins (the
        duration min-max; recorded for duration only).
    bin_scheme (str)
        ``"quantile"`` or ``"quantile_capped"``.
    n_bins (int)
        Quantile bins.
    scale_loss_by_batch (bool)
        Whether each step's loss was scaled by ``rows / batch_size``.
    batch_size (int)
        Training batch size.
    decode_grid_step (float)
        Spacing of ``condition_bins.npz``'s ``decode_grid``.

    Returns
    -------
    condition (dict)
        The JSON-serializable block.
    """
    capped = (
        ", every bin wider than the widest inner bin split into equal-width pieces"
        if bin_scheme == "quantile_capped" else ""
    )
    scaling = f"; loss scaled by rows / {batch_size}" if scale_loss_by_batch else "; batch-mean loss"
    decode = CONDITION_DECODE[conditional]
    decode_rule = (
        "the call's own c, clamped to [train_c_min, train_c_max] of condition_bins.npz" if decode == "exact" else
        f"nearest point of condition_bins.npz decode_grid ({decode_grid_step:g} apart over the training c range); "
        f"c beyond the grid decodes at its end point"
    )
    definitions = {
        "duration": {
            "duration_min": int(duration_range[0]),
            "duration_max": int(duration_range[1]),
            "epsilon": CONDITION_EPSILON,
            "source": "pre-resize duration in time bins",
        },
        "mean_freq": {
            "spectrogram": "masked",
            "epsilon": CONDITION_EPSILON,
            "source": "usv_summary mean_freq_hz; equal to the masked-spectrogram centroid to 1.5e-7",
            "c": (
                f"(f_hz - {MEAN_FREQ_LOW_HZ:g}) * {FREQUENCY_ROWS - 1} / ({FREQUENCY_ROWS} * {FREQUENCY_SPAN_HZ:g})"
            ),
        },
        "bandwidth": {
            "source": "usv_summary freq_bandwidth_hz",
            "c": f"clip(bw_hz / {FREQUENCY_SPAN_HZ:g}, 0, 1)",
        },
        "loudness": {
            "source": (
                "masked image-level loudness in dB from the session audio "
                "(usv_summary loudness_db, compute_usv_loudness.session_image_level_db)"
            ),
            "db_range": list(LOUDNESS_DB_RANGE),
            "c": f"clip((dB - {LOUDNESS_DB_RANGE[0]}) / ({LOUDNESS_DB_RANGE[1]} - {LOUDNESS_DB_RANGE[0]}), 0, 1)",
        },
    }
    return {
        "name": conditional,
        **definitions[conditional],
        "decode": decode,
        "decode_rule": decode_rule,
        "training_bins": f"{bin_scheme}: {n_bins} quantile bins{capped}{scaling}",
        "condition_bins_format": "phase11 (edges, group_means, decode_grid); no bin_mean",
    }


def raw_condition_values(split: dict, column: str, table: dict | None, split_name: str) -> np.ndarray:
    """
    Description
    -----------
    Each row's raw conditioning value (Hz or dB) in one split: from ``table`` (a
    per-call table, ``spec_id`` -> value, :func:`load_condition_table`) when one is
    given, else from the split's own ``column``. Every row must have a finite value
    (a row without one would otherwise train at an arbitrary conditioning value),
    so a missing ``spec_id``, a missing column or a non-finite value raises.

    Parameters
    ----------
    split (dict)
        A split as :func:`load_split` returns it.
    column (str)
        The raw column (``CONDITION_RAW_COLUMNS``).
    table (dict | None)
        ``spec_id`` -> raw value, or None to read the split's column.
    split_name (str)
        The split's file name, for the messages.

    Returns
    -------
    raw (np.ndarray)
        ``(N,)`` float64 raw values.
    """
    if table is not None:
        if split['spec_id'] is None:
            error_message = f"{split_name} has no spec_id, so its rows cannot be looked up in the condition_table."
            raise ValueError(error_message)
        missing = [spec_id for spec_id in split['spec_id'] if spec_id not in table]
        if missing:
            error_message = (
                f"{len(missing)} of {split['spec_id'].size} rows of {split_name} are not in the condition_table "
                f"(first {missing[0]!r})."
            )
            raise ValueError(error_message)
        raw = np.array([table[spec_id] for spec_id in split['spec_id']], dtype=np.float64)
    elif column in split['condition_columns']:
        raw = split['condition_columns'][column]
    else:
        error_message = (
            f"{split_name} has no {column} column (sets built by build-qlvm-training-set before it copied the "
            f"USV summary values, or built outside the repository); rebuild the set or pass a condition_table."
        )
        raise ValueError(error_message)
    not_finite = ~np.isfinite(raw)
    if np.any(not_finite):
        first = split['spec_id'][int(np.flatnonzero(not_finite)[0])] if split['spec_id'] is not None else int(np.flatnonzero(not_finite)[0])
        error_message = (
            f"{int(not_finite.sum())} rows of {split_name} have no finite {column} (first {first!r}); "
            f"run generate-usv-acoustic-features on their sessions and rebuild the set."
        )
        raise ValueError(error_message)
    return raw


def load_condition_table(table_path: pathlib.Path, column: str) -> dict:
    """
    Description
    -----------
    Reads one raw column of a per-call conditioning table: an ``.npz`` with a
    ``spec_id`` array and the column (``mean_freq_hz``, ``freq_bandwidth_hz`` or
    ``loudness_db``, the last also accepted under the v3 package's name
    ``image_level_db``), e.g. the package's ``corpus/cond_table.npz``.

    Parameters
    ----------
    table_path (pathlib.Path)
        The table.
    column (str)
        The raw column (``CONDITION_RAW_COLUMNS``).

    Returns
    -------
    table (dict)
        ``spec_id`` (str) -> raw value (float).
    """
    with np.load(table_path, allow_pickle=False) as table_file:
        if column in table_file.files:
            name = column
        elif column in _CONDITION_TABLE_ALIASES and _CONDITION_TABLE_ALIASES[column] in table_file.files:
            name = _CONDITION_TABLE_ALIASES[column]
        else:
            error_message = f"{table_path} has no {column} column (columns: {sorted(table_file.files)})."
            raise ValueError(error_message)
        spec_ids = table_file["spec_id"].astype(str).tolist()
        values = np.asarray(table_file[name], dtype=np.float64).tolist()
    return dict(zip(spec_ids, values, strict=True))


def condition_values(conditional: str, raw: np.ndarray | None, durations: np.ndarray, condition: dict) -> np.ndarray:
    """
    Description
    -----------
    Each row's conditioning value by the frozen map of its condition (module
    docstring): duration, bandwidth and loudness through
    :func:`qlvm_latents.compute_condition_values` (the map ``infer-qlvm-latents``
    applies), mean frequency by the closed form of the summary's Hz,
    ``(f - 30 kHz) * 127 / (128 * 90 kHz)`` (inference computes the same value as
    the resized SAM-masked spectrogram's row centroid).

    Parameters
    ----------
    conditional (str)
        The condition.
    raw (np.ndarray | None)
        ``(N,)`` raw values (Hz, dB); None for duration.
    durations (np.ndarray)
        ``(N,)`` durations in time bins.
    condition (dict)
        The contract's ``condition`` block (:func:`condition_contract_block`).

    Returns
    -------
    values (np.ndarray)
        ``(N,)`` float32 conditioning values.
    """
    if conditional == "mean_freq":
        return ((raw - MEAN_FREQ_LOW_HZ) * (FREQUENCY_ROWS - 1) / (FREQUENCY_ROWS * FREQUENCY_SPAN_HZ)).astype(np.float32)
    return compute_condition_values(condition, durations, None, raw)


def condition_value_stats(conditional: str, values: np.ndarray) -> dict:
    """
    Description
    -----------
    The run record's summary of one split's conditioning values (the reference
    run record's ``cond_table_stats``): the row count, the range and, for the
    clipped maps (bandwidth, loudness), how many rows the clip pinned at exactly 0
    or 1 (raw values beyond the frozen range).

    Parameters
    ----------
    conditional (str)
        The condition.
    values (np.ndarray)
        ``(N,)`` conditioning values.

    Returns
    -------
    stats (dict)
        ``n``, ``c_min``, ``c_max`` and ``n_clamped``.
    """
    clipped = conditional in ("bandwidth", "loudness")
    return {
        "n": int(values.size),
        "c_min": float(values.min()),
        "c_max": float(values.max()),
        "n_clamped": int(np.count_nonzero((values <= 0.0) | (values >= 1.0))) if clipped else 0,
    }


def make_step_functions(
    optimizer: optax.GradientTransformation,
    training_lattice: jnp.ndarray,
    validation_lattice: jnp.ndarray,
) -> tuple[Callable, Callable]:
    """
    Description
    -----------
    Builds the two compiled functions of a run: one optimizer step on a batch
    against a shifted training lattice, and the validation loss of one
    spectrogram against a shifted validation lattice.

    Parameters
    ----------
    optimizer (optax.GradientTransformation)
        The optimizer (Adam at a constant learning rate).
    training_lattice (jnp.ndarray)
        ``(K_train, 2)`` training lattice.
    validation_lattice (jnp.ndarray)
        ``(K_val, 2)`` validation lattice.

    Returns
    -------
    train_step (Callable)
        ``(params, opt_state, batch, row_weights, condition, shift, step_scale) ->
        (params, opt_state, loss)``, with ``batch`` ``(B, 1, F, T)``,
        ``row_weights`` ``(B,)`` or None, ``condition`` ``(1, c_dim)`` or None,
        ``shift`` ``(1, 2)`` and ``step_scale`` a scalar.
    validation_loss (Callable)
        ``(params, spectrogram, shift, condition) -> loss`` for one ``(F, T)``
        spectrogram (``condition`` ``(1, c_dim)`` or None).
    """

    def batch_loss(
        params: dict,
        batch: jnp.ndarray,
        row_weights: jnp.ndarray | None,
        condition: jnp.ndarray | None,
        shift: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Description
        -----------
        The objective of one batch: the negative (row-weighted) mean QMC log
        evidence of the batch under the training lattice shifted by ``shift`` and
        decoded at ``condition``.

        Parameters
        ----------
        params (dict)
            Decoder weights.
        batch (jnp.ndarray)
            ``(B, 1, F, T)`` decoder inputs.
        row_weights (jnp.ndarray | None)
            ``(B,)`` 0/1 weights (0 for padding rows), or None for all rows.
        condition (jnp.ndarray | None)
            ``(1, c_dim)`` conditioning vector, or None for an unconditional decoder.
        shift (jnp.ndarray)
            ``(1, 2)`` torus shift.

        Returns
        -------
        loss (jnp.ndarray)
            Scalar loss.
        """
        return binary_evidence(decode_shifted_lattice(training_lattice, shift, params, condition), batch, row_weights)

    @jax.jit
    def train_step(
        params: dict,
        opt_state: optax.OptState,
        batch: jnp.ndarray,
        row_weights: jnp.ndarray | None,
        condition: jnp.ndarray | None,
        shift: jnp.ndarray,
        step_scale: jnp.ndarray,
    ) -> tuple:
        """
        Description
        -----------
        One Adam step on the gradient of ``step_scale`` times :func:`batch_loss`
        (``step_scale`` is ``rows / batch_size`` when the loss is scaled by the
        batch's size, else 1; qmc_deep_gen ``train/train.py:train_epoch``'s
        ``scale_to_batch``).

        Parameters
        ----------
        params (dict)
            Decoder weights.
        opt_state (optax.OptState)
            Adam state.
        batch (jnp.ndarray)
            ``(B, 1, F, T)`` decoder inputs.
        row_weights (jnp.ndarray | None)
            ``(B,)`` 0/1 weights, or None.
        condition (jnp.ndarray | None)
            ``(1, c_dim)`` conditioning vector, or None.
        shift (jnp.ndarray)
            ``(1, 2)`` torus shift.
        step_scale (jnp.ndarray)
            Scalar multiplying the loss the gradient is taken of.

        Returns
        -------
        params (dict)
            Updated decoder weights.
        opt_state (optax.OptState)
            Updated Adam state.
        loss (jnp.ndarray)
            The UNSCALED batch loss before the update (the value recorded).
        """

        def scaled_loss(weights: dict) -> tuple[jnp.ndarray, jnp.ndarray]:
            """
            Description
            -----------
            The loss the gradient is taken of, with the unscaled loss as auxiliary
            output.

            Parameters
            ----------
            weights (dict)
                Decoder weights.

            Returns
            -------
            scaled (jnp.ndarray)
                ``step_scale`` times the batch loss.
            loss (jnp.ndarray)
                The batch loss.
            """
            loss = batch_loss(weights, batch, row_weights, condition, shift)
            return loss * step_scale, loss

        (_, loss), grads = jax.value_and_grad(scaled_loss, has_aux=True)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    @jax.jit
    def validation_loss(
        params: dict,
        spectrogram: jnp.ndarray,
        shift: jnp.ndarray,
        condition: jnp.ndarray | None,
    ) -> jnp.ndarray:
        """
        Description
        -----------
        The objective of one validation spectrogram against the validation
        lattice shifted by ``shift`` and decoded at the spectrogram's own
        ``condition`` (the reference trainer validates with batches of one, each
        with its own shift, and a one-row batch's mean value is its own).

        Parameters
        ----------
        params (dict)
            Decoder weights.
        spectrogram (jnp.ndarray)
            ``(F, T)`` decoder input.
        shift (jnp.ndarray)
            ``(1, 2)`` torus shift.
        condition (jnp.ndarray | None)
            ``(1, c_dim)`` conditioning vector, or None.

        Returns
        -------
        loss (jnp.ndarray)
            Scalar loss.
        """
        return binary_evidence(decode_shifted_lattice(validation_lattice, shift, params, condition), spectrogram[None, None])

    return train_step, validation_loss


def torch_checkpoint(
    params: dict[str, jnp.ndarray],
    opt_state: optax.OptState,
    head: str,
    learning_rate: float,
    batch_losses: list[float],
    c_dim: int,
) -> dict:
    """
    Description
    -----------
    Assembles the checkpoint in the structure the reference trainer's
    ``train.model_saving_loading.save`` writes, and every QLVM model package
    cell's ``checkpoint.tar`` holds: ``"model"``, the ``QMCLVM`` ``state_dict``
    (the decoder's tensors under ``decoder.<layer_idx>.weight`` / ``.bias``, in
    parameter order); ``"optimizer"``, a torch ``Adam`` ``state_dict`` (per
    parameter ``step``, ``exp_avg`` and ``exp_avg_sq`` from the optax Adam state,
    and one parameter group with torch's defaults and ``learning_rate``); and
    ``"run info"``, the training loss of every batch.

    Parameters
    ----------
    params (dict[str, jnp.ndarray])
        Trained decoder weights.
    opt_state (optax.OptState)
        The optax Adam state after the last step.
    head (str)
        Decoder head, ``"relu"`` or ``"legacy"``.
    learning_rate (float)
        The learning rate.
    batch_losses (list[float])
        Training loss of every batch, in order.
    c_dim (int)
        Conditioning width of the decoder (0 unconditional, 1 conditional), which
        sets the first Linear layer's input width.

    Returns
    -------
    checkpoint (dict)
        ``{"model": OrderedDict, "optimizer": dict, "run info": list}`` of torch
        tensors and Python values, ready for ``torch.save``.
    """
    names = list(decoder_parameter_shapes(latent_dim=LATENT_DIM, c_dim=c_dim, head=head))
    adam_state = next(state for state in opt_state if isinstance(state, optax.ScaleByAdamState))
    step = torch.tensor(float(adam_state.count), dtype=torch.float32)
    model_state = collections.OrderedDict(
        (f"decoder.{name}", torch.from_numpy(np.array(params[name], dtype=np.float32))) for name in names
    )
    optimizer_state = {
        "state": {
            index: {
                "step": step.clone(),
                "exp_avg": torch.from_numpy(np.array(adam_state.mu[name], dtype=np.float32)),
                "exp_avg_sq": torch.from_numpy(np.array(adam_state.nu[name], dtype=np.float32)),
            }
            for index, name in enumerate(names)
        },
        "param_groups": [
            {
                "lr": learning_rate,
                "betas": _ADAM_BETAS,
                "eps": _ADAM_EPSILON,
                "weight_decay": 0,
                "amsgrad": False,
                "maximize": False,
                "foreach": None,
                "capturable": False,
                "differentiable": False,
                "fused": None,
                "params": list(range(len(names))),
            }
        ],
    }
    return {"model": model_state, "optimizer": optimizer_state, "run info": [float(loss) for loss in batch_losses]}


def write_json(path: pathlib.Path, payload: dict) -> None:
    """
    Description
    -----------
    Writes a JSON file atomically (:func:`os_utils.atomic_output_path`), indented.

    Parameters
    ----------
    path (pathlib.Path)
        Destination; its parent must exist.
    payload (dict)
        JSON-serializable content.

    Returns
    -------
    None
    """
    with atomic_output_path(path) as tmp_path, tmp_path.open("w") as json_file:
        json.dump(payload, json_file, indent=2)


class QLVMTrainer:
    """
    Description
    -----------
    Trains a QLVM decoder on a prebuilt ``.npz`` training set with the shipped
    recipe (see the module docstring) and writes the run as a model package cell:
    ``checkpoint.tar``, ``config/training_contract.json``,
    ``config/run_config.json`` and ``metrics/val_diagnostics.npz``.
    """

    def __init__(
        self,
        dataset_directory: str | None = None,
        output_directory: str | None = None,
        input_parameter_dict: dict | None = None,
        message_output: Callable | None = None,
    ) -> None:
        """
        Description
        -----------
        Initializes the QLVMTrainer.

        Parameters
        ----------
        dataset_directory (str)
            Directory holding the training set: ``train_data.npz`` (or
            ``full_data.npz``), optionally ``val_data.npz``, and ``metadata.npz``.
        output_directory (str)
            Directory the model package cell is written to (created if missing).
        input_parameter_dict (dict)
            Processing settings; the ``train_qlvm`` block supplies the recipe.
        message_output (Callable)
            Logging callback; defaults to ``print``.

        Returns
        -------
        None
        """

        self.dataset_directory = dataset_directory
        self.output_directory = output_directory
        self.input_parameter_dict = input_parameter_dict if input_parameter_dict is not None else {}
        self.message_output = message_output if message_output is not None else print
        self.app_context_bool = is_gui_context()

    def _validate_settings(self, cfg: dict) -> None:
        """
        Description
        -----------
        Checks the ``train_qlvm`` block before any data is read, raising one
        ``ValueError`` that lists every invalid value.

        Parameters
        ----------
        cfg (dict)
            The ``train_qlvm`` settings block.

        Returns
        -------
        None
        """
        problems = [
            f"{key} must be >= 1, got {cfg[key]!r}"
            for key in ("n_epochs", "batch_size", "val_freq", "val_samples_per_mask_count")
            if cfg[key] < 1
        ]
        problems += [
            f"{key} must be >= 3 (a Fibonacci lattice of fib(m) points), got {cfg[key]!r}"
            for key in ("training_fib_m", "validation_fib_m", "embedding_fib_m")
            if cfg[key] < 3
        ]
        if cfg["decoder_head"] not in DECODER_HEADS:
            problems.append(f"decoder_head must be one of {DECODER_HEADS}, got {cfg['decoder_head']!r}")
        if not cfg["learning_rate"] > 0:
            problems.append(f"learning_rate must be > 0, got {cfg['learning_rate']!r}")
        if cfg["conditional"] is not None and cfg["conditional"] not in CONDITION_NAMES:
            problems.append(f"conditional must be null or one of {CONDITION_NAMES}, got {cfg['conditional']!r}")
        if cfg["condition_n_bins"] < 1:
            problems.append(f"condition_n_bins must be >= 1, got {cfg['condition_n_bins']!r}")
        if cfg["condition_bin_scheme"] not in CONDITION_BIN_SCHEMES:
            problems.append(f"condition_bin_scheme must be one of {CONDITION_BIN_SCHEMES}, got {cfg['condition_bin_scheme']!r}")
        if not cfg["condition_decode_grid_step"] > 0:
            problems.append(f"condition_decode_grid_step must be > 0, got {cfg['condition_decode_grid_step']!r}")
        if cfg["condition_table"] is not None and not pathlib.Path(cfg["condition_table"]).is_file():
            problems.append(f"condition_table must be null or an existing .npz, got {cfg['condition_table']!r}")
        if problems:
            error_message = "train_qlvm settings are invalid:\n  " + "\n  ".join(problems)
            raise ValueError(error_message)

    def _locate_splits(self) -> tuple[pathlib.Path, pathlib.Path | None, dict]:
        """
        Description
        -----------
        Finds the training split (``train_data.npz``, else ``full_data.npz``), the
        validation split (``val_data.npz``, when present) and reads the set's
        ``metadata.npz``. Refuses a set without ``metadata.npz`` (the training
        contract is built from it) and a ``train_data.npz`` / ``val_data.npz`` older
        than a ``full_data.npz`` in the same directory (a split left over from an
        earlier build, which would otherwise win the lookup).

        Parameters
        ----------

        Returns
        -------
        train_path (pathlib.Path)
            The training split.
        val_path (pathlib.Path | None)
            The validation split, or ``None``.
        metadata (dict)
            The fields of ``metadata.npz`` the run records: ``masking_type``,
            ``target_shape``, ``time_stretch``, ``length_threshold`` (``None`` when
            the set records NaN, i.e. no duration bound), ``require_mask``
            (``False`` when not recorded) and ``floor`` (``None`` when not recorded).
        """
        dataset_dir = pathlib.Path(self.dataset_directory)
        train_npz = dataset_dir / "train_data.npz"
        full_npz = dataset_dir / "full_data.npz"
        val_npz = dataset_dir / "val_data.npz"
        if train_npz.is_file():
            train_path = train_npz
        elif full_npz.is_file():
            train_path = full_npz
        else:
            error_message = f"No training set found in {dataset_dir} (expected train_data.npz or full_data.npz)."
            raise FileNotFoundError(error_message)

        if full_npz.is_file():
            stale_splits = [
                split.name for split in (train_npz, val_npz)
                if split.is_file() and split.stat().st_mtime < full_npz.stat().st_mtime
            ]
            if stale_splits:
                error_message = (
                    f"{', '.join(stale_splits)} in {dataset_dir} predate full_data.npz, so they are left over "
                    f"from an earlier build. Delete them to train on full_data.npz, or rebuild the split set."
                )
                raise ValueError(error_message)

        metadata_path = dataset_dir / "metadata.npz"
        if not metadata_path.is_file():
            error_message = (
                f"No metadata.npz in {dataset_dir}. The training contract records the set's masking, shape, "
                f"duration window and loudness floor from it."
            )
            raise FileNotFoundError(error_message)
        with np.load(metadata_path, allow_pickle=False) as metadata_file:
            length_threshold = float(metadata_file["length_threshold"])
            metadata = {
                "masking_type": str(metadata_file["masking_type"]),
                "target_shape": [int(value) for value in metadata_file["target_shape"]],
                "time_stretch": bool(metadata_file["time_stretch"]),
                "length_threshold": None if math.isnan(length_threshold) else length_threshold,
                "require_mask": bool(metadata_file["require_mask"]) if "require_mask" in metadata_file.files else False,
                "floor": float(metadata_file["floor"]) if "floor" in metadata_file.files else None,
            }
        return train_path, (val_npz if val_npz.is_file() else None), metadata

    def train(self) -> None:
        """
        Description
        -----------
        Runs the recipe (module docstring): reads and prepares the splits, draws
        the validation subset, initializes the decoder and trains it for
        ``n_epochs`` epochs of Adam steps on shuffled batches (one random lattice
        shift per batch), evaluating the validation loss every ``val_freq`` epochs
        and after the last one, then writes the last-epoch weights and the run's
        records into the output directory. With ``conditional`` set it first
        computes every row's conditioning value, cuts the training values into
        bins and writes ``config/condition_bins.npz``, then trains on batches drawn
        one bin at a time, each decoded at its rows' mean value, with the loss
        optionally scaled by the batch's share of ``batch_size``, and validates each
        call at its own value (module docstring).

        Parameters
        ----------

        Returns
        -------
        ``checkpoint.tar`` + ``config/training_contract.json`` + ``config/run_config.json`` + ``metrics/val_diagnostics.npz`` (+ ``config/condition_bins.npz``)
            The model package cell, written to the output directory.
        """

        started = datetime.now()
        self.message_output(f"QLVM training started at: {started.hour:02d}:{started.minute:02d}:{started.second:02d}.")
        smart_wait(app_context_bool=self.app_context_bool, seconds=1)

        cfg = self.input_parameter_dict['train_qlvm']
        self._validate_settings(cfg)
        n_epochs = cfg['n_epochs']
        head = cfg['decoder_head']
        batch_size = cfg['batch_size']
        learning_rate = cfg['learning_rate']
        val_freq = cfg['val_freq']
        seed = cfg['seed']
        conditional = cfg['conditional']
        c_dim = 0 if conditional is None else 1

        train_path, val_path, metadata = self._locate_splits()
        output_dir = pathlib.Path(self.output_directory)

        train_split = load_split(train_path, metadata['masking_type'])
        train_inputs = train_split['inputs']
        n_train = int(train_inputs.shape[0])
        if n_train < 1:
            error_message = f"{train_path} holds no spectrograms; there is nothing to train on."
            raise ValueError(error_message)
        self.message_output(
            f"Loaded {n_train} training spectrograms from {train_path.name} (masks applied: {train_split['apply_mask']})."
        )

        # Conditional runs: every row's conditioning value (the training split's
        # duration range fixes the duration min-max for both splits and for inference).
        condition = None
        condition_table = None
        train_c = None
        if conditional is not None:
            condition = condition_contract_block(
                conditional,
                (int(train_split['durations'].min()), int(train_split['durations'].max())),
                cfg['condition_bin_scheme'],
                cfg['condition_n_bins'],
                cfg['condition_scale_loss_by_batch'],
                batch_size,
                cfg['condition_decode_grid_step'],
            )
            if conditional != "duration" and cfg['condition_table'] is not None:
                condition_table = load_condition_table(pathlib.Path(cfg['condition_table']), CONDITION_RAW_COLUMNS[conditional])
            train_raw = None if conditional == "duration" else raw_condition_values(
                train_split, CONDITION_RAW_COLUMNS[conditional], condition_table, train_path.name,
            )
            train_c = condition_values(conditional, train_raw, train_split['durations'], condition)

        val_inputs = None
        val_rows = np.zeros(0, dtype=np.int64)
        val_masks_len = np.zeros(0, dtype=np.int64)
        val_durations = np.zeros(0, dtype=np.int64)
        val_c = None
        condition_stats = {}
        if val_path is not None:
            val_split = load_split(val_path, metadata['masking_type'])
            if val_split['apply_mask'] != train_split['apply_mask']:
                error_message = (
                    f"{train_path.name} and {val_path.name} disagree on apply_mask "
                    f"({train_split['apply_mask']} vs {val_split['apply_mask']}); they come from different builds."
                )
                raise ValueError(error_message)
            val_rows = validation_subset_indices(val_split['masks_len'], cfg['val_samples_per_mask_count'], seed)
            val_inputs = val_split['inputs'][val_rows]
            val_masks_len = val_split['masks_len'][val_rows]
            val_durations = val_split['durations'][val_rows]
            if conditional is not None:
                val_raw = None if conditional == "duration" else raw_condition_values(
                    val_split, CONDITION_RAW_COLUMNS[conditional], condition_table, val_path.name,
                )
                val_c_all = condition_values(conditional, val_raw, val_split['durations'], condition)
                condition_stats["val"] = condition_value_stats(conditional, val_c_all)
                val_c = val_c_all[val_rows]
            self.message_output(
                f"Validation subset: {val_rows.shape[0]} of {val_split['inputs'].shape[0]} spectrograms in "
                f"{val_path.name} (at most {cfg['val_samples_per_mask_count']} per masks_len value)."
            )
            del val_split

        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / CONFIG_DIRECTORY).mkdir(exist_ok=True)
        (output_dir / METRICS_DIRECTORY).mkdir(exist_ok=True)

        # Conditional runs: the bins, written to config/condition_bins.npz with the
        # training range and the decode grid, and the batches drawn from them.
        condition_record = {}
        train_groups = None
        if conditional is not None:
            condition_stats["train"] = condition_value_stats(conditional, train_c)
            edges = condition_bin_edges(train_c, cfg['condition_n_bins'], cfg['condition_bin_scheme'])
            train_groups = condition_group_ids(train_c, edges)
            group_labels = np.unique(train_groups)
            group_sizes = np.array([int(np.count_nonzero(train_groups == group)) for group in group_labels], dtype=np.int64)
            group_means = np.array([train_c[train_groups == group].mean() for group in group_labels])
            train_c_min, train_c_max = train_c.min(), train_c.max()
            decode_grid = condition_decode_grid(float(train_c_min), float(train_c_max), cfg['condition_decode_grid_step'])
            condition_bins_path = output_dir / CONFIG_DIRECTORY / "condition_bins.npz"
            with atomic_output_path(condition_bins_path) as tmp_bins_path, tmp_bins_path.open("wb") as bins_file:
                np.savez(
                    bins_file,
                    conditional=np.array(conditional),
                    bin_scheme=np.array(cfg['condition_bin_scheme']),
                    n_bins=np.array(cfg['condition_n_bins']),
                    edges=edges,
                    group_ids=group_labels,
                    group_sizes=group_sizes,
                    group_means=group_means,
                    train_c_min=np.array(train_c_min),
                    train_c_max=np.array(train_c_max),
                    decode_grid=decode_grid,
                    decode_grid_step=np.array(float(cfg['condition_decode_grid_step'])),
                )
            n_batches = int(sum(math.ceil(size / batch_size) for size in group_sizes))
            condition_record = {
                "condition_source": (
                    "the split's durations" if conditional == "duration" else
                    f"condition_table {cfg['condition_table']}" if condition_table is not None else
                    f"the splits' {CONDITION_RAW_COLUMNS[conditional]} column"
                ),
                "condition_stats": condition_stats,
                "condition_bins": f"{CONFIG_DIRECTORY}/condition_bins.npz",
                "n_groups": int(group_labels.size),
                "group_size_min": int(group_sizes.min()),
                "widest_bin": float(np.diff(edges).max()) if edges.size > 1 else 0.0,
                "train_c_range": [float(train_c_min), float(train_c_max)],
                "decode_grid": {"step": float(cfg['condition_decode_grid_step']), "n": int(decode_grid.size)},
            }
            self.message_output(
                f"Conditioning on {conditional}: {group_labels.size} {cfg['condition_bin_scheme']} bins of "
                f"{group_sizes.min()}-{group_sizes.max()} rows, training values {float(train_c_min):.4f}-"
                f"{float(train_c_max):.4f}, loss {'scaled by rows / ' + str(batch_size) if cfg['condition_scale_loss_by_batch'] else 'not scaled'}."
            )
        else:
            n_batches = math.ceil(n_train / batch_size)

        training_lattice = jnp.asarray(gen_fib_basis(cfg['training_fib_m']), dtype=jnp.float32)
        validation_lattice = jnp.asarray(gen_fib_basis(cfg['validation_fib_m']), dtype=jnp.float32)
        optimizer = optax.adam(learning_rate, b1=_ADAM_BETAS[0], b2=_ADAM_BETAS[1], eps=_ADAM_EPSILON)
        train_step, validation_loss = make_step_functions(optimizer, training_lattice, validation_lattice)

        init_key, shift_key, validation_key = jax.random.split(jax.random.PRNGKey(seed), 3)
        params = init_decoder_params(init_key, latent_dim=LATENT_DIM, c_dim=c_dim, head=head)
        opt_state = optimizer.init(params)
        shuffle_generator = np.random.default_rng(seed)
        self.message_output(
            f"Training the {head!r}-head decoder on {jax.devices()[0].platform}: {n_epochs} epochs x {n_batches} "
            f"batches of {batch_size}, a {training_lattice.shape[0]}-point training lattice and a "
            f"{validation_lattice.shape[0]}-point validation lattice."
        )

        # A conditional run pads every short batch to batch_size with zero-weight rows,
        # so one compiled step serves every batch of the run.
        padded_batch = None if conditional is None else np.zeros((batch_size, 1, *train_inputs.shape[1:]), dtype=np.float32)
        batch_losses: list[float] = []
        epoch_train_losses: list[float] = []
        val_epochs: list[int] = []
        val_losses: list[float] = []
        epoch_seconds: list[float] = []
        for epoch in range(n_epochs):
            epoch_start = time.perf_counter()
            if conditional is None:
                order = shuffle_generator.permutation(n_train)
                epoch_batches = [order[n_batch * batch_size:(n_batch + 1) * batch_size] for n_batch in range(n_batches)]
            else:
                # The shipped runs' sampler advanced twice per epoch (module docstring).
                epoch_batches = grouped_batches(train_groups, batch_size, np.random.default_rng(seed + 2 * epoch + 1))
            shift_key, epoch_key = jax.random.split(shift_key)
            shifts = jax.random.uniform(epoch_key, (n_batches, 1, LATENT_DIM), dtype=jnp.float32)
            device_losses = []
            for n_batch, rows in enumerate(epoch_batches):
                if conditional is None:
                    batch = jnp.asarray(train_inputs[rows][:, None])
                    row_weights, batch_condition, step_scale = None, None, 1.0
                else:
                    padded_batch[:rows.size, 0] = train_inputs[rows]
                    padded_batch[rows.size:] = 0.0
                    batch = jnp.asarray(padded_batch)
                    row_weights = jnp.asarray((np.arange(batch_size) < rows.size).astype(np.float32))
                    batch_condition = jnp.full((1, c_dim), train_c[rows].mean(), dtype=jnp.float32)
                    step_scale = rows.size / batch_size if cfg['condition_scale_loss_by_batch'] else 1.0
                params, opt_state, loss = train_step(
                    params, opt_state, batch, row_weights, batch_condition, shifts[n_batch], jnp.float32(step_scale),
                )
                device_losses.append(loss)
            epoch_losses = [float(loss) for loss in device_losses]
            batch_losses += epoch_losses
            epoch_train_losses.append(float(np.mean(epoch_losses)))
            epoch_seconds.append(time.perf_counter() - epoch_start)
            if not np.isfinite(epoch_train_losses[-1]):
                error_message = f"The training loss of epoch {epoch + 1} is {epoch_train_losses[-1]}; training diverged."
                raise FloatingPointError(error_message)

            if (epoch + 1) % val_freq == 0 or epoch == n_epochs - 1:
                message = f"  Epoch {epoch + 1}/{n_epochs}: train evidence loss {epoch_train_losses[-1]:.4f}"
                if val_inputs is not None:
                    validation_start = time.perf_counter()
                    validation_key, epoch_validation_key = jax.random.split(validation_key)
                    val_shifts = jax.random.uniform(
                        epoch_validation_key, (val_inputs.shape[0], 1, LATENT_DIM), dtype=jnp.float32,
                    )
                    device_val_losses = [
                        validation_loss(
                            params,
                            jnp.asarray(val_inputs[n_row]),
                            val_shifts[n_row],
                            None if conditional is None else jnp.full((1, c_dim), val_c[n_row], dtype=jnp.float32),
                        )
                        for n_row in range(val_inputs.shape[0])
                    ]
                    val_epochs.append(epoch + 1)
                    val_losses.append(float(np.mean([float(loss) for loss in device_val_losses])))
                    message += (
                        f", val evidence loss {val_losses[-1]:.4f} "
                        f"(validation took {time.perf_counter() - validation_start:.1f} s)"
                    )
                self.message_output(f"{message}; {epoch_seconds[-1]:.2f} s per training epoch.")

        checkpoint_path = output_dir / CHECKPOINT_NAME
        with atomic_output_path(checkpoint_path) as tmp_checkpoint_path, tmp_checkpoint_path.open("wb") as checkpoint_file:
            torch.save(torch_checkpoint(params, opt_state, head, learning_rate, batch_losses, c_dim), checkpoint_file)

        diagnostics_path = output_dir / METRICS_DIRECTORY / "val_diagnostics.npz"
        condition_diagnostics = {} if conditional is None else {
            "conditional": np.array(conditional),
            "val_diag_c": np.zeros(0, dtype=np.float32) if val_c is None else val_c.astype(np.float32),
        }
        with atomic_output_path(diagnostics_path) as tmp_diagnostics_path, tmp_diagnostics_path.open("wb") as diagnostics_file:
            np.savez(
                diagnostics_file,
                train_loss_epochs=np.arange(1, n_epochs + 1),
                train_losses=np.array(epoch_train_losses),
                epoch_seconds=np.array(epoch_seconds),
                val_loss_epochs=np.array(val_epochs, dtype=np.int64),
                val_losses=np.array(val_losses),
                val_diag_indices=val_rows,
                val_diag_ml=val_masks_len.astype(np.float32),
                val_diag_dur=val_durations.astype(np.float32),
                decoder_head=np.array(head),
                **condition_diagnostics,
            )

        # The training contract, in the fields a v3 model package cell's contract has;
        # infer-qlvm-latents checks its settings against it (enforce_training_contract)
        # and, for a conditional cell, reads condition_bins.npz through it.
        contract = {
            "decoder_head": head,
            "latent_dim": LATENT_DIM,
            "c_dim": c_dim,
            "conditional": conditional,
            "input_normalization": "minmax",
            "normalization_epsilon": NORMALIZATION_EPSILON,
            "masking_type": "sam" if train_split['apply_mask'] else "none",
            "floor": metadata['floor'],
            "target_shape": metadata['target_shape'],
            "time_stretch": metadata['time_stretch'],
            "length_threshold": metadata['length_threshold'],
            "require_mask": metadata['require_mask'],
            "embedding_lattice_type": "fibonacci",
            "embedding_fib_m": cfg['embedding_fib_m'],
            "training_lattice_type": "fibonacci",
            "training_fib_m": cfg['training_fib_m'],
            "validation_fib_m": cfg['validation_fib_m'],
            "condition": condition,
        }
        if conditional is not None:
            contract["condition_bins"] = f"{CONFIG_DIRECTORY}/condition_bins.npz"
        write_json(output_dir / CONFIG_DIRECTORY / "training_contract.json", contract)

        finished = datetime.now()
        run_config = {
            "dataset": pathlib.Path(self.dataset_directory).name,
            "dataset_path": str(pathlib.Path(self.dataset_directory)),
            "train_split": train_path.name,
            "val_split": None if val_path is None else val_path.name,
            "n_train": n_train,
            "n_val_diagnostic": int(val_rows.shape[0]),
            "dataset_apply_mask": train_split['apply_mask'],
            "dataset_floor": metadata['floor'],
            "latent_dim": LATENT_DIM,
            "decoder_head": head,
            "seed": seed,
            "conditional": conditional,
            "c_dim": c_dim,
            "batches_per_epoch": n_batches,
            **condition_record,
            "protocol": "constant learning rate, last-epoch checkpoint",
            "train_qlvm": cfg,
            "device": jax.devices()[0].platform,
            "mean_seconds_per_epoch": float(np.mean(epoch_seconds)),
            "started": started.isoformat(timespec="seconds"),
            "finished": finished.isoformat(timespec="seconds"),
        }
        write_json(output_dir / CONFIG_DIRECTORY / "run_config.json", run_config)

        self.message_output(
            f"Wrote {checkpoint_path}, the training contract and run config in {output_dir / CONFIG_DIRECTORY} "
            f"and the losses in {diagnostics_path}."
        )
        self.message_output(
            f"QLVM training ended at: {finished.hour:02d}:{finished.minute:02d}:{finished.second:02d}."
        )


@click.command(name="train-qlvm")
@click.option('--dataset-directory', type=click.Path(exists=True, file_okay=False, dir_okay=True), required=True, help='Directory holding the .npz training set (train_data.npz or full_data.npz, val_data.npz, metadata.npz).')
@click.option('--output-directory', type=click.Path(file_okay=False, dir_okay=True), required=True, help='Directory the model package cell (checkpoint.tar, config/, metrics/) is written to.')
@click.option('--n-epochs', 'n_epochs', type=int, default=None, required=False, help='Number of training epochs.')
@click.option('--decoder-head', 'decoder_head', type=click.Choice(['relu', 'legacy']), default=None, required=False, help='Decoder head: relu (ReLU between the two Linear layers) or legacy (none).')
@click.option('--training-fib-m', 'training_fib_m', type=int, default=None, required=False, help='Fibonacci index of the training lattice (fib(m) points).')
@click.option('--validation-fib-m', 'validation_fib_m', type=int, default=None, required=False, help='Fibonacci index of the validation lattice (fib(m) points).')
@click.option('--embedding-fib-m', 'embedding_fib_m', type=int, default=None, required=False, help='Fibonacci index of the embedding lattice recorded in the training contract.')
@click.option('--batch-size', 'batch_size', type=int, default=None, required=False, help='Training batch size.')
@click.option('--learning-rate', 'learning_rate', type=float, default=None, required=False, help='Constant Adam learning rate.')
@click.option('--val-freq', 'val_freq', type=int, default=None, required=False, help='Compute the validation loss every N epochs (and after the last).')
@click.option('--val-samples-per-mask-count', 'val_samples_per_mask_count', type=int, default=None, required=False, help='Validation subset: at most this many spectrograms per masks_len value.')
@click.option('--seed', 'seed', type=int, default=None, required=False, help='Seed of the initialization, shuffling, lattice shifts and validation subset.')
@click.option('--conditional', 'conditional', type=click.Choice(['none', 'duration', 'mean_freq', 'bandwidth', 'loudness']), default=None, required=False, help='Train a decoder conditioned on this per-call value (the phase 11 recipe), or none for an unconditional decoder.')
@click.option('--condition-n-bins', 'condition_n_bins', type=int, default=None, required=False, help='Quantile bins the training values are cut into (conditional runs).')
@click.option('--condition-bin-scheme', 'condition_bin_scheme', type=click.Choice(['quantile', 'quantile_capped']), default=None, required=False, help='quantile, or quantile_capped (bins wider than the widest inner bin split into equal-width pieces).')
@click.option('--condition-scale-loss-by-batch/--no-condition-scale-loss-by-batch', 'condition_scale_loss_by_batch', default=None, required=False, help='Scale each step\'s loss by rows / batch size (conditional runs).')
@click.option('--condition-decode-grid-step', 'condition_decode_grid_step', type=float, default=None, required=False, help='Spacing of the decode grid written to condition_bins.npz.')
@click.option('--condition-table', 'condition_table', type=str, default=None, required=False, help='Per-call .npz (spec_id + mean_freq_hz / freq_bandwidth_hz / loudness_db or image_level_db) to read the raw values from instead of the splits\' columns, or none.')
@click.pass_context
def train_qlvm_cli(ctx, dataset_directory, output_directory, **kwargs) -> None:
    """
    Description
    -----------
    A command-line tool to train a QLVM decoder on a prebuilt ``.npz`` training
    set and write it as a model package cell (``checkpoint.tar``,
    ``config/training_contract.json``, ``config/run_config.json``,
    ``metrics/val_diagnostics.npz``, and ``config/condition_bins.npz`` for a
    ``--conditional`` decoder). ``--conditional none`` and ``--condition-table
    none`` set those settings to null.

    Parameters
    ----------

    Returns
    -------
    None
    """

    provided_params = [key for key in kwargs if ctx.get_parameter_source(key) == ParameterSource.COMMANDLINE]
    # "none" on the command line is the settings' null (an unconditional run, no table).
    for key in ('conditional', 'condition_table'):
        if key in provided_params and ctx.params[key].strip().lower() == "none":
            ctx.params[key] = None

    processing_settings_dict = modify_settings_json_for_cli(
        ctx=ctx,
        provided_params=provided_params,
        settings_dict='processing_settings',
        block='train_qlvm',
    )

    QLVMTrainer(
        dataset_directory=dataset_directory,
        output_directory=output_directory,
        input_parameter_dict=processing_settings_dict,
        message_output=print,
    ).train()
