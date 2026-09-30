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

The decoder takes no conditioning input (``c_dim`` 0). A conditional decoder
would append its conditioning value to the torus basis of the whole lattice once
per batch (:func:`qlvm_model.decode_shifted_lattice`), with batches drawn so
their rows share one value; that is not implemented here.
"""

from __future__ import annotations

import collections
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
from .qlvm_latents import minmax_per_spectrogram
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
        ``durations`` ``(N,)`` int64, and ``apply_mask`` (bool).
    """
    with np.load(npz_path, allow_pickle=False) as split:
        apply_mask = bool(split["apply_mask"]) if "apply_mask" in split.files else metadata_masking_type == "sam"
        spectrograms = np.asarray(split["spectrograms"], dtype=np.float32)
        masks = split["masks"] if apply_mask else None
        masks_len = np.asarray(split["masks_len"], dtype=np.int64)
        durations = np.asarray(split["durations"], dtype=np.int64)
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
        ``(params, opt_state, batch, shift) -> (params, opt_state, loss)``, with
        ``batch`` ``(B, 1, F, T)`` and ``shift`` ``(1, 2)``.
    validation_loss (Callable)
        ``(params, spectrogram, shift) -> loss`` for one ``(F, T)`` spectrogram.
    """

    def batch_loss(params: dict, batch: jnp.ndarray, shift: jnp.ndarray) -> jnp.ndarray:
        """
        Description
        -----------
        The objective of one batch: the negative mean QMC log evidence of the
        batch under the training lattice shifted by ``shift``.

        Parameters
        ----------
        params (dict)
            Decoder weights.
        batch (jnp.ndarray)
            ``(B, 1, F, T)`` decoder inputs.
        shift (jnp.ndarray)
            ``(1, 2)`` torus shift.

        Returns
        -------
        loss (jnp.ndarray)
            Scalar loss.
        """
        return binary_evidence(decode_shifted_lattice(training_lattice, shift, params), batch)

    @jax.jit
    def train_step(params: dict, opt_state: optax.OptState, batch: jnp.ndarray, shift: jnp.ndarray) -> tuple:
        """
        Description
        -----------
        One Adam step on the gradient of :func:`batch_loss`.

        Parameters
        ----------
        params (dict)
            Decoder weights.
        opt_state (optax.OptState)
            Adam state.
        batch (jnp.ndarray)
            ``(B, 1, F, T)`` decoder inputs.
        shift (jnp.ndarray)
            ``(1, 2)`` torus shift.

        Returns
        -------
        params (dict)
            Updated decoder weights.
        opt_state (optax.OptState)
            Updated Adam state.
        loss (jnp.ndarray)
            The batch loss before the update.
        """
        loss, grads = jax.value_and_grad(batch_loss)(params, batch, shift)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    @jax.jit
    def validation_loss(params: dict, spectrogram: jnp.ndarray, shift: jnp.ndarray) -> jnp.ndarray:
        """
        Description
        -----------
        The objective of one validation spectrogram against the validation
        lattice shifted by ``shift`` (the reference trainer validates with batches
        of one, each with its own shift).

        Parameters
        ----------
        params (dict)
            Decoder weights.
        spectrogram (jnp.ndarray)
            ``(F, T)`` decoder input.
        shift (jnp.ndarray)
            ``(1, 2)`` torus shift.

        Returns
        -------
        loss (jnp.ndarray)
            Scalar loss.
        """
        return binary_evidence(decode_shifted_lattice(validation_lattice, shift, params), spectrogram[None, None])

    return train_step, validation_loss


def torch_checkpoint(
    params: dict[str, jnp.ndarray],
    opt_state: optax.OptState,
    head: str,
    learning_rate: float,
    batch_losses: list[float],
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

    Returns
    -------
    checkpoint (dict)
        ``{"model": OrderedDict, "optimizer": dict, "run info": list}`` of torch
        tensors and Python values, ready for ``torch.save``.
    """
    names = list(decoder_parameter_shapes(latent_dim=LATENT_DIM, c_dim=0, head=head))
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
        records into the output directory.

        Parameters
        ----------

        Returns
        -------
        ``checkpoint.tar`` + ``config/training_contract.json`` + ``config/run_config.json`` + ``metrics/val_diagnostics.npz``
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

        val_inputs = None
        val_rows = np.zeros(0, dtype=np.int64)
        val_masks_len = np.zeros(0, dtype=np.int64)
        val_durations = np.zeros(0, dtype=np.int64)
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
            self.message_output(
                f"Validation subset: {val_rows.shape[0]} of {val_split['inputs'].shape[0]} spectrograms in "
                f"{val_path.name} (at most {cfg['val_samples_per_mask_count']} per masks_len value)."
            )
            del val_split

        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / CONFIG_DIRECTORY).mkdir(exist_ok=True)
        (output_dir / METRICS_DIRECTORY).mkdir(exist_ok=True)

        training_lattice = jnp.asarray(gen_fib_basis(cfg['training_fib_m']), dtype=jnp.float32)
        validation_lattice = jnp.asarray(gen_fib_basis(cfg['validation_fib_m']), dtype=jnp.float32)
        optimizer = optax.adam(learning_rate, b1=_ADAM_BETAS[0], b2=_ADAM_BETAS[1], eps=_ADAM_EPSILON)
        train_step, validation_loss = make_step_functions(optimizer, training_lattice, validation_lattice)

        init_key, shift_key, validation_key = jax.random.split(jax.random.PRNGKey(seed), 3)
        params = init_decoder_params(init_key, latent_dim=LATENT_DIM, c_dim=0, head=head)
        opt_state = optimizer.init(params)
        shuffle_generator = np.random.default_rng(seed)
        n_batches = math.ceil(n_train / batch_size)
        self.message_output(
            f"Training the {head!r}-head decoder on {jax.devices()[0].platform}: {n_epochs} epochs x {n_batches} "
            f"batches of {batch_size}, a {training_lattice.shape[0]}-point training lattice and a "
            f"{validation_lattice.shape[0]}-point validation lattice."
        )

        batch_losses: list[float] = []
        epoch_train_losses: list[float] = []
        val_epochs: list[int] = []
        val_losses: list[float] = []
        epoch_seconds: list[float] = []
        for epoch in range(n_epochs):
            epoch_start = time.perf_counter()
            order = shuffle_generator.permutation(n_train)
            shift_key, epoch_key = jax.random.split(shift_key)
            shifts = jax.random.uniform(epoch_key, (n_batches, 1, LATENT_DIM), dtype=jnp.float32)
            device_losses = []
            for n_batch in range(n_batches):
                rows = order[n_batch * batch_size:(n_batch + 1) * batch_size]
                batch = jnp.asarray(train_inputs[rows][:, None])
                params, opt_state, loss = train_step(params, opt_state, batch, shifts[n_batch])
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
                        validation_loss(params, jnp.asarray(val_inputs[n_row]), val_shifts[n_row])
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
            torch.save(torch_checkpoint(params, opt_state, head, learning_rate, batch_losses), checkpoint_file)

        diagnostics_path = output_dir / METRICS_DIRECTORY / "val_diagnostics.npz"
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
            )

        # The training contract, in the fields a v3 model package cell's contract has;
        # infer-qlvm-latents checks its settings against it (enforce_training_contract).
        contract = {
            "decoder_head": head,
            "latent_dim": LATENT_DIM,
            "c_dim": 0,
            "conditional": None,
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
            "condition": None,
        }
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
@click.pass_context
def train_qlvm_cli(ctx, dataset_directory, output_directory, **kwargs) -> None:
    """
    Description
    -----------
    A command-line tool to train a QLVM decoder on a prebuilt ``.npz`` training
    set and write it as a model package cell (``checkpoint.tar``,
    ``config/training_contract.json``, ``config/run_config.json``,
    ``metrics/val_diagnostics.npz``).

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
        block='train_qlvm',
    )

    QLVMTrainer(
        dataset_directory=dataset_directory,
        output_directory=output_directory,
        input_parameter_dict=processing_settings_dict,
        message_output=print,
    ).train()
