"""
@author: bartulem
Embed a session's USV spectrograms into the trained QLVM toroidal latent space
and merge the latent coordinates + watershed categories into its
``*_usv_summary.csv``.

This is the in-house, JAX (torch-free) inference driver. It loads the frozen
decoder weights (a ``.npz`` converted once from the training checkpoint's
``state_dict``), rebuilds the fixed lattice, embeds the session's spectrograms
via :func:`qlvm_model.embed_data`, and assigns each USV a cluster by **spatial
lookup into two fixed reference watershed grids** — a FINE grid and a COARSE grid
(the torus-periodic ``ws_labels_periodic`` field of a fine and a coarse reference
``arrays.npz``) — NOT a per-session re-watershed, so clusters are comparable
across every session embedded into the same torus.

Columns written into ``usv_summary.csv`` (the ones the visualizations/tuning
code already consume): ``qlvm1``, ``qlvm2`` (torus coordinates),
``qlvm_category`` (FINE cluster label, e.g. 12 classes) and ``qlvm_supercategory``
(COARSE cluster label, e.g. 7 classes). The reference grids label every pixel
from 1, so there is no background / noise label 0; USVs that are not embedded
get nulls.

Fidelity: the session spectrograms are preprocessed with the SAME resize /
time-stretch used to build the training set (:func:`stretch_specs`), so they are
in-distribution for the decoder.

Model packages: with ``model_cell_directory`` set, one cell of a QLVM model
package (``qlvm_models_latest/v2``) replaces the weights / reference-arrays /
lattice settings (:func:`load_model_cell`): its ``checkpoint.tar`` is read without
torch, its ``training_contract.json`` fixes the head (legacy or ReLU), the input
normalization (min-max, and the loudness floor of floor-trained cells) and the
duration window, its Fibonacci embedding lattice is rebuilt, and its
``cluster/fine`` and ``cluster/coarse`` ``label_grid.npy`` supply the categories.
"""

from __future__ import annotations

import collections
import json
import pathlib
import pickle
import zipfile
from collections.abc import Callable
from datetime import datetime
from typing import BinaryIO

import click
import h5py
import jax.numpy as jnp
import numpy as np
import polars as pls
from click.core import ParameterSource

from ..cli_utils import modify_settings_json_for_cli
from ..os_utils import atomic_output_path, configure_path, derive_spectrogram_model_paths, first_match_or_raise
from ..processing.build_qlvm_training_set import build_session_masks, stretch_specs
from ..time_utils import is_gui_context, smart_wait
from .qlvm_model import decoder_head, embed_data, gen_fib_basis, gen_korobov_basis, roberts_sequence

# QLVM columns written into the USV summary CSV (consumed downstream).
QLVM_COLUMNS = ("qlvm1", "qlvm2", "qlvm_category", "qlvm_supercategory")


class _TorchCheckpointUnpickler(pickle.Unpickler):
    """
    Description
    -----------
    Unpickles the ``data.pkl`` of a torch zip checkpoint into numpy arrays without
    importing torch. Only the globals a saved ``state_dict`` of float tensors
    needs are resolved (``collections.OrderedDict``,
    ``torch._utils._rebuild_tensor_v2`` and the typed storage classes); anything
    else raises, so the file cannot run code. Each storage is read from the
    archive's ``data/<key>`` record.
    """

    def __init__(self, file: BinaryIO, archive: zipfile.ZipFile, record_prefix: str) -> None:
        """
        Description
        -----------
        Initializes the unpickler over one open ``data.pkl``.

        Parameters
        ----------
        file (BinaryIO)
            The open ``data.pkl`` record.
        archive (zipfile.ZipFile)
            The checkpoint archive the storages are read from.
        record_prefix (str)
            The archive's top-level folder name (records are ``<prefix>/data/<key>``).

        Returns
        -------
        None
        """
        super().__init__(file)
        self.archive = archive
        self.record_prefix = record_prefix

    def find_class(self, module: str, name: str) -> object:
        """
        Description
        -----------
        Resolves the few globals a tensor ``state_dict`` pickle refers to: the
        ordered dict, the tensor rebuild function (replaced by
        :func:`_rebuild_tensor_as_array`) and the typed storage classes (replaced by
        their numpy dtype). Every other global is refused.

        Parameters
        ----------
        module (str)
            Module of the global.
        name (str)
            Name of the global.

        Returns
        -------
        resolved (object)
            The replacement object.
        """
        storage_dtypes = {
            "FloatStorage": np.float32, "DoubleStorage": np.float64, "HalfStorage": np.float16,
            "LongStorage": np.int64, "IntStorage": np.int32, "ShortStorage": np.int16,
            "ByteStorage": np.uint8, "CharStorage": np.int8, "BoolStorage": np.bool_,
        }
        if (module, name) == ("collections", "OrderedDict"):
            return collections.OrderedDict
        if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
            return _rebuild_tensor_as_array
        if module == "torch" and name in storage_dtypes:
            return np.dtype(storage_dtypes[name])
        error_message = f"torch checkpoint refers to {module}.{name}, which a decoder state_dict does not need; refusing to load it."
        raise pickle.UnpicklingError(error_message)

    def persistent_load(self, pid: tuple) -> np.ndarray:
        """
        Description
        -----------
        Returns the flat storage a persistent id names, read from the archive as
        little-endian bytes of the storage's dtype.

        Parameters
        ----------
        pid (tuple)
            ``("storage", dtype, key, location, numel)``.

        Returns
        -------
        storage (np.ndarray)
            One-dimensional array over the storage bytes.
        """
        kind, dtype, key, _location, numel = pid
        if kind != "storage":
            error_message = f"torch checkpoint persistent id of kind {kind!r}; only 'storage' is supported."
            raise pickle.UnpicklingError(error_message)
        storage = np.frombuffer(self.archive.read(f"{self.record_prefix}/data/{key}"), dtype=dtype.newbyteorder("<"))
        return storage[:numel]


def _rebuild_tensor_as_array(
    storage: np.ndarray,
    storage_offset: int,
    size: tuple,
    stride: tuple,
    *_unused: object,
) -> np.ndarray:
    """
    Description
    -----------
    Stands in for ``torch._utils._rebuild_tensor_v2`` while unpickling a
    checkpoint: views the storage with the tensor's offset, shape and stride and
    returns a contiguous copy.

    Parameters
    ----------
    storage (np.ndarray)
        Flat storage from :meth:`_TorchCheckpointUnpickler.persistent_load`.
    storage_offset (int)
        Offset of the tensor's first element in the storage.
    size (tuple)
        Tensor shape.
    stride (tuple)
        Tensor stride, in elements.
    _unused (object)
        ``requires_grad``, backward hooks and metadata, which a plain array drops.

    Returns
    -------
    array (np.ndarray)
        The tensor as a numpy array.
    """
    view = np.lib.stride_tricks.as_strided(
        storage[storage_offset:],
        shape=tuple(size),
        strides=tuple(step * storage.itemsize for step in stride),
        writeable=False,
    )
    return np.array(view)


def read_torch_checkpoint(checkpoint_path: str | pathlib.Path) -> object:
    """
    Description
    -----------
    Reads a torch zip checkpoint (``torch.save``'s default format, e.g. a QLVM
    model package's ``checkpoint.tar``) into plain Python containers of numpy
    arrays, without importing torch, via :class:`_TorchCheckpointUnpickler`.

    Parameters
    ----------
    checkpoint_path (str | pathlib.Path)
        Path to the checkpoint.

    Returns
    -------
    checkpoint (object)
        The unpickled object, with every tensor as a numpy array (for a QLVM
        checkpoint, a dict with ``"model"``, ``"optimizer"`` and ``"run info"``).
    """
    with zipfile.ZipFile(checkpoint_path) as archive:
        pickle_records = [name for name in archive.namelist() if name.endswith("/data.pkl")]
        if len(pickle_records) != 1:
            error_message = f"{checkpoint_path} is not a torch zip checkpoint (data.pkl records: {pickle_records})."
            raise ValueError(error_message)
        record_prefix = pickle_records[0][: -len("/data.pkl")]
        byteorder_record = f"{record_prefix}/byteorder"
        if byteorder_record in archive.namelist() and archive.read(byteorder_record) != b"little":
            error_message = f"{checkpoint_path} was saved big-endian; only little-endian checkpoints are supported."
            raise ValueError(error_message)
        with archive.open(pickle_records[0]) as pickle_file:
            return _TorchCheckpointUnpickler(pickle_file, archive, record_prefix).load()


def load_decoder_params(weights_npz_path: str) -> dict[str, jnp.ndarray]:
    """
    Description
    -----------
    Loads the frozen decoder weights into the key form
    :func:`qlvm_model.decoder_forward` expects, from either the converted
    ``.npz`` (one array per ``state_dict`` entry, as ``train-qlvm`` writes it) or,
    for any other suffix, a torch zip checkpoint read without torch
    (:func:`read_torch_checkpoint`; its ``"model"`` entry when present). A leading
    ``decoder.`` prefix (present when the full QMCLVM ``state_dict`` is dumped) is
    stripped.

    Parameters
    ----------
    weights_npz_path (str)
        Path to the ``.npz`` of decoder weights or to a torch checkpoint.

    Returns
    -------
    params (dict[str, jnp.ndarray])
        Decoder weights keyed by ``"<layer_idx>.weight"`` / ``"<layer_idx>.bias"``.
    """
    weights_path = pathlib.Path(configure_path(weights_npz_path))
    if weights_path.suffix == ".npz":
        # Context manager closes the zip-backed NpzFile handle; every array is copied
        # out inside the block, so closing on exit is safe.
        with np.load(weights_path) as raw:
            state = {key: raw[key] for key in raw.files}
    else:
        checkpoint = read_torch_checkpoint(weights_path)
        state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    params: dict[str, jnp.ndarray] = {}
    for key, value in state.items():
        clean = key[len("decoder."):] if key.startswith("decoder.") else key
        params[clean] = jnp.asarray(value)
    return params


def load_training_contract(weights_npz_path: str) -> dict | None:
    """
    Description
    -----------
    Reads the training contract ``train-qlvm`` writes beside the decoder weights
    (same stem, ``.json``; e.g. ``qmc_decoder_weights.json``). Weights trained
    before the contract existed have none, which is reported as ``None`` rather
    than raised so they keep embedding.

    Parameters
    ----------
    weights_npz_path (str)
        Path to the decoder weights ``.npz``.

    Returns
    -------
    contract (dict | None)
        The parsed contract, or ``None`` when no ``.json`` sits beside the weights.
    """
    contract_path = pathlib.Path(configure_path(weights_npz_path)).with_suffix(".json")
    if not contract_path.is_file():
        return None
    with contract_path.open() as contract_file:
        return json.load(contract_file)


def load_model_cell(model_cell_directory: str) -> dict:
    """
    Description
    -----------
    Loads one cell of a QLVM model package (the ``qlvm_models_latest/v2`` layout):
    the decoder weights from its torch ``checkpoint.tar`` (read without torch), its
    ``training_contract.json``, the Fibonacci lattice the package embedded its
    corpus on (``embedding_fib_m`` of the contract), and the ``label_grid.npy`` of
    its ``cluster/fine`` and ``cluster/coarse`` levels (the grids its per-call
    labels were read from, indexed ``[y, x]`` like the reference ``arrays.npz``).

    Parameters
    ----------
    model_cell_directory (str)
        Path to the package cell, e.g.
        ``.../qlvm_models_latest/v2/phase9_USVs_masked_relu/natural_3strata_N65000_masked``.

    Returns
    -------
    model (dict)
        ``params`` (decoder weights), ``contract`` (dict), ``lattice``
        (``(fib(m), 2)``), ``fine_grid`` and ``coarse_grid`` (``(res, res)`` label
        grids), and ``model_id`` (``<package>/<phase>/<cell>``, the last three path
        components).
    """
    cell = pathlib.Path(configure_path(model_cell_directory))
    with (cell / "training_contract.json").open() as contract_file:
        contract = json.load(contract_file)
    if contract["embedding_lattice_type"] != "fibonacci":
        error_message = (
            f"{cell}: training_contract.json embedding_lattice_type is {contract['embedding_lattice_type']!r}; "
            f"only 'fibonacci' package embeddings are supported."
        )
        raise ValueError(error_message)
    return {
        "params": load_decoder_params(str(cell / "checkpoint.tar")),
        "contract": contract,
        "lattice": gen_fib_basis(contract["embedding_fib_m"]),
        "fine_grid": np.load(cell / "cluster" / "fine" / "label_grid.npy", allow_pickle=False),
        "coarse_grid": np.load(cell / "cluster" / "coarse" / "label_grid.npy", allow_pickle=False),
        "model_id": "/".join(cell.parts[-3:]),
    }


def normalize_model_inputs(spectrograms: np.ndarray, contract: dict | None) -> np.ndarray:
    """
    Description
    -----------
    Applies the input normalization a decoder was trained with to resized
    spectrograms, as its training contract records it. ``input_normalization``
    ``"none"`` (``train-qlvm`` decoders, and weights without a contract) leaves the
    stored values as they are. ``"minmax"`` (QLVM model packages) rescales each
    spectrogram to ``(x - min) / (max - min + normalization_epsilon)`` in float32,
    and a non-null ``floor`` then applies ``clip((x - floor) / (1 - floor), 0, 1)``
    followed by a second min-max -- the order the package's ``model_input`` uses.
    SAM masking, when the contract asks for it, has already zeroed the background,
    and a min-max keeps those zeros.

    Parameters
    ----------
    spectrograms (np.ndarray)
        Resized spectrograms, shape ``(N, F, T)``.
    contract (dict | None)
        The training contract, or ``None`` for weights without one.

    Returns
    -------
    inputs (np.ndarray)
        ``(N, F, T)`` float32 decoder inputs.
    """
    inputs = np.asarray(spectrograms, dtype=np.float32)
    if contract is None or contract["input_normalization"] == "none":
        return inputs
    epsilon = np.float32(contract["normalization_epsilon"])

    def _minmax(x: np.ndarray) -> np.ndarray:
        low = x.min(axis=(1, 2), keepdims=True)
        high = x.max(axis=(1, 2), keepdims=True)
        return (x - low) / ((high - low) + epsilon)

    inputs = _minmax(inputs)
    if contract["floor"] is not None:
        floor = np.float32(contract["floor"])
        inputs = np.clip((inputs - floor) / np.float32(1.0 - floor), np.float32(0.0), np.float32(1.0))
        inputs = _minmax(inputs)
    return inputs.astype(np.float32, copy=False)


def enforce_training_contract(contract: dict, cfg: dict, params: dict[str, jnp.ndarray]) -> float:
    """
    Description
    -----------
    Checks the ``infer_qlvm_latents`` settings and the loaded weights against the
    decoder's training contract (``train-qlvm``'s, see
    :func:`load_training_contract`, or a model package cell's, see
    :func:`load_model_cell`) and returns the duration window to embed. Every
    disagreement is collected and raised together: ``masking_type``,
    ``target_shape``, ``time_stretch`` and ``latent_dim`` must equal the
    contract's; a ``length_threshold`` set in the settings must equal the training
    set's; the weights' head (:func:`qlvm_model.decoder_head`) must be the
    contract's ``decoder_head``. The contract must also describe a decoder this
    module can run: no conditioning input (``c_dim`` 0), an
    ``input_normalization`` of ``"none"`` or ``"minmax"``, and a ``floor`` that is
    null or in ``[0, 1)``.

    Parameters
    ----------
    contract (dict)
        The training contract.
    cfg (dict)
        The ``infer_qlvm_latents`` settings block.
    params (dict[str, jnp.ndarray])
        The loaded decoder weights.

    Returns
    -------
    length_threshold (float)
        The training set's duration bound: calls with ``duration >= length_threshold``
        were never trained on.
    """
    mismatches = [
        f"{key}: settings {cfg[key]!r}, trained {contract[key]!r}"
        for key in ("masking_type", "time_stretch", "latent_dim")
        if cfg[key] != contract[key]
    ]
    if [int(value) for value in cfg["target_shape"]] != contract["target_shape"]:
        mismatches.append(f"target_shape: settings {list(cfg['target_shape'])!r}, trained {contract['target_shape']!r}")
    if cfg["length_threshold"] is not None and float(cfg["length_threshold"]) != contract["length_threshold"]:
        mismatches.append(f"length_threshold: settings {cfg['length_threshold']!r}, trained {contract['length_threshold']!r}")
    weights_head = decoder_head(params)
    if contract["decoder_head"] != weights_head:
        mismatches.append(f"decoder_head: the weights are {weights_head!r}, the contract says {contract['decoder_head']!r}")
    if contract["c_dim"] != 0:
        mismatches.append(f"c_dim: this module embeds unconditional decoders (c_dim 0), trained {contract['c_dim']!r}")
    if contract["input_normalization"] not in ("none", "minmax"):
        mismatches.append(f"input_normalization: expected 'none' or 'minmax', trained {contract['input_normalization']!r}")
    if contract["floor"] is not None and not 0.0 <= contract["floor"] < 1.0:
        mismatches.append(f"floor: expected null or a value in [0, 1), trained {contract['floor']!r}")
    if mismatches:
        error_message = (
            "infer_qlvm_latents settings disagree with the decoder's training contract:\n  "
            + "\n  ".join(mismatches)
        )
        raise ValueError(error_message)
    return float(contract["length_threshold"])


def build_lattice(cfg: dict) -> jnp.ndarray:
    """
    Description
    -----------
    Rebuilds the fixed QLVM lattice from the training configuration so inference
    uses the exact grid the model was trained on.

    Parameters
    ----------
    cfg (dict)
        The ``infer_qlvm_latents`` settings block (``lattice_type``,
        ``latent_dim``, ``n_points``, ``korobov_a``, ``fib_m``).

    Returns
    -------
    lattice (jnp.ndarray)
        Lattice points, shape ``(n_points, latent_dim)``.
    """
    lattice_type = cfg["lattice_type"]
    if lattice_type == "korobov":
        return gen_korobov_basis(cfg["korobov_a"], cfg["latent_dim"], cfg["n_points"])
    if lattice_type == "roberts":
        return roberts_sequence(cfg["n_points"], cfg["latent_dim"])
    if lattice_type == "fibonacci":
        return gen_fib_basis(cfg["fib_m"])
    msg = f"build_lattice: unknown lattice_type {lattice_type!r} (expected korobov|roberts|fibonacci)."
    raise ValueError(msg)


def labels_for_coords(
    coords: np.ndarray,
    fine_grid: np.ndarray,
    coarse_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Description
    -----------
    Looks up each torus coordinate's cluster label in the FINE and COARSE
    reference watershed grids, using the convention
    ``label = grid[int(y * res), int(x * res)]`` (with clipping to each grid's
    resolution). Each grid is the torus-periodic ``ws_labels_periodic`` field of
    its reference ``arrays.npz`` (periodic = correct for the native-torus QLVM
    coordinates, which wrap at the seam). The fine grid yields the per-USV
    ``qlvm_category`` (e.g. 12 clusters); the coarse grid yields the broader
    ``qlvm_supercategory`` (e.g. 7 clusters).

    Parameters
    ----------
    coords (np.ndarray)
        Torus coordinates in ``[0, 1)``, shape ``(N, 2)`` ordered ``(x, y)``.
    fine_grid (np.ndarray)
        Fine-granularity periodic watershed label grid, shape ``(res, res)``.
    coarse_grid (np.ndarray)
        Coarse-granularity periodic watershed label grid, shape ``(res, res)``.

    Returns
    -------
    category (np.ndarray)
        Fine cluster labels, shape ``(N,)``.
    supercategory (np.ndarray)
        Coarse cluster labels, shape ``(N,)``.
    """

    def _lookup(grid: np.ndarray) -> np.ndarray:
        res = grid.shape[0]
        px = np.clip((coords[:, 0] * res).astype(int), 0, res - 1)
        py = np.clip((coords[:, 1] * res).astype(int), 0, res - 1)
        return grid[py, px]

    return _lookup(fine_grid), _lookup(coarse_grid)


class QLVMLatentInference:
    """
    Description
    -----------
    Embeds one session's spectrograms into the trained QLVM torus and merges the
    coordinates + watershed categories into its ``*_usv_summary.csv``.
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
        Initializes the QLVMLatentInference.

        Parameters
        ----------
        root_directory (str)
            Session root directory (contains the ``audio`` tree).
        input_parameter_dict (dict)
            Processing settings; the ``infer_qlvm_latents`` block supplies the
            weight/reference paths and lattice configuration.
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

    def infer_and_merge(self) -> None:
        """
        Description
        -----------
        Loads the decoder weights, lattice and reference watershed grids; reads
        the session spectrogram H5, preprocesses identically to training, embeds
        into the torus, assigns categories by reference lookup, and merges
        ``qlvm_*`` columns into the matching USV summary rows (joined on the
        positional USV row index, since the spectrogram rows are 1:1 with the
        ``usv_summary.csv`` rows; USVs with non-positive duration, or with a
        duration at or above the training set's ``length_threshold``, are skipped
        and get nulls; any pre-existing ``qlvm_*`` columns are replaced). When the
        weights carry a training contract (:func:`load_training_contract`), the
        settings are checked against it first (:func:`enforce_training_contract`)
        and its ``length_threshold`` applies; otherwise the settings'
        ``length_threshold`` does (``null`` embeds every positive duration). The
        summary is rewritten atomically.

        Parameters
        ----------

        Returns
        -------
        Updated ``*_usv_summary.csv`` with the ``qlvm_*`` columns.
        """
        self.message_output(
            f"QLVM latent inference started at: {datetime.now().hour:02d}:{datetime.now().minute:02d}:{datetime.now().second:02d}."
        )
        smart_wait(app_context_bool=self.app_context_bool, seconds=1)

        derive_spectrogram_model_paths(self.input_parameter_dict)
        cfg = self.input_parameter_dict['infer_qlvm_latents']
        if cfg['model_cell_directory']:
            # A QLVM model package cell brings its own weights, contract, embedding
            # lattice and label grids; the settings' paths and lattice keys are unused.
            model = load_model_cell(cfg['model_cell_directory'])
            params, contract, lattice = model['params'], model['contract'], model['lattice']
            fine_grid, coarse_grid = model['fine_grid'], model['coarse_grid']
            self.message_output(
                f"Embedding with model package cell {model['model_id']} ({decoder_head(params)} head, "
                f"{lattice.shape[0]}-point Fibonacci lattice)."
            )
        else:
            params = load_decoder_params(cfg['weights_npz_path'])
            lattice = build_lattice(cfg)
            contract = load_training_contract(cfg['weights_npz_path'])
            # Fine grid -> qlvm_category; coarse grid -> qlvm_supercategory. Both are
            # the torus-periodic watershed (ws_labels_periodic) of their reference file.
            # Context managers close each zip-backed NpzFile handle; the grid array is
            # fully materialized on access inside the block, so closing on exit is safe.
            with np.load(configure_path(cfg['reference_arrays_fine_npz_path'])) as fine_ref:
                fine_grid = fine_ref['ws_labels_periodic']
            with np.load(configure_path(cfg['reference_arrays_coarse_npz_path'])) as coarse_ref:
                coarse_grid = coarse_ref['ws_labels_periodic']

        # The decoder only knows calls shaped like its training set: check the
        # preprocessing settings against its contract and embed only calls inside the
        # set's duration window. Weights with no contract fall back to the settings.
        if contract is None:
            length_threshold = cfg['length_threshold']
            self.message_output(
                "No training contract beside the decoder weights; the infer_qlvm_latents settings are used as given "
                f"(length_threshold={length_threshold})."
            )
        else:
            length_threshold = enforce_training_contract(contract, cfg, params)

        root = pathlib.Path(self.root_directory)
        # Session-keyed, NOT "*_spectrograms.h5": a session can hold other files
        # ending in that suffix (e.g. a sonic-band
        # "<session>_3_30khz_spectrograms.h5"), which a wildcard may select
        # ahead of the real one.
        h5_loc = first_match_or_raise(
            root=root / "audio" / "spectrograms",
            pattern=f"{root.name}_spectrograms.h5",
            label="per-session spectrogram H5",
        )
        with h5py.File(h5_loc, "r") as h5_file:
            session_group = h5_file[f"spectrogram/{root.name}"]
            specs = session_group["spectrograms"][:]
            durations = session_group["durations"][:]
            # spectrogram rows are 1:1 with usv_summary.csv; embed only the real
            # (duration > 0) USVs inside the training duration window (the
            # 0 < duration < length_threshold rule build_qlvm_training_set applies)
            # and remember their row positions for the merge.
            in_window = durations > 0
            if length_threshold is not None:
                in_window &= durations < length_threshold
                n_too_long = int(np.count_nonzero((durations > 0) & (durations >= length_threshold)))
                self.message_output(
                    f"{n_too_long} USVs with duration >= {length_threshold} (outside the training set) get null qlvm_* columns."
                )
            usv_indices = np.flatnonzero(in_window).astype(np.uint32)
            specs = specs[usv_indices].astype(np.float32)
            durations = durations[usv_indices]
            # Apply the SAM mask exactly as build_qlvm_training_set does, so the
            # decoder -- trained on masked (background-zeroed) spectrograms -- receives
            # in-distribution input. Embedding raw spectrograms into a masked-trained
            # decoder is out-of-distribution and yields unreliable coordinates.
            # masking_type "none" keeps raw spectrograms (correct only if the decoder
            # was trained without masking).
            if cfg['masking_type'] == 'sam':
                masks, _ = build_session_masks(
                    h5_file, root.name, usv_indices, specs.shape[1], specs.shape[2]
                )
                specs = specs * masks

        # Preprocess identically to the training set (same resize/time-stretch), then
        # normalize the way the decoder's contract says it was fed.
        target_shape = tuple(int(v) for v in cfg['target_shape'])
        resized = stretch_specs(specs, durations, target_shape, cfg['time_stretch'])
        data = jnp.asarray(normalize_model_inputs(resized, contract)[:, None, :, :])

        coords = np.asarray(embed_data(
            lattice, data, params, cfg['lattice_batch_size'], cfg['data_batch_size']
        ))                                                               # (N, 2)
        category, supercategory = labels_for_coords(coords, fine_grid, coarse_grid)

        qlvm_df = pls.DataFrame({
            "_usv_row": usv_indices,
            "qlvm1": coords[:, 0].astype(np.float64),
            "qlvm2": coords[:, 1].astype(np.float64),
            "qlvm_category": category.astype(np.int64),
            "qlvm_supercategory": supercategory.astype(np.int64),
        })

        usv_summary_loc = first_match_or_raise(
            root=root / "audio",
            pattern="*_usv_summary.csv",
            recursive=True,
            label="USV summary CSV",
        )
        usv_df = pls.read_csv(source=str(usv_summary_loc), schema_overrides={"usv_id": pls.String})
        usv_df = usv_df.drop([c for c in QLVM_COLUMNS if c in usv_df.columns])
        usv_df = usv_df.with_row_index(name="_usv_row")
        merged = usv_df.join(qlvm_df, on="_usv_row", how="left").drop("_usv_row")
        # usv_summary.csv holds every other per-USV column too: publish atomically so
        # a failed write leaves the previous file intact instead of a truncated one.
        with atomic_output_path(usv_summary_loc) as tmp_summary_path:
            merged.write_csv(file=str(tmp_summary_path))

        self.message_output(
            f"Merged QLVM latents/categories for {len(usv_indices)} USVs into {usv_summary_loc.name}."
        )
        self.message_output(
            f"QLVM latent inference ended at: {datetime.now().hour:02d}:{datetime.now().minute:02d}:{datetime.now().second:02d}."
        )


@click.command(name="infer-qlvm-latents")
@click.option('--root-directory', type=click.Path(exists=True, file_okay=False, dir_okay=True), required=True, help='Session root directory path.')
@click.option('--model-cell-directory', 'model_cell_directory', type=str, default=None, required=False, help='A QLVM model package cell (e.g. .../qlvm_models_latest/v2/phase9_USVs_masked_relu/natural_3strata_N65000_masked); when set, its checkpoint, training_contract.json, embedding lattice and label grids replace the weights, reference-arrays and lattice settings.')
@click.option('--weights-npz-path', 'weights_npz_path', type=str, default=None, required=False, help='Path to the converted decoder weights .npz.')
@click.option('--reference-arrays-fine-npz-path', 'reference_arrays_fine_npz_path', type=str, default=None, required=False, help='Path to the FINE reference arrays.npz (ws_labels_periodic -> qlvm_category).')
@click.option('--reference-arrays-coarse-npz-path', 'reference_arrays_coarse_npz_path', type=str, default=None, required=False, help='Path to the COARSE reference arrays.npz (ws_labels_periodic -> qlvm_supercategory).')
@click.option('--lattice-type', 'lattice_type', type=click.Choice(['korobov', 'roberts', 'fibonacci']), default=None, required=False, help='Quasi-random lattice generator used to rebuild the fixed QLVM (quasi-Monte Carlo latent variable model) lattice at inference.')
@click.option('--latent-dim', 'latent_dim', type=int, default=None, required=False, help='Dimensionality of the toroidal latent space.')
@click.option('--n-points', 'n_points', type=int, default=None, required=False, help='Number of lattice points used at inference.')
@click.option('--korobov-a', 'korobov_a', type=int, default=None, required=False, help='Korobov generating integer (used when lattice-type=korobov).')
@click.option('--fib-m', 'fib_m', type=int, default=None, required=False, help='Fibonacci lattice order m (used when lattice-type=fibonacci).')
@click.option('--time-stretch/--no-time-stretch', 'time_stretch', default=None, required=False, help='Whether to time-stretch each spectrogram to the fixed size (matching training preprocessing) instead of a plain resize.')
@click.option('--masking-type', 'masking_type', type=click.Choice(['sam', 'none']), default=None, required=False, help='Apply SAM mask regions before embedding ("sam", matching training) or embed raw spectrograms ("none").')
@click.option('--target-shape', 'target_shape', nargs=2, type=int, default=None, required=False, help='Output spectrogram (freq, time) shape as two ints, matching the training preprocessing, e.g. --target-shape 128 128.')
@click.option('--length-threshold', 'length_threshold', type=float, default=None, required=False, help='Embed only USVs with duration below this (time bins); must equal the training contract when the weights carry one. Unset in the settings (null) with no contract, every positive duration is embedded.')
@click.option('--lattice-batch-size', 'lattice_batch_size', type=int, default=None, required=False, help='Lattice points decoded and scored per block; lower it to cut memory on large lattices.')
@click.option('--data-batch-size', 'data_batch_size', type=int, default=None, required=False, help='Spectrograms whose lattice posteriors are computed together; memory grows with this times the lattice size.')
@click.pass_context
def infer_qlvm_latents_cli(ctx, root_directory, **kwargs) -> None:
    """
    Description
    -----------
    A command-line tool to embed a session's USV spectrograms into the QLVM
    torus and merge the latents/categories into its USV summary CSV.

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
        parameters_lists=['target_shape'],
        block='infer_qlvm_latents',
    )

    QLVMLatentInference(
        root_directory=root_directory,
        input_parameter_dict=processing_settings_dict,
        message_output=print,
    ).infer_and_merge()
