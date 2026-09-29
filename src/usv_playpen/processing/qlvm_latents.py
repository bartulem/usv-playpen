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
get nulls. ``qlvm_model`` names the model the other four came from (the model
package cell ``<package>/<phase>/<cell>``, or the decoder weights path), because
labels of different models share the column names but not their meaning.

Fidelity: the session spectrograms are preprocessed with the SAME resize /
time-stretch used to build the training set (:func:`stretch_specs`), so they are
in-distribution for the decoder.

Model packages: with ``model_cell_directory`` set, one cell of a QLVM model
package (``qlvm_models_latest/v2``) replaces the weights / reference-arrays /
lattice settings (:func:`load_model_cell`): its ``checkpoint.tar`` is read without
torch, its ``training_contract.json`` fixes the head (legacy or ReLU), the input
normalization (min-max, and the loudness floor of floor-trained cells) and the
duration window, its Fibonacci embedding lattice is rebuilt, and its
fine and coarse ``label_grid.npy`` (``inference/clusters_<level>/`` in v3,
``cluster/<level>/`` in v2 / v2.1) supply the categories.
Conditional cells take one conditioning value per call: phase 10 cells
(``qlvm_models_latest/v2``, duration or mean frequency) decode it at the frozen
corpus bin mean, phase 11 cells (``qlvm_models_latest/v3``, duration, mean
frequency, bandwidth or loudness) at the call's own value clamped to the training
range or snapped to the cell's decode grid, as the contract's ``condition.decode``
says (:func:`frozen_condition_values`).

Several models in one run: with ``model_cells`` (column prefix -> package cell)
the session is placed on the torus of every listed cell and each prefix ``P``
gets the float columns ``P1`` / ``P2`` and integer cluster labels read off the
cell's ``label_grid.npy`` at the pixel of those coordinates
(:func:`label_grid_lookup`, labels ``1..k`` with 1 the largest cluster, nulls where
the call was not placed; no ``qlvm_model``). Which levels a prefix writes is the
``model_cell_label_levels`` setting (prefix -> levels among ``"fine"`` and
``"coarse"``); its default ``{}`` writes ``qlvm_category`` (fine) and
``qlvm_supercategory`` (coarse) for the regular model's prefix ``"qlvm"`` and
``P_category`` (fine) for every other prefix, e.g. ``qlvm_dur_category``
(:func:`model_cell_label_columns`).
Per model, a corpus session whose spectrogram H5 is verifiably the one the
package was built from (``SESSION_H5_BASELINE.tsv`` SHA-256, row count, and the
package's per-row durations and mask counts) takes the package's own coordinates
(:func:`package_route_verdict`, :func:`load_package_session_rows`); every other
session is embedded with the cell as above.
"""

from __future__ import annotations

import collections
import functools
import json
import pathlib
import pickle
import zipfile
from collections.abc import Callable, Iterable
from datetime import datetime
from typing import BinaryIO

import click
import h5py
import jax.numpy as jnp
import numpy as np
import polars as pls
from click.core import ParameterSource

from ..cli_utils import modify_settings_json_for_cli
from ..os_utils import (
    QLVM_PRODUCTION_MODEL_CELLS,
    USV_SUMMARY_COLUMN_ORDER,
    atomic_output_path,
    cell_cluster_directory,
    configure_path,
    derive_spectrogram_model_paths,
    first_match_or_raise,
    order_usv_summary_columns,
)
from ..processing.build_qlvm_training_set import (
    build_session_masks,
    file_sha256,
    stretch_specs,
)
from ..time_utils import is_gui_context, smart_wait
from .compute_usv_loudness import session_image_level_db
from .qlvm_model import (
    decoder_head,
    embed_data,
    gen_fib_basis,
    gen_korobov_basis,
    roberts_sequence,
    torus_basis_reverse,
)

# QLVM columns written into the USV summary CSV (consumed downstream).
QLVM_COLUMNS = ("qlvm1", "qlvm2", "qlvm_category", "qlvm_supercategory", "qlvm_model")

# Cluster-label levels a model package cell holds (inference/clusters_<level>/ in v3),
# and the column-name suffix each gets in a model_cells run: <prefix>_category for the
# fine level, <prefix>_supercategory for the coarse one (qlvm_category /
# qlvm_supercategory for the regular model's prefix, see model_cell_label_column).
LABEL_LEVELS = ("fine", "coarse")
LABEL_LEVEL_SUFFIXES = {"fine": "category", "coarse": "supercategory"}

# The model_cells prefix of the regular (unconditional) model: it writes both label
# levels by default, every other prefix the fine level only.
REGULAR_MODEL_PREFIX = "qlvm"

# Conditions a package decoder may be trained on (phase 10: the first two; phase 11: all four).
CONDITION_NAMES = ("duration", "mean_freq", "bandwidth", "loudness")

# The file that marks a QLVM model package's root: the SHA-256 and row counts of the
# spectrogram H5 of every session its corpus was built from.
PACKAGE_BASELINE_NAME = "SESSION_H5_BASELINE.tsv"

# Where a package keeps its files. v3 (restructured 2026-09-28) puts the baseline in
# ``<package>/corpus/`` and, inside a cell, the contract and bins in ``config/``, the
# per-call tables in ``inference/`` and each cluster level in ``inference/clusters_<level>/``;
# v2 / v2.1 keep all of these at the package or cell top level and the clusters in
# ``cluster/<level>/``. Both layouts are read; the v3 location is tried first.
PACKAGE_BASELINE_SUBDIRECTORIES = ("corpus", "")
CELL_FILE_SUBDIRECTORIES = ("config", "inference", "")


def cell_file(cell: pathlib.Path, name: str) -> pathlib.Path:
    """
    Description
    -----------
    Locates one file of a QLVM model package cell in either package layout: in the
    cell's ``config/`` or ``inference/`` subfolder (v3) or at the cell's top level
    (v2 / v2.1), in that order.

    Parameters
    ----------
    cell (pathlib.Path)
        The package cell directory.
    name (str)
        The file name, e.g. ``"training_contract.json"`` or ``"posterior_cache.npz"``.

    Returns
    -------
    path (pathlib.Path)
        The first existing candidate.

    Raises
    ------
    FileNotFoundError
        No candidate exists.
    """
    candidates = [cell / subdirectory / name if subdirectory else cell / name for subdirectory in CELL_FILE_SUBDIRECTORIES]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    error_message = f"{cell}: no {name} in {', '.join(str(candidate.parent) for candidate in candidates)}."
    raise FileNotFoundError(error_message)


def package_baseline_path(package_root: pathlib.Path) -> pathlib.Path | None:
    """
    Description
    -----------
    Locates a QLVM model package's ``SESSION_H5_BASELINE.tsv`` in either layout:
    ``corpus/`` (v3) or the package root (v2 / v2.1).

    Parameters
    ----------
    package_root (pathlib.Path)
        A candidate package root.

    Returns
    -------
    path (pathlib.Path | None)
        The baseline file, or ``None`` when the directory holds none.
    """
    for subdirectory in PACKAGE_BASELINE_SUBDIRECTORIES:
        candidate = package_root / subdirectory / PACKAGE_BASELINE_NAME if subdirectory else package_root / PACKAGE_BASELINE_NAME
        if candidate.is_file():
            return candidate
    return None


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
    with cell_file(cell, "training_contract.json").open() as contract_file:
        contract = json.load(contract_file)
    if contract["embedding_lattice_type"] != "fibonacci":
        error_message = (
            f"{cell}: training_contract.json embedding_lattice_type is {contract['embedding_lattice_type']!r}; "
            f"only 'fibonacci' package embeddings are supported."
        )
        raise ValueError(error_message)
    condition_bins = None
    if contract["c_dim"]:
        # Conditional cells: the frozen table new calls are decoded from. Phase 10
        # holds the corpus quantile bins of c and each bin's mean c; phase 11 holds
        # its training range and decode grid instead (see frozen_condition_values).
        with np.load(cell_file(cell, pathlib.Path(contract["condition_bins"]).name), allow_pickle=False) as bins:
            condition_bins = {key: bins[key] for key in bins.files}
        # Phase 10 contracts carry no decode rule; phase 11 contracts name one.
        condition = contract["condition"]
        decode = condition["decode"] if condition is not None and "decode" in condition else None
        phase11_bins = "decode_grid" in condition_bins
        if phase11_bins != (decode is not None) or (not phase11_bins and "bin_mean" not in condition_bins):
            error_message = (
                f"{cell}: training_contract.json condition.decode is {decode!r} but condition_bins.npz holds "
                f"{sorted(condition_bins)}; a phase 10 cell needs edges + bin_mean and no decode, a phase 11 cell "
                f"decode_grid + train_c_min/train_c_max and a decode of 'grid' or 'exact'."
            )
            raise ValueError(error_message)
    return {
        "params": load_decoder_params(str(cell / "checkpoint.tar")),
        "contract": contract,
        "lattice": gen_fib_basis(contract["embedding_fib_m"]),
        "fine_grid": np.load(cell_cluster_directory(cell, "fine") / "label_grid.npy", allow_pickle=False),
        "coarse_grid": np.load(cell_cluster_directory(cell, "coarse") / "label_grid.npy", allow_pickle=False),
        "condition_bins": condition_bins,
        "model_id": "/".join(cell.parts[-3:]),
    }


def compute_condition_values(
    condition: dict,
    durations: np.ndarray,
    masked_spectrograms: np.ndarray | None,
    raw_values: np.ndarray | None = None,
) -> np.ndarray:
    """
    Description
    -----------
    Each call's own conditioning value, as a conditional QLVM model package
    defines it in its training contract's ``condition`` block (float32, the
    package's ``condition_value`` / ``condition_from_raw``). ``"duration"``:
    ``(d - duration_min) / (duration_max - duration_min + epsilon)`` on the
    pre-resize duration in time bins (the corpus range fixes min and max).
    ``"mean_freq"``: the energy-weighted centroid of the frequency rows of the
    resized, SAM-masked spectrogram before any normalization, divided by the
    number of rows, with the energy clamped at ``epsilon`` (phase 11 trained on
    ``(mean_freq_hz - 30000) * 127 / (128 * 90000)``, which equals it to 1.5e-7).
    ``"bandwidth"`` (phase 11): ``clip(freq_bandwidth_hz / 90000, 0, 1)``.
    ``"loudness"`` (phase 11): ``clip((dB - lo) / (hi - lo), 0, 1)`` of the masked
    image-level loudness, with ``[lo, hi]`` the contract's ``db_range``. The two
    raw-unit maps run in float64 and round to float32, as the training sets
    stored them.

    Parameters
    ----------
    condition (dict)
        The contract's ``condition`` block.
    durations (np.ndarray)
        ``(N,)`` durations in time bins.
    masked_spectrograms (np.ndarray | None)
        ``(N, F, T)`` resized SAM-masked spectrograms; required for ``"mean_freq"``.
    raw_values (np.ndarray | None)
        ``(N,)`` raw values in Hz (``"bandwidth"``: ``freq_bandwidth_hz``) or dB
        (``"loudness"``: ``image_level_db``); required for those two. Defaults to None.

    Returns
    -------
    values (np.ndarray)
        ``(N,)`` float32 conditioning values.
    """
    if condition["name"] in ("bandwidth", "loudness"):
        if raw_values is None:
            error_message = f"compute_condition_values: {condition['name']} needs each call's raw value."
            raise ValueError(error_message)
        raw = np.asarray(raw_values, dtype=np.float64)
        if condition["name"] == "bandwidth":
            return np.clip(raw / 90000.0, 0.0, 1.0).astype(np.float32)
        low, high = (float(value) for value in condition["db_range"])
        return np.clip((raw - low) / (high - low), 0.0, 1.0).astype(np.float32)
    if condition["name"] == "duration":
        duration = np.asarray(durations).astype(np.float32)
        low, high = np.float32(condition["duration_min"]), np.float32(condition["duration_max"])
        return (duration - low) / (np.float32(high - low) + np.float32(condition["epsilon"]))
    if condition["name"] == "mean_freq":
        if masked_spectrograms is None:
            error_message = "compute_condition_values: mean_freq needs the SAM-masked spectrograms."
            raise ValueError(error_message)
        spectrograms = np.asarray(masked_spectrograms, dtype=np.float32)
        n_rows = spectrograms.shape[1]
        rows = np.arange(n_rows, dtype=np.float32)[None, :, None]
        energy = np.maximum(spectrograms.sum(axis=(1, 2)), np.float32(condition["epsilon"]))
        return ((spectrograms * rows).sum(axis=(1, 2)) / energy / np.float32(n_rows)).astype(np.float32)
    error_message = (
        f"compute_condition_values: unknown condition {condition['name']!r} "
        f"(expected {'|'.join(CONDITION_NAMES)})."
    )
    raise ValueError(error_message)


def frozen_condition_values(values: np.ndarray, condition_bins: dict, decode: str | None = None) -> np.ndarray:
    """
    Description
    -----------
    The conditioning value to decode new calls at (the package's
    ``frozen_condition_value``), by the rule the cell's embedding used.

    Phase 11 (``condition_bins`` has a ``decode_grid``): with ``decode``
    ``"exact"`` (duration, bandwidth) each call's own value clamped to the
    training range ``[train_c_min, train_c_max]``; with ``"grid"`` (mean
    frequency, loudness) the nearest point of ``decode_grid``, a value beyond
    the grid decoding at its end point. The package embedded every corpus call
    this way, so a new call is decoded exactly like a corpus call.

    Phase 10 (``bin_mean``): the corpus mean of the quantile bin each call's own
    value falls in, ``bin_mean[digitize(value, edges[1:-1])]``. That corpus
    embedding decoded batches at their mean value, which a new session cannot
    reproduce; the frozen bin mean comes closest. Values beyond the corpus range
    fall in the first or last bin.

    Parameters
    ----------
    values (np.ndarray)
        ``(N,)`` each call's own conditioning value.
    condition_bins (dict)
        The cell's ``condition_bins.npz``: ``edges`` and ``bin_mean`` (phase 10),
        or ``decode_grid``, ``train_c_min`` and ``train_c_max`` (phase 11).
    decode (str | None)
        The contract's ``condition.decode``, ``"grid"`` or ``"exact"``; required
        for phase 11 bins, unused for phase 10. Defaults to None.

    Returns
    -------
    frozen (np.ndarray)
        ``(N,)`` float32 values.
    """
    if "decode_grid" in condition_bins:
        own = np.asarray(values, dtype=np.float32).reshape(-1).astype(np.float64)
        if decode == "exact":
            low, high = float(condition_bins["train_c_min"]), float(condition_bins["train_c_max"])
            return np.clip(own, low, high).astype(np.float32)
        if decode == "grid":
            grid = np.asarray(condition_bins["decode_grid"], dtype=np.float64)
            midpoints = 0.5 * (grid[:-1] + grid[1:])
            return grid[np.searchsorted(midpoints, np.clip(own, grid[0], grid[-1]), side="left")].astype(np.float32)
        error_message = f"frozen_condition_values: phase 11 bins need decode 'grid' or 'exact', got {decode!r}."
        raise ValueError(error_message)
    edges = np.asarray(condition_bins["edges"])
    bin_mean = np.asarray(condition_bins["bin_mean"], dtype=np.float32)
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    return bin_mean[np.digitize(values, edges[1:-1], right=False)]


def _minmax_per_spectrogram(spectrograms: np.ndarray, epsilon: np.float32) -> np.ndarray:
    """
    Description
    -----------
    Rescales each spectrogram on its own to ``(x - min) / (max - min + epsilon)``,
    the min-max a QLVM model package applies to its decoder inputs. Zeros stay
    zeros when they are the spectrogram's minimum (a SAM-masked background).

    Parameters
    ----------
    spectrograms (np.ndarray)
        ``(N, F, T)`` float32 spectrograms.
    epsilon (np.float32)
        Added to each spectrogram's range, so a constant spectrogram maps to zeros.

    Returns
    -------
    rescaled (np.ndarray)
        ``(N, F, T)`` float32 spectrograms in ``[0, 1)``.
    """
    low = spectrograms.min(axis=(1, 2), keepdims=True)
    high = spectrograms.max(axis=(1, 2), keepdims=True)
    return (spectrograms - low) / ((high - low) + epsilon)


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
    inputs = _minmax_per_spectrogram(inputs, epsilon)
    if contract["floor"] is not None:
        floor = np.float32(contract["floor"])
        inputs = np.clip((inputs - floor) / np.float32(1.0 - floor), np.float32(0.0), np.float32(1.0))
        inputs = _minmax_per_spectrogram(inputs, epsilon)
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
    module can run: no conditioning input (``c_dim`` 0) or one conditioning value
    of a model package's ``"duration"`` / ``"mean_freq"`` / ``"bandwidth"`` /
    ``"loudness"`` condition (a phase 11 ``decode`` of ``"grid"`` or ``"exact"``), an
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
    # Unconditional decoders (every train-qlvm contract) record "condition": null.
    condition = contract["condition"]
    if contract["c_dim"] != 0 and not (
        contract["c_dim"] == 1 and condition is not None and condition["name"] in CONDITION_NAMES
    ):
        mismatches.append(
            f"c_dim: this module embeds unconditional decoders and decoders conditioned on one of "
            f"{', '.join(CONDITION_NAMES)}, trained c_dim {contract['c_dim']!r} with condition {condition!r}"
        )
    if condition is not None and "decode" in condition and condition["decode"] not in ("grid", "exact"):
        mismatches.append(f"condition.decode: expected 'grid' or 'exact', trained {condition['decode']!r}")
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


def label_grid_lookup(coords: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """
    Description
    -----------
    Looks up each torus coordinate's cluster label in one periodic label grid:
    ``label = grid[floor(y * res) mod res, floor(x * res) mod res]``, with ``res``
    the grid's resolution (``grid.shape[0]``). This is the pixel rule of the QLVM
    model packages: on the posterior-mean coordinates of the v3 production cells
    it reproduces the package's ``inference/clusters_<level>/cluster_labels.csv``
    on every one of the 445,742 corpus calls but one (a ``qlvm_dur`` call whose
    coordinate lies within float32 rounding of a pixel edge).
    The ``mod`` wraps a coordinate of exactly ``1.0`` (or one whose product with
    ``res`` rounds up to ``res``) to pixel 0, the pixel it shares on the torus,
    and a negative coordinate to its periodic image, instead of clipping either
    to an edge pixel.

    Parameters
    ----------
    coords (np.ndarray)
        Torus coordinates, shape ``(N, 2)`` ordered ``(x, y)``; finite values
        (nominally in ``[0, 1)``).
    grid (np.ndarray)
        Periodic label grid, shape ``(res, res)``, indexed ``[y, x]``.

    Returns
    -------
    labels (np.ndarray)
        The label of each coordinate's pixel, shape ``(N,)``, in the grid's dtype.
    """
    resolution = grid.shape[0]
    coords = np.asarray(coords, dtype=np.float64).reshape(-1, 2)
    pixel_x = np.floor(coords[:, 0] * resolution).astype(np.int64) % resolution
    pixel_y = np.floor(coords[:, 1] * resolution).astype(np.int64) % resolution
    return grid[pixel_y, pixel_x]


def labels_for_coords(
    coords: np.ndarray,
    fine_grid: np.ndarray,
    coarse_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Description
    -----------
    Looks up each torus coordinate's cluster label in the FINE and COARSE
    reference watershed grids with :func:`label_grid_lookup`
    (``label = grid[floor(y * res) mod res, floor(x * res) mod res]``, per each
    grid's own resolution). Each grid is the torus-periodic ``ws_labels_periodic``
    field of its reference ``arrays.npz``, or a model package cell's
    ``label_grid.npy`` (periodic = correct for the native-torus QLVM
    coordinates, which wrap at the seam). The fine grid yields the per-USV
    ``qlvm_category`` (e.g. 15 clusters in the v3 regular model); the coarse grid
    yields the broader ``qlvm_supercategory`` (e.g. 9 clusters).

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
    return label_grid_lookup(coords, fine_grid), label_grid_lookup(coords, coarse_grid)


def validate_model_cells(model_cells: Iterable[tuple[str, str]]) -> dict[str, str]:
    """
    Description
    -----------
    Checks the ``infer_qlvm_latents.model_cells`` mapping (column prefix -> model
    package cell directory) and returns it as a dict in the given order. Each
    prefix ``P`` names the two float columns ``P1`` / ``P2`` a run writes, so it
    must be a non-empty Python identifier (letters, digits and underscores, not
    starting with a digit), may be listed only once, and ``P1`` / ``P2`` must not
    be any other column of the USV summary (``USV_SUMMARY_COLUMN_ORDER``; the
    torus-coordinate and label columns of the production prefixes of
    ``os_utils.QLVM_PRODUCTION_MODEL_CELLS`` -- ``qlvm1`` / ``qlvm2``,
    ``qlvm_category`` / ``qlvm_supercategory``, ``qlvm_dur1`` / ``qlvm_dur2``,
    ``qlvm_dur_category``, ... -- are allowed, since writing them is what a run is
    for; see :func:`model_cell_reserved_columns`). Each cell directory must be a
    non-empty string. Every problem is collected and raised together. The label
    columns a prefix writes are checked separately
    (:func:`model_cell_label_columns`).

    Parameters
    ----------
    model_cells (Iterable[tuple[str, str]])
        ``(prefix, model_cell_directory)`` pairs, e.g. ``cfg['model_cells'].items()``
        or the pairs of repeated ``--model-cell`` CLI options (which, unlike a JSON
        object, can repeat a prefix).

    Returns
    -------
    model_cells (dict[str, str])
        The validated mapping, prefix -> cell directory, in the given order.
    """
    pairs = list(model_cells)
    problems = []
    prefixes = [prefix for prefix, _cell in pairs]
    duplicates = sorted({prefix for prefix in prefixes if isinstance(prefix, str) and prefixes.count(prefix) > 1})
    if duplicates:
        problems.append(f"prefixes listed more than once: {duplicates}")
    reserved = model_cell_reserved_columns()
    for prefix, cell_directory in pairs:
        if not isinstance(prefix, str) or not prefix.isidentifier():
            problems.append(f"prefix {prefix!r} is not a non-empty identifier (letters, digits, underscores)")
            continue
        clashing = [column for column in (f"{prefix}1", f"{prefix}2") if column in reserved]
        if clashing:
            problems.append(f"prefix {prefix!r} would overwrite the summary column(s) {clashing}")
        if not isinstance(cell_directory, str) or not cell_directory:
            problems.append(f"prefix {prefix!r} has no model cell directory ({cell_directory!r})")
    if problems:
        error_message = "infer_qlvm_latents.model_cells is invalid:\n  " + "\n  ".join(problems)
        raise ValueError(error_message)
    return dict(pairs)


def model_cell_label_column(prefix: str, level: str) -> str:
    """
    Description
    -----------
    Names the cluster-label column a ``model_cells`` prefix writes for one label
    level. The regular model's prefix ``"qlvm"`` keeps the historical names --
    ``"fine"`` -> ``qlvm_category``, ``"coarse"`` -> ``qlvm_supercategory`` --
    and every other prefix ``P`` gets ``P_category`` (fine) and
    ``P_supercategory`` (coarse), e.g. ``qlvm_dur_category``.

    Parameters
    ----------
    prefix (str)
        The column prefix (a key of ``infer_qlvm_latents.model_cells``).
    level (str)
        ``"fine"`` or ``"coarse"`` (``LABEL_LEVELS``).

    Returns
    -------
    column (str)
        The label column name.
    """
    suffix = LABEL_LEVEL_SUFFIXES[level]
    if prefix == REGULAR_MODEL_PREFIX:
        return f"qlvm_{suffix}"
    return f"{prefix}_{suffix}"


def default_model_cell_label_levels(prefix: str) -> list[str]:
    """
    Description
    -----------
    The label levels a ``model_cells`` prefix writes when
    ``infer_qlvm_latents.model_cell_label_levels`` does not list it: both levels
    (``["fine", "coarse"]``, i.e. ``qlvm_category`` and ``qlvm_supercategory``)
    for the regular model's prefix ``"qlvm"``, the fine level only
    (``["fine"]``, i.e. ``P_category``) for every other prefix ``P``.

    Parameters
    ----------
    prefix (str)
        The column prefix.

    Returns
    -------
    levels (list[str])
        The default levels, in column order.
    """
    if prefix == REGULAR_MODEL_PREFIX:
        return list(LABEL_LEVELS)
    return ["fine"]


def model_cell_reserved_columns() -> set[str]:
    """
    Description
    -----------
    The USV summary columns a ``model_cells`` run may never write: every column
    of ``USV_SUMMARY_COLUMN_ORDER`` except the torus-coordinate columns
    (``P1`` / ``P2``) and the label columns of both levels
    (:func:`model_cell_label_column`) of the production prefixes of
    ``os_utils.QLVM_PRODUCTION_MODEL_CELLS`` -- writing those is what a run is for.
    Every other summary column (DAS event, acoustic features, ...) must stay
    untouched.

    Parameters
    ----------

    Returns
    -------
    reserved (set[str])
        The reserved column names.
    """
    writable = {f"{prefix}{axis}" for prefix in QLVM_PRODUCTION_MODEL_CELLS for axis in (1, 2)}
    writable |= {
        model_cell_label_column(prefix, level) for prefix in QLVM_PRODUCTION_MODEL_CELLS for level in LABEL_LEVELS
    }
    return set(USV_SUMMARY_COLUMN_ORDER) - writable


def model_cell_label_columns(model_cells: dict[str, str], label_levels: object) -> dict[str, dict[str, str]]:
    """
    Description
    -----------
    Resolves which cluster-label columns a ``model_cells`` run writes, per prefix,
    from the ``infer_qlvm_latents.model_cell_label_levels`` setting (prefix ->
    list of levels among ``"fine"`` and ``"coarse"``). A prefix the setting does
    not list takes :func:`default_model_cell_label_levels` (so the shipped ``{}``
    gives ``qlvm_category`` + ``qlvm_supercategory`` for ``"qlvm"`` and
    ``P_category`` for every other prefix ``P``); an empty list writes no label
    column for that prefix. Column names follow :func:`model_cell_label_column`.

    The setting is validated and every problem is raised together: it must be an
    object; each key must be a prefix of ``model_cells``; each value a list of
    distinct levels from ``LABEL_LEVELS``; and no label column may be a reserved
    summary column (:func:`model_cell_reserved_columns`) or a column another
    listed prefix or level also writes.

    Parameters
    ----------
    model_cells (dict[str, str])
        The validated ``model_cells`` mapping (:func:`validate_model_cells`).
    label_levels (object)
        The ``model_cell_label_levels`` setting (expected: dict of prefix -> list
        of levels).

    Returns
    -------
    label_columns (dict[str, dict[str, str]])
        Prefix -> (level -> label column), for every prefix of ``model_cells`` in
        its order, levels in the order ``LABEL_LEVELS`` lists them.
    """
    if not isinstance(label_levels, dict):
        error_message = (
            f"infer_qlvm_latents.model_cell_label_levels must be an object of column prefix -> list of label levels "
            f"(among {list(LABEL_LEVELS)}), got {type(label_levels).__name__}."
        )
        raise ValueError(error_message)
    problems = []
    unknown = [prefix for prefix in label_levels if prefix not in model_cells]
    if unknown:
        problems.append(f"prefixes not in model_cells: {unknown} (model_cells lists {list(model_cells)})")
    label_columns = {}
    for prefix in model_cells:
        levels = label_levels[prefix] if prefix in label_levels else default_model_cell_label_levels(prefix)
        if not isinstance(levels, (list, tuple)):
            problems.append(f"prefix {prefix!r}: levels must be a list, got {type(levels).__name__}")
            continue
        invalid = [level for level in levels if level not in LABEL_LEVELS]
        if invalid:
            problems.append(f"prefix {prefix!r}: invalid level(s) {invalid} (allowed: {list(LABEL_LEVELS)})")
            continue
        repeated = sorted({level for level in levels if list(levels).count(level) > 1})
        if repeated:
            problems.append(f"prefix {prefix!r}: level(s) listed more than once: {repeated}")
            continue
        label_columns[prefix] = {
            level: model_cell_label_column(prefix, level) for level in LABEL_LEVELS if level in levels
        }
    reserved = model_cell_reserved_columns()
    for prefix, columns in label_columns.items():
        clashing = [column for column in columns.values() if column in reserved]
        if clashing:
            problems.append(f"prefix {prefix!r}: label column(s) {clashing} would overwrite summary column(s)")
    written = collections.Counter(f"{prefix}{axis}" for prefix in model_cells for axis in (1, 2))
    written.update(column for columns in label_columns.values() for column in columns.values())
    duplicated = sorted(column for column, count in written.items() if count > 1)
    if duplicated:
        problems.append(f"columns written by more than one prefix or level: {duplicated}")
    if problems:
        error_message = "infer_qlvm_latents.model_cell_label_levels is invalid:\n  " + "\n  ".join(problems)
        raise ValueError(error_message)
    return label_columns


def parse_model_cell_label_levels(pairs: Iterable[tuple[str, str]]) -> dict[str, list[str]]:
    """
    Description
    -----------
    Turns the repeated ``--model-cell-labels PREFIX LEVELS`` CLI options into the
    ``infer_qlvm_latents.model_cell_label_levels`` object (prefix -> list of
    levels). ``LEVELS`` is a comma-separated list (``"fine"``, ``"coarse"``,
    ``"fine,coarse"``; blanks around commas are ignored, and an empty string
    means no label column). The levels themselves are validated when the run
    resolves them against ``model_cells`` (:func:`model_cell_label_columns`).

    Parameters
    ----------
    pairs (Iterable[tuple[str, str]])
        ``(prefix, levels)`` pairs as click passes them.

    Returns
    -------
    label_levels (dict[str, list[str]])
        Prefix -> levels, in the given order.

    Raises
    ------
    ValueError
        A prefix is given more than once.
    """
    label_levels = {}
    for prefix, levels_text in pairs:
        if prefix in label_levels:
            error_message = f"--model-cell-labels lists prefix {prefix!r} more than once."
            raise ValueError(error_message)
        label_levels[prefix] = [level.strip() for level in levels_text.split(",") if level.strip()]
    return label_levels


def find_package_root(model_cell_directory: str) -> pathlib.Path | None:
    """
    Description
    -----------
    Finds the root of the QLVM model package a cell belongs to: the nearest of
    the cell directory and its parents that holds ``SESSION_H5_BASELINE.tsv``
    (the per-session SHA-256 baseline of the spectrogram H5s the package's corpus
    was built from; ``<package>/<phase>/<cell>`` has it two levels up, in
    ``corpus/`` for v3 and at the top level for v2 / v2.1).

    Parameters
    ----------
    model_cell_directory (str)
        Path to the package cell.

    Returns
    -------
    package_root (pathlib.Path | None)
        The package root, or ``None`` when no directory on the way up holds the
        baseline (the package route cannot then be decided).
    """
    cell = pathlib.Path(configure_path(model_cell_directory)).resolve()
    for directory in (cell, *cell.parents):
        if package_baseline_path(directory) is not None:
            return directory
    return None


def load_package_baseline(package_root: pathlib.Path) -> dict[str, str]:
    """
    Description
    -----------
    Reads a QLVM model package's ``SESSION_H5_BASELINE.tsv`` (in ``corpus/`` for v3,
    at the package root for v2 / v2.1; tab-separated,
    columns ``session``, ``h5_rows``, ``corpus_rows``, ``bytes``, ``sha256``,
    ``path``): the SHA-256 of each corpus session's spectrogram H5 at the time the
    package's per-call rows were checked against it.

    Parameters
    ----------
    package_root (pathlib.Path)
        The package root (see :func:`find_package_root`).

    Returns
    -------
    baseline (dict[str, str])
        Session id -> lowercase hexadecimal SHA-256 of its spectrogram H5.
    """
    baseline_path = package_baseline_path(package_root)
    if baseline_path is None:
        error_message = f"{package_root}: no {PACKAGE_BASELINE_NAME} in it or in its corpus/ folder."
        raise FileNotFoundError(error_message)
    table = pls.read_csv(baseline_path, separator="\t", infer_schema_length=0)
    missing = [column for column in ("session", "sha256") if column not in table.columns]
    if missing:
        error_message = f"{baseline_path} has no {missing} column(s); its columns are {table.columns}."
        raise ValueError(error_message)
    return dict(zip(table["session"].to_list(), [digest.lower() for digest in table["sha256"].to_list()], strict=True))


def load_package_session_rows(model_cell_directory: str, session_id: str) -> dict[str, np.ndarray]:
    """
    Description
    -----------
    The rows of one session in a QLVM model package cell's corpus embedding:
    ``recon_mse_breakdown.npz`` names every corpus call by ``spec_id``
    (``<session>_<row of the session's spectrogram H5>``) and records the
    ``durations`` and ``mask_counts`` it was built with, and ``posterior_cache.npz``
    holds, in the same row order, its posterior-mean torus embedding
    ``torus_weighted`` (``(N, 4)`` ``[cos, sin]`` basis), which
    :func:`qlvm_model.torus_basis_reverse` turns into coordinates in ``[0, 1)``.

    Parameters
    ----------
    model_cell_directory (str)
        Path to the package cell.
    session_id (str)
        The session whose rows to take.

    Returns
    -------
    rows (dict[str, np.ndarray])
        ``row`` (``(n,)`` int64 spectrogram-H5 rows, ascending), ``durations`` and
        ``mask_counts`` (``(n,)`` float64, as the package recorded them) and
        ``coords`` (``(n, 2)`` float64 torus coordinates ``(x, y)``); ``n`` is 0
        when the cell holds no call of the session.
    """
    cell = pathlib.Path(configure_path(model_cell_directory))
    with np.load(cell_file(cell, "recon_mse_breakdown.npz"), allow_pickle=False) as breakdown:
        spec_id = breakdown["spec_id"].astype(str)
        durations = breakdown["durations"]
        mask_counts = breakdown["mask_counts"]
    with np.load(cell_file(cell, "posterior_cache.npz"), allow_pickle=False) as cache:
        torus_weighted = cache["torus_weighted"]
    if torus_weighted.shape[0] != spec_id.shape[0]:
        error_message = (
            f"{cell}: posterior_cache.npz holds {torus_weighted.shape[0]} rows but recon_mse_breakdown.npz "
            f"{spec_id.shape[0]}; they must describe the same corpus rows in the same order."
        )
        raise ValueError(error_message)
    # A spec_id is "<session>_<row>"; the digit check keeps a longer id that merely
    # starts with this session's out.
    prefix = f"{session_id}_"
    selected = np.flatnonzero(np.char.startswith(spec_id, prefix))
    row_text = [identifier[len(prefix):] for identifier in spec_id[selected]]
    keep = np.array([text.isdigit() for text in row_text], dtype=bool)
    selected = selected[keep]
    rows = np.array([int(text) for text, kept in zip(row_text, keep, strict=True) if kept], dtype=np.int64)
    order = np.argsort(rows, kind="stable")
    selected, rows = selected[order], rows[order]
    coords = np.asarray(torus_basis_reverse(jnp.asarray(torus_weighted[selected])), dtype=np.float64)
    return {
        "row": rows,
        "durations": durations[selected].astype(np.float64),
        "mask_counts": mask_counts[selected].astype(np.float64),
        "coords": coords.reshape(-1, 2),
    }


def session_h5_call_table(h5_loc: pathlib.Path, session_id: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Description
    -----------
    Each spectrogram-H5 row's duration and number of SAM mask instances, the two
    per-call quantities a QLVM model package records for its corpus rows
    (``recon_mse_breakdown.npz`` ``durations`` / ``mask_counts``). Durations are
    ``spectrogram/<session>/durations``; the mask count of a row is how many
    entries of ``mask/<session>/spectrogram_index`` name it (0 for every row when
    the H5 has no mask group). Only these two small datasets are read.

    Parameters
    ----------
    h5_loc (pathlib.Path)
        The session's spectrogram H5.
    session_id (str)
        The session id naming its groups.

    Returns
    -------
    durations (np.ndarray)
        ``(n_rows,)`` durations in time bins.
    mask_counts (np.ndarray)
        ``(n_rows,)`` int64 mask-instance counts.
    """
    with h5py.File(h5_loc, "r") as h5_file:
        durations = h5_file[f"spectrogram/{session_id}/durations"][:]
        mask_counts = np.zeros(durations.shape[0], dtype=np.int64)
        if f"mask/{session_id}" in h5_file:
            spectrogram_index = h5_file[f"mask/{session_id}/spectrogram_index"][:].astype(np.int64)
            mask_counts = np.bincount(spectrogram_index, minlength=durations.shape[0])[: durations.shape[0]]
    return durations, mask_counts


def package_route_verdict(
    session_id: str,
    baseline: dict[str, str],
    h5_sha256: Callable[[], str],
    n_summary_rows: int,
    h5_durations: np.ndarray,
    h5_mask_counts: np.ndarray,
    package_rows: Callable[[], dict[str, np.ndarray]],
) -> tuple[bool, str, dict[str, np.ndarray] | None]:
    """
    Description
    -----------
    Decides whether a session's coordinates under one model can be taken from the
    model package instead of being inferred. A package row names its call only by
    position (``spec_id`` = ``<session>_<H5 row>``), so the package's values are
    used only when all of these hold, checked in this order:

    * the session is in the package's ``SESSION_H5_BASELINE.tsv``;
    * the SHA-256 of the session's spectrogram H5 equals the baseline's (the file
      was not rebuilt since, so its rows are the rows the package embedded);
    * ``usv_summary.csv`` has as many rows as the H5 (summary and H5 rows are 1:1,
      which the positional join relies on);
    * the cell holds rows of the session, all inside the H5, and each row's
      ``durations`` and ``mask_counts`` in the package equal the H5's.

    The two costly inputs are callables, so the H5 is hashed and the cell's rows
    are read only when the checks before them passed.

    Parameters
    ----------
    session_id (str)
        The session.
    baseline (dict[str, str])
        The package baseline (:func:`load_package_baseline`).
    h5_sha256 (Callable[[], str])
        Returns the SHA-256 of the session's spectrogram H5 (cached by the caller,
        so the file is hashed once per session, not once per model).
    n_summary_rows (int)
        Rows of ``usv_summary.csv``.
    h5_durations (np.ndarray)
        ``(n_rows,)`` H5 durations (:func:`session_h5_call_table`).
    h5_mask_counts (np.ndarray)
        ``(n_rows,)`` H5 mask-instance counts (:func:`session_h5_call_table`).
    package_rows (Callable[[], dict[str, np.ndarray]])
        Returns the cell's rows of the session (:func:`load_package_session_rows`).

    Returns
    -------
    use_package (bool)
        True to take the package's coordinates.
    reason (str)
        The route and why, for the log (e.g. ``"package values (sha256 + 907 rows
        verified)"`` or ``"inference (session not in the package corpus)"``).
    rows (dict[str, np.ndarray] | None)
        The cell's rows of the session when they were read (always when
        ``use_package`` is True), else None.
    """
    if session_id not in baseline:
        return False, "inference (session not in the package corpus)", None
    if h5_sha256() != baseline[session_id]:
        return False, "inference (spectrogram H5 changed since the package: sha256 mismatch)", None
    n_h5_rows = h5_durations.shape[0]
    if n_summary_rows != n_h5_rows:
        return False, f"inference (usv_summary.csv has {n_summary_rows} rows, the spectrogram H5 {n_h5_rows})", None
    rows = package_rows()
    n_rows = rows["row"].shape[0]
    if n_rows == 0:
        return False, "inference (the package cell holds no rows of this session)", rows
    if rows["row"][-1] >= n_h5_rows:
        return False, f"inference (the package names H5 row {int(rows['row'][-1])}, the H5 has {n_h5_rows} rows)", rows
    n_disagree = int(np.count_nonzero(
        (rows["durations"] != h5_durations[rows["row"]].astype(np.float64))
        | (rows["mask_counts"] != h5_mask_counts[rows["row"]].astype(np.float64))
    ))
    if n_disagree:
        return False, (
            f"inference (the package's durations or mask counts disagree with the spectrogram H5 "
            f"on {n_disagree} of {n_rows} rows)"
        ), rows
    return True, f"package values (sha256 + {n_rows} rows verified)", rows


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
        ``length_threshold`` does (``null`` embeds every positive duration). When
        the contract's training set kept only calls with a SAM mask
        (``require_mask``) or the decoder conditions on mean frequency or
        loudness, USVs without a mask instance are skipped and get nulls too.
        When the decoder needs SAM masks at all (those cases, or ``masking_type``
        ``"sam"``), a session H5 without a ``mask/<session>`` group raises. A
        conditional package decoder is decoded at :func:`frozen_condition_values`
        of each call's own value (:func:`compute_condition_values`); a bandwidth
        condition reads it from the summary's ``freq_bandwidth_hz`` (missing
        column raises), a loudness condition measures it from the session audio
        (:func:`compute_usv_loudness.session_image_level_db`), and calls with no
        value get nulls. The summary is rewritten atomically.

        With a non-empty ``model_cells`` setting (column prefix -> model package
        cell) the session is instead placed on the torus of every listed cell, and
        each prefix ``P`` gets the float columns ``P1`` / ``P2`` plus the cluster
        labels of its ``model_cell_label_levels`` (by default ``qlvm_category`` and
        ``qlvm_supercategory`` for ``"qlvm"``, ``P_category`` for every other
        prefix); no model column is written, and a stale ``qlvm_model`` and the
        earlier coordinate and label columns of the listed prefixes are removed
        first (see :meth:`_merge_model_cells`). ``model_cells`` and
        ``model_cell_directory`` cannot both be set.

        Parameters
        ----------

        Returns
        -------
        Updated ``*_usv_summary.csv`` with the ``qlvm_*`` columns (or the
        ``P1`` / ``P2`` and label columns of every ``model_cells`` prefix).
        """
        self.message_output(
            f"QLVM latent inference started at: {datetime.now().hour:02d}:{datetime.now().minute:02d}:{datetime.now().second:02d}."
        )
        smart_wait(app_context_bool=self.app_context_bool, seconds=1)

        derive_spectrogram_model_paths(self.input_parameter_dict)
        cfg = self.input_parameter_dict['infer_qlvm_latents']
        if cfg['model_cells']:
            self._merge_model_cells(cfg)
        else:
            self._merge_single_model(cfg)

        self.message_output(
            f"QLVM latent inference ended at: {datetime.now().hour:02d}:{datetime.now().minute:02d}:{datetime.now().second:02d}."
        )

    def _locate_session_files(self) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path, pls.DataFrame]:
        """
        Description
        -----------
        Finds the session's spectrogram H5 and its ``*_usv_summary.csv`` and reads
        the summary.

        Parameters
        ----------

        Returns
        -------
        root (pathlib.Path)
            The session root directory (its name is the session id).
        h5_loc (pathlib.Path)
            ``audio/spectrograms/<session>_spectrograms.h5``.
        usv_summary_loc (pathlib.Path)
            The USV summary CSV.
        usv_df (pls.DataFrame)
            The summary, ``usv_id`` read as a string.
        """
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
        usv_summary_loc = first_match_or_raise(
            root=root / "audio",
            pattern="*_usv_summary.csv",
            recursive=True,
            label="USV summary CSV",
        )
        usv_df = pls.read_csv(source=str(usv_summary_loc), schema_overrides={"usv_id": pls.String})
        return root, h5_loc, usv_summary_loc, usv_df

    def _merge_single_model(self, cfg: dict) -> None:
        """
        Description
        -----------
        The single-model run of :meth:`infer_and_merge`: embeds the session with
        the model package cell of ``model_cell_directory``, or with the decoder
        weights of ``weights_npz_path`` and the lattice and reference-array
        settings, and merges ``qlvm1``, ``qlvm2``, ``qlvm_category``,
        ``qlvm_supercategory`` and ``qlvm_model`` into the summary.

        Parameters
        ----------
        cfg (dict)
            The ``infer_qlvm_latents`` settings block.

        Returns
        -------
        None
        """
        if cfg['model_cell_directory']:
            # A QLVM model package cell brings its own weights, contract, embedding
            # lattice and label grids; the settings' paths and lattice keys are unused.
            model = load_model_cell(cfg['model_cell_directory'])
            self.message_output(
                f"Embedding with model package cell {model['model_id']} ({decoder_head(model['params'])} head, "
                f"{model['lattice'].shape[0]}-point Fibonacci lattice)."
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
            model = {
                "params": params,
                "contract": contract,
                "lattice": lattice,
                "fine_grid": fine_grid,
                "coarse_grid": coarse_grid,
                "condition_bins": None,
                "model_id": cfg['weights_npz_path'],
            }

        # The decoder only knows calls shaped like its training set: check the
        # preprocessing settings against its contract and embed only calls inside the
        # set's duration window. Weights with no contract fall back to the settings.
        if model['contract'] is None:
            length_threshold = cfg['length_threshold']
            self.message_output(
                "No training contract beside the decoder weights; the infer_qlvm_latents settings are used as given "
                f"(length_threshold={length_threshold})."
            )
        else:
            length_threshold = enforce_training_contract(model['contract'], cfg, model['params'])

        root, h5_loc, usv_summary_loc, usv_df = self._locate_session_files()
        usv_indices, coords = self._embed_session(
            model, cfg, length_threshold, root, h5_loc, usv_df, usv_summary_loc, "qlvm_*"
        )
        category, supercategory = labels_for_coords(coords, model['fine_grid'], model['coarse_grid'])

        qlvm_df = pls.DataFrame({
            "_usv_row": usv_indices,
            "qlvm1": coords[:, 0].astype(np.float64),
            "qlvm2": coords[:, 1].astype(np.float64),
            "qlvm_category": category.astype(np.int64),
            "qlvm_supercategory": supercategory.astype(np.int64),
            # Which model's torus and clusters these are: labels from different models
            # share column names but not meanings, so an analysis can check this first.
            "qlvm_model": [model['model_id']] * len(usv_indices),
        }, schema_overrides={"qlvm_model": pls.String})

        usv_df = usv_df.drop([c for c in QLVM_COLUMNS if c in usv_df.columns])
        usv_df = usv_df.with_row_index(name="_usv_row")
        merged = order_usv_summary_columns(usv_df.join(qlvm_df, on="_usv_row", how="left").drop("_usv_row"))
        # usv_summary.csv holds every other per-USV column too: publish atomically so
        # a failed write leaves the previous file intact instead of a truncated one.
        with atomic_output_path(usv_summary_loc) as tmp_summary_path:
            merged.write_csv(file=str(tmp_summary_path))

        self.message_output(
            f"Merged QLVM latents/categories for {len(usv_indices)} USVs into {usv_summary_loc.name}."
        )

    def _merge_model_cells(self, cfg: dict) -> None:
        """
        Description
        -----------
        The multi-model run of :meth:`infer_and_merge`: places the session on the
        torus of every cell of ``model_cells`` and merges, per prefix ``P``, the
        float columns ``P1`` / ``P2`` and the integer cluster-label columns of the
        prefix's label levels into the summary (nulls where a call was not placed).
        The levels come from ``model_cell_label_levels``
        (:func:`model_cell_label_columns`; by default ``qlvm_category`` (fine) and
        ``qlvm_supercategory`` (coarse) for prefix ``"qlvm"``, ``P_category``
        (fine) for every other prefix). Every label is the cell's
        ``label_grid.npy`` of that level at the pixel of the call's written
        coordinates (:func:`label_grid_lookup`): ``1..k``, 1 the largest cluster,
        as the package numbers them. The settings are validated, and every cell
        is loaded once and checked against them
        (:func:`enforce_training_contract`), before anything is embedded.

        For each cell the coordinates come from one of two routes, and the log
        names the route and why. The package route takes the cell's own corpus
        embedding (``posterior_cache.npz`` joined on ``spec_id``,
        :func:`load_package_session_rows`) when ``prefer_package_values`` is true,
        a ``SESSION_H5_BASELINE.tsv`` sits in or above the cell
        (:func:`find_package_root`), and :func:`package_route_verdict` finds the
        session in that baseline with an unchanged spectrogram H5 (SHA-256, hashed
        once per session), a summary as long as the H5, and the package's
        durations and mask counts equal to the H5's on every one of its rows. Any
        other case embeds the session with the cell (:meth:`_embed_session`),
        exactly as the single-model run does. Both routes label by the same grid
        lookup.

        No ``qlvm_model`` column is written; a stale one, and every earlier
        ``P1`` / ``P2`` and label column of the listed prefixes (both levels,
        whichever this run writes), are dropped before the merge. The summary is
        rewritten atomically.

        Parameters
        ----------
        cfg (dict)
            The ``infer_qlvm_latents`` settings block.

        Returns
        -------
        None
        """
        if cfg['model_cell_directory']:
            error_message = (
                "infer_qlvm_latents: model_cells and model_cell_directory are both set. model_cells embeds the "
                "session with every listed cell and writes <prefix>1/<prefix>2 and label columns; model_cell_directory "
                "embeds it with one cell and writes the qlvm_* columns. Set one of them and leave the other empty."
            )
            raise ValueError(error_message)
        if not isinstance(cfg['model_cells'], dict):
            error_message = (
                f"infer_qlvm_latents.model_cells must be an object of column prefix -> model cell directory, "
                f"got {type(cfg['model_cells']).__name__}."
            )
            raise ValueError(error_message)
        model_cells = validate_model_cells(cfg['model_cells'].items())
        label_columns = model_cell_label_columns(model_cells, cfg['model_cell_label_levels'])

        models = {}
        for prefix, cell_directory in model_cells.items():
            model = load_model_cell(cell_directory)
            model['length_threshold'] = enforce_training_contract(model['contract'], cfg, model['params'])
            models[prefix] = model
            self.message_output(
                f"{prefix}1/{prefix}2: model package cell {model['model_id']} ({decoder_head(model['params'])} head, "
                f"{model['lattice'].shape[0]}-point Fibonacci lattice)."
            )

        root, h5_loc, usv_summary_loc, usv_df = self._locate_session_files()
        h5_durations, h5_mask_counts = session_h5_call_table(h5_loc, root.name)
        # Hashed on first use and reused for every cell: one read of the H5 per session.
        h5_sha256 = functools.cache(functools.partial(file_sha256, h5_loc))
        baselines = {}
        coordinate_frames = []
        for prefix, model in models.items():
            use_package, rows = False, None
            if not cfg['prefer_package_values']:
                reason = "inference (prefer_package_values is false)"
            else:
                package_root = find_package_root(model_cells[prefix])
                if package_root is None:
                    reason = (
                        f"inference (no {PACKAGE_BASELINE_NAME} in or above the cell, so the package route "
                        f"cannot be decided)"
                    )
                else:
                    if package_root not in baselines:
                        baselines[package_root] = load_package_baseline(package_root)
                    use_package, reason, rows = package_route_verdict(
                        session_id=root.name,
                        baseline=baselines[package_root],
                        h5_sha256=h5_sha256,
                        n_summary_rows=usv_df.height,
                        h5_durations=h5_durations,
                        h5_mask_counts=h5_mask_counts,
                        package_rows=functools.partial(load_package_session_rows, model_cells[prefix], root.name),
                    )
            self.message_output(f"{prefix} ({model['model_id']}): {reason}.")
            if use_package:
                usv_indices, coords = rows['row'], rows['coords']
            else:
                usv_indices, coords = self._embed_session(
                    model, cfg, model['length_threshold'], root, h5_loc, usv_df, usv_summary_loc,
                    f"{prefix}1/{prefix}2",
                )
            self.message_output(f"{prefix}1/{prefix}2: {len(usv_indices)} of {usv_df.height} USVs placed.")
            # The labels are read off the coordinates as written (float64), with the
            # package's pixel rule, so a summary's labels can always be re-derived from
            # its own P1/P2 and the cell's label grids, whichever route placed the call.
            placed_coords = np.asarray(coords, dtype=np.float64).reshape(-1, 2)
            frame_columns = {
                "_usv_row": np.asarray(usv_indices).astype(np.uint32),
                f"{prefix}1": placed_coords[:, 0],
                f"{prefix}2": placed_coords[:, 1],
            }
            for level, column in label_columns[prefix].items():
                frame_columns[column] = label_grid_lookup(placed_coords, model[f"{level}_grid"]).astype(np.int64)
            coordinate_frames.append(pls.DataFrame(frame_columns))

        # Provenance of these models is kept outside the summary, so a qlvm_model left
        # by a single-model run goes, together with this run's own earlier coordinate
        # and label columns (both levels, whichever this run writes) of every listed prefix.
        stale = ["qlvm_model"]
        stale += [f"{prefix}{axis}" for prefix in models for axis in (1, 2)]
        stale += [model_cell_label_column(prefix, level) for prefix in models for level in LABEL_LEVELS]
        merged = usv_df.drop([column for column in stale if column in usv_df.columns]).with_row_index(name="_usv_row")
        for frame in coordinate_frames:
            merged = merged.join(frame, on="_usv_row", how="left")
        merged = order_usv_summary_columns(merged.drop("_usv_row"))
        # usv_summary.csv holds every other per-USV column too: publish atomically so
        # a failed write leaves the previous file intact instead of a truncated one.
        with atomic_output_path(usv_summary_loc) as tmp_summary_path:
            merged.write_csv(file=str(tmp_summary_path))

        self.message_output(
            f"Merged the torus coordinates and cluster labels of {len(models)} models "
            f"({', '.join('/'.join([f'{prefix}1', f'{prefix}2', *label_columns[prefix].values()]) for prefix in models)}) "
            f"into {usv_summary_loc.name}."
        )

    def _embed_session(
        self,
        model: dict,
        cfg: dict,
        length_threshold: float | None,
        root: pathlib.Path,
        h5_loc: pathlib.Path,
        usv_df: pls.DataFrame,
        usv_summary_loc: pathlib.Path,
        null_columns: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Description
        -----------
        Embeds the session's USVs with one decoder: reads the spectrogram H5,
        keeps the calls inside the duration window (and, where the decoder needs
        it, with a SAM mask and a conditioning value), preprocesses them as the
        training set was built (SAM masking, resize / time-stretch, the contract's
        input normalization), computes each call's conditioning value for a
        conditional package decoder, and embeds them (:func:`qlvm_model.embed_data`).

        Parameters
        ----------
        model (dict)
            ``params``, ``contract`` (dict or None), ``lattice`` and
            ``condition_bins`` (dict or None), as :func:`load_model_cell` returns them.
        cfg (dict)
            The ``infer_qlvm_latents`` settings block.
        length_threshold (float | None)
            Embed only calls with ``duration < length_threshold`` (None: every
            positive duration).
        root (pathlib.Path)
            The session root directory.
        h5_loc (pathlib.Path)
            The session's spectrogram H5.
        usv_df (pls.DataFrame)
            The session's USV summary (rows 1:1 with the H5 rows).
        usv_summary_loc (pathlib.Path)
            Path of the summary, named in errors.
        null_columns (str)
            How the log names the columns a skipped call leaves null (e.g.
            ``"qlvm_*"`` or ``"qlvm_dur1/qlvm_dur2"``).

        Returns
        -------
        usv_indices (np.ndarray)
            ``(N,)`` uint32 summary rows of the embedded calls.
        coords (np.ndarray)
            ``(N, 2)`` torus coordinates ``(x, y)`` in ``[0, 1)``.
        """
        params, contract, lattice = model['params'], model['contract'], model['lattice']
        condition_bins = model['condition_bins']
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
                    f"{n_too_long} USVs with duration >= {length_threshold} (outside the training set) get null {null_columns} columns."
                )
            usv_indices = np.flatnonzero(in_window).astype(np.uint32)
            # Apply the SAM mask exactly as build_qlvm_training_set does, so the
            # decoder -- trained on masked (background-zeroed) spectrograms -- receives
            # in-distribution input. Embedding raw spectrograms into a masked-trained
            # decoder is out-of-distribution and yields unreliable coordinates.
            # masking_type "none" keeps raw spectrograms (correct only if the decoder
            # was trained without masking).
            # A mean-frequency or loudness condition is always computed on the masked
            # call, even for a decoder fed unmasked (floored) spectrograms, so it needs
            # the masks too.
            condition = contract['condition'] if contract is not None and contract['c_dim'] else None
            masked_call_condition = condition is not None and condition['name'] in ('mean_freq', 'loudness')
            require_mask = contract is not None and contract['require_mask']
            # build_session_masks gives a call without mask instances an all-ones mask.
            # A set built with require_mask left such calls out, and their mean frequency
            # or loudness would span the whole call, so those decoders give them null
            # columns. Sets without require_mask (the shipped model, train-qlvm on a
            # main-built set) trained on them under that all-ones mask, so they are
            # embedded as before.
            drop_maskless = require_mask or masked_call_condition
            masks = None
            if cfg['masking_type'] == 'sam' or drop_maskless:
                # Without a mask group every call would get the all-ones fallback.
                if f"mask/{root.name}" not in h5_file:
                    error_message = (
                        f"{h5_loc} has no mask/{root.name} group. This decoder needs SAM masks (masking_type 'sam', "
                        f"a mean-frequency or loudness condition, or a training contract with require_mask), and "
                        f"without them every USV would be embedded unmasked. Run generate-usv-masks on the session first."
                    )
                    raise ValueError(error_message)
                masks, mask_counts = build_session_masks(
                    h5_file, root.name, usv_indices, specs.shape[1], specs.shape[2]
                )
                if drop_maskless:
                    has_mask = mask_counts > 0
                    self.message_output(
                        f"{int(np.count_nonzero(~has_mask))} USVs without a SAM mask get null {null_columns} columns."
                    )
                    usv_indices = usv_indices[has_mask]
                    masks = masks[has_mask]
            specs = specs[usv_indices].astype(np.float32)
            durations = durations[usv_indices]

        # Bandwidth and loudness conditions (phase 11) take each call's raw value from
        # outside the stored spectrogram; a call without one gets null columns.
        raw_values = None
        if condition is not None and condition['name'] == 'bandwidth':
            if "freq_bandwidth_hz" not in usv_df.columns:
                error_message = (
                    f"{usv_summary_loc.name} has no freq_bandwidth_hz column, which this bandwidth-conditioned "
                    f"decoder is decoded at. Run generate-usv-acoustic-features on the session first."
                )
                raise ValueError(error_message)
            raw_values = usv_df["freq_bandwidth_hz"].cast(pls.Float64).fill_null(np.nan).to_numpy()[usv_indices]
        elif condition is not None and condition['name'] == 'loudness':
            self.message_output(f"Measuring the image-level loudness of {len(usv_indices)} USVs from the audio.")
            raw_values = session_image_level_db(
                root_directory=str(root),
                starts=usv_df["start"].to_numpy()[usv_indices],
                stops=usv_df["stop"].to_numpy()[usv_indices],
                regions=masks > 0.5,
                spec_params=self.input_parameter_dict['generate_spectrograms'],
                message_output=self.message_output,
            )
        if raw_values is not None:
            has_value = np.isfinite(raw_values)
            self.message_output(
                f"{int(np.count_nonzero(~has_value))} USVs without a {condition['name']} value get null {null_columns} columns."
            )
            usv_indices, specs, durations, raw_values = (
                usv_indices[has_value], specs[has_value], durations[has_value], raw_values[has_value]
            )
            masks = masks[has_value] if masks is not None else None

        if cfg['masking_type'] == 'sam':
            specs = specs * masks

        # Preprocess identically to the training set (same resize/time-stretch), then
        # normalize the way the decoder's contract says it was fed.
        target_shape = tuple(int(v) for v in cfg['target_shape'])
        resized = stretch_specs(specs, durations, target_shape, cfg['time_stretch'])
        data = jnp.asarray(normalize_model_inputs(resized, contract)[:, None, :, :])

        # A conditional package decoder is decoded at the value its cell's rule gives
        # each call's own condition value (frozen_condition_values).
        condition_values = None
        if condition is not None:
            masked_resized = None
            if condition['name'] == 'mean_freq':
                masked_resized = resized if cfg['masking_type'] == 'sam' else stretch_specs(
                    specs * masks, durations, target_shape, cfg['time_stretch']
                )
            own_values = compute_condition_values(condition, durations, masked_resized, raw_values)
            decode = condition['decode'] if 'decode' in condition else None
            condition_values = frozen_condition_values(own_values, condition_bins, decode)
            if decode is None:
                rule = "frozen corpus bin means"
            else:
                rule = f"'{decode}' decode values"
                # An 'exact' cell clamps to its training range; a 'grid' cell clips to the ends
                # of its decode grid, which can lie past the training range, so each rule is
                # counted against the bounds it actually applies.
                if decode == 'exact':
                    low, high, bound_name = condition_bins['train_c_min'], condition_bins['train_c_max'], "training range"
                else:
                    low, high, bound_name = condition_bins['decode_grid'][0], condition_bins['decode_grid'][-1], "decode grid"
                n_clamped = int(np.count_nonzero((own_values < low) | (own_values > high)))
                self.message_output(
                    f"{n_clamped} USVs have a {condition['name']} value outside the {bound_name} "
                    f"[{float(low):.4f}, {float(high):.4f}] and are decoded at the end of it nearest them."
                )
            self.message_output(
                f"Conditioning on {condition['name']}: {len(np.unique(condition_values))} {rule} "
                f"for {len(own_values)} USVs."
            )

        coords = np.asarray(embed_data(
            lattice, data, params, cfg['lattice_batch_size'], cfg['data_batch_size'], condition_values
        ))                                                               # (N, 2)
        return usv_indices, coords


def export_model_cell_arrays(
    model_cell_directory: str,
    output_directory: str,
    message_output: Callable | None = None,
) -> list[pathlib.Path]:
    """
    Description
    -----------
    Writes a QLVM model package cell's clustering in the layout of the reference
    ``arrays_fine.npz`` / ``arrays_coarse.npz`` that the visualization and modeling
    readers load (``qlvm-torus-traversal-video``, the sequence embedding map, the
    manifold atlas), so they can draw the package's clusters by pointing at
    ``output_directory`` instead of the reference arrays. Per level:

    * ``ws_labels_periodic`` and ``ws_labels`` -- the cell's ``label_grid.npy``
      (``(res, res)`` int16, indexed ``[y, x]``; the package grid is periodic, so
      both keys hold it);
    * ``centers`` -- ``(K, 2)`` float32 ``(peak_x, peak_y)`` of ``clusters.csv``,
      row ``i`` for label ``i + 1``;
    * ``latent_coords`` -- ``(N, 2)`` float32 torus coordinates of the package's
      corpus calls, from ``posterior_cache.npz``'s ``torus_weighted``;
    * ``sample_ws`` and ``sample_ws_periodic`` -- ``(N,)`` int16 labels of those
      calls from ``cluster_labels.csv``;
    * ``heatmap`` -- ``(res, res)`` float32 aggregated posterior: the
      ``aggregated`` mass of every point of the embedding lattice
      (``embedding_fib_m`` of the contract) added to its pixel, summing to the
      number of corpus calls. The reference arrays' heatmap was built by a
      different, unrecorded smoothing, so the two look alike but are not equal;
    * ``model_id`` -- ``<package>/<phase>/<cell>``.

    Both files are published atomically.

    Parameters
    ----------
    model_cell_directory (str)
        Path to the package cell.
    output_directory (str)
        Directory to write ``arrays_fine.npz`` and ``arrays_coarse.npz`` into
        (created if missing).
    message_output (Callable)
        Logging callback; defaults to ``print``.

    Returns
    -------
    written (list[pathlib.Path])
        The two files written.
    """
    message_output = message_output if message_output is not None else print
    cell = pathlib.Path(configure_path(model_cell_directory))
    output_dir = pathlib.Path(configure_path(output_directory))
    output_dir.mkdir(parents=True, exist_ok=True)
    with cell_file(cell, "training_contract.json").open() as contract_file:
        contract = json.load(contract_file)
    with np.load(cell_file(cell, "posterior_cache.npz"), allow_pickle=False) as cache:
        torus_weighted = cache["torus_weighted"]
        aggregated = cache["aggregated"]
    latent_coords = np.asarray(torus_basis_reverse(jnp.asarray(torus_weighted)), dtype=np.float32)
    lattice = np.asarray(gen_fib_basis(contract["embedding_fib_m"])) % 1.0
    if lattice.shape[0] != aggregated.shape[0]:
        error_message = (
            f"{cell}: posterior_cache.npz aggregates {aggregated.shape[0]} lattice points but the contract's "
            f"embedding lattice has {lattice.shape[0]}."
        )
        raise ValueError(error_message)
    model_id = "/".join(cell.parts[-3:])
    written = []
    for level in ("fine", "coarse"):
        cluster_directory = cell_cluster_directory(cell, level)
        label_grid = np.load(cluster_directory / "label_grid.npy", allow_pickle=False)
        resolution = label_grid.shape[0]
        pixel_x = np.clip((lattice[:, 0] * resolution).astype(int), 0, resolution - 1)
        pixel_y = np.clip((lattice[:, 1] * resolution).astype(int), 0, resolution - 1)
        heatmap = np.bincount(
            pixel_y * resolution + pixel_x, weights=aggregated, minlength=resolution * resolution
        ).reshape(resolution, resolution)
        clusters = pls.read_csv(cluster_directory / "clusters.csv").sort("label")
        labels = pls.read_csv(cluster_directory / "cluster_labels.csv")["label"].to_numpy()
        if labels.shape[0] != latent_coords.shape[0]:
            error_message = (
                f"{cell}: cluster/{level}/cluster_labels.csv has {labels.shape[0]} rows but posterior_cache.npz "
                f"{latent_coords.shape[0]}."
            )
            raise ValueError(error_message)
        destination = output_dir / f"arrays_{level}.npz"
        with atomic_output_path(destination) as tmp_path, tmp_path.open("wb") as array_file:
            np.savez(
                array_file,
                ws_labels_periodic=label_grid.astype(np.int16),
                ws_labels=label_grid.astype(np.int16),
                centers=clusters.select(["peak_x", "peak_y"]).to_numpy().astype(np.float32),
                latent_coords=latent_coords,
                sample_ws=labels.astype(np.int16),
                sample_ws_periodic=labels.astype(np.int16),
                heatmap=heatmap.astype(np.float32),
                model_id=np.array(model_id),
            )
        written.append(destination)
        message_output(
            f"Wrote {destination} ({clusters.height} {level} clusters, {labels.shape[0]} calls) from {model_id}."
        )
    return written


@click.command(name="export-qlvm-reference-arrays")
@click.option('--model-cell-directory', 'model_cell_directory', type=click.Path(exists=True, file_okay=False, dir_okay=True), required=True, help='A QLVM model package cell, e.g. .../qlvm_models_latest/v2/phase9_USVs_masked_relu/natural_3strata_N65000_masked.')
@click.option('--output-directory', 'output_directory', type=click.Path(file_okay=False, dir_okay=True), required=True, help='Directory to write arrays_fine.npz and arrays_coarse.npz into (created if missing), e.g. <spectrograms_dir>/qlvm.')
def export_qlvm_reference_arrays_cli(model_cell_directory, output_directory) -> None:
    """
    Description
    -----------
    A command-line tool to write a QLVM model package cell's clustering as the
    ``arrays_fine.npz`` / ``arrays_coarse.npz`` reference arrays the QLVM
    visualizations read.

    Parameters
    ----------

    Returns
    -------
    None
    """

    export_model_cell_arrays(
        model_cell_directory=model_cell_directory,
        output_directory=output_directory,
        message_output=print,
    )


@click.command(name="infer-qlvm-latents")
@click.option('--root-directory', type=click.Path(exists=True, file_okay=False, dir_okay=True), required=True, help='Session root directory path.')
@click.option('--model-cell-directory', 'model_cell_directory', type=str, default=None, required=False, help='A QLVM model package cell (e.g. .../qlvm_models_latest/v2/phase9_USVs_masked_relu/natural_3strata_N65000_masked); when set, its checkpoint, training_contract.json, embedding lattice and label grids replace the weights, reference-arrays and lattice settings.')
@click.option('--model-cell', 'model_cells', type=(str, str), multiple=True, default=None, required=False, help='A column prefix and a QLVM model package cell (e.g. --model-cell qlvm_dur .../qlvm_models_latest/v3/phase11_cond_duration_floor/natural_5strata_N29000_unmasked_floor); repeat once per model. When given, these pairs replace the model_cells setting: the session is placed on the torus of every listed cell and <prefix>1/<prefix>2 plus the cluster-label columns of each prefix (see --model-cell-labels) are written. Cannot be combined with a model cell directory.')
@click.option('--model-cell-labels', 'model_cell_label_levels', type=(str, str), multiple=True, default=None, required=False, help='A model_cells column prefix and the comma-separated cluster-label levels it writes, among fine and coarse (e.g. --model-cell-labels qlvm_dur fine,coarse writes qlvm_dur_category and qlvm_dur_supercategory; an empty string writes none); repeat once per prefix. When given, these pairs replace the model_cell_label_levels setting; prefixes not listed keep the default (qlvm: fine and coarse -> qlvm_category, qlvm_supercategory; every other prefix P: fine -> P_category).')
@click.option('--prefer-package-values/--no-prefer-package-values', 'prefer_package_values', default=None, required=False, help='With model cells: take a corpus session\'s coordinates from the package\'s own embedding when its spectrogram H5 is unchanged since the package (SHA-256, row count, durations and mask counts verified), else infer them; --no-prefer-package-values infers every session.')
@click.option('--weights-npz-path', 'weights_npz_path', type=str, default=None, required=False, help='Path to the converted decoder weights .npz.')
@click.option('--reference-arrays-fine-npz-path', 'reference_arrays_fine_npz_path', type=str, default=None, required=False, help='Path to the FINE reference arrays.npz (ws_labels_periodic -> qlvm_category).')
@click.option('--reference-arrays-coarse-npz-path', 'reference_arrays_coarse_npz_path', type=str, default=None, required=False, help='Path to the COARSE reference arrays.npz (ws_labels_periodic -> qlvm_supercategory).')
@click.option('--lattice-type', 'lattice_type', type=click.Choice(['korobov', 'roberts', 'fibonacci']), default=None, required=False, help='Quasi-random lattice generator used to rebuild the fixed QLVM (quasi-Monte Carlo latent variable model) lattice at inference.')
@click.option('--latent-dim', 'latent_dim', type=int, default=None, required=False, help='Dimensionality of the toroidal latent space.')
@click.option('--n-points', 'n_points', type=int, default=None, required=False, help='Number of lattice points used at inference.')
@click.option('--korobov-a', 'korobov_a', type=int, default=None, required=False, help='Korobov generating integer (used when lattice-type=korobov).')
@click.option('--fib-m', 'fib_m', type=int, default=None, required=False, help='Fibonacci lattice order m (used when lattice-type=fibonacci).')
@click.option('--time-stretch/--no-time-stretch', 'time_stretch', default=None, required=False, help='Whether to time-stretch each spectrogram to the fixed size (matching training preprocessing) instead of a plain resize.')
@click.option('--masking-type', 'masking_type', type=click.Choice(['sam', 'none']), default=None, required=False, help='Apply SAM mask regions before embedding ("sam", matching training) or embed raw spectrograms ("none"). With "sam", a session without a mask group raises.')
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
    torus and merge the latents/categories into its USV summary CSV (or, with
    ``--model-cell`` pairs, the torus coordinates and cluster labels of every
    listed model; ``--model-cell-labels`` pairs choose each prefix's label levels).

    Parameters
    ----------

    Returns
    -------
    None
    """
    provided_params = [key for key in kwargs if ctx.get_parameter_source(key) == ParameterSource.COMMANDLINE]

    # --model-cell pairs become the model_cells object (prefix -> cell) and
    # --model-cell-labels pairs the model_cell_label_levels object (prefix -> levels),
    # which the generic key-by-key override cannot build; they are written after the others.
    processing_settings_dict = modify_settings_json_for_cli(
        ctx=ctx,
        provided_params=[key for key in provided_params if key not in ('model_cells', 'model_cell_label_levels')],
        settings_dict='processing_settings',
        parameters_lists=['target_shape'],
        block='infer_qlvm_latents',
    )
    if 'model_cells' in provided_params:
        processing_settings_dict['infer_qlvm_latents']['model_cells'] = validate_model_cells(kwargs['model_cells'])
    if 'model_cell_label_levels' in provided_params:
        processing_settings_dict['infer_qlvm_latents']['model_cell_label_levels'] = parse_model_cell_label_levels(
            kwargs['model_cell_label_levels']
        )

    QLVMLatentInference(
        root_directory=root_directory,
        input_parameter_dict=processing_settings_dict,
        message_output=print,
    ).infer_and_merge()
