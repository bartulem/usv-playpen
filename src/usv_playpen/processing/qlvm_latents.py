"""
@author: bartulem
Embed a session's USV spectrograms into the tori of trained QLVM model package
cells and merge the torus coordinates into its ``*_usv_summary.csv``.

This is the in-house, JAX (torch-free) inference driver. Every model it embeds
with is one cell of a QLVM model package (``qlvm_models_latest/v3``, and the
``v2`` / ``v2.1`` layout), listed in the ``model_cells`` setting (column prefix ->
package cell; by default the production mapping ``QLVM_PRODUCTION_MODEL_CELLS``,
filled in by :func:`os_utils.derive_spectrogram_model_paths`). A cell brings
everything inference needs (:func:`load_model_cell`): its ``checkpoint.tar`` is
read without torch, its ``training_contract.json`` fixes the head (legacy or
ReLU), the input normalization (min-max, and the loudness floor of floor-trained
cells) and the duration window, and its Fibonacci embedding lattice is rebuilt.
A clustered v2 / v3 package cell may also ship fine and coarse ``label_grid.npy``
files; they are not read here. The summary's one category column,
``qlvm_category``, comes from ``assign-qlvm-categories`` (see below).
Conditional cells take one conditioning value per call: phase 10 cells
(``qlvm_models_latest/v2``, duration or mean frequency) decode it at the frozen
corpus bin mean, phase 11 cells (``qlvm_models_latest/v3``, duration, mean
frequency, bandwidth or loudness) at the call's own value clamped to the training
range or snapped to the cell's decode grid, as the contract's ``condition.decode``
says (:func:`frozen_condition_values`); a phase 11 recipe cell trained by
``train-qlvm`` on spectral entropy (the summary's ``spectral_entropy``, min-max
scaled by its training split's range) is decoded on its grid the same way.

The session is placed on the torus of every listed cell and each prefix ``P``
gets the float columns ``P1`` / ``P2`` (nulls where the call was not placed), and
nothing else: the summary's category column of the regular map, ``qlvm_category``,
holds the content-ridge categories ``assign-qlvm-categories`` writes after this
step (:mod:`qlvm_categories`), and the canonical layout
(``os_utils.USV_SUMMARY_COLUMN_ORDER``) has no coarse level and no category
column on the conditional maps. Re-embedding a prefix drops every earlier
coordinate, category and category confidence column of it (``P_category``, the
legacy ``P_supercategory``, ``P_category_agreement``, ``P_category_uncertain``;
:func:`model_cell_stale_columns`), since categories read off the old coordinates
no longer hold. No model-provenance column is written; the legacy ``qlvm_model`` column that
summaries embedded by the retired single-model run still carry is dropped.

``tidy-usv-summary-columns`` (:func:`tidy_usv_summary_columns_cli`) migrates an
existing summary to the canonical layout: it drops the obsolete columns
(``os_utils.USV_SUMMARY_OBSOLETE_COLUMNS`` and every category agreement /
uncertain column) and reorders the rest, with a dry-run mode that only reports.

Fidelity: the session spectrograms are preprocessed with the SAME resize /
time-stretch used to build the training set (:func:`stretch_specs`), so they are
in-distribution for the decoder. A masked cell (``masking_type`` ``"sam"``) gets
exactly what a masked training set fed it: call and SAM mask union resized
separately, the resized mask binarized at 0.5 and multiplied in, then a min-max
and the binarized mask once more (``build-qlvm-training-set --apply-mask`` and
``train_qlvm.prepare_split``). A cell that has not been clustered (no
``label_grid.npy``) embeds like any other.

Per model, a corpus session whose spectrogram H5 is verifiably the one the
package was built from (``SESSION_H5_BASELINE.tsv`` SHA-256, row count, and the
package's per-row durations and mask counts) takes the package's own coordinates
(:func:`package_route_verdict`, :func:`load_package_session_rows`); every other
session is embedded with the cell as above.

Pure squeaks. The USV maps were trained on ultrasonic calls, so a pure squeak --
``squeak`` true and ``usv`` false in the summary (written by
``detect-usv-squeaks``), a broadband squeak with no ultrasonic call in it -- is not
a USV the maps can place: it is skipped by every model cell and gets null
coordinates (squeaks have their own map,
``infer-qlvm-squeak-latents``). A segment holding both (``usv`` and ``squeak``
true) is embedded like any other call. Rows with null booleans (noise, or too
short to score) are treated as before. Because the rule needs the two booleans, a
summary without them raises: run ``detect-usv-noise`` and ``detect-usv-squeaks`` on
the session first.
"""

from __future__ import annotations

import collections
import functools
import json
import pathlib
import pickle
import shutil
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
    CATEGORY_CONFIDENCE_SUFFIXES,
    QLVM_PRODUCTION_MODEL_CELLS,
    QLVM_SUMMARY_MAP_PREFIXES,
    USV_SUMMARY_COLUMN_ORDER,
    atomic_output_path,
    configure_path,
    derive_spectrogram_model_paths,
    first_match_or_raise,
    order_usv_summary_columns,
    pure_squeak_mask,
    tidy_usv_summary_columns,
)
from ..processing.build_qlvm_training_set import (
    build_session_masks,
    file_sha256,
    stretch_specs,
)
from ..time_utils import is_gui_context, smart_wait
from .qlvm_model import (
    decoder_head,
    embed_data,
    gen_fib_basis_float32,
    torus_basis_reverse,
)

# Conditions a package decoder may be trained on (phase 10: the first two; phase 11: the
# first four; train-qlvm: all five, spectral_entropy only there).
CONDITION_NAMES = ("duration", "mean_freq", "bandwidth", "loudness", "spectral_entropy")

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


def load_decoder_params(checkpoint_path: str) -> dict[str, jnp.ndarray]:
    """
    Description
    -----------
    Loads the frozen decoder weights of a model package cell's torch zip
    checkpoint (``checkpoint.tar``) into the key form
    :func:`qlvm_model.decoder_forward` expects, read without torch
    (:func:`read_torch_checkpoint`; its ``"model"`` entry when present). A leading
    ``decoder.`` prefix (present when the full QMCLVM ``state_dict`` is dumped) is
    stripped. The converted ``.npz`` weights the retired single-model route read
    are no longer accepted.

    Parameters
    ----------
    checkpoint_path (str)
        Path to the torch checkpoint.

    Returns
    -------
    params (dict[str, jnp.ndarray])
        Decoder weights keyed by ``"<layer_idx>.weight"`` / ``"<layer_idx>.bias"``.
    """
    checkpoint = read_torch_checkpoint(pathlib.Path(configure_path(checkpoint_path)))
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    params: dict[str, jnp.ndarray] = {}
    for key, value in state.items():
        clean = key[len("decoder."):] if key.startswith("decoder.") else key
        params[clean] = jnp.asarray(value)
    return params


def load_model_cell(model_cell_directory: str) -> dict:
    """
    Description
    -----------
    Loads one cell of a QLVM model package (the ``qlvm_models_latest/v2`` layout):
    the decoder weights from its torch ``checkpoint.tar`` (read without torch), its
    ``training_contract.json``, the Fibonacci lattice the package embedded its
    corpus on (``embedding_fib_m`` of the contract) and, for a conditional cell,
    its frozen condition table. A package cell's cluster ``label_grid.npy`` files,
    if any, are not read: inference writes torus coordinates only.

    Parameters
    ----------
    model_cell_directory (str)
        Path to the package cell, e.g.
        ``.../qlvm_models_latest/v2/phase9_USVs_masked_relu/natural_3strata_N65000_masked``.

    Returns
    -------
    model (dict)
        ``params`` (decoder weights), ``contract`` (dict), ``lattice``
        (``(fib(m), 2)``), ``condition_bins`` (dict, or None for an unconditional cell) and
        ``model_id`` (``<package>/<phase>/<cell>``, the last three path components).
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
        # The torch float32 lattice the package's corpus embedding was computed on (the
        # exact lattice, cast to float32 by JAX, lands up to ~1e-3 off it and moves 8% of
        # calls by more than 1e-3; measured on 9,390 calls of 6 corpus sessions).
        "lattice": gen_fib_basis_float32(contract["embedding_fib_m"]),
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
    image-level loudness, with ``[lo, hi]`` the contract's ``db_range``.
    ``"spectral_entropy"`` (``train-qlvm``):
    ``clip((H - entropy_min) / (entropy_max - entropy_min), 0, 1)`` of the
    summary's ``spectral_entropy`` in nats, with ``entropy_min`` / ``entropy_max``
    the training split's range the contract records, so a call outside it is
    clamped to 0 or 1. The three raw-unit maps run in float64 and round to
    float32, as the training sets stored them; a NaN raw value stays NaN.

    Parameters
    ----------
    condition (dict)
        The contract's ``condition`` block.
    durations (np.ndarray)
        ``(N,)`` durations in time bins.
    masked_spectrograms (np.ndarray | None)
        ``(N, F, T)`` resized SAM-masked spectrograms; required for ``"mean_freq"``.
    raw_values (np.ndarray | None)
        ``(N,)`` raw values in Hz (``"bandwidth"``: ``freq_bandwidth_hz``), dB
        (``"loudness"``: ``image_level_db``) or nats (``"spectral_entropy"``:
        ``spectral_entropy``); required for those three. Defaults to None.

    Returns
    -------
    values (np.ndarray)
        ``(N,)`` float32 conditioning values.
    """
    if condition["name"] in ("bandwidth", "loudness", "spectral_entropy"):
        if raw_values is None:
            error_message = f"compute_condition_values: {condition['name']} needs each call's raw value."
            raise ValueError(error_message)
        raw = np.asarray(raw_values, dtype=np.float64)
        if condition["name"] == "bandwidth":
            return np.clip(raw / 90000.0, 0.0, 1.0).astype(np.float32)
        if condition["name"] == "spectral_entropy":
            low, high = float(condition["entropy_min"]), float(condition["entropy_max"])
            return np.clip((raw - low) / (high - low), 0.0, 1.0).astype(np.float32)
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
    frequency, loudness, spectral entropy) the nearest point of ``decode_grid``, a value beyond
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


def condition_quantile_value(condition_bins: dict, decode: str | None, quantile: float) -> np.float32:
    """
    Description
    -----------
    One fixed conditioning value of a conditional QLVM cell: the ``quantile`` of
    the conditioning distribution of the corpus the cell was trained on, decoded
    by the cell's own rule (:func:`frozen_condition_values`: clamped to the
    training range for ``decode`` ``"exact"``, snapped to the decode grid for
    ``"grid"``, the bin mean for phase 10 bins), so the value is one the cell's
    embedding itself decodes calls at.

    A conditional decoder maps a torus position AND a conditioning value to a
    spectrogram, so anything that decodes torus positions without a call (the
    pullback metric ``G(z) = J(z)^T J(z)`` of the torus geodesics, the decoded
    vocal-space atlas of the manifold filter atlas) has to fix that value. This
    fixes it from the cell alone: the training corpus distribution is read from
    ``condition_bins.npz``, whose ``edges`` are the bin edges of ``c`` and whose
    ``group_sizes`` count the training rows in each bin (phase 11 /
    ``train-qlvm`` cells); the quantile is interpolated linearly inside the bin it
    falls in. A phase 10 table (``edges`` + ``bin_mean``, no ``group_sizes``) holds
    corpus quantile bins, so each of its bins is taken to hold an equal share. The
    value is therefore the same for every run, cohort and session subset (it does
    not depend on which calls a run happens to hold), and ``quantile`` ``0.5`` is
    the corpus median call.

    Parameters
    ----------
    condition_bins (dict)
        The cell's ``condition_bins.npz`` (``load_model_cell``'s
        ``condition_bins``).
    decode (str | None)
        The contract's ``condition.decode`` (``"exact"`` or ``"grid"``; None for
        phase 10 bins).
    quantile (float)
        The corpus quantile of ``c`` to decode at, in ``[0, 1]``.

    Returns
    -------
    value (np.float32)
        The conditioning value.

    Raises
    ------
    ValueError
        ``quantile`` is outside ``[0, 1]``, or the table holds no usable ``edges``
        (or its ``group_sizes`` do not match them).
    """
    quantile = float(quantile)
    if not 0.0 <= quantile <= 1.0:
        error_message = f"condition_quantile_value: quantile must be in [0, 1], got {quantile!r}."
        raise ValueError(error_message)
    if "edges" not in condition_bins:
        error_message = f"condition_quantile_value: condition_bins holds {sorted(condition_bins)}, no 'edges'."
        raise ValueError(error_message)
    edges = np.asarray(condition_bins["edges"], dtype=np.float64).reshape(-1)
    if "group_sizes" in condition_bins:
        sizes = np.asarray(condition_bins["group_sizes"], dtype=np.float64).reshape(-1)
    else:
        sizes = np.ones(edges.shape[0] - 1, dtype=np.float64)
    if edges.shape[0] != sizes.shape[0] + 1 or sizes.sum() <= 0.0:
        error_message = (
            f"condition_quantile_value: {edges.shape[0]} edges for {sizes.shape[0]} bins "
            f"(total {sizes.sum()} rows); the table needs one more edge than bins and at least one row."
        )
        raise ValueError(error_message)
    cumulative = np.concatenate([[0.0], np.cumsum(sizes)]) / sizes.sum()
    value = np.interp(quantile, cumulative, edges)
    return frozen_condition_values(np.array([value]), condition_bins, decode)[0]


def model_decode_condition(model: dict, condition_quantile: float) -> np.float32 | None:
    """
    Description
    -----------
    The conditioning value a cell is decoded at when there is no call to take it
    from: None for an unconditional cell (``c_dim`` 0, which takes none), else
    :func:`condition_quantile_value` of the cell's training corpus at
    ``condition_quantile``.

    Parameters
    ----------
    model (dict)
        A cell as :func:`load_model_cell` returns it (``contract`` and
        ``condition_bins`` are read).
    condition_quantile (float)
        The corpus quantile of the conditioning value, in ``[0, 1]`` (ignored for
        an unconditional cell).

    Returns
    -------
    value (np.float32 | None)
        The conditioning value, or None for an unconditional cell.
    """
    if not model["contract"]["c_dim"]:
        return None
    condition = model["contract"]["condition"]
    decode = condition["decode"] if condition is not None and "decode" in condition else None
    return condition_quantile_value(model["condition_bins"], decode, condition_quantile)


def minmax_per_spectrogram(spectrograms: np.ndarray, epsilon: np.float32) -> np.ndarray:
    """
    Description
    -----------
    Rescales each spectrogram on its own to ``(x - min) / (max - min + epsilon)``,
    the min-max a QLVM model package applies to its decoder inputs (and
    :mod:`train_qlvm` to its training inputs). Zeros stay zeros when they are the
    spectrogram's minimum (a SAM-masked background).

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


def normalize_model_inputs(spectrograms: np.ndarray, contract: dict) -> np.ndarray:
    """
    Description
    -----------
    Applies the input normalization a decoder was trained with to resized
    spectrograms, as its training contract records it. ``input_normalization``
    ``"none"`` leaves the stored values as they are. ``"minmax"`` (QLVM model
    packages, and every contract ``train-qlvm`` writes) rescales each
    spectrogram to ``(x - min) / (max - min + normalization_epsilon)`` in float32,
    and a non-null ``floor`` then applies ``clip((x - floor) / (1 - floor), 0, 1)``
    followed by a second min-max -- the order the package's ``model_input`` uses.
    SAM masking, when the contract asks for it, has already zeroed the background,
    and a min-max keeps those zeros.

    Parameters
    ----------
    spectrograms (np.ndarray)
        Resized spectrograms, shape ``(N, F, T)``.
    contract (dict)
        The training contract (a model package cell's ``training_contract.json``).

    Returns
    -------
    inputs (np.ndarray)
        ``(N, F, T)`` float32 decoder inputs.
    """
    inputs = np.asarray(spectrograms, dtype=np.float32)
    if contract["input_normalization"] == "none":
        return inputs
    epsilon = np.float32(contract["normalization_epsilon"])
    inputs = minmax_per_spectrogram(inputs, epsilon)
    if contract["floor"] is not None:
        floor = np.float32(contract["floor"])
        inputs = np.clip((inputs - floor) / np.float32(1.0 - floor), np.float32(0.0), np.float32(1.0))
        inputs = minmax_per_spectrogram(inputs, epsilon)
    return inputs.astype(np.float32, copy=False)


def enforce_training_contract(contract: dict, cfg: dict, params: dict[str, jnp.ndarray]) -> float:
    """
    Description
    -----------
    Checks the ``infer_qlvm_latents`` settings and the loaded weights against the
    decoder's training contract (a model package cell's ``training_contract.json``,
    see :func:`load_model_cell`) and returns the duration window to embed. Every
    disagreement is collected and raised together: ``masking_type``,
    ``target_shape``, ``time_stretch`` and ``latent_dim`` must equal the
    contract's; a ``length_threshold`` set in the settings must equal the training
    set's; the weights' head (:func:`qlvm_model.decoder_head`) must be the
    contract's ``decoder_head``. The contract must also describe a decoder this
    module can run: no conditioning input (``c_dim`` 0) or one conditioning value
    of a model package's ``"duration"`` / ``"mean_freq"`` / ``"bandwidth"`` /
    ``"loudness"`` condition or of ``train-qlvm``'s ``"spectral_entropy"`` (a
    phase 11 ``decode`` of ``"grid"`` or ``"exact"``; a spectral entropy block
    must carry ``entropy_min`` < ``entropy_max``), an
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
    if condition is not None and condition["name"] == "spectral_entropy" and not (
        "entropy_min" in condition and "entropy_max" in condition and condition["entropy_max"] > condition["entropy_min"]
    ):
        mismatches.append(
            f"condition: a spectral_entropy cell needs entropy_min < entropy_max (its training split's range), "
            f"trained {condition!r}"
        )
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


def label_grid_lookup(coords: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """
    Description
    -----------
    Looks up each torus coordinate's cluster label in one periodic label grid:
    ``label = grid[floor(y * res) mod res, floor(x * res) mod res]``, with ``res``
    the grid's resolution (``grid.shape[0]``). This is the pixel rule of the QLVM
    model packages: on the posterior-mean coordinates of the v3 production cells
    it reproduces the package's ``inference/clusters_<level>/cluster_labels.csv``
    on every one of the 445,742 corpus calls but one (a duration-map call whose
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
    torus-coordinate columns of the summary's QLVM map prefixes
    ``os_utils.QLVM_SUMMARY_MAP_PREFIXES`` and of the production prefixes of
    ``os_utils.QLVM_PRODUCTION_MODEL_CELLS`` -- ``qlvm1`` / ``qlvm2``,
    ``qlvm_duration1`` / ``qlvm_duration2``,
    ``qlvm_entropy1`` / ``qlvm_entropy2``, ``qlvm_bandwidth1`` /
    ``qlvm_bandwidth2``, ``qlvm_loudness1`` / ``qlvm_loudness2``, ... -- are allowed, since writing them is what a run is for;
    see :func:`model_cell_reserved_columns`). Each cell directory must be a
    non-empty string. Every problem is collected and raised together.

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


def model_cell_reserved_columns() -> set[str]:
    """
    Description
    -----------
    The USV summary columns a ``model_cells`` run may never write: every column
    of ``USV_SUMMARY_COLUMN_ORDER`` except the torus-coordinate columns
    (``P1`` / ``P2``) of the summary's QLVM map prefixes
    (``os_utils.QLVM_SUMMARY_MAP_PREFIXES``: ``qlvm``, ``qlvm_duration``,
    ``qlvm_entropy``, ``qlvm_bandwidth``, ``qlvm_loudness``)
    and of the production prefixes of ``os_utils.QLVM_PRODUCTION_MODEL_CELLS`` --
    writing those is what a run is for. Every other summary column (DAS event,
    acoustic features, the category columns ``assign-qlvm-categories`` writes, the
    squeak torus coordinates, ...) must stay untouched.

    Parameters
    ----------

    Returns
    -------
    reserved (set[str])
        The reserved column names.
    """
    prefixes = (*QLVM_SUMMARY_MAP_PREFIXES, *QLVM_PRODUCTION_MODEL_CELLS)
    writable = {f"{prefix}{axis}" for prefix in prefixes for axis in (1, 2)}
    return set(USV_SUMMARY_COLUMN_ORDER) - writable


def model_cell_stale_columns(prefix: str) -> list[str]:
    """
    Description
    -----------
    The summary columns that describe a ``model_cells`` prefix's placement, and so
    go stale when the prefix is embedded again: its coordinates ``P1`` / ``P2``,
    the category column ``P_category`` an earlier ``assign-qlvm-categories`` wrote
    for it (``qlvm_category`` for the regular map), the coarse-level
    ``P_supercategory`` older summaries may still carry (the canonical layout has no
    coarse level), and the per-call category confidence columns of the category
    column (its name plus each suffix of ``os_utils.CATEGORY_CONFIDENCE_SUFFIXES``
    minus its leading ``_category``, e.g. ``qlvm_category_agreement``,
    ``qlvm_category_uncertain``).
    A category assigned from the old coordinates no longer holds for the new
    ones, so ``assign-qlvm-categories`` must run again after a re-embedding.

    Parameters
    ----------
    prefix (str)
        The column prefix (a key of ``infer_qlvm_latents.model_cells``).

    Returns
    -------
    stale (list[str])
        The column names, coordinates first.
    """
    category_column = f"{prefix}_category"
    stale = [f"{prefix}{axis}" for axis in (1, 2)]
    stale += [category_column, f"{prefix}_supercategory"]
    stale += [f"{category_column}{suffix.removeprefix('_category')}" for suffix in CATEGORY_CONFIDENCE_SUFFIXES]
    return stale


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
    Embeds one session's spectrograms into the torus of every QLVM model package
    cell of the ``model_cells`` setting and merges the torus coordinates into its
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
        Initializes the QLVMLatentInference.

        Parameters
        ----------
        root_directory (str)
            Session root directory (contains the ``audio`` tree).
        input_parameter_dict (dict)
            Processing settings; the ``infer_qlvm_latents`` block supplies the
            model package cells (``model_cells``) and the preprocessing settings checked against each cell's training contract.
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
        Places the session on the torus of every QLVM model package cell of the
        ``model_cells`` setting (column prefix -> model package cell; filled with
        the production mapping by :func:`os_utils.derive_spectrogram_model_paths`
        when the settings name no model and carry a ``spectrograms_root``). Per
        cell: loads its decoder weights, training contract and embedding lattice
        (:func:`load_model_cell`); checks the settings against the
        contract (:func:`enforce_training_contract`), whose ``length_threshold``
        applies; reads the session spectrogram H5, preprocesses identically to
        training, embeds into the torus (or takes the package's own coordinates,
        see :meth:`_merge_model_cells`), and merges the coordinate columns into the matching USV summary rows (joined on the positional USV
        row index, since the spectrogram rows are 1:1 with the
        ``usv_summary.csv`` rows; USVs with non-positive duration, or with a
        duration at or above the training set's ``length_threshold``, are skipped
        and get nulls). Pure squeaks (``squeak & ~usv``) are skipped by
        every cell and get nulls; segments holding both are embedded as usual; a
        summary without ``usv`` / ``squeak`` raises KeyError (run ``detect-usv-squeaks``
        first). When the contract's training set kept only calls with a SAM mask
        (``require_mask``) or the decoder conditions on mean frequency or
        loudness, USVs without a mask instance are skipped and get nulls too.
        When the decoder needs SAM masks at all (those cases, or ``masking_type``
        ``"sam"``), a session H5 without a ``mask/<session>`` group raises. A
        conditional package decoder is decoded at :func:`frozen_condition_values`
        of each call's own value (:func:`compute_condition_values`); a bandwidth
        condition reads it from the summary's ``freq_bandwidth_hz`` (missing
        column raises), a loudness condition from the summary's ``loudness_db``
        (the absolute loudness ``generate-usv-acoustic-features`` measured from the
        session audio with :func:`compute_usv_loudness.session_image_level_db`;
        missing column raises), a spectral entropy condition from the summary's
        ``spectral_entropy`` (scaled by the contract's ``entropy_min`` /
        ``entropy_max`` and clamped to ``[0, 1]``; missing column raises), and
        calls with no value get nulls. The summary is rewritten atomically.

        Each prefix ``P`` gets the float columns ``P1`` / ``P2`` only (the regular
        map's ``qlvm_category`` comes from ``assign-qlvm-categories``); no model
        column is written, and the legacy ``qlvm_model`` column (written by the
        retired single-model run) and the earlier coordinate, category and category
        confidence columns of the listed prefixes are removed first (see :meth:`_merge_model_cells`). An empty ``model_cells``
        raises ValueError before anything is read: model package cells are the
        only models this module embeds with.

        Parameters
        ----------

        Returns
        -------
        Updated ``*_usv_summary.csv`` with the ``P1`` / ``P2`` columns of every
        ``model_cells`` prefix.
        """
        self.message_output(
            f"QLVM latent inference started at: {datetime.now().hour:02d}:{datetime.now().minute:02d}:{datetime.now().second:02d}."
        )
        smart_wait(app_context_bool=self.app_context_bool, seconds=1)

        derive_spectrogram_model_paths(self.input_parameter_dict)
        cfg = self.input_parameter_dict['infer_qlvm_latents']
        if not cfg['model_cells']:
            error_message = (
                "infer_qlvm_latents: model_cells is empty, so there is no model to embed with. List the QLVM "
                "model package cells to embed with (column prefix -> cell directory, e.g. via --model-cell "
                "PREFIX CELL), or set spectrograms_root so the production cells (os_utils."
                "QLVM_PRODUCTION_MODEL_CELLS) are filled in. The single-model route (model_cell_directory / "
                "weights_npz_path) is retired."
            )
            raise ValueError(error_message)
        self._merge_model_cells(cfg)

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

    def _merge_model_cells(self, cfg: dict) -> None:
        """
        Description
        -----------
        The run of :meth:`infer_and_merge`: places the session on the
        torus of every cell of ``model_cells`` and merges, per prefix ``P``, the
        float columns ``P1`` / ``P2`` into the summary (nulls where a call was not
        placed). The settings are validated, and every cell
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
        other case embeds the session with the cell (:meth:`_embed_session`).
        Pure squeaks (``squeak & ~usv``)
        are left out of both routes -- not embedded by the cell, and dropped from
        the package's rows -- so their coordinates are null; a summary
        without ``usv`` / ``squeak`` raises before anything is embedded.

        No model-provenance column is written; the legacy ``qlvm_model`` column
        (which summaries embedded by the retired single-model run still carry)
        and every earlier coordinate, category and category confidence column of
        the listed prefixes (:func:`model_cell_stale_columns`: ``P1`` / ``P2``,
        ``P_category``, the legacy ``P_supercategory``, ``P_category_agreement`` and
        ``P_category_uncertain``) are dropped before the merge, so a re-embedded
        prefix never keeps categories read off its old coordinates. The summary is
        rewritten atomically and in canonical column order.

        Parameters
        ----------
        cfg (dict)
            The ``infer_qlvm_latents`` settings block.

        Returns
        -------
        None
        """
        if not isinstance(cfg['model_cells'], dict):
            error_message = (
                f"infer_qlvm_latents.model_cells must be an object of column prefix -> model cell directory, "
                f"got {type(cfg['model_cells']).__name__}."
            )
            raise ValueError(error_message)
        model_cells = validate_model_cells(cfg['model_cells'].items())

        models = {}
        for prefix, cell_directory in model_cells.items():
            model = load_model_cell(cell_directory)
            model['length_threshold'] = enforce_training_contract(model['contract'], cfg, model['params'])
            models[prefix] = model
        for prefix, model in models.items():
            self.message_output(
                f"{prefix}1/{prefix}2: model package cell {model['model_id']} ({decoder_head(model['params'])} head, "
                f"{model['lattice'].shape[0]}-point Fibonacci lattice)."
            )

        root, h5_loc, usv_summary_loc, usv_df = self._locate_session_files()
        # Pure squeaks are not USVs the maps can place; every cell skips them (null
        # coordinates). "both" segments hold a USV too and are embedded.
        is_pure_squeak = pure_squeak_mask(usv_df, usv_summary_loc.name).to_numpy()
        self.message_output(
            f"{int(np.count_nonzero(is_pure_squeak))} pure squeak(s) (squeak true, usv false) are skipped by every "
            f"model and get null coordinates."
        )
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
                not_squeak = ~is_pure_squeak[np.asarray(rows['row'], dtype=np.int64)]
                usv_indices = np.asarray(rows['row'])[not_squeak]
                coords = np.asarray(rows['coords']).reshape(-1, 2)[not_squeak]
            else:
                usv_indices, coords = self._embed_session(
                    model, cfg, model['length_threshold'], root, h5_loc, usv_df, usv_summary_loc,
                    f"{prefix}1/{prefix}2", is_pure_squeak,
                )
            self.message_output(f"{prefix}1/{prefix}2: {len(usv_indices)} of {usv_df.height} USVs placed.")
            placed_coords = np.asarray(coords, dtype=np.float64).reshape(-1, 2)
            coordinate_frames.append(pls.DataFrame({
                "_usv_row": np.asarray(usv_indices).astype(np.uint32),
                f"{prefix}1": placed_coords[:, 0],
                f"{prefix}2": placed_coords[:, 1],
            }))

        # Provenance of these models is kept outside the summary, so the legacy
        # qlvm_model column (written by the retired single-model run; older summaries
        # may still carry it) goes, together with every earlier coordinate, category and
        # category confidence column of every listed prefix (model_cell_stale_columns):
        # categories read off the old coordinates do not hold for the new ones.
        stale = ["qlvm_model"]
        stale += [column for prefix in models for column in model_cell_stale_columns(prefix)]
        merged = usv_df.drop([column for column in stale if column in usv_df.columns]).with_row_index(name="_usv_row")
        for frame in coordinate_frames:
            merged = merged.join(frame, on="_usv_row", how="left")
        merged = order_usv_summary_columns(merged.drop("_usv_row"))
        # usv_summary.csv holds every other per-USV column too: publish atomically so
        # a failed write leaves the previous file intact instead of a truncated one.
        with atomic_output_path(usv_summary_loc) as tmp_summary_path:
            merged.write_csv(file=str(tmp_summary_path))

        self.message_output(
            f"Merged the torus coordinates of {len(models)} models "
            f"({', '.join(f'{prefix}1/{prefix}2' for prefix in models)}) "
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
        skip_rows: np.ndarray,
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
            ``params``, ``contract`` (dict), ``lattice`` and
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
            ``"qlvm_duration1/qlvm_duration2"``).
        skip_rows (np.ndarray)
            ``(n_rows,)`` boolean, True for summary rows never to embed (the pure
            squeaks); they get null columns whatever their duration.

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
            # skip_rows follows the summary, which a stale session may hold more or
            # fewer rows of than the H5; only the rows both share can be skipped.
            skipped = np.zeros(durations.size, dtype=bool)
            n_shared = min(durations.size, len(skip_rows))
            skipped[:n_shared] = np.asarray(skip_rows, dtype=bool)[:n_shared]
            in_window = (durations > 0) & ~skipped
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
            condition = contract['condition'] if contract['c_dim'] else None
            masked_call_condition = condition is not None and condition['name'] in ('mean_freq', 'loudness')
            require_mask = contract['require_mask']
            # build_session_masks gives a call without mask instances an all-ones mask.
            # A set built with require_mask left such calls out, and their mean frequency
            # or loudness would span the whole call, so those decoders give them null
            # columns. Sets without require_mask (e.g. a build-qlvm-training-set set)
            # trained on them under that all-ones mask, so they are embedded as before.
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

        # Bandwidth, loudness and spectral entropy conditions take each call's raw value
        # from outside the stored spectrogram; a call without one gets null columns.
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
            # The absolute loudness generate-usv-acoustic-features measured from the audio
            # over the same SAM mask union (compute_usv_loudness.session_image_level_db),
            # read instead of re-measured; a call without it gets null columns.
            if "loudness_db" not in usv_df.columns:
                error_message = (
                    f"{usv_summary_loc.name} has no loudness_db column, which this loudness-conditioned decoder "
                    f"is decoded at. Run generate-usv-acoustic-features on the session first."
                )
                raise ValueError(error_message)
            raw_values = usv_df["loudness_db"].cast(pls.Float64).fill_null(np.nan).to_numpy()[usv_indices]
        elif condition is not None and condition['name'] == 'spectral_entropy':
            # The spectral entropy generate-usv-acoustic-features wrote; the contract's
            # training-split min-max scales it (compute_condition_values).
            if "spectral_entropy" not in usv_df.columns:
                error_message = (
                    f"{usv_summary_loc.name} has no spectral_entropy column, which this spectral-entropy-conditioned "
                    f"decoder is decoded at. Run generate-usv-acoustic-features on the session first."
                )
                raise ValueError(error_message)
            raw_values = usv_df["spectral_entropy"].cast(pls.Float64).fill_null(np.nan).to_numpy()[usv_indices]
            # The min-max clamps these to 0 or 1 before the decode rule sees them, so the
            # decode-grid count below cannot see them; they are counted here instead.
            n_outside = int(np.count_nonzero(
                (raw_values < condition['entropy_min']) | (raw_values > condition['entropy_max'])
            ))
            self.message_output(
                f"{n_outside} USVs have a spectral_entropy outside the training range "
                f"[{condition['entropy_min']:.4f}, {condition['entropy_max']:.4f}] and are clamped to it."
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

        # Preprocess identically to the training set (same resize/time-stretch), then
        # normalize the way the decoder's contract says it was fed.
        target_shape = tuple(int(v) for v in cfg['target_shape'])
        resized = stretch_specs(specs, durations, target_shape, cfg['time_stretch'])
        if cfg['masking_type'] == 'sam':
            # A masked set (build-qlvm-training-set with apply_mask) resizes the call and
            # its SAM mask union separately, binarizes the resized mask at 0.5 and
            # multiplies it in; train-qlvm then min-maxes each spectrogram and multiplies
            # the binarized mask in again (train_qlvm.prepare_split). Masking the native
            # spectrogram before the resize instead lets the interpolation smear signal
            # across the mask edge, which the decoder never saw.
            binary_masks = (stretch_specs(masks, durations, target_shape, cfg['time_stretch']) >= 0.5).astype(np.float32)
            inputs = normalize_model_inputs(resized * binary_masks, contract) * binary_masks
        else:
            inputs = normalize_model_inputs(resized, contract)
        data = jnp.asarray(inputs[:, None, :, :])

        # A conditional package decoder is decoded at the value its cell's rule gives
        # each call's own condition value (frozen_condition_values).
        condition_values = None
        if condition is not None:
            masked_resized = None
            if condition['name'] == 'mean_freq':
                # The mean frequency is defined on the native call masked by its SAM
                # mask union, then resized (the definition the packages were built with).
                masked_resized = stretch_specs(specs * masks, durations, target_shape, cfg['time_stretch'])
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


def tidy_session_usv_summary(
    root_directory: str,
    dry_run: bool,
    backup_directory: str | None,
    message_output: Callable = print,
) -> dict:
    """
    Description
    -----------
    Migrates one session's ``*_usv_summary.csv`` to the canonical layout
    (:func:`os_utils.tidy_usv_summary_columns`): drops the obsolete columns
    (``os_utils.USV_SUMMARY_OBSOLETE_COLUMNS`` -- ``qlvm_supercategory``, the
    duration map's labels, the retired ``qlvm_mf`` / ``qlvm_bw`` / ``qlvm_loud``
    maps, the legacy ``qlvm_model``, the retired squeak-detector columns
    ``squeak_probability`` / ``squeak_frame_runs`` / ``call_class`` /
    ``squeak_spans`` / ``n_squeaks`` -- and every ``*_category_agreement`` /
    ``*_category_uncertain`` column) and puts the rest in
    ``os_utils.USV_SUMMARY_COLUMN_ORDER``, any column that order does not list
    kept last in its existing order. No column is created and no row is touched.

    Every column is read and written as text, so the value of every kept cell is
    written back exactly as it was read (no float re-formatting, no type
    inference). A summary already in the canonical layout is left untouched (not
    rewritten). With ``dry_run`` only the report is produced and logged; nothing
    is written. Otherwise, when ``backup_directory`` is given, the original file
    is first copied to ``<backup_directory>/<session>/<file name>`` (an existing
    backup there is never overwritten: the call raises instead, so a second run
    cannot replace the original with an already-tidied copy), and the summary is
    then rewritten atomically.

    Parameters
    ----------
    root_directory (str)
        Session root directory (contains the ``audio`` tree).
    dry_run (bool)
        Report what would change without writing anything.
    backup_directory (str | None)
        Directory to copy the original summary into before it is rewritten; None
        writes no backup.
    message_output (Callable)
        Logging callback; defaults to ``print``.

    Returns
    -------
    report (dict)
        :func:`os_utils.tidy_usv_summary_columns`'s report (``dropped``,
        ``unknown``, ``columns_before``, ``columns_after``, ``changed``) plus
        ``summary_path`` (str), ``n_rows`` (int), ``written`` (bool, True when the
        file was rewritten) and ``backup_path`` (str, or None when no backup was
        written).
    """

    root = pathlib.Path(root_directory)
    usv_summary_loc = first_match_or_raise(
        root=root / "audio",
        pattern="*_usv_summary.csv",
        recursive=True,
        label="USV summary CSV",
    )
    usv_df = pls.read_csv(source=str(usv_summary_loc), infer_schema=False)
    tidied, report = tidy_usv_summary_columns(usv_df)
    report['summary_path'] = str(usv_summary_loc)
    report['n_rows'] = usv_df.height
    report['written'] = False
    report['backup_path'] = None

    mode = "DRY RUN, " if dry_run else ""
    if not report['changed']:
        message_output(f"{root.name}: {mode}{usv_summary_loc.name} is already in the canonical layout; left untouched.")
        return report
    moved = report['columns_after'] != [column for column in report['columns_before'] if column not in report['dropped']]
    dropped_note = str(report['dropped']) if report['dropped'] else "no column"
    unknown_note = f"; not in the canonical order, kept last: {report['unknown']}" if report['unknown'] else ""
    message_output(
        f"{root.name}: {mode}{usv_summary_loc.name} ({usv_df.height} rows): "
        f"{'would drop' if dry_run else 'dropping'} {dropped_note}; "
        f"column order {'changes' if moved else 'unchanged'}{unknown_note}."
    )
    message_output(f"    {'would be' if dry_run else 'now'}: {', '.join(report['columns_after'])}")
    if dry_run:
        return report

    if backup_directory is not None:
        backup_path = pathlib.Path(backup_directory) / root.name / usv_summary_loc.name
        if backup_path.exists():
            error_message = (
                f"{backup_path} already exists; refusing to overwrite a backup (it may hold the original summary). "
                f"Move it away or pick another backup directory."
            )
            raise FileExistsError(error_message)
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(usv_summary_loc, backup_path)
        report['backup_path'] = str(backup_path)
        message_output(f"    original copied to {backup_path}.")
    with atomic_output_path(usv_summary_loc) as tmp_summary_path:
        tidied.write_csv(file=str(tmp_summary_path))
    report['written'] = True
    return report


@click.command(name="infer-qlvm-latents")
@click.option('--root-directory', type=click.Path(exists=True, file_okay=False, dir_okay=True), required=True, help='Session root directory path.')
@click.option('--model-cell', 'model_cells', type=(str, str), multiple=True, default=None, required=False, help='A column prefix and a QLVM model package cell (e.g. --model-cell qlvm_duration .../qlvm_time_stretch/masked_clean/conditionals/cell/duration); repeat once per model. When given, these pairs replace the model_cells setting: the session is placed on the torus of every listed cell and <prefix>1/<prefix>2 are written, and earlier coordinate and category columns of each listed prefix are removed. Without it, the model_cells setting is used (by default the production cells).')
@click.option('--prefer-package-values/--no-prefer-package-values', 'prefer_package_values', default=None, required=False, help='With model cells: take a corpus session\'s coordinates from the package\'s own embedding when its spectrogram H5 is unchanged since the package (SHA-256, row count, durations and mask counts verified), else infer them; --no-prefer-package-values infers every session.')
@click.option('--latent-dim', 'latent_dim', type=int, default=None, required=False, help='Dimensionality of the toroidal latent space; must equal every model cell\'s training contract.')
@click.option('--time-stretch/--no-time-stretch', 'time_stretch', default=None, required=False, help='Whether to time-stretch each spectrogram to the fixed size (matching training preprocessing; true for the production cells) instead of a plain resize; must match every cell\'s training contract.')
@click.option('--masking-type', 'masking_type', type=click.Choice(['sam', 'none']), default=None, required=False, help='Apply SAM mask regions as a masked training set did ("sam", the default; the production cells, phase 9 cells and masked train-qlvm cells) or embed raw spectrograms ("none", the v3 phase 6 and 11 cells); must match every cell\'s training contract. With "sam", a session without a mask group raises.')
@click.option('--target-shape', 'target_shape', nargs=2, type=int, default=None, required=False, help='Output spectrogram (freq, time) shape as two ints, matching the training preprocessing, e.g. --target-shape 128 128.')
@click.option('--length-threshold', 'length_threshold', type=float, default=None, required=False, help='Embed only USVs with duration below this (time bins); when set, must equal every model cell\'s training contract. Unset in the settings (null), each cell\'s contract sets it.')
@click.option('--lattice-batch-size', 'lattice_batch_size', type=int, default=None, required=False, help='Lattice points decoded and scored per block; lower it to cut memory on large lattices.')
@click.option('--data-batch-size', 'data_batch_size', type=int, default=None, required=False, help='Spectrograms whose lattice posteriors are computed together; memory grows with this times the lattice size.')
@click.pass_context
def infer_qlvm_latents_cli(ctx, root_directory, **kwargs) -> None:
    """
    Description
    -----------
    A command-line tool to embed a session's USV spectrograms into the torus of
    every QLVM model package cell of the ``model_cells`` setting (or of the
    ``--model-cell`` pairs, which replace it) and merge the torus coordinates of
    every listed model into its USV summary CSV (the regular map's
    ``qlvm_category`` is written afterwards by ``assign-qlvm-categories``).

    Parameters
    ----------

    Returns
    -------
    None
    """
    provided_params = [key for key in kwargs if ctx.get_parameter_source(key) == ParameterSource.COMMANDLINE]

    # --model-cell pairs become the model_cells object (prefix -> cell), which the
    # generic key-by-key override cannot build; it is written after the others.
    processing_settings_dict = modify_settings_json_for_cli(
        ctx=ctx,
        provided_params=[key for key in provided_params if key != 'model_cells'],
        settings_dict='processing_settings',
        parameters_lists=['target_shape'],
        block='infer_qlvm_latents',
    )
    if 'model_cells' in provided_params:
        processing_settings_dict['infer_qlvm_latents']['model_cells'] = validate_model_cells(kwargs['model_cells'])

    QLVMLatentInference(
        root_directory=root_directory,
        input_parameter_dict=processing_settings_dict,
        message_output=print,
    ).infer_and_merge()


@click.command(name="tidy-usv-summary-columns")
@click.option('--root-directory', 'root_directories', type=click.Path(exists=True, file_okay=False, dir_okay=True), multiple=True, required=False, help='Session root directory path; repeat once per session.')
@click.option('--sessions-file', 'sessions_file', type=click.Path(exists=True, file_okay=True, dir_okay=False), default=None, required=False, help='Text file with one session root directory per line (blank lines and lines starting with # are skipped); added to the --root-directory sessions.')
@click.option('--dry-run', 'dry_run', is_flag=True, default=False, help='Only report, per session, which columns would be dropped and the resulting column order; nothing is written.')
@click.option('--backup-directory', 'backup_directory', type=click.Path(file_okay=False, dir_okay=True), default=None, required=False, help='Copy each original summary to <backup-directory>/<session>/<file name> before rewriting it (an existing backup is never overwritten; that session fails instead).')
def tidy_usv_summary_columns_cli(root_directories, sessions_file, dry_run, backup_directory) -> None:
    """
    Description
    -----------
    A command-line tool to migrate existing USV summary CSVs to the canonical
    column layout (``os_utils.USV_SUMMARY_COLUMN_ORDER``): drops the obsolete
    columns (``os_utils.USV_SUMMARY_OBSOLETE_COLUMNS`` and every
    ``*_category_agreement`` / ``*_category_uncertain`` column) and reorders the
    rest (:func:`tidy_session_usv_summary`). Every session is attempted; the
    sessions that failed are listed at the end and make the command exit with an
    error.

    Parameters
    ----------

    Returns
    -------
    None
    """

    sessions = list(root_directories)
    if sessions_file is not None:
        lines = pathlib.Path(sessions_file).read_text().splitlines()
        sessions += [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]
    if not sessions:
        error_message = "tidy-usv-summary-columns needs at least one session (--root-directory or --sessions-file)."
        raise click.UsageError(error_message)

    n_changed = 0
    failed = []
    for session in sessions:
        try:
            report = tidy_session_usv_summary(
                root_directory=session,
                dry_run=dry_run,
                backup_directory=backup_directory,
                message_output=print,
            )
        except (OSError, pls.exceptions.PolarsError) as error:
            failed.append(f"{session}: {error}")
            print(f"{session}: FAILED -- {error}")
            continue
        n_changed += int(report['changed'])
    print(
        f"{len(sessions)} session(s): {n_changed} {'would change' if dry_run else 'changed'}, "
        f"{len(sessions) - n_changed - len(failed)} already canonical, {len(failed)} failed."
    )
    if failed:
        error_message = "tidy-usv-summary-columns failed for:\n  " + "\n  ".join(failed)
        raise click.ClickException(error_message)
