"""
@author: bartulem
Consolidate per-session spectrogram/mask H5 files and the QLVM columns of their
USV summaries into one multi-session store, in one of two modes
(``consolidate_spectrogram_store.store_mode`` in processing_settings.json, or
``--store-mode`` on the command line; ``STORE_MODES``).

Each session's ``audio/spectrograms/<session>_spectrograms.h5`` (written by
``generate-usv-spectrograms`` + ``generate-usv-masks``) already uses the
consolidated-store layout -- a shared top-level ``frequency_bins`` axis plus
per-session ``spectrogram/<session>`` and ``mask/<session>`` groups -- so in
both modes those groups are group-copied, one session at a time, and every
session is validated before anything is written.

Production mode (``"production"``) archives the CURRENT production QLVM maps:
the five USV maps of ``os_utils.QLVM_MAPS`` (``qlvm``, ``qlvm_duration``,
``qlvm_entropy``, ``qlvm_bandwidth``, ``qlvm_loudness``; cells resolved by
``os_utils.qlvm_map_cell_directory``), the squeak map ``os_utils.QLVM_SQUEAK_MAP``
(``qlvm_squeak``) and the category column ``os_utils.QLVM_CATEGORY_COLUMN``
(``qlvm_category``) with the per-call agreement and uncertain flag of the
category bundle (``os_utils.load_qlvm_category_bundle``), which the summaries do
not carry and the store computes with the assigner's pixel rule
(``qlvm_categories.assign_categories``). Its layout:

* ``qlvm/<session>/<map>`` -- ``(n, 2)`` float64 coordinates ``<map>1`` /
  ``<map>2`` of every USV map and of the squeak map, NaN where the call was not
  embedded (the squeak map is NaN on every row without a squeak);
* ``qlvm/<session>/status`` -- ``(n,)`` int8 bit field of the USV maps'
  embedding rule (``PRODUCTION_STATUS_CODES``): 0 embedded, bit 0 (1) too long
  (duration at or above the regular cell's ``length_threshold``), bit 1 (2) no
  SAM mask, bit 2 (4) pure squeak (``squeak`` true, ``usv`` false), bit 3 (8)
  no call frames (duration 0);
* ``qlvm/<session>/qlvm_category`` -- ``(n,)`` int16 category ``1..k`` (0 where
  the call is not on the regular map);
* ``qlvm/<session>/qlvm_category_agreement`` -- ``(n,)`` float64 consensus
  agreement of the call's pixel (NaN where not on the regular map);
* ``qlvm/<session>/qlvm_category_uncertain`` -- ``(n,)`` bool, agreement below
  the bundle's ``uncertain_agreement``;
* ``qlvm_models/<map>/`` -- provenance attrs of every map's production cell
  (``cell_directory``, ``model_id``, ``checkpoint_sha256``,
  ``training_contract``, ``condition``);
* ``qlvm_categories/`` -- the bundle's ``label_grid`` and ``agreement`` grids
  and its identity, directory, model id, names, descriptions, nomenclature and
  build config as attrs;
* ``sessions`` -- one row per session: id, the SHA-256 of its spectrogram H5
  and of its USV summary, its row count, the number of calls on the regular map
  and on the squeak map.

A production session is refused when its summary lacks a coordinate column of
any map, ``qlvm_category`` or the ``usv`` / ``squeak`` flags; when the regular
map's coordinates are not present exactly where the status is 0 (the rule
``infer-qlvm-latents`` embeds by), a conditional map places a call the status
excludes, or the squeak map places a call without a squeak; or when
``qlvm_category`` is not present exactly where ``qlvm1`` / ``qlvm2`` are or
differs from the bundle's category of the call's pixel (the summary was
labelled with another bundle). The output is
``spectrograms_production_<S>sessions_<N>vocalizations_<UTC timestamp>.h5``.

V3 mode (``"v3"``) writes the archive of the QLVM model package v3
(``V3_PACKAGE_ROOT``, with the cells ``V3_MODEL_CELLS``): everything the
package contributed to the sessions' ``*_usv_summary.csv``, plus the package
tables needed to interpret it:

* ``spectrogram/<session>/qlvm_dim`` -- ``(n, 2)`` float64 coordinates of the
  regular (phase 6) model, ``qlvm1`` / ``qlvm2``, NaN where a call was not
  embedded; kept for the readers that pool ``qlvm_dim`` (the torus-traversal
  video);
* ``qlvm/<session>/<prefix>`` -- ``(n, 2)`` float64 coordinates of every
  production model (``qlvm``, ``qlvm_dur``, ``qlvm_mf``, ``qlvm_bw``,
  ``qlvm_loud``), NaN where not embedded;
* ``qlvm/<session>/<label column>`` -- ``(n,)`` int16 cluster labels
  (``qlvm_category``, ``qlvm_supercategory``, ``qlvm_dur_category``, ...),
  0 where the summary has no label (package labels start at 1);
* ``qlvm/<session>/status`` -- ``(n,)`` int8 per call: 0 embedded, 1 too long
  (duration at or above the regular cell's ``length_threshold``), 2 no SAM
  mask, 3 both;
* ``qlvm_models/<prefix>/`` -- provenance attrs of each model's package cell
  and its cluster tables (``clusters_<level>``, ``boundaries_<level>``,
  ``fine_to_coarse``) and ``label_grid_<level>`` grids;
* ``sessions`` -- one row per session: id, session type, the SHA-256 of its
  spectrogram H5 and of its USV summary, its row count and embedded count.

The v3 store is an archive and stays pinned to that package: the production QLVM
cells (``os_utils.QLVM_PRODUCTION_MODEL_CELLS``: masked, time-stretched regular,
duration, spectral-entropy, bandwidth and loudness conditional cells) have no ``SESSION_H5_BASELINE.tsv``,
``MANIFEST.sha256``, ``recon_mse_breakdown.npz`` or cluster tables, which every
step below needs, and summaries migrated to the canonical layout no longer carry
the v3 ``qlvm_mf*`` / ``qlvm_bw*`` / ``qlvm_loud*`` columns, so such summaries are
refused (missing QLVM columns) rather than mixed into a v3 store; the
production mode above archives the production cells instead.

The v3 store is meant for the package's corpus sessions (``--package-corpus``
resolves them from the package's ``SESSION_H5_BASELINE.tsv``). Every session is
validated before anything is written: its spectrogram H5 must be the one the
package was built from (baseline SHA-256), its summary must have one row per
H5 row and carry the QLVM columns, and the embedded calls must be exactly the
calls the package's rules admit.

The v3 output is written to ``spectrograms_root`` as
``spectrograms_qlvmv3_<S>sessions_<N>vocalizations_<UTC timestamp>.h5``.
``os_utils.resolve_consolidated_h5_path`` picks the newest ``spectrograms_*.h5``
(a production, a v3 or an earlier ``spectrograms_sam2masks_*`` store), so a
fresh consolidation activates on the next read with no configuration change.
"""

from __future__ import annotations

import json
import pathlib
from collections.abc import Callable
from datetime import UTC, datetime

import click
import h5py
import numpy as np
import polars as pls
from click.core import ParameterSource

from ..analyses.usv_interval_archive import _polars_to_h5, git_sha_for_provenance
from ..cli_utils import modify_settings_json_for_cli
from ..os_utils import (
    CATEGORY_CONFIDENCE_SUFFIXES,
    QLVM_CATEGORY_COLUMN,
    QLVM_CATEGORY_MAP,
    QLVM_MAPS,
    QLVM_REGULAR_MAP,
    QLVM_SQUEAK_MAP,
    SQUEAK_FLAG_COLUMN,
    USV_FLAG_COLUMN,
    atomic_output_path,
    call_class_mask,
    cell_cluster_directory,
    configure_path,
    first_match_or_raise,
    load_qlvm_category_bundle,
    pure_squeak_mask,
    qlvm_cell_model_id,
    qlvm_map_cell_directory,
)
from ..time_utils import is_gui_context, smart_wait
from .build_qlvm_training_set import file_sha256
from .build_squeak_spectrogram_store import read_session_lists
from .qlvm_categories import assign_categories
from .qlvm_latents import (
    PACKAGE_BASELINE_NAME,
    cell_file,
    load_package_baseline,
    package_baseline_path,
)

# Cluster-label levels a v3 model package cell holds (inference/clusters_<level>/), and
# the column-name suffix each carries in the summaries this store archives:
# <prefix>_category for the fine level, <prefix>_supercategory for the coarse one
# (qlvm_category / qlvm_supercategory for the regular model's prefix, see
# model_cell_label_column). Kept here: infer-qlvm-latents no longer writes package
# cluster labels, and this store is the only reader of those columns.
LABEL_LEVELS = ("fine", "coarse")
LABEL_LEVEL_SUFFIXES = {"fine": "category", "coarse": "supercategory"}

# The prefix of the regular (unconditional) model: its label columns keep the
# historical names qlvm_category / qlvm_supercategory.
REGULAR_MODEL_PREFIX = "qlvm"


def model_cell_label_column(prefix: str, level: str) -> str:
    """
    Description
    -----------
    Names the cluster-label column a model prefix carries for one label level in
    the summaries this store archives. The regular model's prefix ``"qlvm"`` keeps
    the historical names -- ``"fine"`` -> ``qlvm_category``, ``"coarse"`` ->
    ``qlvm_supercategory`` -- and every other prefix ``P`` gets ``P_category``
    (fine) and ``P_supercategory`` (coarse), e.g. ``qlvm_dur_category``.

    Parameters
    ----------
    prefix (str)
        The column prefix (a key of ``V3_MODEL_CELLS``).
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


def model_cell_label_columns(model_cells: dict[str, str], label_levels: dict[str, list[str]]) -> dict[str, dict[str, str]]:
    """
    Description
    -----------
    Resolves which cluster-label columns each model prefix must carry: the levels
    ``label_levels`` lists for the prefix (a prefix it does not list carries none,
    so ``{}`` asks for no label column), named by :func:`model_cell_label_column`.

    Parameters
    ----------
    model_cells (dict[str, str])
        Prefix -> model cell directory; only the prefixes (and their order) are used.
    label_levels (dict[str, list[str]])
        Prefix -> list of levels among ``LABEL_LEVELS``.

    Returns
    -------
    label_columns (dict[str, dict[str, str]])
        Prefix -> (level -> label column), for every prefix of ``model_cells`` in
        its order, levels in the order ``LABEL_LEVELS`` lists them.

    Raises
    ------
    ValueError
        A listed level is not one of ``LABEL_LEVELS``.
    """
    label_columns = {}
    for prefix in model_cells:
        levels = label_levels[prefix] if prefix in label_levels else []
        invalid = [level for level in levels if level not in LABEL_LEVELS]
        if invalid:
            error_message = f"prefix {prefix!r}: invalid label level(s) {invalid} (allowed: {list(LABEL_LEVELS)})."
            raise ValueError(error_message)
        label_columns[prefix] = {level: model_cell_label_column(prefix, level) for level in LABEL_LEVELS if level in levels}
    return label_columns

# File-name stem of the stores this module writes (the store's layout version).
STORE_NAME_PREFIX = "spectrograms_qlvmv3"

# The QLVM model package v3 the store archives (read-only) and its cells: the
# unconditional phase 6 regular model (qlvm1/qlvm2) and the four phase 11 tail-bin
# conditional models (qlvm_dur, qlvm_mf, qlvm_bw, qlvm_loud), every one the
# natural_5strata_N29000 design, unmasked, floored. Kept here, not in os_utils: the
# production embedding has moved to other cells (os_utils.QLVM_PRODUCTION_MODEL_CELLS)
# that this store's v3 provenance checks cannot describe.
V3_PACKAGE_ROOT = "/mnt/falkner/Dexter/vocal_beh/models/qlvm_models/qlvm_models_latest/v3"
V3_MODEL_CELLS = {
    "qlvm": "phase6_USVs_unmasked_floor/natural_5strata_N29000_unmasked_floor",
    "qlvm_dur": "phase11_cond_duration_floor/natural_5strata_N29000_unmasked_floor",
    "qlvm_mf": "phase11_cond_mean_freq_floor/natural_5strata_N29000_unmasked_floor",
    "qlvm_bw": "phase11_cond_bandwidth_floor/natural_5strata_N29000_unmasked_floor",
    "qlvm_loud": "phase11_cond_loudness_floor/natural_5strata_N29000_unmasked_floor",
}

# Per-call embedding status codes of qlvm/<session>/status (bit 0: too long, bit 1: no SAM mask).
STATUS_EMBEDDED = 0
STATUS_TOO_LONG = 1
STATUS_NO_MASK = 2
STATUS_CODES = {
    STATUS_EMBEDDED: "embedded",
    STATUS_TOO_LONG: "too long (duration >= length_threshold of the regular cell)",
    STATUS_NO_MASK: "no SAM mask",
    STATUS_TOO_LONG | STATUS_NO_MASK: "too long and no SAM mask",
}

# The file at the package root whose SHA-256 identifies the package's exact contents.
PACKAGE_MANIFEST_NAME = "MANIFEST.sha256"

# The two store modes (consolidate_spectrogram_store.store_mode): the production
# QLVM maps, or the v3 model package archive.
STORE_MODE_PRODUCTION = "production"
STORE_MODE_V3 = "v3"
STORE_MODES = (STORE_MODE_PRODUCTION, STORE_MODE_V3)

# File-name stem (and store_layout attr) of the production-mode stores.
PRODUCTION_STORE_NAME_PREFIX = "spectrograms_production"

# Every map whose coordinates a production store carries: the five USV maps, then
# the squeak map.
PRODUCTION_STORE_MAPS = (*QLVM_MAPS, QLVM_SQUEAK_MAP)

# Per-call status bits of a production store's qlvm/<session>/status: the reasons
# the USV maps' embedding rule (infer-qlvm-latents) leaves a call off their tori.
# 0 means the call is embedded on the regular map; any set bit means it is not.
PRODUCTION_STATUS_TOO_LONG = 1
PRODUCTION_STATUS_NO_MASK = 2
PRODUCTION_STATUS_PURE_SQUEAK = 4
PRODUCTION_STATUS_NO_FRAMES = 8
PRODUCTION_STATUS_CODES = {
    STATUS_EMBEDDED: "embedded (no bit set)",
    PRODUCTION_STATUS_TOO_LONG: "bit 0: too long (duration >= length_threshold of the regular cell)",
    PRODUCTION_STATUS_NO_MASK: "bit 1: no SAM mask",
    PRODUCTION_STATUS_PURE_SQUEAK: "bit 2: pure squeak (squeak true, usv false)",
    PRODUCTION_STATUS_NO_FRAMES: "bit 3: no call frames (duration 0)",
}

# Names of the per-call category agreement / uncertain datasets of a production
# store (qlvm_category_agreement, qlvm_category_uncertain): the confidence the
# summary does not carry, named as the columns an earlier assigner wrote.
CATEGORY_AGREEMENT_DATASET = f"{QLVM_CATEGORY_MAP}{CATEGORY_CONFIDENCE_SUFFIXES[0]}"
CATEGORY_UNCERTAIN_DATASET = f"{QLVM_CATEGORY_MAP}{CATEGORY_CONFIDENCE_SUFFIXES[1]}"


def package_corpus_root_directories(package_root: str | pathlib.Path) -> list[str]:
    """
    Description
    -----------
    The session root directories of a QLVM model package's corpus, in the order
    its ``SESSION_H5_BASELINE.tsv`` lists them. The baseline's ``path`` column
    names each session's spectrogram H5 relative to the lab share
    (``<experimenter>/Data/<session>/audio/spectrograms/<session>_spectrograms.h5``);
    the session root is that file's third parent under ``/mnt/falkner``, run
    through ``configure_path`` so it resolves on the host OS (and on the
    cluster).

    Parameters
    ----------
    package_root (str | pathlib.Path)
        The package root (e.g. ``V3_PACKAGE_ROOT``).

    Returns
    -------
    root_directories (list[str])
        One session root directory per baseline row.

    Raises
    ------
    FileNotFoundError
        The package holds no ``SESSION_H5_BASELINE.tsv``.
    ValueError
        The baseline has no ``session`` or ``path`` column, or a path's
        session folder is not the session it is listed for.
    """
    baseline_path = package_baseline_path(pathlib.Path(package_root))
    if baseline_path is None:
        error_message = f"{package_root}: no {PACKAGE_BASELINE_NAME} in it or in its corpus/ folder."
        raise FileNotFoundError(error_message)
    table = pls.read_csv(baseline_path, separator="\t", infer_schema_length=0)
    missing = [column for column in ("session", "path") if column not in table.columns]
    if missing:
        error_message = f"{baseline_path} has no {missing} column(s); its columns are {table.columns}."
        raise ValueError(error_message)
    root_directories = []
    for session_id, relative_path in zip(table["session"].to_list(), table["path"].to_list(), strict=True):
        root = pathlib.PurePosixPath("/mnt/falkner", relative_path).parents[2]
        if root.name != session_id:
            error_message = (
                f"{baseline_path}: the path of session {session_id} ({relative_path}) does not lie in a "
                f"folder named after the session."
            )
            raise ValueError(error_message)
        root_directories.append(configure_path(str(root)))
    return root_directories


def package_session_types(regular_cell: pathlib.Path) -> dict[str, str]:
    """
    Description
    -----------
    Each corpus session's type (``FF``, ``MF_intact``, ``MF_mute``, ``MM`` or
    ``lone_male``) as the package records it per call in the regular cell's
    ``inference/recon_mse_breakdown.npz`` (``session_types``, aligned with
    ``spec_id`` = ``<session>_<H5 row>``). The package README names this array,
    not the corpus ``session_type`` column (which does not separate
    ``MF_intact`` from ``MF_mute``), as the source of a call's session type.

    Parameters
    ----------
    regular_cell (pathlib.Path)
        The regular (phase 6) model's package cell.

    Returns
    -------
    session_types (dict[str, str])
        Session id -> session type.

    Raises
    ------
    ValueError
        A session's calls carry more than one type.
    """
    with np.load(cell_file(regular_cell, "recon_mse_breakdown.npz"), allow_pickle=False) as breakdown:
        spec_id = breakdown["spec_id"].astype(str)
        session_types = breakdown["session_types"].astype(str)
    sessions = np.char.rpartition(spec_id, "_")[:, 0]
    pairs = pls.DataFrame({"session": sessions, "session_type": session_types}).unique(maintain_order=True)
    ambiguous = pairs.filter(pls.col("session").is_duplicated())["session"].unique().to_list()
    if ambiguous:
        error_message = f"{regular_cell}: recon_mse_breakdown.npz gives sessions {sorted(ambiguous)} more than one session type."
        raise ValueError(error_message)
    return dict(zip(pairs["session"].to_list(), pairs["session_type"].to_list(), strict=True))


def model_cell_metadata(package_root: pathlib.Path, relative_cell: str, manifest_sha256: str) -> dict:
    """
    Description
    -----------
    The provenance of one package cell as the store records it in the attrs of
    ``qlvm_models/<prefix>``: the package root and its name and version (the
    root's parent and own folder names, e.g. ``qlvm_models_latest`` / ``v3``),
    the cell path relative to the package, the draw design (the cell name
    without the phase's ``mask_tag`` suffix of ``config/run_config.json``, e.g.
    ``natural_5strata_N29000``), the phase (the cell's first path component),
    the condition the decoder is conditioned on (``none`` for an unconditional
    cell), the SHA-256 of the package's ``MANIFEST.sha256`` and the cell's
    ``training_contract.json`` as a JSON string.

    Parameters
    ----------
    package_root (pathlib.Path)
        The package root.
    relative_cell (str)
        The cell path relative to the package root (a value of
        ``V3_MODEL_CELLS``).
    manifest_sha256 (str)
        SHA-256 of the package's ``MANIFEST.sha256``.

    Returns
    -------
    metadata (dict)
        Attr name -> value (strings), plus ``contract`` (the parsed training
        contract, not written as an attr).
    """
    cell = package_root / relative_cell
    with cell_file(cell, "training_contract.json").open() as contract_file:
        contract = json.load(contract_file)
    with cell_file(cell, "run_config.json").open() as run_config_file:
        run_config = json.load(run_config_file)
    mask_suffix = f"_{run_config['mask_tag']}"
    design = cell.name[: -len(mask_suffix)] if cell.name.endswith(mask_suffix) else cell.name
    return {
        "package_root": str(package_root),
        "package_name": package_root.parent.name,
        "package_version": package_root.name,
        "cell": relative_cell,
        "design": design,
        "phase": pathlib.PurePosixPath(relative_cell).parts[0],
        "condition": contract["conditional"] if contract["conditional"] is not None else "none",
        "manifest_sha256": manifest_sha256,
        "training_contract": json.dumps(contract, sort_keys=True),
        "contract": contract,
    }


def embedding_status(durations: np.ndarray, mask_counts: np.ndarray, length_threshold: float) -> np.ndarray:
    """
    Description
    -----------
    Each call's embedding status under the package's corpus rules: a call is
    embedded (0) when its duration is below the cell contract's
    ``length_threshold`` and it has at least one SAM mask instance; otherwise
    its status sets bit 0 when it is too long (``duration >= length_threshold``)
    and bit 1 when it has no SAM mask, so 1 = too long, 2 = no mask, 3 = both.

    Parameters
    ----------
    durations (np.ndarray)
        ``(n,)`` spectrogram-H5 durations in time bins.
    mask_counts (np.ndarray)
        ``(n,)`` SAM mask instances per row (bincount of
        ``mask/<session>/spectrogram_index``).
    length_threshold (float)
        The regular cell's ``length_threshold``.

    Returns
    -------
    status (np.ndarray)
        ``(n,)`` int8 status codes.
    """
    too_long = (np.asarray(durations) >= length_threshold).astype(np.int8)
    no_mask = (np.asarray(mask_counts) == 0).astype(np.int8)
    return (too_long * STATUS_TOO_LONG + no_mask * STATUS_NO_MASK).astype(np.int8)


def production_embedding_status(
    durations: np.ndarray,
    mask_counts: np.ndarray,
    pure_squeak: np.ndarray,
    length_threshold: float,
) -> np.ndarray:
    """
    Description
    -----------
    Each call's embedding status under the rule ``infer-qlvm-latents`` embeds the
    production USV maps by (the regular cell's training contract: SAM-masked,
    ``require_mask`` true, ``length_threshold``): a call is embedded (0) when it
    has call frames (``duration > 0``), its duration is below
    ``length_threshold``, it has at least one SAM mask instance and it is not a
    pure squeak (``squeak`` true, ``usv`` false; the USV maps skip those). Every
    reason a call is not embedded sets one bit (``PRODUCTION_STATUS_CODES``):
    1 too long, 2 no SAM mask, 4 pure squeak, 8 no call frames, so e.g. 6 is a
    pure squeak without a SAM mask.

    Parameters
    ----------
    durations (np.ndarray)
        ``(n,)`` spectrogram-H5 durations in time bins.
    mask_counts (np.ndarray)
        ``(n,)`` SAM mask instances per row (bincount of
        ``mask/<session>/spectrogram_index``).
    pure_squeak (np.ndarray)
        ``(n,)`` bool, True on the pure squeak rows of the summary
        (``os_utils.pure_squeak_mask``).
    length_threshold (float)
        The regular cell contract's ``length_threshold``.

    Returns
    -------
    status (np.ndarray)
        ``(n,)`` int8 status bit fields.
    """

    durations = np.asarray(durations)
    status = np.zeros(durations.shape[0], dtype=np.int8)
    status[durations >= length_threshold] |= PRODUCTION_STATUS_TOO_LONG
    status[np.asarray(mask_counts) == 0] |= PRODUCTION_STATUS_NO_MASK
    status[np.asarray(pure_squeak, dtype=bool)] |= PRODUCTION_STATUS_PURE_SQUEAK
    status[durations <= 0] |= PRODUCTION_STATUS_NO_FRAMES
    return status


def production_map_provenance(qlvm_map: str) -> dict:
    """
    Description
    -----------
    The provenance of one production map's model cell as a production store
    records it in the attrs of ``qlvm_models/<map>``: the canonical cell
    directory (``os_utils.qlvm_map_cell_directory``), its model id
    (``os_utils.qlvm_cell_model_id``, the ``<package>/<phase>/<cell>`` identifier
    ``processing.qlvm_latents.load_model_cell`` reports), the SHA-256 of its
    ``checkpoint.tar`` (the decoder weights the coordinates came from), the
    condition the decoder is conditioned on (``none`` for an unconditional
    cell) and the cell's ``training_contract.json`` as a JSON string. The cell is
    read on the host mount (``configure_path``).

    Parameters
    ----------
    qlvm_map (str)
        One of ``PRODUCTION_STORE_MAPS`` (a USV map of ``os_utils.QLVM_MAPS`` or
        the squeak map ``os_utils.QLVM_SQUEAK_MAP``).

    Returns
    -------
    metadata (dict)
        Attr name -> value (``map``, ``cell_directory``, ``model_id``,
        ``checkpoint_sha256``, ``condition``, ``training_contract``), plus
        ``contract`` (the parsed training contract, not written as an attr).

    Raises
    ------
    FileNotFoundError
        The cell has no ``checkpoint.tar`` or no ``training_contract.json``.
    """

    cell_directory = qlvm_map_cell_directory(qlvm_map)
    cell = pathlib.Path(configure_path(cell_directory))
    checkpoint_path = cell / "checkpoint.tar"
    if not checkpoint_path.is_file():
        error_message = f"The {qlvm_map} production cell {cell} has no checkpoint.tar."
        raise FileNotFoundError(error_message)
    with cell_file(cell, "training_contract.json").open() as contract_file:
        contract = json.load(contract_file)
    return {
        "map": qlvm_map,
        "cell_directory": cell_directory,
        "model_id": qlvm_cell_model_id(cell_directory),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "condition": contract["conditional"] if contract["conditional"] is not None else "none",
        "training_contract": json.dumps(contract, sort_keys=True),
        "contract": contract,
    }


class SpectrogramStoreConsolidator:

    def __init__(self,
                 root_directories: list[str] | None = None,
                 input_parameter_dict: dict | None = None,
                 message_output: Callable = print,
                 package_root: str | None = None) -> None:
        """
        Description
        -----------
        Initializes the SpectrogramStoreConsolidator class.

        Parameters
        ----------
        root_directories (list[str])
            Session root directories whose per-session spectrogram H5 files and
            USV summaries are consolidated, in the order they should appear in
            the store (e.g. :func:`package_corpus_root_directories`).
        input_parameter_dict (dict)
            Processing settings; ``spectrograms_root`` names the output
            directory the store is written to, ``generate_spectrograms`` is
            recorded in the store's attrs and
            ``consolidate_spectrogram_store.store_mode`` picks the mode
            (``STORE_MODES``: ``"production"`` or ``"v3"``).
        message_output (Callable)
            Defines output messages; defaults to ``print``.
        package_root (str | None)
            The QLVM model package whose models' columns and tables a v3 store
            carries; defaults to ``V3_PACKAGE_ROOT``. Its cells are
            ``V3_MODEL_CELLS``. Not read in production mode.

        Returns
        -------
        None
        """

        self.root_directories = root_directories if root_directories is not None else []
        self.input_parameter_dict = input_parameter_dict if input_parameter_dict is not None else {}
        self.message_output = message_output
        self.package_root = package_root if package_root is not None else V3_PACKAGE_ROOT
        self.app_context_bool = is_gui_context()

    def consolidate_spectrogram_store(self) -> pathlib.Path | None:
        """
        Description
        -----------
        Builds the store in the mode ``consolidate_spectrogram_store.store_mode``
        names: ``"production"`` hands over to
        :meth:`_consolidate_production_store` (layout and checks in the module
        docstring), ``"v3"`` builds the v3 archive (layout in the module
        docstring) in two passes, as follows.

        The first pass reads and validates every session without writing
        anything; every problem is collected and raised together in one
        ``ValueError``: a missing spectrogram H5 or USV summary; a
        ``frequency_bins`` axis that differs from the first session's; a
        summary whose row count is not the H5's; a session absent from the
        package's ``SESSION_H5_BASELINE.tsv`` or whose H5 SHA-256 differs from
        the baseline's (the H5 was rebuilt, so its rows are not the rows the
        package embedded); a session lacking a QLVM coordinate column
        (``<prefix>1`` / ``<prefix>2`` of every production prefix) or a default
        label column (``qlvm_category`` and ``qlvm_supercategory`` for ``qlvm``,
        ``<prefix>_category`` and ``<prefix>_supercategory`` for the others,
        :func:`model_cell_label_columns`); a session without a session type
        in the package; and an embedding that disagrees with the package's rules
        -- the per-call status (:func:`embedding_status`, from the H5 durations
        and the bincount of ``mask/<session>/spectrogram_index``, against the
        regular cell's ``length_threshold``) must be 0 exactly where
        ``qlvm1`` / ``qlvm2`` are non-null, and each model's labels must be
        present exactly where its coordinates are. A label column outside the
        defaults is stored when every session has it.

        The second pass writes the store atomically, copying one session at a
        time so memory stays bounded by one session's arrays.

        Parameters
        ----------

        Returns
        -------
        store_path (pathlib.Path | None)
            ``spectrograms_production_<S>sessions_<N>vocalizations_<ts>.h5``
            (production mode) or
            ``spectrograms_qlvmv3_<S>sessions_<N>vocalizations_<ts>.h5`` (v3
            mode) under ``spectrograms_root``, or None when no session was given.

        Raises
        ------
        ValueError
            ``store_mode`` is not one of ``STORE_MODES``, or a session fails
            validation (nothing is written).
        """

        store_mode = self.input_parameter_dict['consolidate_spectrogram_store']['store_mode']
        if store_mode not in STORE_MODES:
            error_message = f"consolidate_spectrogram_store.store_mode must be one of {list(STORE_MODES)}, got {store_mode!r}."
            raise ValueError(error_message)

        self.message_output(
            f"Spectrogram-store consolidation ({store_mode} mode) started at: {datetime.now().hour:02d}:{datetime.now().minute:02d}:{datetime.now().second:02d}."
        )
        smart_wait(app_context_bool=self.app_context_bool, seconds=1)

        spectrograms_root = pathlib.Path(self.input_parameter_dict['spectrograms_root'])
        if not spectrograms_root.is_dir():
            err_msg = f"spectrograms_root directory '{spectrograms_root}' does not exist."
            raise FileNotFoundError(err_msg)
        if not self.root_directories:
            self.message_output("No session root directories given; nothing to consolidate.")
            return None

        if store_mode == STORE_MODE_PRODUCTION:
            return self._consolidate_production_store(spectrograms_root)

        package_root = pathlib.Path(configure_path(str(self.package_root)))
        baseline = load_package_baseline(package_root)
        manifest_sha256 = file_sha256(package_root / PACKAGE_MANIFEST_NAME)
        models = {
            prefix: model_cell_metadata(package_root, relative_cell, manifest_sha256)
            for prefix, relative_cell in V3_MODEL_CELLS.items()
        }
        length_threshold = float(models[REGULAR_MODEL_PREFIX]['contract']['length_threshold'])
        session_types = package_session_types(package_root / V3_MODEL_CELLS[REGULAR_MODEL_PREFIX])
        default_label_columns = model_cell_label_columns(
            {prefix: str(package_root / relative_cell) for prefix, relative_cell in V3_MODEL_CELLS.items()},
            {},
        )
        coordinate_columns = [f"{prefix}{axis}" for prefix in models for axis in (1, 2)]
        candidate_label_columns = [model_cell_label_column(prefix, level) for prefix in models for level in LABEL_LEVELS]
        self.message_output(
            f"Package {package_root} ({len(baseline)} corpus sessions); models {', '.join(models)}; "
            f"length_threshold {length_threshold:g}."
        )

        # Pass 1: read and validate every session; nothing is written until all pass.
        problems = []
        sessions = []
        frequency_bins = None
        for one_root in self.root_directories:
            root = pathlib.Path(one_root)
            session_id = root.name
            h5_path = root / "audio" / "spectrograms" / f"{session_id}_spectrograms.h5"
            if not h5_path.is_file():
                problems.append(f"{session_id}: no per-session spectrogram H5 at {h5_path}")
                continue
            try:
                summary_path = first_match_or_raise(
                    root=root / "audio",
                    pattern="*_usv_summary.csv",
                    label="USV summary CSV",
                )
            except FileNotFoundError as error:
                problems.append(f"{session_id}: {error}")
                continue
            summary_df = pls.read_csv(
                source=str(summary_path), infer_schema_length=None, schema_overrides={"usv_id": pls.String}
            )
            with h5py.File(h5_path, mode="r") as session_h5:
                session_bins = session_h5["frequency_bins"][:]
                durations = session_h5[f"spectrogram/{session_id}/durations"][:]
                mask_counts = np.zeros(durations.shape[0], dtype=np.int64)
                if f"mask/{session_id}" in session_h5:
                    spectrogram_index = session_h5[f"mask/{session_id}/spectrogram_index"][:].astype(np.int64)
                    mask_counts = np.bincount(spectrogram_index, minlength=durations.shape[0])[: durations.shape[0]]
            n_rows = int(durations.shape[0])
            if frequency_bins is None:
                frequency_bins = session_bins
            elif session_bins.shape != frequency_bins.shape or not np.allclose(frequency_bins, session_bins):
                problems.append(
                    f"{session_id}: its frequency_bins axis differs from the store's shared axis; all sessions "
                    f"must share one spectrogram frequency grid"
                )
            if n_rows != summary_df.height:
                problems.append(
                    f"{session_id}: inconsistent, {n_rows} spectrogram rows but {summary_df.height} USV summary rows"
                )
            h5_sha256 = file_sha256(h5_path)
            if session_id not in baseline:
                problems.append(f"{session_id}: not in the package corpus ({PACKAGE_BASELINE_NAME})")
            elif h5_sha256 != baseline[session_id]:
                problems.append(
                    f"{session_id}: spectrogram H5 SHA-256 {h5_sha256} differs from the package baseline's "
                    f"{baseline[session_id]} (the H5 was rebuilt after the package was made)"
                )
            required = coordinate_columns + [column for columns in default_label_columns.values() for column in columns.values()]
            missing_columns = [column for column in required if column not in summary_df.columns]
            if missing_columns:
                problems.append(f"{session_id}: {summary_path.name} lacks the QLVM column(s) {missing_columns}")
            if session_id not in session_types:
                problems.append(f"{session_id}: the package's regular cell records no session type for it")
            sessions.append({
                "session_id": session_id,
                "h5_path": h5_path,
                "h5_sha256": h5_sha256,
                "summary_sha256": file_sha256(summary_path),
                "n_rows": n_rows,
                "status": embedding_status(durations, mask_counts, length_threshold),
                "qlvm_frame": summary_df.select([
                    column for column in coordinate_columns + candidate_label_columns if column in summary_df.columns
                ]),
            })
            self.message_output(f"Validated '{session_id}' ({n_rows} rows).")

        # A label column is stored when it is a default one (required above) or every session has it.
        label_columns = {prefix: {} for prefix in models}
        for prefix in models:
            for level in LABEL_LEVELS:
                column = model_cell_label_column(prefix, level)
                if level in default_label_columns[prefix] or (
                    sessions and all(column in session['qlvm_frame'].columns for session in sessions)
                ):
                    label_columns[prefix][level] = column

        if not problems:
            for session in sessions:
                problems.extend(self._extract_qlvm_arrays(session, label_columns))
        if problems:
            error_message = (
                f"Spectrogram-store consolidation refused ({len(problems)} problem(s)); nothing was written:\n  "
                + "\n  ".join(problems)
            )
            raise ValueError(error_message)

        n_vocalizations_total = sum(session['n_rows'] for session in sessions)
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%SZ")
        store_name = f"{STORE_NAME_PREFIX}_{len(sessions)}sessions_{n_vocalizations_total}vocalizations_{timestamp}.h5"
        store_path = spectrograms_root / store_name

        # Pass 2: write the store, one session at a time.
        with atomic_output_path(store_path) as tmp_store_path, h5py.File(tmp_store_path, mode="w") as store_h5:
            store_h5.attrs["created_by"] = "consolidate-spectrogram-store"
            store_h5.attrs["created_date"] = datetime.now(UTC).isoformat()
            store_h5.attrs["git_commit"] = git_sha_for_provenance(pathlib.Path(__file__).resolve().parent)
            store_h5.attrs["store_layout"] = STORE_NAME_PREFIX
            store_h5.attrs["n_sessions"] = len(sessions)
            store_h5.attrs["n_vocalizations"] = n_vocalizations_total
            store_h5.attrs["package_root"] = str(package_root)
            store_h5.attrs["package_manifest_sha256"] = manifest_sha256
            store_h5.attrs["generate_spectrograms_settings"] = json.dumps(
                self.input_parameter_dict['generate_spectrograms'], sort_keys=True
            )
            store_h5.attrs["generate_spectrograms_settings_note"] = (
                "The generate_spectrograms processing settings in force when this store was built; the settings each "
                "session's spectrograms were originally generated with were not recorded per session."
            )
            store_h5.create_dataset("frequency_bins", data=frequency_bins, compression="gzip", compression_opts=6)

            qlvm_group = store_h5.create_group("qlvm")
            qlvm_group.attrs["status_codes"] = json.dumps({str(code): meaning for code, meaning in STATUS_CODES.items()})
            qlvm_group.attrs["length_threshold"] = length_threshold
            qlvm_group.attrs["coordinates_not_embedded"] = "NaN"
            qlvm_group.attrs["label_not_embedded"] = 0
            qlvm_group.attrs["coordinate_datasets"] = json.dumps(list(models))
            qlvm_group.attrs["label_datasets"] = json.dumps(
                [column for prefix in models for column in label_columns[prefix].values()]
            )

            for session in sessions:
                session_id = session['session_id']
                with h5py.File(session['h5_path'], mode="r") as session_h5:
                    session_h5.copy(f"spectrogram/{session_id}", store_h5, name=f"spectrogram/{session_id}")
                    if f"mask/{session_id}" in session_h5:
                        session_h5.copy(f"mask/{session_id}", store_h5, name=f"mask/{session_id}")
                    else:
                        self.message_output(f"Session '{session_id}' has no mask group; copied without masks.")
                spectrogram_group = store_h5[f"spectrogram/{session_id}"]
                if "qlvm_dim" in spectrogram_group:
                    del spectrogram_group["qlvm_dim"]
                spectrogram_group.create_dataset(
                    "qlvm_dim", data=session['coordinates'][REGULAR_MODEL_PREFIX], compression="gzip", compression_opts=6
                )
                session_group = qlvm_group.create_group(session_id)
                for prefix in models:
                    session_group.create_dataset(
                        prefix, data=session['coordinates'][prefix], compression="gzip", compression_opts=6
                    )
                    for column in label_columns[prefix].values():
                        session_group.create_dataset(
                            column, data=session['labels'][column], compression="gzip", compression_opts=6
                        )
                session_group.create_dataset("status", data=session['status'], compression="gzip", compression_opts=6)
                self.message_output(f"Consolidated session '{session_id}'.")

            models_group = store_h5.create_group("qlvm_models")
            for prefix, metadata in models.items():
                model_group = models_group.create_group(prefix)
                for key, value in metadata.items():
                    if key != "contract":
                        model_group.attrs[key] = value
                model_group.attrs["label_columns"] = json.dumps(label_columns[prefix])
                cell = package_root / metadata['cell']
                for level in label_columns[prefix]:
                    cluster_directory = cell_cluster_directory(cell, level)
                    _polars_to_h5(model_group, f"clusters_{level}", pls.read_csv(cluster_directory / "clusters.csv"))
                    _polars_to_h5(model_group, f"boundaries_{level}", pls.read_csv(cluster_directory / "boundaries.csv"))
                    model_group.create_dataset(
                        f"label_grid_{level}",
                        data=np.load(cluster_directory / "label_grid.npy", allow_pickle=False).astype(np.int16),
                        compression="gzip",
                        compression_opts=6,
                    )
                    if level == "coarse":
                        _polars_to_h5(model_group, "fine_to_coarse", pls.read_csv(cluster_directory / "fine_to_coarse.csv"))

            _polars_to_h5(store_h5, "sessions", pls.DataFrame({
                "session_id": [session['session_id'] for session in sessions],
                "session_type": [session_types[session['session_id']] for session in sessions],
                "spectrogram_h5_sha256": [session['h5_sha256'] for session in sessions],
                "usv_summary_sha256": [session['summary_sha256'] for session in sessions],
                "n_rows": [session['n_rows'] for session in sessions],
                "n_embedded": [int(np.count_nonzero(session['status'] == STATUS_EMBEDDED)) for session in sessions],
            }, schema={
                "session_id": pls.String, "session_type": pls.String, "spectrogram_h5_sha256": pls.String,
                "usv_summary_sha256": pls.String, "n_rows": pls.Int64, "n_embedded": pls.Int64,
            }))

        self.message_output(
            f"Consolidated {len(sessions)} sessions / {n_vocalizations_total} vocalizations -> {store_path}."
        )
        self.message_output(
            f"Spectrogram-store consolidation ended at: {datetime.now().hour:02d}:{datetime.now().minute:02d}:{datetime.now().second:02d}."
        )
        return store_path

    @staticmethod
    def _extract_qlvm_arrays(session: dict, label_columns: dict[str, dict[str, str]]) -> list[str]:
        """
        Description
        -----------
        Turns one validated session's QLVM summary columns into the arrays the
        store writes and checks them against the package's rules. Coordinates
        become ``(n, 2)`` float64 per prefix (null -> NaN) and labels ``(n,)``
        int16 per label column (null -> 0). The regular model's coordinates
        must be non-null exactly where the status is 0, and every model's
        labels must be present exactly where its coordinates are. The arrays are
        stored in ``session['coordinates']`` / ``session['labels']`` and the
        summary frame is dropped.

        Parameters
        ----------
        session (dict)
            One session entry of the first pass (``qlvm_frame``, ``status``,
            ``session_id``).
        label_columns (dict[str, dict[str, str]])
            Prefix -> (level -> label column) the store writes.

        Returns
        -------
        problems (list[str])
            One message per violated rule (empty when the session is consistent).
        """
        frame = session.pop('qlvm_frame')
        session_id = session['session_id']
        problems = []
        session['coordinates'] = {}
        session['labels'] = {}
        for prefix, columns in label_columns.items():
            coordinates = frame.select(
                pls.col(f"{prefix}1").cast(pls.Float64), pls.col(f"{prefix}2").cast(pls.Float64)
            ).fill_null(np.nan).to_numpy().astype(np.float64)
            null_entries = np.isnan(coordinates)
            placed = ~null_entries.any(axis=1)
            if (null_entries[:, 0] != null_entries[:, 1]).any():
                problems.append(f"{session_id}: {prefix}1 / {prefix}2 are null on different rows")
            session['coordinates'][prefix] = coordinates
            if prefix == REGULAR_MODEL_PREFIX:
                embedded = session['status'] == STATUS_EMBEDDED
                if not np.array_equal(embedded, placed):
                    problems.append(
                        f"{session_id}: {int(np.count_nonzero(embedded & ~placed))} rows with status 0 have no "
                        f"{prefix}1/{prefix}2 and {int(np.count_nonzero(~embedded & placed))} rows with a nonzero "
                        f"status have them; the embedding does not follow the package's rules"
                    )
            for column in columns.values():
                labels = frame[column].cast(pls.Int64).fill_null(0).to_numpy()
                if (labels < 0).any() or (labels > np.iinfo(np.int16).max).any():
                    problems.append(f"{session_id}: {column} holds labels outside 0..{np.iinfo(np.int16).max}")
                if not np.array_equal(labels > 0, placed):
                    problems.append(f"{session_id}: {column} is not present exactly where {prefix}1/{prefix}2 are")
                session['labels'][column] = labels.astype(np.int16)
        return problems

    def _consolidate_production_store(self, spectrograms_root: pathlib.Path) -> pathlib.Path:
        """
        Description
        -----------
        Builds the production store (layout in the module docstring) in two
        passes, from the production QLVM maps (``PRODUCTION_STORE_MAPS``: the
        five USV maps of ``os_utils.QLVM_MAPS`` and the squeak map) and the
        category bundle (``os_utils.load_qlvm_category_bundle``).

        Before any session is read, every map's production cell is described
        (:func:`production_map_provenance`: cell directory, model id,
        checkpoint SHA-256, training contract) and the bundle is loaded; the
        regular cell's contract must set ``require_mask`` (the status rule of
        :func:`production_embedding_status` assumes it) and its
        ``length_threshold`` bounds the status.

        The first pass reads and validates every session without writing
        anything; every problem is collected and raised together in one
        ``ValueError``: a missing spectrogram H5 or USV summary; a
        ``frequency_bins`` axis that differs from the first session's; a
        summary whose row count is not the H5's; a summary lacking a coordinate
        column of any map (``<map>1`` / ``<map>2``), ``qlvm_category`` or the
        ``usv`` / ``squeak`` flags; and every violation
        :meth:`_extract_production_arrays` finds (coordinates against the
        status, the squeak flag and the bundle's categories).

        The second pass writes the store atomically, copying one session's
        spectrogram and mask groups at a time so memory stays bounded by one
        session's arrays.

        Parameters
        ----------
        spectrograms_root (pathlib.Path)
            The existing output directory.

        Returns
        -------
        store_path (pathlib.Path)
            ``spectrograms_production_<S>sessions_<N>vocalizations_<ts>.h5``
            under ``spectrograms_root``.

        Raises
        ------
        ValueError
            The regular cell's contract does not set ``require_mask``, or a
            session fails validation (nothing is written).
        """

        models = {qlvm_map: production_map_provenance(qlvm_map) for qlvm_map in PRODUCTION_STORE_MAPS}
        regular_contract = models[QLVM_REGULAR_MAP]['contract']
        if regular_contract['require_mask'] is not True:
            error_message = (
                f"The regular production cell {models[QLVM_REGULAR_MAP]['cell_directory']} does not set require_mask "
                f"in its training contract; the production store's status rule (a call without a SAM mask is not "
                f"embedded) would not describe its embedding."
            )
            raise ValueError(error_message)
        length_threshold = float(regular_contract['length_threshold'])
        bundle = load_qlvm_category_bundle()
        bundle['uncertain_agreement'] = float(bundle['nomenclature']['uncertain_agreement'])
        qlvm_columns = [
            *(f"{qlvm_map}{axis}" for qlvm_map in PRODUCTION_STORE_MAPS for axis in (1, 2)),
            QLVM_CATEGORY_COLUMN,
        ]
        required_columns = [*qlvm_columns, USV_FLAG_COLUMN, SQUEAK_FLAG_COLUMN]
        model_listing = ", ".join(f"{qlvm_map} ({metadata['model_id']})" for qlvm_map, metadata in models.items())
        self.message_output(
            f"Production maps {model_listing}; length_threshold {length_threshold:g}; "
            f"category bundle {bundle['identity']}."
        )

        # Pass 1: read and validate every session; nothing is written until all pass.
        problems = []
        sessions = []
        frequency_bins = None
        for one_root in self.root_directories:
            root = pathlib.Path(one_root)
            session_id = root.name
            h5_path = root / "audio" / "spectrograms" / f"{session_id}_spectrograms.h5"
            if not h5_path.is_file():
                problems.append(f"{session_id}: no per-session spectrogram H5 at {h5_path}")
                continue
            try:
                summary_path = first_match_or_raise(
                    root=root / "audio",
                    pattern="*_usv_summary.csv",
                    label="USV summary CSV",
                )
            except FileNotFoundError as error:
                problems.append(f"{session_id}: {error}")
                continue
            summary_df = pls.read_csv(
                source=str(summary_path), infer_schema_length=None, schema_overrides={"usv_id": pls.String}
            )
            with h5py.File(h5_path, mode="r") as session_h5:
                session_bins = session_h5["frequency_bins"][:]
                durations = session_h5[f"spectrogram/{session_id}/durations"][:]
                mask_counts = np.zeros(durations.shape[0], dtype=np.int64)
                if f"mask/{session_id}" in session_h5:
                    spectrogram_index = session_h5[f"mask/{session_id}/spectrogram_index"][:].astype(np.int64)
                    mask_counts = np.bincount(spectrogram_index, minlength=durations.shape[0])[: durations.shape[0]]
            n_rows = int(durations.shape[0])
            if frequency_bins is None:
                frequency_bins = session_bins
            elif session_bins.shape != frequency_bins.shape or not np.allclose(frequency_bins, session_bins):
                problems.append(
                    f"{session_id}: its frequency_bins axis differs from the store's shared axis; all sessions "
                    f"must share one spectrogram frequency grid"
                )
            if n_rows != summary_df.height:
                problems.append(
                    f"{session_id}: inconsistent, {n_rows} spectrogram rows but {summary_df.height} USV summary rows"
                )
                continue
            missing_columns = [column for column in required_columns if column not in summary_df.columns]
            if missing_columns:
                problems.append(f"{session_id}: {summary_path.name} lacks the column(s) {missing_columns}")
                continue
            pure_squeak = pure_squeak_mask(summary_df, summary_path.name).to_numpy()
            sessions.append({
                "session_id": session_id,
                "h5_path": h5_path,
                "h5_sha256": file_sha256(h5_path),
                "summary_sha256": file_sha256(summary_path),
                "n_rows": n_rows,
                "status": production_embedding_status(durations, mask_counts, pure_squeak, length_threshold),
                "has_squeak": call_class_mask(summary_df, ("squeak", "both"), summary_path.name).to_numpy(),
                "qlvm_frame": summary_df.select(qlvm_columns),
            })
            self.message_output(f"Validated '{session_id}' ({n_rows} rows).")

        for session in sessions:
            problems.extend(self._extract_production_arrays(session, bundle))
        if problems:
            error_message = (
                f"Spectrogram-store consolidation refused ({len(problems)} problem(s)); nothing was written:\n  "
                + "\n  ".join(problems)
            )
            raise ValueError(error_message)

        n_vocalizations_total = sum(session['n_rows'] for session in sessions)
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%SZ")
        store_name = f"{PRODUCTION_STORE_NAME_PREFIX}_{len(sessions)}sessions_{n_vocalizations_total}vocalizations_{timestamp}.h5"
        store_path = spectrograms_root / store_name

        # Pass 2: write the store, one session at a time.
        with atomic_output_path(store_path) as tmp_store_path, h5py.File(tmp_store_path, mode="w") as store_h5:
            store_h5.attrs["created_by"] = "consolidate-spectrogram-store"
            store_h5.attrs["created_date"] = datetime.now(UTC).isoformat()
            store_h5.attrs["git_commit"] = git_sha_for_provenance(pathlib.Path(__file__).resolve().parent)
            store_h5.attrs["store_layout"] = PRODUCTION_STORE_NAME_PREFIX
            store_h5.attrs["n_sessions"] = len(sessions)
            store_h5.attrs["n_vocalizations"] = n_vocalizations_total
            store_h5.attrs["category_bundle_identity"] = bundle['identity']
            store_h5.attrs["generate_spectrograms_settings"] = json.dumps(
                self.input_parameter_dict['generate_spectrograms'], sort_keys=True
            )
            store_h5.attrs["generate_spectrograms_settings_note"] = (
                "The generate_spectrograms processing settings in force when this store was built; the settings each "
                "session's spectrograms were originally generated with were not recorded per session."
            )
            store_h5.create_dataset("frequency_bins", data=frequency_bins, compression="gzip", compression_opts=6)

            qlvm_group = store_h5.create_group("qlvm")
            qlvm_group.attrs["status_codes"] = json.dumps({str(code): meaning for code, meaning in PRODUCTION_STATUS_CODES.items()})
            qlvm_group.attrs["status_is_bit_field"] = True
            qlvm_group.attrs["length_threshold"] = length_threshold
            qlvm_group.attrs["coordinates_not_embedded"] = "NaN"
            qlvm_group.attrs["category_not_embedded"] = 0
            qlvm_group.attrs["coordinate_datasets"] = json.dumps(list(PRODUCTION_STORE_MAPS))
            qlvm_group.attrs["category_datasets"] = json.dumps(
                [QLVM_CATEGORY_COLUMN, CATEGORY_AGREEMENT_DATASET, CATEGORY_UNCERTAIN_DATASET]
            )

            for session in sessions:
                session_id = session['session_id']
                with h5py.File(session['h5_path'], mode="r") as session_h5:
                    session_h5.copy(f"spectrogram/{session_id}", store_h5, name=f"spectrogram/{session_id}")
                    if f"mask/{session_id}" in session_h5:
                        session_h5.copy(f"mask/{session_id}", store_h5, name=f"mask/{session_id}")
                    else:
                        self.message_output(f"Session '{session_id}' has no mask group; copied without masks.")
                session_group = qlvm_group.create_group(session_id)
                for qlvm_map in PRODUCTION_STORE_MAPS:
                    session_group.create_dataset(
                        qlvm_map, data=session['coordinates'][qlvm_map], compression="gzip", compression_opts=6
                    )
                session_group.create_dataset("status", data=session['status'], compression="gzip", compression_opts=6)
                session_group.create_dataset(
                    QLVM_CATEGORY_COLUMN, data=session['category'], compression="gzip", compression_opts=6
                )
                session_group.create_dataset(
                    CATEGORY_AGREEMENT_DATASET, data=session['agreement'], compression="gzip", compression_opts=6
                )
                session_group.create_dataset(
                    CATEGORY_UNCERTAIN_DATASET, data=session['uncertain'], compression="gzip", compression_opts=6
                )
                self.message_output(f"Consolidated session '{session_id}'.")

            models_group = store_h5.create_group("qlvm_models")
            for qlvm_map, metadata in models.items():
                model_group = models_group.create_group(qlvm_map)
                for key, value in metadata.items():
                    if key != "contract":
                        model_group.attrs[key] = value

            categories_group = store_h5.create_group("qlvm_categories")
            categories_group.attrs["column"] = QLVM_CATEGORY_COLUMN
            categories_group.attrs["map"] = bundle['map']
            categories_group.attrs["model_id"] = bundle['model_id']
            categories_group.attrs["identity"] = bundle['identity']
            categories_group.attrs["directory"] = bundle['directory']
            categories_group.attrs["uncertain_agreement"] = bundle['uncertain_agreement']
            categories_group.attrs["names"] = json.dumps(bundle['names'])
            categories_group.attrs["descriptions"] = json.dumps(bundle['descriptions'])
            categories_group.attrs["nomenclature"] = json.dumps(bundle['nomenclature'], sort_keys=True)
            categories_group.attrs["build_config"] = json.dumps(bundle['build_config'], sort_keys=True)
            categories_group.create_dataset(
                "label_grid", data=bundle['label_grid'].astype(np.int16), compression="gzip", compression_opts=6
            )
            categories_group.create_dataset(
                "agreement", data=bundle['agreement'], compression="gzip", compression_opts=6
            )

            _polars_to_h5(store_h5, "sessions", pls.DataFrame({
                "session_id": [session['session_id'] for session in sessions],
                "spectrogram_h5_sha256": [session['h5_sha256'] for session in sessions],
                "usv_summary_sha256": [session['summary_sha256'] for session in sessions],
                "n_rows": [session['n_rows'] for session in sessions],
                "n_embedded": [int(np.count_nonzero(session['status'] == STATUS_EMBEDDED)) for session in sessions],
                "n_squeak_embedded": [
                    int(np.count_nonzero(~np.isnan(session['coordinates'][QLVM_SQUEAK_MAP][:, 0]))) for session in sessions
                ],
            }, schema={
                "session_id": pls.String, "spectrogram_h5_sha256": pls.String, "usv_summary_sha256": pls.String,
                "n_rows": pls.Int64, "n_embedded": pls.Int64, "n_squeak_embedded": pls.Int64,
            }))

        self.message_output(
            f"Consolidated {len(sessions)} sessions / {n_vocalizations_total} vocalizations -> {store_path}."
        )
        self.message_output(
            f"Spectrogram-store consolidation ended at: {datetime.now().hour:02d}:{datetime.now().minute:02d}:{datetime.now().second:02d}."
        )
        return store_path

    @staticmethod
    def _extract_production_arrays(session: dict, bundle: dict) -> list[str]:
        """
        Description
        -----------
        Turns one validated session's production QLVM summary columns into the
        arrays the production store writes and checks them. Coordinates become
        ``(n, 2)`` float64 per map (null -> NaN); ``<map>1`` and ``<map>2`` must
        be null on the same rows. The regular map must place a call exactly
        where the status is 0, a conditional map only where it is 0 (a
        conditional decoder may also leave out a call without a conditioning
        value) and the squeak map only where the summary's ``squeak`` flag is
        true. ``qlvm_category`` must be present exactly where ``qlvm1`` /
        ``qlvm2`` are and equal the bundle's category of each call's pixel
        (:func:`qlvm_categories.assign_categories`, the assigner's rule), whose
        agreement and uncertain flag are kept too. The arrays are stored in
        ``session['coordinates']``, ``session['category']`` (int16, 0 off the
        regular map), ``session['agreement']`` (float64, NaN off it) and
        ``session['uncertain']`` (bool); the summary frame and the squeak flag
        are dropped.

        Parameters
        ----------
        session (dict)
            One session entry of the first pass (``qlvm_frame``, ``status``,
            ``has_squeak``, ``session_id``).
        bundle (dict)
            The category bundle (``os_utils.load_qlvm_category_bundle``) with
            ``uncertain_agreement`` added.

        Returns
        -------
        problems (list[str])
            One message per violated rule (empty when the session is consistent).
        """

        frame = session.pop('qlvm_frame')
        has_squeak = session.pop('has_squeak')
        session_id = session['session_id']
        embedded = session['status'] == STATUS_EMBEDDED
        problems = []
        session['coordinates'] = {}
        for qlvm_map in PRODUCTION_STORE_MAPS:
            coordinates = frame.select(
                pls.col(f"{qlvm_map}1").cast(pls.Float64), pls.col(f"{qlvm_map}2").cast(pls.Float64)
            ).fill_null(np.nan).to_numpy().astype(np.float64)
            null_entries = np.isnan(coordinates)
            placed = ~null_entries.any(axis=1)
            if (null_entries[:, 0] != null_entries[:, 1]).any():
                problems.append(f"{session_id}: {qlvm_map}1 / {qlvm_map}2 are null on different rows")
            session['coordinates'][qlvm_map] = coordinates
            if qlvm_map == QLVM_REGULAR_MAP:
                if not np.array_equal(embedded, placed):
                    problems.append(
                        f"{session_id}: {int(np.count_nonzero(embedded & ~placed))} rows with status 0 have no "
                        f"{qlvm_map}1/{qlvm_map}2 and {int(np.count_nonzero(~embedded & placed))} rows with a nonzero "
                        f"status have them; the embedding does not follow the production rule"
                    )
            elif qlvm_map == QLVM_SQUEAK_MAP:
                if (placed & ~has_squeak).any():
                    problems.append(
                        f"{session_id}: {int(np.count_nonzero(placed & ~has_squeak))} rows without a squeak have "
                        f"{qlvm_map}1/{qlvm_map}2"
                    )
            elif (placed & ~embedded).any():
                problems.append(
                    f"{session_id}: {int(np.count_nonzero(placed & ~embedded))} rows with a nonzero status have "
                    f"{qlvm_map}1/{qlvm_map}2; the embedding does not follow the production rule"
                )

        assigned = assign_categories(session['coordinates'][QLVM_CATEGORY_MAP], bundle)
        stored = frame[QLVM_CATEGORY_COLUMN].cast(pls.Int64).fill_null(0).to_numpy()
        if not np.array_equal(stored > 0, assigned['placed']):
            problems.append(
                f"{session_id}: {QLVM_CATEGORY_COLUMN} is not present exactly where "
                f"{QLVM_CATEGORY_MAP}1/{QLVM_CATEGORY_MAP}2 are"
            )
        else:
            n_differ = int(np.count_nonzero(stored != assigned['category']))
            if n_differ:
                problems.append(
                    f"{session_id}: {n_differ} calls' {QLVM_CATEGORY_COLUMN} differs from the category bundle's "
                    f"({bundle['identity']}); the summary was labelled with another bundle, run "
                    f"assign-qlvm-categories on it"
                )
        session['category'] = assigned['category'].astype(np.int16)
        session['agreement'] = assigned['agreement'].astype(np.float64)
        session['uncertain'] = assigned['uncertain'].astype(bool)
        return problems


@click.command(name="consolidate-spectrogram-store")
@click.option('--root-directories', type=str, default=None, required=False, help='Comma-separated string of session root directory paths, in store order. Give this and/or --session-lists, or --package-corpus.')
@click.option('--session-lists', 'session_lists', type=str, default=None, required=False, help='Comma-separated string of session-list .txt files (one session root per line, # comments and blank lines skipped), appended after --root-directories in file order. Give this and/or --root-directories, or --package-corpus.')
@click.option('--package-corpus', 'package_corpus', is_flag=True, default=False, help='Consolidate exactly the QLVM model package\'s corpus sessions, in the order of its SESSION_H5_BASELINE.tsv (in production mode only the session list is taken from it). Give this or --root-directories.')
@click.option('--package-root', 'package_root', type=click.Path(exists=True, file_okay=False, dir_okay=True), default=None, required=False, help='QLVM model package root whose models\' columns and tables a v3 store carries, and whose corpus --package-corpus lists; defaults to the v3 package (qlvm_models_latest/v3).')
@click.option('--spectrograms-root', 'spectrograms_root', type=click.Path(exists=True, file_okay=False, dir_okay=True), default=None, required=False, help='Output directory the consolidated store is written to (the spectrograms_root setting when not given).')
@click.option('--store-mode', 'store_mode', type=click.Choice(STORE_MODES), default=None, required=False, help='Store mode: production (the production QLVM maps, squeak map and category bundle) or v3 (the v3 model package archive).')
@click.pass_context
def consolidate_spectrogram_store_cli(ctx, root_directories, session_lists, package_corpus, package_root, spectrograms_root, **kwargs) -> None:
    """
    Description
    -----------
    A command-line tool to consolidate per-session spectrogram/mask H5 files
    and the QLVM columns of their USV summaries into one multi-session store,
    in production or v3 mode (``--store-mode``, else the
    ``consolidate_spectrogram_store.store_mode`` setting), for an explicit list
    of sessions (``--root-directories`` and / or ``--session-lists``, duplicates
    kept once in first-seen order) or for the v3 model package's corpus sessions
    (``--package-corpus``).

    Parameters
    ----------

    Returns
    -------
    None
    """

    if bool(root_directories or session_lists) == package_corpus:
        usage_message = "Give --root-directories and / or --session-lists, or --package-corpus (not both)."
        raise click.UsageError(usage_message)

    provided_params = [key for key in kwargs if ctx.get_parameter_source(key) == ParameterSource.COMMANDLINE]

    processing_settings_dict = modify_settings_json_for_cli(
        ctx=ctx,
        provided_params=provided_params,
        settings_dict='processing_settings',
        block='consolidate_spectrogram_store',
    )
    if spectrograms_root is not None:
        processing_settings_dict['spectrograms_root'] = spectrograms_root

    resolved_package_root = package_root if package_root is not None else V3_PACKAGE_ROOT
    if package_corpus:
        all_paths = package_corpus_root_directories(configure_path(resolved_package_root))
    else:
        all_paths = [one_dir.strip() for one_dir in (root_directories or '').split(',') if one_dir.strip()]
        if session_lists:
            all_paths += read_session_lists([one_list.strip() for one_list in session_lists.split(',') if one_list.strip()])
        all_paths = list(dict.fromkeys(all_paths))

    SpectrogramStoreConsolidator(
        root_directories=all_paths,
        input_parameter_dict=processing_settings_dict,
        message_output=print,
        package_root=resolved_package_root,
    ).consolidate_spectrogram_store()
