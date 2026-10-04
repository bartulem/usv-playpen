"""
@author: bartulem
Configure path to the OS in use and small subprocess/glob helpers shared
across the codebase.
"""

from __future__ import annotations

import contextlib
import json
import os
import pathlib
import platform
import re
import subprocess
import time as _time
from collections.abc import Callable, Iterable, Iterator
from typing import Any, Optional

import numpy as np
import toml

# The lab CUP shares are defined ONCE, in the ``lab_shares`` / ``file_server``
# entries of the host config (``_config/behavioral_experiments_settings.toml``),
# read by ``_host_lab_shares`` below -- the single source also consumed by the
# recording GUI and behavioral_experiments. Each share stores only the
# irreducible tokens: the share ``name`` (falkner), the Windows drive LETTER
# (``F``), and the per-OS mount PARENT (``/Volumes``, ``/mnt``, ``/mnt/cup/labs``
# for ``darwin``/``linux``/``cluster``). ``expand_lab_share`` turns those into
# the full leading mount roots (``F:``, ``/Volumes/falkner``, ...) and the
# ``\\<file_server>\<name>`` UNC. The first share (falkner) is the default
# returned by ``find_base_path``. Only a leading mount root is ever rewritten, so
# look-alike substrings elsewhere in the path are never touched, and additional
# shares (murthy, ...) are handled by construction, not special-casing.
#
# These tokens are read once and expanded by ``_host_lab_shares``; if that config
# is missing/unparseable or lacks a ``lab_shares`` / ``file_server`` entry, path
# translation raises rather than guessing -- a broken or incomplete host config
# should fail loud, not silently fall back to assumed shares.
_OS_KEYS = {"Windows": "windows", "Darwin": "darwin", "Linux": "linux"}


def expand_lab_share(share: dict, file_server: str) -> dict:
    """
    Description
    -----------
    Expands a token-form lab share into its full leading mount roots. The host
    config stores only the irreducible tokens per share (the drive LETTER and the
    per-OS mount PARENT, with the share ``name`` factored out); this appends the
    name (and the ``:`` for Windows) to build each OS's leading mount root, plus
    the ``\\<file_server>\\<name>`` UNC path. It is the single place this
    derivation happens, used by ``_host_lab_shares`` (path translation), the
    recording GUI, and behavioral_experiments.

    Parameters
    ----------
    share (dict)
        Token-form share: ``name`` + ``windows`` (drive letter, e.g. ``F``) +
        ``darwin``/``linux``/``cluster`` mount parents (e.g. ``/Volumes``,
        ``/mnt``, ``/mnt/cup/labs``).
    file_server (str)
        The SMB server name (e.g. ``cup``) -- the ``\\<file_server>\\...`` host.

    Returns
    -------
    expanded (dict)
        ``name`` plus the full leading roots ``windows`` (``F:``), ``darwin``
        (``/Volumes/falkner``), ``linux`` (``/mnt/falkner``), ``cluster``
        (``/mnt/cup/labs/falkner``), and ``unc`` (``\\cup\\falkner``).
    """

    name = share["name"]
    return {
        "name": name,
        "windows": f"{share['windows']}:",
        "darwin": f"{share['darwin']}/{name}",
        "linux": f"{share['linux']}/{name}",
        "cluster": f"{share['cluster']}/{name}",
        "unc": rf"\\{file_server}\{name}",
    }


def recording_destinations(lab_shares, file_server: str, selected_labs, experimenter: str) -> tuple[list[str], list[str]]:
    """
    Description
    -----------
    Builds the per-OS recording destination lists for the selected labs --
    ``<mount root>/<experimenter>/Data`` for each selected share, in both Linux
    and Windows forms. The single place recording destinations are composed,
    used by the recording GUI and behavioral_experiments so the destination
    layout lives in one spot and is never persisted as hardcoded full paths.

    Parameters
    ----------
    lab_shares (iterable of dict)
        Token-form shares (see ``expand_lab_share``).
    file_server (str)
        The SMB server name.
    selected_labs (iterable of str)
        The ``name`` of each lab this host should write recordings to.
    experimenter (str)
        The experimenter folder placed under the share root.

    Returns
    -------
    (linux_destinations, win_destinations) (tuple[list[str], list[str]])
        Parallel lists of full destination directories in Linux and Windows
        form, one per selected lab, in ``lab_shares`` order.
    """

    selected = set(selected_labs)
    linux_destinations = []
    win_destinations = []
    for share in lab_shares:
        if share["name"] in selected:
            roots = expand_lab_share(share, file_server)
            linux_destinations.append(f"{roots['linux']}/{experimenter}/Data")
            win_destinations.append(f"{roots['windows']}\\{experimenter}\\Data")
    return linux_destinations, win_destinations

_HOST_CONFIG_PATH = pathlib.Path(__file__).parent / "_config" / "behavioral_experiments_settings.toml"
# One-element cache for the resolved (shares, file_server); a list so it can be
# populated by mutation (no ``global``). Cleared in tests to force a re-read.
_HOST_SHARES_CACHE: list[tuple[tuple[dict[str, str], ...], str]] = []


def _host_lab_shares() -> tuple[tuple[dict[str, str], ...], str]:
    """
    Description
    -----------
    Returns the host's EXPANDED lab share table and file-server name, resolved
    once and cached for the process. The token-form values are read from the
    ``lab_shares`` and ``file_server`` entries of the host config
    (``_config/behavioral_experiments_settings.toml``) -- the single place the
    drive letters / mount roots are defined, also consumed by the recording GUI
    and behavioral_experiments -- and expanded into full leading roots via
    ``expand_lab_share`` so ``configure_path``/``find_base_path``/
    ``to_cluster_path`` see ready-to-use mount roots. If that file is missing or
    unparseable, or lacks a ``lab_shares`` / ``file_server`` entry, a clear error
    is raised rather than falling back to assumed shares -- a broken or incomplete
    host config fails loud at the first path translation.

    Parameters
    ----------
    None

    Returns
    -------
    (shares, file_server) (tuple[tuple[dict[str, str], ...], str])
        ``shares`` is the ordered per-lab table of EXPANDED shares (first entry =
        default for ``find_base_path``), each a dict with full
        ``name``/``windows``/``darwin``/``linux``/``cluster``/``unc`` roots;
        ``file_server`` is the SMB server name (e.g. ``cup``).
    """

    if _HOST_SHARES_CACHE:
        return _HOST_SHARES_CACHE[0]

    try:
        host_config = toml.load(_HOST_CONFIG_PATH)
    except (OSError, toml.TomlDecodeError) as exc:
        msg = (
            f"Cannot read the host config '{_HOST_CONFIG_PATH}' required to resolve "
            f"the lab CUP shares for path translation: {type(exc).__name__}: {exc}"
        )
        raise RuntimeError(msg) from exc

    if "lab_shares" not in host_config or not host_config["lab_shares"]:
        msg = (
            f"Host config '{_HOST_CONFIG_PATH}' has no non-empty 'lab_shares' table; "
            "cannot resolve the lab CUP share mount roots."
        )
        raise KeyError(msg)
    if "file_server" not in host_config:
        msg = (
            f"Host config '{_HOST_CONFIG_PATH}' has no 'file_server' entry; "
            "cannot form the file-server UNC roots."
        )
        raise KeyError(msg)

    raw_shares = tuple(host_config["lab_shares"])
    file_server = host_config["file_server"]

    expanded = tuple(expand_lab_share(share, file_server) for share in raw_shares)
    _HOST_SHARES_CACHE.append((expanded, file_server))
    return _HOST_SHARES_CACHE[0]


# One-element cache for the resolved experimenter id; a list so it can be
# populated by mutation (no ``global``). Cleared in tests to force a re-read.
_HOST_EXPERIMENTER_CACHE: list[str] = []


def _host_experimenter() -> str:
    """
    Description
    -----------
    Returns the canonical experimenter id used to re-key experimenter-scoped
    paths (in the analysis data / model settings) to the host / CLI experimenter,
    via :func:`rebase_experimenter_in_paths`. It
    is resolved once and cached for the process, from two sources in order:

    1. the ``EXPERIMENTER_ID`` environment variable, when set and non-empty --
       so a cluster / headless run can select the experimenter without editing
       this checkout's ``behavioral_experiments_settings.toml`` (the shared
       convention is to set ``EXPERIMENTER_ID`` at the top of the cluster
       scripts, which export it into the generated SLURM job); used verbatim.
    2. otherwise the top-level ``experimenter`` key of the host config TOML
       (``_config/behavioral_experiments_settings.toml``) -- the same key the
       recording GUI writes and ``exp_id`` is derived from.

    When the environment variable is unset, a missing or unparseable host config,
    or one lacking an ``experimenter`` entry, raises rather than guessing, so a
    broken config fails loud at the first templated-path read.

    Parameters
    ----------
    None

    Returns
    -------
    experimenter (str)
        The experimenter id (e.g. ``Bartul``).
    """

    if _HOST_EXPERIMENTER_CACHE:
        return _HOST_EXPERIMENTER_CACHE[0]

    # An EXPERIMENTER_ID environment variable overrides the host TOML, so a
    # cluster / headless run selects the experimenter without editing this
    # checkout's behavioral_experiments_settings.toml. Used verbatim when
    # non-empty; the host TOML is the fallback.
    if "EXPERIMENTER_ID" in os.environ and os.environ["EXPERIMENTER_ID"].strip():
        _HOST_EXPERIMENTER_CACHE.append(os.environ["EXPERIMENTER_ID"].strip())
        return _HOST_EXPERIMENTER_CACHE[0]

    try:
        host_config = toml.load(_HOST_CONFIG_PATH)
    except (OSError, toml.TomlDecodeError) as exc:
        msg = (
            f"Cannot read the host config '{_HOST_CONFIG_PATH}' required to resolve "
            f"the experimenter id for path templating: {type(exc).__name__}: {exc}"
        )
        raise RuntimeError(msg) from exc

    if "experimenter" not in host_config:
        msg = (
            f"Host config '{_HOST_CONFIG_PATH}' has no 'experimenter' entry; "
            "cannot re-key experimenter-scoped data paths."
        )
        raise KeyError(msg)

    _HOST_EXPERIMENTER_CACHE.append(host_config["experimenter"])
    return _HOST_EXPERIMENTER_CACHE[0]


# One-element cache for the resolved experimenter roster; a list-of-lists so it
# can be populated by mutation (no ``global``). Cleared in tests to force a
# re-read.
_HOST_EXPERIMENTER_LIST_CACHE: list[list] = []


def _host_experimenter_list() -> list:
    """
    Description
    -----------
    Returns the configured experimenter roster from the host config TOML (the
    top-level ``experimenter_list`` key of
    ``_config/behavioral_experiments_settings.toml``), resolved once and cached
    for the process. This is the set of names :func:`rebase_experimenter_in_paths`
    matches (as whole path components / standalone values) when re-keying the
    experimenter-scoped paths in the ``*_settings.json`` files to the host
    experimenter. A missing or unparseable host config, or one lacking an
    ``experimenter_list`` entry, raises rather than guessing, so a broken config
    fails loud rather than silently leaving paths scoped to the wrong person.

    Parameters
    ----------
    None

    Returns
    -------
    experimenter_list (list)
        The configured experimenter names (e.g. ``["Annegret", "Bartul", ...]``).
    """

    if _HOST_EXPERIMENTER_LIST_CACHE:
        return _HOST_EXPERIMENTER_LIST_CACHE[0]

    try:
        host_config = toml.load(_HOST_CONFIG_PATH)
    except (OSError, toml.TomlDecodeError) as exc:
        msg = (
            f"Cannot read the host config '{_HOST_CONFIG_PATH}' required to resolve "
            f"the experimenter roster for path re-keying: {type(exc).__name__}: {exc}"
        )
        raise RuntimeError(msg) from exc

    if "experimenter_list" not in host_config:
        msg = (
            f"Host config '{_HOST_CONFIG_PATH}' has no 'experimenter_list' entry; "
            "cannot re-key experimenter-scoped paths."
        )
        raise KeyError(msg)

    _HOST_EXPERIMENTER_LIST_CACHE.append(list(host_config["experimenter_list"]))
    return _HOST_EXPERIMENTER_LIST_CACHE[0]


def rebase_experimenter_in_paths(obj: object = None,
                                 experimenter_list: list = None,
                                 exp_id: str = None) -> object:
    """
    Description
    -----------
    Recursively rewrites every experimenter reference found in the string
    leaves of a (possibly nested) settings structure to ``exp_id``, so the
    experimenter-scoped paths in the ``*_settings.json`` files follow the
    experimenter in use rather than the shipped default. Used both by the GUI
    (target = the front-page selection) and by the headless CLI loader
    (target = the host ``experimenter`` from the TOML), so the two contexts
    re-key paths identically.

    Every name in ``experimenter_list`` that occurs either as a complete path
    component (bounded by ``/`` or ``\\`` or a string end) or as the entire
    string (e.g. the ``send_email.experimenter`` field) is rewritten to
    ``exp_id``.

    Strings that merely contain a name as an unbounded substring (e.g. a PC
    label such as ``"A84I Linux"``) are left untouched, and a reference that
    already equals ``exp_id`` is a no-op -- so repeated application (on load
    and on every experimenter change) is idempotent.

    Parameters
    ----------
    obj (dict | list | str | object)
        The structure (or leaf) to rewrite; dicts and lists are recursed
        into, strings are rewritten, all other types are returned as-is.
    experimenter_list (list)
        Known experimenter names matched as path components / standalone names.
    exp_id (str)
        The selected experimenter id substituted in place of every match.

    Returns
    -------
    rebased (dict | list | str | object)
        A structurally identical copy with experimenter references rewritten.
    """

    if isinstance(obj, dict):
        return {key: rebase_experimenter_in_paths(value, experimenter_list, exp_id) for key, value in obj.items()}
    if isinstance(obj, list):
        return [rebase_experimenter_in_paths(value, experimenter_list, exp_id) for value in obj]
    if isinstance(obj, str):
        rebased_string = obj
        is_pathlike = ('/' in rebased_string) or ('\\' in rebased_string)
        for name in experimenter_list:
            if name == exp_id or name not in rebased_string:
                continue
            if rebased_string == name:
                return exp_id
            if is_pathlike:
                name_start = rebased_string.find(name)
                boundary_before_ok = name_start == 0 or rebased_string[name_start - 1] in ('/', '\\')
                name_end = name_start + len(name)
                boundary_after_ok = name_end == len(rebased_string) or rebased_string[name_end] in ('/', '\\')
                if boundary_before_ok and boundary_after_ok:
                    rebased_string = rebased_string[:name_start] + exp_id + rebased_string[name_end:]
        return rebased_string
    return obj


# The folder holding the production QLVM model cells the usv_summary.csv torus
# columns come from (read-only). A module constant rather than a
# processing_settings.json key: the GUI and the CLI re-key every experimenter name
# in those settings to the active experimenter (`rebase_experimenter_in_paths`),
# which would rewrite this path under another experimenter's directory to the
# active one's, where no cell exists. The cells do not live under
# `spectrograms_root`, so they are not derived from it either (see
# `derive_spectrogram_model_paths`). The folder is not a full model package: it has
# no SESSION_H5_BASELINE.tsv, MANIFEST.sha256 or corpus embedding, so
# infer-qlvm-latents always infers (the package route needs the baseline).
QLVM_MODEL_PACKAGE_ROOT = "/mnt/falkner/Bartul/PC_transfer/qlvm_time_stretch/masked_clean"

# The production embedding: column prefix -> cell under the folder above. The
# unconditional regular model gives qlvm1/qlvm2; the duration, spectral-entropy,
# bandwidth and loudness conditional models give qlvm_duration1/2, qlvm_entropy1/2,
# qlvm_bandwidth1/2 and qlvm_loudness1/2. Every cell was trained on
# SAM-masked, time-stretched 128 x 128 spectrograms (training_contract.json:
# masking_type "sam", time_stretch true, length_threshold 128, no floor) and
# carries no cluster label grid: the regular map's category column qlvm_category
# is written by assign-qlvm-categories from a build-qlvm-categories directory, and
# the conditional maps carry coordinates only.
QLVM_PRODUCTION_MODEL_CELLS = {
    "qlvm": "cell/masked",
    "qlvm_duration": "conditionals/cell/duration",
    "qlvm_entropy": "conditionals/cell/spectral_entropy",
    "qlvm_bandwidth": "conditionals/cell/bandwidth",
    "qlvm_loudness": "conditionals/cell/loudness",
}

# The production squeak (broadband vocalization) embedding, written by
# infer-qlvm-squeak-latents as qlvm_squeak1/qlvm_squeak2: the train-qlvm cell
# stretch_nofloor of the squeak package (11,000 class-balanced squeak crops, 3-30 kHz,
# 128 linear bins, envelope +/- 2 frames, time-stretched to 128 frames, unmasked, no
# loudness floor; its config/training_contract.json records masking_type "none",
# time_stretch true and floor null, and infer-qlvm-squeak-latents reads and checks
# them). It replaces the phase 3 BBV cell natural_session_N11000_nomask (zero-padded,
# un-stretched), which the step still reads in its old package layout when named
# explicitly. A constant for the same reason as the package root above.
QLVM_SQUEAK_PACKAGE_ROOT = "/mnt/falkner/Bartul/PC_transfer/qlvm_final/squeaks"
QLVM_SQUEAK_PRODUCTION_CELL = "cell/stretch_nofloor"

# The spectrogram preprocessing every production cell was trained with (their
# training_contract.json `masking_type` and `time_stretch`): SAM-masked
# spectrograms, time-stretched to the target shape.
QLVM_PRODUCTION_MASKING_TYPE = "sam"
QLVM_PRODUCTION_TIME_STRETCH = True

# The QLVM maps a visualization or analysis can use, one per production model: the
# column prefix of QLVM_PRODUCTION_MODEL_CELLS. A map `P` places calls at `P1`/`P2`;
# the visualizations pick one with `shared_resources.qlvm_map` in
# visualizations_settings.json.
QLVM_MAPS = tuple(QLVM_PRODUCTION_MODEL_CELLS)

# The regular (unconditional) QLVM map: the only map whose decoder is a function of
# the torus position alone (the geodesic pullback metric and the manifold filter
# atlas decode with its cell) and the map the category bundle is defined on.
QLVM_REGULAR_MAP = "qlvm"

# The QLVM category bundle: the content-ridge categories of the regular map, written
# once by build-qlvm-categories (processing.qlvm_categories) from the regular map's
# corpus embedding (read-only). It holds `category_grids.npz` (the periodic label
# grid of the categories, 1..k meaning R-1..R-k, indexed [y, x] over the unit torus,
# plus the per-pixel bootstrap agreement, the full-data reference grid, the
# content-change field, the watershed basins and the smoothed call density),
# `category_nomenclature.json` (names, descriptions, call counts and shares, label
# positions), `category_call_labels.csv` and `build_config.json` (the settings and
# inputs of the build). It is the ONE source of category geometry: every consumer
# that draws category boundaries, places category centres or reads region arrays
# (the neuronal tuning watersheds, the sequence figure, the embedding thumbnails and
# explorer, the torus-traversal video, the manifold filter atlas, the category
# embedding panel) loads it through `load_qlvm_category_bundle`, and
# assign-qlvm-categories labels the summaries with it by default
# (`derive_spectrogram_model_paths`). A module constant rather than a settings path
# for the reason given at QLVM_MODEL_PACKAGE_ROOT: the GUI and the CLI re-key every
# experimenter folder in the settings to the active experimenter.
QLVM_CATEGORY_BUNDLE_DIRECTORY = "/mnt/falkner/Bartul/PC_transfer/qlvm_time_stretch/regions/clustering_clean/category_bundle"

# The files of a category bundle (build-qlvm-categories writes them, every reader
# finds them by these names).
QLVM_CATEGORY_GRIDS_NAME = "category_grids.npz"
QLVM_CATEGORY_NOMENCLATURE_NAME = "category_nomenclature.json"
QLVM_CATEGORY_CALL_LABELS_NAME = "category_call_labels.csv"
QLVM_CATEGORY_BUILD_CONFIG_NAME = "build_config.json"

# The QLVM map the category bundle is defined on: the regular map. Its grid is a
# partition of THAT torus only, so it draws boundaries / centres on the regular map
# alone; a conditional map (qlvm_duration, qlvm_entropy, qlvm_bandwidth,
# qlvm_loudness) places the same calls elsewhere, so
# a figure of a conditional map draws no category boundaries and shows each call's
# category through its qlvm_category label instead.
QLVM_CATEGORY_MAP = QLVM_REGULAR_MAP

# The one categorical QLVM label column of the summary, qlvm_category: the
# category-bundle category of each call's position on the regular map (1..k meaning
# R-1..R-k; 4 in production), written by assign-qlvm-categories. There is no coarse
# level and no category column of the conditional maps: an analysis of ANY map
# (qlvm, qlvm_duration, qlvm_entropy, qlvm_bandwidth, qlvm_loudness) that needs a
# per-call region / category label reads
# this column.
QLVM_CATEGORY_COLUMN = f"{QLVM_CATEGORY_MAP}_category"
QLVM_CATEGORY_COLUMNS = (QLVM_CATEGORY_COLUMN,)

# File name of the cohort pooled-embeddings parquet cache under
# `<spectrograms_dir>/embeddings/`. Its summaries fingerprint makes a cache pooled
# from older summaries rebuild when the summaries change.
POOLED_EMBEDDINGS_CACHE_NAME = "pooled_embeddings_qlvmv3.parquet"


def qlvm_production_cell_directory(qlvm_map: str) -> str:
    """
    Description
    -----------
    The production QLVM model package cell of one map, in its canonical
    ``/mnt/falkner`` form: ``QLVM_PRODUCTION_MODEL_CELLS[qlvm_map]`` under
    ``QLVM_MODEL_PACKAGE_ROOT`` (``.../masked_clean/cell/masked`` for the regular
    map). Every reader of a production cell that is not
    ``infer-qlvm-latents`` (the torus geodesic pullback metric, the manifold filter
    atlas decoder, the torus-traversal video's provenance check) takes it from here,
    so no settings path the experimenter re-keying rewrites can point them at
    another cell. Callers translate it to the host mount with ``configure_path``.

    Parameters
    ----------
    qlvm_map (str)
        One of ``QLVM_MAPS``.

    Returns
    -------
    cell_directory (str)
        The canonical cell path (not checked for existence).

    Raises
    ------
    ValueError
        ``qlvm_map`` is not one of ``QLVM_MAPS``.
    """

    if qlvm_map not in QLVM_MAPS:
        error_message = f"qlvm_map must be one of {QLVM_MAPS}, got {qlvm_map!r}."
        raise ValueError(error_message)
    return f"{QLVM_MODEL_PACKAGE_ROOT}/{QLVM_PRODUCTION_MODEL_CELLS[qlvm_map]}"


def qlvm_cell_model_id(cell_directory: str | pathlib.Path) -> str:
    """
    Description
    -----------
    The identifier of a QLVM model package cell: the last three components of its
    path (``<package>/<phase>/<cell>``; ``masked_clean/cell/masked`` for the
    production regular cell), the ``model_id`` that
    ``processing.qlvm_latents.load_model_cell`` reports for the same directory.
    Mount-independent, so a cell read on one host compares equal to the same cell
    named in canonical form.

    Parameters
    ----------
    cell_directory (str | pathlib.Path)
        The cell directory, canonical or host-translated.

    Returns
    -------
    model_id (str)
        ``"<package>/<phase>/<cell>"``.
    """

    return "/".join(pathlib.PurePosixPath(str(cell_directory).replace("\\", "/")).parts[-3:])


def read_qlvm_category_bundle(category_directory: str | pathlib.Path) -> dict:
    """
    Description
    -----------
    Reads and checks a category bundle written by ``build-qlvm-categories``
    (``processing.qlvm_categories.QLVMCategoryBuilder``): ``category_grids.npz``,
    ``category_nomenclature.json`` and ``build_config.json``. The checks keep a
    malformed or mismatched bundle from being drawn silently: the label grid must
    be square and hold exactly the labels ``1..k`` of the nomenclature's ``k``
    categories (``grid_label`` ``1..k`` in order), and the agreement and density
    grids must have its shape. Light (NumPy and JSON only), so figures and
    notebooks can read it without the JAX stack ``processing.qlvm_categories``
    imports.

    Parameters
    ----------
    category_directory (str | pathlib.Path)
        The bundle directory (canonical or host form; run through
        ``configure_path``).

    Returns
    -------
    bundle (dict)
        * ``directory`` (str) -- the host-resolved bundle directory;
        * ``label_grid`` (``(res, res)`` int64, ``1..k``, indexed ``[y, x]``:
          pixel ``[y, x]`` covers torus positions
          ``[x / res, (x + 1) / res) x [y / res, (y + 1) / res)``, the pixel
          rule assign-qlvm-categories labels a call with);
        * ``agreement`` (``(res, res)`` float64) -- the consensus share of every
          pixel's category over the session resamples;
        * ``density`` (``(res, res)`` float64) -- the smoothed corpus call count
          per pixel (the regular map's density landscape);
        * ``resolution`` (int);
        * ``axis`` (``(res,)`` float64) -- the pixel centres ``(i + 0.5) / res``,
          the ``X`` / ``Y`` of a contour over the grid;
        * ``centers`` (``(k, 2)`` float64) -- each category's label position
          ``(label_x, label_y)`` (the pixel farthest from its boundary on the
          torus), row ``i`` for category ``i + 1``;
        * ``names`` / ``descriptions`` (list[str]) -- ``R-1`` ... ``R-k`` and their
          short descriptions;
        * ``nomenclature`` / ``build_config`` (dict) -- the two JSON files;
        * ``map`` (str) -- ``QLVM_CATEGORY_MAP``, the map the grid partitions;
        * ``model_id`` (str) -- that map's production cell
          (:func:`qlvm_cell_model_id` of :func:`qlvm_production_cell_directory`);
        * ``identity`` (str) -- a one-line provenance of the bundle (directory,
          build time, positions file SHA-256 and call count from
          ``build_config.json``), for logs and figure records.

    Raises
    ------
    FileNotFoundError
        A bundle file is missing.
    ValueError
        The grids or the nomenclature fail the checks above.
    """

    directory = pathlib.Path(configure_path(str(category_directory)))
    for name in (QLVM_CATEGORY_GRIDS_NAME, QLVM_CATEGORY_NOMENCLATURE_NAME, QLVM_CATEGORY_BUILD_CONFIG_NAME):
        if not (directory / name).is_file():
            error_message = (
                f"QLVM category bundle {directory} has no {name}; point os_utils.QLVM_CATEGORY_BUNDLE_DIRECTORY at a "
                f"build-qlvm-categories output directory."
            )
            raise FileNotFoundError(error_message)
    with np.load(directory / QLVM_CATEGORY_GRIDS_NAME, allow_pickle=False) as grids:
        label_grid = grids['label_grid'].astype(np.int64)
        agreement = grids['agreement'].astype(np.float64)
        density = grids['density'].astype(np.float64)
    with (directory / QLVM_CATEGORY_NOMENCLATURE_NAME).open() as handle:
        nomenclature = json.load(handle)
    with (directory / QLVM_CATEGORY_BUILD_CONFIG_NAME).open() as handle:
        build_config = json.load(handle)
    resolution = label_grid.shape[0]
    if label_grid.ndim != 2 or label_grid.shape[1] != resolution:
        error_message = f"{directory}: label_grid {label_grid.shape} must be a square grid."
        raise ValueError(error_message)
    if agreement.shape != label_grid.shape or density.shape != label_grid.shape:
        error_message = (
            f"{directory}: agreement {agreement.shape} and density {density.shape} must have the label grid's "
            f"shape {label_grid.shape}."
        )
        raise ValueError(error_message)
    categories = nomenclature['categories']
    n_categories = int(nomenclature['n_categories'])
    grid_labels = [int(category['grid_label']) for category in categories]
    if grid_labels != list(range(1, n_categories + 1)):
        error_message = f"{directory}: the nomenclature's grid labels {grid_labels} are not 1..{n_categories}."
        raise ValueError(error_message)
    present = np.unique(label_grid).tolist()
    if present != list(range(1, n_categories + 1)):
        error_message = f"{directory}: the label grid holds the labels {present}, not 1..{n_categories}."
        raise ValueError(error_message)
    built = build_config['built']
    positions_sha = build_config['positions_file_sha256']
    return {
        'directory': str(directory),
        'label_grid': label_grid,
        'agreement': agreement,
        'density': density,
        'resolution': resolution,
        'axis': (np.arange(resolution) + 0.5) / resolution,
        'centers': np.array([[float(category['label_x']), float(category['label_y'])] for category in categories]),
        'names': [str(category['name']) for category in categories],
        'descriptions': [str(category['description']) for category in categories],
        'nomenclature': nomenclature,
        'build_config': build_config,
        'map': QLVM_CATEGORY_MAP,
        'model_id': qlvm_cell_model_id(qlvm_production_cell_directory(QLVM_CATEGORY_MAP)),
        'identity': (
            f"{directory} (built {built}, positions sha256 {positions_sha[:12]}, "
            f"{int(build_config['n_calls'])} calls, {n_categories} categories)"
        ),
    }


def load_qlvm_category_bundle() -> dict:
    """
    Description
    -----------
    Loads THE category bundle, ``QLVM_CATEGORY_BUNDLE_DIRECTORY``, with
    :func:`read_qlvm_category_bundle`. The one entry point every category-grid
    consumer goes through, so they all draw the partition the summaries'
    ``qlvm_category`` column was assigned from. The directory is read from the
    module attribute at call time.

    Parameters
    ----------
    None

    Returns
    -------
    bundle (dict)
        See :func:`read_qlvm_category_bundle`.
    """

    return read_qlvm_category_bundle(QLVM_CATEGORY_BUNDLE_DIRECTORY)


def cell_cluster_directory(cell: pathlib.Path, level: str) -> pathlib.Path:
    """
    Description
    -----------
    Locates one cluster level of a QLVM model package cell in either package layout:
    ``inference/clusters_<level>/`` (v3) or ``cluster/<level>/`` (v2 / v2.1). Kept
    here, free of the JAX stack ``processing.qlvm_latents`` imports, so light
    readers of a cell's ``label_grid.npy`` (e.g. ``consolidate-spectrogram-store``)
    can find it too.

    Parameters
    ----------
    cell (pathlib.Path)
        The package cell directory.
    level (str)
        ``"fine"`` or ``"coarse"``.

    Returns
    -------
    directory (pathlib.Path)
        The first existing candidate.

    Raises
    ------
    FileNotFoundError
        Neither exists.
    """
    candidates = (cell / "inference" / f"clusters_{level}", cell / "cluster" / level)
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    error_message = f"{cell}: no {level} cluster folder ({' or '.join(str(candidate) for candidate in candidates)})."
    raise FileNotFoundError(error_message)


def derive_spectrogram_model_paths(settings: dict = None) -> dict:
    """
    Description
    -----------
    Fills the spectrogram-pipeline model paths from the single
    ``spectrograms_root`` setting, so the user configures one directory instead
    of many. The shipped ``processing_settings.json`` leaves the granular keys
    empty and carries only ``spectrograms_root``; this helper resolves the
    conventional layout beneath it:

    * ``generate_masks.sam2_model_dir``  -> ``<root>/sam``
    * ``generate_masks.sam2_model_path`` -> ``<root>/sam/checkpoint.pt``
    * ``generate_masks.yolo_weights``    -> ``<root>/sam/best.pt``
    * ``detect_usv_squeaks.squeak_model_path`` -> ``<root>/squeak/usv_squeak_timemil_ens5_n2476_20260930_reviewed.pt``
    * ``detect_usv_noise.noise_model_path`` -> ``<root>/noise/noise_timemil_ens5_n4680_20260926.pt``

    A granular key is filled only when it is empty, so an explicit path set in
    the JSON (or via a CLI flag) wins -- the root supplies defaults, it never
    overrides. ``generate_masks.sam2_model_cfg`` is a SAM2 config NAME resolved
    by the installed package's Hydra search path (not a file under the root),
    so it is left untouched. The derived paths keep the canonical
    ``/mnt/falkner`` form; ``configure_path`` translates
    them to the host mount downstream, exactly like the other model paths. The
    mutation is in place and idempotent.

    The QLVM embedding is NOT derived from ``spectrograms_root`` any more: the
    old in-house model under ``<root>/qlvm`` (``qmc_decoder_weights.npz`` and
    its ``arrays_{fine,coarse}.npz`` watershed grids) would re-embed sessions on
    a different torus and write its own ``qlvm_category`` / ``qlvm_supercategory``,
    and ``infer-qlvm-latents`` no longer reads such a decoder at all (model
    package cells are its only models). The production cells do not live under
    ``spectrograms_root`` at all: they are the module constants
    ``QLVM_PRODUCTION_MODEL_CELLS`` under ``QLVM_MODEL_PACKAGE_ROOT``. When
    ``infer_qlvm_latents`` names no model (``model_cells`` empty),
    ``infer_qlvm_latents.model_cells`` is filled with that mapping (prefixes
    ``qlvm``, ``qlvm_duration``, ``qlvm_entropy``, ``qlvm_bandwidth``,
    ``qlvm_loudness``), ``infer_qlvm_latents.masking_type`` is
    set to the cells' trained ``QLVM_PRODUCTION_MASKING_TYPE`` (``"sam"``) and
    ``infer_qlvm_latents.time_stretch`` to their ``QLVM_PRODUCTION_TIME_STRETCH``
    (true). Both are part of the derived model, not separate choices:
    ``infer-qlvm-latents`` checks them against each cell's training contract and
    refuses to embed on a mismatch (the shipped defaults already match; this keeps
    a derived run correct when a user's settings still carry the earlier
    ``"none"`` / false). An explicitly configured ``model_cells`` is left entirely
    alone, ``masking_type`` and ``time_stretch`` included.
    ``infer_qlvm_latents.model_cell_label_levels`` is never touched: its shipped
    ``{}`` writes coordinates only, which is the production layout (the production
    cells carry no label grids; ``qlvm_category`` comes from
    ``assign-qlvm-categories``). Likewise an empty
    ``infer_qlvm_squeak_latents.model_cell_directory`` is filled with the
    production squeak cell ``QLVM_SQUEAK_PRODUCTION_CELL`` under
    ``QLVM_SQUEAK_PACKAGE_ROOT`` (the time-stretched, unmasked, unfloored
    ``train-qlvm`` cell ``squeaks/cell/stretch_nofloor``); an
    explicitly configured squeak cell is left alone. An empty
    ``assign_qlvm_categories.category_directory`` is filled with the category
    bundle ``QLVM_CATEGORY_BUNDLE_DIRECTORY`` and an empty
    ``assign_qlvm_categories.coordinate_prefix`` with the map it is defined on,
    ``QLVM_CATEGORY_MAP`` (``qlvm``), so the summaries' ``qlvm_category`` is
    assigned from the same bundle every figure draws; an explicitly configured
    directory or prefix is left alone.

    Parameters
    ----------
    settings (dict)
        The full processing-settings dictionary. When ``spectrograms_root`` is
        absent or empty the dictionary is returned unchanged (legacy settings
        files that set the granular ``generate_masks`` paths and
        ``infer_qlvm_latents.model_cells`` directly keep working); otherwise the ``generate_masks``,
        ``infer_qlvm_latents``, ``infer_qlvm_squeak_latents``, ``assign_qlvm_categories``,
        ``detect_usv_squeaks`` and ``detect_usv_noise`` blocks must exist.

    Returns
    -------
    settings (dict)
        The same dictionary, with any empty spectrogram-model paths filled in.
    """

    if 'spectrograms_root' not in settings or not settings['spectrograms_root']:
        return settings
    root = settings['spectrograms_root']
    sam_dir = f'{root}/sam'
    squeak_dir = f'{root}/squeak'
    # The noise model file name carries its training: TimeMIL, 5-seed ensemble, 3,562 labels, build date.
    # The call-class (usv / squeak / both) model's likewise: 5-member ensemble, 2,476 non-unsure labels,
    # build date, and "_reviewed" for the label set with the review overrides applied.
    noise_dir = f'{root}/noise'
    derived = (
        ('generate_masks', 'sam2_model_dir', sam_dir),
        ('generate_masks', 'sam2_model_path', f'{sam_dir}/checkpoint.pt'),
        ('generate_masks', 'yolo_weights', f'{sam_dir}/best.pt'),
        ('detect_usv_squeaks', 'squeak_model_path', f'{squeak_dir}/usv_squeak_timemil_ens5_n2476_20260930_reviewed.pt'),
        ('detect_usv_noise', 'noise_model_path', f'{noise_dir}/noise_timemil_ens5_n4680_20260926.pt'),
    )
    for block, key, derived_path in derived:
        if not settings[block][key]:
            settings[block][key] = derived_path
    qlvm_cfg = settings['infer_qlvm_latents']
    if not qlvm_cfg['model_cells']:
        qlvm_cfg['model_cells'] = {prefix: qlvm_production_cell_directory(prefix) for prefix in QLVM_PRODUCTION_MODEL_CELLS}
        qlvm_cfg['masking_type'] = QLVM_PRODUCTION_MASKING_TYPE
        qlvm_cfg['time_stretch'] = QLVM_PRODUCTION_TIME_STRETCH
    squeak_qlvm_cfg = settings['infer_qlvm_squeak_latents']
    if not squeak_qlvm_cfg['model_cell_directory']:
        squeak_qlvm_cfg['model_cell_directory'] = f'{QLVM_SQUEAK_PACKAGE_ROOT}/{QLVM_SQUEAK_PRODUCTION_CELL}'
    category_cfg = settings['assign_qlvm_categories']
    if not category_cfg['category_directory']:
        category_cfg['category_directory'] = QLVM_CATEGORY_BUNDLE_DIRECTORY
    if not category_cfg['coordinate_prefix']:
        category_cfg['coordinate_prefix'] = QLVM_CATEGORY_MAP
    return settings


def find_base_path() -> str | None:
    """
    Description
    -----------
    Returns the primary (falkner) CUP share's mount root for the OS currently
    in use: ``F:\\`` on Windows, ``/Volumes/falkner`` on macOS, ``/mnt/falkner``
    on Linux. Derived from the first entry of the resolved lab-share table
    (``_host_lab_shares``) so the roots are defined in exactly one place.

    Parameters
    ----------
    None

    Returns
    -------
    base_path (str | None)
        The falkner mount root for the host OS, or ``None`` on an unrecognised
        platform (callers must handle the ``None`` case before using the value).
    """

    system = platform.system()
    if system not in _OS_KEYS:
        return None
    base = _host_lab_shares()[0][0][_OS_KEYS[system]]
    return f"{base}\\" if system == "Windows" else base


_ON_CLUSTER_CACHE: list[bool] = []


def _on_cluster() -> bool:
    """
    Description
    -----------
    Reports whether this process is running on the compute cluster (spock/della),
    detected by the PRESENCE OF THE CLUSTER MOUNT rather than a host-name pattern:
    every lab share is mounted under ``/mnt/cup/labs/<lab>`` there, so the primary
    share's ``cluster`` root exists as a directory only where that mount is
    present. This lets ``configure_path`` resolve share paths to their cluster
    form on the cluster without a fragile ``spock*``/``della*`` name match (which
    would rot on renames and break on other clusters) and without relying on
    SLURM env vars (set only inside a running job, missing on the login node).

    The result is cached for the process -- the mount does not appear or disappear
    within a run. If the host share table can not be resolved (e.g. no host config
    on this machine), the answer is False: an unconfigured host is treated as a
    workstation and the host-OS form is used.

    Parameters
    ----------
    None

    Returns
    -------
    (bool)
        True when the primary lab share's cluster mount root is present.
    """

    if _ON_CLUSTER_CACHE:
        return _ON_CLUSTER_CACHE[0]
    try:
        cluster_root = _host_lab_shares()[0][0]["cluster"]
        result = os.path.isdir(cluster_root)
    except (RuntimeError, KeyError, IndexError):
        result = False
    _ON_CLUSTER_CACHE.append(result)
    return result


def configure_path(pa: str) -> str:
    """
    Description
    -----------
    Translates a CUP-share path from whichever OS form it was written in into
    the form expected by the OS currently in use, for any share listed in
    the resolved lab-share table (falkner, murthy, ...).

    Only the leading mount root is rewritten; the remainder of the path is kept
    verbatim apart from normalising the path separator to the target OS
    (``\\`` on Windows, ``/`` elsewhere). A root must be followed by a path
    separator (or be the whole string) to match, so:

    * embedded look-alike substrings are never corrupted -- e.g.
      ``/mnt/falkner/exp_mnt_2025`` on macOS becomes
      ``/Volumes/falkner/exp_mnt_2025`` (the inner ``mnt`` is left alone), and
    * a path that is already in the host-OS form, or that does not begin with
      any known share root, is returned unchanged (passthrough).

    Parameters
    ----------
    pa (str)
        Original path, in any OS's form for a known CUP share, or an unrelated
        path (returned unchanged).

    Returns
    -------
    pa (str)
        OS-converted path, or the original string if no known share root
        matched or the host OS is unrecognised. On the compute cluster (detected
        by the cluster mount's presence, see ``_on_cluster``) the cluster form
        (``/mnt/cup/labs/<lab>/...``) is returned instead, via ``to_cluster_path``.
    """

    # On the compute cluster every lab share is mounted under /mnt/cup/labs/<lab>,
    # a form the host-OS (linux) target never produces; when that mount is present
    # resolve to the cluster form, so a path in ANY host form -- e.g. the canonical
    # /mnt/falkner form the *_settings.json store -- points at where the share
    # actually lives on this machine. to_cluster_path leaves an already-cluster-form
    # path (e.g. an explicitly-passed session root) unchanged.
    if _on_cluster():
        return to_cluster_path(pa)

    system = platform.system()
    if system not in _OS_KEYS:
        return pa
    target_key = _OS_KEYS[system]

    for share in _host_lab_shares()[0]:
        # Only the host-OS forms are translation sources; the non-OS ``cluster``
        # form is handled by ``to_cluster_path`` and must never match here.
        for src_key in _OS_KEYS.values():
            if src_key == target_key:
                continue
            root = share[src_key]
            if pa == root or (pa.startswith(root) and pa[len(root):len(root) + 1] in ("/", "\\")):
                remainder = pa[len(root):]
                remainder = remainder.replace("/", "\\") if target_key == "windows" else remainder.replace("\\", "/")
                return f"{share[target_key]}{remainder}"

    return pa


def resolve_experimenter_path(pa: str) -> str:
    """
    Description
    -----------
    Resolve a shipped data path to the experimenter currently in use: re-key any
    experimenter name in the path (the shipped default -- e.g. ``Bartul``) to the
    host / CLI experimenter via :func:`rebase_experimenter_in_paths`, then
    OS-translate the leading mount root with :func:`configure_path`. This is the
    non-GUI counterpart of the GUI's front-page re-keying: the GUI rebases the
    loaded settings dicts to its selected experimenter, whereas headless callers
    (the CLI, the analysis notebooks, the marimo explorer) resolve one path at a
    time to :func:`_host_experimenter` (the ``EXPERIMENTER_ID`` env var, else the
    host config TOML's ``experimenter`` key).

    Parameters
    ----------
    pa (str)
        A path carrying a shipped experimenter name (e.g.
        ``/mnt/falkner/Bartul/EPHYS``).

    Returns
    -------
    resolved (str)
        The path with its experimenter component re-keyed to the host / CLI
        experimenter and its mount root translated to the host OS.
    """

    rebased = rebase_experimenter_in_paths(pa, _host_experimenter_list(), _host_experimenter())
    return configure_path(rebased)


def find_cluster_path() -> str:
    """
    Description
    -----------
    Returns the primary (falkner) CUP share's **cluster** mount root
    (e.g. ``/mnt/cup/labs/falkner``), regardless of the host OS. This is the
    location of the share when a job runs ON the HPC cluster (where the share
    is not locally mounted under the host-OS root from :func:`find_base_path`).
    Derived from the resolved lab-share table so the cluster root is defined in
    exactly one place (``_config/behavioral_experiments_settings.toml``).

    Parameters
    ----------
    None

    Returns
    -------
    cluster_path (str)
        The falkner cluster mount root.
    """

    return _host_lab_shares()[0][0]["cluster"]


_ANALYSES_SETTINGS_PATH = pathlib.Path(__file__).parent / "_parameter_settings" / "analyses_settings.json"
_MODELING_SETTINGS_PATH = pathlib.Path(__file__).parent / "_parameter_settings" / "modeling_settings.json"


def resolve_modeling_setting(block: str, key: str) -> Any:
    """
    Description
    -----------
    Reads a single configuration value from ``modeling_settings.json[block][key]``.
    The modeling counterpart to :func:`resolve_analyses_setting`, it lets modeling
    modules source a value (e.g. the model-selection significance level, the
    calibration-bin count, the initial session-split tolerance) from the settings
    file as a module-level default instead of hard-coding it as a bare function
    default, so a single shipped value drives every call site.

    Parameters
    ----------
    block (str)
        A top-level block name in ``modeling_settings.json`` (e.g. ``'model_params'``,
        ``'diagnostics'``).
    key (str)
        A key within that block (e.g. ``'selection_p_val'``, ``'ece_n_bins'``).

    Returns
    -------
    value (Any)
        The value stored at ``modeling_settings.json[block][key]``, returned verbatim
        (no path resolution or type coercion is applied).
    """

    with _MODELING_SETTINGS_PATH.open() as settings_file:
        return json.load(settings_file)[block][key]


def resolve_data_root(key: str) -> pathlib.Path:
    """
    Description
    -----------
    Reads a canonical data-location path from
    ``analyses_settings.json['data_roots'][key]`` and resolves it via
    :func:`resolve_experimenter_path`, which re-keys the shipped experimenter
    name in the path to the host / CLI experimenter and translates the leading
    mount root to the host OS. This is the single place the analysis data roots (the EPHYS /
    histology / Data trees, the unit catalog, the aggregator output directory,
    ...) are defined, so they are user-editable configuration that is also
    OS-portable and experimenter-keyed -- rather than hard-coded
    ``/mnt/<experimenter>/...`` constants.

    Parameters
    ----------
    key (str)
        A key under the ``data_roots`` block of ``analyses_settings.json``
        (e.g. ``'ephys_root'``, ``'histology_root'``, ``'catalog_path'``).

    Returns
    -------
    data_root (pathlib.Path)
        The configured path, re-keyed to the host / CLI experimenter and with
        its leading mount root translated to the host OS.
    """

    with _ANALYSES_SETTINGS_PATH.open() as settings_file:
        data_roots = json.load(settings_file)["data_roots"]
    return pathlib.Path(resolve_experimenter_path(data_roots[key]))


def resolve_analyses_setting(block: str, key: str) -> Any:
    """
    Description
    -----------
    Reads a single configuration value from
    ``analyses_settings.json[block][key]``. This is the non-path counterpart to
    :func:`resolve_data_root`: it exposes settings that live outside the
    ``data_roots`` block (e.g. the per-probe hemisphere map, the Kilosort
    version) to modules that would otherwise hard-code them as module-level
    constants, keeping those values user-editable configuration.

    Parameters
    ----------
    block (str)
        A top-level block name in ``analyses_settings.json``
        (e.g. ``'npx_histology_ibl_alignment_export'``).
    key (str)
        A key within that block (e.g. ``'probe_to_hemisphere'``).

    Returns
    -------
    value (Any)
        The value stored at ``analyses_settings.json[block][key]``, returned
        verbatim (no path resolution or type coercion is applied).
    """

    with _ANALYSES_SETTINGS_PATH.open() as settings_file:
        return json.load(settings_file)[block][key]


def ephys_base_for_data_root(data_root_directory: str) -> pathlib.Path:
    """
    Description
    -----------
    Maps a session's ``Data``-tree root directory to the parent of its sibling
    ``EPHYS``-tree directory. The lab mirrors every ``Data`` directory with an
    ``EPHYS`` directory at the same level, so a session stored at
    ``<base>/Data/<session_id>`` keeps its electrophysiology recordings under
    ``<base>/EPHYS/...``.

    Only the final path *component* that is exactly ``Data`` in the **parent**
    of ``data_root_directory`` is swapped for ``EPHYS``; look-alike substrings
    elsewhere in the path -- an experimenter directory such as ``Database``, or
    a session id that merely contains the text ``Data`` -- are therefore never
    corrupted. This replaces the previous unanchored
    ``str(parent).replace('Data', 'EPHYS')`` idiom, which rewrote every
    occurrence of the substring and so was prone to silently mangling such
    paths.

    Parameters
    ----------
    data_root_directory (str)
        Absolute path to a single session's ``Data``-tree root directory, e.g.
        ``/mnt/falkner/Bartul/Data/20230101_120000``.

    Returns
    -------
    ephys_base (pathlib.Path)
        The session root's parent with its final ``Data`` component replaced by
        ``EPHYS``, e.g. ``/mnt/falkner/Bartul/EPHYS``. If the parent contains no
        ``Data`` component it is returned unchanged.
    """

    parent = pathlib.Path(data_root_directory).parent
    parts = list(parent.parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index] == "Data":
            parts[index] = "EPHYS"
            break
    return pathlib.Path(*parts)


def to_cluster_path(pa: str) -> str:
    """
    Description
    -----------
    Translates a CUP-share path written in any host-OS form (Windows drive
    letter, macOS ``/Volumes`` mount, or Linux ``/mnt`` mount) into the form the
    compute cluster (spock/della) uses, where every lab share is mounted under
    ``/mnt/cup/labs/<lab>``. Unlike ``configure_path``, whose target is the host
    OS, the target here is fixed (the cluster), because this is used when
    *submitting* jobs from a workstation to run remotely.

    Both lab shares in the resolved table (falkner, murthy) are handled from every
    host form, so a ``M:\\...`` / ``/Volumes/murthy/...`` / ``/mnt/murthy/...``
    path maps correctly to ``/mnt/cup/labs/murthy/...`` -- the previous
    per-OS ``.replace`` only special-cased falkner on Windows and silently left
    murthy paths unconverted.

    Matching is anchored on a full mount root followed by a separator (or the
    whole string), so embedded look-alike substrings are never corrupted
    (the previous Linux ``.replace('mnt', 'mnt/cup/labs')`` mangled any inner
    ``mnt`` token). Separators in the remainder are normalised to ``/`` because
    the cluster is Linux.

    Parameters
    ----------
    pa (str)
        Original path in any host-OS form for a known CUP share. A path that
        does not begin with a known share root is returned with its separators
        normalised to ``/`` but otherwise unchanged.

    Returns
    -------
    pa (str)
        Cluster-form path (``/mnt/cup/labs/<lab>/...``), or the separator-
        normalised original if no known share root matched.
    """

    normalised = pa.replace("\\", "/")
    for share in _host_lab_shares()[0]:
        cluster_root = share["cluster"]
        for src_key in _OS_KEYS.values():
            root = share[src_key].replace("\\", "/")
            if normalised == root or (normalised.startswith(root) and normalised[len(root):len(root) + 1] == "/"):
                return f"{cluster_root}{normalised[len(root):]}"
    return normalised


@contextlib.contextmanager
def atomic_output_path(final_path: str | pathlib.Path) -> Iterator[pathlib.Path]:
    """
    Description
    -----------
    Context manager for crash-safe, atomic publishing of precious files. It
    yields a temporary sibling path to write into and, on clean exit, replaces
    ``final_path`` with it via ``os.replace`` (an atomic rename on the same
    filesystem).

    Writing irreplaceable data (session metadata, h5 archives, pickles)
    straight to its final path with mode ``'w'`` truncates the existing file
    before the new bytes land, so a crash, kill, or full disk mid-write leaves
    a corrupt or empty file -- and the original is already gone. Writing to a
    sibling temp file and renaming makes the publish all-or-nothing: a reader
    sees either the complete old file or the complete new file, never a partial
    one.

    The temp file is created in the same directory as ``final_path`` so the
    rename stays on one filesystem (cross-filesystem ``os.replace`` is not
    atomic and may fail). If the body raises, the temp file is removed and the
    exception re-raised, leaving any existing ``final_path`` untouched.

    Parameters
    ----------
    final_path (str | pathlib.Path)
        Destination path to publish atomically. Its parent directory must
        already exist (this helper does not create it, matching the callers'
        existing assumption).

    Yields
    ------
    tmp_path (pathlib.Path)
        Sibling temporary path the caller writes its bytes to. After a clean
        exit it has been renamed onto ``final_path`` and no longer exists under
        the temporary name.
    """

    final = pathlib.Path(final_path)
    tmp = final.with_name(f".{final.name}.tmp-{os.getpid()}")
    try:
        yield tmp
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            tmp.unlink()
        raise
    os.replace(tmp, final)


def wait_for_subprocesses(
    subps: Iterable[subprocess.Popen],
    max_seconds: float,
    label: str,
    poll_interval_s: float = 1.0,
    message_output: Optional[Callable] = None,
    raise_on_nonzero: bool = False,
    raise_on_timeout: bool = True,
) -> list[Optional[int]]:
    """
    Description
    -----------
    Polls a collection of subprocess.Popen handles until every one has
    terminated or a timeout is reached. Replaces the previous 'while True:
    poll()' idiom that appeared across the codebase (behavioral_experiments,
    synchronize_files, modify_files, das_inference, anipose_operations), which
    had no timeout and silently ignored non-zero return codes.

    On timeout, still-running subprocesses are terminated (SIGTERM) and given
    a short grace period before being killed (SIGKILL), so the parent process
    does not leave orphaned Popen handles.

    Parameters
    ----------
    subps (Iterable[subprocess.Popen])
        The subprocess handles to wait on. Empty iterables are a no-op.
    max_seconds (float)
        Hard timeout for the entire group. A TimeoutError is raised if the
        group does not finish within this budget (unless raise_on_timeout is
        False, in which case the still-running subprocesses are terminated and
        their slots in the returned list carry whatever return code poll()
        reports after termination -- typically a negative signal code, or None
        only if a process still has not exited at the final poll).
    label (str)
        Short human-readable label used in log / exception messages so the
        caller knows which phase timed out (e.g., 'audio file copy').
    poll_interval_s (float)
        Seconds between successive poll() calls.
    message_output (Callable, optional)
        Function used to surface progress and failure messages. Defaults to
        the built-in print() when None.
    raise_on_nonzero (bool)
        If True, raises RuntimeError when any subprocess exits with a
        non-zero return code.
    raise_on_timeout (bool)
        If True, raises TimeoutError when the group exceeds max_seconds.

    Returns
    -------
    return_codes (list[Optional[int]])
        The return code of each subprocess, in the same order as the input.
        Slots for subprocesses that had to be terminated on timeout carry the
        return code poll() reports after termination (typically a negative
        signal code), or None only if a process still has not exited at the
        final poll.
    """

    log = message_output or print

    subps_list = list(subps)
    if not subps_list:
        return []

    deadline = _time.monotonic() + max_seconds

    while True:
        status = [p.poll() for p in subps_list]
        if all(s is not None for s in status):
            break
        if _time.monotonic() >= deadline:
            still_running_idx = [i for i, s in enumerate(status) if s is None]
            log(
                f"[{label}] timed out after {max_seconds:.0f} s with "
                f"{len(still_running_idx)}/{len(subps_list)} subprocess(es) still running; terminating."
            )
            for i in still_running_idx:
                try:
                    subps_list[i].terminate()
                except OSError:
                    pass
            # Brief grace period for terminate() to take effect
            grace_end = _time.monotonic() + 3
            while _time.monotonic() < grace_end and any(p.poll() is None for p in subps_list):
                _time.sleep(0.25)
            for i in still_running_idx:
                if subps_list[i].poll() is None:
                    try:
                        subps_list[i].kill()
                    except OSError:
                        pass
            if raise_on_timeout:
                raise TimeoutError(
                    f"{label}: {len(still_running_idx)} subprocess(es) did not finish within {max_seconds:.0f} s."
                )
            # refresh status after termination
            status = [p.poll() for p in subps_list]
            break
        _time.sleep(poll_interval_s)

    failed = [(i, s) for i, s in enumerate(status) if s is not None and s != 0]
    if failed:
        failures_str = ", ".join(f"#{i}(rc={s})" for i, s in failed)
        log(f"[{label}] {len(failed)}/{len(subps_list)} subprocess(es) exited with non-zero status: {failures_str}.")
        if raise_on_nonzero:
            raise RuntimeError(
                f"{label}: {len(failed)} subprocess(es) failed — {failures_str}."
            )

    return status


# Canonical column order of a session's ``*_usv_summary.csv`` -- the single source of
# truth every step that writes the summary reorders to (order_usv_summary_columns), so
# a column's position never depends on which step ran last. Six blocks:
#   1. the DAS event: usv_id, start, stop, duration (das_summarize);
#   2. the call-level class labels: noise / noise_probability (detect_usv_noise), then
#      the vocal-class block of detect_usv_squeaks -- the usv / squeak booleans, the
#      three class probabilities p_usv / p_squeak / p_both and the squeak extent
#      squeak_start / squeak_end;
#   3. the emitter (vocal assignment) and the DAS channel statistics peak_amp_ch,
#      mean_amp_ch, chs_count, chs_detected (das_summarize);
#   4. the acoustic descriptors (compute_usv_acoustic_features, including the absolute
#      loudness_db, the spectral_entropy and the SAM mask_number);
#   5. the QLVM maps (infer_qlvm_latents): the regular map qlvm1 / qlvm2 with its
#      content-ridge category qlvm_category (assign_qlvm_categories; 1..k meaning
#      R-1..R-k), then the duration, spectral-entropy, bandwidth and loudness
#      conditional maps qlvm_duration1 / qlvm_duration2, qlvm_entropy1 /
#      qlvm_entropy2, qlvm_bandwidth1 / qlvm_bandwidth2 and qlvm_loudness1 /
#      qlvm_loudness2 (coordinates only);
#   6. the squeak torus coordinates qlvm_squeak1 / qlvm_squeak2
#      (infer_qlvm_squeak_latents; reserved, so no model_cells prefix can write them).
# A column a session does not carry yet is simply absent; a column not listed here
# (an obsolete one, or a new one not yet placed) is kept after the listed ones, in its
# existing order, so a reorder never loses data. USV_SUMMARY_OBSOLETE_COLUMNS lists the
# columns tidy_usv_summary_columns removes from older summaries.
USV_SUMMARY_COLUMN_ORDER = (
    "usv_id", "start", "stop", "duration",
    "noise", "noise_probability", "usv", "squeak", "p_usv", "p_squeak", "p_both", "squeak_start", "squeak_end",
    "emitter", "peak_amp_ch", "mean_amp_ch", "chs_count", "chs_detected",
    "mean_freq_hz", "peak_freq_hz", "freq_bandwidth_hz", "mean_amplitude", "max_amplitude", "loudness_db",
    "spectral_entropy", "mask_number",
    "qlvm1", "qlvm2", "qlvm_category",
    "qlvm_duration1", "qlvm_duration2", "qlvm_entropy1", "qlvm_entropy2",
    "qlvm_bandwidth1", "qlvm_bandwidth2", "qlvm_loudness1", "qlvm_loudness2",
    "qlvm_squeak1", "qlvm_squeak2",
)

# The column prefixes of the QLVM maps the summary holds (block 5 above): the regular
# map ``qlvm`` and its duration (``qlvm_duration``), spectral-entropy
# (``qlvm_entropy``), bandwidth (``qlvm_bandwidth``) and loudness (``qlvm_loudness``)
# conditional maps. infer_qlvm_latents may write ``<prefix>1`` / ``<prefix>2`` of these
# prefixes (and of the QLVM_PRODUCTION_MODEL_CELLS prefixes); every other canonical
# column is reserved for the step that owns it.
QLVM_SUMMARY_MAP_PREFIXES = ("qlvm", "qlvm_duration", "qlvm_entropy", "qlvm_bandwidth", "qlvm_loudness")

# Columns older summaries carry that the canonical layout no longer has, removed by
# tidy_usv_summary_columns (the tidy-usv-summary-columns command): the coarse cluster
# level of the regular map (qlvm_supercategory), the short-prefix duration and
# spectral-entropy maps qlvm_dur / qlvm_ent that the qlvm_duration / qlvm_entropy
# maps replace, with the cluster labels of the duration map (conditional maps carry
# no category columns), the retired v3 mean-frequency (qlvm_mf), bandwidth (qlvm_bw)
# and loudness (qlvm_loud) conditional maps with their labels, the legacy provenance
# column qlvm_model of the retired single-model run, the per-call category
# agreement / uncertain flag of the regular map (kept outside the summary), and the
# columns of the retired squeak detectors that the usv / squeak booleans, the class
# probabilities and the one squeak_start / squeak_end extent replace: the binary
# detector's squeak_probability / squeak_frame_runs and the first three-class
# encoding's call_class / squeak_spans / n_squeaks. ``squeak`` itself is not listed:
# it is a current column (the squeak boolean of detect_usv_squeaks). The names are
# matched EXACTLY (never as prefixes or globs), so the current qlvm_bandwidth1/2 and
# qlvm_loudness1/2 are never taken for the retired qlvm_bw* / qlvm_loud* columns.
USV_SUMMARY_OBSOLETE_COLUMNS = (
    "qlvm_supercategory",
    "qlvm_dur1", "qlvm_dur2", "qlvm_dur_category", "qlvm_dur_supercategory",
    "qlvm_ent1", "qlvm_ent2", "qlvm_ent_category", "qlvm_ent_supercategory",
    "qlvm_mf1", "qlvm_mf2", "qlvm_mf_category", "qlvm_mf_supercategory",
    "qlvm_bw1", "qlvm_bw2", "qlvm_bw_category", "qlvm_bw_supercategory",
    "qlvm_loud1", "qlvm_loud2", "qlvm_loud_category", "qlvm_loud_supercategory",
    "qlvm_model",
    "qlvm_category_agreement", "qlvm_category_uncertain",
    "squeak_probability", "squeak_frame_runs",
    "call_class", "squeak_spans", "n_squeaks",
)

# Suffixes of the per-call category agreement / uncertain columns an earlier
# assign_qlvm_categories wrote for a prefix P (P_category_agreement,
# P_category_uncertain); they never belong in the summary, whatever the prefix.
CATEGORY_CONFIDENCE_SUFFIXES = ("_category_agreement", "_category_uncertain")


# Column `detect_usv_noise` writes: True when the segment holds no vocalization at all.
NOISE_COLUMN = "noise"


def drop_noise_usvs(usv_summary: Any, source: str, message_output: Callable = print) -> tuple[Any, int]:
    """
    Description
    -----------
    Drops the USV segments a trained classifier flagged as noise -- the single definition of "noise"
    every analysis, figure and model shares, so the rule cannot drift between them. A row survives when
    its ``noise`` value is False, or null (a segment too short to score is a detection, not a verdict).

    A summary WITHOUT the column raises rather than passing every row through: the previous convention
    (a category value treated as noise) degraded silently when its column was missing, so analyses went
    on reporting numbers that quietly included noise. Run ``detect-usv-noise`` on the session, or turn
    the filter off in the settings.

    Parameters
    ----------
    usv_summary (polars.DataFrame)
        A session's USV summary table.
    source (str)
        What the table came from (a session id or file name), named in the error.
    message_output (Callable)
        Logging callback; defaults to ``print``.

    Returns
    -------
    kept (polars.DataFrame)
        The rows that are not noise.
    n_dropped (int)
        How many rows were dropped.
    """

    if NOISE_COLUMN not in usv_summary.columns:
        error_message = (
            f"{source} has no '{NOISE_COLUMN}' column, so its noise segments cannot be excluded. Run "
            f"detect-usv-noise on the session, or set the analysis' exclude_noise_usvs to false to keep "
            f"every detection."
        )
        raise KeyError(error_message)
    kept = usv_summary.filter(~usv_summary[NOISE_COLUMN].fill_null(False))
    n_dropped = usv_summary.height - kept.height
    if n_dropped:
        message_output(f"    {source}: dropped {n_dropped} noise segment(s) of {usv_summary.height}.")
    return kept, n_dropped


# Columns `detect_usv_squeaks` writes: two booleans per segment that is not noise -- `usv` (the
# segment holds an ultrasonic call) and `squeak` (it holds a broadband squeak) -- both null on noise
# rows and on rows too short to score. A pure USV is (true, false), a pure squeak (false, true), and a
# segment holding both is (true, true). `usv` alone is NOT "USV only": a pure-USV filter must also
# require `squeak` false, which is what `call_class_mask` / `pure_usv_mask` do.
USV_FLAG_COLUMN = "usv"
SQUEAK_FLAG_COLUMN = "squeak"
VOCAL_FLAG_COLUMNS = (USV_FLAG_COLUMN, SQUEAK_FLAG_COLUMN)

# The three call classes the two booleans encode: "usv" = pure USV (usv & ~squeak), "squeak" = pure
# squeak (squeak & ~usv), "both" = a squeak and a USV in one segment (usv & squeak).
CALL_CLASSES = ("usv", "squeak", "both")

# The three selections every squeak-side consumer (squeak figures, the explorer's squeak map)
# offers: pure squeaks, segments holding a squeak and a USV, or both kinds together (squeak true).
SQUEAK_CLASS_SELECTIONS = {
    "squeak": ("squeak",),
    "both": ("both",),
    "squeak+both": ("squeak", "both"),
}


def require_vocal_flags(usv_summary: Any, source: str) -> None:
    """
    Description
    -----------
    Checks that a USV summary table carries the ``usv`` and ``squeak`` booleans ``detect-usv-squeaks``
    writes, so a consumer that splits USVs from squeaks never silently treats every row as one class.
    A summary scored only by the retired binary squeak detector has a ``squeak`` column but no ``usv``
    (and its ``squeak`` meant something else), so it fails here too.

    Parameters
    ----------
    usv_summary (polars.DataFrame)
        A session's USV summary table (or any table holding its columns).
    source (str)
        What the table came from (a session id or file name), named in the error.

    Returns
    -------
    None

    Raises
    ------
    KeyError
        The table lacks ``usv`` or ``squeak``.
    """

    missing = [column for column in VOCAL_FLAG_COLUMNS if column not in usv_summary.columns]
    if missing:
        error_message = (
            f"{source} has no {missing} column(s), so its USVs cannot be told from its squeaks. "
            f"Run detect-usv-squeaks on the session (after detect-usv-noise)."
        )
        raise KeyError(error_message)


def _vocal_flag(usv_summary: Any, column: str) -> Any:
    """
    Description
    -----------
    One of the two booleans as a null-free boolean Series (null -> False). The column is cast through
    text, so a flag read from CSV as Boolean, as the strings "true" / "false", or as an all-null
    column of any type all give the same answer.

    Parameters
    ----------
    usv_summary (polars.DataFrame)
        A table holding ``column``.
    column (str)
        ``"usv"`` or ``"squeak"``.

    Returns
    -------
    flag (polars.Series)
        Boolean, False where the value is null.
    """

    return usv_summary[column].cast(str).str.to_lowercase().eq("true").fill_null(False)


def call_class_mask(usv_summary: Any, classes: Iterable[str], source: str) -> Any:
    """
    Description
    -----------
    Marks the rows of a USV summary whose call class -- derived from the two booleans, never from
    ``usv`` alone -- is one of ``classes``: ``"usv"`` = pure USV (``usv & ~squeak``), ``"squeak"`` =
    pure squeak (``squeak & ~usv``), ``"both"`` = ``usv & squeak``. A null flag counts as false, so a
    noise row (both null) is in no selection and a mask built here never admits noise.

    Parameters
    ----------
    usv_summary (polars.DataFrame)
        A session's USV summary table (or any table holding ``usv`` and ``squeak``).
    classes (Iterable[str])
        Call classes to keep, each one of ``CALL_CLASSES``.
    source (str)
        What the table came from, named in the error when a flag is missing.

    Returns
    -------
    mask (polars.Series)
        Boolean, one value per row.

    Raises
    ------
    KeyError
        The table lacks ``usv`` or ``squeak``.
    ValueError
        A requested class is not one of ``CALL_CLASSES``.
    """

    classes = list(classes)
    unknown = [value for value in classes if value not in CALL_CLASSES]
    if unknown:
        error_message = f"call_class_mask: unknown call class(es) {unknown}; the classes are {list(CALL_CLASSES)}."
        raise ValueError(error_message)
    require_vocal_flags(usv_summary, source)
    usv = _vocal_flag(usv_summary, USV_FLAG_COLUMN)
    squeak = _vocal_flag(usv_summary, SQUEAK_FLAG_COLUMN)
    by_class = {"usv": usv & ~squeak, "squeak": squeak & ~usv, "both": usv & squeak}
    mask = usv & ~usv
    for value in classes:
        mask = mask | by_class[value]
    return mask.alias("call_class_mask")


def pure_usv_mask(usv_summary: Any, source: str) -> Any:
    """
    Description
    -----------
    Rows holding an ultrasonic call and no squeak (``usv & ~squeak``): what every USV-only consumer
    keeps. Shorthand for ``call_class_mask(usv_summary, ("usv",), source)``.

    Parameters
    ----------
    usv_summary (polars.DataFrame)
        A table holding ``usv`` and ``squeak``.
    source (str)
        What the table came from, named in the error when a flag is missing.

    Returns
    -------
    mask (polars.Series)
        Boolean, one value per row.
    """

    return call_class_mask(usv_summary, ("usv",), source)


def pure_squeak_mask(usv_summary: Any, source: str) -> Any:
    """
    Description
    -----------
    Rows holding a squeak and no ultrasonic call (``squeak & ~usv``). Shorthand for
    ``call_class_mask(usv_summary, ("squeak",), source)``.

    Parameters
    ----------
    usv_summary (polars.DataFrame)
        A table holding ``usv`` and ``squeak``.
    source (str)
        What the table came from, named in the error when a flag is missing.

    Returns
    -------
    mask (polars.Series)
        Boolean, one value per row.
    """

    return call_class_mask(usv_summary, ("squeak",), source)


def both_mask(usv_summary: Any, source: str) -> Any:
    """
    Description
    -----------
    Rows holding a squeak and an ultrasonic call (``usv & squeak``). Shorthand for
    ``call_class_mask(usv_summary, ("both",), source)``.

    Parameters
    ----------
    usv_summary (polars.DataFrame)
        A table holding ``usv`` and ``squeak``.
    source (str)
        What the table came from, named in the error when a flag is missing.

    Returns
    -------
    mask (polars.Series)
        Boolean, one value per row.
    """

    return call_class_mask(usv_summary, ("both",), source)


def squeak_bearing_mask(usv_summary: Any, source: str) -> Any:
    """
    Description
    -----------
    Rows holding a squeak, with or without an ultrasonic call (``squeak`` true: pure squeaks and
    "both"): the rows the squeak QLVM embedding, its training-set builder and the squeak spectrogram
    store take. Shorthand for ``call_class_mask(usv_summary, ("squeak", "both"), source)``.

    Parameters
    ----------
    usv_summary (polars.DataFrame)
        A table holding ``usv`` and ``squeak``.
    source (str)
        What the table came from, named in the error when a flag is missing.

    Returns
    -------
    mask (polars.Series)
        Boolean, one value per row.
    """

    return call_class_mask(usv_summary, ("squeak", "both"), source)


def squeak_class_selection(selection: str) -> tuple[str, ...]:
    """
    Description
    -----------
    Resolves a squeak-class selection name (``"squeak"``, ``"both"`` or ``"squeak+both"``) to
    the call classes it covers.

    Parameters
    ----------
    selection (str)
        One of the keys of ``SQUEAK_CLASS_SELECTIONS``.

    Returns
    -------
    classes (tuple[str, ...])
        The call classes the selection keeps.

    Raises
    ------
    ValueError
        The name is not a known selection.
    """

    if selection not in SQUEAK_CLASS_SELECTIONS:
        error_message = f"Unknown squeak class selection {selection!r}; choose one of {list(SQUEAK_CLASS_SELECTIONS)}."
        raise ValueError(error_message)
    return SQUEAK_CLASS_SELECTIONS[selection]


def order_usv_summary_columns(usv_summary: Any) -> Any:
    """
    Description
    -----------
    Reorders a USV summary table to ``USV_SUMMARY_COLUMN_ORDER``. Canonical columns
    that are present come first, in canonical order; any other column keeps its
    relative order and follows them, so nothing is dropped and a column this
    function does not know about is never lost. Only the column order changes: no
    value, dtype or row is touched.

    Parameters
    ----------
    usv_summary (polars.DataFrame)
        The summary table about to be written.

    Returns
    -------
    ordered (polars.DataFrame)
        The same table with its columns in canonical order.
    """

    present = set(usv_summary.columns)
    canonical = [column for column in USV_SUMMARY_COLUMN_ORDER if column in present]
    extra = [column for column in usv_summary.columns if column not in USV_SUMMARY_COLUMN_ORDER]
    return usv_summary.select(canonical + extra)


def obsolete_usv_summary_columns(columns: Iterable[str]) -> list[str]:
    """
    Description
    -----------
    Picks, from a USV summary's column names, the columns the canonical layout no
    longer has: every name in ``USV_SUMMARY_OBSOLETE_COLUMNS`` and every per-call
    category confidence column (a name ending in one of
    ``CATEGORY_CONFIDENCE_SUFFIXES``, e.g. ``qlvm_category_agreement``). A
    canonical column (``USV_SUMMARY_COLUMN_ORDER``) is never picked.

    Parameters
    ----------
    columns (Iterable[str])
        The summary's column names, in file order.

    Returns
    -------
    obsolete (list[str])
        The obsolete column names, in the order given.
    """

    obsolete = []
    for column in columns:
        if column in USV_SUMMARY_COLUMN_ORDER:
            continue
        if column in USV_SUMMARY_OBSOLETE_COLUMNS or column.endswith(CATEGORY_CONFIDENCE_SUFFIXES):
            obsolete.append(column)
    return obsolete


def tidy_usv_summary_columns(usv_summary: Any) -> tuple[Any, dict]:
    """
    Description
    -----------
    Brings an existing USV summary table to the canonical layout: drops its
    obsolete columns (:func:`obsolete_usv_summary_columns`) and reorders the rest
    with :func:`order_usv_summary_columns` (canonical columns first, any other
    column after them in its existing order). No row or value of a kept column is
    touched, and no column is created. Used by the ``tidy-usv-summary-columns``
    command to migrate summaries written before the layout was fixed.

    Parameters
    ----------
    usv_summary (polars.DataFrame)
        A session's USV summary table.

    Returns
    -------
    tidied (polars.DataFrame)
        The table without its obsolete columns, in canonical column order.
    report (dict)
        What the tidy changes: ``dropped`` (list of the removed columns),
        ``unknown`` (list of the kept columns the canonical order does not list,
        which end up last), ``columns_before`` / ``columns_after`` (lists of the
        column names) and ``changed`` (bool, True when the columns or their order
        differ).
    """

    dropped = obsolete_usv_summary_columns(usv_summary.columns)
    tidied = order_usv_summary_columns(usv_summary.drop(dropped))
    unknown = [column for column in tidied.columns if column not in USV_SUMMARY_COLUMN_ORDER]
    report = {
        'dropped': dropped,
        'unknown': unknown,
        'columns_before': list(usv_summary.columns),
        'columns_after': list(tidied.columns),
        'changed': list(usv_summary.columns) != list(tidied.columns),
    }
    return tidied, report


def first_match_or_raise(
    root: pathlib.Path,
    pattern: str,
    recursive: bool = False,
    label: Optional[str] = None,
) -> pathlib.Path:
    """
    Description
    -----------
    Returns the alphabetically-first path matching a glob pattern under
    ``root``, or raises a FileNotFoundError with a clear, debuggable message
    naming both the pattern and the root that produced zero matches. Replaces
    the common 'sorted(root.glob(...))[0]' / 'list(root.rglob(...))[0]' idiom,
    which surfaced as bare IndexError with no hint about which pattern failed.

    Matches are sorted deterministically before selection so that callers get
    the same answer across runs, platforms, and filesystems. Earlier revisions
    used 'next(iter(glob))' which returned matches in directory-entry order;
    that caused non-deterministic behavior when the pattern matched multiple
    files (ext4's hash-order directory listing, in particular, is effectively
    random). Every known caller was the post-refactor equivalent of
    'sorted(glob)[0]', so sorting is restored as the default — there is no
    opt-out because non-deterministic first-match is not a feature we want.

    Parameters
    ----------
    root (pathlib.Path)
        The directory to search.
    pattern (str)
        The glob pattern (relative to root) to match.
    recursive (bool)
        If True, uses rglob to walk the directory tree; otherwise uses glob.
    label (str, optional)
        Short context label to include in the error message (e.g.,
        'camera frame count JSON'). Defaults to the pattern itself.

    Returns
    -------
    match (pathlib.Path)
        The alphabetically-first match.
    """

    root = pathlib.Path(root)
    if not root.exists():
        raise FileNotFoundError(
            f"{label or pattern}: search root '{root}' does not exist."
        )
    matches = sorted(root.rglob(pattern) if recursive else root.glob(pattern))
    if not matches:
        kind = "rglob" if recursive else "glob"
        raise FileNotFoundError(
            f"{label or pattern}: no match for {kind} pattern '{pattern}' under '{root}'."
        )
    return matches[0]


# The concatenated multi-channel audio memmaps of a session, one per frequency
# band, each in its own exact folder under ``<root>/audio``: ``usv`` is the
# 30 kHz high-passed HPSS audio every USV reader (DAS summary, spectrograms,
# loudness, vocalocator, figures, videos) was built and trained on; ``broadband``
# is the 2 kHz high-passed, line-noise-cleaned HPSS audio written by
# ``Operator.broadband_filter_audio``.
AUDIO_MMAP_BAND_FOLDERS = {"usv": "hpss_filtered", "broadband": "broadband_filtered"}


def audio_mmap_name_regex(band: str) -> re.Pattern:
    """
    Description
    -----------
    Compiled regular expression matching the exact file name of a session's
    concatenated audio memmap for one band:
    ``<id>_concatenated_audio_<folder>_<sampling rate>_<samples>_<channels>_int16.mmap``,
    where ``<folder>`` is the band's folder (``hpss_filtered`` for ``usv``,
    ``broadband_filtered`` for ``broadband``) and ``<id>`` the recording id token
    of the source wav names (no underscore). The pattern is anchored at both
    ends, so temporary siblings (``.<name>.tmp-<pid>``), copies with a suffix and
    the memmap of another band never match.

    Parameters
    ----------
    band (str)
        ``'usv'`` or ``'broadband'``.

    Returns
    -------
    regex (re.Pattern)
        Pattern with the named groups ``id``, ``sr``, ``n_samples`` and ``n_ch``.
    """

    if band not in AUDIO_MMAP_BAND_FOLDERS:
        raise ValueError(f"Unknown audio band {band!r}; expected one of {sorted(AUDIO_MMAP_BAND_FOLDERS)}.")
    folder = AUDIO_MMAP_BAND_FOLDERS[band]
    return re.compile(
        rf"^(?P<id>[^_]+)_concatenated_audio_{folder}_(?P<sr>\d+)_(?P<n_samples>\d+)_(?P<n_ch>\d+)_int16\.mmap$"
    )


def find_audio_mmap(root_directory: str | pathlib.Path, band: str) -> pathlib.Path:
    """
    Description
    -----------
    Returns the ONE concatenated audio memmap of a session for the requested
    band, searched in that band's exact folder (``<root>/audio/hpss_filtered``
    for ``'usv'``, ``<root>/audio/broadband_filtered`` for ``'broadband'``, never
    recursively) with the exact name pattern of :func:`audio_mmap_name_regex`.

    This replaces the older ``first_match_or_raise`` lookups with a ``*.mmap``
    glob (some of them recursive), which returned the alphabetically first memmap
    anywhere under ``audio/`` and so silently picked up any second memmap (a
    stray unfiltered file in ``cropped_to_video``, or a memmap of another band
    in a folder that sorts first). Here a reader of one band can never be handed
    the file of another band, and an ambiguous session fails loudly.

    Parameters
    ----------
    root_directory (str | pathlib.Path)
        Session root directory (contains ``audio``).
    band (str)
        ``'usv'`` (the 30 kHz high-passed HPSS memmap the USV pipeline reads) or
        ``'broadband'`` (the 2 kHz high-passed, line-noise-cleaned memmap).

    Returns
    -------
    mmap_path (pathlib.Path)
        Path of the single matching memmap.

    Raises
    ------
    ValueError
        If ``band`` is not a known band.
    FileNotFoundError
        If the band folder does not exist or holds no matching memmap.
    RuntimeError
        If the band folder holds more than one matching memmap.
    """

    regex = audio_mmap_name_regex(band)
    folder = pathlib.Path(root_directory) / "audio" / AUDIO_MMAP_BAND_FOLDERS[band]
    if not folder.is_dir():
        raise FileNotFoundError(f"{band} audio memmap: folder '{folder}' does not exist.")
    matches = sorted(path for path in folder.iterdir() if path.is_file() and regex.match(path.name))
    if not matches:
        raise FileNotFoundError(f"{band} audio memmap: no file matching '{regex.pattern}' in '{folder}'.")
    if len(matches) > 1:
        raise RuntimeError(
            f"{band} audio memmap: {len(matches)} files match in '{folder}' "
            f"({', '.join(path.name for path in matches)}); exactly one is required."
        )
    return matches[0]


def parse_audio_mmap_name(mmap_path: str | pathlib.Path) -> dict:
    """
    Description
    -----------
    Reads the layout a concatenated audio memmap encodes in its file name
    (``<id>_concatenated_audio_<folder>_<sr>_<n_samples>_<n_ch>_int16.mmap``),
    for either band.

    Parameters
    ----------
    mmap_path (str | pathlib.Path)
        Path (or name) of the memmap.

    Returns
    -------
    layout (dict)
        ``band`` (str), ``id`` (str), ``sampling_rate`` (int), ``n_samples``
        (int), ``n_channels`` (int) and ``dtype`` (``'int16'``).

    Raises
    ------
    ValueError
        If the name matches no band's pattern.
    """

    name = pathlib.Path(mmap_path).name
    for band in AUDIO_MMAP_BAND_FOLDERS:
        match = audio_mmap_name_regex(band).match(name)
        if match is not None:
            return {
                "band": band,
                "id": match["id"],
                "sampling_rate": int(match["sr"]),
                "n_samples": int(match["n_samples"]),
                "n_channels": int(match["n_ch"]),
                "dtype": "int16",
            }
    raise ValueError(f"'{name}' is not a concatenated audio memmap name of any band.")


def newest_match_or_raise(
    root: pathlib.Path,
    pattern: str,
    key: Optional[Callable[[pathlib.Path], float]] = None,
    recursive: bool = False,
    label: Optional[str] = None,
) -> pathlib.Path:
    """
    Description
    -----------
    Returns the single "largest" (newest, by default) path matching a glob
    pattern under ``root``, or raises FileNotFoundError with a clear, named
    message when the glob produces zero matches. Replaces the common
    'max(root.glob(...), key=lambda p: p.stat().st_ctime)' idiom, which
    raised a bare ValueError ("max() arg is an empty sequence") with no
    hint about which directory or pattern produced the empty result.

    Parameters
    ----------
    root (pathlib.Path)
        The directory to search.
    pattern (str)
        The glob pattern (relative to root) to match.
    key (Callable[[pathlib.Path], float], optional)
        Ordering key passed to max(). Defaults to creation time
        (lambda p: p.stat().st_ctime).
    recursive (bool)
        If True, uses rglob to walk the directory tree; otherwise uses glob.
    label (str, optional)
        Short context label to include in the error message (e.g.,
        'most recent Avisoft .wav'). Defaults to the pattern itself.

    Returns
    -------
    match (pathlib.Path)
        The maximum-key match.
    """

    root = pathlib.Path(root)
    if not root.exists():
        raise FileNotFoundError(
            f"{label or pattern}: search root '{root}' does not exist."
        )
    if key is None:
        def key(p):
            return p.stat().st_ctime
    matches = list(root.rglob(pattern) if recursive else root.glob(pattern))
    if not matches:
        kind = "rglob" if recursive else "glob"
        raise FileNotFoundError(
            f"{label or pattern}: no match for {kind} pattern '{pattern}' under '{root}'."
        )
    return max(matches, key=key)


# Embedding-landscape resolution. The visualization layer reads its precomputed
# cohort artifacts from a single base directory (``shared_resources.spectrograms_dir``)
# by convention, rather than from several hard-coded file paths:
#   <dir>/spectrograms_*.h5                   consolidated spectrogram/mask/latent store
#   <dir>/squeak_spectrograms_*.h5            2-125 kHz log-frequency squeak spectrogram store
#   <dir>/embeddings/pooled_embeddings_qlvmv3.parquet  pooled cohort embeddings cache
#                                             (POOLED_EMBEDDINGS_CACHE_NAME)
# The QLVM category geometry (label grid, density, centres) is not under it: it is
# the category bundle, QLVM_CATEGORY_BUNDLE_DIRECTORY (load_qlvm_category_bundle).
def resolve_consolidated_h5_path(spectrograms_dir: str) -> str:
    """
    Description
    -----------
    Resolve the consolidated spectrogram/mask/latent ``.h5`` store: the most
    recently modified ``spectrograms_*.h5`` directly under the base directory. The
    name pattern is deliberate -- it selects the consolidated store and skips other
    ``.h5`` siblings (e.g. the legacy ``qlvm_clusters_*.h5``). Raises ``FileNotFoundError`` with
    a clear message if the directory is missing or holds no matching store.

    Parameters
    ----------
    spectrograms_dir (str)
        Base spectrograms directory (run through ``configure_path``).

    Returns
    -------
    path (str)
        The OS-resolved path to the newest ``spectrograms_*.h5``.
    """

    base = pathlib.Path(configure_path(spectrograms_dir))
    return str(
        newest_match_or_raise(
            base,
            "spectrograms_*.h5",
            key=lambda p: p.stat().st_mtime,
            label="consolidated spectrogram .h5",
        )
    )


def resolve_squeak_spectrogram_store_path(spectrograms_dir: str) -> str:
    """
    Description
    -----------
    Resolve the squeak spectrogram store written by
    ``build-squeak-spectrogram-store`` (2-125 kHz, log-frequency spectrograms of
    the squeaks the squeak QLVM map holds): the most recently modified
    ``squeak_spectrograms_*.h5`` directly under the base directory. The pattern
    and the consolidated store's ``spectrograms_*.h5`` never match each other's
    files. Raises ``FileNotFoundError`` with a clear message if the directory is
    missing or holds no matching store.

    Parameters
    ----------
    spectrograms_dir (str)
        Base spectrograms directory (run through ``configure_path``).

    Returns
    -------
    path (str)
        The OS-resolved path to the newest ``squeak_spectrograms_*.h5``.
    """

    base = pathlib.Path(configure_path(spectrograms_dir))
    return str(
        newest_match_or_raise(
            base,
            "squeak_spectrograms_*.h5",
            key=lambda p: p.stat().st_mtime,
            label="squeak spectrogram store .h5",
        )
    )


def resolve_pooled_embeddings_cache(spectrograms_dir: str) -> str:
    """
    Description
    -----------
    Build the path to the cohort pooled-embeddings parquet cache under the
    spectrograms base directory, by convention:
    ``<dir>/embeddings/<POOLED_EMBEDDINGS_CACHE_NAME>``
    (``<dir>/embeddings/pooled_embeddings_qlvmv3.parquet``). The name is
    versioned by the QLVM model, so the cache pooled from the v3 summaries sits
    beside (and never overwrites) the old model's ``pooled_embeddings.parquet``.
    This is a pure path builder (run through ``configure_path``); whether the file
    exists is the caller's concern -- the embedding figures pass it as
    ``embeddings_cache_path`` so that ``build_pooled_embeddings_df`` loads it when
    present and its summaries fingerprint matches (one combined table that
    carries the QLVM coordinates and the coarse + fine labels), and
    otherwise pools the cohort and writes it there.

    Parameters
    ----------
    spectrograms_dir (str)
        Base directory where the precomputed embedding artifacts live.

    Returns
    -------
    path (str)
        The OS-resolved parquet path (not checked for existence).
    """

    base = pathlib.Path(configure_path(spectrograms_dir))
    return str(base / "embeddings" / POOLED_EMBEDDINGS_CACHE_NAME)
