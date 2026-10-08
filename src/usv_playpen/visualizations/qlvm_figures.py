"""
@author: bartulem

Cohort-level figures of the regular QLVM map and its content-ridge categories.

Three figures, each split into a compute step and a plotting step so the numbers
can be inspected (and tested) without drawing:

* ``plot_qlvm_overview`` -- the map at a glance: the USV density, the content
  change of the category bundle with the category boundaries, the USVs coloured
  by category with ten example spectrograms per category (sampled along a spiral
  around each category's densest point), and the USVs coloured by duration,
  spectral entropy, frequency bandwidth, mean frequency and loudness.
* ``compute_qlvm_quality_metrics`` / ``plot_qlvm_quality`` -- how good a map of
  the USVs the regular QLVM is, against PCA and UMAP of the same decoder inputs:
  the three maps coloured by the six measured USV properties; (a) the acoustic
  readout from 2-D position (kNN rank R^2, whole sessions held out); (b1-b3)
  neighborhood preservation against the six properties (neighbor overlap,
  false-neighbor severity = 1 - trustworthiness, torn-apart severity =
  1 - continuity); (b') distance fidelity (distance correlation of map distances
  with property distances, all pairs; the QLVM with the decoder pullback
  geodesic, the density geodesic and the flat torus distance of
  ``modeling.modeling_torus_geodesics``); (c) the readout per QLVM training seed.
* ``compute_category_boundary_fields`` / ``compute_category_similarity`` /
  ``plot_category_boundaries`` -- what the content-ridge category boundaries
  follow and whether the categories group similar USVs: the aligned mean
  spectrogram of each category; the USV density and the content change with the
  boundaries; the along-boundary test (boundary pattern against the same pattern
  shifted rigidly around the torus); each property's share of the content change
  on the boundaries; the content change per property; and, on the spectrograms
  themselves, (i) the mean normalized similarity between categories, (ii) each
  USV's similarity to its own category minus to the others, (iii) where each
  USV's closest matches fall, (iv) the categories against the density-watershed
  clusters of the same map.

Data. The USVs are the cohort's (``make_usv_spectrograms.load_regular_map_cohort_usvs``:
every non-playback ``*sessions_list.txt`` under the settings'
``input_files_directory``, pure USVs with a regular-map position and all six
properties). The spectrogram-based panels use a per-session sample
(``build_qlvm_figure_sample``: up to ``usvs_per_session`` random USVs of every
session) whose spectrograms are read from each session's own
``audio/spectrograms/<session>_spectrograms.h5`` (row = summary row), masked by
the union of the USV's SAM masks, both time-stretched to 128 x 128 exactly as the
regular QLVM's decoder input and at native length (zeros after the USV).

Normalized similarity (category figure). The raw similarity of two USVs is the
Pearson correlation of their masked, not time-stretched spectrograms (averaged
to 64 x 64) after alignment: the largest correlation over shifts of up to
``similarity_max_shift`` bins in time and frequency (zero padding, nothing wraps).
It is normalized per USV: for USV i, the similarity to j becomes its percentile
among i's similarities to a reference set of USVs (i itself left out), then the
i-to-j and j-to-i percentiles are averaged. 0.5 is the similarity of a typical
random pair for those two USVs.

All figure text uses the project style (``apply_plot_style``: Helvetica Light);
every colour is a hex string. Settings: the ``qlvm_figures`` block of
``visualizations_settings.json``.
"""

from __future__ import annotations

import pathlib
from collections.abc import Callable

import h5py
import numpy as np
import polars as pls
import umap
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial.distance import cdist
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors

from ..modeling.manifold_metric import distance_correlation
from ..modeling.modeling_torus_geodesics import (
    build_torus_geodesic_context,
    flat_torus_distance_matrix,
    make_qlvm_decode_fn_from_model_cell,
)
from ..os_utils import QLVM_CATEGORY_COLUMN, QLVM_CATEGORY_GRIDS_NAME, configure_path
from ..processing.build_qlvm_training_set import build_session_masks, stretch_specs
from ..processing.qlvm_categories import call_pixels, content_fields, property_ranks
from ..processing.qlvm_latents import load_model_cell, normalize_model_inputs
from ..processing.qlvm_model import embed_data
from .auxiliary_plot_functions import draw_category_outlines, periodic_density
from .make_usv_spectrograms import REGULAR_MAP_PROPERTY_COLUMNS, _pick_spiral_with_grid
from .plot_style import apply_plot_style

# (summary column, label with unit, factor from the stored value to the shown unit), in the order of
# make_usv_spectrograms.REGULAR_MAP_PROPERTY_COLUMNS.
FIGURE_PROPERTIES = (
    ("duration", "duration (ms)", 1000.0),
    ("spectral_entropy", "spectral entropy (nats)", 1.0),
    ("freq_bandwidth_hz", "frequency bandwidth (kHz)", 1e-3),
    ("mean_freq_hz", "mean frequency (kHz)", 1e-3),
    ("loudness_db", "loudness (dB)", 1.0),
    ("mask_number", "mask count (masks)", 1.0),
)
# The five properties of the overview's second row (the mask count is the categories' business).
OVERVIEW_PROPERTIES = FIGURE_PROPERTIES[:5]
# Short property names of the readout panel's x axis.
PROPERTY_SHORT_LABELS = {"duration": "duration", "spectral_entropy": "spectral\nentropy", "freq_bandwidth_hz": "bandwidth",
                         "mean_freq_hz": "mean\nfrequency", "loudness_db": "loudness", "mask_number": "mask\ncount"}
# The category figure's property order (content change per property, share on the boundaries).
BOUNDARY_PROPERTIES = (("mask_number", "mask count"), ("duration", "duration"), ("freq_bandwidth_hz", "bandwidth"),
                       ("spectral_entropy", "spectral entropy"), ("mean_freq_hz", "mean frequency"), ("loudness_db", "loudness"))
SPECTROGRAM_SIZE = 128
FREQUENCY_RANGE_KHZ = (30.0, 120.0)
TEXT_COLOR = "#000000"
FIGURE_FACE = "#FFFFFF"
BOUNDARY_COLOR = "#FFFFFF"
PROPERTY_BOUNDARY_COLOR = "#000000"
MAP_COLORS = {"QLVM": "#000000", "PCA": "#9A9A9A", "UMAP": "#E69F00"}
DISTANCE_COLORS = {"QLVM pullback geodesic": "#000000", "QLVM density geodesic": "#555555", "QLVM flat torus": "#8C8C8C",
                   "PCA Euclidean": "#BDBDBD", "UMAP Euclidean": "#E69F00"}
SEED_COLORS = ("#000000", "#666666", "#A6A6A6", "#CCCCCC")
NULL_COLOR = "#D62728"
DENSITY_COLOR = "#4E79A7"
CHANGE_COLOR = "#E15759"
BAR_COLOR = "#59595B"
WATERSHED_COLOR = "#BDBDBD"
GROUP_DOT_COLOR = "#FFFFFF"
CHANCE_COLORS = ("#2166AC", "#92C5DE", "#FFFFFF", "#F4A582", "#B2182B")
TITLE_SIZE = 14
LABEL_SIZE = 14
TICK_SIZE = 12
COLORBAR_TICK_SIZE = 13
COLORBAR_LABEL_SIZE = 15
QUALITY_LABEL_SIZE = 16
QUALITY_TICK_SIZE = 14
QUALITY_TITLE_SIZE = 16


def session_roots_by_id(session_roots: list[str]) -> dict[str, pathlib.Path]:
    """
    Description
    -----------
    Maps each session id (the root directory's name) to its host-resolved root,
    so a pooled USV (``session_id``, ``row_index``) can be traced back to its
    session's spectrogram H5.

    Parameters
    ----------
    session_roots (list[str])
        Session roots (``make_usv_spectrograms.read_cohort_session_roots``).

    Returns
    -------
    roots (dict[str, pathlib.Path])
        Session id -> root directory.
    """

    resolved = [pathlib.Path(configure_path(root)) for root in session_roots]
    return {root.name: root for root in resolved}


def session_spectrogram_file(root: pathlib.Path) -> pathlib.Path:
    """
    Description
    -----------
    The per-session spectrogram store written by ``generate-usv-spectrograms``:
    ``<root>/audio/spectrograms/<session>_spectrograms.h5``.

    Parameters
    ----------
    root (pathlib.Path)
        Session root directory.

    Returns
    -------
    path (pathlib.Path)
        The H5 path (not checked for existence).
    """

    return root / "audio" / "spectrograms" / f"{root.name}_spectrograms.h5"


def check_categories_match_bundle(usvs: pls.DataFrame, label_grid: np.ndarray) -> None:
    """
    Description
    -----------
    Checks that every USV's ``qlvm_category`` is the category the bundle's grid
    gives its regular-map position (the pixel rule ``assign-qlvm-categories``
    applies). A mismatch means the summaries were categorized with another bundle
    or before their positions were last written, so the figures would mix two
    partitions.

    Parameters
    ----------
    usvs (pls.DataFrame)
        USVs with ``session_id``, ``qlvm1``, ``qlvm2`` and ``qlvm_category``.
    label_grid (np.ndarray)
        The bundle's ``(res, res)`` grid (``[y, x]``).

    Returns
    -------
    None

    Raises
    ------
    ValueError
        Some USVs' categories disagree with the grid; the message names the sessions.
    """

    if QLVM_CATEGORY_COLUMN not in usvs.columns:
        error_message = f"The pooled summaries carry no {QLVM_CATEGORY_COLUMN} column; run assign-qlvm-categories first."
        raise ValueError(error_message)
    rows, cols = call_pixels(np.column_stack([usvs["qlvm1"].to_numpy(), usvs["qlvm2"].to_numpy()]), label_grid.shape[0])
    wrong = label_grid[rows, cols] != usvs[QLVM_CATEGORY_COLUMN].to_numpy()
    if wrong.any():
        sessions = sorted(set(usvs["session_id"].to_numpy()[wrong].tolist()))
        error_message = (f"{int(wrong.sum()):,} USVs in {len(sessions)} session(s) carry a {QLVM_CATEGORY_COLUMN} other than the "
                         f"category bundle's grid gives their position (e.g. {', '.join(sessions[:5])}); re-run infer-qlvm-latents / "
                         "assign-qlvm-categories on them, or leave them out.")
        raise ValueError(error_message)


def build_qlvm_figure_sample(
    usvs: pls.DataFrame,
    roots: dict[str, pathlib.Path],
    usvs_per_session: int,
    seed: int,
    model_cell_directory: str,
    cache_file: str | None,
    message_output: Callable | None = None,
) -> dict:
    """
    Description
    -----------
    Draws the per-session sample of the spectrogram-based panels and reads its
    spectrograms: at most ``usvs_per_session`` random USVs of every session of
    ``usvs`` (sessions in sorted order, rows in summary order, one generator of
    ``seed``). For each USV the session H5 gives its spectrogram (128 frequency x
    128 time bins, the USV from the first time bin at its native length), its
    length in time bins and the union of its SAM masks
    (``build_qlvm_training_set.build_session_masks``). Two images are kept:

    * the decoder input of the regular map: spectrogram and mask union
      time-stretched to 128 x 128 separately (``stretch_specs``), the mask
      binarized at 0.5, the masked spectrogram normalized by the cell's training
      contract and masked again -- exactly the input ``infer-qlvm-latents`` embeds;
    * the masked, not time-stretched spectrogram: the spectrogram with the time
      bins after the USV zeroed, times the binarized mask union.

    With ``cache_file`` the arrays are written there, and served from there on
    later calls when the cached USVs (session, row and regular-map position) are
    exactly the ones drawn now; a cache built from other or older summaries is
    rebuilt and overwritten.

    Parameters
    ----------
    usvs (pls.DataFrame)
        Cohort USVs (``load_regular_map_cohort_usvs``): ``session_id``,
        ``row_index``, ``qlvm1``, ``qlvm2``, ``qlvm_category`` and the six properties.
    roots (dict[str, pathlib.Path])
        Session id -> root directory (``session_roots_by_id``).
    usvs_per_session (int)
        Most USVs drawn per session.
    seed (int)
        Seed of the draw.
    model_cell_directory (str)
        The regular map's production cell (its training contract normalizes the decoder input).
    cache_file (str | None)
        ``.npz`` cache of the sample, or None for no cache.
    message_output (Callable | None)
        Logger; defaults to ``print``.

    Returns
    -------
    sample (dict)
        ``session_id`` (str), ``row_index``, ``xy`` ``(n, 2)``, ``category``,
        ``properties`` ``(n, 6)`` (``REGULAR_MAP_PROPERTY_COLUMNS`` order),
        ``inputs`` ``(n, 128, 128)`` float16 decoder inputs, ``images``
        ``(n, 128, 128)`` float16 masked, not time-stretched spectrograms and
        ``lengths`` (time bins).
    """

    log = message_output or print
    rng = np.random.default_rng(seed)
    parts = []
    for session_id in sorted(usvs["session_id"].unique().to_list()):
        session = usvs.filter(pls.col("session_id") == session_id).sort("row_index")
        chosen = np.sort(rng.choice(session.height, size=min(usvs_per_session, session.height), replace=False))
        parts.append(session[chosen.tolist()])
    drawn = pls.concat(parts, how="vertical")
    session_ids = drawn["session_id"].to_numpy().astype(str)
    row_index = drawn["row_index"].to_numpy().astype(np.int64)
    xy = np.column_stack([drawn["qlvm1"].to_numpy(), drawn["qlvm2"].to_numpy()]).astype(np.float64)

    if cache_file:
        cache_path = pathlib.Path(configure_path(cache_file))
        if cache_path.exists():
            with np.load(cache_path) as cached:
                same = (cached["session_id"].shape == session_ids.shape and np.array_equal(cached["session_id"], session_ids)
                        and np.array_equal(cached["row_index"], row_index) and np.allclose(cached["xy"], xy, atol=1e-9))
                if same:
                    log(f"[qlvm-figures] sample of {session_ids.size:,} USVs served from {cache_path}.")
                    return {key: cached[key] for key in cached.files}
            log(f"[qlvm-figures] sample cache {cache_path} holds other USVs; rebuilding it.")

    contract = load_model_cell(model_cell_directory)["contract"]
    n = session_ids.size
    inputs = np.zeros((n, SPECTROGRAM_SIZE, SPECTROGRAM_SIZE), dtype=np.float16)
    images = np.zeros((n, SPECTROGRAM_SIZE, SPECTROGRAM_SIZE), dtype=np.float16)
    lengths = np.zeros(n, dtype=np.int32)
    unique_sessions = np.unique(session_ids)
    for number, session_id in enumerate(unique_sessions, start=1):
        positions = np.flatnonzero(session_ids == session_id)
        rows = row_index[positions].astype(np.uint32)
        with h5py.File(session_spectrogram_file(roots[session_id]), "r") as h5:
            group = h5[f"spectrogram/{session_id}"]
            specs = np.stack([group["spectrograms"][int(row)] for row in rows]).astype(np.float32)
            durations = group["durations"][:][rows].astype(int)
            masks, _ = build_session_masks(h5, session_id, rows, specs.shape[1], specs.shape[2])
        stretched = stretch_specs(specs, durations, (SPECTROGRAM_SIZE, SPECTROGRAM_SIZE), True)
        binary = (stretch_specs(masks, durations, (SPECTROGRAM_SIZE, SPECTROGRAM_SIZE), True) >= 0.5).astype(np.float32)
        inputs[positions] = (normalize_model_inputs(stretched * binary, contract) * binary).astype(np.float16)
        for index, duration in enumerate(durations):
            specs[index, :, duration:] = 0.0
        images[positions] = (specs * (masks >= 0.5)).astype(np.float16)
        lengths[positions] = durations
        if number % 50 == 0 or number == unique_sessions.size:
            log(f"[qlvm-figures] read {number}/{unique_sessions.size} sessions.")

    sample = {"session_id": session_ids, "row_index": row_index, "xy": xy,
              "category": drawn[QLVM_CATEGORY_COLUMN].to_numpy().astype(np.int64),
              "properties": np.column_stack([drawn[column].to_numpy() for column in REGULAR_MAP_PROPERTY_COLUMNS]).astype(np.float64),
              "inputs": inputs, "images": images, "lengths": lengths}
    if cache_file:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, **sample)
        log(f"[qlvm-figures] sample of {n:,} USVs from {unique_sessions.size} sessions written to {cache_path}.")
    return sample


def embed_with_cells(inputs: np.ndarray, cell_directories: list[str], lattice_batch_size: int, data_batch_size: int,
                     message_output: Callable | None = None) -> dict[str, np.ndarray]:
    """
    Description
    -----------
    Embeds decoder inputs with other unconditional QLVM cells (e.g. retrainings of
    the regular map's recipe under other seeds): each cell's posterior-mean torus
    position of every input (``qlvm_model.embed_data`` over the cell's own lattice).

    Parameters
    ----------
    inputs (np.ndarray)
        ``(n, 128, 128)`` decoder inputs (``build_qlvm_figure_sample``).
    cell_directories (list[str])
        Model package cells (run through ``configure_path``).
    lattice_batch_size (int)
        Lattice points decoded at once.
    data_batch_size (int)
        Inputs scored at once.
    message_output (Callable | None)
        Logger; defaults to ``print``.

    Returns
    -------
    positions (dict[str, np.ndarray])
        Cell directory name -> ``(n, 2)`` torus positions.
    """

    log = message_output or print
    positions = {}
    for directory in cell_directories:
        cell = load_model_cell(configure_path(directory))
        name = pathlib.Path(configure_path(directory)).name
        positions[name] = np.asarray(embed_data(cell["lattice"], inputs.astype(np.float32)[:, None], cell["params"],
                                                lattice_batch_size, data_batch_size), dtype=np.float64)
        log(f"[qlvm-figures] embedded {inputs.shape[0]:,} USVs with {name}.")
    return positions


def neighbor_indices(points: np.ndarray, queries: np.ndarray, k: int, torus: bool, exclude_self: bool) -> np.ndarray:
    """
    Description
    -----------
    Exact k nearest neighbors: Euclidean on a plane, or the flat-torus shortest
    wrapped distance (found by searching the 3 x 3 periodic tiling of ``points``,
    which holds every wrapped copy).

    Parameters
    ----------
    points (np.ndarray)
        ``(n, 2)`` candidate positions.
    queries (np.ndarray)
        ``(m, 2)`` query positions.
    k (int)
        Neighbors per query.
    torus (bool)
        Unit-torus positions (True) or plane positions (False).
    exclude_self (bool)
        The queries are ``points`` themselves: drop each one's own entry.

    Returns
    -------
    indices (np.ndarray)
        ``(m, k)`` indices into ``points``.
    """

    extra = 1 if exclude_self else 0
    if not torus:
        found = NearestNeighbors(n_neighbors=k + extra).fit(points).kneighbors(queries, return_distance=False)
        return found[:, extra:]
    offsets = np.array([(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)], dtype=float)
    tiled = (points[None, :, :] + offsets[:, None, :]).reshape(-1, 2)
    found = NearestNeighbors(n_neighbors=k + extra).fit(tiled).kneighbors(queries, return_distance=False) % points.shape[0]
    return found[:, extra:]


def knn_rank_r2(points: np.ndarray, target: np.ndarray, groups: np.ndarray, torus: bool, n_neighbors: int,
                n_folds: int) -> tuple[float, list[float]]:
    """
    Description
    -----------
    Acoustic readout from map position: the target's ranks (scaled to [0, 1])
    predicted for every USV of a held-out fold as the mean rank of its
    ``n_neighbors`` nearest USVs of the other folds, in the map's own geometry;
    folds group whole sessions (``GroupKFold``). Scored by the coefficient of
    determination, ``1 - SS_res / SS_tot``.

    Parameters
    ----------
    points (np.ndarray)
        ``(n, 2)`` positions.
    target (np.ndarray)
        ``(n,)`` property values.
    groups (np.ndarray)
        ``(n,)`` session ids.
    torus (bool)
        Torus positions (flat-torus neighbors) or plane positions (Euclidean).
    n_neighbors (int)
        Neighbors averaged.
    n_folds (int)
        Session-grouped folds.

    Returns
    -------
    r2 (float)
        R^2 of the pooled out-of-fold predictions.
    per_fold (list[float])
        The R^2 within each fold.
    """

    ranks = rankdata(target) / target.size
    predicted = np.empty_like(ranks)
    per_fold = []
    for train_rows, test_rows in GroupKFold(n_splits=n_folds).split(points, ranks, groups):
        neighbors = neighbor_indices(points[train_rows], points[test_rows], n_neighbors, torus, exclude_self=False)
        predicted[test_rows] = ranks[train_rows][neighbors].mean(axis=1)
        truth = ranks[test_rows]
        per_fold.append(float(1.0 - ((truth - predicted[test_rows]) ** 2).sum() / ((truth - truth.mean()) ** 2).sum()))
    return float(1.0 - ((ranks - predicted) ** 2).sum() / ((ranks - ranks.mean()) ** 2).sum()), per_fold


def neighborhood_scores(map_distances: np.ndarray, reference_distances: np.ndarray, k: int) -> dict:
    """
    Description
    -----------
    Neighborhood preservation of a map against a reference space, from two
    precomputed distance matrices. A USV's true neighbors are its ``k`` nearest
    in the reference space, its map neighbors its ``k`` nearest on the map. The
    number of intruders (map neighbors that are not true neighbors) always equals
    the number of missed true neighbors; the two severities weigh them from the
    two directions (Venna and Kaski): false-neighbor severity = 1 -
    trustworthiness (how dissimilar the intruders are, by their reference rank),
    torn-apart severity = 1 - continuity (how far on the map the missed true
    neighbors were placed, by their map rank).

    Parameters
    ----------
    map_distances (np.ndarray)
        ``(n, n)`` distances on the map.
    reference_distances (np.ndarray)
        ``(n, n)`` distances in the reference space.
    k (int)
        Neighborhood size.

    Returns
    -------
    scores (dict)
        ``overlap`` (share of the true neighbors that are map neighbors),
        ``false_neighbors`` and ``torn_neighbors``.
    """

    n = map_distances.shape[0]
    map_d = map_distances.copy()
    np.fill_diagonal(map_d, np.inf)
    ref_d = reference_distances.copy()
    np.fill_diagonal(ref_d, np.inf)
    map_rank = np.empty((n, n), dtype=np.int64)
    map_rank[np.arange(n)[:, None], np.argsort(map_d, axis=1)] = np.arange(1, n + 1)[None, :]
    ref_rank = np.empty((n, n), dtype=np.int64)
    ref_rank[np.arange(n)[:, None], np.argsort(ref_d, axis=1)] = np.arange(1, n + 1)[None, :]
    norm = 2.0 / (n * k * (2 * n - 3 * k - 1))
    in_map = map_rank <= k
    in_ref = ref_rank <= k
    return {"overlap": float((in_map & in_ref).sum() / (n * k)),
            "false_neighbors": float(norm * np.where(in_map & ~in_ref, ref_rank - k, 0).sum()),
            "torn_neighbors": float(norm * np.where(in_ref & ~in_map, map_rank - k, 0).sum())}


def standardized_property_ranks(properties: np.ndarray) -> np.ndarray:
    """
    Description
    -----------
    The six properties as ranks, each standardized to zero mean and unit variance
    (missing values set to the property's median first), so all six weigh equally
    in the Euclidean acoustic distance.

    Parameters
    ----------
    properties (np.ndarray)
        ``(n, p)`` property values.

    Returns
    -------
    ranks (np.ndarray)
        ``(n, p)`` standardized ranks.
    """

    filled = np.column_stack([np.nan_to_num(properties[:, i], nan=np.nanmedian(properties[:, i])) for i in range(properties.shape[1])])
    ranks = np.column_stack([rankdata(filled[:, i]) for i in range(filled.shape[1])])
    return (ranks - ranks.mean(axis=0)) / ranks.std(axis=0)


def compute_qlvm_quality_metrics(
    sample: dict,
    cfg: dict,
    seed: int,
    model_cell_directory: str,
    seed_positions: dict[str, np.ndarray],
    message_output: Callable | None = None,
) -> dict:
    """
    Description
    -----------
    Every metric of the QLVM quality figure on the per-session sample (module
    docstring). The baselines are fitted to the sample's decoder inputs: PCA
    (``n_pcs`` components, randomized SVD, ``seed``), the first two components
    being the PCA map; UMAP (two components) of those components
    (``umap_n_neighbors``, ``umap_min_dist``, ``seed``; a seeded UMAP runs on one
    thread, so ``n_jobs=1`` is set explicitly). The QLVM map is the
    regular map's stored position. The reference space of (b) and (b') is the six
    standardized property ranks.

    Parameters
    ----------
    sample (dict)
        ``build_qlvm_figure_sample`` output.
    cfg (dict)
        The ``qlvm_figures`` settings block.
    seed (int)
        Seed of the fits and of every random draw.
    model_cell_directory (str)
        The regular map's production cell (its decoder defines the pullback geodesic).
    seed_positions (dict[str, np.ndarray])
        Name -> ``(n, 2)`` positions of the same USVs under other trainings of the
        regular map (``embed_with_cells``); empty to skip panel (c).
    message_output (Callable | None)
        Logger; defaults to ``print``.

    Returns
    -------
    metrics (dict)
        ``maps`` (name -> (positions, is_torus)), ``readout_folds`` /
        ``readout_null`` (map -> property -> values), ``neighborhoods``
        (map -> score -> per-subset values), ``distance_fidelity`` (distance ->
        per-sample values), ``per_seed`` (name -> property -> per-fold R^2) and
        ``n_usvs`` / ``n_sessions``.
    """

    log = message_output or print
    quality = cfg["quality"]
    groups = sample["session_id"]
    properties = sample["properties"]
    n = groups.size
    rng = np.random.default_rng(seed)
    flat_inputs = sample["inputs"].astype(np.float32).reshape(n, -1)
    components = PCA(n_components=quality["n_pcs"], svd_solver="randomized", random_state=seed).fit_transform(flat_inputs)
    del flat_inputs
    umap_xy = umap.UMAP(n_components=2, n_neighbors=quality["umap_n_neighbors"], min_dist=quality["umap_min_dist"],
                        random_state=seed, n_jobs=1).fit_transform(components)
    log("[qlvm-quality] PCA and UMAP fitted.")
    maps = {"QLVM": (sample["xy"], True), "PCA": (components[:, :2], False), "UMAP": (np.asarray(umap_xy, dtype=np.float64), False)}
    metrics = {"n_usvs": int(n), "n_sessions": int(np.unique(groups).size), "maps": maps, "readout_folds": {}, "readout_null": {},
               "neighborhoods": {}, "distance_fidelity": {}, "per_seed": {}}

    for map_name, (xy, torus) in maps.items():
        shuffled = xy[rng.permutation(n)]
        metrics["readout_folds"][map_name] = {}
        metrics["readout_null"][map_name] = {}
        for index, (column, _, _) in enumerate(FIGURE_PROPERTIES):
            metrics["readout_folds"][map_name][column] = knn_rank_r2(xy, properties[:, index], groups, torus,
                                                                     quality["readout_neighbors"], quality["readout_folds"])[1]
            metrics["readout_null"][map_name][column] = knn_rank_r2(shuffled, properties[:, index], groups, torus,
                                                                    quality["readout_neighbors"], quality["readout_folds"])[0]
    log("[qlvm-quality] readout done.")

    reference_ranks = standardized_property_ranks(properties)
    order = rng.permutation(n)
    size = quality["neighborhood_subset_size"]
    for map_name in maps:
        metrics["neighborhoods"][map_name] = {"overlap": [], "false_neighbors": [], "torn_neighbors": []}
    for subset_index in range(quality["n_neighborhood_subsets"]):
        subset = np.sort(order[subset_index * size:(subset_index + 1) * size])
        reference = cdist(reference_ranks[subset], reference_ranks[subset]).astype(np.float32)
        for map_name, (xy, torus) in maps.items():
            distances = (flat_torus_distance_matrix(xy[subset]) if torus else cdist(xy[subset], xy[subset])).astype(np.float32)
            for key, value in neighborhood_scores(distances, reference, quality["neighborhood_k"]).items():
                metrics["neighborhoods"][map_name][key].append(value)
    log("[qlvm-quality] neighborhoods done.")

    context = build_torus_geodesic_context(sample["xy"], decode_fn=make_qlvm_decode_fn_from_model_cell(model_cell_directory),
                                           n_per_dim=quality["pullback_grid"])
    nodes = neighbor_indices(context.grid, sample["xy"], 1, torus=True, exclude_self=False)[:, 0]
    order = rng.permutation(n)
    size = quality["distance_sample_size"]
    labels = tuple(DISTANCE_COLORS)
    metrics["distance_fidelity"] = {label: [] for label in labels}
    for sample_index in range(quality["n_distance_samples"]):
        pairs = np.sort(order[sample_index * size:(sample_index + 1) * size])
        reference = cdist(reference_ranks[pairs], reference_ranks[pairs])
        distances = {"QLVM pullback geodesic": context.pullback_matrix[np.ix_(nodes[pairs], nodes[pairs])],
                     "QLVM density geodesic": context.density_matrix[np.ix_(nodes[pairs], nodes[pairs])],
                     "QLVM flat torus": flat_torus_distance_matrix(sample["xy"][pairs]),
                     "PCA Euclidean": cdist(maps["PCA"][0][pairs], maps["PCA"][0][pairs]),
                     "UMAP Euclidean": cdist(maps["UMAP"][0][pairs], maps["UMAP"][0][pairs])}
        for label in labels:
            metrics["distance_fidelity"][label].append(float(distance_correlation(distances[label], reference)))
    log("[qlvm-quality] distance fidelity done.")

    if seed_positions:
        for name, xy in {"production": sample["xy"], **seed_positions}.items():
            metrics["per_seed"][name] = {column: knn_rank_r2(xy, properties[:, index], groups, True, quality["readout_neighbors"],
                                                             quality["readout_folds"])[1]
                                         for index, (column, _, _) in enumerate(FIGURE_PROPERTIES)}
        log("[qlvm-quality] per-seed readout done.")
    return metrics


def bar_with_dots(axis, x: float, values, color: str, jitter: float, half_width: float) -> None:
    """
    Description
    -----------
    Draws repeated values (folds, subsets or samples) as dots over a bar at their mean.

    Parameters
    ----------
    axis (matplotlib.axes.Axes)
        Target axis.
    x (float)
        Horizontal position of the group.
    values (array-like)
        The repeated values.
    color (str)
        Hex colour of the group.
    jitter (float)
        Half-spread of the dots around ``x``.
    half_width (float)
        Half-width of the bar.

    Returns
    -------
    None
    """

    values = np.asarray(values, dtype=float)
    axis.bar(x, values.mean(), width=2 * half_width, color=color, alpha=0.35, zorder=2)
    axis.scatter(x + np.linspace(-jitter, jitter, values.size), values, s=26, color=color, linewidths=0, zorder=3)


def style_summary_axis(axis, title: str, ylabel: str) -> None:
    """
    Description
    -----------
    Applies the quality figure's title, y label and tick sizes to one summary axis.

    Parameters
    ----------
    axis (matplotlib.axes.Axes)
        Target axis.
    title (str)
        Panel title.
    ylabel (str)
        Y-axis label.

    Returns
    -------
    None
    """

    axis.set_title(title, fontsize=QUALITY_TITLE_SIZE, color=TEXT_COLOR)
    axis.set_ylabel(ylabel, fontsize=QUALITY_LABEL_SIZE, color=TEXT_COLOR)
    axis.tick_params(labelsize=QUALITY_TICK_SIZE)


def plot_qlvm_quality(sample: dict, metrics: dict, cfg: dict, cmap_name: str, seed: int) -> Figure:
    """
    Description
    -----------
    Draws the QLVM quality figure from ``compute_qlvm_quality_metrics``: the three
    maps (rows) coloured by the six properties (columns, 1st-99th percentile);
    (a) readout, bars = mean of the folds, dots = folds, red = shuffled positions;
    (c) readout per training seed (when computed); (b1-b3) neighborhood scores,
    dots = disjoint subsets; (b') distance fidelity, dots = disjoint samples.

    Parameters
    ----------
    sample (dict)
        ``build_qlvm_figure_sample`` output.
    metrics (dict)
        ``compute_qlvm_quality_metrics`` output.
    cfg (dict)
        The ``qlvm_figures`` settings block.
    cmap_name (str)
        Sequential colormap of the maps (the project's ``sequential_cmap``).
    seed (int)
        Seed of the drawing order of the map dots.

    Returns
    -------
    fig (Figure)
        The figure.
    """

    apply_plot_style()
    quality = cfg["quality"]
    properties = sample["properties"]
    maps = metrics["maps"]
    n = properties.shape[0]
    fig = Figure(figsize=(20, 22.5), facecolor=FIGURE_FACE)
    rows = fig.add_gridspec(3, 1, height_ratios=[1.0, 0.8, 0.8], hspace=0.24)
    top = rows[0].subgridspec(3, len(FIGURE_PROPERTIES), wspace=0.2, hspace=0.05)
    middle = rows[1].subgridspec(1, 2, width_ratios=[1.45, 1.0], wspace=0.2)
    bottom = rows[2].subgridspec(1, 4, width_ratios=[1, 1, 1, 1.5], wspace=0.42)
    draw_order = np.random.default_rng(seed).permutation(n)
    for row, (map_name, (xy, torus)) in enumerate(maps.items()):
        for column, (_, label, scale) in enumerate(FIGURE_PROPERTIES):
            axis = fig.add_subplot(top[row, column])
            values = properties[:, column] * scale
            low, high = np.nanpercentile(values, [1, 99])
            image = axis.scatter(xy[draw_order, 0], xy[draw_order, 1], c=values[draw_order], s=0.3, cmap=cmap_name,
                                 vmin=low, vmax=high, linewidths=0, rasterized=True)
            axis.set_xticks([])
            axis.set_yticks([])
            if torus:
                axis.set_xlim(0, 1)
                axis.set_ylim(0, 1)
            else:
                center = (xy.min(axis=0) + xy.max(axis=0)) / 2.0
                half = 0.52 * (xy.max(axis=0) - xy.min(axis=0)).max()
                axis.set_xlim(center[0] - half, center[0] + half)
                axis.set_ylim(center[1] - half, center[1] + half)
            axis.set_aspect("equal")
            if row == 0:
                axis.set_title(label, fontsize=11, color=TEXT_COLOR)
            if column == 0:
                axis.set_ylabel(map_name, fontsize=13, color=TEXT_COLOR)
            colorbar = fig.colorbar(image, ax=axis, fraction=0.046, pad=0.015)
            colorbar.ax.tick_params(length=0, labelsize=9)

    columns = [column for column, _, _ in FIGURE_PROPERTIES]
    positions = np.arange(len(columns))
    axis = fig.add_subplot(middle[0])
    for offset, map_name in zip((-0.27, 0.0, 0.27), maps, strict=True):
        for index, column in enumerate(columns):
            bar_with_dots(axis, positions[index] + offset, metrics["readout_folds"][map_name][column], MAP_COLORS[map_name],
                          jitter=0.07, half_width=0.135)
            axis.scatter(positions[index] + offset, metrics["readout_null"][map_name][column], s=22, color=NULL_COLOR, zorder=3)
        axis.bar([0], [0], color=MAP_COLORS[map_name], alpha=0.35, label=map_name)
    axis.scatter([], [], s=22, color=NULL_COLOR, label="shuffled positions")
    axis.axhline(0, color=TEXT_COLOR, linewidth=0.8)
    axis.set_ylim(-0.1, 1.0)
    axis.set_xticks(positions)
    axis.set_xticklabels([PROPERTY_SHORT_LABELS[column] for column in columns])
    style_summary_axis(axis, f"(a) acoustic readout from 2-D position (dots: {quality['readout_folds']} session-held-out folds)",
                       "kNN rank R²")
    axis.legend(frameon=False, fontsize=QUALITY_TICK_SIZE, ncol=1, loc="upper right")

    axis = fig.add_subplot(middle[1])
    if metrics["per_seed"]:
        seeds = list(metrics["per_seed"])
        offsets = np.linspace(-0.27, 0.27, len(seeds)) if len(seeds) > 1 else np.zeros(1)
        width = 0.135 if len(seeds) <= 3 else 0.54 / len(seeds) / 2
        for offset, name, color in zip(offsets, seeds, SEED_COLORS, strict=False):
            for index, column in enumerate(columns):
                bar_with_dots(axis, positions[index] + offset, metrics["per_seed"][name][column], color, jitter=0.07, half_width=width)
            axis.bar([0], [0], color=color, alpha=0.35, label=name)
        axis.set_ylim(0, 1.0)
        axis.set_xticks(positions)
        axis.set_xticklabels([PROPERTY_SHORT_LABELS[column] for column in columns])
        axis.legend(frameon=False, fontsize=QUALITY_TICK_SIZE, ncol=1, loc="upper right")
    else:
        axis.set_axis_off()
        axis.text(0.5, 0.5, "no other trainings set\n(qlvm_figures.seed_cell_directories)", ha="center", va="center",
                  fontsize=QUALITY_TICK_SIZE, color=TEXT_COLOR, transform=axis.transAxes)
    style_summary_axis(axis, "(c) readout per QLVM training seed", "kNN rank R²")

    panels = (("overlap", "(b1) neighbor overlap", "share of true neighbors found\n(higher is better)"),
              ("false_neighbors", "(b2) false-neighbor severity", "1 - trustworthiness\n(lower is better)"),
              ("torn_neighbors", "(b3) torn-apart severity", "1 - continuity\n(lower is better)"))
    for column, (key, title, ylabel) in enumerate(panels):
        axis = fig.add_subplot(bottom[column])
        for index, map_name in enumerate(maps):
            bar_with_dots(axis, index, metrics["neighborhoods"][map_name][key], MAP_COLORS[map_name], jitter=0.18, half_width=0.5)
        axis.set_xticks(range(len(maps)))
        axis.set_xticklabels(list(maps))
        axis.set_xlim(-0.9, len(maps) - 0.1)
        style_summary_axis(axis, title, ylabel)
    fig.text(0.5 * (bottom[0].get_position(fig).x0 + bottom[2].get_position(fig).x1), bottom[0].get_position(fig).y0 - 0.025,
             f"k = {quality['neighborhood_k']}; dots: {quality['n_neighborhood_subsets']} disjoint subsets of "
             f"{quality['neighborhood_subset_size']:,} USVs", ha="center", fontsize=QUALITY_TICK_SIZE)

    axis = fig.add_subplot(bottom[3])
    for index, label in enumerate(DISTANCE_COLORS):
        values = np.asarray(metrics["distance_fidelity"][label])
        axis.barh(index, values.mean(), height=1.0, color=DISTANCE_COLORS[label], alpha=0.35, zorder=2)
        axis.scatter(values, index + np.linspace(-0.18, 0.18, values.size), s=26, color=DISTANCE_COLORS[label], linewidths=0, zorder=3)
        axis.text(0.01, index, label, va="center", ha="left", fontsize=QUALITY_TICK_SIZE, color=TEXT_COLOR, zorder=4)
    axis.set_yticks([])
    axis.set_ylim(len(DISTANCE_COLORS) - 0.5, -0.5)
    axis.set_xlim(0, 0.7)
    axis.set_xlabel("distance correlation:\nmap distance vs acoustic difference", fontsize=QUALITY_LABEL_SIZE, color=TEXT_COLOR)
    axis.set_title(f"(b') distance fidelity, all pairs\n({quality['n_distance_samples']} samples of "
                   f"{quality['distance_sample_size']:,} USVs)", fontsize=QUALITY_TITLE_SIZE, color=TEXT_COLOR)
    axis.tick_params(labelsize=QUALITY_TICK_SIZE)
    return fig


def similarity_features(images: np.ndarray, downsample: int) -> np.ndarray:
    """
    Description
    -----------
    Prepares masked spectrograms for the aligned correlation: averaged over
    ``downsample`` x ``downsample`` blocks.

    Parameters
    ----------
    images (np.ndarray)
        ``(m, 128, 128)`` masked spectrograms.
    downsample (int)
        Block size (a divisor of 128).

    Returns
    -------
    features (np.ndarray)
        ``(m, 128 / downsample, 128 / downsample)`` float32.
    """

    m, height, width = images.shape
    return images.astype(np.float32).reshape(m, height // downsample, downsample, width // downsample, downsample).mean(axis=(2, 4))


def aligned_correlation(features: np.ndarray, max_shift: int) -> np.ndarray:
    """
    Description
    -----------
    Pairwise correlation after alignment: the largest Pearson correlation over
    shifts of up to ``max_shift`` bins along both image axes, the shifted image
    padded with zeros (nothing wraps around), each shifted copy re-centred and
    re-scaled to unit norm. Made symmetric by taking the larger of the two
    directions.

    Parameters
    ----------
    features (np.ndarray)
        ``(m, h, w)`` images.
    max_shift (int)
        Largest shift in bins.

    Returns
    -------
    similarity (np.ndarray)
        ``(m, m)`` float32.
    """

    def unit(stack: np.ndarray) -> np.ndarray:
        centred = stack - stack.mean(axis=(1, 2), keepdims=True)
        scale = np.maximum(np.sqrt((centred ** 2).sum(axis=(1, 2), keepdims=True)), 1e-12)
        return (centred / scale).reshape(stack.shape[0], -1).astype(np.float32)

    m, height, width = features.shape
    base = unit(features)
    padded = np.pad(features, ((0, 0), (max_shift, max_shift), (max_shift, max_shift)))
    similarity = np.full((m, m), -np.inf, dtype=np.float32)
    for shift_a in range(-max_shift, max_shift + 1):
        for shift_b in range(-max_shift, max_shift + 1):
            window = padded[:, max_shift + shift_a:max_shift + shift_a + height, max_shift + shift_b:max_shift + shift_b + width]
            np.maximum(similarity, base @ unit(window).T, out=similarity)
    return np.maximum(similarity, similarity.T)


def rank_similarity(similarity: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Description
    -----------
    Normalized similarity: for each USV i, its similarity to j becomes the share
    of i's similarities to the reference USVs (i itself left out) that are lower
    -- a percentile against i's own typical level, which keeps a multi-part USV
    (low correlation with everything) from looking dissimilar by default. The
    i-to-j and j-to-i percentiles are then averaged, so the matrix is symmetric.
    The diagonal is NaN.

    Parameters
    ----------
    similarity (np.ndarray)
        ``(m, m)`` raw similarities.
    reference (np.ndarray)
        Indices of the reference USVs.

    Returns
    -------
    ranked (np.ndarray)
        ``(m, m)`` float32 percentiles in [0, 1].
    """

    ranked = np.empty_like(similarity)
    for row in range(similarity.shape[0]):
        reference_values = np.sort(similarity[row, reference[reference != row]])
        ranked[row] = np.searchsorted(reference_values, similarity[row], side="left") / reference_values.size
    ranked = (ranked + ranked.T) / 2.0
    np.fill_diagonal(ranked, np.nan)
    return ranked


def group_contrast(filled: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Description
    -----------
    Per USV: its mean normalized similarity to the other USVs of its own group
    minus its mean normalized similarity to the USVs of all other groups.

    Parameters
    ----------
    filled (np.ndarray)
        ``(m, m)`` normalized similarities with a zero diagonal.
    labels (np.ndarray)
        ``(m,)`` group of every USV.

    Returns
    -------
    delta (np.ndarray)
        ``(m,)`` differences.
    """

    groups = np.unique(labels)
    counts = np.array([(labels == group).sum() for group in groups])
    onehot = (labels[:, None] == groups[None, :]).astype(np.float32)
    sums = filled @ onehot
    own_index = np.searchsorted(groups, labels)
    own_sum = sums[np.arange(labels.size), own_index]
    own_count = counts[own_index] - 1
    return own_sum / own_count - (sums.sum(axis=1) - own_sum) / (labels.size - 1 - own_count)


def aligned_category_mean(stack: np.ndarray, max_shift: int, iterations: int) -> np.ndarray:
    """
    Description
    -----------
    The typical shape of a set of spectrograms: starting from their plain mean,
    every spectrogram is shifted (up to ``max_shift`` bins in time and frequency)
    to where it overlaps the current template best, the shifted spectrograms are
    averaged into the next template, ``iterations`` times. It removes differences
    in pitch and timing within the shift range, so it shows a category's typical
    shape but understates its spread.

    Parameters
    ----------
    stack (np.ndarray)
        ``(m, h, w)`` spectrograms.
    max_shift (int)
        Largest shift in bins.
    iterations (int)
        Alignment rounds.

    Returns
    -------
    template (np.ndarray)
        ``(h, w)`` aligned mean.
    """

    stack = stack.astype(np.float32)
    template = stack.mean(axis=0)
    for _ in range(iterations):
        centred = template - template.mean()
        best = np.full(stack.shape[0], -np.inf)
        moved = np.empty_like(stack)
        for shift_f in range(-max_shift, max_shift + 1):
            for shift_t in range(-max_shift, max_shift + 1):
                candidate = np.roll(stack, (shift_f, shift_t), axis=(1, 2))
                score = candidate.reshape(stack.shape[0], -1) @ centred.ravel()
                better = score > best
                best[better] = score[better]
                moved[better] = candidate[better]
        template = moved.mean(axis=0)
    return template


def compute_category_similarity(sample: dict, label_grid: np.ndarray, watershed_grids: dict[str, np.ndarray], cfg: dict,
                                seed: int, message_output: Callable | None = None) -> dict:
    """
    Description
    -----------
    The spectrogram-similarity panels of the category figure, on the per-session
    sample (normalized similarity: module docstring).

    * Balanced sample: ``balanced_per_category`` random USVs per category,
      ranked against ``balanced_reference_per_category`` reference USVs per
      category of the same sample. (i) the category x category mean normalized
      similarity over all pairs of distinct USVs; (ii) per USV, its mean
      similarity to its own category minus to all other categories; (iii) per
      USV, the share of its ``closest_matches`` most similar USVs in each
      category, averaged per category (rows sum to 1, chance 1 / k).
    * Random sample: ``partition_sample_size`` random USVs (not balanced, so no
      partition is favoured), ranked against ``partition_reference_size`` random
      references. (iv) per partition, the per-USV own-group-minus-other-groups
      similarity averaged per group; the partition score is the mean of its group
      values.
    * Category averages: the aligned mean (``aligned_category_mean``) of
      ``average_size`` random masked, not time-stretched spectrograms per category.

    Parameters
    ----------
    sample (dict)
        ``build_qlvm_figure_sample`` output.
    label_grid (np.ndarray)
        The category bundle's ``(res, res)`` grid (``[y, x]``).
    watershed_grids (dict[str, np.ndarray])
        Display name -> label grid of each comparison partition of the same map;
        may be empty.
    cfg (dict)
        The ``qlvm_figures`` settings block.
    seed (int)
        Seed of every random draw.
    message_output (Callable | None)
        Logger; defaults to ``print``.

    Returns
    -------
    similarity (dict)
        ``matrix``, ``difference`` and ``difference_labels``, ``neighbor_shares``,
        ``partition_scores`` (name -> per-group values; the categories first),
        ``aligned_means`` ``(k, 128, 128)``, ``ms_per_bin``.

    Raises
    ------
    ValueError
        The sample's categories disagree with ``label_grid`` (``check_categories_match_bundle``).
    """

    log = message_output or print
    categories_cfg = cfg["categories"]
    downsample = categories_cfg["similarity_downsample"]
    max_shift = categories_cfg["similarity_max_shift"]
    labels = sample["category"].astype(int)
    images = sample["images"]
    check_categories_match_bundle(pls.DataFrame({"session_id": sample["session_id"], "qlvm1": sample["xy"][:, 0],
                                                 "qlvm2": sample["xy"][:, 1], QLVM_CATEGORY_COLUMN: labels}), label_grid)
    n_categories = int(label_grid.max())
    category_ids = np.arange(1, n_categories + 1)
    rng = np.random.default_rng(seed)

    balanced = np.concatenate([np.sort(rng.choice(np.flatnonzero(labels == a), categories_cfg["balanced_per_category"], replace=False))
                               for a in category_ids])
    balanced_labels = labels[balanced]
    reference = np.concatenate([rng.choice(np.flatnonzero(balanced_labels == a), categories_cfg["balanced_reference_per_category"],
                                           replace=False) for a in category_ids])
    ranked = rank_similarity(aligned_correlation(similarity_features(images[balanced], downsample), max_shift), reference)
    matrix = np.array([[float(np.nanmean(ranked[np.ix_(balanced_labels == a, balanced_labels == b)])) for b in category_ids]
                       for a in category_ids])
    filled = np.nan_to_num(ranked, nan=0.0)
    difference = group_contrast(filled, balanced_labels)
    np.fill_diagonal(ranked, -np.inf)
    closest = balanced_labels[np.argpartition(-ranked, categories_cfg["closest_matches"], axis=1)[:, :categories_cfg["closest_matches"]]]
    neighbor_shares = np.array([[(closest[balanced_labels == a] == b).mean() for b in category_ids] for a in category_ids])
    del ranked, filled
    log("[qlvm-categories] balanced similarity done.")

    chosen = np.sort(rng.choice(labels.size, categories_cfg["partition_sample_size"], replace=False))
    reference = rng.choice(chosen.size, categories_cfg["partition_reference_size"], replace=False)
    filled = np.nan_to_num(rank_similarity(aligned_correlation(similarity_features(images[chosen], downsample), max_shift), reference),
                           nan=0.0).astype(np.float32)
    partition_scores = {}
    for name, grid in {f"content ridges\n({n_categories})": label_grid, **watershed_grids}.items():
        grid_rows, grid_cols = call_pixels(sample["xy"][chosen], grid.shape[0])
        partition_labels = grid[grid_rows, grid_cols]
        delta = group_contrast(filled, partition_labels)
        partition_scores[name] = np.array([delta[partition_labels == group].mean() for group in np.unique(partition_labels)])
    del filled
    log("[qlvm-categories] partition scores done.")

    aligned_means = np.array([aligned_category_mean(images[rng.choice(np.flatnonzero(labels == a), categories_cfg["average_size"],
                                                                      replace=False)],
                                                    categories_cfg["align_max_shift"], categories_cfg["align_iterations"])
                              for a in category_ids])
    durations = sample["properties"][:, REGULAR_MAP_PROPERTY_COLUMNS.index("duration")]
    valid = sample["lengths"] > 0
    return {"matrix": matrix, "difference": difference, "difference_labels": balanced_labels, "neighbor_shares": neighbor_shares,
            "partition_scores": partition_scores, "aligned_means": aligned_means,
            "ms_per_bin": float(np.median(durations[valid] / sample["lengths"][valid]) * 1000.0)}


def compute_category_boundary_fields(usvs: pls.DataFrame, bundle: dict, cfg: dict, seed: int) -> dict:
    """
    Description
    -----------
    The boundary panels of the category figure, on at most ``n_usvs`` random
    cohort USVs. The category boundaries are the bundle's (fixed); the density
    (``periodic_density``, bandwidth ``overview.density_bandwidth``) and the
    content fields are recomputed from these USVs with the build's own functions
    and settings (``qlvm_categories.property_ranks``, ``content_fields`` with the
    bundle's ``field_sigma``, the content change with its ``change_span``), the
    content change split by property (each property's own term of the root sum
    of squares). Along-boundary test: the mean density and mean content change
    on the boundary pixels, each divided by its map mean, against the same
    boundary pattern shifted rigidly around the torus (``categories.n_shifts``
    shifts). Each property's share of the content change on the boundary pixels.

    Parameters
    ----------
    usvs (pls.DataFrame)
        Cohort USVs (``load_regular_map_cohort_usvs``).
    bundle (dict)
        ``os_utils.load_qlvm_category_bundle()``.
    cfg (dict)
        The ``qlvm_figures`` settings block.
    seed (int)
        Seed of the draw and of the shifts.

    Returns
    -------
    fields (dict)
        ``n_usvs``, ``density``, ``change``, ``per_property`` ``(6, res, res)``,
        ``nulls`` / ``observed`` (measure -> shifted values / real value) and
        ``share`` ``(6,)``.
    """

    check_categories_match_bundle(usvs, bundle["label_grid"])
    if usvs.height > cfg["n_usvs"]:
        usvs = usvs.sample(n=cfg["n_usvs"], seed=seed)
    build = bundle["build_config"]["build_qlvm_categories"]
    label_grid = bundle["label_grid"]
    resolution = label_grid.shape[0]
    rows, cols = call_pixels(np.column_stack([usvs["qlvm1"].to_numpy(), usvs["qlvm2"].to_numpy()]), resolution)
    values = np.column_stack([usvs[column].to_numpy() for column, _ in BOUNDARY_PROPERTIES])
    fields = content_fields(rows, cols, property_ranks(values), np.ones(usvs.height), float(build["field_sigma"]), resolution)
    span = int(build["change_span"])
    per_property = np.sqrt((np.roll(fields, -span, axis=2) - np.roll(fields, span, axis=2)) ** 2
                           + (np.roll(fields, -span, axis=1) - np.roll(fields, span, axis=1)) ** 2)
    change = np.sqrt((per_property ** 2).sum(axis=0))
    density = periodic_density(rows, cols, resolution, cfg["overview"]["density_bandwidth"])
    boundary = (label_grid != np.roll(label_grid, 1, axis=0)) | (label_grid != np.roll(label_grid, 1, axis=1))
    shifts = np.random.default_rng(seed).integers(0, resolution, size=(cfg["categories"]["n_shifts"], 2))
    nulls, observed = {}, {}
    for name, field in (("density", density), ("content_change", change)):
        relative = field / field.mean()
        observed[name] = float(relative[boundary].mean())
        nulls[name] = np.array([relative[np.roll(boundary, (int(a), int(b)), axis=(0, 1))].mean() for a, b in shifts])
    share = per_property[:, boundary] ** 2
    share = (share / share.sum(axis=0, keepdims=True)).mean(axis=1)
    return {"n_usvs": int(usvs.height), "density": density, "change": change, "per_property": per_property,
            "nulls": nulls, "observed": observed, "share": share}


def style_torus(axis, title: str, title_size: float) -> None:
    """
    Description
    -----------
    Square unit-torus axes without ticks, with a title.

    Parameters
    ----------
    axis (matplotlib.axes.Axes)
        Axes to style.
    title (str)
        Panel title.
    title_size (float)
        Title font size.

    Returns
    -------
    None
    """

    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_title(title, fontsize=title_size, color=TEXT_COLOR)


def draw_category_field(axis, field: np.ndarray, title: str, bundle: dict, colorbar_label: str, cmap_name: str) -> None:
    """
    Description
    -----------
    A torus field (1st-99th percentile) with the category boundaries and numbers,
    and a labelled colour bar without tick marks.

    Parameters
    ----------
    axis (matplotlib.axes.Axes)
        Axes to draw on.
    field (np.ndarray)
        ``(res, res)`` field indexed ``[y, x]``.
    title (str)
        Panel title.
    bundle (dict)
        ``os_utils.load_qlvm_category_bundle()``.
    colorbar_label (str)
        Colour-bar label with its unit in parentheses.
    cmap_name (str)
        Sequential colormap.

    Returns
    -------
    None
    """

    image = axis.imshow(field, origin="lower", extent=(0, 1, 0, 1), cmap=cmap_name,
                        vmin=np.percentile(field, 1), vmax=np.percentile(field, 99), interpolation="nearest")
    colorbar = axis.figure.colorbar(image, ax=axis, fraction=0.046, pad=0.02)
    colorbar.ax.tick_params(length=0, labelsize=COLORBAR_TICK_SIZE)
    colorbar.set_label(colorbar_label, fontsize=COLORBAR_LABEL_SIZE)
    draw_category_outlines(axis, bundle["axis"], bundle["axis"], bundle["label_grid"], colors=BOUNDARY_COLOR, linewidths=1.4, zorder=3)
    for index, (label_x, label_y) in enumerate(bundle["centers"]):
        axis.text(min(max(label_x, 0.04), 0.96), min(max(label_y, 0.04), 0.96), str(index + 1), color=BOUNDARY_COLOR,
                  fontsize=13, ha="center", va="center", zorder=4)
    style_torus(axis, title, TITLE_SIZE)


def plot_category_boundaries(fields: dict, similarity: dict, bundle: dict, cfg: dict, cmap_name: str) -> Figure:
    """
    Description
    -----------
    Draws the category figure (module docstring) from
    ``compute_category_boundary_fields`` and ``compute_category_similarity``.
    Row 1: the aligned category means (2 x 2), the density, the content change,
    the along-boundary test and the property shares. Row 2: the content change
    per property. Row 3: (i) the category similarity matrix on a diverging scale
    centred on chance (0.5); (ii) the per-USV own-minus-outside similarity
    (violins, median marked, share above 0 to one decimal); (iii) where the
    closest matches fall, coloured by log2(share / chance) symmetric to
    +-``categories.closest_log2_limit`` (lower shares take the deepest blue, the
    arrow on the colour bar); (iv) the partition scores (bars) with their group
    values (dots). (iii) is shifted so its visible gap to (ii) equals the gap
    between (i) and (ii).

    Parameters
    ----------
    fields (dict)
        ``compute_category_boundary_fields`` output.
    similarity (dict)
        ``compute_category_similarity`` output.
    bundle (dict)
        ``os_utils.load_qlvm_category_bundle()``.
    cfg (dict)
        The ``qlvm_figures`` settings block.
    cmap_name (str)
        Sequential colormap of the maps and spectrograms.

    Returns
    -------
    fig (Figure)
        The figure.
    """

    apply_plot_style()
    categories_cfg = cfg["categories"]
    colors = cfg["category_colors"]
    n_categories = similarity["matrix"].shape[0]
    chance_cmap = LinearSegmentedColormap.from_list("chance", CHANCE_COLORS)
    fig = Figure(figsize=(32.5, 16.6), facecolor=FIGURE_FACE)
    outer = fig.add_gridspec(3, 1, height_ratios=[0.9, 0.75, 0.95], hspace=0.24)
    top = outer[0].subgridspec(1, 5, width_ratios=[0.95, 1, 1, 0.8, 0.8], wspace=0.5)
    middle = outer[1].subgridspec(1, len(BOUNDARY_PROPERTIES), wspace=0.30)
    bottom = outer[2].subgridspec(1, 4, width_ratios=[0.9, 0.8, 0.8, 1.15], wspace=0.5)
    averages_grid = top[0].subgridspec(2, 2, wspace=0.22, hspace=0.32)

    extent = (0, SPECTROGRAM_SIZE * similarity["ms_per_bin"], FREQUENCY_RANGE_KHZ[0], FREQUENCY_RANGE_KHZ[1])
    for category in range(n_categories):
        row, column = divmod(category, 2)
        axis = fig.add_subplot(averages_grid[row, column])
        axis.imshow(similarity["aligned_means"][category], origin="lower", cmap=cmap_name, aspect="auto", extent=extent)
        axis.set_xlim(0, categories_cfg["average_time_ms"])
        axis.set_xticks(np.arange(0, categories_cfg["average_time_ms"] + 1, 50))
        axis.set_yticks([40, 80, 120])
        axis.set_title(f"category {category + 1}", fontsize=TICK_SIZE, color=TEXT_COLOR)
        if row == 1:
            axis.set_xlabel("time (ms)", fontsize=TICK_SIZE)
        else:
            axis.set_xticklabels([])
        if column == 0:
            axis.set_ylabel("frequency (kHz)", fontsize=TICK_SIZE)
        else:
            axis.set_yticklabels([])
        axis.tick_params(labelsize=TICK_SIZE - 2)

    draw_category_field(fig.add_subplot(top[1]), fields["density"], f"USV density ({fields['n_usvs']:,} USVs)", bundle,
                        "USV density (smoothed count)", cmap_name)
    draw_category_field(fig.add_subplot(top[2]), fields["change"], "content change (all six properties)", bundle,
                        "content change (Δ rank)", cmap_name)

    axis = fig.add_subplot(top[3])
    nulls, observed = fields["nulls"], fields["observed"]
    bins = np.linspace(min(nulls["density"].min(), nulls["content_change"].min(), 0.6),
                       max(observed["content_change"], nulls["content_change"].max()) * 1.05, categories_cfg["histogram_bins"] + 1)
    measures = (("density", DENSITY_COLOR, "density"), ("content_change", CHANGE_COLOR, "content change"))
    for name, color, label in measures:
        axis.hist(nulls[name], bins=bins, histtype="stepfilled", color=color, alpha=0.55,
                  label=f"{label}: shifted boundaries (mean {nulls[name].mean():.2f})")
    for name, color, label in measures:
        axis.axvline(observed[name], color=color, linewidth=2.4, label=f"{label}: real boundaries ({observed[name]:.2f})")
    axis.axvline(1.0, color=TEXT_COLOR, linewidth=0.8, linestyle=":")
    axis.set_xlabel("boundary pattern mean / map mean", fontsize=LABEL_SIZE)
    axis.set_ylabel(f"shifted patterns (of {nulls['density'].size})", fontsize=LABEL_SIZE)
    axis.set_title("along-boundary test (real vs shifted boundaries)", fontsize=TITLE_SIZE, color=TEXT_COLOR)
    axis.tick_params(labelsize=TICK_SIZE)
    axis.set_ylim(0, axis.get_ylim()[1] * 1.75)
    axis.legend(frameon=False, fontsize=TICK_SIZE - 2, loc="upper left")

    axis = fig.add_subplot(top[4])
    labels = [label for _, label in BOUNDARY_PROPERTIES]
    order = np.argsort(fields["share"])[::-1]
    axis.barh(np.arange(len(labels)), fields["share"][order] * 100, color=BAR_COLOR)
    axis.set_yticks(np.arange(len(labels)))
    axis.set_yticklabels([labels[i] for i in order], fontsize=LABEL_SIZE)
    axis.invert_yaxis()
    axis.set_xlabel("share of the content change on the boundaries (%)", fontsize=LABEL_SIZE)
    axis.set_title("what changes across the boundaries", fontsize=TITLE_SIZE, color=TEXT_COLOR)
    axis.tick_params(labelsize=TICK_SIZE)

    for column, (_, label) in enumerate(BOUNDARY_PROPERTIES):
        draw_category_field(fig.add_subplot(middle[column]), fields["per_property"][column], f"content change: {label}", bundle,
                            f"{label} change (Δ rank)", cmap_name)

    category_ticks = range(n_categories)
    category_labels = [str(category) for category in range(1, n_categories + 1)]
    axis = fig.add_subplot(bottom[0])
    table = similarity["matrix"]
    half = float(np.abs(table - 0.5).max())
    image = axis.imshow(table, cmap=chance_cmap, origin="upper", vmin=0.5 - half, vmax=0.5 + half)
    for a in range(n_categories):
        for b in range(n_categories):
            axis.text(b, a, f"{table[a, b]:.2f}", ha="center", va="center", fontsize=TICK_SIZE,
                      color=BOUNDARY_COLOR if abs(table[a, b] - 0.5) > 0.7 * half else TEXT_COLOR)
    axis.set_xticks(category_ticks)
    axis.set_xticklabels(category_labels)
    axis.set_yticks(category_ticks)
    axis.set_yticklabels(category_labels)
    axis.set_xlabel("category", fontsize=LABEL_SIZE)
    axis.set_ylabel("category", fontsize=LABEL_SIZE)
    axis.tick_params(labelsize=TICK_SIZE, length=0)
    axis.set_title(f"(i) mean similarity between categories\n({categories_cfg['balanced_per_category']:,} USVs per category)",
                   fontsize=TITLE_SIZE, color=TEXT_COLOR)
    colorbar = fig.colorbar(image, cax=make_axes_locatable(axis).append_axes("right", size="5%", pad=0.12))
    similarity_colorbar_axis = colorbar.ax
    colorbar.ax.tick_params(length=0, labelsize=COLORBAR_TICK_SIZE)
    colorbar.set_ticks([0.5 - half, 0.5, 0.5 + half])
    colorbar.set_ticklabels([f"{value:.2f}" for value in (0.5 - half, 0.5, 0.5 + half)])
    colorbar.set_label("mean normalized similarity (percentile)", fontsize=COLORBAR_LABEL_SIZE)

    axis = fig.add_subplot(bottom[1])
    attachment_axis = axis
    groups = [similarity["difference"][similarity["difference_labels"] == a + 1] for a in range(n_categories)]
    violins = axis.violinplot(groups, positions=np.arange(1, n_categories + 1), widths=0.8, showextrema=False, showmedians=True)
    for body, color in zip(violins["bodies"], colors, strict=False):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.75)
    violins["cmedians"].set_color(TEXT_COLOR)
    top_value = max(np.percentile(group, 99.5) for group in groups)
    for a, group in enumerate(groups):
        axis.text(a + 1, top_value + 0.06, f"{100 * (group > 0).mean():.1f}% > 0", ha="center", va="bottom", fontsize=TICK_SIZE)
    axis.axhline(0, color=TEXT_COLOR, linewidth=0.9, linestyle=":")
    axis.set_ylim(min(group.min() for group in groups) - 0.03, max(top_value + 0.13, max(group.max() for group in groups) + 0.02))
    axis.set_xticks(np.arange(1, n_categories + 1))
    axis.set_xticklabels(category_labels)
    axis.set_xlabel("category", fontsize=LABEL_SIZE)
    axis.set_ylabel("Δ mean normalized similarity", fontsize=LABEL_SIZE)
    axis.set_title(f"(ii) per-USV attachment to its category\n({categories_cfg['balanced_per_category']:,} USVs per category)",
                   fontsize=TITLE_SIZE, color=TEXT_COLOR)
    axis.tick_params(labelsize=TICK_SIZE)

    axis = fig.add_subplot(bottom[2])
    matches_axis = axis
    shares = similarity["neighbor_shares"]
    chance = 1.0 / n_categories
    limit = categories_cfg["closest_log2_limit"]
    # Shares below the scale (including 0, whose log is undefined) take the deepest blue; the cell text keeps the value.
    log_ratio = np.log2(np.maximum(shares, chance * 2.0 ** (-limit - 1)) / chance)
    image = axis.imshow(log_ratio, cmap=chance_cmap, origin="upper", vmin=-limit, vmax=limit)
    for a in range(n_categories):
        for b in range(n_categories):
            strong = abs(log_ratio[a, b]) > 0.7 * limit
            axis.text(b, a, f"{shares[a, b]:.2f}", ha="center", va="center", fontsize=TICK_SIZE, color=BOUNDARY_COLOR if strong else TEXT_COLOR)
    axis.set_xticks(category_ticks)
    axis.set_xticklabels(category_labels)
    axis.set_yticks(category_ticks)
    axis.set_yticklabels(category_labels)
    axis.set_xlabel("category of the closest matches", fontsize=LABEL_SIZE)
    axis.set_ylabel("category of the USV", fontsize=LABEL_SIZE)
    axis.tick_params(labelsize=TICK_SIZE, length=0)
    axis.set_title(f"(iii) where the {categories_cfg['closest_matches']} closest matches fall\n"
                   f"({categories_cfg['balanced_per_category']:,} USVs per category)", fontsize=TITLE_SIZE, color=TEXT_COLOR)
    colorbar = fig.colorbar(image, cax=make_axes_locatable(axis).append_axes("right", size="5%", pad=0.12), extend="min")
    matches_colorbar_axis = colorbar.ax
    tick_shares = [chance * 2.0 ** power for power in np.arange(-limit, limit + 1)]
    colorbar.set_ticks(np.log2(np.array(tick_shares) / chance))
    colorbar.set_ticklabels([f"{value:g}" for value in tick_shares])
    colorbar.ax.tick_params(length=0, labelsize=COLORBAR_TICK_SIZE)
    colorbar.set_label("share of closest matches (log scale)", fontsize=COLORBAR_LABEL_SIZE)

    axis = fig.add_subplot(bottom[3])
    for position, group_values in enumerate(similarity["partition_scores"].values()):
        axis.bar(position, np.mean(group_values), width=0.65, color=BAR_COLOR if position == 0 else WATERSHED_COLOR, zorder=2)
        jitter = np.linspace(-0.15, 0.15, len(group_values)) if len(group_values) > 1 else np.zeros(1)
        axis.scatter(position + jitter, group_values, s=34, color=GROUP_DOT_COLOR, edgecolors=TEXT_COLOR, linewidths=0.8, zorder=3)
        axis.text(position, np.mean(group_values) + 0.004, f"{np.mean(group_values):.3f}", ha="center", va="bottom", fontsize=TICK_SIZE,
                  zorder=4)
    axis.axhline(0, color=TEXT_COLOR, linewidth=0.8)
    axis.set_xticks(range(len(similarity["partition_scores"])))
    axis.set_xticklabels(list(similarity["partition_scores"]), fontsize=TICK_SIZE)
    axis.set_ylabel("partition score\n(Δ mean normalized similarity)", fontsize=LABEL_SIZE)
    axis.set_title(f"(iv) content-ridge categories vs watershed clusters\n({categories_cfg['partition_sample_size']:,} random USVs; "
                   "dots: groups)", fontsize=TITLE_SIZE, color=TEXT_COLOR)
    axis.tick_params(labelsize=TICK_SIZE)

    # Equal visible gaps in the similarity row: (iii) and its colour bar are shifted so the gap from (ii)
    # to (iii) equals the gap from (i)'s colour bar to (ii), labels included.
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    gap_before = attachment_axis.get_tightbbox(renderer).x0 - similarity_colorbar_axis.get_tightbbox(renderer).x1
    gap_after = matches_axis.get_tightbbox(renderer).x0 - attachment_axis.get_tightbbox(renderer).x1
    shift = (gap_after - gap_before) / fig.bbox.width
    for moved in (matches_axis, matches_colorbar_axis):
        position = moved.get_position()
        moved.set_axes_locator(None)
        moved.set_position([position.x0 - shift, position.y0, position.width, position.height])
    return fig


def read_thumbnail(h5: h5py.File, session_id: str, row_index: int) -> np.ndarray:
    """
    Description
    -----------
    One USV's thumbnail as the embedding explorer builds it: the spectrogram's
    valid ``[:, :duration]`` slice times the union of its SAM masks, centred in
    the storage window.

    Parameters
    ----------
    h5 (h5py.File)
        The session's spectrogram H5.
    session_id (str)
        Session id.
    row_index (int)
        Summary row of the USV.

    Returns
    -------
    tile (np.ndarray)
        ``(n_freq, window)`` array.
    """

    group = h5[f"spectrogram/{session_id}"]
    spec = group["spectrograms"][row_index, :, :].astype(np.float32)
    window = spec.shape[1]
    duration = max(1, min(int(group["durations"][row_index]), window))
    shown = spec[:, :duration]
    mask_key = f"mask/{session_id}"
    if mask_key in h5:
        matching = np.where(h5[mask_key]["spectrogram_index"][:] == row_index)[0]
        if matching.size:
            shown = shown * np.any(h5[mask_key]["segmentations"][matching, :, :duration], axis=0).astype(np.float32)
    tile = np.zeros((spec.shape[0], window), dtype=np.float32)
    pad_left = (window - duration) // 2
    tile[:, pad_left:pad_left + duration] = shown
    return tile


def category_examples(usvs: pls.DataFrame, label_grid: np.ndarray, cfg: dict, seed: int) -> list[tuple]:
    """
    Description
    -----------
    Per category: its densest point (the maximum of the density of the
    category's own USVs, restricted to its pixels of the grid) and
    ``overview.n_examples`` of its USVs sampled along a spiral of radius
    ``overview.spiral_radius`` around that point (``_pick_spiral_with_grid`` in
    torus-displacement coordinates).

    Parameters
    ----------
    usvs (pls.DataFrame)
        The overview's USVs (``session_id``, ``row_index``, ``qlvm1``, ``qlvm2``, ``qlvm_category``).
    label_grid (np.ndarray)
        The bundle's ``(res, res)`` category grid (``[y, x]``).
    cfg (dict)
        The ``qlvm_figures`` settings block.
    seed (int)
        Seed of the spiral sampler.

    Returns
    -------
    examples (list[tuple])
        Per category ``(centre, picks, path)``: centre ``(2,)``, picks
        (pls.DataFrame) in spiral order, path ``(m, 2)`` absolute spiral positions.
    """

    overview = cfg["overview"]
    resolution = label_grid.shape[0]
    rng = np.random.default_rng(seed)
    examples = []
    for category in range(1, int(label_grid.max()) + 1):
        members = usvs.filter(pls.col(QLVM_CATEGORY_COLUMN) == category)
        points = np.column_stack([members["qlvm1"].to_numpy(), members["qlvm2"].to_numpy()])
        rows, cols = call_pixels(points, resolution)
        density = periodic_density(rows, cols, resolution, overview["density_bandwidth"])
        density[label_grid != category] = -np.inf
        row, col = np.unravel_index(np.argmax(density), density.shape)
        centre = np.array([(col + 0.5) / resolution, (row + 0.5) / resolution])
        local, path_x, path_y = _pick_spiral_with_grid((points - centre + 0.5) % 1.0 - 0.5, n_per=overview["n_examples"], cx0=0.0,
                                                       cy0=0.0, r_max=overview["spiral_radius"], labels_grid=None, xx=None, yy=None,
                                                       cluster_label=-1, n_turns=overview["spiral_turns"], rng=rng)
        examples.append((centre, members[local.tolist()], (np.column_stack([path_x, path_y]) + centre) % 1.0))
    return examples


def plot_qlvm_overview(usvs: pls.DataFrame, bundle: dict, roots: dict[str, pathlib.Path], cfg: dict, cmap_name: str,
                       seed: int) -> tuple[Figure, pls.DataFrame]:
    """
    Description
    -----------
    Draws the QLVM overview figure on at most ``n_usvs`` random cohort USVs.
    Row 1: the USV density (``periodic_density``, 1st-99th percentile of the
    occupied pixels); the bundle's content-change field with the category
    boundaries and numbers; the USVs coloured by category with each category's
    spiral and densest point, the legend giving each category's share; and per
    category ``overview.n_examples`` spectrograms along its spiral (masked, at
    the true duration, centred). Row 2: the USVs coloured by the five
    ``OVERVIEW_PROPERTIES`` (1st-99th percentile) with the boundaries in black.

    Parameters
    ----------
    usvs (pls.DataFrame)
        Cohort USVs (``load_regular_map_cohort_usvs``).
    bundle (dict)
        ``os_utils.load_qlvm_category_bundle()``.
    roots (dict[str, pathlib.Path])
        Session id -> root directory (``session_roots_by_id``).
    cfg (dict)
        The ``qlvm_figures`` settings block.
    cmap_name (str)
        Sequential colormap.
    seed (int)
        Seed of the draw, the drawing order and the spiral sampler.

    Returns
    -------
    fig (Figure)
        The figure.
    examples_table (pls.DataFrame)
        The example USVs: ``session_id``, ``row_index``, ``category``,
        ``description``, ``spiral_order``, ``centre_x``, ``centre_y``.
    """

    apply_plot_style()
    check_categories_match_bundle(usvs, bundle["label_grid"])
    if usvs.height > cfg["n_usvs"]:
        usvs = usvs.sample(n=cfg["n_usvs"], seed=seed)
    overview = cfg["overview"]
    colors = cfg["category_colors"]
    label_grid = bundle["label_grid"]
    resolution = label_grid.shape[0]
    descriptions = bundle["descriptions"]
    n_categories = len(descriptions)
    change = np.load(pathlib.Path(bundle["directory"]) / QLVM_CATEGORY_GRIDS_NAME)["content_change"]
    examples = category_examples(usvs, label_grid, cfg, seed)

    fig = Figure(figsize=(30, 12.5), facecolor=FIGURE_FACE)
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.14)
    top = outer[0].subgridspec(1, 4, width_ratios=[1, 1, 1, 2.35], wspace=0.06)
    bottom = outer[1].subgridspec(1, len(OVERVIEW_PROPERTIES), wspace=0.06)

    points = np.column_stack([usvs["qlvm1"].to_numpy(), usvs["qlvm2"].to_numpy()])
    rows, cols = call_pixels(points, resolution)
    density = periodic_density(rows, cols, resolution, overview["density_bandwidth"])
    occupied = density[density >= 1.0]
    axis = fig.add_subplot(top[0])
    axis.imshow(density, origin="lower", extent=(0, 1, 0, 1), cmap=cmap_name, vmin=np.percentile(occupied, 1),
                vmax=np.percentile(occupied, 99), interpolation="nearest")
    style_torus(axis, f"USV density ({usvs.height:,} USVs)", 11)

    axis = fig.add_subplot(top[1])
    axis.imshow(change, origin="lower", extent=(0, 1, 0, 1), cmap=cmap_name, vmin=np.percentile(change, 1),
                vmax=np.percentile(change, 99), interpolation="nearest")
    draw_category_outlines(axis, bundle["axis"], bundle["axis"], label_grid, colors=BOUNDARY_COLOR, linewidths=1.6, zorder=3)
    for index, (label_x, label_y) in enumerate(bundle["centers"]):
        axis.text(min(max(label_x, 0.04), 0.96), min(max(label_y, 0.04), 0.96), str(index + 1), color=BOUNDARY_COLOR, fontsize=13,
                  fontweight="bold", ha="center", va="center", zorder=4)
    style_torus(axis, "content ridges and category boundaries", 11)

    axis = fig.add_subplot(top[2])
    order = np.random.default_rng(seed).permutation(usvs.height)
    labels = usvs[QLVM_CATEGORY_COLUMN].to_numpy()
    axis.scatter(points[order, 0], points[order, 1], s=0.6, c=np.array(colors)[labels[order] - 1], linewidths=0, rasterized=True)
    for centre, _, path in examples:
        jumps = np.flatnonzero(np.abs(np.diff(path, axis=0)).max(axis=1) > 0.5) + 1
        for segment in np.split(path, jumps):
            axis.plot(segment[:, 0], segment[:, 1], color=TEXT_COLOR, linewidth=0.8, zorder=4)
        axis.scatter(*centre, s=40, color=FIGURE_FACE, edgecolor=TEXT_COLOR, linewidth=1.0, zorder=5)
    handles = [Line2D([], [], marker="o", linestyle="", markersize=7, color=colors[index],
                      label=f"{index + 1} {descriptions[index]} ({100.0 * float((labels == index + 1).mean()):.1f}%)")
               for index in range(n_categories)]
    axis.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=2, frameon=False, fontsize=9)
    style_torus(axis, "categories (spiral centres = densest point)", 11)

    thumbnails = top[3].subgridspec(n_categories, overview["n_examples"], wspace=0.04, hspace=0.12)
    records = []
    for row, (centre, picks, _) in enumerate(examples):
        for column in range(overview["n_examples"]):
            axis = fig.add_subplot(thumbnails[row, column])
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_edgecolor(colors[row])
                spine.set_linewidth(2.0)
            if column >= picks.height:
                axis.set_facecolor(FIGURE_FACE)
                continue
            pick = picks.row(column, named=True)
            with h5py.File(session_spectrogram_file(roots[pick["session_id"]]), "r") as h5:
                tile = read_thumbnail(h5, pick["session_id"], int(pick["row_index"]))
            axis.imshow(tile, origin="lower", aspect="auto", cmap=cmap_name, vmin=0.0, vmax=float(tile.max()) or 1.0)
            if column == 0:
                axis.set_ylabel(f"{row + 1} {descriptions[row]}", fontsize=10, color=TEXT_COLOR)
            records.append({"session_id": pick["session_id"], "row_index": int(pick["row_index"]), "category": row + 1,
                            "description": descriptions[row], "spiral_order": column + 1, "centre_x": float(centre[0]),
                            "centre_y": float(centre[1])})

    shuffle = np.random.default_rng(seed).permutation(usvs.height)
    for column, (name, label, scale) in enumerate(OVERVIEW_PROPERTIES):
        axis = fig.add_subplot(bottom[column])
        values = usvs[name].to_numpy() * scale
        image = axis.scatter(points[shuffle, 0], points[shuffle, 1], c=values[shuffle], s=1.2, cmap=cmap_name,
                             vmin=np.percentile(values, 1), vmax=np.percentile(values, 99), linewidths=0, rasterized=True)
        draw_category_outlines(axis, bundle["axis"], bundle["axis"], label_grid, colors=PROPERTY_BOUNDARY_COLOR, linewidths=0.8, zorder=3)
        style_torus(axis, label, 11)
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.02)
    return fig, pls.DataFrame(records)
