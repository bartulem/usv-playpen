"""
@author: bartulem

Content-ridge categories of a QLVM torus: build them once from a corpus of
embedded calls, then label the calls of any session with them.

A category is a part of the torus inside which the calls' acoustic content
changes little, bounded by ridges where it changes fast. The build
(:class:`QLVMCategoryBuilder`, ``build-qlvm-categories``), from every corpus
call's torus position and six acoustic properties:

1. **Ranks.** Each property (by default ``mask_count``, ``durations``,
   ``freq_bandwidth_hz``, ``spectral_entropy``, ``mean_freq_hz``,
   ``loudness_db``) becomes its rank over all calls, ``(rank - 0.5) / n`` in
   ``(0, 1)`` (average rank for ties), so every property counts equally whatever
   its unit (:func:`property_ranks`).
2. **Content fields.** Every call sits in one pixel of a periodic
   ``grid_resolution`` x ``grid_resolution`` grid (``floor(y * res)``,
   ``floor(x * res)``); each property's field is the Gaussian kernel average of
   its ranks (``field_sigma`` pixels, wrapped at the seams): the smoothed sum of
   the calls' (weighted) ranks over the smoothed (weighted) call count
   (:func:`content_fields`).
3. **Content change.** At every pixel, the root sum of squares, over x, y and the
   properties, of the periodic central differences of the fields at offset
   ``change_span`` pixels (:func:`content_change`).
4. **Basins.** Marker watershed of the content-change field on a 3 x 3 tiling of
   the torus, so floods cross the seams: the field is blurred by
   ``marker_sigma`` pixels, its local minima over a ``2 * marker_distance + 1``
   pixel window are the markers (every copy of a marker carries the same label),
   and the central tile is kept (:func:`periodic_basins`). Basin boundaries thus
   run along the ridges of content change.
5. **Joins.** Neighbouring basins are joined one pair at a time. While a region
   holds less than ``size_floor`` of the (weighted) calls, the smallest such
   region joins the neighbour it shares the lowest mean ridge with; otherwise
   the pair of neighbours with the lowest mean ridge height along their shared
   boundary joins; down to ``n_categories`` regions, which must all satisfy the
   floor (:func:`join_basins`). Categories are numbered by (weighted) call
   count, largest first (:func:`partition_grid`).
6. **Consensus.** ``n_bootstraps`` resamples of the sessions with replacement
   (seed ``bootstrap_seed + i``; every call weighted by how often its session was
   drawn) rebuild steps 2-5; each resample's categories are matched one-to-one
   to the full-data categories by their pixel overlap (Hungarian assignment,
   :func:`align_to_reference`), and every pixel collects one vote per resample
   that reached ``n_categories`` categories under the floor. The consensus
   category of a pixel is the one with most votes; its ``agreement`` is that
   category's share of the votes (:func:`consensus_vote`).

Each call takes the consensus category of the pixel under its position, with
that pixel's agreement; a call is ``uncertain`` when its agreement is below
``uncertain_agreement``. Categories are numbered ``1`` ... ``k`` and named ``R-1``
... ``R-k`` in the order ``category_order`` gives (category ``i`` is the
category whose full-data call count ranks ``category_order[i - 1]``, largest
first; empty keeps the size order, :func:`apply_category_order`), each with the
description ``category_descriptions`` gives it (``category_order`` ``[1, 4, 2, 3]``
with ``simple``, ``biphones``, ``intermediate``, ``complex`` in the shipped
settings).

The build writes, into its output directory: ``category_grids.npz``
(``label_grid`` (int16, ``1..k``, indexed ``[y, x]``), ``agreement``,
``reference_grid`` (the full-data partition, ``1..k``), ``content_change``,
``basins`` and ``density`` (the smoothed call count)), ``category_nomenclature.json``
(the categories' names, descriptions, call counts and label positions, plus the
uncertainty threshold and grid resolution), ``category_call_labels.csv`` (the
corpus calls' categories) and ``build_config.json`` (the settings and inputs).

The assignment (:class:`QLVMCategoryAssigner`, ``assign-qlvm-categories``) reads
such a directory and labels the calls of one session from their torus
coordinates ``<P>1`` / ``<P>2`` in its ``*_usv_summary.csv``, with the same pixel
rule (:func:`qlvm_latents.label_grid_lookup`), writing ``<P>_category`` (int,
``1..k`` for ``R-1`` ... ``R-k``) only; calls without coordinates get nulls. The
per-call agreement and uncertain flag do not go into the summary: they stay
computable from the category directory and the call's coordinates
(:func:`assign_categories` on :func:`load_category_bundle`), and any
``<P>_category_agreement`` / ``<P>_category_uncertain`` column an earlier version
wrote is removed. Run it after ``infer-qlvm-latents``, which drops the
``<P>_category`` column (and those confidence columns) of every prefix it embeds.
With an empty ``category_directory`` / ``coordinate_prefix`` it labels with the
production bundle, ``os_utils.QLVM_CATEGORY_BUNDLE_DIRECTORY``, on the regular map
``qlvm`` (``os_utils.derive_spectrogram_model_paths``): the bundle every QLVM
figure loads its category grid from (``os_utils.load_qlvm_category_bundle``), so
the summaries' ``qlvm_category`` and the drawn boundaries are one partition.
"""

from __future__ import annotations

import json
import pathlib
from collections.abc import Callable
from datetime import datetime

import click
import numpy as np
import pandas as pd
import polars as pls
from click.core import ParameterSource
from joblib import Parallel, delayed
from scipy.ndimage import distance_transform_edt, gaussian_filter, label, minimum_filter
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata
from skimage.segmentation import watershed

from ..cli_utils import modify_settings_json_for_cli
from ..os_utils import (
    CATEGORY_CONFIDENCE_SUFFIXES,
    QLVM_CATEGORY_BUILD_CONFIG_NAME,
    QLVM_CATEGORY_CALL_LABELS_NAME,
    QLVM_CATEGORY_GRIDS_NAME,
    QLVM_CATEGORY_NOMENCLATURE_NAME,
    atomic_output_path,
    derive_spectrogram_model_paths,
    first_match_or_raise,
    order_usv_summary_columns,
    read_qlvm_category_bundle,
)
from ..time_utils import is_gui_context, smart_wait
from .build_qlvm_training_set import file_sha256
from .qlvm_latents import label_grid_lookup

# Files a category directory holds (named once, in os_utils, where every reader of
# a bundle finds them too).
CATEGORY_GRIDS_NAME = QLVM_CATEGORY_GRIDS_NAME
CATEGORY_NOMENCLATURE_NAME = QLVM_CATEGORY_NOMENCLATURE_NAME
CATEGORY_CALL_LABELS_NAME = QLVM_CATEGORY_CALL_LABELS_NAME
CATEGORY_BUILD_CONFIG_NAME = QLVM_CATEGORY_BUILD_CONFIG_NAME

# The columns a positions table and a properties table must carry besides the properties.
POSITION_COLUMNS = ("spec_id", "x", "y")
PROPERTY_ID_COLUMNS = ("spec_id", "session_id")


def property_ranks(values: np.ndarray) -> np.ndarray:
    """
    Description
    -----------
    Ranks every property over all calls: ``(rank - 0.5) / n``, ``rank`` the
    1-based average rank (``scipy.stats.rankdata``, ties sharing their mean rank),
    so each column lies in ``(0, 1)`` whatever the property's unit.

    Parameters
    ----------
    values (np.ndarray)
        ``(n_calls, n_properties)`` raw property values; all finite.

    Returns
    -------
    ranks (np.ndarray)
        ``(n_calls, n_properties)`` float64 ranks.
    """

    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        error_message = f"property_ranks: {int((~np.isfinite(values)).sum())} property values are not finite."
        raise ValueError(error_message)
    n_calls = values.shape[0]
    return np.column_stack([(rankdata(values[:, index]) - 0.5) / n_calls for index in range(values.shape[1])])


def call_pixels(coordinates: np.ndarray, resolution: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Description
    -----------
    The grid pixel of every call: row ``int(y * res)``, column ``int(x * res)``,
    each clipped to ``[0, res - 1]`` (truncation equals the floor for coordinates
    in ``[0, 1)``, where every QLVM coordinate lies).

    Parameters
    ----------
    coordinates (np.ndarray)
        ``(n_calls, 2)`` torus coordinates ``(x, y)``.
    resolution (int)
        Grid resolution in pixels per side.

    Returns
    -------
    rows (np.ndarray)
        ``(n_calls,)`` int64 pixel rows.
    cols (np.ndarray)
        ``(n_calls,)`` int64 pixel columns.
    """

    coordinates = np.asarray(coordinates, dtype=np.float64).reshape(-1, 2)
    rows = np.clip((coordinates[:, 1] * resolution).astype(int), 0, resolution - 1)
    cols = np.clip((coordinates[:, 0] * resolution).astype(int), 0, resolution - 1)
    return rows, cols


def content_fields(
    rows: np.ndarray,
    cols: np.ndarray,
    features: np.ndarray,
    weights: np.ndarray,
    sigma: float,
    resolution: int,
) -> np.ndarray:
    """
    Description
    -----------
    Kernel-average field of every feature on the periodic grid: the Gaussian
    smoothing (``sigma`` pixels, ``mode='wrap'``) of the weighted feature sums
    per pixel divided by the smoothing of the weighted call counts per pixel
    (floored at 1e-9, so a pixel far from every call does not divide by zero).

    Parameters
    ----------
    rows (np.ndarray)
        ``(n_calls,)`` pixel row of each call.
    cols (np.ndarray)
        ``(n_calls,)`` pixel column of each call.
    features (np.ndarray)
        ``(n_calls, p)`` features (property ranks).
    weights (np.ndarray)
        ``(n_calls,)`` call weights (1 for a plain build; session draw counts in a
        bootstrap).
    sigma (float)
        Gaussian smoothing in pixels.
    resolution (int)
        Grid resolution in pixels per side.

    Returns
    -------
    fields (np.ndarray)
        ``(p, resolution, resolution)`` float64 fields, indexed ``[property, y, x]``.
    """

    count = np.zeros((resolution, resolution))
    np.add.at(count, (rows, cols), weights)
    denominator = np.maximum(gaussian_filter(count, sigma, mode='wrap'), 1e-9)
    fields = np.empty((features.shape[1], resolution, resolution))
    for index in range(features.shape[1]):
        total = np.zeros((resolution, resolution))
        np.add.at(total, (rows, cols), features[:, index] * weights)
        fields[index] = gaussian_filter(total, sigma, mode='wrap') / denominator
    return fields


def content_change(fields: np.ndarray, span: int) -> np.ndarray:
    """
    Description
    -----------
    Content-change magnitude of a stack of fields: the periodic central
    differences at offset ``span`` pixels along x and y,
    ``f(i + span) - f(i - span)``, squared, summed over both axes and every
    field, and square-rooted.

    Parameters
    ----------
    fields (np.ndarray)
        ``(p, res, res)`` fields.
    span (int)
        Offset in pixels of each side of the difference.

    Returns
    -------
    change (np.ndarray)
        ``(res, res)`` float64 content change.
    """

    dx = np.roll(fields, -span, axis=2) - np.roll(fields, span, axis=2)
    dy = np.roll(fields, -span, axis=1) - np.roll(fields, span, axis=1)
    return np.sqrt((dx ** 2 + dy ** 2).sum(axis=0))


def periodic_basins(change: np.ndarray, marker_sigma: float, marker_distance: int) -> np.ndarray:
    """
    Description
    -----------
    Marker watershed of the content-change field on the torus. The field is
    blurred (``marker_sigma`` pixels, wrapped); its markers are the pixels equal
    to the minimum of the blurred field over a ``2 * marker_distance + 1`` pixel
    window (wrapped), joined into connected markers (8-connectivity) on a 3 x 3
    tiling so a marker crossing a seam stays one marker, and numbered ``1..n`` in
    the order of their labels in the central tile. The flood runs on the 3 x 3
    tiling of the blurred field with every copy of a marker carrying the same
    label, and the central tile is returned.

    Parameters
    ----------
    change (np.ndarray)
        ``(res, res)`` content-change field.
    marker_sigma (float)
        Blur of the field before its minima are found and flooded, in pixels.
    marker_distance (int)
        Half-width in pixels of the window a marker is the minimum of.

    Returns
    -------
    basins (np.ndarray)
        ``(res, res)`` basin labels ``1..n``.
    """

    resolution = change.shape[0]
    smooth = gaussian_filter(change, marker_sigma, mode='wrap')
    minima = smooth == minimum_filter(smooth, size=2 * marker_distance + 1, mode='wrap')
    tiled_labels, _ = label(np.tile(minima, (3, 3)), structure=np.ones((3, 3)))
    centre = tiled_labels[resolution:2 * resolution, resolution:2 * resolution]
    markers = np.zeros((resolution, resolution), dtype=int)
    for new, old in enumerate(np.unique(centre[centre > 0]), start=1):
        markers[centre == old] = new
    flooded = watershed(np.tile(smooth, (3, 3)), np.tile(markers, (3, 3)))
    return flooded[resolution:2 * resolution, resolution:2 * resolution]


def basin_ridges(basins: np.ndarray, change: np.ndarray) -> tuple[dict, dict]:
    """
    Description
    -----------
    The ridge between every pair of neighbouring basins: for each pair of
    4-neighbouring pixels (periodic, rows then columns) in different basins, the
    larger content change of the two pixels is one ridge sample of that pair. The
    pairs are keyed ``(smaller label, larger label)`` in the order they are first
    met, which fixes how ties are broken during the joins.

    Parameters
    ----------
    basins (np.ndarray)
        ``(res, res)`` basin labels ``1..n``.
    change (np.ndarray)
        ``(res, res)`` content-change field.

    Returns
    -------
    ridge_sum (dict)
        ``(a, b)`` -> summed ridge height.
    ridge_count (dict)
        ``(a, b)`` -> number of ridge samples.
    """

    ridge_sum, ridge_count = {}, {}
    for axis in (0, 1):
        other = np.roll(basins, 1, axis=axis)
        height = np.maximum(change, np.roll(change, 1, axis=axis))
        different = basins != other
        low = np.minimum(basins[different], other[different])
        high = np.maximum(basins[different], other[different])
        for first, second, sample in zip(low, high, height[different], strict=True):
            ridge_sum[(first, second)] = ridge_sum[(first, second)] + sample if (first, second) in ridge_sum else sample
            ridge_count[(first, second)] = ridge_count[(first, second)] + 1 if (first, second) in ridge_count else 1
    return ridge_sum, ridge_count


def find_root(parent: np.ndarray, index: int) -> int:
    """
    Description
    -----------
    Root of a basin in the join forest (with path halving).

    Parameters
    ----------
    parent (np.ndarray)
        Parent of every basin label.
    index (int)
        Basin label.

    Returns
    -------
    root (int)
        Label of the region the basin belongs to.
    """

    while parent[index] != index:
        parent[index] = parent[parent[index]]
        index = parent[index]
    return int(index)


def join_basins(basins: np.ndarray, change: np.ndarray, basin_size: np.ndarray, size_floor: float, n_categories: int) -> np.ndarray | None:
    """
    Description
    -----------
    Joins neighbouring basins one pair at a time until ``n_categories`` regions
    remain, every one of them holding at least ``size_floor`` of the (weighted)
    calls. At each step: if some region holds less than ``size_floor`` of the
    calls, the smallest of them joins the neighbour it shares the lowest mean
    ridge with; otherwise the pair of neighbours with the lowest mean ridge
    height (:func:`basin_ridges`; ties go to the pair met first) joins. The
    larger region (by weighted calls) keeps its label, and the ridges of the
    absorbed one are added to the survivor's.

    Parameters
    ----------
    basins (np.ndarray)
        ``(res, res)`` basin labels ``1..n``.
    change (np.ndarray)
        ``(res, res)`` content-change field.
    basin_size (np.ndarray)
        ``(n + 1,)`` weighted call count of every basin (index 0 unused).
    size_floor (float)
        Smallest share of the calls a final region may hold.
    n_categories (int)
        Number of regions to stop at.

    Returns
    -------
    lookup (np.ndarray | None)
        ``(n + 1,)`` basin label -> region label (the surviving basin label), or
        None when ``n_categories`` regions satisfying the floor are never reached.
    """

    n_basins = int(basins.max())
    ridge_sum, ridge_count = basin_ridges(basins, change)
    total_size = float(basin_size.sum())
    size = basin_size.astype(float).copy()
    parent = np.arange(n_basins + 1)
    alive = set(range(1, n_basins + 1))
    while len(alive) >= n_categories:
        small = [region for region in alive if size[region] < size_floor * total_size]
        if not small and len(alive) == n_categories:
            return np.array([find_root(parent, index) for index in range(n_basins + 1)])
        if small:
            smallest = min(small, key=lambda region: size[region])
            candidates = [pair for pair in ridge_sum if smallest in pair]
        else:
            candidates = list(ridge_sum)
        if not candidates:
            return None
        first, second = min(candidates, key=lambda pair: ridge_sum[pair] / ridge_count[pair])
        if size[first] < size[second]:
            first, second = second, first
        parent[second] = first
        alive.discard(second)
        size[first] += size[second]
        for old in [pair for pair in ridge_sum if second in pair]:
            neighbour = old[0] if old[1] == second else old[1]
            ridge_height, ridge_samples = ridge_sum.pop(old), ridge_count.pop(old)
            if neighbour == first:
                continue
            key = (min(first, neighbour), max(first, neighbour))
            ridge_sum[key] = ridge_sum[key] + ridge_height if key in ridge_sum else ridge_height
            ridge_count[key] = ridge_count[key] + ridge_samples if key in ridge_count else ridge_samples
    return None


def partition_grid(rows: np.ndarray, cols: np.ndarray, ranks: np.ndarray, weights: np.ndarray, cfg: dict) -> tuple[np.ndarray | None, dict]:
    """
    Description
    -----------
    Builds the categories from weighted calls (module docstring, steps 2-5) and
    numbers them ``0..k-1`` by weighted call count, largest first (ties keep the
    order ``pandas.Series.sort_values`` gives them).

    Parameters
    ----------
    rows (np.ndarray)
        ``(n_calls,)`` pixel row of each call.
    cols (np.ndarray)
        ``(n_calls,)`` pixel column of each call.
    ranks (np.ndarray)
        ``(n_calls, p)`` property ranks.
    weights (np.ndarray)
        ``(n_calls,)`` call weights.
    cfg (dict)
        The ``build_qlvm_categories`` settings (``grid_resolution``,
        ``field_sigma``, ``change_span``, ``marker_sigma``, ``marker_distance``,
        ``size_floor``, ``n_categories``).

    Returns
    -------
    grid (np.ndarray | None)
        ``(res, res)`` category labels ``0..k-1``, or None when the joins never
        reach ``n_categories`` regions satisfying the floor.
    layers (dict)
        ``content_change`` and ``basins`` of this build.
    """

    resolution = int(cfg['grid_resolution'])
    fields = content_fields(rows, cols, ranks, weights, float(cfg['field_sigma']), resolution)
    change = content_change(fields, int(cfg['change_span']))
    basins = periodic_basins(change, float(cfg['marker_sigma']), int(cfg['marker_distance']))
    basin_size = np.bincount(basins[rows, cols], weights=weights, minlength=int(basins.max()) + 1)
    layers = {'content_change': change, 'basins': basins}
    lookup = join_basins(basins, change, basin_size, float(cfg['size_floor']), int(cfg['n_categories']))
    if lookup is None:
        return None, layers
    raw = lookup[basins]
    counts = pd.Series(np.bincount(raw[rows, cols], weights=weights, minlength=int(raw.max()) + 1))
    present = set(np.unique(raw).tolist())
    order = [region for region in counts.sort_values(ascending=False).index if region in present]
    relabel = np.zeros(int(raw.max()) + 1, dtype=int)
    for number, region in enumerate(order):
        relabel[region] = number
    return relabel[raw], layers


def apply_category_order(grid: np.ndarray, category_order: list, n_categories: int) -> np.ndarray:
    """
    Description
    -----------
    Renumbers a partition whose categories are numbered ``0..k-1`` by call count,
    largest first (:func:`partition_grid`), into the order ``category_order``
    gives: category ``i`` (``0``-based) of the result is the category of size
    rank ``category_order[i]`` (``1``-based, ``1`` = most calls). ``[1, 4, 2, 3]``
    makes the largest category the first, the smallest the second, the second
    largest the third and the third largest the fourth. An empty order keeps the
    size order. Cells outside ``0..k-1`` (none in a complete partition) keep
    their value.

    Parameters
    ----------
    grid (np.ndarray)
        Integer partition, categories ``0..k-1`` by size rank.
    category_order (list)
        Size rank (``1..k``) of each output category, the first output category
        first; empty for the size order.
    n_categories (int)
        Number of categories ``k``.

    Returns
    -------
    ordered (np.ndarray)
        The partition renumbered, same shape and dtype.

    Raises
    ------
    ValueError
        ``category_order`` is not empty and not a permutation of ``1..k``.
    """

    if not category_order:
        return grid.copy()
    ranks = [int(rank) for rank in category_order]
    if sorted(ranks) != list(range(1, n_categories + 1)):
        error_message = f"category_order must be a permutation of 1..{n_categories}, got {category_order!r}."
        raise ValueError(error_message)
    new_of_rank = np.empty(n_categories, dtype=np.int64)
    for new_category, rank in enumerate(ranks):
        new_of_rank[rank - 1] = new_category
    ordered = grid.copy()
    inside = (grid >= 0) & (grid < n_categories)
    ordered[inside] = new_of_rank[grid[inside]].astype(grid.dtype)
    return ordered


def align_to_reference(reference: np.ndarray, grid: np.ndarray, n_categories: int) -> np.ndarray:
    """
    Description
    -----------
    Relabels a category grid so each of its categories carries the reference
    category it is matched to by a one-to-one assignment maximizing the summed
    pixel overlap (Hungarian assignment).

    Parameters
    ----------
    reference (np.ndarray)
        ``(res, res)`` reference labels ``0..k-1``.
    grid (np.ndarray)
        ``(res, res)`` labels ``0..k-1`` to relabel.
    n_categories (int)
        ``k``.

    Returns
    -------
    aligned (np.ndarray)
        ``(res, res)`` relabelled grid.
    """

    overlap = np.zeros((n_categories, n_categories))
    np.add.at(overlap, (reference.ravel(), grid.ravel()), 1.0)
    row, column = linear_sum_assignment(-overlap)
    mapping = np.zeros(n_categories, dtype=int)
    mapping[column] = row
    return mapping[grid]


def session_bootstrap_weights(sessions: np.ndarray, pool: np.ndarray, seed: int) -> np.ndarray:
    """
    Description
    -----------
    Call weights of one session resample: ``len(pool)`` sessions are drawn from
    ``pool`` with replacement (``numpy.random.default_rng(seed).choice``), and
    every call is weighted by how many times its session was drawn (0 when never).

    Parameters
    ----------
    sessions (np.ndarray)
        ``(n_calls,)`` session id of each call.
    pool (np.ndarray)
        Sessions to draw from (sorted unique session ids).
    seed (int)
        Seed of the draw.

    Returns
    -------
    weights (np.ndarray)
        ``(n_calls,)`` float64 weights.
    """

    drawn, draw_counts = np.unique(np.random.default_rng(seed).choice(pool, size=len(pool), replace=True), return_counts=True)
    count_of = dict(zip(drawn.tolist(), draw_counts.tolist(), strict=True))
    return np.array([float(count_of[session]) if session in count_of else 0.0 for session in sessions.tolist()])


def bootstrap_vote(
    seed: int,
    rows: np.ndarray,
    cols: np.ndarray,
    ranks: np.ndarray,
    sessions: np.ndarray,
    pool: np.ndarray,
    reference: np.ndarray,
    cfg: dict,
) -> np.ndarray | None:
    """
    Description
    -----------
    One session resample: its weights (:func:`session_bootstrap_weights`), its
    categories (:func:`partition_grid`) and their alignment to the reference
    (:func:`align_to_reference`).

    Parameters
    ----------
    seed (int)
        Seed of the resample.
    rows (np.ndarray)
        ``(n_calls,)`` pixel row of each call.
    cols (np.ndarray)
        ``(n_calls,)`` pixel column of each call.
    ranks (np.ndarray)
        ``(n_calls, p)`` property ranks.
    sessions (np.ndarray)
        ``(n_calls,)`` session id of each call.
    pool (np.ndarray)
        Sessions to draw from.
    reference (np.ndarray)
        ``(res, res)`` reference labels ``0..k-1``.
    cfg (dict)
        The ``build_qlvm_categories`` settings.

    Returns
    -------
    aligned (np.ndarray | None)
        ``(res, res)`` aligned labels, or None when this resample does not reach
        ``n_categories`` categories under the floor (it then casts no vote).
    """

    grid, _layers = partition_grid(rows, cols, ranks, session_bootstrap_weights(sessions, pool, seed), cfg)
    return None if grid is None else align_to_reference(reference, grid, int(cfg['n_categories']))


def consensus_vote(aligned: list, n_categories: int, resolution: int) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Description
    -----------
    Majority vote of aligned category grids: every grid gives each pixel one vote
    for its category; the winner is the category with most votes (the lowest
    label on a tie) and the agreement its share of the votes.

    Parameters
    ----------
    aligned (list)
        Aligned ``(res, res)`` grids; None entries cast no vote.
    n_categories (int)
        ``k``.
    resolution (int)
        Grid resolution in pixels per side.

    Returns
    -------
    grid (np.ndarray)
        ``(res, res)`` consensus labels ``0..k-1``.
    agreement (np.ndarray)
        ``(res, res)`` winning share of the votes.
    n_votes (int)
        Number of grids that voted.
    """

    votes = np.zeros((n_categories, resolution, resolution))
    used = [grid for grid in aligned if grid is not None]
    for grid in used:
        for category in range(n_categories):
            votes[category] += grid == category
    return votes.argmax(axis=0), votes.max(axis=0) / max(len(used), 1), len(used)


def category_label_position(grid: np.ndarray, category: int) -> tuple[float, float]:
    """
    Description
    -----------
    Where to write a category's name on the map: the centre of its pixel
    farthest from its boundary (Euclidean distance transform on a 3 x 3 tiling,
    so the torus seams are not boundaries).

    Parameters
    ----------
    grid (np.ndarray)
        ``(res, res)`` category labels.
    category (int)
        The category's label in ``grid``.

    Returns
    -------
    position (tuple[float, float])
        ``(x, y)`` in ``[0, 1)``.
    """

    resolution = grid.shape[0]
    depth = distance_transform_edt(np.tile(grid == category, (3, 3)))[resolution:2 * resolution, resolution:2 * resolution]
    row, column = np.unravel_index(np.argmax(depth), depth.shape)
    return (column + 0.5) / resolution, (row + 0.5) / resolution


def read_table(path: str | pathlib.Path, columns: list[str]) -> pd.DataFrame:
    """
    Description
    -----------
    Reads the named columns of a per-call table from a ``.npz`` (one array per
    column; e.g. a ``build-qlvm-training-set`` split, read lazily so its
    spectrograms are never loaded), ``.parquet`` or ``.csv`` file.

    Parameters
    ----------
    path (str | pathlib.Path)
        The table.
    columns (list[str])
        Columns to read.

    Returns
    -------
    table (pd.DataFrame)
        The columns, in file row order.

    Raises
    ------
    ValueError
        A column is missing or the file type is not one of the three.
    """

    path = pathlib.Path(path)
    if path.suffix == '.npz':
        with np.load(path, allow_pickle=False) as handle:
            missing = [column for column in columns if column not in handle.files]
            if missing:
                error_message = f"{path} has no {missing} array(s)."
                raise ValueError(error_message)
            return pd.DataFrame({column: handle[column] for column in columns})
    if path.suffix == '.parquet':
        table = pd.read_parquet(path)
    elif path.suffix == '.csv':
        table = pd.read_csv(path, dtype={'spec_id': str, 'session_id': str})
    else:
        error_message = f"{path}: a per-call table must be a .npz, .parquet or .csv file."
        raise ValueError(error_message)
    missing = [column for column in columns if column not in table.columns]
    if missing:
        error_message = f"{path} has no {missing} column(s)."
        raise ValueError(error_message)
    return table[columns].reset_index(drop=True)


def load_category_bundle(category_directory: str) -> dict:
    """
    Description
    -----------
    Loads a category directory written by :class:`QLVMCategoryBuilder`, through
    the one bundle reader every figure uses too,
    :func:`os_utils.read_qlvm_category_bundle` (which checks that the grids are
    square, share one shape and hold exactly the nomenclature's labels ``1..k``).

    Parameters
    ----------
    category_directory (str)
        The directory.

    Returns
    -------
    bundle (dict)
        :func:`os_utils.read_qlvm_category_bundle`'s dictionary (``label_grid``
        (``(res, res)`` int, ``1..k``), ``agreement`` (``(res, res)`` float),
        ``nomenclature`` (dict, the JSON), ``build_config``, ``density``,
        ``centers``, ...), plus ``uncertain_agreement`` (float, the
        nomenclature's threshold).
    """

    bundle = read_qlvm_category_bundle(category_directory)
    bundle['uncertain_agreement'] = float(bundle['nomenclature']['uncertain_agreement'])
    return bundle


def assign_categories(coordinates: np.ndarray, bundle: dict) -> dict[str, np.ndarray]:
    """
    Description
    -----------
    Labels torus coordinates with a category bundle: each finite coordinate takes
    the category and agreement of its pixel (:func:`qlvm_latents.label_grid_lookup`,
    the QLVM pixel rule ``grid[floor(y * res) mod res, floor(x * res) mod res]``),
    and is uncertain when that agreement is below the bundle's
    ``uncertain_agreement``. A coordinate with a NaN gets category 0, agreement NaN
    and uncertain False, and ``placed`` False.

    Parameters
    ----------
    coordinates (np.ndarray)
        ``(n, 2)`` torus coordinates ``(x, y)``.
    bundle (dict)
        :func:`load_category_bundle` output.

    Returns
    -------
    labels (dict[str, np.ndarray])
        ``placed`` (bool), ``category`` (int64), ``agreement`` (float64) and
        ``uncertain`` (bool), each ``(n,)``.
    """

    coordinates = np.asarray(coordinates, dtype=np.float64).reshape(-1, 2)
    placed = np.isfinite(coordinates).all(axis=1)
    category = np.zeros(coordinates.shape[0], dtype=np.int64)
    agreement = np.full(coordinates.shape[0], np.nan)
    category[placed] = label_grid_lookup(coordinates[placed], bundle['label_grid'])
    agreement[placed] = label_grid_lookup(coordinates[placed], bundle['agreement'])
    uncertain = placed & (agreement < bundle['uncertain_agreement'])
    return {'placed': placed, 'category': category, 'agreement': agreement, 'uncertain': uncertain}


class QLVMCategoryBuilder:
    """
    Description
    -----------
    Builds the content-ridge categories of a QLVM torus from a corpus of embedded
    calls and writes them as a category directory (see the module docstring).
    """

    def __init__(
        self,
        positions_file: str | None = None,
        properties_file: str | None = None,
        output_directory: str | None = None,
        input_parameter_dict: dict | None = None,
        message_output: Callable | None = None,
    ) -> None:
        """
        Description
        -----------
        Initializes the QLVMCategoryBuilder.

        Parameters
        ----------
        positions_file (str)
            Per-call torus positions: ``.npz`` / ``.parquet`` / ``.csv`` with
            ``spec_id``, ``x`` and ``y`` in ``[0, 1)``.
        properties_file (str)
            Per-call properties: ``.npz`` / ``.parquet`` / ``.csv`` with
            ``spec_id``, ``session_id`` and every column of the ``properties``
            setting (e.g. a ``build-qlvm-training-set`` split, whose ``mask_count``,
            ``durations``, ``freq_bandwidth_hz``, ``spectral_entropy``,
            ``mean_freq_hz`` and ``loudness_db`` arrays are the default properties).
            Its rows, in its order, are the corpus; each must have a position.
        output_directory (str)
            Directory the category files are written to (created if missing).
        input_parameter_dict (dict)
            Processing settings; the ``build_qlvm_categories`` block supplies
            every parameter.
        message_output (Callable)
            Logging callback; defaults to ``print``.

        Returns
        -------
        None
        """

        self.positions_file = positions_file
        self.properties_file = properties_file
        self.output_directory = output_directory
        self.input_parameter_dict = input_parameter_dict if input_parameter_dict is not None else {}
        self.message_output = message_output if message_output is not None else print
        self.app_context_bool = is_gui_context()

    def _validate_settings(self, cfg: dict) -> None:
        """
        Description
        -----------
        Checks the ``build_qlvm_categories`` block before any data is read,
        raising one error that lists every problem.

        Parameters
        ----------
        cfg (dict)
            The ``build_qlvm_categories`` settings block.

        Returns
        -------
        None
        """

        problems = []
        if not cfg['properties'] or len(set(cfg['properties'])) != len(cfg['properties']):
            problems.append(f"properties must be a non-empty list of distinct columns, got {cfg['properties']!r}")
        if int(cfg['grid_resolution']) < 8:
            problems.append(f"grid_resolution must be at least 8 pixels, got {cfg['grid_resolution']!r}")
        if float(cfg['field_sigma']) <= 0 or float(cfg['marker_sigma']) <= 0:
            problems.append("field_sigma and marker_sigma must be positive")
        if int(cfg['change_span']) < 1 or int(cfg['marker_distance']) < 1:
            problems.append("change_span and marker_distance must be at least 1 pixel")
        if not 0.0 <= float(cfg['size_floor']) < 1.0 / int(cfg['n_categories']):
            problems.append(f"size_floor must lie in [0, 1 / n_categories), got {cfg['size_floor']!r}")
        if int(cfg['n_categories']) < 2:
            problems.append(f"n_categories must be at least 2, got {cfg['n_categories']!r}")
        if int(cfg['n_bootstraps']) < 1:
            problems.append(f"n_bootstraps must be at least 1, got {cfg['n_bootstraps']!r}")
        if not 0.0 < float(cfg['uncertain_agreement']) <= 1.0:
            problems.append(f"uncertain_agreement must lie in (0, 1], got {cfg['uncertain_agreement']!r}")
        if cfg['category_order'] and sorted(int(rank) for rank in cfg['category_order']) != list(range(1, int(cfg['n_categories']) + 1)):
            problems.append(
                f"category_order must be empty or a permutation of 1..n_categories ({cfg['n_categories']}), "
                f"got {cfg['category_order']!r}"
            )
        if cfg['category_descriptions'] and len(cfg['category_descriptions']) != int(cfg['n_categories']):
            problems.append(
                f"category_descriptions must be empty or hold one description per category "
                f"({cfg['n_categories']}), got {len(cfg['category_descriptions'])}"
            )
        if problems:
            error_message = "build_qlvm_categories settings are invalid:\n  " + "\n  ".join(problems)
            raise ValueError(error_message)

    def _load_corpus(self, cfg: dict) -> tuple[pd.DataFrame, np.ndarray]:
        """
        Description
        -----------
        Reads the properties table (the corpus, in its row order) and joins every
        call's position from the positions table on ``spec_id``.

        Parameters
        ----------
        cfg (dict)
            The ``build_qlvm_categories`` settings block.

        Returns
        -------
        corpus (pd.DataFrame)
            ``spec_id``, ``session_id``, ``x``, ``y`` and the properties, one row
            per call.
        values (np.ndarray)
            ``(n_calls, n_properties)`` float64 property values.
        """

        properties = list(cfg['properties'])
        corpus = read_table(self.properties_file, [*PROPERTY_ID_COLUMNS, *properties])
        positions = read_table(self.positions_file, list(POSITION_COLUMNS))
        corpus['spec_id'] = corpus['spec_id'].astype(str)
        corpus['session_id'] = corpus['session_id'].astype(str)
        positions['spec_id'] = positions['spec_id'].astype(str)
        if corpus['spec_id'].duplicated().any() or positions['spec_id'].duplicated().any():
            error_message = "build_qlvm_categories: spec_id must be unique in both the properties and the positions table."
            raise ValueError(error_message)
        corpus = corpus.merge(positions, on='spec_id', how='left', validate='one_to_one')
        unplaced = int(corpus[['x', 'y']].isna().any(axis=1).sum())
        if unplaced:
            error_message = f"build_qlvm_categories: {unplaced} calls of {self.properties_file} have no position in {self.positions_file}."
            raise ValueError(error_message)
        coordinates = corpus[['x', 'y']].to_numpy(dtype=np.float64)
        if (coordinates < 0).any() or (coordinates >= 1).any():
            error_message = "build_qlvm_categories: every position must lie in [0, 1)."
            raise ValueError(error_message)
        n_extra = positions.shape[0] - corpus.shape[0]
        if n_extra:
            self.message_output(f"{n_extra} positions belong to no call of the properties table and are ignored.")
        return corpus, corpus[properties].to_numpy(dtype=np.float64)

    def build(self) -> None:
        """
        Description
        -----------
        Runs the build (module docstring, steps 1-6) and writes the category
        directory: ``category_grids.npz``, ``category_nomenclature.json``,
        ``category_call_labels.csv`` and ``build_config.json``.

        Parameters
        ----------

        Returns
        -------
        None
        """

        self.message_output(
            f"QLVM category build started at: {datetime.now().hour:02d}:{datetime.now().minute:02d}:{datetime.now().second:02d}."
        )
        smart_wait(app_context_bool=self.app_context_bool, seconds=1)

        cfg = self.input_parameter_dict['build_qlvm_categories']
        self._validate_settings(cfg)
        resolution = int(cfg['grid_resolution'])
        n_categories = int(cfg['n_categories'])
        corpus, values = self._load_corpus(cfg)
        ranks = property_ranks(values)
        rows, cols = call_pixels(corpus[['x', 'y']].to_numpy(dtype=np.float64), resolution)
        sessions = corpus['session_id'].to_numpy().astype(str)
        pool = np.unique(sessions)
        self.message_output(f"{corpus.shape[0]:,} calls from {pool.size} sessions; properties {list(cfg['properties'])}.")

        reference, layers = partition_grid(rows, cols, ranks, np.ones(corpus.shape[0]), cfg)
        if reference is None:
            error_message = (
                f"build_qlvm_categories: the full corpus does not reach {n_categories} categories that each hold "
                f"at least {cfg['size_floor']} of the calls."
            )
            raise ValueError(error_message)
        aligned = Parallel(n_jobs=int(cfg['n_jobs']))(
            delayed(bootstrap_vote)(int(cfg['bootstrap_seed']) + index, rows, cols, ranks, sessions, pool, reference, cfg)
            for index in range(int(cfg['n_bootstraps']))
        )
        grid, agreement, n_votes = consensus_vote(aligned, n_categories, resolution)
        if n_votes == 0:
            error_message = f"build_qlvm_categories: none of the {cfg['n_bootstraps']} session resamples reached {n_categories} categories."
            raise ValueError(error_message)
        grid = apply_category_order(grid, cfg['category_order'], n_categories)
        reference = apply_category_order(reference, cfg['category_order'], n_categories)
        call_category = grid[rows, cols] + 1
        call_agreement = agreement[rows, cols]
        uncertain = call_agreement < float(cfg['uncertain_agreement'])
        self.message_output(
            f"{n_votes} of {cfg['n_bootstraps']} session resamples voted; {int((grid[rows, cols] != reference[rows, cols]).sum()):,} calls "
            f"changed category from the single full-data partition; {int(uncertain.sum()):,} calls "
            f"({uncertain.mean():.1%}) are uncertain (agreement < {cfg['uncertain_agreement']})."
        )

        descriptions = list(cfg['category_descriptions']) if cfg['category_descriptions'] else [''] * n_categories
        categories = []
        for category in range(1, n_categories + 1):
            label_x, label_y = category_label_position(grid + 1, category)
            categories.append({
                'grid_label': category,
                'name': f'R-{category}',
                'description': descriptions[category - 1],
                'n_calls': int((call_category == category).sum()),
                'share': round(float((call_category == category).mean()), 4),
                'uncertain_share': round(float(uncertain[call_category == category].mean()), 4) if (call_category == category).any() else 0.0,
                'label_x': round(label_x, 4),
                'label_y': round(label_y, 4),
            })
            self.message_output(
                f"  R-{category} ({descriptions[category - 1] or 'no description'}): {categories[-1]['n_calls']:,} calls "
                f"({categories[-1]['share']:.1%}), {categories[-1]['uncertain_share']:.1%} uncertain."
            )

        output_dir = pathlib.Path(self.output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        density = gaussian_filter(np.bincount(rows * resolution + cols, minlength=resolution * resolution).reshape(resolution, resolution).astype(float),
                                  float(cfg['field_sigma']), mode='wrap')
        with atomic_output_path(output_dir / CATEGORY_GRIDS_NAME) as tmp_path, tmp_path.open('wb') as grids_file:
            np.savez_compressed(
                grids_file,
                label_grid=(grid + 1).astype(np.int16),
                agreement=agreement,
                reference_grid=(reference + 1).astype(np.int16),
                content_change=layers['content_change'],
                basins=layers['basins'].astype(np.int32),
                density=density,
            )
        nomenclature = {
            'grid_resolution': resolution,
            'n_categories': n_categories,
            'uncertain_agreement': float(cfg['uncertain_agreement']),
            'n_votes': int(n_votes),
            'n_calls': int(corpus.shape[0]),
            'categories': categories,
        }
        with atomic_output_path(output_dir / CATEGORY_NOMENCLATURE_NAME) as tmp_path:
            tmp_path.write_text(json.dumps(nomenclature, indent=1) + '\n')
        names = np.array([category['name'] for category in categories])
        call_labels = corpus[['spec_id', 'session_id', 'x', 'y']].assign(
            category=call_category,
            name=names[call_category - 1],
            description=np.array(descriptions)[call_category - 1],
            agreement=call_agreement,
            uncertain=uncertain,
        )
        with atomic_output_path(output_dir / CATEGORY_CALL_LABELS_NAME) as tmp_path:
            call_labels.to_csv(tmp_path, index=False)
        build_config = {
            'build_qlvm_categories': cfg,
            'positions_file': str(self.positions_file),
            'positions_file_sha256': file_sha256(self.positions_file),
            'properties_file': str(self.properties_file),
            'n_calls': int(corpus.shape[0]),
            'n_sessions': int(pool.size),
            'n_basins': int(layers['basins'].max()),
            'n_votes': int(n_votes),
            'built': datetime.now().isoformat(timespec='seconds'),
        }
        with atomic_output_path(output_dir / CATEGORY_BUILD_CONFIG_NAME) as tmp_path:
            tmp_path.write_text(json.dumps(build_config, indent=2) + '\n')
        self.message_output(f"Wrote the {n_categories} categories to {output_dir}.")
        self.message_output(
            f"QLVM category build ended at: {datetime.now().hour:02d}:{datetime.now().minute:02d}:{datetime.now().second:02d}."
        )


class QLVMCategoryAssigner:
    """
    Description
    -----------
    Labels the calls of one session with a category directory, from their torus
    coordinates in the session's ``*_usv_summary.csv`` (see the module docstring).
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
        Initializes the QLVMCategoryAssigner.

        Parameters
        ----------
        root_directory (str)
            Session root directory (contains the ``audio`` tree).
        input_parameter_dict (dict)
            Processing settings; the ``assign_qlvm_categories`` block supplies the
            category directory and the coordinate prefix.
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

    def assign_and_merge(self) -> None:
        """
        Description
        -----------
        Reads the category directory (``category_directory``) and the session's
        USV summary, labels every call that has both ``<P>1`` and ``<P>2``
        (``P`` = ``coordinate_prefix``; :func:`assign_categories`) and writes
        ``<P>_category`` (``1..k``) into the summary (null for calls without
        coordinates), replacing an earlier column of that name. The per-call
        agreement and uncertain flag are computed (the log counts the uncertain
        calls) but not written; an earlier ``<P>_category_agreement`` /
        ``<P>_category_uncertain`` column is removed. The summary is rewritten
        atomically, in canonical column order
        (:func:`os_utils.order_usv_summary_columns`).

        Parameters
        ----------

        Returns
        -------
        None
        """

        self.message_output(
            f"QLVM category assignment started at: {datetime.now().hour:02d}:{datetime.now().minute:02d}:{datetime.now().second:02d}."
        )
        smart_wait(app_context_bool=self.app_context_bool, seconds=1)

        # An empty category directory / coordinate prefix is the production bundle
        # (os_utils.QLVM_CATEGORY_BUNDLE_DIRECTORY) on the map it is defined on (qlvm).
        derive_spectrogram_model_paths(self.input_parameter_dict)
        cfg = self.input_parameter_dict['assign_qlvm_categories']
        prefix = cfg['coordinate_prefix']
        if not cfg['category_directory'] or not isinstance(prefix, str) or not prefix.isidentifier():
            error_message = (
                f"assign_qlvm_categories needs a category_directory (a build-qlvm-categories output) and a "
                f"coordinate_prefix P whose P1 / P2 columns hold the torus coordinates of the map the categories were "
                f"built on; got {cfg['category_directory']!r} and {prefix!r}."
            )
            raise ValueError(error_message)
        bundle = load_category_bundle(cfg['category_directory'])
        usv_summary_loc = first_match_or_raise(
            root=pathlib.Path(self.root_directory) / "audio",
            pattern="*_usv_summary.csv",
            recursive=True,
            label="USV summary CSV",
        )
        usv_df = pls.read_csv(source=str(usv_summary_loc), schema_overrides={"usv_id": pls.String})
        missing = [column for column in (f"{prefix}1", f"{prefix}2") if column not in usv_df.columns]
        if missing:
            error_message = f"{usv_summary_loc.name} has no {missing} column(s); run infer-qlvm-latents with prefix {prefix!r} first."
            raise ValueError(error_message)
        coordinates = usv_df.select([f"{prefix}1", f"{prefix}2"]).cast(pls.Float64).fill_null(np.nan).to_numpy()
        labels = assign_categories(coordinates, bundle)
        placed = labels['placed']
        category_column = f"{prefix}_category"
        new_columns = pls.DataFrame({
            "_placed": placed,
            category_column: labels['category'],
        })
        # Calls without coordinates get a null category.
        new_columns = new_columns.select(
            pls.when(pls.col("_placed")).then(pls.col(category_column)).otherwise(None).alias(category_column)
        )
        # The category column is replaced; the confidence columns an earlier version
        # wrote are removed (the agreement and uncertain flag stay outside the summary).
        replaced = [category_column, *(f"{prefix}{suffix}" for suffix in CATEGORY_CONFIDENCE_SUFFIXES)]
        merged = usv_df.drop([column for column in replaced if column in usv_df.columns]).hstack(new_columns)
        merged = order_usv_summary_columns(merged)
        with atomic_output_path(usv_summary_loc) as tmp_summary_path:
            merged.write_csv(file=str(tmp_summary_path))
        counts = np.bincount(labels['category'][placed], minlength=len(bundle['nomenclature']['categories']) + 1)[1:]
        self.message_output(
            f"{int(placed.sum())} of {usv_df.height} USVs labelled with {cfg['category_directory']} "
            f"({', '.join(f'R-{index + 1}: {int(count)}' for index, count in enumerate(counts))}; "
            f"{int(labels['uncertain'].sum())} uncertain, not written) into {category_column} of {usv_summary_loc.name}."
        )
        self.message_output(
            f"QLVM category assignment ended at: {datetime.now().hour:02d}:{datetime.now().minute:02d}:{datetime.now().second:02d}."
        )


@click.command(name="build-qlvm-categories")
@click.option('--positions-file', type=click.Path(exists=True, file_okay=True, dir_okay=False), required=True, help='Per-call torus positions (.npz / .parquet / .csv with spec_id, x, y in [0, 1)), e.g. the embedding of the training set the map was trained on.')
@click.option('--properties-file', type=click.Path(exists=True, file_okay=True, dir_okay=False), required=True, help='Per-call properties (.npz / .parquet / .csv with spec_id, session_id and every property column), e.g. a build-qlvm-training-set split; its rows are the corpus.')
@click.option('--output-directory', type=click.Path(file_okay=False, dir_okay=True), required=True, help='Directory to write category_grids.npz, category_nomenclature.json, category_call_labels.csv and build_config.json into.')
@click.option('--properties', 'properties', type=str, default=None, required=False, help='Comma-separated property columns the categories are built on, e.g. mask_count,durations,freq_bandwidth_hz,spectral_entropy,mean_freq_hz,loudness_db.')
@click.option('--grid-resolution', 'grid_resolution', type=int, default=None, required=False, help='Pixels per side of the periodic grid.')
@click.option('--field-sigma', 'field_sigma', type=float, default=None, required=False, help='Gaussian smoothing of the property fields, in pixels.')
@click.option('--change-span', 'change_span', type=int, default=None, required=False, help='Offset of the central differences of the content-change field, in pixels.')
@click.option('--marker-sigma', 'marker_sigma', type=float, default=None, required=False, help='Blur of the content-change field before the watershed, in pixels.')
@click.option('--marker-distance', 'marker_distance', type=int, default=None, required=False, help='Half-width of the window a watershed marker is the minimum of, in pixels.')
@click.option('--size-floor', 'size_floor', type=float, default=None, required=False, help='Smallest share of the calls a category may hold.')
@click.option('--n-categories', 'n_categories', type=int, default=None, required=False, help='Number of categories.')
@click.option('--n-bootstraps', 'n_bootstraps', type=int, default=None, required=False, help='Session resamples of the consensus vote.')
@click.option('--bootstrap-seed', 'bootstrap_seed', type=int, default=None, required=False, help='Seed of the first session resample (resample i uses seed + i).')
@click.option('--uncertain-agreement', 'uncertain_agreement', type=float, default=None, required=False, help='A call whose pixel agreement is below this is flagged uncertain.')
@click.option('--category-order', 'category_order', type=str, default=None, required=False, help='Comma-separated size rank (1 = most calls) of each category, R-1 first (e.g. "1,4,2,3"), or an empty string for size order.')
@click.option('--category-descriptions', 'category_descriptions', type=str, default=None, required=False, help='Comma-separated short description of each category, R-1 first (e.g. "simple,biphones,intermediate,complex"), or an empty string for none.')
@click.option('--n-jobs', 'n_jobs', type=int, default=None, required=False, help='Parallel workers of the session resamples.')
@click.pass_context
def build_qlvm_categories_cli(ctx, positions_file, properties_file, output_directory, **kwargs) -> None:
    """
    Description
    -----------
    A command-line tool to build the content-ridge categories of a QLVM torus
    from a corpus of embedded calls and write them as a category directory.

    Parameters
    ----------

    Returns
    -------
    None
    """

    provided_params = [key for key in kwargs if ctx.get_parameter_source(key) == ParameterSource.COMMANDLINE]
    # Comma-separated lists become the settings' JSON lists.
    for key in ('properties', 'category_descriptions'):
        if key in provided_params:
            ctx.params[key] = [entry.strip() for entry in ctx.params[key].split(',') if entry.strip()]
    if 'category_order' in provided_params:
        ctx.params['category_order'] = [int(entry) for entry in ctx.params['category_order'].split(',') if entry.strip()]

    processing_settings_dict = modify_settings_json_for_cli(
        ctx=ctx,
        provided_params=provided_params,
        settings_dict='processing_settings',
        block='build_qlvm_categories',
    )

    QLVMCategoryBuilder(
        positions_file=positions_file,
        properties_file=properties_file,
        output_directory=output_directory,
        input_parameter_dict=processing_settings_dict,
        message_output=print,
    ).build()


@click.command(name="assign-qlvm-categories")
@click.option('--root-directory', type=click.Path(exists=True, file_okay=False, dir_okay=True), required=True, help='Session root directory path.')
@click.option('--category-directory', 'category_directory', type=click.Path(exists=True, file_okay=False, dir_okay=True), default=None, required=False, help='A build-qlvm-categories output directory; when neither this nor the category_directory setting names one, the production bundle (os_utils.QLVM_CATEGORY_BUNDLE_DIRECTORY, the categories every QLVM figure draws).')
@click.option('--coordinate-prefix', 'coordinate_prefix', type=str, default=None, required=False, help='Prefix P of the summary columns P1 / P2 holding the torus coordinates of the map the categories were built on (the infer-qlvm-latents --model-cell prefix; qlvm, the regular map the production bundle is defined on, when neither this nor the setting names one); P_category is written (earlier P_category_agreement / P_category_uncertain columns are removed).')
@click.pass_context
def assign_qlvm_categories_cli(ctx, root_directory, **kwargs) -> None:
    """
    Description
    -----------
    A command-line tool to label a session's USVs with the content-ridge
    categories of a category directory, from their torus coordinates in its USV
    summary CSV.

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
        block='assign_qlvm_categories',
    )

    QLVMCategoryAssigner(
        root_directory=root_directory,
        input_parameter_dict=processing_settings_dict,
        message_output=print,
    ).assign_and_merge()
