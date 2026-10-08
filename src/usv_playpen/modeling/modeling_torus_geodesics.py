"""
@author: bartulem
Distance geometries on the QLVM torus latent space.

This module provides three interchangeable ways to measure distance between
points on the (period-`period`) flat 2-torus that the QLVM embeds vocalizations
into. It is an ANALYSIS-ONLY toolkit -- it does not touch the von Mises
selection objective in ``manifold_metric`` -- built so the three geometries can
be computed side by side and compared (e.g. for bout acoustic self-similarity,
repertoire structure, or centroid-to-centroid paths):

(1) **Flat-torus wrapped distance** -- the intrinsic metric of the periodic
    parametrization: the wrap-aware Euclidean norm (``manifold_metric.signed_diff``).
    Geometry-only; ignores where the data density sits and how the decoder warps
    the space. Cheap and exact per point.

(2) **Density-ratio graph geodesic** -- in the spirit of Martinez & Williams
    (QLVM paper, Appendix C): a shortest path on a wrap-aware k-nearest-neighbour
    graph whose edges are the flat-torus lengths reweighted by the aggregate
    posterior density, so paths are cheap through high-density regions and
    expensive across low-density valleys (the class boundaries where the decoded
    spectrogram changes abruptly). Implemented as the symmetric inverse-density
    (Fermat-style) arc length ``integral (1/density)^alpha dl``, which realises
    the paper's "route through dense regions" intent as a proper metric. Needs
    only the embedded points (no decoder).

(3) **Decoder-Jacobian pullback geodesic** -- the Arvanitidis et al. (2018)
    pullback the paper explicitly defers: the frozen QLVM ConvTranspose decoder
    ``g: z -> spectrogram`` induces a Riemannian metric ``G(z) = J(z)^T J(z)``
    (``J = dg/dz``) that measures distance by how much the decoded spectrogram
    changes; the geodesic is the shortest path under ``G``. Because the decoder
    is a differentiable JAX function (``qlvm_model.decoder_forward`` composed with
    ``torus_basis_forward``), ``J`` is available exactly via forward-mode autodiff
    (``jax.jacfwd``; 2 latent inputs << spectrogram outputs). The pullback core is
    written decoder-agnostically (it takes any ``decode_fn``), so it is unit-
    testable against an analytic linear map; ``make_qlvm_decode_fn`` wires the
    real decoder. The decoder is always the one of the map the coordinates come
    from (``resolve_geodesic_decoder_source``: the regular, a conditional or the
    squeak cell); a conditional decoder is evaluated at one fixed conditioning
    value, the ``pullback_condition_quantile`` of its training corpus's
    conditioning distribution, so ``G`` stays a function of ``z`` alone.

Geometries (2) and (3) share one k-NN-graph + Dijkstra core and differ only in
the edge-weight source (density ratio vs local G-length). All shortest paths are
computed on a manageable node set -- a regular ``torus_grid`` or a set of cluster
centroids -- because all-pairs geodesics over every USV are infeasible; per-USV
queries snap to their nearest node.
"""

from __future__ import annotations

from collections import namedtuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.stats import gaussian_kde

from ..os_utils import QLVM_DECODER_MAPS, qlvm_map_cell_directory
from ..processing.qlvm_latents import load_model_cell, model_decode_condition
from ..processing.qlvm_model import decoder_forward, torus_basis_forward
from .manifold_metric import _geodesic_distance_matrix, signed_diff
from .modeling_utils import manifold_tag_segment

_DENSITY_FLOOR = 1e-12

# Precomputed geometry for turning a (y_pred, y_true) pair into a per-event
# geodesic distance by snap-to-grid + lookup. `density_matrix` / `pullback_matrix`
# are the all-pairs grid geodesic distance matrices (either may be None when that
# geometry was not requested / no decoder was available).
TorusGeodesicContext = namedtuple(
    "TorusGeodesicContext",
    ["grid", "n_per_dim", "period", "density", "density_matrix", "pullback_matrix"],
)


def flat_torus_distance_matrix(nodes: np.ndarray, *, period: float = 1.0) -> np.ndarray:
    """
    Full pairwise flat-torus (wrap-aware Euclidean) distance matrix.

    This is geometry (1): the intrinsic metric of the periodic latent square,
    delegated to :func:`manifold_metric._geodesic_distance_matrix` so the
    wrapping convention matches the rest of the pipeline exactly.

    Parameters
    ----------
    nodes : np.ndarray
        ``(n, 2)`` torus coordinates in ``[0, period)``.
    period : float, default 1.0
        Per-axis wrap period (1.0 for the native QLVM torus).

    Returns
    -------
    np.ndarray
        ``(n, n)`` symmetric flat-torus distance matrix.
    """

    return _geodesic_distance_matrix(np.asarray(nodes, dtype=np.float64),
                                     metric='torus', period=period)


def torus_grid(n_per_dim: int, *, period: float = 1.0) -> np.ndarray:
    """
    Regular grid of node coordinates covering the flat 2-torus.

    Nodes are placed at cell centres of an ``n_per_dim x n_per_dim`` partition of
    ``[0, period)^2``, so no node sits exactly on the wrap seam and coverage is
    uniform. This is the default node set for the graph geodesics (2) and (3);
    per-USV queries are answered by snapping to the nearest grid node.

    Parameters
    ----------
    n_per_dim : int
        Number of grid cells per axis; the grid has ``n_per_dim ** 2`` nodes.
    period : float, default 1.0
        Per-axis wrap period.

    Returns
    -------
    np.ndarray
        ``(n_per_dim ** 2, 2)`` grid coordinates in ``[0, period)``.
    """

    axis = (np.arange(int(n_per_dim)) + 0.5) * (float(period) / int(n_per_dim))
    xx, yy = np.meshgrid(axis, axis, indexing='ij')
    return np.stack([xx.ravel(), yy.ravel()], axis=1)


def torus_kde_density(samples: np.ndarray, queries: np.ndarray, *,
                      period: float = 1.0, bw_method=None) -> np.ndarray:
    """
    Wrap-aware Gaussian-KDE density at ``queries``, fit on ``samples``.

    Fits a :class:`scipy.stats.gaussian_kde` on the (non-wrapped) sample cloud
    and evaluates it as a sum over the 3x3 lattice shifts of the torus, so the
    estimate is periodic and a query near the seam correctly aggregates mass from
    the opposite edge -- the same periodic-KDE construction used in
    ``acoustic_manifold_geometry``. This is the aggregate-posterior density that
    weights the density-ratio geodesic (2).

    Parameters
    ----------
    samples : np.ndarray
        ``(m, 2)`` embedded points the density is estimated from.
    queries : np.ndarray
        ``(n, 2)`` points to evaluate the density at (e.g. grid nodes).
    period : float, default 1.0
        Per-axis wrap period.
    bw_method : str | float | callable | None, default None
        Forwarded to :class:`scipy.stats.gaussian_kde` (``None`` -> Scott's rule).

    Returns
    -------
    np.ndarray
        ``(n,)`` non-negative density at each query point.
    """

    samples = np.asarray(samples, dtype=np.float64)
    queries = np.asarray(queries, dtype=np.float64)
    kde = gaussian_kde(samples.T, bw_method=bw_method)
    shifts = [np.array([dx, dy], dtype=np.float64) * float(period)
              for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
    dens = np.zeros(queries.shape[0], dtype=np.float64)
    for s in shifts:
        dens += kde(queries.T - s[:, None])
    return dens


def _torus_knn_edges(nodes: np.ndarray, k: int, period: float):
    """
    Directed k-nearest-neighbour edge list over ``nodes`` under the flat-torus
    metric, returned as ``(rows, cols, flat_len)`` for the graph geodesics.

    Each node is connected to its ``k`` nearest neighbours (self excluded); the
    edge length is the flat-torus distance. The caller symmetrises the graph, so
    returning the directed k-NN edges is sufficient and keeps every node with at
    least ``k`` incident edges (guaranteeing connectivity for a dense grid).

    Parameters
    ----------
    nodes : np.ndarray
        ``(n, 2)`` node coordinates.
    k : int
        Number of neighbours per node.
    period : float
        Per-axis wrap period.

    Returns
    -------
    tuple of np.ndarray
        ``(rows, cols, flat_len)`` each of length ``n * k``.
    """

    dmat = flat_torus_distance_matrix(nodes, period=period)
    n = nodes.shape[0]
    k = int(min(k, n - 1))
    nn = np.argsort(dmat, axis=1)[:, 1:k + 1]
    rows = np.repeat(np.arange(n), k)
    cols = nn.ravel()
    flat_len = dmat[rows, cols]
    return rows, cols, flat_len


def _dijkstra_from_edges(n: int, rows: np.ndarray, cols: np.ndarray,
                         weights: np.ndarray, sources=None) -> np.ndarray:
    """
    Symmetric-graph Dijkstra shortest-path distances from an edge list.

    Builds an undirected sparse graph from the (possibly directed) edge list by
    taking the elementwise maximum with its transpose (so an edge present in one
    direction is mirrored), then runs :func:`scipy.sparse.csgraph.dijkstra`.

    Parameters
    ----------
    n : int
        Number of nodes.
    rows, cols : np.ndarray
        Edge endpoints.
    weights : np.ndarray
        Non-negative edge weights aligned with ``rows`` / ``cols``.
    sources : array-like | None, default None
        If given, compute distances only from these source nodes (rows of the
        returned matrix); otherwise all-pairs.

    Returns
    -------
    np.ndarray
        ``(len(sources), n)`` (or ``(n, n)``) shortest-path distance matrix;
        unreachable pairs are ``inf``.
    """

    graph = csr_matrix((np.asarray(weights, dtype=np.float64), (rows, cols)),
                       shape=(n, n))
    graph = graph.maximum(graph.transpose())
    return dijkstra(graph, directed=False, indices=sources)


def density_geodesic_matrix(nodes: np.ndarray, density: np.ndarray, *,
                            period: float = 1.0, k: int = 8,
                            density_exponent: float = 1.0,
                            sources=None) -> np.ndarray:
    """
    Density-ratio (Fermat-style) graph geodesic distances over ``nodes``.

    Geometry (2). On a wrap-aware k-NN graph, each edge's flat-torus length is
    reweighted by the inverse aggregate-posterior density at the edge (the mean
    of its endpoints), raised to ``density_exponent``:

        ``w_ij = flat_len_ij * (1 / density_mid_ij) ** density_exponent``.

    Shortest paths therefore prefer high-density corridors and pay a large cost
    to cross low-density valleys -- the "route through dense regions, avoid the
    class boundaries" behaviour of the paper's Appendix C geodesic, realised as a
    symmetric proper metric.

    Parameters
    ----------
    nodes : np.ndarray
        ``(n, 2)`` node coordinates.
    density : np.ndarray
        ``(n,)`` non-negative density at each node (e.g. from
        :func:`torus_kde_density`).
    period : float, default 1.0
        Per-axis wrap period.
    k : int, default 8
        Neighbours per node in the graph.
    density_exponent : float, default 1.0
        Exponent ``alpha`` on the inverse density. ``0`` recovers the flat
        graph metric; larger values push paths harder toward dense regions.
    sources : array-like | None, default None
        Restrict to these source nodes (else all-pairs).

    Returns
    -------
    np.ndarray
        Shortest-path distance matrix (``inf`` for unreachable pairs).
    """

    nodes = np.asarray(nodes, dtype=np.float64)
    density = np.asarray(density, dtype=np.float64)
    rows, cols, flat_len = _torus_knn_edges(nodes, k, period)
    dens_mid = 0.5 * (density[rows] + density[cols])
    inv = 1.0 / np.maximum(dens_mid, _DENSITY_FLOOR)
    weights = flat_len * inv ** float(density_exponent)
    return _dijkstra_from_edges(nodes.shape[0], rows, cols, weights, sources)


def pullback_metric_at_nodes(nodes: np.ndarray, decode_fn) -> np.ndarray:
    """
    Decoder-pullback Riemannian metric ``G(z) = J(z)^T J(z)`` at each node.

    ``decode_fn`` maps a single latent coordinate ``z`` (shape ``(2,)``) to a
    flat decoded output ``g(z)`` (shape ``(D,)``); its Jacobian ``J = dg/dz``
    (shape ``(D, 2)``) is taken with forward-mode autodiff (efficient here since
    there are 2 inputs and many outputs), and ``G = J^T J`` (shape ``(2, 2)``) is
    the local metric measuring how fast the decoded output moves per unit latent
    displacement. Vectorised over all nodes.

    Parameters
    ----------
    nodes : np.ndarray
        ``(n, 2)`` latent coordinates.
    decode_fn : callable
        ``z (2,) -> g(z) (D,)`` differentiable JAX function.

    Returns
    -------
    np.ndarray
        ``(n, 2, 2)`` symmetric positive-semidefinite metric tensors.
    """

    z = jnp.asarray(np.asarray(nodes, dtype=np.float64))
    jac = jax.vmap(jax.jacfwd(decode_fn))(z)          # (n, D, 2)
    g = jnp.einsum('ndi,ndj->nij', jac, jac)          # (n, 2, 2)
    return np.asarray(g, dtype=np.float64)


def pullback_geodesic_matrix(nodes: np.ndarray, decode_fn, *,
                             period: float = 1.0, k: int = 8,
                             sources=None,
                             metric_tensors: np.ndarray = None) -> np.ndarray:
    """
    Decoder-Jacobian pullback graph geodesic distances over ``nodes``.

    Geometry (3). On a wrap-aware k-NN graph, each edge ``i -> j`` is given the
    local pullback length ``sqrt(delta^T G_mid delta)`` where ``delta`` is the
    wrap-aware displacement ``signed_diff(z_j, z_i)`` and ``G_mid = (G_i + G_j)/2``
    averages the endpoint metrics (:func:`pullback_metric_at_nodes`). Shortest
    paths under these weights approximate geodesics of the decoder-pulled-back
    metric -- distances measured in decoded-spectrogram change rather than in raw
    latent coordinates.

    Parameters
    ----------
    nodes : np.ndarray
        ``(n, 2)`` node coordinates.
    decode_fn : callable
        ``z (2,) -> g(z) (D,)`` differentiable JAX decoder. Ignored when
        ``metric_tensors`` is supplied.
    period : float, default 1.0
        Per-axis wrap period (for the wrap-aware edge displacement).
    k : int, default 8
        Neighbours per node in the graph.
    sources : array-like | None, default None
        Restrict to these source nodes (else all-pairs).
    metric_tensors : np.ndarray | None, default None
        Optional precomputed ``(n, 2, 2)`` metric tensors to reuse instead of
        recomputing them from ``decode_fn`` (they are the expensive part).

    Returns
    -------
    np.ndarray
        Shortest-path distance matrix (``inf`` for unreachable pairs).
    """

    nodes = np.asarray(nodes, dtype=np.float64)
    g = metric_tensors if metric_tensors is not None else \
        pullback_metric_at_nodes(nodes, decode_fn)
    g = np.asarray(g, dtype=np.float64)
    rows, cols, _ = _torus_knn_edges(nodes, k, period)
    delta = signed_diff(nodes[cols], nodes[rows], metric='torus', period=period)  # (E, 2)
    g_mid = 0.5 * (g[rows] + g[cols])                                             # (E, 2, 2)
    quad = np.einsum('ei,eij,ej->e', delta, g_mid, delta)
    weights = np.sqrt(np.maximum(quad, 0.0))
    return _dijkstra_from_edges(nodes.shape[0], rows, cols, weights, sources)


def make_qlvm_decode_fn(params: dict, condition: float = None):
    """
    Build the differentiable QLVM decode function ``z -> flattened spectrogram``.

    Composes the fixed sin/cos torus basis with the frozen ConvTranspose decoder
    (``qlvm_model.torus_basis_forward`` then ``decoder_forward``) into a single
    ``z (2,) -> (128*128,)`` JAX function suitable for
    :func:`pullback_metric_at_nodes`. The QLVM torus is native period 1.0, so the
    latent is reduced mod 1 before the basis (matching ``decode_lattice_atlas``).
    A conditional decoder (``c_dim`` 1) takes its conditioning value appended to
    the basis, exactly as ``qlvm_model.decode_shifted_lattice`` appends it; the
    value is a constant of the function, so the Jacobian is taken with respect to
    the torus position only.

    Parameters
    ----------
    params : dict
        Decoder weights as returned by
        ``processing.qlvm_latents.load_decoder_params`` (the ``params`` entry of
        ``processing.qlvm_latents.load_model_cell``).
    condition : float | None, default None
        The conditioning value of a conditional decoder (e.g.
        ``processing.qlvm_latents.model_decode_condition``); None for an
        unconditional decoder.

    Returns
    -------
    callable
        ``z (2,) -> g(z) (16384,)`` differentiable decode function.
    """

    condition_row = None if condition is None else jnp.full((1, 1), float(condition), dtype=jnp.float32)

    def decode_fn(z: jnp.ndarray) -> jnp.ndarray:
        basis = torus_basis_forward((z % 1.0)[None, :])   # (1, 4)
        if condition_row is not None:
            basis = jnp.concatenate([basis, condition_row.astype(basis.dtype)], axis=-1)   # (1, 5)
        rec = decoder_forward(basis, params)              # (1, 1, 128, 128)
        return rec.reshape(-1)                            # (16384,)

    return decode_fn


def make_qlvm_decode_fn_from_model_cell(model_cell_directory: str, condition_quantile: float = None):
    """
    Description
    -----------
    Loads the decoder of one QLVM model package cell and returns its decode
    function, so the pullback metric is computed with the same decoder whose
    torus the manifold coordinates live on (for the production summaries, the
    cell of the coordinates' map, ``os_utils.qlvm_map_cell_directory``:
    ``qlvm1`` / ``qlvm2`` -> ``.../spectrograms/qlvm/qlvm``,
    ``qlvm_duration1`` / ``qlvm_duration2`` ->
    ``.../spectrograms/qlvm/qlvm_duration``, ``qlvm_squeak1`` /
    ``qlvm_squeak2`` -> the squeak cell). The cell is read by
    :func:`processing.qlvm_latents.load_model_cell` (its ``checkpoint.tar``
    without torch, plus its training contract), and the weights go through
    :func:`make_qlvm_decode_fn`, whose ``qlvm_model.decoder_forward`` runs either
    decoder head (``"legacy"`` or ``"relu"``) exactly as the embedding's lattice
    decode does.

    A conditional cell (``c_dim`` 1) decodes ``(z, c)``, so ``z -> spectrogram``
    is a function of the torus position only once ``c`` is fixed: it is decoded
    at ``processing.qlvm_latents.condition_quantile_value`` of
    ``condition_quantile`` (the corpus quantile of the cell's own training
    conditioning distribution, decoded by the cell's exact / grid rule), and
    without a ``condition_quantile`` it is refused. An unconditional cell ignores
    ``condition_quantile``.

    Parameters
    ----------
    model_cell_directory (str)
        Path to the package cell (routed through ``configure_path`` by
        ``load_model_cell``).
    condition_quantile (float | None)
        The training-corpus quantile of the conditioning value a conditional
        cell is decoded at, in ``[0, 1]``; None (the default) refuses a
        conditional cell.

    Returns
    -------
    decode_fn (callable)
        ``z (2,) -> g(z) (16384,)`` differentiable decode function.
    """

    model = load_model_cell(model_cell_directory)
    if model['contract']['c_dim'] != 0 and condition_quantile is None:
        error_message = (
            f"make_qlvm_decode_fn_from_model_cell: {model['model_id']} is a conditional cell "
            f"(c_dim {model['contract']['c_dim']}, condition {model['contract']['condition']!r}); the "
            f"pullback metric needs the conditioning value fixed, so pass condition_quantile."
        )
        raise ValueError(error_message)
    condition = model_decode_condition(model, condition_quantile) if model['contract']['c_dim'] else None
    if condition is not None:
        print(f"    [geodesic] {model['model_id']} decoded at {model['contract']['condition']['name']} "
              f"c = {float(condition):.4f} (training-corpus quantile {float(condition_quantile):.3g})")
    return make_qlvm_decode_fn(model['params'], condition=condition)


def resolve_manifold_column_names(input_metadata: dict | None, vocal_feature_settings: dict) -> list:
    """
    Description
    -----------
    The two summary columns the manifold target ``Y`` of a modeling input pickle
    was read from, which name the QLVM map (and so the decoder) the torus
    geodesics work on. The pickle's own record comes first
    (``_input_metadata.analysis_specific.usv_manifold_column_names``, written by
    the extraction), because it describes the coordinates actually in ``Y`` even
    if the settings were edited after the extraction; a pickle written before
    that record existed falls back to the current
    ``vocal_features.usv_manifold_column_names`` setting.

    Parameters
    ----------
    input_metadata (dict | None)
        The input pickle's ``_input_metadata`` block, or None when it has none.
    vocal_feature_settings (dict)
        The ``vocal_features`` settings block.

    Returns
    -------
    manifold_column_names (list)
        The manifold column names (e.g. ``['qlvm_duration1', 'qlvm_duration2']``).
    """

    if input_metadata is not None and 'analysis_specific' in input_metadata:
        analysis_specific = input_metadata['analysis_specific']
        if analysis_specific is not None and 'usv_manifold_column_names' in analysis_specific:
            return list(analysis_specific['usv_manifold_column_names'])
    return list(vocal_feature_settings['usv_manifold_column_names'])


def resolve_geodesic_decoder_source(geodesic_settings: dict,
                                    manifold_column_names: list) -> tuple[str, str, float] | None:
    """
    Description
    -----------
    Reads whether the pullback geodesic metric is computed from the
    ``vocal_features.usv_manifold_geodesic_metrics`` block and, when it is, which
    decoder defines it. The decoder is not a settings path: it is the production
    cell of the map the manifold coordinates come from, the map prefix of
    ``manifold_column_names`` (``modeling_utils.manifold_tag_segment``) resolved
    by ``os_utils.qlvm_map_cell_directory`` (``qlvm`` ->
    ``.../spectrograms/qlvm/qlvm``, ``qlvm_duration`` / ``qlvm_entropy`` /
    ``qlvm_bandwidth`` / ``qlvm_loudness`` -> their cells
    ``.../spectrograms/qlvm/<map>``, ``qlvm_squeak`` -> the squeak cell).
    A pullback metric of one map's coordinates under another map's decoder would
    measure distances on a torus the coordinates do not live on. Taking the cell
    from the code constants keeps the GUI / CLI experimenter re-keying of settings
    paths from ever pointing the metric at another experimenter's folder (where no
    cell exists).

    ``pullback_metric`` (bool) switches the pullback metric: false means no
    pullback metric (``pullback_geodesic_mae`` is NaN), the density-ratio
    geodesic is unaffected. ``pullback_condition_quantile`` (float in ``[0, 1]``)
    fixes the conditioning value of a conditional map's decoder: the decoder is
    evaluated at that quantile of its training corpus's conditioning distribution
    (``processing.qlvm_latents.condition_quantile_value``; ``0.5`` is the corpus
    median call), the same value for every grid node, so the metric is one
    Riemannian metric of the torus; the regular and squeak decoders take no
    conditioning value and ignore it. This is read before any geometry is built,
    so a configuration error (a missing key, a manifold whose columns name no QLVM
    map) is not swallowed by the geometry's soft-failure handling.

    Parameters
    ----------
    geodesic_settings (dict)
        The ``usv_manifold_geodesic_metrics`` settings block; its
        ``pullback_metric`` flag and ``pullback_condition_quantile`` are read.
    manifold_column_names (list)
        The manifold columns of the run (:func:`resolve_manifold_column_names`).

    Returns
    -------
    source (tuple[str, str, float] | None)
        ``('model_cell', <cell of the map, canonical form>,
        <pullback_condition_quantile>)``, or None when ``pullback_metric`` is
        false.

    Raises
    ------
    ValueError
        ``pullback_metric`` is on, but the manifold columns name no map of
        ``os_utils.QLVM_DECODER_MAPS`` (no decoder defines their torus), or
        ``pullback_condition_quantile`` is outside ``[0, 1]``.
    """

    if not geodesic_settings['pullback_metric']:
        return None
    condition_quantile = float(geodesic_settings['pullback_condition_quantile'])
    if not 0.0 <= condition_quantile <= 1.0:
        error_message = (
            f"usv_manifold_geodesic_metrics.pullback_condition_quantile must be in [0, 1], "
            f"got {condition_quantile!r}."
        )
        raise ValueError(error_message)
    qlvm_map = manifold_tag_segment(list(manifold_column_names))
    if qlvm_map not in QLVM_DECODER_MAPS:
        error_message = (
            f"usv_manifold_geodesic_metrics.pullback_metric is on, but the manifold columns "
            f"{list(manifold_column_names)} name the map {qlvm_map!r}, which has no decoder (one of "
            f"{QLVM_DECODER_MAPS}); switch pullback_metric off for this manifold."
        )
        raise ValueError(error_message)
    return 'model_cell', qlvm_map_cell_directory(qlvm_map), condition_quantile


def make_qlvm_decode_fn_from_source(source: tuple[str, str, float]):
    """
    Description
    -----------
    Builds the decode function for a decoder source returned by
    :func:`resolve_geodesic_decoder_source`.

    Parameters
    ----------
    source (tuple[str, str, float])
        ``('model_cell', <directory>, <condition quantile>)``; the quantile fixes
        the conditioning value of a conditional cell and is ignored by an
        unconditional one.

    Returns
    -------
    decode_fn (callable)
        ``z (2,) -> g(z) (16384,)`` differentiable decode function.
    """

    kind, location, condition_quantile = source
    if kind == 'model_cell':
        return make_qlvm_decode_fn_from_model_cell(location, condition_quantile=condition_quantile)
    error_message = f"make_qlvm_decode_fn_from_source: unknown decoder source kind {kind!r}."
    raise ValueError(error_message)


def torus_distance(nodes: np.ndarray, method: str, *,
                   period: float = 1.0, k: int = 8,
                   density: np.ndarray = None,
                   decode_fn=None,
                   metric_tensors: np.ndarray = None,
                   density_exponent: float = 1.0,
                   sources=None) -> np.ndarray:
    """
    Unified dispatcher for the three torus distance geometries over ``nodes``.

    Parameters
    ----------
    nodes : np.ndarray
        ``(n, 2)`` node coordinates in ``[0, period)``.
    method : str
        ``'flat'``            -> :func:`flat_torus_distance_matrix` (ignores
                                 ``k`` / ``density`` / ``decode_fn``);
        ``'density_geodesic'``-> :func:`density_geodesic_matrix` (needs
                                 ``density``);
        ``'pullback_geodesic'``-> :func:`pullback_geodesic_matrix` (needs
                                 ``decode_fn`` or ``metric_tensors``).
    period : float, default 1.0
        Per-axis wrap period.
    k : int, default 8
        Graph neighbours (geodesic methods only).
    density : np.ndarray | None
        ``(n,)`` node densities, required for ``'density_geodesic'``.
    decode_fn : callable | None
        Differentiable decoder, required for ``'pullback_geodesic'`` unless
        ``metric_tensors`` is given.
    metric_tensors : np.ndarray | None
        Optional precomputed ``(n, 2, 2)`` pullback metrics.
    density_exponent : float, default 1.0
        Inverse-density exponent for ``'density_geodesic'``.
    sources : array-like | None
        Restrict to these source nodes (else all-pairs).

    Returns
    -------
    np.ndarray
        Distance matrix under the chosen geometry.

    Raises
    ------
    ValueError
        If ``method`` is unknown or a required input for it is missing.
    """

    if method == 'flat':
        full = flat_torus_distance_matrix(nodes, period=period)
        return full if sources is None else full[np.asarray(sources)]
    if method == 'density_geodesic':
        if density is None:
            raise ValueError("method='density_geodesic' requires `density` at each node.")
        return density_geodesic_matrix(nodes, density, period=period, k=k,
                                       density_exponent=density_exponent, sources=sources)
    if method == 'pullback_geodesic':
        if decode_fn is None and metric_tensors is None:
            raise ValueError("method='pullback_geodesic' requires `decode_fn` or "
                             "`metric_tensors`.")
        return pullback_geodesic_matrix(nodes, decode_fn, period=period, k=k,
                                        sources=sources, metric_tensors=metric_tensors)
    raise ValueError(f"Unknown method {method!r}; expected 'flat', "
                     f"'density_geodesic', or 'pullback_geodesic'.")


def _snap_to_grid(points: np.ndarray, n_per_dim: int, period: float) -> np.ndarray:
    """
    Nearest regular-grid node index for each point (wrap-aware).

    The grid from :func:`torus_grid` places node ``(i, j)`` at cell centre
    ``((i + 0.5), (j + 0.5)) * period / n_per_dim`` and ravels it in ``'ij'``
    order (flat index ``i * n_per_dim + j``). The nearest node per axis is
    ``round(coord / cell - 0.5) mod n_per_dim``, which wraps correctly at the
    seam.

    Parameters
    ----------
    points : np.ndarray
        ``(m, 2)`` coordinates in ``[0, period)``.
    n_per_dim : int
        Grid cells per axis (must match the grid the indices are used against).
    period : float
        Per-axis wrap period.

    Returns
    -------
    np.ndarray
        ``(m,)`` flat node indices into an ``n_per_dim ** 2`` grid.
    """

    pts = np.asarray(points, dtype=np.float64)
    n = int(n_per_dim)
    cell = float(period) / n
    idx = np.round(pts / cell - 0.5).astype(np.int64) % n      # (m, 2)
    return idx[:, 0] * n + idx[:, 1]


def build_torus_geodesic_context(embedded_points: np.ndarray, *, decode_fn=None,
                                 n_per_dim: int = 40, period: float = 1.0, k: int = 8,
                                 density_exponent: float = 1.0,
                                 bw_method=None) -> TorusGeodesicContext:
    """
    Precompute the grid geodesic geometry once, so per-event prediction errors
    are cheap snap-to-grid lookups.

    Builds a regular ``torus_grid``, the wrap-aware KDE density at its nodes
    (from ``embedded_points``), the all-pairs density-ratio geodesic distance
    matrix, and -- when a differentiable ``decode_fn`` is supplied -- the
    all-pairs decoder-Jacobian pullback geodesic distance matrix. The geometry
    is fixed (it depends on the embedding + decoder, not on any behavioural
    prediction), so this is computed a single time and reused across folds.

    Parameters
    ----------
    embedded_points : np.ndarray
        ``(m, 2)`` embedded torus coordinates the density is estimated from
        (e.g. all training-fold ``Y``).
    decode_fn : callable | None, default None
        Differentiable ``z (2,) -> g(z) (D,)`` decoder for the pullback geometry
        (e.g. :func:`make_qlvm_decode_fn`). ``None`` -> ``pullback_matrix`` is
        ``None`` and the pullback error is reported as ``nan``.
    n_per_dim : int, default 40
        Grid resolution per axis (``n_per_dim ** 2`` nodes).
    period : float, default 1.0
        Per-axis wrap period.
    k : int, default 8
        Graph neighbours per node.
    density_exponent : float, default 1.0
        Inverse-density exponent for the density geodesic.
    bw_method : str | float | callable | None, default None
        Forwarded to the KDE (``None`` -> Scott's rule).

    Returns
    -------
    TorusGeodesicContext
        The grid, its resolution/period, node densities, and the density /
        pullback geodesic distance matrices (the latter ``None`` without a
        decoder).
    """

    grid = torus_grid(n_per_dim, period=period)
    density = torus_kde_density(embedded_points, grid, period=period, bw_method=bw_method)
    density_matrix = density_geodesic_matrix(grid, density, period=period, k=k,
                                             density_exponent=density_exponent)
    pullback_matrix = None
    if decode_fn is not None:
        pullback_matrix = pullback_geodesic_matrix(grid, decode_fn, period=period, k=k)
    return TorusGeodesicContext(grid=grid, n_per_dim=int(n_per_dim), period=float(period),
                                density=density, density_matrix=density_matrix,
                                pullback_matrix=pullback_matrix)


def per_event_geodesic_error(y_pred: np.ndarray, y_true: np.ndarray,
                             ctx: TorusGeodesicContext, *, method: str) -> np.ndarray:
    """
    Per-event geodesic distance between prediction and truth via snap + lookup.

    Snaps both ``y_pred`` and ``y_true`` to their nearest grid nodes and reads
    the requested precomputed geodesic distance between them -- the per-event
    analogue of the flat-torus residual, but under the density or pullback
    geometry. Returns all-``nan`` (rather than raising) when the requested
    geometry was not precomputed (e.g. pullback with no decoder), so callers can
    add the column unconditionally.

    Parameters
    ----------
    y_pred, y_true : np.ndarray
        ``(n, 2)`` predicted / true torus coordinates.
    ctx : TorusGeodesicContext | None
        Precomputed geometry from :func:`build_torus_geodesic_context`. ``None``
        (or a context whose matrix for ``method`` is ``None``) yields all-``nan``.
    method : str
        ``'density_geodesic'`` or ``'pullback_geodesic'``.

    Returns
    -------
    np.ndarray
        ``(n,)`` per-event geodesic distances (``nan`` where unavailable).

    Raises
    ------
    ValueError
        If ``method`` is not a geodesic method.
    """

    if method not in ('density_geodesic', 'pullback_geodesic'):
        raise ValueError(f"method must be 'density_geodesic' or 'pullback_geodesic', "
                         f"got {method!r}.")
    n = np.asarray(y_pred).shape[0]
    matrix = None if ctx is None else (
        ctx.density_matrix if method == 'density_geodesic' else ctx.pullback_matrix)
    if matrix is None:
        return np.full(n, np.nan)
    ip = _snap_to_grid(y_pred, ctx.n_per_dim, ctx.period)
    it = _snap_to_grid(y_true, ctx.n_per_dim, ctx.period)
    return np.asarray(matrix)[ip, it]


def geodesic_mae_columns(y_pred: np.ndarray, y_true: np.ndarray,
                         ctx: TorusGeodesicContext) -> dict:
    """
    The two geodesic prediction-error metrics as a metric-bundle fragment.

    Computes the mean per-event density-geodesic and pullback-geodesic distance
    between prediction and truth, ready to merge into a fold's metric dict
    alongside ``euclidean_mae`` / ``mahalanobis_mae``. Each is ``nan`` when its
    geometry was not precomputed or no event has a finite distance.

    Parameters
    ----------
    y_pred, y_true : np.ndarray
        ``(n, 2)`` predicted / true torus coordinates.
    ctx : TorusGeodesicContext | None
        Precomputed geometry; ``None`` (or a missing per-method matrix) yields a
        ``nan`` for that column.

    Returns
    -------
    dict
        ``{'density_geodesic_mae': float, 'pullback_geodesic_mae': float}``.
    """

    def _safe_mae(d):
        d = np.asarray(d, dtype=np.float64)
        finite = np.isfinite(d)
        return float(d[finite].mean()) if finite.any() else float('nan')

    return {
        'density_geodesic_mae': _safe_mae(
            per_event_geodesic_error(y_pred, y_true, ctx, method='density_geodesic')),
        'pullback_geodesic_mae': _safe_mae(
            per_event_geodesic_error(y_pred, y_true, ctx, method='pullback_geodesic')),
    }
