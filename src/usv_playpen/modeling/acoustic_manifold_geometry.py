"""
@author: bartulem
Acoustic-manifold geometry helpers for downstream saliency /
region-of-interest analyses on USV manifold embeddings.

Three concerns live here, and only here, so the same logic is shared
between the CNN saliency phase, the linear-univariate post-hocs, and any
future region-conditional analysis:

1. **Empirical cluster centres** — for each unique label in a per-USV
   (manifold-position, label) pool, locate the highest-density point of
   the per-label distribution via a 2-D Gaussian KDE on a regular grid.
   This is the principled "where does cluster X actually live on the
   manifold" answer, robust to outliers (unlike the arithmetic mean) and
   independent of any upstream watershed-or-other clustering algorithm's
   bookkeeping. On a torus the KDE is built on a 3x3 lattice-replicated
   copy of the points so a cluster straddling the wrap boundary is not
   spuriously pulled toward the geometric centre of `[0, period)^2`.

2. **Per-cluster radius** — given a set of centres, derive a circular
   region of inclusion per cluster, either *adaptive* (each cluster gets
   `alpha * d_nn(i) / 2` where `d_nn(i)` is its distance to the nearest
   other centre, so dense neighbourhoods get small circles and isolated
   clusters get larger ones) or *uniform* (a single radius
   `alpha * min_i d_nn(i) / 2` shared across all clusters, sized by the
   globally tightest pair). With `alpha <= 1.0` no two circles ever
   overlap each other under either mode, because for any pair `(i, j)`
   each radius is bounded above by `alpha * d_ij / 2`.

3. **Wrap-aware circle membership** — a vectorised "is this point inside
   the circle centred at `c` with radius `r`?" test that uses the same
   metric / period as the rest of the pipeline. Used at saliency-extraction
   time to gate USVs by spatial proximity to a cluster centre, in
   combination with the categorical label check.

Notes on noise filtering
------------------------
The pipeline's `noise_vocal_categories` setting (typically `[0]`) marks
USVs that are not biological vocalisations. `derive_cluster_centers_empirically`
accepts a `drop_label` argument so the noise label is excluded from the
returned centres without the caller having to pre-filter; the default
is `0` to match the project convention.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from .manifold_metric import _validate_metric_period, signed_diff


def derive_cluster_centers_empirically(Y: np.ndarray,
                                       labels: np.ndarray,
                                       *,
                                       drop_label: int | float | None = 0,
                                       min_points_per_label: int = 20,
                                       max_points_per_label: int = 30000,
                                       grid_resolution: int = 200,
                                       kde_bandwidth: float | str | None = None,
                                       metric: str = 'euclidean',
                                       period: float = 1.0,
                                       rng_seed: int = 0) -> dict:
    """
    Description
    -----------
    Locate the highest-density point of each unique label in a per-USV
    `(Y, labels)` pool, using a 2-D Gaussian KDE evaluated on a regular
    grid. Designed for USV-manifold cluster-centre estimation where the
    KDE mode is the principled stand-in for the arithmetic mean of a
    cluster's points (robust to outliers).

    On a torus the KDE is fit on a `3x3` lattice-replicated copy of the
    label's points so a cluster straddling the wrap boundary returns a
    centre on the actual cluster instead of on the antipode. The grid the
    KDE is evaluated on is built from the *original* (un-replicated)
    per-label data range, with a small margin, so the returned centre is
    always inside the canonical `[0, period)^2` cell on torus.

    Parameters
    ----------
    Y : numpy.ndarray
        `(N, 2)` manifold positions per USV.
    labels : numpy.ndarray
        `(N,)` per-USV cluster labels, integer- or float-valued. NaN
        labels are skipped automatically.
    drop_label : int or float or None, default 0
        Label value to exclude from the returned centres entirely (no
        KDE fit). The project convention is `0` for "noise" / discarded
        USVs. Pass `None` to keep every label.
    min_points_per_label : int, default 20
        Skip labels with fewer than this many finite-coordinate points
        (the per-label KDE is numerically unstable below this).
    max_points_per_label : int, default 30_000
        When a label has more than this many points, a uniform
        sub-sample of size `max_points_per_label` is used. The KDE-mode
        estimate is robust at this sample size; the cap keeps wall-clock
        per label bounded for large pools.
    grid_resolution : int, default 200
        Resolution of the regular grid the KDE is evaluated on, per
        axis. The returned centre is the grid point of maximum density,
        so the quantisation floor on the centre is `range / grid_resolution`.
    kde_bandwidth : float or str or None, default None
        Forwarded to `scipy.stats.gaussian_kde`'s `bw_method`. `None`
        selects Scott's rule. To reproduce the QLVM watershed pipeline's
        seeding, pass `0.8` (matches the `params_bandwidth` attribute on
        the precomputed QLVM cluster H5).
    metric : str, default 'euclidean'
        `'euclidean'` for flat-plane manifolds (e.g. VAE-UMAP) or
        `'torus'` for periodic ones (e.g. QLVM-UMAP). Controls the
        wrap-aware KDE on torus.
    period : float, default 1.0
        Per-axis wrap period; ignored when `metric == 'euclidean'`.
    rng_seed : int, default 0
        Seed for the per-label sub-sample draw so the centres are
        deterministic given identical inputs.

    Returns
    -------
    dict
        ``{label_value: numpy.ndarray of shape (2,)}``. Labels appear in
        sorted order. Labels equal to ``drop_label`` are absent, as are
        labels with fewer than ``min_points_per_label`` finite points.
    """

    _validate_metric_period(metric, period)
    rng = np.random.default_rng(rng_seed)

    finite_xy = np.isfinite(Y).all(axis=1)
    out: dict = {}
    for lbl in sorted(np.unique(labels[~pd.isna(labels)])):
        if drop_label is not None and lbl == drop_label:
            continue
        mask = (labels == lbl) & finite_xy
        n_total = int(mask.sum())
        if n_total < min_points_per_label:
            continue

        pts = Y[mask]
        if n_total > max_points_per_label:
            pts = pts[rng.choice(n_total, max_points_per_label, replace=False)]

        if metric == 'torus':
            # Lattice-replicate so wrap-straddling clusters are handled.
            # The KDE consumes the (9*N, 2) replicated point cloud; the
            # density on the canonical cell sums contributions from all
            # nine shifts of the original points.
            shifts = np.array(
                [(dx, dy) for dx in (-period, 0.0, period) for dy in (-period, 0.0, period)],
                dtype=np.float64,
            )
            pts_kde = np.concatenate([pts + s for s in shifts], axis=0)
        else:
            pts_kde = pts

        kde = gaussian_kde(pts_kde.T, bw_method=kde_bandwidth)

        # Grid covers the original per-label data range (not the replicated
        # one) with a small margin so the returned centre is inside the
        # canonical cell on torus.
        margin = 0.05 * (pts.max(axis=0) - pts.min(axis=0)).max()
        xmin, ymin = pts.min(axis=0) - margin
        xmax, ymax = pts.max(axis=0) + margin

        if metric == 'torus':
            xmin = max(xmin, 0.0)
            ymin = max(ymin, 0.0)
            xmax = min(xmax, period)
            ymax = min(ymax, period)

        xx, yy = np.mgrid[xmin:xmax:complex(0, grid_resolution),
                          ymin:ymax:complex(0, grid_resolution)]
        zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
        idx = np.unravel_index(int(np.argmax(zz)), zz.shape)
        out[lbl] = np.array([float(xx[idx]), float(yy[idx])])

    return out


def derive_cluster_geometry(centres: dict,
                            *,
                            alpha: float = 0.5,
                            mode: str = 'adaptive',
                            max_radius: float | None = None,
                            metric: str = 'euclidean',
                            period: float = 1.0) -> dict:
    """
    Description
    -----------
    Compute per-cluster circle geometry (`centroid`, `radius`) from a set
    of cluster centres. The radius rule is the same family in both modes:

        adaptive : r_i = alpha * d_nearest_neighbour(i) / 2
        uniform  : r   = alpha * min_i d_nearest_neighbour(i) / 2  (all clusters)

    where ``d_nearest_neighbour(i)`` is the distance from centre `i` to
    its single closest other centre under the supplied metric. With
    ``alpha <= 1.0`` no two circles overlap each other, because for any
    pair `(i, j)` each radius is at most ``alpha * d_ij / 2`` and the sum
    is at most ``alpha * d_ij <= d_ij``.

    The adaptive mode lets isolated clusters get larger circles while
    keeping circles in dense neighbourhoods tight. The uniform mode gives
    every cluster the same circle size, sized by the globally tightest
    pair; this is the right choice when the downstream analysis needs a
    constant region area across clusters (e.g. equal-area sampling for
    saliency-map normalisation).

    Parameters
    ----------
    centres : dict
        ``{label: (cx, cy)}`` or ``{label: numpy.ndarray of shape (2,)}``,
        as produced by :func:`derive_cluster_centers_empirically` or read
        from an upstream watershed-cluster artifact.
    alpha : float, default 0.5
        Gap factor in `(0, 1]`. ``alpha = 1.0`` produces tangent circles
        (touching at the midpoint between adjacent centres); smaller
        values leave proportional gaps. The project default is `0.5`,
        which produces a clearly visible gap and makes label-purity
        violations (USVs in cluster A's circle but assigned to cluster B
        by the labelling) extremely rare.
    mode : str, default 'adaptive'
        Either ``'adaptive'`` (per-cluster radius scaled by local
        nearest-neighbour distance) or ``'uniform'`` (one global radius
        sized by the globally tightest pair).
    max_radius : float or None, default None
        Optional cap applied after the rule above; when supplied,
        ``radius = min(radius, max_radius)``. Useful when an isolated
        cluster's natural radius would otherwise engulf a large fraction
        of the manifold; not needed when the cluster centres are
        approximately balanced.
    metric : str, default 'euclidean'
        ``'euclidean'`` or ``'torus'``. The pairwise distance between
        centres is computed under this metric.
    period : float, default 1.0
        Per-axis wrap period; ignored when ``metric == 'euclidean'``.

    Returns
    -------
    dict
        ``{label: {'centroid': numpy.ndarray of shape (2,), 'radius': float,
        'nearest_neighbour_distance': float}}``. The
        ``'nearest_neighbour_distance'`` is preserved verbatim for
        diagnostics even when the radius has been clamped by
        ``max_radius`` or made uniform.
    """

    _validate_metric_period(metric, period)
    if alpha <= 0.0:
        raise ValueError(f"alpha must be > 0; got {alpha!r}")
    if mode not in ('adaptive', 'uniform'):
        raise ValueError(f"mode must be 'adaptive' or 'uniform'; got {mode!r}")
    if len(centres) < 2:
        raise ValueError(
            f"derive_cluster_geometry needs at least 2 centres to define a nearest-neighbour distance; "
            f"got {len(centres)}."
        )

    labels = list(centres.keys())
    C = np.asarray([np.asarray(centres[k], dtype=np.float64) for k in labels])

    # Per-centre nearest-neighbour distance under the chosen metric.
    n = len(C)
    d_nn = np.full(n, np.inf, dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d_ij = float(np.sqrt((signed_diff(C[i], C[j], metric=metric, period=period) ** 2).sum()))
            if d_ij < d_nn[i]:
                d_nn[i] = d_ij

    if mode == 'uniform':
        uniform_radius = alpha * float(d_nn.min()) / 2.0
        if max_radius is not None:
            uniform_radius = min(uniform_radius, max_radius)
        radii = np.full(n, uniform_radius, dtype=np.float64)
    else:
        radii = alpha * d_nn / 2.0
        if max_radius is not None:
            radii = np.minimum(radii, max_radius)

    return {
        labels[i]: {
            'centroid': C[i],
            'radius': float(radii[i]),
            'nearest_neighbour_distance': float(d_nn[i]),
        }
        for i in range(n)
    }


def usv_in_circle(Y: np.ndarray,
                  centroid: np.ndarray,
                  radius: float,
                  *,
                  metric: str = 'euclidean',
                  period: float = 1.0) -> np.ndarray:
    """
    Description
    -----------
    Vectorised wrap-aware "is this point inside the circle centred at
    ``centroid`` with radius ``radius``" membership test. On torus, the
    signed difference between each point and the centroid is wrapped into
    `(-period/2, period/2]` per axis before squaring, so points near the
    wrap boundary are correctly judged close to a boundary-adjacent
    centroid.

    Parameters
    ----------
    Y : numpy.ndarray
        ``(N, 2)`` array of per-USV manifold positions.
    centroid : numpy.ndarray
        Length-2 cluster centre.
    radius : float
        Circle radius in the same units as the manifold coordinates.
        ``radius <= 0`` is allowed and returns an all-``False`` mask.
    metric : str, default 'euclidean'
        ``'euclidean'`` or ``'torus'``.
    period : float, default 1.0
        Per-axis wrap period; ignored when ``metric == 'euclidean'``.

    Returns
    -------
    numpy.ndarray
        Boolean array of shape ``(N,)``; ``True`` where the USV is
        inside the closed disk of radius ``radius`` around ``centroid``
        under the supplied metric.
    """

    _validate_metric_period(metric, period)
    if radius <= 0.0:
        return np.zeros(len(Y), dtype=bool)

    Y_arr = np.asarray(Y, dtype=np.float64)
    centroid_arr = np.asarray(centroid, dtype=np.float64).reshape(1, 2)
    diff = signed_diff(Y_arr, centroid_arr, metric=metric, period=period)
    dist_sq = np.sum(diff ** 2, axis=1)
    return dist_sq <= float(radius) ** 2
