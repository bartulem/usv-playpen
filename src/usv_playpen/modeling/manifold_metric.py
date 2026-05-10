"""
@author: bartulem
Distance / centroid helpers for USV manifold targets.

The continuous regression and CNN pipelines in this project predict 2-D
acoustic-manifold positions from behavioural-kinematic histories. Two
manifolds are currently supported:

- **`'euclidean'`** — the VAE UMAP manifold (`vae_umap1`, `vae_umap2`).
  Treats the 2-D plane as flat R^2; standard Euclidean distance,
  arithmetic mean, sample covariance.
- **`'torus'`** — the QLVM UMAP manifold (`qlvm_umap1`, `qlvm_umap2`).
  Each axis is periodic with period `P` (so the manifold is the
  product of two circles, T^2 = R/P x R/P). Distances must be the
  shortest wrap-aware distance, the centroid must be the circular mean,
  and any algorithm that internally assumes Euclidean geometry (KMeans,
  KDTree) needs the canonical 4-D embedding
  `(x, y) -> (cos(2pi x/P), sin(2pi x/P), cos(2pi y/P), sin(2pi y/P))`.

Single source of truth: every consumer reads
`settings['vocal_features']['usv_manifold_metric']` (`'euclidean'` or
`'torus'`) and `settings['vocal_features']['usv_manifold_period']` (a
positive float; ignored when metric is `'euclidean'`). Pipelines pass
the resolved `(metric, period)` pair through every site that compares
predictions to ground truth.

JAX vs NumPy
------------
The `*_jax` variants use `jax.numpy` so they can live inside the JIT-ed
training loops. The plain variants accept NumPy arrays and are used by
the per-fold metric blocks, the splitter, and the visualisation code.
Both rely on the same wrap rule, so they agree numerically up to
floating-point noise.
"""

import numpy as np
import jax.numpy as jnp


VALID_METRICS = ('euclidean', 'torus')


def _validate_metric_period(metric: str, period: float) -> None:
    """
    Validates that `metric` is a supported tag and that `period` is a
    strictly positive float when the metric is `'torus'`. Called from
    every helper at the top so the failure mode is a clear `ValueError`
    at call time rather than an arithmetic surprise deep inside a JIT
    trace.

    Parameters
    ----------
    metric : str
        Either `'euclidean'` or `'torus'`.
    period : float
        Positive period of each axis (ignored when `metric` is
        `'euclidean'`, but a non-positive value is still rejected so
        the caller cannot silently mis-specify on euclidean runs and
        then flip to torus later).

    Raises
    ------
    ValueError
        On unknown metric tag or non-positive period.
    """

    if metric not in VALID_METRICS:
        raise ValueError(
            f"manifold_metric must be one of {VALID_METRICS}; got {metric!r}"
        )
    if not (np.isfinite(period) and period > 0):
        raise ValueError(
            f"manifold_period must be a positive finite number; got {period!r}"
        )


def signed_diff(a: np.ndarray, b: np.ndarray, *,
                metric: str, period: float) -> np.ndarray:
    """
    Returns the wrap-aware signed difference `a - b` per coordinate.

    For `metric='euclidean'` this is just `a - b`. For `metric='torus'`
    each component is shifted by an integer multiple of `period` so it
    lies in `(-period/2, period/2]`, i.e. the shortest-wrap-direction
    representation. Used everywhere that needs a directional residual:
    loss gradients, regression coefficients, signed bias diagnostics.

    Parameters
    ----------
    a, b : np.ndarray
        Same-shape coordinate arrays. Last axis is the per-coordinate
        axis (typically 2 for `(x, y)` UMAP).
    metric : str
        `'euclidean'` or `'torus'`.
    period : float
        Per-axis wrap period; ignored when `metric == 'euclidean'`.

    Returns
    -------
    np.ndarray
        Same shape as `a - b`, with each component wrapped into
        `(-period/2, period/2]` on torus.
    """

    _validate_metric_period(metric, period)
    diff = a - b
    if metric == 'torus':
        diff = diff - period * np.round(diff / period)
    return diff


def signed_diff_jax(a: jnp.ndarray, b: jnp.ndarray, *,
                    metric: str, period: float) -> jnp.ndarray:
    """
    JAX-friendly mirror of `signed_diff`.

    Used inside JIT-ed loss / gradient routines where `np` would break
    the trace. The wrap behaviour is identical to `signed_diff` and the
    two agree numerically up to floating-point noise.

    No internal validation: `metric` is captured at trace time as a
    Python string (the if-branch is resolved before the JIT compile)
    and `period` is typically a JAX traced scalar inside the regressor's
    JIT path, where `np.isfinite` would raise. Callers must validate
    `metric` and `period` themselves at construction time — the
    regressor's `__init__` does this via `_validate_metric_period` so
    every downstream `signed_diff_jax` call is guaranteed clean inputs.

    Parameters
    ----------
    a, b : jnp.ndarray
        Same-shape coordinate arrays.
    metric : str
        `'euclidean'` or `'torus'`. The argument is captured at trace
        time as a Python string, not a traced value, so the if-branch
        is resolved before the JIT compile.
    period : float or jnp.ndarray
        Per-axis wrap period. May be a traced JAX scalar.

    Returns
    -------
    jnp.ndarray
        Wrap-aware signed difference, same shape as `a - b`.
    """

    diff = a - b
    if metric == 'torus':
        diff = diff - period * jnp.round(diff / period)
    return diff


def pairwise_distance(a: np.ndarray, b: np.ndarray, *,
                      metric: str, period: float) -> np.ndarray:
    """
    Returns the per-row Euclidean norm of the wrap-aware signed diff.

    On euclidean this is just `||a - b||_2` along the last axis. On
    torus each component is wrapped into `(-period/2, period/2]` first,
    then squared and summed; the result is the **shortest** path on
    the torus between `a` and `b`, identical to the user's reference
    `torus_dist` helper.

    Parameters
    ----------
    a, b : np.ndarray
        Coordinate arrays of shape `(..., D)`. Distances are taken
        along the last axis.
    metric : str
        `'euclidean'` or `'torus'`.
    period : float
        Per-axis wrap period.

    Returns
    -------
    np.ndarray
        Distance array of shape `a.shape[:-1]`.
    """

    diff = signed_diff(a, b, metric=metric, period=period)
    return np.sqrt(np.sum(diff ** 2, axis=-1))


def circular_mean(Y: np.ndarray, *,
                  metric: str, period: float,
                  weights: np.ndarray = None) -> np.ndarray:
    """
    Returns the per-axis (weighted) centroid of `Y` in the metric.

    For euclidean this is the standard arithmetic mean (or weighted
    mean) along axis 0. For torus each axis is treated as a circular
    variable: the centroid is the angle whose `(cos, sin)` average
    matches the `(cos, sin)` average of `Y` on that axis. The
    canonical fix-up wraps the result back into `[0, period)`.

    Why circular mean on torus
    --------------------------
    The arithmetic mean of `{0.05, 0.95}` on a unit-period torus is
    `0.5`, but the actual cluster of those two points sits at `0.0`
    (with shortest-wrap distance `0.1` between them). Using the
    arithmetic mean would place the "centroid" on the opposite side
    of the manifold and then claim an average distance of `~0.45` —
    which is wrong by roughly the period.

    Parameters
    ----------
    Y : np.ndarray
        `(N, D)` coordinate matrix; axes are independent torus circles
        on torus, independent real lines on euclidean.
    metric : str
        `'euclidean'` or `'torus'`.
    period : float
        Per-axis wrap period.
    weights : np.ndarray, optional
        Length-`N` non-negative weights. Falls back to uniform weights
        when not supplied. Renormalised to unit sum internally.

    Returns
    -------
    np.ndarray
        `(D,)` centroid on the manifold.
    """

    _validate_metric_period(metric, period)
    Y = np.asarray(Y, dtype=np.float64)
    if weights is None:
        w = np.full(Y.shape[0], 1.0 / max(Y.shape[0], 1), dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)
        s = np.sum(weights)
        if s <= 0:
            raise ValueError("weights must have positive sum")
        w = weights / s

    if metric == 'euclidean':
        return np.sum(w[:, None] * Y, axis=0)

    # Torus: per-axis circular mean. Map to angle, average the unit
    # vector, recover angle from atan2, fold back into [0, period).
    # Handle the negative-branch fold explicitly (rather than via
    # `np.mod`) so a tiny negative `mean_ang` produced by float noise
    # near the wrap boundary doesn't get snapped to the upper edge of
    # the period — `np.mod(-1e-17, 1.0)` evaluates to `1.0`, which
    # downstream is mathematically equivalent to `0.0` but visually
    # confusing.
    angles = (2.0 * np.pi / period) * Y                       # shape (N, D)
    s_avg = np.sum(w[:, None] * np.sin(angles), axis=0)
    c_avg = np.sum(w[:, None] * np.cos(angles), axis=0)
    mean_ang = np.arctan2(s_avg, c_avg)                       # in (-pi, pi]
    mean_xy = (period / (2.0 * np.pi)) * mean_ang
    return np.where(mean_xy < 0, mean_xy + period, mean_xy)


def total_dispersion(Y: np.ndarray, *,
                     metric: str, period: float,
                     weights: np.ndarray = None) -> float:
    """
    Returns the total squared dispersion of `Y` around its (metric-
    aware) centroid: `sum_i ||Y_i - centroid||_2 ^ 2`.

    Used as the denominator of `r2_spatial`, where the numerator is
    the sum of squared wrap-aware residuals between predictions and
    truth. Computing both numerator and denominator under the same
    metric is what keeps `r2_spatial` interpretable on a torus —
    using `np.var(Y)` against a sum-of-squared-wrap-residuals would
    give nonsense near the boundary.

    Parameters
    ----------
    Y : np.ndarray
        `(N, D)` coordinate matrix.
    metric : str
        `'euclidean'` or `'torus'`.
    period : float
        Per-axis wrap period.
    weights : np.ndarray, optional
        Length-`N` weights. Defaults to uniform.

    Returns
    -------
    float
        `sum_i w_i * ||Y_i - centroid||_2 ^ 2` (under the metric).
        Note the `w_i` are renormalised to unit sum inside, so when
        called with uniform weights this reduces to the unweighted
        sum of squared distances times `1/N`. The R^2 numerator and
        denominator must therefore be computed with the **same**
        weight convention; the consumers in this project pass
        `weights=None` on both sides, so the `1/N` cancels.
    """

    _validate_metric_period(metric, period)
    Y = np.asarray(Y, dtype=np.float64)
    centroid = circular_mean(Y, metric=metric, period=period, weights=weights)
    diff = signed_diff(Y, centroid[None, :], metric=metric, period=period)
    if weights is None:
        return float(np.sum(diff ** 2))
    weights = np.asarray(weights, dtype=np.float64)
    s = np.sum(weights)
    if s <= 0:
        raise ValueError("weights must have positive sum")
    w = weights / s
    return float(np.sum(w[:, None] * diff ** 2) * len(Y))  # scale to match unweighted convention


def torus_embed(Y: np.ndarray, period: float) -> np.ndarray:
    """
    Maps torus coordinates `(x, y) in [0, period)^D` into the
    canonical `2D`-dimensional Euclidean embedding via
    `(x, y) -> (cos(2pi x/P), sin(2pi x/P), cos(2pi y/P), sin(2pi y/P))`.

    The embedding is an isometric immersion of the torus into R^{2D}
    (up to a constant scale factor): Euclidean distance in the embedded
    space is a strictly monotone function of the toroidal distance, so
    Euclidean-only algorithms (KMeans, KDTree) operate "as if" they
    were aware of the wrap.

    Bandwidth note
    --------------
    A bandwidth `h_torus` in original-space units corresponds to
    `h_4D = 2 sin(pi h_torus / P)` in the embedded space (4D-Euclidean).
    For small `h_torus / P` this is approximately
    `(2 pi / P) * h_torus`. Callers picking bandwidths by
    cross-validation should re-pick on the embedded scale; callers
    setting bandwidth from a torus-space target should apply the
    above conversion.

    Parameters
    ----------
    Y : np.ndarray
        `(N, D)` coordinates on the torus; values may lie outside
        `[0, period)`, the trig functions wrap automatically.
    period : float
        Per-axis wrap period.

    Returns
    -------
    np.ndarray
        `(N, 2*D)` embedded coordinates, dtype `float64`.
    """

    if not (np.isfinite(period) and period > 0):
        raise ValueError(f"period must be positive finite; got {period!r}")
    Y = np.asarray(Y, dtype=np.float64)
    angles = (2.0 * np.pi / period) * Y
    return np.concatenate([np.cos(angles), np.sin(angles)], axis=-1)


def resolve_manifold_metric(modeling_settings: dict) -> tuple:
    """
    Reads `(metric, period)` out of a modeling-settings dictionary.

    Looks up `vocal_features.usv_manifold_metric` and
    `vocal_features.usv_manifold_period` directly (no `.get()` defaults,
    matching the project convention for strict settings access). Both
    keys must be present and valid; a missing key is treated as a
    settings-file bug, not a soft-fallback opportunity.

    Parameters
    ----------
    modeling_settings : dict
        Fully loaded `modeling_settings.json`.

    Returns
    -------
    tuple
        `(metric: str, period: float)` ready to forward into any of
        the helpers above.
    """

    voc = modeling_settings['vocal_features']
    metric = voc['usv_manifold_metric']
    period = float(voc['usv_manifold_period'])
    _validate_metric_period(metric, period)
    return metric, period
