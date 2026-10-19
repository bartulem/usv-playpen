"""
@author: bartulem
Circular-shift nulls and empirical p-values for the neural encoding tests.

The null is a circular shift of the unit's spike train, which keeps its firing rate, burstiness and
slow drift and destroys only its alignment with behaviour and vocalizations. Lags are drawn from the
full wrap minus a guard band, and held-out models are frozen across draws rather than refitted.

- `shift_range_seconds`         — the lag range ``(guard, duration - guard)``.
- `sample_circular_shift`       — one lag drawn uniformly over that range.
- `shifted_spike_frames`        — a spike-frame train shifted and wrapped modulo the session length.
- `empirical_pvalue`            — the one-sided empirical p and whether it sits on its floor.
- `escalated_empirical_pvalue`  — tops a floored p up with more draws along a ladder; the shipped
                                   ladder is empty, so no p is escalated by default.
- `bh_floor_requirement`        — how many floor-tied units Benjamini-Hochberg needs before it can
                                   reject any of them.
"""

from __future__ import annotations

import numpy as np


def shift_range_seconds(duration: float, guard_seconds: float) -> tuple[float, float]:
    """
    Description
    -----------
    The circular-shift lag range ``(guard, duration - guard)``, the full wrap-around minus a symmetric guard
    band. This is the one definition every shift null in the project draws from.

    The guard excludes lags at which the wrapped train is nearly back in register.

    Parameters
    ----------
    duration (float)
        Session duration in seconds.
    guard_seconds (float)
        Excluded band at both ends of the wrap.

    Returns
    -------
    lo, hi (tuple)
        Inclusive lag bounds in seconds.
    """

    hi = duration - guard_seconds
    if hi <= guard_seconds:
        msg = f"session too short ({duration:.1f} s) for a {guard_seconds} s guard band"
        raise ValueError(msg)
    return guard_seconds, hi


def sample_circular_shift(rng, n_frames: int, fps: float, guard_seconds: float) -> float:
    """
    Description
    -----------
    Draw one circular-shift lag in seconds, uniformly over :func:`shift_range_seconds`.

    Parameters
    ----------
    rng (np.random.Generator)
        Seeded generator.
    n_frames (int)
        Session frame count.
    fps (float)
        Camera frame rate.
    guard_seconds (float)
        Excluded band at both ends of the wrap.

    Returns
    -------
    shift_seconds (float)
        Lag in seconds.
    """

    return float(rng.uniform(*shift_range_seconds(n_frames / fps, guard_seconds)))


def shifted_spike_frames(spike_frames: np.ndarray, shift_seconds: float, fps: float,
                         n_frames: int) -> np.ndarray:
    """
    Description
    -----------
    Apply a circular shift to a spike-frame train, wrapping modulo the session length.

    Parameters
    ----------
    spike_frames (np.ndarray)
        Integer spike-frame indices.
    shift_seconds (float)
        Lag in seconds.
    fps (float)
        Camera frame rate.
    n_frames (int)
        Session frame count.

    Returns
    -------
    shifted (np.ndarray)
        Sorted, wrapped spike frames.
    """

    return np.sort((spike_frames + shift_seconds * fps) % n_frames).astype(np.int64)


def escalated_empirical_pvalue(observed: float, draw_more, ladder: list, initial_null: np.ndarray,
                               message_output=print) -> tuple[float, bool, np.ndarray]:
    """
    Description
    -----------
    Resolve an empirical p that has bottomed out, by drawing more shuffles rather than extrapolating.

    An empirical p cannot go below ``1 / (n + 1)``, so a unit that beats every draw is indistinguishable
    from one far more extreme. Each rung of the ladder tops the null up to its target count, keeping the
    draws already taken; this is valid because the decision to continue depends only on whether the
    floor was reached. It stops as soon as the p lifts off the floor. No parametric tail is fitted.

    Parameters
    ----------
    observed (float)
        The observed statistic.
    draw_more (Callable)
        ``draw_more(n) -> np.ndarray`` returning ``n`` FRESH null scores.
    ladder (list)
        Ascending target counts of FINITE draws, e.g. ``[10000, 100000, 1000000]``; NaN draws are kept in
        the returned null for auditing but count toward no target. Each rung tops up once, by the
        shortfall in finite draws.
    initial_null (np.ndarray)
        Null scores already in hand.
    message_output (Callable)
        Where escalation steps are reported.

    Returns
    -------
    p_value (float)
        The empirical p at the deepest level reached.
    at_floor (bool)
        Whether it is STILL at the floor after the whole ladder.
    null (np.ndarray)
        Every null score drawn, so the escalation is auditable.
    """

    null = np.asarray(initial_null, dtype=np.float64)
    p_value, at_floor = empirical_pvalue(null, observed)
    for target in ladder:
        # Only finite draws count toward a rung's target, as only they enter the p-value.
        n_finite = int(np.isfinite(null).sum())
        if not at_floor or target <= n_finite:
            break
        message_output(f"      escalating {n_finite} -> {target} finite draws (p at the {p_value:.3e} floor)")
        null = np.concatenate([null, np.asarray(draw_more(target - n_finite), dtype=np.float64)])
        p_value, at_floor = empirical_pvalue(null, observed)
    return p_value, at_floor, null

def bh_floor_requirement(n_tests: int, fdr_q: float, n_shuffles: int) -> float:
    """
    Description
    -----------
    The number of units that must be tied at the empirical p-value floor before Benjamini-Hochberg can
    reject any of them.

    BH rejects the k-th smallest p-value when ``p_(k) <= k * q / m``, while an empirical permutation p
    cannot fall below ``1 / (n + 1)``, so a floor-tied unit is rejectable only once
    ``k >= m / (q * (n + 1))``.

    Parameters
    ----------
    n_tests (int)
        Number of units entering the FDR correction.
    fdr_q (float)
        Target false-discovery rate.
    n_shuffles (int)
        Draws behind each empirical p-value.

    Returns
    -------
    k_floor (float)
        Smallest rank at which a floor-tied p-value becomes rejectable.
    """

    return n_tests / (fdr_q * (n_shuffles + 1.0))


def empirical_pvalue(null_values: np.ndarray, observed: float) -> tuple[float, bool]:
    """
    Description
    -----------
    The empirical one-sided permutation p-value ``(exceedances + 1) / (draws + 1)``, plus whether it is
    sitting on the floor and therefore a candidate for escalation.

    No parametric tail is fitted beyond the largest draw.

    Parameters
    ----------
    null_values (np.ndarray)
        Null distribution of the statistic; higher is more extreme.
    observed (float)
        The observed statistic.

    Returns
    -------
    p (float)
        The empirical p-value.
    at_floor (bool)
        True when no draw reached the observed value, so the p-value is the floor and more draws would
        resolve it further.
    """

    null = np.asarray(null_values, dtype=np.float64)
    null = null[np.isfinite(null)]
    if null.size == 0 or not np.isfinite(observed):
        return np.nan, False
    n_exceed = int(np.sum(null >= observed))
    return (n_exceed + 1.0) / (null.size + 1.0), n_exceed == 0
