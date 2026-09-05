"""
@author: bartulem
Circular-shift nulls and tail-extrapolated p-values for the neural encoding tests.

The null throughout is a circular shift of the unit's spike train: it preserves firing rate, burstiness and
slow within-session drift while destroying the alignment between spikes and behaviour. Two properties of
this project make the details matter more than usual.

The shift range is the full wrap minus a guard band, not a narrow window. Every shift beyond the
decorrelation time is individually a valid draw, which was never the issue. But the null distribution is
estimated from the draws, and two shifts closer together than the statistic's decorrelation lag give
near-identical statistics. Since the predictor is built from a 4 s history of slow behavioural variables,
that lag is at least 4 s and plausibly tens of seconds, so a fixed 40 s window carried only about ten
effectively independent draws regardless of how many were taken. That capped the usable p at roughly
1 / (N_eff + 1), around 0.09, left a two-parameter tail fit about two points to work with, and gave the
null spread a relative error near 22% that every reported z divides by. Over a 1,200 s session the full
range yields roughly 290 draws at identical runtime.

Models are frozen rather than refitted per draw. This is legitimate because every statistic here is
held-out: the model was fitted on other sessions and never saw the train being shuffled, so conditioning on
it is valid. Under the null hypothesis the fitted filter is itself arbitrary, and the question each draw
asks, given this filter is its alignment with the spike train better than chance, is the right one for a
held-out score. It is also what makes a thousand draws cost seconds instead of days.

P-values are empirical with escalation, never tail-extrapolated. The number of draws is set by the bar the
test in question has to clear, not by a global constant: a screen judged against a Bonferroni threshold of
alpha / n_features needs a floor below that, while a p-value entering Benjamini-Hochberg over a
2,525-unit cohort needs far more. Since an empirical permutation p floors at 1 / (n + 1), the answer is
more draws rather than a fitted tail.

Escalation climbs a ladder and accumulates rather than restarting, and it is driven by the COHORT, not by
the unit. BH rejects the k-th smallest p when p_(k) <= k * q / m, so a p-value on the floor is rejectable
once k reaches m / (q * (n + 1)) -- 252 units at a thousand draws, 25 at ten thousand, 3 at a hundred
thousand. Which rung is needed therefore depends on how many units are tied at the floor, which is only
knowable after the whole cohort has run. The shape is: run everything at the base count, count the
floor-tied, compute the requirement with :func:`bh_floor_requirement`, and draw more only where it still
binds. Measured cost of the top rung, at the sizes claim 1 uses: about fifteen minutes per unit.

Escalation is only meaningful for a POOLED statistic, and that is a measured constraint rather than a
stylistic one. The null statistic is smooth in the shift lag: its autocorrelation is 0.997 at a lag
difference of 0.1 s and reaches 1/e only at 2.2 s. A single session therefore carries about 528 effectively
independent shifts out of 174,000 possible frame lags, so a per-session statistic floors near 1/529 and no
number of draws improves it. Pooling shifts every session independently, giving a joint space of
528^n_sessions -- about 1.5e8 for three sessions -- against which a hundred thousand draws are all
effectively distinct.
"""

from __future__ import annotations

import numpy as np


def shift_range_seconds(duration: float, guard_seconds: float) -> tuple[float, float]:
    """
    Description
    -----------
    The circular-shift lag range ``(guard, duration - guard)``, the full wrap-around minus a symmetric guard
    band. This is the one definition every shift null in the project draws from.

    The guard excludes lags where the wrapped train is nearly back in register. Large lags cross slow
    within-session rate drift, which per-session intercept recalibration absorbs, so the effect is to
    inflate null variance slightly, which is conservative, rather than to bias it.

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
        raise ValueError(f"session too short ({duration:.1f} s) for a {guard_seconds} s guard band")
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

    An empirical p cannot go below ``1 / (n + 1)``, so a unit whose observed value beats every draw is
    reported at that floor and nothing distinguishes it from a unit a thousand times more extreme. At
    1,000 draws the floor is 9.99e-4, while BH across a 2,525-unit cohort needs about 2e-5 for the
    top-ranked unit -- so the floor, not the data, would decide the cohort.

    The escalation is by DECADE and it ACCUMULATES: a run that floors at 1,000 draws is topped up to
    10,000, then to 100,000, then to 1,000,000, keeping the draws already taken rather than discarding
    them. Accumulating is valid here because the draws are exchangeable and the decision to continue
    depends only on whether the floor was reached, never on where the observed value sits among the new
    draws. It stops as soon as the p lifts off the floor, so only genuinely extreme units pay for the
    deep ladder.

    Resolution is bought with draws and never with a parametric tail. A GPD fit was used earlier and is
    dropped project-wide: it silently fell back to the floor when the observed exceeded its fitted
    support, and in the gating pilot it extrapolated 3-sigma borderline features into false positives.

    Parameters
    ----------
    observed (float)
        The observed statistic.
    draw_more (Callable)
        ``draw_more(n) -> np.ndarray`` returning ``n`` FRESH null scores.
    ladder (list)
        Ascending target draw counts, e.g. ``[10000, 100000, 1000000]``.
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
        if not at_floor or target <= null.size:
            break
        message_output(f"      escalating {null.size} -> {target} draws (p at the {p_value:.3e} floor)")
        null = np.concatenate([null, np.asarray(draw_more(target - null.size), dtype=np.float64)])
        p_value, at_floor = empirical_pvalue(null, observed)
    return p_value, at_floor, null

def bh_floor_requirement(n_tests: int, fdr_q: float, n_shuffles: int) -> float:
    """
    Description
    -----------
    The number of units that must be tied at the empirical p-value floor before Benjamini-Hochberg can
    reject any of them.

    BH rejects the k-th smallest p-value when ``p_(k) <= k * q / m``, while an empirical permutation p
    cannot fall below ``1 / (n + 1)``. A unit sitting on that floor is therefore only rejectable once
    ``k >= m / (q * (n + 1))``. Reporting this per run makes the constraint visible instead of leaving a
    whole cohort silently unrejectable: for 2,525 units at q = 0.01 with 1,000 shuffles it takes 253
    floor-tied units before any of them passes.

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

    No parametric tail is fitted. The alternative -- extrapolating a Generalized Pareto beyond the largest
    observed draw -- buys resolution at the cost of a distributional assumption about a region where, by
    construction, there is no data. Since the frozen nulls are cheap, taking more draws is both simpler and
    assumption-free.

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
