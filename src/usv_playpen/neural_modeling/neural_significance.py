"""
@author: bartulem
Cohort-level multiplicity control and the exact within-session permutation null.

Shared by every analysis:

- `benjamini_hochberg`          — step-up FDR control over a cohort of p-values; returns both the
                                   rejection mask and the threshold.
- `within_session_permutation`  — one exact permutation of events within each session.
- `permutation_null`            — the null distribution of a statistic under that permutation, refit
                                   inside every draw.

The permutation null destroys only the pairing of spike counts with a per-USV target, leaving the
counts and the spike train in place; it is used by the identity tests instead of the circular shift
of ``shift_null_inference``, which would also destroy the alignment of spikes with vocal onsets.
Permutation is always within session, never across.
"""

from __future__ import annotations

import numpy as np


def benjamini_hochberg(p_values: np.ndarray, q: float) -> tuple[np.ndarray, float, int]:
    """
    Description
    -----------
    Benjamini-Hochberg step-up procedure over a cohort of p-values.

    Rejects the k-th smallest p when ``p_(k) <= k * q / m``, taking the largest such k, and rejects
    everything ranked below it. NaN p-values are carried through as non-rejections rather than dropped,
    so a unit that could not be tested never silently changes the multiplicity for the units that could.

    An empirical p cannot fall below ``1 / (n_draws + 1)``, so units tied at that floor are rejectable
    only when ``k >= m / (q * (n_draws + 1))``.

    Parameters
    ----------
    p_values (np.ndarray)
        One p-value per unit. NaN marks a unit that could not be tested.
    q (float)
        Target false-discovery rate.

    Returns
    -------
    rejected (np.ndarray)
        Boolean mask, True where the null is rejected.
    threshold (float)
        The largest p-value rejected, or 0.0 when nothing is.
    n_rejected (int)
        How many were rejected.
    """

    values = np.asarray(p_values, dtype=np.float64)
    rejected = np.zeros(values.size, dtype=bool)
    testable = np.flatnonzero(np.isfinite(values))
    if testable.size == 0:
        return rejected, 0.0, 0

    order = testable[np.argsort(values[testable])]
    ranked = values[order]
    m = ranked.size
    below = np.flatnonzero(ranked <= (np.arange(1, m + 1) / m) * q)
    if below.size == 0:
        return rejected, 0.0, 0

    cutoff = int(below.max())
    rejected[order[:cutoff + 1]] = True
    return rejected, float(ranked[cutoff]), int(cutoff + 1)


def within_session_permutation(session_index: np.ndarray, rng) -> np.ndarray:
    """
    Description
    -----------
    Draw one exact within-session permutation of the events.

    Returns a reordering of the event indices in which every event is replaced by another event FROM THE
    SAME SESSION. Applying it to the target (torus position, or a category label) while leaving the spike
    counts in place destroys the count-to-target pairing and nothing else.

    Sessions are permuted independently and a session of one event is a fixed point, which is correct
    rather than a special case: such a session carries no pairing to destroy and contributes the same
    value to the observed statistic and to every draw.

    Parameters
    ----------
    session_index (np.ndarray)
        Session id per event, as integers. Events sharing a value are exchangeable with one another.
    rng (np.random.Generator)
        Seeded generator.

    Returns
    -------
    permutation (np.ndarray)
        Index array of the same length; ``target[permutation]`` is one permuted draw.
    """

    sessions = np.asarray(session_index)
    permutation = np.arange(sessions.size)
    for value in np.unique(sessions):
        members = np.flatnonzero(sessions == value)
        permutation[members] = members[rng.permutation(members.size)]
    return permutation


def permutation_null(statistic, session_index: np.ndarray, n_draws: int, seed: int) -> np.ndarray:
    """
    Description
    -----------
    Null distribution of a statistic under exact within-session permutation of the target.

    The statistic is handed a permutation and returns one number, so everything about the model --
    ridge, basis, amplitude calibration, the leave-one-session-out rotation -- is re-run inside every
    draw exactly as it was for the observed value. That is what keeps the test exact: any advantage the
    fitting procedure confers is conferred on the null too.

    Parameters
    ----------
    statistic (Callable)
        ``statistic(permutation) -> float``.
    session_index (np.ndarray)
        Session id per event.
    n_draws (int)
        Number of permutations.
    seed (int)
        Base seed; draw ``i`` uses ``seed + i`` so a run is reproducible and resumable.

    Returns
    -------
    null (np.ndarray)
        ``n_draws`` null values.
    """

    null = np.empty(n_draws, dtype=np.float64)
    for draw in range(n_draws):
        generator = np.random.default_rng(seed + draw)
        null[draw] = statistic(within_session_permutation(session_index, generator))
    return null
