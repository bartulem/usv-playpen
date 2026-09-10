"""
@author: bartulem
Cohort-level multiplicity control and the exact within-session permutation null.

Two things live here, both shared by every claim rather than reimplemented per analysis.

Benjamini-Hochberg existed in seven near-identical copies across the prototype scripts, in two
incompatible shapes -- some returning a discovery count and a threshold, others a boolean mask -- which
is how a cohort ends up with two different answers to the same question. One implementation returns
both.

The permutation null is what the claim-2 WHAT axis tests against, and it is a DIFFERENT null from the
circular shift used by claim 1, claim 3, the gating analysis and the claim-2 WHEN axis. The distinction
is not stylistic. Shifting a spike train destroys the alignment between spikes and vocal onsets IN
ADDITION to the pairing under test, so a strongly onset-responsive unit gets a null centred far too low:
52.5% of position-permuted control units exceeded |z| > 2 under the shift, and 10% reached the 0/1000
floor. Permuting the TARGET across events within a session destroys only the count-to-position pairing,
leaving counts, the spike train and the bout envelope untouched, and is exact by exchangeability for any
statistic computed from them.

Permutation is within SESSION, never across. Sessions differ in rate, in behavioural state and in
repertoire, so a global permutation would manufacture structure by mixing them and would be testing a
different, easier null.
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

    Note the floor interaction that governs every empirical p in this project: a p cannot fall below
    ``1 / (n_draws + 1)``, so units tied at that floor are rejectable only when
    ``k >= m / (q * (n_draws + 1))``. For a 2,525-unit cohort at ``q = 0.01`` that is 253 units at 1,000
    draws but under 3 at 100,000, which is what the escalation ladder in ``shift_null_inference`` exists
    to deliver.

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
