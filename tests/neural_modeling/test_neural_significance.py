"""
@author: bartulem
Unit tests for ``usv_playpen.neural_modeling.neural_significance``.

Coverage: the shared Benjamini-Hochberg, and the exact within-session permutation. The permutation's
defining property -- events are exchanged only with events of the SAME session -- is asserted directly,
since permuting across sessions would mix rate and repertoire differences and test an easier null.
"""

from __future__ import annotations

import numpy as np
import pytest

from usv_playpen.neural_modeling.neural_significance import (
    benjamini_hochberg,
    permutation_null,
    within_session_permutation,
)


class TestBenjaminiHochberg:

    def test_step_up_rejects_everything_below_the_largest_passing_rank(self):
        """BH is a step-UP procedure: once the largest passing rank is found, every smaller p goes with
        it, including any that individually fail their own threshold."""
        p_values = np.array([0.001, 0.008, 0.030, 0.600])
        rejected, threshold, count = benjamini_hochberg(p_values, 0.05)

        assert rejected.tolist() == [True, True, True, False]
        assert threshold == pytest.approx(0.030)
        assert count == 3

    def test_nothing_passes_when_every_p_is_large(self):
        rejected, threshold, count = benjamini_hochberg(np.array([0.4, 0.6, 0.9]), 0.01)

        assert not rejected.any()
        assert threshold == 0.0
        assert count == 0

    def test_nan_units_are_non_rejections_and_do_not_change_the_multiplicity(self):
        """A unit that could not be tested must not make its neighbours easier or harder to reject."""
        without = benjamini_hochberg(np.array([0.001, 0.008, 0.030]), 0.05)
        with_nan = benjamini_hochberg(np.array([0.001, np.nan, 0.008, 0.030, np.nan]), 0.05)

        assert with_nan[2] == without[2]
        assert with_nan[1] == pytest.approx(without[1])
        assert with_nan[0].tolist() == [True, False, True, True, False]

    def test_all_nan_is_handled(self):
        rejected, threshold, count = benjamini_hochberg(np.array([np.nan, np.nan]), 0.05)
        assert not rejected.any()
        assert threshold == 0.0
        assert count == 0

    def test_the_floor_requirement_is_about_HOW_MANY_units_are_tied(self):
        """The arithmetic the escalation ladder exists for, stated correctly. A unit sitting at the
        empirical floor 1/(n+1) is rejectable only at a rank whose threshold k*q/m reaches it, so at
        least ``m / (q * (n + 1))`` units must be tied there. For 2,525 units at q = 0.01 that is 253
        at 1,000 draws -- and under 3 at 100,000, which is why the ladder lifts the constraint.

        Note it is NOT true that a floor of 9.99e-4 blocks rejection outright: if every unit is tied,
        rank m carries threshold q itself and they all pass. The floor binds when the tied group is
        SMALL relative to the cohort."""
        m, q, floor = 2525, 0.01, 1.0 / 1001.0
        required = m / (q * 1001.0)
        assert required == pytest.approx(252.2, abs=0.1)

        def cohort(n_tied):
            values = np.full(m, 0.9)
            values[:n_tied] = floor
            return benjamini_hochberg(values, q)[2]

        assert cohort(200) == 0                  # below the requirement: none reject
        assert cohort(300) == 300                # above it: the whole tied group rejects

        deeper = 1.0 / 100_001.0
        assert m / (q * 100_001.0) < 3.0         # the ladder makes the requirement trivial
        values = np.full(m, 0.9)
        values[:3] = deeper
        assert benjamini_hochberg(values, q)[2] == 3


class TestWithinSessionPermutation:

    def test_events_are_exchanged_only_within_their_own_session(self):
        sessions = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2])
        generator = np.random.default_rng(0)

        for _ in range(50):
            permutation = within_session_permutation(sessions, generator)
            assert np.array_equal(sessions[permutation], sessions)      # session labels never move
            assert np.array_equal(np.sort(permutation), np.arange(sessions.size))

    def test_a_single_event_session_is_a_fixed_point(self):
        """Correct rather than a special case: such a session carries no pairing to destroy."""
        sessions = np.array([0, 1, 1, 1])
        permutation = within_session_permutation(sessions, np.random.default_rng(1))

        assert permutation[0] == 0

    def test_it_actually_permutes(self):
        sessions = np.zeros(200, dtype=int)
        generator = np.random.default_rng(2)
        moved = [np.mean(within_session_permutation(sessions, generator) != np.arange(200))
                 for _ in range(20)]

        assert np.mean(moved) > 0.9


class TestPermutationNull:

    def test_the_null_is_reproducible_and_the_statistic_sees_every_draw(self):
        sessions = np.array([0, 0, 0, 1, 1, 1])
        seen = []

        def statistic(permutation):
            seen.append(permutation.copy())
            return float(permutation[0])

        first = permutation_null(statistic, sessions, 25, seed=7)
        second = permutation_null(statistic, sessions, 25, seed=7)

        assert first.size == 25
        assert np.array_equal(first, second)
        assert len(seen) == 50

    def test_a_target_carried_through_the_permutation_stays_within_session(self):
        """The end-to-end property the WHAT null relies on: positions move, sessions do not."""
        sessions = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        target = np.arange(8, dtype=float)
        generator = np.random.default_rng(3)

        permuted = target[within_session_permutation(sessions, generator)]
        assert set(permuted[:4]) <= set(target[:4])
        assert set(permuted[4:]) <= set(target[4:])
