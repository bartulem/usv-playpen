"""
@author: bartulem
Unit tests for ``usv_playpen.neural_modeling.shift_null_inference``.

Coverage: the circular-shift range, the empirical p and its floor, the BH floor requirement, and the
escalation ladder. The ladder is the piece that decides whether a cohort's strongest units can clear
BH at all, so its accumulate-and-stop-early behaviour is asserted directly rather than assumed.
"""

from __future__ import annotations

import numpy as np
import pytest

from usv_playpen.neural_modeling.shift_null_inference import (
    bh_floor_requirement,
    empirical_pvalue,
    escalated_empirical_pvalue,
    sample_circular_shift,
    shift_range_seconds,
    shifted_spike_frames,
)


class TestShiftRange:

    def test_range_is_the_full_wrap_minus_a_guard_at_both_ends(self):
        """A narrow window holds few INDEPENDENT draws however many are nominally taken, which is why
        the range is the whole session rather than a fixed 20-60 s band."""
        low, high = shift_range_seconds(1200.0, guard_seconds=20.0)
        assert low == pytest.approx(20.0)
        assert high == pytest.approx(1180.0)

    def test_a_session_too_short_for_the_guard_raises(self):
        with pytest.raises(ValueError, match="too short"):
            shift_range_seconds(30.0, guard_seconds=20.0)

    def test_sampled_shifts_respect_the_guard(self):
        generator = np.random.default_rng(0)
        fps, n_frames = 150.0, 180_000
        # the sampler returns SECONDS, which is what `shifted_spike_frames` takes
        seconds = np.asarray([sample_circular_shift(generator, n_frames, fps, 20.0)
                              for _ in range(200)])

        assert seconds.min() >= 20.0
        assert seconds.max() <= n_frames / fps - 20.0

    def test_shifting_preserves_the_spike_count(self):
        """A circular shift moves the train without creating or destroying spikes -- that is what makes
        it preserve rate, burstiness and slow drift while breaking alignment to behaviour."""
        spikes = np.sort(np.random.default_rng(1).choice(50_000, 3_000, replace=False))
        shifted = shifted_spike_frames(spikes, 51.85, 150.0, 50_000)   # seconds

        assert shifted.size == spikes.size
        assert shifted.min() >= 0
        assert shifted.max() < 50_000
        assert np.all(np.diff(shifted) >= 0)


class TestEmpiricalPvalue:

    def test_p_floors_at_one_over_n_plus_one(self):
        null = np.random.default_rng(2).normal(size=999)
        p_value, at_floor = empirical_pvalue(null, 50.0)

        assert p_value == pytest.approx(1.0 / 1000.0)
        assert at_floor

    def test_an_ordinary_observation_is_not_at_the_floor(self):
        null = np.random.default_rng(3).normal(size=999)
        p_value, at_floor = empirical_pvalue(null, 0.0)

        assert 0.3 < p_value < 0.7
        assert not at_floor


class TestBhFloorRequirement:

    def test_the_requirement_falls_as_draws_are_added(self):
        """This is the arithmetic that motivates the ladder: at 1,000 draws a 2,525-unit cohort needs
        253 units tied at the floor before BH can reject any of them."""
        shallow = bh_floor_requirement(2525, 0.01, 1000)
        deep = bh_floor_requirement(2525, 0.01, 100_000)

        assert shallow > 250.0
        assert deep < 3.0


class TestEscalationLadder:

    def test_it_accumulates_and_stops_when_the_p_lifts_off_the_floor(self):
        generator = np.random.default_rng(4)
        calls = []

        def draw_more(count):
            calls.append(count)
            return generator.normal(size=count)

        initial = generator.normal(size=1000)
        p_value, at_floor, null = escalated_empirical_pvalue(
            5.0, draw_more, [10_000, 100_000], initial, message_output=lambda _m: None)

        assert calls == [9_000, 90_000]          # tops up, never re-draws from scratch
        assert null.size == 100_000
        assert p_value == pytest.approx(1.0 / 100_001.0)
        assert at_floor

    def test_an_unremarkable_observation_never_escalates(self):
        generator = np.random.default_rng(5)
        calls = []

        def draw_more(count):
            calls.append(count)
            return generator.normal(size=count)

        p_value, at_floor, null = escalated_empirical_pvalue(
            0.0, draw_more, [10_000, 100_000], generator.normal(size=1000),
            message_output=lambda _m: None)

        assert calls == []
        assert null.size == 1000
        assert not at_floor
        assert 0.3 < p_value < 0.7
