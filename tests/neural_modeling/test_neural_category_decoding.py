"""
@author: bartulem

Coverage: the categorical decoding target -- code mapping, per-category rates, the posterior, the
leave-one-session-out context, the amplitude calibration and the gain's key roster.
"""
from __future__ import annotations

import numpy as np
import pytest

from usv_playpen.neural_modeling.neural_category_decoding import (category_codes, category_context,
                                                                  category_gain, category_posterior,
                                                                  category_rates,
                                                                  decoding_spike_sufficiency,
                                                                  tuned_category_amplitude)

SETTINGS = {"category_surface": {"ridge_selection": {"mode": "count_normalized_fixed",
                                                     "ridge_frac": 0.02},
                                 "amplitude_calibration": {"enabled": True,
                                                           "mode": "train_tuned_inner_loso",
                                                           "alpha_grid": [0.1, 0.25, 0.5, 1.0]},
                                 "category_column": "qlvm_category"}}


def _synthetic(n_sessions=3, per_session=200, n_categories=4, strength=3.0, seed=0):
    """A unit whose rate depends on category, with the strength of that dependence controllable."""
    rng = np.random.default_rng(seed)
    labels, counts, session = [], [], []
    rates = np.linspace(1.0, 1.0 + strength, n_categories)
    for s in range(n_sessions):
        c = rng.integers(0, n_categories, per_session)
        labels.append(c)
        counts.append(rng.poisson(rates[c]))
        session.append(np.full(per_session, s))
    return (np.concatenate(counts).astype(float), np.concatenate(labels),
            np.concatenate(session))


class TestCategoryCodes:
    def test_labels_are_made_contiguous(self):
        codes, kept, values = category_codes(np.array([1.0, 4.0, 4.0, 2.0]))
        assert codes.tolist() == [0, 2, 2, 1]
        assert values.tolist() == [1.0, 2.0, 4.0]
        assert kept.all()

    def test_unlabelled_events_are_dropped_not_renumbered(self):
        """A NaN label must remove its event, never shift the codes of the events after it."""
        codes, kept, values = category_codes(np.array([1.0, np.nan, 3.0]))
        assert kept.tolist() == [True, False, True]
        assert codes.tolist() == [0, 1]
        assert values.tolist() == [1.0, 3.0]


class TestRates:
    def test_rates_recover_the_generating_values(self):
        counts, labels, _ = _synthetic(n_sessions=1, per_session=4000, strength=3.0)
        rates = category_rates(counts, labels, 4, ridge=0.0)
        assert np.allclose(rates, np.linspace(1.0, 4.0, 4), atol=0.15)

    def test_an_absent_category_falls_back_to_the_pooled_rate(self):
        """Falling back to zero would send any event of that category to -inf log-likelihood."""
        counts = np.array([2.0, 3.0, 2.0, 3.0])
        rates = category_rates(counts, np.array([0, 0, 1, 1]), 3, ridge=0.0)
        assert rates[2] == pytest.approx(counts.mean())
        assert (rates > 0).all()

    def test_the_ridge_shrinks_toward_the_pooled_rate(self):
        counts = np.array([10.0, 10.0, 0.0, 0.0])
        labels = np.array([0, 0, 1, 1])
        loose = category_rates(counts, labels, 2, ridge=0.0)
        tight = category_rates(counts, labels, 2, ridge=100.0)
        assert loose[0] > tight[0] > counts.mean() * 0.9
        assert loose[1] < tight[1]


class TestPosterior:
    def test_alpha_zero_collapses_to_the_prior(self):
        """With no amplitude the categories share a rate, so the count carries nothing."""
        prior = np.array([0.5, 0.3, 0.2])
        post = category_posterior(np.array([5.0]), np.array([1.0, 5.0, 20.0]), prior, 0.0, 1.0)
        assert np.allclose(np.exp(post[0]), prior, atol=1e-9)

    def test_alpha_cannot_invert_the_profile(self):
        """It may temper a profile but never reverse its ordering -- the one-sided property."""
        rates = np.array([1.0, 5.0])
        for alpha in (0.1, 0.5, 1.0, 1.5):
            post = category_posterior(np.array([8.0]), rates, np.array([0.5, 0.5]), alpha, 1.0)
            assert post[0, 1] > post[0, 0]

    def test_rows_are_normalised(self):
        post = category_posterior(np.arange(1.0, 6.0), np.array([1.0, 2.0, 3.0]),
                                  np.array([0.2, 0.3, 0.5]), 1.0, 1.0)
        assert np.allclose(np.exp(post).sum(axis=1), 1.0)


class TestContext:
    def test_one_session_is_refused(self):
        with pytest.raises(ValueError, match="at least two sessions"):
            category_context(np.array([0, 1, 0]), np.zeros(3, dtype=int))

    def test_the_prior_uses_training_sessions_only(self):
        """A prior built from all events would leak the held-out session's label distribution."""
        labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        session = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        folds = category_context(labels, session)
        held0 = next(f for f in folds if f["held_out"] == 0)
        assert held0["prior"][1] > held0["prior"][0]

    def test_an_absent_category_keeps_prior_mass(self):
        labels = np.array([0, 0, 1, 1])
        folds = category_context(labels, np.array([0, 0, 1, 1]))
        assert all((f["prior"] > 0).all() for f in folds)


class TestAmplitude:
    def test_a_single_training_session_cannot_tune(self):
        """Two-session units have no inner rotation; the fallback must be explicit, not silent."""
        counts, labels, session = _synthetic(n_sessions=2, per_session=100)
        folds = category_context(labels, session)
        assert tuned_category_amplitude(counts, labels, folds[0], SETTINGS) == 1.0

    def test_disabling_the_calibration_returns_one(self):
        counts, labels, session = _synthetic()
        folds = category_context(labels, session)
        off = {"category_surface": {**SETTINGS["category_surface"],
                                    "amplitude_calibration": {
                                        **SETTINGS["category_surface"]["amplitude_calibration"],
                                        "enabled": False}}}
        assert tuned_category_amplitude(counts, labels, folds[0], off) == 1.0


class TestGain:
    def test_a_category_tuned_unit_scores_above_zero(self):
        counts, labels, session = _synthetic(strength=4.0)
        folds = category_context(labels, session)
        assert category_gain(counts, labels, folds, SETTINGS)["gain"] > 0.05

    def test_an_untuned_unit_scores_near_zero(self):
        counts, labels, session = _synthetic(strength=0.0)
        folds = category_context(labels, session)
        assert abs(category_gain(counts, labels, folds, SETTINGS)["gain"]) < 0.02

    def test_a_permutation_destroys_the_gain(self):
        counts, labels, session = _synthetic(strength=4.0)
        folds = category_context(labels, session)
        rng = np.random.default_rng(0)
        perm = np.arange(counts.size)
        for s in np.unique(session):
            rows = np.flatnonzero(session == s)
            perm[rows] = rng.permutation(rows)
        observed = category_gain(counts, labels, folds, SETTINGS)["gain"]
        shuffled = category_gain(counts, labels, folds, SETTINGS, permutation=perm)["gain"]
        assert observed > shuffled + 0.03

    def test_the_permutation_stays_within_session(self):
        """Permuting across sessions would also break the session structure the folds rely on."""
        counts, labels, session = _synthetic()
        folds = category_context(labels, session)
        rng = np.random.default_rng(1)
        perm = np.arange(counts.size)
        for s in np.unique(session):
            rows = np.flatnonzero(session == s)
            perm[rows] = rng.permutation(rows)
        assert (session[perm] == session).all()

    def test_the_key_roster_matches_the_continuous_decoder(self):
        """Downstream code is target-agnostic, so the two must return the same keys."""
        counts, labels, session = _synthetic()
        folds = category_context(labels, session)
        r = category_gain(counts, labels, folds, SETTINGS, with_detail=True)
        for key in ("gain", "gain_uncalibrated", "nats_per_spike", "n_masked", "n_folds_positive",
                    "min_leave_one_fold_out", "per_fold", "alphas", "ridges", "per_event_gain",
                    "posterior_entropy", "mean_posterior_entropy"):
            assert key in r, key
        assert len(r["per_fold"]) == 3
        assert len(r["alphas"]) == 3

    def test_the_leave_one_fold_out_minimum_is_scored_on_n_minus_one_folds(self):
        counts, labels, session = _synthetic(strength=4.0)
        folds = category_context(labels, session)
        r = category_gain(counts, labels, folds, SETTINGS)
        assert r["min_leave_one_fold_out"] <= max(f["gain"] for f in r["per_fold"]) + 1e-9


class TestSilentSessions:
    """A unit that never fires in a held-out session must score ZERO there, not NaN.

    With no spikes the count cannot discriminate, so the posterior equals the prior and the gain is
    exactly zero -- that is the limit, and it is a usable result. Before the exposure was floored this
    produced `0 * log(0)` and returned NaN, which on the first cohort pass silently removed 48 of 1,231
    units, every one of them near-silent rather than uninformative in any interesting way.
    """

    def test_a_silent_held_out_session_scores_zero_not_nan(self):
        counts, labels, session = _synthetic(strength=3.0)
        counts[session == 1] = 0.0
        folds = category_context(labels, session)
        result = category_gain(counts, labels, folds, SETTINGS)
        assert np.isfinite(result["gain"])
        silent = next(f for f in result["per_fold"] if f["held_out"] == 1)
        # Not exactly zero: the floored exposure leaves a residual term per category, bounded by the
        # floor itself and two orders below any real gain.
        assert abs(silent["gain"]) < 1e-3

    def test_a_unit_silent_everywhere_is_finite_and_flat(self):
        counts, labels, session = _synthetic()
        folds = category_context(labels, session)
        result = category_gain(np.zeros_like(counts), labels, folds, SETTINGS)
        assert np.isfinite(result["gain"])
        assert abs(result["gain"]) < 1e-3

    def test_the_posterior_stays_finite_at_zero_exposure(self):
        post = category_posterior(np.zeros(3), np.array([1.0, 2.0, 3.0, 4.0]),
                                  np.full(4, 0.25), 1.0, 0.0)
        assert np.isfinite(post).all()
        assert np.allclose(np.exp(post[0]), 0.25, atol=1e-3)


class TestSpikeSufficiency:
    """Near-silent units destabilise the gain even though the null handles them correctly.

    Measured over 1,231 units: sd(gain) is 0.029 below 50 spikes against 0.014 above, and a unit with
    fewer than 25 spikes returned 0.70 -- three times the cohort's strongest genuinely tuned unit. The
    gate therefore protects every analysis that reads the gain rather than the p-value.
    """

    SETTINGS = {"data_sufficiency": {"min_decoding_spikes": 50}}

    def test_a_well_firing_unit_is_sufficient(self):
        v = decoding_spike_sufficiency(np.full(100, 2.0), self.SETTINGS)
        assert v["sufficient"] and v["n_spikes"] == 200 and v["reason"] is None

    def test_a_near_silent_unit_is_flagged_with_a_reason(self):
        v = decoding_spike_sufficiency(np.array([1.0, 1.0]), self.SETTINGS)
        assert not v["sufficient"]
        assert "below min_decoding_spikes=50" in v["reason"]
        assert v["n_spikes"] == 2

    def test_the_count_is_reported_even_when_the_gate_is_off(self):
        """The standing rule: the quantity is recorded whether or not anything gates on it."""
        v = decoding_spike_sufficiency(np.array([1.0, 1.0]),
                                       {"data_sufficiency": {"min_decoding_spikes": None}})
        assert v["sufficient"] and v["n_spikes"] == 2 and v["threshold"] is None

    def test_the_boundary_is_inclusive(self):
        assert decoding_spike_sufficiency(np.full(50, 1.0), self.SETTINGS)["sufficient"]
        assert not decoding_spike_sufficiency(np.full(49, 1.0), self.SETTINGS)["sufficient"]
