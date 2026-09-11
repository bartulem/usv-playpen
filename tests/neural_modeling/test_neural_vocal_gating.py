"""
@author: bartulem
Coverage: the vocal-gating statistic, its bases, and the three-condition verdict.

Every test here pins a failure that actually occurred during the pilot work, because the gating
result is decided by WHICH frames each effect is measured on and WHICH base it is scored over, and
both can be got wrong in ways that flatter the answer without erroring.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import polars as pl
import pytest

from usv_playpen.neural_modeling import neural_vocal_gating as gating_module
from usv_playpen.neural_modeling.neural_vocal_gating import (
    check_settings,
    feature_gating_statistic,
    gating_survivors,
    gating_terms,
    gating_universe,
    gating_verdict,
    loso_added_deviance,
    settings_bonferroni_bar,
    should_run_gating,
)

SETTINGS_PATH = (pathlib.Path(__file__).resolve().parents[2] / "src" / "usv_playpen"
                 / "_parameter_settings" / "neural_modeling_settings.json")


def _gating() -> dict:
    """The SHIPPED vocal_gating block, so a key the code stops honouring fails here."""
    with SETTINGS_PATH.open() as handle:
        return json.load(handle)["vocal_gating"]


def _universe(n_per_session=900, n_sessions=3, seed=0, gate=0.0, main=0.0, vocal_drive=1.0):
    """A synthetic unit whose vocal response is scaled by a feature to a controllable degree.

    `gate` is the interaction: how much the feature scales firing INSIDE vocalizations. `main` is the
    feature's effect during quiet. `vocal_drive` is the plain vocal main effect."""
    rng = np.random.default_rng(seed)
    n = n_per_session * n_sessions
    feature = rng.normal(size=n)
    vocal = (rng.random(n) < 0.35).astype(np.float64)
    eta = -2.0 + vocal_drive * vocal + main * feature + gate * feature * vocal
    spikes = (rng.random(n) < 1.0 / (1.0 + np.exp(-eta))).astype(np.float64)
    session = np.repeat(np.arange(n_sessions), n_per_session)
    quiet = vocal == 0.0
    return {"feature": feature, "vocal": vocal, "spikes": spikes, "session": session,
            "feature_quiet": feature[quiet], "spikes_quiet": spikes[quiet],
            "session_quiet": session[quiet]}


def _statistic(data, settings=None):
    return feature_gating_statistic(data["feature"], data["feature_quiet"], data["vocal"],
                                    data["spikes"], data["spikes_quiet"], data["session"],
                                    data["session_quiet"], settings or _gating())


class TestTheBases:
    """The three effects are not interchangeable, and pairing one with the wrong base flatters the
    result -- in opposite directions, which is why both mistakes went unnoticed at the time."""

    def test_the_vocal_main_base_excludes_the_interaction(self):
        """Marginality runs one way: V does not require the interaction, the interaction requires V.
        Putting the product in beta_V's base shrank it 410 -> 85 and inflated delta/beta to a fake 11x."""
        blocks = gating_terms(np.arange(6.0), np.array([0.0, 0, 0, 1, 1, 1]))
        full, base = blocks["vocal_main"]
        assert full.shape[1] == 2          # feature + V
        assert base.shape[1] == 1          # feature alone -- no product anywhere

    def test_the_interaction_base_carries_both_main_effects(self):
        blocks = gating_terms(np.arange(6.0), np.array([0.0, 0, 0, 1, 1, 1]))
        full, base = blocks["interaction"]
        assert full.shape[1] == 3
        assert base.shape[1] == 2

    def test_the_feature_main_is_scored_over_an_intercept_alone(self):
        blocks = gating_terms(np.arange(6.0), np.zeros(6))
        full, base = blocks["feature_main"]
        assert full.shape[1] == 1
        assert base.shape[1] == 0

    def test_the_product_is_identically_zero_on_quiet_frames(self):
        """Which is why the interaction cannot be contaminated by quiet rows at all."""
        vocal = np.array([0.0, 0, 1, 1])
        full, _base = gating_terms(np.array([3.0, -2.0, 3.0, -2.0]), vocal)["interaction"]
        assert np.all(full[vocal == 0.0, 2] == 0.0)


class TestTheStatistic:

    def test_a_gated_unit_shows_an_interaction_a_flat_one_does_not(self):
        gated = _statistic(_universe(gate=1.5, seed=1))
        flat = _statistic(_universe(gate=0.0, seed=1))
        assert gated["interaction"] > flat["interaction"]
        assert gated["interaction"] > 0

    def test_the_gating_sign_follows_the_interaction(self):
        assert _statistic(_universe(gate=1.5, seed=2))["gating_sign"] > 0
        assert _statistic(_universe(gate=-1.5, seed=2))["gating_sign"] < 0

    def test_a_quiet_main_effect_is_detected_on_quiet_frames(self):
        with_main = _statistic(_universe(main=1.2, gate=0.0, seed=3))
        without = _statistic(_universe(main=0.0, gate=0.0, seed=3))
        assert with_main["feature_main"] > without["feature_main"]

    def test_a_pure_vocal_main_effect_does_not_masquerade_as_a_gate(self):
        """A unit that simply fires during calls, with no feature dependence, must show a vocal main
        effect WITHOUT an interaction -- the distinction the delta > beta_V condition enforces."""
        data = _universe(gate=0.0, vocal_drive=2.0, seed=4)
        statistic = _statistic(data)
        assert statistic["vocal_main"] > 0
        assert statistic["interaction_minus_vocal"] < 0

    def test_a_socially_gated_unit_shows_delta_greater_than_beta(self):
        """The calibration unit's signature: the response IS the gated part, measured at
        delta 971 against beta_V 410, a ratio of 2.37. This synthetic lands at 2.27, and it is the
        case the delta > beta_V condition exists to separate from a vocal-motor unit."""
        statistic = _statistic(_universe(gate=3.0, vocal_drive=0.25, seed=4))
        assert statistic["interaction"] > statistic["vocal_main"]
        assert statistic["interaction_minus_vocal"] > 0

    def test_a_vocal_motor_unit_with_mild_modulation_is_not_a_gate(self):
        """Real interactions that are dwarfed by their vocal main effect are secondary modulations,
        not gates -- on the calibration unit every other feature landed here, ratios 0.00-0.37."""
        statistic = _statistic(_universe(gate=1.5, vocal_drive=1.0, seed=4))
        assert statistic["interaction"] > 0            # the interaction is REAL
        assert statistic["interaction_minus_vocal"] < 0   # and still smaller than the vocal main

    def test_a_single_session_is_refused_rather_than_silently_scored(self):
        blocks = gating_terms(np.arange(10.0), np.zeros(10))
        with pytest.raises(ValueError, match="at least two sessions"):
            loso_added_deviance(*blocks["feature_main"], np.zeros(10), np.zeros(10),
                                0.02, 5, 5)


class TestTheVerdict:

    NULL = np.linspace(-5.0, 5.0, 4001)

    def test_all_three_conditions_must_hold_for_a_clean_gate(self):
        observed = {"interaction": 50.0, "interaction_minus_vocal": 20.0, "feature_main": -3.0}
        verdict = gating_verdict(observed, self.NULL, self.NULL, self.NULL, _gating(), 19)
        assert verdict["label"] == "GATE"

    def test_a_present_main_effect_is_labelled_not_hidden(self):
        """The alpha bar is deliberately GENEROUS toward detecting a main effect, so the clean GATE
        label is reserved for convincingly-absent mains."""
        observed = {"interaction": 50.0, "interaction_minus_vocal": 20.0, "feature_main": 50.0}
        verdict = gating_verdict(observed, self.NULL, self.NULL, self.NULL, _gating(), 19)
        assert verdict["label"] == "GATE+silentME"

    def test_an_interaction_smaller_than_the_vocal_main_is_not_a_gate(self):
        observed = {"interaction": 50.0, "interaction_minus_vocal": -20.0, "feature_main": -3.0}
        verdict = gating_verdict(observed, self.NULL, self.NULL, self.NULL, _gating(), 19)
        assert verdict["label"] == "int<vocal"

    def test_an_unremarkable_interaction_is_ns(self):
        observed = {"interaction": 0.0, "interaction_minus_vocal": 20.0, "feature_main": -3.0}
        verdict = gating_verdict(observed, self.NULL, self.NULL, self.NULL, _gating(), 19)
        assert verdict["label"] == "ns"

    def test_the_borderline_flag_is_diagnostic_and_not_a_second_gate(self):
        """A pass with a small sigma margin is FLAGGED for eyes-on review, not demoted -- keeping the
        formal alpha interpretable."""
        observed = {"interaction": 6.0, "interaction_minus_vocal": 20.0, "feature_main": -3.0}
        verdict = gating_verdict(observed, self.NULL, self.NULL, self.NULL, _gating(), 19)
        assert verdict["label"] == "GATE"
        assert verdict["borderline"]


class TestTheShuffleCountCanReachTheBar:
    """An empirical p floors at 1/(n+1). A shuffle count too small to reach the Bonferroni bar makes
    every feature unpassable however strong -- silently, since nothing else would complain."""

    def test_the_shipped_count_clears_the_bar_for_nineteen_features(self):
        assert settings_bonferroni_bar(_gating(), 19) == pytest.approx(0.01 / 19)
        assert 1.0 / (_gating()["gating_screen_n_shuffles"] + 1.0) <= 0.01 / 19

    def test_too_few_shuffles_is_refused_with_the_count_that_would_work(self):
        settings = _gating()
        settings["gating_screen_n_shuffles"] = 1000
        with pytest.raises(ValueError, match="no feature could pass"):
            settings_bonferroni_bar(settings, 19)

    def test_the_shipped_override_of_the_shared_default_is_deliberate(self):
        """1,000 is the project-wide default and CANNOT clear 0.01/19 = 5.3e-4; gating overrides it."""
        assert _gating()["gating_screen_n_shuffles"] == 2000


class TestActivation:
    """Two independent reasons to skip a unit, and they mean different things: one says the unit has
    nothing for a gate to modulate, the other says we could not test it."""

    def test_a_unit_with_claim_two_tuning_is_run(self):
        run, reason = should_run_gating(True, False, 500, 1.0, _gating())
        assert run
        assert reason is None
        assert should_run_gating(False, True, 500, 1.0, _gating())[0]

    def test_a_unit_with_no_vocal_tuning_is_skipped(self):
        """User-ruled: the screen is 19 features x 2,000 shuffles, and a unit with no vocal tuning has
        nothing for a gate to modulate."""
        run, reason = should_run_gating(False, False, 500, 1.0, _gating())
        assert not run
        assert reason == "no_claim2_tuning"

    def test_the_filter_can_be_turned_off(self):
        settings = _gating()
        settings["require_claim2_tuning"] = False
        assert should_run_gating(False, False, 500, 1.0, settings)[0]

    def test_an_unidentifiable_unit_reads_not_testable_not_ns(self):
        """`ns` says we tested it and found nothing; `not_testable` says we could not test it.
        Collapsing the two converts missing power into evidence of absence."""
        settings = _gating()
        settings["min_vocal_spikes"] = 100
        run, reason = should_run_gating(True, True, 10, 1.0, settings)
        assert not run
        assert reason == "not_testable"

    def test_a_feature_flat_during_vocal_frames_is_unidentifiable(self):
        """delta_f is identified purely from vocal frames, so a feature with no spread there cannot
        be estimated at all."""
        settings = _gating()
        settings["min_feature_iqr_vocal"] = 0.5
        run, reason = should_run_gating(True, True, 500, 0.01, settings)
        assert not run
        assert reason == "not_testable"

    def test_identifiability_is_checked_before_the_tuning_filter(self):
        """A unit that is BOTH untestable and untuned should report the stronger fact -- that the test
        could not be run -- rather than implying it was skipped merely for lacking tuning."""
        settings = _gating()
        settings["min_vocal_spikes"] = 100
        assert should_run_gating(False, False, 10, 1.0, settings)[1] == "not_testable"

    def test_the_gates_default_off_like_every_other_min_knob(self):
        assert _gating()["min_vocal_spikes"] is None
        assert _gating()["min_feature_iqr_vocal"] is None


class TestImplementedOptions:

    def test_the_shipped_block_is_accepted(self):
        check_settings(_gating())

    def test_the_vocal_window_names_what_it_means(self):
        """`start_stop` named the MECHANISM -- it reads the start and stop columns. `during_call`
        names the meaning. The nameable alternative is a fixed peri-onset window, rejected because
        gating asks about VOCALIZING rather than peri-vocalizing."""
        assert _gating()["vocal_window"] == "during_call"

    def test_a_peri_onset_vocal_window_is_refused(self):
        settings = _gating()
        settings["vocal_window"] = "peri_onset"
        with pytest.raises(ValueError, match="is not implemented"):
            check_settings(settings)

    def test_content_gating_cannot_be_switched_on_yet(self):
        """Deferred, and it must be gated on a claim-2 pass -- run blind on a non-tuned unit the
        content screen produces false positives."""
        settings = _gating()
        settings["content_gating_compute"] = True
        with pytest.raises(ValueError, match="is not implemented"):
            check_settings(settings)


class TestSurvivorReporting:
    """All survivors are reported with their collinearity, rather than forward-selected to a minimal
    set -- under collinearity a greedy search is knife-edge, and which proxy it keeps is unstable
    rather than meaningful. Feature identity is DESCRIPTION here, so a stable description beats an
    unstable selection."""

    @staticmethod
    def _verdicts(labels_and_margins):
        return {name: {"label": label, "sigma_margin": margin}
                for name, label, margin in labels_and_margins}

    def test_only_gated_labels_survive(self):
        per_feature = self._verdicts([("a", "GATE", 40.0), ("b", "int<vocal", 30.0),
                                      ("c", "ns", 1.0), ("d", "GATE+silentME", 20.0)])
        values = {name: np.arange(10.0) for name in per_feature}
        summary = gating_survivors(per_feature, values)
        assert summary["survivors"] == ["a", "d"]
        assert summary["n_survivors"] == 2

    def test_survivors_are_ordered_strongest_first(self):
        per_feature = self._verdicts([("weak", "GATE", 6.0), ("strong", "GATE", 46.0)])
        values = {name: np.arange(10.0) for name in per_feature}
        assert gating_survivors(per_feature, values)["survivors"] == ["strong", "weak"]

    def test_collinear_survivors_are_flagged_by_their_correlation(self):
        """Four survivors correlating at 0.9 are one story told four ways; the reader needs to see
        that without the pipeline having committed to which one is 'the' feature."""
        rng = np.random.default_rng(0)
        base = rng.normal(size=400)
        per_feature = self._verdicts([("x", "GATE", 40.0), ("x_proxy", "GATE", 38.0)])
        values = {"x": base, "x_proxy": base + 0.05 * rng.normal(size=400)}
        summary = gating_survivors(per_feature, values)
        assert summary["max_abs_correlation"] > 0.9
        assert "x|x_proxy" in summary["correlations"]

    def test_independent_survivors_are_not_flagged(self):
        rng = np.random.default_rng(1)
        per_feature = self._verdicts([("x", "GATE", 40.0), ("y", "GATE", 38.0)])
        values = {"x": rng.normal(size=400), "y": rng.normal(size=400)}
        assert abs(gating_survivors(per_feature, values)["max_abs_correlation"]) < 0.3

    def test_a_flat_feature_gives_nan_rather_than_a_spurious_correlation(self):
        per_feature = self._verdicts([("x", "GATE", 40.0), ("flat", "GATE", 38.0)])
        values = {"x": np.arange(50.0), "flat": np.zeros(50)}
        assert np.isnan(gating_survivors(per_feature, values)["correlations"]["x|flat"])

    def test_no_survivors_reports_cleanly(self):
        summary = gating_survivors(self._verdicts([("a", "ns", 1.0)]), {"a": np.arange(10.0)})
        assert summary["survivors"] == []
        assert np.isnan(summary["max_abs_correlation"])

    def test_the_forward_selection_knob_is_gone(self):
        """`forward_stop_gain` configured a greedy search that is no longer performed."""
        assert "forward_stop_gain" not in _gating()


class TestTheUniverse:
    """Three frame categories, two of which the model sees. The third -- peri-vocal -- must be absent
    from BOTH, because it holds most of the firing and folding it into quiet manufactures a feature
    main effect out of nothing."""

    FPS, N_FRAMES = 100.0, 4000

    def _patch(self, monkeypatch, calls):
        """Two sessions with identical structure, and features that ARE the frame index so an
        instantaneous read can be checked exactly."""
        module = gating_module
        monkeypatch.setattr(module, "session_timebase",
                            lambda _root, _sid: (["male", "female"], self.FPS, self.N_FRAMES))
        monkeypatch.setattr(module, "load_session_usvs", lambda _root, _sid: pl.DataFrame(
            {"start": [c[0] for c in calls], "stop": [c[1] for c in calls],
             "emitter": [c[2] for c in calls]}))

    def _per_session(self, session_ids, quiet, spike_frames):
        series = np.column_stack([np.arange(self.N_FRAMES, dtype=float),
                                  -np.arange(self.N_FRAMES, dtype=float)])
        return {sid: {"feature_time_series": series, "feature_names": ["f0", "f1"],
                      "fps": self.FPS, "n_frames": self.N_FRAMES,
                      "quiet": np.asarray(quiet, dtype=np.int64),
                      "spike_frames": np.asarray(spike_frames, dtype=np.int64)}
                for sid in session_ids}

    def _unit(self):
        return {"mouse_id": "male", "vocal_sessions": ["s0", "s1"]}

    def test_peri_vocal_frames_reach_neither_class(self, monkeypatch):
        """A frame just outside a call is neither vocal nor quiet, and must appear in no row."""
        self._patch(monkeypatch, [(10.0, 10.5, "male")])          # frames 1000-1049
        quiet = np.arange(3000, 3500)                              # far from the call
        universe = gating_universe(self._unit(), "/data",
                                   self._per_session(["s0", "s1"], quiet, [1005, 3010]),
                                   _gating(), {"vocal_emitter": "self"})
        frames_used = universe["features"][:, 0]                   # feature 0 IS the frame index
        assert 1005 in frames_used                                 # inside the call -> vocal
        assert 3010 in frames_used                                 # a quiet anchor
        assert 1100 not in frames_used                             # just after it -> peri-vocal, gone
        assert 2000 not in frames_used

    def test_the_vocal_indicator_marks_exactly_the_call_frames(self, monkeypatch):
        self._patch(monkeypatch, [(10.0, 10.5, "male")])
        universe = gating_universe(self._unit(), "/data",
                                   self._per_session(["s0", "s1"], np.arange(3000, 3500), [1005]),
                                   _gating(), {"vocal_emitter": "self"})
        frames = universe["features"][:, 0]
        inside = (frames >= 1000) & (frames < 1050)
        np.testing.assert_array_equal(universe["vocal"] > 0, inside)

    def test_features_are_instantaneous_not_lagged(self, monkeypatch):
        """Gating asks whether the CURRENT context scales the response; it neither needs nor uses
        claim 1's 600 lags."""
        self._patch(monkeypatch, [(10.0, 10.2, "male")])
        universe = gating_universe(self._unit(), "/data",
                                   self._per_session(["s0", "s1"], np.arange(3000, 3100), [1005]),
                                   _gating(), {"vocal_emitter": "self"})
        assert universe["features"].shape[1] == 2                  # one column per feature, no lags
        np.testing.assert_allclose(universe["features"][:, 1], -universe["features"][:, 0])

    def test_a_partner_call_does_not_create_vocal_frames(self, monkeypatch):
        """V is the RECORDED animal vocalizing; a partner call is not that."""
        self._patch(monkeypatch, [(10.0, 10.5, "female")])
        universe = gating_universe(self._unit(), "/data",
                                   self._per_session(["s0", "s1"], np.arange(3000, 3100), [1005]),
                                   _gating(), {"vocal_emitter": "self"})
        assert universe["vocal"].sum() == 0

    def test_the_quiet_cap_is_honoured(self, monkeypatch):
        self._patch(monkeypatch, [(10.0, 10.5, "male")])
        settings = _gating()
        settings["max_quiet_frames_per_session"] = 50
        universe = gating_universe(self._unit(), "/data",
                                   self._per_session(["s0", "s1"], np.arange(2000, 3000), [1005]),
                                   settings, {"vocal_emitter": "self"})
        for book in universe["per_session"].values():
            assert book["n_quiet_frames_available"] == 1000
            assert book["n_quiet_frames_used"] == 50

    def test_the_quiet_only_set_bounds_zeros_per_spike(self, monkeypatch):
        """alpha_f is a rare-event logistic there, so the ZERO count is what needs bounding -- a
        different question from the universe's cap, which bounds silence against vocal frames."""
        self._patch(monkeypatch, [(10.0, 10.2, "male")])
        settings = _gating()
        settings["quiet_zeros_per_spike"] = 3
        quiet = np.arange(2000, 3000)
        universe = gating_universe(self._unit(), "/data",
                                   self._per_session(["s0", "s1"], quiet, [1005, 2500]),
                                   settings, {"vocal_emitter": "self"})
        per_session_rows = universe["spikes_quiet"][universe["session_quiet"] == 0]
        assert per_session_rows.sum() == 1                         # the one quiet spike is kept
        assert per_session_rows.size == 4                          # plus 3 zeros, not 999

    def test_the_bookkeeping_counts_what_was_excluded(self, monkeypatch):
        """The peri-vocal count is reported rather than silently dropped, because its size is the
        whole reason the category exists."""
        self._patch(monkeypatch, [(10.0, 10.5, "male")])
        universe = gating_universe(self._unit(), "/data",
                                   self._per_session(["s0", "s1"], np.arange(3000, 3500), [1005]),
                                   _gating(), {"vocal_emitter": "self"})
        book = universe["per_session"]["s0"]
        assert book["n_vocal_frames"] == 50
        assert book["n_quiet_frames_available"] == 500
        assert book["n_peri_vocal_frames_excluded"] == self.N_FRAMES - 50 - 500
        assert book["n_focal_calls"] == 1
