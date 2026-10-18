"""
@author: bartulem
Unit tests for ``usv_playpen.neural_modeling.neural_vocal_gating``.

Coverage: the vocal-gating statistic, its bases, and the three-condition verdict; the shuffle-count
bar, the activation rules, the implemented options, survivor reporting, the frame universe, the
spike-train shift null, the alpha settings and the per-unit p-value.

Every test here pins a failure that actually occurred during the pilot work, because the gating
result is decided by WHICH frames each effect is measured on and WHICH base it is scored over, and
both can be got wrong in ways that flatter the answer without erroring.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import polars as pls
import pytest

from usv_playpen.neural_modeling import neural_vocal_gating as gating_module
from usv_playpen.neural_modeling.neural_significance import benjamini_hochberg
from usv_playpen.neural_modeling.neural_vocal_gating import (
    check_settings,
    feature_gating_statistic,
    gating_null,
    gating_survivors,
    gating_terms,
    gating_unit_pvalue,
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
    """
    Description
    -----------
    Build a synthetic unit whose vocal response is scaled by a feature to a controllable degree.

    `gate` is the interaction: how much the feature scales firing INSIDE vocalizations. `main` is the
    feature's effect during quiet. `vocal_drive` is the plain vocal main effect.

    Parameters
    ----------
    n_per_session (int)
        Frames per session; defaults to 900.
    n_sessions (int)
        Number of sessions; defaults to 3.
    seed (int)
        Seed for ``np.random.default_rng``; defaults to 0.
    gate (float)
        Feature x vocal interaction weight on the log-odds; defaults to 0.0.
    main (float)
        Feature main-effect weight; defaults to 0.0.
    vocal_drive (float)
        Vocal main-effect weight; defaults to 1.0.

    Returns
    -------
    data (dict)
        ``feature``, ``vocal``, ``spikes`` and ``session`` over all frames, plus their quiet-only
        subsets ``feature_quiet``, ``spikes_quiet`` and ``session_quiet``.
    """
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
    """
    Description
    -----------
    Run ``feature_gating_statistic`` on a ``_universe`` dict.

    Parameters
    ----------
    data (dict)
        Output of ``_universe``.
    settings (dict | None)
        A ``vocal_gating`` block; the shipped block when ``None``.

    Returns
    -------
    statistic (dict)
        The per-feature gating statistic (``interaction``, ``vocal_main``, ``feature_main``,
        ``interaction_minus_vocal``, ``gating_sign``, ...).
    """
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
        """The interaction is scored against a base holding the feature AND V, so delta measures only
        what the product adds beyond both main effects."""
        blocks = gating_terms(np.arange(6.0), np.array([0.0, 0, 0, 1, 1, 1]))
        full, base = blocks["interaction"]
        assert full.shape[1] == 3
        assert base.shape[1] == 2

    def test_the_feature_main_is_scored_over_an_intercept_alone(self):
        """The quiet feature main effect has no other term to compete with, so its base is the
        intercept alone (zero columns)."""
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
    """The per-feature statistic on synthetic units with planted gating, main and vocal effects."""

    def test_a_gated_unit_shows_an_interaction_a_flat_one_does_not(self):
        """A planted gate raises the interaction above the same data without one, and above zero."""
        gated = _statistic(_universe(gate=1.5, seed=1))
        flat = _statistic(_universe(gate=0.0, seed=1))
        assert gated["interaction"] > flat["interaction"]
        assert gated["interaction"] > 0

    def test_the_gating_sign_follows_the_interaction(self):
        """``gating_sign`` reports the direction of the planted interaction, so suppressive and
        facilitating gates are not reported alike."""
        assert _statistic(_universe(gate=1.5, seed=2))["gating_sign"] > 0
        assert _statistic(_universe(gate=-1.5, seed=2))["gating_sign"] < 0

    def test_a_quiet_main_effect_is_detected_on_quiet_frames(self):
        """A planted quiet main effect raises ``feature_main``, so the third verdict condition has
        something real to detect."""
        with_main = _statistic(_universe(main=1.2, gate=0.0, seed=3))
        without = _statistic(_universe(main=0.0, gate=0.0, seed=3))
        assert with_main["feature_main"] > without["feature_main"]

    def test_a_pure_vocal_main_effect_does_not_masquerade_as_a_gate(self):
        """A unit that simply fires during USVs, with no feature dependence, must show a vocal main
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
        """Leave-one-session-out needs at least two sessions; one session must raise instead of
        returning a score fitted and tested on the same data."""
        blocks = gating_terms(np.arange(10.0), np.zeros(10))
        with pytest.raises(ValueError, match="at least two sessions"):
            loso_added_deviance(*blocks["feature_main"], np.zeros(10), np.zeros(10),
                                0.02, 5, 5)


class TestTheVerdict:
    """The three-condition verdict labels against a fixed null."""

    NULL = np.linspace(-5.0, 5.0, 4001)

    def test_all_three_conditions_must_hold_for_a_clean_gate(self):
        """Significant interaction, interaction above the vocal main, and no quiet main effect together
        give the clean ``GATE`` label."""
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
        """A significant interaction that does not beat the vocal main effect is labelled
        ``int<vocal``, separating secondary modulation from gating."""
        observed = {"interaction": 50.0, "interaction_minus_vocal": -20.0, "feature_main": -3.0}
        verdict = gating_verdict(observed, self.NULL, self.NULL, self.NULL, _gating(), 19)
        assert verdict["label"] == "int<vocal"

    def test_an_unremarkable_interaction_is_ns(self):
        """An interaction at the centre of its null is ``ns`` whatever the other two conditions say."""
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
        """The Bonferroni bar is screen_alpha / 19 and the shipped shuffle count's p floor lies at or
        below it, so a strong feature can actually pass."""
        assert settings_bonferroni_bar(_gating(), 19) == pytest.approx(0.01 / 19)
        assert 1.0 / (_gating()["gating_screen_n_shuffles"] + 1.0) <= 0.01 / 19

    def test_too_few_shuffles_is_refused_with_the_count_that_would_work(self):
        """1,000 shuffles cannot reach 0.01/19, so the bar computation must raise rather than run a
        screen no feature could pass."""
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

    def test_a_unit_with_vocal_tuning_is_run(self):
        """A unit tuned on either vocal axis, with enough spikes and feature spread, is run with no
        skip reason."""
        run, reason = should_run_gating(True, False, 500, 1.0, _gating())
        assert run
        assert reason is None
        assert should_run_gating(False, True, 500, 1.0, _gating())[0]

    def test_a_unit_with_no_vocal_tuning_is_skipped(self):
        """User-ruled: the screen is 19 features x 2,000 shuffles, and a unit with no vocal tuning has
        nothing for a gate to modulate."""
        run, reason = should_run_gating(False, False, 500, 1.0, _gating())
        assert not run
        assert reason == "no_vocal_tuning"

    def test_the_filter_can_be_turned_off(self):
        """``require_vocal_tuning=False`` runs an untuned unit, so the tuning filter is a real knob."""
        settings = _gating()
        settings["require_vocal_tuning"] = False
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
        """Both identifiability minimums ship as ``None`` (disabled), matching the other ``min_*``
        knobs."""
        assert _gating()["min_vocal_spikes"] is None
        assert _gating()["min_feature_iqr_vocal"] is None


class TestImplementedOptions:
    """``check_settings`` accepts the shipped block and refuses options that are not implemented."""

    def test_the_shipped_block_is_accepted(self):
        """The shipped ``vocal_gating`` block passes validation, so a default run is never refused."""
        check_settings(_gating())

    def test_the_vocal_window_names_what_it_means(self):
        """`start_stop` named the MECHANISM -- it reads the start and stop columns.
        `during_vocalization` names the meaning. The nameable alternative is a fixed peri-onset window,
        rejected because gating asks about VOCALIZING rather than peri-vocalizing."""
        assert _gating()["vocal_window"] == "during_vocalization"

    def test_the_retired_window_name_is_refused(self):
        """The value was renamed to ``during_vocalization``; the retired spelling is refused rather than
        silently accepted, so a stale settings file fails loudly."""
        settings = _gating()
        settings["vocal_window"] = "during_" + "call"
        with pytest.raises(ValueError, match="is not implemented"):
            check_settings(settings)

    def test_a_peri_onset_vocal_window_is_refused(self):
        """The rejected peri-onset window must raise rather than be silently treated as the
        during-vocalization window."""
        settings = _gating()
        settings["vocal_window"] = "peri_onset"
        with pytest.raises(ValueError, match="is not implemented"):
            check_settings(settings)

    def test_content_gating_cannot_be_switched_on_yet(self):
        """Deferred, and it must be gated on a vocalization-identity pass -- run blind on a non-tuned unit the
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
        """
        Description
        -----------
        Build per-feature verdicts carrying only a label and a sigma margin.

        Parameters
        ----------
        labels_and_margins (list[tuple[str, str, float]])
            ``(feature_name, label, sigma_margin)`` per feature.

        Returns
        -------
        per_feature (dict)
            ``{feature_name: {"label": ..., "sigma_margin": ...}}``.
        """
        return {name: {"label": label, "sigma_margin": margin}
                for name, label, margin in labels_and_margins}

    def test_only_gated_labels_survive(self):
        """Only ``GATE`` and ``GATE+silentME`` count as survivors; ``int<vocal`` and ``ns`` do not."""
        per_feature = self._verdicts([("a", "GATE", 40.0), ("b", "int<vocal", 30.0),
                                      ("c", "ns", 1.0), ("d", "GATE+silentME", 20.0)])
        values = {name: np.arange(10.0) for name in per_feature}
        summary = gating_survivors(per_feature, values)
        assert summary["survivors"] == ["a", "d"]
        assert summary["n_survivors"] == 2

    def test_survivors_are_ordered_strongest_first(self):
        """Survivors are listed by descending sigma margin, so the first one named is the strongest."""
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
        """Uncorrelated survivors report a small maximum correlation, so the collinearity flag does not
        fire on noise."""
        rng = np.random.default_rng(1)
        per_feature = self._verdicts([("x", "GATE", 40.0), ("y", "GATE", 38.0)])
        values = {"x": rng.normal(size=400), "y": rng.normal(size=400)}
        assert abs(gating_survivors(per_feature, values)["max_abs_correlation"]) < 0.3

    def test_a_flat_feature_gives_nan_rather_than_a_spurious_correlation(self):
        """A zero-variance feature has an undefined correlation, reported as NaN rather than 0 or 1."""
        per_feature = self._verdicts([("x", "GATE", 40.0), ("flat", "GATE", 38.0)])
        values = {"x": np.arange(50.0), "flat": np.zeros(50)}
        assert np.isnan(gating_survivors(per_feature, values)["correlations"]["x|flat"])

    def test_no_survivors_reports_cleanly(self):
        """With no survivor the list is empty and the maximum correlation NaN, rather than an error."""
        summary = gating_survivors(self._verdicts([("a", "ns", 1.0)]), {"a": np.arange(10.0)})
        assert summary["survivors"] == []
        assert np.isnan(summary["max_abs_correlation"])

    def test_the_forward_selection_knob_is_gone(self):
        """`forward_stop_gain` configured a greedy search that is no longer performed."""
        assert "forward_stop_gain" not in _gating()


VOCAL_SETTINGS = {"vocal_emitter": "self", "exclude_noise": True, "exclude_squeaks": True,
                  "squeaks_block_quiet": True}


class TestTheUniverse:
    """Three frame categories, two of which the model sees. The third -- peri-vocal -- must be absent
    from BOTH, because it holds most of the firing and folding it into quiet manufactures a feature
    main effect out of nothing."""

    FPS, N_FRAMES = 100.0, 4000

    def _patch(self, monkeypatch, usvs):
        """
        Description
        -----------
        Patch the session timebase and USV loader so every session has identical structure
        (male/female tracks, ``FPS``, ``N_FRAMES``) and the given USV table.

        Parameters
        ----------
        monkeypatch (pytest.MonkeyPatch)
            The test's monkeypatch fixture.
        usvs (list[tuple[float, float, str]])
            ``(start, stop, emitter)`` per USV; written as clean (non-noise, non-squeak) rows.

        Returns
        -------
        None
            The patches stay active for the test.
        """
        module = gating_module
        monkeypatch.setattr(module, "session_timebase",
                            lambda _root, _sid: (["male", "female"], self.FPS, self.N_FRAMES))
        monkeypatch.setattr(module, "load_session_usvs", lambda _root, _sid: pls.DataFrame(
            {"start": [c[0] for c in usvs], "stop": [c[1] for c in usvs],
             "emitter": [c[2] for c in usvs],
             # The classifier columns the vocalization filter reads, as the real summaries carry them.
             "noise": [False] * len(usvs), "squeak": [False] * len(usvs)}))

    def _per_session(self, session_ids, quiet, spike_frames):
        """
        Description
        -----------
        Build per-session kinematic records whose two features ARE the frame index and its negative,
        so an instantaneous read can be checked exactly.

        Parameters
        ----------
        session_ids (list[str])
            Session ids to build (all identical).
        quiet (array-like)
            Quiet anchor frames.
        spike_frames (array-like)
            Spike frames.

        Returns
        -------
        per_session (dict)
            ``{session_id: {...}}`` with ``feature_time_series``, ``feature_names``, ``fps``,
            ``n_frames``, ``quiet`` and ``spike_frames``.
        """
        series = np.column_stack([np.arange(self.N_FRAMES, dtype=float),
                                  -np.arange(self.N_FRAMES, dtype=float)])
        return {sid: {"feature_time_series": series, "feature_names": ["f0", "f1"],
                      "fps": self.FPS, "n_frames": self.N_FRAMES,
                      "quiet": np.asarray(quiet, dtype=np.int64),
                      "spike_frames": np.asarray(spike_frames, dtype=np.int64)}
                for sid in session_ids}

    def _unit(self):
        """
        Description
        -----------
        Return the unit record the universe builder reads.

        Parameters
        ----------
        None

        Returns
        -------
        unit (dict)
            ``mouse_id`` ``"male"`` with vocal sessions ``s0`` and ``s1``.
        """
        return {"mouse_id": "male", "vocal_sessions": ["s0", "s1"]}

    def test_peri_vocal_frames_reach_neither_class(self, monkeypatch):
        """A frame just outside a USV is neither vocal nor quiet, and must appear in no row."""
        self._patch(monkeypatch, [(10.0, 10.5, "male")])          # frames 1000-1049
        quiet = np.arange(3000, 3500)                              # far from the USV
        universe = gating_universe(self._unit(), "/data",
                                   self._per_session(["s0", "s1"], quiet, [1005, 3010]),
                                   _gating(), VOCAL_SETTINGS)
        frames_used = universe["features"][:, 0]                   # feature 0 IS the frame index
        assert 1005 in frames_used                                 # inside the USV -> vocal
        assert 3010 in frames_used                                 # a quiet anchor
        assert 1100 not in frames_used                             # just after it -> peri-vocal, gone
        assert 2000 not in frames_used

    def test_the_vocal_indicator_marks_exactly_the_usv_frames(self, monkeypatch):
        """V is 1 on exactly the frames between the focal USV's start and stop and 0 elsewhere, so the
        interaction is measured on vocalizing frames only."""
        self._patch(monkeypatch, [(10.0, 10.5, "male")])
        universe = gating_universe(self._unit(), "/data",
                                   self._per_session(["s0", "s1"], np.arange(3000, 3500), [1005]),
                                   _gating(), VOCAL_SETTINGS)
        frames = universe["features"][:, 0]
        inside = (frames >= 1000) & (frames < 1050)
        np.testing.assert_array_equal(universe["vocal"] > 0, inside)

    def test_features_are_instantaneous_not_lagged(self, monkeypatch):
        """Gating asks whether the CURRENT context scales the response; it neither needs nor uses
        the kinematic encoding's 600 lags."""
        self._patch(monkeypatch, [(10.0, 10.2, "male")])
        universe = gating_universe(self._unit(), "/data",
                                   self._per_session(["s0", "s1"], np.arange(3000, 3100), [1005]),
                                   _gating(), VOCAL_SETTINGS)
        assert universe["features"].shape[1] == 2                  # one column per feature, no lags
        np.testing.assert_allclose(universe["features"][:, 1], -universe["features"][:, 0])

    def test_a_partner_usv_does_not_create_vocal_frames(self, monkeypatch):
        """V is the RECORDED animal vocalizing; a partner USV is not that."""
        self._patch(monkeypatch, [(10.0, 10.5, "female")])
        universe = gating_universe(self._unit(), "/data",
                                   self._per_session(["s0", "s1"], np.arange(3000, 3100), [1005]),
                                   _gating(), VOCAL_SETTINGS)
        assert universe["vocal"].sum() == 0

    def test_vocal_emitter_all_counts_every_emitter_instead_of_raising(self, monkeypatch):
        """``vocal_emitter = 'all'`` resolves to no emitter filter; it used to index the empty list
        ``emitter_names`` returns for it and raise IndexError."""
        self._patch(monkeypatch, [(10.0, 10.5, "female"), (20.0, 20.5, "male")])
        universe = gating_universe(self._unit(), "/data",
                                   self._per_session(["s0", "s1"], np.arange(3000, 3100), [1005]),
                                   _gating(), {**VOCAL_SETTINGS, "vocal_emitter": "all"})
        assert universe["per_session"]["s0"]["n_focal_vocalizations"] == 2
        assert universe["vocal"].sum() == 2 * 2 * 50

    def test_the_quiet_cap_is_honoured(self, monkeypatch):
        """``max_quiet_frames_per_session`` limits the quiet rows in the FIT only; every quiet frame is
        still scored, which is what keeps the cap from setting the effect size."""
        self._patch(monkeypatch, [(10.0, 10.5, "male")])
        settings = _gating()
        settings["max_quiet_frames_per_session"] = 50
        universe = gating_universe(self._unit(), "/data",
                                   self._per_session(["s0", "s1"], np.arange(2000, 3000), [1005]),
                                   settings, VOCAL_SETTINGS)
        for book in universe["per_session"].values():
            assert book["n_quiet_frames_available"] == 1000
            assert book["n_quiet_frames_fitted"] == 50
        # Every quiet frame stays in the universe and is SCORED; the cap only withholds rows from the fit.
        assert int(np.sum(universe["vocal"] < 0.5)) == 2000
        assert int(np.sum(universe["fit_rows"] & (universe["vocal"] < 0.5))) == 100

    def test_the_quiet_only_set_bounds_zeros_per_spike(self, monkeypatch):
        """alpha_f is a rare-event logistic there, so the ZERO count is what needs bounding -- a
        different question from the universe's cap, which bounds silence against vocal frames."""
        self._patch(monkeypatch, [(10.0, 10.2, "male")])
        settings = _gating()
        settings["quiet_zeros_per_spike"] = 3
        quiet = np.arange(2000, 3000)
        universe = gating_universe(self._unit(), "/data",
                                   self._per_session(["s0", "s1"], quiet, [1005, 2500]),
                                   settings, VOCAL_SETTINGS)
        per_session_rows = universe["spikes_quiet"][universe["session_quiet"] == 0]
        assert per_session_rows.sum() == 1                         # the one quiet spike is kept
        assert per_session_rows.size == 4                          # plus 3 zeros, not 999

    def test_the_bookkeeping_counts_what_was_excluded(self, monkeypatch):
        """The peri-vocal count is reported rather than silently dropped, because its size is the
        whole reason the category exists."""
        self._patch(monkeypatch, [(10.0, 10.5, "male")])
        universe = gating_universe(self._unit(), "/data",
                                   self._per_session(["s0", "s1"], np.arange(3000, 3500), [1005]),
                                   _gating(), VOCAL_SETTINGS)
        book = universe["per_session"]["s0"]
        assert book["n_vocal_frames"] == 50
        assert book["n_quiet_frames_available"] == 500
        assert book["n_peri_vocal_frames_excluded"] == self.N_FRAMES - 50 - 500
        assert book["n_focal_vocalizations"] == 1


class TestTheNull:
    """A circular shift of the SPIKE TRAIN. The features, the vocal indicator and the frame
    membership never move -- only which frames carry a spike."""

    @staticmethod
    def _universe(n=1200, n_sessions=2, seed=0):
        """
        Description
        -----------
        Build a minimal gating universe (one feature, every third row vocal, random spikes) with
        every key ``gating_null`` reads.

        Parameters
        ----------
        n (int)
            Total rows; defaults to 1200.
        n_sessions (int)
            Number of sessions; defaults to 2.
        seed (int)
            Seed for ``np.random.default_rng``; defaults to 0.

        Returns
        -------
        universe (dict)
            Row arrays, their quiet subsets (the observed subsample and every quiet row), per-session
            spike frames, frame counts and rates.
        """
        rng = np.random.default_rng(seed)
        per = n // n_sessions
        index = np.repeat(np.arange(n_sessions), per)
        frames = np.tile(np.arange(per) * 3, n_sessions)
        vocal = (np.tile(np.arange(per), n_sessions) % 3 == 0).astype(float)
        feature = rng.normal(size=n)
        quiet_rows = vocal == 0.0
        return {"features": feature[:, None], "vocal": vocal,
                "spikes": (rng.random(n) < 0.2).astype(float), "session_index": index,
                "frames": frames.astype(np.int64),
                "features_quiet": feature[quiet_rows][:, None],
                "spikes_quiet": (rng.random(int(quiet_rows.sum())) < 0.2).astype(float),
                "session_quiet": index[quiet_rows],
                "frames_quiet": frames[quiet_rows].astype(np.int64),
                "features_quiet_all": feature[quiet_rows][:, None],
                "session_quiet_all": index[quiet_rows],
                "frames_quiet_all": frames[quiet_rows].astype(np.int64),
                "spike_frames": {s: np.sort(rng.choice(per * 3, 400, replace=False))
                                 for s in range(n_sessions)},
                "n_frames": dict.fromkeys(range(n_sessions), per * 3),
                "fps": dict.fromkeys(range(n_sessions), 100.0),
                "fit_rows": np.ones(n, dtype=bool),
                "feature_names": ["f0"], "per_session": {}}

    def test_all_four_arrays_come_back_aligned(self):
        """The null returns exactly the four statistics, each with one value per draw, so they can be
        compared draw by draw."""
        null = gating_null(self._universe(), 0, _gating(), 1.0, 6, seed=0)
        assert set(null) == {"interaction", "vocal_main", "feature_main", "difference"}
        for values in null.values():
            assert values.shape == (6,)

    def test_the_difference_is_PAIRED_draw_by_draw(self):
        """Scoring delta > beta_V against its own paired null is the correction the superseded pilot
        lacked -- it compared the two point estimates directly, which says nothing about whether the
        gap could have arisen by chance."""
        null = gating_null(self._universe(), 0, _gating(), 1.0, 8, seed=0)
        np.testing.assert_allclose(null["difference"],
                                   null["interaction"] - null["vocal_main"], atol=1e-9)

    def test_draws_differ_from_each_other(self):
        """Different shifts give different interaction values; identical draws would mean the shift is
        not applied."""
        null = gating_null(self._universe(), 0, _gating(), 1.0, 8, seed=0)
        assert np.unique(null["interaction"]).size > 1

    def test_the_same_seed_reproduces_the_null(self):
        """A fixed seed reproduces the null exactly, so a stored p-value can be regenerated."""
        first = gating_null(self._universe(), 0, _gating(), 1.0, 5, seed=7)
        second = gating_null(self._universe(), 0, _gating(), 1.0, 5, seed=7)
        np.testing.assert_allclose(first["interaction"], second["interaction"])

    def test_a_null_interaction_centres_near_zero(self):
        """A decoupled train should add nothing, which is what makes the observed value readable."""
        null = gating_null(self._universe(seed=3), 0, _gating(), 1.0, 40, seed=0)
        spread = np.std(null["interaction"])
        assert abs(np.mean(null["interaction"])) < 5.0 * spread


class TestTheQuietOnlyNullRedrawsItsRows:
    """The quiet-only null must choose its rows from each draw's SHIFTED labels -- every shifted spike
    frame plus ``quiet_zeros_per_spike`` zeros each -- exactly as the observed rows are chosen from the
    real labels. Relabelling the OBSERVED rows (chosen around the real spikes) gave the null the base
    rate as its positive fraction (0.0067 against the observed 0.024 at 1 Hz), and the quiet p came out
    too conservative (median 0.79 on null units)."""

    FPS, N_FRAMES, N_SESSIONS = 150.0, 9000, 3

    def _null_unit(self, monkeypatch, seed, rate_hz):
        """
        Description
        -----------
        Build a gating universe through ``gating_universe`` for a unit with NO feature dependence: a
        slow feature, Bernoulli spikes at ``rate_hz``, and 40 focal USVs of 0.1 s per session; every
        non-vocal frame is a quiet anchor.

        Parameters
        ----------
        monkeypatch (pytest.MonkeyPatch)
            The test's monkeypatch fixture (patches the session timebase and the USV loader).
        seed (int)
            Seed for the feature, the spikes and the USV placement.
        rate_hz (float)
            Firing rate of the unit.

        Returns
        -------
        universe (dict)
            As ``gating_universe`` returns it.
        settings (dict)
            The shipped ``vocal_gating`` block with a small fit cap, for speed.
        """
        rng = np.random.default_rng(seed)
        onsets = np.sort(rng.choice(np.arange(10, self.N_FRAMES / self.FPS - 1.0, 0.5), 40, replace=False))
        usvs = pls.DataFrame({"start": onsets, "stop": onsets + 0.1, "emitter": ["male"] * onsets.size,
                              "noise": [False] * onsets.size, "squeak": [False] * onsets.size})
        monkeypatch.setattr(gating_module, "session_timebase",
                            lambda _root, _sid: (["male", "female"], self.FPS, self.N_FRAMES))
        monkeypatch.setattr(gating_module, "load_session_usvs", lambda _root, _sid: usvs)
        vocal = np.zeros(self.N_FRAMES, dtype=bool)
        for onset in onsets:
            vocal[int(np.floor(onset * self.FPS)):int(np.ceil((onset + 0.1) * self.FPS))] = True
        per_session = {}
        for slot in range(self.N_SESSIONS):
            feature = np.convolve(rng.normal(size=self.N_FRAMES + 300), np.ones(300) / np.sqrt(300),
                                  "valid")[:self.N_FRAMES]
            per_session[f"s{slot}"] = {
                "feature_time_series": feature[:, None], "feature_names": ["f0"], "fps": self.FPS,
                "n_frames": self.N_FRAMES, "quiet": np.flatnonzero(~vocal).astype(np.int64),
                "spike_frames": np.flatnonzero(rng.random(self.N_FRAMES) < rate_hz / self.FPS)}
        settings = _gating()
        settings["max_quiet_frames_per_session"] = 1500
        unit = {"mouse_id": "male", "vocal_sessions": list(per_session)}
        return gating_universe(unit, "/data", per_session, settings, VOCAL_SETTINGS, seed=seed), settings

    def test_every_draw_keeps_the_observed_class_balance(self, monkeypatch):
        """Each null draw's quiet-only rows carry the observed positive fraction ~1 / (1 + zeros per
        spike), not the base rate a relabelling of the observed rows gave."""
        universe, settings = self._null_unit(monkeypatch, seed=0, rate_hz=1.0)
        balance = 1.0 / (1.0 + settings["quiet_zeros_per_spike"])
        assert universe["spikes_quiet"].mean() == pytest.approx(balance, rel=0.05)
        seen = []
        real_statistic = gating_module.feature_gating_statistic

        def recording_statistic(feature_vocal, feature_quiet, vocal, spikes_universe, spikes_quiet,
                                *args, **kwargs):
            """Records each draw's quiet-only positive fraction, then runs the real statistic."""
            seen.append(float(np.mean(spikes_quiet)))
            return real_statistic(feature_vocal, feature_quiet, vocal, spikes_universe, spikes_quiet,
                                  *args, **kwargs)

        monkeypatch.setattr(gating_module, "feature_gating_statistic", recording_statistic)
        gating_null(universe, 0, settings, 20.0, 5, seed=1)
        assert len(seen) == 5
        np.testing.assert_allclose(seen, balance, rtol=0.1)

    def test_the_quiet_main_effect_p_is_calibrated_at_one_hz(self, monkeypatch):
        """On units with no feature dependence the quiet-only main-effect p is uniform: its median
        over simulated null units sits near 0.5 (it was 0.78 before the redraw)."""
        n_draws, p_values = 19, []
        for seed in range(24):
            universe, settings = self._null_unit(monkeypatch, seed=seed, rate_hz=1.0)
            observed = feature_gating_statistic(
                universe["features"][:, 0], universe["features_quiet"][:, 0], universe["vocal"],
                universe["spikes"], universe["spikes_quiet"], universe["session_index"],
                universe["session_quiet"], settings, universe["fit_rows"])
            null = gating_null(universe, 0, settings, 20.0, n_draws, seed=1000 * seed)
            p_values.append((1.0 + np.sum(null["feature_main"] >= observed["feature_main"])) / (1.0 + n_draws))
        # 24 units x 19 draws: the median of a calibrated p sits near 0.5 with a standard error of
        # about 0.06 (measured 0.65 here, 0.56 at 40 x 30); the relabelled-rows null gave 0.78.
        assert 0.35 < float(np.median(p_values)) < 0.72


class TestTheAlphasAreSettings:
    """Three SEPARATE bars, none of which is the FDR level -- `significance.fdr_q` is also 0.01, and
    conflating a cohort FDR level with an uncorrected within-unit alpha is the confusion the plan
    warns against."""

    def test_all_three_are_present_and_independent(self):
        """Each of the three alphas is its own shipped key at 0.01."""
        settings = _gating()
        for key in ("screen_alpha", "interaction_gt_vocal_alpha", "feature_main_alpha"):
            assert settings[key] == 0.01

    def test_the_screen_bar_follows_its_own_alpha(self):
        """Changing ``screen_alpha`` moves the Bonferroni bar to screen_alpha / 19."""
        settings = _gating()
        settings["screen_alpha"] = 0.05
        assert settings_bonferroni_bar(settings, 19) == pytest.approx(0.05 / 19)

    def test_the_generous_main_bar_can_be_tightened_alone(self):
        """It is deliberately generous toward DETECTING a main effect, so the clean GATE label is
        reserved for convincingly-absent mains. Tightening it must not touch the other two."""
        observed = {"interaction": 50.0, "interaction_minus_vocal": 20.0, "feature_main": 6.0}
        null = np.linspace(-5.0, 5.0, 4001)
        loose = _gating()
        assert gating_verdict(observed, null, null, null, loose, 19)["label"] == "GATE+silentME"
        strict = _gating()
        strict["feature_main_alpha"] = 1e-9
        assert gating_verdict(observed, null, null, null, strict, 19)["label"] == "GATE"

    def test_the_difference_bar_can_be_relaxed_alone(self):
        """Relaxing ``interaction_gt_vocal_alpha`` alone turns ``int<vocal`` into ``GATE``, so that
        condition reads its own alpha."""
        observed = {"interaction": 50.0, "interaction_minus_vocal": 0.0, "feature_main": -3.0}
        null = np.linspace(-5.0, 5.0, 4001)
        assert gating_verdict(observed, null, null, null, _gating(), 19)["label"] == "int<vocal"
        relaxed = _gating()
        relaxed["interaction_gt_vocal_alpha"] = 0.99
        assert gating_verdict(observed, null, null, null, relaxed, 19)["label"] == "GATE"


class TestTheUnitPvalue:
    """One p per UNIT for the cohort correction, from three per FEATURE."""

    @staticmethod
    def _verdicts(rows):
        """
        Description
        -----------
        Build per-feature verdicts carrying label, margin and the two p-values the unit p combines.

        Parameters
        ----------
        rows (list[tuple[str, str, float, float, float]])
            ``(feature_name, label, sigma_margin, p_interaction, p_interaction_gt_vocal)``.

        Returns
        -------
        per_feature (dict)
            ``{feature_name: {...}}`` with those four fields.
        """
        return {name: {"label": label, "sigma_margin": margin,
                       "p_interaction": p_int, "p_interaction_gt_vocal": p_diff}
                for name, label, margin, p_int, p_diff in rows}

    def test_it_is_the_max_of_the_two_conditions(self):
        """Both must hold, so the weaker decides -- as the intersection-union test does across the analyses it combines."""
        per_feature = self._verdicts([("a", "GATE", 40.0, 5.0e-4, 3.0e-3)])
        assert gating_unit_pvalue(per_feature, _gating())["p_unit"] == pytest.approx(3.0e-3)

    def test_it_is_the_bonferroni_adjusted_best_feature(self):
        """The unit p is the smallest feature conjunction p times the number of features tested, and
        that feature is named."""
        per_feature = self._verdicts([("weak", "GATE", 6.0, 5.0e-4, 9.0e-3),
                                      ("strong", "GATE", 72.0, 5.0e-4, 5.0e-4)])
        result = gating_unit_pvalue(per_feature, _gating())
        assert result["feature"] == "strong"
        assert result["p_unit"] == pytest.approx(2 * 5.0e-4)

    def test_the_adjusted_p_is_capped_at_one(self):
        """``n_features * min p`` is capped at 1, so the adjusted value is still a p-value."""
        per_feature = self._verdicts([(f"f{i}", "ns", 0.0, 0.4, 0.4) for i in range(19)])
        assert gating_unit_pvalue(per_feature, _gating())["p_unit"] == 1.0

    def test_a_tested_unit_with_no_survivor_carries_its_real_p(self):
        """A tested unit without a surviving feature enters the cohort correction with its real
        adjusted p; it used to return NaN, which kept it out of BH's denominator."""
        per_feature = self._verdicts([("a", "int<vocal", 40.0, 5.0e-4, 1.0),
                                      ("b", "ns", 1.0, 0.3, 0.2)])
        result = gating_unit_pvalue(per_feature, _gating())
        assert result["p_unit"] == pytest.approx(min(1.0, 2 * 0.3))
        assert result["feature"] == "b"
        assert result["label"] == "ns"

    def test_only_an_untested_unit_is_nan(self):
        """NaN is reserved for a unit with no tested feature -- the one case BH must leave out."""
        result = gating_unit_pvalue({}, _gating())
        assert np.isnan(result["p_unit"])
        assert result["feature"] is None

    def test_a_lone_survivor_is_not_rejected_against_ninety_nine_tested_non_survivors(self):
        """
        Repro basis: one unit at p = 4e-4 among 100 TESTED units. With the 99 non-survivors entering
        as NaN, BH ran over the survivor alone and rejected it at q = 0.01; with their real adjusted
        p-values in the denominator (m = 100, rank-1 bar 1e-4) it is not rejected.
        """
        survivor = gating_unit_pvalue(self._verdicts([("a", "GATE", 50.0, 4.0e-4, 4.0e-4)]), _gating())
        non_survivors = [gating_unit_pvalue(self._verdicts([(f"f{i}", "ns", 1.0, 0.05 + 0.009 * unit, 0.5)
                                                            for i in range(19)]), _gating())
                         for unit in range(99)]
        p_values = np.array([survivor["p_unit"]] + [result["p_unit"] for result in non_survivors])
        assert np.all(np.isfinite(p_values))
        rejected, _threshold, n_rejected = benjamini_hochberg(p_values, 0.01)
        assert n_rejected == 0
        assert not rejected[0]

    def test_the_floor_is_flagged_because_strong_units_all_tie_there(self):
        """At 2,000 shuffles a strong feature sits at exactly 1/2001, which is the point -- it is what
        clears the Bonferroni bar -- so BH over the cohort sees ties and behaves discretely."""
        settings = _gating()
        floor = 1.0 / (settings["gating_screen_n_shuffles"] + 1.0)
        per_feature = self._verdicts([("a", "GATE", 72.0, floor, floor)])
        assert gating_unit_pvalue(per_feature, settings)["at_floor"]

    def test_a_resolved_p_is_not_flagged_as_floored(self):
        """A p above the shuffle floor is not flagged, so ``at_floor`` marks only true ties."""
        per_feature = self._verdicts([("a", "GATE", 72.0, 5.0e-4, 4.0e-3)])
        assert not gating_unit_pvalue(per_feature, _gating())["at_floor"]
