"""
@author: bartulem
Unit tests for ``usv_playpen.neural_modeling.main_neural_vocal_dispatcher``.

Coverage: the job that runs vocal occurrence, vocalization identity, nested vocal-manifold position decoding
and vocal gating for one unit. Its own logic is thin, so these
tests pin the properties that would fail SILENTLY rather than loudly -- that nulls split across spawned
worker processes come back identical to the package's serial nulls, that escalation continues a draw
sequence instead of repeating it, that the written sections follow the 2026-09-14 rulings (held-out
sessions by name, no gating decisions), and that vocal gating's on/off switch reads its threshold from settings
and skips an untuned unit without loading its kinematics.
"""

from __future__ import annotations

import copy
import json
import pathlib
import warnings

import numpy as np
import pytest

# The dispatcher imports the nested decoding's estimator, whose JAX/Optax parent sets a flag JAX has deprecated;
# that one-time DeprecationWarning fires at import, where `filterwarnings = ["error"]` would make it a
# collection error. Suppressed just for these imports, as in `test_neural_nested_decoding.py`.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from usv_playpen.neural_modeling import main_neural_vocal_dispatcher as vocal
    from usv_playpen.neural_modeling.neural_artifacts import (
        decision_key_paths,
        read_unit_artifact,
        write_unit_section,
    )
    from usv_playpen.neural_modeling.neural_nested_decoding import (
        nested_settings,
        null_draw_factory,
        reduced_predictions,
    )
    from usv_playpen.neural_modeling.neural_significance import permutation_null
    from usv_playpen.neural_modeling.neural_vocal_decoding import (
        decode_gain,
        decoding_context,
    )

SETTINGS_PATH = (pathlib.Path(__file__).resolve().parents[2] / "src" / "usv_playpen"
                 / "_parameter_settings" / "neural_modeling_settings.json")
UNIT = {"unit_id": "imec1_cl0000_ch000_good", "unit_uid": "m_20250101_imec1_cl0000_ch000_good",
        "mouse_id": "m", "rec_date": 20250101, "courtship_sessions": ["s0", "s1"],
        "vocal_sessions": ["s0", "s1"]}


def _settings() -> dict:
    with SETTINGS_PATH.open() as handle:
        return json.load(handle)


def _quiet(*_args, **_kwargs) -> None:
    return None


class TestParallelNullsAreExact:
    """A null split across workers must be the SAME null, value for value, or a cohort run would report
    p-values no serial run could reproduce."""

    def test_the_vocalization_identity_null_across_workers_equals_the_serial_permutation_null(self):
        generator = np.random.default_rng(3)
        n_sessions, per_session = 3, 120
        positions = generator.random((n_sessions * per_session, 2))
        sessions = np.repeat(np.arange(n_sessions), per_session)
        shape = np.cos(2 * np.pi * positions[:, 0]) + np.cos(2 * np.pi * positions[:, 1])
        counts = generator.poisson(np.exp(1.4 * shape - 1.4 * shape.max() + np.log(3.0))).astype(float)
        decoding = copy.deepcopy(_settings()["vocal_decoding"])
        decoding["tuning_surface"].update({"grid_n": 24, "kde_bandwidth": 0.10})
        folds = decoding_context(positions, sessions, decoding)

        serial = permutation_null(lambda permutation: decode_gain(counts, folds, decoding, permutation)["gain"],
                                  sessions, 5, 11)
        state = {"counts": counts, "folds": folds, "decoding": decoding, "session_index": sessions, "seed": 11}
        with vocal.spawn_pool(2, state) as pool:
            parallel = vocal.parallel_draw_more(pool, vocal.vocalization_identity_null_chunk, 2)(5)
        np.testing.assert_array_equal(parallel, serial)

    def test_the_nested_decoding_null_across_workers_equals_the_package_factory_and_continues(self):
        """Two escalation steps of 2 draws must reproduce the factory's first 4 -- continuing the
        sequence, not restarting it."""
        generator = np.random.default_rng(5)
        n_sessions, per_session, n_features, n_lags = 3, 60, 2, 3
        n = n_sessions * per_session
        behaviour = generator.normal(size=(n, n_features * n_lags))
        angle = behaviour[:, n_lags - 1] + 0.05 * generator.normal(size=n)
        sessions = np.repeat(np.arange(n_sessions), per_session)
        duration, width, guard = 300.0, 0.1, 20.0
        edges = np.concatenate([np.sort(generator.uniform(5.0, duration - 5.0, per_session))
                                for _ in range(n_sessions)])
        spikes = {slot: np.sort(generator.uniform(0.0, duration, 800)) for slot in range(n_sessions)}
        durations = dict.fromkeys(range(n_sessions), duration)
        design = {"behaviour": behaviour, "neural": generator.normal(size=(n, 1)),
                  "positions": np.column_stack([(0.15 * angle) % 1.0, (0.07 * angle) % 1.0]),
                  "session_index": sessions,
                  "region_labels": np.tile(np.arange(6, dtype=float), n // 6 + 1)[:n],
                  "feature_names": [f"f{i}" for i in range(n_features)], "kept": np.arange(n),
                  "per_session": {}}
        nested = nested_settings(_settings())
        nested["min_region_events"] = 2
        reduced = reduced_predictions(design, nested, n_lags)

        serial = null_draw_factory(design, nested, n_lags, edges, spikes, durations, width, guard, reduced, 13,
                                   nested["sigma_floor"])(4)
        state = {"design": design, "settings": nested, "n_lags": n_lags, "window_edges": edges,
                 "spike_seconds": spikes, "durations": durations, "width": width, "seed": 13,
                 "guard_seconds": guard, "sigma_floor": nested["sigma_floor"], "reduced_predicted": reduced}
        with vocal.spawn_pool(2, state) as pool:
            draw_more = vocal.parallel_draw_more(pool, vocal.nested_decoding_null_chunk, 2)
            stepwise = np.concatenate([draw_more(2), draw_more(2)])
        np.testing.assert_array_equal(stepwise, serial)


class TestTheWrittenSections:
    """The sections follow the 2026-09-14 rulings."""

    def test_the_nested_decoding_names_the_held_out_session_rather_than_its_slot(self):
        scores = {"added": 0.01, "reduced": -3.70, "full": -3.69, "n_events": 30, "n_regions_scored": 7,
                  "min_leave_one_fold_out": 0.005,
                  "per_fold": [{"held_out": 1, "n_events": 10, "added": 0.02},
                               {"held_out": 0, "n_events": 20, "added": 0.00}]}
        control = {"features": ["nose-nose", "other.speed"], "held_out_day": "20250919",
                   "settings_sha256": "abc123"}
        payload = vocal.nested_decoding_payload(scores, ["s_a", "s_b"], np.zeros(10), 0.09, False,
                                                -3.74, control)
        assert [fold["held_out_session"] for fold in payload["per_fold"]] == ["s_b", "s_a"]
        assert "held_out" not in payload["per_fold"][0]
        # the behaviour model's own improvement over no behaviour, stored so the cohort step can judge it
        assert payload["behaviour_over_no_behaviour"] == pytest.approx(-3.70 - -3.74)
        # WHICH control produced the reduced model. It is an external artifact now, not a refit, so an
        # added score that cannot be traced to the control that made it is not reproducible.
        assert payload["behaviour_control"]["held_out_day"] == "20250919"
        assert payload["behaviour_control"]["features"] == ["nose-nose", "other.speed"]
        assert payload["behaviour_control"]["settings_sha256"] == "abc123"

    def test_vocal_gating_writes_statistics_and_p_values_but_no_decisions(self, tmp_path):
        """Vocal gating's label, the borderline flag and the unit's best label are uncorrected by construction; the
        artifact writer refuses them, so the payload must not carry them at any depth."""
        draws = np.zeros(4)
        observed = {"f0": {"feature_main": -3.0, "vocal_main": 400.0, "interaction": 560.0,
                           "additive": 420.0, "baseline": 9000.0, "quiet_baseline": 700.0,
                           "interaction_minus_vocal": 160.0, "gating_sign": 1.0}}
        nulls = {"f0": {"interaction": draws, "vocal_main": draws, "feature_main": draws, "difference": draws}}
        verdicts = {"f0": {"label": "GATE", "p_interaction": 5e-4, "p_interaction_gt_vocal": 5e-4,
                           "p_feature_main_quiet": 0.4, "sigma_margin": 12.0, "borderline": False,
                           "bonferroni_bar": 5e-4}}
        unit_p = {"p_unit": 5e-4, "feature": "f0", "label": "GATE", "at_floor": True}
        payload = vocal.vocal_gating_payload(["f0", "f1"], observed, nulls, verdicts, unit_p,
                                       {"f1": "not_testable"}, 2000)
        assert decision_key_paths(payload) == []
        assert payload["per_feature"]["f1"] == {"tested": False, "reason": "not_testable"}
        assert payload["per_feature"]["f0"]["p_interaction"] == 5e-4
        assert payload["p_unit_feature"] == "f0"
        write_unit_section(str(tmp_path), UNIT, "vocal_gating", payload, _settings())

    def test_the_vocalization_identity_section_derives_its_counts_from_the_null_and_events(self):
        identity = {"gain": 0.05}
        events = {"counts": np.array([1.0, 2.0, 3.0, 4.0]), "session_index": np.array([0, 0, 1, 1]),
                  "session_ids": ["a", "b"]}
        payload = vocal.vocalization_identity_payload(identity, np.array([0.0, 0.06, 0.01]), 0.5, False, events)
        assert payload["exceed_count"] == 1
        assert payload["mid_p"] == pytest.approx(1.5 / 4.0)
        assert payload["per_session_counts"] == {"a": 3, "b": 7}
        assert payload["n_events"] == 4


class TestTheGatingSwitch:
    """Vocal gating runs only on a unit whose vocal occurrence or vocalization identity shows tuning, at a
    threshold read from settings."""

    def test_the_threshold_is_a_setting(self):
        """Ruled 2026-09-14: 0.01, the level every other per-unit threshold in vocal gating already uses."""
        assert _settings()["vocal_gating"]["vocal_tuning_alpha"] == 0.01

    def test_an_untuned_unit_is_skipped_without_loading_its_kinematics(self, tmp_path, monkeypatch):
        def refuse(*_args, **_kwargs):
            msg = "an untuned unit must not load kinematics"
            raise AssertionError(msg)

        monkeypatch.setattr(vocal, "assemble_unit_sessions", refuse)
        payload = vocal.run_vocal_gating(UNIT, _settings(), "unused", str(tmp_path), 1, 0.2, 0.3, _quiet)
        assert payload == {"ran": False, "reason": "no_vocal_tuning"}
        assert read_unit_artifact(str(tmp_path), UNIT["unit_uid"])["vocal_gating"] == payload

    def test_the_threshold_is_read_not_assumed(self, tmp_path, monkeypatch):
        """At 0.5 the same p-values count as tuned, so the switch opens and reaches for the kinematics."""

        class Reached(Exception):
            pass

        def reached(*_args, **_kwargs):
            raise Reached

        settings = _settings()
        settings["vocal_gating"]["vocal_tuning_alpha"] = 0.5
        monkeypatch.setattr(vocal, "assemble_unit_sessions", reached)
        with pytest.raises(Reached):
            vocal.run_vocal_gating(UNIT, settings, "unused", str(tmp_path), 1, 0.2, 0.3, _quiet)

    def test_vocal_gating_alone_refuses_without_occurrence_and_identity_results(self, tmp_path):
        with pytest.raises(ValueError, match="run vocal_occurrence first"):
            vocal.run_vocal(UNIT, _settings(), "unused", str(tmp_path), 1, steps=("vocal_gating",),
                            message_output=_quiet)

    def test_a_threshold_too_few_draws_can_clear_is_refused(self, tmp_path):
        """Found on a real smoke run: at 20 draws every p-value floors at 1/21 = 0.048, above the 0.01
        threshold, so a strongly tuned unit read as untuned. Silently skipping every unit is refused."""
        write_unit_section(str(tmp_path), UNIT, "vocal_occurrence", {"p": 1 / 21, "n_shuffles": 20}, _settings())
        write_unit_section(str(tmp_path), UNIT, "vocalization_identity", {"p": 1 / 21, "n_permutations": 20},
                           _settings())
        with pytest.raises(ValueError, match="cannot fall below"):
            vocal.run_vocal(UNIT, _settings(), "unused", str(tmp_path), 1, steps=("vocal_gating",),
                            message_output=_quiet)

    def test_the_fewest_draws_that_can_clear_the_threshold_are_accepted(self):
        """1/(n+1) must be strictly below alpha: at 0.01 that is 100 draws, and 99 is refused."""
        vocal.check_tuning_threshold_is_reachable(0.01, {"vocal_occurrence": 100, "vocalization_identity": 1000})
        with pytest.raises(ValueError, match="at least 100 draws"):
            vocal.check_tuning_threshold_is_reachable(0.01, {"vocal_occurrence": 99})

    def test_an_unknown_step_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="unknown vocal step"):
            vocal.run_vocal(UNIT, _settings(), "unused", str(tmp_path), 1, steps=("not_an_analysis",),
                            message_output=_quiet)
