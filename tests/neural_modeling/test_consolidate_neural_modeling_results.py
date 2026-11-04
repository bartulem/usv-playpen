"""
@author: bartulem
Unit tests for ``usv_playpen.neural_modeling.consolidate_neural_modeling_results``.

Coverage: reading one unit's p-values with every way a test can be missing, the false-discovery
procedures, the category rules (including that a missing test is neither a pass nor a failure and that
Transformation is the conjunction test), and the cohort consolidation -- correction within each area or
pooled, the conjunction as the largest component p-value, the exclusion of single-session units, the
report's floor requirements and exclusion counts, a group left with nothing to correct -- and the command
line, which writes the cohort table and report and warns when the unit files disagree on their settings.
"""

from __future__ import annotations

import pickle
import sys

import numpy as np
import polars as pls
import pytest

from usv_playpen.neural_modeling import (
    consolidate_neural_modeling_results as consolidation,
)
from usv_playpen.neural_modeling.consolidate_neural_modeling_results import (
    CATEGORIES,
    TESTS,
    classify_unit,
    consolidate,
    correlation_matrix,
    fdr_reject,
    null_z,
    unit_row,
)
from usv_playpen.neural_modeling.neural_significance import benjamini_hochberg


def _record(uid: str, area: str = "PAG", single: bool = False) -> dict:
    """
    Description
    -----------
    A cohort-manifest record with the fields the consolidation reads.

    Parameters
    ----------
    uid (str)
        The unit's identifier.
    area (str)
        Its brain area.
    single (bool)
        Whether it was recorded in one session (tested on blocks).

    Returns
    -------
    record (dict)
        The manifest record.
    """
    return {"unit_uid": uid, "brain_area": area, "block_split": {"n_blocks": 5} if single else None,
            "n_courtship_sessions": 1 if single else 3, "n_vocal_sessions": 3}


def _artifact(p: float = 1e-3, slope: float = 0.4, gating=None, null_draws: int = 1000) -> dict:
    """
    Description
    -----------
    A result file in which every test ran with the same p-value.

    Parameters
    ----------
    p (float)
        The p-value of every test.
    slope (float)
        The transfer slope.
    gating (dict | None)
        The gating section; ``None`` writes it as not run.
    null_draws (int)
        Length of the stored encoding nulls.

    Returns
    -------
    artifact (dict)
        The result file.
    """
    null = np.linspace(-1.0, 1.0, null_draws)
    return {"kinematic_encoding": {"quiet_p": p, "transfer_p": p, "transfer_slope": slope,
                                   "quiet_score": 3.0, "transfer_score": 2.0,
                                   "quiet_null": null, "transfer_null": null},
            "vocal_occurrence": {"p": p, "z": 9.0, "gain": 0.3, "response_sign": "elevated", "n_shuffles": 1000},
            "vocalization_identity_manifold": {"p": p, "z": 8.0, "gain": 0.05, "target": "manifold",
                                               "n_permutations": 1000},
            "nested_vocal_decoding_manifold": {"p": p, "z": 7.0, "added": 0.02, "n_permutations": 1000,
                                               "behaviour_over_no_behaviour": 0.01},
            # The categorical target carries its own, distinct numbers, so a test reading the wrong
            # target would see them.
            "vocalization_identity_multinomial": {"p": 0.5, "z": 0.1, "gain": 0.001, "target": "multinomial",
                                                  "n_permutations": 5000},
            "nested_vocal_decoding_multinomial": {"p": 0.6, "z": -0.2, "added": -0.001, "n_permutations": 1000,
                                                  "behaviour_over_no_behaviour": 0.03},
            "nested_vocal_decoding_onset": {"p": p, "z": 6.0, "added": 0.015, "n_shuffles": 1000,
                                            "behaviour_over_no_behaviour": 0.02, "target": "onset"},
            "vocal_gating": gating if gating is not None else {"ran": False, "reason": "no_vocal_tuning"}}


SETTINGS = {"inference": {"fdr_q": 0.05, "fdr_method": "bh", "include_single_session_units": False},
            "cohort": {"pool_areas": False}, "vocalizations": {"vocal_target": "manifold"}}


class TestUnitRow:
    """Reading one unit's p-values, and every way a test can be missing."""

    def test_a_unit_with_every_test_reads_every_p_value(self):
        """Each test's p-value is read from its own section, the encoding z-scores from the stored nulls."""
        row = unit_row(_record("u"), _artifact(p=2e-3), "manifold")
        assert all(row[f"p_{test}"] == pytest.approx(2e-3) for test in TESTS if test != "gating")
        assert row["z_quiet"] == pytest.approx(null_z(3.0, np.linspace(-1, 1, 1000)))
        assert row["reason_gating"] == "no_vocal_tuning"
        assert np.isnan(row["p_gating"])

    def test_an_unrun_unit_is_missing_everywhere_never_failed(self):
        """A unit with no result file has a NaN p-value and a ``not_run`` reason for every test."""
        row = unit_row(_record("u"), {}, "manifold")
        assert all(np.isnan(row[f"p_{test}"]) for test in TESTS)
        assert all(row[f"reason_{test}"] == "not_run" for test in TESTS)
        assert not row["has_results"]

    def test_a_section_written_as_not_run_carries_its_reason(self):
        """A vocal section written as not vocal-testable is missing, with that reason."""
        artifact = _artifact()
        artifact["vocal_occurrence"] = {"ran": False, "reason": "not_vocal_testable", "n_vocal_sessions": 1}
        row = unit_row(_record("u"), artifact, "manifold")
        assert np.isnan(row["p_occurrence"])
        assert row["reason_occurrence"] == "not_vocal_testable"

    @pytest.mark.parametrize(("flags", "reason"), [({"no_scoreable_fold": True}, "no_scoreable_fold"),
                                                   ({"no_model": True}, "no_model")])
    def test_an_encoding_without_a_p_value_names_why(self, flags, reason):
        """A NaN transfer p from a unit with no scoreable fold or no model is missing with that reason."""
        artifact = _artifact()
        artifact["kinematic_encoding"].update({"transfer_p": float("nan"), **flags})
        assert unit_row(_record("u"), artifact, "manifold")["reason_transfer"] == reason

    def test_the_tested_target_supplies_identity_and_nested_and_both_are_described(self):
        """``target`` picks which sections are tested; both targets' numbers are copied as descriptors."""
        artifact = _artifact(p=2e-3)
        manifold = unit_row(_record("u"), artifact, "manifold")
        multinomial = unit_row(_record("u"), artifact, "multinomial")
        assert manifold["p_identity"] == pytest.approx(2e-3)
        assert manifold["p_nested"] == pytest.approx(2e-3)
        assert manifold["behaviour_gain"] == pytest.approx(0.01)
        assert multinomial["p_identity"] == pytest.approx(0.5)
        assert multinomial["p_nested"] == pytest.approx(0.6)
        assert multinomial["behaviour_gain"] == pytest.approx(0.03)
        assert multinomial["identity_target"] == "multinomial"
        for row in (manifold, multinomial):
            assert row["p_identity_manifold"] == pytest.approx(2e-3)
            assert row["p_identity_multinomial"] == pytest.approx(0.5)
            assert row["nested_effect_multinomial"] == pytest.approx(-0.001)
            assert row["behaviour_gain_multinomial"] == pytest.approx(0.03)

    def test_null_z_is_undefined_for_a_constant_or_absent_null(self):
        """A z-score needs a null with spread."""
        assert np.isnan(null_z(1.0, None))
        assert np.isnan(null_z(1.0, np.ones(10)))


class TestFdrReject:
    """The false-discovery procedures."""

    def test_bh_is_the_shared_benjamini_hochberg(self):
        """``'bh'`` gives exactly the shared procedure's rejections, and NaN p-values are never counted."""
        p_values = np.array([1e-4, 0.004, 0.02, 0.5, np.nan])
        expected = benjamini_hochberg(p_values, 0.05)[0]
        np.testing.assert_array_equal(fdr_reject(p_values, 0.05, "bh"), expected)
        assert not fdr_reject(p_values, 0.05, "bh")[-1]

    def test_by_is_stricter(self):
        """Benjamini-Yekutieli divides q by the harmonic number of m, so it rejects no more than BH."""
        p_values = np.array([1e-4, 0.004, 0.012, 0.02, 0.5])
        assert fdr_reject(p_values, 0.05, "by").sum() < fdr_reject(p_values, 0.05, "bh").sum()

    def test_an_unknown_method_is_refused(self):
        """A misspelled method must raise naming the key, not fall back to BH."""
        with pytest.raises(ValueError, match="fdr_method"):
            fdr_reject(np.array([0.01]), 0.05, "BH")


class TestClassifyUnit:
    """The six categories, first match wins."""

    @staticmethod
    def _passes(**states) -> dict:
        """
        Description
        -----------
        Every test failed, then the given overrides.

        Parameters
        ----------
        **states (bool | None)
            ``test=state`` overrides.

        Returns
        -------
        passes (dict)
            ``{test: state}``.
        """
        return {**dict.fromkeys(TESTS, False), **states}

    def test_transformation_is_the_conjunction_with_a_positive_slope(self):
        """The conjunction test decides Transformation; a negative slope makes the same unit Regime-switching."""
        passes = self._passes(quiet=True, transfer=True, identity=True, nested=True)
        assert classify_unit(passes, 0.4, True)[0] == "Transformation"
        assert classify_unit(passes, -0.4, True)[0] == "Regime-switching"

    def test_passing_each_test_but_not_the_conjunction_is_flagged(self):
        """Four separate passes without the conjunction fall through to Vocal and are flagged."""
        passes = self._passes(quiet=True, transfer=True, identity=True, nested=True)
        category, modifiers = classify_unit(passes, 0.4, False)
        assert category == "Vocal"
        assert modifiers["separate_pass_conjunction_fail"]

    def test_relay_needs_a_failed_nested_test_not_a_missing_one(self):
        """A missing nested test is neither a pass nor a failure, so the unit is Vocal, not Relay."""
        relay = self._passes(quiet=True, transfer=True, occurrence=True, nested=False)
        missing = self._passes(quiet=True, transfer=True, occurrence=True, nested=None)
        assert classify_unit(relay, 0.4, False)[0] == "Relay"
        assert classify_unit(missing, 0.4, False)[0] == "Vocal"

    @pytest.mark.parametrize(("states", "category"), [({"nested": True}, "Vocal"), ({"quiet": True}, "Kinematic"),
                                                      ({}, "None")])
    def test_the_remaining_rules(self, states, category):
        """Nested alone is Vocal, quiet alone Kinematic, nothing None."""
        assert classify_unit(self._passes(**states), float("nan"), False)[0] == category

    def test_every_category_is_named(self):
        """The rules use exactly the six declared categories."""
        assert CATEGORIES == ("Transformation", "Relay", "Regime-switching", "Vocal", "Kinematic", "None")


class TestConsolidate:
    """The cohort consolidation."""

    def test_each_area_is_corrected_on_its_own(self):
        """A PAG unit at p = 0.004 passes among 5 PAG units; adding 200 null VTA units changes nothing for it,
        while pooling the areas makes it fail."""
        cohort = [_record(f"pag{i}") for i in range(5)] + [_record(f"vta{i}", "VTA") for i in range(200)]
        artifacts = {f"pag{i}": _artifact(p=0.004) for i in range(5)}
        artifacts.update({f"vta{i}": _artifact(p=0.9) for i in range(200)})
        table, report = consolidate(cohort, artifacts, SETTINGS)
        assert table.filter(table["unit_uid"] == "pag0")["pass_quiet"][0] is True
        assert set(report) >= {"PAG", "VTA"}
        pooled, _ = consolidate(cohort, artifacts, {**SETTINGS, "cohort": {"pool_areas": True}})
        assert pooled.filter(pooled["unit_uid"] == "pag0")["pass_quiet"][0] is False

    def test_the_conjunction_is_the_largest_component(self):
        """The conjunction p-value is the largest of quiet, transfer, identity and nested; occurrence is not in it."""
        artifact = _artifact(p=1e-3)
        artifact["vocalization_identity_manifold"]["p"] = 0.03
        artifact["vocal_occurrence"]["p"] = 0.9
        table, _ = consolidate([_record("u")], {"u": artifact}, SETTINGS)
        assert table["p_conjunction"][0] == pytest.approx(0.03)

    def test_switching_the_target_needs_only_the_consolidation(self):
        """The same result files classify differently under the other target: here the categorical
        identity and nested tests fail, so the unit is no longer a Transformation."""
        artifact = _artifact(p=1e-3)
        manifold, _ = consolidate([_record("u")], {"u": artifact}, SETTINGS)
        multinomial, _ = consolidate([_record("u")], {"u": artifact},
                                     {**SETTINGS, "vocalizations": {"vocal_target": "multinomial"}})
        assert manifold["category"][0] == "Transformation"
        # The categorical nested test added -0.001: a non-positive effect fails the conjunction outright.
        assert multinomial["p_conjunction"][0] == pytest.approx(1.0)
        assert multinomial["category"][0] != "Transformation"

    def test_a_decoding_with_a_non_positive_effect_never_passes(self):
        """A floored p with a gain at or below zero is a fail for identity, nested and nested onset alike,
        and it fails the conjunction; occurrence follows the same rule."""
        artifact = _artifact(p=1e-3)
        artifact["vocalization_identity_manifold"]["gain"] = -0.02
        artifact["nested_vocal_decoding_onset"]["added"] = 0.0
        table, _ = consolidate([_record("u")], {"u": artifact}, SETTINGS)
        assert table["pass_identity"][0] is False
        assert table["pass_nested_onset"][0] is False
        assert table["pass_nested"][0] is True
        assert table["p_conjunction"][0] == pytest.approx(1.0)
        assert table["category"][0] != "Transformation"

    def test_a_best_screened_only_unit_enters_quiet_as_a_fail_and_transfer_as_untested(self):
        """Whatever descriptive p-values a no-model section carries, the corrections read quiet 1.0 and
        transfer NaN, so a maximum over the screened features never enters a correction."""
        artifact = _artifact(p=1e-3)
        artifact["kinematic_encoding"].update({"no_model": True, "best_screened_only": True})
        table, _ = consolidate([_record("u")], {"u": artifact}, SETTINGS)
        assert table["p_quiet"][0] == 1.0
        assert np.isnan(table["p_transfer"][0])
        assert table["reason_transfer"][0] == "no_model"

    def test_a_group_without_any_nested_result_still_reports(self):
        """When no unit in a group carries a nested result the modifier column is Null-typed; the report
        must still be built rather than fail on the comparison."""
        artifact = _artifact(p=1e-3)
        for name in ("nested_vocal_decoding_manifold", "nested_vocal_decoding_multinomial",
                     "nested_vocal_decoding_onset"):
            artifact[name] = {"ran": False, "reason": "not_vocal_testable"}
        _, report = consolidate([_record("u")], {"u": artifact}, SETTINGS)
        assert report["PAG"]["transformation_where_behaviour_uninformative"] == 0

    def test_single_session_units_are_listed_but_not_corrected_by_default(self):
        """Excluded by default, admitted by the switch; listed either way."""
        cohort = [_record("multi"), _record("block", single=True)]
        artifacts = {"multi": _artifact(), "block": _artifact()}
        table, report = consolidate(cohort, artifacts, SETTINGS)
        assert report["excluded_single_session"] == ["block"]
        assert table.filter(table["unit_uid"] == "block")["category"][0] is None
        admitted = {**SETTINGS, "inference": {**SETTINGS["inference"], "include_single_session_units": True}}
        table, report = consolidate(cohort, artifacts, admitted)
        assert report["excluded_single_session"] == []
        assert table.filter(table["unit_uid"] == "block")["category"][0] == "Transformation"

    def test_the_report_carries_the_floor_and_the_exclusions(self):
        """Each test reports its floor requirement at the draws behind it, and why units were excluded."""
        cohort = [_record(f"u{i}") for i in range(4)]
        artifacts = {f"u{i}": _artifact() for i in range(3)}
        _table, report = consolidate(cohort, artifacts, SETTINGS)
        quiet = report["PAG"]["tests"]["quiet"]
        assert quiet["n_tested"] == 3
        assert quiet["excluded"] == {"not_run": 1}
        assert quiet["floor_requirement"] == pytest.approx(3 / (0.05 * 1001))
        assert report["PAG"]["tests"]["gating"]["excluded"] == {"no_vocal_tuning": 3, "not_run": 1}
        assert report["PAG"]["tests"]["gating"]["floor_requirement"] is None
        assert sum(report["PAG"]["categories"].values()) == 4


class TestCorrelationMatrix:
    """The cross-test correlation of z-scores."""

    def test_matches_numpy_and_leaves_a_constant_column_undefined(self):
        """Varying columns match ``np.corrcoef``; a column with no spread gives NaN rather than a warning."""
        generator = np.random.default_rng(0)
        values = generator.normal(size=(50, 3))
        np.testing.assert_allclose(correlation_matrix(values), np.corrcoef(values, rowvar=False), atol=1e-12)
        values[:, 1] = 2.0
        result = correlation_matrix(values)
        assert np.isnan(result[1]).all()
        assert result[0, 2] == pytest.approx(np.corrcoef(values[:, 0], values[:, 2])[0, 1])


    def test_the_gating_floor_is_the_bonferroni_adjusted_one(self):
        """Gating p-values floor at F / (N + 1), so its floor requirement is m * that floor / q."""
        cohort = [_record(f"u{i}") for i in range(4)]
        gating = {"ran": True, "p_unit": 19 / 2001, "p_unit_at_floor": True, "p_unit_feature": "head-head",
                  "n_draws": 2000}
        artifacts = {f"u{i}": _artifact(gating=dict(gating)) for i in range(4)}
        _table, report = consolidate(cohort, artifacts, SETTINGS)
        entry = report["PAG"]["tests"]["gating"]
        assert entry["floor_p"] == pytest.approx(19 / 2001)
        assert entry["floor_requirement"] == pytest.approx(4 * (19 / 2001) / 0.05)


class TestBehaviourModifier:
    """Units whose day the behaviour control does not help are kept, flagged and counted."""

    def test_an_uninformative_control_is_flagged_not_dropped(self):
        """A transformation unit with a non-positive behaviour gain stays a Transformation, is flagged, and is
        counted in the report."""
        informative, uninformative = _artifact(), _artifact()
        uninformative["nested_vocal_decoding_manifold"]["behaviour_over_no_behaviour"] = -0.002
        table, report = consolidate([_record("a"), _record("b")], {"a": informative, "b": uninformative},
                                    SETTINGS)
        assert table["category"].to_list() == ["Transformation", "Transformation"]
        assert table["behaviour_informative"].to_list() == [True, False]
        assert report["PAG"]["transformation_where_behaviour_uninformative"] == 1


class TestGapsAndCommandLine:
    """The less common paths of the consolidation and its command line."""

    def test_a_section_that_ran_without_a_p_value_names_why(self):
        """A test that ran but stored no p-value is missing with the reason ``no_p_value``, never failed."""
        artifact = _artifact()
        artifact["vocal_occurrence"]["p"] = None
        artifact["vocal_gating"] = {"p_unit": None}
        row = unit_row(_record("u0"), artifact, "manifold")
        assert np.isnan(row["p_occurrence"])
        assert row["reason_occurrence"] == "no_p_value"
        assert row["reason_gating"] == "no_p_value"
        assert row["gating_feature"] is None

    def test_an_encoding_without_stored_nulls_still_consolidates(self):
        """Missing encoding nulls leave the floor requirement without a draw count, not a crash."""
        artifact = _artifact()
        artifact["kinematic_encoding"]["quiet_null"] = None
        artifact["kinematic_encoding"]["transfer_null"] = np.zeros(0)
        table, _report = consolidate([_record("u0")], {"u0": artifact}, SETTINGS)
        assert table.height == 1

    def test_a_group_left_with_nothing_to_correct_is_not_reported(self):
        """An area whose only units are excluded single-session units has no correction to report."""
        cohort = [_record("s0", single=True), _record("s1", single=True)]
        _table, report = consolidate(cohort, {"s0": _artifact(), "s1": _artifact()}, SETTINGS)
        assert "PAG" not in report

    def test_the_command_line_writes_the_table_and_report(self, tmp_path, monkeypatch, capsys):
        """The command reads every unit's file, writes the table and the report, and warns when the unit
        files were written under different settings."""
        cohort = [_record("u0"), _record("u1")]
        artifacts = {"u0": {**_artifact(), "provenance": {"settings_sha256": "a" * 64}},
                     "u1": {**_artifact(p=0.5), "provenance": {"settings_sha256": "b" * 64}}}
        monkeypatch.setattr(consolidation, "build_cohort_manifest", lambda _settings: cohort)
        monkeypatch.setattr(consolidation, "read_unit_artifact", lambda _directory, uid: artifacts[uid])
        monkeypatch.setattr(consolidation, "load_settings",
                            lambda _path: {**SETTINGS, "nested_vocal_decoding": {}, "x": 1})
        monkeypatch.setattr(consolidation, "build_provenance", lambda _settings: {"settings_sha256": "c" * 64})
        monkeypatch.setattr(sys, "argv", ["consolidate", "--results-directory", str(tmp_path),
                                          "--output-directory", str(tmp_path / "out")])
        assert consolidation.main() == 0
        assert (tmp_path / "out" / "neural_cohort_table.csv").exists()
        with (tmp_path / "out" / "neural_cohort_report.pkl").open("rb") as handle:
            stored = pickle.load(handle)
        assert stored["unit_settings_sha256"] == ["a" * 64, "b" * 64]
        assert "different settings" in capsys.readouterr().out


class TestNestedOnset:
    """The nested onset test is read, corrected and reported, but stays out of the conjunction."""

    def test_the_onset_section_feeds_its_own_test(self):
        """``p_nested_onset`` and its effect come from ``nested_vocal_decoding_onset``; a missing section
        is ``not_run``."""
        row = unit_row(_record("u"), _artifact(p=2e-3), "manifold")
        assert row["p_nested_onset"] == pytest.approx(2e-3)
        assert row["z_nested_onset"] == 6.0
        assert row["nested_onset_effect"] == 0.015
        artifact = _artifact()
        del artifact["nested_vocal_decoding_onset"]
        row = unit_row(_record("u"), artifact, "manifold")
        assert np.isnan(row["p_nested_onset"])
        assert row["reason_nested_onset"] == "not_run"

    def test_it_is_corrected_and_a_modifier_but_not_part_of_the_conjunction(self):
        """A unit failing only the onset nested test is still a Transformation; the test's pass state is
        corrected per group and carried as the ``nested_onset`` modifier."""
        strong = _artifact(p=1e-3)
        strong["nested_vocal_decoding_onset"]["p"] = 0.9
        table, report = consolidate([_record("u")], {"u": strong}, SETTINGS)
        row = table.row(0, named=True)
        assert row["category"] == "Transformation"
        assert row["pass_nested_onset"] is False
        assert row["nested_onset"] is False
        assert report["PAG"]["tests"]["nested_onset"]["n_tested"] == 1
        assert report["PAG"]["tests"]["nested_onset"]["n_draws"] == 1000
        assert "nested_onset" in TESTS


class TestStaleSections:
    """Sections written under other settings than the consolidation's are refused, not silently mixed."""

    def test_a_section_under_another_hash_is_listed(self):
        """A section whose provenance hash differs, or a section without provenance in a file that records
        per-section provenance, is stale; a file without per-section provenance is not judged."""
        settings = {"x": 1}
        current = consolidation.compute_settings_sha256(settings)
        artifacts = {"u0": {"identity": {}, "vocal_occurrence": {}, "vocal_gating": {},
                            "section_provenance": {"vocal_occurrence": {"settings_sha256": current},
                                                   "vocal_gating": {"settings_sha256": "0" * 64}}},
                     "u1": {"identity": {}, "vocal_occurrence": {}},
                     "u2": {}}
        assert consolidation.stale_sections(artifacts, settings) == {"u0": ["vocal_gating"]}


class TestDescriptors:
    """The cohort table carries the descriptors the figures read beside the p-values."""

    def test_identity_draws_and_model_descriptors_are_columns(self):
        """Probe and date come from the record; draw counts from the stored nulls and sections; the
        gating z is the winning feature's sigma margin."""
        artifact = _artifact(p=1e-3, gating={"ran": True, "p_unit": 1e-3, "p_unit_feature": "f0",
                                             "p_unit_at_floor": False, "n_features_tested": 3,
                                             "per_feature": {"f0": {"sigma_margin": 7.5}}})
        artifact["kinematic_encoding"].update({"selected": ["head-head", "self.speed"],
                                               "spike_counts_per_session": {"s0": {"spikes": 100,
                                                                                    "quiet_spikes": 60}}})
        record = {**_record("u"), "unit_id": "imec1_cl0401_ch220_good", "mouse_id": "m1", "rec_date": 20250919}
        table, _ = consolidate([record], {"u": artifact}, SETTINGS)
        assert table["probe"][0] == "imec1"
        assert table["rec_date"][0] == 20250919
        assert table["n_selected"][0] == 2
        assert table["selected_features"][0] == "head-head+self.speed"
        assert table["n_draws_quiet"][0] == 1000
        assert table["n_draws_occurrence"][0] == 1000
        assert table["n_spikes_total"][0] == 100
        assert table["z_gating"][0] == 7.5
        assert table["gating_n_features_tested"][0] == 3


class TestStaleSectionsOnTheCommandLine:
    """The command refuses stale sections by default and excludes them only when told to."""

    def _run(self, tmp_path, monkeypatch, extra: list) -> int:
        """
        Description
        -----------
        Run the command on one unit whose gating section was written under other settings.

        Parameters
        ----------
        tmp_path (pathlib.Path)
            Results and output directory.
        monkeypatch (pytest.MonkeyPatch)
            Installs the stubs.
        extra (list)
            Extra command-line arguments.

        Returns
        -------
        status (int)
            The command's exit status.
        """
        settings = {**SETTINGS, "nested_vocal_decoding": {}}
        current = consolidation.compute_settings_sha256(settings)
        artifact = _artifact(p=1e-3, gating={"ran": True, "p_unit": 1e-3, "p_unit_feature": "f0",
                                             "p_unit_at_floor": False})
        artifact["section_provenance"] = {name: {"settings_sha256": current} for name in artifact}
        artifact["section_provenance"]["vocal_gating"] = {"settings_sha256": "0" * 64}
        self.artifact = artifact
        monkeypatch.setattr(consolidation, "build_cohort_manifest", lambda _settings: [_record("u")])
        monkeypatch.setattr(consolidation, "read_unit_artifact", lambda _directory, _uid: artifact)
        monkeypatch.setattr(consolidation, "load_settings", lambda _path: settings)
        monkeypatch.setattr(consolidation, "build_provenance", lambda _settings: {"settings_sha256": current})
        monkeypatch.setattr(sys, "argv", ["consolidate", "--results-directory", str(tmp_path),
                                          "--output-directory", str(tmp_path / "out"), *extra])
        return consolidation.main()

    def test_a_stale_section_stops_the_step(self, tmp_path, monkeypatch, capsys):
        """Without the flag the command returns 1 and writes nothing."""
        assert self._run(tmp_path, monkeypatch, []) == 1
        assert not (tmp_path / "out" / "neural_cohort_table.csv").exists()
        assert "--allow-stale-sections" in capsys.readouterr().out

    def test_the_flag_excludes_the_stale_section_with_its_reason(self, tmp_path, monkeypatch):
        """With the flag the stale section is excluded as not run, reason ``stale_settings``."""
        assert self._run(tmp_path, monkeypatch, ["--allow-stale-sections"]) == 0
        assert self.artifact["vocal_gating"] == {"ran": False, "reason": "stale_settings"}
        table = pls.read_csv(tmp_path / "out" / "neural_cohort_table.csv")
        assert table["reason_gating"][0] == "stale_settings"
