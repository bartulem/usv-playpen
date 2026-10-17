"""
@author: bartulem
Unit tests for ``usv_playpen.neural_modeling.neural_artifacts``.

Coverage: the merged per-unit file. The properties that matter are the ones the design turns on --
one analysis's section survives another analysis's write, provenance travels with the results, and a
decision-shaped key is refused rather than stored.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from usv_playpen.neural_modeling.neural_artifacts import (
    build_provenance,
    read_unit_artifact,
    unit_artifact_path,
    write_unit_section,
)

SETTINGS_PATH = (pathlib.Path(__file__).resolve().parents[2] / "src" / "usv_playpen"
                 / "_parameter_settings" / "neural_modeling_settings.json")
UNIT = {"unit_uid": "m1_20250101_imec0_cl0001", "mouse_id": "m1", "unit_id": "imec0_cl0001",
        "brain_area": "PAG", "courtship_sessions": ["a", "b"], "vocal_sessions": ["a"],
        "emitter_usvs_per_session": {"a": 500, "b": 3}}


def _settings() -> dict:
    with SETTINGS_PATH.open() as handle:
        return json.load(handle)


class TestMergedFile:

    def test_a_later_section_does_not_clobber_an_earlier_one(self, tmp_path):
        """The whole point of one file per unit: analyses run at different times, on different cohorts,
        and each must be able to write without destroying the others."""
        settings = _settings()
        write_unit_section(str(tmp_path), UNIT, "kinematic_encoding", {"quiet_p": 1e-6}, settings)
        write_unit_section(str(tmp_path), UNIT, "vocalization_identity", {"gain": 0.057}, settings)
        write_unit_section(str(tmp_path), UNIT, "vocal_occurrence", {"nats": 0.14}, settings)

        artifact = read_unit_artifact(str(tmp_path), UNIT["unit_uid"])
        assert artifact["kinematic_encoding"] == {"quiet_p": 1e-6}
        assert artifact["vocalization_identity"] == {"gain": 0.057}
        assert artifact["vocal_occurrence"] == {"nats": 0.14}

    def test_rewriting_a_section_replaces_only_that_section(self, tmp_path):
        settings = _settings()
        write_unit_section(str(tmp_path), UNIT, "kinematic_encoding", {"quiet_p": 1e-6}, settings)
        write_unit_section(str(tmp_path), UNIT, "vocalization_identity", {"gain": 0.057}, settings)
        write_unit_section(str(tmp_path), UNIT, "vocalization_identity", {"gain": 0.061}, settings)

        artifact = read_unit_artifact(str(tmp_path), UNIT["unit_uid"])
        assert artifact["vocalization_identity"] == {"gain": 0.061}
        assert artifact["kinematic_encoding"] == {"quiet_p": 1e-6}

    def test_identity_and_provenance_travel_with_the_results(self, tmp_path):
        """A number without the configuration that produced it cannot be checked later."""
        write_unit_section(str(tmp_path), UNIT, "kinematic_encoding", {"quiet_p": 1e-6}, _settings())
        artifact = read_unit_artifact(str(tmp_path), UNIT["unit_uid"])

        assert artifact["identity"]["vocal_sessions"] == ["a"]
        assert artifact["identity"]["emitter_usvs_per_session"] == {"a": 500, "b": 3}
        assert set(artifact["provenance"]) == {"settings_sha256", "git_commit", "git_dirty",
                                               "package_version"}
        assert len(artifact["provenance"]["settings_sha256"]) == 64

    def test_the_settings_hash_moves_when_the_settings_do(self):
        settings = _settings()
        before = build_provenance(settings)["settings_sha256"]
        settings["significance"]["fdr_q"] = 0.05
        assert build_provenance(settings)["settings_sha256"] != before

    def test_a_missing_file_reads_as_empty_rather_than_raising(self, tmp_path):
        assert read_unit_artifact(str(tmp_path), "nobody") == {}

    def test_the_path_is_named_for_the_unit(self, tmp_path):
        assert unit_artifact_path(str(tmp_path), "u1").name == "u1.pkl"


class TestDecisionsAreRefused:
    """A verdict in a per-unit file is uncorrected by construction -- false-discovery control needs
    the cohort, and a per-unit file has never seen it. Storing one invites it being read as though it
    had been corrected."""

    @pytest.mark.parametrize("key", ["verdict", "passed", "significant", "fdr_flag", "rejected",
                                     "is_transformation", "label", "borderline", "survivors"])
    def test_decision_shaped_keys_are_refused(self, tmp_path, key):
        with pytest.raises(ValueError, match="not decisions"):
            write_unit_section(str(tmp_path), UNIT, "kinematic_encoding", {key: True, "quiet_p": 1e-6},
                               _settings())

    def test_the_refusal_leaves_the_existing_file_untouched(self, tmp_path):
        settings = _settings()
        write_unit_section(str(tmp_path), UNIT, "kinematic_encoding", {"quiet_p": 1e-6}, settings)
        with pytest.raises(ValueError, match="not decisions"):
            write_unit_section(str(tmp_path), UNIT, "vocalization_identity", {"verdict": "PASS"}, settings)
        artifact = read_unit_artifact(str(tmp_path), UNIT["unit_uid"])
        assert artifact["kinematic_encoding"] == {"quiet_p": 1e-6}
        assert "vocalization_identity" not in artifact

    def test_a_decision_nested_inside_a_result_is_refused_too(self, tmp_path):
        """Gating's label sits inside a per-feature entry, never at the top of the payload. A guard that
        only looked at the top would store all of them."""
        payload = {"per_feature": {"nose-nose": {"p_interaction": 5e-4, "label": "GATE"}}, "p_unit": 5e-4}
        with pytest.raises(ValueError, match=r"per_feature\.nose-nose\.label"):
            write_unit_section(str(tmp_path), UNIT, "vocal_gating", payload, _settings())

    def test_a_decision_inside_a_list_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="not decisions"):
            write_unit_section(str(tmp_path), UNIT, "vocal_gating",
                               {"per_feature": [{"feature": "nose-nose", "borderline": True}]}, _settings())

    def test_data_named_labels_is_not_a_decision(self, tmp_path):
        """The kinematic encoding stores every vocal frame's 0/1 spike labels as `labels`. Exact-key matching keeps real
        data from being refused as if it were a verdict."""
        payload = {"transfer_per_session": [{"labels": [0.0, 1.0, 0.0], "auroc": 0.6}]}
        write_unit_section(str(tmp_path), UNIT, "kinematic_encoding", payload, _settings())
        stored = read_unit_artifact(str(tmp_path), UNIT["unit_uid"])["kinematic_encoding"]
        assert stored["transfer_per_session"][0]["labels"] == [0.0, 1.0, 0.0]

    def test_p_values_and_metrics_are_of_course_fine(self, tmp_path):
        write_unit_section(str(tmp_path), UNIT, "kinematic_encoding",
                           {"quiet_p": 1e-6, "quiet_d2": 0.15, "transfer_slope": 0.44}, _settings())
        assert read_unit_artifact(str(tmp_path), UNIT["unit_uid"])["kinematic_encoding"]["quiet_d2"] == 0.15
