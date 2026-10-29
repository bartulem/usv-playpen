"""
@author: bartulem
Unit tests for ``usv_playpen.neural_modeling.main_neural_cohort_dispatcher``.

Coverage: the cohort runner's bookkeeping -- recorded sessions behind block identifiers, the sections each
vocal step writes, which steps a resumed run still owes, whether a fold file is current, the manifest's
settings check, the input check before planning, the written manifest -- ``run_unit``'s orchestration
with the heavy stages stubbed (what is skipped when current, what is re-run, a session set that changed
since planning, an encoding failure that must not stop the vocal steps), and the monitoring: timestamped
log lines, the per-section result line, the per-unit status file and the cohort status summary.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from usv_playpen.modeling.modeling_metadata import compute_settings_sha256
from usv_playpen.neural_modeling import (
    main_neural_cohort_dispatcher as cohort_dispatcher,
)
from usv_playpen.neural_modeling.main_neural_cohort_dispatcher import (
    MANIFEST_NAME,
    check_inputs,
    cohort_status,
    fold_is_current,
    load_manifest,
    pending_vocal_steps,
    plan,
    recorded_sessions,
    run_unit,
    section_summary,
    step_sections,
    timestamped,
)
from usv_playpen.neural_modeling.main_neural_encoding_dispatcher import (
    load_settings,
    write_fold_artifact,
)
from usv_playpen.neural_modeling.main_neural_vocal_dispatcher import VOCAL_STEPS
from usv_playpen.neural_modeling.neural_artifacts import (
    read_unit_artifact,
    write_unit_section,
)

RECORD = {"unit_uid": "m1_20250101_imec0_cl0001_ch001_good", "unit_id": "imec0_cl0001_ch001_good",
          "mouse_id": "m1", "rec_date": 20250101, "brain_area": "PAG", "block_split": None,
          "courtship_sessions": ["20250101_100000", "20250101_110000"],
          "vocal_sessions": ["20250101_100000", "20250101_110000"],
          "recorded_sessions": ["20250101_100000", "20250101_110000"]}


def _all_sections(settings: dict) -> dict:
    """
    Description
    -----------
    A result file in which every vocal step's sections were written under ``settings``.

    Parameters
    ----------
    settings (dict)
        The settings the sections record.

    Returns
    -------
    artifact (dict)
        Sections and their per-section provenance.
    """

    sections = [section for step in VOCAL_STEPS for section in step_sections(step)]
    sha = compute_settings_sha256(settings)
    return {**{section: {} for section in sections},
            "section_provenance": {section: {"settings_sha256": sha} for section in sections}}


class TestBookkeeping:
    """Session sets, sections, pending steps, fold currency and the manifest."""

    def test_recorded_sessions_strip_block_identifiers(self):
        """A one-session unit's blocks map back to the one recording its fold processes must load."""
        record = {"courtship_sessions": ["20250101_100000#block0", "20250101_100000#block1"]}
        assert recorded_sessions(record) == ["20250101_100000"]

    def test_per_target_steps_write_one_section_per_target(self):
        """Identity and the nested decoding write both targets' sections; occurrence and gating write one."""
        assert step_sections("vocalization_identity") == ["vocalization_identity_manifold",
                                                          "vocalization_identity_multinomial"]
        assert step_sections("vocal_occurrence") == ["vocal_occurrence"]

    def test_nothing_is_pending_when_every_section_is_current(self):
        """A finished unit must not be recomputed."""
        settings = load_settings()
        assert pending_vocal_steps(_all_sections(settings), settings) == ()

    def test_a_stale_step_reruns_with_every_later_step(self):
        """Gating reads occurrence and identity, so recomputing identity must recompute what follows."""
        settings = load_settings()
        artifact = _all_sections(settings)
        artifact["section_provenance"]["vocalization_identity_multinomial"]["settings_sha256"] = "0" * 64
        assert pending_vocal_steps(artifact, settings) == VOCAL_STEPS[1:]
        assert pending_vocal_steps({}, settings) == VOCAL_STEPS

    def test_a_fold_is_current_only_under_its_own_settings(self, tmp_path):
        """A fold file written under other settings, or absent, must be re-run."""
        settings, other = load_settings(), load_settings()
        other["null"]["shuffle_seed"] = int(other["null"]["shuffle_seed"]) + 1
        assert not fold_is_current(str(tmp_path), RECORD, 0, settings)
        write_fold_artifact({"fold_index": 0, "no_model": True}, RECORD, 0, str(tmp_path), settings,
                            message_output=lambda *_a: None)
        assert fold_is_current(str(tmp_path), RECORD, 0, settings)
        assert not fold_is_current(str(tmp_path), RECORD, 0, other)

    def test_a_manifest_planned_under_other_settings_is_refused(self, tmp_path):
        """Which units and sessions the cohort holds depends on the settings, so a mismatch stops the run."""
        manifest = tmp_path / MANIFEST_NAME
        manifest.write_text(json.dumps({"settings_sha256": "0" * 64, "n_units": 0, "units": []}))
        with pytest.raises(ValueError, match="other settings"):
            load_manifest(str(manifest), load_settings())

    def test_missing_inputs_are_named(self, tmp_path):
        """Every absent input is reported before an array is launched."""
        settings = load_settings()
        settings["data_roots"]["catalog_path"] = str(tmp_path / "absent_catalog.csv")
        settings["data_roots"]["data_root"] = str(tmp_path)
        for target in ("manifold", "multinomial"):
            settings["nested_vocal_decoding"]["behaviour_control_model_path"][target] = str(tmp_path)
            settings["data"]["behaviour_selection_result_path"][target] = str(tmp_path)
        missing = check_inputs(settings)
        assert len(missing) == 1
        assert "unit catalog" in missing[0]

    def test_plan_writes_the_manifest_with_recorded_sessions(self, tmp_path, monkeypatch):
        """The manifest carries the settings hash, the count and each unit's recorded sessions."""
        settings = load_settings()
        block_record = {**RECORD, "unit_uid": "m1_20250101_imec0_cl0002_ch002_good",
                        "courtship_sessions": ["20250101_100000#block0", "20250101_100000#block1"],
                        "vocal_sessions": ["20250101_100000#block0", "20250101_100000#block1"],
                        "block_split": {"n_blocks": 2}}
        monkeypatch.setattr(cohort_dispatcher, "check_inputs", lambda _settings: [])
        monkeypatch.setattr(cohort_dispatcher, "build_cohort_manifest", lambda _settings: [dict(block_record)])
        monkeypatch.setattr(cohort_dispatcher, "summarize_cohort", lambda cohort: {"n_units": len(cohort)})
        path = plan(str(tmp_path), message_output=lambda *_a: None)
        manifest = json.loads(path.read_text())
        assert manifest["n_units"] == 1
        assert manifest["settings_sha256"] == compute_settings_sha256(settings)
        assert manifest["units"][0]["recorded_sessions"] == ["20250101_100000"]

    def test_plan_stops_when_an_input_is_missing(self, tmp_path, monkeypatch):
        """Nothing is planned when an input is absent."""
        monkeypatch.setattr(cohort_dispatcher, "check_inputs", lambda _settings: ["unit catalog: /x"])
        with pytest.raises(FileNotFoundError, match="unit catalog"):
            plan(str(tmp_path), message_output=lambda *_a: None)
        assert not (tmp_path / MANIFEST_NAME).exists()


class TestRunUnit:
    """``run_unit`` with the folds, the combination and the vocal steps stubbed."""

    @staticmethod
    def _stub(monkeypatch, calls: dict, fold_failures: list | None = None, rebuilt: dict | None = None) -> None:
        """
        Description
        -----------
        Replace the heavy stages with recorders; the combination writes a real section so resumption can
        be observed.

        Parameters
        ----------
        monkeypatch (pytest.MonkeyPatch)
            The fixture.
        calls (dict)
            Filled with what each stage was asked to do.
        fold_failures (list | None)
            Fold indices the fold stage reports as failed.
        rebuilt (dict | None)
            Overrides of the rebuilt unit record (e.g. a changed session set).

        Returns
        -------
        None
        """

        def build(_unit_id, _mouse_id, _rec_date, _sessions, _data_root, _settings):
            unit = {key: RECORD[key] for key in ("unit_uid", "unit_id", "mouse_id", "rec_date",
                                                 "courtship_sessions", "vocal_sessions")}
            return {**unit, **(rebuilt or {})}

        def run_folds(unit, folds, output_directory, _data_root, _settings_path, _parallel, _threads, _message_output):
            calls.setdefault("folds", []).append(list(folds))
            for fold_index in folds:
                if fold_index not in (fold_failures or []):
                    write_fold_artifact({"fold_index": fold_index}, unit, fold_index, output_directory,
                                        load_settings(), message_output=lambda *_a: None)
            return list(fold_failures or [])

        def combine(unit, settings, _data_root, output_directory, _message_output):
            calls.setdefault("combine", 0)
            calls["combine"] += 1
            write_unit_section(output_directory, unit, "kinematic_encoding", {"quiet_p": 0.5}, settings)

        def run_vocal(unit, settings, _data_root, output_directory, _n_workers, steps, _message_output):
            calls.setdefault("vocal", []).append(tuple(steps))
            for step in steps:
                for section in step_sections(step):
                    write_unit_section(output_directory, unit, section, {"p": 0.5}, settings)

        monkeypatch.setattr(cohort_dispatcher, "build_unit_record", build)
        monkeypatch.setattr(cohort_dispatcher, "run_folds", run_folds)
        monkeypatch.setattr(cohort_dispatcher, "combine", combine)
        monkeypatch.setattr(cohort_dispatcher, "run_vocal", run_vocal)

    def test_a_first_run_does_everything_and_a_second_run_nothing(self, tmp_path, monkeypatch):
        """Resumption: a finished unit re-run under the same settings recomputes nothing."""
        calls: dict = {}
        self._stub(monkeypatch, calls)
        settings = load_settings()
        assert run_unit(RECORD, settings, None, "/data", str(tmp_path), 4, 2, message_output=lambda *_a: None)
        assert calls == {"folds": [[0, 1]], "combine": 1, "vocal": [(step,) for step in VOCAL_STEPS]}
        calls.clear()
        assert run_unit(RECORD, settings, None, "/data", str(tmp_path), 4, 2, message_output=lambda *_a: None)
        assert calls == {"folds": [[]]}
        artifact = read_unit_artifact(str(tmp_path), RECORD["unit_uid"])
        assert artifact["identity"]["brain_area"] == "PAG"

    def test_a_session_set_changed_since_planning_is_refused(self, tmp_path, monkeypatch):
        """A unit is never analysed on other sessions than the cohort lists."""
        self._stub(monkeypatch, {}, rebuilt={"vocal_sessions": ["20250101_100000"]})
        with pytest.raises(ValueError, match="re-plan"):
            run_unit(RECORD, load_settings(), None, "/data", str(tmp_path), 4, 2, message_output=lambda *_a: None)

    def test_an_encoding_failure_does_not_stop_the_vocal_steps(self, tmp_path, monkeypatch):
        """The vocal steps do not depend on the encoding; the task still reports the failure."""
        calls: dict = {}
        self._stub(monkeypatch, calls, fold_failures=[1])
        assert not run_unit(RECORD, load_settings(), None, "/data", str(tmp_path), 4, 2, message_output=lambda *_a: None)
        assert "combine" not in calls
        assert calls["vocal"] == [(step,) for step in VOCAL_STEPS]
        assert np.load(tmp_path / f"{RECORD['unit_uid']}_fold0.npz", allow_pickle=True)["fold_index"] == 0

    def test_the_status_file_records_every_stage_and_the_outcome(self, tmp_path, monkeypatch):
        """The status file must name each stage's state and the task's outcome, for the status command."""
        self._stub(monkeypatch, {}, fold_failures=[1])
        run_unit(RECORD, load_settings(), None, "/data", str(tmp_path), 4, 2, {"index": 7},
                 message_output=lambda *_a: None)
        status = json.loads((tmp_path / "status" / f"{RECORD['unit_uid']}.json").read_text())
        assert status["state"] == "failed"
        assert status["index"] == 7
        assert status["stages"]["kinematic_encoding"]["state"] == "failed"
        assert all(status["stages"][step]["state"] == "done" for step in VOCAL_STEPS)

    def test_the_log_is_a_timestamped_timeline_with_a_summary(self, tmp_path, monkeypatch):
        """Every line carries a time, and the run closes with a per-stage summary."""
        self._stub(monkeypatch, {})
        lines: list = []
        run_unit(RECORD, load_settings(), None, "/data", str(tmp_path), 4, 2, message_output=lines.append)
        assert all(line.startswith("[20") for line in lines)
        assert any("SUMMARY" in line and "OK" in line for line in lines)
        assert any("STAGE vocal_gating DONE" in line for line in lines)


class TestMonitoring:
    """The pieces that make a running cohort readable."""

    def test_timestamped_prefixes_the_wall_clock_time(self):
        """A stamped line starts with the date and time."""
        lines: list = []
        timestamped(lines.append)("hello")
        assert lines[0].startswith("[20")
        assert lines[0].endswith("] hello")

    def test_section_summary_reports_p_effect_or_reason(self):
        """The one-line result names the p-value and effect, or why the section did not run."""
        artifact = {"vocal_occurrence": {"p": 0.001, "at_floor": True, "gain": 0.3},
                    "vocal_gating": {"ran": False, "reason": "no_vocal_tuning"}}
        assert section_summary(artifact, "vocal_occurrence") == "p 1.0e-03 | (floor) | gain +0.3000"
        assert section_summary(artifact, "vocal_gating") == "not run: no_vocal_tuning"
        assert section_summary(artifact, "nested_vocal_decoding_manifold") == "missing"

    def test_cohort_status_counts_every_state(self, tmp_path):
        """Done, failed, running and never-started units are counted, and failures name their stage."""
        units = [{"unit_uid": f"u{index}"} for index in range(4)]
        manifest = tmp_path / MANIFEST_NAME
        manifest.write_text(json.dumps({"n_units": 4, "units": units}))
        status_directory = tmp_path / "status"
        status_directory.mkdir()
        for uid, state, stages in (("u0", "done", {"vocal_occurrence": {"state": "done", "minutes": 2.0}}),
                                   ("u1", "failed", {"kinematic_encoding": {"state": "failed", "minutes": 1.0}}),
                                   ("u2", "running", {})):
            (status_directory / f"{uid}.json").write_text(json.dumps({"state": state, "stages": stages}))
        summary = cohort_status(str(manifest), str(tmp_path), message_output=lambda *_a: None)
        assert summary["counts"] == {"done": 1, "running": 1, "failed": 1, "not_started": 1}
        assert summary["failed"] == {"u1": ["kinematic_encoding"]}
        assert summary["median_minutes"] == {"vocal_occurrence": 2.0}
