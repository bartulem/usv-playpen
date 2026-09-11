"""
@author: bartulem
Unit tests for the call-grain data layer in ``usv_playpen.neural_modeling.neural_design_assembly``.

Coverage: the clean-prevocal filter's exact overlap rule, the header-only tracking read, and the
end-to-end event assembler run against a synthetic session tree written under ``tmp_path``. The
properties asserted are the ones both claim-2 axes depend on -- a call never excludes itself, a partner's
call excludes just as a focal one does, calls with no torus position are dropped and counted rather than
carried as NaN, and the WHEN negatives are the same quiet definition the encoding claim uses, tiled at
the analysis window's own width so the two counts are comparable.

Also guards the settings invariants the assembler reads: the analysis windows must agree with the
baseline width, and the neural copy of ``kinematic_features`` must still equal the behavioural one, since
claim 3's "beyond behaviour" means "beyond exactly what P1 fits".
"""

from __future__ import annotations

import inspect
import json
import pathlib
from typing import ClassVar

import h5py
import numpy as np
import polars as pl
import pytest

from usv_playpen.neural_modeling.main_neural_encoding_dispatcher import (
    focal_vocal_frames,
)
from usv_playpen.neural_modeling.neural_design_assembly import (
    assemble_unit_sessions,
    assemble_unit_vocal_events,
    build_zscored_feature_frames,
    clean_window_mask,
    emitter_names,
    session_timebase,
)
from usv_playpen.neural_modeling.neural_when import counts_in_windows

SETTINGS_DIR = pathlib.Path(__file__).resolve().parents[2] / "src" / "usv_playpen" / "_parameter_settings"


def _load_settings() -> dict:
    """Read the shipped neural-modeling settings."""
    with (SETTINGS_DIR / "neural_modeling_settings.json").open() as handle:
        return json.load(handle)


class TestCleanWindowMask:

    def test_a_call_never_excludes_itself(self):
        """A call's own span starts exactly at its onset, so it can never sit inside its own window."""
        starts = np.array([1.0, 5.0])
        stops = np.array([1.2, 5.3])
        keep = clean_window_mask(starts, starts, stops, 0.05)
        assert keep.tolist() == [True, True]

    def test_a_preceding_call_that_reaches_into_the_window_excludes(self):
        starts = np.array([1.0, 1.22])
        stops = np.array([1.2, 1.4])
        # the second call's window is (1.17, 1.22); the first call runs to 1.2 and so overlaps it
        keep = clean_window_mask(starts, starts, stops, 0.05)
        assert keep.tolist() == [True, False]

    def test_a_preceding_call_that_stops_before_the_window_does_not(self):
        starts = np.array([1.0, 1.30])
        stops = np.array([1.2, 1.5])
        # the window is (1.25, 1.30) and the previous call ended at 1.2
        keep = clean_window_mask(starts, starts, stops, 0.05)
        assert keep.tolist() == [True, True]

    def test_a_partner_call_excludes_exactly_as_a_focal_one_does(self):
        focal = np.array([2.0])
        every_start = np.array([1.98, 2.0])
        every_stop = np.array([1.99, 2.1])
        assert clean_window_mask(focal, every_start, every_stop, 0.05).tolist() == [False]
        assert clean_window_mask(focal, every_start[1:], every_stop[1:], 0.05).tolist() == [True]

    def test_empty_inputs_return_the_right_shapes(self):
        assert clean_window_mask(np.array([]), np.array([1.0]), np.array([2.0]), 0.05).size == 0
        assert clean_window_mask(np.array([1.0]), np.array([]), np.array([]), 0.05).tolist() == [True]


def _write_session(root: pathlib.Path, session_id: str, unit_id: str, *, n_frames: int,
                   frame_rate: float, calls: list[tuple[float, float, str, float, float]],
                   spike_seconds: np.ndarray) -> None:
    """Write one synthetic session tree: tracking H5, USV summary CSV and a cluster_data spike file."""
    video_dir = root / session_id / "video" / f"{session_id}_cam"
    video_dir.mkdir(parents=True)
    with h5py.File(video_dir / f"{session_id}_points3d_translated_rotated_metric.h5", "w") as handle:
        handle["tracks"] = np.zeros((n_frames, 2, 15, 3), dtype=np.float64)
        handle["recording_frame_rate"] = frame_rate
        handle.create_dataset("track_names", data=np.array([b"male", b"female"], dtype="S6"))

    audio_dir = root / session_id / "audio"
    audio_dir.mkdir(parents=True)
    pl.DataFrame({
        "usv_id": list(range(len(calls))),
        "start": [c[0] for c in calls],
        "stop": [c[1] for c in calls],
        "emitter": [c[2] for c in calls],
        "qlvm1": [c[3] for c in calls],
        "qlvm2": [c[4] for c in calls],
    }).write_csv(audio_dir / f"{session_id}_usv_summary.csv")

    ephys_dir = root / session_id / "ephys" / "imec0" / "cluster_data"
    ephys_dir.mkdir(parents=True)
    np.save(ephys_dir / f"{unit_id}.npy",
            np.vstack([spike_seconds, spike_seconds * frame_rate]))


class TestSessionTimebase:

    def test_the_header_read_returns_names_rate_and_frame_count(self, tmp_path):
        _write_session(tmp_path, "20250101_000000", "imec0_cl0001_ch001_good", n_frames=1500,
                       frame_rate=150.0, calls=[(1.0, 1.1, "male", 0.1, 0.2)],
                       spike_seconds=np.array([0.5]))
        names, rate, n_frames = session_timebase(str(tmp_path), "20250101_000000")
        assert names == ["male", "female"]
        assert rate == pytest.approx(150.0)
        assert n_frames == 1500

    def test_a_missing_tracking_file_fails_loudly(self, tmp_path):
        (tmp_path / "20250101_000000" / "video").mkdir(parents=True)
        with pytest.raises(FileNotFoundError):
            session_timebase(str(tmp_path), "20250101_000000")


class TestAssembleUnitVocalEvents:

    UNIT = "imec0_cl0001_ch001_good"

    def _settings(self) -> dict:
        settings = _load_settings()
        # a tiny cap keeps the synthetic sessions' tile subsampling exercised without a huge tile set
        settings["vocal_decoding"]["max_quiet_tiles_per_session"] = 20
        return settings

    def _build(self, tmp_path, *, calls, spike_seconds, n_frames=150000, frame_rate=150.0):
        _write_session(tmp_path, "20250101_000000", self.UNIT, n_frames=n_frames,
                       frame_rate=frame_rate, calls=calls, spike_seconds=spike_seconds)
        return {"unit_id": self.UNIT, "mouse_id": "male",
                "courtship_sessions": ["20250101_000000"],
                "vocal_sessions": ["20250101_000000"]}

    def test_only_focal_calls_become_events(self, tmp_path):
        calls = [(100.0, 100.1, "male", 0.10, 0.20),
                 (200.0, 200.1, "female", 0.30, 0.40),
                 (300.0, 300.1, "male", 0.50, 0.60)]
        unit = self._build(tmp_path, calls=calls, spike_seconds=np.array([99.98, 299.97]))
        events = assemble_unit_vocal_events(unit, str(tmp_path), self._settings(), 0.05, 0.05)

        assert events["positions"].shape == (2, 2)
        assert events["positions"][:, 0].tolist() == [0.10, 0.50]
        assert events["session_index"].tolist() == [0, 0]
        assert events["per_session"]["20250101_000000"]["n_focal_calls"] == 2

    def test_counts_come_from_the_window_before_onset(self, tmp_path):
        calls = [(100.0, 100.1, "male", 0.10, 0.20), (300.0, 300.1, "male", 0.50, 0.60)]
        # two spikes inside the first call's [99.95, 100.0) window, none inside the second's
        spikes = np.array([99.96, 99.99, 300.05])
        unit = self._build(tmp_path, calls=calls, spike_seconds=spikes)
        events = assemble_unit_vocal_events(unit, str(tmp_path), self._settings(), 0.05, 0.05)
        assert events["counts"].tolist() == [2.0, 0.0]

    def test_a_missing_torus_position_is_dropped_and_counted(self, tmp_path):
        calls = [(100.0, 100.1, "male", 0.10, 0.20), (300.0, 300.1, "male", float("nan"), 0.60)]
        unit = self._build(tmp_path, calls=calls, spike_seconds=np.array([99.98]))
        events = assemble_unit_vocal_events(unit, str(tmp_path), self._settings(), 0.05, 0.05)

        assert events["positions"].shape == (1, 2)
        assert np.isfinite(events["positions"]).all()
        assert events["per_session"]["20250101_000000"]["n_dropped_missing_position"] == 1

    def test_the_clean_filter_is_recorded_and_can_be_switched_off(self, tmp_path):
        # the second call opens 20 ms after the first closes, so its 50 ms window is contaminated
        calls = [(100.0, 100.10, "male", 0.10, 0.20), (100.12, 100.22, "male", 0.50, 0.60)]
        unit = self._build(tmp_path, calls=calls, spike_seconds=np.array([99.98]))

        settings = self._settings()
        events = assemble_unit_vocal_events(unit, str(tmp_path), settings, 0.05, 0.05)
        assert events["positions"].shape[0] == 1
        assert events["per_session"]["20250101_000000"]["n_dropped_unclean"] == 1

        settings["vocal_decoding"]["require_clean_prevocal_bool"] = False
        loose = assemble_unit_vocal_events(unit, str(tmp_path), settings, 0.05, 0.05)
        assert loose["positions"].shape[0] == 2

    def test_the_when_arrays_stack_positives_over_quiet_tiles(self, tmp_path):
        calls = [(100.0, 100.1, "male", 0.10, 0.20), (300.0, 300.1, "male", 0.50, 0.60)]
        unit = self._build(tmp_path, calls=calls, spike_seconds=np.array([99.98, 500.0]))
        events = assemble_unit_vocal_events(unit, str(tmp_path), self._settings(), 0.05, 0.05)

        when = events["when"]
        n_events = events["counts"].size
        n_tiles = events["per_session"]["20250101_000000"]["n_baseline_tiles"]
        assert when["counts"].size == n_events + n_tiles
        assert when["labels"][:n_events].tolist() == [1.0] * n_events
        assert not when["labels"][n_events:].any()
        assert when["counts"][:n_events].tolist() == events["counts"].tolist()

    def test_the_when_edges_reconstruct_the_when_counts(self, tmp_path):
        """The WHEN null is a circular shift that recounts BOTH classes from the shifted train. With
        counts alone that is impossible -- a caller can only recount the positives and has to leave
        the tiles frozen, which is a different null from the specified one. The edges make it a
        per-session recount, so they have to line up with the counts exactly."""
        calls = [(100.0, 100.1, "male", 0.10, 0.20), (300.0, 300.1, "male", 0.50, 0.60)]
        spikes = np.sort(np.random.default_rng(0).uniform(0, 1000, 4000))
        unit = self._build(tmp_path, calls=calls, spike_seconds=spikes)
        events = assemble_unit_vocal_events(unit, str(tmp_path), self._settings(), 0.05, 0.05)

        when = events["when"]
        assert when["edges"].size == when["counts"].size
        rebuilt = counts_in_windows(np.sort(spikes), when["edges"], when["width"])
        assert np.array_equal(rebuilt, when["counts"])

    def test_tiles_are_disjoint_at_the_analysis_width(self, tmp_path):
        """Tiles are laid at the analysed window's OWN width, so consecutive negatives never share a
        frame. The earlier design laid them at a fixed baseline width and re-centred wider windows on
        the same lattice, which made adjacent tiles overlap by half and their counts correlated. With
        one window per run there is no second width, and the negatives are a genuine sample."""
        calls = [(100.0, 100.1, "male", 0.10, 0.20)]
        unit = self._build(tmp_path, calls=calls, spike_seconds=np.array([99.98]))
        for width in (0.05, 0.10):
            events = assemble_unit_vocal_events(unit, str(tmp_path), self._settings(), 0.05, width)
            n_events = int(events["counts"].size)
            tiles = np.sort(events["when"]["edges"][n_events:])
            gaps = np.diff(tiles)
            assert np.all(gaps >= width - 1e-9)

    def test_tiles_are_capped_per_session(self, tmp_path):
        calls = [(100.0, 100.1, "male", 0.10, 0.20)]
        unit = self._build(tmp_path, calls=calls, spike_seconds=np.array([99.98]))
        events = assemble_unit_vocal_events(unit, str(tmp_path), self._settings(), 0.05, 0.05)
        assert events["per_session"]["20250101_000000"]["n_baseline_tiles"] == 20

    def test_a_legacy_position_column_name_fails_loudly(self, tmp_path):
        """The columns were renamed repo-wide; a loader that silently harmonised them would hide
        which embedding a result was actually computed on."""
        calls = [(100.0, 100.1, "male", 0.10, 0.20)]
        unit = self._build(tmp_path, calls=calls, spike_seconds=np.array([99.98]))
        settings = self._settings()
        settings["vocal_decoding"]["usv_manifold_column_names"] = ["qlvm_dim1", "qlvm_dim2"]
        with pytest.raises(KeyError, match="qlvm_dim1"):
            assemble_unit_vocal_events(unit, str(tmp_path), settings, 0.05, 0.05)

    def test_a_unit_whose_mouse_is_absent_fails_loudly(self, tmp_path):
        """The focal animal is resolved by LOOKUP from the unit's own mouse_id, never by track slot --
        a slot says "whatever is in position 0", which is a different claim. If the unit's mouse is
        not in this session's track_names, that is unrecoverable and must not be guessed at."""
        calls = [(100.0, 100.1, "male", 0.10, 0.20)]
        unit = self._build(tmp_path, calls=calls, spike_seconds=np.array([99.98]))
        unit["mouse_id"] = "some_other_mouse"
        with pytest.raises(ValueError, match="not in track_names"):
            assemble_unit_vocal_events(unit, str(tmp_path), self._settings(), 0.05, 0.05)

    def test_unsorted_spike_files_still_count_correctly(self, tmp_path):
        """Window counts are read by bisection, so an out-of-order train would return wrong counts
        with no error at all. The order on disk is the writer's habit, not a guarantee here."""
        calls = [(100.0, 100.1, "male", 0.10, 0.20)]
        shuffled = np.array([99.99, 99.96, 300.0, 99.97])
        unit = self._build(tmp_path, calls=calls, spike_seconds=shuffled)
        events = assemble_unit_vocal_events(unit, str(tmp_path), self._settings(), 0.05, 0.05)
        assert events["counts"].tolist() == [3.0]

    def test_out_of_order_calls_fail_loudly(self, tmp_path):
        """The gap covariate is start[i] - stop[i-1], which is silently meaningless -- negative, even
        -- if the rows are not in temporal order."""
        calls = [(300.0, 300.1, "male", 0.50, 0.60), (100.0, 100.1, "male", 0.10, 0.20)]
        unit = self._build(tmp_path, calls=calls, spike_seconds=np.array([99.98]))
        with pytest.raises(ValueError, match="ordered by start"):
            assemble_unit_vocal_events(unit, str(tmp_path), self._settings(), 0.05, 0.05)

    def test_the_recorded_gap_is_to_the_previous_focal_call(self, tmp_path):
        calls = [(100.0, 100.10, "male", 0.10, 0.20),
                 (100.50, 100.60, "male", 0.30, 0.40),
                 (102.00, 102.10, "male", 0.50, 0.60)]
        unit = self._build(tmp_path, calls=calls, spike_seconds=np.array([99.98]))
        events = assemble_unit_vocal_events(unit, str(tmp_path), self._settings(), 0.05, 0.05)
        assert np.isinf(events["silent_gap"][0])          # nothing precedes the first call
        assert events["silent_gap"][1] == pytest.approx(0.40)
        assert events["silent_gap"][2] == pytest.approx(1.40)

    def test_clean_against_selects_whose_calls_contaminate(self, tmp_path):
        """`vocalization_settings.clean_against` was read nowhere: every window was judged against every emitter, so
        setting it to a mouse index changed nothing at all."""
        calls = [(100.00, 100.10, "female", 0.90, 0.90),   # partner call, 20 ms before the focal one
                 (100.12, 100.22, "male", 0.50, 0.60)]
        unit = self._build(tmp_path, calls=calls, spike_seconds=np.array([99.98]))

        against_all = self._settings()
        assert against_all["vocalization_settings"]["clean_against"] == "all"
        assert against_all["vocalization_settings"]["vocal_emitter"] == "self"
        strict = assemble_unit_vocal_events(unit, str(tmp_path), against_all, 0.05, 0.05)
        assert strict["positions"].shape[0] == 0          # the partner's call contaminates it

        against_self = self._settings()
        against_self["vocalization_settings"]["clean_against"] = "self"    # only the recorded animal's own calls
        loose = assemble_unit_vocal_events(unit, str(tmp_path), against_self, 0.05, 0.05)
        assert loose["positions"].shape[0] == 1

    def test_sessions_are_indexed_by_slot(self, tmp_path):
        for slot, session_id in enumerate(["20250101_000000", "20250101_010000"]):
            _write_session(tmp_path, session_id, self.UNIT, n_frames=150000, frame_rate=150.0,
                           calls=[(100.0 + slot, 100.1 + slot, "male", 0.1, 0.2)],
                           spike_seconds=np.array([99.98 + slot]))
        unit = {"unit_id": self.UNIT, "mouse_id": "male",
                "courtship_sessions": ["20250101_000000", "20250101_010000"],
                "vocal_sessions": ["20250101_000000", "20250101_010000"]}
        events = assemble_unit_vocal_events(unit, str(tmp_path), self._settings(), 0.05, 0.05)
        assert events["session_index"].tolist() == [0, 1]
        assert events["session_ids"] == ["20250101_000000", "20250101_010000"]


class TestSettingsInvariants:

    def test_one_window_per_run_lives_in_the_shared_anchors_block(self):
        """ONE window is analysed per run, not a pair looped over. It sits in `vocalization_settings` because it is
        a cross-claim event definition -- claim 2 and claim 3 must see the same window, and claim 3
        has its own block, so a window inside `vocal_decoding` could not serve it."""
        settings = _load_settings()
        window = settings["vocalization_settings"]["spike_window"]
        assert set(window) == {"pre_offset_seconds", "width_seconds"}
        assert window["pre_offset_seconds"] == 0.05
        assert window["width_seconds"] == 0.10
        assert "analysis_windows" not in settings["vocal_decoding"]

    def test_the_shipped_window_straddles_onset_deliberately(self):
        """RULED 2026-09-11 (user): width 0.10 against a pre-offset of 0.05, so the window runs
        [start - 50 ms, start + 50 ms) and roughly 40-55% of a typical call (median duration 87-127
        ms) falls INSIDE it.

        This is a deliberate trade and it reverses the earlier prediction-clean default. It buys
        POWER -- banked gains rise from +0.05726 to +0.07199 on cl0401 and from +0.0031 to +0.0114 on
        cl0499, a 3.7x lift on the weak unit, and passes are decided at the faint margin. It costs the
        claim that the gain is purely PREDICTIVE: part of it is response to the call in progress, and
        the design cannot separate the two. The user ruled that showing the activity is purely
        premotor is not required.

        The pre-window is unchanged at 50 ms, so contamination from the PREVIOUS call is unaffected
        (3.7% of USVs cohort-wide have a gap below 50 ms); the post-onset half is inside the call by
        construction and needs no cleanliness guard.

        The two numbers stay independent rather than collapsing into one duration precisely so this
        choice has to be made explicitly."""
        window = _load_settings()["vocalization_settings"]["spike_window"]
        assert window["width_seconds"] > window["pre_offset_seconds"]
        inside = window["width_seconds"] - window["pre_offset_seconds"]
        assert inside == pytest.approx(0.05)

    def test_vocal_decoding_holds_claim_two_only(self):
        """Claim 3 is a separate claim with its own model, null and statistic; its configuration
        living inside claim 2's block was the file's worst piece of mis-filing."""
        settings = _load_settings()
        assert set(settings["vocal_decoding"]) == {
            "require_clean_prevocal_bool", "usv_manifold_column_names",
            "max_quiet_tiles_per_session", "when_axis", "geodesic_metrics", "tuning_surface",
            "discrimination_null", "record_overdispersion_index"}
        assert set(settings["nested_position_decoding"]) == {
            "behaviour_control", "reduced_model_features", "compute_matched_window_control",
            "vm_score_mode", "region_label_column", "min_region_events", "lambda_smooth", "l2_reg",
            "smoothness_derivative_order", "sigma_floor", "prevocal_window_n_bins", "rate_transform",
            "rate_basis"}

    def test_claim_three_matches_p1_where_it_must(self):
        """Claim 3 is "P1 plus the neuron", so the control has to be fitted and SCORED the way P1
        fitted and scored it. A silent divergence here would make the reduced model suboptimal for
        the test, which biases claim 3 toward passing -- the wrong direction for a positive claim."""
        nested = _load_settings()["nested_position_decoding"]
        modeling = (pathlib.Path(__file__).resolve().parents[2] / "src" / "usv_playpen"
                    / "_parameter_settings" / "modeling_settings.json")
        with modeling.open() as handle:
            p1 = json.load(handle)
        manifold = p1["hyperparameters"]["linear_models"]["manifold_regression"]
        vocal = p1["vocal_features"]
        assert nested["lambda_smooth"] == manifold["lambda_smooth_fixed"]
        assert nested["l2_reg"] == manifold["l2_reg_fixed"]
        assert nested["smoothness_derivative_order"] == manifold["smoothness_derivative_order"]
        assert nested["vm_score_mode"] == vocal["usv_manifold_selection_score"]
        assert nested["min_region_events"] == vocal["usv_manifold_min_region_events"]

    def test_the_dead_min_gates_are_gone(self):
        """`data_sufficiency` once held six `min_*` keys that nothing read, every value null, whose
        promised behaviour (NaN plus a report line for a unit failing an enabled gate) was never
        implemented. Four were deleted and stay deleted.

        TWO WERE RESTORED deliberately (2026-09-11) into `vocal_gating`, where they ARE read: gating
        needs them to distinguish `not_testable` -- a feature with no spread during vocal frames
        leaves delta unidentifiable -- from `ns`, which says the test ran and found nothing.
        Collapsing those two would convert missing power into evidence of absence."""
        settings = _load_settings()
        leaves = {key for block in settings.values() if isinstance(block, dict) for key in block}
        assert not leaves & {"min_quiet_anchors", "min_quiet_spikes", "min_focal_usvs_total",
                             "min_baseline_tiles"}
        assert set(settings["vocal_gating"]) >= {"min_vocal_spikes", "min_feature_iqr_vocal"}

    def test_the_sufficiency_block_holds_only_live_keys(self):
        """Every key here is read by something -- unlike the block that previously carried the name."""
        assert set(_load_settings()["data_sufficiency"]) == {
            "min_courtship_sessions", "min_emitter_usvs_per_session", "min_vocal_sessions",
            "single_session_inner_split_blocks"}

    def test_the_session_gate_is_named_for_the_emitter_it_follows(self):
        """The gate counts calls from whoever `vocalization_settings.vocal_emitter` names, so a name
        saying `focal` would misdescribe it for two of that knob's three values -- reading as correct
        while gating on the partner."""
        settings = _load_settings()
        assert settings["data_sufficiency"]["min_emitter_usvs_per_session"] == 20
        assert not any("focal" in key for block in settings.values()
                       if isinstance(block, dict) for key in block)

    def test_the_name_collisions_are_gone(self):
        """Each of these cost a false positive in an audit: two keys spelling the same words in a
        different order, two spelling the same concept differently, and two `min_focal_usvs` meaning
        different things."""
        settings = _load_settings()
        assert "n_shuffles_screen" not in settings["vocal_gating"]
        assert settings["vocal_gating"]["gating_screen_n_shuffles"] == 2000
        assert "calib_n_steps" not in settings["vocal_gating"]["solver"]
        assert "calib_n_steps" not in settings["vocal_decoding"]["when_axis"]["solver"]
        assert "min_focal_usvs" not in settings["vocal_decoding"]
        assert "decorrelation_lag_seconds" not in settings["null"]

    def test_the_neural_kinematic_features_still_equal_the_behavioural_ones(self):
        """Claim 3 asks whether a neuron adds beyond behaviour, where "behaviour" is the feature set P1
        fits. The neural settings own their copy so a neural run can diverge deliberately, but the
        shipped default has to match or the comparison silently stops meaning what it says."""
        with (SETTINGS_DIR / "modeling_settings.json").open() as handle:
            modeling = json.load(handle)
        assert _load_settings()["kinematic_features"] == modeling["kinematic_features"]


class TestFocalVocalFrames:
    """The claim-1 transfer resolves the focal animal separately from the assembler, and did it by
    substring match with a bare [0] index."""

    UNIT = "imec0_cl0001_ch001_good"

    def test_a_session_where_the_focal_animal_never_called_yields_no_frames(self, tmp_path):
        """Ordinary, not exceptional: it happens in 8 of 77 courtship sessions. It used to raise
        IndexError and kill the whole fold."""
        _write_session(tmp_path, "20250101_000000", self.UNIT, n_frames=150000, frame_rate=150.0,
                       calls=[(100.0, 100.1, "female", 0.1, 0.2)],   # partner only
                       spike_seconds=np.array([99.98]))
        session = {"n_frames": 150000}
        frames, pointer, gaps = focal_vocal_frames(session, "20250101_000000", str(tmp_path),
                                                   "male", 150.0, 600)
        assert frames.size == 0
        assert gaps.size == 0
        assert pointer.tolist() == [0]

    def test_the_emitter_is_matched_exactly_not_by_substring(self, tmp_path):
        """A bare mouse id is a prefix of the same animal's suffixed form, so a substring match can
        silently pick up another animal's calls."""
        _write_session(tmp_path, "20250101_000000", self.UNIT, n_frames=150000, frame_rate=150.0,
                       calls=[(100.0, 100.1, "147366_1", 0.1, 0.2),
                              (200.0, 200.1, "147366", 0.3, 0.4)],
                       spike_seconds=np.array([99.98]))
        session = {"n_frames": 150000}
        frames, _pointer, gaps = focal_vocal_frames(session, "20250101_000000", str(tmp_path),
                                                    "147366", 150.0, 600)
        assert gaps.size == 1                     # only the exact-match call, not the "147366_1" one
        assert frames.min() >= int(200.0 * 150.0)


class TestQuietSideKeepsEverySession:
    """The load-bearing half of the two-session-set design, and the easier one to break by accident.

    A session in which the recorded animal barely called is EXCLUDED from everything scored on
    vocalizations and must stay in everything fitted on silence -- it has more quiet, not less. If a
    future change points the kinematic assembler at `vocal_sessions`, claim 1 would quietly lose
    training data for the exact sessions that offer the most of it."""

    UNIT = "imec0_cl0001_ch001_good"

    def test_the_kinematic_assembler_reads_courtship_sessions_not_vocal_sessions(self, monkeypatch):
        seen = {}

        def _spy(session_dirs, **kwargs):
            seen["dirs"] = list(session_dirs)
            seen["mouse"] = kwargs["recorded_mouse_id"]
            return {}, {}, {}, {}

        monkeypatch.setattr(
            "usv_playpen.neural_modeling.neural_design_assembly.build_zscored_feature_frames", _spy)
        unit = {"unit_id": self.UNIT, "mouse_id": "male",
                "courtship_sessions": ["s_rich", "s_thin"],
                "vocal_sessions": ["s_rich"]}
        assemble_unit_sessions(unit, "/data", {}, 4.0, 2.0, "all", "self")

        assert seen["dirs"] == ["/data/s_rich", "/data/s_thin"]
        assert "/data/s_thin" in seen["dirs"]        # the thin session is FITTED on, not dropped
        # the `self.` role comes from the unit's own mouse, not a configured slot index
        assert seen["mouse"] == "male"


class TestSelfAndPartnerAreResolvedByName:
    """Who is `self` and who is `partner` must never come from a track SLOT. A slot says "whatever
    sits in position 0", which is a different claim from "this animal" and needs its own assertion to
    be safe. Both halves of the pipeline -- the emitter filter on the vocal side and the
    `self.`/`other.` feature split on the kinematic side -- now answer it the same way: `self` by NAME
    lookup of the unit's own `mouse_id`, `partner` by EXCLUSION."""

    NAMES: ClassVar[list[str]] = ["male", "female"]

    def test_self_is_the_units_own_mouse_not_a_slot(self):
        assert emitter_names(self.NAMES, "female", "self") == ["female"]
        assert emitter_names(self.NAMES, "male", "self") == ["male"]

    def test_partner_is_everyone_else_and_does_not_assume_two_tracks(self):
        """Exclusion is length-agnostic; a two-slot flip (`abs(idx - 1)`) is not, and would pick an
        arbitrary animal for a third track with no error."""
        assert emitter_names(self.NAMES, "male", "partner") == ["female"]
        trio = ["male", "female", "pup"]
        assert emitter_names(trio, "female", "partner") == ["male", "pup"]

    def test_all_is_no_filter_rather_than_the_two_track_names(self):
        """A session's USV table can carry rows whose emitter was never assigned, and a call nobody was
        credited with is still a call the animal heard."""
        assert emitter_names(self.NAMES, "male", "all") == []

    def test_an_absent_mouse_raises_rather_than_being_guessed_at(self):
        with pytest.raises(ValueError, match="not in track_names"):
            emitter_names(self.NAMES, "some_other_mouse", "self")

    def test_an_unknown_spec_raises(self):
        with pytest.raises(ValueError, match="emitter spec must be"):
            emitter_names(self.NAMES, "male", "focal")

    def test_the_feature_split_asserts_a_dyad(self, monkeypatch):
        """The P1 utility downstream derives the other animal as `abs(idx - 1)`, which is the partner
        only in a dyad. Measured 2 track names in 73 of 73 cohort sessions -- so a third one would go
        unnoticed, which is what the assertion is for."""
        def _three_tracks(**_kwargs):
            return ({"s1": pl.DataFrame({"male.speed": [0.0]})}, {"s1": 150.0},
                    {"s1": ["male", "female", "pup"]})

        monkeypatch.setattr(
            "usv_playpen.neural_modeling.neural_design_assembly.load_behavioral_feature_data",
            _three_tracks)
        with pytest.raises(ValueError, match="assumes a dyad"):
            build_zscored_feature_frames(["/data/s1"], {}, "male")


class TestOneQuietDefinition:
    """The plan requires ONE quiet definition across claim 1, the WHEN negatives and the claim-2
    baselines. Two code paths that can disagree is how that breaks -- and it did: `clean_against` was
    read by the vocal-side assembler and ignored by the kinematic one, so setting it to a mouse index
    would have moved claim 2's tiles while leaving claim 1's anchors all-emitter."""

    def test_the_kinematic_assembler_takes_clean_against(self):
        assert "clean_against" in inspect.signature(assemble_unit_sessions).parameters

    def test_the_kinematic_assembler_takes_vocal_emitter(self):
        """Same failure mode one key over: the kinematic side resolved the onset emitter from the
        unit's mouse_id directly, so pointing `vocalization_settings.vocal_emitter` at the partner would have moved
        claim 2's events while leaving claim 1's onsets on the recorded animal."""
        assert "vocal_emitter" in inspect.signature(assemble_unit_sessions).parameters

    def test_the_dispatcher_passes_the_same_setting_to_both_sides(self):
        source = (pathlib.Path(__file__).resolve().parents[2] / "src" / "usv_playpen"
                  / "neural_modeling" / "main_neural_encoding_dispatcher.py").read_text()
        # every assemble_unit_sessions call site must hand it the shared anchors values
        assert source.count("assemble_unit_sessions(") == source.count(
            'settings["vocalization_settings"]["clean_against"]')
        assert source.count('settings["vocalization_settings"]["vocal_emitter"]') >= source.count(
            "assemble_unit_sessions(")


class TestVocalSessionScope:
    """The cohort builder hands out two session sets, and the vocal-side assembler must read the
    vocal one. A session the male barely called in stays available to the encoding claim, which
    needs silence, and is kept away from everything scored on the calls themselves."""

    UNIT = "imec0_cl0001_ch001_good"

    def test_only_vocal_sessions_contribute_events(self, tmp_path):
        for slot, session_id in enumerate(["20250101_000000", "20250101_010000"]):
            _write_session(tmp_path, session_id, self.UNIT, n_frames=150000, frame_rate=150.0,
                           calls=[(100.0 + slot, 100.1 + slot, "male", 0.1, 0.2)],
                           spike_seconds=np.array([99.98 + slot]))
        settings = _load_settings()
        settings["vocal_decoding"]["max_quiet_tiles_per_session"] = 20

        both = {"unit_id": self.UNIT, "mouse_id": "male",
                "courtship_sessions": ["20250101_000000", "20250101_010000"],
                "vocal_sessions": ["20250101_000000", "20250101_010000"]}
        one = {**both, "vocal_sessions": ["20250101_000000"]}

        assert assemble_unit_vocal_events(both, str(tmp_path), settings, 0.05, 0.05
                                          )["session_ids"] == both["vocal_sessions"]
        gated = assemble_unit_vocal_events(one, str(tmp_path), settings, 0.05, 0.05)
        assert gated["session_ids"] == ["20250101_000000"]
        assert set(gated["session_index"].tolist()) == {0}
