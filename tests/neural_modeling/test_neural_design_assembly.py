"""
@author: bartulem
Unit tests for the USV-grain data layer in ``usv_playpen.neural_modeling.neural_design_assembly``.

Coverage: the clean-prevocal filter's exact overlap rule, the header-only tracking read, and the
end-to-end event assembler run against a synthetic session tree written under ``tmp_path``. The
properties asserted are the ones both vocal-decoding axes depend on -- a USV never excludes itself, a partner's
USV excludes just as a focal one does, USVs with no torus position are dropped and counted rather than
carried as NaN, and the vocal-occurrence negatives are the same quiet definition the kinematic encoding uses, tiled at
the analysis window's own width so the two counts are comparable.

Also guards the settings invariants the assembler reads: the analysis windows must agree with the
baseline width, and the neural copy of ``kinematic_features`` must still equal the behavioural one, since
the nested decoding's "beyond behaviour" means "beyond exactly what P1 fits".

The kinematic half is covered against known answers on synthetic files: the quiet-anchor rasteriser at
the recording edges, the loaders' missing-file errors, the pooled role-named z-scoring (zero mean and
unit SD over the pooled rows, exact values from the pooled formula, ``self``/``other`` following the
recorded mouse, skipped sessions without track names, and the slot and column-set guards), and the
per-session assembly (quiet frames, spike frames and imputed feature rows lined up exactly, under both
``clean_against`` settings). The vocal-event assembler's uncapped tile path is checked against the
tiler it wraps.
"""

from __future__ import annotations

import copy
import inspect
import json
import pathlib
from typing import ClassVar

import h5py
import numpy as np
import polars as pls
import pytest

from usv_playpen.modeling.modeling_utils import harmonize_session_columns
from usv_playpen.neural_modeling.main_neural_encoding_dispatcher import (
    filter_vocalizations,
    focal_vocal_frames,
)
from usv_playpen.neural_modeling.neural_design_assembly import (
    assemble_unit_sessions,
    assemble_unit_vocal_events,
    build_zscored_feature_frames,
    clean_window_mask,
    emitter_names,
    load_session_usvs,
    load_unit_spike_frames,
    quiet_anchor_frames,
    session_timebase,
)
from usv_playpen.neural_modeling.neural_vocal_occurrence import (
    counts_in_windows,
    quiet_tile_edges,
)

SETTINGS_DIR = pathlib.Path(__file__).resolve().parents[2] / "src" / "usv_playpen" / "_parameter_settings"


def _load_settings() -> dict:
    """Read the shipped neural-modeling settings."""
    with (SETTINGS_DIR / "neural_modeling_settings.json").open() as handle:
        return json.load(handle)


class TestCleanWindowMask:
    """The clean-prevocal overlap rule: which USVs (focal or partner) reach into an event's pre-onset
    window and so disqualify it."""

    def test_a_usv_never_excludes_itself(self):
        """A USV's own span starts exactly at its onset, so it can never sit inside its own window."""
        starts = np.array([1.0, 5.0])
        stops = np.array([1.2, 5.3])
        keep = clean_window_mask(starts, starts, stops, 0.05)
        assert keep.tolist() == [True, True]

    def test_a_preceding_usv_that_reaches_into_the_window_excludes(self):
        """A previous USV whose stop falls inside the next event's pre-onset window must drop that event,
        or the window's spikes could be a response to the earlier USV rather than a prediction."""
        starts = np.array([1.0, 1.22])
        stops = np.array([1.2, 1.4])
        # the second USV's window is (1.17, 1.22); the first USV runs to 1.2 and so overlaps it
        keep = clean_window_mask(starts, starts, stops, 0.05)
        assert keep.tolist() == [True, False]

    def test_a_preceding_usv_that_stops_before_the_window_does_not(self):
        """The overlap rule must not over-exclude: a previous USV that ended before the window opened
        leaves the event clean, otherwise dense bouts would lose events for no reason."""
        starts = np.array([1.0, 1.30])
        stops = np.array([1.2, 1.5])
        # the window is (1.25, 1.30) and the previous USV ended at 1.2
        keep = clean_window_mask(starts, starts, stops, 0.05)
        assert keep.tolist() == [True, True]

    def test_a_partner_usv_excludes_exactly_as_a_focal_one_does(self):
        """Contamination is judged against every emitter passed in: a partner USV just before the focal
        onset excludes it, and removing that partner USV from the contaminant list restores it."""
        focal = np.array([2.0])
        every_start = np.array([1.98, 2.0])
        every_stop = np.array([1.99, 2.1])
        assert clean_window_mask(focal, every_start, every_stop, 0.05).tolist() == [False]
        assert clean_window_mask(focal, every_start[1:], every_stop[1:], 0.05).tolist() == [True]

    def test_empty_inputs_return_the_right_shapes(self):
        """No events gives an empty mask, and no contaminants keeps every event, so a session with no
        USVs on either side cannot crash the assembler or drop events."""
        assert clean_window_mask(np.array([]), np.array([1.0]), np.array([2.0]), 0.05).size == 0
        assert clean_window_mask(np.array([1.0]), np.array([]), np.array([]), 0.05).tolist() == [True]


VOCAL_SETTINGS = {"vocal_emitter": "self", "exclude_noise": True, "exclude_squeaks": True,
                  "squeaks_block_quiet": True}


def _write_session(root: pathlib.Path, session_id: str, unit_id: str, *, n_frames: int,
                   frame_rate: float, usvs: list[tuple[float, float, str, float, float]],
                   spike_seconds: np.ndarray) -> None:
    """
    Description
    -----------
    Write one synthetic session tree under ``root``: the tracking H5 (zero tracks, a frame rate and
    the ``male``/``female`` track names), the USV summary CSV with the torus-position and classifier
    columns the assembler reads, and one ``cluster_data`` spike file for the unit.

    Parameters
    ----------
    root (pathlib.Path)
        Directory the session folder is created in (normally ``tmp_path``).
    session_id (str)
        Session folder name, e.g. ``"20250101_000000"``.
    unit_id (str)
        Unit identifier; the spike file is written as ``<unit_id>.npy``.
    n_frames (int)
        Number of tracking frames in the H5.
    frame_rate (float)
        Tracking frame rate in frames per second.
    usvs (list[tuple])
        One tuple per USV: ``(start, stop, emitter, qlvm1, qlvm2)``, optionally followed by a
        ``noise`` and a ``squeak`` flag (both default to ``False``, a clean vocalization).
    spike_seconds (np.ndarray)
        Spike times in seconds; written with their frame equivalents as a 2-row array.

    Returns
    -------
    None
        The files are written to disk.
    """
    video_dir = root / session_id / "video" / f"{session_id}_cam"
    video_dir.mkdir(parents=True)
    with h5py.File(video_dir / f"{session_id}_points3d_translated_rotated_metric.h5", "w") as handle:
        handle["tracks"] = np.zeros((n_frames, 2, 15, 3), dtype=np.float64)
        handle["recording_frame_rate"] = frame_rate
        handle.create_dataset("track_names", data=np.array([b"male", b"female"], dtype="S6"))

    audio_dir = root / session_id / "audio"
    audio_dir.mkdir(parents=True)
    pls.DataFrame({
        "usv_id": list(range(len(usvs))),
        "start": [c[0] for c in usvs],
        "stop": [c[1] for c in usvs],
        "emitter": [c[2] for c in usvs],
        "qlvm1": [c[3] for c in usvs],
        "qlvm2": [c[4] for c in usvs],
        # The classifier columns the vocalization filter reads. A USV tuple may carry them as a
        # 6th and 7th element; the default is a clean vocalization, which is what most tests want.
        "noise": [bool(c[5]) if len(c) > 5 else False for c in usvs],
        "squeak": [bool(c[6]) if len(c) > 6 else False for c in usvs],
    }).write_csv(audio_dir / f"{session_id}_usv_summary.csv")

    ephys_dir = root / session_id / "ephys" / "imec0" / "cluster_data"
    ephys_dir.mkdir(parents=True)
    np.save(ephys_dir / f"{unit_id}.npy",
            np.vstack([spike_seconds, spike_seconds * frame_rate]))


class TestSessionTimebase:
    """The header-only tracking read that supplies track names, frame rate and frame count."""

    def test_the_header_read_returns_names_rate_and_frame_count(self, tmp_path):
        """The three header values must come back exactly as written, since every frame index and the
        focal-animal lookup downstream are built from them."""
        _write_session(tmp_path, "20250101_000000", "imec0_cl0001_ch001_good", n_frames=1500,
                       frame_rate=150.0, usvs=[(1.0, 1.1, "male", 0.1, 0.2)],
                       spike_seconds=np.array([0.5]))
        names, rate, n_frames = session_timebase(str(tmp_path), "20250101_000000")
        assert names == ["male", "female"]
        assert rate == pytest.approx(150.0)
        assert n_frames == 1500

    def test_a_missing_tracking_file_fails_loudly(self, tmp_path):
        """A session without its tracking H5 must raise FileNotFoundError rather than return defaults
        that would silently misalign frames."""
        (tmp_path / "20250101_000000" / "video").mkdir(parents=True)
        with pytest.raises(FileNotFoundError):
            session_timebase(str(tmp_path), "20250101_000000")


class TestAssembleUnitVocalEvents:
    """The end-to-end vocal-event assembler run on synthetic session trees: which USVs become events,
    how their counts, positions and gaps are read, and how the quiet tiles are laid."""

    UNIT = "imec0_cl0001_ch001_good"

    def _settings(self) -> dict:
        """
        Description
        -----------
        Return the shipped neural-modeling settings with ``max_quiet_tiles_per_session`` lowered
        to 20, so tile subsampling is exercised on small synthetic sessions.

        Parameters
        ----------
        None

        Returns
        -------
        settings (dict)
            A fresh copy of the shipped settings with the tile cap overridden.
        """
        settings = _load_settings()
        # a tiny cap keeps the synthetic sessions' tile subsampling exercised without a huge tile set
        settings["vocal_decoding"]["max_quiet_tiles_per_session"] = 20
        return settings

    def _build(self, tmp_path, *, usvs, spike_seconds, n_frames=150000, frame_rate=150.0):
        """
        Description
        -----------
        Write one synthetic session for ``UNIT`` and return the cohort-style unit record that
        points at it as both its courtship and its vocal session.

        Parameters
        ----------
        tmp_path (pathlib.Path)
            Root the session tree is written under.
        usvs (list[tuple])
            USV tuples in the ``_write_session`` format.
        spike_seconds (np.ndarray)
            Spike times in seconds for the unit.
        n_frames (int)
            Number of tracking frames; defaults to 150000 (1000 s at 150 fps).
        frame_rate (float)
            Tracking frame rate; defaults to 150.0.

        Returns
        -------
        unit (dict)
            Unit record with ``unit_id``, ``mouse_id`` (``"male"``), ``courtship_sessions`` and
            ``vocal_sessions``.
        """
        _write_session(tmp_path, "20250101_000000", self.UNIT, n_frames=n_frames,
                       frame_rate=frame_rate, usvs=usvs, spike_seconds=spike_seconds)
        return {"unit_id": self.UNIT, "mouse_id": "male",
                "courtship_sessions": ["20250101_000000"],
                "vocal_sessions": ["20250101_000000"]}

    def test_only_focal_usvs_become_events(self, tmp_path):
        """Only the recorded animal's own USVs are decoded; a partner USV turning into an event would
        attribute the partner's repertoire to this unit."""
        usvs = [(100.0, 100.1, "male", 0.10, 0.20),
                 (200.0, 200.1, "female", 0.30, 0.40),
                 (300.0, 300.1, "male", 0.50, 0.60)]
        unit = self._build(tmp_path, usvs=usvs, spike_seconds=np.array([99.98, 299.97]))
        events = assemble_unit_vocal_events(unit, str(tmp_path), self._settings(), 0.05, 0.05)

        assert events["positions"].shape == (2, 2)
        assert events["positions"][:, 0].tolist() == [0.10, 0.50]
        assert events["session_index"].tolist() == [0, 0]
        assert events["per_session"]["20250101_000000"]["n_focal_vocalizations"] == 2

    def test_vocal_emitter_all_decodes_every_emitter_instead_of_raising(self, tmp_path):
        """``vocal_emitter = 'all'`` is no emitter filter, as in ``assemble_unit_sessions``; it used to
        index the empty list ``emitter_names`` returns for it and raise IndexError."""
        usvs = [(100.0, 100.1, "male", 0.10, 0.20),
                (200.0, 200.1, "female", 0.30, 0.40),
                (300.0, 300.1, "male", 0.50, 0.60)]
        unit = self._build(tmp_path, usvs=usvs, spike_seconds=np.array([99.98, 299.97]))
        settings = self._settings()
        settings["vocalization_settings"]["vocal_emitter"] = "all"
        events = assemble_unit_vocal_events(unit, str(tmp_path), settings, 0.05, 0.05)
        assert events["positions"][:, 0].tolist() == [0.10, 0.30, 0.50]
        np.testing.assert_allclose(events["vocalization_start"], [100.0, 200.0, 300.0])
        assert events["per_session"]["20250101_000000"]["n_focal_vocalizations"] == 3

    def test_counts_come_from_the_window_before_onset(self, tmp_path):
        """Spike counts are read from [onset - pre_offset, onset - pre_offset + width), so a spike just
        after the onset of a later USV must not leak into its count."""
        usvs = [(100.0, 100.1, "male", 0.10, 0.20), (300.0, 300.1, "male", 0.50, 0.60)]
        # two spikes inside the first USV's [99.95, 100.0) window, none inside the second's
        spikes = np.array([99.96, 99.99, 300.05])
        unit = self._build(tmp_path, usvs=usvs, spike_seconds=spikes)
        events = assemble_unit_vocal_events(unit, str(tmp_path), self._settings(), 0.05, 0.05)
        assert events["counts"].tolist() == [2.0, 0.0]

    def test_a_missing_torus_position_is_dropped_and_counted(self, tmp_path):
        """A USV with a NaN torus coordinate is removed and tallied, never carried as NaN into a
        regression that would either crash or silently down-weight it."""
        usvs = [(100.0, 100.1, "male", 0.10, 0.20), (300.0, 300.1, "male", float("nan"), 0.60)]
        unit = self._build(tmp_path, usvs=usvs, spike_seconds=np.array([99.98]))
        events = assemble_unit_vocal_events(unit, str(tmp_path), self._settings(), 0.05, 0.05)

        assert events["positions"].shape == (1, 2)
        assert np.isfinite(events["positions"]).all()
        assert events["per_session"]["20250101_000000"]["n_dropped_missing_position"] == 1

    def test_the_clean_filter_is_recorded_and_can_be_switched_off(self, tmp_path):
        """The clean-prevocal filter drops a contaminated event and records the drop, and
        ``require_clean_prevocal_bool=False`` really disables it, so the knob is honoured both ways."""
        # the second USV opens 20 ms after the first closes, so its 50 ms window is contaminated
        usvs = [(100.0, 100.10, "male", 0.10, 0.20), (100.12, 100.22, "male", 0.50, 0.60)]
        unit = self._build(tmp_path, usvs=usvs, spike_seconds=np.array([99.98]))

        settings = self._settings()
        events = assemble_unit_vocal_events(unit, str(tmp_path), settings, 0.05, 0.05)
        assert events["positions"].shape[0] == 1
        assert events["per_session"]["20250101_000000"]["n_dropped_unclean"] == 1

        settings["vocal_decoding"]["require_clean_prevocal_bool"] = False
        loose = assemble_unit_vocal_events(unit, str(tmp_path), settings, 0.05, 0.05)
        assert loose["positions"].shape[0] == 2

    def test_the_when_arrays_stack_positives_over_quiet_tiles(self, tmp_path):
        """The vocal-occurrence arrays put the USV events first (label 1, same counts as the decoding
        events) and the quiet tiles after (label 0), the layout the occurrence classifier indexes by."""
        usvs = [(100.0, 100.1, "male", 0.10, 0.20), (300.0, 300.1, "male", 0.50, 0.60)]
        unit = self._build(tmp_path, usvs=usvs, spike_seconds=np.array([99.98, 500.0]))
        events = assemble_unit_vocal_events(unit, str(tmp_path), self._settings(), 0.05, 0.05)

        windows = events["occurrence_windows"]
        n_events = events["counts"].size
        n_tiles = events["per_session"]["20250101_000000"]["n_baseline_tiles"]
        assert windows["counts"].size == n_events + n_tiles
        assert windows["labels"][:n_events].tolist() == [1.0] * n_events
        assert not windows["labels"][n_events:].any()
        assert windows["counts"][:n_events].tolist() == events["counts"].tolist()

    def test_the_when_edges_reconstruct_the_when_counts(self, tmp_path):
        """The vocal-occurrence null is a circular shift that recounts BOTH classes from the shifted train. With
        counts alone that is impossible -- a consumer can only recount the positives and has to leave
        the tiles frozen, which is a different null from the specified one. The edges make it a
        per-session recount, so they have to line up with the counts exactly."""
        usvs = [(100.0, 100.1, "male", 0.10, 0.20), (300.0, 300.1, "male", 0.50, 0.60)]
        spikes = np.sort(np.random.default_rng(0).uniform(0, 1000, 4000))
        unit = self._build(tmp_path, usvs=usvs, spike_seconds=spikes)
        events = assemble_unit_vocal_events(unit, str(tmp_path), self._settings(), 0.05, 0.05)

        windows = events["occurrence_windows"]
        assert windows["edges"].size == windows["counts"].size
        rebuilt = counts_in_windows(np.sort(spikes), windows["edges"], windows["width"])
        assert np.array_equal(rebuilt, windows["counts"])

    def test_tiles_are_disjoint_at_the_analysis_width(self, tmp_path):
        """Tiles are laid at the analysed window's OWN width, so consecutive negatives never share a
        frame. The earlier design laid them at a fixed baseline width and re-centred wider windows on
        the same lattice, which made adjacent tiles overlap by half and their counts correlated. With
        one window per run there is no second width, and the negatives are a genuine sample."""
        usvs = [(100.0, 100.1, "male", 0.10, 0.20)]
        unit = self._build(tmp_path, usvs=usvs, spike_seconds=np.array([99.98]))
        for width in (0.05, 0.10):
            events = assemble_unit_vocal_events(unit, str(tmp_path), self._settings(), 0.05, width)
            n_events = int(events["counts"].size)
            tiles = np.sort(events["occurrence_windows"]["edges"][n_events:])
            gaps = np.diff(tiles)
            assert np.all(gaps >= width - 1e-9)

    def test_tiles_are_capped_per_session(self, tmp_path):
        """``max_quiet_tiles_per_session`` bounds the negatives per session, so one long session cannot
        dominate the vocal-occurrence baseline."""
        usvs = [(100.0, 100.1, "male", 0.10, 0.20)]
        unit = self._build(tmp_path, usvs=usvs, spike_seconds=np.array([99.98]))
        events = assemble_unit_vocal_events(unit, str(tmp_path), self._settings(), 0.05, 0.05)
        assert events["per_session"]["20250101_000000"]["n_baseline_tiles"] == 20

    def test_without_a_cap_every_quiet_tile_is_kept(self, tmp_path):
        """
        Description
        -----------
        With ``max_quiet_tiles_per_session = None`` no subsampling happens: the negatives
        must be exactly the tiles ``quiet_tile_edges`` lays over the session for the
        settings' guard band, in the same order, far more than the 20 the capped test keeps.

        Parameters
        ----------
        tmp_path (pathlib.Path)
            Root the session tree is written under.

        Returns
        -------
        None
        """
        usvs = [(100.0, 100.1, "male", 0.10, 0.20)]
        unit = self._build(
            tmp_path, usvs=usvs, spike_seconds=np.array([99.98]), n_frames=30000
        )
        settings = self._settings()
        settings["vocal_decoding"]["max_quiet_tiles_per_session"] = None
        events = assemble_unit_vocal_events(unit, str(tmp_path), settings, 0.05, 0.05)
        expected = quiet_tile_edges(
            np.array([100.0]),
            np.array([100.1]),
            200.0,
            0.05,
            settings["kinematic_encoding"]["history_pre_seconds"],
            settings["kinematic_encoding"]["clean_post_seconds"],
        )
        n_events = int(events["counts"].size)
        assert expected.size > 20
        assert (
            events["per_session"]["20250101_000000"]["n_baseline_tiles"]
            == expected.size
        )
        np.testing.assert_array_equal(
            events["occurrence_windows"]["edges"][n_events:], expected
        )

    def test_a_legacy_position_column_name_fails_loudly(self, tmp_path):
        """The columns were renamed repo-wide; a loader that silently harmonised them would hide
        which embedding a result was actually computed on."""
        usvs = [(100.0, 100.1, "male", 0.10, 0.20)]
        unit = self._build(tmp_path, usvs=usvs, spike_seconds=np.array([99.98]))
        settings = self._settings()
        settings["vocal_decoding"]["usv_manifold_column_names"] = ["qlvm_dim1", "qlvm_dim2"]
        with pytest.raises(KeyError, match="qlvm_dim1"):
            assemble_unit_vocal_events(unit, str(tmp_path), settings, 0.05, 0.05)

    def test_a_unit_whose_mouse_is_absent_fails_loudly(self, tmp_path):
        """The focal animal is resolved by LOOKUP from the unit's own mouse_id, never by track slot --
        a slot says "whatever is in position 0", which is a different question. If the unit's mouse is
        not in this session's track_names, that is unrecoverable and must not be guessed at."""
        usvs = [(100.0, 100.1, "male", 0.10, 0.20)]
        unit = self._build(tmp_path, usvs=usvs, spike_seconds=np.array([99.98]))
        unit["mouse_id"] = "some_other_mouse"
        with pytest.raises(ValueError, match="not in track_names"):
            assemble_unit_vocal_events(unit, str(tmp_path), self._settings(), 0.05, 0.05)

    def test_unsorted_spike_files_still_count_correctly(self, tmp_path):
        """Window counts are read by bisection, so an out-of-order train would return wrong counts
        with no error at all. The order on disk is the writer's habit, not a guarantee here."""
        usvs = [(100.0, 100.1, "male", 0.10, 0.20)]
        shuffled = np.array([99.99, 99.96, 300.0, 99.97])
        unit = self._build(tmp_path, usvs=usvs, spike_seconds=shuffled)
        events = assemble_unit_vocal_events(unit, str(tmp_path), self._settings(), 0.05, 0.05)
        assert events["counts"].tolist() == [3.0]

    def test_out_of_order_usvs_fail_loudly(self, tmp_path):
        """The gap covariate is start[i] - stop[i-1], which is silently meaningless -- negative, even
        -- if the rows are not in temporal order."""
        usvs = [(300.0, 300.1, "male", 0.50, 0.60), (100.0, 100.1, "male", 0.10, 0.20)]
        unit = self._build(tmp_path, usvs=usvs, spike_seconds=np.array([99.98]))
        with pytest.raises(ValueError, match="ordered by start"):
            assemble_unit_vocal_events(unit, str(tmp_path), self._settings(), 0.05, 0.05)

    def test_the_recorded_gap_is_to_the_previous_focal_usv(self, tmp_path):
        """``silent_gap`` is the start of each USV minus the stop of the previous focal USV, with +inf
        for the first, so the gap covariate reflects real silence and never a made-up value."""
        usvs = [(100.0, 100.10, "male", 0.10, 0.20),
                 (100.50, 100.60, "male", 0.30, 0.40),
                 (102.00, 102.10, "male", 0.50, 0.60)]
        unit = self._build(tmp_path, usvs=usvs, spike_seconds=np.array([99.98]))
        events = assemble_unit_vocal_events(unit, str(tmp_path), self._settings(), 0.05, 0.05)
        assert np.isinf(events["silent_gap"][0])          # nothing precedes the first USV
        assert events["silent_gap"][1] == pytest.approx(0.40)
        assert events["silent_gap"][2] == pytest.approx(1.40)

    def test_clean_against_selects_whose_usvs_contaminate(self, tmp_path):
        """`vocalization_settings.clean_against` was read nowhere: every window was judged against every emitter, so
        setting it to a mouse index changed nothing at all."""
        usvs = [(100.00, 100.10, "female", 0.90, 0.90),   # partner USV, 20 ms before the focal one
                 (100.12, 100.22, "male", 0.50, 0.60)]
        unit = self._build(tmp_path, usvs=usvs, spike_seconds=np.array([99.98]))

        against_all = self._settings()
        assert against_all["vocalization_settings"]["clean_against"] == "all"
        assert against_all["vocalization_settings"]["vocal_emitter"] == "self"
        strict = assemble_unit_vocal_events(unit, str(tmp_path), against_all, 0.05, 0.05)
        assert strict["positions"].shape[0] == 0          # the partner's USV contaminates it

        against_self = self._settings()
        against_self["vocalization_settings"]["clean_against"] = "self"    # only the recorded animal's own USVs
        loose = assemble_unit_vocal_events(unit, str(tmp_path), against_self, 0.05, 0.05)
        assert loose["positions"].shape[0] == 1

    def test_sessions_are_indexed_by_slot(self, tmp_path):
        """``session_index`` is each event's position in ``vocal_sessions``, so the per-session
        permutation and folds group events by the right session."""
        for slot, session_id in enumerate(["20250101_000000", "20250101_010000"]):
            _write_session(tmp_path, session_id, self.UNIT, n_frames=150000, frame_rate=150.0,
                           usvs=[(100.0 + slot, 100.1 + slot, "male", 0.1, 0.2)],
                           spike_seconds=np.array([99.98 + slot]))
        unit = {"unit_id": self.UNIT, "mouse_id": "male",
                "courtship_sessions": ["20250101_000000", "20250101_010000"],
                "vocal_sessions": ["20250101_000000", "20250101_010000"]}
        events = assemble_unit_vocal_events(unit, str(tmp_path), self._settings(), 0.05, 0.05)
        assert events["session_index"].tolist() == [0, 1]
        assert events["session_ids"] == ["20250101_000000", "20250101_010000"]


class TestSettingsInvariants:
    """Shipped-settings invariants the assembler and the downstream decodings rely on."""

    def test_one_window_per_run_lives_in_the_shared_anchors_block(self):
        """ONE window is analysed per run, not a pair looped over. It sits in `vocalization_settings` because it is
        an event definition shared across analyses -- the vocal decoding and the nested decoding must see the
        same window, and the nested decoding
        has its own block, so a window inside `vocal_decoding` could not serve it."""
        settings = _load_settings()
        window = settings["vocalization_settings"]["spike_window"]
        assert set(window) == {"pre_offset_seconds", "width_seconds"}
        assert window["pre_offset_seconds"] == 0.05
        assert window["width_seconds"] == 0.10
        assert "analysis_windows" not in settings["vocal_decoding"]

    def test_the_shipped_window_straddles_onset_deliberately(self):
        """RULED 2026-09-11 (user): width 0.10 against a pre-offset of 0.05, so the window runs
        [start - 50 ms, start + 50 ms) and roughly 40-55% of a typical USV (median duration 87-127
        ms) falls INSIDE it.

        This is a deliberate trade and it reverses the earlier prediction-clean default. It buys
        POWER -- banked gains rise from +0.05726 to +0.07199 on cl0401 and from +0.0031 to +0.0114 on
        cl0499, a 3.7x lift on the weak unit, and passes are decided at the faint margin. It costs the
        ability to say the gain is purely PREDICTIVE: part of it is response to the USV in progress, and
        the design cannot separate the two. The user ruled that showing the activity is purely
        premotor is not required.

        The pre-window is unchanged at 50 ms, so contamination from the PREVIOUS USV is unaffected
        (3.7% of USVs cohort-wide have a gap below 50 ms); the post-onset half is inside the USV by
        construction and needs no cleanliness guard.

        The two numbers stay independent rather than collapsing into one duration precisely so this
        choice has to be made explicitly."""
        window = _load_settings()["vocalization_settings"]["spike_window"]
        assert window["width_seconds"] > window["pre_offset_seconds"]
        inside = window["width_seconds"] - window["pre_offset_seconds"]
        assert inside == pytest.approx(0.05)

    def test_vocal_decoding_holds_only_the_vocal_decoding_axes(self):
        """The nested decoding is a separate analysis with its own model, null and statistic; its
        configuration living inside the vocal decoding's block was the file's worst piece of mis-filing.

        `category_surface` and `targets` DO belong here: the categorical decode is the same claim's
        axis on a different target -- same events, same window, same folds, same currency and the same
        permutation draws -- so it is a vocal-decoding axis rather than a separate analysis."""
        settings = _load_settings()
        assert set(settings["vocal_decoding"]) == {
            "require_clean_prevocal_bool", "usv_manifold_column_names",
            "max_quiet_tiles_per_session", "vocal_occurrence", "geodesic_metrics", "tuning_surface",
            "category_surface", "targets", "discrimination_null", "record_overdispersion_index"}
        assert set(settings["nested_vocal_manifold_position_decoding"]) == {
            "behaviour_control", "reduced_model_features", "behaviour_control_model_path",
            "vm_score_mode", "region_label_column", "min_region_events", "lambda_smooth", "l2_reg",
            "smoothness_derivative_order", "huber_delta", "learning_rate", "max_iter", "tol", "random_state",
            "behaviour_control_n_lags", "sigma_floor", "prevocal_window_n_bins", "rate_transform",
            "rate_basis", "statistic", "discrimination_null", "behaviour_prior"}

    def test_the_nested_decoding_matches_p1_where_it_must(self):
        """The nested decoding is "P1 plus the neuron", so the control has to be fitted and SCORED the way P1
        fitted and scored it. A silent divergence here would make the reduced model suboptimal for
        the test, which biases the test toward passing -- the wrong direction for a positive result."""
        nested = _load_settings()["nested_vocal_manifold_position_decoding"]
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
        """Every key here is read by something -- unlike the block that previously carried the name.

        `min_decoding_spikes` is read by `neural_category_decoding.decoding_spike_sufficiency`. It was
        added because the decoding axes had NO spike gate at all: measured over 1,231 units, sd(gain)
        runs 0.029 below 50 spikes against 0.014 above, and a unit with under 25 spikes returned a gain
        of 0.70 -- three times the cohort's strongest tuned unit."""
        assert set(_load_settings()["data_sufficiency"]) == {
            "min_courtship_sessions", "min_emitter_usvs_per_session", "min_vocal_sessions",
            "min_decoding_spikes", "single_session_inner_split_blocks"}

    def test_the_session_gate_is_named_for_the_emitter_it_follows(self):
        """The gate counts USVs from whoever `vocalization_settings.vocal_emitter` names, so a name
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
        assert "calib_n_steps" not in settings["vocal_decoding"]["vocal_occurrence"]["solver"]
        assert "min_focal_usvs" not in settings["vocal_decoding"]
        assert "decorrelation_lag_seconds" not in settings["null"]

    def test_the_neural_kinematic_features_still_equal_the_behavioural_ones(self):
        """The nested decoding asks whether a neuron adds beyond behaviour, where "behaviour" is the feature set P1
        fits. The neural settings own their copy so a neural run can diverge deliberately, but the
        shipped default has to match or the comparison silently stops meaning what it says."""
        with (SETTINGS_DIR / "modeling_settings.json").open() as handle:
            modeling = json.load(handle)
        assert _load_settings()["kinematic_features"] == modeling["kinematic_features"]


class TestFocalVocalFrames:
    """The kinematic encoding's transfer resolves the focal animal separately from the assembler, and did it by
    substring match with a bare [0] index."""

    UNIT = "imec0_cl0001_ch001_good"

    def test_a_session_where_the_focal_animal_never_vocalized_yields_no_frames(self, tmp_path):
        """Ordinary, not exceptional: it happens in 8 of 77 courtship sessions. It used to raise
        IndexError and kill the whole fold."""
        _write_session(tmp_path, "20250101_000000", self.UNIT, n_frames=150000, frame_rate=150.0,
                       usvs=[(100.0, 100.1, "female", 0.1, 0.2)],   # partner only
                       spike_seconds=np.array([99.98]))
        session = {"n_frames": 150000, "track_names": ["male", "female"]}
        frames, pointer, gaps = focal_vocal_frames(session, "20250101_000000", str(tmp_path),
                                                   "male", 150.0, 600, VOCAL_SETTINGS)
        assert frames.size == 0
        assert gaps.size == 0
        assert pointer.tolist() == [0]

    def test_the_emitter_is_matched_exactly_not_by_substring(self, tmp_path):
        """A bare mouse id is a prefix of the same animal's suffixed form, so a substring match can
        silently pick up another animal's USVs."""
        _write_session(tmp_path, "20250101_000000", self.UNIT, n_frames=150000, frame_rate=150.0,
                       usvs=[(100.0, 100.1, "147366_1", 0.1, 0.2),
                              (200.0, 200.1, "147366", 0.3, 0.4)],
                       spike_seconds=np.array([99.98]))
        session = {"n_frames": 150000, "track_names": ["147366", "147366_1"]}
        frames, _pointer, gaps = focal_vocal_frames(session, "20250101_000000", str(tmp_path),
                                                    "147366", 150.0, 600, VOCAL_SETTINGS)
        assert gaps.size == 1                     # only the exact-match USV, not the "147366_1" one
        assert frames.min() >= int(200.0 * 150.0)

    @pytest.mark.parametrize(("emitter", "expected_starts"), [("self", [100.0]), ("partner", [200.0]),
                                                              ("all", [100.0, 200.0])])
    def test_the_transfer_frames_follow_vocal_emitter(self, tmp_path, emitter, expected_starts):
        """The transfer frames read ``vocalization_settings.vocal_emitter`` exactly as the vocal-side
        assembler does, so pointing it at the partner (or at every emitter) moves the transfer and the
        vocal decoding together; they used to match the unit's own mouse unconditionally."""
        _write_session(tmp_path, "20250101_000000", self.UNIT, n_frames=150000, frame_rate=150.0,
                       usvs=[(100.0, 100.1, "male", 0.1, 0.2), (200.0, 200.1, "female", 0.3, 0.4)],
                       spike_seconds=np.array([99.98]))
        session = {"n_frames": 150000, "track_names": ["male", "female"]}
        frames, pointer, gaps = focal_vocal_frames(session, "20250101_000000", str(tmp_path), "male",
                                                   150.0, 600, {**VOCAL_SETTINGS, "vocal_emitter": emitter})
        assert gaps.size == len(expected_starts)
        span_starts = frames[pointer[:-1]]
        np.testing.assert_array_equal(span_starts, [int(start * 150.0) for start in expected_starts])


class TestQuietSideKeepsEverySession:
    """The load-bearing half of the two-session-set design, and the easier one to break by accident.

    A session in which the recorded animal barely vocalized is EXCLUDED from everything scored on
    vocalizations and must stay in everything fitted on silence -- it has more quiet, not less. If a
    future change points the kinematic assembler at `vocal_sessions`, the kinematic encoding would quietly lose
    training data for the exact sessions that offer the most of it."""

    UNIT = "imec0_cl0001_ch001_good"

    def test_the_kinematic_assembler_reads_courtship_sessions_not_vocal_sessions(self, monkeypatch):
        """The kinematic feature loader must receive every courtship session (the thin one included) and
        the unit's own mouse id, so quiet-side training data is never lost to the vocal-session gate."""
        seen = {}

        def _spy(session_dirs, **kwargs):
            """
            Description
            -----------
            Stand-in for ``build_zscored_feature_frames`` that records the session directories and
            the ``recorded_mouse_id`` it was handed, then returns empty feature containers.

            Parameters
            ----------
            session_dirs (list[str])
                Session directories the assembler asked for.
            **kwargs (dict)
                Remaining keyword arguments; ``recorded_mouse_id`` is read.

            Returns
            -------
            empty (tuple[dict, dict, dict, dict])
                Four empty dicts, the shape of the real loader's return.
            """
            seen["dirs"] = list(session_dirs)
            seen["mouse"] = kwargs["recorded_mouse_id"]
            return {}, {}, {}, {}

        monkeypatch.setattr(
            "usv_playpen.neural_modeling.neural_design_assembly.build_zscored_feature_frames", _spy)
        unit = {"unit_id": self.UNIT, "mouse_id": "male",
                "courtship_sessions": ["s_rich", "s_thin"],
                "vocal_sessions": ["s_rich"]}
        assemble_unit_sessions(unit, "/data", {}, 4.0, 2.0, "all", VOCAL_SETTINGS)

        assert seen["dirs"] == ["/data/s_rich", "/data/s_thin"]
        assert "/data/s_thin" in seen["dirs"]        # the thin session is FITTED on, not dropped
        # the `self.` role comes from the unit's own mouse, not a configured slot index
        assert seen["mouse"] == "male"


class TestSelfAndPartnerAreResolvedByName:
    """Who is `self` and who is `partner` must never come from a track SLOT. A slot says "whatever
    sits in position 0", which is a different statement from "this animal" and needs its own assertion to
    be safe. Both halves of the pipeline -- the emitter filter on the vocal side and the
    `self.`/`other.` feature split on the kinematic side -- now answer it the same way: `self` by NAME
    lookup of the unit's own `mouse_id`, `partner` by EXCLUSION."""

    NAMES: ClassVar[list[str]] = ["male", "female"]

    def test_self_is_the_units_own_mouse_not_a_slot(self):
        """``self`` resolves to whichever name the unit's mouse carries, whatever slot it occupies, so
        flipping the track order cannot flip which animal is focal."""
        assert emitter_names(self.NAMES, "female", "self") == ["female"]
        assert emitter_names(self.NAMES, "male", "self") == ["male"]

    def test_partner_is_everyone_else_and_does_not_assume_two_tracks(self):
        """Exclusion is length-agnostic; a two-slot flip (`abs(idx - 1)`) is not, and would pick an
        arbitrary animal for a third track with no error."""
        assert emitter_names(self.NAMES, "male", "partner") == ["female"]
        trio = ["male", "female", "pup"]
        assert emitter_names(trio, "female", "partner") == ["male", "pup"]

    def test_all_is_no_filter_rather_than_the_two_track_names(self):
        """A session's USV table can carry rows whose emitter was never assigned, and a USV nobody was
        credited with is still a USV the animal heard."""
        assert emitter_names(self.NAMES, "male", "all") == []

    def test_an_absent_mouse_raises_rather_than_being_guessed_at(self):
        """A mouse id missing from the track names must raise, never fall back to a slot."""
        with pytest.raises(ValueError, match="not in track_names"):
            emitter_names(self.NAMES, "some_other_mouse", "self")

    def test_an_unknown_spec_raises(self):
        """Only ``self``/``partner``/``all`` are valid; a legacy ``focal`` spec must raise rather than
        be interpreted."""
        with pytest.raises(ValueError, match="emitter spec must be"):
            emitter_names(self.NAMES, "male", "focal")

    def test_the_feature_split_asserts_a_dyad(self, monkeypatch):
        """The P1 utility downstream derives the other animal as `abs(idx - 1)`, which is the partner
        only in a dyad. Measured 2 track names in 73 of 73 cohort sessions -- so a third one would go
        unnoticed, which is what the assertion is for."""
        def _three_tracks(**_kwargs):
            """
            Description
            -----------
            Stand-in for ``load_behavioral_feature_data`` that reports three track names for one
            session, which the dyad assertion must reject.

            Parameters
            ----------
            **_kwargs (dict)
                Ignored loader arguments.

            Returns
            -------
            loaded (tuple[dict, dict, dict])
                Feature frames, frame rates and track names keyed by session.
            """
            return ({"s1": pls.DataFrame({"male.speed": [0.0]})}, {"s1": 150.0},
                    {"s1": ["male", "female", "pup"]})

        monkeypatch.setattr(
            "usv_playpen.neural_modeling.neural_design_assembly.load_behavioral_feature_data",
            _three_tracks)
        with pytest.raises(ValueError, match="assumes a dyad"):
            build_zscored_feature_frames(["/data/s1"], {}, "male")


class TestOneQuietDefinition:
    """The plan requires ONE quiet definition across the kinematic encoding, the vocal-occurrence
    negatives and the vocal-decoding baselines. Two code paths that can disagree is how that breaks -- and it did: `clean_against` was
    read by the vocal-side assembler and ignored by the kinematic one, so setting it to a mouse index
    would have moved the vocal decoding's tiles while leaving the kinematic encoding's anchors all-emitter."""

    def test_the_kinematic_assembler_takes_clean_against(self):
        """The kinematic assembler must accept ``clean_against`` so both sides can share one quiet
        definition."""
        assert "clean_against" in inspect.signature(assemble_unit_sessions).parameters

    def test_the_kinematic_assembler_carries_no_dead_onset_set(self):
        """The assembler used to build USV-onset anchors (and take ``vocal_emitter`` for them) that no
        analysis read; the kinematic encoding's vocal frames come from ``focal_vocal_frames``, which
        reads ``vocal_emitter`` itself. The dead set and its argument are gone."""
        assert "vocal_emitter" not in inspect.signature(assemble_unit_sessions).parameters

    def test_the_dispatcher_passes_the_same_setting_to_both_sides(self):
        """Every ``assemble_unit_sessions`` site in the dispatcher must forward the shared
        ``clean_against`` value, so the kinematic side cannot drift from the vocal side through a
        hard-coded argument."""
        source = (pathlib.Path(__file__).resolve().parents[2] / "src" / "usv_playpen"
                  / "neural_modeling" / "main_neural_encoding_dispatcher.py").read_text()
        # every assemble_unit_sessions call site must hand it the shared anchors values
        assert source.count("assemble_unit_sessions(") == source.count(
            'settings["vocalization_settings"]["clean_against"]')


class TestVocalSessionScope:
    """The cohort builder hands out two session sets, and the vocal-side assembler must read the
    vocal one. A session the male barely vocalized in stays available to the kinematic encoding,
    which needs silence, and is kept away from everything scored on the USVs themselves."""

    UNIT = "imec0_cl0001_ch001_good"

    def test_only_vocal_sessions_contribute_events(self, tmp_path):
        """A courtship session absent from ``vocal_sessions`` must contribute no events and no session
        index, while listing both sessions yields both."""
        for slot, session_id in enumerate(["20250101_000000", "20250101_010000"]):
            _write_session(tmp_path, session_id, self.UNIT, n_frames=150000, frame_rate=150.0,
                           usvs=[(100.0 + slot, 100.1 + slot, "male", 0.1, 0.2)],
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


class TestVocalizationFilter:
    """Noise and squeaks are excluded, and the two categories deliberately have DIFFERENT scopes.

    Noise is not a vocalization, so it is dropped everywhere including the quiet guard. A squeak is a
    real sound of a different type, so it leaves the analysed USV set but -- at the shipped default --
    still stops a frame counting as quiet.
    """

    TABLE = pls.DataFrame({"start": [0.0, 1.0, 2.0, 3.0],
                          "noise": [True, False, False, False],
                          "squeak": [False, True, False, False]})

    def test_analysed_scope_drops_both(self):
        """The analysed set excludes both noise and squeaks, so only genuine USVs are decoded."""
        kept = filter_vocalizations(self.TABLE, VOCAL_SETTINGS, "analysed")
        assert kept["start"].to_list() == [2.0, 3.0]

    def test_guard_keeps_squeaks_by_default(self):
        """The animal DID vocalize, so the surrounding frames are not silent."""
        kept = filter_vocalizations(self.TABLE, VOCAL_SETTINGS, "guard")
        assert kept["start"].to_list() == [1.0, 2.0, 3.0]

    def test_guard_drops_squeaks_when_they_do_not_block_quiet(self):
        """With ``squeaks_block_quiet=False`` the guard drops squeaks too, so the knob really controls
        whether squeaks veto quiet frames."""
        settings = dict(VOCAL_SETTINGS, squeaks_block_quiet=False)
        kept = filter_vocalizations(self.TABLE, settings, "guard")
        assert kept["start"].to_list() == [2.0, 3.0]

    def test_noise_is_dropped_from_the_guard_too(self):
        """A microphone artefact must never veto a genuinely silent frame."""
        for scope in ("analysed", "guard"):
            kept = filter_vocalizations(self.TABLE, VOCAL_SETTINGS, scope)
            assert 0.0 not in kept["start"].to_list()

    def test_flags_off_keeps_everything(self):
        """With both exclusion flags off neither scope removes a row, so the flags are the only thing
        that filters."""
        settings = {"exclude_noise": False, "exclude_squeaks": False, "squeaks_block_quiet": True}
        for scope in ("analysed", "guard"):
            assert filter_vocalizations(self.TABLE, settings, scope).height == 4

    def test_a_missing_column_raises_rather_than_silently_skipping(self):
        """An unfiltered run must never be indistinguishable from a filtered one in its numbers."""
        table = pls.DataFrame({"start": [0.0, 1.0]})
        with pytest.raises(KeyError, match="no such column"):
            filter_vocalizations(table, VOCAL_SETTINGS, "analysed")

    def test_an_unknown_scope_raises(self):
        """A misspelled scope must raise rather than default to one of the two filters."""
        with pytest.raises(ValueError, match="scope must be one of"):
            filter_vocalizations(self.TABLE, VOCAL_SETTINGS, "everything")

    def test_the_filter_preserves_row_order(self):
        """Filtering must not reorder rows, since the assembler's start-order check and the gap
        covariate read the table in its stored order."""
        table = pls.DataFrame({"start": [5.0, 1.0, 9.0], "noise": [False] * 3, "squeak": [False] * 3})
        assert filter_vocalizations(table, VOCAL_SETTINGS, "analysed")["start"].to_list() == [5.0, 1.0, 9.0]


class TestQuietAnchorEdgeCases:
    """Boundary behaviour of the quiet-anchor rasteriser."""

    def test_a_usv_past_the_recording_forbids_nothing(self):
        """
        Description
        -----------
        A USV whose forbidden span starts after the last frame clips to an empty span and
        must veto no frame; with ``clean_post_seconds = 0`` there is no tail trim either,
        so the anchors are every frame with a full 1 s history: ``10 .. 99`` at 10 fps.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        frames = quiet_anchor_frames(
            np.array([50.0]), np.array([51.0]), 100, 10.0, 1.0, 0.0
        )
        np.testing.assert_array_equal(frames, np.arange(10, 100))

    def test_a_post_buffer_trims_the_recording_tail(self):
        """
        Description
        -----------
        The same session with a 0.5 s post-buffer loses its last five frames, whose
        forward window runs past the end of the recording and so is not fully observed.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        frames = quiet_anchor_frames(
            np.array([50.0]), np.array([51.0]), 100, 10.0, 1.0, 0.5
        )
        np.testing.assert_array_equal(frames, np.arange(10, 95))


class TestMissingInputFiles:
    """The per-session loaders fail loudly when their file is absent."""

    def test_a_missing_spike_file_raises(self, tmp_path):
        """
        Description
        -----------
        A unit with no ``cluster_data`` file in the session must raise
        ``FileNotFoundError`` naming the path, never return an empty train that would read
        as a silent unit.

        Parameters
        ----------
        tmp_path (pathlib.Path)
            Root the (empty) session tree is created under.

        Returns
        -------
        None
        """
        (tmp_path / "20250101_000000" / "ephys" / "imec0" / "cluster_data").mkdir(
            parents=True
        )
        with pytest.raises(FileNotFoundError, match="spike file not found"):
            load_unit_spike_frames(
                str(tmp_path), "20250101_000000", "imec0_cl0001_ch001_good"
            )

    def test_a_missing_usv_summary_raises(self, tmp_path):
        """
        Description
        -----------
        A session with no ``usv_summary`` CSV must raise ``FileNotFoundError`` rather than
        be treated as a session without vocalizations, which would make every frame quiet.

        Parameters
        ----------
        tmp_path (pathlib.Path)
            Root the (empty) session tree is created under.

        Returns
        -------
        None
        """
        (tmp_path / "20250101_000000" / "audio").mkdir(parents=True)
        with pytest.raises(FileNotFoundError, match="usv_summary not found"):
            load_session_usvs(str(tmp_path), "20250101_000000")


def _write_behaviour(root: pathlib.Path, session_id: str, frame: pls.DataFrame) -> None:
    """
    Description
    -----------
    Write a session's behavioural-features CSV next to the tracking H5 that
    ``_write_session`` created, under the file name the behavioural loader globs for.

    Parameters
    ----------
    root (pathlib.Path)
        Directory holding the session folder.
    session_id (str)
        Session folder name.
    frame (pls.DataFrame)
        The per-frame feature table to write.

    Returns
    -------
    None
        The CSV is written to disk.
    """
    video_dir = root / session_id / "video" / f"{session_id}_cam"
    frame.write_csv(
        video_dir
        / f"{session_id}_points3d_translated_rotated_metric_behavioral_features.csv"
    )


def _behaviour_frame(seed: int, n_frames: int) -> pls.DataFrame:
    """
    Description
    -----------
    A synthetic behavioural-features table for a ``male``/``female`` dyad: both animals'
    ``speed``, the dyadic ``head-head`` angle, and one column (``male.nose_x``) that is in
    no kinematic bucket and so must not survive column selection. All values sit inside
    the physical bounds the z-scoring clips to.

    Parameters
    ----------
    seed (int)
        Seed for the feature values.
    n_frames (int)
        Number of rows.

    Returns
    -------
    frame (pls.DataFrame)
        The table, columns ``male.speed``, ``female.speed``, ``male-female.head-head``,
        ``male.nose_x``.
    """
    rng = np.random.default_rng(seed)
    return pls.DataFrame(
        {
            "male.speed": rng.uniform(0.0, 40.0, n_frames),
            "female.speed": rng.uniform(5.0, 50.0, n_frames),
            "male-female.head-head": rng.uniform(0.0, 80.0, n_frames),
            "male.nose_x": rng.normal(size=n_frames),
        }
    )


class TestBuildZscoredFeatureFrames:
    """The pooled, role-named z-scored feature frames, built from synthetic files on disk."""

    SESSIONS: ClassVar[list[str]] = ["20250101_000000", "20250101_010000"]
    N_FRAMES = 200

    def _sessions(self, tmp_path) -> dict:
        """
        Description
        -----------
        Write two synthetic sessions (tracking H5 with ``male``/``female`` track names at
        10 fps, behavioural features, an empty-ish USV summary and a spike file), with one
        out-of-range ``male.speed`` value (60 against the 0-54 bound) at row 3 of the
        first session.

        Parameters
        ----------
        tmp_path (pathlib.Path)
            Root the session trees are written under.

        Returns
        -------
        frames (dict)
            The raw behavioural table written for each session, keyed by session id.
        """
        frames = {}
        for slot, session_id in enumerate(self.SESSIONS):
            _write_session(
                tmp_path,
                session_id,
                "imec0_cl0001_ch001_good",
                n_frames=self.N_FRAMES,
                frame_rate=10.0,
                usvs=[(5.0, 5.25, "male", 0.1, 0.2)],
                spike_seconds=np.array([1.0]),
            )
            frame = _behaviour_frame(slot, self.N_FRAMES)
            if slot == 0:
                frame = frame.with_columns(
                    pls.when(pls.int_range(pls.len()) == 3)
                    .then(60.0)
                    .otherwise(pls.col("male.speed"))
                    .alias("male.speed")
                )
            _write_behaviour(tmp_path, session_id, frame)
            frames[session_id] = frame
        return frames

    @staticmethod
    def _pooled(frames: dict, columns: list[str], upper: float) -> tuple[float, float]:
        """
        Description
        -----------
        The pooled mean and standard deviation (``ddof = 1``, float32 values, as the
        z-scoring computes them) of the given raw columns over every session, after
        dropping values above the physical bound.

        Parameters
        ----------
        frames (dict)
            Raw behavioural tables keyed by session id.
        columns (list[str])
            The columns sharing one feature suffix.
        upper (float)
            Physical upper bound; larger values are excluded, as the clip nulls them.

        Returns
        -------
        mean (float)
            Pooled mean.
        std (float)
            Pooled sample standard deviation.
        """
        pooled = np.concatenate(
            [
                frame[column].to_numpy()
                for frame in frames.values()
                for column in columns
            ]
        ).astype(np.float32)
        pooled = pooled[pooled <= upper].astype(np.float64)
        return float(pooled.mean()), float(pooled.std(ddof=1))

    def test_every_feature_has_zero_mean_and_unit_sd_over_the_pooled_rows(
        self, tmp_path
    ):
        """
        Description
        -----------
        The ruler is pooled over ALL sessions and over both role columns sharing a suffix
        (``self.speed`` with ``other.speed``), so over those pooled non-null values each
        feature must have mean 0 and sample SD 1, while the unselected column is gone and
        the out-of-range value is null.

        Parameters
        ----------
        tmp_path (pathlib.Path)
            Root the session trees are written under.

        Returns
        -------
        None
        """
        self._sessions(tmp_path)
        settings = _load_settings()
        feats, camera_fr, names, suffixes = build_zscored_feature_frames(
            [str(tmp_path / s) for s in self.SESSIONS],
            settings["kinematic_features"],
            "male",
        )
        assert sorted(feats) == self.SESSIONS
        assert suffixes == ["head-head", "speed"]
        assert camera_fr == dict.fromkeys(self.SESSIONS, 10.0)
        assert names == {s: ["male", "female"] for s in self.SESSIONS}
        for frame in feats.values():
            assert frame.columns == ["head-head", "other.speed", "self.speed"]
        assert feats[self.SESSIONS[0]]["self.speed"][3] is None

        for columns in (["head-head"], ["self.speed", "other.speed"]):
            pooled = np.concatenate(
                [
                    frame[column].drop_nulls().to_numpy()
                    for frame in feats.values()
                    for column in columns
                ]
            )
            assert pooled.mean() == pytest.approx(0.0, abs=1e-6)
            assert pooled.std(ddof=1) == pytest.approx(1.0, abs=1e-6)

    def test_values_follow_the_pooled_formula_with_roles_from_the_recorded_mouse(
        self, tmp_path
    ):
        """
        Description
        -----------
        With the FEMALE as the recorded animal, ``self.speed`` must be her raw speed and
        ``other.speed`` the male's, each z-scored with the speed statistics pooled over
        both animals and both sessions; ``head-head`` uses its own pooled statistics.

        Parameters
        ----------
        tmp_path (pathlib.Path)
            Root the session trees are written under.

        Returns
        -------
        None
        """
        frames = self._sessions(tmp_path)
        settings = _load_settings()
        feats, _camera_fr, _names, _suffixes = build_zscored_feature_frames(
            [str(tmp_path / s) for s in self.SESSIONS],
            settings["kinematic_features"],
            "female",
        )
        speed_mean, speed_std = self._pooled(
            frames, ["male.speed", "female.speed"], 54.0
        )
        angle_mean, angle_std = self._pooled(frames, ["male-female.head-head"], 90.0)
        for session_id in self.SESSIONS:
            raw, zscored = frames[session_id], feats[session_id]
            np.testing.assert_allclose(
                zscored["self.speed"].to_numpy(),
                (raw["female.speed"].to_numpy() - speed_mean) / speed_std,
                atol=1e-5,
            )
            other = zscored["other.speed"].to_numpy().astype(np.float64)
            expected_other = (raw["male.speed"].to_numpy() - speed_mean) / speed_std
            expected_other[raw["male.speed"].to_numpy() > 54.0] = np.nan
            np.testing.assert_allclose(other, expected_other, atol=1e-5)
            np.testing.assert_allclose(
                zscored["head-head"].to_numpy(),
                (raw["male-female.head-head"].to_numpy() - angle_mean) / angle_std,
                atol=1e-5,
            )

    def test_a_session_without_track_names_is_skipped(self, monkeypatch):
        """
        Description
        -----------
        A session the loader returned features for but no track names cannot have its
        ``self``/``other`` roles resolved, so it is left out of the output rather than
        guessed at; the remaining session is still z-scored (one row each side of the
        mean gives exactly -0.7071 and +0.7071).

        Parameters
        ----------
        monkeypatch (pytest.MonkeyPatch)
            Replaces the behavioural loader.

        Returns
        -------
        None
        """

        def _loader(**_kwargs):
            """
            Description
            -----------
            Stand-in for ``load_behavioral_feature_data`` returning two sessions' features
            but track names for ``s1`` only.

            Parameters
            ----------
            **_kwargs (dict)
                Ignored loader arguments.

            Returns
            -------
            loaded (tuple[dict, dict, dict])
                Feature frames, frame rates and track names keyed by session.
            """
            frame = pls.DataFrame(
                {"male.speed": [1.0, 3.0], "female.speed": [2.0, 2.0]}
            )
            return (
                {"s1": frame, "s2": frame.clone()},
                {"s1": 10.0, "s2": 10.0},
                {"s1": ["male", "female"]},
            )

        monkeypatch.setattr(
            "usv_playpen.neural_modeling.neural_design_assembly.load_behavioral_feature_data",
            _loader,
        )
        feats, _camera_fr, _names, _suffixes = build_zscored_feature_frames(
            ["/data/s1", "/data/s2"], _load_settings()["kinematic_features"], "male"
        )
        assert list(feats) == ["s1"]
        # pooled speed values {1, 3, 2, 2}: mean 2, sample SD sqrt(2/3)
        np.testing.assert_allclose(
            feats["s1"]["self.speed"].to_numpy(),
            np.array([-1.0, 1.0]) / np.sqrt(2.0 / 3.0),
            rtol=1e-6,
        )

    def test_a_recorded_mouse_in_different_slots_raises(self, monkeypatch):
        """
        Description
        -----------
        The column harmonisation takes one pair of slot indices, so a recorded animal at
        slot 0 in one session and slot 1 in another must raise ``ValueError`` rather than
        silently harmonise with whichever session came last.

        Parameters
        ----------
        monkeypatch (pytest.MonkeyPatch)
            Replaces the behavioural loader.

        Returns
        -------
        None
        """

        def _loader(**_kwargs):
            """
            Description
            -----------
            Stand-in for ``load_behavioral_feature_data`` whose second session lists the
            two animals in the opposite track order.

            Parameters
            ----------
            **_kwargs (dict)
                Ignored loader arguments.

            Returns
            -------
            loaded (tuple[dict, dict, dict])
                Feature frames, frame rates and track names keyed by session.
            """
            frame = pls.DataFrame({"male.speed": [1.0], "female.speed": [2.0]})
            return (
                {"s1": frame, "s2": frame.clone()},
                {"s1": 10.0, "s2": 10.0},
                {"s1": ["male", "female"], "s2": ["female", "male"]},
            )

        monkeypatch.setattr(
            "usv_playpen.neural_modeling.neural_design_assembly.load_behavioral_feature_data",
            _loader,
        )
        with pytest.raises(ValueError, match="different track slot"):
            build_zscored_feature_frames(
                ["/data/s1", "/data/s2"], _load_settings()["kinematic_features"], "male"
            )

    def test_a_canonical_column_mismatch_raises(self, monkeypatch, tmp_path):
        """
        Description
        -----------
        After canonicalisation every session must carry the same column set, because the
        design is read by POSITION across sessions; if a session ends up missing a column
        (forced here by dropping ``female.speed`` from the second session after
        harmonisation), the assembler must raise rather than misalign features.

        Parameters
        ----------
        monkeypatch (pytest.MonkeyPatch)
            Wraps the column harmonisation.
        tmp_path (pathlib.Path)
            Root the session trees are written under.

        Returns
        -------
        None
        """
        self._sessions(tmp_path)

        def _drop_one(**kwargs):
            """
            Description
            -----------
            Run the real ``harmonize_session_columns`` and then drop ``female.speed`` from
            the second session, producing the inconsistent column sets the guard exists for.

            Parameters
            ----------
            **kwargs (dict)
                Forwarded to ``harmonize_session_columns``.

            Returns
            -------
            harmonised (tuple[dict, list])
                The per-session frames and the suffix list.
            """
            processed, suffixes = harmonize_session_columns(**kwargs)
            processed[self.SESSIONS[1]] = processed[self.SESSIONS[1]].drop(
                "female.speed"
            )
            return processed, suffixes

        monkeypatch.setattr(
            "usv_playpen.neural_modeling.neural_design_assembly.harmonize_session_columns",
            _drop_one,
        )
        with pytest.raises(ValueError, match="differs from the reference"):
            build_zscored_feature_frames(
                [str(tmp_path / s) for s in self.SESSIONS],
                _load_settings()["kinematic_features"],
                "male",
            )


class TestAssembleUnitSessions:
    """The per-session arrays the kinematic encoder is fitted on, from synthetic files."""

    UNIT = "imec0_cl0001_ch001_good"
    SESSIONS: ClassVar[list[str]] = ["20250101_000000", "20250101_010000"]

    def _unit(self, tmp_path) -> tuple[dict, dict]:
        """
        Description
        -----------
        Write two complete sessions (200 frames at 10 fps) and a third courtship session
        with no behavioural-features file. Each complete session holds a male USV at
        ``[5.0, 5.25]`` s, a female USV at ``[12.0, 12.5]`` s and a male NOISE row at
        ``[16.0, 16.25]`` s, and spikes at 1.0, 2.6 and 7.3 s. Row 3 of the first session's
        ``male.speed`` is out of range.

        Parameters
        ----------
        tmp_path (pathlib.Path)
            Root the session trees are written under.

        Returns
        -------
        unit (dict)
            Cohort-style record listing all three sessions as courtship sessions.
        frames (dict)
            The raw behavioural table written for each complete session.
        """
        usvs = [
            (5.0, 5.25, "male", 0.1, 0.2),
            (12.0, 12.5, "female", 0.3, 0.4),
            (16.0, 16.25, "male", 0.5, 0.6, True),
        ]
        frames = {}
        for slot, session_id in enumerate([*self.SESSIONS, "20250101_020000"]):
            _write_session(
                tmp_path,
                session_id,
                self.UNIT,
                n_frames=200,
                frame_rate=10.0,
                usvs=usvs,
                spike_seconds=np.array([1.0, 2.6, 7.3]),
            )
            if slot < len(self.SESSIONS):
                frame = _behaviour_frame(slot, 200)
                if slot == 0:
                    frame = frame.with_columns(
                        pls.when(pls.int_range(pls.len()) == 3)
                        .then(60.0)
                        .otherwise(pls.col("male.speed"))
                        .alias("male.speed")
                    )
                _write_behaviour(tmp_path, session_id, frame)
                frames[session_id] = frame
        unit = {
            "unit_id": self.UNIT,
            "mouse_id": "male",
            "courtship_sessions": [*self.SESSIONS, "20250101_020000"],
            "vocal_sessions": list(self.SESSIONS),
        }
        return unit, frames

    def test_rows_units_and_quiet_frames_line_up(self, tmp_path):
        """
        Description
        -----------
        Under ``clean_against = 'all'`` with a 1 s history and a 0.5 s post-buffer, the
        male USV forbids frames 45-62, the female USV 115-134 and the recording edges
        0-9 and 195-199, while the noise row forbids nothing; spike frames are
        ``round(seconds * fps)``; the feature time series is the pooled z-score with the
        out-of-range frame imputed to 0; and the session without behavioural features is
        absent.

        Parameters
        ----------
        tmp_path (pathlib.Path)
            Root the session trees are written under.

        Returns
        -------
        None
        """
        unit, frames = self._unit(tmp_path)
        settings = _load_settings()
        per_session = assemble_unit_sessions(
            unit,
            str(tmp_path),
            settings["kinematic_features"],
            1.0,
            0.5,
            "all",
            copy.deepcopy(settings["vocalization_settings"]),
        )
        assert sorted(per_session) == self.SESSIONS
        speed = np.concatenate(
            [
                frame[c].to_numpy()
                for frame in frames.values()
                for c in ("male.speed", "female.speed")
            ]
        ).astype(np.float32)
        speed = speed[speed <= 54.0].astype(np.float64)
        expected_quiet = np.r_[
            np.arange(10, 45), np.arange(63, 115), np.arange(135, 195)
        ]
        for session_id in self.SESSIONS:
            session = per_session[session_id]
            assert session["feature_names"] == [
                "head-head",
                "other.speed",
                "self.speed",
            ]
            assert session["fps"] == 10.0
            assert session["n_frames"] == 200
            assert session["feature_time_series"].shape == (200, 3)
            assert session["track_names"] == ["male", "female"]
            np.testing.assert_array_equal(session["spike_frames"], [10, 26, 73])
            np.testing.assert_array_equal(session["quiet"], expected_quiet)
            expected_self = (
                frames[session_id]["male.speed"].to_numpy() - speed.mean()
            ) / speed.std(ddof=1)
            if session_id == self.SESSIONS[0]:
                expected_self[3] = 0.0
            np.testing.assert_allclose(
                session["feature_time_series"][:, 2], expected_self, atol=1e-5
            )

    def test_clean_against_self_ignores_the_partner_usv(self, tmp_path):
        """
        Description
        -----------
        Under ``clean_against = 'self'`` only the recorded male's own USV makes frames
        unclean, so the female USV's frames 115-134 become quiet again.

        Parameters
        ----------
        tmp_path (pathlib.Path)
            Root the session trees are written under.

        Returns
        -------
        None
        """
        unit, _frames = self._unit(tmp_path)
        settings = _load_settings()
        per_session = assemble_unit_sessions(
            unit,
            str(tmp_path),
            settings["kinematic_features"],
            1.0,
            0.5,
            "self",
            copy.deepcopy(settings["vocalization_settings"]),
        )
        expected_quiet = np.r_[np.arange(10, 45), np.arange(63, 195)]
        for session_id in self.SESSIONS:
            np.testing.assert_array_equal(
                per_session[session_id]["quiet"], expected_quiet
            )
