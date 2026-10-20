"""
@author: bartulem
Unit tests for ``usv_playpen.neural_modeling.neural_cohort``.

Coverage: the session-list union, the brain-area scheme, and the two session sets the cohort hands
out. The property that matters is the asymmetry: a session in which the recorded animal barely vocalized
stays available to the kinematic encoding, which is fitted on silence and is if anything better off, and
is withheld from everything scored on the vocalizations themselves, where it would contribute a fold resting
on a handful of events.

Also covered with hand-countable synthetic data: the session-to-mouse map read off the catalog, the
per-session focal USV counts (analysed-set filtering, emitter resolution by name, and a missing file
counting as zero), every unit-level filter's rejection path, and the end-to-end manifest built from a
settings dict pointing at a synthetic data tree.
"""

from __future__ import annotations

import copy
import json
import pathlib

import h5py
import numpy as np
import polars as pls
import pytest

from usv_playpen.neural_modeling.neural_cohort import (
    _area_pass,
    build_cohort_manifest,
    emitter_usv_counts,
    load_courtship_session_ids,
    recorded_mouse_by_session,
    select_cohort,
    summarize_cohort,
)

SETTINGS_PATH = (pathlib.Path(__file__).resolve().parents[2] / "src" / "usv_playpen"
                 / "_parameter_settings" / "neural_modeling_settings.json")


def _catalog(tmp_path) -> str:
    """
    Description
    -----------
    Writes a four-unit synthetic unit catalog: two PAG somatic good units, one MUA unit and one
    non-somatic good unit, all from mouse ``m1``. The first unit was recorded in a thin session
    (``s_thin``) as well as the two rich ones.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Directory the ``unit_catalog.csv`` is written to.

    Returns
    -------
    catalog_path (str)
        Path to the written CSV.
    """
    path = tmp_path / "unit_catalog.csv"
    pls.DataFrame({
        "mouse_id": ["m1", "m1", "m1", "m1"],
        "rec_date": [20250101, 20250101, 20250101, 20250101],
        "unit_id": ["imec0_cl0001_ch001_good", "imec0_cl0002_ch002_good",
                    "imec0_cl0003_ch003_mua", "imec0_cl0004_ch004_good"],
        "brain_area": ["PAG", "PAG", "PAG", "PAG"],
        "cluster_group": ["good", "good", "mua", "good"],
        "somatic": [True, True, True, False],
        "rec_sessions": ["['s_rich', 's_thin', 's_rich2']", "['s_rich', 's_rich2']",
                         "['s_rich', 's_rich2']", "['s_rich', 's_rich2']"],
    }).write_csv(path)
    return str(path)


COUNTS = {"s_rich": 500, "s_rich2": 300, "s_thin": 3}
MOUSE_BY_SESSION = {"s_rich": "m1", "s_rich2": "m1", "s_thin": "m1"}


def _select(tmp_path, **overrides):
    """
    Description
    -----------
    Runs ``select_cohort`` on the four-unit synthetic catalog with the shipped-like defaults (PAG,
    include, good, somatic, two courtship sessions, 20 emitter USVs per session, two vocal sessions),
    letting a test override any single keyword.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Directory the synthetic catalog is written to.
    **overrides
        Keyword arguments replacing the defaults passed to ``select_cohort``.

    Returns
    -------
    cohort (list[dict])
        The per-unit records ``select_cohort`` returns.
    """
    kwargs = {"catalog_path": _catalog(tmp_path),
              "courtship_session_ids": {"s_rich", "s_rich2", "s_thin"},
              "brain_areas": ["PAG"], "area_mode": "include", "quality": ["good"],
              "somatic": "somatic", "min_courtship_sessions": 2, "emitter_counts": COUNTS,
              "min_emitter_usvs_per_session": 20, "min_vocal_sessions": 2}
    kwargs.update(overrides)
    return select_cohort(**kwargs)


class TestSessionLists:

    """Loading the courtship session ids from the session-list text files."""
    def test_ids_are_unioned_across_files_and_reduced_to_basenames(self, tmp_path):
        """Ids repeated across files and blank lines must collapse to one set of directory basenames, so a
        session listed twice is not counted twice."""
        (tmp_path / "a.txt").write_text("/mnt/x/Data/20250101_000000\n/mnt/x/Data/20250101_010000\n")
        (tmp_path / "b.txt").write_text("/mnt/x/Data/20250101_010000\n\n/mnt/x/Data/20250102_000000\n")
        ids = load_courtship_session_ids([str(tmp_path / "a.txt"), str(tmp_path / "b.txt")])
        assert ids == {"20250101_000000", "20250101_010000", "20250102_000000"}


class TestAreaScheme:

    """The include / exclude / all brain-area filter."""
    def test_include_exclude_and_all(self):
        """``include`` keeps only listed areas, ``exclude`` keeps only unlisted ones and ``all`` keeps every
        area, so each mode selects exactly the population it names."""
        assert _area_pass("PAG", ["PAG"], "include")
        assert not _area_pass("VTA", ["PAG"], "include")
        assert _area_pass("VTA", ["PAG"], "exclude")
        assert not _area_pass("PAG", ["PAG"], "exclude")
        assert _area_pass("anything", [], "all")

    def test_an_unknown_mode_fails_loudly(self):
        """An unrecognised mode must raise ``ValueError`` naming ``area_mode`` rather than silently passing or
        failing every unit."""
        with pytest.raises(ValueError, match="area_mode"):
            _area_pass("PAG", ["PAG"], "sometimes")


class TestTwoSessionSets:

    """The courtship (quiet) and vocal session sets handed out per unit."""
    def test_a_thin_session_is_kept_for_quiet_and_withheld_from_vocal(self, tmp_path):
        """A session with 3 emitter USVs must stay in the unit's courtship set but be dropped from its vocal set,
        with the counts and the vocal-testable flag following from that split."""
        unit = next(u for u in _select(tmp_path) if u["unit_id"].endswith("cl0001_ch001_good"))
        assert unit["courtship_sessions"] == ["s_rich", "s_rich2", "s_thin"]
        assert unit["vocal_sessions"] == ["s_rich", "s_rich2"]
        assert unit["n_courtship_sessions"] == 3
        assert unit["n_vocal_sessions"] == 2
        assert unit["vocal_testable"]

    def test_the_per_session_counts_are_recorded_not_just_applied(self, tmp_path):
        """A gate whose effect is not stated is a silent change to every reported denominator."""
        unit = next(u for u in _select(tmp_path) if u["unit_id"].endswith("cl0001_ch001_good"))
        assert unit["emitter_usvs_per_session"] == {"s_rich": 500, "s_rich2": 300, "s_thin": 3}

    def test_a_unit_short_of_vocal_sessions_is_flagged_not_dropped(self, tmp_path):
        """It still has its kinematic encoding to be tested, and dropping it here would quietly shrink the
        denominator the transformation fraction is quoted against."""
        cohort = _select(tmp_path, min_vocal_sessions=3)
        assert len(cohort) == 2                        # both good+somatic units survive
        assert not any(u["vocal_testable"] for u in cohort)

    def test_the_threshold_moves_the_vocal_set_and_nothing_else(self, tmp_path):
        """Lowering the per-session USV threshold to 2 must restore the thin session to the vocal set while the
        courtship set is unchanged, so the threshold acts on the vocal side only."""
        loose = next(u for u in _select(tmp_path, min_emitter_usvs_per_session=2)
                     if u["unit_id"].endswith("cl0001_ch001_good"))
        assert loose["vocal_sessions"] == ["s_rich", "s_rich2", "s_thin"]
        assert loose["courtship_sessions"] == ["s_rich", "s_rich2", "s_thin"]

    def test_quality_and_somatic_filters_still_apply(self, tmp_path):
        """The MUA unit and the non-somatic unit must be excluded, so the session-set logic did not bypass the
        quality and somatic filters."""
        kept = {u["unit_id"] for u in _select(tmp_path)}
        assert kept == {"imec0_cl0001_ch001_good", "imec0_cl0002_ch002_good"}

    def test_min_sessions_counts_courtship_sessions_not_vocal_ones(self, tmp_path):
        """The floor that decides whether a unit exists at all is about the QUIET side, so a unit
        with two courtship sessions of which one is thin is still a unit."""
        cohort = _select(tmp_path, emitter_counts={"s_rich": 500, "s_rich2": 3, "s_thin": 3})
        unit = next(u for u in cohort if u["unit_id"].endswith("cl0002_ch002_good"))
        assert unit["n_courtship_sessions"] == 2
        assert unit["vocal_sessions"] == ["s_rich"]
        assert not unit["vocal_testable"]


class TestSummary:

    """The cohort summary that reports what the sufficiency gate removed."""
    def test_the_summary_reports_what_the_gate_removed(self, tmp_path):
        """The summary must count the units, the courtship and vocal session slots, and the total versus
        retained focal USVs, so the gate's effect on every denominator is stated."""
        summary = summarize_cohort(_select(tmp_path))
        assert summary["n_units"] == 2
        assert summary["session_slots"] == 5              # 3 + 2
        assert summary["vocal_session_slots"] == 4        # the thin session is withheld once
        assert summary["focal_usvs_total"] == 500 + 300 + 3 + 500 + 300
        assert summary["focal_usvs_retained"] == 500 + 300 + 500 + 300
        assert summary["n_units_losing_a_session"] == 1


class TestShippedSettings:

    """The data-sufficiency and kinematic-encoding keys of the shipped settings file."""
    def test_the_gate_keys_are_present_and_the_threshold_is_twenty(self):
        """The shipped settings must carry the three gate keys at 20 USVs per session, two vocal sessions and
        two courtship sessions, so the cohort a run uses is the intended one."""
        with SETTINGS_PATH.open() as handle:
            sufficiency = json.load(handle)["data_sufficiency"]
        assert sufficiency["min_emitter_usvs_per_session"] == 20
        assert sufficiency["min_vocal_sessions"] == 2
        assert sufficiency["min_courtship_sessions"] == 2

    def test_the_block_split_gap_is_derived_not_configured(self):
        """A gap shorter than the predictor history leaks and is silently overridden; a longer one
        only discards anchors. A key here could therefore only mislead, so there must not be one."""
        with SETTINGS_PATH.open() as handle:
            settings = json.load(handle)
        assert "block_cv_gap_seconds" not in settings["data_sufficiency"]
        assert settings["data_sufficiency"]["single_session_inner_split_blocks"] == 5

    def test_the_filter_band_knob_is_absent_until_the_band_is_built(self):
        """The reported filter band is a RESAMPLING of the representative model's quiet anchors with
        its feature set frozen, not a fold count -- the rotation yields one filter per session, never
        ten. The knob returns, named for resampling, with the figure layer that computes it."""
        with SETTINGS_PATH.open() as handle:
            settings = json.load(handle)
        assert "filter_band_n_folds" not in settings["kinematic_encoding"]


class TestUnitFilters:

    """Each unit-level filter's rejection path, checked against the four-unit catalog by hand."""
    def test_an_unknown_somatic_mode_fails_loudly(self, tmp_path):
        """
        Description
        -----------
        A somatic mode outside ``'somatic' | 'non_somatic' | 'both'`` must raise ``ValueError`` naming the
        argument before any unit is read, rather than silently passing or failing every unit.

        Parameters
        ----------
        tmp_path (pathlib.Path)
            Directory the synthetic catalog is written to.

        Returns
        -------
        None
        """
        with pytest.raises(ValueError, match="somatic must be"):
            _select(tmp_path, somatic="mostly")

    def test_non_somatic_keeps_only_the_non_somatic_good_unit(self, tmp_path):
        """
        Description
        -----------
        Of the four units, three are somatic and one (``cl0004``, good) is not, so ``'non_somatic'`` must keep
        exactly ``cl0004`` and ``'both'`` must keep both somatic good units plus ``cl0004`` (the MUA unit
        still fails quality).

        Parameters
        ----------
        tmp_path (pathlib.Path)
            Directory the synthetic catalog is written to.

        Returns
        -------
        None
        """
        assert [u["unit_id"] for u in _select(tmp_path, somatic="non_somatic")] == ["imec0_cl0004_ch004_good"]
        both = {u["unit_id"] for u in _select(tmp_path, somatic="both")}
        assert both == {"imec0_cl0001_ch001_good", "imec0_cl0002_ch002_good", "imec0_cl0004_ch004_good"}

    def test_a_failing_area_drops_every_unit(self, tmp_path):
        """
        Description
        -----------
        Every unit is in PAG, so ``'exclude'`` on PAG must drop all of them and ``'include'`` on another
        area must too, while ``'all'`` keeps the same two good somatic units as the default.

        Parameters
        ----------
        tmp_path (pathlib.Path)
            Directory the synthetic catalog is written to.

        Returns
        -------
        None
        """
        assert _select(tmp_path, area_mode="exclude") == []
        assert _select(tmp_path, brain_areas=["VTA"]) == []
        assert len(_select(tmp_path, area_mode="all", brain_areas=[])) == 2

    def test_a_unit_short_of_courtship_sessions_is_dropped(self, tmp_path):
        """
        Description
        -----------
        Only ``cl0001`` was recorded in three courtship sessions (the others in two), so a floor of three must
        keep exactly ``cl0001``; and with the courtship set narrowed to ``s_rich`` alone, a floor of two must
        keep no unit at all, since sessions outside the courtship set do not count.

        Parameters
        ----------
        tmp_path (pathlib.Path)
            Directory the synthetic catalog is written to.

        Returns
        -------
        None
        """
        assert [u["unit_id"] for u in _select(tmp_path, min_courtship_sessions=3)] == ["imec0_cl0001_ch001_good"]
        assert _select(tmp_path, courtship_session_ids={"s_rich"}) == []


class TestRecordedMouseBySession:

    """The session-to-mouse map is read off the catalog."""
    def test_every_listed_session_maps_to_its_mouse(self, tmp_path):
        """
        Description
        -----------
        Two mice: ``m1`` units list ``s1``/``s2`` (one unit lists ``s2`` again), an ``m2`` unit lists ``s3``,
        and a fourth unit has an EMPTY ``rec_sessions`` cell (read as null). The map must be exactly
        ``{s1: m1, s2: m1, s3: m2}``, the empty cell contributing nothing; a numeric mouse id (``7``) must
        come back as the string ``"7"``.

        Parameters
        ----------
        tmp_path (pathlib.Path)
            Directory the catalog is written to.

        Returns
        -------
        None
        """
        path = tmp_path / "unit_catalog.csv"
        pls.DataFrame({
            "mouse_id": ["m1", "m1", "m2", "m2", "7"],
            "rec_sessions": ["['s1', 's2']", "['s2']", "['s3']", None, "['s4']"],
        }).write_csv(path)
        assert recorded_mouse_by_session(str(path)) == {"s1": "m1", "s2": "m1", "s3": "m2", "s4": "7"}


SETTINGS_VOCAL = {"exclude_noise": True, "exclude_squeaks": True, "squeaks_block_quiet": True}

# Synthetic session ids start with a digit like the real ones: the tracking-file glob is digit-led
# (``[0-9]*``), so a basename starting with a letter is never found.
SESSION_A = "20250101_100000"
SESSION_B = "20250102_100000"
SESSION_EMPTY = "20250103_100000"
SESSION_MISSING = "20250104_100000"
SESSION_NEVER = "20250105_100000"
SESSION_ORPHAN = "20250106_100000"


def _write_session(root: pathlib.Path, session_id: str, track_names: list, usvs: list | None) -> None:
    """
    Description
    -----------
    Write a synthetic session: the tracking-H5 header (``track_names``, a frame rate and a tiny ``tracks``
    array) and, unless ``usvs`` is None, a USV summary with ``start``/``stop``/``emitter``/``noise``/``squeak``
    columns.

    Parameters
    ----------
    root (pathlib.Path)
        The data root.
    session_id (str)
        Session directory basename.
    track_names (list[str])
        The session's ``track_names``.
    usvs (list[tuple] | None)
        ``(emitter, noise, squeak)`` per USV row, or None to write no summary at all.

    Returns
    -------
    None
        The files are written to disk.
    """
    video_dir = root / session_id / "video" / f"{session_id}_cam"
    video_dir.mkdir(parents=True)
    with h5py.File(video_dir / f"{session_id}_points3d_translated_rotated_metric.h5", "w") as handle:
        handle["tracks"] = np.zeros((10, len(track_names), 3, 3), dtype=np.float64)
        handle["recording_frame_rate"] = 150.0
        handle.create_dataset("track_names", data=np.array([name.encode() for name in track_names], dtype="S8"))
    if usvs is None:
        return
    audio_dir = root / session_id / "audio"
    audio_dir.mkdir(parents=True)
    pls.DataFrame({"start": [float(i) for i in range(len(usvs))],
                   "stop": [float(i) + 0.05 for i in range(len(usvs))],
                   "emitter": pls.Series([row[0] for row in usvs], dtype=pls.String),
                   "noise": pls.Series([row[1] for row in usvs], dtype=pls.Boolean),
                   "squeak": pls.Series([row[2] for row in usvs], dtype=pls.Boolean),
                   }).write_csv(audio_dir / f"{session_id}_usv_summary.csv")


def _write_tree(root: pathlib.Path) -> None:
    """
    Description
    -----------
    Write four synthetic sessions with hand-countable USVs, the recorded animal always ``m1``:

    - ``SESSION_A``: 3 clean ``m1`` rows, 1 ``m1`` noise row, 1 ``m1`` squeak row, 2 clean ``f1`` rows
      (self 3, partner 2, all 5 after the analysed-set filter);
    - ``SESSION_B``: 1 clean ``m1`` row and 1 clean ``f1`` row (self 1, partner 1, all 2), with the
      RECORDED animal in track slot 1, so a slot assumption would swap the self/partner counts;
    - ``SESSION_EMPTY``: a summary with no rows (0 for every emitter);
    - ``SESSION_MISSING``: tracking but no summary (counted as 0, not raised).

    Parameters
    ----------
    root (pathlib.Path)
        The data root.

    Returns
    -------
    None
        The files are written to disk.
    """
    _write_session(root, SESSION_A, ["m1", "f1"],
                   [("m1", False, False), ("m1", False, False), ("m1", False, False), ("m1", True, False),
                    ("m1", False, True), ("f1", False, False), ("f1", False, False)])
    _write_session(root, SESSION_B, ["f1", "m1"], [("m1", False, False), ("f1", False, False)])
    _write_session(root, SESSION_EMPTY, ["m1", "f1"], [])
    _write_session(root, SESSION_MISSING, ["m1", "f1"], None)


class TestEmitterUsvCounts:

    """Per-session focal counts match a hand count on a tiny synthetic tree."""
    @pytest.mark.parametrize(("vocal_emitter", "expected"), [
        ("self", {SESSION_A: 3, SESSION_B: 1, SESSION_EMPTY: 0, SESSION_MISSING: 0}),
        ("partner", {SESSION_A: 2, SESSION_B: 1, SESSION_EMPTY: 0, SESSION_MISSING: 0}),
        ("all", {SESSION_A: 5, SESSION_B: 2, SESSION_EMPTY: 0, SESSION_MISSING: 0}),
    ])
    def test_counts_match_the_hand_count(self, tmp_path, vocal_emitter, expected):
        """
        Description
        -----------
        The counts must be the analysed-set rows (noise and squeak dropped) from the emitter the spec names,
        with the recorded animal resolved by NAME from the session-to-mouse map (``SESSION_B`` stores it in
        track slot 1), an empty summary counting 0 and a session without a summary counting 0 instead of
        raising. The returned keys must be every requested session, whatever order they were passed in.

        Parameters
        ----------
        tmp_path (pathlib.Path)
            The synthetic data root.
        vocal_emitter (str)
            ``'self'`` | ``'partner'`` | ``'all'``.
        expected (dict)
            The hand count per session.

        Returns
        -------
        None
        """
        _write_tree(tmp_path)
        mouse = dict.fromkeys((SESSION_A, SESSION_B, SESSION_EMPTY, SESSION_MISSING), "m1")
        counts = emitter_usv_counts({SESSION_B, SESSION_MISSING, SESSION_A, SESSION_EMPTY}, str(tmp_path),
                                    mouse, vocal_emitter, SETTINGS_VOCAL)
        assert counts == expected

    def test_a_session_without_tracking_counts_zero(self, tmp_path):
        """
        Description
        -----------
        A session listed but never processed (no directory at all) must count zero rather than abort the
        manifest, and must not need an entry in the session-to-mouse map (the lookup happens only after the
        files are found).

        Parameters
        ----------
        tmp_path (pathlib.Path)
            The (empty) data root.

        Returns
        -------
        None
        """
        assert emitter_usv_counts([SESSION_NEVER], str(tmp_path), {}, "self", SETTINGS_VOCAL) == {SESSION_NEVER: 0}


class TestBuildCohortManifest:

    """The end-to-end manifest from a settings dict pointing at a synthetic tree."""
    def test_the_manifest_holds_exactly_the_expected_units_and_sessions(self, tmp_path):
        """
        Description
        -----------
        The two session lists name ``SESSION_A`` (first list) and ``SESSION_B`` (second list, twice, with a
        blank line) plus ``SESSION_ORPHAN``, which no catalog unit lists and which therefore has no recorded
        mouse and must not be counted. The catalog holds ``u1`` (good, somatic, PAG; A and B), ``u2`` (good,
        somatic, PAG; A only -- short of two courtship sessions), ``u3`` (MUA; A and B) and ``u4`` (VTA; A
        and B). With the vocal floor overridden in a deep copy to 2 USVs per session, the manifest must be
        exactly ``u1``, with both courtship sessions, the hand counts ``{A: 3, B: 1}`` (B's recorded male sits
        in track slot 1), only A vocal-eligible, and therefore not vocal-testable at two vocal sessions.
        The shipped settings dict itself must be left untouched.

        Parameters
        ----------
        tmp_path (pathlib.Path)
            Root for the synthetic data tree, session lists and catalog.

        Returns
        -------
        None
        """
        data_root = tmp_path / "Data"
        _write_tree(data_root)
        (tmp_path / "intact.txt").write_text(f"{data_root / SESSION_A}\n{data_root / SESSION_ORPHAN}\n")
        (tmp_path / "mute.txt").write_text(f"{data_root / SESSION_B}\n\n{data_root / SESSION_B}\n")
        catalog = tmp_path / "unit_catalog.csv"
        both = f"['{SESSION_A}', '{SESSION_B}']"
        pls.DataFrame({
            "mouse_id": ["m1", "m1", "m1", "m1"],
            "rec_date": [20250101] * 4,
            "unit_id": ["u1", "u2", "u3", "u4"],
            "brain_area": ["PAG", "PAG", "PAG", "VTA"],
            "cluster_group": ["good", "good", "mua", "good"],
            "somatic": [True, True, True, True],
            "rec_sessions": [both, f"['{SESSION_A}']", both, both],
        }).write_csv(catalog)
        with SETTINGS_PATH.open() as handle:
            shipped = json.load(handle)
        settings = copy.deepcopy(shipped)
        settings["data"]["session_list_files"] = [str(tmp_path / "intact.txt"), str(tmp_path / "mute.txt")]
        settings["data_roots"]["catalog_path"] = str(catalog)
        settings["data_roots"]["data_root"] = str(data_root)
        settings["data_sufficiency"]["min_emitter_usvs_per_session"] = 2
        settings["vocalization_settings"]["vocal_emitter"] = "self"

        manifest = build_cohort_manifest(settings)
        assert [unit["unit_uid"] for unit in manifest] == ["m1_20250101_u1"]
        unit = manifest[0]
        assert unit["courtship_sessions"] == [SESSION_A, SESSION_B]
        assert unit["emitter_usvs_per_session"] == {SESSION_A: 3, SESSION_B: 1}
        assert unit["vocal_sessions"] == [SESSION_A]
        assert unit["vocal_testable"] is False
        assert unit["rec_date"] == 20250101
        assert shipped["data_sufficiency"]["min_emitter_usvs_per_session"] == 20
