"""
@author: bartulem
Unit tests for ``usv_playpen.neural_modeling.neural_cohort``.

Coverage: the session-list union, the brain-area scheme, and the two session sets the cohort hands
out. The property that matters is the asymmetry: a session in which the recorded animal barely vocalized
stays available to the kinematic encoding, which is fitted on silence and is if anything better off, and
is withheld from everything scored on the vocalizations themselves, where it would contribute a fold resting
on a handful of events.
"""

from __future__ import annotations

import json
import pathlib

import polars as pls
import pytest

from usv_playpen.neural_modeling.neural_cohort import (
    _area_pass,
    load_courtship_session_ids,
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
    def test_the_gate_keys_are_present_and_the_threshold_is_the_ruled_one(self):
        """The shipped settings must carry the three gate keys at 20 USVs per session, two vocal sessions and
        two courtship sessions, so the cohort a run uses is the agreed one."""
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
