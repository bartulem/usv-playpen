"""
@author: bartulem
Test the vocal-bout occupancy decoder and the bout-onset statistics.

Coverage: IBI bout grouping and its minimum-USV rule; the labelling, whose single-USV-bin
rule and kept-time floor are where a regression would hide; raw counting and the common-lag
shift; the class-balance guard; the bout-onset descriptors against a hand-built spike train;
and the decoding figure's refusal to draw when its assumptions do not hold.
"""

from __future__ import annotations

import json
import pathlib

import matplotlib
import numpy as np
import polars as pls
import pytest

matplotlib.use("Agg")

from usv_playpen.analyses.compute_neuronal_tuning_curves import (  # noqa: E402
    ibi_bout_extents,
    ibi_bout_onset_statistics,
    ibi_bout_runs,
    ibi_bout_threshold,
    weighted_usv_response,
)
from usv_playpen.analyses.decode_usv_bout_occupancy import (  # noqa: E402
    bout_occupancy_bins,
    count_matrix,
    flatten_spans,
    free_intervals,
    nearest_neighbour,
)
from usv_playpen.visualizations.make_neuronal_decoding_figures import (  # noqa: E402
    NeuronalDecodingFigureMaker,
    blend,
)

SETTINGS_DIR = pathlib.Path(__file__).parents[2] / "src" / "usv_playpen" / "_parameter_settings"


def test_ibi_threshold_matches_the_respiratory_component() -> None:
    """The male's first mixture component at z = 2.58 is the 115.5 ms respiratory threshold."""
    assert ibi_bout_threshold("male", 2.58, 0) == pytest.approx(0.1155, abs=1e-4)
    assert ibi_bout_threshold("female", 2.58, 0) > ibi_bout_threshold("male", 2.58, 0)
    with pytest.raises(KeyError):
        ibi_bout_threshold("unscored", 2.58, 0)


def test_bout_grouping_splits_on_the_threshold_and_honours_the_minimum() -> None:
    """A gap at or above the threshold ends a bout; runs under the minimum are not bouts."""
    starts = np.array([0.0, 0.2, 0.4, 5.0, 5.2])
    stops = starts + 0.1
    runs = ibi_bout_runs(starts, stops, 0.1155, 3)
    assert [list(r) for r in runs] == [[0, 1, 2]]

    low, high = ibi_bout_extents(starts, stops, 0.1155, 3)
    assert low.tolist() == [0.0]
    assert high.tolist() == pytest.approx([0.5])

    # the same USVs with a minimum of two admit the trailing pair as well
    assert len(ibi_bout_runs(starts, stops, 0.1155, 2)) == 2
    # a gap exactly at the threshold splits, since the comparison is `>=`
    split = np.array([0.0, 0.1 + 0.1155, 0.4])
    assert ibi_bout_runs(split, split + 0.1, 0.1155, 3) == []


def test_free_intervals_returns_only_uncovered_time() -> None:
    """The kept sub-intervals of a bin are exactly those no USV covers."""
    starts, stops = np.array([0.10, 0.30]), np.array([0.20, 0.35])
    free = free_intervals(0.0, 0.5, starts, stops)
    assert free == [(0.0, 0.10), (0.20, 0.30), (0.35, 0.5)]
    assert free_intervals(0.12, 0.18, starts, stops) == []


def test_labelling_drops_single_usv_bins_and_respects_the_kept_floor() -> None:
    """A bin holding one isolated USV is neither a positive nor a negative."""
    starts = np.array([0.0, 0.2, 0.4, 20.0])
    stops = starts + 0.1
    spans, labels, mids, positions = bout_occupancy_bins(
        0.25, starts, stops, starts, stops, 40.0, 0.1155, 3, 0.03)

    # the three-USV run is one bout of extent [0.0, 0.5] -> two whole positive bins
    assert int((labels == 1).sum()) == 2
    assert sorted(set(positions[labels == 1].tolist())) == [1, 2]
    # the isolated USV at 20 s belongs to no bout, so its bin is DROPPED, not made a negative
    lattice = int(40.0 / 0.25)
    assert labels.size == lattice - 1
    assert not any(a <= 20.0 < b for gaps in spans for a, b in gaps if len(gaps) == 1
                   and b - a == pytest.approx(0.25))
    assert mids.size == labels.size

    # a floor above the available non-USV time removes every positive
    _spans, strict_labels, _mids, _pos = bout_occupancy_bins(
        0.25, starts, stops, starts, stops, 40.0, 0.1155, 3, 0.20)
    assert int((strict_labels == 1).sum()) == 0


def test_silence_is_judged_against_every_vocalization() -> None:
    """Bouts come from the pure USVs, but a squeak or an overlap segment (or an unassigned
    vocalization) is not silence: a lattice bin holding one is no negative, and inside a bout its
    time is cut from a positive bin's kept time just like a USV's."""
    starts = np.array([0.0, 0.2, 0.4])
    stops = starts + 0.1
    squeak_starts, squeak_stops = np.array([0.16, 10.1]), np.array([0.19, 10.2])
    silence_starts = np.sort(np.concatenate([starts, squeak_starts]))
    silence_stops = np.concatenate([stops, squeak_stops])[np.argsort(np.concatenate([starts, squeak_starts]))]

    usv_only = bout_occupancy_bins(0.25, starts, stops, starts, stops, 40.0, 0.1155, 3, 0.03)
    every = bout_occupancy_bins(0.25, starts, stops, silence_starts, silence_stops, 40.0, 0.1155, 3, 0.03)

    # the squeak at 10.1 s sits in the lattice bin [10.0, 10.25): a negative under pure-USV
    # silence, dropped once every vocalization counts
    def negative_at(result, t):
        spans, labels = result[0], result[1]
        return any(label == 0 and spans[i][0][0] <= t < spans[i][0][1] for i, label in enumerate(labels))
    assert negative_at(usv_only, 10.15)
    assert not negative_at(every, 10.15)
    assert int((every[1] == 0).sum()) == int((usv_only[1] == 0).sum()) - 1

    # inside the bout the squeak's 30 ms is cut from the first positive bin's kept time
    def first_positive_kept(result):
        spans, labels = result[0], result[1]
        index = int(np.flatnonzero(labels == 1)[0])
        return sum(b - a for a, b in spans[index])
    assert first_positive_kept(usv_only) - first_positive_kept(every) == pytest.approx(0.03)


def test_counts_are_raw_and_the_shift_is_common_to_every_unit() -> None:
    """Counting divides by nothing, and one lag moves all units together."""
    spans = [[(0.0, 1.0)], [(2.0, 3.0)]]
    flat = flatten_spans(spans)
    trains = [np.array([0.1, 0.2, 0.3]), np.array([2.5])]
    counts = count_matrix(flat, 2, trains, 10.0)
    assert counts.tolist() == [[3.0, 0.0], [0.0, 1.0]]

    shifted = count_matrix(flat, 2, trains, 10.0, lag_seconds=2.0)
    # Every spike moves by the same 2 s, so unit 0's three land in the second bin. Unit 1's
    # single spike moves to 4.5 s, which no kept interval covers -- the bins deliberately do
    # NOT tile the session, so a shift can carry spikes out of all of them and the total is
    # not conserved. That is the null doing its job, not counting going wrong.
    assert shifted[:, 0].tolist() == [0.0, 3.0]
    assert shifted[:, 1].tolist() == [0.0, 0.0]


def test_nearest_neighbour_refuses_a_fold_with_one_class() -> None:
    """A training fold missing a class cannot be balanced, and says so."""
    train = np.array([[1.0, 2.0], [1.1, 2.1]])
    with pytest.raises(ValueError, match="no training examples of one class"):
        nearest_neighbour(train, np.array([1, 1]), train, np.random.default_rng(0))


def test_weighted_response_is_continuous_and_bounded_by_the_count() -> None:
    """The Gaussian weighting cannot exceed the spikes present, and varies continuously."""
    spikes = np.array([0.01, 0.05, 0.09])
    inside = weighted_usv_response(spikes, 0.0, 0.150, 0.020)
    assert 0.0 < inside <= 3.0
    assert weighted_usv_response(np.empty(0), 0.0, 0.150, 0.020) == 0.0
    # a USV far from every spike draws essentially nothing
    assert weighted_usv_response(spikes, 50.0, 0.150, 0.020) == pytest.approx(0.0, abs=1e-9)


def test_bout_onset_statistics_recover_a_built_in_onset_response() -> None:
    """A train with extra spikes just after each bout onset reads as an elevated onset gain."""
    cfg = json.loads((SETTINGS_DIR / "analyses_settings.json").read_text())[
        "calculate_neuronal_tuning_curves"]["ibi_bout_onset"]
    rng = np.random.default_rng(0)
    duration = 600.0
    onsets = np.arange(20.0, 560.0, 10.0)
    USVs = np.concatenate([onsets + offset for offset in (0.0, 0.2, 0.4)])
    USVs.sort()
    runs = ibi_bout_runs(USVs, USVs + 0.1, 0.1155, 3)
    assert len(runs) == onsets.size

    background = np.sort(rng.uniform(0.0, duration, size=3000))
    burst = np.concatenate([o + rng.uniform(0.0, 0.2, size=25) for o in onsets])
    spikes = np.sort(np.concatenate([background, burst]))

    stats = ibi_bout_onset_statistics(spikes, USVs, runs, duration, cfg)
    assert stats["onset_gain"] > 2.0
    assert -0.1 <= stats["peak_latency"] <= 0.4
    assert 0.0 <= stats["fraction_first_above"] <= 1.0

    flat_stats = ibi_bout_onset_statistics(background, USVs, runs, duration, cfg)
    assert flat_stats["onset_gain"] == pytest.approx(1.0, abs=0.25)
    assert np.isnan(ibi_bout_onset_statistics(spikes, USVs, [], duration, cfg)["onset_gain"])


def test_blend_fades_toward_the_page_without_an_alpha_channel() -> None:
    """Full weight returns the colour, zero weight returns the page."""
    assert blend("#677470", 1.0) == "#677470"
    assert blend("#677470", 0.0) == "#ffffff"
    assert blend("#000000", 0.5) == "#808080"


def test_decoding_figure_refuses_rather_than_drawing(tmp_path: pathlib.Path) -> None:
    """Every assumption the figure cannot verify raises instead of producing a panel."""
    maker = NeuronalDecodingFigureMaker(
        decoding_directory=str(tmp_path),
        visualizations_parameter_dict=json.loads(
            (SETTINGS_DIR / "visualizations_settings.json").read_text()),
        message_output=lambda *_: None,
    )
    focal = (("m1", "20250101_000000", 100),)
    three = (("MRN", "20250101_000000", "20250101_000000", "same session"),) * 3

    with pytest.raises(ValueError, match="exactly three comparison panels"):
        maker.make_bout_decoding_figure(focal_area="PAG", focal_sessions=focal,
                                        comparisons=three[:2], unit_counts=(5,),
                                        out_dir=tmp_path)
    with pytest.raises(ValueError, match="nothing to draw"):
        maker.make_bout_decoding_figure(focal_area="PAG", focal_sessions=(),
                                        comparisons=three, unit_counts=(5,), out_dir=tmp_path)
    with pytest.raises(FileNotFoundError, match="No decode output"):
        maker.make_bout_decoding_figure(focal_area="PAG", focal_sessions=focal,
                                        comparisons=three, unit_counts=(5,), out_dir=tmp_path)
    with pytest.raises(FileNotFoundError, match="No decoder summary"):
        maker.summarize(sessions=("20250101_000000",))


def test_decoding_figure_pools_catalog_acronyms_onto_palette_buckets(
        tmp_path: pathlib.Path) -> None:
    """`CENT2` is a catalog acronym; the palette is keyed by its bucket, `CENT`."""
    maker = NeuronalDecodingFigureMaker(
        decoding_directory=str(tmp_path),
        visualizations_parameter_dict=json.loads(
            (SETTINGS_DIR / "visualizations_settings.json").read_text()),
        message_output=lambda *_: None,
    )
    palette = maker.visualizations_parameter_dict["brain_area_colors"]
    assert maker._area_colour("CENT2") == palette["CENT"]
    assert maker._area_colour("PAG") == palette["PAG"]
    assert maker._area_colour("Su3") == palette["other"]


def test_decoding_figure_draws_and_summarizes_decoded_sessions(tmp_path: pathlib.Path) -> None:
    """With every named session and area decoded, the figure is written and the per-session
    tables are stacked into one frame, one row per area-session-rung."""
    rng = np.random.default_rng(0)
    session_a, session_b = "20250101_000000", "20250102_000000"
    for session, areas in ((session_a, ("PAG", "MRN", "CENT2")), (session_b, ("PAG", "VTA"))):
        rows = []
        for area in areas:
            draws = {str(c): 0.6 + 0.01 * rng.standard_normal(10) for c in (5, 10)}
            nulls = {f"null_{c}": 0.5 + 0.01 * rng.standard_normal(200) for c in (5, 10)}
            np.savez_compressed(tmp_path / f"{session}_{area}.npz", labels=np.array([0, 1]),
                                positions=np.array([0, 1]), ladder=np.array([5, 10]), **draws, **nulls)
            rows += [{"session": session, "area": area, "cells": c, "accuracy": float(draws[str(c)].mean())}
                     for c in (5, 10)]
        pls.DataFrame(rows).write_parquet(tmp_path / f"{session}_bout_decoding.parquet")

    maker = NeuronalDecodingFigureMaker(
        decoding_directory=str(tmp_path),
        visualizations_parameter_dict=json.loads(
            (SETTINGS_DIR / "visualizations_settings.json").read_text()),
        message_output=lambda *_: None,
    )
    destination = maker.make_bout_decoding_figure(
        focal_area="PAG",
        focal_sessions=(("m1", session_a, 100), ("m2", session_b, 300)),
        comparisons=(("MRN", session_a, session_a, "same session"),
                     ("CENT2", session_a, session_a, "same session"),
                     ("VTA", session_b, session_a, "density matched")),
        unit_counts=(5, 10), out_dir=tmp_path / "figures")
    assert pathlib.Path(destination).exists()

    curve = maker._curve(session_a, "PAG", (5, 10, 20))
    assert curve["rungs"] == [5, 10] and all(curve["clears"])

    summary = maker.summarize(sessions=(session_a, session_b))
    assert summary.height == 2 * 3 + 2 * 2
    assert set(summary["session"].to_list()) == {session_a, session_b}
