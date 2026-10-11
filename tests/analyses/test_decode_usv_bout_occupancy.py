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

from usv_playpen.analyses import decode_usv_bout_occupancy as bout_decoder
from usv_playpen.analyses.compute_neuronal_tuning_curves import (  # noqa: E402
    ibi_bout_extents,
    ibi_bout_onset_block,
    ibi_bout_onset_statistics,
    ibi_bout_runs,
    ibi_bout_threshold,
    weighted_usv_response,
)
from usv_playpen.analyses.decode_usv_bout_occupancy import (  # noqa: E402
    BoutOccupancyDecoder,
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


SESSION_ID = "20250101_000000"


def _synthetic_bundle(*, extra_rows: tuple = (), emitter_of_bouts: str = "m1") -> dict:
    """
    Description
    -----------
    Build a synthetic stand-in for the `load_vocal_inputs` bundle of a 300 s session.

    One animal (`emitter_of_bouts`) produces 28 bouts, one every 10 s from 10 s to 280 s, each of
    four 100 ms pure USVs 200 ms apart, so every bout spans 0.7 s and yields two whole 0.25 s
    positive bins. A single unassigned USV (emitter None) sits at 295 s, which must be treated as
    a vocalization silence has to avoid. Additional rows are appended verbatim.

    Parameters
    ----------
    extra_rows (tuple)
        Extra `(start, stop, emitter, is_usv)` rows appended after the bouts, e.g. a squeak
        (`is_usv` False) placed in an otherwise silent lattice bin.
    emitter_of_bouts (str)
        Track name credited with the bout USVs.

    Returns
    -------
    bundle (dict)
        A dict with the documented `load_vocal_inputs` keys: `usv_df`, `track_names`,
        `animal_sex`, `duration_seconds`, `starts`, `stops`, `emitters` and `is_usv`.
    """

    rows = [(onset + offset, onset + offset + 0.1, emitter_of_bouts, True)
            for onset in np.arange(10.0, 281.0, 10.0) for offset in (0.0, 0.2, 0.4, 0.6)]
    rows.append((295.0, 295.1, None, True))
    rows.extend(extra_rows)
    rows.sort(key=lambda row: row[0])
    starts = np.array([row[0] for row in rows], dtype=float)
    stops = np.array([row[1] for row in rows], dtype=float)
    emitters = np.array([row[2] for row in rows], dtype=object)
    is_usv = np.array([row[3] for row in rows], dtype=bool)
    usv_df = pls.DataFrame({"start": starts, "stop": stops,
                            "emitter": [row[2] for row in rows]})
    return {"usv_df": usv_df, "track_names": ["m1", "f1"],
            "animal_sex": {"m1": "male", "f1": "female"}, "duration_seconds": 300.0,
            "starts": starts, "stops": stops, "emitters": emitters, "is_usv": is_usv}


def _write_decoder_session(tmp_path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    """
    Description
    -----------
    Lay down a session root with spike files and a unit catalog for the session driver.

    The catalog names six somatic good PAG units and two somatic good MRN units in the session,
    plus three rows the quality filters must drop: a non-somatic PAG unit, a `mua` PAG unit and
    a PAG unit recorded in another session. Every in-session unit gets a spike file at
    `ephys/imec0/cluster_data/<unit_id>.npy` whose first row holds spike times (the second row is
    a dummy row, as in the real cluster files); PAG units fire more inside bouts so the observed
    decode has something to read.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Pytest-provided temporary directory.

    Returns
    -------
    root, catalog_path (tuple[pathlib.Path, pathlib.Path])
        The session root directory and the written catalog CSV.
    """

    rng = np.random.default_rng(1)
    root = tmp_path / SESSION_ID
    cluster_dir = root / "ephys" / "imec0" / "cluster_data"
    cluster_dir.mkdir(parents=True)
    bout_onsets = np.arange(10.0, 281.0, 10.0)
    records = []
    for index in range(6):
        unit_id = f"imec0_cl{index:04d}"
        records.append((f"['{SESSION_ID}']", "good", "PAG", True, "m1", unit_id))
    for index in range(6, 8):
        records.append((f"['{SESSION_ID}']", "good", "MRN", True, "m1", f"imec0_cl{index:04d}"))
    records.append((f"['{SESSION_ID}']", "good", "PAG", False, "m1", "imec0_cl0008"))
    records.append((f"['{SESSION_ID}']", "mua", "PAG", True, "m1", "imec0_cl0009"))
    records.append(("['20250102_000000']", "good", "PAG", True, "m1", "imec0_cl0010"))
    for _, _, area, _, _, unit_id in records[:10]:
        background = rng.uniform(0.0, 300.0, size=900)
        burst = (np.concatenate([o + rng.uniform(0.0, 0.7, size=6) for o in bout_onsets])
                 if area == "PAG" else np.empty(0))
        spikes = np.sort(np.concatenate([background, burst]))
        np.save(cluster_dir / f"{unit_id}.npy", np.vstack([spikes, np.zeros_like(spikes)]))
    catalog = pls.DataFrame(records, schema=["rec_sessions", "cluster_group", "brain_area",
                                             "somatic", "mouse_id", "unit_id"], orient="row")
    catalog_path = tmp_path / "unit_catalog.csv"
    catalog.write_csv(catalog_path)
    return root, catalog_path


def _decoder_parameters(catalog_path: pathlib.Path, out_dir: pathlib.Path, **overrides) -> dict:
    """
    Description
    -----------
    A small `decoding_parameters_dict`: the shipped bout definition (male respiratory threshold,
    three-USV minimum, 0.25 s bins, 30 ms kept floor), a short `[2, 4]` unit ladder, two runs per
    rung, three shuffles, three folds and a four-unit area floor, so the whole drive takes a few
    seconds. Any key can be overridden.

    Parameters
    ----------
    catalog_path (pathlib.Path)
        The unit catalog CSV.
    out_dir (pathlib.Path)
        Where the per-area archives and the session table are written.
    **overrides
        Keys replacing the defaults.

    Returns
    -------
    params (dict)
        The decoding-parameter dict consumed by `BoutOccupancyDecoder`.
    """

    params = {"emitter": "self", "ibi_z_score": 2.58, "ibi_component_index": 0,
              "min_vocalizations": 3, "bin_seconds": 0.25, "min_kept_seconds": 0.03,
              "brain_areas": None, "min_units_per_area": 4, "cluster_group": ["good"],
              "somatic_bool": True, "unit_ladder": [2, 4], "runs_per_count": 2, "n_folds": 3,
              "n_shuffles": 3, "shuffle_guard_seconds": 20.0, "shuffle_seed": 0,
              "catalog_path": str(catalog_path), "output_directory": str(out_dir)}
    params.update(overrides)
    return params


def _run_decoder(monkeypatch: pytest.MonkeyPatch, root: pathlib.Path, params: dict,
                 bundle: dict | None) -> list:
    """
    Description
    -----------
    Drive `BoutOccupancyDecoder.decode_usv_bout_occupancy` on one session with the loader
    replaced by a fixed bundle, the interactive wait disabled and experimenter path resolution
    made the identity, collecting every message the driver emits.

    Parameters
    ----------
    monkeypatch (pytest.MonkeyPatch)
        Used to patch the module-level loader, `smart_wait` and `resolve_experimenter_path`.
    root (pathlib.Path)
        Session root handed to the decoder.
    params (dict)
        The decoding-parameter dict.
    bundle (dict | None)
        What the patched `load_vocal_inputs` returns; None exercises the missing-input skip.

    Returns
    -------
    messages (list)
        Every string passed to `message_output`, in order.
    """

    monkeypatch.setattr(bout_decoder, "load_vocal_inputs", lambda _root: bundle)
    monkeypatch.setattr(bout_decoder, "smart_wait", lambda **_kwargs: None)
    monkeypatch.setattr(bout_decoder, "resolve_experimenter_path", lambda p: p)
    messages: list = []
    BoutOccupancyDecoder(root_directory=str(root), decoding_parameters_dict=params,
                         message_output=messages.append).decode_usv_bout_occupancy()
    return messages


def test_decoder_rejects_unexpected_keyword_arguments() -> None:
    """
    Description
    -----------
    The constructor accepts only `root_directory`, `decoding_parameters_dict` and
    `message_output`; a misspelled or stray keyword raises `TypeError` naming it instead of being
    silently stored and never read.
    """

    with pytest.raises(TypeError, match="'n_shuffle'"):
        BoutOccupancyDecoder(root_directory="/tmp", n_shuffle=5)


def test_emitter_names_resolve_self_partner_and_all() -> None:
    """
    Description
    -----------
    `self` is the recorded animal, `partner` every other track and `all` an empty list meaning no
    filter at all (so unassigned vocalizations are kept). A recorded animal absent from the
    tracks, a session with no partner track and an unknown emitter word each raise `ValueError`
    rather than decoding the wrong animal's bouts.
    """

    bundle = {"track_names": ["m1", "f1"]}

    def names(emitter: str, focal: str, tracks: dict = bundle) -> list:
        """Resolve `emitter` for `focal` through a decoder carrying only that setting."""
        decoder = BoutOccupancyDecoder(decoding_parameters_dict={"emitter": emitter},
                                       message_output=lambda *_: None)
        return decoder._emitter_names(tracks, focal)

    assert names("self", "m1") == ["m1"]
    assert names("partner", "m1") == ["f1"]
    assert names("all", "m1") == []
    with pytest.raises(ValueError, match="cannot resolve emitter 'self'"):
        names("self", "m9")
    with pytest.raises(ValueError, match="No partner track"):
        names("partner", "m1", {"track_names": ["m1"]})
    with pytest.raises(ValueError, match="emitter must be 'self', 'partner' or 'all'"):
        names("everyone", "m1")


def test_session_driver_decodes_each_area_over_the_floor(tmp_path: pathlib.Path,
                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Description
    -----------
    End to end on a synthetic session: PAG (six units after the somatic, cluster-group and
    session filters drop three rows) clears the four-unit floor and is decoded; MRN (two units) is
    reported as held back and writes nothing. The PAG archive carries the labels, the bout
    positions, the ladder `[2, 4, 6]` (the area's full roster is appended as the last rung) and an
    observed and a null array per rung, sized by `runs_per_count` and `n_shuffles`. The session
    table holds one row per area-rung with consistent bin counts, accuracies in [0, 1] and a
    shift-null p in (0, 1].
    """

    root, catalog_path = _write_decoder_session(tmp_path)
    out_dir = tmp_path / "decoding"
    messages = _run_decoder(monkeypatch, root, _decoder_parameters(catalog_path, out_dir),
                            _synthetic_bundle())

    assert any("below the 4-unit floor, not decoded: MRN (2)" in m for m in messages)
    assert any("56 in-bout bins" in m for m in messages)
    assert messages[-1].startswith("Decoding USV bout occupancy finished at")
    assert not (out_dir / f"{SESSION_ID}_MRN.npz").exists()

    archive = np.load(out_dir / f"{SESSION_ID}_PAG.npz")
    assert archive["ladder"].tolist() == [2, 4, 6]
    assert {"labels", "positions", "2", "4", "6", "null_2", "null_4", "null_6"} <= set(
        archive.files)
    for count in (2, 4, 6):
        assert archive[str(count)].shape == (2,)
        assert archive[f"null_{count}"].shape == (3,)
    labels, positions = archive["labels"], archive["positions"]
    assert int((labels == 1).sum()) == 56
    assert sorted(set(positions[labels == 1].tolist())) == [1, 2]
    assert not positions[labels == 0].any()

    table = pls.read_parquet(out_dir / f"{SESSION_ID}_bout_decoding.parquet")
    assert table.height == 3
    assert table["area"].to_list() == ["PAG"] * 3
    assert table["cells"].to_list() == [2, 4, 6]
    assert set(table["n_units"].to_list()) == {6}
    assert set(table["n_positive"].to_list()) == {56}
    assert set(table["n_negative"].to_list()) == {int((labels == 0).sum())}
    assert set(table["mouse"].to_list()) == {"m1"}
    assert table["ibi_threshold_seconds"][0] == pytest.approx(ibi_bout_threshold("male", 2.58, 0))
    assert table["accuracy"].is_between(0.0, 1.0).all()
    assert ((table["p"] > 0.0) & (table["p"] <= 1.0)).all()
    assert (table["null_p995"] >= table["null"] - 1e-12).all()


def test_session_driver_judges_silence_against_every_vocalization(
        tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Description
    -----------
    The silence rule end to end: a pure squeak by the emitter (`usv` False, `squeak` True, so
    `is_usv` is False) placed in an otherwise silent lattice bin joins no bout but costs that bin
    its negative label, so the session table reports exactly one negative fewer than the same
    session without it, with the positives unchanged. A squeak by the partner, whom `self` does
    not listen to, removes nothing.
    """

    root, catalog_path = _write_decoder_session(tmp_path)
    counts = {}
    for name, extra in (("plain", ()), ("own_squeak", ((155.05, 155.10, "m1", False),)),
                        ("partner_squeak", ((155.05, 155.10, "f1", False),))):
        out_dir = tmp_path / name
        _run_decoder(monkeypatch, root, _decoder_parameters(catalog_path, out_dir,
                                                            unit_ladder=[]),
                     _synthetic_bundle(extra_rows=extra))
        table = pls.read_parquet(out_dir / f"{SESSION_ID}_bout_decoding.parquet")
        counts[name] = (table["n_positive"][0], table["n_negative"][0])

    assert counts["own_squeak"][0] == counts["plain"][0]
    assert counts["own_squeak"][1] == counts["plain"][1] - 1
    assert counts["partner_squeak"] == counts["plain"]


def test_session_driver_honours_requested_brain_areas(tmp_path: pathlib.Path,
                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Description
    -----------
    With `brain_areas` set to PAG and VTA, the area the session never recorded (VTA) is named as
    absent, MRN is filtered out before the floor so it is neither decoded nor reported as held
    back, and only PAG is written. The run uses emitter `all`, whose emitter set is mixed, so the
    bout threshold falls back to the recorded animal's sex.
    """

    root, catalog_path = _write_decoder_session(tmp_path)
    out_dir = tmp_path / "decoding"
    messages = _run_decoder(monkeypatch, root,
                            _decoder_parameters(catalog_path, out_dir, emitter="all",
                                                brain_areas=["PAG", "VTA"], unit_ladder=[]),
                            _synthetic_bundle())

    assert any("requested areas absent from this session: VTA" in m for m in messages)
    assert not any("floor, not decoded" in m for m in messages)
    table = pls.read_parquet(out_dir / f"{SESSION_ID}_bout_decoding.parquet")
    assert table["area"].to_list() == ["PAG"]
    assert table["cells"].to_list() == [6]
    assert table["ibi_threshold_seconds"][0] == pytest.approx(ibi_bout_threshold("male", 2.58, 0))
    assert not (out_dir / f"{SESSION_ID}_MRN.npz").exists()


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("no_bundle", "missing *_usv_summary.csv or tracking .h5"),
        ("no_catalog_units", "no catalog units for"),
        ("empty_class", "one of the two classes is empty"),
        ("no_area_over_floor", "no area clears the unit floor"),
    ],
)
def test_session_driver_skips_with_a_reason(tmp_path: pathlib.Path,
                                            monkeypatch: pytest.MonkeyPatch, case: str,
                                            expected: str) -> None:
    """
    Description
    -----------
    Every way a session can contribute nothing ends with a message saying why and writes no
    output: the loader finds no USV summary or tracking file (bundle None), the catalog holds no
    unit of this session under the quality filters (`cluster_group` set to a group no unit is
    in), the emitter has no bouts so the in-bout class is empty (every bout credited to the
    partner while decoding `self`), and no area reaches `min_units_per_area`.
    """

    root, catalog_path = _write_decoder_session(tmp_path)
    out_dir = tmp_path / "decoding"
    params = _decoder_parameters(catalog_path, out_dir)
    bundle = _synthetic_bundle()
    if case == "no_bundle":
        bundle = None
    elif case == "no_catalog_units":
        params["cluster_group"] = ["noise"]
    elif case == "empty_class":
        bundle = _synthetic_bundle(emitter_of_bouts="f1")
    else:
        params["min_units_per_area"] = 100

    messages = _run_decoder(monkeypatch, root, params, bundle)
    assert expected in messages[-1]
    assert not out_dir.exists() or not any(out_dir.iterdir())


def test_bout_helpers_handle_no_usvs_and_no_qualifying_bouts() -> None:
    """
    Description
    -----------
    With no USVs at all `ibi_bout_runs` returns no bouts, and when USVs exist but no run reaches
    `min_vocalizations` the extents come back as two empty float arrays rather than failing on an
    empty stack.
    """

    assert ibi_bout_runs(np.empty(0), np.empty(0), 0.1155, 3) == []
    starts = np.array([0.0, 5.0, 10.0])
    low, high = ibi_bout_extents(starts, starts + 0.1, 0.1155, 3)
    assert low.size == 0
    assert high.size == 0
    assert low.dtype == float
    assert high.dtype == float


def test_bout_onset_statistics_without_later_usvs_leave_the_ratio_undefined() -> None:
    """
    Description
    -----------
    Single-USV bouts (allowed when `min_vocalizations` is 1) have no later USVs, so they are
    skipped by the first-versus-later comparison: the ratio and the fraction of bouts whose first
    USV drew more are NaN, while the rate descriptors are still defined.
    """

    cfg = json.loads((SETTINGS_DIR / "analyses_settings.json").read_text())[
        "calculate_neuronal_tuning_curves"]["ibi_bout_onset"]
    usv_starts = np.arange(10.0, 100.0, 10.0)
    runs = ibi_bout_runs(usv_starts, usv_starts + 0.1, 0.1155, 1)
    assert all(run.size == 1 for run in runs)
    spikes = np.sort(np.random.default_rng(2).uniform(0.0, 120.0, size=400))
    stats = ibi_bout_onset_statistics(spikes, usv_starts, runs, 120.0, cfg)
    assert np.isnan(stats["first_over_later"])
    assert np.isnan(stats["fraction_first_above"])
    assert np.isfinite(stats["onset_gain"])


def test_bout_onset_block_of_a_silent_unit_is_undefined_everywhere() -> None:
    """
    Description
    -----------
    A unit with no spikes has no defined descriptor in the observed data or in any shift draw,
    so every null aggregate is filled with NaN directly (rather than handing an all-NaN array to
    the percentile helper, which warns) and every two-sided p is NaN; the bookkeeping fields still
    record the bout count, the draw count and the threshold used.
    """

    cfg = json.loads((SETTINGS_DIR / "analyses_settings.json").read_text())[
        "calculate_neuronal_tuning_curves"]["ibi_bout_onset"]
    usv_starts = np.concatenate([o + np.array([0.0, 0.2, 0.4]) for o in (10.0, 30.0, 50.0)])
    runs = ibi_bout_runs(usv_starts, usv_starts + 0.1, 0.1155, 3)
    payload = ibi_bout_onset_block(np.empty(0), usv_starts, runs, 60.0, cfg, 0.1155,
                                   np.array([5.0, 15.0, 25.0]))
    assert payload["n_bouts"] == 3
    assert payload["n_shuffles"] == 3
    assert payload["ibi_threshold_seconds"] == pytest.approx(0.1155)
    for key in ("onset_gain", "peak_latency", "pre_onset_gain", "first_over_later",
                "fraction_first_above"):
        entry = payload[key]
        assert np.isnan(entry["observed"])
        assert np.isnan(entry["p_two_sided"])
        for name in ("null_mean", "null_std", "null_p0_5", "null_p2_5", "null_p97_5",
                     "null_p99_5"):
            assert np.isnan(entry[name])


def _decoding_maker(tmp_path: pathlib.Path, **overrides) -> NeuronalDecodingFigureMaker:
    """
    Description
    -----------
    A `NeuronalDecodingFigureMaker` reading archives from `tmp_path`, with the shipped
    visualization settings (any top-level key replaced by `overrides`) and a silent message sink.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Decoding directory.
    **overrides
        Top-level visualization-settings keys to replace.

    Returns
    -------
    maker (NeuronalDecodingFigureMaker)
        The figure maker.
    """

    settings = json.loads((SETTINGS_DIR / "visualizations_settings.json").read_text())
    settings.update(overrides)
    return NeuronalDecodingFigureMaker(decoding_directory=str(tmp_path),
                                       visualizations_parameter_dict=settings,
                                       message_output=lambda *_: None)


def _write_area_archive(tmp_path: pathlib.Path, session: str, area: str, rungs: tuple) -> None:
    """
    Description
    -----------
    Write one decoder archive `<session>_<area>.npz` holding ten observed draws and two hundred
    null draws per rung, in the layout `decode_usv_bout_occupancy` writes.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Decoding directory.
    session (str)
        Session identifier.
    area (str)
        Brain area acronym.
    rungs (tuple)
        Unit counts the archive carries.

    Returns
    -------
    None
    """

    rng = np.random.default_rng(3)
    draws = {str(c): 0.6 + 0.01 * rng.standard_normal(10) for c in rungs}
    nulls = {f"null_{c}": 0.5 + 0.01 * rng.standard_normal(200) for c in rungs}
    np.savez_compressed(tmp_path / f"{session}_{area}.npz", labels=np.array([0, 1]),
                        positions=np.array([0, 1]), ladder=np.array(rungs), **draws, **nulls)


def test_decoding_figure_rejects_bad_construction_and_unknown_buckets(
        tmp_path: pathlib.Path) -> None:
    """
    Description
    -----------
    The figure maker refuses a stray constructor keyword with `TypeError`, and an area whose
    display bucket the palette does not carry raises `KeyError` naming the bucket instead of
    falling back to an arbitrary colour.
    """

    with pytest.raises(TypeError, match="'out_dir'"):
        NeuronalDecodingFigureMaker(decoding_directory=str(tmp_path), out_dir=str(tmp_path))
    maker = _decoding_maker(tmp_path, brain_area_colors={"PAG": "#677470"})
    assert maker._area_colour("PAG") == "#677470"
    with pytest.raises(KeyError, match="does not carry"):
        maker._area_colour("MRN")


def test_decoding_figure_refuses_curves_and_panels_without_shared_rungs(
        tmp_path: pathlib.Path) -> None:
    """
    Description
    -----------
    A curve asked for rungs its archive does not carry raises rather than drawing an empty line,
    and a comparison panel whose area and focal area share no requested rung raises rather than
    drawing a panel that compares nothing.
    """

    session = "20250101_000000"
    _write_area_archive(tmp_path, session, "PAG", (5, 10))
    _write_area_archive(tmp_path, session, "MRN", (20,))
    maker = _decoding_maker(tmp_path)
    with pytest.raises(ValueError, match="carries no rung from the requested ladder"):
        maker._curve(session, "MRN", (5, 10))
    with pytest.raises(ValueError, match="share no rung"):
        maker.make_bout_decoding_figure(
            focal_area="PAG", focal_sessions=(("m1", session, 100),),
            comparisons=(("MRN", session, session, "same session"),) * 3,
            unit_counts=(5, 10), out_dir=tmp_path / "figures")


def test_silent_lattice_bin_inside_a_bout_is_not_a_negative() -> None:
    """
    Description
    -----------
    With bins narrower than the inter-USV silences of a bout, a lattice bin can hold no
    vocalization yet lie inside the bout's extent. Such a bin is bout time, so it must not become
    a negative: every negative stays clear of the bout `[0.0, 0.5]`, while the silent lattice bin
    `[0.10, 0.15)` and its siblings inside the bout are absent from the negatives.
    """

    starts = np.array([0.0, 0.2, 0.4])
    stops = starts + 0.1
    spans, labels, _mids, _positions = bout_occupancy_bins(
        0.05, starts, stops, starts, stops, 2.0, 0.1155, 3, 0.01)
    negatives = [spans[i][0] for i in np.flatnonzero(labels == 0)]
    assert negatives
    assert all(lo >= 0.5 - 1e-9 for lo, _hi in negatives)
    assert not any(lo == pytest.approx(0.10) for lo, _hi in negatives)
