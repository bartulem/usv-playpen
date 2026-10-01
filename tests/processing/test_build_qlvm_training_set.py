"""
@author: bartulem
Tests for processing/build_qlvm_training_set.

Covers the pure helpers directly (session typing, mask-count strata, uniform
headroom, capped-even water-filling, the natural / uniform within-session draw,
the type-budgeted draw, the type-stratified session split, the loudness floor and
the resize), and end-to-end builds on synthetic sessions (per-session spectrogram
H5 with a SAM mask group, a metadata YAML with the subjects' sexes and a USV
summary with the ``usv`` / ``squeak`` booleans): the masked and the unmasked-floor
twin of one draw, the squeak exclusion (pure squeaks and segments holding both out,
pure USVs in), the reference squeak index's strict rule, the left-out session types,
full_dataset, the settings checks and the per-session H5 fingerprints.
"""

from __future__ import annotations

import hashlib
import json

import h5py
import numpy as np
import polars as pls
import pytest
import yaml

from usv_playpen.processing.build_qlvm_training_set import (
    QLVMTrainingSetBuilder,
    allocate_even_quotas,
    apply_loudness_floor,
    compute_selected_rows_by_type,
    eligible_rows,
    file_sha256,
    mask_count_bin_labels,
    mask_count_bins,
    parse_int_list,
    parse_optional_float,
    parse_session_type_targets,
    read_reference_squeak_index,
    reference_strict_squeak_rows,
    select_rows_stratified,
    session_mask_counts,
    session_type_from_subject_sexes,
    split_sessions_by_type,
    stretch_specs,
    uniform_sampling_headroom,
    waterfill_level,
)

# Mirrors the shipped processing_settings.json block, with small targets.
_CFG = {
    "session_type_targets": {"MF": 6, "FF": 6, "MM": None},
    "draw_mode": "natural",
    "mask_count_bin_edges": [1, 2, 3],
    "length_threshold": 128.0,
    "require_mask": True,
    "exclude_squeaks": True,
    "strict_squeak_exclusion": False,
    "reference_squeak_index_path": "",
    "exclude_noise": False,
    "masking_type": "sam",
    "apply_mask": False,
    "floor": 0.2,
    "validation_split": 0.2,
    "random_state": 42,
    "full_dataset": False,
    "target_shape": [32, 32],
    "time_stretch": False,
}

_SEXES = {"MF": ["male", "female"], "FF": ["female", "female"], "MM": ["male", "male"], "lone_male": ["male"]}


def test_session_type_from_subject_sexes():
    """Two-subject sessions are typed by their sex pair in any order and case; one
    subject gives a lone type; everything else is "other"."""
    assert session_type_from_subject_sexes(["Female", "male"]) == "MF"
    assert session_type_from_subject_sexes(["male", "male"]) == "MM"
    assert session_type_from_subject_sexes(["female", "female"]) == "FF"
    assert session_type_from_subject_sexes(["male"]) == "lone_male"
    assert session_type_from_subject_sexes(["female"]) == "lone_female"
    assert session_type_from_subject_sexes(["male", "female", "male"]) == "other"
    assert session_type_from_subject_sexes(["male", "unknown"]) == "other"


def test_mask_count_bins_and_labels():
    """Edges are inclusive lower bounds above stratum 0 and the top stratum is open."""
    counts = np.array([0, 1, 2, 3, 4, 9])
    assert mask_count_bins(counts, [1, 2, 3]).tolist() == [0, 1, 2, 3, 3, 3]
    assert mask_count_bins(counts, [1, 2, 3, 4, 5]).tolist() == [0, 1, 2, 3, 4, 5]
    assert mask_count_bin_labels([1, 2, 3]) == ["0", "1", "2", "3+"]
    assert mask_count_bin_labels([1, 3]) == ["0", "1-2", "3+"]


def test_session_mask_counts_from_spectrogram_index(tmp_path):
    """Instance counts come from mask/<session>/spectrogram_index; a session without a
    mask group has zero instances on every row."""
    with h5py.File(tmp_path / "a.h5", "w") as h5_file:
        h5_file.create_dataset("mask/s/spectrogram_index", data=np.array([0, 0, 2, 2, 2]))
        assert session_mask_counts(h5_file, "s", 4).tolist() == [2, 0, 3, 0]
        assert session_mask_counts(h5_file, "t", 3).tolist() == [0, 0, 0]


def test_eligible_rows_filters():
    """Duration window, exclusions and require_mask each remove rows."""
    durations = np.array([0, 10, 200, 10, 10])
    counts = np.array([1, 1, 1, 0, 2])
    excluded = np.array([False, False, False, False, True])
    assert eligible_rows(durations, counts, excluded, 128.0, True).tolist() == [1]
    assert eligible_rows(durations, counts, excluded, 128.0, False).tolist() == [1, 3]


def test_uniform_sampling_headroom():
    """Headroom is the number of non-empty strata times the smallest of them."""
    assert uniform_sampling_headroom(np.array([1, 1, 1, 2, 2, 3])) == 3
    assert uniform_sampling_headroom(np.array([1, 1, 3, 3, 3])) == 4
    assert uniform_sampling_headroom(np.array([], dtype=np.int64)) == 0


def test_allocate_even_quotas_exact_capped_and_seeded():
    """Quotas sum to the target exactly, respect every cap, hand the remainder only
    to entries with room, and depend on the seed alone."""
    available = np.array([2, 50, 7, 30, 30])
    quotas = allocate_even_quotas(available, 60, 3)
    assert quotas.sum() == 60
    assert np.all(quotas <= available)
    level = waterfill_level(available, 60)
    assert np.all(quotas[available <= level] == available[available <= level])
    assert set(quotas[available > level].tolist()) <= {level, level + 1}
    assert np.array_equal(quotas, allocate_even_quotas(available, 60, 3))
    assert allocate_even_quotas(available, 1000, 3).tolist() == available.tolist()


def test_allocate_even_quotas_remainder_follows_the_seeded_permutation():
    """The remainder goes to the entries with room in the order of
    default_rng(seed).permutation over all entries."""
    available = np.array([10, 10, 10, 10])
    quotas = allocate_even_quotas(available, 6, 7)
    order = np.random.default_rng(7).permutation(4)
    expected = np.ones(4, dtype=np.int64)
    expected[order[:2]] += 1
    assert quotas.tolist() == expected.tolist()


def test_select_rows_stratified_natural_is_a_plain_choice():
    """The natural draw is rng.choice over the row positions, sorted; a quota that
    covers every row keeps them all without using the generator."""
    rows = np.arange(10, 30)
    bins = np.ones(20, dtype=np.int64)
    got = select_rows_stratified(rows, bins, 5, np.random.default_rng(1), "natural")
    assert got.tolist() == np.sort(rows[np.random.default_rng(1).choice(20, size=5, replace=False)]).tolist()
    rng = np.random.default_rng(1)
    assert select_rows_stratified(rows, bins, 25, rng, "natural").tolist() == rows.tolist()
    assert rng.random() == np.random.default_rng(1).random()


def test_select_rows_stratified_uniform_equalizes_strata():
    """The uniform draw takes equal shares of the non-empty strata (plus a remainder
    of one row per stratum)."""
    rows = np.arange(100)
    bins = np.array([1] * 60 + [2] * 25 + [3] * 15)
    got = select_rows_stratified(rows, bins, 31, np.random.default_rng(0), "uniform")
    per_stratum = np.bincount(bins[got], minlength=4)[1:]
    assert got.size == 31
    assert sorted(per_stratum.tolist()) == [10, 10, 11]
    with pytest.raises(ValueError, match="draw_mode"):
        select_rows_stratified(rows, bins, 3, np.random.default_rng(0), "bad")


def test_compute_selected_rows_by_type_budgets_and_take_all():
    """Budgeted types hit their target exactly; a None target takes the type whole;
    the report records availability, target and per-stratum counts."""
    eligible = {"a": np.arange(10), "b": np.arange(4), "c": np.arange(6)}
    bins = {key: np.ones(value.size, dtype=np.int64) for key, value in eligible.items()}
    types = {"a": "MF", "b": "MF", "c": "MM"}
    selected, report = compute_selected_rows_by_type(eligible, bins, types, {"MF": 8, "MM": None}, "natural", 0, ["0", "1+"])
    assert selected["a"].size + selected["b"].size == 8
    assert selected["b"].size == 4
    assert selected["c"].tolist() == list(range(6))
    assert report["MF"] == {"n_sessions": 2, "n_sessions_drawn": 2, "available": 14, "capacity": 14,
                            "target": 8, "drawn": 8, "per_bin": {"0": 0, "1+": 8}}
    assert report["MM"]["target"] == "all"


def test_split_sessions_by_type_whole_sessions_per_type():
    """Every session lands on exactly one side, each type holds out at least the
    requested share of its rows, and the result depends on the seed alone."""
    counts = {f"s{i}": 10 + i for i in range(10)}
    types = {key: ("MF" if i < 6 else "FF") for i, key in enumerate(counts)}
    train, val = split_sessions_by_type(list(counts), types, counts, 0.2, 5)
    assert set(train).isdisjoint(val)
    assert set(train) | set(val) == set(counts)
    for session_type in ("MF", "FF"):
        total = sum(c for k, c in counts.items() if types[k] == session_type)
        held = sum(counts[k] for k in val if types[k] == session_type)
        assert held >= 0.2 * total
    assert (train, val) == split_sessions_by_type(list(reversed(counts)), types, counts, 0.2, 5)


def test_apply_loudness_floor_range():
    """The floor maps each spectrogram to [0, 1] and silences its bottom fraction."""
    specs = np.linspace(0.0, 10.0, 2 * 4 * 5, dtype=np.float32).reshape(2, 4, 5)
    floored = apply_loudness_floor(specs, 0.2)
    assert floored.dtype == np.float32
    assert floored.min() == 0.0
    assert floored.max() == pytest.approx(1.0, abs=1e-6)
    assert np.count_nonzero(floored[0] == 0.0) >= 4


def test_cli_value_parsers():
    """The JSON targets, the comma-separated edges and the optional floor decode."""
    assert parse_session_type_targets('{"MF": 5, "MM": null}') == {"MF": 5, "MM": None}
    assert parse_int_list("1,2,3") == [1, 2, 3]
    assert parse_optional_float("none") is None
    assert parse_optional_float("0.2") == 0.2


def test_stretch_specs_outputs_target_shape():
    """Every spectrogram is resized to target_shape regardless of input size."""
    specs = np.random.default_rng(0).random((3, 64, 90)).astype(np.float32)
    durations = np.array([90, 45, 10])
    out = stretch_specs(specs, durations, (128, 128), time_stretch=False)
    assert out.shape == (3, 128, 128)
    out_ts = stretch_specs(specs, durations, (128, 128), time_stretch=True)
    assert out_ts.shape == (3, 128, 128)


def test_simple_resize_preserves_full_signal_on_time_upsampling():
    """The non-warp (time_stretch=False) resize must keep the WHOLE signal window
    when time-upsampling (target wider than the native spec). A regression that
    slices the zoomed array with a native-coordinate length would truncate it to
    the first ``duration`` columns and discard the rest of the signal."""
    n_f, n_t, target_t = 8, 10, 100
    spec = np.ones((n_f, n_t), dtype=np.float32)   # the entire spectrogram is signal
    durations = np.array([n_t])
    out = stretch_specs(spec[None], durations, (n_f, target_t), time_stretch=False)[0]
    # The all-ones signal (duration=10) zoomed by 100/10=10 fills all 100 target
    # columns; a native-coordinate slice would keep only ~10 and zero the other 90.
    nonzero_cols = int(np.count_nonzero(out.sum(axis=0)))
    assert nonzero_cols == target_t, f"expected {target_t} signal columns, got {nonzero_cols}"
    assert out.sum() == pytest.approx(n_f * target_t, rel=0.02)


def _write_session(tmp_path, session_id, session_type, n=8, n_f=16, n_t=20, squeak_rows=(), both_rows=(), with_summary=True):
    """
    Create a synthetic session root: ``audio/spectrograms/<session>_spectrograms.h5``
    (a ``spectrogram/<session>`` group with random spectrograms and durations of 12
    bins, and a ``mask/<session>`` group giving row i ``1 + i % 3`` mask instances
    except row 0, which has none), ``<session>_metadata.yaml`` with the subjects'
    sexes of ``session_type``, and ``audio/<session>_usv_summary.csv`` whose
    ``(usv, squeak)`` booleans are ``(false, true)`` (pure squeak) on ``squeak_rows``,
    ``(true, true)`` (both) on ``both_rows`` and ``(true, false)`` (pure USV)
    elsewhere. Returns the session root.
    """
    rng = np.random.default_rng(abs(hash(session_id)) % (2**32))
    root = tmp_path / session_id
    spec_dir = root / "audio" / "spectrograms"
    spec_dir.mkdir(parents=True)
    with h5py.File(spec_dir / f"{session_id}_spectrograms.h5", "w") as f:
        f.create_dataset("frequency_bins", data=np.linspace(30000.0, 120000.0, n_f))
        group = f.create_group(f"spectrogram/{session_id}")
        group.create_dataset("spectrograms", data=rng.random((n, n_f, n_t)).astype(np.float32))
        group.create_dataset("durations", data=np.full(n, 12, dtype=np.int64))
        segmentations, index = [], []
        for row in range(1, n):
            for instance in range(1 + row % 3):
                seg = np.zeros((n_f, n_t), dtype=bool)
                seg[2 + instance:8 + instance, 1:9] = True
                segmentations.append(seg)
                index.append(row)
        mask_group = f.create_group(f"mask/{session_id}")
        mask_group.create_dataset("segmentations", data=np.stack(segmentations))
        mask_group.create_dataset("spectrogram_index", data=np.array(index, dtype=np.int64))
    with (root / f"{session_id}_metadata.yaml").open("w") as handle:
        yaml.safe_dump({"Subjects": [{"subject_id": str(i), "sex": sex} for i, sex in enumerate(_SEXES[session_type])]}, handle)
    if with_summary:
        pls.DataFrame({
            "usv_id": [f"{i:06d}" for i in range(n)],
            "start": np.arange(n, dtype=np.float64),
            "stop": np.arange(n, dtype=np.float64) + 0.05,
            "usv": [i not in squeak_rows for i in range(n)],
            "squeak": [i in squeak_rows or i in both_rows for i in range(n)],
        }).write_csv(root / "audio" / f"{session_id}_usv_summary.csv")
    return root


def _build(tmp_path, roots, cfg, name="out", **kwargs):
    out_dir = tmp_path / name
    QLVMTrainingSetBuilder(
        root_directories=[str(root) for root in roots],
        output_directory=str(out_dir),
        input_parameter_dict={"build_qlvm_training_set": cfg},
        message_output=lambda *_a, **_kw: None,
        **kwargs,
    ).build()
    return out_dir


@pytest.fixture
def cohort(tmp_path, mocker):
    """Two MF, two FF, one MM and one lone_male session; one MF session has a squeak (row 1) and a both (row 2)."""
    mocker.patch("usv_playpen.processing.build_qlvm_training_set.smart_wait")
    return [
        _write_session(tmp_path, "20230101_100000", "MF", squeak_rows=(1,), both_rows=(2,)),
        _write_session(tmp_path, "20230101_110000", "MF"),
        _write_session(tmp_path, "20230101_120000", "FF"),
        _write_session(tmp_path, "20230101_130000", "FF"),
        _write_session(tmp_path, "20230101_140000", "MM"),
        _write_session(tmp_path, "20230101_150000", "lone_male"),
    ]


def test_build_masked_and_floor_twins_share_rows(tmp_path, cohort):
    """The masked build (apply_mask) and the unmasked-floor build of the same draw
    hold the same rows in the same order and split; the masked spectrograms are zero
    off the mask, the floored ones lie in [0, 1]; targets, take-all types, left-out
    types, squeak exclusion and the rows without a mask all behave."""
    floor_dir = _build(tmp_path, cohort, _CFG, "floor")
    masked_dir = _build(tmp_path, cohort, {**_CFG, "apply_mask": True, "floor": None}, "masked")
    for split in ("train_data.npz", "val_data.npz"):
        floored, masked = np.load(floor_dir / split), np.load(masked_dir / split)
        for key in ("spec_id", "session_id", "session_type", "masks", "masks_len", "mask_count", "durations"):
            assert np.array_equal(floored[key], masked[key]), key
        assert not bool(floored["apply_mask"])
        assert bool(masked["apply_mask"])
        assert np.all(masked["spectrograms"][masked["masks"] == 0] == 0)
        assert floored["spectrograms"].min() >= 0.0
        assert floored["spectrograms"].max() <= 1.0
        assert floored["spectrograms"].shape[1:] == (32, 32)
    meta = np.load(floor_dir / "metadata.npz")
    assert float(meta["floor"]) == 0.2
    assert "floor" not in np.load(masked_dir / "metadata.npz").files
    report = json.loads(str(meta["type_report"]))
    assert set(report) == {"MF", "FF", "MM"}
    assert report["MF"]["drawn"] == 6
    assert report["FF"]["drawn"] == 6
    assert report["MM"]["drawn"] == 7
    ids = np.concatenate([np.load(floor_dir / s)["spec_id"] for s in ("train_data.npz", "val_data.npz")]).tolist()
    assert not any(i.endswith(("_0", "100000_1", "100000_2")) for i in ids)
    assert not any(i.startswith("20230101_150000") for i in ids)
    split = json.loads(str(meta["split_sessions"]))
    assert set(split["train"]).isdisjoint(split["validation"])
    assert json.loads(str(meta["excluded_by_type"]))["MF"] == 2


def test_build_uniform_draw_equalizes_strata(tmp_path, cohort):
    """A uniform draw within capacity gives the budgeted types equal strata."""
    out_dir = _build(tmp_path, cohort, {**_CFG, "draw_mode": "uniform", "session_type_targets": {"FF": 12}})
    report = json.loads(str(np.load(out_dir / "metadata.npz")["type_report"]))
    assert report["FF"]["per_bin"] == {"0": 0, "1": 4, "2": 4, "3+": 4}


def test_build_row_exclusions_override_the_summary(tmp_path, cohort):
    """row_exclusions replaces the summary flags of the sessions it names."""
    exclusions = {"20230101_100000": np.zeros(8, dtype=bool)}
    out_dir = _build(tmp_path, cohort, {**_CFG, "full_dataset": True}, row_exclusions=exclusions)
    full = np.load(out_dir / "full_data.npz")
    assert "20230101_100000_1" in full["spec_id"].tolist()
    assert bool(np.load(out_dir / "metadata.npz")["row_exclusions_override"])


def _write_reference_index(tmp_path, roots, flagged=None, missing_session=None):
    """
    Write a synthetic reference squeak index CSV (the columns the strict rule reads
    plus one it ignores): every row of every session in ``roots`` (except
    ``missing_session``), with ``is_bbv`` / ``n_bouts_min3`` taken from
    ``flagged`` (session id -> {row: (is_bbv, n_bouts_min3)}), False / 0 elsewhere.
    Returns the CSV path.
    """
    flagged = {} if flagged is None else flagged
    rows = []
    for root in roots:
        if root.name == missing_session:
            continue
        for row in range(8):
            is_bbv, n_bouts = flagged.get(root.name, {}).get(row, (False, 0))
            rows.append({"session_id": root.name, "seg_index": row, "p_bbv": 0.1, "is_bbv": is_bbv, "n_bouts_min3": n_bouts})
    path = tmp_path / "bbv_segment_index.csv"
    pls.DataFrame(rows).write_csv(path)
    return path


def test_build_copies_the_summary_condition_columns_row_for_row(tmp_path, cohort):
    """Every split carries mean_freq_hz, freq_bandwidth_hz, loudness_db and
    spectral_entropy of the session's USV summary, row for row with its
    spectrograms (the raw values a conditional train-qlvm run conditions on); a
    summary without them, or with a null, gives NaN."""
    values = {}
    for n_session, root in enumerate(cohort[:-1]):
        summary = root / "audio" / f"{root.name}_usv_summary.csv"
        frame = pls.read_csv(summary)
        if n_session == 0:
            continue
        rng = np.random.default_rng(n_session)
        columns = {
            "mean_freq_hz": rng.uniform(40000.0, 100000.0, frame.height),
            "freq_bandwidth_hz": rng.uniform(1000.0, 30000.0, frame.height),
            "loudness_db": rng.uniform(30.0, 90.0, frame.height),
            "spectral_entropy": rng.uniform(1.0, 5.0, frame.height),
        }
        loudness = [None if row == 3 else value for row, value in enumerate(columns["loudness_db"].tolist())]
        entropy = [None if row == 5 else value for row, value in enumerate(columns["spectral_entropy"].tolist())]
        frame.with_columns(
            pls.Series("mean_freq_hz", columns["mean_freq_hz"]),
            pls.Series("freq_bandwidth_hz", columns["freq_bandwidth_hz"]),
            pls.Series("loudness_db", loudness, dtype=pls.Float64),
            pls.Series("spectral_entropy", entropy, dtype=pls.Float64),
        ).write_csv(summary)
        columns["loudness_db"][3] = np.nan
        columns["spectral_entropy"][5] = np.nan
        values[root.name] = columns
    out_dir = _build(tmp_path, cohort, _CFG)
    for split_name in ("train_data.npz", "val_data.npz"):
        with np.load(out_dir / split_name) as split:
            for n_row, spec_id in enumerate(split["spec_id"].tolist()):
                session_id, row = spec_id.rsplit("_", 1)
                for column in ("mean_freq_hz", "freq_bandwidth_hz", "loudness_db", "spectral_entropy"):
                    expected = values[session_id][column][int(row)] if session_id in values else np.nan
                    np.testing.assert_array_equal(split[column][n_row], expected)


def test_build_exclude_squeaks_keeps_usv_rows_only(tmp_path, cohort):
    """exclude_squeaks leaves out pure squeaks (squeak true, usv false) and segments
    holding both (usv and squeak true) alike -- usv true alone does not admit a row --
    so only pure USVs are drawn; without it both come back, and null booleans (noise
    or unscorable) do not mark a squeak."""
    summary = cohort[1] / "audio" / f"{cohort[1].name}_usv_summary.csv"
    pls.read_csv(summary).with_columns(
        pls.Series("usv", [True, True, True, None, True, True, True, True], dtype=pls.Boolean),
        pls.Series("squeak", [False, False, False, None, False, False, False, False], dtype=pls.Boolean),
    ).write_csv(summary)
    kept_dir = _build(tmp_path, cohort, {**_CFG, "full_dataset": True}, "usv_only")
    all_dir = _build(tmp_path, cohort, {**_CFG, "full_dataset": True, "exclude_squeaks": False}, "all")
    kept_ids = np.load(kept_dir / "full_data.npz")["spec_id"].tolist()
    all_ids = np.load(all_dir / "full_data.npz")["spec_id"].tolist()
    assert "20230101_100000_1" not in kept_ids and "20230101_100000_2" not in kept_ids
    assert "20230101_110000_3" in kept_ids
    assert sorted(set(all_ids) - set(kept_ids)) == ["20230101_100000_1", "20230101_100000_2"]


def test_build_exclude_squeaks_needs_vocal_flags(tmp_path, cohort):
    """A summary without the usv boolean (never scored by detect-usv-squeaks, or scored only by the
    retired binary detector, whose summaries carry a squeak column alone) stops a build that
    excludes squeaks."""
    summary = cohort[0] / "audio" / f"{cohort[0].name}_usv_summary.csv"
    pls.read_csv(summary).drop("usv").write_csv(summary)
    with pytest.raises(ValueError, match="usv"):
        _build(tmp_path, cohort, _CFG)


def test_read_reference_squeak_index_applies_the_strict_rule(tmp_path, cohort):
    """The strict rule marks a row when is_bbv is true OR n_bouts_min3 >= 1; rows the
    index does not list are not marked; a missing session or an out-of-range row raises."""
    path = _write_reference_index(tmp_path, cohort, {"20230101_100000": {3: (True, 0), 4: (False, 2), 5: (False, 0)}})
    index = read_reference_squeak_index(str(path))
    squeak = reference_strict_squeak_rows(index, "20230101_100000", 8)
    assert np.flatnonzero(squeak).tolist() == [3, 4]
    assert not reference_strict_squeak_rows(index, "20230101_110000", 10).any()
    with pytest.raises(ValueError, match="not in the reference squeak index"):
        reference_strict_squeak_rows(index, "20990101_000000", 8)
    with pytest.raises(ValueError, match="does not describe"):
        reference_strict_squeak_rows(index, "20230101_100000", 5)


def test_build_strict_squeak_exclusion_reads_the_reference_index(tmp_path, cohort):
    """
    strict_squeak_exclusion takes the squeak rows from the reference squeak index's
    strict rule (is_bbv OR n_bouts_min3 >= 1) INSTEAD of the summary's squeak flag: the
    index's rows go, the squeak-bearing rows the index does not flag come back.
    """
    path = _write_reference_index(tmp_path, cohort, {"20230101_100000": {3: (True, 0), 4: (False, 1)}})
    default_dir = _build(tmp_path, cohort, {**_CFG, "full_dataset": True}, "default")
    strict_cfg = {**_CFG, "full_dataset": True, "strict_squeak_exclusion": True, "reference_squeak_index_path": str(path)}
    strict_dir = _build(tmp_path, cohort, strict_cfg, "strict")
    default_ids = set(np.load(default_dir / "full_data.npz")["spec_id"].tolist())
    strict_ids = set(np.load(strict_dir / "full_data.npz")["spec_id"].tolist())
    assert sorted(default_ids - strict_ids) == ["20230101_100000_3", "20230101_100000_4"]
    assert sorted(strict_ids - default_ids) == ["20230101_100000_1", "20230101_100000_2"]
    meta = np.load(strict_dir / "metadata.npz")
    assert bool(meta["strict_squeak_exclusion"])
    assert str(meta["reference_squeak_index_path"]) == str(path)
    assert not bool(np.load(default_dir / "metadata.npz")["strict_squeak_exclusion"])


def test_build_strict_squeak_exclusion_needs_every_session_in_the_index(tmp_path, cohort):
    """A session the reference index does not list has unknown strict exclusions and stops the build."""
    path = _write_reference_index(tmp_path, cohort, missing_session="20230101_120000")
    with pytest.raises(ValueError, match="20230101_120000 is not in the reference squeak index"):
        _build(tmp_path, cohort, {**_CFG, "strict_squeak_exclusion": True, "reference_squeak_index_path": str(path)})


def test_build_full_dataset_writes_every_eligible_row(tmp_path, cohort):
    """full_dataset takes every eligible row of the listed types, writes the split
    pair and then full_data.npz, whose rows are the union of the pair's."""
    out_dir = _build(tmp_path, cohort, {**_CFG, "full_dataset": True})
    full = np.load(out_dir / "full_data.npz")
    assert full["spec_id"].size == 5 + 7 + 7 + 7 + 7
    pair = np.concatenate([np.load(out_dir / s)["spec_id"] for s in ("train_data.npz", "val_data.npz")])
    assert sorted(pair.tolist()) == sorted(full["spec_id"].tolist())
    assert (out_dir / "full_data.npz").stat().st_mtime >= (out_dir / "train_data.npz").stat().st_mtime


@pytest.mark.parametrize(("override", "message"), [
    ({"validation_split": 0.0}, "validation_split"),
    ({"masking_type": "none"}, "masking_type 'none'"),
    ({"apply_mask": True}, "loudness floor"),
    ({"mask_count_bin_edges": [2, 1]}, "mask_count_bin_edges"),
    ({"exclude_squeaks": False, "strict_squeak_exclusion": True, "reference_squeak_index_path": "x.csv"}, "needs exclude_squeaks"),
    ({"strict_squeak_exclusion": True}, "reference_squeak_index_path is empty"),
])
def test_build_rejects_inconsistent_settings(tmp_path, cohort, override, message):
    """Inconsistent settings stop the build before anything is read."""
    with pytest.raises(ValueError, match=message):
        _build(tmp_path, cohort, {**_CFG, **override})


def test_build_summary_row_mismatch_raises(tmp_path, cohort):
    """A summary whose row count differs from the H5's cannot be joined by row."""
    summary = cohort[0] / "audio" / "20230101_100000_usv_summary.csv"
    pls.read_csv(summary).head(5).write_csv(summary)
    with pytest.raises(ValueError, match="rows"):
        _build(tmp_path, cohort, _CFG)


def test_build_fingerprints_every_session_h5(tmp_path, cohort):
    """
    The builder records each session H5's SHA-256 (identical to hashing its bytes),
    its row count and how many rows entered the set: in metadata.npz, in a
    sha256sum-format SESSION_H5.sha256, and in SESSION_H5.tsv laid out like the v2
    package's baseline. Only sessions of listed types are read.
    """
    out_dir = _build(tmp_path, cohort[:2], {**_CFG, "full_dataset": True})
    h5_a = cohort[0] / "audio" / "spectrograms" / "20230101_100000_spectrograms.h5"
    h5_b = cohort[1] / "audio" / "spectrograms" / "20230101_110000_spectrograms.h5"
    expected = {str(h5_a): hashlib.sha256(h5_a.read_bytes()).hexdigest(),
                str(h5_b): hashlib.sha256(h5_b.read_bytes()).hexdigest()}
    assert file_sha256(h5_a, chunk_bytes=7) == expected[str(h5_a)]

    meta = np.load(out_dir / "metadata.npz", allow_pickle=True)
    paths = meta["spectrogram_h5_paths"].tolist()
    assert meta["spectrogram_h5_sha256"].tolist() == [expected[p] for p in paths]
    assert meta["session_ids"].tolist() == ["20230101_100000", "20230101_110000"]
    assert meta["spectrogram_h5_rows"].tolist() == [8, 8]
    assert meta["spectrogram_h5_corpus_rows"].tolist() == [5, 7]

    lines = (out_dir / "SESSION_H5.sha256").read_text().splitlines()
    assert lines == [f"{expected[p]}  {p}" for p in paths]
    tsv = (out_dir / "SESSION_H5.tsv").read_text().splitlines()
    assert tsv[0].split("\t") == ["session", "h5_rows", "corpus_rows", "bytes", "sha256", "path"]
    assert tsv[1].split("\t")[:3] == ["20230101_100000", "8", "5"]
    assert tsv[2].split("\t")[3] == str(h5_b.stat().st_size)
