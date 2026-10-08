"""
@author: bartulem
Tests for processing/build_qlvm_squeak_training_set.

The frame count, the candidate selection from the USV summary (squeak true:
pure squeaks and both, the crop window that widens a segment to hold a squeak running
past it), the crop gates of both crop windows, the per-session/stratum cap, the
natural and uniform duration-stratified draws and the crop normalizations are
tested directly; the end-to-end build runs on synthetic sessions (a metadata YAML
and a USV summary with the call-class columns) with the audio spectrogram rebuild
mocked by a deterministic function of the window, so the crops, the draw, the
split and the written columns can be checked without wav files.
"""

from __future__ import annotations

import json

import numpy as np
import polars as pls
import pytest
import yaml

from usv_playpen.processing import build_qlvm_squeak_training_set as squeak_set
from usv_playpen.processing.detect_usv_squeaks import FRAME_DT_S, squeak_window_n_frames

_CFG = {
    "draw_mode": "natural",
    "n_total": 10,
    "per_session_bin_cap": 0,
    "duration_bin_edges": [12, 20],
    "context_frames": 2,
    "min_trimmed_frames": 8,
    "crop_window": "full_length",
    "crop_normalization": "per_crop",
    "exclude_noise": False,
    "exclude_metadata_audio_channels": True,
    "validation_split": 0.2,
    "random_state": 42,
    "full_dataset": False,
    "target_shape": [128, 128],
}


def test_squeak_window_n_frames():
    """1 + n_samples // 512 frames, and none below one 2048-sample window."""
    start = np.array([0.0, 0.0, 1.0])
    stop = np.array([0.1, 0.008, 1.0 + 2048 / 250000])
    assert squeak_window_n_frames(start, stop).tolist() == [1 + 25000 // 512, 0, 5]


def test_squeak_candidates_from_summary():
    """squeak and both rows with an envelope are candidates (usv rows and rows without an
    envelope are not); noise is left out on request; the window grows to hold a squeak
    that starts before its segment; extents become frame indices of the window; the
    probability is p_squeak + p_both."""
    dt = FRAME_DT_S
    summary = pls.DataFrame({
        "start": [1.0, 2.0, 3.0, 4.0, 5.0],
        "stop": [1.1, 2.1, 3.1, 4.1, 5.1],
        "usv": [False, True, True, True, False],
        "squeak": [True, False, True, True, True],
        "p_squeak": [0.8, 0.1, 0.3, 0.2, 0.6],
        "p_both": [0.1, 0.0, 0.6, 0.7, 0.3],
        "squeak_start": [1.0 + 3 * dt, None, 3.0 - 5 * dt, 4.0 + 2 * dt, None],
        "squeak_end": [1.0 + 20 * dt, None, 3.0 + 9 * dt, 4.0 + 30 * dt, None],
        "noise": [False, False, False, True, False],
    })
    candidates = squeak_set.squeak_candidates_from_summary(summary, exclude_noise=False)
    assert candidates["row"].tolist() == [0, 2, 3]
    assert candidates["start_s"].tolist() == pytest.approx([1.0, 3.0 - 7 * dt, 4.0])
    assert candidates["stop_s"].tolist() == pytest.approx([1.1, 3.1, 4.1])
    assert candidates["first_raw"].tolist() == [3, 2, 2]
    assert candidates["last_raw"].tolist() == [20, 16, 30]
    assert candidates["probability"].tolist() == pytest.approx([0.9, 0.9, 0.9])
    assert squeak_set.squeak_candidates_from_summary(summary, exclude_noise=True)["row"].tolist() == [0, 2]
    with pytest.raises(ValueError, match="column"):
        squeak_set.squeak_candidates_from_summary(summary.drop("squeak_end"), exclude_noise=False)
    with pytest.raises(ValueError, match="usv"):
        squeak_set.squeak_candidates_from_summary(summary.drop("usv"), exclude_noise=False)


def test_squeak_crop_gates_windows():
    """first_128_frames drops right-censored squeaks and clips to frame 127;
    full_length keeps them but drops crops wider than 128 frames; both drop
    narrow crops."""
    n_frames = np.array([50, 300, 300, 300, 50])
    first_raw = np.array([10, 100, 10, 10, 20])
    last_raw = np.array([30, 200, 60, 250, 21])
    keep, first, last, gates = squeak_set.squeak_crop_gates(n_frames, first_raw, last_raw, "first_128_frames", 2, 8)
    assert keep.tolist() == [True, False, True, False, False]
    assert (first[0], last[0]) == (8, 32)
    assert [label for label, _ in gates] == ["squeak candidates", "drop right-censored", "drop trimmed < 8"]
    keep, first, last, _ = squeak_set.squeak_crop_gates(n_frames, first_raw, last_raw, "full_length", 2, 8)
    assert keep.tolist() == [True, True, True, False, False]
    assert (first[1], last[1]) == (98, 202)
    with pytest.raises(ValueError, match="crop_window"):
        squeak_set.squeak_crop_gates(n_frames, first_raw, last_raw, "bad", 2, 8)


def test_cap_rows_per_session_bin():
    """No (session, stratum) keeps more than the cap, smaller groups are kept whole,
    and the groups are drawn in order of first appearance."""
    sessions = np.array(["a"] * 5 + ["b"] * 2 + ["a"] * 3)
    bins = np.array([0] * 5 + [0] * 2 + [1] * 3)
    keep = squeak_set.cap_rows_per_session_bin(sessions, bins, 2, np.random.default_rng(0))
    assert keep[5:7].all()
    assert keep[:5].sum() == 2
    assert keep[7:].sum() == 2
    rng = np.random.default_rng(0)
    first_group = rng.choice(np.arange(5), size=2, replace=False)
    assert np.flatnonzero(keep[:5]).tolist() == sorted(first_group.tolist())


def test_stratified_duration_draw():
    """natural draws n rows; uniform shares them equally across strata and gives a
    small stratum all it has."""
    bins = np.array([0] * 30 + [1] * 30 + [2] * 3)
    natural = squeak_set.stratified_duration_draw(bins, 12, 3, "natural", np.random.default_rng(0))
    assert natural.size == 12
    assert np.unique(natural).size == 12
    uniform = squeak_set.stratified_duration_draw(bins, 21, 3, "uniform", np.random.default_rng(0))
    assert np.bincount(bins[uniform], minlength=3).tolist() == [9, 9, 3]
    with pytest.raises(ValueError, match="draw_mode"):
        squeak_set.stratified_duration_draw(bins, 5, 3, "bad", np.random.default_rng(0))


def test_normalize_crop():
    """per_crop maps a crop to [0, 1); absolute is the fixed dB transform."""
    crop = np.array([[-100.0, 0.0], [20.0, 50.0]], dtype=np.float32)
    per_crop = squeak_set.normalize_crop(crop, "per_crop")
    assert per_crop.dtype == np.float32
    assert per_crop.min() == 0.0
    assert per_crop.max() == pytest.approx(1.0, abs=1e-6)
    assert np.allclose(squeak_set.normalize_crop(crop, "absolute"), [[-1.0, 1 / 3], [0.6, 1.0]])


def _write_session(tmp_path, session_id, sexes, n_squeaks):
    """
    Create a synthetic session root with ``<session>_metadata.yaml`` and
    ``audio/<session>_usv_summary.csv``: ``n_squeaks`` rows of 0.2 s (98 frames), squeak
    and both alternating, whose envelopes cover frames 5 .. 10 + 4 * (row % 6) (inside the
    segment with its context, so the window is the segment), plus one usv row.
    """
    root = tmp_path / session_id
    (root / "audio").mkdir(parents=True)
    with (root / f"{session_id}_metadata.yaml").open("w") as handle:
        yaml.safe_dump({"Subjects": [{"subject_id": str(i), "sex": sex} for i, sex in enumerate(sexes)]}, handle)
    n = n_squeaks + 1
    start = np.arange(n, dtype=np.float64)
    pls.DataFrame({
        "usv_id": [f"{i:06d}" for i in range(n)],
        "start": start,
        "stop": start + 0.2,
        "usv": [(i % 2 == 0) if i < n_squeaks else True for i in range(n)],
        "squeak": [i < n_squeaks for i in range(n)],
        "p_squeak": [0.5 if i < n_squeaks else 0.0 for i in range(n)],
        "p_both": [0.4 if i < n_squeaks else 0.0 for i in range(n)],
        "squeak_start": [start[i] + 5 * FRAME_DT_S if i < n_squeaks else None for i in range(n)],
        "squeak_end": [start[i] + (10 + 4 * (i % 6)) * FRAME_DT_S if i < n_squeaks else None for i in range(n)],
    }).write_csv(root / "audio" / f"{session_id}_usv_summary.csv")
    return root


def _fake_spectrograms(window_start_s, window_stop_s, **_kwargs):
    """A deterministic (128, n_frames) dB 'spectrogram' per window, the window start in
    every cell of column t plus t, standing in for the audio rebuild."""
    n_frames = squeak_window_n_frames(window_start_s, window_stop_s)
    return [np.tile(np.arange(n, dtype=np.float32), (128, 1)) - 60.0 + float(start) for start, n in zip(window_start_s, n_frames, strict=True)]


@pytest.fixture
def cohort(tmp_path, mocker):
    """Three MF sessions and two FF sessions, with the spectrogram rebuild mocked."""
    mocker.patch("usv_playpen.processing.build_qlvm_squeak_training_set.smart_wait")
    mocker.patch("usv_playpen.processing.build_qlvm_squeak_training_set.squeak_window_spectrograms", side_effect=_fake_spectrograms)
    return [
        _write_session(tmp_path, "20230101_120000", ["male", "female"], 9),
        _write_session(tmp_path, "20230101_100000", ["male", "female"], 7),
        _write_session(tmp_path, "20230101_110000", ["female", "male"], 5),
        _write_session(tmp_path, "20230101_130000", ["female", "female"], 6),
        _write_session(tmp_path, "20230101_140000", ["female", "female"], 4),
    ]


def _build(tmp_path, roots, cfg, name="out"):
    out_dir = tmp_path / name
    squeak_set.QLVMSqueakTrainingSetBuilder(
        root_directories=[str(root) for root in roots],
        output_directory=str(out_dir),
        input_parameter_dict={"build_qlvm_squeak_training_set": cfg},
        message_output=lambda *_a, **_kw: None,
    ).build()
    return out_dir


def test_build_draw_crop_and_split(tmp_path, cohort):
    """The build draws n_total squeaks, crops each to its extent plus context,
    writes the train-qlvm columns and splits whole sessions."""
    out_dir = _build(tmp_path, cohort, _CFG)
    train, val = np.load(out_dir / "train_data.npz"), np.load(out_dir / "val_data.npz")
    assert train["spec_id"].size + val["spec_id"].size == 10
    assert set(train["session_id"].tolist()).isdisjoint(val["session_id"].tolist())
    assert train["spectrograms"].shape[1:] == (128, 128)
    assert not train["masks"].any()
    assert not bool(train["apply_mask"])
    assert np.array_equal(train["durations"], train["crop_last"] - train["crop_first"] + 1)
    assert (train["crop_first"] == 3).all()
    assert train["duration_bin"].tolist() == np.digitize(train["durations"], [12, 20]).tolist()
    assert set(train["session_type"].tolist()) <= {"MF", "FF"}
    meta = np.load(out_dir / "metadata.npz")
    assert str(meta["crop_window"]) == "full_length"
    assert int(meta["n_total_target"]) == 10
    assert json.loads(str(meta["curation_gates"]))[0] == ["squeak candidates", 31]


def test_build_crop_values_and_centring(tmp_path, cohort):
    """Every stored spectrogram is the per-crop min-max of its crop, centred in the
    128-frame frame (columns outside the crop are zero)."""
    out_dir = _build(tmp_path, cohort, {**_CFG, "full_dataset": True})
    full = np.load(out_dir / "full_data.npz")
    for position in range(full["spec_id"].size):
        width = int(full["durations"][position])
        columns = np.flatnonzero(full["spectrograms"][position].any(axis=0))
        assert columns.size == width - 1
        left = (128 - width) // 2
        assert full["spectrograms"][position][0, left + width - 1] == pytest.approx(1.0, abs=1e-5)
    assert full["spec_id"].size == 31


def test_build_uniform_session_cap(tmp_path, cohort):
    """With a per-session/stratum cap no session gives more than the cap to any
    stratum, and the result is the same for a reordered session list."""
    cfg = {**_CFG, "per_session_bin_cap": 1, "draw_mode": "uniform", "n_total": 6}
    out_a = _build(tmp_path, cohort, cfg, "a")
    out_b = _build(tmp_path, list(reversed(cohort)), cfg, "b")
    ids_a = np.concatenate([np.load(out_a / s)["spec_id"] for s in ("train_data.npz", "val_data.npz")])
    ids_b = np.concatenate([np.load(out_b / s)["spec_id"] for s in ("train_data.npz", "val_data.npz")])
    assert ids_a.tolist() == ids_b.tolist()
    both = [np.load(out_a / s) for s in ("train_data.npz", "val_data.npz")]
    pairs = [(sid, b) for split in both for sid, b in zip(split["session_id"].tolist(), split["duration_bin"].tolist(), strict=True)]
    assert len(pairs) == len(set(pairs))


def test_build_rejects_degenerate_split(tmp_path, cohort):
    """A validation split outside (0, 1) stops the build."""
    with pytest.raises(ValueError, match="validation_split"):
        _build(tmp_path, cohort, {**_CFG, "validation_split": 1.0})
