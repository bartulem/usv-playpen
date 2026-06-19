"""
@author: bartulem
Tests for processing/compute_usv_acoustic_features.

Synthesizes a per-session spectrogram H5 (a few specs with a known bright pixel)
plus a matching ``*_usv_summary.csv``, then checks the computed acoustic-feature
columns are merged into the right summary rows (joined on the per-USV index in
each spec_id), with USVs absent from the H5 left null.
"""

from __future__ import annotations

import h5py
import numpy as np
import polars as pls

from usv_playpen.processing.compute_usv_acoustic_features import (
    FEATURE_COLUMNS,
    USVAcousticFeatureExtractor,
    build_mask_region_masks,
    build_time_window_masks,
    compute_acoustic_features,
)

_CFG = {"low_energy_frac": 0.05, "high_energy_frac": 0.95}


def test_build_time_window_masks_clips_and_floors():
    """Region masks are 1 over [0, min(duration, T)) per row, with duration
    clamped to at least 1 and at most T."""
    region = build_time_window_masks(np.array([3, 200, 0]), n_time_bins=8)
    assert region.shape == (3, 8)
    assert region[0].tolist() == [1, 1, 1, 0, 0, 0, 0, 0]
    assert region[1].tolist() == [1] * 8          # 200 clamped to T=8
    assert region[2].tolist() == [1, 0, 0, 0, 0, 0, 0, 0]  # 0 floored to 1


def test_compute_acoustic_features_peak_frequency():
    """A single bright pixel inside the window drives peak_freq to that row's
    frequency and yields finite features."""
    n_f, n_t = 16, 16
    specs = np.zeros((1, n_f, n_t), dtype=np.float64)
    specs[0, 4, 2] = 1.0  # bright pixel at freq-row 4, time 2 (inside window)
    freq_axis = np.linspace(30000.0, 120000.0, n_f)
    feats = compute_acoustic_features(specs, np.array([8]), freq_axis, 0.05, 0.95)
    assert set(feats) == set(FEATURE_COLUMNS)
    assert feats["peak_freq_hz"][0] == freq_axis[4]
    assert feats["max_amplitude"][0] == 1.0
    assert np.all(np.isfinite(np.concatenate([feats[c] for c in FEATURE_COLUMNS])))


def test_build_mask_region_masks_unions_and_falls_back():
    """Each valid USV's region is the union of its segmentation rows; a valid USV
    with no mask rows falls back to its [0, duration) time-window."""
    n_f, n_t = 16, 16
    # Two valid USVs at summary rows 0 and 3; durations 8 and 4.
    usv_indices = np.array([0, 3], dtype=np.uint32)
    durations = np.array([8, 4], dtype=np.int64)
    # Row 0 has two masks; row 3 has none (-> time-window fallback).
    seg0 = np.zeros((n_f, n_t), dtype=bool)
    seg0[4:7, 0:4] = True
    seg1 = np.zeros((n_f, n_t), dtype=bool)
    seg1[10, 0:3] = True
    segmentations = np.stack([seg0, seg1], axis=0)
    spectrogram_index = np.array([0, 0], dtype=np.int64)

    region, fallback = build_mask_region_masks(
        durations=durations,
        usv_indices=usv_indices,
        segmentations=segmentations,
        spectrogram_index=spectrogram_index,
        n_freq=n_f,
        n_time=n_t,
    )
    assert region.shape == (2, n_f, n_t)
    assert region.dtype == np.float32
    assert fallback == 1
    # Row 0 == union of the two masks.
    assert np.array_equal(region[0].astype(bool), seg0 | seg1)
    # Row 3 fell back to the [0, 4) time-window over all frequencies.
    assert region[1, :, :4].all()
    assert not region[1, :, 4:].any()


def test_compute_acoustic_features_respects_region_mask():
    """With a 2D region mask, a bright pixel OUTSIDE the mask is ignored and the
    in-mask pixel drives peak_freq / max_amplitude (vs the time-window default,
    which would pick the brighter out-of-mask pixel)."""
    n_f, n_t = 16, 16
    specs = np.zeros((1, n_f, n_t), dtype=np.float64)
    specs[0, 4, 2] = 1.0    # dim pixel, inside the mask
    specs[0, 10, 8] = 2.0   # brighter pixel, OUTSIDE the mask
    freq_axis = np.linspace(30000.0, 120000.0, n_f)

    region = np.zeros((1, n_f, n_t), dtype=np.float32)
    region[0, 3:6, 0:4] = 1.0  # mask covers the dim pixel only

    masked = compute_acoustic_features(specs, np.array([16]), freq_axis, 0.05, 0.95, region_masks=region)
    assert masked["peak_freq_hz"][0] == freq_axis[4]
    assert masked["max_amplitude"][0] == 1.0

    # Time-window default sees the whole row and picks the brighter pixel.
    windowed = compute_acoustic_features(specs, np.array([16]), freq_axis, 0.05, 0.95)
    assert windowed["peak_freq_hz"][0] == freq_axis[10]
    assert windowed["max_amplitude"][0] == 2.0


def _write_h5(path, session_id, n_usv, valid_rows, n_f=16, n_t=16):
    """Consolidated layout, rows 1:1 with usv_summary: rows in ``valid_rows``
    carry a real spectrogram (duration > 0); the rest are all-zero placeholders
    (duration 0), exactly as ``generate_spectrograms`` writes them."""
    rng = np.random.default_rng(0)
    specs = np.zeros((n_usv, n_f, n_t), dtype=np.float32)
    durations = np.zeros(n_usv, dtype=np.int64)
    for r in valid_rows:
        specs[r] = rng.random((n_f, n_t)).astype(np.float32)
        durations[r] = n_t
    with h5py.File(path, "w") as f:
        f.create_dataset("frequency_bins", data=np.linspace(30000.0, 120000.0, n_f))
        session_group = f.create_group(f"spectrogram/{session_id}")
        session_group.create_dataset("spectrograms", data=specs)
        session_group.create_dataset("durations", data=durations)


def test_merge_features_into_summary(tmp_path, mocker):
    """Features merge into the summary by per-USV index; a USV missing from the
    H5 gets null features; pre-existing feature columns are replaced."""
    session_id = "20230119_155302"
    root = tmp_path / session_id
    (root / "audio" / "spectrograms").mkdir(parents=True)

    # 4 USVs in the summary; only indices 0, 1, 3 have specs (2 is skipped).
    pls.DataFrame({
        "usv_id": [f"{i:04d}" for i in range(4)],
        "start": [0.1, 0.3, 0.5, 0.7],
        "stop": [0.15, 0.35, 0.55, 0.75],
        "mean_amplitude": [9.0, 9.0, 9.0, 9.0],  # stale values to be replaced
    }).write_csv(root / "audio" / f"{session_id}_usv_summary.csv")

    _write_h5(root / "audio" / "spectrograms" / f"{session_id}_spectrograms.h5",
              session_id, n_usv=4, valid_rows=[0, 1, 3])

    mocker.patch("usv_playpen.processing.compute_usv_acoustic_features.smart_wait")
    USVAcousticFeatureExtractor(
        root_directory=str(root),
        input_parameter_dict={"compute_usv_acoustic_features": _CFG},
        message_output=lambda *_a, **_kw: None,
    ).merge_features_into_summary()

    df = pls.read_csv(root / "audio" / f"{session_id}_usv_summary.csv")
    # All feature columns present, original rows preserved (4), schema kept.
    assert set(FEATURE_COLUMNS).issubset(df.columns)
    assert df.height == 4
    # Rows 0,1,3 populated; row 2 (no spec) is null.
    assert df["mean_amplitude"][2] is None
    assert df["mean_amplitude"][0] is not None
    # Stale value was replaced, not duplicated into a suffixed column.
    assert "mean_amplitude_right" not in df.columns
    assert df["mean_amplitude"][0] != 9.0


def test_merge_features_uses_mask_group_when_present(tmp_path, mocker):
    """When the H5 carries a mask/<session> group, features are restricted to the
    masked region: a bright pixel outside the mask does not become peak_freq."""
    session_id = "20230119_155302"
    root = tmp_path / session_id
    (root / "audio" / "spectrograms").mkdir(parents=True)
    n_f = n_t = 16

    pls.DataFrame({
        "usv_id": ["0000"],
        "start": [0.1],
        "stop": [0.15],
    }).write_csv(root / "audio" / f"{session_id}_usv_summary.csv")

    # One valid USV: dim pixel inside the future mask, brighter pixel outside it.
    specs = np.zeros((1, n_f, n_t), dtype=np.float32)
    specs[0, 4, 2] = 1.0
    specs[0, 10, 8] = 2.0
    freq_axis = np.linspace(30000.0, 120000.0, n_f)
    seg = np.zeros((1, n_f, n_t), dtype=bool)
    seg[0, 3:6, 0:4] = True  # covers the dim in-mask pixel only

    h5_path = root / "audio" / "spectrograms" / f"{session_id}_spectrograms.h5"
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("frequency_bins", data=freq_axis)
        grp = f.create_group(f"spectrogram/{session_id}")
        grp.create_dataset("spectrograms", data=specs)
        grp.create_dataset("durations", data=np.array([n_t], dtype=np.int64))
        mask_grp = f.create_group(f"mask/{session_id}")
        mask_grp.create_dataset("segmentations", data=seg)
        mask_grp.create_dataset("spectrogram_index", data=np.array([0], dtype=np.int64))

    mocker.patch("usv_playpen.processing.compute_usv_acoustic_features.smart_wait")
    USVAcousticFeatureExtractor(
        root_directory=str(root),
        input_parameter_dict={"compute_usv_acoustic_features": _CFG},
        message_output=lambda *_a, **_kw: None,
    ).merge_features_into_summary()

    df = pls.read_csv(root / "audio" / f"{session_id}_usv_summary.csv")
    # The masked region keeps the in-mask pixel (row 4), not the brighter row-10 pixel.
    assert df["peak_freq_hz"][0] == freq_axis[4]
    assert df["max_amplitude"][0] == 1.0
