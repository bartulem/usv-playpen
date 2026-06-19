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


def _write_h5(path, session_id, usv_indices, n_f=16, n_t=16):
    rng = np.random.default_rng(0)
    n = len(usv_indices)
    specs = rng.random((n, n_f, n_t)).astype(np.float32)
    durations = np.full(n, n_t, dtype=np.int64)
    freq_bins = np.linspace(30000.0, 120000.0, n_f)
    with h5py.File(path, "w") as f:
        f.attrs["session_id"] = session_id
        f.create_dataset("spectrograms", data=specs)
        f.create_dataset("durations", data=durations)
        f.create_dataset("freq_bins", data=freq_bins)
        f.create_dataset("spectrogram_ids", data=np.asarray(usv_indices, dtype=np.int64))


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
              session_id, usv_indices=[0, 1, 3])

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
