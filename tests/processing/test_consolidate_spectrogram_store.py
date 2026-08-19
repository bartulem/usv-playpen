"""
@author: bartulem
Tests for processing/consolidate_spectrogram_store.SpectrogramStoreConsolidator.

Coverage: two synthetic sessions consolidate into one store with shared
frequency_bins, per-session spectrogram/mask groups, qlvm_dim injection from
the summary's qlvm1/qlvm2 columns, and correct file-level attrs; a session
with a mismatching frequency axis fails loud; a session whose spectrogram
row count disagrees with its summary fails loud; a session without the
per-session H5 is skipped with a message.
"""

from __future__ import annotations

import h5py
import numpy as np
import polars as pls
import pytest

from usv_playpen.processing.consolidate_spectrogram_store import SpectrogramStoreConsolidator


def _build_session(base_dir, session_id, n_usv=3, freq0=30000.0, with_mask=True, with_qlvm=True):
    """Create one session dir with a per-session spectrograms H5 (+ optional
    mask group) and a usv_summary CSV (+ optional qlvm1/qlvm2), returning
    the session root path."""
    root = base_dir / session_id
    spec_dir = root / "audio" / "spectrograms"
    spec_dir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    with h5py.File(spec_dir / f"{session_id}_spectrograms.h5", "w") as f:
        f.create_dataset("frequency_bins", data=np.linspace(freq0, 120000.0, 16))
        grp = f.create_group(f"spectrogram/{session_id}")
        grp.create_dataset("spectrograms", data=rng.random((n_usv, 16, 8)).astype(np.float32))
        grp.create_dataset("durations", data=np.full(n_usv, 5, dtype=np.int64))
        if with_mask:
            mgrp = f.create_group(f"mask/{session_id}")
            mgrp.create_dataset("segmentations", data=np.ones((n_usv, 16, 8), dtype=bool))
            mgrp.create_dataset("spectrogram_index", data=np.arange(n_usv, dtype=np.int64))
    rows = {
        "usv_id": [f"{i:04d}" for i in range(n_usv)],
        "start": [0.1 + 0.2 * i for i in range(n_usv)],
        "stop": [0.15 + 0.2 * i for i in range(n_usv)],
    }
    if with_qlvm:
        rows["qlvm1"] = rng.standard_normal(n_usv).round(6).tolist()
        rows["qlvm2"] = rng.standard_normal(n_usv).round(6).tolist()
    pls.DataFrame(rows).write_csv(root / "audio" / f"{session_id}_usv_summary.csv")
    return root


def _consolidate(tmp_path, roots, mocker):
    mocker.patch("usv_playpen.processing.consolidate_spectrogram_store.smart_wait")
    out_dir = tmp_path / "store"
    out_dir.mkdir(exist_ok=True)
    SpectrogramStoreConsolidator(
        root_directories=[str(r) for r in roots],
        input_parameter_dict={"spectrograms_root": str(out_dir)},
        message_output=lambda *_a, **_kw: None,
    ).consolidate_spectrogram_store()
    stores = sorted(out_dir.glob("spectrograms_*.h5"))
    return stores


def test_consolidates_two_sessions_with_qlvm_injection(tmp_path, mocker):
    """Two sessions land in one store: shared axis, both groups, masks,
    injected qlvm_dim, and correct n_sessions/n_vocalizations attrs."""
    roots = [
        _build_session(tmp_path / "data", "20260101_120000", n_usv=3),
        _build_session(tmp_path / "data", "20260102_120000", n_usv=2),
    ]
    stores = _consolidate(tmp_path, roots, mocker)
    assert len(stores) == 1
    with h5py.File(stores[0], "r") as f:
        assert f.attrs["n_sessions"] == 2
        assert f.attrs["n_vocalizations"] == 5
        assert f["frequency_bins"].shape == (16,)
        for sid, n in (("20260101_120000", 3), ("20260102_120000", 2)):
            assert f[f"spectrogram/{sid}/spectrograms"].shape == (n, 16, 8)
            assert f[f"spectrogram/{sid}/qlvm_dim"].shape == (n, 2)
            assert f[f"mask/{sid}/segmentations"].shape == (n, 16, 8)


def test_session_without_qlvm_copied_without_dataset(tmp_path, mocker):
    """A session lacking qlvm1/qlvm2 (e.g. playback) is copied, just without
    the qlvm_dim dataset."""
    roots = [_build_session(tmp_path / "data", "20260103_120000", with_qlvm=False)]
    stores = _consolidate(tmp_path, roots, mocker)
    with h5py.File(stores[0], "r") as f:
        assert "qlvm_dim" not in f["spectrogram/20260103_120000"]


def test_frequency_axis_mismatch_fails_loud(tmp_path, mocker):
    """Sessions must share one frequency grid; a divergent axis raises."""
    roots = [
        _build_session(tmp_path / "data", "20260104_120000"),
        _build_session(tmp_path / "data", "20260105_120000", freq0=25000.0),
    ]
    with pytest.raises(ValueError, match="frequency_bins"):
        _consolidate(tmp_path, roots, mocker)


def test_rowcount_mismatch_fails_loud(tmp_path, mocker):
    """A session whose summary row count disagrees with its spectrogram rows
    must raise rather than silently desync the store."""
    root = _build_session(tmp_path / "data", "20260106_120000", n_usv=3)
    summary = root / "audio" / "20260106_120000_usv_summary.csv"
    df = pls.read_csv(summary).head(2)
    df.write_csv(summary)
    with pytest.raises(ValueError, match="inconsistent"):
        _consolidate(tmp_path, [root], mocker)


def test_missing_session_h5_is_skipped(tmp_path, mocker):
    """A session without the per-session H5 is skipped; the rest consolidate."""
    good = _build_session(tmp_path / "data", "20260107_120000", n_usv=2)
    bare = tmp_path / "data" / "20260108_120000"
    (bare / "audio").mkdir(parents=True)
    stores = _consolidate(tmp_path, [good, bare], mocker)
    with h5py.File(stores[0], "r") as f:
        assert f.attrs["n_sessions"] == 1
        assert "20260108_120000" not in f["spectrogram"]
