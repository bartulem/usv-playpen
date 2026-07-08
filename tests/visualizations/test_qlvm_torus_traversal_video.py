"""
@author: bartulem
Tests for visualizations/qlvm_torus_traversal_video.

Covers the torus embedding helper, pooling per-USV latent coords from the
consolidated H5 (`spectrogram/<key>/qlvm_dim`), the phase-script builder, a small
end-to-end three-part render to a temporary GIF (Pillow writer, no FFmpeg) that
sources coords + spectrograms from the H5, and CLI routing.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest
from click.testing import CliRunner

from usv_playpen.visualizations.qlvm_torus_traversal_video import (
    QLVMTorusTraversalVideo,
    build_phases,
    pool_latents_from_h5,
    qlvm_torus_traversal_video_cli,
    torus_forward,
)


def test_torus_forward_shape_and_values():
    """Embedding doubles the last dim and equals [cos, sin] of 2*pi*coords."""
    coords = np.array([[0.0, 0.25]])
    emb = torus_forward(coords)
    assert emb.shape == (1, 4)
    assert np.allclose(emb[0], [np.cos(0), np.cos(np.pi / 2), np.sin(0), np.sin(np.pi / 2)])


def test_pool_latents_from_h5(tmp_path):
    """Pools qlvm_dim across sessions in order, drops NaN rows, aligns the index."""
    h5_path = tmp_path / "store.h5"
    with h5py.File(h5_path, "w") as h5:
        h5.create_dataset("spectrogram/20230101_000000/qlvm_dim",
                          data=np.array([[0.1, 0.2], [0.3, 0.4]]))
        h5.create_dataset("spectrogram/20230102_000000/qlvm_dim",
                          data=np.array([[0.5, 0.6], [np.nan, np.nan]]))
    with h5py.File(h5_path, "r") as h5:
        coords, index = pool_latents_from_h5(h5)
    assert coords.shape == (3, 2)
    assert index == [("20230101_000000", 0), ("20230101_000000", 1), ("20230102_000000", 0)]


def test_build_phases_structure():
    """peaks_only -> Part 1 only; full -> 3 parts with peak + boundary traversals."""
    rng = np.random.default_rng(0)
    centers = np.array([[0.2, 0.2], [0.5, 0.5], [0.8, 0.8]])
    peaks = build_phases(centers, 3, True, 1, 2, 4, 4, 0.01, 0.1, rng)
    assert [p['name'] for p in peaks] == ['title_card', 'cluster', 'cluster', 'cluster']

    rng = np.random.default_rng(0)
    full = build_phases(centers, 3, False, 1, 2, 4, 4, 0.01, 0.1, rng)
    n_traversals = sum(1 for p in full if p['name'] == 'traversal')
    n_cards = sum(1 for p in full if p['name'] == 'title_card')
    # min(5, K*(K-1)) peak pairs + 5 boundary walks; 3 title cards.
    assert n_traversals == min(5, 3 * 2) + 5
    assert n_cards == 3


def _tiny_cfg(spectrograms_dir, peaks_only=False):
    """A small render input dict (the qlvm block + the shared_resources block)
    that exercises all phases quickly at low dpi. The QLVM arrays + the
    consolidated store are resolved from ``spectrograms_dir``."""
    return {
        "shared_resources": {
            "spectrograms_dir": str(spectrograms_dir),
        },
        "figures": {
            "cmap": "inferno",
        },
        "qlvm_torus_traversal_video": {
            "clustering": "coarse",
            "fps": 4, "dpi": 30, "m": 4,
            "cluster_hold_frames": 2, "peak_traverse_frames": 4,
            "boundary_traverse_frames": 4, "title_card_frames": 1,
            "samples_per_trace": 3, "peak_jitter_sigma": 0.01,
            "boundary_curve_amplitude": 0.1, "boundary_positions_per_walk": 3,
            "boundary_neighbors": 3, "seed": 0,
            "peaks_only": peaks_only, "spec_cache_size": 64, "apply_mask": True,
            "accent_color": "#00FFFF",
        },
    }


def _write_inputs(tmp_path, n=12, res=8, n_f=16, n_t=16):
    """Build a shared spectrograms dir: <dir>/qlvm/arrays_{coarse,fine}.npz
    (heatmap/ws/centers) + <dir>/spectrograms_<key>.h5 carrying per-session
    spectrograms AND qlvm_dim coords. Returns str(<dir>)."""
    rng = np.random.default_rng(0)
    session_key = "20250907_190610"
    spec_dir = tmp_path / "spectrograms"
    (spec_dir / "qlvm").mkdir(parents=True, exist_ok=True)
    for tag in ("coarse", "fine"):
        np.savez(
            spec_dir / "qlvm" / f"arrays_{tag}.npz",
            heatmap=rng.random((res, res)).astype(np.float32),
            ws_labels_periodic=rng.integers(0, 3, size=(res, res)).astype(np.int16),
            centers=np.array([[0.25, 0.25], [0.7, 0.7]], dtype=np.float32),
        )
    h5_path = spec_dir / f"spectrograms_{session_key}.h5"
    with h5py.File(h5_path, "w") as h5:
        h5.create_dataset(f"spectrogram/{session_key}/spectrograms",
                          data=rng.random((n, n_f, n_t)).astype(np.float32))
        h5.create_dataset(f"spectrogram/{session_key}/durations",
                          data=rng.integers(4, n_t, size=n).astype(np.int64))
        h5.create_dataset(f"spectrogram/{session_key}/qlvm_dim",
                          data=rng.random((n, 2)).astype(np.float64))
        # One SAM2 mask per spectrogram, so the apply_mask path is exercised.
        h5.create_dataset(f"mask/{session_key}/segmentations",
                          data=(rng.random((n, n_f, n_t)) > 0.5))
        h5.create_dataset(f"mask/{session_key}/spectrogram_index",
                          data=np.arange(n, dtype=np.int64))
    return str(spec_dir)


def test_make_video_writes_gif(tmp_path):
    """End-to-end three-part render to a small GIF (coords + specs from H5)."""
    spec_dir = _write_inputs(tmp_path)
    out = tmp_path / "traversal.gif"
    QLVMTorusTraversalVideo(
        output_path=str(out),
        input_parameter_dict=_tiny_cfg(spec_dir),
        message_output=lambda *_a, **_kw: None,
    ).make_video()
    assert out.is_file()
    assert out.stat().st_size > 0


def test_make_video_errors_without_qlvm_dim(tmp_path):
    """If the H5 has no qlvm_dim, render fails with a clear, actionable error."""
    rng = np.random.default_rng(1)
    spec_dir = tmp_path / "spectrograms"
    (spec_dir / "qlvm").mkdir(parents=True, exist_ok=True)
    for tag in ("coarse", "fine"):
        np.savez(spec_dir / "qlvm" / f"arrays_{tag}.npz",
                 heatmap=rng.random((8, 8)).astype(np.float32),
                 ws_labels_periodic=rng.integers(0, 3, size=(8, 8)).astype(np.int16),
                 centers=np.array([[0.25, 0.25], [0.7, 0.7]], dtype=np.float32))
    with h5py.File(spec_dir / "spectrograms_20250907_190610.h5", "w") as h5:
        h5.create_dataset("spectrogram/20250907_190610/spectrograms",
                          data=rng.random((5, 16, 16)).astype(np.float32))
    with pytest.raises(ValueError, match="qlvm_dim"):
        QLVMTorusTraversalVideo(
            output_path=str(tmp_path / "x.gif"),
            input_parameter_dict=_tiny_cfg(spec_dir),
            message_output=lambda *_a, **_kw: None,
        ).make_video()


@pytest.fixture
def runner():
    """Provides a CliRunner instance for invoking commands."""
    return CliRunner()


def test_qlvm_torus_traversal_video_cli_routes(runner, mocker, tmp_path):
    """The command resolves settings and calls QLVMTorusTraversalVideo once."""
    mock_cls = mocker.patch(
        "usv_playpen.visualizations.qlvm_torus_traversal_video.QLVMTorusTraversalVideo"
    )
    mocker.patch(
        "usv_playpen.visualizations.qlvm_torus_traversal_video.modify_settings_json_for_cli",
        return_value={"qlvm_torus_traversal_video": {}},
    )
    result = runner.invoke(qlvm_torus_traversal_video_cli, ["--output-path", str(tmp_path / "v.mp4")])
    assert result.exit_code == 0, result.output
    mock_cls.assert_called_once()
    mock_cls.return_value.make_video.assert_called_once()
