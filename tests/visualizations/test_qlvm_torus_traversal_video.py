"""
@author: bartulem
Tests for visualizations/qlvm_torus_traversal_video.

Covers the torus embedding + traversal-path helpers, pooling per-USV latent
coords from the consolidated H5 (`spectrogram/<key>/qlvm_dim`), a small
end-to-end render to a temporary GIF (Pillow writer, no FFmpeg needed) that
sources both coords and spectrograms from the H5, and CLI routing.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest
from click.testing import CliRunner

from usv_playpen.visualizations.qlvm_torus_traversal_video import (
    QLVMTorusTraversalVideo,
    build_traversal_path,
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


def test_build_traversal_path_shape_and_on_torus():
    """The path loops over all centers, has K*frames points, and stays in [0,1)."""
    centers = np.array([[0.1, 0.1], [0.8, 0.8], [0.2, 0.9]])
    path = build_traversal_path(centers, frames_per_segment=10, curvature=0.1)
    assert path.shape == (30, 2)
    assert path.min() >= 0.0
    assert path.max() < 1.0


def test_pool_latents_from_h5(tmp_path):
    """Pools qlvm_dim across sessions in order, drops NaN rows, aligns the index."""
    h5_path = tmp_path / "store.h5"
    with h5py.File(h5_path, "w") as h5:
        h5.create_dataset("spectrogram/20230101_000000/qlvm_dim",
                          data=np.array([[0.1, 0.2], [0.3, 0.4]]))
        # one valid row + one NaN row (should be dropped)
        h5.create_dataset("spectrogram/20230102_000000/qlvm_dim",
                          data=np.array([[0.5, 0.6], [np.nan, np.nan]]))
    with h5py.File(h5_path, "r") as h5:
        coords, index = pool_latents_from_h5(h5)
    assert coords.shape == (3, 2)
    assert index == [("20230101_000000", 0), ("20230101_000000", 1), ("20230102_000000", 0)]


def _write_inputs(tmp_path, n=12, res=8, n_f=16, n_t=16):
    """Write a synthetic arrays_coarse.npz (heatmap/ws/centers) and a consolidated
    H5 carrying per-session spectrograms AND qlvm_dim coords."""
    rng = np.random.default_rng(0)
    session_key = "20250907_190610"

    arrays = tmp_path / "arrays_coarse.npz"
    np.savez(
        arrays,
        latent_coords=rng.random((n, 2)).astype(np.float32),  # unused by the video now
        heatmap=rng.random((res, res)).astype(np.float32),
        ws_labels_periodic=rng.integers(0, 3, size=(res, res)).astype(np.int16),
        centers=np.array([[0.2, 0.2], [0.7, 0.7]], dtype=np.float32),
    )

    h5_path = tmp_path / "consolidated.h5"
    with h5py.File(h5_path, "w") as h5:
        h5.create_dataset(f"spectrogram/{session_key}/spectrograms",
                          data=rng.random((n, n_f, n_t)).astype(np.float32))
        h5.create_dataset(f"spectrogram/{session_key}/qlvm_dim",
                          data=rng.random((n, 2)).astype(np.float64))
    return str(arrays), str(h5_path)


def test_make_video_writes_gif(tmp_path):
    """End-to-end render to a small GIF (coords + specs from the H5) is non-empty."""
    arrays_path, h5_path = _write_inputs(tmp_path)
    out = tmp_path / "traversal.gif"
    cfg = {
        "clustering": "coarse",
        "arrays_npz_path_coarse": arrays_path,
        "arrays_npz_path_fine": arrays_path,
        "consolidated_h5_path": h5_path,
        "frames_per_segment": 4,   # 2 centers -> 8 frames
        "fps": 4,
        "dpi": 50,
        "path_curvature": 0.1,
        "trail_length": 20,
    }
    QLVMTorusTraversalVideo(
        output_path=str(out),
        input_parameter_dict={"qlvm_torus_traversal_video": cfg},
        message_output=lambda *_a, **_kw: None,
    ).make_video()
    assert out.is_file()
    assert out.stat().st_size > 0


def test_make_video_errors_without_qlvm_dim(tmp_path):
    """If the H5 has no qlvm_dim, render fails with a clear, actionable error."""
    rng = np.random.default_rng(1)
    arrays = tmp_path / "arrays_coarse.npz"
    np.savez(arrays, heatmap=rng.random((8, 8)).astype(np.float32),
             ws_labels_periodic=rng.integers(0, 3, size=(8, 8)).astype(np.int16),
             centers=np.array([[0.2, 0.2], [0.7, 0.7]], dtype=np.float32))
    h5_path = tmp_path / "consolidated.h5"
    with h5py.File(h5_path, "w") as h5:
        h5.create_dataset("spectrogram/20250907_190610/spectrograms",
                          data=rng.random((5, 16, 16)).astype(np.float32))
    cfg = {
        "clustering": "coarse", "arrays_npz_path_coarse": str(arrays),
        "arrays_npz_path_fine": str(arrays), "consolidated_h5_path": str(h5_path),
        "frames_per_segment": 4, "fps": 4, "dpi": 50, "path_curvature": 0.1, "trail_length": 20,
    }
    with pytest.raises(ValueError, match="qlvm_dim"):
        QLVMTorusTraversalVideo(
            output_path=str(tmp_path / "x.gif"),
            input_parameter_dict={"qlvm_torus_traversal_video": cfg},
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
