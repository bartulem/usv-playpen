"""
@author: bartulem
Tests for visualizations/qlvm_torus_traversal_video.

Covers the torus embedding + traversal-path helpers, a small end-to-end render
to a temporary GIF (Pillow writer, no FFmpeg needed), and CLI routing.
"""

from __future__ import annotations

import numpy as np
import pytest
from click.testing import CliRunner

from usv_playpen.visualizations.qlvm_torus_traversal_video import (
    QLVMTorusTraversalVideo,
    build_traversal_path,
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


def _write_inputs(tmp_path, n=12, res=8, n_f=16, n_t=16):
    rng = np.random.default_rng(0)
    arrays = tmp_path / "arrays.npz"
    np.savez(
        arrays,
        latent_coords=rng.random((n, 2)).astype(np.float32),
        heatmap=rng.random((res, res)).astype(np.float32),
        ws_labels_periodic=rng.integers(0, 3, size=(res, res)).astype(np.int16),
        centers=np.array([[0.2, 0.2], [0.7, 0.7]], dtype=np.float32),
    )
    specs = tmp_path / "full_data.npz"
    np.savez(specs, spectrograms=rng.random((n, n_f, n_t)).astype(np.float32))
    return str(arrays), str(specs)


def test_make_video_writes_gif(tmp_path):
    """End-to-end render to a small GIF produces a non-empty file."""
    arrays_path, specs_path = _write_inputs(tmp_path)
    out = tmp_path / "traversal.gif"
    cfg = {
        "arrays_npz_path": arrays_path,
        "specs_npz_path": specs_path,
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
