"""
@author: bartulem
Tests for visualizations/qlvm_torus_traversal_video.

Covers the torus embedding + traversal-path helpers, the flat-index ->
(session, row) provenance map, a small end-to-end render to a temporary GIF
(Pillow writer, no FFmpeg needed) that sources spectrograms from a synthetic
consolidated H5, and CLI routing.
"""

from __future__ import annotations

import pickle

import h5py
import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from usv_playpen.visualizations.qlvm_torus_traversal_video import (
    QLVMTorusTraversalVideo,
    build_flat_to_session_row,
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


def test_build_flat_to_session_row(tmp_path):
    """Two-session synthetic provenance -> correct (h5_key, row) flat map.

    Session names ``<date>_<time>_<suffix>`` map to the bare ``<date>_<time>``
    H5 key, dict order is preserved, and ``original_index`` becomes the row."""
    provenance = {
        "20250907_190610_Male-Female_avg": pd.DataFrame({"original_index": [0, 1, 2]}),
        "20241107_114630_ephys_avg": pd.DataFrame({"original_index": [0, 1]}),
    }
    pkl_path = tmp_path / "provenance.pkl"
    with open(pkl_path, "wb") as pkl_file:
        pickle.dump(provenance, pkl_file)

    flat_map = build_flat_to_session_row(str(pkl_path))

    assert flat_map == [
        ("20250907_190610", 0),
        ("20250907_190610", 1),
        ("20250907_190610", 2),
        ("20241107_114630", 0),
        ("20241107_114630", 1),
    ]
    assert len(flat_map) == sum(len(df) for df in provenance.values())


def _write_inputs(tmp_path, n=12, res=8, n_f=16, n_t=16):
    """Write a synthetic arrays.npz, provenance pkl (one session), and a
    consolidated H5 whose per-session spectrograms align with latent_coords."""
    rng = np.random.default_rng(0)
    session_key = "20250907_190610"

    arrays = tmp_path / "arrays_coarse.npz"
    np.savez(
        arrays,
        latent_coords=rng.random((n, 2)).astype(np.float32),
        heatmap=rng.random((res, res)).astype(np.float32),
        ws_labels_periodic=rng.integers(0, 3, size=(res, res)).astype(np.int16),
        centers=np.array([[0.2, 0.2], [0.7, 0.7]], dtype=np.float32),
    )

    # Provenance pkl: a single session with original_index 0..n-1 (so the flat
    # map is in latent_coords order).
    pkl = tmp_path / "provenance.pkl"
    with open(pkl, "wb") as pkl_file:
        pickle.dump({f"{session_key}_test_avg": pd.DataFrame({"original_index": np.arange(n)})}, pkl_file)

    # Consolidated H5: spectrogram/<key>/spectrograms (n, F, T).
    h5_path = tmp_path / "consolidated.h5"
    with h5py.File(h5_path, "w") as h5:
        h5.create_dataset(f"spectrogram/{session_key}/spectrograms",
                          data=rng.random((n, n_f, n_t)).astype(np.float32))
    return str(arrays), str(pkl), str(h5_path)


def test_make_video_writes_gif(tmp_path):
    """End-to-end render to a small GIF (specs pulled from the H5) is non-empty."""
    arrays_path, pkl_path, h5_path = _write_inputs(tmp_path)
    out = tmp_path / "traversal.gif"
    cfg = {
        "clustering": "coarse",
        "arrays_npz_path_coarse": arrays_path,
        "arrays_npz_path_fine": arrays_path,
        "provenance_pkl_path": pkl_path,
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
