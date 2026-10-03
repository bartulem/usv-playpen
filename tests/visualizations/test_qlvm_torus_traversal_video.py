"""
@author: bartulem
Tests for visualizations/qlvm_torus_traversal_video.

Covers the torus embedding helper, pooling per-USV latent coords from the
consolidated H5 (`qlvm/<key>/<map>`), the phase-script builder, the store
provenance check against the cell the QLVM category bundle is defined on, a small
end-to-end three-part render to a temporary GIF (Pillow writer, no FFmpeg) that
sources coords + spectrograms from the H5 and the landscape from the category
bundle, and CLI routing.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest
from click.testing import CliRunner

from usv_playpen.visualizations.qlvm_torus_traversal_video import (
    QLVMTorusTraversalVideo,
    build_phases,
    check_store_map_provenance,
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
    """Pools one map's qlvm/<session>/<map> coords across sessions in order, drops NaN
    rows, aligns the index, and skips sessions without that map."""
    h5_path = tmp_path / "store.h5"
    with h5py.File(h5_path, "w") as h5:
        h5.create_dataset("qlvm/20230101_000000/qlvm", data=np.array([[0.1, 0.2], [0.3, 0.4]]))
        h5.create_dataset("qlvm/20230101_000000/qlvm_dur", data=np.array([[0.9, 0.8], [np.nan, np.nan]]))
        h5.create_dataset("qlvm/20230102_000000/qlvm", data=np.array([[0.5, 0.6], [np.nan, np.nan]]))
    with h5py.File(h5_path, "r") as h5:
        coords, index = pool_latents_from_h5(h5, "qlvm")
        dur_coords, dur_index = pool_latents_from_h5(h5, "qlvm_dur")
    assert coords.shape == (3, 2)
    assert index == [("20230101_000000", 0), ("20230101_000000", 1), ("20230102_000000", 0)]
    np.testing.assert_array_equal(dur_coords, [[0.9, 0.8]])
    assert dur_index == [("20230101_000000", 0)]


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


def _tiny_cfg(spectrograms_dir, peaks_only=False, qlvm_map="qlvm"):
    """A small render input dict (the qlvm block + the shared_resources block)
    that exercises all phases quickly at low dpi. The consolidated store is
    resolved from ``spectrograms_dir``; the landscape is the category bundle."""
    return {
        "shared_resources": {
            "spectrograms_dir": str(spectrograms_dir),
            "qlvm_map": qlvm_map,
        },
        "figures": {
            "sequential_cmap": "inferno",
        },
        "qlvm_torus_traversal_video": {
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


_PRODUCTION_PACKAGE = "/mnt/falkner/Bartul/PC_transfer/qlvm_time_stretch/masked_clean"


def _write_inputs(tmp_path, n=12, n_f=16, n_t=16, package_root=_PRODUCTION_PACKAGE, cell="cell/masked"):
    """Build a shared spectrograms dir holding <dir>/spectrograms_<key>.h5 with
    per-session spectrograms, qlvm/<key>/qlvm coords and the qlvm_models/qlvm
    provenance attrs (package_root / cell, the production regular cell by default).
    Returns str(<dir>)."""
    rng = np.random.default_rng(0)
    session_key = "20250907_190610"
    spec_dir = tmp_path / "spectrograms"
    spec_dir.mkdir(parents=True, exist_ok=True)
    h5_path = spec_dir / f"spectrograms_{session_key}.h5"
    with h5py.File(h5_path, "w") as h5:
        h5.create_dataset(f"spectrogram/{session_key}/spectrograms",
                          data=rng.random((n, n_f, n_t)).astype(np.float32))
        h5.create_dataset(f"spectrogram/{session_key}/durations",
                          data=rng.integers(4, n_t, size=n).astype(np.int64))
        h5.create_dataset(f"qlvm/{session_key}/qlvm",
                          data=rng.random((n, 2)).astype(np.float64))
        # One SAM2 mask per spectrogram, so the apply_mask path is exercised.
        h5.create_dataset(f"mask/{session_key}/segmentations",
                          data=(rng.random((n, n_f, n_t)) > 0.5))
        h5.create_dataset(f"mask/{session_key}/spectrogram_index",
                          data=np.arange(n, dtype=np.int64))
        model_group = h5.create_group("qlvm_models/qlvm")
        model_group.attrs["package_root"] = package_root
        model_group.attrs["cell"] = cell
    return str(spec_dir)


def test_make_video_writes_gif(tmp_path, qlvm_category_bundle):
    """End-to-end three-part render to a small GIF (coords + specs from H5, the
    density / boundaries / category peaks from the category bundle)."""
    spec_dir = _write_inputs(tmp_path)
    out = tmp_path / "traversal.gif"
    messages = []
    QLVMTorusTraversalVideo(
        output_path=str(out),
        input_parameter_dict=_tiny_cfg(spec_dir),
        message_output=messages.append,
    ).make_video()
    assert out.is_file()
    assert out.stat().st_size > 0
    assert any("4 categories" in message and str(qlvm_category_bundle) in message for message in messages)


def test_make_video_refuses_a_conditional_map(tmp_path, qlvm_category_bundle):
    """The bundle's regions are defined on the regular map only, so a conditional
    shared qlvm_map is refused before anything is read, naming the map."""
    spec_dir = _write_inputs(tmp_path)
    with pytest.raises(ValueError, match="'qlvm_dur'"):
        QLVMTorusTraversalVideo(
            output_path=str(tmp_path / "x.gif"),
            input_parameter_dict=_tiny_cfg(spec_dir, qlvm_map="qlvm_dur"),
            message_output=lambda *_a, **_kw: None,
        ).make_video()


def test_make_video_refuses_a_store_of_another_cell(tmp_path, qlvm_category_bundle):
    """A store whose qlvm coordinates come from another cell (e.g. the v3 archive)
    places the calls on another torus than the bundle partitions: refused."""
    spec_dir = _write_inputs(
        tmp_path,
        package_root="/mnt/falkner/Dexter/vocal_beh/models/qlvm_models/qlvm_models_latest/v3",
        cell="phase6_USVs_unmasked_floor/natural_5strata_N29000_unmasked_floor",
    )
    with pytest.raises(ValueError, match="mislabel"):
        QLVMTorusTraversalVideo(
            output_path=str(tmp_path / "x.gif"),
            input_parameter_dict=_tiny_cfg(spec_dir),
            message_output=lambda *_a, **_kw: None,
        ).make_video()


def test_check_store_map_provenance(tmp_path):
    """The store must record the production regular cell for the map; a store without
    provenance cannot be shown to match and is refused."""
    h5_path = tmp_path / "store.h5"
    with h5py.File(h5_path, "w") as h5:
        group = h5.create_group("qlvm_models/qlvm")
        group.attrs["package_root"] = "/Volumes/falkner/Bartul/PC_transfer/qlvm_time_stretch/masked_clean"
        group.attrs["cell"] = "cell/masked"
    with h5py.File(h5_path, "r") as h5:
        assert check_store_map_provenance(h5, "qlvm") == "masked_clean/cell/masked"
    with h5py.File(tmp_path / "bare.h5", "w") as h5:
        h5.create_dataset("qlvm/s/qlvm", data=np.zeros((2, 2)))
    with h5py.File(tmp_path / "bare.h5", "r") as h5, pytest.raises(ValueError, match="no qlvm_models/qlvm"):
        check_store_map_provenance(h5, "qlvm")


def test_make_video_errors_without_map_coordinates(tmp_path, qlvm_category_bundle):
    """If the H5 has no coordinates of the regular map, render fails with a clear,
    actionable error naming the map."""
    rng = np.random.default_rng(1)
    spec_dir = tmp_path / "spectrograms"
    spec_dir.mkdir()
    with h5py.File(spec_dir / "spectrograms_20250907_190610.h5", "w") as h5:
        h5.create_dataset("spectrogram/20250907_190610/spectrograms",
                          data=rng.random((5, 16, 16)).astype(np.float32))
        h5.create_dataset("qlvm/20250907_190610/qlvm_dur", data=rng.random((5, 2)))
        model_group = h5.create_group("qlvm_models/qlvm")
        model_group.attrs["package_root"] = _PRODUCTION_PACKAGE
        model_group.attrs["cell"] = "cell/masked"
    with pytest.raises(ValueError, match="qlvm/<session>/qlvm"):
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
