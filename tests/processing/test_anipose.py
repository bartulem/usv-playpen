"""
@author: bartulem
Mock-based tests for ConvertTo3D in anipose_operations.py.

The class wraps four orchestration methods around external tools
(sleap-convert, sleap_anipose.draw_board / calibrate / triangulate); we
substitute every external call so the orchestration itself can be
exercised against tmp_path-based session layouts.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import cv2
import h5py
import numpy as np
import pytest
import sleap_anipose

import usv_playpen
from usv_playpen.processing.anipose_operations import ConvertTo3D, find_mouse_names

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def processing_settings():
    """Loads processing_settings.json from the package once per test."""
    package_dir = Path(usv_playpen.__file__).parent
    with (package_dir / '_parameter_settings' / 'processing_settings.json').open('r') as f:
        return json.load(f)


@pytest.fixture
def session_with_video_dir(tmp_path):
    """Creates `<tmp_path>/video/<sessdate>/` (no underscore in subdir name);
    that's the directory ConvertTo3D's __init__ scans for to set
    session_root_joint_date_dir."""
    sess_dir = tmp_path / "video" / "20260101120000"
    sess_dir.mkdir(parents=True)
    return tmp_path, sess_dir


# ---------------------------------------------------------------------------
# ConvertTo3D.__init__
# ---------------------------------------------------------------------------


def test_convert_to_3d_init_resolves_session_root_joint_date_dir(
    processing_settings, session_with_video_dir
):
    """Init scans <root>/video/ for the first dir whose name has no "_"; this
    becomes session_root_joint_date_dir / session_root_name. A camera-named
    dir like "21372315" qualifies; "session_21372315" (with underscore) does
    not — that's the convention enforced here."""
    root, sess_dir = session_with_video_dir
    converter = ConvertTo3D(
        root_directory=str(root),
        input_parameter_dict=processing_settings,
        message_output=lambda *_a, **_kw: None,
    )
    assert converter.session_root_joint_date_dir == sess_dir
    assert converter.session_root_name == sess_dir.name
    # Settings reduced to the right sub-block:
    assert "conduct_anipose_calibration" in converter.input_parameter_dict


def test_convert_to_3d_init_raises_when_no_matching_session_dir(processing_settings, tmp_path):
    """If no <root>/video subdir matches (an underscore-free directory),
    construction raises FileNotFoundError instead of silently defaulting the
    session root to the cwd ('.'), which would route downstream reads/writes
    into the working directory."""
    (tmp_path / "video").mkdir()
    with pytest.raises(FileNotFoundError, match="No session joint-date directory"):
        ConvertTo3D(
            root_directory=str(tmp_path),
            input_parameter_dict=processing_settings,
            message_output=lambda *_a, **_kw: None,
        )


def test_convert_to_3d_init_picks_first_sorted_session_dir(processing_settings, tmp_path):
    """With multiple underscore-free subdirs the FIRST sorted one is chosen
    deterministically (not the filesystem-order last entry); underscore-bearing
    session dirs are ignored."""
    video = tmp_path / "video"
    video.mkdir()
    (video / "20230301").mkdir()
    (video / "20230101").mkdir()
    (video / "20230207_213549").mkdir()   # has an underscore -> ignored
    converter = ConvertTo3D(
        root_directory=str(tmp_path),
        input_parameter_dict=processing_settings,
        message_output=lambda *_a, **_kw: None,
    )
    assert converter.session_root_name == "20230101"


# ---------------------------------------------------------------------------
# sleap_file_conversion (subprocess + wait_for_subprocesses)
# ---------------------------------------------------------------------------


def test_sleap_file_conversion_invokes_subprocess_per_slp(
    processing_settings, session_with_video_dir, mocker
):
    """For every .slp file under any camera subdir, one Popen call is fired
    with `uvx --from sleap[nn] sleap-convert ...`."""
    root, sess_dir = session_with_video_dir
    cam1 = sess_dir / "cam1"
    cam2 = sess_dir / "cam2"
    cam1.mkdir()
    cam2.mkdir()
    (cam1 / "v1.slp").write_bytes(b"")
    (cam1 / "v2.slp").write_bytes(b"")
    (cam2 / "v3.slp").write_bytes(b"")
    # Non-.slp file should be ignored
    (cam1 / "ignore.txt").write_bytes(b"")

    popen_mock = mocker.patch(
        "usv_playpen.processing.anipose_operations.subprocess.Popen",
        return_value=MagicMock(returncode=0),
    )
    mocker.patch("usv_playpen.processing.anipose_operations.wait_for_subprocesses",
                 return_value=[0, 0, 0])
    mocker.patch("usv_playpen.processing.anipose_operations.smart_wait")

    converter = ConvertTo3D(
        root_directory=str(root),
        input_parameter_dict=processing_settings,
        message_output=lambda *_a, **_kw: None,
    )
    converter.sleap_file_conversion()

    assert popen_mock.call_count == 3  # exactly one per .slp
    # Every subprocess call should target sleap-convert with --format analysis
    for call in popen_mock.call_args_list:
        argv = call.kwargs.get("args") or call.args[0]
        assert "sleap-convert" in argv
        assert "--format" in argv
        assert "analysis" in argv


def test_sleap_file_conversion_no_slp_files(processing_settings,
                                              session_with_video_dir, mocker):
    """No .slp files anywhere → wait_for_subprocesses called with [], no Popen."""
    root, sess_dir = session_with_video_dir
    (sess_dir / "cam1").mkdir()
    popen_mock = mocker.patch(
        "usv_playpen.processing.anipose_operations.subprocess.Popen",
        return_value=MagicMock(returncode=0),
    )
    waiter = mocker.patch(
        "usv_playpen.processing.anipose_operations.wait_for_subprocesses",
        return_value=[],
    )
    mocker.patch("usv_playpen.processing.anipose_operations.smart_wait")

    converter = ConvertTo3D(
        root_directory=str(root),
        input_parameter_dict=processing_settings,
        message_output=lambda *_a, **_kw: None,
    )
    converter.sleap_file_conversion()

    assert popen_mock.call_count == 0
    assert waiter.call_args.kwargs["subps"] == []


# ---------------------------------------------------------------------------
# conduct_anipose_calibration (sleap_anipose.draw_board / .calibrate)
# ---------------------------------------------------------------------------


def test_conduct_anipose_calibration_draws_board_when_not_provided(
    processing_settings, session_with_video_dir, mocker
):
    """If board_provided_bool=False, draw_board is invoked before calibrate
    with parameters pulled out of the settings dict."""
    root, _ = session_with_video_dir
    processing_settings["anipose_operations"]["ConvertTo3D"]["conduct_anipose_calibration"]["board_provided_bool"] = False

    draw_mock = mocker.patch("usv_playpen.processing.anipose_operations.sleap_anipose.draw_board")
    calib_mock = mocker.patch("usv_playpen.processing.anipose_operations.sleap_anipose.calibrate")
    mocker.patch("usv_playpen.processing.anipose_operations.smart_wait")

    converter = ConvertTo3D(
        root_directory=str(root),
        input_parameter_dict=processing_settings,
        message_output=lambda *_a, **_kw: None,
    )
    converter.conduct_anipose_calibration()

    assert draw_mock.call_count == 1
    assert calib_mock.call_count == 1


def test_conduct_anipose_calibration_skips_draw_when_board_provided(
    processing_settings, session_with_video_dir, mocker
):
    """board_provided_bool=True → draw_board is NOT invoked, but calibrate is."""
    root, _ = session_with_video_dir
    processing_settings["anipose_operations"]["ConvertTo3D"]["conduct_anipose_calibration"]["board_provided_bool"] = True

    draw_mock = mocker.patch("usv_playpen.processing.anipose_operations.sleap_anipose.draw_board")
    calib_mock = mocker.patch("usv_playpen.processing.anipose_operations.sleap_anipose.calibrate")
    mocker.patch("usv_playpen.processing.anipose_operations.smart_wait")

    converter = ConvertTo3D(
        root_directory=str(root),
        input_parameter_dict=processing_settings,
        message_output=lambda *_a, **_kw: None,
    )
    converter.conduct_anipose_calibration()

    assert draw_mock.call_count == 0
    assert calib_mock.call_count == 1


# ---------------------------------------------------------------------------
# conduct_anipose_triangulation
# ---------------------------------------------------------------------------


def test_conduct_anipose_triangulation_logs_when_no_calibration(
    processing_settings, session_with_video_dir, mocker, tmp_path
):
    """No *_calibration.toml under calibration_file_loc → log skip + no
    triangulate call."""
    root, _ = session_with_video_dir
    # Point calibration_file_loc at an empty dir so the rglob finds nothing
    empty_calib_dir = tmp_path / "no_calib"
    empty_calib_dir.mkdir()
    cfg = processing_settings["anipose_operations"]["ConvertTo3D"]
    cfg["conduct_anipose_triangulation"]["calibration_file_loc"] = str(empty_calib_dir)

    triangulate_mock = mocker.patch(
        "usv_playpen.processing.anipose_operations.sleap_anipose.triangulate"
    )
    mocker.patch("usv_playpen.processing.anipose_operations.smart_wait")
    msgs: list[str] = []

    converter = ConvertTo3D(
        root_directory=str(root),
        input_parameter_dict=processing_settings,
        message_output=msgs.append,
    )
    converter.conduct_anipose_triangulation()

    assert triangulate_mock.call_count == 0
    assert any("Calibration directory not found" in m for m in msgs)


def test_conduct_anipose_triangulation_arena_branch_uses_one_frame(
    processing_settings, session_with_video_dir, mocker, tmp_path
):
    """When triangulate_arena_points_bool=True, frames default to [0, 1] so
    only a single frame is triangulated for arena calibration."""
    root, _ = session_with_video_dir
    # Provide a calibration toml under calibration_file_loc.
    calib_dir = tmp_path / "calib"
    (calib_dir / "video").mkdir(parents=True)
    (calib_dir / "video" / "session_calibration.toml").write_text("[cameras]\n")
    cfg = processing_settings["anipose_operations"]["ConvertTo3D"]
    cfg["conduct_anipose_triangulation"]["calibration_file_loc"] = str(calib_dir)
    cfg["conduct_anipose_triangulation"]["triangulate_arena_points_bool"] = True
    cfg["conduct_anipose_triangulation"]["frame_restriction"] = None

    triangulate_mock = mocker.patch(
        "usv_playpen.processing.anipose_operations.sleap_anipose.triangulate"
    )
    mocker.patch("usv_playpen.processing.anipose_operations.smart_wait")

    converter = ConvertTo3D(
        root_directory=str(root),
        input_parameter_dict=processing_settings,
        message_output=lambda *_a, **_kw: None,
    )
    converter.conduct_anipose_triangulation()

    assert triangulate_mock.call_count == 1
    # The arena branch sets frames=(0, 1) — confirm that's what was passed.
    assert triangulate_mock.call_args.kwargs["frames"] == (0, 1)
    # And the temporary frame_restriction should have been reverted to None.
    assert cfg["conduct_anipose_triangulation"]["frame_restriction"] is None


def test_conduct_anipose_triangulation_session_branch_reads_frame_count(
    processing_settings, session_with_video_dir, mocker, tmp_path
):
    """When triangulate_arena_points_bool=False and frame_restriction=None,
    the method reads camera_frame_count_dict.json to derive the frame range."""
    root, _ = session_with_video_dir
    # JSON with the expected key
    fc_path = root / "video" / "x_camera_frame_count_dict.json"
    fc_path.write_text(json.dumps({"total_frame_number_least": 5000,
                                   "median_empirical_camera_sr": 150.0}))
    # Calibration toml
    calib_dir = tmp_path / "calib"
    (calib_dir / "video").mkdir(parents=True)
    (calib_dir / "video" / "session_calibration.toml").write_text("[cameras]\n")

    cfg = processing_settings["anipose_operations"]["ConvertTo3D"]
    cfg["conduct_anipose_triangulation"]["calibration_file_loc"] = str(calib_dir)
    cfg["conduct_anipose_triangulation"]["triangulate_arena_points_bool"] = False
    cfg["conduct_anipose_triangulation"]["frame_restriction"] = None

    triangulate_mock = mocker.patch(
        "usv_playpen.processing.anipose_operations.sleap_anipose.triangulate"
    )
    mocker.patch("usv_playpen.processing.anipose_operations.smart_wait")

    converter = ConvertTo3D(
        root_directory=str(root),
        input_parameter_dict=processing_settings,
        message_output=lambda *_a, **_kw: None,
    )
    converter.conduct_anipose_triangulation()

    assert triangulate_mock.call_count == 1
    assert triangulate_mock.call_args.kwargs["frames"] == (0, 5000)
    # And the temporary frame_restriction should have been reverted to None.
    assert cfg["conduct_anipose_triangulation"]["frame_restriction"] is None


# ---------------------------------------------------------------------------
# translate_rotate_metric — arena + animal transform branches
#
# Builds a synthetic square arena points3d.h5 (the four North/West/South/East
# corner nodes laid out as a unit square, the 24 channel nodes left at the
# origin) plus the session's camera-frame-count JSON, then drives the full
# metric-conversion + translate + multi-axis rotation pipeline through to a
# written *_translated_rotated_metric.h5. The 'animal' branch additionally
# transforms a synthetic 15-node mouse points3d.h5.
# ---------------------------------------------------------------------------


# Real arena skeleton: North/West/South/East occupy indices 0..3, followed by
# 24 ch_* nodes (28 total). Lay the corners out as a unit square in z=0.
_ARENA_N_NODES = 28
_ARENA_CORNERS = {0: (0.0, 1.0, 0.0),    # North
                  1: (-1.0, 0.0, 0.0),   # West
                  2: (0.0, -1.0, 0.0),   # South
                  3: (1.0, 0.0, 0.0)}    # East


def _write_arena_h5(arena_dir: Path) -> Path:
    """
    Description
    -----------
    Write a synthetic arena `*_points3d.h5` with a single frame / single
    'animal' (the arena) and the real 28-node arena skeleton layout: the
    four corner nodes form a unit square in the z=0 plane, every channel
    node sits at the origin. Only the `tracks` dataset is read back by
    `translate_rotate_metric`.

    Parameters
    ----------
    arena_dir (pathlib.Path)
        Directory to write `arena_points3d.h5` into (created if missing).

    Returns
    -------
    h5_path (pathlib.Path)
        Path to the written arena H5.
    """

    arena_dir.mkdir(parents=True, exist_ok=True)
    tracks = np.zeros((1, 1, _ARENA_N_NODES, 3), dtype="float64")
    for idx, xyz in _ARENA_CORNERS.items():
        tracks[0, 0, idx, :] = xyz
    h5_path = arena_dir / "arena_points3d.h5"
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("tracks", data=tracks)
    return h5_path


def _trm_settings(processing_settings, arena_dir, *, mode, exp_codes, delete=False):
    """
    Description
    -----------
    Return a copy of the processing-settings dict with the
    `translate_rotate_metric` block pointed at a synthetic arena directory
    and configured for the requested transform mode.

    Parameters
    ----------
    processing_settings (dict)
        Base settings loaded from the package.
    arena_dir (pathlib.Path)
        Directory holding `arena_points3d.h5`.
    mode (str)
        `'arena'` or `'animal'`.
    exp_codes (list[str])
        Per-session experimental codes.
    delete (bool)
        Whether to delete the original mouse H5 after the animal transform.

    Returns
    -------
    settings (dict)
        The mutated settings dict (safe to pass to ConvertTo3D).
    """

    processing_settings["anipose_operations"]["ConvertTo3D"]["translate_rotate_metric"] = {
        "original_arena_file_loc": str(arena_dir),
        "save_transformed_data":   mode,
        "delete_original_h5":      delete,
        "static_reference_len":    0.615,
        "experimental_codes":      exp_codes,
    }
    return processing_settings


def test_translate_rotate_metric_arena_branch_writes_transformed_arena(
    processing_settings, tmp_path, mocker,
):
    """
    Description
    -----------
    With `save_transformed_data='arena'`, `translate_rotate_metric` must
    metric-scale, translate-to-midpoint, and multi-axis rotate the arena
    points, then write `<arena>_translated_rotated_metric.h5` carrying the
    transformed `tracks` and the arena `node_names`. Asserts the output H5
    exists with the expected (1, 1, 28, 3) shape and node-name count.

    Parameters
    ----------
    processing_settings (dict)
        Package settings fixture.
    tmp_path (pathlib.Path)
        Per-test temp directory.
    mocker (pytest_mock.MockerFixture)
        Used to no-op `smart_wait`.

    Returns
    -------
    None
    """

    mocker.patch("usv_playpen.processing.anipose_operations.smart_wait")
    video_dir = tmp_path / "video"
    video_dir.mkdir()
    (video_dir / "20230101120000").mkdir()   # underscore-free session joint-date dir
    (video_dir / "sess_camera_frame_count_dict.json").write_text(
        json.dumps({"median_empirical_camera_sr": 150.0})
    )
    arena_dir = tmp_path / "arena_src"
    _write_arena_h5(arena_dir)

    settings = _trm_settings(processing_settings, arena_dir, mode="arena", exp_codes=[])
    converter = ConvertTo3D(
        root_directory=str(tmp_path),
        input_parameter_dict=settings,
        message_output=lambda *_a, **_k: None,
    )
    converter.translate_rotate_metric()

    out_h5 = arena_dir / "arena_points3d_translated_rotated_metric.h5"
    assert out_h5.is_file(), "transformed arena H5 not written"
    with h5py.File(out_h5, "r") as f:
        assert f["tracks"].shape == (1, 1, _ARENA_N_NODES, 3)
        assert len(f["node_names"]) == _ARENA_N_NODES


def test_translate_rotate_metric_animal_branch_writes_transformed_mouse(
    processing_settings, tmp_path, mocker,
):
    """
    Description
    -----------
    With `save_transformed_data='animal'`, `translate_rotate_metric` must
    apply the arena-derived metric/translate/rotation transform to the
    session's 15-node mouse points, zero any negative z, and write
    `<session>_points3d_translated_rotated_metric.h5` carrying `tracks`,
    `node_names`, `track_names`, `experimental_code`, and
    `recording_frame_rate`. With `delete_original_h5=True` the source mouse
    H5 must be removed. A session metadata file is provided, so the
    metadata-update block runs and `find_mouse_names` resolves the track
    names from the `Subjects` list (here `["m1"]`).

    Parameters
    ----------
    processing_settings (dict)
        Package settings fixture.
    tmp_path (pathlib.Path)
        Per-test temp directory.
    mocker (pytest_mock.MockerFixture)
        Used to no-op `smart_wait`.

    Returns
    -------
    None
    """

    mocker.patch("usv_playpen.processing.anipose_operations.smart_wait")
    video_dir = tmp_path / "video"
    video_dir.mkdir()
    (video_dir / "sess_camera_frame_count_dict.json").write_text(
        json.dumps({"median_empirical_camera_sr": 150.0})
    )
    # Session metadata so the metadata-update block runs and find_mouse_names
    # reads track names from the Subjects list.
    (tmp_path / "sess_metadata.yaml").write_text(
        "Session:\n  id: s001\nSubjects:\n  - subject_id: m1\n"
    )
    # No-underscore session dir is what __init__ latches onto.
    session_dir = video_dir / "20260101120000"
    session_dir.mkdir()
    mouse_tracks = np.zeros((5, 1, 15, 3), dtype="float64")
    rng = np.random.default_rng(0)
    mouse_tracks[...] = rng.uniform(-0.5, 0.5, size=mouse_tracks.shape)
    with h5py.File(session_dir / "20260101120000_points3d.h5", "w") as f:
        f.create_dataset("tracks", data=mouse_tracks)

    arena_dir = tmp_path / "arena_src"
    _write_arena_h5(arena_dir)

    settings = _trm_settings(
        processing_settings, arena_dir, mode="animal",
        exp_codes=["E2MF"], delete=True,
    )
    converter = ConvertTo3D(
        root_directory=str(tmp_path),
        input_parameter_dict=settings,
        message_output=lambda *_a, **_k: None,
    )
    converter.translate_rotate_metric(session_idx=0)

    out_h5 = session_dir / "20260101120000_points3d_translated_rotated_metric.h5"
    assert out_h5.is_file(), "transformed mouse H5 not written"
    with h5py.File(out_h5, "r") as f:
        assert f["tracks"].shape == (5, 1, 15, 3)
        assert f["experimental_code"][()].decode("utf-8") == "E2MF"
        assert float(f["recording_frame_rate"][()]) == 150.0
        assert [n.decode("utf-8") for n in f["track_names"]] == ["m1"]
        # negative z must have been clamped to zero
        assert float(np.asarray(f["tracks"])[:, :, :, 2].min()) >= 0.0
    assert not (session_dir / "20260101120000_points3d.h5").exists(), (
        "original mouse H5 should be deleted when delete_original_h5=True"
    )


# find_mouse_names — legacy imgstore metadata branch (no cage/subject keys)
def test_find_mouse_names_legacy_keeps_present_m2_with_empty_cage(tmp_path, mocker):
    """
    Description
    -----------
    In the legacy imgstore-metadata layout (no flat `cage`/`subject` keys, only
    suffixed `..._cage_ID_mN` / `..._mouse_ID_mN` keys), a second animal whose
    `mouse_ID_m2` is populated must never be silently dropped just because its
    `cage_ID_m2` field happens to be empty. The previous nested `if` skipped
    exactly that case; the fix mirrors the m1 logic and falls back to the bare
    mouse ID.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Per-test temp directory.
    mocker (pytest_mock.MockerFixture)
        Used to stub the imgstore `new_for_filename` factory.

    Returns
    -------
    None
    """

    sub = tmp_path / "video" / "20260101_1200.imgstore"
    sub.mkdir(parents=True)
    store = MagicMock()
    store.user_metadata = {
        "p_cage_ID_m1": "c1", "p_mouse_ID_m1": "m1",
        "p_cage_ID_m2": "",   "p_mouse_ID_m2": "m2",
    }
    mocker.patch(
        "usv_playpen.processing.anipose_operations.new_for_filename",
        return_value=store,
    )
    assert find_mouse_names(root_directory=str(tmp_path), metadata=None) == [
        "c1_m1",
        "m2",
    ]


def test_find_mouse_names_legacy_missing_m1_keys_raises(tmp_path, mocker):
    """
    Description
    -----------
    If the mandatory first-animal keys (`..._cage_ID_m1` / `..._mouse_ID_m1`)
    are absent from the legacy imgstore metadata, `find_mouse_names` must fail
    with a clear KeyError naming the missing keys, rather than raising a bare,
    cryptic KeyError from the downstream dictionary lookup.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Per-test temp directory.
    mocker (pytest_mock.MockerFixture)
        Used to stub the imgstore `new_for_filename` factory.

    Returns
    -------
    None
    """

    sub = tmp_path / "video" / "20260101_1200.imgstore"
    sub.mkdir(parents=True)
    store = MagicMock()
    store.user_metadata = {"p_cage_ID_m2": "c2", "p_mouse_ID_m2": "m2"}
    mocker.patch(
        "usv_playpen.processing.anipose_operations.new_for_filename",
        return_value=store,
    )
    with pytest.raises(KeyError, match="missing required key"):
        find_mouse_names(root_directory=str(tmp_path), metadata=None)


# conduct_anipose_triangulation — frame_restriction must not leak on error
def test_conduct_anipose_triangulation_error_keeps_frame_restriction_none(
    processing_settings, session_with_video_dir, mocker, tmp_path
):
    """
    Description
    -----------
    When `sleap_anipose.triangulate` raises, the shared settings dict's
    `frame_restriction` must remain `None`. The old code wrote the computed
    `[0, N]` back into the dict and only reset it *after* a successful
    triangulate, so a raised triangulate left the stale value behind and
    contaminated the next session in a batch run; the fix resolves the range
    into a local variable and never mutates the shared dict.

    Parameters
    ----------
    processing_settings (dict)
        Package settings fixture.
    session_with_video_dir (tuple)
        (root, session_dir) scaffolding fixture.
    mocker (pytest_mock.MockerFixture)
        Used to force triangulate to raise and to no-op smart_wait.
    tmp_path (pathlib.Path)
        Per-test temp directory.

    Returns
    -------
    None
    """

    root, _ = session_with_video_dir
    fc_path = root / "video" / "x_camera_frame_count_dict.json"
    fc_path.write_text(
        json.dumps({"total_frame_number_least": 5000, "median_empirical_camera_sr": 150.0})
    )
    calib_dir = tmp_path / "calib"
    (calib_dir / "video").mkdir(parents=True)
    (calib_dir / "video" / "session_calibration.toml").write_text("[cameras]\n")

    cfg = processing_settings["anipose_operations"]["ConvertTo3D"]
    cfg["conduct_anipose_triangulation"]["calibration_file_loc"] = str(calib_dir)
    cfg["conduct_anipose_triangulation"]["triangulate_arena_points_bool"] = False
    cfg["conduct_anipose_triangulation"]["frame_restriction"] = None

    mocker.patch(
        "usv_playpen.processing.anipose_operations.sleap_anipose.triangulate",
        side_effect=RuntimeError("boom"),
    )
    mocker.patch("usv_playpen.processing.anipose_operations.smart_wait")

    converter = ConvertTo3D(
        root_directory=str(root),
        input_parameter_dict=processing_settings,
        message_output=lambda *_a, **_k: None,
    )
    with pytest.raises(RuntimeError, match="boom"):
        converter.conduct_anipose_triangulation()

    assert cfg["conduct_anipose_triangulation"]["frame_restriction"] is None


# translate_rotate_metric — session_idx bounds guard
def test_translate_rotate_metric_session_idx_out_of_range_raises(
    processing_settings, tmp_path, mocker
):
    """
    Description
    -----------
    In the animal branch, an out-of-range `session_idx` (here 5 against a single
    configured experimental code) must raise a clear IndexError before the
    metadata update / h5 write, rather than a bare positional IndexError deep
    inside those writes.

    Parameters
    ----------
    processing_settings (dict)
        Package settings fixture.
    tmp_path (pathlib.Path)
        Per-test temp directory.
    mocker (pytest_mock.MockerFixture)
        Used to no-op smart_wait.

    Returns
    -------
    None
    """

    mocker.patch("usv_playpen.processing.anipose_operations.smart_wait")
    video_dir = tmp_path / "video"
    video_dir.mkdir()
    (video_dir / "sess_camera_frame_count_dict.json").write_text(
        json.dumps({"median_empirical_camera_sr": 150.0})
    )
    (tmp_path / "sess_metadata.yaml").write_text(
        "Session:\n  id: s001\nSubjects:\n  - subject_id: m1\n"
    )
    session_dir = video_dir / "20260101120000"
    session_dir.mkdir()
    mouse_tracks = np.zeros((5, 1, 15, 3), dtype="float64")
    with h5py.File(session_dir / "20260101120000_points3d.h5", "w") as f:
        f.create_dataset("tracks", data=mouse_tracks)

    arena_dir = tmp_path / "arena_src"
    _write_arena_h5(arena_dir)

    settings = _trm_settings(
        processing_settings, arena_dir, mode="animal",
        exp_codes=["E2MF"], delete=False,
    )
    converter = ConvertTo3D(
        root_directory=str(tmp_path),
        input_parameter_dict=settings,
        message_output=lambda *_a, **_k: None,
    )
    with pytest.raises(IndexError, match="out of range"):
        converter.translate_rotate_metric(session_idx=5)


# Real (un-mocked) opencv/aruco checks. The orchestration tests above mock
# sleap_anipose, so they pass even when the active opencv has lost the
# function-style cv2.aruco API that aniposelib's calibration depends on. These
# two exercise that API directly, so a broken opencv install (e.g. base
# opencv-python pulled by ultralytics shadowing opencv-contrib-python) fails the
# suite instead of only surfacing at calibration time on the rig.


def test_cv2_aruco_function_api_present():
    """Guard the opencv install: aniposelib's calibration calls the function-style
    cv2.aruco API (detectMarkers / refineDetectedMarkers / interpolateCornersCharuco
    / estimatePoseCharucoBoard), which only ``opencv-contrib-python`` ships. Base
    ``opencv-python`` (pulled transitively by ultralytics) lacks these symbols and
    clobbers the same ``cv2`` files, so this fails loudly if the ``opencv-python``
    exclusion in ``pyproject.toml`` ever stops keeping contrib the sole provider."""
    required = (
        "detectMarkers",
        "refineDetectedMarkers",
        "interpolateCornersCharuco",
        "estimatePoseCharucoBoard",
    )
    missing = [name for name in required if not hasattr(cv2.aruco, name)]
    assert not missing, (
        f"cv2.aruco is missing {missing} (active cv2 {cv2.__version__}); the "
        f"installed opencv is not opencv-contrib-python. Check the opencv-python "
        f"override in pyproject.toml."
    )


def test_charuco_pose_pipeline_is_geometrically_correct(tmp_path):
    """Full calibration-math check (not just symbol presence): render a ChArUco
    board via the real ``sleap_anipose.draw_board``, then run the exact deprecated
    function-style ``cv2.aruco`` pipeline aniposelib's ``calibrate()`` uses per frame
    -- ``detectMarkers`` -> ``interpolateCornersCharuco`` -> ``estimatePoseCharucoBoard``
    -- and reproject the known 3D board corners with the recovered pose. A small mean
    reprojection error proves the pose is computed CORRECTLY, so this guards the
    calibration geometry against a broken opencv, not merely that the calls exist."""
    board_x, board_y = 5, 7
    square_length, marker_length = 24.0, 18.0
    board_path = tmp_path / "charuco_board.jpg"
    sleap_anipose.draw_board(
        str(board_path), board_x, board_y, square_length, marker_length, 4, 50, 1440, 1080
    )
    assert board_path.is_file()

    image = cv2.imread(str(board_path), cv2.IMREAD_GRAYSCALE)
    assert image is not None
    height, width = image.shape

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    board = cv2.aruco.CharucoBoard((board_x, board_y), square_length, marker_length, aruco_dict)

    corners, ids, _rejected = cv2.aruco.detectMarkers(image, aruco_dict)
    assert ids is not None, "no aruco markers detected on the rendered board"
    assert len(ids) > 0, "no aruco markers detected on the rendered board"

    n_charuco, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        corners, ids, image, board
    )
    assert n_charuco > 0, "no charuco corners interpolated"

    # Recover the board pose with a plausible pinhole camera, then reproject the
    # known 3D chessboard corners and measure the error against the detected ones.
    camera_matrix = np.array(
        [[float(width), 0.0, width / 2.0], [0.0, float(width), height / 2.0], [0.0, 0.0, 1.0]]
    )
    dist_coeffs = np.zeros((5,), dtype=float)
    ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
        charuco_corners, charuco_ids, board, camera_matrix, dist_coeffs, None, None
    )
    assert ok, "estimatePoseCharucoBoard failed"

    object_points = board.getChessboardCorners()[charuco_ids.flatten()]
    reprojected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
    mean_error = float(
        np.linalg.norm(reprojected.reshape(-1, 2) - charuco_corners.reshape(-1, 2), axis=1).mean()
    )
    assert mean_error < 2.0, f"charuco pose reprojection error too large ({mean_error:.3f} px)"
