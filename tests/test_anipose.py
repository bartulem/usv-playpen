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

import pytest

from usv_playpen.anipose_operations import ConvertTo3D


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def processing_settings():
    """Loads processing_settings.json from the package once per test."""
    import usv_playpen
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


def test_convert_to_3d_init_handles_session_with_no_matching_dir(processing_settings, tmp_path):
    """If no <root>/video subdir matches, both attrs stay at default empty values."""
    (tmp_path / "video").mkdir()
    converter = ConvertTo3D(
        root_directory=str(tmp_path),
        input_parameter_dict=processing_settings,
        message_output=lambda *_a, **_kw: None,
    )
    # session_root_joint_date_dir starts as an empty pathlib.Path()
    assert str(converter.session_root_joint_date_dir) == "."
    assert converter.session_root_name == ""


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
        "usv_playpen.anipose_operations.subprocess.Popen",
        return_value=MagicMock(returncode=0),
    )
    mocker.patch("usv_playpen.anipose_operations.wait_for_subprocesses",
                 return_value=[0, 0, 0])
    mocker.patch("usv_playpen.anipose_operations.smart_wait")

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
        assert "--format" in argv and "analysis" in argv


def test_sleap_file_conversion_no_slp_files(processing_settings,
                                              session_with_video_dir, mocker):
    """No .slp files anywhere → wait_for_subprocesses called with [], no Popen."""
    root, sess_dir = session_with_video_dir
    (sess_dir / "cam1").mkdir()
    popen_mock = mocker.patch(
        "usv_playpen.anipose_operations.subprocess.Popen",
        return_value=MagicMock(returncode=0),
    )
    waiter = mocker.patch(
        "usv_playpen.anipose_operations.wait_for_subprocesses",
        return_value=[],
    )
    mocker.patch("usv_playpen.anipose_operations.smart_wait")

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

    draw_mock = mocker.patch("usv_playpen.anipose_operations.sleap_anipose.draw_board")
    calib_mock = mocker.patch("usv_playpen.anipose_operations.sleap_anipose.calibrate")
    mocker.patch("usv_playpen.anipose_operations.smart_wait")

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

    draw_mock = mocker.patch("usv_playpen.anipose_operations.sleap_anipose.draw_board")
    calib_mock = mocker.patch("usv_playpen.anipose_operations.sleap_anipose.calibrate")
    mocker.patch("usv_playpen.anipose_operations.smart_wait")

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
        "usv_playpen.anipose_operations.sleap_anipose.triangulate"
    )
    mocker.patch("usv_playpen.anipose_operations.smart_wait")
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
        "usv_playpen.anipose_operations.sleap_anipose.triangulate"
    )
    mocker.patch("usv_playpen.anipose_operations.smart_wait")

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
        "usv_playpen.anipose_operations.sleap_anipose.triangulate"
    )
    mocker.patch("usv_playpen.anipose_operations.smart_wait")

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
