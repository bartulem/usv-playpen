"""
@author: bartulem
Tests for usv_playpen.processing.extract_phidget_data.Gatherer.prepare_data_for_analyses:
the happy path (single + multi file, missing-key -> NaN) and the three error
paths that previously crashed with AttributeError / IndexError / a raw
iterdir FileNotFoundError (no matching camera dir, no json files, no video dir).

Coverage: Gatherer construction (a required root_directory), the camera
folder choice (calibration folders skipped, exactly one recording folder) and
prepare_data_for_analyses.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from usv_playpen.processing.extract_phidget_data import Gatherer

CAM = "21372315"


def _input_dict(camera=CAM):
    """Return the minimal settings dict Gatherer reads: the
    ``extract_phidget_data`` block naming ``camera`` as the camera whose directory
    holds the phidget ``*.extra_data.json`` files."""
    return {"extract_phidget_data": {"Gatherer": {
        "prepare_data_for_analyses": {"extra_data_camera": camera}}}}


def _make_session(tmp_path, records_per_file, cam=CAM, make_json=True):
    """Create ``<tmp_path>/video/20250101_120000.<cam>/`` and, unless ``make_json`` is
    False, write one ``NNNNNN.extra_data.json`` per entry of ``records_per_file``
    holding that entry's records. Returns the session root (``tmp_path``)."""
    cam_dir = tmp_path / "video" / f"20250101_120000.{cam}"
    cam_dir.mkdir(parents=True)
    if make_json:
        for i, recs in enumerate(records_per_file):
            (cam_dir / f"{i:06d}.extra_data.json").write_text(json.dumps(recs))
    return tmp_path


def _gatherer(tmp_path):
    """Return a Gatherer on the session root ``tmp_path`` configured for the default
    camera ``CAM``."""
    return Gatherer(input_parameter_dict=_input_dict(), root_directory=str(tmp_path))


def test_happy_path_single_file_sorted_by_sensor_time(tmp_path):
    """Records written out of order come back sorted by sensor_time, with hum_h,
    lux and hum_t mapped to humidity, lux and temperature, so the sensor traces
    line up with time."""
    recs = [
        {"sensor_time": 2.0, "hum_h": 51.0, "lux": 110.0, "hum_t": 23.0},
        {"sensor_time": 1.0, "hum_h": 50.0, "lux": 100.0, "hum_t": 22.0},
    ]
    root = _make_session(tmp_path, [recs])
    out = _gatherer(root).prepare_data_for_analyses()
    assert list(out["humidity"]) == [50.0, 51.0]
    assert list(out["lux"]) == [100.0, 110.0]
    assert list(out["temperature"]) == [22.0, 23.0]


def test_multiple_files_concatenated(tmp_path):
    """Records spread over several json files are concatenated into one trace, so
    a recording split across files loses no samples."""
    root = _make_session(tmp_path, [
        [{"sensor_time": 1.0, "hum_h": 50.0}],
        [{"sensor_time": 2.0, "hum_h": 60.0}],
    ])
    out = _gatherer(root).prepare_data_for_analyses()
    assert list(out["humidity"]) == [50.0, 60.0]


def test_missing_keys_become_nan(tmp_path):
    """A record lacking a sensor key yields NaN for that channel instead of a
    KeyError, while the channels it does hold keep their values."""
    root = _make_session(tmp_path, [[{"sensor_time": 1.0, "lux": 100.0}]])
    out = _gatherer(root).prepare_data_for_analyses()
    assert np.isnan(out["humidity"][0])
    assert out["lux"][0] == 100.0
    assert np.isnan(out["temperature"][0])


def _realistic_record(sensor_time, hum_h, lux, hum_t, is_recording=True):
    """A record shaped like a real '*.extra_data.json' entry (10 keys); the code
    reads only sensor_time/hum_h/lux/hum_t and must ignore the rest."""
    return {
        "sensor_time": sensor_time, "lux": lux, "hum_h": hum_h, "hum_t": hum_t,
        "frame_time": sensor_time - 13.0, "frame_number": 35, "frame_index": 34,
        "is_recording": is_recording, "recording_sequence_number": 90,
        "recording_start": 1675823751.657399,
    }


def test_extracts_from_full_schema_records_ignoring_extra_keys(tmp_path):
    """Real records carry 10 keys; only the three sensor channels are extracted,
    sorted by the Unix-timestamp sensor_time, and the extra keys are ignored."""
    recs = [
        _realistic_record(1675823753.0, hum_h=35.9, lux=40.75, hum_t=26.63),
        _realistic_record(1675823752.0, hum_h=35.8, lux=40.50, hum_t=26.60),
    ]
    root = _make_session(tmp_path, [recs])
    out = _gatherer(root).prepare_data_for_analyses()
    assert list(out["humidity"]) == [35.8, 35.9]
    assert list(out["lux"]) == [40.50, 40.75]
    assert list(out["temperature"]) == [26.60, 26.63]
    assert set(out.keys()) == {"humidity", "lux", "temperature"}


def test_stray_non_extra_data_json_is_ignored(tmp_path):
    """A non-phidget .json in the camera dir (e.g. some metadata file) must not be
    loaded as phidget data; only '*extra_data*.json' files are read."""
    root = _make_session(tmp_path, [[_realistic_record(1675823752.0, 35.8, 40.5, 26.6)]])
    cam_dir = next((root / "video").iterdir())
    (cam_dir / "frame_metadata.json").write_text(json.dumps({"not": "phidget data"}))
    out = _gatherer(root).prepare_data_for_analyses()
    assert list(out["humidity"]) == [35.8]  # stray file ignored, no crash


def test_no_matching_camera_dir_raises(tmp_path):
    """No directory of the configured camera raises a FileNotFoundError that says
    so, instead of the AttributeError / IndexError it used to crash with."""
    _make_session(tmp_path, [[{"sensor_time": 1.0}]], cam="99999999")
    with pytest.raises(FileNotFoundError, match="No camera directory"):
        _gatherer(tmp_path).prepare_data_for_analyses()


def test_empty_json_set_raises(tmp_path):
    """A camera directory with no phidget json files raises a FileNotFoundError
    naming the missing phidget data rather than returning empty traces."""
    _make_session(tmp_path, [], make_json=False)
    with pytest.raises(FileNotFoundError, match="No phidget"):
        _gatherer(tmp_path).prepare_data_for_analyses()


def test_missing_video_dir_raises(tmp_path):
    """A session without a video directory raises a FileNotFoundError that names
    it, instead of a raw iterdir error."""
    with pytest.raises(FileNotFoundError, match="Video directory not found"):
        _gatherer(tmp_path).prepare_data_for_analyses()


def test_gatherer_requires_a_root_directory():
    """root_directory is required: the processing settings hold no session root, so
    None raises a ValueError instead of a KeyError on a key that does not exist."""
    with pytest.raises(ValueError, match="needs a root_directory"):
        Gatherer(input_parameter_dict=None, root_directory=None)


def test_camera_calibration_directory_is_skipped(tmp_path):
    """A 'calibration_<ts>.<serial>' directory of the camera is not its recording
    directory and is skipped even where it would sort first."""
    _make_session(tmp_path, [[{"sensor_time": 1.0, "hum_h": 50.0, "lux": 100.0, "hum_t": 22.0}]])
    calibration_dir = tmp_path / "video" / f"calibration_20241107_110512.{CAM}"
    calibration_dir.mkdir()
    (calibration_dir / "000000.extra_data.json").write_text(json.dumps([{"sensor_time": 9.0, "lux": 1.0}]))
    out = _gatherer(tmp_path).prepare_data_for_analyses()
    assert out["lux"].tolist() == [100.0]


def test_two_recording_directories_of_the_camera_raise(tmp_path):
    """Two recording directories of the camera (e.g. an aborted first recording)
    raise a ValueError naming both instead of the first sorted one being read."""
    _make_session(tmp_path, [[{"sensor_time": 1.0, "lux": 100.0}]])
    second = tmp_path / "video" / f"20250101_120500.{CAM}"
    second.mkdir()
    (second / "000000.extra_data.json").write_text(json.dumps([{"sensor_time": 2.0, "lux": 5.0}]))
    with pytest.raises(ValueError, match=rf"2 recording directories of camera '{CAM}'"):
        _gatherer(tmp_path).prepare_data_for_analyses()
