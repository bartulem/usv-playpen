"""
Tests for usv_playpen.processing.extract_phidget_data.Gatherer.prepare_data_for_analyses:
the happy path (single + multi file, missing-key -> NaN) and the three error
paths that previously crashed with AttributeError / IndexError / a raw
iterdir FileNotFoundError (no matching camera dir, no json files, no video dir).
"""

import json

import numpy as np
import pytest

from usv_playpen.processing.extract_phidget_data import Gatherer

CAM = "21372315"


def _input_dict(camera=CAM):
    return {"extract_phidget_data": {"Gatherer": {
        "prepare_data_for_analyses": {"extra_data_camera": camera}}}}


def _make_session(tmp_path, records_per_file, cam=CAM, make_json=True):
    cam_dir = tmp_path / "video" / f"20250101_120000.{cam}"
    cam_dir.mkdir(parents=True)
    if make_json:
        for i, recs in enumerate(records_per_file):
            (cam_dir / f"{i:06d}.extra_data.json").write_text(json.dumps(recs))
    return tmp_path


def _gatherer(tmp_path):
    return Gatherer(input_parameter_dict=_input_dict(), root_directory=str(tmp_path))


def test_happy_path_single_file_sorted_by_sensor_time(tmp_path):
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
    root = _make_session(tmp_path, [
        [{"sensor_time": 1.0, "hum_h": 50.0}],
        [{"sensor_time": 2.0, "hum_h": 60.0}],
    ])
    out = _gatherer(root).prepare_data_for_analyses()
    assert list(out["humidity"]) == [50.0, 60.0]


def test_missing_keys_become_nan(tmp_path):
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
    _make_session(tmp_path, [[{"sensor_time": 1.0}]], cam="99999999")
    with pytest.raises(FileNotFoundError, match="No camera directory"):
        _gatherer(tmp_path).prepare_data_for_analyses()


def test_empty_json_set_raises(tmp_path):
    _make_session(tmp_path, [], make_json=False)
    with pytest.raises(FileNotFoundError, match="No phidget"):
        _gatherer(tmp_path).prepare_data_for_analyses()


def test_missing_video_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Video directory not found"):
        _gatherer(tmp_path).prepare_data_for_analyses()
