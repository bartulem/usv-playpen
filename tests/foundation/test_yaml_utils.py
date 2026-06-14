"""
@author: bartulem
Tests for yaml_utils: SmartDumper formatting, sync_equipment_dynamic_fields,
load/save_session_metadata round-trip, set_sync_LEDs_device_port.
"""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from usv_playpen.yaml_utils import (
    SmartDumper,
    load_session_metadata,
    save_session_metadata,
    sync_equipment_dynamic_fields,
    set_sync_LEDs_device_port,
    _set_block_field_in_position,
)


# ---------------------------------------------------------------------------
# SmartDumper.represent_list — flow vs block style
# ---------------------------------------------------------------------------


def test_smart_dumper_simple_list_uses_flow_style():
    """Simple lists of scalars should be emitted in flow style ([1, 2, 3])."""
    out = yaml.dump({"nums": [1, 2, 3]}, Dumper=SmartDumper,
                    default_flow_style=False, sort_keys=False)
    # Flow style for the list — square brackets on a single line
    assert "[1, 2, 3]" in out


def test_smart_dumper_nested_list_uses_block_style():
    """A list whose elements are themselves lists/dicts should fall back
    to block style (one item per line)."""
    out = yaml.dump({"matrix": [[1, 2], [3, 4]]}, Dumper=SmartDumper,
                    default_flow_style=False, sort_keys=False)
    # Block style → no flat "[[1, 2], [3, 4]]" form
    assert "- - 1" in out or "- -1" in out or "matrix:\n" in out


def test_smart_dumper_mixed_list_uses_block_style():
    """A list mixing dicts and scalars should still emit in block style."""
    out = yaml.dump({"items": [{"a": 1}, "scalar"]}, Dumper=SmartDumper,
                    default_flow_style=False, sort_keys=False)
    assert "- a:" in out


# ---------------------------------------------------------------------------
# SmartDumper.represent_str — single-quoting YAML 1.1 coercion candidates
# ---------------------------------------------------------------------------


def test_smart_dumper_quotes_integer_looking_strings():
    """A string that LOOKS like an int must be single-quoted to prevent
    YAML 1.1 readers from coercing it to int on load."""
    out = yaml.dump({"port": "1234"}, Dumper=SmartDumper,
                    default_flow_style=False, sort_keys=False)
    assert "port: '1234'" in out


def test_smart_dumper_quotes_iso_date_strings():
    """ISO-date strings (YYYY-MM-DD) must be quoted."""
    out = yaml.dump({"date": "2026-01-15"}, Dumper=SmartDumper,
                    default_flow_style=False, sort_keys=False)
    assert "date: '2026-01-15'" in out


@pytest.mark.parametrize("yaml_bool", ["yes", "No", "TRUE", "off", "On"])
def test_smart_dumper_quotes_yaml11_boolean_literals(yaml_bool):
    """YAML 1.1 boolean literals (yes/no/on/off/true/false in any case) must
    be single-quoted to preserve them as strings."""
    out = yaml.dump({"flag": yaml_bool}, Dumper=SmartDumper,
                    default_flow_style=False, sort_keys=False)
    assert f"flag: '{yaml_bool}'" in out


def test_smart_dumper_does_not_quote_plain_string():
    """A plain ASCII string with no coercion risk should not be quoted."""
    out = yaml.dump({"name": "hello_world"}, Dumper=SmartDumper,
                    default_flow_style=False, sort_keys=False)
    assert "name: hello_world" in out


@pytest.mark.parametrize("coercible", [
    "3.14", ".5", "1.0",
    ".inf", ".nan",
    "null", "~", "",
    "0x1A", "012", "1:2:3",
])
def test_smart_dumper_quotes_all_yaml11_coercible_strings(coercible):
    """Any string YAML 1.1 would coerce (float / .inf / .nan / null / non-decimal
    int) must survive a dump+load round-trip as the original string."""
    out = yaml.dump({"v": coercible}, Dumper=SmartDumper,
                    default_flow_style=False, sort_keys=False)
    assert yaml.safe_load(out)["v"] == coercible


def test_smart_dumper_leaves_non_coercible_lookalike_unquoted():
    """'1e3' is NOT a YAML 1.1 float (it lacks a dot), so it stays a string
    without quoting: the resolver only quotes values that would truly coerce."""
    out = yaml.dump({"v": "1e3"}, Dumper=SmartDumper,
                    default_flow_style=False, sort_keys=False)
    assert "v: 1e3" in out
    assert yaml.safe_load(out)["v"] == "1e3"


# ---------------------------------------------------------------------------
# SmartDumper.represent_numpy_scalar — convert np scalars to native Python
# ---------------------------------------------------------------------------


def test_smart_dumper_numpy_int64_emits_as_python_int():
    """np.int64 should be emitted as a plain integer, not a tagged numpy obj.

    Note: the key name avoids YAML 1.1 boolean literals like 'n' (=no), 'y'
    (=yes), etc. — those would themselves be quoted by represent_str."""
    out = yaml.dump({"count": np.int64(42)}, Dumper=SmartDumper,
                    default_flow_style=False, sort_keys=False)
    assert "count: 42" in out
    assert "numpy" not in out


def test_smart_dumper_numpy_float64_emits_as_python_float():
    """np.float64 should be emitted as a plain float."""
    out = yaml.dump({"x": np.float64(3.14)}, Dumper=SmartDumper,
                    default_flow_style=False, sort_keys=False)
    assert "x: 3.14" in out
    assert "numpy" not in out


def test_smart_dumper_numpy_bool_emits_as_python_bool():
    """np.bool_ should be emitted as 'true' / 'false'."""
    out = yaml.dump({"flag": np.bool_(True)}, Dumper=SmartDumper,
                    default_flow_style=False, sort_keys=False)
    assert "flag: true" in out
    assert "numpy" not in out


# ---------------------------------------------------------------------------
# load_session_metadata / save_session_metadata round-trip
# ---------------------------------------------------------------------------


def test_load_session_metadata_returns_none_when_no_yaml(tmp_path):
    """No *_metadata.yaml in the directory → (None, None) without raising."""
    data, path = load_session_metadata(str(tmp_path))
    assert data is None
    assert path is None


def test_load_session_metadata_loads_first_match(tmp_path):
    """A *_metadata.yaml file is found and parsed."""
    md = tmp_path / "session1_metadata.yaml"
    md.write_text("Session:\n  session_id: '20260101_120000'\n")
    data, path = load_session_metadata(str(tmp_path))
    assert data == {"Session": {"session_id": "20260101_120000"}}
    assert path == md


def test_load_session_metadata_handles_yaml_error(tmp_path):
    """Malformed YAML → logs an error and returns (None, None)."""
    md = tmp_path / "session1_metadata.yaml"
    md.write_text("Session:\n  bad: : :\n")
    msgs: list[str] = []
    data, path = load_session_metadata(str(tmp_path), logger=msgs.append)
    assert data is None
    assert path is None
    assert any("Error loading metadata" in m for m in msgs)


def test_save_then_load_round_trip(tmp_path):
    """save_session_metadata followed by load_session_metadata should
    recover the original dict (modulo any SmartDumper formatting)."""
    payload = {
        "Session": {"session_id": "20260101_120000"},
        "Equipment": {"sync_LEDs": {"device_br": 115200}},
    }
    out_path = tmp_path / "test_metadata.yaml"
    save_session_metadata(payload, out_path, logger=lambda *_: None)
    assert out_path.is_file()

    data, found_path = load_session_metadata(str(tmp_path))
    assert data == payload
    assert found_path == out_path


# ---------------------------------------------------------------------------
# _set_block_field_in_position — canonical-order insertion
# ---------------------------------------------------------------------------


def test_set_block_field_existing_value_unchanged_returns_false():
    """Already-present field with same value → no-op, returns False."""
    block = {"a": 1, "b": 2}
    changed = _set_block_field_in_position(block, "a", 1, after_key="b")
    assert changed is False
    assert list(block.keys()) == ["a", "b"]


def test_set_block_field_existing_key_value_change_returns_true():
    """Already-present key with different value → updates in place, True."""
    block = {"a": 1, "b": 2}
    changed = _set_block_field_in_position(block, "a", 99, after_key="b")
    assert changed is True
    assert block["a"] == 99
    # Order preserved
    assert list(block.keys()) == ["a", "b"]


def test_set_block_field_inserts_after_anchor_key():
    """New field is inserted immediately after `after_key`."""
    block = {"a": 1, "b": 2, "c": 3}
    changed = _set_block_field_in_position(block, "new", 99, after_key="b")
    assert changed is True
    # Expected order: a, b, new, c
    assert list(block.keys()) == ["a", "b", "new", "c"]
    assert block["new"] == 99


def test_set_block_field_appends_when_anchor_missing():
    """If `after_key` is not in the block, new field goes to the end."""
    block = {"a": 1, "b": 2}
    changed = _set_block_field_in_position(block, "new", 99, after_key="zzzz")
    assert changed is True
    assert list(block.keys())[-1] == "new"


# ---------------------------------------------------------------------------
# sync_equipment_dynamic_fields — full reconciliation
# ---------------------------------------------------------------------------


def _equip_fixture() -> dict:
    """Builds a metadata dict with the three equipment blocks."""
    return {
        "Equipment": {
            "sync_LEDs": {"device_br": 115200, "device_port": "COM3"},
            "audio_Avisoft": {
                "device_sn": "abc123",
                "device_sync": False,
                "device_sr": 250000,
            },
            "video_Loopbio": {
                "device_count": 1,
                "device_sr": 100,
                "device_sr_calibration": 30,
                "sensor_model": "blackfly",
                "sensor_lens": "50mm",
                "sensor_sn": ["111"],
                "sensor_count": 1,
                "sensor_exposure_time": [5000],
                "sensor_gain": [1.0],
                "output_file_extension": "mp4",
                "output_file_codec": "h264",
            },
        },
    }


def test_sync_equipment_returns_false_when_metadata_not_dict():
    """Defensive guard: non-dict metadata_settings → False without raising."""
    assert sync_equipment_dynamic_fields("not a dict", {}) is False


def test_sync_equipment_returns_false_when_no_equipment_block():
    """No 'Equipment' key in the metadata → no-op, returns False."""
    md = {"Session": {"session_id": "x"}}
    assert sync_equipment_dynamic_fields(md, {}) is False


def test_sync_equipment_writes_arduino_sync_port():
    """A new arduino_sync_port string lands in Equipment.sync_LEDs.device_port."""
    md = _equip_fixture()
    exp = {"arduino_sync_port": "COM7"}
    changed = sync_equipment_dynamic_fields(md, exp)
    assert changed is True
    assert md["Equipment"]["sync_LEDs"]["device_port"] == "COM7"


def test_sync_equipment_writes_audio_settings():
    """audio_devices.fabtast and audio.usgh_devices_sync flow into Avisoft block."""
    md = _equip_fixture()
    exp = {
        "audio": {
            "usgh_devices_sync": True,
            "devices": {"fabtast": 300000},
        },
    }
    changed = sync_equipment_dynamic_fields(md, exp)
    assert changed is True
    assert md["Equipment"]["audio_Avisoft"]["device_sync"] is True
    assert md["Equipment"]["audio_Avisoft"]["device_sr"] == 300000


def test_sync_equipment_writes_video_settings_with_codec_long_name():
    """recording_codec='hq' is translated to its Loopbio long-name form."""
    md = _equip_fixture()
    exp = {
        "video": {
            "general": {
                "recording_frame_rate": 150,
                "calibration_frame_rate": 60,
                "recording_codec": "hq",
                "expected_cameras": ["222", "111"],
            },
            "cameras_config": {
                "111": {"exposure_time": 5000, "gain": 1.0},
                "222": {"exposure_time": 6000, "gain": 1.5},
            },
        },
    }
    changed = sync_equipment_dynamic_fields(md, exp)
    assert changed is True
    vl = md["Equipment"]["video_Loopbio"]
    assert vl["device_sr"] == 150
    assert vl["device_sr_calibration"] == 60
    assert vl["output_file_codec"] == "nvenc-slow-yuv420"
    assert vl["sensor_count"] == 2
    # sensor_sn comes back ascending — and digit-only strings become ints
    assert vl["sensor_sn"] == [111, 222]
    # Per-camera exposures / gains follow the sorted serial order
    assert vl["sensor_exposure_time"] == [5000, 6000]
    assert vl["sensor_gain"] == [1.0, 1.5]


def test_sync_equipment_sorts_serials_numerically_not_lexicographically():
    """Serials of differing digit-length must sort numerically (2, 9, 10), not
    lexicographically (10, 2, 9); the exposure / gain lists must stay aligned
    with their serial under that order."""
    md = _equip_fixture()
    exp = {
        "video": {
            "general": {
                "recording_frame_rate": 150,
                "expected_cameras": ["9", "10", "2"],
            },
            "cameras_config": {
                "2":  {"exposure_time": 2000, "gain": 0.2},
                "9":  {"exposure_time": 9000, "gain": 0.9},
                "10": {"exposure_time": 1000, "gain": 1.0},
            },
        },
    }
    sync_equipment_dynamic_fields(md, exp)
    vl = md["Equipment"]["video_Loopbio"]
    assert vl["sensor_sn"] == [2, 9, 10]
    assert vl["sensor_exposure_time"] == [2000, 9000, 1000]
    assert vl["sensor_gain"] == [0.2, 0.9, 1.0]


def test_sync_equipment_no_op_when_already_current():
    """If every field already equals the live setting, returns False."""
    md = _equip_fixture()
    md["Equipment"]["sync_LEDs"]["device_port"] = "COM7"
    md["Equipment"]["audio_Avisoft"]["device_sr"] = 250000
    exp = {
        "arduino_sync_port": "COM7",
        "audio": {"devices": {"fabtast": 250000}},
    }
    changed = sync_equipment_dynamic_fields(md, exp)
    assert changed is False


def test_sync_equipment_skips_missing_block():
    """If the metadata has only sync_LEDs, the audio_Avisoft writes are no-ops."""
    md = {"Equipment": {"sync_LEDs": {"device_br": 115200}}}
    exp = {
        "arduino_sync_port": "COM7",
        "audio": {"devices": {"fabtast": 250000}},
    }
    changed = sync_equipment_dynamic_fields(md, exp)
    # sync_LEDs got the port; audio_Avisoft was absent, so no field landed there
    assert changed is True
    assert md["Equipment"]["sync_LEDs"]["device_port"] == "COM7"
    assert "audio_Avisoft" not in md["Equipment"]


# ---------------------------------------------------------------------------
# set_sync_LEDs_device_port — repositioning helper
# ---------------------------------------------------------------------------


def test_set_sync_LEDs_device_port_reorders_existing_field():
    """If device_port is already present (last), it moves to right after device_br."""
    md = {"Equipment": {"sync_LEDs": {
        "device_br": 115200,
        "software": "v1",
        "device_port": "OLD",
    }}}
    set_sync_LEDs_device_port(md, "COM9")
    keys = list(md["Equipment"]["sync_LEDs"].keys())
    assert keys == ["device_br", "device_port", "software"]
    assert md["Equipment"]["sync_LEDs"]["device_port"] == "COM9"


def test_set_sync_LEDs_device_port_inserts_when_missing():
    """device_port absent → inserted right after device_br."""
    md = {"Equipment": {"sync_LEDs": {
        "device_br": 115200,
        "software": "v1",
    }}}
    set_sync_LEDs_device_port(md, "COM9")
    keys = list(md["Equipment"]["sync_LEDs"].keys())
    assert keys == ["device_br", "device_port", "software"]


def test_set_sync_LEDs_device_port_no_op_when_block_missing():
    """sync_LEDs block missing → no changes, no error."""
    md = {"Equipment": {}}
    set_sync_LEDs_device_port(md, "COM9")
    assert md == {"Equipment": {}}


def test_set_sync_LEDs_device_port_no_op_when_metadata_not_dict():
    """Defensive: non-dict metadata → no-op, no error."""
    set_sync_LEDs_device_port("garbage", "COM9")  # must not raise
