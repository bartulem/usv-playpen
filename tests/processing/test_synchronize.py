"""
@author: bartulem
Mock-based tests for synchronize_files.Synchronizer.

The class wraps four big orchestration methods around camera frame counts,
audio WAV cropping, e-phys sync channels, and Arduino IPI matching. We
exercise the constructor, the static helpers (_build_led_px_dict,
find_lsb_changes), and the early skip / file-missing paths in the four
public methods.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from usv_playpen.processing.synchronize_files import Synchronizer


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def processing_settings():
    """Loads processing_settings.json from the package."""
    import usv_playpen
    package_dir = Path(usv_playpen.__file__).parent
    with (package_dir / '_parameter_settings' / 'processing_settings.json').open('r') as f:
        return json.load(f)


def _make_sync(tmp_path, processing_settings) -> Synchronizer:
    return Synchronizer(
        root_directory=str(tmp_path),
        input_parameter_dict=processing_settings,
        message_output=lambda *_a, **_kw: None,
    )


# ---------------------------------------------------------------------------
# _build_led_px_dict — pure dict constructor
# ---------------------------------------------------------------------------


def test_build_led_px_dict_returns_distinct_dict_per_call():
    """Each call returns a fresh dict (not a shared mutable singleton),
    so two Synchronizer instances cannot accidentally share state."""
    d1 = Synchronizer._build_led_px_dict()
    d2 = Synchronizer._build_led_px_dict()
    assert d1 == d2  # equal contents
    assert d1 is not d2  # but distinct objects
    # Mutating one must not affect the other.
    d1['current']['21241563']['LED_top'] = [0, 0]
    assert d2['current']['21241563']['LED_top'] != [0, 0]


def test_build_led_px_dict_has_expected_keys():
    """Every date-fenced key has at least one camera serial number, and
    every camera entry has the three LED-position keys."""
    d = Synchronizer._build_led_px_dict()
    for date_key, cameras in d.items():
        assert isinstance(date_key, str)
        assert len(cameras) >= 1
        for serial, leds in cameras.items():
            assert serial.isdigit()
            assert {"LED_top", "LED_middle", "LED_bottom"} <= set(leds.keys())
            for pos in leds.values():
                assert len(pos) == 2  # (x, y)


# ---------------------------------------------------------------------------
# Synchronizer.__init__
# ---------------------------------------------------------------------------


def test_synchronizer_init_with_full_kwargs(processing_settings, tmp_path):
    """Both args provided → uses them directly, no JSON load."""
    sync = _make_sync(tmp_path, processing_settings)
    assert sync.root_directory == str(tmp_path)
    assert "find_audio_sync_trains" in sync.input_parameter_dict
    # led_px_dict was built and attached
    assert "current" in sync.led_px_dict


def test_synchronizer_init_loads_defaults_when_input_dict_none(tmp_path):
    """input_parameter_dict=None → loads processing_settings.json defaults."""
    sync = Synchronizer(
        root_directory=str(tmp_path),
        message_output=lambda *_a, **_kw: None,
    )
    assert sync.root_directory == str(tmp_path)
    assert "find_audio_sync_trains" in sync.input_parameter_dict


# ---------------------------------------------------------------------------
# find_lsb_changes (static method, lsb_bool=True/False, edge cases)
# ---------------------------------------------------------------------------


def test_find_lsb_changes_lsb_true_returns_correct_endpoints():
    """LSB pattern: 0,0,1,0,0,...,1,0,0,1 → off→on rising edges. The
    function returns (start_first_relevant_sample, end_last_relevant_sample,
    largest_break_duration, ttl_break_end_samples, largest_break_end_hop).
    """
    # Construct an array with N=4 rising edges in the LSB; the largest
    # gap is between edges 0→1 (gap=10) and 1→2 (gap=2), 2→3 (gap=2).
    # ttl_break_end_samples = positions where LSB rises. After
    # largest_break_end_hop=1 we have N=2 frames remaining.
    sound = np.zeros(64, dtype=np.int16)
    # rises at 5, 15, 17, 19
    for pos in (5, 15, 17, 19):
        sound[pos:pos+1] = 1
    out = Synchronizer.find_lsb_changes(
        relevant_array=sound, lsb_bool=True, total_frame_number=2,
    )
    start, end, largest_break, ttl_breaks, hop = out
    # Should have detected 4 rising edges → 3 inter-edge gaps; largest gap
    # is between edge 0 and edge 1 (10 samples).
    assert ttl_breaks.size >= 3
    assert largest_break == 10
    # start / end are computed from ttl_breaks[hop:] so they should be ints
    assert isinstance(start, int)
    assert isinstance(end, int)
    assert end > start


def test_find_lsb_changes_returns_none_when_total_frames_exceeds_capacity():
    """If total_frame_number is larger than the available rising edges past
    the largest break, return (None, None, ...)."""
    # Two rising edges only — total_frame_number=10 cannot fit.
    sound = np.zeros(20, dtype=np.int16)
    sound[5] = 1
    sound[15] = 1
    out = Synchronizer.find_lsb_changes(
        relevant_array=sound, lsb_bool=True, total_frame_number=10,
    )
    start, end, largest_break, ttl_breaks, hop = out
    assert start is None
    assert end is None
    # The duration of the largest break is still computed and returned
    assert largest_break > 0


def test_find_lsb_changes_lsb_false_uses_raw_signal():
    """lsb_bool=False uses raw differences instead of LSB-only — useful for
    e-phys sync channels where the sync bit is whole-int."""
    raw = np.zeros(64, dtype=np.int16)
    # rises at 5, 25, 27, 29 (largest gap = 20 between edges 0 and 1)
    raw[5:25] = 100
    raw[25:27] = 0
    raw[27:29] = 100
    raw[29:] = 0
    out = Synchronizer.find_lsb_changes(
        relevant_array=raw, lsb_bool=False, total_frame_number=2,
    )
    start, end, largest_break, ttl_breaks, hop = out
    assert ttl_breaks.size >= 2
    assert largest_break > 0


# ---------------------------------------------------------------------------
# validate_ephys_video_sync — file-missing branches
# ---------------------------------------------------------------------------


def test_validate_ephys_video_sync_no_camera_json_raises(processing_settings, tmp_path,
                                                          mocker):
    """No camera_frame_count_dict.json under root_directory → FileNotFoundError
    with an actionable label."""
    mocker.patch("usv_playpen.processing.synchronize_files.smart_wait")
    sync = _make_sync(tmp_path, processing_settings)
    with pytest.raises(FileNotFoundError, match="camera frame count JSON"):
        sync.validate_ephys_video_sync()


# ---------------------------------------------------------------------------
# crop_wav_files_to_video — file-missing branches
# ---------------------------------------------------------------------------


def test_crop_wav_files_to_video_raises_when_camera_json_missing(processing_settings,
                                                                  tmp_path, mocker):
    """Missing video/<...>_camera_frame_count_dict.json → a clear
    FileNotFoundError from first_match_or_raise (naming the missing pattern),
    instead of the previous bare IndexError from sorted(...)[0]."""
    (tmp_path / "video").mkdir()
    mocker.patch("usv_playpen.processing.synchronize_files.smart_wait")
    sync = _make_sync(tmp_path, processing_settings)
    with pytest.raises(FileNotFoundError, match=r"camera_frame_count_dict"):
        sync.crop_wav_files_to_video()


# ---------------------------------------------------------------------------
# find_audio_sync_trains — early-fail branches
# ---------------------------------------------------------------------------


def test_find_audio_sync_trains_raises_when_camera_json_missing(processing_settings,
                                                                 tmp_path, mocker):
    """Missing camera frame count JSON → DataLoader returns an empty dict but
    the helper subsequently calls first_match_or_raise which raises. We just
    confirm the function refuses to silently no-op."""
    mocker.patch("usv_playpen.processing.synchronize_files.smart_wait")
    sync = _make_sync(tmp_path, processing_settings)
    # Empty wave_data_dict is fine (DataLoader returns {} when the path
    # doesn't exist), but the next step queries the JSON, which raises.
    mocker.patch("usv_playpen.processing.synchronize_files.DataLoader",
                 return_value=MagicMock(load_wavefile_data=lambda: {}))
    with pytest.raises(FileNotFoundError, match="camera frame count JSON"):
        sync.find_audio_sync_trains()


# ---------------------------------------------------------------------------
# find_video_sync_trains — no-video-subdirs path
# ---------------------------------------------------------------------------


def test_find_video_sync_trains_no_video_dir_returns_empty(processing_settings,
                                                            tmp_path, mocker):
    """When <root>/video/ has no qualifying subdirectories, the method
    iterates through nothing and returns (empty array, empty dict)."""
    (tmp_path / "video").mkdir()
    # Only a single underscore-named dir, which the loop skips.
    (tmp_path / "video" / "session_data").mkdir()
    mocker.patch("usv_playpen.processing.synchronize_files.smart_wait")
    sync = _make_sync(tmp_path, processing_settings)
    ipi_starts, sync_dict = sync.find_video_sync_trains(camera_fps=[150.0],
                                                         total_frame_number=100)
    assert isinstance(ipi_starts, np.ndarray)
    assert ipi_starts.size == 0
    assert sync_dict == {}
