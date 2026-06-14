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

import glob
import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from scipy.io import wavfile

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


def _write_video_sync_fixture(root, serial) -> int:
    """
    Description
    -----------
    Write the LED-brightness `sync_px_*` memmap + a dummy `.mp4` + a CoolTerm
    Arduino-IPI log into a session root, so `find_video_sync_trains` can run its
    full detection/match pipeline without decoding a real video (the memmap
    pre-exists, so `gather_px_information` is skipped). The pulse train is four
    equal 45-frame dark gaps -> ~300 ms IPIs at 150 fps.

    Parameters
    ----------
    root (pathlib.Path)
        Session root directory.
    serial (str)
        Sync-camera serial number.

    Returns
    -------
    n_frames (int)
        Number of frames in the synthetic LED memmap.
    """

    video_name = f"{serial}-20260421185830.mp4"
    cam_dir = root / "video" / "20260421185830" / serial
    cam_dir.mkdir(parents=True)
    (cam_dir / video_name).write_bytes(b"\x00")          # dummy; never decoded
    sync_dir = root / "sync"
    sync_dir.mkdir(exist_ok=True)

    segs = [(250, 50), (5, 45), (250, 45), (5, 45), (250, 45),
            (5, 45), (250, 45), (5, 45), (250, 50)]
    brightness = np.concatenate([np.full(n, v, dtype=np.uint8) for v, n in segs])
    n_frames = brightness.size
    mm = np.memmap(sync_dir / f"sync_px_{video_name[:-4]}", dtype=np.uint8,
                   mode="w+", shape=(n_frames, 3, 3))
    mm[:] = brightness[:, None, None]                     # all 3 LEDs x RGB = brightness
    mm.flush()
    # exactly four IPIs (== the four detected dark gaps) so a single arduino
    # window matches; a longer log would ravel multiple matching windows and
    # inflate the returned sequence length.
    (sync_dir / "CoolTerm Capture test.txt").write_text(
        "header0\nheader1\nheader2\n" + "\n".join(["300"] * 4) + "\n"
    )
    return n_frames


def test_find_video_sync_trains_matches_led_pulse_train(tmp_path, processing_settings, mocker):
    """
    Description
    -----------
    End-to-end happy path for `find_video_sync_trains`: the LED memmap +
    CoolTerm log fixture drives pulse detection + sequence matching, returning
    non-empty start frames and a per-camera sequence dict.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Per-test session root.
    processing_settings (dict)
        Package processing-settings fixture.
    mocker (pytest_mock.MockerFixture)
        No-ops the interactive `smart_wait`.

    Returns
    -------
    None
    """

    mocker.patch("usv_playpen.processing.synchronize_files.smart_wait")
    serial = processing_settings["synchronize_files"]["Synchronizer"]["find_video_sync_trains"]["sync_camera_serial_num"][0]
    n_frames = _write_video_sync_fixture(tmp_path, serial)

    sync = _make_sync(tmp_path, processing_settings)
    ipi_starts, sync_seq = sync.find_video_sync_trains(camera_fps=[150.0],
                                                       total_frame_number=n_frames)
    assert len(sync_seq) == 1
    assert any(Path(k).name == serial for k in sync_seq)   # keyed by camera dir
    assert ipi_starts.size > 0


def test_find_audio_sync_trains_matches_video(tmp_path, processing_settings, mocker):
    """
    Description
    -----------
    End-to-end happy path for `find_audio_sync_trains`: the video LED fixture
    (four ~300 ms IPIs) plus a `_ch02` cropped WAV whose LSB carries four
    matching ~300 ms OFF gaps. The audio IPI durations match the video
    sequence within tolerance, so the returned discrepancy dict carries the
    per-file `ipi_discrepancy_ms` result.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Per-test session root.
    processing_settings (dict)
        Package processing-settings fixture.
    mocker (pytest_mock.MockerFixture)
        No-ops the interactive `smart_wait`.

    Returns
    -------
    None
    """

    mocker.patch("usv_playpen.processing.synchronize_files.smart_wait")
    serial = processing_settings["synchronize_files"]["Synchronizer"]["find_video_sync_trains"]["sync_camera_serial_num"][0]
    n_frames = _write_video_sync_fixture(tmp_path, serial)

    (tmp_path / "video" / "sess_camera_frame_count_dict.json").write_text(json.dumps({
        serial: [n_frames, 150.0],
        "total_frame_number_least": n_frames,
        "total_video_time_least": n_frames / 150.0,
    }))

    # _ch02 cropped WAV: LSB pulse train with four ~300 ms OFF gaps at 250 kHz
    # (75000 OFF samples -> 300.004 ms, rounds to the video's 300 ms IPIs).
    sr = 250000
    on = np.ones(1000, dtype=np.int16)        # LSB = 1 (pulse ON)
    off = np.zeros(75000, dtype=np.int16)     # LSB = 0 (IPI gap)
    data = np.concatenate([on] + [np.concatenate([off, on]) for _ in range(4)])
    cropped = tmp_path / "audio" / "cropped_to_video"
    cropped.mkdir(parents=True)
    wavfile.write(cropped / "m_240101_ch02_cropped_to_video.wav", sr, data)

    sync = _make_sync(tmp_path, processing_settings)
    result = sync.find_audio_sync_trains()

    key = "m_240101_ch02_cropped_to_video"
    assert key in result
    assert "ipi_discrepancy_ms" in result[key]


def test_validate_ephys_video_sync_writes_changepoints(tmp_path, processing_settings, mocker):
    """
    Description
    -----------
    End-to-end happy path for `validate_ephys_video_sync`: a synthetic
    SpikeGLX `*.ap.bin` (5 channels int16, last = SY sync TTL) carries a
    camera-frame pulse train with one large break followed by
    `total_frame_number_least + 1` evenly-spaced edges. The recovered
    tracking window's duration is made to equal `total_video_time_least`
    (zero divergence), so the method writes the `changepoints_info_*.json`
    into the mirrored EPHYS tree.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Per-test temp dir; session is rooted at `<tmp>/Data/<id>` so the
        EPHYS mirror resolves to `<tmp>/EPHYS`.
    processing_settings (dict)
        Package processing-settings fixture.
    mocker (pytest_mock.MockerFixture)
        No-ops the interactive `smart_wait`.

    Returns
    -------
    None
    """

    import configparser

    mocker.patch("usv_playpen.processing.synchronize_files.smart_wait")

    # SR for the headstage SN we put in the synthetic .meta (must be in the ini).
    import usv_playpen
    ini = configparser.ConfigParser()
    ini.read(Path(usv_playpen.__file__).parent / "_config" / "calibrated_sample_rates_imec.ini")
    headstage_sn = "23280356"
    sr = float(ini["CalibratedHeadStages"][headstage_sn])

    total_frames = 5
    spacing = 6000
    # rising-edge sample positions: 3 pre-break edges, a 50000-sample break,
    # then total_frames + 1 post-break edges spaced `spacing` apart.
    pre = [100, 200, 300]
    first_post = 50300
    post = [first_post + i * spacing for i in range(total_frames + 1)]
    edges = pre + post
    n_samples = edges[-1] + 10

    # sync channel: isolated 5-sample-wide pulses -> one rising edge each.
    sync_ch = np.zeros(n_samples, dtype=np.int16)
    for e in edges:
        sync_ch[e + 1:e + 6] = 1000

    # interleave into a (n_samples, 5) int16 frame so reshape((5, n), 'F')[-1]
    # recovers the sync channel as the last row.
    n_ch = 5
    frame = np.zeros((n_samples, n_ch), dtype=np.int16)
    frame[:, -1] = sync_ch

    root = tmp_path / "Data" / "20250919_155842"
    imec_dir = root / "ephys" / "imec0"
    imec_dir.mkdir(parents=True)
    frame.tofile(imec_dir / "20250919_155842.imec0.ap.bin")
    (imec_dir / "20250919_155842.imec0.ap.meta").write_text(
        f"acqApLfSy=4,0,1\nimDatHs_sn={headstage_sn}\nimDatPrb_sn=22420014283\n"
    )

    # tracking window = ttl[hop+total]-ttl[hop] = total_frames*spacing samples;
    # set the video duration to the same so divergence is ~0.
    tracking_samples = total_frames * spacing
    (root / "video").mkdir()
    (root / "video" / "sess_camera_frame_count_dict.json").write_text(json.dumps({
        "total_frame_number_least": total_frames,
        "total_video_time_least": tracking_samples / sr,
    }))

    sync = Synchronizer(
        root_directory=str(root),
        input_parameter_dict=processing_settings,
        message_output=lambda *_a, **_k: None,
    )
    sync.validate_ephys_video_sync()

    cp = root.parent.parent / "EPHYS" / "20250919_imec0" / "changepoints_info_20250919_imec0.json"
    assert cp.is_file(), "changepoints JSON was not written to the EPHYS mirror"
    info = json.loads(cp.read_text())
    rec = info["20250919_155842.imec0"]
    assert rec["tracking_start_end"] == [50301, 50301 + tracking_samples]
    assert rec["total_num_channels"] == 5


def _processing_settings_full():
    """
    Description
    -----------
    Load the package ``processing_settings.json`` fresh as a mutable dict so a
    test can override sub-keys before handing it to ``Synchronizer`` without
    touching the on-disk settings.

    Parameters
    ----------

    Returns
    -------
    settings (dict)
        The parsed processing-settings dict.
    """

    path = glob.glob("**/processing_settings.json", recursive=True)[0]
    return json.loads(Path(path).read_text())


def test_crop_wav_files_to_video_single_device(tmp_path, mocker):
    """
    Description
    -----------
    End-to-end single-device crop: a triggerbox-channel WAV whose LSB carries a
    camera-frame TTL train (with one large recording break) is parsed by
    ``find_lsb_changes`` to recover the tracking window, the per-frame
    audio-sample offsets and sync-info JSON are written, and every original WAV
    is trimmed to that window via the bundled ``static_sox`` into
    ``audio/cropped_to_video/``.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Per-test temp directory used as the session root.
    mocker (pytest_mock.MockerFixture)
        Used to no-op the interactive ``smart_wait``.

    Returns
    -------
    None
    """

    mocker.patch("usv_playpen.processing.synchronize_files.smart_wait")

    root = tmp_path
    (root / "video").mkdir()
    (root / "video" / "sess_camera_frame_count_dict.json").write_text(
        json.dumps({
            "total_frame_number_least": 5,
            "total_video_time_least": 0.001,
        })
    )
    (root / "sync").mkdir()
    audio_orig = root / "audio" / "original"
    audio_orig.mkdir(parents=True)

    # LSB rising-edge train: three pre-break edges, a large gap, then six
    # post-break edges (>= total_frame_number) so find_lsb_changes resolves a
    # tracking window after the largest break.
    edges = [10, 20, 30, 200, 210, 220, 230, 240, 250]
    data = np.zeros(320, dtype=np.int16)
    for e in edges:
        data[e] = 1  # odd value -> LSB high -> 0->1 rising edge at e
    wavfile.write(audio_orig / "m_240101_ch04.wav", 250000, data)

    settings = _processing_settings_full()
    crop = settings["synchronize_files"]["Synchronizer"]["crop_wav_files_to_video"]
    crop["device_receiving_input"] = "m"
    crop["triggerbox_ch_receiving_input"] = 4

    sync = Synchronizer(
        root_directory=str(root),
        input_parameter_dict=settings,
        message_output=lambda *_a, **_k: None,
    )
    sync.crop_wav_files_to_video()

    # sync-info JSON written with a resolved (non-zero) tracking window.
    info = json.loads((root / "audio" / "audio_triggerbox_sync_info.json").read_text())
    assert info["m"]["duration_samples"] > 0
    # tracking window spans the first to last post-break TTL edge (200..250).
    assert info["m"]["start_first_recorded_frame"] == 200
    assert info["m"]["end_last_recorded_frame"] == 250
    # per-frame offsets file + the static_sox-trimmed output WAV.
    assert (root / "sync" / "m_video_frames_in_audio_samples.txt").is_file()
    cropped = list((root / "audio" / "cropped_to_video").glob("*_cropped_to_video.wav"))
    assert cropped, "static_sox did not produce a cropped WAV"
