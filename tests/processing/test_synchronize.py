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
from scipy.io.wavfile import read as _real_wavfile_read

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


def test_find_audio_sync_trains_warns_on_cross_device_pulse_count_mismatch(
    tmp_path, processing_settings, mocker
):
    """When the master and slave audio devices detect a DIFFERENT number of IPI
    sync pulses (a dropped/extra pulse), the cross-device start-sample subtraction
    must not crash with a broadcast ValueError -- it logs a clear count-mismatch
    warning and compares only the aligned prefix."""
    mocker.patch("usv_playpen.processing.synchronize_files.smart_wait")
    serial = processing_settings["synchronize_files"]["Synchronizer"]["find_video_sync_trains"]["sync_camera_serial_num"][0]
    n_frames = _write_video_sync_fixture(tmp_path, serial)
    (tmp_path / "video" / "sess_camera_frame_count_dict.json").write_text(json.dumps({
        serial: [n_frames, 150.0],
        "total_frame_number_least": n_frames,
        "total_video_time_least": n_frames / 150.0,
    }))

    sr = 250000
    on = np.ones(1000, dtype=np.int16)
    off = np.zeros(75000, dtype=np.int16)
    cropped = tmp_path / "audio" / "cropped_to_video"
    cropped.mkdir(parents=True)
    # master: 4 IPI gaps; slave: 3 IPI gaps -> cross-device pulse-count mismatch.
    data_m = np.concatenate([on] + [np.concatenate([off, on]) for _ in range(4)])
    data_s = np.concatenate([on] + [np.concatenate([off, on]) for _ in range(3)])
    wavfile.write(cropped / "m_240101_ch02_cropped_to_video.wav", sr, data_m)
    wavfile.write(cropped / "s_240101_ch02_cropped_to_video.wav", sr, data_s)

    msgs: list[str] = []
    sync = Synchronizer(
        root_directory=str(tmp_path),
        input_parameter_dict=processing_settings,
        message_output=msgs.append,
    )
    sync.find_audio_sync_trains()   # must not raise a broadcast ValueError

    assert any("DIFFERENT number of IPI sync pulses" in m for m in msgs)


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


def test_gather_px_information_writes_led_memmap(tmp_path, processing_settings):
    """
    Description
    -----------
    `gather_px_information` decodes a real (generated) video, locates the three
    sync-LED centroids via the Otsu/moments path, and writes the per-frame
    `sync_px_*` LED-intensity memmap. A 15-frame 1280x720 clip with bright
    blobs at the camera's known LED coordinates exercises the cv2 read +
    centroid + memmap-write path.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Per-test session root.
    processing_settings (dict)
        Package processing-settings fixture.

    Returns
    -------
    None
    """

    import cv2

    serial = "21372315"
    coords = list(Synchronizer._build_led_px_dict()["current"][serial].values())  # [y, x] each
    h, w = 720, 1280
    video_name = f"{serial}-ts.mp4"
    video_path = tmp_path / video_name

    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (w, h))
    if not writer.isOpened():
        pytest.skip("cv2 mp4v VideoWriter unavailable in this environment")
    for _ in range(15):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        for (y, x) in coords:
            frame[y - 5:y + 6, x - 5:x + 6] = 255      # bright LED blob
        writer.write(frame)
    writer.release()

    (tmp_path / "sync").mkdir()
    sync = _make_sync(tmp_path, processing_settings)
    sync.gather_px_information(video_of_interest=str(video_path), sync_camera_fps=10,
                              camera_id=serial, video_name=video_name, total_frame_number=10)

    mm = np.memmap(tmp_path / "sync" / f"sync_px_{serial}-ts", dtype=np.uint8,
                   mode="r", shape=(10, 3, 3))
    assert mm.shape == (10, 3, 3)
    assert int(mm.max()) > 0      # LED pixels sampled as bright


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


def _triggerbox_wav(path, edges, sr=250000):
    """
    Description
    -----------
    Write a triggerbox-channel WAV whose LSB carries a camera-frame TTL train:
    a single odd sample at each edge index produces a 0->1 LSB rising edge for
    `find_lsb_changes` (lsb_bool=True).

    Parameters
    ----------
    path (pathlib.Path)
        Output WAV path.
    edges (list[int])
        Sample indices of the rising edges.
    sr (int)
        Sampling rate.

    Returns
    -------
    None
    """

    data = np.zeros(edges[-1] + 10, dtype=np.int16)
    for e in edges:
        data[e] = 1
    wavfile.write(path, sr, data)


def test_crop_wav_files_to_video_both_devices_tempo_adjust(tmp_path, mocker):
    """
    Description
    -----------
    Two-device ("both") crop where the master tracking window is longer than the
    slave's: exercises the `m_longer` branch that resamples (SoX `tempo`) the
    master files down to the slave duration, writes per-device offset files and
    the sync-info JSON for both devices, and crops every original WAV.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Per-test session root.
    mocker (pytest_mock.MockerFixture)
        No-ops the interactive `smart_wait`.

    Returns
    -------
    None
    """

    mocker.patch("usv_playpen.processing.synchronize_files.smart_wait")
    root = tmp_path
    (root / "video").mkdir()
    (root / "video" / "sess_camera_frame_count_dict.json").write_text(json.dumps({
        "total_frame_number_least": 5,
        "total_video_time_least": 0.001,
    }))
    (root / "sync").mkdir()
    audio_orig = root / "audio" / "original"
    audio_orig.mkdir(parents=True)

    # master: post-break edges spaced 10 -> window 200..250 (longer);
    # slave: spaced 5 -> window 200..225 (shorter) -> m_longer branch.
    _triggerbox_wav(audio_orig / "m_240101_ch04.wav",
                    [10, 20, 30, 200, 210, 220, 230, 240, 250])
    _triggerbox_wav(audio_orig / "s_240101_ch04.wav",
                    [10, 20, 30, 200, 205, 210, 215, 220, 225])

    settings = _processing_settings_full()
    crop = settings["synchronize_files"]["Synchronizer"]["crop_wav_files_to_video"]
    crop["device_receiving_input"] = "both"
    crop["triggerbox_ch_receiving_input"] = 4

    sync = Synchronizer(
        root_directory=str(root),
        input_parameter_dict=settings,
        message_output=lambda *_a, **_k: None,
    )
    sync.crop_wav_files_to_video()

    info = json.loads((root / "audio" / "audio_triggerbox_sync_info.json").read_text())
    assert info["m"]["duration_samples"] > info["s"]["duration_samples"]   # m_longer path
    assert (root / "sync" / "m_video_frames_in_audio_samples.txt").is_file()
    assert (root / "sync" / "s_video_frames_in_audio_samples.txt").is_file()
    cropped = list((root / "audio" / "cropped_to_video").glob("*_cropped_to_video.wav"))
    assert len(cropped) >= 2      # both devices cropped


# ---------------------------------------------------------------------------
# find_audio_sync_trains — pulse-sequence shape-mismatch reconciliation
# ---------------------------------------------------------------------------


def _wire_audio_sync(sync, mocker, *, audio_durations_ms, audio_starts,
                     video_seq, video_frames, root, wave_keys=("m_240101_ch02_cropped_to_video.wav",),
                     sampling_rate=250000):
    """
    Description
    -----------
    Wire up ``find_audio_sync_trains`` so the audio/video pulse-sequence
    reconciliation block (source lines 926-1026) runs against fully crafted
    inputs. The expensive detectors are replaced: ``DataLoader`` yields the
    given wave keys, ``find_ipi_intervals`` returns the supplied audio IPI
    durations + start samples, and ``find_video_sync_trains`` returns the
    supplied video frames + a single-camera sequence dict (so the
    all-equal-sequence guard at line 926 passes trivially). A camera frame
    count JSON is written so the early ``first_match_or_raise`` succeeds.

    Parameters
    ----------
    sync (Synchronizer)
        Instance under test.
    mocker (pytest_mock.MockerFixture)
        Patch factory.
    audio_durations_ms (np.ndarray)
        IPI durations (ms) the mocked ``find_ipi_intervals`` returns.
    audio_starts (np.ndarray)
        Audio IPI start samples the mocked ``find_ipi_intervals`` returns.
    video_seq (np.ndarray)
        Per-camera video IPI duration sequence.
    video_frames (np.ndarray)
        Video IPI start frame indices.
    root (pathlib.Path)
        Session root.
    wave_keys (tuple[str])
        Wave-data dict keys (cropped WAV file names).
    sampling_rate (int)
        Reported WAV sampling rate.

    Returns
    -------
    video_key (str)
        The synthetic camera directory name used as the sequence-dict key.
    """

    (root / "video").mkdir(exist_ok=True)
    (root / "video" / "sess_camera_frame_count_dict.json").write_text(json.dumps({
        "21372315": [len(video_frames), 150.0],
        "total_frame_number_least": int(max(video_frames) + 1) if len(video_frames) else 1,
        "total_video_time_least": 1.0,
    }))

    wave_data_dict = {k: {"wav_data": np.zeros(8, dtype=np.int16), "sampling_rate": sampling_rate}
                      for k in wave_keys}
    mocker.patch("usv_playpen.processing.synchronize_files.DataLoader",
                 return_value=MagicMock(load_wavefile_data=lambda: wave_data_dict))

    video_key = "21372315"
    video_sync_sequence_dict = {video_key: np.asarray(video_seq)}
    mocker.patch.object(sync, "find_video_sync_trains",
                        return_value=(np.asarray(video_frames), video_sync_sequence_dict))
    mocker.patch.object(sync, "find_ipi_intervals",
                        return_value=(np.asarray(audio_durations_ms, dtype=float),
                                      np.asarray(audio_starts, dtype=np.int64)))
    return video_key


def test_find_audio_sync_trains_equal_shapes_within_tolerance(tmp_path, processing_settings, mocker):
    """
    Description
    -----------
    Equal audio/video pulse counts whose rounded durations agree within the
    12 ms tolerance: drives the ``n_a == n_v`` diff branch (line 938) through
    the successful-match tail that records ``ipi_discrepancy_ms`` and
    ``video_ipi_start_frames`` (lines 966, 982, 1011-1017). Audio sample
    starts are aligned to the video-frame timing so the discrepancy stays
    below tolerance.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Per-test session root.
    processing_settings (dict)
        Package processing-settings fixture.
    mocker (pytest_mock.MockerFixture)
        Patch factory; no-ops ``smart_wait``.

    Returns
    -------
    None
    """

    mocker.patch("usv_playpen.processing.synchronize_files.smart_wait")
    sync = _make_sync(tmp_path, processing_settings)
    camera_fps = 150.0
    video_frames = np.array([10, 20, 30, 40])
    # audio start samples placed exactly at video-frame times -> zero discrepancy.
    audio_starts = (video_frames / camera_fps * 250000).astype(np.int64)
    _wire_audio_sync(sync, mocker,
                     audio_durations_ms=np.array([300.0, 300.0, 300.0, 300.0]),
                     audio_starts=audio_starts,
                     video_seq=np.array([300, 300, 300, 300]),
                     video_frames=video_frames, root=tmp_path)

    # write the per-device video-frame-in-audio-samples map so the audio->video
    # frame-index reconciliation block (source lines 985-1008) runs: each audio
    # IPI start sample is attributed to the last recorded video frame before it.
    (tmp_path / "sync").mkdir(exist_ok=True)
    frame_samples = np.linspace(0, int(audio_starts.max()) + 5000, num=200).astype(np.int64)
    (tmp_path / "sync" / "m_video_frames_in_audio_samples.txt").write_text(
        "\n".join(str(int(s)) for s in frame_samples) + "\n"
    )

    # an existing audio/original dir so the acceptable-sync cleanup that deletes
    # it (source lines 1012-1014) actually runs and removes the directory.
    (tmp_path / "audio" / "original").mkdir(parents=True, exist_ok=True)

    result = sync.find_audio_sync_trains()
    key = "m_240101_ch02_cropped_to_video"
    assert "ipi_discrepancy_ms" in result[key]
    assert np.max(np.abs(result[key]["ipi_discrepancy_ms"])) < 12
    assert not (tmp_path / "audio" / "original").exists()   # cleaned up (line 1014)


def test_find_audio_sync_trains_two_devices_start_sample_difference(tmp_path, processing_settings, mocker):
    """
    Description
    -----------
    Two cropped WAV files (master + slave) so the second iteration runs the
    cross-device IPI-start-sample subtraction (source line 924) and the trailing
    master/slave start-sample difference summary (lines 1028-1031). The mocked
    ``find_ipi_intervals`` returns identical starts for both files so the
    difference reduces to zero.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Per-test session root.
    processing_settings (dict)
        Package processing-settings fixture.
    mocker (pytest_mock.MockerFixture)
        Patch factory; no-ops ``smart_wait``.

    Returns
    -------
    None
    """

    mocker.patch("usv_playpen.processing.synchronize_files.smart_wait")
    sync = _make_sync(tmp_path, processing_settings)
    camera_fps = 150.0
    video_frames = np.array([10, 20, 30, 40])
    audio_starts = (video_frames / camera_fps * 250000).astype(np.int64)
    _wire_audio_sync(sync, mocker,
                     audio_durations_ms=np.array([300.0, 300.0, 300.0, 300.0]),
                     audio_starts=audio_starts,
                     video_seq=np.array([300, 300, 300, 300]),
                     video_frames=video_frames, root=tmp_path,
                     wave_keys=("m_240101_ch02_cropped_to_video.wav",
                                "s_240101_ch02_cropped_to_video.wav"))
    result = sync.find_audio_sync_trains()
    # both per-file entries created; second iteration drove the subtraction.
    assert "m_240101_ch02_cropped_to_video" in result
    assert "s_240101_ch02_cropped_to_video" in result


def test_find_audio_sync_trains_audio_start_precedes_all_frames_nan(tmp_path, processing_settings, mocker):
    """
    Description
    -----------
    Equal audio/video pulse counts with a ``m_video_frames_in_audio_samples.txt``
    map whose every recorded video-frame start sample lies AFTER the first audio
    IPI start sample: drives the NaN-attribution guard (source lines 993-1003)
    where an audio start precedes all video frames and its frame index is marked
    NaN rather than crashing on ``max()`` of an empty slice.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Per-test session root.
    processing_settings (dict)
        Package processing-settings fixture.
    mocker (pytest_mock.MockerFixture)
        Patch factory; no-ops ``smart_wait``.

    Returns
    -------
    None
    """

    mocker.patch("usv_playpen.processing.synchronize_files.smart_wait")
    sync = _make_sync(tmp_path, processing_settings)
    camera_fps = 150.0
    video_frames = np.array([10, 20, 30, 40])
    audio_starts = (video_frames / camera_fps * 250000).astype(np.int64)
    _wire_audio_sync(sync, mocker,
                     audio_durations_ms=np.array([300.0, 300.0, 300.0, 300.0]),
                     audio_starts=audio_starts,
                     video_seq=np.array([300, 300, 300, 300]),
                     video_frames=video_frames, root=tmp_path)

    # all recorded frame-start samples lie AFTER the earliest audio IPI start ->
    # the first audio start has no preceding frame -> NaN-attribution branch.
    (tmp_path / "sync").mkdir(exist_ok=True)
    offset = int(audio_starts.min()) + 1000
    frame_samples = (offset + np.arange(200) * 10).astype(np.int64)
    (tmp_path / "sync" / "m_video_frames_in_audio_samples.txt").write_text(
        "\n".join(str(int(s)) for s in frame_samples) + "\n"
    )

    result = sync.find_audio_sync_trains()
    key = "m_240101_ch02_cropped_to_video"
    assert "ipi_discrepancy_ms" in result[key]


def test_find_audio_sync_trains_drop_first_audio_pulse(tmp_path, processing_settings, mocker):
    """
    Description
    -----------
    One extra audio pulse (n_a == n_v + 1) where dropping the FIRST audio pulse
    aligns the sequences within tolerance: drives the ``abs(n_a - n_v) == 1``
    /``n_a > n_v`` candidate loop and the "dropped first audio pulse" resolution
    (source lines 939-957). A spurious leading audio IPI is prepended.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Per-test session root.
    processing_settings (dict)
        Package processing-settings fixture.
    mocker (pytest_mock.MockerFixture)
        Patch factory; no-ops ``smart_wait``.

    Returns
    -------
    None
    """

    mocker.patch("usv_playpen.processing.synchronize_files.smart_wait")
    sync = _make_sync(tmp_path, processing_settings)
    camera_fps = 150.0
    video_seq = np.array([300, 300, 300])
    video_frames = np.array([20, 30, 40])
    # real audio aligned to video; a bogus leading pulse of wildly wrong duration.
    real_starts = (video_frames / camera_fps * 250000).astype(np.int64)
    audio_starts = np.concatenate([[1], real_starts])
    audio_durations = np.array([9999.0, 300.0, 300.0, 300.0])
    _wire_audio_sync(sync, mocker,
                     audio_durations_ms=audio_durations,
                     audio_starts=audio_starts,
                     video_seq=video_seq, video_frames=video_frames, root=tmp_path)
    result = sync.find_audio_sync_trains()
    key = "m_240101_ch02_cropped_to_video"
    assert "ipi_discrepancy_ms" in result[key]
    # dropped-first resolution -> three retained frames equal to video_frames.
    np.testing.assert_array_equal(result[key]["video_ipi_start_frames"], video_frames)


def test_find_audio_sync_trains_drop_last_audio_pulse(tmp_path, processing_settings, mocker):
    """
    Description
    -----------
    One extra audio pulse (n_a == n_v + 1) where dropping the LAST audio pulse
    aligns the sequences: the first candidate ("dropped first audio pulse")
    fails tolerance, so the loop falls through to the second candidate
    ("dropped last audio pulse"). Drives the second ``n_a > n_v`` candidate
    (source lines 943, 951-957).

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Per-test session root.
    processing_settings (dict)
        Package processing-settings fixture.
    mocker (pytest_mock.MockerFixture)
        Patch factory; no-ops ``smart_wait``.

    Returns
    -------
    None
    """

    mocker.patch("usv_playpen.processing.synchronize_files.smart_wait")
    sync = _make_sync(tmp_path, processing_settings)
    camera_fps = 150.0
    video_seq = np.array([300, 300, 300])
    video_frames = np.array([10, 20, 30])
    real_starts = (video_frames / camera_fps * 250000).astype(np.int64)
    audio_starts = np.concatenate([real_starts, [999999]])
    # trailing bogus pulse -> dropping first would misalign, dropping last works.
    audio_durations = np.array([300.0, 300.0, 300.0, 9999.0])
    _wire_audio_sync(sync, mocker,
                     audio_durations_ms=audio_durations,
                     audio_starts=audio_starts,
                     video_seq=video_seq, video_frames=video_frames, root=tmp_path)
    result = sync.find_audio_sync_trains()
    key = "m_240101_ch02_cropped_to_video"
    assert "ipi_discrepancy_ms" in result[key]
    np.testing.assert_array_equal(result[key]["video_ipi_start_frames"], video_frames)


def test_find_audio_sync_trains_drop_first_video_pulse(tmp_path, processing_settings, mocker):
    """
    Description
    -----------
    One extra VIDEO pulse (n_v == n_a + 1) where dropping the FIRST video pulse
    aligns the sequences: drives the ``n_a < n_v`` candidate branch and the
    "dropped first video pulse" resolution (source lines 945-957). A spurious
    leading video IPI is prepended to ``video_seq``/``video_frames``.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Per-test session root.
    processing_settings (dict)
        Package processing-settings fixture.
    mocker (pytest_mock.MockerFixture)
        Patch factory; no-ops ``smart_wait``.

    Returns
    -------
    None
    """

    mocker.patch("usv_playpen.processing.synchronize_files.smart_wait")
    sync = _make_sync(tmp_path, processing_settings)
    camera_fps = 150.0
    video_seq = np.array([9999, 300, 300, 300])
    video_frames = np.array([5, 20, 30, 40])
    retained_frames = video_frames[1:]
    audio_starts = (retained_frames / camera_fps * 250000).astype(np.int64)
    audio_durations = np.array([300.0, 300.0, 300.0])
    _wire_audio_sync(sync, mocker,
                     audio_durations_ms=audio_durations,
                     audio_starts=audio_starts,
                     video_seq=video_seq, video_frames=video_frames, root=tmp_path)
    result = sync.find_audio_sync_trains()
    key = "m_240101_ch02_cropped_to_video"
    assert "ipi_discrepancy_ms" in result[key]
    np.testing.assert_array_equal(result[key]["video_ipi_start_frames"], retained_frames)


def test_find_audio_sync_trains_drop_last_video_pulse(tmp_path, processing_settings, mocker):
    """
    Description
    -----------
    One extra VIDEO pulse (n_v == n_a + 1) where dropping the LAST video pulse
    aligns the sequences: the first candidate ("dropped first video pulse")
    fails tolerance, so the loop falls through to "dropped last video pulse"
    (source lines 948, 951-957). A spurious trailing video IPI is appended.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Per-test session root.
    processing_settings (dict)
        Package processing-settings fixture.
    mocker (pytest_mock.MockerFixture)
        Patch factory; no-ops ``smart_wait``.

    Returns
    -------
    None
    """

    mocker.patch("usv_playpen.processing.synchronize_files.smart_wait")
    sync = _make_sync(tmp_path, processing_settings)
    camera_fps = 150.0
    video_seq = np.array([300, 300, 300, 9999])
    video_frames = np.array([10, 20, 30, 99])
    retained_frames = video_frames[:-1]
    audio_starts = (retained_frames / camera_fps * 250000).astype(np.int64)
    audio_durations = np.array([300.0, 300.0, 300.0])
    _wire_audio_sync(sync, mocker,
                     audio_durations_ms=audio_durations,
                     audio_starts=audio_starts,
                     video_seq=video_seq, video_frames=video_frames, root=tmp_path)
    result = sync.find_audio_sync_trains()
    key = "m_240101_ch02_cropped_to_video"
    assert "ipi_discrepancy_ms" in result[key]
    np.testing.assert_array_equal(result[key]["video_ipi_start_frames"], retained_frames)


def test_find_audio_sync_trains_shape_mismatch_fallback_truncation(tmp_path, processing_settings, mocker):
    """
    Description
    -----------
    Shape mismatch of 1 where NEITHER drop-candidate clears tolerance: drives
    the ``diff_array is None`` fallback that truncates both sequences to
    ``n_min`` (source lines 958-962). Because the truncated diffs still exceed
    tolerance, the "IPI sequence match NOT found" branch (lines 967-969) fires
    and no ``ipi_discrepancy_ms`` is recorded.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Per-test session root.
    processing_settings (dict)
        Package processing-settings fixture.
    mocker (pytest_mock.MockerFixture)
        Patch factory; no-ops ``smart_wait``.

    Returns
    -------
    None
    """

    mocker.patch("usv_playpen.processing.synchronize_files.smart_wait")
    sync = _make_sync(tmp_path, processing_settings)
    # n_a = 4, n_v = 3; every alignment is far out of tolerance.
    audio_durations = np.array([100.0, 200.0, 400.0, 800.0])
    audio_starts = np.array([1, 2, 3, 4], dtype=np.int64)
    video_seq = np.array([300, 300, 300])
    video_frames = np.array([10, 20, 30])
    _wire_audio_sync(sync, mocker,
                     audio_durations_ms=audio_durations,
                     audio_starts=audio_starts,
                     video_seq=video_seq, video_frames=video_frames, root=tmp_path)
    result = sync.find_audio_sync_trains()
    key = "m_240101_ch02_cropped_to_video"
    # match failed -> the per-file dict was created but never populated.
    assert result[key] == {}


def test_find_audio_sync_trains_shape_mismatch_gt_one_infinite(tmp_path, processing_settings, mocker):
    """
    Description
    -----------
    Audio/video pulse counts differing by more than one: drives the final
    ``else`` that sets ``diff_array = np.array([np.inf])`` (source lines
    963-964), which then fails the tolerance check and reports no match.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Per-test session root.
    processing_settings (dict)
        Package processing-settings fixture.
    mocker (pytest_mock.MockerFixture)
        Patch factory; no-ops ``smart_wait``.

    Returns
    -------
    None
    """

    mocker.patch("usv_playpen.processing.synchronize_files.smart_wait")
    sync = _make_sync(tmp_path, processing_settings)
    audio_durations = np.array([300.0, 300.0, 300.0, 300.0, 300.0])
    audio_starts = np.arange(5, dtype=np.int64)
    video_seq = np.array([300, 300])
    video_frames = np.array([10, 20])
    _wire_audio_sync(sync, mocker,
                     audio_durations_ms=audio_durations,
                     audio_starts=audio_starts,
                     video_seq=video_seq, video_frames=video_frames, root=tmp_path)
    result = sync.find_audio_sync_trains()
    key = "m_240101_ch02_cropped_to_video"
    assert result[key] == {}


def test_find_audio_sync_trains_unequal_video_sequences_skip(tmp_path, processing_settings, mocker):
    """
    Description
    -----------
    Two cameras whose matched IPI sequences disagree: the all-equal guard at
    source line 926 is False, so the reconciliation block is skipped and the
    "IPI sequences on different videos do not match" branch (lines 1025-1026)
    runs. ``find_video_sync_trains`` is mocked to return a two-key sequence
    dict with differing rows.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Per-test session root.
    processing_settings (dict)
        Package processing-settings fixture.
    mocker (pytest_mock.MockerFixture)
        Patch factory; no-ops ``smart_wait``.

    Returns
    -------
    None
    """

    mocker.patch("usv_playpen.processing.synchronize_files.smart_wait")
    sync = _make_sync(tmp_path, processing_settings)
    (tmp_path / "video").mkdir()
    (tmp_path / "video" / "sess_camera_frame_count_dict.json").write_text(json.dumps({
        "21372315": [3, 150.0],
        "total_frame_number_least": 3,
        "total_video_time_least": 1.0,
    }))
    wave_data_dict = {"m_240101_ch02_cropped_to_video.wav":
                      {"wav_data": np.zeros(8, dtype=np.int16), "sampling_rate": 250000}}
    mocker.patch("usv_playpen.processing.synchronize_files.DataLoader",
                 return_value=MagicMock(load_wavefile_data=lambda: wave_data_dict))
    # two cameras with DIFFERENT sequences -> guard at line 926 is False.
    video_sync_sequence_dict = {"21372315": np.array([300, 300, 300]),
                                "21372316": np.array([300, 250, 300])}
    mocker.patch.object(sync, "find_video_sync_trains",
                        return_value=(np.array([10, 20, 30]), video_sync_sequence_dict))
    mocker.patch.object(sync, "find_ipi_intervals",
                        return_value=(np.array([300.0, 300.0, 300.0]),
                                      np.array([10, 20, 30], dtype=np.int64)))
    result = sync.find_audio_sync_trains()
    key = "m_240101_ch02_cropped_to_video"
    assert result[key] == {}


def test_find_audio_sync_trains_with_nidq_block(tmp_path, processing_settings, mocker):
    """
    Description
    -----------
    Exercises the NIDQ sync-extraction block (source lines 866-908): a real
    ``*.nidq.bin`` memmap with a triggerbox bit carrying a camera-frame pulse
    train (one large break + ``total_frame_number`` post-break edges) and a
    sync bit carrying OFF/ON IPI gaps. ``nidq_bool`` is enabled so the recording
    window is recovered, IPI durations + start samples are computed and saved to
    ``sync/nidq_ipi_data.npy`` (line 908). The audio/video match then succeeds
    and the NIDQ tail (lines 1018-1022) attaches NIDQ fields to the result.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Per-test session root.
    processing_settings (dict)
        Package processing-settings fixture (mutated locally to enable NIDQ).
    mocker (pytest_mock.MockerFixture)
        Patch factory; no-ops ``smart_wait``.

    Returns
    -------
    None
    """

    mocker.patch("usv_playpen.processing.synchronize_files.smart_wait")
    settings = json.loads(json.dumps(processing_settings))  # deep copy
    fa = settings["synchronize_files"]["Synchronizer"]["find_audio_sync_trains"]
    fa["nidq_bool"] = True
    fa["nidq_num_channels"] = 2
    fa["nidq_triggerbox_input_bit_position"] = 5
    fa["nidq_sync_input_bit_position"] = 7

    nidq_sr = fa["nidq_sr"]
    n_ch = 2
    total_frame_number = 4
    camera_fps = 150.0

    # build the 16-bit digital word per sample for the LAST nidq channel.
    n_samples = 4000
    digital_word = np.zeros(n_samples, dtype=np.int16)
    trig_bit = 1 << fa["nidq_triggerbox_input_bit_position"]
    sync_bit = 1 << fa["nidq_sync_input_bit_position"]

    # triggerbox rising edges: 3 pre-break edges, a large break, then
    # total_frame_number + 1 post-break edges (so [hop+total_frame_number] indexes).
    pre = [50, 100, 150]
    first_post = 1500
    spacing = 300
    post = [first_post + i * spacing for i in range(total_frame_number + 1)]
    for e in pre + post:
        digital_word[e] |= trig_bit          # single-sample triggerbox pulse

    # sync bit: held ON across the post-break window with OFF gaps (IPIs).
    window_start = first_post + 1
    window_end = post[-1] + 1
    digital_word[window_start:window_end] |= sync_bit
    # punch four OFF gaps (sync bit -> 0) to create IPI start/end edges,
    # one per video frame so the NIDQ-vs-video broadcast (line 1021) aligns.
    for gap_start in (1550, 1850, 2150, 2450):
        digital_word[gap_start:gap_start + 40] &= ~sync_bit

    # interleave digital word as the LAST channel of an (n_ch, n_samples) 'F' layout.
    frame = np.zeros((n_samples, n_ch), dtype=np.int16)
    frame[:, -1] = digital_word
    sync_dir = tmp_path / "sync"
    sync_dir.mkdir()
    frame.tofile(sync_dir / "recording.nidq.bin")

    # camera frame count JSON.
    (tmp_path / "video").mkdir()
    video_frames = np.array([10, 20, 30, 40])
    (tmp_path / "video" / "sess_camera_frame_count_dict.json").write_text(json.dumps({
        "21372315": [total_frame_number, camera_fps],
        "total_frame_number_least": total_frame_number,
        "total_video_time_least": (post[-1] - first_post) / nidq_sr,
    }))

    wave_data_dict = {"m_240101_ch02_cropped_to_video.wav":
                      {"wav_data": np.zeros(8, dtype=np.int16), "sampling_rate": 250000}}
    mocker.patch("usv_playpen.processing.synchronize_files.DataLoader",
                 return_value=MagicMock(load_wavefile_data=lambda: wave_data_dict))

    video_seq = np.array([300, 300, 300, 300])
    audio_starts = (video_frames / camera_fps * 250000).astype(np.int64)
    sync = Synchronizer(root_directory=str(tmp_path), input_parameter_dict=settings,
                        message_output=lambda *_a, **_k: None)
    mocker.patch.object(sync, "find_video_sync_trains",
                        return_value=(video_frames, {"21372315": video_seq}))
    mocker.patch.object(sync, "find_ipi_intervals",
                        return_value=(np.array([300.0, 300.0, 300.0, 300.0]), audio_starts))

    result = sync.find_audio_sync_trains()
    assert (sync_dir / "nidq_ipi_data.npy").is_file()    # NIDQ IPI data saved (line 908)
    key = "m_240101_ch02_cropped_to_video"
    assert "nidq_ipi_durations_ms" in result[key]         # NIDQ tail attached (lines 1019-1022)
    assert "nidq_ipi_discrepancy_ms" in result[key]
    assert "nidq_ipi_start_samples" in result[key]


# ---------------------------------------------------------------------------
# crop_wav_files_to_video — s_longer branch + KeyError guards
# ---------------------------------------------------------------------------


def test_crop_wav_files_to_video_both_devices_s_longer(tmp_path, mocker):
    """
    Description
    -----------
    Two-device ("both") crop where the SLAVE tracking window is longer than the
    master's: exercises the ``s_longer`` branch (source lines 1146-1157,
    1208-1220, 1279-1306) that resamples (SoX ``tempo``) the slave files down to
    the master duration, then rewrites the cropped slave WAV's LSB. The post-crop
    LSB overwrite covers the size-equal / size-greater / padding sub-branches
    depending on what ``static_sox`` produced.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Per-test session root.
    mocker (pytest_mock.MockerFixture)
        Patch factory; no-ops ``smart_wait``.

    Returns
    -------
    None
    """

    mocker.patch("usv_playpen.processing.synchronize_files.smart_wait")
    root = tmp_path
    (root / "video").mkdir()
    (root / "video" / "sess_camera_frame_count_dict.json").write_text(json.dumps({
        "total_frame_number_least": 5,
        "total_video_time_least": 0.001,
    }))
    (root / "sync").mkdir()
    audio_orig = root / "audio" / "original"
    audio_orig.mkdir(parents=True)

    # master: post-break edges spaced 5 -> window 200..225 (shorter);
    # slave: spaced 10 -> window 200..250 (longer) -> s_longer branch.
    _triggerbox_wav(audio_orig / "m_240101_ch04.wav",
                    [10, 20, 30, 200, 205, 210, 215, 220, 225])
    _triggerbox_wav(audio_orig / "s_240101_ch04.wav",
                    [10, 20, 30, 200, 210, 220, 230, 240, 250])

    settings = _processing_settings_full()
    crop = settings["synchronize_files"]["Synchronizer"]["crop_wav_files_to_video"]
    crop["device_receiving_input"] = "both"
    crop["triggerbox_ch_receiving_input"] = 4

    sync = Synchronizer(root_directory=str(root), input_parameter_dict=settings,
                        message_output=lambda *_a, **_k: None)
    sync.crop_wav_files_to_video()

    info = json.loads((root / "audio" / "audio_triggerbox_sync_info.json").read_text())
    assert info["s"]["duration_samples"] > info["m"]["duration_samples"]   # s_longer path
    cropped = list((root / "audio" / "cropped_to_video").glob("*_cropped_to_video.wav"))
    assert len(cropped) >= 2      # both devices cropped


def test_crop_wav_files_to_video_m_longer_missing_slave_key_raises(tmp_path, mocker):
    """
    Description
    -----------
    Two-device crop where only the MASTER triggerbox WAV exists: the slave
    duration stays 0 so ``m_longer`` is True, but no ``s_*`` triggerbox key is
    present in ``wave_data_dict``, driving the defensive ``KeyError`` guard
    (source lines 1139-1144).

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Per-test session root.
    mocker (pytest_mock.MockerFixture)
        Patch factory; no-ops ``smart_wait``.

    Returns
    -------
    None
    """

    mocker.patch("usv_playpen.processing.synchronize_files.smart_wait")
    root = tmp_path
    (root / "video").mkdir()
    (root / "video" / "sess_camera_frame_count_dict.json").write_text(json.dumps({
        "total_frame_number_least": 5,
        "total_video_time_least": 0.001,
    }))
    (root / "sync").mkdir()
    audio_orig = root / "audio" / "original"
    audio_orig.mkdir(parents=True)
    # only the master triggerbox channel present -> slave duration stays 0.
    _triggerbox_wav(audio_orig / "m_240101_ch04.wav",
                    [10, 20, 30, 200, 210, 220, 230, 240, 250])

    settings = _processing_settings_full()
    crop = settings["synchronize_files"]["Synchronizer"]["crop_wav_files_to_video"]
    crop["device_receiving_input"] = "both"
    crop["triggerbox_ch_receiving_input"] = 4

    sync = Synchronizer(root_directory=str(root), input_parameter_dict=settings,
                        message_output=lambda *_a, **_k: None)
    with pytest.raises(KeyError, match=r"s_\*"):
        sync.crop_wav_files_to_video()


def test_crop_wav_files_to_video_s_longer_missing_master_key_raises(tmp_path, mocker):
    """
    Description
    -----------
    Two-device crop where only the SLAVE triggerbox WAV exists: the master
    duration stays 0 so ``s_longer`` is True, but no ``m_*`` triggerbox key is
    present in ``wave_data_dict``, driving the defensive ``KeyError`` guard
    (source lines 1151-1156).

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Per-test session root.
    mocker (pytest_mock.MockerFixture)
        Patch factory; no-ops ``smart_wait``.

    Returns
    -------
    None
    """

    mocker.patch("usv_playpen.processing.synchronize_files.smart_wait")
    root = tmp_path
    (root / "video").mkdir()
    (root / "video" / "sess_camera_frame_count_dict.json").write_text(json.dumps({
        "total_frame_number_least": 5,
        "total_video_time_least": 0.001,
    }))
    (root / "sync").mkdir()
    audio_orig = root / "audio" / "original"
    audio_orig.mkdir(parents=True)
    # only the slave triggerbox channel present -> master duration stays 0.
    _triggerbox_wav(audio_orig / "s_240101_ch04.wav",
                    [10, 20, 30, 200, 210, 220, 230, 240, 250])

    settings = _processing_settings_full()
    crop = settings["synchronize_files"]["Synchronizer"]["crop_wav_files_to_video"]
    crop["device_receiving_input"] = "both"
    crop["triggerbox_ch_receiving_input"] = 4

    sync = Synchronizer(root_directory=str(root), input_parameter_dict=settings,
                        message_output=lambda *_a, **_k: None)
    with pytest.raises(KeyError, match=r"m_\*"):
        sync.crop_wav_files_to_video()


def _crop_both_m_longer(tmp_path, mocker, read_side_effect):
    """
    Description
    -----------
    Run the two-device ("both") ``m_longer`` crop with a custom
    ``wavfile.read`` side effect, used to force the post-tempo LSB-overwrite
    size sub-branches (source lines 1262-1274). The master tracking window is
    longer than the slave's, so the master cropped WAV is rewritten by reading
    it back; intercepting that read lets a test inject an over- or under-sized
    array to drive the ``>`` and padding branches.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Per-test session root.
    mocker (pytest_mock.MockerFixture)
        Patch factory; no-ops ``smart_wait``.
    read_side_effect (callable)
        Replacement for ``synchronize_files.wavfile.read``.

    Returns
    -------
    None
    """

    mocker.patch("usv_playpen.processing.synchronize_files.smart_wait")
    root = tmp_path
    (root / "video").mkdir()
    (root / "video" / "sess_camera_frame_count_dict.json").write_text(json.dumps({
        "total_frame_number_least": 5,
        "total_video_time_least": 0.001,
    }))
    (root / "sync").mkdir()
    audio_orig = root / "audio" / "original"
    audio_orig.mkdir(parents=True)
    _triggerbox_wav(audio_orig / "m_240101_ch04.wav",
                    [10, 20, 30, 200, 210, 220, 230, 240, 250])
    _triggerbox_wav(audio_orig / "s_240101_ch04.wav",
                    [10, 20, 30, 200, 205, 210, 215, 220, 225])

    settings = _processing_settings_full()
    crop = settings["synchronize_files"]["Synchronizer"]["crop_wav_files_to_video"]
    crop["device_receiving_input"] = "both"
    crop["triggerbox_ch_receiving_input"] = 4

    mocker.patch("usv_playpen.processing.synchronize_files.wavfile.read",
                 side_effect=read_side_effect)
    mocker.patch("usv_playpen.processing.synchronize_files.wavfile.write")

    sync = Synchronizer(root_directory=str(root), input_parameter_dict=settings,
                        message_output=lambda *_a, **_k: None)
    sync.crop_wav_files_to_video()


def _make_read_side_effect(cropped_size, real_read):
    """
    Description
    -----------
    Build a ``wavfile.read`` side effect that returns a zero array of
    ``cropped_size`` samples (with a non-zero final sample) for any path under
    ``cropped_to_video``, and passes every other path through to the supplied
    unpatched ``real_read``. Used to force a specific post-tempo size branch in
    the ``crop_wav_files_to_video`` LSB-overwrite block. The real reader must be
    captured BEFORE patching, otherwise the side effect would recurse into
    itself (the module attribute and ``scipy.io.wavfile.read`` are the same
    object).

    Parameters
    ----------
    cropped_size (int)
        Sample count returned for cropped-file reads.
    real_read (callable)
        The original, unpatched ``scipy.io.wavfile.read`` function.

    Returns
    -------
    side_effect (callable)
        Replacement for ``synchronize_files.wavfile.read``.
    """

    def _side_effect(path):
        path = str(path)
        if "cropped_to_video" in path:
            data = np.zeros(cropped_size, dtype=np.int16)
            data[-1] = 7
            return 250000, data
        return real_read(path)

    return _side_effect


def test_crop_wav_files_to_video_m_longer_oversized_cropped(tmp_path, mocker):
    """
    Description
    -----------
    Forces the ``m_longer`` LSB-overwrite ``>`` branch (source lines
    1264-1265): the cropped master WAV is read back as an OVERSIZED array
    (larger than the slave duration), so it is truncated to the slave duration
    before the LSB is XOR-merged. ``wavfile.read`` of the cropped file is
    intercepted; original-file reads pass through to real data.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Per-test session root.
    mocker (pytest_mock.MockerFixture)
        Patch factory; no-ops ``smart_wait`` and stubs ``wavfile.write``.

    Returns
    -------
    None
    """

    # slave window 200..225 inclusive = 26 samples; 40 > 26 -> '>' branch.
    _crop_both_m_longer(tmp_path, mocker, _make_read_side_effect(cropped_size=40, real_read=_real_wavfile_read))


def _crop_both_s_longer(tmp_path, mocker, read_side_effect):
    """
    Description
    -----------
    Run the two-device ("both") ``s_longer`` crop with a custom
    ``wavfile.read`` side effect, used to force the post-tempo slave
    LSB-overwrite size sub-branches (source lines 1292-1304). The slave
    tracking window is longer than the master's, so the slave cropped WAV is
    rewritten; intercepting its read injects an over- or under-sized array.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Per-test session root.
    mocker (pytest_mock.MockerFixture)
        Patch factory; no-ops ``smart_wait`` and stubs ``wavfile.write``.
    read_side_effect (callable)
        Replacement for ``synchronize_files.wavfile.read``.

    Returns
    -------
    None
    """

    mocker.patch("usv_playpen.processing.synchronize_files.smart_wait")
    root = tmp_path
    (root / "video").mkdir()
    (root / "video" / "sess_camera_frame_count_dict.json").write_text(json.dumps({
        "total_frame_number_least": 5,
        "total_video_time_least": 0.001,
    }))
    (root / "sync").mkdir()
    audio_orig = root / "audio" / "original"
    audio_orig.mkdir(parents=True)
    # slave longer (spaced 10 -> 200..250), master shorter (spaced 5 -> 200..225).
    _triggerbox_wav(audio_orig / "m_240101_ch04.wav",
                    [10, 20, 30, 200, 205, 210, 215, 220, 225])
    _triggerbox_wav(audio_orig / "s_240101_ch04.wav",
                    [10, 20, 30, 200, 210, 220, 230, 240, 250])

    settings = _processing_settings_full()
    crop = settings["synchronize_files"]["Synchronizer"]["crop_wav_files_to_video"]
    crop["device_receiving_input"] = "both"
    crop["triggerbox_ch_receiving_input"] = 4

    mocker.patch("usv_playpen.processing.synchronize_files.wavfile.read",
                 side_effect=read_side_effect)
    mocker.patch("usv_playpen.processing.synchronize_files.wavfile.write")

    sync = Synchronizer(root_directory=str(root), input_parameter_dict=settings,
                        message_output=lambda *_a, **_k: None)
    sync.crop_wav_files_to_video()


def test_crop_wav_files_to_video_s_longer_oversized_cropped(tmp_path, mocker):
    """
    Description
    -----------
    Forces the ``s_longer`` LSB-overwrite ``>`` branch (source lines
    1294-1295): the cropped slave WAV is read back OVERSIZED relative to the
    master duration and truncated before the XOR-merge.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Per-test session root.
    mocker (pytest_mock.MockerFixture)
        Patch factory; no-ops ``smart_wait`` and stubs ``wavfile.write``.

    Returns
    -------
    None
    """

    # master window 200..225 inclusive = 26 samples; 40 > 26 -> '>' branch.
    _crop_both_s_longer(tmp_path, mocker, _make_read_side_effect(cropped_size=40, real_read=_real_wavfile_read))
