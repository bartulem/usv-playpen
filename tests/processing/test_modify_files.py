"""
@author: bartulem
Targeted unit tests for processing.modify_files.Operator.

These tests drive the previously-uncovered branches of three Operator
methods without performing heavy real I/O:

(1) concatenate_audio_files - the ">1 audio files" memmap-concatenation path
    (name_origin parsing, memmap allocation, per-file column copy, flush).
(2) concatenate_binary_files - the POSIX `cat` concatenation command on tiny
    synthetic SpikeGLX .bin/.meta pairs, the changepoints-JSON *merge* path
    taken on a second run (existing JSON, tracking_start_end offsetting), and
    the non-zero subprocess return-code warning.
(3) rectify_video_fps - the `conduct_concat=False` copy-from-camera-subdir
    path, the metadata session_duration write, and the calibration-camera
    move/cleanup branch, with imgstore + ffmpeg invocation mocked out.

All external heavy tools (imgstore frame reads, ffmpeg) are mocked; only the
bundled `static_sox`/`cat`/`copy` real binaries are exercised where they are
fast and deterministic on tiny inputs.
"""

from __future__ import annotations

import configparser
import json
import pathlib

import numpy as np
import pytest
from scipy.io import wavfile

import usv_playpen
from usv_playpen.processing.modify_files import Operator


@pytest.fixture
def processing_settings():
    """
    Description
    -----------
    Load the package ``processing_settings.json`` fresh as a mutable dict so a
    test can override sub-keys before handing it to ``Operator`` without
    touching the on-disk settings file.

    Parameters
    ----------

    Returns
    -------
    settings (dict)
        The parsed processing-settings dict.
    """

    package_dir = pathlib.Path(usv_playpen.__file__).parent
    with (package_dir / '_parameter_settings' / 'processing_settings.json').open('r') as settings_file:
        return json.load(settings_file)


def _make_operator(root_directory, processing_settings, messages):
    """
    Description
    -----------
    Build an ``Operator`` whose ``message_output`` appends to a list, so tests
    can both silence the chatty status prints and assert on emitted warnings.

    Parameters
    ----------
    root_directory (str / list of str)
        Root directory (single string) or list of root directories.
    processing_settings (dict)
        Full processing-settings dict (already containing both the
        ``modify_files`` and ``synchronize_files`` sub-trees).
    messages (list)
        List accumulating every emitted message string.

    Returns
    -------
    operator (Operator)
        Configured Operator instance with patched message output.
    """

    return Operator(
        root_directory=root_directory,
        input_parameter_dict=processing_settings,
        message_output=messages.append,
    )


def test_concatenate_audio_files_writes_memmap(tmp_path, processing_settings, mocker):
    """
    Description
    -----------
    The ">1 audio files" path of ``concatenate_audio_files``: two single-channel
    WAVs in the configured concat directory are loaded, the destination memmap
    name is derived from the first key's ``split('_')[1]`` token plus the
    sampling rate / sample-count / file-count, the (n_samples x n_files) int16
    memmap is allocated, each file's samples are written into its column, and
    the array is flushed to disk. Verifies the memmap exists and round-trips
    the original per-channel sample values.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Per-test session root.
    processing_settings (dict)
        Package processing-settings fixture.
    mocker (pytest_mock.MockerFixture)
        No-ops the interactive ``smart_wait``.

    Returns
    -------
    None
    """

    mocker.patch("usv_playpen.processing.modify_files.smart_wait")

    concat_dir_name = "hpss_filtered"
    processing_settings['modify_files']['Operator']['concatenate_audio_files']['concat_dirs'] = [concat_dir_name]
    processing_settings['modify_files']['Operator']['concatenate_audio_files']['concatenate_audio_format'] = "wav"

    audio_type_dir = tmp_path / "audio" / concat_dir_name
    audio_type_dir.mkdir(parents=True)

    sampling_rate = 10000
    n_samples = 256
    rng = np.random.default_rng(0)
    ch01_data = rng.integers(-1000, 1000, size=n_samples, dtype=np.int16)
    ch02_data = rng.integers(-1000, 1000, size=n_samples, dtype=np.int16)
    wavfile.write(audio_type_dir / "m_sess_ch01.wav", sampling_rate, ch01_data)
    wavfile.write(audio_type_dir / "m_sess_ch02.wav", sampling_rate, ch02_data)

    messages = []
    operator = _make_operator(str(tmp_path), processing_settings, messages)
    operator.concatenate_audio_files()

    mmap_files = list(audio_type_dir.glob("sess_concatenated_audio_*_int16.mmap"))
    assert len(mmap_files) == 1, f"expected exactly one concatenated memmap, got {mmap_files}"

    expected_name = audio_type_dir / f"sess_concatenated_audio_{concat_dir_name}_{sampling_rate}_{n_samples}_2_int16.mmap"
    assert mmap_files[0] == expected_name

    written = np.memmap(filename=str(expected_name), dtype='int16', mode='r', shape=(n_samples, 2))
    np.testing.assert_array_equal(written[:, 0], ch01_data)
    np.testing.assert_array_equal(written[:, 1], ch02_data)


def test_concatenate_audio_files_skips_when_fewer_than_two(tmp_path, processing_settings, mocker):
    """
    Description
    -----------
    The ``else`` branch of ``concatenate_audio_files``: with a single WAV in the
    configured directory, concatenation is impossible, so no memmap is written
    and an explanatory "<2 audio files" message is emitted.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Per-test session root.
    processing_settings (dict)
        Package processing-settings fixture.
    mocker (pytest_mock.MockerFixture)
        No-ops the interactive ``smart_wait``.

    Returns
    -------
    None
    """

    mocker.patch("usv_playpen.processing.modify_files.smart_wait")

    concat_dir_name = "hpss_filtered"
    processing_settings['modify_files']['Operator']['concatenate_audio_files']['concat_dirs'] = [concat_dir_name]
    processing_settings['modify_files']['Operator']['concatenate_audio_files']['concatenate_audio_format'] = "wav"

    audio_type_dir = tmp_path / "audio" / concat_dir_name
    audio_type_dir.mkdir(parents=True)
    wavfile.write(audio_type_dir / "m_sess_ch01.wav", 10000, np.zeros(64, dtype=np.int16))

    messages = []
    operator = _make_operator(str(tmp_path), processing_settings, messages)
    operator.concatenate_audio_files()

    assert not list(audio_type_dir.glob("*.mmap"))
    assert any("<2 audio files" in m for m in messages)


def _write_synthetic_binary(imec_dir, file_stem, headstage_sn, n_samples_per_channel, total_num_channels=5):
    """
    Description
    -----------
    Write a tiny synthetic SpikeGLX ``.ap.bin`` / ``.ap.meta`` pair into an
    ``imec`` directory so ``concatenate_binary_files`` can memmap the binary and
    parse the metadata without any real recording hardware.

    Parameters
    ----------
    imec_dir (pathlib.Path)
        Per-probe ``ephys/imec<N>`` directory (created by the caller).
    file_stem (str)
        Recording stem, e.g. ``20250101_120000.imec0`` (the ``.ap.bin`` /
        ``.ap.meta`` suffixes are appended here).
    headstage_sn (str)
        Headstage serial number; must be present in the calibrated-SR ini.
    n_samples_per_channel (int)
        Number of int16 samples per channel in the binary file.
    total_num_channels (int)
        Total channel count (AP + SY) encoded in ``acqApLfSy``.

    Returns
    -------
    None
    """

    data = np.zeros(n_samples_per_channel * total_num_channels, dtype=np.int16)
    data.tofile(imec_dir / f"{file_stem}.ap.bin")
    (imec_dir / f"{file_stem}.ap.meta").write_text(
        f"acqApLfSy={total_num_channels - 1},0,1\n"
        f"imDatHs_sn={headstage_sn}\n"
        f"imDatPrb_sn=22420014283\n"
        f"fileSizeBytes={data.nbytes}\n"
        f"fileTimeSecs=0.5\n"
    )


def _headstage_sn():
    """
    Description
    -----------
    Return a headstage serial number that is guaranteed to be present in the
    package ``calibrated_sample_rates_imec.ini`` so the calibrated-SR lookup in
    ``concatenate_binary_files`` succeeds.

    Parameters
    ----------

    Returns
    -------
    headstage_sn (str)
        A calibrated headstage serial number.
    """

    ini = configparser.ConfigParser()
    ini.read(pathlib.Path(usv_playpen.__file__).parent / "_config" / "calibrated_sample_rates_imec.ini")
    return next(iter(ini["CalibratedHeadStages"].keys()))


def test_concatenate_binary_files_writes_outputs(tmp_path, processing_settings, mocker):
    """
    Description
    -----------
    First-run happy path for ``concatenate_binary_files`` (POSIX ``cat``): a
    single synthetic ``imec0`` recording is concatenated into the mirrored
    ``EPHYS`` tree. Verifies the concatenated ``.bin``, the concatenated
    ``.meta`` (with summed ``fileSizeBytes`` / ``fileTimeSecs``), and the
    freshly-written ``changepoints_info_*.json`` are produced.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Per-test temp dir; session rooted at ``<tmp>/Data/<id>`` so the EPHYS
        mirror resolves to ``<tmp>/EPHYS``.
    processing_settings (dict)
        Package processing-settings fixture.
    mocker (pytest_mock.MockerFixture)
        No-ops the interactive ``smart_wait``.

    Returns
    -------
    None
    """

    mocker.patch("usv_playpen.processing.modify_files.smart_wait")

    headstage_sn = _headstage_sn()
    root = tmp_path / "Data" / "20250101_120000"
    imec_dir = root / "ephys" / "imec0"
    imec_dir.mkdir(parents=True)
    _write_synthetic_binary(imec_dir, "20250101_120000.imec0", headstage_sn, n_samples_per_channel=64)

    messages = []
    operator = _make_operator([str(root)], processing_settings, messages)
    operator.concatenate_binary_files()

    ephys_base = tmp_path / "EPHYS" / "20250101_imec0"
    assert (ephys_base / "concatenated_20250101_imec0.ap.bin").is_file()
    assert (ephys_base / "concatenated_20250101_imec0.ap.meta").is_file()

    changepoints_json = ephys_base / "changepoints_info_20250101_imec0.json"
    assert changepoints_json.is_file()
    info = json.loads(changepoints_json.read_text())
    rec = info["20250101_120000.imec0"]
    assert rec["total_num_channels"] == 5
    assert rec["headstage_sn"] == headstage_sn
    assert rec["session_start_end"] == [0, 64]

    meta_text = (ephys_base / "concatenated_20250101_imec0.ap.meta").read_text()
    assert "fileSizeBytes=" in meta_text
    assert "fileTimeSecs=" in meta_text


def test_concatenate_binary_files_two_sessions_chains_changepoints(tmp_path, processing_settings, mocker):
    """
    Description
    -----------
    The multi-session changepoint-chaining branch of
    ``concatenate_binary_files``: two root directories each contribute one
    ``imec0`` recording, so the second file's ``session_start_end`` is offset by
    the running changepoint (the non-first-file ``else`` branch), and the POSIX
    ``cat`` command appends the second file rather than redirecting it.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Per-test temp dir; both sessions rooted under ``<tmp>/Data``.
    processing_settings (dict)
        Package processing-settings fixture.
    mocker (pytest_mock.MockerFixture)
        No-ops the interactive ``smart_wait``.

    Returns
    -------
    None
    """

    mocker.patch("usv_playpen.processing.modify_files.smart_wait")

    headstage_sn = _headstage_sn()
    root_a = tmp_path / "Data" / "20250101_120000"
    root_b = tmp_path / "Data" / "20250101_130000"
    for root, stem, n_samp in ((root_a, "20250101_120000.imec0", 64),
                               (root_b, "20250101_130000.imec0", 32)):
        imec_dir = root / "ephys" / "imec0"
        imec_dir.mkdir(parents=True)
        _write_synthetic_binary(imec_dir, stem, headstage_sn, n_samples_per_channel=n_samp)

    messages = []
    operator = _make_operator([str(root_a), str(root_b)], processing_settings, messages)
    operator.concatenate_binary_files()

    ephys_base = tmp_path / "EPHYS" / "20250101_imec0"
    info = json.loads((ephys_base / "changepoints_info_20250101_imec0.json").read_text())
    # first file occupies [0, 64], second is chained onto the running changepoint.
    assert info["20250101_120000.imec0"]["session_start_end"] == [0, 64]
    assert info["20250101_130000.imec0"]["session_start_end"] == [64, 96]
    # the concatenated .bin holds both recordings' bytes (96 samples x 5 ch x 2 B).
    assert (ephys_base / "concatenated_20250101_imec0.ap.bin").stat().st_size == 96 * 5 * 2


def test_concatenate_binary_files_merges_existing_changepoints(tmp_path, processing_settings, mocker):
    """
    Description
    -----------
    The changepoints-JSON *merge* branch of ``concatenate_binary_files``: when a
    ``changepoints_info_*.json`` already exists in the EPHYS mirror, the method
    reads it, replaces stale numeric components with the freshly-computed ones,
    and offsets a non-NaN ``tracking_start_end`` by the file's
    ``session_start_end`` start. We pre-seed the JSON with a manually-tracked
    window for the recording so both the "component differs" and the
    "tracking_start_end offset" sub-branches run, then re-write the merged JSON.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Per-test temp dir; session rooted at ``<tmp>/Data/<id>``.
    processing_settings (dict)
        Package processing-settings fixture.
    mocker (pytest_mock.MockerFixture)
        No-ops the interactive ``smart_wait``.

    Returns
    -------
    None
    """

    mocker.patch("usv_playpen.processing.modify_files.smart_wait")

    headstage_sn = _headstage_sn()
    root = tmp_path / "Data" / "20250101_120000"
    imec_dir = root / "ephys" / "imec0"
    imec_dir.mkdir(parents=True)
    _write_synthetic_binary(imec_dir, "20250101_120000.imec0", headstage_sn, n_samples_per_channel=64)

    ephys_base = tmp_path / "EPHYS" / "20250101_imec0"
    ephys_base.mkdir(parents=True)
    changepoints_json = ephys_base / "changepoints_info_20250101_imec0.json"
    existing = {
        "20250101_120000.imec0": {
            "session_start_end": [0, 999],
            "tracking_start_end": [10, 20],
            "largest_camera_break_duration": 1.0,
            "file_duration_samples": 999,
            "root_directory": str(root),
            "total_num_channels": 999,
            "headstage_sn": "OLD_STALE_SN",
            "imec_probe_sn": "OLD_PROBE",
        }
    }
    changepoints_json.write_text(json.dumps(existing, indent=4))

    messages = []
    operator = _make_operator([str(root)], processing_settings, messages)
    operator.concatenate_binary_files()

    merged = json.loads(changepoints_json.read_text())
    rec = merged["20250101_120000.imec0"]
    # stale numeric components were overwritten with the freshly-computed ones.
    assert rec["total_num_channels"] == 5
    assert rec["headstage_sn"] == headstage_sn
    # tracking_start_end was offset by session_start_end[0] (== 0 on first file).
    assert rec["tracking_start_end"] == [10, 20]


def test_concatenate_binary_files_adds_new_file_key_to_existing_json(tmp_path, processing_settings, mocker):
    """
    Description
    -----------
    The "new file key" sub-branch of the changepoints-JSON merge in
    ``concatenate_binary_files``: when the existing JSON describes a *different*
    recording than the one found this run, the current recording is inserted
    verbatim as a brand-new key while the pre-existing entry is left untouched.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Per-test temp dir; session rooted at ``<tmp>/Data/<id>``.
    processing_settings (dict)
        Package processing-settings fixture.
    mocker (pytest_mock.MockerFixture)
        No-ops the interactive ``smart_wait``.

    Returns
    -------
    None
    """

    mocker.patch("usv_playpen.processing.modify_files.smart_wait")

    headstage_sn = _headstage_sn()
    root = tmp_path / "Data" / "20250101_120000"
    imec_dir = root / "ephys" / "imec0"
    imec_dir.mkdir(parents=True)
    _write_synthetic_binary(imec_dir, "20250101_120000.imec0", headstage_sn, n_samples_per_channel=64)

    ephys_base = tmp_path / "EPHYS" / "20250101_imec0"
    ephys_base.mkdir(parents=True)
    changepoints_json = ephys_base / "changepoints_info_20250101_imec0.json"
    pre_existing = {
        "20240101_080000.imec0": {
            "session_start_end": [0, 100],
            "tracking_start_end": [np.nan, np.nan],
            "largest_camera_break_duration": np.nan,
            "file_duration_samples": 100,
            "root_directory": str(root),
            "total_num_channels": 5,
            "headstage_sn": headstage_sn,
            "imec_probe_sn": "OLD_PROBE",
        }
    }
    changepoints_json.write_text(json.dumps(pre_existing, indent=4))

    messages = []
    operator = _make_operator([str(root)], processing_settings, messages)
    operator.concatenate_binary_files()

    merged = json.loads(changepoints_json.read_text())
    # the unrelated pre-existing entry is preserved.
    assert "20240101_080000.imec0" in merged
    # the freshly-found recording was added as a new key.
    assert "20250101_120000.imec0" in merged
    assert merged["20250101_120000.imec0"]["session_start_end"] == [0, 64]


def test_concatenate_binary_files_warns_on_nonzero_return(tmp_path, processing_settings, mocker):
    """
    Description
    -----------
    The non-zero subprocess-return-code branch of ``concatenate_binary_files``:
    the concatenation ``subprocess.Popen`` is patched to return a fake process
    whose ``wait()`` reports a non-zero status, so the method emits the loud
    "may be incomplete or corrupt" warning instead of trusting the binary.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Per-test temp dir; session rooted at ``<tmp>/Data/<id>``.
    processing_settings (dict)
        Package processing-settings fixture.
    mocker (pytest_mock.MockerFixture)
        No-ops the interactive ``smart_wait`` and stubs ``subprocess.Popen``.

    Returns
    -------
    None
    """

    mocker.patch("usv_playpen.processing.modify_files.smart_wait")

    headstage_sn = _headstage_sn()
    root = tmp_path / "Data" / "20250101_120000"
    imec_dir = root / "ephys" / "imec0"
    imec_dir.mkdir(parents=True)
    _write_synthetic_binary(imec_dir, "20250101_120000.imec0", headstage_sn, n_samples_per_channel=64)

    fake_process = mocker.MagicMock()
    fake_process.wait.return_value = 13
    mocker.patch("usv_playpen.processing.modify_files.subprocess.Popen", return_value=fake_process)

    messages = []
    operator = _make_operator([str(root)], processing_settings, messages)
    operator.concatenate_binary_files()

    assert any("non-zero status 13" in m for m in messages)


def _patch_imgstore(mocker, total_frame_num, esr_frame_times, has_dropped=False):
    """
    Description
    -----------
    Patch ``new_for_filename`` so ``rectify_video_fps`` reads a synthetic
    imgstore: a fixed frame count, an optional dropped-frame mismatch
    (``frame_max`` > ``frame_count``), and a linear ``frame_time`` vector whose
    span sets the empirical sampling rate.

    Parameters
    ----------
    mocker (pytest_mock.MockerFixture)
        The mocker fixture used to patch ``new_for_filename``.
    total_frame_num (int)
        Number of frames reported by the store.
    esr_frame_times (numpy.ndarray)
        Per-frame timestamps returned by ``get_frame_metadata``.
    has_dropped (bool)
        If True, report ``frame_max`` larger than ``frame_count`` to exercise
        the dropped-frame WARNING path.

    Returns
    -------
    mock_store (unittest.mock.MagicMock)
        The configured imgstore mock.
    """

    mock_store = mocker.MagicMock()
    mock_store.frame_count = total_frame_num
    mock_store.frame_max = total_frame_num + 1 if has_dropped else total_frame_num
    mock_store.get_frame_metadata.return_value = {'frame_time': esr_frame_times}
    mocker.patch("usv_playpen.processing.modify_files.new_for_filename", return_value=mock_store)
    return mock_store


def test_rectify_video_fps_no_concat_copies_and_handles_calibration(tmp_path, processing_settings, mocker):
    """
    Description
    -----------
    Drives the rarely-hit branches of ``rectify_video_fps`` with
    ``conduct_concat=False``:

    (a) the pre-loop copy path (no non-hidden files in ``video/`` yet, so the
        per-camera ``conversion_target_file`` is copied up from the camera
        sub-directory),
    (b) the metadata ``session_duration`` write (metadata is present),
    (c) the calibration-camera branch (``current_working_dir`` becomes the
        camera dir, ``000000`` source / ``*-calibration`` destination),
    (d) the post-encode move + ``delete_old_file`` cleanup for both a normal and
        a calibration camera.

    ``new_for_filename`` (imgstore) and ``subprocess.Popen`` (ffmpeg) are mocked;
    the ffmpeg mock additionally creates the expected re-encoded output files so
    the subsequent ``shutil.move`` calls succeed deterministically.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Per-test session root.
    processing_settings (dict)
        Package processing-settings fixture.
    mocker (pytest_mock.MockerFixture)
        No-ops ``smart_wait`` and patches imgstore + ffmpeg.

    Returns
    -------
    None
    """

    mocker.patch("usv_playpen.processing.modify_files.smart_wait")

    rectify = processing_settings['modify_files']['Operator']['rectify_video_fps']
    normal_serial = "21372315"
    rectify['encode_camera_serial_num'] = [normal_serial]
    rectify['delete_old_file'] = True
    conv_target = rectify['conversion_target_file']
    vid_ext = rectify['encode_video_extension']

    date_token = "20250101_120000"
    date_joint = "20250101120000"
    video_dir = tmp_path / "video"

    # normal camera sub-directory holding the conversion target file.
    normal_cam_dir = video_dir / f"{date_token}.{normal_serial}"
    normal_cam_dir.mkdir(parents=True)
    (normal_cam_dir / f"{conv_target}.{vid_ext}").write_bytes(b"\x00")

    # calibration camera sub-directory holding a 000000 source file.
    calib_cam_dir = video_dir / f"calibration_{date_token}.{normal_serial}"
    calib_cam_dir.mkdir(parents=True)
    (calib_cam_dir / f"000000.{vid_ext}").write_bytes(b"\x00")

    # imgstore + metadata
    esr_times = np.linspace(0, 1.0, 30)
    _patch_imgstore(mocker, total_frame_num=30, esr_frame_times=esr_times, has_dropped=False)

    metadata = {'Session': {'session_duration': 0.0, 'camera_serials': [normal_serial]}}
    metadata_path = tmp_path / f"{tmp_path.name}_metadata.yaml"
    mocker.patch(
        "usv_playpen.processing.modify_files.load_session_metadata",
        return_value=(metadata, metadata_path),
    )
    saved = {}

    def _fake_save(data, filepath, logger):
        saved['data'] = data
        saved['filepath'] = filepath

    mocker.patch("usv_playpen.processing.modify_files.save_session_metadata", side_effect=_fake_save)

    # ffmpeg mock: create the new_file output in cwd so the later move succeeds.
    def _fake_popen(args, stdout=None, stderr=None, cwd=None, shell=None):
        new_file = args[-1]
        (pathlib.Path(cwd) / new_file).write_bytes(b"\x00")
        proc = mocker.MagicMock()
        proc.poll.return_value = 0
        proc.wait.return_value = 0
        return proc

    mocker.patch("usv_playpen.processing.modify_files.subprocess.Popen", side_effect=_fake_popen)

    messages = []
    operator = _make_operator(str(tmp_path), processing_settings, messages)
    operator.rectify_video_fps(conduct_concat=False)

    # (a) the no-concat copy lifted the conversion target into video/.
    assert (video_dir / f"{conv_target}_{normal_serial}.{vid_ext}").exists() or \
        (video_dir / date_joint / normal_serial / f"{normal_serial}-{date_joint}.{vid_ext}").exists()

    # (b) metadata session_duration was written via save_session_metadata.
    assert 'data' in saved
    assert saved['data']['Session']['session_duration'] == pytest.approx(round(esr_times[-1] - esr_times[0], 3))

    # (c)/(d) the re-encoded normal video was moved into the deep date/serial dir.
    moved_normal = video_dir / date_joint / normal_serial / f"{normal_serial}-{date_joint}.{vid_ext}"
    assert moved_normal.exists()

    # calibration output moved into calibration_images.
    moved_calib = video_dir / date_joint / normal_serial / 'calibration_images' / f"{normal_serial}-{date_joint}-calibration.{vid_ext}"
    assert moved_calib.exists()

    # The RAW calibration source must NEVER be deleted, even with delete_old_file=True:
    # that flag cleans up the disposable concatenation intermediate only. Regression guard
    # for the bug where the calibration '000000.<ext>' -- original loopbio footage -- was
    # silently destroyed after re-encoding.
    assert (calib_cam_dir / f"000000.{vid_ext}").is_file(), \
        "raw calibration source 000000.<ext> was deleted; it must be preserved"

    # camera frame count JSON written for the session.
    assert (video_dir / f"{date_joint}_camera_frame_count_dict.json").is_file()
