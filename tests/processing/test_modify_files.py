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
(4) broadband_filter_audio - line-noise tone removal, the 2 kHz linear-phase
    high-pass response, memmap column order, chunk-length independence,
    idempotency / revalidation, and the batch runner, on small synthetic
    250 kHz sessions.

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
from scipy import signal
from scipy.io import wavfile

import usv_playpen
from usv_playpen.os_utils import find_audio_mmap
from usv_playpen.processing.modify_files import (
    BROADBAND_BATCH_REPORT_COLUMNS,
    BROADBAND_REPORT_NAME,
    Operator,
    block_mean_phasors,
    broadband_filter_sessions,
    broadband_highpass_cutoff,
    broadband_output_settings,
    design_broadband_highpass,
    estimate_line_noise,
    read_broadband_session_list,
    unit_phasor,
)


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


def test_split_clusters_to_sessions_removes_duplicate_spikes(tmp_path, processing_settings, mocker):
    """Duplicate-spike removal in split_clusters_to_sessions: with the toggle on, a
    unit's near-coincident spikes are dropped before the per-session save (exactly
    the injected duplicates), and the saved count never exceeds the toggle-off count."""

    mocker.patch("usv_playpen.processing.modify_files.smart_wait")

    headstage_sn = _headstage_sn()
    ini = configparser.ConfigParser()
    ini.read(pathlib.Path(usv_playpen.__file__).parent / "_config" / "calibrated_sample_rates_imec.ini")
    sampling_rate = float(ini["CalibratedHeadStages"][headstage_sn])
    censored_samples = round(0.3e-3 * sampling_rate)  # matches shipped duplicate_censored_period_ms

    # one good unit: 10 well-separated spikes + 3 duplicates one sample after the
    # first three (well inside the censored period) -> exactly 3 must be removed
    base_spikes = np.arange(1, 11, dtype=np.int64) * 1000
    duplicate_spikes = base_spikes[:3] + 1
    assert censored_samples > 1  # the injected duplicates must fall inside the window
    spike_times = np.sort(np.concatenate([base_spikes, duplicate_spikes]))
    spike_clusters = np.zeros(spike_times.shape[0], dtype=np.int64)

    root = tmp_path / "Data" / "20250101_120000"
    (root / "ephys" / "imec0").mkdir(parents=True)
    (root / "video").mkdir(parents=True)
    (root / "video" / "sess_camera_frame_count_dict.json").write_text(
        json.dumps({"median_empirical_camera_sr": 150.0, "total_frame_number_least": 100000})
    )

    ks_dir = tmp_path / "EPHYS" / "20250101_imec0" / "kilosort4"
    ks_dir.mkdir(parents=True)
    np.save(ks_dir / "spike_times.npy", spike_times)
    np.save(ks_dir / "spike_clusters.npy", spike_clusters)
    (ks_dir / "cluster_info.tsv").write_text("cluster_id\tch\tgroup\n0\t10\tgood\n")

    changepoints = {
        "20250101_120000.imec0": {
            "session_start_end": [0, 20000],
            "tracking_start_end": [0, 20000],
            "largest_camera_break_duration": 0,
            "file_duration_samples": 20000,
            "root_directory": str(root),
            "total_num_channels": 5,
            "headstage_sn": headstage_sn,
            "imec_probe_sn": "22420014283",
        }
    }
    (ks_dir.parent / "changepoints_info_20250101_imec0.json").write_text(json.dumps(changepoints))

    processing_settings['modify_files']['Operator']['get_spike_times']['min_spike_num'] = 0
    saved_npy = root / "ephys" / "imec0" / "cluster_data" / "imec0_cl0000_ch010_good.npy"

    def run_and_count(remove_duplicate_spikes):
        processing_settings['modify_files']['Operator']['get_spike_times']['remove_duplicate_spikes'] = remove_duplicate_spikes
        _make_operator([str(root)], processing_settings, []).split_clusters_to_sessions()
        return int(np.load(saved_npy).shape[1])

    off_count = run_and_count(False)
    on_count = run_and_count(True)

    assert off_count == spike_times.shape[0]        # nothing dropped when off
    assert on_count == base_spikes.shape[0]         # the 3 duplicates removed when on
    assert off_count - on_count == duplicate_spikes.shape[0]
    assert on_count <= off_count                    # monotonic: dedup never adds spikes


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


_BB_SR = 250000
_BB_SECONDS = 6.0
_BB_TONES = {0: (8000.17, 200.0), 1: (8000.67, 100.0)}


def _broadband_settings(processing_settings, chunk_s=2.0):
    """
    Description
    -----------
    Shrinks the broadband settings to a 6 s synthetic session: three 2 s
    line-noise estimation windows and one thread, the rest as shipped.

    Parameters
    ----------
    processing_settings (dict)
        Package processing-settings fixture (mutated in place).
    chunk_s (float)
        Processing chunk length (s).

    Returns
    -------
    settings (dict)
        The mutated ``broadband_filter_audio`` block.
    """

    settings = processing_settings['modify_files']['Operator']['broadband_filter_audio']
    settings['line_noise_estimation_windows'] = 3
    settings['line_noise_estimation_window_s'] = 2.0
    settings['chunk_s'] = chunk_s
    settings['n_threads'] = 1
    return settings


def _write_broadband_session(root):
    """
    Description
    -----------
    Writes three full-band HPSS wavs (``audio/hpss/[ms]_230101120000_chNN_cropped_to_video_hpss.wav``,
    250 kHz, 6 s) plus a stray empty ``output.wav``. Every channel carries white
    noise (sd 300) and a 500 Hz sine (amplitude 3000, below the high-pass);
    master channel 1 adds an 8000.17 Hz line (amplitude 200), master channel 2 an
    8000.67 Hz line (amplitude 100), and slave channel 1 a 30 kHz marker tone
    (amplitude 1000) that only its own memmap column may contain.

    Parameters
    ----------
    root (pathlib.Path)
        Session root.

    Returns
    -------
    signals (list of np.ndarray)
        The int16 channel signals, in sorted file-name (memmap column) order.
    """

    hpss_dir = root / "audio" / "hpss"
    hpss_dir.mkdir(parents=True)
    n_samples = int(_BB_SECONDS * _BB_SR)
    t = np.arange(n_samples) / _BB_SR
    rng = np.random.default_rng(7)
    names = ["m_230101120000_ch01_cropped_to_video_hpss.wav", "m_230101120000_ch02_cropped_to_video_hpss.wav",
             "s_230101120000_ch01_cropped_to_video_hpss.wav"]
    signals = []
    for column, name in enumerate(names):
        x = rng.normal(0.0, 300.0, n_samples) + 3000.0 * np.sin(2 * np.pi * 500.0 * t)
        if column in _BB_TONES:
            frequency, amplitude = _BB_TONES[column]
            x += amplitude * np.cos(2 * np.pi * frequency * t + 0.3 * column)
        if column == 2:
            x += 1000.0 * np.sin(2 * np.pi * 30000.0 * t)
        x = np.clip(np.rint(x), -32768, 32767).astype(np.int16)
        wavfile.write(hpss_dir / name, _BB_SR, x)
        signals.append(x)
    (hpss_dir / "output.wav").write_bytes(b"")
    return signals


def _tone_amplitude(x, frequency):
    """
    Description
    -----------
    Amplitude of a tone of known frequency in a signal (complex demodulation
    over the whole signal).

    Parameters
    ----------
    x (np.ndarray)
        Signal.
    frequency (float)
        Tone frequency (Hz).

    Returns
    -------
    amplitude (float)
        Estimated peak amplitude.
    """

    return float(2 * np.abs(block_mean_phasors(np.asarray(x, dtype=np.float64), frequency, _BB_SR, 0, x.shape[0])[0]))


def _read_broadband(root):
    """
    Description
    -----------
    Opens the session's broadband memmap through the exact-one lookup.

    Parameters
    ----------
    root (pathlib.Path)
        Session root.

    Returns
    -------
    audio (np.ndarray)
        ``(n_samples, n_channels)`` int16 copy of the memmap.
    """

    path = find_audio_mmap(root, "broadband")
    n_samples, n_channels = (int(token) for token in path.name.split("_")[-3:-1])
    return np.array(np.memmap(path, dtype=np.int16, mode="r", shape=(n_samples, n_channels)))


def test_broadband_highpass_cutoff_reads_the_upper_edge_of_the_removed_band(processing_settings):
    """
    The broadband block's ``filter_freq_bounds`` follows the ``filter_audio_files``
    convention (the band between the two bounds is removed): the shipped
    ``[0, 2000]`` is the 2 kHz high-pass, whose -6 dB point is the upper edge.
    """
    settings = processing_settings['modify_files']['Operator']['broadband_filter_audio']
    assert settings['filter_freq_bounds'] == [0, 2000]
    assert broadband_highpass_cutoff(settings) == 2000.0


@pytest.mark.parametrize("bounds", [[500, 2000], [0, 0], [0, -10], [2000]])
def test_broadband_highpass_cutoff_rejects_bounds_that_are_not_a_highpass(bounds):
    """
    The broadband filter is a high-pass only: a nonzero lower bound (a
    band-stop), a non-positive upper bound or a wrong length must raise rather
    than silently filter something else.
    """
    with pytest.raises(ValueError, match="filter_freq_bounds"):
        broadband_highpass_cutoff({'filter_freq_bounds': bounds})


def test_broadband_output_settings_record_the_cutoff_as_one_number(processing_settings):
    """
    The settings stored in line_noise.json (and compared to decide whether an
    output is current) hold the cutoff as ``highpass_cutoff_hz``, the upper
    edge of ``filter_freq_bounds``, and not the bounds themselves: that is the
    form of every report written before the bounds setting existed, so those
    outputs stay current while the filter is unchanged.
    """
    settings = processing_settings['modify_files']['Operator']['broadband_filter_audio']
    recorded = broadband_output_settings(settings)
    assert recorded['highpass_cutoff_hz'] == 2000
    assert 'filter_freq_bounds' not in recorded
    changed = dict(settings, filter_freq_bounds=[0, 3000])
    assert broadband_output_settings(changed)['highpass_cutoff_hz'] == 3000
    assert broadband_output_settings(changed) != recorded


def test_design_broadband_highpass_matches_sox_sinc_t1000_2k():
    """
    Description
    -----------
    The FIR equivalent of sox ``sinc -t 1000 2k`` at 250 kHz: odd length,
    symmetric (linear phase), -6 dB at 2 kHz, >= 100 dB down at and below
    1.5 kHz, within 0.01 dB of unity from 2.5 kHz up.

    Returns
    -------
    None
    """

    taps, kaiser_beta = design_broadband_highpass(_BB_SR, 2000, 1000, 120)
    assert taps.shape[0] % 2 == 1
    np.testing.assert_allclose(taps, taps[::-1], atol=1e-15)
    assert kaiser_beta > 10
    _, response = signal.freqz(taps, worN=np.array([500.0, 1000.0, 1500.0, 2000.0, 2500.0, 3000.0, 30000.0, 100000.0]), fs=_BB_SR)
    gains = 20 * np.log10(np.abs(response))
    assert np.all(gains[:3] <= -100)
    assert abs(gains[3] + 6.02) < 0.05
    assert np.all(np.abs(gains[4:]) < 0.01)


def test_unit_phasor_matches_direct_exponential_at_large_offsets():
    """
    Description
    -----------
    The outer-product phasor equals the direct complex exponential, also at
    absolute sample indices of a 20 min recording.

    Returns
    -------
    None
    """

    start = 299_000_000
    n = np.arange(start, start + 5000, dtype=np.float64)
    direct = np.exp(2j * np.pi * np.mod(8000.17 / _BB_SR * n, 1.0))
    np.testing.assert_allclose(unit_phasor(8000.17, _BB_SR, start, 5000), direct, atol=1e-9)


def test_estimate_line_noise_finds_comb_and_doublet_on_a_sloping_floor(tmp_path):
    """
    Description
    -----------
    In one search band the estimator keeps every line of a comb, including a
    doublet 0.26 Hz apart (2090.05 / 2090.31 Hz plus 2150.00 Hz), at the right
    frequencies, on a background that drops by ~80 dB inside the band (noise
    low-passed at 2125 Hz with a steep elliptic filter), and keeps no band-edge
    or slope artefact.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Per-test directory.

    Returns
    -------
    None
    """

    n_samples = 32 * _BB_SR
    t = np.arange(n_samples) / _BB_SR
    rng = np.random.default_rng(3)
    steep = signal.ellip(8, 0.1, 80, 2125.0, btype="lowpass", fs=_BB_SR, output="sos")
    x = signal.sosfilt(steep, rng.normal(0.0, 2000.0, n_samples)) + rng.normal(0.0, 2.0, n_samples)
    for frequency in (2090.05, 2090.31, 2150.00):
        x += 30.0 * np.cos(2 * np.pi * frequency * t)
    path = tmp_path / "m_230101120000_ch01_cropped_to_video_hpss.wav"
    wavfile.write(path, _BB_SR, np.clip(np.rint(x), -32768, 32767).astype(np.int16))

    tones = estimate_line_noise(wav_path=path, search_bands_hz=[[2080, 2165]], min_height_db=8.0, n_windows=3, window_s=10.0,
                                max_tones_per_band=6, min_separation_hz=0.25, floor_window_hz=4.0)
    kept = sorted(tone['frequency_hz'] for tone in tones if tone['kept'])
    assert len(kept) == 3, tones
    np.testing.assert_allclose(kept, [2090.05, 2090.31, 2150.00], atol=0.02)


def test_broadband_filter_removes_tones_highpasses_and_keeps_column_order(tmp_path, processing_settings, mocker):
    """
    Description
    -----------
    On a synthetic session: the memmap lands at the exact broadband path with
    columns in sorted wav order (the 30 kHz marker of slave channel 1 is only in
    column 2, at unchanged amplitude); the per-device 8 kHz lines are found on
    their own channels at their own frequencies and removed (residual < 5 % of
    the line), and no line is kept on the channel without one; the 500 Hz sine
    is removed by the high-pass; the stray empty ``output.wav`` is ignored; and
    the line-free channel equals the input convolved with the FIR within 1 LSB.

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
    _broadband_settings(processing_settings)
    signals = _write_broadband_session(tmp_path)
    messages = []
    summary = _make_operator(str(tmp_path), processing_settings, messages).broadband_filter_audio()

    assert summary['status'] == 'written'
    n_samples = signals[0].shape[0]
    expected = tmp_path / "audio" / "broadband_filtered" / f"230101120000_concatenated_audio_broadband_filtered_{_BB_SR}_{n_samples}_3_int16.mmap"
    assert find_audio_mmap(tmp_path, "broadband") == expected
    output = _read_broadband(tmp_path)

    with open(tmp_path / "audio" / "broadband_filtered" / BROADBAND_REPORT_NAME, encoding="utf-8") as report_file:
        report = json.load(report_file)
    assert report['complete'] is True
    assert [source['file'] for source in report['sources']] == [
        "m_230101120000_ch01_cropped_to_video_hpss.wav", "m_230101120000_ch02_cropped_to_video_hpss.wav",
        "s_230101120000_ch01_cropped_to_video_hpss.wav"]
    for column, (frequency, amplitude) in _BB_TONES.items():
        line = report['line_noise']['channels'][column]['tones'][0]
        assert line['kept'] is True
        assert abs(line['frequency_hz'] - frequency) < 0.02
        assert abs(line['amplitude_median_lsb'] - amplitude) / amplitude < 0.1
        assert _tone_amplitude(signals[column], frequency) > 0.9 * amplitude
        assert _tone_amplitude(output[:, column], frequency) < 0.05 * amplitude
    assert report['line_noise']['channels'][2]['tones'][0]['kept'] is False

    for column in range(3):
        assert _tone_amplitude(output[:, column], 500.0) < 1.0
    assert abs(_tone_amplitude(output[:, 2], 30000.0) - 1000.0) < 10.0
    assert _tone_amplitude(output[:, 0], 30000.0) < 5.0
    assert _tone_amplitude(output[:, 1], 30000.0) < 5.0

    taps, _ = design_broadband_highpass(_BB_SR, 2000, 1000, 120)
    reference = np.convolve(signals[2].astype(np.float64), taps, mode="same")
    assert np.max(np.abs(output[:, 2].astype(np.float64) - reference)) <= 1.0


def test_broadband_filter_is_independent_of_chunk_length(tmp_path, processing_settings, mocker):
    """
    Description
    -----------
    The output does not depend on the processing chunk length (block grid,
    median windows and FIR are all on absolute sample indices): 0.37 s and 2 s
    chunks agree within 1 LSB (FFT-convolution rounding only).

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Per-test directory (two session roots).
    processing_settings (dict)
        Package processing-settings fixture.
    mocker (pytest_mock.MockerFixture)
        No-ops the interactive ``smart_wait``.

    Returns
    -------
    None
    """

    mocker.patch("usv_playpen.processing.modify_files.smart_wait")
    outputs = []
    for chunk_s in (0.37, 2.0):
        root = tmp_path / f"chunk_{chunk_s}"
        _write_broadband_session(root)
        _broadband_settings(processing_settings, chunk_s=chunk_s)
        _make_operator(str(root), processing_settings, []).broadband_filter_audio()
        outputs.append(_read_broadband(root).astype(np.int32))
    assert np.max(np.abs(outputs[0] - outputs[1])) <= 1


def test_broadband_filter_is_idempotent_and_revalidates(tmp_path, processing_settings, mocker):
    """
    Description
    -----------
    A second run on a complete, current output is skipped without touching the
    memmap; a stale temporary of an interrupted run is removed; changing an
    output-defining setting, or deleting the report, makes the next run rewrite
    the memmap; the exactly-one broadband lookup holds throughout.

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
    settings = _broadband_settings(processing_settings)
    _write_broadband_session(tmp_path)
    first = _make_operator(str(tmp_path), processing_settings, []).broadband_filter_audio()
    mmap_path = find_audio_mmap(tmp_path, "broadband")
    first_mtime = mmap_path.stat().st_mtime_ns

    second = _make_operator(str(tmp_path), processing_settings, []).broadband_filter_audio()
    assert first['status'] == 'written'
    assert second['status'] == 'skipped'
    assert mmap_path.stat().st_mtime_ns == first_mtime

    stale = mmap_path.parent / f".{mmap_path.name}.tmp-99999"
    stale.write_bytes(b"partial")
    settings['line_noise_min_height_db'] = 7.0
    third = _make_operator(str(tmp_path), processing_settings, []).broadband_filter_audio()
    assert third['status'] == 'written'
    assert third['reason'] == 'settings changed'
    assert not stale.exists()
    assert find_audio_mmap(tmp_path, "broadband") == mmap_path

    (mmap_path.parent / BROADBAND_REPORT_NAME).unlink()
    fourth = _make_operator(str(tmp_path), processing_settings, []).broadband_filter_audio()
    assert fourth['status'] == 'written'
    assert (mmap_path.parent / BROADBAND_REPORT_NAME).is_file()


def test_broadband_batch_runner_reports_and_resumes(tmp_path, processing_settings):
    """
    Description
    -----------
    The batch runner processes every listed session (one of them missing its
    wavs, which must fail without stopping the batch), appends one report row
    per session under the documented header, logs, and on a second run skips
    the already written session.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Per-test directory.
    processing_settings (dict)
        Package processing-settings fixture.

    Returns
    -------
    None
    """

    _broadband_settings(processing_settings)
    good = tmp_path / "good"
    _write_broadband_session(good)
    bad = tmp_path / "bad"
    (bad / "audio" / "hpss").mkdir(parents=True)
    sessions_file = tmp_path / "sessions.txt"
    sessions_file.write_text(f"# backfill\n{good}\n\n{bad}\n{good}\n")
    session_roots = read_broadband_session_list(sessions_file=str(sessions_file))
    assert session_roots == [str(good), str(bad)]

    log_path = tmp_path / "logs" / "batch.log"
    report_path = tmp_path / "logs" / "report.csv"
    rows = broadband_filter_sessions(session_roots, processing_settings, n_workers=2, log_path=str(log_path),
                                     report_csv_path=str(report_path), message_output=lambda *_args: None)
    status = {row['session_root']: row['status'] for row in rows}
    assert status == {str(good): 'written', str(bad): 'failed'}
    rows_again = broadband_filter_sessions([str(good)], processing_settings, n_workers=1, log_path=str(log_path),
                                           report_csv_path=str(report_path), message_output=lambda *_args: None)
    assert rows_again[0]['status'] == 'skipped'
    with open(report_path, encoding="utf-8") as report_file:
        lines = report_file.read().splitlines()
    assert lines[0] == ",".join(BROADBAND_BATCH_REPORT_COLUMNS)
    assert len(lines) == 4
    assert "FAILED" in log_path.read_text()


def test_read_broadband_session_list_from_usv_counts_table(tmp_path):
    """
    Description
    -----------
    From a session table only the rows with the requested tag are kept, in
    order.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Per-test directory.

    Returns
    -------
    None
    """

    table = tmp_path / "session_usv_counts.csv"
    table.write_text(f"session_id,dir,n_usv,tag\na,{tmp_path / 'a'},3,ok\nb,{tmp_path / 'b'},0,playback\nc,{tmp_path / 'c'},9,ok\n")
    assert read_broadband_session_list(usv_counts_csv=str(table)) == [str(tmp_path / 'a'), str(tmp_path / 'c')]
    with pytest.raises(ValueError, match="exactly one"):
        read_broadband_session_list()
