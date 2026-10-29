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
(3) rectify_video_fps - how each camera's input is found (the concatenated
    file, else the single video read in place from its camera folder), the
    errors for several videos without concatenation and for no video, the
    metadata session_duration write, and the calibration-camera move/cleanup
    branch, with imgstore + ffmpeg invocation mocked out.
(4) broadband_filter_audio - line-noise tone removal, the 2 kHz linear-phase
    high-pass response, memmap column order, chunk-length independence,
    idempotency / revalidation, and the batch runner, on small synthetic
    250 kHz sessions.

All external heavy tools (imgstore frame reads, ffmpeg) are mocked; only the
bundled `static_sox`/`cat`/`copy` real binaries are exercised where they are
fast and deterministic on tiny inputs.

Coverage: ``concatenate_audio_files`` (memmap write, fewer-than-two skip),
``concatenate_binary_files`` (outputs, chained / merged / extended
changepoints, the concatenated tracking clock on rerun, failed-rewrite
atomicity, the non-zero return warning, a .meta missing required keys, no
template .meta, and a probe absent from the last session),
``concatenate_video_files`` (stale output excluded and replaced, ``-y``
overwrite, a single video left untouched, a stale output beside a single
chunk, calibration and unlisted folders ignored, an empty folder skipped, the
timeout tunable), ``split_clusters_to_sessions``
(duplicate-spike removal, failed-save atomicity), ``rectify_video_fps``
(input resolution, the two input errors, calibration handling), the broadband helpers (high-pass cutoff
and design, unit phasors and block demodulation edge cases, line-noise
estimation incl. silence and the separation rule, the stream's short-read
guard, low-band variance incl. a too-short memmap, its backfill function and
CLI, every rejection reason of ``validate_broadband_output``),
``broadband_filter_audio`` (tone removal, column order, chunk-length
independence, idempotency / revalidation, input checks, a short output never
published, an older memmap replaced, an unknown code version), the batch
runner and its in-process worker, the session-list reader, and the required
root directory.
"""

from __future__ import annotations

import configparser
import json
import pathlib
import re
from importlib import metadata as importlib_metadata

import numpy as np
import pytest
import soundfile as sf
from click.testing import CliRunner
from scipy import signal
from scipy.io import wavfile

import usv_playpen
from tests.conftest import write_broadband_audio
from usv_playpen.os_utils import find_audio_mmap
from usv_playpen.processing import modify_files, preprocess_data
from usv_playpen.processing.modify_files import (
    BROADBAND_BATCH_REPORT_COLUMNS,
    BROADBAND_REPORT_NAME,
    Operator,
    _BroadbandChannelStream,
    add_broadband_low_band_variance,
    block_mean_phasors,
    broadband_filter_sessions,
    broadband_highpass_cutoff,
    broadband_output_settings,
    design_broadband_highpass,
    estimate_line_noise,
    estimate_low_band_variance,
    read_broadband_session_list,
    unit_phasor,
    validate_broadband_output,
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
        """Split clusters with duplicate removal set to ``remove_duplicate_spikes`` and return the saved spike count."""
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


def test_concatenate_binary_files_failed_changepoints_rewrite_keeps_the_old_json(tmp_path, processing_settings, mocker):
    """The changepoints JSON is rewritten through a temporary sibling, so a dump
    that dies part-way leaves the existing JSON byte-identical (the session's
    tracking alignment lives only there) and no temporary file behind."""
    mocker.patch("usv_playpen.processing.modify_files.smart_wait")
    headstage_sn = _headstage_sn()
    root = tmp_path / "Data" / "20250101_120000"
    imec_dir = root / "ephys" / "imec0"
    imec_dir.mkdir(parents=True)
    _write_synthetic_binary(imec_dir, "20250101_120000.imec0", headstage_sn, n_samples_per_channel=64)
    ephys_base = tmp_path / "EPHYS" / "20250101_imec0"
    ephys_base.mkdir(parents=True)
    changepoints_json = ephys_base / "changepoints_info_20250101_imec0.json"
    changepoints_json.write_text(json.dumps({
        "20250101_120000.imec0": {
            "session_start_end": [0, 999], "tracking_start_end": [10, 20], "largest_camera_break_duration": 1.0,
            "file_duration_samples": 999, "root_directory": str(root), "total_num_channels": 999,
            "headstage_sn": "OLD_STALE_SN", "imec_probe_sn": "OLD_PROBE",
        }
    }, indent=4))
    before = changepoints_json.read_bytes()

    def _failing_dump(_obj, file_handle, **_kwargs):
        """Write a truncated JSON prefix into the open handle and fail, like a disk filling mid-dump."""
        file_handle.write('{"20250101_120000.imec0": {"session_')
        message = "disk full"
        raise OSError(message)

    mocker.patch("usv_playpen.processing.modify_files.json.dump", side_effect=_failing_dump)
    with pytest.raises(OSError, match="disk full"):
        _make_operator([str(root)], processing_settings, []).concatenate_binary_files()
    assert changepoints_json.read_bytes() == before
    assert not [path.name for path in ephys_base.iterdir() if path.name.startswith(".")]


def test_split_clusters_to_sessions_failed_save_keeps_the_old_spike_file(tmp_path, processing_settings, mocker):
    """Each unit's cluster_data spike file is written through a temporary sibling,
    so a save that dies part-way leaves the earlier file loadable and unchanged."""
    mocker.patch("usv_playpen.processing.modify_files.smart_wait")
    headstage_sn = _headstage_sn()
    root = tmp_path / "Data" / "20250101_120000"
    (root / "ephys" / "imec0").mkdir(parents=True)
    (root / "video").mkdir(parents=True)
    (root / "video" / "sess_camera_frame_count_dict.json").write_text(
        json.dumps({"median_empirical_camera_sr": 150.0, "total_frame_number_least": 100000})
    )
    ks_dir = tmp_path / "EPHYS" / "20250101_imec0" / "kilosort4"
    ks_dir.mkdir(parents=True)
    np.save(ks_dir / "spike_times.npy", np.arange(1, 11, dtype=np.int64) * 1000)
    np.save(ks_dir / "spike_clusters.npy", np.zeros(10, dtype=np.int64))
    (ks_dir / "cluster_info.tsv").write_text("cluster_id\tch\tgroup\n0\t10\tgood\n")
    (ks_dir.parent / "changepoints_info_20250101_imec0.json").write_text(json.dumps({
        "20250101_120000.imec0": {
            "session_start_end": [0, 20000], "tracking_start_end": [0, 20000], "largest_camera_break_duration": 0,
            "file_duration_samples": 20000, "root_directory": str(root), "total_num_channels": 5,
            "headstage_sn": headstage_sn, "imec_probe_sn": "22420014283",
        }
    }))
    processing_settings['modify_files']['Operator']['get_spike_times']['min_spike_num'] = 0
    _make_operator([str(root)], processing_settings, []).split_clusters_to_sessions()
    saved_npy = root / "ephys" / "imec0" / "cluster_data" / "imec0_cl0000_ch010_good.npy"
    before = saved_npy.read_bytes()

    def _failing_save(file, **_kwargs):
        """Write a truncated .npy magic into the open handle and fail, like a disk filling mid-save."""
        file.write(b"\x93NUMPY")
        message = "disk full"
        raise OSError(message)

    mocker.patch("usv_playpen.processing.modify_files.np.save", side_effect=_failing_save)
    with pytest.raises(OSError, match="disk full"):
        _make_operator([str(root)], processing_settings, []).split_clusters_to_sessions()
    assert saved_npy.read_bytes() == before
    assert [path.name for path in saved_npy.parent.iterdir()] == [saved_npy.name]


@pytest.mark.parametrize(("stored_start", "stored_tracking"), [(None, [10, 20]), (64, [74, 84])])
def test_concatenate_binary_files_rerun_keeps_tracking_on_the_concatenated_clock(
        tmp_path, processing_settings, mocker, stored_start, stored_tracking):
    """
    Description
    -----------
    The second of two recordings starts at sample 64 of the concatenation and was
    tracked over samples 10-20 of its own file. Whether the stored tracking window is
    file-relative (no concatenation ran yet: session start null) or already on the
    concatenated clock (an earlier concatenation), concatenating, and concatenating
    again, leaves it at 74-84: the file's offset is applied once, never per run.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Per-test temp dir; sessions rooted at ``<tmp>/Data/<id>``.
    processing_settings (dict)
        Package processing-settings fixture.
    mocker (pytest_mock.MockerFixture)
        No-ops the interactive ``smart_wait``.
    stored_start (int | None)
        The second recording's stored session start before the run (None = NaN).
    stored_tracking (list[int])
        Its stored tracking window before the run.

    Returns
    -------
    None
    """

    mocker.patch("usv_playpen.processing.modify_files.smart_wait")
    headstage_sn = _headstage_sn()
    roots = []
    for session_id in ("20250101_120000", "20250101_130000"):
        root = tmp_path / "Data" / session_id
        imec_dir = root / "ephys" / "imec0"
        imec_dir.mkdir(parents=True)
        _write_synthetic_binary(imec_dir, f"{session_id}.imec0", headstage_sn, n_samples_per_channel=64)
        roots.append(str(root))
    ephys_base = tmp_path / "EPHYS" / "20250101_imec0"
    ephys_base.mkdir(parents=True)
    changepoints_json = ephys_base / "changepoints_info_20250101_imec0.json"
    second = {"session_start_end": [stored_start if stored_start is not None else float("nan"),
                                    stored_start + 64 if stored_start is not None else float("nan")],
              "tracking_start_end": stored_tracking, "largest_camera_break_duration": 1.0,
              "file_duration_samples": 64, "root_directory": roots[1], "total_num_channels": 5,
              "headstage_sn": headstage_sn, "imec_probe_sn": "P"}
    changepoints_json.write_text(json.dumps({"20250101_130000.imec0": second}, indent=4))

    for _ in range(2):
        _make_operator(roots, processing_settings, []).concatenate_binary_files()
        rec = json.loads(changepoints_json.read_text())["20250101_130000.imec0"]
        assert rec["session_start_end"] == [64, 128]
        assert rec["tracking_start_end"] == [74, 84]


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


def _patch_concat_ffmpeg(mocker, written_bytes):
    """
    Description
    -----------
    Stand in for the ffmpeg concat-demuxer runs of ``concatenate_video_files``:
    ``subprocess.Popen`` records each invocation (argv + working directory) and
    ``wait_for_subprocesses`` writes ``written_bytes`` to every requested output
    file, the way a successful ffmpeg run would, and reports a zero exit code.

    Parameters
    ----------
    mocker (pytest_mock.MockerFixture)
        Patches ``smart_wait``, ``subprocess.Popen`` and ``wait_for_subprocesses``
        in ``usv_playpen.processing.modify_files``.
    written_bytes (bytes)
        Content written to each concatenated output file.

    Returns
    -------
    invocations (list of dict)
        One ``{'args': list, 'cwd': pathlib.Path}`` entry per ffmpeg invocation,
        filled as the method under test runs.
    """

    invocations = []

    def _fake_popen(**kwargs):
        """Record the ffmpeg argv and working directory of one invocation."""
        invocations.append({'args': kwargs['args'], 'cwd': pathlib.Path(kwargs['cwd'])})
        return mocker.MagicMock(returncode=0)

    def _fake_wait(**kwargs):
        """Write each invocation's output file (the last argv entry) and report success."""
        for invocation in invocations:
            (invocation['cwd'] / invocation['args'][-1]).write_bytes(written_bytes)
        return [0] * len(kwargs['subps'])

    mocker.patch("usv_playpen.processing.modify_files.smart_wait")
    mocker.patch("usv_playpen.processing.modify_files.subprocess.Popen", side_effect=_fake_popen)
    mocker.patch("usv_playpen.processing.modify_files.wait_for_subprocesses", side_effect=_fake_wait)
    return invocations


def test_concatenate_video_files_replaces_stale_output(tmp_path, processing_settings, mocker):
    """
    Description
    -----------
    Regression: a concatenated file left in the camera folder by a crashed run
    used to be globbed into its own ffmpeg input list, and because ffmpeg ran
    without ``-y`` it refused to overwrite ("Not overwriting - exiting") yet
    exited 0, so the OLD file was moved out and published as the session video.

    The stale file must be excluded from the concat list and removed (with a
    logged message), ffmpeg must run with ``-y``, and the published file must be
    the freshly concatenated one.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Per-test session root.
    processing_settings (dict)
        Package processing-settings fixture.
    mocker (pytest_mock.MockerFixture)
        Stubs ffmpeg via ``_patch_concat_ffmpeg``.

    Returns
    -------
    None
    """

    concat = processing_settings['modify_files']['Operator']['concatenate_video_files']
    serial = "21372315"
    concat['concatenate_camera_serial_num'] = [serial]
    concat_name = concat['concatenated_video_name']
    vid_ext = concat['concatenate_video_extension']

    cam_dir = tmp_path / "video" / f"20250101_120000.{serial}"
    cam_dir.mkdir(parents=True)
    (cam_dir / f"000000.{vid_ext}").write_bytes(b"chunk0")
    (cam_dir / f"000001.{vid_ext}").write_bytes(b"chunk1")
    (cam_dir / f"{concat_name}_{serial}.{vid_ext}").write_bytes(b"stale")

    invocations = _patch_concat_ffmpeg(mocker, written_bytes=b"fresh")
    list_contents = []
    original_unlink = pathlib.Path.unlink

    def _capture_list_then_unlink(self, *args, **kwargs):
        """Record the concat list's lines just before the method deletes it."""
        if self.name.startswith("file_concatenation_list_"):
            list_contents.extend(self.read_text(encoding="utf-8").splitlines())
        return original_unlink(self, *args, **kwargs)

    mocker.patch.object(pathlib.Path, "unlink", _capture_list_then_unlink)

    messages = []
    _make_operator(str(tmp_path), processing_settings, messages).concatenate_video_files()

    assert len(invocations) == 1
    assert "-y" in invocations[0]['args']
    assert list_contents == [f"file '000000.{vid_ext}'", f"file '000001.{vid_ext}'"]
    published = tmp_path / "video" / f"{concat_name}_{serial}.{vid_ext}"
    assert published.read_bytes() == b"fresh"
    assert not (cam_dir / f"{concat_name}_{serial}.{vid_ext}").exists()
    assert any("stale" in message.lower() for message in messages)


def test_concatenate_video_files_leaves_single_video_and_skips_empty_folder(tmp_path, processing_settings, mocker):
    """
    Description
    -----------
    Regression: the copy-back loop used to delete the concat list and move the
    concatenated file of EVERY camera folder, but both exist only for folders
    with more than one video chunk, so a folder with one video (or none) raised
    FileNotFoundError after the other cameras had already been concatenated.

    A folder with two chunks is concatenated and published. The single-video
    folder has nothing to join, so it is left exactly as it was (no ffmpeg run,
    no concat list, no copy or renamed file in ``video/``) with a logged
    message: re-encoding reads that video straight from the camera folder. The
    empty folder is skipped with a logged message.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Per-test session root.
    processing_settings (dict)
        Package processing-settings fixture.
    mocker (pytest_mock.MockerFixture)
        Stubs ffmpeg via ``_patch_concat_ffmpeg``.

    Returns
    -------
    None
    """

    concat = processing_settings['modify_files']['Operator']['concatenate_video_files']
    multi_serial, single_serial, empty_serial = "21372315", "21372316", "21369048"
    concat['concatenate_camera_serial_num'] = [multi_serial, single_serial, empty_serial]
    concat_name = concat['concatenated_video_name']
    vid_ext = concat['concatenate_video_extension']

    video_dir = tmp_path / "video"
    multi_dir = video_dir / f"20250101_120000.{multi_serial}"
    single_dir = video_dir / f"20250101_120000.{single_serial}"
    empty_dir = video_dir / f"20250101_120000.{empty_serial}"
    for cam_dir in (multi_dir, single_dir, empty_dir):
        cam_dir.mkdir(parents=True)
    (multi_dir / f"000000.{vid_ext}").write_bytes(b"chunk0")
    (multi_dir / f"000001.{vid_ext}").write_bytes(b"chunk1")
    (single_dir / f"000000.{vid_ext}").write_bytes(b"only")

    invocations = _patch_concat_ffmpeg(mocker, written_bytes=b"fresh")

    messages = []
    _make_operator(str(tmp_path), processing_settings, messages).concatenate_video_files()

    assert [invocation['cwd'] for invocation in invocations] == [multi_dir]
    assert (video_dir / f"{concat_name}_{multi_serial}.{vid_ext}").read_bytes() == b"fresh"
    assert not (multi_dir / f"file_concatenation_list_{multi_serial}.txt").exists()
    assert not (video_dir / f"{concat_name}_{single_serial}.{vid_ext}").exists()
    assert sorted(path.name for path in single_dir.iterdir()) == [f"000000.{vid_ext}"]
    assert (single_dir / f"000000.{vid_ext}").read_bytes() == b"only"
    assert sum("nothing to join" in message for message in messages) == 1
    assert not (video_dir / f"{concat_name}_{empty_serial}.{vid_ext}").exists()
    assert sum("holds no video files" in message for message in messages) == 1


def _patch_rectify_ffmpeg(mocker):
    """
    Description
    -----------
    Stand in for the ffmpeg re-encoding runs of ``rectify_video_fps``:
    ``subprocess.Popen`` records each invocation (argv + working directory),
    creates the output file (the last argv entry) in the working directory so
    the later move succeeds, and returns a process that exits with 0.

    Parameters
    ----------
    mocker (pytest_mock.MockerFixture)
        Patches ``smart_wait`` and ``subprocess.Popen`` in
        ``usv_playpen.processing.modify_files``.

    Returns
    -------
    invocations (list of dict)
        One ``{'args': list, 'cwd': pathlib.Path}`` entry per ffmpeg invocation,
        filled as the method under test runs.
    """

    invocations = []

    def _fake_popen(args, cwd=None, **_kwargs):
        """Record the invocation, create its output file in ``cwd`` and return a process that exits with 0."""
        invocations.append({'args': args, 'cwd': pathlib.Path(cwd)})
        (pathlib.Path(cwd) / args[-1]).write_bytes(b"\x00")
        proc = mocker.MagicMock()
        proc.poll.return_value = 0
        proc.wait.return_value = 0
        return proc

    mocker.patch("usv_playpen.processing.modify_files.smart_wait")
    mocker.patch("usv_playpen.processing.modify_files.subprocess.Popen", side_effect=_fake_popen)
    return invocations


def _rectify_input(invocation):
    """
    Description
    -----------
    Return the input video path of one recorded ffmpeg re-encoding invocation
    (the argv entry after ``-i``).

    Parameters
    ----------
    invocation (dict)
        One entry recorded by ``_patch_rectify_ffmpeg``.

    Returns
    -------
    input_path (str)
        The ``-i`` argument.
    """

    return invocation['args'][invocation['args'].index('-i') + 1]


def test_rectify_video_fps_finds_each_camera_input_and_handles_calibration(tmp_path, processing_settings, mocker):
    """
    Description
    -----------
    ``rectify_video_fps`` finds each camera's input on disk, with no file name
    setting:

    (a) a camera whose concatenated ``<concatenated_video_name>_<serial>`` file
        exists in ``video/`` re-encodes that file, which ``delete_old_file``
        then deletes as a disposable intermediate, even though its camera folder
        still holds several chunks;
    (b) a camera with a single ``000000`` video re-encodes it straight from the
        camera folder (absolute ``-i`` path, output still written to ``video/``),
        a stale concatenation output beside it is not counted as a second
        video, and the raw recording is never deleted;
    (c) the calibration camera re-encodes its ``000000`` source in its own folder
        into ``calibration_images`` and keeps the raw source;
    (d) the metadata ``session_duration`` and the frame-count JSON are written.

    ``new_for_filename`` (imgstore) and ``subprocess.Popen`` (ffmpeg) are mocked.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Per-test session root.
    processing_settings (dict)
        Package processing-settings fixture.
    mocker (pytest_mock.MockerFixture)
        Patches imgstore, metadata I/O and ffmpeg.

    Returns
    -------
    None
    """

    rectify = processing_settings['modify_files']['Operator']['rectify_video_fps']
    concat_serial, single_serial = "21372315", "21372316"
    rectify['encode_camera_serial_num'] = [concat_serial, single_serial]
    rectify['delete_old_file'] = True
    vid_ext = rectify['encode_video_extension']
    concat_name = processing_settings['modify_files']['Operator']['concatenate_video_files']['concatenated_video_name']

    date_token = "20250101_120000"
    date_joint = "20250101120000"
    video_dir = tmp_path / "video"

    concat_cam_dir = video_dir / f"{date_token}.{concat_serial}"
    concat_cam_dir.mkdir(parents=True)
    (concat_cam_dir / f"000000.{vid_ext}").write_bytes(b"chunk0")
    (concat_cam_dir / f"000001.{vid_ext}").write_bytes(b"chunk1")
    concat_product = video_dir / f"{concat_name}_{concat_serial}.{vid_ext}"
    concat_product.write_bytes(b"joined")

    single_cam_dir = video_dir / f"{date_token}.{single_serial}"
    single_cam_dir.mkdir(parents=True)
    (single_cam_dir / f"000000.{vid_ext}").write_bytes(b"raw")
    (single_cam_dir / f"{concat_name}_{single_serial}.{vid_ext}").write_bytes(b"stale")

    calib_cam_dir = video_dir / f"calibration_{date_token}.{concat_serial}"
    calib_cam_dir.mkdir(parents=True)
    (calib_cam_dir / f"000000.{vid_ext}").write_bytes(b"calib")

    esr_times = np.linspace(0, 1.0, 30)
    _patch_imgstore(mocker, total_frame_num=30, esr_frame_times=esr_times, has_dropped=False)

    metadata = {'Session': {'session_duration': 0.0, 'camera_serials': [concat_serial, single_serial]}}
    metadata_path = tmp_path / f"{tmp_path.name}_metadata.yaml"
    mocker.patch(
        "usv_playpen.processing.modify_files.load_session_metadata",
        return_value=(metadata, metadata_path),
    )
    saved = {}

    def _fake_save(data, filepath, **_kwargs):
        """Record the metadata dict and target path instead of writing the YAML."""
        saved['data'] = data
        saved['filepath'] = filepath

    mocker.patch("usv_playpen.processing.modify_files.save_session_metadata", side_effect=_fake_save)
    invocations = _patch_rectify_ffmpeg(mocker)

    messages = []
    _make_operator(str(tmp_path), processing_settings, messages).rectify_video_fps()

    by_input = {_rectify_input(invocation): invocation for invocation in invocations}
    assert len(invocations) == 3

    # (a) the concatenated file is the input and is deleted afterwards
    assert by_input[str(concat_product)]['cwd'] == video_dir
    assert not concat_product.exists()
    assert (video_dir / date_joint / concat_serial / f"{concat_serial}-{date_joint}.{vid_ext}").is_file()

    # (b) the single video is read in place, the stale file ignored, the raw kept
    single_input = single_cam_dir / f"000000.{vid_ext}"
    assert by_input[str(single_input)]['cwd'] == video_dir
    assert single_input.read_bytes() == b"raw"
    assert (video_dir / date_joint / single_serial / f"{single_serial}-{date_joint}.{vid_ext}").is_file()
    assert sum("single video (000000" in message for message in messages) == 1

    # (c) calibration: own folder, calibration_images destination, raw kept
    assert by_input[f"000000.{vid_ext}"]['cwd'] == calib_cam_dir
    assert (video_dir / date_joint / concat_serial / 'calibration_images' / f"{concat_serial}-{date_joint}-calibration.{vid_ext}").is_file()
    assert (calib_cam_dir / f"000000.{vid_ext}").read_bytes() == b"calib"

    # (d) metadata and frame counts
    assert saved['data']['Session']['session_duration'] == pytest.approx(round(esr_times[-1] - esr_times[0], 3))
    assert (video_dir / f"{date_joint}_camera_frame_count_dict.json").is_file()


def test_rectify_video_fps_several_videos_without_concatenation_raises(tmp_path, processing_settings, mocker):
    """
    Description
    -----------
    A camera folder holding several videos with no concatenated file in
    ``video/`` stops re-encoding with a ValueError asking to run concatenation
    first (re-encoding one chunk would give a partial video). Every camera is
    checked before any ffmpeg process starts, so the camera listed first, whose
    single video could be re-encoded, has not been started either.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Per-test session root.
    processing_settings (dict)
        Package processing-settings fixture.
    mocker (pytest_mock.MockerFixture)
        Patches ffmpeg.

    Returns
    -------
    None
    """

    rectify = processing_settings['modify_files']['Operator']['rectify_video_fps']
    single_serial, multi_serial = "21372315", "21372316"
    rectify['encode_camera_serial_num'] = [single_serial, multi_serial]
    vid_ext = rectify['encode_video_extension']

    video_dir = tmp_path / "video"
    single_cam_dir = video_dir / f"20250101_120000.{single_serial}"
    multi_cam_dir = video_dir / f"20250101_120000.{multi_serial}"
    single_cam_dir.mkdir(parents=True)
    multi_cam_dir.mkdir(parents=True)
    (single_cam_dir / f"000000.{vid_ext}").write_bytes(b"raw")
    (multi_cam_dir / f"000000.{vid_ext}").write_bytes(b"chunk0")
    (multi_cam_dir / f"000001.{vid_ext}").write_bytes(b"chunk1")

    invocations = _patch_rectify_ffmpeg(mocker)

    with pytest.raises(ValueError, match=rf"Camera {multi_serial}: .* holds 2 videos .*run video concatenation first"):
        _make_operator(str(tmp_path), processing_settings, []).rectify_video_fps()

    assert invocations == []
    assert sorted(path.name for path in multi_cam_dir.iterdir()) == [f"000000.{vid_ext}", f"000001.{vid_ext}"]


def test_rectify_video_fps_camera_without_video_raises(tmp_path, processing_settings, mocker):
    """
    Description
    -----------
    A listed camera folder with no video and no concatenated file in ``video/``
    stops re-encoding with a FileNotFoundError before any ffmpeg process starts.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Per-test session root.
    processing_settings (dict)
        Package processing-settings fixture.
    mocker (pytest_mock.MockerFixture)
        Patches ffmpeg.

    Returns
    -------
    None
    """

    rectify = processing_settings['modify_files']['Operator']['rectify_video_fps']
    serial = "21372315"
    rectify['encode_camera_serial_num'] = [serial]

    (tmp_path / "video" / f"20250101_120000.{serial}").mkdir(parents=True)
    invocations = _patch_rectify_ffmpeg(mocker)

    with pytest.raises(FileNotFoundError, match=rf"Camera {serial}: .* holds no .* video .*nothing to re-encode"):
        _make_operator(str(tmp_path), processing_settings, []).rectify_video_fps()

    assert invocations == []


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

    with (tmp_path / "audio" / "broadband_filtered" / BROADBAND_REPORT_NAME).open(encoding="utf-8") as report_file:
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

    # the report stores each column's low-band variance: here the 500 Hz sine (amplitude 3000) the
    # high-pass removed, (3000 / 32768)^2 / 2 on soundfile's float scale, plus the removed tones
    low_band = report['low_band_variance']
    assert low_band['windows'] == processing_settings['modify_files']['Operator']['broadband_filter_audio']['low_band_variance_windows']
    expected = (3000.0 / 32768.0) ** 2 / 2
    for column, value in enumerate(low_band['per_column']):
        assert abs(value - expected) / expected < 0.05, (column, value, expected)


def test_estimate_low_band_variance_recovers_the_removed_band(tmp_path):
    """Wavs = low tone + high tone + noise, memmap = high tone + noise (what a high-pass leaves):
    the estimate is the low tone's variance on the float scale, per column."""
    sr = 250000
    rng = np.random.default_rng(11)
    t = np.arange(2 * sr) / sr
    noise = rng.normal(0.0, 100.0, size=(t.size, 2))
    high = 500.0 * np.sin(2 * np.pi * 40000.0 * t)
    low_amplitudes = (2000.0, 800.0)
    hpss = tmp_path / "audio" / "hpss"
    hpss.mkdir(parents=True)
    names = ["m_230101120000_ch01_cropped_to_video_hpss.wav", "m_230101120000_ch02_cropped_to_video_hpss.wav"]
    channels = []
    for column, (name, amplitude) in enumerate(zip(names, low_amplitudes, strict=True)):
        low = amplitude * np.sin(2 * np.pi * 300.0 * t)
        sf.write(str(hpss / name), np.clip(low + high + noise[:, column], -32768, 32767).astype(np.int16), sr, subtype="PCM_16")
        channels.append((name, np.clip(high + noise[:, column], -32768, 32767).astype(np.int16)))
    mmap_path = write_broadband_audio(tmp_path, channels, sr)
    estimate = estimate_low_band_variance([hpss / name for name in names], mmap_path, n_windows=12, window_s=0.2, seed=0)
    assert estimate['windows'] == 12
    assert estimate['unit'] == "(int16 / 32768)^2"
    for value, amplitude in zip(estimate['per_column'], low_amplitudes, strict=True):
        expected = (amplitude / 32768.0) ** 2 / 2
        assert abs(value - expected) / expected < 0.05


def test_add_broadband_low_band_variance_fills_an_older_report(tmp_path, processing_settings, mocker):
    """A report written before the field existed gets it added (everything else kept); a report that
    has it is skipped unless forced; an incomplete session is refused."""
    mocker.patch("usv_playpen.processing.modify_files.smart_wait")
    _broadband_settings(processing_settings)
    _write_broadband_session(tmp_path)
    _make_operator(str(tmp_path), processing_settings, []).broadband_filter_audio()
    settings = processing_settings['modify_files']['Operator']['broadband_filter_audio']
    report_path = tmp_path / "audio" / "broadband_filtered" / BROADBAND_REPORT_NAME
    with report_path.open(encoding="utf-8") as report_file:
        written = json.load(report_file)
    older = {key: value for key, value in written.items() if key != 'low_band_variance'}
    report_path.write_text(json.dumps(older, indent=2))

    result = add_broadband_low_band_variance(tmp_path, settings, message_output=lambda *_a, **_kw: None)
    assert result['status'] == 'written'
    with report_path.open(encoding="utf-8") as report_file:
        restored = json.load(report_file)
    assert {key: value for key, value in restored.items() if key != 'low_band_variance'} == older
    assert restored['low_band_variance']['per_column'] == pytest.approx(written['low_band_variance']['per_column'])
    assert add_broadband_low_band_variance(tmp_path, settings, message_output=lambda *_a, **_kw: None)['status'] == 'skipped'
    assert add_broadband_low_band_variance(tmp_path, settings, message_output=lambda *_a, **_kw: None, force=True)['status'] == 'written'
    report_path.write_text(json.dumps({**restored, 'complete': False}, indent=2))
    with pytest.raises(ValueError, match="not complete and current"):
        add_broadband_low_band_variance(tmp_path, settings, message_output=lambda *_a, **_kw: None)


def test_add_broadband_low_band_variance_cli_runs_every_listed_session(tmp_path, mocker):
    """The command collects sessions from --root-directories and --sessions-file (each once) and
    reports each one's status."""
    added = mocker.patch("usv_playpen.processing.preprocess_data.add_broadband_low_band_variance",
                         side_effect=lambda _root, _settings, **_kwargs: {'status': 'written', 'reason': 'added'})
    sessions_file = tmp_path / "sessions.txt"
    sessions_file.write_text("/data/b\n/data/c\n")
    result = CliRunner().invoke(preprocess_data.add_broadband_low_band_variance_cli,
                                ["--root-directories", "/data/a,/data/b", "--sessions-file", str(sessions_file), "--workers", "2", "--force"])
    assert result.exit_code == 0, result.output
    assert sorted(call.args[0] for call in added.call_args_list) == ["/data/a", "/data/b", "/data/c"]
    assert all(call.kwargs["force"] is True for call in added.call_args_list)
    assert result.output.count("written (added)") == 3
    result = CliRunner().invoke(preprocess_data.add_broadband_low_band_variance_cli, [])
    assert result.exit_code == 2


def test_add_broadband_low_band_variance_cli_lists_the_failed_sessions(mocker):
    """A session that raises is reported with its error type and the batch goes on;
    the run ends with a line naming every failed (skipped) session."""

    def _added(root, _settings, **_kwargs):
        """Fail for ``/data/b`` and report every other session as written."""
        if root == "/data/b":
            message = "memmap missing"
            raise OSError(message)
        return {'status': 'written', 'reason': 'added'}

    mocker.patch("usv_playpen.processing.preprocess_data.add_broadband_low_band_variance", side_effect=_added)
    result = CliRunner().invoke(preprocess_data.add_broadband_low_band_variance_cli,
                                ["--root-directories", "/data/a,/data/b,/data/c", "--workers", "1"])
    assert result.exit_code == 0, result.output
    assert "/data/b: FAILED (OSError: memmap missing)" in result.output
    assert result.output.count("written (added)") == 2
    assert "1 of 3 session(s) FAILED and were skipped: /data/b" in result.output


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
    with report_path.open(encoding="utf-8") as report_file:
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


def test_operator_requires_a_root_directory(processing_settings):
    """root_directory is required: None raises a ValueError instead of a KeyError on
    a settings key that does not exist."""
    with pytest.raises(ValueError, match="needs a root_directory"):
        Operator(root_directory=None, input_parameter_dict=processing_settings)
    with pytest.raises(ValueError, match="needs a root_directory"):
        Operator()


def test_concatenate_video_files_stale_output_beside_one_chunk_is_removed(tmp_path, processing_settings, mocker):
    """
    Description
    -----------
    A camera folder holding ONE real video chunk next to a stale
    ``<concatenated_video_name>_<serial>`` output of a crashed run: the stale file
    must not count as a second input (which would have sent the folder down the
    ffmpeg concatenation path with the old output as one of its own inputs), it
    must be deleted with a logged message, and the single real chunk is left in
    its camera folder with nothing copied to ``video/`` (re-encoding reads it in
    place). A calibration folder and a
    folder whose serial is not listed in ``concatenate_camera_serial_num`` are
    left untouched, no ffmpeg process is started, and the ffmpeg wait receives
    the ``subprocess_timeout_hours`` tunable converted to seconds.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Per-test session root.
    processing_settings (dict)
        Package processing-settings fixture.
    mocker (pytest_mock.MockerFixture)
        Stubs ffmpeg via ``_patch_concat_ffmpeg``.

    Returns
    -------
    None
    """

    concat = processing_settings['modify_files']['Operator']['concatenate_video_files']
    serial, unlisted_serial = "21372315", "21372399"
    concat['concatenate_camera_serial_num'] = [serial]
    concat['subprocess_timeout_hours'] = 0.5
    concat_name = concat['concatenated_video_name']
    vid_ext = concat['concatenate_video_extension']

    video_dir = tmp_path / "video"
    cam_dir = video_dir / f"20250101_120000.{serial}"
    calibration_dir = video_dir / f"calibration_20250101_120000.{serial}"
    unlisted_dir = video_dir / f"20250101_120000.{unlisted_serial}"
    for one_dir in (cam_dir, calibration_dir, unlisted_dir):
        one_dir.mkdir(parents=True)
    (cam_dir / f"000000.{vid_ext}").write_bytes(b"raw-chunk")
    (cam_dir / f"{concat_name}_{serial}.{vid_ext}").write_bytes(b"stale")
    (calibration_dir / f"000000.{vid_ext}").write_bytes(b"calib")
    (unlisted_dir / f"000000.{vid_ext}").write_bytes(b"other")

    invocations = _patch_concat_ffmpeg(mocker, written_bytes=b"fresh")

    messages = []
    _make_operator(str(tmp_path), processing_settings, messages).concatenate_video_files()

    assert invocations == []
    assert modify_files.wait_for_subprocesses.call_args.kwargs['max_seconds'] == 0.5 * 60 * 60
    assert modify_files.wait_for_subprocesses.call_args.kwargs['subps'] == []
    assert not (video_dir / f"{concat_name}_{serial}.{vid_ext}").exists()
    assert (cam_dir / f"000000.{vid_ext}").read_bytes() == b"raw-chunk"
    assert not (cam_dir / f"{concat_name}_{serial}.{vid_ext}").exists()
    assert sorted(path.name for path in cam_dir.iterdir()) == [f"000000.{vid_ext}"]
    assert sum("stale" in message.lower() and "1 video chunk(s)" in message for message in messages) == 1
    assert sum("nothing to join" in message for message in messages) == 1
    assert sorted(path.name for path in calibration_dir.iterdir()) == [f"000000.{vid_ext}"]
    assert sorted(path.name for path in unlisted_dir.iterdir()) == [f"000000.{vid_ext}"]
    assert not (video_dir / f"{concat_name}_{unlisted_serial}.{vid_ext}").exists()
    assert sorted(path.name for path in video_dir.iterdir() if path.is_file()) == []


def test_concatenate_binary_files_rejects_a_meta_missing_required_keys(tmp_path, processing_settings, mocker):
    """
    Description
    -----------
    A SpikeGLX ``.meta`` without ``imDatPrb_sn`` (or any other of the three
    required keys) must fail loudly with a ``KeyError`` that names the file and
    the missing key, instead of silently reusing the previous file's parsed
    value or crashing later in the sample arithmetic. No concatenated output is
    written.

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

    root = tmp_path / "Data" / "20250101_120000"
    imec_dir = root / "ephys" / "imec0"
    imec_dir.mkdir(parents=True)
    _write_synthetic_binary(imec_dir, "20250101_120000.imec0", _headstage_sn(), n_samples_per_channel=16)
    meta_path = imec_dir / "20250101_120000.imec0.ap.meta"
    meta_path.write_text("".join(line + "\n" for line in meta_path.read_text().splitlines() if not line.startswith("imDatPrb_sn=")))

    with pytest.raises(KeyError, match=r"missing required meta key\(s\): imDatPrb_sn"):
        _make_operator([str(root)], processing_settings, []).concatenate_binary_files()
    assert not (tmp_path / "EPHYS" / "20250101_imec0" / "concatenated_20250101_imec0.ap.bin").exists()


def test_concatenate_binary_files_without_any_source_meta_raises(tmp_path, processing_settings, mocker):
    """
    Description
    -----------
    A probe folder holding a ``.bin`` without its ``.meta`` contributes no
    recording, and with no ``.meta`` anywhere to serve as the template of the
    concatenated ``.meta`` the method must raise a ``FileNotFoundError`` that
    names the file pattern, the probe folder and the session searched, before
    any concatenated ``.meta`` or changepoints JSON is written.

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

    root = tmp_path / "Data" / "20250101_120000"
    imec_dir = root / "ephys" / "imec0"
    imec_dir.mkdir(parents=True)
    _write_synthetic_binary(imec_dir, "20250101_120000.imec0", _headstage_sn(), n_samples_per_channel=16)
    (imec_dir / "20250101_120000.imec0.ap.meta").unlink()

    with pytest.raises(FileNotFoundError, match=r"no '\*ap\.bin' with a sibling \.meta under 'ephys/imec0'") as raised:
        _make_operator([str(root)], processing_settings, []).concatenate_binary_files()
    assert str(root) in str(raised.value)
    ephys_base = tmp_path / "EPHYS" / "20250101_imec0"
    assert not (ephys_base / "concatenated_20250101_imec0.ap.meta").exists()
    assert not (ephys_base / "changepoints_info_20250101_imec0.json").exists()


def test_concatenate_binary_files_probe_absent_from_the_last_session(tmp_path, processing_settings, mocker):
    """
    Description
    -----------
    Two sessions are concatenated; ``imec0`` is recorded in both, ``imec1``
    only in the first. The ``imec1`` output must be the first session's
    recording alone, with its concatenated ``.meta`` built from that session's
    ``.meta`` (the template is the first concatenated recording's ``.meta``,
    not a search of the last session's probe folder, which does not exist for
    ``imec1``). For ``imec0`` the template is likewise the FIRST session's
    ``.meta``: every line except the size and duration fields matches it.

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
    for root in (root_a, root_b):
        (root / "ephys" / "imec0").mkdir(parents=True)
        _write_synthetic_binary(root / "ephys" / "imec0", f"{root.name}.imec0", headstage_sn, n_samples_per_channel=32)
        # A per-session start field, so the output reveals WHICH session's .meta was the template.
        imec0_meta = root / "ephys" / "imec0" / f"{root.name}.imec0.ap.meta"
        imec0_meta.write_text(imec0_meta.read_text() + f"fileCreateTime={root.name}\n")
    (root_a / "ephys" / "imec1").mkdir()
    _write_synthetic_binary(root_a / "ephys" / "imec1", f"{root_a.name}.imec1", headstage_sn, n_samples_per_channel=32)

    _make_operator([str(root_a), str(root_b)], processing_settings, []).concatenate_binary_files()

    imec1_base = tmp_path / "EPHYS" / "20250101_imec1"
    info = json.loads((imec1_base / "changepoints_info_20250101_imec1.json").read_text())
    assert info["20250101_120000.imec1"]["session_start_end"] == [0, 32]
    assert (imec1_base / "concatenated_20250101_imec1.ap.meta").is_file()

    for probe_id in ("imec0", "imec1"):
        source_lines = (root_a / "ephys" / probe_id / f"{root_a.name}.{probe_id}.ap.meta").read_text().splitlines()
        output_lines = (tmp_path / "EPHYS" / f"20250101_{probe_id}"
                        / f"concatenated_20250101_{probe_id}.ap.meta").read_text().splitlines()
        assert len(output_lines) == len(source_lines)
        for source_line, output_line in zip(source_lines, output_lines, strict=True):
            if not source_line.startswith(("fileSizeBytes=", "fileTimeSecs=")):
                assert output_line == source_line


def test_unit_phasor_and_block_mean_phasors_handle_empty_and_short_input():
    """
    Description
    -----------
    Edge cases of the complex demodulation helpers: a zero-length phasor and
    an empty sample stretch both return an empty complex128 array, and a stretch
    shorter than one block is demodulated as a single (partial) tail block whose
    value is the plain mean of ``x[n] exp(-i w n)`` over the absolute sample
    indices ``n`` of that stretch; for a tone ``A cos(w n + phi)`` spanning a
    whole number of cycles that is ``(A / 2) exp(i phi)``.

    Parameters
    ----------

    Returns
    -------
    None
    """

    empty_phasor = unit_phasor(1000.0, 250000, 12345, 0)
    assert empty_phasor.shape == (0,)
    assert empty_phasor.dtype == np.complex128
    empty_blocks = block_mean_phasors(np.zeros(0), 1000.0, 250000, 0, 250)
    assert empty_blocks.shape == (0,)
    assert empty_blocks.dtype == np.complex128

    sampling_rate, frequency, start = 250000, 1000.0, 1000
    n = np.arange(start, start + 500)
    x = 40.0 * np.cos(2 * np.pi * frequency * n / sampling_rate + 0.7)
    phasors = block_mean_phasors(x, frequency, sampling_rate, start, block_length=4096)
    expected = np.mean(x * np.exp(-2j * np.pi * frequency * n / sampling_rate))
    assert phasors.shape == (1,)
    np.testing.assert_allclose(phasors[0], expected, rtol=1e-9)
    np.testing.assert_allclose(phasors[0], 20.0 * np.exp(0.7j), rtol=1e-9)


def test_estimate_line_noise_on_silence_reports_one_unkept_candidate(tmp_path):
    """
    Description
    -----------
    On an all-zero channel the whitened spectrum is zero everywhere, so the
    first (and only) candidate stands at the ``tiny``-clamped floor (about
    -3077 dB), far below ``min_height_db``: it is reported with ``kept`` False,
    at a frequency inside the band, without parabolic refinement (the zero power
    has no log), and the search stops there.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Per-test directory.

    Returns
    -------
    None
    """

    path = tmp_path / "m_230101120000_ch01_cropped_to_video_hpss.wav"
    wavfile.write(path, 250000, np.zeros(250000, dtype=np.int16))
    tones = estimate_line_noise(wav_path=path, search_bands_hz=[[7950, 8050]], min_height_db=8.0, n_windows=2, window_s=0.5,
                                max_tones_per_band=4, min_separation_hz=0.25, floor_window_hz=4.0)
    assert len(tones) == 1
    assert tones[0]['kept'] is False
    assert tones[0]['search_band_hz'] == [7950.0, 8050.0]
    assert 7950.0 <= tones[0]['frequency_hz'] <= 8050.0
    assert tones[0]['height_db'] == pytest.approx(10 * np.log10(np.finfo(np.float64).tiny), abs=0.01)


def test_estimate_line_noise_keeps_peaks_at_least_the_separation_apart(tmp_path):
    """
    Description
    -----------
    With a threshold every bin passes and no cap on the tone count, the greedy
    search walks the whole band: every bin closer than
    ``max(min_separation_hz, main lobe)`` to an already kept peak is skipped, so
    the kept tones are pairwise at least ``min_separation_hz`` apart (here 10 Hz
    on a 100 Hz band, so between 6 and 11 of them), all inside the band and all
    marked kept.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Per-test directory.

    Returns
    -------
    None
    """

    path = tmp_path / "m_230101120000_ch01_cropped_to_video_hpss.wav"
    wavfile.write(path, 250000, np.zeros(250000, dtype=np.int16))
    tones = estimate_line_noise(wav_path=path, search_bands_hz=[[7950, 8050]], min_height_db=-1e5, n_windows=2, window_s=0.5,
                                max_tones_per_band=1000, min_separation_hz=10.0, floor_window_hz=4.0)
    frequencies = np.sort([tone['frequency_hz'] for tone in tones])
    assert all(tone['kept'] for tone in tones)
    assert 6 <= frequencies.size <= 11
    assert np.min(np.diff(frequencies)) >= 10.0 - 1e-6
    assert frequencies[0] >= 7950.0
    assert frequencies[-1] <= 8050.0


def test_broadband_channel_stream_raises_on_a_short_read(tmp_path):
    """
    Description
    -----------
    A wav that holds fewer samples than the stream was told (a truncated or
    still-growing file) must raise an ``OSError`` naming the expected and the
    delivered sample counts, instead of silently padding the output.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Per-test directory.

    Returns
    -------
    None
    """

    path = tmp_path / "m_230101120000_ch01_cropped_to_video_hpss.wav"
    wavfile.write(path, 250000, np.zeros(1000, dtype=np.int16))
    taps, _ = design_broadband_highpass(250000, 2000, 1000, 120)
    stream = _BroadbandChannelStream(wav_path=path, n_samples=5000, sampling_rate=250000, taps=taps,
                                     tone_frequencies_hz=[], block_length=250, half_window=1)
    try:
        with pytest.raises(OSError, match=r"Short read from .* expected 5000 samples, got 1000"):
            stream.process(0, 5000)
    finally:
        stream.close()


def test_estimate_low_band_variance_rejects_a_memmap_shorter_than_one_window(tmp_path):
    """
    Description
    -----------
    A broadband memmap with fewer samples than one ``window_s`` window cannot
    give any variance window, so the estimate raises a ``ValueError`` that names
    the sample count and the window length.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Per-test session root.

    Returns
    -------
    None
    """

    name = "m_230101120000_ch01_cropped_to_video_hpss.wav"
    mmap_path = write_broadband_audio(tmp_path, [(name, np.zeros(1000, dtype=np.int16))], 250000)
    with pytest.raises(ValueError, match=re.escape("holds 1000 samples, fewer than one 0.2 s window")):
        estimate_low_band_variance([tmp_path / name], mmap_path, n_windows=3, window_s=0.2, seed=0)


def test_validate_broadband_output_names_every_reason_for_rejecting_an_output(tmp_path, processing_settings, mocker):
    """
    Description
    -----------
    After one real broadband run the output validates as ``complete and
    current``; each kind of damage is then rejected with its own reason: no
    source wavs, a memmap of the wrong byte size, a second broadband memmap, an
    unreadable report, a report naming another memmap, changed source files, and
    a report missing a required key.

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
    assert validate_broadband_output(tmp_path, settings) == (False, "no source wavs")
    _write_broadband_session(tmp_path)
    _make_operator(str(tmp_path), processing_settings, []).broadband_filter_audio()
    mmap_path = find_audio_mmap(tmp_path, "broadband")
    report_path = mmap_path.parent / BROADBAND_REPORT_NAME
    report_text = report_path.read_text(encoding="utf-8")
    report = json.loads(report_text)
    assert validate_broadband_output(tmp_path, settings) == (True, "complete and current")

    original_bytes = mmap_path.read_bytes()
    mmap_path.write_bytes(original_bytes[:-2])
    valid, reason = validate_broadband_output(tmp_path, settings)
    assert valid is False
    assert reason == f"{mmap_path.name} has {len(original_bytes) - 2} bytes, expected {len(original_bytes)}"
    mmap_path.write_bytes(original_bytes)

    other = mmap_path.parent / "230101120000_concatenated_audio_broadband_filtered_250000_10_3_int16.mmap"
    other.write_bytes(b"")
    assert validate_broadband_output(tmp_path, settings) == (False, f"other broadband memmaps present: {[other.name]}")
    other.unlink()

    report_path.write_text("{not json", encoding="utf-8")
    valid, reason = validate_broadband_output(tmp_path, settings)
    assert valid is False
    assert reason.startswith(f"unreadable {BROADBAND_REPORT_NAME}: ")

    report_path.write_text(json.dumps({**report, 'output': {**report['output'], 'file': other.name}}), encoding="utf-8")
    assert validate_broadband_output(tmp_path, settings) == (False, "report names another memmap")

    changed_sources = [{**source, 'bytes': source['bytes'] + 1} for source in report['sources']]
    report_path.write_text(json.dumps({**report, 'sources': changed_sources}), encoding="utf-8")
    assert validate_broadband_output(tmp_path, settings) == (False, "source files changed")

    report_path.write_text(json.dumps({key: value for key, value in report.items() if key != 'settings'}), encoding="utf-8")
    assert validate_broadband_output(tmp_path, settings) == (False, f"malformed {BROADBAND_REPORT_NAME}: missing 'settings'")

    report_path.write_text(report_text, encoding="utf-8")
    assert validate_broadband_output(tmp_path, settings) == (True, "complete and current")


def test_broadband_filter_audio_rejects_missing_or_mismatched_sources_and_even_smoothing(tmp_path, processing_settings, mocker):
    """
    Description
    -----------
    The input checks of ``broadband_filter_audio``: a session without source
    wavs raises a ``FileNotFoundError`` naming the glob and folder; a source
    wav at another sampling rate raises a ``ValueError`` naming the offending
    file; and an even ``line_noise_smoothing_blocks`` (the running median needs
    a centre block) raises a ``ValueError`` naming the value. None of them leaves
    a broadband memmap behind.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Per-test directory (three session roots).
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

    empty_root = tmp_path / "empty"
    (empty_root / "audio" / settings['source_dir']).mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match=f"No '{re.escape(settings['source_glob'])}' files in"):
        _make_operator(str(empty_root), processing_settings, []).broadband_filter_audio()

    mismatched_root = tmp_path / "mismatched"
    hpss_dir = mismatched_root / "audio" / "hpss"
    hpss_dir.mkdir(parents=True)
    wavfile.write(hpss_dir / "m_230101120000_ch01_cropped_to_video_hpss.wav", 250000, np.zeros(2000, dtype=np.int16))
    wavfile.write(hpss_dir / "m_230101120000_ch02_cropped_to_video_hpss.wav", 125000, np.zeros(2000, dtype=np.int16))
    with pytest.raises(ValueError, match=re.escape("'m_230101120000_ch02_cropped_to_video_hpss.wav' is 1 ch PCM_16 at 125000 Hz")) as raised:
        _make_operator(str(mismatched_root), processing_settings, []).broadband_filter_audio()
    assert "every source must be 1 ch PCM_16 at 250000 Hz with 2000 samples" in str(raised.value)

    even_root = tmp_path / "even"
    _write_broadband_session(even_root)
    settings['line_noise_smoothing_blocks'] = 4
    with pytest.raises(ValueError, match="line_noise_smoothing_blocks must be a positive odd integer, got 4"):
        _make_operator(str(even_root), processing_settings, []).broadband_filter_audio()

    for root in (empty_root, mismatched_root, even_root):
        assert list((root / "audio").rglob("*.mmap")) == []


def test_broadband_filter_audio_short_output_is_never_published(tmp_path, processing_settings, mocker):
    """
    Description
    -----------
    If the written memmap ends up with fewer bytes than ``n_samples x
    n_channels x 2`` (here every chunk is made one sample short), the step must
    raise an ``OSError`` naming both sizes and leave neither the memmap nor its
    hidden temporary nor a report in the output folder.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Per-test session root.
    processing_settings (dict)
        Package processing-settings fixture.
    mocker (pytest_mock.MockerFixture)
        No-ops the interactive ``smart_wait`` and shortens each streamed chunk.

    Returns
    -------
    None
    """

    mocker.patch("usv_playpen.processing.modify_files.smart_wait")
    _broadband_settings(processing_settings)
    signals = _write_broadband_session(tmp_path)

    def _one_sample_short(_stream, start, end):
        """
        Description
        -----------
        Stand-in for ``_BroadbandChannelStream.process`` returning one sample
        fewer than requested.

        Parameters
        ----------
        _stream (_BroadbandChannelStream)
            The stream instance (unused).
        start (int)
            First absolute sample of the chunk.
        end (int)
            One past the last absolute sample of the chunk.

        Returns
        -------
        column (np.ndarray)
            int16 zeros of length ``end - start - 1``.
        """

        return np.zeros(end - start - 1, dtype=np.int16)

    mocker.patch.object(_BroadbandChannelStream, "process", _one_sample_short)
    n_samples = signals[0].shape[0]
    n_chunks = -(-n_samples // round(2.0 * _BB_SR))
    expected_bytes = n_samples * 3 * 2
    with pytest.raises(OSError, match=f"Broadband memmap has {expected_bytes - n_chunks * 3 * 2} bytes, expected {expected_bytes}"):
        _make_operator(str(tmp_path), processing_settings, []).broadband_filter_audio()
    output_dir = tmp_path / "audio" / "broadband_filtered"
    assert sorted(path.name for path in output_dir.iterdir()) == []


def test_broadband_filter_audio_replaces_an_older_memmap_and_records_an_unknown_version(tmp_path, processing_settings, mocker):
    """
    Description
    -----------
    Exactly one broadband memmap may exist: an older one with another name
    (e.g. from a run on a different sample count) is deleted once the new one
    is in place. When the installed package metadata cannot be found, the
    report records ``code_version`` as ``unknown`` instead of failing.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Per-test session root.
    processing_settings (dict)
        Package processing-settings fixture.
    mocker (pytest_mock.MockerFixture)
        No-ops the interactive ``smart_wait`` and makes the version lookup fail.

    Returns
    -------
    None
    """

    mocker.patch("usv_playpen.processing.modify_files.smart_wait")
    mocker.patch("usv_playpen.processing.modify_files.metadata.version", side_effect=importlib_metadata.PackageNotFoundError("usv-playpen"))
    _broadband_settings(processing_settings)
    signals = _write_broadband_session(tmp_path)
    output_dir = tmp_path / "audio" / "broadband_filtered"
    output_dir.mkdir(parents=True)
    older = output_dir / "230101120000_concatenated_audio_broadband_filtered_250000_10_3_int16.mmap"
    older.write_bytes(np.zeros(30, dtype=np.int16).tobytes())

    summary = _make_operator(str(tmp_path), processing_settings, []).broadband_filter_audio()

    expected_name = f"230101120000_concatenated_audio_broadband_filtered_{_BB_SR}_{signals[0].shape[0]}_3_int16.mmap"
    assert summary['status'] == 'written'
    assert not older.exists()
    assert sorted(path.name for path in output_dir.glob("*.mmap")) == [expected_name]
    with (output_dir / BROADBAND_REPORT_NAME).open(encoding="utf-8") as report_file:
        report = json.load(report_file)
    assert report['code_version'] == 'unknown'
    assert report['output']['file'] == expected_name


def test_broadband_filter_session_worker_returns_written_and_failed_rows(tmp_path, processing_settings):
    """
    Description
    -----------
    The batch worker, run in-process: a good session yields a ``written`` row
    carrying the step's summary (3 channels, the exact sample count and byte
    size, an empty error) and logs the session's messages with its name; a
    session without wavs yields a ``failed`` row with empty numeric fields and
    the exception type and message in ``error``, and logs ``FAILED``. Both rows
    carry every report column.

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
    signals = _write_broadband_session(good)
    bad = tmp_path / "bad"
    (bad / "audio" / "hpss").mkdir(parents=True)
    log_path = tmp_path / "batch.log"

    good_row = modify_files._broadband_filter_session_worker(str(good), processing_settings, str(log_path))
    bad_row = modify_files._broadband_filter_session_worker(str(bad), processing_settings, str(log_path))

    n_samples = signals[0].shape[0]
    assert good_row['status'] == 'written'
    assert good_row['n_channels'] == 3
    assert good_row['n_samples'] == n_samples
    assert good_row['output_bytes'] == n_samples * 3 * 2
    assert good_row['error'] == ''
    assert bad_row['status'] == 'failed'
    assert bad_row['n_channels'] == ''
    assert bad_row['tones_kept'] == ''
    assert bad_row['error'].startswith("FileNotFoundError: No ")
    for row in (good_row, bad_row):
        assert set(row) == set(BROADBAND_BATCH_REPORT_COLUMNS)
    log_text = log_path.read_text(encoding="utf-8")
    assert "[good] Broadband filtering of" in log_text
    assert "[bad] FAILED: FileNotFoundError" in log_text
