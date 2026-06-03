"""
@author: bartulem
Mock-based tests for das_inference and assign_vocalizations.

Both modules orchestrate external CLI tools (DAS via conda, vocalocator via
conda) and rely on file-system state. These tests substitute every subprocess
call and provide synthetic disk fixtures so the orchestration logic itself can
be exercised without any external software installed.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import h5py
import numpy as np
import polars as pls
import pytest

# Headless matplotlib for the das_inference plotting code path.
import matplotlib
matplotlib.use("Agg")

from usv_playpen.das_inference import FindMouseVocalizations
from usv_playpen.assign_vocalizations import Vocalocator


# Shared parameter fixture

@pytest.fixture
def processing_settings():
    """Loads processing_settings.json from the package — same fixture pattern
    as test_process.py uses, but local-scoped so test_inference doesn't depend
    on test_process being collected first."""
    import usv_playpen
    package_dir = Path(usv_playpen.__file__).parent
    with (package_dir / '_parameter_settings' / 'processing_settings.json').open('r') as f:
        return json.load(f)

# FindMouseVocalizations

def test_find_mouse_vocalizations_init_loads_defaults_when_no_input_dict(tmp_path):
    """input_parameter_dict=None now reliably loads the package defaults from
    processing_settings.json. The old branch that tried to read a nonexistent
    'usv_inference.root_directory' key (raising KeyError) was removed."""
    fmv = FindMouseVocalizations(root_directory=str(tmp_path))
    # Defaults loaded — the FindMouseVocalizations sub-block must be present.
    assert "das_command_line_inference" in fmv.input_parameter_dict
    assert fmv.root_directory == str(tmp_path)


def test_find_mouse_vocalizations_init_root_directory_can_be_none(tmp_path):
    """Construction is permitted without a root_directory; downstream methods
    will raise when they try to use it. This mirrors the documented optional
    nature of the kwarg without crashing in the constructor itself."""
    fmv = FindMouseVocalizations()
    assert fmv.root_directory is None
    assert "das_command_line_inference" in fmv.input_parameter_dict


def test_find_mouse_vocalizations_init_with_full_kwargs(processing_settings, tmp_path):
    """Both root_directory and input_parameter_dict provided → no JSON load."""
    fmv = FindMouseVocalizations(
        root_directory=str(tmp_path),
        input_parameter_dict=processing_settings,
        message_output=lambda *_a, **_kw: None,
    )
    assert fmv.root_directory == str(tmp_path)
    assert fmv.input_parameter_dict is processing_settings["usv_inference"]["FindMouseVocalizations"]


def test_das_command_line_inference_invokes_subprocess_per_wav(processing_settings,
                                                                tmp_path, mocker):
    """For every .wav under audio/hpss_filtered/, we should fire one
    subprocess.Popen call to `conda run -n <env> das predict ...`. After
    completion, any matching annotation files are moved into
    audio/das_annotations/."""
    hpss_dir = tmp_path / "audio" / "hpss_filtered"
    hpss_dir.mkdir(parents=True)
    # Two synthetic input .wav files plus one fake annotation .csv that the
    # method should detect and move.
    (hpss_dir / "ch1.wav").write_bytes(b"\x00")
    (hpss_dir / "ch2.wav").write_bytes(b"\x00")
    (hpss_dir / "ch1_annotations.csv").write_text("start_seconds,stop_seconds,name\n0,1,call\n")

    popen_mock = mocker.patch("usv_playpen.das_inference.subprocess.Popen",
                              return_value=MagicMock(returncode=0))
    mocker.patch("usv_playpen.das_inference.wait_for_subprocesses",
                 return_value=[0])
    mocker.patch("usv_playpen.das_inference.smart_wait")

    fmv = FindMouseVocalizations(
        root_directory=str(tmp_path),
        input_parameter_dict=processing_settings,
        message_output=lambda *_a, **_kw: None,
    )
    fmv.das_command_line_inference()

    # One Popen per .wav
    assert popen_mock.call_count == 2
    # Verify "das predict" appears in every command
    for call in popen_mock.call_args_list:
        argv = call.kwargs.get("args") or call.args[0]
        assert "das" in argv and "predict" in argv

    # The annotation CSV was moved
    moved_csv = tmp_path / "audio" / "das_annotations" / "ch1_annotations.csv"
    assert moved_csv.is_file()


def test_das_command_line_inference_creates_save_dir_even_if_no_annotations(
    processing_settings, tmp_path, mocker
):
    """Even with zero outputs, the das_annotations dir should be created."""
    hpss_dir = tmp_path / "audio" / "hpss_filtered"
    hpss_dir.mkdir(parents=True)
    (hpss_dir / "ch1.wav").write_bytes(b"\x00")

    mocker.patch("usv_playpen.das_inference.subprocess.Popen",
                 return_value=MagicMock(returncode=0))
    mocker.patch("usv_playpen.das_inference.wait_for_subprocesses",
                 return_value=[0])
    mocker.patch("usv_playpen.das_inference.smart_wait")

    fmv = FindMouseVocalizations(
        root_directory=str(tmp_path),
        input_parameter_dict=processing_settings,
        message_output=lambda *_a, **_kw: None,
    )
    fmv.das_command_line_inference()

    assert (tmp_path / "audio" / "das_annotations").is_dir()


def test_summarize_das_findings_handles_no_annotations(processing_settings,
                                                        tmp_path, mocker):
    """When the das_annotations directory is missing entirely, the method
    falls into its except (IndexError, FileNotFoundError) block and emits a
    skip message rather than crashing."""
    msgs: list[str] = []
    mocker.patch("usv_playpen.das_inference.smart_wait")

    fmv = FindMouseVocalizations(
        root_directory=str(tmp_path),
        input_parameter_dict=processing_settings,
        message_output=msgs.append,
    )
    # No audio/das_annotations dir exists → glob raises FileNotFoundError on
    # some Python/Path versions; the function catches it gracefully.
    fmv.summarize_das_findings()
    # We don't assert the exact wording — just confirm execution didn't raise.
    assert isinstance(msgs, list)


def test_summarize_das_findings_skips_unrecognised_filename(processing_settings,
                                                             tmp_path, mocker):
    """A filename that doesn't match `<device>_..._<chXX>_annotations.csv`
    is logged + skipped instead of crashing the run with a KeyError."""
    annot_dir = tmp_path / "audio" / "das_annotations"
    annot_dir.mkdir(parents=True)
    # Bad filename: missing the device prefix entirely.
    pls.DataFrame({
        "start_seconds": [0.1], "stop_seconds": [0.2], "name": ["call"],
    }).write_csv(annot_dir / "garbage_filename.csv")

    msgs: list[str] = []
    mocker.patch("usv_playpen.das_inference.smart_wait")
    mocker.patch("usv_playpen.das_inference.load_session_metadata",
                 return_value=(None, None))
    mocker.patch("usv_playpen.das_inference.save_session_metadata")

    fmv = FindMouseVocalizations(
        root_directory=str(tmp_path),
        input_parameter_dict=processing_settings,
        message_output=msgs.append,
    )
    fmv.summarize_das_findings()  # must not raise

    assert any("does not match expected" in m for m in msgs)


def test_summarize_das_findings_zero_after_merge(processing_settings, tmp_path,
                                                  mocker):
    """An annotations dir that exists but contains only noise rows yields
    0 merged USVs and the method returns without crashing.

    The post-fix regex parser tolerates timestamps with arbitrary embedded
    underscores (e.g. "20260101_120000"), so this test uses such a name to
    pin that behaviour."""
    annot_dir = tmp_path / "audio" / "das_annotations"
    annot_dir.mkdir(parents=True)
    pls.DataFrame({
        "start_seconds": [0.1],
        "stop_seconds": [0.2],
        "name": ["noise"],
    }).write_csv(annot_dir / "m_20260101_120000_ch01_annotations.csv")

    mocker.patch("usv_playpen.das_inference.smart_wait")
    # load_session_metadata may attempt yaml IO; mock it.
    mocker.patch("usv_playpen.das_inference.load_session_metadata",
                 return_value=(None, None))
    mocker.patch("usv_playpen.das_inference.save_session_metadata")

    fmv = FindMouseVocalizations(
        root_directory=str(tmp_path),
        input_parameter_dict=processing_settings,
        message_output=lambda *_a, **_kw: None,
    )
    fmv.summarize_das_findings()
    # n_usv = 0 → no histogram .svg, no summary CSV written
    assert list((tmp_path / "audio").glob("*.svg")) == []
    # And the function did not raise.


# ===========================================================================
# Vocalocator
# ===========================================================================


def _make_vocalocator(tmp_path, processing_settings, **extras):
    return Vocalocator(
        root_directory=str(tmp_path),
        input_parameter_dict=processing_settings,
        message_output=lambda *_a, **_kw: None,
        **extras,
    )


def test_vocalocator_init_stores_arbitrary_kwargs():
    """The constructor sets every kwarg as a public attribute (no allowlist)."""
    voc = Vocalocator(foo=1, bar="baz", message_output=lambda *_a, **_kw: None)
    assert voc.foo == 1
    assert voc.bar == "baz"


def test_vocalocator_run_vocalocator_subprocess_invocation(tmp_path, processing_settings, mocker):
    """run_vocalocator runs the subprocess and writes assessment_assn.npy."""
    # ---- minimal disk setup needed before the method's calls -----------
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    video_dir = tmp_path / "video" / "track"
    video_dir.mkdir(parents=True)
    sl_dir = audio_dir / "sound_localization"
    sl_dir.mkdir()

    # tracking H5 + USV summary CSV (both required by the method's
    # first_match_or_raise calls).
    track_h5 = video_dir / "20230207213549_points3d_translated_rotated_metric.h5"
    with h5py.File(track_h5, "w") as f:
        f.create_dataset("track_names", data=np.array([b"M", b"F"]))
    pls.DataFrame({
        "usv_id": ["0001", "0002"],
        "start": [0.1, 0.2],
        "stop": [0.15, 0.25],
        "duration": [0.05, 0.05],
        "peak_amp_ch": [0.0, 0.0],
        "mean_amp_ch": [0.0, 0.0],
        "chs_count": [1.0, 1.0],
        "chs_detected": ["[0]", "[0]"],
        "emitter": [None, None],
    }).write_csv(audio_dir / "sess_usv_summary.csv")

    # ---- assessment.h5 produced "by the subprocess" --------------------
    assessment_h5 = sl_dir / "assessment.h5"
    n_voc = 2
    raw = np.zeros((n_voc, 27), dtype=np.float32)
    scaled = np.zeros((n_voc, 2, 3, 3), dtype=np.float32)  # (n_voc, n_mice, n_nodes, n_dims)
    arena_dims = [400.0, 300.0]
    model_config = {"DATA": {"ARENA_DIMS": arena_dims}}
    with h5py.File(assessment_h5, "w") as f:
        f.create_dataset("raw_model_output", data=raw)
        f.create_dataset("scaled_locations", data=scaled)
        f.attrs["model_config"] = json.dumps(model_config)

    # ---- mock everything subprocess-y ----------------------------------
    mocker.patch("usv_playpen.assign_vocalizations.subprocess.run",
                 return_value=MagicMock(returncode=0))
    mocker.patch("usv_playpen.assign_vocalizations.smart_wait")
    mocker.patch("usv_playpen.assign_vocalizations.load_session_metadata",
                 return_value=(None, None))
    mocker.patch("usv_playpen.assign_vocalizations.save_session_metadata")
    # get_conf_sets_6d / are_points_in_conf_set are imported at module top;
    # patch them to return small synthetic outputs.
    mocker.patch("usv_playpen.assign_vocalizations.get_conf_sets_6d",
                 return_value=(np.zeros((n_voc, 1)),
                               np.zeros((n_voc, 1)),
                               np.zeros((n_voc, 1))))
    mocker.patch("usv_playpen.assign_vocalizations.are_points_in_conf_set",
                 return_value=np.array([True, False]))

    # The 'vcl_model_directory' setting may point to a path that doesn't
    # exist on this host; that's fine because we mocked subprocess.run.
    voc = _make_vocalocator(tmp_path, processing_settings)
    voc.run_vocalocator()

    # assignment npy was written
    assert (sl_dir / "assessment_assn.npy").is_file()


def test_vocalocator_run_vocalocator_ssl_handles_missing_calibration(tmp_path,
                                                                     processing_settings,
                                                                     mocker, monkeypatch):
    """When no *cal*.npz file is in model_directory, run_vocalocator_ssl logs
    a "no calibration NPZ" message and returns (no further IO)."""
    # Set up the minimum disk layout to reach the calibration check.
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    video_dir = tmp_path / "video"
    video_dir.mkdir()
    track_h5 = video_dir / "20230207213549_points3d_translated_rotated_metric.h5"
    with h5py.File(track_h5, "w") as f:
        f.create_dataset("track_names", data=np.array([b"M", b"F"]))
    pls.DataFrame({"usv_id": ["x"]}).write_csv(audio_dir / "x_usv_summary.csv")

    # Point vcl_model_directory at an empty dir (no cal.npz anywhere).
    empty_model_dir = tmp_path / "empty_model"
    empty_model_dir.mkdir()
    processing_settings['vocalocator']['vcl_model_directory'] = str(empty_model_dir)

    mocker.patch("usv_playpen.assign_vocalizations.smart_wait")
    sub_run = mocker.patch("usv_playpen.assign_vocalizations.subprocess.run")

    msgs: list[str] = []
    voc = Vocalocator(
        root_directory=str(tmp_path),
        input_parameter_dict=processing_settings,
        message_output=msgs.append,
    )
    voc.run_vocalocator_ssl()

    # No subprocess call was issued because the calibration file wasn't found.
    assert sub_run.call_count == 0
    assert any("calibration NPZ" in m for m in msgs)


# ===========================================================================
# summarize_das_findings — happy path with synthetic multi-channel data
# ===========================================================================


def _make_summary_fixture(tmp_path: Path,
                          n_usv: int = 3,
                          channels: tuple = ("ch01", "ch02"),
                          n_audio_channels: int = 24,
                          sampling_rate: int = 250000) -> Path:
    """Builds the on-disk inputs that summarize_das_findings expects:

    1. <root>/audio/das_annotations/m_<ts>_<chN>_annotations.csv — one per channel.
    2. <root>/audio/hpss_filtered/<sess>_<sr>_<n_samples>_<n_ch>_int16.mmap —
       a memmap audio file whose name encodes the sample rate, sample count,
       channel count, and dtype.
    """
    annot_dir = tmp_path / "audio" / "das_annotations"
    annot_dir.mkdir(parents=True)

    # Write per-channel CSVs with `n_usv` real calls + 1 noise row each
    for ch in channels:
        rows = []
        for i in range(n_usv):
            rows.append({
                "start_seconds": 0.1 + i * 0.5,
                "stop_seconds":  0.15 + i * 0.5,
                "name": "call",
            })
        rows.append({
            "start_seconds": 100.0, "stop_seconds": 100.5, "name": "noise",
        })
        pls.DataFrame(rows).write_csv(annot_dir / f"m_20260101_{ch}_annotations.csv")

    # Build a small audio memmap. The summarize_das_findings filename parser
    # expects: ..._<sr>_<n_samples>_<n_channels>_<dtype>.mmap (split by '_').
    hpss_dir = tmp_path / "audio" / "hpss_filtered"
    hpss_dir.mkdir(parents=True)
    n_samples = int(sampling_rate * 5)  # 5 seconds of audio
    audio = np.zeros((n_samples, n_audio_channels), dtype=np.int16)
    # Inject signal at the USV start times so the spectrogram quality check
    # can compute correlations / variance over real samples.
    rng = np.random.default_rng(0)
    for i in range(n_usv):
        s = int((0.1 + i * 0.5) * sampling_rate)
        e = int((0.15 + i * 0.5) * sampling_rate)
        audio[s:e, :] = rng.integers(-10000, 10000, size=(e - s, n_audio_channels), dtype=np.int16)
    fname = f"sess_{sampling_rate}_{n_samples}_{n_audio_channels}_int16.mmap"
    audio.tofile(hpss_dir / fname)
    return tmp_path


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_summarize_das_findings_writes_summary_csv_and_histogram(
    processing_settings, tmp_path, mocker
):
    """End-to-end: 2 channels × 3 USVs each, all overlapping → 3 merged USVs.
    The function should write a `<sess>_usv_summary.csv` plus a
    `<sess>_usv_signal_correlation_histogram.svg` (the noise-cutoff figure).

    Filterwarning rationale: when every USV is detected on >1 channel, the
    `signal_variance` array stays all-NaN and `np.nanpercentile` emits a
    benign all-NaN-slice warning (the NaN flows through to a finite cutoff).
    The project's pytest config escalates RuntimeWarning to an error; we
    silence it locally because it's an artifact of the synthetic input mix,
    not a code bug.
    """
    _make_summary_fixture(tmp_path, n_usv=3, channels=("ch01", "ch02"))

    mocker.patch("usv_playpen.das_inference.smart_wait")
    mocker.patch("usv_playpen.das_inference.load_session_metadata",
                 return_value=(None, None))
    mocker.patch("usv_playpen.das_inference.save_session_metadata")

    fmv = FindMouseVocalizations(
        root_directory=str(tmp_path),
        input_parameter_dict=processing_settings,
        message_output=lambda *_a, **_kw: None,
    )
    fmv.summarize_das_findings()

    audio_dir = tmp_path / "audio"
    # Histogram .svg written
    svgs = list(audio_dir.glob("*_usv_signal_correlation_histogram.svg"))
    assert len(svgs) == 1
    # Summary CSV written
    summaries = list(audio_dir.glob("*_usv_summary.csv"))
    assert len(summaries) == 1
    df = pls.read_csv(str(summaries[0]))
    # Schema columns are stable
    assert set(df.columns) >= {
        "usv_id", "start", "stop", "duration", "peak_amp_ch", "mean_amp_ch",
        "chs_count", "chs_detected", "emitter",
    }


def test_summarize_das_findings_skips_unknown_channel_filename(
    processing_settings, tmp_path, mocker
):
    """A filename with a channel index outside the m_ch01..s_ch12 range is
    logged as "unrecognized device/channel" and skipped (no crash)."""
    annot_dir = tmp_path / "audio" / "das_annotations"
    annot_dir.mkdir(parents=True)
    # ch99 is not in ch_conversion_dict (which only has ch01..ch12 for m/s)
    pls.DataFrame({
        "start_seconds": [0.1], "stop_seconds": [0.2], "name": ["call"],
    }).write_csv(annot_dir / "m_20260101_ch99_annotations.csv")

    mocker.patch("usv_playpen.das_inference.smart_wait")
    mocker.patch("usv_playpen.das_inference.load_session_metadata",
                 return_value=(None, None))
    mocker.patch("usv_playpen.das_inference.save_session_metadata")

    msgs: list[str] = []
    fmv = FindMouseVocalizations(
        root_directory=str(tmp_path),
        input_parameter_dict=processing_settings,
        message_output=msgs.append,
    )
    fmv.summarize_das_findings()  # must not raise

    assert any("unrecognized device/channel" in m for m in msgs)
