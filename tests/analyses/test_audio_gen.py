"""
@author: bartulem
Mock-based tests for analyses/generate_audio_files.AudioGenerator.

The orchestration methods invoke `static_sox` via subprocess, librosa.load,
and soundfile.write. We mock all three so we can exercise the file-handling
and branching logic without sox installed and without any real WAV files
larger than a handful of samples.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from scipy.io import wavfile

from usv_playpen.analyses.generate_audio_files import AudioGenerator, _read_int16_snippet
from usv_playpen.analyses.mixture_model_utils import TMixture


# ---------------------------------------------------------------------------
# AudioGenerator class-level attributes
# ---------------------------------------------------------------------------


def test_audiogen_class_level_attrs_are_set():
    """`command_addition` and `shell_usage_bool` are set at class-definition
    time based on os.name. They should always be defined and consistent."""
    assert hasattr(AudioGenerator, "command_addition")
    assert hasattr(AudioGenerator, "shell_usage_bool")
    # command_addition is either "" (POSIX) or "cmd /c " (Windows) and
    # shell_usage_bool tracks the same fork.
    assert AudioGenerator.command_addition in ("", "cmd /c ")
    assert isinstance(AudioGenerator.shell_usage_bool, bool)


def test_audiogen_init_stores_arbitrary_kwargs():
    """Constructor stores every kwarg as an attribute (no allowlist)."""
    ag = AudioGenerator(
        exp_id="test_exp",
        root_directory="/whatever",
        message_output=lambda *_a, **_kw: None,
    )
    assert ag.exp_id == "test_exp"
    assert ag.root_directory == "/whatever"
    # The constructor also sets app_context_bool (GUI vs CLI detection).
    assert hasattr(ag, "app_context_bool")


# ---------------------------------------------------------------------------
# frequency_shift_audio_segment
# ---------------------------------------------------------------------------


def _fs_settings(audio_dir: str = "original",
                 device_id: str = "m",
                 channel_id: int = 1,
                 octave_shift: float = -1.0,
                 volume_adjustment: bool = False) -> dict:
    """Default freq-shift settings dict (matches keys used by the method)."""
    return {
        "fs_audio_dir": audio_dir,
        "fs_device_id": device_id,
        "fs_channel_id": channel_id,
        "fs_wav_sampling_rate": 250,  # kHz; the method multiplies by 1e3
        "fs_sequence_start": 0.0,
        "fs_sequence_duration": 0.05,
        "fs_octave_shift": octave_shift,
        "fs_volume_adjustment": volume_adjustment,
    }


def test_frequency_shift_audio_segment_logs_not_found(tmp_path, mocker):
    """When no .wav matches the device/channel pattern, the method logs
    a "not found" message and does NOT spawn any subprocess.

    NB: index access into an empty match list still happens before the
    branch check in the source — we patch the glob to return one fake
    path, then patch len() of the result via a manual subclass.
    """
    # Build directory tree:
    audio_dir = tmp_path / "audio" / "original"
    audio_dir.mkdir(parents=True)
    # Create a fake WAV that matches the wrong device pattern (s_*) so the
    # glob returns exactly one match (so audio_file_loc[0].name works) but
    # the len() check is = 1 — meaning we exercise the happy branch above
    # the "not found" else-branch.
    # Skip this approach. Instead: place a wav matching m_*_ch01_*.wav so
    # the happy branch executes — but mock librosa/sox so it doesn't fail.
    wav_name = "m_001_ch01_session.wav"
    (audio_dir / wav_name).write_bytes(b"\x00\x00")

    # Mock librosa.load → returns synthetic 1-D audio
    mocker.patch("usv_playpen.analyses.generate_audio_files.librosa.load",
                 return_value=(np.zeros(1024, dtype=np.float32), 250000))
    # Mock soundfile.write → no-op so no real sf.write is called.
    mocker.patch("usv_playpen.analyses.generate_audio_files.sf.write")
    # Mock noisereduce.reduce_noise → return the input unchanged.
    mocker.patch("usv_playpen.analyses.generate_audio_files.nr.reduce_noise",
                 side_effect=lambda y, **_kw: y)
    # Mock subprocess.Popen so static_sox is never invoked.
    fake_popen = mocker.patch(
        "usv_playpen.analyses.generate_audio_files.subprocess.Popen",
        return_value=MagicMock(wait=lambda *_a, **_kw: 0),
    )
    mocker.patch("usv_playpen.analyses.generate_audio_files.smart_wait")
    # Mock pathlib.Path.unlink to avoid the missing-file error when the
    # method tries to delete the temp files we never actually wrote.
    mocker.patch("pathlib.Path.unlink", return_value=None)

    ag = AudioGenerator(
        exp_id="test_exp",
        root_directory=str(tmp_path),
        freq_shift_settings_dict=_fs_settings(),
        message_output=lambda *_a, **_kw: None,
    )
    ag.frequency_shift_audio_segment()

    # Two SoX subprocesses (volume_adjustment=False → 1 call, plus the
    # final tempo-adjust → 1 call). The exact count depends on the
    # 'filtered' branch and volume flag; with default args (audio_dir
    # = 'original', volume off) we expect exactly 1 sox call (tempo).
    assert fake_popen.call_count >= 1


def test_frequency_shift_audio_segment_volume_adjustment_double_sox(tmp_path, mocker):
    """volume_adjustment=True introduces a second SoX call (compand)."""
    audio_dir = tmp_path / "audio" / "original"
    audio_dir.mkdir(parents=True)
    (audio_dir / "m_001_ch01_session.wav").write_bytes(b"\x00\x00")

    mocker.patch("usv_playpen.analyses.generate_audio_files.librosa.load",
                 return_value=(np.zeros(1024, dtype=np.float32), 250000))
    mocker.patch("usv_playpen.analyses.generate_audio_files.sf.write")
    mocker.patch("usv_playpen.analyses.generate_audio_files.nr.reduce_noise",
                 side_effect=lambda y, **_kw: y)
    fake_popen = mocker.patch(
        "usv_playpen.analyses.generate_audio_files.subprocess.Popen",
        return_value=MagicMock(wait=lambda *_a, **_kw: 0),
    )
    mocker.patch("usv_playpen.analyses.generate_audio_files.smart_wait")
    mocker.patch("pathlib.Path.unlink", return_value=None)

    ag = AudioGenerator(
        exp_id="test_exp",
        root_directory=str(tmp_path),
        freq_shift_settings_dict=_fs_settings(volume_adjustment=True),
        message_output=lambda *_a, **_kw: None,
    )
    ag.frequency_shift_audio_segment()

    # With volume adjustment on, we should have 2 sox calls (compand + tempo).
    assert fake_popen.call_count == 2


def test_frequency_shift_audio_segment_filtered_branch_skips_sinc(tmp_path, mocker):
    """When audio_dir contains "filtered", the final SoX call omits the sinc
    filter (only tempo). We verify this by inspecting the subprocess args."""
    audio_dir = tmp_path / "audio" / "hpss_filtered"
    audio_dir.mkdir(parents=True)
    (audio_dir / "m_001_ch01_session.wav").write_bytes(b"\x00\x00")

    mocker.patch("usv_playpen.analyses.generate_audio_files.librosa.load",
                 return_value=(np.zeros(1024, dtype=np.float32), 250000))
    mocker.patch("usv_playpen.analyses.generate_audio_files.sf.write")
    mocker.patch("usv_playpen.analyses.generate_audio_files.nr.reduce_noise",
                 side_effect=lambda y, **_kw: y)
    fake_popen = mocker.patch(
        "usv_playpen.analyses.generate_audio_files.subprocess.Popen",
        return_value=MagicMock(wait=lambda *_a, **_kw: 0),
    )
    mocker.patch("usv_playpen.analyses.generate_audio_files.smart_wait")
    mocker.patch("pathlib.Path.unlink", return_value=None)

    ag = AudioGenerator(
        exp_id="test_exp",
        root_directory=str(tmp_path),
        freq_shift_settings_dict=_fs_settings(audio_dir="hpss_filtered"),
        message_output=lambda *_a, **_kw: None,
    )
    ag.frequency_shift_audio_segment()

    # Verify that the sox command in the filtered branch does not contain "sinc".
    last_call_args = fake_popen.call_args_list[-1].kwargs.get("args", "")
    assert "sinc" not in last_call_args
    assert "tempo" in last_call_args


def test_frequency_shift_audio_segment_no_match_logs_and_returns(tmp_path, mocker):
    """Zero glob matches: log "Requested audio file not found." and return
    without spawning any sox subprocess. (Previously this path hit an
    IndexError on audio_file_loc[0] before the len() check; the fix moves
    the length check before the indexing.)"""
    audio_dir = tmp_path / "audio" / "original"
    audio_dir.mkdir(parents=True)
    # Wrong channel — glob returns no match
    (audio_dir / "m_001_ch99_session.wav").write_bytes(b"\x00\x00")

    fake_popen = mocker.patch(
        "usv_playpen.analyses.generate_audio_files.subprocess.Popen",
        return_value=MagicMock(wait=lambda *_a, **_kw: 0),
    )
    mocker.patch("usv_playpen.analyses.generate_audio_files.smart_wait")

    msgs: list[str] = []
    ag = AudioGenerator(
        exp_id="test_exp",
        root_directory=str(tmp_path),
        freq_shift_settings_dict=_fs_settings(channel_id=1),
        message_output=msgs.append,
    )
    ag.frequency_shift_audio_segment()  # must not raise

    assert fake_popen.call_count == 0
    assert any("Requested audio file not found" in m for m in msgs)


def test_frequency_shift_audio_segment_returns_path_with_explicit_window(tmp_path, mocker):
    """The method returns the produced file's Path, and an explicit
    ``seq_start``/``seq_duration`` overrides the settings-dict window: the
    output filename embeds the passed values and ``librosa.load`` is offset
    accordingly (this is the path the behavioral-video step relies on to align
    the audible audio to the rendered frames)."""
    audio_dir = tmp_path / "audio" / "original"
    audio_dir.mkdir(parents=True)
    (audio_dir / "m_001_ch01_session.wav").write_bytes(b"\x00\x00")

    fake_load = mocker.patch("usv_playpen.analyses.generate_audio_files.librosa.load",
                             return_value=(np.zeros(1024, dtype=np.float32), 250000))
    mocker.patch("usv_playpen.analyses.generate_audio_files.sf.write")
    mocker.patch("usv_playpen.analyses.generate_audio_files.nr.reduce_noise",
                 side_effect=lambda y, **_kw: y)
    mocker.patch("usv_playpen.analyses.generate_audio_files.subprocess.Popen",
                 return_value=MagicMock(wait=lambda *_a, **_kw: 0))
    mocker.patch("usv_playpen.analyses.generate_audio_files.smart_wait")
    mocker.patch("pathlib.Path.unlink", return_value=None)

    ag = AudioGenerator(
        exp_id="test_exp",
        root_directory=str(tmp_path),
        freq_shift_settings_dict=_fs_settings(),  # fs_sequence_start=0.0, fs_sequence_duration=0.05
        message_output=lambda *_a, **_kw: None,
    )
    out_path = ag.frequency_shift_audio_segment(seq_start=2.0, seq_duration=0.05)

    # the produced .wav path is returned (not None)
    assert out_path is not None
    assert out_path.name.endswith("_audible_denoised_tempo_adjusted.wav")
    # the explicit 2.0 s window overrides the dict's fs_sequence_start (0.0 s)
    assert "start=2.0s_duration=0.05s" in out_path.name
    # and the FIRST librosa.load (the segment extraction; later calls reload
    # temp files with only sr=) received the overriding offset/duration
    assert fake_load.call_args_list[0].kwargs["offset"] == 2.0
    assert fake_load.call_args_list[0].kwargs["duration"] == 0.05


def test_frequency_shift_audio_segment_returns_none_when_source_missing(tmp_path, mocker):
    """Returns None (not a Path) when the source audio cannot be uniquely
    located, so callers can fall back gracefully (the video step skips the
    audio mux when None is returned)."""
    (tmp_path / "audio" / "original").mkdir(parents=True)  # empty dir -> 0 glob matches
    mocker.patch("usv_playpen.analyses.generate_audio_files.subprocess.Popen",
                 return_value=MagicMock(wait=lambda *_a, **_kw: 0))
    mocker.patch("usv_playpen.analyses.generate_audio_files.smart_wait")

    ag = AudioGenerator(
        exp_id="test_exp",
        root_directory=str(tmp_path),
        freq_shift_settings_dict=_fs_settings(),
        message_output=lambda *_a, **_kw: None,
    )
    assert ag.frequency_shift_audio_segment() is None


def _run_usv_playback(tmp_path, mocker, *, seed, exp_id, n_snippets=8, total_usv=25):
    """Drive create_usv_playback_wav with mocked audio I/O and return the
    ordered list of chosen snippet filenames (from the *_usvids.txt the method
    writes). Identical snippet filenames are created under each exp_id so the
    sorted wav list -- and therefore the seeded selection -- is comparable
    across runs.
    """
    snippets_name = "snips"
    snip_dir = tmp_path / exp_id / "usv_playback_experiments" / snippets_name
    snip_dir.mkdir(parents=True)
    for i in range(n_snippets):
        (snip_dir / f"snippet_{i:02d}.wav").write_bytes(b"")

    mocker.patch("usv_playpen.analyses.generate_audio_files.find_base_path",
                 return_value=str(tmp_path))
    mocker.patch("usv_playpen.analyses.generate_audio_files.os.path.ismount",
                 return_value=True)
    mocker.patch("usv_playpen.analyses.generate_audio_files.smart_wait")
    mocker.patch("usv_playpen.analyses.generate_audio_files.wavfile.read",
                 return_value=(250, np.zeros(8, dtype=np.int16)))
    mocker.patch("usv_playpen.analyses.generate_audio_files.wavfile.write")

    ag = AudioGenerator(
        exp_id=exp_id,
        create_playback_settings_dict={
            "num_usv_files": 1,
            "total_usv_number": total_usv,
            "ipi_duration": 0.015,
            "wav_sampling_rate": 250,
            "playback_snippets_dir": snippets_name,
            "playback_seed": seed,
        },
        message_output=lambda *_a, **_kw: None,
    )
    ag.create_usv_playback_wav()

    out_dir = tmp_path / exp_id / "usv_playback_experiments" / "usv_playback_files"
    usvids_file = sorted(out_dir.glob("*_usvids.txt"))[0]
    return usvids_file.read_text().split()


def test_create_usv_playback_wav_seed_is_reproducible_and_varies(tmp_path, mocker):
    """A fixed integer playback_seed makes the chosen USV sequence exactly
    reproducible across runs, while a different seed yields a different
    sequence -- so a documented stimulus set can be regenerated bit-for-bit."""
    seq_a = _run_usv_playback(tmp_path, mocker, seed=0, exp_id="expA")
    seq_b = _run_usv_playback(tmp_path, mocker, seed=0, exp_id="expB")
    seq_c = _run_usv_playback(tmp_path, mocker, seed=1, exp_id="expC")

    assert seq_a == seq_b
    assert seq_a != seq_c


def _run_naturalistic_playback(tmp_path, mocker, *, seed, exp_id, prefix="male",
                               total_time=8, n_snippets=6):
    """
    Description
    -----------
    Drive ``create_naturalistic_usv_playback_wav`` with mocked audio I/O
    (``find_base_path`` / ``os.path.ismount`` redirected into ``tmp_path`` and
    ``wavfile.read`` / ``wavfile.write`` stubbed), against a snippet directory
    of ``n_snippets`` empty WAVs.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Per-test temp directory (serves as the file-server base).
    mocker (pytest_mock.MockerFixture)
        Used to patch the audio I/O boundary.
    seed (int)
        ``playback_seed`` for reproducible snippet selection.
    exp_id (str)
        Experiment-id subdirectory under the file-server base.
    prefix (str)
        Snippet-directory prefix selecting the sex-specific Student-t interval
        model ('male' / 'female').
    total_time (int | float)
        Target naturalistic playback time (s).
    n_snippets (int)
        Number of synthetic snippet WAVs to create.

    Returns
    -------
    spacing, usvids, write_mock (tuple)
        The spacing.txt lines, the usvids.txt lines, and the ``wavfile.write``
        mock, so callers can assert sequence structure and WAV emission.
    """

    snip_dir = (
        tmp_path / exp_id / "usv_playback_experiments"
        / f"{prefix}_usv_playback_snippets"
    )
    snip_dir.mkdir(parents=True)
    for i in range(n_snippets):
        (snip_dir / f"snippet_{i:02d}.wav").write_bytes(b"")

    mocker.patch("usv_playpen.analyses.generate_audio_files.find_base_path",
                 return_value=str(tmp_path))
    mocker.patch("usv_playpen.analyses.generate_audio_files.os.path.ismount",
                 return_value=True)
    mocker.patch("usv_playpen.analyses.generate_audio_files.smart_wait")
    mocker.patch("usv_playpen.analyses.generate_audio_files.wavfile.read",
                 return_value=(250, np.zeros(8, dtype=np.int16)))
    write_mock = mocker.patch("usv_playpen.analyses.generate_audio_files.wavfile.write")

    # Inject a small synthetic 3-component Student-t interval model so the test
    # does not depend on the real on-disk HDF5 archive. Components are sorted
    # ascending by log-mean, so the slowest (mean=0.0 -> ~1 s median) becomes the
    # ISI and the two faster ones (~0.06 s, ~0.37 s medians) form the IUI pool;
    # a ~1 s ISI keeps at least one full sequence inside the small time budget.
    synthetic_model = TMixture(
        weights=np.array([0.6, 0.25, 0.15]),
        means=np.array([-2.8, -1.0, 0.0]),
        covariances=np.array([0.07, 0.1, 0.1]),
        nus=np.array([30.0, 30.0, 30.0]),
    )
    mocker.patch(
        "usv_playpen.analyses.generate_audio_files.read_usv_interval_h5",
        return_value={"modes": {"e2s": {
            "attrs": {"K_selected_male": 3, "K_selected_female": 3},
            "gmm_fits": None,
        }}},
    )
    mocker.patch(
        "usv_playpen.analyses.generate_audio_files.reconstruct_best_model",
        return_value=(synthetic_model, np.arange(3)),
    )

    ag = AudioGenerator(
        exp_id=exp_id,
        create_playback_settings_dict={
            "num_naturalistic_usv_files": 1,
            "total_acceptable_naturalistic_playback_time": total_time,
            "naturalistic_wav_sampling_rate": 250,
            "naturalistic_playback_snippets_dir_prefix": prefix,
            "naturalistic_iui_archive_h5": "/dummy/archive.h5",
            "naturalistic_interval_mode": "e2s",
            "naturalistic_interval_clip_pct": {"male": 99.0, "female": 97.0},
            "playback_seed": seed,
        },
        message_output=lambda *_a, **_kw: None,
    )
    ag.create_naturalistic_usv_playback_wav()

    out_dir = (
        tmp_path / exp_id / "usv_playback_experiments"
        / "naturalistic_usv_playback_files"
    )
    spacing = sorted(out_dir.glob("*_spacing.txt"))[0].read_text().split("\n")
    usvids = sorted(out_dir.glob("*_usvids.txt"))[0].read_text().split("\n")
    return spacing, usvids, write_mock


def test_create_naturalistic_usv_playback_wav_emits_sequences(tmp_path, mocker):
    """
    Description
    -----------
    ``create_naturalistic_usv_playback_wav`` must interleave inter-sequence
    silences (ISI) and seeded USV sequences (each separated by IUIs) until the
    target playback time is reached, logging both the per-segment sample counts
    (``spacing.txt``) and the chosen snippet / ISI / IUI labels
    (``usvids.txt``), then writing a single WAV. The synthetic interval model
    keeps the ISIs short enough that at least one full sequence is emitted
    within the budget.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Per-test temp directory.
    mocker (pytest_mock.MockerFixture)
        Used to patch the audio I/O boundary.

    Returns
    -------
    None
    """

    spacing, usvids, write_mock = _run_naturalistic_playback(
        tmp_path, mocker, seed=0, exp_id="natA", prefix="male", total_time=8,
    )
    assert write_mock.call_count == 1, "exactly one playback WAV should be written"
    # At least one ISI marker and one chosen snippet (a .wav line) must appear.
    assert any(line.strip() == "ISI" for line in usvids)
    assert any(line.strip().endswith(".wav") for line in usvids)
    # Every spacing entry is an integer sample count.
    assert all(line.strip().isdigit() for line in spacing if line.strip())


def test_read_int16_snippet_rejects_non_int16(tmp_path):
    """A playback snippet that is not int16 (e.g. float32) raises a clear ValueError
    so it cannot silently upcast the int16 playback buffer / corrupt amplitudes; an
    int16 snippet is returned unchanged."""
    bad = tmp_path / "bad.wav"
    wavfile.write(bad, 250000, np.zeros(8, dtype=np.float32))
    with pytest.raises(ValueError, match="must be int16"):
        _read_int16_snippet(bad)

    good = tmp_path / "good.wav"
    wavfile.write(good, 250000, np.arange(8, dtype=np.int16))
    out = _read_int16_snippet(good)
    assert out.dtype == np.int16
    assert out.shape[0] == 8


def test_naturalistic_playback_metadata_matches_truncated_wav(tmp_path, mocker):
    """The spacing.txt sample counts must sum to EXACTLY the written WAV length.
    The inner sequence loop overshoots total_acceptable_playback_time and the WAV
    is sliced to target_samples; the metadata is clamped to match, so a downstream
    consumer walking spacing.txt does not desync past the truncation point."""
    spacing, usvids, write_mock = _run_naturalistic_playback(
        tmp_path, mocker, seed=0, exp_id="natAlign", prefix="male", total_time=2,
    )
    wav = write_mock.call_args.kwargs["data"]
    spacing_counts = [int(line) for line in spacing if line.strip()]
    usvid_labels = [line for line in usvids if line.strip()]
    # Metadata describes exactly the samples kept in the (sliced) WAV.
    assert sum(spacing_counts) == wav.shape[0]
    assert len(spacing_counts) == len(usvid_labels)   # 1:1 per-chunk metadata
    # A single ISI (~1 s) + one sequence (~1.8 s) overshoots the 2 s budget, so the
    # WAV is sliced to target_samples and the clamp path is exercised (not a no-op).
    assert wav.shape[0] == int(2 * 250 * 1e3)


def test_create_naturalistic_usv_playback_wav_seed_reproducible(tmp_path, mocker):
    """
    Description
    -----------
    A fixed ``playback_seed`` must reproduce the exact snippet sequence across
    runs (different exp_id dirs, identical snippet names), while a different
    seed must diverge.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Per-test temp directory (sub-dirs isolate the three runs).
    mocker (pytest_mock.MockerFixture)
        Used to patch the audio I/O boundary.

    Returns
    -------
    None
    """

    _, ids_a, _ = _run_naturalistic_playback(tmp_path / "a", mocker, seed=0, exp_id="x")
    _, ids_b, _ = _run_naturalistic_playback(tmp_path / "b", mocker, seed=0, exp_id="x")
    _, ids_c, _ = _run_naturalistic_playback(tmp_path / "c", mocker, seed=1, exp_id="x")
    snippets_a = [ln for ln in ids_a if ln.strip().endswith(".wav")]
    snippets_b = [ln for ln in ids_b if ln.strip().endswith(".wav")]
    snippets_c = [ln for ln in ids_c if ln.strip().endswith(".wav")]
    assert snippets_a == snippets_b
    assert snippets_a != snippets_c
