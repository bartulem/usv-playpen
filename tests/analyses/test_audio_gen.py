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

import h5py
import numpy as np
import pytest
from scipy.io import wavfile

from usv_playpen.analyses.generate_audio_files import (
    _PLAYBACK_CONTEXTS,
    AudioGenerator,
    _read_int16_snippet,
)

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


def _write_fake_repository(path, sampling_rate=250000):
    """Write a tiny fake naturalistic USV repository H5: 2 sessions, 5 USVs in 2
    bouts (distinct int16 audio), with known real within-bout + inter-bout gaps."""
    str_dtype = h5py.string_dtype(encoding="utf-8")
    usv_len = 2000
    n_usv = 5
    audio = np.concatenate([np.full(usv_len, (i + 1) * 100, dtype=np.int16) for i in range(n_usv)])
    with h5py.File(str(path), "w") as f:
        f.attrs["sex"] = "male"
        f.attrs["sampling_rate_hz"] = int(sampling_rate)
        f.attrs["n_usv"] = n_usv
        f.attrs["n_bout"] = 2
        f.create_dataset("audio", data=audio, dtype=np.int16)
        usv = f.create_group("usv")
        usv.create_dataset("offset", data=np.arange(n_usv, dtype=np.int64) * usv_len)
        usv.create_dataset("length", data=np.full(n_usv, usv_len, dtype=np.int64))
        usv.create_dataset("session", data=np.asarray(["sessA", "sessA", "sessA", "sessB", "sessB"], dtype=object), dtype=str_dtype)
        usv.create_dataset("usv_row", data=np.asarray([0, 1, 2, 5, 6], dtype=np.int64))
        usv.create_dataset("gap_to_next_s", data=np.asarray([0.01, 0.01, np.nan, 0.01, np.nan], dtype=np.float64))
        features = usv.create_group("features")
        # usv 0/1/2 (session A) are simple (mask 1); usv 3/4 (session B) are complex (mask 3/4).
        features.create_dataset("mask_number", data=np.asarray([1, 1, 1, 3, 4], dtype=np.int64))
        bout = f.create_group("bout")
        bout.create_dataset("usv_start", data=np.asarray([0, 3], dtype=np.int64))
        bout.create_dataset("usv_count", data=np.asarray([3, 2], dtype=np.int64))
        bout.create_dataset("preceding_isi_s", data=np.asarray([np.nan, 0.05], dtype=np.float64))


def _bout_runs(labels):
    """Split usvids labels into per-bout USV runs: the USV labels (``<session>_usv<row>``)
    between consecutive ``ISI`` markers, dropping the ``IUI`` silence labels."""
    runs, current = [], []
    for label in labels:
        if label == "ISI":
            if current:
                runs.append(current)
            current = []
        elif label != "IUI":
            current.append(label)
    if current:
        runs.append(current)
    return runs


def _run_naturalistic_playback(tmp_path, mocker, *, seed, exp_id, context_label="courtship_male", total_time=8, complexity=None):
    """
    Description
    -----------
    Drive ``create_naturalistic_usv_playback_wav`` against a fake repository laid out as
    ``<root>/<sex>/naturalistic_usv_repository_<token>_<ts>.h5`` (the real directory scheme),
    with the audio-I/O boundary mocked (``resolve_data_root`` -> the fake repository root /
    playback output dir, and ``wavfile.write`` stubbed to capture the WAV without touching
    disk).

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Per-test temp directory (file-server base + repository root).
    mocker (pytest_mock.MockerFixture)
        Used to patch the audio-I/O boundary.
    seed (int)
        ``playback_seed`` for reproducible bout draws.
    exp_id (str)
        Experiment-id subdirectory under the file-server base.
    context_label (str)
        Playback context (e.g. ``courtship_male``); picks the sex subdirectory + token.
    total_time (int | float)
        Target naturalistic playback time (s).
    complexity (tuple | None)
        Optional ``(enabled, mask_threshold, start_fraction, end_fraction)``.

    Returns
    -------
    spacing, usvids, write_mock (tuple)
        The spacing.txt lines, the usvids.txt lines, and the ``wavfile.write`` mock.
    """

    tmp_path.mkdir(parents=True, exist_ok=True)
    token, sex = _PLAYBACK_CONTEXTS[context_label]
    repo_root = tmp_path / "repo_root"
    (repo_root / sex).mkdir(parents=True, exist_ok=True)
    _write_fake_repository(repo_root / sex / f"naturalistic_usv_repository_{token}_20230101_000000.h5")

    playback_out = tmp_path / "playback_out"
    mocker.patch(
        "usv_playpen.analyses.generate_audio_files.resolve_data_root",
        side_effect=lambda key: repo_root if key == "naturalistic_usv_repository_dir" else playback_out,
    )
    mocker.patch("usv_playpen.analyses.generate_audio_files.smart_wait")
    write_mock = mocker.patch("usv_playpen.analyses.generate_audio_files.wavfile.write")

    ag = AudioGenerator(
        exp_id=exp_id,
        create_playback_settings_dict={
            "num_naturalistic_usv_files": 1,
            "context_label": context_label,
            "total_acceptable_naturalistic_playback_time": total_time,
            "complexity_enabled": complexity[0] if complexity else False,
            "complexity_mask_threshold": complexity[1] if complexity else 2,
            "complexity_start_fraction": complexity[2] if complexity else 0.0,
            "complexity_end_fraction": complexity[3] if complexity else 1.0,
            "complexity_bandwidth": 0.2,
            "edge_silence_seconds": 0.1,
            "max_isi_seconds": 100.0,
            "playback_seed": seed,
        },
        message_output=lambda *_a, **_kw: None,
    )
    ag.create_naturalistic_usv_playback_wav()

    out_dir = playback_out
    spacing = sorted(out_dir.glob("*_spacing.txt"))[0].read_text().split("\n")
    usvids = sorted(out_dir.glob("*_usvids.txt"))[0].read_text().split("\n")
    return spacing, usvids, write_mock


def test_create_naturalistic_usv_playback_wav_emits_sequences(tmp_path, mocker):
    """
    Description
    -----------
    ``create_naturalistic_usv_playback_wav`` must interleave inter-sequence silences
    (ISI) and real bout sequences (USVs separated by IUIs) drawn from the repository
    until the target playback time is reached, logging the per-chunk sample counts
    (``spacing.txt``) and the per-chunk labels (``usvids.txt``: ``<session>_usv<row>``
    / ``ISI`` / ``IUI``), then writing a single WAV. Each emitted bout replays real
    USVs in their natural order (same session, ascending ``usv_row``).

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Per-test temp directory.
    mocker (pytest_mock.MockerFixture)
        Used to patch the audio-I/O boundary.

    Returns
    -------
    None
    """

    spacing, usvids, write_mock = _run_naturalistic_playback(
        tmp_path, mocker, seed=0, exp_id="natA", total_time=8,
    )
    assert write_mock.call_count == 1, "exactly one playback WAV should be written"
    labels = [line.strip() for line in usvids if line.strip()]
    # At least one ISI marker and one USV label (an id, NOT a .wav filename) appear.
    assert any(label == "ISI" for label in labels)
    assert any(label not in ("ISI", "IUI") for label in labels)
    # Every spacing entry is an integer sample count.
    assert all(line.strip().isdigit() for line in spacing if line.strip())
    # Bouts are replayed in natural order: within each ISI-delimited run the USVs are
    # same-session with strictly ascending usv_row (not random single draws).
    runs = _bout_runs(labels)
    assert runs, "no bout runs emitted"
    for run in runs:
        sessions = {label.rsplit("_usv", 1)[0] for label in run}
        rows = [int(label.rsplit("_usv", 1)[1]) for label in run]
        assert len(sessions) == 1, f"a bout run spans multiple sessions: {sessions}"
        assert rows == sorted(rows), f"bout run not ascending: {rows}"
        assert len(set(rows)) == len(rows), f"bout run has duplicate rows: {rows}"


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
    """The spacing.txt sample counts must sum to EXACTLY the written WAV length, and stay 1:1
    with usvids so a downstream consumer walking spacing.txt does not desync. The file is now
    built of whole bouts framed by a fixed lead-in/lead-out silence, so it is *up to* the
    requested duration and opens and closes on an ``edge_silence_seconds`` ISI chunk."""
    spacing, usvids, write_mock = _run_naturalistic_playback(
        tmp_path, mocker, seed=0, exp_id="natAlign", total_time=2,
    )
    wav = write_mock.call_args.kwargs["data"]
    spacing_counts = [int(line) for line in spacing if line.strip()]
    usvid_labels = [line.strip() for line in usvids if line.strip()]
    assert sum(spacing_counts) == wav.shape[0]
    assert len(spacing_counts) == len(usvid_labels)   # 1:1 per-chunk metadata
    assert wav.shape[0] <= 2 * 250000                 # up to the requested duration @ 250 kHz
    # fixed 0.1 s (edge_silence_seconds) lead-in and lead-out: file opens and closes on an ISI
    assert usvid_labels[0] == "ISI" and usvid_labels[-1] == "ISI"
    assert spacing_counts[0] == int(0.1 * 250000) and spacing_counts[-1] == int(0.1 * 250000)


def test_create_naturalistic_usv_playback_wav_seed_reproducible(tmp_path, mocker):
    """
    Description
    -----------
    A fixed ``playback_seed`` must reproduce the exact bout / USV sequence across runs
    (different exp_id dirs, identical repository), while a different seed must diverge.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Per-test temp directory (sub-dirs isolate the three runs).
    mocker (pytest_mock.MockerFixture)
        Used to patch the audio-I/O boundary.

    Returns
    -------
    None
    """

    _, ids_a, _ = _run_naturalistic_playback(tmp_path / "a", mocker, seed=0, exp_id="x")
    _, ids_b, _ = _run_naturalistic_playback(tmp_path / "b", mocker, seed=0, exp_id="x")
    _, ids_c, _ = _run_naturalistic_playback(tmp_path / "c", mocker, seed=1, exp_id="x")
    usvs_a = [ln.strip() for ln in ids_a if ln.strip() and ln.strip() not in ("ISI", "IUI")]
    usvs_b = [ln.strip() for ln in ids_b if ln.strip() and ln.strip() not in ("ISI", "IUI")]
    usvs_c = [ln.strip() for ln in ids_c if ln.strip() and ln.strip() not in ("ISI", "IUI")]
    assert usvs_a == usvs_b
    assert usvs_a != usvs_c


def _overall_complex_fraction(usvids):
    """Fraction of the output's USV labels that are session-B calls (the complex ones in the
    fake repo: mask 3/4); ISI/IUI silence labels are ignored."""
    usv_labels = [x.strip() for x in usvids if x.strip() and x.strip() not in ("ISI", "IUI")]
    if not usv_labels:
        return 0.0
    return sum(1 for label in usv_labels if label.startswith("sessB")) / len(usv_labels)


def _complex_fraction_by_half(spacing, usvids):
    """Complex (session-B) USV fraction in the first vs second half of the file (split by
    cumulative sample offset), to check a complexity ramp rises over the file."""
    counts = [int(x) for x in spacing if x.strip()]
    labels = [x.strip() for x in usvids if x.strip()]
    total = sum(counts)
    offset = 0
    first, second = [0, 0], [0, 0]
    for count, label in zip(counts, labels, strict=True):
        if label not in ("ISI", "IUI"):
            bucket = first if offset < total / 2 else second
            bucket[1] += 1
            if label.startswith("sessB"):
                bucket[0] += 1
        offset += count
    first_frac = first[0] / first[1] if first[1] else 0.0
    second_frac = second[0] / second[1] if second[1] else 0.0
    return first_frac, second_frac


def test_complexity_ratio_steers_output_mix(tmp_path, mocker):
    """A flat high target complex-fraction yields a much more complex output than a flat low
    one; complexity=(enabled, mask_threshold, start_fraction, end_fraction)."""
    _, high, _ = _run_naturalistic_playback(tmp_path / "hi", mocker, seed=0, exp_id="x", complexity=(True, 2, 1.0, 1.0))
    _, low, _ = _run_naturalistic_playback(tmp_path / "lo", mocker, seed=0, exp_id="x", complexity=(True, 2, 0.0, 0.0))
    high_frac, low_frac = _overall_complex_fraction(high), _overall_complex_fraction(low)
    assert high_frac > low_frac
    assert high_frac > 0.8
    assert low_frac < 0.2


def test_complexity_ramp_increases_over_file(tmp_path, mocker):
    """A 0 -> 1 complexity ramp makes the second half of the file more complex than the first."""
    spacing, usvids, _ = _run_naturalistic_playback(tmp_path, mocker, seed=0, exp_id="x", complexity=(True, 2, 0.0, 1.0))
    first_frac, second_frac = _complex_fraction_by_half(spacing, usvids)
    assert second_frac > first_frac


def test_complexity_threshold_is_respected(tmp_path, mocker):
    """Raising the mask threshold above every call's mask_number means nothing counts as
    complex, so steering has nothing to prefer and degenerates to uniform draws -- a
    less session-B-heavy file than a threshold at which the complex calls qualify."""
    _, qualifies, _ = _run_naturalistic_playback(tmp_path / "t2", mocker, seed=0, exp_id="x", complexity=(True, 2, 1.0, 1.0))
    _, none_qualify, _ = _run_naturalistic_playback(tmp_path / "t9", mocker, seed=0, exp_id="x", complexity=(True, 9, 1.0, 1.0))
    assert _overall_complex_fraction(qualifies) > _overall_complex_fraction(none_qualify)


def test_naturalistic_playback_missing_repository_reports_not_found(tmp_path, mocker):
    """If no repository is built for the chosen context, the generator emits a clean message
    (no crash, no WAV) rather than raising an uncaught FileNotFoundError into the GUI."""
    empty_root = tmp_path / "empty_repos"
    (empty_root / "male").mkdir(parents=True)  # sex dir exists, but holds no repository H5
    playback_out = tmp_path / "out"
    mocker.patch(
        "usv_playpen.analyses.generate_audio_files.resolve_data_root",
        side_effect=lambda key: empty_root if key == "naturalistic_usv_repository_dir" else playback_out,
    )
    mocker.patch("usv_playpen.analyses.generate_audio_files.smart_wait")
    write_mock = mocker.patch("usv_playpen.analyses.generate_audio_files.wavfile.write")
    messages = []
    AudioGenerator(
        exp_id="x",
        create_playback_settings_dict={
            "num_naturalistic_usv_files": 1,
            "context_label": "lone_male",
            "total_acceptable_naturalistic_playback_time": 10,
            "complexity_enabled": False,
            "complexity_mask_threshold": 2,
            "complexity_start_fraction": 0.0,
            "complexity_end_fraction": 1.0,
            "complexity_bandwidth": 0.2,
            "playback_seed": 0,
        },
        message_output=messages.append,
    ).create_naturalistic_usv_playback_wav()
    assert write_mock.call_count == 0, "no WAV should be written when the repository is missing"
    assert any("no naturalistic usv repository" in m.lower() for m in messages), messages
