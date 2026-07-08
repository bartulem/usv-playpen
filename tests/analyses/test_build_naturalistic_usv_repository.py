"""
@author: bartulem
Tests for ``usv_playpen.analyses.build_naturalistic_usv_repository`` — the
naturalistic-USV-playback repository builder.

The builder reconstructs clean per-USV audio from a session's raw
``hpss_filtered`` mmap + stored SAM masks, segments the emitter's USVs into
natural bouts, and writes one timestamped H5 per (sex, social context) holding
the concatenated int16 audio plus the per-USV / per-bout structure a playback
function replays. These tests cover the pure helpers (emitter normalization,
bout segmentation), the true-phase masked-ISTFT reconstruction, and a full
``build()`` run against a fully synthetic session laid out on disk (no real
data): a raw noise mmap, a ``usv_summary.csv``, and a spectrogram H5 carrying
``durations`` + a ``mask/<session>`` group. Batch robustness (a bad session is
skipped, not fatal), the no-bouts short-circuit, and the CLI routing are checked
too.
"""

from __future__ import annotations

import json
import pathlib

import h5py
import numpy as np
import polars as pls
import pytest
from click.testing import CliRunner

from usv_playpen.analyses import build_naturalistic_usv_repository as bnr
from usv_playpen.analyses.build_naturalistic_usv_repository import (
    NaturalisticUsvRepositoryBuilder,
    _normalize_emitter,
    _segment_bouts,
    build_naturalistic_usv_repository_cli,
    reconstruct_usv_waveform,
)

_SPEC_PARAMS = json.loads(
    (pathlib.Path(bnr.__file__).resolve().parent.parent
     / "_parameter_settings" / "processing_settings.json").read_text()
)["generate_spectrograms"]


# ---------------------------------------------------------------------------
# _normalize_emitter
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value,expected", [
    (None, ""),
    ("mouse_1", "mouse_1"),
    ("mouse_1\x00\x00", "mouse_1"),   # NUL padding stripped
    ("  spaced  ", "spaced"),
    ("\x00 pad \x00", "pad"),
])
def test_normalize_emitter(value, expected):
    """Emitter ids are NUL- and whitespace-stripped; ``None`` maps to ''."""

    assert _normalize_emitter(value) == expected


# ---------------------------------------------------------------------------
# _segment_bouts
# ---------------------------------------------------------------------------


def test_segment_bouts_empty():
    """An empty USV set returns an empty (int) bout-index array."""

    out = _segment_bouts(np.empty(0), np.empty(0), ibi_threshold=1.0)
    assert out.shape == (0,)
    assert out.dtype.kind == "i"


def test_segment_bouts_groups_by_gap():
    """A new bout starts at the first USV and whenever the end-to-start gap
    reaches the IBI threshold; sub-threshold gaps stay in the same bout."""

    starts = np.array([0.0, 1.0, 2.0, 10.0])
    stops = np.array([0.5, 1.5, 2.5, 10.5])
    # gaps to previous stop: 0.5, 0.5, 7.5 -> only the last opens a new bout.
    out = _segment_bouts(starts, stops, ibi_threshold=1.0)
    np.testing.assert_array_equal(out, [0, 0, 0, 1])


def test_segment_bouts_threshold_is_inclusive():
    """A gap exactly equal to the threshold opens a new bout (``>=``)."""

    starts = np.array([0.0, 2.0])
    stops = np.array([0.0, 1.0])   # gap = 2.0 - 1.0 = 1.0 == threshold
    out = _segment_bouts(starts, stops, ibi_threshold=1.0)
    np.testing.assert_array_equal(out, [0, 1])


# ---------------------------------------------------------------------------
# reconstruct_usv_waveform
# ---------------------------------------------------------------------------


def _tight_mask(num_freq: int = 128, num_time: int = 128) -> np.ndarray:
    """A tight boolean mask covering a mid-frequency band across all time."""

    mask = np.zeros((num_freq, num_time), dtype=bool)
    mask[40:80, :] = True
    return mask


def test_reconstruct_returns_none_for_short_segment():
    """A segment shorter than one STFT window cannot be inverted -> None."""

    segment = np.zeros((_SPEC_PARAMS["nperseg"] - 1, 4), dtype=np.int16)
    out = reconstruct_usv_waveform(
        audio_segment_channels=segment, mask_2d=_tight_mask(), sampling_rate=250000,
        spec_params=_SPEC_PARAMS, mask_dilation=0, feather_sigma_time=0.8, fade_ms=4.0,
    )
    assert out is None


def test_reconstruct_shape_and_faded_ends():
    """A valid segment reconstructs to a finite float waveform the length of
    the input signal, whose raised-cosine fade drives the first/last samples to
    exactly zero (the fade ramp starts at 0)."""

    rng = np.random.default_rng(0)
    n_samples = 6000
    segment = rng.integers(-2000, 2000, size=(n_samples, 4)).astype(np.int16)
    out = reconstruct_usv_waveform(
        audio_segment_channels=segment, mask_2d=_tight_mask(), sampling_rate=250000,
        spec_params=_SPEC_PARAMS, mask_dilation=0, feather_sigma_time=0.8, fade_ms=4.0,
    )
    assert out is not None
    assert out.shape == (n_samples,)
    assert np.all(np.isfinite(out))
    assert out[0] == pytest.approx(0.0, abs=1e-9)
    assert out[-1] == pytest.approx(0.0, abs=1e-9)
    # The masked-in band carries energy, so the reconstruction is not all-zero.
    assert float(np.max(np.abs(out))) > 0.0


# ---------------------------------------------------------------------------
# build() — full synthetic-session integration
# ---------------------------------------------------------------------------


def _write_fake_session(
    session_root: pathlib.Path,
    *,
    session_id: str = "20230101_120000",
    starts: list[float],
    stops: list[float],
    sampling_rate: int = 250000,
    n_channels: int = 4,
    durations_bins: int = 64,
    emitter: str = "male",
) -> None:
    """
    Write a minimal on-disk session the builder can consume: a raw noise
    ``hpss_filtered`` mmap (its filename encodes sr / samples / channels /
    dtype), a ``usv_summary.csv`` (start / stop / emitter + one feature column),
    and a spectrogram H5 carrying per-row ``durations`` plus a ``mask/<session>``
    group with one segmentation per USV (so every row has a detected mask).
    """

    n_usv = len(starts)
    # Enough samples to cover the last USV segment end.
    n_samples = int(np.ceil((max(stops) + 0.05) * sampling_rate))

    audio_dir = session_root / "audio"
    hpss_dir = audio_dir / "hpss_filtered"
    spec_dir = audio_dir / "spectrograms"
    hpss_dir.mkdir(parents=True, exist_ok=True)
    spec_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(1)
    audio = rng.integers(-3000, 3000, size=(n_samples, n_channels)).astype(np.int16)
    mmap_name = f"{session_id}_audio_hpss_filtered_{sampling_rate}_{n_samples}_{n_channels}_int16.mmap"
    audio.tofile(hpss_dir / mmap_name)

    pls.DataFrame({
        "start": starts,
        "stop": stops,
        "emitter": [emitter] * n_usv,
        "duration_s": [float(b - a) for a, b in zip(starts, stops)],
    }).write_csv(str(audio_dir / f"{session_id}_usv_summary.csv"))

    num_freq = int(_SPEC_PARAMS["num_freq_bins"])
    num_time = int(_SPEC_PARAMS["num_time_bins"])
    segmentations = np.zeros((n_usv, num_freq, num_time), dtype=bool)
    segmentations[:, 40:80, :] = True
    with h5py.File(str(spec_dir / f"{session_id}_spectrograms.h5"), "w") as h5_file:
        spec_grp = h5_file.create_group(f"spectrogram/{session_id}")
        spec_grp.create_dataset("durations", data=np.full(n_usv, durations_bins, dtype=np.int64))
        mask_grp = h5_file.create_group(f"mask/{session_id}")
        mask_grp.create_dataset("segmentations", data=segmentations)
        mask_grp.create_dataset("spectrogram_index", data=np.arange(n_usv, dtype=np.int64))


def _build_cfg(context_label: str = "same_sex_male") -> dict:
    """The ``build_naturalistic_usv_repository`` settings block (shipped defaults)."""

    return {
        "session_lists": [],
        "context_label": context_label,
        "ibi_z_score": 2.58,
        "ibi_component_index": 0,
        "min_vocalizations": 2,
        "length_threshold": 128,
        "min_duration": 1,
        "mask_dilation": 0,
        "feather_sigma_time": 0.8,
        "fade_ms": 4.0,
        "peak_normalize": True,
        "peak_target_fraction": 0.85,
    }


@pytest.fixture
def _patched_env(mocker):
    """No-op ``smart_wait`` and identity ``resolve_experimenter_path`` so the
    build runs fast and does not depend on a host experimenter config."""

    mocker.patch.object(bnr, "smart_wait")
    mocker.patch.object(bnr, "resolve_experimenter_path", side_effect=lambda p: p)


def test_build_writes_repository_h5(tmp_path, _patched_env):
    """
    Description
    -----------
    A full ``build()`` over one synthetic session reconstructs the two complete
    bouts (3 + 2 USVs; the sub-threshold within-bout gaps keep them together, the
    large gap between them opens the second bout) and writes a single timestamped
    H5 under ``<repo>/male/`` whose root attrs, ``usv`` / ``bout`` groups,
    concatenated ``audio``, per-USV feature table, and provenance all match.
    """

    session_root = tmp_path / "session_root"
    # bout 1: starts 0.01/0.05/0.09 (within-bout gaps ~0.02 s < IBI ~0.12 s);
    # bout 2: starts 0.30/0.34 (0.19 s gap from bout 1 opens a new bout).
    _write_fake_session(
        session_root,
        starts=[0.01, 0.05, 0.09, 0.30, 0.34],
        stops=[0.03, 0.07, 0.11, 0.32, 0.36],
    )
    repo_out = tmp_path / "repo"
    builder = NaturalisticUsvRepositoryBuilder(
        root_directories=[str(session_root)],
        input_parameter_dict={
            "build_naturalistic_usv_repository": _build_cfg("same_sex_male"),
            "data_roots": {"naturalistic_usv_repository_dir": str(repo_out)},
        },
        message_output=lambda *_a, **_kw: None,
    )
    builder.build()

    written = list((repo_out / "male").glob("naturalistic_usv_repository_same_sex_*.h5"))
    assert len(written) == 1, "exactly one repository H5 should be written under the male dir"
    with h5py.File(str(written[0]), "r") as h5_file:
        assert h5_file.attrs["sex"] == "male"
        assert h5_file.attrs["social_context"] == "same_sex"
        assert h5_file.attrs["context_label"] == "same_sex_male"
        assert int(h5_file.attrs["n_usv"]) == 5
        assert int(h5_file.attrs["n_bout"]) == 2
        assert int(h5_file.attrs["sampling_rate_hz"]) == 250000
        # Audio is one concatenation indexed by per-USV offset/length.
        assert h5_file["audio"].dtype == np.int16
        offsets = h5_file["usv/offset"][:]
        lengths = h5_file["usv/length"][:]
        assert offsets.shape == (5,) and lengths.shape == (5,)
        assert int(offsets[0]) == 0
        assert int(offsets[-1] + lengths[-1]) == h5_file["audio"].shape[0]
        # Bout structure: two bouts of 3 then 2 USVs.
        np.testing.assert_array_equal(h5_file["bout/usv_count"][:], [3, 2])
        np.testing.assert_array_equal(h5_file["bout/usv_start"][:], [0, 3])
        # Per-USV feature table carries the usv_summary columns.
        assert "duration_s" in h5_file["usv/features"]
        assert h5_file["usv/features/duration_s"].shape == (5,)
        # Provenance records the resolved session root.
        roots = [r.decode() if isinstance(r, bytes) else r for r in h5_file["provenance/session_roots"][:]]
        assert str(session_root) in roots


def test_build_no_complete_bouts_writes_nothing(tmp_path, _patched_env):
    """When no bout meets ``min_vocalizations`` (every USV is isolated by a
    supra-threshold gap) nothing is written and the build still completes."""

    session_root = tmp_path / "session_root"
    # Every USV separated by > IBI threshold -> all singleton bouts -> dropped.
    _write_fake_session(session_root, starts=[0.01, 0.50, 1.00], stops=[0.03, 0.52, 1.02])
    repo_out = tmp_path / "repo"
    messages = []
    NaturalisticUsvRepositoryBuilder(
        root_directories=[str(session_root)],
        input_parameter_dict={
            "build_naturalistic_usv_repository": _build_cfg("same_sex_male"),
            "data_roots": {"naturalistic_usv_repository_dir": str(repo_out)},
        },
        message_output=lambda *m, **_k: messages.append(" ".join(str(x) for x in m)),
    ).build()

    assert not (repo_out / "male").exists() or not list((repo_out / "male").glob("*.h5"))
    assert any("no repository written" in m.lower() for m in messages)


def test_build_skips_unreadable_session(tmp_path, _patched_env):
    """A session missing its raw ``hpss_filtered`` mmap is skipped (logged), so a
    batch that also contains a good session still completes and writes it."""

    good_root = tmp_path / "good_session"
    _write_fake_session(good_root, starts=[0.01, 0.05, 0.09], stops=[0.03, 0.07, 0.11])
    bad_root = tmp_path / "bad_session"
    (bad_root / "audio").mkdir(parents=True, exist_ok=True)   # no mmap / csv / h5

    repo_out = tmp_path / "repo"
    messages = []
    NaturalisticUsvRepositoryBuilder(
        root_directories=[str(bad_root), str(good_root)],
        input_parameter_dict={
            "build_naturalistic_usv_repository": _build_cfg("same_sex_male"),
            "data_roots": {"naturalistic_usv_repository_dir": str(repo_out)},
        },
        message_output=lambda *m, **_k: messages.append(" ".join(str(x) for x in m)),
    ).build()

    assert any("skipping bad_session" in m.lower() for m in messages)
    assert len(list((repo_out / "male").glob("*.h5"))) == 1


def test_build_courtship_emitter_filter_keeps_target_sex(tmp_path, mocker, _patched_env):
    """
    Description
    -----------
    A courtship build (``emitter`` mode) reads the target sex's track id from the
    session metadata and keeps only USVs attributed to that emitter. With a mix of
    male / female emitters, only the male bout survives.
    """

    session_root = tmp_path / "session_root"
    # Three male USVs forming a bout + two female USVs (a different emitter).
    _write_fake_session(
        session_root,
        starts=[0.01, 0.05, 0.09, 0.30, 0.34],
        stops=[0.03, 0.07, 0.11, 0.32, 0.36],
        emitter="male_mouse",
    )
    # Overwrite the emitter column so the last two rows are the female's.
    summary_path = session_root / "audio" / "20230101_120000_usv_summary.csv"
    df = pls.read_csv(str(summary_path)).with_columns(
        pls.Series("emitter", ["male_mouse", "male_mouse", "male_mouse", "female_mouse", "female_mouse"])
    )
    df.write_csv(str(summary_path))

    mocker.patch.object(
        bnr, "extract_session_metadata",
        return_value={"male_id": "male_mouse", "female_id": "female_mouse"},
    )
    repo_out = tmp_path / "repo"
    NaturalisticUsvRepositoryBuilder(
        root_directories=[str(session_root)],
        input_parameter_dict={
            "build_naturalistic_usv_repository": _build_cfg("courtship_male"),
            "data_roots": {"naturalistic_usv_repository_dir": str(repo_out)},
        },
        message_output=lambda *_a, **_kw: None,
    ).build()

    written = list((repo_out / "male").glob("naturalistic_usv_repository_courtship_*.h5"))
    assert len(written) == 1
    with h5py.File(str(written[0]), "r") as h5_file:
        assert int(h5_file.attrs["n_usv"]) == 3   # only the male bout
        assert set(e.decode() if isinstance(e, bytes) else e for e in h5_file["usv/emitter"][:]) == {"male_mouse"}


def test_build_rejects_unknown_context_label(tmp_path, _patched_env):
    """An unrecognised ``context_label`` fails fast with a clear ValueError."""

    cfg = _build_cfg("not_a_context")
    with pytest.raises(ValueError, match="context_label must be one of"):
        NaturalisticUsvRepositoryBuilder(
            root_directories=[str(tmp_path)],
            input_parameter_dict={
                "build_naturalistic_usv_repository": cfg,
                "data_roots": {"naturalistic_usv_repository_dir": str(tmp_path / "repo")},
            },
            message_output=lambda *_a, **_kw: None,
        ).build()


# ---------------------------------------------------------------------------
# CLI routing
# ---------------------------------------------------------------------------


def test_cli_routes_to_builder(mocker, tmp_path):
    """``build-naturalistic-usv-repository`` loads the settings block and drives
    the builder exactly once."""

    mocker.patch(
        "usv_playpen.analyses.build_naturalistic_usv_repository.modify_settings_json_for_cli",
        return_value={"build_naturalistic_usv_repository": {}},
    )
    fake_builder = mocker.patch(
        "usv_playpen.analyses.build_naturalistic_usv_repository.NaturalisticUsvRepositoryBuilder"
    )
    result = CliRunner().invoke(build_naturalistic_usv_repository_cli, [])
    assert result.exit_code == 0, result.output
    fake_builder.assert_called_once()
    fake_builder.return_value.build.assert_called_once()
