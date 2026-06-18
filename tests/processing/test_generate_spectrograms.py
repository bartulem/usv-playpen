"""
@author: bartulem
Tests for processing/generate_spectrograms.SpectrogramGenerator.

Builds a synthetic HPSS-filtered audio memmap (whose filename encodes the
sampling rate / sample count / channel count / dtype the way das_inference
writes it) plus a small ``*_usv_summary.csv``, then checks that one session's
spectrograms are written to ``audio/spectrograms/<session>_spectrograms.h5``
with the expected datasets and shapes.
"""

from __future__ import annotations

import h5py
import numpy as np
import polars as pls

from usv_playpen.processing.generate_spectrograms import (
    SpectrogramGenerator,
    compute_usv_spectrogram,
)

_SPEC_PARAMS = {
    "num_freq_bins": 128,
    "num_time_bins": 128,
    "nperseg": 2048,
    "noverlap": 1792,
    "min_freq": 30000.0,
    "max_freq": 120000.0,
    "hop_length": 512,
    "window": "blackmanharris",
    "offset": 0.0,
    "normalize": True,
}


def _build_session(tmp_path, *, sr=250000, n_channels=4, n_usv=3):
    """Create a session dir with a synthetic mmap + usv_summary.csv and return
    its root path. Signal is injected at each USV interval so the variance
    weighting has something to weight."""
    session_id = "20230119_155302"
    root = tmp_path / session_id
    hpss_dir = root / "audio" / "hpss_filtered"
    hpss_dir.mkdir(parents=True)

    n_samples = int(sr * 1.0)  # 1 second
    audio = np.zeros((n_samples, n_channels), dtype=np.int16)
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n_usv):
        start = 0.1 + i * 0.2
        stop = start + 0.05  # 50 ms -> 12500 samples > nperseg
        s0, s1 = int(start * sr), int(stop * sr)
        audio[s0:s1, :] = rng.integers(-8000, 8000, size=(s1 - s0, n_channels), dtype=np.int16)
        rows.append({"usv_id": f"{i:04d}", "start": start, "stop": stop, "duration": stop - start})

    # Filename: ..._<sr>_<n_samples>_<n_channels>_<dtype>.mmap
    mmap_name = f"{session_id}_audio_hpss_filtered_{sr}_{n_samples}_{n_channels}_int16.mmap"
    audio.tofile(hpss_dir / mmap_name)

    pls.DataFrame(rows).write_csv(root / "audio" / f"{session_id}_usv_summary.csv")
    return root, session_id, n_usv


def test_compute_usv_spectrogram_shape_and_range():
    """A single segment yields a (num_freq_bins, num_time_bins) spectrogram
    normalized into [0, 1], plus a positive native time-bin count."""
    sr = 250000
    rng = np.random.default_rng(1)
    segment = rng.integers(-8000, 8000, size=(12500, 4), dtype=np.int16).astype(np.float64)
    spec, n_time = compute_usv_spectrogram(segment, sr, _SPEC_PARAMS, normalize=True)
    assert spec is not None
    assert spec.shape == (128, 128)
    assert spec.min() >= 0.0
    assert spec.max() <= 1.0
    assert n_time > 0


def test_compute_usv_spectrogram_too_short_returns_none():
    """A segment shorter than nperseg on every channel yields no spectrogram."""
    segment = np.zeros((100, 4), dtype=np.float64)
    spec, n_time = compute_usv_spectrogram(segment, 250000, _SPEC_PARAMS)
    assert spec is None
    assert n_time == 0


def test_generate_session_spectrograms_writes_h5(tmp_path, mocker):
    """End-to-end: a session with N USVs writes an H5 with specs/durations/
    spec_ids/freq_bins, row-aligned and shaped (N, F, T)."""
    root, session_id, n_usv = _build_session(tmp_path)
    mocker.patch("usv_playpen.processing.generate_spectrograms.smart_wait")

    SpectrogramGenerator(
        root_directory=str(root),
        input_parameter_dict={"generate_spectrograms": _SPEC_PARAMS},
        message_output=lambda *_a, **_kw: None,
    ).generate_session_spectrograms()

    h5_path = root / "audio" / "spectrograms" / f"{session_id}_spectrograms.h5"
    assert h5_path.is_file()
    with h5py.File(h5_path, "r") as f:
        assert set(f.keys()) == {"specs", "durations", "freq_bins", "spec_ids"}
        assert f["specs"].shape == (n_usv, 128, 128)
        assert f["durations"].shape == (n_usv,)
        assert f["spec_ids"].shape == (n_usv,)
        assert f["freq_bins"].shape == (128,)
        ids = [s.decode() if isinstance(s, bytes) else s for s in f["spec_ids"][:]]
        assert ids == [f"{session_id}_{i}" for i in range(n_usv)]


def test_generate_session_spectrograms_no_usvs_writes_nothing(tmp_path, mocker):
    """A session whose USV intervals are all degenerate writes no H5 and does
    not raise."""
    root, session_id, _ = _build_session(tmp_path, n_usv=0)
    # Overwrite the summary with a zero-length interval.
    pls.DataFrame([{"usv_id": "0000", "start": 0.5, "stop": 0.5, "duration": 0.0}]).write_csv(
        root / "audio" / f"{session_id}_usv_summary.csv"
    )
    mocker.patch("usv_playpen.processing.generate_spectrograms.smart_wait")

    msgs: list[str] = []
    SpectrogramGenerator(
        root_directory=str(root),
        input_parameter_dict={"generate_spectrograms": _SPEC_PARAMS},
        message_output=msgs.append,
    ).generate_session_spectrograms()

    assert not (root / "audio" / "spectrograms" / f"{session_id}_spectrograms.h5").exists()
    assert any("No spectrograms generated" in m for m in msgs)
