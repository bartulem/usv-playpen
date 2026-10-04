"""
@author: bartulem
Tests for processing/compute_usv_loudness: the absolute, masked, variance-weighted
image-level loudness a QLVM loudness conditional is decoded at.

The port was checked against the package's corpus values outside the test suite
(every checked call of two corpus sessions reproduced ``image_level_db``
exactly); these tests pin the properties that check relies on with synthetic audio.
"""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from usv_playpen.processing.compute_usv_loudness import (
    call_image_level_db,
    session_image_level_db,
)

_SR = 250000
_SPEC_PARAMS = {
    "num_freq_bins": 128,
    "num_time_bins": 128,
    "nperseg": 2048,
    "min_freq": 30000.0,
    "max_freq": 120000.0,
    "hop_length": 512,
    "window": "blackmanharris",
    "offset": 0.0,
}
_FULL = np.ones((128, 128), dtype=bool)


def _tone(n_samples, freq_hz, amplitude, rng):
    """A tone plus a little noise, as int16 counts in float64."""
    t = np.arange(n_samples) / _SR
    return amplitude * np.sin(2 * np.pi * freq_hz * t) + rng.normal(0.0, 20.0, n_samples)


def test_call_image_level_db_scales_with_the_audio_level():
    """Power is kept absolute: ten times the amplitude is 20 dB louder."""
    rng = np.random.default_rng(0)
    segment = _tone(12500, 60000.0, 3000.0, rng)[:, None]
    quiet = call_image_level_db(segment, _SR, _SPEC_PARAMS, _FULL)
    loud = call_image_level_db(10.0 * segment, _SR, _SPEC_PARAMS, _FULL)
    assert loud - quiet == pytest.approx(20.0, abs=1e-4)


def test_call_image_level_db_weights_channels_by_their_variance():
    """Channels combine as sum(w * L) / sum(w), w the de-meaned segment variance,
    so the loud channel dominates."""
    rng = np.random.default_rng(1)
    loud = _tone(12500, 60000.0, 6000.0, rng)
    quiet = _tone(12500, 60000.0, 600.0, rng)
    level_loud = call_image_level_db(loud[:, None], _SR, _SPEC_PARAMS, _FULL)
    level_quiet = call_image_level_db(quiet[:, None], _SR, _SPEC_PARAMS, _FULL)
    w_loud, w_quiet = np.var(loud - loud.mean()), np.var(quiet - quiet.mean())
    expected = (w_loud * level_loud + w_quiet * level_quiet) / (w_loud + w_quiet)
    got = call_image_level_db(np.stack([loud, quiet], axis=1), _SR, _SPEC_PARAMS, _FULL)
    assert got == pytest.approx(expected, abs=1e-4)
    assert level_quiet < got < level_loud


def test_call_image_level_db_averages_only_inside_the_region():
    """The level is the call's mask region's: the rows holding the tone are
    louder than rows of noise only."""
    rng = np.random.default_rng(2)
    segment = _tone(12500, 60000.0, 3000.0, rng)[:, None]
    tone_row = round((60000.0 - 30000.0) / 90000.0 * 127)
    on_tone, off_tone = np.zeros((128, 128), dtype=bool), np.zeros((128, 128), dtype=bool)
    on_tone[tone_row - 1:tone_row + 2, :] = True
    off_tone[110:, :] = True
    assert call_image_level_db(segment, _SR, _SPEC_PARAMS, on_tone) > call_image_level_db(segment, _SR, _SPEC_PARAMS, off_tone) + 10.0


def test_call_image_level_db_is_nan_without_a_usable_call():
    """A segment shorter than one STFT window, or a region with no pixel over the
    call's frames, has no level."""
    rng = np.random.default_rng(3)
    assert np.isnan(call_image_level_db(_tone(1000, 60000.0, 3000.0, rng)[:, None], _SR, _SPEC_PARAMS, _FULL))
    late = np.zeros((128, 128), dtype=bool)
    late[:, 120:] = True                              # a 12,500-sample call spans 25 frames
    assert np.isnan(call_image_level_db(_tone(12500, 60000.0, 3000.0, rng)[:, None], _SR, _SPEC_PARAMS, late))


def test_session_image_level_db_slices_like_the_generator_and_drops_excluded_channels(tmp_path):
    """Calls are cut from the HPSS memmap at floor(start * sr):ceil(stop * sr) and
    the metadata's excluded channels are left out, as the spectrograms were made."""
    session_id = "20230119_155302"
    root = tmp_path / session_id
    (root / "audio" / "hpss_filtered").mkdir(parents=True)
    rng = np.random.default_rng(4)
    n_samples, n_channels = _SR, 3
    audio = np.zeros((n_samples, n_channels), dtype=np.int16)
    starts, stops = np.array([0.1000013, 0.5]), np.array([0.1500007, 0.56])
    for start, stop in zip(starts, stops, strict=True):
        s0, s1 = int(np.floor(start * _SR)), int(np.ceil(stop * _SR))
        for ch in range(n_channels):
            amplitude = 20000.0 if ch == 1 else 2000.0 * (ch + 1)    # column 1 = m_ch02, excluded and loudest
            audio[s0:s1, ch] = np.clip(_tone(s1 - s0, 60000.0, amplitude, rng), -32768, 32767).astype(np.int16)
    audio.tofile(root / "audio" / "hpss_filtered" / f"sess_concatenated_audio_hpss_filtered_{_SR}_{n_samples}_{n_channels}_int16.mmap")
    (root / f"{session_id}_metadata.yaml").write_text(
        yaml.dump({"Equipment": {"audio_Avisoft": {"excluded_channels": ["m_ch02"]}}})
    )
    regions = np.ones((2, 128, 128), dtype=bool)

    got = session_image_level_db(str(root), starts, stops, regions, _SPEC_PARAMS, message_output=lambda *_a: None)

    assert got.dtype == np.float32
    for call, (start, stop) in enumerate(zip(starts, stops, strict=True)):
        segment = audio[int(np.floor(start * _SR)):int(np.ceil(stop * _SR)), [0, 2]]
        assert got[call] == np.float32(call_image_level_db(segment, _SR, _SPEC_PARAMS, regions[call]))
