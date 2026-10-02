"""
@author: bartulem
Tests for processing/build_squeak_spectrogram_store.

The log-frequency bins and their weights (averaging where linear bins fall
inside a log bin, interpolation where none does, rows summing to 1), the
channel-averaged log spectrogram of pure tones, the dB quantization round trip,
the display-window placement and the explorer's tile reader are tested
directly. The builder runs end to end on a synthetic session (per-channel
PCM_16 wavs under ``audio/hpss`` plus a ``*_usv_summary.csv`` with squeak and
noise columns) and the written store is read back: which rows it holds, their
frame counts, the window of a segment longer than the stored window, the
quantized values against a direct recomputation, the attrs, and the resolver
that finds it without matching the consolidated store's pattern.
"""

from __future__ import annotations

import pathlib

import h5py
import librosa
import numpy as np
import polars as pls
import pytest
import soundfile as sf
import yaml

from usv_playpen.os_utils import (
    resolve_consolidated_h5_path,
    resolve_squeak_spectrogram_store_path,
)
from usv_playpen.processing import build_squeak_spectrogram_store as store
from usv_playpen.processing.detect_usv_squeaks import (
    SQUEAK_SPEC_PARAMS,
    squeak_crop_frames,
)

SESSION_ID = "20250913_193920"
SAMPLING_RATE = 250000

CFG = {
    "min_freq": 2000.0,
    "max_freq": 125000.0,
    "n_frequency_bins": 64,
    "window_frames": 64,
    "db_floor": -100.0,
    "db_ceil": 60.0,
    "exclude_metadata_audio_channels": True,
    "n_workers": 1,
}


def _linear_freqs() -> np.ndarray:
    """
    Description
    -----------
    Centre frequencies of the squeak front end's linear STFT bins.

    Parameters
    ----------

    Returns
    -------
    freqs (np.ndarray)
        ``(1025,)`` frequencies in Hz.
    """

    return librosa.fft_frequencies(sr=SAMPLING_RATE, n_fft=SQUEAK_SPEC_PARAMS["nperseg"])


def _build_session(tmp_path: pathlib.Path, excluded_channels: list[str] | None = None) -> pathlib.Path:
    """
    Description
    -----------
    Creates a synthetic session: four PCM_16 HPSS wavs (two master, two slave
    channels) of 2 s of noise plus a 5 kHz tone, and a summary with a 50 ms
    squeak (row 0), a 0.6 s squeak whose extent lies late in the segment
    (row 1, longer than the 64-frame test window), a squeak flagged as noise
    (row 2), a non-squeak (row 3) and a 4 ms squeak too short for one STFT
    frame (row 4). Optionally writes session metadata excluding channels.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Pytest temporary directory.
    excluded_channels (list[str] | None)
        Channel names to record as excluded in the session metadata.

    Returns
    -------
    root (pathlib.Path)
        Session root directory.
    """

    root = tmp_path / SESSION_ID
    hpss_dir = root / "audio" / "hpss"
    hpss_dir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    time_s = np.arange(2 * SAMPLING_RATE) / SAMPLING_RATE
    for device, channel in (("m", 1), ("m", 2), ("s", 1), ("s", 2)):
        audio = rng.normal(0.0, 300.0, size=time_s.size) + 8000.0 * np.sin(2 * np.pi * 5000.0 * time_s)
        sf.write(
            str(hpss_dir / f"{device}_250913193920_ch{channel:02d}_cropped_to_video_hpss.wav"),
            np.clip(audio, -32768, 32767).astype(np.int16),
            SAMPLING_RATE,
            subtype="PCM_16",
        )
    pls.DataFrame(
        {
            "usv_id": ["0000", "0001", "0002", "0003", "0004"],
            "start": [0.10, 0.30, 1.00, 1.20, 1.50],
            "stop": [0.15, 0.90, 1.05, 1.25, 1.504],
            "squeak": [True, True, True, False, True],
            "squeak_probability": [0.9, 0.8, 0.7, None, 0.6],
            "squeak_start": [0.11, 0.70, 1.01, None, None],
            "squeak_end": [0.14, 0.80, 1.04, None, None],
            "noise": [False, None, True, False, False],
        },
        schema={
            "usv_id": pls.String, "start": pls.Float64, "stop": pls.Float64, "squeak": pls.Boolean,
            "squeak_probability": pls.Float64, "squeak_start": pls.Float64, "squeak_end": pls.Float64,
            "noise": pls.Boolean,
        },
    ).write_csv(root / "audio" / f"{SESSION_ID}_usv_summary.csv")
    if excluded_channels is not None:
        metadata = {"Equipment": {"audio_Avisoft": {"excluded_channels": excluded_channels}}}
        (root / f"{SESSION_ID}_metadata.yaml").write_text(yaml.dump(metadata))
    return root


def test_log_frequency_bins_are_geometric():
    """Edges are geometric from min to max frequency and centres are the geometric means of the edges."""
    edges, centres = store.log_frequency_bins(2000.0, 128000.0, 6)
    np.testing.assert_allclose(edges, [2000, 4000, 8000, 16000, 32000, 64000, 128000])
    np.testing.assert_allclose(centres, np.sqrt(edges[:-1] * edges[1:]))
    with pytest.raises(ValueError, match="min_freq < max_freq"):
        store.log_frequency_bins(5000.0, 2000.0, 8)


def test_log_frequency_weights_average_inside_and_interpolate_below_the_spacing():
    """Rows sum to 1; a wide log bin averages the linear bins inside it; a narrow one interpolates at its centre."""
    freqs = _linear_freqs()
    edges, centres = store.log_frequency_bins(2000.0, 125000.0, 128)
    weights = store.log_frequency_weights(freqs, edges)
    assert weights.shape == (128, freqs.size)
    np.testing.assert_allclose(weights.sum(axis=1), 1.0)
    np.testing.assert_allclose(weights @ np.ones(freqs.size), 1.0)

    top = weights[-1]
    inside = (freqs >= edges[-2]) & (freqs <= edges[-1])
    np.testing.assert_allclose(top[inside], 1.0 / inside.sum())
    assert np.all(top[~inside] == 0.0)

    first = weights[0]
    assert np.count_nonzero(first) == 2
    left, right = np.flatnonzero(first)
    assert right == left + 1
    assert freqs[left] <= centres[0] <= freqs[right]
    np.testing.assert_allclose(first[left] * freqs[left] + first[right] * freqs[right], centres[0])

    linear_power = freqs.copy()
    mapped = weights @ linear_power
    assert np.all(np.diff(mapped) > 0)

    with pytest.raises(ValueError, match="outside the linear"):
        store.log_frequency_weights(freqs, np.array([1000.0, 200000.0]))


@pytest.mark.parametrize("tone_hz", [3000.0, 5000.0, 60000.0])
def test_log_frequency_spectrogram_puts_a_tone_in_its_log_bin(tone_hz):
    """A pure tone peaks in the log bin whose edges contain it, on every frame, in absolute dB."""
    edges, _ = store.log_frequency_bins(2000.0, 125000.0, 128)
    weights = store.log_frequency_weights(_linear_freqs(), edges)
    time_s = np.arange(int(0.05 * SAMPLING_RATE)) / SAMPLING_RATE
    tone = 0.25 * np.sin(2 * np.pi * tone_hz * time_s)
    spectrogram_db, n_frames = store.log_frequency_spectrogram(np.stack([tone, 0.5 * tone], axis=1), weights)
    assert spectrogram_db.dtype == np.float32
    assert n_frames == 1 + time_s.size // SQUEAK_SPEC_PARAMS["hop_length"]
    expected_bin = int(np.searchsorted(edges, tone_hz, side="right")) - 1
    peak_bins = spectrogram_db[:, 2:-2].argmax(axis=0)
    assert np.all(np.abs(peak_bins - expected_bin) <= 1)
    assert spectrogram_db.max() > 0.0


def test_log_frequency_spectrogram_without_a_full_window_is_none():
    """Channels shorter than one STFT window give no spectrogram."""
    edges, _ = store.log_frequency_bins(2000.0, 125000.0, 16)
    weights = store.log_frequency_weights(_linear_freqs(), edges)
    spectrogram_db, n_frames = store.log_frequency_spectrogram(np.zeros((1000, 3)), weights)
    assert spectrogram_db is None
    assert n_frames == 0


def test_quantization_round_trip_and_clipping():
    """Codes span 0..255 over [db_floor, db_ceil]; decoding is within half a step; values outside are clipped."""
    values = np.linspace(-100.0, 60.0, 1001)
    codes = store.quantize_db(values, -100.0, 60.0)
    assert codes.dtype == np.uint8
    assert codes[0] == 0
    assert codes[-1] == 255
    decoded = store.dequantize_db(codes, -100.0, 60.0)
    assert np.max(np.abs(decoded - values)) <= 160.0 / 510.0 + 1e-4
    np.testing.assert_array_equal(store.quantize_db(np.array([-150.0, 90.0]), -100.0, 60.0), [0, 255])


def test_display_window_first():
    """A segment that fits starts at 0; a longer one is centred on its crop and clipped to the segment."""
    assert store.display_window_first(50, 10, 20, 64) == 0
    assert store.display_window_first(300, None, None, 64) == 0
    assert store.display_window_first(300, 140, 160, 64) == 118
    assert store.display_window_first(300, 0, 4, 64) == 0
    assert store.display_window_first(300, 290, 299, 64) == 236


def test_builder_writes_the_squeak_rows_of_every_session(tmp_path):
    """The store holds the squeak rows that are not noise, their frames, the centred long window and the attrs."""
    root = _build_session(tmp_path, excluded_channels=["s_ch02"])
    output = tmp_path / "spectrograms"
    output.mkdir()
    messages = []
    store_path = store.SqueakSpectrogramStoreBuilder(
        root_directories=[str(root)],
        input_parameter_dict={"build_squeak_spectrogram_store": CFG, "spectrograms_root": str(output)},
        message_output=messages.append,
    ).build()

    assert store_path.parent == output
    assert store_path.name.startswith("squeak_spectrograms_64logbins_1sessions_3squeaks_")
    assert resolve_squeak_spectrogram_store_path(str(output)) == str(store_path)
    with pytest.raises(FileNotFoundError):
        resolve_consolidated_h5_path(str(output))
    assert any("s_ch02" in message for message in messages)

    summary = pls.read_csv(root / "audio" / f"{SESSION_ID}_usv_summary.csv")
    with h5py.File(store_path, "r") as h5:
        assert h5.attrs["window_frames"] == 64
        assert h5.attrs["db_floor"] == -100.0
        assert h5.attrs["db_ceil"] == 60.0
        assert h5.attrs["n_squeaks"] == 3
        edges, centres = store.log_frequency_bins(2000.0, 125000.0, 64)
        np.testing.assert_allclose(h5["frequency_bins"][:], centres)
        np.testing.assert_allclose(h5["frequency_bin_edges"][:], edges)
        group = h5[f"spectrogram/{SESSION_ID}"]
        np.testing.assert_array_equal(group["row_index"][:], [0, 1, 4])
        assert group["spectrograms"].shape == (3, 64, 64)
        assert group["spectrograms"].dtype == np.uint8
        assert group["spectrograms"].chunks == (1, 64, 64)

        n_samples = np.round(summary["stop"].to_numpy() * SAMPLING_RATE) - np.round(summary["start"].to_numpy() * SAMPLING_RATE)
        expected_frames = (1 + n_samples // SQUEAK_SPEC_PARAMS["hop_length"]).astype(int)
        np.testing.assert_array_equal(group["n_frames"][:], [expected_frames[0], expected_frames[1], 0])
        np.testing.assert_array_equal(group["durations"][:], [expected_frames[0], 64, 0])

        first, last = squeak_crop_frames(
            segment_start_s=np.array([0.30]), squeak_start_s=np.array([0.70]), squeak_end_s=np.array([0.80]),
            n_frames=np.array([expected_frames[1]]),
        )
        expected_first = store.display_window_first(int(expected_frames[1]), int(first[0]), int(last[0]), 64)
        assert expected_first > 0
        np.testing.assert_array_equal(group["window_first"][:], [0, expected_first, 0])
        assert np.all(group["spectrograms"][0, :, expected_frames[0]:] == 0)
        assert np.all(group["spectrograms"][2] == 0)

        wav_paths = sorted((root / "audio" / "hpss").glob("*_cropped_to_video_hpss.wav"))
        kept = [path for path in wav_paths if not path.name.startswith("s_") or "_ch02_" not in path.name]
        audio = np.stack([sf.read(str(path), dtype="float64")[0][25000:37500] for path in kept], axis=1)
        weights = store.log_frequency_weights(_linear_freqs(), edges)
        direct_db, _ = store.log_frequency_spectrogram(audio, weights)
        np.testing.assert_array_equal(group["spectrograms"][0, :, :direct_db.shape[1]], store.quantize_db(direct_db, -100.0, 60.0))

        sessions = h5["sessions"][:]
        assert sessions["session_id"][0].decode() == SESSION_ID
        assert int(sessions["n_squeaks"][0]) == 3


def test_builder_reports_a_broken_session_and_writes_the_rest(tmp_path):
    """A session without audio is reported and left out; the store still holds the working session."""
    root = _build_session(tmp_path)
    broken = tmp_path / "20250101_000000"
    (broken / "audio").mkdir(parents=True)
    output = tmp_path / "spectrograms"
    output.mkdir()
    messages = []
    store_path = store.SqueakSpectrogramStoreBuilder(
        root_directories=[str(root), str(broken)],
        input_parameter_dict={"build_squeak_spectrogram_store": CFG, "spectrograms_root": str(output)},
        message_output=messages.append,
    ).build()
    assert any(message.startswith("FAILED") and "20250101_000000" in message for message in messages)
    with h5py.File(store_path, "r") as h5:
        assert list(h5["spectrogram"]) == [SESSION_ID]
        assert "20250101_000000" in h5.attrs["failed_sessions"]


def test_squeak_store_thumbnail_reads_a_row_and_scales_it(tmp_path):
    """The tile is the call's stored frames scaled over the top dynamic range; unknown or empty rows give None."""
    with h5py.File(tmp_path / "tiny.h5", "w") as h5:
        group = h5.create_group("spectrogram/s")
        tile_db = np.full((4, 10), -80.0)
        tile_db[1, 2:5] = 20.0
        codes = np.zeros((2, 4, 10), dtype=np.uint8)
        codes[0, :, :6] = store.quantize_db(tile_db[:, :6], -100.0, 60.0)
        group.create_dataset("spectrograms", data=codes)
        group.create_dataset("row_index", data=np.array([3, 7], dtype=np.int64))
        group.create_dataset("durations", data=np.array([6, 0], dtype=np.int32))
        tile = store.squeak_store_thumbnail(group, 3, -100.0, 60.0, 60.0)
        assert tile.shape == (4, 6)
        assert tile.max() == pytest.approx(1.0)
        assert tile[0, 0] == 0.0
        assert store.squeak_store_thumbnail(group, 7, -100.0, 60.0, 60.0) is None
        assert store.squeak_store_thumbnail(group, 5, -100.0, 60.0, 60.0) is None
        assert store.squeak_store_thumbnail(group, 99, -100.0, 60.0, 60.0) is None


def test_read_session_lists_skips_comments_and_duplicates(tmp_path):
    """Blank lines, comments and repeated roots are skipped; a missing list raises."""
    list_path = tmp_path / "sessions.txt"
    list_path.write_text("/a/one\n\n# comment\n/a/two\n/a/one\n")
    assert store.read_session_lists([str(list_path)]) == ["/a/one", "/a/two"]
    with pytest.raises(FileNotFoundError):
        store.read_session_lists([str(tmp_path / "missing.txt")])


def test_squeak_store_resolver_raises_without_a_store(tmp_path):
    """With only a consolidated store present the squeak resolver raises (the explorer's fallback signal) and vice versa."""
    (tmp_path / "spectrograms_qlvmv3_1sessions_1vocalizations_20260101_000000Z.h5").touch()
    with pytest.raises(FileNotFoundError, match="squeak spectrogram store"):
        resolve_squeak_spectrogram_store_path(str(tmp_path))
    (tmp_path / "squeak_spectrograms_128logbins_1sessions_1squeaks_20260101_000000Z.h5").touch()
    assert resolve_squeak_spectrogram_store_path(str(tmp_path)).endswith("squeak_spectrograms_128logbins_1sessions_1squeaks_20260101_000000Z.h5")
    assert resolve_consolidated_h5_path(str(tmp_path)).endswith("spectrograms_qlvmv3_1sessions_1vocalizations_20260101_000000Z.h5")
