"""
@author: bartulem
Tests for processing/detect_usv_squeaks.

The call-class model: the network (and its noise-trunk initialization), the squeak-extent rule (the
envelope of above-threshold frames in runs touching the segment, no minimum run, half-hop edges, and
the highest-scoring-frame fallback), the window input (segment indicator, frame grid) and the bundle
loader with its refusals are tested directly. The end-to-end classifier runs on a synthetic session
(a four-channel broadband memmap plus a small ``*_usv_summary.csv``) with a stub ensemble
whose class logits are set by the segment length and whose frame logits are set per frame relative to
the segment, so every row's class, probabilities and extent are known: that pins the boolean encoding
(pure USV (true, false), pure squeak (false, true), both (true, true), nulls on noise rows),
probabilities summing to 1, the single extent on squeak rows only, the fallback count, and the merge
that removes the columns of earlier encodings.

Training: the labelling tool's label sets with review overrides, the frame targets, one training seed
from a noise trunk, and an end-to-end training run on the synthetic session whose bundle the detector's
loader accepts.

The squeak QLVM embedding is tested on the same kind of synthetic session: the crop window that
widens a segment to hold a squeak running past it, the frame rule, the crop normalization and
centring, the row selection (squeak and both, never noise or usv) and exclusions, the old-layout cell
loader and its checks, and the merge of ``qlvm_squeak1`` / ``qlvm_squeak2`` with the decoder mocked.
"""

from __future__ import annotations

import json
import pathlib

import jax.numpy as jnp
import numpy as np
import polars as pls
import pytest
import torch
import yaml
from click.testing import CliRunner

from tests.conftest import write_broadband_audio
from usv_playpen.processing import detect_usv_squeaks as squeaks
from usv_playpen.processing.detect_usv_noise import HOP_SAMPLES, NoiseTimeMIL

SESSION_ID = "20250913_193920"
SAMPLING_RATE = 250000
HOP_S = HOP_SAMPLES / SAMPLING_RATE


class _StubCallClassModel(torch.nn.Module):
    """
    Description
    -----------
    Stand-in for USVSqueakTimeMIL. Its class logits depend only on the number of segment frames of each
    input (a lookup, default "usv"), and its frame logits are +5 on the frames listed relative to the
    first segment frame and -5 elsewhere, so every row's class and squeak track are known.
    """

    def __init__(self, class_by_segment_frames: dict[int, int], above_offsets: list[int], peak_offset: int | None = None) -> None:
        """
        Description
        -----------
        Stores the class lookup and the above-threshold frame offsets.

        Parameters
        ----------
        class_by_segment_frames (dict[int, int])
            Segment frame count -> class index whose logit is +4 (the others 0).
        above_offsets (list[int])
            Frame offsets from the first segment frame (negative = context before) whose logit is +5.
        peak_offset (int | None)
            Optional frame offset whose logit is -1 (below the threshold but above every other frame
            left at -5), the frame the extent rule's fallback must pick.

        Returns
        -------
        None
        """

        super().__init__()
        self.class_by_segment_frames = class_by_segment_frames
        self.above_offsets = above_offsets
        self.peak_offset = peak_offset

    def forward(self, x: torch.Tensor, valid: torch.Tensor, segment: torch.Tensor, scalars: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # noqa: ARG002
        """
        Description
        -----------
        Returns the stub class logits and frame logits.

        Parameters
        ----------
        x (torch.Tensor)
            ``(B, 3, 128, T)`` inputs (unused but for the shape).
        valid (torch.Tensor)
            ``(B, T)`` validity.
        segment (torch.Tensor)
            ``(B, T)`` segment frames.
        scalars (torch.Tensor)
            ``(B, 2)`` scalars (unused).

        Returns
        -------
        class_logits (torch.Tensor)
            ``(B, 3)``.
        frame_logits (torch.Tensor)
            ``(B, T)``.
        """

        n_batch, n_time = x.shape[0], x.shape[-1]
        class_logits = torch.zeros((n_batch, 3))
        frame_logits = torch.full((n_batch, n_time), -5.0)
        for item in range(n_batch):
            n_segment = int(segment[item].sum())
            class_logits[item, self.class_by_segment_frames.get(n_segment, 0)] = 4.0
            first = int(torch.nonzero(segment[item])[0, 0])
            if self.peak_offset is not None:
                frame_logits[item, first + self.peak_offset] = -1.0
            for offset in self.above_offsets:
                if 0 <= first + offset < n_time and bool(valid[item, first + offset]):
                    frame_logits[item, first + offset] = 5.0
        return class_logits, frame_logits


def _write_wavs(root: pathlib.Path, seconds: float = 1.0) -> None:
    """
    Description
    -----------
    Writes a broadband memmap of four channels (two master, two slave) of noise, with its record.

    Parameters
    ----------
    root (pathlib.Path)
        Session root directory.
    seconds (float)
        Length of every wav.

    Returns
    -------
    None
    """

    rng = np.random.default_rng(0)
    write_broadband_audio(root, [
        (f"{device}_250913193920_ch{channel:02d}_cropped_to_video_hpss.wav", rng.integers(-3000, 3000, size=int(SAMPLING_RATE * seconds), dtype=np.int16))
        for device, channel in (("m", 1), ("m", 2), ("s", 1), ("s", 2))
    ], SAMPLING_RATE)


def _build_session(tmp_path: pathlib.Path, excluded_channels: list[str] | None = None) -> pathlib.Path:
    """
    Description
    -----------
    Creates a synthetic session: a broadband memmap and a summary with five rows: a 50 ms segment, a 400 ms
    segment, a 4 ms segment (too short for one STFT window), a noise segment and a 100 ms segment.
    Optionally writes session metadata excluding channels.

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
    _write_wavs(root)
    pls.DataFrame(
        {
            "usv_id": ["0000", "0001", "0002", "0003", "0004"],
            "start": [0.10, 0.30, 0.80, 0.85, 0.20],
            "stop": [0.15, 0.70, 0.804, 0.90, 0.30],
            "duration": [0.05, 0.40, 0.004, 0.05, 0.10],
            "chs_count": [10, 20, 3, 5, 8],
            "emitter": [None, None, None, None, None],
            "noise": [False, None, False, True, False],
        },
        schema={"usv_id": pls.String, "start": pls.Float64, "stop": pls.Float64, "duration": pls.Float64,
                "chs_count": pls.Int64, "emitter": pls.String, "noise": pls.Boolean},
    ).write_csv(root / "audio" / f"{SESSION_ID}_usv_summary.csv")
    if excluded_channels is not None:
        metadata = {"Equipment": {"audio_Avisoft": {"excluded_channels": excluded_channels}}}
        (root / f"{SESSION_ID}_metadata.yaml").write_text(yaml.dump(metadata))
    return root


def _segment_frames(start: float, stop: float) -> int:
    """
    Description
    -----------
    The segment-frame count the window input marks: ``1 + (ceil(stop * fs) - floor(start * fs)) // hop``.

    Parameters
    ----------
    start (float)
        Segment start (s).
    stop (float)
        Segment stop (s).

    Returns
    -------
    n (int)
        Segment frames.
    """

    return 1 + (int(np.ceil(stop * SAMPLING_RATE)) - int(np.floor(start * SAMPLING_RATE))) // HOP_SAMPLES


def _stub_bundle(fallback: bool = False) -> dict:
    """
    Description
    -----------
    A loaded-bundle dict around the stub model: the 50 ms row is "both", the 400 ms row "squeak", every
    other row "usv". By default the frame track is above the threshold on frames -30 .. -16 from the
    first segment frame (a run wholly in the context before the segment, which the extent ignores),
    2 .. 29 (a run that starts inside the segment and, for the 25-frame 50 ms segment, extends past its
    end) and 32 .. 36 (a 5-frame run, counted when it touches the segment: there is no minimum run). With ``fallback`` only the context run is above the threshold and frame 7 holds the
    highest segment score, so every squeak row falls back to frame 7.

    Parameters
    ----------
    fallback (bool)
        Build the fallback track instead.

    Returns
    -------
    bundle (dict)
        The keys :func:`classify_usv_squeak_rows` reads.
    """

    above = list(range(-30, -15)) if fallback else list(range(-30, -15)) + list(range(2, 30)) + list(range(32, 37))
    model = _StubCallClassModel(
        {_segment_frames(0.10, 0.15): 2, _segment_frames(0.30, 0.70): 1, _segment_frames(0.20, 0.30): 0},
        above,
        peak_offset=7 if fallback else None,
    )
    return {
        **squeaks.USV_SQUEAK_INPUT_CONTRACT,
        "models": [model],
        "class_names": squeaks.CALL_CLASSES,
        "scalar_mean": np.zeros(2, dtype=np.float32),
        "scalar_std": np.ones(2, dtype=np.float32),
        "span_threshold": 0.6,
        "labels": [],
    }


def _write_bundle(path: pathlib.Path, model_name: str = squeaks.USV_SQUEAK_MODEL_NAME) -> None:
    """
    Description
    -----------
    Writes a randomly initialized two-member call-class bundle in the trainer's format.

    Parameters
    ----------
    path (pathlib.Path)
        Bundle path.
    model_name (str)
        ``model`` entry (a wrong one must be refused).

    Returns
    -------
    None
    """

    torch.manual_seed(0)
    state_dicts = [squeaks.USVSqueakTimeMIL(in_channels=3, n_scalars=2).state_dict() for _ in range(2)]
    torch.save({
        "model": model_name, "state_dicts": state_dicts, "class_names": ["usv", "squeak", "both"],
        "in_channels": 3, "n_scalars": 2, "scalar_names": ["log_chs_count", "log_duration_s"],
        "scalar_mean": [2.9, -2.2], "scalar_std": [0.5, 0.6],
        **{key: ([list(band) for band in value] if key == "bands_hz" else value) for key, value in squeaks.USV_SQUEAK_INPUT_CONTRACT.items()},
        "sampling_rate": SAMPLING_RATE, "hop_samples": HOP_SAMPLES,
        "extent_rule": {"threshold": 0.6, "min_run_frames": 12}, "labels": ["a.csv"],
    }, path)


def test_usv_squeak_timemil_shapes_and_noise_trunk():
    """The network returns (B, 3) class logits and (B, T) frame logits, and a noise model's trunk
    weights load into it without unexpected keys."""
    model = squeaks.USVSqueakTimeMIL(in_channels=3, n_scalars=2).eval()
    noise_trunk = {key: value for key, value in NoiseTimeMIL(in_channels=3, n_scalars=2).state_dict().items() if key.startswith(("body.", "tcn."))}
    loaded = model.load_state_dict(noise_trunk, strict=False)
    assert not loaded.unexpected_keys
    assert all(key.startswith(("class_", "squeak_frame")) for key in loaded.missing_keys)
    x = torch.randn(2, 3, 128, 40)
    valid = torch.ones(2, 40, dtype=torch.bool)
    segment = torch.zeros(2, 40, dtype=torch.bool)
    segment[:, 10:20] = True
    class_logits, frame_logits = model(x, valid, segment, torch.zeros(2, 2))
    assert class_logits.shape == (2, 3)
    assert frame_logits.shape == (2, 40)


def test_squeak_envelope_frames_rule():
    """The extent runs from the first to the last frame strictly above the threshold, over every run
    that touches the segment (no minimum run; gaps included), ignoring runs wholly in the context."""
    track = np.zeros(60)
    track[2:6] = 0.9
    track[12:14] = 0.9
    track[20:22] = 0.61
    track[30:40] = 0.9
    track[45:50] = 0.6
    assert squeaks.squeak_envelope_frames(track, 0.6, 10, 25) == (12, 21, False)
    assert squeaks.squeak_envelope_frames(track, 0.6, 5, 25) == (2, 21, False)
    assert squeaks.squeak_envelope_frames(track, 0.6, 21, 31) == (20, 39, False)
    assert squeaks.squeak_envelope_frames(track, 0.6, 45, 55) == (45 + int(np.argmax(track[45:56])), 45 + int(np.argmax(track[45:56])), True)


def test_squeak_envelope_frames_fallback_takes_the_best_segment_frame():
    """With no frame above the threshold touching the segment, the extent is the one segment frame of
    highest score, even when a context frame scores higher."""
    track = np.full(40, 0.1)
    track[3] = 0.95
    track[17] = 0.4
    track[22] = 0.3
    assert squeaks.squeak_envelope_frames(track, 0.6, 10, 30) == (17, 17, True)


def test_frames_to_session_seconds_uses_half_hop_edges():
    """An extent runs from half a hop before its first frame's centre to half a hop after its last."""
    start, end = squeaks.frames_to_session_seconds(3, 14, 10.0)
    assert start == pytest.approx(10.0 + 2.5 * HOP_S, abs=1e-6)
    assert end == pytest.approx(10.0 + 14.5 * HOP_S, abs=1e-6)


def test_usv_squeak_window_input_marks_the_segment(tmp_path):
    """The window holds 49 context hops before the segment (fewer at the file start), frame 0 is
    centred at the read start, and the indicator marks exactly the segment frames."""
    root = _build_session(tmp_path)
    handles = squeaks.squeak_audio_channels(root, False, print)
    try:
        window = squeaks.usv_squeak_window_input(handles, handles[0].frames, 0.30, 0.70, squeaks.USV_SQUEAK_INPUT_CONTRACT)
        early = squeaks.usv_squeak_window_input(handles, handles[0].frames, 0.05, 0.10, squeaks.USV_SQUEAK_INPUT_CONTRACT)
        short = squeaks.usv_squeak_window_input(handles, handles[0].frames, 0.0, 0.001, squeaks.USV_SQUEAK_INPUT_CONTRACT)
    finally:
        for handle in handles:
            handle.close()
    assert window["x"].shape[:2] == (3, 128)
    assert window["first_frame"] == 49
    assert window["read_start_s"] == pytest.approx((np.floor(0.30 * SAMPLING_RATE) - 49 * HOP_SAMPLES) / SAMPLING_RATE)
    indicator = window["x"][2, 0]
    assert np.flatnonzero(indicator).tolist() == list(range(49, 49 + _segment_frames(0.30, 0.70)))
    assert window["n_segment_frames"] == _segment_frames(0.30, 0.70)
    assert np.all(window["x"][2] == indicator[None, :])
    assert early["first_frame"] == int(np.floor(0.05 * SAMPLING_RATE)) // HOP_SAMPLES
    assert early["read_start_s"] == pytest.approx(0.05 - early["first_frame"] * HOP_S, abs=1e-6)
    assert short is not None
    assert short["n_segment_frames"] == 1


def test_load_usv_squeak_model_round_trip_and_refusals(tmp_path):
    """A trainer-format bundle loads with its members, rule and constants; a missing file, a wrong
    model name or wrong class names are refused."""
    _write_bundle(tmp_path / "ok.pt")
    bundle = squeaks.load_usv_squeak_model(str(tmp_path / "ok.pt"), torch.device("cpu"))
    assert len(bundle["models"]) == 2
    assert bundle["class_names"] == ("usv", "squeak", "both")
    assert bundle["span_threshold"] == 0.6
    assert bundle["max_frames"] == 1024
    assert bundle["context_frames"] == 49
    with pytest.raises(FileNotFoundError):
        squeaks.load_usv_squeak_model(str(tmp_path / "missing.pt"), torch.device("cpu"))
    _write_bundle(tmp_path / "old.pt", model_name="timemil_binary")
    with pytest.raises(ValueError, match="retired binary"):
        squeaks.load_usv_squeak_model(str(tmp_path / "old.pt"), torch.device("cpu"))


def test_classify_usv_squeak_rows_encoding(tmp_path):
    """
    The boolean encoding on every kind of row: the "both" (50 ms) row is (true, true), the "squeak"
    (400 ms) row (false, true), the "usv" rows (true, false), and the noise row is not read and gets
    nulls everywhere (a null noise value counts as not noise). Squeak rows get one extent, the envelope
    of the runs touching the segment: on the 400 ms row frames 2 .. 36 (the context-only run dropped,
    the 5-frame run kept, the gap included); on the 25-frame 50 ms row frames 2 .. 29, past the
    segment's stop, because the 32 .. 36 run lies wholly in the context after it. usv rows get none
    although their tracks hold the same runs.
    A 4 ms segment is still scored: its window holds 49 context hops either side.
    """
    root = _build_session(tmp_path)
    summary = pls.read_csv(root / "audio" / f"{SESSION_ID}_usv_summary.csv", schema_overrides={"usv_id": pls.String})
    messages = []
    scores = squeaks.classify_usv_squeak_rows(root, summary, _stub_bundle(), torch.device("cpu"), True, 64, messages.append)
    assert scores.columns == list(squeaks.VOCAL_CLASS_COLUMNS)
    assert scores.schema["usv"] == pls.Boolean
    assert scores.schema["squeak"] == pls.Boolean
    assert scores["usv"].to_list() == [True, False, True, None, True]
    assert scores["squeak"].to_list() == [True, True, False, None, False]
    probabilities = scores.select("p_usv", "p_squeak", "p_both").to_numpy()
    assert np.allclose(probabilities[[0, 1, 2, 4]].sum(axis=1), 1.0)
    assert probabilities[0].argmax() == 2
    assert np.isnan(probabilities[3]).all()
    assert scores["p_usv"].null_count() == 1
    for row, start, last in ((0, 0.10, 29), (1, 0.30, 36)):
        first_centre = np.floor(start * SAMPLING_RATE) / SAMPLING_RATE
        assert scores["squeak_start"][row] == pytest.approx(first_centre + 1.5 * HOP_S, abs=2e-6)
        assert scores["squeak_end"][row] == pytest.approx(first_centre + (last + 0.5) * HOP_S, abs=2e-6)
    assert scores["squeak_end"][0] > 0.15
    assert scores["squeak_start"].is_null().to_list() == [False, False, True, True, True]
    assert scores["squeak_end"].is_null().to_list() == [False, False, True, True, True]
    assert any("0 of 2 squeak-bearing" in message for message in messages)


def test_classify_usv_squeak_rows_fallback_extent(tmp_path):
    """Squeak rows whose track has no frame above the threshold touching the segment take the
    highest-scoring segment frame (one frame, half-hop edges), and the step reports how many did."""
    root = _build_session(tmp_path)
    summary = pls.read_csv(root / "audio" / f"{SESSION_ID}_usv_summary.csv", schema_overrides={"usv_id": pls.String})
    messages = []
    scores = squeaks.classify_usv_squeak_rows(root, summary, _stub_bundle(fallback=True), torch.device("cpu"), True, 64, messages.append)
    for row, start in ((0, 0.10), (1, 0.30)):
        first_centre = np.floor(start * SAMPLING_RATE) / SAMPLING_RATE
        assert scores["squeak_start"][row] == pytest.approx(first_centre + 6.5 * HOP_S, abs=2e-6)
        assert scores["squeak_end"][row] == pytest.approx(first_centre + 7.5 * HOP_S, abs=2e-6)
    assert any("2 of 2 squeak-bearing" in message for message in messages)


def test_classify_usv_squeak_rows_needs_the_noise_column(tmp_path):
    """Without detect-usv-noise's column the step refuses to run."""
    root = _build_session(tmp_path)
    summary = pls.read_csv(root / "audio" / f"{SESSION_ID}_usv_summary.csv").drop("noise")
    with pytest.raises(ValueError, match="detect-usv-noise"):
        squeaks.classify_usv_squeak_rows(root, summary, _stub_bundle(), torch.device("cpu"), True, 64, print)


def test_squeak_audio_channels_honours_metadata_exclusion(tmp_path):
    """Metadata-excluded channels are dropped only when exclusion is switched on; each reader serves
    its own memmap column, scaled like soundfile's PCM_16 reading."""
    root = _build_session(tmp_path, excluded_channels=["m_ch02", "s_ch01"])
    kept = squeaks.squeak_audio_channels(root, exclude_metadata_audio_channels=True, message_output=lambda *_a, **_kw: None)
    assert [reader.name.split("_cropped")[0] for reader in kept] == ["m_250913193920_ch01", "s_250913193920_ch02"]
    everything = squeaks.squeak_audio_channels(root, exclude_metadata_audio_channels=False, message_output=lambda *_a, **_kw: None)
    assert len(everything) == 4
    rng = np.random.default_rng(0)
    first_column = rng.integers(-3000, 3000, size=SAMPLING_RATE, dtype=np.int16)
    everything[0].seek(100)
    np.testing.assert_array_equal(everything[0].read(frames=50), first_column[100:150] / 32768.0)
    assert everything[0].read(frames=10 * SAMPLING_RATE).size == SAMPLING_RATE - 150


def test_squeak_audio_channels_refuses_an_incomplete_record(tmp_path):
    """A broadband record that is incomplete, or a session without one, stops the read."""
    root = _build_session(tmp_path)
    record_path = root / "audio" / "broadband_filtered" / "line_noise.json"
    record = json.loads(record_path.read_text())
    record_path.write_text(json.dumps({**record, "complete": False}))
    with pytest.raises(ValueError, match="incomplete"):
        squeaks.squeak_audio_channels(root, False, print)
    record_path.write_text(json.dumps({k: v for k, v in record.items() if k != "low_band_variance"}))
    with pytest.raises(ValueError, match="add-broadband-low-band-variance"):
        squeaks.squeak_audio_channels(root, False, print)
    record_path.unlink()
    with pytest.raises(FileNotFoundError, match="line_noise.json"):
        squeaks.squeak_audio_channels(root, False, print)


def test_readers_carry_the_low_band_variance_and_the_window_input_weights_with_it(tmp_path):
    """Each reader carries its column's stored low-band variance, and the window input adds it to
    the window variance when weighting the channels: a dominant stored value makes the input equal
    that channel's single-channel input."""
    root = tmp_path / SESSION_ID
    rng = np.random.default_rng(5)
    channels = [(f"m_250913193920_ch{channel:02d}_cropped_to_video_hpss.wav", rng.integers(-3000, 3000, size=SAMPLING_RATE, dtype=np.int16))
                for channel in (1, 2, 3)]
    write_broadband_audio(root, channels, SAMPLING_RATE, low_band_variance=[0.0, 50.0, 0.0])
    readers = squeaks.squeak_audio_channels(root, False, lambda *_a, **_kw: None)
    assert [reader.low_band_variance for reader in readers] == [0.0, 50.0, 0.0]
    window = squeaks.usv_squeak_window_input(readers, readers[0].frames, 0.30, 0.40, squeaks.USV_SQUEAK_INPUT_CONTRACT)
    only_second = squeaks.usv_squeak_window_input([readers[1]], readers[1].frames, 0.30, 0.40, squeaks.USV_SQUEAK_INPUT_CONTRACT)
    np.testing.assert_allclose(window["x"][:2], only_second["x"][:2], atol=1e-4)
    for reader in readers:
        reader.low_band_variance = 0.0
    plain = squeaks.usv_squeak_window_input(readers, readers[0].frames, 0.30, 0.40, squeaks.USV_SQUEAK_INPUT_CONTRACT)
    assert np.abs(plain["x"][:2] - only_second["x"][:2]).max() > 1e-2


def test_detect_and_merge_replaces_the_retired_columns(tmp_path, mocker):
    """
    The merge removes the columns of earlier encodings (squeak_probability, squeak_frame_runs,
    call_class, squeak_spans, n_squeaks), replaces a stale squeak flag and extent, keeps every other
    column (and the usv_id zero-padding) unchanged, and leaves the summary in the canonical order: the
    DAS event, the noise block, the vocal-class block, then emitter and the DAS channel statistics, then
    the acoustic features.
    """
    root = _build_session(tmp_path)
    summary_path = root / "audio" / f"{SESSION_ID}_usv_summary.csv"
    original = pls.read_csv(summary_path, schema_overrides={"usv_id": pls.String})
    original.with_columns(
        pls.lit(True).alias("squeak"), pls.lit(0.9).alias("squeak_probability"), pls.lit(1.0).alias("squeak_start"),
        pls.lit(9).alias("squeak_frame_runs"), pls.lit("usv").alias("call_class"), pls.lit("[]").alias("squeak_spans"),
        pls.lit(0).alias("n_squeaks"), pls.lit(40000.0).alias("mean_freq_hz"),
    ).write_csv(summary_path)

    mocker.patch("usv_playpen.processing.detect_usv_squeaks.smart_wait")
    mocker.patch("usv_playpen.processing.detect_usv_squeaks.torch.cuda.is_available", return_value=False)
    mocker.patch("usv_playpen.processing.detect_usv_squeaks.load_usv_squeak_model", return_value=_stub_bundle())
    squeaks.USVSqueakDetector(
        root_directory=str(root),
        input_parameter_dict={"detect_usv_squeaks": {"squeak_model_path": "unused.pt", "exclude_metadata_audio_channels": True, "batch_size": 64}},
        message_output=lambda *_a, **_kw: None,
    ).detect_and_merge()

    written = pls.read_csv(summary_path, schema_overrides={"usv_id": pls.String})
    assert written.columns == [
        "usv_id", "start", "stop", "duration", "noise", *squeaks.VOCAL_CLASS_COLUMNS, "emitter", "chs_count", "mean_freq_hz",
    ]
    assert not set(squeaks.RETIRED_SQUEAK_COLUMNS) & set(written.columns)
    assert written.select(original.columns).equals(original)
    assert written["mean_freq_hz"].to_list() == [40000.0] * 5
    assert written["usv"].to_list() == [True, False, True, None, True]
    assert written["squeak"].to_list() == [True, True, False, None, False]
    assert written["squeak_start"].is_null().to_list() == [False, False, True, True, True]


def test_detect_usv_squeaks_cli_routes(mocker, tmp_path):
    """detect-usv-squeaks resolves settings and calls USVSqueakDetector.detect_and_merge once."""
    mock_cls = mocker.patch("usv_playpen.processing.detect_usv_squeaks.USVSqueakDetector")
    mocker.patch(
        "usv_playpen.processing.detect_usv_squeaks.modify_settings_json_for_cli",
        return_value={"detect_usv_squeaks": {}},
    )
    result = CliRunner().invoke(squeaks.detect_usv_squeaks_cli, ["--root-directory", str(tmp_path)])
    assert result.exit_code == 0, result.output
    mock_cls.assert_called_once()
    mock_cls.return_value.detect_and_merge.assert_called_once()


def _write_label_set(directory: pathlib.Path, session_root: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    """
    Description
    -----------
    Writes a labelling-tool label set on the synthetic session (rows 0, 1, 4 and the noise row 3 as an
    unsure answer) and a review CSV that turns panel P0 from usv into both.

    Parameters
    ----------
    directory (pathlib.Path)
        Directory to write into.
    session_root (pathlib.Path)
        The synthetic session.

    Returns
    -------
    paths (tuple[pathlib.Path, pathlib.Path, pathlib.Path])
        Labels CSV, sample CSV, review CSV.
    """

    directory.mkdir(parents=True, exist_ok=True)
    labels = directory / "labels.csv"
    sample = directory / "sample.csv"
    review = directory / "review.csv"
    pls.DataFrame({
        "panel_id": ["P0", "P1", "P4", "P3"],
        "label": [0, 1, 0, 3],
        "squeak_extents_s": ["[]", "[[0.35, 0.40]]", "[]", "[]"],
        "note": [None, None, None, None],
    }).write_csv(labels)
    pls.DataFrame({
        "panel_id": ["P0", "P1", "P3", "P4"],
        "session_id": [SESSION_ID] * 4,
        "session_dir": [str(session_root)] * 4,
        "row_index": [0, 1, 3, 4],
        "start": [0.10, 0.30, 0.85, 0.20],
        "stop": [0.15, 0.70, 0.90, 0.30],
    }).write_csv(sample)
    pls.DataFrame({"panel_id": ["P0"], "label": [2], "squeak_extents_s": ["[[0.09, 0.12]]"]}).write_csv(review)
    return labels, sample, review


def test_read_usv_squeak_labels_sets_and_overrides(tmp_path):
    """Sample ids carry the set name, a review overrides a panel's label and spans, unsure answers are
    dropped, and an override naming an unknown panel or an inconsistent label stops the read."""
    root = _build_session(tmp_path)
    labels, sample, review = _write_label_set(tmp_path / "labels", root)
    table = squeaks.read_usv_squeak_labels(
        [{"name": "r1", "labels_csv": str(labels), "sample_csv": str(sample)}],
        [{"name": "r1", "override_csv": str(review)}],
        lambda *_a, **_kw: None,
    )
    assert table["sample_id"].to_list() == ["r1_P0", "r1_P1", "r1_P4"]
    assert table["label"].to_list() == [2, 1, 0]
    assert table["reviewed"].to_list() == [True, False, False]
    assert json.loads(table["squeak_extents_s"][0]) == [[0.09, 0.12]]
    pls.DataFrame({"panel_id": ["P9"], "label": [1], "squeak_extents_s": ["[[1.0, 1.1]]"]}).write_csv(tmp_path / "bad_review.csv")
    with pytest.raises(ValueError, match="not in label set"):
        squeaks.read_usv_squeak_labels(
            [{"name": "r1", "labels_csv": str(labels), "sample_csv": str(sample)}],
            [{"name": "r1", "override_csv": str(tmp_path / "bad_review.csv")}],
            print,
        )
    pls.DataFrame({"panel_id": ["P0"], "label": [1], "squeak_extents_s": ["[]"]}).write_csv(tmp_path / "no_span.csv")
    with pytest.raises(ValueError, match="at least one"):
        squeaks.read_usv_squeak_labels(
            [{"name": "r1", "labels_csv": str(labels), "sample_csv": str(sample)}],
            [{"name": "r1", "override_csv": str(tmp_path / "no_span.csv")}],
            print,
        )


def test_usv_squeak_frame_targets_mark_frame_centres_inside_spans():
    """A frame is 1 when its centre lies inside a labelled span (closed), in the context too."""
    window = {"x": np.zeros((3, 128, 20), dtype=np.float32), "read_start_s": 1.0}
    target = squeaks.usv_squeak_frame_targets([window], [json.dumps([[1.0 + 2 * HOP_S, 1.0 + 4.5 * HOP_S], [1.0 + 10 * HOP_S, 1.0 + 10 * HOP_S]])])[0]
    assert np.flatnonzero(target).tolist() == [2, 3, 4, 10]


def test_train_usv_squeak_seed_from_a_noise_trunk():
    """One member trains on tiny inputs from a noise trunk and comes back in eval mode with the trunk
    moved by training."""
    rng = np.random.default_rng(0)
    inputs = []
    for n_frames in (20, 30, 25, 18):
        x = rng.uniform(-1, 1, size=(3, 128, n_frames)).astype(np.float32)
        x[2] = 0.0
        x[2, :, 5:12] = 1.0
        inputs.append(x)
    targets = [np.zeros(item.shape[2], dtype=np.float32) for item in inputs]
    targets[1][6:10] = 1.0
    trunk = NoiseTimeMIL(in_channels=3, n_scalars=2).state_dict()
    model = squeaks.train_usv_squeak_seed(
        inputs, np.zeros((4, 2), dtype=np.float32), np.array([0, 1, 2, 0]), targets, 3,
        {"epochs": 2, "batch_size": 2, "learning_rate": 1e-3, "weight_decay": 1e-4, "label_smoothing": 0.05,
         "frame_loss_weight": 1.0, "class_weighted": True},
        trunk, torch.device("cpu"), lambda *_a, **_kw: None,
    )
    assert not model.training
    assert not torch.equal(model.state_dict()["body.0.0.weight"], trunk["body.0.0.weight"])


def test_usv_squeak_model_trainer_writes_a_loadable_bundle(tmp_path, mocker):
    """The trainer reads the label set, builds the inputs with the detector's window, trains and writes
    a bundle the detector's loader accepts, carrying the extent threshold and the label provenance; an existing
    bundle path is refused."""
    root = _build_session(tmp_path)
    labels, sample, review = _write_label_set(tmp_path / "labels", root)
    settings = {
        "detect_usv_noise": {"noise_model_path": ""},
        "train_usv_squeak_model": {
            "label_sets": [{"name": "r1", "labels_csv": str(labels), "sample_csv": str(sample)}],
            "label_overrides": [{"name": "r1", "override_csv": str(review)}],
            "pretrained": False, "span_threshold": 0.6,
            "exclude_metadata_audio_channels": True, "n_workers": 2, "seeds": [0, 1],
            "epochs": 1, "batch_size": 2, "learning_rate": 1e-3, "weight_decay": 1e-4,
            "label_smoothing": 0.05, "frame_loss_weight": 1.0, "class_weighted": False,
        },
    }
    mocker.patch("usv_playpen.processing.detect_usv_squeaks.torch.cuda.is_available", return_value=False)
    bundle_path = squeaks.USVSqueakModelTrainer(str(tmp_path / "bundle.pt"), settings, lambda *_a, **_kw: None).train()
    raw = torch.load(bundle_path, weights_only=True)
    assert raw["n_labels"] == {"usv": 1, "squeak": 1, "both": 1}
    assert raw["extent_rule"]["threshold"] == 0.6
    assert raw["label_overrides"] == [["r1", str(review)]]
    assert raw["max_frames"] == 1024
    assert len(squeaks.load_usv_squeak_model(str(bundle_path), torch.device("cpu"))["models"]) == 2
    with pytest.raises(FileExistsError):
        squeaks.USVSqueakModelTrainer(str(bundle_path), settings, print).train()


def test_train_usv_squeak_model_cli_routes_label_sets(mocker, tmp_path):
    """train-usv-squeak-model turns repeated --label-set / --label-override into the settings lists and
    calls the trainer once."""
    mock_cls = mocker.patch("usv_playpen.processing.detect_usv_squeaks.USVSqueakModelTrainer")
    settings = {"train_usv_squeak_model": {"label_sets": [], "label_overrides": []}}
    mocker.patch("usv_playpen.processing.detect_usv_squeaks.modify_settings_json_for_cli", return_value=settings)
    result = CliRunner().invoke(squeaks.train_usv_squeak_model_cli, [
        "--bundle-path", str(tmp_path / "b.pt"),
        "--label-set", "r1", "l1.csv", "s1.csv", "--label-set", "r2", "l2.csv", "s2.csv",
        "--label-override", "r1", "rev.csv",
    ])
    assert result.exit_code == 0, result.output
    assert settings["train_usv_squeak_model"]["label_sets"] == [
        {"name": "r1", "labels_csv": "l1.csv", "sample_csv": "s1.csv"},
        {"name": "r2", "labels_csv": "l2.csv", "sample_csv": "s2.csv"},
    ]
    assert settings["train_usv_squeak_model"]["label_overrides"] == [{"name": "r1", "override_csv": "rev.csv"}]
    mock_cls.return_value.train.assert_called_once()


def _build_squeak_embedding_session(tmp_path: pathlib.Path) -> pathlib.Path:
    """
    Description
    -----------
    Creates a synthetic session for the squeak QLVM embedding: a broadband memmap and a summary whose rows
    cover every selection and crop case (``dt`` the frame hop):

    * row 0 -- squeak, envelope ``start + 4.5 dt .. start + 20.5 dt`` of a 100 ms segment: the window
      is the segment, extent frames 5 .. 20, crop 3 .. 22 (embedded);
    * row 1 -- a noise row (no call class; not a candidate);
    * row 2 -- both, a squeak starting 10 frames BEFORE its segment: the window starts 12 frames before
      the segment, extent frames 2 .. 42, crop 0 .. 44 (embedded);
    * row 3 -- squeak, extent frames 10 .. 180 of a 400 ms segment (crop of 175 frames, too wide);
    * row 4 -- both, an envelope one frame wide (crop of 5 frames, too narrow);
    * row 5 -- squeak on a 1 ms segment at the very start of the recording with a 0.5 ms envelope (the
      window cannot extend before 0 s and stays shorter than one STFT window: no spectrogram);
    * row 6 -- usv (not a candidate);
    * row 7 -- both without an extent (only a hand-edited summary has one: no squeak extent).

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Pytest temporary directory.

    Returns
    -------
    root (pathlib.Path)
        Session root directory.
    """

    root = tmp_path / SESSION_ID
    _write_wavs(root)
    dt = squeaks.FRAME_DT_S
    pls.DataFrame(
        {
            "usv_id": [f"{i:04d}" for i in range(8)],
            "start": [0.10, 0.20, 0.40, 0.30, 0.75, 0.0, 0.85, 0.05],
            "stop": [0.20, 0.25, 0.60, 0.70, 0.78, 0.001, 0.90, 0.08],
            "noise": [False, True, None, False, False, False, False, False],
            "usv": [False, None, True, False, True, False, True, True],
            "squeak": [True, None, True, True, True, True, False, True],
            "squeak_start": [0.10 + 4.5 * dt, None, 0.40 - 10 * dt, 0.30 + 10 * dt, 0.75 + 5 * dt, 0.0, None, None],
            "squeak_end": [0.10 + 20.5 * dt, None, 0.40 + 30 * dt, 0.30 + 180 * dt, 0.75 + 5 * dt, 0.0005, None, None],
            "mean_freq_hz": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        },
        schema={
            "usv_id": pls.String, "start": pls.Float64, "stop": pls.Float64, "noise": pls.Boolean,
            "usv": pls.Boolean, "squeak": pls.Boolean, "squeak_start": pls.Float64, "squeak_end": pls.Float64, "mean_freq_hz": pls.Float64,
        },
    ).write_csv(root / "audio" / f"{SESSION_ID}_usv_summary.csv")
    return root


def test_squeak_crop_window_widens_the_segment_only_when_needed():
    """The window is the segment when the envelope and its two context frames lie inside it, and grows
    to hold them otherwise (never before 0 s)."""
    dt = squeaks.FRAME_DT_S
    start, stop = squeaks.squeak_crop_window(
        np.array([1.0, 1.0, 1.0, 0.0]), np.array([1.2, 1.2, 1.2, 0.1]),
        np.array([1.05, 0.95, 1.05, 0.0]), np.array([1.15, 1.10, 1.25, 0.05]),
    )
    assert np.allclose(start, [1.0, 0.95 - 2 * dt, 1.0, 0.0])
    assert np.allclose(stop, [1.2, 1.2, 1.25 + 2 * dt, 0.1])


def test_squeak_crop_frames_take_frame_centres_inside_the_extent_plus_context():
    """Extent frames are the frame centres inside [squeak_start, squeak_end]; two context frames are
    added and clipped to 0 .. n_frames - 1."""
    dt = squeaks.FRAME_DT_S
    window_start = np.array([10.0, 10.0, 10.0])
    first, last = squeaks.squeak_crop_frames(
        window_start_s=window_start,
        squeak_start_s=window_start + np.array([0.5, 5.0, 39.5]) * dt,
        squeak_end_s=window_start + np.array([30.5, 60.0, 199.0]) * dt,
        n_frames=np.array([100, 61, 200]),
    )
    assert first.tolist() == [0, 3, 38]
    assert last.tolist() == [32, 60, 199]
    assert first.dtype == np.int64
    assert last.dtype == np.int64


def test_squeak_crop_inputs_normalizes_per_crop_and_centres_the_crop():
    """
    The crop is min-max normalized on its own (the dB values outside it play no part), centred in the
    128-frame frame with (128 - width) // 2 zero columns on the left, and min-max normalized again; a
    crop wider than 128 frames raises.
    """
    rng = np.random.default_rng(3)
    spectrogram = rng.uniform(-90.0, 10.0, size=(128, 60)).astype(np.float32)
    spectrogram[:, :5] = 500.0
    inputs = squeaks.squeak_crop_inputs([spectrogram], np.array([10]), np.array([29]), False, squeaks.SQUEAK_QLVM_INPUT_CONTRACT)
    assert inputs.shape == (1, 128, 128)
    assert inputs.dtype == np.float32
    crop = spectrogram[:, 10:30]
    low, high = float(crop.min()), float(crop.max())
    once = (crop - low) / (high - low + 1e-6)
    expected = (once - once.min()) / (once.max() - once.min() + np.float32(1e-8))
    left = (128 - 20) // 2
    assert np.allclose(inputs[0, :, left:left + 20], expected, atol=1e-6)
    assert not inputs[0, :, :left].any()
    assert not inputs[0, :, left + 20:].any()
    with pytest.raises(ValueError, match="1 to 128 frames"):
        squeaks.squeak_crop_inputs(
            [np.zeros((128, 200), dtype=np.float32)], np.array([0]), np.array([150]), False, squeaks.SQUEAK_QLVM_INPUT_CONTRACT
        )


def test_squeak_crop_inputs_time_stretches_and_floors_as_the_contract_says():
    """
    With time_stretch true the per-crop normalized crop is stretched over all 128 frames exactly as the
    training-set builder stretched it (stretch_specs(..., True) of the zero-padded frame, no zero
    columns left), then min-max normalized; a contract floor is applied after the resize and the input
    min-maxed again (qlvm_latents.normalize_model_inputs), as the floored training sets were.
    """
    rng = np.random.default_rng(5)
    spectrogram = rng.uniform(-90.0, 10.0, size=(128, 60)).astype(np.float32)
    contract = {"input_normalization": "minmax", "normalization_epsilon": 1e-8, "floor": None}
    inputs = squeaks.squeak_crop_inputs([spectrogram], np.array([10]), np.array([29]), True, contract)
    crop = spectrogram[:, 10:30]
    low, high = float(crop.min()), float(crop.max())
    padded = np.zeros((1, 128, 128), dtype=np.float32)
    padded[0, :, :20] = (crop - low) / (high - low + 1e-6)
    stretched = squeaks.stretch_specs(padded, np.array([20]), (128, 128), True)
    expected = squeaks.normalize_model_inputs(stretched, contract)
    assert np.array_equal(inputs, expected)
    assert inputs[0, :, 0].any()
    assert inputs[0, :, -1].any()
    floored = squeaks.squeak_crop_inputs([spectrogram], np.array([10]), np.array([29]), True, {**contract, "floor": 0.2})
    once = (expected - expected.min()) / np.float32(expected.max() - expected.min() + np.float32(1e-8))
    clipped = np.clip((once - np.float32(0.2)) / np.float32(0.8), 0.0, 1.0)
    assert np.allclose(floored, (clipped - clipped.min()) / (clipped.max() - clipped.min() + np.float32(1e-8)), atol=1e-6)
    assert not np.array_equal(floored, inputs)


def test_squeak_qlvm_rows_selects_squeak_and_both_that_are_not_noise():
    """squeak and both rows with noise false or null are selected, never usv or unclassed rows; a
    summary without noise or the usv / squeak booleans raises."""
    summary = pls.DataFrame(
        {"noise": [False, True, None, False, False, False],
         "usv": [False, False, True, True, None, True], "squeak": [True, True, True, False, None, True]},
        schema={"noise": pls.Boolean, "usv": pls.Boolean, "squeak": pls.Boolean},
    ).with_columns(pls.lit(None, dtype=pls.Float64).alias(column) for column in ("squeak_start", "squeak_end"))
    assert squeaks.squeak_qlvm_rows(summary).tolist() == [0, 2, 5]
    with pytest.raises(ValueError, match="detect-usv-noise"):
        squeaks.squeak_qlvm_rows(summary.drop("noise"))
    with pytest.raises(ValueError, match="usv"):
        squeaks.squeak_qlvm_rows(summary.drop("usv"))


def test_squeak_qlvm_inputs_crops_and_excludes(tmp_path):
    """Two segments are embedded (one whose squeak starts before the segment, cropped whole from the
    widened window); the other candidates are counted by reason."""
    root = _build_squeak_embedding_session(tmp_path)
    summary = pls.read_csv(root / "audio" / f"{SESSION_ID}_usv_summary.csv", schema_overrides={"usv_id": pls.String})
    built = squeaks.squeak_qlvm_inputs(root, summary, True, lambda *_a, **_kw: None, False, squeaks.SQUEAK_QLVM_INPUT_CONTRACT)
    assert built["row_index"].tolist() == [0, 2]
    assert built["window_start_s"] == pytest.approx([0.10, 0.40 - 12 * squeaks.FRAME_DT_S])
    assert built["first"].tolist() == [3, 0]
    assert built["last"].tolist() == [22, 44]
    assert built["inputs"].shape == (2, 128, 128)
    assert built["n_candidates"] == 6
    assert built["excluded"] == {"no squeak extent": 1, "no spectrogram": 1, "crop < 8 frames": 1, "crop > 128 frames": 1}
    spectrograms = squeaks.squeak_window_spectrograms(root, np.array([0.10]), np.array([0.20]), True, lambda *_a, **_kw: None)
    assert np.array_equal(
        built["inputs"][:1],
        squeaks.squeak_crop_inputs(spectrograms, np.array([3]), np.array([22]), False, squeaks.SQUEAK_QLVM_INPUT_CONTRACT),
    )
    stretched = squeaks.squeak_qlvm_inputs(root, summary, True, lambda *_a, **_kw: None, True, squeaks.SQUEAK_QLVM_INPUT_CONTRACT)
    assert stretched["row_index"].tolist() == [0, 2]
    assert np.array_equal(
        stretched["inputs"][:1],
        squeaks.squeak_crop_inputs(spectrograms, np.array([3]), np.array([22]), True, squeaks.SQUEAK_QLVM_INPUT_CONTRACT),
    )


def _write_squeak_cell(cell: pathlib.Path, mask_tag: str = "nomask", masking_type: str = "none") -> None:
    """
    Description
    -----------
    Writes a minimal phase 3 squeak cell in the old package layout (every file at the cell root): a
    torch zip checkpoint holding a tiny unconditional legacy-head decoder prefix, ``run_config.json``
    and ``manifest.json``.

    Parameters
    ----------
    cell (pathlib.Path)
        Cell directory to create.
    mask_tag (str)
        ``run_config.json`` ``mask_tag``.
    masking_type (str)
        ``manifest.json`` ``dataset.masking_type``.

    Returns
    -------
    None
    """

    cell.mkdir(parents=True)
    torch.save(
        {"model": {"decoder.0.weight": torch.zeros(8, 4), "decoder.1.weight": torch.zeros(2, 8)}},
        cell / "checkpoint.tar",
    )
    (cell / "run_config.json").write_text(
        '{"phase": "phase3_BBVs_masked", "mask_tag": "' + mask_tag + '", "dataset": "bbv-natural_dur-26-40-62_lumped_N11000_seed42", "latent_dim": 2, "seed": 42}'
    )
    (cell / "manifest.json").write_text(
        '{"dataset": {"masking_type": "' + masking_type + '", "apply_mask": false, "target_shape": [128, 128], '
        '"time_stretch": false}, "analysis": {"lattice_m": 8}}'
    )


def test_load_squeak_qlvm_cell_reads_the_old_layout_and_checks_it(tmp_path):
    """The old-layout cell loads with its float32 lattice; a masked cell is refused with every problem listed."""
    _write_squeak_cell(tmp_path / "phase3_BBVs_qlvm" / "natural_lumped_N11000_nomask")
    model = squeaks.load_squeak_qlvm_cell(str(tmp_path / "phase3_BBVs_qlvm" / "natural_lumped_N11000_nomask"))
    assert model["model_id"] == "phase3_BBVs_qlvm/natural_lumped_N11000_nomask"
    assert model["lattice_m"] == 8
    assert model["head"] == "legacy"
    assert np.array_equal(np.asarray(model["lattice"]), np.asarray(squeaks.gen_fib_basis_float32(8)))
    assert model["layout"] == "phase 3"
    assert model["time_stretch"] is False
    assert model["input_contract"] == squeaks.SQUEAK_QLVM_INPUT_CONTRACT
    _write_squeak_cell(tmp_path / "bad", mask_tag="masked", masking_type="sam")
    with pytest.raises(ValueError, match="mask_tag") as error:
        squeaks.load_squeak_qlvm_cell(str(tmp_path / "bad"))
    assert "masking_type" in str(error.value)


def _write_contract_squeak_cell(cell: pathlib.Path, **contract_overrides) -> None:
    """
    Description
    -----------
    Writes a minimal squeak cell in the train-qlvm layout: a torch zip checkpoint holding a tiny
    unconditional ReLU-head decoder prefix (layers 0 and 2) and ``config/training_contract.json``
    with the production stretch_nofloor contract, any field replaced by ``contract_overrides``.

    Parameters
    ----------
    cell (pathlib.Path)
        Cell directory to create.
    contract_overrides (dict)
        Contract fields to replace.

    Returns
    -------
    None
    """

    (cell / "config").mkdir(parents=True)
    torch.save(
        {"model": {"decoder.0.weight": torch.zeros(8, 4), "decoder.2.weight": torch.zeros(2, 8)}},
        cell / "checkpoint.tar",
    )
    contract = {
        "decoder_head": "relu", "latent_dim": 2, "c_dim": 0, "conditional": None,
        "input_normalization": "minmax", "normalization_epsilon": 1e-08, "masking_type": "none", "floor": None,
        "target_shape": [128, 128], "time_stretch": True, "length_threshold": None, "require_mask": False,
        "embedding_lattice_type": "fibonacci", "embedding_fib_m": 8, "training_lattice_type": "fibonacci",
        "training_fib_m": 5, "validation_fib_m": 6, "condition": None,
    }
    contract.update(contract_overrides)
    (cell / "config" / "training_contract.json").write_text(json.dumps(contract))


def test_load_squeak_qlvm_cell_reads_the_contract_layout_and_checks_it(tmp_path):
    """A train-qlvm cell is read from its training_contract.json: time stretch, floor and input
    normalization come from the contract and the lattice from embedding_fib_m; a masked, conditional,
    mis-headed or badly floored contract is refused with every problem listed."""
    _write_contract_squeak_cell(tmp_path / "squeaks" / "cell" / "stretch_nofloor")
    model = squeaks.load_squeak_qlvm_cell(str(tmp_path / "squeaks" / "cell" / "stretch_nofloor"))
    assert model["layout"] == "train-qlvm"
    assert model["model_id"] == "squeaks/cell/stretch_nofloor"
    assert model["head"] == "relu"
    assert model["time_stretch"] is True
    assert model["input_contract"] == {"input_normalization": "minmax", "normalization_epsilon": 1e-08, "floor": None}
    assert model["lattice_m"] == 8
    assert np.array_equal(np.asarray(model["lattice"]), np.asarray(squeaks.gen_fib_basis_float32(8)))
    _write_contract_squeak_cell(tmp_path / "padded_floor", time_stretch=False, floor=0.2)
    padded = squeaks.load_squeak_qlvm_cell(str(tmp_path / "padded_floor"))
    assert padded["time_stretch"] is False
    assert padded["input_contract"]["floor"] == 0.2
    _write_contract_squeak_cell(
        tmp_path / "bad", masking_type="sam", c_dim=1, condition={"name": "duration"}, decoder_head="legacy", floor=1.5,
    )
    with pytest.raises(ValueError, match="masking_type") as error:
        squeaks.load_squeak_qlvm_cell(str(tmp_path / "bad"))
    for problem in ("c_dim", "decoder_head", "floor"):
        assert problem in str(error.value)


def test_embed_and_merge_writes_the_two_squeak_torus_columns(tmp_path, mocker):
    """
    The embedded segments get their coordinates, every other row nulls, stale squeak coordinates are
    replaced, the other columns are kept and the two columns follow the canonical order (after every
    other known column).
    """
    root = _build_squeak_embedding_session(tmp_path)
    summary_path = root / "audio" / f"{SESSION_ID}_usv_summary.csv"
    pls.read_csv(summary_path, schema_overrides={"usv_id": pls.String}).with_columns(
        pls.lit(0.5).alias("qlvm_squeak1"), pls.lit(0.5).alias("qlvm_squeak2")
    ).write_csv(summary_path)
    mocker.patch("usv_playpen.processing.detect_usv_squeaks.smart_wait")
    mocker.patch(
        "usv_playpen.processing.detect_usv_squeaks.load_squeak_qlvm_cell",
        return_value={
            "params": {}, "lattice": np.zeros((21, 2)), "lattice_m": 8, "head": "legacy", "model_id": "p/c",
            "layout": "phase 3", "time_stretch": False, "input_contract": dict(squeaks.SQUEAK_QLVM_INPUT_CONTRACT),
        },
    )
    embed = mocker.patch(
        "usv_playpen.processing.detect_usv_squeaks.embed_data",
        return_value=jnp.asarray(np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)),
    )
    squeaks.USVSqueakQLVMEmbedder(
        root_directory=str(root),
        input_parameter_dict={"infer_qlvm_squeak_latents": {
            "model_cell_directory": "cell", "exclude_metadata_audio_channels": True,
            "lattice_batch_size": 7, "data_batch_size": 3,
        }},
        message_output=lambda *_a, **_kw: None,
    ).embed_and_merge()

    assert embed.call_args.args[1].shape == (2, 1, 128, 128)
    assert embed.call_args.args[3:] == (7, 3)
    written = pls.read_csv(summary_path, schema_overrides={"usv_id": pls.String})
    assert written.columns[-2:] == list(squeaks.SQUEAK_QLVM_COLUMNS)
    assert written["mean_freq_hz"].to_list() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    assert written["qlvm_squeak1"].is_null().to_list() == [False, True, False, True, True, True, True, True]
    assert written["qlvm_squeak1"][0] == pytest.approx(0.1, abs=1e-6)
    assert written["qlvm_squeak2"][2] == pytest.approx(0.4, abs=1e-6)


def test_embed_and_merge_refuses_an_empty_model_cell(tmp_path, mocker):
    """With no spectrograms_root to derive the production squeak cell from, an empty
    model_cell_directory stops the run before anything is read."""
    mocker.patch("usv_playpen.processing.detect_usv_squeaks.smart_wait")
    embedder = squeaks.USVSqueakQLVMEmbedder(
        root_directory=str(tmp_path),
        input_parameter_dict={"infer_qlvm_squeak_latents": {
            "model_cell_directory": "", "exclude_metadata_audio_channels": True,
            "lattice_batch_size": 4096, "data_batch_size": 1024,
        }},
        message_output=lambda *_a, **_kw: None,
    )
    with pytest.raises(ValueError, match="production squeak QLVM cell is not filled in"):
        embedder.embed_and_merge()


def test_infer_qlvm_squeak_latents_cli_routes(mocker, tmp_path):
    """infer-qlvm-squeak-latents resolves its settings block and calls USVSqueakQLVMEmbedder.embed_and_merge once."""
    mock_cls = mocker.patch("usv_playpen.processing.detect_usv_squeaks.USVSqueakQLVMEmbedder")
    modify = mocker.patch(
        "usv_playpen.processing.detect_usv_squeaks.modify_settings_json_for_cli",
        return_value={"infer_qlvm_squeak_latents": {}},
    )
    result = CliRunner().invoke(
        squeaks.infer_qlvm_squeak_latents_cli,
        ["--root-directory", str(tmp_path), "--model-cell-directory", "some/cell"],
    )
    assert result.exit_code == 0, result.output
    assert modify.call_args.kwargs["block"] == "infer_qlvm_squeak_latents"
    assert modify.call_args.kwargs["provided_params"] == ["model_cell_directory"]
    mock_cls.return_value.embed_and_merge.assert_called_once()
