"""
@author: bartulem
Tests for processing/detect_usv_squeaks.

The input transform, the 128-frame window, the onset/offset conversion and the
checkpoint contract are tested directly. The end-to-end scorer runs on a
synthetic session (per-channel PCM_16 wavs under ``audio/hpss`` plus a small
``*_usv_summary.csv``) with a TimeMIL whose frame head is forced to a constant
logit, so every valid frame and every scorable segment has a known probability:
that pins the squeak call, the probability, the full-length onset/offset of a
segment longer than the model window, the unscorable too-short row and the
metadata channel exclusion without needing the real checkpoint. The frame-run
count (``squeak_frame_runs``, the reference squeak index's ``n_bouts_min3``) is
pinned with a stub model whose frame logits are set per frame index, which shows
that runs are counted on the 128-frame window only and need 3 frames.

The squeak QLVM embedding is tested on the same kind of synthetic session: the
crop frames and their context, the per-crop and model-input normalization and
centring, the row selection (squeaks that are not noise) and exclusions, the
old-layout cell loader and its checks, and the merge of ``qlvm_squeak1`` /
``qlvm_squeak2`` with the decoder mocked.
"""

from __future__ import annotations

import pathlib

import jax.numpy as jnp
import numpy as np
import polars as pls
import pytest
import soundfile as sf
import torch
import yaml
from click.testing import CliRunner

from usv_playpen.processing import detect_usv_squeaks as squeaks

SESSION_ID = "20250913_193920"
SAMPLING_RATE = 250000


def _forced_model(frame_logit: float) -> squeaks.TimeMIL:
    """
    Description
    -----------
    Builds a TimeMIL in eval mode whose frame head outputs the same logit at
    every frame, so the attention-weighted segment logit equals it too.

    Parameters
    ----------
    frame_logit (float)
        The constant frame (and therefore segment) logit.

    Returns
    -------
    model (squeaks.TimeMIL)
        Model with a zeroed frame-head weight and the given bias.
    """

    model = squeaks.TimeMIL()
    with torch.no_grad():
        model.frame.weight.zero_()
        model.frame.bias.fill_(frame_logit)
    model.eval()
    return model


class _FramePatternModel(torch.nn.Module):
    """
    Description
    -----------
    Stand-in for TimeMIL whose frame logit is +5 at the frame indices given and
    -5 elsewhere, whatever the input, and whose segment logit is always +5, so
    every scorable segment is a squeak and its per-frame pattern is known.
    """

    def __init__(self, above_frames: list[int]) -> None:
        """
        Description
        -----------
        Stores the frame indices whose logit is +5.

        Parameters
        ----------
        above_frames (list[int])
            Frame indices (from the start of the input) that reach the threshold.

        Returns
        -------
        None
        """

        super().__init__()
        self.above_frames = above_frames

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # noqa: ARG002
        """
        Description
        -----------
        Returns the fixed segment logit and the per-frame logit pattern.

        Parameters
        ----------
        x (torch.Tensor)
            Input batch, shape ``(B, 1, 128, T)``.
        mask (torch.Tensor)
            Validity mask, shape ``(B, T)`` (unused).

        Returns
        -------
        segment_logit (torch.Tensor)
            ``(B,)`` all +5.
        frame_logit (torch.Tensor)
            ``(B, T)``: +5 at ``above_frames`` (those below ``T``), -5 elsewhere.
        """

        n_batch, n_time = x.shape[0], x.shape[-1]
        frame_logit = torch.full((n_batch, n_time), -5.0)
        frame_logit[:, [frame for frame in self.above_frames if frame < n_time]] = 5.0
        return torch.full((n_batch,), 5.0), frame_logit


def _build_session(tmp_path: pathlib.Path, excluded_channels: list[str] | None = None) -> pathlib.Path:
    """
    Description
    -----------
    Creates a synthetic session: four PCM_16 HPSS wavs (two master, two slave
    channels) of noise, and a summary with a normal segment (50 ms), a segment
    longer than the 128-frame window (400 ms) and one too short for a single
    STFT frame (4 ms). Optionally writes session metadata excluding channels.

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
    for device, channel in (("m", 1), ("m", 2), ("s", 1), ("s", 2)):
        audio = rng.integers(-3000, 3000, size=SAMPLING_RATE, dtype=np.int16)
        sf.write(
            str(hpss_dir / f"{device}_250913193920_ch{channel:02d}_cropped_to_video_hpss.wav"),
            audio,
            SAMPLING_RATE,
            subtype="PCM_16",
        )
    pls.DataFrame(
        {
            "usv_id": ["0000", "0001", "0002"],
            "start": [0.10, 0.30, 0.80],
            "stop": [0.15, 0.70, 0.804],
            "duration": [0.05, 0.40, 0.004],
            "emitter": [None, None, None],
        },
        schema={"usv_id": pls.String, "start": pls.Float64, "stop": pls.Float64, "duration": pls.Float64, "emitter": pls.String},
    ).write_csv(root / "audio" / f"{SESSION_ID}_usv_summary.csv")
    if excluded_channels is not None:
        metadata = {"Equipment": {"audio_Avisoft": {"excluded_channels": excluded_channels}}}
        (root / f"{SESSION_ID}_metadata.yaml").write_text(yaml.dump(metadata))
    return root


def test_normalize_absolute_db_is_the_fixed_affine_map():
    """-100 dB maps to -1, +50 dB to +1, -25 dB to 0, and values beyond are clipped."""
    normalized = squeaks.normalize_absolute_db(np.array([-150.0, -100.0, -25.0, 50.0, 80.0]))
    assert normalized.dtype == np.float32
    np.testing.assert_allclose(normalized, [-1.0, -1.0, 0.0, 1.0, 1.0])


def test_model_window_pads_short_and_truncates_long():
    """A short segment is padded at -100 dB with padding masked out; a long one keeps its first 128 frames."""
    short = np.full((128, 40), -30.0, dtype=np.float32)
    window, valid = squeaks.model_window(short)
    assert window.shape == (128, 128)
    assert np.all(window[:, :40] == -30.0)
    assert np.all(window[:, 40:] == squeaks.PAD_DB)
    assert valid.sum() == 40
    assert valid[:40].all()

    long = np.arange(128 * 200, dtype=np.float32).reshape(128, 200)
    window, valid = squeaks.model_window(long)
    np.testing.assert_array_equal(window, long[:, :128])
    assert valid.all()


def test_squeak_extent_seconds_uses_frame_centres_on_the_session_clock():
    """Onset / offset are the centres of the first and last above-threshold frames, in session seconds."""
    probability = np.array([0.1, 0.2, 0.5, 0.9, 0.4, 0.1])
    start, end = squeaks.squeak_extent_seconds(probability, segment_start_s=12.0, threshold=0.385)
    assert start == pytest.approx(12.0 + 2 * squeaks.FRAME_DT_S)
    assert end == pytest.approx(12.0 + 4 * squeaks.FRAME_DT_S)
    assert squeaks.squeak_extent_seconds(np.array([0.1, 0.2]), 12.0, 0.385) == (None, None)


def test_squeak_frame_run_count_counts_runs_of_at_least_three_frames():
    """Maximal runs of p >= threshold are counted when they span >= 3 frames (the
    reference n_bouts_min3); a threshold hit is inclusive and shorter runs are ignored."""
    threshold = 0.385
    probability = np.array([0.9, 0.9, 0.1, 0.385, 0.5, 0.6, 0.1, 0.9, 0.9, 0.9, 0.9, 0.2, 0.9])
    assert squeaks.squeak_frame_run_count(probability, threshold) == 2
    assert squeaks.squeak_frame_run_count(probability, threshold, min_run_frames=1) == 4
    assert squeaks.squeak_frame_run_count(probability, threshold, min_run_frames=4) == 1
    assert squeaks.squeak_frame_run_count(np.full(128, 0.9), threshold) == 1
    assert squeaks.squeak_frame_run_count(np.array([0.9, 0.9]), threshold) == 0
    assert squeaks.squeak_frame_run_count(np.empty(0), threshold) == 0


def test_timemil_is_fully_convolutional_along_time():
    """The network accepts the 128-frame window and any longer segment, returning one logit per frame."""
    model = squeaks.TimeMIL().eval()
    for n_frames in (128, 300):
        with torch.no_grad():
            segment_logit, frame_logit = model(torch.zeros(2, 1, 128, n_frames), torch.ones(2, n_frames, dtype=torch.bool))
        assert segment_logit.shape == (2,)
        assert frame_logit.shape == (2, n_frames)


def test_load_squeak_model_enforces_the_input_contract(tmp_path):
    """A checkpoint recorded under a different normalization or mask rule is refused; a matching one loads."""
    norm = {"kind": squeaks.CHECKPOINT_NORM_KIND, "db_floor": -100.0, "db_ceil": 50.0, "center": -25.0, "half": 75.0}
    base = {"state_dict": squeaks.TimeMIL().state_dict(), "ch": 32, "pool": "attn", "mask": squeaks.CHECKPOINT_MASK_RULE}

    good_path = tmp_path / "good.pt"
    torch.save({**base, "norm": norm}, good_path)
    model = squeaks.load_squeak_model(str(good_path), torch.device("cpu"))
    assert not model.training

    bad_norm_path = tmp_path / "bad_norm.pt"
    torch.save({**base, "norm": {**norm, "center": -20.0}}, bad_norm_path)
    with pytest.raises(ValueError, match="normalization"):
        squeaks.load_squeak_model(str(bad_norm_path), torch.device("cpu"))

    bad_mask_path = tmp_path / "bad_mask.pt"
    torch.save({**base, "norm": norm, "mask": "arange(128) < n_frames"}, bad_mask_path)
    with pytest.raises(ValueError, match="mask rule"):
        squeaks.load_squeak_model(str(bad_mask_path), torch.device("cpu"))


def test_score_squeak_rows_calls_times_and_unscorable_rows(tmp_path):
    """
    With every frame forced to p = sigmoid(5): scorable segments are squeaks
    with that probability; the 400 ms segment's offset comes from the full-length
    pass (past frame 127); the 4 ms segment is unscorable and not a squeak.
    """
    root = _build_session(tmp_path)
    summary = pls.read_csv(root / "audio" / f"{SESSION_ID}_usv_summary.csv", schema_overrides={"usv_id": pls.String})
    scores = squeaks.score_squeak_rows(
        session_root=root,
        usv_summary=summary,
        row_indices=np.arange(summary.height),
        model=_forced_model(5.0),
        device=torch.device("cpu"),
        threshold=0.385,
        exclude_metadata_audio_channels=True,
        batch_size=2,
        message_output=lambda *_a, **_kw: None,
    )
    expected_p = float(torch.sigmoid(torch.tensor(5.0)))

    assert scores["squeak"].to_list() == [True, True, False]
    assert scores["squeak_probability"][0] == pytest.approx(expected_p, abs=1e-6)
    assert scores["squeak_probability"][1] == pytest.approx(expected_p, abs=1e-6)
    assert scores["squeak_probability"][2] is None

    long_frames = int(scores["n_frames"][1])
    assert long_frames > squeaks.MODEL_WINDOW_FRAMES
    assert scores["squeak_start"][1] == pytest.approx(0.30)
    assert scores["squeak_end"][1] == pytest.approx(0.30 + (long_frames - 1) * squeaks.FRAME_DT_S)
    assert scores["squeak_end"][1] > 0.30 + (squeaks.MODEL_WINDOW_FRAMES - 1) * squeaks.FRAME_DT_S

    assert scores["n_frames"][2] == 0
    assert np.isnan(scores["raw_probability"][2])
    assert scores["squeak_start"][2] is None
    assert scores["squeak_end"][2] is None

    assert scores[squeaks.SQUEAK_FRAME_RUNS_COLUMN].dtype == pls.Int64
    assert scores[squeaks.SQUEAK_FRAME_RUNS_COLUMN].to_list() == [1, 1, 0]


def test_score_squeak_rows_counts_frame_runs_on_the_model_window_only(tmp_path):
    """
    The frame runs are counted over the 128-frame window the segment call is made
    on, as the reference squeak index did: on the 400 ms segment the 12-frame run
    at frames 130-141 lies past the window, so it sets the full-length offset but
    is not counted, while the 2-frame run at frames 0-1 is too short and the
    3-frame run at frames 60-62 counts. The 50 ms segment (about 25 frames)
    holds only the two short-run frames and the 60-62 run lies past its end.
    """
    root = _build_session(tmp_path)
    summary = pls.read_csv(root / "audio" / f"{SESSION_ID}_usv_summary.csv", schema_overrides={"usv_id": pls.String})
    scores = squeaks.score_squeak_rows(
        session_root=root,
        usv_summary=summary,
        row_indices=np.array([0, 1]),
        model=_FramePatternModel([0, 1, 60, 61, 62, *range(130, 142)]),
        device=torch.device("cpu"),
        threshold=0.385,
        exclude_metadata_audio_channels=True,
        batch_size=256,
        message_output=lambda *_a, **_kw: None,
    )
    assert scores["squeak"].to_list() == [True, True]
    assert scores[squeaks.SQUEAK_FRAME_RUNS_COLUMN].to_list() == [0, 1]
    assert scores["squeak_end"][1] == pytest.approx(0.30 + 141 * squeaks.FRAME_DT_S)


def test_score_squeak_rows_below_threshold_leaves_squeak_columns_empty(tmp_path):
    """With every frame forced to p = sigmoid(-5) nothing is a squeak, but the raw probability is still returned."""
    root = _build_session(tmp_path)
    summary = pls.read_csv(root / "audio" / f"{SESSION_ID}_usv_summary.csv", schema_overrides={"usv_id": pls.String})
    scores = squeaks.score_squeak_rows(
        session_root=root,
        usv_summary=summary,
        row_indices=np.array([0, 1]),
        model=_forced_model(-5.0),
        device=torch.device("cpu"),
        threshold=0.385,
        exclude_metadata_audio_channels=True,
        batch_size=256,
        message_output=lambda *_a, **_kw: None,
    )
    assert scores["squeak"].to_list() == [False, False]
    assert scores["squeak_probability"].null_count() == 2
    assert scores["squeak_start"].null_count() == 2
    assert scores["raw_probability"][0] == pytest.approx(float(torch.sigmoid(torch.tensor(-5.0))), abs=1e-6)
    assert scores[squeaks.SQUEAK_FRAME_RUNS_COLUMN].to_list() == [0, 0]


def test_squeak_wav_channels_honours_metadata_exclusion(tmp_path):
    """Metadata-excluded channels are dropped only when exclusion is switched on."""
    root = _build_session(tmp_path, excluded_channels=["m_ch02", "s_ch01"])
    kept = squeaks.squeak_wav_channels(root, exclude_metadata_audio_channels=True, message_output=lambda *_a, **_kw: None)
    assert [path.name.split("_cropped")[0] for path in kept] == ["m_250913193920_ch01", "s_250913193920_ch02"]
    everything = squeaks.squeak_wav_channels(root, exclude_metadata_audio_channels=False, message_output=lambda *_a, **_kw: None)
    assert len(everything) == 4


def test_detect_and_merge_writes_the_four_columns(tmp_path, mocker):
    """
    The merge replaces any existing squeak columns (squeak_frame_runs included),
    keeps every other column and the usv_id zero-padding, places the squeak block
    and squeak_frame_runs between emitter and the acoustic features (the
    canonical summary order), writes True / False and an integer run count in
    every row and leaves the probability and timing empty on non-squeak rows.
    """
    root = _build_session(tmp_path)
    summary_path = root / "audio" / f"{SESSION_ID}_usv_summary.csv"
    stale = pls.read_csv(summary_path, schema_overrides={"usv_id": pls.String}).with_columns(
        pls.lit(True).alias("squeak"), pls.lit(40000.0).alias("mean_freq_hz"), pls.lit(9).alias("squeak_frame_runs")
    )
    stale.write_csv(summary_path)

    mocker.patch("usv_playpen.processing.detect_usv_squeaks.smart_wait")
    mocker.patch("usv_playpen.processing.detect_usv_squeaks.torch.cuda.is_available", return_value=False)
    mocker.patch("usv_playpen.processing.detect_usv_squeaks.load_squeak_model", return_value=_forced_model(5.0))
    detector = squeaks.USVSqueakDetector(
        root_directory=str(root),
        input_parameter_dict={
            "detect_usv_squeaks": {
                "squeak_model_path": "unused.pt",
                "squeak_threshold": 0.385,
                "exclude_metadata_audio_channels": True,
                "batch_size": 256,
            }
        },
        message_output=lambda *_a, **_kw: None,
    )
    detector.detect_and_merge()

    written = pls.read_csv(summary_path, schema_overrides={"usv_id": pls.String})
    assert written.columns == [
        "usv_id", "start", "stop", "duration", "emitter", *squeaks.SQUEAK_COLUMNS, squeaks.SQUEAK_FRAME_RUNS_COLUMN, "mean_freq_hz",
    ]
    assert written[squeaks.SQUEAK_FRAME_RUNS_COLUMN].to_list() == [1, 1, 0]
    assert written["mean_freq_hz"].to_list() == [40000.0, 40000.0, 40000.0]
    assert written["usv_id"].to_list() == ["0000", "0001", "0002"]
    assert written["squeak"].to_list() == [True, True, False]
    assert written["squeak_probability"].null_count() == 1
    assert written["squeak_start"][2] is None
    assert written["squeak_end"][2] is None


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


def _build_squeak_embedding_session(tmp_path: pathlib.Path) -> pathlib.Path:
    """
    Description
    -----------
    Creates a synthetic session for the squeak QLVM embedding: the HPSS wavs of
    :func:`_build_session` and a summary whose rows cover every selection and
    crop case, with squeak extents placed on exact frame centres:

    * row 0 -- a squeak, not noise, frames 2 .. 20 of a 50 ms segment
      (crop 0 .. 22, embedded);
    * row 1 -- a squeak that is noise (not a candidate);
    * row 2 -- a squeak with a null noise value (counts as not noise), frames
      150 .. 190 of a 400 ms segment, i.e. past the 128-frame window (crop
      148 .. 192, embedded);
    * row 3 -- a squeak, frames 10 .. 180 of the 400 ms segment (crop of 175
      frames, too wide);
    * row 4 -- a squeak at frame 5 only of a 30 ms segment (crop of 5 frames,
      too narrow);
    * row 5 -- a squeak on a 4 ms segment (no spectrogram);
    * row 6 -- not a squeak.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Pytest temporary directory.

    Returns
    -------
    root (pathlib.Path)
        Session root directory.
    """

    root = _build_session(tmp_path)
    dt = squeaks.FRAME_DT_S
    pls.DataFrame(
        {
            "usv_id": ["0000", "0001", "0002", "0003", "0004", "0005", "0006"],
            "start": [0.10, 0.20, 0.30, 0.30, 0.75, 0.80, 0.85],
            "stop": [0.15, 0.25, 0.70, 0.70, 0.78, 0.804, 0.90],
            "noise": [False, True, None, False, False, False, False],
            "squeak": [True, True, True, True, True, True, False],
            "squeak_probability": [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, None],
            "squeak_start": [0.10 + 2 * dt, 0.20, 0.30 + 150 * dt, 0.30 + 10 * dt, 0.75 + 5 * dt, 0.80, None],
            "squeak_end": [0.10 + 20 * dt, 0.21, 0.30 + 190 * dt, 0.30 + 180 * dt, 0.75 + 5 * dt, 0.80, None],
            "mean_freq_hz": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        },
        schema={
            "usv_id": pls.String, "start": pls.Float64, "stop": pls.Float64, "noise": pls.Boolean,
            "squeak": pls.Boolean, "squeak_probability": pls.Float64, "squeak_start": pls.Float64,
            "squeak_end": pls.Float64, "mean_freq_hz": pls.Float64,
        },
    ).write_csv(root / "audio" / f"{SESSION_ID}_usv_summary.csv")
    return root


def test_squeak_crop_frames_adds_context_and_clips_to_the_segment():
    """Frame centres invert squeak_extent_seconds; two context frames are added and clipped to 0 .. n_frames - 1."""
    dt = squeaks.FRAME_DT_S
    start = np.array([10.0, 10.0, 10.0])
    first, last = squeaks.squeak_crop_frames(
        segment_start_s=start,
        squeak_start_s=start + np.array([1, 5, 40]) * dt,
        squeak_end_s=start + np.array([30, 60, 199]) * dt,
        n_frames=np.array([100, 61, 200]),
    )
    assert first.tolist() == [0, 3, 38]
    assert last.tolist() == [32, 60, 199]
    assert first.dtype == np.int64
    assert last.dtype == np.int64


def test_squeak_crop_inputs_normalizes_per_crop_and_centres_the_crop():
    """
    The crop is min-max normalized on its own (the dB values outside it play no
    part), centred in the 128-frame frame with (128 - width) // 2 zero columns on
    the left, and min-max normalized again; a crop wider than 128 frames raises.
    """
    rng = np.random.default_rng(3)
    spectrogram = rng.uniform(-90.0, 10.0, size=(128, 60)).astype(np.float32)
    spectrogram[:, :5] = 500.0
    inputs = squeaks.squeak_crop_inputs([spectrogram], np.array([10]), np.array([29]))
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
        squeaks.squeak_crop_inputs([np.zeros((128, 200), dtype=np.float32)], np.array([0]), np.array([150]))


def test_squeak_qlvm_rows_selects_squeaks_that_are_not_noise():
    """Squeak rows with noise false or null are selected; a summary without noise or squeak columns raises."""
    summary = pls.DataFrame(
        {"noise": [False, True, None, False], "squeak": [True, True, True, None]},
        schema={"noise": pls.Boolean, "squeak": pls.Boolean},
    ).with_columns(
        pls.lit(None, dtype=pls.Float64).alias(column) for column in ("squeak_probability", "squeak_start", "squeak_end")
    )
    assert squeaks.squeak_qlvm_rows(summary).tolist() == [0, 2]
    with pytest.raises(ValueError, match="detect-usv-noise"):
        squeaks.squeak_qlvm_rows(summary.drop("noise"))


def test_squeak_qlvm_inputs_crops_and_excludes(tmp_path):
    """Two squeaks are embedded (one past frame 127 of its segment); the other candidates are counted by reason."""
    root = _build_squeak_embedding_session(tmp_path)
    summary = pls.read_csv(root / "audio" / f"{SESSION_ID}_usv_summary.csv", schema_overrides={"usv_id": pls.String})
    built = squeaks.squeak_qlvm_inputs(root, summary, True, lambda *_a, **_kw: None)
    assert built["row_index"].tolist() == [0, 2]
    assert built["first"].tolist() == [0, 148]
    assert built["last"].tolist() == [22, 192]
    assert built["inputs"].shape == (2, 128, 128)
    assert built["n_candidates"] == 5
    assert built["excluded"] == {"no spectrogram": 1, "no squeak extent": 0, "crop < 8 frames": 1, "crop > 128 frames": 1}
    spectrograms = squeaks.squeak_segment_spectrograms(root, summary, np.array([0]), True, lambda *_a, **_kw: None)
    assert np.array_equal(built["inputs"][:1], squeaks.squeak_crop_inputs(spectrograms, np.array([0]), np.array([22])))


def _write_squeak_cell(cell: pathlib.Path, mask_tag: str = "nomask", masking_type: str = "none") -> None:
    """
    Description
    -----------
    Writes a minimal phase 3 squeak cell in the old package layout (every file at
    the cell root): a torch zip checkpoint holding a tiny unconditional
    legacy-head decoder prefix, ``run_config.json`` and ``manifest.json``.

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
    _write_squeak_cell(tmp_path / "bad", mask_tag="masked", masking_type="sam")
    with pytest.raises(ValueError, match="mask_tag") as error:
        squeaks.load_squeak_qlvm_cell(str(tmp_path / "bad"))
    assert "masking_type" in str(error.value)


def test_embed_and_merge_writes_the_two_squeak_torus_columns(tmp_path, mocker):
    """
    The embedded squeaks get their coordinates, every other row nulls, stale
    squeak coordinates are replaced, the other columns are kept and the two
    columns follow the canonical order (after every other known column).
    """
    root = _build_squeak_embedding_session(tmp_path)
    summary_path = root / "audio" / f"{SESSION_ID}_usv_summary.csv"
    pls.read_csv(summary_path, schema_overrides={"usv_id": pls.String}).with_columns(
        pls.lit(0.5).alias("qlvm_squeak1"), pls.lit(0.5).alias("qlvm_squeak2")
    ).write_csv(summary_path)
    mocker.patch("usv_playpen.processing.detect_usv_squeaks.smart_wait")
    mocker.patch(
        "usv_playpen.processing.detect_usv_squeaks.load_squeak_qlvm_cell",
        return_value={"params": {}, "lattice": np.zeros((21, 2)), "lattice_m": 8, "head": "legacy", "model_id": "p/c"},
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
    assert written["mean_freq_hz"].to_list() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    assert written["qlvm_squeak1"].null_count() == 5
    assert written["qlvm_squeak1"].is_null().to_list() == [False, True, False, True, True, True, True]
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
