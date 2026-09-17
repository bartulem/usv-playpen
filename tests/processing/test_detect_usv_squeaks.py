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
metadata channel exclusion without needing the real checkpoint.
"""

from __future__ import annotations

import pathlib

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


def test_squeak_wav_channels_honours_metadata_exclusion(tmp_path):
    """Metadata-excluded channels are dropped only when exclusion is switched on."""
    root = _build_session(tmp_path, excluded_channels=["m_ch02", "s_ch01"])
    kept = squeaks.squeak_wav_channels(root, exclude_metadata_audio_channels=True, message_output=lambda *_a, **_kw: None)
    assert [path.name.split("_cropped")[0] for path in kept] == ["m_250913193920_ch01", "s_250913193920_ch02"]
    everything = squeaks.squeak_wav_channels(root, exclude_metadata_audio_channels=False, message_output=lambda *_a, **_kw: None)
    assert len(everything) == 4


def test_detect_and_merge_writes_the_four_columns(tmp_path, mocker):
    """
    The merge replaces any existing squeak columns, keeps every other column and
    the usv_id zero-padding, places the squeak block between emitter and the
    acoustic features (the canonical summary order), writes True / False in every
    row and leaves the probability and timing empty on non-squeak rows.
    """
    root = _build_session(tmp_path)
    summary_path = root / "audio" / f"{SESSION_ID}_usv_summary.csv"
    stale = pls.read_csv(summary_path, schema_overrides={"usv_id": pls.String}).with_columns(
        pls.lit(True).alias("squeak"), pls.lit(40000.0).alias("mean_freq_hz")
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
    assert written.columns == ["usv_id", "start", "stop", "duration", "emitter", *squeaks.SQUEAK_COLUMNS, "mean_freq_hz"]
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
