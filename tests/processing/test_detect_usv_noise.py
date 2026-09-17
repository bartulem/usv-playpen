"""
@author: bartulem
Tests for processing/detect_usv_noise -- the step that flags USV segments holding no vocalization.

The load-bearing checks are the ones a wrong answer would be silent about: the threshold the settings'
``noise_min_precision`` resolves to, the input the models are handed (segment frames only, cut out of a
context window), and the merge, which must replace stale noise columns and leave the summary in the
canonical column order with the noise block before the squeak block.
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

from usv_playpen.processing import detect_usv_noise as noise

SESSION_ID = "20250913_193920"
SAMPLING_RATE = noise.NOISE_SAMPLING_RATE
CALIBRATION = [
    {"threshold": 0.30, "precision": 0.949, "recall": 0.992, "flagged_per_10000": 342.0, "calls_lost_per_10000": 17.6},
    {"threshold": 0.45, "precision": 0.987, "recall": 0.973, "flagged_per_10000": 322.0, "calls_lost_per_10000": 4.0},
    {"threshold": 0.60, "precision": 1.000, "recall": 0.930, "flagged_per_10000": 304.0, "calls_lost_per_10000": 0.0},
]


def _forced_bundle(logit: float, in_channels: int = 3) -> dict:
    """A bundle whose single model returns a fixed logit for every segment, with a two-row calibration."""
    model = noise.NoiseTimeMIL(in_channels=in_channels, n_scalars=2)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.head.frame.bias.fill_(logit)
    model.eval()
    return {
        "models": [model],
        "scalar_mean": np.zeros(2, dtype=np.float32), "scalar_std": np.ones(2, dtype=np.float32),
        "db_floor": -100.0, "db_ceil": 50.0, "db_center": -25.0, "db_half": 75.0,
        "max_frames": 512, "context_frames": 49, "bands_hz": noise.NOISE_BANDS_HZ, "calibration": CALIBRATION,
    }


def _build_session(tmp_path: pathlib.Path, excluded_channels: list[str] | None = None) -> pathlib.Path:
    """
    Description
    -----------
    Creates a synthetic session: four PCM_16 HPSS wavs (two master, two slave channels) of noise, and a
    summary with two normal segments (50 and 100 ms) and a 4 ms one, far shorter than a single STFT
    window. Optionally writes session metadata excluding channels.

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
        sf.write(str(hpss_dir / f"{device}_250913193920_ch{channel:02d}_cropped_to_video_hpss.wav"), audio, SAMPLING_RATE, subtype="PCM_16")
    pls.DataFrame(
        {
            "usv_id": ["0000", "0001", "0002"],
            "start": [0.10, 0.30, 0.80],
            "stop": [0.15, 0.40, 0.804],
            "duration": [0.05, 0.10, 0.004],
            "chs_count": [4.0, 2.0, 1.0],
            "emitter": [None, None, None],
        },
        schema={"usv_id": pls.String, "start": pls.Float64, "stop": pls.Float64, "duration": pls.Float64,
                "chs_count": pls.Float64, "emitter": pls.String},
    ).write_csv(root / "audio" / f"{SESSION_ID}_usv_summary.csv")
    if excluded_channels is not None:
        metadata = {"Equipment": {"audio_Avisoft": {"excluded_channels": excluded_channels}}}
        (root / f"{SESSION_ID}_metadata.yaml").write_text(yaml.dump(metadata))
    return root


def test_resolve_threshold_takes_the_lowest_threshold_reaching_the_target():
    """The lowest reaching threshold is the one that also catches the most noise, so that is the pick."""
    assert noise.resolve_threshold(CALIBRATION, 0.98)["threshold"] == 0.45
    assert noise.resolve_threshold(CALIBRATION, 0.94)["threshold"] == 0.30
    assert noise.resolve_threshold(CALIBRATION, 1.0)["threshold"] == 0.60


def test_resolve_threshold_refuses_an_unreachable_target_and_shows_the_table():
    """Flagging more real calls than the user allowed is worse than stopping, so a target no threshold
    reaches raises, with the table in the message so the user can see what is on offer."""
    with pytest.raises(ValueError, match="calibration") as excinfo:
        noise.resolve_threshold(CALIBRATION[:2], 0.999)
    assert "0.45" in str(excinfo.value)


def test_segment_input_keeps_only_the_segment_frames():
    """The context window sets the channel weights and the STFT edges, but the model sees the segment."""
    rng = np.random.default_rng(1)
    window = rng.uniform(-0.2, 0.2, size=(60 * noise.HOP_SAMPLES, 3))
    bundle = _forced_bundle(0.0)
    x, n_used = noise.segment_input(window, first_frame=10, n_frames=8, bundle=bundle)
    assert x.shape == (3, 128, 8)
    assert n_used == 8
    assert np.array_equal(x[2], np.ones((128, 8), dtype=np.float32))     # the segment-indicator channel
    assert x[:2].min() >= -1.0
    assert x[:2].max() <= 1.0
    short, n_short = noise.segment_input(window[: noise.NOISE_SPEC_BASE["nperseg"] // 2], 0, 1, bundle)
    assert short is None
    assert n_short == 0


def test_frame_budget_batches_keep_a_long_call_from_inflating_a_batch():
    """Batches are padded to their longest call, so a batch is budgeted in frame slots, not segments: one
    512-frame call among short ones would otherwise make the batch 30x larger and exhaust the GPU."""
    short = [np.zeros((3, 128, 16), dtype=np.float32)] * 6
    long_call = np.zeros((3, 128, 512), dtype=np.float32)
    batches = noise.frame_budget_batches([*short, long_call, *short], batch_size=1)
    assert batches[0] == (0, 6)                       # the short calls fill one batch (6 * 16 <= 128)
    assert (6, 7) in batches                          # the long call is scored on its own
    assert batches[-1][1] == 13                       # every segment is covered, in order
    assert noise.frame_budget_batches([long_call], batch_size=1) == [(0, 1)]   # never an empty batch


def test_score_noise_rows_flags_every_row_above_the_threshold(tmp_path):
    """Every row gets a probability, including a 4 ms segment: the audio window extends a context either
    side, so even a segment far shorter than one STFT window still yields its own frame to score."""
    root = _build_session(tmp_path)
    summary = pls.read_csv(root / "audio" / f"{SESSION_ID}_usv_summary.csv", schema_overrides={"usv_id": pls.String})
    scores = noise.score_noise_rows(
        session_root=root, usv_summary=summary, bundle=_forced_bundle(4.0), device=torch.device("cpu"),
        threshold=0.5, exclude_metadata_audio_channels=True, batch_size=8, message_output=lambda *_a, **_kw: None,
    )
    assert scores["noise"].to_list() == [True, True, True]
    assert scores["noise_probability"][0] == pytest.approx(1 / (1 + np.exp(-4.0)), abs=1e-6)
    assert scores["noise_probability"].null_count() == 0


def test_score_noise_rows_below_threshold_flags_nothing(tmp_path):
    """A model that is confident every segment holds a call must flag no row."""
    root = _build_session(tmp_path)
    summary = pls.read_csv(root / "audio" / f"{SESSION_ID}_usv_summary.csv", schema_overrides={"usv_id": pls.String})
    scores = noise.score_noise_rows(
        session_root=root, usv_summary=summary, bundle=_forced_bundle(-4.0), device=torch.device("cpu"),
        threshold=0.5, exclude_metadata_audio_channels=True, batch_size=8, message_output=lambda *_a, **_kw: None,
    )
    assert scores["noise"].to_list() == [False, False, False]
    assert scores["noise_probability"][0] < 0.5


def test_load_noise_model_refuses_a_bundle_trained_on_other_bands(tmp_path):
    """A checkpoint whose bands differ from the ones this module builds would be fed the wrong input."""
    model = noise.NoiseTimeMIL(in_channels=3, n_scalars=2)
    path = tmp_path / "bundle.pt"
    torch.save({
        "state_dicts": [model.state_dict()], "in_channels": 3, "n_scalars": 2,
        "scalar_mean": [0.0, 0.0], "scalar_std": [1.0, 1.0],
        "db_floor": -100.0, "db_ceil": 50.0, "db_center": -25.0, "db_half": 75.0,
        "max_frames": 512, "context_frames": 49, "bands_hz": [[20000.0, 100000.0], [3000.0, 30000.0]],
        "calibration": CALIBRATION,
    }, path)
    with pytest.raises(ValueError, match="bands"):
        noise.load_noise_model(str(path), torch.device("cpu"))


def test_load_noise_model_reports_a_missing_bundle(tmp_path):
    """The derived path is the usual cause, so the error names the setting that overrides it."""
    with pytest.raises(FileNotFoundError, match="noise_model_path"):
        noise.load_noise_model(str(tmp_path / "absent.pt"), torch.device("cpu"))


def test_detect_and_merge_writes_the_noise_columns_before_the_squeak_block(tmp_path, mocker):
    """
    The merge replaces stale noise columns, keeps every other column and the usv_id zero-padding, and
    leaves the summary in the canonical order: emitter, the noise block, then the squeak block.
    """
    root = _build_session(tmp_path)
    summary_path = root / "audio" / f"{SESSION_ID}_usv_summary.csv"
    stale = pls.read_csv(summary_path, schema_overrides={"usv_id": pls.String}).with_columns(
        pls.lit(True).alias("noise"), pls.lit(0.1).alias("noise_probability"),
        pls.lit(False).alias("squeak"), pls.lit(40000.0).alias("mean_freq_hz"),
    )
    stale.write_csv(summary_path)

    mocker.patch("usv_playpen.processing.detect_usv_noise.smart_wait")
    mocker.patch("usv_playpen.processing.detect_usv_noise.torch.cuda.is_available", return_value=False)
    mocker.patch("usv_playpen.processing.detect_usv_noise.load_noise_model", return_value=_forced_bundle(4.0))
    messages: list[str] = []
    noise.USVNoiseDetector(
        root_directory=str(root),
        input_parameter_dict={
            "detect_usv_noise": {
                "noise_model_path": "unused.pt",
                "noise_min_precision": 0.98,
                "exclude_metadata_audio_channels": True,
                "batch_size": 256,
            }
        },
        message_output=messages.append,
    ).detect_and_merge()

    written = pls.read_csv(summary_path, schema_overrides={"usv_id": pls.String})
    assert written.columns == ["usv_id", "start", "stop", "duration", "chs_count", "emitter", *noise.NOISE_COLUMNS, "squeak", "mean_freq_hz"]
    assert written["usv_id"].to_list() == ["0000", "0001", "0002"]
    assert written["noise"].to_list() == [True, True, True]
    assert written["noise_probability"].null_count() == 0
    assert written["mean_freq_hz"].to_list() == [40000.0, 40000.0, 40000.0]
    assert any("noise_min_precision 0.98 -> p >= 0.45" in message for message in messages)


def test_detect_usv_noise_cli_routes(mocker, tmp_path):
    """detect-usv-noise resolves settings and calls USVNoiseDetector.detect_and_merge once."""
    mock_cls = mocker.patch("usv_playpen.processing.detect_usv_noise.USVNoiseDetector")
    mocker.patch("usv_playpen.processing.detect_usv_noise.modify_settings_json_for_cli", return_value={"detect_usv_noise": {}})
    result = CliRunner().invoke(noise.detect_usv_noise_cli, ["--root-directory", str(tmp_path)])
    assert result.exit_code == 0, result.output
    mock_cls.assert_called_once()
    mock_cls.return_value.detect_and_merge.assert_called_once()
