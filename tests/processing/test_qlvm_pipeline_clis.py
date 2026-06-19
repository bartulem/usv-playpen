"""
@author: bartulem
CLI-level routing tests for the QLVM pipeline commands.

Each click command should resolve settings via modify_settings_json_for_cli and
dispatch to its backend class' single public method exactly once. The backend
classes and the settings resolver are mocked, so these exercise only the CLI
wiring (no file I/O / no real compute).
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from usv_playpen.processing.qlvm_latents import infer_qlvm_latents_cli
from usv_playpen.processing.build_qlvm_training_set import build_qlvm_training_set_cli
from usv_playpen.processing.compute_usv_acoustic_features import (
    compute_usv_acoustic_features_cli,
)
from usv_playpen.processing.generate_spectrograms import generate_spectrograms_cli


@pytest.fixture
def runner():
    """Provides a CliRunner instance for invoking commands."""
    return CliRunner()


def test_generate_spectrograms_cli_routes(runner, mocker, tmp_path):
    """generate-spectrograms resolves settings and calls SpectrogramGenerator once."""
    mock_cls = mocker.patch("usv_playpen.processing.generate_spectrograms.SpectrogramGenerator")
    mocker.patch(
        "usv_playpen.processing.generate_spectrograms.modify_settings_json_for_cli",
        return_value={"generate_spectrograms": {}},
    )
    result = runner.invoke(generate_spectrograms_cli, ["--root-directory", str(tmp_path)])
    assert result.exit_code == 0, result.output
    mock_cls.assert_called_once()
    mock_cls.return_value.generate_session_spectrograms.assert_called_once()


def test_compute_usv_acoustic_features_cli_routes(runner, mocker, tmp_path):
    """generate-usv-acoustic-features calls USVAcousticFeatureExtractor once."""
    mock_cls = mocker.patch("usv_playpen.processing.compute_usv_acoustic_features.USVAcousticFeatureExtractor")
    mocker.patch(
        "usv_playpen.processing.compute_usv_acoustic_features.modify_settings_json_for_cli",
        return_value={"compute_usv_acoustic_features": {}},
    )
    result = runner.invoke(compute_usv_acoustic_features_cli, ["--root-directory", str(tmp_path)])
    assert result.exit_code == 0, result.output
    mock_cls.assert_called_once()
    mock_cls.return_value.merge_features_into_summary.assert_called_once()


def test_build_qlvm_training_set_cli_routes_and_splits_paths(runner, mocker, tmp_path):
    """build-qlvm-training-set splits the comma-separated H5 paths and calls the
    builder once with the parsed list."""
    mock_cls = mocker.patch("usv_playpen.processing.build_qlvm_training_set.QLVMTrainingSetBuilder")
    mocker.patch(
        "usv_playpen.processing.build_qlvm_training_set.modify_settings_json_for_cli",
        return_value={"build_qlvm_training_set": {}},
    )
    result = runner.invoke(build_qlvm_training_set_cli, [
        "--spectrogram-h5-paths", "/a/x.h5,/b/y.h5",
        "--output-directory", str(tmp_path / "out"),
    ])
    assert result.exit_code == 0, result.output
    mock_cls.assert_called_once()
    assert mock_cls.call_args.kwargs["spectrogram_h5_paths"] == ["/a/x.h5", "/b/y.h5"]
    mock_cls.return_value.build.assert_called_once()


def test_infer_qlvm_latents_cli_routes(runner, mocker, tmp_path):
    """infer-qlvm-latents calls QLVMLatentInference once."""
    mock_cls = mocker.patch("usv_playpen.processing.qlvm_latents.QLVMLatentInference")
    mocker.patch(
        "usv_playpen.processing.qlvm_latents.modify_settings_json_for_cli",
        return_value={"infer_qlvm_latents": {}},
    )
    result = runner.invoke(infer_qlvm_latents_cli, ["--root-directory", str(tmp_path)])
    assert result.exit_code == 0, result.output
    mock_cls.assert_called_once()
    mock_cls.return_value.infer_and_merge.assert_called_once()
