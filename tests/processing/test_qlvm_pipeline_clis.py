"""
@author: bartulem
CLI-level routing tests for the in-house USV spectrogram / mask / latent pipeline
commands (per-session inference + cross-session model training).

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
from usv_playpen.processing.train_qlvm import train_qlvm_cli
from usv_playpen.processing.export_yolo_dataset import export_yolo_dataset_cli
from usv_playpen.processing.train_masks import train_masks_cli


@pytest.fixture
def runner():
    """Provides a CliRunner instance for invoking commands."""
    return CliRunner()


def test_generate_spectrograms_cli_routes(runner, mocker, tmp_path):
    """generate-usv-spectrograms resolves settings and calls SpectrogramGenerator once."""
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
    """build-qlvm-training-set splits the comma-separated root directories and calls
    the builder once with the parsed list."""
    mock_cls = mocker.patch("usv_playpen.processing.build_qlvm_training_set.QLVMTrainingSetBuilder")
    mocker.patch(
        "usv_playpen.processing.build_qlvm_training_set.modify_settings_json_for_cli",
        return_value={"build_qlvm_training_set": {}},
    )
    result = runner.invoke(build_qlvm_training_set_cli, [
        "--root-directories", "/a/sess1,/b/sess2",
        "--output-directory", str(tmp_path / "out"),
    ])
    assert result.exit_code == 0, result.output
    mock_cls.assert_called_once()
    assert mock_cls.call_args.kwargs["root_directories"] == ["/a/sess1", "/b/sess2"]
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


def test_train_qlvm_cli_routes(runner, mocker, tmp_path):
    """train-qlvm resolves settings and calls QLVMTrainer.train once."""
    mock_cls = mocker.patch("usv_playpen.processing.train_qlvm.QLVMTrainer")
    mocker.patch(
        "usv_playpen.processing.train_qlvm.modify_settings_json_for_cli",
        return_value={"train_qlvm": {}},
    )
    result = runner.invoke(train_qlvm_cli, [
        "--dataset-directory", str(tmp_path),       # exists=True
        "--output-directory", str(tmp_path / "out"),
    ])
    assert result.exit_code == 0, result.output
    mock_cls.assert_called_once()
    assert mock_cls.call_args.kwargs["dataset_directory"] == str(tmp_path)
    mock_cls.return_value.train.assert_called_once()


def test_export_yolo_dataset_cli_routes_and_splits_paths(runner, mocker, tmp_path):
    """export-yolo-dataset splits the comma-separated root directories and calls
    the exporter once with the parsed list."""
    mock_cls = mocker.patch("usv_playpen.processing.export_yolo_dataset.YOLODatasetExporter")
    mocker.patch(
        "usv_playpen.processing.export_yolo_dataset.modify_settings_json_for_cli",
        return_value={"export_yolo_dataset": {}},
    )
    result = runner.invoke(export_yolo_dataset_cli, [
        "--root-directories", "/a/sess1, /b/sess2",
        "--output-directory", str(tmp_path / "out"),
    ])
    assert result.exit_code == 0, result.output
    mock_cls.assert_called_once()
    assert mock_cls.call_args.kwargs["root_directories"] == ["/a/sess1", "/b/sess2"]
    mock_cls.return_value.export.assert_called_once()


def test_train_masks_cli_routes(runner, mocker, tmp_path):
    """train-masks resolves settings and calls MaskDetectorTrainer.train once."""
    mock_cls = mocker.patch("usv_playpen.processing.train_masks.MaskDetectorTrainer")
    mocker.patch(
        "usv_playpen.processing.train_masks.modify_settings_json_for_cli",
        return_value={"train_masks": {}},
    )
    result = runner.invoke(train_masks_cli, [
        "--dataset-directory", str(tmp_path),       # exists=True
        "--output-directory", str(tmp_path / "out"),
    ])
    assert result.exit_code == 0, result.output
    mock_cls.assert_called_once()
    assert mock_cls.call_args.kwargs["dataset_directory"] == str(tmp_path)
    mock_cls.return_value.train.assert_called_once()
