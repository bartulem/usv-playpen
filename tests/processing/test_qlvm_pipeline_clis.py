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

import json
import pathlib
import warnings

import click
import pytest
from click.testing import CliRunner

from usv_playpen.processing.qlvm_latents import infer_qlvm_latents_cli
from usv_playpen.processing.qlvm_categories import assign_qlvm_categories_cli, build_qlvm_categories_cli
from usv_playpen.processing.build_qlvm_squeak_training_set import build_qlvm_squeak_training_set_cli
from usv_playpen.processing.build_squeak_spectrogram_store import build_squeak_spectrogram_store_cli
from usv_playpen.processing.build_qlvm_training_set import build_qlvm_training_set_cli
from usv_playpen.processing.compute_usv_acoustic_features import (
    compute_usv_acoustic_features_cli,
)
from usv_playpen.processing.generate_masks import generate_masks_cli
from usv_playpen.processing.generate_spectrograms import generate_spectrograms_cli
from usv_playpen.processing.export_yolo_dataset import export_yolo_dataset_cli
from usv_playpen.processing.train_masks import train_masks_cli

# train_qlvm pulls optax -> a one-time JAX DeprecationWarning at import.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from usv_playpen.processing.train_qlvm import train_qlvm_cli


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


def test_build_qlvm_training_set_cli_decodes_structured_options(runner, mocker, tmp_path):
    """--session-type-targets (JSON), --mask-count-bin-edges (comma-separated) and
    --floor none reach the settings override decoded, as a dict, a list and None."""
    mocker.patch("usv_playpen.processing.build_qlvm_training_set.QLVMTrainingSetBuilder")
    seen = {}

    def _capture(ctx, provided_params, **_kwargs):
        seen.update({key: ctx.params[key] for key in provided_params})
        return {"build_qlvm_training_set": {}}

    mocker.patch("usv_playpen.processing.build_qlvm_training_set.modify_settings_json_for_cli", side_effect=_capture)
    result = runner.invoke(build_qlvm_training_set_cli, [
        "--root-directories", "/a/sess1", "--output-directory", str(tmp_path / "out"),
        "--session-type-targets", '{"MF": 10, "MM": null}', "--mask-count-bin-edges", "1,2,3", "--floor", "none",
        "--row-exclusion-table", "none",
    ])
    assert result.exit_code == 0, result.output
    assert seen == {"session_type_targets": {"MF": 10, "MM": None}, "mask_count_bin_edges": [1, 2, 3], "floor": None,
                    "row_exclusion_table": None}
    seen.clear()
    result = runner.invoke(build_qlvm_training_set_cli, [
        "--root-directories", "/a/sess1", "--output-directory", str(tmp_path / "out"), "--row-exclusion-table", "/x/rows.csv",
    ])
    assert result.exit_code == 0, result.output
    assert seen == {"row_exclusion_table": "/x/rows.csv"}


def test_build_qlvm_categories_cli_exposes_every_settings_key():
    """Every key of the build_qlvm_categories block has a --flag (the command takes
    file arguments, so it is checked here rather than in the parametrized guard)."""
    option_dests = {p.name for p in build_qlvm_categories_cli.params if isinstance(p, click.Option)}
    missing = set(_PROCESSING_SETTINGS["build_qlvm_categories"]) - option_dests
    assert not missing, f"build_qlvm_categories: settings keys with no CLI flag: {sorted(missing)}"


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


def test_infer_qlvm_latents_cli_model_cell_pairs_become_model_cells(runner, mocker, tmp_path):
    """Repeated --model-cell PREFIX CELL options become the model_cells object (prefix
    -> cell, in the given order), written after the generic overrides, which never see
    them; a prefix given twice is refused."""
    mock_cls = mocker.patch("usv_playpen.processing.qlvm_latents.QLVMLatentInference")
    spy = mocker.patch(
        "usv_playpen.processing.qlvm_latents.modify_settings_json_for_cli",
        return_value={"infer_qlvm_latents": {"model_cells": {}}},
    )
    result = runner.invoke(infer_qlvm_latents_cli, [
        "--root-directory", str(tmp_path),
        "--model-cell", "qlvm", "/pkg/phase6/cell",
        "--model-cell", "qlvm_duration", "/pkg/phase11_duration/cell",
        "--no-prefer-package-values",
    ])
    assert result.exit_code == 0, result.output
    assert spy.call_args.kwargs["provided_params"] == ["prefer_package_values"]
    settings = mock_cls.call_args.kwargs["input_parameter_dict"]["infer_qlvm_latents"]
    assert settings["model_cells"] == {"qlvm": "/pkg/phase6/cell", "qlvm_duration": "/pkg/phase11_duration/cell"}
    assert list(settings["model_cells"]) == ["qlvm", "qlvm_duration"]

    result = runner.invoke(infer_qlvm_latents_cli, [
        "--root-directory", str(tmp_path),
        "--model-cell", "qlvm", "/a",
        "--model-cell", "qlvm", "/b",
    ])
    assert result.exit_code != 0
    assert "prefixes listed more than once" in str(result.exception)


def test_infer_qlvm_latents_cli_has_no_label_level_option(runner, mocker, tmp_path):
    """infer-qlvm-latents writes torus coordinates only: the retired --model-cell-labels
    option is refused as unknown and the inference never runs."""
    mock_cls = mocker.patch("usv_playpen.processing.qlvm_latents.QLVMLatentInference")
    result = runner.invoke(infer_qlvm_latents_cli, [
        "--root-directory", str(tmp_path),
        "--model-cell-labels", "qlvm", "fine",
    ])
    assert result.exit_code != 0
    assert "No such option: --model-cell-labels" in result.output
    mock_cls.assert_not_called()


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


def test_train_qlvm_cli_none_is_the_null_setting(runner, mocker, tmp_path):
    """train-qlvm's --conditional none and --condition-table none reach the settings
    as null (an unconditional run, no table); a named condition passes through."""
    mocker.patch("usv_playpen.processing.train_qlvm.QLVMTrainer")
    modify = mocker.patch(
        "usv_playpen.processing.train_qlvm.modify_settings_json_for_cli",
        return_value={"train_qlvm": {}},
    )
    base = ["--dataset-directory", str(tmp_path), "--output-directory", str(tmp_path / "out")]
    result = runner.invoke(train_qlvm_cli, [*base, "--conditional", "none", "--condition-table", "None"])
    assert result.exit_code == 0, result.output
    ctx = modify.call_args.kwargs["ctx"]
    assert ctx.params["conditional"] is None
    assert ctx.params["condition_table"] is None
    assert {"conditional", "condition_table"} <= set(modify.call_args.kwargs["provided_params"])

    result = runner.invoke(train_qlvm_cli, [*base, "--conditional", "loudness", "--no-condition-scale-loss-by-batch"])
    assert result.exit_code == 0, result.output
    ctx = modify.call_args.kwargs["ctx"]
    assert ctx.params["conditional"] == "loudness"
    assert ctx.params["condition_scale_loss_by_batch"] is False


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


# Full-coverage + block-scoping guards for every in-house pipeline command.
# (cli, backend module path, backend class, settings block, invoke args templated
# on {d} = an existing dir, {o} = an output dir).
_PIPELINE_CLIS = [
    (generate_spectrograms_cli, "usv_playpen.processing.generate_spectrograms", "SpectrogramGenerator", "generate_spectrograms", ["--root-directory", "{d}"]),
    (generate_masks_cli, "usv_playpen.processing.generate_masks", "MaskGenerator", "generate_masks", ["--root-directory", "{d}"]),
    (compute_usv_acoustic_features_cli, "usv_playpen.processing.compute_usv_acoustic_features", "USVAcousticFeatureExtractor", "compute_usv_acoustic_features", ["--root-directory", "{d}"]),
    (build_qlvm_training_set_cli, "usv_playpen.processing.build_qlvm_training_set", "QLVMTrainingSetBuilder", "build_qlvm_training_set", ["--root-directories", "/a,/b", "--output-directory", "{o}"]),
    (build_qlvm_squeak_training_set_cli, "usv_playpen.processing.build_qlvm_squeak_training_set", "QLVMSqueakTrainingSetBuilder", "build_qlvm_squeak_training_set", ["--root-directories", "/a,/b", "--output-directory", "{o}"]),
    (build_squeak_spectrogram_store_cli, "usv_playpen.processing.build_squeak_spectrogram_store", "SqueakSpectrogramStoreBuilder", "build_squeak_spectrogram_store", ["--root-directories", "/a,/b"]),
    (train_qlvm_cli, "usv_playpen.processing.train_qlvm", "QLVMTrainer", "train_qlvm", ["--dataset-directory", "{d}", "--output-directory", "{o}"]),
    (infer_qlvm_latents_cli, "usv_playpen.processing.qlvm_latents", "QLVMLatentInference", "infer_qlvm_latents", ["--root-directory", "{d}"]),
    (assign_qlvm_categories_cli, "usv_playpen.processing.qlvm_categories", "QLVMCategoryAssigner", "assign_qlvm_categories", ["--root-directory", "{d}"]),
    (export_yolo_dataset_cli, "usv_playpen.processing.export_yolo_dataset", "YOLODatasetExporter", "export_yolo_dataset", ["--root-directories", "/a,/b", "--output-directory", "{o}"]),
    (train_masks_cli, "usv_playpen.processing.train_masks", "MaskDetectorTrainer", "train_masks", ["--dataset-directory", "{d}", "--output-directory", "{o}"]),
]

_PROCESSING_SETTINGS = json.loads(
    (pathlib.Path(__file__).parents[2] / "src/usv_playpen/_parameter_settings/processing_settings.json").read_text(encoding="utf-8")
)


@pytest.mark.parametrize(
    "cli,block",
    [(entry[0], entry[3]) for entry in _PIPELINE_CLIS],
    ids=[entry[3] for entry in _PIPELINE_CLIS],
)
def test_pipeline_cli_exposes_every_settings_key(cli, block):
    """Every key in a command's processing_settings.json block must be overridable
    from the CLI. A setting added to the JSON without its matching ``--flag`` (the
    incompleteness this expansion fixed) fails here. A click ``Option``'s ``.name``
    is the bare setting key it writes to."""
    option_dests = {p.name for p in cli.params if isinstance(p, click.Option)}
    missing = set(_PROCESSING_SETTINGS[block]) - option_dests
    assert not missing, f"{block}: settings keys with no CLI flag: {sorted(missing)}"


@pytest.mark.parametrize(
    "cli,module_path,backend,block,args",
    _PIPELINE_CLIS,
    ids=[entry[3] for entry in _PIPELINE_CLIS],
)
def test_pipeline_cli_scopes_overrides_to_its_own_block(
    cli, module_path, backend, block, args, runner, mocker, tmp_path
):
    """Each command must call ``modify_settings_json_for_cli`` with
    ``block='<its own block>'`` so a key shared across blocks (``n_epochs`` /
    ``batch_size`` / ``latent_dim``) can never be written into another command's
    block — the failure mode block-scoping fixed. Guards against a command dropping
    or mistyping its ``block=`` argument."""
    mocker.patch(f"{module_path}.{backend}")
    spy = mocker.patch(f"{module_path}.modify_settings_json_for_cli", return_value={block: {}})
    concrete_args = [arg.format(d=str(tmp_path), o=str(tmp_path / "out")) for arg in args]
    result = runner.invoke(cli, concrete_args)
    assert result.exit_code == 0, result.output
    call_kwargs = spy.call_args.kwargs
    assert "block" in call_kwargs and call_kwargs["block"] == block
