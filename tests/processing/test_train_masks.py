"""
@author: bartulem
Tests for processing/train_masks.MaskDetectorTrainer.

The real Ultralytics fine-tune needs a GPU + network (to fetch the base weights)
and is not exercised here. ``ultralytics`` is replaced with a light fake injected
into ``sys.modules`` (its lazy ``from ultralytics import YOLO`` then picks it up),
and the test checks that the trainer copies the run's ``best.pt`` to the stable
output path and that a missing ``data.yaml`` is rejected before any import.
"""

from __future__ import annotations

import pathlib
import sys
import types

import pytest

from usv_playpen.processing.train_masks import MaskDetectorTrainer

_CFG = {
    "train_masks": {
        "base_weights": "yolo11n.pt",
        "n_epochs": 1,
        "imgsz": 128,
        "batch_size": 2,
        "device": None,
        "run_name": "usv_yolo_detector",
    }
}


def _install_fake_ultralytics(monkeypatch):
    """Replace ultralytics with a fake YOLO whose train() drops a best.pt under
    project/name/weights and returns a save_dir, like the real trainer."""
    module = types.ModuleType("ultralytics")

    class _FakeYOLO:
        def __init__(self, weights):
            self.weights = weights

        def train(self, **kwargs):
            save_dir = pathlib.Path(kwargs["project"]) / kwargs["name"]
            (save_dir / "weights").mkdir(parents=True, exist_ok=True)
            (save_dir / "weights" / "best.pt").write_bytes(b"fake-weights")
            return types.SimpleNamespace(save_dir=str(save_dir))

    module.YOLO = _FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", module)


def _write_dataset(tmp_path):
    """A minimal YOLO dataset dir (only data.yaml is needed for the faked run)."""
    dataset_dir = tmp_path / "yolo"
    dataset_dir.mkdir()
    (dataset_dir / "data.yaml").write_text("path: .\ntrain: images/train\nval: images/val\nnc: 1\nnames:\n  0: usv\n")
    return dataset_dir


def test_train_copies_best_weights(tmp_path, monkeypatch, mocker):
    """A faked train() run produces best.pt, which the trainer copies to
    <output_directory>/best.pt."""
    dataset_dir = _write_dataset(tmp_path)
    output_dir = tmp_path / "detector"
    _install_fake_ultralytics(monkeypatch)
    mocker.patch("usv_playpen.processing.train_masks.smart_wait")

    MaskDetectorTrainer(
        dataset_directory=str(dataset_dir),
        output_directory=str(output_dir),
        input_parameter_dict=_CFG,
        message_output=lambda *_a, **_kw: None,
    ).train()

    assert (output_dir / "best.pt").is_file()
    assert (output_dir / "best.pt").read_bytes() == b"fake-weights"


def test_train_raises_when_best_weights_missing(tmp_path, monkeypatch, mocker):
    """If the Ultralytics run produces no best.pt, the trainer raises instead of
    silently leaving a stale/absent weights file."""
    dataset_dir = _write_dataset(tmp_path)

    module = types.ModuleType("ultralytics")

    class _NoWeightsYOLO:
        def __init__(self, weights):
            self.weights = weights

        def train(self, **kwargs):
            save_dir = pathlib.Path(kwargs["project"]) / kwargs["name"]
            save_dir.mkdir(parents=True, exist_ok=True)  # deliberately writes no weights/best.pt
            return types.SimpleNamespace(save_dir=str(save_dir))

    module.YOLO = _NoWeightsYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", module)
    mocker.patch("usv_playpen.processing.train_masks.smart_wait")

    with pytest.raises(FileNotFoundError, match="best.pt"):
        MaskDetectorTrainer(
            dataset_directory=str(dataset_dir),
            output_directory=str(tmp_path / "detector"),
            input_parameter_dict=_CFG,
            message_output=lambda *_a, **_kw: None,
        ).train()


def test_train_missing_data_yaml_raises(tmp_path, mocker):
    """A dataset directory without data.yaml raises FileNotFoundError (before any
    ultralytics import)."""
    dataset_dir = tmp_path / "empty"
    dataset_dir.mkdir()
    mocker.patch("usv_playpen.processing.train_masks.smart_wait")
    with pytest.raises(FileNotFoundError, match="data.yaml"):
        MaskDetectorTrainer(
            dataset_directory=str(dataset_dir),
            output_directory=str(tmp_path / "out"),
            input_parameter_dict=_CFG,
            message_output=lambda *_a, **_kw: None,
        ).train()
