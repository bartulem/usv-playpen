"""
@author: bartulem
Tests for processing/export_yolo_dataset.YOLODatasetExporter.

Synthesizes a per-session spectrogram H5, then checks the Ultralytics dataset
layout is written (images/{train,val}, labels/{train,val}, data.yaml), that the
rendering helper matches the inference window shape, that the connected-component
label source produces a label file per valid spectrogram, that the manual source
copies hand-verified labels verbatim, and that manual/merge without a labels
directory is rejected.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest
from PIL import Image

from usv_playpen.processing.export_yolo_dataset import (
    YOLODatasetExporter,
    spec_to_yolo_image,
)

_SESSION_ID = "20230119_155302"


def _cfg(**overrides):
    base = {
        "label_source": "cc",
        "validation_split": 0.0,  # all -> train, deterministic for assertions
        "random_state": 0,
        "colormap": "viridis",
        "manual_labels_directory": "",
    }
    base.update(overrides)
    return {"export_yolo_dataset": base}


def _write_session_h5(path, durations):
    """A consolidated-layout spectrogram H5 (no mask group needed here)."""
    rng = np.random.default_rng(0)
    n = len(durations)
    specs = rng.random((n, 128, 128)).astype(np.float32)
    with h5py.File(path, "w") as f:
        f.create_dataset("frequency_bins", data=np.linspace(30000.0, 120000.0, 128))
        grp = f.create_group(f"spectrogram/{_SESSION_ID}")
        grp.create_dataset("spectrograms", data=specs)
        grp.create_dataset("durations", data=np.asarray(durations, dtype=np.int64))


def test_spec_to_yolo_image_shape():
    """The rendered image is the flipped signal window (H == F, W == duration)."""
    spec = np.random.default_rng(0).random((128, 128)).astype(np.float32)
    image, width, height = spec_to_yolo_image(spec, duration=40, colormap="viridis")
    assert image.dtype == np.uint8
    assert image.shape == (128, 40, 3)
    assert (width, height) == (40, 128)


def test_export_cc_writes_dataset(tmp_path):
    """cc export writes images + a label file per valid spectrogram + data.yaml;
    the duration==0 placeholder row is skipped."""
    h5_path = tmp_path / f"{_SESSION_ID}_spectrograms.h5"
    _write_session_h5(h5_path, durations=[40, 0, 60])  # row 1 is a placeholder
    out_dir = tmp_path / "yolo"

    YOLODatasetExporter(
        spectrogram_h5_paths=[str(h5_path)],
        output_directory=str(out_dir),
        input_parameter_dict=_cfg(),
        message_output=lambda *_a, **_kw: None,
    ).export()

    assert (out_dir / "data.yaml").is_file()
    images = sorted((out_dir / "images" / "train").glob("*.png"))
    labels = sorted((out_dir / "labels" / "train").glob("*.txt"))
    # Two valid USVs (rows 0 and 2); the placeholder row 1 is excluded.
    assert len(images) == 2
    assert len(labels) == 2
    assert {p.stem for p in images} == {f"{_SESSION_ID}_0", f"{_SESSION_ID}_2"}
    # Row 0's image is the 40-wide signal window.
    with Image.open(out_dir / "images" / "train" / f"{_SESSION_ID}_0.png") as img:
        assert img.size == (40, 128)  # PIL size is (width, height)


def test_export_manual_copies_labels(tmp_path):
    """manual export writes the hand-verified label lines verbatim."""
    h5_path = tmp_path / f"{_SESSION_ID}_spectrograms.h5"
    _write_session_h5(h5_path, durations=[40])
    manual_dir = tmp_path / "manual"
    manual_dir.mkdir()
    (manual_dir / f"{_SESSION_ID}_0.txt").write_text("0 0.5 0.5 0.3 0.2")
    out_dir = tmp_path / "yolo"

    YOLODatasetExporter(
        spectrogram_h5_paths=[str(h5_path)],
        output_directory=str(out_dir),
        input_parameter_dict=_cfg(label_source="manual", manual_labels_directory=str(manual_dir)),
        message_output=lambda *_a, **_kw: None,
    ).export()

    label = (out_dir / "labels" / "train" / f"{_SESSION_ID}_0.txt").read_text()
    assert label.strip() == "0 0.5 0.5 0.3 0.2"


def test_export_manual_without_directory_raises(tmp_path):
    """manual/merge require a manual_labels_directory."""
    h5_path = tmp_path / f"{_SESSION_ID}_spectrograms.h5"
    _write_session_h5(h5_path, durations=[40])
    with pytest.raises(ValueError, match="manual_labels_directory"):
        YOLODatasetExporter(
            spectrogram_h5_paths=[str(h5_path)],
            output_directory=str(tmp_path / "yolo"),
            input_parameter_dict=_cfg(label_source="manual"),
            message_output=lambda *_a, **_kw: None,
        ).export()
