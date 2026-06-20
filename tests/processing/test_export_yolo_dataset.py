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


def _write_session_h5(tmp_path, durations):
    """Create a session root with audio/spectrograms/<session>_spectrograms.h5
    (consolidated layout, no mask group needed here) and return the session root."""
    rng = np.random.default_rng(0)
    n = len(durations)
    specs = rng.random((n, 128, 128)).astype(np.float32)
    root = tmp_path / _SESSION_ID
    spec_dir = root / "audio" / "spectrograms"
    spec_dir.mkdir(parents=True)
    with h5py.File(spec_dir / f"{_SESSION_ID}_spectrograms.h5", "w") as f:
        f.create_dataset("frequency_bins", data=np.linspace(30000.0, 120000.0, 128))
        grp = f.create_group(f"spectrogram/{_SESSION_ID}")
        grp.create_dataset("spectrograms", data=specs)
        grp.create_dataset("durations", data=np.asarray(durations, dtype=np.int64))
    return root


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
    root = _write_session_h5(tmp_path, durations=[40, 0, 60])  # row 1 is a placeholder
    out_dir = tmp_path / "yolo"

    YOLODatasetExporter(
        root_directories=[str(root)],
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
    root = _write_session_h5(tmp_path, durations=[40])
    manual_dir = tmp_path / "manual"
    manual_dir.mkdir()
    (manual_dir / f"{_SESSION_ID}_0.txt").write_text("0 0.5 0.5 0.3 0.2")
    out_dir = tmp_path / "yolo"

    YOLODatasetExporter(
        root_directories=[str(root)],
        output_directory=str(out_dir),
        input_parameter_dict=_cfg(label_source="manual", manual_labels_directory=str(manual_dir)),
        message_output=lambda *_a, **_kw: None,
    ).export()

    label = (out_dir / "labels" / "train" / f"{_SESSION_ID}_0.txt").read_text()
    assert label.strip() == "0 0.5 0.5 0.3 0.2"


def test_cc_labels_normalization():
    """_cc_labels turns detector (t,l,b,r) boxes into one normalized YOLO line each,
    with class 0 and all coordinates in [0, 1]."""
    exporter = YOLODatasetExporter(input_parameter_dict=_cfg())
    # Fake detector returning a single box (top=10, left=20, bottom=40, right=60).
    lines = exporter._cc_labels(lambda _spec, _dur: [(10, 20, 40, 60)], spec=None, duration=64, width=128, height=128)
    assert len(lines) == 1
    parts = lines[0].split()
    assert parts[0] == "0"
    coords = [float(v) for v in parts[1:]]
    assert len(coords) == 4
    assert all(0.0 <= c <= 1.0 for c in coords)


def test_export_merge_uses_manual_then_cc(tmp_path):
    """merge writes the hand-verified label where a {spec_id}.txt exists and falls
    back to cc pseudo-labels for spectrograms without one."""
    root = _write_session_h5(tmp_path, durations=[40, 60])  # rows 0 and 1
    manual_dir = tmp_path / "manual"
    manual_dir.mkdir()
    (manual_dir / f"{_SESSION_ID}_0.txt").write_text("0 0.4 0.4 0.2 0.2")  # only row 0 hand-labeled
    out_dir = tmp_path / "yolo"

    YOLODatasetExporter(
        root_directories=[str(root)],
        output_directory=str(out_dir),
        input_parameter_dict=_cfg(label_source="merge", manual_labels_directory=str(manual_dir)),
        message_output=lambda *_a, **_kw: None,
    ).export()

    # Row 0 -> verbatim manual; row 1 -> cc fallback (label file exists, possibly empty).
    assert (out_dir / "labels" / "train" / f"{_SESSION_ID}_0.txt").read_text().strip() == "0 0.4 0.4 0.2 0.2"
    assert (out_dir / "labels" / "train" / f"{_SESSION_ID}_1.txt").is_file()


def test_export_split_is_exact_fraction(tmp_path):
    """validation_split yields an exact, reproducible val count (not a per-image flip)."""
    root = _write_session_h5(tmp_path, durations=[10, 20, 30, 40])  # 4 valid USVs
    out_dir = tmp_path / "yolo"

    YOLODatasetExporter(
        root_directories=[str(root)],
        output_directory=str(out_dir),
        input_parameter_dict=_cfg(validation_split=0.5),
        message_output=lambda *_a, **_kw: None,
    ).export()

    assert len(list((out_dir / "images" / "val").glob("*.png"))) == 2
    assert len(list((out_dir / "images" / "train").glob("*.png"))) == 2


def test_export_validation_split_zero_routes_all_to_train(tmp_path):
    """validation_split=0.0 sends every image to train and leaves val empty."""
    root = _write_session_h5(tmp_path, durations=[10, 20, 30, 40])
    out_dir = tmp_path / "yolo"

    YOLODatasetExporter(
        root_directories=[str(root)],
        output_directory=str(out_dir),
        input_parameter_dict=_cfg(validation_split=0.0),
        message_output=lambda *_a, **_kw: None,
    ).export()

    assert len(list((out_dir / "images" / "train").glob("*.png"))) == 4
    assert len(list((out_dir / "images" / "val").glob("*.png"))) == 0


def test_export_validation_split_one_routes_all_to_val(tmp_path):
    """validation_split=1.0 sends every image to val and leaves train empty."""
    root = _write_session_h5(tmp_path, durations=[10, 20, 30, 40])
    out_dir = tmp_path / "yolo"

    YOLODatasetExporter(
        root_directories=[str(root)],
        output_directory=str(out_dir),
        input_parameter_dict=_cfg(validation_split=1.0),
        message_output=lambda *_a, **_kw: None,
    ).export()

    assert len(list((out_dir / "images" / "train").glob("*.png"))) == 0
    assert len(list((out_dir / "images" / "val").glob("*.png"))) == 4


def test_export_small_cohort_val_not_empty(tmp_path):
    """On a tiny cohort the seeded-permutation split still rounds to a non-empty val
    set (a per-image coin flip could leave it empty)."""
    root = _write_session_h5(tmp_path, durations=[10, 20, 30])  # 3 valid USVs
    out_dir = tmp_path / "yolo"

    YOLODatasetExporter(
        root_directories=[str(root)],
        output_directory=str(out_dir),
        input_parameter_dict=_cfg(validation_split=0.34),  # round(3 * 0.34) == 1
        message_output=lambda *_a, **_kw: None,
    ).export()

    assert len(list((out_dir / "images" / "val").glob("*.png"))) == 1
    assert len(list((out_dir / "images" / "train").glob("*.png"))) == 2


def test_export_manual_without_directory_raises(tmp_path):
    """manual/merge require a manual_labels_directory."""
    root = _write_session_h5(tmp_path, durations=[40])
    with pytest.raises(ValueError, match="manual_labels_directory"):
        YOLODatasetExporter(
            root_directories=[str(root)],
            output_directory=str(tmp_path / "yolo"),
            input_parameter_dict=_cfg(label_source="manual"),
            message_output=lambda *_a, **_kw: None,
        ).export()
