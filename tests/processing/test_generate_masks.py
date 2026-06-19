"""
@author: bartulem
Tests for processing/generate_masks.MaskGenerator + flatten_session_masks.

The live SAM2/YOLO segmentation needs a GPU plus model checkpoints, so it is not
exercised here. Instead the heavy leaf kernels (``_common_memory.setup_device``,
``boxprompt_utils.build_predictor`` / ``process_session_batch_boxprompt`` and
``box_detectors.detect.get_detector``) are replaced with light fakes injected into
``sys.modules`` -- importing ``torch`` / ``sam2`` / ``ultralytics`` is therefore
avoided -- and the test verifies the usv-playpen-native orchestration around them:
the ``mask/<session>`` group is written into the per-session spectrogram H5 with the
right dataset shapes/dtypes and a ``spectrogram_index`` that maps every mask back to
the usv_summary row it belongs to (placeholder ``duration == 0`` rows getting none).
"""

from __future__ import annotations

import sys
import types

import h5py
import numpy as np
import pytest

from usv_playpen.processing.generate_masks import MaskGenerator, flatten_session_masks

_SESSION_ID = "20230119_155302"
_N_FREQ = 128
_N_TIME = 128


def _base_settings(tmp_path):
    """Build a ``generate_masks`` settings dict pointing at dummy (existing) model
    paths so the up-front path validation passes without real checkpoints."""
    sam2_dir = tmp_path / "sam2_model_dir"
    sam2_dir.mkdir()
    ckpt = sam2_dir / "sam2.pt"
    ckpt.write_bytes(b"not-a-real-checkpoint")
    yolo = tmp_path / "best.pt"
    yolo.write_bytes(b"not-a-real-yolo")
    return {
        "generate_masks": {
            "method": "boxprompt",
            "detector": "yolo",
            "sam2_model_dir": str(sam2_dir),
            "sam2_model_cfg": "sam2_hiera_s.yaml",
            "sam2_model_path": str(ckpt),
            "yolo_weights": str(yolo),
            "yolo_conf": 0.25,
            "yolo_iou": 0.7,
            "yolo_imgsz": 128,
            "mask_cmap": "viridis",
            "duration_min": 10,
            "batch_size": 12,
            "multimask_output": True,
            "iou_floor": 0.7,
            "drop_below_iou": False,
            "split_disconnected": True,
            "max_iters": 1,
            "merge_instances": True,
            "merge_iou": 0.5,
            "merge_containment": 0.8,
            "mask_intensity_floor": 0.0,
            "tiny_mask_floor_px": 12,
            "min_box_area": 0,
        }
    }


def _make_session_h5(tmp_path, durations):
    """Create a session dir holding a per-session spectrogram H5 in the
    consolidated-store layout (top-level ``frequency_bins`` + ``spectrogram/<session>``
    group with ``spectrograms`` and ``durations``). Returns the session root path."""
    root = tmp_path / _SESSION_ID
    spec_dir = root / "audio" / "spectrograms"
    spec_dir.mkdir(parents=True)

    n = len(durations)
    rng = np.random.default_rng(0)
    specs = rng.random((n, _N_FREQ, _N_TIME)).astype(np.float32)
    freq_bins = np.linspace(30000.0, 120000.0, _N_FREQ)

    h5_path = spec_dir / f"{_SESSION_ID}_spectrograms.h5"
    with h5py.File(h5_path, "w") as h5_file:
        h5_file.attrs["created"] = "generate_spectrograms"
        h5_file.attrs["total_spectrograms"] = n
        h5_file.attrs["valid_spectrograms"] = int(np.count_nonzero(np.asarray(durations) > 0))
        h5_file.create_dataset("frequency_bins", data=freq_bins)
        grp = h5_file.create_group(f"spectrogram/{_SESSION_ID}")
        grp.create_dataset("spectrograms", data=specs)
        grp.create_dataset("durations", data=np.asarray(durations, dtype=np.int64))
    return root


def _seg(width, *, rows=(10, 20)):
    """A boolean ``(F, width)`` segmentation with a couple of True frequency rows."""
    seg = np.zeros((_N_FREQ, width), dtype=bool)
    for r in rows:
        seg[r, :] = True
    return seg


def _install_fake_kernels(monkeypatch, canned_masks):
    """Replace the heavy mask leaf modules with light fakes so no GPU library is
    imported. ``process_session_batch_boxprompt`` returns ``canned_masks`` verbatim."""
    common = types.ModuleType("usv_playpen.processing.masks._common_memory")
    common.setup_device = lambda **_kwargs: "cpu"

    boxprompt = types.ModuleType("usv_playpen.processing.masks.boxprompt_utils")
    boxprompt.build_predictor = lambda **_kwargs: object()

    def _fake_process(session_name, *_args, **_kwargs):
        return session_name, canned_masks

    boxprompt.process_session_batch_boxprompt = _fake_process

    detect = types.ModuleType("usv_playpen.processing.masks.box_detectors.detect")
    detect.get_detector = lambda *_args, **_kwargs: (lambda *_a, **_k: [])

    monkeypatch.setitem(sys.modules, "usv_playpen.processing.masks._common_memory", common)
    monkeypatch.setitem(sys.modules, "usv_playpen.processing.masks.boxprompt_utils", boxprompt)
    monkeypatch.setitem(sys.modules, "usv_playpen.processing.masks.box_detectors.detect", detect)


def test_flatten_session_masks_packs_and_indexes():
    """flatten_session_masks pads each mask to (F, T), records the owning spec row,
    and preserves only the valid-width columns."""
    canned = {
        0: [{"segmentation": _seg(40)}, {"segmentation": _seg(40, rows=(5,))}],
        1: [],
        2: [{"segmentation": _seg(60)}],
    }
    segmentations, spectrogram_index = flatten_session_masks(canned, _N_FREQ, _N_TIME)

    assert segmentations.shape == (3, _N_FREQ, _N_TIME)
    assert segmentations.dtype == np.bool_
    assert spectrogram_index.tolist() == [0, 0, 2]
    # Row 0's first mask spans 40 columns; everything at/after 40 is padding.
    assert not segmentations[0, :, 40:].any()
    assert segmentations[0, 10, :40].all()
    # Cropping back to the spec duration recovers the original segmentation.
    assert np.array_equal(segmentations[2, :, :60], _seg(60))


def test_flatten_session_masks_empty():
    """With no masks the datasets are correctly-shaped empties."""
    segmentations, spectrogram_index = flatten_session_masks({0: [], 1: []}, _N_FREQ, _N_TIME)
    assert segmentations.shape == (0, _N_FREQ, _N_TIME)
    assert segmentations.dtype == np.bool_
    assert spectrogram_index.shape == (0,)
    assert spectrogram_index.dtype == np.int64


def test_generate_session_masks_writes_mask_group(tmp_path, monkeypatch):
    """End-to-end (kernels faked): the mask group lands in the spectrogram H5 with
    the right shapes/dtype, masks index the right USV rows, and the duration==0
    placeholder row gets no masks."""
    durations = [40, 0, 60, 25]  # row 1 is a placeholder
    root = _make_session_h5(tmp_path, durations)

    # The faked kernel keys every row (valid -> masks, placeholder -> []) exactly as
    # the real process_session_batch_boxprompt does when passed the full stack.
    canned = {
        0: [{"segmentation": _seg(40)}, {"segmentation": _seg(40, rows=(7,))}],
        1: [],
        2: [{"segmentation": _seg(60)}],
        3: [{"segmentation": _seg(25)}],
    }
    _install_fake_kernels(monkeypatch, canned)

    messages = []
    MaskGenerator(
        root_directory=str(root),
        input_parameter_dict=_base_settings(tmp_path),
        message_output=messages.append,
    ).generate_session_masks()

    h5_path = root / "audio" / "spectrograms" / f"{_SESSION_ID}_spectrograms.h5"
    with h5py.File(h5_path, "r") as h5_file:
        assert f"mask/{_SESSION_ID}" in h5_file
        mask_grp = h5_file[f"mask/{_SESSION_ID}"]
        segmentations = mask_grp["segmentations"][:]
        spectrogram_index = mask_grp["spectrogram_index"][:]
        assert mask_grp.attrs["created"] == "generate_masks"
        assert int(mask_grp.attrs["total_masks"]) == 4
        # The spectrogram group is left untouched alongside the new mask group.
        assert f"spectrogram/{_SESSION_ID}" in h5_file

    assert segmentations.shape == (4, _N_FREQ, _N_TIME)
    assert segmentations.dtype == np.bool_
    assert spectrogram_index.tolist() == [0, 0, 2, 3]
    # The placeholder (duration == 0) USV row never receives a mask.
    assert 1 not in spectrogram_index.tolist()
    # Each mask only occupies its spec's valid-width columns.
    assert not segmentations[0, :, 40:].any()
    assert not segmentations[3, :, 25:].any()


def test_generate_session_masks_overwrites_existing_group(tmp_path, monkeypatch):
    """A second run replaces a pre-existing mask group rather than appending."""
    durations = [40, 60]
    root = _make_session_h5(tmp_path, durations)
    h5_path = root / "audio" / "spectrograms" / f"{_SESSION_ID}_spectrograms.h5"

    # Seed a stale mask group that the run must overwrite.
    with h5py.File(h5_path, "a") as h5_file:
        stale = h5_file.create_group(f"mask/{_SESSION_ID}")
        stale.create_dataset("segmentations", data=np.ones((9, _N_FREQ, _N_TIME), dtype=bool))
        stale.create_dataset("spectrogram_index", data=np.zeros((9,), dtype=np.int64))

    canned = {0: [{"segmentation": _seg(40)}], 1: [{"segmentation": _seg(60)}]}
    _install_fake_kernels(monkeypatch, canned)

    MaskGenerator(
        root_directory=str(root),
        input_parameter_dict=_base_settings(tmp_path),
        message_output=lambda *_: None,
    ).generate_session_masks()

    with h5py.File(h5_path, "r") as h5_file:
        spectrogram_index = h5_file[f"mask/{_SESSION_ID}"]["spectrogram_index"][:]
    assert spectrogram_index.tolist() == [0, 1]


def test_generate_session_masks_cc_detector_skips_yolo_weights(tmp_path, monkeypatch):
    """The cc detector path needs no YOLO weights: a blank yolo_weights is fine and
    the mask group is still written (detect_fn is None, handled by the kernel)."""
    durations = [40, 60]
    root = _make_session_h5(tmp_path, durations)
    settings = _base_settings(tmp_path)
    settings["generate_masks"]["detector"] = "cc"
    settings["generate_masks"]["yolo_weights"] = ""

    canned = {0: [{"segmentation": _seg(40)}], 1: [{"segmentation": _seg(60)}]}
    _install_fake_kernels(monkeypatch, canned)

    MaskGenerator(
        root_directory=str(root),
        input_parameter_dict=settings,
        message_output=lambda *_a, **_kw: None,
    ).generate_session_masks()

    h5_path = root / "audio" / "spectrograms" / f"{_SESSION_ID}_spectrograms.h5"
    with h5py.File(h5_path, "r") as h5_file:
        assert f"mask/{_SESSION_ID}" in h5_file


def test_generate_session_masks_missing_yolo_weights_raises(tmp_path):
    """With detector=yolo, missing YOLO weights raises FileNotFoundError up front."""
    root = _make_session_h5(tmp_path, [40, 60])
    settings = _base_settings(tmp_path)
    settings["generate_masks"]["yolo_weights"] = str(tmp_path / "does_not_exist.pt")
    with pytest.raises(FileNotFoundError, match="YOLO weights"):
        MaskGenerator(
            root_directory=str(root),
            input_parameter_dict=settings,
            message_output=lambda *_a, **_kw: None,
        ).generate_session_masks()


def test_generate_session_masks_missing_checkpoint_raises(tmp_path):
    """A missing SAM2 checkpoint raises FileNotFoundError before any GPU import."""
    root = _make_session_h5(tmp_path, [40, 60])
    settings = _base_settings(tmp_path)
    settings["generate_masks"]["sam2_model_path"] = str(tmp_path / "does_not_exist.pt")
    # No fakes installed: if validation did not short-circuit, the heavy import would.
    with pytest.raises(FileNotFoundError, match="SAM2 checkpoint"):
        MaskGenerator(
            root_directory=str(root),
            input_parameter_dict=settings,
            message_output=lambda *_: None,
        ).generate_session_masks()


def test_generate_session_masks_rejects_non_boxprompt(tmp_path):
    """Only the boxprompt method is supported; anything else raises ValueError."""
    root = _make_session_h5(tmp_path, [40, 60])
    settings = _base_settings(tmp_path)
    settings["generate_masks"]["method"] = "grid"
    with pytest.raises(ValueError, match="boxprompt"):
        MaskGenerator(
            root_directory=str(root),
            input_parameter_dict=settings,
            message_output=lambda *_: None,
        ).generate_session_masks()
