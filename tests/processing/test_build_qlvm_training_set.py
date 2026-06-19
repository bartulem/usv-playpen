"""
@author: bartulem
Tests for processing/build_qlvm_training_set.

Covers the index selection + resize helpers directly, and an end-to-end build
from two synthetic per-session spectrogram H5 files into train/val (and full)
``.npz`` outputs with the expected keys/shapes and provenance preserved.
"""

from __future__ import annotations

import h5py
import numpy as np

from usv_playpen.processing.build_qlvm_training_set import (
    QLVMTrainingSetBuilder,
    compute_selected_indices,
    stretch_specs,
)

_CFG = {
    "length_threshold": 50.0,
    "dataset_size_constraint": None,
    "validation_split": 0.5,
    "random_state": 42,
    "full_dataset": False,
    "target_shape": [128, 128],
    "time_stretch": False,
    "masking_type": "none",
}


def test_compute_selected_indices_length_filter():
    """Indices with duration >= threshold are dropped; the rest are sorted."""
    durs = {"s": np.array([10, 60, 20, 200, 5])}
    sel = compute_selected_indices(durs, length_threshold=50.0, dataset_size_constraint=None, random_state=0)
    assert sel["s"].tolist() == [0, 2, 4]


def test_compute_selected_indices_subsample_is_capped_and_reproducible():
    """An absolute size constraint caps per-session count and is reproducible."""
    durs = {"a": np.arange(10), "b": np.arange(10)}  # all < threshold
    a = compute_selected_indices(durs, 50.0, dataset_size_constraint=8, random_state=1)
    b = compute_selected_indices(durs, 50.0, dataset_size_constraint=8, random_state=1)
    assert len(a["a"]) == 4  # 8 total / 2 sessions
    assert len(a["b"]) == 4
    assert a["a"].tolist() == b["a"].tolist()


def test_stretch_specs_outputs_target_shape():
    """Every spectrogram is resized to target_shape regardless of input size."""
    specs = np.random.default_rng(0).random((3, 64, 90)).astype(np.float32)
    durations = np.array([90, 45, 10])
    out = stretch_specs(specs, durations, (128, 128), time_stretch=False)
    assert out.shape == (3, 128, 128)
    out_ts = stretch_specs(specs, durations, (128, 128), time_stretch=True)
    assert out_ts.shape == (3, 128, 128)


def _write_session_h5(path, session_id, n, n_f=32, n_t=40):
    """Consolidated layout: top-level ``frequency_bins`` + a ``spectrogram/<session>``
    group whose ``spectrograms`` rows are 1:1 with usv_summary (all real here)."""
    rng = np.random.default_rng(abs(hash(session_id)) % (2**32))
    with h5py.File(path, "w") as f:
        f.create_dataset("frequency_bins", data=np.linspace(30000.0, 120000.0, n_f))
        session_group = f.create_group(f"spectrogram/{session_id}")
        session_group.create_dataset("spectrograms", data=rng.random((n, n_f, n_t)).astype(np.float32))
        session_group.create_dataset("durations", data=np.full(n, n_t, dtype=np.int64))


def test_build_train_val_npz(tmp_path, mocker):
    """End-to-end train/val build: two sessions -> train_data.npz + val_data.npz
    + metadata.npz, with the expected keys, target shape, and zero masks."""
    h5a = tmp_path / "20230119_155302_spectrograms.h5"
    h5b = tmp_path / "20230119_162529_spectrograms.h5"
    _write_session_h5(h5a, "20230119_155302", n=6)
    _write_session_h5(h5b, "20230119_162529", n=6)
    out_dir = tmp_path / "out"

    mocker.patch("usv_playpen.processing.build_qlvm_training_set.smart_wait")
    QLVMTrainingSetBuilder(
        spectrogram_h5_paths=[str(h5a), str(h5b)],
        output_directory=str(out_dir),
        input_parameter_dict={"build_qlvm_training_set": _CFG},
        message_output=lambda *_a, **_kw: None,
    ).build()

    train = np.load(out_dir / "train_data.npz", allow_pickle=True)
    val = np.load(out_dir / "val_data.npz", allow_pickle=True)
    assert set(train.files) == {"spectrograms", "masks", "masks_len", "durations", "spec_id"}
    assert train["spectrograms"].shape[1:] == (128, 128)
    assert train["spectrograms"].shape[0] + val["spectrograms"].shape[0] == 12
    # mask-free: masks all zero.
    assert not train["masks"].any()
    assert (train["masks_len"] == 0).all()
    # provenance carried.
    assert any(s.startswith("20230119_") for s in train["spec_id"].tolist())
    assert (out_dir / "metadata.npz").is_file()


def test_build_full_dataset_single_npz(tmp_path, mocker):
    """full_dataset mode writes one full_data.npz with all kept samples."""
    h5a = tmp_path / "20230119_155302_spectrograms.h5"
    _write_session_h5(h5a, "20230119_155302", n=5)
    out_dir = tmp_path / "out_full"

    mocker.patch("usv_playpen.processing.build_qlvm_training_set.smart_wait")
    cfg = {**_CFG, "full_dataset": True}
    QLVMTrainingSetBuilder(
        spectrogram_h5_paths=[str(h5a)],
        output_directory=str(out_dir),
        input_parameter_dict={"build_qlvm_training_set": cfg},
        message_output=lambda *_a, **_kw: None,
    ).build()

    assert (out_dir / "full_data.npz").is_file()
    assert not (out_dir / "train_data.npz").exists()
    full = np.load(out_dir / "full_data.npz", allow_pickle=True)
    assert full["spectrograms"].shape == (5, 128, 128)


def _write_session_h5_with_masks(path, session_id, n, n_f=32, n_t=40):
    """Like _write_session_h5 but also writes a mask/<session> group: row 0 gets
    two mask instances (a shared region), the remaining rows get none (so they
    fall back to an all-ones mask)."""
    rng = np.random.default_rng(abs(hash(session_id)) % (2**32))
    seg0 = np.zeros((n_f, n_t), dtype=bool)
    seg0[5:10, 2:8] = True
    seg1 = np.zeros((n_f, n_t), dtype=bool)
    seg1[12:15, 3:6] = True
    with h5py.File(path, "w") as f:
        f.create_dataset("frequency_bins", data=np.linspace(30000.0, 120000.0, n_f))
        session_group = f.create_group(f"spectrogram/{session_id}")
        session_group.create_dataset("spectrograms", data=rng.random((n, n_f, n_t)).astype(np.float32))
        session_group.create_dataset("durations", data=np.full(n, n_t, dtype=np.int64))
        mask_group = f.create_group(f"mask/{session_id}")
        mask_group.create_dataset("segmentations", data=np.stack([seg0, seg1], axis=0))
        mask_group.create_dataset("spectrogram_index", data=np.array([0, 0], dtype=np.int64))


def test_build_sam_masking_applies_masks_and_counts(tmp_path, mocker):
    """masking_type='sam' writes binary masks, masks the spectrograms (zeroed
    outside the mask), and records per-row instance counts; rows with no detected
    mask fall back to an all-ones mask."""
    h5a = tmp_path / "20230119_155302_spectrograms.h5"
    _write_session_h5_with_masks(h5a, "20230119_155302", n=3)
    out_dir = tmp_path / "out_sam"

    mocker.patch("usv_playpen.processing.build_qlvm_training_set.smart_wait")
    cfg = {**_CFG, "full_dataset": True, "masking_type": "sam"}
    QLVMTrainingSetBuilder(
        spectrogram_h5_paths=[str(h5a)],
        output_directory=str(out_dir),
        input_parameter_dict={"build_qlvm_training_set": cfg},
        message_output=lambda *_a, **_kw: None,
    ).build()

    full = np.load(out_dir / "full_data.npz", allow_pickle=True)
    masks = full["masks"]
    specs = full["spectrograms"]
    # Masks are binary, and the spectrogram is zeroed everywhere the mask is zero.
    assert set(np.unique(masks).tolist()).issubset({0.0, 1.0})
    assert np.all(specs[masks == 0] == 0)
    # Row 0 had two mask instances; the fallback rows have a zero instance count.
    assert full["masks_len"].tolist() == [2, 0, 0]
    # The fallback rows keep signal (all-ones mask over the signal window).
    assert specs[1].any()


def test_build_sam_masking_without_mask_group_falls_back(tmp_path, mocker):
    """masking_type='sam' over a session with NO mask/<session> group gives every
    row an all-ones mask (spectrogram preserved, zero instance counts)."""
    h5a = tmp_path / "20230119_155302_spectrograms.h5"
    _write_session_h5(h5a, "20230119_155302", n=4)  # writes no mask group
    out_dir = tmp_path / "out_nomask"

    mocker.patch("usv_playpen.processing.build_qlvm_training_set.smart_wait")
    cfg = {**_CFG, "full_dataset": True, "masking_type": "sam"}
    QLVMTrainingSetBuilder(
        spectrogram_h5_paths=[str(h5a)],
        output_directory=str(out_dir),
        input_parameter_dict={"build_qlvm_training_set": cfg},
        message_output=lambda *_a, **_kw: None,
    ).build()

    full = np.load(out_dir / "full_data.npz", allow_pickle=True)
    assert (full["masks_len"] == 0).all()          # no detected instances
    assert full["spectrograms"].any()              # signal preserved (all-ones mask)
    assert set(np.unique(full["masks"]).tolist()).issubset({0.0, 1.0})
