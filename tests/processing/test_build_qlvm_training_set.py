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
    rng = np.random.default_rng(abs(hash(session_id)) % (2**32))
    with h5py.File(path, "w") as f:
        f.create_dataset("specs", data=rng.random((n, n_f, n_t)).astype(np.float32))
        f.create_dataset("durations", data=np.full(n, n_t, dtype=np.int64))
        f.create_dataset("freq_bins", data=np.linspace(30000.0, 120000.0, n_f))
        f.create_dataset("spec_ids", data=np.array([f"{session_id}_{i}" for i in range(n)], dtype=h5py.string_dtype()))


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
