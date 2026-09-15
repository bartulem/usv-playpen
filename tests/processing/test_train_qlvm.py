"""
@author: bartulem
Tests for processing/train_qlvm.QLVMTrainer + the decoder helpers.

Synthesizes a tiny ``.npz`` training set (a handful of 128x128 spectrograms, the
exact shape ``build_qlvm_training_set`` writes), runs a 2-epoch CPU training run
with a small lattice, and checks that (1) the torch checkpoint and the
decoder-weights ``.npz`` are written, (2) the weights carry the expected
``nn.Sequential`` ``state_dict`` keys, and (3) the exported weights load straight
into the torch-free JAX inference decoder (``processing/qlvm_model``) and decode a
lattice to ``(K, 1, 128, 128)`` reconstructions in ``[0, 1]`` -- i.e. the
train -> infer bridge holds end to end. It also checks the training contract
written beside the weights, the refusal of stale splits and of a set without
``metadata.npz``, and that a seeded torch decoder gives the same images and
lattice posterior through the torch model and the JAX inference port. The
full-scale GPU training run is not exercised here.
"""

from __future__ import annotations

import json
import os

import jax.numpy as jnp
import numpy as np
import pytest
import torch

from usv_playpen.processing.qlvm_model import (
    decode_lattice_atlas,
    decoder_forward,
    decoder_head,
    gen_korobov_basis,
    posterior_over_lattice,
    torus_basis_forward,
)
from usv_playpen.processing.qlvm_training.losses import binary_lp as torch_binary_lp
from usv_playpen.processing.qlvm_training.qmc_base import QMCLVM, TorusBasis
from usv_playpen.processing.train_qlvm import (
    QLVMTrainer,
    build_lattice,
    build_qmc_decoder,
)

_TINY_CFG = {
    "train_qlvm": {
        "n_epochs": 2,
        "latent_dim": 2,
        "lattice_type": "korobov",
        "korobov_a": 3,
        "train_n_points": 17,
        "test_n_points": 11,
        "fib_m": 5,
        "batch_size": 4,
        "learning_rate": 0.001,
        "val_freq": 1,
        "seed": 0,
        "num_workers": 0,
    }
}

# Decoder state_dict keys: Linear layers 0,1 + ConvTranspose layers 3,5,7,9.
_EXPECTED_WEIGHT_KEYS = ("0.weight", "0.bias", "1.weight", "3.weight", "5.weight", "7.weight", "9.weight")


def _write_training_npz(path, n_samples, *, seed=0):
    """Write a tiny train/val .npz in the build_qlvm_training_set layout
    (spectrograms (N,128,128) float32 in [0,1] + masks/masks_len/durations/spec_id)."""
    rng = np.random.default_rng(seed)
    specs = rng.random((n_samples, 128, 128)).astype(np.float32)
    np.savez(
        path,
        spectrograms=specs,
        masks=np.zeros_like(specs),
        masks_len=np.zeros(n_samples, dtype=np.int64),
        durations=np.full(n_samples, 128, dtype=np.int64),
        spec_id=np.array([f"sess_{i}" for i in range(n_samples)]),
    )


def _write_metadata_npz(dataset_dir, *, length_threshold=128.0, masking_type="sam"):
    """Write the metadata.npz sidecar build_qlvm_training_set puts beside the splits
    (only the keys the training contract reads, plus a few it always carries)."""
    np.savez(
        dataset_dir / "metadata.npz",
        length_threshold=length_threshold,
        validation_split=0.2,
        random_state=42,
        full_dataset=False,
        target_shape=np.array([128, 128]),
        time_stretch=False,
        masking_type=masking_type,
    )


def test_build_qmc_decoder_state_dict_keys():
    """The decoder exposes exactly the nn.Sequential keys the JAX inference path
    reconstructs, and maps a (G, 2*latent_dim) torus embedding to (G,1,128,128)."""
    decoder = build_qmc_decoder(latent_dim=2)
    keys = set(decoder.state_dict().keys())
    for key in _EXPECTED_WEIGHT_KEYS:
        assert key in keys
    out = decoder(torch.zeros((5, 4), dtype=torch.float32))  # 2*latent_dim == 4
    assert tuple(out.shape) == (5, 1, 128, 128)


def test_build_lattice_shapes():
    """Each lattice type returns a (n_points, latent_dim) tensor."""
    korobov = build_lattice("korobov", latent_dim=2, korobov_a=3, n_points=17, fib_m=5)
    roberts = build_lattice("roberts", latent_dim=2, korobov_a=3, n_points=17, fib_m=5)
    assert korobov.shape[1] == 2
    assert roberts.shape == (17, 2)


def test_train_writes_checkpoint_and_bridge_weights(tmp_path, mocker):
    """A short CPU run writes the checkpoint + decoder-weights .npz, and the
    exported weights load straight into the JAX inference decoder."""
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    _write_training_npz(dataset_dir / "train_data.npz", n_samples=12, seed=0)
    _write_training_npz(dataset_dir / "val_data.npz", n_samples=4, seed=1)
    _write_metadata_npz(dataset_dir, length_threshold=90.0)
    output_dir = tmp_path / "model"

    mocker.patch("usv_playpen.processing.train_qlvm.smart_wait")
    QLVMTrainer(
        dataset_directory=str(dataset_dir),
        output_directory=str(output_dir),
        input_parameter_dict=_TINY_CFG,
        message_output=lambda *_a, **_kw: None,
    ).train()

    checkpoint_path = output_dir / "qmc_train_qlvm.tar"
    weights_path = output_dir / "qmc_decoder_weights.npz"
    assert checkpoint_path.is_file()
    assert weights_path.is_file()

    # The training contract sits beside the weights and records the decoder and the
    # dataset's preprocessing, which infer-qlvm-latents checks its settings against.
    contract = json.loads((output_dir / "qmc_decoder_weights.json").read_text())
    assert contract["decoder_head"] == "legacy"
    assert contract["c_dim"] == 0
    assert contract["condition"] is None
    assert contract["latent_dim"] == _TINY_CFG["train_qlvm"]["latent_dim"]
    assert contract["input_normalization"] == "none"
    assert contract["floor"] is None
    assert contract["masking_type"] == "sam"
    assert contract["target_shape"] == [128, 128]
    assert contract["time_stretch"] is False
    assert contract["length_threshold"] == 90.0
    assert contract["lattice_type"] == "korobov"
    assert contract["train_n_points"] == _TINY_CFG["train_qlvm"]["train_n_points"]

    # Bridge: the exported weights carry the expected keys and decode through the
    # torch-free JAX inference path to correctly-shaped reconstructions in [0, 1].
    with np.load(weights_path) as weights:
        for key in _EXPECTED_WEIGHT_KEYS:
            assert key in weights.files
        params = {key: jnp.asarray(weights[key]) for key in weights.files}

    lattice = jnp.asarray(np.random.default_rng(0).random((5, 2)), dtype=jnp.float32)
    atlas = decode_lattice_atlas(lattice, params)
    assert tuple(atlas.shape) == (5, 1, 128, 128)
    assert np.all(np.isfinite(np.asarray(atlas)))
    assert float(atlas.min()) >= 0.0
    assert float(atlas.max()) <= 1.0


def test_jax_inference_matches_torch_decoder_and_posterior():
    """The JAX port stands in for the torch model at inference, so the same weights
    must give the same images and the same lattice posterior in both: a seeded torch
    decoder is exported the way train-qlvm exports it and run through both paths."""
    torch.manual_seed(0)
    decoder = build_qmc_decoder(latent_dim=2).eval()
    params = {key: jnp.asarray(value.detach().numpy()) for key, value in decoder.state_dict().items()}
    model = QMCLVM(latent_dim=2, device=torch.device("cpu"), decoder=decoder, basis=TorusBasis())

    lattice_np = np.array(gen_korobov_basis(a=76, num_dims=2, num_points=1021), dtype=np.float32)
    lattice_torch = torch.from_numpy(lattice_np)

    # Decoder: identical inputs (the torus basis of the wrapped lattice) through both.
    basis_np = np.array(torus_basis_forward(jnp.asarray(lattice_np % 1)), dtype=np.float32)
    with torch.no_grad():
        images_torch = decoder(torch.from_numpy(basis_np)).numpy()
    images_jax = np.asarray(decoder_forward(jnp.asarray(basis_np), params))
    assert images_jax.shape == images_torch.shape == (1021, 1, 128, 128)
    np.testing.assert_allclose(images_jax, images_torch, atol=1e-5)

    # Posterior over the lattice for binarized decoded images of a few lattice points
    # (distinct likelihoods, so the comparison is not of two flat posteriors).
    rows = np.array([3, 250, 511, 1000])
    data_np = (images_torch[rows] > np.median(images_torch[rows], axis=(1, 2, 3), keepdims=True)).astype(np.float32)
    with torch.no_grad():
        posterior_torch = model.posterior_probability(lattice_torch, torch.from_numpy(data_np), torch_binary_lp).numpy()
    posterior_jax = np.asarray(posterior_over_lattice(jnp.asarray(images_jax), jnp.asarray(data_np)))
    assert posterior_jax.shape == posterior_torch.shape == (4, 1021)
    # Each posterior normalizes 16,384 summed float32 log-likelihoods per lattice point,
    # whose rounding differs across platforms (up to 1.14e-4 on macOS runners).
    np.testing.assert_allclose(posterior_jax, posterior_torch, atol=5e-4)
    assert np.array_equal(posterior_jax.argmax(axis=1), posterior_torch.argmax(axis=1))


def test_jax_decoder_runs_the_relu_head_like_torch():
    """QLVM model package decoders put a ReLU between the two Linear layers, shifting
    every later state_dict index by one; the JAX decoder must detect that head from
    the keys and reproduce torch's output."""
    torch.manual_seed(1)
    relu_decoder = torch.nn.Sequential(
        torch.nn.Linear(4, 2048),
        torch.nn.ReLU(),
        torch.nn.Linear(2048, 64 * 8 * 8),
        torch.nn.Unflatten(1, (64, 8, 8)),
        torch.nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
        torch.nn.ReLU(),
        torch.nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
        torch.nn.ReLU(),
        torch.nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
        torch.nn.ReLU(),
        torch.nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1),
        torch.nn.Sigmoid(),
    ).eval()
    params = {key: jnp.asarray(value.detach().numpy()) for key, value in relu_decoder.state_dict().items()}
    assert decoder_head(params) == "relu"
    assert decoder_head(dict(build_qmc_decoder(latent_dim=2).state_dict())) == "legacy"

    basis_np = np.random.default_rng(3).uniform(-1.0, 1.0, size=(64, 4)).astype(np.float32)
    with torch.no_grad():
        images_torch = relu_decoder(torch.from_numpy(basis_np)).numpy()
    images_jax = np.asarray(decoder_forward(jnp.asarray(basis_np), params))
    np.testing.assert_allclose(images_jax, images_torch, atol=1e-5)


def test_build_lattice_fib_requires_2d():
    """The Fibonacci lattice is 2D only; latent_dim != 2 raises rather than
    silently producing a lattice that mismatches the decoder input width."""
    with pytest.raises(ValueError, match="Fibonacci"):
        build_lattice("fibonacci", latent_dim=3, korobov_a=3, n_points=17, fib_m=5)


def test_train_full_dataset_no_val(tmp_path, mocker):
    """With only full_data.npz (no val split) the run still writes both artifacts
    and skips validation cleanly."""
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    _write_training_npz(dataset_dir / "full_data.npz", n_samples=8, seed=0)
    _write_metadata_npz(dataset_dir)
    output_dir = tmp_path / "model"

    mocker.patch("usv_playpen.processing.train_qlvm.smart_wait")
    QLVMTrainer(
        dataset_directory=str(dataset_dir),
        output_directory=str(output_dir),
        input_parameter_dict=_TINY_CFG,
        message_output=lambda *_a, **_kw: None,
    ).train()

    assert (output_dir / "qmc_train_qlvm.tar").is_file()
    assert (output_dir / "qmc_decoder_weights.npz").is_file()


def test_train_rejects_zero_val_freq(tmp_path, mocker):
    """val_freq < 1 raises (it gates `epoch % val_freq`), before touching data."""
    cfg = {"train_qlvm": {**_TINY_CFG["train_qlvm"], "val_freq": 0}}
    mocker.patch("usv_playpen.processing.train_qlvm.smart_wait")
    with pytest.raises(ValueError, match="val_freq"):
        QLVMTrainer(
            dataset_directory=str(tmp_path),
            output_directory=str(tmp_path / "out"),
            input_parameter_dict=cfg,
            message_output=lambda *_a, **_kw: None,
        ).train()


def test_train_missing_dataset_raises(tmp_path):
    """A dataset directory without train_data.npz/full_data.npz raises FileNotFoundError."""
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="No training set found"):
        QLVMTrainer(
            dataset_directory=str(tmp_path / "empty"),
            output_directory=str(tmp_path / "out"),
            input_parameter_dict=_TINY_CFG,
            message_output=lambda *_a, **_kw: None,
        ).train()


def test_train_refuses_splits_older_than_full_data(tmp_path, mocker):
    """A set rebuilt with full_dataset keeps its old splits, and train_data.npz would
    win the lookup: a split older than full_data.npz is refused before training."""
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    _write_training_npz(dataset_dir / "train_data.npz", n_samples=4, seed=0)
    _write_training_npz(dataset_dir / "val_data.npz", n_samples=4, seed=1)
    _write_training_npz(dataset_dir / "full_data.npz", n_samples=8, seed=2)
    _write_metadata_npz(dataset_dir)
    newest = (dataset_dir / "full_data.npz").stat().st_mtime
    for split in ("train_data.npz", "val_data.npz"):
        os.utime(dataset_dir / split, (newest - 60, newest - 60))

    mocker.patch("usv_playpen.processing.train_qlvm.smart_wait")
    with pytest.raises(ValueError, match=r"predate full_data\.npz"):
        QLVMTrainer(
            dataset_directory=str(dataset_dir),
            output_directory=str(tmp_path / "out"),
            input_parameter_dict=_TINY_CFG,
            message_output=lambda *_a, **_kw: None,
        ).train()
    assert not (tmp_path / "out").exists()


def test_train_missing_metadata_raises(tmp_path, mocker):
    """Without metadata.npz the training contract cannot record the set's
    preprocessing, so the run stops before training."""
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    _write_training_npz(dataset_dir / "full_data.npz", n_samples=8, seed=0)

    mocker.patch("usv_playpen.processing.train_qlvm.smart_wait")
    with pytest.raises(FileNotFoundError, match=r"No metadata\.npz"):
        QLVMTrainer(
            dataset_directory=str(dataset_dir),
            output_directory=str(tmp_path / "out"),
            input_parameter_dict=_TINY_CFG,
            message_output=lambda *_a, **_kw: None,
        ).train()
