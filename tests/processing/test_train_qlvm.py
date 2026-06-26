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
train -> infer bridge holds end to end. The full-scale GPU training run is not
exercised here.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
import torch

from usv_playpen.processing.qlvm_model import decode_lattice_atlas
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
