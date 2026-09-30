"""
@author: bartulem
Tests for processing/train_qlvm (the JAX port of the shipped QLVM training recipe)
and the model half of training in processing/qlvm_model.

Synthesizes a tiny ``.npz`` training set in the model-package layout (a few dozen
128x128 spectrograms, a ``metadata.npz``), trains on the CPU for a handful of
epochs with small Fibonacci lattices, and checks that (1) the training loss
decreases, (2) ``checkpoint.tar`` has the structure of a shipped model package
checkpoint, loads with ``torch.load`` into the reference torch decoder, and loads
without torch through the inference path (``read_torch_checkpoint``,
``load_decoder_params``, ``load_model_cell``), (3) the training contract passes
``enforce_training_contract`` and embeds calls, and (4) the data preparation
(min-max, masking, the validation subset) follows the reference trainer. The
objective and its gradients, the decoder initialization, and the JAX inference
decoder and posterior are checked against torch transcriptions of the reference
code. The full-scale GPU run is not exercised here.
"""

from __future__ import annotations

import json
import os
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from usv_playpen.processing.qlvm_latents import (
    enforce_training_contract,
    load_decoder_params,
    load_model_cell,
    minmax_per_spectrogram,
    read_torch_checkpoint,
)
from usv_playpen.processing.qlvm_model import (
    binary_evidence,
    decode_shifted_lattice,
    decoder_forward,
    decoder_head,
    decoder_parameter_shapes,
    embed_data,
    gen_fib_basis,
    init_decoder_params,
    posterior_over_lattice,
    torus_basis_forward,
)

# train_qlvm pulls optax -> a one-time JAX DeprecationWarning at import.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from usv_playpen.processing.train_qlvm import (
        QLVMTrainer,
        load_split,
        validation_subset_indices,
    )

_TINY_CFG = {
    "train_qlvm": {
        "n_epochs": 6,
        "decoder_head": "relu",
        "training_fib_m": 6,
        "validation_fib_m": 7,
        "embedding_fib_m": 8,
        "batch_size": 8,
        "learning_rate": 0.001,
        "val_freq": 3,
        "val_samples_per_mask_count": 4,
        "seed": 0,
    }
}

_INFERENCE_CFG = {
    "masking_type": "none",
    "time_stretch": False,
    "latent_dim": 2,
    "target_shape": [128, 128],
    "length_threshold": None,
}


def _torch_decoder(head):
    """The reference torch decoder (qmc_deep_gen's build_qmc_decoder, unconditional, 2-D torus)."""
    front = [torch.nn.Linear(4, 2048)]
    if head == "relu":
        front.append(torch.nn.ReLU())
    front.append(torch.nn.Linear(2048, 64 * 8 * 8))
    return torch.nn.Sequential(
        *front,
        torch.nn.Unflatten(1, (64, 8, 8)),
        torch.nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
        torch.nn.ReLU(),
        torch.nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
        torch.nn.ReLU(),
        torch.nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
        torch.nn.ReLU(),
        torch.nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1),
        torch.nn.Sigmoid(),
    )


def _torch_binary_lp(samples, data):
    """Transcription of the reference train/losses.py:binary_lp (no importance weights)."""
    samples = torch.clamp(samples, min=1e-6, max=1 - 1e-6)
    t1 = torch.einsum("bjdl,sjdl->bs", data, torch.log(samples))
    t2 = torch.einsum("bjdl,sjdl->bs", 1 - data, torch.log(1 - samples))
    return t1 + t2


def _torch_binary_evidence(samples, data):
    """Transcription of the reference train/losses.py:binary_evidence with its defaults."""
    log_evidence = torch.special.logsumexp(_torch_binary_lp(samples, data), axis=1) - np.log(samples.shape[0])
    return -1 * torch.mean(log_evidence)


def _blob_spectrograms(n_samples, rng):
    """(N, 128, 128) float32 spectrograms: one bright Gaussian blob on a dim noisy
    background, at a random place -- structure a decoder can learn in a few steps."""
    grid = np.arange(128, dtype=np.float32)
    rows, cols = rng.uniform(20, 108, size=(2, n_samples))
    blobs = np.exp(-(((grid[None, :, None] - rows[:, None, None]) ** 2) + ((grid[None, None, :] - cols[:, None, None]) ** 2)) / 60.0)
    return (blobs + 0.05 * rng.random((n_samples, 128, 128))).astype(np.float32)


def _write_split(path, n_samples, *, seed, apply_mask=None, masks=None, masks_len=None):
    """Write one split in the model-package layout; ``apply_mask`` None leaves the key out."""
    rng = np.random.default_rng(seed)
    specs = _blob_spectrograms(n_samples, rng)
    arrays = {
        "spectrograms": specs,
        "masks": np.ones_like(specs) if masks is None else masks,
        "masks_len": rng.integers(1, 4, n_samples) if masks_len is None else masks_len,
        "durations": rng.integers(8, 120, n_samples),
        "spec_id": np.array([f"sess_{i}" for i in range(n_samples)]),
    }
    if apply_mask is not None:
        arrays["apply_mask"] = np.array(apply_mask)
    np.savez(path, **arrays)


def _write_metadata(dataset_dir, *, masking_type="sam", require_mask=True, floor=0.2, length_threshold=128.0):
    """Write metadata.npz; ``require_mask`` / ``floor`` None leave those keys out."""
    metadata = {
        "length_threshold": length_threshold,
        "target_shape": np.array([128, 128]),
        "time_stretch": False,
        "masking_type": masking_type,
    }
    if require_mask is not None:
        metadata["require_mask"] = require_mask
    if floor is not None:
        metadata["floor"] = floor
    np.savez(dataset_dir / "metadata.npz", **metadata)


def _train(dataset_dir, output_dir, cfg, mocker):
    """Run QLVMTrainer quietly."""
    mocker.patch("usv_playpen.processing.train_qlvm.smart_wait")
    QLVMTrainer(
        dataset_directory=str(dataset_dir),
        output_directory=str(output_dir),
        input_parameter_dict=cfg,
        message_output=lambda *_a, **_kw: None,
    ).train()


@pytest.fixture(scope="module")
def trained_cell(tmp_path_factory):
    """One short CPU training run on an unmasked, floored set (the phase 6 layout),
    shared by the tests that read its outputs."""
    root = tmp_path_factory.mktemp("qlvm_train")
    dataset_dir = root / "dataset"
    dataset_dir.mkdir()
    _write_split(dataset_dir / "train_data.npz", 48, seed=0, apply_mask=False)
    _write_split(dataset_dir / "val_data.npz", 16, seed=1, apply_mask=False)
    _write_metadata(dataset_dir)
    output_dir = root / "cell"
    QLVMTrainer(
        dataset_directory=str(dataset_dir),
        output_directory=str(output_dir),
        input_parameter_dict=_TINY_CFG,
        message_output=lambda *_a, **_kw: None,
    ).train()
    return output_dir


def test_decoder_parameter_shapes_match_the_torch_decoder():
    """Both heads' keys, shapes and parameter order are those of the reference torch
    decoder's state_dict, which is what the checkpoint and the Adam state index by."""
    for head in ("relu", "legacy"):
        torch_state = _torch_decoder(head).state_dict()
        shapes = decoder_parameter_shapes(latent_dim=2, c_dim=0, head=head)
        assert list(shapes) == list(torch_state)
        assert all(shapes[name] == tuple(torch_state[name].shape) for name in shapes)
    with pytest.raises(ValueError, match="head"):
        decoder_parameter_shapes(latent_dim=2, c_dim=0, head="harmonic")


def test_init_decoder_params_follows_torch_default_init():
    """Every weight and bias is U(-1/sqrt(fan_in), 1/sqrt(fan_in)) with torch's fan_in
    (the input width for Linear, out_channels * 9 for ConvTranspose2d)."""
    params = init_decoder_params(jax.random.PRNGKey(0), latent_dim=2, c_dim=0, head="relu")
    assert decoder_head(params) == "relu"
    fan_ins = {"0": 4, "2": 2048, "4": 32 * 9, "6": 16 * 9, "8": 8 * 9, "10": 1 * 9}
    for name, value in params.items():
        bound = 1.0 / np.sqrt(fan_ins[name.split(".")[0]])
        values = np.asarray(value)
        assert values.dtype == np.float32
        assert np.abs(values).max() <= bound
        if values.size >= 1000:
            # A uniform on [-b, b] has standard deviation b / sqrt(3).
            assert np.std(values) == pytest.approx(bound / np.sqrt(3.0), rel=0.05)


def test_binary_evidence_and_its_gradients_match_torch():
    """The training objective and its gradient with respect to every decoder weight
    equal a torch transcription of the reference binary_evidence through the
    reference torch decoder, for the same weights, lattice shift and data."""
    params = init_decoder_params(jax.random.PRNGKey(3), latent_dim=2, c_dim=0, head="relu")
    lattice = gen_fib_basis(7)
    shift = jnp.asarray([[0.3, 0.71]], dtype=jnp.float32)
    data = _blob_spectrograms(5, np.random.default_rng(4))[:, None]
    data = data / data.max(axis=(1, 2, 3), keepdims=True)

    loss_jax, grads_jax = jax.value_and_grad(
        lambda p: binary_evidence(decode_shifted_lattice(lattice, shift, p), jnp.asarray(data))
    )(params)

    decoder = _torch_decoder("relu")
    decoder.load_state_dict({name: torch.from_numpy(np.array(value)) for name, value in params.items()})
    shifted = (torch.from_numpy(np.array(lattice)) + torch.from_numpy(np.array(shift))) % 1
    basis = torch.cat([torch.cos(2 * torch.pi * shifted), torch.sin(2 * torch.pi * shifted)], dim=-1)
    loss_torch = _torch_binary_evidence(decoder(basis), torch.from_numpy(data))
    loss_torch.backward()

    assert float(loss_jax) == pytest.approx(loss_torch.item(), rel=1e-5)
    # Float32 sums over 16,384 pixels and 21 lattice points round differently in the
    # two frameworks, so each gradient is compared as a whole (relative L2 error).
    # Against a float64 torch run, float32 torch and JAX are both off by up to ~1e-3
    # in the first layer, so 5e-3 separates rounding from a wrong objective.
    for name, parameter in decoder.named_parameters():
        grad_torch = parameter.grad.numpy()
        relative_error = np.linalg.norm(np.asarray(grads_jax[name]) - grad_torch) / np.linalg.norm(grad_torch)
        assert relative_error < 5e-3, name


def test_jax_inference_matches_torch_decoder_and_posterior():
    """Inference stands in for the torch model, so the same weights must give the same
    images and lattice posterior in both, for both heads."""
    for head, seed in (("legacy", 0), ("relu", 1)):
        torch.manual_seed(seed)
        decoder = _torch_decoder(head).eval()
        params = {key: jnp.asarray(value.detach().numpy()) for key, value in decoder.state_dict().items()}
        assert decoder_head(params) == head

        lattice_np = np.array(gen_fib_basis(15), dtype=np.float32)
        basis_np = np.array(torus_basis_forward(jnp.asarray(lattice_np % 1)), dtype=np.float32)
        with torch.no_grad():
            images_torch = decoder(torch.from_numpy(basis_np)).numpy()
        images_jax = np.asarray(decoder_forward(jnp.asarray(basis_np), params))
        assert images_jax.shape == images_torch.shape == (610, 1, 128, 128)
        np.testing.assert_allclose(images_jax, images_torch, atol=1e-5)

        # The reference QMCLVM.posterior_probability, transcribed.
        rows = np.array([3, 250, 511])
        data_np = (images_torch[rows] > np.median(images_torch[rows], axis=(1, 2, 3), keepdims=True)).astype(np.float32)
        with torch.no_grad():
            lls = _torch_binary_lp(torch.from_numpy(images_torch), torch.from_numpy(data_np))
            evidence = torch.special.logsumexp(lls, dim=1, keepdim=True) - np.log(lls.shape[1])
            posterior_torch = torch.nn.Softmax(dim=1)(lls - evidence).numpy()
        posterior_jax = np.asarray(posterior_over_lattice(jnp.asarray(images_jax), jnp.asarray(data_np)))
        np.testing.assert_allclose(posterior_jax, posterior_torch, atol=5e-4)
        assert np.array_equal(posterior_jax.argmax(axis=1), posterior_torch.argmax(axis=1))


def test_training_loss_decreases(trained_cell):
    """The per-epoch training loss falls over the run and the validation loss is
    recorded at every val_freq-th epoch and the last."""
    with np.load(trained_cell / "metrics" / "val_diagnostics.npz") as diagnostics:
        train_losses = diagnostics["train_losses"]
        assert train_losses.shape == (_TINY_CFG["train_qlvm"]["n_epochs"],)
        assert np.all(np.isfinite(train_losses))
        assert train_losses[-1] < train_losses[0]
        assert diagnostics["val_loss_epochs"].tolist() == [3, 6]
        assert np.all(np.isfinite(diagnostics["val_losses"]))
        assert str(diagnostics["decoder_head"]) == "relu"


def test_checkpoint_has_the_shipped_structure_and_loads_in_torch(trained_cell):
    """checkpoint.tar holds what a shipped cell's does -- the QMCLVM state_dict with
    decoder. keys, a torch Adam state_dict and the per-batch losses -- and its weights
    load into the reference torch decoder, which then decodes like the JAX one."""
    checkpoint = torch.load(trained_cell / "checkpoint.tar", map_location="cpu", weights_only=False)
    assert set(checkpoint) == {"model", "optimizer", "run info"}
    shapes = decoder_parameter_shapes(latent_dim=2, c_dim=0, head="relu")
    assert list(checkpoint["model"]) == [f"decoder.{name}" for name in shapes]
    n_batches = _TINY_CFG["train_qlvm"]["n_epochs"] * int(np.ceil(48 / _TINY_CFG["train_qlvm"]["batch_size"]))
    assert len(checkpoint["run info"]) == n_batches
    assert all(isinstance(loss, float) for loss in checkpoint["run info"])

    decoder = _torch_decoder("relu")
    decoder.load_state_dict({key[len("decoder."):]: value for key, value in checkpoint["model"].items()})
    optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3)
    optimizer.load_state_dict(checkpoint["optimizer"])
    assert float(optimizer.state_dict()["state"][0]["step"]) == n_batches

    basis = np.array(torus_basis_forward(gen_fib_basis(6)), dtype=np.float32)
    with torch.no_grad():
        images_torch = decoder.eval()(torch.from_numpy(basis)).numpy()
    params = load_decoder_params(str(trained_cell / "checkpoint.tar"))
    np.testing.assert_allclose(np.asarray(decoder_forward(jnp.asarray(basis), params)), images_torch, atol=1e-5)


def test_checkpoint_round_trips_through_the_inference_loader(trained_cell, tmp_path):
    """Without torch, the inference path reads the checkpoint (read_torch_checkpoint,
    load_decoder_params), accepts the training contract (enforce_training_contract),
    loads the cell once its label grids exist (load_model_cell) and embeds calls."""
    raw = read_torch_checkpoint(trained_cell / "checkpoint.tar")
    assert set(raw) == {"model", "optimizer", "run info"}

    contract = json.loads((trained_cell / "config" / "training_contract.json").read_text())
    assert contract["decoder_head"] == "relu"
    assert contract["input_normalization"] == "minmax"
    assert contract["normalization_epsilon"] == 1e-8
    assert contract["masking_type"] == "none"
    assert contract["floor"] == 0.2
    assert contract["require_mask"] is True
    assert contract["c_dim"] == 0
    assert contract["condition"] is None
    assert contract["embedding_fib_m"] == _TINY_CFG["train_qlvm"]["embedding_fib_m"]
    assert contract["training_fib_m"] == _TINY_CFG["train_qlvm"]["training_fib_m"]
    run_config = json.loads((trained_cell / "config" / "run_config.json").read_text())
    assert run_config["train_qlvm"] == _TINY_CFG["train_qlvm"]
    assert run_config["n_train"] == 48

    params = load_decoder_params(str(trained_cell / "checkpoint.tar"))
    assert enforce_training_contract(contract, _INFERENCE_CFG, params) == 128.0

    # A cell becomes loadable once clustering adds its label grids.
    cell = tmp_path / "package" / "phase" / "cell"
    cell.mkdir(parents=True)
    (cell / "checkpoint.tar").write_bytes((trained_cell / "checkpoint.tar").read_bytes())
    (cell / "config").mkdir()
    (cell / "config" / "training_contract.json").write_text(json.dumps(contract))
    for level in ("fine", "coarse"):
        (cell / "inference" / f"clusters_{level}").mkdir(parents=True)
        np.save(cell / "inference" / f"clusters_{level}" / "label_grid.npy", np.ones((8, 8), dtype=np.int64))
    model = load_model_cell(str(cell))
    assert model["lattice"].shape == (21, 2)
    coords = np.asarray(embed_data(
        model["lattice"],
        jnp.asarray(_blob_spectrograms(3, np.random.default_rng(9))[:, None]),
        model["params"],
        lattice_batch_size=8,
        data_batch_size=2,
    ))
    assert coords.shape == (3, 2)
    assert np.all((coords >= 0.0) & (coords < 1.0))


def test_load_split_minmax_and_mask_rules(tmp_path):
    """Inputs are min-max normalized per spectrogram (epsilon 1e-8) and then masked by
    mask > 0.5 when the split applies masks: its own apply_mask decides, and a split
    that does not declare one follows the set's masking_type."""
    rng = np.random.default_rng(0)
    masks = (rng.random((6, 128, 128)) > 0.5).astype(np.float32)
    _write_split(tmp_path / "declared.npz", 6, seed=0, apply_mask=True, masks=masks)
    _write_split(tmp_path / "silent.npz", 6, seed=0, masks=masks)
    with np.load(tmp_path / "declared.npz") as split:
        expected = minmax_per_spectrogram(split["spectrograms"], np.float32(1e-8))

    declared = load_split(tmp_path / "declared.npz", "none")
    assert declared["apply_mask"] is True
    np.testing.assert_array_equal(declared["inputs"], expected * masks)

    assert load_split(tmp_path / "silent.npz", "sam")["apply_mask"] is True
    unmasked = load_split(tmp_path / "silent.npz", "none")
    assert unmasked["apply_mask"] is False
    np.testing.assert_array_equal(unmasked["inputs"], expected)

    _write_split(tmp_path / "zero_masks.npz", 6, seed=0, apply_mask=True, masks=np.zeros_like(masks))
    with pytest.raises(ValueError, match="zero every spectrogram"):
        load_split(tmp_path / "zero_masks.npz", "sam")


def test_validation_subset_matches_the_reference_draw():
    """The subset is the reference trainer's: np.random.seed(seed), then per masks_len
    value in increasing order all rows or np.random.choice(rows, cap, replace=False)."""
    masks_len = np.random.default_rng(1).integers(0, 5, 300)
    np.random.seed(42)  # noqa: NPY002 (the reference draw uses the legacy global generator)
    expected = []
    for value in np.unique(masks_len):
        rows = np.where(masks_len == value)[0]
        expected.extend((rows if len(rows) <= 40 else np.random.choice(rows, 40, replace=False)).tolist())  # noqa: NPY002
    np.testing.assert_array_equal(validation_subset_indices(masks_len, 40, 42), np.array(sorted(expected)))


def test_train_legacy_head_on_full_data_without_validation(tmp_path, mocker):
    """With only full_data.npz there is no validation; the legacy head is written as
    such, a set without require_mask / floor records false / null, and a NaN
    length_threshold (no duration bound) is recorded as null."""
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    _write_split(dataset_dir / "full_data.npz", 16, seed=0, apply_mask=False)
    _write_metadata(dataset_dir, masking_type="none", require_mask=None, floor=None, length_threshold=np.nan)
    cfg = {"train_qlvm": {**_TINY_CFG["train_qlvm"], "decoder_head": "legacy", "n_epochs": 2}}
    _train(dataset_dir, tmp_path / "cell", cfg, mocker)

    params = load_decoder_params(str(tmp_path / "cell" / "checkpoint.tar"))
    assert decoder_head(params) == "legacy"
    contract = json.loads((tmp_path / "cell" / "config" / "training_contract.json").read_text())
    assert contract["decoder_head"] == "legacy"
    assert contract["require_mask"] is False
    assert contract["floor"] is None
    assert contract["length_threshold"] is None
    with np.load(tmp_path / "cell" / "metrics" / "val_diagnostics.npz") as diagnostics:
        assert diagnostics["val_loss_epochs"].size == 0


@pytest.mark.parametrize(
    ("key", "value"),
    [("val_freq", 0), ("n_epochs", 0), ("batch_size", 0), ("training_fib_m", 2), ("decoder_head", "harmonic"), ("learning_rate", 0.0)],
)
def test_train_rejects_invalid_settings(tmp_path, mocker, key, value):
    """Invalid settings raise before any data is read."""
    cfg = {"train_qlvm": {**_TINY_CFG["train_qlvm"], key: value}}
    with pytest.raises(ValueError, match=key):
        _train(tmp_path, tmp_path / "out", cfg, mocker)


def test_train_missing_dataset_raises(tmp_path, mocker):
    """A dataset directory without train_data.npz/full_data.npz raises FileNotFoundError."""
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="No training set found"):
        _train(tmp_path / "empty", tmp_path / "out", _TINY_CFG, mocker)


def test_train_refuses_splits_older_than_full_data(tmp_path, mocker):
    """A set rebuilt with full_dataset keeps its old splits, and train_data.npz would
    win the lookup: a split older than full_data.npz is refused before training."""
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    _write_split(dataset_dir / "train_data.npz", 4, seed=0)
    _write_split(dataset_dir / "val_data.npz", 4, seed=1)
    _write_split(dataset_dir / "full_data.npz", 8, seed=2)
    _write_metadata(dataset_dir)
    newest = (dataset_dir / "full_data.npz").stat().st_mtime
    for split in ("train_data.npz", "val_data.npz"):
        os.utime(dataset_dir / split, (newest - 60, newest - 60))
    with pytest.raises(ValueError, match=r"predate full_data\.npz"):
        _train(dataset_dir, tmp_path / "out", _TINY_CFG, mocker)
    assert not (tmp_path / "out").exists()


def test_train_missing_metadata_raises(tmp_path, mocker):
    """Without metadata.npz the training contract cannot record the set's
    preprocessing, so the run stops before training."""
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    _write_split(dataset_dir / "full_data.npz", 8, seed=0)
    with pytest.raises(FileNotFoundError, match=r"No metadata\.npz"):
        _train(dataset_dir, tmp_path / "out", _TINY_CFG, mocker)
