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

import itertools
import json
import os
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from usv_playpen.processing.qlvm_latents import (
    compute_condition_values,
    enforce_training_contract,
    frozen_condition_values,
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

# optax (and train_qlvm, which pulls it) -> a one-time JAX DeprecationWarning at import.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import optax

    from usv_playpen.processing.train_qlvm import (
        QLVMTrainer,
        condition_bin_edges,
        condition_decode_grid,
        condition_group_ids,
        grouped_batches,
        load_split,
        make_step_functions,
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
        "conditional": None,
        "condition_n_bins": 32,
        "condition_bin_scheme": "quantile_capped",
        "condition_scale_loss_by_batch": True,
        "condition_decode_grid_step": 0.0025,
        "condition_table": None,
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


def _write_split(path, n_samples, *, seed, apply_mask=None, masks=None, masks_len=None, condition_columns=None):
    """Write one split in the model-package layout; ``apply_mask`` None leaves the key
    out, and ``condition_columns`` (name -> (N,) float64) adds the USV summary columns
    build-qlvm-training-set copies into every split."""
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
    if condition_columns is not None:
        arrays.update(condition_columns)
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
    [
        ("val_freq", 0),
        ("n_epochs", 0),
        ("batch_size", 0),
        ("training_fib_m", 2),
        ("decoder_head", "harmonic"),
        ("learning_rate", 0.0),
        ("conditional", "pitch"),
        ("condition_n_bins", 0),
        ("condition_bin_scheme", "equal_width"),
        ("condition_decode_grid_step", 0.0),
        ("condition_table", "/no/such/table.npz"),
    ],
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


def _reference_bin_edges(values, n_bins, scheme):
    """Transcription of the reference data/conditionals.py:bin_edges."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1)))
    if scheme == "quantile" or len(edges) < 4:
        return edges
    cap = float(np.diff(edges)[1:-1].max())
    out = [edges[0]]
    for lo, hi in itertools.pairwise(edges):
        pieces = int(np.ceil((hi - lo) / cap - 1e-9))
        out.extend(np.linspace(lo, hi, pieces + 1)[1:].tolist())
    return np.asarray(out)


def _reference_sampler_epoch(group_ids, batch_size, seed, epoch):
    """Transcription of one epoch of the reference ConditionGroupedBatchSampler
    (shuffle=True, drop_last=False)."""
    rng = np.random.default_rng(seed + epoch)
    batches = []
    for gid in sorted(np.unique(group_ids).tolist()):
        idx = rng.permutation(np.where(group_ids == gid)[0])
        batches.extend([idx[start:start + batch_size].tolist() for start in range(0, len(idx), batch_size)])
    return [batches[i] for i in rng.permutation(len(batches))]


def _conditional_cfg(conditional, **overrides):
    """The tiny recipe for a conditional run: 4 bins, so a few dozen rows give groups of several rows."""
    return {"train_qlvm": {**_TINY_CFG["train_qlvm"], "conditional": conditional, "condition_n_bins": 4, **overrides}}


def _summary_columns(n_samples, seed):
    """Raw per-call values in the units of the USV summary columns a split carries."""
    rng = np.random.default_rng(seed)
    return {
        "mean_freq_hz": rng.uniform(35000.0, 110000.0, n_samples),
        "freq_bandwidth_hz": rng.uniform(1000.0, 40000.0, n_samples),
        "loudness_db": rng.uniform(20.0, 100.0, n_samples),
    }


@pytest.fixture(scope="module")
def loudness_cell(tmp_path_factory):
    """One short CPU loudness-conditioned run whose raw values come from the splits'
    own loudness_db columns (some beyond the frozen 28.83-95.85 dB range, so the clip
    is exercised)."""
    root = tmp_path_factory.mktemp("qlvm_train_conditional")
    dataset_dir = root / "dataset"
    dataset_dir.mkdir()
    _write_split(dataset_dir / "train_data.npz", 48, seed=0, apply_mask=False, condition_columns=_summary_columns(48, 10))
    _write_split(dataset_dir / "val_data.npz", 16, seed=1, apply_mask=False, condition_columns=_summary_columns(16, 11))
    _write_metadata(dataset_dir)
    output_dir = root / "cell"
    QLVMTrainer(
        dataset_directory=str(dataset_dir),
        output_directory=str(output_dir),
        input_parameter_dict=_conditional_cfg("loudness"),
        message_output=lambda *_a, **_kw: None,
    ).train()
    return dataset_dir, output_dir


def test_condition_bin_edges_follow_the_reference():
    """Both bin schemes reproduce the reference edges exactly; the capped scheme only
    splits the end bins, into pieces no wider than the widest inner bin."""
    values = np.random.default_rng(3).gamma(1.5, 1.0, 5000).astype(np.float32)
    for scheme in ("quantile", "quantile_capped"):
        np.testing.assert_array_equal(condition_bin_edges(values, 32, scheme), _reference_bin_edges(values, 32, scheme))
    plain = condition_bin_edges(values, 32, "quantile")
    capped = condition_bin_edges(values, 32, "quantile_capped")
    widest_inner = np.diff(plain)[1:-1].max()
    assert capped.size > plain.size
    assert np.diff(capped).max() <= widest_inner + 1e-12
    np.testing.assert_array_equal(capped[1:-1][np.isin(capped[1:-1], plain)], plain[1:-1])
    np.testing.assert_array_equal(condition_group_ids(np.array([plain[0] - 1.0, plain[-1] + 1.0]), plain), [0, plain.size - 2])


def test_grouped_batches_follow_the_reference_sampler():
    """Every epoch's batches are the reference sampler's, draw for draw: homogeneous
    in group, at most batch_size rows, every row exactly once."""
    group_ids = np.random.default_rng(4).integers(0, 7, 300)
    for epoch in range(3):
        batches = grouped_batches(group_ids, 16, np.random.default_rng(42 + epoch))
        assert [batch.tolist() for batch in batches] == _reference_sampler_epoch(group_ids, 16, 42, epoch)
        assert all(np.unique(group_ids[batch]).size == 1 and batch.size <= 16 for batch in batches)
        np.testing.assert_array_equal(np.sort(np.concatenate(batches)), np.arange(300))


def test_condition_decode_grid_covers_the_training_range():
    """The grid starts at the training minimum, steps by exactly step and ends at or
    just past the training maximum (the reference data/conditionals.py:decode_grid)."""
    grid = condition_decode_grid(0.0041258004, 0.9495616, 0.0025)
    assert grid.size == 380
    assert grid[0] == 0.0041258004
    np.testing.assert_allclose(np.diff(grid), 0.0025)
    assert grid[-2] < 0.9495616 <= grid[-1]
    np.testing.assert_array_equal(condition_decode_grid(0.0, 1.0, 0.01), 0.01 * np.arange(101))


def test_condition_enters_the_decoder():
    """A conditional decoder's first Linear layer takes 2 * latent_dim + 1 inputs, the
    condition appended to every lattice point's torus basis (QMCLVM.forward's
    torch.cat([basis, c.repeat(K, 1)], -1)), so different conditions decode different images."""
    params = init_decoder_params(jax.random.PRNGKey(0), latent_dim=2, c_dim=1, head="relu")
    assert params["0.weight"].shape == (2048, 5)
    lattice = jnp.asarray(gen_fib_basis(6), dtype=jnp.float32)
    shift = jnp.array([[0.3, 0.7]], dtype=jnp.float32)
    low = decode_shifted_lattice(lattice, shift, params, jnp.array([[0.1]], dtype=jnp.float32))
    high = decode_shifted_lattice(lattice, shift, params, jnp.array([[0.9]], dtype=jnp.float32))
    basis = torus_basis_forward((lattice + shift) % 1)
    expected = decoder_forward(jnp.concatenate([basis, jnp.full((lattice.shape[0], 1), 0.1, dtype=jnp.float32)], axis=-1), params)
    np.testing.assert_allclose(np.asarray(low), np.asarray(expected), atol=1e-6)
    assert float(jnp.max(jnp.abs(low - high))) > 1e-4


def test_scaled_padded_step_is_the_reference_step():
    """A short batch padded with zero-weight rows and its loss scaled by rows /
    batch_size takes exactly the step the reference takes on the unpadded batch
    (train/train.py's scale_to_batch), and the recorded loss is the unscaled mean."""
    lattice = jnp.asarray(gen_fib_basis(6), dtype=jnp.float32)
    params = init_decoder_params(jax.random.PRNGKey(1), latent_dim=2, c_dim=1, head="relu")
    optimizer = optax.sgd(1.0)
    train_step, _ = make_step_functions(optimizer, lattice, lattice)
    rows = jnp.asarray(minmax_per_spectrogram(_blob_spectrograms(3, np.random.default_rng(5)), np.float32(1e-8))[:, None])
    padded = jnp.concatenate([rows, jnp.zeros((5, 1, 128, 128), dtype=jnp.float32)])
    weights = jnp.asarray(np.array([1, 1, 1, 0, 0, 0, 0, 0], dtype=np.float32))
    condition = jnp.array([[0.4]], dtype=jnp.float32)
    shift = jnp.array([[0.2, 0.6]], dtype=jnp.float32)

    new_params, _, loss = train_step(params, optimizer.init(params), padded, weights, condition, shift, jnp.float32(3 / 8))

    def reference_loss(weights_dict):
        """The reference objective of the unpadded 3-row batch."""
        return binary_evidence(decode_shifted_lattice(lattice, shift, weights_dict, condition), rows)

    reference_value, reference_grads = jax.value_and_grad(reference_loss)(params)
    np.testing.assert_allclose(float(loss), float(reference_value), rtol=1e-6)
    # float32 sums over a padded vs an unpadded batch round differently (more so on a
    # GPU, whose reductions order differently than the CPU's: up to 1.5e-3 measured),
    # so each layer is compared to 5e-3 of its largest gradient entry.
    for name in params:
        expected_step = np.asarray(reference_grads[name]) * 3 / 8
        np.testing.assert_allclose(
            np.asarray(params[name] - new_params[name]), expected_step, rtol=0.0, atol=5e-3 * np.abs(expected_step).max(),
        )


def test_conditional_cell_records_bins_and_condition(loudness_cell):
    """A loudness run writes the phase 11 condition block and a condition_bins.npz
    whose bins, group sizes and means, training range and decode grid follow from
    the training split's values (the frozen dB map, clipped)."""
    dataset_dir, cell = loudness_cell
    contract = json.loads((cell / "config" / "training_contract.json").read_text())
    assert contract["c_dim"] == 1
    assert contract["conditional"] == "loudness"
    assert contract["condition_bins"] == "config/condition_bins.npz"
    condition = contract["condition"]
    assert condition["name"] == "loudness"
    assert condition["db_range"] == [28.83, 95.85]
    assert condition["decode"] == "grid"
    assert condition["training_bins"].startswith("quantile_capped: 4 quantile bins")

    with np.load(dataset_dir / "train_data.npz") as split:
        train_c = compute_condition_values(condition, split["durations"], None, split["loudness_db"])
    assert train_c.min() == 0.0
    assert train_c.max() == 1.0
    with np.load(cell / "config" / "condition_bins.npz") as bins:
        np.testing.assert_array_equal(bins["edges"], _reference_bin_edges(train_c, 4, "quantile_capped"))
        groups = condition_group_ids(train_c, bins["edges"])
        labels = np.unique(groups)
        np.testing.assert_array_equal(bins["group_ids"], labels)
        np.testing.assert_array_equal(bins["group_sizes"], np.bincount(groups)[labels])
        np.testing.assert_array_equal(bins["group_means"], [train_c[groups == group].mean() for group in labels])
        assert bins["group_means"].dtype == np.float32
        assert bins["train_c_min"].dtype == np.float32
        assert float(bins["train_c_min"]) == 0.0
        assert float(bins["train_c_max"]) == 1.0
        np.testing.assert_allclose(bins["decode_grid"], 0.0025 * np.arange(401))
        assert str(bins["conditional"]) == "loudness"
        assert str(bins["bin_scheme"]) == "quantile_capped"

    run_config = json.loads((cell / "config" / "run_config.json").read_text())
    assert run_config["batches_per_epoch"] == int(sum(-(-size // 8) for size in np.bincount(groups)[labels]))
    assert run_config["condition_stats"]["train"]["n_clamped"] == int(np.count_nonzero((train_c <= 0.0) | (train_c >= 1.0)))
    with np.load(cell / "metrics" / "val_diagnostics.npz") as diagnostics, np.load(dataset_dir / "val_data.npz") as split:
        assert str(diagnostics["conditional"]) == "loudness"
        val_c = compute_condition_values(condition, split["durations"], None, split["loudness_db"])
        np.testing.assert_array_equal(diagnostics["val_diag_c"], val_c[diagnostics["val_diag_indices"]])
        assert np.all(np.isfinite(diagnostics["val_losses"]))
        assert np.all(np.isfinite(diagnostics["train_losses"]))


def test_conditional_checkpoint_round_trips_through_the_phase11_decode_path(loudness_cell, tmp_path):
    """The conditional cell passes enforce_training_contract, loads with load_model_cell
    (phase 11 bins + decode rule), and embeds calls at their frozen conditioning values."""
    _, trained = loudness_cell
    contract = json.loads((trained / "config" / "training_contract.json").read_text())
    params = load_decoder_params(str(trained / "checkpoint.tar"))
    assert params["0.weight"].shape == (2048, 5)
    assert enforce_training_contract(contract, _INFERENCE_CFG, params) == 128.0

    cell = tmp_path / "package" / "phase" / "cell"
    (cell / "config").mkdir(parents=True)
    (cell / "checkpoint.tar").write_bytes((trained / "checkpoint.tar").read_bytes())
    for name in ("training_contract.json", "condition_bins.npz"):
        (cell / "config" / name).write_bytes((trained / "config" / name).read_bytes())
    for level in ("fine", "coarse"):
        (cell / "inference" / f"clusters_{level}").mkdir(parents=True)
        np.save(cell / "inference" / f"clusters_{level}" / "label_grid.npy", np.ones((8, 8), dtype=np.int64))
    model = load_model_cell(str(cell))
    condition = model["contract"]["condition"]
    own_values = compute_condition_values(condition, np.array([20, 40, 60]), None, np.array([10.0, 50.0, 120.0]))
    frozen = frozen_condition_values(own_values, model["condition_bins"], condition["decode"])
    np.testing.assert_allclose(frozen, [0.0, own_values[1], 1.0], atol=0.00125 + 1e-6)
    coords = np.asarray(embed_data(
        model["lattice"],
        jnp.asarray(_blob_spectrograms(3, np.random.default_rng(9))[:, None]),
        model["params"],
        lattice_batch_size=8,
        data_batch_size=2,
        condition_values=frozen,
    ))
    assert coords.shape == (3, 2)
    assert np.all((coords >= 0.0) & (coords < 1.0))


def test_duration_condition_uses_the_training_duration_range(tmp_path, mocker):
    """A duration run needs no raw column: the training split's shortest and longest
    durations are recorded as the min-max, the decode is exact, and the validation
    calls are conditioned on the same map."""
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    _write_split(dataset_dir / "train_data.npz", 32, seed=0, apply_mask=False)
    _write_split(dataset_dir / "val_data.npz", 8, seed=1, apply_mask=False)
    _write_metadata(dataset_dir)
    _train(dataset_dir, tmp_path / "cell", _conditional_cfg("duration", n_epochs=2, condition_decode_grid_step=0.01), mocker)

    condition = json.loads((tmp_path / "cell" / "config" / "training_contract.json").read_text())["condition"]
    with np.load(dataset_dir / "train_data.npz") as split:
        durations = split["durations"]
    assert (condition["duration_min"], condition["duration_max"]) == (int(durations.min()), int(durations.max()))
    assert condition["decode"] == "exact"
    with np.load(tmp_path / "cell" / "config" / "condition_bins.npz") as bins:
        assert float(bins["train_c_min"]) == 0.0
        np.testing.assert_allclose(float(bins["train_c_max"]), 1.0, atol=1e-6)
        assert bins["decode_grid"].size == 101
    with np.load(tmp_path / "cell" / "metrics" / "val_diagnostics.npz") as diagnostics, np.load(dataset_dir / "val_data.npz") as split:
        expected = compute_condition_values(condition, split["durations"], None)[diagnostics["val_diag_indices"]]
        np.testing.assert_array_equal(diagnostics["val_diag_c"], expected)


def test_condition_table_supplies_values_and_gaps_stop_the_run(tmp_path, mocker):
    """Without split columns, a per-call table keyed by spec_id supplies the raw values
    (the package's cond_table.npz names loudness image_level_db); a split row missing
    from the table, a split without the column and no table, and a non-finite value
    each stop the run before training."""
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    _write_split(dataset_dir / "train_data.npz", 24, seed=0, apply_mask=False)
    _write_metadata(dataset_dir)
    spec_ids = np.array([f"sess_{i}" for i in range(24)])
    decibels = np.random.default_rng(2).uniform(30.0, 90.0, 24).astype(np.float32)
    table = tmp_path / "cond_table.npz"
    np.savez(table, spec_id=spec_ids, image_level_db=decibels)
    _train(dataset_dir, tmp_path / "cell", _conditional_cfg("loudness", n_epochs=1, condition_table=str(table)), mocker)
    run_config = json.loads((tmp_path / "cell" / "config" / "run_config.json").read_text())
    assert run_config["condition_source"] == f"condition_table {table}"
    expected = np.clip((decibels.astype(np.float64) - 28.83) / (95.85 - 28.83), 0.0, 1.0).astype(np.float32)
    assert run_config["condition_stats"]["train"]["c_min"] == float(expected.min())
    assert run_config["condition_stats"]["train"]["c_max"] == float(expected.max())

    np.savez(table, spec_id=spec_ids[1:], image_level_db=decibels[1:])
    with pytest.raises(ValueError, match="not in the condition_table"):
        _train(dataset_dir, tmp_path / "out_missing", _conditional_cfg("loudness", condition_table=str(table)), mocker)
    with pytest.raises(ValueError, match="no loudness_db column"):
        _train(dataset_dir, tmp_path / "out_no_column", _conditional_cfg("loudness"), mocker)
    decibels[3] = np.nan
    np.savez(table, spec_id=spec_ids, image_level_db=decibels)
    with pytest.raises(ValueError, match="no finite loudness_db"):
        _train(dataset_dir, tmp_path / "out_nan", _conditional_cfg("loudness", condition_table=str(table)), mocker)
