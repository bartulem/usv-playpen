"""
@author: bartulem
Tests for processing/qlvm_latents — the QLVM inference driver.

Covers the helper pieces (weight loading + ``decoder.`` prefix stripping, the
label-grid pixel rule, training-contract checks, input normalization, condition
values) and end-to-end runs that synthesize QLVM model package cells, a session
spectrogram H5 and a ``usv_summary.csv``, then check the ``model_cells`` columns
are merged into the right rows.
"""

from __future__ import annotations

import json
import pathlib
import shutil
import pickle
import warnings
import zipfile

import h5py
import jax.numpy as jnp
import numpy as np
import polars as pls
import pytest
from click.testing import CliRunner

from usv_playpen.os_utils import cell_cluster_directory
from usv_playpen.processing import qlvm_latents as ql
from usv_playpen.processing.build_qlvm_training_set import build_session_masks, file_sha256, stretch_specs

# train_qlvm pulls optax -> a one-time JAX DeprecationWarning at import.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from usv_playpen.processing.train_qlvm import prepare_split


def test_label_grid_lookup_is_the_package_pixel_rule_at_the_edges():
    """label_grid_lookup is the model packages' rule px = floor(x * res) mod res,
    py = floor(y * res) mod res, label = grid[py, px] -- checked against that formula
    written out independently on the edge values: 0.0, coordinates exactly on pixel
    boundaries (which belong to the pixel they open), values just below a boundary and
    just below 1.0 (the last pixel), exactly 1.0 (wraps to pixel 0, as on the torus) and
    a tiny negative value (wraps to the last pixel)."""
    resolution = 200
    grid = (np.arange(resolution * resolution).reshape(resolution, resolution) + 1).astype(np.int32)
    below_one = np.nextafter(1.0, 0.0)
    edges = np.array([
        0.0, 1 / resolution, 2 / resolution, 0.5, 191 / resolution, np.nextafter(191 / resolution, 0.0),
        199 / resolution, 0.999999, below_one, 1.0, -1e-9,
    ])
    xs, ys = np.meshgrid(edges, edges[::-1])
    coords = np.column_stack([xs.ravel(), ys.ravel()])

    labels = ql.label_grid_lookup(coords, grid)

    expected_px = np.floor(coords[:, 0] * resolution).astype(np.int64) % resolution
    expected_py = np.floor(coords[:, 1] * resolution).astype(np.int64) % resolution
    np.testing.assert_array_equal(labels, grid[expected_py, expected_px])
    by_value = {value: ql.label_grid_lookup(np.array([[value, 0.0]]), grid)[0] for value in edges}
    assert by_value[0.0] == grid[0, 0]
    assert by_value[1 / resolution] == grid[0, 1]
    assert by_value[191 / resolution] == grid[0, 191]
    assert by_value[np.nextafter(191 / resolution, 0.0)] == grid[0, 190]
    assert by_value[below_one] == grid[0, 199]
    assert by_value[1.0] == grid[0, 0]
    assert by_value[-1e-9] == grid[0, 199]


def _write_session_masks(root, session_id, masks_by_row):
    """Write the session H5's ``mask/<session>`` group the way generate-usv-masks
    does: one ``segmentations`` row per mask instance, keyed by ``spectrogram_index``
    (``masks_by_row`` maps a spectrogram row to one boolean mask)."""
    rows = sorted(masks_by_row)
    with h5py.File(root / "audio" / "spectrograms" / f"{session_id}_spectrograms.h5", "a") as f:
        group = f.create_group(f"mask/{session_id}")
        group.create_dataset("segmentations", data=np.stack([masks_by_row[row] for row in rows]))
        group.create_dataset("spectrogram_index", data=np.array(rows, dtype=np.int64))


def _make_inference_session(tmp_path, rng, *, with_masks=True):
    """Synthesize a session (spectrogram H5 + usv_summary) for an end-to-end
    QLVMLatentInference run and return (root, session_id, cfg), with ``cfg`` an
    ``infer_qlvm_latents`` block whose ``model_cells`` is still empty (a test lists
    the cells it embeds with, see :func:`_make_model_cell`). Rows 0 and 2 are real
    64-bin calls (inside the synthetic cells' duration window of 100), row 1 is a
    placeholder; with ``with_masks``, rows 0 and 2 each get one all-ones SAM mask.
    The summary carries the ``usv`` / ``squeak`` booleans detect-usv-squeaks writes (rows 0
    and 2 pure USVs, ``(true, false)``; the placeholder row null), which inference requires."""
    session_id = "20230119_155302"
    root = tmp_path / session_id
    (root / "audio" / "spectrograms").mkdir(parents=True)

    n_f = n_t = 128
    specs = np.zeros((3, n_f, n_t), dtype=np.float32)
    specs[0] = rng.random((n_f, n_t)).astype(np.float32)
    specs[2] = rng.random((n_f, n_t)).astype(np.float32)
    with h5py.File(root / "audio" / "spectrograms" / f"{session_id}_spectrograms.h5", "w") as f:
        f.create_dataset("frequency_bins", data=np.linspace(30000.0, 120000.0, n_f))
        session_group = f.create_group(f"spectrogram/{session_id}")
        session_group.create_dataset("spectrograms", data=specs)
        session_group.create_dataset("durations", data=np.array([64, 0, 64], dtype=np.int64))
    if with_masks:
        _write_session_masks(root, session_id, {0: np.ones((n_f, n_t), dtype=bool), 2: np.ones((n_f, n_t), dtype=bool)})

    pls.DataFrame({
        "usv_id": [f"{i:04d}" for i in range(3)],
        "start": [0.1, 0.3, 0.5],
        "stop": [0.15, 0.35, 0.55],
        "usv": pls.Series([True, None, True], dtype=pls.Boolean),
        "squeak": pls.Series([False, None, False], dtype=pls.Boolean),
    }).write_csv(root / "audio" / f"{session_id}_usv_summary.csv")

    cfg = {
        "model_cells": {},
        "prefer_package_values": True,
        "latent_dim": 2,
        "time_stretch": False,
        "masking_type": "none",
        "target_shape": [128, 128],
        "length_threshold": None,
        "lattice_batch_size": 4096,
        "data_batch_size": 8192,
    }
    return root, session_id, cfg


def test_infer_and_merge_masking_type_applies_or_skips_sam_mask(tmp_path, mocker):
    """The decoder is trained on SAM-masked (background-zeroed) spectrograms, so
    ``masking_type='sam'`` must zero every pixel outside the call's SAM mask region
    before embedding, while ``masking_type='none'`` must embed the raw spectrogram.
    This pins the train/inference masking parity: embedding raw specs into a
    masked-trained decoder is out-of-distribution and yields unreliable latents.
    The spectrogram fed to ``embed_data`` is captured under both settings, each with
    a model cell trained with that masking (the settings must match the contract)."""
    rng = np.random.default_rng(3)
    grid = np.ones((8, 8), dtype=np.int16)
    root, session_id, cfg = _make_inference_session(tmp_path, rng, with_masks=False)
    cells = {
        masking_type: str(_make_model_cell(tmp_path, rng, masking_type=masking_type, floor=None, fine_grid=grid,
                                           coarse_grid=grid, cell_name=f"cell_{masking_type}"))
        for masking_type in ("sam", "none")
    }

    # Add a SAM mask group covering only the top half (rows 0:64) of each real USV
    # (spectrogram rows 0 and 2); build_session_masks unions per spectrogram_index.
    n_f = n_t = 128
    top_half = np.zeros((n_f, n_t), dtype=bool)
    top_half[:64, :] = True
    _write_session_masks(root, session_id, {0: top_half, 2: top_half})

    # Capture the spectrogram batch handed to the decoder without needing a real
    # embedding; return valid torus coordinates so the downstream lookup succeeds.
    captured = {}

    def _fake_embed(lattice, data, params, *_block_sizes):
        captured["data"] = np.asarray(data)
        return np.full((data.shape[0], 2), 0.5, dtype=np.float64)

    mocker.patch("usv_playpen.processing.qlvm_latents.smart_wait")
    mocker.patch("usv_playpen.processing.qlvm_latents.embed_data", side_effect=_fake_embed)

    def _run(masking_type):
        cfg["masking_type"] = masking_type
        cfg["model_cells"] = {"qlvm": cells[masking_type]}
        ql.QLVMLatentInference(
            root_directory=str(root),
            input_parameter_dict={"infer_qlvm_latents": cfg},
            message_output=lambda *_a, **_kw: None,
        ).infer_and_merge()
        return captured["data"]

    data_sam = _run("sam")
    assert np.all(data_sam[:, 0, 64:, :] == 0.0), "masking_type='sam' must zero the region outside the SAM mask"
    assert np.any(data_sam[:, 0, :64, :] != 0.0), "the masked-in (kept) region must retain the spectrogram"

    data_none = _run("none")
    assert np.any(data_none[:, 0, 64:, :] != 0.0), "masking_type='none' must embed the raw (unmasked) spectrogram"


def test_infer_and_merge_honors_target_shape(tmp_path, mocker):
    """The ``target_shape`` setting must drive the resize the embedder applies:
    the spectrogram batch handed to ``embed_data`` must carry exactly the
    configured ``(freq, time)`` shape, not the hard-coded 128x128 default. This
    pins train/inference preprocessing parity, since the decoder can only accept
    the fixed shape it was trained on."""
    rng = np.random.default_rng(4)
    grid = np.ones((8, 8), dtype=np.int16)
    root, _session_id, cfg = _make_inference_session(tmp_path, rng)
    cell = _make_model_cell(tmp_path, rng, masking_type="none", floor=None, fine_grid=grid, coarse_grid=grid,
                            contract_overrides={"target_shape": [96, 112]})
    cfg["model_cells"] = {"qlvm": str(cell)}
    cfg["target_shape"] = [96, 112]

    captured = {}

    def _fake_embed(lattice, data, params, *_block_sizes):
        captured["data"] = np.asarray(data)
        return np.full((data.shape[0], 2), 0.5, dtype=np.float64)

    mocker.patch("usv_playpen.processing.qlvm_latents.smart_wait")
    mocker.patch("usv_playpen.processing.qlvm_latents.embed_data", side_effect=_fake_embed)

    ql.QLVMLatentInference(
        root_directory=str(root),
        input_parameter_dict={"infer_qlvm_latents": cfg},
        message_output=lambda *_a, **_kw: None,
    ).infer_and_merge()

    assert captured["data"].shape[-2:] == (96, 112), "the embed batch must match the configured target_shape"


def test_infer_and_merge_idempotent_preserves_other_columns(tmp_path, mocker):
    """Re-running inference rewrites only the qlvm_* columns: unrelated columns
    survive, the row count is unchanged, and the qlvm columns are refreshed (not
    duplicated or left stale). With the shipped (empty) label-level setting only the
    coordinates are written, and every earlier category / supercategory / category
    confidence column of the re-embedded prefix is removed."""
    rng = np.random.default_rng(2)
    grid = rng.integers(1, 8, size=(8, 8)).astype(np.int16)
    root, session_id, cfg = _make_inference_session(tmp_path, rng)
    cell = _make_model_cell(tmp_path, rng, masking_type="none", floor=0.2, fine_grid=grid, coarse_grid=grid)
    cfg["model_cells"] = {"qlvm": str(cell)}

    # Add an unrelated column the merge must leave intact.
    summary_path = root / "audio" / f"{session_id}_usv_summary.csv"
    df0 = pls.read_csv(summary_path).with_columns(
        pls.Series("quality", [0.11, 0.22, 0.33]),
        pls.Series("qlvm_category", [1, 2, 3]),
        pls.Series("qlvm_supercategory", [1, 1, 1]),
        pls.Series("qlvm_category_agreement", [0.9, 0.9, 0.9]),
        pls.Series("qlvm_category_uncertain", [False, False, False]),
    )
    df0.write_csv(summary_path)

    mocker.patch("usv_playpen.processing.qlvm_latents.smart_wait")
    inference = ql.QLVMLatentInference(
        root_directory=str(root),
        input_parameter_dict={"infer_qlvm_latents": cfg},
        message_output=lambda *_a, **_kw: None,
    )
    inference.infer_and_merge()
    inference.infer_and_merge()  # second run must be a clean overwrite

    df = pls.read_csv(summary_path)
    assert df.height == 3
    # the unrelated column is untouched, and each qlvm column appears exactly once.
    assert df["quality"].to_list() == [0.11, 0.22, 0.33]
    for column in ("qlvm1", "qlvm2"):
        assert df.columns.count(column) == 1
    for column in ("qlvm_model", "qlvm_category", "qlvm_supercategory", "qlvm_category_agreement", "qlvm_category_uncertain"):
        assert column not in df.columns
    assert df.columns == ["usv_id", "start", "stop", "usv", "squeak", "qlvm1", "qlvm2", "quality"]
    # the embedded rows still carry latents after the re-run.
    assert df["qlvm1"][0] is not None
    assert df["qlvm1"][1] is None


def _matching_contract(cfg, **overrides):
    """A training contract that agrees with ``cfg`` (the fields a model package cell's
    contract records, for an unconditional legacy-head decoder), with ``overrides``
    applied."""
    contract = {
        "decoder_head": "legacy",
        "latent_dim": cfg["latent_dim"],
        "c_dim": 0,
        "condition": None,
        "input_normalization": "none",
        "floor": None,
        "masking_type": cfg["masking_type"],
        "target_shape": list(cfg["target_shape"]),
        "time_stretch": cfg["time_stretch"],
        "length_threshold": 128.0,
        "require_mask": False,
        "embedding_lattice_type": "fibonacci",
        "embedding_fib_m": 8,
    }
    contract.update(overrides)
    return contract


def _set_session_durations(root, session_id, durations):
    """Replace the synthetic session's per-row durations (rows stay 1:1 with the summary)."""
    with h5py.File(root / "audio" / "spectrograms" / f"{session_id}_spectrograms.h5", "a") as f:
        group = f[f"spectrogram/{session_id}"]
        del group["durations"]
        group.create_dataset("durations", data=np.asarray(durations, dtype=np.int64))


def _contract_cfg():
    """The contract-relevant slice of an infer_qlvm_latents block."""
    return {"latent_dim": 2, "masking_type": "sam", "target_shape": [128, 128], "time_stretch": False,
            "length_threshold": None}


def test_enforce_training_contract_returns_the_training_window():
    """Settings that agree with the contract pass, and the contract's duration bound
    is what inference applies."""
    cfg = _contract_cfg()
    params = {"0.weight": np.zeros((2, 4)), "1.weight": np.zeros((2, 2))}
    assert ql.enforce_training_contract(_matching_contract(cfg, length_threshold=100.0), cfg, params) == 100.0
    cfg["length_threshold"] = 100
    assert ql.enforce_training_contract(_matching_contract(cfg, length_threshold=100.0), cfg, params) == 100.0


def test_enforce_training_contract_names_every_mismatch():
    """All disagreements are reported at once, so one run shows everything to fix."""
    cfg = _contract_cfg()
    cfg["length_threshold"] = 50.0
    contract = _matching_contract(cfg, masking_type="none", target_shape=[96, 96], decoder_head="relu", c_dim=1)
    legacy_params = {"0.weight": np.zeros((2, 4)), "1.weight": np.zeros((2, 2))}
    with pytest.raises(ValueError, match="training contract") as excinfo:
        ql.enforce_training_contract(contract, cfg, legacy_params)
    message = str(excinfo.value)
    for key in ("masking_type", "target_shape", "length_threshold", "decoder_head", "c_dim"):
        assert key in message
    assert "time_stretch" not in message


def test_infer_and_merge_failed_write_keeps_previous_summary(tmp_path, mocker):
    """usv_summary.csv carries every other per-USV column, so a write that fails
    part-way must leave the previous file whole and no temporary file behind."""
    rng = np.random.default_rng(10)
    grid = np.ones((8, 8), dtype=np.int16)
    root, session_id, cfg = _make_inference_session(tmp_path, rng)
    cell = _make_model_cell(tmp_path, rng, masking_type="none", floor=0.2, fine_grid=grid, coarse_grid=grid)
    cfg["model_cells"] = {"qlvm": str(cell)}
    summary_path = root / "audio" / f"{session_id}_usv_summary.csv"
    before = summary_path.read_bytes()

    def _partial_write(_self, file, **_kwargs):
        pathlib.Path(file).write_text("usv_id,start\n0000,")
        message = "disk full"
        raise OSError(message)

    mocker.patch("usv_playpen.processing.qlvm_latents.smart_wait")
    mocker.patch.object(pls.DataFrame, "write_csv", _partial_write)
    with pytest.raises(OSError, match="disk full"):
        ql.QLVMLatentInference(
            root_directory=str(root),
            input_parameter_dict={"infer_qlvm_latents": cfg},
            message_output=lambda *_a, **_kw: None,
        ).infer_and_merge()
    assert summary_path.read_bytes() == before
    assert sorted(path.name for path in summary_path.parent.iterdir() if path.is_file()) == [summary_path.name]


def _relu_state_dict(rng, latent_dim=2, prefix="decoder."):
    """Decoder weights in the ReLU-head layout (Linear 0, ReLU 1, Linear 2, convs 4-10),
    as a QLVM model package checkpoint stores them."""
    shapes = {
        "0": ((2048, 2 * latent_dim), (2048,)), "2": ((64 * 8 * 8, 2048), (64 * 8 * 8,)),
        "4": ((64, 32, 3, 3), (32,)), "6": ((32, 16, 3, 3), (16,)), "8": ((16, 8, 3, 3), (8,)), "10": ((8, 1, 3, 3), (1,)),
    }
    return {
        f"{prefix}{idx}.{part}": (rng.standard_normal(shape) * 0.05).astype(np.float32)
        for idx, (weight_shape, bias_shape) in shapes.items()
        for part, shape in (("weight", weight_shape), ("bias", bias_shape))
    }


def test_read_torch_checkpoint_matches_torch_load(tmp_path):
    """A torch zip checkpoint is read bitwise without torch, including non-contiguous
    tensors and nested containers, so a package checkpoint needs no torch install."""
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(11)
    state = {key: torch.from_numpy(value) for key, value in _relu_state_dict(rng).items()}
    state["decoder.transposed"] = torch.arange(12, dtype=torch.float32).reshape(3, 4).t()  # a strided view
    path = tmp_path / "checkpoint.tar"
    torch.save({"model": state, "optimizer": {"state": {0: {"step": torch.tensor(3.0)}}}, "run info": [1.0, 2.0]}, path)

    checkpoint = ql.read_torch_checkpoint(path)
    assert set(checkpoint) == {"model", "optimizer", "run info"}
    for key, tensor in state.items():
        np.testing.assert_array_equal(checkpoint["model"][key], tensor.numpy())
        assert checkpoint["model"][key].dtype == tensor.numpy().dtype
    assert checkpoint["run info"] == [1.0, 2.0]
    params = ql.load_decoder_params(str(path))
    assert "2.weight" in params
    assert "decoder.2.weight" not in params


def test_read_torch_checkpoint_refuses_foreign_globals(tmp_path):
    """Unpickling may only rebuild tensors: a checkpoint naming any other global is
    refused instead of executed."""
    pickled = pickle.dumps({"model": print})
    path = tmp_path / "checkpoint.tar"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("checkpoint/data.pkl", pickled)
        archive.writestr("checkpoint/byteorder", "little")
    with pytest.raises(pickle.UnpicklingError, match=r"builtins\.print"):
        ql.read_torch_checkpoint(path)


def test_normalize_model_inputs_follows_the_contract():
    """Package decoders were fed min-maxed (and, for floor cells, floored then
    re-min-maxed) spectrograms; an input_normalization of "none" the stored values."""
    rng = np.random.default_rng(12)
    specs = rng.uniform(0.1, 3.0, size=(2, 16, 16)).astype(np.float32)
    assert np.array_equal(ql.normalize_model_inputs(specs, {"input_normalization": "none", "floor": None}), specs)

    contract = {"input_normalization": "minmax", "normalization_epsilon": 1e-8, "floor": None}
    got = ql.normalize_model_inputs(specs, contract)
    for row in range(2):
        x = specs[row]
        expected = (x - x.min()) / (np.float32(x.max() - x.min()) + np.float32(1e-8))
        np.testing.assert_array_equal(got[row], expected)

    floored = ql.normalize_model_inputs(specs, {**contract, "floor": 0.2})
    for row in range(2):
        x = specs[row]
        x = (x - x.min()) / (np.float32(x.max() - x.min()) + np.float32(1e-8))
        x = np.clip((x - np.float32(0.2)) / np.float32(0.8), np.float32(0.0), np.float32(1.0))
        expected = (x - x.min()) / (np.float32(x.max() - x.min()) + np.float32(1e-8))
        np.testing.assert_array_equal(floored[row], expected)
    assert floored.dtype == np.float32


def _make_model_cell(
    tmp_path, rng, *, masking_type, floor, fine_grid, coarse_grid, fib_m=8, condition=None, bins=None, require_mask=True,
    cell_name="cell_test", contract_overrides=None,
):
    """Synthesize a QLVM model package cell (checkpoint.tar, training_contract.json,
    cluster/{fine,coarse}/label_grid.npy) with a ReLU-head decoder; with a
    ``condition`` block, a conditional one (one extra decoder input) and its
    ``condition_bins.npz`` (``bins`` = (edges, bin_mean) for phase 10, or a dict of
    the phase 11 keys, see :func:`_phase11_bins`). ``require_mask`` True, as in every
    v2 and v3 cell, says its corpus kept only calls with a SAM mask. The cell is
    ``<tmp_path>/pkg/phase_test/<cell_name>``, so several cells share one package.
    ``contract_overrides`` replaces training-contract entries (e.g. ``target_shape``)."""
    torch = pytest.importorskip("torch")
    cell = tmp_path / "pkg" / "phase_test" / cell_name
    (cell / "cluster" / "fine").mkdir(parents=True)
    (cell / "cluster" / "coarse").mkdir(parents=True)
    arrays = _relu_state_dict(rng)
    if condition is not None:
        arrays["decoder.0.weight"] = (rng.standard_normal((2048, 5)) * 0.05).astype(np.float32)
        if isinstance(bins, dict):
            np.savez(cell / "condition_bins.npz", **bins)
        else:
            edges, bin_mean = bins
            np.savez(cell / "condition_bins.npz", conditional=np.array(condition["name"]), n_bins=np.array(len(bin_mean)),
                     edges=np.asarray(edges, dtype=np.float64), bin_mean=np.asarray(bin_mean, dtype=np.float32),
                     bin_count=np.ones(len(bin_mean), dtype=np.int64))
    state = {key: torch.from_numpy(value) for key, value in arrays.items()}
    torch.save({"model": state, "optimizer": {}, "run info": []}, cell / "checkpoint.tar")
    contract = {
        "decoder_head": "relu", "latent_dim": 2, "c_dim": 0 if condition is None else 1,
        "conditional": None if condition is None else condition["name"],
        "input_normalization": "minmax", "normalization_epsilon": 1e-8,
        "masking_type": masking_type, "floor": floor,
        "target_shape": [128, 128], "time_stretch": False, "length_threshold": 100.0, "require_mask": require_mask,
        "embedding_lattice_type": "fibonacci", "embedding_fib_m": fib_m,
        "training_lattice_type": "fibonacci", "training_fib_m": 5, "validation_fib_m": 6, "condition": condition,
    }
    if condition is not None:
        contract["condition_bins"] = "condition_bins.npz"
    if contract_overrides is not None:
        contract.update(contract_overrides)
    (cell / "training_contract.json").write_text(json.dumps(contract))
    np.save(cell / "cluster" / "fine" / "label_grid.npy", fine_grid)
    np.save(cell / "cluster" / "coarse" / "label_grid.npy", coarse_grid)
    return cell


def _phase11_bins(name, c_min, c_max, step):
    """A phase 11 condition_bins.npz, as the v3 package writes it: the capped training
    edges and groups, the training c range and a decode grid from its low end; no
    bin_mean."""
    grid = c_min + step * np.arange(round((c_max - c_min) / step) + 1)
    return {
        "conditional": np.array(name), "bin_scheme": np.array("quantile_capped"), "n_bins": np.array(2),
        "edges": np.array([c_min, 0.5, c_max]), "group_ids": np.array([0, 1]), "group_sizes": np.array([10, 10]),
        "group_means": np.array([0.25, 0.75], dtype=np.float32),
        "train_c_min": np.float32(c_min), "train_c_max": np.float32(c_max),
        "decode_grid": grid.astype(np.float64), "decode_grid_step": np.float64(step), "run_decode_grid_step": np.float64(0.01),
    }


def test_infer_and_merge_with_a_model_package_cell(tmp_path, mocker):
    """A model_cells cell's ReLU checkpoint, contract and Fibonacci lattice are used:
    calls outside its duration window are null and inputs are min-maxed and floored.
    The cell's label_grid.npy files are not read: only the qlvm1 / qlvm2 coordinates are
    written (no qlvm_category / qlvm_supercategory), and no qlvm_model column."""
    rng = np.random.default_rng(13)
    res = 8
    fine_grid = rng.integers(101, 117, size=(res, res)).astype(np.int16)
    coarse_grid = rng.integers(201, 208, size=(res, res)).astype(np.int16)
    root, session_id, cfg = _make_inference_session(tmp_path, rng)
    _set_session_durations(root, session_id, [128, 0, 64])  # row 0 is outside the cell's window (100)
    cell = _make_model_cell(tmp_path, rng, masking_type="none", floor=0.2, fine_grid=fine_grid, coarse_grid=coarse_grid)
    cfg["model_cells"] = {"qlvm": str(cell)}

    captured = {}
    real_embed = ql.embed_data

    def _capture(lattice, data, params, *block_sizes):
        captured["lattice"], captured["data"], captured["params"] = np.asarray(lattice), np.asarray(data), params
        return real_embed(lattice, data, params, *block_sizes)

    mocker.patch("usv_playpen.processing.qlvm_latents.smart_wait")
    mocker.patch("usv_playpen.processing.qlvm_latents.embed_data", side_effect=_capture)
    messages = []
    ql.QLVMLatentInference(
        root_directory=str(root),
        input_parameter_dict={"infer_qlvm_latents": cfg},
        message_output=messages.append,
    ).infer_and_merge()

    assert captured["lattice"].shape == (21, 2)                       # fib(8) points
    assert "2.weight" in captured["params"]
    assert captured["data"].shape == (1, 1, 128, 128)
    assert captured["data"].min() == 0.0
    assert captured["data"].max() == pytest.approx(1.0, abs=1e-6)
    assert any("cell_test" in message and "relu head" in message for message in messages)
    df = pls.read_csv(root / "audio" / f"{session_id}_usv_summary.csv")
    assert df["qlvm1"][0] is None
    assert df["qlvm1"][2] is not None
    for column in ("qlvm_category", "qlvm_supercategory", "qlvm_model"):
        assert column not in df.columns


def test_compute_condition_values_follow_the_contract_definitions():
    """Duration c is normalized on the corpus range; mean-frequency c is the energy
    centroid of the masked call's frequency rows, before any min-max."""
    durations = np.array([8, 60, 127], dtype=np.int64)
    duration = {"name": "duration", "duration_min": 8, "duration_max": 127, "epsilon": 1e-8}
    np.testing.assert_array_equal(
        ql.compute_condition_values(duration, durations, None),
        (durations.astype(np.float32) - np.float32(8)) / (np.float32(119) + np.float32(1e-8)),
    )
    specs = np.zeros((2, 4, 3), dtype=np.float32)
    specs[0, 1, :] = 2.0                                    # all energy in row 1 of 4
    specs[1, 0, 0], specs[1, 3, 2] = 1.0, 3.0               # centroid (0*1 + 3*3) / 4 = 2.25
    mean_freq = {"name": "mean_freq", "spectrogram": "masked", "epsilon": 1e-8}
    np.testing.assert_allclose(ql.compute_condition_values(mean_freq, durations[:2], specs), [0.25, 2.25 / 4], rtol=1e-6)
    with pytest.raises(ValueError, match="masked spectrograms"):
        ql.compute_condition_values(mean_freq, durations[:2], None)


def test_frozen_condition_values_take_bin_means_and_clamp_the_range():
    """New calls decode at the corpus mean of their bin; values beyond the corpus
    range fall in the outermost bins, and a value on an edge belongs to the upper bin."""
    bins = {"edges": np.array([0.0, 0.3, 0.6, 1.0]), "bin_mean": np.array([0.1, 0.45, 0.8], dtype=np.float32)}
    got = ql.frozen_condition_values(np.array([-0.2, 0.1, 0.3, 0.59, 0.61, 1.4]), bins)
    np.testing.assert_array_equal(got, np.array([0.1, 0.1, 0.45, 0.45, 0.8, 0.8], dtype=np.float32))
    assert got.dtype == np.float32


def test_infer_and_merge_conditional_cell_decodes_at_frozen_mean_freq(tmp_path, mocker):
    """A mean-frequency cell fed unmasked spectrograms still needs the masks for c:
    each call is decoded at the frozen bin mean of its masked centroid."""
    rng = np.random.default_rng(16)
    res = 8
    grid = rng.integers(1, 5, size=(res, res)).astype(np.int16)
    root, session_id, cfg = _make_inference_session(tmp_path, rng, with_masks=False)
    _set_session_durations(root, session_id, [64, 0, 64])
    n_f = n_t = 128
    low_rows, high_rows = np.zeros((n_f, n_t), dtype=bool), np.zeros((n_f, n_t), dtype=bool)
    low_rows[:32, :] = True                                # row 0's call sits in the low rows
    high_rows[96:, :] = True                               # row 2's call in the high rows
    _write_session_masks(root, session_id, {0: low_rows, 2: high_rows})
    condition = {"name": "mean_freq", "spectrogram": "masked", "epsilon": 1e-8}
    edges, bin_mean = np.array([0.0, 0.5, 1.0]), np.array([0.2, 0.8], dtype=np.float32)
    cell = _make_model_cell(tmp_path, rng, masking_type="none", floor=0.2, fine_grid=grid, coarse_grid=grid,
                            condition=condition, bins=(edges, bin_mean))
    cfg["model_cells"] = {"qlvm": str(cell)}

    captured = {}
    real_embed = ql.embed_data

    def _capture(lattice, data, params, lattice_batch_size, data_batch_size, condition_values=None):
        captured["condition_values"] = condition_values
        return real_embed(lattice, data, params, lattice_batch_size, data_batch_size, condition_values)

    mocker.patch("usv_playpen.processing.qlvm_latents.smart_wait")
    mocker.patch("usv_playpen.processing.qlvm_latents.embed_data", side_effect=_capture)
    messages = []
    ql.QLVMLatentInference(
        root_directory=str(root),
        input_parameter_dict={"infer_qlvm_latents": cfg},
        message_output=messages.append,
    ).infer_and_merge()

    np.testing.assert_array_equal(captured["condition_values"], np.array([0.2, 0.8], dtype=np.float32))
    assert any("Conditioning on mean_freq" in message for message in messages)
    df = pls.read_csv(root / "audio" / f"{session_id}_usv_summary.csv")
    assert df["qlvm1"][0] is not None
    assert df["qlvm1"][2] is not None


def test_frozen_condition_values_phase11_exact_clamps_to_the_training_range():
    """Duration and bandwidth cells decode each call at its own c, clamped to the
    range the cell was trained on."""
    bins = _phase11_bins("bandwidth", 0.05, 0.95, 0.01)
    got = ql.frozen_condition_values(np.array([-0.1, 0.05, 0.3337, 0.95, 1.2]), bins, "exact")
    np.testing.assert_array_equal(got, np.array([0.05, 0.05, 0.3337, 0.95, 0.95], dtype=np.float32))
    assert got.dtype == np.float32


def test_frozen_condition_values_phase11_grid_snaps_to_the_nearest_grid_point():
    """Mean-frequency and loudness cells decode at the nearest decode_grid point: a
    midpoint goes to the lower point and a c beyond the grid to its end point."""
    bins = _phase11_bins("loudness", 0.0, 1.0, 0.25)
    got = ql.frozen_condition_values(np.array([-0.3, 0.1, 0.125, 0.126, 0.6, 1.4]), bins, "grid")
    np.testing.assert_array_equal(got, np.array([0.0, 0.0, 0.0, 0.25, 0.5, 1.0], dtype=np.float32))
    with pytest.raises(ValueError, match="decode 'grid' or 'exact'"):
        ql.frozen_condition_values(np.array([0.5]), bins)


def test_compute_condition_values_maps_bandwidth_and_loudness_from_raw_units():
    """Phase 11's frozen maps: bandwidth Hz / 90 kHz and loudness dB over the
    contract's db_range, clipped to [0, 1], in float64 rounded to float32."""
    durations = np.zeros(5, dtype=np.int64)
    bandwidth = {"name": "bandwidth", "decode": "exact"}
    np.testing.assert_array_equal(
        ql.compute_condition_values(bandwidth, durations, None, np.array([0.0, 45000.0, 90000.0, 99000.0, 3000.3])),
        np.array([0.0, 0.5, 1.0, 1.0, 3000.3 / 90000.0], dtype=np.float32),
    )
    loudness = {"name": "loudness", "db_range": [28.83, 95.85], "decode": "grid"}
    np.testing.assert_array_equal(
        ql.compute_condition_values(loudness, durations, None, np.array([28.83, 95.85, 62.34, 10.0, 120.0])),
        np.array([0.0, 1.0, (62.34 - 28.83) / (95.85 - 28.83), 0.0, 1.0], dtype=np.float32),
    )
    with pytest.raises(ValueError, match="needs each call's raw value"):
        ql.compute_condition_values(loudness, durations, None)


def test_enforce_training_contract_accepts_the_phase11_conditions():
    """Bandwidth and loudness conditions are embeddable; an unknown condition or
    decode rule is refused."""
    cfg = _contract_cfg()
    params = {"0.weight": np.zeros((2, 4)), "1.weight": np.zeros((2, 2))}
    for condition in ({"name": "bandwidth", "decode": "exact"}, {"name": "loudness", "db_range": [0, 1], "decode": "grid"}):
        assert ql.enforce_training_contract(_matching_contract(cfg, c_dim=1, condition=condition), cfg, params) == 128.0
    with pytest.raises(ValueError, match=r"condition\.decode"):
        ql.enforce_training_contract(
            _matching_contract(cfg, c_dim=1, condition={"name": "loudness", "decode": "batch_mean"}), cfg, params
        )
    with pytest.raises(ValueError, match="c_dim"):
        ql.enforce_training_contract(_matching_contract(cfg, c_dim=1, condition={"name": "pitch"}), cfg, params)


@pytest.mark.parametrize(
    ("condition", "bins"),
    [
        ({"name": "duration", "duration_min": 8, "duration_max": 127, "epsilon": 1e-8, "decode": "exact"},
         (np.array([0.0, 0.5, 1.0]), np.array([0.2, 0.8], dtype=np.float32))),
        ({"name": "duration", "duration_min": 8, "duration_max": 127, "epsilon": 1e-8},
         _phase11_bins("duration", 0.0, 1.0, 0.01)),
    ],
)
def test_load_model_cell_refuses_bins_that_disagree_with_the_contract(tmp_path, condition, bins):
    """Phase 10 bins (bin means) with a phase 11 decode rule, or phase 11 bins (no
    bin_mean) without one, would decode at the wrong c: the cell is refused."""
    rng = np.random.default_rng(20)
    grid = np.ones((8, 8), dtype=np.int16)
    cell = _make_model_cell(tmp_path, rng, masking_type="none", floor=0.2, fine_grid=grid, coarse_grid=grid,
                            condition=condition, bins=bins)
    with pytest.raises(ValueError, match=r"condition_bins\.npz holds"):
        ql.load_model_cell(str(cell))


def _phase11_session(tmp_path, rng, condition, bins, summary_columns=None):
    """A session whose rows 0 and 2 are real calls with SAM masks in the low and the
    high frequency rows, a phase 11 cell for ``condition`` and settings pointing at
    it; ``summary_columns`` are added to usv_summary.csv."""
    grid = rng.integers(1, 5, size=(8, 8)).astype(np.int16)
    root, session_id, cfg = _make_inference_session(tmp_path, rng, with_masks=False)
    low_rows, high_rows = np.zeros((128, 128), dtype=bool), np.zeros((128, 128), dtype=bool)
    low_rows[:32, :] = True
    high_rows[96:, :] = True
    _write_session_masks(root, session_id, {0: low_rows, 2: high_rows})
    if summary_columns:
        summary_path = root / "audio" / f"{session_id}_usv_summary.csv"
        pls.read_csv(summary_path).with_columns(**summary_columns).write_csv(summary_path)
    cell = _make_model_cell(tmp_path, rng, masking_type="none", floor=0.2, fine_grid=grid, coarse_grid=grid,
                            condition=condition, bins=bins)
    cfg["model_cells"] = {"qlvm": str(cell)}
    return root, session_id, cfg, (low_rows, high_rows)


def _run_capturing_c(root, cfg, mocker, extra_settings=None):
    """Run infer_and_merge, returning the condition values embed_data was given and
    the messages."""
    captured = {}
    real_embed = ql.embed_data

    def _capture(lattice, data, params, lattice_batch_size, data_batch_size, condition_values=None):
        captured["condition_values"] = condition_values
        return real_embed(lattice, data, params, lattice_batch_size, data_batch_size, condition_values)

    mocker.patch("usv_playpen.processing.qlvm_latents.smart_wait")
    mocker.patch("usv_playpen.processing.qlvm_latents.embed_data", side_effect=_capture)
    messages = []
    ql.QLVMLatentInference(
        root_directory=str(root),
        input_parameter_dict={"infer_qlvm_latents": cfg, **(extra_settings or {})},
        message_output=messages.append,
    ).infer_and_merge()
    return captured["condition_values"], messages


def test_infer_and_merge_phase11_bandwidth_reads_the_summary(tmp_path, mocker):
    """A bandwidth cell decodes each call at its summary freq_bandwidth_hz / 90 kHz,
    clamped to the training range; a call without a bandwidth gets null columns."""
    rng = np.random.default_rng(21)
    condition = {"name": "bandwidth", "source": "usv_summary freq_bandwidth_hz", "decode": "exact"}
    root, session_id, cfg, _masks = _phase11_session(
        tmp_path, rng, condition, _phase11_bins("bandwidth", 0.05, 0.95, 0.01),
        summary_columns={"freq_bandwidth_hz": pls.Series([89100.0, 5000.0, None])},
    )

    condition_values, messages = _run_capturing_c(root, cfg, mocker)

    np.testing.assert_array_equal(condition_values, np.array([0.95], dtype=np.float32))   # 0.99, clamped
    assert any("1 USVs without a bandwidth value" in message for message in messages)
    assert any("1 USVs have a bandwidth value outside the training range" in message for message in messages)
    df = pls.read_csv(root / "audio" / f"{session_id}_usv_summary.csv")
    assert df["qlvm1"][0] is not None
    assert df["qlvm1"][1] is None
    assert df["qlvm1"][2] is None
    assert df["freq_bandwidth_hz"][0] == 89100.0


def test_infer_and_merge_phase11_grid_counts_against_the_decode_grid(tmp_path, mocker):
    """A 'grid' cell clips to the ends of its decode grid, which can extend past the
    training range (as the v3 mean-frequency grid does), so the log counts calls against
    the grid's ends: a call between the training maximum and the grid's end decodes at
    its own grid point and is not counted, while one beyond the grid's end is."""
    rng = np.random.default_rng(23)
    condition = {"name": "bandwidth", "source": "usv_summary freq_bandwidth_hz", "decode": "grid"}
    bins = _phase11_bins("bandwidth", 0.05, 0.95, 0.01)
    bins["train_c_max"] = np.float32(0.90)
    root, _session_id, cfg, _masks = _phase11_session(
        tmp_path, rng, condition, bins,
        summary_columns={"freq_bandwidth_hz": pls.Series([0.93 * 90000.0, 5000.0, 0.99 * 90000.0])},
    )

    condition_values, messages = _run_capturing_c(root, cfg, mocker)

    # Row 0 (0.93, past the training maximum but inside the grid) keeps its own grid point;
    # row 2 (0.99, beyond the grid) decodes at the grid's end.
    np.testing.assert_allclose(condition_values, np.array([0.93, 0.95], dtype=np.float32), atol=1e-6)
    assert any("1 USVs have a bandwidth value outside the decode grid [0.0500, 0.9500]" in message
               for message in messages)


def test_infer_and_merge_phase11_bandwidth_needs_the_summary_column(tmp_path, mocker):
    """Without generate-usv-acoustic-features' freq_bandwidth_hz there is no c: the
    run stops before anything is written."""
    rng = np.random.default_rng(22)
    condition = {"name": "bandwidth", "decode": "exact"}
    root, session_id, cfg, _masks = _phase11_session(tmp_path, rng, condition, _phase11_bins("bandwidth", 0.05, 0.95, 0.01))
    summary_path = root / "audio" / f"{session_id}_usv_summary.csv"
    before = summary_path.read_bytes()
    mocker.patch("usv_playpen.processing.qlvm_latents.smart_wait")
    with pytest.raises(ValueError, match="no freq_bandwidth_hz column"):
        ql.QLVMLatentInference(
            root_directory=str(root),
            input_parameter_dict={"infer_qlvm_latents": cfg},
            message_output=lambda *_a, **_kw: None,
        ).infer_and_merge()
    assert summary_path.read_bytes() == before


def test_infer_and_merge_phase11_loudness_reads_the_summary_loudness(tmp_path, mocker):
    """A loudness cell is decoded at each call's loudness_db (the absolute level
    generate-usv-acoustic-features measured from the audio over the SAM mask), mapped
    through db_range and snapped to the decode grid; a call without a loudness_db
    gets null columns, and a summary without the column raises."""
    rng = np.random.default_rng(23)
    condition = {"name": "loudness", "db_range": [28.83, 95.85], "decode": "grid"}
    root, session_id, cfg, _masks = _phase11_session(
        tmp_path, rng, condition, _phase11_bins("loudness", 0.0, 1.0, 0.0025),
        summary_columns={"loudness_db": pls.Series([62.34, 50.0, None])},
    )

    condition_values, messages = _run_capturing_c(root, cfg, mocker)

    own = np.clip((62.34 - 28.83) / (95.85 - 28.83), 0.0, 1.0)
    grid = 0.0025 * np.arange(401)
    np.testing.assert_array_equal(condition_values, np.array([grid[np.argmin(np.abs(grid - own))]], dtype=np.float32))
    assert any("1 USVs without a loudness value" in message for message in messages)
    df = pls.read_csv(root / "audio" / f"{session_id}_usv_summary.csv")
    assert df["qlvm1"][0] is not None
    assert df["qlvm1"][2] is None

    summary_path = root / "audio" / f"{session_id}_usv_summary.csv"
    pls.read_csv(summary_path).drop("loudness_db").write_csv(summary_path)
    with pytest.raises(ValueError, match="no loudness_db column"):
        _run_capturing_c(root, cfg, mocker)


_ENTROPY_CONDITION = {
    "name": "spectral_entropy",
    "source": "usv_summary spectral_entropy",
    "entropy_min": 1.051373310783529,
    "entropy_max": 4.82961777804439,
    "decode": "grid",
}


def test_compute_condition_values_scales_spectral_entropy_by_the_training_range():
    """The spectral entropy map: (H - entropy_min) / (entropy_max - entropy_min) in
    float64 rounded to float32, clamped to [0, 1] beyond the training range, NaN
    kept for a call without an entropy."""
    low, high = _ENTROPY_CONDITION["entropy_min"], _ENTROPY_CONDITION["entropy_max"]
    raw = np.array([low, high, 2.5, 0.3, 6.0, np.nan])
    got = ql.compute_condition_values(_ENTROPY_CONDITION, np.zeros(6, dtype=np.int64), None, raw)
    assert got.dtype == np.float32
    np.testing.assert_array_equal(got[:5], np.array([0.0, 1.0, (2.5 - low) / (high - low), 0.0, 1.0], dtype=np.float32))
    assert np.isnan(got[5])
    with pytest.raises(ValueError, match="needs each call's raw value"):
        ql.compute_condition_values(_ENTROPY_CONDITION, np.zeros(1, dtype=np.int64), None)


def test_enforce_training_contract_accepts_spectral_entropy_with_a_range():
    """A spectral entropy cell is embeddable when its block carries a non-empty
    training range; a missing or degenerate range is refused."""
    cfg = _contract_cfg()
    params = {"0.weight": np.zeros((2, 4)), "1.weight": np.zeros((2, 2))}
    assert ql.enforce_training_contract(_matching_contract(cfg, c_dim=1, condition=_ENTROPY_CONDITION), cfg, params) == 128.0
    for broken in ({**_ENTROPY_CONDITION, "entropy_max": _ENTROPY_CONDITION["entropy_min"]},
                   {key: value for key, value in _ENTROPY_CONDITION.items() if key != "entropy_min"}):
        with pytest.raises(ValueError, match="entropy_min < entropy_max"):
            ql.enforce_training_contract(_matching_contract(cfg, c_dim=1, condition=broken), cfg, params)


def test_infer_and_merge_spectral_entropy_reads_the_summary_and_snaps_to_the_grid(tmp_path, mocker):
    """A spectral entropy cell is decoded at each call's summary spectral_entropy,
    scaled by the contract's training range, clamped to [0, 1] and snapped to the
    nearest decode-grid point; a call without an entropy gets null columns, and a
    summary without the column raises."""
    rng = np.random.default_rng(24)
    low, high = _ENTROPY_CONDITION["entropy_min"], _ENTROPY_CONDITION["entropy_max"]
    root, session_id, cfg, _masks = _phase11_session(
        tmp_path, rng, _ENTROPY_CONDITION, _phase11_bins("spectral_entropy", 0.0, 1.0, 0.0025),
        summary_columns={"spectral_entropy": pls.Series([low + 0.6182 * (high - low), 3.0, None])},
    )

    condition_values, messages = _run_capturing_c(root, cfg, mocker)

    np.testing.assert_allclose(condition_values, np.array([0.6175], dtype=np.float32), atol=1e-6)
    assert any("1 USVs without a spectral_entropy value" in message for message in messages)
    df = pls.read_csv(root / "audio" / f"{session_id}_usv_summary.csv")
    assert df["qlvm1"][0] is not None
    assert df["qlvm1"][2] is None

    summary_path = root / "audio" / f"{session_id}_usv_summary.csv"
    pls.read_csv(summary_path).with_columns(spectral_entropy=pls.Series([high + 2.0, 3.0, low - 0.5])).write_csv(summary_path)
    condition_values, messages = _run_capturing_c(root, cfg, mocker)
    np.testing.assert_array_equal(condition_values, np.array([1.0, 0.0], dtype=np.float32))
    assert any("2 USVs have a spectral_entropy outside the training range" in message for message in messages)

    pls.read_csv(summary_path).drop("spectral_entropy").write_csv(summary_path)
    with pytest.raises(ValueError, match="no spectral_entropy column"):
        _run_capturing_c(root, cfg, mocker)


def test_infer_and_merge_model_cell_refuses_wrong_masking(tmp_path, mocker):
    """A masked package decoder must not be fed unmasked spectrograms: the settings'
    masking_type is checked against the cell's contract before anything runs, and the
    summary is left as it was."""
    rng = np.random.default_rng(14)
    res = 8
    grid = np.ones((res, res), dtype=np.int16)
    root, session_id, cfg = _make_inference_session(tmp_path, rng)
    cell = _make_model_cell(tmp_path, rng, masking_type="sam", floor=None, fine_grid=grid, coarse_grid=grid)
    cfg["model_cells"] = {"qlvm": str(cell)}
    cfg["masking_type"] = "none"
    summary_path = root / "audio" / f"{session_id}_usv_summary.csv"
    before = summary_path.read_bytes()

    mocker.patch("usv_playpen.processing.qlvm_latents.smart_wait")
    with pytest.raises(ValueError, match="masking_type: settings 'none', trained 'sam'"):
        ql.QLVMLatentInference(
            root_directory=str(root),
            input_parameter_dict={"infer_qlvm_latents": cfg},
            message_output=lambda *_a, **_kw: None,
        ).infer_and_merge()
    assert summary_path.read_bytes() == before


def _use_decoder(tmp_path, rng, cfg, decoder, grid):
    """Point ``cfg`` at one kind of model cell for the SAM-mask rule: ``"sam"`` (a
    cell trained with masking_type sam on every call), ``"require_mask"`` (an
    unmasked cell whose set kept only calls with a mask), ``"mean_freq"`` (an
    unmasked cell conditioned on mean frequency, require_mask false) or
    ``"unmasked"`` (an unmasked cell whose set kept every call)."""
    condition, bins = None, None
    if decoder == "mean_freq":
        condition = {"name": "mean_freq", "spectrogram": "masked", "epsilon": 1e-8}
        bins = (np.array([0.0, 0.5, 1.0]), np.array([0.2, 0.8], dtype=np.float32))
    cfg["masking_type"] = "sam" if decoder == "sam" else "none"
    cell = _make_model_cell(tmp_path, rng, masking_type=cfg["masking_type"], floor=None, fine_grid=grid,
                            coarse_grid=grid, condition=condition, bins=bins,
                            require_mask=decoder == "require_mask")
    cfg["model_cells"] = {"qlvm": str(cell)}


def _fake_embed_counting(captured):
    """An embed_data stand-in that records how many spectrograms it was given and
    places every one at the torus center."""

    def _fake_embed(lattice, data, params, *_rest):
        captured["n_embedded"] = data.shape[0]
        return np.full((data.shape[0], 2), 0.5, dtype=np.float64)

    return _fake_embed


@pytest.mark.parametrize(
    ("decoder", "maskless_embedded"),
    [("sam", True), ("require_mask", False), ("mean_freq", False), ("unmasked", True)],
)
def test_infer_and_merge_nulls_calls_without_a_sam_mask(tmp_path, mocker, decoder, maskless_embedded):
    """build_session_masks gives a call without mask instances an all-ones mask. A
    require_mask training set left such calls out, and a mean-frequency cell would
    compute c over the whole call, so those decoders give them null qlvm_* columns.
    A sam decoder whose set kept them under that all-ones mask still embeds them, as
    does an unmasked decoder."""
    rng = np.random.default_rng(17)
    grid = np.ones((8, 8), dtype=np.int16)
    root, session_id, cfg = _make_inference_session(tmp_path, rng, with_masks=False)
    _write_session_masks(root, session_id, {0: np.ones((128, 128), dtype=bool)})  # row 2: no mask instance
    _use_decoder(tmp_path, rng, cfg, decoder, grid)

    captured = {}
    mocker.patch("usv_playpen.processing.qlvm_latents.smart_wait")
    mocker.patch("usv_playpen.processing.qlvm_latents.embed_data", side_effect=_fake_embed_counting(captured))
    messages = []
    ql.QLVMLatentInference(
        root_directory=str(root),
        input_parameter_dict={"infer_qlvm_latents": cfg},
        message_output=messages.append,
    ).infer_and_merge()

    df = pls.read_csv(root / "audio" / f"{session_id}_usv_summary.csv")
    assert df["qlvm1"][0] is not None
    assert df["qlvm1"][1] is None
    assert (df["qlvm1"][2] is not None) is maskless_embedded
    assert captured["n_embedded"] == (2 if maskless_embedded else 1)
    assert any("1 USVs without a SAM mask" in message for message in messages) is not maskless_embedded


@pytest.mark.parametrize("decoder", ["sam", "require_mask", "mean_freq"])
def test_infer_and_merge_refuses_a_session_without_masks(tmp_path, mocker, decoder):
    """A session H5 without a mask/<session> group would give every call an all-ones
    mask, so a decoder that needs masks stops before anything is written."""
    rng = np.random.default_rng(18)
    grid = np.ones((8, 8), dtype=np.int16)
    root, session_id, cfg = _make_inference_session(tmp_path, rng, with_masks=False)
    _use_decoder(tmp_path, rng, cfg, decoder, grid)
    summary_path = root / "audio" / f"{session_id}_usv_summary.csv"
    before = summary_path.read_bytes()

    mocker.patch("usv_playpen.processing.qlvm_latents.smart_wait")
    with pytest.raises(ValueError, match=rf"no mask/{session_id} group"):
        ql.QLVMLatentInference(
            root_directory=str(root),
            input_parameter_dict={"infer_qlvm_latents": cfg},
            message_output=lambda *_a, **_kw: None,
        ).infer_and_merge()
    assert summary_path.read_bytes() == before


def test_infer_and_merge_unmasked_decoder_needs_no_masks(tmp_path, mocker):
    """A decoder trained on every call without masks embeds a session that has no
    mask/<session> group."""
    rng = np.random.default_rng(19)
    grid = np.ones((8, 8), dtype=np.int16)
    root, _session_id, cfg = _make_inference_session(tmp_path, rng, with_masks=False)
    _use_decoder(tmp_path, rng, cfg, "unmasked", grid)

    captured = {}
    mocker.patch("usv_playpen.processing.qlvm_latents.smart_wait")
    mocker.patch("usv_playpen.processing.qlvm_latents.embed_data", side_effect=_fake_embed_counting(captured))
    ql.QLVMLatentInference(
        root_directory=str(root),
        input_parameter_dict={"infer_qlvm_latents": cfg},
        message_output=lambda *_a, **_kw: None,
    ).infer_and_merge()

    assert captured["n_embedded"] == 2


def _model_cells_session(tmp_path, rng, prefixes=("qlvm", "qlvm_x")):
    """A session whose rows 0 and 2 are real 64-bin calls with a SAM mask (row 1 a
    placeholder), one unmasked floor-trained package cell per prefix (all in one
    package, ``<tmp_path>/pkg``) and settings listing them in ``model_cells``. Every
    cell ships its own (unread) label grids (:func:`_model_cell_grids`), as a clustered
    package cell does."""
    root, session_id, cfg = _make_inference_session(tmp_path, rng)
    for index, prefix in enumerate(prefixes):
        fine_grid, coarse_grid = _model_cell_grids(index)
        cfg["model_cells"][prefix] = str(_make_model_cell(
            tmp_path, rng, masking_type="none", floor=0.2, fine_grid=fine_grid, coarse_grid=coarse_grid,
            cell_name=f"cell_{prefix}",
        ))
    return root, session_id, cfg


def _model_cell_grids(index):
    """Distinct (8, 8) fine and coarse label grids of the ``index``-th cell of
    :func:`_model_cells_session`: every fine pixel its own label (1..64, offset by
    ``1000 * index``), every coarse row its own label (101..108, same offset)."""
    fine = (np.arange(64).reshape(8, 8) + 1 + 1000 * index).astype(np.int16)
    coarse = (np.repeat(np.arange(8)[:, None], 8, axis=1) + 101 + 1000 * index).astype(np.int16)
    return fine, coarse


def _write_fake_package(tmp_path, root, session_id, cfg, rng, *, baseline_session=None, sha256=None,
                        durations=(64.0, 64.0), mask_counts=(1.0, 1.0)):
    """Give the ``<tmp_path>/pkg`` package of ``cfg['model_cells']`` what the package
    route reads: a SESSION_H5_BASELINE.tsv (for ``baseline_session``, default the
    session, with ``sha256``, default the session H5's) and, per cell, a
    recon_mse_breakdown.npz / posterior_cache.npz holding the session's rows 0 and 2
    (with the given package ``durations`` / ``mask_counts``) plus one row of another
    session. Returns each prefix's expected (2, 2) coordinates of rows 0 and 2."""
    h5_path = root / "audio" / "spectrograms" / f"{session_id}_spectrograms.h5"
    digest = file_sha256(h5_path) if sha256 is None else sha256
    listed = session_id if baseline_session is None else baseline_session
    (tmp_path / "pkg" / "SESSION_H5_BASELINE.tsv").write_text(
        "session\th5_rows\tcorpus_rows\tbytes\tsha256\tpath\n"
        f"{listed}\t3\t2\t{h5_path.stat().st_size}\t{digest}\tBartul/Data/{listed}/audio/spectrograms/{listed}_spectrograms.h5\n"
    )
    expected = {}
    for prefix, cell_directory in cfg["model_cells"].items():
        cell = pathlib.Path(cell_directory)
        angles = rng.uniform(0.0, 2 * np.pi, size=(3, 2))
        torus_weighted = np.concatenate([np.cos(angles), np.sin(angles)], axis=1).astype(np.float32)
        np.savez(cell / "posterior_cache.npz", torus_weighted=torus_weighted)
        np.savez(
            cell / "recon_mse_breakdown.npz",
            spec_id=np.array(["20990101_000000_0", f"{session_id}_0", f"{session_id}_2"]),
            durations=np.array([50.0, *durations], dtype=np.float32),
            mask_counts=np.array([2.0, *mask_counts], dtype=np.float32),
        )
        expected[prefix] = np.asarray(ql.torus_basis_reverse(jnp.asarray(torus_weighted[1:])), dtype=np.float64)
    return expected


def _run_model_cells(root, cfg, mocker):
    """Run infer_and_merge with embed_data replaced by a stand-in that records how many
    spectrograms each call embedded and places every one at the torus center; returns
    (embedded counts per call, messages)."""
    embedded = []

    def _fake_embed(lattice, data, params, *_rest):
        embedded.append(data.shape[0])
        return np.full((data.shape[0], 2), 0.5, dtype=np.float64)

    mocker.patch("usv_playpen.processing.qlvm_latents.smart_wait")
    mocker.patch("usv_playpen.processing.qlvm_latents.embed_data", side_effect=_fake_embed)
    messages = []
    ql.QLVMLatentInference(
        root_directory=str(root),
        input_parameter_dict={"infer_qlvm_latents": cfg},
        message_output=messages.append,
    ).infer_and_merge()
    return embedded, messages


def test_model_cells_write_prefixed_coordinates_only(tmp_path, mocker):
    """Every listed cell places the session and each prefix gets exactly its coordinates
    <prefix>1/<prefix>2, null where the call was not placed, and no cluster-label column
    (the cells' label grids are not read). No model column is written; the stale category
    (qlvm_category), legacy coarse-level (qlvm_supercategory, qlvm_x_supercategory),
    category confidence, model and coordinate columns of the listed prefixes are replaced
    or removed; other columns stay, after the canonical ones."""
    rng = np.random.default_rng(30)
    root, session_id, cfg = _model_cells_session(tmp_path, rng)
    summary_path = root / "audio" / f"{session_id}_usv_summary.csv"
    pls.read_csv(summary_path).with_columns(
        quality=pls.Series([0.11, 0.22, 0.33]),
        qlvm_category_agreement=pls.Series([0.5, None, 0.5]),
        qlvm_category=pls.Series([3, None, 4]),
        qlvm_supercategory=pls.Series([1, None, 2]),
        qlvm_model=pls.Series(["old", None, "old"]),
        qlvm_x1=pls.Series([9.0, 9.0, 9.0]),
        qlvm_x_supercategory=pls.Series([5, 5, 5]),
    ).write_csv(summary_path)

    mocker.patch("usv_playpen.processing.qlvm_latents.smart_wait")
    messages = []
    ql.QLVMLatentInference(
        root_directory=str(root),
        input_parameter_dict={"infer_qlvm_latents": cfg},
        message_output=messages.append,
    ).infer_and_merge()

    df = pls.read_csv(summary_path)
    assert df.columns == [
        "usv_id", "start", "stop", "usv", "squeak", "qlvm1", "qlvm2", "quality", "qlvm_x1", "qlvm_x2",
    ]
    for column in ("qlvm1", "qlvm2", "qlvm_x1", "qlvm_x2"):
        assert df[column].dtype == pls.Float64
        assert df[column][1] is None
        assert 0.0 <= df[column][0] < 1.0
        assert 0.0 <= df[column][2] < 1.0
    assert df["quality"].to_list() == [0.11, 0.22, 0.33]
    assert any("Merged the torus coordinates of 2 models (qlvm1/qlvm2, qlvm_x1/qlvm_x2)" in message
               for message in messages)
    # No package baseline above these cells: both are inferred, and the log says why.
    assert sum("inference (no SESSION_H5_BASELINE.tsv in or above the cell" in message for message in messages) == 2
    assert any("qlvm_x1/qlvm_x2: 2 of 3 USVs placed" in message for message in messages)


def _to_v3_layout(package_root):
    """Rearrange a fake package from the v2 / v2.1 layout (every file at the cell's or
    package's top level, clusters in ``cluster/<level>/``) into the v3 layout of
    2026-09-28: ``config/`` for the contract and bins, ``inference/`` for the per-call
    tables, ``inference/clusters_<level>/`` for the clusters, and the package's
    ``SESSION_H5_BASELINE.tsv`` in ``corpus/``."""
    for cell in sorted(package_root.glob("phase_*/*")):
        (cell / "config").mkdir()
        (cell / "inference").mkdir()
        for name, subdirectory in (("training_contract.json", "config"), ("condition_bins.npz", "config"),
                                   ("posterior_cache.npz", "inference"), ("recon_mse_breakdown.npz", "inference")):
            if (cell / name).is_file():
                shutil.move(cell / name, cell / subdirectory / name)
        for level in ("fine", "coarse"):
            shutil.move(cell / "cluster" / level, cell / "inference" / f"clusters_{level}")
        (cell / "cluster").rmdir()
    if (package_root / "SESSION_H5_BASELINE.tsv").is_file():
        (package_root / "corpus").mkdir()
        shutil.move(package_root / "SESSION_H5_BASELINE.tsv", package_root / "corpus" / "SESSION_H5_BASELINE.tsv")


def test_load_model_cell_reads_the_v3_layout(tmp_path):
    """A v3 cell (contract and bins in config/, clusters in inference/clusters_<level>/)
    loads to the same model as the same cell in the v2 layout: the same contract,
    decoder parameters and phase 11 bins; neither layout's label grids are loaded."""
    rng = np.random.default_rng(41)
    condition = {"name": "bandwidth", "source": "usv_summary freq_bandwidth_hz", "decode": "exact"}
    fine = rng.integers(1, 5, size=(8, 8)).astype(np.int16)
    coarse = rng.integers(1, 3, size=(8, 8)).astype(np.int16)
    cell = _make_model_cell(tmp_path, rng, masking_type="none", floor=0.2, fine_grid=fine, coarse_grid=coarse,
                            condition=condition, bins=_phase11_bins("bandwidth", 0.05, 0.95, 0.01))
    before = ql.load_model_cell(str(cell))
    _to_v3_layout(tmp_path / "pkg")
    assert not (cell / "training_contract.json").exists() and (cell / "config" / "training_contract.json").is_file()

    after = ql.load_model_cell(str(cell))

    assert after["contract"] == before["contract"]
    # the embedding lattice is the torch float32 one the package corpus was embedded on
    np.testing.assert_array_equal(
        np.asarray(after["lattice"]),
        np.asarray(ql.gen_fib_basis_float32(after["contract"]["embedding_fib_m"])),
    )
    assert "fine_grid" not in after and "coarse_grid" not in after
    for key in before["condition_bins"]:
        np.testing.assert_array_equal(after["condition_bins"][key], before["condition_bins"][key])
    for key in before["params"]:
        for left, right in zip(np.atleast_1d(before["params"][key]), np.atleast_1d(after["params"][key]), strict=True):
            np.testing.assert_array_equal(np.asarray(left), np.asarray(right))


def test_cell_file_names_what_it_looked_for(tmp_path):
    """A file in neither layout raises FileNotFoundError naming every folder searched."""
    (tmp_path / "cell").mkdir()
    with pytest.raises(FileNotFoundError, match="no posterior_cache.npz in"):
        ql.cell_file(tmp_path / "cell", "posterior_cache.npz")
    with pytest.raises(FileNotFoundError, match="no fine cluster folder"):
        cell_cluster_directory(tmp_path / "cell", "fine")


@pytest.mark.parametrize("layout", ["v2", "v3"])
def test_model_cells_take_package_values_when_the_session_is_verified(tmp_path, mocker, layout):
    """A corpus session whose H5 is unchanged (baseline SHA-256, row count, durations
    and mask counts) takes each cell's own corpus coordinates, joined on spec_id; the
    decoder is never run and the H5 is hashed once for all cells. Both package layouts
    are read: v2 / v2.1 (flat) and v3 (config/, inference/, corpus/)."""
    rng = np.random.default_rng(31)
    root, session_id, cfg = _model_cells_session(tmp_path, rng)
    expected = _write_fake_package(tmp_path, root, session_id, cfg, rng)
    if layout == "v3":
        _to_v3_layout(tmp_path / "pkg")
    hashes = mocker.patch("usv_playpen.processing.qlvm_latents.file_sha256", side_effect=file_sha256)

    embedded, messages = _run_model_cells(root, cfg, mocker)

    assert embedded == []
    assert hashes.call_count == 1
    assert sum("package values (sha256 + 2 rows verified)" in message for message in messages) == 2
    df = pls.read_csv(root / "audio" / f"{session_id}_usv_summary.csv")
    for prefix, coords in expected.items():
        assert df[f"{prefix}1"][1] is None
        np.testing.assert_array_equal(df[f"{prefix}1"].to_numpy()[[0, 2]], coords[:, 0])
        np.testing.assert_array_equal(df[f"{prefix}2"].to_numpy()[[0, 2]], coords[:, 1])
    # The package route writes coordinates only, as the inference route does.
    assert not [column for column in df.columns if column.endswith(("_category", "_supercategory"))]


@pytest.mark.parametrize(
    ("failure", "reason"),
    [
        ("not_in_baseline", "inference (session not in the package corpus)"),
        ("sha256", "inference (spectrogram H5 changed since the package: sha256 mismatch)"),
        ("durations", "inference (the package's durations or mask counts disagree with the spectrogram H5 on 1 of 2 rows)"),
        ("mask_counts", "inference (the package's durations or mask counts disagree with the spectrogram H5 on 1 of 2 rows)"),
        ("summary_rows", "inference (usv_summary.csv has 4 rows, the spectrogram H5 3)"),
    ],
)
def test_model_cells_fall_back_to_inference_when_the_gate_fails(tmp_path, mocker, failure, reason):
    """Package rows name calls only by H5 position, so any doubt that the session's
    rows are the package's -- not in its corpus, H5 rehashed, a call's duration or
    mask count changed, or a summary out of step with the H5 -- embeds the session."""
    rng = np.random.default_rng(32)
    root, session_id, cfg = _model_cells_session(tmp_path, rng, prefixes=("qlvm_duration",))
    package = {
        "not_in_baseline": {"baseline_session": "20990101_000000"},
        "sha256": {"sha256": "0" * 64},
        "durations": {"durations": (64.0, 65.0)},
        "mask_counts": {"mask_counts": (1.0, 3.0)},
        "summary_rows": {},
    }[failure]
    _write_fake_package(tmp_path, root, session_id, cfg, rng, **package)
    if failure == "summary_rows":
        summary_path = root / "audio" / f"{session_id}_usv_summary.csv"
        summary = pls.read_csv(summary_path, schema_overrides={"usv_id": pls.String})
        pls.concat([summary, pls.DataFrame({"usv_id": ["0003"], "start": [0.7], "stop": [0.75], "usv": [True], "squeak": [False]})]).write_csv(summary_path)

    embedded, messages = _run_model_cells(root, cfg, mocker)

    assert embedded == [2]
    assert any(reason in message for message in messages), messages
    df = pls.read_csv(root / "audio" / f"{session_id}_usv_summary.csv")
    assert df["qlvm_duration1"].to_list()[:3] == [0.5, None, 0.5]


def test_model_cells_prefer_package_values_false_always_infers(tmp_path, mocker):
    """prefer_package_values false embeds even a verified corpus session, without
    hashing its H5."""
    rng = np.random.default_rng(33)
    root, session_id, cfg = _model_cells_session(tmp_path, rng, prefixes=("qlvm",))
    _write_fake_package(tmp_path, root, session_id, cfg, rng)
    cfg["prefer_package_values"] = False
    hashes = mocker.patch("usv_playpen.processing.qlvm_latents.file_sha256", side_effect=file_sha256)

    embedded, messages = _run_model_cells(root, cfg, mocker)

    assert embedded == [2]
    assert hashes.call_count == 0
    assert any("qlvm (pkg/phase_test/cell_qlvm): inference (prefer_package_values is false)" in message
               for message in messages)


def test_infer_and_merge_refuses_empty_model_cells(tmp_path, mocker):
    """Model package cells are the only models infer_and_merge embeds with: an empty
    model_cells (no spectrograms_root to derive the production cells from) stops the
    run with a message naming the setting, before any cell is loaded or the summary
    is touched."""
    rng = np.random.default_rng(34)
    root, session_id, cfg = _make_inference_session(tmp_path, rng)
    summary_path = root / "audio" / f"{session_id}_usv_summary.csv"
    before = summary_path.read_bytes()
    loads = mocker.patch("usv_playpen.processing.qlvm_latents.load_model_cell", side_effect=ql.load_model_cell)

    mocker.patch("usv_playpen.processing.qlvm_latents.smart_wait")
    with pytest.raises(ValueError, match="infer_qlvm_latents: model_cells is empty"):
        ql.QLVMLatentInference(
            root_directory=str(root),
            input_parameter_dict={"infer_qlvm_latents": cfg},
            message_output=lambda *_a, **_kw: None,
        ).infer_and_merge()
    assert loads.call_count == 0
    assert summary_path.read_bytes() == before


@pytest.mark.parametrize(
    ("pairs", "match"),
    [
        ([("", "/cell")], "is not a non-empty identifier"),
        ([("1qlvm", "/cell")], "is not a non-empty identifier"),
        ([("qlvm-dur", "/cell")], "is not a non-empty identifier"),
        ([("qlvm_duration", "/a"), ("qlvm_duration", "/b")], r"prefixes listed more than once: \['qlvm_duration'\]"),
        ([("qlvm_duration", "")], "has no model cell directory"),
    ],
)
def test_validate_model_cells_refuses_bad_prefixes(pairs, match):
    """A prefix names two summary columns, so it must be an identifier listed once,
    with a cell to embed."""
    with pytest.raises(ValueError, match=match):
        ql.validate_model_cells(pairs)


def test_validate_model_cells_refuses_prefixes_that_overwrite_summary_columns(mocker):
    """<prefix>1/<prefix>2 may not be another summary column; qlvm1/qlvm2 of prefix
    'qlvm' are the torus coordinates and are allowed."""
    mocker.patch.object(ql, "USV_SUMMARY_COLUMN_ORDER", (*ql.USV_SUMMARY_COLUMN_ORDER, "peak1"))
    with pytest.raises(ValueError, match=r"prefix 'peak' would overwrite the summary column\(s\) \['peak1'\]"):
        ql.validate_model_cells([("peak", "/cell")])
    assert ql.validate_model_cells([("qlvm", "/a"), ("qlvm_duration", "/b")]) == {"qlvm": "/a", "qlvm_duration": "/b"}


def test_model_cells_refuse_invalid_prefixes_before_writing(tmp_path, mocker):
    """An invalid model_cells prefix stops the run before the summary is touched."""
    rng = np.random.default_rng(35)
    root, session_id, cfg = _model_cells_session(tmp_path, rng, prefixes=("qlvm",))
    cfg["model_cells"] = {"2d": cfg["model_cells"]["qlvm"]}
    summary_path = root / "audio" / f"{session_id}_usv_summary.csv"
    before = summary_path.read_bytes()

    mocker.patch("usv_playpen.processing.qlvm_latents.smart_wait")
    with pytest.raises(ValueError, match="model_cells is invalid"):
        ql.QLVMLatentInference(
            root_directory=str(root),
            input_parameter_dict={"infer_qlvm_latents": cfg},
            message_output=lambda *_a, **_kw: None,
        ).infer_and_merge()
    assert summary_path.read_bytes() == before


def test_model_cell_stale_columns_cover_coordinates_and_categories():
    """The stale columns of a prefix are its coordinates, its category column (written by
    assign-qlvm-categories), the legacy coarse-level column older summaries may carry and
    the category confidence columns."""
    assert ql.model_cell_stale_columns("qlvm") == [
        "qlvm1", "qlvm2", "qlvm_category", "qlvm_supercategory", "qlvm_category_agreement", "qlvm_category_uncertain",
    ]
    assert ql.model_cell_stale_columns("qlvm_entropy") == [
        "qlvm_entropy1", "qlvm_entropy2", "qlvm_entropy_category", "qlvm_entropy_supercategory",
        "qlvm_entropy_category_agreement", "qlvm_entropy_category_uncertain",
    ]


def test_model_cell_coordinates_of_the_summary_and_production_maps_are_writable():
    """The coordinates of the summary map prefixes and the production prefixes
    (qlvm_entropy1 / qlvm_entropy2 included) are writable; the squeak map's are not, and
    no category column is (assign-qlvm-categories writes those)."""
    summary_maps = dict.fromkeys(ql.QLVM_SUMMARY_MAP_PREFIXES, "/cell")
    assert ql.validate_model_cells(summary_maps.items()) == summary_maps
    with pytest.raises(ValueError, match=r"prefix 'qlvm_squeak' would overwrite"):
        ql.validate_model_cells([("qlvm_squeak", "/cell")])
    production = {prefix: "/cell" for prefix in ql.QLVM_PRODUCTION_MODEL_CELLS}
    assert ql.validate_model_cells(production.items()) == production
    assert list(production) == ["qlvm", "qlvm_duration", "qlvm_entropy", "qlvm_bandwidth", "qlvm_loudness"]
    assert "qlvm_category" in ql.model_cell_reserved_columns()


def test_infer_and_merge_sam_inputs_equal_the_masked_training_inputs(tmp_path, mocker):
    """A masked cell is fed exactly what train-qlvm fed it: the call and its SAM mask
    union resized separately (here time-stretched), the resized mask binarized at 0.5
    and multiplied in (build-qlvm-training-set with apply_mask), then each spectrogram
    min-maxed and the binarized mask multiplied in again (train_qlvm.prepare_split).
    The mask edge (row 50, column 40 of a 64-bin call) does not fall on the resized
    grid, so masking the native spectrogram before the resize would differ there."""
    rng = np.random.default_rng(41)
    grid = np.ones((8, 8), dtype=np.int16)
    root, session_id, cfg = _make_inference_session(tmp_path, rng, with_masks=False)
    region = np.zeros((128, 128), dtype=bool)
    region[10:50, 5:40] = True
    _write_session_masks(root, session_id, {0: region, 2: region})
    cell = _make_model_cell(tmp_path, rng, masking_type="sam", floor=None, fine_grid=grid, coarse_grid=grid,
                            contract_overrides={"time_stretch": True})
    cfg.update(masking_type="sam", time_stretch=True, model_cells={"qlvm_masked": str(cell)})
    captured = {}

    def _fake_embed(lattice, data, params, *_block_sizes):
        captured["data"] = np.asarray(data)
        return np.full((data.shape[0], 2), 0.5, dtype=np.float64)

    mocker.patch("usv_playpen.processing.qlvm_latents.smart_wait")
    mocker.patch("usv_playpen.processing.qlvm_latents.embed_data", side_effect=_fake_embed)
    ql.QLVMLatentInference(
        root_directory=str(root), input_parameter_dict={"infer_qlvm_latents": cfg}, message_output=lambda *_a, **_kw: None,
    ).infer_and_merge()

    rows = np.array([0, 2])
    with h5py.File(root / "audio" / "spectrograms" / f"{session_id}_spectrograms.h5", "r") as h5_file:
        native = h5_file[f"spectrogram/{session_id}/spectrograms"][rows]
        durations = h5_file[f"spectrogram/{session_id}/durations"][rows]
        native_masks, _counts = build_session_masks(h5_file, session_id, rows, 128, 128)
    stored_masks = (stretch_specs(native_masks, durations, (128, 128), True) >= 0.5).astype(np.float32)
    stored = (stretch_specs(native, durations, (128, 128), True) * stored_masks).astype(np.float32)
    expected = prepare_split(stored, stored_masks)
    np.testing.assert_array_equal(captured["data"][:, 0], expected)
    pre_masked = ql.normalize_model_inputs(
        stretch_specs(native * native_masks, durations, (128, 128), True), json.loads((cell / "training_contract.json").read_text())
    )
    assert not np.array_equal(pre_masked, expected), "the test must tell the two orders apart"


def test_model_cells_embed_an_unclustered_cell(tmp_path, mocker):
    """A cell train-qlvm wrote has no cluster folders: it loads and embeds like any other
    (coordinates only, no category or supercategory column)."""
    rng = np.random.default_rng(42)
    grid = np.ones((8, 8), dtype=np.int16)
    root, session_id, cfg = _make_inference_session(tmp_path, rng)
    cell = _make_model_cell(tmp_path, rng, masking_type="none", floor=None, fine_grid=grid, coarse_grid=grid)
    shutil.rmtree(cell / "cluster")
    mocker.patch("usv_playpen.processing.qlvm_latents.smart_wait")
    mocker.patch("usv_playpen.processing.qlvm_latents.embed_data",
                 side_effect=lambda lattice, data, *_rest: np.full((data.shape[0], 2), 0.25, dtype=np.float64))
    summary_path = root / "audio" / f"{session_id}_usv_summary.csv"

    cfg["model_cells"] = {"qlvm_new": str(cell)}
    ql.QLVMLatentInference(
        root_directory=str(root), input_parameter_dict={"infer_qlvm_latents": cfg}, message_output=lambda *_a, **_kw: None,
    ).infer_and_merge()
    df = pls.read_csv(summary_path)
    assert df.columns == ["usv_id", "start", "stop", "usv", "squeak", "qlvm_new1", "qlvm_new2"]
    assert df["qlvm_new1"].to_list() == [0.25, None, 0.25]


def _old_layout_session(tmp_path, name="20250101_120000"):
    """A session folder whose summary has the pre-canonical layout: obsolete QLVM
    columns, the category confidence columns and columns out of canonical order,
    with values that a type-inferring rewrite would re-format (trailing zeros,
    zero-padded ids, an empty field)."""
    root = tmp_path / name
    (root / "audio").mkdir(parents=True)
    summary_path = root / "audio" / f"{name}_usv_summary.csv"
    summary_path.write_text(
        "usv_id,start,stop,duration,peak_amp_ch,emitter,noise,qlvm1,qlvm2,qlvm_category,qlvm_supercategory,"
        "qlvm_mf1,qlvm_mf2,qlvm_category_agreement,qlvm_category_uncertain,custom\n"
        "000000,0.10,0.2000,0.1,3.0,m1,false,0.25,0.50,2,1,0.1,0.2,0.75,false,a\n"
        "000001,1.00,1.1000,0.1,4.0,,true,,,,,,,,,b\n"
    )
    return root, summary_path


def test_tidy_session_usv_summary_dry_run_reports_without_writing(tmp_path):
    """A dry run reports the columns it would drop and the new order, and leaves the file
    byte-for-byte untouched."""
    root, summary_path = _old_layout_session(tmp_path)
    before = summary_path.read_bytes()
    messages = []
    report = ql.tidy_session_usv_summary(str(root), dry_run=True, backup_directory=None, message_output=messages.append)
    assert summary_path.read_bytes() == before
    assert report["written"] is False
    assert report["changed"] is True
    assert report["dropped"] == ["qlvm_supercategory", "qlvm_mf1", "qlvm_mf2", "qlvm_category_agreement", "qlvm_category_uncertain"]
    assert report["columns_after"] == [
        "usv_id", "start", "stop", "duration", "noise", "emitter", "peak_amp_ch", "qlvm1", "qlvm2", "qlvm_category", "custom",
    ]
    assert any("would drop" in message for message in messages)


def test_tidy_session_usv_summary_rewrites_with_a_backup_and_keeps_values_verbatim(tmp_path):
    """The rewrite drops the obsolete columns, reorders the rest, keeps every kept value as
    the text it was (no re-formatting), backs the original up, refuses to overwrite that
    backup on a second run, and leaves an already-canonical summary untouched."""
    root, summary_path = _old_layout_session(tmp_path)
    original = summary_path.read_text()
    backups = tmp_path / "backups"
    report = ql.tidy_session_usv_summary(str(root), dry_run=False, backup_directory=str(backups), message_output=lambda *_a: None)
    assert report["written"] is True
    assert (backups / root.name / summary_path.name).read_text() == original
    assert summary_path.read_text() == (
        "usv_id,start,stop,duration,noise,emitter,peak_amp_ch,qlvm1,qlvm2,qlvm_category,custom\n"
        "000000,0.10,0.2000,0.1,false,m1,3.0,0.25,0.50,2,a\n"
        "000001,1.00,1.1000,0.1,true,,4.0,,,,b\n"
    )
    tidied = summary_path.read_bytes()
    # A canonical summary is left alone (no backup is attempted for it).
    report = ql.tidy_session_usv_summary(str(root), dry_run=False, backup_directory=str(backups), message_output=lambda *_a: None)
    assert report["written"] is False
    assert report["backup_path"] is None
    # A summary that needs a rewrite while a backup already sits there is refused, untouched.
    summary_path.write_text(original)
    with pytest.raises(FileExistsError, match="refusing to overwrite a backup"):
        ql.tidy_session_usv_summary(str(root), dry_run=False, backup_directory=str(backups), message_output=lambda *_a: None)
    assert summary_path.read_text() == original
    summary_path.write_bytes(tidied)
    report = ql.tidy_session_usv_summary(str(root), dry_run=False, backup_directory=None, message_output=lambda *_a: None)
    assert report["changed"] is False
    assert report["written"] is False
    assert summary_path.read_bytes() == tidied


def test_tidy_usv_summary_columns_cli_runs_every_session(tmp_path):
    """The command takes sessions from --root-directory and --sessions-file, honours
    --dry-run, rewrites otherwise, and exits with an error listing a session that failed
    while still processing the others."""
    first, first_path = _old_layout_session(tmp_path, "20250101_120000")
    second, second_path = _old_layout_session(tmp_path, "20250102_120000")
    broken = tmp_path / "20250103_120000"
    (broken / "audio").mkdir(parents=True)
    sessions_file = tmp_path / "sessions.txt"
    sessions_file.write_text(f"# sessions\n{second}\n\n")
    runner = CliRunner()

    before = (first_path.read_bytes(), second_path.read_bytes())
    result = runner.invoke(ql.tidy_usv_summary_columns_cli, ["--root-directory", str(first), "--sessions-file", str(sessions_file), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "2 session(s): 2 would change" in result.output
    assert (first_path.read_bytes(), second_path.read_bytes()) == before

    result = runner.invoke(ql.tidy_usv_summary_columns_cli, [
        "--root-directory", str(first), "--root-directory", str(broken), "--sessions-file", str(sessions_file),
    ])
    assert result.exit_code != 0
    assert str(broken) in result.output
    for path in (first_path, second_path):
        assert pls.read_csv(path).columns[-1] == "custom"
        assert "qlvm_supercategory" not in pls.read_csv(path).columns

    result = runner.invoke(ql.tidy_usv_summary_columns_cli, [])
    assert result.exit_code != 0
    assert "needs at least one session" in result.output


def _set_vocal_flags(root, session_id, flags):
    """Overwrite the session summary's ``usv`` / ``squeak`` booleans with ``flags``, one
    ``(usv, squeak)`` pair (or ``(None, None)``) per row."""
    summary_path = root / "audio" / f"{session_id}_usv_summary.csv"
    summary = pls.read_csv(summary_path, schema_overrides={"usv_id": pls.String})
    summary.with_columns(
        usv=pls.Series([pair[0] for pair in flags], dtype=pls.Boolean),
        squeak=pls.Series([pair[1] for pair in flags], dtype=pls.Boolean),
    ).write_csv(summary_path)


@pytest.mark.parametrize("route", ["inference", "package"])
def test_model_cells_skip_pure_squeaks_on_both_routes(tmp_path, mocker, route):
    """A pure squeak (squeak true, usv false) is not a USV the maps can place: on the
    inference route the cell never embeds it, on the package route the package's
    coordinates are dropped for it, and either way its coordinates are null. A segment holding both (usv and squeak true) is placed as before: only pure
    squeaks are nulled, never a row whose usv is true."""
    rng = np.random.default_rng(35)
    root, session_id, cfg = _model_cells_session(tmp_path, rng, prefixes=("qlvm",))
    expected = _write_fake_package(tmp_path, root, session_id, cfg, rng)
    cfg["prefer_package_values"] = route == "package"
    _set_vocal_flags(root, session_id, [(False, True), (None, None), (True, True)])

    embedded, messages = _run_model_cells(root, cfg, mocker)

    assert embedded == ([1] if route == "inference" else [])
    assert any("1 pure squeak(s)" in message for message in messages)
    df = pls.read_csv(root / "audio" / f"{session_id}_usv_summary.csv")
    for column in ("qlvm1", "qlvm2"):
        assert df[column][0] is None
        assert df[column][1] is None
        assert df[column][2] is not None
    if route == "package":
        assert df["qlvm1"][2] == expected["qlvm"][1, 0]


def test_infer_and_merge_refuses_a_summary_without_vocal_flags(tmp_path, mocker):
    """Without the usv / squeak booleans the pure squeaks cannot be kept off the USV maps, so the run
    stops with a KeyError naming detect-usv-squeaks and leaves the summary untouched."""
    rng = np.random.default_rng(36)
    root, session_id, cfg = _model_cells_session(tmp_path, rng, prefixes=("qlvm",))
    summary_path = root / "audio" / f"{session_id}_usv_summary.csv"
    pls.read_csv(summary_path).drop("usv").write_csv(summary_path)
    before = summary_path.read_bytes()

    with pytest.raises(KeyError, match="detect-usv-squeaks"):
        _run_model_cells(root, cfg, mocker)
    assert summary_path.read_bytes() == before
