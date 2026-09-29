"""
@author: bartulem
Tests for processing/qlvm_latents — the QLVM inference driver.

Covers the helper pieces (weight loading + ``decoder.`` prefix stripping,
lattice rebuild, fine/coarse reference label lookup) and an end-to-end run that
synthesizes a decoder-weights ``.npz``, FINE + COARSE reference ``arrays.npz``
(periodic ws grids), a session spectrogram H5 and a ``usv_summary.csv``, then
checks the ``qlvm_*`` columns are merged into the right rows.
"""

from __future__ import annotations

import json
import pathlib
import shutil
import pickle
import zipfile

import h5py
import jax.numpy as jnp
import numpy as np
import polars as pls
import pytest

from usv_playpen.processing import qlvm_latents as ql
from usv_playpen.processing.build_qlvm_training_set import file_sha256


def test_load_decoder_params_strips_prefix(tmp_path):
    """Weights are loaded and a leading ``decoder.`` prefix is stripped."""
    p = tmp_path / "w.npz"
    np.savez(p, **{"decoder.0.weight": np.ones((4, 4)), "decoder.0.bias": np.zeros(4)})
    params = ql.load_decoder_params(str(p))
    assert set(params) == {"0.weight", "0.bias"}


def test_build_lattice_korobov_and_roberts():
    """build_lattice dispatches on lattice_type with the right point count."""
    kor = ql.build_lattice({"lattice_type": "korobov", "latent_dim": 2, "n_points": 21, "korobov_a": 3})
    assert kor.shape == (21, 2)
    rob = ql.build_lattice({"lattice_type": "roberts", "latent_dim": 2, "n_points": 30})
    assert rob.shape == (30, 2)


def test_labels_for_coords_lookup_convention():
    """Coordinate (x, y) maps to grid[int(y*res), int(x*res)] in each grid (per its
    own resolution): category from the fine grid, supercategory from the coarse grid."""
    fine = np.arange(16).reshape(4, 4).astype(np.int16)      # res = 4 -> qlvm_category
    coarse = np.arange(4).reshape(2, 2).astype(np.int16)     # res = 2 -> qlvm_supercategory
    coords = np.array([[0.0, 0.0], [0.9, 0.1], [0.1, 0.9]])
    cat, supercat = ql.labels_for_coords(coords, fine, coarse)
    # fine (res=4): (px,py) = (0,0),(3,0),(0,3)
    assert cat.tolist() == [fine[0, 0], fine[0, 3], fine[3, 0]]
    # coarse (res=2): (px,py) = (0,0),(1,0),(0,1)
    assert supercat.tolist() == [coarse[0, 0], coarse[0, 1], coarse[1, 0]]


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
    # labels_for_coords applies the same rule to its two grids.
    fine, coarse = ql.labels_for_coords(coords, grid, grid.T.copy())
    np.testing.assert_array_equal(fine, labels)
    np.testing.assert_array_equal(coarse, grid.T[expected_py, expected_px])


def _decoder_weights_npz(path, rng, latent_dim=2):
    np.savez(
        path,
        **{
            "decoder.0.weight": rng.standard_normal((2048, 2 * latent_dim)) * 0.05,
            "decoder.0.bias": rng.standard_normal(2048) * 0.05,
            "decoder.1.weight": rng.standard_normal((64 * 8 * 8, 2048)) * 0.01,
            "decoder.1.bias": rng.standard_normal(64 * 8 * 8) * 0.05,
            "decoder.3.weight": rng.standard_normal((64, 32, 3, 3)) * 0.05,
            "decoder.3.bias": rng.standard_normal(32) * 0.05,
            "decoder.5.weight": rng.standard_normal((32, 16, 3, 3)) * 0.05,
            "decoder.5.bias": rng.standard_normal(16) * 0.05,
            "decoder.7.weight": rng.standard_normal((16, 8, 3, 3)) * 0.05,
            "decoder.7.bias": rng.standard_normal(8) * 0.05,
            "decoder.9.weight": rng.standard_normal((8, 1, 3, 3)) * 0.05,
            "decoder.9.bias": rng.standard_normal(1) * 0.05,
        },
    )


def test_infer_and_merge_writes_qlvm_columns(tmp_path, mocker):
    """End-to-end: embed a session's specs and merge qlvm_* columns into the
    summary, joined on the per-USV index; missing USVs are null."""
    rng = np.random.default_rng(0)
    session_id = "20230119_155302"
    root = tmp_path / session_id
    (root / "audio" / "spectrograms").mkdir(parents=True)

    # weights + FINE/COARSE reference grids (the code reads ws_labels_periodic
    # from each); fine has more clusters than coarse.
    weights = tmp_path / "qmc_decoder_weights.npz"
    _decoder_weights_npz(weights, rng)
    fine_arrays = tmp_path / "arrays_fine.npz"
    coarse_arrays = tmp_path / "arrays_coarse.npz"
    res = 8
    np.savez(fine_arrays, ws_labels_periodic=(rng.integers(0, 12, size=(res, res))).astype(np.int16))
    np.savez(coarse_arrays, ws_labels_periodic=(rng.integers(0, 7, size=(res, res))).astype(np.int16))

    # consolidated layout, rows 1:1 with the 3-USV summary: rows 0 and 2 are
    # real (duration > 0), row 1 is an all-zero placeholder (duration 0).
    n_f = n_t = 128
    specs = np.zeros((3, n_f, n_t), dtype=np.float32)
    specs[0] = rng.random((n_f, n_t)).astype(np.float32)
    specs[2] = rng.random((n_f, n_t)).astype(np.float32)
    with h5py.File(root / "audio" / "spectrograms" / f"{session_id}_spectrograms.h5", "w") as f:
        f.create_dataset("frequency_bins", data=np.linspace(30000.0, 120000.0, n_f))
        session_group = f.create_group(f"spectrogram/{session_id}")
        session_group.create_dataset("spectrograms", data=specs)
        session_group.create_dataset("durations", data=np.array([128, 0, 128], dtype=np.int64))
    _write_session_masks(root, session_id, {0: np.ones((n_f, n_t), dtype=bool), 2: np.ones((n_f, n_t), dtype=bool)})

    pls.DataFrame({
        "usv_id": [f"{i:04d}" for i in range(3)],
        "start": [0.1, 0.3, 0.5],
        "stop": [0.15, 0.35, 0.55],
    }).write_csv(root / "audio" / f"{session_id}_usv_summary.csv")

    cfg = {
        "model_cell_directory": "",
        "model_cells": {},
        "model_cell_label_levels": {},
        "prefer_package_values": True,
        "weights_npz_path": str(weights),
        "reference_arrays_fine_npz_path": str(fine_arrays),
        "reference_arrays_coarse_npz_path": str(coarse_arrays),
        "lattice_type": "korobov",
        "latent_dim": 2,
        "n_points": 16,
        "korobov_a": 3,
        "fib_m": 16,
        "time_stretch": False,
        "masking_type": "sam",
        "target_shape": [128, 128],
        "length_threshold": None,
        "lattice_batch_size": 4096,
        "data_batch_size": 8192,
    }
    mocker.patch("usv_playpen.processing.qlvm_latents.smart_wait")
    ql.QLVMLatentInference(
        root_directory=str(root),
        input_parameter_dict={"infer_qlvm_latents": cfg},
        message_output=lambda *_a, **_kw: None,
    ).infer_and_merge()

    df = pls.read_csv(root / "audio" / f"{session_id}_usv_summary.csv")
    assert set(ql.QLVM_COLUMNS).issubset(df.columns)
    assert df.height == 3
    # rows 0 and 2 embedded; row 1 (no spec) is null.
    assert df["qlvm1"][0] is not None
    assert df["qlvm1"][1] is None
    assert df["qlvm1"][2] is not None
    # coordinates live on the torus [0, 1).
    assert 0.0 <= df["qlvm1"][0] < 1.0
    # every embedded row names the model its latents and labels came from.
    assert df["qlvm_model"][0] == str(weights)
    assert df["qlvm_model"][1] is None


def _write_session_masks(root, session_id, masks_by_row):
    """Write the session H5's ``mask/<session>`` group the way generate-usv-masks
    does: one ``segmentations`` row per mask instance, keyed by ``spectrogram_index``
    (``masks_by_row`` maps a spectrogram row to one boolean mask)."""
    rows = sorted(masks_by_row)
    with h5py.File(root / "audio" / "spectrograms" / f"{session_id}_spectrograms.h5", "a") as f:
        group = f.create_group(f"mask/{session_id}")
        group.create_dataset("segmentations", data=np.stack([masks_by_row[row] for row in rows]))
        group.create_dataset("spectrogram_index", data=np.array(rows, dtype=np.int64))


def _make_inference_session(tmp_path, rng, *, fine_grid, coarse_grid, with_masks=True):
    """Synthesize a session (weights + fine/coarse reference grids + spectrogram
    H5 + usv_summary) for an end-to-end QLVMLatentInference run and return
    (root, session_id, cfg). Rows 0 and 2 are real, row 1 is a placeholder; with
    ``with_masks``, rows 0 and 2 each get one all-ones SAM mask."""
    session_id = "20230119_155302"
    root = tmp_path / session_id
    (root / "audio" / "spectrograms").mkdir(parents=True)

    weights = tmp_path / "qmc_decoder_weights.npz"
    _decoder_weights_npz(weights, rng)
    fine_arrays = tmp_path / "arrays_fine.npz"
    coarse_arrays = tmp_path / "arrays_coarse.npz"
    np.savez(fine_arrays, ws_labels_periodic=fine_grid)
    np.savez(coarse_arrays, ws_labels_periodic=coarse_grid)

    n_f = n_t = 128
    specs = np.zeros((3, n_f, n_t), dtype=np.float32)
    specs[0] = rng.random((n_f, n_t)).astype(np.float32)
    specs[2] = rng.random((n_f, n_t)).astype(np.float32)
    with h5py.File(root / "audio" / "spectrograms" / f"{session_id}_spectrograms.h5", "w") as f:
        f.create_dataset("frequency_bins", data=np.linspace(30000.0, 120000.0, n_f))
        session_group = f.create_group(f"spectrogram/{session_id}")
        session_group.create_dataset("spectrograms", data=specs)
        session_group.create_dataset("durations", data=np.array([128, 0, 128], dtype=np.int64))
    if with_masks:
        _write_session_masks(root, session_id, {0: np.ones((n_f, n_t), dtype=bool), 2: np.ones((n_f, n_t), dtype=bool)})

    pls.DataFrame({
        "usv_id": [f"{i:04d}" for i in range(3)],
        "start": [0.1, 0.3, 0.5],
        "stop": [0.15, 0.35, 0.55],
    }).write_csv(root / "audio" / f"{session_id}_usv_summary.csv")

    cfg = {
        "model_cell_directory": "",
        "model_cells": {},
        "model_cell_label_levels": {},
        "prefer_package_values": True,
        "weights_npz_path": str(weights),
        "reference_arrays_fine_npz_path": str(fine_arrays),
        "reference_arrays_coarse_npz_path": str(coarse_arrays),
        "lattice_type": "korobov",
        "latent_dim": 2,
        "n_points": 16,
        "korobov_a": 3,
        "fib_m": 16,
        "time_stretch": False,
        "masking_type": "sam",
        "target_shape": [128, 128],
        "length_threshold": None,
        "lattice_batch_size": 4096,
        "data_batch_size": 8192,
    }
    return root, session_id, cfg


def test_infer_and_merge_category_vs_supercategory_semantics(tmp_path, mocker):
    """Regression guard for the fine/coarse mapping: qlvm_category must be read
    from the FINE reference grid and qlvm_supercategory from the COARSE one. Using
    grids with DISJOINT value ranges (fine 100..115, coarse 0..6) means a swapped
    file/assignment would land values in the wrong column and fail this test."""
    rng = np.random.default_rng(1)
    res = 8
    fine_grid = rng.integers(100, 116, size=(res, res)).astype(np.int16)   # 100..115
    coarse_grid = rng.integers(0, 7, size=(res, res)).astype(np.int16)     # 0..6
    root, session_id, cfg = _make_inference_session(tmp_path, rng, fine_grid=fine_grid, coarse_grid=coarse_grid)

    mocker.patch("usv_playpen.processing.qlvm_latents.smart_wait")
    ql.QLVMLatentInference(
        root_directory=str(root),
        input_parameter_dict={"infer_qlvm_latents": cfg},
        message_output=lambda *_a, **_kw: None,
    ).infer_and_merge()

    df = pls.read_csv(root / "audio" / f"{session_id}_usv_summary.csv")
    cats = df["qlvm_category"].drop_nulls().to_list()
    supercats = df["qlvm_supercategory"].drop_nulls().to_list()
    assert cats, "expected at least one embedded USV"
    # FINE labels land in qlvm_category (100..115), COARSE in qlvm_supercategory (0..6).
    assert all(100 <= c <= 115 for c in cats)
    assert all(0 <= s <= 6 for s in supercats)


def test_infer_and_merge_masking_type_applies_or_skips_sam_mask(tmp_path, mocker):
    """The decoder is trained on SAM-masked (background-zeroed) spectrograms, so
    ``masking_type='sam'`` must zero every pixel outside the call's SAM mask region
    before embedding, while ``masking_type='none'`` must embed the raw spectrogram.
    This pins the train/inference masking parity: embedding raw specs into a
    masked-trained decoder is out-of-distribution and yields unreliable latents.
    The spectrogram fed to ``embed_data`` is captured under both settings."""
    rng = np.random.default_rng(3)
    res = 8
    fine_grid = rng.integers(0, 12, size=(res, res)).astype(np.int16)
    coarse_grid = rng.integers(0, 7, size=(res, res)).astype(np.int16)
    root, session_id, cfg = _make_inference_session(
        tmp_path, rng, fine_grid=fine_grid, coarse_grid=coarse_grid, with_masks=False
    )

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
    res = 8
    fine_grid = rng.integers(0, 12, size=(res, res)).astype(np.int16)
    coarse_grid = rng.integers(0, 7, size=(res, res)).astype(np.int16)
    root, session_id, cfg = _make_inference_session(tmp_path, rng, fine_grid=fine_grid, coarse_grid=coarse_grid)

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
    duplicated or left stale)."""
    rng = np.random.default_rng(2)
    res = 8
    fine_grid = rng.integers(0, 12, size=(res, res)).astype(np.int16)
    coarse_grid = rng.integers(0, 7, size=(res, res)).astype(np.int16)
    root, session_id, cfg = _make_inference_session(tmp_path, rng, fine_grid=fine_grid, coarse_grid=coarse_grid)

    # Add an unrelated column the merge must leave intact.
    summary_path = root / "audio" / f"{session_id}_usv_summary.csv"
    df0 = pls.read_csv(summary_path).with_columns(pls.Series("quality", [0.11, 0.22, 0.33]))
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
    for column in ql.QLVM_COLUMNS:
        assert df.columns.count(column) == 1
    # the embedded rows still carry latents after the re-run.
    assert df["qlvm1"][0] is not None
    assert df["qlvm1"][1] is None


def _matching_contract(cfg, **overrides):
    """A training contract that agrees with ``cfg`` (as train-qlvm would write it for
    the synthetic session), with ``overrides`` applied."""
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
        "lattice_type": cfg["lattice_type"],
        "korobov_a": cfg["korobov_a"],
        "train_n_points": cfg["n_points"],
        "test_n_points": cfg["n_points"],
        "fib_m": cfg["fib_m"],
        "dataset_directory": "/synthetic",
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
            "length_threshold": None, "lattice_type": "korobov", "korobov_a": 3, "n_points": 16, "fib_m": 16}


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


def test_infer_and_merge_nulls_calls_outside_the_training_window(tmp_path, mocker):
    """A call at or above the training set's duration bound was never trained on:
    with a contract beside the weights it gets null qlvm_* columns, shorter calls embed."""
    rng = np.random.default_rng(7)
    res = 8
    fine_grid = rng.integers(0, 12, size=(res, res)).astype(np.int16)
    coarse_grid = rng.integers(0, 7, size=(res, res)).astype(np.int16)
    root, session_id, cfg = _make_inference_session(tmp_path, rng, fine_grid=fine_grid, coarse_grid=coarse_grid)
    _set_session_durations(root, session_id, [128, 0, 64])
    contract_path = tmp_path / "qmc_decoder_weights.json"
    contract_path.write_text(json.dumps(_matching_contract(cfg, length_threshold=100.0)))

    mocker.patch("usv_playpen.processing.qlvm_latents.smart_wait")
    ql.QLVMLatentInference(
        root_directory=str(root),
        input_parameter_dict={"infer_qlvm_latents": cfg},
        message_output=lambda *_a, **_kw: None,
    ).infer_and_merge()

    df = pls.read_csv(root / "audio" / f"{session_id}_usv_summary.csv")
    assert df["qlvm1"][0] is None      # 128 >= 100: outside the window
    assert df["qlvm1"][1] is None      # duration 0: no call
    assert df["qlvm1"][2] is not None  # 64 < 100: embedded


def test_infer_and_merge_without_contract_applies_settings_threshold(tmp_path, mocker):
    """Weights trained before contracts existed still run; the settings'
    length_threshold then sets the window."""
    rng = np.random.default_rng(8)
    res = 8
    fine_grid = rng.integers(0, 12, size=(res, res)).astype(np.int16)
    coarse_grid = rng.integers(0, 7, size=(res, res)).astype(np.int16)
    root, session_id, cfg = _make_inference_session(tmp_path, rng, fine_grid=fine_grid, coarse_grid=coarse_grid)
    _set_session_durations(root, session_id, [128, 0, 64])
    cfg["length_threshold"] = 100.0

    mocker.patch("usv_playpen.processing.qlvm_latents.smart_wait")
    messages = []
    ql.QLVMLatentInference(
        root_directory=str(root),
        input_parameter_dict={"infer_qlvm_latents": cfg},
        message_output=messages.append,
    ).infer_and_merge()

    df = pls.read_csv(root / "audio" / f"{session_id}_usv_summary.csv")
    assert df["qlvm1"][0] is None
    assert df["qlvm1"][2] is not None
    assert any("No training contract" in message for message in messages)


def test_infer_and_merge_refuses_settings_that_break_the_contract(tmp_path, mocker):
    """Embedding with preprocessing the decoder was not trained on must stop before
    anything is written."""
    rng = np.random.default_rng(9)
    res = 8
    fine_grid = rng.integers(0, 12, size=(res, res)).astype(np.int16)
    coarse_grid = rng.integers(0, 7, size=(res, res)).astype(np.int16)
    root, session_id, cfg = _make_inference_session(tmp_path, rng, fine_grid=fine_grid, coarse_grid=coarse_grid)
    (tmp_path / "qmc_decoder_weights.json").write_text(json.dumps(_matching_contract(cfg, masking_type="none")))
    summary_path = root / "audio" / f"{session_id}_usv_summary.csv"
    before = summary_path.read_bytes()

    mocker.patch("usv_playpen.processing.qlvm_latents.smart_wait")
    with pytest.raises(ValueError, match="masking_type"):
        ql.QLVMLatentInference(
            root_directory=str(root),
            input_parameter_dict={"infer_qlvm_latents": cfg},
            message_output=lambda *_a, **_kw: None,
        ).infer_and_merge()
    assert summary_path.read_bytes() == before


def test_infer_and_merge_failed_write_keeps_previous_summary(tmp_path, mocker):
    """usv_summary.csv carries every other per-USV column, so a write that fails
    part-way must leave the previous file whole and no temporary file behind."""
    rng = np.random.default_rng(10)
    res = 8
    fine_grid = rng.integers(0, 12, size=(res, res)).astype(np.int16)
    coarse_grid = rng.integers(0, 7, size=(res, res)).astype(np.int16)
    root, session_id, cfg = _make_inference_session(tmp_path, rng, fine_grid=fine_grid, coarse_grid=coarse_grid)
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
    re-min-maxed) spectrograms; train-qlvm decoders the stored values."""
    rng = np.random.default_rng(12)
    specs = rng.uniform(0.1, 3.0, size=(2, 16, 16)).astype(np.float32)
    assert np.array_equal(ql.normalize_model_inputs(specs, None), specs)
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
    cell_name="cell_test",
):
    """Synthesize a QLVM model package cell (checkpoint.tar, training_contract.json,
    cluster/{fine,coarse}/label_grid.npy) with a ReLU-head decoder; with a
    ``condition`` block, a conditional one (one extra decoder input) and its
    ``condition_bins.npz`` (``bins`` = (edges, bin_mean) for phase 10, or a dict of
    the phase 11 keys, see :func:`_phase11_bins`). ``require_mask`` True, as in every
    v2 and v3 cell, says its corpus kept only calls with a SAM mask. The cell is
    ``<tmp_path>/pkg/phase_test/<cell_name>``, so several cells share one package."""
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
    """With model_cell_directory set, the cell's ReLU checkpoint, contract, Fibonacci
    lattice and label grids are used: calls outside its duration window are null,
    inputs are min-maxed and floored, and labels come from label_grid.npy."""
    rng = np.random.default_rng(13)
    res = 8
    fine_grid = rng.integers(101, 117, size=(res, res)).astype(np.int16)
    coarse_grid = rng.integers(201, 208, size=(res, res)).astype(np.int16)
    root, session_id, cfg = _make_inference_session(
        tmp_path, rng,
        fine_grid=np.zeros((res, res), dtype=np.int16), coarse_grid=np.zeros((res, res), dtype=np.int16),
    )
    _set_session_durations(root, session_id, [128, 0, 64])  # row 0 is outside the cell's window (100)
    cell = _make_model_cell(tmp_path, rng, masking_type="none", floor=0.2, fine_grid=fine_grid, coarse_grid=coarse_grid)
    cfg["model_cell_directory"] = str(cell)
    cfg["masking_type"] = "none"
    cfg["weights_npz_path"] = str(tmp_path / "does_not_exist.npz")  # unused in package mode

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
    assert 101 <= df["qlvm_category"][2] <= 116
    assert 201 <= df["qlvm_supercategory"][2] <= 207
    assert df["qlvm_model"][2] == "pkg/phase_test/cell_test"


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
    root, session_id, cfg = _make_inference_session(tmp_path, rng, fine_grid=grid, coarse_grid=grid, with_masks=False)
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
    cfg["model_cell_directory"] = str(cell)
    cfg["masking_type"] = "none"

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
    root, session_id, cfg = _make_inference_session(tmp_path, rng, fine_grid=grid, coarse_grid=grid, with_masks=False)
    _set_session_durations(root, session_id, [64, 0, 64])
    low_rows, high_rows = np.zeros((128, 128), dtype=bool), np.zeros((128, 128), dtype=bool)
    low_rows[:32, :] = True
    high_rows[96:, :] = True
    _write_session_masks(root, session_id, {0: low_rows, 2: high_rows})
    if summary_columns:
        summary_path = root / "audio" / f"{session_id}_usv_summary.csv"
        pls.read_csv(summary_path).with_columns(**summary_columns).write_csv(summary_path)
    cell = _make_model_cell(tmp_path, rng, masking_type="none", floor=0.2, fine_grid=grid, coarse_grid=grid,
                            condition=condition, bins=bins)
    cfg["model_cell_directory"] = str(cell)
    cfg["masking_type"] = "none"
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


def test_infer_and_merge_phase11_loudness_measures_the_masked_calls(tmp_path, mocker):
    """A loudness cell measures each call's image-level dB over its SAM mask region
    from the audio, maps it through db_range and snaps it to the decode grid; a
    call with no measurable loudness gets null columns."""
    rng = np.random.default_rng(23)
    condition = {"name": "loudness", "db_range": [28.83, 95.85], "decode": "grid"}
    root, session_id, cfg, (low_rows, high_rows) = _phase11_session(
        tmp_path, rng, condition, _phase11_bins("loudness", 0.0, 1.0, 0.0025)
    )
    measured = {}

    def _loudness(**kwargs):
        measured.update(kwargs)
        return np.array([62.34, np.nan], dtype=np.float32)

    mocker.patch("usv_playpen.processing.qlvm_latents.session_image_level_db", side_effect=_loudness)
    spec_params = {"offset": 0.0}
    condition_values, messages = _run_capturing_c(root, cfg, mocker, {"generate_spectrograms": spec_params})

    np.testing.assert_array_equal(measured["starts"], [0.1, 0.5])
    np.testing.assert_array_equal(measured["regions"], np.stack([low_rows, high_rows]))
    assert measured["spec_params"] is spec_params
    own = np.clip((np.float64(np.float32(62.34)) - 28.83) / (95.85 - 28.83), 0.0, 1.0)
    grid = 0.0025 * np.arange(401)
    np.testing.assert_array_equal(condition_values, np.array([grid[np.argmin(np.abs(grid - own))]], dtype=np.float32))
    assert any("1 USVs without a loudness value" in message for message in messages)
    df = pls.read_csv(root / "audio" / f"{session_id}_usv_summary.csv")
    assert df["qlvm1"][0] is not None
    assert df["qlvm1"][2] is None


def test_export_model_cell_arrays_writes_the_reference_layout(tmp_path):
    """The visualizations read arrays_{fine,coarse}.npz: a package cell exported to
    that layout must carry its grid, its peaks in label order, its calls with their
    labels, and an aggregated-posterior heatmap holding all the posterior mass."""
    rng = np.random.default_rng(15)
    res, n_calls, fib_m = 8, 30, 8
    cell = tmp_path / "pkg" / "phase_x" / "cell_x"
    for level in ("fine", "coarse"):
        (cell / "cluster" / level).mkdir(parents=True)
    (cell / "training_contract.json").write_text(json.dumps({"embedding_fib_m": fib_m}))
    angles = rng.uniform(0.0, 2 * np.pi, size=(n_calls, 2))
    torus_weighted = np.concatenate([np.cos(angles), np.sin(angles)], axis=1).astype(np.float32)
    aggregated = rng.random(21) * 3.0                                 # fib(8) = 21 lattice points
    np.savez(cell / "posterior_cache.npz", torus_weighted=torus_weighted, aggregated=aggregated)
    coords = (angles / (2 * np.pi)).astype(np.float32)
    grids = {"fine": rng.integers(1, 5, size=(res, res)).astype(np.int16), "coarse": rng.integers(1, 3, size=(res, res)).astype(np.int16)}
    for level, grid in grids.items():
        np.save(cell / "cluster" / level / "label_grid.npy", grid)
        pixel_y = np.clip((coords[:, 1] * res).astype(int), 0, res - 1)
        pixel_x = np.clip((coords[:, 0] * res).astype(int), 0, res - 1)
        pls.DataFrame({"spec_id": [f"s_{i}" for i in range(n_calls)], "label": grid[pixel_y, pixel_x].astype(np.int64)}).write_csv(
            cell / "cluster" / level / "cluster_labels.csv"
        )
        k = int(grid.max())
        pls.DataFrame({"label": list(range(k, 0, -1)), "peak_x": [0.1 * label for label in range(k, 0, -1)],
                       "peak_y": [0.05 * label for label in range(k, 0, -1)]}).write_csv(cell / "cluster" / level / "clusters.csv")

    written = ql.export_model_cell_arrays(str(cell), str(tmp_path / "out"), message_output=lambda *_a: None)

    assert [path.name for path in written] == ["arrays_fine.npz", "arrays_coarse.npz"]
    for level, grid in grids.items():
        with np.load(tmp_path / "out" / f"arrays_{level}.npz", allow_pickle=False) as arrays:
            np.testing.assert_array_equal(arrays["ws_labels_periodic"], grid)
            np.testing.assert_array_equal(arrays["ws_labels"], grid)
            k = int(grid.max())
            np.testing.assert_allclose(arrays["centers"], [[0.1 * label, 0.05 * label] for label in range(1, k + 1)], rtol=1e-6)
            np.testing.assert_allclose(arrays["latent_coords"], coords, atol=1e-5)
            np.testing.assert_array_equal(ql.labels_for_coords(arrays["latent_coords"], grid, grid)[0], arrays["sample_ws_periodic"])
            assert arrays["heatmap"].shape == (res, res)
            assert arrays["heatmap"].sum() == pytest.approx(aggregated.sum(), rel=1e-5)
            assert str(arrays["model_id"]) == "pkg/phase_x/cell_x"


def test_infer_and_merge_model_cell_refuses_wrong_masking(tmp_path, mocker):
    """A masked package decoder must not be fed unmasked spectrograms: the settings'
    masking_type is checked against the cell's contract before anything runs."""
    rng = np.random.default_rng(14)
    res = 8
    grid = np.ones((res, res), dtype=np.int16)
    root, _session_id, cfg = _make_inference_session(tmp_path, rng, fine_grid=grid, coarse_grid=grid)
    cell = _make_model_cell(tmp_path, rng, masking_type="sam", floor=None, fine_grid=grid, coarse_grid=grid)
    cfg["model_cell_directory"] = str(cell)
    cfg["masking_type"] = "none"

    mocker.patch("usv_playpen.processing.qlvm_latents.smart_wait")
    with pytest.raises(ValueError, match="masking_type: settings 'none', trained 'sam'"):
        ql.QLVMLatentInference(
            root_directory=str(root),
            input_parameter_dict={"infer_qlvm_latents": cfg},
            message_output=lambda *_a, **_kw: None,
        ).infer_and_merge()


def _use_decoder(tmp_path, rng, cfg, decoder, grid):
    """Point ``cfg`` at one kind of decoder for the SAM-mask rule: ``"sam"`` (a
    train-qlvm contract with masking_type sam), ``"require_mask"`` (an unmasked
    train-qlvm contract whose set kept only calls with a mask), ``"mean_freq"`` (an
    unmasked package cell conditioned on mean frequency, require_mask false) or
    ``"unmasked"`` (an unmasked train-qlvm contract whose set kept every call)."""
    if decoder == "mean_freq":
        condition = {"name": "mean_freq", "spectrogram": "masked", "epsilon": 1e-8}
        bins = (np.array([0.0, 0.5, 1.0]), np.array([0.2, 0.8], dtype=np.float32))
        cell = _make_model_cell(tmp_path, rng, masking_type="none", floor=0.2, fine_grid=grid, coarse_grid=grid,
                                condition=condition, bins=bins, require_mask=False)
        cfg["model_cell_directory"] = str(cell)
        cfg["masking_type"] = "none"
        return
    cfg["masking_type"] = "sam" if decoder == "sam" else "none"
    contract = _matching_contract(cfg, require_mask=decoder == "require_mask")
    (tmp_path / "qmc_decoder_weights.json").write_text(json.dumps(contract))


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
    A sam decoder whose set kept them under that all-ones mask (the shipped model,
    train-qlvm on a main-built set) still embeds them, as does an unmasked decoder."""
    rng = np.random.default_rng(17)
    grid = np.ones((8, 8), dtype=np.int16)
    root, session_id, cfg = _make_inference_session(tmp_path, rng, fine_grid=grid, coarse_grid=grid, with_masks=False)
    _set_session_durations(root, session_id, [64, 0, 64])
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
    root, session_id, cfg = _make_inference_session(tmp_path, rng, fine_grid=grid, coarse_grid=grid, with_masks=False)
    _set_session_durations(root, session_id, [64, 0, 64])
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
    root, session_id, cfg = _make_inference_session(tmp_path, rng, fine_grid=grid, coarse_grid=grid, with_masks=False)
    _set_session_durations(root, session_id, [64, 0, 64])
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
    cell gets its own label grids (:func:`_model_cell_grids`), so a label read from the
    wrong cell or level shows."""
    grid = np.ones((8, 8), dtype=np.int16)
    root, session_id, cfg = _make_inference_session(tmp_path, rng, fine_grid=grid, coarse_grid=grid)
    _set_session_durations(root, session_id, [64, 0, 64])
    cfg["masking_type"] = "none"
    cfg["model_cells"] = {}
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


def _assert_labels_follow_coordinates(df, prefix, index, levels):
    """Every label column of ``prefix`` in ``levels`` equals label_grid_lookup of the
    prefix's written coordinates in the ``index``-th cell's grid of that level, as Int64,
    and is null exactly where the coordinates are."""
    grids = dict(zip(("fine", "coarse"), _model_cell_grids(index), strict=True))
    placed = df[f"{prefix}1"].is_not_null().to_numpy()
    coords = df.select(f"{prefix}1", f"{prefix}2").to_numpy()[placed]
    for level in levels:
        column = ql.model_cell_label_column(prefix, level)
        assert df[column].dtype == pls.Int64
        np.testing.assert_array_equal(df[column].is_not_null().to_numpy(), placed)
        np.testing.assert_array_equal(
            df[column].to_numpy()[placed], ql.label_grid_lookup(coords, grids[level]).astype(np.int64)
        )


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


def test_model_cells_write_prefixed_coordinates_and_labels(tmp_path, mocker):
    """With model_cells and the shipped (empty) model_cell_label_levels, every listed cell
    places the session and each prefix gets exactly its coordinates <prefix>1/<prefix>2
    plus its default labels -- qlvm_category (fine) and qlvm_supercategory (coarse) for
    'qlvm', qlvm_x_category (fine) only for 'qlvm_x' -- each the cell's grid of that level
    at the pixel of the written coordinates, null where the call was not placed. No model
    column is written; stale label, model and coordinate columns of the listed prefixes
    (including a label level this run does not write) are removed; other columns stay."""
    rng = np.random.default_rng(30)
    root, session_id, cfg = _model_cells_session(tmp_path, rng)
    summary_path = root / "audio" / f"{session_id}_usv_summary.csv"
    pls.read_csv(summary_path).with_columns(
        quality=pls.Series([0.11, 0.22, 0.33]),
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
        "usv_id", "start", "stop", "qlvm1", "qlvm2", "qlvm_category", "qlvm_supercategory",
        "quality", "qlvm_x1", "qlvm_x2", "qlvm_x_category",
    ]
    for column in ("qlvm1", "qlvm2", "qlvm_x1", "qlvm_x2"):
        assert df[column].dtype == pls.Float64
        assert df[column][1] is None
        assert 0.0 <= df[column][0] < 1.0
        assert 0.0 <= df[column][2] < 1.0
    _assert_labels_follow_coordinates(df, "qlvm", 0, ("fine", "coarse"))
    _assert_labels_follow_coordinates(df, "qlvm_x", 1, ("fine",))
    assert df["quality"].to_list() == [0.11, 0.22, 0.33]
    assert any("qlvm1/qlvm2/qlvm_category/qlvm_supercategory, qlvm_x1/qlvm_x2/qlvm_x_category" in message
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
    decoder parameters, label grids and phase 11 bins."""
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
    np.testing.assert_array_equal(after["fine_grid"], fine)
    np.testing.assert_array_equal(after["coarse_grid"], coarse)
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
        ql.cell_cluster_directory(tmp_path / "cell", "fine")


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
    # The package route labels by the same grid lookup, on the package's coordinates.
    _assert_labels_follow_coordinates(df, "qlvm", 0, ("fine", "coarse"))
    _assert_labels_follow_coordinates(df, "qlvm_x", 1, ("fine",))
    assert "qlvm_x_supercategory" not in df.columns


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
    root, session_id, cfg = _model_cells_session(tmp_path, rng, prefixes=("qlvm_dur",))
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
        pls.concat([summary, pls.DataFrame({"usv_id": ["0003"], "start": [0.7], "stop": [0.75]})]).write_csv(summary_path)

    embedded, messages = _run_model_cells(root, cfg, mocker)

    assert embedded == [2]
    assert any(reason in message for message in messages), messages
    df = pls.read_csv(root / "audio" / f"{session_id}_usv_summary.csv")
    assert df["qlvm_dur1"].to_list()[:3] == [0.5, None, 0.5]


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


def test_model_cells_and_model_cell_directory_are_exclusive(tmp_path, mocker):
    """model_cells writes <prefix>1/<prefix>2 of several cells, model_cell_directory the
    qlvm_* columns of one: setting both stops the run before anything is written."""
    rng = np.random.default_rng(34)
    root, session_id, cfg = _model_cells_session(tmp_path, rng, prefixes=("qlvm",))
    cfg["model_cell_directory"] = cfg["model_cells"]["qlvm"]
    summary_path = root / "audio" / f"{session_id}_usv_summary.csv"
    before = summary_path.read_bytes()

    mocker.patch("usv_playpen.processing.qlvm_latents.smart_wait")
    with pytest.raises(ValueError, match="model_cells and model_cell_directory are both set"):
        ql.QLVMLatentInference(
            root_directory=str(root),
            input_parameter_dict={"infer_qlvm_latents": cfg},
            message_output=lambda *_a, **_kw: None,
        ).infer_and_merge()
    assert summary_path.read_bytes() == before


@pytest.mark.parametrize(
    ("pairs", "match"),
    [
        ([("", "/cell")], "is not a non-empty identifier"),
        ([("1qlvm", "/cell")], "is not a non-empty identifier"),
        ([("qlvm-dur", "/cell")], "is not a non-empty identifier"),
        ([("qlvm_dur", "/a"), ("qlvm_dur", "/b")], r"prefixes listed more than once: \['qlvm_dur'\]"),
        ([("qlvm_dur", "")], "has no model cell directory"),
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
    assert ql.validate_model_cells([("qlvm", "/a"), ("qlvm_dur", "/b")]) == {"qlvm": "/a", "qlvm_dur": "/b"}


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


def test_model_cell_label_column_names_follow_the_rule():
    """Prefix 'qlvm' keeps qlvm_category (fine) / qlvm_supercategory (coarse); every other
    prefix P gets P_category / P_supercategory. The defaults are both levels for 'qlvm',
    the fine level for every other prefix."""
    assert ql.model_cell_label_column("qlvm", "fine") == "qlvm_category"
    assert ql.model_cell_label_column("qlvm", "coarse") == "qlvm_supercategory"
    assert ql.model_cell_label_column("qlvm_dur", "fine") == "qlvm_dur_category"
    assert ql.model_cell_label_column("qlvm_loud", "coarse") == "qlvm_loud_supercategory"
    assert ql.model_cell_label_columns({"qlvm": "/a", "qlvm_mf": "/b"}, {}) == {
        "qlvm": {"fine": "qlvm_category", "coarse": "qlvm_supercategory"},
        "qlvm_mf": {"fine": "qlvm_mf_category"},
    }
    # Levels come out in the fine, coarse order whatever order the setting lists them in.
    assert list(ql.model_cell_label_columns({"qlvm_bw": "/a"}, {"qlvm_bw": ["coarse", "fine"]})["qlvm_bw"]) == [
        "fine", "coarse",
    ]


def test_model_cells_write_the_configured_label_levels(tmp_path, mocker):
    """model_cell_label_levels overrides the default per prefix: 'qlvm' fine only drops
    qlvm_supercategory (a stale one is removed, not kept), 'qlvm_x' both levels adds
    qlvm_x_supercategory, and an empty list writes coordinates only. Each label is the
    grid lookup of the written coordinates."""
    rng = np.random.default_rng(36)
    root, session_id, cfg = _model_cells_session(tmp_path, rng, prefixes=("qlvm", "qlvm_x", "qlvm_y"))
    cfg["model_cell_label_levels"] = {"qlvm": ["fine"], "qlvm_x": ["fine", "coarse"], "qlvm_y": []}
    summary_path = root / "audio" / f"{session_id}_usv_summary.csv"
    pls.read_csv(summary_path).with_columns(
        qlvm_supercategory=pls.Series([1, None, 2]),
        qlvm_y_category=pls.Series([7, 7, 7]),
    ).write_csv(summary_path)

    _run_model_cells(root, cfg, mocker)

    df = pls.read_csv(summary_path)
    assert df.columns == [
        "usv_id", "start", "stop", "qlvm1", "qlvm2", "qlvm_category",
        "qlvm_x1", "qlvm_x2", "qlvm_x_category", "qlvm_x_supercategory", "qlvm_y1", "qlvm_y2",
    ]
    _assert_labels_follow_coordinates(df, "qlvm", 0, ("fine",))
    _assert_labels_follow_coordinates(df, "qlvm_x", 1, ("fine", "coarse"))
    # The stand-in embedding places every call at the torus center: pixel (4, 4) of the 8 x 8 grids.
    assert df["qlvm_x_category"].to_list() == [_model_cell_grids(1)[0][4, 4], None, _model_cell_grids(1)[0][4, 4]]


@pytest.mark.parametrize(
    ("label_levels", "match"),
    [
        ([], "must be an object of column prefix -> list of label levels"),
        ({"qlvm": ["medium"]}, r"prefix 'qlvm': invalid level\(s\) \['medium'\]"),
        ({"qlvm": "fine"}, "prefix 'qlvm': levels must be a list, got str"),
        ({"qlvm": ["fine", "fine"]}, r"prefix 'qlvm': level\(s\) listed more than once: \['fine'\]"),
        ({"qlvm_z": ["fine"]}, r"prefixes not in model_cells: \['qlvm_z'\]"),
    ],
)
def test_model_cells_refuse_invalid_label_levels_before_writing(tmp_path, mocker, label_levels, match):
    """An invalid model_cell_label_levels -- not an object, an unknown level, a level that is
    not in a list, a repeated level, or a prefix model_cells does not list -- stops the run
    before any cell is loaded or the summary is touched."""
    rng = np.random.default_rng(37)
    root, session_id, cfg = _model_cells_session(tmp_path, rng, prefixes=("qlvm",))
    cfg["model_cell_label_levels"] = label_levels
    summary_path = root / "audio" / f"{session_id}_usv_summary.csv"
    before = summary_path.read_bytes()
    loads = mocker.patch("usv_playpen.processing.qlvm_latents.load_model_cell", side_effect=ql.load_model_cell)

    mocker.patch("usv_playpen.processing.qlvm_latents.smart_wait")
    with pytest.raises(ValueError, match=match):
        ql.QLVMLatentInference(
            root_directory=str(root),
            input_parameter_dict={"infer_qlvm_latents": cfg},
            message_output=lambda *_a, **_kw: None,
        ).infer_and_merge()
    assert loads.call_count == 0
    assert summary_path.read_bytes() == before


def test_model_cell_label_columns_refuse_to_overwrite_summary_columns(mocker):
    """A label column may not be another summary column; the production label columns
    (qlvm_category, qlvm_supercategory, qlvm_dur_category, ...) are in the canonical column
    order and are allowed, and so are the production prefixes' coordinates."""
    mocker.patch.object(ql, "USV_SUMMARY_COLUMN_ORDER", (*ql.USV_SUMMARY_COLUMN_ORDER, "peak_category"))
    with pytest.raises(ValueError, match=r"prefix 'peak': label column\(s\) \['peak_category'\] would overwrite"):
        ql.model_cell_label_columns({"peak": "/cell"}, {})
    production = {prefix: "/cell" for prefix in ql.QLVM_PRODUCTION_MODEL_CELLS}
    assert ql.validate_model_cells(production.items()) == production
    both = {prefix: ["fine", "coarse"] for prefix in production}
    assert ql.model_cell_label_columns(production, both)["qlvm_loud"] == {
        "fine": "qlvm_loud_category", "coarse": "qlvm_loud_supercategory",
    }
