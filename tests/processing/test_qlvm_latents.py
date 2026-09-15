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
import pickle
import zipfile

import h5py
import numpy as np
import polars as pls
import pytest

from usv_playpen.processing import qlvm_latents as ql


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

    pls.DataFrame({
        "usv_id": [f"{i:04d}" for i in range(3)],
        "start": [0.1, 0.3, 0.5],
        "stop": [0.15, 0.35, 0.55],
    }).write_csv(root / "audio" / f"{session_id}_usv_summary.csv")

    cfg = {
        "model_cell_directory": "",
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


def _make_inference_session(tmp_path, rng, *, fine_grid, coarse_grid):
    """Synthesize a session (weights + fine/coarse reference grids + spectrogram
    H5 + usv_summary) for an end-to-end QLVMLatentInference run and return
    (root, session_id, cfg). Rows 0 and 2 are real, row 1 is a placeholder."""
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

    pls.DataFrame({
        "usv_id": [f"{i:04d}" for i in range(3)],
        "start": [0.1, 0.3, 0.5],
        "stop": [0.15, 0.35, 0.55],
    }).write_csv(root / "audio" / f"{session_id}_usv_summary.csv")

    cfg = {
        "model_cell_directory": "",
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
    root, session_id, cfg = _make_inference_session(tmp_path, rng, fine_grid=fine_grid, coarse_grid=coarse_grid)

    # Add a SAM mask group covering only the top half (rows 0:64) of each real USV
    # (spectrogram rows 0 and 2); build_session_masks unions per spectrogram_index.
    n_f = n_t = 128
    seg = np.zeros((2, n_f, n_t), dtype=bool)
    seg[:, :64, :] = True
    h5_loc = root / "audio" / "spectrograms" / f"{session_id}_spectrograms.h5"
    with h5py.File(h5_loc, "a") as f:
        mask_group = f.create_group(f"mask/{session_id}")
        mask_group.create_dataset("segmentations", data=seg)
        mask_group.create_dataset("spectrogram_index", data=np.array([0, 2], dtype=np.int64))

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
        "input_normalization": "none",
        "floor": None,
        "masking_type": cfg["masking_type"],
        "target_shape": list(cfg["target_shape"]),
        "time_stretch": cfg["time_stretch"],
        "length_threshold": 128.0,
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


def _make_model_cell(tmp_path, rng, *, masking_type, floor, fine_grid, coarse_grid, fib_m=8):
    """Synthesize a QLVM model package cell (checkpoint.tar, training_contract.json,
    cluster/{fine,coarse}/label_grid.npy) with a ReLU-head decoder."""
    torch = pytest.importorskip("torch")
    cell = tmp_path / "pkg" / "phase_test" / "cell_test"
    (cell / "cluster" / "fine").mkdir(parents=True)
    (cell / "cluster" / "coarse").mkdir(parents=True)
    state = {key: torch.from_numpy(value) for key, value in _relu_state_dict(rng).items()}
    torch.save({"model": state, "optimizer": {}, "run info": []}, cell / "checkpoint.tar")
    contract = {
        "decoder_head": "relu", "latent_dim": 2, "c_dim": 0, "conditional": None,
        "input_normalization": "minmax", "normalization_epsilon": 1e-8,
        "masking_type": masking_type, "floor": floor,
        "target_shape": [128, 128], "time_stretch": False, "length_threshold": 100.0,
        "embedding_lattice_type": "fibonacci", "embedding_fib_m": fib_m,
        "training_lattice_type": "fibonacci", "training_fib_m": 5, "validation_fib_m": 6, "condition": None,
    }
    (cell / "training_contract.json").write_text(json.dumps(contract))
    np.save(cell / "cluster" / "fine" / "label_grid.npy", fine_grid)
    np.save(cell / "cluster" / "coarse" / "label_grid.npy", coarse_grid)
    return cell


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
