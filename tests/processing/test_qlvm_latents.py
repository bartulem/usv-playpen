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
    _write_session_masks(root, session_id, {0: np.ones((n_f, n_t), dtype=bool), 2: np.ones((n_f, n_t), dtype=bool)})

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
    tmp_path, rng, *, masking_type, floor, fine_grid, coarse_grid, fib_m=8, condition=None, bins=None, require_mask=True
):
    """Synthesize a QLVM model package cell (checkpoint.tar, training_contract.json,
    cluster/{fine,coarse}/label_grid.npy) with a ReLU-head decoder; with a
    ``condition`` block, a conditional one (one extra decoder input) and its
    ``condition_bins.npz`` (``bins`` = (edges, bin_mean)). ``require_mask`` True, as
    in every v2 cell, says its corpus kept only calls with a SAM mask."""
    torch = pytest.importorskip("torch")
    cell = tmp_path / "pkg" / "phase_test" / "cell_test"
    (cell / "cluster" / "fine").mkdir(parents=True)
    (cell / "cluster" / "coarse").mkdir(parents=True)
    arrays = _relu_state_dict(rng)
    if condition is not None:
        arrays["decoder.0.weight"] = (rng.standard_normal((2048, 5)) * 0.05).astype(np.float32)
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
