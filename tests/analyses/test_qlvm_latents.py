"""
@author: bartulem
Tests for analyses/qlvm_latents — the QLVM inference driver.

Covers the helper pieces (weight loading + ``decoder.`` prefix stripping,
lattice rebuild, reference label lookup) and an end-to-end run that synthesizes
a decoder-weights ``.npz``, a reference ``arrays.npz`` (ws grids), a session
spectrogram H5 and a ``usv_summary.csv``, then checks the ``qlvm_*`` columns are
merged into the right rows.
"""

from __future__ import annotations

import h5py
import numpy as np
import polars as pls

from usv_playpen.analyses import qlvm_latents as ql


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
    """Coordinate (x, y) maps to grid[int(y*res), int(x*res)] with clipping."""
    ws = np.arange(16).reshape(4, 4).astype(np.int16)        # res = 4
    ws_per = (ws + 100).astype(np.int16)
    coords = np.array([[0.0, 0.0], [0.9, 0.1], [0.1, 0.9]])  # -> (px,py): (0,0),(3,0),(0,3)
    cat, supercat = ql.labels_for_coords(coords, ws, ws_per)
    assert cat.tolist() == [ws[0, 0], ws[0, 3], ws[3, 0]]
    assert supercat.tolist() == [ws_per[0, 0], ws_per[0, 3], ws_per[3, 0]]


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

    # weights + reference grids (res=8)
    weights = tmp_path / "qmc_decoder_weights.npz"
    _decoder_weights_npz(weights, rng)
    arrays = tmp_path / "arrays.npz"
    res = 8
    np.savez(
        arrays,
        ws_labels=(rng.integers(0, 4, size=(res, res))).astype(np.int16),
        ws_labels_periodic=(rng.integers(0, 4, size=(res, res))).astype(np.int16),
    )

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
        "weights_npz_path": str(weights),
        "reference_arrays_npz_path": str(arrays),
        "lattice_type": "korobov",
        "latent_dim": 2,
        "n_points": 16,
        "korobov_a": 3,
        "fib_m": 16,
        "time_stretch": False,
    }
    mocker.patch("usv_playpen.analyses.qlvm_latents.smart_wait")
    ql.QLVMLatentInference(
        root_directory=str(root),
        input_parameter_dict={"infer_qlvm_latents": cfg},
        message_output=lambda *_a, **_kw: None,
    ).infer_and_merge()

    df = pls.read_csv(root / "audio" / f"{session_id}_usv_summary.csv")
    assert set(ql.QLVM_COLUMNS).issubset(df.columns)
    assert df.height == 3
    # rows 0 and 2 embedded; row 1 (no spec) is null.
    assert df["qlvm_umap1"][0] is not None
    assert df["qlvm_umap1"][1] is None
    assert df["qlvm_umap1"][2] is not None
    # coordinates live on the torus [0, 1).
    assert 0.0 <= df["qlvm_umap1"][0] < 1.0
