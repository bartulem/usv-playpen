"""
@author: bartulem

Tests for visualizations/qlvm_figures.

Covers the metric helpers (torus neighbors, kNN rank readout, neighborhood
scores, aligned correlation, normalized similarity, group contrast, aligned
means), the per-session sample (decoder inputs, masked native-length images,
the cache served only for the same USVs), and a small end-to-end render of the
three figures on synthetic sessions (spectrogram H5s written to a temporary
directory, a two-band category grid, a stubbed decoder geometry).
"""

from __future__ import annotations

import pathlib

import h5py
import numpy as np
import polars as pls
import pytest
from matplotlib.figure import Figure

from usv_playpen.visualizations import qlvm_figures as qf


def _cfg(n_usvs: int = 5000) -> dict:
    """
    Description
    -----------
    A small ``qlvm_figures`` settings block for synthetic data.

    Parameters
    ----------
    n_usvs (int)
        Most USVs drawn for the map panels.

    Returns
    -------
    cfg (dict)
        The block.
    """

    return {"n_usvs": n_usvs, "usvs_per_session": 40, "sample_cache_file": "", "seed_cell_directories": [], "watershed_levels_file": "",
            "category_colors": ["#4E79A7", "#F28E2B"],
            "overview": {"n_examples": 3, "spiral_radius": 0.1, "spiral_turns": 2, "density_bandwidth": 0.02},
            "quality": {"n_pcs": 5, "umap_n_neighbors": 10, "umap_min_dist": 0.1, "readout_neighbors": 5, "readout_folds": 2,
                        "neighborhood_k": 5, "n_neighborhood_subsets": 2, "neighborhood_subset_size": 60, "distance_sample_size": 50,
                        "n_distance_samples": 2, "pullback_grid": 8, "embed_lattice_batch_size": 64, "embed_data_batch_size": 32},
            "categories": {"n_shifts": 20, "histogram_bins": 10, "similarity_downsample": 2, "similarity_max_shift": 1,
                           "balanced_per_category": 30, "balanced_reference_per_category": 10, "closest_matches": 5,
                           "closest_log2_limit": 2.0, "partition_sample_size": 60, "partition_reference_size": 20, "average_size": 10,
                           "align_max_shift": 2, "align_iterations": 2, "average_time_ms": 150.0}}


def _two_band_grid(resolution: int = 40) -> np.ndarray:
    """
    Description
    -----------
    A two-category grid: category 1 left half, category 2 right half (``[y, x]``).

    Parameters
    ----------
    resolution (int)
        Pixels per side.

    Returns
    -------
    grid (np.ndarray)
        ``(resolution, resolution)`` labels 1 and 2.
    """

    grid = np.ones((resolution, resolution), dtype=np.int64)
    grid[:, resolution // 2:] = 2
    return grid


def _synthetic_cohort(tmp_path: pathlib.Path, n_sessions: int = 3, per_session: int = 40) -> tuple[pls.DataFrame, dict]:
    """
    Description
    -----------
    Synthetic sessions: per session a spectrogram H5 (``spectrogram/<id>`` with
    ``spectrograms`` and ``durations``, no mask group) and a pooled table whose
    USVs of category 1 (left half of the torus) are low tones and of category 2
    (right half) high tones, the six properties following the position.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Directory the session folders are written to.
    n_sessions (int)
        Number of sessions.
    per_session (int)
        USVs per session.

    Returns
    -------
    usvs (pls.DataFrame)
        The pooled table.
    roots (dict)
        Session id -> root.
    """

    rng = np.random.default_rng(0)
    frames, roots = [], {}
    for index in range(n_sessions):
        session_id = f"2025010{index}_120000"
        root = tmp_path / session_id
        (root / "audio" / "spectrograms").mkdir(parents=True)
        x = rng.random(per_session)
        y = rng.random(per_session)
        category = np.where(x < 0.5, 1, 2)
        durations = rng.integers(10, 40, per_session)
        specs = np.zeros((per_session, 128, 128), dtype=np.float32)
        for row in range(per_session):
            band = 30 if category[row] == 1 else 90
            specs[row, band:band + 4, :durations[row]] = 1.0
        with h5py.File(root / "audio" / "spectrograms" / f"{session_id}_spectrograms.h5", "w") as h5:
            h5.create_dataset(f"spectrogram/{session_id}/spectrograms", data=specs)
            h5.create_dataset(f"spectrogram/{session_id}/durations", data=durations)
        roots[session_id] = root
        frames.append(pls.DataFrame({
            "session_id": [session_id] * per_session, "row_index": np.arange(per_session, dtype=np.uint32),
            "qlvm1": x, "qlvm2": y, "qlvm_category": category,
            "duration": durations * 0.002 + 0.001 * x, "spectral_entropy": 2.0 + x, "freq_bandwidth_hz": 1e4 * (1 + x),
            "mean_freq_hz": 5e4 + 3e4 * x, "loudness_db": 60 + 10 * y, "mask_number": 1.0 + (x > 0.5)}))
    return pls.concat(frames), roots


def _bundle(tmp_path: pathlib.Path, grid: np.ndarray) -> dict:
    """
    Description
    -----------
    A minimal category bundle (the keys the figures read) with a content-change
    grid written to ``category_grids.npz``.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Directory of the bundle.
    grid (np.ndarray)
        The label grid.

    Returns
    -------
    bundle (dict)
        ``directory``, ``label_grid``, ``axis``, ``centers``, ``descriptions``, ``build_config``.
    """

    resolution = grid.shape[0]
    directory = tmp_path / "bundle"
    directory.mkdir()
    np.savez(directory / qf.QLVM_CATEGORY_GRIDS_NAME, label_grid=grid, content_change=np.random.default_rng(1).random(grid.shape))
    return {"directory": str(directory), "label_grid": grid, "axis": (np.arange(resolution) + 0.5) / resolution,
            "centers": np.array([[0.25, 0.5], [0.75, 0.5]]), "descriptions": ["low", "high"],
            "build_config": {"build_qlvm_categories": {"field_sigma": 2.0, "change_span": 2}}}


def test_torus_neighbors_wrap_across_the_seam():
    """On the torus a point at x = 0.99 is the nearest neighbor of one at x = 0.01; on the plane it is not."""
    points = np.array([[0.99, 0.5], [0.3, 0.5], [0.01, 0.5]])
    assert qf.neighbor_indices(points, points[[2]], 1, torus=True, exclude_self=True)[0, 0] == 0
    assert qf.neighbor_indices(points, points[[2]], 1, torus=False, exclude_self=True)[0, 0] == 1


def test_knn_rank_readout_separates_signal_from_shuffle():
    """A property that follows position reads out well; the same positions shuffled read out near 0 or below."""
    rng = np.random.default_rng(0)
    xy = rng.random((600, 2))
    target = np.sin(2 * np.pi * xy[:, 0])
    groups = np.repeat(np.arange(6), 100)
    r2, folds = qf.knn_rank_r2(xy, target, groups, True, 10, 3)
    assert r2 > 0.8 and len(folds) == 3
    assert qf.knn_rank_r2(xy[rng.permutation(600)], target, groups, True, 10, 3)[0] < 0.1


def test_neighborhood_scores_of_a_perfect_and_a_random_map():
    """A map identical to the reference finds every true neighbor with no error; a random map does not."""
    rng = np.random.default_rng(0)
    points = rng.random((80, 3))
    reference = np.linalg.norm(points[:, None] - points[None], axis=2)
    perfect = qf.neighborhood_scores(reference, reference, 5)
    assert perfect["overlap"] == pytest.approx(1.0)
    assert perfect["false_neighbors"] == pytest.approx(0.0) and perfect["torn_neighbors"] == pytest.approx(0.0)
    other = rng.random((80, 2))
    random_map = qf.neighborhood_scores(np.linalg.norm(other[:, None] - other[None], axis=2), reference, 5)
    assert random_map["overlap"] < 0.3 and random_map["false_neighbors"] > 0.1


def test_aligned_correlation_finds_a_shifted_copy():
    """A copy shifted by one bin correlates 1 after alignment, and the matrix is symmetric."""
    base = np.zeros((3, 16, 16), dtype=np.float32)
    base[0, 4:6, 2:10] = 1.0
    base[1, 5:7, 2:10] = 1.0
    base[2, 12:14, 1:3] = 1.0
    similarity = qf.aligned_correlation(base, 1)
    assert similarity[0, 1] == pytest.approx(1.0, abs=1e-5)
    assert np.allclose(similarity, similarity.T)
    assert similarity[0, 2] < 0.9


def test_rank_similarity_excludes_self_and_is_symmetric():
    """Worked example (references USVs 0 and 1): USV 0's yardstick is USV 1 only, so 0.8 beats all of it
    (1.0); USV 2's yardstick is USVs 0 and 1, where 0.8 beats half (0.5); the pair's value is their mean
    0.75. Counting USV 0 in its own yardstick (self-similarity 1) would give 0.5 instead."""
    raw = np.array([[1.0, 0.5, 0.8],
                    [0.5, 1.0, 0.3],
                    [0.8, 0.3, 1.0]], dtype=np.float32)
    ranked = qf.rank_similarity(raw, np.array([0, 1]))
    assert ranked[0, 2] == pytest.approx(0.75)
    assert ranked[0, 1] == pytest.approx(0.0)
    assert np.allclose(ranked[[0, 0, 1], [1, 2, 2]], ranked[[1, 2, 2], [0, 0, 1]])
    assert np.all(np.isnan(np.diag(ranked)))


def test_group_contrast_is_positive_for_coherent_groups():
    """USVs more similar within their group than across get positive own-minus-other values."""
    labels = np.array([1, 1, 1, 2, 2, 2])
    filled = np.where(labels[:, None] == labels[None, :], 0.9, 0.2).astype(np.float32)
    np.fill_diagonal(filled, 0.0)
    assert np.allclose(qf.group_contrast(filled, labels), 0.7)


def test_aligned_category_mean_recovers_a_jittered_shape():
    """Spectrograms of one shape at jittered positions align to a sharper template than their plain mean."""
    rng = np.random.default_rng(0)
    stack = np.zeros((30, 32, 32), dtype=np.float32)
    for index in range(30):
        shift = rng.integers(-2, 3)
        stack[index, 15 + shift:17 + shift, 5:20] = 1.0
    plain = stack.mean(axis=0)
    aligned = qf.aligned_category_mean(stack, 3, 3)
    assert aligned.max() > plain.max()


def test_figure_sample_reads_images_and_reuses_its_cache(tmp_path, monkeypatch):
    """The sample holds masked native-length images (zeros after the USV), decoder inputs of the same
    USVs, and is served from the cache only for the same USVs."""
    usvs, roots = _synthetic_cohort(tmp_path)
    monkeypatch.setattr(qf, "load_model_cell", lambda directory: {"contract": {"input_normalization": "none"}})
    cache = tmp_path / "cache" / "sample.npz"
    sample = qf.build_qlvm_figure_sample(usvs, roots, 10, 0, "cell", str(cache), message_output=lambda *_a, **_k: None)
    assert sample["images"].shape == (30, 128, 128) and sample["inputs"].shape == (30, 128, 128)
    first = int(sample["lengths"][0])
    assert np.all(sample["images"][0, :, first:] == 0) and sample["images"][0].max() > 0
    assert cache.exists()
    monkeypatch.setattr(qf, "load_model_cell", lambda directory: pytest.fail("the cache should have been served"))
    again = qf.build_qlvm_figure_sample(usvs, roots, 10, 0, "cell", str(cache), message_output=lambda *_a, **_k: None)
    assert np.array_equal(again["row_index"], sample["row_index"])


def test_three_figures_render_on_synthetic_sessions(tmp_path, monkeypatch):
    """The overview, the quality figure (stubbed decoder geometry) and the category figure render, and the
    categories of the synthetic data (two tones) score above a band partition orthogonal to them."""
    usvs, roots = _synthetic_cohort(tmp_path, n_sessions=4, per_session=60)
    grid = _two_band_grid()
    bundle = _bundle(tmp_path, grid)
    cfg = _cfg()
    monkeypatch.setattr(qf, "load_model_cell", lambda directory: {"contract": {"input_normalization": "none"}})
    sample = qf.build_qlvm_figure_sample(usvs, roots, cfg["usvs_per_session"], 0, "cell", None, message_output=lambda *_a, **_k: None)

    fig, table = qf.plot_qlvm_overview(usvs, bundle, roots, cfg, "inferno", 0)
    assert isinstance(fig, Figure) and table.height == 2 * cfg["overview"]["n_examples"]

    class Geometry:
        grid = np.stack(np.meshgrid((np.arange(8) + 0.5) / 8, (np.arange(8) + 0.5) / 8), axis=-1).reshape(-1, 2)
        pullback_matrix = np.random.default_rng(2).random((64, 64))
        density_matrix = np.random.default_rng(3).random((64, 64))

    monkeypatch.setattr(qf, "make_qlvm_decode_fn_from_model_cell", lambda directory: None)
    monkeypatch.setattr(qf, "build_torus_geodesic_context", lambda points, decode_fn, n_per_dim: Geometry())
    metrics = qf.compute_qlvm_quality_metrics(sample, cfg, 0, "cell", {}, message_output=lambda *_a, **_k: None)
    assert set(metrics["maps"]) == {"QLVM", "PCA", "UMAP"}
    assert np.mean(metrics["readout_folds"]["QLVM"]["mean_freq_hz"]) > 0.5
    assert isinstance(qf.plot_qlvm_quality(sample, metrics, cfg, "inferno", 0), Figure)

    orthogonal = np.ones_like(grid)
    orthogonal[grid.shape[0] // 2:, :] = 2
    fields = qf.compute_category_boundary_fields(usvs, bundle, cfg, 0)
    similarity = qf.compute_category_similarity(sample, grid, {"rows": orthogonal}, cfg, 0, message_output=lambda *_a, **_k: None)
    scores = {name: float(np.mean(values)) for name, values in similarity["partition_scores"].items()}
    assert scores["content ridges\n(2)"] > scores["rows"]
    assert similarity["matrix"][0, 0] > similarity["matrix"][0, 1]
    assert np.allclose(similarity["neighbor_shares"].sum(axis=1), 1.0)
    assert isinstance(qf.plot_category_boundaries(fields, similarity, bundle, cfg, "inferno"), Figure)
