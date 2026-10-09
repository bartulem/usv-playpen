"""
@author: bartulem

Tests for visualizations/qlvm_property_sweep_video.

Covers the pooling filter of load_property_sweep_usvs (through
make_usv_spectrograms.load_regular_map_cohort_usvs: pure USVs with a position
and all six properties, the temporary cohort session list removed afterwards),
the random order within tied values, a still-frame render and a short video
render (skipped without ffmpeg). The periodic density the panels draw is tested
with its module, in test_auxiliary_plot_functions.
"""

from __future__ import annotations

import pathlib
import shutil

import numpy as np
import polars as pls
import pytest

from usv_playpen.visualizations import qlvm_property_sweep_video as sweep


def _usvs(n: int, seed: int = 0) -> pls.DataFrame:
    """
    Description
    -----------
    A synthetic pooled table: torus positions and the six swept properties,
    the mask count integer (so it ties heavily).

    Parameters
    ----------
    n (int)
        Number of USVs.
    seed (int)
        Random seed.

    Returns
    -------
    usvs (pls.DataFrame)
        ``qlvm1``, ``qlvm2`` and the ``SWEEP_PROPERTIES`` columns.
    """

    rng = np.random.default_rng(seed)
    return pls.DataFrame({
        "qlvm1": rng.random(n), "qlvm2": rng.random(n),
        "mask_number": rng.integers(1, 4, n).astype(float), "duration": rng.uniform(0.01, 0.2, n),
        "freq_bandwidth_hz": rng.uniform(1e3, 7e4, n), "spectral_entropy": rng.uniform(1.5, 4.5, n),
        "mean_freq_hz": rng.uniform(4e4, 9e4, n), "loudness_db": rng.uniform(40, 90, n),
    })


def _label_grid(resolution: int = 40) -> np.ndarray:
    """
    Description
    -----------
    A four-category grid (quadrants), indexed ``[y, x]``.

    Parameters
    ----------
    resolution (int)
        Pixels per side.

    Returns
    -------
    label_grid (np.ndarray)
        ``(resolution, resolution)`` labels 1..4.
    """

    half = resolution // 2
    grid = np.ones((resolution, resolution), dtype=int)
    grid[:half, half:] = 2
    grid[half:, :half] = 3
    grid[half:, half:] = 4
    return grid


def test_load_property_sweep_usvs_filters_and_cleans_up(tmp_path, monkeypatch):
    """Only pure USVs with a position and six finite properties are kept, the cohort list
    pools every non-playback session list, and the temporary list is removed."""
    input_dir = tmp_path / "input_files"
    input_dir.mkdir()
    (input_dir / "a_sessions_list.txt").write_text("/root/sessA\n")
    (input_dir / "ephys_playback_sessions_list.txt").write_text("/root/sessP\n")
    pooled = _usvs(5).with_columns(
        pls.Series("usv", [True, True, False, True, True]),
        pls.Series("squeak", [False, True, True, False, False]),
        pls.Series("session_id", ["sessA"] * 5),
        pls.Series("row_index", list(range(5)), dtype=pls.UInt32),
    ).with_columns(pls.when(pls.int_range(pls.len()) == 3).then(None).otherwise(pls.col("qlvm1")).alias("qlvm1"),
                   pls.when(pls.int_range(pls.len()) == 4).then(float("nan")).otherwise(pls.col("duration")).alias("duration"))
    captured = {}

    def _stub(sessions_txt_path, cache_path, exclude_noise_usvs, message_output):
        captured["roots"] = pathlib.Path(sessions_txt_path).read_text().splitlines()
        captured["path"] = sessions_txt_path
        captured["cache"] = cache_path
        return pooled

    monkeypatch.setattr("usv_playpen.visualizations.make_usv_spectrograms.build_pooled_embeddings_df", _stub)
    usvs = sweep.load_property_sweep_usvs(str(input_dir), None, True, message_output=lambda *_a, **_k: None)
    assert captured["roots"] == ["/root/sessA"]
    assert captured["cache"] is None
    assert not pathlib.Path(captured["path"]).exists()
    assert usvs.height == 1
    assert usvs.columns == ["qlvm1", "qlvm2", *(column for column, _, _, _ in sweep.SWEEP_PROPERTIES)]


def test_ties_are_put_in_random_order():
    """Within a tie the order follows the seeded tie-breaker, not the file order."""
    values = np.array([1.0] * 50 + [2.0] * 50)
    tie_breaker = np.random.default_rng(0).random(values.size)
    order = np.lexsort((tie_breaker, values))
    assert set(order[:50]) == set(range(50))
    assert not np.array_equal(order[:50], np.arange(50))


def test_preview_frame_is_written(tmp_path):
    """A preview renders one still of the window at the given quantile, next to the output path."""
    fig = sweep.render_qlvm_property_sweep_video(
        _usvs(2000), _label_grid(), str(tmp_path / "sweep.mp4"), cmap_name="inferno", colormap_floor=0.2, window=0.1,
        step=0.1, hold=1, fps=5, dpi=30, density_bandwidth=0.02, seed=0, preview_quantile=0.5,
        message_output=lambda *_a, **_k: None)
    assert (tmp_path / "sweep.png").exists()
    assert not (tmp_path / "sweep.mp4").exists()
    assert "45th and 55th percentile" in fig.texts[0].get_text()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not on the PATH")
def test_video_is_written(tmp_path):
    """A short sweep (coarse step) writes a non-empty .mp4."""
    sweep.render_qlvm_property_sweep_video(
        _usvs(2000), _label_grid(), str(tmp_path / "sweep.mp4"), cmap_name="inferno", colormap_floor=0.2, window=0.2,
        step=0.3, hold=1, fps=5, dpi=30, density_bandwidth=0.02, seed=0, message_output=lambda *_a, **_k: None)
    assert (tmp_path / "sweep.mp4").stat().st_size > 0
