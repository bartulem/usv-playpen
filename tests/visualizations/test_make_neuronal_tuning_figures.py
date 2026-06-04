"""
@author: bartulem
End-to-end regression tests for `NeuronalTuningFigureMaker`.

These tests exist to catch the bug class where `generate-rm-figs` reports
"success" (zero non-zero exit code, no fatal exceptions) but produces
zero output files on disk. The original failure was a combination of:

  1. The figure side's `has_vocal` predicate looking for keys starting
     with `vocal_q`, while the compute side writes keys starting with
     `usv_`. Result: vocal pages were silently skipped on every cluster.
  2. The figure side's tracking-H5 lookup restricted to `<root>/video/`,
     while the analysis side searches the whole session tree. Result:
     `mouse_id_list` came back empty on sessions whose tracking H5 sits
     at the session root, and behavioral pages were skipped.
  3. The per-cluster broad `except Exception` swallowing the
     `RuntimeError("Cannot save empty PDF file")` raised when a
     `PdfPages` context closes with zero pages, leaving a clean log
     trail of "failed to render" messages and no output files.

A passing test here therefore requires:
  * the compute side actually writes `usv_*` keys when vocal payload
    exists, AND the figure side actually reads them
  * the figure side actually finds the tracking H5 wherever it lives,
    AND uses it to render behavioral pages
  * the figure-render context actually emits the PDF to disk

We do not assert byte-exact figure contents — only that real files
appear on disk and contain a sensible number of pages.
"""

from __future__ import annotations

import json
import pathlib
import pickle
import re

import h5py
import matplotlib
import matplotlib.collections
import matplotlib.pyplot as plt
import numpy as np
import polars as pls
import pytest

from usv_playpen.analyses.compute_neuronal_tuning_curves import (
    CATEGORICAL_FEATURES,
    CONTINUOUS_PROPERTIES,
    NeuronalTuning,
)
from usv_playpen.visualizations.make_neuronal_tuning_figures import (
    DISPLAY_FACTOR,
    NeuronalTuningFigureMaker,
)


def _build_synthetic_figure_session(
    root: pathlib.Path,
    *,
    tracking_h5_at_session_root: bool,
    n_frames: int = 1500,
    n_usvs: int = 120,
    fps: float = 150.0,
) -> None:
    """
    Description
    -----------
    Build a minimal session directory tree carrying every input the
    `generate-rm` -> `generate-rm-figs` pipeline reads: a behavioral
    features CSV, a translated/rotated tracking H5, a USV summary CSV,
    and a per-cluster spike `.npy`. Single-mouse layout (`m1`).

    Two layout variants are supported via `tracking_h5_at_session_root`:
      * False -> the tracking H5 sits under `<root>/video/vid1/` (the
        layout the figure code originally assumed).
      * True  -> the tracking H5 sits directly at `<root>/` (the
        scratch-cluster layout that triggered the original bug, where
        the figure code's `root / "video"` lookup silently returned
        nothing).

    Parameters
    ----------
    root (pathlib.Path)
        Session root to populate.
    tracking_h5_at_session_root (bool)
        Whether to write the tracking H5 at the session root (True) or
        under the conventional `video/vid1/` subdirectory (False).
    n_frames (int)
        Length of the synthetic tracking series; session duration is
        `n_frames / fps`.
    n_usvs (int)
        Number of USVs written into the USV summary CSV. All are
        attributed to `m1` so the `n_usv_min_self` gate passes.
    fps (float)
        Camera frame rate the compute reads from the H5.

    Returns
    -------
    None
    """

    (root / "video" / "vid1").mkdir(parents=True)
    (root / "audio" / "sync").mkdir(parents=True)
    (root / "ephys" / "imec0" / "cluster_data").mkdir(parents=True)

    rng = np.random.default_rng(0)

    df_beh = pls.DataFrame({
        "m1.speed":        rng.uniform(0, 30, n_frames),
        "m1.acceleration": rng.uniform(-100, 100, n_frames),
        "m1.allo_yaw":     rng.uniform(-180, 180, n_frames),
        "m1.spaceX":       rng.uniform(-30, 30, n_frames),
        "m1.spaceY":       rng.uniform(-30, 30, n_frames),
    })
    df_beh.write_csv(root / "video" / "vid1" / "vid1_behavioral_features.csv")

    if tracking_h5_at_session_root:
        h5_path = root / "vid1_points3d_translated_rotated_metric.h5"
    else:
        h5_path = root / "video" / "vid1" / "vid1_points3d_translated_rotated_metric.h5"
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("track_names", data=np.array([b"m1"]))
        f.create_dataset("recording_frame_rate", data=np.float64(fps))
        # `E1M` decodes to {experiment_type: [ephys], mouse_number: 1,
        # mouse_sex: [male], ...} — `choose_animal_colors` needs a
        # non-empty `mouse_sex` of the right length to iterate.
        f.create_dataset("experimental_code", data=b"E1M")
        f.create_dataset("tracks", data=np.zeros((n_frames, 1, 1, 3), dtype=float))

    duration_s = float(n_frames / fps)
    sync_json = root / "audio" / "sync" / "audio_triggerbox_sync_info.json"
    sync_json.write_text(json.dumps({"m": {"duration_seconds": duration_s}}))

    starts = np.sort(rng.uniform(2, duration_s - 2, n_usvs))
    durations = rng.uniform(0.05, 0.2, n_usvs)
    stops = starts + durations
    usv_df = pls.DataFrame({
        "start":             starts.tolist(),
        "stop":              stops.tolist(),
        "duration":          durations.tolist(),
        "emitter":           ["m1"] * n_usvs,
        "vae_supercategory": rng.integers(1, 5, size=n_usvs).tolist(),
        "vae_category":      rng.integers(1, 8, size=n_usvs).tolist(),
        "qlvm_supercategory": rng.integers(1, 4, size=n_usvs).tolist(),
        "qlvm_category":     rng.integers(1, 6, size=n_usvs).tolist(),
        "mean_freq_hz":      rng.uniform(40000, 90000, n_usvs).tolist(),
        "peak_freq_hz":      rng.uniform(40000, 90000, n_usvs).tolist(),
        "freq_bandwidth_hz": rng.uniform(5000, 30000, n_usvs).tolist(),
        "mean_amplitude":    rng.uniform(0.1, 3.0, n_usvs).tolist(),
        "max_amplitude":     rng.uniform(0.5, 8.0, n_usvs).tolist(),
        "spectral_entropy":  rng.uniform(1.0, 4.0, n_usvs).tolist(),
        "mask_number":       rng.integers(1, 12, n_usvs).tolist(),
    })
    usv_df.write_csv(root / "audio" / "sync" / "vid1_usv_summary.csv")

    n_spikes = 800
    spike_times = np.sort(rng.uniform(0, duration_s, n_spikes))
    spike_frames = np.clip(
        np.floor(spike_times * fps).astype(np.int64), 0, n_frames - 1,
    )
    cluster_arr = np.vstack([spike_times, spike_frames.astype(float)])
    np.save(
        root / "ephys" / "imec0" / "cluster_data" / "imec0_cl0001_ch001_good.npy",
        cluster_arr,
    )


def _make_tuning_parameters() -> dict:
    """
    Description
    -----------
    Return a small, fast `tuning_parameters_dict` suitable for the
    full-pipeline test. `n_shuffles=5`, low bin counts, low USV minimums.

    Parameters
    ----------

    Returns
    -------
    params (dict)
        Tuning-parameter dict consumed by
        `NeuronalTuning.calculate_neuronal_tuning_curves`.
    """

    return {
        "temporal_offsets":                         [0],
        "n_shuffles":                               5,
        "shuffle_seed":                             0,
        "total_bin_num":                            10,
        "n_spatial_bins":                           36,
        "spatial_scale_cm":                         32,
        "shuffle_seconds_range":                    [3.0, 6.0],
        "peth_window_seconds":                      [-2.0, 0.0],
        "peth_bin_seconds":                         0.05,
        "bout_quiet_seconds":                       2.0,
        "vocal_require_clean_post_anchor":          True,
        "vocal_require_clean_prior_anchor":         False,
        "n_usv_min_self":                           5,
        "n_usv_min_partner":                        30,
        "n_usv_min_category":                       3,
        "behavioral_min_occupancy_seconds":         0.1,
        "usv_property_min_occupancy_seconds":       0.05,
        "include_partner_vocalization_tuning_bool": False,
        "shuffle_chunk_size":                       4,
        "smoothing_sd":                             0.0,
        "circular_features":                        ["allo_yaw", "body_dir"],
    }


def _make_visualizations_parameters() -> dict:
    """
    Description
    -----------
    Return the minimum `visualizations_parameter_dict` the figure maker
    reads: `male_colors` / `female_colors` palettes (only first entry of
    each is touched). The `figures` block is left out so the code path
    falls back to `fig_format='pdf'` and `cmap='inferno'`.

    Parameters
    ----------

    Returns
    -------
    viz (dict)
        Visualizations-parameter dict consumed by the figure maker.
    """

    return {
        "male_colors":   ["#202020"],
        "female_colors": ["#a83232"],
    }


def _count_pdf_pages(pdf_path: pathlib.Path) -> int:
    """
    Description
    -----------
    Count the page objects in a matplotlib-generated PDF by counting
    `/Type /Page` occurrences (the page-tree node `/Type /Pages` is
    excluded via the negative lookahead `(?!s)`). Empirically reliable
    for matplotlib's `PdfPages` output and avoids adding a runtime
    PDF-library dependency to the test suite.

    Parameters
    ----------
    pdf_path (pathlib.Path)
        Path to the PDF file to inspect.

    Returns
    -------
    n_pages (int)
        Number of page objects in the PDF.
    """

    data = pdf_path.read_bytes()
    return len(re.findall(rb"/Type\s*/Page(?!s)", data))


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_full_pipeline_writes_pdf_with_behavioral_and_vocal_pages(tmp_path):
    """
    Description
    -----------
    End-to-end regression: building a complete synthetic session (with
    both behavioral and vocal inputs), running
    `NeuronalTuning.calculate_neuronal_tuning_curves` to populate the
    per-cluster pkl, and then running
    `NeuronalTuningFigureMaker.make_neuronal_tuning_figures` must result
    in a real, multi-page PDF on disk under
    `<root>/ephys/tuning_curves/`. Asserts:

      * the cluster pkl carries both `beh_offset=*` and `usv_*` keys
        (catches a regression where compute/figure key names drift apart)
      * exactly one PDF exists for the single cluster
      * the PDF has >= 3 pages (1 behavioral + at least 2 vocal),
        catching the historical "zero-page PdfPages" silent failure

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Pytest-provided per-test temp directory.

    Returns
    -------
    None
    """

    root = tmp_path / "session"
    root.mkdir()
    _build_synthetic_figure_session(root, tracking_h5_at_session_root=False)

    nt = NeuronalTuning(
        root_directory=str(root),
        tuning_parameters_dict=_make_tuning_parameters(),
        message_output=lambda *a, **k: None,
    )
    nt.calculate_neuronal_tuning_curves()

    tuning_dir = root / "ephys" / "tuning_curves"
    pkls = sorted(tuning_dir.glob("*_tuning_curves_data.pkl"))
    assert len(pkls) == 1, f"expected 1 cluster pkl, got {len(pkls)}"

    with pkls[0].open("rb") as fh:
        cluster_data = pickle.load(fh)
    has_behavioral = any(k.startswith("beh_offset=") for k in cluster_data)
    has_vocal = any(k.startswith("usv_") for k in cluster_data)
    assert has_behavioral, "compute did not write any beh_offset=* keys"
    assert has_vocal, (
        "compute did not write any usv_* keys — vocal compute silently "
        "short-circuited (check n_usv_min_self, tracking H5 lookup, or the "
        "_load_vocal_inputs gates)"
    )

    figure_maker = NeuronalTuningFigureMaker(
        root_directory=str(root),
        visualizations_parameter_dict=_make_visualizations_parameters(),
        message_output=lambda *a, **k: None,
    )
    figure_maker.make_neuronal_tuning_figures()

    pdfs = sorted(tuning_dir.glob("*_neuronal_tuning.pdf"))
    assert len(pdfs) == 1, (
        f"expected 1 rendered PDF, found {len(pdfs)} — the figure maker "
        "ran but produced no output files (likely empty PdfPages closed by "
        "the broad-except swallowing 'Cannot save empty PDF file')"
    )
    pdf = pdfs[0]
    assert pdf.stat().st_size > 10_000, (
        f"PDF too small ({pdf.stat().st_size} bytes); matplotlib closed a "
        "near-empty PDF — payload-keys/predicate mismatch likely"
    )
    n_pages = _count_pdf_pages(pdf)
    assert n_pages >= 3, (
        f"PDF has only {n_pages} pages; expected behavioral + vocal "
        "(>=3 total). A 1-page PDF means only one of the two halves "
        "rendered — usually the figure-side `has_vocal` predicate not "
        "matching the compute side's key names, or the tracking-H5 "
        "lookup returning an empty mouse_id_list."
    )


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_figures_render_when_tracking_h5_at_session_root(tmp_path):
    """
    Description
    -----------
    Regression test for the specific scratch-cluster session layout
    where the translated/rotated tracking H5 sits directly at the
    session root rather than inside a `video/` subdirectory. The
    original figure code restricted its lookup to `<root>/video/`,
    which silently produced `mouse_id_list=[]` on this layout and
    therefore zero behavioral pages. With the lookup now mirroring the
    analysis side (`root=root`, `recursive=True`), the figure pipeline
    must still produce a real multi-page PDF.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Pytest-provided per-test temp directory.

    Returns
    -------
    None
    """

    root = tmp_path / "session"
    root.mkdir()
    _build_synthetic_figure_session(root, tracking_h5_at_session_root=True)

    nt = NeuronalTuning(
        root_directory=str(root),
        tuning_parameters_dict=_make_tuning_parameters(),
        message_output=lambda *a, **k: None,
    )
    nt.calculate_neuronal_tuning_curves()

    figure_maker = NeuronalTuningFigureMaker(
        root_directory=str(root),
        visualizations_parameter_dict=_make_visualizations_parameters(),
        message_output=lambda *a, **k: None,
    )
    figure_maker.make_neuronal_tuning_figures()

    tuning_dir = root / "ephys" / "tuning_curves"
    pdfs = sorted(tuning_dir.glob("*_neuronal_tuning.pdf"))
    assert len(pdfs) == 1, (
        f"expected 1 PDF on scratch-style layout (tracking H5 at session "
        f"root), got {len(pdfs)}; the figure code likely still scopes its "
        "tracking-H5 lookup to <root>/video/."
    )
    n_pages = _count_pdf_pages(pdfs[0])
    assert n_pages >= 3, (
        f"scratch-layout PDF has only {n_pages} pages — behavioral pages "
        "likely skipped due to empty mouse_id_list (tracking-H5 lookup "
        "not finding the session-root H5)."
    )


# ---------------------------------------------------------------------------
# Per-bug rendering regressions: each test exercises a single private
# rendering method with a hand-crafted payload that surfaces exactly one
# class of "axis bound vs rendered data extent" mismatch, then inspects
# the resulting matplotlib axes for the regression. These tests are
# fast (no compute, no PDF I/O) and pinpoint the offending code path
# precisely when a fix regresses.
# ---------------------------------------------------------------------------


def _make_figure_maker() -> NeuronalTuningFigureMaker:
    """
    Description
    -----------
    Construct a `NeuronalTuningFigureMaker` for unit-testing private
    rendering methods. Only the attributes those methods touch
    (`visualizations_parameter_dict`, `vocal_labels` inherited from
    `FeatureZoo`, `_segmentation_cache`, etc.) need to be populated;
    `root_directory` is a placeholder since no pkls are loaded.

    Parameters
    ----------

    Returns
    -------
    maker (NeuronalTuningFigureMaker)
        Maker instance with `_make_visualizations_parameters()` settings.
    """

    return NeuronalTuningFigureMaker(
        root_directory="/tmp",
        visualizations_parameter_dict=_make_visualizations_parameters(),
        message_output=lambda *a, **k: None,
    )


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_property_cell_xlim_matches_finite_data_extent_not_occupancy():
    """
    Description
    -----------
    Regression for bug #1: an outlier bin at the far-right of the
    property axis with `occ > 0` but `rate=NaN` (occupancy below
    `usv_property_min_occupancy_seconds`, which the compute side uses
    to NaN out the rate) used to drag the x-axis tick out to that
    bin's center. The visible rate / shuffle band stopped half the
    axis earlier, producing a wide empty gap between the rendered
    data and the right spine. Filter the visible extent by finite
    rate / shuffle bounds instead.

    The hand-crafted payload mirrors the real cluster:
      - 36 mean_freq_hz bins spanning 30–120 kHz
      - finite rate / shuffle at indices 5..20 (kHz centers 43.75–81.25)
      - outlier bin at index 31 (kHz center 108.75) with occ > 0 but
        rate=NaN — the bin the OLD code would have followed

    Parameters
    ----------

    Returns
    -------
    None
    """

    n_bins = 36
    edges = np.linspace(30000.0, 120000.0, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    rate = np.full(n_bins, np.nan)
    rate[5:21] = 9.0
    p0_5 = np.full(n_bins, np.nan); p0_5[5:21] = 2.0
    p99_5 = np.full(n_bins, np.nan); p99_5[5:21] = 16.0
    occ = np.zeros(n_bins)
    occ[5:21] = 5.0
    occ[31] = 0.12  # outlier: positive but below min-occ -> rate stays NaN

    payload = {
        "bin_centers":       centers,
        "bin_edges":         edges,
        "rate":              rate,
        "occupancy_seconds": occ,
        "null_p0_5":         p0_5,
        "null_p99_5":        p99_5,
    }

    maker = _make_figure_maker()
    fig, (ax_line, ax_occ) = plt.subplots(1, 2)
    try:
        maker._draw_property_pair(
            ax_line=ax_line,
            ax_occ=ax_occ,
            prop="mean_freq_hz",
            cluster_payload=payload,
            line_color="#202020",
        )

        # Display factor for mean_freq_hz is Hz -> kHz.
        f = DISPLAY_FACTOR["mean_freq_hz"]
        last_finite_kHz = float(centers[20] * f)   # 81.25
        outlier_kHz = float(centers[31] * f)       # 108.75

        x_hi = ax_line.get_xlim()[1]
        assert x_hi >= last_finite_kHz, (
            f"x-axis upper bound {x_hi:.2f} kHz cuts off before the last "
            f"finite-rate bin at {last_finite_kHz:.2f} kHz."
        )
        midpoint = 0.5 * (last_finite_kHz + outlier_kHz)
        assert x_hi < midpoint, (
            f"x-axis upper bound {x_hi:.2f} kHz extends past the last "
            f"finite-rate bin ({last_finite_kHz:.2f}) toward the outlier-"
            f"occupancy bin at {outlier_kHz:.2f} kHz — the visible-extent "
            "filter is matching `occ > 0` instead of `isfinite(rate)`."
        )
    finally:
        plt.close(fig)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_property_cell_clamps_negative_shuffle_floor_at_zero():
    """
    Description
    -----------
    Regression for bug #2: the rendered shuffle band must never reach
    below zero firing rate, even if upstream numerics ever produce a
    tiny negative value. Inject `null_p0_5 = -0.1` everywhere and
    inspect the resulting `fill_between` polygon vertices.

    Parameters
    ----------

    Returns
    -------
    None
    """

    n_bins = 12
    edges = np.linspace(30000.0, 120000.0, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    rate = np.full(n_bins, 5.0)
    p0_5 = np.full(n_bins, -0.1)
    p99_5 = np.full(n_bins, 10.0)
    occ = np.full(n_bins, 1.0)

    payload = {
        "bin_centers":       centers,
        "bin_edges":         edges,
        "rate":              rate,
        "occupancy_seconds": occ,
        "null_p0_5":         p0_5,
        "null_p99_5":        p99_5,
    }

    maker = _make_figure_maker()
    fig, (ax_line, ax_occ) = plt.subplots(1, 2)
    try:
        maker._draw_property_pair(
            ax_line=ax_line,
            ax_occ=ax_occ,
            prop="mean_freq_hz",
            cluster_payload=payload,
            line_color="#202020",
        )

        polys = [
            c for c in ax_line.collections
            if isinstance(c, matplotlib.collections.PolyCollection)
        ]
        assert polys, (
            "no PolyCollection from fill_between in the line axes — "
            "rendering changed shape, recheck which artist holds the band"
        )
        min_y = min(
            float(path.vertices[:, 1].min())
            for poly in polys
            for path in poly.get_paths()
        )
        assert min_y >= 0.0, (
            f"rendered shuffle band reaches y={min_y:.4f} < 0; the "
            "defensive `p0_5 = max(p0_5, 0)` clamp is missing or broken."
        )
    finally:
        plt.close(fig)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_categorical_strip_symlog_xlim_uses_geometric_buffer():
    """
    Description
    -----------
    Regression for bug #3: on a symlog x-scale, the old code applied a
    LINEAR 5%-of-(hi-lo) buffer below `x_lo_data`. With small positive
    `x_lo_data` and a wide dynamic range, that buffer was orders of
    magnitude wider than the leftmost data point itself, producing a
    huge visible gap. The fix uses a MULTIPLICATIVE (geometric) buffer
    when the data is all positive.

    Crafts a payload with `max/min(positive) >> log_threshold` so the
    strip plot is forced into symlog, then asserts:
      * x-scale is symlog
      * the lower xlim stays above the old-linear-buffer floor
      * the lower xlim is within a factor of 2 of `x_lo_data` (the
        leftmost rendered point) — log-space proportionality

    Parameters
    ----------

    Returns
    -------
    None
    """

    cats  = np.arange(8)
    rate  = np.array([ 5.0,  8.0, 30.0, 45.0, 60.0,  70.0,  80.0, 100.0])
    p0_5  = np.array([ 0.5,  0.7,  2.0,  3.0,  4.0,   5.0,   6.0,  10.0])
    p99_5 = np.array([20.0, 30.0, 50.0, 70.0, 80.0,  90.0, 100.0, 120.0])

    payload = {
        "categories": cats,
        "rate":       rate,
        "null_p0_5":  p0_5,
        "null_p99_5": p99_5,
    }

    maker = _make_figure_maker()
    fig, ax = plt.subplots()
    try:
        maker._draw_categorical_strip(
            ax=ax,
            payload=payload,
            dot_color="#202020",
            log_threshold=10.0,
            symlog_linthresh=0.5,
        )

        assert ax.get_xscale() == "symlog", (
            "expected symlog scale given max(120) / min_positive(0.5) "
            "ratio = 240 >> log_threshold 10"
        )
        x_lo_lim, x_hi_lim = ax.get_xlim()
        x_lo_data = float(p0_5.min())   # 0.5
        x_hi_data = float(p99_5.max())  # 120.0
        old_linear_lower = x_lo_data - 0.05 * (x_hi_data - x_lo_data)
        assert x_lo_lim > old_linear_lower, (
            f"x_lo_lim={x_lo_lim:.3f} is at or below the old-linear-buffer "
            f"floor ({old_linear_lower:.3f}); the symlog branch is not "
            "taking the geometric path."
        )
        assert x_lo_lim > 0.0, (
            f"x_lo_lim={x_lo_lim:.3f} crossed zero — geometric buffer "
            "should keep the lower bound positive when the data is positive."
        )
        assert x_lo_lim < x_lo_data, (
            f"x_lo_lim={x_lo_lim:.3f} >= x_lo_data={x_lo_data:.3f}; some "
            "buffer below the leftmost data point is expected."
        )
        assert x_lo_lim > x_lo_data * 0.5, (
            f"x_lo_lim={x_lo_lim:.3f} is more than 2x below "
            f"x_lo_data={x_lo_data:.3f} — the buffer is still linearly "
            "scaled in symlog space (the original bug)."
        )
    finally:
        plt.close(fig)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_category_peth_cell_curve_reaches_anchor_t_zero():
    """
    Description
    -----------
    Regression for bug #4: the per-category PETH cell anchors the
    right x-edge at t=0 but `bin_centers_s[-1]` is half-a-bin-width
    left of zero (e.g. -0.025 s with 50 ms bins in a [-2, 0] window).
    The OLD code plotted `ax.plot(bin_centers, rate)` directly, so the
    rendered curve stopped at the rightmost bin center, leaving a
    visible air gap between the curve and the t=0 tick. The fix
    appends a synthetic anchor point `(0.0, rate[last_finite])` to
    the curve and the shuffle band.

    Parameters
    ----------

    Returns
    -------
    None
    """

    bin_centers = np.arange(-1.975, 0.0, 0.05)
    assert abs(bin_centers[-1] - (-0.025)) < 1e-9
    n_bins = bin_centers.size

    cats = np.array([1])
    rate = np.full((1, n_bins), np.nan)
    rate[0, -10:] = 8.0
    p0_5 = np.full((1, n_bins), np.nan)
    p0_5[0, -10:] = 2.0
    p99_5 = np.full((1, n_bins), np.nan)
    p99_5[0, -10:] = 16.0

    payload_for_cat = {
        "categories":     cats,
        "bin_centers_s":  bin_centers,
        "rate":           rate,
        "null_p0_5":      p0_5,
        "null_p99_5":     p99_5,
        "sex":            "male",
    }
    # `_draw_section_d` iterates every CATEGORICAL_FEATURES entry; provide
    # an empty payload for the others so the lookup doesn't KeyError.
    empty_payload = {
        "categories":     np.array([], dtype=int),
        "bin_centers_s":  bin_centers,
        "rate":           np.zeros((0, n_bins), dtype=float),
        "null_p0_5":      np.zeros((0, n_bins), dtype=float),
        "null_p99_5":     np.zeros((0, n_bins), dtype=float),
        "sex":            "male",
    }
    cluster_data = {
        "usv_category_peth": {
            "m1": {
                cf: (payload_for_cat if cf == CATEGORICAL_FEATURES[0]
                     else empty_payload)
                for cf in CATEGORICAL_FEATURES
            }
        }
    }

    maker = _make_figure_maker()
    fig = plt.figure()
    gs = fig.add_gridspec(1, 1)
    try:
        maker._draw_section_d(
            fig=fig, gs=gs, emitter="m1",
            cluster_data=cluster_data, n_cols=1,
        )

        axes = fig.axes
        assert len(axes) == 1, (
            f"expected 1 rendered cell (one finite category), got {len(axes)}"
        )
        lines = axes[0].get_lines()
        assert lines, "no Line2D found in the rendered cell"
        rate_xdata = lines[0].get_xdata()
        assert abs(float(rate_xdata[-1]) - 0.0) < 1e-9, (
            f"PETH curve right edge is at x={float(rate_xdata[-1]):.4f}, "
            "not at the t=0 anchor — the line-extension fix is missing "
            "or broken (the curve stops half-a-binwidth left of zero)."
        )
    finally:
        plt.close(fig)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_categorical_strip_has_exactly_two_xticks():
    """
    Description
    -----------
    The "tuning vs shuffled" categorical strip is a small panel; the
    matplotlib auto-ticker (especially under symlog) places enough
    ticks that their labels overlap one another. The renderer now
    forces exactly two major ticks — one at `x_lo_data` and one at
    `x_hi_data` — and suppresses any minor-tick labels.

    Uses the same wide-dynamic-range payload as the symlog buffer
    test so we exercise the more aggressive auto-ticker.

    Parameters
    ----------

    Returns
    -------
    None
    """

    cats  = np.arange(8)
    rate  = np.array([ 5.0,  8.0, 30.0, 45.0, 60.0,  70.0,  80.0, 100.0])
    p0_5  = np.array([ 0.5,  0.7,  2.0,  3.0,  4.0,   5.0,   6.0,  10.0])
    p99_5 = np.array([20.0, 30.0, 50.0, 70.0, 80.0,  90.0, 100.0, 120.0])

    payload = {
        "categories": cats,
        "rate":       rate,
        "null_p0_5":  p0_5,
        "null_p99_5": p99_5,
    }

    maker = _make_figure_maker()
    fig, ax = plt.subplots()
    try:
        maker._draw_categorical_strip(
            ax=ax,
            payload=payload,
            dot_color="#202020",
            log_threshold=10.0,
            symlog_linthresh=0.5,
        )

        major_ticks = ax.get_xticks()
        assert len(major_ticks) == 2, (
            f"expected exactly 2 x-ticks on the categorical strip, got "
            f"{len(major_ticks)} ({major_ticks}); the auto-ticker is "
            "back and will overlap labels on small panels."
        )
        # Labels should match the data extrema (with .3g formatting).
        tick_labels = [t.get_text() for t in ax.get_xticklabels()]
        assert tick_labels == [
            f"{float(p0_5.min()):.3g}",
            f"{float(p99_5.max()):.3g}",
        ], (
            f"expected labels at data extrema, got {tick_labels}"
        )
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Full pkl contract: walks the ENTIRE payload structure
# `NeuronalTuning.calculate_neuronal_tuning_curves` writes per cluster and
# asserts every expected key and shape consistency at every level. This
# is the safety net for the renaming-class bug (compute writes `usv_*`
# but figure code looked for `vocal_q*`) at every nested layer.
# ---------------------------------------------------------------------------


# 1D behavioral feature cell stat keys (smoothing_sd == 0 path: no
# `_smoothed` variants). 2D spatial features use a smaller set because
# spatial Skaggs info / coherence are evaluated via triage_stats.
_BEH_1D_STAT_KEYS = frozenset({
    "bin_centers", "bin_edges", "rate", "occupancy_seconds",
    "null_mean", "null_std",
    "null_p0_5", "null_p2_5", "null_p97_5", "null_p99_5",
})
_BEH_2D_STAT_KEYS = frozenset({
    "bin_centers", "bin_edges", "rate", "occupancy_seconds",
})

# Vocal block stat keys (smoothing_sd == 0 path).
_USV_PETH_KEYS = frozenset({
    "bin_centers_s", "denom_seconds", "n_anchors", "rate",
    "null_mean", "null_std",
    "null_p0_5", "null_p2_5", "null_p97_5", "null_p99_5",
    "role", "sex",
})
_USV_PROPERTY_TUNING_KEYS = frozenset({
    "bin_centers", "bin_edges", "rate", "occupancy_seconds",
    "null_mean", "null_std",
    "null_p0_5", "null_p2_5", "null_p97_5", "null_p99_5",
    "role", "sex",
})
_USV_CATEGORY_TUNING_KEYS = frozenset({
    "categories", "rate", "occupancy_seconds", "occupancy_count",
    "null_mean", "null_std",
    "null_p0_5", "null_p2_5", "null_p97_5", "null_p99_5",
    "role", "sex",
})
_USV_CATEGORY_PETH_KEYS = frozenset({
    "categories", "bin_centers_s", "rate", "denom_seconds",
    "null_mean", "null_std",
    "null_p0_5", "null_p2_5", "null_p97_5", "null_p99_5",
    "role", "sex",
})

# Top-level cluster-pkl keys after a clean compute pass.
_TOP_LEVEL_KEYS = frozenset({
    "beh_offset=0s",
    "behavioral_metadata",
    "triage_stats",
    "usv_peth",
    "usv_property_tuning",
    "usv_category_tuning",
    "usv_category_peth",
    "usv_metadata",
})

# Scalar/list keys recorded in each metadata block.
_BEHAVIORAL_METADATA_KEYS = frozenset({
    "behavioral_min_occupancy_seconds",
    "cluster_id",
    "empirical_camera_sr",
    "generated_at",
    "n_shuffles",
    "n_spatial_bins",
    "session_root",
    "smoothing_sd",
    "spatial_scale_cm",
    "temporal_offsets",
    "total_bin_num",
})
_USV_METADATA_KEYS = frozenset({
    "bout_quiet_seconds",
    "cluster_id",
    "duration_seconds",
    "generated_at",
    "n_shuffles",
    "n_usv_min_category",
    "peth_bin_seconds",
    "peth_window_seconds",
    "session_root",
    "shuffle_seconds_range",
    "smoothing_sd",
    "usv_property_min_occupancy_seconds",
    "vocal_require_clean_post_anchor",
    "vocal_require_clean_prior_anchor",
})
_TRIAGE_STATS_KEYS = frozenset({
    "behavioral", "spatial", "vmi",
    "usv_peth", "usv_property_tuning",
    "usv_category_tuning", "usv_category_peth",
})


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_pkl_contract_includes_entire_expected_payload_structure(tmp_path):
    """
    Description
    -----------
    Run a full synthetic-session compute end-to-end and walk every
    level of the resulting cluster pkl, asserting:

    * top-level keys exactly match the expected set
    * `beh_offset=Ns` block: one entry per session feature with the
      expected 1D / 2D stat keys; shapes consistent across rate +
      bin_centers + null_* arrays
    * `behavioral_metadata` carries the named scalar keys
    * `usv_peth[emitter]` carries the named stat keys with rate /
      bin_centers_s / null_* shape consistency
    * `usv_property_tuning[emitter][prop]` exists for every property
      in `CONTINUOUS_PROPERTIES`, with the named stat keys and shape
      consistency
    * `usv_category_tuning[emitter][cat_feat]` and
      `usv_category_peth[emitter][cat_feat]` exist for every entry in
      `CATEGORICAL_FEATURES`, each with the named stat keys
    * `usv_metadata` carries the named scalar keys
    * `triage_stats` carries every per-modality summary key

    This is the contract test that would have caught the renaming
    regression at the source (compute side) — if a future refactor
    renames a key, drops a key, or breaks a shape invariant, this
    test fails before the figure side gets a chance to silently skip
    rendering.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Pytest-provided temp directory.

    Returns
    -------
    None
    """

    root = tmp_path / "session"
    root.mkdir()
    _build_synthetic_figure_session(root, tracking_h5_at_session_root=False)

    nt = NeuronalTuning(
        root_directory=str(root),
        tuning_parameters_dict=_make_tuning_parameters(),
        message_output=lambda *a, **k: None,
    )
    nt.calculate_neuronal_tuning_curves()

    pkls = sorted((root / "ephys" / "tuning_curves").glob("*_tuning_curves_data.pkl"))
    assert len(pkls) == 1, f"expected 1 pkl, got {len(pkls)}"
    with pkls[0].open("rb") as fh:
        d = pickle.load(fh)

    # ---- top level ------------------------------------------------------
    assert set(d.keys()) == _TOP_LEVEL_KEYS, (
        f"top-level keys mismatch.\n"
        f"  expected: {sorted(_TOP_LEVEL_KEYS)}\n"
        f"  got:      {sorted(d.keys())}\n"
        f"  missing:  {sorted(_TOP_LEVEL_KEYS - set(d.keys()))}\n"
        f"  extra:    {sorted(set(d.keys()) - _TOP_LEVEL_KEYS)}"
    )

    # ---- beh_offset=0s --------------------------------------------------
    beh = d["beh_offset=0s"]
    assert isinstance(beh, dict) and beh, "beh_offset=0s is empty"
    for feat_name, feat_payload in beh.items():
        is_2d = feat_name.endswith(".space")
        expected = _BEH_2D_STAT_KEYS if is_2d else _BEH_1D_STAT_KEYS
        got = set(feat_payload.keys())
        missing = expected - got
        assert not missing, (
            f"beh_offset=0s[{feat_name}] missing keys: {sorted(missing)}; "
            f"got: {sorted(got)}"
        )
        # shape consistency: rate / bin_centers / null_* same length for 1D
        if not is_2d:
            n = len(feat_payload["bin_centers"])
            for k in ("rate", "null_p0_5", "null_p99_5", "occupancy_seconds"):
                assert len(feat_payload[k]) == n, (
                    f"beh_offset=0s[{feat_name}][{k}] shape={len(feat_payload[k])} "
                    f"!= bin_centers shape={n}"
                )

    # ---- behavioral_metadata -------------------------------------------
    bm = d["behavioral_metadata"]
    missing = _BEHAVIORAL_METADATA_KEYS - set(bm.keys())
    assert not missing, f"behavioral_metadata missing keys: {sorted(missing)}"

    # ---- usv_peth[emitter] ---------------------------------------------
    usv_peth = d["usv_peth"]
    assert isinstance(usv_peth, dict) and usv_peth, "usv_peth is empty"
    for emitter, payload in usv_peth.items():
        missing = _USV_PETH_KEYS - set(payload.keys())
        assert not missing, (
            f"usv_peth[{emitter}] missing keys: {sorted(missing)}"
        )
        n_bins = len(payload["bin_centers_s"])
        for k in ("rate", "null_p0_5", "null_p99_5", "denom_seconds"):
            assert len(payload[k]) == n_bins, (
                f"usv_peth[{emitter}][{k}] shape mismatch with bin_centers_s"
            )

    # ---- usv_property_tuning[emitter][prop] ----------------------------
    upt = d["usv_property_tuning"]
    for emitter, by_prop in upt.items():
        assert set(by_prop.keys()) == set(CONTINUOUS_PROPERTIES), (
            f"usv_property_tuning[{emitter}] property set differs from "
            f"CONTINUOUS_PROPERTIES.\n"
            f"  expected: {sorted(CONTINUOUS_PROPERTIES)}\n"
            f"  got:      {sorted(by_prop.keys())}"
        )
        for prop, payload in by_prop.items():
            missing = _USV_PROPERTY_TUNING_KEYS - set(payload.keys())
            assert not missing, (
                f"usv_property_tuning[{emitter}][{prop}] missing keys: "
                f"{sorted(missing)}"
            )
            n = len(payload["bin_centers"])
            for k in ("rate", "null_p0_5", "null_p99_5", "occupancy_seconds"):
                assert len(payload[k]) == n, (
                    f"usv_property_tuning[{emitter}][{prop}][{k}] shape "
                    f"mismatch with bin_centers"
                )

    # ---- usv_category_tuning[emitter][cat_feat] ------------------------
    uct = d["usv_category_tuning"]
    for emitter, by_cat in uct.items():
        assert set(by_cat.keys()) == set(CATEGORICAL_FEATURES), (
            f"usv_category_tuning[{emitter}] cat set differs from "
            f"CATEGORICAL_FEATURES.\n"
            f"  expected: {sorted(CATEGORICAL_FEATURES)}\n"
            f"  got:      {sorted(by_cat.keys())}"
        )
        for cat_feat, payload in by_cat.items():
            missing = _USV_CATEGORY_TUNING_KEYS - set(payload.keys())
            assert not missing, (
                f"usv_category_tuning[{emitter}][{cat_feat}] missing "
                f"keys: {sorted(missing)}"
            )
            n_cats = len(payload["categories"])
            for k in ("rate", "null_p0_5", "null_p99_5",
                      "occupancy_seconds", "occupancy_count"):
                assert len(payload[k]) == n_cats, (
                    f"usv_category_tuning[{emitter}][{cat_feat}][{k}] "
                    f"shape mismatch with categories"
                )

    # ---- usv_category_peth[emitter][cat_feat] --------------------------
    ucp = d["usv_category_peth"]
    for emitter, by_cat in ucp.items():
        assert set(by_cat.keys()) == set(CATEGORICAL_FEATURES), (
            f"usv_category_peth[{emitter}] cat set differs from "
            f"CATEGORICAL_FEATURES"
        )
        for cat_feat, payload in by_cat.items():
            missing = _USV_CATEGORY_PETH_KEYS - set(payload.keys())
            assert not missing, (
                f"usv_category_peth[{emitter}][{cat_feat}] missing "
                f"keys: {sorted(missing)}"
            )
            n_cats = len(payload["categories"])
            n_bins = len(payload["bin_centers_s"])
            for k in ("rate", "null_p0_5", "null_p99_5"):
                arr = np.asarray(payload[k])
                assert arr.shape == (n_cats, n_bins), (
                    f"usv_category_peth[{emitter}][{cat_feat}][{k}] "
                    f"shape={arr.shape} != (n_cats={n_cats}, n_bins={n_bins})"
                )

    # ---- usv_metadata ---------------------------------------------------
    um = d["usv_metadata"]
    missing = _USV_METADATA_KEYS - set(um.keys())
    assert not missing, f"usv_metadata missing keys: {sorted(missing)}"

    # ---- triage_stats ---------------------------------------------------
    ts = d["triage_stats"]
    missing = _TRIAGE_STATS_KEYS - set(ts.keys())
    assert not missing, (
        f"triage_stats missing top-level keys: {sorted(missing)}"
    )
    # behavioral / spatial keyed by `beh_offset=Ns`
    for section in ("behavioral", "spatial"):
        assert "beh_offset=0s" in ts[section], (
            f"triage_stats[{section}] missing 'beh_offset=0s' entry"
        )
    # vmi / usv_* keyed by emitter id
    for section in ("vmi", "usv_peth", "usv_property_tuning",
                    "usv_category_tuning", "usv_category_peth"):
        assert ts[section], (
            f"triage_stats[{section}] is empty — expected at least one "
            f"emitter entry"
        )
