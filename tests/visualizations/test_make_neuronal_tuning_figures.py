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
    USV_CATEGORY_SEGMENTATIONS,
    USV_PROPERTY_ORDER,
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


# ---------------------------------------------------------------------------
# Population-level "aggregate" figures: every `make_*_figure(s)` method that
# consumes a cross-session `unit_triage_*.pkl` (produced by
# `unit_triage_aggregator.py`) plus the companion `unit_catalog.csv`. These
# methods walk the triage pickle's `units` structure, classify units per
# brain-area group / modality / consistency rule, and render one or more
# population summary figures.
#
# Rather than running the full compute+aggregator chain (slow, and whose
# random synthetic stats rarely trip the "significant" branches), we
# hand-build a triage pickle + catalog whose `units` deliberately populate
# every interesting branch: significant-positive / significant-negative /
# sign-flipping / non-significant VMI units, units with consistent PETH /
# property / category peaks, and units spanning the pose / movement / social
# behavioral buckets — across all seven canonical region groups and both the
# `intact_female` / `mute_female` conditions. The schema mirrors exactly what
# `aggregate_units_across_conditions` writes (per-modality `per_session` lists
# AND the top-level `n_significant` / `n_tested` counts the behavioral / vocal
# flag helpers read).
# ---------------------------------------------------------------------------

_COND_A = "intact_female"
_COND_B = "mute_female"
_TRIAGE_ALPHA = 0.05
_TRIAGE_MIN_BOUTS = 8

# anatomy_region -> canonical region group (one raw region per group so the
# fixture exercises the region_to_group lookup for each VMI_REGION_ORDER label;
# "FOO" is deliberately unknown so it lands in the "Other" catch-all).
_REGION_ANATOMY = {
    "PAG":   "PAG",
    "MRN":   "MRN",
    "VTA":   "VTA",
    "MB":    "MB",
    "CENT":  "CENT2",
    "SC":    "SCdw",
    "Other": "FOO",
}


def _vmi_session(session, vmi, p, *, n_bouts=40, fr_baseline=2.5, fr_usv=4.0):
    """
    Description
    -----------
    Build one per-session VMI entry exactly as the triage aggregator
    records it under a `vmi_self_{excit,suppress}` modality block:
    `session`, `n_bouts`, `p`, `vmi`, `fr_baseline`, `fr_usv`,
    `significant`. Significance is derived from `p < _TRIAGE_ALPHA` so
    callers can drive the sig / non-sig branches purely by the supplied
    p-value.

    Parameters
    ----------
    session (str)
        Session id string stored on the entry.
    vmi (float | None)
        Signed vocalization-modulation index for the session.
    p (float | None)
        Per-session two-sided p-value; `< _TRIAGE_ALPHA` marks the
        entry significant.
    n_bouts (int)
        USV-bout count; entries below `_TRIAGE_MIN_BOUTS` are filtered
        out by every VMI collector.
    fr_baseline (float)
        Pre-bout baseline firing rate (sp/s).
    fr_usv (float)
        During-bout firing rate (sp/s).

    Returns
    -------
    entry (dict)
        One per-session VMI record.
    """

    return {
        "session":     session,
        "n_bouts":     n_bouts,
        "p":           p,
        "vmi":         vmi,
        "fr_baseline": fr_baseline,
        "fr_usv":      fr_usv,
        "significant": (p is not None and p < _TRIAGE_ALPHA),
    }


def _mod_block(per_session):
    """
    Description
    -----------
    Wrap a `per_session` list into a full modality block, computing the
    top-level `n_tested` / `n_significant` / `consistency` scalars the
    aggregator derives in its post-pass (and which the behavioral /
    vocal flag helpers read directly off the block). An empty
    `aggregate` dict is attached for schema parity.

    Parameters
    ----------
    per_session (list[dict])
        Per-session entries (any modality shape).

    Returns
    -------
    block (dict)
        Modality block with `per_session`, `n_tested`,
        `n_significant`, `consistency`, `aggregate`.
    """

    n_test = len(per_session)
    n_sig = sum(1 for e in per_session if e.get("significant"))
    return {
        "per_session":   per_session,
        "n_tested":      n_test,
        "n_significant": n_sig,
        "consistency":   (n_sig / n_test) if n_test else 0.0,
        "aggregate":     {},
    }


def _behavioral_block(n_sig, n_tested):
    """
    Description
    -----------
    Build a behavioral-modality block as the flag helpers see it: only
    the top-level `n_significant` / `n_tested` counts are read (the
    consistency rule never touches a `per_session` list for behavioral
    keys), so the block carries exactly those two scalars.

    Parameters
    ----------
    n_sig (int)
        Number of significant sessions for the feature.
    n_tested (int)
        Number of tested sessions for the feature.

    Returns
    -------
    block (dict)
        `{"n_significant": int, "n_tested": int}`.
    """

    return {"n_significant": int(n_sig), "n_tested": int(n_tested)}


def _vocal_modalities(archetype):
    """
    Description
    -----------
    Build the full set of self-vocal modality blocks for one condition,
    parameterised by `archetype`. The archetype drives the VMI sign /
    significance pattern and whether the unit carries consistent PETH /
    property / category peaks:

      * "pos"    — two significant +VMI sessions; consistent excit PETH,
                   consistent property peaks for every property, and
                   consistent up-category peaks for every segmentation.
      * "neg"    — two significant -VMI suppress sessions; no consistent
                   vocal-peak content (exercises the sig-negative branch).
      * "flip"   — one significant +VMI and one significant -VMI session
                   (the cross-session sign-flipper).
      * "nonsig" — two non-significant VMI sessions only.

    Parameters
    ----------
    archetype (str)
        One of {"pos", "neg", "flip", "nonsig"}.

    Returns
    -------
    mods (dict)
        Mapping from modality key to modality block.
    """

    mods: dict = {}

    if archetype == "pos":
        mods["vmi_self_excit"] = _mod_block([
            _vmi_session("s1", 0.62, 0.005),
            _vmi_session("s2", 0.55, 0.02),
        ])
    elif archetype == "neg":
        mods["vmi_self_suppress"] = _mod_block([
            _vmi_session("s1", -0.60, 0.004),
            _vmi_session("s2", -0.52, 0.03),
        ])
    elif archetype == "flip":
        mods["vmi_self_excit"] = _mod_block([_vmi_session("s1", 0.65, 0.004)])
        mods["vmi_self_suppress"] = _mod_block([_vmi_session("s2", -0.70, 0.004)])
    else:  # nonsig
        mods["vmi_self_excit"] = _mod_block([
            _vmi_session("s1", 0.12, 0.40),
            _vmi_session("s2", 0.08, 0.55),
        ])

    if archetype != "pos":
        return mods

    # Consistent excit PETH: two sig sessions whose peak_t cluster within
    # the 100 ms tolerance, plus a sig suppress block for the suppress
    # figure variants.
    mods["usv_peth_self_excit"] = _mod_block([
        {"session": "s1", "significant": True, "peak_t": -0.50, "peak_z": 4.2},
        {"session": "s2", "significant": True, "peak_t": -0.52, "peak_z": 4.6},
    ])
    mods["usv_peth_self_suppress"] = _mod_block([
        {"session": "s1", "significant": True, "peak_t": -0.30, "peak_z": -4.1},
        {"session": "s2", "significant": True, "peak_t": -0.31, "peak_z": -4.4},
    ])

    # Consistent property peaks for every property: two sig sessions with
    # peak_bin_value within the per-property tolerance.
    _prop_base = {
        "duration":          (0.05, 0.06),
        "mean_freq_hz":      (60000.0, 61000.0),
        "peak_freq_hz":      (62000.0, 63000.0),
        "freq_bandwidth_hz": (15000.0, 16000.0),
        "mean_amplitude":    (1.0, 1.1),
        "max_amplitude":     (2.0, 2.3),
        "spectral_entropy":  (2.0, 2.1),
        "mask_number":       (3.0, 4.0),
    }
    for prop in USV_PROPERTY_ORDER:
        v1, v2 = _prop_base[prop]
        mods[f"usv_property_self_{prop}_excit"] = _mod_block([
            {"session": "s1", "significant": True,
             "peak_bin_value": v1, "peak_z": 4.0},
            {"session": "s2", "significant": True,
             "peak_bin_value": v2, "peak_z": 4.5},
        ])

    # Consistent up-category peaks for every segmentation: two sig
    # sessions agreeing on best_cat=2 with positive signed z.
    for seg in USV_CATEGORY_SEGMENTATIONS:
        mods[f"usv_category_self_{seg}"] = _mod_block([
            {"session": "s1", "significant": True, "peak_signed_z": 3.1,
             "best_cat": 2, "n_sig_categories": 2, "selectivity": 0.62},
            {"session": "s2", "significant": True, "peak_signed_z": 3.6,
             "best_cat": 2, "n_sig_categories": 3, "selectivity": 0.71},
        ])

    return mods


def _behavioral_modalities(mouse_id, partner_id, buckets):
    """
    Description
    -----------
    Build the behavioral modality keys for one condition, in the exact
    `behavioral_beh_offset=0s_<prefix>.<feat>_<direction>` shape the
    aggregator emits. `buckets` selects which of the pose / movement /
    social buckets are "tuned" (consistent in >= k_min sessions with a
    strict majority) so the caller can place a unit in any of the eight
    behavioral tiers.

    Parameters
    ----------
    mouse_id (str)
        Recorded-mouse id (self-pose / self-movement key prefix).
    partner_id (str)
        Partner mouse id used to form the hyphenated dyadic (social)
        key prefix `<mouse>-<partner>`.
    buckets (set[str])
        Subset of {"pose", "movement", "social"} to flag as tuned.

    Returns
    -------
    mods (dict)
        Mapping from behavioral modality key to its `{n_significant,
        n_tested}` block. Tuned buckets get `2/2`; untuned buckets get
        `0/2` so the feature is present-but-not-tuned.
    """

    def _counts(tuned):
        """Return the ``(n_significant, n_tested)`` pair for a behavioral
        bucket: ``(2, 2)`` when tuned (consistent + majority), else
        ``(0, 2)`` (present but not tuned)."""
        return (2, 2) if tuned else (0, 2)

    mods: dict = {}
    ns, nt = _counts("pose" in buckets)
    mods[f"behavioral_beh_offset=0s_{mouse_id}.allo_yaw_excit"] = _behavioral_block(ns, nt)
    ns, nt = _counts("movement" in buckets)
    mods[f"behavioral_beh_offset=0s_{mouse_id}.speed_excit"] = _behavioral_block(ns, nt)
    ns, nt = _counts("social" in buckets)
    mods[f"behavioral_beh_offset=0s_{mouse_id}-{partner_id}.nose-nose_excit"] = _behavioral_block(ns, nt)
    return mods


def _make_unit(
    *,
    mouse_id,
    rec_date,
    unit_id,
    anatomy,
    archetype,
    both_conditions,
    behavioral_buckets,
    kslabel="good",
):
    """
    Description
    -----------
    Assemble one `triage["units"][...]` entry. The unit carries a full
    self-vocal + behavioral modality set under `_COND_A`, and the same
    set duplicated under `_COND_B` when `both_conditions` is True (so the
    cross-condition stability figure has paired data). Identity fields
    (`mouse_id`, `rec_date`, `unit_id`, `kslabel`, `anatomy_region`)
    mirror the aggregator's per-unit header.

    Parameters
    ----------
    mouse_id (str)
        Recorded-mouse id.
    rec_date (int)
        8-digit recording date (also the catalog join key).
    unit_id (str)
        Per-day Kilosort unit id string.
    anatomy (str)
        Raw `anatomy_region` value (mapped to a canonical group
        downstream).
    archetype (str)
        Vocal archetype passed through to `_vocal_modalities`.
    both_conditions (bool)
        Whether to populate `_COND_B` in addition to `_COND_A`.
    behavioral_buckets (set[str])
        Behavioral buckets to flag tuned (see `_behavioral_modalities`).
    kslabel (str)
        Kilosort label; only `"good"` units survive the collectors'
        filter.

    Returns
    -------
    unit (dict)
        One unit record keyed-ready for `triage["units"]`.
    """

    partner_id = "p9"

    def _cond_block():
        """Build one condition block: the archetype's self-vocal modalities
        merged with the behavioral modalities, under a two-session header."""
        mods = dict(_vocal_modalities(archetype))
        mods.update(_behavioral_modalities(mouse_id, partner_id, behavioral_buckets))
        return {"sessions_tested": ["s1", "s2"], "modalities": mods}

    conditions = {_COND_A: _cond_block()}
    if both_conditions:
        conditions[_COND_B] = _cond_block()

    return {
        "unit_uid":       f"{mouse_id}_{rec_date}_{unit_id}",
        "mouse_id":       mouse_id,
        "rec_date":       rec_date,
        "unit_id":        unit_id,
        "imec":           0,
        "cluster_num":    int(unit_id[-2:]) if unit_id[-2:].isdigit() else 1,
        "peak_channel":   1,
        "kslabel":        kslabel,
        "anatomy_region": anatomy,
        "conditions":     conditions,
    }


def _build_synthetic_triage_pkl(tmp_path):
    """
    Description
    -----------
    Hand-build a `unit_triage_*.pkl` plus its companion
    `unit_catalog.csv` rich enough to drive every aggregate figure
    through its non-trivial branches. Produces, per canonical region
    group, a mix of significant-positive / significant-negative /
    sign-flipping / non-significant VMI units, with the PAG group given
    Allen-CCF coordinates and >= 3 significant units so the anatomical
    KDE path runs. Behavioral buckets are varied so the tier matrix and
    Venn figures populate multiple tiers.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Pytest temp directory; both artifacts are written beneath it.

    Returns
    -------
    triage_path (pathlib.Path)
        Path to the written triage pickle.
    catalog_path (pathlib.Path)
        Path to the written catalog CSV.
    """

    units: dict = {}
    catalog_rows: list[dict] = []

    # Per-region unit recipes: (archetype, both_conditions, behavioral_buckets).
    region_recipes = {
        "PAG": [
            ("pos",   True,  {"pose", "movement", "social"}),
            ("pos",   True,  {"pose", "movement"}),
            ("neg",   True,  {"pose"}),
            ("flip",  True,  {"movement", "social"}),
            ("nonsig", False, set()),
        ],
        "MRN": [
            ("pos",   True,  {"pose", "social"}),
            ("neg",   True,  {"movement"}),
            ("flip",  False, {"social"}),
            ("nonsig", True,  set()),
        ],
        "VTA": [
            ("pos",   True,  {"movement", "social"}),
            ("neg",   False, {"pose", "movement"}),
            ("nonsig", True,  set()),
        ],
        "MB": [
            ("pos",   True,  {"pose"}),
            ("flip",  True,  {"movement"}),
        ],
        "CENT": [
            ("pos",   False, {"social"}),
            ("nonsig", True,  set()),
        ],
        "SC": [
            ("neg",   True,  {"pose", "movement", "social"}),
            ("pos",   True,  set()),
        ],
        "Other": [
            ("pos",   True,  {"pose"}),
            ("nonsig", False, {"movement"}),
        ],
    }

    # PAG Allen-CCF coordinates (µm), spread so gaussian_kde stays
    # non-singular; non-PAG units get blank coords (never read).
    pag_coords = [
        (-4400.0, 300.0, 2400.0),
        (-4450.0, 280.0, 2500.0),
        (-4500.0, 350.0, 2600.0),
        (-4550.0, 260.0, 2450.0),
        (-4600.0, 330.0, 2550.0),
    ]

    date_counter = 20240101
    for region, recipes in region_recipes.items():
        anatomy = _REGION_ANATOMY[region]
        for idx, (archetype, both, buckets) in enumerate(recipes):
            mouse_id = f"m{region}"
            rec_date = date_counter
            date_counter += 1
            unit_id = f"imec0_cl{idx:02d}_ch001_good"

            unit = _make_unit(
                mouse_id=mouse_id,
                rec_date=rec_date,
                unit_id=unit_id,
                anatomy=anatomy,
                archetype=archetype,
                both_conditions=both,
                behavioral_buckets=buckets,
            )
            units[unit["unit_uid"]] = unit

            row = {
                "mouse_id":      mouse_id,
                "rec_date":      str(rec_date),
                "unit_id":       unit_id,
                "brain_area":    anatomy,
                "cluster_group": "good",
                "somatic":       "True",
                "loc_ap":        "",
                "loc_ml":        "",
                "loc_dv":        "",
            }
            if region == "PAG":
                ap, ml, dv = pag_coords[idx % len(pag_coords)]
                row["loc_ap"], row["loc_ml"], row["loc_dv"] = (
                    str(ap), str(ml), str(dv),
                )
            catalog_rows.append(row)

    # One extra non-somatic + one non-good unit so the catalog/kslabel
    # filters in every collector get exercised (both must be skipped).
    skip_specs = [
        ("mSKIP", 20240901, "imec0_cl90_ch001_good", "good", "False"),
        ("mSKIP", 20240902, "imec0_cl91_ch001_mua",  "mua",  "True"),
    ]
    for mouse_id, rec_date, unit_id, kslabel, somatic in skip_specs:
        unit = _make_unit(
            mouse_id=mouse_id,
            rec_date=rec_date,
            unit_id=unit_id,
            anatomy="PAG",
            archetype="pos",
            both_conditions=True,
            behavioral_buckets={"pose"},
            kslabel=kslabel,
        )
        units[unit["unit_uid"]] = unit
        catalog_rows.append({
            "mouse_id":      mouse_id,
            "rec_date":      str(rec_date),
            "unit_id":       unit_id,
            "brain_area":    "PAG",
            "cluster_group": kslabel,
            "somatic":       somatic,
            "loc_ap":        "-4400.0",
            "loc_ml":        "300.0",
            "loc_dv":        "2400.0",
        })

    # Every catalog value is stored as a string (matching how the
    # collectors read the CSV back via `csv.DictReader` and then cast),
    # so a single string-typed polars frame round-trips cleanly — empty
    # `loc_*` fields come back as "" and trip the collectors' float()
    # guard exactly as a real catalog's blank coordinates would.
    catalog_path = tmp_path / "unit_catalog.csv"
    fieldnames = [
        "mouse_id", "rec_date", "unit_id", "brain_area",
        "cluster_group", "somatic", "loc_ap", "loc_ml", "loc_dv",
    ]
    catalog_columns = {
        field: [row[field] for row in catalog_rows] for field in fieldnames
    }
    pls.DataFrame(catalog_columns).write_csv(catalog_path)

    triage = {
        "generated_at":     "2026-06-04T00:00:00",
        "thresholds_used":  {
            "vmi_alpha":     _TRIAGE_ALPHA,
            "vmi_min_bouts": _TRIAGE_MIN_BOUTS,
        },
        "catalog_path":     str(catalog_path),
        "data_root":        str(tmp_path),
        "conditions_included": {_COND_A: [], _COND_B: []},
        "sessions_skipped": {_COND_A: [], _COND_B: []},
        "n_units_total":    len(units),
        "n_units_per_condition": {_COND_A: len(units), _COND_B: len(units)},
        "units":            units,
    }
    triage_path = tmp_path / "unit_triage_20260604_000000.pkl"
    with triage_path.open("wb") as fh:
        pickle.dump(triage, fh)

    return triage_path, catalog_path


def _make_aggregate_visualizations_parameters() -> dict:
    """
    Description
    -----------
    Return a `visualizations_parameter_dict` carrying every palette /
    settings key the aggregate figures read: `brain_area_colors` (with
    the lowercase `other` entry `_resolve_region_colors` maps to),
    `unassigned_colors`, `male_colors`, `female_colors`, `social_colors`,
    and a `figures` block with a `cmap` (read by the category-peak
    figure). Values mirror the project's `visualizations_settings.json`.

    Parameters
    ----------

    Returns
    -------
    viz (dict)
        Visualizations-parameter dict for the aggregate figures.
    """

    return {
        "brain_area_colors": {
            "PAG":   "#677470",
            "MRN":   "#939884",
            "VTA":   "#F5D27A",
            "SC":    "#9FB7D8",
            "CENT":  "#D88080",
            "MB":    "#9BBE85",
            "other": "#B8B8B8",
        },
        "unassigned_colors": ["#C0C0C0"],
        "male_colors":       ["#9AC0CD", "#8CA252"],
        "female_colors":     ["#FF6347", "#B851B4"],
        "social_colors":     ["#5A6470"],
        "figures":           {"cmap": "inferno"},
    }


def _aggregate_maker(tmp_path):
    """
    Description
    -----------
    Construct a `NeuronalTuningFigureMaker` wired with the aggregate
    visualizations palette. `root_directory` is a throwaway path since
    the aggregate methods read the triage pickle / catalog directly.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Pytest temp directory (used as the placeholder root).

    Returns
    -------
    maker (NeuronalTuningFigureMaker)
        Configured figure maker.
    """

    return NeuronalTuningFigureMaker(
        root_directory=str(tmp_path),
        visualizations_parameter_dict=_make_aggregate_visualizations_parameters(),
        message_output=lambda *a, **k: None,
    )


def _assert_figure_written(out_path, tmp_path):
    """
    Description
    -----------
    Assert the figure method returned a real, non-trivially-sized PNG
    that lives under the requested output directory.

    Parameters
    ----------
    out_path (pathlib.Path)
        Path returned by the figure method.
    tmp_path (pathlib.Path)
        The `out_dir` the method was asked to write under.

    Returns
    -------
    None
    """

    out_path = pathlib.Path(out_path)
    assert out_path.exists(), f"figure not written: {out_path}"
    assert out_path.suffix == ".png", f"unexpected format: {out_path.suffix}"
    assert tmp_path in out_path.parents, (
        f"figure {out_path} not under requested out_dir {tmp_path}"
    )
    assert out_path.stat().st_size > 5_000, (
        f"figure {out_path} suspiciously small ({out_path.stat().st_size} bytes)"
    )


@pytest.fixture
def triage_fixture(tmp_path):
    """
    Description
    -----------
    Pytest fixture yielding `(maker, triage_path, catalog_path, out_dir)`
    for the aggregate-figure tests: a hand-built triage pickle + catalog
    under `tmp_path/data`, a configured figure maker, and a dedicated
    `tmp_path/figs` output directory.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Pytest temp directory.

    Returns
    -------
    fixture (tuple)
        `(maker, triage_path, catalog_path, out_dir)`.
    """

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    triage_path, catalog_path = _build_synthetic_triage_pkl(data_dir)
    out_dir = tmp_path / "figs"
    out_dir.mkdir()
    return _aggregate_maker(tmp_path), triage_path, catalog_path, out_dir


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_vmi_fr_confound_figure_writes_png(triage_fixture):
    """
    Description
    -----------
    `make_vmi_fr_confound_figure` renders the FR-baseline × |VMI|
    confound diagnostic (7 per-region scatter panels + a pooled ECDF
    panel) from the triage pickle, writing a real PNG under the
    requested output directory.

    Parameters
    ----------
    triage_fixture (tuple)
        `(maker, triage_path, catalog_path, out_dir)`.

    Returns
    -------
    None
    """

    maker, triage_path, _catalog_path, out_dir = triage_fixture
    out_path = maker.make_vmi_fr_confound_figure(
        triage_pkl_path=triage_path,
        out_dir=out_dir,
        fig_format="png",
    )
    _assert_figure_written(out_path, out_dir)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_vmi_cross_condition_stability_figure_writes_png(triage_fixture):
    """
    Description
    -----------
    `make_vmi_cross_condition_stability_figure` pairs per-unit median
    VMI across `intact_female` / `mute_female` and renders the
    cross-condition stability scatter (with a small bootstrap so the
    test stays fast). Asserts a real PNG is produced.

    Parameters
    ----------
    triage_fixture (tuple)
        `(maker, triage_path, catalog_path, out_dir)`.

    Returns
    -------
    None
    """

    maker, triage_path, _catalog_path, out_dir = triage_fixture
    out_path = maker.make_vmi_cross_condition_stability_figure(
        triage_pkl_path=triage_path,
        n_bootstrap=50,
        out_dir=out_dir,
        fig_format="png",
    )
    _assert_figure_written(out_path, out_dir)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_vmi_magnitude_consistency_figure_writes_png(triage_fixture):
    """
    Description
    -----------
    `make_vmi_magnitude_consistency_figure` plots per-unit max-|VMI|
    against cross-session significance consistency for units tested in
    at least `n_tested_min` sessions. Asserts a real PNG is produced.

    Parameters
    ----------
    triage_fixture (tuple)
        `(maker, triage_path, catalog_path, out_dir)`.

    Returns
    -------
    None
    """

    maker, triage_path, _catalog_path, out_dir = triage_fixture
    out_path = maker.make_vmi_magnitude_consistency_figure(
        triage_pkl_path=triage_path,
        out_dir=out_dir,
        fig_format="png",
    )
    _assert_figure_written(out_path, out_dir)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_vmi_sign_flip_summary_figure_writes_png(triage_fixture):
    """
    Description
    -----------
    `make_vmi_sign_flip_summary_figure` tallies units into
    significant-positive-only / negative-only / both / never tiers
    (the `flip` archetype populates the sig-both tier). Asserts a real
    PNG is produced.

    Parameters
    ----------
    triage_fixture (tuple)
        `(maker, triage_path, catalog_path, out_dir)`.

    Returns
    -------
    None
    """

    maker, triage_path, _catalog_path, out_dir = triage_fixture
    out_path = maker.make_vmi_sign_flip_summary_figure(
        triage_pkl_path=triage_path,
        out_dir=out_dir,
        fig_format="png",
    )
    _assert_figure_written(out_path, out_dir)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_pag_anatomical_gradient_figure_writes_png(triage_fixture):
    """
    Description
    -----------
    `make_pag_anatomical_gradient_figure` projects PAG units' best-
    session signed VMI onto their Allen-CCF coordinates and overlays a
    significant-unit KDE. The fixture's five spread PAG units (>= 3
    significant) keep the gaussian-KDE path non-singular. Asserts a
    real PNG is produced.

    Parameters
    ----------
    triage_fixture (tuple)
        `(maker, triage_path, catalog_path, out_dir)`.

    Returns
    -------
    None
    """

    maker, triage_path, _catalog_path, out_dir = triage_fixture
    out_path = maker.make_pag_anatomical_gradient_figure(
        triage_pkl_path=triage_path,
        kde_grid_resolution=40,
        out_dir=out_dir,
        fig_format="png",
    )
    _assert_figure_written(out_path, out_dir)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_vmi_distribution_figure_writes_png(triage_fixture):
    """
    Description
    -----------
    `make_vmi_distribution_figure` renders per-region signed-VMI
    histograms with sig-positive / sig-negative overlays computed via
    the hybrid per-unit assignment rule. Asserts a real PNG is produced.

    Parameters
    ----------
    triage_fixture (tuple)
        `(maker, triage_path, catalog_path, out_dir)`.

    Returns
    -------
    None
    """

    maker, triage_path, _catalog_path, out_dir = triage_fixture
    out_path = maker.make_vmi_distribution_figure(
        triage_pkl_path=triage_path,
        out_dir=out_dir,
        fig_format="png",
    )
    _assert_figure_written(out_path, out_dir)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("direction", ["excit", "suppress"])
def test_peth_timing_distribution_figure_writes_png(triage_fixture, direction):
    """
    Description
    -----------
    `make_peth_timing_distribution_figure` renders the population
    distribution of consistent per-unit PETH peak times for the given
    direction. The `pos` archetype supplies consistent excit and
    suppress PETH content. Asserts a real PNG is produced for both
    directions.

    Parameters
    ----------
    triage_fixture (tuple)
        `(maker, triage_path, catalog_path, out_dir)`.
    direction (str)
        PETH direction under test ("excit" / "suppress").

    Returns
    -------
    None
    """

    maker, triage_path, _catalog_path, out_dir = triage_fixture
    out_path = maker.make_peth_timing_distribution_figure(
        triage_pkl_path=triage_path,
        direction=direction,
        out_dir=out_dir,
        fig_format="png",
    )
    _assert_figure_written(out_path, out_dir)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_all_property_tuning_distribution_figures_writes_pngs(triage_fixture):
    """
    Description
    -----------
    `make_all_property_tuning_distribution_figures` dispatches one
    property-tuning distribution figure per entry in
    `USV_PROPERTY_ORDER`. Asserts one real PNG per property is written.

    Parameters
    ----------
    triage_fixture (tuple)
        `(maker, triage_path, catalog_path, out_dir)`.

    Returns
    -------
    None
    """

    maker, triage_path, _catalog_path, out_dir = triage_fixture
    out_paths = maker.make_all_property_tuning_distribution_figures(
        triage_pkl_path=triage_path,
        out_dir=out_dir,
        fig_format="png",
    )
    assert len(out_paths) == len(USV_PROPERTY_ORDER), (
        f"expected {len(USV_PROPERTY_ORDER)} property figures, got "
        f"{len(out_paths)}"
    )
    for out_path in out_paths:
        _assert_figure_written(out_path, out_dir)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_all_category_figures_writes_pngs(triage_fixture):
    """
    Description
    -----------
    `make_all_category_figures` dispatches a category-peak-distribution
    figure AND a selectivity-breadth figure per entry in
    `USV_CATEGORY_SEGMENTATIONS` (eight figures total). Asserts every
    returned path is a real PNG.

    Parameters
    ----------
    triage_fixture (tuple)
        `(maker, triage_path, catalog_path, out_dir)`.

    Returns
    -------
    None
    """

    maker, triage_path, _catalog_path, out_dir = triage_fixture
    out_paths = maker.make_all_category_figures(
        triage_pkl_path=triage_path,
        out_dir=out_dir,
        fig_format="png",
    )
    assert len(out_paths) == 2 * len(USV_CATEGORY_SEGMENTATIONS), (
        f"expected {2 * len(USV_CATEGORY_SEGMENTATIONS)} category figures, "
        f"got {len(out_paths)}"
    )
    for out_path in out_paths:
        _assert_figure_written(out_path, out_dir)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_behavioral_tuning_summary_figure_writes_png(triage_fixture):
    """
    Description
    -----------
    `make_behavioral_tuning_summary_figure` classifies each unit into
    one of the eight pose/movement/social tiers and renders the
    region × tier matrix. The fixture's varied behavioral buckets
    populate multiple tiers. Asserts a real PNG is produced.

    Parameters
    ----------
    triage_fixture (tuple)
        `(maker, triage_path, catalog_path, out_dir)`.

    Returns
    -------
    None
    """

    maker, triage_path, _catalog_path, out_dir = triage_fixture
    out_path = maker.make_behavioral_tuning_summary_figure(
        triage_pkl_path=triage_path,
        out_dir=out_dir,
        fig_format="png",
    )
    _assert_figure_written(out_path, out_dir)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_three_set_overlap_venn_figure_writes_png(triage_fixture):
    """
    Description
    -----------
    `make_three_set_overlap_venn_figure` renders the population-level
    behavioral / social / vocal 3-set Venn from the per-unit bucket and
    vocal flags. Asserts a real PNG is produced.

    Parameters
    ----------
    triage_fixture (tuple)
        `(maker, triage_path, catalog_path, out_dir)`.

    Returns
    -------
    None
    """

    maker, triage_path, _catalog_path, out_dir = triage_fixture
    out_path = maker.make_three_set_overlap_venn_figure(
        triage_pkl_path=triage_path,
        out_dir=out_dir,
        fig_format="png",
    )
    _assert_figure_written(out_path, out_dir)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_per_region_overlap_venn_figure_writes_png(triage_fixture):
    """
    Description
    -----------
    `make_per_region_overlap_venn_figure` renders the per-region
    small-multiples Venn grid (one panel per region group plus an
    all-regions aggregate). Asserts a real PNG is produced.

    Parameters
    ----------
    triage_fixture (tuple)
        `(maker, triage_path, catalog_path, out_dir)`.

    Returns
    -------
    None
    """

    maker, triage_path, _catalog_path, out_dir = triage_fixture
    out_path = maker.make_per_region_overlap_venn_figure(
        triage_pkl_path=triage_path,
        out_dir=out_dir,
        fig_format="png",
    )
    _assert_figure_written(out_path, out_dir)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_collectors_skip_non_good_and_non_somatic_units(triage_fixture):
    """
    Description
    -----------
    Contract check on the catalog / kslabel filters shared by every
    collector: the two deliberately-skipped units (one `kslabel='mua'`,
    one `somatic='False'`) must never appear in the population counted
    by `_collect_three_set_overlap_counts`. The eligible count must
    therefore equal the number of good + somatic units in the fixture.

    Parameters
    ----------
    triage_fixture (tuple)
        `(maker, triage_path, catalog_path, out_dir)`.

    Returns
    -------
    None
    """

    maker, triage_path, catalog_path, _out_dir = triage_fixture
    _counts, n_eligible = maker._collect_three_set_overlap_counts(
        triage_pkl_path=triage_path,
        catalog_csv_path=catalog_path,
        condition=_COND_A,
        k_min=2,
        require_majority=True,
    )

    with triage_path.open("rb") as fh:
        triage = pickle.load(fh)
    n_good_somatic = sum(
        1 for u in triage["units"].values()
        if u["kslabel"] == "good" and u["unit_id"] != "imec0_cl90_ch001_good"
    )
    assert n_eligible == n_good_somatic, (
        f"eligible-unit count {n_eligible} should equal the good+somatic "
        f"count {n_good_somatic}; the kslabel/somatic skip filters drifted"
    )


def test_unit_passes_filter(tmp_path):
    """
    Description
    -----------
    Direct contract check on the configurable unit filter
    (`_unit_passes_filter`): the default `("good",)` + `"somatic"`
    keeps only good somatic units, `("good", "mua")` + `"both"` admits
    everything, `"non_somatic"` inverts the somatic test, and an invalid
    `somatic_filter` is rejected at construction.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Pytest temp directory (throwaway root).

    Returns
    -------
    None
    """

    viz = _make_aggregate_visualizations_parameters()

    def _maker(**kw):
        return NeuronalTuningFigureMaker(
            root_directory=str(tmp_path),
            visualizations_parameter_dict=viz,
            message_output=lambda *a, **k: None,
            **kw,
        )

    default = _maker()
    assert default._unit_passes_filter({"kslabel": "good"}, {"somatic": "True"})
    assert not default._unit_passes_filter({"kslabel": "good"}, {"somatic": "False"})
    assert not default._unit_passes_filter({"kslabel": "mua"}, {"somatic": "True"})

    both = _maker(kslabels=("good", "mua"), somatic_filter="both")
    assert both._unit_passes_filter({"kslabel": "mua"}, {"somatic": "False"})
    assert both._unit_passes_filter({"kslabel": "good"}, {"somatic": "True"})

    non_somatic = _maker(somatic_filter="non_somatic")
    assert non_somatic._unit_passes_filter({"kslabel": "good"}, {"somatic": "False"})
    assert not non_somatic._unit_passes_filter({"kslabel": "good"}, {"somatic": "True"})

    with pytest.raises(ValueError):
        _maker(somatic_filter="bogus")


def test_compute_behavioral_bucket_flags_no_condition(tmp_path):
    """
    Description
    -----------
    A unit with no block for the requested condition yields all-False
    bucket flags (the early-return guard), not a crash.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Pytest temp directory (throwaway maker root).

    Returns
    -------
    None
    """

    maker = _aggregate_maker(tmp_path)
    unit = {
        "mouse_id": "M1", "rec_date": 1, "unit_id": "u",
        "kslabel": "good", "anatomy_region": "PAG",
        "conditions": {"intact_female": {"sessions_tested": [], "modalities": {}}},
    }
    flags = maker._compute_behavioral_bucket_flags(
        unit, recorded_mouse_id="M1", condition="mute_female",
        k_min=2, require_majority=True,
    )
    assert flags == {"pose": False, "movement": False, "social": False}


def test_compute_behavioral_bucket_flags_dyadic_pooling(tmp_path):
    """
    Description
    -----------
    Two dyadic (social) modality keys for the same `(feat, direction)`
    but different partners — each `n_tested=1` — must be pooled before
    the consistency gate, so the social bucket flips True while pose /
    movement stay False.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Pytest temp directory (throwaway maker root).

    Returns
    -------
    None
    """

    maker = _aggregate_maker(tmp_path)
    mods = {
        "behavioral_beh_offset=0s_M1-pA.nose_excit": {"n_significant": 1, "n_tested": 1},
        "behavioral_beh_offset=0s_M1-pB.nose_excit": {"n_significant": 1, "n_tested": 1},
    }
    unit = {
        "mouse_id": "M1", "rec_date": 1, "unit_id": "u",
        "kslabel": "good", "anatomy_region": "PAG",
        "conditions": {"intact_female": {"sessions_tested": [], "modalities": mods}},
    }
    flags = maker._compute_behavioral_bucket_flags(
        unit, recorded_mouse_id="M1", condition="intact_female",
        k_min=2, require_majority=True,
    )
    assert flags["social"] is True
    assert flags["pose"] is False and flags["movement"] is False


def test_collect_vmi_consistency_below_n_tested_min(tmp_path):
    """
    Description
    -----------
    A unit with only one valid VMI session is dropped by
    `_collect_vmi_consistency` (`n_tested < n_tested_min`), leaving every
    region group empty.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Pytest temp directory for the hand-built triage + catalog.

    Returns
    -------
    None
    """

    units = {
        "M1_1_u": {
            "mouse_id": "M1", "rec_date": 1, "unit_id": "u",
            "kslabel": "good", "anatomy_region": "PAG",
            "conditions": {
                "intact_female": {
                    "sessions_tested": [],
                    "modalities": {
                        "vmi_self_excit": {"per_session": [
                            {"session": "s1", "vmi": 0.5, "p": 0.001,
                             "n_bouts": 20, "significant": True},
                        ]},
                    },
                },
            },
        },
    }
    catalog_path = tmp_path / "catalog.csv"
    catalog_path.write_text(
        "mouse_id,rec_date,unit_id,somatic\nM1,1,u,True\n"
    )
    triage_path = tmp_path / "unit_triage_min.pkl"
    with triage_path.open("wb") as fh:
        pickle.dump({
            "catalog_path": str(catalog_path),
            "thresholds_used": {"vmi_alpha": 0.01, "vmi_min_bouts": 10},
            "units": units,
        }, fh)

    maker = _aggregate_maker(tmp_path)
    per_group = maker._collect_vmi_consistency(
        triage_pkl_path=triage_path,
        catalog_csv_path=catalog_path,
        n_tested_min=2,
    )
    assert all(len(v) == 0 for v in per_group.values())


def test_consistent_peth_majority_is_strict(tmp_path):
    """
    Description
    -----------
    The `require_majority` gate is **strict** (`> 0.5`): a unit whose
    largest in-tolerance peak_t cluster is exactly half of its
    significant sessions (`k / n_sig == 0.5`) is NOT consistent, while a
    unit just above half (2/3) is. Locks the harmonised strict-majority
    boundary shared by the PETH / property / category collectors.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Pytest temp directory for the hand-built triage + catalog.

    Returns
    -------
    None
    """

    def _peth_unit(unit_id, peak_ts):
        per_session = [
            {"significant": True, "peak_t": t, "peak_z": 5.0} for t in peak_ts
        ]
        return {
            "unit_uid": f"M01_20240101_{unit_id}",
            "mouse_id": "M01", "rec_date": 20240101, "unit_id": unit_id,
            "imec": 0, "cluster_num": 1, "peak_channel": 1,
            "kslabel": "good", "anatomy_region": "PAG",
            "conditions": {
                "intact_female": {
                    "sessions_tested": [],
                    "modalities": {
                        "usv_peth_self_excit": {"per_session": per_session},
                    },
                },
            },
        }

    # Unit A: largest in-tol cluster k=2 of n_sig=4 -> k/n_sig == 0.5 -> EXCLUDED.
    unit_a = _peth_unit("imec0_cl01_ch1_good", [0.00, 0.05, 0.50, 0.55])
    # Unit B: k=2 of n_sig=3 -> 0.667 > 0.5 -> INCLUDED.
    unit_b = _peth_unit("imec0_cl02_ch1_good", [0.00, 0.05, 0.50])
    units = {u["unit_uid"]: u for u in (unit_a, unit_b)}

    catalog_path = tmp_path / "catalog.csv"
    catalog_path.write_text(
        "mouse_id,rec_date,unit_id,somatic\n"
        "M01,20240101,imec0_cl01_ch1_good,True\n"
        "M01,20240101,imec0_cl02_ch1_good,True\n"
    )
    triage_path = tmp_path / "unit_triage_strictmaj.pkl"
    with triage_path.open("wb") as fh:
        pickle.dump({"catalog_path": str(catalog_path), "units": units}, fh)

    maker = _aggregate_maker(tmp_path)
    per_group = maker._collect_consistent_peth(
        triage_pkl_path=triage_path,
        catalog_csv_path=catalog_path,
        direction="excit",
        tol_s=0.100,
        k_min=2,
        require_majority=True,
    )

    pag = per_group["PAG"]
    assert len(pag) == 1, "only the >0.5-majority unit should survive strict majority"
    assert pag[0]["n_sig"] == 3


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_collectors_honor_configurable_filter(triage_fixture, tmp_path):
    """
    Description
    -----------
    A maker configured with `kslabels=("good", "mua")` + `"both"` must
    re-admit the two units the default filter skips (one `kslabel='mua'`,
    one `somatic='False'`), so its eligible count is exactly two larger
    than the default good + somatic count over the same pickle.

    Parameters
    ----------
    triage_fixture (tuple)
        `(maker, triage_path, catalog_path, out_dir)`.
    tmp_path (pathlib.Path)
        Pytest temp directory (throwaway root for the permissive maker).

    Returns
    -------
    None
    """

    default_maker, triage_path, catalog_path, _out_dir = triage_fixture
    _c0, n_default = default_maker._collect_three_set_overlap_counts(
        triage_pkl_path=triage_path,
        catalog_csv_path=catalog_path,
        condition=_COND_A,
        k_min=2,
        require_majority=True,
    )

    permissive = NeuronalTuningFigureMaker(
        root_directory=str(tmp_path),
        visualizations_parameter_dict=_make_aggregate_visualizations_parameters(),
        message_output=lambda *a, **k: None,
        kslabels=("good", "mua"),
        somatic_filter="both",
    )
    _c1, n_permissive = permissive._collect_three_set_overlap_counts(
        triage_pkl_path=triage_path,
        catalog_csv_path=catalog_path,
        condition=_COND_A,
        k_min=2,
        require_majority=True,
    )

    assert n_permissive == n_default + 2, (
        f"permissive filter eligible count {n_permissive} should be the "
        f"default {n_default} plus the two re-admitted skip units"
    )


# ---------------------------------------------------------------------------
# Per-cluster behavioral-page rendering: the `make_neuronal_tuning_figures`
# dispatcher's non-PDF (one-file-per-page) save path, plus the social /
# directional-SEI branches of `_render_behavioral_pages` that the
# single-mouse full-pipeline fixtures never reach. The full-pipeline tests
# above already exercise the 1D self-feature + 2D spatial happy path via the
# default PDF backend; these add the format / multi-animal branches.
# ---------------------------------------------------------------------------


def _beh_1d_payload(n_bins: int = 12) -> dict:
    """
    Description
    -----------
    Build a minimal 1D behavioral-feature payload as
    `_render_behavioral_pages` reads it: `bin_edges`, `bin_centers`,
    `rate`, `occupancy_seconds` (all above the occupancy threshold so
    the rate line / shuffle band render), and the `null_p0_5` /
    `null_p99_5` shuffle bounds. No `_smoothed` variants are included so
    the renderer takes its un-smoothed `.get(..., default)` fallbacks.

    Parameters
    ----------
    n_bins (int)
        Number of feature bins.

    Returns
    -------
    payload (dict)
        1D feature payload.
    """

    edges = np.linspace(0.0, 30.0, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return {
        "bin_edges":         edges,
        "bin_centers":       centers,
        "rate":              np.linspace(1.0, 5.0, n_bins),
        "occupancy_seconds": np.full(n_bins, 5.0),
        "null_p0_5":         np.full(n_bins, 0.5),
        "null_p99_5":        np.full(n_bins, 6.0),
    }


def _beh_2d_payload(side: int = 8) -> dict:
    """
    Description
    -----------
    Build a minimal 2D spatial (`.space`) behavioral-feature payload:
    a `side x side` `rate` ratemap and matching `occupancy_seconds`
    grid. Deterministic ramp values keep the colorbar tick formatting
    branch exercised without depending on RNG.

    Parameters
    ----------
    side (int)
        Edge length of the square ratemap.

    Returns
    -------
    payload (dict)
        2D spatial feature payload.
    """

    grid = np.arange(side * side, dtype=float).reshape(side, side)
    return {
        "rate":              grid,
        "occupancy_seconds": grid + 0.5,
    }


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_per_cluster_pipeline_png_writes_per_page_files(tmp_path):
    """
    Description
    -----------
    Drive `make_neuronal_tuning_figures` with `figures.fig_format='png'`
    so the dispatcher takes the non-PDF branch of `_open_save_target`:
    each rendered page is written as its own
    `<cluster>_neuronal_tuning_p{N}_{label}.png` file (rather than one
    multi-page PDF). Asserts several per-page PNGs land on disk and that
    no combined PDF is produced.

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

    viz = _make_visualizations_parameters()
    viz["figures"] = {"fig_format": "png", "dpi": 80}
    figure_maker = NeuronalTuningFigureMaker(
        root_directory=str(root),
        visualizations_parameter_dict=viz,
        message_output=lambda *a, **k: None,
    )
    figure_maker.make_neuronal_tuning_figures()

    tuning_dir = root / "ephys" / "tuning_curves"
    per_page = sorted(tuning_dir.glob("*_neuronal_tuning_p*.png"))
    assert len(per_page) >= 3, (
        f"expected >= 3 per-page PNGs (behavioral + 2 vocal pages), got "
        f"{[p.name for p in per_page]}"
    )
    assert not list(tuning_dir.glob("*_neuronal_tuning.pdf")), (
        "a combined PDF was written despite fig_format='png' — the non-PDF "
        "branch of _open_save_target did not take"
    )
    for png in per_page:
        assert png.stat().st_size > 1_000, f"per-page PNG too small: {png}"


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_render_behavioral_pages_renders_social_and_directional_sei():
    """
    Description
    -----------
    Unit-test `_render_behavioral_pages` directly with a two-mouse,
    hand-crafted `cluster_data` so the branches the single-mouse
    full-pipeline fixtures never reach all execute:

      * the `social` plot-feature group (dyadic `m1-m2.*` keys routed to
        the social bucket with the social ratemap color),
      * the directional-SEI title path (`orofacial-sei` under a
        hyphenated `m1-m2` prefix → `feat (1→2)` mathtext title),
      * the 2D spatial (`.space`) cell with its own colorbar, AND the
        `gs_y > 5` row-wrap after both a spatial cell and a 1D cell.

    Feature ordering places the `.space` cell at column 4 (so its
    post-cell `gs_y += 2` wraps the row) and a 1D cell at column 4 on
    the next row (so the 1D wrap fires too). `save_fig` records one
    label per rendered page; we assert both the individual-mouse page
    and the social page were committed.

    Parameters
    ----------

    Returns
    -------
    None
    """

    cluster_data = {
        "beh_offset=0s": {
            "m1.speed":            _beh_1d_payload(),
            "m1.acceleration":     _beh_1d_payload(),
            "m1.space":            _beh_2d_payload(),
            "m1.allo_yaw":         _beh_1d_payload(),
            "m1.ego_yaw":          _beh_1d_payload(),
            "m1.neck_elevation":   _beh_1d_payload(),
            "m1-m2.nose-nose":     _beh_1d_payload(),
            "m1-m2.orofacial-sei": _beh_1d_payload(),
        },
        "behavioral_metadata": {"behavioral_min_occupancy_seconds": 0.1},
    }

    maker = NeuronalTuningFigureMaker(
        root_directory="/tmp",
        visualizations_parameter_dict=_make_aggregate_visualizations_parameters(),
        message_output=lambda *a, **k: None,
    )

    saved_labels: list[str] = []

    def _save_fig(fig, label: str) -> None:
        """Record the rendered page's label and close its figure (stands in
        for the per-cluster PDF/PNG save target)."""
        saved_labels.append(label)
        plt.close(fig)

    maker._render_behavioral_pages(
        cluster_data=cluster_data,
        mouse_id_list=["m1", "m2"],
        mouse_colors=["#9AC0CD", "#8CA252"],
        save_fig=_save_fig,
    )

    assert any("individual.m1" in lbl for lbl in saved_labels), (
        f"individual-mouse behavioral page not rendered; got {saved_labels}"
    )
    assert any(lbl.endswith("social") for lbl in saved_labels), (
        f"social behavioral page not rendered; got {saved_labels}"
    )


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_render_behavioral_pages_returns_early_without_beh_offset():
    """
    Description
    -----------
    `_render_behavioral_pages` must early-return (rendering nothing)
    when the cluster payload carries no `beh_offset=*` keys — the guard
    that stops `make_neuronal_tuning_figures` from opening a page target
    for a vocal-only cluster's behavioral half. Asserts `save_fig` is
    never invoked.

    Parameters
    ----------

    Returns
    -------
    None
    """

    maker = NeuronalTuningFigureMaker(
        root_directory="/tmp",
        visualizations_parameter_dict=_make_aggregate_visualizations_parameters(),
        message_output=lambda *a, **k: None,
    )

    calls: list[str] = []

    def _save_fig(fig, label: str) -> None:  # pragma: no cover - must not run
        """Record the page label and close its figure; must never be called on
        a payload with no ``beh_offset`` keys (asserted by the test)."""
        calls.append(label)
        plt.close(fig)

    maker._render_behavioral_pages(
        cluster_data={"usv_peth": {}},
        mouse_id_list=["m1"],
        mouse_colors=["#9AC0CD"],
        save_fig=_save_fig,
    )

    assert calls == [], (
        f"_render_behavioral_pages rendered pages for a payload with no "
        f"beh_offset keys; save_fig called with {calls}"
    )
