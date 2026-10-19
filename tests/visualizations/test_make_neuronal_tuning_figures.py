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

Coverage: the full compute-to-figure pipeline (PDF and per-page PNG) with the tracking file at
either location; the pickle payload contract; the property / categorical panel axis rules; every
cross-unit summary figure (VMI, PETH timing, property and category tuning, behavioral tuning, Venn
overlaps, PAG gradient); the unit filter and the collectors' consistency rules; the behavioral,
spatial and vocal page renderers, the bout-onset panels and the QLVM segmentation loaded from the
category bundle; figures closed when a page or cluster fails; and the settings the maker requires.
"""

from __future__ import annotations

import json
import pathlib
import pickle
import re

import matplotlib as mpl

mpl.use("Agg")
import h5py
import matplotlib.collections
import matplotlib.colors
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import polars as pls
import pytest
from matplotlib import gridspec
from matplotlib.figure import Figure

from usv_playpen import os_utils
from usv_playpen.analyses.compute_neuronal_tuning_curves import (
    CATEGORICAL_FEATURES,
    CONTINUOUS_PROPERTIES,
    NeuronalTuning,
)
from usv_playpen.visualizations import make_neuronal_tuning_figures as tuning_figures
from usv_playpen.visualizations.make_neuronal_tuning_figures import (
    DISPLAY_FACTOR,
    USV_CATEGORY_SEGMENTATIONS,
    USV_PROPERTY_ORDER,
    NeuronalTuningFigureMaker,
    _category_class_count,
    load_qlvm_category_segmentation,
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
        h5_path = root / "20230101000000_points3d_translated_rotated_metric.h5"
    else:
        h5_path = root / "video" / "vid1" / "20230101000000_points3d_translated_rotated_metric.h5"
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("track_names", data=np.array([b"m1"]))
        f.create_dataset("recording_frame_rate", data=np.float64(fps))
        # `E1M` decodes to {experiment_type: [ephys], mouse_number: 1,
        # mouse_sex: [male], ...} — `choose_animal_colors` needs a
        # non-empty `mouse_sex` of the right length to iterate.
        f.create_dataset("experimental_code", data=b"E1M")
        f.create_dataset("tracks", data=np.zeros((n_frames, 1, 1, 3), dtype=float))

    # The vocal compute reads each animal's sex from the session metadata Subjects.
    (root / f"{root.name}_metadata.yaml").write_text("Subjects:\n- subject_id: 'm1'\n  sex: male\n")

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
        "usv":               [True] * n_usvs,
        "squeak":            [False] * n_usvs,
        "squeak_start":      [None] * n_usvs,
        "squeak_end":        [None] * n_usvs,
        "qlvm_category":     rng.integers(1, 5, size=n_usvs).tolist(),
        "mean_freq_hz":      rng.uniform(40000, 90000, n_usvs).tolist(),
        "peak_freq_hz":      rng.uniform(40000, 90000, n_usvs).tolist(),
        "freq_bandwidth_hz": rng.uniform(5000, 30000, n_usvs).tolist(),
        "mean_amplitude":    rng.uniform(0.1, 3.0, n_usvs).tolist(),
        "max_amplitude":     rng.uniform(0.5, 8.0, n_usvs).tolist(),
        "spectral_entropy":  rng.uniform(1.0, 4.0, n_usvs).tolist(),
        "mask_number":       rng.integers(1, 12, n_usvs).tolist(),
        # Every synthetic USV is real; the figures drop noise-flagged rows.
        "noise":             [False] * n_usvs,
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
        "peth_window_seconds":                      [-2.0, 0.5],
        "peth_bin_seconds":                         0.05,
        "bout_quiet_seconds":                       2.0,
        "ramp_anchor_near_seconds":                 -0.1,
        "ramp_anchor_far_seconds":                  -1.5,
        "vmi_wilcoxon_exact_max_n":                 25,
        "ibi_bout_onset": {
            "ibi_z_score":              2.58,
            "ibi_component_index":      0,
            "min_vocalizations":        3,
            "shuffle_guard_seconds":    1.0,
            "trigger_window_seconds":   [-1.0, 2.0],
            "trigger_bin_seconds":      0.02,
            "smoothing_bins":           2.0,
            "gain_window_seconds":      [0.0, 0.25],
            "pre_window_seconds":       [-1.0, -0.5],
            "peak_search_seconds":      [-0.25, 1.0],
            "response_window_seconds":  0.15,
            "response_kernel_seconds":  0.02,
        },
        "vocal_require_clean_post_anchor":          True,
        "vocal_require_clean_prior_anchor":         False,
        "n_usv_min_self":                           5,
        "n_usv_min_partner":                        30,
        "n_usv_min_category":                       3,
        "behavioral_min_occupancy_seconds":         0.1,
        "usv_property_min_occupancy_seconds":       0.05,
        "include_partner_vocalization_tuning_bool": False,
        "exclude_squeaks_self":                     True,
        "exclude_squeaks_partner":                  False,
        "excluded_behavioral_features":             ["nose-nose", "allo_yaw-nose", "nose-allo_yaw",
                                                     "allo_pitch-nose", "nose-allo_pitch"],
        "smoothing_sd":                             0.0,
        "circular_features":                        ["allo_yaw", "body_dir"],
    }


# The shipped visualization settings, for the blocks the tests take as shipped.
with (pathlib.Path(tuning_figures.__file__).resolve().parent.parent / "_parameter_settings"
      / "visualizations_settings.json").open(encoding="utf-8") as _settings_file:
    _SHIPPED_VIZ_SETTINGS = json.load(_settings_file)


def _make_visualizations_parameters() -> dict:
    """
    Description
    -----------
    Return the minimum `visualizations_parameter_dict` the figure maker
    reads: `male_colors` / `female_colors` palettes (only first entry of
    each is touched), a `figures` block (PDF output, dpi, seed and the
    colormaps, read directly by the figure maker) and the shipped
    `neuronal_tuning_figures` block (the bout panels' binning, smoothing and
    contour settings).

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
        "figures":       {"sequential_cmap": "inferno", "diverging_cmap": "BrBG_r",
                          "fig_format": "pdf", "dpi": 300, "seed": 0, "timestamp_in_name": False},
        "neuronal_tuning_figures": _SHIPPED_VIZ_SETTINGS["neuronal_tuning_figures"],
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
@pytest.mark.usefixtures("qlvm_category_bundle")
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
      * the PDF has >= 2 pages (behavioral + one vocal page per
        emitter), catching the historical "zero-page PdfPages" silent
        failure

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Pytest-provided per-test temp directory.
    qlvm_category_bundle (pathlib.Path)
        The synthetic category bundle the section-(c) watersheds read.

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
        message_output=lambda *_args, **_kwargs: None,
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
        message_output=lambda *_args, **_kwargs: None,
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
    assert n_pages >= 2, (
        f"PDF has only {n_pages} pages; expected behavioral + vocal "
        "(>=2 total). A 1-page PDF means only one of the two halves "
        "rendered — usually the figure-side `has_vocal` predicate not "
        "matching the compute side's key names, or the tracking-H5 "
        "lookup returning an empty mouse_id_list."
    )


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.usefixtures("qlvm_category_bundle")
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
        message_output=lambda *_args, **_kwargs: None,
    )
    nt.calculate_neuronal_tuning_curves()

    figure_maker = NeuronalTuningFigureMaker(
        root_directory=str(root),
        visualizations_parameter_dict=_make_visualizations_parameters(),
        message_output=lambda *_args, **_kwargs: None,
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
    assert n_pages >= 2, (
        f"scratch-layout PDF has only {n_pages} pages — behavioral pages "
        "likely skipped due to empty mouse_id_list (tracking-H5 lookup "
        "not finding the session-root H5)."
    )


# Per-bug rendering regressions: each test exercises a single private
# rendering method with a hand-crafted payload that surfaces exactly one
# class of "axis bound vs rendered data extent" mismatch, then inspects
# the resulting matplotlib axes for the regression. These tests are
# fast (no compute, no PDF I/O) and pinpoint the offending code path
# precisely when a fix regresses.


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
        message_output=lambda *_args, **_kwargs: None,
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
    p0_5 = np.full(n_bins, np.nan)
    p0_5[5:21] = 2.0
    p99_5 = np.full(n_bins, np.nan)
    p99_5[5:21] = 16.0
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
        x_lo_lim, _x_hi_lim = ax.get_xlim()
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


# Full pkl contract: walks the ENTIRE payload structure
# `NeuronalTuning.calculate_neuronal_tuning_curves` writes per cluster and
# asserts every expected key and shape consistency at every level. This
# is the safety net for the renaming-class bug (compute writes `usv_*`
# but figure code looked for `vocal_q*`) at every nested layer.


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
    "usv_ibi_bout_onset",
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
        message_output=lambda *_args, **_kwargs: None,
    )
    nt.calculate_neuronal_tuning_curves()

    pkls = sorted((root / "ephys" / "tuning_curves").glob("*_tuning_curves_data.pkl"))
    assert len(pkls) == 1, f"expected 1 pkl, got {len(pkls)}"
    with pkls[0].open("rb") as fh:
        d = pickle.load(fh)

    # top level
    assert set(d.keys()) == _TOP_LEVEL_KEYS, (
        f"top-level keys mismatch.\n"
        f"  expected: {sorted(_TOP_LEVEL_KEYS)}\n"
        f"  got:      {sorted(d.keys())}\n"
        f"  missing:  {sorted(_TOP_LEVEL_KEYS - set(d.keys()))}\n"
        f"  extra:    {sorted(set(d.keys()) - _TOP_LEVEL_KEYS)}"
    )

    # beh_offset=0s
    beh = d["beh_offset=0s"]
    assert isinstance(beh, dict), "beh_offset=0s is empty"
    assert beh, "beh_offset=0s is empty"
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

    # behavioral_metadata
    bm = d["behavioral_metadata"]
    missing = _BEHAVIORAL_METADATA_KEYS - set(bm.keys())
    assert not missing, f"behavioral_metadata missing keys: {sorted(missing)}"

    # usv_peth[emitter]
    usv_peth = d["usv_peth"]
    assert isinstance(usv_peth, dict), "usv_peth is empty"
    assert usv_peth, "usv_peth is empty"
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

    # usv_property_tuning[emitter][prop]
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

    # usv_category_tuning[emitter][cat_feat]
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

    # usv_category_peth[emitter][cat_feat]
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

    # usv_metadata
    um = d["usv_metadata"]
    missing = _USV_METADATA_KEYS - set(um.keys())
    assert not missing, f"usv_metadata missing keys: {sorted(missing)}"

    # triage_stats
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
    n_sig = sum(1 for e in per_session if e["significant"])
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
        "mean_amplitude":    (0.50, 0.52),   # per-USV normalized [0, 1] scale
        "max_amplitude":     (0.93, 0.96),
        "loudness_db":       (60.0, 62.0),   # absolute dB
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
    and a `figures` block (colormaps, format, dpi and timestamp rule, read
    by the figures and by `figure_io`). Values mirror the project's
    `visualizations_settings.json`.

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
        "figures":           {"sequential_cmap": "inferno", "diverging_cmap": "BrBG_r",
                              "fig_format": "png", "dpi": 80, "seed": 0, "timestamp_in_name": False},
        "neuronal_tuning_figures": _SHIPPED_VIZ_SETTINGS["neuronal_tuning_figures"],
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
        message_output=lambda *_args, **_kwargs: None,
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
    # The knob is read from settings; a copied block keeps the shipped one untouched.
    maker.visualizations_parameter_dict = {
        **maker.visualizations_parameter_dict,
        "neuronal_tuning_figures": {
            **maker.visualizations_parameter_dict["neuronal_tuning_figures"],
            "vmi_bootstrap_n": 50,
        },
    }
    out_path = maker.make_vmi_cross_condition_stability_figure(
        triage_pkl_path=triage_path,
        out_dir=out_dir,
        fig_format="png",
    )
    _assert_figure_written(out_path, out_dir)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_vmi_stability_figure_renders_when_the_interval_excludes_the_estimate(triage_fixture):
    """
    Description
    -----------
    A percentile bootstrap interval can exclude its own point estimate. With the
    per-region unit minimum lowered to 3 so the fixture's regions are scored, setting
    both bootstrap quantiles to 0 makes every region's interval collapse onto its smallest
    bootstrap r, which lies below the point estimate, so the upper error length
    ``hi - r`` is negative. The error bars are clipped at 0, so the figure must still be
    written instead of matplotlib raising on a negative error.

    Parameters
    ----------
    triage_fixture (tuple)
        `(maker, triage_path, catalog_path, out_dir)`.

    Returns
    -------
    None
    """

    maker, triage_path, _catalog_path, out_dir = triage_fixture
    maker.visualizations_parameter_dict = {
        **maker.visualizations_parameter_dict,
        "neuronal_tuning_figures": {
            **maker.visualizations_parameter_dict["neuronal_tuning_figures"],
            "vmi_bootstrap_n": 50,
            "vmi_bootstrap_quantiles": [0.0, 0.0],
            "stability_min_units": 3,
        },
    }
    out_path = maker.make_vmi_cross_condition_stability_figure(
        triage_pkl_path=triage_path,
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
    # The knob is read from settings; a copied block keeps the shipped one untouched.
    maker.visualizations_parameter_dict = {
        **maker.visualizations_parameter_dict,
        "neuronal_tuning_figures": {
            **maker.visualizations_parameter_dict["neuronal_tuning_figures"],
            "pag_kde_grid_resolution": 40,
        },
    }
    out_path = maker.make_pag_anatomical_gradient_figure(
        triage_pkl_path=triage_path,
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
@pytest.mark.usefixtures("qlvm_category_bundle")
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
        """
        Description
        -----------
        Construct a silent `NeuronalTuningFigureMaker` on the throwaway root
        with the aggregate settings, overriding only the unit-filter knobs.

        Parameters
        ----------
        **kw
            Constructor overrides (``kslabels``, ``somatic_filter``).

        Returns
        -------
        maker (NeuronalTuningFigureMaker)
            The configured figure maker.
        """

        return NeuronalTuningFigureMaker(
            root_directory=str(tmp_path),
            visualizations_parameter_dict=viz,
            message_output=lambda *_args, **_kwargs: None,
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

    with pytest.raises(ValueError, match="somatic_filter must be one of"):
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
    assert flags["pose"] is False
    assert flags["movement"] is False


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
        """
        Description
        -----------
        Build one triage unit whose only modality is a self-excitatory USV
        PETH, with one significant session per requested peak time, so the
        peak-time clustering rule alone decides whether the unit is kept.

        Parameters
        ----------
        unit_id (str)
            Unit identifier, also embedded in ``unit_uid``.
        peak_ts (list[float])
            Per-session PETH peak times (s).

        Returns
        -------
        unit (dict)
            Triage unit record in the shape the collectors read.
        """

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
        message_output=lambda *_args, **_kwargs: None,
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


# Per-cluster behavioral-page rendering: the `make_neuronal_tuning_figures`
# dispatcher's non-PDF (one-file-per-page) save path, plus the social /
# directional-SEI branches of `_render_behavioral_pages` that the
# single-mouse full-pipeline fixtures never reach. The full-pipeline tests
# above already exercise the 1D self-feature + 2D spatial happy path via the
# default PDF backend; these add the format / multi-animal branches.


def _beh_1d_payload(n_bins: int = 12) -> dict:
    """
    Description
    -----------
    Build a minimal 1D behavioral-feature payload as
    `_render_behavioral_pages` reads it: `bin_edges`, `bin_centers`,
    `rate`, `occupancy_seconds` (all above the occupancy threshold so
    the rate line / shuffle band render), and the `null_p0_5` /
    `null_p99_5` shuffle bounds. No `_smoothed` variants are included so
    the renderer falls back to the un-smoothed keys.

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
@pytest.mark.usefixtures("qlvm_category_bundle")
def test_per_cluster_pipeline_png_writes_per_page_files(tmp_path):
    """
    Description
    -----------
    Drive `make_neuronal_tuning_figures` with `figures.fig_format='png'`
    so the dispatcher takes the non-PDF branch of `_open_save_target`:
    each rendered page is written as its own
    `<cluster>_neuronal_tuning_p{N}_{label}.png` file (rather than one
    multi-page PDF). Asserts several per-page PNGs land on disk, that each
    emitter has exactly one vocal page (`vocal_<sex>`: section (c), the
    single QLVM category row, sits on the vocal page instead of a page of
    its own) and that no combined PDF is produced.

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
        message_output=lambda *_args, **_kwargs: None,
    )
    nt.calculate_neuronal_tuning_curves()

    viz = _make_visualizations_parameters()
    viz["figures"] = {"fig_format": "png", "dpi": 80, "seed": 0, "timestamp_in_name": False,
                      "sequential_cmap": "inferno", "diverging_cmap": "BrBG_r"}
    figure_maker = NeuronalTuningFigureMaker(
        root_directory=str(root),
        visualizations_parameter_dict=viz,
        message_output=lambda *_args, **_kwargs: None,
    )
    figure_maker.make_neuronal_tuning_figures()

    tuning_dir = root / "ephys" / "tuning_curves"
    per_page = sorted(tuning_dir.glob("*_neuronal_tuning_p*.png"))
    assert len(per_page) >= 2, (
        f"expected >= 2 per-page PNGs (behavioral + vocal pages), got "
        f"{[p.name for p in per_page]}"
    )
    vocal_pages = [p.name for p in per_page if "_vocal_" in p.name]
    assert vocal_pages, f"no vocal page among {[p.name for p in per_page]}"
    assert not any("_vocal_a_" in name or "_vocal_b_" in name for name in vocal_pages), vocal_pages
    vocal_labels = [name.split("_vocal_", 1)[1] for name in vocal_pages]
    assert len(vocal_labels) == len(set(vocal_labels)), f"an emitter has more than one vocal page: {vocal_pages}"
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
        message_output=lambda *_args, **_kwargs: None,
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
def test_render_behavioral_pages_draws_spatial_maps_x_across_y_up():
    """
    Description
    -----------
    The spatial rate and occupancy maps are indexed ``[x_bin, y_bin]``; the cells must
    draw X across and Y up to match their "X (cm)" / "Y (cm)" labels: the image array is
    the transposed map and the origin is at the bottom. A map that is hot only at large x
    (last row) must show its hot column on the right.

    Parameters
    ----------

    Returns
    -------
    None
    """

    rate = np.zeros((6, 6))
    rate[-1, :] = 5.0
    cluster_data = {
        "beh_offset=0s": {"m1.space": {"rate": rate, "occupancy_seconds": rate + 1.0}},
        "behavioral_metadata": {"behavioral_min_occupancy_seconds": 0.1},
    }
    maker = NeuronalTuningFigureMaker(
        root_directory="/tmp",
        visualizations_parameter_dict=_make_aggregate_visualizations_parameters(),
        message_output=lambda *_args, **_kwargs: None,
    )
    images = []

    def _save_fig(fig, _label: str) -> None:
        """Collect every image drawn on the page, then close it."""
        for ax in fig.axes:
            if ax.get_xlabel() == "X (cm)":
                images.extend(ax.get_images())
        plt.close(fig)

    maker._render_behavioral_pages(
        cluster_data=cluster_data, mouse_id_list=["m1"], mouse_colors=["#9AC0CD"], save_fig=_save_fig,
    )
    assert len(images) == 2
    for image, grid in zip(images, (rate, rate + 1.0), strict=True):
        np.testing.assert_array_equal(np.asarray(image.get_array()), grid.T)
        assert image.origin == "lower"
        assert np.asarray(image.get_array())[:, -1].min() > np.asarray(image.get_array())[:, 0].max()


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
        message_output=lambda *_args, **_kwargs: None,
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


def test_category_class_count_grows_with_the_labels_units_hold():
    """The QLVM class count comes from the label grid when there is one (here 15
    categories); units tuned to a category above the base count widen the axis instead
    of dropping out, and with no grid (the production cells carry none) the units
    alone set the count."""
    grids = {"qlvm_category": {"unique_labels": list(range(1, 16))}}
    per_group = {"PAG": [{"best_cat": 3}, {"best_cat": 12}], "VMH": []}
    assert _category_class_count("qlvm_category", per_group, grids) == 15
    assert _category_class_count("qlvm_category", {"PAG": [{"best_cat": 17}]}, grids) == 17
    assert _category_class_count("qlvm_category", {"PAG": [{"best_cat": 4}]}, {}) == 4


def test_qlvm_segmentation_comes_from_the_category_bundle(qlvm_category_bundle):
    """The only categorical feature, qlvm_category, takes its watersheds and class count
    from the category bundle's label grid (os_utils.load_qlvm_category_bundle), on the
    unit torus with pixel [y, x] at (x, y); no coarse level and no conditional-map
    feature is loaded, and nothing is reported."""
    messages = []
    maker = NeuronalTuningFigureMaker(
        root_directory="/tmp",
        visualizations_parameter_dict=_make_visualizations_parameters(),
        message_output=messages.append,
    )

    segmentation = maker._load_segmentation()

    with np.load(qlvm_category_bundle / "category_grids.npz") as grids:
        expected = grids["label_grid"]
    assert set(segmentation) == {"qlvm_category"}
    np.testing.assert_array_equal(segmentation["qlvm_category"]["label_grid"], expected)
    assert segmentation["qlvm_category"]["unique_labels"] == [1, 2, 3, 4]
    assert segmentation["qlvm_category"]["names"] == ["R-1", "R-2", "R-3", "R-4"]
    np.testing.assert_array_equal(segmentation["qlvm_category"]["bounds"], [0.0, 1.0, 0.0, 1.0])
    # Pixel [row=y, col=x] is centred at ((x + 0.5) / res, (y + 0.5) / res).
    res = expected.shape[0]
    assert segmentation["qlvm_category"]["xx"][1, 3] == 3.5 / res
    assert segmentation["qlvm_category"]["yy"][1, 3] == 1.5 / res
    assert _category_class_count("qlvm_category", {}, segmentation) == 4
    assert messages == []
    assert tuning_figures.SECTION_C_ROWS == (("qlvm_category",),)
    assert tuning_figures.USV_CATEGORY_SEGMENTATIONS == ("qlvm_category",)


def test_qlvm_segmentation_unreadable_bundle_raises(tmp_path, monkeypatch):
    """With the category bundle unreadable the segmentation raises, naming the bundle:
    there are no placeholder panels and no grid of another partition."""
    monkeypatch.setattr(os_utils, "QLVM_CATEGORY_BUNDLE_DIRECTORY", str(tmp_path / "missing"))
    maker = NeuronalTuningFigureMaker(
        root_directory="/tmp",
        visualizations_parameter_dict=_make_visualizations_parameters(),
        message_output=lambda *_a: None,
    )

    with pytest.raises(FileNotFoundError, match="missing"):
        maker._load_segmentation()
    with pytest.raises(FileNotFoundError):
        load_qlvm_category_segmentation()


def test_vocal_page_layout_puts_section_c_on_the_vocal_page():
    """Section (c) is a single row, so it shares the vocal page with sections (a) and
    (b) instead of a page of its own. The property-grid row height keeps 17.5 units for
    the grid's rows and their 0.45-row gaps; the constants of the retired two-section
    layout (the page size and the three outer height ratios / gap) are gone."""
    assert len(tuning_figures.SECTION_C_ROWS) == 1
    n_rows = len(tuning_figures.PROPERTY_ROW_ORDER)
    assert tuning_figures.VOCAL_SECTION_B_ROW_HEIGHT * (n_rows + 0.45 * (n_rows - 1)) == pytest.approx(17.5)
    for retired in ("VOCAL_PAGE_FIGSIZE_INCHES", "VOCAL_SECTION_HEIGHT_RATIOS", "VOCAL_SECTION_HSPACE"):
        assert not hasattr(tuning_figures, retired)
    assert not hasattr(tuning_figures, "VOCAL_PAGE2_FIGSIZE_INCHES")
    assert not hasattr(NeuronalTuningFigureMaker, "_render_page2")


def _bout_onset_figure_inputs(tmp_path: pathlib.Path) -> tuple[NeuronalTuningFigureMaker, dict,
                                                               pls.DataFrame]:
    """
    Description
    -----------
    Build the inputs the bout-onset panels read, on a synthetic 120 s session.

    The male `m1` produces 19 IBI bouts, one every 6 s from 5 s, each of three 100 ms USVs 200 ms
    apart; two USVs are assigned to no animal and the female `f1` never vocalizes. The unit fires
    a Poisson background plus a burst in the 300 ms after every bout onset, and its spike file is
    written at `<tmp_path>/ephys/imec0/cluster_data/imec0_cl0001.npy` (spike times in row 0), the
    layout `_cluster_spike_times` searches under the rendered directory. The pkl payload carries
    the `usv_metadata` fields the panels read, a bout-onset block for both emitters and a pooled
    `usv_peth` block for `m1`.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Pytest-provided temporary directory, used as the rendered session root.

    Returns
    -------
    maker, cluster_data, usv_summary_df (tuple)
        A figure maker rooted at `tmp_path`, the synthetic pkl payload and the USV summary.
    """

    rng = np.random.default_rng(0)
    onsets = np.arange(5.0, 115.0, 6.0)
    starts = np.concatenate([onsets + offset for offset in (0.0, 0.2, 0.4)] + [[2.0, 60.0]])
    emitters = ["m1"] * (3 * onsets.size) + [None, None]
    usv_summary_df = pls.DataFrame({"start": starts, "stop": starts + 0.1,
                                    "emitter": emitters}).sort("start")
    spikes = np.sort(np.concatenate([rng.uniform(0.0, 120.0, size=600)]
                                    + [o + rng.uniform(0.0, 0.3, size=8) for o in onsets]))
    cluster_dir = tmp_path / "ephys" / "imec0" / "cluster_data"
    cluster_dir.mkdir(parents=True)
    np.save(cluster_dir / "imec0_cl0001.npy", np.vstack([spikes, np.zeros_like(spikes)]))
    cfg = _make_tuning_parameters()["ibi_bout_onset"]
    bin_centers = np.arange(-2.0, 0.5, 0.05) + 0.025
    cluster_data = {
        "usv_metadata": {"cluster_id": "imec0_cl0001", "session_root": None,
                         "ibi_bout_onset": cfg, "duration_seconds": 120.0},
        "usv_ibi_bout_onset": {
            "m1": {"ibi_threshold_seconds": 0.1155, "sex": "male", "n_bouts": onsets.size},
            "f1": {"ibi_threshold_seconds": 0.1155, "sex": "female", "n_bouts": 0},
        },
        "usv_peth": {"m1": {"sex": "male", "bin_centers_s": bin_centers,
                            "rate": np.full(bin_centers.size, 5.0),
                            "null_p0_5": np.full(bin_centers.size, 3.0),
                            "null_p99_5": np.full(bin_centers.size, 7.0),
                            "n_anchors": 3 * onsets.size}},
    }
    maker = NeuronalTuningFigureMaker(root_directory=str(tmp_path),
                                      visualizations_parameter_dict=_make_visualizations_parameters(),
                                      message_output=lambda *_args, **_kwargs: None)
    return maker, cluster_data, usv_summary_df


def _bout_page_grid(figure: Figure) -> gridspec.GridSpec:
    """
    Description
    -----------
    The top three rows of the vocal page grid, which is all the bout-onset panels address.

    Parameters
    ----------
    figure (matplotlib.figure.Figure)
        Target figure.

    Returns
    -------
    outer (gridspec.GridSpec)
        A three-row grid with `VOCAL_PAGE_COLUMNS` columns.
    """

    return gridspec.GridSpec(nrows=3, ncols=tuning_figures.VOCAL_PAGE_COLUMNS, figure=figure)


def test_bout_onset_panels_draw_session_zoom_scatter_and_raster(tmp_path):
    """
    Description
    -----------
    With a bout-onset payload the page draws the session rate trace, the zoom on the densest
    20 s window, the first-versus-later scatter and the bout-triggered raster. The titles report
    the 19 bouts the emitter's USVs form under the payload's own IBI threshold, the zoom carries a
    spike raster, the scatter has a density image and contours, and the raster gains a twinned
    rate axis.
    """

    maker, cluster_data, usv_summary_df = _bout_onset_figure_inputs(tmp_path)
    figure = Figure(figsize=(16, 9))
    outer = _bout_page_grid(figure)
    maker._draw_bout_rows(fig=figure, outer=outer, emitter="m1", cluster_data=cluster_data,
                          usv_summary_df=usv_summary_df,
                          payload=cluster_data["usv_ibi_bout_onset"]["m1"])
    ax_session, ax_zoom, ax_scatter = figure.axes
    assert "19 bouts" in ax_session.get_title(loc="left")
    assert ax_session.get_xlim() == pytest.approx((0.0, 120.0))
    assert "the densest window" in ax_zoom.get_title(loc="left")
    assert ax_zoom.get_xlim()[1] - ax_zoom.get_xlim()[0] == pytest.approx(
        maker.visualizations_parameter_dict["neuronal_tuning_figures"]["bout_zoom_seconds"])
    assert any(isinstance(c, matplotlib.collections.EventCollection) for c in ax_zoom.collections)
    assert "first vs later USV" in ax_scatter.get_title(loc="left")
    assert "19 bouts" in ax_scatter.get_title(loc="left")
    assert ax_scatter.images

    ax_raster = figure.add_subplot(outer[0, tuning_figures.VOCAL_PAGE_COLUMNS - 1])
    maker._draw_bout_raster(ax=ax_raster, emitter="m1", usv_summary_df=usv_summary_df,
                            cluster_data=cluster_data)
    assert "19 bouts" in ax_raster.get_title(loc="left")
    assert ax_raster.get_ylim() == pytest.approx((-0.5, 18.5))
    assert len(figure.axes) == 5


def test_bout_onset_panels_without_payload_or_bouts_say_so(tmp_path):
    """
    Description
    -----------
    A pickle with no bout-onset block, or an emitter the block does not name, leaves the raster
    with the text "no bout-onset payload". An emitter that has a payload but never vocalizes gets
    "no bouts" in the raster, a session trace whose zoom falls back to the start of the session,
    and "too few multi-USV bouts" instead of a scatter. With no USV summary at all the raster
    panel says it was omitted while the pooled PETH beside it is still drawn.
    """

    maker, cluster_data, usv_summary_df = _bout_onset_figure_inputs(tmp_path)
    figure = Figure(figsize=(16, 9))
    outer = _bout_page_grid(figure)

    no_block = {key: value for key, value in cluster_data.items() if key != "usv_ibi_bout_onset"}
    for payload_source, emitter in ((no_block, "m1"), (cluster_data, "u9")):
        ax = figure.add_subplot(outer[2, 0])
        maker._draw_bout_raster(ax=ax, emitter=emitter, usv_summary_df=usv_summary_df,
                                cluster_data=payload_source)
        assert [t.get_text() for t in ax.texts] == ["no bout-onset payload"]
        figure.delaxes(ax)

    ax = figure.add_subplot(outer[2, 0])
    maker._draw_bout_raster(ax=ax, emitter="f1", usv_summary_df=usv_summary_df,
                            cluster_data=cluster_data)
    assert [t.get_text() for t in ax.texts] == ["no bouts"]
    figure.delaxes(ax)

    maker._draw_bout_rows(fig=figure, outer=outer, emitter="f1", cluster_data=cluster_data,
                          usv_summary_df=usv_summary_df,
                          payload=cluster_data["usv_ibi_bout_onset"]["f1"])
    ax_session, ax_zoom, ax_scatter = figure.axes
    assert "0 bouts" in ax_session.get_title(loc="left")
    assert ax_zoom.get_xlim() == pytest.approx(
        (0.0, maker.visualizations_parameter_dict["neuronal_tuning_figures"]["bout_zoom_seconds"]))
    assert [t.get_text() for t in ax_scatter.texts] == ["too few multi-USV bouts"]

    peth_figure = Figure(figsize=(8, 4))
    ax_raster, ax_peth = peth_figure.subplots(1, 2)
    maker._draw_section_a(ax_raster=ax_raster, ax_peth=ax_peth, emitter="m1",
                          cluster_data=cluster_data, usv_summary_df=None)
    assert "raster omitted" in ax_raster.texts[0].get_text()
    assert ax_peth.get_title() == "PETH (n_USVs = 57)"


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_draw_vocal_page_closes_its_figure_when_drawing_fails():
    """
    Description
    -----------
    The vocal page's figure is closed by `save_fig` only once drawing has finished; when a
    panel raises, the page must close the figure before the error propagates, or each failing
    cluster (the per-cluster loop catches the error and moves on) leaks one large figure.

    Parameters
    ----------

    Returns
    -------
    None
    """

    maker = _make_figure_maker()
    open_before = set(plt.get_fignums())
    with pytest.raises(KeyError):
        maker._draw_vocal_page(emitter="m1", cluster_data={}, usv_summary_df=None,
                               segmentation={}, save_fig=lambda fig, _label: plt.close(fig),
                               page_label="vocal_m1")
    assert set(plt.get_fignums()) == open_before


def test_failed_cluster_render_closes_the_figures_it_opened(tmp_path, monkeypatch):
    """
    Description
    -----------
    When rendering a cluster raises, the per-cluster loop reports it and moves on; the figures
    that cluster opened (a behavioral page drawn halfway, for example) are closed rather than
    left open for the rest of the run, while figures open before the run are left alone. After
    the loop a RuntimeError names every failed cluster.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Pytest-provided per-test temp directory.
    monkeypatch (pytest.MonkeyPatch)
        Replaces the segmentation loader and the vocal renderer.

    Returns
    -------
    None
    """

    tuning_dir = tmp_path / "ephys" / "tuning_curves"
    tuning_dir.mkdir(parents=True)
    for cluster in ("imec0_cl0001", "imec0_cl0002"):
        with (tuning_dir / f"{cluster}_tuning_curves_data.pkl").open("wb") as fh:
            pickle.dump({"usv_peth": {}}, fh)

    def _half_drawn_page(_self, **_kwargs):
        """Open a page figure and fail before it is saved."""
        plt.figure()
        msg = "panel failed"
        raise ValueError(msg)

    messages: list[str] = []
    maker = NeuronalTuningFigureMaker(
        root_directory=str(tmp_path),
        visualizations_parameter_dict=_make_visualizations_parameters(),
        message_output=messages.append,
    )
    monkeypatch.setattr(NeuronalTuningFigureMaker, "_load_segmentation", lambda _self: {})
    monkeypatch.setattr(NeuronalTuningFigureMaker, "_render_vocal_pages", _half_drawn_page)
    monkeypatch.setattr(tuning_figures, "smart_wait", lambda **_kwargs: None)
    kept = plt.figure()
    open_before = set(plt.get_fignums())
    # both clusters are attempted, then the run fails loudly naming both (the broad handler
    # used to log each failure and report a finished run)
    with pytest.raises(RuntimeError, match=r"failed for 2 of 2 clusters.*imec0_cl0001.*imec0_cl0002"):
        maker.make_neuronal_tuning_figures()
    assert sum("failed to render" in message for message in messages) == 2
    assert set(plt.get_fignums()) == open_before
    plt.close(kept)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("density_levels", [[0.5, 0.9], [0.9, 0.5, 0.75]])
def test_first_vs_later_contour_styles_match_their_legend_entries(density_levels):
    """
    Description
    -----------
    Each density contour is drawn in the line style its legend entry shows for the same share
    of bouts. The contours and the legend were paired in different orders, which agreed only
    for exactly three ascending shares: with two shares, or shares out of order, a contour
    carried another share's style.

    Parameters
    ----------
    density_levels (list)
        Bout shares the contours enclose.

    Returns
    -------
    None
    """

    rng = np.random.default_rng(4)
    usv_starts = np.sort(rng.uniform(0.0, 300.0, 90))
    runs = [np.arange(3 * k, 3 * k + 3) for k in range(30)]
    spike_times = np.sort(rng.uniform(0.0, 300.0, 3000))
    fig, ax = plt.subplots()
    contour_calls, legend_styles = [], {}
    draw_contour, draw_line = ax.contour, ax.plot

    def _record_contour(*args, **kwargs):
        """Record the level and line style of every contour drawn."""
        contour_calls.append((float(kwargs["levels"][0]), kwargs["linestyles"]))
        return draw_contour(*args, **kwargs)

    def _record_line(*args, **kwargs):
        """Record the line style of every labelled legend proxy."""
        if "label" in kwargs:
            legend_styles[kwargs["label"]] = kwargs["linestyle"]
        return draw_line(*args, **kwargs)

    ax.contour, ax.plot = _record_contour, _record_line
    NeuronalTuningFigureMaker._draw_first_vs_later(
        ax, spike_times, usv_starts, runs,
        {"response_window_seconds": 0.2, "response_kernel_seconds": 0.05}, "#9AC0CD",
        {"bout_density_grid": 60, "bout_density_levels": density_levels,
         "ink_color": "#1F1F1F", "dash_color": "#888888"})
    plt.close(fig)
    assert len(contour_calls) == len(density_levels)
    # a larger share of bouts is enclosed by a LOWER density level
    by_level = [style for _level, style in sorted(contour_calls, reverse=True)]
    expected = [legend_styles[f"{100 * share:.0f}%"] for share in sorted(density_levels)]
    assert by_level == expected


def test_figure_maker_requires_the_figures_dpi_and_seed(tmp_path):
    """A settings dict without ``figures.dpi`` / ``figures.seed`` is a configuration error; the
    figure maker used to substitute 300 dpi / seed 0 silently and must now raise KeyError."""
    viz = _make_visualizations_parameters()
    del viz["figures"]["seed"]
    with pytest.raises(KeyError, match="seed"):
        NeuronalTuningFigureMaker(root_directory=str(tmp_path), visualizations_parameter_dict=viz,
                                  message_output=lambda *_a: None)


def _capture_saved_figure(monkeypatch, captured: dict) -> None:
    """
    Description
    -----------
    Replace the module's `save_figure` with a stand-in that hands the figure to the test
    instead of writing it, so the artists the settings drove can be inspected before the
    figure method closes the figure. The stand-in stores the figure under `captured["fig"]`
    and the requested stem under `captured["stem"]`, and returns a placeholder path.

    Parameters
    ----------
    monkeypatch (pytest.MonkeyPatch)
        Replaces `make_neuronal_tuning_figures.save_figure`.
    captured (dict)
        Receives the `fig` and `stem` the figure method saved.

    Returns
    -------
    None
    """

    def _fake_save_figure(fig, stem, **_kwargs):
        """
        Description
        -----------
        Record the figure and its stem instead of writing a file.

        Parameters
        ----------
        fig (matplotlib.figure.Figure)
            Figure the method would save.
        stem (str)
            File stem the method requested.
        **_kwargs (dict)
            The settings and the output directory / format overrides (unused).

        Returns
        -------
        out_path (pathlib.Path)
            A placeholder path named after the stem.
        """

        captured["fig"] = fig
        captured["stem"] = stem
        return pathlib.Path(f"{stem}.png")

    monkeypatch.setattr(tuning_figures, "save_figure", _fake_save_figure)


def test_unreadable_cluster_pickle_fails_the_run_naming_the_load_error(tmp_path, monkeypatch):
    """
    Description
    -----------
    A per-cluster pickle that cannot be loaded is reported, collected, and the run raises a
    RuntimeError after the loop naming the cluster file and the load error type, instead of
    the failure disappearing into a log line while the run reports success.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Pytest-provided per-test temp directory.
    monkeypatch (pytest.MonkeyPatch)
        Replaces the segmentation loader and the wait helper.

    Returns
    -------
    None
    """

    tuning_dir = tmp_path / "ephys" / "tuning_curves"
    tuning_dir.mkdir(parents=True)
    (tuning_dir / "imec0_cl0007_tuning_curves_data.pkl").write_bytes(b"not a pickle")
    messages: list[str] = []
    maker = NeuronalTuningFigureMaker(
        root_directory=str(tmp_path),
        visualizations_parameter_dict=_make_visualizations_parameters(),
        message_output=messages.append,
    )
    monkeypatch.setattr(NeuronalTuningFigureMaker, "_load_segmentation", lambda _self: {})
    monkeypatch.setattr(tuning_figures, "smart_wait", lambda **_kwargs: None)
    with pytest.raises(RuntimeError, match=r"failed for 1 of 1 clusters.*imec0_cl0007_tuning_curves_data\.pkl \(load: UnpicklingError"):
        maker.make_neuronal_tuning_figures()
    assert sum("failed to load imec0_cl0007_tuning_curves_data.pkl" in message for message in messages) == 1
    assert not list(tuning_dir.glob("*_neuronal_tuning.*"))


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_vmi_stability_bootstrap_interval_uses_the_configured_quantiles(tmp_path, monkeypatch):
    """
    Description
    -----------
    The per-region bootstrap interval of the cross-condition correlation panel is read at the
    `vmi_bootstrap_quantiles` of the `neuronal_tuning_figures` settings: the quantile call
    receives exactly the configured pair, the errorbar drawn on the region's bar spans exactly
    the two quantiles it returned, and a region below `stability_min_units` gets no bar.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Pytest-provided per-test temp directory.
    monkeypatch (pytest.MonkeyPatch)
        Replaces the collector, `save_figure` and `np.nanquantile` (recorded, not changed).

    Returns
    -------
    None
    """

    rng = np.random.default_rng(3)
    vmi_a = rng.uniform(-0.8, 0.8, 12)
    vmi_b = 0.6 * vmi_a + rng.normal(0.0, 0.2, 12)
    per_group = {region: [] for region in tuning_figures.VMI_REGION_ORDER}
    per_group["PAG"] = [{"vmi_a": float(a), "vmi_b": float(b), "sig_a": bool(k % 2), "sig_b": bool(k % 3),
                         "fr_baseline_median": 1.0 + k} for k, (a, b) in enumerate(zip(vmi_a, vmi_b, strict=True))]
    per_group["MRN"] = per_group["PAG"][:2]
    maker = _aggregate_maker(tmp_path)
    maker.visualizations_parameter_dict = {
        **maker.visualizations_parameter_dict,
        "neuronal_tuning_figures": {
            **maker.visualizations_parameter_dict["neuronal_tuning_figures"],
            "vmi_bootstrap_n": 200,
            "stability_min_units": 5,
            "vmi_bootstrap_quantiles": [0.1, 0.9],
        },
    }
    monkeypatch.setattr(maker, "_collect_vmi_per_condition_medians", lambda **_kwargs: per_group)
    quantile_calls: list[tuple] = []
    original_nanquantile = np.nanquantile

    def _recording_nanquantile(values, q, *args, **kwargs):
        """
        Description
        -----------
        Forward to `np.nanquantile` and record the requested quantiles and the result.

        Parameters
        ----------
        values (np.ndarray)
            Values to take quantiles of.
        q (array-like)
            Requested quantiles.
        *args, **kwargs
            Forwarded unchanged.

        Returns
        -------
        result (np.ndarray)
            The quantiles `np.nanquantile` returns.
        """

        result = original_nanquantile(values, q, *args, **kwargs)
        quantile_calls.append((q, np.asarray(result, dtype=float)))
        return result

    monkeypatch.setattr(np, "nanquantile", _recording_nanquantile)
    captured: dict = {}
    _capture_saved_figure(monkeypatch, captured)
    maker.make_vmi_cross_condition_stability_figure(triage_pkl_path=tmp_path / "unused.pkl")
    monkeypatch.setattr(np, "nanquantile", original_nanquantile)

    assert captured["stem"] == "vmi_cross_condition_stability"
    assert len(quantile_calls) == 1
    requested, (lo, hi) = quantile_calls[0]
    assert list(requested) == [0.1, 0.9]
    ax_r = next(ax for ax in captured["fig"].axes if ax.get_title() == "cross-condition VMI correlation")
    segments = next(c for c in ax_r.collections if isinstance(c, matplotlib.collections.LineCollection)).get_segments()
    pag_index = tuning_figures.VMI_REGION_ORDER.index("PAG")
    assert segments[pag_index][0][1] == pytest.approx(lo)
    assert segments[pag_index][1][1] == pytest.approx(hi)
    mrn_index = tuning_figures.VMI_REGION_ORDER.index("MRN")
    assert np.isnan(ax_r.patches[mrn_index].get_height())
    assert [label.get_text() for label in ax_r.get_xticklabels()][mrn_index] == "MRN\nN=2"


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_peth_timing_axes_without_units_span_the_configured_peth_window(tmp_path, monkeypatch):
    """
    Description
    -----------
    With no consistent unit in any region, the shared time axis of the PETH-timing figure is
    exactly the `peth_window_seconds` of the shipped analyses settings padded by 2 % on each
    side (nothing widens it), and the peak-z axis falls back to [0, 1] padded by 5 %.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Pytest-provided per-test temp directory.
    monkeypatch (pytest.MonkeyPatch)
        Replaces the consistency collector and `save_figure`.

    Returns
    -------
    None
    """

    with (pathlib.Path(tuning_figures.__file__).resolve().parent.parent / "_parameter_settings"
          / "analyses_settings.json").open(encoding="utf-8") as settings_file:
        window_lo, window_hi = json.load(settings_file)["calculate_neuronal_tuning_curves"]["peth_window_seconds"]
    maker = _aggregate_maker(tmp_path)
    monkeypatch.setattr(maker, "_collect_consistent_peth",
                        lambda **_kwargs: {region: [] for region in tuning_figures.VMI_REGION_ORDER})
    captured: dict = {}
    _capture_saved_figure(monkeypatch, captured)
    maker.make_peth_timing_distribution_figure(triage_pkl_path=tmp_path / "unused.pkl", direction="suppress")

    assert captured["stem"] == "peth_suppress_timing_distribution"
    pad = 0.02 * (window_hi - window_lo)
    axes = captured["fig"].axes
    assert len(axes) == 2 * len(tuning_figures.VMI_REGION_ORDER)
    for ax in axes:
        assert ax.get_xlim() == pytest.approx((window_lo - pad, window_hi + pad))
    assert axes[1].get_ylim() == pytest.approx((-0.05, 1.05))
    assert axes[0].get_title() == "PAG  (N=0)"


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_render_behavioral_pages_skips_excluded_features_and_titles_unknown_pairs_plainly():
    """
    Description
    -----------
    A feature listed in `PLOT_EXCLUDED_BEHAVIORAL_FEATURES` gets no panel, and a directional
    SEI feature whose pair names an animal absent from the session's mouse list is titled by
    its plain display name (no direction arrow) instead of failing the page.

    Parameters
    ----------

    Returns
    -------
    None
    """

    excluded = tuning_figures.PLOT_EXCLUDED_BEHAVIORAL_FEATURES[0]
    cluster_data = {
        "beh_offset=0s": {
            "m1.speed":            _beh_1d_payload(),
            "m1-m3.orofacial-sei": _beh_1d_payload(),
            f"m1-m2.{excluded}":   _beh_1d_payload(),
        },
        "behavioral_metadata": {"behavioral_min_occupancy_seconds": 0.1},
    }
    maker = NeuronalTuningFigureMaker(
        root_directory="/tmp",
        visualizations_parameter_dict=_make_aggregate_visualizations_parameters(),
        message_output=lambda *_args, **_kwargs: None,
    )
    titles: dict[str, list[str]] = {}

    def _save_fig(fig, label: str) -> None:
        """
        Description
        -----------
        Record every non-empty axis title on the page under its label, then close it.

        Parameters
        ----------
        fig (matplotlib.figure.Figure)
            Rendered page.
        label (str)
            Page label.

        Returns
        -------
        None
        """

        titles[label] = [ax.get_title() for ax in fig.axes if ax.get_title()]
        plt.close(fig)

    maker._render_behavioral_pages(cluster_data=cluster_data, mouse_id_list=["m1", "m2"],
                                   mouse_colors=["#9AC0CD", "#8CA252"], save_fig=_save_fig)
    social_titles = next(page_titles for label, page_titles in titles.items() if label.endswith("social"))
    assert social_titles == [maker._feature_title("orofacial-sei")]
    assert "rightarrow" not in social_titles[0]
    all_titles = [title for page_titles in titles.values() for title in page_titles]
    assert maker._feature_title(excluded) not in all_titles


def test_feature_title_falls_back_to_the_opened_out_name():
    """
    Description
    -----------
    A feature the display-name map carries is titled by its display name whatever its role
    prefix, and a feature the map does not carry is titled by its own name with underscores
    replaced by spaces.

    Parameters
    ----------

    Returns
    -------
    None
    """

    maker = _make_figure_maker()
    known = next(iter(maker.feature_display_names))
    assert maker._feature_title(f"self.{known}") == maker.feature_display_names[known]
    assert "brand_new_feature" not in maker.feature_display_names
    assert maker._feature_title("other.brand_new_feature") == "brand new feature"


def test_cluster_spike_times_searches_the_rendered_then_the_recorded_root(tmp_path, monkeypatch):
    """
    Description
    -----------
    `_cluster_spike_times` returns no spikes for a payload without a cluster id; skips a root
    whose search raises OSError and falls through to the recorded session root, returning that
    file's sorted row-0 spike times; and returns no spikes when neither root holds the file.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Pytest-provided per-test temp directory.
    monkeypatch (pytest.MonkeyPatch)
        Makes the search under the rendered root raise OSError.

    Returns
    -------
    None
    """

    rendered_root = tmp_path / "rendered"
    rendered_root.mkdir()
    recorded_root = tmp_path / "recorded"
    cluster_dir = recorded_root / "ephys" / "imec1" / "cluster_data"
    cluster_dir.mkdir(parents=True)
    np.save(cluster_dir / "imec1_cl0042.npy", np.vstack([[3.0, 1.0, 2.0], [0.0, 0.0, 0.0]]))
    maker = NeuronalTuningFigureMaker(root_directory=str(rendered_root),
                                      visualizations_parameter_dict=_make_visualizations_parameters(),
                                      message_output=lambda *_args, **_kwargs: None)

    empty = maker._cluster_spike_times({"usv_metadata": {"cluster_id": "", "session_root": str(recorded_root)}})
    assert empty.size == 0
    assert empty.dtype == float

    original_rglob = pathlib.Path.rglob

    def _rglob_refusing_rendered_root(self, pattern):
        """
        Description
        -----------
        Raise PermissionError when searching the rendered root, search normally elsewhere.

        Parameters
        ----------
        self (pathlib.Path)
            Directory being searched.
        pattern (str)
            Glob pattern.

        Returns
        -------
        matches (generator)
            The normal `rglob` matches outside the rendered root.
        """

        if self == rendered_root:
            msg = "permission denied"
            raise PermissionError(msg)
        return original_rglob(self, pattern)

    monkeypatch.setattr(pathlib.Path, "rglob", _rglob_refusing_rendered_root)
    found = maker._cluster_spike_times({"usv_metadata": {"cluster_id": "imec1_cl0042",
                                                         "session_root": str(recorded_root)}})
    monkeypatch.setattr(pathlib.Path, "rglob", original_rglob)
    np.testing.assert_array_equal(found, [1.0, 2.0, 3.0])

    missing = maker._cluster_spike_times({"usv_metadata": {"cluster_id": "imec1_cl0099", "session_root": None}})
    assert missing.size == 0


def test_first_vs_later_skips_single_usv_bouts_and_refuses_too_many_levels():
    """
    Description
    -----------
    Single-USV bouts have no later USVs and are left out of the first-versus-later scatter, so
    two multi-USV bouts among many single ones still read "too few multi-USV bouts"; and more
    `bout_density_levels` than there are contour line styles is a configuration error that
    raises ValueError naming the setting.

    Parameters
    ----------

    Returns
    -------
    None
    """

    rng = np.random.default_rng(5)
    usv_starts = np.sort(rng.uniform(0.0, 300.0, 90))
    spike_times = np.sort(rng.uniform(0.0, 300.0, 3000))
    cfg = {"response_window_seconds": 0.2, "response_kernel_seconds": 0.05}
    bout_cfg = {"bout_density_grid": 30, "bout_density_levels": [0.5],
                "ink_color": "#1F1F1F", "dash_color": "#888888"}
    singles = [np.array([k]) for k in range(10, 40)]
    fig, ax = plt.subplots()
    NeuronalTuningFigureMaker._draw_first_vs_later(ax, spike_times, usv_starts,
                                                   [np.arange(0, 3), np.arange(3, 6), *singles],
                                                   cfg, "#9AC0CD", bout_cfg)
    assert [t.get_text() for t in ax.texts] == ["too few multi-USV bouts"]
    assert ax.texts[0].get_color() == "#888888"
    plt.close(fig)

    too_many = {**bout_cfg, "bout_density_levels": [0.25, 0.5, 0.75, 0.9]}
    runs = [np.arange(3 * k, 3 * k + 3) for k in range(30)]
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match=r"bout_density_levels holds 4 levels; at most 3"):
        NeuronalTuningFigureMaker._draw_first_vs_later(ax, spike_times, usv_starts, runs, cfg,
                                                       "#9AC0CD", too_many)
    plt.close(fig)


def test_bout_raster_without_spikes_draws_bouts_over_the_configured_window(tmp_path):
    """
    Description
    -----------
    When the unit's spike file cannot be found the bout-triggered raster still draws every
    bout's block but no spike rows, over the `bout_trigger_window_seconds` x-range the
    settings give (here a non-default window). Every bout block is a solid rectangle in the
    emitter's colour (no colour ramp, so no images), one row tall and inside the window.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Pytest-provided per-test temp directory.

    Returns
    -------
    None
    """

    maker, cluster_data, usv_summary_df = _bout_onset_figure_inputs(tmp_path)
    (tmp_path / "ephys" / "imec0" / "cluster_data" / "imec0_cl0001.npy").unlink()
    maker.visualizations_parameter_dict = {
        **maker.visualizations_parameter_dict,
        "neuronal_tuning_figures": {
            **maker.visualizations_parameter_dict["neuronal_tuning_figures"],
            "bout_trigger_window_seconds": [-0.5, 1.5],
        },
    }
    figure = Figure(figsize=(16, 9))
    ax = figure.add_subplot(_bout_page_grid(figure)[0, 0])
    maker._draw_bout_raster(ax=ax, emitter="m1", usv_summary_df=usv_summary_df, cluster_data=cluster_data)
    assert not any(isinstance(c, matplotlib.collections.EventCollection) for c in ax.collections)
    assert not ax.images
    blocks = [patch for patch in ax.patches if isinstance(patch, matplotlib.patches.Rectangle)]
    assert blocks
    emitter_rgba = matplotlib.colors.to_rgba(maker._sex_color(cluster_data["usv_ibi_bout_onset"]["m1"]["sex"]))
    for block in blocks:
        assert block.get_facecolor() == pytest.approx(emitter_rgba)
        assert block.get_height() == pytest.approx(1.0)
        assert block.get_x() >= -0.5 - 1e-9
        assert block.get_x() + block.get_width() <= 1.5 + 1e-9
    assert ax.get_xlim() == pytest.approx((-0.5, 1.5))
    assert ax.get_ylim() == pytest.approx((-0.5, 18.5))


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_watershed_label_sits_in_the_largest_piece_of_a_split_category():
    """
    Description
    -----------
    A category split into two pieces (as one that wraps the embedding boundary is in the
    unwrapped view) carries its ID label inside its larger piece, held inside the frame by the
    `qlvm_label_inset` share of the span, at the `qlvm_category_label_fontsize` and in the
    `ink_color` of the settings.

    Parameters
    ----------

    Returns
    -------
    None
    """

    maker = _make_figure_maker()
    settings = maker.visualizations_parameter_dict["neuronal_tuning_figures"]
    side = 40
    xx, yy = np.meshgrid(np.linspace(0.0, 1.0, side), np.linspace(0.0, 1.0, side))
    label_grid = np.full((side, side), 2, dtype=int)
    label_grid[:, :4] = 1
    label_grid[:, 20:] = 1
    seg = {"label_grid": label_grid, "xx": xx, "yy": yy, "bounds": (0.0, 1.0, 0.0, 1.0)}
    fig, ax = plt.subplots()
    maker._draw_categorical_watershed(ax=ax, seg=seg, categories=[1, 2], values=np.array([2.0, 4.0]),
                                      cbar_label="rate", fig=fig, cmap="inferno", annotate_categories=True)
    labels = {text.get_text(): text for text in ax.texts}
    plt.close(fig)
    assert set(labels) == {"1", "2"}
    label_x, label_y = labels["1"].get_position()
    assert xx[0, 20] <= label_x <= 1.0 - settings["qlvm_label_inset"]
    assert settings["qlvm_label_inset"] <= label_y <= 1.0 - settings["qlvm_label_inset"]
    assert labels["1"].get_fontsize() == settings["qlvm_category_label_fontsize"]
    assert labels["1"].get_color() == settings["ink_color"]
