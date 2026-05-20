"""
@author: bartulem
Makes per-cluster neuronal tuning figures: a single multi-page output
combining the behavioral feature tuning grid (one page per temporal
offset) and the vocal pages (Page 1: bout raster + pooled `usv_peth`
on top, `usv_property_tuning` 4x4 grid below; Page 2:
`usv_category_tuning` watersheds + `usv_category_peth` grid). Output
format is configurable via `neuronal_tuning_figures.fig_format` in
visualizations_settings.json (`pdf` default; PDF is multi-page in one
file, other formats produce one file per page).

Output:
  ephys/tuning_curves/{cluster_id}_neuronal_tuning.{fig_format}
  (or, for non-PDF formats, ..._neuronal_tuning_p{N}_{label}.{fig_format})
"""

from __future__ import annotations

import contextlib
import pathlib
import pickle
import warnings
from datetime import datetime

import h5py
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import polars as pls
from matplotlib import gridspec
from matplotlib import patheffects as mpe
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

from ..analyses.compute_behavioral_features import FeatureZoo
from ..analyses.compute_neuronal_tuning_curves import (
    CONTINUOUS_PROPERTIES,
    CATEGORICAL_FEATURES,
)
from ..analyses.decode_experiment_label import extract_information
from ..os_utils import first_match_or_raise
from ..time_utils import is_gui_context, smart_wait
from .auxiliary_plot_functions import choose_animal_colors, create_colormap

for _ttf in (
    "Helvetica.ttf",
    "Helvetica-Bold.ttf",
    "Helvetica-Oblique.ttf",
    "Helvetica-BoldOblique.ttf",
    "Helvetica-Light.ttf",
):
    fm.fontManager.addfont(pathlib.Path(__file__).parent.parent / "fonts" / _ttf)
plt.style.use(pathlib.Path(__file__).parent.parent / "_config/usv_playpen.mplstyle")


# Display-unit conversions: vocal_boundaries store CSV-native units; the
# plotter shows ms / kHz / etc. for readability. Axis-label strings live
# on FeatureZoo.vocal_labels (e.g. "low -- (kHz) -- high"), inherited via
# the class hierarchy and accessed as self.vocal_labels[prop] at draw time.
DISPLAY_FACTOR = {
    "duration": 1000.0,
    "mean_freq_hz": 1.0 / 1000.0,
    "peak_freq_hz": 1.0 / 1000.0,
    "freq_bandwidth_hz": 1.0 / 1000.0,
    "mean_amplitude": 1.0,
    "max_amplitude": 1.0,
    "spectral_entropy": 1.0,
    "mask_number": 1.0,
}

# Section-(b) feature ordering: 2 features per row × 4 rows
PROPERTY_ROW_ORDER = (
    ("duration", "mean_freq_hz"),
    ("peak_freq_hz", "freq_bandwidth_hz"),
    ("mean_amplitude", "max_amplitude"),
    ("spectral_entropy", "mask_number"),
)

# Page-2 section (c) layout: each row is one segmentation method (VAE
# in the first row, QLVM in the second), and each row holds the
# `_category` AND `_supercategory` variants side-by-side, contributing
# 3 cells (rate / occupancy / strip) each — so 6 cells per row in
# total, matching the behavioral feature grid's density.
SECTION_C_ROWS = (
    ("vae_category", "vae_supercategory"),
    ("qlvm_category", "qlvm_supercategory"),
)

# Hex palette
COLOR_BLACK = "#1F1F1F"
COLOR_GRAY_BAND = "#D3D3D3"
COLOR_GRAY_DASH = "#888888"
COLOR_LIGHT = "#CCCCCC"
COLOR_HATCH = "#BBBBBB"

# Page sizes are fixed by the layout invariants of each page (Page 1 has
# the section-(a) raster + usv_peth on top of the 4×4 usv_property_tuning grid; Page 2 has
# section-(c) 2×6 above section-(d) flowing 6 cols/row). They scale with
# the page dimensions, not with anything user-tunable, so are constants
# rather than settings.
VOCAL_PAGE1_FIGSIZE_INCHES = (16, 22)
VOCAL_PAGE2_FIGSIZE_INCHES = (16, 22)

# Section-(c) per-category strip plot scaling: switch to symlog when
# (max / min) firing-rate dynamic range exceeds the threshold; the
# linthresh defines the linear region around 0 in symlog units. Both are
# numerical knobs of the rendering algorithm rather than artist-tunable
# preferences.
VOCAL_STRIP_LOG_RATIO_THRESHOLD = 10.0
VOCAL_STRIP_SYMLOG_LINTHRESH = 0.5


def _decide_strip_xscale(
    observed: np.ndarray,
    null_p0_5: np.ndarray,
    null_p99_5: np.ndarray,
    log_ratio_threshold: float,
) -> str:
    """
    Description
    -----------
    Decide whether the per-category strip x-axis should be linear or
    symlog based on dynamic range. Min is computed only over strictly
    positive values to avoid division by zero; the presence of zeros
    does not by itself force linear scaling.

    Parameters
    ----------
    observed (np.ndarray)
        Per-category observed firing rates.
    null_p0_5, null_p99_5 (np.ndarray)
        Per-category shuffle 0.5 / 99.5 percentile bounds.
    log_ratio_threshold (float)
        If max/min(positive) > this, use symlog.

    Returns
    -------
    scale (str)
        "linear" or "symlog".
    """

    all_values = np.concatenate(
        [
            np.atleast_1d(observed).astype(float),
            np.atleast_1d(null_p0_5).astype(float),
            np.atleast_1d(null_p99_5).astype(float),
        ]
    )
    finite = all_values[np.isfinite(all_values)]
    if finite.size == 0:
        return "linear"
    positive = finite[finite > 0]
    if positive.size == 0:
        return "linear"
    if (finite.max() / positive.min()) > log_ratio_threshold:
        return "symlog"
    return "linear"


class NeuronalTuningFigureMaker(FeatureZoo):
    """
    Description
    -----------
    Per-cluster combined neuronal-tuning figure renderer. Reads each
    cluster's `*_tuning_curves_data.pkl` (carrying the unified behavioral
    + vocal payload) and emits one multi-page output per cluster: the
    behavioral feature pages (one per temporal offset, per plot-feature
    group) followed by the vocal pages (Page 1: bout raster + pooled
    `usv_peth` on top, `usv_property_tuning` 4x4 grid below; Page 2:
    `usv_category_tuning` watersheds + `usv_category_peth` grid). Pkls
    with neither behavioral nor vocal payload are skipped silently.
    Output filename is `{cluster_id}_neuronal_tuning.{fig_format}` (or,
    for non-PDF formats, `..._p{N}_{label}.{fig_format}`).

    Parameters
    ----------
    root_directory (str)
        Session root directory.
    visualizations_parameter_dict (dict)
        Settings dictionary; reads `male_colors`, `female_colors`, and
        the `neuronal_tuning_figures` block.
    message_output (Callable)
        Logger; defaults to print.
    """

    def __init__(self, **kwargs):
        """
        Description
        -----------
        Initialize the per-cluster figure maker. Loads `FeatureZoo`
        feature definitions (vocal boundaries / vocal labels / display
        units), stashes any keyword arguments as attributes (notably
        `root_directory`, `visualizations_parameter_dict`, and
        `message_output`), records GUI-vs-CLI context, pins the path of
        the bundled UMAP segmentation file used by section (c), and
        primes a lazy segmentation cache.

        Parameters
        ----------
        **kwargs
            Forwarded as-is to `self.__dict__`. Expected keys include
            `root_directory`, `visualizations_parameter_dict`,
            `message_output`.

        Returns
        -------
        None
        """

        FeatureZoo.__init__(self)
        for kw_arg, kw_val in kwargs.items():
            self.__dict__[kw_arg] = kw_val
        self.app_context_bool = is_gui_context()
        self._segmentation_path = (
            pathlib.Path(__file__).parent.parent
            / "_config"
            / "vocal_umap_segmentation.npz"
        )
        self._segmentation_cache: dict | None = None

    # segmentation loading (lazy)

    def _load_segmentation(self) -> dict:
        """
        Description
        -----------
        Lazy-load the bundled UMAP segmentation file
        (`_config/vocal_umap_segmentation.npz`) used to render the
        section-(c) categorical watersheds. Returns an empty dict if
        the file is absent. Cached on first call.

        Parameters
        ----------

        Returns
        -------
        seg (dict)
            Mapping `cat_feat -> {label_grid, xx, yy, bounds,
            unique_labels}` for each categorical feature present in
            the segmentation file.
        """

        if self._segmentation_cache is not None:
            return self._segmentation_cache
        if not self._segmentation_path.exists():
            self._segmentation_cache = {}
            return self._segmentation_cache
        data = np.load(self._segmentation_path, allow_pickle=True)
        out: dict[str, dict] = {}
        for cat_feat in CATEGORICAL_FEATURES:
            if f"{cat_feat}__label_grid" not in data.files:
                continue
            out[cat_feat] = {
                "label_grid": data[f"{cat_feat}__label_grid"],
                "xx": data[f"{cat_feat}__xx"],
                "yy": data[f"{cat_feat}__yy"],
                "bounds": data[f"{cat_feat}__bounds"],
                "unique_labels": data[f"{cat_feat}__unique_labels"].tolist(),
            }
        self._segmentation_cache = out
        return out

    # color resolution

    def _sex_color(self, sex: str) -> str:
        """
        Description
        -----------
        Resolve the per-sex hex color used as the "primary" color for
        an emitter side. Reads `male_colors[0]` / `female_colors[0]`
        from the visualizations settings dict.

        Parameters
        ----------
        sex (str)
            Either "male" or "female"; anything other than "male"
            returns the female color.

        Returns
        -------
        hex_color (str)
            Six-digit hex color string with leading "#".
        """

        if sex == "male":
            return self.visualizations_parameter_dict["male_colors"][0]
        return self.visualizations_parameter_dict["female_colors"][0]

    # main entry point

    def make_neuronal_tuning_figures(self) -> None:
        """
        Description
        -----------
        Iterate over per-cluster pkls under `<root>/ephys/tuning_curves/`,
        and for each cluster render one combined output containing both
        the behavioral feature tuning pages and the vocal pages. Output
        format and per-cluster file naming are controlled by
        `neuronal_tuning_figures.fig_format` (`pdf` produces a single
        multi-page file; other formats produce one file per page). Pkls
        with neither behavioral nor vocal payload are skipped silently.

        Parameters
        ----------

        Returns
        -------
        None
        """

        message_output = self.message_output if hasattr(self, "message_output") else print
        message_output(
            f"Making neuronal tuning figures started at: "
            f"{datetime.now().hour:02d}:{datetime.now().minute:02d}:{datetime.now().second:02d}"
        )
        smart_wait(app_context_bool=self.app_context_bool, seconds=1)

        root = pathlib.Path(self.root_directory)
        tuning_dir = root / "ephys" / "tuning_curves"
        if not tuning_dir.exists():
            message_output(f"  neuronal-figs: {tuning_dir} not found; skipping.")
            return

        pkl_files = sorted(tuning_dir.glob("*_tuning_curves_data.pkl"))
        if not pkl_files:
            message_output(f"  neuronal-figs: no tuning pkls in {tuning_dir}; skipping.")
            return

        viz_params = self.visualizations_parameter_dict["neuronal_tuning_figures"]
        fig_format = str(viz_params.get("fig_format", "pdf")).lower()

        # try to load USV summary for the bout raster (left subplot of section a).
        usv_summary_df = None
        try:
            csv_path = first_match_or_raise(
                root=root,
                pattern="*_usv_summary.csv",
                recursive=True,
                label="USV summary CSV",
            )
            usv_summary_df = pls.read_csv(str(csv_path)).filter(pls.col("vae_supercategory") != 0)
        except (StopIteration, FileNotFoundError):
            pass

        # try to load tracking H5 (for behavioral mouse colors / animal IDs).
        # Search the whole session tree — mirrors `_load_behavioral_inputs`
        # and `_load_vocal_inputs` in compute_neuronal_tuning_curves.py.
        # Earlier revisions restricted to `<root>/video/`, which silently
        # produced an empty mouse_id_list (and therefore zero rendered
        # pages) on session layouts where the tracking H5 sits at the
        # session root instead of inside a `video/` subdirectory.
        mouse_id_list: list[str] = []
        mouse_colors: list = []
        try:
            tracked_file_loc = first_match_or_raise(
                root=root,
                pattern="[!speaker]*_points3d_translated_rotated_metric.h5",
                recursive=True,
                label="translated/rotated mouse points3d .h5",
            )
            with h5py.File(tracked_file_loc, mode="r") as tracking_data_3d:
                mouse_id_list = [
                    elem.decode("utf-8") for elem in tracking_data_3d["track_names"]
                ]
                session_exp_code = (
                    tracking_data_3d["experimental_code"][()].decode("utf-8")
                )
            experiment_info_dict = extract_information(experiment_code=session_exp_code)
            mouse_colors = choose_animal_colors(
                exp_info_dict=experiment_info_dict,
                visualizations_parameter_dict=self.visualizations_parameter_dict,
            )
        except (StopIteration, FileNotFoundError):
            pass

        segmentation = self._load_segmentation()

        # One-time warning when the tracking H5 was not found at all:
        # without it `mouse_id_list` is empty and behavioral pages cannot
        # render for any cluster, which used to fail silently per cluster.
        if not mouse_id_list:
            message_output(
                f"  neuronal-figs: no tracking H5 found under {root}; "
                "behavioral pages will be skipped for every cluster."
            )

        # double the spine thickness for all axes (mplstyle uses 0.75 default).
        spine_lw = 2.0 * float(plt.rcParams["axes.linewidth"])
        for pkl_file in tqdm(pkl_files, desc="neuronal tuning figures"):
            try:
                with pkl_file.open("rb") as fh:
                    cluster_data = pickle.load(fh)
            except Exception as exc:  # noqa: BLE001
                message_output(f"  neuronal-figs: failed to load {pkl_file.name}: {exc}")
                continue
            cluster_id = pkl_file.stem.replace("_tuning_curves_data", "")
            has_behavioral = any(k.startswith("beh_offset=") for k in cluster_data)
            # Vocal payload is written by `_compute_one_cluster_vocal` under
            # top-level keys `usv_peth`, `usv_property_tuning`,
            # `usv_category_tuning`, `usv_category_peth`, `usv_metadata`.
            # An earlier revision of this predicate looked for `vocal_q*`,
            # which matches nothing the compute side ever writes — every
            # cluster would silently skip the vocal half.
            has_vocal = any(k.startswith("usv_") for k in cluster_data)
            # Skip whenever nothing can actually render: behavioral needs
            # both behavioral payload AND a populated mouse_id_list, vocal
            # needs vocal payload. If neither is renderable, do not open a
            # PdfPages — closing one with zero pages raises and the broad
            # except below would then mask the real cause.
            behavioral_renderable = has_behavioral and bool(mouse_id_list)
            if not (behavioral_renderable or has_vocal):
                continue
            out_path = tuning_dir / f"{cluster_id}_neuronal_tuning.{fig_format}"
            try:
                with plt.rc_context({"axes.linewidth": spine_lw}), self._open_save_target(
                    out_path=out_path,
                    cluster_id=cluster_id,
                    fig_format=fig_format,
                ) as save_fig:
                    if has_behavioral and mouse_id_list:
                        self._render_behavioral_pages(
                            cluster_data=cluster_data,
                            mouse_id_list=mouse_id_list,
                            mouse_colors=mouse_colors,
                            save_fig=save_fig,
                        )
                    if has_vocal:
                        self._render_vocal_pages(
                            cluster_data=cluster_data,
                            usv_summary_df=usv_summary_df,
                            segmentation=segmentation,
                            viz_params=viz_params,
                            save_fig=save_fig,
                        )
            except Exception as exc:  # noqa: BLE001
                message_output(
                    f"  neuronal-figs: failed to render {cluster_id}: {exc}"
                )
                continue

    @contextlib.contextmanager
    def _open_save_target(
        self,
        out_path: pathlib.Path,
        cluster_id: str,
        fig_format: str,
    ):
        """
        Description
        -----------
        Yield a `save_fig(fig, label)` callable that persists each
        rendered figure. For PDF, all pages go into one file via
        `PdfPages`. For other formats (PNG / SVG / EPS / etc.), each
        page is saved as its own file with `_p{N}_{label}` suffix.

        Parameters
        ----------
        out_path (pathlib.Path)
            Target output path; the suffix is the chosen `fig_format`.
        cluster_id (str)
            Cluster identifier (used in per-page file names for non-PDF
            formats).
        fig_format (str)
            Extension to save under (lowercase, no leading dot).

        Yields
        ------
        save_fig (Callable[[matplotlib.figure.Figure, str], None])
            Callable that accepts an open figure and a short label and
            commits the figure to the appropriate output target.

        Returns
        -------
        None
        """

        if fig_format == "pdf":
            with PdfPages(out_path) as pdf:
                page_idx = {"n": 0}

                def save_fig(fig, label: str) -> None:  # noqa: ARG001
                    """
                    Description
                    -----------
                    PDF backend: append the rendered figure to the
                    open `PdfPages` document at 200 DPI, then close
                    it. The `label` argument is unused (PDF pages have
                    no per-page filename) but kept for signature
                    parity with the non-PDF branch.

                    Parameters
                    ----------
                    fig (matplotlib.figure.Figure)
                        Figure to commit and close.
                    label (str)
                        Short page label (ignored in this branch).

                    Returns
                    -------
                    None
                    """
                    page_idx["n"] += 1
                    pdf.savefig(fig, dpi=200)
                    plt.close(fig)

                yield save_fig
        else:
            page_idx = {"n": 0}
            out_dir = out_path.parent

            def save_fig(fig, label: str) -> None:
                """
                Description
                -----------
                Non-PDF backend: write the rendered figure to its own
                file at `<cluster_id>_neuronal_tuning_p{N}_{label}.{fig_format}`
                with 200 DPI, then close the figure. `N` is the running
                page index across all pages of this cluster.

                Parameters
                ----------
                fig (matplotlib.figure.Figure)
                    Figure to commit and close.
                label (str)
                    Short page label inserted into the filename.

                Returns
                -------
                None
                """
                page_idx["n"] += 1
                per_page_path = (
                    out_dir
                    / f"{cluster_id}_neuronal_tuning_p{page_idx['n']}_{label}.{fig_format}"
                )
                fig.savefig(per_page_path, dpi=200, bbox_inches="tight")
                plt.close(fig)

            yield save_fig

    # per-cluster page rendering

    # behavioral rendering

    def _render_behavioral_pages(
        self,
        cluster_data: dict,
        mouse_id_list: list[str],
        mouse_colors: list,
        save_fig,
    ) -> None:
        """
        Description
        -----------
        For each behavioral temporal offset key (`beh_offset=*s`) in the
        per-cluster pkl, render one page per plot-feature group
        (`individual.<mouse_id>` and `social`). Each page is a small
        gridspec of (line + occupancy) pairs for 1D features plus
        per-animal 2D spatial ratemap pairs. Smoothing, occupancy
        thresholding, and colorbar placement are handled exactly as the
        legacy behavioral plotter did.

        Parameters
        ----------
        cluster_data (dict)
            Loaded pkl payload (top-level keys include `beh_offset=*s`
            entries plus optional vocal keys).
        mouse_id_list (list[str])
            Animal IDs from the tracking H5; drives per-mouse coloring.
        mouse_colors (list)
            Hex colors per mouse, in the same order as `mouse_id_list`.
        save_fig (Callable[[Figure, str], None])
            Persists each rendered page; closes the figure.

        Returns
        -------
        None
        """

        beh_offset_keys = [k for k in cluster_data if k.startswith("beh_offset=")]
        if not beh_offset_keys:
            return

        mouse_color_dict = {"social": "#5A6470"}
        mouse_colormap_dict: dict = {}
        for mouse_idx, mouse in enumerate(mouse_id_list):
            mouse_color_dict[mouse] = mouse_colors[mouse_idx]
            mouse_colormap_dict[mouse] = create_colormap(
                input_parameter_dict={
                    "cm_length": 255,
                    "cm_name": f"{mouse}",
                    "cm_type": "sequential",
                    "cm_start": (
                        int(mouse_colors[mouse_idx][1:3], 16),
                        int(mouse_colors[mouse_idx][3:5], 16),
                        int(mouse_colors[mouse_idx][5:7], 16),
                    ),
                    "cm_end": (255, 255, 255),
                    "equalize_luminance": True,
                    "match_luminance_by": "max",
                    "change_saturation": 0.5,
                    "cm_opacity": 1,
                }
            )

        plot_features: dict[str, list[str]] = {}
        for feature_key in cluster_data[beh_offset_keys[0]]:
            mouse_id = feature_key.split(".")[0]
            if (
                f"individual.{mouse_id}" not in plot_features
                and "-" not in mouse_id
            ):
                plot_features[f"individual.{mouse_id}"] = []
            if "-" not in mouse_id:
                plot_features[f"individual.{mouse_id}"].append(feature_key)
            else:
                plot_features.setdefault("social", []).append(feature_key)

        viz_params = self.visualizations_parameter_dict["neuronal_tuning_figures"]
        ratemap_cmap = viz_params.get("ratemap_cmap", "inferno")
        # behavioral occupancy threshold lives on the analyses side (it's
        # a tuning-curve config alongside usv_property_min_occupancy_seconds);
        # compute writes it into behavioral_metadata, plotter reads from
        # there with a sane fallback for older pkls.
        occ_threshold_setting = float(
            cluster_data.get("behavioral_metadata", {}).get(
                "behavioral_min_occupancy_seconds", 1.0
            )
        )

        for offset in beh_offset_keys:
            for plot_feature_key, features in plot_features.items():
                if not features:
                    continue
                if "social" in plot_feature_key:
                    rm_color = mouse_color_dict["social"]
                    mouse_colormap = 0
                else:
                    rm_color = mouse_color_dict[plot_feature_key.split(".")[-1]]
                    mouse_colormap = mouse_colormap_dict[
                        plot_feature_key.split(".")[-1]
                    ]

                with warnings.catch_warnings():
                    warnings.simplefilter(action="ignore")
                    row_num = int(np.ceil((len(features) * 2) / 6))
                    # the legacy plotter used (6.4, row_num) with ~4 pt
                    # tick / 5 pt title fonts; bumping fonts to 8/9 pt
                    # without enlarging the figure crushes the layout.
                    # scale both axes ~2.5× to give the new fonts room
                    # and tighten wspace/hspace so cells stay roughly
                    # square.
                    fig = plt.figure(
                        figsize=(16.0, max(2.6, 2.6 * row_num)), tight_layout=False
                    )
                    gs = gridspec.GridSpec(
                        nrows=row_num,
                        ncols=6,
                        wspace=0.40,
                        hspace=0.45,
                        left=0.06,
                        right=0.96,
                        bottom=0.07,
                        top=0.94,
                    )
                    gs_x, gs_y = 0, 0

                    for feature in features:
                        feat_payload = cluster_data[offset][feature]
                        if "space" in feature:
                            cbar_width = 0.005
                            cbar_height = 0.04
                            ratemap_2d = feat_payload.get(
                                "rate_smoothed", feat_payload["rate"]
                            )
                            ax1 = fig.add_subplot(gs[gs_x, gs_y])
                            rm = ax1.imshow(
                                X=ratemap_2d, cmap=ratemap_cmap, vmin=0,
                                interpolation="gaussian", aspect="equal",
                            )
                            ax1.set_title(
                                "Spatial tuning", fontsize=13, pad=8.0,
                                fontweight="bold",
                            )
                            ax1.set_xticks([])
                            ax1.set_yticks([])
                            ax1.set_xlabel("X (cm)", fontsize=10, labelpad=1)
                            ax1.set_ylabel("Y (cm)", fontsize=10, labelpad=1)

                            ax1_pos = gs[gs_x, gs_y].get_position(fig)
                            cb_ax = fig.add_axes(
                                (ax1_pos.x1, ax1_pos.y1 - cbar_height,
                                 cbar_width, cbar_height)
                            )
                            cbar = fig.colorbar(rm, cax=cb_ax, orientation="vertical")
                            cbar_vmin, cbar_vmax = cbar.mappable.get_clim()
                            cbar.set_ticks([cbar_vmin, cbar_vmax])
                            cbar.set_ticklabels(
                                [f"{int(cbar_vmin)}", f"{round(cbar_vmax, 1)}"],
                                fontsize=7,
                            )
                            cbar.ax.tick_params(length=0, pad=0.5)
                            cbar.outline.set_visible(True)

                            ax2 = fig.add_subplot(gs[gs_x, gs_y + 1])
                            occ = ax2.imshow(
                                X=feat_payload["occupancy_seconds"],
                                cmap=mouse_colormap, vmin=0,
                                interpolation="gaussian", aspect="equal",
                            )
                            ax2.set_xticks([])
                            ax2.set_yticks([])
                            ax2.set_xlabel("X (cm)", fontsize=10, labelpad=1)
                            ax2.set_ylabel("Y (cm)", fontsize=10, labelpad=1)

                            ax2_pos = gs[gs_x, gs_y + 1].get_position(fig)
                            cb_ax2 = fig.add_axes(
                                (ax2_pos.x1, ax2_pos.y1 - cbar_height,
                                 cbar_width, cbar_height)
                            )
                            cbar2 = fig.colorbar(occ, cax=cb_ax2, orientation="vertical")
                            cbar2_vmin, cbar2_vmax = cbar2.mappable.get_clim()
                            cbar2.set_ticks([cbar2_vmin, cbar2_vmax])
                            cbar2.set_ticklabels(
                                [f"{int(cbar2_vmin)}", f"{int(np.ceil(cbar2_vmax))}"],
                                fontsize=7,
                            )
                            cbar2.ax.tick_params(length=0, pad=0.5)
                            cbar2.outline.set_visible(True)
                            gs_y += 2
                            if gs_y > 5:
                                gs_y = 0; gs_x += 1
                        else:
                            occ_mask = (
                                feat_payload["occupancy_seconds"] > occ_threshold_setting
                            )
                            low_end_sh = feat_payload.get(
                                "null_p0_5_smoothed", feat_payload["null_p0_5"]
                            )
                            high_end_sh = feat_payload.get(
                                "null_p99_5_smoothed", feat_payload["null_p99_5"]
                            )
                            ratemap = feat_payload.get(
                                "rate_smoothed", feat_payload["rate"]
                            )
                            bin_edges = feat_payload["bin_edges"]
                            bin_centers = feat_payload["bin_centers"]

                            # raw orofacial/anogenital SEI are the only
                            # social features that exist in both directions
                            # (A→B and B→A); disambiguate those two with a
                            # directional mouse-index pair in the title.
                            # Their 1st / 2nd derivatives are also stored
                            # twice but should NOT receive the suffix per
                            # convention; all other social features are
                            # bidirectional (one entry only) and stay as-is.
                            prefix_str, feat_name = feature.split(".", 1)
                            directional_sei = {"orofacial-sei", "anogenital-sei"}
                            if "-" in prefix_str and feat_name in directional_sei:
                                m_a, m_b = prefix_str.split("-", 1)
                                try:
                                    idx_a = mouse_id_list.index(m_a) + 1
                                    idx_b = mouse_id_list.index(m_b) + 1
                                    # the bundled Helvetica.ttf has no
                                    # arrow glyphs (U+2192 etc.), so render
                                    # the arrow via matplotlib mathtext.
                                    title_text = (
                                        rf"{feat_name} ({idx_a}$\rightarrow${idx_b})"
                                    )
                                except ValueError:
                                    title_text = feat_name
                            else:
                                title_text = feat_name

                            ax1 = fig.add_subplot(gs[gs_x, gs_y])
                            # split x/y tick params: push xticklabels down
                            # so the leftmost xtick label no longer kisses
                            # the bottom yticklabel; keep yticks tight.
                            ax1.tick_params(axis="x", length=1.5, pad=4.0, labelsize=10)
                            ax1.tick_params(axis="y", length=1.5, pad=0.5, labelsize=10)
                            ax1.fill_between(
                                bin_centers[occ_mask],
                                low_end_sh[occ_mask],
                                high_end_sh[occ_mask],
                                where=high_end_sh[occ_mask] >= low_end_sh[occ_mask],
                                facecolor=COLOR_GRAY_BAND,
                                interpolate=True,
                            )
                            ax1.plot(
                                bin_centers[occ_mask],
                                ratemap[occ_mask],
                                lw=4.0, ls="-", c=rm_color, alpha=1.0,
                            )
                            ax1.set_title(
                                title_text, fontsize=13, pad=8.0,
                                fontweight="bold",
                            )
                            tx_min, tx_max = ax1.get_xlim()
                            x1_lo = max(
                                self.feature_boundaries[feat_name][0],
                                tx_min,
                            )
                            x1_hi = min(
                                self.feature_boundaries[feat_name][1],
                                tx_max,
                            )
                            ax1.set_xticks(
                                ticks=[x1_lo, x1_hi],
                                labels=[f"{x1_lo:.1f}", f"{x1_hi:.1f}"],
                                fontsize=10,
                            )
                            ax1.set_xlabel(
                                self.feature_labels[plot_feature_key.split(".")[0]][
                                    feat_name
                                ],
                                fontsize=10, labelpad=2,
                            )
                            ty_min, ty_max = ax1.get_ylim()
                            ax1.set_yticks(
                                ticks=[max(np.floor(ty_min), 0), np.ceil(ty_max)],
                                labels=[
                                    f"{max(np.floor(ty_min), 0):.1f}",
                                    f"{ty_max:.1f}",
                                ],
                                fontsize=10,
                            )
                            ax1.set_ylabel(
                                "Firing rate (sp/s)", fontsize=10, labelpad=-2,
                            )
                            ax1.set_box_aspect(1)

                            ax2 = fig.add_subplot(gs[gs_x, gs_y + 1])
                            ax2.tick_params(axis="x", length=1.5, pad=4.0, labelsize=10)
                            ax2.tick_params(axis="y", length=1.5, pad=0.5, labelsize=10)
                            ax2.stairs(
                                values=feat_payload["occupancy_seconds"],
                                edges=bin_edges,
                                fill=True,
                                color=rm_color,
                                edgecolor=COLOR_BLACK,
                                linewidth=0.3,
                            )
                            ax2.set_xticks(
                                ticks=[
                                    self.feature_boundaries[feat_name][0],
                                    self.feature_boundaries[feat_name][1],
                                ],
                                labels=[
                                    f"{self.feature_boundaries[feat_name][0]:.1f}",
                                    f"{self.feature_boundaries[feat_name][1]:.1f}",
                                ],
                                fontsize=10,
                            )
                            ax2.set_xlabel(
                                self.feature_labels[plot_feature_key.split(".")[0]][
                                    feat_name
                                ],
                                fontsize=10, labelpad=2,
                            )
                            _, ty2_max = ax2.get_ylim()
                            ax2.set_yticks(
                                ticks=[0, int(np.ceil(ty2_max))],
                                labels=["0", f"{int(np.ceil(ty2_max))}"],
                                fontsize=10,
                            )
                            ax2.set_ylabel(
                                "Occupancy (s)", fontsize=10, labelpad=-2,
                            )
                            ax2.set_box_aspect(1)
                            gs_y += 2
                            if gs_y > 5:
                                gs_y = 0; gs_x += 1

                save_fig(fig, f"behavioral_{offset}_{plot_feature_key}")

    def _render_vocal_pages(
        self,
        cluster_data: dict,
        usv_summary_df,
        segmentation: dict,
        viz_params: dict,
        save_fig,
    ) -> None:
        """
        Description
        -----------
        For each emitter present in the pkl, render Page 1 (sections a +
        b) and Page 2 (sections c + d) via the supplied `save_fig`
        callable.

        Parameters
        ----------
        cluster_data (dict)
            Loaded pkl payload with keys usv_peth, usv_property_tuning,
            usv_category_tuning, usv_category_peth, usv_metadata.
        usv_summary_df (pls.DataFrame | None)
            Filtered (post `vae_supercategory != 0`) USV summary if
            available; required for the bout raster.
        segmentation (dict)
            Loaded UMAP segmentation per categorical feature.
        viz_params (dict)
            `neuronal_tuning_figures` settings block.
        save_fig (Callable[[Figure, str], None])
            Persists each rendered page; closes the figure.

        Returns
        -------
        None
        """

        for emitter in cluster_data["usv_peth"]:
            sex_label = cluster_data["usv_peth"][emitter].get("sex", "x")
            self._render_page1(
                emitter=emitter,
                cluster_data=cluster_data,
                usv_summary_df=usv_summary_df,
                viz_params=viz_params,
                save_fig=save_fig,
                page_label=f"vocal_a_{sex_label}",
            )
            self._render_page2(
                emitter=emitter,
                cluster_data=cluster_data,
                segmentation=segmentation,
                viz_params=viz_params,
                save_fig=save_fig,
                page_label=f"vocal_b_{sex_label}",
            )

    # Page 1 — sections (a) and (b)

    def _render_page1(
        self,
        emitter: str,
        cluster_data: dict,
        usv_summary_df,
        viz_params: dict,
        save_fig,
        page_label: str,
    ) -> None:
        """
        Description
        -----------
        Render Page 1 of the vocal output for one emitter side: the
        bout raster + pooled `usv_peth` (section a) on top of the
        4×4 `usv_property_tuning` grid (section b), then commit the
        page via `save_fig`.

        Parameters
        ----------
        emitter (str)
            Emitter ID (e.g. mouse track name) keying into the vocal
            payloads.
        cluster_data (dict)
            Loaded per-cluster pkl payload.
        usv_summary_df (pls.DataFrame | None)
            Filtered USV summary (post `vae_supercategory != 0`); used
            by the bout raster. Pass `None` to skip the raster.
        viz_params (dict)
            `neuronal_tuning_figures` settings block.
        save_fig (Callable[[Figure, str], None])
            Persists the rendered figure and closes it.
        page_label (str)
            Short label embedded in non-PDF per-page filenames.

        Returns
        -------
        None
        """

        fig = plt.figure(figsize=VOCAL_PAGE1_FIGSIZE_INCHES, tight_layout=False)
        outer = gridspec.GridSpec(
            nrows=2, ncols=1, height_ratios=[5, 14], hspace=0.30,
            left=0.06, right=0.97, top=0.96, bottom=0.04,
        )

        # section (a)
        gs_a = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0, 0], wspace=0.18)
        ax_raster = fig.add_subplot(gs_a[0, 0])
        ax_peth = fig.add_subplot(gs_a[0, 1])
        self._draw_section_a(
            ax_raster=ax_raster,
            ax_peth=ax_peth,
            emitter=emitter,
            cluster_data=cluster_data,
            usv_summary_df=usv_summary_df,
        )

        # section (b) — 4x4 grid of square usv_property_tuning cells
        gs_b = gridspec.GridSpecFromSubplotSpec(
            4, 4, subplot_spec=outer[1, 0], wspace=0.40, hspace=0.45
        )
        self._draw_section_b(
            fig=fig,
            gs=gs_b,
            emitter=emitter,
            cluster_data=cluster_data,
        )

        save_fig(fig, page_label)

    # section (a) — raster + PETH

    def _draw_section_a(
        self,
        ax_raster,
        ax_peth,
        emitter: str,
        cluster_data: dict,
        usv_summary_df,
    ) -> None:
        """
        Description
        -----------
        Draw section (a) of vocal Page 1: bout raster on the left axes
        (per-bout spike raster + per-USV color bars), pooled `usv_peth`
        on the right axes (peri-USV PETH with shuffle band and
        observed line).

        Parameters
        ----------
        ax_raster (matplotlib.axes.Axes)
            Target axes for the bout raster (left).
        ax_peth (matplotlib.axes.Axes)
            Target axes for the PETH (right).
        emitter (str)
            Emitter ID keying into the vocal payloads.
        cluster_data (dict)
            Per-cluster pkl payload.
        usv_summary_df (pls.DataFrame | None)
            Filtered USV summary; if `None` the raster is skipped.

        Returns
        -------
        None
        """

        sex = cluster_data["usv_peth"][emitter].get("sex", "male")
        line_color = self._sex_color(sex)

        # left: bout raster
        if usv_summary_df is None:
            ax_raster.text(
                0.5, 0.5, "USV summary CSV not available\n(raster omitted)",
                transform=ax_raster.transAxes, ha="center", va="center", fontsize=13, color=COLOR_GRAY_DASH,
            )
            ax_raster.set_xticks([])
            ax_raster.set_yticks([])
        else:
            self._draw_bout_raster(
                ax=ax_raster,
                emitter=emitter,
                usv_summary_df=usv_summary_df,
                cluster_data=cluster_data,
            )

        # right: PETH
        peth = cluster_data["usv_peth"][emitter]
        bin_centers = peth["bin_centers_s"]
        # smoothed versions are pre-computed in compute (smooth-then-percentile);
        # fall back to raw if smoothing was disabled (smoothing_sd == 0).
        rate = peth.get("rate_smoothed", peth["rate"])
        p0_5 = peth.get("null_p0_5_smoothed", peth["null_p0_5"])
        p99_5 = peth.get("null_p99_5_smoothed", peth["null_p99_5"])

        ax_peth.fill_between(
            bin_centers,
            p0_5,
            p99_5,
            where=np.isfinite(p0_5) & np.isfinite(p99_5),
            facecolor=COLOR_GRAY_BAND,
            interpolate=True,
            zorder=1,
        )
        ax_peth.plot(
            bin_centers,
            rate,
            lw=5.6,
            color=line_color,
            zorder=2,
        )
        ax_peth.axvline(x=0.0, color=COLOR_BLACK, ls="--", lw=0.5)
        ax_peth.set_xlim(bin_centers.min(), bin_centers.max())
        # explicit X ticks at 0.25 s spacing so 0 s is always visible
        peth_xticks = np.arange(-2.0, 0.0001, 0.25)
        ax_peth.set_xticks(peth_xticks)
        ax_peth.set_xticklabels([f"{v:g}" for v in peth_xticks])
        ax_peth.set_xlabel("time relative to USV onset (s)", fontsize=12)
        ax_peth.set_ylabel("firing rate (sp/s)", fontsize=12)
        n_anchors = peth.get("n_anchors", 0)
        ax_peth.set_title(f"PETH (n_USVs = {n_anchors})", fontsize=13, pad=3)
        ax_peth.tick_params(labelsize=11)

    def _draw_bout_raster(
        self,
        ax,
        emitter: str,
        usv_summary_df,
        cluster_data: dict,
    ) -> None:
        """
        Description
        -----------
        Draw the bout-onset-aligned raster on `ax`. Each bout (defined as
        a same-emitter USV preceded by ≥ `bout_quiet_seconds` of silence)
        becomes one row showing every USV (any emitter) within
        [-2, +2] s of the bout onset, plus this cluster's spike events.

        Parameters
        ----------
        ax (matplotlib axis)
            Target axis.
        emitter (str)
            Anchor-emitter string for bout definition.
        usv_summary_df (pls.DataFrame)
            Filtered USV summary.
        cluster_data (dict)
            Loaded vocal tuning pkl — needed for spike times and metadata.
            The bout-quiet-seconds setting is read from
            `cluster_data["usv_metadata"]["bout_quiet_seconds"]` so
            that the raster's bout definition matches what compute used.

        Returns
        -------
        None
        """

        bout_quiet_s = cluster_data.get("usv_metadata", {}).get("bout_quiet_seconds", 2.0)

        # find bouts of the anchor emitter
        emitters = [
            (e.strip() if e is not None else None)
            for e in usv_summary_df["emitter"].to_list()
        ]
        starts = usv_summary_df["start"].to_numpy()
        stops = usv_summary_df["stop"].to_numpy()
        anchor_mask = np.array([e == emitter for e in emitters])
        anchor_starts = starts[anchor_mask]
        anchor_stops = stops[anchor_mask]
        if anchor_starts.size == 0:
            ax.text(
                0.5, 0.5, "no anchor USVs",
                transform=ax.transAxes, ha="center", va="center", fontsize=13, color=COLOR_GRAY_DASH,
            )
            return

        order = np.argsort(anchor_starts)
        anchor_starts = anchor_starts[order]
        anchor_stops = anchor_stops[order]
        is_new_bout = np.empty(anchor_starts.size, dtype=bool)
        is_new_bout[0] = True
        is_new_bout[1:] = (anchor_starts[1:] - anchor_stops[:-1]) >= bout_quiet_s
        bout_onsets = anchor_starts[is_new_bout]

        if bout_onsets.size == 0:
            ax.text(
                0.5, 0.5, "no bouts", transform=ax.transAxes, ha="center", va="center",
                fontsize=13, color=COLOR_GRAY_DASH,
            )
            return

        # spike data: the spike file path is not stored in the pkl;
        # locate the cluster's spike .npy under <session_root>/ephys
        # via session_root + cluster_id stored in metadata.
        spike_path_hint = cluster_data.get("usv_metadata", {}).get("session_root", None)
        if spike_path_hint is not None:
            cluster_id = cluster_data.get("usv_metadata", {}).get("cluster_id", None)
            if cluster_id:
                spike_npy_candidates = list(
                    pathlib.Path(spike_path_hint).rglob(f"cluster_data/{cluster_id}.npy")
                )
                if spike_npy_candidates:
                    arr = np.load(spike_npy_candidates[0])
                    spike_times = np.sort(np.asarray(arr[0, :], dtype=float))
                else:
                    spike_times = np.empty(0, dtype=float)
            else:
                spike_times = np.empty(0, dtype=float)
        else:
            spike_times = np.empty(0, dtype=float)

        x_lo, x_hi = -2.0, 2.0

        # USV-bar color: anchor's own sex for anchor-emitter USVs, the
        # opposite sex for any other emitter's USVs falling in the window.
        anchor_sex = cluster_data["usv_peth"][emitter].get("sex", "male")
        anchor_partner_sex = "female" if anchor_sex == "male" else "male"

        # for each bout, draw spikes (black) and USV-bars (sex-colored)
        # within [-2, +2] of the bout onset
        all_starts = starts
        all_stops = stops
        spike_y = []
        spike_x = []
        for row_idx, t0 in enumerate(bout_onsets):
            # USVs in window:
            mask = (all_stops > t0 + x_lo) & (all_starts < t0 + x_hi)
            for j in np.where(mask)[0]:
                s = max(all_starts[j] - t0, x_lo)
                e = min(all_stops[j] - t0, x_hi)
                em = emitters[j]
                if em == emitter:
                    color = self._sex_color(anchor_sex)
                else:
                    color = self._sex_color(anchor_partner_sex) if em is not None else COLOR_LIGHT
                # USV bars at the TOP of each row (just inside the upper edge);
                # spikes occupy the body of the row centered at row_idx + 0.5.
                # plot only anchor-emitter USVs (sex-colored) and
                # unassigned USVs (gray); skip partner-emitter USVs
                # entirely so the raster reflects this side's vocal
                # activity uncluttered by the other animal's calls.
                if em is not None and em != emitter:
                    continue
                ax.hlines(y=row_idx + 0.95, xmin=s, xmax=e, colors=color, linewidth=1.5)

            # spikes in window:
            if spike_times.size > 0:
                lo = np.searchsorted(spike_times, t0 + x_lo, "left")
                hi = np.searchsorted(spike_times, t0 + x_hi, "left")
                rel = spike_times[lo:hi] - t0
                spike_x.extend(rel.tolist())
                spike_y.extend([row_idx + 0.5] * rel.size)

        if spike_x:
            ax.vlines(
                x=spike_x, ymin=np.array(spike_y) - 0.4, ymax=np.array(spike_y) + 0.4,
                colors=COLOR_BLACK, linewidth=0.15,
            )

        ax.axvline(x=0.0, color=COLOR_BLACK, ls="--", lw=0.5)
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(-0.25, bout_onsets.size + 0.25)
        ax.set_xlabel("time relative to bout onset (s)", fontsize=12)
        ax.set_ylabel("bout #", fontsize=12)
        ax.set_title(f"raster (n_bouts = {bout_onsets.size})", fontsize=13, pad=3)
        ax.tick_params(labelsize=11)

    # section (b) — within-USV continuous tuning grid

    def _draw_section_b(
        self,
        fig,
        gs,
        emitter: str,
        cluster_data: dict,
    ) -> None:
        """
        Description
        -----------
        Draw section (b) of vocal Page 1: a 4×4 grid of
        (line + occupancy) cell pairs, one row per pair of continuous
        USV properties (per `PROPERTY_ROW_ORDER`). Dispatches to
        `_draw_property_pair` for each cell pair.

        Parameters
        ----------
        fig (matplotlib.figure.Figure)
            Parent figure (used to add subplots).
        gs (matplotlib.gridspec.GridSpecFromSubplotSpec)
            4×4 gridspec slot for section (b).
        emitter (str)
            Emitter ID keying into `usv_property_tuning`.
        cluster_data (dict)
            Per-cluster pkl payload.

        Returns
        -------
        None
        """

        sex = cluster_data["usv_property_tuning"][emitter][CONTINUOUS_PROPERTIES[0]].get("sex", "male")
        line_color = self._sex_color(sex)

        for row_idx, prop_pair in enumerate(PROPERTY_ROW_ORDER):
            for pair_idx, prop in enumerate(prop_pair):
                ax_line = fig.add_subplot(gs[row_idx, 2 * pair_idx])
                ax_occ = fig.add_subplot(gs[row_idx, 2 * pair_idx + 1])
                self._draw_property_pair(
                    ax_line=ax_line,
                    ax_occ=ax_occ,
                    prop=prop,
                    cluster_payload=cluster_data["usv_property_tuning"][emitter][prop],
                    line_color=line_color,
                )

    def _draw_property_pair(
        self,
        ax_line,
        ax_occ,
        prop: str,
        cluster_payload: dict,
        line_color: str,
    ) -> None:
        """
        Description
        -----------
        Draw one (line, occupancy) cell pair for the section-(b) grid.
        The line axes shows observed firing rate (smoothed if
        available) plus the smoothed shuffle band; the occupancy axes
        shows a step-filled per-bin occupancy histogram. The x-axis is
        trimmed to the non-zero-occupancy range (with a 5% margin on
        each side) and given two ticks (min, max) of the data extent;
        the y-axis is given two ticks (floor, ceil) over the joint
        observed + shuffle range, also with a small auto-margin.

        Parameters
        ----------
        ax_line (matplotlib.axes.Axes)
            Target axes for the rate-vs-property line plot.
        ax_occ (matplotlib.axes.Axes)
            Target axes for the per-property occupancy histogram.
        prop (str)
            One of `CONTINUOUS_PROPERTIES`.
        cluster_payload (dict)
            The `usv_property_tuning[emitter][prop]` block.
        line_color (str)
            Hex color for the observed-rate line and occupancy fill;
            typically the per-emitter sex color.

        Returns
        -------
        None
        """

        f = DISPLAY_FACTOR[prop]
        centers = cluster_payload["bin_centers"] * f
        edges = cluster_payload["bin_edges"] * f
        # smoothed versions are pre-computed in compute; fall back to raw
        # if smoothing was disabled (smoothing_sd == 0).
        rate = cluster_payload.get("rate_smoothed", cluster_payload["rate"])
        occ = cluster_payload["occupancy_seconds"]
        p0_5 = cluster_payload.get("null_p0_5_smoothed", cluster_payload["null_p0_5"])
        p99_5 = cluster_payload.get("null_p99_5_smoothed", cluster_payload["null_p99_5"])

        # Firing rates are non-negative by construction (counts /
        # occupancy, both >= 0) and Gaussian smoothing preserves that;
        # clamp the shuffle floor at zero anyway to insulate the
        # rendered band from any negative values that future numerics
        # / smoothing changes could introduce.
        p0_5 = np.maximum(p0_5, 0.0)

        # Natural bounds: trim x-axis to bins where there is actually
        # something to render. The compute side sets `rate=NaN` (and the
        # shuffle stats follow) whenever a bin's occupancy is below
        # `usv_property_min_occupancy_seconds`. An earlier revision used
        # `occ > 0` to pick the visible bins, which accepted outlier
        # bins (tiny non-zero occupancy, NaN rate / NaN shuffle band) at
        # the far end of the property range and produced an x-axis
        # extending well past the rightmost rendered point. Filter by
        # `isfinite(rate)` AND the shuffle band so the displayed extent
        # matches the visible data extent exactly.
        #
        # CRITICAL — use bin CENTERS (not bin edges) for both xlim and
        # tick positions. Different properties have different numbers of
        # occupied bins, so edge-based extents pick up a half-binwidth
        # offset that varies per plot, which breaks visual buffer
        # uniformity. Center-based extents with a percent margin give a
        # constant 5%-on-each-side buffer regardless of bin width
        # (matches the matplotlib auto-margin that behavioral plots use).
        visible = (
            np.isfinite(rate) & np.isfinite(p0_5) & np.isfinite(p99_5)
        )
        nz = np.where(visible)[0]
        if nz.size > 0:
            sl = slice(int(nz[0]), int(nz[-1]) + 1)
            data_x_lo = float(centers[nz[0]])
            data_x_hi = float(centers[nz[-1]])
        else:
            sl = slice(None)
            data_x_lo, data_x_hi = float(centers[0]), float(centers[-1])
        x_margin = 0.05 * (data_x_hi - data_x_lo) if data_x_hi > data_x_lo else 0.0
        x_lo_lim = data_x_lo - x_margin
        x_hi_lim = data_x_hi + x_margin

        ax_line.fill_between(
            centers[sl],
            p0_5[sl],
            p99_5[sl],
            where=np.isfinite(p0_5[sl]) & np.isfinite(p99_5[sl]),
            facecolor=COLOR_GRAY_BAND,
            interpolate=True,
        )
        ax_line.plot(centers[sl], rate[sl], lw=4.0, color=line_color)
        ax_line.set_xlim(x_lo_lim, x_hi_lim)
        # two ticks per axis, at the visible data extents
        ax_line.set_xticks(
            ticks=[data_x_lo, data_x_hi],
            labels=[f"{data_x_lo:.1f}", f"{data_x_hi:.1f}"],
            fontsize=11,
        )
        # y-buffer: read matplotlib's auto-fit ylim (already includes
        # a small auto-margin from fill_between + plot), then floor /
        # ceil to nearest integer (clamped at 0) so the line never sits
        # exactly on the top / bottom spine.
        ty_min, ty_max = ax_line.get_ylim()
        y_lo = max(float(np.floor(ty_min)), 0.0)
        y_hi = float(np.ceil(ty_max))
        if y_hi <= y_lo:
            y_hi = y_lo + 1.0
        ax_line.set_ylim(y_lo, y_hi)
        ax_line.set_yticks(
            ticks=[y_lo, y_hi],
            labels=[f"{y_lo:.1f}", f"{y_hi:.1f}"],
            fontsize=11,
        )
        ax_line.set_xlabel(self.vocal_labels[prop], fontsize=12, labelpad=2)
        ax_line.set_ylabel("Firing rate (sp/s)", fontsize=12, labelpad=-2)
        ax_line.tick_params(axis="x", labelsize=11, pad=4.0)
        ax_line.tick_params(axis="y", labelsize=11, pad=0.5)
        ax_line.set_title(
            prop.split(".")[-1], fontsize=15, pad=8.0,
            fontweight="bold",
        )
        ax_line.set_box_aspect(1)

        # stepfilled histogram-style for occupancy (matplotlib's stairs
        # primitive consumes pre-binned values + edges and renders the
        # equivalent of histtype='stepfilled' without needing the raw
        # samples).
        ax_occ.stairs(
            values=occ,
            edges=edges,
            fill=True,
            color=line_color,
            edgecolor=COLOR_BLACK,
            linewidth=0.4,
        )
        ax_occ.set_xlim(x_lo_lim, x_hi_lim)
        ax_occ.set_xticks(
            ticks=[data_x_lo, data_x_hi],
            labels=[f"{data_x_lo:.1f}", f"{data_x_hi:.1f}"],
            fontsize=11,
        )
        _, ty2_max = ax_occ.get_ylim()
        occ_max = max(int(np.ceil(float(ty2_max))), 1)
        ax_occ.set_yticks(
            ticks=[0, occ_max],
            labels=["0", f"{occ_max}"],
            fontsize=11,
        )
        ax_occ.set_xlabel(self.vocal_labels[prop], fontsize=12, labelpad=2)
        ax_occ.set_ylabel("Occupancy (s)", fontsize=12, labelpad=-2)
        ax_occ.tick_params(axis="x", labelsize=11, pad=4.0)
        ax_occ.tick_params(axis="y", labelsize=11, pad=0.5)
        ax_occ.set_box_aspect(1)

    # Page 2 — sections (c) and (d)

    def _render_page2(
        self,
        emitter: str,
        cluster_data: dict,
        segmentation: dict,
        viz_params: dict,
        save_fig,
        page_label: str,
    ) -> None:
        """
        Description
        -----------
        Render Page 2 of the vocal output for one emitter side: section
        (c) categorical watersheds (`usv_category_tuning` rate / occupancy
        / strip — 2 rows × 6 cols, paired by VAE / QLVM method) on top
        of section (d) `usv_category_peth` per-category PETH grid (flat
        sequential 6 cols / row, all-NaN entries skipped). Outer
        `height_ratios` adapts to the actual `usv_category_peth` row
        count so cells stay square in both sections.

        Parameters
        ----------
        emitter (str)
            Emitter ID keying into the vocal payloads.
        cluster_data (dict)
            Per-cluster pkl payload.
        segmentation (dict)
            Loaded UMAP segmentation per categorical feature.
        viz_params (dict)
            `neuronal_tuning_figures` settings block.
        save_fig (Callable[[Figure, str], None])
            Persists the rendered figure and closes it.
        page_label (str)
            Short label embedded in non-PDF per-page filenames.

        Returns
        -------
        None
        """

        fig = plt.figure(figsize=VOCAL_PAGE2_FIGSIZE_INCHES, tight_layout=False)

        # pre-count section (d) non-empty cells across all four
        # categorical features so we can size the gridspec to a flat
        # 6-col flow (same density / spacing as the behavioral grid).
        d_cols = 6
        total_d_cells = 0
        for cf in CATEGORICAL_FEATURES:
            payload_d = cluster_data["usv_category_peth"][emitter][cf]
            rate_arr_d = payload_d.get("rate_smoothed", payload_d["rate"])
            for i in range(rate_arr_d.shape[0]):
                if np.isfinite(rate_arr_d[i]).any():
                    total_d_cells += 1
        d_rows = max(1, (total_d_cells + d_cols - 1) // d_cols)

        # outer ratio reflects the row count of each section so the
        # cell height in (c) and (d) matches (both use 6 cols).
        outer = gridspec.GridSpec(
            nrows=2, ncols=1,
            height_ratios=[len(SECTION_C_ROWS), d_rows],
            hspace=0.12,
            left=0.04, right=0.98, top=0.94, bottom=0.05,
        )

        # section (c): 2 rows × 6 cols. Each row pairs the `_category`
        # (3 cols) and `_supercategory` (3 cols) variants of one
        # segmentation method. wspace/hspace match the behavioral grid.
        gs_c = gridspec.GridSpecFromSubplotSpec(
            len(SECTION_C_ROWS), 6, subplot_spec=outer[0, 0],
            wspace=0.40, hspace=0.45,
        )
        self._draw_section_c(
            fig=fig,
            gs=gs_c,
            emitter=emitter,
            cluster_data=cluster_data,
            segmentation=segmentation,
            viz_params=viz_params,
        )

        # section (d): flat sequential PETH flow at 6 cols / row,
        # ordered VAE cat → VAE supercat → QLVM cat → QLVM supercat;
        # all-NaN entries are skipped so the row stays packed.
        gs_d = gridspec.GridSpecFromSubplotSpec(
            d_rows, d_cols, subplot_spec=outer[1, 0],
            wspace=0.40, hspace=0.45,
        )
        self._draw_section_d(
            fig=fig,
            gs=gs_d,
            emitter=emitter,
            cluster_data=cluster_data,
            n_cols=d_cols,
        )

        save_fig(fig, page_label)

    # section (c) — categorical 2 rows × 6 cols

    def _draw_section_c(
        self,
        fig,
        gs,
        emitter: str,
        cluster_data: dict,
        segmentation: dict,
        viz_params: dict,
    ) -> None:
        """
        Description
        -----------
        Draw section (c): 2 rows × 6 cols. Each row pairs the
        `_category` (3 cols: rate watershed | occupancy watershed |
        strip) and `_supercategory` (3 cols, same structure) variants
        of one segmentation method (`SECTION_C_ROWS`). The rate
        watershed uses the configurable `ratemap_cmap`; the occupancy
        watershed uses a per-emitter sequential colormap built from
        the emitter's sex color; the strip plot ranks per-category
        observed firing rate against the shuffle band.

        Parameters
        ----------
        fig (matplotlib.figure.Figure)
            Parent figure.
        gs (matplotlib.gridspec.GridSpecFromSubplotSpec)
            2×6 gridspec slot for section (c).
        emitter (str)
            Emitter ID keying into `usv_category_tuning`.
        cluster_data (dict)
            Per-cluster pkl payload.
        segmentation (dict)
            Per-categorical-feature watershed segmentation.
        viz_params (dict)
            `neuronal_tuning_figures` settings block; reads
            `ratemap_cmap`, `vocal_strip_log_ratio_threshold`,
            `vocal_strip_symlog_linthresh`.

        Returns
        -------
        None
        """

        sex = cluster_data["usv_category_tuning"][emitter][SECTION_C_ROWS[0][0]].get("sex", "male")
        dot_color = self._sex_color(sex)

        log_threshold = VOCAL_STRIP_LOG_RATIO_THRESHOLD
        symlog_linthresh = VOCAL_STRIP_SYMLOG_LINTHRESH
        ratemap_cmap = viz_params.get("ratemap_cmap", "inferno")

        # build a sequential colormap for the occupancy watershed from
        # the emitter's sex color (mirrors the per-mouse colormap that
        # behavioral occupancy plots use).
        emitter_color = dot_color
        emitter_cmap = create_colormap(
            input_parameter_dict={
                "cm_length": 255,
                "cm_name": f"emitter_{emitter}",
                "cm_type": "sequential",
                "cm_start": (
                    int(emitter_color[1:3], 16),
                    int(emitter_color[3:5], 16),
                    int(emitter_color[5:7], 16),
                ),
                "cm_end": (255, 255, 255),
                "equalize_luminance": True,
                "match_luminance_by": "max",
                "change_saturation": 0.5,
                "cm_opacity": 1,
            }
        )

        for row_idx, row_feats in enumerate(SECTION_C_ROWS):
            for sub_idx, cat_feat in enumerate(row_feats):
                col_offset = sub_idx * 3
                payload = cluster_data["usv_category_tuning"][emitter][cat_feat]
                seg = segmentation.get(cat_feat)

                ax_rate = fig.add_subplot(gs[row_idx, col_offset + 0])
                ax_occ = fig.add_subplot(gs[row_idx, col_offset + 1])
                ax_strip = fig.add_subplot(gs[row_idx, col_offset + 2])

                # build a human-readable method label, e.g. vae_category
                # → "VAE category", qlvm_supercategory → "QLVM supercategory".
                method_token, granularity = cat_feat.split("_", 1)
                method_label = method_token.upper()
                tuning_title = f"{method_label} {granularity} tuning"

                # column N+0: firing rate watershed (configurable cmap)
                self._draw_categorical_watershed(
                    ax=ax_rate,
                    seg=seg,
                    categories=payload["categories"],
                    values=payload["rate"],
                    cbar_label="firing rate (sp/s)",
                    fig=fig,
                    cmap=ratemap_cmap,
                )
                ax_rate.set_title(tuning_title, fontsize=12, pad=3)

                # column N+1: occupancy watershed (per-emitter cmap), with
                # category IDs overlaid at each region's pole of
                # inaccessibility (distance-transform peak)
                self._draw_categorical_watershed(
                    ax=ax_occ,
                    seg=seg,
                    categories=payload["categories"],
                    values=payload["occupancy_count"].astype(float),
                    cbar_label="USV count",
                    fig=fig,
                    cmap=emitter_cmap,
                    annotate_categories=True,
                )
                ax_occ.set_title("Occupancy", fontsize=12, pad=3)

                # column N+2: per-category strip
                self._draw_categorical_strip(
                    ax=ax_strip,
                    payload=payload,
                    dot_color=dot_color,
                    log_threshold=log_threshold,
                    symlog_linthresh=symlog_linthresh,
                )
                ax_strip.set_title("Tuning vs Shuffled", fontsize=12, pad=3)

    def _draw_categorical_watershed(
        self,
        ax,
        seg: dict | None,
        categories,
        values: np.ndarray,
        cbar_label: str,
        fig,
        cmap="inferno",
        annotate_categories: bool = False,
    ) -> None:
        """
        Description
        -----------
        Render a watershed-style imshow of per-category values over the
        bundled UMAP segmentation grid. Pixels with no cluster-side
        category get a hatched fill; per-category 1-NN boundaries are
        contour-drawn in `COLOR_BLACK`. When `annotate_categories=True`,
        each region's category ID is overlaid at its centroid (or a
        circular-mean fallback for QLVM regions that wrap the embedding
        boundary). When `seg` is `None`, draws a placeholder text and
        returns.

        Parameters
        ----------
        ax (matplotlib.axes.Axes)
            Target axes.
        seg (dict | None)
            Segmentation block for this categorical feature
            (`label_grid`, `xx`, `yy`, `bounds`, `unique_labels`).
        categories (array-like)
            Category IDs present in the cluster's data.
        values (np.ndarray)
            Per-category value to color (firing rate or USV count).
        cbar_label (str)
            Colorbar label string.
        fig (matplotlib.figure.Figure)
            Parent figure (needed for the colorbar).
        cmap (str | matplotlib.colors.Colormap)
            Colormap name or instance.
        annotate_categories (bool)
            If True, overlay each category's ID at its region centroid.

        Returns
        -------
        None
        """

        if seg is None:
            ax.text(
                0.5, 0.5, "segmentation\nunavailable",
                transform=ax.transAxes, ha="center", va="center", fontsize=12, color=COLOR_GRAY_DASH,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_box_aspect(1)
            return

        label_grid = seg["label_grid"]
        x_min, x_max, y_min, y_max = seg["bounds"]
        xx, yy = seg["xx"], seg["yy"]

        # build a "value per pixel" image: NaN where the pixel's label is
        # missing from the cluster's category list (e.g., this cluster
        # never saw that category)
        cat_to_value = dict(
            zip(
                np.asarray(categories).tolist(),
                np.asarray(values).tolist(),
                strict=False,
            )
        )
        flat = label_grid.ravel()
        value_grid = np.array(
            [cat_to_value.get(int(l), np.nan) for l in flat],
            dtype=float,
        ).reshape(label_grid.shape)

        # shade: hatched gray where NaN; sequential colormap elsewhere
        finite = np.isfinite(value_grid)
        if finite.any():
            vmin = 0.0
            vmax = float(np.nanmax(value_grid))
            if vmax <= 0:
                vmax = 1.0
            im = ax.imshow(
                np.ma.masked_invalid(value_grid),
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                extent=[x_min, x_max, y_min, y_max],
                origin="lower",
                aspect="auto",
                interpolation="nearest",
            )
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cbar.set_label(cbar_label, fontsize=10)
            cbar.ax.tick_params(labelsize=7)
        # hatched fill for invalid pixels (under-supported categories)
        invalid_mask = ~finite
        if invalid_mask.any():
            ax.contourf(
                xx, yy, invalid_mask.astype(float),
                levels=[0.5, 1.5], colors=[COLOR_HATCH], alpha=0.6, hatches=["//"],
            )

        # 1-NN boundary contours
        # remap label_grid to dense indices for clean half-integer contours
        unique_labels_in_grid = np.unique(label_grid)
        label_to_dense = {int(l): i for i, l in enumerate(unique_labels_in_grid)}
        dense = np.vectorize(label_to_dense.get)(label_grid).astype(np.int32)
        ax.contour(
            xx, yy, dense,
            levels=np.arange(unique_labels_in_grid.size + 1) - 0.5,
            colors=COLOR_BLACK, linewidths=0.8,
        )

        # optionally annotate each category region with its category ID
        # at the centroid of the category mask. For non-toroidal
        # embeddings (VAE) the arithmetic mean of (xx, yy) over the mask
        # lands inside the blob and we use it. For toroidal embeddings
        # (QLVM) a category can wrap across the embedding boundary, in
        # which case the arithmetic centroid falls in the empty gap
        # between halves; we detect that case (closest pixel to the
        # arithmetic centroid is NOT inside the mask) and fall back to
        # the circular mean on each axis using period = bounds range.
        if annotate_categories:
            period_x = x_max - x_min
            period_y = y_max - y_min
            cluster_cats = set(int(c) for c in np.asarray(categories).tolist())
            for cat_label in unique_labels_in_grid:
                cat_id = int(cat_label)
                if cat_id not in cluster_cats:
                    continue
                mask = label_grid == cat_id
                if not mask.any():
                    continue
                masked_x = xx[mask]
                masked_y = yy[mask]
                cx = float(masked_x.mean())
                cy = float(masked_y.mean())
                # check whether the arithmetic centroid lands inside the
                # category mask (find nearest pixel and inspect its label)
                d_a = (xx - cx) ** 2 + (yy - cy) ** 2
                nearest = np.unravel_index(np.argmin(d_a), d_a.shape)
                if not mask[nearest] and period_x > 0 and period_y > 0:
                    # circular mean per axis (atan2 of mean sin / mean cos),
                    # then map back into [x_min, x_max] / [y_min, y_max]
                    theta_x = 2.0 * np.pi * (masked_x - x_min) / period_x
                    theta_y = 2.0 * np.pi * (masked_y - y_min) / period_y
                    cx_c = (
                        x_min
                        + np.arctan2(
                            np.sin(theta_x).mean(),
                            np.cos(theta_x).mean(),
                        )
                        * period_x / (2.0 * np.pi)
                    )
                    cy_c = (
                        y_min
                        + np.arctan2(
                            np.sin(theta_y).mean(),
                            np.cos(theta_y).mean(),
                        )
                        * period_y / (2.0 * np.pi)
                    )
                    cx_c = x_min + ((cx_c - x_min) % period_x)
                    cy_c = y_min + ((cy_c - y_min) % period_y)
                    d_c = (xx - cx_c) ** 2 + (yy - cy_c) ** 2
                    nearest_c = np.unravel_index(np.argmin(d_c), d_c.shape)
                    if mask[nearest_c]:
                        cx, cy = float(cx_c), float(cy_c)
                ax.text(
                    cx, cy, str(cat_id),
                    ha="center", va="center",
                    fontsize=9, fontweight="bold", color=COLOR_BLACK,
                    path_effects=[
                        mpe.withStroke(linewidth=2.0, foreground="white"),
                    ],
                )
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_box_aspect(1)

    def _draw_categorical_strip(
        self,
        ax,
        payload: dict,
        dot_color: str,
        log_threshold: float,
        symlog_linthresh: float,
    ) -> None:
        """
        Description
        -----------
        Per-category strip plot — one row per category showing the
        observed firing rate (dot, colored by emitter sex) on top of
        the [p0.5, p99.5] shuffle band (gray rectangle). All-NaN
        categories are dropped (the row is removed); the x-axis spans
        `[min(data, shuffle) − buffer, max(data, shuffle) + buffer]`
        and switches to symlog when the dynamic range exceeds
        `log_threshold`.

        Parameters
        ----------
        ax (matplotlib.axes.Axes)
            Target axes.
        payload (dict)
            `usv_category_tuning[emitter][cat_feat]` block.
        dot_color (str)
            Hex color for observed-rate dots (per-emitter sex color).
        log_threshold (float)
            If `max / min > log_threshold` over positive values, use
            symlog x-scale.
        symlog_linthresh (float)
            Width of the linear region around 0 in symlog units.

        Returns
        -------
        None
        """

        cats_all = np.asarray(payload["categories"])
        rate_all = np.asarray(payload["rate"], dtype=float)
        p0_5_all = np.asarray(payload["null_p0_5"], dtype=float)
        p99_5_all = np.asarray(payload["null_p99_5"], dtype=float)

        # Defensive non-negative clamp on the shuffle floor — firing
        # rates are non-negative by construction but a hard clamp at
        # zero guarantees the rendered shuffle rectangle never crosses
        # below the firing-rate floor regardless of upstream changes.
        p0_5_all = np.maximum(p0_5_all, 0.0)

        # drop categories with no observed firing rate (insufficient data
        # in the cluster); we don't want their numeric label on the y-axis
        # if there is nothing to show next to it.
        keep = np.isfinite(rate_all)
        cats = cats_all[keep]
        rate = rate_all[keep]
        p0_5 = p0_5_all[keep]
        p99_5 = p99_5_all[keep]

        # decide x-axis scale on the kept (finite) values
        scale = _decide_strip_xscale(rate, p0_5, p99_5, log_threshold)

        n = cats.size
        ys = np.arange(n)
        for i in range(n):
            lo = p0_5[i] if np.isfinite(p0_5[i]) else 0.0
            hi = p99_5[i] if np.isfinite(p99_5[i]) else 0.0
            ax.add_patch(
                plt.Rectangle(
                    (min(lo, hi), i - 0.3),
                    abs(hi - lo),
                    0.6,
                    facecolor=COLOR_GRAY_BAND,
                    edgecolor="none",
                )
            )
        ax.scatter(
            rate, ys, s=18, color=dot_color, edgecolor=COLOR_BLACK,
            linewidths=0.4, zorder=3,
        )

        ax.set_yticks(ys)
        ax.set_yticklabels([str(int(c)) for c in cats], fontsize=10)
        if n > 0:
            ax.set_ylim(n - 0.5, -0.5)  # category 0 at top
        ax.set_xlabel(f"firing rate (sp/s) [{scale}]", fontsize=10)
        ax.tick_params(axis="x", labelsize=10)

        if scale == "symlog":
            ax.set_xscale("symlog", linthresh=symlog_linthresh)
        # x-axis from joint min(observed, shuffle) − buffer to max + buffer
        # (no longer pinned to 0; firing-rate floor doesn't have to be
        # visible on every plot). For symlog, a LINEAR buffer near zero
        # is enormous in log space and visually separates the leftmost
        # data point from the spine by an order of magnitude or more —
        # use a multiplicative (geometric) buffer when both endpoints
        # are positive so the proportional log-space margin is 5% on
        # each side, matching the visual proportion of a linear axis.
        all_vals = np.concatenate(
            [
                rate[np.isfinite(rate)],
                p0_5[np.isfinite(p0_5)],
                p99_5[np.isfinite(p99_5)],
            ]
        )
        if all_vals.size > 0:
            x_lo_data = float(all_vals.min())
            x_hi_data = float(all_vals.max())
            # 10% buffer on each side (linear or log-space) so the
            # leftmost / rightmost observed-rate dots never sit on the
            # spine. Earlier 5% buffers occasionally placed the dot
            # essentially on the axis edge, which looked clipped.
            if (
                scale == "symlog"
                and x_lo_data > 0
                and x_hi_data > x_lo_data
            ):
                pad = (x_hi_data / x_lo_data) ** 0.10
                x_lo_lim = x_lo_data / pad
                x_hi_lim = x_hi_data * pad
            else:
                x_range = x_hi_data - x_lo_data
                buffer_x = (
                    0.10 * x_range if x_range > 0
                    else max(0.10 * abs(x_hi_data), 0.10)
                )
                x_lo_lim = x_lo_data - buffer_x
                x_hi_lim = x_hi_data + buffer_x
            ax.set_xlim(x_lo_lim, x_hi_lim)
            # Two explicit ticks at the data extrema: matplotlib's auto-
            # ticker (especially with symlog) places several ticks that
            # overlap one another given the small panel size. Two ticks
            # at min / max guarantee legibility regardless of scale.
            # `.3g` keeps the labels concise across magnitudes (0.5 stays
            # "0.5", 120 stays "120", 0.005 stays "0.005").
            ax.set_xticks(
                ticks=[x_lo_data, x_hi_data],
                labels=[f"{x_lo_data:.3g}", f"{x_hi_data:.3g}"],
                fontsize=10,
            )
            # Suppress any minor-tick labels symlog might add — keep just
            # the two majors we just set.
            ax.tick_params(axis="x", which="minor", labelbottom=False)
        ax.set_box_aspect(1)

    # section (d) — per-category PETH grid (tiny)

    def _draw_section_d(
        self,
        fig,
        gs,
        emitter: str,
        cluster_data: dict,
        n_cols: int,
    ) -> None:
        """
        Description
        -----------
        Draw section (d) of vocal Page 2: a flat sequential
        `usv_category_peth` PETH grid at `n_cols` cols/row. Items flow
        VAE category → VAE supercategory → QLVM category →
        QLVM supercategory; each non-NaN per-category PETH gets one
        cell. Each cell has its x-axis tightened to its own finite
        data extent (left), with the right edge anchored at t = 0.
        Two ticks per axis; lines doubled in width for visibility;
        per-cell title formatted "{METHOD} {cat|supercat} #{N}".

        Parameters
        ----------
        fig (matplotlib.figure.Figure)
            Parent figure.
        gs (matplotlib.gridspec.GridSpecFromSubplotSpec)
            (`ceil(total_items/n_cols)`, `n_cols`) gridspec slot.
        emitter (str)
            Emitter ID keying into `usv_category_peth`.
        cluster_data (dict)
            Per-cluster pkl payload.
        n_cols (int)
            Number of columns per row in the flat flow.

        Returns
        -------
        None
        """

        sex = cluster_data["usv_category_peth"][emitter][CATEGORICAL_FEATURES[0]].get("sex", "male")
        line_color = self._sex_color(sex)

        # build a flat sequential list of (title, rate, p0_5, p99_5,
        # bin_centers) tuples in the order VAE cat → VAE supercat →
        # QLVM cat → QLVM supercat, dropping all-NaN entries so the
        # display row stays packed.
        items: list[dict] = []
        for cat_feat in CATEGORICAL_FEATURES:
            method_token, granularity = cat_feat.split("_", 1)
            method_label = method_token.upper()
            short_gran = "supercat" if "super" in granularity else "cat"
            title_prefix = f"{method_label} {short_gran}"

            payload = cluster_data["usv_category_peth"][emitter][cat_feat]
            cats = np.asarray(payload["categories"])
            bin_centers = payload["bin_centers_s"]
            # smoothed versions are pre-computed in compute; fall back
            # to raw if smoothing was disabled (smoothing_sd == 0).
            rate_arr = payload.get("rate_smoothed", payload["rate"])
            p0_5_arr = payload.get("null_p0_5_smoothed", payload["null_p0_5"])
            p99_5_arr = payload.get("null_p99_5_smoothed", payload["null_p99_5"])
            for i in range(cats.size):
                rate = rate_arr[i, :]
                if not np.isfinite(rate).any():
                    continue
                items.append(
                    {
                        "title": f"{title_prefix} #{int(cats[i])}",
                        "rate": rate,
                        "p0_5": p0_5_arr[i, :],
                        "p99_5": p99_5_arr[i, :],
                        "bin_centers": bin_centers,
                    }
                )

        for display_idx, item in enumerate(items):
            row = display_idx // n_cols
            col = display_idx % n_cols
            rate = item["rate"]
            p0_5 = item["p0_5"]
            p99_5 = item["p99_5"]
            bin_centers = item["bin_centers"]

            # Defensive non-negative clamp on the shuffle floor (see
            # property-tuning cell for the same rationale).
            p0_5 = np.maximum(p0_5, 0.0)

            # tighten x-axis to the finite-rate extent of THIS cell;
            # 5% margin on the left, with the right side anchored at 0
            # so the anchor reference always shows up as a tick.
            finite = np.isfinite(rate)
            nz = np.where(finite)[0]
            x_lo_cell = float(bin_centers[nz[0]])
            x_hi_cell = 0.0
            cell_range = x_hi_cell - x_lo_cell
            cell_margin = (
                0.05 * cell_range if cell_range > 0
                else max(0.05 * abs(x_lo_cell), 0.05)
            )

            # Extend the rendered curve / shuffle band by a single
            # synthetic point at the anchor (t=0) repeating the last
            # finite value, so the visible data reaches the right axis
            # bound. Without this the line stops at `bin_centers[-1]`
            # (half-a-binwidth left of the anchor) and leaves a visual
            # air gap between the data and the t=0 tick.
            last_idx = int(nz[-1])
            bin_centers_ext = np.append(bin_centers, 0.0)
            rate_ext = np.append(rate, rate[last_idx])
            p0_5_last = p0_5[last_idx] if np.isfinite(p0_5[last_idx]) else 0.0
            p99_5_last = p99_5[last_idx] if np.isfinite(p99_5[last_idx]) else 0.0
            p0_5_ext = np.append(p0_5, p0_5_last)
            p99_5_ext = np.append(p99_5, p99_5_last)

            ax = fig.add_subplot(gs[row, col])
            ax.fill_between(
                bin_centers_ext, p0_5_ext, p99_5_ext,
                where=np.isfinite(p0_5_ext) & np.isfinite(p99_5_ext),
                facecolor=COLOR_GRAY_BAND, interpolate=True,
            )
            ax.plot(bin_centers_ext, rate_ext, lw=3.2, color=line_color)
            ax.set_xlim(x_lo_cell - cell_margin, x_hi_cell + cell_margin)
            ax.set_xticks(
                ticks=[x_lo_cell, 0.0],
                labels=[f"{x_lo_cell:.1f}", "0"],
                fontsize=8,
            )
            # y-axis: matplotlib's auto-fit ylim already adds a small
            # margin from fill_between + plot; floor / ceil to int (with
            # min clamped at 0) gives the same buffer treatment as the
            # continuous-feature plots.
            ty_min, ty_max = ax.get_ylim()
            y_lo = max(float(np.floor(ty_min)), 0.0)
            y_hi = float(np.ceil(ty_max))
            if y_hi <= y_lo:
                y_hi = y_lo + 1.0
            ax.set_ylim(y_lo, y_hi)
            ax.set_yticks(
                ticks=[y_lo, y_hi],
                labels=[f"{y_lo:.1f}", f"{y_hi:.1f}"],
                fontsize=8,
            )
            ax.tick_params(axis="x", labelsize=8, length=1.5, pad=2.0)
            ax.tick_params(axis="y", labelsize=8, length=1.5, pad=0.5)
            ax.set_xlabel("Pre-USV time (s)", fontsize=10, labelpad=2)
            ax.set_ylabel("Firing rate (sp/s)", fontsize=10, labelpad=-2)
            ax.set_title(item["title"], fontsize=11, pad=2)
            ax.set_box_aspect(1)
