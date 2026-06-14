"""
@author: bartulem
Anatomy / dataset-overview figure rendering.

This module emits the multi-panel "Figure 1"-style anatomy summary
of the Neuropixels recording dataset. It reads `unit_catalog.csv`
(the authoritative scope of every curated unit across the experiment)
plus the `brain_area_colors` block of `visualizations_settings.json`,
and renders the panels as standalone SVG files.

Current coverage:

  (a) recording yield  - `make_recording_yield_figure()`
      Two side-by-side panels.
        * Panel A: per-mouse stacked bar (SU somatic / SU non-somatic
          / MUA), one bar per mouse, sorted by total unit count
          descending. Coloured with the three-element mono-grey
          "strong contrast" cell-type triad.
        * Panel B: per-cell-type stacked bar (SU somatic, SU non-
          somatic, MUA), each bar split into the seven brain-area
          buckets (PAG, MRN, VTA, MB, CENT, SC, other) using the
          palette from `visualizations_settings.json["brain_area_colors"]`.

  (b) 3D unit positions  - `make_unit_positions_figure` /
      `build_unit_positions_figure` / `make_unit_positions_video`.

  (c) probe + raw-trace overlay  - `make_unit_waveform_figure`.

The catalog's raw `brain_area` acronyms (e.g. `SCdw`, `CENT2`) are
pooled to the seven canonical display buckets via
`make_behavioral_videos.pool_brain_area`, keeping bucketing consistent
with the raster / 3D-video colour code.
"""

from __future__ import annotations

import json
import pathlib
import urllib.request
from collections.abc import Callable
from datetime import datetime

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from usv_playpen.visualizations.plot_style import apply_plot_style
from usv_playpen.visualizations.figure_io import resolve_pdf_path, save_figure
from usv_playpen.visualizations.make_behavioral_videos import pool_brain_area

apply_plot_style()


# Canonical bucket ordering (most-populated first; matches the order
# the user picked for the figure legend / stack order).
_BRAIN_AREA_BUCKETS: tuple[str, ...] = (
    "PAG", "MRN", "VTA", "MB", "CENT", "SC", "other",
)

# Cell-type categories — bottom-to-top stack order in Panel A.
_CELL_TYPE_LABELS: tuple[str, ...] = (
    "Somatic", "Non-somatic", "Multi-unit",
)

# Strong-contrast mono-grey triad agreed for the cell-type stacks.
_CELL_TYPE_PALETTE: tuple[str, ...] = (
    "#1A1A1A", "#7A7A7A", "#CFCFCF",
)

# Mouse IDs excluded from anatomy figures (insufficient yield / not used
# in any downstream analysis).
_EXCLUDED_MOUSE_IDS: frozenset[str] = frozenset({"147366"})


# Allen CCFv3 structure IDs for the whole-brain shell and for the six
# display buckets. Looked up from the published structure ontology
# (`structure_graph_download/1.json`). SC is split into motor + sensory
# components — both rendered in the SC bucket colour.
_ALLEN_STRUCTURE_IDS: dict[str, int] = {
    "root": 997,
    "PAG":  795,
    "MRN":  128,
    "VTA":  749,
    "MB":   313,
    "CENT": 920,
    "SCm":  294,
    "SCs":  302,
}

# Mapping from sub-region mesh ID to the bucket colour it should be
# painted with. SCm and SCs both fall under the SC bucket; MB is the
# parent of PAG/MRN/VTA so it gets a near-transparent render to avoid
# engulfing the inner regions.
_BUCKET_FOR_MESH: dict[str, str] = {
    "PAG": "PAG", "MRN": "MRN", "VTA": "VTA",
    "MB": "MB",   "CENT": "CENT",
    "SCm": "SC",  "SCs": "SC",
}

# Bregma offset in CCF µm (Chon et al. 2019; matches the Allen
# ccf-streamlines reference). Used to transform Allen meshes from CCF
# coordinates into the stereotaxic Bregma-referenced frame the catalog
# `loc_ap` / `loc_dv` / `loc_ml` columns live in.
_BREGMA_AP_CCF: float = 5400.0
_BREGMA_DV_CCF: float = 332.0
_BREGMA_ML_CCF: float = 5739.0

# Default root directory holding the per-day ephys folders. Each
# directory contains the concatenated `.ap.bin` plus its
# `kilosort4/` and `changepoints_info_<YYYYMMDD>_imec<i>.json`.
_DEFAULT_EPHYS_ROOT: pathlib.Path = pathlib.Path("/mnt/falkner/Bartul/EPHYS")

# Default root for IBL histology output (one channel_locations.json
# per (mouse, session_date, hemisphere)).
_DEFAULT_HISTOLOGY_ROOT: pathlib.Path = pathlib.Path("/mnt/falkner/Bartul/histology")


# Local cache directory for Allen mesh OBJ files.
_ALLEN_MESH_CACHE: pathlib.Path = pathlib.Path(
    "/mnt/falkner/Bartul/EPHYS/allen_meshes"
)
_ALLEN_MESH_URL: str = (
    "https://download.alleninstitute.org/informatics-archive/"
    "current-release/mouse_ccf/annotation/ccf_2017/"
    "structure_meshes/{structure_id}.obj"
)


def _download_allen_mesh(structure_id: int) -> pathlib.Path:
    """
    Description
    -----------
    Return the local path to the Allen CCFv3 OBJ mesh for the given
    structure ID, downloading it into `_ALLEN_MESH_CACHE` on first
    access. Each mesh is a few hundred KB; the whole-brain `root` mesh
    is ~10 MB. After download the file is reused indefinitely (cached
    by structure ID alone).

    Parameters
    ----------
    structure_id (int)
        Allen CCFv3 structure ID (e.g. `997` for `root`, `795` for
        `PAG`).

    Returns
    -------
    path (pathlib.Path)
        Local path to the cached OBJ file.
    """

    _ALLEN_MESH_CACHE.mkdir(parents=True, exist_ok=True)
    path = _ALLEN_MESH_CACHE / f"{structure_id}.obj"
    if path.exists():
        return path
    url = _ALLEN_MESH_URL.format(structure_id=structure_id)
    with urllib.request.urlopen(url, timeout=60) as resp:
        data = resp.read()
    path.write_bytes(data)
    return path


def _load_obj_mesh(path: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Description
    -----------
    Minimal Wavefront-OBJ parser sufficient for the Allen structure
    meshes (vertices + triangle faces only — no textures, normals, or
    materials matter for our render). Returns the parsed vertex and
    face arrays.

    Parameters
    ----------
    path (pathlib.Path)
        Path to the OBJ file.

    Returns
    -------
    vertices (np.ndarray)
        `(N, 3)` float array of vertex coordinates, columns ordered as
        the OBJ file stores them (Allen convention: column 0 = AP,
        column 1 = DV, column 2 = ML; all in CCF µm).
    faces (np.ndarray)
        `(M, 3)` int array of zero-based vertex indices defining each
        triangle (OBJ files use one-based indices; we adjust on read).
    """

    verts: list[list[float]] = []
    faces: list[list[int]] = []
    with path.open("r") as fh:
        for line in fh:
            if line.startswith("v "):
                tokens = line.split()
                verts.append([float(tokens[1]), float(tokens[2]), float(tokens[3])])
            elif line.startswith("f "):
                tokens = line.split()[1:]
                face = [int(t.split("/")[0]) - 1 for t in tokens[:3]]
                faces.append(face)
    return np.asarray(verts, dtype=float), np.asarray(faces, dtype=int)


def _ccf_to_bregma(verts_ccf: np.ndarray) -> np.ndarray:
    """
    Description
    -----------
    Transform Allen-CCF µm vertices into stereotaxic Bregma-referenced
    coordinates:

      AP_bregma = BREGMA_AP_CCF - AP_ccf   (positive = anterior)
      DV_bregma = BREGMA_DV_CCF - DV_ccf   (positive = dorsal)
      ML_bregma = ML_ccf - BREGMA_ML_CCF   (positive = right)

    Bregma offset taken from Chon et al. 2019 / Allen ccf-streamlines
    (AP=5400 µm, DV=332 µm, ML=5739 µm). After the transform, the
    catalog's `loc_ap` / `loc_dv` / `loc_ml` columns live in the same
    frame as the returned vertices.

    Parameters
    ----------
    verts_ccf (np.ndarray)
        `(N, 3)` array of CCF µm coordinates (AP, DV, ML).

    Returns
    -------
    verts_bregma (np.ndarray)
        `(N, 3)` array of stereotaxic-Bregma µm coordinates
        (AP_stereo, DV_stereo, ML_stereo).
    """

    out = np.empty_like(verts_ccf)
    out[:, 0] = _BREGMA_AP_CCF - verts_ccf[:, 0]
    out[:, 1] = _BREGMA_DV_CCF - verts_ccf[:, 1]
    out[:, 2] = verts_ccf[:, 2] - _BREGMA_ML_CCF
    return out


class AnatomyFigureMaker:
    """
    Description
    -----------
    Dataset-overview figure renderer. One instance is bound to one
    `unit_catalog.csv` and one settings dict; each `make_*` method
    writes one timestamped SVG to a caller-supplied output directory.

    Parameters
    ----------
    catalog_path (str | pathlib.Path)
        Path to `unit_catalog.csv`. Read fully into memory on
        construction; downstream methods slice as needed.
    visualizations_parameter_dict (dict)
        Visualization settings; reads `brain_area_colors` (the seven-
        bucket display palette).
    message_output (Callable)
        Logger; defaults to `print`.
    """

    def __init__(self,
                 catalog_path: str | pathlib.Path,
                 visualizations_parameter_dict: dict,
                 message_output: Callable = print) -> None:
        """
        Description
        -----------
        Cache the catalog path and settings, pre-load the catalog
        DataFrame, and pin the bucket / cell-type column derivations
        so every figure method sees a consistent view.

        Parameters
        ----------
        catalog_path (str | pathlib.Path)
            Path to `unit_catalog.csv`.
        visualizations_parameter_dict (dict)
            Settings dictionary; must contain a `brain_area_colors`
            block keyed by bucket name (PAG / MRN / VTA / MB / CENT /
            SC / other).
        message_output (Callable)
            Logger; defaults to `print`.

        Returns
        -------
        None
        """

        self.catalog_path = pathlib.Path(catalog_path)
        self.visualizations_parameter_dict = visualizations_parameter_dict
        self.message_output = message_output

        # Canonical bucket-keyed colour palette.
        self.brain_area_colors = dict(
            visualizations_parameter_dict['brain_area_colors']
        )

        self.catalog = self._load_catalog()

    def _load_catalog(self) -> pd.DataFrame:
        """
        Description
        -----------
        Read `unit_catalog.csv` and decorate each row with the two
        derived columns the figure methods rely on:
          * `bucket`    -> pooled brain-area bucket (one of seven)
          * `cell_type` -> one of `Somatic`, `Non-somatic`, `Multi-unit`

        Parameters
        ----------
        None

        Returns
        -------
        df (pd.DataFrame)
            Catalog with the `bucket` and `cell_type` derived columns
            appended. Original column order preserved otherwise.
        """

        if not self.catalog_path.exists():
            raise FileNotFoundError(
                f"unit_catalog.csv not found: {self.catalog_path} "
                "(the catalog is the authoritative unit scope for every "
                "anatomy figure)."
            )
        df = pd.read_csv(
            self.catalog_path,
            usecols=[
                "mouse_id", "rec_date", "unit_id",
                "cluster_group", "somatic", "brain_area",
                "closest_ch",
                "loc_ap", "loc_ml", "loc_dv",
            ],
        )
        df["mouse_id"] = df["mouse_id"].astype(str)
        df["brain_area"] = df["brain_area"].astype(str)
        if _EXCLUDED_MOUSE_IDS:
            df = df[~df["mouse_id"].isin(_EXCLUDED_MOUSE_IDS)].reset_index(drop=True)
        df["bucket"] = df["brain_area"].map(pool_brain_area)
        df["cell_type"] = df.apply(self._classify_cell_type, axis=1)
        return df

    @staticmethod
    def _classify_cell_type(row: pd.Series) -> str:
        """
        Description
        -----------
        Three-way cell-type label combining the Kilosort curation flag
        (`cluster_group`) with the somatic / non-somatic post-curation
        annotation (`somatic`). Everything Kilosort labelled `mua`
        falls through to the `Multi-unit` bucket regardless of the
        somatic flag (the somatic / non-somatic split is only
        meaningful for well-isolated SUs).

        Parameters
        ----------
        row (pd.Series)
            One catalog row; reads `cluster_group` and `somatic`.

        Returns
        -------
        cell_type (str)
            One of `'Somatic'`, `'Non-somatic'`, `'Multi-unit'`.
        """

        if row["cluster_group"] == "mua":
            return "Multi-unit"
        return "Somatic" if row["somatic"] else "Non-somatic"

    def make_recording_yield_figure(
            self,
            out_dir: str | pathlib.Path | None = None,
            *,
            fig_size_inches: tuple[float, float] = (7.0, 2.6),
            fig_format: str | None = None,
    ) -> pathlib.Path:
        """
        Description
        -----------
        Render the sub-figure (a) "recording yield" SVG as a two-panel
        side-by-side figure and write it. When `out_dir` is `None` the
        path is resolved through `figure_io.save_figure` using the
        `figures` block of `visualizations_settings.json`; pass an
        explicit `out_dir` to override.

          * Panel A (left): per-mouse stacked bar; one bar per mouse,
            sorted by total unit count descending. Stacks bottom-up:
            SU-somatic / SU-non-somatic / MUA, mono-grey triad.
          * Panel B (right): per-cell-type stacked bar (SU-somatic /
            SU-non-somatic / MUA on the x-axis), each bar split into
            the seven brain-area buckets using the bucket palette.

        Parameters
        ----------
        out_dir (str | pathlib.Path | None)
            Directory override. `None` falls back to
            `figures.save_directory`.
        fig_size_inches (tuple[float, float])
            Figure width / height in inches. Default targets a Nature-
            style single-column band (~89 mm wide, ~66 mm tall).
        fig_format (str | None)
            Matplotlib output format override (e.g. `'svg'`, `'pdf'`,
            `'png'`). `None` falls back to `figures.fig_format`.

        Returns
        -------
        out_path (pathlib.Path)
            Path to the written figure file.
        """

        fig, (ax_left, ax_right) = plt.subplots(
            nrows=1, ncols=2,
            figsize=fig_size_inches,
            gridspec_kw={"width_ratios": [1.4, 1.0], "wspace": 0.35},
        )

        self._render_per_mouse_panel(ax_left)
        self._render_per_celltype_panel(ax_right)

        out_path = save_figure(
            fig, "anatomy_yield", self.visualizations_parameter_dict,
            override_dir=out_dir, override_format=fig_format,
        )
        plt.close(fig)

        self.message_output(
            f"  anatomy: wrote recording-yield figure to {out_path}"
        )
        return out_path

    def _render_per_mouse_panel(self, ax: plt.Axes) -> None:
        """
        Description
        -----------
        Render Panel A. Per-mouse stacked bar (SU-somatic /
        SU-non-somatic / MUA), one bar per mouse_id, sorted by total
        unit count descending. Annotates each bar's total at the top.

        Parameters
        ----------
        ax (plt.Axes)
            Matplotlib axes to draw into.

        Returns
        -------
        None
        """

        df = self.catalog
        pivot = (
            df.groupby(["mouse_id", "cell_type"]).size()
              .unstack(fill_value=0)
              .reindex(columns=list(_CELL_TYPE_LABELS), fill_value=0)
        )
        pivot["TOTAL"] = pivot.sum(axis=1)
        pivot = pivot.sort_values("TOTAL", ascending=False)
        totals = pivot["TOTAL"].to_numpy()
        pivot = pivot.drop(columns="TOTAL")

        x_positions = range(len(pivot.index))
        bottom = [0] * len(pivot.index)
        for label, hex_code in zip(_CELL_TYPE_LABELS, _CELL_TYPE_PALETTE):
            heights = pivot[label].to_numpy()
            ax.bar(
                x_positions,
                heights,
                bottom=bottom,
                width=0.78,
                color=hex_code,
                edgecolor="#FFFFFF",
                linewidth=0.4,
                label=label,
            )
            bottom = [b + h for b, h in zip(bottom, heights)]

        # total annotations above each bar
        for x, tot in zip(x_positions, totals):
            ax.text(
                x, tot * 1.02, str(int(tot)),
                ha="center", va="bottom",
                fontsize=6, color="#000000",
            )

        ax.set_xticks(list(x_positions))
        ax.set_xticklabels(pivot.index, rotation=30, ha="right", fontsize=8)
        ax.set_xlabel("Mouse ID", fontsize=8)
        ax.set_ylabel("Unit count", fontsize=8)
        ax.tick_params(axis="y", labelsize=6, length=2)
        ax.tick_params(axis="x", length=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim(0, max(totals) * 1.12)
        ax.legend(
            loc="upper right",
            fontsize=6,
            frameon=False,
            handlelength=1.0,
            handleheight=0.8,
            borderpad=0.2,
            labelspacing=0.3,
        )

    def _render_per_celltype_panel(self, ax: plt.Axes) -> None:
        """
        Description
        -----------
        Render Panel B. Per-cell-type stacked bar with three bars
        (SU-somatic / SU-non-somatic / MUA), each split into the seven
        brain-area buckets using `brain_area_colors`. Stack order
        bottom-up follows `_BRAIN_AREA_BUCKETS` (PAG, MRN, ..., other).
        Annotates each bar's total at the top.

        Parameters
        ----------
        ax (plt.Axes)
            Matplotlib axes to draw into.

        Returns
        -------
        None
        """

        df = self.catalog
        pivot = (
            df.groupby(["cell_type", "bucket"]).size()
              .unstack(fill_value=0)
              .reindex(index=list(_CELL_TYPE_LABELS), fill_value=0)
              .reindex(columns=list(_BRAIN_AREA_BUCKETS), fill_value=0)
        )
        totals = pivot.sum(axis=1).to_numpy()

        x_positions = range(len(pivot.index))
        bottom = [0] * len(pivot.index)
        for bucket in _BRAIN_AREA_BUCKETS:
            heights = pivot[bucket].to_numpy()
            ax.bar(
                x_positions,
                heights,
                bottom=bottom,
                width=0.62,
                color=self.brain_area_colors[bucket],
                edgecolor="#FFFFFF",
                linewidth=0.4,
                label=bucket,
            )
            bottom = [b + h for b, h in zip(bottom, heights)]

        for x, tot in zip(x_positions, totals):
            ax.text(
                x, tot * 1.02, str(int(tot)),
                ha="center", va="bottom",
                fontsize=6, color="#000000",
            )

        ax.set_xticks(list(x_positions))
        ax.set_xticklabels(pivot.index, rotation=0, ha="center", fontsize=8)
        ax.set_xlabel("Unit type", fontsize=8)
        ax.set_ylabel("Unit count", fontsize=8)
        ax.tick_params(axis="y", labelsize=6, length=2)
        ax.tick_params(axis="x", length=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim(0, max(totals) * 1.12)
        # Anchor the legend just outside the right axis edge so it no
        # longer overlaps the rightmost bar at higher unit counts.
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            fontsize=6,
            frameon=False,
            handlelength=1.0,
            handleheight=0.8,
            borderpad=0.2,
            labelspacing=0.3,
            ncols=1,
        )

    # sub-figure (b): 3D unit positions

    def make_unit_positions_figure(
            self,
            out_dir: str | pathlib.Path | None = None,
            *,
            fig_size_inches: tuple[float, float] = (6.0, 5.0),
            fig_format: str | None = None,
            view_elev: float = 18.0,
            view_azim: float = -60.0,
            shell_vertex_stride: int = 6,
            shell_point_size: float = 0.7,
            shell_point_alpha: float = 0.50,
            region_alpha: float = 0.10,
            mb_alpha: float = 0.04,
            dot_size: float = 9.0,
            dot_alpha: float = 1.0,
            legend_marker_scale: float = 3.0,
            filter_outliers: bool = True,
            rasterize_dense: bool = True,
    ) -> pathlib.Path:
        """
        Description
        -----------
        Render sub-figure (b) and save it as a static file. Thin wrapper
        around `build_unit_positions_figure`; use the builder directly
        from a notebook (with `%matplotlib widget`) for an interactive
        rotatable view.

        Parameters
        ----------
        out_dir (str | pathlib.Path | None)
            Directory override; `None` falls back to
            `figures.save_directory`.
        fig_format (str | None)
            Output format override; `None` falls back to
            `figures.fig_format`.
        rasterize_dense (bool, default True)
            When True, the six translucent Allen-CCF bucket meshes
            are flattened to an embedded raster layer inside the SVG.
            Those meshes are the dominant filesize contributor
            (~thousands of triangles each x six buckets) so this
            alone shrinks a 9 MB file to ~660 KB and makes downstream
            editing in Illustrator / Inkscape responsive. Axis labels,
            the legend, the brain-shell point cloud and the per-unit
            dots stay vector — matplotlib's 3D backend silently
            ignores `rasterized` on `Path3DCollection` (scatter
            points), so those layers remain editable per-element.
            Set False to emit a fully vector SVG (mesh polygons
            included; large but cleanly editable everywhere).
        All other kwargs
            See `build_unit_positions_figure`.

        Returns
        -------
        out_path (pathlib.Path)
            Path to the written figure file.
        """

        fig = self.build_unit_positions_figure(
            fig_size_inches=fig_size_inches,
            view_elev=view_elev,
            view_azim=view_azim,
            shell_vertex_stride=shell_vertex_stride,
            shell_point_size=shell_point_size,
            shell_point_alpha=shell_point_alpha,
            region_alpha=region_alpha,
            mb_alpha=mb_alpha,
            dot_size=dot_size,
            dot_alpha=dot_alpha,
            legend_marker_scale=legend_marker_scale,
            filter_outliers=filter_outliers,
            rasterize_dense=rasterize_dense,
        )

        out_path = save_figure(
            fig, "anatomy_unit_positions", self.visualizations_parameter_dict,
            override_dir=out_dir, override_format=fig_format,
        )
        plt.close(fig)

        self.message_output(
            f"  anatomy: wrote unit-positions figure to {out_path}"
        )
        return out_path

    def build_unit_positions_figure(
            self,
            *,
            fig_size_inches: tuple[float, float] = (6.0, 5.0),
            view_elev: float = 18.0,
            view_azim: float = -60.0,
            shell_vertex_stride: int = 6,
            shell_point_size: float = 0.7,
            shell_point_alpha: float = 0.50,
            region_alpha: float = 0.10,
            mb_alpha: float = 0.04,
            dot_size: float = 9.0,
            dot_alpha: float = 1.0,
            legend_marker_scale: float = 3.0,
            filter_outliers: bool = True,
            rasterize_dense: bool = True,
    ) -> plt.Figure:
        """
        Description
        -----------
        Build sub-figure (b) and return the matplotlib `Figure` without
        saving it. Call this from a notebook with `%matplotlib widget`
        active to get a rotatable / zoomable rendering; call
        `make_unit_positions_figure` instead for a static export.

        Render contents:
          * Whole-brain shell — sparse scatter of root-mesh vertices
            (every `shell_vertex_stride`-th vertex), in neutral grey.
            Scattering is preferred over triangulated `Poly3DCollection`
            here because matplotlib's 3D back-end does not depth-sort
            triangles; scatter points are correctly ordered through
            rotation and look like a clean dotted silhouette of the
            brain.
          * Six bucket volumes as translucent `Poly3DCollection`
            (PAG, MRN, VTA, MB, CENT, SC = SCm ∪ SCs) in the bucket
            palette colours. MB sits at a much lower alpha than the
            other five because it is the parent of PAG / MRN / VTA;
            higher MB alpha would occlude the inner sub-regions.
          * Somatic-SU dots from the catalog (`loc_ap`, `loc_ml`,
            `loc_dv`), one scatter per bucket. When `filter_outliers`
            is true (default), dots whose stereotaxic coordinates
            fall outside their bucket's Allen mesh bounding box are
            dropped — a small number (~0.3 %) of units land just
            outside the canonical reference mesh due to per-session
            probe-registration noise.

        The 3D pane fills, gridlines, axis spines, ticks, and axis
        labels are all switched off so the brain floats cleanly in
        white space; box-aspect is locked to true mouse-brain
        proportions (~13.2 mm AP × ~10.4 mm ML × ~7.4 mm DV).

        Parameters
        ----------
        fig_size_inches (tuple[float, float])
            Figure size in inches.
        view_elev (float)
            Elevation angle for `ax.view_init`.
        view_azim (float)
            Azimuth angle for `ax.view_init`.
        shell_vertex_stride (int)
            Take every Nth vertex of the root mesh for the shell
            scatter. Lower -> denser, more visible shell.
        shell_point_size (float)
            Marker size for shell dots.
        shell_point_alpha (float)
            Alpha for shell dots.
        region_alpha (float)
            Face alpha for the highlighted sub-region volumes
            (PAG, MRN, VTA, CENT, SC).
        mb_alpha (float)
            Face alpha for the MB parent volume; intentionally kept
            very low so it doesn't visually occlude PAG / MRN / VTA.
        dot_size (float)
            Scatter marker size for unit dots.
        dot_alpha (float)
            Alpha for unit dots.
        legend_marker_scale (float)
            Multiplier applied to the in-axes dot size when drawing
            the legend swatches. Keeps the brain dots tight while
            making the legend keys easy to read.
        filter_outliers (bool)
            When True, drop dots whose stereotaxic coordinates fall
            outside their bucket's Allen mesh bounding box.

        Returns
        -------
        fig (matplotlib.figure.Figure)
            The built figure. The caller is responsible for either
            displaying it (in a notebook) or saving / closing it.
        """

        fig = plt.figure(figsize=fig_size_inches)
        ax = fig.add_subplot(111, projection="3d")

        # 1) Whole-brain shell rendered as a sparse vertex scatter for
        # depth-correct rotation. `_load_obj_mesh` returns CCF µm; we
        # transform to stereotaxic-Bregma to match the catalog frame.
        root_path = _download_allen_mesh(_ALLEN_STRUCTURE_IDS["root"])
        root_verts_ccf, _ = _load_obj_mesh(root_path)
        root_verts = _ccf_to_bregma(root_verts_ccf)[::shell_vertex_stride]
        shell_scatter = ax.scatter(
            root_verts[:, 0],   # AP_stereo
            root_verts[:, 2],   # ML_stereo
            root_verts[:, 1],   # DV_stereo
            c="#888888",
            s=shell_point_size,
            alpha=shell_point_alpha,
            linewidths=0,
            depthshade=False,
            rasterized=rasterize_dense,
        )
        # Force the shell into the deepest layer so its 8k+ points
        # never obscure the data dots from any rotation angle. The
        # bucket meshes use a smaller positive sort_zpos so they sit
        # ABOVE the shell but still BELOW the data dots.
        shell_scatter.set_sort_zpos(1e12)

        # 2) Six bucket volumes — bucket colours, filled translucent.
        # MB is painted at a much lower alpha because it engulfs PAG /
        # MRN / VTA. Pre-compute per-bucket bounding boxes from the
        # same meshes so we can drop the dot outliers further down.
        bucket_bboxes = self._compute_bucket_bboxes()
        for mesh_key, bucket in _BUCKET_FOR_MESH.items():
            self._add_mesh_to_axes(
                ax,
                structure_id=_ALLEN_STRUCTURE_IDS[mesh_key],
                face_color=self.brain_area_colors[bucket],
                face_alpha=(mb_alpha if mesh_key == "MB" else region_alpha),
                face_stride=1,
                rasterized=rasterize_dense,
            )

        # 3) Somatic-SU dots, coloured by bucket. All dots from every
        # bucket are pooled into ONE `Path3DCollection` so matplotlib
        # depth-sorts them per-POINT (not per-bucket); without that,
        # bucket-vs-bucket centroid ordering flips with view angle and
        # smaller buckets like SC can disappear behind PAG. With
        # `filter_outliers=True`, drop dots that lie outside their
        # bucket's mesh bounding box (per-session probe-registration
        # noise; ~9 of 3477 units across the experiment). The legend
        # is built from proxy `Line2D` handles since the single
        # scatter call carries no per-bucket label.
        df_su = self.catalog[
            (self.catalog["cluster_group"] == "good")
            & self.catalog["somatic"]
        ]
        all_ap: list[np.ndarray] = []
        all_ml: list[np.ndarray] = []
        all_dv: list[np.ndarray] = []
        all_colors: list[str] = []
        legend_handles: list[Line2D] = []
        for bucket in _BRAIN_AREA_BUCKETS:
            sub = df_su[df_su["bucket"] == bucket]
            if filter_outliers and bucket in bucket_bboxes:
                bb = bucket_bboxes[bucket]
                inside = (
                    (sub["loc_ap"] >= bb["ap"][0])
                    & (sub["loc_ap"] <= bb["ap"][1])
                    & (sub["loc_dv"] >= bb["dv"][0])
                    & (sub["loc_dv"] <= bb["dv"][1])
                    & (sub["loc_ml"] >= bb["ml"][0])
                    & (sub["loc_ml"] <= bb["ml"][1])
                )
                sub = sub[inside]
            if len(sub) == 0:
                continue
            colour = self.brain_area_colors[bucket]
            all_ap.append(sub["loc_ap"].to_numpy())
            all_ml.append(sub["loc_ml"].to_numpy())
            all_dv.append(sub["loc_dv"].to_numpy())
            all_colors.extend([colour] * len(sub))
            legend_handles.append(
                Line2D(
                    [0], [0],
                    marker="o", linestyle="",
                    markerfacecolor=colour,
                    markeredgecolor="none",
                    markersize=float(np.sqrt(dot_size) * legend_marker_scale),
                    label=f"{bucket} (n={len(sub)})",
                )
            )

        if all_ap:
            dot_scatter = ax.scatter(
                np.concatenate(all_ap),
                np.concatenate(all_ml),
                np.concatenate(all_dv),
                c=all_colors,
                s=dot_size,
                alpha=dot_alpha,
                edgecolors="none",
                linewidths=0,
                depthshade=False,
                zorder=10,
                rasterized=rasterize_dense,
            )
            # Pool the data dots into the front-most layer so they
            # always paint on top of both the shell scatter and the
            # translucent bucket meshes, regardless of camera angle.
            dot_scatter.set_sort_zpos(-1e10)

        # 4) Axes / view. Box-aspect set to the true (AP, ML, DV) span
        # of the Allen root mesh so the brain shows up with realistic
        # proportions (~13.2 mm AP × ~10.4 mm ML × ~7.4 mm DV). All
        # axis decoration (panes, grid, ticks, labels) is hidden so
        # the brain reads as a clean floating object — the bucket
        # legend stays as the only on-figure annotation.
        ax.view_init(elev=view_elev, azim=view_azim)
        # `zoom` scales the 3D content INSIDE the axes cube so the
        # brain fills more of the canvas; matplotlib 3D otherwise
        # leaves a chunky default margin around the content.
        ax.set_box_aspect((13.2, 10.4, 7.4), zoom=1.4)
        ax.set_axis_off()
        # Maximise the brain's screen real-estate by stretching the
        # 3D axes to fill the whole figure (anim.save does not honour
        # `bbox_inches='tight'`). The legend is anchored INSIDE the
        # axes (top-left) so it stays on screen after the axes
        # expand.
        ax.set_position([0.0, 0.0, 1.0, 1.0])
        ax.legend(
            handles=legend_handles,
            bbox_to_anchor=(0.02, 0.98),
            loc="upper left",
            fontsize=7,
            frameon=False,
            handlelength=1.0,
            handleheight=0.8,
            borderpad=0.2,
            labelspacing=0.5,
        )

        return fig

    def make_unit_positions_video(
            self,
            out_dir: str | pathlib.Path | None = None,
            *,
            fig_size_inches: tuple[float, float] = (6.0, 5.0),
            view_elev: float = 18.0,
            view_azim_start: float = -60.0,
            n_frames: int = 240,
            fps: int = 30,
            video_format: str = "mp4",
            dpi: int = 600,
            **build_kwargs,
    ) -> pathlib.Path:
        """
        Description
        -----------
        Render sub-figure (b) as a short rotating video instead of a
        static image. The 3D view starts at `view_azim_start` and
        sweeps a full 360 degrees of azimuth in `n_frames` steps,
        re-rendering each frame and concatenating them at `fps` into a
        single video file. Elevation, scatter, meshes, and all other
        rendering settings are forwarded to
        `build_unit_positions_figure`.

        Parameters
        ----------
        out_dir (str | pathlib.Path)
            Output directory; created if absent.
        fig_size_inches (tuple[float, float])
            Figure size in inches.
        view_elev (float)
            Constant elevation angle held over the whole sweep.
        view_azim_start (float)
            Initial azimuth; the sweep covers
            `[view_azim_start, view_azim_start + 360)`.
        n_frames (int)
            Number of frames in the sweep. With `fps=30` and
            `n_frames=180` the video is 6 s long.
        fps (int)
            Output frame rate (frames per second).
        video_format (str)
            Container extension (`'mp4'`, `'gif'`, `'mov'`, ...).
            mp4 / mov use the ffmpeg writer; gif uses Pillow.
        dpi (int)
            Resolution for each rendered frame.
        **build_kwargs
            Forwarded to `build_unit_positions_figure` so the caller
            can dial in shell density, region alpha, dot size, etc.

        Returns
        -------
        out_path (pathlib.Path)
            Path to the written video file.
        """

        # Directory + format resolution is handled by `resolve_pdf_path`
        # below; the legacy `out_dir = pathlib.Path(out_dir)` call here
        # used to break when `out_dir is None` (which is now the default
        # — the helper falls back to `figures.save_directory`).

        fig = self.build_unit_positions_figure(
            fig_size_inches=fig_size_inches,
            view_elev=view_elev,
            view_azim=view_azim_start,
            **build_kwargs,
        )
        # The 3D axes are the only Axes object on the figure.
        ax3d = fig.axes[0]

        def update(frame_idx: int) -> tuple:
            """
            Description
            -----------
            Animation callback. Rotates the 3D view to the azimuth
            for the current frame. Returns an empty tuple because the
            azimuth change is not a `set_data`-style artist update.

            Parameters
            ----------
            frame_idx (int)
                Index of the current frame in `[0, n_frames)`.

            Returns
            -------
            artists (tuple)
                Empty — `view_init` is a property of the axes, not an
                artist, so the animation does not need to track one.
            """

            # Divide by `n_frames - 1` (not `n_frames`) so the LAST
            # rendered frame lands at exactly `view_azim_start + 360°`
            # — i.e. visually identical to the first frame. That makes
            # the mp4 loop seamlessly in a player without the small
            # angular gap a `frame_idx / n_frames` cycle leaves behind.
            denom = max(1, n_frames - 1)
            azim = view_azim_start + (360.0 * frame_idx / denom)
            ax3d.view_init(elev=view_elev, azim=azim)
            return ()

        anim = animation.FuncAnimation(
            fig,
            update,
            frames=n_frames,
            interval=1000.0 / fps,
            blit=False,
        )

        out_path, _video_dpi = resolve_pdf_path(
            "anatomy_unit_positions", self.visualizations_parameter_dict,
            override_dir=out_dir,
        )
        # resolve_pdf_path returns a `.pdf` path; we override the suffix
        # to match the requested video container.
        out_path = out_path.with_suffix(f".{video_format}")
        if video_format.lower() == "gif":
            writer = animation.PillowWriter(fps=fps)
        else:
            writer = animation.FFMpegWriter(fps=fps, bitrate=4000)
        anim.save(out_path, writer=writer, dpi=dpi)
        plt.close(fig)

        self.message_output(
            f"  anatomy: wrote unit-positions video to {out_path}"
        )
        return out_path

    # sub-figure (c): probe + raw-trace overlay

    def make_unit_waveform_figure(
            self,
            mouse_id: str,
            session_id: str,
            *,
            out_dir: str | pathlib.Path | None = None,
            probes: tuple[str, ...] = ("imec0", "imec1"),
            probe_to_hemisphere: dict[str, str] | None = None,
            ephys_root: str | pathlib.Path = _DEFAULT_EPHYS_ROOT,
            histology_root: str | pathlib.Path = _DEFAULT_HISTOLOGY_ROOT,
            n_top_units: int | None = None,
            probe_filter: str | None = "imec0",
            shank_filter: int | None = None,
            lateral_jitter_um: float = 12.0,
            waveform_width_um: float = 12.0,
            peak_amplitude_target_um: float = 25.0,
            inter_shank_wspace: float = 0.05,
            schematic_side: str = "right",
            opacity_sigma_um: float = 20.0,
            n_neighbors_each_side: int = 5,
            ap_padding_um: float = 20.0,
            dv_padding_um: float = 30.0,
            fig_size_inches: tuple[float, float] = (4.5, 7.0),
            fig_format: str | None = None,
    ) -> pathlib.Path:
        """
        Description
        -----------
        Render sub-figure (c): the mean Kilosort waveform of the
        top-`n_top_units` highest-amplitude SU-somatic clusters in the
        session, drawn side-by-side. Each sub-panel shows one cluster
        on its peak channel plus the N nearest same-shank channels
        above and below (`n_neighbors_each_side`), plus the "sibling"
        channel at the same axial as the peak on the other column of
        the shank — 5 + peak + sibling + 5 = 12 channels per unit.
        Opacity decays Gaussianly with probe-local distance from the
        peak; voltage is rendered raw (no per-cluster normalisation),
        so high-amplitude units draw a bigger spike than low-amplitude
        ones.

        Parameters
        ----------
        out_dir (str | pathlib.Path)
            Output directory; created if absent.
        mouse_id (str)
            Catalog `mouse_id` for the session.
        session_id (str)
            Session timestamp string, e.g. `'20241115_162223'`.
        probes (tuple[str, ...])
            Probes to search across when ranking clusters.
        ephys_root (str | pathlib.Path)
            Root containing `<YYYYMMDD>_imec<i>/` per-day directories.
        n_top_units (int)
            Number of top-amplitude clusters to render (left to right).
        waveform_width_um (float)
            Lateral-axis width allocated to each waveform.
        peak_amplitude_target_um (float)
            Axial-µm height that the largest rendered unit's peak
            amplitude maps to. One fixed scale across all units, so peak
            amplitudes vary visibly across the row.
        opacity_sigma_um (float)
            Gaussian σ for opacity vs probe-local distance from peak.
        n_neighbors_each_side (int)
            Number of nearest same-shank channels above and below the
            peak (peak's row sibling is added separately).
        fig_size_inches (tuple[float, float])
            Figure size in inches.
        fig_format (str)
            Output format (e.g. `'svg'`, `'pdf'`, `'png'`).

        Returns
        -------
        out_path (pathlib.Path)
            Path to the written figure file.
        """

        ephys_root = pathlib.Path(ephys_root)
        histology_root = pathlib.Path(histology_root)
        if probe_to_hemisphere is None:
            probe_to_hemisphere = {"imec0": "R", "imec1": "L"}

        rec_date = int(session_id[:8])
        clusters = self._collect_session_clusters(mouse_id, rec_date)

        # Per-probe templates / cluster bookkeeping. The
        # `probe_to_hemisphere` / `histology_root` args are no longer
        # used for this view (we plot in probe-local coords) — kept in
        # the signature so older callers don't have to change.
        del histology_root, probe_to_hemisphere
        per_probe: dict[str, dict] = {}
        for probe in probes:
            per_probe[probe] = self._gather_probe_context_for_unit(
                ephys_root, rec_date, probe, clusters,
            )

        # Rank every SU-somatic cluster across all probes by template
        # peak-to-peak on its TRUE peak channel
        # (`np.ptp(template, axis=0).argmax()`), not the catalog's
        # `closest_ch` — those disagree (the catalog uses monopolar
        # triangulation, which can place the unit's geometric centre
        # on a different channel than the one with the largest
        # amplitude).
        all_units: list[dict] = []
        for probe, ctx in per_probe.items():
            for cluster_num, template in ctx["cluster_templates"].items():
                ptp_per_channel = np.ptp(template, axis=0)
                peakch = int(ptp_per_channel.argmax())
                ptp = float(ptp_per_channel[peakch])
                all_units.append({
                    "probe": probe,
                    "cluster_num": cluster_num,
                    "ptp": ptp,
                    "peakch": peakch,
                    "template": template,
                    "ctx": ctx,
                })
        if not all_units:
            raise RuntimeError(
                "No SU-somatic clusters found in any probe for this session."
            )
        all_units.sort(key=lambda u: -u["ptp"])
        top_units = (
            all_units if n_top_units is None else all_units[:n_top_units]
        )

        # Optional filters: probe and/or shank.
        if probe_filter is not None:
            top_units = [u for u in top_units if u["probe"] == probe_filter]
        if shank_filter is not None:
            keep = []
            for u in top_units:
                cs = np.load(u["ctx"]["ks_dir"] / "channel_shanks.npy").astype(int)
                if int(cs[u["peakch"]]) == shank_filter:
                    keep.append(u)
            top_units = keep
        if not top_units:
            raise RuntimeError(
                "No units left after probe / shank filters."
            )

        # Per-unit deterministic lateral jitter so units sharing a peak
        # channel don't fully overlap. Seed by cluster_num so the same
        # unit always lands at the same offset run-to-run.
        for u in top_units:
            rng = np.random.default_rng(u["cluster_num"])
            u["lateral_offset_um"] = float(
                rng.uniform(-lateral_jitter_um, +lateral_jitter_um)
            )

        # Group units by shank so each shank gets its own subplot.
        # Each subplot uses real probe-local lateral µm; the visible
        # inter-shank break comes from the gap between subplots
        # (`inter_shank_wspace`).
        for u in top_units:
            cs = np.load(u["ctx"]["ks_dir"] / "channel_shanks.npy").astype(int)
            u["shank"] = int(cs[u["peakch"]])

        # Use the probe's full shank set (1..4) so all four bands
        # render even if some have no SU-somatic units. For the LH
        # probe (schematic_on_left=True) we REVERSE the shank order
        # so the data panels and the schematic both read 4 → 1
        # left-to-right; combined with the RH probe rendering
        # 1 → 4 left-to-right, this puts rostral on the left in
        # BOTH probes (shank 1 = rostral for RH, shank 4 = rostral
        # for LH).
        schematic_on_left = schematic_side.lower().startswith("l")
        any_ks_dir = top_units[0]["ctx"]["ks_dir"]
        all_shanks = sorted(
            int(s) for s in np.unique(
                np.load(any_ks_dir / "channel_shanks.npy").astype(int)
            )
        )
        if schematic_on_left:
            all_shanks = list(reversed(all_shanks))

        fig = plt.figure(figsize=fig_size_inches)
        if schematic_on_left:
            width_ratios = [1.5] + [1.0] * len(all_shanks)
            gs = fig.add_gridspec(
                1, len(all_shanks) + 1,
                width_ratios=width_ratios,
                wspace=inter_shank_wspace,
            )
            schematic_ax = fig.add_subplot(gs[0, 0])
            axes = [fig.add_subplot(gs[0, 1])]
            for col in range(2, len(all_shanks) + 1):
                axes.append(fig.add_subplot(gs[0, col], sharey=axes[0]))
        else:
            width_ratios = [1.0] * len(all_shanks) + [1.5]
            gs = fig.add_gridspec(
                1, len(all_shanks) + 1,
                width_ratios=width_ratios,
                wspace=inter_shank_wspace,
            )
            axes = [fig.add_subplot(gs[0, 0])]
            for col in range(1, len(all_shanks)):
                axes.append(fig.add_subplot(gs[0, col], sharey=axes[0]))
            schematic_ax = fig.add_subplot(gs[0, -1])

        all_axi: list[np.ndarray] = []
        # Precompute per-shank lateral centre for axis bounds.
        positions_any = top_units[0]["ctx"]["channel_positions"]
        channel_shanks_any = np.load(any_ks_dir / "channel_shanks.npy").astype(int)
        shank_lat_range = {
            int(s): (
                float(positions_any[channel_shanks_any == s, 0].min()),
                float(positions_any[channel_shanks_any == s, 0].max()),
            )
            for s in all_shanks
        }

        for ax_idx, shank_id in enumerate(all_shanks):
            ax = axes[ax_idx]
            shank_units = [u for u in top_units if u["shank"] == shank_id]
            for unit in shank_units:
                effective_scale = peak_amplitude_target_um / max(unit["ptp"], 1e-6)
                _, axi_drawn = self._draw_single_unit_waveforms_in_brain_space(
                    ax,
                    ctx=unit["ctx"],
                    cluster_num=unit["cluster_num"],
                    template=unit["template"],
                    peakch=unit["peakch"],
                    waveform_width_um=waveform_width_um,
                    waveform_voltage_uv_scale=effective_scale,
                    opacity_sigma_um=opacity_sigma_um,
                    n_neighbors_each_side=n_neighbors_each_side,
                    lateral_offset_um=unit["lateral_offset_um"],
                )
                all_axi.append(axi_drawn)
            # Tight x-axis bounds per shank.
            lat_lo, lat_hi = shank_lat_range[shank_id]
            ax.set_xlim(
                lat_lo - lateral_jitter_um - waveform_width_um / 2 - ap_padding_um,
                lat_hi + lateral_jitter_um + waveform_width_um / 2 + ap_padding_um,
            )
            # Hide the tick marks but keep the numerical labels;
            # match label size across both axes.
            ax.tick_params(axis="both", length=0, labelsize=6)
            ax.spines["top"].set_visible(False)
            # The y-axis lives on the side OPPOSITE the schematic:
            #   schematic right → y-axis on the leftmost data panel
            #   schematic left  → y-axis on the rightmost data panel
            is_first_data_axes = ax_idx == 0
            is_last_data_axes = ax_idx == len(all_shanks) - 1
            keep_y_here = (
                is_last_data_axes if schematic_on_left else is_first_data_axes
            )
            if keep_y_here:
                if schematic_on_left:
                    ax.spines["left"].set_visible(False)
                    ax.spines["right"].set_visible(True)
                    ax.yaxis.tick_right()
                    ax.yaxis.set_label_position("right")
                else:
                    ax.spines["right"].set_visible(False)
            else:
                ax.spines["left"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.tick_params(axis="y", labelleft=False, labelright=False)
            n_units_on_shank = sum(
                1 for u in top_units if u["shank"] == shank_id
            )
            ax.set_xlabel(
                f"{probe_filter}\nshank {shank_id}\n"
                f"$n_\\mathrm{{units}}={n_units_on_shank}$",
                fontsize=7, linespacing=1.1,
            )
        ylabel_ax = axes[-1] if schematic_on_left else axes[0]
        ylabel_ax.set_ylabel("ventral-dorsal (µm)", fontsize=7)
        # Right-side schematic panel: small grey-square depiction of
        # the four-shank Neuropixels 2.0 contact layout, anchored to
        # the data panels' y-range with extra padding above and below
        # for "channels continue" fade marks plus shank-tip
        # triangles. Y ticks are hidden (redundant with the leftmost
        # data panel).
        positions_full = top_units[0]["ctx"]["channel_positions"]
        channel_shanks_full = np.load(
            top_units[0]["ctx"]["ks_dir"] / "channel_shanks.npy"
        ).astype(int)
        axial_lo_rec = float(positions_full[:, 1].min())
        axial_hi_rec = float(positions_full[:, 1].max())
        sch_pad_above = 80.0
        sch_pad_below = 110.0

        # Compressed schematic lateral: keep each shank's internal
        # 32 µm column geometry, but shrink the inter-shank gap so
        # the four shanks sit close together. `schematic_shank_step`
        # is the centre-to-centre spacing used in the schematic
        # (physical is ~250 µm).
        schematic_shank_step = 90.0
        true_centres = {
            int(s): float(positions_full[channel_shanks_full == s, 0].mean())
            for s in all_shanks
        }
        base_centre = true_centres[all_shanks[0]]
        new_centres = {
            int(s): base_centre + i * schematic_shank_step
            for i, s in enumerate(all_shanks)
        }
        shifts = {
            s: new_centres[s] - true_centres[s] for s in all_shanks
        }
        compressed_lat = np.array([
            positions_full[ch, 0]
            + shifts[int(channel_shanks_full[ch])]
            for ch in range(positions_full.shape[0])
        ])

        # Colour each schematic square by the anatomy of the
        # PHYSICAL electrode at that contact. We do NOT trust the
        # `neuropixels_sites_to_anatomy_converter.json` channel-index
        # ranges here because that converter is keyed by the
        # histology pipeline's channel ordering (geometric: shank-major
        # then axial), which differs from Kilosort's IMRO-driven
        # channel index. Joining converter[ks_ch] would systematically
        # mis-label every contact. Instead we read the IBL
        # `channel_locations.json` directly and key by the
        # `(lateral, axial)` physical position that BOTH IBL and KS
        # agree on, then look up each KS channel's region by its
        # physical position.
        n_channels = positions_full.shape[0]
        hemisphere = "R" if str(probe_filter).endswith("0") else "L"
        ibl_dir = "ibl_RH" if hemisphere == "R" else "ibl_LH"
        ibl_path = (
            pathlib.Path("/mnt/falkner/Bartul/histology")
            / str(mouse_id) / str(rec_date) / ibl_dir / "channel_locations.json"
        )
        with ibl_path.open() as fh:
            ibl_payload = json.load(fh)
        pos_to_region: dict[tuple[int, int], str] = {}
        for key, entry in ibl_payload.items():
            if not key.startswith("channel_"):
                continue
            pos_to_region[(int(entry["lateral"]), int(entry["axial"]))] = (
                entry["brain_region"]
            )
        contact_colours = [
            self.brain_area_colors[pool_brain_area(
                pos_to_region.get(
                    (int(positions_full[ch, 0]), int(positions_full[ch, 1]))
                )
            )]
            for ch in range(n_channels)
        ]
        schematic_ax.scatter(
            compressed_lat, positions_full[:, 1],
            s=14, marker="s",
            facecolor=contact_colours,
            edgecolor="none",
        )
        # Fade rows above the recorded range — alpha decays linearly.
        ghost_axials_above = axial_hi_rec + 15.0 * np.arange(1, 6)
        ghost_axials_below = axial_lo_rec - 15.0 * np.arange(1, 6)
        ghost_alphas = np.linspace(0.40, 0.05, 5)
        for shank_id in all_shanks:
            mask = channel_shanks_full == shank_id
            lats = sorted(set(compressed_lat[mask]))
            for ax_val, alpha in zip(ghost_axials_above, ghost_alphas):
                schematic_ax.scatter(
                    lats, [ax_val] * len(lats),
                    s=10, marker="s",
                    facecolor="#9E9E9E", edgecolor="none",
                    alpha=alpha,
                )
            for ax_val, alpha in zip(ghost_axials_below, ghost_alphas):
                schematic_ax.scatter(
                    lats, [ax_val] * len(lats),
                    s=10, marker="s",
                    facecolor="#9E9E9E", edgecolor="none",
                    alpha=alpha,
                )
        # Thin vertical guide lines flanking each shank's two
        # contact columns — frames the shank visually and connects
        # the contact dots to the tip triangle.
        line_top = axial_hi_rec + sch_pad_above
        line_bot = axial_lo_rec - sch_pad_below + 40.0
        shank_half_width_um = 14.0  # snug margin around contact columns
        for shank_id in all_shanks:
            mask = channel_shanks_full == shank_id
            lats = compressed_lat[mask]
            x_left = float(lats.min()) - shank_half_width_um
            x_right = float(lats.max()) + shank_half_width_um
            for x in (x_left, x_right):
                schematic_ax.plot(
                    [x, x], [line_bot, line_top],
                    color="#BBBBBB", linewidth=0.45, alpha=0.7,
                )

        # Shank-tip triangles, one per shank, centred on the shank
        # and placed below the lower fade.
        triangle_y_top = axial_lo_rec - sch_pad_below + 40.0
        triangle_y_tip = axial_lo_rec - sch_pad_below
        for shank_id in all_shanks:
            mask = channel_shanks_full == shank_id
            lats = compressed_lat[mask]
            x_centre = float(lats.mean())
            # Match the triangle's top edge to the shank's flanking
            # guide lines so the tip reads as a continuation of the
            # shank body.
            tri_half = (float(lats.max()) - float(lats.min())) / 2 + shank_half_width_um
            schematic_ax.add_patch(plt.Polygon(
                [
                    (x_centre - tri_half, triangle_y_top),
                    (x_centre + tri_half, triangle_y_top),
                    (x_centre,            triangle_y_tip),
                ],
                facecolor="#9E9E9E", edgecolor="none",
            ))
        schematic_ax.set_xlim(
            float(compressed_lat.min()) - 30,
            float(compressed_lat.max()) + 30,
        )
        schematic_ax.set_ylim(
            triangle_y_tip - 20,
            axial_hi_rec + sch_pad_above,
        )
        schematic_ax.set_xticks([])
        schematic_ax.set_yticks([])
        for spine in schematic_ax.spines.values():
            spine.set_visible(False)

        # Indicate the x-axis is also in µm. Single label centred
        # below the four data subplots (positioned below the per-shank
        # captions). Note: schematic's x-axis is the same probe-local
        # lateral µm, just shown without ticks.
        # Centre the supxlabel under the four data panels (not the
        # schematic) — shift right when schematic is on the left.
        supx = 0.62 if schematic_on_left else 0.42
        fig.supxlabel("rostro-caudal (µm)", fontsize=7, y=0.045, x=supx)

        # Header: mouse / session. Probe identifier is already part
        # of the per-shank xlabels, so it's intentionally omitted here.
        fig.suptitle(
            f"{mouse_id} / {session_id}",
            fontsize=8, y=0.99,
        )

        # Bucket-colour legend: one swatch per anatomy bucket that
        # actually appears in the rendered units (avoids putting
        # SC / VTA / CENT chips on a probe-session that has none).
        buckets_present_in_units = sorted({
            u["ctx"]["cluster_to_bucket"][u["cluster_num"]]
            for u in top_units
        })
        legend_handles = [
            Line2D(
                [0], [0],
                marker="s", linestyle="",
                markerfacecolor=self.brain_area_colors[b],
                markeredgecolor="none",
                markersize=8,
                label=b,
            )
            for b in buckets_present_in_units
        ]
        # Legend: top-center, horizontal (one row), aligned with the
        # supxlabel x so it sits over the centre of the data-panel
        # row (NOT over the y-axis label).
        legend_supx = 0.62 if schematic_on_left else 0.42
        fig.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(legend_supx, 0.955),
            ncol=len(legend_handles),
            fontsize=7,
            frameon=False,
            handlelength=1.0,
            handleheight=0.8,
            borderpad=0.2,
            columnspacing=1.2,
        )

        # Scale bar in the LOWER-LEFT corner of shank 1 (axes[0] is
        # always shank 1 regardless of which side the schematic is
        # on). Vertical line = the height every unit's peak channel
        # is rendered at (`peak_amplitude_target_um`), labelled
        # "norm µV"; horizontal line spans 30 samples (= 1 ms at
        # 30 kHz), labelled "1 ms".
        # Lock the shared y-axis BEFORE drawing the scale bar so its
        # data-coordinate y values land in the right place (the bar
        # would otherwise be positioned against matplotlib's auto-
        # derived ylim from the LineCollection adds, not the final
        # explicit ylim we set below).
        axi_concat = np.concatenate(all_axi)
        ax_y_lo = axi_concat.min() - dv_padding_um
        ax_y_hi = axi_concat.max() + dv_padding_um
        axes[0].set_ylim(ax_y_lo, ax_y_hi)

        # Horizontal bar spans 30 sample bins (= 1 ms at 30 kHz),
        # which is `(30 / n_samples)` of the full waveform width.
        sb_ax = axes[0]
        sb_xlim = sb_ax.get_xlim()
        n_samples = top_units[0]["template"].shape[0]
        ms_bar_um = (30.0 / n_samples) * waveform_width_um
        # For imec1 (schematic_on_left) the leftmost data panel
        # sits next to the schematic and the shank itself is at high
        # absolute lateral (e.g. shank 4 ≈ 777 µm), so the same
        # x-offset that worked for imec0 lands the bar inside the
        # waveform cloud. Push it further left for imec1.
        sb_x0 = sb_xlim[0] + (5.0 if schematic_on_left else 18.0)
        sb_y_bot = ax_y_lo + 25.0
        sb_y_top = sb_y_bot + peak_amplitude_target_um
        sb_x_right = sb_x0 + ms_bar_um
        sb_ax.plot(
            [sb_x0, sb_x0], [sb_y_bot, sb_y_top],
            color="#000000", linewidth=0.8, clip_on=False, zorder=10,
        )
        sb_ax.plot(
            [sb_x0, sb_x_right], [sb_y_bot, sb_y_bot],
            color="#000000", linewidth=0.8, clip_on=False, zorder=10,
        )
        sb_ax.text(
            sb_x0 - 7.0, sb_y_bot + 0.92 * peak_amplitude_target_um,
            "norm µV", fontsize=6, ha="center", va="center",
            rotation=90,
        )
        sb_ax.text(
            (sb_x0 + sb_x_right) / 2, sb_y_bot - 3.0,
            "1 ms", fontsize=6, ha="center", va="top",
        )
        # Don't invert: for this probe insertion, LARGER axial = dorsal
        # (back of probe near surface). Matplotlib default puts larger
        # y at the top, which lands dorsal at the top correctly.
        # Per-probe margins: when the schematic is on the left, the
        # y-axis lives on the right side of the rightmost data panel,
        # so we want more right padding (room for the ylabel) and
        # less left padding (no ylabel there, just the schematic).
        if schematic_on_left:
            fig.subplots_adjust(left=0.04, right=0.91, top=0.93, bottom=0.16)
        else:
            fig.subplots_adjust(left=0.10, right=0.98, top=0.93, bottom=0.16)

        # Force savefig to use the exact figure bbox — without an
        # explicit bbox matplotlib sometimes auto-grows the canvas
        # to enclose stray artists (here the multi-line shank
        # captions and the off-axes scale-bar text), which produces
        # a hugely-elongated SVG. The transform below pins the
        # output to `fig.get_size_inches()` exactly.
        size_w, size_h = fig.get_size_inches()
        out_path = save_figure(
            fig, "anatomy_unit_waveform", self.visualizations_parameter_dict,
            override_dir=out_dir, override_format=fig_format,
            bbox_inches=Bbox.from_bounds(0, 0, size_w, size_h),
        )
        plt.close(fig)

        ptps = ", ".join(f"{u['ptp']:.0f}" for u in top_units)
        self.message_output(
            f"  anatomy: wrote single-unit waveform figure to {out_path}  "
            f"(top {len(top_units)} ptps: [{ptps}])"
        )
        return out_path

    def _collect_session_clusters(
            self, mouse_id: str, rec_date: int
    ) -> pd.DataFrame:
        """
        Description
        -----------
        Slice the catalog to the SU-somatic clusters belonging to one
        recording day of one mouse, decorating each row with the
        probe identifier (`imec0`/`imec1`) and the integer
        `cluster_num` parsed from `unit_id`. Used by the probe-trace
        figure to map spike-sort cluster IDs back to bucket colours.

        Parameters
        ----------
        mouse_id (str)
            Catalog `mouse_id` string.
        rec_date (int)
            Recording date as YYYYMMDD integer.

        Returns
        -------
        df (pd.DataFrame)
            Subset of the catalog with extra columns:
              probe (str, e.g. `'imec0'`)
              cluster_num (int)
        """

        df = self.catalog
        sub = df[
            (df["mouse_id"] == mouse_id)
            & (df["rec_date"] == rec_date)
            & (df["cluster_group"] == "good")
            & df["somatic"]
        ].copy()
        sub["probe"] = sub["unit_id"].str.extract(r"(imec\d)", expand=False)
        sub["cluster_num"] = (
            sub["unit_id"].str.extract(r"cl(\d{4})", expand=False).astype(int)
        )
        return sub

    def _gather_probe_context_for_unit(
            self,
            ephys_root: pathlib.Path,
            rec_date: int,
            probe: str,
            clusters: pd.DataFrame,
    ) -> dict:
        """
        Description
        -----------
        Load the per-probe assets needed for the single-unit waveform
        figure: per-cluster Kilosort template, on-probe
        `channel_positions.npy`, and the cluster -> bucket / peak-channel
        mapping derived from the catalog.

        Parameters
        ----------
        ephys_root (pathlib.Path)
            Root containing `<YYYYMMDD>_imec<i>/` directories.
        rec_date (int)
            Recording date as YYYYMMDD integer.
        probe (str)
            `'imec0'` or `'imec1'`.
        clusters (pd.DataFrame)
            Output of `_collect_session_clusters`, filtered to the
            relevant mouse/date.

        Returns
        -------
        ctx (dict)
            See body for keys.
        """

        probe_dir = ephys_root / f"{rec_date}_{probe}"
        ks_dir = probe_dir / "kilosort4"

        probe_clusters = clusters[clusters["probe"] == probe]
        cluster_to_bucket = {
            int(row["cluster_num"]): row["bucket"]
            for _, row in probe_clusters.iterrows()
        }
        cluster_to_peakch = {
            int(row["cluster_num"]): int(row["closest_ch"])
            for _, row in probe_clusters.iterrows()
        }

        # Per-cluster primary template: mode of `spike_templates` over
        # the cluster's spikes. Robust to manual merges where one
        # cluster covers more than one Kilosort template.
        spike_clusters_arr = (
            np.load(ks_dir / "spike_clusters.npy").flatten().astype(np.int64)
        )
        spike_templates_arr = (
            np.load(ks_dir / "spike_templates.npy").flatten().astype(np.int64)
        )
        cluster_to_template: dict[int, int] = {}
        for cluster_num in cluster_to_bucket:
            mask = spike_clusters_arr == cluster_num
            if not mask.any():
                continue
            template_ids, counts = np.unique(
                spike_templates_arr[mask], return_counts=True
            )
            cluster_to_template[cluster_num] = int(template_ids[counts.argmax()])

        templates = np.load(ks_dir / "templates.npy", mmap_mode="r")
        cluster_templates: dict[int, np.ndarray] = {}
        for cluster_num, template_idx in cluster_to_template.items():
            cluster_templates[cluster_num] = np.asarray(
                templates[template_idx], dtype=np.float32
            )

        return {
            "probe": probe,
            "probe_dir": probe_dir,
            "ks_dir": ks_dir,
            "channel_positions": np.load(ks_dir / "channel_positions.npy"),
            "cluster_to_bucket": cluster_to_bucket,
            "cluster_to_peakch": cluster_to_peakch,
            "cluster_templates": cluster_templates,
        }

    @staticmethod
    def _load_ibl_brain_coords(
            histology_root: pathlib.Path,
            mouse_id: str,
            rec_date: int,
            hemisphere: str,
    ) -> dict[str, np.ndarray]:
        """
        Description
        -----------
        Load the IBL-aligned per-channel brain coordinates from one
        session's `channel_locations.json`. Returns three arrays
        indexed by channel number giving (ML, AP, DV) in Bregma µm.
        Hemisphere selects `ibl_LH/` or `ibl_RH/`.

        Parameters
        ----------
        histology_root (pathlib.Path)
            Root containing `<mouse_id>/<rec_date>/ibl_{LH,RH}/
            channel_locations.json`.
        mouse_id (str)
            Catalog `mouse_id` string.
        rec_date (int)
            Recording date as YYYYMMDD integer.
        hemisphere (str)
            `'L'` or `'R'`.

        Returns
        -------
        coords (dict[str, np.ndarray])
            `{'ml': (n_channels,), 'ap': (n_channels,),
              'dv': (n_channels,)}` in Bregma µm.
        """

        hemi_dir = "ibl_RH" if hemisphere.upper().startswith("R") else "ibl_LH"
        ch_loc_path = (
            histology_root / mouse_id / str(rec_date)
            / hemi_dir / "channel_locations.json"
        )
        with ch_loc_path.open() as fh:
            raw = json.load(fh)
        channel_keys = [k for k in raw if k.startswith("channel_")]
        n_channels = len(channel_keys)
        ml = np.empty(n_channels, dtype=float)
        ap = np.empty(n_channels, dtype=float)
        dv = np.empty(n_channels, dtype=float)
        for ch in range(n_channels):
            entry = raw[f"channel_{ch}"]
            ml[ch] = float(entry["x"])
            ap[ch] = float(entry["y"])
            dv[ch] = float(entry["z"])
        return {"ml": ml, "ap": ap, "dv": dv}

    def _draw_single_unit_waveforms_in_brain_space(
            self,
            ax: plt.Axes,
            *,
            ctx: dict,
            cluster_num: int,
            template: np.ndarray,
            peakch: int,
            waveform_width_um: float,
            waveform_voltage_uv_scale: float,
            opacity_sigma_um: float,
            n_neighbors_each_side: int,
            lateral_offset_um: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Description
        -----------
        Same neighbour-selection logic as `_draw_single_unit_waveforms`
        (12 channels on the same shank: peak + sibling + N nearest
        above + N nearest below), but the waveforms are placed at the
        IBL brain `(AP, DV)` of each channel instead of the probe-local
        `(lateral, axial)`. Used to put multiple units in one shared
        anatomical panel.

        Parameters
        ----------
        ax (plt.Axes)
            Target axes.
        ctx (dict)
            Per-probe context dict. Must additionally contain
            `brain_coords` (output of `_load_ibl_brain_coords`).
        cluster_num (int)
            Cluster ID (catalog `cluster_num`).
        template (np.ndarray)
            `(n_samples, n_channels)` Kilosort mean template.
        peakch (int)
            True peak channel of the template (`ptp(...).argmax()`).
        waveform_width_um (float)
            AP-axis width allocated to one waveform.
        waveform_voltage_uv_scale (float)
            Multiplier mapping template voltage units to DV µm.
        opacity_sigma_um (float)
            Gaussian σ for opacity vs probe-local distance from peak.
        n_neighbors_each_side (int)
            Number of nearest same-shank channels above and below the
            peak.

        Returns
        -------
        (ap_drawn, dv_drawn) (tuple[np.ndarray, np.ndarray])
            Brain AP and DV centres of every channel that was
            rendered; used by the caller to compute axis bounds.
        """

        positions = ctx["channel_positions"].astype(float)
        lateral = positions[:, 0]
        axial = positions[:, 1]
        channel_shanks = np.load(
            ctx["ks_dir"] / "channel_shanks.npy"
        ).astype(int)
        bucket = ctx["cluster_to_bucket"][cluster_num]
        colour = self.brain_area_colors[bucket]

        peak_shank = channel_shanks[peakch]
        same_shank_mask = channel_shanks == peak_shank
        same_shank_idx = np.where(same_shank_mask)[0]

        peak_lat = lateral[peakch]
        peak_axi = axial[peakch]
        distances_same = np.sqrt(
            (lateral[same_shank_idx] - peak_lat) ** 2
            + (axial[same_shank_idx] - peak_axi) ** 2
        )
        axial_offset = axial[same_shank_idx] - peak_axi
        above_idx = same_shank_idx[axial_offset < 0]
        below_idx = same_shank_idx[axial_offset > 0]
        above_dist = distances_same[axial_offset < 0]
        below_dist = distances_same[axial_offset > 0]
        same_row_idx = same_shank_idx[axial_offset == 0]
        sibling_idx = same_row_idx[same_row_idx != peakch]
        order_above = np.argsort(above_dist)[:n_neighbors_each_side]
        order_below = np.argsort(below_dist)[:n_neighbors_each_side]
        keep_channels = np.concatenate([
            above_idx[order_above],
            np.array([peakch], dtype=int),
            sibling_idx,
            below_idx[order_below],
        ])

        # Opacity from probe-local distance (same physics as the
        # per-unit panel renderer).
        keep_lat = lateral[keep_channels]
        keep_axi = axial[keep_channels]
        keep_dist = np.sqrt(
            (keep_lat - peak_lat) ** 2 + (keep_axi - peak_axi) ** 2
        )
        sigma2 = float(opacity_sigma_um) ** 2
        keep_opacities = np.exp(-(keep_dist ** 2) / (2.0 * sigma2))

        # Use raw probe-local lateral (real µm). The caller routes
        # each unit to the appropriate per-shank subplot, so each
        # axis only needs to render its own shank's units. The
        # per-unit `lateral_offset_um` shifts the whole 12-channel
        # group sideways so units sharing a peak channel don't fully
        # overlap.
        plot_lat = keep_lat + lateral_offset_um
        plot_axi = keep_axi

        n_samples = template.shape[0]
        t_um = np.linspace(
            -waveform_width_um / 2.0, +waveform_width_um / 2.0, n_samples
        ).astype(np.float32)

        segs_by_op: dict[float, list[np.ndarray]] = {}
        for ch, op, lat_pos, axi_pos in zip(
            keep_channels, keep_opacities, plot_lat, plot_axi
        ):
            op_bin = float(np.round(op, 2))
            # Template is in raw units: somatic AP is a NEGATIVE
            # trough. With matplotlib's default y-axis (no inversion),
            # plotting `wf + axial` puts the trough BELOW the channel
            # position on screen — exactly the downward-spike look we
            # want.
            wf = template[:, ch] * waveform_voltage_uv_scale
            pts = np.column_stack([
                t_um + lat_pos,
                wf + axi_pos,
            ])
            segs_by_op.setdefault(op_bin, []).append(pts)

        for op_bin, segs in segs_by_op.items():
            lw = 0.25 + 0.45 * op_bin
            ax.add_collection(LineCollection(
                segs, colors=colour, linewidths=lw,
                alpha=op_bin,
                zorder=5,
            ))

        return plot_lat, plot_axi

    def _draw_single_unit_waveforms(
            self,
            ax: plt.Axes,
            *,
            ctx: dict,
            cluster_num: int,
            template: np.ndarray,
            peakch: int,
            waveform_width_um: float,
            waveform_voltage_uv_scale: float,
            opacity_sigma_um: float,
            n_neighbors_each_side: int,
            zoom_axial_um: float,
            zoom_lateral_um: float,
    ) -> None:
        """
        Description
        -----------
        Draw one SU-somatic cluster's mean Kilosort waveform on its
        peak channel and on neighbouring channels. Each channel's
        waveform is plotted at its probe-local `(lateral, axial)`
        position; the peak channel is drawn at full opacity, and
        surrounding channels at an opacity that decays as
        `exp(-d² / (2σ²))` with the probe-local distance `d` from the
        peak. The axes are zoomed to a small box around the peak
        channel so the spike shape and its spatial decay are the
        centerpiece.

        Parameters
        ----------
        ax (plt.Axes)
            Target axes.
        ctx (dict)
            Per-probe context dict from
            `_gather_probe_context_for_unit`.
        cluster_num (int)
            Cluster ID (catalog `cluster_num`).
        template (np.ndarray)
            `(n_samples, n_channels)` Kilosort mean template for this
            cluster (already extracted from `templates.npy`).
        waveform_width_um (float)
            Lateral-axis width allocated to one waveform.
        waveform_voltage_uv_scale (float)
            Multiplier mapping template voltage units to axial µm.
        opacity_sigma_um (float)
            Gaussian σ for opacity vs distance-from-peak.
        n_neighbors_each_side (int)
            Number of nearest same-shank channels above and below the
            peak to include in the panel.
        zoom_axial_um (float)
            Half-extent (µm) above and below the peak channel.
        zoom_lateral_um (float)
            Half-extent (µm) left and right of the peak channel.

        Returns
        -------
        None
        """

        positions = ctx["channel_positions"].astype(float)
        lateral = positions[:, 0]
        axial = positions[:, 1]
        channel_shanks = np.load(
            ctx["ks_dir"] / "channel_shanks.npy"
        ).astype(int)
        bucket = ctx["cluster_to_bucket"][cluster_num]
        colour = self.brain_area_colors[bucket]

        peak_lat = lateral[peakch]
        peak_axi = axial[peakch]
        peak_shank = channel_shanks[peakch]

        # Restrict to channels on the same shank as the peak. Spike
        # spread does not cross to neighbouring shanks (250 µm gap).
        same_shank_mask = channel_shanks == peak_shank
        same_shank_idx = np.where(same_shank_mask)[0]

        # Probe-local physical distance from peak (lateral + axial),
        # computed only for same-shank channels.
        distances_same = np.sqrt(
            (lateral[same_shank_idx] - peak_lat) ** 2
            + (axial[same_shank_idx] - peak_axi) ** 2
        )

        # Partition by axial side: strictly above the peak vs.
        # strictly below, plus the "sibling" at the SAME axial row as
        # the peak (the other column of the shank — the very nearest
        # contact to the peak, 32 µm lateral away).
        axial_offset = axial[same_shank_idx] - peak_axi
        above_idx = same_shank_idx[axial_offset < 0]
        below_idx = same_shank_idx[axial_offset > 0]
        above_dist = distances_same[axial_offset < 0]
        below_dist = distances_same[axial_offset > 0]

        same_row_idx = same_shank_idx[axial_offset == 0]
        sibling_idx = same_row_idx[same_row_idx != peakch]

        # Pick the N nearest on each side.
        order_above = np.argsort(above_dist)[:n_neighbors_each_side]
        order_below = np.argsort(below_dist)[:n_neighbors_each_side]
        keep_channels = np.concatenate([
            above_idx[order_above],
            np.array([peakch], dtype=int),
            sibling_idx,
            below_idx[order_below],
        ])

        # Physical distance from peak for each kept channel, used to
        # set the per-channel opacity.
        keep_lat = lateral[keep_channels]
        keep_axi = axial[keep_channels]
        keep_dist = np.sqrt(
            (keep_lat - peak_lat) ** 2 + (keep_axi - peak_axi) ** 2
        )
        sigma2 = float(opacity_sigma_um) ** 2
        keep_opacities = np.exp(-(keep_dist ** 2) / (2.0 * sigma2))

        n_samples = template.shape[0]
        t_um = np.linspace(
            -waveform_width_um / 2.0, +waveform_width_um / 2.0, n_samples
        ).astype(np.float32)

        # Group waveform segments by rounded opacity so we can issue a
        # small number of LineCollection draws.
        segs_by_op: dict[float, list[np.ndarray]] = {}
        for ch, op in zip(keep_channels, keep_opacities):
            op_bin = float(np.round(op, 2))
            wf = template[:, ch] * waveform_voltage_uv_scale
            pts = np.column_stack([
                t_um + lateral[ch],
                wf + axial[ch],
            ])
            segs_by_op.setdefault(op_bin, []).append(pts)

        for op_bin, segs in segs_by_op.items():
            lw = 0.25 + 0.45 * op_bin
            ax.add_collection(LineCollection(
                segs, colors=colour, linewidths=lw,
                alpha=op_bin,
                zorder=5,
            ))

        ax.set_xlim(peak_lat - zoom_lateral_um, peak_lat + zoom_lateral_um)
        ax.set_ylim(peak_axi - zoom_axial_um, peak_axi + zoom_axial_um)
        ax.set_aspect("equal")
        ax.set_axis_off()


    def _compute_bucket_bboxes(self) -> dict[str, dict]:
        """
        Description
        -----------
        Compute per-bucket Allen-mesh bounding boxes in stereotaxic-
        Bregma µm. For buckets backed by multiple meshes (only `SC`,
        which is `SCm ∪ SCs`), the bounding box is the union of all
        member-mesh vertex spans. The result feeds the outlier filter
        in `build_unit_positions_figure`.

        Parameters
        ----------
        None

        Returns
        -------
        bboxes (dict[str, dict])
            Mapping `bucket -> {'ap': (lo, hi), 'dv': (lo, hi),
            'ml': (lo, hi)}` for the six anatomically-defined buckets
            (PAG, MRN, VTA, MB, CENT, SC). The `'other'` bucket has
            no mesh and is intentionally absent.
        """

        bucket_to_meshes = {
            "PAG":  ["PAG"],
            "MRN":  ["MRN"],
            "VTA":  ["VTA"],
            "MB":   ["MB"],
            "CENT": ["CENT"],
            "SC":   ["SCm", "SCs"],
        }
        bboxes: dict[str, dict] = {}
        for bucket, mesh_keys in bucket_to_meshes.items():
            verts_list = []
            for k in mesh_keys:
                v_ccf, _ = _load_obj_mesh(
                    _download_allen_mesh(_ALLEN_STRUCTURE_IDS[k])
                )
                verts_list.append(_ccf_to_bregma(v_ccf))
            verts = np.vstack(verts_list)
            bboxes[bucket] = {
                "ap": (float(verts[:, 0].min()), float(verts[:, 0].max())),
                "dv": (float(verts[:, 1].min()), float(verts[:, 1].max())),
                "ml": (float(verts[:, 2].min()), float(verts[:, 2].max())),
            }
        return bboxes

    def _add_mesh_to_axes(
            self,
            ax,
            *,
            structure_id: int,
            face_color: str,
            face_alpha: float,
            face_stride: int = 1,
            rasterized: bool = False,
    ) -> None:
        """
        Description
        -----------
        Download (if needed), parse, transform, and add one Allen
        CCFv3 structure mesh as a translucent `Poly3DCollection` to
        the given 3D axes. Vertex coordinates are converted from CCF
        µm to stereotaxic-Bregma µm via `_ccf_to_bregma`, then
        re-ordered so the matplotlib `(x, y, z)` axes correspond to
        `(AP_stereo, ML_stereo, DV_stereo)` — which lines them up with
        the scatter dots that use the catalog's
        `(loc_ap, loc_ml, loc_dv)` columns in the same order.

        Parameters
        ----------
        ax (mpl_toolkits.mplot3d.axes3d.Axes3D)
            Target 3D axes.
        structure_id (int)
            Allen CCFv3 structure ID.
        face_color (str)
            Hex colour for the triangulated face fill.
        face_alpha (float)
            Alpha applied to the face fill.
        face_stride (int)
            Take every Nth face. `1` keeps all faces; >1 decimates
            (used for the whole-brain root mesh to stay responsive).
        rasterized (bool)
            When True, mark the resulting `Poly3DCollection` as
            rasterized so an SVG export bakes the (thousands of)
            translucent triangles into an embedded PNG rather than
            emitting each one as a separate `<polygon>`. Cuts SVG
            file size by ~100x and makes downstream editing
            (Illustrator, Inkscape) actually usable. Text / legend /
            axis labels stay vector regardless.

        Returns
        -------
        None
        """

        mesh_path = _download_allen_mesh(structure_id)
        verts_ccf, faces = _load_obj_mesh(mesh_path)
        verts_bregma = _ccf_to_bregma(verts_ccf)
        if face_stride > 1:
            faces = faces[::face_stride]
        # tris: (n_tri, 3 vertices, 3 coords) in (AP, DV, ML) order.
        tris = verts_bregma[faces]
        # Re-order to (AP, ML, DV) so matplotlib (x, y, z) lines up
        # with the scatter dots.
        tris_plot = np.stack(
            [tris[..., 0], tris[..., 2], tris[..., 1]],
            axis=-1,
        )
        coll = Poly3DCollection(
            tris_plot,
            alpha=face_alpha,
            facecolor=face_color,
            edgecolor="none",
            linewidth=0,
        )
        if rasterized:
            coll.set_rasterized(True)
        # Force the painter's algorithm to treat this mesh as the
        # FARTHEST artist on every frame, so the scatter dots layered
        # later always paint on top of it. Fixes the rotation-dependent
        # "dots disappear inside the colored volume" artefact caused by
        # matplotlib 3D's per-artist centroid depth sort.
        coll.set_sort_zpos(1e10)
        ax.add_collection3d(coll)
