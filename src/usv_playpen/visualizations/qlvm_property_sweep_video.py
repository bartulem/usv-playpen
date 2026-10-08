"""
@author: bartulem

Property-sweep video of the regular QLVM map: where the low, middle and high
USVs of every measured property sit on the torus.

Six panels (2 x 3), one per USV property -- SAM-mask count, duration, frequency
bandwidth, spectral entropy, mean frequency and loudness -- sweep together. At
every frame each panel shows the USVs between the same two percentiles of ITS
property: an equal-count window holding ``window`` of the USVs, moved in steps
of ``step`` from the lowest to the highest values and held ``hold`` frames at
both ends. Each panel draws:

* the window's USV density on the category grid of the regular map (counts per
  pixel convolved with a periodic Gaussian of ``density_bandwidth`` torus
  units, divided by the field's own 99th percentile), in the colour of the
  window's median value on the property's colour scale (the 1st-99th
  percentile of all USVs on the colormap), brightness = density, on black;
* the QLVM category boundaries (``os_utils.load_qlvm_category_bundle``) as
  dotted white lines;
* a colour bar in the property's unit, with the current window shaded and its
  median marked.

All text renders in the project font (Helvetica Light) through the shared
apply_plot_style, which the renderer applies itself.

The USVs come from the per-session ``*_usv_summary.csv`` files of the cohort,
pooled with ``build_pooled_embeddings_df`` (parquet cache under
``<spectrograms_dir>/embeddings``, rebuilt automatically when the summaries
change): non-noise, pure USVs only (``usv & ~squeak``; squeaks and segments
holding both are left out), positions on the regular map (``qlvm1`` /
``qlvm2``), and only USVs with all six properties. USVs with equal values (the
mask count is an integer, so about half the USVs tie at one mask) are put in a
random order within each tie (``seed``), so a window inside a tie is a random
subset rather than a block of sessions in file order.

The colormap is the project's ``sequential_cmap`` (``figures`` settings block),
used from ``colormap_floor`` upward: the density is drawn as brightness on
black, so the black low end of a colormap such as inferno would make the
windows of the lowest values invisible.

The video is written with matplotlib's ``FFMpegWriter`` (ffmpeg on the PATH);
``render_qlvm_property_sweep_video`` can also return one still frame instead.
"""

from __future__ import annotations

import pathlib
from collections.abc import Callable

import numpy as np
import polars as pls
from matplotlib import colormaps
from matplotlib.animation import FFMpegWriter
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.figure import Figure

from ..os_utils import configure_path
from .auxiliary_plot_functions import periodic_density
from .make_usv_spectrograms import load_regular_map_cohort_usvs
from .plot_style import apply_plot_style

# (usv_summary column, panel label, factor from the stored value to the shown unit, unit), in panel order.
SWEEP_PROPERTIES = (
    ("mask_number", "mask count", 1.0, "masks"),
    ("duration", "duration", 1e3, "ms"),
    ("freq_bandwidth_hz", "bandwidth", 1e-3, "kHz"),
    ("spectral_entropy", "spectral entropy", 1.0, "nats"),
    ("mean_freq_hz", "mean frequency", 1e-3, "kHz"),
    ("loudness_db", "loudness", 1.0, "dB"),
)
SWEEP_POSITION_COLUMNS = ("qlvm1", "qlvm2")
GRID_SHAPE = (2, 3)
FIGURE_SIZE = (19.0, 11.4)
FIGURE_FACE = "#FFFFFF"
BOUNDARY_COLOR = "#FFFFFF"
MEDIAN_MARK_COLOR = "#000000"
WINDOW_SHADE_COLOR = "#FFFFFF"
COLORBAR_LABEL_SIZE = 17
COLORBAR_TICK_SIZE = 14


def load_property_sweep_usvs(
    input_files_directory: str,
    cache_path: str | None,
    exclude_noise_usvs: bool,
    message_output: Callable | None = None,
) -> pls.DataFrame:
    """
    Description
    -----------
    Pools the cohort's USVs for the property-sweep video with
    ``make_usv_spectrograms.load_regular_map_cohort_usvs``: every session of the
    ``*sessions_list.txt`` files under ``input_files_directory`` (playback lists
    dropped), the pure USVs (``usv & ~squeak``) with a regular-map position and
    all six ``SWEEP_PROPERTIES`` finite.

    Parameters
    ----------
    input_files_directory (str)
        Directory holding the cohort's ``*sessions_list.txt`` files.
    cache_path (str | None)
        Pooled-embeddings parquet cache (``os_utils.resolve_pooled_embeddings_cache``);
        ``None`` pools from the summaries without reading or writing a cache.
    exclude_noise_usvs (bool)
        Whether to drop the segments ``detect_usv_noise`` flagged as holding no vocalization.
    message_output (Callable | None)
        Logger; defaults to ``print``.

    Returns
    -------
    usvs (pls.DataFrame)
        ``qlvm1``, ``qlvm2`` and the six property columns, one row per USV.
    """

    usvs = load_regular_map_cohort_usvs(input_files_directory, cache_path, exclude_noise_usvs, "property-sweep",
                                        message_output=message_output)
    return usvs.select([*SWEEP_POSITION_COLUMNS, *(column for column, _, _, _ in SWEEP_PROPERTIES)])


class PropertySweepPanel:
    """
    Description
    -----------
    One property's panel of the sweep: the torus image, the category boundaries
    and the colour bar are drawn once; ``update`` redraws the current window.
    """

    def __init__(self, subfigure, rows: np.ndarray, cols: np.ndarray, values: np.ndarray, order: np.ndarray,
                 label: str, scale: float, unit: str, label_grid: np.ndarray, colormap: ListedColormap,
                 window: float, density_bandwidth: float) -> None:
        """
        Description
        -----------
        Builds the panel's axes, boundaries and colour bar.

        Parameters
        ----------
        subfigure (matplotlib.figure.SubFigure)
            The subfigure the panel fills.
        rows (np.ndarray)
            Pixel row (y) of each USV on the category grid.
        cols (np.ndarray)
            Pixel column (x) of each USV on the category grid.
        values (np.ndarray)
            The property's stored value for each USV.
        order (np.ndarray)
            USV indices sorted by value, ties in random order.
        label (str)
            Property name shown in the title and on the colour bar.
        scale (float)
            Factor from the stored value to the shown unit.
        unit (str)
            Shown unit.
        label_grid (np.ndarray)
            Category grid of the regular map, ``[y, x]``.
        colormap (ListedColormap)
            The property colour scale (the project colormap from its floor up).
        window (float)
            Share of the USVs in each window.
        density_bandwidth (float)
            Gaussian standard deviation of the density, in torus units.

        Returns
        -------
        None
        """

        self.rows, self.cols, self.values, self.order = rows, cols, values, order
        self.label, self.scale, self.unit = label, scale, unit
        self.colormap, self.window, self.density_bandwidth = colormap, window, density_bandwidth
        self.resolution = label_grid.shape[0]
        self.vmin, self.vmax = (float(np.percentile(values, q)) * scale for q in (1, 99))
        self.norm = Normalize(self.vmin, self.vmax)

        subfigure.subplots_adjust(left=0.02, right=0.88, bottom=0.03, top=0.86)
        grid = subfigure.add_gridspec(1, 2, width_ratios=[1, 0.04], wspace=0.04)
        self.axis = subfigure.add_subplot(grid[0, 0])
        self.image = self.axis.imshow(np.zeros((self.resolution, self.resolution, 3)), origin="lower",
                                      extent=(0, 1, 0, 1), interpolation="nearest")
        centres = (np.arange(self.resolution) + 0.5) / self.resolution
        for category in np.unique(label_grid):
            self.axis.contour(centres, centres, (label_grid == category).astype(float), levels=[0.5], zorder=3,
                              colors=BOUNDARY_COLOR, linewidths=1.6, linestyles="dotted", alpha=0.95)
        self.axis.set_xlim(0, 1)
        self.axis.set_ylim(0, 1)
        self.axis.set_aspect("equal")
        self.axis.set_xticks([])
        self.axis.set_yticks([])
        self.colorbar_axis = subfigure.add_subplot(grid[0, 1])
        colorbar = subfigure.colorbar(ScalarMappable(self.norm, colormap), cax=self.colorbar_axis)
        colorbar.set_label(f"{label} ({unit})" if unit else label, fontsize=COLORBAR_LABEL_SIZE)
        colorbar.ax.tick_params(labelsize=COLORBAR_TICK_SIZE)
        self.marks = []

    def update(self, position: float) -> None:
        """
        Description
        -----------
        Draws the window of USVs centred on the ``position`` quantile of the
        property: its density in the colour of its median value, the title
        with its value range, and the window and median on the colour bar.

        Parameters
        ----------
        position (float)
            Window centre, as a quantile in [window / 2, 1 - window / 2].

        Returns
        -------
        None
        """

        n = self.values.size
        width = int(self.window * n)
        start = int(np.clip(round(position * n - width / 2), 0, n - width))
        chosen = self.order[start:start + width]
        low, high, median = (float(np.percentile(self.values[chosen], q)) * self.scale for q in (0, 100, 50))
        colour = np.array(self.colormap(self.norm(median))[:3])
        density = periodic_density(self.rows[chosen], self.cols[chosen], self.resolution, self.density_bandwidth)
        brightness = np.clip(density / np.percentile(density, 99), 0, 1)
        self.image.set_data(brightness[..., None] * colour[None, None, :])
        self.axis.set_title(f"{self.label}: {low:.3g} to {high:.3g} {self.unit}".strip(), fontsize=15)
        for mark in self.marks:
            mark.remove()
        self.marks = [self.colorbar_axis.axhspan(max(low, self.vmin), min(high, self.vmax), color=WINDOW_SHADE_COLOR,
                                                 alpha=0.35, zorder=3),
                      self.colorbar_axis.axhline(min(max(median, self.vmin), self.vmax), color=MEDIAN_MARK_COLOR,
                                                 linewidth=2.5, zorder=4)]


def render_qlvm_property_sweep_video(
    usvs: pls.DataFrame,
    label_grid: np.ndarray,
    output_path: str,
    cmap_name: str,
    colormap_floor: float,
    window: float,
    step: float,
    hold: int,
    fps: int,
    dpi: int,
    density_bandwidth: float,
    seed: int,
    preview_quantile: float | None = None,
    message_output: Callable | None = None,
) -> Figure:
    """
    Description
    -----------
    Renders the property-sweep video (module docstring) to ``output_path``, or,
    with ``preview_quantile``, one still frame of the window centred on that
    quantile, saved as a ``.png`` next to ``output_path``.

    Parameters
    ----------
    usvs (pls.DataFrame)
        ``qlvm1``, ``qlvm2`` and the six ``SWEEP_PROPERTIES`` columns
        (``load_property_sweep_usvs``).
    label_grid (np.ndarray)
        Category grid of the regular map, ``[y, x]`` (``os_utils.load_qlvm_category_bundle()['label_grid']``).
    output_path (str)
        The ``.mp4`` to write (run through ``configure_path``; parent created).
    cmap_name (str)
        Matplotlib colormap of the property colour scales (the project's ``sequential_cmap``).
    colormap_floor (float)
        Lowest point of the colormap used, in [0, 1): the scale runs from this
        point to the top so the lowest values stay visible on black.
    window (float)
        Share of the USVs in each window (e.g. 0.10).
    step (float)
        Window-centre step between frames, as a share of the USVs (e.g. 0.01).
    hold (int)
        Frames held at the first and at the last window.
    fps (int)
        Video frame rate.
    dpi (int)
        Frame resolution (the frame is ``FIGURE_SIZE`` inches).
    density_bandwidth (float)
        Gaussian standard deviation of the density, in torus units.
    seed (int)
        Seed of the random order within tied values.
    preview_quantile (float | None)
        When set, render only the window centred on this quantile and save it
        as a still instead of the video.
    message_output (Callable | None)
        Logger; defaults to ``print``.

    Returns
    -------
    fig (Figure)
        The figure, showing the last frame drawn.
    """

    log = message_output or print
    apply_plot_style()
    resolution = label_grid.shape[0]
    x = usvs[SWEEP_POSITION_COLUMNS[0]].to_numpy()
    y = usvs[SWEEP_POSITION_COLUMNS[1]].to_numpy()
    rows = np.clip((y * resolution).astype(int), 0, resolution - 1)
    cols = np.clip((x * resolution).astype(int), 0, resolution - 1)
    base = colormaps[cmap_name]
    colormap = ListedColormap(base(np.linspace(colormap_floor, 1.0, 256)), name=f"{cmap_name}_from_{colormap_floor:g}")
    tie_breaker = np.random.default_rng(seed).random(usvs.height)

    fig = Figure(figsize=FIGURE_SIZE, facecolor=FIGURE_FACE)
    subfigures = fig.subfigures(*GRID_SHAPE, wspace=0.02, hspace=0.04).ravel()
    panels = []
    for subfigure, (column, label, scale, unit) in zip(subfigures, SWEEP_PROPERTIES, strict=True):
        values = usvs[column].to_numpy().astype(float)
        panels.append(PropertySweepPanel(subfigure, rows, cols, values, np.lexsort((tie_breaker, values)), label, scale,
                                         unit, label_grid, colormap, window, density_bandwidth))
    header = fig.suptitle("", y=0.995, fontsize=16)

    def show(position: float) -> None:
        for panel in panels:
            panel.update(position)
        header.set_text(f"USVs between the {100 * (position - window / 2):.0f}th and "
                        f"{100 * (position + window / 2):.0f}th percentile of each property ({usvs.height:,} USVs)")

    out = pathlib.Path(configure_path(output_path))
    out.parent.mkdir(parents=True, exist_ok=True)
    if preview_quantile is not None:
        show(preview_quantile)
        still = out.with_suffix(".png")
        fig.savefig(still, dpi=dpi, facecolor=FIGURE_FACE)
        log(f"[property-sweep] preview -> {still}")
        return fig

    positions = np.arange(window / 2, 1 - window / 2 + 1e-9, step)
    sequence = [positions[0]] * hold + list(positions) + [positions[-1]] * hold
    writer = FFMpegWriter(fps=fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-crf", "18"])
    with writer.saving(fig, str(out), dpi):
        for position in sequence:
            show(position)
            writer.grab_frame(facecolor=FIGURE_FACE)
    log(f"[property-sweep] {len(sequence)} frames -> {out} ({len(sequence) / fps:.1f} s)")
    return fig
