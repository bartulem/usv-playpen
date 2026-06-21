"""
@author: bartulem
Render a two-panel QLVM toroidal-latent traversal video.

Animates a guided walk across the trained QLVM torus: the left panel shows the
latent-density heatmap with periodic-watershed cluster boundaries, the cluster
centers, and a moving trajectory with a fading trail; the right panel shows the
spectrogram of the data sample nearest (on the torus) to the current trajectory
position. The result is written as ``.mp4`` (FFmpeg) or ``.gif`` (Pillow).

Inputs:
* ``arrays_coarse.npz`` / ``arrays_fine.npz`` -- used ONLY for the ``heatmap``
  (res, res) background, ``ws_labels_periodic`` (res, res) cluster contours, and
  ``centers`` (K, 2) path waypoints (coarse = 7 clusters, fine = 12).
* the consolidated spectrogram H5 -- supplies BOTH the per-USV latent coords
  (per-session ``spectrogram/<key>/qlvm_dim``, written once by a one-off
  enrichment of the store) for the nearest-neighbour lookup AND the spectrograms
  (``spectrogram/<key>/spectrograms``) shown in the right panel. No latents
  pickle / flat ``spectrograms`` npz is needed at render time.

This is the in-house, torch-free port of ``qmc_deep_gen``'s
``inference_latents_video.py``. The original ``torus_forward`` is pure numpy and
the video shows real data spectrograms (it never runs the decoder), so no model
is needed here. It is a faithful *core* of the original: the two-panel torus
traversal between cluster centers. The original's additional decorative phases
(title cards, concentric-ring and grid montages) are intentionally omitted for
maintainability; the trajectory + nearest-sample panels are preserved.
"""

from __future__ import annotations

import itertools
import pathlib
from collections.abc import Callable
from datetime import datetime

import click
import h5py
import matplotlib
import numpy as np
from click.core import ParameterSource
from sklearn.neighbors import NearestNeighbors

from ..cli_utils import modify_settings_json_for_cli
from ..os_utils import configure_path
from .plot_style import apply_plot_style

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # Agg backend is selected above, before importing pyplot
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter

# Register the bundled Helvetica weights + activate the project mplstyle so all
# text in the video renders in Helvetica (titles forced to Light below), instead
# of matplotlib's default font.
apply_plot_style()

# Hex palette (every matplotlib color= arg is a hex string).
_TRAIL_COLOR = "#DC143C"      # crimson trajectory trail
_MARKER_COLOR = "#FFFFFF"     # current-position marker
_CENTER_COLOR = "#000000"     # cluster-center markers
_CONTOUR_COLOR = "#000000"    # watershed boundary contours


def torus_forward(coords: np.ndarray) -> np.ndarray:
    """
    Description
    -----------
    Pure-numpy torus embedding ``[cos(2*pi*coords), sin(2*pi*coords)]`` used for
    nearest-neighbour queries on the torus (so wrap-around is respected).

    Parameters
    ----------
    coords (np.ndarray)
        Latent coordinates in ``[0, 1)``, shape ``(N, d)``.

    Returns
    -------
    embedding (np.ndarray)
        Torus embedding, shape ``(N, 2d)``.
    """
    return np.concatenate([np.cos(2 * np.pi * coords), np.sin(2 * np.pi * coords)], axis=1)


def pool_latents_from_h5(h5) -> tuple[np.ndarray, list[tuple[str, int]]]:
    """
    Description
    -----------
    Pools every session's per-USV latent coordinates from the consolidated H5's
    ``spectrogram/<key>/qlvm_dim`` datasets (written once by a one-off
    enrichment of the store) into one array, with a parallel
    ``(session key, spectrogram row)`` list so a nearest-neighbour hit can be
    mapped straight back to a spectrogram. Rows with NaN coords are dropped.

    Parameters
    ----------
    h5 (h5py.File)
        Open consolidated spectrogram store (read mode).

    Returns
    -------
    coords, index (tuple)
        ``coords`` is an ``(N, 2)`` float array of latent coordinates; ``index``
        is a length-``N`` ``list[(session_key, row)]`` aligned with it.
    """
    coords_chunks: list[np.ndarray] = []
    index: list[tuple[str, int]] = []
    spec_group = h5["spectrogram"]
    for session_key in spec_group:
        session_h5 = spec_group[session_key]
        if "qlvm_dim" not in session_h5:
            continue
        session_coords = session_h5["qlvm_dim"][:]
        valid_rows = ~np.isnan(session_coords).any(axis=1)
        if not valid_rows.any():
            continue
        coords_chunks.append(session_coords[valid_rows])
        index.extend((session_key, int(row)) for row in np.nonzero(valid_rows)[0])

    if not coords_chunks:
        return np.empty((0, 2), dtype=np.float64), index
    return np.concatenate(coords_chunks, axis=0), index


def build_traversal_path(centers: np.ndarray, frames_per_segment: int, curvature: float) -> np.ndarray:
    """
    Description
    -----------
    Builds a looped trajectory visiting each cluster center in turn. Each segment
    takes the shortest toroidal route (wrapping when the gap exceeds 0.5) and is
    bowed by a sinusoidal perpendicular offset of magnitude ``curvature``; the
    final path is reduced mod 1 back onto the torus.

    Parameters
    ----------
    centers (np.ndarray)
        Cluster centers in ``[0, 1)^2``, shape ``(K, 2)``.
    frames_per_segment (int)
        Interpolation frames between consecutive centers.
    curvature (float)
        Amplitude of the perpendicular bow on each segment.

    Returns
    -------
    path (np.ndarray)
        Trajectory points in ``[0, 1)^2``, shape ``(K * frames_per_segment, 2)``.
    """
    ordered = np.vstack([centers, centers[:1]])  # loop back to the first center
    t = np.linspace(0.0, 1.0, frames_per_segment, endpoint=False)
    segments = []
    for a, b in itertools.pairwise(ordered):
        target = b.copy()
        delta = target - a
        target = target - np.where(np.abs(delta) > 0.5, np.sign(delta), 0.0)  # shortest wrapped route
        straight = a[None, :] + t[:, None] * (target - a)[None, :]
        # Perpendicular bow that vanishes at both endpoints.
        direction = target - a
        norm = np.linalg.norm(direction)
        if norm > 1e-9:
            perp = np.array([-direction[1], direction[0]]) / norm
            straight = straight + (curvature * np.sin(np.pi * t))[:, None] * perp[None, :]
        segments.append(straight)
    return np.concatenate(segments, axis=0) % 1.0


class QLVMTorusTraversalVideo:
    """
    Description
    -----------
    Builds the two-panel QLVM torus-traversal animation from the inference
    ``arrays.npz`` and a curated spectrogram ``.npz``.
    """

    def __init__(
        self,
        output_path: str | None = None,
        input_parameter_dict: dict | None = None,
        message_output: Callable | None = None,
    ) -> None:
        """
        Description
        -----------
        Initializes the QLVMTorusTraversalVideo.

        Parameters
        ----------
        output_path (str)
            Output ``.mp4`` / ``.gif`` path.
        input_parameter_dict (dict)
            Visualization settings; the ``qlvm_torus_traversal_video`` block
            supplies the input paths and render parameters.
        message_output (Callable)
            Logging callback; defaults to ``print``.

        Returns
        -------
        None
        """
        self.output_path = output_path
        self.input_parameter_dict = input_parameter_dict if input_parameter_dict is not None else {}
        self.message_output = message_output if message_output is not None else print

    def make_video(self) -> None:
        """
        Description
        -----------
        Loads the latent arrays + spectrograms, builds the traversal path, and
        renders the two-panel animation to ``output_path`` (``.gif`` via Pillow,
        otherwise ``.mp4`` via FFmpeg).

        Parameters
        ----------

        Returns
        -------
        A video file at ``output_path``.
        """
        self.message_output(
            f"QLVM torus-traversal video started at: {datetime.now().hour:02d}:{datetime.now().minute:02d}:{datetime.now().second:02d}."
        )

        cfg = self.input_parameter_dict['qlvm_torus_traversal_video']

        # Coarse (fewer clusters) vs fine: the two arrays files share latent
        # coords / heatmap and differ only in centers / watershed labels. Here
        # the arrays are used ONLY for the heatmap background, the watershed
        # contours, and the cluster-center path waypoints.
        clustering = cfg['clustering']
        arrays_path = cfg['arrays_npz_path_fine'] if clustering == "fine" else cfg['arrays_npz_path_coarse']
        arrays = np.load(configure_path(arrays_path))
        heatmap = arrays['heatmap']
        ws_labels_periodic = arrays['ws_labels_periodic']
        centers = arrays['centers']

        path = build_traversal_path(centers, cfg['frames_per_segment'], cfg['path_curvature'])
        trail_length = cfg['trail_length']
        res = heatmap.shape[0]
        grid = np.linspace(0, 1, res)

        # Auto-derive the output path (figures.save_directory + timestamp) when the
        # caller did not supply one (the GUI path); the CLI may still pass one.
        if self.output_path is None:
            stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
            out_path = pathlib.Path(
                configure_path(self.input_parameter_dict['figures']['save_directory'])
            ) / f"qlvm_torus_traversal_{stamp}.mp4"
        else:
            out_path = pathlib.Path(self.output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(configure_path(cfg['consolidated_h5_path']), "r") as h5:
            # Latent coords + (session, row) come straight from the H5's per-session
            # qlvm_dim datasets (written once by a one-off enrichment); the nearest
            # sample's spectrogram is then read from the same store.
            pooled_coords, pooled_index = pool_latents_from_h5(h5)
            if pooled_coords.shape[0] == 0:
                raise ValueError(
                    "No `qlvm_dim` coordinates found in the consolidated H5 — the "
                    "store must be enriched once with per-session qlvm_dim latent "
                    "coords before rendering."
                )
            nn_index = NearestNeighbors(n_neighbors=1, algorithm="ball_tree").fit(torus_forward(pooled_coords))

            def spec_for(flat_idx: int) -> np.ndarray:
                """Spectrogram for a pooled latent index, read from the consolidated H5."""
                session_key, row = pooled_index[flat_idx]
                return h5["spectrogram"][session_key]["spectrograms"][row]

            fig, (ax_map, ax_spec) = plt.subplots(1, 2, figsize=(12, 6), dpi=cfg['dpi'])

            ax_map.imshow(heatmap, origin="lower", extent=(0, 1, 0, 1), cmap="magma", aspect="auto")
            ax_map.contour(grid, grid, ws_labels_periodic, levels=np.arange(0.5, ws_labels_periodic.max() + 1),
                           colors=_CONTOUR_COLOR, linewidths=0.6)
            ax_map.scatter(centers[:, 0], centers[:, 1], c=_CENTER_COLOR, s=20, zorder=4)
            (trail_line,) = ax_map.plot([], [], color=_TRAIL_COLOR, lw=2.0, zorder=5)
            (marker,) = ax_map.plot([], [], marker="o", color=_MARKER_COLOR, markersize=8,
                                    markeredgecolor=_CENTER_COLOR, zorder=6)
            ax_map.set_xlim(0, 1)
            ax_map.set_ylim(0, 1)
            ax_map.set_title("QLVM torus", fontsize=12, fontweight="light")

            first_nearest = int(nn_index.kneighbors(torus_forward(path[0][None]), return_distance=False)[0, 0])
            spec_im = ax_spec.imshow(spec_for(first_nearest), origin="lower", aspect="auto", cmap="viridis")
            ax_spec.set_title("nearest USV spectrogram", fontsize=12, fontweight="light")
            ax_spec.set_xticks([])
            ax_spec.set_yticks([])

            def update(frame_idx: int):
                """Advance the marker/trail and show the nearest sample's spectrogram."""
                xy = path[frame_idx]
                lo = max(0, frame_idx - trail_length)
                trail_line.set_data(path[lo:frame_idx + 1, 0], path[lo:frame_idx + 1, 1])
                marker.set_data([xy[0]], [xy[1]])
                nearest = int(nn_index.kneighbors(torus_forward(xy[None]), return_distance=False)[0, 0])
                spec_im.set_data(spec_for(nearest))
                return trail_line, marker, spec_im

            ani = FuncAnimation(fig, update, frames=len(path), interval=1000 / cfg['fps'], blit=False)

            if out_path.suffix.lower() == ".gif":
                writer: PillowWriter | FFMpegWriter = PillowWriter(fps=cfg['fps'])
            else:
                writer = FFMpegWriter(fps=cfg['fps'], bitrate=4000)
            ani.save(str(out_path), writer=writer, dpi=cfg['dpi'])
            plt.close(fig)

        self.message_output(
            f"Wrote {len(path)}-frame traversal video -> {out_path}."
        )
        self.message_output(
            f"QLVM torus-traversal video ended at: {datetime.now().hour:02d}:{datetime.now().minute:02d}:{datetime.now().second:02d}."
        )


@click.command(name="qlvm-torus-traversal-video")
@click.option('--output-path', type=str, default=None, required=False, help='Output .mp4 / .gif path (default: figures.save_directory + timestamp).')
@click.option('--frames-per-segment', 'frames_per_segment', type=int, default=None, required=False, help='Interpolation frames between cluster centers.')
@click.option('--fps', 'fps', type=int, default=None, required=False, help='Video frames per second.')
@click.pass_context
def qlvm_torus_traversal_video_cli(ctx, output_path, **kwargs) -> None:
    """
    Description
    -----------
    A command-line tool to render the QLVM torus-traversal video.

    Parameters
    ----------

    Returns
    -------
    None
    """
    provided_params = [key for key in kwargs if ctx.get_parameter_source(key) == ParameterSource.COMMANDLINE]

    visualizations_settings_dict = modify_settings_json_for_cli(
        ctx=ctx,
        provided_params=provided_params,
        settings_dict='visualizations_settings',
    )

    QLVMTorusTraversalVideo(
        output_path=output_path,
        input_parameter_dict=visualizations_settings_dict,
        message_output=print,
    ).make_video()
