"""
@author: bartulem
Render a two-panel traversal video over the QLVM latent torus.

A matplotlib ``FuncAnimation`` builder that renders a guided "torus walkthrough"
from the QLVM analysis arrays + the consolidated spectrogram store. It builds a
ball-tree nearest-neighbour index over the torus embedding of every USV's latent
coordinate, assembles an ordered list of animation "phases", precomputes
per-traversal reveal/inset positions, then animates the two panels frame by frame
and writes an ``.mp4`` (FFmpeg) or ``.gif`` (Pillow).

This is the in-house, torch-free port of ``qmc_deep_gen``'s
``inference_latents_video.py``. The original loaded a ``mouse_data`` dataset
(``full_data.pt``) index-aligned with ``latent_coords``; here the per-USV latent
coordinates AND the spectrograms both come from the consolidated H5 (per-session
``spectrogram/<key>/qlvm_dim`` -- written once by a one-off enrichment -- and
``spectrogram/<key>/spectrograms``), pooled into one ordered array. The arrays
``.npz`` is used only for the ``heatmap`` background, ``ws_labels_periodic``
contours, and cluster ``centers``.

Layout:
  left  = [0,1]^2 latent map (heatmap + watershed contours, no axes/ticks) with a
          recency-coloured trajectory trail (cyan at the moving head, fading to
          white going back) and a cyan head marker during traversals.
  right = phase-specific spectrogram panel -- a concentric ring of tiles
          (Part 1) or a 5x15 = 75-slot grid (Parts 2 & 3). All spectrograms are
          SAM2-masked and centred with equal padding on both sides.

Phases (each preceded by a full-figure white title card):
  * Part 1 -- Cluster peaks. For each cluster the right ring shows the peak
    (center tile) surrounded by its ``m`` nearest samples in up to three
    concentric rings; the active cluster is outlined in a thick pulsating cyan
    contour with a cyan dot at its centre.
  * Part 2 -- Peak-to-peak walks (up to 5). Shortest torus path + small smooth
    jitter between random cluster peaks; the right 5x15 grid fills row-major.
  * Part 3 -- Boundary crossings (5). Curved walks that wrap edges/corners; the
    right 5x15 grid uses columns = trajectory positions, rows = nearest
    neighbours.

All text renders in Helvetica via the shared ``apply_plot_style`` helper; every
``color=`` argument is a hex string.
"""

from __future__ import annotations

import pathlib
from collections.abc import Callable
from datetime import datetime
from functools import lru_cache

import click
import h5py
import matplotlib
import matplotlib.gridspec as gridspec
import numpy as np
from click.core import ParameterSource
from sklearn.neighbors import NearestNeighbors

from ..cli_utils import modify_settings_json_for_cli
from ..os_utils import configure_path
from .auxiliary_plot_functions import create_colormap
from .plot_style import apply_plot_style

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # Agg backend is selected above, before importing pyplot
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb

# Register the bundled Helvetica weights + activate the project mplstyle so all
# text in the video renders in Helvetica, instead of matplotlib's default font.
apply_plot_style()

# Hex palette (every matplotlib color= arg is a hex string). The accent
# (highlight) color is NOT here -- it is read per-render from the settings
# block so the user can tune it.
_C_WHITE = "#FFFFFF"
_C_BLACK = "#000000"
_C_GRAY = "#808080"
_C_LIGHTGRAY = "#D3D3D3"
_C_SUBTITLE = "#444444"
_SPEC_CMAP = "inferno"


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
    ``spectrogram/<key>/qlvm_dim`` datasets (written once by a one-off enrichment)
    into one array, with a parallel ``(session key, spectrogram row)`` list so a
    nearest-neighbour hit can be mapped straight back to a spectrogram. Rows with
    NaN coords are dropped.

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


# ---------------------------------------------------------------------------
# Geometry / trajectory helpers (pure, no matplotlib state)
# ---------------------------------------------------------------------------


def _lin(a, b, n):
    """Linear interpolation a -> b in ``n`` steps, returning an (n, len(a)) array."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return a + np.linspace(0.0, 1.0, n)[:, None] * (b - a)


def _smooth_jitter(rng, n, sigma):
    """Small smooth (n, 2) perturbation pinned at both endpoints."""
    if n <= 2 or sigma <= 0:
        return np.zeros((n, 2))
    win = max(3, n // 8)
    kernel = np.ones(win) / win
    raw = rng.normal(0.0, sigma, size=(n, 2))
    sm = np.stack([np.convolve(raw[:, d], kernel, mode='same') for d in range(2)], axis=1)
    t = np.linspace(0, 1, n)[:, None]
    sm = sm - (1 - t) * sm[0] - t * sm[-1]
    return sm


def _curved_path(a, b, n, amplitude):
    """Deterministic curved path from a to b: straight line + half-sine bend."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    t = np.linspace(0.0, 1.0, n)
    line = a[None] + t[:, None] * (b - a)[None]
    direction = b - a
    norm = float(np.linalg.norm(direction))
    if norm < 1e-12:
        return line
    perp = np.array([-direction[1], direction[0]]) / norm
    bend = amplitude * np.sin(np.pi * t)
    return line + bend[:, None] * perp[None, :]


def _shortest_extended(a, b):
    """Return (a, b') so that the straight line a->b' is the shortest torus path."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = (b - a + 0.5) % 1.0 - 0.5
    return a, a + d


# ---------------------------------------------------------------------------
# Matplotlib artist helpers (mutate / create axes & inset artists)
# ---------------------------------------------------------------------------


def _set_border(ax, color, width):
    """Set every spine of ``ax`` to the given color/width (used for tile borders)."""
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_edgecolor(color)
        sp.set_linewidth(width)


def _contour_visible(cs, vis):
    """Toggle visibility of a QuadContourSet across matplotlib versions."""
    try:
        cs.set_visible(vis)
    except AttributeError:
        pass
    if hasattr(cs, 'collections'):
        for coll in cs.collections:
            coll.set_visible(vis)


def _set_contour_style(cs, lw, alpha):
    """Set a QuadContourSet's linewidth + alpha across matplotlib versions."""
    try:
        cs.set_linewidth(lw)
        cs.set_alpha(alpha)
    except AttributeError:
        pass
    if hasattr(cs, 'collections'):
        for coll in cs.collections:
            coll.set_linewidth(lw)
            coll.set_alpha(alpha)


def _fading_trail_segments(traj_unit, wrap_thresh=0.5):
    """Consecutive-point segments + a recency array (0 oldest .. 1 newest).

    Wrap discontinuities (jumps > ``wrap_thresh`` on the torus) are dropped so the
    line never streaks across the panel. The recency array is fed to a colormap
    so the trail is brightest at the current position and fades going back.
    """
    pts = np.asarray(traj_unit)
    if len(pts) < 2:
        return np.empty((0, 2, 2)), np.empty((0,))
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    recency = np.linspace(0.0, 1.0, len(pts) - 1)
    jumps = np.abs(pts[1:] - pts[:-1]).max(axis=1)
    keep = jumps <= wrap_thresh
    return segs[keep], recency[keep]


# ---------------------------------------------------------------------------
# Phase-list construction (pure: builds the ordered animation script)
# ---------------------------------------------------------------------------


def build_phases(centers, K, peaks_only, title_card_frames, cluster_hold_frames,
                 peak_traverse_frames, boundary_traverse_frames,
                 peak_jitter_sigma, boundary_curve_amplitude, rng):
    """Build the ordered list of animation phases (the "script").

    Mixes ``title_card`` / ``cluster`` / ``traversal`` phase dicts: Part 1 title
    card + one cluster phase each; then (unless ``peaks_only``) Part 2 title card
    + up to 5 peak->peak traversals, and Part 3 title card + 5 boundary walks.
    ``rng`` is consumed in a fixed order (peak-pair choice, then per-pair jitter).
    """
    phases = []

    phases.append({
        'name': 'title_card', 'part': 1,
        'title': 'QLVM clusters walkthrough' if peaks_only else 'Part 1 - Cluster Peaks',
        'subtitle': 'Peak and nearest samples in three concentric rings',
        'duration': title_card_frames,
    })
    for ci in range(K):
        phases.append({'name': 'cluster', 'active_ci': ci, 'duration': cluster_hold_frames})

    if not peaks_only:
        phases.append({
            'name': 'title_card', 'part': 2,
            'title': 'Part 2 - Peak-to-Peak Walks',
            'subtitle': 'Shortest torus path between random cluster peaks',
            'duration': title_card_frames,
        })
        n_peak_pairs = min(5, K * (K - 1))
        pair_idx = rng.choice(K * (K - 1), size=n_peak_pairs, replace=False)
        all_pairs = []
        for p in pair_idx:
            i, j = divmod(int(p), K - 1)
            if j >= i:
                j += 1
            all_pairs.append((i, j))
        for (i, j) in all_pairs:
            a, b = _shortest_extended(centers[i], centers[j])
            traj = _lin(a, b, peak_traverse_frames) + \
                   _smooth_jitter(rng, peak_traverse_frames, peak_jitter_sigma)
            phases.append({
                'name': 'traversal', 'kind': 'peak',
                'sub': f'Peak {i + 1} -> Peak {j + 1}',
                'traj': traj, 'duration': peak_traverse_frames,
            })

        phases.append({
            'name': 'title_card', 'part': 3,
            'title': 'Part 3 - Boundary Crossings',
            'subtitle': 'Curved walks that wrap edges and corners of the torus',
            'duration': title_card_frames,
        })
        boundary_walks = [
            ((0.92, 0.30), (1.20, 0.70), 1.0, 'wrap right edge'),
            ((0.30, 0.92), (0.70, 1.20), 1.0, 'wrap top edge'),
            ((0.94, 0.94), (1.20, 1.20), 1.0, 'wrap corner'),
            ((0.08, 0.70), (-0.22, 0.30), -1.0, 'wrap left edge'),
            ((0.30, 0.30), (1.45, 1.35), 0.6, 'multi-wrap diagonal'),
        ]
        for start, end, amp_scale, lab in boundary_walks:
            traj = _curved_path(np.asarray(start), np.asarray(end),
                                boundary_traverse_frames, amp_scale * boundary_curve_amplitude)
            phases.append({
                'name': 'traversal', 'kind': 'boundary',
                'sub': lab, 'traj': traj, 'duration': boundary_traverse_frames,
            })

    return phases


def precompute_reveals(phases, nn_query, samples_per_trace,
                       boundary_positions_per_walk, boundary_neighbors):
    """Annotate each ``traversal`` phase in-place with its right-panel reveal plan.

    Boundary walks reveal ``boundary_positions_per_walk`` evenly-spaced positions,
    each with its ``boundary_neighbors`` nearest neighbours; peak walks reveal
    ``samples_per_trace`` positions with one NN each.
    """
    for ph in phases:
        if ph['name'] != 'traversal':
            continue
        n = ph['duration']
        if ph.get('kind') == 'boundary':
            reveal_frames = np.linspace(0, n - 1, boundary_positions_per_walk).astype(int)
            reveal_xy = ph['traj'][reveal_frames]
            reveal_nn = np.array([nn_query(xy % 1.0, k=boundary_neighbors) for xy in reveal_xy])
            ph['reveal_frames'] = reveal_frames
            ph['reveal_xy'] = reveal_xy
            ph['reveal_nn'] = reveal_nn
        else:
            reveal_frames = np.linspace(0, n - 1, samples_per_trace).astype(int)
            reveal_xy = ph['traj'][reveal_frames]
            reveal_idx = np.array([nn_query(xy % 1.0, k=1)[0] for xy in reveal_xy])
            ph['reveal_frames'] = reveal_frames
            ph['reveal_xy'] = reveal_xy
            ph['reveal_idx'] = reveal_idx


class QLVMTorusTraversalVideo:
    """
    Description
    -----------
    Builds the two-panel QLVM torus-traversal animation (cluster peaks +
    peak-to-peak / boundary walks) from the QLVM arrays + the consolidated
    spectrogram H5.
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
            Output ``.mp4`` / ``.gif`` path; if None, derived from
            ``figures.save_directory`` + a timestamp.
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
        Loads the QLVM arrays + pools per-USV latent coords / spectrograms from
        the consolidated H5, builds the phase script, and renders the two-panel
        animation to ``output_path`` (``.gif`` via Pillow, otherwise ``.mp4`` via
        FFmpeg).

        Parameters
        ----------

        Returns
        -------
        A video file at ``output_path``.
        """
        self.message_output(
            f"QLVM torus-traversal video started at: "
            f"{datetime.now().hour:02d}:{datetime.now().minute:02d}:{datetime.now().second:02d}."
        )

        cfg = self.input_parameter_dict['qlvm_torus_traversal_video']
        clustering = cfg['clustering']
        arrays_path = cfg['arrays_npz_path_fine'] if clustering == "fine" else cfg['arrays_npz_path_coarse']
        fps = cfg['fps']
        dpi = cfg['dpi']
        m = cfg['m']
        cluster_hold_frames = cfg['cluster_hold_frames']
        peak_traverse_frames = cfg['peak_traverse_frames']
        boundary_traverse_frames = cfg['boundary_traverse_frames']
        title_card_frames = cfg['title_card_frames']
        samples_per_trace = cfg['samples_per_trace']
        peak_jitter_sigma = cfg['peak_jitter_sigma']
        boundary_curve_amplitude = cfg['boundary_curve_amplitude']
        boundary_positions_per_walk = cfg['boundary_positions_per_walk']
        boundary_neighbors = cfg['boundary_neighbors']
        seed = cfg['seed']
        peaks_only = cfg['peaks_only']
        spec_cache_size = cfg['spec_cache_size']
        apply_mask = cfg['apply_mask']
        # User-tunable accent (highlight) color for the head marker, cluster
        # outline/centre dot, current-tile borders, and the trail colormap.
        accent_color = cfg['accent_color']
        accent_rgb255 = tuple(int(round(channel * 255)) for channel in to_rgb(accent_color))

        # Accent recency colormap for the trajectory trail (newest = accent
        # color, fading to white going back), via the create_colormap helper.
        trail_cmap = create_colormap(input_parameter_dict={
            'cm_length': 256, 'cm_name': 'accent_fade', 'cm_type': 'sequential',
            'cm_start': accent_rgb255, 'cm_end': (255, 255, 255),
            'equalize_luminance': False, 'match_luminance_by': 'max',
            'change_saturation': 1, 'cm_opacity': 1,
        })

        # Arrays supply ONLY the heatmap background, watershed contours, and
        # cluster centers (coarse = 7 / fine = 12).
        arrays = np.load(configure_path(arrays_path))
        heatmap = arrays['heatmap']
        ws_labels = arrays['ws_labels_periodic']
        centers = arrays['centers']
        res = heatmap.shape[0]
        K = len(centers)

        if self.output_path is None:
            stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
            out_path = pathlib.Path(
                configure_path(self.input_parameter_dict['figures']['save_directory'])
            ) / f"qlvm_torus_traversal_{stamp}.mp4"
        else:
            out_path = pathlib.Path(self.output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(configure_path(cfg['consolidated_h5_path']), "r") as h5:
            # Per-USV latent coords + (session, row) come from the H5's qlvm_dim
            # datasets; spectrograms are read from the same store on demand.
            pooled_coords, pooled_index = pool_latents_from_h5(h5)
            n_samples = pooled_coords.shape[0]
            if n_samples == 0:
                raise ValueError(
                    "No `qlvm_dim` coordinates found in the consolidated H5 — the "
                    "store must be enriched once with per-session qlvm_dim latent "
                    "coords before rendering."
                )
            self.message_output(f"Pooled {n_samples} latents, {K} clusters ({clustering}).")

            n_neighbors_max = min(max(m + 1, boundary_neighbors, 1), n_samples)
            nn_index = NearestNeighbors(n_neighbors=n_neighbors_max,
                                        algorithm="ball_tree").fit(torus_forward(pooled_coords))

            def nn_query(xy_unit, k):
                """Return the k nearest latent indices to a single [0,1]^2 point."""
                z = torus_forward(np.asarray(xy_unit)[None])
                _, idx = nn_index.kneighbors(z, n_neighbors=min(k, n_samples))
                return idx[0]

            # Per-session SAM2 spectrogram_index cache (read once per session when
            # masking, like the marimo explorer).
            mask_index_cache: dict = {}

            @lru_cache(maxsize=spec_cache_size)
            def _get_spec(data_idx):
                """Read (and LRU-cache) the spectrogram for a pooled index from the H5.

                Optionally applies the combined SAM2 mask (``apply_mask``), then
                centers the call in its fixed window with equal zero-padding on
                both sides (using the per-USV ``durations``) so short calls are not
                stretched -- the same treatment as the marimo embedding explorer.
                """
                session_key, row = pooled_index[int(data_idx)]
                grp = h5["spectrogram"][session_key]
                spec = np.asarray(grp["spectrograms"][row])
                spec_h, spec_w = spec.shape
                if "durations" in grp:
                    dur = int(grp["durations"][row])
                    dur = max(1, min(dur, spec_w))
                else:
                    dur = spec_w
                call = spec[:, :dur]

                if apply_mask and "mask" in h5 and session_key in h5["mask"]:
                    mask_grp = h5["mask"][session_key]
                    if session_key not in mask_index_cache:
                        mask_index_cache[session_key] = mask_grp["spectrogram_index"][:]
                    matching = np.where(mask_index_cache[session_key] == row)[0]
                    if matching.size > 0:
                        masks_for_spec = mask_grp["segmentations"][matching, :, :dur]
                        combined_mask = np.any(masks_for_spec, axis=0)
                        call = call * combined_mask.astype(call.dtype)

                canvas = np.zeros((spec_h, spec_w), dtype=spec.dtype)
                pad_left = (spec_w - dur) // 2
                canvas[:, pad_left:pad_left + dur] = call
                return canvas

            # m+1 neighbours per cluster: index 0 = peak (center tile), 1.. = ring.
            cluster_nn = [nn_query(c, k=m + 1) for c in centers]

            rng = np.random.default_rng(seed)
            phases = build_phases(
                centers, K, peaks_only, title_card_frames, cluster_hold_frames,
                peak_traverse_frames, boundary_traverse_frames,
                peak_jitter_sigma, boundary_curve_amplitude, rng,
            )
            precompute_reveals(
                phases, nn_query, samples_per_trace,
                boundary_positions_per_walk, boundary_neighbors,
            )

            frames_info = []
            for pi, ph in enumerate(phases):
                for fi in range(ph['duration']):
                    frames_info.append((pi, fi))

            # --- Figure layout ---
            sample_spec = _get_spec(0)
            spec_H, spec_W = sample_spec.shape

            grid_nrows = 5
            grid_ncols = 15
            n_grid_slots = grid_nrows * grid_ncols
            boundary_row_nn = [2, 1, 0, 3, 4]
            boundary_mid_row = boundary_row_nn.index(0)

            fig_w = 18.0
            fig_h = 8.6
            fig = plt.figure(figsize=(fig_w, fig_h))
            outer = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.2, 1.45], wspace=0.08)
            ax_l = fig.add_subplot(outer[0, 0])
            rg = gridspec.GridSpecFromSubplotSpec(
                grid_nrows, grid_ncols, subplot_spec=outer[0, 1], hspace=0.55, wspace=0.08,
            )
            ax_grid = [fig.add_subplot(rg[r, c])
                       for r in range(grid_nrows) for c in range(grid_ncols)]

            fig.canvas.draw()
            right_bbox = outer[0, 1].get_position(fig)
            rx0, ry0 = right_bbox.x0, right_bbox.y0
            rw, rh = right_bbox.width, right_bbox.height
            ccx = rx0 + rw / 2
            ccy = ry0 + rh / 2

            tile_in = 0.78
            tile_w = tile_in / fig_w
            tile_h = tile_in / fig_h

            n_inner = min(m, 6)
            n_middle = min(max(m - n_inner, 0), 12)
            n_outer = max(m - n_inner - n_middle, 0)
            n_outer = min(n_outer, 18)
            r_inner_in = 1.05
            r_middle_in = 2.05
            r_outer_in = 3.05

            ring_axes = []
            for n_pts, r_in in [(n_inner, r_inner_in), (n_middle, r_middle_in), (n_outer, r_outer_in)]:
                if n_pts <= 0:
                    continue
                for i in range(n_pts):
                    ang = 2 * np.pi * i / n_pts - np.pi / 2
                    dx = r_in * np.cos(ang) / fig_w
                    dy = r_in * np.sin(ang) / fig_h
                    ax = fig.add_axes([ccx + dx - tile_w / 2, ccy + dy - tile_h / 2, tile_w, tile_h])
                    ax.set_xticks([]); ax.set_yticks([])
                    ring_axes.append(ax)

            ax_center_ring = fig.add_axes([ccx - tile_w / 2, ccy - tile_h / 2, tile_w, tile_h])
            ax_center_ring.set_xticks([]); ax_center_ring.set_yticks([])
            ring_all_axes = [ax_center_ring] + ring_axes  # idx 0 = peak

            ax_card = fig.add_axes([0, 0, 1, 1], zorder=50)
            ax_card.set_facecolor(_C_WHITE)
            ax_card.set_xticks([]); ax_card.set_yticks([])
            for sp in ax_card.spines.values():
                sp.set_visible(False)
            card_title = ax_card.text(0.5, 0.58, '', ha='center', va='center',
                                      fontsize=36, fontweight='light', color=_C_BLACK)
            card_sub = ax_card.text(0.5, 0.44, '', ha='center', va='center',
                                    fontsize=20, fontweight='light', color=_C_SUBTITLE, style='italic')
            ax_card.set_visible(False)

            # --- Static left panel ---
            nz = heatmap[heatmap > 0]
            vmax = float(np.percentile(nz, 95)) if nz.size else None
            ax_l.imshow(heatmap, origin='lower', extent=(0, 1, 0, 1),
                        cmap=_SPEC_CMAP, vmin=0, vmax=vmax, aspect='equal')
            xx = np.linspace(0, 1, res)
            yy = np.linspace(0, 1, res)
            ax_l.contour(xx, yy, ws_labels, levels=np.arange(0.5, K + 1.5),
                         colors=_C_WHITE, linewidths=2.5)
            ax_l.set_xlim(0, 1); ax_l.set_ylim(0, 1); ax_l.set_aspect('equal')
            # No axis labels / ticks / ticklabels on the latent map -- just the map.
            ax_l.set_xticks([]); ax_l.set_yticks([])

            cluster_contours = {}
            active_contour_ci = {'value': None}

            def show_cluster_contour(ci):
                """Show cluster ``ci``'s cyan outline on the left panel, hiding any prior."""
                prev = active_contour_ci['value']
                if prev is not None and prev != ci and prev in cluster_contours:
                    _contour_visible(cluster_contours[prev], False)
                if ci not in cluster_contours:
                    cs = ax_l.contour(xx, yy, (ws_labels == ci + 1).astype(int),
                                      levels=[0.5], colors=accent_color, linewidths=5.0, alpha=0.95)
                    cluster_contours[ci] = cs
                _contour_visible(cluster_contours[ci], True)
                active_contour_ci['value'] = ci

            def hide_active_contour():
                """Hide the currently-active cluster contour (if any)."""
                prev = active_contour_ci['value']
                if prev is not None and prev in cluster_contours:
                    _contour_visible(cluster_contours[prev], False)
                active_contour_ci['value'] = None

            # Trajectory trail: a recency-colored line (cyan at the current
            # position, fading to white going back) instead of inset thumbnails.
            trail_lc = LineCollection([], cmap=trail_cmap, linewidths=4.0, zorder=6)
            trail_lc.set_clim(0.0, 1.0)
            ax_l.add_collection(trail_lc)
            head_marker = ax_l.scatter([], [], c=accent_color, s=70, marker='o',
                                        edgecolors=_C_BLACK, linewidths=1.0, zorder=11)
            # Cyan circle marking the active cluster's center (Part 1); pulses
            # with the cluster outline.
            cluster_center_marker = ax_l.scatter([], [], c=accent_color, s=120, marker='o',
                                                 edgecolors=_C_BLACK, linewidths=1.2, zorder=12)
            title = ax_l.set_title('', fontsize=13, loc='left', fontweight='light')
            sup = fig.suptitle('', fontsize=18, fontweight='light')

            BLANK = np.zeros((spec_H, spec_W))

            def _init_tile(ax, fontsize):
                """Initialise one tile axes: blank imshow + empty title; return (im, t)."""
                ax.set_xticks([]); ax.set_yticks([])
                _set_border(ax, _C_LIGHTGRAY, 0.5)
                im = ax.imshow(np.zeros((spec_H, spec_W)), cmap=_SPEC_CMAP,
                               origin='lower', aspect='auto', vmin=0, vmax=1)
                t = ax.set_title('', fontsize=fontsize, pad=1)
                return im, t

            grid_imgs, grid_titles = [], []
            for ax in ax_grid:
                im, t = _init_tile(ax, fontsize=8.5)
                grid_imgs.append(im); grid_titles.append(t)

            ring_imgs, ring_titles = [], []
            for ax in ring_all_axes:
                im, t = _init_tile(ax, fontsize=8.0)
                ring_imgs.append(im); ring_titles.append(t)

            def _draw(im, ax, t, data_idx, title_str, border, border_w):
                """Paint one tile: spectrogram (or blank) + per-tile clim + title + border."""
                if data_idx is None:
                    im.set_data(BLANK); im.set_clim(0, 1)
                else:
                    spec = _get_spec(int(data_idx))
                    im.set_data(spec)
                    mx = float(spec.max())
                    im.set_clim(0.0, max(mx, 1e-8))
                t.set_text(title_str)
                _set_border(ax, border, border_w)

            def render_grid_slot(slot, data_idx, title_str='', border=_C_LIGHTGRAY, border_w=0.5):
                """Draw spectrogram ``data_idx`` into right-grid tile ``slot`` (in place)."""
                _draw(grid_imgs[slot], ax_grid[slot], grid_titles[slot],
                      data_idx, title_str, border, border_w)

            def render_ring_slot(slot, data_idx, title_str='', border=_C_LIGHTGRAY, border_w=0.5):
                """Draw spectrogram ``data_idx`` into ring tile ``slot`` (in place)."""
                _draw(ring_imgs[slot], ring_all_axes[slot], ring_titles[slot],
                      data_idx, title_str, border, border_w)

            def blank_grid():
                """Blank every right-grid tile."""
                for s in range(n_grid_slots):
                    render_grid_slot(s, None, '', _C_LIGHTGRAY, 0.3)

            def set_grid_visible(vis):
                """Show/hide all right-grid tile axes."""
                for ax in ax_grid:
                    ax.set_visible(vis)

            def set_ring_visible(vis):
                """Show/hide all ring tile axes."""
                for ax in ring_all_axes:
                    ax.set_visible(vis)

            set_grid_visible(False)
            set_ring_visible(False)

            state = {'last_phase': -1, 'last_reveal_count': 0, 'reached_part': 0}

            def on_phase_enter(pi):
                """Reconfigure all shared artists for the first frame of phase ``pi``."""
                ph = phases[pi]
                if ax_card.get_visible() and ph['name'] != 'title_card':
                    ax_card.set_visible(False)
                hide_active_contour()
                trail_lc.set_segments([])
                head_marker.set_offsets(np.empty((0, 2)))
                cluster_center_marker.set_offsets(np.empty((0, 2)))

                if ph['name'] == 'title_card':
                    part = ph.get('part', 0)
                    state['reached_part'] = max(state['reached_part'], part)
                    set_grid_visible(False)
                    set_ring_visible(False)
                    ax_card.set_visible(True)
                    card_title.set_text(ph['title'])
                    card_sub.set_text(ph.get('subtitle', ''))
                    sup.set_text('')
                    title.set_text('')
                    return

                if ph['name'] == 'cluster':
                    set_grid_visible(False)
                    set_ring_visible(True)
                    ci = ph['active_ci']
                    show_cluster_contour(ci)
                    cluster_center_marker.set_offsets([[float(centers[ci][0]), float(centers[ci][1])]])
                    nn_idx = cluster_nn[ci]
                    render_ring_slot(0, int(nn_idx[0]), title_str='', border=accent_color, border_w=2.8)
                    for s in range(1, len(ring_all_axes)):
                        if s <= m and s < len(nn_idx):
                            render_ring_slot(s, int(nn_idx[s]), title_str='', border=_C_GRAY, border_w=0.5)
                        else:
                            render_ring_slot(s, None, '', _C_LIGHTGRAY, 0.3)
                    sup.set_text(f'Cluster {ci + 1} / {K}  -  peak + {m} nearest samples')
                    title.set_text('')

                elif ph['name'] == 'traversal':
                    set_ring_visible(False)
                    set_grid_visible(True)
                    blank_grid()
                    kind = ph.get('kind', '')
                    if kind == 'boundary':
                        sup.set_text(f"Boundary walk  -  {ph['sub']}")
                        title.set_text(f'Curved path  -  right columns = '
                                       f'{boundary_positions_per_walk} positions, '
                                       f'rows = {boundary_neighbors} nearest neighbors')
                    else:
                        sup.set_text(f"Peak-to-peak walk  -  {ph['sub']}")
                        title.set_text('Red trail = path on torus  -  '
                                       'right grid fills as samples are visited')

                state['last_reveal_count'] = 0

            def update(frame_idx):
                """Advance the animation to global frame ``frame_idx`` (mutates artists)."""
                pi, fi = frames_info[frame_idx]
                ph = phases[pi]
                if pi != state['last_phase']:
                    on_phase_enter(pi)
                    state['last_phase'] = pi

                if ph['name'] == 'title_card':
                    return []

                if ph['name'] == 'cluster':
                    # Pulsate the active cluster's (thick) cyan outline so it is
                    # clearly visible: width + opacity oscillate over ~24 frames.
                    pulse = 0.5 + 0.5 * np.sin(2 * np.pi * fi / 24.0)
                    cs = cluster_contours.get(ph['active_ci'])
                    if cs is not None:
                        _set_contour_style(cs, lw=4.0 + 5.0 * pulse, alpha=0.45 + 0.55 * pulse)
                    cluster_center_marker.set_sizes([120.0 + 160.0 * pulse])
                    return []

                traj = ph['traj']
                traj_unit = traj[:fi + 1] % 1.0
                segs, recency = _fading_trail_segments(traj_unit)
                trail_lc.set_segments(segs)
                trail_lc.set_array(recency)
                # The cyan ball rides the current head of the walk (where the
                # right-panel spectrogram is sampled); the trail is cyan next to
                # it and fades to white going back into the past.
                head_marker.set_offsets([[traj[fi, 0] % 1.0, traj[fi, 1] % 1.0]])

                reveal_frames = ph['reveal_frames']
                count = int(np.sum(reveal_frames <= fi))
                last_count = state['last_reveal_count']
                kind = ph.get('kind', '')

                if kind == 'boundary':
                    if count > last_count:
                        if last_count > 0:
                            prev_s = last_count - 1
                            if prev_s < grid_ncols:
                                _set_border(ax_grid[boundary_mid_row * grid_ncols + prev_s], _C_GRAY, 0.5)
                        for s in range(last_count, count):
                            if s >= boundary_positions_per_walk:
                                break
                            nn = ph['reveal_nn'][s]
                            is_current = (s == count - 1)
                            for row in range(grid_nrows):
                                slot = row * grid_ncols + s
                                if slot >= n_grid_slots:
                                    continue
                                nn_k = boundary_row_nn[row]
                                if nn_k >= len(nn):
                                    render_grid_slot(slot, None, '', border=_C_LIGHTGRAY, border_w=0.3)
                                    continue
                                didx = int(nn[nn_k])
                                is_mid = (row == boundary_mid_row)
                                border = accent_color if (is_current and is_mid) else _C_GRAY
                                border_w = 2.0 if (is_current and is_mid) else 0.5
                                render_grid_slot(slot, didx, '', border=border, border_w=border_w)
                        state['last_reveal_count'] = count
                    title.set_text(f"positions revealed: "
                                   f"{min(count, boundary_positions_per_walk)}/{boundary_positions_per_walk}")
                else:  # peak walk
                    if count > last_count:
                        if last_count > 0:
                            _set_border(ax_grid[last_count - 1], _C_GRAY, 0.5)
                        for s in range(last_count, count):
                            if s >= n_grid_slots:
                                break
                            didx = int(ph['reveal_idx'][s])
                            is_current = (s == count - 1)
                            render_grid_slot(s, didx, '', border=accent_color if is_current else _C_GRAY,
                                             border_w=2.2 if is_current else 0.5)
                        state['last_reveal_count'] = count
                    title.set_text(f"samples revealed: {count}/{samples_per_trace}")
                return []

            ani = FuncAnimation(fig, update, frames=len(frames_info), interval=1000 / fps, blit=False)

            total_frames = len(frames_info)
            self.message_output(f"Writing {total_frames} frames @ {fps} fps -> {out_path} ...")
            if out_path.suffix.lower() == ".gif":
                writer: PillowWriter | FFMpegWriter = PillowWriter(fps=fps)
            else:
                writer = FFMpegWriter(fps=fps, bitrate=4000)
            ani.save(str(out_path), writer=writer, dpi=dpi)
            plt.close(fig)

        self.message_output(f"Wrote {total_frames}-frame traversal video -> {out_path}.")
        self.message_output(
            f"QLVM torus-traversal video ended at: "
            f"{datetime.now().hour:02d}:{datetime.now().minute:02d}:{datetime.now().second:02d}."
        )


@click.command(name="qlvm-torus-traversal-video")
@click.option('--output-path', type=str, default=None, required=False, help='Output .mp4 / .gif path (default: figures.save_directory + timestamp).')
@click.option('--clustering', 'clustering', type=str, default=None, required=False, help='coarse / fine.')
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
