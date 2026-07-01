"""
@author: bartulem
Makes behavioral videos from 3D tracked points.
"""

import json
import os
import pathlib
import platform
import re
import subprocess
import warnings
from datetime import datetime
from typing import Any

import h5py
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import polars as pls
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from numba import njit
from scipy.io import wavfile

from ..analyses.decode_experiment_label import extract_information
from ..analyses.generate_audio_files import AudioGenerator
from ..os_utils import first_match_or_raise
from .plot_style import apply_plot_style
from ..time_utils import is_gui_context, smart_wait
from .auxiliary_plot_functions import create_colormap, choose_animal_colors
from .figure_io import save_figure

apply_plot_style()


# Behavioral-video figure geometry + palette: the single source of truth for the
# hand-tuned video composite. The arena/panel layout is optimised for this fixed
# figure size, so the size is hard-coded rather than down-scaled at runtime.
# _VIDEO_DPI is chosen so figsize * dpi stays within the 4096 px hardware-H.264 /
# PowerPoint decoder limit on the long (7.5") edge (7.5 * 540 = 4050 px), which
# removes the need for any runtime clamp. The figure-creation dpi AND every
# encoder/static-save dpi must read _VIDEO_DPI from here, or the rendered frame
# size desyncs from the figure and the video glitches in PowerPoint.
_VIDEO_FIGSIZE = (7.5, 4.8)
_VIDEO_DPI = 540
_VIDEO_COLOR_MODES = {
    "light_mode": {
        "background_color": "#FFFFFF",
        "node_edge_color": "#8B8B8B",
        "body_edge_color": "#8B8B8B",
        "text_color": "#202020",
        "tick_color": "#202020",
        "arena_line_color": "#202020",
        "arena_mic_color": "#202020",
        "arena_mesh_color": "#202020",
        "spectrogram_text_color": "#202020",
        "speaker_color": "#202020"
    },
    "dark_mode": {
        "background_color": "#202020",
        "node_edge_color": "#8B8B8B",
        "body_edge_color": "#8B8B8B",
        "text_color": "#FFFFFF",
        "tick_color": "#FFFFFF",
        "arena_line_color": "#FFFFFF",
        "arena_mic_color": "#FFFFFF",
        "arena_mesh_color": "#FFFFFF",
        "spectrogram_text_color": "#FFFFFF",
        "speaker_color": "#FFFFFF"
    }
}

# Per-case behavioral-video panel layout, hand-tuned per
# (view_angle, has-companion-panels) combination and kept here as one
# inspectable table. Each entry is the full set of [left, bottom, width, height]
# panel rectangles, 3-D view angles, axis limits, arena zoom and text offsets for
# that case; the rationale for individual values is preserved inline. `view_azimuth`
# is None for the side views (resolved at runtime from `side_azimuth_start`).
# `spec_fig_position` / `raster_fig_position` are None for layouts whose
# spectrogram / raster panel is absent — the value is never read there, since each
# panel is gated on its own `*_bool` downstream.
_VIDEO_BEH_FEATURES_POSITION = [0.09, 0.85, 0.158, 0.035]
_VIDEO_LAYOUT = {
    "top_panels": {
        "view_elevation": 90,
        "view_azimuth": 0,
        "plot_xlim": 0.5, "plot_ylim": 0.5, "plot_zlim": 0.5,
        # arena_zoom is the 3-D content scale factor (set_box_aspect zoom): 1.0 =
        # matplotlib default, larger = bigger arena. Decoupled from layout, subplot
        # count and bbox cropping; identical for static + video frames. THE
        # enlargement knob.
        "arena_zoom": 1.6,
        # arena_position is the [left, bottom, width, height] placement of the
        # arena axis, independent of arena_zoom (which sets size). Shifting `left`
        # re-centres the arena between the companion panels.
        "arena_position": [-0.02, 0.0, 1.0, 1.0],
        "text_start_coords": [-0.15, 0.98],
        # bottom is raised off the frame edge so the spectrogram's x-tick labels
        # and 'Time (s)' label sit ABOVE y=0: video frames grab the fixed [0,1]
        # canvas with no crop, so anything below y=0 is cut off.
        "spec_fig_position": [0.275, 0.03, 0.49, 0.125],
        # raster_fig_position: widen to de-squeeze the right-hand spike raster (it
        # grows leftward into the arena's empty right margin). For video, left +
        # width must stay <= ~0.96 so the right spine and '+0.5' tick label fit
        # inside the fixed [0,1] frame.
        "raster_fig_position": [0.734, 0.18, 0.221, 0.70],
        "mouse_id_text_offset": 0.05,
        # main_text_offset is the vertical gap between consecutive text lines
        # (title -> frame counter, and between animal-ID rows).
        "main_text_offset": 0.025,
    },
    "top_plain": {
        "view_elevation": 90,
        "view_azimuth": 0,
        "plot_xlim": 0.5, "plot_ylim": 0.5, "plot_zlim": 0.5,
        # No companion panels: a larger zoom fills more of the frame.
        "arena_zoom": 2.2,
        "arena_position": [0.0, 0.0, 1.0, 1.0],
        # y=0.96: the session-title text2D is baseline-aligned (glyphs extend
        # upward), and on the no-crop video [0,1] canvas it needs headroom — 0.99
        # clipped the title's top edge, 0.98 just cleared it, 0.96 leaves a small
        # top margin. Figures crop tight (bbox_inches='tight') so they're unaffected.
        "text_start_coords": [-0.25, 0.96],
        "spec_fig_position": None,
        "raster_fig_position": None,
        "mouse_id_text_offset": 0.05,
        "main_text_offset": 0.025,
    },
    "side_panels": {
        "view_elevation": 45,
        "view_azimuth": None,
        "plot_xlim": 0.6, "plot_ylim": 0.6, "plot_zlim": 0.4,
        # arena_zoom sets size; arena_position sets placement. The raster occupies
        # the right, so shift the arena left to re-centre it between the left-hand
        # features and the raster — which also frees room to grow it via arena_zoom.
        "arena_zoom": 1.2,
        "arena_position": [-0.05, 0.0, 1.0, 1.0],
        "text_start_coords": [-0.15, 0.98],
        "spec_fig_position": [0.265, 0.835, 0.5, 0.125],
        "raster_fig_position": [0.72, 0.18, 0.26, 0.70],
        "mouse_id_text_offset": 0.05,
        "main_text_offset": 0.025,
    },
    "side_plain": {
        "view_elevation": 45,
        "view_azimuth": None,
        # No companion panels: full-frame placement, offset is moot.
        "plot_xlim": 0.5, "plot_ylim": 0.5, "plot_zlim": 0.4,
        "arena_zoom": 1.3,
        "arena_position": [-0.05, 0.0, 1.0, 1.0],
        "text_start_coords": [-0.15, 0.98],
        "spec_fig_position": [0.265, 0.835, 0.5, 0.125],
        "raster_fig_position": None,
        "mouse_id_text_offset": 0.05,
        "main_text_offset": 0.025,
    },
}


@njit
def read_ttl_events(input_array: np.ndarray) -> tuple:
    """
    Description
    -----------
    Return TTL ON and OFF in the least significant bit array.

    Parameters
    ----------
    input_array (np.ndarray)
        A (n_samples) shape ndarray of audio data.

    Returns
    -------
     off_to_on, on_to_off (tuple)
        Samples when the TTL pulse starts and ends.
    """

    lsb_array = input_array & 1
    off_to_on = np.where(np.diff(lsb_array) < 0)[0] + 1
    on_to_off = np.where(np.diff(lsb_array) > 0)[0]
    return off_to_on[0], on_to_off[0]


@njit
def filter_spikes_for_raster(input_arr: np.ndarray,
                             ra_st_fr: int,
                             ra_end_fr: int,
                             fr_start: int) -> np.ndarray:
    """
    Description
    -----------
    Return spike times relative to current frame.

    Parameters
    ----------
    input_arr (np.ndarray)
        A (n_spikes) shape ndarray of spike train.
    ra_st_fr (int)
        Start frame of raster.
    ra_end_fr (int)
        End frame of raster.
    fr_start (int)
        Current frame 0 in raster.

    Returns
    -------
    input_arr (np.ndarray)
        Spike times relative to current frame.
    """

    return input_arr[(input_arr >= ra_st_fr) & (input_arr < ra_end_fr)] - fr_start


def pool_brain_area(brain_region: str | None) -> str:
    """
    Description
    -----------
    Map a raw CCF region acronym (as stored in the per-mouse
    `neuropixels_sites_to_anatomy_converter.json`) to one of the
    six display buckets used by the plotting palette:

        PAG, MRN, VTA, MB -> themselves
        CENT*             -> 'CENT'   (e.g. CENT2, CENT3)
        SC*               -> 'SC'     (e.g. SCdw, SCdg, SCop, SCsg, ...)
        everything else   -> 'other'  (incl. None / empty string)

    The bucket names match keys in
    `visualizations_settings.json["brain_area_colors"]`; the `'other'`
    bucket exists explicitly so unknown / unselected regions get a
    well-defined grey colour rather than tripping a KeyError.

    Parameters
    ----------
    brain_region (str | None)
        Raw region acronym from the anatomy converter. May be `None`
        when a cluster's channel falls outside any labelled range.

    Returns
    -------
    bucket (str)
        One of `{'PAG', 'MRN', 'VTA', 'MB', 'CENT', 'SC', 'other'}`.
    """

    if not brain_region:
        return 'other'
    if brain_region in ('PAG', 'MRN', 'VTA', 'MB'):
        return brain_region
    if brain_region.startswith('CENT'):
        return 'CENT'
    if brain_region.startswith('SC'):
        return 'SC'
    return 'other'


def _resolve_brain_area_color(brain_region: str | None,
                              brain_color_scheme: dict) -> str:
    """
    Description
    -----------
    Resolve the display colour for a raw brain-region acronym by
    pooling it to a bucket (via `pool_brain_area`) and looking the
    bucket up in `brain_color_scheme`. Falls back to the `'other'`
    entry of the scheme when the bucket isn't explicitly listed —
    callers therefore never need to pre-pool or guard against
    KeyError.

    Parameters
    ----------
    brain_region (str | None)
        Raw region acronym from the anatomy converter (or `None`).
    brain_color_scheme (dict)
        Bucket-keyed palette, typically the
        `visualizations_settings.json["brain_area_colors"]` block.
        Must contain at least an `'other'` entry.

    Returns
    -------
    color (str)
        Hex colour string for the resolved bucket.
    """

    bucket = pool_brain_area(brain_region)
    if bucket in brain_color_scheme:
        return brain_color_scheme[bucket]
    return brain_color_scheme['other']


def find_region_by_channel(cluster_id: str,
                           brain_area_dict: dict,
                           brain_color_scheme: dict,
                           return_only_color: bool = True,
                           return_only_area: bool = False) -> tuple[Any, Any] | Any:
    """
    Description
    -----------
    Returns name and color of particular brain region.

    `brain_color_scheme` is keyed by display bucket (PAG / MRN / VTA /
    MB / CENT / SC / other), not by raw region acronym; the colour for
    a raw region is resolved through `_resolve_brain_area_color` so
    `SCdw` / `SCdg` / `CENT2` / etc. all map to the canonical bucket
    colour. The raw acronym is still returned for area-only queries
    (so filter selectors that key on subdivisions keep working).

    Parameters
    ----------
    cluster_id (str)
        Cluster ID.
    brain_area_dict (dict)
        Contains brain area information.
    brain_color_scheme (dict)
        Bucket-keyed display palette (see
        `visualizations_settings.json["brain_area_colors"]`).
    return_only_color (bool)
        If True, returns only color.
    return_only_area (bool)
        If True, returns only area.

    Returns
    -------
    brain_region, brain_color, (brain_region, brain_color)  (str | tuple)
        Brain region (raw acronym) and/or color (bucket-resolved).
    """

    cluster_ch = int(cluster_id[cluster_id.index('_ch') + 3:cluster_id.index('_ch') + 6])
    for probe_id, probe_regions in brain_area_dict.items():
        for brain_region, channel_groups in probe_regions.items():
            for channel_group in channel_groups:
                if channel_group[0] <= cluster_ch < channel_group[1]:
                    if return_only_color:
                        return _resolve_brain_area_color(brain_region, brain_color_scheme)
                    if return_only_area:
                        return brain_region
                    return brain_region, _resolve_brain_area_color(brain_region, brain_color_scheme)

    # Channel falls outside every labelled anatomy range -> assign the 'other'
    # bucket (region 'other', bucket-resolved colour) instead of returning None,
    # so callers (eventplot colours / (region, colour) unpacking) receive a valid
    # value rather than crashing on None. Unlabelled clusters render in the
    # 'other' colour and still pool to the 'other' bucket for area filtering.
    if return_only_color:
        return _resolve_brain_area_color('other', brain_color_scheme)
    if return_only_area:
        return 'other'
    return 'other', _resolve_brain_area_color('other', brain_color_scheme)


def load_audio_data(root_directory: str) -> tuple[np.ndarray, int]:
    """
    Description
    -----------
    Returns audio data w/ sampling rate.
    NB: Audio is loaded from mmap file!

    Parameters
    ----------
    root_directory (str)
        Root directory.

    Returns
    -------
    audio_data, sampling_rate (tuple (np.ndarray, int))
       Audio data and audio sampling rate.
    """

    audio_loc = first_match_or_raise(
        root=pathlib.Path(root_directory),
        pattern='*_int16.mmap*',
        recursive=True,
        label="concatenated int16 audio memmap",
    )
    channel_num = int(audio_loc.name.split('_')[-2])
    sample_num = int(audio_loc.name.split('_')[-3])
    sampling_rate = int(audio_loc.name.split('_')[-4])

    audio_data = np.memmap(filename=audio_loc,
                           dtype=np.int16,
                           mode='r',
                           shape=(sample_num, channel_num),
                           order='C')

    return audio_data, sampling_rate


def plot_mouse_data(data: np.ndarray,
                    plot_axes: plt.Axes,
                    frame_number: int,
                    animal_node_names: list[str],
                    animal_color: list[str],
                    animal_cm: list[plt.cm],
                    animal_line_width: int,
                    node_connections: list[str],
                    node_polygons: list[str],
                    node_lw: int|float,
                    node_size: int|float,
                    node_opacity: int|float,
                    node_edge_color: str,
                    polygon_color: list[str],
                    polygon_opacity: int|float,
                    body_edge_color: str,
                    history_frame_span: int,
                    history_point: str,
                    history_ls: str,
                    history_lw: int|float,
                    xlim_: int|float,
                    ylim_: int|float,
                    zlim_: int|float,
                    node_bool: bool = False,
                    history_bool: bool = False) -> None:

    """
    Description
    -----------
    Plots mouse 3D data.

    Parameters
    ----------
    Contains the following set of parameters
        data (np.ndarray)
            A (n_frames, n_mice, n_nodes, n_dim) shape ndarray of 3D mouse data.
        plot_axes (plt.Axes)
            Axes object for plotting.
        frame_number (int)
            Frame to be plotted.
        animal_node_names (list)
            Mouse node names.
        animal_color (list)
            List of mouse colors.
        animal_cm (list)
            List of mouse colormaps.
        animal_line_width (int / float)
            Line width of lines connecting mouse nodes.
        node_bool (bool)
            If true, plots body nodes.
        node_connections (list)
            Mouse node connections.
        node_polygons (list)
            Mouse node polygon vertices.
        node_lw (int / float)
            Line width of mouse nodes.
        node_size (int / float)
            Size of mouse nodes.
        node_opacity (int / float)
            Opacity of mouse nodes.
        node_edge_color (str)
            Mouse node edge color.
        polygon_color (list)
            List of face colors of mouse polygons.
        polygon_opacity (int/ float)
            Opacity of mouse polygons.
        body_edge_color (str)
            Color of mouse body outer edges
        history_bool (bool)
            Boolean for plotting mouse movement history.
        history_frame_span (int)
            Number of frames to plot for mouse movement history.
        history_point (str)
            Node to use for plotting mouse movement history.
        history_ls (str)
            Line style of mouse movement history.
        history_lw (int / float)
            Line width of mouse movement history.
        xlim_ (int / float)
            X-axis limit.
        ylim_ (int / float)
            Y-axis limit.
        zlim_ (int / float)
            Z-axis limit.

    Returns
    -------
    None
    """

    # Resolve every node name to its index once: this function runs once per animation
    # frame and the mapping never changes, so the repeated O(n) animal_node_names.index(...)
    # scans in the loops below are pure repeated work. setdefault in enumerate order stores
    # each name's first occurrence, exactly matching list.index.
    node_idx = {}
    for _i_node, _node_name in enumerate(animal_node_names):
        node_idx.setdefault(_node_name, _i_node)

    for mouse_idx in range(data.shape[1]):
        if history_bool:
            # plot history of animal paths
            for hist_point_idx, hist_point in enumerate(range(frame_number - history_frame_span, frame_number)):
                plot_axes.plot([data[hist_point, mouse_idx, node_idx[history_point], 0], data[hist_point + 1, mouse_idx, node_idx[history_point], 0]],
                               [data[hist_point, mouse_idx, node_idx[history_point], 1], data[hist_point + 1, mouse_idx, node_idx[history_point], 1]],
                               [data[hist_point, mouse_idx, node_idx[history_point], 2], data[hist_point + 1, mouse_idx, node_idx[history_point], 2]],
                               color=animal_cm[mouse_idx](int(255*hist_point_idx/history_frame_span)), ls=history_ls, lw=history_lw)

        # plot node connection lines
        for nc_idx, nc in enumerate(node_connections):
            line_color = body_edge_color if nc_idx <= 7 else animal_color[mouse_idx]
            nc = nc.split('-')
            plot_axes.plot([data[frame_number, mouse_idx, node_idx[nc[0]], 0], data[frame_number, mouse_idx, node_idx[nc[1]], 0]],
                           [data[frame_number, mouse_idx, node_idx[nc[0]], 1], data[frame_number, mouse_idx, node_idx[nc[1]], 1]],
                           [data[frame_number, mouse_idx, node_idx[nc[0]], 2], data[frame_number, mouse_idx, node_idx[nc[1]], 2]],
                           c=line_color, lw=animal_line_width)

        # plot node polygon shading
        for npol in node_polygons:
            npol = npol.split('-')
            xs, ys, zs = np.zeros(len(npol)), np.zeros(len(npol)), np.zeros(len(npol))
            for i in range(len(npol)):
                xs[i] = data[frame_number, mouse_idx, node_idx[npol[i]], 0]
                ys[i] = data[frame_number, mouse_idx, node_idx[npol[i]], 1]
                zs[i] = data[frame_number, mouse_idx, node_idx[npol[i]], 2]

            vertices = [list(zip(xs, ys, zs, strict=True))]
            plot_axes.add_collection3d(Poly3DCollection(verts=vertices, facecolors=[polygon_color[mouse_idx]], alpha=polygon_opacity))

        if node_bool:
            plot_axes.scatter(data[frame_number, mouse_idx, :, 0], data[frame_number, mouse_idx, :, 1], data[frame_number, mouse_idx, :, 2],
                              c=animal_color[mouse_idx], edgecolor=node_edge_color, linewidth=node_lw, s=node_size, alpha=node_opacity)

    plot_axes.grid(False)
    plot_axes.set_axis_off()
    plot_axes.set_xlim3d(-xlim_, xlim_)
    plot_axes.set_ylim3d(-ylim_, ylim_)
    plot_axes.set_zlim3d(0, zlim_)


def plot_speaker_data(speaker_data: np.ndarray,
                      plot_axes: plt.Axes,
                      frame_number: int,
                      speaker_color: str,
                      speaker_alpha: int|float) -> None:
    """
    Description
    -----------
    Plots speaker object as a sphere.

    Parameters
    ----------
    Contains the following set of parameters
        speaker_data (np.ndarray)
            A (n_frames, 1, 1, n_dim) shape ndarray of the 3D speaker point.
        plot_axes (ax)
            Axes object for plotting.
        frame_number (int)
            Frame to be plotted.
        speaker_color (str)
            Speaker color.
        speaker_alpha (int / float)
            Speaker opacity.

    Returns
    -------
    None
    """

    plot_axes.scatter(speaker_data[frame_number, 0, 0, 0],
                      speaker_data[frame_number, 0, 0, 1],
                      speaker_data[frame_number, 0, 0, 2],
                      c=speaker_color, s=10, alpha=speaker_alpha)


def plot_spectrogram(plot_axes: plt.Axes,
                     figure_object: plt.Figure,
                     spec_start: int,
                     spec_end: int,
                     audio_sr: int,
                     stft_hop: int,
                     half_window_size_sec: int|float,
                     color_mode_preferences: dict,
                     spectrogram_amplitude: np.ndarray,
                     power_limit: list[int|float],
                     freq_limit: list[int|float],
                     freq_yticks: list[int|float],
                     usv_segments_list: list[tuple],
                     usv_segment_lw: int|float,
                     usv_segment_colors_list: list[str],
                     usv_segments_ypos: int|float,
                     cbar_bool: bool = False,
                     plot_usv_segments_bool: bool = False) -> None:
    """
    Description
    -----------
    Plots a spectrogram.

    Parameters
    ----------
    Contains the following set of parameters
        plot_axes (ax)
            Axes object for plotting.
        figure_object (fig)
            Figure object for plotting.
        spec_start (int)
            Start index of spectrogram.
        spec_end (int)
            End index of spectrogram.
        audio_sr (int)
            Audio sampling rate.
        stft_hop (int)
            STFT hop size.
        half_window_size_sec (int / float)
            Half window size in seconds.
        cbar_bool (bool)
            Boolean for plotting color bar.
        color_mode_preferences (dict)
            Contains color mode preferences.
        spectrogram_amplitude (np.ndarray)
            Input data w/ spectrogram amplitude.
        power_limit (list)
            Power limits [vmin, vmax] for spectrogram.
        freq_limit (list)
            Frequency y-axis limits [fmin, fmax] for spectrogram.
        freq_yticks (list)
            Frequency y-axis ticks for spectrogram.
        plot_usv_segments_bool (bool)
            Boolean for plotting assignment of USV segments.
        usv_segments_list (list)
            Contains information about start/stop of each USV segment.
        usv_segment_lw (int / float)
            Line width of USV segments.
        usv_segment_colors_list (list)
            USVs segments colored by mouse identity.
        usv_segments_ypos (int / float)
            Y-position of USV segments.

    Returns
    -------
    None
    """

    img = librosa.display.specshow(data=spectrogram_amplitude[:, spec_start:spec_end],
                                   sr=audio_sr,
                                   hop_length=stft_hop,
                                   y_axis='linear',
                                   x_axis='time',
                                   cmap='inferno',
                                   vmin=power_limit[0],
                                   vmax=power_limit[1],
                                   ax=plot_axes)
    plot_axes.axvline(x=half_window_size_sec,
                      color='#FFFFFF',
                      linewidth=.25)
    if plot_usv_segments_bool:
        for idx_usv_, usv_ in enumerate(usv_segments_list):
            plot_axes.axhline(y=usv_segments_ypos,
                              xmin=((max(-half_window_size_sec, usv_[0]) - (-half_window_size_sec)) / (half_window_size_sec - (-half_window_size_sec))),
                              xmax=((min(half_window_size_sec, usv_[1]) - (-half_window_size_sec)) / (half_window_size_sec - (-half_window_size_sec))),
                              color=usv_segment_colors_list[idx_usv_],
                              lw=usv_segment_lw)
    # x-axis: keep only the two bounding labels (the window is centred, so they
    # read -half .. +half — e.g. -0.5 / +0.5 for a 1 s window), drop the tick
    # marks, and add an enlarged 'Time (s)' label.
    spec_xlim = plot_axes.get_xlim()
    plot_axes.set_xticks([spec_xlim[0], spec_xlim[1]])
    plot_axes.set_xticklabels([f"{-half_window_size_sec:g}", f"{half_window_size_sec:g}"])
    plot_axes.set_xlim(spec_xlim)
    plot_axes.xaxis.set_tick_params(which='minor', bottom=False)
    plot_axes.set_xlabel(xlabel='Time (s)',
                         labelpad=-6,
                         fontsize=6,
                         color=color_mode_preferences['spectrogram_text_color'])
    # y-axis: only the 'Freq (kHz)' label, pulled in close to the axis; no
    # frequency tick marks or numeric labels.
    plot_axes.set_ylim(freq_limit)
    plot_axes.set_yticks([])
    plot_axes.yaxis.set_tick_params(which='minor', left=False)
    plot_axes.spines['bottom'].set_color(color_mode_preferences['tick_color'])
    plot_axes.spines['top'].set_color(color_mode_preferences['tick_color'])
    plot_axes.spines['left'].set_color(color_mode_preferences['tick_color'])
    plot_axes.spines['right'].set_color(color_mode_preferences['tick_color'])
    plot_axes.tick_params(axis='x',
                          which='major',
                          length=0,
                          colors=color_mode_preferences['tick_color'])
    plot_axes.set_ylabel(ylabel='Freq (kHz)',
                         labelpad=3,
                         fontsize=6,
                         color=color_mode_preferences['spectrogram_text_color'])

    if cbar_bool:
        cbar = figure_object.colorbar(mappable=img,
                                      ax=plot_axes,
                                      pad=.01)
        # strip every colorbar tick and tick label; keep only the label, pulled
        # in close to the bar.
        cbar.set_ticks([])
        cbar.set_label(label=r'Amplitude (dB)',
                       labelpad=3,
                       size='xx-small',
                       color=color_mode_preferences['spectrogram_text_color'])
        cbar.outline.set_edgecolor(color_mode_preferences['tick_color'])


def plot_raster(plot_axes: plt.Axes,
                figure_object: plt.Figure,
                unit_num: int,
                raster_data: list[np.ndarray],
                raster_half_window: int,
                raster_half_window_sec: int|float,
                raster_brain_area: dict,
                raster_line_lengths: list[int|float],
                raster_line_widths: list[int|float],
                filtered_brain_areas: list[str],
                color_mode_preferences: dict,
                event_plot_colors: list[str],
                brain_area_color_scheme: dict) -> None:
    """
    Description
    -----------
    Makes a raster plot.

    Parameters
    ----------
    Contains the following set of parameters
        plot_axes (ax)
            Axes object for plotting.
        figure_object (fig)
            Figure object for plotting.
        unit_num (int)
            Number of units.
        raster_data (list of np.ndarray)
            Input data for plotting a spike raster.
        raster_half_window (int)
            Half window size in frames.
        raster_half_window_sec (int / float)
            Half window size in seconds.
        raster_brain_area (dict)
            Brain areas recorded from by each probe.
        raster_line_lengths (list)
            Line lengths for raster events.
        raster_line_widths (list)
            Line widths for raster events.
        filtered_brain_areas (list)
            Brain areas to be denoted on plot.
        color_mode_preferences (dict)
            Contains color mode preferences.
        event_plot_colors (list)
            Colors for plotting raster events.
        brain_area_color_scheme (dict)
            Brain area color scheme.

    Returns
    -------
    None
    """

    plot_axes.spines['bottom'].set_color(color_mode_preferences['tick_color'])
    plot_axes.spines['top'].set_visible(False)
    plot_axes.spines['left'].set_visible(False)
    plot_axes.spines['right'].set_visible(False)
    plot_axes.axvline(x=0,
                      color='#FFFFFF',
                      linewidth=.05)
    plot_axes.eventplot(positions=raster_data,
                        orientation='horizontal',
                        lineoffsets=range(len(raster_data)),
                        linelengths=raster_line_lengths,
                        linewidths=raster_line_widths,
                        colors=event_plot_colors,
                        linestyles='solid')
    plot_axes.tick_params(axis='both',
                          colors=color_mode_preferences['tick_color'],
                          pad=1.5,
                          length=1,
                          labelsize=6,
                          labelcolor=color_mode_preferences['text_color'])
    plot_axes.set_xlim(-raster_half_window, raster_half_window)
    plot_axes.set_xticks([-raster_half_window, 0, raster_half_window])
    # keep all three ticks but drop the central "0" label; the two bounding
    # labels are enlarged via the tick_params labelsize above.
    plot_axes.set_xticklabels([f"{-raster_half_window_sec:g}", '', f"{raster_half_window_sec:g}"])
    plot_axes.set_xlabel(xlabel='Time (s)',
                         fontsize=6,
                         fontweight='bold',
                         color=color_mode_preferences['text_color'])
    plot_axes.xaxis.set_label_coords(0.5, -0.025)
    plot_axes.set_ylim(-.5, unit_num + .5)
    plot_axes.set_yticks([])
    fig_renderer = figure_object.canvas.get_renderer()
    txt_x_start = 0
    # Fixed left-to-right display order for the area labels; this matches the
    # top-to-bottom unit ordering imposed on the raster. A representative raw
    # region is stored per bucket so its bucket colour can be resolved.
    brain_area_label_order = ['CENT', 'SC', 'PAG', 'MRN', 'MB', 'VTA', 'other']
    bucket_to_region = {}
    for probe_id in raster_brain_area:
        for brain_region in raster_brain_area[probe_id]:
            bucket = pool_brain_area(brain_region)
            area_selected = (
                len(filtered_brain_areas) == 0
                or brain_region in filtered_brain_areas
                or bucket in filtered_brain_areas
            )
            if area_selected and bucket not in bucket_to_region:
                bucket_to_region[bucket] = brain_region
    for bucket in brain_area_label_order:
        if bucket not in bucket_to_region:
            continue
        txt = plot_axes.text(x=txt_x_start,
                             y=1.01,
                             s=bucket,
                             fontsize=6,
                             fontweight='bold',
                             color=_resolve_brain_area_color(bucket_to_region[bucket], brain_area_color_scheme),
                             transform=plot_axes.transAxes)
        txt_x_start = txt.get_window_extent(renderer=fig_renderer).transformed(plot_axes.transAxes.inverted()).x1 + .01


def plot_behavioral_features(plot_axes: plt.Axes,
                             figure_object: plt.Figure,
                             mouse_track_names: list[str],
                             special_features: list[str],
                             beh_features_to_plot: list,
                             beh_feature_data: pls.DataFrame,
                             beh_features_fig_position: list[float],
                             beh_window_size_sec: int|float,
                             beh_window_size_frames: int,
                             beh_half_window_size_frames: int,
                             beh_features_ylabels: dict,
                             feature_ts_fr_start: int,
                             feature_ts_fr_end: int,
                             feature_ts_fr_middle: int,
                             x_axis_start: int,
                             x_axis_middle: int,
                             x_axis_end: int,
                             ylim_dict: dict,
                             plot_theme: str,
                             color_mode_preferences: dict,
                             animal_colors: list[str],
                             remove_axes_bool: bool = False) -> None:
    """
    Description
    -----------
    Plots behavioral feature dynamics.

    Parameters
    ----------
    Contains the following set of parameters
        plot_axes (ax)
            Full axes object for plotting.
        figure_object (fig)
            Figure object for plotting.
        mouse_track_names (list)
            List of mouse track names.
        special_features (list)
            Features that will be accentuated, and others made transparent.
        beh_features_to_plot (list)
            List of behavioral features to plot.
        beh_feature_data (DataFrame)
            Input data for plotting behavioral features.
        beh_features_fig_position (list)
            List of figure positions for behavioral features.
        beh_window_size_frames (int)
            Window size in frames.
        beh_window_size_sec (int / float)
            Half window size in seconds (per-side +/- x-axis bound; equals beh_features_window_size / 2).
        beh_half_window_size_frames (int)
            Half window size in frames.
        beh_features_ylabels (dict)
            Dictionary containing y-axis labels for behavioral features.
        feature_ts_fr_start (int)
            Start index of feature time series.
        feature_ts_fr_end (int)
            End index of feature time series.
        feature_ts_fr_middle (int)
            Middle index of feature time series.
        x_axis_start (int)
            Start index of x-axis.
        x_axis_middle (int)
            Middle index of x-axis.
        x_axis_end (int)
            End index of x-axis.
        ylim_dict (dict)
            Dictionary containing y-axis limits for behavioral features.
        plot_theme (str)
            If 'dark', dark theme.
        color_mode_preferences (dict)
            Dictionary containing color mode preferences.
        animal_colors (list)
            List of animal colors.
        remove_axes_bool (bool)
            Boolean for removing axes.

    Returns
    -------
    None
    """

    for feature_idx, feature_name in enumerate(beh_features_to_plot):
        ax_num = 3 + feature_idx
        feature_col_idx = beh_feature_data.columns.index(feature_name)
        if remove_axes_bool:
            plot_axes[ax_num].remove()

        if len(special_features) == 0:
            if feature_name.split('.')[0] == mouse_track_names[0]:
                feature_color = animal_colors[0]
                x_axis_feature_color = animal_colors[0]
            elif feature_name.split('.')[0] == mouse_track_names[1]:
                feature_color = animal_colors[1]
                x_axis_feature_color = animal_colors[1]
            else:
                if '-sei' not in feature_name.split('.')[1]:
                    if plot_theme == 'dark':
                        feature_color = '#FFFFFF'
                        x_axis_feature_color = '#FFFFFF'
                    else:
                        feature_color = '#202020'
                        x_axis_feature_color = '#202020'
                else:
                    if feature_name.split('.')[0].split('-')[0] == mouse_track_names[0]:
                        feature_color = animal_colors[0]
                        x_axis_feature_color = animal_colors[0]
                    else:
                        feature_color = animal_colors[1]
                        x_axis_feature_color = animal_colors[1]
        else:
            if feature_name.split('.')[0] == mouse_track_names[0]:
                if feature_name.split('.')[1] in special_features:
                    feature_color = animal_colors[0]
                    x_axis_feature_color = animal_colors[0]
                else:
                    feature_color = f"{animal_colors[0]}33"
                    x_axis_feature_color = animal_colors[0]
            elif feature_name.split('.')[0] == mouse_track_names[1]:
                if feature_name.split('.')[1] in special_features:
                    feature_color = animal_colors[1]
                    x_axis_feature_color = animal_colors[1]
                else:
                    feature_color = f"{animal_colors[1]}33"
                    x_axis_feature_color = animal_colors[1]
            else:
                if plot_theme == 'dark':
                    if feature_name.split('.')[1] in special_features:
                        feature_color = '#FFFFFF'
                        x_axis_feature_color = '#FFFFFF'
                    else:
                        feature_color = '#FFFFFF33'
                        x_axis_feature_color = '#FFFFFF'
                else:
                    if feature_name.split('.')[1] in special_features:
                        feature_color = '#202020'
                        x_axis_feature_color = '#202020'
                    else:
                        feature_color = '#00000033'
                        x_axis_feature_color = '#202020'

        plot_axes[ax_num] = figure_object.add_axes([beh_features_fig_position[0],
                                                    beh_features_fig_position[1] - (feature_idx * 0.042),
                                                    beh_features_fig_position[2],
                                                    beh_features_fig_position[3]])
        plot_axes[ax_num].set_facecolor(color_mode_preferences['background_color'])
        plot_axes[ax_num].plot(range(beh_window_size_frames),
                               beh_feature_data[feature_ts_fr_start:feature_ts_fr_end,
                               feature_col_idx],
                               ls='-',
                               lw=.5,
                               color=feature_color)
        plot_axes[ax_num].plot(beh_half_window_size_frames,
                               beh_feature_data[feature_ts_fr_middle, feature_col_idx],
                               marker='o',
                               markersize=2,
                               color=feature_color,
                               clip_on=False)
        plot_axes[ax_num].tick_params(axis='x',
                                      which='both',
                                      length=1,
                                      color=x_axis_feature_color,
                                      pad=1.5,
                                      labelsize=6,
                                      labelcolor=x_axis_feature_color)
        plot_axes[ax_num].tick_params(axis='y',
                                      which='both',
                                      length=1,
                                      color=feature_color,
                                      pad=1.5,
                                      labelsize=2,
                                      labelcolor=feature_color)
        if feature_idx == len(beh_features_to_plot) - 1:
            plot_axes[ax_num].set_xticks(ticks=[x_axis_start, x_axis_middle, x_axis_end])
            # keep all three ticks but drop the central "0" label
            plot_axes[ax_num].set_xticklabels(labels=[-beh_window_size_sec, '', beh_window_size_sec])
        else:
            plot_axes[ax_num].set_xticks([])
        plot_axes[ax_num].set_ylim(ylim_dict[feature_name.split('.')[1]][0],
                                   ylim_dict[feature_name.split('.')[1]][1])
        # y-axis keeps only its (enlarged, Helvetica-Light) label — drop the
        # numeric tick labels and ticks to free room next to the spectrogram
        plot_axes[ax_num].set_yticks([])
        plot_axes[ax_num].spines['top'].set_visible(False)
        plot_axes[ax_num].spines['right'].set_visible(False)
        if feature_idx == len(beh_features_to_plot) - 1:
            plot_axes[ax_num].spines['bottom'].set_position(('axes', -.35))
            plot_axes[ax_num].spines['bottom'].set_color(x_axis_feature_color)
            plot_axes[ax_num].spines['bottom'].set_bounds(x_axis_start, x_axis_end)
            plot_axes[ax_num].set_xlabel(xlabel='Time (s)',
                                         fontsize=6,
                                         fontweight='bold',
                                         color=x_axis_feature_color,
                                         alpha=1.)
            plot_axes[ax_num].xaxis.set_label_coords(.5, -.55)
        else:
            plot_axes[ax_num].spines['bottom'].set_visible(False)
        plot_axes[ax_num].spines['left'].set_color(feature_color)

        ylabel_for_feature = beh_features_ylabels[feature_name.split('.')[1]]
        plot_axes[ax_num].set_ylabel(ylabel=ylabel_for_feature[:ylabel_for_feature.index('(')] + '\n' + ylabel_for_feature[ylabel_for_feature.index('('):],
                                     fontsize=5,
                                     fontweight='light',
                                     rotation=0,
                                     color=feature_color)
        plot_axes[ax_num].yaxis.set_label_coords(-.12, .25)


def plot_arena_corners_mics(data: np.ndarray,
                            plot_axes: plt.Axes,
                            frame_number: int,
                            session_id: str,
                            esr: int|float,
                            animal_id: dict,
                            animal_colors: list[str],
                            color_mode_preferences: dict,
                            arena_node_connections: list[str],
                            arena_node_names: list[str],
                            arena_axes_lw: int|float,
                            arena_mics_lw: int|float,
                            arena_mics_opacity: int|float,
                            corner_size: int|float,
                            corner_opacity: int|float,
                            mesh_color: str,
                            mesh_opacity: int|float,
                            active_mic_position: int,
                            active_mic_color: str,
                            inactive_mic_color: str,
                            text_start_coords: list[int|float],
                            main_text_offset: int|float,
                            mouse_id_text_offset: int|float,
                            text_fontsize: int,
                            arena_node_connections_bool: bool = False,
                            plot_corners_bool: bool = False,
                            plot_mesh_walls_bool: bool = False,
                            active_mic_bool: bool = False,
                            inactive_mic_bool: bool = False) -> None:
    """
    Description
    -----------
    This function plots arena coordinates w/ microphones.

    Parameters
    ----------
    Contains the following set of parameters
        data (np.ndarray)
            Input data w/ 3D arena and mic points.
        plot_axes (ax)
            Axes object for plotting.
        frame_number (int)
            Frame to be plotted.
        session_id (str)
            Session ID.
        esr (int / float)
            Empirical camera frame rate.
        animal_id (dict)
            Contains animal ID (name: sex symbol) information.
        animal_colors (list)
            List of animal colors.
        color_mode_preferences (dict)
            Contains color mode preferences.
        arena_node_connections_bool (bool)
            If True, plots arena node connections.
        arena_node_connections (list)
            List of arena node connections.
        arena_node_names (list)
            List of arena node names.
        arena_axes_lw (int / float)
            Line width of arena axes.
        arena_mics_lw (int / float)
            Line width of arena-mic depictions.
        arena_mics_opacity (int / float)
            Opacity of arena-mic depictions.
        plot_corners_bool (bool)
            If True, plots corner points.
        corner_size (int / float)
            Size of corner points.
        corner_opacity (int / float)
            Opacity of corner points.
        plot_mesh_walls_bool (bool)
           If True, plots mesh walls around playpen.
        mesh_color (str)
            Color of mesh walls.
        mesh_opacity (int / float)
            Opacity of mesh walls.
        active_mic_bool (bool)
            If True, plots active mic.
        active_mic_position (int)
            Channel of active mic (required; the caller supplies 0 when no spectrogram channel is selected).
        active_mic_color (str)
            Color of active mic.
        inactive_mic_bool (bool)
            If True, plots inactive mics.
        inactive_mic_color (str)
            Color of inactive mics.
        text_start_coords (list)
            Starting coordinates for text.
        main_text_offset (int / float)
            Offset for main text.
        mouse_id_text_offset (int / float)
            Offset for mouse ID text.
        text_fontsize (int)
            Font size for text.


    Returns
    -------
    None
    """

    # Resolve every node name to its index once: this function runs once per animation
    # frame and the arena geometry is frame-invariant, so the dozens of O(n)
    # arena_node_names.index(...) scans below are pure repeated work. setdefault in
    # enumerate order stores each name's first occurrence, exactly matching list.index.
    node_idx = {}
    for _i_node, _node_name in enumerate(arena_node_names):
        node_idx.setdefault(_node_name, _i_node)

    if active_mic_bool:
        plot_axes.scatter(data[0, 0, 4 + active_mic_position, 0],
                          data[0, 0, 4 + active_mic_position, 1],
                          data[0, 0, 4 + active_mic_position, 2],
                          c=active_mic_color, s=20, alpha=arena_mics_opacity)

        if inactive_mic_bool:
            inactive_mic_list = list(range(4, 28))
            inactive_mic_list.remove(4 + active_mic_position)
            plot_axes.scatter(data[0, 0, inactive_mic_list, 0],
                              data[0, 0, inactive_mic_list, 1],
                              data[0, 0, inactive_mic_list, 2],
                              c=inactive_mic_color, s=20, alpha=arena_mics_opacity)

    if plot_corners_bool:
        plot_axes.scatter(data[0, 0, node_idx['North'], 0], data[0, 0, node_idx['North'], 1], data[0, 0, node_idx['North'], 2], c='#FF0000', s=corner_size, alpha=corner_opacity)
        plot_axes.scatter(data[0, 0, node_idx['West'], 0], data[0, 0, node_idx['West'], 1], data[0, 0, node_idx['West'], 2], c='#FFFF00', s=corner_size, alpha=corner_opacity)
        plot_axes.scatter(data[0, 0, node_idx['South'], 0], data[0, 0, node_idx['South'], 1], data[0, 0, node_idx['South'], 2], c='#008000', s=corner_size, alpha=corner_opacity)
        plot_axes.scatter(data[0, 0, node_idx['East'], 0], data[0, 0, node_idx['East'], 1], data[0, 0, node_idx['East'], 2], c='#0000FF', s=corner_size, alpha=corner_opacity)

    # plot bottom sides
    plot_axes.plot([data[0, 0, node_idx['North'], 0], data[0, 0, node_idx['West'], 0]],
                   [data[0, 0, node_idx['North'], 1], data[0, 0, node_idx['West'], 1]],
                   [data[0, 0, node_idx['North'], 2], data[0, 0, node_idx['West'], 2]],
                   c=color_mode_preferences['arena_line_color'], lw=arena_axes_lw)

    plot_axes.plot([data[0, 0, node_idx['North'], 0], data[0, 0, node_idx['East'], 0]],
                   [data[0, 0, node_idx['North'], 1], data[0, 0, node_idx['East'], 1]],
                   [data[0, 0, node_idx['North'], 2], data[0, 0, node_idx['East'], 2]],
                   c=color_mode_preferences['arena_line_color'], lw=arena_axes_lw)

    plot_axes.plot([data[0, 0, node_idx['South'], 0], data[0, 0, node_idx['West'], 0]],
                   [data[0, 0, node_idx['South'], 1], data[0, 0, node_idx['West'], 1]],
                   [data[0, 0, node_idx['South'], 2], data[0, 0, node_idx['West'], 2]],
                   c=color_mode_preferences['arena_line_color'], lw=arena_axes_lw)

    plot_axes.plot([data[0, 0, node_idx['South'], 0], data[0, 0, node_idx['East'], 0]],
                   [data[0, 0, node_idx['South'], 1], data[0, 0, node_idx['East'], 1]],
                   [data[0, 0, node_idx['South'], 2], data[0, 0, node_idx['East'], 2]],
                   c=color_mode_preferences['arena_line_color'], lw=arena_axes_lw)

    # plot vertical sides
    plot_axes.plot([data[0, 0, node_idx['North'], 0], data[0, 0, node_idx['North'], 0]],
                   [data[0, 0, node_idx['North'], 1], data[0, 0, node_idx['North'], 1]],
                   [data[0, 0, node_idx['North'], 2], .25],
                   c=color_mode_preferences['arena_line_color'], lw=arena_axes_lw)

    plot_axes.plot([data[0, 0, node_idx['West'], 0], data[0, 0, node_idx['West'], 0]],
                   [data[0, 0, node_idx['West'], 1], data[0, 0, node_idx['West'], 1]],
                   [data[0, 0, node_idx['West'], 2], .25],
                   c=color_mode_preferences['arena_line_color'], lw=arena_axes_lw)

    plot_axes.plot([data[0, 0, node_idx['South'], 0], data[0, 0, node_idx['South'], 0]],
                   [data[0, 0, node_idx['South'], 1], data[0, 0, node_idx['South'], 1]],
                   [data[0, 0, node_idx['South'], 2], .25],
                   c=color_mode_preferences['arena_line_color'], lw=arena_axes_lw)

    plot_axes.plot([data[0, 0, node_idx['East'], 0], data[0, 0, node_idx['East'], 0]],
                   [data[0, 0, node_idx['East'], 1], data[0, 0, node_idx['East'], 1]],
                   [data[0, 0, node_idx['East'], 2], .25],
                   c=color_mode_preferences['arena_line_color'], lw=arena_axes_lw)

    # plot mesh walls
    if plot_mesh_walls_bool:
        for wall in ['North-West', 'North-East', 'South-West', 'South-East']:
            wall = wall.split('-')
            mesh_wall = [[data[0, 0, node_idx[wall[0]], :],
                         np.array([data[0, 0, node_idx[wall[0]], 0], data[0, 0, node_idx[wall[0]], 1], 0.25]),
                         np.array([data[0, 0, node_idx[wall[1]], 0], data[0, 0, node_idx[wall[1]], 1], 0.25]),
                         data[0, 0, node_idx[wall[1]], :]]]
            plot_axes.add_collection3d(Poly3DCollection(verts=mesh_wall, facecolors=mesh_color, alpha=mesh_opacity))

    if arena_node_connections_bool:
        for arena_nc in arena_node_connections:
            arena_nc = arena_nc.split('-')
            plot_axes.plot([data[0, 0, node_idx[arena_nc[0]], 0], data[0, 0, node_idx[arena_nc[1]], 0]],
                           [data[0, 0, node_idx[arena_nc[0]], 1], data[0, 0, node_idx[arena_nc[1]], 1]],
                           [0, data[0, 0, node_idx[arena_nc[1]], 2]],
                           c=color_mode_preferences['arena_line_color'], lw=arena_mics_lw, alpha=arena_mics_opacity)

    plot_axes.text2D(x=text_start_coords[0],
                     y=text_start_coords[1],
                     s=f"{session_id}",
                     fontsize=text_fontsize,
                     color=color_mode_preferences['text_color'],
                     transform=plot_axes.transAxes)
    plot_axes.text2D(x=text_start_coords[0],
                     y=text_start_coords[1] - main_text_offset,
                     s=f"fr {frame_number:06d} | "
                       f"{frame_number / esr:07.2f}s",
                     fontsize=text_fontsize-2,
                     color=color_mode_preferences['text_color'],
                     transform=plot_axes.transAxes)
    for animal_idx, animal_key in enumerate(animal_id.keys()):
        # An explicit family *list* (not the generic 'sans-serif' alias) triggers
        # matplotlib's silent per-glyph fallback: the ID renders in Helvetica-Light
        # and the ♂ / ♀ sign, which Helvetica lacks, is supplied by DejaVu Sans.
        # apply_plot_style() de-shadows the glyph-poor system DejaVuSans-ExtraLight,
        # so this fallback lands on a DejaVu that carries the signs even at the
        # project's light font.weight (otherwise they would render as tofu boxes).
        plot_axes.text2D(x=text_start_coords[0],
                         y=text_start_coords[1] - mouse_id_text_offset - (animal_idx * main_text_offset),
                         s=f"{animal_key} {animal_id[animal_key]}",
                         fontsize=text_fontsize,
                         color=animal_colors[animal_idx],
                         fontfamily=['Helvetica', 'DejaVu Sans'],
                         transform=plot_axes.transAxes)


def create_spike_sound_file(audio_duration: int|float,
                            spike_array: np.ndarray,
                            sound_save_directory: str,
                            sound_session_id: str,
                            sound_frame_start: int,
                            sound_frame_span: int,
                            tracking_esr: int|float,
                            unit_id: str) -> None:
    """
    Description
    -----------
    Creates a WAV file with spiking sounds.

    Parameters
    ----------
    audio_duration (int / float)
        Duration of audio file.
    spike_array (np.ndarray)
        Spike events relative to segment start.
    sound_save_directory (str)
        Directory to save spike sound file.
    sound_session_id (str)
        Session ID.
    sound_frame_start (int)
        Tracking frame start.
    sound_frame_span (int)
        Tracking frame span.
    tracking_esr (int)
        Tracking acquisition rate.
    unit_id (str)
        Unit ID.

    Returns
    -------
    spike_sound (.wav)
        File containing relevant spike sounds.
    """

    spike_sound_sr, spike_sound = wavfile.read(pathlib.Path(__file__).parent.parent / '_config/spike.wav')

    new_spike_sound_array = np.zeros(shape=int(np.floor(spike_sound_sr * audio_duration)))

    for event in spike_array:
        sound_start = int(np.floor((event / tracking_esr) * spike_sound_sr))
        # Clip the write to what fits before the array end -- a spike landing
        # within spike_sound.shape[0] samples of the end would otherwise make the
        # left-hand slice shorter than spike_sound and raise a broadcast error.
        end = min(sound_start + spike_sound.shape[0], new_spike_sound_array.shape[0])
        # ADD (mix) rather than overwrite, so spikes closer together than the spike
        # sound's duration sum into an audible buzz instead of the later spike
        # clobbering the earlier one (which silently dropped spikes in dense bursts).
        new_spike_sound_array[sound_start:end] += spike_sound[:end - sound_start]
    # Clip BEFORE the int16 cast: summed overlaps can exceed the int16 range, and a
    # bare astype(int16) would wrap around to garbage rather than saturating.
    new_spike_sound_array = np.clip(new_spike_sound_array, -32768, 32767).astype(np.int16)

    sound_filename = pathlib.Path(sound_save_directory) / f"{sound_session_id}_3D_{sound_frame_start}-{sound_frame_start + sound_frame_span}fr_spike_sound_{unit_id}.wav"
    wavfile.write(filename=sound_filename,
                  rate=spike_sound_sr,
                  data=new_spike_sound_array)


class Create3DVideo:

    if os.name == 'nt':
        command_addition = 'cmd /c '
        shell_usage_bool = False
    else:
        command_addition = ''
        shell_usage_bool = True

    def __init__(self, **kwargs):

        """
        Description
        -----------
        Initializes the Create3DVideo class.

        Parameters
        ----------
        exp_id (str)
            Experiment ID (needed for figure naming).
        root_directory (str)
            Root directory containing mouse tracking data.
        arena_directory (str)
            Root directory containing arena tracking data.
        speaker_audio_file (str)
            File path to speaker tracking data.
        visualizations_parameter_dict (dict)
            Visualization params; defaults to None.
        message_output (function)
            Defines output messages; defaults to None.

        Returns
        -------
        None
        """

        expected_kwargs = {'exp_id', 'root_directory', 'arena_directory', 'speaker_audio_file',
                           'visualizations_parameter_dict', 'message_output'}
        unexpected_kwargs = set(kwargs) - expected_kwargs
        if unexpected_kwargs:
            raise TypeError(f"{type(self).__name__}() got unexpected keyword argument(s) "
                            f"{', '.join(map(repr, sorted(unexpected_kwargs)))}; expected only "
                            f"{', '.join(map(repr, sorted(expected_kwargs)))}.")
        for kw_arg, kw_val in kwargs.items():
            self.__dict__[kw_arg] = kw_val

        self.app_context_bool = is_gui_context()

        self.brain_area_color_scheme = dict(self.visualizations_parameter_dict['brain_area_colors'])

        self.node_connections = ['TTI-Haunch_left', 'TTI-Haunch_right', 'Shoulder_left-Haunch_left',
                                 'Shoulder_right-Haunch_right', 'Nose-Ear_L', 'Nose-Ear_R',
                                 'Head-Shoulder_left', 'Head-Shoulder_right',
                                 'TailTip-Tail_2', 'Tail_2-Tail_1', 'Tail_1-Tail_0', 'Tail_0-TTI',
                                 'TTI-Trunk', 'Trunk-Haunch_left', 'Trunk-Haunch_right',
                                 'Trunk-Neck', 'Shoulder_left-Neck', 'Shoulder_right-Neck',
                                 'Nose-Head', 'Ear_L-Head', 'Ear_R-Head', 'Head-Neck']

        self.node_polygons = ['Nose-Ear_L-Head', 'Nose-Ear_R-Head', 'TTI-Haunch_left-Trunk', 'TTI-Haunch_right-Trunk',
                              'Shoulder_left-Neck-Head', 'Shoulder_right-Neck-Head',
                              'Haunch_left-Trunk-Neck-Shoulder_left', 'Haunch_right-Trunk-Neck-Shoulder_right']

        self.arena_node_connections = ['North-ch_9', 'North-ch_10', 'North-ch_11', 'North-ch_12', 'North-ch_13', 'North-ch_14',
                                       'East-ch_3', 'East-ch_4', 'East-ch_5', 'East-ch_6', 'East-ch_7', 'East-ch_8',
                                       'South-ch_0', 'South-ch_1', 'South-ch_2', 'South-ch_21', 'South-ch_22', 'South-ch_23',
                                       'West-ch_15', 'West-ch_16', 'West-ch_17', 'West-ch_18', 'West-ch_19', 'West-ch_20']

        self.beh_features_ylabels = {"spaceX": "Disp(cm)", "spaceY": "Disp(cm)", "spaceZ": "Disp(cm)", "speed": "Speed(cm/s)", "acceleration": "Acc(cm/s²)",
                                     "neck_elevation": "Elev(cm)", "neck_elevation_1st_der": "Elev'(cm/s)", "neck_elevation_2nd_der": "Elev''(cm/s)",
                                     "body_dir": "BodyDir(°)", "body_dir_1st_der": "BodyDir'(°/s)", "body_dir_2nd_der": "BodyDir''(°/s²)",
                                     "ego_yaw": "EgoYaw(°)", "ego_yaw_1st_der": "EgoYaw'(°/s)", "ego_yaw_2nd_der": "EgoYaw''(°/s²)",
                                     "allo_roll": "Roll(°)", "allo_roll_1st_der": "Roll'(°/s)", "allo_roll_2nd_der": "Roll''(°/s²)",
                                     "allo_pitch": "Pitch(°)", "allo_pitch_1st_der": "Pitch'(°/s)", "allo_pitch_2nd_der": "Pitch''(°/s²)",
                                     "allo_yaw": "Yaw(°)", "allo_yaw_1st_der": "Yaw'(°/s)", "allo_yaw_2nd_der": "Yaw''(°/s²)",
                                     "back_pitch": "BPitch(°)", "back_pitch_1st_der": "BPitch'(°/s)", "back_pitch_2nd_der": "BPitch''(°/s²)",
                                     "back_yaw": "BYaw(°)", "back_yaw_1st_der": "BYaw'(°/s)", "back_yaw_2nd_der": "BYaw''(°/s²)",
                                     "tail_curvature": "Tail(a.u.)", "tail_curvature_1st_der": "Tail'(a.u.)", "tail_curvature_2nd_der": "Tail''(a.u.)",

                                     "nose-nose": "ΔN(cm)", "nose-nose_1st_der": "ΔN'(cm/s)", "nose-nose_2nd_der": "ΔN''(cm/s²)",
                                     "TTI-TTI": "ΔT(cm)", "TTI-TTI_1st_der": "ΔT'(cm/s)", "TTI-TTI_2nd_der": "ΔT''(cm/s²)",
                                     "nose-TTI": "ΔNT(cm)", "nose-TTI_1st_der": "ΔNT'(cm/s)", "nose-TTI_2nd_der": "ΔNT''(cm/s²)",
                                     "TTI-nose": "ΔTN(cm)", "TTI-nose_1st_der": "ΔTN'(cm/s)", "TTI-nose_2nd_der": "ΔTN''(cm/s²)",
                                     "allo_yaw-nose": "Yaw-N(°)", "allo_yaw-nose_1st_der": "Yaw-N'(°/s)", "allo_yaw-nose_2nd_der": "Yaw-N''(°/s²)",
                                     "nose-allo_yaw": "N-Yaw(°)", "nose-allo_yaw_1st_der": "N-Yaw'(°/s)", "nose-allo_yaw_2nd_der": "N-Yaw''(°/s²)",
                                     "allo_yaw-TTI": "Yaw-T(°)", "allo_yaw-TTI_1st_der": "Yaw-T'(°/s)", "allo_yaw-TTI_2nd_der": "Yaw-T''(°/s²)",
                                     "TTI-allo_yaw": "T-Yaw(°)", "TTI-allo_yaw_1st_der": "T-Yaw'(°/s)", "TTI-allo_yaw_2nd_der": "T-Yaw''(°/s²)",
                                     "allo_pitch-nose": "Pitch-N(°)", "allo_pitch-nose_1st_der": "Pitch-N'(°/s)", "allo_pitch-nose_2nd_der": "Pitch-N''(°/s²)",
                                     "nose-allo_pitch": "N-Pitch(°)", "nose-allo_pitch_1st_der": "N-Pitch'(°/s)", "nose-allo_pitch_2nd_der": "N-Pitch''(°/s²)",
                                     "allo_pitch-TTI": "Pitch-T(°)", "allo_pitch-TTI_1st_der": "Pitch-T'(°/s)", "allo_pitch-TTI_2nd_der": "Pitch-T''(°/s²)",
                                     "TTI-allo_pitch": "T-Pitch(°)", "TTI-allo_pitch_1st_der": "T-Pitch'(°/s)", "TTI-allo_pitch_2nd_der": "T-Pitch''(°/s²)",
                                     "orofacial-sei": "SEI(a.u.)", "orofacial-sei_1st_der": "SEI'(a.u./s)", "orofacial-sei_2nd_der": "SEI''(a.u./s²)",
                                     "anogenital-sei": "SEI(a.u.)", "anogenital-sei_1st_der": "SEI'(a.u./s)", "anogenital-sei_2nd_der": "SEI''(a.u./s²)"}

        self.color_mode_preferences = _VIDEO_COLOR_MODES


    def load_beh_features_file(self) -> pls.DataFrame:
        """
        Description
        -----------
        Loads the CSV file containing 3D behavioral features.

        Parameters
        ----------

        Returns
        -------
        beh_feature_data (DataFrame)
            Table (N_frames X N_features) containing 3D behavioral features.
        """

        # load behavioral feature data
        behavioral_data_file = first_match_or_raise(
            root=pathlib.Path(self.root_directory),
            pattern='*_behavioral_features.csv*',
            recursive=True,
            label="behavioral features CSV",
        )
        beh_feature_data = pls.read_csv(str(behavioral_data_file))

        return beh_feature_data

    def load_h5_file(self) -> tuple:
        """
        Description
        -----------
        Loads the HDF5 file containing 3D tracked points.

        Parameters
        ----------

        Returns
        -------
        arena, mouse (np.ndarray)
            Numpy arrays containing 3D tracked point data for arena and animals.
        """

        h5_file_arena = first_match_or_raise(
            root=pathlib.Path(self.arena_directory) / 'video',
            pattern='*_points3d_translated_rotated_metric.h5',
            recursive=True,
            label="arena points3d .h5",
        )
        h5_file_mouse = first_match_or_raise(
            root=pathlib.Path(self.root_directory) / 'video',
            pattern='[!speaker]*_points3d_translated_rotated_metric.h5',
            recursive=True,
            label="translated/rotated mouse points3d .h5",
        )

        # load HDF5 file
        with h5py.File(name=h5_file_arena, mode='r') as h5_file_arena_obj:
            arena_tracks = np.array(h5_file_arena_obj['tracks'])
            arena_node_names = [item.decode('utf-8') for item in h5_file_arena_obj['node_names']]
        with h5py.File(name=h5_file_mouse, mode='r') as h5_file_mouse_obj:
            mouse_tracks = np.array(h5_file_mouse_obj['tracks'])
            mouse_track_names = [item.decode('utf-8') for item in h5_file_mouse_obj['track_names']]
            mouse_node_names = [item.decode('utf-8') for item in h5_file_mouse_obj['node_names']]
            mouse_experimental_code = h5_file_mouse_obj['experimental_code'][()].decode('utf-8')
            empirical_camera_sr = float(h5_file_mouse_obj['recording_frame_rate'][()])

        if self.visualizations_parameter_dict['make_behavioral_videos']['speaker_bool']:
            h5_file_speaker = first_match_or_raise(
                root=pathlib.Path(self.root_directory) / 'video',
                pattern='speaker*_points3d_translated_rotated_metric.h5',
                recursive=True,
                label="speaker points3d .h5",
            )
            with h5py.File(name=h5_file_speaker, mode='r') as h5_file_speaker_obj:
                speaker_tracks = np.array(h5_file_speaker_obj['tracks'])
                speaker_track_name = h5_file_speaker_obj['track_names'][0].decode('utf-8')
                speaker_node_name = h5_file_speaker_obj['node_names'][0].decode('utf-8')

            return (h5_file_arena.parent, arena_tracks, arena_node_names,
                    h5_file_mouse.parent, mouse_tracks, mouse_track_names, mouse_node_names,
                    mouse_experimental_code, empirical_camera_sr,
                    speaker_tracks, speaker_track_name, speaker_node_name)

        else:
            return (h5_file_arena.parent, arena_tracks, arena_node_names,
                    h5_file_mouse.parent, mouse_tracks, mouse_track_names, mouse_node_names,
                    mouse_experimental_code, empirical_camera_sr)

    def visualize_in_video(self) -> None:
        """
        Description
        -----------
        Plots/animates 3D tracked mice.

        Parameters
        ----------

        Returns
        -------
        plot, figure, video (.svg | .mp4)
            Visualization of mouse (social) behavior in 3D.
        """

        self.message_output(f"Creating data visualization started at: {datetime.now().hour:02d}:{datetime.now().minute:02d}:{datetime.now().second:02d}")
        smart_wait(app_context_bool=self.app_context_bool, seconds=1)

        if self.visualizations_parameter_dict['make_behavioral_videos']['speaker_bool']:
            (arena_dir_name,
             arena_data,
             arena_node_names,
             mouse_dir_name,
             mouse_data,
             mouse_track_names,
             mouse_node_names,
             mouse_experimental_code,
             empirical_camera_sr,
             speaker_data,
             speaker_track_name,
             speaker_node_name) = self.load_h5_file()
        else:
            (arena_dir_name,
             arena_data,
             arena_node_names,
             mouse_dir_name,
             mouse_data,
             mouse_track_names,
             mouse_node_names,
             mouse_experimental_code,
             empirical_camera_sr) = self.load_h5_file()

        # Create the save directory only AFTER load_h5_file() has validated that
        # the session actually has data. Creating it at method entry left an
        # empty 'data_animation_examples' directory behind on any data-less run
        # (e.g. a smoke test pointed at an empty root); git silently ignored it
        # because the directory had no files, so the debris went unnoticed. The
        # directory is not referenced until well below (first use ~line 1913),
        # so deferring its creation past the validation gate is free.
        putative_save_directory = pathlib.Path(self.root_directory) / 'data_animation_examples'
        putative_save_directory.mkdir(exist_ok=True, parents=True)

        experiment_info_dict = extract_information(experiment_code=mouse_experimental_code)
        # Plain Unicode (not mathtext "$\u2642$"): Helvetica lacks the \u2642 / \u2640
        # signs, so the symbols resolve through the silent Helvetica -> DejaVu
        # Sans text-fallback chain. Wrapping them in mathtext instead would
        # route them through the custom math fontset and emit the noisy
        # "Font family ['cursive'] not found" findfont message.
        animal_id_sex_dict = {mouse_name: "\u2642" if mouse_sex == 'male' else "\u2640" for mouse_name, mouse_sex in zip(mouse_track_names, experiment_info_dict['mouse_sex'], strict=True)}

        animal_colors = choose_animal_colors(exp_info_dict=experiment_info_dict, visualizations_parameter_dict=self.visualizations_parameter_dict)
        animal_colors_dict = {mouse_name: animal_colors[mouse_idx] for mouse_idx, mouse_name in enumerate(mouse_track_names)}

        # The GUI / JSON default for "raster_special_units" is [""] (a lone
        # empty-string sentinel), not an actual unit. Because every downstream
        # guard tests `len(...) == 0`, that sentinel is truthy enough to flip
        # the raster onto the all-grey "special units" code path, which paints
        # every cluster (and the brain-area legend) with the 'other' colour.
        # Strip empty entries here so the per-brain-area colouring path runs
        # whenever no genuine special unit was requested.
        self.visualizations_parameter_dict['make_behavioral_videos']['raster_special_units'] = [
            one_special_unit
            for one_special_unit in self.visualizations_parameter_dict['make_behavioral_videos']['raster_special_units']
            if one_special_unit
        ]

        animal_colormaps = []
        for mouse_idx, mouse in enumerate(mouse_track_names):
            if self.visualizations_parameter_dict['make_behavioral_videos']['plot_theme'] == 'dark':
                cm_end = (0, 0, 0)
            else:
                cm_end = (255, 255, 255)
            animal_colormaps.append(create_colormap(input_parameter_dict={'cm_length': 255,
                                                                          'cm_name': f'{mouse}',
                                                                          'cm_type': 'sequential',
                                                                          'cm_start': (int(animal_colors[mouse_idx][1:3], 16),
                                                                                       int(animal_colors[mouse_idx][3:5], 16),
                                                                                       int(animal_colors[mouse_idx][5:7], 16)),
                                                                          'cm_end': cm_end,
                                                                          'equalize_luminance': True,
                                                                          'match_luminance_by': 'max',
                                                                          'change_saturation': .5,
                                                                          'cm_opacity': 1}))

        session_id = mouse_dir_name.parent.parent.name

        frame_start = int(np.floor(self.visualizations_parameter_dict['make_behavioral_videos']['video_start_time'] * empirical_camera_sr))
        frame_span = int(np.floor(self.visualizations_parameter_dict['make_behavioral_videos']['video_duration'] * empirical_camera_sr))

        active_mic_position = 0
        beh_features_fig_position = _VIDEO_BEH_FEATURES_POSITION
        # Select the hand-tuned layout for this (view, companion-panel) case from
        # the _VIDEO_LAYOUT table above. Any non-'top' view_angle is treated as
        # 'side' (mirrors the original if/else). view_azimuth for the side views is
        # resolved here from the live `side_azimuth_start` setting.
        _has_companion_panels = (self.visualizations_parameter_dict['make_behavioral_videos']['spectrogram_bool'] or
                                 self.visualizations_parameter_dict['make_behavioral_videos']['beh_features_bool'] or
                                 self.visualizations_parameter_dict['make_behavioral_videos']['raster_plot_bool'])
        _view_key = 'top' if self.visualizations_parameter_dict['make_behavioral_videos']['view_angle'] == 'top' else 'side'
        _layout = _VIDEO_LAYOUT[f"{_view_key}_{'panels' if _has_companion_panels else 'plain'}"]
        view_elevation = _layout['view_elevation']
        view_azimuth = (self.visualizations_parameter_dict['make_behavioral_videos']['side_azimuth_start']
                        if _view_key == 'side' else _layout['view_azimuth'])
        plot_xlim = _layout['plot_xlim']
        plot_ylim = _layout['plot_ylim']
        plot_zlim = _layout['plot_zlim']
        arena_zoom = _layout['arena_zoom']
        arena_position = _layout['arena_position']
        text_start_coords = _layout['text_start_coords']
        spec_fig_position = _layout['spec_fig_position']
        raster_fig_position = _layout['raster_fig_position']
        mouse_id_text_offset = _layout['mouse_id_text_offset']
        main_text_offset = _layout['main_text_offset']


        if (self.visualizations_parameter_dict['make_behavioral_videos']['spectrogram_bool'] or
                self.visualizations_parameter_dict['make_behavioral_videos']['beh_features_bool'] or
                self.visualizations_parameter_dict['make_behavioral_videos']['raster_plot_bool']):

            if self.visualizations_parameter_dict['make_behavioral_videos']['raster_plot_bool']:

                data_start_index = re.search(f"Data{os.sep}{pathlib.Path(self.root_directory).name}", self.root_directory, re.IGNORECASE).start()
                ephys_directory = self.root_directory[:data_start_index] + 'EPHYS'

                with open(
                    first_match_or_raise(
                        root=pathlib.Path(ephys_directory),
                        pattern='neuropixels_sites_to_anatomy_converter.json',
                        label="neuropixels-to-anatomy converter JSON",
                    ),
                    'r',
                ) as anatomy_converter_json:
                    neuropixels_sites_to_anatomy_converter = json.load(anatomy_converter_json)

                # find cluster data files and sort them in ascending order (0 channel first)
                cluster_files = sorted((pathlib.Path(self.root_directory) / 'ephys').rglob('cluster_data/*.npy'), key=lambda p: p.name)

                # filter cluster data files
                if (len(self.visualizations_parameter_dict['make_behavioral_videos']['raster_selection_criteria']['other']) > 0 or
                        len(self.visualizations_parameter_dict['make_behavioral_videos']['raster_selection_criteria']["brain_areas"]) > 0):
                    cluster_files_filtered = []
                    for one_file in cluster_files:
                        one_cluster_file_name = one_file.stem
                        select_other_bool = True
                        select_area_bool = True
                        if len(self.visualizations_parameter_dict['make_behavioral_videos']['raster_selection_criteria']["other"]) > 0:
                            select_other_bool = all(keyword in one_cluster_file_name for keyword in self.visualizations_parameter_dict['make_behavioral_videos']['raster_selection_criteria']["other"])
                        if len(self.visualizations_parameter_dict['make_behavioral_videos']['raster_selection_criteria']["brain_areas"]) > 0:
                            cl_brain_region = find_region_by_channel(cluster_id=one_cluster_file_name,
                                                                     brain_area_dict=neuropixels_sites_to_anatomy_converter[mouse_track_names[0]][session_id],
                                                                     brain_color_scheme=self.brain_area_color_scheme,
                                                                     return_only_color=False,
                                                                     return_only_area=True)
                            cl_brain_bucket = pool_brain_area(cl_brain_region)
                            select_area_bool = any(
                                area_keyword == cl_brain_region or area_keyword == cl_brain_bucket
                                for area_keyword in self.visualizations_parameter_dict['make_behavioral_videos']['raster_selection_criteria']["brain_areas"]
                            )
                        if select_other_bool and select_area_bool:
                            cluster_files_filtered.append(one_file)
                    cluster_files = cluster_files_filtered

                # load cluster data files
                cluster_data_dict = {}
                for cluster_file in cluster_files:
                    cluster_data_dict[cluster_file.stem] = np.load(file=cluster_file)[1, :]

                # Order units by brain-area bucket so the raster groups areas in
                # a fixed sequence (CENT, SC, PAG, MRN, MB, VTA, other) read top
                # to bottom, matching the area-label order drawn above the raster;
                # WITHIN each area block, order units anatomically by their
                # dorsoventral coordinate (loc_dv) read from the unit catalog.
                # eventplot offsets increase upward, so the bucket key uses the
                # reversed (bottom-to-top) sequence. This single reordering
                # propagates to the raster data, colours and line specs built from
                # the dict below.
                #
                # The catalog keys units by `unit_id`, which equals the cluster-file
                # stem; a `rec_sessions` timestamp is a globally-unique recording,
                # so filtering on the session alone scopes the lookup to this
                # session's units (a single mouse).
                unit_catalog_path = first_match_or_raise(
                    root=pathlib.Path(ephys_directory),
                    pattern='unit_catalog.csv',
                    label="unit catalog CSV",
                )
                unit_catalog_df = pls.read_csv(str(unit_catalog_path),
                                               columns=['unit_id', 'rec_sessions', 'loc_dv'])
                unit_catalog_df = unit_catalog_df.with_columns(pls.col('loc_dv').cast(pls.Float64, strict=False))
                unit_catalog_df = unit_catalog_df.filter(pls.col('rec_sessions').str.contains(session_id, literal=True))
                unit_dv_lookup = dict(zip(unit_catalog_df['unit_id'], unit_catalog_df['loc_dv'], strict=True))

                raster_area_order_top_to_bottom = ['CENT', 'SC', 'PAG', 'MRN', 'MB', 'VTA', 'other']
                raster_area_order_bottom_to_top = raster_area_order_top_to_bottom[::-1]
                cluster_bucket_rank = {}
                cluster_depth = {}
                for cluster_id in cluster_data_dict:
                    cluster_region = find_region_by_channel(cluster_id=cluster_id,
                                                            brain_area_dict=neuropixels_sites_to_anatomy_converter[mouse_track_names[0]][session_id],
                                                            brain_color_scheme=self.brain_area_color_scheme,
                                                            return_only_color=False,
                                                            return_only_area=True)
                    cluster_bucket_rank[cluster_id] = raster_area_order_bottom_to_top.index(pool_brain_area(cluster_region))
                    # units absent from the catalog sort to the end of their block
                    cluster_dv = unit_dv_lookup[cluster_id] if cluster_id in unit_dv_lookup else None
                    cluster_depth[cluster_id] = float(cluster_dv) if cluster_dv is not None else float('inf')
                cluster_data_dict = {cluster_id: cluster_data_dict[cluster_id]
                                     for cluster_id in sorted(cluster_data_dict,
                                                              key=lambda one_cluster_id: (cluster_bucket_rank[one_cluster_id], cluster_depth[one_cluster_id]))}

            if self.visualizations_parameter_dict['make_behavioral_videos']['beh_features_bool']:

                # check if video start time and end time is within spectrogram window
                beh_window_size_frames = int(np.floor(self.visualizations_parameter_dict['make_behavioral_videos']['subplot_specs']['beh_features_window_size'] * empirical_camera_sr))
                beh_half_window_size_frames = int(np.floor((self.visualizations_parameter_dict['make_behavioral_videos']['subplot_specs']['beh_features_window_size'] / 2) * empirical_camera_sr))
                if frame_start - beh_half_window_size_frames < 0 or frame_start + frame_span + beh_half_window_size_frames > mouse_data.shape[0]:
                    self.message_output("Video start time is either too early or too late for behavioral features.")
                    return

                # determine behavioral features to plot
                if len(self.visualizations_parameter_dict['make_behavioral_videos']['beh_features_to_plot']) == 0:
                    beh_features_to_plot = []
                    if len(mouse_track_names) == 1:
                        tentative_features = ["speed", "acceleration", "neck_elevation", "neck_elevation_1st_der",
                                              "allo_roll", "allo_roll_1st_der", "allo_pitch", "allo_pitch_1st_der",
                                              "allo_yaw", "allo_yaw_1st_der", "ego_yaw", "ego_yaw_1st_der",
                                              "body_dir", "body_dir_1st_der", "tail_curvature"]
                        for tentative_feature in tentative_features:
                            beh_features_to_plot.append(f"{mouse_track_names[0]}.{tentative_feature}")
                    else:
                        tentative_features = ["speed", "neck_elevation", "allo_yaw", "allo_pitch", "tail_curvature"]
                        tentative_social_features = ["nose-nose", "allo_yaw-nose", "nose-allo_yaw", "allo_pitch-nose", "nose-allo_pitch", "orofacial-sei"]
                        for mouse_name in mouse_track_names:
                            for tentative_feature in tentative_features:
                                beh_features_to_plot.append(f"{mouse_name}.{tentative_feature}")
                        for tentative_social_feature in tentative_social_features:
                            if '-sei' not in tentative_social_feature:
                                beh_features_to_plot.append(f"{mouse_track_names[0]}-{mouse_track_names[1]}.{tentative_social_feature}")
                            else:
                                beh_features_to_plot.append(f"{mouse_track_names[0]}-{mouse_track_names[1]}.{tentative_social_feature}")
                                beh_features_to_plot.append(f"{mouse_track_names[1]}-{mouse_track_names[0]}.{tentative_social_feature}")

                # get feature data
                beh_feature_data = self.load_beh_features_file()

                beh_window_start = frame_start - beh_half_window_size_frames
                beh_window_end = frame_start + frame_span + beh_half_window_size_frames

                beh_feature_data = beh_feature_data[beh_window_start:beh_window_end, beh_features_to_plot]

            if self.visualizations_parameter_dict['make_behavioral_videos']['spectrogram_bool']:

                # check if video start time and end time is within spectrogram window
                half_window_size_frames = int(np.floor((self.visualizations_parameter_dict['make_behavioral_videos']['subplot_specs']['spectrogram_plot_window_size'] / 2) * empirical_camera_sr))
                if frame_start - half_window_size_frames < 0 or frame_start + frame_span + half_window_size_frames > mouse_data.shape[0]:
                    self.message_output("Video start time is either too early or too late for spectrogram.")
                    return

                active_mic_position = self.visualizations_parameter_dict['make_behavioral_videos']['spectrogram_ch']

                # load microphone data
                audio_data, audio_sr = load_audio_data(root_directory=self.root_directory)

                half_window_size_sec = self.visualizations_parameter_dict['make_behavioral_videos']['subplot_specs']['spectrogram_plot_window_size'] / 2
                camera_frame_in_samples = int(np.ceil(1 / empirical_camera_sr * audio_sr))

                half_window_size_samples = int(np.floor(half_window_size_sec * audio_sr))
                window_start_signal = int(self.visualizations_parameter_dict['make_behavioral_videos']['video_start_time'] * audio_sr) - half_window_size_samples
                window_end_signal = int((self.visualizations_parameter_dict['make_behavioral_videos']['video_start_time'] + self.visualizations_parameter_dict['make_behavioral_videos']['video_duration']) * audio_sr) + half_window_size_samples

                stft_hop = self.visualizations_parameter_dict['make_behavioral_videos']['subplot_specs']['spectrogram_stft_nfft'] // 4
                spectrogram_step = int(camera_frame_in_samples // stft_hop)
                first_fr_spectrogram_center = int((audio_sr * half_window_size_sec) // stft_hop)

                if self.speaker_audio_file != '' and pathlib.Path(self.speaker_audio_file).is_file():
                    speaker_audio_sr, speaker_audio_data = wavfile.read(self.speaker_audio_file)
                    raspi_input_loc = first_match_or_raise(
                        root=pathlib.Path(self.root_directory) / 'audio' / 'cropped_to_video',
                        pattern='m_*ch03_*.wav',
                        label="raspi sync audio (ch03)",
                    )
                    raspi_input_mic_sr, raspi_input_mic_data = wavfile.read(raspi_input_loc)
                    ttl_start, ttl_end = read_ttl_events(raspi_input_mic_data)
                    time_correction_coefficient = 20000  # 80 ms
                    window_start_signal = window_start_signal - ttl_start - time_correction_coefficient
                    window_end_signal = window_end_signal - ttl_start - time_correction_coefficient
                    # The in-range check above validated the window in camera frames
                    # BEFORE this TTL/time correction shifted it in sample space; a
                    # negative start would make speaker_audio_data[start:end] wrap
                    # (numpy negative indexing) and silently render a spectrogram over
                    # the wrong audio. Re-validate against the speaker-audio bounds.
                    if window_start_signal < 0 or window_end_signal > speaker_audio_data.shape[0]:
                        self.message_output("TTL-corrected spectrogram window falls outside the speaker audio bounds; skipping the spectrogram.")
                        return
                    spectrogram_data = librosa.stft(y=speaker_audio_data[window_start_signal:window_end_signal].astype(np.float32),
                                                    n_fft=self.visualizations_parameter_dict['make_behavioral_videos']['subplot_specs']['spectrogram_stft_nfft'])
                else:
                    spectrogram_data = librosa.stft(y=audio_data[window_start_signal:window_end_signal, self.visualizations_parameter_dict['make_behavioral_videos']['spectrogram_ch']].astype(np.float32),
                                                    n_fft=self.visualizations_parameter_dict['make_behavioral_videos']['subplot_specs']['spectrogram_stft_nfft'])

                spectrogram_amplitude = librosa.amplitude_to_db(np.abs(spectrogram_data), ref=np.max(np.abs(spectrogram_data)))

                # find USV onset and offset times for epoch of interest
                if self.speaker_audio_file == '' and not self.visualizations_parameter_dict['make_behavioral_videos']['speaker_bool']:
                    usv_summary_file = first_match_or_raise(
                        root=pathlib.Path(self.root_directory) / 'audio',
                        pattern='*_usv_summary.csv',
                        label="USV summary CSV",
                    )
                    usv_summary_df = pls.read_csv(str(usv_summary_file))
                    usv_summary_df = usv_summary_df.filter((pls.col('stop') >= self.visualizations_parameter_dict['make_behavioral_videos']['video_start_time'] - half_window_size_sec) &
                                                           (pls.col('start') <= self.visualizations_parameter_dict['make_behavioral_videos']['video_start_time'] + self.visualizations_parameter_dict['make_behavioral_videos']['video_duration'] + half_window_size_sec))

        if self.visualizations_parameter_dict['make_behavioral_videos']['plot_theme'] == 'dark':
            color_mode_preferences = self.color_mode_preferences['dark_mode']
        else:
            color_mode_preferences = self.color_mode_preferences['light_mode']

        if self.visualizations_parameter_dict['make_behavioral_videos']['beh_features_bool']:
            n_plot_cols = np.sum([self.visualizations_parameter_dict['make_behavioral_videos']['spectrogram_bool'],
                                  self.visualizations_parameter_dict['make_behavioral_videos']['raster_plot_bool'],
                                  len(beh_features_to_plot)])
        else:
            n_plot_cols = np.sum([self.visualizations_parameter_dict['make_behavioral_videos']['spectrogram_bool'],
                                  self.visualizations_parameter_dict['make_behavioral_videos']['raster_plot_bool']])

        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore")

            fig, ax = plt.subplots(figsize=_VIDEO_FIGSIZE,
                                   nrows=1,
                                   ncols=3 + n_plot_cols,
                                   dpi=_VIDEO_DPI,
                                   squeeze=True)

            # plt.subplots only builds the indexable `ax` scaffold; every panel
            # below is drawn on a freshly created axis (a 3-D subplot for the
            # arena, add_axes for the spectrogram / raster / behavioural-feature
            # panels), so detach the auto-created grid axes now to stop their
            # empty 0-1 frames from showing through the composite. The indices
            # that get reused (ax[0], ax[1], ax[2], ax[3:]) are all rebound
            # before they are touched again.
            for scaffold_axis in ax:
                scaffold_axis.remove()

            # Darken the figure patch in dark mode (light_mode background is
            # #FFFFFF, so this is a no-op there). Without it the figure stays
            # white behind the dark axes and shows through wherever a panel does
            # not cover the canvas — the "white square" border in dark videos.
            fig.set_facecolor(color_mode_preferences['background_color'])

            # The arena is placed full-frame; its companion panels are add_axes
            # drawn on top. The 3-D content size is controlled by arena_zoom via
            # set_box_aspect — an intuitive, layout-independent scale factor that
            # replaces the old plt.subplot + tight_layout(pad) enlargement.
            ax[0] = fig.add_axes(arena_position, projection='3d')
            ax[0].view_init(elev=view_elevation, azim=view_azimuth, roll=0)
            ax[0].set_box_aspect(None, zoom=arena_zoom)
            ax[0].set_facecolor(color_mode_preferences['background_color'])

            plot_arena_corners_mics(data=arena_data,
                                    plot_axes=ax[0],
                                    frame_number=frame_start,
                                    session_id=session_id,
                                    esr=empirical_camera_sr,
                                    animal_id=animal_id_sex_dict,
                                    animal_colors=animal_colors,
                                    color_mode_preferences=color_mode_preferences,
                                    arena_node_connections_bool=self.visualizations_parameter_dict['make_behavioral_videos']['arena_figure_specs']['arena_node_connections_bool'],
                                    arena_node_connections=self.arena_node_connections,
                                    arena_node_names=arena_node_names,
                                    arena_axes_lw=self.visualizations_parameter_dict['make_behavioral_videos']['arena_figure_specs']['arena_axes_lw'],
                                    arena_mics_lw=self.visualizations_parameter_dict['make_behavioral_videos']['arena_figure_specs']['arena_mics_lw'],
                                    arena_mics_opacity=self.visualizations_parameter_dict['make_behavioral_videos']['arena_figure_specs']['arena_mics_opacity'],
                                    plot_corners_bool=self.visualizations_parameter_dict['make_behavioral_videos']['arena_figure_specs']['plot_corners_bool'],
                                    corner_size=self.visualizations_parameter_dict['make_behavioral_videos']['arena_figure_specs']['corner_size'],
                                    corner_opacity=self.visualizations_parameter_dict['make_behavioral_videos']['arena_figure_specs']['corner_opacity'],
                                    plot_mesh_walls_bool=self.visualizations_parameter_dict['make_behavioral_videos']['arena_figure_specs']['plot_mesh_walls_bool'],
                                    mesh_color=color_mode_preferences['arena_mesh_color'],
                                    mesh_opacity=self.visualizations_parameter_dict['make_behavioral_videos']['arena_figure_specs']['mesh_opacity'],
                                    active_mic_bool=self.visualizations_parameter_dict['make_behavioral_videos']['arena_figure_specs']['active_mic_bool'],
                                    active_mic_position=active_mic_position,
                                    active_mic_color=color_mode_preferences['arena_mic_color'],
                                    inactive_mic_bool=self.visualizations_parameter_dict['make_behavioral_videos']['arena_figure_specs']['inactive_mic_bool'],
                                    inactive_mic_color=self.visualizations_parameter_dict['make_behavioral_videos']['arena_figure_specs']['inactive_mic_color'],
                                    text_start_coords=text_start_coords,
                                    main_text_offset=main_text_offset,
                                    mouse_id_text_offset=mouse_id_text_offset,
                                    text_fontsize=self.visualizations_parameter_dict['make_behavioral_videos']['arena_figure_specs']['text_fontsize'])


            plot_mouse_data(data=mouse_data,
                            plot_axes=ax[0],
                            frame_number=frame_start,
                            animal_node_names=mouse_node_names,
                            animal_color=animal_colors,
                            animal_cm=animal_colormaps,
                            animal_line_width=self.visualizations_parameter_dict['make_behavioral_videos']['mouse_figure_specs']['node_connection_lw'],
                            node_bool=self.visualizations_parameter_dict['make_behavioral_videos']['mouse_figure_specs']['node_bool'],
                            node_connections=self.node_connections,
                            node_polygons=self.node_polygons,
                            node_lw=self.visualizations_parameter_dict['make_behavioral_videos']['mouse_figure_specs']['node_lw'],
                            node_size=self.visualizations_parameter_dict['make_behavioral_videos']['mouse_figure_specs']['node_size'],
                            node_opacity=self.visualizations_parameter_dict['make_behavioral_videos']['mouse_figure_specs']['node_opacity'],
                            node_edge_color=color_mode_preferences['node_edge_color'],
                            polygon_color=animal_colors,
                            polygon_opacity=self.visualizations_parameter_dict['make_behavioral_videos']['mouse_figure_specs']['body_opacity'],
                            body_edge_color=color_mode_preferences['body_edge_color'],
                            history_bool=self.visualizations_parameter_dict['make_behavioral_videos']['history_bool'],
                            history_frame_span=int(np.ceil(self.visualizations_parameter_dict['make_behavioral_videos']['mouse_figure_specs']['history_span_sec'] * empirical_camera_sr)),
                            history_point=self.visualizations_parameter_dict['make_behavioral_videos']['mouse_figure_specs']['history_point'],
                            history_ls=self.visualizations_parameter_dict['make_behavioral_videos']['mouse_figure_specs']['history_ls'],
                            history_lw=self.visualizations_parameter_dict['make_behavioral_videos']['mouse_figure_specs']['history_lw'],
                            xlim_=plot_xlim,
                            ylim_=plot_ylim,
                            zlim_=plot_zlim)

            if self.visualizations_parameter_dict['make_behavioral_videos']['speaker_bool']:
                plot_speaker_data(speaker_data=speaker_data,
                                  plot_axes=ax[0],
                                  frame_number=frame_start,
                                  speaker_color=color_mode_preferences['speaker_color'],
                                  speaker_alpha=self.visualizations_parameter_dict['make_behavioral_videos']['arena_figure_specs']['speaker_opacity'])

            if self.visualizations_parameter_dict['make_behavioral_videos']['spectrogram_bool']:
                left, bottom, width, height = spec_fig_position
                ax[1] = fig.add_axes([left, bottom, width, height])
                ax[1].set_facecolor(color_mode_preferences['background_color'])

                if self.speaker_audio_file == '' and not self.visualizations_parameter_dict['make_behavioral_videos']['speaker_bool']:
                    # find USV onsets/offsets for given frame
                    frame_usv_summary_df = usv_summary_df.filter((pls.col('stop') >= self.visualizations_parameter_dict['make_behavioral_videos']['video_start_time'] - half_window_size_sec) &
                                                                 (pls.col('start') <= self.visualizations_parameter_dict['make_behavioral_videos']['video_start_time'] + half_window_size_sec))

                    usv_segments_list = [(usv_start - self.visualizations_parameter_dict['make_behavioral_videos']['video_start_time'],
                                          usv_stop - self.visualizations_parameter_dict['make_behavioral_videos']['video_start_time'])
                                         for usv_start, usv_stop in zip(frame_usv_summary_df['start'], frame_usv_summary_df['stop'], strict=True)]

                    _unassigned = self.visualizations_parameter_dict["unassigned_colors"][0]
                    usv_segments_colors = [animal_colors_dict.get(emitter_id, _unassigned) for emitter_id in frame_usv_summary_df['emitter']]
                else:
                    usv_segments_list = []
                    usv_segments_colors = []

                plot_usv_segments_bool = self.visualizations_parameter_dict['make_behavioral_videos']['subplot_specs']['plot_usv_segments_bool'] and self.speaker_audio_file == '' and not self.visualizations_parameter_dict['make_behavioral_videos']['speaker_bool']

                relative_fr = 0
                spec_start = relative_fr * spectrogram_step
                spec_end = (first_fr_spectrogram_center * 2) + (relative_fr * spectrogram_step)

                plot_spectrogram(plot_axes=ax[1],
                                 figure_object=fig,
                                 spec_start=spec_start,
                                 spec_end=spec_end,
                                 audio_sr=audio_sr,
                                 stft_hop=stft_hop,
                                 half_window_size_sec=half_window_size_sec,
                                 cbar_bool=self.visualizations_parameter_dict['make_behavioral_videos']['subplot_specs']['spectrogram_cbar_bool'],
                                 color_mode_preferences=color_mode_preferences,
                                 spectrogram_amplitude=spectrogram_amplitude,
                                 power_limit=self.visualizations_parameter_dict['make_behavioral_videos']['subplot_specs']['spectrogram_power_limit'],
                                 freq_limit=self.visualizations_parameter_dict['make_behavioral_videos']['subplot_specs']['spectrogram_frequency_limit'],
                                 freq_yticks=self.visualizations_parameter_dict['make_behavioral_videos']['subplot_specs']['spectrogram_yticks'],
                                 plot_usv_segments_bool=plot_usv_segments_bool,
                                 usv_segments_list=usv_segments_list,
                                 usv_segment_lw=self.visualizations_parameter_dict['make_behavioral_videos']['subplot_specs']['usv_segments_lw'],
                                 usv_segments_ypos=self.visualizations_parameter_dict['make_behavioral_videos']['subplot_specs']['usv_segments_ypos'],
                                 usv_segment_colors_list=usv_segments_colors)

            if self.visualizations_parameter_dict['make_behavioral_videos']['raster_plot_bool']:
                if (self.visualizations_parameter_dict['make_behavioral_videos']['animate_bool'] and
                        self.visualizations_parameter_dict['make_behavioral_videos']['spike_sound_bool']):
                    for special_unit in self.visualizations_parameter_dict['make_behavioral_videos']['raster_special_units']:
                        spikes_for_sound = filter_spikes_for_raster(input_arr=cluster_data_dict[special_unit],
                                                                    ra_st_fr=frame_start,
                                                                    ra_end_fr=frame_start + frame_span,
                                                                    fr_start=frame_start)
                        create_spike_sound_file(audio_duration=self.visualizations_parameter_dict['make_behavioral_videos']['video_duration'],
                                                spike_array=spikes_for_sound,
                                                sound_save_directory=putative_save_directory,
                                                sound_session_id=session_id,
                                                sound_frame_start=frame_start,
                                                sound_frame_span=frame_span,
                                                tracking_esr=empirical_camera_sr,
                                                unit_id=special_unit)

                raster_half_window_sec = round(self.visualizations_parameter_dict['make_behavioral_videos']['subplot_specs']['raster_window_size'] / 2, 2)
                raster_half_window = int(np.floor(self.visualizations_parameter_dict['make_behavioral_videos']['subplot_specs']['raster_window_size'] * empirical_camera_sr / 2))
                raster_start_frame = frame_start - raster_half_window
                raster_end_frame = frame_start + raster_half_window
                raster_data = []
                event_plot_colors = []
                if len(self.visualizations_parameter_dict['make_behavioral_videos']['raster_special_units']) == 0:
                    event_plot_line_lengths = [self.visualizations_parameter_dict['make_behavioral_videos']['subplot_specs']['raster_ll']] * len(cluster_data_dict.keys())
                    event_plot_line_widths = [self.visualizations_parameter_dict['make_behavioral_videos']['subplot_specs']['raster_lw']] * len(cluster_data_dict.keys())
                else:
                    special_unit_ll = max(1, int(np.floor(len(cluster_data_dict.keys()) / 36)))
                    event_plot_line_lengths = [self.visualizations_parameter_dict['make_behavioral_videos']['subplot_specs']['raster_ll']
                                               if one_unit not in self.visualizations_parameter_dict['make_behavioral_videos']['raster_special_units'] else special_unit_ll for one_unit in cluster_data_dict.keys()]
                    event_plot_line_widths = [self.visualizations_parameter_dict['make_behavioral_videos']['subplot_specs']['raster_lw']
                                              if one_unit not in self.visualizations_parameter_dict['make_behavioral_videos']['raster_special_units'] else .2 for one_unit in cluster_data_dict.keys()]
                    change_brain_area_colors = []
                for cluster_key in cluster_data_dict.keys():
                    raster_data.append(filter_spikes_for_raster(input_arr=cluster_data_dict[cluster_key],
                                                                ra_st_fr=raster_start_frame,
                                                                ra_end_fr=raster_end_frame,
                                                                fr_start=frame_start))

                    if len(self.visualizations_parameter_dict['make_behavioral_videos']['raster_special_units']) == 0:
                        event_plot_colors.append(find_region_by_channel(cluster_id=cluster_key,
                                                                        brain_area_dict=neuropixels_sites_to_anatomy_converter[mouse_track_names[0]][session_id],
                                                                        brain_color_scheme=self.brain_area_color_scheme))
                    elif cluster_key in self.visualizations_parameter_dict['make_behavioral_videos']['raster_special_units']:
                        special_brain_region, special_color = find_region_by_channel(cluster_id=cluster_key,
                                                                                     brain_area_dict=neuropixels_sites_to_anatomy_converter[mouse_track_names[0]][session_id],
                                                                                     brain_color_scheme=self.brain_area_color_scheme,
                                                                                     return_only_color=False)
                        event_plot_colors.append(special_color)
                        special_bucket = pool_brain_area(special_brain_region)
                        if special_bucket not in change_brain_area_colors:
                            change_brain_area_colors.append(special_bucket)
                    else:
                        event_plot_colors.append(self.brain_area_color_scheme['other'])

                if len(self.visualizations_parameter_dict['make_behavioral_videos']['raster_special_units']) > 0:
                    grey_hex = self.brain_area_color_scheme['other']
                    self.brain_area_color_scheme = {one_key: (grey_hex if one_key not in change_brain_area_colors else one_value) for one_key, one_value in self.brain_area_color_scheme.items()}

                left, bottom, width, height = raster_fig_position
                ax[2] = fig.add_axes([left, bottom, width, height])
                ax[2].set_facecolor(color_mode_preferences['background_color'])

                plot_raster(plot_axes=ax[2],
                            figure_object=fig,
                            unit_num=len(cluster_data_dict.keys()),
                            raster_data=raster_data,
                            raster_half_window=raster_half_window,
                            raster_half_window_sec=raster_half_window_sec,
                            raster_brain_area=neuropixels_sites_to_anatomy_converter[mouse_track_names[0]][session_id],
                            raster_line_lengths=event_plot_line_lengths,
                            raster_line_widths=event_plot_line_widths,
                            filtered_brain_areas=self.visualizations_parameter_dict['make_behavioral_videos']['raster_selection_criteria']['brain_areas'],
                            color_mode_preferences=color_mode_preferences,
                            event_plot_colors=event_plot_colors,
                            brain_area_color_scheme=self.brain_area_color_scheme)

            if self.visualizations_parameter_dict['make_behavioral_videos']['beh_features_bool']:
                ylim_dict = {}
                for feature_name in beh_features_to_plot:
                    feature_name_alone = feature_name.split('.')[1]
                    new_min = beh_feature_data[feature_name].min()
                    new_max = beh_feature_data[feature_name].max()
                    if feature_name_alone not in ylim_dict:
                        ylim_dict[feature_name_alone] = [new_min, new_max]
                    else:
                        ylim_dict[feature_name_alone][0] = min(ylim_dict[feature_name_alone][0], new_min)
                        ylim_dict[feature_name_alone][1] = max(ylim_dict[feature_name_alone][1], new_max)

                beginning_feature_ts_fr_start = 0
                beginning_feature_ts_fr_end = beh_window_size_frames
                beginning_feature_ts_fr_middle = beh_half_window_size_frames

                plot_behavioral_features(plot_axes=ax,
                                         figure_object=fig,
                                         mouse_track_names=mouse_track_names,
                                         special_features=self.visualizations_parameter_dict['make_behavioral_videos']['special_beh_features'],
                                         beh_features_to_plot=beh_features_to_plot,
                                         beh_feature_data=beh_feature_data,
                                         beh_features_fig_position=beh_features_fig_position,
                                         beh_window_size_sec=self.visualizations_parameter_dict['make_behavioral_videos']['subplot_specs']['beh_features_window_size'] / 2,
                                         beh_window_size_frames=beh_window_size_frames,
                                         beh_half_window_size_frames=beh_half_window_size_frames,
                                         beh_features_ylabels=self.beh_features_ylabels,
                                         feature_ts_fr_start=beginning_feature_ts_fr_start,
                                         feature_ts_fr_end=beginning_feature_ts_fr_end,
                                         feature_ts_fr_middle=beginning_feature_ts_fr_middle,
                                         x_axis_start=beginning_feature_ts_fr_start,
                                         x_axis_middle=beginning_feature_ts_fr_middle,
                                         x_axis_end=beginning_feature_ts_fr_end,
                                         ylim_dict=ylim_dict,
                                         plot_theme=self.visualizations_parameter_dict['make_behavioral_videos']['plot_theme'],
                                         color_mode_preferences=color_mode_preferences,
                                         animal_colors=animal_colors)

            def animate(frame_num):
                """
                Description
                -----------
                Matplotlib FuncAnimation callback. Renders a single video frame
                by clearing the 3D arena axis, (optionally) rotating the side
                view, and re-plotting arena geometry, mouse skeletons, and any
                accompanying time-series panels for the current frame index.

                Parameters
                ----------
                frame_num (int)
                    Frame index relative to frame_start (0-based).

                Returns
                -------
                (None)
                    The function is used for its side effects on the current
                    matplotlib figure.
                """

                ax[0].clear()

                # rotates plot
                if (self.visualizations_parameter_dict['make_behavioral_videos']['view_angle'] != 'top'
                        and self.visualizations_parameter_dict['make_behavioral_videos']['rotate_side_view_bool']):
                    azim = (self.visualizations_parameter_dict['make_behavioral_videos']['rotation_speed']* ((frame_num + frame_start) / empirical_camera_sr)
                            + self.visualizations_parameter_dict['make_behavioral_videos']['side_azimuth_start']) % 360
                    ax[0].view_init(elev=view_elevation, azim=azim, roll=0)

                # Axes3D.clear() above resets the box aspect, so re-apply the
                # zoom every frame to keep the arena at its intended size.
                ax[0].set_box_aspect(None, zoom=arena_zoom)

                plot_arena_corners_mics(data=arena_data,
                                        plot_axes=ax[0],
                                        frame_number=frame_num,
                                        session_id=session_id,
                                        esr=empirical_camera_sr,
                                        animal_id=animal_id_sex_dict,
                                        animal_colors=animal_colors,
                                        color_mode_preferences=color_mode_preferences,
                                        arena_node_connections_bool=self.visualizations_parameter_dict['make_behavioral_videos']['arena_figure_specs']['arena_node_connections_bool'],
                                        arena_node_connections=self.arena_node_connections,
                                        arena_node_names=arena_node_names,
                                        arena_axes_lw=self.visualizations_parameter_dict['make_behavioral_videos']['arena_figure_specs']['arena_axes_lw'],
                                        arena_mics_lw=self.visualizations_parameter_dict['make_behavioral_videos']['arena_figure_specs']['arena_mics_lw'],
                                        arena_mics_opacity=self.visualizations_parameter_dict['make_behavioral_videos']['arena_figure_specs']['arena_mics_opacity'],
                                        plot_corners_bool=self.visualizations_parameter_dict['make_behavioral_videos']['arena_figure_specs']['plot_corners_bool'],
                                        corner_size=self.visualizations_parameter_dict['make_behavioral_videos']['arena_figure_specs']['corner_size'],
                                        corner_opacity=self.visualizations_parameter_dict['make_behavioral_videos']['arena_figure_specs']['corner_opacity'],
                                        plot_mesh_walls_bool=self.visualizations_parameter_dict['make_behavioral_videos']['arena_figure_specs']['plot_mesh_walls_bool'],
                                        mesh_color=color_mode_preferences['arena_mesh_color'],
                                        mesh_opacity=self.visualizations_parameter_dict['make_behavioral_videos']['arena_figure_specs']['mesh_opacity'],
                                        active_mic_bool=self.visualizations_parameter_dict['make_behavioral_videos']['arena_figure_specs']['active_mic_bool'],
                                        active_mic_position=active_mic_position,
                                        active_mic_color=color_mode_preferences['arena_mic_color'],
                                        inactive_mic_bool=self.visualizations_parameter_dict['make_behavioral_videos']['arena_figure_specs']['inactive_mic_bool'],
                                        inactive_mic_color=self.visualizations_parameter_dict['make_behavioral_videos']['arena_figure_specs']['inactive_mic_color'],
                                        text_start_coords=text_start_coords,
                                        main_text_offset=main_text_offset,
                                        mouse_id_text_offset=mouse_id_text_offset,
                                        text_fontsize=self.visualizations_parameter_dict['make_behavioral_videos']['arena_figure_specs']['text_fontsize'])

                plot_mouse_data(data=mouse_data,
                                plot_axes=ax[0],
                                frame_number=frame_num,
                                animal_node_names=mouse_node_names,
                                animal_color=animal_colors,
                                animal_cm=animal_colormaps,
                                animal_line_width=self.visualizations_parameter_dict['make_behavioral_videos']['mouse_figure_specs']['node_connection_lw'],
                                node_bool=self.visualizations_parameter_dict['make_behavioral_videos']['mouse_figure_specs']['node_bool'],
                                node_connections=self.node_connections,
                                node_polygons=self.node_polygons,
                                node_lw=self.visualizations_parameter_dict['make_behavioral_videos']['mouse_figure_specs']['node_lw'],
                                node_size=self.visualizations_parameter_dict['make_behavioral_videos']['mouse_figure_specs']['node_size'],
                                node_opacity=self.visualizations_parameter_dict['make_behavioral_videos']['mouse_figure_specs']['node_opacity'],
                                node_edge_color=color_mode_preferences['node_edge_color'],
                                polygon_color=animal_colors,
                                polygon_opacity=self.visualizations_parameter_dict['make_behavioral_videos']['mouse_figure_specs']['body_opacity'],
                                body_edge_color=color_mode_preferences['body_edge_color'],
                                history_bool=self.visualizations_parameter_dict['make_behavioral_videos']['history_bool'],
                                history_frame_span=int(np.ceil(self.visualizations_parameter_dict['make_behavioral_videos']['mouse_figure_specs']['history_span_sec'] * empirical_camera_sr)),
                                history_point=self.visualizations_parameter_dict['make_behavioral_videos']['mouse_figure_specs']['history_point'],
                                history_ls=self.visualizations_parameter_dict['make_behavioral_videos']['mouse_figure_specs']['history_ls'],
                                history_lw=self.visualizations_parameter_dict['make_behavioral_videos']['mouse_figure_specs']['history_lw'],
                                xlim_=plot_xlim,
                                ylim_=plot_ylim,
                                zlim_=plot_zlim)

                if self.visualizations_parameter_dict['make_behavioral_videos']['speaker_bool']:
                    plot_speaker_data(speaker_data=speaker_data,
                                      plot_axes=ax[0],
                                      frame_number=frame_num,
                                      speaker_color=color_mode_preferences['speaker_color'],
                                      speaker_alpha=self.visualizations_parameter_dict['make_behavioral_videos']['arena_figure_specs']['speaker_opacity'])

                if self.visualizations_parameter_dict['make_behavioral_videos']['spectrogram_bool']:
                    ax[1].clear()
                    current_relative_fr = frame_num - frame_start
                    current_spec_start = current_relative_fr * spectrogram_step
                    current_spec_end = (first_fr_spectrogram_center * 2) + (current_relative_fr * spectrogram_step)

                    if self.speaker_audio_file == '' and not self.visualizations_parameter_dict['make_behavioral_videos']['speaker_bool']:
                        # find USV onsets/offsets for given frame
                        current_video_time = frame_num / empirical_camera_sr
                        frame_usv_summary_df_temp = usv_summary_df.filter((pls.col('stop') >= current_video_time - half_window_size_sec) &
                                                                          (pls.col('start') <= current_video_time + half_window_size_sec))

                        usv_segments_list_temp = [(usv_start - current_video_time,
                                                   usv_stop - current_video_time)
                                                  for usv_start, usv_stop in zip(frame_usv_summary_df_temp['start'], frame_usv_summary_df_temp['stop'], strict=True)]

                        _unassigned = self.visualizations_parameter_dict["unassigned_colors"][0]
                        usv_segments_colors_temp = [animal_colors_dict.get(emitter_id, _unassigned) for emitter_id in frame_usv_summary_df_temp['emitter']]
                    else:
                        usv_segments_list_temp = []
                        usv_segments_colors_temp = []

                    plot_spectrogram(plot_axes=ax[1],
                                     figure_object=fig,
                                     spec_start=current_spec_start,
                                     spec_end=current_spec_end,
                                     audio_sr=audio_sr,
                                     stft_hop=stft_hop,
                                     half_window_size_sec=half_window_size_sec,
                                     cbar_bool=False,
                                     color_mode_preferences=color_mode_preferences,
                                     spectrogram_amplitude=spectrogram_amplitude,
                                     power_limit=self.visualizations_parameter_dict['make_behavioral_videos']['subplot_specs']['spectrogram_power_limit'],
                                     freq_limit=self.visualizations_parameter_dict['make_behavioral_videos']['subplot_specs']['spectrogram_frequency_limit'],
                                     freq_yticks=self.visualizations_parameter_dict['make_behavioral_videos']['subplot_specs']['spectrogram_yticks'],
                                     plot_usv_segments_bool=plot_usv_segments_bool,
                                     usv_segments_list=usv_segments_list_temp,
                                     usv_segment_lw=self.visualizations_parameter_dict['make_behavioral_videos']['subplot_specs']['usv_segments_lw'],
                                     usv_segments_ypos=self.visualizations_parameter_dict['make_behavioral_videos']['subplot_specs']['usv_segments_ypos'],
                                     usv_segment_colors_list=usv_segments_colors_temp)

                if self.visualizations_parameter_dict['make_behavioral_videos']['raster_plot_bool']:
                    ax[2].clear()
                    current_raster_start_frame = frame_num - raster_half_window
                    current_raster_end_frame = frame_num + raster_half_window
                    current_raster_data = []
                    for cluster_name in cluster_data_dict.keys():
                        current_raster_data.append(filter_spikes_for_raster(input_arr=cluster_data_dict[cluster_name],
                                                                            ra_st_fr=current_raster_start_frame,
                                                                            ra_end_fr=current_raster_end_frame,
                                                                            fr_start=frame_num))

                    plot_raster(plot_axes=ax[2],
                                figure_object=fig,
                                unit_num=len(cluster_data_dict.keys()),
                                raster_data=current_raster_data,
                                raster_half_window=raster_half_window,
                                raster_half_window_sec=raster_half_window_sec,
                                raster_brain_area=neuropixels_sites_to_anatomy_converter[mouse_track_names[0]][session_id],
                                raster_line_lengths=event_plot_line_lengths,
                                raster_line_widths=event_plot_line_widths,
                                filtered_brain_areas=self.visualizations_parameter_dict['make_behavioral_videos']['raster_selection_criteria']['brain_areas'],
                                color_mode_preferences=color_mode_preferences,
                                event_plot_colors=event_plot_colors,
                                brain_area_color_scheme=self.brain_area_color_scheme)

                if self.visualizations_parameter_dict['make_behavioral_videos']['beh_features_bool']:
                    current_feature_ts_fr_start = frame_num - frame_start
                    current_feature_ts_fr_end = (frame_num - frame_start) + beh_window_size_frames
                    current_feature_ts_fr_middle = (frame_num - frame_start) + beh_half_window_size_frames

                    plot_behavioral_features(plot_axes=ax,
                                             figure_object=fig,
                                             mouse_track_names=mouse_track_names,
                                             special_features=self.visualizations_parameter_dict['make_behavioral_videos']['special_beh_features'],
                                             beh_features_to_plot=beh_features_to_plot,
                                             beh_feature_data=beh_feature_data,
                                             beh_features_fig_position=beh_features_fig_position,
                                             beh_window_size_sec=self.visualizations_parameter_dict['make_behavioral_videos']['subplot_specs']['beh_features_window_size'] / 2,
                                             beh_window_size_frames=beh_window_size_frames,
                                             beh_half_window_size_frames=beh_half_window_size_frames,
                                             beh_features_ylabels=self.beh_features_ylabels,
                                             feature_ts_fr_start=current_feature_ts_fr_start,
                                             feature_ts_fr_end=current_feature_ts_fr_end,
                                             feature_ts_fr_middle=current_feature_ts_fr_middle,
                                             x_axis_start=beginning_feature_ts_fr_start,
                                             x_axis_middle=beginning_feature_ts_fr_middle,
                                             x_axis_end=beginning_feature_ts_fr_end,
                                             ylim_dict=ylim_dict,
                                             plot_theme=self.visualizations_parameter_dict['make_behavioral_videos']['plot_theme'],
                                             color_mode_preferences=color_mode_preferences,
                                             animal_colors=animal_colors,
                                             remove_axes_bool=True)

            # Persist the dark-mode patch colour through both the per-frame video
            # encoder and the static savefig. The mplstyle pins savefig.facecolor
            # to #FFFFFF, so without these the saved frames would revert to a
            # white background regardless of fig.set_facecolor above.
            frame_savefig_kwargs = {'facecolor': color_mode_preferences['background_color'],
                                    'edgecolor': color_mode_preferences['background_color']}

            name_addition = f"{self.visualizations_parameter_dict['make_behavioral_videos']['plot_theme']}"
            name_addition = f"{name_addition}_{self.visualizations_parameter_dict['make_behavioral_videos']['view_angle']}view"
            if self.visualizations_parameter_dict['make_behavioral_videos']['view_angle'] == 'side':
                name_addition = f"{name_addition}_{self.visualizations_parameter_dict['make_behavioral_videos']['side_azimuth_start']}azim"
                if self.visualizations_parameter_dict['make_behavioral_videos']['rotate_side_view_bool'] and self.visualizations_parameter_dict['make_behavioral_videos']['animate_bool']:
                    name_addition = f"{name_addition}_rotate"
            if self.visualizations_parameter_dict['make_behavioral_videos']['beh_features_bool']:
                name_addition = f"{name_addition}_{len(beh_features_to_plot)}features"
            if self.visualizations_parameter_dict['make_behavioral_videos']['spectrogram_bool']:
                name_addition = f"{name_addition}_spectrogram_ch{active_mic_position}"
            if self.visualizations_parameter_dict['make_behavioral_videos']['raster_plot_bool']:
                name_addition = f"{name_addition}_raster"
            if self.visualizations_parameter_dict['make_behavioral_videos']['history_bool']:
                name_addition = f"{name_addition}_history"
            if self.visualizations_parameter_dict['make_behavioral_videos']['speaker_bool']:
                name_addition = f"{name_addition}_speaker"
            name_addition = f"{name_addition}_{self.exp_id}"

            if self.visualizations_parameter_dict['make_behavioral_videos']['animate_bool']:
                anima = FuncAnimation(fig=fig,
                                      func=animate,
                                      frames=range(frame_start, frame_start + frame_span, 1),
                                      interval=round(1 / empirical_camera_sr * 1000, 3))

                animation_file_name = f"{session_id}_3D_{frame_start}-{frame_start + frame_span}fr_{name_addition}"
                animation_file_path = str(putative_save_directory / f"{animation_file_name}.{self.visualizations_parameter_dict['make_behavioral_videos']['general_figure_specs']['animation_format']}")

                if self.visualizations_parameter_dict['make_behavioral_videos']['general_figure_specs']['animation_codec'] is not None:
                    # create a custom writer for the configured codec (e.g. h264_nvenc for NVIDIA GPU acceleration)
                    animation_writer = FFMpegWriter(
                        fps=int(np.floor(empirical_camera_sr)),
                        codec=self.visualizations_parameter_dict['make_behavioral_videos']['general_figure_specs']['animation_codec'],
                        extra_args=['-preset',
                                    self.visualizations_parameter_dict['make_behavioral_videos']['general_figure_specs']['animation_codec_preset_flag'],
                                    '-tune',
                                    self.visualizations_parameter_dict['make_behavioral_videos']['general_figure_specs']['animation_codec_tune_flag']]
                    )
                else:
                    animation_writer = self.visualizations_parameter_dict['make_behavioral_videos']['general_figure_specs']['animation_writer']

                try:
                    if isinstance(animation_writer, FFMpegWriter):
                        self.message_output(f"Using configured codec ({self.visualizations_parameter_dict['make_behavioral_videos']['general_figure_specs']['animation_codec']}) for video encoding...")
                        smart_wait(app_context_bool=self.app_context_bool, seconds=1)
                        anima.save(filename=animation_file_path,
                                   writer=animation_writer,
                                   dpi=_VIDEO_DPI,
                                   savefig_kwargs=frame_savefig_kwargs)
                    else:
                        # CPU attempt: Pass the string AND the fps.
                        self.message_output("Using default CPU for video encoding...")
                        smart_wait(app_context_bool=self.app_context_bool, seconds=1)
                        anima.save(filename=animation_file_path,
                                   writer=animation_writer,
                                   fps=int(np.floor(empirical_camera_sr)),
                                   dpi=_VIDEO_DPI,
                                   savefig_kwargs=frame_savefig_kwargs)

                except Exception as write_error:
                    self.message_output(f"WARNING: Video saving failed. Error: {write_error}")
                    smart_wait(app_context_bool=self.app_context_bool, seconds=1)

                    # Check if the failed attempt was with the GPU
                    if isinstance(animation_writer, FFMpegWriter):
                        self.message_output("Falling back to the default CPU encoder...")
                        smart_wait(app_context_bool=self.app_context_bool, seconds=1)
                        cpu_writer_fallback = self.visualizations_parameter_dict['make_behavioral_videos']['general_figure_specs']['animation_writer']

                        # The fallback CPU call MUST include fps.
                        anima.save(filename=animation_file_path,
                                   writer=cpu_writer_fallback,
                                   fps=int(np.floor(empirical_camera_sr)),
                                   dpi=_VIDEO_DPI,
                                   savefig_kwargs=frame_savefig_kwargs)
                    else:
                        self.message_output("CPU encoding failed. No fallback available.")
                        smart_wait(app_context_bool=self.app_context_bool, seconds=1)
                        raise write_error

                if (self.visualizations_parameter_dict['make_behavioral_videos']['animate_bool'] and
                        self.visualizations_parameter_dict['make_behavioral_videos']['spike_sound_bool']):
                    for unit_id in self.visualizations_parameter_dict['make_behavioral_videos']['raster_special_units']:
                        audio_file_name = f"{session_id}_3D_{frame_start}-{frame_start + frame_span}fr_spike_sound_{unit_id}.wav"
                        output_video_name = animation_file_name + f"_spike_sound_{unit_id}.{self.visualizations_parameter_dict['make_behavioral_videos']['general_figure_specs']['animation_format']}"
                        subprocess.Popen(args=f"{self.command_addition}ffmpeg -y -i {animation_file_name}.{self.visualizations_parameter_dict['make_behavioral_videos']['general_figure_specs']['animation_format']} -i {audio_file_name} -c:v copy -c:a aac {output_video_name}",
                                         stdout=subprocess.DEVNULL,
                                         stderr=subprocess.STDOUT,
                                         cwd=putative_save_directory,
                                         shell=self.shell_usage_bool).wait()
                        (putative_save_directory / audio_file_name).unlink()

                mbv = self.visualizations_parameter_dict['make_behavioral_videos']
                if mbv['pitch_shifted_audio_bool']:
                    # auto-produce the audible (pitch-shifted) audio for THIS video's time window
                    fs_dict = dict(mbv['pitch_shifted_audio_specs'])
                    fs_dict['fs_sequence_start'] = mbv['video_start_time']
                    fs_dict['fs_sequence_duration'] = mbv['video_duration']
                    usv_sound_path = AudioGenerator(root_directory=self.root_directory,
                                                    freq_shift_settings_dict=fs_dict,
                                                    message_output=self.message_output).frequency_shift_audio_segment(seq_start=mbv['video_start_time'],
                                                                                                                      seq_duration=mbv['video_duration'])
                    if usv_sound_path is not None and pathlib.Path(usv_sound_path).is_file():
                        output_video_name = animation_file_name + f"_with_USV_sound.{mbv['general_figure_specs']['animation_format']}"
                        mux_return_code = subprocess.Popen(args=f"{self.command_addition}ffmpeg -y -i {animation_file_name}.{mbv['general_figure_specs']['animation_format']} -i {usv_sound_path} -c:v copy -c:a aac {output_video_name}",
                                                           stdout=subprocess.DEVNULL,
                                                           stderr=subprocess.STDOUT,
                                                           cwd=putative_save_directory,
                                                           shell=self.shell_usage_bool).wait()
                        if mux_return_code != 0:
                            self.message_output(f"WARNING: ffmpeg mux of pitch-shifted audio failed (return code {mux_return_code}).")

            else:
                if self.visualizations_parameter_dict['make_behavioral_videos']['save_fig']:
                    video_fig_specs = self.visualizations_parameter_dict['make_behavioral_videos']['general_figure_specs']
                    # bbox_inches=None (NOT 'tight'): the static frame is a single
                    # frame of the same composite as the video, so it must use the
                    # identical fixed [0,1] canvas. Tight-cropping here would hide
                    # any panel that overflows the frame edge, making the preview
                    # disagree with the video (where overflow is clipped). Saving
                    # the full frame keeps the preview truthful — tune once, match
                    # both.
                    fig_loc = save_figure(
                        fig,
                        stem=f"{session_id}_3D_{frame_start}fr_{name_addition}",
                        viz_settings=self.visualizations_parameter_dict,
                        override_dir=putative_save_directory,
                        override_format=video_fig_specs['fig_format'],
                        override_dpi=_VIDEO_DPI,
                        bbox_inches=None,
                        **frame_savefig_kwargs,
                    )

                    # open image in default viewer if display available
                    os_type = platform.system()
                    if os_type == 'Windows':
                        if 'WT_SESSION' in os.environ or 'USERNAME' in os.environ:
                            os.startfile(str(fig_loc.resolve()))
                    elif os_type == 'Darwin':
                        if 'DISPLAY' in os.environ:
                            subprocess.run(args=['open', str(fig_loc.resolve())], check=True)
                    elif os_type == 'Linux':
                        if 'DISPLAY' in os.environ:
                            subprocess.run(args=['xdg-open', str(fig_loc.resolve())], check=True)
                    else:
                        self.message_output("Unsupported operating system for opening image.")

            # Close the figure for BOTH branches (video + static): pyplot keeps
            # figures in its registry until closed, so a per-session run would
            # otherwise accumulate one open figure per session (the animate
            # branch previously never closed it).
            plt.close(fig)
