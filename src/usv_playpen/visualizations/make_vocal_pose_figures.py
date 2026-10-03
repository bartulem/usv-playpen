"""
@author: bartulem
Vocal-pose figures: a spectrogram of the calls in a short window above the two animals' 3D pose at its last frame.

The still (:class:`VocalPoseStillMaker`) stacks two panels. On top is the spectrogram of one
microphone over the window, every time column coloured by the emitter of the call covering it
(white to the male's colour, white to the female's, white to the unassigned colour outside any
listed call), each emitter's USVs and squeaks scaled to their own loudest sound, with a time bar
beneath whose strength follows the fade of the pose trails. Below are both animals at the last
frame, drawn with the shared mouse renderer (:func:`plot_mouse_data`) in one colour each, see-through
bodies, over one silhouette per camera frame of their preceding movement that fades with age and ends
after ``trail_end_seconds``. The camera can stand on the male's side of the pair automatically; the
skeleton's line widths scale with the panel's zoom so the animals look the same in every figure; a
scale mark of three perpendicular arms, the arena floor and nearest wall, and a two-swatch colour key
are optional. The PNG is cropped to its content; an SVG with live text and vector poses can be written
beside it.

The video (:class:`VocalPoseVideoMaker`) plays the window at a chosen speed with a short fading trail
behind each animal, a fixed, following or turning camera, and a scrolling spectrogram (inset or strip)
that is sharp only around the present.

:func:`find_vocal_pose_windows` ranks the windows of a session that hold only the male's USVs with the
animals close together, and :func:`plot_vocal_pose_window_candidates` stacks their spectrograms for
choosing by eye. :func:`vocal_pose_view_picker_html` writes a page on which the frame can be turned by
hand to read off a camera azimuth and elevation.

Every tunable lives in the ``vocal_pose_figures`` block of ``visualizations_settings.json``; the
animal colours come from the ``male_colors`` / ``female_colors`` / ``unassigned_colors`` palette and
the arena from ``make_behavioral_videos.arena_directory``.
"""

from __future__ import annotations

import base64
import json
import os
import pathlib
import shutil
import subprocess
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import h5py
import librosa
import matplotlib.pyplot as plt
import numpy as np
import polars as pls
import soundfile as sf
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.transforms import Bbox, TransformedBbox
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image
from scipy.ndimage import gaussian_filter1d

from ..analyses.decode_experiment_label import extract_information
from ..os_utils import drop_noise_usvs, first_match_or_raise
from .auxiliary_plot_functions import choose_animal_colors
from .figure_io import resolve_save_path
from .make_behavioral_videos import (
    MOUSE_NODE_CONNECTIONS,
    MOUSE_NODE_POLYGONS,
    load_arena_tracks,
    plot_mouse_data,
)
from .plot_style import apply_plot_style


def hex_to_rgb(color: str) -> np.ndarray:
    """
    Description
    -----------
    A hex colour as an RGB triple in 0-1.

    Parameters
    ----------
    color (str)
        Hex colour such as ``"#9AC0CD"``.

    Returns
    -------
    rgb (np.ndarray)
        Shape (3,), floats in 0-1.
    """
    return np.array([int(color[i:i + 2], 16) for i in (1, 3, 5)], dtype=float) / 255.0


def rgb_to_hex(rgb: np.ndarray) -> str:
    """
    Description
    -----------
    An RGB triple in 0-1 as a hex colour.

    Parameters
    ----------
    rgb (np.ndarray)
        Shape (3,), floats in 0-1.

    Returns
    -------
    color (str)
        Hex colour.
    """
    return '#{:02X}{:02X}{:02X}'.format(*np.round(np.clip(rgb, 0.0, 1.0) * 255).astype(int))


def blend_colors(color: str, target: str, strength: float) -> str:
    """
    Description
    -----------
    A colour mixed toward a target colour: ``strength`` 1 keeps the colour, 0 gives the target.

    Parameters
    ----------
    color (str)
        Hex colour.
    target (str)
        Hex colour mixed in (the page or surface the colour fades into).
    strength (float)
        0-1.

    Returns
    -------
    blended (str)
        Hex colour.
    """
    return rgb_to_hex(hex_to_rgb(target) + (hex_to_rgb(color) - hex_to_rgb(target)) * strength)


def trail_strength(age: np.ndarray | float, trail: dict[str, Any], trail_end_seconds: float) -> np.ndarray:
    """
    Description
    -----------
    Colour strength of a trail silhouette of a given age: an exponential decay from
    ``trail['strongest']`` to ``trail['floor']`` with time constant ``trail['fade_seconds']``,
    which, when the trail has an end, tapers linearly to nothing over the last
    ``trail['taper_fraction']`` of the time before that end.

    Parameters
    ----------
    age (np.ndarray | float)
        Time before the frame (s).
    trail (dict)
        The ``trail`` settings sub-block (``strongest``, ``fade_seconds``, ``floor``, ``taper_fraction``).
    trail_end_seconds (float)
        Age at which the trail has faded to nothing (s); 0 or less keeps it for the whole window.

    Returns
    -------
    strength (np.ndarray)
        0 (invisible) to 1 (the animal's full colour), same shape as ``age``.
    """
    ages = np.asarray(age, dtype=float)
    strength = trail['floor'] + (trail['strongest'] - trail['floor']) * np.exp(-ages / trail['fade_seconds'])
    if trail_end_seconds > 0:
        taper = (trail_end_seconds - ages) / (trail['taper_fraction'] * trail_end_seconds)
        strength = strength * np.clip(taper, 0.0, 1.0)
    return np.asarray(strength, dtype=float)


def microphone_wav(root_directory: pathlib.Path, microphone: str) -> pathlib.Path:
    """
    Description
    -----------
    The unfiltered, video-cropped wav of one microphone, named by device and channel.

    Parameters
    ----------
    root_directory (pathlib.Path)
        Session directory.
    microphone (str)
        Device letter and 1-based channel, e.g. ``"m07"`` or ``"s04"``.

    Returns
    -------
    wav_path (pathlib.Path)
        The wav file.
    """
    return first_match_or_raise(root=root_directory / 'audio' / 'cropped_to_video',
                                pattern=f'{microphone[0]}_*_ch{int(microphone[1:]):02d}_cropped_to_video.wav',
                                recursive=False, label=f'cropped_to_video wav of microphone {microphone}')


def peak_microphone(calls: pls.DataFrame) -> str:
    """
    Description
    -----------
    The microphone on which most of the given calls peak. ``peak_amp_ch`` indexes the 24-channel
    recording: 0-11 are device ``m`` channels 1-12, 12-23 device ``s`` channels 1-12.

    Parameters
    ----------
    calls (pls.DataFrame)
        Rows of a USV summary with a ``peak_amp_ch`` column.

    Returns
    -------
    microphone (str)
        Device letter and channel, e.g. ``"s04"``.
    """
    index = int(np.bincount(calls['peak_amp_ch'].to_numpy().astype(int)).argmax())
    return f'm{index + 1:02d}' if index < 12 else f's{index - 11:02d}'


def read_session_calls(root_directory: pathlib.Path, message_output: Callable[..., Any]) -> pls.DataFrame:
    """
    Description
    -----------
    The session's USV summary without its noise rows.

    Parameters
    ----------
    root_directory (pathlib.Path)
        Session directory.
    message_output (Callable)
        Message sink.

    Returns
    -------
    calls (pls.DataFrame)
        Non-noise rows, sorted by ``start``.
    """
    summary_path = first_match_or_raise(root=root_directory / 'audio', pattern='*_usv_summary.csv', recursive=False, label='USV summary')
    calls = pls.read_csv(summary_path, schema_overrides={'emitter': pls.Utf8})
    kept, _ = drop_noise_usvs(calls, source=summary_path.name, message_output=message_output)
    sorted_calls: pls.DataFrame = kept.sort('start')
    return sorted_calls


def load_session_poses(root_directory: pathlib.Path, visualizations_parameter_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Description
    -----------
    The session's 3D tracks with everything the figures need about the animals.

    Parameters
    ----------
    root_directory (pathlib.Path)
        Session directory.
    visualizations_parameter_dict (dict)
        The visualizations settings (for the sex palette).

    Returns
    -------
    session (dict)
        ``tracks`` (frames, animals, nodes, 3), ``names`` (track names), ``nodes`` (node names), ``fps``,
        ``sex_of`` (track name -> ``'male'`` / ``'female'``), ``colors`` (hex per animal, in track order).
    """
    h5_path = first_match_or_raise(root=root_directory / 'video', pattern='[!speaker]*_points3d_translated_rotated_metric.h5',
                                   recursive=True, label='translated/rotated mouse points3d .h5')
    with h5py.File(h5_path, 'r') as h5_file:
        tracks = np.array(h5_file['tracks'])
        names = [item.decode('utf-8') for item in h5_file['track_names']]
        nodes = [item.decode('utf-8') for item in h5_file['node_names']]
        fps = float(h5_file['recording_frame_rate'][()])
        experimental_code = h5_file['experimental_code'][()].decode('utf-8')
    experiment_info = extract_information(experiment_code=experimental_code)
    if experiment_info is None:
        message = f'The experimental code {experimental_code!r} of {h5_path} could not be decoded.'
        raise ValueError(message)
    sex_of = dict(zip(names, experiment_info['mouse_sex'], strict=True))
    colors = choose_animal_colors(exp_info_dict=experiment_info, visualizations_parameter_dict=visualizations_parameter_dict)
    return {'tracks': tracks, 'names': names, 'nodes': nodes, 'fps': fps, 'sex_of': sex_of, 'colors': list(colors)}


def emitter_spectrogram(root_directory: pathlib.Path, calls: pls.DataFrame, sex_of: dict[str, Any], sex_color: dict[str, Any], none_color: str,
                        end_seconds: float, history_seconds: float, microphone: str, spectrogram: dict[str, Any],
                        message_output: Callable[..., Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Description
    -----------
    The spectrogram of a window as an RGB image in which every time column is coloured by the emitter
    of the call covering it: white to the animal's colour, continuing to a darker shade of the same
    hue when that sex's ``*_deep`` factor is below 1 (the loud core of a narrowband USV then stands
    out on white), and white to ``none_color`` outside any listed call. Each emitter's USVs and squeaks
    are scaled separately to their own loudest sound (a squeak is tens of dB louder than a USV), the
    floor is shared, so silence is white throughout. With ``db_floor`` set to ``"adaptive"`` the floor
    sits ``adaptive_floor_above_median_db`` above the band's median power (the noise floor, calls
    occupying few time-frequency bins).

    Parameters
    ----------
    root_directory (pathlib.Path)
        Session directory.
    calls (pls.DataFrame)
        Non-noise USV summary rows (``start``, ``stop``, ``emitter``, ``squeak``).
    sex_of (dict)
        Track name -> sex.
    sex_color (dict)
        Sex -> hex colour.
    none_color (str)
        Hex colour of sound outside any listed call.
    end_seconds (float)
        End of the window from video start (s).
    history_seconds (float)
        Window length (s).
    microphone (str)
        Device letter and channel, e.g. ``"m07"``.
    spectrogram (dict)
        The ``spectrogram`` settings sub-block.
    message_output (Callable)
        Message sink.

    Returns
    -------
    image, band, times (tuple[np.ndarray, np.ndarray, np.ndarray])
        ``image`` (frequencies in band, columns, 3) RGB in 0-1 with the lowest frequency first, ``band``
        the row frequencies (kHz), ``times`` the column times relative to ``end_seconds`` (s, negative).
    """
    n_fft = int(spectrogram['n_fft'])
    hop = n_fft // int(spectrogram['hop_fraction'])
    t0 = end_seconds - history_seconds
    with sf.SoundFile(str(microphone_wav(root_directory, microphone))) as handle:
        rate = handle.samplerate
        handle.seek(round(t0 * rate))
        audio = handle.read(round(history_seconds * rate), dtype='float32')
    power_db = librosa.power_to_db(np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop)) ** 2, ref=np.max)
    freqs = librosa.fft_frequencies(sr=rate, n_fft=n_fft) / 1000.0
    times = librosa.frames_to_time(np.arange(power_db.shape[1]), sr=rate, hop_length=hop) - history_seconds
    white = np.ones(3)
    column_rgb = np.tile(hex_to_rgb(none_color), (times.size, 1))
    column_deep = column_rgb.copy()
    column_split = np.ones(times.size)
    column_owner = np.full(times.size, 'none', dtype=object)
    inside_window = calls.filter((pls.col('stop') >= t0) & (pls.col('start') <= end_seconds))
    pad = float(spectrogram['call_pad_seconds'])
    for row in inside_window.iter_rows(named=True):
        emitter = row['emitter']
        sex = sex_of[emitter] if emitter is not None and emitter in sex_of else None
        if sex is None:
            continue
        inside = (times >= row['start'] - end_seconds - pad) & (times <= row['stop'] - end_seconds + pad)
        deep = float(spectrogram[f'{sex}_deep'])
        column_rgb[inside] = hex_to_rgb(sex_color[sex])
        column_deep[inside] = hex_to_rgb(sex_color[sex]) * deep
        column_split[inside] = float(spectrogram['ramp_split']) if deep < 1.0 else 1.0
        column_owner[inside] = f"{sex} {'squeak' if row['squeak'] else 'USV'}"
    audible = (freqs >= spectrogram['freq_range_khz'][0]) & (freqs <= spectrogram['freq_range_khz'][1])
    if spectrogram['db_floor'] == 'adaptive':
        db_floor = float(np.median(power_db[audible])) + float(spectrogram['adaptive_floor_above_median_db'])
    else:
        db_floor = float(spectrogram['db_floor'])
    ceilings = {owner: float(np.percentile(power_db[np.ix_(audible, column_owner == owner)], spectrogram['ceiling_percentile']))
                for owner in sorted(set(column_owner) - {'none'})}
    ceilings['none'] = max(ceilings.values()) if ceilings else float(np.max(power_db[audible]))
    message_output(f'  spectrogram of {microphone}: floor {db_floor:.0f} dB re window max; ceilings '
                   + ', '.join(f'{owner} {value:.1f}' for owner, value in ceilings.items()))
    column_ceiling = np.array([ceilings[owner] for owner in column_owner])
    intensity = np.clip((power_db - db_floor) / (column_ceiling[None, :] - db_floor), 0.0, 1.0) ** float(spectrogram['gamma'])
    lower = np.clip(intensity / column_split[None, :], 0.0, 1.0)
    upper = np.clip((intensity - column_split[None, :]) / np.maximum(1.0 - column_split[None, :], np.finfo(float).eps), 0.0, 1.0)
    image = (white[None, None, :] + (column_rgb[None, :, :] - white[None, None, :]) * lower[:, :, None]
             + (column_deep[None, :, :] - column_rgb[None, :, :]) * upper[:, :, None])
    return image[audible], freqs[audible], times


def male_side_azimuth(pose: np.ndarray, names: list[str], nodes: list[str], sex_of: dict[str, Any]) -> float:
    """
    Description
    -----------
    The camera azimuth that puts the camera on the male's side of the pair: it looks along the line
    from the female's body centre to the male's (tails excluded), so the male is nearest the viewer
    and lowest in the picture.

    Parameters
    ----------
    pose (np.ndarray)
        (animals, nodes, 3) of one frame.
    names (list[str])
        Track names, in the pose's animal order.
    nodes (list[str])
        Node names.
    sex_of (dict)
        Track name -> sex.

    Returns
    -------
    azimuth (float)
        Degrees, in matplotlib's convention.
    """
    body = [i for i, name in enumerate(nodes) if not name.startswith('Tail')]
    male = next(i for i, name in enumerate(names) if sex_of[name] == 'male')
    female = next(i for i, name in enumerate(names) if sex_of[name] == 'female')
    offset = np.nanmean(pose[male][body, :2], axis=0) - np.nanmean(pose[female][body, :2], axis=0)
    return float(np.degrees(np.arctan2(offset[1], offset[0])))


def resolve_azimuth(camera: dict[str, Any], pose: np.ndarray, names: list[str], nodes: list[str], sex_of: dict[str, Any]) -> float:
    """
    Description
    -----------
    The camera azimuth from the ``camera`` settings: a number, or ``"male_side"`` for the automatic rule.

    Parameters
    ----------
    camera (dict)
        The ``camera`` settings sub-block.
    pose (np.ndarray)
        (animals, nodes, 3) of the frame the rule is applied to.
    names (list[str])
        Track names.
    nodes (list[str])
        Node names.
    sex_of (dict)
        Track name -> sex.

    Returns
    -------
    azimuth (float)
        Degrees.
    """
    if camera['azimuth'] == 'male_side':
        return male_side_azimuth(pose, names, nodes, sex_of)
    return float(camera['azimuth'])


def screen_axes(azimuth: float, elevation: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Description
    -----------
    The world directions that project to screen right and screen up for an orthographic matplotlib
    camera at the given azimuth and elevation.

    Parameters
    ----------
    azimuth (float)
        Degrees.
    elevation (float)
        Degrees.

    Returns
    -------
    right, up (tuple[np.ndarray, np.ndarray])
        Unit vectors, shape (3,) each.
    """
    a, e = np.radians(azimuth), np.radians(elevation)
    right = np.array([-np.sin(a), np.cos(a), 0.0])
    up = np.array([-np.sin(e) * np.cos(a), -np.sin(e) * np.sin(a), np.cos(e)])
    return right, up


class PosePanel:
    """
    Description
    -----------
    The 3D panel shared by the still's PNG and SVG renders: both animals' trails and current poses,
    the optional scale mark and arena surfaces, drawn into any figure with the same camera and zoom.
    Holds the centred poses of the frames still visible, their ages, the camera, the skeleton
    widths scaled to the panel's zoom, and the draw order of the animals.

    Parameters
    ----------
    session (dict)
        From :func:`load_session_poses`.
    settings (dict)
        The ``vocal_pose_figures`` settings block.
    end_seconds (float)
        Time of the frame shown (s from video start).
    arena (dict | None)
        From :func:`arena_geometry`, or None when no surfaces are drawn.
    message_output (Callable)
        Message sink.

    Returns
    -------
    None
    """

    def __init__(self, session: dict[str, Any], settings: dict[str, Any], end_seconds: float, arena: dict[str, Any] | None, message_output: Callable[..., Any]) -> None:
        self.settings = settings
        self.message_output = message_output
        self.names = session['names']
        self.nodes = session['nodes']
        self.sex_of = session['sex_of']
        self.colors = session['colors']
        self.arena = arena
        window, trail, skeleton, layout = settings['window'], settings['trail'], settings['skeleton'], settings['layout']
        fps = session['fps']
        last = int(np.floor(end_seconds * fps))
        first = last - round(window['history_seconds'] * fps)
        poses = session['tracks'][first:last + 1]
        ages = (last - np.arange(first, last + 1)) / fps
        self.trail_end_seconds = float(window['trail_end_seconds'])
        if self.trail_end_seconds > 0:
            poses, ages = poses[ages <= self.trail_end_seconds], ages[ages <= self.trail_end_seconds]
        self.ages = ages
        self.strengths = trail_strength(ages, trail, self.trail_end_seconds)
        self.azimuth = resolve_azimuth(settings['camera'], poses[-1], self.names, self.nodes, self.sex_of)
        self.elevation = float(settings['camera']['elevation'])
        self.centre = 0.5 * (np.nanmin(poses[..., :2].reshape(-1, 2), axis=0) + np.nanmax(poses[..., :2].reshape(-1, 2), axis=0))
        self.poses = poses.copy()
        self.poses[..., 0] -= self.centre[0]
        self.poses[..., 1] -= self.centre[1]
        self.half_extent = float(np.nanmax(np.abs(self.poses[..., :2]))) + float(skeleton['margin_m'])
        self.z_top = float(np.nanmax(self.poses[..., 2])) + float(skeleton['z_margin_m'])
        self.box_zoom = float(layout['box_zoom'])
        self.base_line_width = float(skeleton['line_width'])
        self.base_node_size = float(skeleton['node_size'])
        self.set_box_zoom(self.box_zoom)
        # Within each moment the animals are drawn in this order, so the last one lies on top where they overlap.
        self.draw_order = sorted(range(len(self.names)), key=lambda slot: self.sex_of[self.names[slot]] == settings['camera']['top_sex'])
        self.trail_ground = settings['surfaces']['color'] if settings['surfaces']['enabled'] else settings['layout']['background_color']
        self.message_output(f'  camera azimuth {self.azimuth:.1f}, elevation {self.elevation:.1f}; '
                            f'{self.poses.shape[0]} frames drawn; half extent {self.half_extent * 100:.1f} cm')

    def set_box_zoom(self, box_zoom: float) -> None:
        """
        Description
        -----------
        Set how much the 3D box is enlarged inside the panel, and rescale the skeleton's line width
        and node size with it so the skeleton keeps the same thickness relative to the body (the
        settings' widths apply at ``skeleton['reference_half_extent_m']`` and the settings' ``box_zoom``).

        Parameters
        ----------
        box_zoom (float)
            Enlargement factor passed to ``set_box_aspect``.

        Returns
        -------
        None
        """
        self.box_zoom = box_zoom
        self.zoom = float(self.settings['skeleton']['reference_half_extent_m']) / self.half_extent * box_zoom / float(self.settings['layout']['box_zoom'])
        self.line_width = self.base_line_width * self.zoom
        self.node_size = self.base_node_size * self.zoom ** 2

    def current_zorder(self) -> int:
        """
        Description
        -----------
        The z-order of the first artist of the current (last) frame; everything of the trail lies below it.

        Parameters
        ----------

        Returns
        -------
        zorder (int)
            Integer z-order.
        """
        return 10 * ((self.poses.shape[0] - 1) * len(self.colors) + 1)

    def add_axes(self, figure: Any, rect: tuple[float, float, float, float], rasterize_trail: bool) -> Any:
        """
        Description
        -----------
        Add the pose panel to a figure: the trail silhouettes (oldest first, newer covering older
        whichever animal they belong to) and the current poses on top, then the scale mark.

        Parameters
        ----------
        figure (matplotlib.figure.Figure)
            Figure to draw into.
        rect (tuple[float, float, float, float])
            Axes rectangle in figure fractions.
        rasterize_trail (bool)
            Rasterise the trail into one image and keep the current poses as vector shapes (for SVG output).

        Returns
        -------
        ax_pose (matplotlib.axes.Axes)
            The 3D axes.
        """
        skeleton = self.settings['skeleton']
        ax_pose = figure.add_axes(rect, projection='3d')
        ax_pose.view_init(elev=self.elevation, azim=self.azimuth, roll=0)
        ax_pose.set_proj_type('ortho')
        ax_pose.set_box_aspect((2 * self.half_extent, 2 * self.half_extent, self.z_top), zoom=self.box_zoom)
        ax_pose.set_facecolor(self.settings['layout']['background_color'])
        ax_pose.computed_zorder = False
        n_frames = self.poses.shape[0]
        for index in range(n_frames):
            is_current = index == n_frames - 1
            for rank, slot in enumerate(self.draw_order):
                color = self.colors[slot]
                shade = color if is_current else blend_colors(color, self.trail_ground, float(self.strengths[index]))
                n_lines, n_collections = len(ax_pose.lines), len(ax_pose.collections)
                plot_mouse_data(data=self.poses[index:index + 1, slot:slot + 1], plot_axes=ax_pose, frame_number=0, animal_node_names=self.nodes,
                                animal_color=[shade], animal_cm=[None], animal_line_width=self.line_width,
                                node_connections=list(MOUSE_NODE_CONNECTIONS), node_polygons=list(MOUSE_NODE_POLYGONS),
                                node_lw=0.0, node_size=self.node_size, node_opacity=1.0, node_edge_color=shade,
                                polygon_color=[shade], polygon_opacity=skeleton['body_opacity'] if is_current else skeleton['trail_body_opacity'],
                                body_edge_color=shade, history_frame_span=0, history_point=skeleton['history_point'], history_ls='-',
                                history_lw=0.75, xlim_=self.half_extent, ylim_=self.half_extent, zlim_=self.z_top,
                                node_bool=True, history_bool=False)
                base = 10 * (index * len(self.colors) + rank + 1)
                for artist in ax_pose.collections[n_collections:]:
                    artist.set_zorder(base + (2 if artist.__class__.__name__ == 'Path3DCollection' else 0))
                for artist in ax_pose.lines[n_lines:]:
                    artist.set_zorder(base + 1)
        if float(self.settings['scale_mark']['length_cm']) > 0:
            self.add_scale_mark(ax_pose)
        if rasterize_trail:
            ax_pose.patch.set_visible(False)
            ax_pose.set_rasterization_zorder(self.current_zorder())
        return ax_pose

    def add_scale_mark(self, ax_pose: Any) -> None:
        """
        Description
        -----------
        Three perpendicular arms of equal length from one point on the floor, along the arena's axes
        (each optionally reversed, and the two floor arms optionally turned about the vertical), in
        the panel's own projection, set beside the male on the picture's right; labelled at the tips
        and captioned with the length underneath.

        Parameters
        ----------
        ax_pose (matplotlib.axes.Axes)
            The 3D pose axes.

        Returns
        -------
        None
        """
        mark = self.settings['scale_mark']
        right, up = screen_axes(self.azimuth, self.elevation)
        male_slot = next(i for i, name in enumerate(self.names) if self.sex_of[name] == 'male')
        male_pose = self.poses[-1, male_slot]
        anchor = np.append(np.nanmean(male_pose[:, :2], axis=0), 0.0)
        reach = float(np.nanmax((male_pose - anchor) @ right))
        arm = float(mark['length_cm']) / 100.0
        turn = -np.radians(float(mark['turn_degrees']))
        spin = np.array([[np.cos(turn), -np.sin(turn), 0.0], [np.sin(turn), np.cos(turn), 0.0], [0.0, 0.0, 1.0]])
        arms = (np.eye(3) * np.asarray(mark['signs'], dtype=float)[:, None]) @ spin.T
        screen = [np.array([axis @ right, axis @ up]) for axis in arms]
        origin = anchor + right * (reach + float(mark['gap_cm']) / 100.0 + arm * max(0.0, -min(vector[0] for vector in screen)))
        label_gap = float(mark['label_gap_fraction']) * self.half_extent
        top = self.current_zorder() + 10 * len(self.colors) + 10
        lowest = 0.0
        for index, letter in enumerate(mark['labels']):
            tip = origin + arms[index] * arm
            ax_pose.plot(*np.stack([origin, tip]).T, color=mark['color'], linewidth=float(mark['line_width']) * self.zoom,
                         solid_capstyle='round', zorder=top)
            direction = screen[index] / max(np.linalg.norm(screen[index]), np.finfo(float).eps)
            push = np.zeros(2)
            for other in range(3):
                if other != index and np.linalg.norm(screen[other]) > np.finfo(float).eps:
                    other_direction = screen[other] / np.linalg.norm(screen[other])
                    if direction @ other_direction > np.cos(np.radians(float(mark['label_separation_degrees']))):
                        side = direction - other_direction
                        push += side / max(np.linalg.norm(side), np.finfo(float).eps)
            spot = screen[index] * arm + (direction + push) * label_gap
            label_at = origin + right * spot[0] + up * spot[1]
            ax_pose.text(*label_at, letter, color=mark['color'], fontsize=mark['font_size'], ha='center', va='center', zorder=top + 1)
            lowest = min(lowest, float(spot[1]))
        caption_at = origin + up * (lowest - float(mark['caption_gap_factor']) * label_gap)
        ax_pose.text(*caption_at, f"{float(mark['length_cm']):g} cm", color=mark['color'], fontsize=mark['font_size'],
                     ha='center', va='center', zorder=top + 1)

    def add_surfaces(self, ax_pose: Any, clip_box: TransformedBbox | None) -> None:
        """
        Description
        -----------
        The arena floor and the wall nearest the animals as flat surfaces behind everything, and the
        line where they meet; the line is drawn above the trail and below the current poses so faded
        silhouettes (opaque shapes in near-surface colours) cannot erase it.

        Parameters
        ----------
        ax_pose (matplotlib.axes.Axes)
            The 3D pose axes.
        clip_box (TransformedBbox | None)
            Region the surfaces are confined to when the axes is larger than the panel; None leaves
            them clipped by the axes alone.

        Returns
        -------
        None
        """
        if self.arena is None:
            message = 'Surfaces were requested but no arena geometry was loaded.'
            raise ValueError(message)
        surfaces = self.settings['surfaces']
        limits = (ax_pose.get_xlim3d(), ax_pose.get_ylim3d(), ax_pose.get_zlim3d())
        low, high = self.arena['low'] - self.centre, self.arena['high'] - self.centre
        floor_corners = np.array([[low[0], low[1], 0.0], [high[0], low[1], 0.0], [high[0], high[1], 0.0], [low[0], high[1], 0.0]])
        if self.arena['wall_axis'] == 'y':
            at = self.arena['wall_at'] - self.centre[1]
            base = np.array([[low[0], at, 0.0], [high[0], at, 0.0]])
        else:
            at = self.arena['wall_at'] - self.centre[0]
            base = np.array([[at, low[1], 0.0], [at, high[1], 0.0]])
        wall_corners = np.vstack([base, base[::-1] + np.array([0.0, 0.0, float(surfaces['wall_height_m'])])])
        artists: list[Any] = []
        for order, corner_set in enumerate((floor_corners, wall_corners)):
            surface = Poly3DCollection([corner_set], facecolors=surfaces['color'], edgecolors='none', zorder=1 + order)
            ax_pose.add_collection3d(surface)
            artists.append(surface)
        artists.extend(ax_pose.plot(*base.T, color=surfaces['junction_color'], linewidth=float(surfaces['junction_width']) * self.zoom,
                                    solid_capstyle='butt', zorder=self.current_zorder() - 5))
        ax_pose.set_xlim3d(limits[0])
        ax_pose.set_ylim3d(limits[1])
        ax_pose.set_zlim3d(limits[2])
        if clip_box is not None:
            for artist in artists:
                artist.set_clip_box(clip_box)


def arena_geometry(arena_directory: str, pose: np.ndarray, nodes: list[str]) -> dict[str, Any]:
    """
    Description
    -----------
    The arena's floor extent and the wall nearest the animals' noses at a frame.

    Parameters
    ----------
    arena_directory (str)
        The arena session directory (``make_behavioral_videos.arena_directory``).
    pose (np.ndarray)
        (animals, nodes, 3) of the frame.
    nodes (list[str])
        Node names.

    Returns
    -------
    arena (dict)
        ``low`` / ``high`` (floor corners' min / max x, y), ``wall_axis`` (``'x'`` or ``'y'``), ``wall_at``
        (that coordinate of the nearest wall), ``gap_m`` (distance from the noses to it).
    """
    _, arena_tracks, arena_nodes = load_arena_tracks(arena_directory)
    corners = np.stack([arena_tracks[0, 0, arena_nodes.index(name), :2] for name in ('North', 'West', 'South', 'East')])
    low, high = corners.min(axis=0), corners.max(axis=0)
    nose = np.nanmean(pose[:, nodes.index('Nose'), :2], axis=0)
    gaps = {('x', float(low[0])): nose[0] - low[0], ('x', float(high[0])): high[0] - nose[0],
            ('y', float(low[1])): nose[1] - low[1], ('y', float(high[1])): high[1] - nose[1]}
    wall_axis, wall_at = min(gaps, key=lambda key: gaps[key])
    return {'low': low, 'high': high, 'wall_axis': wall_axis, 'wall_at': wall_at, 'gap_m': float(gaps[(wall_axis, wall_at)])}


class VocalPoseStillMaker:
    """
    Description
    -----------
    Renders the vocal-pose still of one session: the emitter-coloured spectrogram of the window
    ending at ``window.end_time`` above both animals' pose at that frame with their fading trails.

    Parameters
    ----------
    root_directory (str)
        Session directory.
    visualizations_parameter_dict (dict | None)
        The visualizations settings; loaded from the shipped JSON when None.
    message_output (Callable | None)
        Message sink; ``print`` when None.

    Returns
    -------
    None
    """

    def __init__(self, root_directory: str, visualizations_parameter_dict: dict[str, Any] | None = None, message_output: Callable[..., Any] | None = None) -> None:
        self.root_directory = pathlib.Path(root_directory)
        if visualizations_parameter_dict is None:
            with (pathlib.Path(__file__).parent.parent / '_parameter_settings' / 'visualizations_settings.json').open('r') as settings_file:
                visualizations_parameter_dict = json.load(settings_file)
        self.visualizations_parameter_dict = visualizations_parameter_dict
        self.settings = visualizations_parameter_dict['vocal_pose_figures']
        self.message_output = print if message_output is None else message_output

    def make_vocal_pose_still(self) -> dict[str, Any]:
        """
        Description
        -----------
        Build and save the still (PNG, and SVG when ``output.save_svg``) in ``figures.save_directory``,
        named ``vocal_pose_<session>_<start>-<end>s``.

        Parameters
        ----------

        Returns
        -------
        paths (dict)
            ``png`` and, when written, ``svg`` (pathlib.Path each).
        """
        apply_plot_style()
        settings, layout, output = self.settings, self.settings['layout'], self.settings['output']
        window, spectrogram_settings = settings['window'], settings['spectrogram']
        end_seconds = float(window['end_time'])
        history_seconds = float(window['history_seconds'])
        dpi = int(output['dpi'])
        session = load_session_poses(self.root_directory, self.visualizations_parameter_dict)
        calls = read_session_calls(self.root_directory, self.message_output)
        sex_color = {'male': self.visualizations_parameter_dict['male_colors'][0], 'female': self.visualizations_parameter_dict['female_colors'][0]}
        none_color = self.visualizations_parameter_dict['unassigned_colors'][0]
        window_calls = calls.filter((pls.col('stop') >= end_seconds - history_seconds) & (pls.col('start') <= end_seconds))
        microphone = peak_microphone(window_calls) if spectrogram_settings['microphone'] == 'peak' else str(spectrogram_settings['microphone'])
        image, band, _ = emitter_spectrogram(self.root_directory, calls, session['sex_of'], sex_color, none_color, end_seconds,
                                             history_seconds, microphone, spectrogram_settings, self.message_output)
        last_pose = session['tracks'][int(np.floor(end_seconds * session['fps']))]
        arena = None
        if settings['surfaces']['enabled']:
            arena = arena_geometry(self.visualizations_parameter_dict['make_behavioral_videos']['arena_directory'], last_pose, session['nodes'])
            self.message_output(f"  surfaces: wall at {arena['wall_axis']} = {arena['wall_at'] * 100:.1f} cm, {arena['gap_m'] * 100:.1f} cm from the noses")
        panel = PosePanel(session, settings, end_seconds, arena, self.message_output)
        width_in = float(layout['figure_width_inches'])
        background = layout['background_color']

        # Render the square pose panel on its own; if anything drawn touches its edge, shrink the box and render again.
        # An explicit Agg canvas, so the panel's pixel size is the figure's dpi whatever the session's backend.
        for _ in range(int(layout['max_fit_passes'])):
            pose_fig = Figure(figsize=(width_in, width_in), dpi=dpi, facecolor=background)
            pose_canvas = FigureCanvasAgg(pose_fig)
            pose_axes = panel.add_axes(pose_fig, (0.0, 0.0, 1.0, 1.0), False)
            pose_canvas.draw()
            drawn = (np.asarray(pose_canvas.buffer_rgba())[..., :3] < layout['content_threshold']).any(axis=2)
            if not (drawn[:2].any() or drawn[-2:].any() or drawn[:, :2].any() or drawn[:, -2:].any()):
                break
            panel.set_box_zoom(panel.box_zoom * float(layout['fit_shrink']))
        self.message_output(f'  box zoom {panel.box_zoom:.2f}; line width {panel.line_width:.1f} pt; node size {panel.node_size:.0f}')
        pose_image = np.asarray(pose_canvas.buffer_rgba())[..., :3].copy()
        # Content is whatever differs visibly from the page and, with surfaces, from the colour the trail fades into.
        ground = np.round(hex_to_rgb(panel.trail_ground) * 255)
        tolerance = float(settings['surfaces']['content_tolerance']) if settings['surfaces']['enabled'] else 0.0
        visible = (pose_image < layout['content_threshold']).any(axis=2) & (np.abs(pose_image.astype(float) - ground) > tolerance).any(axis=2)
        content_rows = np.where(visible.any(axis=1))[0]
        if settings['surfaces']['enabled']:
            panel.add_surfaces(pose_axes, None)
            pose_canvas.draw()
            pose_image = np.asarray(pose_canvas.buffer_rgba())[..., :3].copy()
        pad_inches = float(settings['surfaces']['pad_inches'] if settings['surfaces']['enabled'] else layout['pose_pad_inches'])
        pad = round(pad_inches * dpi)
        pose_side = pose_image.shape[0]
        pose_row0, pose_row1 = max(int(content_rows[0]) - pad, 0), min(int(content_rows[-1]) + pad, pose_side)
        pose_image = pose_image[pose_row0:pose_row1]
        pose_inches = pose_image.shape[0] / (pose_image.shape[1] / width_in)
        spec_left, spec_width = float(layout['spectrogram_left']), float(layout['spectrogram_width'])
        drawn_columns = np.where((pose_image < layout['content_threshold']).any(axis=(0, 2)))[0]
        pose_shift = 0.0
        if layout['align_pose']:
            # Everything in the pose panel moves left so that its right edge sits under the spectrogram's right edge.
            pose_shift = (drawn_columns[-1] + 1) / pose_image.shape[1] - (spec_left + spec_width)
        key_rows = self._key_rows(pose_image, pose_shift) if layout['color_key'] and layout['key_position'] == 'pose' else None
        top_block = float(layout['top_block_inches'])
        fig_height = top_block + pose_inches

        def build_figure(vector_pose: bool) -> Any:
            """
            Description
            -----------
            Assemble the composite: spectrogram, time bar, colour key and pose panel.

            Parameters
            ----------
            vector_pose (bool)
                Draw the pose panel as live 3D axes (vector current poses, rasterised trail) instead
                of pasting its pre-rendered image.

            Returns
            -------
            fig (matplotlib.figure.Figure)
                The assembled figure.
            """
            fig = plt.figure(figsize=(width_in, fig_height), dpi=dpi, facecolor=background)
            inch = 1.0 / fig_height
            spec_top, spec_height, bar_height, bar_gap = (float(layout[key]) for key in ('spectrogram_top_inches', 'spectrogram_height_inches',
                                                                                          'bar_height_inches', 'bar_gap_inches'))
            ax_spec = fig.add_axes((spec_left, 1.0 - (spec_top + spec_height) * inch, spec_width, spec_height * inch))
            ax_bar = fig.add_axes((spec_left, 1.0 - (spec_top + spec_height + bar_gap + bar_height) * inch, spec_width, bar_height * inch))
            if vector_pose:
                vector_axes = panel.add_axes(fig, (-pose_shift, -(pose_side - pose_row1) / dpi * inch, 1.0, width_in * inch), True)
                if settings['surfaces']['enabled']:
                    panel.add_surfaces(vector_axes, TransformedBbox(Bbox([[0.0, 0.0], [1.0, pose_inches * inch]]), fig.transFigure))
            else:
                ax_image = fig.add_axes((-pose_shift, 0.0, 1.0, pose_inches * inch))
                ax_image.imshow(pose_image, aspect='auto', interpolation='none')
                ax_image.axis('off')
            ax_spec.imshow(image, origin='lower', aspect='auto', extent=(-history_seconds, 0.0, band[0], band[-1]), interpolation='antialiased')
            ax_spec.set_ylim(*spectrogram_settings['freq_range_khz'])
            ax_spec.set_yticks(list(spectrogram_settings['freq_ticks_khz']))
            ax_spec.set_yticklabels([f'{value:g}' for value in spectrogram_settings['freq_ticks_khz']])
            ax_spec.tick_params(axis='y', length=0, labelsize=layout['tick_font_size'], pad=layout['tick_pad'], colors=layout['ink_color'])
            ax_spec.set_xticks([])
            ax_spec.set_ylabel(layout['frequency_label'], fontsize=layout['label_font_size'], color=layout['ink_color'])
            for spine in ax_spec.spines.values():
                spine.set_visible(False)
            # The bar's strength at each time is the trail's own fade at that age.
            bar_ages = np.linspace(history_seconds, 0.0, int(layout['bar_samples']))
            ramp = trail_strength(bar_ages, settings['trail'], panel.trail_end_seconds)
            white = np.ones(3)
            mid_rgb = 0.5 * (hex_to_rgb(sex_color['male']) + hex_to_rgb(sex_color['female']))
            bar = white[None, None, :] + (mid_rgb[None, None, :] - white[None, None, :]) * ramp[None, :, None]
            ax_bar.imshow(bar, aspect='auto', extent=(-history_seconds, 0.0, 0.0, 1.0))
            ax_bar.set_xticks([])
            ax_bar.set_yticks([])
            for spine in ax_bar.spines.values():
                spine.set_visible(False)
            label_y = -float(layout['bar_label_offset'])
            ax_bar.text(0.0, label_y, f'\u2212{history_seconds:g}', transform=ax_bar.transAxes, ha='left', va='top',
                        fontsize=layout['tick_font_size'], color=layout['ink_color'])
            ax_bar.text(1.0, label_y, '0', transform=ax_bar.transAxes, ha='right', va='top', fontsize=layout['tick_font_size'], color=layout['ink_color'])
            ax_bar.text(0.5, label_y, layout['time_label'], transform=ax_bar.transAxes, ha='center', va='top',
                        fontsize=layout['label_font_size'], color=layout['ink_color'])
            if layout['color_key']:
                side = float(layout['key_swatch_inches'])
                if layout['key_position'] == 'pose' and key_rows is not None:
                    left, wide = spec_left, float(layout['key_wide_inches'])
                    middle = (pose_inches - key_rows / dpi - side - 0.5 * float(layout['key_spacing_inches'])) * inch
                else:
                    left, wide = spec_left + spec_width + float(layout['key_gap_inches']) / width_in, side
                    middle = 1.0 - (spec_top + 0.5 * spec_height) * inch
                for row, sex in enumerate(('male', 'female')):
                    bottom = middle + (0.5 * float(layout['key_spacing_inches']) if row == 0 else -0.5 * float(layout['key_spacing_inches']) - side) * inch
                    fig.add_artist(Rectangle((left, bottom), wide / width_in, side * inch, transform=fig.transFigure,
                                             facecolor=sex_color[sex], edgecolor='none'))
            return fig

        stem = f'vocal_pose_{self.root_directory.name}_{end_seconds - history_seconds:.2f}-{end_seconds:.2f}s'
        png_path, _, _ = resolve_save_path(stem=stem, viz_settings=self.visualizations_parameter_dict, override_format='png', override_dpi=dpi)
        fig = build_figure(False)
        # The whole figure, exactly: pixel positions in this image are then figure positions, which the SVG crop relies on.
        fig.savefig(png_path, dpi=dpi, facecolor=background, bbox_inches=Bbox([[0.0, 0.0], [width_in, fig_height]]), pad_inches=0.0)
        plt.close(fig)
        saved = Image.open(png_path).convert('RGB')
        pixels = np.asarray(saved)
        rows = np.where((pixels < layout['content_threshold']).any(axis=(1, 2)))[0]
        cols = np.where((pixels < layout['content_threshold']).any(axis=(0, 2)))[0]
        edge = int(layout['crop_edge_pixels'])
        box = (max(int(cols[0]) - edge, 0), max(int(rows[0]) - edge, 0),
               min(int(cols[-1]) + 1 + edge, saved.width), min(int(rows[-1]) + 1 + edge, saved.height))
        saved.crop(box).save(png_path)
        paths = {'png': png_path}
        self.message_output(f'  still written to {png_path}')
        if output['save_svg']:
            svg_path, _, _ = resolve_save_path(stem=stem, viz_settings=self.visualizations_parameter_dict, override_format='svg', override_dpi=dpi)
            svg_pad = float(layout['svg_pad_inches'])
            svg_fig = build_figure(True)
            svg_fig.savefig(svg_path, dpi=dpi, facecolor=background, pad_inches=0.0,
                            bbox_inches=Bbox([[box[0] / dpi - svg_pad, (saved.height - box[3]) / dpi - svg_pad],
                                              [box[2] / dpi + svg_pad, (saved.height - box[1]) / dpi + svg_pad]]))
            plt.close(svg_fig)
            paths['svg'] = svg_path
            self.message_output(f'  svg written to {svg_path}')
        return paths

    def _key_rows(self, pose_image: np.ndarray, pose_shift: float) -> int:
        """
        Description
        -----------
        The top row (in the cropped pose image) of the colour key when it sits beside the animals: the
        key goes at the spectrogram's left edge, in the middle of the tallest band of rows that nothing
        drawn enters there.

        Parameters
        ----------
        pose_image (np.ndarray)
            The cropped pose panel image.
        pose_shift (float)
            How far the panel is moved left (figure fraction).

        Returns
        -------
        key_rows (int)
            Row index of the key block's top.
        """
        layout, dpi = self.settings['layout'], int(self.settings['output']['dpi'])
        width_in = float(layout['figure_width_inches'])
        key_width = float(layout['key_wide_inches']) / width_in
        clear = round(float(layout['key_clear_inches']) * pose_image.shape[1] / width_in)
        first_column = int(max((float(layout['spectrogram_left']) + pose_shift) * pose_image.shape[1] - clear, 0))
        last_column = int((float(layout['spectrogram_left']) + key_width + pose_shift) * pose_image.shape[1] + clear)
        busy = (pose_image[:, first_column:last_column] < layout['content_threshold']).any(axis=(1, 2))
        best, start = (0, 0), None
        for row, flag in enumerate(np.append(busy, True)):
            if not flag and start is None:
                start = row
            elif flag and start is not None:
                best, start = max(best, (row - start, start)), None
        block = round((2 * float(layout['key_swatch_inches']) + float(layout['key_spacing_inches'])) * dpi)
        if best[0] < block:
            message = 'No free space for the colour key to the left of the animals; set layout.key_position to "spectrogram".'
            raise RuntimeError(message)
        return best[1] + (best[0] - block) // 2


def render_video_frames(job: dict[str, Any]) -> tuple[tuple[int, int, int, int], np.ndarray]:
    """
    Description
    -----------
    Worker for :class:`VocalPoseVideoMaker`: render a subset of the video's frames to PNG files, each
    with the animals' current poses over their short fading trails, and report where on the square
    canvas anything was drawn.

    Parameters
    ----------
    job (dict)
        ``frames`` (clip frame indices to render), ``poses`` (n, animals, nodes, 3; centred; the first
        ``n_trail`` rows precede the clip), ``targets`` (per clip frame, the floor point the camera is
        centred on), ``n_trail``, ``fps``, ``azimuths`` (per clip frame), ``elevation``, ``colors``,
        ``nodes``, ``draw_order``, ``half_extent``, ``z_top``, ``shades`` (per animal, per trail step,
        the silhouette colour), ``skeleton`` (the video's skeleton keys), ``inches``, ``dpi``,
        ``box_zoom``, ``background_color``, ``content_threshold``, ``frame_dir``.

    Returns
    -------
    bounds, union (tuple[tuple[int, int, int, int], np.ndarray])
        (first row, last row, first column, last column) of the drawn pixels over the worker's frames,
        and the union of those pixels as a boolean image.
    """
    poses, n_trail, colors, skeleton = job['poses'], job['n_trail'], job['colors'], job['skeleton']
    fig = Figure(figsize=(job['inches'], job['inches']), dpi=job['dpi'], facecolor=job['background_color'])
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0), projection='3d')
    bounds = [10 ** 9, -1, 10 ** 9, -1]
    union: np.ndarray | None = None
    for frame in job['frames']:
        ax.cla()
        ax.view_init(elev=job['elevation'], azim=float(job['azimuths'][frame]), roll=0)
        ax.set_proj_type('ortho')
        ax.set_box_aspect((2 * job['half_extent'], 2 * job['half_extent'], job['z_top']), zoom=job['box_zoom'])
        ax.set_facecolor(job['background_color'])
        ax.computed_zorder = False
        view = poses[frame:frame + n_trail + 1].copy()
        view[..., :2] -= job['targets'][frame]
        for step in range(n_trail + 1):
            is_current = step == n_trail
            for rank, slot in enumerate(job['draw_order']):
                shade = job['shades'][slot][step]
                n_lines, n_collections = len(ax.lines), len(ax.collections)
                plot_mouse_data(data=view[step:step + 1, slot:slot + 1], plot_axes=ax, frame_number=0, animal_node_names=job['nodes'],
                                animal_color=[shade], animal_cm=[None], animal_line_width=skeleton['line_width'],
                                node_connections=list(MOUSE_NODE_CONNECTIONS), node_polygons=list(MOUSE_NODE_POLYGONS),
                                node_lw=0.0, node_size=skeleton['node_size'], node_opacity=1.0, node_edge_color=shade,
                                polygon_color=[shade], polygon_opacity=skeleton['body_opacity'] if is_current else skeleton['trail_body_opacity'],
                                body_edge_color=shade, history_frame_span=0, history_point=skeleton['history_point'], history_ls='-',
                                history_lw=0.75, xlim_=job['half_extent'], ylim_=job['half_extent'], zlim_=job['z_top'],
                                node_bool=True, history_bool=False)
                base = 10 * (step * len(colors) + rank + 1)
                for artist in ax.collections[n_collections:]:
                    artist.set_zorder(base + (2 if artist.__class__.__name__ == 'Path3DCollection' else 0))
                for artist in ax.lines[n_lines:]:
                    artist.set_zorder(base + 1)
        canvas.draw()
        pixels = np.asarray(canvas.buffer_rgba())[..., :3]
        Image.fromarray(pixels).save(pathlib.Path(job['frame_dir']) / f'frame_{frame:05d}.png')
        content = (pixels < job['content_threshold']).any(axis=2)
        union = content if union is None else union | content
        rows = np.where(content.any(axis=1))[0]
        cols = np.where(content.any(axis=0))[0]
        bounds = [min(bounds[0], int(rows[0])), max(bounds[1], int(rows[-1])), min(bounds[2], int(cols[0])), max(bounds[3], int(cols[-1]))]
    if union is None:
        union = np.zeros((int(job['inches'] * job['dpi']), int(job['inches'] * job['dpi'])), dtype=bool)
    return (bounds[0], bounds[1], bounds[2], bounds[3]), union


class VocalPoseVideoMaker:
    """
    Description
    -----------
    Renders the vocal-pose video of one session: the window ending at ``window.end_time``, played at
    ``video.speed`` with a short fading trail behind each animal, a fixed, following or turning
    camera, and a scrolling spectrogram that is sharp only around the present. The clip ends on the
    still's frame and camera, so the two match.

    Parameters
    ----------
    root_directory (str)
        Session directory.
    visualizations_parameter_dict (dict | None)
        The visualizations settings; loaded from the shipped JSON when None.
    message_output (Callable | None)
        Message sink; ``print`` when None.

    Returns
    -------
    None
    """

    def __init__(self, root_directory: str, visualizations_parameter_dict: dict[str, Any] | None = None, message_output: Callable[..., Any] | None = None) -> None:
        self.root_directory = pathlib.Path(root_directory)
        if visualizations_parameter_dict is None:
            with (pathlib.Path(__file__).parent.parent / '_parameter_settings' / 'visualizations_settings.json').open('r') as settings_file:
                visualizations_parameter_dict = json.load(settings_file)
        self.visualizations_parameter_dict = visualizations_parameter_dict
        self.settings = visualizations_parameter_dict['vocal_pose_figures']
        self.message_output = print if message_output is None else message_output

    def make_vocal_pose_video(self) -> pathlib.Path:
        """
        Description
        -----------
        Render the frames in parallel worker processes, crop them to the union of their content
        (symmetrically about the camera target when the camera follows the pair), add the scrolling
        spectrogram, and encode the clip with ffmpeg into ``figures.save_directory`` as
        ``vocal_pose_<session>_<start>-<end>s.mp4``.

        Parameters
        ----------

        Returns
        -------
        mp4_path (pathlib.Path)
            The written clip.
        """
        apply_plot_style()
        settings, video, layout = self.settings, self.settings['video'], self.settings['layout']
        trail, skeleton, camera = video['trail'], video['skeleton'], settings['camera']
        session = load_session_poses(self.root_directory, self.visualizations_parameter_dict)
        fps, names, nodes = session['fps'], session['names'], session['nodes']
        end_seconds = float(settings['window']['end_time'])
        last = int(np.floor(end_seconds * fps))
        n_frames = round(float(video['duration_seconds']) * fps)
        n_trail = round(float(trail['trail_seconds']) * fps)
        first = last - n_frames + 1
        poses = session['tracks'][first - n_trail:last + 1]
        end_azimuth = resolve_azimuth(camera, session['tracks'][last], names, nodes, session['sex_of'])
        elevation = float(camera['elevation'])
        colors = session['colors']
        draw_order = sorted(range(len(names)), key=lambda slot: session['sex_of'][names[slot]] == camera['top_sex'])
        window = poses[n_trail - 1:]
        centre = 0.5 * (np.nanmin(window[..., :2].reshape(-1, 2), axis=0) + np.nanmax(window[..., :2].reshape(-1, 2), axis=0))
        centred = poses.copy()
        centred[..., 0] -= centre[0]
        centred[..., 1] -= centre[1]
        follow_seconds = float(video['follow_seconds'])
        targets = np.zeros((n_frames, 2))
        if follow_seconds >= 0:
            # Camera target: the centre of the box around every keypoint of both animals, smoothed in time so the view glides.
            outline = centred[n_trail:, :, :, :2].reshape(n_frames, -1, 2)
            targets = 0.5 * (np.nanmin(outline, axis=1) + np.nanmax(outline, axis=1))
            if follow_seconds > 0:
                targets = gaussian_filter1d(targets, sigma=follow_seconds * fps, axis=0, mode='nearest')
        half_extent = max(float(np.nanmax(np.hypot(centred[i:i + n_trail + 1, ..., 0] - targets[i, 0], centred[i:i + n_trail + 1, ..., 1] - targets[i, 1])))
                          for i in range(n_frames)) + float(skeleton['margin_m'])
        z_top = float(np.nanmax(centred[..., 2])) + float(skeleton['z_margin_m'])
        azimuths = end_azimuth + float(video['degrees_per_second']) * (n_frames - 1 - np.arange(n_frames)) / fps
        ages = np.arange(n_trail, 0, -1) / fps
        strengths = trail_strength(ages, trail, 0.0)
        shades = [[blend_colors(color, layout['background_color'], float(strength)) for strength in strengths] + [color] for color in colors]
        stem = f'vocal_pose_{self.root_directory.name}_{(first / fps):.2f}-{end_seconds:.2f}s'
        mp4_path, _, _ = resolve_save_path(stem=stem, viz_settings=self.visualizations_parameter_dict, override_format='mp4')
        frame_dir = mp4_path.parent / f'{mp4_path.stem}_frames'
        frame_dir.mkdir(parents=True, exist_ok=True)
        dpi, inches = int(video['dpi']), float(video['inches'])
        job = {'poses': centred, 'targets': targets, 'n_trail': n_trail, 'fps': fps, 'azimuths': azimuths, 'elevation': elevation,
               'colors': colors, 'nodes': nodes, 'draw_order': draw_order, 'half_extent': half_extent, 'z_top': z_top, 'shades': shades,
               'skeleton': skeleton, 'inches': inches, 'dpi': dpi, 'box_zoom': float(layout['box_zoom']),
               'background_color': layout['background_color'], 'content_threshold': layout['content_threshold'], 'frame_dir': str(frame_dir)}
        self.message_output(f'  frames {first}-{last} ({n_frames}), trail {n_trail} frames, azimuth {azimuths[0]:.1f} -> {azimuths[-1]:.1f}, '
                            f'half extent {half_extent * 100:.1f} cm')
        n_workers = int(video['workers']) if int(video['workers']) > 0 else max((os.cpu_count() or 1) - 1, 1)
        chunks = [list(range(n_frames))[i::n_workers] for i in range(n_workers)]
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            results = list(pool.map(render_video_frames, [{**job, 'frames': chunk} for chunk in chunks]))
        side = round(inches * dpi)
        edge = int(video['crop_edge_pixels'])
        row0 = max(min(result[0][0] for result in results) - edge, 0)
        row1 = min(max(result[0][1] for result in results) + edge, side - 1)
        col0 = max(min(result[0][2] for result in results) - edge, 0)
        col1 = min(max(result[0][3] for result in results) + edge, side - 1)
        if follow_seconds >= 0:
            # The camera target projects to the middle of the square; crop symmetrically about it so the pair stays centred.
            half_width = max(side // 2 - col0, col1 - side // 2)
            half_height = max(side // 2 - row0, row1 - side // 2)
            col0, col1 = max(side // 2 - half_width, 0), min(side // 2 + half_width, side - 1)
            row0, row1 = max(side // 2 - half_height, 0), min(side // 2 + half_height, side - 1)
        width = (col1 - col0 + 1) // 2 * 2
        height = (row1 - row0 + 1) // 2 * 2
        occupied = np.logical_or.reduce([result[1] for result in results])[row0:row0 + height, col0:col0 + width]
        self.message_output(f'  rendered {n_frames} frames; crop {width}x{height} of {side}x{side}')
        playback_fps = fps * float(video['speed'])
        encode = ['ffmpeg', '-loglevel', 'error', '-y']
        if not video['spectrogram']['enabled']:
            subprocess.run([*encode, '-framerate', f'{playback_fps:.6f}', '-i', str(frame_dir / 'frame_%05d.png'),
                            '-vf', f'crop={width}:{height}:{col0}:{row0}', '-c:v', video['codec'], '-pix_fmt', video['pixel_format'],
                            '-crf', str(video['crf']), str(mp4_path)], check=True, timeout=float(video['encode_timeout_seconds']))
        else:
            self._encode_with_spectrogram(session, first, last, n_frames, frame_dir, mp4_path, (row0, col0, width, height), occupied, playback_fps)
        shutil.rmtree(frame_dir)
        self.message_output(f'  video written to {mp4_path}')
        return mp4_path

    def _encode_with_spectrogram(self, session: dict[str, Any], first: int, last: int, n_frames: int, frame_dir: pathlib.Path, mp4_path: pathlib.Path,
                                 crop: tuple[int, int, int, int], occupied: np.ndarray, playback_fps: float) -> None:
        """
        Description
        -----------
        Compose each rendered frame with a slice of the window's spectrogram, the present fixed at the
        slice's centre and only ``reach_seconds`` visible on each side, sharp within ``sharp_seconds``
        and fading toward white beyond, as an inset in a corner no animal ever enters (the lower right
        when free, then the roomiest corner, slid left by ``inset_shift`` of the free room) or as a
        strip below the animals; then pipe the frames to ffmpeg.

        Parameters
        ----------
        session (dict)
            From :func:`load_session_poses`.
        first (int)
            First clip frame.
        last (int)
            Last clip frame.
        n_frames (int)
            Number of clip frames.
        frame_dir (pathlib.Path)
            Directory of the rendered frames.
        mp4_path (pathlib.Path)
            Output clip.
        crop (tuple[int, int, int, int])
            (row0, col0, width, height) of the crop applied to every rendered frame.
        occupied (np.ndarray)
            Boolean image of the cropped frame: where any animal or trail ever is.
        playback_fps (float)
            Frame rate of the clip.

        Returns
        -------
        None
        """
        settings, video, layout = self.settings, self.settings['video'], self.settings['layout']
        spec, spectrogram_settings = video['spectrogram'], settings['spectrogram']
        row0, col0, width, height = crop
        fps = session['fps']
        reach = float(spec['reach_seconds'])
        spec_end = last / fps + reach
        calls = read_session_calls(self.root_directory, self.message_output)
        sex_color = {'male': self.visualizations_parameter_dict['male_colors'][0], 'female': self.visualizations_parameter_dict['female_colors'][0]}
        none_color = self.visualizations_parameter_dict['unassigned_colors'][0]
        span = (last - first) / fps + 2 * reach
        window_calls = calls.filter((pls.col('stop') >= spec_end - span) & (pls.col('start') <= spec_end))
        microphone = peak_microphone(window_calls) if spectrogram_settings['microphone'] == 'peak' else str(spectrogram_settings['microphone'])
        image, _, times = emitter_spectrogram(self.root_directory, calls, session['sex_of'], sex_color, none_color, spec_end, span, microphone,
                                              spectrogram_settings, self.message_output)
        white = np.ones(3)
        n_columns = round(2 * reach / (times[1] - times[0]))
        dpi = int(video['dpi'])
        margin, gap = int(spec['inset_margin_pixels']), int(spec['inset_gap_pixels'])
        panel_left_pixels, panel_right_pixels, panel_edge_pixels = (int(spec[key]) for key in ('panel_left_pixels', 'panel_right_pixels', 'panel_edge_pixels'))
        if spec['layout'] == 'below':
            panel_width, panel_height, panel_top, panel_left = width, round(float(spec['height_fraction']) * width) // 2 * 2, height, 0
        else:
            fits = {}
            corners = ('lower right', 'lower left', 'upper right', 'upper left') if spec['inset_corner'] == 'auto' else (spec['inset_corner'],)
            for corner in corners:
                candidate = round(float(spec['inset_max_width']) * width)
                while candidate > 1.5 * panel_left_pixels:
                    tall = round(float(spec['height_fraction']) * candidate)
                    rows = slice(max(height - margin - tall - gap, 0), height) if corner.startswith('lower') else slice(0, margin + tall + gap)
                    cols = slice(max(width - margin - candidate - gap, 0), width) if corner.endswith('right') else slice(0, margin + candidate + gap)
                    if not occupied[rows, cols].any():
                        fits[corner] = candidate
                        break
                    candidate -= 2
            if not fits:
                message = 'No free corner for the spectrogram inset; set video.spectrogram.layout to "below".'
                raise RuntimeError(message)
            corner = max(fits, key=lambda name: (fits[name], name == 'lower right'))
            if 'lower right' in fits and fits['lower right'] >= float(spec['corner_preference']) * fits[corner]:
                corner = 'lower right'
            panel_width = fits[corner]
            panel_height = round(float(spec['height_fraction']) * panel_width)
            panel_top = height - margin - panel_height if corner.startswith('lower') else margin
            panel_left = width - margin - panel_width if corner.endswith('right') else margin
            if corner == 'lower right':
                band = occupied[max(height - margin - panel_height - gap, 0):]
                leftmost = panel_left
                while leftmost > 0 and not band[:, max(leftmost - 1 - gap, 0):leftmost - 1 + panel_width + gap].any():
                    leftmost -= 1
                panel_left -= round(float(spec['inset_shift']) * (panel_left - leftmost))
            self.message_output(f'  spectrogram inset {panel_width}x{panel_height} px in the {corner} at column {panel_left}; free corners {fits}')
        # The static part of the panel (y label and tick labels), drawn once; the moving slice is pasted into its axes.
        text_scale = min(1.0, panel_height / float(spec['panel_full_text_pixels']))
        panel_fig = Figure(figsize=(panel_width / dpi, panel_height / dpi), dpi=dpi, facecolor=layout['background_color'])
        panel_canvas = FigureCanvasAgg(panel_fig)
        panel_ax = panel_fig.add_axes((panel_left_pixels * text_scale / panel_width, panel_edge_pixels / panel_height,
                                       1.0 - (panel_left_pixels * text_scale + panel_right_pixels) / panel_width, 1.0 - 2 * panel_edge_pixels / panel_height))
        panel_ax.set_xlim(0, 1)
        panel_ax.set_ylim(*spectrogram_settings['freq_range_khz'])
        panel_ax.set_yticks(list(spectrogram_settings['freq_ticks_khz']))
        panel_ax.set_yticklabels([f'{value:g}' for value in spectrogram_settings['freq_ticks_khz']])
        panel_ax.tick_params(axis='y', length=0, labelsize=layout['tick_font_size'] * text_scale, pad=layout['tick_pad'], colors=layout['ink_color'])
        panel_ax.set_xticks([])
        panel_ax.set_ylabel(layout['frequency_label'], fontsize=layout['label_font_size'] * text_scale, color=layout['ink_color'])
        panel_ax.set_facecolor(layout['background_color'])
        for spine in panel_ax.spines.values():
            spine.set_visible(False)
        panel_canvas.draw()
        panel = np.asarray(panel_canvas.buffer_rgba())[..., :3].copy()
        extent = panel_ax.get_window_extent()
        # The canvas can come out a pixel off the requested size; pad with background or trim to the exact size.
        exact = np.full((panel_height, panel_width, 3), 255, dtype=np.uint8)
        exact[:min(panel_height, panel.shape[0]), :min(panel_width, panel.shape[1])] = panel[:panel_height, :panel_width]
        ax_col0, ax_col1 = round(extent.x0), round(extent.x1)
        ax_row0, ax_row1 = panel.shape[0] - round(extent.y1), panel.shape[0] - round(extent.y0)
        out_height = height + (panel_height if spec['layout'] == 'below' else 0)
        encoder = subprocess.Popen(['ffmpeg', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{width}x{out_height}',
                                    '-framerate', f'{playback_fps:.6f}', '-i', '-', '-c:v', video['codec'], '-pix_fmt', video['pixel_format'],
                                    '-crf', str(video['crf']), str(mp4_path)], stdin=subprocess.PIPE)
        if encoder.stdin is None:
            message = 'ffmpeg did not open a pipe for the frames.'
            raise RuntimeError(message)
        for frame in range(n_frames):
            now = (first + frame) / fps - spec_end
            start = int(np.clip(np.searchsorted(times, now - reach), 0, times.size - n_columns))
            lag = np.abs(times[start:start + n_columns] - now)
            weight = np.where(lag <= float(spec['sharp_seconds']), 1.0,
                              float(spec['floor']) + (1.0 - float(spec['floor'])) * np.exp(-(lag - float(spec['sharp_seconds'])) / float(spec['fade_seconds'])))
            piece = white[None, None, :] + (image[:, start:start + n_columns] - white[None, None, :]) * weight[None, :, None]
            resized = Image.fromarray((np.clip(piece[::-1], 0.0, 1.0) * 255).astype(np.uint8)).resize((ax_col1 - ax_col0, ax_row1 - ax_row0), Image.Resampling.LANCZOS)
            strip = exact.copy()
            strip[ax_row0:ax_row1, ax_col0:ax_col1] = np.asarray(resized)
            pose = np.asarray(Image.open(frame_dir / f'frame_{frame:05d}.png').convert('RGB'))[row0:row0 + height, col0:col0 + width].copy()
            if spec['layout'] == 'below':
                pose = np.vstack([pose, strip])
            else:
                pose[panel_top:panel_top + panel_height, panel_left:panel_left + panel_width] = strip
            encoder.stdin.write(pose.tobytes())
        encoder.stdin.close()
        if encoder.wait(timeout=float(video['encode_timeout_seconds'])) != 0:
            message = f'ffmpeg failed while encoding {mp4_path}.'
            raise RuntimeError(message)


def find_vocal_pose_windows(root_directory: str, visualizations_parameter_dict: dict[str, Any] | None = None,
                            message_output: Callable[..., Any] | None = None) -> pls.DataFrame:
    """
    Description
    -----------
    Rank the windows of a session for a vocal-pose figure. A window of ``candidates.window_seconds``,
    stepped by ``candidates.step_seconds``, qualifies when every call overlapping it is a non-noise,
    non-squeak USV of the male, no call is cut by either edge, no detection flagged as noise falls
    inside it, both animals are tracked on every frame, and the smallest distance between any body
    keypoints of the two animals (tails excluded) never exceeds ``candidates.max_gap_cm``. Qualifying
    windows are scored by the sum of three z-scores across windows: the fraction of the window filled
    with calling, the duration-weighted mean frequency bandwidth of its calls, and their mean duration.

    Parameters
    ----------
    root_directory (str)
        Session directory.
    visualizations_parameter_dict (dict | None)
        The visualizations settings; loaded from the shipped JSON when None.
    message_output (Callable | None)
        Message sink; ``print`` when None.

    Returns
    -------
    windows (pls.DataFrame)
        One row per qualifying window, best first: ``start``, ``end`` (s), ``n_calls``, ``vocal_fraction``,
        ``bandwidth_khz``, ``mean_duration_ms``, ``gap_median_cm``, ``gap_max_cm``, ``microphone``, ``score``.
    """
    sink: Callable[..., Any] = print if message_output is None else message_output
    root = pathlib.Path(root_directory)
    if visualizations_parameter_dict is None:
        with (pathlib.Path(__file__).parent.parent / '_parameter_settings' / 'visualizations_settings.json').open('r') as settings_file:
            visualizations_parameter_dict = json.load(settings_file)
    candidates = visualizations_parameter_dict['vocal_pose_figures']['candidates']
    window_seconds, step_seconds, max_gap_cm = (float(candidates[key]) for key in ('window_seconds', 'step_seconds', 'max_gap_cm'))
    session = load_session_poses(root, visualizations_parameter_dict)
    summary_path = first_match_or_raise(root=root / 'audio', pattern='*_usv_summary.csv', recursive=False, label='USV summary')
    all_calls = pls.read_csv(summary_path, schema_overrides={'emitter': pls.Utf8})
    noise_rows = all_calls.filter(pls.col('noise') == True)  # noqa: E712
    noise_start, noise_stop = noise_rows['start'].to_numpy(), noise_rows['stop'].to_numpy()
    calls = read_session_calls(root, sink)
    start, stop = calls['start'].to_numpy(), calls['stop'].to_numpy()
    duration, bandwidth = calls['duration'].to_numpy(), calls['freq_bandwidth_hz'].to_numpy()
    is_male_usv = np.array([(emitter in session['sex_of'] and session['sex_of'][emitter] == 'male') and not bool(squeak)
                            for emitter, squeak in zip(calls['emitter'].to_list(), calls['squeak'].to_list(), strict=True)])
    tracks, fps = session['tracks'], session['fps']
    body = [i for i, name in enumerate(session['nodes']) if not name.startswith('Tail')]
    tracked = np.isfinite(tracks).all(axis=(1, 2, 3))
    gap_cm = np.sqrt(((tracks[:, 0][:, body][:, :, None, :] - tracks[:, 1][:, body][:, None, :, :]) ** 2).sum(axis=-1)).reshape(tracks.shape[0], -1).min(axis=1) * 100.0
    rows = []
    for t0 in np.arange(0.0, tracks.shape[0] / fps - window_seconds, step_seconds):
        t1 = t0 + window_seconds
        inside = (stop > t0) & (start < t1)
        if not inside.any() or not is_male_usv[inside].all() or (start[inside] < t0).any() or (stop[inside] > t1).any():
            continue
        if ((noise_stop > t0) & (noise_start < t1)).any():
            continue
        frames = slice(int(np.floor(t0 * fps)), int(np.floor(t1 * fps)) + 1)
        if not tracked[frames].all() or gap_cm[frames].max() > max_gap_cm:
            continue
        rows.append({'start': float(t0), 'end': float(t1), 'n_calls': int(inside.sum()), 'vocal_fraction': float(duration[inside].sum() / window_seconds),
                     'bandwidth_khz': float(np.average(bandwidth[inside], weights=duration[inside]) / 1000.0),
                     'mean_duration_ms': float(duration[inside].mean() * 1000.0), 'gap_median_cm': float(np.median(gap_cm[frames])),
                     'gap_max_cm': float(gap_cm[frames].max()), 'microphone': peak_microphone(calls.filter(pls.Series(inside)))})
    sink(f'  {len(rows)} qualifying windows of {window_seconds:g} s; male USVs {int(is_male_usv.sum())} of {start.size} non-noise calls')
    windows = pls.DataFrame(rows)
    if windows.height == 0:
        return windows
    features = windows.select(['vocal_fraction', 'bandwidth_khz', 'mean_duration_ms']).to_numpy()
    spread = features.std(axis=0)
    spread[spread == 0] = 1.0
    score = ((features - features.mean(axis=0)) / spread).sum(axis=1)
    return windows.with_columns(pls.Series('score', score)).sort('score', descending=True)


def plot_vocal_pose_window_candidates(root_directory: str, windows: pls.DataFrame, visualizations_parameter_dict: dict[str, Any] | None = None,
                                      message_output: Callable[..., Any] | None = None) -> Any:
    """
    Description
    -----------
    Stack the emitter-coloured spectrograms of the best non-overlapping candidate windows (up to
    ``candidates.n_shown``), each from the microphone most of its calls peak on and captioned with its
    rank letter, times and summary numbers, for choosing a window by eye.

    Parameters
    ----------
    root_directory (str)
        Session directory.
    windows (pls.DataFrame)
        From :func:`find_vocal_pose_windows`.
    visualizations_parameter_dict (dict | None)
        The visualizations settings; loaded from the shipped JSON when None.
    message_output (Callable | None)
        Message sink; ``print`` when None.

    Returns
    -------
    fig (matplotlib.figure.Figure)
        The figure, one row per candidate.
    """
    sink: Callable[..., Any] = print if message_output is None else message_output
    root = pathlib.Path(root_directory)
    if visualizations_parameter_dict is None:
        with (pathlib.Path(__file__).parent.parent / '_parameter_settings' / 'visualizations_settings.json').open('r') as settings_file:
            visualizations_parameter_dict = json.load(settings_file)
    apply_plot_style()
    settings = visualizations_parameter_dict['vocal_pose_figures']
    candidates, layout, spectrogram_settings = settings['candidates'], settings['layout'], settings['spectrogram']
    session = load_session_poses(root, visualizations_parameter_dict)
    calls = read_session_calls(root, sink)
    sex_color = {'male': visualizations_parameter_dict['male_colors'][0], 'female': visualizations_parameter_dict['female_colors'][0]}
    none_color = visualizations_parameter_dict['unassigned_colors'][0]
    chosen: list[dict[str, Any]] = []
    for row in windows.iter_rows(named=True):
        if all(abs(row['start'] - other['start']) >= row['end'] - row['start'] for other in chosen):
            chosen.append(row)
        if len(chosen) == int(candidates['n_shown']):
            break
    fig, axes = plt.subplots(len(chosen), 1, figsize=(float(candidates['sheet_width_inches']), float(candidates['row_height_inches']) * len(chosen)),
                             dpi=int(candidates['sheet_dpi']), facecolor=layout['background_color'])
    for rank, (ax, row) in enumerate(zip(np.atleast_1d(axes), chosen, strict=True)):
        span = row['end'] - row['start']
        image, band, _ = emitter_spectrogram(root, calls, session['sex_of'], sex_color, none_color, row['end'], span, row['microphone'],
                                             spectrogram_settings, sink)
        ax.imshow(image, origin='lower', aspect='auto', extent=(0.0, span, band[0], band[-1]), interpolation='antialiased')
        ax.set_ylim(*spectrogram_settings['freq_range_khz'])
        ax.set_yticks(list(spectrogram_settings['freq_ticks_khz']))
        ax.tick_params(axis='y', length=0, labelsize=layout['tick_font_size'], pad=layout['tick_pad'], colors=layout['ink_color'])
        ax.set_xticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(f"{chr(ord('A') + rank)}   {row['start']:.2f} to {row['end']:.2f} s   {row['n_calls']} calls   "
                     f"{row['vocal_fraction'] * 100:.0f}% vocal   bandwidth {row['bandwidth_khz']:.0f} kHz   mean {row['mean_duration_ms']:.0f} ms   "
                     f"animals {row['gap_median_cm']:.0f} cm apart (max {row['gap_max_cm']:.0f})   mic {row['microphone']}",
                     loc='left', fontsize=layout['tick_font_size'], color=layout['ink_color'], pad=layout['tick_pad'])
    fig.supylabel(layout['frequency_label'], fontsize=layout['tick_font_size'], color=layout['ink_color'])
    fig.subplots_adjust(**candidates['sheet_margins'])
    return fig


def vocal_pose_view_picker_html(root_directory: str, visualizations_parameter_dict: dict[str, Any] | None = None, title: str = 'Pose View Picker') -> str:
    """
    Description
    -----------
    A self-contained page for choosing the still's camera: the two animals at ``window.end_time``
    with a sparse version of their trail, drawn on a canvas that turns when dragged or with two
    sliders, reporting the azimuth and elevation in the figure's convention so the numbers can be
    copied into ``camera.azimuth`` / ``camera.elevation``. Presets jump to the male-side, heads-down,
    tails-down and top views. Display it in a notebook with ``IPython.display.HTML``.

    Parameters
    ----------
    root_directory (str)
        Session directory.
    visualizations_parameter_dict (dict | None)
        The visualizations settings; loaded from the shipped JSON when None.
    title (str)
        Page title.

    Returns
    -------
    html (str)
        The page.
    """
    root = pathlib.Path(root_directory)
    if visualizations_parameter_dict is None:
        with (pathlib.Path(__file__).parent.parent / '_parameter_settings' / 'visualizations_settings.json').open('r') as settings_file:
            visualizations_parameter_dict = json.load(settings_file)
    settings = visualizations_parameter_dict['vocal_pose_figures']
    picker, window = settings['view_picker'], settings['window']
    session = load_session_poses(root, visualizations_parameter_dict)
    names, nodes, fps, sex_of = session['names'], session['nodes'], session['fps'], session['sex_of']
    end_seconds = float(window['end_time'])
    last = int(np.floor(end_seconds * fps))
    first = last - round(float(window['history_seconds']) * fps)
    poses = session['tracks'][first:last + 1].copy()
    centre = 0.5 * (np.nanmin(poses[..., :2].reshape(-1, 2), axis=0) + np.nanmax(poses[..., :2].reshape(-1, 2), axis=0))
    poses[..., 0] -= centre[0]
    poses[..., 1] -= centre[1]
    male_side = male_side_azimuth(poses[-1], names, nodes, sex_of)
    heading = poses[-1, :, nodes.index('Nose'), :2] - poses[-1, :, nodes.index('TTI'), :2]
    heading = (heading / np.linalg.norm(heading, axis=1, keepdims=True)).mean(axis=0)
    heads_down = float(np.degrees(np.arctan2(heading[1], heading[0])))
    tails_down = ((heads_down + 360.0) % 360.0) - 180.0
    elevation = float(settings['camera']['elevation'])
    steps = np.unique(np.linspace(0, poses.shape[0] - 2, int(picker['n_trail'])).round().astype(int))
    trail = [{'strength': round(float(trail_strength((poses.shape[0] - 1 - step) / fps, picker, 0.0)), 3), 'poses': np.round(poses[step], 4).tolist()}
             for step in steps]
    data = {'colors': session['colors'],
            'connections': [[nodes.index(a), nodes.index(b)] for a, b in (pair.split('-') for pair in MOUSE_NODE_CONNECTIONS)],
            'polygons': [[nodes.index(name) for name in polygon.split('-')] for polygon in MOUSE_NODE_POLYGONS],
            'current': np.round(poses[-1], 4).tolist(), 'trail': trail,
            'radius': round(float(np.nanmax(np.hypot(poses[..., 0], poses[..., 1]))), 4), 'height': round(float(np.nanmax(poses[..., 2])), 4),
            'start': {'azimuth': round(male_side), 'elevation': round(elevation)},
            'presets': [{'name': 'Male side', 'azimuth': round(male_side), 'elevation': round(elevation)},
                        {'name': 'Heads down', 'azimuth': round(heads_down), 'elevation': round(elevation)},
                        {'name': 'Tails down', 'azimuth': round(tails_down), 'elevation': round(elevation)},
                        {'name': 'From above', 'azimuth': round(male_side), 'elevation': 90}],
            'body_opacity': float(settings['skeleton']['body_opacity']), 'ink': settings['layout']['ink_color'],
            'background': settings['layout']['background_color']}
    where = f'{root.name} \u00b7 frame at {end_seconds:.2f} s \u00b7 trail {float(window["history_seconds"]):g} s'
    # Element ids and the script's scope are keyed by the frame, so several pickers can sit in one notebook.
    page = '''<style>
/* Layout: one working surface (the canvas) with its controls stacked beneath; a single column at every width. */
:root {
  --bg: #F3F5F6; --paper: #FFFFFF; --fg: #1D2327; --muted: #5B6770; --line: #D5DBDF; --accent: #2C6E85; --accent-fg: #FFFFFF;
  --sans: system-ui, -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
  --mono: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
}
.picker { background: var(--bg); color: var(--fg); font-family: var(--sans); padding: 20px 16px; }
.picker main { max-width: 760px; margin-inline: auto; display: flex; flex-direction: column; gap: 14px; }
.picker h1 { font-size: 1.15rem; font-weight: 600; margin: 0; }
.picker .where { color: var(--muted); font-family: var(--mono); font-size: 0.82rem; margin: 0; }
.picker .stage { background: var(--paper); border: 1px solid var(--line); border-radius: 6px; touch-action: none; cursor: grab; }
.picker .stage:active { cursor: grabbing; }
.picker canvas { display: block; width: 100%; aspect-ratio: 4 / 3; }
.picker .readout { display: flex; flex-wrap: wrap; align-items: center; gap: 10px 16px; }
.picker .numbers { font-family: var(--mono); font-variant-numeric: tabular-nums; font-size: 1.05rem; }
.picker .rows { display: grid; grid-template-columns: auto 1fr auto; gap: 8px 12px; align-items: center; }
.picker .rows label { font-size: 0.9rem; }
.picker .rows output { font-family: var(--mono); font-variant-numeric: tabular-nums; min-width: 4.5ch; text-align: right; }
.picker input[type="range"] { width: 100%; min-width: 0; accent-color: var(--accent); }
.picker .presets { display: flex; flex-wrap: wrap; gap: 8px; }
.picker button { font: inherit; font-size: 0.88rem; padding: 6px 12px; border-radius: 5px; border: 1px solid var(--line); background: var(--paper); color: var(--fg); cursor: pointer; }
.picker button:hover { border-color: var(--accent); }
.picker button.primary { background: var(--accent); color: var(--accent-fg); border-color: var(--accent); }
.picker .hint { color: var(--muted); font-size: 0.88rem; margin: 0; max-width: 65ch; }
</style>
<div class="picker" id="picker-__ID__">
<main>
  <h1>__TITLE__</h1>
  <p class="where">__WHERE__</p>
  <div class="stage" data-role="stage"><canvas data-role="view" aria-label="The two animals at the last frame, seen from the chosen camera. Drag to turn."></canvas></div>
  <div class="readout">
    <span class="numbers" data-role="numbers">azimuth 0, elevation 45</span>
    <button class="primary" data-role="copy" type="button">Copy these numbers</button>
    <span class="hint" data-role="copied" hidden>Copied.</span>
  </div>
  <div class="rows">
    <label>Turn (azimuth)</label><input data-role="azimuth" type="range" min="-180" max="180" step="1"><output data-role="azimuth-out"></output>
    <label>Height (elevation)</label><input data-role="elevation" type="range" min="0" max="90" step="1"><output data-role="elevation-out"></output>
  </div>
  <div class="presets" data-role="presets"></div>
  <p class="hint">Drag the picture to turn it: sideways changes the azimuth, up and down changes the elevation. Elevation 90 is straight down, 0 is level with the floor. Copy the two numbers into camera.azimuth and camera.elevation. The two animals are in their palette colours; the faint shapes are their trails.</p>
</main>
</div>
<script>
(function () {
const DATA = __DATA__;
const rootElement = document.getElementById('picker-__ID__');
const byRole = role => rootElement.querySelector('[data-role="' + role + '"]');
const canvas = byRole('view');
const context = canvas.getContext('2d');
const azimuthInput = byRole('azimuth');
const elevationInput = byRole('elevation');
let azimuth = DATA.start.azimuth;
let elevation = DATA.start.elevation;

function project(point, scale, cx, cy) {
  // matplotlib's convention: the camera sits at (azimuth, elevation) looking at the origin; orthographic.
  const a = azimuth * Math.PI / 180, e = elevation * Math.PI / 180;
  const sx = -Math.sin(a) * point[0] + Math.cos(a) * point[1];
  const sy = -Math.sin(e) * (Math.cos(a) * point[0] + Math.sin(a) * point[1]) + Math.cos(e) * point[2];
  return [cx + sx * scale, cy - sy * scale];
}
function mix(hex, strength) {
  const parse = h => [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16));
  const c = parse(hex), p = parse(DATA.background);
  return 'rgb(' + c.map((v, i) => Math.round(p[i] + (v - p[i]) * strength)).join(',') + ')';
}
function drawPose(pose, color, scale, cx, cy, width, withBody) {
  const pts = pose.map(p => project(p, scale, cx, cy));
  if (withBody) {
    context.globalAlpha = DATA.body_opacity; context.fillStyle = color;
    for (const polygon of DATA.polygons) { context.beginPath(); polygon.forEach((n, i) => i ? context.lineTo(...pts[n]) : context.moveTo(...pts[n])); context.closePath(); context.fill(); }
    context.globalAlpha = 1;
  }
  context.strokeStyle = color; context.lineWidth = width; context.lineCap = 'round';
  for (const [i, j] of DATA.connections) { context.beginPath(); context.moveTo(...pts[i]); context.lineTo(...pts[j]); context.stroke(); }
  if (withBody) { context.fillStyle = color; for (const p of pts) { context.beginPath(); context.arc(p[0], p[1], width * 0.85, 0, 2 * Math.PI); context.fill(); } }
}
function draw() {
  const ratio = window.devicePixelRatio || 1;
  const box = canvas.getBoundingClientRect();
  if (box.width === 0) { return; }
  canvas.width = Math.round(box.width * ratio); canvas.height = Math.round(box.height * ratio);
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.fillStyle = DATA.background; context.fillRect(0, 0, box.width, box.height);
  const scale = 0.46 * Math.min(box.width, box.height) / DATA.radius;
  const cx = box.width / 2, cy = box.height / 2 + 0.25 * DATA.height * scale * Math.cos(elevation * Math.PI / 180);
  const width = Math.max(2, 0.0042 * scale);
  DATA.trail.forEach(step => step.poses.forEach((pose, animal) => drawPose(pose, mix(DATA.colors[animal], step.strength), scale, cx, cy, width, false)));
  DATA.current.forEach((pose, animal) => drawPose(pose, DATA.colors[animal], scale, cx, cy, width, true));
  byRole('numbers').textContent = 'azimuth ' + Math.round(azimuth) + ', elevation ' + Math.round(elevation);
  azimuthInput.value = Math.round(azimuth); elevationInput.value = Math.round(elevation);
  byRole('azimuth-out').textContent = Math.round(azimuth) + '°';
  byRole('elevation-out').textContent = Math.round(elevation) + '°';
  byRole('copied').hidden = true;
}
function wrap(angle) { return ((angle + 180) % 360 + 360) % 360 - 180; }
azimuthInput.addEventListener('input', () => { azimuth = Number(azimuthInput.value); draw(); });
elevationInput.addEventListener('input', () => { elevation = Number(elevationInput.value); draw(); });
const stage = byRole('stage');
let last = null;
stage.addEventListener('pointerdown', event => { last = [event.clientX, event.clientY]; stage.setPointerCapture(event.pointerId); });
stage.addEventListener('pointermove', event => {
  if (!last) return;
  azimuth = wrap(azimuth - (event.clientX - last[0]) * 0.5);
  elevation = Math.min(90, Math.max(0, elevation + (event.clientY - last[1]) * 0.5));
  last = [event.clientX, event.clientY]; draw();
});
stage.addEventListener('pointerup', () => { last = null; });
stage.addEventListener('pointercancel', () => { last = null; });
for (const preset of DATA.presets) {
  const button = document.createElement('button');
  button.type = 'button'; button.textContent = preset.name;
  button.addEventListener('click', () => { azimuth = preset.azimuth; elevation = preset.elevation; draw(); });
  byRole('presets').appendChild(button);
}
byRole('copy').addEventListener('click', () => {
  const text = byRole('numbers').textContent;
  const done = () => { byRole('copied').hidden = false; };
  if (navigator.clipboard && navigator.clipboard.writeText) { navigator.clipboard.writeText(text).then(done).catch(() => {}); }
});
new ResizeObserver(draw).observe(canvas);
draw();
})();
</script>
'''
    return (page.replace('__TITLE__', title).replace('__WHERE__', where).replace('__ID__', f'{root.name}-{last}')
            .replace('__DATA__', json.dumps(data, separators=(',', ':'))))


def vocal_pose_image_html(png_path: pathlib.Path) -> str:
    """
    Description
    -----------
    A one-image HTML snippet embedding a PNG, for showing a still in a notebook at full width.

    Parameters
    ----------
    png_path (pathlib.Path)
        The image.

    Returns
    -------
    html (str)
        The snippet.
    """
    uri = base64.b64encode(png_path.read_bytes()).decode('ascii')
    return f'<img src="data:image/png;base64,{uri}" style="max-width:100%;height:auto;display:block" alt="{png_path.name}">'
