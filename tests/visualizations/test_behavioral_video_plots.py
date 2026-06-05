"""
@author: bartulem
Tests for the per-frame plotting helpers in
visualizations/make_behavioral_videos.py.

These module-level functions render one composited video frame's worth of
content (3D mouse skeleton, speaker, arena + microphones, spectrogram, spike
raster, behavioral-feature traces) onto pre-existing matplotlib axes. They are
the pure-rendering core of the otherwise opencv/ffmpeg-bound video builder, so
they can be exercised directly against tiny synthetic inputs on the Agg
backend without writing any video.
"""

from __future__ import annotations

import numpy as np
import polars as pls
import pytest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from usv_playpen.visualizations.make_behavioral_videos import (
    plot_mouse_data,
    plot_speaker_data,
    plot_spectrogram,
    plot_raster,
    plot_behavioral_features,
    plot_arena_corners_mics,
)


_COLOR_MODE = {
    "tick_color": "#888888",
    "text_color": "#202020",
    "spectrogram_text_color": "#FFFFFF",
    "background_color": "#000000",
    "arena_line_color": "#444444",
}

_MOUSE_NODES = ["Nose", "Head", "Neck", "Trunk", "TTI"]
_MOUSE_CONNECTIONS = [
    "Nose-Head", "Head-Neck", "Neck-Trunk", "Trunk-TTI", "Nose-Neck",
    "Head-Trunk", "Neck-TTI", "Nose-Trunk", "Head-TTI",
]


def _ax3d():
    """
    Description
    -----------
    Create a fresh figure with a single 3D axes for the 3D plotting helpers.

    Parameters
    ----------

    Returns
    -------
    fig, ax (tuple)
        The figure and its 3D axes.
    """

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    return fig, ax


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_plot_mouse_data_renders_history_polygons_and_nodes():
    """
    Description
    -----------
    `plot_mouse_data` must draw, for each mouse, the movement-history trail,
    the node-connection skeleton (with the body-edge vs. animal-colour split),
    the shaded body polygons, and the body nodes — leaving artists on the 3D
    axes and applying the configured axis limits.

    Parameters
    ----------

    Returns
    -------
    None
    """

    rng = np.random.default_rng(0)
    data = rng.uniform(-0.2, 0.2, size=(20, 2, len(_MOUSE_NODES), 3))
    cmaps = [matplotlib.colormaps["Reds"], matplotlib.colormaps["Greens"]]
    fig, ax = _ax3d()
    try:
        plot_mouse_data(
            data=data, plot_axes=ax, frame_number=10,
            animal_node_names=_MOUSE_NODES,
            animal_color=["#FF0000", "#00FF00"],
            animal_cm=cmaps, animal_line_width=1,
            node_connections=_MOUSE_CONNECTIONS,
            node_polygons=["Nose-Head-Neck"],
            node_lw=0.5, node_size=10, node_opacity=0.8,
            node_edge_color="#202020",
            polygon_color=["#FF000033", "#00FF0033"], polygon_opacity=0.3,
            body_edge_color="#101010",
            history_frame_span=5, history_point="Trunk",
            history_ls="-", history_lw=0.5,
            xlim_=0.5, ylim_=0.5, zlim_=0.5,
            node_bool=True, history_bool=True,
        )
        assert len(ax.lines) > 0, "no skeleton/history lines drawn"
        assert ax.get_zlim3d() == (0.0, 0.5)
    finally:
        plt.close(fig)


def test_plot_speaker_data_adds_scatter_point():
    """
    Description
    -----------
    `plot_speaker_data` must scatter the speaker's 3D position for the frame.

    Parameters
    ----------

    Returns
    -------
    None
    """

    speaker = np.zeros((5, 1, 1, 3))
    speaker[2, 0, 0, :] = [0.1, -0.1, 0.05]
    fig, ax = _ax3d()
    try:
        plot_speaker_data(
            speaker_data=speaker, plot_axes=ax, frame_number=2,
            speaker_color="#3333FF", speaker_alpha=0.7,
        )
        assert len(ax.collections) >= 1, "speaker scatter not drawn"
    finally:
        plt.close(fig)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_plot_spectrogram_renders_with_usv_segments_and_cbar():
    """
    Description
    -----------
    `plot_spectrogram` must render the dB spectrogram via librosa specshow,
    mark the window centre, overlay the USV segment bars, and (when requested)
    attach a colorbar — formatting the bounded time / frequency axes.

    Parameters
    ----------

    Returns
    -------
    None
    """

    rng = np.random.default_rng(1)
    spec = rng.uniform(-80.0, 0.0, size=(128, 200)).astype(np.float32)
    fig, ax = plt.subplots()
    try:
        plot_spectrogram(
            plot_axes=ax, figure_object=fig,
            spec_start=0, spec_end=200, audio_sr=250000, stft_hop=128,
            half_window_size_sec=0.5, color_mode_preferences=_COLOR_MODE,
            spectrogram_amplitude=spec, power_limit=[-80, 0],
            freq_limit=[0, 125000], freq_yticks=[],
            usv_segments_list=[(-0.2, 0.2)], usv_segment_lw=2.0,
            usv_segment_colors_list=["#FF0000"], usv_segments_ypos=60000,
            cbar_bool=True, plot_usv_segments_bool=True,
        )
        # librosa specshow renders via pcolormesh (a QuadMesh collection),
        # and the requested colorbar adds a second axes to the figure.
        assert len(ax.collections) >= 1, "spectrogram mesh not drawn"
        assert len(fig.axes) >= 2, "colorbar axes not added"
    finally:
        plt.close(fig)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_plot_raster_draws_eventplot_and_area_labels():
    """
    Description
    -----------
    `plot_raster` must lay down the per-unit spike eventplot and render the
    per-bucket brain-area labels (coloured via the bucket palette) for the
    probe's recorded regions.

    Parameters
    ----------

    Returns
    -------
    None
    """

    raster_data = [
        np.array([-0.4, -0.1, 0.0, 0.2]),
        np.array([-0.3, 0.05, 0.3]),
    ]
    scheme = {
        "PAG": "#677470", "MRN": "#939884", "VTA": "#F5D27A",
        "SC": "#9FB7D8", "CENT": "#D88080", "MB": "#9BBE85", "other": "#B8B8B8",
    }
    fig, ax = plt.subplots()
    fig.canvas.draw()  # ensure a renderer exists for the label-extent math
    try:
        plot_raster(
            plot_axes=ax, figure_object=fig, unit_num=2,
            raster_data=raster_data, raster_half_window=150,
            raster_half_window_sec=1.0,
            raster_brain_area={"imec0": ["PAG", "MRN"]},
            raster_line_lengths=[0.8, 0.8], raster_line_widths=[0.5, 0.5],
            filtered_brain_areas=[], color_mode_preferences=_COLOR_MODE,
            event_plot_colors=["#FF0000", "#00FF00"],
            brain_area_color_scheme=scheme,
        )
        assert len(ax.collections) >= 1, "raster eventplot not drawn"
        texts = [t.get_text() for t in ax.texts]
        assert "PAG" in texts and "MRN" in texts
    finally:
        plt.close(fig)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("special_features", [[], ["speed"]])
def test_plot_behavioral_features_traces(special_features):
    """
    Description
    -----------
    `plot_behavioral_features` must add one trace axes per behavioral feature,
    colouring self-features by animal and dyadic features by theme, in both the
    no-accent (`special_features=[]`) and accentuated modes.

    Parameters
    ----------
    special_features (list[str])
        Features to accentuate (empty for the plain path).

    Returns
    -------
    None
    """

    n = 30
    rng = np.random.default_rng(2)
    beh_df = pls.DataFrame({
        "m1.speed": rng.uniform(0, 30, n),
        "m2.speed": rng.uniform(0, 30, n),
        "m1-m2.nose-nose": rng.uniform(0, 90, n),
    })
    features = ["m1.speed", "m2.speed", "m1-m2.nose-nose"]
    plot_axes: dict = {}
    fig = plt.figure()
    try:
        plot_behavioral_features(
            plot_axes=plot_axes, figure_object=fig,
            mouse_track_names=["m1", "m2"], special_features=special_features,
            beh_features_to_plot=features, beh_feature_data=beh_df,
            beh_features_fig_position=[0.1, 0.8, 0.3, 0.03],
            beh_window_size_sec=0.1, beh_window_size_frames=n,
            beh_half_window_size_frames=15,
            beh_features_ylabels={
                "speed": "speed (cm/s)", "nose-nose": "distance (cm)",
            },
            feature_ts_fr_start=0, feature_ts_fr_end=n, feature_ts_fr_middle=15,
            x_axis_start=0, x_axis_middle=15, x_axis_end=29,
            ylim_dict={"speed": [0, 30], "nose-nose": [0, 90]},
            plot_theme="dark", color_mode_preferences=_COLOR_MODE,
            animal_colors=["#FF0000", "#00FF00"], remove_axes_bool=False,
        )
        # One trace axes created per feature (keyed 3, 4, 5).
        assert set(plot_axes.keys()) == {3, 4, 5}
    finally:
        plt.close(fig)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_plot_arena_corners_mics_full_render():
    """
    Description
    -----------
    `plot_arena_corners_mics` must draw the arena rails (bottom + vertical),
    the active / inactive microphones, the corner markers, the mesh walls, the
    node-connection lines, and the session / frame / animal-ID text overlays.

    Parameters
    ----------

    Returns
    -------
    None
    """

    # 28-node arena: North/West/South/East at 0..3, 24 mics at 4..27.
    arena_names = ["North", "West", "South", "East"] + [f"ch_{i}" for i in range(24)]
    data = np.zeros((1, 1, 28, 3))
    data[0, 0, 0, :] = [0.0, 0.5, 0.0]    # North
    data[0, 0, 1, :] = [-0.5, 0.0, 0.0]   # West
    data[0, 0, 2, :] = [0.0, -0.5, 0.0]   # South
    data[0, 0, 3, :] = [0.5, 0.0, 0.0]    # East
    rng = np.random.default_rng(3)
    data[0, 0, 4:, :] = rng.uniform(-0.5, 0.5, size=(24, 3))

    arena_connections = ["North-West", "North-East", "South-West", "South-East"]
    fig, ax = _ax3d()
    try:
        plot_arena_corners_mics(
            data=data, plot_axes=ax, frame_number=42, session_id="20240101_120000",
            esr=150.0, animal_id={"m1": "♂", "m2": "♀"},
            animal_colors=["#FF0000", "#00FF00"], color_mode_preferences=_COLOR_MODE,
            arena_node_connections=arena_connections, arena_node_names=arena_names,
            arena_axes_lw=1.0, arena_mics_lw=0.5, arena_mics_opacity=0.6,
            corner_size=20, corner_opacity=0.8, mesh_color="#333333",
            mesh_opacity=0.2, active_mic_position=0, active_mic_color="#FF0000",
            inactive_mic_color="#777777", text_start_coords=[0.02, 0.95],
            main_text_offset=0.03, mouse_id_text_offset=0.06, text_fontsize=8,
            arena_node_connections_bool=True, plot_corners_bool=True,
            plot_mesh_walls_bool=True, active_mic_bool=True, inactive_mic_bool=True,
        )
        assert len(ax.collections) >= 1, "no mic/corner scatter or mesh drawn"
        assert any("20240101_120000" in t.get_text() for t in ax.texts)
    finally:
        plt.close(fig)
