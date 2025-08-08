"""
@author: bartulem
Visualizes 3D tracking, vocalization and neural data.
"""

from __future__ import annotations

import json
import pathlib
import traceback
from datetime import datetime
from typing import Union

import click
from click.core import ParameterSource

from .cli_utils import modify_settings_json_for_cli
from .send_email import Messenger
from .visualizations.make_behavioral_tuning_figures import RatemapFigureMaker
from .visualizations.make_behavioral_videos import Create3DVideo


class Visualizer:
    def __init__(
        self,
        input_parameter_dict: dict = None,
        root_directories: list = None,
        message_output: callable = None,
    ) -> None:
        """
        Initializes the Visualizer class.

        Parameter
        ---------
        root_directories (list)
            Root directories for data; defaults to None.
        input_parameter_dict (dict)
            Analyses parameters; defaults to None.
        message_output (function)
            Defines output messages; defaults to None.

        Returns
        -------
        -------
        """

        if root_directories is None:
            with open(
                pathlib.Path(__file__).parent
                / "_parameter_settings/visualizations_settings.json"
            ) as json_file:
                self.root_directories = json.load(json_file)["visualize_data"][
                    "root_directories"
                ]
        else:
            self.root_directories = root_directories

        if input_parameter_dict is None:
            with open(
                pathlib.Path(__file__).parent
                / "_parameter_settings/visualizations_settings.json"
            ) as json_file:
                self.input_parameter_dict = json.load(json_file)
        else:
            self.input_parameter_dict = input_parameter_dict

        if message_output is None:
            self.message_output = print
        else:
            self.message_output = message_output

    def visualize_data(self) -> None:
        """
        Description
        ----------
        This method performs the following analyses:
        (1) create behavioral tuning curve figures
        (2) visualizes (plot or video) 3D tracking, vocalization and neural data
        ----------

        Parameters
        ----------
        ----------

        Returns
        ----------
        ----------
        """

        Messenger(
            message_output=self.message_output,
            receivers=self.input_parameter_dict["send_email"]["send_message"][
                "receivers"
            ],
            exp_settings_dict=None,
        ).send_message(
            subject=f"{self.input_parameter_dict['send_email']['visualizations_pc_choice']} PC is busy, do NOT attempt to remote in!",
            message=f"Data visualizations in progress, started at "
            f"{datetime.now().hour:02d}:{datetime.now().minute:02d}.{datetime.now().second:02d} "
            f"and run by @{self.input_parameter_dict['send_email']['experimenter']}. "
            f"You will be notified upon completion. \n \n ***This is an automatic e-mail, please do NOT respond.***",
        )

        for one_directory in self.root_directories:
            try:
                self.message_output(
                    f"Visualizing data in {one_directory} started at: "
                    f"{datetime.now().hour:02d}:{datetime.now().minute:02d}.{datetime.now().second:02d}."
                )

                # # # # plot behavioral tuning curves
                if self.input_parameter_dict["visualize_booleans"][
                    "make_behavioral_tuning_figures_bool"
                ]:
                    RatemapFigureMaker(
                        root_directory=one_directory,
                        visualizations_parameter_dict=self.input_parameter_dict,
                        message_output=self.message_output,
                    ).neuronal_tuning_figures()

                # # # # make behavioral videos
                if self.input_parameter_dict["visualize_booleans"][
                    "make_behavioral_videos_bool"
                ]:
                    Create3DVideo(
                        root_directory=one_directory,
                        arena_directory=self.input_parameter_dict[
                            "make_behavioral_videos"
                        ]["arena_directory"],
                        speaker_audio_file=self.input_parameter_dict[
                            "make_behavioral_videos"
                        ]["speaker_audio_file"],
                        exp_id=self.input_parameter_dict["send_email"]["experimenter"],
                        visualizations_parameter_dict=self.input_parameter_dict,
                        message_output=self.message_output,
                    ).visualize_in_video()

                self.message_output(
                    f"Visualizing data in {one_directory} finished at: "
                    f"{datetime.now().hour:02d}:{datetime.now().minute:02d}.{datetime.now().second:02d}."
                )

            except (
                OSError,
                RuntimeError,
                TypeError,
                IndexError,
                EOFError,
                TimeoutError,
                NameError,
                KeyError,
                ValueError,
                AttributeError,
            ):
                self.message_output(traceback.format_exc())

        Messenger(
            message_output=self.message_output,
            no_receivers_notification=False,
            receivers=self.input_parameter_dict["send_email"]["send_message"][
                "receivers"
            ],
            exp_settings_dict=None,
        ).send_message(
            subject=f"{self.input_parameter_dict['send_email']['visualizations_pc_choice']} PC is available again, visualizations have been completed",
            message=f"Data visualizations have been completed at "
            f"{datetime.now().hour:02d}:{datetime.now().minute:02d}.{datetime.now().second:02d} "
            f"by @{self.input_parameter_dict['send_email']['experimenter']}. "
            f"You will be notified about further PC usage "
            f"should it occur. \n \n ***This is an automatic e-mail, please do NOT respond.***",
        )


@click.command(name="generate-viz")
@click.option(
    "--root-directory",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default=None,
    required=True,
    help="Session root directory path.",
)
@click.option(
    "--arena-directory",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default=None,
    required=True,
    help="Arena session path.",
)
@click.option(
    "--exp-id",
    "experimenter",
    type=str,
    default=None,
    required=True,
    help="Experimenter ID.",
)
@click.option(
    "--speaker-audio-file",
    "speaker_audio_file",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    default=None,
    required=False,
    help="Speaker audio file path.",
)
@click.option(
    "--sequence-audio-file",
    "sequence_audio_file",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    default=None,
    required=False,
    help="Audible audio sequence file path.",
)
@click.option(
    "--animate/--no-animate",
    "animate_bool",
    default=None,
    help="Animate visualization.",
)
@click.option(
    "--video-start-time",
    "video_start_time",
    type=click.IntRange(min=0),
    default=None,
    required=False,
    help="Video start time (in s).",
)
@click.option(
    "--video-duration",
    "video_duration",
    type=Union[int, float],
    default=None,
    required=False,
    help="Video duration (in s).",
)
@click.option(
    "--plot-theme",
    "plot_theme",
    type=str,
    default=None,
    required=False,
    help="Plot background theme (light or dark).",
)
@click.option(
    "--save-fig/--no-save-fig",
    "save_fig",
    default=None,
    help="Save plot as figure to file.",
)
@click.option(
    "--view-angle",
    "view_angle",
    type=str,
    default=None,
    required=False,
    help='View angle for 3D visualization ("top" or "side").',
)
@click.option(
    "--side-azimuth-start",
    "side_azimuth_start",
    type=Union[int, float],
    default=None,
    required=False,
    help="Azimuth angle for side view (in degrees).",
)
@click.option(
    "--rotate-side-view/--no-rotate-side-view",
    "rotate_side_view_bool",
    default=None,
    help="Rotate side view in animation.",
)
@click.option(
    "--rotation-speed",
    "rotation_speed",
    type=Union[int, float],
    default=None,
    required=False,
    help="Rotation speed (in degrees/s).",
)
@click.option(
    "--history/--no-history",
    "history_bool",
    default=None,
    help="Display history of single mouse node.",
)
@click.option(
    "--speaker/--no-speaker",
    "speaker_bool",
    default=None,
    help="Display speaker node in visualization.",
)
@click.option(
    "--spectrogram/--no-spectrogram",
    "spectrogram_bool",
    default=None,
    help="Display spectrogram of audio sequence.",
)
@click.option(
    "--spectrogram-ch",
    "spectrogram_ch",
    type=int,
    default=None,
    required=False,
    help="Spectrogram channel (0-23).",
)
@click.option(
    "--raster-plot/--no-raster-plot",
    "raster_plot_bool",
    default=None,
    help="Display spike raster plot in visualization.",
)
@click.option(
    "--brain-areas",
    "brain_areas",
    multiple=True,
    type=str,
    default=None,
    required=False,
    help="Brain areas to display in raster plot.",
)
@click.option(
    "--other",
    "other",
    multiple=True,
    type=str,
    default=None,
    required=False,
    help="Other spike cluster features to use for filtering.",
)
@click.option(
    "--raster-special-units",
    "raster_special_units",
    multiple=True,
    type=str,
    default=None,
    required=False,
    help="Clusters to accentuate in raster plot.",
)
@click.option(
    "--spike-sound/--no-spike-sound",
    "spike_sound_bool",
    default=None,
    help="Play sound each time the cluster spikes.",
)
@click.option(
    "--beh-features/--no-beh-features",
    "beh_features_bool",
    default=None,
    help="Display behavioral feature dynamics.",
)
@click.option(
    "--beh-features-to-plot",
    "beh_features_to_plot",
    multiple=True,
    type=str,
    default=None,
    required=False,
    help="Behavioral feature(s) to display.",
)
@click.option(
    "--special-beh-features",
    "special_beh_features",
    multiple=True,
    type=str,
    default=None,
    required=False,
    help="Behavioral feature(s) to accentuate in display.",
)
@click.option(
    "--fig-format",
    "fig_format",
    type=str,
    default=None,
    required=False,
    help="Figure format.",
)
@click.option(
    "--fig-dpi",
    "fig_dpi",
    type=int,
    default=None,
    required=False,
    help="Figure resolution in dots per inch.",
)
@click.option(
    "--animation-writer",
    "animation_writer",
    type=str,
    default=None,
    required=False,
    help="Animation writer backend.",
)
@click.option(
    "--animation-format",
    "animation_format",
    type=str,
    default=None,
    required=False,
    help="Video format.",
)
@click.option(
    "--arena-node-connections/--no-arena-node-connections",
    "arena_node_connections_bool",
    default=None,
    help="Display connections between arena nodes.",
)
@click.option(
    "--arena-axes-lw",
    "arena_axes_lw",
    type=float,
    default=None,
    required=False,
    help="Line width for the arena axes.",
)
@click.option(
    "--arena-mics-lw",
    "arena_mics_lw",
    type=float,
    default=None,
    required=False,
    help="Line width for the microphone markers.",
)
@click.option(
    "--arena-mics-opacity",
    "arena_mics_opacity",
    type=float,
    default=None,
    required=False,
    help="Opacity for the microphone markers.",
)
@click.option(
    "--plot-corners/--no-plot-corners",
    "plot_corners_bool",
    default=None,
    help="Display arena corner markers.",
)
@click.option(
    "--corner-size",
    "corner_size",
    type=float,
    default=None,
    required=False,
    help="Size of the arena corner markers.",
)
@click.option(
    "--corner-opacity",
    "corner_opacity",
    type=float,
    default=None,
    required=False,
    help="Opacity of the arena corner markers.",
)
@click.option(
    "--plot-mesh-walls/--no-plot-mesh-walls",
    "plot_mesh_walls_bool",
    default=None,
    help="Display arena walls as a mesh.",
)
@click.option(
    "--mesh-opacity",
    "mesh_opacity",
    type=float,
    default=None,
    required=False,
    help="Opacity of the arena wall mesh.",
)
@click.option(
    "--active-mic/--no-active-mic",
    "active_mic_bool",
    default=None,
    help="Display the active microphone marker.",
)
@click.option(
    "--inactive-mic/--no-inactive-mic",
    "inactive_mic_bool",
    default=None,
    help="Display inactive microphone markers.",
)
@click.option(
    "--inactive-mic-color",
    "inactive_mic_color",
    type=str,
    default=None,
    required=False,
    help="Color for inactive microphone markers.",
)
@click.option(
    "--text-fontsize",
    "text_fontsize",
    type=int,
    default=None,
    required=False,
    help="Font size for text elements in the plot.",
)
@click.option(
    "--speaker-opacity",
    "speaker_opacity",
    type=float,
    default=None,
    required=False,
    help="Opacity of the speaker node.",
)
@click.option(
    "--nodes/--no-nodes", "node_bool", default=None, help="Display mouse nodes."
)
@click.option(
    "--node-size",
    "node_size",
    type=float,
    default=None,
    required=False,
    help="Size of the mouse nodes.",
)
@click.option(
    "--node-opacity",
    "node_opacity",
    type=float,
    default=None,
    required=False,
    help="Opacity of the mouse nodes.",
)
@click.option(
    "--node-lw",
    "node_lw",
    type=float,
    default=None,
    required=False,
    help="Line width for the mouse node connections.",
)
@click.option(
    "--node-connection-lw",
    "node_connection_lw",
    type=float,
    default=None,
    required=False,
    help="Line width for mouse node connections.",
)
@click.option(
    "--body-opacity",
    "body_opacity",
    type=float,
    default=None,
    required=False,
    help="Opacity of the mouse body.",
)
@click.option(
    "--history-point",
    "history_point",
    type=str,
    default=None,
    required=False,
    help="Node to use for the history trail.",
)
@click.option(
    "--history-span-sec",
    "history_span_sec",
    type=int,
    default=None,
    required=False,
    help="Duration of the history trail (s).",
)
@click.option(
    "--history-ls",
    "history_ls",
    type=str,
    default=None,
    required=False,
    help="Line style for the history trail.",
)
@click.option(
    "--history-lw",
    "history_lw",
    type=float,
    default=None,
    required=False,
    help="Line width for the history trail.",
)
@click.option(
    "--beh-features-window-size",
    "beh_features_window_size",
    type=int,
    default=None,
    required=False,
    help="Window size for behavioral features (s).",
)
@click.option(
    "--raster-window-size",
    "raster_window_size",
    type=int,
    default=None,
    required=False,
    help="Window size for the raster plot (s).",
)
@click.option(
    "--raster-lw",
    "raster_lw",
    type=float,
    default=None,
    required=False,
    help="Line width for spikes in the raster plot.",
)
@click.option(
    "--raster-ll",
    "raster_ll",
    type=float,
    default=None,
    required=False,
    help="Line length for spikes in the raster plot.",
)
@click.option(
    "--spectrogram-cbar/--no-spectrogram-cbar",
    "spectrogram_cbar_bool",
    default=None,
    help="Display the color bar for the spectrogram.",
)
@click.option(
    "--spectrogram-plot-window-size",
    "spectrogram_plot_window_size",
    type=int,
    default=None,
    required=False,
    help="Window size for the spectrogram plot (s).",
)
@click.option(
    "--spectrogram-power-limit",
    "spectrogram_power_limit",
    nargs=2,
    type=int,
    default=None,
    required=False,
    help="Power limits (min/max) for spectrogram color scale.",
)
@click.option(
    "--spectrogram-frequency-limit",
    "spectrogram_frequency_limit",
    nargs=2,
    type=int,
    default=None,
    required=False,
    help="Frequency limits (min/max) for spectrogram y-axis (Hz).",
)
@click.option(
    "--spectrogram-yticks",
    "spectrogram_yticks",
    multiple=True,
    type=int,
    default=None,
    required=False,
    help="Y-tick position for spectrogram",
)
@click.option(
    "--spectrogram-stft-nfft",
    "spectrogram_stft_nfft",
    type=int,
    default=None,
    required=False,
    help="NFFT for the spectrogram STFT calculation.",
)
@click.option(
    "--plot-usv-segments/--no-plot-usv-segments",
    "plot_usv_segments_bool",
    default=None,
    help="Display USV assignments on the spectrogram.",
)
@click.option(
    "--usv-segments-ypos",
    "usv_segments_ypos",
    type=int,
    default=None,
    required=False,
    help="Y-axis position for USV segment markers (Hz).",
)
@click.option(
    "--usv-segments-lw",
    "usv_segments_lw",
    type=float,
    default=None,
    required=False,
    help="Line width for USV segment markers.",
)
@click.pass_context
def visualize_3D_data_cli(
    ctx, root_directory, arena_directory, exp_id, speaker_audio_file, **kwargs
) -> None:
    """
    Description
    ----------
    A command-line tool to plot/animate 3D tracked mice.
    ----------

    Parameters
    ----------
    ----------

    Returns
    ----------
    ----------
    """

    parameters_lists = [
        "brain_areas",
        "other",
        "raster_special_units",
        "beh_features_to_plot",
        "special_beh_features",
        "spectrogram_power_limit",
        "spectrogram_frequency_limit",
        "spectrogram_yticks",
    ]

    provided_params = [
        key
        for key in kwargs
        if ctx.get_parameter_source(key) == ParameterSource.COMMANDLINE
    ]

    visualizations_settings_parameter_dict = modify_settings_json_for_cli(
        ctx=ctx,
        provided_params=provided_params,
        parameters_lists=parameters_lists,
        settings_dict="visualizations_settings",
    )

    Create3DVideo(
        root_directory=root_directory,
        arena_directory=arena_directory,
        exp_id=exp_id,
        speaker_audio_file=speaker_audio_file,
        visualizations_parameter_dict=visualizations_settings_parameter_dict,
        message_output=print,
    ).visualize_in_video()


@click.command(name="generate-rm-figs")
@click.option(
    "--root-directory",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default=None,
    required=True,
    help="Session root directory path.",
)
@click.option(
    "--smoothing-sd",
    "smoothing_sd",
    type=float,
    default=None,
    required=False,
    help="Standard deviation of smoothing (in bins).",
)
@click.option(
    "--occ-threshold",
    "occ_threshold",
    type=float,
    default=None,
    required=False,
    help="Minimum acceptable occupancy (in s).",
)
@click.pass_context
def generate_rm_figures_cli(ctx, root_directory, **kwargs) -> None:
    """
    Description
    ----------
    A command-line tool to generate behavioral tuning curve figures.
    ----------

    Parameters
    ----------
    ----------

    Returns
    ----------
    ----------
    """

    provided_params = [
        key
        for key in kwargs
        if ctx.get_parameter_source(key) == ParameterSource.COMMANDLINE
    ]

    visualizations_settings_parameter_dict = modify_settings_json_for_cli(
        ctx=ctx,
        provided_params=provided_params,
        settings_dict="visualizations_settings",
    )
    RatemapFigureMaker(
        root_directory=root_directory,
        visualizations_parameter_dict=visualizations_settings_parameter_dict,
        message_output=print,
    ).neuronal_tuning_figures()
