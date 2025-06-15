"""
@author: bartulem
Preprocess data after running experiments.
"""

import os
import traceback
from click.core import ParameterSource
from datetime import datetime
from .anipose_operations import ConvertTo3D
from .assign_vocalizations import Vocalocator
from .cli_utils import *
from .das_inference import FindMouseVocalizations
from .extract_phidget_data import Gatherer
from .modify_files import Operator
from .prepare_cluster_job import PrepareClusterJob
from .preprocessing_plot import SummaryPlotter
from .send_email import Messenger
from .synchronize_files import Synchronizer


class Stylist:

    def __init__(self,
                 exp_settings_dict: dict = None,
                 input_parameter_dict: dict = None,
                 root_directories: list = None,
                 message_output: callable = None) -> None:

        """
        Initializes the Stylist class.

        Parameter
        ---------
        exp_settings_dict (dict)
            Experimental settings; defaults to None.
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
            with open((pathlib.Path(__file__).parent / '_parameter_settings/processing_settings.json'), 'r') as json_file:
                self.root_directories = json.load(json_file)['preprocess_data']['root_directories']
        else:
            self.root_directories = root_directories

        if input_parameter_dict is None:
            with open((pathlib.Path(__file__).parent / '_parameter_settings/processing_settings.json'), 'r') as json_file:
                self.input_parameter_dict = json.load(json_file)
        else:
            self.input_parameter_dict = input_parameter_dict

        if exp_settings_dict is None:
            self.exp_settings_dict = None
        else:
            self.exp_settings_dict = exp_settings_dict

        if message_output is None:
            self.message_output = print
        else:
            self.message_output = message_output

    def prepare_data_for_analyses(self) -> None:
        """
        Description
        ----------
        This method performs the following data preprocessing steps:
        (1) concatenates video files (necessary for sessions >15 min)
        (2) re-encodes videos (compresses and adjusts sampling rate)
        (3) prepares SLEAP inference cluster job for videos
        (4) converts SLP to H5 files after proofreading
        (5) conducts SLEAP-Anipose calibration
        (6) conducts SLEAP-Anipose triangulation
        (7) conducts SLEAP-Anipose coordinate transformation
        (8) splits multichannel to single-channel audio files
        (9) crops audio files to match video length
        (10) cleans audio background with harmonic-percussive source separation
        (11) band-pass filters audio files
        (12) vertically stacks all audio files in one memmap file
        (13) conducts DAS inference on audio data
        (14) summarizes DAS findings across different audio channels
        (15) extracts phidget-measured data during the experiment
        (16) checks audio-video sync using  Arduino-controlled IR lights pulses
        (17) plots the summary of AV sync and phidget measurements
        (18) concatenates e-phys binary files
        (19) splits e-phys clusters to individual sessions
        (20) conducts ephys-video sync validation

        ----------

        Parameters
        ----------
        ----------

        Returns
        ----------
        preprocessing_plot (.svg)
            Figure summarizing the preprocessing of experimental data;
            the figure is saved in /mnt/LAB/CUP-subdirectory/root_directory/sync/root_directory_summary.svg.
        ----------
        """

        Messenger(message_output=self.message_output,
                  receivers=self.input_parameter_dict['send_email']['Messenger']['send_message']['receivers'],
                  exp_settings_dict=self.exp_settings_dict).send_message(subject=f"{self.input_parameter_dict['send_email']['Messenger']['processing_pc_choice']} PC is busy, do NOT attempt to remote in!",
                                                                         message=f"Data preprocessing in progress, started at "
                                                                                 f"{datetime.now().hour:02d}:{datetime.now().minute:02d}.{datetime.now().second:02d} "
                                                                                 f"and run by @{self.input_parameter_dict['send_email']['Messenger']['experimenter']}. "
                                                                                 f"You will be notified upon completion. \n \n ***This is an automatic e-mail, please do NOT respond.***")

        # analyze data in all root directories at once
        if self.input_parameter_dict['processing_booleans']['conduct_ephys_file_chaining'] or self.input_parameter_dict['processing_booleans']['split_cluster_spikes'] or self.input_parameter_dict['processing_booleans']['prepare_sleap_cluster']:
            try:
                self.message_output(f"Preprocessing data started at: "
                                    f"{datetime.now().hour:02d}:{datetime.now().minute:02d}.{datetime.now().second:02d}")

                # # # concatenate e-phys files
                if self.input_parameter_dict['processing_booleans']['conduct_ephys_file_chaining']:
                    Operator(root_directory=self.root_directories,
                             input_parameter_dict=self.input_parameter_dict,
                             message_output=self.message_output).concatenate_binary_files()

                # # # split clusters to individual sessions
                if self.input_parameter_dict['processing_booleans']['split_cluster_spikes']:
                    Operator(root_directory=self.root_directories,
                             input_parameter_dict=self.input_parameter_dict,
                             message_output=self.message_output).split_clusters_to_sessions()

                # # # prepare SLEAP cluster job
                if self.input_parameter_dict['processing_booleans']['prepare_sleap_cluster']:
                    PrepareClusterJob(root_directory=self.root_directories,
                                      input_parameter_dict=self.input_parameter_dict,
                                      message_output=self.message_output).video_list_to_txt()

                self.message_output(f"Preprocessing data finished at: "
                                    f"{datetime.now().hour:02d}:{datetime.now().minute:02d}.{datetime.now().second:02d}")

            except (OSError, RuntimeError, TypeError, IndexError, IOError, EOFError, TimeoutError, NameError, KeyError, ValueError, AttributeError):
                self.message_output(traceback.format_exc())

        # analyze data in each root directory separately
        else:
            for one_directory_idx, one_directory in enumerate(self.root_directories):
                try:
                    self.message_output(f"Preprocessing data in {one_directory} started at: "
                                        f"{datetime.now().hour:02d}:{datetime.now().minute:02d}.{datetime.now().second:02d}.")

                    # # # configure video properties via ffmpeg
                    if self.input_parameter_dict['processing_booleans']['conduct_video_concatenation']:
                        Operator(root_directory=one_directory,
                                 input_parameter_dict=self.input_parameter_dict,
                                 message_output=self.message_output).concatenate_video_files()

                    if self.input_parameter_dict['processing_booleans']['conduct_video_fps_change']:
                        Operator(root_directory=one_directory,
                                 input_parameter_dict=self.input_parameter_dict,
                                 message_output=self.message_output).rectify_video_fps(conduct_concat=self.input_parameter_dict['processing_booleans']['conduct_video_concatenation'])

                    # # # convert multichannel to single channel files
                    if self.input_parameter_dict['processing_booleans']['conduct_audio_multichannel_to_single_ch']:
                        if len(os.listdir(f"{one_directory}{os.sep}audio{os.sep}original")) == 0:
                            Operator(root_directory=one_directory,
                                     input_parameter_dict=self.input_parameter_dict,
                                     message_output=self.message_output).multichannel_to_channel_audio()

                    # # # crop audio files to match video
                    if self.input_parameter_dict['processing_booleans']['conduct_audio_cropping']:
                        Synchronizer(root_directory=one_directory,
                                     input_parameter_dict=self.input_parameter_dict,
                                     message_output=self.message_output).crop_wav_files_to_video()

                    # # # check audio-video sync
                    if self.input_parameter_dict['processing_booleans']['conduct_audio_video_sync']:
                        phidget_data_dictionary = Gatherer(root_directory=one_directory,
                                                           input_parameter_dict=self.input_parameter_dict).prepare_data_for_analyses()

                        ipi_discrepancy_dict = Synchronizer(root_directory=one_directory,
                                                            input_parameter_dict=self.input_parameter_dict,
                                                            message_output=self.message_output).find_audio_sync_trains()

                        SummaryPlotter(root_directory=one_directory,
                                       input_parameter_dict=self.input_parameter_dict).preprocessing_summary(ipi_discrepancy_dict=ipi_discrepancy_dict,
                                                                                                             phidget_data_dictionary=phidget_data_dictionary)

                    # # # check e-phys-video sync
                    if self.input_parameter_dict['processing_booleans']['conduct_ephys_video_sync']:
                        Synchronizer(root_directory=one_directory,
                                     input_parameter_dict=self.input_parameter_dict,
                                     message_output=self.message_output).validate_ephys_video_sync()

                    # # # harmonic-percussive source separation
                    if self.input_parameter_dict['processing_booleans']['conduct_hpss']:
                        Operator(root_directory=one_directory,
                                 input_parameter_dict=self.input_parameter_dict,
                                 message_output=self.message_output).hpss_audio()

                    # # # band-pass filter audio files in memmap
                    if self.input_parameter_dict['processing_booleans']['conduct_audio_filtering']:
                        Operator(root_directory=one_directory,
                                 input_parameter_dict=self.input_parameter_dict,
                                 message_output=self.message_output).filter_audio_files()

                    # # # stack audio files in memmap
                    if self.input_parameter_dict['processing_booleans']['conduct_audio_to_mmap']:
                        Operator(root_directory=one_directory,
                                 input_parameter_dict=self.input_parameter_dict,
                                 message_output=self.message_output).concatenate_audio_files()

                    # # # convert .slp to .h5 files
                    if self.input_parameter_dict['processing_booleans']['sleap_h5_conversion']:
                        ConvertTo3D(root_directory=one_directory,
                                    input_parameter_dict=self.input_parameter_dict,
                                    message_output=self.message_output).sleap_file_conversion()

                    # # # conduct Anipose calibration
                    if self.input_parameter_dict['processing_booleans']['anipose_calibration']:
                        ConvertTo3D(root_directory=one_directory,
                                    input_parameter_dict=self.input_parameter_dict,
                                    message_output=self.message_output).conduct_anipose_calibration()

                    # # # conduct Anipose triangulation
                    if self.input_parameter_dict['processing_booleans']['anipose_triangulation']:
                        ConvertTo3D(root_directory=one_directory,
                                    input_parameter_dict=self.input_parameter_dict,
                                    message_output=self.message_output).conduct_anipose_triangulation()

                    # # # conduct coordinate transformation of Anipose data
                    if self.input_parameter_dict['processing_booleans']['anipose_trm']:
                        if (self.input_parameter_dict['anipose_operations']['ConvertTo3D']['conduct_anipose_triangulation']['triangulate_arena_points_bool'] or
                                len(self.input_parameter_dict['anipose_operations']['ConvertTo3D']['translate_rotate_metric']['experimental_codes']) == len(self.root_directories)):
                            ConvertTo3D(root_directory=one_directory,
                                        input_parameter_dict=self.input_parameter_dict,
                                        message_output=self.message_output).translate_rotate_metric(session_idx=one_directory_idx)
                        else:
                            self.message_output("Please provide the experimental code for each session in the root directory, their number does not match.")

                    # # # conduct DAS inference on audio data
                    if self.input_parameter_dict['processing_booleans']['das_infer']:
                        FindMouseVocalizations(root_directory=one_directory,
                                               input_parameter_dict=self.input_parameter_dict,
                                               message_output=self.message_output).das_command_line_inference()

                    # # # conduct DAS summary
                    if self.input_parameter_dict['processing_booleans']['das_summarize']:
                        FindMouseVocalizations(root_directory=one_directory,
                                               input_parameter_dict=self.input_parameter_dict,
                                               message_output=self.message_output).summarize_das_findings()

                    # # # prepare data for vocal assignment
                    if self.input_parameter_dict['processing_booleans']['prepare_assign_vocalizations']:
                        Vocalocator(root_directory=one_directory,
                                    input_parameter_dict=self.input_parameter_dict,
                                    message_output=self.message_output).prepare_for_vocalocator()

                    # # # assign vocalizations to mice
                    if self.input_parameter_dict['processing_booleans']['assign_vocalizations']:
                        Vocalocator(root_directory=one_directory,
                                    input_parameter_dict=self.input_parameter_dict,
                                    message_output=self.message_output).run_vocalocator()

                    self.message_output(f"Preprocessing data in {one_directory} finished at: "
                                        f"{datetime.now().hour:02d}:{datetime.now().minute:02d}.{datetime.now().second:02d}.")

                except (OSError, RuntimeError, TypeError, IndexError, IOError, EOFError, TimeoutError, NameError, KeyError, ValueError, AttributeError):
                    self.message_output(traceback.format_exc())

        Messenger(message_output=self.message_output,
                  no_receivers_notification=False,
                  receivers=self.input_parameter_dict['send_email']['Messenger']['send_message']['receivers'],
                  exp_settings_dict=self.exp_settings_dict).send_message(subject=f"{self.input_parameter_dict['send_email']['Messenger']['processing_pc_choice']} PC is available again, processing has been completed",
                                                                         message=f"Data preprocessing has been completed at "
                                                                                 f"{datetime.now().hour:02d}:{datetime.now().minute:02d}.{datetime.now().second:02d} "
                                                                                 f"by @{self.input_parameter_dict['send_email']['Messenger']['experimenter']}. "
                                                                                 f"You will be notified about further PC usage "
                                                                                 f"should it occur. \n \n ***This is an automatic e-mail, please do NOT respond.***")

@click.command(name="concatenate-video-files")
@click.option('--root-directory', type=click.Path(exists=True, file_okay=False, dir_okay=True), required=True, help='Session root directory path.')
@click.option('--camera-serial', 'concatenate_camera_serial_num', multiple=True, type=str, default=None, required=False, help='Camera serial number(s).')
@click.option('--extension', 'concatenate_video_extension', type=str, default=None, required=False, help='Video file extension.')
@click.option('--output-name', 'concatenated_video_name', type=str, default=None, required=False, help='Name of the concatenated file.')
@click.pass_context
def concatenate_video_files_cli(ctx, root_directory, **kwargs) -> None:
    """
    Description
    ----------
    A command-line tool to concatenate video files from multiple cameras.
    ----------

    Parameters
    ----------
    ----------

    Returns
    ----------
    ----------
    """

    parameters_lists = ['concatenate_camera_serial_num']

    provided_params = [key for key in kwargs if ctx.get_parameter_source(key) == ParameterSource.COMMANDLINE]

    processing_settings_dict = modify_settings_json_for_cli(
        ctx=ctx,
        parameters_lists=parameters_lists,
        provided_params=provided_params,
        settings_dict='processing_settings'
    )

    Operator(
        root_directory=root_directory,
        input_parameter_dict=processing_settings_dict
    ).concatenate_video_files()

@click.command(name="rectify-video-fps")
@click.option('--root-directory', type=click.Path(exists=True, file_okay=False, dir_okay=True), required=True, help='Session root directory path.')
@click.option('--camera-serial', 'encode_camera_serial_num', multiple=True, type=str, default=None, required=False, help='Camera serial number(s).')
@click.option('--target-file', 'conversion_target_file', type=str, default=None, required=False, help='Name of the target video file.')
@click.option('--extension', 'encode_video_extension', type=str, default=None, required=False, help='Video file extension.')
@click.option('--crf', 'constant_rate_factor', type=int, default=None, required=False, help='FFMPEG -crf (e.g., 16).')
@click.option('--preset', 'encoding_preset', type=click.Choice(['ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium', 'slow', 'slower', 'veryslow'], case_sensitive=False), default=None, required=False, help='FFMPEG encoding speed preset.')
@click.option('--delete-old-file/--no-delete-old-file', 'delete_old_file', default=None, help='Delete the original file after encoding.')
@click.option('--conduct-concat/--no-conduct-concat', 'conduct_concat', default=None, help='Indicate if prior concatenation was performed')
@click.pass_context
def rectify_video_fps_cli(ctx, root_directory, conduct_concat, **kwargs) -> None:
    """
    Description
    ----------
    A command-line tool to re-encode videos to an appropriate frame rate.
    ----------

    Parameters
    ----------
    ----------

    Returns
    ----------
    ----------
    """

    parameters_lists = ['encode_camera_serial_num']

    provided_params = [key for key in kwargs if ctx.get_parameter_source(key) == ParameterSource.COMMANDLINE]

    processing_settings_dict = modify_settings_json_for_cli(
        ctx=ctx,
        parameters_lists=parameters_lists,
        provided_params=provided_params,
        settings_dict='processing_settings'
    )

    Operator(
        root_directory=root_directory,
        input_parameter_dict=processing_settings_dict
    ).rectify_video_fps(conduct_concat=conduct_concat)

@click.command(name="multichannel-to-single-ch")
@click.option('--root-directory', type=click.Path(exists=True, file_okay=False, dir_okay=True), required=True, help='Session root directory path.')
def multichannel_to_channel_audio_cli(root_directory) -> None:
    """
    Description
    ----------
    A command-line tool to split multichannel audio files into single-channel files.
    ----------

    Parameters
    ----------
    ----------

    Returns
    ----------
    ----------
    """

    Operator(root_directory=root_directory).multichannel_to_channel_audio()

@click.command(name="crop-wav-files")
@click.option('--root-directory', type=click.Path(exists=True, file_okay=False, dir_okay=True), required=True, help='Session root directory path.')
@click.option('--trigger-device', 'device_receiving_input', type=click.Choice(['both', 'm', 'r'], case_sensitive=False), default=None, required=False, help='USGH device(s) receiving triggerbox input.')
@click.option('--trigger-channel', 'triggerbox_ch_receiving_input', type=int, default=None, required=False, help='USGH channel receiving triggerbox input.')
@click.pass_context
def crop_wav_files_to_video_cli(ctx, root_directory, **kwargs) -> None:
    """
    Description
    ----------
    A command-line tool to crop audio WAV files to match video length.
    ----------

    Parameters
    ----------
    ----------

    Returns
    ----------
    ----------
    """

    provided_params = [key for key in kwargs if ctx.get_parameter_source(key) == ParameterSource.COMMANDLINE]

    processing_settings_dict = modify_settings_json_for_cli(
        ctx=ctx,
        provided_params=provided_params,
        settings_dict='processing_settings'
    )

    Synchronizer(
        root_directory=root_directory,
        input_parameter_dict=processing_settings_dict
    ).crop_wav_files_to_video()

@click.command(name="av-sync-check")
@click.option('--root-directory', type=click.Path(exists=True, file_okay=False, dir_okay=True), required=True, help='Session root directory path.')
@click.option('--extra-camera', 'extra_data_camera', type=str, default=None, required=False, help='Camera serial number for extra data.')
@click.option('--audio-sync-ch', 'sync_ch_receiving_input', type=int, default=None, required=False, help='Audio channel receiving sync input.')
@click.option('--exact-frame-times/--no-exact-frame-times', 'extract_exact_video_frame_times_bool', default=None, help='Extract exact video frame times.')
@click.option('--nidq-sr', type=float, default=None, required=False, help='NI-DAQ sampling rate (Hz).')
@click.option('--nidq-channels', 'nidq_num_channels', type=int, default=None, required=False, help='Number of NI-DAQ channels.')
@click.option('--nidq-trigger-bit', 'nidq_triggerbox_input_bit_position', type=int, default=None, required=False, help='NI-DAQ triggerbox input bit position.')
@click.option('--nidq-sync-bit', 'nidq_sync_input_bit_position', type=int, default=None, required=False, help='NI-DAQ sync input bit position.')
@click.option('--video-sync-camera', 'sync_camera_serial_num', multiple=True, type=str, default=None, required=False, help='Camera serial number for video sync.')
@click.option('--led-version', 'led_px_version', type=str, default=None, required=False, help='Version of the LED pixel used for sync.')
@click.option('--led-dev', 'led_px_dev', type=int, default=None, required=False, help='LED pixel deviation value.')
@click.option('--video-extension', 'sync_video_extension', type=str, default=None, required=False, help='Video extension for sync files.')
@click.option('--intensity-thresh', 'relative_intensity_threshold', type=float, default=None, required=False, help='Relative intensity threshold for LED detection.')
@click.option('--ms-tolerance', 'millisecond_divergence_tolerance', type=int, default=None, required=False, help='Divergence tolerance (in ms).')
@click.pass_context
def av_sync_check_cli(ctx, root_directory, **kwargs) -> None:
    """
    Description
    ----------
    A tool to check audio-video sync and generate a summary plot.
    ----------

    Parameters
    ----------
    ----------

    Returns
    ----------
    ----------
    """

    parameters_lists=['sync_camera_serial_num']

    provided_params = [key for key in kwargs if ctx.get_parameter_source(key) == ParameterSource.COMMANDLINE]

    processing_settings_dict = modify_settings_json_for_cli(
        ctx=ctx,
        parameters_lists=parameters_lists,
        provided_params=provided_params,
        settings_dict='processing_settings'
    )

    phidget_data_dictionary = Gatherer(
        root_directory=root_directory,
        input_parameter_dict=processing_settings_dict
    ).prepare_data_for_analyses()

    ipi_discrepancy_dict = Synchronizer(
        root_directory=root_directory,
        input_parameter_dict=processing_settings_dict
    ).find_audio_sync_trains()

    SummaryPlotter(
        root_directory=root_directory,
        input_parameter_dict=processing_settings_dict
    ).preprocessing_summary(
        ipi_discrepancy_dict=ipi_discrepancy_dict,
        phidget_data_dictionary=phidget_data_dictionary
    )

click.command(name="ev-sync-check")
@click.option('--root-directory', type=click.Path(exists=True, file_okay=False, dir_okay=True), required=True, help='Session root directory path.')
@click.option('--file-type', 'npx_file_type', type=click.Choice(['ap', 'lf'], case_sensitive=False), default=None, required=False, help='Neuropixels file type (ap or lf).')
@click.option('--tolerance', 'npx_ms_divergence_tolerance', type=float, default=None, required=False, help='Divergence tolerance (in ms).')
@click.pass_context
def ev_sync_check_cli(ctx, root_directory, **kwargs) -> None:
    """
    Description
    ----------
    A command-line tool to validate ephys-video synchronization.
    ----------

    Parameters
    ----------
    ----------

    Returns
    ----------
    ----------
    """

    provided_params = [key for key in kwargs if ctx.get_parameter_source(key) == ParameterSource.COMMANDLINE]

    processing_settings_dict = modify_settings_json_for_cli(
        ctx=ctx,
        provided_params=provided_params,
        settings_dict='processing_settings'
    )

    Synchronizer(
        root_directory=root_directory,
        input_parameter_dict=processing_settings_dict
    ).validate_ephys_video_sync()

@click.command(name="hpss-audio")
@click.option('--root-directory', type=click.Path(exists=True, file_okay=False, dir_okay=True), required=True, help='Session root directory path.')
@click.option('--stft-params', 'stft_window_length_hop_size', nargs=2, type=int, default=None, required=False, help='STFT window length and hop size.')
@click.option('--kernel-size', 'kernel_size', nargs=2, type=int, default=None, required=False, help='Median filter kernel size (harmonic, percussive).')
@click.option('--power', 'hpss_power', type=float, default=None, required=False, help='HPSS power parameter.')
@click.option('--margin', 'margin', nargs=2, type=int, default=None, required=False, help='HPSS margin (harmonic, percussive).')
@click.pass_context
def hpss_audio_cli(ctx, root_directory, **kwargs) -> None:
    """
    Description
    ----------
    A command-line tool to perform HPSS on audio files.
    ----------

    Parameters
    ----------
    ----------

    Returns
    ----------
    ----------
    """

    parameters_lists = ['stft_window_length_hop_size', 'kernel_size', 'margin']

    provided_params = [key for key in kwargs if ctx.get_parameter_source(key) == ParameterSource.COMMANDLINE]

    processing_settings_dict = modify_settings_json_for_cli(
        ctx=ctx,
        parameters_lists=parameters_lists,
        provided_params=provided_params,
        settings_dict='processing_settings'
    )

    Operator(
        root_directory=root_directory,
        input_parameter_dict=processing_settings_dict
    ).hpss_audio()

@click.command(name="bp-filter-audio")
@click.option('--root-directory', type=click.Path(exists=True, file_okay=False, dir_okay=True), required=True, help='Session root directory path.')
@click.option('--format', 'filter_audio_format', type=str, default=None, required=False, help='Audio file format.')
@click.option('--dirs', 'filter_dirs', multiple=True, type=str, default=None, required=False, help='Directory/ies containing files to filter.')
@click.option('--freq-bounds', 'filter_freq_bounds', nargs=2, type=int, default=None, required=False, help='Frequency bounds for the band-pass filter (Hz).')
@click.pass_context
def bp_filter_audio_files_cli(ctx, root_directory, **kwargs) -> None:
    """
    Description
    ----------
    A command-line tool to band-pass filter audio files.
    ----------

    Parameters
    ----------
    ----------

    Returns
    ----------
    ----------
    """

    parameters_lists = ['filter_dirs', 'filter_freq_bounds']

    provided_params = [key for key in kwargs if ctx.get_parameter_source(key) == ParameterSource.COMMANDLINE]

    processing_settings_dict = modify_settings_json_for_cli(
        ctx=ctx,
        parameters_lists=parameters_lists,
        provided_params=provided_params,
        settings_dict='processing_settings'
    )

    Operator(
        root_directory=root_directory,
        input_parameter_dict=processing_settings_dict
    ).filter_audio_files()

@click.command(name="concatenate-audio-files")
@click.option('--root-directory', type=click.Path(exists=True, file_okay=False, dir_okay=True), required=True, help='Session root directory path.')
@click.option('--format', 'concatenate_audio_format', type=str, default=None, required=False, help='Audio file format.')
@click.option('--dirs', 'concat_dirs', multiple=True, type=str, default=None, required=False, help='Directory/ies to search for files to concatenate.')
@click.pass_context
def concatenate_audio_files_cli(ctx, root_directory, **kwargs) -> None:
    """
    Description
    ----------
    A command-line tool to vertically stack audio files into a single memmap file.
    ----------

    Parameters
    ----------
    ----------

    Returns
    ----------
    ----------
    """

    parameters_lists = ['concat_dirs']

    provided_params = [key for key in kwargs if ctx.get_parameter_source(key) == ParameterSource.COMMANDLINE]

    processing_settings_dict = modify_settings_json_for_cli(
        ctx=ctx,
        parameters_lists=parameters_lists,
        provided_params=provided_params,
        settings_dict='processing_settings'
    )

    Operator(
        root_directory=root_directory,
        input_parameter_dict=processing_settings_dict
    ).concatenate_audio_files()

@click.command(name="sleap-to-h5")
@click.option('--root-directory', type=click.Path(exists=True, file_okay=False, dir_okay=True), required=True, help='Session root directory path.')
@click.option('--env-name', 'sleap_conda_env_name', type=str, default=None, required=False, help='SLEAP conda environment.')
@click.pass_context
def sleap_file_conversion_cli(ctx, root_directory, **kwargs) -> None:
    """
    Description
    ----------
    A command-line tool to convert SLEAP (.slp) files to HDF5 (.h5) files.
    ----------

    Parameters
    ----------
    ----------

    Returns
    ----------
    ----------
    """

    provided_params = [key for key in kwargs if ctx.get_parameter_source(key) == ParameterSource.COMMANDLINE]

    processing_settings_dict = modify_settings_json_for_cli(
        ctx=ctx,
        provided_params=provided_params,
        settings_dict='processing_settings'
    )

    ConvertTo3D(
        root_directory=root_directory,
        input_parameter_dict=processing_settings_dict
    ).sleap_file_conversion()

@click.command(name="anipose-calibrate")
@click.option('--root-directory', type=click.Path(exists=True, file_okay=False, dir_okay=True), required=True, help='Session root directory path.')
@click.option('--board-provided', 'board_provided_bool', is_flag=True, help='Indicate that the calibration board is provided.')
@click.option('--board-dims', 'board_xy', nargs=2, type=int, default=None, required=False, help='Checkerboard dimensions (squares_x, squares_y).')
@click.option('--square-len', type=int, default=None, required=False, help='Length of a checkerboard square (mm).')
@click.option('--marker-params', 'marker_len_bits', nargs=2, type=float, default=None, required=False, help='ArUco marker length (mm) and dictionary bits.')
@click.option('--dict-size', type=int, default=None, required=False, help='Size of the ArUco dictionary.')
@click.option('--img-dims', 'img_width_height', nargs=2, type=int, default=None, required=False, help='Image dimensions (width, height) in pixels.')
@click.pass_context
def conduct_anipose_calibration_cli(ctx, root_directory, **kwargs) -> None:
    """
    Description
    ----------
    A command-line tool to conduct Anipose camera calibration.
    ----------

    Parameters
    ----------
    ----------

    Returns
    ----------
    ----------
    """

    parameters_lists = ['board_xy', 'marker_len_bits', 'img_width_height']

    provided_params = [key for key in kwargs if ctx.get_parameter_source(key) == ParameterSource.COMMANDLINE]

    processing_settings_dict = modify_settings_json_for_cli(
        ctx=ctx,
        parameters_lists=parameters_lists,
        provided_params=provided_params,
        settings_dict='processing_settings'
    )

    ConvertTo3D(
        root_directory=root_directory,
        input_parameter_dict=processing_settings_dict
    ).conduct_anipose_calibration()

@click.command(name="anipose-triangulate")
@click.option('--root-directory', type=click.Path(exists=True, file_okay=False, dir_okay=True), required=True, help='Session root directory path.')
@click.option('--cal-directory', 'calibration_file_loc', type=click.Path(exists=True, file_okay=False, dir_okay=True), default=None, required=True, help='Path to the Anipose calibration session.')
@click.option('--arena-points/--no-arena-points', 'triangulate_arena_points_bool', default=None, help='Triangulate arena points instead of animal points.')
@click.option('--frame-restriction', type=int, default=None, required=False, help='Restrict triangulation to a specific number of frames.')
@click.option('--exclude-views', 'excluded_views', multiple=True, type=str, default=None, required=False, help='Camera views to exclude from triangulation.')
@click.option('--display-progress/--no-display-progress', 'display_progress_bool', default=None, help='Display the progress bar during triangulation.')
@click.option('--use-ransac/--no-use-ransac', 'ransac_bool', default=None, help='Use RANSAC for robust triangulation.')
@click.option('--rigid-constraint', 'rigid_body_constraints', type=StringTuple(), multiple=True, default=None, required=False, help='Pair(s) of nodes for a rigid constraint.')
@click.option('--weak-constraint', 'weak_body_constraints', type=StringTuple(), multiple=True, default=None, required=False, help='Pair(s) of nodes for a weak constraint.')
@click.option('--smooth-scale', type=float, default=None, required=False, help='Scaling factor for smoothing.')
@click.option('--weight-weak', type=int, default=None, required=False, help='Weight for weak constraints.')
@click.option('--weight-rigid', type=int, default=None, required=False, help='Weight for rigid constraints.')
@click.option('--reprojection-threshold', 'reprojection_error_threshold', type=int, default=None, required=False, help='Reprojection error threshold.')
@click.option('--regularization', 'regularization_function', type=click.Choice(['l1', 'l2'], case_sensitive=False), default=None, required=False, help='Regularization function to use.')
@click.option('--n-deriv-smooth', type=int, default=None, required=False, help='Number of derivatives to use for smoothing.')
@click.pass_context
def conduct_anipose_triangulation_cli(ctx, root_directory, calibration_file_loc, **kwargs) -> None:
    """
    Description
    ----------
    A command-line tool to conduct Anipose 3D triangulation.
    ----------

    Parameters
    ----------
    ----------

    Returns
    ----------
    ----------
    """

    parameters_lists = ['excluded_views', 'rigid_body_constraints', 'weak_body_constraints']

    provided_params = [key for key in kwargs if ctx.get_parameter_source(key) == ParameterSource.COMMANDLINE]

    processing_settings_dict = modify_settings_json_for_cli(
        ctx=ctx,
        parameters_lists=parameters_lists,
        provided_params=provided_params,
        settings_dict='processing_settings'
    )

    ConvertTo3D(
        root_directory=root_directory,
        input_parameter_dict=processing_settings_dict
    ).conduct_anipose_triangulation()

@click.command(name="anipose-trm")
@click.option('--root-directory', type=click.Path(exists=True, file_okay=False, dir_okay=True), required=True, help='Session root directory path.')
@click.option('--exp-code', 'experimental_codes', type=str, default=None, required=True, help='Experimental code.')
@click.option('--arena-directory', 'original_arena_file_loc', type=click.Path(exists=True, file_okay=False, dir_okay=True), default=None, required=True, help='Path to the original arena session.')
@click.option('--save-data-for', 'save_transformed_data', type=click.Choice(['animal', 'arena'], case_sensitive=False), default=None, required=False, help='Data to save after transformation.')
@click.option('--delete-original/--no-delete-original', 'delete_original_h5', help='If set, deletes the original HDF5 file.')
@click.option('--ref-len', 'static_reference_len', type=float, default=None, required=False, help='Length of the static reference object (e.g., arena side).')
@click.pass_context
def translate_rotate_metric_cli(ctx, root_directory, experimental_codes, original_arena_file_loc, **kwargs) -> None:
    """
    Description
    ----------
    A command-line tool to translate, rotate, and scale 3D point data.
    ----------

    Parameters
    ----------
    ----------

    Returns
    ----------
    ----------
    """

    parameters_lists = ['experimental_codes']

    provided_params = [key for key in kwargs if ctx.get_parameter_source(key) == ParameterSource.COMMANDLINE]

    processing_settings_dict = modify_settings_json_for_cli(
        ctx=ctx,
        parameters_lists=parameters_lists,
        provided_params=provided_params,
        settings_dict='processing_settings'
    )

    ConvertTo3D(
        root_directory=root_directory,
        input_parameter_dict=processing_settings_dict
    ).translate_rotate_metric(session_idx=0)

@click.command(name="das-infer")
@click.option('--root-directory', type=click.Path(exists=True, file_okay=False, dir_okay=True), required=True, help='Session root directory path.')
@click.option('--env-name', 'das_conda_env_name', type=str, default=None, required=False, help='Name of the DAS conda environment.')
@click.option('--model-dir', 'das_model_directory', type=click.Path(exists=True, file_okay=False, dir_okay=True), default=None, required=False, help='Directory of the DAS model.')
@click.option('--model-name', 'model_name_base', type=str, default=None, required=False, help='Base name of the DAS model.')
@click.option('--output-type', 'output_file_type', type=click.Choice(['csv', 'hdf5'], case_sensitive=False), default=None, required=False, help='Output file type for DAS predictions.')
@click.option('--confidence-thresh', 'segment_confidence_threshold', type=float, default=None, required=False, help='Confidence threshold for segment detection.')
@click.option('--min-len', 'segment_minlen', type=float, default=None, required=False, help='Minimum length for a detected segment (s).')
@click.option('--fill-gap', 'segment_fillgap', type=float, default=None, required=False, help='Gap duration to fill between segments (s).')
@click.pass_context
def das_command_line_inference_cli(ctx, root_directory, **kwargs):
    """
    Description
    ----------
    A command-line tool to run DAS inference on audio data.
    ----------

    Parameters
    ----------
    ----------

    Returns
    ----------
    ----------
    """

    provided_params = [key for key in kwargs if ctx.get_parameter_source(key) == ParameterSource.COMMANDLINE]

    processing_settings_dict = modify_settings_json_for_cli(
        ctx=ctx,
        provided_params=provided_params,
        settings_dict='processing_settings'
    )

    FindMouseVocalizations(
        root_directory=root_directory,
        input_parameter_dict=processing_settings_dict
    ).das_command_line_inference()

@click.command(name="das-summarize")
@click.option('--root-directory', type=click.Path(exists=True, file_okay=False, dir_okay=True), required=True, help='Session root directory path.')
@click.option('--win-len', 'len_win_signal', type=int, default=None, required=False, help='Window length of the signal.')
@click.option('--freq-cutoff', 'low_freq_cutoff', type=int, default=None, required=False, help='Low frequency cutoff (Hz).')
@click.option('--corr-cutoff', 'noise_corr_cutoff_min', type=float, default=None, required=False, help='Minimum noise correlation cutoff.')
@click.option('--var-cutoff', 'noise_var_cutoff_max', type=float, default=None, required=False, help='Maximum noise variance cutoff.')
@click.pass_context
def summarize_das_findings_cli(ctx, root_directory, **kwargs):
    """
    Description
    ----------
    A command-line tool to summarize DAS inference findings.
    ----------

    Parameters
    ----------
    ----------

    Returns
    ----------
    ----------
    """

    provided_params = [key for key in kwargs if ctx.get_parameter_source(key) == ParameterSource.COMMANDLINE]

    processing_settings_dict = modify_settings_json_for_cli(
        ctx=ctx,
        provided_params=provided_params,
        settings_dict='processing_settings'
    )

    FindMouseVocalizations(
        root_directory=root_directory,
        input_parameter_dict=processing_settings_dict
    ).summarize_das_findings()

@click.command(name="concatenate-ephys-files")
@click.option('--root-directories', type=str, required=True, help='Comma-separated string of session root directory paths.')
def concatenate_binary_files_cli(root_directories):
    """
    Description
    ----------
    A command-line tool to concatenate ephys binary files across multiple sessions.
    ----------

    Parameters
    ----------
    ----------

    Returns
    ----------
    ----------
    """

    all_paths = [d.strip() for d in root_directories.split(',')]
    valid_dirs = [path for path in all_paths if pathlib.Path(path).is_dir()]

    if len(valid_dirs) > 0:
        Operator(
            root_directory=valid_dirs
        ).concatenate_binary_files()


@click.command(name="split-clusters")
@click.option('--root-directories', type=str, required=True, help='A comma-separated string of session root directory paths.')
@click.option('--min-spikes', 'min_spike_num', type=int, default=None, required=False, help='Minimum number of spikes for a cluster to be saved.')
@click.option('--kilosort-version', type=str, default=None, required=False, help='Version of Kilosort used for spike sorting.')
@click.pass_context
def split_clusters_to_sessions_cli(ctx, root_directories, **kwargs):
    """
    Description
    ----------
    A command-line tool to split curated ephys clusters into individual session files.
    ----------

    Parameters
    ----------
    ----------

    Returns
    ----------
    ----------
    """

    all_paths = [d.strip() for d in root_directories.split(',')]
    valid_dirs = [path for path in all_paths if pathlib.Path(path).is_dir()]

    provided_params = [key for key in kwargs if ctx.get_parameter_source(key) == ParameterSource.COMMANDLINE]

    processing_settings_dict = modify_settings_json_for_cli(
        ctx=ctx,
        provided_params=provided_params,
        settings_dict='processing_settings'
    )

    if len(valid_dirs) > 0:
        Operator(
            root_directory=valid_dirs,
            input_parameter_dict=processing_settings_dict
        ).split_clusters_to_sessions()
