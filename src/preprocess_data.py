"""
@author: bartulem
Preprocess data after running experiments.
"""

import json
import os
import pathlib
import traceback
from datetime import datetime
from .anipose_operations import ConvertTo3D
from .assign_vocalizations import Vocalocator
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
                             message_output=self.message_output,
                             exp_settings_dict=self.exp_settings_dict).concatenate_binary_files()

                # # # split clusters to individual sessions
                if self.input_parameter_dict['processing_booleans']['split_cluster_spikes']:
                    Operator(root_directory=self.root_directories,
                             input_parameter_dict=self.input_parameter_dict,
                             message_output=self.message_output,
                             exp_settings_dict=self.exp_settings_dict).split_clusters_to_sessions()

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
                                     message_output=self.message_output,
                                     exp_settings_dict=self.exp_settings_dict).crop_wav_files_to_video()

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
                                     message_output=self.message_output,
                                     exp_settings_dict=self.exp_settings_dict).validate_ephys_video_sync()

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
                                        message_output=self.message_output,
                                        exp_settings_dict=self.exp_settings_dict).translate_rotate_metric(session_idx=one_directory_idx)
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
