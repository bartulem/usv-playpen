"""
@author: bartulem
Code to preprocess data after running experiments.
"""

import json
import os
import traceback
from datetime import datetime
from anipose_operations import ConvertTo3D
from extract_phidget_data import Gatherer
from file_manipulation import Operator
from preprocessing_plot import SummaryPlotter
from send_email import Messenger
from synchronize_files import Synchronizer
from usv_inference import FindMouseVocalizations


class Stylist:

    def __init__(self, input_parameter_dict=None, root_directories=None,
                 exp_settings_dict=None, message_output=None):

        if root_directories is None:
            with open('input_parameters.json', 'r') as json_file:
                self.root_directory = json.load(json_file)['preprocess_data']['root_directories']
        else:
            self.root_directories = root_directories

        if input_parameter_dict is None:
            with open('input_parameters.json', 'r') as json_file:
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

    def prepare_data_for_analyses(self):
        """
        Description
        ----------
        This method preprocesses experimental data: (1) concatenates video files
        (necessary for sessions >15 min), (2) corrects videos to appropriate FPS,
        (3) crops audio files to match video length, (4) checks audio-video sync
        using IPI pulses generated by the Arduino-controlled IR lights,
        (5) vertically stacks all audio files in one memmap file, (6) extracts
        phidget-measured data during the experiment, (7) plots the summary of
        these analyses.
        ----------

        Parameters
        ----------
        Contains the following set of parameters
            root_directory (list)
                List of root directories for recording sessions.
        ----------

        Returns
        ----------
        preprocessing_plot (fig)
            Figure summarizing the preprocessing of experimental data.
        ----------
        """

        Messenger(message_output=self.message_output,
                  receivers=self.input_parameter_dict['send_email']['Messenger']['send_message']['receivers'],
                  exp_settings_dict=self.exp_settings_dict).send_message(subject="Audio PC in 165B is busy, do NOT attempt to remote in!",
                                                                         message=f"Data preprocessing in progress, started at "
                                                                                 f"{datetime.now().hour:02d}:{datetime.now().minute:02d}.{datetime.now().second:02d} "
                                                                                 f"and run by @{self.input_parameter_dict['send_email']['Messenger']['experimenter']}. "
                                                                                 f"You will be notified upon completion. \n \n ***This is an automatic e-mail, please do NOT respond.***")

        # analyze data in all root directories at once
        if self.input_parameter_dict['processing_booleans']['conduct_ephys_file_chaining'] or self.input_parameter_dict['processing_booleans']['split_cluster_spikes']:
            try:
                self.message_output(f"Preprocessing e-phys data started at: "
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

                self.message_output(f"Preprocessing e-phys data finished at: "
                                    f"{datetime.now().hour:02d}:{datetime.now().minute:02d}.{datetime.now().second:02d}")

            except (OSError, RuntimeError, TypeError, IndexError, IOError, EOFError, TimeoutError, NameError, KeyError, ValueError, AttributeError):
                self.message_output(traceback.format_exc())

        # analyze data in each root directory separately
        else:
            for one_directory in self.root_directories:
                try:
                    self.message_output(f"Preprocessing data in {one_directory} started at: "
                                        f"{datetime.now().hour:02d}:{datetime.now().minute:02d}.{datetime.now().second:02d}")

                    # # # configure video properties via ffmpeg
                    if self.input_parameter_dict['processing_booleans']['conduct_video_concatenation']:
                        Operator(root_directory=one_directory,
                                 input_parameter_dict=self.input_parameter_dict,
                                 message_output=self.message_output).concatenate_video_files()

                    if self.input_parameter_dict['processing_booleans']['conduct_video_fps_change']:
                        Operator(root_directory=one_directory,
                                 input_parameter_dict=self.input_parameter_dict,
                                 message_output=self.message_output).rectify_video_fps()

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

                        prediction_error_dict = Synchronizer(root_directory=one_directory,
                                                             input_parameter_dict=self.input_parameter_dict,
                                                             message_output=self.message_output).find_audio_sync_trains()

                        SummaryPlotter(root_directory=one_directory,
                                       input_parameter_dict=self.input_parameter_dict).preprocessing_summary(prediction_error_dict=prediction_error_dict,
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

                    # # # vstack audio files in memmap
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
                        ConvertTo3D(root_directory=one_directory,
                                    input_parameter_dict=self.input_parameter_dict,
                                    message_output=self.message_output,
                                    exp_settings_dict=self.exp_settings_dict).translate_rotate_metric()

                    # # # conduct DAS inference on audio data
                    if self.input_parameter_dict['processing_booleans']['das_infer']:
                        FindMouseVocalizations(root_directory=one_directory,
                                               input_parameter_dict=self.input_parameter_dict,
                                               message_output=self.message_output).das_command_line_inference()

                    self.message_output(f"Preprocessing data in {one_directory} finished at: "
                                        f"{datetime.now().hour:02d}:{datetime.now().minute:02d}.{datetime.now().second:02d}")

                except (OSError, RuntimeError, TypeError, IndexError, IOError, EOFError, TimeoutError, NameError, KeyError, ValueError, AttributeError):
                    self.message_output(traceback.format_exc())

        Messenger(message_output=self.message_output,
                  no_receivers_notification=False,
                  receivers=self.input_parameter_dict['send_email']['Messenger']['send_message']['receivers'],
                  exp_settings_dict=self.exp_settings_dict).send_message(subject="Audio PC in 165B is available again, processing has been completed",
                                                                         message=f"Data preprocessing has been completed at "
                                                                                 f"{datetime.now().hour:02d}:{datetime.now().minute:02d}.{datetime.now().second:02d} "
                                                                                 f"by @{self.input_parameter_dict['send_email']['Messenger']['experimenter']}. "
                                                                                 f"You will be notified about further PC usage "
                                                                                 f"should it occur. \n \n ***This is an automatic e-mail, please do NOT respond.***")
