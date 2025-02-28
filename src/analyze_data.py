"""
@author: bartulem
Conduct analyses of choice on the data of choice (on the PC of choice).
"""

from datetime import datetime
import json
import pathlib
import traceback
import warnings
from .send_email import Messenger
from .analyses.compute_behavioral_features import FeatureZoo
from .analyses.compute_behavioral_tuning_curves import NeuronalTuning
from .analyses.generate_audio_files import AudioGenerator

warnings.simplefilter('ignore')


class Analyst:

    def __init__(self, input_parameter_dict=None, root_directories=None,
                 message_output=None):

        if root_directories is None:
            with open((pathlib.Path(__file__).parent / '_parameter_settings/analyses_settings.json'), 'r') as json_file:
                self.root_directories = json.load(json_file)['analyze_data']['root_directories']
        else:
            self.root_directories = root_directories

        if input_parameter_dict is None:
            with open((pathlib.Path(__file__).parent / '_parameter_settings/analyses_settings.json'), 'r') as json_file:
                self.input_parameter_dict = json.load(json_file)
        else:
            self.input_parameter_dict = input_parameter_dict

        if message_output is None:
            self.message_output = print
        else:
            self.message_output = message_output

    def analyze_data(self) -> None:
        """
        Description
        ----------
        This method performs the following analyses:
        (1) computes behavioral features and plots their distributions
        (2) computes behavioral tuning curves
        (3) generates playback WAV files
        (4) frequency shifts audio segments
        ----------

        Parameters
        ----------
        ----------

        Returns
        ----------
        ----------
        """

        Messenger(message_output=self.message_output,
                  receivers=self.input_parameter_dict['send_email']['send_message']['receivers'],
                  exp_settings_dict=None).send_message(subject=f"{self.input_parameter_dict['send_email']['analyses_pc_choice']} PC is busy, do NOT attempt to remote in!",
                                                       message=f"Data analyses in progress, started at "
                                                               f"{datetime.now().hour:02d}:{datetime.now().minute:02d}.{datetime.now().second:02d} "
                                                               f"and run by @{self.input_parameter_dict['send_email']['experimenter']}. "
                                                               f"You will be notified upon completion. \n \n ***This is an automatic e-mail, please do NOT respond.***")

        # # # create USV playback WAV files
        if self.input_parameter_dict['analyses_booleans']['create_usv_playback_wav_bool']:
            AudioGenerator(exp_id=self.input_parameter_dict['send_email']['experimenter'],
                           create_playback_settings_dict=self.input_parameter_dict['create_usv_playback_wav'],
                           message_output=self.message_output).create_usv_playback_wav()

        for one_directory in self.root_directories:
            try:
                self.message_output(f"Analyzing data in {one_directory} started at: "
                                    f"{datetime.now().hour:02d}:{datetime.now().minute:02d}.{datetime.now().second:02d}.")

                # # # compute behavioral features and plot their distributions
                if self.input_parameter_dict['analyses_booleans']['compute_behavioral_features_bool']:
                    FeatureZoo(root_directory=one_directory,
                               behavioral_parameters_dict=self.input_parameter_dict['compute_behavioral_features'],
                               message_output=self.message_output).save_behavioral_features_to_file()

                # # # compute behavioral tuning curves
                if self.input_parameter_dict['analyses_booleans']['compute_behavioral_tuning_bool']:
                    NeuronalTuning(root_directory=one_directory,
                                   tuning_parameters_dict=self.input_parameter_dict['calculate_neuronal_tuning_curves'],
                                   message_output=self.message_output).calculate_neuronal_tuning_curves()

                # # # frequency shift audio segments
                if self.input_parameter_dict['analyses_booleans']['frequency_shift_audio_segment_bool']:
                    AudioGenerator(root_directory=one_directory,
                                   freq_shift_settings_dict=self.input_parameter_dict['frequency_shift_audio_segment'],
                                   message_output=self.message_output).frequency_shift_audio_segment()

                self.message_output(f"Analyzing data in {one_directory} finished at: "
                                    f"{datetime.now().hour:02d}:{datetime.now().minute:02d}.{datetime.now().second:02d}.")

            except (OSError, RuntimeError, TypeError, IndexError, IOError, EOFError, TimeoutError, NameError, KeyError, ValueError, AttributeError):
                self.message_output(traceback.format_exc())

        Messenger(message_output=self.message_output,
                  no_receivers_notification=False,
                  receivers=self.input_parameter_dict['send_email']['send_message']['receivers'],
                  exp_settings_dict=None).send_message(subject=f"{self.input_parameter_dict['send_email']['analyses_pc_choice']} PC is available again, analyses have been completed",
                                                       message=f"Data analyses have been completed at "
                                                               f"{datetime.now().hour:02d}:{datetime.now().minute:02d}.{datetime.now().second:02d} "
                                                               f"by @{self.input_parameter_dict['send_email']['experimenter']}. "
                                                               f"You will be notified about further PC usage "
                                                               f"should it occur. \n \n ***This is an automatic e-mail, please do NOT respond.***")
