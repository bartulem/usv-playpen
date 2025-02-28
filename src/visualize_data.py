"""
@author: bartulem
Visualize 3D tracking, vocalization and neural data.
"""

import json
import pathlib
import traceback
from datetime import datetime
from .send_email import Messenger
from src.visualizations.make_behavioral_tuning_figures import RatemapFigureMaker
from src.visualizations.make_behavioral_videos import Create3DVideo


class Visualizer:

    def __init__(self, input_parameter_dict=None, root_directories=None,
                 message_output=None):

        if root_directories is None:
            with open((pathlib.Path(__file__).parent / '_parameter_settings/visualizations_settings.json'), 'r') as json_file:
                self.root_directories = json.load(json_file)['visualize_data']['root_directories']
        else:
            self.root_directories = root_directories

        if input_parameter_dict is None:
            with open((pathlib.Path(__file__).parent / '_parameter_settings/visualizations_settings.json'), 'r') as json_file:
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

        Messenger(message_output=self.message_output,
                  receivers=self.input_parameter_dict['send_email']['send_message']['receivers']).send_message(subject=f"{self.input_parameter_dict['send_email']['visualizations_pc_choice']} PC is busy, do NOT attempt to remote in!",
                                                                                                               message=f"Data visualizations in progress, started at "
                                                                                                                       f"{datetime.now().hour:02d}:{datetime.now().minute:02d}.{datetime.now().second:02d} "
                                                                                                                       f"and run by @{self.input_parameter_dict['send_email']['experimenter']}. "
                                                                                                                       f"You will be notified upon completion. \n \n ***This is an automatic e-mail, please do NOT respond.***")

        for one_directory in self.root_directories:
            try:
                self.message_output(f"Visualizing data in {one_directory} started at: "
                                    f"{datetime.now().hour:02d}:{datetime.now().minute:02d}.{datetime.now().second:02d}.")

                # # # # plot behavioral tuning curves
                if self.input_parameter_dict['visualize_booleans']['make_behavioral_tuning_figures_bool']:
                    RatemapFigureMaker(root_directory=one_directory,
                                       visualizations_parameter_dict=self.input_parameter_dict,
                                       message_output=self.message_output).neuronal_tuning_figures()

                # # # # make behavioral videos
                if self.input_parameter_dict['visualize_booleans']['make_behavioral_videos_bool']:
                    Create3DVideo(root_directory=one_directory,
                                  arena_directory=self.input_parameter_dict['make_behavioral_videos']['arena_directory'],
                                  speaker_audio_file=self.input_parameter_dict['make_behavioral_videos']['speaker_audio_file'],
                                  exp_id=self.input_parameter_dict['send_email']['experimenter'],
                                  visualizations_parameter_dict=self.input_parameter_dict,
                                  message_output=self.message_output).visualize_in_video()

                self.message_output(f"Visualizing data in {one_directory} finished at: "
                                    f"{datetime.now().hour:02d}:{datetime.now().minute:02d}.{datetime.now().second:02d}.")

            except (OSError, RuntimeError, TypeError, IndexError, IOError, EOFError, TimeoutError, NameError, KeyError, ValueError, AttributeError):
                self.message_output(traceback.format_exc())

        Messenger(message_output=self.message_output,
                  no_receivers_notification=False,
                  receivers=self.input_parameter_dict['send_email']['send_message']['receivers']).send_message(subject=f"{self.input_parameter_dict['send_email']['visualizations_pc_choice']} PC is available again, visualizations have been completed",
                                                                                                               message=f"Data visualizations have been completed at "
                                                                                                                       f"{datetime.now().hour:02d}:{datetime.now().minute:02d}.{datetime.now().second:02d} "
                                                                                                                       f"by @{self.input_parameter_dict['send_email']['experimenter']}. "
                                                                                                                       f"You will be notified about further PC usage "
                                                                                                                       f"should it occur. \n \n ***This is an automatic e-mail, please do NOT respond.***")
