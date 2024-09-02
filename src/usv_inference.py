"""
@author: bartulem
Run USV inference on WAV files and create annotations.
"""

from PyQt6.QtTest import QTest
import glob
import json
import os
import pathlib
import shutil
import subprocess
from datetime import datetime


class FindMouseVocalizations:

    def __init__(self, root_directory=None, input_parameter_dict=None,
                 exp_settings_dict=None, message_output=None):
        if input_parameter_dict is None:
            with open('input_parameters.json', 'r') as json_file:
                self.input_parameter_dict = json.load(json_file)['usv_inference']['FindMouseVocalizations']
        else:
            self.input_parameter_dict = input_parameter_dict['usv_inference']['FindMouseVocalizations']

        if root_directory is None:
            with open('input_parameters.json', 'r') as json_file:
                self.root_directory = json.load(json_file)['usv_inference']['root_directory']
        else:
            self.root_directory = root_directory

        if exp_settings_dict is None:
            self.exp_settings_dict = None
        else:
            self.exp_settings_dict = exp_settings_dict

        if message_output is None:
            self.message_output = print
        else:
            self.message_output = message_output

        if exp_settings_dict is None:
            self.exp_settings_dict = None
        else:
            self.exp_settings_dict = exp_settings_dict

    def das_command_line_inference(self):
        """
        Description
        ----------
        This function takes WAV files as input and runs DAS inference on them to generate
        tentative USV segments in the recording.
        ----------

        Parameters
        ----------
        Contains the following set of parameters
            model_directory (str)
                Directory containing DAS model files.
            model_name_base (str)
                The base of the DAS model name.
            output_file_type (str)
                Type of annotation output file; defaults to 'csv'.
            segment_threshold (int / float)
                Confidence threshold for detecting segments, range 0-1; defaults to 0.5.
            segment_minlen (int / float)
                Minimal duration of a segment used for filtering out spurious detections; defaults to 0.015 (s).
            segment_fillgap (int / float)
                Gap between adjacent segments to be filled; defaults to 0.015 (s).
        ----------

        Returns
        ----------
        .csv annotation files
            CSV files w/ onsets and offsets of all detected USV segments,
            shape: (N_USV, VOC_TYPE, START_SEC, END_SEC).
        ----------
        """

        self.message_output(f"DAS inference started at: {datetime.now().hour:02d}:{datetime.now().minute:02d}.{datetime.now().second:02d}. Please be patient, this can take >5 min/file.")
        QTest.qWait(1000)

        das_conda_name = self.input_parameter_dict['das_command_line_inference']['das_conda_env_name']
        model_base = f"{self.input_parameter_dict['das_command_line_inference']['model_directory']}{os.sep}{self.input_parameter_dict['das_command_line_inference']['model_name_base']}"
        thresh = self.input_parameter_dict['das_command_line_inference']['segment_tmf'][0]
        min_len = self.input_parameter_dict['das_command_line_inference']['segment_tmf'][1]
        fill_gap = self.input_parameter_dict['das_command_line_inference']['segment_tmf'][2]
        save_format = self.input_parameter_dict['das_command_line_inference']['output_file_type']

        if os.name == 'nt':
            command_addition = 'cmd /c '
            shell_usage_bool = False
        else:
            command_addition = 'eval "$(conda shell.bash hook)" && '
            shell_usage_bool = True

        # run inference
        for one_file in glob.glob(pathname=os.path.join(f"{self.root_directory}{os.sep}audio{os.sep}hpss_filtered", "*.wav*")):
            self.message_output(f"Running DAS inference on: {os.path.basename(one_file)}")
            QTest.qWait(2000)

            inference_subp = subprocess.Popen(f'''{command_addition}conda activate {das_conda_name} && das predict {one_file} {model_base} --segment-thres {thresh} --segment-minlen {min_len} --segment-fillgap {fill_gap} --save-format {save_format}''',
                                              stdout=subprocess.DEVNULL,
                                              stderr=subprocess.STDOUT,
                                              cwd=f"{self.root_directory}{os.sep}audio{os.sep}hpss_filtered",
                                              shell=shell_usage_bool)

            while True:
                status_poll = inference_subp.poll()
                if status_poll is None:
                    QTest.qWait(5000)
                else:
                    break

        # create save directory if it doesn't exist
        pathlib.Path(f"{self.root_directory}{os.sep}audio{os.sep}das_annotations").mkdir(parents=True, exist_ok=True)

        # move CSV files to save directory and remove them from WAV directory
        for one_file in os.listdir(f"{self.root_directory}{os.sep}audio{os.sep}hpss_filtered"):
            if f".{save_format}" in one_file:
                shutil.move(f"{self.root_directory}{os.sep}audio{os.sep}hpss_filtered{os.sep}{one_file}",
                            f"{self.root_directory}{os.sep}audio{os.sep}das_annotations{os.sep}{one_file}")
