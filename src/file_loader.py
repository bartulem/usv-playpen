"""
@author: bartulem
Loads WAV files.
"""

import json
import librosa
import os
from scipy.io import wavfile
import struct
import subprocess
import warnings


class DataLoader:

    if os.name == 'nt':
        command_addition = 'cmd /c '
        move_command = 'move'
        remove_command = 'del'
        shell_usage_bool = False
    else:
        command_addition = ''
        move_command = 'mv'
        remove_command = 'rm'
        shell_usage_bool = True

    known_dtypes = {
        'int': int,
        'np.int8': 'int8',
        'int8': 'int8',
        'np.int16': 'int16',
        'int16': 'int16',
        'np.int32': 'int32',
        'int32': 'int32',
        'np.int64': 'int64',
        'int64': 'int64',
        'np.uint8': 'uint8',
        'uint8': 'uint8',
        'np.uint16': 'uint16',
        'uint16': 'uint16',
        'np.uint32': 'uint32',
        'uint32': 'uint32',
        'np.uint64': 'uint64',
        'uint64': 'uint64',
        'float': float,
        'np.float16': 'float16',
        'float16': 'float16',
        'np.float32': 'float32',
        'float32': 'float32',
        'np.float64': 'float64',
        'float64': 'float64',
        'str': str,
        'dict': dict
    }

    def __init__(self, input_parameter_dict=None):
        if input_parameter_dict is None:
            with open('input_parameters.json', 'r') as json_file:
                self.input_parameter_dict = json.load(json_file)['file_loader']['DataLoader']
        else:
            self.input_parameter_dict = input_parameter_dict

    def load_wavefile_data(self):
        """
        Description
        ----------
        This method loads the .wav file(s) of interest.
        ----------

        Parameters
        ----------
        Contains the following set of parameters
            library (str)
                Which module/library to use to load data; defaults to 'scipy'.
            conditional_arg (list)
                String to search in file names; defaults to empty list.
        ----------

        Returns
        ----------
        wave_data_dict (dict)
            A dictionary with all desired sound outputs.
        ----------
        """

        # spits out warnings if .wav file has header, the line below suppresses it
        warnings.simplefilter('ignore')

        wave_data_dict = {}
        for one_dir in self.input_parameter_dict['wave_data_loc']:
            for one_file in os.listdir(one_dir):

                # additional conditional argument to reduce numbers of files loaded
                if len(self.input_parameter_dict['load_wavefile_data']['conditional_arg']) == 0:
                    additional_condition = True
                else:
                    additional_condition = all([cond in one_file for cond in self.input_parameter_dict['load_wavefile_data']['conditional_arg']])

                if '.wav' in one_file and additional_condition:
                    wave_data_dict[one_file] = {'sampling_rate': 0, 'wav_data': 0, 'dtype': 0}
                    if self.input_parameter_dict['load_wavefile_data']['library'] == 'scipy':
                        try:
                            wave_data_dict[one_file]['sampling_rate'], wave_data_dict[one_file]['wav_data'] = wavfile.read(f'{one_dir}{os.sep}{one_file}')
                        except struct.error:
                            subprocess.run(args=f'''{self.command_addition}sox {one_file} {one_file[:-4]}_correct.wav && {self.remove_command} {one_file} && {self.move_command} {one_file[:-4]}_correct.wav {one_file}''',
                                           shell=self.shell_usage_bool,
                                           cwd=one_dir,
                                           stdout=subprocess.DEVNULL,
                                           stderr=subprocess.STDOUT)
                            wave_data_dict[one_file]['sampling_rate'], wave_data_dict[one_file]['wav_data'] = wavfile.read(f'{one_dir}{os.sep}{one_file}')
                    else:
                        wave_data_dict[one_file]['wav_data'], wave_data_dict[one_file]['sampling_rate'] = librosa.load(f'{one_dir}{os.sep}{one_file}')
                    wave_data_dict[one_file]['dtype'] = self.known_dtypes[type(wave_data_dict[one_file]['wav_data'].ravel()[0]).__name__]
        return wave_data_dict
