"""
@author: bartulem
Writes WAV files.
"""

import json
import os
import numpy as np
from scipy.io import wavfile
import soundfile


class DataWriter:

    def __init__(self, input_parameter_dict=None, wav_data=None):
        if input_parameter_dict is None:
            with open('input_parameters.json', 'r') as json_file:
                self.input_parameter_dict = json.load(json_file)['file_writer']['DataWriter']
                self.wave_write_loc = self.input_parameter_dict['wave_write_loc']
                self.wav_data = wav_data
        else:
            self.input_parameter_dict = input_parameter_dict
            self.wave_write_loc = self.input_parameter_dict['wave_write_loc']
            self.wav_data = wav_data

    def write_wavefile_data(self):
        """
        Description
        ----------
        This method writes a .wav file to disk.
        ----------

        Parameters
        ----------
        Contains the following set of parameters
            file_name (str)
                The desired name of the .wav file (without extension).
            sampling_rate (int)
                The sampling rate of the .wav file; defaults to 300 (kHz).
            library (str)
                Which module/library to use to write data; defaults to 'scipy'.
        ----------

        Returns
        ----------
        wave_file (dict)
            A .wav file.
        ----------
        """

        if os.path.exists(self.wave_write_loc):
            if self.input_parameter_dict['write_wavefile_data']['library'] == 'scipy':
                wavfile.write(filename=f'{self.wave_write_loc}{os.sep}{self.input_parameter_dict["write_wavefile_data"]["file_name"]}.wav',
                              rate=int(np.ceil(self.input_parameter_dict['write_wavefile_data']['sampling_rate']*1e3)),
                              data=self.wav_data)
            else:
                soundfile.write(file=f'{self.wave_write_loc}{os.sep}{self.input_parameter_dict["write_wavefile_data"]["file_name"]}.wav',
                                data=self.wav_data,
                                samplerate=int(np.ceil(self.input_parameter_dict['write_wavefile_data']['sampling_rate']*1e3)))
