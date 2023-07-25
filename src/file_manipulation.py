"""
@author: bartulem
Manipulates files:
(1) break from multi to single channel, band-pass filter and temporally concatenate audio (e.g., wav) files
(2) concatenate video (e.g., mp4) files
(3) change video (e.g., mp4) sampling rate (fps)
"""

from PyQt6.QtTest import QTest
import glob
import json
import os
import shutil
import sys
import numpy as np
import subprocess
from datetime import datetime
from scipy.signal import square
from file_loader import DataLoader
from file_writer import DataWriter


class Operator:
    tone_functions = {
        'np.sin': np.sin,
        'np.cos': np.cos,
        'square': square
    }

    def __init__(self, root_directory=None, input_parameter_dict=None, message_output=None):
        if input_parameter_dict is None:
            with open('input_parameters.json', 'r') as json_file:
                self.input_parameter_dict = json.load(json_file)['file_manipulation']['Operator']
        else:
            self.input_parameter_dict = input_parameter_dict['file_manipulation']['Operator']

        if root_directory is None:
            with open('input_parameters.json', 'r') as json_file:
                self.root_directory = json.load(json_file)['file_manipulation']['root_directory']
        else:
            self.root_directory = root_directory

        if message_output is None:
            self.message_output = print
        else:
            self.message_output = message_output

    def multichannel_to_channel_audio(self):
        """
        Description
        ----------
        This method splits multichannel audio file into single channel files and
        concatenates single channel files via Sox, since multichannel files where
        split due to a size limitation.

        NB: You need to install sox: https://sourceforge.net/projects/sox/files/sox/
            and add the sox directory to your system PATH prior to running this
        ----------

        Parameters
        ----------
        ----------

        Returns
        ----------
        wave_file (.wav files)
            Concatenated single channel wave files.
        ----------
        """

        self.message_output(f"Multichannel to single channel audio conversion started at: {datetime.now().hour:02d}:{datetime.now().minute:02d}.{datetime.now().second:02d}")

        QTest.qWait(2000)

        if not os.path.isdir(f"{self.root_directory}{os.sep}audio{os.sep}temp"):

            os.makedirs(f"{self.root_directory}{os.sep}audio{os.sep}temp", exist_ok=False)

        mc_audio_files = DataLoader(input_parameter_dict={'wave_data_loc': [f"{self.root_directory}{os.sep}audio{os.sep}original_mc"],
                                                          'load_wavefile_data': {'library': 'scipy', 'conditional_arg': []}}).load_wavefile_data()

        # split multichannel files
        for mc_audio_file in mc_audio_files.keys():
            for ch in range(mc_audio_files[mc_audio_file]['wav_data'].shape[1]):
                DataWriter(wav_data=mc_audio_files[mc_audio_file]['wav_data'][:, ch],
                           input_parameter_dict={'wave_write_loc': f"{self.root_directory}{os.sep}audio{os.sep}temp",
                                                 'write_wavefile_data': {'file_name': f"{mc_audio_file[:-4]}_ch{ch + 1:02d}",
                                                                         'sampling_rate': mc_audio_files[mc_audio_file]['sampling_rate'] / 1e3,
                                                                         'library': 'scipy'}}).write_wavefile_data()
        # release dict from memory
        mc_audio_files = 0

        # find name origin for file naming purposes
        name_origin = glob.glob(f"{self.root_directory}{os.sep}audio{os.sep}temp{os.sep}m_*_ch*.wav")[0].split('_')[2]

        # concatenate single channel files for master/slave
        for device_id in ['m', 's']:
            for ch in range(1, 13):
                mc_to_sc_subp = subprocess.Popen(f'''cmd /c "sox {device_id}_*_ch{ch:02d}.wav -q {self.root_directory}{os.sep}audio{os.sep}original{os.sep}{device_id}_{name_origin}_ch{ch:02d}.wav"''',
                                                 cwd=f"{self.root_directory}{os.sep}audio{os.sep}temp")

                while True:
                    status_poll = mc_to_sc_subp.poll()
                    if status_poll is None:
                        QTest.qWait(1000)
                    else:
                        break

        # delete temp directory (w/ all files in it)
        shutil.rmtree(f"{self.root_directory}{os.sep}audio{os.sep}temp")

        # self.message_output(f"Multichannel to single channel audio conversion completed at: {datetime.now().hour:02d}:{datetime.now().minute:02d}.{datetime.now().second:02d}")

    def filter_audio_files(self):
        """
        Description
        ----------
        This method filters audio files via Sox.

        NB: You need to install sox: https://sourceforge.net/projects/sox/files/sox/
            and add the sox directory to your system PATH prior to running this

        It applies a sinc kaiser-windowed low-pass, high-pass, band-pass, or band-reject filter
        to the signal. The freqHP and freqLP parameters give the frequencies of the 6dB points
        of a high-pass and low-pass filter that may be invoked individually, or together. If
        both are given, then freqHP less than freqLP creates a band-pass filter, freqHP greater
        than freqLP creates a band-reject filter. For example, the invocations:
           sinc 3k
           sinc -4k
           sinc 3k-4k
           sinc 4k-3k
        create a high-pass, low-pass, band-pass, and band-reject filter respectively.
        ----------

        Parameters
        ----------
        Contains the following set of parameters
            audio_format (str)
                The format of the audio files; defaults to 'wav'.
            freq_hp (int, float)
                High pass filter frequency; defaults to 2000 (Hz).
            freq_lp (int, float)
                Low pass filter frequency; defaults to 0 (Hz).
        ----------

        Returns
        ----------
        wave_file (.wav file)
            Filtered wave file.
        ----------
        """

        freq_hp = self.input_parameter_dict['filter_audio_files']['freq_hp']
        freq_lp = self.input_parameter_dict['filter_audio_files']['freq_lp']

        self.message_output(f"Filtering out signal between {freq_lp} and {freq_hp} Hz in audio files started at: "
                            f"{datetime.now().hour:02d}:{datetime.now().minute:02d}.{datetime.now().second:02d}")

        QTest.qWait(2000)

        if not os.path.exists(f"{self.root_directory}{os.sep}audio{os.sep}filtered"):
            os.makedirs(f"{self.root_directory}{os.sep}audio{os.sep}filtered", exist_ok=False)

        if os.path.exists(f"{self.root_directory}{os.sep}audio{os.sep}cropped_to_video"):
            all_audio_files = glob.glob(f"{self.root_directory}{os.sep}audio{os.sep}cropped_to_video{os.sep}"
                                        f"*.{self.input_parameter_dict['filter_audio_files']['audio_format']}")

            if len(all_audio_files) > 0:
                for one_file in all_audio_files:
                    filter_subp = subprocess.Popen(f'''cmd /c "sox {one_file.split(os.sep)[-1]} {self.root_directory}{os.sep}audio{os.sep}filtered{os.sep}{one_file.split(os.sep)[-1][:-4]}_filtered.wav sinc {freq_hp}-{freq_lp}"''',
                                                   cwd=f"{self.root_directory}{os.sep}audio{os.sep}cropped_to_video")

                    while True:
                        status_poll = filter_subp.poll()
                        if status_poll is None:
                            QTest.qWait(1000)
                        else:
                            break

    def concatenate_audio_files(self):
        """
        Description
        ----------
        This method concatenates audio files via Sox.

        NB: You need to install sox: https://sourceforge.net/projects/sox/files/sox/
            and add the sox directory to your system PATH prior to running this

        NB: Size of RIFF (.wav file format) chunk data is stored in 32 bits.
            (max. unsigned value is 4 294 967 295), i.e., RIFF is limited to ~4.2 GBytes per file
        ----------

        Parameters
        ----------
        Contains the following set of parameters
            audio_format (str)
                The format of the audio files; defaults to 'wav'.
            concat_type (str)
                Either 'vstack' or 'hstack'.
        ----------

        Returns
        ----------
        wave_file (.wav file) or memmap file
            Concatenated wave file.
        ----------
        """

        self.message_output(f"Audio concatenation started at: {datetime.now().hour:02d}:{datetime.now().minute:02d}.{datetime.now().second:02d}")

        QTest.qWait(2000)

        name_origin = ""
        if os.path.exists(f"{self.root_directory}{os.sep}audio{os.sep}cropped_to_video"):
            all_audio_files = glob.glob(f"{self.root_directory}{os.sep}audio{os.sep}cropped_to_video{os.sep}"
                                        f"*.{self.input_parameter_dict['concatenate_audio_files']['audio_format']}")

            if len(all_audio_files) > 1:
                if self.input_parameter_dict['concatenate_audio_files']['concat_type'] == 'hstack':
                    total_file_size_in_gb = 0
                    for file_idx, one_file in enumerate(all_audio_files):
                        total_file_size_in_gb += os.path.getsize(one_file) / (1024*1024*1024)
                        if file_idx == 0:
                            name_origin = one_file.split(os.sep)[-1].split('_')[-1][:-4]

                    if self.input_parameter_dict['concatenate_audio_files']['audio_format'] != 'wav' or total_file_size_in_gb < 4.2:
                        hstack_subp = subprocess.Popen(f'''cmd /c "sox *.{self.input_parameter_dict['concatenate_audio_files']['audio_format']} -S 
                                                       {name_origin}_concatenated.{self.input_parameter_dict['concatenate_audio_files']['audio_format']}"''',
                                                       cwd=f"{self.root_directory}{os.sep}audio{os.sep}cropped_to_video")

                        while True:
                            status_poll = hstack_subp.poll()
                            if status_poll is None:
                                QTest.qWait(1000)
                            else:
                                break

                    else:
                        self.message_output("The combined size of these files exceeds 4.2 Gb, so concatenation won't work in .wav form. Saving file as memmap. NB: memory exhaustive process!")

                        data_dict = DataLoader(input_parameter_dict={'wave_data_loc': [f"{self.root_directory}{os.sep}audio{os.sep}cropped_to_video"],
                                                                     'load_wavefile_data': {'library': 'scipy', 'conditional_arg': []}}).load_wavefile_data()

                        dim_1 = 0
                        dim_2 = 0
                        data_type = 0
                        sr = 0
                        for file_idx, one_file in enumerate(data_dict.keys()):
                            if file_idx == 0:
                                data_type = data_dict[one_file]['dtype']
                                sr = data_dict[one_file]['sampling_rate']
                                dim_1 += data_dict[one_file]['wav_data'].shape[0]
                                dim_2 = data_dict[one_file]['wav_data'].shape[1]
                            else:
                                dim_1 += data_dict[one_file]['wav_data'].shape[0]

                        audio_mm_arr = np.memmap(f"{self.root_directory}{os.sep}audio{os.sep}cropped_to_video{os.sep}{name_origin}"
                                                 f"_concatenated_audio_{sr}_{dim_1}_{dim_2}_{str(data_type).split('.')[-1][:-2]}.mmap",
                                                 dtype=data_type,
                                                 mode='w+',
                                                 shape=(dim_1, dim_2))

                        counter = 0
                        for one_file in data_dict.keys():
                            audio_mm_arr[counter:counter+data_dict[one_file]['wav_data'].shape[0], :] = data_dict[one_file]['wav_data']
                            counter += data_dict[one_file]['wav_data'].shape[0]

                        audio_mm_arr.flush()
                        audio_mm_arr = 0

                    self.message_output(f"Audio concatenation completed at: {datetime.now().hour:02d}:{datetime.now().minute:02d}.{datetime.now().second:02d}")

                elif self.input_parameter_dict['concatenate_audio_files']['concat_type'] == 'vstack':

                    # self.message_output("Saving the vstacked files as memmap. NB: memory exhaustive process!")

                    data_dict = DataLoader(input_parameter_dict={'wave_data_loc': [f"{self.root_directory}{os.sep}audio{os.sep}cropped_to_video"],
                                                                 'load_wavefile_data': {'library': 'scipy', 'conditional_arg': []}}).load_wavefile_data()
                    dim_1 = 0
                    dim_2 = len(data_dict.keys())
                    data_type = 0
                    sr = 0
                    for file_idx, one_file in enumerate(data_dict.keys()):
                        if file_idx == 0:
                            name_origin = one_file.split('_')[1]
                            data_type = data_dict[one_file]['dtype']
                            dim_1 = data_dict[one_file]['wav_data'].shape[0]
                            sr = data_dict[one_file]['sampling_rate']
                            break

                    audio_mm_arr = np.memmap(f"{self.root_directory}{os.sep}audio{os.sep}cropped_to_video{os.sep}{name_origin}"
                                             f"_concatenated_audio_{sr}_{dim_1}_{dim_2}_{str(data_type).split('.')[-1][:-2]}.mmap",
                                             dtype=data_type,
                                             mode='w+',
                                             shape=(dim_1, dim_2))

                    for file_idx, one_file in enumerate(data_dict.keys()):
                        audio_mm_arr[:, file_idx] = data_dict[one_file]['wav_data']

                    audio_mm_arr.flush()
                    audio_mm_arr = 0

                    # self.message_output(f"Audio concatenation completed at: {datetime.now().hour:02d}:{datetime.now().minute:02d}.{datetime.now().second:02d}")

                else:
                    self.message_output(f"Concatenation type input parameter {self.input_parameter_dict['concatenate_audio_files']['concat_type']} not recognized!")
                    sys.exit()

            else:
                self.message_output(f"There are <2 audio files per provided directory: '{self.root_directory}{os.sep}audio{os.sep}cropped_to_video', "
                                    f"so concatenation impossible.")

        else:
            self.message_output(f"Directory {self.root_directory}{os.sep}audio{os.sep}cropped_to_video does not exist.")

    def concatenate_video_files(self):
        """
        Description
        ----------
        This method concatenates video files via ffmpeg.
        ----------

        Parameters
        ----------
        Contains the following set of parameters
            camera_serial_num (list)
                Serial numbers of cameras used.
            video_extension (str)
                Video extension; defaults to 'mp4'.
            concatenated_video_name (str)
                Temporary name for concatenated video; defaults to 'concatenated_temp'.
        ----------

        Returns
        ----------
        concatenated_temp (video (e.g., .mp4) file)
            Concatenated video file.
        ----------
        """

        self.message_output(f"Video concatenation started at: {datetime.now().hour:02d}:{datetime.now().minute:02d}.{datetime.now().second:02d}")

        if os.path.exists(f"{self.root_directory}{os.sep}video"):
            for sub_directory in os.listdir(f"{self.root_directory}{os.sep}video"):
                if 'calibration' not in sub_directory \
                        and sub_directory.split('.')[-1] in self.input_parameter_dict['concatenate_video_files']['camera_serial_num']:

                    current_working_dir = f"{self.root_directory}{os.sep}video{os.sep}{sub_directory}"

                    vid_name = self.input_parameter_dict['concatenate_video_files']['concatenated_video_name']
                    vid_extension = self.input_parameter_dict['concatenate_video_files']['video_extension']
                    all_video_files = glob.glob(f"{current_working_dir}{os.sep}*.{vid_extension}")

                    if len(all_video_files) > 1:

                        # create .txt file with video files to concatenate
                        with open(f"{current_working_dir}{os.sep}file_concatenation_list.txt", 'w', encoding="utf-8") as concat_txt_file:
                            for file_path in all_video_files:
                                concat_txt_file.write(f"file '{file_path.split(os.sep)[-1]}'\n")

                        # concatenate videos
                        concat_subp = subprocess.Popen(f'''cmd /c "ffmpeg -loglevel warning -f concat -i file_concatenation_list.txt -c copy {vid_name}.{vid_extension}"''', cwd=current_working_dir)

                        while True:
                            status_poll = concat_subp.poll()
                            if status_poll is None:
                                QTest.qWait(1000)
                            else:
                                break

                    else:
                        continue
                        # self.message_output(f"There are <2 video files per provided subdirectory: '{sub_directory}', so concatenation impossible.")

    def rectify_video_fps(self):
        """
        Description
        ----------
        This method changes video sampling rate via ffmpeg.
        ----------

        Parameters
        ----------
        Contains the following set of parameters
            camera_serial_num (list)
                Serial numbers of cameras used.
            conversion_target_file (str)
                File to modify; defaults to 'concatenated_temp'.
            video_extension (str)
                Video extension; defaults to 'mp4'.
            calibration_fps (int)
                Desired sampling rate in calibration; defaults to 10 (fps).
            recording_fps (int)
                Desired sampling rate in recording; defaults to 150 (fps).
            delete_old_file (bool)
                Delete original file; defaults to True.
        ----------

        Returns
        ----------
        fps_corrected_video (video (e.g., .mp4) file)
            FPS modified video file.
        ----------
        """

        self.message_output(f"FPS modification started at: {datetime.now().hour:02d}:{datetime.now().minute:02d}.{datetime.now().second:02d}")

        date_joint = ''
        if os.path.exists(f"{self.root_directory}{os.sep}video"):
            for sd_idx, sub_directory in enumerate(os.listdir(f"{self.root_directory}{os.sep}video")):
                if sub_directory.split('.')[-1] in self.input_parameter_dict['rectify_video_fps']['camera_serial_num']:

                    if sd_idx == 0:
                        date_joint = sub_directory.split('.')[0].split('_')[-2] + sub_directory.split('.')[0].split('_')[-1]

                    current_working_dir = f"{self.root_directory}{os.sep}video{os.sep}{sub_directory}"
                    if os.path.isfile(f"{current_working_dir}{os.sep}{self.input_parameter_dict['rectify_video_fps']['conversion_target_file']}.{self.input_parameter_dict['rectify_video_fps']['video_extension']}"):
                        target_file = f"{self.input_parameter_dict['rectify_video_fps']['conversion_target_file']}.{self.input_parameter_dict['rectify_video_fps']['video_extension']}"
                    else:
                        target_file = f"000000.{self.input_parameter_dict['rectify_video_fps']['video_extension']}"
                    if 'calibration' in sub_directory:
                        new_file = f"{sub_directory.split('.')[-1]}-{date_joint}-calibration.{self.input_parameter_dict['rectify_video_fps']['video_extension']}"
                        new_fps = f"{self.input_parameter_dict['rectify_video_fps']['calibration_fps']}"
                    else:
                        new_file = f"{sub_directory.split('.')[-1]}-{date_joint}.{self.input_parameter_dict['rectify_video_fps']['video_extension']}"
                        new_fps = f"{self.input_parameter_dict['rectify_video_fps']['recording_fps']}"

                    if os.path.isfile(f"{current_working_dir}{os.sep}{target_file}"):

                        # change video sampling rate
                        fps_subp = subprocess.Popen(f'''cmd /c "ffmpeg -loglevel warning -y -r {new_fps} -i {target_file} -fps_mode passthrough -crf 16.4 -preset veryfast {new_file}"''', cwd=current_working_dir)

                        while True:
                            status_poll = fps_subp.poll()
                            if status_poll is None:
                                QTest.qWait(1000)
                            else:
                                break

                        if not os.path.exists(f"{self.root_directory}{os.sep}video{os.sep}{date_joint}"):
                            os.makedirs(f"{self.root_directory}{os.sep}video{os.sep}{date_joint}", exist_ok=False)

                        if not os.path.exists(f"{self.root_directory}{os.sep}video{os.sep}{date_joint}{os.sep}{sub_directory.split('.')[-1]}"):
                            os.makedirs(f"{self.root_directory}{os.sep}video{os.sep}{date_joint}{os.sep}{sub_directory.split('.')[-1]}", exist_ok=False)

                        if not os.path.exists(f"{self.root_directory}{os.sep}video{os.sep}{date_joint}{os.sep}{sub_directory.split('.')[-1]}{os.sep}calibration_images"):
                            os.makedirs(f"{self.root_directory}{os.sep}video{os.sep}{date_joint}{os.sep}{sub_directory.split('.')[-1]}{os.sep}calibration_images", exist_ok=False)

                        # copy files to special directory
                        if 'calibration' in sub_directory:
                            shutil.move(f"{current_working_dir}{os.sep}{new_file}",
                                        f"{self.root_directory}{os.sep}video{os.sep}{date_joint}{os.sep}{sub_directory.split('.')[-1]}{os.sep}calibration_images{os.sep}{new_file}")
                        else:
                            shutil.move(f"{current_working_dir}{os.sep}{new_file}",
                                        f"{self.root_directory}{os.sep}video{os.sep}{date_joint}{os.sep}{sub_directory.split('.')[-1]}{os.sep}{new_file}")

                        # clean video directory of all unnecessary files
                        if self.input_parameter_dict['rectify_video_fps']['delete_old_file']:
                            if os.path.isfile(f"{current_working_dir}{os.sep}{self.input_parameter_dict['rectify_video_fps']['conversion_target_file']}.{self.input_parameter_dict['rectify_video_fps']['video_extension']}"):
                                os.remove(f"{current_working_dir}{os.sep}{target_file}")
                                os.remove(f"{current_working_dir}{os.sep}file_concatenation_list.txt")

                    else:
                        self.message_output(f"In subdirectory '{sub_directory}', the file '{target_file}' does not exist.")
