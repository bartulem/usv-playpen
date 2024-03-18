"""
@author: bartulem
Manipulates files:
(1) break from multi to single channel, band-pass filter and temporally concatenate audio (e.g., wav) files
(2) concatenate video (e.g., mp4) files and change video (e.g., mp4) sampling rate (fps)
"""

from PyQt6.QtTest import QTest
import glob
import json
import os
import shutil
import numpy as np
import subprocess
from datetime import datetime
from imgstore import new_for_filename
from file_loader import DataLoader
from file_writer import DataWriter


class Operator:

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
            os.makedirs(f"{self.root_directory}{os.sep}audio{os.sep}temp")

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
        separation_subprocesses = []
        for device_id in ['m', 's']:
            for ch in range(1, 13):
                mc_to_sc_subp = subprocess.Popen(f'''cmd /c "sox {device_id}_*_ch{ch:02d}.wav -q {self.root_directory}{os.sep}audio{os.sep}original{os.sep}{device_id}_{name_origin}_ch{ch:02d}.wav"''',
                                                 cwd=f"{self.root_directory}{os.sep}audio{os.sep}temp")

                separation_subprocesses.append(mc_to_sc_subp)

        while True:
            status_poll = [query_subp.poll() for query_subp in separation_subprocesses]
            if any(elem is None for elem in status_poll):
                QTest.qWait(5000)
            else:
                break

        # delete temp directory (w/ all files in it)
        shutil.rmtree(f"{self.root_directory}{os.sep}audio{os.sep}temp")

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

        QTest.qWait(1000)

        for one_dir in self.input_parameter_dict['filter_audio_files']['filter_dirs']:

            if not os.path.exists(f"{self.root_directory}{os.sep}audio{os.sep}{one_dir}_filtered"):
                os.makedirs(f"{self.root_directory}{os.sep}audio{os.sep}{one_dir}_filtered")

            filter_subprocesses = []
            if os.path.exists(f"{self.root_directory}{os.sep}audio{os.sep}{one_dir}"):
                all_audio_files = glob.glob(f"{self.root_directory}{os.sep}audio{os.sep}{one_dir}{os.sep}"
                                            f"*.{self.input_parameter_dict['filter_audio_files']['audio_format']}")

                if len(all_audio_files) > 0:
                    for one_file in all_audio_files:
                        filter_subp = subprocess.Popen(f'''cmd /c "sox {one_file.split(os.sep)[-1]} {self.root_directory}{os.sep}audio{os.sep}{one_dir}_filtered{os.sep}{one_file.split(os.sep)[-1][:-4]}_filtered.wav sinc {freq_hp}-{freq_lp}"''',
                                                       cwd=f"{self.root_directory}{os.sep}audio{os.sep}{one_dir}")

                        filter_subprocesses.append(filter_subp)

            while True:
                status_poll = [query_subp.poll() for query_subp in filter_subprocesses]
                if any(elem is None for elem in status_poll):
                    QTest.qWait(5000)
                else:
                    break

    def concatenate_audio_files(self):
        """
        Description
        ----------
        This method concatenates audio files into a memmap array.
        ----------

        Parameters
        ----------
        Contains the following set of parameters
            audio_format (str)
                The format of the audio files; defaults to 'wav'.
        ----------

        Returns
        ----------
        memmap file
            Concatenated wave file (shape: n_channels X n_samples).
        ----------
        """

        self.message_output(f"Audio concatenation started at: {datetime.now().hour:02d}:{datetime.now().minute:02d}.{datetime.now().second:02d}")

        QTest.qWait(2000)

        for audio_file_type in self.input_parameter_dict['concatenate_audio_files']['concat_dirs']:

            all_audio_files = glob.glob(f"{self.root_directory}{os.sep}audio{os.sep}{audio_file_type}{os.sep}"
                                        f"*.{self.input_parameter_dict['concatenate_audio_files']['audio_format']}")

            if len(all_audio_files) > 1:

                data_dict = DataLoader(input_parameter_dict={'wave_data_loc': [f"{self.root_directory}{os.sep}audio{os.sep}{audio_file_type}"],
                                                             'load_wavefile_data': {'library': 'scipy', 'conditional_arg': []}}).load_wavefile_data()

                name_origin = list(data_dict.keys())[0].split('_')[1]
                dim_1 = data_dict[list(data_dict.keys())[0]]['wav_data'].shape[0]
                dim_2 = len(data_dict.keys())
                data_type = data_dict[list(data_dict.keys())[0]]['dtype']
                sr = data_dict[list(data_dict.keys())[0]]['sampling_rate']
                complete_mm_file_name = (f"{self.root_directory}{os.sep}audio{os.sep}{audio_file_type}{os.sep}{name_origin}"
                                         f"_concatenated_audio_{audio_file_type}_{sr}_{dim_1}_{dim_2}_{str(data_type).split('.')[-1][:-2]}.mmap")

                with np.memmap(filename=complete_mm_file_name, dtype=data_type, mode='w+', shape=(dim_1, dim_2)) as audio_mm_arr:
                    for file_idx, one_file in enumerate(data_dict.keys()):
                        audio_mm_arr[:, file_idx] = data_dict[one_file]['wav_data']

            else:
                self.message_output(f"There are <2 audio files per provided directory: '{self.root_directory}{os.sep}audio{os.sep}cropped_to_video', "
                                    f"so concatenation impossible.")

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

        subprocesses = []

        for sub_directory in os.listdir(f"{self.root_directory}{os.sep}video"):
            if 'calibration' not in sub_directory \
                    and sub_directory.split('.')[-1] in self.input_parameter_dict['concatenate_video_files']['camera_serial_num']:

                current_working_dir = f"{self.root_directory}{os.sep}video{os.sep}{sub_directory}"

                vid_name = f"{self.input_parameter_dict['concatenate_video_files']['concatenated_video_name']}_{sub_directory.split('.')[-1]}"
                vid_extension = self.input_parameter_dict['concatenate_video_files']['video_extension']
                all_video_files = glob.glob(f"{current_working_dir}{os.sep}*.{vid_extension}")

                if len(all_video_files) > 1:

                    # create .txt file with video files to concatenate
                    with open(f"{current_working_dir}{os.sep}file_concatenation_list_{sub_directory.split('.')[-1]}.txt", 'w', encoding="utf-8") as concat_txt_file:
                        for file_path in all_video_files:
                            concat_txt_file.write(f"file '{file_path.split(os.sep)[-1]}'\n")

                    # concatenate videos
                    one_subprocess = subprocess.Popen(f'''cmd /c "ffmpeg -loglevel warning -f concat -i file_concatenation_list_{sub_directory.split('.')[-1]}.txt -c copy {vid_name}.{vid_extension}"''',
                                                      stdout=subprocess.PIPE,
                                                      cwd=current_working_dir)

                    subprocesses.append(one_subprocess)

        while True:
            status_poll = [query_subp.poll() for query_subp in subprocesses]
            if any(elem is None for elem in status_poll):
                QTest.qWait(5000)
            else:
                break

        #  copy files over to video directory
        for sub_directory in os.listdir(f"{self.root_directory}{os.sep}video"):
            if 'calibration' not in sub_directory \
                    and sub_directory.split('.')[-1] in self.input_parameter_dict['concatenate_video_files']['camera_serial_num']:

                current_working_dir = f"{self.root_directory}{os.sep}video{os.sep}{sub_directory}"

                os.remove(f"{current_working_dir}{os.sep}file_concatenation_list_{sub_directory.split('.')[-1]}.txt")
                shutil.move(f"{current_working_dir}{os.sep}{self.input_parameter_dict['concatenate_video_files']['concatenated_video_name']}_{sub_directory.split('.')[-1]}.{self.input_parameter_dict['concatenate_video_files']['video_extension']}",
                            f"{self.root_directory}{os.sep}video{os.sep}{self.input_parameter_dict['concatenate_video_files']['concatenated_video_name']}_{sub_directory.split('.')[-1]}.{self.input_parameter_dict['concatenate_video_files']['video_extension']}")

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
        camera_frame_count_dict (.json file)
            Dictionary with camera frame counts,
            empirical capture rates and total video time.
        ----------
        """

        self.message_output(f"FPS modification started at: {datetime.now().hour:02d}:{datetime.now().minute:02d}.{datetime.now().second:02d}")

        date_joint = ''
        total_frame_number = 1e9
        total_video_time = 1e9
        camera_frame_count_dict = {}
        empirical_camera_sr = np.zeros(len(self.input_parameter_dict['rectify_video_fps']['camera_serial_num']))
        empirical_camera_sr[:] = np.nan
        camera_idx = 0

        fsp_subprocesses = []
        for sd_idx, sub_directory in enumerate(sorted(os.listdir(f"{self.root_directory}{os.sep}video"))):
            if (os.path.isdir(f"{self.root_directory}{os.sep}video{os.sep}{sub_directory}")
                    and '.' in sub_directory
                    and sub_directory.split('.')[-1] in self.input_parameter_dict['rectify_video_fps']['camera_serial_num']):

                if camera_idx == 0:
                    date_joint = sub_directory.split('.')[0].split('_')[-2] + sub_directory.split('.')[0].split('_')[-1]

                # get frame count and empirical sampling rate
                img_store = new_for_filename(f"{self.root_directory}{os.sep}video{os.sep}{sub_directory}{os.sep}metadata.yaml")
                total_frame_num = img_store.frame_count
                last_frame_num = img_store.frame_max
                frame_times = img_store.get_frame_metadata()['frame_time']
                video_duration = frame_times[-1] - frame_times[0]
                esr = round(number=total_frame_num / video_duration, ndigits=3)
                if 'calibration' not in sub_directory:
                    empirical_camera_sr[camera_idx] = esr
                    camera_frame_count_dict[sub_directory.split('.')[-1]] = (total_frame_num, esr)
                    if total_frame_num == last_frame_num:
                        self.message_output(f"Camera {sub_directory.split('.')[-1]} has {total_frame_num} total frames, no dropped frames, "
                                            f"video duration of {video_duration:.4f} seconds, and empirical sampling rate of {esr} fps.")
                        if total_frame_num < total_frame_number:
                            total_frame_number = total_frame_num
                        if video_duration < total_video_time:
                            total_video_time = video_duration
                    else:
                        self.message_output(f"WARNING: The last frame on camera {sub_directory.split('.')[-1]} is {last_frame_num}, which is more than {total_frame_num} in total, "
                                            f"suggesting dropped frames. The video duration is {video_duration:.4f} seconds")
                    camera_idx += 1

                crf = self.input_parameter_dict['rectify_video_fps']['constant_rate_factor']
                enc_preset = self.input_parameter_dict['rectify_video_fps']['encoding_preset']

                current_working_dir = f"{self.root_directory}{os.sep}video"
                if os.path.isfile(f"{current_working_dir}{os.sep}{self.input_parameter_dict['rectify_video_fps']['conversion_target_file']}_{sub_directory.split('.')[-1]}.{self.input_parameter_dict['rectify_video_fps']['video_extension']}"):
                    target_file = f"{self.input_parameter_dict['rectify_video_fps']['conversion_target_file']}_{sub_directory.split('.')[-1]}.{self.input_parameter_dict['rectify_video_fps']['video_extension']}"
                else:
                    current_working_dir = f"{self.root_directory}{os.sep}video{os.sep}{sub_directory}"
                    target_file = f"000000.{self.input_parameter_dict['rectify_video_fps']['video_extension']}"

                if 'calibration' in sub_directory:
                    new_file = f"{sub_directory.split('.')[-1]}-{date_joint}-calibration.{self.input_parameter_dict['rectify_video_fps']['video_extension']}"
                else:
                    new_file = f"{sub_directory.split('.')[-1]}-{date_joint}.{self.input_parameter_dict['rectify_video_fps']['video_extension']}"

                # change video sampling rate
                fps_subp = subprocess.Popen(f'''cmd /c "ffmpeg -loglevel warning -y -r {esr} -i {target_file} -fps_mode passthrough -crf {crf} -preset {enc_preset} {new_file}"''',
                                            stdout=subprocess.PIPE,
                                            cwd=current_working_dir)

                fsp_subprocesses.append(fps_subp)

        while True:
            status_poll = [query_subp.poll() for query_subp in fsp_subprocesses]
            if any(elem is None for elem in status_poll):
                QTest.qWait(5000)
            else:
                break

        # move files to special directory
        for sd_idx, sub_directory in enumerate(sorted(os.listdir(f"{self.root_directory}{os.sep}video"))):
            if (os.path.isdir(f"{self.root_directory}{os.sep}video{os.sep}{sub_directory}")
                    and '.' in sub_directory
                    and sub_directory.split('.')[-1] in self.input_parameter_dict['rectify_video_fps']['camera_serial_num']):

                if not os.path.exists(f"{self.root_directory}{os.sep}video{os.sep}{date_joint}{os.sep}{sub_directory.split('.')[-1]}{os.sep}calibration_images"):
                    os.makedirs(f"{self.root_directory}{os.sep}video{os.sep}{date_joint}{os.sep}{sub_directory.split('.')[-1]}{os.sep}calibration_images")

                current_working_dir = f"{self.root_directory}{os.sep}video"
                if os.path.isfile(f"{current_working_dir}{os.sep}{self.input_parameter_dict['rectify_video_fps']['conversion_target_file']}_{sub_directory.split('.')[-1]}.{self.input_parameter_dict['rectify_video_fps']['video_extension']}"):
                    target_file = f"{self.input_parameter_dict['rectify_video_fps']['conversion_target_file']}_{sub_directory.split('.')[-1]}.{self.input_parameter_dict['rectify_video_fps']['video_extension']}"
                else:
                    current_working_dir = f"{self.root_directory}{os.sep}video{os.sep}{sub_directory}"
                    target_file = f"000000.{self.input_parameter_dict['rectify_video_fps']['video_extension']}"
                if 'calibration' in sub_directory:
                    new_file = f"{sub_directory.split('.')[-1]}-{date_joint}-calibration.{self.input_parameter_dict['rectify_video_fps']['video_extension']}"
                else:
                    new_file = f"{sub_directory.split('.')[-1]}-{date_joint}.{self.input_parameter_dict['rectify_video_fps']['video_extension']}"

                if 'calibration' in sub_directory:
                    shutil.move(f"{current_working_dir}{os.sep}{new_file}",
                                f"{self.root_directory}{os.sep}video{os.sep}{date_joint}{os.sep}{sub_directory.split('.')[-1]}{os.sep}calibration_images{os.sep}{new_file}")
                else:
                    shutil.move(f"{current_working_dir}{os.sep}{new_file}",
                                f"{self.root_directory}{os.sep}video{os.sep}{date_joint}{os.sep}{sub_directory.split('.')[-1]}{os.sep}{new_file}")

                # clean video directory of all unnecessary files
                if self.input_parameter_dict['rectify_video_fps']['delete_old_file']:
                    if os.path.isfile(f"{current_working_dir}{os.sep}{self.input_parameter_dict['rectify_video_fps']['conversion_target_file']}_{sub_directory.split('.')[-1]}.{self.input_parameter_dict['rectify_video_fps']['video_extension']}"):
                        os.remove(f"{current_working_dir}{os.sep}{target_file}")

        # save camera_frame_count_dict to a file
        camera_frame_count_dict['total_frame_number_least'] = total_frame_number
        camera_frame_count_dict['total_video_time_least'] = total_video_time
        camera_frame_count_dict['median_empirical_camera_sr'] = round(number=np.median(empirical_camera_sr), ndigits=3)
        with open(f"{self.root_directory}{os.sep}video{os.sep}{date_joint}_camera_frame_count_dict.json", 'w') as frame_count_outfile:
            json.dump(camera_frame_count_dict, frame_count_outfile, indent=4)
