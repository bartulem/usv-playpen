"""
@author: bartulem
Different functions for modifying files:
(1a) break from multi to single channel
(1b) perform harmonic-percussive source separation
(1c) perform band-pass filtering
(1d) concatenate single channel audio (e.g., wav) files
(2a) concatenate video (e.g., mp4) files
(2b) change video (e.g., mp4) sampling rate (fps)
(3a) concatenate e-phys binary files
(3b) split manually curated clusters into sessions
"""

from PyQt6.QtTest import QTest
import configparser
import glob
import json
import librosa
import os
import pandas as pd
import pathlib
import shutil
import numpy as np
import subprocess
from datetime import datetime
from imgstore import new_for_filename
from scipy.io import wavfile
from tqdm import tqdm
from .load_audio_files import DataLoader
from .os_utils import configure_path


class Operator:
    if os.name == 'nt':
        command_addition = 'cmd /c '
        shell_usage_bool = False
    else:
        command_addition = ''
        shell_usage_bool = True

    def __init__(self, root_directory=None, input_parameter_dict=None,
                 exp_settings_dict=None, message_output=None):
        if input_parameter_dict is None:
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '_parameter_settings/processing_settings.json'), 'r') as json_file:
                self.input_parameter_dict = json.load(json_file)['modify_files']['Operator']
                self.input_parameter_dict_2 = json.load(json_file)['synchronize_files']['Synchronizer']
        else:
            self.input_parameter_dict = input_parameter_dict['modify_files']['Operator']
            self.input_parameter_dict_2 = input_parameter_dict['synchronize_files']['Synchronizer']

        if root_directory is None:
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '_parameter_settings/processing_settings.json'), 'r') as json_file:
                self.root_directory = json.load(json_file)['modify_files']['root_directory']
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

    def split_clusters_to_sessions(self):
        """
        Description
        ----------
        This method converts every spike sample time into seconds,
        relative to tracking start and splits spikes back into
        individual sessions (if binary files were concatenated).

        NB: If you have recorded multiple sessions in one day,
        it is sufficient to put only one root directory for that day,
        e.g., the first one. The script will find EPHYS root directory,
        and split spikes from all probes into sessions based on the
        inputs in the changepoints JSON file.
        ----------

        Parameter
        ---------
        root_directory : list
             Directories of recording files of interest;
        calibrated_sample_rates_file : str
            Configuration file containing calibrated sampling rates for headstages.
        kilosort_version : str
            Kilosort version used for spike sorting.
        min_num_spikes : int
            Minimum relevant number of spikes per session.

        Returns
        -------
         spike times : np.ndarray
            Arrays that contain spike times (in seconds and frames);
            saved as .npy files in a separate directory.
        """

        self.message_output(f"Splitting clusters to sessions started at: {datetime.now().hour:02d}:{datetime.now().minute:02d}.{datetime.now().second:02d}")
        QTest.qWait(1000)

        # read headstage sampling rates
        calibrated_sr_config = configparser.ConfigParser()
        calibrated_sr_config.read(os.path.join(os.path.dirname(os.path.abspath(__file__)), '_config/calibrated_sample_rates_imec.ini'))

        for one_root_dir in self.root_directory:
            for ephys_dir in sorted(glob.glob(pathname=f"{one_root_dir.replace('Data', 'EPHYS')[:-7]}_imec*", recursive=True)):

                probe_id = ephys_dir.split('_')[-1]

                self.message_output(f"Working on getting spike times from clusters in: {ephys_dir}, started at {datetime.now()}.")

                # load the changepoint .json file
                with open(sorted(glob.glob(pathname=f'{ephys_dir}{os.sep}changepoints_info_*.json', recursive=True))[0], 'r') as binary_info_input_file:
                    binary_files_info = json.load(binary_info_input_file)

                    for session_key in binary_files_info.keys():
                        binary_files_info[session_key]['root_directory'] = configure_path(pa=binary_files_info[session_key]['root_directory'])

                # get info about session start
                se_dict = {}
                esr_dict = {}
                frame_least_dict = {}
                root_dict = {}
                unit_count_dict = {'noise': 0, 'unsorted': 0}
                for session_key in binary_files_info.keys():

                    unit_count_dict[session_key] = {'good': 0, 'mua': 0}

                    # load info from camera_frame_count_dict
                    with open(sorted(glob.glob(f"{binary_files_info[session_key]['root_directory']}{os.sep}video{os.sep}*_camera_frame_count_dict.json"))[0], 'r') as frame_count_infile:
                        camera_frame_info = json.load(frame_count_infile)
                        esr_dict[session_key] = camera_frame_info['median_empirical_camera_sr']
                        frame_least_dict[session_key] = camera_frame_info['total_frame_number_least']
                        root_dict[session_key] = binary_files_info[session_key]['root_directory']

                    if any([np.isnan(value) for value in binary_files_info[session_key]['tracking_start_end']]):
                        se_dict[session_key] = binary_files_info[session_key]['session_start_end']
                    else:
                        se_dict[session_key] = binary_files_info[session_key]['tracking_start_end']

                    pathlib.Path(f'{root_dict[session_key]}{os.sep}ephys{os.sep}{probe_id}{os.sep}cluster_data').mkdir(parents=True, exist_ok=True)

                # load the Kilosort output files
                phy_curation_bool = os.path.isfile(f"{ephys_dir}{os.sep}kilosort{self.input_parameter_dict['get_spike_times']['kilosort_version']}{os.sep}cluster_info.tsv")
                spike_clusters = np.load(f"{ephys_dir}{os.sep}kilosort{self.input_parameter_dict['get_spike_times']['kilosort_version']}{os.sep}spike_clusters.npy")
                spike_times = np.load(f"{ephys_dir}{os.sep}kilosort{self.input_parameter_dict['get_spike_times']['kilosort_version']}{os.sep}spike_times.npy")

                if phy_curation_bool:
                    cluster_info = pd.read_csv(filepath_or_buffer=f"{ephys_dir}{os.sep}kilosort{self.input_parameter_dict['get_spike_times']['kilosort_version']}{os.sep}cluster_info.tsv",
                                               sep='\t')
                else:
                    self.message_output("Phy2 curation has not been done for this session, no cluster_info.tsv file exists.")
                    continue

                QTest.qWait(1000)

                for idx in tqdm(range(cluster_info.shape[0])):
                    if cluster_info.loc[idx, 'group'] == 'good' or cluster_info.loc[idx, 'group'] == 'mua':

                        # collect all spikes for any given cluster
                        cluster_indices = np.sort(np.where(spike_clusters == cluster_info.loc[idx, 'cluster_id'])[0])
                        spike_events = np.take(spike_times, cluster_indices)

                        # filter spikes for each session
                        for session_key in binary_files_info.keys():
                            session_spikes_sec = ((spike_events[(spike_events >= se_dict[session_key][0]) & (spike_events < se_dict[session_key][1])] - se_dict[session_key][0]) /
                                                  float(calibrated_sr_config['CalibratedHeadStages'][binary_files_info[session_key]['headstage_sn']]))

                            session_spikes_fps = np.round(session_spikes_sec * esr_dict[session_key])
                            session_spikes_fps[session_spikes_fps == frame_least_dict[session_key]] = frame_least_dict[session_key]-1

                            session_spikes = np.vstack((session_spikes_sec, session_spikes_fps))

                            # save spiking data
                            if session_spikes_sec.shape[0] > self.input_parameter_dict['get_spike_times']['min_spike_num']:
                                cluster_id = f"{probe_id}_cl{cluster_info.loc[idx, 'cluster_id']:04d}_ch{cluster_info.loc[idx, 'ch']:03d}_{cluster_info.loc[idx, 'group']}"
                                np.save(file=f'{root_dict[session_key]}{os.sep}ephys{os.sep}{probe_id}{os.sep}cluster_data{os.sep}{cluster_id}', arr=session_spikes)

                                unit_count_dict[session_key][cluster_info.loc[idx, 'group']] += 1

                    elif cluster_info.loc[idx, 'group'] == 'noise':
                        unit_count_dict['noise'] += 1

                    else:
                        unit_count_dict['unsorted'] += 1

                self.message_output(f"For {ephys_dir}, there were {unit_count_dict['noise']} noise clusters and {unit_count_dict['unsorted']} unsorted clusters.")
                for session_key in binary_files_info.keys():
                    self.message_output(f"For {root_dict[session_key]} probe {probe_id}, there were {unit_count_dict[session_key]['good']} good and {unit_count_dict[session_key]['mua']} MUA clusters.")

    def concatenate_binary_files(self):
        """
        Description
        ----------
        This method concatenates binary files from Neuropixels recordings into one
        .bin file (can be used from "ap" or "lf" files). It goes through all root
        directories and concatenates all binary files for any given probe, say "imec0",
        into one binary file.

        NB: If you have recorded multiple sessions in one day,
        it is necessary to list all of their root directories
        to conduct concatenation. The script operates by first
        locating all available probes in the root directories,
        and then concatenates all binary files for each probe.
        ----------

        Parameter
        ---------
        root_directory : list
             Directories of recording files of interest;
        npx_file_type : str
            AP or LF binary file; defaults to 'ap'.
        calibrated_sample_rates_file : str
            Configuration file containing calibrated sampling rates for headstages.

        Returns
        -------
        binary_files_info : .json
            Dictionary w/ information about changepoints and binary file lengths.
        concatenated : .bin
           Concatenated binary file.
        """

        self.message_output(f"E-phys file concatenation started at: {datetime.now().hour:02d}:{datetime.now().minute:02d}.{datetime.now().second:02d}. "
                            f"Please be patient - this could take >1 hour.")
        QTest.qWait(1000)

        # read headstage sampling rates
        calibrated_sr_config = configparser.ConfigParser()
        calibrated_sr_config.read(os.path.join(os.path.dirname(os.path.abspath(__file__)), '_config/calibrated_sample_rates_imec.ini'))

        # create list of directories to save concatenated files in
        concat_save_dir = []
        available_probes = []
        for ord_idx, one_root_dir in enumerate(self.root_directory):
            ephys_save_dir_base = one_root_dir.replace('Data', 'EPHYS')[:-7]
            for one_probe_dir in os.listdir(f'{one_root_dir}{os.sep}ephys'):
                if one_probe_dir not in available_probes:
                    available_probes.append(one_probe_dir)
                if os.path.isdir(f'{one_root_dir}{os.sep}ephys{os.sep}{one_probe_dir}') and 'imec' in one_probe_dir:
                    if not any([True if one_probe_dir in one_concat_dir else False for one_concat_dir in concat_save_dir]):
                        concat_save_dir.append(f'{ephys_save_dir_base}_{one_probe_dir}')

        npx_file_type = self.input_parameter_dict_2['validate_ephys_video_sync']['npx_file_type']
        # create dictionary to store information about binary files and generate stitching command
        for probe_idx, probe_id in enumerate(available_probes):
            binary_files_info = {}
            changepoints = [0]
            concatenation_command = 'copy /b ' if os.name == 'nt' else 'cat '
            for ord_idx, one_root_dir in enumerate(self.root_directory):
                if os.path.isdir(f'{one_root_dir}{os.sep}ephys{os.sep}{probe_id}'):
                    for one_file, one_meta in zip(sorted(list(pathlib.Path(f'{one_root_dir}{os.sep}ephys{os.sep}{probe_id}').glob(f"*{npx_file_type}.bin*"))),
                                                  sorted(list(pathlib.Path(f'{one_root_dir}{os.sep}ephys{os.sep}{probe_id}').glob(f"*{npx_file_type}.meta*")))):
                        if one_file.is_file() and one_meta.is_file():

                            # parse metadata file for channel and headstage information
                            with open(one_meta) as meta_data_file:
                                for line in meta_data_file:
                                    key, value = line.strip().split("=")
                                    if key == 'acqApLfSy':
                                        total_num_channels = int(value.split(',')[0]) + int(value.split(',')[-1])
                                    elif key == 'imDatHs_sn':
                                        headstage_sn = value
                                        spike_glx_sr = float(calibrated_sr_config['CalibratedHeadStages'][headstage_sn])
                                    elif key == 'imDatPrb_sn':
                                        imec_probe_sn = value

                            binary_file_info_id = pathlib.Path(one_file).name.split(os.sep)[-1][:-7]
                            binary_files_info[binary_file_info_id] = {'session_start_end': [np.nan, np.nan],
                                                                      'tracking_start_end': [np.nan, np.nan],
                                                                      'largest_camera_break_duration': np.nan,
                                                                      'file_duration_samples': np.nan,
                                                                      'root_directory': one_root_dir,
                                                                      'total_num_channels': total_num_channels,
                                                                      'headstage_sn': headstage_sn,
                                                                      'imec_probe_sn': imec_probe_sn}

                            one_recording = np.memmap(filename=one_file, mode='r', dtype='int16', order='C')

                            self.message_output(f"File {pathlib.Path(one_file).name}, recorded with hs #{headstage_sn} & probe #{imec_probe_sn} has total length {one_recording.shape[0]}, or {one_recording.shape[0] // total_num_channels} "
                                                f"samples on {total_num_channels} channels, totaling {round((one_recording.shape[0] // total_num_channels) / (spike_glx_sr * 60), 2)} minutes of recording.")

                            binary_files_info[binary_file_info_id]['file_duration_samples'] = int(one_recording.shape[0] // total_num_channels)

                            if len(changepoints) == 1:
                                binary_files_info[binary_file_info_id]['session_start_end'][0] = 0
                                binary_files_info[binary_file_info_id]['session_start_end'][1] = int(one_recording.shape[0] // total_num_channels)
                                changepoints.append(int(one_recording.shape[0] // total_num_channels))
                                concatenation_command += '{} '.format(one_file)
                            else:
                                binary_files_info[binary_file_info_id]['session_start_end'][0] = changepoints[-1]
                                binary_files_info[binary_file_info_id]['session_start_end'][1] = int(one_recording.shape[0] // total_num_channels) + changepoints[-1]
                                changepoints.append(int(one_recording.shape[0] // total_num_channels) + changepoints[-1])
                                if os.name == 'nt':
                                    concatenation_command += '+ {} '.format(one_file)
                                else:
                                    concatenation_command += '{} '.format(one_file)

            if os.name == 'nt':
                concatenation_command += f'"{concat_save_dir[probe_idx]}{os.sep}concatenated_{concat_save_dir[probe_idx].split(os.sep)[-1]}.{npx_file_type}.bin"'
            else:
                concatenation_command += f'> {concat_save_dir[probe_idx]}{os.sep}concatenated_{concat_save_dir[probe_idx].split(os.sep)[-1]}.{npx_file_type}.bin'

            # create save directory if one doesn't exist already
            pathlib.Path(concat_save_dir[probe_idx]).mkdir(parents=True, exist_ok=True)

            # save changepoint information in JSON file
            if not os.path.exists(f'{concat_save_dir[probe_idx]}{os.sep}changepoints_info_{concat_save_dir[probe_idx].split(os.sep)[-1]}.json'):
                with open(f'{concat_save_dir[probe_idx]}{os.sep}changepoints_info_{concat_save_dir[probe_idx].split(os.sep)[-1]}.json', 'w') as binary_info_output_file:
                    json.dump(binary_files_info, binary_info_output_file, indent=4)
            else:
                with open(f'{concat_save_dir[probe_idx]}{os.sep}changepoints_info_{concat_save_dir[probe_idx].split(os.sep)[-1]}.json', 'r') as existing_json_file:
                    changepoint_info_data = json.load(existing_json_file)

                for file_key in binary_files_info.keys():
                    if file_key not in changepoint_info_data.keys():
                        changepoint_info_data[file_key] = binary_files_info[file_key]
                    else:
                        for component_key in changepoint_info_data[file_key].keys():
                            if component_key != 'tracking_start_end' and component_key != 'largest_camera_break_duration' and component_key != 'root_directory' and changepoint_info_data[file_key][component_key] != binary_files_info[file_key][component_key]:
                                changepoint_info_data[file_key][component_key] = binary_files_info[file_key][component_key]
                            elif component_key == 'tracking_start_end' and changepoint_info_data[file_key][component_key] != [np.nan, np.nan]:
                                changepoint_info_data[file_key][component_key][0] = changepoint_info_data[file_key][component_key][0] + binary_files_info[file_key]['session_start_end'][0]
                                changepoint_info_data[file_key][component_key][1] = changepoint_info_data[file_key][component_key][1] + binary_files_info[file_key]['session_start_end'][0]

                with open(f'{concat_save_dir[probe_idx]}{os.sep}changepoints_info_{concat_save_dir[probe_idx].split(os.sep)[-1]}.json', 'w') as binary_info_output_file:
                    json.dump(changepoint_info_data, binary_info_output_file, indent=4)

            QTest.qWait(2000)

            # run command in shell
            subprocess.Popen(args=f"{self.command_addition}{concatenation_command}",
                             shell=True,
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.STDOUT,
                             cwd=concat_save_dir[probe_idx]).wait()

    def multichannel_to_channel_audio(self):
        """
        Description
        ----------
        This method splits multichannel audio file into single channel files and
        concatenates single channel files via Sox, since multichannel files where
        split due to a size limitation.

        NB: You need to install sox: https://sourceforge.net/projects/sox/files/sox/
        and add the sox directory to your system PATH prior to running this.
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
        QTest.qWait(1000)

        pathlib.Path(f"{self.root_directory}{os.sep}audio{os.sep}temp").mkdir(parents=True, exist_ok=True)

        mc_audio_files = DataLoader(input_parameter_dict={'wave_data_loc': [f"{self.root_directory}{os.sep}audio{os.sep}original_mc"],
                                                          'load_wavefile_data': {'library': 'scipy', 'conditional_arg': []}}).load_wavefile_data()

        # split multichannel files
        for mc_audio_file in mc_audio_files.keys():
            for ch in range(mc_audio_files[mc_audio_file]['wav_data'].shape[1]):
                wavfile.write(filename=f"{self.root_directory}{os.sep}audio{os.sep}temp{os.sep}{mc_audio_file[:-4]}_ch{ch + 1:02d}.wav",
                              rate=int(mc_audio_files[mc_audio_file]['sampling_rate']),
                              data=mc_audio_files[mc_audio_file]['wav_data'][:, ch])

        # release dict from memory
        mc_audio_files = 0

        # find name origin for file naming purposes
        name_origin = sorted(glob.glob(f"{self.root_directory}{os.sep}audio{os.sep}temp{os.sep}m_*_ch*.wav"))[0].split('_')[2]

        # concatenate single channel files for master/slave
        separation_subprocesses = []
        for device_id in ['m', 's']:
            for ch in range(1, 13):
                mc_to_sc_subp = subprocess.Popen(args=f'''{self.command_addition}sox {device_id}_*_ch{ch:02d}.wav -q {self.root_directory}{os.sep}audio{os.sep}original{os.sep}{device_id}_{name_origin}_ch{ch:02d}.wav''',
                                                 cwd=f"{self.root_directory}{os.sep}audio{os.sep}temp",
                                                 shell=self.shell_usage_bool)

                separation_subprocesses.append(mc_to_sc_subp)

        while True:
            status_poll = [query_subp.poll() for query_subp in separation_subprocesses]
            if any(elem is None for elem in status_poll):
                QTest.qWait(5000)
            else:
                break

        # delete temp directory (w/ all files in it)
        shutil.rmtree(f"{self.root_directory}{os.sep}audio{os.sep}temp")

    def hpss_audio(self):
        """
        Description
        ----------
        This function performs the harmonic/percussive source separation (HPSS)
        on the provided audio (WAV) files. The harmonic component is then converted
        back to the time domain and saved as a new WAV file.
        ----------

        Parameter
        ---------
        stft_window_length_hop_size : list
            Length of the window and hop size for the STFT (Short-Time Fourier Transform).
        kernel_size : tuple
            Size of the kernel for the HPSS (Harmonic / Percussive components).
        hpss_power : float
            Exponent for the HPSS.
        margin : tuple
            Margin for the HPSS (Harmonic / Percussive components).

        Returns
        -------
        harmonic_data_clipped : WAV file(s)
            Output audio file w/ only the harmonics component.
        """

        self.message_output(f"Harmonic-percussive source separation started at: {datetime.now().hour:02d}:{datetime.now().minute:02d}.{datetime.now().second:02d}")
        QTest.qWait(1000)

        wav_file_lst = sorted(glob.glob(f'{self.root_directory}{os.sep}audio{os.sep}cropped_to_video{os.sep}*.wav'))

        for one_wav_file in wav_file_lst:
            self.message_output(f"Working on file: {one_wav_file}")
            QTest.qWait(1000)

            # read the audio file (use Scipy, not Librosa because Librosa performs scaling)
            sampling_rate_audio, audio_data = wavfile.read(one_wav_file)

            # convert to float32 because librosa.stft() requires float32
            audio_data = np.array(audio_data, dtype='float32')

            # perform Short-Time Fourier Transform (STFT) on the audio data
            spectrogram_data = librosa.stft(y=audio_data,
                                            n_fft=self.input_parameter_dict['hpss_audio']['stft_window_length_hop_size'][0],
                                            hop_length=self.input_parameter_dict['hpss_audio']['stft_window_length_hop_size'][1])

            # perform HPSS on the spectrogram data
            D_harmonic, D_percussive = librosa.decompose.hpss(S=spectrogram_data,
                                                              kernel_size=self.input_parameter_dict['hpss_audio']['kernel_size'],
                                                              power=self.input_parameter_dict['hpss_audio']['hpss_power'],
                                                              mask=False,
                                                              margin=self.input_parameter_dict['hpss_audio']['margin'])

            # convert the harmonic component back to the time domain
            harmonic_data = librosa.istft(stft_matrix=D_harmonic,
                                          length=audio_data.shape[0],
                                          win_length=self.input_parameter_dict['hpss_audio']['stft_window_length_hop_size'][0],
                                          hop_length=self.input_parameter_dict['hpss_audio']['stft_window_length_hop_size'][1])

            # ensure the float values are within the range of 16-bit integers
            # clip values outside the range to the minimum and maximum representable values
            harmonic_data_clipped = np.clip(a=harmonic_data,
                                            a_min=-32768,
                                            a_max=32767).astype('int16')

            # save the harmonic component as a new WAV file
            new_dir = f"{self.root_directory}{os.sep}audio{os.sep}hpss"
            pathlib.Path(new_dir).mkdir(parents=True, exist_ok=True)

            raw_file_name = os.path.basename(one_wav_file)
            wavfile.write(filename=f'{new_dir}{os.sep}{raw_file_name[:-4]}_hpss.wav',
                          rate=sampling_rate_audio,
                          data=harmonic_data_clipped)

    def filter_audio_files(self):
        """
        Description
        ----------
        This method filters audio files via Sox.

        NB: You need to install sox: https://sourceforge.net/projects/sox/files/sox/
        and add the sox directory to your system PATH prior to running this.

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

        freq_lp = self.input_parameter_dict['filter_audio_files']['filter_freq_bounds'][0]
        freq_hp = self.input_parameter_dict['filter_audio_files']['filter_freq_bounds'][1]

        self.message_output(f"Filtering out signal between {freq_lp} and {freq_hp} Hz in audio files started at: "
                            f"{datetime.now().hour:02d}:{datetime.now().minute:02d}.{datetime.now().second:02d}")
        QTest.qWait(1000)

        for one_dir in self.input_parameter_dict['filter_audio_files']['filter_dirs']:

            pathlib.Path(f"{self.root_directory}{os.sep}audio{os.sep}{one_dir}_filtered").mkdir(parents=True, exist_ok=True)

            filter_subprocesses = []
            all_audio_files = sorted(glob.glob(f"{self.root_directory}{os.sep}audio{os.sep}{one_dir}{os.sep}"
                                               f"*.{self.input_parameter_dict['filter_audio_files']['audio_format']}"))

            if len(all_audio_files) > 0:
                for one_file in all_audio_files:
                    filter_subp = subprocess.Popen(args=f'''{self.command_addition}sox {one_file.split(os.sep)[-1]} {self.root_directory}{os.sep}audio{os.sep}{one_dir}_filtered{os.sep}{one_file.split(os.sep)[-1][:-4]}_filtered.wav sinc {freq_hp}-{freq_lp}''',
                                                   cwd=f"{self.root_directory}{os.sep}audio{os.sep}{one_dir}",
                                                   shell=self.shell_usage_bool)

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
        QTest.qWait(1000)

        for audio_file_type in self.input_parameter_dict['concatenate_audio_files']['concat_dirs']:

            all_audio_files = sorted(glob.glob(f"{self.root_directory}{os.sep}audio{os.sep}{audio_file_type}{os.sep}"
                                               f"*.{self.input_parameter_dict['concatenate_audio_files']['audio_format']}"))

            if len(all_audio_files) > 1:

                data_dict = DataLoader(input_parameter_dict={'wave_data_loc': [f"{self.root_directory}{os.sep}audio{os.sep}{audio_file_type}"],
                                                             'load_wavefile_data': {'library': 'scipy', 'conditional_arg': []}}).load_wavefile_data()

                name_origin = list(data_dict.keys())[0].split('_')[1]
                dim_1 = data_dict[list(data_dict.keys())[0]]['wav_data'].shape[0]
                dim_2 = len(data_dict.keys())
                sr = data_dict[list(data_dict.keys())[0]]['sampling_rate']
                complete_mm_file_name = (f"{self.root_directory}{os.sep}audio{os.sep}{audio_file_type}{os.sep}{name_origin}"
                                         f"_concatenated_audio_{audio_file_type}_{sr}_{dim_1}_{dim_2}_int16.mmap")

                audio_mm_arr = np.memmap(filename=complete_mm_file_name,
                                         dtype='int16',
                                         mode='w+',
                                         shape=(dim_1, dim_2))

                for file_idx, one_file in enumerate(data_dict.keys()):
                    audio_mm_arr[:, file_idx] = data_dict[one_file]['wav_data']

                audio_mm_arr.flush()

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
                all_video_files = sorted(glob.glob(f"{current_working_dir}{os.sep}*.{vid_extension}"))

                if len(all_video_files) > 1:

                    # create .txt file with video files to concatenate
                    with open(f"{current_working_dir}{os.sep}file_concatenation_list_{sub_directory.split('.')[-1]}.txt", 'w', encoding="utf-8") as concat_txt_file:
                        for file_path in all_video_files:
                            concat_txt_file.write(f"file '{file_path.split(os.sep)[-1]}'\n")

                    # concatenate videos
                    one_subprocess = subprocess.Popen(args=f'''{self.command_addition}ffmpeg -loglevel warning -f concat -i file_concatenation_list_{sub_directory.split('.')[-1]}.txt -c copy {vid_name}.{vid_extension}''',
                                                      stdout=subprocess.PIPE,
                                                      cwd=current_working_dir,
                                                      shell=self.shell_usage_bool)

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

    def rectify_video_fps(self, conduct_concat=True):
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

        self.message_output(f"Video re-encoding started at: {datetime.now().hour:02d}:{datetime.now().minute:02d}.{datetime.now().second:02d}")

        if not conduct_concat and len(next(os.walk(f"{self.root_directory}{os.sep}video"))[2]) == 0:
            for sub_directory in os.listdir(f"{self.root_directory}{os.sep}video"):
                if 'calibration' not in sub_directory \
                        and sub_directory.split('.')[-1] in self.input_parameter_dict['rectify_video_fps']['camera_serial_num']:
                    current_working_dir = f"{self.root_directory}{os.sep}video{os.sep}{sub_directory}"

                    shutil.copy(src=f"{current_working_dir}{os.sep}{self.input_parameter_dict['rectify_video_fps']['conversion_target_file']}.{self.input_parameter_dict['rectify_video_fps']['video_extension']}",
                                dst=f"{self.root_directory}{os.sep}video{os.sep}{self.input_parameter_dict['rectify_video_fps']['conversion_target_file']}_{sub_directory.split('.')[-1]}.{self.input_parameter_dict['rectify_video_fps']['video_extension']}")

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
                                            f"video duration of {video_duration:.4f} seconds, and sampling rate of {esr} fps.")
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
                if 'calibration' not in sub_directory:
                    target_file = f"{self.input_parameter_dict['rectify_video_fps']['conversion_target_file']}_{sub_directory.split('.')[-1]}.{self.input_parameter_dict['rectify_video_fps']['video_extension']}"
                    new_file = f"{sub_directory.split('.')[-1]}-{date_joint}.{self.input_parameter_dict['rectify_video_fps']['video_extension']}"
                else:
                    current_working_dir = f"{self.root_directory}{os.sep}video{os.sep}{sub_directory}"
                    target_file = f"000000.{self.input_parameter_dict['rectify_video_fps']['video_extension']}"
                    new_file = f"{sub_directory.split('.')[-1]}-{date_joint}-calibration.{self.input_parameter_dict['rectify_video_fps']['video_extension']}"

                # change video sampling rate
                fps_subp = subprocess.Popen(args=f'''{self.command_addition}ffmpeg -loglevel warning -y -r {esr} -i {target_file} -fps_mode passthrough -crf {crf} -preset {enc_preset} {new_file}''',
                                            stdout=subprocess.PIPE,
                                            cwd=current_working_dir,
                                            shell=self.shell_usage_bool)

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

                pathlib.Path(f"{self.root_directory}{os.sep}video{os.sep}{date_joint}{os.sep}{sub_directory.split('.')[-1]}{os.sep}calibration_images").mkdir(parents=True, exist_ok=True)

                current_working_dir = f"{self.root_directory}{os.sep}video"
                if 'calibration' not in sub_directory:
                    target_file = f"{self.input_parameter_dict['rectify_video_fps']['conversion_target_file']}_{sub_directory.split('.')[-1]}.{self.input_parameter_dict['rectify_video_fps']['video_extension']}"
                    new_file = f"{sub_directory.split('.')[-1]}-{date_joint}.{self.input_parameter_dict['rectify_video_fps']['video_extension']}"

                    shutil.move(src=f"{current_working_dir}{os.sep}{new_file}",
                                dst=f"{self.root_directory}{os.sep}video{os.sep}{date_joint}{os.sep}{sub_directory.split('.')[-1]}{os.sep}{new_file}")
                else:
                    current_working_dir = f"{self.root_directory}{os.sep}video{os.sep}{sub_directory}"
                    target_file = f"000000.{self.input_parameter_dict['rectify_video_fps']['video_extension']}"
                    new_file = f"{sub_directory.split('.')[-1]}-{date_joint}-calibration.{self.input_parameter_dict['rectify_video_fps']['video_extension']}"

                    shutil.move(src=f"{current_working_dir}{os.sep}{new_file}",
                                dst=f"{self.root_directory}{os.sep}video{os.sep}{date_joint}{os.sep}{sub_directory.split('.')[-1]}{os.sep}calibration_images{os.sep}{new_file}")

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
