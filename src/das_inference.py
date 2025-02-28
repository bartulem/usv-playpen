"""
@author: bartulem
Run USV inference on WAV files and create annotations.
"""

from PyQt6.QtTest import QTest
import glob
import json
import librosa
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
import pandas as pd
import shutil
import subprocess
from tqdm import tqdm
from datetime import datetime

plt.style.use(pathlib.Path(__file__).parent / '_config/usv_playpen.mplstyle')


class FindMouseVocalizations:

    def __init__(self, root_directory=None, input_parameter_dict=None,
                 exp_settings_dict=None, message_output=None):
        if input_parameter_dict is None:
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '_parameter_settings/processing_settings.json'), 'r') as json_file:
                self.input_parameter_dict = json.load(json_file)['usv_inference']['FindMouseVocalizations']
        else:
            self.input_parameter_dict = input_parameter_dict['usv_inference']['FindMouseVocalizations']

        if root_directory is None:
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '_parameter_settings/processing_settings.json'), 'r') as json_file:
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
        This method takes WAV files as input and runs DAS inference on them to generate
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
        thresh = self.input_parameter_dict['das_command_line_inference']['segment_confidence_threshold']
        min_len = self.input_parameter_dict['das_command_line_inference']['segment_minlen']
        fill_gap = self.input_parameter_dict['das_command_line_inference']['segment_fillgap']
        save_format = self.input_parameter_dict['das_command_line_inference']['output_file_type']

        if os.name == 'nt':
            command_addition = 'cmd /c '
            shell_usage_bool = False
        else:
            command_addition = 'eval "$(conda shell.bash hook)" && '
            shell_usage_bool = True

        # run inference
        for one_file in sorted(glob.glob(pathname=os.path.join(f"{self.root_directory}{os.sep}audio{os.sep}hpss_filtered", "*.wav*"))):
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
                shutil.move(src=f"{self.root_directory}{os.sep}audio{os.sep}hpss_filtered{os.sep}{one_file}",
                            dst=f"{self.root_directory}{os.sep}audio{os.sep}das_annotations{os.sep}{one_file}")

    def summarize_das_findings(self):
        """
        Description
        ----------
        This function takes WAV files as input and runs DAS inference on them to generate
        tentative USV segments in the recording.
        This method takes CSV files generated by DAS inference and creates a summary file,
        containing information about individual USV segment start and stop times, duration,
        peak amplitude channel, mean amplitude channel, total number of channels it was
        detected on, list of channels it was detected on, and emitter ID.
        ----------

        Parameters
        ----------
        ----------

        Returns
        ----------
        .csv summary file
            CSV file w/ information about all detected USV segments,
            shape: (N_USV, START, STOP, DURATION, PEAK_AMP_CH,
            MEAN_AMP_CH, CHs_COUNT, CHS_DETECTED).
        ----------
        """

        self.message_output(f"DAS summary started at: {datetime.now().hour:02d}:{datetime.now().minute:02d}.{datetime.now().second:02d}.")
        QTest.qWait(1000)

        ch_conversion_dict = {'m_ch01': 0, 'm_ch02': 1, 'm_ch03': 2, 'm_ch04': 3, 'm_ch05': 4, 'm_ch06': 5,
                              'm_ch07': 6, 'm_ch08': 7, 'm_ch09': 8, 'm_ch10': 9, 'm_ch11': 10, 'm_ch12': 11,
                              's_ch01': 12, 's_ch02': 13, 's_ch03': 14, 's_ch04': 15, 's_ch05': 16, 's_ch06': 17,
                              's_ch07': 18, 's_ch08': 19, 's_ch09': 20, 's_ch10': 21, 's_ch11': 22, 's_ch12': 23}

        session_id = self.root_directory.split(os.sep)[-1]

        try:
            das_annotation_files = sorted(glob.glob(f"{self.root_directory}{os.sep}audio{os.sep}das_annotations{os.sep}*.csv"))

            # extract data from all CSV files
            usv_data = {}
            for one_file in das_annotation_files:
                file_id = one_file.split(os.sep)[-1].split('_')[0] + '_' + one_file.split(os.sep)[-1].split('_')[2]
                usv_data[file_id] = pd.read_csv(filepath_or_buffer=one_file, sep=',', index_col=0)

            # filter noise (usually the last row of every DAS file)
            for channel_id, channel_data in usv_data.items():
                usv_data[channel_id] = channel_data.loc[channel_data['name'] != 'noise']

            usv_summary = pd.DataFrame.from_dict(data={'usv_id': [0.],
                                                       'start': [0.],
                                                       'stop': [0.],
                                                       'duration': [0.],
                                                       'peak_amp_ch': [0.],
                                                       'mean_amp_ch': [0.],
                                                       'chs_count': [0.],
                                                       'chs_detected': [[0.]],
                                                       'emitter': [np.nan]}, orient='columns')

            usv_summary_null = usv_summary.copy()

            # extract USV onsets and offsets, and channels they are detected on
            progress_bar = tqdm(list(usv_data.keys()), desc="USV match search on {:s}".format(""), position=0, leave=True)
            usv_num = 0
            for channel_id in progress_bar:
                progress_bar.set_description("USV match search on {:s}".format(channel_id), refresh=True)
                channel_data = usv_data[channel_id]
                if not channel_data.empty:
                    for index, row in channel_data.iterrows():
                        try:
                            next_idx_overlap_with_previous = pd.Interval(left=usv_summary.iloc[-1, 1],
                                                                         right=usv_summary.iloc[-1, 2],
                                                                         closed='neither').overlaps(pd.Interval(left=channel_data.loc[index, 'start_seconds'],
                                                                                                                right=channel_data.loc[index, 'stop_seconds'],
                                                                                                                closed='neither'))
                        except (IndexError, KeyError):
                            next_idx_overlap_with_previous = False

                        if not next_idx_overlap_with_previous:
                            if usv_num > 0:
                                usv_summary = usv_summary._append(usv_summary_null, ignore_index=True)
                            chs_count_temp = 1
                            chs_detected_temp = [ch_conversion_dict[channel_id]]
                            usv_summary.iloc[-1, 1] = row['start_seconds']
                            usv_summary.iloc[-1, 2] = row['stop_seconds']
                            usv_num += 1
                            for channel_id_other, channel_data_other in usv_data.items():
                                if channel_id != channel_id_other and not channel_data_other.empty:
                                    for index_other, row_other in channel_data_other.iterrows():
                                        if pd.Interval(left=row['start_seconds'],
                                                       right=row['stop_seconds'],
                                                       closed='neither').overlaps(pd.Interval(left=row_other['start_seconds'],
                                                                                              right=row_other['stop_seconds'],
                                                                                              closed='neither')):
                                            if row_other['start_seconds'] < usv_summary.iloc[-1, 1]:
                                                usv_summary.iloc[-1, 1] = row_other['start_seconds']
                                            if row_other['stop_seconds'] > usv_summary.iloc[-1, 2]:
                                                usv_summary.iloc[-1, 2] = row_other['stop_seconds']
                                            chs_count_temp += 1
                                            chs_detected_temp.append(ch_conversion_dict[channel_id_other])
                                            # the following segment checks for USV segments that are complete in some channels but broken in pieces in others
                                            try:
                                                if index_other + 1 <= channel_data_other.shape[0] and pd.Interval(left=row['start_seconds'],
                                                                                                                  right=row['stop_seconds'],
                                                                                                                  closed='neither').overlaps(pd.Interval(left=channel_data_other.loc[index_other + 1, 'start_seconds'],
                                                                                                                                                         right=channel_data_other.loc[index_other + 1, 'stop_seconds'],
                                                                                                                                                         closed='neither')):
                                                    channel_data_other.drop(labels=[index_other, index_other + 1], inplace=True)
                                                else:
                                                    channel_data_other.drop(labels=index_other, inplace=True)
                                            except (IndexError, KeyError):
                                                channel_data_other.drop(labels=index_other, inplace=True)

                                            break

                            channel_data.drop(labels=index, inplace=True)

                            usv_summary.iloc[-1, 6] = chs_count_temp
                            usv_summary.iat[-1, 7] = chs_detected_temp

            # compute USV durations and order them by start time
            usv_summary['duration'] = usv_summary['stop'] - usv_summary['start']
            usv_summary.sort_values(by='start', ascending=True, inplace=True)

            # find peak and mean amplitude channels and filter out noise
            mean_signal_correlations = np.zeros(usv_summary.shape[0])
            mean_signal_correlations[:] = np.nan
            audio_file_loc = sorted(glob.glob(f"{self.root_directory}{os.sep}audio{os.sep}hpss_filtered{os.sep}*.mmap"))[0]
            audio_file_name = os.path.basename(audio_file_loc)
            data_type, channel_num, sample_num, audio_sampling_rate = audio_file_name.split("_")[-1][:-5], int(audio_file_name.split("_")[-2]), int(audio_file_name.split("_")[-3]), int(audio_file_name.split("_")[-4])
            audio_file_data = np.memmap(filename=audio_file_loc, mode='r', dtype=data_type, shape=(sample_num, channel_num))

            len_win_signal = 512
            low_freq_cutoff = 30000
            noise_corr_cutoff = 0.3
            frequency_resolution = audio_sampling_rate / len_win_signal
            lower_bin = int(np.floor(low_freq_cutoff / frequency_resolution))

            for index, row in tqdm(usv_summary.iterrows(), desc="USV clean-up progress", total=usv_summary.shape[0], position=0, leave=True):
                start_usv = int(np.floor(row['start'] * audio_sampling_rate))
                stop_usv = int(np.ceil(row['stop'] * audio_sampling_rate))
                peak_amp_ch = np.unravel_index(np.argmax(audio_file_data[start_usv:stop_usv, :]), audio_file_data.shape)[1]
                mean_amp_ch = np.argmax(np.abs(audio_file_data[start_usv:stop_usv, :]).mean(axis=0))
                usv_detected_chs = row['chs_detected']

                # the following section computes channel-wise signal correlations in the frequency domain
                # if the signal has a mean channel-wise correlation of less than 0.3 (or 0.4 for <4 channels!), it is likely noise
                if len(usv_detected_chs) > 1:
                    usv_data_selected_ch = audio_file_data[start_usv:stop_usv, usv_detected_chs].astype('float32').T
                    spectrogram_data_selected_ch = np.abs(librosa.stft(usv_data_selected_ch, n_fft=len_win_signal))
                    reshaped_spectrogram = spectrogram_data_selected_ch[:, lower_bin:, :].reshape(len(usv_detected_chs), -1)
                    correlation_matrix = np.corrcoef(reshaped_spectrogram)
                    unique_correlations = correlation_matrix[np.triu_indices(n=len(usv_detected_chs), k=1)]
                    mean_signal_correlations[index] = np.mean(unique_correlations)
                    if len(usv_detected_chs) > 3:
                        condition_4 = np.mean(unique_correlations) < noise_corr_cutoff
                    else:
                        condition_4 = np.mean(unique_correlations) < (noise_corr_cutoff + 0.1)
                else:
                    condition_4 = False

                # remove USV segments if they appear only on one channel; this gets rid of some true positives, but false positives to a larger extent
                condition_1 = len(usv_detected_chs) < 2
                # remove USV segments if they don't appear on both peak and mean amplitude channels; this is clearly noise
                condition_2 = peak_amp_ch not in usv_detected_chs or mean_amp_ch not in usv_detected_chs
                # if the USV is detected on two channels, but the peak amplitude channel is not the same as the mean amplitude channel, it is likely noise
                condition_3 = len(usv_detected_chs) == 2 and peak_amp_ch != mean_amp_ch

                if condition_1 or condition_2 or condition_3 or condition_4:
                    usv_summary.drop(labels=index, inplace=True)
                else:
                    usv_summary.at[index, 'peak_amp_ch'] = peak_amp_ch
                    usv_summary.at[index, 'mean_amp_ch'] = mean_amp_ch

            fig, ax = plt.subplots(nrows=1, ncols=1)
            ax.hist(x=mean_signal_correlations,
                    bins=20,
                    histtype='stepfilled',
                    color='#BBD5E8',
                    ec='#000000',
                    alpha=.5)
            ax.set_xlabel('Mean signal/spectral correlation')
            ax.set_ylabel('Number of putative USVs')
            ax.axvline(x=noise_corr_cutoff, ls='-.', lw=1.2, c='#000000')
            fig.savefig(f"{self.root_directory}{os.sep}audio{os.sep}{session_id}_usv_signal_correlation_histogram.svg", dpi=300)
            plt.close()

            # give ID number to each USV
            usv_summary['usv_id'] = [f"{_num:04d}" for _num in range(usv_summary.shape[0])]

            # save summary file
            usv_summary.to_csv(path_or_buf=f"{self.root_directory}{os.sep}audio{os.sep}{session_id}_usv_summary.csv",
                               sep=',',
                               index=False)

        except (IndexError, FileNotFoundError):
            self.message_output(f"No DAS annotations found in directory: {self.root_directory}. Skipping summary generation.")
