"""
@author: bartulem
Synchronizes files:
(1) the recorded .wav file with tracking file (cuts them to video length).
(2) find audio and video sync trains and check whether they match.
(3) performs a check on the e-phys data stream to see if the video duration matches the e-phys recording.
"""

from PyQt6.QtTest import QTest
import configparser
import glob
import json
import operator
import os
import pathlib
import pims
import shutil
import subprocess
import numpy as np
from collections import Counter
from datetime import datetime
from numba import njit
from scipy.io import wavfile
from .load_audio_files import DataLoader

@pims.pipeline
def modify_memmap_array(frame: np.ndarray = None,
                        mmap_arr: np.ndarray = None,
                        frame_idx: int = None,
                        led_0: list = None,
                        led_1: list = None,
                        led_2: list = None) -> np.ndarray | None:
    """
    Description
    ----------
    This function .
    ----------

    Parameters
    ----------
    frame (np.ndarray)
        The frame to perform extraction on.
    mmap_arr (memmap np.ndarray)
        The array to fill data with.
    frame_idx (int)
        The corresponding frame index.
    led_0 (list)
        XY px coordinates for top LED.
    led_1 (list)
        XY px coordinates for middle LED.
    led_2 (list)
        XY px coordinates for bottom LED.
    ----------

    Returns
    ----------
    ----------
    """
    mmap_arr[frame_idx, 0, :] = np.array(frame)[led_0[0], led_0[1], :]
    mmap_arr[frame_idx, 1, :] = np.array(frame)[led_1[0], led_1[1], :]
    mmap_arr[frame_idx, 2, :] = np.array(frame)[led_2[0], led_2[1], :]


class Synchronizer:

    """In the dictionary below, you can find px values
    for extracting intensity changes from sync LEDs
    NB: changes in camera positions will change
    these values!
    """
    led_px_dict = {'<2022_08_15': {'21241563': {'LED_top': [276, 1248], 'LED_middle': [348, 1260], 'LED_bottom': [377, 1227]},
                                   '21372315': {'LED_top': [499, 1251], 'LED_middle': [567, 1225], 'LED_bottom': [575, 1249]}},
                   '<2022_12_09': {'21241563': {'LED_top': [276, 1243], 'LED_middle': [348, 1258], 'LED_bottom': [377, 1225]},
                                   '21372315': {'LED_top': [518, 1262], 'LED_middle': [587, 1237], 'LED_bottom': [593, 1260]},
                                   '21372316': {'LED_top': [1000, 603], 'LED_middle': [1003, 598], 'LED_bottom': [1004, 691]}},
                   '<2023_01_19': {'21241563': {'LED_top': [275, 1266], 'LED_middle': [345, 1272], 'LED_bottom': [375, 1245]},
                                   '21372315': {'LED_top': [520, 1260], 'LED_middle': [590, 1230], 'LED_bottom': [595, 1260]},
                                   '21372316': {'LED_top': [1000, 605], 'LED_middle': [1004, 601], 'LED_bottom': [1005, 694]}},
                   '<2023_08_01': {'21241563': {'LED_top': [275, 1260], 'LED_middle': [345, 1270], 'LED_bottom': [380, 1233]},
                                   '21372315': {'LED_top': [520, 1255], 'LED_middle': [590, 1230], 'LED_bottom': [595, 1257]}},
                   '<2024_01_01': {'21372315': {'LED_top': [514, 1255], 'LED_middle': [575, 1235], 'LED_bottom': [590, 1261]}},
                   '<2024_09_20': {'21241563': {'LED_top': [315, 1250], 'LED_middle': [355, 1255], 'LED_bottom': [400, 1264]},
                                   '21372315': {'LED_top': [510, 1268], 'LED_middle': [555, 1268], 'LED_bottom': [603, 1266]}},
                   'current': {'21241563': {'LED_top': [317, 1247], 'LED_middle': [360, 1254], 'LED_bottom': [403, 1262]},
                               '21372315': {'LED_top': [507, 1267], 'LED_middle': [554, 1267], 'LED_bottom': [601, 1266]}}}

    if os.name == 'nt':
        command_addition = 'cmd /c '
        shell_usage_bool = False
    else:
        command_addition = ''
        shell_usage_bool = True

    def __init__(self, root_directory: str = None,
                 input_parameter_dict: dict = None,
                 message_output: callable = None,
                 exp_settings_dict: dict = None) -> None:
        """
        Initializes the Synchronizer class.

        Parameter
        ---------
        root_directory (str)
            Root directory for data; defaults to None.
        input_parameter_dict (dict)
            Processing parameters; defaults to None.
        exp_settings_dict (dict)
            Experimental settings; defaults to None.
        message_output (function)
            Defines output messages; defaults to None.

        Returns
        -------
        -------
        """

        if input_parameter_dict is None:
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '_parameter_settings/processing_settings.json'), 'r') as json_file:
                self.input_parameter_dict = json.load(json_file)['synchronize_files']['Synchronizer']
        else:
            self.input_parameter_dict = input_parameter_dict['synchronize_files']['Synchronizer']

        if root_directory is None:
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '_parameter_settings/processing_settings.json'), 'r') as json_file:
                self.root_directory = json.load(json_file)['synchronize_files']['root_directory']
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

    def validate_ephys_video_sync(self) -> None:
        """
        Description
        ----------
        This method checks whether the time recorded between
        first and last camera signals in the e-phys data stream
        match the total video duration.
        ----------

        Parameters
        ----------
        ----------

        Returns
        -------
        binary_files_info (.json file)
            Dictionary w/ information about changepoints, binary file lengths and tracking start/end.
        """

        self.message_output(f"Checking e-phys/video sync started at: {datetime.now().hour:02d}:{datetime.now().minute:02d}.{datetime.now().second:02d}")
        QTest.qWait(1000)

        # read headstage sampling rates
        calibrated_sr_config = configparser.ConfigParser()
        calibrated_sr_config.read(os.path.join(os.path.dirname(os.path.abspath(__file__)), '_config/calibrated_sample_rates_imec.ini'))

        # load info from camera_frame_count_dict
        with open(sorted(glob.glob(pathname=f'{self.root_directory}{os.sep}**{os.sep}*_camera_frame_count_dict.json', recursive=True))[0], 'r') as frame_count_infile:
            camera_frame_count_dict = json.load(frame_count_infile)
            total_frame_number_least = camera_frame_count_dict['total_frame_number_least']
            total_video_time_least = camera_frame_count_dict['total_video_time_least']

        for npx_idx, npx_recording in enumerate(sorted(glob.glob(pathname=f"{self.root_directory}{os.sep}**{os.sep}*{self.input_parameter_dict['validate_ephys_video_sync']['npx_file_type']}.bin", recursive=True))):

            # parse metadata file for channel and headstage information
            with open(f"{npx_recording[:-3]}meta") as meta_data_file:
                for line in meta_data_file:
                    key, value = line.strip().split('=')
                    if key == 'acqApLfSy':
                        total_probe_ch = int(value.split(',')[0]) + int(value.split(',')[-1])
                    elif key == 'imDatHs_sn':
                        headstage_sn = value
                    elif key == 'imDatPrb_sn':
                        imec_probe_sn = value

            recording_date = self.root_directory.split(os.sep)[-1].split('_')[0]
            recording_file_name = npx_recording.split(os.sep)[-1]
            imec_probe_id = npx_recording.split('.')[-3]

            self.message_output(f"N/V sync for {recording_file_name} with {total_probe_ch} channels, recorded w/ probe #{imec_probe_sn} & headstage #{headstage_sn}.")

            sync_ch_file = f"{os.path.dirname(npx_recording)}{os.sep}{os.path.basename(npx_recording)[:-7]}_sync_ch_data".replace('.', '_')
            if not os.path.isfile(f'{sync_ch_file}.npy'):

                # load the binary file data
                one_recording = np.memmap(filename=npx_recording, mode='r', dtype='int16', order='C')
                one_sample_num = one_recording.shape[0] // total_probe_ch

                # reshape the array such that channels are rows and samples are columns
                sync_data = one_recording.reshape((total_probe_ch, one_sample_num), order='F')[-1, :]

                # save sync channel data
                np.save(file=sync_ch_file, arr=sync_data)

            # search for tracking start and end
            ch_sync_data = np.load(file=f'{sync_ch_file}.npy')
            tracking_start, tracking_end, largest_break_duration = self.find_lsb_changes(relevant_array=ch_sync_data,
                                                                                         lsb_bool=False,
                                                                                         total_frame_number=total_frame_number_least)

            largest_break_duration_sec = round(largest_break_duration / float(calibrated_sr_config['CalibratedHeadStages'][headstage_sn]), 3)
            if (tracking_start, tracking_end) != (None, None) or largest_break_duration_sec < 2:
                spike_glx_sr = float(calibrated_sr_config['CalibratedHeadStages'][headstage_sn])
                total_npx_recording_duration = (tracking_end - tracking_start) / spike_glx_sr

                duration_difference = round(number=((total_npx_recording_duration - total_video_time_least) * 1000), ndigits=2)
                if duration_difference < 0:
                    comparator_word = 'shorter'
                else:
                    comparator_word = 'longer'

                self.message_output(f"{recording_file_name} is {abs(duration_difference)} ms {comparator_word} than the video recording with {largest_break_duration_sec} s largest camera break duration.")

                if abs(duration_difference) < self.input_parameter_dict['validate_ephys_video_sync']['npx_ms_divergence_tolerance']:

                    # save tracking start and end in changepoint information JSON file
                    root_ephys = self.root_directory.replace('Data', 'EPHYS').replace(self.root_directory.split(os.sep)[-1], recording_date) + f'_{imec_probe_id}'
                    pathlib.Path(root_ephys).mkdir(parents=True, exist_ok=True)
                    if len(sorted(glob.glob(pathname=f'{root_ephys}{os.sep}changepoints_info_*.json', recursive=True))) > 0:
                        with open(sorted(glob.glob(pathname=f'{root_ephys}{os.sep}changepoints_info_*.json', recursive=True))[0], 'r') as binary_info_input_file:
                            binary_files_info = json.load(binary_info_input_file)

                        binary_files_info[recording_file_name[:-7]] = {'session_start_end': [np.nan, np.nan],
                                                                       'tracking_start_end': [np.nan, np.nan],
                                                                       'largest_camera_break_duration': np.nan,
                                                                       'file_duration_samples': np.nan,
                                                                       'root_directory': self.root_directory,
                                                                       'total_num_channels': total_probe_ch,
                                                                       'headstage_sn': headstage_sn,
                                                                       'imec_probe_sn': imec_probe_sn}
                    else:
                        binary_files_info = {recording_file_name[:-7]: {'session_start_end': [np.nan, np.nan],
                                                                        'tracking_start_end': [np.nan, np.nan],
                                                                        'largest_camera_break_duration': np.nan,
                                                                        'file_duration_samples': np.nan,
                                                                        'root_directory': self.root_directory,
                                                                        'total_num_channels': total_probe_ch,
                                                                        'headstage_sn': headstage_sn,
                                                                        'imec_probe_sn': imec_probe_sn}}

                    binary_files_info[recording_file_name[:-7]]['tracking_start_end'] = [int(tracking_start), int(tracking_end)]
                    binary_files_info[recording_file_name[:-7]]['largest_camera_break_duration'] = int(largest_break_duration)

                    with open(f'{root_ephys}{os.sep}changepoints_info_{recording_date}_{imec_probe_id}.json', 'w') as binary_info_output_file:
                        json.dump(binary_files_info, binary_info_output_file, indent=4)

                    self.message_output(f"SUCCESS! Tracking start/end sample times saved in {sorted(glob.glob(pathname=f'{root_ephys}{os.sep}changepoints_info_*.json', recursive=True))[0]}.")

                else:
                    count_values_in_sync_data = sorted(dict(Counter(ch_sync_data)).items(), key=operator.itemgetter(1), reverse=True)
                    self.message_output(f'{recording_file_name} has a duration difference (e-phys/tracking) of {duration_difference} ms, so above threshold. '
                                        f'Values in original sync data: {count_values_in_sync_data}. Inspect further before proceeding.')

            else:
                self.message_output(f"Tracking end exceeds e-phys recording boundary, so not found for {recording_file_name}.")
                continue

    @staticmethod
    # @njit(parallel=True)
    def find_lsb_changes(relevant_array: np.ndarray = None,
                         lsb_bool: bool = True,
                         total_frame_number: int = 0) -> tuple:

        """
        Description
        ----------
        This method takes a  WAV channel sound array or Neuropixels
        sync channel, extracts the LSB part (for WAV files) and
        finds start and end of tracking pulses.
        ----------

        Parameters
        ----------
        relevant_array (np.ndarray)
            Array to extract sync signal from.
        lsb_bool (bool)
            Whether to extract the least significant bit.
        total_frame_number (int)
            Number of frames on the camera containing the minimum total number of frames.
        ----------

        Returns
        ----------
        start_first_relevant_sample, end_last_relevant_sample, largest_break_duration (tuple)
            Start and end of tracking in audio/e-phys samples, and the duration of largest break.
        ----------
        """

        if lsb_bool:
            lsb_array = relevant_array & 1
            ttl_break_end_samples = np.where((lsb_array[1:] - lsb_array[:-1]) > 0)[0]
            largest_break_end_hop = np.argmax(ttl_break_end_samples[1:] - ttl_break_end_samples[:-1]) + 1
        else:
            ttl_break_end_samples = np.where((relevant_array[1:] - relevant_array[:-1]) > 0)[0]
            largest_break_end_hop = np.argmax(ttl_break_end_samples[1:] - ttl_break_end_samples[:-1]) + 1

        largest_break_duration = np.max(ttl_break_end_samples[1:] - ttl_break_end_samples[:-1])

        if (total_frame_number + largest_break_end_hop) <= ttl_break_end_samples.shape[0]:
            return int(ttl_break_end_samples[largest_break_end_hop] + 1), int(ttl_break_end_samples[largest_break_end_hop + total_frame_number] + 1), int(largest_break_duration)
        else:
            return None, None, int(largest_break_duration)

    @staticmethod
    @njit(parallel=True)
    def find_ipi_intervals(sound_array: np.ndarray = None,
                           audio_sr_rate: int = 250000) -> tuple:

        """
        Description
        ----------
        This method takes a WAV channel sound array, extracts the LSB
        part and finds durations and starts of Arduino sync pulses.
        ----------

        Parameters
        ----------
        sound_array (np.ndarray)
            Sound data array.
        audio_sr_rate (int)
            Sampling rate of audio device; defaults to 250 kHz.
        ----------

        Returns
        ----------
        ipi_durations_ms (np.ndarray), audio_ipi_start_samples (np.ndarray)
            Durations of all found IPI intervals (in ms) and
            start samples of all found IPI intervals.
        ----------
        """

        # get the least significant bit array
        lsb_array = sound_array & 1

        # get switches from ON to OFF and vice versa (both look at the 0 value positions)
        ipi_start_samples = np.where(np.diff(lsb_array) < 0)[0] + 1
        ipi_end_samples = np.where(np.diff(lsb_array) > 0)[0]

        # find IPI starts and durations in milliseconds
        if ipi_start_samples[0] < ipi_end_samples[0]:
            if ipi_start_samples.shape[0] == ipi_end_samples.shape[0]:
                ipi_durations_ms = (((ipi_end_samples - ipi_start_samples) + 1) * 1000 / audio_sr_rate)
                audio_ipi_start_samples = ipi_start_samples
            else:
                ipi_durations_ms = (((ipi_end_samples - ipi_start_samples[:-1]) + 1) * 1000 / audio_sr_rate)
                audio_ipi_start_samples = ipi_start_samples[:-1]
        else:
            if ipi_start_samples.shape[0] == ipi_end_samples.shape[0]:
                ipi_durations_ms = (((ipi_end_samples[1:] - ipi_start_samples[:-1]) + 1) * 1000 / audio_sr_rate)
                audio_ipi_start_samples = ipi_start_samples[:-1]
            else:
                ipi_durations_ms = (((ipi_end_samples[1:] - ipi_start_samples) + 1) * 1000 / audio_sr_rate)
                audio_ipi_start_samples = ipi_start_samples

        return ipi_durations_ms, audio_ipi_start_samples

    @staticmethod
    @njit(parallel=True)
    def relative_change_across_array(input_array: np.ndarray = None,
                                     desired_axis: int = 0) -> np.ndarray:

        """
        Description
        ----------
        This method takes a 2-D array, and computes the relative
        change, element-wise along the first (X) or second (Y) axis.
        ----------

        Parameters
        ----------
        input_array (np.ndarray)
            Input array.
        desired_axis (int)
            Axis to compute the change over.
        ----------

        Returns
        ----------
        percent_change_array (np.ndarray)
            Proportional change relative to previous element along the desired axis.
        ----------
        """

        if desired_axis == 0:
            relative_change_array = 1 - (input_array[1:, :] / input_array[:-1, :])
        else:
            relative_change_array = 1 - (input_array[:, 1:] / input_array[:, :-1])

        return relative_change_array

    def gather_px_information(self, video_of_interest: str = None,
                              sync_camera_fps: int | float = None,
                              camera_id: str = None,
                              video_name: str = None,
                              total_frame_number: int = None) -> None:
        """
        ----------
        This method takes find sync LEDs in video frames,
        and gathers information about their intensity changes
        over time.
        ----------

        Parameters
        ----------
        video_of_interest (str)
            Location of relevant sync video.
        sync_camera_fps (int / float)
            Sampling rate of given sync camera.
        camera_id (str)
            ID of sync camera.
        video_name (str)
            Full name of sync video.
        total_frame_number (int)
            Total least number of frames of all cameras.
        ----------

        Returns
        ----------
        mm_arr (memmap file)
            Memory map file containing pixel intensities of sync LEDs.
        ----------
        """

        # load video
        loaded_video = pims.Video(video_of_interest)

        # find camera sync pxl
        max_frame_num = int(round(sync_camera_fps + (sync_camera_fps / 2)))
        led_px_version = self.input_parameter_dict['find_video_sync_trains']["led_px_version"]
        led_px_dev = self.input_parameter_dict['find_video_sync_trains']["led_px_dev"]
        used_camera = camera_id

        peak_intensity_frame_loc = 'zero'
        for led_idx, led_position in enumerate(self.led_px_dict[led_px_version][used_camera].keys()):
            led_dim1, led_dim2 = self.led_px_dict[led_px_version][used_camera][led_position]
            dim1_floor_pxl = max(0, led_dim1 - led_px_dev)
            dim1_ceil_pxl = min(loaded_video[0].shape[0], led_dim1 + led_px_dev)
            dim2_floor_pxl = max(0, led_dim2 - led_px_dev)
            dim2_ceil_pxl = min(loaded_video[0].shape[1], led_dim2 + led_px_dev)

            if led_idx == 0:
                fame_pxl_values = np.zeros(max_frame_num)
                for one_num in range(max_frame_num):
                    fame_pxl_values[one_num] = loaded_video[one_num][dim1_floor_pxl:dim1_ceil_pxl, dim2_floor_pxl:dim2_ceil_pxl].max()
                peak_intensity_frame_loc = fame_pxl_values.argmax()

            frame_of_choice = loaded_video[peak_intensity_frame_loc][dim1_floor_pxl:dim1_ceil_pxl, dim2_floor_pxl:dim2_ceil_pxl]
            screen_results = np.where(frame_of_choice == frame_of_choice.max())
            result_dim1, result_dim2 = [dim1_floor_pxl + screen_results[0][0], dim2_floor_pxl + screen_results[1][0]]
            self.led_px_dict[led_px_version][used_camera][led_position] = [result_dim1, result_dim2]
            self.message_output(f"For camera {used_camera}, {led_position} the highest intensity pixel is in position {result_dim1},{result_dim2}.")

        # create memmap array to store the data for posterity
        mm_arr = np.memmap(filename=f"{self.root_directory}{os.sep}sync{os.sep}sync_px_{video_name[:-4]}",
                           dtype=DataLoader(input_parameter_dict={}).known_dtypes['np.uint8'], mode='w+', shape=(total_frame_number, 3, 3))

        for fr_idx in range(total_frame_number):
            processed_frame = modify_memmap_array(loaded_video[fr_idx], mm_arr, fr_idx,
                                                  self.led_px_dict[led_px_version][used_camera]['LED_top'],
                                                  self.led_px_dict[led_px_version][used_camera]['LED_middle'],
                                                  self.led_px_dict[led_px_version][used_camera]['LED_bottom'])

        # the following line is important for saving memmap file changes
        mm_arr.flush()
        mm_arr = 0

    def find_video_sync_trains(self, camera_fps: list = None,
                               total_frame_number: int = None) -> tuple:
        """
        Description
        ----------
        This method takes video(s) and identifies sync events (from intensity
        changes of sync LEDs) to check sync between different data streams.
        ----------

        Parameters
        ----------
        camera_fps (list)
            List of relevant video sampling rates (in fps).
        total_frame_number (int)
            Number of frames on the camera containing the minimum total number of frames.
        ----------

        Returns
        ----------
        led_on_frames (np.ndarray), sync_sequence_dict (dict)
            Frames when LED on events start and
            dictionary for IPI values (in ms) for each camera.
        ----------
        """

        sync_sequence_dict = {}
        ipi_start_frames = 0

        for video_subdir in os.listdir(f"{self.root_directory}{os.sep}video"):
            if '_' not in video_subdir and os.path.isdir(f"{self.root_directory}{os.sep}video{os.sep}{video_subdir}"):
                sync_cam_idx = 0
                for camera_dir in os.listdir(f"{self.root_directory}{os.sep}video{os.sep}{video_subdir}"):
                    if camera_dir != '.DS_Store' and os.path.isdir(f"{self.root_directory}{os.sep}video{os.sep}{video_subdir}{os.sep}{camera_dir}") and camera_dir in self.input_parameter_dict['find_video_sync_trains']['camera_serial_num']:
                        video_name = sorted(glob.glob(f"{self.root_directory}{os.sep}video{os.sep}{video_subdir}{os.sep}{camera_dir}{os.sep}*.mp4"))[0].split(os.sep)[-1]
                        if 'calibration' not in video_name \
                                and video_name.split('-')[0] in self.input_parameter_dict['find_video_sync_trains']['camera_serial_num'] \
                                and self.input_parameter_dict['find_video_sync_trains']['video_extension'] in video_name:

                            current_working_dir = f"{self.root_directory}{os.sep}video{os.sep}{video_subdir}{os.sep}{camera_dir}"
                            video_of_interest = f"{current_working_dir}{os.sep}{video_name}"

                            if not os.path.exists(f"{self.root_directory}{os.sep}sync{os.sep}sync_px_{video_name[:-4]}"):
                                self.gather_px_information(video_of_interest=video_of_interest,
                                                           sync_camera_fps=camera_fps[sync_cam_idx],
                                                           camera_id=camera_dir,
                                                           video_name=video_name,
                                                           total_frame_number=total_frame_number)

                            # load memmap data
                            leds_array = np.memmap(f"{self.root_directory}{os.sep}sync{os.sep}sync_px_{video_name[:-4]}",
                                                   dtype=DataLoader(input_parameter_dict={}).known_dtypes['np.uint8'], mode='r', shape=(total_frame_number, 3, 3))

                            # take mean across all three (RGB) channels
                            mean_across_rgb = leds_array.mean(axis=-1)

                            # compute relative change across frames and take median across LEDs
                            diff_across_leds = np.median(self.relative_change_across_array(input_array=mean_across_rgb), axis=1)

                            # get actual IPI from CoolTerm-recorded .txt file
                            arduino_ipi_durations = []
                            for txt_file in os.listdir(f"{self.root_directory}{os.sep}sync"):
                                if 'CoolTerm' in txt_file:
                                    with open(f"{self.root_directory}{os.sep}sync{os.sep}{txt_file}", 'r') as ipi_txt_file:
                                        for line_num, line in enumerate(ipi_txt_file.readlines()):
                                            if line_num > 2 and line.strip():
                                                arduino_ipi_durations.append(int(line.strip()))
                                    break
                            arduino_ipi_durations = np.array(arduino_ipi_durations)

                            # find indices where the largest changes occur
                            relative_intensity_threshold = self.input_parameter_dict['find_video_sync_trains']['relative_intensity_threshold']
                            sequence_found = False
                            for threshold_value in np.arange(0.2, relative_intensity_threshold, .01)[::-1]:
                                if not sequence_found:

                                    diff_mask_neg = np.logical_and(np.logical_and(diff_across_leds[:-1] > -threshold_value, diff_across_leds[:-1] < threshold_value), diff_across_leds[1:] < -threshold_value)
                                    neg_significant_events = np.where(diff_mask_neg)[0] + 1
                                    neg_significant_events = np.delete(neg_significant_events, np.argwhere(np.ediff1d(neg_significant_events) <= int(np.ceil(camera_fps[sync_cam_idx] / 2.5))) + 1)

                                    diff_mask_pos = np.logical_and(np.logical_and(diff_across_leds[:-1] > -threshold_value, diff_across_leds[:-1] < threshold_value), diff_across_leds[1:] > threshold_value)
                                    pos_significant_events = np.where(diff_mask_pos)[0]
                                    pos_significant_events = np.delete(pos_significant_events, np.argwhere(np.ediff1d(pos_significant_events) <= int(np.ceil(camera_fps[sync_cam_idx]/2.5))) + 1)

                                    if 0 <= (pos_significant_events.size - neg_significant_events.size) < 2 or (0 <= np.abs(pos_significant_events.size - neg_significant_events.size) < 2 and threshold_value < 0.35):
                                        if neg_significant_events.size > pos_significant_events.size:
                                            neg_significant_events = neg_significant_events[1:]

                                        if pos_significant_events[0] < neg_significant_events[0]:
                                            if pos_significant_events.size == neg_significant_events.size:
                                                ipi_durations_frames = (neg_significant_events - pos_significant_events) - 1
                                                temp_ipi_start_frames = pos_significant_events + 1
                                            else:
                                                ipi_durations_frames = (neg_significant_events - pos_significant_events[:-1]) - 1
                                                temp_ipi_start_frames = pos_significant_events[:-1] + 1
                                        else:
                                            if pos_significant_events.size == neg_significant_events.size:
                                                ipi_durations_frames = (neg_significant_events[1:] - pos_significant_events[:-1]) - 1
                                                temp_ipi_start_frames = pos_significant_events[:-1] + 1
                                            else:
                                                ipi_durations_frames = (neg_significant_events[1:] - pos_significant_events) - 1
                                                temp_ipi_start_frames = pos_significant_events + 1

                                        # compute IPI durations in milliseconds
                                        ipi_durations_ms = np.round(ipi_durations_frames * (1000 / camera_fps[sync_cam_idx]))

                                        # match IPI sequences
                                        subarray_size = ipi_durations_ms.shape[0]
                                        start_indices = np.arange(len(arduino_ipi_durations) - subarray_size + 1)
                                        index_matrix = start_indices[:, np.newaxis] + np.arange(subarray_size)
                                        arduino_ipi_durations_subarrays = arduino_ipi_durations[index_matrix]

                                        result_array = arduino_ipi_durations_subarrays - ipi_durations_ms
                                        all_zero_matches = np.all(result_array <= self.input_parameter_dict['find_video_sync_trains']['millisecond_divergence_tolerance'],
                                                                  axis=1)
                                        any_all_zeros = np.any(all_zero_matches)
                                        if any_all_zeros:
                                            sync_sequence_dict[camera_dir] = np.ravel(arduino_ipi_durations_subarrays[all_zero_matches])
                                            ipi_start_frames = temp_ipi_start_frames
                                            sequence_found = True
                                else:
                                    break
                            else:
                                self.message_output(f"No IPI sequence match found in video '{video_of_interest.split(os.sep)[-1]}'!")

                            sync_cam_idx += 1

        return ipi_start_frames, sync_sequence_dict

    def find_audio_sync_trains(self) -> dict:
        """
        Description
        ----------
        This method takes audio files and identifies sync events (from the least
        significant bit inputs) to check sync between different data streams.
        ----------

        Parameters
        ----------
        ----------

        Returns
        ----------
        prediction_error_array (np.ndarray)
            The difference between predicted LED on start video frames and observed LED on start frames.
        ----------
        """

        self.message_output(f"A/V synchronization started at: {datetime.now().hour:02d}:{datetime.now().minute:02d}.{datetime.now().second:02d}")
        QTest.qWait(1000)

        wave_data_dict = DataLoader(input_parameter_dict={'wave_data_loc': [f"{self.root_directory}{os.sep}audio{os.sep}cropped_to_video"],
                                                          'load_wavefile_data': {'library': 'scipy',
                                                                                 'conditional_arg': [f"_ch{self.input_parameter_dict['find_audio_sync_trains']['ch_receiving_input']:02d}"]}}).load_wavefile_data()

        # get the total number of frames in the video
        json_loc = sorted(glob.glob(f"{self.root_directory}{os.sep}video{os.sep}*_camera_frame_count_dict.json"))[0]
        with open(json_loc, 'r') as camera_count_json_file:
            camera_fr_count_dict = json.load(camera_count_json_file)
            total_frame_number = camera_fr_count_dict['total_frame_number_least']
            camera_fr = [value[1] for key, value in camera_fr_count_dict.items() if key in self.input_parameter_dict['find_video_sync_trains']['camera_serial_num']]

        # find video sync trains
        video_ipi_start_frames, video_sync_sequence_dict = self.find_video_sync_trains(total_frame_number=total_frame_number,
                                                                                       camera_fps=camera_fr)
        video_sync_sequence_array = np.array(list(video_sync_sequence_dict.values()))

        ipi_discrepancy_dict = {}
        audio_devices_start_sample_differences = 0
        for af_idx, audio_file in enumerate(sorted(wave_data_dict.keys())):
            self.message_output(f"Working on sync data in audio file: {audio_file[:-4]}")
            QTest.qWait(1000)

            ipi_durations_ms, audio_ipi_start_samples = self.find_ipi_intervals(sound_array=wave_data_dict[audio_file]['wav_data'],
                                                                                audio_sr_rate=wave_data_dict[audio_file]['sampling_rate'])

            if af_idx == 0:
                audio_devices_start_sample_differences = audio_ipi_start_samples
            else:
                audio_devices_start_sample_differences = audio_devices_start_sample_differences - audio_ipi_start_samples

            if (video_sync_sequence_array == video_sync_sequence_array[0]).all():
                for video_idx, video_key in enumerate(video_sync_sequence_dict.keys()):
                    if video_idx == 0:
                        diff_array = np.absolute(np.round(ipi_durations_ms) - video_sync_sequence_dict[video_key])
                        bool_condition_array = diff_array <= self.input_parameter_dict['find_video_sync_trains']['millisecond_divergence_tolerance']
                        if not np.all(bool_condition_array):
                            self.message_output(f"IPI sequence match NOT found in audio file! There is/are {(~bool_condition_array).sum()} difference(s) larger "
                                                f"than the tolerance and the largest one is {diff_array.max()} ms")
                        else:
                            audio_video_ipi_discrepancy_ms = ((audio_ipi_start_samples / wave_data_dict[audio_file]['sampling_rate']) - (video_ipi_start_frames / camera_fr[0])) * 1000

                            # if the SYNC is acceptable, delete the original audio files
                            if np.max(np.abs(audio_video_ipi_discrepancy_ms)) < 18:
                                if os.path.exists(f"{self.root_directory}{os.sep}audio{os.sep}original"):
                                    shutil.rmtree(f"{self.root_directory}{os.sep}audio{os.sep}original")

                            ipi_discrepancy_dict[audio_file[:-4]] = audio_video_ipi_discrepancy_ms

            else:
                self.message_output("The IPI sequences on different videos do not match.")

        # check if the audio devices match on IPI start samples
        audio_devices_start_sample_differences = np.abs(audio_devices_start_sample_differences)
        self.message_output(f"The smallest IPI start sample difference across master/slave audio devices is {np.nanmin(audio_devices_start_sample_differences)}, "
                            f"the largest is {np.nanmax(audio_devices_start_sample_differences)}, and the mean is {np.nanmean(audio_devices_start_sample_differences)}.")

        return ipi_discrepancy_dict

    def crop_wav_files_to_video(self) -> None:
        """
        Description
        ----------
        This method takes a WAV file audio recording to find sequences of recorded
        video frames in the LSB of the triggerbox input channel, and then crops the audio file to
        match the length from the beginning of the first to the end of the last video frame.

        NB: If there are two audio recording devices and if they are not synchronized, both
        sets of audio files are cut to the length of the shorter one. This entails resampling
        longer audio files to match the shorter duration (on one device) using SoX, and the
        LSB of those files is resampled and then maintained in the final audio file.
        ----------

        Parameters
        ----------
        ----------

        Returns
        ----------
        cropped_to_video (.wav file)
            Cropped channel file(s) to match video file.
        ----------
        """

        self.message_output(f"Cropping WAV files to video started at: {datetime.now().hour:02d}:{datetime.now().minute:02d}.{datetime.now().second:02d}")
        QTest.qWait(1000)

        # load info from camera_frame_count_dict
        with open(sorted(glob.glob(f"{self.root_directory}{os.sep}video{os.sep}*_camera_frame_count_dict.json"))[0], 'r') as frame_count_infile:
            camera_frame_count_dict = json.load(frame_count_infile)
            total_frame_number = camera_frame_count_dict['total_frame_number_least']
            total_video_time = camera_frame_count_dict['total_video_time_least']

        # load audio channels receiving camera triggerbox input
        wave_data_dict = DataLoader(input_parameter_dict={'wave_data_loc': [f"{self.root_directory}{os.sep}audio{os.sep}original"],
                                                          'load_wavefile_data': {'library': 'scipy',
                                                                                 'conditional_arg': [f"_ch{self.input_parameter_dict['crop_wav_files_to_video']['ch_receiving_input']:02d}"]}}).load_wavefile_data()

        # determine device ID(s) that get(s) camera frame trigger pulses
        if self.input_parameter_dict['crop_wav_files_to_video']['device_receiving_input'] == 'both':
            device_ids = ['m', 's']
        else:
            device_ids = [self.input_parameter_dict['crop_wav_files_to_video']['device_receiving_input']]

        # find camera frame trigger pulses and IPIs in channel file
        start_end_video = {device: {'start_first_recorded_frame': 0, 'end_last_recorded_frame': 0, 'largest_break_duration': 0,
                                    'duration_samples': 0, 'duration_seconds': 0, 'audio_tracking_diff_seconds': 0} for device in device_ids}

        for device in device_ids:
            for audio_file in wave_data_dict.keys():
                if f'{device}_' in audio_file:

                    (start_end_video[device]['start_first_recorded_frame'],
                     start_end_video[device]['end_last_recorded_frame'],
                     start_end_video[device]['largest_break_duration']) = self.find_lsb_changes(relevant_array=wave_data_dict[audio_file]['wav_data'], lsb_bool=True, total_frame_number=total_frame_number)

                    start_end_video[device]['duration_samples'] = int(start_end_video[device]['end_last_recorded_frame'] - start_end_video[device]['start_first_recorded_frame'] + 1)
                    start_end_video[device]['duration_seconds'] = round(start_end_video[device]['duration_samples'] / wave_data_dict[audio_file]['sampling_rate'], 4)
                    start_end_video[device]['audio_tracking_diff_seconds'] = round(start_end_video[device]['duration_seconds'] - total_video_time, 4)

                    self.message_output(f"On {device} device, the largest break duration lasted {start_end_video[device]['largest_break_duration'] / wave_data_dict[audio_file]['sampling_rate']:.3f} seconds, "
                                        f"so the first tracking frame started at {start_end_video[device]['start_first_recorded_frame']} samples, and the last joint one ended at "
                                        f"{start_end_video[device]['end_last_recorded_frame']} samples, giving a total audio recording time of {start_end_video[device]['duration_seconds']} seconds, "
                                        f"which is {start_end_video[device]['audio_tracking_diff_seconds']} seconds off relative to tracking.")

                    break

        # create new directory for cropped files and HPSS files
        with open(f"{self.root_directory}{os.sep}audio{os.sep}audio_triggerbox_sync_info.json", 'w') as audio_dict_outfile:
            json.dump(start_end_video, audio_dict_outfile, indent=4)
        new_directory_cropped_files = f"{self.root_directory}{os.sep}audio{os.sep}cropped_to_video"
        pathlib.Path(new_directory_cropped_files).mkdir(parents=True, exist_ok=True)

        # find all audio files
        all_audio_files = sorted(glob.glob(f"{self.root_directory}{os.sep}audio{os.sep}original{os.sep}*.wav"))

        m_longer = False
        s_longer = False
        if len(device_ids) > 1:
           if start_end_video['m']['duration_samples'] > start_end_video['s']['duration_samples']:
               m_longer = True
               m_original_arr_indices = np.arange(0, start_end_video['m']['duration_samples'])
               m_new_arr_indices = np.linspace(start=0, stop=start_end_video['m']['duration_samples'] - 1, num=start_end_video['s']['duration_samples'])
           if start_end_video['m']['duration_samples'] < start_end_video['s']['duration_samples']:
               s_longer = True
               s_original_arr_indices = np.arange(0, start_end_video['s']['duration_samples'])
               s_new_arr_indices = np.linspace(start=0, stop=start_end_video['s']['duration_samples'] - 1, num=start_end_video['m']['duration_samples'])

        QTest.qWait(1000)

        cut_audio_subprocesses = []
        for audio_file in all_audio_files:
            outfile_loc = f"{self.root_directory}{os.sep}audio{os.sep}cropped_to_video{os.sep}{os.path.basename(audio_file)[:-4]}_cropped_to_video.wav"

            if len(device_ids) == 1:
                start_cut_sample = start_end_video[device_ids[0]]['start_first_recorded_frame']
                cut_duration_samples = start_end_video[device_ids[0]]['duration_samples']
                cut_audio_subp = subprocess.Popen(args=f'''{self.command_addition}sox {os.path.basename(audio_file)} {outfile_loc} trim {start_cut_sample}s {cut_duration_samples}s''',
                                                  cwd=f"{self.root_directory}{os.sep}audio{os.sep}original",
                                                  shell=self.shell_usage_bool)
                cut_audio_subprocesses.append(cut_audio_subp)
            else:
                if 'm_' in audio_file:
                    m_start_cut_sample = start_end_video['m']['start_first_recorded_frame']
                    m_cut_duration_samples = start_end_video['m']['duration_samples']
                    if m_longer:
                        # extract original LSB data
                        m_sr_original, m_data_original = wavfile.read(f'{audio_file}')
                        m_lsb_original = m_data_original[start_end_video['m']['start_first_recorded_frame']:start_end_video['m']['end_last_recorded_frame']+1] & 1

                        # resample the LSB data
                        m_lsb_modified = np.where(np.interp(x=m_new_arr_indices, xp=m_original_arr_indices, fp=m_lsb_original).astype(np.int16) > 0.5, 1, 0).astype(np.int16)

                        # trim and adjust tempo
                        tempo_adjustment_factor = start_end_video['m']['duration_samples'] / start_end_video['s']['duration_samples']
                        subprocess.Popen(args=f'''{self.command_addition}sox {os.path.basename(audio_file)} {outfile_loc} trim {m_start_cut_sample}s {m_cut_duration_samples}s tempo -s {tempo_adjustment_factor}''',
                                         cwd=f"{self.root_directory}{os.sep}audio{os.sep}original",
                                         shell=self.shell_usage_bool).wait()

                        # load data again and overwrite the LSB
                        m_sr_tempo_adjusted, m_data_tempo_adjusted = wavfile.read(f'{outfile_loc}')
                        if m_data_tempo_adjusted.size == start_end_video['s']['duration_samples']:
                            m_data_modified = (m_data_tempo_adjusted & ~1) ^ m_lsb_modified
                        else:
                            m_data_modified = (m_data_tempo_adjusted[:start_end_video['s']['duration_samples']] & ~1) ^ m_lsb_modified
                        wavfile.write(filename=outfile_loc, rate=m_sr_original, data=m_data_modified)

                    else:
                        cut_audio_subp = subprocess.Popen(args=f'''{self.command_addition}sox {os.path.basename(audio_file)} {outfile_loc} trim {m_start_cut_sample}s {m_cut_duration_samples}s''',
                                                          cwd=f"{self.root_directory}{os.sep}audio{os.sep}original",
                                                          shell=self.shell_usage_bool)
                        cut_audio_subprocesses.append(cut_audio_subp)
                else:
                    s_start_cut_sample = start_end_video['s']['start_first_recorded_frame']
                    s_cut_duration_samples = start_end_video['s']['duration_samples']
                    if s_longer:
                        # extract original LSB data
                        s_sr_original, s_data_original = wavfile.read(f'{audio_file}')
                        s_lsb_original = s_data_original[start_end_video['s']['start_first_recorded_frame']:start_end_video['s']['end_last_recorded_frame'] + 1] & 1

                        # resample the LSB data
                        s_lsb_modified = np.where(np.interp(x=s_new_arr_indices, xp=s_original_arr_indices, fp=s_lsb_original).astype(np.int16) > 0.5, 1, 0).astype(np.int16)

                        # trim and adjust tempo
                        tempo_adjustment_factor = start_end_video['s']['duration_samples'] / start_end_video['m']['duration_samples']
                        subprocess.Popen(args=f'''{self.command_addition}sox {os.path.basename(audio_file)} {outfile_loc} trim {s_start_cut_sample}s {s_cut_duration_samples}s tempo -s {tempo_adjustment_factor}''',
                                         cwd=f"{self.root_directory}{os.sep}audio{os.sep}original",
                                         shell=self.shell_usage_bool).wait()

                        # load data again and overwrite the LSB
                        s_sr_tempo_adjusted, s_data_tempo_adjusted = wavfile.read(f'{outfile_loc}')
                        if s_data_tempo_adjusted.size == start_end_video['m']['duration_samples']:
                            s_data_modified = (s_data_tempo_adjusted & ~1) ^ s_lsb_modified
                        else:
                            s_data_modified = (s_data_tempo_adjusted[:start_end_video['m']['duration_samples']] & ~1) ^ s_lsb_modified

                        wavfile.write(filename=outfile_loc, rate=s_sr_original, data=s_data_modified)

                    else:
                        cut_audio_subp = subprocess.Popen(args=f'''{self.command_addition}sox {os.path.basename(audio_file)} {outfile_loc} trim {s_start_cut_sample}s {s_cut_duration_samples}s''',
                                                          cwd=f"{self.root_directory}{os.sep}audio{os.sep}original",
                                                          shell=self.shell_usage_bool)
                        cut_audio_subprocesses.append(cut_audio_subp)

        while True:
            status_poll = [query_subp.poll() for query_subp in cut_audio_subprocesses]
            if any(elem is None for elem in status_poll):
                QTest.qWait(5000)
            else:
                break

        # create HPSS directory
        pathlib.Path(f"{self.root_directory}{os.sep}audio{os.sep}hpss").mkdir(parents=True, exist_ok=True)
