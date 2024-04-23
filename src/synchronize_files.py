"""
@author: bartulem
Synchronizes files:
(1) the recorded .wav file with tracking file (cuts them to video length).
(2) find audio and video sync trains and check whether they match.
"""

from PyQt6.QtTest import QTest
import configparser
import glob
import json
import operator
import os
import pims
import shutil
import numpy as np
from collections import Counter
from datetime import datetime
from numba import njit
from requests.exceptions import RequestException
from file_loader import DataLoader
from file_writer import DataWriter
from random_pulses import generate_truly_random_seed
from sync_regression import LinRegression


@pims.pipeline
def modify_memmap_array(frame, mmap_arr, frame_idx,
                        led_0, led_1, led_2):
    """
    Description
    ----------
    This function equalizes input colors on luminance.
    ----------

    Parameters
    ----------
    frame : frame object
        The frame to perform extraction on.
    mmap_arr : memmap np.ndarray
        The array to fill data with.
    frame_idx : int
        The corresponding frame index.
    led_0 : list
        XY px coordinates for top LED.
    led_1 : list
        XY px coordinates for middle LED.
    led_2 : list
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
                   'current': {'21241563': {'LED_top': [315, 1250], 'LED_middle': [355, 1255], 'LED_bottom': [400, 1264]},
                               '21372315': {'LED_top': [510, 1268], 'LED_middle': [555, 1268], 'LED_bottom': [603, 1266]}}}

    def __init__(self, root_directory=None, input_parameter_dict=None,
                 message_output=None, exp_settings_dict=None,):
        if input_parameter_dict is None:
            with open('input_parameters.json', 'r') as json_file:
                self.input_parameter_dict = json.load(json_file)['synchronize_files']['Synchronizer']
        else:
            self.input_parameter_dict = input_parameter_dict['synchronize_files']['Synchronizer']
            self.input_parameter_dict_random = input_parameter_dict['random_pulses']['generate_truly_random_seed']

        if root_directory is None:
            with open('input_parameters.json', 'r') as json_file:
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

    def validate_ephys_video_sync(self):
        """
        Description
        ----------
        This method checks whether the time recorded between
        first and last camera signals in the e-phys data stream
        match the total video duration.
        ----------

        Parameter
        ---------
        npx_file_type: str
            AP or LF binary file; defaults to 'ap'.
        ms_divergence_tolerance : int / float
            Max tolerance for divergence between video and e-phys recordings (in ms).

        Returns
        -------
        binary_files_info : .json
            Dictionary w/ information about changepoints, binary file lengths and tracking start/end.
        """

        self.message_output(f"Checking e-phys/video sync started at: {datetime.now().hour:02d}:{datetime.now().minute:02d}.{datetime.now().second:02d}")
        QTest.qWait(1000)

        # read headstage sampling rates
        config = configparser.ConfigParser()
        config.read(f"{self.exp_settings_dict['config_settings_directory']}{os.sep}calibrated_sample_rates_imec.ini")

        # load info from camera_frame_count_dict
        with open(glob.glob(pathname=f'{self.root_directory}{os.sep}**{os.sep}*_camera_frame_count_dict.json', recursive=True)[0], 'r') as frame_count_infile:
            camera_frame_count_dict = json.load(frame_count_infile)
            total_frame_number_least = camera_frame_count_dict['total_frame_number_least']
            total_video_time_least = camera_frame_count_dict['total_video_time_least']

        for npx_idx, npx_recording in enumerate(glob.glob(pathname=f"{self.root_directory}{os.sep}**{os.sep}*{self.input_parameter_dict['validate_ephys_video_sync']['npx_file_type']}.bin", recursive=True)):

            # parse metadata file for channel and headstage information
            with open(f"{npx_recording[:-3]}meta") as meta_data_file:
                for line in meta_data_file:
                    key, value = line.strip().split("=")
                    if key == 'acqApLfSy':
                        total_probe_ch = int(value.split(',')[-1]) + int(value.split(',')[-2])
                    elif key == 'imDatHs_sn':
                        headstage_sn = value
                    elif key == 'imDatPrb_sn':
                        imec_probe_sn = value

            recording_date = self.root_directory.split(os.sep)[-1].split('_')[0]
            recording_file_name = npx_recording.split(os.sep)[-1]
            imec_probe_id = npx_recording.split('.')[-3]

            self.message_output(f"N/V sync for {recording_file_name} with {total_probe_ch} channels, recorded w/ probe #{imec_probe_sn} & headstage #{headstage_sn}.")

            sync_ch_file = npx_recording.replace(recording_file_name, f'{npx_recording.split(os.sep)[-2]}_sync_ch_data')
            if not os.path.isfile(f'{sync_ch_file}.npy'):

                # load the binary file data
                one_recording = np.memmap(filename=npx_recording, mode='r', dtype=np.int16, order='C')
                one_sample_num = one_recording.shape[0] // total_probe_ch

                # reshape the array such that channels are rows and samples are columns
                sync_data = one_recording.reshape((total_probe_ch, one_sample_num), order='F')[-1, :]

                # save sync channel data
                np.save(file=sync_ch_file, arr=sync_data)

            # search for tracking start and end
            ch_sync_data = np.load(file=f'{sync_ch_file}.npy')
            tracking_start, tracking_end = self.find_lsb_changes(relevant_array=ch_sync_data,
                                                                 lsb_bool=False,
                                                                 total_frame_number=total_frame_number_least)

            if (tracking_start, tracking_end) != (None, None):
                spike_glx_sr = float(config['CalibratedHeadStages'][headstage_sn])
                total_npx_recording_duration = (tracking_end - tracking_start) / spike_glx_sr

                duration_difference = round(number=((total_npx_recording_duration - total_video_time_least) * 1000), ndigits=2)
                if duration_difference < 0:
                    comparator_word = 'shorter'
                else:
                    comparator_word = 'longer'

                self.message_output(f"{recording_file_name} is {duration_difference} ms {comparator_word} than the video recording.")

                if abs(duration_difference) < self.input_parameter_dict['validate_ephys_video_sync']['npx_ms_divergence_tolerance']:

                    # save tracking start and end in changepoint information JSON file
                    root_ephys = self.root_directory.replace('Data', 'EPHYS').replace(self.root_directory.split(os.sep)[-1], recording_date) + f'_{imec_probe_id}'
                    if len(glob.glob(pathname=f'{root_ephys}{os.sep}changepoints_info_*.json', recursive=True)) > 0:
                        with open(glob.glob(pathname=f'{root_ephys}{os.sep}changepoints_info_*.json', recursive=True)[0], 'r') as binary_info_input_file:
                            binary_files_info = json.load(binary_info_input_file)
                    else:
                        os.makedirs(root_ephys, exist_ok=True)
                        binary_files_info = {recording_file_name[:-7]: {'session_start_end': [np.nan, np.nan],
                                                                        'tracking_start_end': [np.nan, np.nan],
                                                                        'file_duration_samples': np.nan}}

                    binary_files_info[recording_file_name[:-7]]['tracking_start_end'] = [int(tracking_start), int(tracking_end)]

                    with open(f'{root_ephys}{os.sep}changepoints_info_{recording_date}_imec{imec_probe_id}.json', 'w') as binary_info_output_file:
                        json.dump(binary_files_info, binary_info_output_file, ignore_nan=True, indent=4)

                    self.message_output(f"SUCCESS! Tracking start/end sample times saved in {glob.glob(pathname=f'{root_ephys}{os.sep}changepoints_info_*.json', recursive=True)[0]}.")

                else:
                    count_values_in_sync_data = sorted(dict(Counter(ch_sync_data)).items(), key=operator.itemgetter(1), reverse=True)
                    self.message_output(f'{recording_file_name} has a duration difference (e-phys/tracking) of {duration_difference} ms, so above threshold. '
                                        f'Values in original sync data: {count_values_in_sync_data}. Inspect further before proceeding.')

            else:
                self.message_output(f"Tracking end exceeds e-phys recording boundary, so not found for {recording_file_name}.")
                continue

    @staticmethod
    @njit(parallel=True)
    def find_lsb_changes(relevant_array=None,
                         lsb_bool=True,
                         total_frame_number=0):

        """
        Description
        ----------
        This method takes a  WAV channel sound array or Neuropixels
        sync channel, extracts the LSB part (for WAV files) and
        finds start and end of tracking pulses.
        ----------

        Parameters
        ----------
        Contains the following set of parameters
            relevant_array (np.ndarray)
                Array to extract sync signal from.
            lsb_bool (bool)
                Whether to extract the least significant bit.
            total_frame_number (int)
                Number of frames on the camera containing the minimum total number of frames.
        ----------

        Returns
        ----------
        start_first_relevant_sample, end_last_relevant_sample : tuple
            Start and end of tracking in audio/e-phys samples.
        ----------
        """

        if lsb_bool:
            lsb_array = relevant_array & 1
            ttl_break_end_samples = np.where((lsb_array[1:] - lsb_array[:-1]) > 0)[0]
            largest_break_end_hop = np.argmax(ttl_break_end_samples[1:] - ttl_break_end_samples[:-1]) + 1

        else:
            ttl_break_end_samples = np.where((relevant_array[1:] - relevant_array[:-1]) > 0)[0]
            largest_break_end_hop = np.argmax(ttl_break_end_samples[1:] - ttl_break_end_samples[:-1]) + 1

        if (total_frame_number + largest_break_end_hop) <= ttl_break_end_samples.shape[0]:
            return ttl_break_end_samples[largest_break_end_hop] + 1, ttl_break_end_samples[largest_break_end_hop + total_frame_number] + 1
        else:
            return None, None

    @staticmethod
    @njit(parallel=True)
    def find_ipi_intervals(sound_array=None,
                           audio_sr_rate=250000):

        """
        Description
        ----------
        This method takes a WAV channel sound array, extracts the LSB
        part and finds durations and starts of Arduino sync pulses.
        ----------

        Parameters
        ----------
        Contains the following set of parameters
            sound_array (np.ndarray)
                (Multi-)channel sound array.
            audio_sr_rate (int)
                Sampling rate of audio device; defaults to 250 kHz.
        ----------

        Returns
        ----------
        ipi_durations_ms (np.ndarray)
            Durations of all found IPI intervals (in ms).
        audio_ipi_start_samples (np.ndarray)
            Start samples of all found IPI intervals.
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
    def relative_change_across_array(input_array=None,
                                     desired_axis=0):

        """
        Description
        ----------
        This method takes a 2-D array, and computes the relative
        change, element-wise along the first (X) or second (Y) axis.
        ----------

        Parameters
        ----------
        Contains the following set of parameters
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

    def gather_px_information(self, video_of_interest, sync_camera_fps,
                              camera_id, video_name, total_frame_number):
        """
        ----------
        This method takes find sync LEDs in video frames,
        and gathers information about their intensity changes
        over time.
        ----------

        Parameters
        ----------
        Contains the following set of parameters
            video_of_interest (str)
                Location of relevant sync video.
            sync_camera_fps (int / float)
                Sampling rate of given sync camera.
            camera_id (str)
                ID of sync camera.
            video_name (stR)
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
        mm_arr = np.memmap(f"{self.root_directory}{os.sep}sync{os.sep}sync_px_{video_name[:-4]}",
                           dtype=DataLoader(input_parameter_dict={}).known_dtypes[self.input_parameter_dict['find_video_sync_trains']['mm_dtype']], mode='w+', shape=(total_frame_number, 3, 3))

        for fr_idx in range(total_frame_number):
            processed_frame = modify_memmap_array(loaded_video[fr_idx], mm_arr, fr_idx,
                                                  self.led_px_dict[led_px_version][used_camera]['LED_top'],
                                                  self.led_px_dict[led_px_version][used_camera]['LED_middle'],
                                                  self.led_px_dict[led_px_version][used_camera]['LED_bottom'])

        # the following line is important for saving memmap file changes
        mm_arr.flush()
        mm_arr = 0

    def find_video_sync_trains(self, camera_fps, total_frame_number):
        """
        Description
        ----------
        This method takes video(s) and identifies sync events (from intensity
        changes of sync LEDs) to check sync between different data streams.
        ----------

        Parameters
        ----------
        Contains the following set of parameters
            camera_serial_num (list)
                Serial numbers of sync cameras.
            led_px_version (str)
                Version of LED pxl to be looked at; defaults to current.
            led_px_dev (int)
                Max number of pxl away from some loc to search peak intensity; defaults to 10.
            video_extension (str)
                Video extension; defaults to 'mp4'.
            mm_dtype (str)
                Data type foe memmap file; defaults to 'np.uint8'.
            relative_intensity_threshold (int)
                Relative intensity threshold to categorize important events; defaults to .35.
            millisecond_divergence_tolerance (int / float)
                The amount of variation allowed for sync events converted to ms; defaults to 10 (ms).
            camera_fps (list)
                List of relevant video sampling rates (in fps).
            total_frame_number (int)
                Number of frames on the camera containing the minimum total number of frames.
        ----------

        Returns
        ----------
        led_on_frames (np.ndarray)
            Frames when LED on events start.
        sync_sequence_dict (dict)
            Dictionary for IPI values (in ms) for each camera.
        ----------
        """

        sync_sequence_dict = {}
        ipi_start_frames = 0

        for video_subdir in os.listdir(f"{self.root_directory}{os.sep}video"):
            if '_' not in video_subdir and os.path.isdir(f"{self.root_directory}{os.sep}video{os.sep}{video_subdir}"):
                sync_cam_idx = 0
                for camera_dir in os.listdir(f"{self.root_directory}{os.sep}video{os.sep}{video_subdir}"):
                    video_name = glob.glob(f"{self.root_directory}{os.sep}video{os.sep}{video_subdir}{os.sep}{camera_dir}{os.sep}*.mp4")[0].split(os.sep)[-1]
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
                                               dtype=DataLoader(input_parameter_dict={}).known_dtypes[self.input_parameter_dict['find_video_sync_trains']['mm_dtype']], mode='r', shape=(total_frame_number, 3, 3))

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
                        for threshold_value in np.arange(relative_intensity_threshold-.4, relative_intensity_threshold+.01, .01)[::-1]:
                            if not sequence_found:
                                neg_diff_mask = np.logical_and(np.logical_and(diff_across_leds[:-1] > -threshold_value, diff_across_leds[:-1] < threshold_value), diff_across_leds[1:] < -threshold_value)
                                neg_significant_events = np.where(neg_diff_mask)[0] + 1
                                pos_diff_mask = np.logical_and(np.logical_and(diff_across_leds[:-1] > -threshold_value, diff_across_leds[:-1] < threshold_value), diff_across_leds[1:] > threshold_value)
                                pos_significant_events = np.where(pos_diff_mask)[0]
                                significant_events = np.sort(np.concatenate((neg_significant_events, pos_significant_events)))

                                expected_number_of_pulses = (total_frame_number / camera_fps[sync_cam_idx]) / (0.25 + ((0.25 + 1.5) / 2))
                                if significant_events.shape[0] > expected_number_of_pulses:
                                    # get all significant event durations (i.e., LED on periods and IPIs)
                                    significant_event_durations = np.diff(a=significant_events)

                                    # select only IPI intervals ("-1" is because significant events are computed relative to LED on times,
                                    # so one frame has to be removed)
                                    even_event_durations = significant_event_durations[::2]
                                    odd_event_durations = significant_event_durations[1::2]
                                    if even_event_durations.sum() > odd_event_durations.sum():
                                        ipi_durations_frames = even_event_durations - 1
                                        if type(ipi_start_frames) is int:
                                            temp_ipi_start_frames = np.array(significant_events[::2]) + 1
                                    else:
                                        ipi_durations_frames = odd_event_durations - 1
                                        if type(ipi_start_frames) is int:
                                            temp_ipi_start_frames = np.array(significant_events[1::2]) + 1

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
                                        sync_sequence_dict[camera_dir] = arduino_ipi_durations_subarrays[all_zero_matches]
                                        ipi_start_frames = temp_ipi_start_frames
                                        sequence_found = True
                            else:
                                break
                        else:
                            self.message_output(f"No IPI sequence match found in video '{video_of_interest.split(os.sep)[-1]}'!")

                        sync_cam_idx += 1

        return ipi_start_frames, sync_sequence_dict

    def find_audio_sync_trains(self):
        """
        Description
        ----------
        This method takes audio files and identifies sync events (from the least
        significant bit inputs) to check sync between different data streams.
        ----------

        Parameters
        ----------
        Contains the following set of parameters
            ch_receiving_input (int)
                Audio channel receiving digital input; defaults to 2.
        ----------

        Returns
        ----------
        prediction_error_array (np.ndarray)
            The difference between predicted LED on start video frames and observed LED on start frames.
        ----------
        """

        self.message_output(f"A/V synchronization started at: {datetime.now().hour:02d}:{datetime.now().minute:02d}.{datetime.now().second:02d}")
        QTest.qWait(1000)

        try:
            quantum_seed = generate_truly_random_seed(input_parameter_dict=self.input_parameter_dict_random)
        except RequestException:
            quantum_seed = None

        wave_data_dict = DataLoader(input_parameter_dict={'wave_data_loc': [f"{self.root_directory}{os.sep}audio{os.sep}cropped_to_video"],
                                                          'load_wavefile_data': {'library': 'scipy',
                                                                                 'conditional_arg': [f"_ch{self.input_parameter_dict['find_audio_sync_trains']['ch_receiving_input']:02d}"]}}).load_wavefile_data()

        # get the total number of frames in the video
        json_loc = glob.glob(f"{self.root_directory}{os.sep}video{os.sep}*_camera_frame_count_dict.json")[0]
        with open(json_loc, 'r') as camera_count_json_file:
            camera_fr_count_dict = json.load(camera_count_json_file)
            total_frame_number = camera_fr_count_dict['total_frame_number_least']
            camera_fr = [value[1] for key, value in camera_fr_count_dict.items() if key in self.input_parameter_dict['find_video_sync_trains']['camera_serial_num']]

        # find video sync trains
        video_ipi_start_frames, video_sync_sequence_dict = self.find_video_sync_trains(total_frame_number=total_frame_number,
                                                                                       camera_fps=camera_fr)
        video_sync_sequence_array = np.array(list(video_sync_sequence_dict.values()))

        prediction_error_dict = {}
        audio_devices_start_sample_differences = 0
        for af_idx, audio_file in enumerate(wave_data_dict.keys()):
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
                            prediction_error_array = LinRegression(x_data=audio_ipi_start_samples,
                                                                   y_data=video_ipi_start_frames).split_train_test_and_regress(quantum_seed=quantum_seed)
                            prediction_error_dict[audio_file[:-4]] = prediction_error_array

            else:
                self.message_output("The IPI sequences on different videos do not match.")

        # check if the audio devices match on IPI start samples
        audio_devices_start_sample_differences = np.abs(audio_devices_start_sample_differences)
        self.message_output(f"The smallest IPI start sample difference across master/slave audio devices is {np.nanmin(audio_devices_start_sample_differences)}, "
                            f"the largest is {np.nanmax(audio_devices_start_sample_differences)}, and the mean is {np.nanmean(audio_devices_start_sample_differences)}.")

        return prediction_error_dict

    def crop_wav_files_to_video(self):
        """
        Description
        ----------
        This method takes a (multi-)channel audio recording to find sequences of recorded
        video frames in the LSB of the ch1 recording, and then crops the audio file to
        match the length from the beginning of the first to the end of the last video frame.
        ----------

        Parameters
        ----------
        Contains the following set of parameters
            ch_receiving_input (int)
                Audio channel receiving digital input from Motif; defaults to 1.
        ----------

        Returns
        ----------
        cropped_to_video (.wav file)
            Cropped channel file(s) to match video file.
        ----------
        """

        self.message_output(f"Cropping WAV files started at: {datetime.now().hour:02d}:{datetime.now().minute:02d}.{datetime.now().second:02d}")
        QTest.qWait(1000)

        # load info from camera_frame_count_dict
        with open(glob.glob(f"{self.root_directory}{os.sep}video{os.sep}*_camera_frame_count_dict.json")[0], 'r') as frame_count_infile:
            camera_frame_count_dict = json.load(frame_count_infile)
            total_frame_number = camera_frame_count_dict['total_frame_number_least']
            total_video_time = camera_frame_count_dict['total_video_time_least']

        # audio
        wave_data_dict = DataLoader(input_parameter_dict={'wave_data_loc': [f"{self.root_directory}{os.sep}audio{os.sep}original"],
                                                          'load_wavefile_data': {'library': 'scipy', 'conditional_arg': []}}).load_wavefile_data()

        # determine device ID that gets camera frame trigger pulses
        device_id = self.input_parameter_dict['crop_wav_files_to_video']['device_receiving_input']

        # find camera frame trigger pulses and IPIs in channel file
        start_first_recorded_frame = 0
        end_last_recorded_frame = 0

        for audio_file in wave_data_dict.keys():
            if f"_ch{self.input_parameter_dict['crop_wav_files_to_video']['ch_receiving_input']:02d}" in audio_file and 'm_' in audio_file:

                start_first_recorded_frame, end_last_recorded_frame = self.find_lsb_changes(relevant_array=wave_data_dict[audio_file]['wav_data'],
                                                                                            lsb_bool=True,
                                                                                            total_frame_number=total_frame_number)

                total_audio_recording_during_tracking = (end_last_recorded_frame - start_first_recorded_frame + 1) / wave_data_dict[audio_file]['sampling_rate']
                audio_tracking_difference = total_audio_recording_during_tracking - total_video_time
                self.message_output(f"On device {device_id}, the first tracking frame started at {start_first_recorded_frame} samples, and the last joint one ended at "
                                    f"{end_last_recorded_frame} samples, giving a total audio recording time of {total_audio_recording_during_tracking:.4f} seconds, "
                                    f"which is ~{audio_tracking_difference:.4f} seconds off relative to tracking.")
                break

        QTest.qWait(1000)

        for audio_idx, audio_file in enumerate(wave_data_dict.keys()):

            if wave_data_dict[audio_file]['wav_data'].ndim == 1:
                resized_wav_file = wave_data_dict[audio_file]['wav_data'][start_first_recorded_frame:end_last_recorded_frame + 1]
            else:
                resized_wav_file = wave_data_dict[audio_file]['wav_data'][start_first_recorded_frame:end_last_recorded_frame + 1, :]

            # create new directory for cropped files and HPSS files
            new_directory_cropped_files = f"{self.root_directory}{os.sep}audio{os.sep}cropped_to_video"
            if not os.path.isdir(new_directory_cropped_files):
                os.makedirs(new_directory_cropped_files)
            if not os.path.isdir(f"{self.root_directory}{os.sep}audio{os.sep}hpss"):
                os.makedirs(f"{self.root_directory}{os.sep}audio{os.sep}hpss")

            # write to file
            DataWriter(wav_data=resized_wav_file,
                       input_parameter_dict={'wave_write_loc': new_directory_cropped_files,
                                             'write_wavefile_data': {
                                                 'file_name': f"{audio_file[:-4]}_cropped_to_video",
                                                 'sampling_rate': wave_data_dict[audio_file]['sampling_rate'] / 1e3,
                                                 'library': 'scipy'
                                             }}).write_wavefile_data()

        # delete original directory
        shutil.rmtree(f"{self.root_directory}{os.sep}audio{os.sep}original")

        return total_video_time, total_frame_number
