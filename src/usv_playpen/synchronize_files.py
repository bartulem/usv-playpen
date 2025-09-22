"""
@author: bartulem
Synchronizes files:
(1) the recorded .wav file with a tracking file (cuts them to video length).
(2) find audio and video sync trains and check whether they match.
(3) performs a check on the e-phys data stream to see if the video duration matches the e-phys recording.
"""

import configparser
import cv2
import glob
import json
import operator
import os
import pathlib
import shutil
import subprocess
import numpy as np
from collections import Counter
from datetime import datetime
from imgstore import new_for_filename
from numba import njit
from scipy.io import wavfile
from .load_audio_files import DataLoader
from .time_utils import *


def find_events(diffs: np.ndarray,
                threshold: float,
                min_separation: int) -> tuple:
    """
    Description
    ----------
    This function finds initial event candidates (rising/falling edges) in a
    signal and debounces them by removing duplicate-like detections.
    ----------

    Parameters
    ----------
    diffs (np.ndarray)
        The 1D input signal representing frame-to-frame changes.
    threshold (float)
        The value above which a change is considered significant.
    min_separation (int)
        Minimum frames between two events of the same type to be kept.
    ----------

    Returns
    ----------
    pos_events (np.ndarray), neg_events (np.ndarray)
        A tuple of two arrays containing the frame indices for debounced
        positive (ON) and negative (OFF) events, respectively.
    ----------
    """

    stable = np.abs(diffs[:-1]) < threshold
    rising = diffs[1:] > threshold
    falling = diffs[1:] < -threshold

    pos_events = np.where(stable & rising)[0]
    neg_events = np.where(stable & falling)[0] + 1

    if len(pos_events) > 1:
        keep_mask = np.concatenate(([True], np.diff(pos_events) > min_separation))
        pos_events = pos_events[keep_mask]

    if len(neg_events) > 1:
        keep_mask = np.concatenate(([True], np.diff(neg_events) > min_separation))
        neg_events = neg_events[keep_mask]

    return pos_events, neg_events

def _combine_and_sort_events(pos_events: np.ndarray,
                             neg_events: np.ndarray) -> np.ndarray:
    """
    Description
    ----------
    Internal helper function to combine separate ON and OFF event arrays
    into a single, sorted (N, 2) array of [frame, type].
    ----------

    Parameters
    ----------
    pos_events (np.ndarray)
        Array of frame indices for positive (ON) events.
    neg_events (np.ndarray)
        Array of frame indices for negative (OFF) events.
    ----------

    Returns
    ----------
    (np.ndarray)
        A single (N, 2) array of all events sorted by frame number,
        with type 1 for ON and -1 for OFF.
    ----------
    """

    pos_array = np.stack(arrays=(pos_events, np.ones_like(pos_events)), axis=1)
    neg_array = np.stack(arrays=(neg_events, -np.ones_like(neg_events)), axis=1)
    all_events = np.vstack((pos_array, neg_array))
    return all_events[all_events[:, 0].argsort()]

def filter_events_by_duration(pos_events: np.ndarray,
                              neg_events: np.ndarray,
                              min_duration: int) -> tuple:
    """
    Description
    ----------
    This function filters event pairs that define a state (e.g., an 'ON'
    state) that is shorter than a minimum duration, removing glitches.
    ----------

    Parameters
    ----------
    pos_events (np.ndarray)
        Array of frame indices for candidate positive (ON) events.
    neg_events (np.ndarray)
        Array of frame indices for candidate negative (OFF) events.
    min_duration (int)
        The minimum number of frames a state must last to be considered valid.
    ----------

    Returns
    ----------
    final_pos (np.ndarray), final_neg (np.ndarray)
        A tuple of two arrays containing the filtered frame indices for
        valid positive (ON) and negative (OFF) events.
    ----------
    """

    if len(pos_events) == 0 and len(neg_events) == 0:
        return np.array([]), np.array([])

    all_events = _combine_and_sort_events(pos_events, neg_events)

    durations = np.diff(all_events[:, 0])
    is_short = durations < min_duration
    is_flip = all_events[:-1, 1] == -all_events[1:, 1]

    glitch_starts = np.where(is_short & is_flip)[0]
    indices_to_remove = np.union1d(glitch_starts, glitch_starts + 1)
    valid_events = np.delete(all_events, indices_to_remove, axis=0)

    final_pos = valid_events[valid_events[:, 1] == 1, 0].astype(int)
    final_neg = valid_events[valid_events[:, 1] == -1, 0].astype(int)
    return final_pos, final_neg

def validate_sequence(pos_events: np.ndarray,
                      neg_events: np.ndarray) -> tuple:
    """
    Description
    ----------
    Ensures the final event sequence is logical by enforcing that event
    types strictly alternate (e.g., ON, OFF, ON...).
    ----------

    Parameters
    ----------
    pos_events (np.ndarray)
        Array of frame indices for filtered positive (ON) events.
    neg_events (np.ndarray)
        Array of frame indices for filtered negative (OFF) events.
    ----------

    Returns
    ----------
    final_pos (np.ndarray), final_neg (np.ndarray)
        A tuple of two arrays containing the final, validated frame indices.
    ----------
    """

    if len(pos_events) == 0 and len(neg_events) == 0:
        return np.array([]), np.array([])

    all_events = _combine_and_sort_events(pos_events, neg_events)

    if len(all_events) < 2:
        return pos_events, neg_events

    # find indices where an event is the same type as the one following it
    non_alternating_indices = np.where(all_events[:-1, 1] == all_events[1:, 1])[0]

    if len(non_alternating_indices) > 0:
        # keep the first event of a non-alternating pair, remove the second
        indices_to_remove = non_alternating_indices + 1
        valid_events = np.delete(all_events, indices_to_remove, axis=0)
    else:
        valid_events = all_events

    final_pos = valid_events[valid_events[:, 1] == 1, 0].astype(int)
    final_neg = valid_events[valid_events[:, 1] == -1, 0].astype(int)

    return final_pos, final_neg


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
                   '<2025_05_08': {'21241563': {'LED_top': [317, 1247], 'LED_middle': [360, 1254], 'LED_bottom': [403, 1262]},
                                   '21372315': {'LED_top': [507, 1267], 'LED_middle': [554, 1267], 'LED_bottom': [601, 1266]}},
                   '<2025_09_21': {'21241563': {'LED_top': [310, 1245], 'LED_middle': [358, 1248], 'LED_bottom': [402, 1255]},
                                   '21372315': {'LED_top': [504, 1261], 'LED_middle': [551, 1260], 'LED_bottom': [598, 1260]}},
                   'current': {'21241563': {'LED_top': [296, 1234], 'LED_middle': [339, 1244], 'LED_bottom': [383, 1252]},
                               '21372315': {'LED_top': [504, 1267], 'LED_middle': [551, 1268], 'LED_bottom': [599, 1265]}}}

    if os.name == 'nt':
        command_addition = 'cmd /c '
        shell_usage_bool = False
    else:
        command_addition = ''
        shell_usage_bool = True

    def __init__(self, root_directory: str = None,
                 input_parameter_dict: dict = None,
                 message_output: callable = None) -> None:
        """
        Initializes the Synchronizer class.

        Parameter
        ---------
        root_directory (str)
            Root directory for data; defaults to None.
        input_parameter_dict (dict)
            Processing parameters; defaults to None.
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

        if message_output is None:
            self.message_output = print
        else:
            self.message_output = message_output

        self.app_context_bool = is_gui_context()

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
        smart_wait(app_context_bool=self.app_context_bool, seconds=1)

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
            (tracking_start, tracking_end, largest_break_duration,
             ttl_break_end_samples, largest_break_end_hop) = self.find_lsb_changes(relevant_array=ch_sync_data, lsb_bool=False, total_frame_number=total_frame_number_least)

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
    def find_lsb_changes(relevant_array: np.ndarray = None,
                         lsb_bool: bool = True,
                         total_frame_number: int = 0) -> tuple:

        """
        Description
        ----------
        This method takes a WAV channel sound array or Neuropixels
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
        start_first_relevant_sample, end_last_relevant_sample,
        largest_break_duration, ttl_break_end_samples, largest_break_end_hop (tuple)
            Start and end of tracking in audio/e-phys samples, the duration of largest break,
            all TTL break end samples, and sample position of the largest break.
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
            return int(ttl_break_end_samples[largest_break_end_hop] + 1), int(ttl_break_end_samples[largest_break_end_hop + total_frame_number] + 1), int(largest_break_duration), ttl_break_end_samples, largest_break_end_hop
        else:
            return None, None, int(largest_break_duration), ttl_break_end_samples, largest_break_end_hop

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
            if ipi_start_samples.size == ipi_end_samples.size:
                ipi_durations_ms = (((ipi_end_samples - ipi_start_samples) + 1) * 1000 / audio_sr_rate)
                audio_ipi_start_samples = ipi_start_samples
            else:
                ipi_durations_ms = (((ipi_end_samples - ipi_start_samples[:-1]) + 1) * 1000 / audio_sr_rate)
                audio_ipi_start_samples = ipi_start_samples[:-1]
        else:
            if ipi_start_samples.size == ipi_end_samples.size:
                ipi_durations_ms = (((ipi_end_samples[1:] - ipi_start_samples[:-1]) + 1) * 1000 / audio_sr_rate)
                audio_ipi_start_samples = ipi_start_samples[:-1]
            else:
                ipi_durations_ms = (((ipi_end_samples[1:] - ipi_start_samples) + 1) * 1000 / audio_sr_rate)
                audio_ipi_start_samples = ipi_start_samples

        return ipi_durations_ms, audio_ipi_start_samples

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

        cap = cv2.VideoCapture(video_of_interest)

        # get video dimensions
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        max_frame_num = int(round(sync_camera_fps + (sync_camera_fps / 2)))
        led_px_version = self.input_parameter_dict['find_video_sync_trains']["led_px_version"]
        led_px_dev = self.input_parameter_dict['find_video_sync_trains']["led_px_dev"]
        used_camera = camera_id

        for led_position in self.led_px_dict[led_px_version][used_camera].keys():
            led_dim1, led_dim2 = self.led_px_dict[led_px_version][used_camera][led_position]

            # define a search area (Region of Interest - ROI) around the approximate coordinate
            y_start = max(0, led_dim1 - led_px_dev)
            y_end = min(frame_height, led_dim1 + led_px_dev)
            x_start = max(0, led_dim2 - led_px_dev)
            x_end = min(frame_width, led_dim2 + led_px_dev)

            peak_intensity = -1
            peak_intensity_frame_loc = -1

            # find the brightest frame for THIS SPECIFIC LED
            for frame_num in range(max_frame_num):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                if not ret: continue

                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                roi_intensity = np.max(frame_gray[y_start:y_end, x_start:x_end])

                if roi_intensity > peak_intensity:
                    peak_intensity = roi_intensity
                    peak_intensity_frame_loc = frame_num

            # go to that brightest frame and find the centroid of the LED spot
            cap.set(cv2.CAP_PROP_POS_FRAMES, peak_intensity_frame_loc)
            ret, frame = cap.read()
            if ret:
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                roi = frame_gray[y_start:y_end, x_start:x_end]

                # use Otsu's method to automatically find the best threshold
                # to separate the bright LED from the darker background within the ROI.
                _, binary_roi = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                # calculate image moments of the resulting binary mask
                M = cv2.moments(binary_roi)

                # compute centroid
                if M['m00'] != 0:
                    # find the center (cx, cy) *relative to the top-left corner of the small ROI box*
                    cx_relative = int(M['m10'] / M['m00'])
                    cy_relative = int(M['m01'] / M['m00'])

                    # crucially, add the box's offset (y_start, x_start) to convert back to full-frame coordinates
                    final_y = y_start + cy_relative
                    final_x = x_start + cx_relative

                    self.led_px_dict[led_px_version][used_camera][led_position] = [final_y, final_x]
                    self.message_output(f"For {led_position}, centroid found at frame {peak_intensity_frame_loc}: ({final_y}, {final_x})")
                else:
                    self.message_output(f"Could not find centroid for {led_position}, using original coordinate.")

        mm_arr = np.memmap(filename=f"{self.root_directory}{os.sep}sync{os.sep}sync_px_{video_name[:-4]}",
                           dtype=np.uint8, mode='w+', shape=(total_frame_number, 3, 3))

        led_coords = np.array([
            self.led_px_dict[led_px_version][used_camera]['LED_top'],
            self.led_px_dict[led_px_version][used_camera]['LED_middle'],
            self.led_px_dict[led_px_version][used_camera]['LED_bottom']
        ])

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        for fr_idx in range(total_frame_number):
            ret, frame = cap.read()

            if not ret:
                self.message_output(f"WARNING: Reached end of decodable frames at index {fr_idx}, while total_frame_number was {total_frame_number}.")
                break

            if frame.ndim == 3:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pixel_values = frame_rgb[led_coords[:, 0], led_coords[:, 1]]
                mm_arr[fr_idx] = pixel_values
            else:
                pixel_values = frame[led_coords[:, 0], led_coords[:, 1]]
                mm_arr[fr_idx] = np.repeat(pixel_values[:, np.newaxis], repeats=3, axis=1)

        cap.release()
        mm_arr.flush()

    def attempt_sequence_match(self, brightness_signal: np.ndarray,
                                camera_fps: float,
                                arduino_ipi_durations: np.ndarray,
                                camera_dir: str) -> tuple:
        """
        Description
        ----------
        This helper function takes a 1D brightness signal and attempts to find a
        match for the ground-truth Arduino IPI sequence. It contains the full
        pipeline of event detection, filtering, and sequence comparison.
        ----------

        Parameters
        ----------
        brightness_signal (np.ndarray)
            The 1D brightness signal generated from either median or max of LEDs.
        camera_fps (float)
            The sampling rate of the camera in frames per second.
        arduino_ipi_durations (np.ndarray)
            The ground-truth sequence of IPIs from the CoolTerm log.
        camera_dir (str)
            The identifier for the camera being processed.
        ----------

        Returns
        ----------
        (dict, np.ndarray, bool)
            A tuple containing the sync_sequence_dict, the ipi_start_frames,
            and a boolean indicating if the sequence was found. Returns
            (None, None, False) on failure.
        ----------
        """

        # compute the relative change of the provided signal
        diff_across_leds = 1 - (brightness_signal[1:] / brightness_signal[:-1])

        # find indices where the largest changes occur by iterating through thresholds
        relative_intensity_threshold = self.input_parameter_dict['find_video_sync_trains']['relative_intensity_threshold']

        for threshold_value in np.arange(0.2, relative_intensity_threshold, .01)[::-1]:
            # step 1: find raw candidate events
            pos_significant_events, neg_significant_events = find_events(
                diffs=diff_across_leds,
                threshold=threshold_value,
                min_separation=int(np.ceil(camera_fps / 2.5))
            )

            # step 2: filter out short-lived glitches
            pos_significant_events, neg_significant_events = filter_events_by_duration(
                pos_significant_events, neg_significant_events, min_duration=35
            )

            # step 3: ensure the sequence of events is logical (alternating)
            pos_significant_events, neg_significant_events = validate_sequence(
                pos_significant_events, neg_significant_events
            )

            if pos_significant_events.size > 0 and neg_significant_events.size > 0:
                # check for a reasonable number of ON/OFF events before proceeding
                if 0 <= (pos_significant_events.size - neg_significant_events.size) < 2 or \
                        (0 <= np.abs(pos_significant_events.size - neg_significant_events.size) < 2 and threshold_value < 0.35):

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
                            if pos_significant_events.size > neg_significant_events.size:
                                ipi_durations_frames = (neg_significant_events[1:] - pos_significant_events[:neg_significant_events.size - 1]) - 1
                                temp_ipi_start_frames = pos_significant_events[:neg_significant_events.size - 1] + 1
                            else:
                                ipi_durations_frames = (neg_significant_events[1:] - pos_significant_events) - 1
                                temp_ipi_start_frames = pos_significant_events + 1

                    ipi_durations_ms = np.round(ipi_durations_frames * (1000 / camera_fps))

                    # match IPI sequences
                    if ipi_durations_ms.shape[0] > 0 and ipi_durations_ms.shape[0] <= len(arduino_ipi_durations):
                        subarray_size = ipi_durations_ms.shape[0]
                        start_indices = np.arange(len(arduino_ipi_durations) - subarray_size + 1)
                        index_matrix = start_indices[:, np.newaxis] + np.arange(subarray_size)
                        arduino_ipi_durations_subarrays = arduino_ipi_durations[index_matrix]

                        result_array = arduino_ipi_durations_subarrays - ipi_durations_ms
                        tolerance = self.input_parameter_dict['find_video_sync_trains']['millisecond_divergence_tolerance']
                        all_zero_matches = np.all(np.abs(result_array) <= tolerance, axis=1)

                        if np.any(all_zero_matches):
                            sync_sequence_dict = {camera_dir: np.ravel(arduino_ipi_durations_subarrays[all_zero_matches])}
                            ipi_start_frames = temp_ipi_start_frames
                            return sync_sequence_dict, ipi_start_frames, True

        return None, None, False

    def find_video_sync_trains(self, camera_fps: list = None,
                               total_frame_number: int = None) -> tuple:

        """
        Description
        ----------
        This method takes video(s) and identifies sync events (from intensity
        changes of sync LEDs) to check sync between different data streams. It uses
        a robust temporal validation method based on known pulse durations.
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
        (np.ndarray, dict)
            A tuple containing an array of the OFF-event start frames (as per user
            definition) and a dictionary of the matched IPI sequences for each camera.
        ----------
        """

        sync_sequence_dict = {}
        ipi_start_frames = np.array([])

        for video_subdir in os.listdir(f"{self.root_directory}{os.sep}video"):
            if '_' in video_subdir or not os.path.isdir(f"{self.root_directory}{os.sep}video{os.sep}{video_subdir}"): continue

            sync_cam_idx = 0
            for camera_dir in os.listdir(f"{self.root_directory}{os.sep}video{os.sep}{video_subdir}"):
                if (camera_dir == '.DS_Store' or not os.path.isdir(f"{self.root_directory}{os.sep}video{os.sep}{video_subdir}{os.sep}{camera_dir}")
                        or camera_dir not in self.input_parameter_dict['find_video_sync_trains']['sync_camera_serial_num']): continue

                video_name_glob = glob.glob(f"{self.root_directory}{os.sep}video{os.sep}{video_subdir}{os.sep}{camera_dir}{os.sep}*.mp4")
                if not video_name_glob: continue
                video_name = sorted(video_name_glob)[0].split(os.sep)[-1]

                if ('calibration' in video_name or video_name.split('-')[0] not in self.input_parameter_dict['find_video_sync_trains']['sync_camera_serial_num']
                        or self.input_parameter_dict['find_video_sync_trains']['sync_video_extension'] not in video_name): continue

                video_of_interest = f"{self.root_directory}{os.sep}video{os.sep}{video_subdir}{os.sep}{camera_dir}{os.sep}{video_name}"

                if not os.path.exists(f"{self.root_directory}{os.sep}sync{os.sep}sync_px_{video_name[:-4]}"):
                    self.gather_px_information(
                        video_of_interest=video_of_interest,
                        sync_camera_fps=camera_fps[sync_cam_idx],
                        camera_id=camera_dir,
                        video_name=video_name,
                        total_frame_number=total_frame_number
                    )

                arduino_ipi_durations = []
                for txt_file in os.listdir(f"{self.root_directory}{os.sep}sync"):
                    if 'CoolTerm' in txt_file:
                        with open(f"{self.root_directory}{os.sep}sync{os.sep}{txt_file}", 'r') as ipi_txt_file:
                            for line_num, line in enumerate(ipi_txt_file.readlines()):
                                if line_num > 2 and line.strip():
                                    arduino_ipi_durations.append(int(line.strip()))
                        break
                arduino_ipi_durations = np.array(arduino_ipi_durations)

                leds_array = np.memmap(filename=f"{self.root_directory}{os.sep}sync{os.sep}sync_px_{video_name[:-4]}",
                                       dtype=np.uint8, mode='r', shape=(total_frame_number, 3, 3))

                mean_across_rgb = leds_array.mean(axis=-1)

                # --- ATTEMPT 1: Use MEDIAN (robust to bright noise) ---
                self.message_output(f"Attempting sync detection for {camera_dir} with MEDIAN signal...")
                brightness_signal_median = np.median(mean_across_rgb, axis=1) + 1e-6

                temp_sync_dict, temp_ipi_frames, sequence_found = self.attempt_sequence_match(
                    brightness_signal=brightness_signal_median,
                    camera_fps=camera_fps[sync_cam_idx],
                    arduino_ipi_durations=arduino_ipi_durations,
                    camera_dir=camera_dir
                )

                # --- ATTEMPT 2: Fallback to MAX if MEDIAN fails (robust to occlusions) ---
                if not sequence_found:
                    self.message_output(f"Median method failed for {camera_dir}. Falling back to MAX signal...")
                    brightness_signal_max = np.max(mean_across_rgb, axis=1) + 1e-6

                    temp_sync_dict, temp_ipi_frames, sequence_found = self.attempt_sequence_match(
                        brightness_signal=brightness_signal_max,
                        camera_fps=camera_fps[sync_cam_idx],
                        arduino_ipi_durations=arduino_ipi_durations,
                        camera_dir=camera_dir
                    )

                if sequence_found:
                    self.message_output(f"SUCCESS: Sync sequence found for {camera_dir}!")
                    sync_sequence_dict.update(temp_sync_dict)
                    ipi_start_frames = temp_ipi_frames
                else:
                    self.message_output(f"No sequence match found in '{video_name}'!")

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
        ipi_discrepancy_dict (dict)
            Contains IPI discrepancies between audio and video sync trains and IPI video start frames.
        ----------
        """

        self.message_output(f"A/V synchronization started at: {datetime.now().hour:02d}:{datetime.now().minute:02d}.{datetime.now().second:02d}")
        smart_wait(app_context_bool=self.app_context_bool, seconds=1)

        wave_data_dict = DataLoader(input_parameter_dict={'wave_data_loc': [f"{self.root_directory}{os.sep}audio{os.sep}cropped_to_video"],
                                                          'load_wavefile_data': {'library': 'scipy',
                                                                                 'conditional_arg': [f"_ch{self.input_parameter_dict['find_audio_sync_trains']['sync_ch_receiving_input']:02d}"]}}).load_wavefile_data()

        # get the total number of frames in the video
        json_loc = sorted(glob.glob(f"{self.root_directory}{os.sep}video{os.sep}*_camera_frame_count_dict.json"))[0]
        with open(json_loc, 'r') as camera_count_json_file:
            camera_fr_count_dict = json.load(camera_count_json_file)
            total_frame_number = camera_fr_count_dict['total_frame_number_least']
            total_video_time_least = camera_fr_count_dict['total_video_time_least']
            camera_fr = [value[1] for key, value in camera_fr_count_dict.items() if key in self.input_parameter_dict['find_video_sync_trains']['sync_camera_serial_num']]

        # find video sync trains
        video_ipi_start_frames, video_sync_sequence_dict = self.find_video_sync_trains(total_frame_number=total_frame_number,
                                                                                       camera_fps=camera_fr)
        video_sync_sequence_array = np.array(list(video_sync_sequence_dict.values()))

        # find NIDQ sync trains
        nidq_file = next(pathlib.Path(self.root_directory).glob(f"**{os.sep}*.nidq.bin"), None)
        nidq_ipi_data_file = f"{self.root_directory}{os.sep}sync{os.sep}nidq_ipi_data.npy"
        if nidq_file is not None and not os.path.isfile(nidq_ipi_data_file):
            nidq_recording = np.memmap(filename=nidq_file, mode='r', dtype=np.int16, order='C')
            nidq_sample_num = nidq_recording.shape[0] // self.input_parameter_dict['find_audio_sync_trains']['nidq_num_channels']
            nidq_digital_ch = nidq_recording.reshape((self.input_parameter_dict['find_audio_sync_trains']['nidq_num_channels'], nidq_sample_num), order='F')[-1, :].reshape([-1, 1])
            nidq_digital_bits = (nidq_digital_ch & (2 ** np.arange(16).reshape([1, 16]))).astype(bool).astype(int)

            # find start/end of recording
            if self.input_parameter_dict['find_audio_sync_trains']['nidq_bool']:
                triggerbox_bit_changes = np.where((nidq_digital_bits[1:, self.input_parameter_dict['find_audio_sync_trains']['nidq_triggerbox_input_bit_position']] - nidq_digital_bits[:-1, self.input_parameter_dict['find_audio_sync_trains']['nidq_triggerbox_input_bit_position']]) > 0)[0]
                triggerbox_diffs = triggerbox_bit_changes[1:] - triggerbox_bit_changes[:-1]
                largest_break_end_hop = np.argmax(triggerbox_diffs) + 1
                largest_break_end_hop_sec = round((triggerbox_bit_changes[largest_break_end_hop] - triggerbox_bit_changes[largest_break_end_hop - 1]) / self.input_parameter_dict['find_audio_sync_trains']['nidq_sr'], 3)
                self.message_output(f"For NIDQ, the largest break in video frame recording is {largest_break_end_hop_sec} seconds.")

                loopbio_start_nidq_sample = int(triggerbox_bit_changes[largest_break_end_hop] + 1)
                loopbio_end_nidq_sample = int(triggerbox_bit_changes[largest_break_end_hop + total_frame_number] + 1)
                nidq_rec_duration = (loopbio_end_nidq_sample - loopbio_start_nidq_sample) / self.input_parameter_dict['find_audio_sync_trains']['nidq_sr']
                nidq_video_difference = nidq_rec_duration - total_video_time_least
                self.message_output(f"For NIDQ, video recording starts at {loopbio_start_nidq_sample} NIDQ sample and ends at {loopbio_end_nidq_sample} NIDQ sample, giving a total NIDQ duration of {nidq_rec_duration:.4f}, which is {nidq_video_difference:.4f} off relative to video duration.")

                # find NIDQ IPI starts and durations in milliseconds
                nidq_rec_ = nidq_digital_bits[loopbio_start_nidq_sample:loopbio_end_nidq_sample, self.input_parameter_dict['find_audio_sync_trains']['nidq_sync_input_bit_position']].copy()
                ipi_start_samples = np.where(np.diff(nidq_rec_) < 0)[0] + 1
                ipi_end_samples = np.where(np.diff(nidq_rec_) > 0)[0]

                if ipi_start_samples[0] < ipi_end_samples[0]:
                    if ipi_start_samples.size == ipi_end_samples.size:
                        nidq_ipi_durations_ms = (((ipi_end_samples - ipi_start_samples) + 1) * 1000 / self.input_parameter_dict['find_audio_sync_trains']['nidq_sr'])
                        nidq_ipi_start_samples = ipi_start_samples
                    else:
                        nidq_ipi_durations_ms = (((ipi_end_samples - ipi_start_samples[:-1]) + 1) * 1000 / self.input_parameter_dict['find_audio_sync_trains']['nidq_sr'])
                        nidq_ipi_start_samples = ipi_start_samples[:-1]
                else:
                    if ipi_start_samples.size == ipi_end_samples.size:
                        nidq_ipi_durations_ms = (((ipi_end_samples[1:] - ipi_start_samples[:-1]) + 1) * 1000 / self.input_parameter_dict['find_audio_sync_trains']['nidq_sr'])
                        nidq_ipi_start_samples = ipi_start_samples[:-1]
                    else:
                        nidq_ipi_durations_ms = (((ipi_end_samples[1:] - ipi_start_samples) + 1) * 1000 / self.input_parameter_dict['find_audio_sync_trains']['nidq_sr'])
                        nidq_ipi_start_samples = ipi_start_samples

                # save NIDQ IPI data
                nidq_data_arr = np.vstack((nidq_ipi_durations_ms, nidq_ipi_start_samples))
                np.save(file=nidq_ipi_data_file, arr=nidq_data_arr)

        ipi_discrepancy_dict = {}
        audio_devices_start_sample_differences = 0
        audio_device_prefixes = ['m', 's']
        for af_idx, audio_file in enumerate(sorted(wave_data_dict.keys())):
            ipi_discrepancy_dict[audio_file[:-4]] = {}
            self.message_output(f"Working on sync data in audio file: {audio_file[:-4]}")
            smart_wait(app_context_bool=self.app_context_bool, seconds=1)

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
                            video_metadata_search = next(pathlib.Path(f"{self.root_directory}{os.sep}video{os.sep}").glob(f"*.{video_key}{os.sep}metadata.yaml"), None)
                            if video_metadata_search:
                                img_store = new_for_filename(str(video_metadata_search))
                                frame_times = np.array(img_store.get_frame_metadata()['frame_time'])
                                frame_times = frame_times - frame_times[0]
                                video_ipi_start_times = frame_times[video_ipi_start_frames]

                            if video_metadata_search and self.input_parameter_dict['find_audio_sync_trains']['extract_exact_video_frame_times_bool']:
                                audio_video_ipi_discrepancy_ms = ((audio_ipi_start_samples / wave_data_dict[audio_file]['sampling_rate']) - video_ipi_start_times) * 1000
                            else:
                                # this comparison is fairer, given that the timing on the video PC is not completely accurate (up to ~4 ms jitter), but both should give roughly similar results
                                audio_video_ipi_discrepancy_ms = ((audio_ipi_start_samples / wave_data_dict[audio_file]['sampling_rate']) - (video_ipi_start_frames / camera_fr[0])) * 1000

                                # the following segment checks whether the IPI video frames indices extracted from the audio file match the video frames indices
                                if next(pathlib.Path(f"{self.root_directory}{os.sep}sync{os.sep}").glob(f"*{audio_device_prefixes[af_idx]}_video_frames_in_audio_samples.txt"), None):
                                    with open(f"{self.root_directory}{os.sep}sync{os.sep}{audio_device_prefixes[af_idx]}_video_frames_in_audio_samples.txt", 'r') as txt_file:
                                       video_fr_starts_in_samples = np.array([line.rstrip() for line in txt_file], dtype=np.int64)

                                    audio_ipi_start_frames = []
                                    for ipi_start_sample in audio_ipi_start_samples:
                                       temp_arr = video_fr_starts_in_samples - ipi_start_sample
                                       audio_ipi_start_frames.append((list(temp_arr).index(max(temp_arr[temp_arr<0]))))

                                    discrepancy_arr = np.array(audio_ipi_start_frames) - video_ipi_start_frames
                                    self.message_output(f"On device {audio_device_prefixes[af_idx]}, the first IPI event had a {discrepancy_arr[0]} fr discrepancy, and the last one had a {discrepancy_arr[-1]} fr discrepancy.")
                                    self.message_output(f"Overall, the min discrepancy is {np.min(discrepancy_arr)} fr and the max discrepancy is {np.max(discrepancy_arr)} fr.")

                            # if the SYNC is acceptable, delete the original audio files
                            if np.max(np.abs(audio_video_ipi_discrepancy_ms)) < self.input_parameter_dict['find_video_sync_trains']['millisecond_divergence_tolerance']:
                                if os.path.exists(f"{self.root_directory}{os.sep}audio{os.sep}original"):
                                    shutil.rmtree(f"{self.root_directory}{os.sep}audio{os.sep}original")

                            ipi_discrepancy_dict[audio_file[:-4]]['ipi_discrepancy_ms'] = audio_video_ipi_discrepancy_ms
                            ipi_discrepancy_dict[audio_file[:-4]]['video_ipi_start_frames'] = video_ipi_start_frames
                            if nidq_file is not None and os.path.isfile(nidq_ipi_data_file):
                                nidq_data_arr = np.load(file=nidq_ipi_data_file)
                                ipi_discrepancy_dict[audio_file[:-4]]['nidq_ipi_durations_ms'] = nidq_data_arr[0, :]
                                ipi_discrepancy_dict[audio_file[:-4]]['nidq_ipi_discrepancy_ms'] = ((nidq_data_arr[1, :] / self.input_parameter_dict['find_audio_sync_trains']['nidq_sr']) * 1000)  - ((video_ipi_start_frames / camera_fr[0]) * 1000)
                                ipi_discrepancy_dict[audio_file[:-4]]['nidq_ipi_start_samples'] = nidq_data_arr[1, :]


            else:
                self.message_output("The IPI sequences on different videos do not match.")

        # check if the audio devices match on IPI start samples
        audio_devices_start_sample_differences = np.abs(audio_devices_start_sample_differences)
        self.message_output(f"The smallest IPI start sample difference across master/slave audio devices is {np.nanmin(audio_devices_start_sample_differences)}, "
                            f"the largest is {np.nanmax(audio_devices_start_sample_differences)}, and the mean is {round(np.nanmean(audio_devices_start_sample_differences), 2)}.")

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
            Cropped channel file(s) to match video duration.
        ----------
        """

        self.message_output(f"Cropping WAV files to video started at: {datetime.now().hour:02d}:{datetime.now().minute:02d}.{datetime.now().second:02d}")
        smart_wait(app_context_bool=self.app_context_bool, seconds=1)

        # load info from camera_frame_count_dict
        with open(sorted(glob.glob(f"{self.root_directory}{os.sep}video{os.sep}*_camera_frame_count_dict.json"))[0], 'r') as frame_count_infile:
            camera_frame_count_dict = json.load(frame_count_infile)
            total_frame_number = camera_frame_count_dict['total_frame_number_least']
            total_video_time = camera_frame_count_dict['total_video_time_least']

        # load audio channels receiving camera triggerbox input
        wave_data_dict = DataLoader(input_parameter_dict={'wave_data_loc': [f"{self.root_directory}{os.sep}audio{os.sep}original"],
                                                          'load_wavefile_data': {'library': 'scipy',
                                                                                 'conditional_arg': [f"_ch{self.input_parameter_dict['crop_wav_files_to_video']['triggerbox_ch_receiving_input']:02d}"]}}).load_wavefile_data()

        # determine device ID(s) that get(s) camera frame trigger pulses
        if self.input_parameter_dict['crop_wav_files_to_video']['device_receiving_input'] == 'both':
            device_ids = ['m', 's']
        else:
            device_ids = [self.input_parameter_dict['crop_wav_files_to_video']['device_receiving_input']]

        # find start/end video frame information file or create a new one
        if os.path.isfile(f"{self.root_directory}{os.sep}audio{os.sep}audio_triggerbox_sync_info.json"):
            with open(f"{self.root_directory}{os.sep}audio{os.sep}audio_triggerbox_sync_info.json", 'r') as audio_dict_infile:
                start_end_video = json.load(audio_dict_infile)
        else:
            start_end_video = {device: {'start_first_recorded_frame': 0, 'end_last_recorded_frame': 0, 'largest_break_duration': 0,
                                        'duration_samples': 0, 'duration_seconds': 0, 'audio_tracking_diff_seconds': 0} for device in device_ids}

        # find camera frame trigger pulses and IPIs in channel file
        for device in device_ids:
            for audio_file in wave_data_dict.keys():
                if f'{device}_' in audio_file:

                    (start_end_video[device]['start_first_recorded_frame'],
                     start_end_video[device]['end_last_recorded_frame'],
                     start_end_video[device]['largest_break_duration'],
                     ttl_break_end_samples,
                     largest_break_end_hop) = self.find_lsb_changes(relevant_array=wave_data_dict[audio_file]['wav_data'], lsb_bool=True, total_frame_number=total_frame_number)

                    # for each audio device, write the sync video frame start times in audio samples
                    if not os.path.isfile(f"{self.root_directory}{os.sep}sync{os.sep}{device}_video_frames_in_audio_samples.txt"):
                        with open(f"{self.root_directory}{os.sep}sync{os.sep}{device}_video_frames_in_audio_samples.txt", 'w') as text_file:
                            for fr in range(total_frame_number):
                                text_file.write(f"{int(ttl_break_end_samples[largest_break_end_hop + fr] + 1 - int(ttl_break_end_samples[largest_break_end_hop] + 1))}" + "\n")

                    start_end_video[device]['duration_samples'] = int(start_end_video[device]['end_last_recorded_frame'] - start_end_video[device]['start_first_recorded_frame'] + 1)
                    start_end_video[device]['duration_seconds'] = round(start_end_video[device]['duration_samples'] / wave_data_dict[audio_file]['sampling_rate'], 4)
                    start_end_video[device]['audio_tracking_diff_seconds'] = round(start_end_video[device]['duration_seconds'] - total_video_time, 4)

                    self.message_output(f"On {device} device, the largest break duration lasted {start_end_video[device]['largest_break_duration'] / wave_data_dict[audio_file]['sampling_rate']:.3f} seconds, "
                                        f"so the first tracking frame started at {start_end_video[device]['start_first_recorded_frame']} samples, and the last joint one ended at "
                                        f"{start_end_video[device]['end_last_recorded_frame']} samples, giving a total audio recording time of {start_end_video[device]['duration_seconds']} seconds, "
                                        f"which is {start_end_video[device]['audio_tracking_diff_seconds']} seconds off relative to tracking.")

                    if 'num_dropouts' in start_end_video[device].keys():
                        self.message_output(f"Also, on {device} device, {start_end_video[device]['num_dropouts']} recording dropout instances were detected.")

                    break

        # save start/end video frame information
        with open(f"{self.root_directory}{os.sep}audio{os.sep}audio_triggerbox_sync_info.json", 'w') as audio_dict_outfile:
            json.dump(start_end_video, audio_dict_outfile, indent=4)

        # create new directory for cropped files and HPSS files
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
               base_name_date = next(key for key in wave_data_dict.keys() if 's_' in key and f"_ch{self.input_parameter_dict['crop_wav_files_to_video']['triggerbox_ch_receiving_input']:02d}" in key)[2:-9]
           if start_end_video['m']['duration_samples'] < start_end_video['s']['duration_samples']:
               s_longer = True
               s_original_arr_indices = np.arange(0, start_end_video['s']['duration_samples'])
               s_new_arr_indices = np.linspace(start=0, stop=start_end_video['s']['duration_samples'] - 1, num=start_end_video['m']['duration_samples'])
               base_name_date = next(key for key in wave_data_dict.keys() if 'm_' in key and f"_ch{self.input_parameter_dict['crop_wav_files_to_video']['triggerbox_ch_receiving_input']:02d}" in key)[2:-9]

        smart_wait(app_context_bool=self.app_context_bool, seconds=1)

        cut_audio_subprocesses = []
        for audio_file in all_audio_files:
            if len(device_ids) == 1:
                outfile_loc = f"{self.root_directory}{os.sep}audio{os.sep}cropped_to_video{os.sep}{os.path.basename(audio_file)[:-4]}_cropped_to_video.wav"
                start_cut_sample = start_end_video[device_ids[0]]['start_first_recorded_frame']
                cut_duration_samples = start_end_video[device_ids[0]]['duration_samples']
                cut_audio_subp = subprocess.Popen(args=f'''{self.command_addition}static_sox {os.path.basename(audio_file)} {outfile_loc} trim {start_cut_sample}s {cut_duration_samples}s''',
                                                  stdout=subprocess.DEVNULL,
                                                  stderr=subprocess.STDOUT,
                                                  cwd=f"{self.root_directory}{os.sep}audio{os.sep}original",
                                                  shell=self.shell_usage_bool)
                cut_audio_subprocesses.append(cut_audio_subp)
            else:
                if 'm_' in audio_file:
                    m_start_cut_sample = start_end_video['m']['start_first_recorded_frame']
                    m_cut_duration_samples = start_end_video['m']['duration_samples']
                    if m_longer:
                        # adjust outfile name
                        default_base_name = os.path.basename(audio_file)[:-4]
                        modified_base_name = default_base_name[:2] + base_name_date + default_base_name[2 + len(base_name_date):]
                        outfile_loc = f"{self.root_directory}{os.sep}audio{os.sep}cropped_to_video{os.sep}{modified_base_name}_cropped_to_video.wav"

                        # trim and adjust tempo
                        tempo_adjustment_factor = start_end_video['m']['duration_samples'] / start_end_video['s']['duration_samples']
                        cut_audio_subp = subprocess.Popen(args=f'''{self.command_addition}static_sox {os.path.basename(audio_file)} {outfile_loc} trim {m_start_cut_sample}s {m_cut_duration_samples}s tempo -s {tempo_adjustment_factor}''',
                                                          stdout=subprocess.DEVNULL,
                                                          stderr=subprocess.STDOUT,
                                                          cwd=f"{self.root_directory}{os.sep}audio{os.sep}original",
                                                          shell=self.shell_usage_bool)
                        cut_audio_subprocesses.append(cut_audio_subp)

                    else:
                        outfile_loc = f"{self.root_directory}{os.sep}audio{os.sep}cropped_to_video{os.sep}{os.path.basename(audio_file)[:-4]}_cropped_to_video.wav"
                        cut_audio_subp = subprocess.Popen(args=f'''{self.command_addition}static_sox {os.path.basename(audio_file)} {outfile_loc} trim {m_start_cut_sample}s {m_cut_duration_samples}s''',
                                                          stdout=subprocess.DEVNULL,
                                                          stderr=subprocess.STDOUT,
                                                          cwd=f"{self.root_directory}{os.sep}audio{os.sep}original",
                                                          shell=self.shell_usage_bool)
                        cut_audio_subprocesses.append(cut_audio_subp)
                else:
                    s_start_cut_sample = start_end_video['s']['start_first_recorded_frame']
                    s_cut_duration_samples = start_end_video['s']['duration_samples']
                    if s_longer:
                        # adjust outfile name
                        default_base_name = os.path.basename(audio_file)[:-4]
                        modified_base_name = default_base_name[:2] + base_name_date + default_base_name[2 + len(base_name_date):]
                        outfile_loc = f"{self.root_directory}{os.sep}audio{os.sep}cropped_to_video{os.sep}{modified_base_name}_cropped_to_video.wav"

                        # trim and adjust tempo
                        tempo_adjustment_factor = start_end_video['s']['duration_samples'] / start_end_video['m']['duration_samples']
                        cut_audio_subp = subprocess.Popen(args=f'''{self.command_addition}static_sox {os.path.basename(audio_file)} {outfile_loc} trim {s_start_cut_sample}s {s_cut_duration_samples}s tempo -s {tempo_adjustment_factor}''',
                                                          stdout=subprocess.DEVNULL,
                                                          stderr=subprocess.STDOUT,
                                                          cwd=f"{self.root_directory}{os.sep}audio{os.sep}original",
                                                          shell=self.shell_usage_bool)
                        cut_audio_subprocesses.append(cut_audio_subp)

                    else:
                        outfile_loc = f"{self.root_directory}{os.sep}audio{os.sep}cropped_to_video{os.sep}{os.path.basename(audio_file)[:-4]}_cropped_to_video.wav"
                        cut_audio_subp = subprocess.Popen(args=f'''{self.command_addition}static_sox {os.path.basename(audio_file)} {outfile_loc} trim {s_start_cut_sample}s {s_cut_duration_samples}s''',
                                                          stdout=subprocess.DEVNULL,
                                                          stderr=subprocess.STDOUT,
                                                          cwd=f"{self.root_directory}{os.sep}audio{os.sep}original",
                                                          shell=self.shell_usage_bool)
                        cut_audio_subprocesses.append(cut_audio_subp)

        while True:
            status_poll = [query_subp.poll() for query_subp in cut_audio_subprocesses]
            if any(elem is None for elem in status_poll):
                smart_wait(app_context_bool=self.app_context_bool, seconds=5)
            else:
                break

        if len(device_ids) > 1:
            for audio_file in all_audio_files:
                if 'm_' in audio_file:
                    if m_longer:
                        # adjust outfile name
                        default_base_name = os.path.basename(audio_file)[:-4]
                        modified_base_name = default_base_name[:2] + base_name_date + default_base_name[2 + len(base_name_date):]
                        outfile_loc = f"{self.root_directory}{os.sep}audio{os.sep}cropped_to_video{os.sep}{modified_base_name}_cropped_to_video.wav"

                        # extract original LSB data
                        m_sr_original, m_data_original = wavfile.read(f'{audio_file}')
                        m_lsb_original = m_data_original[start_end_video['m']['start_first_recorded_frame']:start_end_video['m']['end_last_recorded_frame'] + 1] & 1

                        # resample the LSB data
                        m_lsb_modified = np.where(np.interp(x=m_new_arr_indices, xp=m_original_arr_indices, fp=m_lsb_original).astype(np.int16) > 0.5, 1, 0).astype(np.int16)

                        # load data again and overwrite the LSB
                        m_sr_tempo_adjusted, m_data_tempo_adjusted = wavfile.read(f'{outfile_loc}')
                        if m_data_tempo_adjusted.size == start_end_video['s']['duration_samples']:
                            m_data_modified = (m_data_tempo_adjusted & ~1) ^ m_lsb_modified
                        elif m_data_tempo_adjusted.size > start_end_video['s']['duration_samples']:
                            m_data_modified = (m_data_tempo_adjusted[:start_end_video['s']['duration_samples']] & ~1) ^ m_lsb_modified
                        else:
                            padding_needed = start_end_video['s']['duration_samples'] - m_data_tempo_adjusted.size
                            value_for_padded_part = m_data_tempo_adjusted[-1]
                            padding = np.full(padding_needed, value_for_padded_part, dtype=m_data_tempo_adjusted.dtype)
                            padded_data = np.concatenate((m_data_tempo_adjusted, padding))
                            lsb_value_for_padded_part = m_lsb_modified[-1]
                            lsb_padding = np.full(padding_needed, lsb_value_for_padded_part, dtype=m_lsb_modified.dtype)
                            extended_lsb_array = np.concatenate((m_lsb_modified, lsb_padding))
                            m_data_modified = (padded_data & ~1) ^ extended_lsb_array

                        wavfile.write(filename=outfile_loc, rate=m_sr_original, data=m_data_modified)
                else:
                    if s_longer:
                        # adjust outfile name
                        default_base_name = os.path.basename(audio_file)[:-4]
                        modified_base_name = default_base_name[:2] + base_name_date + default_base_name[2 + len(base_name_date):]
                        outfile_loc = f"{self.root_directory}{os.sep}audio{os.sep}cropped_to_video{os.sep}{modified_base_name}_cropped_to_video.wav"

                        # extract original LSB data
                        s_sr_original, s_data_original = wavfile.read(f'{audio_file}')
                        s_lsb_original = s_data_original[start_end_video['s']['start_first_recorded_frame']:start_end_video['s']['end_last_recorded_frame'] + 1] & 1

                        # resample the LSB data
                        s_lsb_modified = np.where(np.interp(x=s_new_arr_indices, xp=s_original_arr_indices, fp=s_lsb_original).astype(np.int16) > 0.5, 1, 0).astype(np.int16)

                        # load data again and overwrite the LSB
                        s_sr_tempo_adjusted, s_data_tempo_adjusted = wavfile.read(f'{outfile_loc}')
                        if s_data_tempo_adjusted.size == start_end_video['m']['duration_samples']:
                            s_data_modified = (s_data_tempo_adjusted & ~1) ^ s_lsb_modified
                        elif s_data_tempo_adjusted.size > start_end_video['m']['duration_samples']:
                            s_data_modified = (s_data_tempo_adjusted[:start_end_video['m']['duration_samples']] & ~1) ^ s_lsb_modified
                        else:
                            padding_needed = start_end_video['m']['duration_samples'] - s_data_tempo_adjusted.size
                            value_for_padded_part = s_data_tempo_adjusted[-1]
                            padding = np.full(padding_needed, value_for_padded_part, dtype=s_data_tempo_adjusted.dtype)
                            padded_data = np.concatenate((s_data_tempo_adjusted, padding))
                            lsb_value_for_padded_part = s_lsb_modified[-1]
                            lsb_padding = np.full(padding_needed, lsb_value_for_padded_part, dtype=s_lsb_modified.dtype)
                            extended_lsb_array = np.concatenate((s_lsb_modified, lsb_padding))
                            s_data_modified = (padded_data & ~1) ^ extended_lsb_array

                        wavfile.write(filename=outfile_loc, rate=s_sr_original, data=s_data_modified)


        # create HPSS directory
        pathlib.Path(f"{self.root_directory}{os.sep}audio{os.sep}hpss").mkdir(parents=True, exist_ok=True)
