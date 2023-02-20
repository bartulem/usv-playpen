"""
@author: bartulem
Synchronizes files:
(1) the recorded .wav file with tracking file (cuts them to video length).
"""

import json
import os
import pims
import numpy as np
from imgstore import new_for_filename
from numba import njit
from behavioral_experiments import _loop_time
from file_loader import DataLoader
from file_writer import DataWriter
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
                   'current': {'21241563': {'LED_top': [275, 1260], 'LED_middle': [345, 1270], 'LED_bottom': [380, 1233]},
                               '21372315': {'LED_top': [520, 1255], 'LED_middle': [590, 1230], 'LED_bottom': [595, 1257]}}}

    def __init__(self, root_directory=None, input_parameter_dict=None, message_output=None):
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

        if message_output is None:
            self.message_output = print
        else:
            self.message_output = message_output

    @staticmethod
    @njit(parallel=True)
    def find_lsb_changes(sound_array=None,
                         ch_receiving_input=0):

        """
        Description
        ----------
        This method takes a (multi-)channel sound array, extract the LSB
        array and finds all instances where 0 changed to 1, and vice versa.
        ----------

        Parameters
        ----------
        Contains the following set of parameters
            sound_array (np.ndarray)
                (Multi-)channel sound array.
            ch_receiving_input (int)
                Channel receiving digital input; defaults to 0.
        ----------

        Returns
        ----------
        on_to_off (np.ndarray)
            Positions where values change from high to low (low positions).
        off_to_on (np.ndarray)
            Positions where values change from low to high (low positions).
        ----------
        """

        # get the least significant bit array
        if sound_array.ndim == 1:
            lsb_array = sound_array & 1
        else:
            lsb_array = sound_array[:, ch_receiving_input] & 1

        # get switches from ON to OFF and vice versa (both look at the 0 value positions)
        on_to_off = np.where(np.diff(lsb_array) < 0)[0] + 1
        off_to_on = np.where(np.diff(lsb_array) > 0)[0]

        return on_to_off, off_to_on

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

    @staticmethod
    @njit(parallel=True)
    def correct_loose_ttl_samples(raw_data,
                                  ch_num=1,
                                  usgh_sr=None,
                                  ttl_pulse_duration=None,
                                  break_proportion_threshold=None,
                                  ttl_proportion_threshold=None):

        """
        Description
        ----------
        When TTL pulses (e.g., sync LEDs) occur, their duration should be fixed,
        i.e. 250 ms, but sometimes signals are recognized spuriously as HIGH,
        even though they should be LOW. This method finds such events and corrects
        their values back to LOW.
        ----------

        Parameters
        ----------
        Contains the following set of parameters
            raw_data (np.ndarray)
                Sound recording input array.
            ch_num (int)
                In a multichannel array, channel where the TTL is recorded.
            usgh_sr (int)
                Sampling rate of the audio recorder.
            ttl_pulse_duration (int / float)
                TTL duration in seconds.
            break_proportion_threshold (float)
                Proportion of the break event (relative to TTL duration) below which
                every o is turned to 1.
            ttl_proportion_threshold (float)
                Proportion of the TTL event that can still be considered significant.
        ----------

        Returns
        ----------
        raw_data (np.ndarray)
            The LSB-corrected sound recording input array.
        ----------
        """

        # find the least significant bit in the raw data
        if raw_data.ndim == 1:
            lsb_arr = raw_data & 1
        else:
            lsb_arr = raw_data[:, ch_num] & 1

        # find all indices where the LSB changes
        critical_points = np.where((np.diff(lsb_arr) < 0) | (np.diff(lsb_arr) > 0))[0]

        # find all TTL (HIGH) durations
        critical_durations = np.diff(critical_points)
        even_events = critical_durations[::2]
        odd_events = critical_durations[1::2]
        if even_events.mean() < odd_events.mean():
            pulses_durations_samples = even_events
            break_durations_samples = odd_events
            start_idx_durations = 0
            start_idx_breaks = 1
        else:
            pulses_durations_samples = odd_events
            break_durations_samples = even_events
            start_idx_durations = 1
            start_idx_breaks = 0

        # turn all LSB zeros that fall below 50 samples to ones
        if break_proportion_threshold is not None:
            for ap_idx, ap in enumerate(break_durations_samples):
                if ap < int(np.round((usgh_sr * ttl_pulse_duration * break_proportion_threshold))):
                    raw_data[critical_points[start_idx_breaks + (2 * ap_idx)]:critical_points[start_idx_breaks + (2 * ap_idx) + 1] + 1] = \
                        raw_data[critical_points[start_idx_breaks + (2 * ap_idx)]:critical_points[start_idx_breaks + (2 * ap_idx) + 1] + 1] | 1

        # turn all LSB ones that fall below a certain duration threshold to zeros
        if ttl_proportion_threshold is not None:
            for ap_idx, ap in enumerate(pulses_durations_samples):
                if ap < int(np.round((usgh_sr * ttl_pulse_duration * ttl_proportion_threshold))):
                    raw_data[critical_points[start_idx_durations+(2*ap_idx)]:critical_points[start_idx_durations+(2*ap_idx)+1]+1] = \
                        raw_data[critical_points[start_idx_durations+(2*ap_idx)]:critical_points[start_idx_durations+(2*ap_idx)+1]+1] & ~1

        return raw_data

    def find_video_sync_trains(self, total_frame_number):
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
            camera_fps (int)
                Video sampling rate; defaults to 150 (fps).
            sync_pulse_duration (int / float)
                Duration of LED sync pulse; defaults to 0.25 (s)
            millisecond_divergence_tolerance (int / float)
                The amount of variation allowed for sync events converted to ms; defaults to 10 (ms).
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
        if os.path.exists(f"{self.root_directory}{os.sep}video{os.sep}fps_corrected_videos"):
            for corrected_video in os.listdir(f"{self.root_directory}{os.sep}video{os.sep}fps_corrected_videos"):
                video_name_sans_mp4 = corrected_video[:-4]
                if 'calibration' not in corrected_video \
                        and video_name_sans_mp4.split('.')[-1] in self.input_parameter_dict['find_video_sync_trains']['camera_serial_num'] \
                        and self.input_parameter_dict['find_video_sync_trains']['video_extension'] in corrected_video:

                    current_working_dir = f"{self.root_directory}{os.sep}video{os.sep}fps_corrected_videos"
                    video_of_interest = f"{current_working_dir}{os.sep}{corrected_video}"
                    loaded_video = pims.Video(video_of_interest)

                    if not os.path.exists(f"{self.root_directory}{os.sep}sync{os.sep}sync_px_{video_name_sans_mp4}"):
                        # find camera sync pxl
                        max_frame_num = int(round(self.input_parameter_dict['find_video_sync_trains']['camera_fps'] + (self.input_parameter_dict['find_video_sync_trains']['camera_fps'] / 2)))
                        led_px_version = self.input_parameter_dict['find_video_sync_trains']["led_px_version"]
                        led_px_dev = self.input_parameter_dict['find_video_sync_trains']["led_px_dev"]
                        used_camera = video_name_sans_mp4.split('.')[-1]

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
                                peak_intensity_frame_loc = np.where(fame_pxl_values == fame_pxl_values.max())[0][0]

                            frame_of_choice = loaded_video[peak_intensity_frame_loc][dim1_floor_pxl:dim1_ceil_pxl, dim2_floor_pxl:dim2_ceil_pxl]
                            screen_results = np.where(frame_of_choice == frame_of_choice.max())
                            result_dim1, result_dim2 = [dim1_floor_pxl + screen_results[0][0], dim2_floor_pxl + screen_results[1][0]]
                            self.led_px_dict[led_px_version][used_camera][led_position] = [result_dim1, result_dim2]
                            # self.message_output(f"For camera {used_camera}, {led_position} highest intensity pixel is in position {result_dim1},{result_dim2}.")

                        # create memmap array to store the data for posterity
                        mm_arr = np.memmap(f"{self.root_directory}{os.sep}sync{os.sep}sync_px_{video_name_sans_mp4}",
                                           dtype=DataLoader().known_dtypes[self.input_parameter_dict['find_video_sync_trains']['mm_dtype']], mode='w+', shape=(total_frame_number, 3, 3))
                        for fr_idx in range(total_frame_number):
                            processed_frame = modify_memmap_array(loaded_video[fr_idx], mm_arr, fr_idx,
                                                                  self.led_px_dict[led_px_version][used_camera]['LED_top'],
                                                                  self.led_px_dict[led_px_version][used_camera]['LED_middle'],
                                                                  self.led_px_dict[led_px_version][used_camera]['LED_bottom'])

                        # the following line is important for saving memmap file changes
                        mm_arr.flush()

                    # load memmap data
                    leds_array = np.memmap(f"{self.root_directory}{os.sep}sync{os.sep}sync_px_{video_name_sans_mp4}",
                                           dtype=DataLoader().known_dtypes[self.input_parameter_dict['find_video_sync_trains']['mm_dtype']], mode='r', shape=(total_frame_number, 3, 3))

                    # take mean across all three (RGB) channels
                    mean_across_rgb = leds_array.mean(axis=-1)

                    # compute relative change across frames and take median across LEDs
                    diff_across_leds = np.median(self.relative_change_across_array(input_array=mean_across_rgb), axis=1)

                    # find indices where the largest changes occur
                    relative_intensity_threshold = self.input_parameter_dict['find_video_sync_trains']['relative_intensity_threshold']
                    sequence_found = False
                    for threshold_value in np.arange(relative_intensity_threshold-.1, relative_intensity_threshold+.01, .01)[::-1]:
                        if not sequence_found:
                            significant_events = []
                            for x_idx, x in enumerate(diff_across_leds):
                                if x < -threshold_value and (-threshold_value < diff_across_leds[x_idx - 1] < threshold_value):
                                    significant_events.append(x_idx + 1)
                                elif x > threshold_value and (-threshold_value < diff_across_leds[x_idx - 1] < threshold_value):
                                    significant_events.append(x_idx)

                            # get all significant event durations (i.e., LED on periods and IPIs)
                            significant_event_durations = np.diff(a=significant_events)

                            # select only IPI intervals ("-1" is because significant events are computed relative to LED on times,
                            # so one frame has to be removed)
                            camera_fps = self.input_parameter_dict['find_video_sync_trains']['camera_fps']
                            even_event_durations = significant_event_durations[::2]
                            odd_event_durations = significant_event_durations[1::2]
                            if even_event_durations.sum() > odd_event_durations.sum():
                                ipi_durations_frames = even_event_durations - 1
                                if type(ipi_start_frames) == int:
                                    temp_ipi_start_frames = np.array(significant_events[::2]) + 1
                            else:
                                ipi_durations_frames = odd_event_durations - 1
                                if type(ipi_start_frames) == int:
                                    temp_ipi_start_frames = np.array(significant_events[1::2]) + 1

                            # compute IPI durations in milliseconds
                            ipi_durations_ms = np.round(ipi_durations_frames * (1000 / camera_fps))

                            # get actual IPI from CoolTerm-recorded .txt file
                            arduino_ipi_durations = []
                            for txt_file in os.listdir(f"{self.root_directory}{os.sep}sync"):
                                if 'CoolTerm' in txt_file:
                                    with open(f"{self.root_directory}{os.sep}sync{os.sep}{txt_file}", 'r') as ipi_txt_file:
                                        for line_num, line in enumerate(ipi_txt_file.readlines()):
                                            if line_num > 0 and line.strip():
                                                arduino_ipi_durations.append(int(line.strip()))
                                    break
                            arduino_ipi_durations = np.array(arduino_ipi_durations)

                            # match IPI sequences
                            sync_sequence_len = ipi_durations_ms.shape[0]
                            for aid_idx, aid in enumerate(arduino_ipi_durations):
                                if aid_idx + sync_sequence_len <= arduino_ipi_durations.shape[0]:
                                    temp_diffs = arduino_ipi_durations[aid_idx:aid_idx + sync_sequence_len]
                                    if np.all((np.absolute(temp_diffs - ipi_durations_ms)
                                               <= self.input_parameter_dict['find_video_sync_trains']['millisecond_divergence_tolerance'])):
                                        # self.message_output(f"IPI sequence match found in video '{video_of_interest.split(os.sep)[-1]}'!")
                                        sync_sequence_dict[video_name_sans_mp4.split('.')[-1]] = temp_diffs
                                        ipi_start_frames = temp_ipi_start_frames
                                        sequence_found = True
                                        break
                            else:
                                continue

                        else:
                            break

                    else:
                        self.message_output(f"No IPI sequence match found in video '{video_of_interest.split(os.sep)[-1]}'!")

        return ipi_start_frames, sync_sequence_dict

    def find_audio_sync_trains(self, total_frame_number):
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
            sync_pulse_duration (float)
                The duration of the LED TTL pulse; defaults to .25 (s)
            time_proportion_threshold (float)
                Proportion of the TTL event that can still be considered not to be spurious.
            total_frame_number (int)
                Number of frames on the camera containing the minimum total number of frames.
        ----------

        Returns
        ----------
        prediction_error_array (np.ndarray)
            The difference between predicted LED on start video frames and observed LED on start frames.
        ----------
        """

        if os.path.exists(f"{self.root_directory}{os.sep}audio{os.sep}cropped_to_video"):
            wave_data_dict = DataLoader(input_parameter_dict={'wave_data_loc': [f"{self.root_directory}{os.sep}audio{os.sep}cropped_to_video"],
                                                              'load_wavefile_data': {'library': 'scipy',
                                                                                     'conditional_arg': [f"_ch{self.input_parameter_dict['find_audio_sync_trains']['ch_receiving_input']:02d}"]}}).load_wavefile_data()
        else:
            self.message_output(f"Audio directory '{self.root_directory}{os.sep}audio' does not exist!")

        video_ipi_start_frames, video_sync_sequence_dict = self.find_video_sync_trains(total_frame_number=total_frame_number)
        video_sync_sequence_array = np.array(list(video_sync_sequence_dict.values()))

        prediction_error_dict = {}
        for audio_file in wave_data_dict.keys():
            self.message_output(f"Working on sync data in audio file: {audio_file[:-4]}")
            _loop_time(2000)
            # correct the sync LED jitters - when LED is recorded as being ON although it ia actually off
            wave_data_dict[audio_file]['wav_data'] = self.correct_loose_ttl_samples(raw_data=wave_data_dict[audio_file]['wav_data'],
                                                                                    ch_num=self.input_parameter_dict['find_audio_sync_trains']['ch_receiving_input'],
                                                                                    usgh_sr=wave_data_dict[audio_file]['sampling_rate'],
                                                                                    ttl_pulse_duration=self.input_parameter_dict['find_audio_sync_trains']['sync_pulse_duration'],
                                                                                    break_proportion_threshold=self.input_parameter_dict['find_audio_sync_trains']['break_proportion_threshold'],
                                                                                    ttl_proportion_threshold=self.input_parameter_dict['find_audio_sync_trains']['ttl_proportion_threshold'])

            ipi_start_samples, ipi_end_samples = self.find_lsb_changes(sound_array=wave_data_dict[audio_file]['wav_data'],
                                                                       ch_receiving_input=self.input_parameter_dict['find_audio_sync_trains']['ch_receiving_input'])

            if ipi_start_samples[0] < ipi_end_samples[0]:
                if ipi_start_samples.shape[0] == ipi_end_samples.shape[0]:
                    ipi_durations_ms = np.round(((ipi_end_samples - ipi_start_samples) + 1) * 1000 / wave_data_dict[audio_file]['sampling_rate'])
                    audio_ipi_start_samples = ipi_start_samples
                else:
                    ipi_durations_ms = np.round(((ipi_end_samples - ipi_start_samples[:-1]) + 1) * 1000 / wave_data_dict[audio_file]['sampling_rate'])
                    audio_ipi_start_samples = ipi_start_samples[:-1]
            else:
                if ipi_start_samples.shape[0] == ipi_end_samples.shape[0]:
                    ipi_durations_ms = np.round(((ipi_end_samples[1:] - ipi_start_samples[:-1]) + 1) * 1000 / wave_data_dict[audio_file]['sampling_rate'])
                    audio_ipi_start_samples = ipi_start_samples[:-1]
                else:
                    ipi_durations_ms = np.round(((ipi_end_samples[1:] - ipi_start_samples) + 1) * 1000 / wave_data_dict[audio_file]['sampling_rate'])
                    audio_ipi_start_samples = ipi_start_samples

            if (video_sync_sequence_array == video_sync_sequence_array[0]).all():
                for video_idx, video_key in enumerate(video_sync_sequence_dict.keys()):
                    if video_idx == 0:
                        diff_array = np.absolute(ipi_durations_ms - video_sync_sequence_dict[video_key])
                        bool_condition_array = diff_array <= self.input_parameter_dict['find_video_sync_trains']['millisecond_divergence_tolerance']
                        if not np.all(bool_condition_array):
                            self.message_output(f"IPI sequence match NOT found in audio file! There is/are {(~bool_condition_array).sum()} difference(s) larger "
                                                f"than the tolerance and the largest one is {diff_array.max()} ms")
                        else:
                            prediction_error_array = LinRegression(x_data=audio_ipi_start_samples,
                                                                   y_data=video_ipi_start_frames).split_train_test_and_regress(random_dict=self.input_parameter_dict_random)
                            prediction_error_dict[audio_file[:-4]] = prediction_error_array

            else:
                self.message_output("The IPI sequences on different videos do not match.")

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
            camera_serial_num (list)
                Serial numbers of cameras used.
            ch_receiving_input (int)
                Audio channel receiving digital input from Motif; defaults to 1.
            ttl_pulse_duration (float)
                The duration of the TTL pulse; defaults to .000116667 (s)
            ttl_proportion_threshold (float)
                Proportion of the TTL event that can still be considered not to be spurious.
        ----------

        Returns
        ----------
        cropped_to_video (.wav file)
            Cropped channel file(s) to match video file.
        ----------
        """

        # video
        total_frame_number = 1e9
        total_video_time = 1e9
        if os.path.exists(f"{self.root_directory}{os.sep}video"):
            for sub_directory in os.listdir(f"{self.root_directory}{os.sep}video"):
                if 'calibration' not in sub_directory \
                        and sub_directory.split('.')[-1] in self.input_parameter_dict['crop_wav_files_to_video']['camera_serial_num']:
                    img_store = new_for_filename(f"{self.root_directory}{os.sep}video{os.sep}{sub_directory}{os.sep}metadata.yaml")
                    total_frame_num = img_store.frame_count
                    last_frame_num = img_store.frame_max
                    frame_times = img_store.get_frame_metadata()['frame_time']
                    video_duration = frame_times[-1] - frame_times[0]
                    if total_frame_num == last_frame_num:
                        self.message_output(f"Camera {sub_directory.split('.')[-1]} has {total_frame_num} total frames, no dropped frames, "
                                            f"and a video duration of {video_duration:.4f} seconds.")
                        if total_frame_num < total_frame_number:
                            total_frame_number = total_frame_num
                        if video_duration < total_video_time:
                            total_video_time = video_duration
                    else:
                        self.message_output(f"WARNING: The last frame on camera {sub_directory.split('.')[-1]} is {last_frame_num}, which is more than {total_frame_num} in total, "
                                            f"suggesting dropped frames. The video duration is {video_duration:.4f} seconds")

        else:
            self.message_output(f"Video directory '{self.root_directory}{os.sep}video' does not exist!")

        # audio
        if os.path.exists(f"{self.root_directory}{os.sep}audio{os.sep}original"):
            wave_data_dict = DataLoader(input_parameter_dict={'wave_data_loc': [f"{self.root_directory}{os.sep}audio{os.sep}original"],
                                                              'load_wavefile_data': {'library': 'scipy', 'conditional_arg': []}}).load_wavefile_data()
        else:
            self.message_output(f"Audio directory '{self.root_directory}{os.sep}audio' does not exist!")

        # find camera frame trigger pulses and IPIs in channel file
        cam_frames_in_audio = {'m': {'start_first_recorded_frame': 0, 'end_last_recorded_frame': 0, 'ttl_on_durations_in_video': 0},
                               's': {'start_first_recorded_frame': 0, 'end_last_recorded_frame': 0, 'ttl_on_durations_in_video': 0}}
        for audio_file in wave_data_dict.keys():
            if f"_ch{self.input_parameter_dict['crop_wav_files_to_video']['ch_receiving_input']:02d}" in audio_file and 'm_' in audio_file:
                # determine device ID
                device_id = 'm'

                # correct camera TTL jitters
                wave_data_dict[audio_file]['wav_data'] = self.correct_loose_ttl_samples(raw_data=wave_data_dict[audio_file]['wav_data'],
                                                                                        ch_num=self.input_parameter_dict['crop_wav_files_to_video']['ch_receiving_input'],
                                                                                        usgh_sr=wave_data_dict[audio_file]['sampling_rate'],
                                                                                        ttl_pulse_duration=self.input_parameter_dict['crop_wav_files_to_video']['ttl_pulse_duration'],
                                                                                        ttl_proportion_threshold=self.input_parameter_dict['crop_wav_files_to_video']['ttl_proportion_threshold'])

                ttl_break_start_samples, ttl_break_end_samples = self.find_lsb_changes(wave_data_dict[audio_file]['wav_data'])
                ttl_on_start_samples, ttl_on_end_samples = ttl_break_end_samples + 1, ttl_break_start_samples - 1

                # cut starts and ends to the same shape
                if ttl_break_start_samples.shape[0] != ttl_break_end_samples.shape[0]:
                    if ttl_break_start_samples.shape[0] > ttl_break_end_samples.shape[0]:
                        ttl_break_start_samples = ttl_break_start_samples[:-1]
                    elif ttl_break_end_samples.shape[0] > ttl_break_start_samples.shape[0]:
                        ttl_break_end_samples = ttl_break_end_samples[1:]

                if ttl_on_start_samples.shape[0] != ttl_break_end_samples.shape[0]:
                    if ttl_on_start_samples.shape[0] > ttl_break_end_samples.shape[0]:
                        ttl_on_start_samples = ttl_on_start_samples[:-1]
                    elif ttl_on_end_samples.shape[0] > ttl_on_start_samples.shape[0]:
                        ttl_on_end_samples = ttl_on_end_samples[1:]

                # get the inter-pulse intervals & camera frame durations
                if ttl_break_start_samples[0] < ttl_break_end_samples[0]:
                    ttl_break_durations = (ttl_break_end_samples - ttl_break_start_samples) + 1
                    ttl_on_durations = (ttl_on_end_samples[1:] - ttl_on_start_samples[:-1]) + 1
                    largest_ttl_break = np.where(ttl_break_durations == ttl_break_durations.max())[0][0]
                    start_first_recorded_frame = ttl_break_end_samples[largest_ttl_break] + 1
                    end_last_recorded_frame = ttl_break_end_samples[largest_ttl_break + total_frame_number]
                    ttl_on_durations_in_video = ttl_on_durations[largest_ttl_break:largest_ttl_break + total_frame_number]
                elif ttl_break_start_samples[0] > ttl_break_end_samples[0]:
                    ttl_break_durations = (ttl_break_end_samples[1:] - ttl_break_start_samples[:-1]) + 1
                    ttl_on_durations = (ttl_on_end_samples - ttl_on_start_samples) + 1
                    largest_ttl_break = np.where(ttl_break_durations == ttl_break_durations.max())[0][0]
                    start_first_recorded_frame = ttl_break_end_samples[largest_ttl_break + 1] + 1
                    end_last_recorded_frame = ttl_break_end_samples[largest_ttl_break + 1 + total_frame_number]
                    ttl_on_durations_in_video = ttl_on_durations[largest_ttl_break + 1:largest_ttl_break + 1 + total_frame_number]

                cam_frames_in_audio[device_id]['start_first_recorded_frame'] = start_first_recorded_frame
                cam_frames_in_audio[device_id]['end_last_recorded_frame'] = end_last_recorded_frame
                cam_frames_in_audio[device_id]['ttl_on_durations_in_video'] = ttl_on_durations_in_video
                break

        for audio_idx, audio_file in enumerate(wave_data_dict.keys()):
            # determine device ID
            device_id = 'm'

            if f"_ch{self.input_parameter_dict['crop_wav_files_to_video']['ch_receiving_input']:02d}" in audio_file and 'm_' in audio_file:
                ttl_sequence_duration_max = int(np.ceil(wave_data_dict[audio_file]['sampling_rate']*self.input_parameter_dict['crop_wav_files_to_video']['ttl_pulse_duration']))
                ttl_sequence_bool = np.all((cam_frames_in_audio[device_id]['ttl_on_durations_in_video'] >= ttl_sequence_duration_max))
                if cam_frames_in_audio[device_id]['ttl_on_durations_in_video'].shape[0] == total_frame_number and ttl_sequence_bool:
                    total_audio_recording_during_tracking = (cam_frames_in_audio[device_id]['end_last_recorded_frame'] - cam_frames_in_audio[device_id]['start_first_recorded_frame'] + 1) / wave_data_dict[audio_file]['sampling_rate']
                    audio_tracking_difference = total_audio_recording_during_tracking - total_video_time
                    self.message_output(f"On device {device_id}, the first tracking frame started at {cam_frames_in_audio[device_id]['start_first_recorded_frame']} samples, and the last joint one ended at "
                                        f"{cam_frames_in_audio[device_id]['end_last_recorded_frame']} samples, giving a total audio recording time of {total_audio_recording_during_tracking:.4f} seconds, "
                                        f"which is ~{audio_tracking_difference:.4f} seconds off relative to tracking.")
                else:
                    self.message_output(f"Check this file in more detail. The total number of video frames is {total_frame_number}, and you have {ttl_on_durations_in_video.shape[0]}"
                                        f" good TTL frames in the audio recording, while it is {str(ttl_sequence_bool).lower()} that every TTL sequence is equal to or greater than {ttl_sequence_duration_max}.")

            if wave_data_dict[audio_file]['wav_data'].ndim == 1:
                resized_wav_file = wave_data_dict[audio_file]['wav_data'][cam_frames_in_audio[device_id]['start_first_recorded_frame']:cam_frames_in_audio[device_id]['end_last_recorded_frame'] + 1]
            else:
                resized_wav_file = wave_data_dict[audio_file]['wav_data'][cam_frames_in_audio[device_id]['start_first_recorded_frame']:cam_frames_in_audio[device_id]['end_last_recorded_frame'] + 1, :]

            # create new directory for cropped files
            new_directory_cropped_files = f"{self.root_directory}{os.sep}audio{os.sep}cropped_to_video"
            if not os.path.isdir(new_directory_cropped_files):
                os.makedirs(new_directory_cropped_files, exist_ok=False)

            # write to file
            DataWriter(wav_data=resized_wav_file,
                       input_parameter_dict={'wave_write_loc': new_directory_cropped_files,
                                             'write_wavefile_data': {
                                                 'file_name': f"{audio_file[:-4]}_cropped_to_video",
                                                 'sampling_rate': wave_data_dict[audio_file]['sampling_rate'] / 1e3,
                                                 'library': 'scipy'
                                             }}).write_wavefile_data()

        return total_video_time, total_frame_number
