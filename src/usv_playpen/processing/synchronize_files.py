"""
@author: bartulem
Synchronizes files:
(1) the recorded .wav file with a tracking file (cuts them to video length).
(2) find audio and video sync trains and check whether they match.
(3) performs a check on the e-phys data stream to see if the video duration matches the e-phys recording.
"""

from __future__ import annotations

import configparser
import json
import operator
import pathlib
import re
import shutil
import subprocess
from collections import Counter
from collections.abc import Callable
from datetime import datetime

import cv2
import numpy as np
import probeinterface
from imgstore import new_for_filename
from numba import njit
from scipy.io import wavfile
from spikeinterface.extractors import read_binary
from spikeinterface.extractors.neuropixels_utils import get_neuropixels_sample_shifts_from_probe
from spikeinterface.preprocessing import phase_shift

from ..os_utils import (
    ephys_base_for_data_root,
    first_match_or_raise,
    wait_for_subprocesses,
)
from ..time_utils import is_gui_context, smart_wait
from .load_audio_files import DataLoader


def find_events(diffs: np.ndarray,
                threshold: float) -> tuple:
    """
    Description
    -----------
    This function finds initial event candidates (rising/falling edges)
    in a signal.

    Parameters
    ----------
    diffs (np.ndarray)
        The 1D input signal representing frame-to-frame changes.
    threshold (float)
        The value above which a change is considered significant.

    Returns
    -------
    pos_events (np.ndarray), neg_events (np.ndarray)
        A tuple of two arrays containing the frame indices for debounced
        positive (ON) and negative (OFF) events, respectively.
    """

    stable = np.abs(diffs[:-1]) < threshold
    rising = diffs[1:] > threshold
    falling = diffs[1:] < -threshold

    pos_events = np.where(stable & rising)[0]
    neg_events = np.where(stable & falling)[0] + 1

    return pos_events, neg_events

def _combine_and_sort_events(pos_events: np.ndarray,
                             neg_events: np.ndarray) -> np.ndarray:
    """
    Description
    -----------
    Internal helper function to combine separate ON and OFF event arrays
    into a single, sorted (N, 2) array of [frame, type].

    Parameters
    ----------
    pos_events (np.ndarray)
        Array of frame indices for positive (ON) events.
    neg_events (np.ndarray)
        Array of frame indices for negative (OFF) events.

    Returns
    -------
    (np.ndarray)
        A single (N, 2) array of all events sorted by frame number,
        with type 1 for ON and -1 for OFF.
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
    -----------
    This function filters event pairs that define a state (e.g., an 'ON'
    state) that is shorter than a minimum duration, removing glitches.

    Parameters
    ----------
    pos_events (np.ndarray)
        Array of frame indices for candidate positive (ON) events.
    neg_events (np.ndarray)
        Array of frame indices for candidate negative (OFF) events.
    min_duration (int)
        The minimum number of frames a state must last to be considered valid.

    Returns
    -------
    final_pos (np.ndarray), final_neg (np.ndarray)
        A tuple of two arrays containing the filtered frame indices for
        valid positive (ON) and negative (OFF) events.
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
    -----------
    Ensures the final event sequence is logical by enforcing that event
    types strictly alternate (e.g., ON, OFF, ON...).

    Parameters
    ----------
    pos_events (np.ndarray)
        Array of frame indices for filtered positive (ON) events.
    neg_events (np.ndarray)
        Array of frame indices for filtered negative (OFF) events.

    Returns
    -------
    final_pos (np.ndarray), final_neg (np.ndarray)
        A tuple of two arrays containing the final, validated frame indices.
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

    """
    Synchronizes the recorded data streams of a session.

    This class cross-checks the audio (.wav), video (tracking) and e-phys
    (Neuropixels) recordings of a session: it crops the audio to the video
    duration, finds and matches the audio/video sync trains (via sync-LED
    intensity changes and least-significant-bit pulses), and validates that
    the e-phys recording duration matches the video duration.
    """

    @staticmethod
    def _build_led_px_dict() -> dict:
        """
        Description
        -----------
        Builds and returns a fresh LED pixel coordinate dictionary for each
        Synchronizer instance to avoid shared mutable state between instances.
        The dictionary holds the pixel coordinates used to extract intensity
        changes from the sync LEDs.
        NB: changes in camera positions will change these values!

        Parameters
        ----------

        Returns
        -------
        led_px_dict (dict)
            Dictionary mapping date/version keys to camera serial numbers and
            LED coordinate lists.
        """

        return {'<2022_08_15': {'21241563': {'LED_top': [276, 1248], 'LED_middle': [348, 1260], 'LED_bottom': [377, 1227]},
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

    def __init__(self, root_directory: str = None,
                 input_parameter_dict: dict = None,
                 message_output: Callable | None = None) -> None:
        """
        Description
        -----------
        Initializes the Synchronizer class.

        Parameters
        ----------
        root_directory (str)
            Root directory for data; defaults to None.
        input_parameter_dict (dict)
            Processing parameters; defaults to None.
        message_output (function)
            Defines output messages; defaults to None.

        Returns
        -------
        None
        """

        if input_parameter_dict is None or root_directory is None:
            with open(pathlib.Path(__file__).parent.parent / '_parameter_settings/processing_settings.json') as json_file:
                _settings = json.load(json_file)['synchronize_files']

        self.input_parameter_dict = (
            input_parameter_dict['synchronize_files']['Synchronizer']
            if input_parameter_dict is not None
            else _settings['Synchronizer']
        )
        self.root_directory = root_directory if root_directory is not None else _settings['root_directory']
        self.message_output = message_output if message_output is not None else print

        self.led_px_dict = self._build_led_px_dict()

        self.app_context_bool = is_gui_context()

    def _phase_shift_correct_in_place(self,
                                      npx_recording: pathlib.Path,
                                      meta_path: pathlib.Path,
                                      num_channels: int,
                                      sampling_frequency: float) -> None:
        """
        Description
        -----------
        De-skew one session's Neuropixels AP binary in place (Neuropixels ADC
        sample-time / phase-shift correction).

        A Neuropixels probe multiplexes its analog channels onto a small number of
        ADCs, so within a single sample period the channels are digitized at
        slightly staggered times (a fixed, per-channel fraction-of-a-sample delay).
        This method removes that skew from the AP channels with a Fourier
        fractional-sample shift (SpikeInterface's ``phase_shift``, an
        ``exp(-i 2 pi f tau)`` phase ramp), so that every channel is aligned to a
        common time base before any channel-combining step (referencing /
        whitening / drift correction) in the sorter reads the data. The trailing
        SpikeGLX sync channel is given a shift of exactly zero, so it is left
        bit-for-bit unchanged (it is a digital timing channel, not an analog
        voltage, and must not be resampled).

        The binary is replaced ON DISK at its exact original path, so every
        downstream consumer that globs for a single ``*.ap.bin`` per probe
        directory sees no change. To make the replacement safe, the corrected data
        is first streamed to a temporary file and then atomically renamed over the
        raw binary, so an interrupted run can never leave a half-written ``.bin``.
        A JSON provenance / done-marker is written next to the binary; a session
        that already carries this marker is skipped, so re-running the step can
        never double-shift the same data.

        Parameters
        ----------
        npx_recording (pathlib.Path)
            Path to the raw per-session AP ``.bin`` (interleaved int16, samples x
            channels) that is corrected in place.
        meta_path (pathlib.Path)
            Path to the matching SpikeGLX ``.ap.meta`` whose probe geometry yields
            the per-channel ADC sample shifts.
        num_channels (int)
            Total channels per sample in the binary, INCLUDING the trailing sync
            channel (e.g. 385 = 384 AP + 1 sync for a Neuropixels probe).
        sampling_frequency (float)
            Sampling rate (Hz) attached to the SpikeInterface recording object. The
            correction is expressed in fractions of a sample, so this value does
            not change the result.

        Returns
        -------
        None
            Replaces ``npx_recording`` with its phase-corrected version and writes
            a ``*_phase_shift_applied.json`` provenance marker next to it.
        """

        # a session that already carries the done-marker is skipped, so re-running
        # can never double-shift; the marker name avoids the 'ap.bin' substring so
        # it is not picked up by the downstream '*ap.bin*' / '*ap.bin' globs
        done_marker = npx_recording.parent / f"{npx_recording.name[:-7]}_phase_shift_applied.json"
        if done_marker.is_file():
            self.message_output(f"Phase-shift already applied to {npx_recording.name} (marker present); skipping.")
            return

        # per-channel AP sample shifts from this session's probe geometry; the
        # trailing sync channel(s) get a zero shift so they pass through untouched
        probe = probeinterface.read_spikeglx(meta_path)
        ap_sample_shifts = get_neuropixels_sample_shifts_from_probe(probe)
        if ap_sample_shifts is None:
            error_message = (f"Could not derive Neuropixels sample shifts from {meta_path}; "
                             f"the probe metadata lacks the required ADC fields.")
            raise ValueError(error_message)
        num_sync_channels = num_channels - int(ap_sample_shifts.shape[0])
        if num_sync_channels < 0:
            error_message = (f"{meta_path} implies {ap_sample_shifts.shape[0]} AP channels, which exceeds the "
                             f"{num_channels} channels in {npx_recording.name}.")
            raise ValueError(error_message)
        inter_sample_shift = np.concatenate([ap_sample_shifts,
                                             np.zeros(num_sync_channels, dtype=ap_sample_shifts.dtype)])

        # de-skew the AP channels (sync channel left bit-exact), streaming to a temp
        # file first, then atomically replace the raw binary and drop the marker
        raw_recording = read_binary(str(npx_recording),
                                    sampling_frequency=sampling_frequency,
                                    dtype='int16',
                                    num_channels=num_channels)
        shifted_recording = phase_shift(raw_recording,
                                        inter_sample_shift=inter_sample_shift,
                                        dtype='int16')

        # write the corrected traces to a temporary file, chunk by chunk, through an
        # explicitly-closed handle. `phase_shift` fetches its own margin on every
        # `get_traces` call, so chunked reads reproduce the whole-signal result; and
        # closing the handle before the rename keeps a stray file descriptor from
        # leaking (SpikeInterface's own `write_binary_recording` leaves it open) and
        # lets the atomic replace succeed on Windows, where renaming a file with an
        # open handle fails.
        # NB: this deliberately does not use ``os_utils.atomic_output_path`` -- its temp
        # name (``.<name>.tmp-<pid>``) still contains the ``ap.bin`` substring, which the
        # concatenation step's ``*ap.bin*`` glob would match if a crash left the temp
        # behind; this temp name is glob-safe. The try/except replicates that helper's
        # cleanup-on-failure so an interrupted write never leaves a partial file.
        temp_path = npx_recording.parent / f"{npx_recording.name[:-7]}_phase_shift_tmp"
        num_frames = shifted_recording.get_num_frames()
        chunk_frames = 300_000
        try:
            with temp_path.open('wb') as temp_file:
                for start_frame in range(0, num_frames, chunk_frames):
                    end_frame = min(start_frame + chunk_frames, num_frames)
                    chunk_traces = shifted_recording.get_traces(start_frame=start_frame, end_frame=end_frame)
                    temp_file.write(np.ascontiguousarray(chunk_traces, dtype='int16').tobytes())
            # release the read handles on the source before the rename, so no memory-
            # mapped file object is left open across it (also required on Windows)
            del shifted_recording, raw_recording
            temp_path.replace(npx_recording)
        except BaseException:
            temp_path.unlink(missing_ok=True)
            raise

        with done_marker.open('w') as marker_file:
            json.dump({'phase_shift_applied': True,
                       'inter_sample_shift_source_meta': str(meta_path),
                       'num_channels': int(num_channels),
                       'num_ap_channels_shifted': int(ap_sample_shifts.shape[0]),
                       'num_sync_channels_unshifted': int(num_sync_channels)},
                      marker_file, indent=4)
        self.message_output(f"Applied Neuropixels ADC phase-shift to {npx_recording.name} "
                            f"({int(ap_sample_shifts.shape[0])} AP channels de-skewed, "
                            f"{int(num_sync_channels)} sync channel(s) left unchanged).")

    def validate_ephys_video_sync(self) -> None:
        """
        Description
        -----------
        This method checks whether the time recorded between
        first and last camera signals in the e-phys data stream
        match the total video duration.

        Parameters
        ----------

        Returns
        -------
        binary_files_info (.json file)
            Dictionary w/ information about changepoints, binary file lengths and tracking start/end.
        """

        self.message_output(f"Checking e-phys/video sync started at: {datetime.now().hour:02d}:{datetime.now().minute:02d}:{datetime.now().second:02d}")
        smart_wait(app_context_bool=self.app_context_bool, seconds=1)

        # read headstage sampling rates
        calibrated_sr_config = configparser.ConfigParser()
        calibrated_sr_config.read(pathlib.Path(__file__).parent.parent / '_config/calibrated_sample_rates_imec.ini')

        # load info from camera_frame_count_dict
        with open(
            first_match_or_raise(
                root=pathlib.Path(self.root_directory),
                pattern='*_camera_frame_count_dict.json',
                recursive=True,
                label="camera frame count JSON (ephys sync)",
            ),
            'r',
        ) as frame_count_infile:
            camera_frame_count_dict = json.load(frame_count_infile)
            total_frame_number_least = camera_frame_count_dict['total_frame_number_least']
            total_video_time_least = camera_frame_count_dict['total_video_time_least']

        # phase-shift correction toggle (Neuropixels ADC sample-time de-skew of the
        # AP band, applied in place per session below); opt-in and AP-band only
        npx_file_type = self.input_parameter_dict['validate_ephys_video_sync']['npx_file_type']
        apply_phase_shift = bool(self.input_parameter_dict['validate_ephys_video_sync']['apply_phase_shift'])

        for npx_recording in sorted(pathlib.Path(self.root_directory).rglob(f"*{self.input_parameter_dict['validate_ephys_video_sync']['npx_file_type']}.bin")):

            # parse metadata file for channel and headstage information
            with open(npx_recording.parent / (npx_recording.name[:-3] + 'meta')) as meta_data_file:
                for line in meta_data_file:
                    key, value = line.strip().split('=')
                    if key == 'acqApLfSy':
                        total_probe_ch = int(value.split(',')[0]) + int(value.split(',')[-1])
                    elif key == 'imDatHs_sn':
                        headstage_sn = value
                    elif key == 'imDatPrb_sn':
                        imec_probe_sn = value

            recording_date = pathlib.Path(self.root_directory).name.split('_')[0]
            recording_file_name = npx_recording.name
            imec_probe_id = re.search(r'imec\d', str(npx_recording)).group()

            self.message_output(f"N/V sync for {recording_file_name} with {total_probe_ch} channels, recorded w/ probe #{imec_probe_sn} & headstage #{headstage_sn}.")

            sync_ch_file = str(npx_recording.parent / f"{npx_recording.name[:-7]}_sync_ch_data").replace('.', '_')
            if not pathlib.Path(f'{sync_ch_file}.npy').is_file():

                # load the binary file data
                one_recording = np.memmap(filename=npx_recording, mode='r', dtype='int16', order='C')
                one_sample_num = one_recording.shape[0] // total_probe_ch

                # reshape the array such that channels are rows and samples are columns
                sync_data = one_recording.reshape((total_probe_ch, one_sample_num), order='F')[-1, :]

                # save sync channel data
                np.save(file=sync_ch_file, arr=sync_data)

            # optionally de-skew this session's AP binary in place (Neuropixels ADC
            # phase-shift); the sync channel extracted above is unchanged by the
            # correction, so the tracking-sync logic below is unaffected. AP-band only.
            if apply_phase_shift and npx_file_type == 'ap':
                self._phase_shift_correct_in_place(npx_recording=npx_recording,
                                                   meta_path=npx_recording.parent / (npx_recording.name[:-3] + 'meta'),
                                                   num_channels=total_probe_ch,
                                                   sampling_frequency=float(calibrated_sr_config['CalibratedHeadStages'][headstage_sn]))

            # search for tracking start and end
            ch_sync_data = np.load(file=f'{sync_ch_file}.npy')
            (tracking_start, tracking_end, largest_break_duration,
             _, _) = self.find_lsb_changes(relevant_array=ch_sync_data, lsb_bool=False, total_frame_number=total_frame_number_least)

            largest_break_duration_sec = round(largest_break_duration / float(calibrated_sr_config['CalibratedHeadStages'][headstage_sn]), 3)
            if (tracking_start, tracking_end) != (None, None) and largest_break_duration_sec < 2:
                spike_glx_sr = float(calibrated_sr_config['CalibratedHeadStages'][headstage_sn])
                total_npx_recording_duration = (tracking_end - tracking_start) / spike_glx_sr

                duration_difference = round(number=((total_npx_recording_duration - total_video_time_least) * 1000), ndigits=2)
                comparator_word = 'shorter' if duration_difference < 0 else 'longer'

                self.message_output(f"{recording_file_name} is {abs(duration_difference)} ms {comparator_word} than the video recording with {largest_break_duration_sec} s largest camera break duration.")

                if abs(duration_difference) < self.input_parameter_dict['validate_ephys_video_sync']['npx_ms_divergence_tolerance']:

                    # save tracking start and end in changepoint information JSON file
                    root_ephys = str(ephys_base_for_data_root(self.root_directory) / f'{recording_date}_{imec_probe_id}')
                    pathlib.Path(root_ephys).mkdir(parents=True, exist_ok=True)
                    existing_changepoint_files = sorted(pathlib.Path(root_ephys).glob('changepoints_info_*.json'))
                    if len(existing_changepoint_files) > 0:
                        with open(existing_changepoint_files[0], 'r') as binary_info_input_file:
                            binary_files_info = json.load(binary_info_input_file)

                        if recording_file_name[:-7] not in binary_files_info:
                            binary_files_info[recording_file_name[:-7]] = {'session_start_end': [np.nan, np.nan],
                                                                           'tracking_start_end': [np.nan, np.nan],
                                                                           'largest_camera_break_duration': np.nan,
                                                                           'file_duration_samples': np.nan,
                                                                           'root_directory': str(pathlib.Path(self.root_directory)),
                                                                           'total_num_channels': total_probe_ch,
                                                                           'headstage_sn': headstage_sn,
                                                                           'imec_probe_sn': imec_probe_sn}
                    else:
                        binary_files_info = {recording_file_name[:-7]: {'session_start_end': [np.nan, np.nan],
                                                                        'tracking_start_end': [np.nan, np.nan],
                                                                        'largest_camera_break_duration': np.nan,
                                                                        'file_duration_samples': np.nan,
                                                                        'root_directory': str(pathlib.Path(self.root_directory)),
                                                                        'total_num_channels': total_probe_ch,
                                                                        'headstage_sn': headstage_sn,
                                                                        'imec_probe_sn': imec_probe_sn}}

                    session_start = binary_files_info[recording_file_name[:-7]]['session_start_end'][0]
                    if not np.isnan(session_start):
                        binary_files_info[recording_file_name[:-7]]['tracking_start_end'] = [int(tracking_start) + int(session_start), int(tracking_end) + int(session_start)]
                    else:
                        binary_files_info[recording_file_name[:-7]]['tracking_start_end'] = [int(tracking_start), int(tracking_end)]
                    binary_files_info[recording_file_name[:-7]]['largest_camera_break_duration'] = int(largest_break_duration)

                    with open(pathlib.Path(root_ephys) / f'changepoints_info_{recording_date}_{imec_probe_id}.json', 'w') as binary_info_output_file:
                        json.dump(binary_files_info, binary_info_output_file, indent=4)

                    self.message_output(f"SUCCESS! Tracking start/end sample times saved in {sorted(pathlib.Path(root_ephys).glob('changepoints_info_*.json'))[0]}.")

                else:
                    count_values_in_sync_data = sorted(dict(Counter(ch_sync_data)).items(), key=operator.itemgetter(1), reverse=True)
                    self.message_output(f'{recording_file_name} has a duration difference (e-phys/tracking) of {duration_difference} ms, so above threshold. '
                                        f'Values in original sync data: {count_values_in_sync_data}. Inspect further before proceeding.')

            else:
                self.message_output(f"Tracking end exceeds e-phys recording boundary, so not found for {recording_file_name}.")
                continue

    @staticmethod
    def find_lsb_changes(relevant_array: np.ndarray,
                         lsb_bool: bool = True,
                         total_frame_number: int = 0) -> tuple:

        """
        Description
        -----------
        This method takes a WAV channel sound array or Neuropixels
        sync channel, extracts the LSB part (for WAV files) and
        finds start and end of tracking pulses.

        Parameters
        ----------
        relevant_array (np.ndarray)
            Array to extract sync signal from.
        lsb_bool (bool)
            Whether to extract the least significant bit.
        total_frame_number (int)
            Number of frames on the camera containing the minimum total number of frames.

        Returns
        -------
        start_first_relevant_sample, end_last_relevant_sample,
        largest_break_duration, ttl_break_end_samples, largest_break_end_hop (tuple)
            Start and end of tracking in audio/e-phys samples, the duration of largest break,
            all TTL break end samples, and the index ('hop') into ttl_break_end_samples that
            marks the end of the largest break (i.e. the recording-start hop), not a sample position.
        """

        if lsb_bool:
            lsb_array = relevant_array & 1
            ttl_break_end_samples = np.where((lsb_array[1:] - lsb_array[:-1]) > 0)[0]
        else:
            ttl_break_end_samples = np.where((relevant_array[1:] - relevant_array[:-1]) > 0)[0]

        # With fewer than two TTL break-ends the inter-break diff array is empty,
        # so np.argmax / np.max would raise on a degenerate signal (no usable sync
        # pulses). Return the same None-start/None-end sentinel the
        # out-of-range branch below uses so the caller handles it uniformly.
        if ttl_break_end_samples.shape[0] < 2:
            return None, None, 0, ttl_break_end_samples, 0

        largest_break_end_hop = np.argmax(ttl_break_end_samples[1:] - ttl_break_end_samples[:-1]) + 1

        largest_break_duration = np.max(ttl_break_end_samples[1:] - ttl_break_end_samples[:-1])

        if (total_frame_number + largest_break_end_hop) < ttl_break_end_samples.shape[0]:
            return int(ttl_break_end_samples[largest_break_end_hop] + 1), int(ttl_break_end_samples[largest_break_end_hop + total_frame_number] + 1), int(largest_break_duration), ttl_break_end_samples, largest_break_end_hop
        else:
            return None, None, int(largest_break_duration), ttl_break_end_samples, largest_break_end_hop

    @staticmethod
    @njit(parallel=True)
    def find_ipi_intervals(sound_array: np.ndarray,
                           audio_sr_rate: int = 250000) -> tuple:

        """
        Description
        -----------
        This method takes a WAV channel sound array, extracts the LSB
        part and finds durations and starts of Arduino sync pulses.

        Parameters
        ----------
        sound_array (np.ndarray)
            Sound data array.
        audio_sr_rate (int)
            Sampling rate of audio device; defaults to 250 kHz.

        Returns
        -------
        ipi_durations_ms (np.ndarray), audio_ipi_start_samples (np.ndarray)
            Durations of all found IPI intervals (in ms) and
            start samples of all found IPI intervals.
        """

        # get the least significant bit array
        lsb_array = sound_array & 1

        # falling edges (1->0) mark IPI starts; rising edges (0->1) mark IPI ends
        ipi_start_samples = np.where(np.diff(lsb_array) < 0)[0] + 1
        ipi_end_samples = np.where(np.diff(lsb_array) > 0)[0]

        # A channel with no detectable Arduino pulses yields an empty start or
        # end array, and the `[0]` comparison below would raise. Return empty
        # duration / start-sample arrays as the sentinel so the caller's
        # size-based handling treats the channel as carrying no IPIs.
        if ipi_start_samples.size == 0 or ipi_end_samples.size == 0:
            empty_durations = np.zeros(0, dtype=np.float64)
            empty_starts = np.zeros(0, dtype=ipi_start_samples.dtype)
            return empty_durations, empty_starts

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

    def gather_px_information(self, video_of_interest: str,
                              sync_camera_fps: int | float,
                              camera_id: str,
                              video_name: str,
                              total_frame_number: int) -> None:
        """
        Description
        -----------
        This method finds the sync LEDs in video frames,
        and gathers information about their intensity changes
        over time.

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

        Returns
        -------
        mm_arr (memmap file)
            Memory map file containing pixel intensities of sync LEDs.
        """

        cap = cv2.VideoCapture(video_of_interest)

        # Release the VideoCapture in a finally so an exception anywhere in the
        # decode/centroid loops below (a corrupt frame, an OpenCV error) cannot
        # leak the open capture / its file handle.
        try:
            # get video dimensions
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

            # scan the first ~1.5 s of frames (1.5x fps) to locate the brightest LED frame
            max_frame_num = int(round(sync_camera_fps + (sync_camera_fps / 2)))
            led_px_version = self.input_parameter_dict['find_video_sync_trains']["led_px_version"]
            led_px_dev = self.input_parameter_dict['find_video_sync_trains']["led_px_dev"]
            used_camera = camera_id

            led_positions = list(self.led_px_dict[led_px_version][used_camera].keys())

            # Define each LED's search ROI once (around its approximate coordinate).
            roi_by_led = {}
            for led_position in led_positions:
                led_dim1, led_dim2 = self.led_px_dict[led_px_version][used_camera][led_position]
                roi_by_led[led_position] = (
                    max(0, led_dim1 - led_px_dev),
                    min(frame_height, led_dim1 + led_px_dev),
                    max(0, led_dim2 - led_px_dev),
                    min(frame_width, led_dim2 + led_px_dev),
                )

            peak_intensity = {led_position: -1 for led_position in led_positions}
            peak_intensity_frame_loc = {led_position: -1 for led_position in led_positions}

            # Single sequential pass over the first max_frame_num frames: decode and
            # grayscale-convert each frame ONCE and update every LED's peak tracker,
            # instead of re-seeking (CAP_PROP_POS_FRAMES) and re-decoding the same frames
            # once per LED. Sequential reads return frames pixel-identical to the
            # per-frame seeks here (verified on real H.264 video: 225/225 frames, zero
            # pixel diffs), so each LED's brightest frame is unchanged.
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            for frame_num in range(max_frame_num):
                ret, frame = cap.read()
                if not ret: continue

                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                for led_position in led_positions:
                    y_start, y_end, x_start, x_end = roi_by_led[led_position]
                    roi_intensity = np.max(frame_gray[y_start:y_end, x_start:x_end])
                    if roi_intensity > peak_intensity[led_position]:
                        peak_intensity[led_position] = roi_intensity
                        peak_intensity_frame_loc[led_position] = frame_num

            # For each LED, seek to its brightest frame and find the LED-spot centroid.
            for led_position in led_positions:
                y_start, y_end, x_start, x_end = roi_by_led[led_position]
                cap.set(cv2.CAP_PROP_POS_FRAMES, peak_intensity_frame_loc[led_position])
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
                        self.message_output(f"For {led_position}, centroid found at frame {peak_intensity_frame_loc[led_position]}: ({final_y}, {final_x})")
                    else:
                        self.message_output(f"Could not find centroid for {led_position}, using original coordinate.")

            mm_arr = np.memmap(filename=pathlib.Path(self.root_directory) / 'sync' / f'sync_px_{video_name[:-4]}',
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

            mm_arr.flush()
        finally:
            cap.release()

    def attempt_sequence_match(self, brightness_signal: np.ndarray,
                                camera_fps: float,
                                arduino_ipi_durations: np.ndarray,
                                camera_dir: str) -> tuple:
        """
        Description
        -----------
        This helper function takes a 1D brightness signal and attempts to find a
        match for the ground-truth Arduino IPI sequence. It contains the full
        pipeline of event detection, filtering, and sequence comparison.

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

        Returns
        -------
        (dict, np.ndarray, bool)
            A tuple containing the sync_sequence_dict, the ipi_start_frames,
            and a boolean indicating if the sequence was found. Returns
            (None, None, False) on failure.
        """

        # compute the relative change of the provided signal
        diff_across_leds = 1 - (brightness_signal[1:] / brightness_signal[:-1])

        # find indices where the largest changes occur by iterating through thresholds
        relative_intensity_threshold = self.input_parameter_dict['find_video_sync_trains']['relative_intensity_threshold']

        for threshold_value in np.arange(0.2, relative_intensity_threshold, .01)[::-1]:
            # step 1: find raw candidate events
            pos_significant_events, neg_significant_events = find_events(
                diffs=diff_across_leds,
                threshold=threshold_value
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
                    if 0 < ipi_durations_ms.shape[0] <= len(arduino_ipi_durations):
                        subarray_size = ipi_durations_ms.shape[0]
                        start_indices = np.arange(len(arduino_ipi_durations) - subarray_size + 1)
                        index_matrix = start_indices[:, np.newaxis] + np.arange(subarray_size)
                        arduino_ipi_durations_subarrays = arduino_ipi_durations[index_matrix]

                        result_array = arduino_ipi_durations_subarrays - ipi_durations_ms
                        tolerance = self.input_parameter_dict['find_video_sync_trains']['millisecond_divergence_tolerance']
                        all_zero_matches = np.all(np.abs(result_array) <= tolerance, axis=1)

                        if np.any(all_zero_matches):
                            # Key by the camera serial (the ``Path`` basename), not the full
                            # ``Path``: the exact-frame-times consumer globs
                            # ``video/*.{serial}/metadata.yaml``, which only matches when the key
                            # is the bare serial string. ``pathlib.Path(...).name`` handles both a
                            # Path (production) and a bare serial string (the function's str-typed
                            # signature / tests) without an AttributeError.
                            sync_sequence_dict = {pathlib.Path(camera_dir).name: np.ravel(arduino_ipi_durations_subarrays[all_zero_matches])}
                            ipi_start_frames = temp_ipi_start_frames
                            return sync_sequence_dict, ipi_start_frames, True

        return None, None, False

    def find_video_sync_trains(self, camera_fps: list,
                               total_frame_number: int) -> tuple:

        """
        Description
        -----------
        This method takes video(s) and identifies sync events (from intensity
        changes of sync LEDs) to check sync between different data streams. It uses
        a robust temporal validation method based on known pulse durations.

        Parameters
        ----------
        camera_fps (list)
            List of relevant video sampling rates (in fps).
        total_frame_number (int)
            Number of frames on the camera containing the minimum total number of frames.

        Returns
        -------
        (np.ndarray, dict)
            A tuple containing an array of the start frames of each detected IPI (the frame
            following each ON edge) and a dictionary of the matched IPI sequences for each camera.
        """

        sync_sequence_dict = {}
        ipi_start_frames = np.array([])

        # Read + parse the CoolTerm Arduino IPI log once: its content lives at a
        # fixed path and does not depend on the video/camera being processed, so it
        # need not be re-read and re-parsed inside the per-camera loop below. The
        # exists() guard preserves the old behavior where this read was only reached
        # inside the camera loop (so a missing sync dir -- e.g. no video to process --
        # returns empty rather than raising on iterdir()).
        arduino_ipi_durations = []
        sync_dir = pathlib.Path(self.root_directory) / 'sync'
        if sync_dir.exists():
            for txt_file in sync_dir.iterdir():
                if 'CoolTerm' in txt_file.name:
                    with open(txt_file, 'r') as ipi_txt_file:
                        for line_num, line in enumerate(ipi_txt_file.readlines()):
                            if line_num > 2 and line.strip():
                                arduino_ipi_durations.append(int(line.strip()))
                    break
        arduino_ipi_durations = np.array(arduino_ipi_durations)

        for video_subdir in (pathlib.Path(self.root_directory) / 'video').iterdir():
            if '_' in video_subdir.name or not video_subdir.is_dir(): continue

            sync_cam_idx = 0
            for camera_dir in video_subdir.iterdir():
                if (camera_dir.name == '.DS_Store' or not camera_dir.is_dir()
                        or camera_dir.name not in self.input_parameter_dict['find_video_sync_trains']['sync_camera_serial_num']): continue

                video_name_glob = list(camera_dir.glob('*.mp4'))
                if not video_name_glob: continue
                video_name = sorted(video_name_glob)[0].name

                if ('calibration' in video_name or video_name.split('-')[0] not in self.input_parameter_dict['find_video_sync_trains']['sync_camera_serial_num']
                        or self.input_parameter_dict['find_video_sync_trains']['sync_video_extension'] not in video_name): continue

                video_of_interest = str(camera_dir / video_name)

                if not (pathlib.Path(self.root_directory) / 'sync' / f'sync_px_{video_name[:-4]}').exists():
                    self.gather_px_information(
                        video_of_interest=video_of_interest,
                        sync_camera_fps=camera_fps[sync_cam_idx],
                        camera_id=camera_dir.name,
                        video_name=video_name,
                        total_frame_number=total_frame_number
                    )

                leds_array = np.memmap(filename=pathlib.Path(self.root_directory) / 'sync' / f'sync_px_{video_name[:-4]}',
                                       dtype=np.uint8, mode='r', shape=(total_frame_number, 3, 3))

                mean_across_rgb = leds_array.mean(axis=-1)

                # Use MEDIAN (robust to bright noise)
                self.message_output(f"Attempting sync detection for {camera_dir.name} with MEDIAN signal...")
                brightness_signal_median = np.median(mean_across_rgb, axis=1) + 1e-6

                temp_sync_dict, temp_ipi_frames, sequence_found = self.attempt_sequence_match(
                    brightness_signal=brightness_signal_median,
                    camera_fps=camera_fps[sync_cam_idx],
                    arduino_ipi_durations=arduino_ipi_durations,
                    camera_dir=camera_dir
                )

                # Fallback to MAX if MEDIAN fails (robust to occlusions)
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
        -----------
        This method takes audio files and identifies sync events (from the least
        significant bit inputs) to check sync between different data streams.

        NB: This method also has consequential side effects. It caches the NIDQ IPI
        data to disk (sync/nidq_ipi_data.npy), and, when the audio/video sync passes
        the divergence tolerance, it irreversibly deletes the original (uncropped)
        audio directory (audio/original) via shutil.rmtree.

        Parameters
        ----------

        Returns
        -------
        ipi_discrepancy_dict (dict)
            Contains IPI discrepancies between audio and video sync trains and IPI video start frames.
        """

        self.message_output(f"A/V synchronization started at: {datetime.now().hour:02d}:{datetime.now().minute:02d}:{datetime.now().second:02d}")
        smart_wait(app_context_bool=self.app_context_bool, seconds=1)

        wave_data_dict = DataLoader(input_parameter_dict={'wave_data_loc': [str(pathlib.Path(self.root_directory) / 'audio' / 'cropped_to_video')],
                                                          'load_wavefile_data': {'library': 'scipy',
                                                                                 'conditional_arg': [f"_ch{self.input_parameter_dict['find_audio_sync_trains']['sync_ch_receiving_input']:02d}"]}}).load_wavefile_data()

        # get the total number of frames in the video
        json_loc = first_match_or_raise(
            root=pathlib.Path(self.root_directory),
            pattern='video/*_camera_frame_count_dict.json',
            label="camera frame count JSON",
        )
        with open(json_loc, 'r') as camera_count_json_file:
            camera_fr_count_dict = json.load(camera_count_json_file)
            total_frame_number = camera_fr_count_dict['total_frame_number_least']
            total_video_time_least = camera_fr_count_dict['total_video_time_least']
            camera_fr = [value[1] for key, value in camera_fr_count_dict.items() if key in self.input_parameter_dict['find_video_sync_trains']['sync_camera_serial_num']]

        # find video sync trains
        video_ipi_start_frames, video_sync_sequence_dict = self.find_video_sync_trains(total_frame_number=total_frame_number,
                                                                                       camera_fps=camera_fr)
        # Every camera's recovered sync sequence must be the same length before
        # stacking into a 2D array; a ragged set would make np.array build an
        # object array (or raise), and the all-equal check downstream would then
        # silently misbehave. Validate the lengths and fail with a clear message.
        video_sync_sequences = list(video_sync_sequence_dict.values())
        sync_sequence_lengths = {len(seq) for seq in video_sync_sequences}
        if len(sync_sequence_lengths) > 1:
            error_message = (
                f"Per-camera video sync sequences have mismatched lengths "
                f"{ {key: len(seq) for key, seq in video_sync_sequence_dict.items()} }; "
                f"cannot stack them into a single comparison array."
            )
            raise ValueError(error_message)
        video_sync_sequence_array = np.array(video_sync_sequences)

        # find NIDQ sync trains
        nidq_file = next(iter(sorted(pathlib.Path(self.root_directory).glob("**/*.nidq.bin"))), None)
        nidq_ipi_data_file = pathlib.Path(self.root_directory) / 'sync' / 'nidq_ipi_data.npy'
        if nidq_file is not None and not nidq_ipi_data_file.is_file():
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
                # Bounds-guard the end index (the LSB path at `find_lsb_changes` guards the
                # identical arithmetic): if the recording ends at/near the last camera frame
                # with no trailing triggerbox edge, `largest_break_end_hop + total_frame_number`
                # overruns `triggerbox_bit_changes` and would raise a cryptic IndexError.
                if largest_break_end_hop + total_frame_number >= triggerbox_bit_changes.shape[0]:
                    raise ValueError(
                        f"NIDQ triggerbox has only {triggerbox_bit_changes.shape[0]} rising edges after the "
                        f"largest break at index {largest_break_end_hop}, but {total_frame_number} video frames "
                        f"were expected. The recording appears to end at/before the last camera frame, so the "
                        f"loopbio end NIDQ sample cannot be located — check the NIDQ triggerbox channel and the "
                        f"expected frame count."
                    )
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
                # Guard the elementwise cross-device subtraction against a pulse-
                # count mismatch: if the master and slave devices detected a
                # different number of IPI sync pulses (a dropped/extra pulse -- the
                # very desync this module exists to detect), report it loudly and
                # compare only the aligned prefix rather than crashing the whole
                # method with a broadcasting ValueError. Local copies keep the full
                # audio_ipi_start_samples intact for the video-sync block below.
                prev = audio_devices_start_sample_differences
                cur = audio_ipi_start_samples
                if prev.shape[0] != cur.shape[0]:
                    self.message_output(
                        f"WARNING: master/slave audio devices detected a DIFFERENT number of IPI "
                        f"sync pulses ({prev.shape[0]} vs {cur.shape[0]}) -- a dropped/extra pulse "
                        f"(device desync). Comparing only the first {min(prev.shape[0], cur.shape[0])} "
                        f"aligned pulses."
                    )
                    n = min(prev.shape[0], cur.shape[0])
                    prev, cur = prev[:n], cur[:n]
                audio_devices_start_sample_differences = prev - cur

            if (video_sync_sequence_array == video_sync_sequence_array[0]).all():
                for video_idx, video_key in enumerate(video_sync_sequence_dict.keys()):
                    if video_idx == 0:
                        audio_rounded = np.round(ipi_durations_ms)
                        video_seq = video_sync_sequence_dict[video_key]
                        tolerance = self.input_parameter_dict['find_video_sync_trains']['millisecond_divergence_tolerance']

                        _audio_starts = audio_ipi_start_samples
                        _video_frames = video_ipi_start_frames
                        n_a, n_v = audio_rounded.shape[0], video_seq.shape[0]

                        if n_a == n_v:
                            diff_array = np.absolute(audio_rounded - video_seq)
                        elif abs(n_a - n_v) == 1:
                            if n_a > n_v:
                                candidates = [
                                    (np.absolute(audio_rounded[1:] - video_seq), audio_ipi_start_samples[1:], video_ipi_start_frames, 'dropped first audio pulse'),
                                    (np.absolute(audio_rounded[:-1] - video_seq), audio_ipi_start_samples[:-1], video_ipi_start_frames, 'dropped last audio pulse'),
                                ]
                            else:
                                candidates = [
                                    (np.absolute(audio_rounded - video_seq[1:]), audio_ipi_start_samples, video_ipi_start_frames[1:], 'dropped first video pulse'),
                                    (np.absolute(audio_rounded - video_seq[:-1]), audio_ipi_start_samples, video_ipi_start_frames[:-1], 'dropped last video pulse'),
                                ]
                            diff_array = None
                            for d, as_, vf_, label in candidates:
                                if np.all(d <= tolerance):
                                    diff_array = d
                                    _audio_starts = as_
                                    _video_frames = vf_
                                    self.message_output(f"Shape mismatch of 1 resolved for {audio_file[:-4]} by: {label}.")
                                    break
                            if diff_array is None:
                                n_min = min(n_a, n_v)
                                diff_array = np.absolute(audio_rounded[:n_min] - video_seq[:n_min])
                                _audio_starts = audio_ipi_start_samples[:n_min]
                                _video_frames = video_ipi_start_frames[:n_min]
                        else:
                            diff_array = np.array([np.inf])

                        bool_condition_array = diff_array <= tolerance
                        if not np.all(bool_condition_array):
                            self.message_output(f"IPI sequence match NOT found in audio file! There is/are {(~bool_condition_array).sum()} difference(s) larger "
                                                f"than the tolerance and the largest one is {diff_array.max()} ms")
                        else:
                            video_metadata_search = next(iter(sorted((pathlib.Path(self.root_directory) / 'video').glob(f'*.{video_key}/metadata.yaml'))), None)
                            if video_metadata_search:
                                # close the store as soon as its frame times are read so it never
                                # keeps the backing video chunk open (a lingering handle blocks a
                                # later move/delete of that file on Windows)
                                img_store = new_for_filename(str(video_metadata_search))
                                try:
                                    frame_times = np.array(img_store.get_frame_metadata()['frame_time'])
                                finally:
                                    img_store.close()
                                frame_times = frame_times - frame_times[0]
                                video_ipi_start_times = frame_times[_video_frames]

                            if video_metadata_search and self.input_parameter_dict['find_audio_sync_trains']['extract_exact_video_frame_times_bool']:
                                audio_video_ipi_discrepancy_ms = ((_audio_starts / wave_data_dict[audio_file]['sampling_rate']) - video_ipi_start_times) * 1000
                            else:
                                # this comparison is fairer, given that the timing on the video PC is not completely accurate (up to ~4 ms jitter), but both should give roughly similar results
                                audio_video_ipi_discrepancy_ms = ((_audio_starts / wave_data_dict[audio_file]['sampling_rate']) - (_video_frames / camera_fr[0])) * 1000

                                # the following segment checks whether the IPI video frames indices extracted from the audio file match the video frames indices
                                if next(iter(sorted((pathlib.Path(self.root_directory) / 'sync').glob(f'*{audio_device_prefixes[af_idx]}_video_frames_in_audio_samples.txt'))), None):
                                    with (pathlib.Path(self.root_directory) / 'sync' / f'{audio_device_prefixes[af_idx]}_video_frames_in_audio_samples.txt').open() as txt_file:
                                       video_fr_starts_in_samples = np.array([line.rstrip() for line in txt_file], dtype=np.int64)

                                    # video_fr_starts_in_samples is strictly increasing (cumulative frame
                                    # sample positions), so the "last video frame that started before each
                                    # audio IPI event" is searchsorted(..., side='left') - 1 -- vectorized,
                                    # vs the previous O(n_audio * n_video) loop with a per-event Python
                                    # list().index(). idx < 0 marks an event preceding every video frame
                                    # (NaN, with the same per-event message, in the same order). Verified
                                    # byte-identical against real strictly-increasing frame-start data.
                                    last_frame_idx = np.searchsorted(
                                        video_fr_starts_in_samples, _audio_starts, side='left'
                                    ) - 1
                                    audio_ipi_start_frames = last_frame_idx.astype(np.float64)
                                    for precede_pos in np.where(last_frame_idx < 0)[0]:
                                        self.message_output(
                                            f"On device {audio_device_prefixes[af_idx]}, audio IPI start sample "
                                            f"{int(_audio_starts[precede_pos])} precedes all video frame starts; "
                                            f"marking its frame index as NaN."
                                        )
                                        audio_ipi_start_frames[precede_pos] = np.nan

                                    discrepancy_arr = np.array(audio_ipi_start_frames) - _video_frames
                                    self.message_output(f"On device {audio_device_prefixes[af_idx]}, the first IPI event had a {discrepancy_arr[0]} fr discrepancy, and the last one had a {discrepancy_arr[-1]} fr discrepancy.")
                                    self.message_output(f"Overall, the min discrepancy is {np.min(discrepancy_arr)} fr and the max discrepancy is {np.max(discrepancy_arr)} fr.")

                            # if the SYNC is acceptable, delete the original audio files
                            if np.max(np.abs(audio_video_ipi_discrepancy_ms)) < self.input_parameter_dict['find_video_sync_trains']['millisecond_divergence_tolerance']:
                                original_audio_dir = pathlib.Path(self.root_directory) / 'audio' / 'original'
                                if original_audio_dir.exists():
                                    shutil.rmtree(original_audio_dir)

                            ipi_discrepancy_dict[audio_file[:-4]]['ipi_discrepancy_ms'] = audio_video_ipi_discrepancy_ms
                            ipi_discrepancy_dict[audio_file[:-4]]['video_ipi_start_frames'] = _video_frames
                            if nidq_file is not None and nidq_ipi_data_file.is_file():
                                nidq_data_arr = np.load(file=nidq_ipi_data_file)
                                ipi_discrepancy_dict[audio_file[:-4]]['nidq_ipi_durations_ms'] = nidq_data_arr[0, :]
                                ipi_discrepancy_dict[audio_file[:-4]]['nidq_ipi_discrepancy_ms'] = ((nidq_data_arr[1, :] / self.input_parameter_dict['find_audio_sync_trains']['nidq_sr']) * 1000) - ((_video_frames / camera_fr[0]) * 1000)
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
        -----------
        This method takes a WAV file audio recording to find sequences of recorded
        video frames in the LSB of the triggerbox input channel, and then crops the audio file to
        match the length from the beginning of the first to the end of the last video frame.

        NB: If there are two audio recording devices and if they are not synchronized, both
        sets of audio files are cut to the length of the shorter one. This entails resampling
        longer audio files to match the shorter duration (on one device) using SoX, and the
        LSB of those files is resampled and then maintained in the final audio file.

        Parameters
        ----------

        Returns
        -------
        cropped_to_video (.wav file)
            Cropped channel file(s) to match video duration.
        """

        self.message_output(f"Cropping WAV files to video started at: {datetime.now().hour:02d}:{datetime.now().minute:02d}:{datetime.now().second:02d}")
        smart_wait(app_context_bool=self.app_context_bool, seconds=1)

        # load info from camera_frame_count_dict
        with first_match_or_raise(
            root=pathlib.Path(self.root_directory),
            pattern='video/*_camera_frame_count_dict.json',
            label="camera_frame_count_dict JSON",
        ).open() as frame_count_infile:
            camera_frame_count_dict = json.load(frame_count_infile)
            total_frame_number = camera_frame_count_dict['total_frame_number_least']
            total_video_time = camera_frame_count_dict['total_video_time_least']

        # load audio channels receiving camera triggerbox input
        wave_data_dict = DataLoader(input_parameter_dict={'wave_data_loc': [str(pathlib.Path(self.root_directory) / 'audio' / 'original')],
                                                          'load_wavefile_data': {'library': 'scipy',
                                                                                 'conditional_arg': [f"_ch{self.input_parameter_dict['crop_wav_files_to_video']['triggerbox_ch_receiving_input']:02d}"]}}).load_wavefile_data()

        # determine device ID(s) that get(s) camera frame trigger pulses
        if self.input_parameter_dict['crop_wav_files_to_video']['device_receiving_input'] == 'both':
            device_ids = ['m', 's']
        else:
            device_ids = [self.input_parameter_dict['crop_wav_files_to_video']['device_receiving_input']]

        # find start/end video frame information file or create a new one
        if (pathlib.Path(self.root_directory) / 'audio' / 'audio_triggerbox_sync_info.json').is_file():
            with (pathlib.Path(self.root_directory) / 'audio' / 'audio_triggerbox_sync_info.json').open() as audio_dict_infile:
                start_end_video = json.load(audio_dict_infile)
        else:
            start_end_video = {device: {'start_first_recorded_frame': 0, 'end_last_recorded_frame': 0, 'largest_break_duration': 0,
                                        'duration_samples': 0, 'duration_seconds': 0, 'audio_tracking_diff_seconds': 0} for device in device_ids}

        # find camera frame trigger pulses and IPIs in channel file
        for device in device_ids:
            for audio_file in wave_data_dict:
                if f'{device}_' in audio_file:

                    (start_end_video[device]['start_first_recorded_frame'],
                     start_end_video[device]['end_last_recorded_frame'],
                     start_end_video[device]['largest_break_duration'],
                     ttl_break_end_samples,
                     largest_break_end_hop) = self.find_lsb_changes(relevant_array=wave_data_dict[audio_file]['wav_data'], lsb_bool=True, total_frame_number=total_frame_number)

                    # for each audio device, write the sync video frame start times in audio samples
                    if not (pathlib.Path(self.root_directory) / 'sync' / f'{device}_video_frames_in_audio_samples.txt').is_file():
                        with (pathlib.Path(self.root_directory) / 'sync' / f'{device}_video_frames_in_audio_samples.txt').open('w') as text_file:
                            for fr in range(total_frame_number):
                                text_file.write(f"{int(ttl_break_end_samples[largest_break_end_hop + fr] + 1 - int(ttl_break_end_samples[largest_break_end_hop] + 1))}" + "\n")

                    start_end_video[device]['duration_samples'] = int(start_end_video[device]['end_last_recorded_frame'] - start_end_video[device]['start_first_recorded_frame'] + 1)
                    start_end_video[device]['duration_seconds'] = round(start_end_video[device]['duration_samples'] / wave_data_dict[audio_file]['sampling_rate'], 4)
                    start_end_video[device]['audio_tracking_diff_seconds'] = round(start_end_video[device]['duration_seconds'] - total_video_time, 4)

                    self.message_output(f"On {device} device, the largest break duration lasted {start_end_video[device]['largest_break_duration'] / wave_data_dict[audio_file]['sampling_rate']:.3f} seconds, "
                                        f"so the first tracking frame started at {start_end_video[device]['start_first_recorded_frame']} samples, and the last joint one ended at "
                                        f"{start_end_video[device]['end_last_recorded_frame']} samples, giving a total audio recording time of {start_end_video[device]['duration_seconds']} seconds, "
                                        f"which is {start_end_video[device]['audio_tracking_diff_seconds']} seconds off relative to tracking.")

                    if 'num_dropouts' in start_end_video[device]:
                        self.message_output(f"Also, on {device} device, {start_end_video[device]['num_dropouts']} recording dropout instances were detected.")

                    break

        # save start/end video frame information
        with (pathlib.Path(self.root_directory) / 'audio' / 'audio_triggerbox_sync_info.json').open('w') as audio_dict_outfile:
            json.dump(start_end_video, audio_dict_outfile, indent=4)

        # create new directory for cropped files and HPSS files
        (pathlib.Path(self.root_directory) / 'audio' / 'cropped_to_video').mkdir(parents=True, exist_ok=True)

        # find all audio files
        all_audio_files = sorted(pathlib.Path(self.root_directory).glob('audio/original/*.wav'))

        m_longer = False
        s_longer = False
        if len(device_ids) > 1:
           triggerbox_ch_str = f"_ch{self.input_parameter_dict['crop_wav_files_to_video']['triggerbox_ch_receiving_input']:02d}"
           if start_end_video['m']['duration_samples'] > start_end_video['s']['duration_samples']:
               m_longer = True
               m_original_arr_indices = np.arange(0, start_end_video['m']['duration_samples'])
               m_new_arr_indices = np.linspace(start=0, stop=start_end_video['m']['duration_samples'] - 1, num=start_end_video['s']['duration_samples'])
               base_name_keys = [key for key in wave_data_dict if 's_' in key and triggerbox_ch_str in key]
               if not base_name_keys:
                   msg = (
                       f"No 's_*{triggerbox_ch_str}*' key found in wave_data_dict "
                       f"(available keys: {list(wave_data_dict.keys())})."
                   )
                   raise KeyError(msg)
               base_name_date = base_name_keys[0][2:-9]
           if start_end_video['m']['duration_samples'] < start_end_video['s']['duration_samples']:
               s_longer = True
               s_original_arr_indices = np.arange(0, start_end_video['s']['duration_samples'])
               s_new_arr_indices = np.linspace(start=0, stop=start_end_video['s']['duration_samples'] - 1, num=start_end_video['m']['duration_samples'])
               base_name_keys = [key for key in wave_data_dict if 'm_' in key and triggerbox_ch_str in key]
               if not base_name_keys:
                   msg = (
                       f"No 'm_*{triggerbox_ch_str}*' key found in wave_data_dict "
                       f"(available keys: {list(wave_data_dict.keys())})."
                   )
                   raise KeyError(msg)
               base_name_date = base_name_keys[0][2:-9]

        smart_wait(app_context_bool=self.app_context_bool, seconds=1)

        cut_audio_subprocesses = []
        for audio_file in all_audio_files:
            if len(device_ids) == 1:
                outfile_loc = str(pathlib.Path(self.root_directory) / 'audio' / 'cropped_to_video' / f'{audio_file.stem}_cropped_to_video.wav')
                start_cut_sample = start_end_video[device_ids[0]]['start_first_recorded_frame']
                cut_duration_samples = start_end_video[device_ids[0]]['duration_samples']
                cut_audio_subp = subprocess.Popen(
                                                  args=["static_sox", audio_file.name, outfile_loc, "trim", f"{start_cut_sample}s", f"{cut_duration_samples}s"],
                                                  stdout=subprocess.DEVNULL,
                                                  stderr=subprocess.STDOUT,
                                                  cwd=pathlib.Path(self.root_directory) / 'audio' / 'original',
                                                  shell=False)
                cut_audio_subprocesses.append(cut_audio_subp)
            else:
                if 'm_' in audio_file.name:
                    m_start_cut_sample = start_end_video['m']['start_first_recorded_frame']
                    m_cut_duration_samples = start_end_video['m']['duration_samples']
                    if m_longer:
                        # adjust outfile name
                        default_base_name = audio_file.stem
                        modified_base_name = default_base_name[:2] + base_name_date + default_base_name[2 + len(base_name_date):]
                        outfile_loc = str(pathlib.Path(self.root_directory) / 'audio' / 'cropped_to_video' / f'{modified_base_name}_cropped_to_video.wav')

                        # trim and adjust tempo
                        tempo_adjustment_factor = start_end_video['m']['duration_samples'] / start_end_video['s']['duration_samples']
                        cut_audio_subp = subprocess.Popen(
                                                          args=["static_sox", audio_file.name, outfile_loc, "trim", f"{m_start_cut_sample}s", f"{m_cut_duration_samples}s", "tempo", "-s", str(tempo_adjustment_factor)],
                                                          stdout=subprocess.DEVNULL,
                                                          stderr=subprocess.STDOUT,
                                                          cwd=pathlib.Path(self.root_directory) / 'audio' / 'original',
                                                          shell=False)
                        cut_audio_subprocesses.append(cut_audio_subp)

                    else:
                        outfile_loc = str(pathlib.Path(self.root_directory) / 'audio' / 'cropped_to_video' / f'{audio_file.stem}_cropped_to_video.wav')
                        cut_audio_subp = subprocess.Popen(
                                                          args=["static_sox", audio_file.name, outfile_loc, "trim", f"{m_start_cut_sample}s", f"{m_cut_duration_samples}s"],
                                                          stdout=subprocess.DEVNULL,
                                                          stderr=subprocess.STDOUT,
                                                          cwd=pathlib.Path(self.root_directory) / 'audio' / 'original',
                                                          shell=False)
                        cut_audio_subprocesses.append(cut_audio_subp)
                else:
                    s_start_cut_sample = start_end_video['s']['start_first_recorded_frame']
                    s_cut_duration_samples = start_end_video['s']['duration_samples']
                    if s_longer:
                        # adjust outfile name
                        default_base_name = audio_file.stem
                        modified_base_name = default_base_name[:2] + base_name_date + default_base_name[2 + len(base_name_date):]
                        outfile_loc = str(pathlib.Path(self.root_directory) / 'audio' / 'cropped_to_video' / f'{modified_base_name}_cropped_to_video.wav')

                        # trim and adjust tempo
                        tempo_adjustment_factor = start_end_video['s']['duration_samples'] / start_end_video['m']['duration_samples']
                        cut_audio_subp = subprocess.Popen(
                                                          args=["static_sox", audio_file.name, outfile_loc, "trim", f"{s_start_cut_sample}s", f"{s_cut_duration_samples}s", "tempo", "-s", str(tempo_adjustment_factor)],
                                                          stdout=subprocess.DEVNULL,
                                                          stderr=subprocess.STDOUT,
                                                          cwd=pathlib.Path(self.root_directory) / 'audio' / 'original',
                                                          shell=False)
                        cut_audio_subprocesses.append(cut_audio_subp)

                    else:
                        outfile_loc = str(pathlib.Path(self.root_directory) / 'audio' / 'cropped_to_video' / f'{audio_file.stem}_cropped_to_video.wav')
                        cut_audio_subp = subprocess.Popen(
                                                          args=["static_sox", audio_file.name, outfile_loc, "trim", f"{s_start_cut_sample}s", f"{s_cut_duration_samples}s"],
                                                          stdout=subprocess.DEVNULL,
                                                          stderr=subprocess.STDOUT,
                                                          cwd=pathlib.Path(self.root_directory) / 'audio' / 'original',
                                                          shell=False)
                        cut_audio_subprocesses.append(cut_audio_subp)

        # 2-hour budget — sox "cut to video" on very long sessions can take a
        # while, but anything beyond this almost certainly means a hang.
        wait_for_subprocesses(
            subps=cut_audio_subprocesses,
            max_seconds=2 * 60 * 60,
            label="audio cut-to-video",
            poll_interval_s=5,
            message_output=self.message_output,
            raise_on_nonzero=False,
            raise_on_timeout=False,
        )

        if len(device_ids) > 1:
            for audio_file in all_audio_files:
                if 'm_' in audio_file.name:
                    if m_longer:
                        # adjust outfile name
                        default_base_name = audio_file.stem
                        modified_base_name = default_base_name[:2] + base_name_date + default_base_name[2 + len(base_name_date):]
                        outfile_loc = str(pathlib.Path(self.root_directory) / 'audio' / 'cropped_to_video' / f'{modified_base_name}_cropped_to_video.wav')

                        # extract original LSB data
                        m_sr_original, m_data_original = wavfile.read(f'{audio_file}')
                        m_lsb_original = m_data_original[start_end_video['m']['start_first_recorded_frame']:start_end_video['m']['end_last_recorded_frame'] + 1] & 1

                        # resample the LSB data
                        m_lsb_modified = np.where(np.interp(x=m_new_arr_indices, xp=m_original_arr_indices, fp=m_lsb_original) > 0.5, 1, 0).astype(np.int16)

                        # load data again and overwrite the LSB
                        _, m_data_tempo_adjusted = wavfile.read(f'{outfile_loc}')
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
                elif s_longer:
                        # adjust outfile name
                        default_base_name = audio_file.stem
                        modified_base_name = default_base_name[:2] + base_name_date + default_base_name[2 + len(base_name_date):]
                        outfile_loc = str(pathlib.Path(self.root_directory) / 'audio' / 'cropped_to_video' / f'{modified_base_name}_cropped_to_video.wav')

                        # extract original LSB data
                        s_sr_original, s_data_original = wavfile.read(f'{audio_file}')
                        s_lsb_original = s_data_original[start_end_video['s']['start_first_recorded_frame']:start_end_video['s']['end_last_recorded_frame'] + 1] & 1

                        # resample the LSB data
                        s_lsb_modified = np.where(np.interp(x=s_new_arr_indices, xp=s_original_arr_indices, fp=s_lsb_original) > 0.5, 1, 0).astype(np.int16)

                        # load data again and overwrite the LSB
                        _, s_data_tempo_adjusted = wavfile.read(f'{outfile_loc}')
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
        (pathlib.Path(self.root_directory) / 'audio' / 'hpss').mkdir(parents=True, exist_ok=True)
