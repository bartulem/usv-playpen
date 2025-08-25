"""
@author: bartulem
Generates playback WAV files and frequency shifts audio segment.
"""

from datetime import datetime
import glob
import noisereduce as nr
import numpy as np
import os
from pathlib import Path
from pydub import AudioSegment
import random
from scipy.io import wavfile
import subprocess
from tqdm import tqdm
from ..os_utils import find_base_path
from ..time_utils import *


class AudioGenerator:

    if os.name == 'nt':
        command_addition = 'cmd /c '
        shell_usage_bool = False
    else:
        command_addition = ''
        shell_usage_bool = True

    def __init__(self, **kwargs):
        """
        Initializes the AudioGenerator class.

        Parameter
        ---------
        exp_id (str)
            Base file server directory.
        root_directory (str)
            Root directory for data; defaults to None.
        create_playback_settings_dict (dict)
            Settings for creating USV playback files; defaults to None.
        freq_shift_settings_dict (dict)
            Frequency shift settings; defaults to None.
        message_output (function)
            Defines output messages; defaults to None.

        Returns
        -------
        -------
        """

        for kw_arg, kw_val in kwargs.items():
            self.__dict__[kw_arg] = kw_val

        self.app_context_bool = is_gui_context()

    def create_naturalistic_usv_playback_wav(self) -> None:
        """
        Description
        ----------
        We tried to extract female-assigned USVs and construct sequences
        using them. It should be noted that ideally in the future, sequences
        should not be constructed from randomly picked USVs, but should be
        taken from a genuine USV sequence.

        The way the code works is as follows:
        (1) it finds all .wav files in the specified directory (female or male or whatever)
        (2) it draws long inter-sequence quiet intervals from the following probability distribution:
            {"2.5s": 0.125, "5s": 0.50, "7.5s": 0.25, "10s": 0.125}
        (3) it draws short inter-USV quiet intervals from the following probability distribution:
            {"0.02s": 0.02, "0.04s": 0.33, "0.06s": 0.45, "0.08s": 0.1, "0.1s": 0.025, "0.15s": 0.045, "0.2s": 0.025, "0.25s": 0.005}
        (4) it draws USV-sequence length from the following probability distribution:
            {"5": 0.5, "10": 0.25, "20": 0.125, "40": 0.0625, "80": 0.0625}
        (5) it starts with a long quiet interval
        (6) it chooses a USV-sequence length
        (7) it plays that many pseudo-randomly chosen USVs, each separated by a short quiet interval
        (8) it goes back to (5) and repeats until exceeds the total playback time

        NB: Run time for ~18 min .wav file is ~2 minutes.
        ----------

        Parameters
        ----------
        ----------

        Returns
        ----------
        usv_playback (.wav file(s))
            Wave file(s) with naturalistic sequences of  USVs.
        """

        self.message_output(f"Creating naturalistic USV playback file(s) started at: {datetime.now().hour:02d}:{datetime.now().minute:02d}.{datetime.now().second:02d}")
        smart_wait(app_context_bool=self.app_context_bool, seconds=1)

        os_base_path = find_base_path()
        local_cup_mount_bool = os.path.ismount(os_base_path)
        if local_cup_mount_bool:
            playback_snippets_dir = f"{os_base_path}{os.sep}{self.exp_id}{os.sep}usv_playback_experiments{os.sep}{self.create_playback_settings_dict['naturalistic_playback_snippets_dir_prefix']}_usv_playback_snippets"
            output_file_dir = f"{os_base_path}{os.sep}{self.exp_id}{os.sep}usv_playback_experiments{os.sep}naturalistic_usv_playback_files"
        else:
            playback_snippets_dir = f"/mnt/cup/labs/falkner/{self.exp_id}/usv_playback_experiments{os.sep}{self.create_playback_settings_dict['naturalistic_playback_snippets_dir_prefix']}_usv_playback_snippets"
            output_file_dir = f"/mnt/cup/labs/falkner/{self.exp_id}/usv_playback_experiments{os.sep}naturalistic_usv_playback_files"

        Path(output_file_dir).mkdir(parents=True, exist_ok=True)

        wav_sampling_rate = self.create_playback_settings_dict['naturalistic_wav_sampling_rate']
        inter_seq_interval_distribution = self.create_playback_settings_dict['inter_seq_interval_distribution']
        usv_seq_length_distribution = self.create_playback_settings_dict['usv_seq_length_distribution']
        inter_usv_interval_distribution = self.create_playback_settings_dict['inter_usv_interval_distribution']
        total_acceptable_playback_time = self.create_playback_settings_dict['total_acceptable_naturalistic_playback_time']

        inter_seq_interval_distribution_periods = [float(isi_key) for isi_key in inter_seq_interval_distribution.keys()]
        inter_seq_interval_distribution_probabilities = list(inter_seq_interval_distribution.values())
        usv_seq_length_distribution_periods = [int(usv_length_key) for usv_length_key in usv_seq_length_distribution.keys()]
        usv_seq_length_distribution_probabilities = list(usv_seq_length_distribution.values())
        inter_usv_interval_distribution_periods = [float(iui_key) for iui_key in inter_usv_interval_distribution.keys()]
        inter_usv_interval_distribution_probabilities = list(inter_usv_interval_distribution.values())

        for number in range(self.create_playback_settings_dict['num_naturalistic_usv_files']):
            smart_wait(app_context_bool=self.app_context_bool, seconds=1)
            current_time = datetime.today().strftime('%Y%m%d_%H%M%S')

            wav_files_list = sorted(glob.glob(f"{playback_snippets_dir}{os.sep}*.wav"))
            replay_wav_arr = np.array(object=[], dtype=np.int16)

            total_playback_time_created = 0
            last_time_updated = 0  # Variable to track progress for tqdm

            replay_txt_path = f"{output_file_dir}{os.sep}{self.create_playback_settings_dict['naturalistic_playback_snippets_dir_prefix']}_usv_playback_{total_acceptable_playback_time}s_{current_time}_spacing.txt"
            usv_id_txt_path = f"{output_file_dir}{os.sep}{self.create_playback_settings_dict['naturalistic_playback_snippets_dir_prefix']}_usv_playback_{total_acceptable_playback_time}s_{current_time}_usvids.txt"

            # This ensures files are opened only once and not overwritten.
            with (open(replay_txt_path, 'w+') as replay_txt_file,
                  open(usv_id_txt_path, 'w+') as usv_id_txt_file,
                  tqdm(total=total_acceptable_playback_time, desc="Generating Playback", unit="s") as pbar):

                while total_playback_time_created < total_acceptable_playback_time:
                    # inter-sequence interval
                    isi = random.choices(inter_seq_interval_distribution_periods, inter_seq_interval_distribution_probabilities, k=1)[0]
                    isi_samples = int(np.ceil(isi * wav_sampling_rate * 1e3))

                    if total_playback_time_created == 0:
                        arr_start_with_isi = np.zeros(isi_samples).astype(np.int16)
                        replay_wav_arr = arr_start_with_isi.copy()
                        usv_id_txt_file.write('ISI \n')
                    else:
                        isi_arr = np.zeros(isi_samples).astype(np.int16)
                        replay_wav_arr = np.concatenate((replay_wav_arr, isi_arr))
                        usv_id_txt_file.write('ISI \n')

                    total_playback_time_created += isi
                    replay_txt_file.write(f'{isi_samples} \n')

                    # choose sequence length
                    usv_seq_length = random.choices(usv_seq_length_distribution_periods, usv_seq_length_distribution_probabilities, k=1)[0]
                    for usv_idx in range(usv_seq_length):
                        # pick USV file
                        random_wav_file = random.choice(wav_files_list)
                        random_wav_file_sr, random_wav_file_data = wavfile.read(random_wav_file)
                        total_playback_time_created += (random_wav_file_data.shape[0] / (wav_sampling_rate * 1e3))

                        replay_txt_file.write(f'{random_wav_file_data.shape[0]} \n')
                        usv_id_txt_file.write(f'{Path(random_wav_file).name} \n')

                        if usv_idx < (usv_seq_length - 1):
                            # inter-USV interval
                            iui = random.choices(inter_usv_interval_distribution_periods, inter_usv_interval_distribution_probabilities, k=1)[0]
                            iui_samples = int(np.ceil(iui * wav_sampling_rate * 1e3))
                            total_playback_time_created += iui

                            replay_wav_arr = np.concatenate((replay_wav_arr, random_wav_file_data, np.zeros(iui_samples).astype(np.int16)))
                            replay_txt_file.write(f'{iui_samples} \n')
                            usv_id_txt_file.write('IUI \n')
                        else:
                            replay_wav_arr = np.concatenate((replay_wav_arr, random_wav_file_data))

                    # manually update the progress bar at the end of each loop
                    update_amount = int(np.floor(total_playback_time_created - last_time_updated))
                    pbar.update(update_amount)
                    last_time_updated = total_playback_time_created

                if pbar.n < pbar.total:
                    pbar.update(pbar.total - pbar.n)

            actual_total_time_sec = int(np.ceil(replay_wav_arr.shape[0] / (wav_sampling_rate * 1e3)))
            self.message_output(f"The total duration of the generated naturalistic playback file is {round(actual_total_time_sec / 60, 2)} min.")

            wavfile.write(filename=f"{output_file_dir}{os.sep}{self.create_playback_settings_dict['naturalistic_playback_snippets_dir_prefix']}_usv_playback_{total_acceptable_playback_time}s_{current_time}.wav",
                          rate=int(wav_sampling_rate * 1e3),
                          data=replay_wav_arr)

        self.message_output(f"Creating naturalistic USV playback file(s) ended at: {datetime.now().hour:02d}:{datetime.now().minute:02d}.{datetime.now().second:02d}")


    def create_usv_playback_wav(self) -> None:
        """
        Description
        ----------
        This method takes .wav files containing individual USVs and concatenates them
        together with a known IPI period between each USV.

        NB: Run time for 10k USVs (~19 min .wav file) is ~18 minutes.
        ----------

        Parameters
        ----------
        spock_cluster_bool (bool)
            If True, the code is run on Spock.
        ----------

        Returns
        ----------
        usv_playback (.wav file(s))
            Wave file(s) with concatenated USVs.
        """

        self.message_output(f"Creating USV playback file(s) started at: {datetime.now().hour:02d}:{datetime.now().minute:02d}.{datetime.now().second:02d}")
        smart_wait(app_context_bool=self.app_context_bool, seconds=1)

        os_base_path = find_base_path()
        local_cup_mount_bool = os.path.ismount(os_base_path)
        if local_cup_mount_bool:
            os_base_path = find_base_path()
            playback_snippets_dir = f"{os_base_path}{os.sep}{self.exp_id}{os.sep}usv_playback_experiments{os.sep}{self.create_playback_settings_dict['playback_snippets_dir']}"
            output_file_dir = f"{os_base_path}{os.sep}{self.exp_id}{os.sep}usv_playback_experiments{os.sep}usv_playback_files"
        else:
            playback_snippets_dir = f"/mnt/cup/labs/falkner/{self.exp_id}/usv_playback_experiments{os.sep}{self.create_playback_settings_dict['playback_snippets_dir']}"
            output_file_dir = f"/mnt/cup/labs/falkner/{self.exp_id}/usv_playback_experiments{os.sep}usv_playback_files"

        Path(output_file_dir).mkdir(parents=True, exist_ok=True)
        ipi_duration = self.create_playback_settings_dict['ipi_duration']
        wav_sampling_rate = self.create_playback_settings_dict['wav_sampling_rate']
        total_usv_number = self.create_playback_settings_dict['total_usv_number']

        for number in range(self.create_playback_settings_dict['num_usv_files']):

            smart_wait(app_context_bool=self.app_context_bool, seconds=1)
            current_time = datetime.today().strftime('%Y%m%d_%H%M%S')

            wav_files_list = sorted(glob.glob(f"{playback_snippets_dir}{os.sep}*.wav"))

            ipi_duration_samples = int(np.ceil(ipi_duration * wav_sampling_rate * 1e3))

            arr_start_with_ipi = np.zeros(ipi_duration_samples).astype(np.int16)
            replay_wav_arr = arr_start_with_ipi.copy()

            with (open(f"{output_file_dir}{os.sep}usv_playback_n={total_usv_number}_{current_time}_spacing.txt",
                       'w+') as replay_txt_file,
                  open(f"{output_file_dir}{os.sep}usv_playback_n={total_usv_number}_{current_time}_usvids.txt",
                       'w+') as usv_id_txt_file):
                replay_txt_file.write(f'{ipi_duration_samples} \n')
                for usv_num in tqdm(range(total_usv_number)):
                    random_wav_file = random.choice(wav_files_list)
                    random_wav_file_sr, random_wav_file_data = wavfile.read(random_wav_file)
                    replay_wav_arr = np.concatenate((replay_wav_arr, random_wav_file_data, arr_start_with_ipi))
                    replay_txt_file.write(f'{random_wav_file_data.shape[0]} \n')
                    replay_txt_file.write(f'{ipi_duration_samples} \n')
                    usv_id_txt_file.write(f'{Path(random_wav_file).name} \n')

            self.message_output(f"The total duration of the generated playback file is {round(replay_wav_arr.shape[0] / (wav_sampling_rate * 1e3) / 60, 2)} min.")

            wavfile.write(filename=f"{output_file_dir}{os.sep}usv_playback_n={total_usv_number}_{current_time}.wav",
                          rate=int(wav_sampling_rate * 1e3),
                          data=replay_wav_arr)

        self.message_output(f"Creating USV playback file(s) ended at: {datetime.now().hour:02d}:{datetime.now().minute:02d}.{datetime.now().second:02d}")

    def frequency_shift_audio_segment(self) -> None:
        """
        Description
        ----------
        This method takes a temporal sequence from an existing USV .wav recording and pitch
        shifts (e.g., shifting down by a tritone) the sequence to human audible range.

        There are several steps in this procedure:
        (1) pitch-shifting occurs on the raw input audio segment
        (2) volume is modulated dynamically on this signal (this is optional):
            the transfer function ('6:-70,...') says that very soft sounds (below -70dB)
            will remain unchanged; sounds in the range -60dB to 0dB (maximum volume)
            will be boosted so that the 60dB dynamic  range of the original music will be
            compressed 3-to-1 into a 20dB range; the -5 (dB) output gain is needed to
            avoid clipping; -90 (dB) for the initial volume will work fine for a clip that
            starts with near silence; the delay of 0.2 (seconds) has the effect of causing
            the compander to react a bit more quickly to sudden volume changes
`       (3) stationary noise reduction is applied to the signal
            3 standard deviations above mean to place the threshold between
            signal and noise
        (4) tempo is adjusted to match the duration of the original audio segment

        These audio files are to be used for presentation purposes only.

        NB, relevant term:
        octave: interval between one pitch and another with double its frequency (12 semitones)
        ----------

        Parameters
        ----------
        ----------

        Returns
        ----------
        audible_chirp (.wav file)
            Wave file with audible chirp data.
        NB: File is saved in the 'audio/frequency_shifted_audio_segments' directory.
        ----------
        """

        self.message_output(f"Frequency shifting of audio segment by {abs(self.freq_shift_settings_dict['fs_octave_shift'])} octaves started at: {datetime.now().hour:02d}:{datetime.now().minute:02d}.{datetime.now().second:02d}")
        smart_wait(app_context_bool=self.app_context_bool, seconds=1)

        audio_dir = self.freq_shift_settings_dict['fs_audio_dir']
        device_id = self.freq_shift_settings_dict['fs_device_id']
        channel_id = self.freq_shift_settings_dict['fs_channel_id']

        wav_sampling_rate = self.freq_shift_settings_dict['fs_wav_sampling_rate']
        seq_start = self.freq_shift_settings_dict['fs_sequence_start']
        seq_duration = self.freq_shift_settings_dict['fs_sequence_duration']
        octave_shift = self.freq_shift_settings_dict['fs_octave_shift']
        volume_adjustment = self.freq_shift_settings_dict['fs_volume_adjustment']

        audio_file_loc = glob.glob(f"{self.root_directory}{os.sep}audio{os.sep}{audio_dir}{os.sep}*{device_id}_*_ch{channel_id:02d}_*.wav")
        output_dir = f"{self.root_directory}{os.sep}audio{os.sep}frequency_shifted_audio_segments"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_file_name = f"{Path(audio_file_loc[0]).name}_start={seq_start}s_duration={seq_duration}s_octave_shift={octave_shift}"

        if len(audio_file_loc) == 1:

            # load audio sequence
            audio_seq = AudioSegment.from_file(file=audio_file_loc[0],
                                               format='wav',
                                               frame_rate=int(wav_sampling_rate * 1e3),
                                               start_second=seq_start,
                                               duration=seq_duration)

            # pitch shift by desired amount
            new_sr = int(audio_seq.frame_rate * (2.0 ** octave_shift))
            audible_seq = audio_seq._spawn(audio_seq.raw_data, overrides={'frame_rate': new_sr})

            audible_seq.export(out_f=f"{output_dir}{os.sep}{output_file_name}.wav",
                               format='wav')

            if volume_adjustment:
                # increase volume
                subprocess.Popen(args=f'''{self.command_addition}sox {output_file_name}.wav {output_file_name}_audible.wav compand 0.3,1 6:-70,-60,-20 -5 -90 0.2''',
                                 cwd=output_dir,
                                 shell=self.shell_usage_bool).wait()

                # noise reduction
                shifted_seq = AudioSegment.from_file(file=f"{output_dir}{os.sep}{output_file_name}_audible.wav",
                                                     format='wav',
                                                     frame_rate=new_sr)
            else:
                shifted_seq = AudioSegment.from_file(file=f"{output_dir}{os.sep}{output_file_name}.wav",
                                                     format='wav',
                                                     frame_rate=new_sr)

            numpy_audio = shifted_seq.get_array_of_samples()
            reduced_noise = nr.reduce_noise(y=numpy_audio, sr=new_sr, stationary=True, n_std_thresh_stationary=3)

            denoised_segment = AudioSegment(np.int16(reduced_noise).tobytes(),
                                           frame_rate=new_sr,
                                           sample_width=shifted_seq.sample_width,
                                           channels=shifted_seq.channels)

            denoised_segment.export(out_f=f"{output_dir}{os.sep}{output_file_name}_audible_denoised.wav",
                                    format='wav')

            # adjust tempo
            tempo_adjustment_factor = audio_seq.frame_rate / denoised_segment.frame_rate
            if 'filtered' not in audio_dir:
                upper_cutoff_freq = int(np.ceil(25000 / (2 ** abs(octave_shift))))
                subprocess.Popen(args=f'''{self.command_addition}sox {output_file_name}_audible_denoised.wav {output_file_name}_audible_denoised_tempo_adjusted.wav sinc {upper_cutoff_freq}-0 tempo -s {tempo_adjustment_factor}''',
                                 cwd=output_dir,
                                 shell=self.shell_usage_bool).wait()
            else:
                subprocess.Popen(args=f'''{self.command_addition}sox {output_file_name}_audible_denoised.wav {output_file_name}_audible_denoised_tempo_adjusted.wav tempo -s {tempo_adjustment_factor}''',
                                 cwd=output_dir,
                                 shell=self.shell_usage_bool).wait()

            os.remove(f"{output_dir}{os.sep}{output_file_name}.wav")
            if volume_adjustment:
                os.remove(f"{output_dir}{os.sep}{output_file_name}_audible.wav")
            os.remove(f"{output_dir}{os.sep}{output_file_name}_audible_denoised.wav")

        else:
            self.message_output(f"Requested audio file not found. Please try again.")
