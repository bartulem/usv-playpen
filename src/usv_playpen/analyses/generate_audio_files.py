"""
@author: bartulem
Generates playback WAV files and frequency shifts audio segment.
"""

import os
import random
import subprocess
from datetime import datetime
from pathlib import Path

import librosa
import noisereduce as nr
import numpy as np
import soundfile as sf
from scipy.io import wavfile
from tqdm import tqdm

from ..os_utils import find_base_path
from ..time_utils import is_gui_context, smart_wait


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
        Constructs naturalistic USV playback sequences by sampling inter-event
        intervals and sequence lengths from empirically derived distributions.

        Inter-USV intervals (IUI) and inter-sequence intervals (ISI) are sampled
        from a sex-specific 3-component Gaussian mixture model (GMM) fit to
        log-transformed empirical interval data:
            - IUI: first Gaussian component (shortest intervals, ~60 ms peak)
            - ISI: third Gaussian component (longest intervals, seconds-scale)

        Sequence length is drawn from N(13, 5) clipped to [3, 23] USVs.
        Sex is inferred from naturalistic_playback_snippets_dir_prefix.

        The way the code works is as follows:
        (1) it finds all .wav files in the specified directory (female or male)
        (2) it draws a long inter-sequence quiet interval (ISI) from the third
            Gaussian of the sex-specific GMM in log space, then exponentiates
        (3) it draws a sequence length from N(13, 5) clipped to [3, 23]
        (4) it plays that many pseudo-randomly chosen USVs, each separated by
            a short inter-USV interval (IUI) drawn from the first Gaussian
            of the sex-specific GMM in log space, then exponentiated
        (5) it goes back to (2) and repeats until exceeding total playback time

        NB: Run time for ~18 min .wav file is ~2 minutes.
        ----------

        Parameters
        ----------
        ----------

        Returns
        ----------
        usv_playback (.wav file(s))
            Wave file(s) with naturalistic sequences of USVs.
        ----------
        """

        _GMM_PARAMS = {
            'male': {
                'means': [-2.78176965, -1.61892112, -0.62569187],
                'sds':   [0.26162863,  0.77768956,  2.2298624],
            },
            'female': {
                'means': [-2.76859759, -1.64223541,  1.88505038],
                'sds':   [0.26499761,  1.07984569,   1.36805932],
            },
        }

        self.message_output(f"Creating naturalistic USV playback file(s) started at: {datetime.now().hour:02d}:{datetime.now().minute:02d}.{datetime.now().second:02d}")
        smart_wait(app_context_bool=self.app_context_bool, seconds=1)

        os_base_path = find_base_path()
        local_cup_mount_bool = os.path.ismount(os_base_path)
        prefix = self.create_playback_settings_dict['naturalistic_playback_snippets_dir_prefix']
        if local_cup_mount_bool:
            playback_snippets_dir = Path(os_base_path) / self.exp_id / 'usv_playback_experiments' / f"{prefix}_usv_playback_snippets"
            output_file_dir = Path(os_base_path) / self.exp_id / 'usv_playback_experiments' / 'naturalistic_usv_playback_files'
        else:
            playback_snippets_dir = Path('/mnt/cup/labs/falkner') / self.exp_id / 'usv_playback_experiments' / f"{prefix}_usv_playback_snippets"
            output_file_dir = Path('/mnt/cup/labs/falkner') / self.exp_id / 'usv_playback_experiments' / 'naturalistic_usv_playback_files'

        output_file_dir.mkdir(parents=True, exist_ok=True)

        wav_sampling_rate = self.create_playback_settings_dict['naturalistic_wav_sampling_rate']
        total_acceptable_playback_time = self.create_playback_settings_dict['total_acceptable_naturalistic_playback_time']

        gmm = _GMM_PARAMS['female'] if 'female' in prefix.lower() else _GMM_PARAMS['male']
        rng = np.random.default_rng()

        for _ in range(self.create_playback_settings_dict['num_naturalistic_usv_files']):
            smart_wait(app_context_bool=self.app_context_bool, seconds=1)
            current_time = datetime.today().strftime('%Y%m%d_%H%M%S')

            wav_files_list = sorted(playback_snippets_dir.glob('*.wav'))
            replay_wav_arr = np.array(object=[], dtype=np.int16)

            total_playback_time_created = 0
            last_time_updated = 0  # Variable to track progress for tqdm

            replay_txt_path = output_file_dir / f"{prefix}_usv_playback_{total_acceptable_playback_time}s_{current_time}_spacing.txt"
            usv_id_txt_path = output_file_dir / f"{prefix}_usv_playback_{total_acceptable_playback_time}s_{current_time}_usvids.txt"

            # This ensures files are opened only once and not overwritten.
            with (replay_txt_path.open('w+') as replay_txt_file,
                  usv_id_txt_path.open('w+') as usv_id_txt_file,
                  tqdm(total=total_acceptable_playback_time, desc="Generating Playback", unit="s") as pbar):

                while total_playback_time_created < total_acceptable_playback_time:
                    # inter-sequence interval: sample from third GMM component in log space
                    isi = np.exp(rng.normal(gmm['means'][2], gmm['sds'][2]))
                    isi_samples = int(np.ceil(isi * wav_sampling_rate * 1e3))

                    if total_playback_time_created == 0:
                        replay_wav_arr = np.zeros(isi_samples, dtype=np.int16)
                        usv_id_txt_file.write('ISI \n')
                    else:
                        replay_wav_arr = np.concatenate((replay_wav_arr, np.zeros(isi_samples, dtype=np.int16)))
                        usv_id_txt_file.write('ISI \n')

                    total_playback_time_created += isi
                    replay_txt_file.write(f'{isi_samples} \n')

                    # sequence length: Gaussian(13, 5) clipped to [3, 23]
                    usv_seq_length = int(np.clip(round(rng.normal(13, 5)), 3, 23))
                    for usv_idx in range(usv_seq_length):
                        # pick USV file
                        random_wav_file = random.choice(wav_files_list)
                        _, random_wav_file_data = wavfile.read(random_wav_file)
                        total_playback_time_created += (random_wav_file_data.shape[0] / (wav_sampling_rate * 1e3))

                        replay_txt_file.write(f'{random_wav_file_data.shape[0]} \n')
                        usv_id_txt_file.write(f'{random_wav_file.name} \n')

                        if usv_idx < (usv_seq_length - 1):
                            # inter-USV interval: sample from first GMM component in log space
                            iui = np.exp(rng.normal(gmm['means'][0], gmm['sds'][0]))
                            iui_samples = int(np.ceil(iui * wav_sampling_rate * 1e3))
                            total_playback_time_created += iui

                            replay_wav_arr = np.concatenate((replay_wav_arr, random_wav_file_data, np.zeros(iui_samples, dtype=np.int16)))
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

            target_samples = int(total_acceptable_playback_time * wav_sampling_rate * 1e3)
            replay_wav_arr = replay_wav_arr[:target_samples]

            actual_total_time_sec = int(np.ceil(replay_wav_arr.shape[0] / (wav_sampling_rate * 1e3)))
            self.message_output(f"The total duration of the generated naturalistic playback file is {round(actual_total_time_sec / 60, 2)} min.")

            wavfile.write(filename=output_file_dir / f"{prefix}_usv_playback_{total_acceptable_playback_time}s_{current_time}.wav",
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
            playback_snippets_dir = Path(os_base_path) / self.exp_id / 'usv_playback_experiments' / self.create_playback_settings_dict['playback_snippets_dir']
            output_file_dir = Path(os_base_path) / self.exp_id / 'usv_playback_experiments' / 'usv_playback_files'
        else:
            playback_snippets_dir = Path('/mnt/cup/labs/falkner') / self.exp_id / 'usv_playback_experiments' / self.create_playback_settings_dict['playback_snippets_dir']
            output_file_dir = Path('/mnt/cup/labs/falkner') / self.exp_id / 'usv_playback_experiments' / 'usv_playback_files'

        output_file_dir.mkdir(parents=True, exist_ok=True)
        ipi_duration = self.create_playback_settings_dict['ipi_duration']
        wav_sampling_rate = self.create_playback_settings_dict['wav_sampling_rate']
        total_usv_number = self.create_playback_settings_dict['total_usv_number']

        for _ in range(self.create_playback_settings_dict['num_usv_files']):

            smart_wait(app_context_bool=self.app_context_bool, seconds=1)
            current_time = datetime.today().strftime('%Y%m%d_%H%M%S')

            wav_files_list = sorted(playback_snippets_dir.glob('*.wav'))

            ipi_duration_samples = int(np.ceil(ipi_duration * wav_sampling_rate * 1e3))

            arr_start_with_ipi = np.zeros(ipi_duration_samples).astype(np.int16)
            replay_wav_arr = arr_start_with_ipi.copy()

            with (open(output_file_dir / f"usv_playback_n={total_usv_number}_{current_time}_spacing.txt",
                       'w+') as replay_txt_file,
                  open(output_file_dir / f"usv_playback_n={total_usv_number}_{current_time}_usvids.txt",
                       'w+') as usv_id_txt_file):
                replay_txt_file.write(f'{ipi_duration_samples} \n')
                for usv_num in tqdm(range(total_usv_number)):
                    random_wav_file = random.choice(wav_files_list)
                    _, random_wav_file_data = wavfile.read(random_wav_file)
                    replay_wav_arr = np.concatenate((replay_wav_arr, random_wav_file_data, arr_start_with_ipi))
                    replay_txt_file.write(f'{random_wav_file_data.shape[0]} \n')
                    replay_txt_file.write(f'{ipi_duration_samples} \n')
                    usv_id_txt_file.write(f'{random_wav_file.name} \n')

            self.message_output(f"The total duration of the generated playback file is {round(replay_wav_arr.shape[0] / (wav_sampling_rate * 1e3) / 60, 2)} min.")

            wavfile.write(filename=output_file_dir / f"usv_playback_n={total_usv_number}_{current_time}.wav",
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

        audio_file_loc = list((Path(self.root_directory) / 'audio' / audio_dir).glob(f"*{device_id}_*_ch{channel_id:02d}_*.wav"))
        output_dir = Path(self.root_directory) / 'audio' / 'frequency_shifted_audio_segments'
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file_name = f"{audio_file_loc[0].name}_start={seq_start}s_duration={seq_duration}s_octave_shift={octave_shift}"

        if len(audio_file_loc) == 1:

            # load audio sequence with Librosa
            original_audio, original_sr = librosa.load(
                audio_file_loc[0],
                sr=int(wav_sampling_rate * 1e3),
                offset=seq_start,
                duration=seq_duration
            )

            # calculate the new sample rate for resampling (changes pitch AND speed)
            new_sr = int(original_sr * (2.0 ** octave_shift))

            # intermediate filenames for the processing pipeline
            temp_resampled_file = output_dir / f"{output_file_name}_temp_resampled.wav"
            temp_audible_file = output_dir / f"{output_file_name}_temp_audible.wav"
            temp_denoised_file = output_dir / f"{output_file_name}_temp_denoised.wav"
            final_output_file = output_dir / f"{output_file_name}_audible_denoised_tempo_adjusted.wav"

            # export the resampled audio (this is the pitch/speed shift)
            sf.write(temp_resampled_file, original_audio, new_sr)

            # perform volume adjustment with SoX (if needed)
            if volume_adjustment:
                subprocess.Popen(args=f'''{self.command_addition}static_sox {temp_resampled_file} {temp_audible_file} compand 0.3,1 6:-70,-60,-20 -5 -90 0.2''',
                                 cwd=output_dir,
                                 stdout=subprocess.DEVNULL,
                                 stderr=subprocess.STDOUT,
                                 shell=self.shell_usage_bool).wait()

                processed_audio, _ = librosa.load(temp_audible_file, sr=new_sr)
            else:
                processed_audio, _ = librosa.load(temp_resampled_file, sr=new_sr)

            # perform noise reduction
            reduced_noise = nr.reduce_noise(y=processed_audio, sr=new_sr, stationary=True, n_std_thresh_stationary=3)
            sf.write(temp_denoised_file, reduced_noise, new_sr)

            # correct the tempo back to the original duration using SoX
            tempo_adjustment_factor = original_sr / new_sr

            if 'filtered' not in audio_dir:
                upper_cutoff_freq = int(np.ceil(25000 / (2 ** abs(octave_shift))))
                subprocess.Popen(args=f'''{self.command_addition}static_sox {temp_denoised_file} {final_output_file} sinc {upper_cutoff_freq}-0 tempo -s {tempo_adjustment_factor}''',
                                 cwd=output_dir,
                                 stdout=subprocess.DEVNULL,
                                 stderr=subprocess.STDOUT,
                                 shell=self.shell_usage_bool).wait()
            else:
                subprocess.Popen(args=f'''{self.command_addition}static_sox {temp_denoised_file} {final_output_file} tempo -s {tempo_adjustment_factor}''',
                                 cwd=output_dir,
                                 stdout=subprocess.DEVNULL,
                                 stderr=subprocess.STDOUT,
                                 shell=self.shell_usage_bool).wait()

            temp_resampled_file.unlink()
            if volume_adjustment:
                temp_audible_file.unlink()
            temp_denoised_file.unlink()

        else:
            self.message_output(f"Requested audio file not found. Please try again.")
