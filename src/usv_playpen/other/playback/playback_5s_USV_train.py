import os

import numpy as np
import pathlib
import RPi.GPIO as io
import subprocess
import time

io.setwarnings(False)

# hyperparameters
usv_sequence_n = 70
io_pin = 14
playback_gain = 5
audio_d = '/home/usv-replay/Music/shorts'


def run_playback_files(num_seq,
                       io_pin_id,
                       audio_directory,
                       audio_playback_gain):
    """
    Description
    Pseudo-randomly selects WAV files from a directory and plays them back
    through the configured ALSA device at 192 kHz. A TTL pulse is driven
    low on the given Raspberry Pi GPIO pin for the duration of each
    playback, then raised high; playback is spaced with a 10 second gap.
    Intended to be run on the USV playback Raspberry Pi.

    Parameters
    num_seq (int)
        Number of playback iterations to run.
    io_pin_id (int)
        BCM-numbered GPIO pin used to emit the TTL sync pulse.
    audio_directory (str)
        Directory containing the WAV files to sample from.
    audio_playback_gain (float or int)
        Linear gain applied by 'play' via the -v flag.

    Returns
    (None)
    """

    # setup Raspberry Pi GPIO pin
    io.setmode(io.BCM)
    io.setup(io_pin_id, io.OUT)

    # find .wav files
    wav_file_lst = sorted(pathlib.Path(audio_d).glob('*.wav'))
    if not wav_file_lst:
        raise FileNotFoundError(
            f"No .wav files found under '{audio_d}' — nothing to play back."
        )

    # get a pseudo-random sequence of audio files — keep the modulo so the
    # random index actually maps onto the discovered file list regardless of
    # how many .wav files are present.
    random_int_lst = np.random.randint(low=0, high=100, size=num_seq, dtype=int)

    play_env = os.environ.copy()
    play_env["AUDIODEV"] = "hw:3,0"

    for i in range(num_seq):

        # TTL change
        io.output(io_pin_id, io.LOW)

        wav_path = wav_file_lst[random_int_lst[i] % len(wav_file_lst)]

        # play audio file — argv list + env= so the path cannot be interpreted
        # as a shell command even if the file name contains spaces or shell
        # metacharacters.
        subprocess.Popen(
            args=["play", "-r", "192k", "-v", str(audio_playback_gain), str(wav_path)],
            shell=False,
            cwd=audio_directory,
            env=play_env,
        ).wait()

        # TTL change
        io.output(io_pin_id, io.HIGH)

        # wait for 10 seconds
        time.sleep(10)


if __name__ == "__main__":
    run_playback_files(num_seq = usv_sequence_n,
                       io_pin_id=io_pin,
                       audio_directory=audio_d,
                       audio_playback_gain=playback_gain)
