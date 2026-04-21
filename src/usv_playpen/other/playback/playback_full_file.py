import os

import RPi.GPIO as io
import subprocess

io.setwarnings(False)

# hyperparameters
io_pin = 14
playback_gain = 3
audio_d = '/home/usv-replay/Music'
audio_f = 'usv_playback_n=10000_20241020_174951_192kHz.wav'


def run_speaker_file(io_pin_id,
                     audio_directory,
                     audio_file,
                     audio_playback_gain):
    """
    Description
    ----------
    Plays back a single audio file through the configured ALSA device at
    192 kHz while driving a TTL sync line low for the duration of playback
    (raised back high afterwards). Intended to be run on the USV playback
    Raspberry Pi.
    ----------

    Parameters
    ----------
    io_pin_id (int)
        BCM-numbered GPIO pin used to emit the TTL sync pulse.
    audio_directory (str)
        Directory containing the WAV file to play.
    audio_file (str)
        Name of the WAV file to play (resolved relative to audio_directory).
    audio_playback_gain (float or int)
        Linear gain applied by 'play' via the -v flag.
    ----------

    Returns
    -------
    (None)
    -------
    """

    # setup Raspberry Pi GPIO pin
    io.setmode(io.BCM)
    io.setup(io_pin_id, io.OUT)

    # TTL change
    io.output(io_pin_id, io.LOW)

    # play audio file — use an argv list and pass AUDIODEV via env= rather
    # than interpolating into a shell string, so a pathological file name
    # cannot be injected as a shell command.
    play_env = os.environ.copy()
    play_env["AUDIODEV"] = "hw:3,0"
    subprocess.Popen(
        args=["play", "-r", "192k", "-v", str(audio_playback_gain), str(audio_file)],
        shell=False,
        cwd=audio_directory,
        env=play_env,
    ).wait()

    # TTL change
    io.output(io_pin_id, io.HIGH)


if __name__ == "__main__":
    run_speaker_file(io_pin_id=io_pin,
                     audio_directory=audio_d,
                     audio_file=audio_f,
                     audio_playback_gain=playback_gain)
