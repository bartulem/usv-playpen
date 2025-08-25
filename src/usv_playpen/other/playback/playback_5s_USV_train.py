import glob
import numpy as np
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

    # setup Raspberry Pi GPIO pin
    io.setmode(io.BCM)
    io.setup(io_pin_id, io.OUT)

    # find .wav files
    wav_file_lst = glob.glob(f"{audio_d}/*.wav")

    # get a pseudo-random sequence of audio files
    random_int_lst = np.random.randint(low=0, high=100, size=num_seq, dtype=int)

    for i in range(num_seq):

        # TTL change
        io.output(io_pin_id, io.LOW)

        # play audio file
        subprocess.Popen(args=f'''AUDIODEV=hw:3,0 play -r 192k -v {audio_playback_gain} {wav_file_lst[random_int_lst[i]]}''',
                         shell=True,
                         cwd=audio_directory).wait()

        # TTL change
        io.output(io_pin_id, io.HIGH)

        # wait for 10 seconds
        time.sleep(10)


if __name__ == "__main__":
    run_playback_files(num_seq = usv_sequence_n,
                       io_pin_id=io_pin,
                       audio_directory=audio_d,
                       audio_playback_gain=playback_gain)
