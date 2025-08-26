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

    # setup Raspberry Pi GPIO pin
    io.setmode(io.BCM)
    io.setup(io_pin_id, io.OUT)

    # TTL change
    io.output(io_pin_id, io.LOW)

    # play audio file
    subprocess.Popen(args=f'''AUDIODEV=hw:3,0 play -r 192k -v {audio_playback_gain} {audio_file}''',
                     shell=True,
                     cwd=audio_directory).wait()

    # TTL change
    io.output(io_pin_id, io.HIGH)


if __name__ == "__main__":
    run_speaker_file(io_pin_id=io_pin,
                     audio_directory=audio_d,
                     audio_file=audio_f,
                     audio_playback_gain=playback_gain)
