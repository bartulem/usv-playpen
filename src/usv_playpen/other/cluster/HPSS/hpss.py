"""
@author: bartulem
Runs HPSS on the cluster.
"""

from __future__ import annotations

import importlib.resources
import json
import pathlib
import sys

import librosa
import numpy as np
from scipy.io import wavfile

# The compute cluster mounts every lab share under this directory
# (``/mnt/cup/labs/<lab share>``, the cluster form ``os_utils.to_cluster_path``
# produces); the session folders this script reads and writes live below it. A
# fixed property of the cluster, not an analysis setting.
CLUSTER_LAB_SHARES_ROOT = "/mnt/cup/labs"


def hpss_func(
    recording_identifier: str,
    cup_recording_directory: str,
    wav_file_idx: int,
) -> None:
    """
    Description
    -----------
    This function performs the harmonic/percussive source separation (HPSS)
    on the provided audio (WAV) files. The harmonic component is then converted
    back to the time domain and saved as a new WAV file.

    Parameters
    ----------
    recording_identifier (str)
        Identifier for the recording session, e.g. "20230801_155857".
    cup_recording_directory (str)
        Lab-share subdirectory holding the session, relative to
        ``CLUSTER_LAB_SHARES_ROOT`` (e.g. "falkner/<experimenter>/Data").
    wav_file_idx (int)
        Index of the WAV file to be processed.

    Returns
    -------
    None
        The harmonic component is written to
        ``<CLUSTER_LAB_SHARES_ROOT>/<cup_recording_directory>/<recording_identifier>/audio/hpss/<wav stem>_hpss.wav``.
    """

    wav_dir = pathlib.Path(CLUSTER_LAB_SHARES_ROOT) / cup_recording_directory / recording_identifier / 'audio' / 'cropped_to_video'
    wav_files = sorted(wav_dir.glob('*.wav'))
    if wav_file_idx < 0 or wav_file_idx >= len(wav_files):
        raise IndexError(
            f"wav_file_idx={wav_file_idx} is out of range for "
            f"{len(wav_files)} .wav file(s) found under '{wav_dir}'."
        )
    wav_file = wav_files[wav_file_idx]

    # load the HPSS parameters from the shipped processing settings so this cluster
    # run stays identical to the local modify_files.hpss_audio implementation instead
    # of re-hard-coding the STFT window/hop, kernel size, power, and margin here.
    settings_traversable = importlib.resources.files('usv_playpen').joinpath(
        '_parameter_settings', 'processing_settings.json'
    )
    hpss_params = json.loads(settings_traversable.read_text())['modify_files']['Operator']['hpss_audio']
    stft_window_length, stft_hop_size = hpss_params['stft_window_length_hop_size']

    # read the audio file (use Scipy, not Librosa because Librosa performs scaling)
    sampling_rate_audio, audio_data = wavfile.read(wav_file)

    # convert to float32 because librosa.stft() requires float32
    audio_data = audio_data.astype(np.float32)

    # perform Short-Time Fourier Transform (STFT) on the audio data
    spectrogram_data = librosa.stft(
        y=audio_data, n_fft=stft_window_length, hop_length=stft_hop_size
    )

    # perform HPSS on the spectrogram data
    D_harmonic, D_percussive = librosa.decompose.hpss(
        S=spectrogram_data,
        kernel_size=hpss_params['kernel_size'],
        power=hpss_params['hpss_power'],
        mask=False,
        margin=hpss_params['margin'],
    )

    # convert the harmonic component back to the time domain
    harmonic_data = librosa.istft(
        stft_matrix=D_harmonic,
        length=audio_data.shape[0],
        win_length=stft_window_length,
        hop_length=stft_hop_size,
    )

    # ensure the float values are within the range of 16-bit integers. Clip values outside the range to the minimum and maximum representable values.
    harmonic_data_clipped = np.clip(
        harmonic_data, np.iinfo(np.int16).min, np.iinfo(np.int16).max
    ).astype(np.int16)

    # save the harmonic component as a new WAV file
    hpss_dir = pathlib.Path(CLUSTER_LAB_SHARES_ROOT) / cup_recording_directory / recording_identifier / 'audio' / 'hpss'
    hpss_dir.mkdir(parents=True, exist_ok=True)

    wavfile.write(
        filename=hpss_dir / f"{wav_file.stem}_hpss.wav",
        rate=sampling_rate_audio,
        data=harmonic_data_clipped,
    )


if __name__ == "__main__":
    hpss_func(str(sys.argv[1]), str(sys.argv[2]), int(sys.argv[3]))
