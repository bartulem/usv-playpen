"""
@author: bartulem
Runs HPSS on the cluster.
"""

import glob
import librosa
import numpy as np
import os
import pathlib
import sys
from scipy.io import wavfile


def hpss_func(recording_identifier: str = None,
              cup_recording_directory: str = None,
              wav_file_idx: int = None) -> None:
    """
    Description
    ----------
    This function performs the harmonic/percussive source separation (HPSS) 
    on the provided audio (WAV) files. The harmonic component is then converted
    back to the time domain and saved as a new WAV file.
    ----------

    Parameter
    ---------
    cup_recording_directory : str
        Personal file server subdirectory name, e.g. "Bartul".
    recording_identifier : str
        Identifier for the recording session, e.g. "20230801_155857".
    wav_file_idx : int
         Index of the WAV file to be processed.

    Returns
    -------
    harmonic_data_clipped : WAV file
        Output audio file w/ only the harmonics component.
    """

    wav_file = sorted(glob.glob(f'/mnt/cup/labs/falkner/{cup_recording_directory}/Data/{recording_identifier}/audio/cropped_to_video/*.wav'))[wav_file_idx]
    
    # read the audio file (use Scipy, not Librosa because Librosa performs scaling)
    sampling_rate_audio, audio_data = wavfile.read(wav_file)

    # convert to float32 because librosa.stft() requires float32
    audio_data = audio_data.astype(np.float32)
    
    # perform Short-Time Fourier Transform (STFT) on the audio data
    spectrogram_data = librosa.stft(y=audio_data, n_fft=512)

    # perform HPSS on the spectrogram data
    D_harmonic, D_percussive = librosa.decompose.hpss(S=spectrogram_data,
                                                      kernel_size=(5, 60),
                                                      power=4.0,
                                                      mask=False,
                                                      margin=(4, 1))
    
    # convert the harmonic component back to the time domain
    harmonic_data = librosa.istft(stft_matrix=D_harmonic, length=audio_data.shape[0], win_length=512, hop_length=128)
    
    # ensure the float values are within the range of 16-bit integers. Clip values outside the range to the minimum and maximum representable values.
    harmonic_data_clipped = np.clip(harmonic_data, np.iinfo(np.int16).min, np.iinfo(np.int16).max).astype(np.int16)

    # save the harmonic component as a new WAV file
    pathlib.Path(f'/mnt/cup/labs/falkner/{cup_recording_directory}/Data/{recording_identifier}/audio/hpss').mkdir(parents=True, exist_ok=True)
    
    raw_file_name = os.path.basename(wav_file)
    wavfile.write(filename=f'/mnt/cup/labs/falkner/{cup_recording_directory}/Data/{recording_identifier}/audio/hpss/{raw_file_name[:-4]}_hpss.wav', 
                  rate=sampling_rate_audio, 
                  data=harmonic_data_clipped)



if __name__ == '__main__':
        hpss_func(str(sys.argv[1]), str(sys.argv[2]), int(sys.argv[3]))
