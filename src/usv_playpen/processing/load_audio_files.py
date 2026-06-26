"""
@author: bartulem
Loads WAV files.
"""

from __future__ import annotations

import json
import pathlib
import struct
import subprocess
import warnings

import librosa
from scipy.io import wavfile


class DataLoader:
    def __init__(self, input_parameter_dict: dict | None = None) -> None:
        """
        Description
        -----------
        Initializes the DataLoader class.

        Parameters
        ----------
        input_parameter_dict (dict)
            Processing parameters; defaults to None.

        Returns
        -------
        None
        """

        # Maps a numpy array's dtype name (numpy.dtype.name) onto the canonical
        # string stored under the "dtype" sub-key. Only the bare numpy scalar
        # names that numpy.dtype.name can actually produce are listed here; the
        # WAV data returned by scipy/librosa never yields any other dtype name.
        self.known_dtypes = {
            "int8": "int8",
            "int16": "int16",
            "int32": "int32",
            "int64": "int64",
            "uint8": "uint8",
            "uint16": "uint16",
            "uint32": "uint32",
            "uint64": "uint64",
            "float16": "float16",
            "float32": "float32",
            "float64": "float64",
        }

        if input_parameter_dict is None:
            with open(
                pathlib.Path(__file__).parent.parent / "_parameter_settings/processing_settings.json"
            ) as json_file:
                self.input_parameter_dict = json.load(json_file)["load_audio_files"][
                    "DataLoader"
                ]
        else:
            self.input_parameter_dict = input_parameter_dict

    def load_wavefile_data(self) -> dict:
        """
        Description
        -----------
        This method loads the .wav file(s) of interest.

        Parameters
        ----------

        Returns
        -------
        wave_data_dict (dict)
            A dictionary with all desired sound outputs;
            the top-level key is the WAV file's name (one_file.name),
            with "sampling_rate", "wav_data" and "dtype" as sub-keys.
        """

        wave_data_dict = {}
        for one_dir in self.input_parameter_dict["wave_data_loc"]:
            for one_file in sorted(pathlib.Path(one_dir).iterdir(), key=lambda p: p.name):
                # additional conditional argument to reduce numbers of files loaded
                if (
                    len(
                        self.input_parameter_dict["load_wavefile_data"][
                            "conditional_arg"
                        ]
                    )
                    == 0
                ):
                    additional_condition = True
                else:
                    additional_condition = all(
                        cond in one_file.name
                        for cond in self.input_parameter_dict["load_wavefile_data"][
                            "conditional_arg"
                        ]
                    )

                if (
                    one_file.is_file()
                    and one_file.suffix.lower() == ".wav"
                    and additional_condition
                ):
                    wave_data_dict[one_file.name] = {
                        "sampling_rate": 0,
                        "wav_data": 0,
                        "dtype": 0,
                    }
                    if (
                        self.input_parameter_dict["load_wavefile_data"]["library"]
                        == "scipy"
                    ):
                        try:
                            with warnings.catch_warnings():
                                # scipy.io.wavfile.read emits a WavFileWarning on
                                # non-standard/extra header chunks; suppress it only
                                # around the read instead of disabling warnings
                                # process-wide for everything that runs afterwards.
                                warnings.simplefilter("ignore")
                                (
                                    wave_data_dict[one_file.name]["sampling_rate"],
                                    wave_data_dict[one_file.name]["wav_data"],
                                ) = wavfile.read(one_file)
                        except struct.error:
                            # The .wav header is malformed; try to rewrite it with sox.
                            # We do NOT delete the original until sox has successfully
                            # produced the corrected file, otherwise a sox failure
                            # (missing codec, path issue, crash) would permanently
                            # destroy the original recording.
                            correct_file = one_file.parent / f"{one_file.stem}_correct.wav"
                            sox_result = subprocess.run(
                                args=["static_sox", one_file.name, correct_file.name],
                                shell=False,
                                cwd=one_file.parent,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                check=False,
                                text=True,
                            )
                            if sox_result.returncode != 0 or not correct_file.is_file():
                                raise RuntimeError(
                                    f"sox failed to repair '{one_file}' "
                                    f"(return code {sox_result.returncode}); "
                                    f"sox output: {sox_result.stdout.strip() if sox_result.stdout else '<empty>'}. "
                                    f"Original file left untouched."
                                )
                            one_file.unlink()
                            correct_file.rename(one_file)
                            with warnings.catch_warnings():
                                # scipy.io.wavfile.read emits a WavFileWarning on
                                # non-standard/extra header chunks; suppress it only
                                # around the read instead of disabling warnings
                                # process-wide for everything that runs afterwards.
                                warnings.simplefilter("ignore")
                                (
                                    wave_data_dict[one_file.name]["sampling_rate"],
                                    wave_data_dict[one_file.name]["wav_data"],
                                ) = wavfile.read(one_file)
                    else:
                        with warnings.catch_warnings():
                            # librosa imports the stdlib `aifc` module, which is
                            # deprecated in Python 3.13; suppress that
                            # DeprecationWarning here rather than disabling warnings
                            # process-wide (the old module-level simplefilter).
                            warnings.simplefilter("ignore")
                            (
                                wave_data_dict[one_file.name]["wav_data"],
                                wave_data_dict[one_file.name]["sampling_rate"],
                            ) = librosa.load(one_file)
                    # Read the dtype name straight off the array (numpy.dtype.name)
                    # rather than materializing a scalar via .ravel()[0], which would
                    # raise IndexError on an empty/zero-sample WAV.
                    wave_data_dict[one_file.name]["dtype"] = self.known_dtypes[
                        wave_data_dict[one_file.name]["wav_data"].dtype.name
                    ]

        return wave_data_dict
