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
        Initializes the DataLoader class.

        Parameter
        ---------
        input_parameter_dict (dict)
            Processing parameters; defaults to None.

        Returns
        -------
        -------
        """

        self.known_dtypes = {
            "int": int,
            "np.int8": "int8",
            "int8": "int8",
            "np.int16": "int16",
            "int16": "int16",
            "np.int32": "int32",
            "int32": "int32",
            "np.int64": "int64",
            "int64": "int64",
            "np.uint8": "uint8",
            "uint8": "uint8",
            "np.uint16": "uint16",
            "uint16": "uint16",
            "np.uint32": "uint32",
            "uint32": "uint32",
            "np.uint64": "uint64",
            "uint64": "uint64",
            "float": float,
            "np.float16": "float16",
            "float16": "float16",
            "np.float32": "float32",
            "float32": "float32",
            "np.float64": "float64",
            "float64": "float64",
            "str": str,
            "dict": dict,
        }

        if input_parameter_dict is None:
            with open(
                pathlib.Path(__file__).parent / "_parameter_settings/processing_settings.json"
            ) as json_file:
                self.input_parameter_dict = json.load(json_file)["load_audio_files"][
                    "DataLoader"
                ]
        else:
            self.input_parameter_dict = input_parameter_dict

    def load_wavefile_data(self) -> dict:
        """
        Description
        ----------
        This method loads the .wav file(s) of interest.
        ----------

        Parameters
        ----------
        ----------

        Returns
        ----------
        wave_data_dict (dict)
            A dictionary with all desired sound outputs;
            starting key in the dictionary is "session_id",
            with "sampling_rate", "wav_data" and "dtype" as sub-keys.
        ----------
        """

        # spits out warnings if .wav file has header, the line below suppresses it
        warnings.simplefilter("ignore")

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

                if ".wav" in one_file.name and additional_condition:
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
                            (
                                wave_data_dict[one_file.name]["sampling_rate"],
                                wave_data_dict[one_file.name]["wav_data"],
                            ) = wavfile.read(one_file)
                        except struct.error:
                            correct_file = one_file.parent / f"{one_file.stem}_correct.wav"
                            subprocess.run(
                                args=["static_sox", one_file.name, correct_file.name],
                                shell=False,
                                cwd=one_file.parent,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.STDOUT,
                                check=False,
                            )
                            one_file.unlink()
                            correct_file.rename(one_file)
                            (
                                wave_data_dict[one_file.name]["sampling_rate"],
                                wave_data_dict[one_file.name]["wav_data"],
                            ) = wavfile.read(one_file)
                    else:
                        (
                            wave_data_dict[one_file.name]["wav_data"],
                            wave_data_dict[one_file.name]["sampling_rate"],
                        ) = librosa.load(one_file)
                    wave_data_dict[one_file.name]["dtype"] = self.known_dtypes[
                        type(wave_data_dict[one_file.name]["wav_data"].ravel()[0]).__name__
                    ]

        return wave_data_dict
