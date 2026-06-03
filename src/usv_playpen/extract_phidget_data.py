"""
@author: bartulem
Code to extract data measured by phidgets.
"""

from __future__ import annotations

import json
import pathlib
from operator import itemgetter

import numpy as np


class Gatherer:
    """
    Description
    -----------
    Reads the phidget sensor logs (illumination, temperature, humidity) recorded
    alongside a session's video and returns them as aligned per-sample arrays.
    """

    def __init__(
        self, input_parameter_dict: dict | None = None, root_directory: str | None = None
    ) -> None:
        """
        Description
        -----------
        Initializes the Gatherer class.

        Parameters
        ----------
        root_directory (str)
            Root directory for data; defaults to None.
        input_parameter_dict (dict)
           Processing parameters; defaults to None.

        Returns
        -------
        None
        """

        if root_directory is None or input_parameter_dict is None:
            with open(
                pathlib.Path(__file__).parent / "_parameter_settings/processing_settings.json"
            ) as json_file:
                _settings = json.load(json_file)["extract_phidget_data"]

        self.root_directory = root_directory if root_directory is not None else _settings["root_directory"]
        self.input_parameter_dict = (
            input_parameter_dict["extract_phidget_data"]["Gatherer"]
            if input_parameter_dict is not None
            else _settings["Gatherer"]
        )

    def prepare_data_for_analyses(self) -> dict:
        """
        Description
        -----------
        This method extracts phidget-measured atmospheric data:
        (1) the amount of illumination (lux)
        (2) temperature (degrees Celsius)
        (3) humidity (%)

        NB: Phidgets' sampling rate is ~1 Hz!

        Parameters
        ----------
        None

        Returns
        -------
        phidget_data_dictionary (dict)
            Contains lux, humidity and temperature data.

        Raises
        ------
        FileNotFoundError
            If the video directory, the requested camera sub-directory, or the
            phidget '*extra_data*.json' files within it are missing.
        """

        # Find the camera sub-directory holding the phidget data. Iterate sorted
        # for a deterministic choice (was iterdir-order-dependent), skip non-dirs,
        # and raise a clear error when none matches (was a None.glob crash).
        video_dir = pathlib.Path(self.root_directory) / "video"
        if not video_dir.exists():
            raise FileNotFoundError(f"Video directory not found: '{video_dir}'.")
        extra_data_camera = self.input_parameter_dict["prepare_data_for_analyses"]["extra_data_camera"]
        sub_directory = next(
            (
                one_dir
                for one_dir in sorted(video_dir.iterdir())
                if one_dir.is_dir() and extra_data_camera in one_dir.name
            ),
            None,
        )
        if sub_directory is None:
            raise FileNotFoundError(
                f"No camera directory containing '{extra_data_camera}' found under '{video_dir}'."
            )

        # Load raw phidget data. The files are named '<n>.extra_data.json' and
        # each holds a list of ~1 Hz sensor records; glob specifically for them so
        # a stray .json in the camera directory is never mistaken for phidget data.
        # One loop covers the single- and multi-file cases and raises on an empty
        # set (was an IndexError on phidget_file_list[0]).
        phidget_file_list = sorted(sub_directory.glob("*extra_data*.json"))
        if not phidget_file_list:
            raise FileNotFoundError(
                f"No phidget '*extra_data*.json' files found in '{sub_directory}'."
            )
        phidget_data = []
        for one_phidget_file in phidget_file_list:
            with open(one_phidget_file) as phidget_file:
                phidget_data += json.load(phidget_file)

        # sort phidget_data by particular dictionary key
        phidget_data_sorted = sorted(
            phidget_data, key=itemgetter("sensor_time")
        )

        # extract data for export
        phidget_data_dictionary = {
            "humidity": np.full((len(phidget_data_sorted),), np.nan),
            "lux": np.full((len(phidget_data_sorted),), np.nan),
            "temperature": np.full((len(phidget_data_sorted),), np.nan),
        }

        for one_dict_idx, one_dict in enumerate(phidget_data_sorted):
            if "hum_h" in one_dict:
                phidget_data_dictionary["humidity"][one_dict_idx] = one_dict["hum_h"]
            if "lux" in one_dict:
                phidget_data_dictionary["lux"][one_dict_idx] = one_dict["lux"]
            if "hum_t" in one_dict:
                phidget_data_dictionary["temperature"][one_dict_idx] = one_dict["hum_t"]

        return phidget_data_dictionary
