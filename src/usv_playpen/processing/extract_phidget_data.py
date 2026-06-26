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
    alongside a session's video and returns them as three arrays that are
    index-aligned by ascending acquisition timestamp ('sensor_time'). Each array
    holds NaN at any index where that sensor's value was absent from the
    corresponding record, so a given index is not guaranteed to carry all three
    sensor values.
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
        input_parameter_dict (dict)
            Processing parameters; defaults to None. When provided, it must
            contain the nesting ['extract_phidget_data']['Gatherer'] (with a
            'prepare_data_for_analyses' sub-dict providing 'extra_data_camera').
        root_directory (str)
            Root directory for data; defaults to None.

        Returns
        -------
        None
        """

        if root_directory is None or input_parameter_dict is None:
            with open(
                pathlib.Path(__file__).parent.parent / "_parameter_settings/processing_settings.json"
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
        KeyError
            If any loaded phidget record lacks the 'sensor_time' key, which is
            required on every record because the exported arrays are sorted and
            index-aligned by it (unlike the optional sensor keys hum_h/lux/hum_t,
            which are guarded with 'in' and default to NaN when absent).
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

        # Sort records by their acquisition timestamp (sensor_time) so the
        # exported arrays are chronological; multi-file loads above are
        # concatenated in filename order, not time order.
        phidget_data_sorted = sorted(
            phidget_data, key=itemgetter("sensor_time")
        )

        # extract data for export
        phidget_data_dictionary = {
            "humidity": np.full((len(phidget_data_sorted),), np.nan),
            "lux": np.full((len(phidget_data_sorted),), np.nan),
            "temperature": np.full((len(phidget_data_sorted),), np.nan),
        }

        # Map the raw record keys onto the exported sensors: 'hum_h' is humidity
        # (%) and 'hum_t' is temperature (degrees Celsius) from the same combined
        # humidity/temperature sensor (note 'hum_t' is temperature despite its
        # 'hum_' prefix), while 'lux' is illumination. Each key is filled
        # independently so absent keys leave the pre-allocated NaN in place.
        for one_dict_idx, one_dict in enumerate(phidget_data_sorted):
            if "hum_h" in one_dict:
                phidget_data_dictionary["humidity"][one_dict_idx] = one_dict["hum_h"]
            if "lux" in one_dict:
                phidget_data_dictionary["lux"][one_dict_idx] = one_dict["lux"]
            if "hum_t" in one_dict:
                phidget_data_dictionary["temperature"][one_dict_idx] = one_dict["hum_t"]

        return phidget_data_dictionary
