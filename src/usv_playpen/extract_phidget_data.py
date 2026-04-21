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
    def __init__(
        self, input_parameter_dict: dict = None, root_directory: str = None
    ) -> None:
        """
        Initializes the Gatherer class.

        Parameters
        ----------
        root_directory (str)
            Root directory for data; defaults to None.
        input_parameter_dict (dict)
           Processing parameters; defaults to None.

        Returns
        -------
        -------
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
        ----------
        This method extracts phidget-measured atmospheric data:
        (1) the amount of illumination (lux)
        (2) temperature (degrees Celsius)
        (3) humidity (%)

        NB: Phidgets' sampling rate is ~1 Hz!
        ----------

        Parameters
        ----------
        ----------

        Returns
        ----------
        phidget_data_dictionary (dict)
            Contains lux, humidity and temperature data.
        ----------
        """

        # find subdirectory with phidget data
        sub_directory = None
        for one_dir in (pathlib.Path(self.root_directory) / "video").iterdir():
            if (
                self.input_parameter_dict["prepare_data_for_analyses"][
                    "extra_data_camera"
                ]
                in one_dir.name
            ):
                sub_directory = one_dir
                break

        phidget_file_list = sorted(sub_directory.glob("*.json"))

        # load raw phidget data
        phidget_data = []
        if len(phidget_file_list) > 1:
            for one_phidget_file in phidget_file_list:
                with open(one_phidget_file) as phidget_file:
                    phidget_data += json.load(phidget_file)

        else:
            with open(phidget_file_list[0]) as phidget_file:
                phidget_data = json.load(phidget_file)

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
