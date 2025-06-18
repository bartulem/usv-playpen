"""
@author: bartulem
Creates perceptually uniform colormaps (per Crameri, F. et al., Nat. Commun. (2020))
"""

from __future__ import annotations

import colorsys
import json
import sys

import numpy as np
from matplotlib.colors import ListedColormap


def choose_animal_colors(
    exp_info_dict: dict | None = None, visualizations_parameter_dict: dict | None = None
) -> list | None:
    """
    Selects colors for male and female mice.

    Parameters
    ----------
    exp_info_dict (dict)
        Information about the experiment.
    visualizations_parameter_dict (dict)
        Information about the male/female color scheme.

    Returns
    -------
    mouse_colors (list)
        Chosen mouse colors in sequence.
    """

    mouse_colors = []
    n_males = 0
    n_females = 0
    for sex_idx, sex in enumerate(exp_info_dict["mouse_sex"]):
        if sex == "male":
            mouse_colors.append(visualizations_parameter_dict["male_colors"][n_males])
            n_males += 1
        else:
            mouse_colors.append(
                visualizations_parameter_dict["female_colors"][n_females]
            )
            n_females += 1

    return mouse_colors


def luminance_equalizer(
    color_start: tuple | None = None,
    color_end: tuple | None = None,
    luminance: bool | None = False,
    match_by: str | None = None,
    saturation: bool | None = False,
) -> tuple | None:
    """
    Description
    ----------
    This function equalizes input colors on luminance.
    ----------

    Parameters
    ----------
    color_start (tuple)
        RGB of spectrum start color.
    color_end (tuple)
        RGB of spectrum end color.
    luminance (bool / float)
        Equalizes luminance of spectrum ends.
    match_by (str)
        Match luminance by 'max', 'min' or 'mean'; defaults to 'max'.
    saturation (bool / float)
        Change saturation of spectrum ends.
    ----------

    Returns
    ----------
    color_start, color_end (tuple)
        Modified start and end colors to match luminance.
    ----------
    """

    # extract hue, luminance and saturation for start and end colors
    hls_start = colorsys.rgb_to_hls(
        r=color_start[0] / 255.0, g=color_start[1] / 255.0, b=color_start[2] / 255.0
    )
    hls_end = colorsys.rgb_to_hls(
        r=color_end[0] / 255.0, g=color_end[1] / 255.0, b=color_end[2] / 255.0
    )

    # match luminance
    if luminance is True or type(luminance) == float:
        if match_by == "max":
            luminance_start, luminance_end = np.repeat(
                np.max([hls_start[1], hls_end[1]]), 2
            )
        elif match_by == "min":
            luminance_start, luminance_end = np.repeat(
                np.min([hls_start[1], hls_end[1]]), 2
            )
        elif match_by == "mean":
            luminance_start, luminance_end = np.repeat(
                np.mean([hls_start[1], hls_end[1]]), 2
            )
        elif match_by == "set":
            luminance_start, luminance_end = [float(luminance), float(luminance)]
        else:
            print("Do not recognize luminance matching approach, try again!")
            sys.exit()
    else:
        luminance_start = hls_start[1]
        luminance_end = hls_end[1]

    if type(saturation) == float:
        saturation_start = saturation
        saturation_end = saturation
    else:
        saturation_start = hls_start[2]
        saturation_end = hls_end[2]

    # convert back to RGB
    color_start = tuple(
        item * 255.0
        for item in colorsys.hls_to_rgb(hls_start[0], luminance_start, saturation_start)
    )
    color_end = tuple(
        item * 255.0
        for item in colorsys.hls_to_rgb(hls_end[0], luminance_end, saturation_end)
    )

    return color_start, color_end


def create_colormap(input_parameter_dict: dict | None = None) -> ListedColormap:
    """
    Description
    ----------
    This function creates colormap(s) of choice.
    ----------

    Parameters
    ----------
    input_parameter_dict (dict)
        Contains the following set of parameters
        cm_length (int)
            Length of colormap; defaults to 255.
        cm_name (str)
            The name of the new colormap; defaults to 'red_green'.
        cm_type (str)
            Colormap type; defaults to 'sequential'.
        cm_start (tuple)
            RGB start of the colormap; defaults to red (255, 0, 0).
        cm_start_div (tuple)
            RGB start of the opposite side in a diverging colormap;
            defaults to green (0, 255, 0).
        cm_end (tuple)
            RGB end of the colormap; defaults to white (255, 255, 255).
        equalize_luminance (bool / float)
            Match luminance at both ends of the color spectrum; defaults to True.
        match_luminance_by (str)
            Match luminance by 'max', 'min', 'mean' or 'set'; defaults to 'max'.
        change_saturation (int / float)
            Saturation of color(s) at the end of spectrum; defaults to 1.
        cm_opacity (int / float)
            Opacity for colors in the new colormap; defaults to 1.
    ----------

    Returns
    ----------
    new_cm (matplotlib.colors.ListedColormap)
        A colormap object.
    ----------
    """

    # load .json parameters
    if input_parameter_dict is None:
        with open("../input_parameters.json") as json_file:
            input_parameter_dict = json.load(json_file)["auxiliary_plot_functions"][
                "create_colormap"
            ]

    # change luminance|saturation
    if (
        input_parameter_dict["equalize_luminance"] is True
        or type(input_parameter_dict["equalize_luminance"]) == float
        or type(input_parameter_dict["change_saturation"]) == float
    ):
        if input_parameter_dict["cm_type"] == "diverging":
            input_parameter_dict["cm_start"], input_parameter_dict["cm_start_div"] = (
                luminance_equalizer(
                    tuple(input_parameter_dict["cm_start"]),
                    tuple(input_parameter_dict["cm_start_div"]),
                    luminance=input_parameter_dict["equalize_luminance"],
                    match_by=input_parameter_dict["match_luminance_by"],
                    saturation=input_parameter_dict["change_saturation"],
                )
            )
        elif input_parameter_dict["cm_type"] == "sequential" and tuple(
            input_parameter_dict["cm_end"]
        ) != (255, 255, 255):
            input_parameter_dict["cm_start"], input_parameter_dict["cm_end"] = (
                luminance_equalizer(
                    tuple(input_parameter_dict["cm_start"]),
                    tuple(input_parameter_dict["cm_end"]),
                    luminance=input_parameter_dict["equalize_luminance"],
                    match_by=input_parameter_dict["match_luminance_by"],
                    saturation=input_parameter_dict["change_saturation"],
                )
            )

    # create colormap
    cm_values = np.ones((input_parameter_dict["cm_length"], 4))
    if input_parameter_dict["cm_type"] == "sequential":
        for rgb in range(3):
            cm_values[:, rgb] = np.linspace(
                input_parameter_dict["cm_end"][rgb] / input_parameter_dict["cm_length"],
                input_parameter_dict["cm_start"][rgb]
                / input_parameter_dict["cm_length"],
                input_parameter_dict["cm_length"],
            )
    elif input_parameter_dict["cm_type"] == "diverging":
        for rgb in range(3):
            cm_values[: input_parameter_dict["cm_length"] // 2 + 1, rgb] = np.linspace(
                input_parameter_dict["cm_start"][rgb]
                / input_parameter_dict["cm_length"],
                input_parameter_dict["cm_end"][rgb] / input_parameter_dict["cm_length"],
                input_parameter_dict["cm_length"] // 2 + 1,
            )
            cm_values[input_parameter_dict["cm_length"] // 2 :, rgb] = np.flip(
                np.linspace(
                    input_parameter_dict["cm_start_div"][rgb]
                    / input_parameter_dict["cm_length"],
                    input_parameter_dict["cm_end"][rgb]
                    / input_parameter_dict["cm_length"],
                    input_parameter_dict["cm_length"] // 2 + 1,
                )
            )
    else:
        print("Do not recognize cm_type, try again!")
        sys.exit()
    cm_values[:, 3] = (
        np.ones(input_parameter_dict["cm_length"]) * input_parameter_dict["cm_opacity"]
    )
    new_cm = ListedColormap(colors=cm_values, name=input_parameter_dict["cm_name"])

    return new_cm
