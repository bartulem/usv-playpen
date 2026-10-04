"""
@author: bartulem
Builds linearly-interpolated matplotlib colormaps between user-specified RGB anchors.

This module constructs sequential (two-anchor) and diverging (three-anchor) colormaps
by linearly interpolating each RGB channel with ``numpy.linspace`` between the supplied
start, end and (for diverging maps) opposite-start RGB colors, with optional HLS
luminance and saturation equalization of the spectrum ends. It also provides a helper
for selecting per-animal colors by sex, and one that outlines every category of an
integer label grid (e.g. the QLVM category bundle's ``label_grid``) with uniform-width
lines.
"""

from __future__ import annotations

import colorsys

from typing import Any

import numpy as np
from matplotlib.colors import ListedColormap


def choose_animal_colors(
    exp_info_dict: dict | None = None, visualizations_parameter_dict: dict | None = None
) -> list:
    """
    Description
    -----------
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
    luminance: bool | float | None = False,
    match_by: str | None = None,
    saturation: bool | float | None = False,
) -> tuple | None:
    """
    Description
    -----------
    This function equalizes input colors on luminance.

    Parameters
    ----------
    color_start (tuple)
        RGB of spectrum start color.
    color_end (tuple)
        RGB of spectrum end color.
    luminance (bool / float)
        Equalizes luminance of spectrum ends; a float value both enables
        equalization and, when match_by='set', is used directly as the
        target luminance applied to both spectrum ends.
    match_by (str)
        Match luminance by 'max', 'min', 'mean' or 'set'; required (no
        default) whenever luminance equalization is requested.
    saturation (bool / float)
        Change saturation of spectrum ends; only a float value takes effect
        (it sets both ends to that saturation), while any non-float (e.g. the
        default False) leaves the original saturation unchanged.

    Returns
    -------
    color_start, color_end (tuple)
        Modified start and end colors to match luminance.
    """

    # extract hue, luminance and saturation for start and end colors
    hls_start = colorsys.rgb_to_hls(
        r=color_start[0] / 255.0, g=color_start[1] / 255.0, b=color_start[2] / 255.0
    )
    hls_end = colorsys.rgb_to_hls(
        r=color_end[0] / 255.0, g=color_end[1] / 255.0, b=color_end[2] / 255.0
    )

    # match luminance
    if luminance is True or isinstance(luminance, float):
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
            msg = (
                f"unrecognized luminance matching approach {match_by!r}; "
                "expected 'min' / 'max' / 'mean' / 'set'"
            )
            raise ValueError(msg)
    else:
        luminance_start = hls_start[1]
        luminance_end = hls_end[1]

    if isinstance(saturation, float):
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
    -----------
    This function creates colormap(s) of choice.

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

    Returns
    -------
    new_cm (matplotlib.colors.ListedColormap)
        A colormap object.
    """

    if input_parameter_dict is None:
        msg = "create_colormap requires an input_parameter_dict (got None)."
        raise ValueError(msg)

    # change luminance|saturation
    if (
        input_parameter_dict["equalize_luminance"] is True
        or isinstance(input_parameter_dict["equalize_luminance"], float)
        or isinstance(input_parameter_dict["change_saturation"], float)
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
                    # length must equal the target slice `[cm_length//2:]` length,
                    # i.e. cm_length - cm_length//2 (== cm_length//2 + 1 for odd
                    # cm_length, but one shorter for even cm_length, which the old
                    # `cm_length//2 + 1` over-counted into a broadcast error).
                    input_parameter_dict["cm_length"]
                    - input_parameter_dict["cm_length"] // 2,
                )
            )
    else:
        cm_type = input_parameter_dict["cm_type"]
        msg = (
            f"unrecognized cm_type {cm_type!r}; expected "
            "'sequential' or 'diverging'"
        )
        raise ValueError(msg)
    cm_values[:, 3] = (
        np.ones(input_parameter_dict["cm_length"]) * input_parameter_dict["cm_opacity"]
    )
    new_cm = ListedColormap(colors=cm_values, name=input_parameter_dict["cm_name"])

    return new_cm


def draw_category_outlines(
    ax: Any,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    label_grid: np.ndarray,
    colors: str,
    linewidths: float,
    zorder: float,
) -> list:
    """
    Description
    -----------
    Outlines EACH category of an integer label grid as the 0.5 contour of its own
    binary mask (``label_grid == category``), the method the embedding explorer, the
    manifold filter atlas and the torus-traversal video use for the QLVM category
    bundle. Contouring the integer grid itself at half-integer levels instead stacks
    several iso-lines wherever two non-consecutive categories touch (every level in
    between crosses there), so such borders render thicker than the rest; one 0.5
    contour per category puts every shared border at exactly one position, so the
    line is the same width everywhere (each shared border is drawn by the two
    categories it separates, on the same path). Pixels that are NaN (a float grid
    with unlabelled pixels) belong to no category.

    Parameters
    ----------
    ax (matplotlib.axes.Axes)
        Axes to draw the outlines on.
    x_axis (np.ndarray)
        1-D pixel-centre coordinates of the grid's columns (``label_grid[:, j]``).
    y_axis (np.ndarray)
        1-D pixel-centre coordinates of the grid's rows (``label_grid[i, :]``); the
        grid is indexed ``[y, x]``.
    label_grid (np.ndarray)
        2-D grid of category labels (integers, or floats with NaN for unlabelled
        pixels), shape ``(len(y_axis), len(x_axis))``.
    colors (str)
        Hex colour of the outlines.
    linewidths (float)
        Outline width in points.
    zorder (float)
        Drawing order of the outlines.

    Returns
    -------
    contour_sets (list[matplotlib.contour.QuadContourSet])
        One contour set per category present in the grid, in ascending label order.
    """

    labels = np.asarray(label_grid, dtype=float)
    present_labels = [label for label in np.unique(labels) if not np.isnan(label)]
    contour_sets = []
    for label in present_labels:
        category_mask = np.where(np.isnan(labels), 0.0, (labels == label).astype(float))
        contour_sets.append(
            ax.contour(x_axis, y_axis, category_mask, levels=[0.5], colors=colors, linewidths=linewidths, zorder=zorder)
        )
    return contour_sets
