"""
@author: bartulem
Unit tests for visualizations/auxiliary_plot_functions.py — the pure colour
helpers (`luminance_equalizer`, `create_colormap`) that back every per-mouse
colormap and luminance-matched palette in the figure suite.
"""

from __future__ import annotations

import colorsys

import numpy as np
import pytest
from matplotlib.colors import ListedColormap

from usv_playpen.visualizations.auxiliary_plot_functions import (
    luminance_equalizer,
    create_colormap,
)


@pytest.mark.parametrize("match_by", ["max", "min", "mean"])
def test_luminance_equalizer_matches_both_ends_to_same_luminance(match_by):
    """
    Description
    -----------
    With `luminance=True`, `luminance_equalizer` must drive both ends of the
    spectrum to a common HLS luminance computed by the requested rule (max /
    min / mean of the two inputs' luminances), leaving hue untouched.

    Parameters
    ----------
    match_by (str)
        Luminance-matching rule under test.

    Returns
    -------
    None
    """

    start = (200, 40, 40)   # reddish, higher luminance
    end = (20, 20, 80)      # dark blue, lower luminance
    out_start, out_end = luminance_equalizer(
        start, end, luminance=True, match_by=match_by, saturation=False,
    )
    lum_start = colorsys.rgb_to_hls(*[c / 255.0 for c in out_start])[1]
    lum_end = colorsys.rgb_to_hls(*[c / 255.0 for c in out_end])[1]
    assert lum_start == pytest.approx(lum_end, abs=1e-6), (
        f"both ends should share luminance under match_by={match_by!r}"
    )


def test_luminance_equalizer_set_uses_explicit_value():
    """
    Description
    -----------
    `match_by='set'` with a float `luminance` must pin both ends to exactly
    that luminance value.

    Parameters
    ----------

    Returns
    -------
    None
    """

    out_start, out_end = luminance_equalizer(
        (200, 40, 40), (20, 20, 80), luminance=0.5, match_by="set", saturation=False,
    )
    lum_start = colorsys.rgb_to_hls(*[c / 255.0 for c in out_start])[1]
    lum_end = colorsys.rgb_to_hls(*[c / 255.0 for c in out_end])[1]
    assert lum_start == pytest.approx(0.5, abs=1e-6)
    assert lum_end == pytest.approx(0.5, abs=1e-6)


def test_luminance_equalizer_unknown_match_by_raises():
    """
    Description
    -----------
    An unrecognised `match_by` (with luminance matching requested) must raise
    ValueError rather than silently passing colours through.

    Parameters
    ----------

    Returns
    -------
    None
    """

    with pytest.raises(ValueError, match="unrecognized luminance matching"):
        luminance_equalizer(
            (200, 40, 40), (20, 20, 80), luminance=True, match_by="median",
        )


def test_luminance_equalizer_saturation_float_overrides_both_ends():
    """
    Description
    -----------
    A float `saturation` must override the saturation of both ends while the
    no-luminance path leaves each end's luminance at its original value.

    Parameters
    ----------

    Returns
    -------
    None
    """

    out_start, out_end = luminance_equalizer(
        (200, 40, 40), (20, 20, 80), luminance=False, saturation=0.25,
    )
    sat_start = colorsys.rgb_to_hls(*[c / 255.0 for c in out_start])[2]
    sat_end = colorsys.rgb_to_hls(*[c / 255.0 for c in out_end])[2]
    assert sat_start == pytest.approx(0.25, abs=1e-6)
    assert sat_end == pytest.approx(0.25, abs=1e-6)


def _cm_params(**overrides) -> dict:
    """
    Description
    -----------
    Build a `create_colormap` parameter dict with sensible defaults that the
    caller can override per test.

    Parameters
    ----------
    **overrides
        Keys to override in the base parameter dict.

    Returns
    -------
    params (dict)
        The merged parameter dict.
    """

    params = {
        "cm_length": 64,
        "cm_name": "test_cm",
        "cm_type": "sequential",
        "cm_start": (255, 0, 0),
        "cm_start_div": (0, 255, 0),
        "cm_end": (255, 255, 255),
        "equalize_luminance": True,
        "match_luminance_by": "max",
        "change_saturation": 1,
        "cm_opacity": 1,
    }
    params.update(overrides)
    return params


def test_create_colormap_sequential_returns_listed_colormap():
    """
    Description
    -----------
    A sequential request must return a `ListedColormap` of the requested
    length; with a non-white `cm_end` the luminance-equalisation branch also
    runs.

    Parameters
    ----------

    Returns
    -------
    None
    """

    cm = create_colormap(_cm_params(cm_end=(10, 10, 90)))
    assert isinstance(cm, ListedColormap)
    assert cm.N == 64


def test_create_colormap_diverging_returns_listed_colormap():
    """
    Description
    -----------
    A diverging request must build a two-sided ramp (start -> end -> start_div)
    and return a `ListedColormap`, exercising the diverging luminance-equalise
    and fill branches.

    Parameters
    ----------

    Returns
    -------
    None
    """

    # The project default diverging length is the odd 255; both halves span
    # cm_length // 2 + 1 and cm_length - cm_length // 2 samples respectively.
    cm = create_colormap(_cm_params(cm_type="diverging", cm_length=65, change_saturation=0.5))
    assert isinstance(cm, ListedColormap)
    assert cm.N == 65


def test_create_colormap_diverging_even_length():
    """A diverging map with an EVEN length must build without a broadcast error.
    The second half's slice `[cm_length//2:]` has length cm_length - cm_length//2,
    which differs from cm_length//2 + 1 for even lengths; the old fill used the
    latter for the linspace and raised ValueError for any even cm_length."""

    cm = create_colormap(_cm_params(cm_type="diverging", cm_length=64, change_saturation=0.5))
    assert isinstance(cm, ListedColormap)
    assert cm.N == 64


def test_create_colormap_unknown_type_raises():
    """
    Description
    -----------
    An unrecognised `cm_type` must raise ValueError.

    Parameters
    ----------

    Returns
    -------
    None
    """

    with pytest.raises(ValueError, match="unrecognized cm_type"):
        create_colormap(_cm_params(cm_type="rainbow", equalize_luminance=False))
