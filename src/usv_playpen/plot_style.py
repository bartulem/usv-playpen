"""
@author: bartulem
Single source of truth for the usv-playpen matplotlib style.

`apply_plot_style()` registers the five bundled Helvetica TTF
weights with matplotlib's font manager and activates the project
mplstyle at ``_config/usv_playpen.mplstyle``. The call is
idempotent — matplotlib's `font_manager.addfont` and
`pyplot.style.use` are both no-op-on-repeat — so callers can invoke
it freely without coordinating.

Two intended callsites:

  1. Top of every visualisation / plotting Python module in the
     package. This guarantees CLI invocations and external scripts
     that import the module without first touching anything else
     still render with Helvetica.

  2. The first cell of every Jupyter notebook that draws plots via
     matplotlib without going through one of the visualisation
     modules. Notebooks that don't draw anything (pure data /
     aggregator notebooks) do not need to call it.

Both ship the call as a single line, replacing the ~10-line
inline boilerplate that historically lived at the top of each
plotting module and notebook.
"""

from __future__ import annotations

import pathlib

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

_HELVETICA_TTFS: tuple[str, ...] = (
    "Helvetica.ttf",
    "Helvetica-Bold.ttf",
    "Helvetica-Oblique.ttf",
    "Helvetica-BoldOblique.ttf",
    "Helvetica-Light.ttf",
)

_PKG_ROOT = pathlib.Path(__file__).parent
_FONTS_DIR = _PKG_ROOT / "fonts"
_STYLE_PATH = _PKG_ROOT / "_config" / "usv_playpen.mplstyle"


def apply_plot_style() -> None:
    """
    Description
    -----------
    Register the five bundled Helvetica TTFs with matplotlib's font
    manager and activate ``_config/usv_playpen.mplstyle``. Safe to
    call repeatedly — both matplotlib operations no-op on already-
    registered fonts and already-applied styles.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """

    for _ttf in _HELVETICA_TTFS:
        fm.fontManager.addfont(_FONTS_DIR / _ttf)
    plt.style.use(_STYLE_PATH)
