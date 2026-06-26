"""
@author: bartulem
Single source of truth for the usv-playpen matplotlib style.

`apply_plot_style()` registers the five bundled Helvetica TTF
weights with matplotlib's font manager and activates the project
mplstyle at ``_config/usv_playpen.mplstyle``. The visual result is
stable across repeated calls — the bundled faces stay authoritative
and the active style is unchanged — so callers can invoke it freely
without coordinating. Note, however, that the call is not a true
no-op on repeat: `font_manager.addfont` unconditionally *appends* to
the manager's ``ttflist`` (it does not deduplicate), so each
invocation re-parses the bundled TTFs and grows ``ttflist`` with
fresh bundled-Helvetica entries that the de-shadowing prune below
deliberately keeps. Repeated calls within a long-lived session are
therefore correctness-safe but not free.

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

# This module lives in usv_playpen/visualizations/, so the package root
# (which holds the bundled fonts/ and _config/ directories) is one level up.
_PKG_ROOT = pathlib.Path(__file__).parent.parent
_FONTS_DIR = _PKG_ROOT / "fonts"
_STYLE_PATH = _PKG_ROOT / "_config" / "usv_playpen.mplstyle"


def apply_plot_style() -> None:
    """
    Description
    -----------
    Register the five bundled Helvetica TTFs with matplotlib's font
    manager, ensure they win lookups over any same-named system font,
    and activate ``_config/usv_playpen.mplstyle``. The rendered result
    is stable when called repeatedly, but the call is not a literal
    no-op: ``font_manager.addfont`` always appends to ``ttflist``
    rather than skipping already-registered fonts, so each invocation
    re-parses the bundled TTFs and adds fresh bundled-Helvetica entries
    (the de-shadowing prune keeps every bundled entry, so duplicates
    accumulate). ``plt.style.use`` re-applies the same style each time.

    `font_manager.addfont` only *appends* to the manager's font list,
    so a system face that happens to be named "Helvetica" (e.g. a
    FreeFont alias under ``/usr/share/fonts``) that was discovered
    first wins ``findfont`` score ties and silently shadows the bundled
    weights — normal / bold lookups then resolve to the system file
    instead of the TTFs shipped with this package. To make the bundled
    faces authoritative we drop every non-bundled "Helvetica" entry
    from the manager's ``ttflist`` and clear the per-lookup cache so the
    change takes effect even if a Helvetica lookup was already cached
    this session.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """

    bundled_helvetica_paths = set()
    for _ttf in _HELVETICA_TTFS:
        _ttf_path = _FONTS_DIR / _ttf
        fm.fontManager.addfont(_ttf_path)
        bundled_helvetica_paths.add(str(_ttf_path.resolve()))

    fm.fontManager.ttflist = [
        _entry for _entry in fm.fontManager.ttflist
        if _entry.name != 'Helvetica' or str(pathlib.Path(_entry.fname).resolve()) in bundled_helvetica_paths
    ]
    fm.fontManager._findfont_cached.cache_clear()

    plt.style.use(_STYLE_PATH)
