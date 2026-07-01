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
from matplotlib.ft2font import FT2Font

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


def _font_has_glyph(font_path: str, codepoint: int) -> bool:
    """
    Description
    -----------
    Return whether the font file at ``font_path`` provides a glyph for the given
    Unicode ``codepoint`` (i.e. FreeType maps it to a non-zero glyph index). Used
    to prune glyph-poor ``DejaVu Sans`` faces from the font manager so the
    Helvetica -> DejaVu Sans per-glyph fallback lands on a face that actually
    carries the requested character. A font that cannot be opened is treated as
    "has the glyph" so a parse failure never over-prunes the manager.

    Parameters
    ----------
    font_path (str)
        Filesystem path to the font file to inspect.
    codepoint (int)
        The Unicode code point to test (e.g. ``0x2640`` for the female sign).

    Returns
    -------
    (bool)
        True if the font provides a glyph for ``codepoint`` (or could not be
        inspected); False if the font lacks the glyph.
    """

    try:
        return FT2Font(font_path).get_char_index(codepoint) != 0
    except (RuntimeError, OSError):
        return True


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
    this session. We likewise drop any ``DejaVu Sans`` face lacking the
    ♀ / ♂ sign glyphs (U+2640 / U+2642) -- e.g. a system
    ``DejaVuSans-ExtraLight`` -- so the Helvetica -> DejaVu Sans per-glyph
    fallback for those signs never lands on a glyph-poor face (tofu box).

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
    # The ♂ / ♀ sex signs (U+2640 / U+2642) in behavioral-video ID labels are
    # absent from Helvetica, so matplotlib's per-glyph fallback supplies them from
    # 'DejaVu Sans'. At the project's light font.weight a system
    # 'DejaVuSans-ExtraLight' can win that lookup, and that thin variant also lacks
    # the signs (tofu box). Drop every 'DejaVu Sans' face whose file has no U+2640
    # glyph so the fallback always resolves to a DejaVu that carries the signs.
    fm.fontManager.ttflist = [
        _entry for _entry in fm.fontManager.ttflist
        if _entry.name != 'DejaVu Sans' or _font_has_glyph(_entry.fname, 0x2640)
    ]
    fm.fontManager._findfont_cached.cache_clear()

    plt.style.use(_STYLE_PATH)
