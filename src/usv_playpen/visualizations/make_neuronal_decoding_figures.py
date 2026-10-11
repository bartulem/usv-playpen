"""
@author: bartulem
Figures for the vocal-bout occupancy decode, read from the per-session decoder outputs.

WHAT THIS FIGURE ASSUMES, and it is a narrow set. It draws ONE focal area across a set of
sessions the caller names, beside up to three comparisons against other areas. That shape came
from a specific question -- is bout occupancy decodable from PAG, and is it any more so than
from areas recorded beside it -- and a different set of sessions will not necessarily fit it:

* the sessions are CHOSEN, not discovered. There is no rule here for picking one session per
  animal; the caller passes the list it wants drawn.
* only a SAME-SESSION comparison is exact. Two areas recorded in one session share bins, labels,
  USVs and animal, so only their units differ. A comparison against an area from a different
  session is confounded by how much that animal vocalised, because accuracy falls steeply with
  USV density, and the partner session has to be matched on that BY HAND -- there is no rule
  here for choosing it either.
* the geometry is fixed at three square comparison panels stacked to the main panel's height.
  Four do not fit and one leaves two holes.
* every named session and area must already have been decoded.

Rather than draw something that looks finished when those do not hold, the builder RAISES. An
empty panel cannot be told apart from a null result by anyone reading the page, which is the
worse failure.

Opacity carries how much the animal vocalised, applied by BLENDING toward the page rather than
through an alpha channel, so markers stay opaque and hide the error bar where it would cross
them. A filled marker means that rung cleared its OWN shift null; significance is exact per
curve rather than read off a shared band, which could not exist here, since the null's 99.5th
percentile spans 0.05-0.10 across sessions and its standard deviation spans more than threefold
at the top rung.
"""

from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import polars as pls
from matplotlib.lines import Line2D
from scipy.stats import t as student_t

from ..os_utils import resolve_experimenter_path
from .figure_io import save_figure
from .make_behavioral_videos import pool_brain_area
from .plot_style import apply_plot_style

apply_plot_style()

COLOR_TRACE = "#1F1F1F"
COLOR_PAGE = "#FFFFFF"
CURVE_STYLES = ("-", "--", "-.", ":", (0, (5, 1, 1, 1, 1, 1)), (0, (7, 2)))
OPACITY_RANGE = (0.45, 1.00)
MAIN_PANEL_INCHES = (5.6, 4.0)
SIDE_GAP_INCHES, PANEL_GAP_INCHES = 0.44, 0.20
PAGE_MARGIN_INCHES = {"left": 0.78, "right": 0.14, "bottom": 0.60, "top": 0.74}


def blend(colour: str, weight: float, background: str = COLOR_PAGE) -> str:
    """
    Description
    -----------
    A colour mixed toward the page, returned opaque.

    Used instead of an alpha channel so a marker drawn over an error bar hides it. Mixing
    against the page gives the same appearance a transparent colour would have had.

    Parameters
    ----------
    colour (str)
        Hex colour to fade.
    weight (float)
        How much of the colour survives, 0 fully faded to 1 untouched.
    background (str)
        Hex colour of the page it fades into.

    Returns
    -------
    mixed (str)
        The resulting hex colour.
    """

    front = [int(colour[1 + 2 * k:3 + 2 * k], 16) for k in range(3)]
    back = [int(background[1 + 2 * k:3 + 2 * k], 16) for k in range(3)]
    return "#" + "".join(f"{round(weight * f + (1 - weight) * b):02x}" for f, b in zip(front, back))


class NeuronalDecodingFigureMaker:

    def __init__(self, **kwargs) -> None:
        """
        Description
        -----------
        Bind one instance to a decoder output directory and one settings dict.

        Parameters
        ----------
        **kwargs
            The expected keys are `decoding_directory`, `visualizations_parameter_dict` and
            `message_output`; each is set as an attribute and any other key raises ``TypeError``.

        Returns
        -------
        None
        """

        expected_kwargs = {"decoding_directory", "visualizations_parameter_dict",
                           "message_output"}
        unexpected_kwargs = set(kwargs) - expected_kwargs
        if unexpected_kwargs:
            msg = (f"{type(self).__name__}() got unexpected keyword argument(s) "
                   f"{', '.join(map(repr, sorted(unexpected_kwargs)))}; expected only "
                   f"{', '.join(map(repr, sorted(expected_kwargs)))}.")
            raise TypeError(msg)
        for kw_arg, kw_val in kwargs.items():
            self.__dict__[kw_arg] = kw_val
        self.root = pathlib.Path(resolve_experimenter_path(self.decoding_directory))

    def _area_colour(self, area: str) -> str:
        """
        Description
        -----------
        The display colour for a catalog brain-area acronym.

        The catalog carries raw acronyms (`CENT2`, `SCdw`) while the palette is keyed by display
        bucket (`CENT`, `SC`), so the acronym is pooled first through the project's own mapper
        rather than through a second lookup table invented here.

        Parameters
        ----------
        area (str)
            Brain area as the catalog names it.

        Returns
        -------
        colour (str)
            Hex colour for that area's bucket.
        """

        palette = self.visualizations_parameter_dict["brain_area_colors"]
        bucket = pool_brain_area(area)
        if bucket not in palette:
            msg = (f"{area!r} pools to bucket {bucket!r}, which `brain_area_colors` does not "
                   f"carry ({sorted(palette)}).")
            raise KeyError(msg)
        return palette[bucket]

    def _archive(self, session: str, area: str):
        """
        Description
        -----------
        One area-session's decode archive, or a clear failure saying which one is missing.

        Parameters
        ----------
        session (str)
            Session identifier.
        area (str)
            Brain area as the catalog names it.

        Returns
        -------
        archive (np.lib.npyio.NpzFile)
            The loaded archive.
        """

        path = self.root / f"{session}_{area}.npz"
        if not path.exists():
            msg = (f"No decode output for {area} in {session} at {path}. Run "
                   f"`decode_usv_bout_occupancy` on that session before drawing this figure.")
            raise FileNotFoundError(msg)
        return np.load(path)

    def _curve(self, session: str, area: str, ladder: tuple) -> dict:
        """
        Description
        -----------
        One area-session's rungs, means, 99% half-widths and whether each rung beat its own null.

        Parameters
        ----------
        session (str)
            Session identifier.
        area (str)
            Brain area as the catalog names it.
        ladder (tuple)
            Unit counts to draw, in order.

        Returns
        -------
        curve (dict)
            Keys `rungs`, `means`, `halves` and `clears`.
        """

        archive = self._archive(session, area)
        rungs, means, halves, clears = [], [], [], []
        for count in ladder:
            if str(count) not in archive:
                continue
            draws = archive[str(count)]
            rungs.append(count)
            means.append(float(draws.mean()))
            halves.append(float(student_t.ppf(0.995, draws.size - 1)
                                * draws.std(ddof=1) / np.sqrt(draws.size)))
            clears.append(bool(draws.mean()
                               > np.percentile(archive[f"null_{count}"], 99.5)))
        if not rungs:
            msg = f"{area} in {session} carries no rung from the requested ladder {ladder}."
            raise ValueError(msg)
        return {"rungs": rungs, "means": means, "halves": halves, "clears": clears}

    @staticmethod
    def _draw_curve(axis, curve: dict, shade: str, style: str, width: float, size: float,
                    label: str = None) -> None:
        """
        Description
        -----------
        One curve: the line, capless bars beneath it, then opaque markers over the bars.

        The bars are drawn first and the markers over them, which leaves each bar extending from
        the marker's edge outward rather than crossing it.

        Parameters
        ----------
        axis (matplotlib axis)
            Target axis.
        curve (dict)
            The output of `_curve`.
        shade (str)
            Hex colour, already blended to its opacity.
        style (str)
            Line style.
        width (float)
            Line width.
        size (float)
            Marker size.
        label (str)
            Legend entry, or None for no entry.

        Returns
        -------
        None
        """

        axis.plot(curve["rungs"], curve["means"], linestyle=style, color=shade, linewidth=width,
                  zorder=2, label=label)
        axis.errorbar(curve["rungs"], curve["means"], yerr=curve["halves"], fmt="none",
                      ecolor=shade, elinewidth=width * 0.85, capsize=0, zorder=3)
        for rung, mean, cleared in zip(curve["rungs"], curve["means"], curve["clears"]):
            axis.plot(rung, mean, marker="o", markersize=size,
                      markerfacecolor=shade if cleared else COLOR_PAGE,
                      markeredgecolor=COLOR_TRACE, markeredgewidth=0.5, zorder=4)

    def make_bout_decoding_figure(
        self,
        focal_area: str,
        focal_sessions: tuple,
        comparisons: tuple,
        unit_counts: tuple,
        out_dir: str | pathlib.Path | None = None,
    ) -> pathlib.Path:
        """
        Description
        -----------
        Draw the focal area's accuracy against unit count for every named session, beside up to
        three comparison panels, and write it out.

        Read the module docstring before using this on a new set of sessions: the session list,
        the focal area and each comparison's partner session are the caller's choices, and only
        a same-session comparison is exact.

        Parameters
        ----------
        focal_area (str)
            Area drawn in the main panel.
        focal_sessions (tuple)
            One ``(mouse, session, usv_count)`` per curve; `usv_count` drives the opacity.
        comparisons (tuple)
            Up to three ``(area, area_session, focal_partner_session, note)`` entries. `note`
            is drawn on the panel and should say whether the comparison is exact.
        unit_counts (tuple)
            Unit counts to draw, in order.
        out_dir (str | pathlib.Path | None)
            Destination directory; None falls back to the `figures` settings block.

        Returns
        -------
        destination (pathlib.Path)
            The written file.
        """

        if len(comparisons) != 3:
            msg = (f"This figure's geometry holds exactly three comparison panels; "
                   f"{len(comparisons)} were given. Four do not fit and fewer leaves holes.")
            raise ValueError(msg)
        if not focal_sessions:
            msg = "No focal sessions given; there is nothing to draw."
            raise ValueError(msg)

        counts = {mouse: int(usvs) for mouse, _session, usvs in focal_sessions}
        low, high = min(counts.values()), max(counts.values())
        focal_colour = self._area_colour(focal_area)

        side = (MAIN_PANEL_INCHES[1] - 2 * PANEL_GAP_INCHES) / 3
        width = (PAGE_MARGIN_INCHES["left"] + MAIN_PANEL_INCHES[0] + SIDE_GAP_INCHES + side
                 + PAGE_MARGIN_INCHES["right"])
        height = PAGE_MARGIN_INCHES["bottom"] + MAIN_PANEL_INCHES[1] + PAGE_MARGIN_INCHES["top"]
        figure = plt.figure(figsize=(width, height))
        axis = figure.add_axes((PAGE_MARGIN_INCHES["left"] / width,
                                PAGE_MARGIN_INCHES["bottom"] / height,
                                MAIN_PANEL_INCHES[0] / width, MAIN_PANEL_INCHES[1] / height))

        for (mouse, session, usvs), style in zip(focal_sessions, CURVE_STYLES):
            curve = self._curve(session, focal_area, unit_counts)
            span = max(high - low, 1)
            weight = (OPACITY_RANGE[0]
                      + (OPACITY_RANGE[1] - OPACITY_RANGE[0]) * (int(usvs) - low) / span)
            n_units = int(self._archive(session, focal_area)["ladder"][-1])
            self._draw_curve(axis, curve, blend(focal_colour, weight), style, 1.4, 5.0,
                             label=f"{mouse}  ({n_units} units, {int(usvs)} USVs)")

        axis.set_xscale("log")
        axis.set_xticks(list(unit_counts))
        axis.set_xticklabels([str(c) for c in unit_counts])
        axis.set_xlabel("units in the decoder", fontsize=9)
        axis.set_ylabel("balanced accuracy", fontsize=9)
        axis.set_title("Nearest neighbor decoding (bout vs no-bout)", fontsize=12, loc="left",
                       pad=18)
        axis.text(0.0, 1.015, "250 ms bins, spikes during any vocalization excluded", transform=axis.transAxes,
                  fontsize=8, color=COLOR_TRACE)
        curves_legend = axis.legend(fontsize=7, frameon=False, loc="upper left",
                                    handlelength=3.0)
        axis.add_artist(curves_legend)
        marks = [Line2D([], [], marker="o", linestyle="none", markersize=5.0,
                        markerfacecolor=face, markeredgecolor=COLOR_TRACE, markeredgewidth=0.5,
                        label=text)
                 for face, text in ((focal_colour, "above its own shift null (99%)"),
                                    (COLOR_PAGE, "within the null"))]
        axis.legend(handles=marks, fontsize=6, frameon=False, loc="lower right",
                    handlelength=1.0, labelspacing=0.15, borderpad=0.2, handletextpad=0.4)
        axis.tick_params(labelsize=8)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)

        left = (PAGE_MARGIN_INCHES["left"] + MAIN_PANEL_INCHES[0] + SIDE_GAP_INCHES) / width
        for index, (area, area_session, focal_session, note) in enumerate(comparisons):
            area_colour = self._area_colour(area)
            bottom = (PAGE_MARGIN_INCHES["bottom"]
                      + (2 - index) * (side + PANEL_GAP_INCHES)) / height
            panel = figure.add_axes((left, bottom, side / width, side / height))
            shared = tuple(c for c in unit_counts
                           if str(c) in self._archive(area_session, area)
                           and str(c) in self._archive(focal_session, focal_area))
            if not shared:
                msg = (f"{area} in {area_session} and {focal_area} in {focal_session} share no "
                       f"rung of {unit_counts}, so the panel would compare nothing.")
                raise ValueError(msg)
            self._draw_curve(panel, self._curve(focal_session, focal_area, shared),
                             focal_colour, "-", 1.0, 3.2)
            self._draw_curve(panel, self._curve(area_session, area, shared), area_colour,
                             "-", 1.0, 3.2)
            panel.set_xscale("log")
            panel.set_xticks(list(unit_counts))
            panel.set_ylim(0.50, 0.80)
            panel.set_yticks((0.5, 0.6, 0.7, 0.8))
            panel.tick_params(labelsize=6, length=2.0, width=0.5)
            for spine in ("top", "right"):
                panel.spines[spine].set_visible(False)
            panel.text(0.03, 0.96, area, transform=panel.transAxes, fontsize=7, va="top",
                       color=area_colour)
            panel.text(0.03, 0.855, focal_area, transform=panel.transAxes, fontsize=7, va="top",
                       color=focal_colour)
            panel.text(0.97, 0.04, note, transform=panel.transAxes, fontsize=5, ha="right",
                       va="bottom", color=COLOR_TRACE)
            if index == len(comparisons) - 1:
                panel.set_xticklabels([str(c) for c in unit_counts])
                panel.set_xlabel("units", fontsize=7, labelpad=2)
            else:
                panel.set_xticklabels([])

        destination = save_figure(figure, f"bout_decoding_{focal_area.lower()}",
                                  self.visualizations_parameter_dict, override_dir=out_dir)
        plt.close(figure)
        self.message_output(f"wrote {destination}")
        return destination

    def summarize(self, sessions: tuple) -> pls.DataFrame:
        """
        Description
        -----------
        Gather the per-session decoder tables into one frame, for the cross-session comparisons
        the analysis itself deliberately does not make.

        The decoder runs per session and pools nothing, so assembling across sessions is the
        caller's step and belongs here rather than in the analysis.

        Parameters
        ----------
        sessions (tuple)
            Session identifiers to gather.

        Returns
        -------
        summary (pls.DataFrame)
            One row per area-session-rung.
        """

        frames = []
        for session in sessions:
            path = self.root / f"{session}_bout_decoding.parquet"
            if not path.exists():
                msg = (f"No decoder summary for {session} at {path}. Run "
                       f"`decode_usv_bout_occupancy` on it first.")
                raise FileNotFoundError(msg)
            frames.append(pls.read_parquet(path))
        return pls.concat(frames, how="vertical")
