"""
@author: bartulem
Centralised figure save-path / format / dpi resolution.

The project's visualisations historically hard-coded one or more of
``(save directory, file format, dpi, timestamp-in-filename)`` at every
``fig.savefig(...)`` callsite. The defaults are now collected in
``visualizations_settings.json`` under the ``figures`` block:

    "figures": {
      "save_directory": "/mnt/falkner/Bartul/figures",
      "fig_format": "png",
      "dpi": 300,
      "timestamp_in_name": true,
      "cmap": "inferno"
    }

Three entry points are provided:

  * ``resolve_save_path(stem, viz_settings, ...)`` — returns
    ``(absolute_path, fig_format, dpi)`` after applying the override →
    settings → hard-coded fallback chain. The directory is run through
    ``configure_path`` so the helper works on both Linux mounts
    (``/mnt/...``) and macOS mounts (``/Volumes/...``).

  * ``save_figure(fig, stem, viz_settings, ...)`` — the most common
    shape: resolves the path/format/dpi and calls
    ``fig.savefig(path, format=fmt, dpi=dpi, bbox_inches='tight')`` for
    you. Returns the resolved path.

  * ``resolve_pdf_path(stem, viz_settings, ...)`` — PdfPages-friendly
    variant for callers that need a fixed ``.pdf`` extension and want
    to manage the ``with PdfPages(path) as pdf`` lifecycle themselves.
    Returns ``(absolute_path, dpi)``.

All three honour the per-call overrides ``override_dir`` (the "save
next to data" callsites pass an explicit session directory),
``override_format`` and ``override_dpi``. The ``timestamp_in_name``
toggle is independent of the directory choice — "save next to data"
callsites typically pass ``timestamp_in_name=False`` since their
filenames already embed a session id or unit id.
"""

from __future__ import annotations

import pathlib
from datetime import datetime
from typing import Any

from ..os_utils import configure_path


_DEFAULT_FIG_FORMAT = "png"
_DEFAULT_DPI = 300
_DEFAULT_TIMESTAMP_IN_NAME = True


def _figures_block(viz_settings: dict | None) -> dict:
    """
    Description
    -----------
    Return the ``figures`` block from ``viz_settings`` if present, an
    empty dict otherwise. Tolerates either ``None`` or a settings
    dict with no ``figures`` key (older configurations).

    Parameters
    ----------
    viz_settings (dict | None)
        Visualisation-settings dict (typically loaded from
        ``visualizations_settings.json``).

    Returns
    -------
    figures (dict)
        The ``figures`` block, or ``{}``.
    """

    if not viz_settings:
        return {}
    return viz_settings.get("figures", {}) or {}


def _append_timestamp(stem: str) -> str:
    """Append ``_<YYYYMMDD>_<HHMMSS>`` to ``stem`` and return it."""
    return f"{stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def resolve_save_path(
        stem: str,
        viz_settings: dict | None,
        *,
        override_dir: str | pathlib.Path | None = None,
        override_format: str | None = None,
        override_dpi: int | None = None,
        timestamp_in_name: bool | None = None,
) -> tuple[pathlib.Path, str, int]:
    """
    Description
    -----------
    Resolve the final save path, file format and dpi for a figure by
    applying the override → settings → hard-coded fallback chain.

    The directory is run through ``configure_path`` so a settings
    value of ``/mnt/falkner/...`` resolves to ``/Volumes/falkner/...``
    on macOS automatically. The directory is created if missing.

    Parameters
    ----------
    stem (str)
        Filename stem without extension and without timestamp suffix.
        The resolver appends ``_<YYYYMMDD>_<HHMMSS>`` and the format
        extension as configured.
    viz_settings (dict | None)
        Visualisation-settings dict; the ``figures`` block is read for
        defaults.
    override_dir (str | pathlib.Path | None)
        Per-call directory override. The "save next to data" callsites
        (per-unit ratemaps, per-session yield figures) pass an explicit
        session path here. ``None`` falls back to
        ``viz_settings["figures"]["save_directory"]``.
    override_format (str | None)
        Per-call format override. Defaults to
        ``viz_settings["figures"]["fig_format"]`` and then to
        ``"png"``.
    override_dpi (int | None)
        Per-call dpi override. Defaults to
        ``viz_settings["figures"]["dpi"]`` and then to ``300``.
    timestamp_in_name (bool | None)
        Per-call timestamp toggle. ``None`` falls back to
        ``viz_settings["figures"]["timestamp_in_name"]`` and then to
        ``True``.

    Returns
    -------
    path (pathlib.Path)
        Absolute path to write to (parent directory created).
    fig_format (str)
        Lower-case format string (e.g. ``"svg"``, ``"pdf"``,
        ``"png"``).
    dpi (int)
        Resolved dpi.
    """

    figures = _figures_block(viz_settings)

    if override_dir is None:
        save_dir = figures.get("save_directory")
        if not save_dir:
            raise ValueError(
                "No save directory: neither `override_dir` nor "
                "`viz_settings['figures']['save_directory']` is set."
            )
    else:
        # An empty-string `override_dir` would otherwise pass through
        # `configure_path("")` and silently resolve to the current
        # working directory; reject it explicitly so the caller fixes
        # the (mis-)configured path instead of scattering figures into
        # the CWD.
        if str(override_dir).strip() == "":
            raise ValueError(
                "`override_dir` is an empty string; pass a real "
                "directory or `None` to fall back to "
                "`viz_settings['figures']['save_directory']`."
            )
        save_dir = override_dir
    save_dir = pathlib.Path(configure_path(str(save_dir)))
    save_dir.mkdir(parents=True, exist_ok=True)

    fig_format = (
        override_format
        if override_format is not None
        else figures.get("fig_format", _DEFAULT_FIG_FORMAT)
    )
    fig_format = str(fig_format).lower().lstrip(".")

    dpi = (
        override_dpi
        if override_dpi is not None
        else int(figures.get("dpi", _DEFAULT_DPI))
    )

    use_timestamp = (
        timestamp_in_name
        if timestamp_in_name is not None
        else bool(figures.get("timestamp_in_name", _DEFAULT_TIMESTAMP_IN_NAME))
    )

    final_stem = _append_timestamp(stem) if use_timestamp else stem
    path = save_dir / f"{final_stem}.{fig_format}"
    return path, fig_format, dpi


def save_figure(
        fig: Any,
        stem: str,
        viz_settings: dict | None,
        *,
        override_dir: str | pathlib.Path | None = None,
        override_format: str | None = None,
        override_dpi: int | None = None,
        timestamp_in_name: bool | None = None,
        bbox_inches: str | None = "tight",
        **savefig_kwargs,
) -> pathlib.Path:
    """
    Description
    -----------
    Resolve the save path/format/dpi and write ``fig`` to disk via
    ``fig.savefig(...)``. Returns the resolved path.

    Parameters
    ----------
    fig (matplotlib.figure.Figure)
        Figure to write.
    stem (str)
        Filename stem (see ``resolve_save_path``).
    viz_settings (dict | None)
        Visualisation-settings dict; ``figures`` block supplies the
        defaults.
    override_dir, override_format, override_dpi, timestamp_in_name
        Per-call overrides; same semantics as ``resolve_save_path``.
    bbox_inches (str | None)
        Forwarded to ``fig.savefig`` (default ``"tight"`` matches the
        repo's prior convention).
    **savefig_kwargs
        Additional kwargs forwarded to ``fig.savefig``. Caller may
        pass e.g. ``transparent=True`` or ``facecolor=...`` here.

    Returns
    -------
    path (pathlib.Path)
        Absolute path that was written.
    """

    path, fig_format, dpi = resolve_save_path(
        stem=stem,
        viz_settings=viz_settings,
        override_dir=override_dir,
        override_format=override_format,
        override_dpi=override_dpi,
        timestamp_in_name=timestamp_in_name,
    )
    fig.savefig(
        path,
        format=fig_format,
        dpi=dpi,
        bbox_inches=bbox_inches,
        **savefig_kwargs,
    )
    return path


def resolve_pdf_path(
        stem: str,
        viz_settings: dict | None,
        *,
        override_dir: str | pathlib.Path | None = None,
        override_dpi: int | None = None,
        timestamp_in_name: bool | None = None,
) -> tuple[pathlib.Path, int]:
    """
    Description
    -----------
    PdfPages-friendly variant of ``resolve_save_path``. Returns
    ``(absolute_path_with_.pdf_extension, dpi)`` so the caller can
    drive ``with PdfPages(path) as pdf: pdf.savefig(fig, dpi=dpi)``
    itself.

    Parameters
    ----------
    stem (str)
        Filename stem (no extension).
    viz_settings (dict | None)
        Visualisation-settings dict; ``figures`` block supplies the
        defaults.
    override_dir, override_dpi, timestamp_in_name
        Per-call overrides; same semantics as ``resolve_save_path``.

    Returns
    -------
    path (pathlib.Path)
        Absolute ``.pdf`` path (parent directory created).
    dpi (int)
        Resolved dpi.
    """

    path, _fmt, dpi = resolve_save_path(
        stem=stem,
        viz_settings=viz_settings,
        override_dir=override_dir,
        override_format="pdf",
        override_dpi=override_dpi,
        timestamp_in_name=timestamp_in_name,
    )
    return path, dpi
