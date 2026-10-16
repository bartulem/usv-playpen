"""
@author: bartulem
Centralised figure save-path / format / dpi resolution.

The project's visualisations historically hard-coded one or more of
``(save directory, file format, dpi, timestamp-in-filename)`` at every
``fig.savefig(...)`` callsite. They are now read from
``visualizations_settings.json`` under the ``figures`` block (a value a caller does
not override must be there; nothing falls back to a hard-coded default):

    "figures": {
      "save_directory": "/mnt/falkner/Bartul/figures",
      "fig_format": "png",
      "dpi": 300,
      "timestamp_in_name": true,
      "sequential_cmap": "inferno"
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


def _figures_setting(viz_settings: dict | None, key: str, override_name: str) -> Any:
    """
    Description
    -----------
    One key of the ``figures`` block of ``viz_settings``, read directly: a value a
    caller did not override must come from the shipped settings, never from a
    hard-coded default, so a missing block or key raises and names both.

    Parameters
    ----------
    viz_settings (dict | None)
        Visualisation-settings dict (typically loaded from
        ``visualizations_settings.json``).
    key (str)
        The key of the ``figures`` block.
    override_name (str)
        The argument that would have overridden it, named in the error.

    Returns
    -------
    value (Any)
        ``viz_settings['figures'][key]``.

    Raises
    ------
    KeyError
        ``viz_settings`` is None or lacks ``figures`` / ``key``.
    """

    if viz_settings is None or "figures" not in viz_settings or key not in viz_settings["figures"]:
        msg = f"viz_settings['figures']['{key}'] is not set and `{override_name}` was not given."
        raise KeyError(msg)
    return viz_settings["figures"][key]


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
        Visualisation-settings dict; the ``figures`` block supplies every value not
        overridden (a missing key raises a KeyError).
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

    if override_dir is None:
        save_dir = _figures_setting(viz_settings, "save_directory", "override_dir")
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
        else _figures_setting(viz_settings, "fig_format", "override_format")
    )
    fig_format = str(fig_format).lower().lstrip(".")

    dpi = (
        override_dpi
        if override_dpi is not None
        else int(_figures_setting(viz_settings, "dpi", "override_dpi"))
    )

    use_timestamp = (
        timestamp_in_name
        if timestamp_in_name is not None
        else bool(_figures_setting(viz_settings, "timestamp_in_name", "timestamp_in_name"))
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
        Visualisation-settings dict; the ``figures`` block supplies every
        value not overridden (a missing key raises a KeyError).
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
        Visualisation-settings dict; the ``figures`` block supplies every
        value not overridden (a missing key raises a KeyError).
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
