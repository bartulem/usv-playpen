"""
@author: bartulem
Tests for `usv_playpen.visualizations.figure_io`.

Coverage:

  * The override → ``figures`` block → hard-coded fallback chain for
    ``resolve_save_path``.
  * Timestamp toggle behaviour (default-on, explicit-off, settings-off).
  * Directory creation side effect (parent created when missing).
  * ``configure_path`` is applied to the resolved directory (we point
    it at a ``tmp_path`` subtree that exists, so the round-trip is
    identity).
  * ``save_figure`` writes a file and forwards extra savefig kwargs.
  * ``resolve_pdf_path`` returns a ``.pdf`` path regardless of the
    ``figures.fig_format`` setting.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from usv_playpen.visualizations.figure_io import (  # noqa: E402
    resolve_save_path,
    save_figure,
    resolve_pdf_path,
)


_TS_RE = re.compile(r"_\d{8}_\d{6}$")


def _settings(tmp_path: Path, **overrides) -> dict:
    figures = {
        "save_directory": str(tmp_path),
        "fig_format": "svg",
        "dpi": 300,
        "timestamp_in_name": True,
        "cmap": "inferno",
    }
    figures.update(overrides)
    return {"figures": figures}


# resolve_save_path — fallback chain


def test_resolve_save_path_uses_settings_defaults(tmp_path):
    """With no overrides, every parameter comes from `figures`."""
    s = _settings(tmp_path)
    path, fmt, dpi = resolve_save_path("yield", s)
    assert path.parent == tmp_path
    assert fmt == "svg"
    assert dpi == 300
    # stem is "yield_<TS>"; extension ".svg"
    assert _TS_RE.search(path.stem)
    assert path.suffix == ".svg"


def test_resolve_save_path_overrides_take_precedence(tmp_path):
    """Per-call overrides win over the settings block."""
    s = _settings(tmp_path)
    sub = tmp_path / "alt"
    path, fmt, dpi = resolve_save_path(
        "x", s,
        override_dir=sub,
        override_format="png",
        override_dpi=600,
        timestamp_in_name=False,
    )
    assert path.parent == sub
    assert fmt == "png"
    assert dpi == 600
    assert path.name == "x.png"  # no timestamp
    assert sub.is_dir()  # parent created


def test_resolve_save_path_format_lowercased(tmp_path):
    """Format is normalised: leading dot stripped, lower-cased."""
    s = _settings(tmp_path)
    _, fmt, _ = resolve_save_path("x", s, override_format=".PDF")
    assert fmt == "pdf"


def test_resolve_save_path_timestamp_setting_off(tmp_path):
    """`timestamp_in_name=False` in settings disables the suffix."""
    s = _settings(tmp_path, timestamp_in_name=False)
    path, _, _ = resolve_save_path("clean_name", s)
    assert path.name == "clean_name.svg"


def test_resolve_save_path_timestamp_per_call_overrides_setting(tmp_path):
    """Per-call timestamp override beats the settings value."""
    s = _settings(tmp_path, timestamp_in_name=False)
    path, _, _ = resolve_save_path("clean_name", s, timestamp_in_name=True)
    assert _TS_RE.search(path.stem)


def test_resolve_save_path_no_settings_raises_without_override_dir():
    """`viz_settings=None` and no `override_dir` → explicit error."""
    with pytest.raises(ValueError, match="No save directory"):
        resolve_save_path("x", None)


def test_resolve_save_path_no_settings_with_override_dir_ok(tmp_path):
    """`viz_settings=None` + `override_dir` works with hard-coded
    fallbacks for the other parameters."""
    path, fmt, dpi = resolve_save_path("x", None, override_dir=tmp_path)
    assert path.parent == tmp_path
    assert fmt == "png"
    assert dpi == 300


def test_resolve_save_path_creates_missing_parent(tmp_path):
    """Save dir is created if it doesn't exist yet."""
    new_dir = tmp_path / "nested" / "deeper"
    assert not new_dir.exists()
    resolve_save_path("x", None, override_dir=new_dir)
    assert new_dir.is_dir()


# save_figure


def test_save_figure_writes_a_file_at_resolved_path(tmp_path):
    """End-to-end: returned path exists on disk."""
    s = _settings(tmp_path)
    fig, _ = plt.subplots(figsize=(1, 1))
    out_path = save_figure(fig, "smoke", s, timestamp_in_name=False)
    plt.close(fig)
    assert out_path == tmp_path / "smoke.svg"
    assert out_path.is_file()


def test_save_figure_forwards_extra_savefig_kwargs(tmp_path):
    """Extra kwargs (e.g. `transparent`) reach `fig.savefig`."""
    captured = {}

    class _FakeFig:
        def savefig(self, path, **kwargs):
            captured["path"] = path
            captured["kwargs"] = kwargs

    s = _settings(tmp_path)
    save_figure(
        _FakeFig(), "k", s, timestamp_in_name=False,
        transparent=True, facecolor="#fff",
    )
    assert captured["kwargs"]["transparent"] is True
    assert captured["kwargs"]["facecolor"] == "#fff"
    assert captured["kwargs"]["format"] == "svg"
    assert captured["kwargs"]["dpi"] == 300
    assert captured["kwargs"]["bbox_inches"] == "tight"


# resolve_pdf_path


def test_resolve_pdf_path_returns_pdf_extension_regardless_of_settings(tmp_path):
    """`resolve_pdf_path` always produces a `.pdf` path even when
    `figures.fig_format` is set to something else."""
    s = _settings(tmp_path, fig_format="svg")
    path, dpi = resolve_pdf_path("doc", s, timestamp_in_name=False)
    assert path.name == "doc.pdf"
    assert dpi == 300


def test_resolve_pdf_path_dpi_override(tmp_path):
    """`override_dpi` propagates."""
    s = _settings(tmp_path)
    _, dpi = resolve_pdf_path("doc", s, override_dpi=600)
    assert dpi == 600
