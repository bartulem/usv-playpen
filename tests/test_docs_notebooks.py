"""
@author: bartulem
Reachability guard for the documentation's notebook catalog.

Every analysis notebook (and the marimo app) shipped under
``src/usv_playpen/notebooks/`` must be referenced by name in
``docs/Notebooks.rst`` (the single, detailed notebook catalog). This converts
"no guarantee every notebook is reachable" from a hope into an enforced
invariant: adding a notebook without documenting it fails the suite rather than
silently orphaning it (as happened previously with
``usv_summary_statistics_plots.ipynb``).
"""

from __future__ import annotations

import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
NOTEBOOK_DIR = REPO_ROOT / "src" / "usv_playpen" / "notebooks"
NOTEBOOKS_RST = REPO_ROOT / "docs" / "Notebooks.rst"


def _shipped_notebooks() -> list[str]:
    """
    Description
    -----------
    Collect the file names of every notebook artifact under
    ``notebooks/`` that should be catalogued: all Jupyter notebooks
    (``*.ipynb``) plus any top-level marimo / script app (``*.py`` that is not a
    dunder module such as ``__init__.py``).

    Parameters
    ----------

    Returns
    -------
    notebooks (list of str)
        Sorted list of file names (with extension) expected in the catalog.
    """

    return sorted(
        entry.name
        for entry in NOTEBOOK_DIR.iterdir()
        if entry.is_file()
        and (entry.suffix == ".ipynb"
             or (entry.suffix == ".py" and not entry.name.startswith("__")))
    )


def test_every_notebook_is_documented():
    """
    Description
    -----------
    Assert that every shipped notebook artifact is named in
    ``docs/Notebooks.rst``. The catalog mentions each notebook by its full file
    name (e.g. ``**modeling_analyses.ipynb**``), so a simple substring check is
    sufficient and robust to layout changes.

    Parameters
    ----------

    Returns
    -------
    None
    """

    catalog_text = NOTEBOOKS_RST.read_text()
    missing = [name for name in _shipped_notebooks() if name not in catalog_text]
    assert not missing, (
        f"notebooks not referenced in docs/Notebooks.rst: {missing}. "
        f"Add a catalog entry (and, for a .ipynb, a 'Rendered notebooks' toctree line)."
    )
