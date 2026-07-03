"""
@author: bartulem
Static soundness guard for the analysis notebooks.

Beyond the reachability check in ``test_docs_notebooks.py``, this module statically
checks every shipped notebook without executing it:

* **syntax** — every code cell must parse (``ast.parse``);
* **undefined names** — each notebook's code cells, concatenated in order (so
  cell-to-cell name flow is modelled as a single module), must be free of
  pyflakes' undefined-name rule (ruff ``F821``): a typo or a reference to a name
  that does not exist fails the suite.

IPython line/cell magics (``%…`` / ``!…``) are stripped before checking, and the
names IPython injects into a live notebook namespace (``display`` etc.) are
whitelisted. This does NOT execute the notebooks — data-dependent runtime
behaviour still needs a real ``jupyter nbconvert --execute`` run on a machine
with the data mounted. The undefined-name check is skipped (not failed) when the
``ruff`` executable is unavailable.
"""
from __future__ import annotations

import ast
import json
import pathlib
import shutil
import subprocess
import tempfile

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
NOTEBOOK_DIR = REPO_ROOT / "src" / "usv_playpen" / "notebooks"

# Names IPython injects into a notebook namespace: defined at runtime, invisible
# to static analysis, so whitelist them to avoid spurious F821 hits.
_IPYTHON_BUILTINS = ("display", "get_ipython", "HTML")


def _notebooks() -> list[pathlib.Path]:
    """
    Description
    -----------
    Collect every Jupyter notebook shipped under ``notebooks/``.

    Parameters
    ----------

    Returns
    -------
    notebooks (list of pathlib.Path)
        Sorted list of ``*.ipynb`` paths.
    """

    return sorted(NOTEBOOK_DIR.glob("*.ipynb"))


def _code_cells(notebook_path: pathlib.Path) -> list[tuple[int, str]]:
    """
    Description
    -----------
    Return each code cell's source with IPython line/cell magics (``%…``) and
    shell escapes (``!…``) stripped, so the source is plain Python.

    Parameters
    ----------
    notebook_path (pathlib.Path)
        Path to the ``.ipynb`` to read.

    Returns
    -------
    cells (list of tuple[int, str])
        ``(cell_index, magic-stripped source)`` for every code cell.
    """

    notebook = json.loads(notebook_path.read_text())
    cells = []
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        stripped = "\n".join(
            line for line in source.split("\n") if not line.lstrip().startswith(("%", "!"))
        )
        cells.append((index, stripped))
    return cells


def test_every_notebook_cell_parses():
    """
    Description
    -----------
    Assert every code cell of every shipped notebook parses as valid Python
    (magics stripped). Pure syntax check; always runs.

    Parameters
    ----------

    Returns
    -------
    None
    """

    errors = []
    for notebook_path in _notebooks():
        for index, source in _code_cells(notebook_path):
            try:
                ast.parse(source)
            except SyntaxError as exc:
                errors.append(f"{notebook_path.name} cell {index}: {exc.msg} (line {exc.lineno})")
    assert not errors, "notebook code cells with syntax errors:\n" + "\n".join(errors)


def test_no_undefined_names_in_notebooks():
    """
    Description
    -----------
    Assert no notebook references an undefined name. Each notebook's code cells
    are concatenated in order (modelling cell-to-cell name flow as one module),
    prefixed with the IPython-builtin whitelist, and checked with ruff's ``F821``
    rule. Skipped when the ``ruff`` executable is not on PATH.

    Parameters
    ----------

    Returns
    -------
    None
    """

    ruff = shutil.which("ruff")
    if ruff is None:
        pytest.skip("ruff executable not available; undefined-name (F821) check skipped")

    whitelist = " = ".join(_IPYTHON_BUILTINS) + " = None  # IPython-injected notebook builtins\n"
    failures = []
    for notebook_path in _notebooks():
        concatenated = whitelist + "\n".join(
            f"# --- cell {index} ---\n{source}" for index, source in _code_cells(notebook_path)
        ) + "\n"
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as handle:
            handle.write(concatenated)
            temp_path = handle.name
        try:
            result = subprocess.run(
                [ruff, "check", "--select", "F821", "--output-format", "concise", temp_path],
                capture_output=True,
                text=True,
                check=False,   # ruff exits non-zero on findings; we parse stdout for F821 instead
            )
        finally:
            pathlib.Path(temp_path).unlink()
        for line in result.stdout.splitlines():
            if "F821" in line:
                failures.append(f"{notebook_path.name}: {line.split(':', 1)[1].strip()}")
    assert not failures, "notebook cells referencing undefined names:\n" + "\n".join(failures)
