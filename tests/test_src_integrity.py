"""
Source-tree integrity check -- the regression guard for the v0.10.1 / v0.10.2
empty-YAML disaster.

Captures SHA-256 hashes of every file under ``src/usv_playpen`` at the start
of the pytest session (via ``pytest_sessionstart`` in ``conftest.py``) and
compares the post-run state here. Any test that mutates a tracked package
file as a side effect -- which is how the bundled ``_config/_metadata.yaml``
got silently zeroed by ``test_conduct_recording_cli_invokes_controller
_and_dumps_metadata`` on every test run before commit 75b0ab2 -- will now
fail this test with a per-file diff instead of leaving a corrupted artifact
in the working tree (or, worse, in a tagged release).

This module is forced to run LAST by ``pytest_collection_modifyitems`` in
``conftest.py``.
"""

from __future__ import annotations

from .conftest import _hash_src_tree


def test_src_tree_unmodified_during_test_session(pytestconfig):
    """
    Description
    -----------
    Re-hashes every file under ``src/usv_playpen`` and compares against the
    session-start baseline captured in ``pytest_sessionstart``. Fails with a
    human-readable per-file report listing additions, deletions, and content
    mutations so the offending test is easy to locate.

    Parameters
    ----------
    pytestconfig (pytest.Config)
        Pytest config fixture; carries the baseline attribute populated by
        the ``pytest_sessionstart`` hook in ``conftest.py``.

    Returns
    -------
    None
    """

    baseline = getattr(pytestconfig, "_src_tree_baseline", None)
    assert baseline is not None, (
        "pytest_sessionstart did not run -- the src-tree integrity baseline "
        "was never captured. Check tests/conftest.py."
    )

    current = _hash_src_tree()

    added = sorted(set(current) - set(baseline))
    removed = sorted(set(baseline) - set(current))
    mutated = sorted(
        path for path in set(baseline) & set(current)
        if baseline[path] != current[path]
    )

    if not (added or removed or mutated):
        return

    lines: list[str] = [
        "Files under src/usv_playpen were modified during the pytest session.",
        "This usually means a test wrote into the package as a side effect "
        "(e.g. patching yaml.dump but forgetting the open(path, 'w') that "
        "truncates the file before yaml.dump runs).",
        "",
    ]
    if mutated:
        lines.append("MUTATED:")
        lines.extend(f"  - {p}" for p in mutated)
    if added:
        lines.append("ADDED:")
        lines.extend(f"  + {p}" for p in added)
    if removed:
        lines.append("REMOVED:")
        lines.extend(f"  - {p}" for p in removed)

    raise AssertionError("\n".join(lines))
