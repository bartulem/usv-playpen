from __future__ import annotations

import hashlib
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

from usv_playpen import os_utils

# If running under WSL
# Ensure Qt uses a headless platform during tests (no X server required).
# This must be set before the first QApplication is created.
if sys.platform.startswith("linux") and "microsoft" in os.uname().release.lower():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ---------------------------------------------------------------------------
# Source-tree integrity baseline
#
# We snapshot SHA-256 hashes of every file under ``src/usv_playpen`` at the
# very start of the pytest session, and ``tests/test_src_integrity.py``
# (forced to run last by ``pytest_collection_modifyitems`` below) compares
# the post-run state against this baseline. Any test that mutates a tracked
# package file as a side effect -- the way
# ``test_conduct_recording_cli_invokes_controller_and_dumps_metadata``
# silently zeroed ``_config/_metadata.yaml`` on every run between
# commits a87717a and 75b0ab2 and shipped the empty YAML in v0.10.1 /
# v0.10.2 -- will now fail loudly with a per-file diff instead of leaving
# a trail of corrupted artifacts in working trees and releases.
# ---------------------------------------------------------------------------

SRC_TREE_ROOT = Path(__file__).resolve().parent.parent / "src" / "usv_playpen"

# Generated / cache artifacts that legitimately fluctuate during a test
# run; never include them in the integrity hash.
_INTEGRITY_EXCLUDE_DIRS = {"__pycache__", ".pytest_cache", ".mypy_cache", "__marimo__"}
_INTEGRITY_EXCLUDE_FILES = {"_version.py", ".DS_Store"}
_INTEGRITY_EXCLUDE_SUFFIXES = {".pyc", ".pyo"}


def _hash_src_tree() -> dict[str, str]:
    """
    Description
    -----------
    Walks ``SRC_TREE_ROOT`` and returns a mapping of relative path -> SHA-256
    hex digest of file contents for every tracked package file, skipping
    runtime / build artifacts (see ``_INTEGRITY_EXCLUDE_*``).

    Parameters
    ----------

    Returns
    -------
    digests (dict[str, str])
        Keys are paths relative to ``SRC_TREE_ROOT`` (POSIX separators);
        values are 64-character lowercase SHA-256 hex digests.
    """

    digests: dict[str, str] = {}
    for path in SRC_TREE_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in _INTEGRITY_EXCLUDE_DIRS for part in path.parts):
            continue
        if path.name in _INTEGRITY_EXCLUDE_FILES:
            continue
        if path.suffix in _INTEGRITY_EXCLUDE_SUFFIXES:
            continue
        rel = path.relative_to(SRC_TREE_ROOT).as_posix()
        digests[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    return digests


def pytest_sessionstart(session):
    """
    Description
    -----------
    Pytest session hook -- captures the SHA-256 baseline of every file
    under ``src/usv_playpen`` before any test runs and stashes it on
    ``session.config`` for the integrity test to read at session end.

    Parameters
    ----------
    session (pytest.Session)
        Pytest session object.

    Returns
    -------
    None
    """

    session.config._src_tree_baseline = _hash_src_tree()


def pytest_collection_modifyitems(config, items):
    """
    Description
    -----------
    Ensure ``tests/test_src_integrity.py`` always runs LAST so the file
    integrity comparison observes the cumulative effect of every other
    test (including ones that legitimately write into the package -- e.g.
    GUI startup re-saving ``_metadata.yaml`` after
    ``sync_equipment_dynamic_fields`` -- AND ones that mutate it as an
    unintended side effect, which is what we want to surface).

    Parameters
    ----------
    config (pytest.Config)
        Pytest configuration object (unused).
    items (list[pytest.Item])
        Collected test items; reordered in place.

    Returns
    -------
    None
    """

    integrity_items = []
    other_items = []
    for item in items:
        if "test_src_integrity" in item.nodeid:
            integrity_items.append(item)
        else:
            other_items.append(item)
    items[:] = other_items + integrity_items


def write_qlvm_category_bundle(directory: Path, resolution: int = 20) -> Path:
    """
    Description
    -----------
    Writes a small synthetic QLVM category bundle in the build-qlvm-categories
    layout (``category_grids.npz``, ``category_nomenclature.json``,
    ``build_config.json``): four quadrant categories R-1..R-4 on a periodic
    ``resolution`` x ``resolution`` grid indexed ``[y, x]`` (R-1 bottom-left,
    R-2 bottom-right, R-3 top-left, R-4 top-right), full agreement, a density that
    grows along x, and label positions at the quadrant centres.

    Parameters
    ----------
    directory (Path)
        Directory to write into (created if missing).
    resolution (int)
        Pixels per side; even.

    Returns
    -------
    directory (Path)
        The bundle directory.
    """

    directory.mkdir(parents=True, exist_ok=True)
    half = resolution // 2
    label_grid = np.ones((resolution, resolution), dtype=np.int16)
    label_grid[:half, half:] = 2
    label_grid[half:, :half] = 3
    label_grid[half:, half:] = 4
    density = np.tile(np.linspace(0.5, 2.0, resolution), (resolution, 1))
    np.savez_compressed(
        directory / "category_grids.npz",
        label_grid=label_grid,
        agreement=np.ones((resolution, resolution)),
        reference_grid=label_grid.copy(),
        content_change=np.zeros((resolution, resolution)),
        basins=label_grid.astype(np.int32),
        density=density,
    )
    centres = {1: (0.25, 0.25), 2: (0.75, 0.25), 3: (0.25, 0.75), 4: (0.75, 0.75)}
    descriptions = ["simple", "mixed", "complex", "wide-band two-mask"]
    nomenclature = {
        "grid_resolution": resolution,
        "n_categories": 4,
        "uncertain_agreement": 0.6,
        "n_votes": 10,
        "n_calls": 400,
        "categories": [
            {"grid_label": label, "name": f"R-{label}", "description": descriptions[label - 1], "n_calls": 100,
             "share": 0.25, "uncertain_share": 0.0, "label_x": centres[label][0], "label_y": centres[label][1]}
            for label in range(1, 5)
        ],
    }
    (directory / "category_nomenclature.json").write_text(json.dumps(nomenclature))
    build_config = {
        "build_qlvm_categories": {"grid_resolution": resolution, "n_categories": 4},
        "positions_file": "/synthetic/positions.csv",
        "positions_file_sha256": "0123456789abcdef" * 4,
        "properties_file": "/synthetic/properties.npz",
        "n_calls": 400,
        "n_sessions": 4,
        "n_basins": 4,
        "n_votes": 10,
        "built": "2026-10-03T12:00:00",
    }
    (directory / "build_config.json").write_text(json.dumps(build_config))
    return directory


@pytest.fixture
def qlvm_category_bundle(tmp_path, monkeypatch):
    """
    Description
    -----------
    A synthetic QLVM category bundle (:func:`write_qlvm_category_bundle`) that
    ``os_utils.load_qlvm_category_bundle`` -- the one loader every category-grid
    consumer goes through -- reads for the duration of the test, instead of the
    production bundle ``os_utils.QLVM_CATEGORY_BUNDLE_DIRECTORY``.

    Parameters
    ----------
    tmp_path (Path)
        Pytest's per-test directory.
    monkeypatch (pytest.MonkeyPatch)
        Pytest's monkeypatch fixture.

    Returns
    -------
    directory (Path)
        The bundle directory.
    """

    directory = write_qlvm_category_bundle(tmp_path / "category_bundle")
    monkeypatch.setattr(os_utils, "QLVM_CATEGORY_BUNDLE_DIRECTORY", str(directory))
    return directory
