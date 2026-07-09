"""
@author: bartulem
Tests for ``get_bbox`` — the one pure mask-geometry helper in
``usv_playpen.processing.masks.sam_utils`` still used by the live box-prompt path.

The bounding-box op is easy to get subtly wrong (index order, empty-mask edge
case), so it is pinned directly here rather than only through the
(detector-stubbed) pipeline tests.
"""

from __future__ import annotations

import numpy as np

from usv_playpen.processing.masks.sam_utils import get_bbox


def _rect(top: int, left: int, bottom: int, right: int, shape=(8, 8)) -> np.ndarray:
    """A boolean mask that is True on the inclusive-exclusive rectangle
    ``[top:bottom, left:right]``."""

    m = np.zeros(shape, dtype=bool)
    m[top:bottom, left:right] = True
    return m


def test_get_bbox_bounds_and_empty():
    """``get_bbox`` returns inclusive ``(top, left, bottom, right)``; an empty
    mask returns the all-zero degenerate box."""

    assert get_bbox(_rect(2, 3, 6, 5)) == (2, 3, 5, 4)   # rows 2..5, cols 3..4 inclusive
    assert get_bbox(np.zeros((8, 8), dtype=bool)) == (0, 0, 0, 0)
