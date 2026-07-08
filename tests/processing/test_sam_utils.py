"""
@author: bartulem
Tests for the pure mask-geometry helpers in
``usv_playpen.processing.masks.sam_utils`` — IoU, overlap tests, pairwise merge,
bounding box, and the iterative overlap-merging loop.

These boolean-array ops feed the SAM2 mask post-processing and are easy to get
subtly wrong (index order, empty-mask edge cases), so they are pinned directly
here rather than only through the (detector-stubbed) pipeline tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from usv_playpen.processing.masks.sam_utils import (
    compute_iou,
    get_bbox,
    iterative_mask_merging,
    masks_overlap,
    merge_two_masks,
)


def _rect(top: int, left: int, bottom: int, right: int, shape=(8, 8)) -> np.ndarray:
    """A boolean mask that is True on the inclusive-exclusive rectangle
    ``[top:bottom, left:right]``."""

    m = np.zeros(shape, dtype=bool)
    m[top:bottom, left:right] = True
    return m


# ---------------------------------------------------------------------------
# compute_iou
# ---------------------------------------------------------------------------


def test_compute_iou_identical_disjoint_partial():
    """IoU is 1.0 for identical masks, 0.0 for disjoint masks, and
    intersection/union for a partial overlap."""

    a = _rect(0, 0, 4, 4)                     # 16 px
    assert compute_iou(a, a) == pytest.approx(1.0)
    assert compute_iou(_rect(0, 0, 2, 2), _rect(4, 4, 6, 6)) == pytest.approx(0.0)
    # a=[0:4,0:4], b=[2:6,2:6] overlap [2:4,2:4]=4; union = 16+16-4 = 28.
    assert compute_iou(a, _rect(2, 2, 6, 6)) == pytest.approx(4 / 28)


def test_compute_iou_two_empty_masks_is_zero():
    """Two empty masks have an empty union -> IoU 0.0 (no division by zero)."""

    empty = np.zeros((8, 8), dtype=bool)
    assert compute_iou(empty, empty) == 0.0


# ---------------------------------------------------------------------------
# get_bbox
# ---------------------------------------------------------------------------


def test_get_bbox_bounds_and_empty():
    """``get_bbox`` returns inclusive ``(top, left, bottom, right)``; an empty
    mask returns the all-zero degenerate box."""

    assert get_bbox(_rect(2, 3, 6, 5)) == (2, 3, 5, 4)   # rows 2..5, cols 3..4 inclusive
    assert get_bbox(np.zeros((8, 8), dtype=bool)) == (0, 0, 0, 0)


# ---------------------------------------------------------------------------
# masks_overlap
# ---------------------------------------------------------------------------


def test_masks_overlap_any_vs_disjoint():
    """``'any'`` is True on a single shared pixel and False for disjoint masks."""

    assert bool(masks_overlap(_rect(0, 0, 4, 4), _rect(3, 3, 6, 6), method='any')) is True
    assert bool(masks_overlap(_rect(0, 0, 2, 2), _rect(4, 4, 6, 6), method='any')) is False


def test_masks_overlap_significant_threshold():
    """``'significant'`` requires IoU > 0.1: a small (IoU 0.03) overlap that
    ``'any'`` accepts is rejected."""

    a = _rect(0, 0, 4, 4)
    b = _rect(3, 3, 7, 7)   # overlap [3:4,3:4]=1; union=16+16-1=31 -> IoU ~0.032
    assert bool(masks_overlap(a, b, method='any')) is True
    assert bool(masks_overlap(a, b, method='significant')) is False


# ---------------------------------------------------------------------------
# merge_two_masks + iterative_mask_merging
# ---------------------------------------------------------------------------


def test_merge_two_masks_is_union():
    """Merging two masks is their logical OR."""

    merged = merge_two_masks(_rect(0, 0, 2, 8), _rect(6, 0, 8, 8))
    assert merged.sum() == 32   # 16 + 16, disjoint rows
    np.testing.assert_array_equal(merged, _rect(0, 0, 2, 8) | _rect(6, 0, 8, 8))


def test_iterative_mask_merging_collapses_overlapping_keeps_disjoint():
    """Two overlapping masks collapse into one union mask; a third disjoint mask
    survives untouched. Output dicts carry 'segmentation', 'area', 'bbox'."""

    masks = [
        {'segmentation': _rect(0, 0, 4, 4)},
        {'segmentation': _rect(2, 2, 6, 6)},   # overlaps the first
        {'segmentation': _rect(0, 6, 2, 8)},   # disjoint from both
    ]
    out = iterative_mask_merging(masks, overlap_method='any')
    assert len(out) == 2
    assert {'segmentation', 'area', 'bbox'} <= set(out[0])
    # the merged mask is the union of the two overlapping rects.
    union = _rect(0, 0, 4, 4) | _rect(2, 2, 6, 6)
    assert any(np.array_equal(m['segmentation'], union) for m in out)


def test_iterative_mask_merging_empty_input():
    """An empty mask list is returned unchanged."""

    assert iterative_mask_merging([]) == []
