"""
@author: bartulem
Tests for ``usv_playpen.processing.masks.cc_box_detector`` — the pure-NumPy
connected-component USV box detector (ridge map -> binarize -> label ->
time-gap merge -> asymmetric dilate).

The mask-generation pipeline tests (``test_generate_masks``) stub the detector
with a fake, so the real algorithm is exercised here directly on tiny synthetic
spectrograms with known bright blobs.
"""

from __future__ import annotations

import numpy as np
import pytest

from usv_playpen.processing.masks.cc_box_detector import (
    BoxDetectorConfig,
    box_to_xyxy,
    detect_boxes,
    merge_time_gaps,
    ridge_map,
)


# ---------------------------------------------------------------------------
# BoxDetectorConfig
# ---------------------------------------------------------------------------


def test_config_coerces_norm_pctl_list_to_tuple():
    """A JSON-loaded ``NORM_PCTL`` list is normalized to a tuple in ``__post_init__``."""

    cfg = BoxDetectorConfig(NORM_PCTL=[2.0, 98.0])
    assert cfg.NORM_PCTL == (2.0, 98.0)


@pytest.mark.parametrize("kwargs", [{"ROW_BG_PCTL": 60.0}, {"TIME_SMOOTH": 0}])
def test_config_rejects_invalid_fields(kwargs):
    """``ROW_BG_PCTL`` must be in [0, 50) and ``TIME_SMOOTH`` >= 1."""

    with pytest.raises(AssertionError):
        BoxDetectorConfig(**kwargs)


# ---------------------------------------------------------------------------
# box_to_xyxy — the single detector(y,x) -> SAM(x,y) swap
# ---------------------------------------------------------------------------


def test_box_to_xyxy_swaps_axes():
    """``(top, left, bottom, right)`` -> ``(x0, y0, x1, y1)`` = ``(left, top, right, bottom)``."""

    assert box_to_xyxy((1, 2, 3, 4)) == (2, 1, 4, 3)


# ---------------------------------------------------------------------------
# merge_time_gaps
# ---------------------------------------------------------------------------


def test_merge_time_gaps_unions_time_adjacent_freq_overlapping():
    """Two fragments of one call — same frequency rows, a small time gap — are
    unioned into a single bounding box."""

    cfg = BoxDetectorConfig()
    boxes = [(10, 0, 20, 10), (10, 12, 20, 20)]   # freq rows 10-20; time gap 12-10=2 <= merge_time_gap
    merged = merge_time_gaps(boxes, cfg)
    assert merged == [(10, 0, 20, 20)]


def test_merge_time_gaps_keeps_frequency_stacked_calls_separate():
    """Two time-overlapping boxes far apart in frequency (gap >> merge_freq_gap)
    are kept separate — distinct stacked calls, not one fragmented call."""

    cfg = BoxDetectorConfig()
    boxes = [(10, 0, 20, 10), (40, 0, 50, 10)]   # same time cols; freq gap 40-20=20 >> merge_freq_gap
    merged = merge_time_gaps(boxes, cfg)
    assert len(merged) == 2


# ---------------------------------------------------------------------------
# ridge_map + detect_boxes
# ---------------------------------------------------------------------------


def _blob_spec(n_freq: int = 64, n_time: int = 64) -> np.ndarray:
    """A spec (freq rows x time cols) with one graded (Gaussian) horizontal blob
    and a zero background — the canonical single-USV case. The blob is *graded*
    (not a flat bar) so the foreground has the dynamic range the Otsu / percentile
    binarization needs, and its bright core clears ``min_area_px``."""

    yy, xx = np.mgrid[0:n_freq, 0:n_time]
    spec = np.exp(-(((yy - 30) / 6.0) ** 2 + ((xx - 32) / 16.0) ** 2))  # peak 1.0 at (30, 32)
    spec[spec < 0.05] = 0.0   # zero the tails -> clear foreground blob on a zero background
    return spec


def test_ridge_map_none_on_degenerate_spec():
    """An all-zero spec fails the degenerate gate (no foreground) -> ``None``."""

    cfg = BoxDetectorConfig()
    assert ridge_map(np.zeros((64, 64)), duration=64, cfg=cfg) is None


def test_detect_boxes_finds_single_bright_blob():
    """A single bright bar yields exactly one box whose (dilated) extent covers
    the blob; ``box_to_xyxy`` then places the blob centre inside the SAM box."""

    boxes = detect_boxes(_blob_spec(), duration=64)
    assert len(boxes) == 1
    top, left, bottom, right = boxes[0]
    # blob centre (row 30, col 30) is inside the detected (dilated) box.
    assert top <= 30 <= bottom
    assert left <= 30 <= right
    x0, y0, x1, y1 = box_to_xyxy(boxes[0])
    assert (x0, y0, x1, y1) == (left, top, right, bottom)


def test_detect_boxes_empty_on_degenerate_spec():
    """A degenerate (all-zero) spec produces no boxes."""

    assert detect_boxes(np.zeros((64, 64)), duration=64) == []


def test_detect_boxes_uniform_spec_no_boxes():
    """A flat, uniform spec has zero peak-to-peak range, so the degenerate gate
    rejects it before any component is formed -> no boxes."""

    assert detect_boxes(np.ones((64, 64)), duration=64) == []
