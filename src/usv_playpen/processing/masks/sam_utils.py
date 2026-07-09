# ABOUTME: Bounding-box helper for the box-prompt mask path.
# ABOUTME: Memory/device helpers live in specgen.common; import them from there.
from typing import Tuple

import numpy as np


def get_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """Get bounding box of mask: (top, left, bottom, right)"""
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return (0, 0, 0, 0)
    return (rows.min(), cols.min(), rows.max(), cols.max())
