# ABOUTME: SAM2 grid (AMG) preset configs + mask post-processing used by the grid path.
# ABOUTME: Memory/device/seed/compile helpers live in specgen.common; import them from there.
from typing import Dict, List, Tuple

import numpy as np


def get_sam2_config(config_name: str) -> Dict:
    """Get the AMG (grid) parameter dict for a named preset.

    Presets live in ``configs/mask_grid_presets.yaml`` (loaded via ``specgen.config``);
    each entry is ``{description, tradeoff, params}`` and only ``params`` (the AMG kwargs)
    is returned here. See that file for the speed/accuracy tradeoff documentation.

    Args:
        config_name: Name of the AMG preset (e.g. ultra_fast, fast, balanced, precision,
            anti_large, quality, max_quality).

    Returns:
        Dict: AMG configuration dictionary (kwargs for SAM2AutomaticMaskGenerator).

    Raises:
        ValueError: If ``config_name`` is not a known preset.
    """
    from . import _config as cfg

    presets = cfg.load_grid_presets()
    if config_name not in presets:
        raise ValueError(
            f"Unknown grid config: {config_name}. Available: {list(presets.keys())}")
    entry = presets[config_name]
    # Tolerate both the documented {description, params} shape and a flat kwargs dict.
    params = entry.get("params", entry) if isinstance(entry, dict) else entry
    return dict(params)


def post_process_masks(masks: List[Dict],
                      image_shape: Tuple[int, int],
                      spectrogram: np.ndarray = None) -> List[Dict]:
    """Post-process masks using iterative merging.

    If spectrogram is provided, uses intelligent filters.
    Otherwise, just merges overlapping masks.

    Args:
        masks: List of mask dictionaries from SAM2.
        image_shape: Tuple of (height, width) for the image.
        spectrogram: Optional spectrogram array for intelligent filtering.

    Returns:
        List[Dict]: Processed and merged masks.
    """
    if not masks:
        return masks

    if spectrogram is not None:
        return simple_pipeline(
            masks,
            spectrogram,
            filter_size=False,            # Remove tiny masks
            filter_intensity=False,       # Keep bright masks
            max_area_ratio=0.1,
            min_area_ratio=0.005
        )
    else:
        # Just merge overlapping masks
        return iterative_mask_merging(masks, overlap_method='any')


def compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Compute Intersection over Union between two masks.

    Args:
        mask1, mask2: Binary masks

    Returns:
        IoU value between 0 and 1
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    if union == 0:
        return 0.0

    return intersection / union


def masks_overlap(mask1: np.ndarray, mask2: np.ndarray,
                 method: str = 'any') -> bool:
    """
    Check if two masks overlap.

    Args:
        mask1, mask2: Binary masks
        method: 'any' (any pixel overlap), 'iou' (IoU > 0),
                'significant' (IoU > 0.1), 'bbox' (bounding boxes overlap)

    Returns:
        True if masks overlap according to method
    """
    if method == 'any' or method == 'iou':
        # Check for any pixel overlap
        return np.any(np.logical_and(mask1, mask2))

    elif method == 'significant':
        # Check for significant overlap (IoU > 0.1)
        iou = compute_iou(mask1, mask2)
        return iou > 0.1

    elif method == 'bbox':
        # Check if bounding boxes overlap
        r1, c1 = np.where(mask1)
        r2, c2 = np.where(mask2)

        if len(r1) == 0 or len(r2) == 0:
            return False

        bbox1 = (r1.min(), c1.min(), r1.max(), c1.max())
        bbox2 = (r2.min(), c2.min(), r2.max(), c2.max())

        # Check if bounding boxes overlap
        return not (bbox1[2] < bbox2[0] or bbox1[0] > bbox2[2] or
                   bbox1[3] < bbox2[1] or bbox1[1] > bbox2[3])

    else:
        raise ValueError(f"Unknown overlap method: {method}")


def merge_two_masks(mask1: np.ndarray, mask2: np.ndarray) -> np.ndarray:
    """
    Merge two masks via logical OR.

    Args:
        mask1, mask2: Binary masks

    Returns:
        Merged mask
    """
    return np.logical_or(mask1, mask2)


def iterative_mask_merging(masks: List[Dict],
                          overlap_method: str = 'any',
                          verbose: bool = False) -> List[Dict]:
    """
    Iteratively merge overlapping masks until no overlaps remain.

    This is the core algorithm:
    1. Find any pair of overlapping masks
    2. Merge them
    3. Repeat until no more overlaps

    Args:
        masks: List of mask dictionaries with 'segmentation' key
        overlap_method: How to define overlap ('any', 'iou', 'significant', 'bbox')
        verbose: Print progress

    Returns:
        List of merged masks (no overlaps)
    """
    if not masks:
        return masks

    # Convert to list of segmentation arrays
    mask_list = [m['segmentation'].copy() for m in masks]

    if verbose:
        print(f"Starting with {len(mask_list)} masks")

    iteration = 0
    while True:
        iteration += 1
        merged_any = False

        # Find first pair of overlapping masks
        n = len(mask_list)
        for i in range(n):
            for j in range(i + 1, n):
                if masks_overlap(mask_list[i], mask_list[j], overlap_method):
                    # Merge j into i
                    mask_list[i] = merge_two_masks(mask_list[i], mask_list[j])
                    # Remove j
                    mask_list.pop(j)
                    merged_any = True
                    break

            if merged_any:
                break

        if not merged_any:
            # No more overlaps found
            break

        if verbose and iteration % 10 == 0:
            print(f"  Iteration {iteration}: {len(mask_list)} masks remaining")

    if verbose:
        print(f"Completed in {iteration} iterations: {len(mask_list)} final masks")

    # Convert back to mask dictionaries
    result = []
    for seg in mask_list:
        result.append({
            'segmentation': seg,
            'area': seg.sum(),
            'bbox': get_bbox(seg)
        })

    return result


def get_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """Get bounding box of mask: (top, left, bottom, right)"""
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return (0, 0, 0, 0)
    return (rows.min(), cols.min(), rows.max(), cols.max())


def simple_pipeline(masks: List[Dict],
                   spectrogram: np.ndarray,
                   merge_overlap_method: str = 'any',
                   use_merge_constraints: bool = False,
                   filter_size: bool = True,
                   filter_intensity: bool = True,
                   min_pixels: int = 100,
                   max_area_ratio: float = .3,
                   min_area_ratio: float = .001,
                   verbose: bool = False) -> List[Dict]:
    """
    Complete simple pipeline: merge + optional filters.

    Args:
        masks: Raw masks from SAM2
        spectrogram: Original spectrogram
        merge_overlap_method: How to define overlap for merging
        use_merge_constraints: Whether to use constrained merging
        filter_size: Whether to filter by size after merging
        filter_intensity: Whether to filter by intensity after merging
        min_pixels: Minimum mask size if filter_size=True
        verbose: Print progress

    Returns:
        Processed masks
    """

    if not masks:
        return masks

    # Step 0: Filter masks by size (we know one spec is full background, some are super small)
    max_area = max_area_ratio * spectrogram.shape[0] * spectrogram.shape[1]
    min_area = min_area_ratio * spectrogram.shape[0] * spectrogram.shape[1]
    masks = [m for m in masks if m['area'] <= max_area and m['area'] >= min_area]

    if verbose:
        print(f"\n=== Simple Pipeline ===")
        print(f"Input: {len(masks)} masks")

    # Step 1: Merge overlapping masks
    if verbose:
        print("\nStep 1: Merging overlapping masks...")
    merged = iterative_mask_merging(
        masks,
        overlap_method=merge_overlap_method,
        verbose=verbose
    )

    if verbose:
        print(f"After merging: {len(merged)} masks")

    if verbose:
        print(f"\nFinal: {len(merged)} masks")
        print("=" * 50)

    return merged
