# ABOUTME: Pure numpy/scipy/skimage box utilities: GT-mask -> instances -> (t,l,b,r) bboxes,
# ABOUTME: clamping, format converters (YOLO xywhn / SAM xyxy), and canonical spec->uint8 rendering.
"""Shared, dependency-light box / mask / rendering helpers for box_detectors.

NO torch, NO ultralytics, NO segmentation_models_pytorch here — only numpy,
scipy.ndimage, skimage, and (optionally, for viridis) cv2. Keeping this module
light means every detector (exporter, U-Net, YOLO) and the unit tests can import
it in any cluster env or on the GPU-less dev host.

Box-format contract
-------------------
A box is the inclusive tuple ``(top, left, bottom, right) == (y0, x0, y1, x1)`` —
the SAME convention as
``spec_gen_full_pipeline/src/specgen/masks/cc_box_detector.py``
(``detect_boxes`` / ``get_bbox`` = ``(rows.min, cols.min, rows.max, cols.max)``).
Rows are FREQUENCY, columns are TIME. ``top <= bottom`` and ``left <= right``.

This is NOT xyxy. SAM2's ``predict(box=...)`` wants XYXY ``(x0, y0, x1, y1)``;
that swap happens downstream (``cc_box_detector.box_to_xyxy`` /
:func:`tlbr_to_xyxy` here), in exactly one place, to avoid tuple-index-shift bugs.

Orientation convention (IMPORTANT — read before training/inference)
-------------------------------------------------------------------
At inference the existing box-prompt path operates on ``np.flipud(working_spec)``
— the orientation SAM actually sees (low frequency at the BOTTOM, like a
conventional spectrogram image). ``cc_box_detector.detect_boxes`` is documented to
run in that SAME flipped space, and so must every detector here.

Concretely:
  * The GT exporter renders each spec with :func:`render_spec_image` AFTER
    ``np.flipud``; the GT masks on disk are already in image (flipped) orientation,
    so exported boxes live in flipped space.
  * U-Net / YOLO are therefore trained on flipped images with flipped-space boxes.
  * At inference, a detector receives the spec, flips it (``np.flipud``) to match,
    runs, and returns flipped-space ``(t,l,b,r)`` boxes — identical handling to
    ``cc_box_detector`` — so downstream ``box_to_xyxy`` + SAM behave unchanged.

The helpers in THIS module are orientation-agnostic (they operate on whatever
array you pass). The flip is the caller's responsibility; it is performed once,
consistently, in the exporter and in each detector's ``detect_boxes_*`` wrapper.
The single canonical rendering used by BOTH the exporter and YOLO inference is
:func:`render_spec_image`, so train-time and test-time images match exactly.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy import ndimage

# Inclusive (top, left, bottom, right) = (y0, x0, y1, x1). Rows=freq, cols=time.
Box = Tuple[int, int, int, int]


# ---------------------------------------------------------------------------
# GT mask -> per-instance boolean masks -> bounding boxes
# ---------------------------------------------------------------------------
def mask_to_instances(mask: np.ndarray, min_area: int = 0) -> List[np.ndarray]:
    """Split a GT mask into a list of per-instance boolean masks.

    Handles BOTH GT mask flavours produced by the SAM2 finetune dataset:

    * **Instance-ID (VOS-style) masks** — a single-channel integer image where
      each distinct nonzero pixel value is one instance ID (0 = background).
      Each unique nonzero value becomes one instance mask. This is detected when
      the mask has more than two distinct values (i.e. not purely binary).
    * **Binary masks** — only {0, nonzero}. Instances are recovered as
      8-connected components via :func:`scipy.ndimage.label`.

    A 3-channel (HxWxC) mask is reduced to a single label channel first:
    if all channels are equal it is collapsed; otherwise the max over channels
    is used (RGB-encoded instance maps still separate by value).

    Args:
        mask: ``[H, W]`` or ``[H, W, C]`` array (any integer/float dtype). Pixel
            values are interpreted as labels (instance-ID) or foreground (binary).
        min_area: Drop instances/components with fewer than this many pixels. The
            key knob for BINARY masks, whose 8-connected components otherwise
            include single-pixel noise fragments (one scattered foreground can
            explode into 100+ tiny "instances"). Mirrors
            ``cc_box_detector.BoxDetectorConfig.min_area_px`` (≈20). 0 = no filter.

    Returns:
        List of boolean ``[H, W]`` arrays, one per instance. Empty if the mask
        has no foreground (or all instances fall below ``min_area``). Order is by
        ascending instance ID (instance-ID masks) or connected-component label
        (binary masks).
    """
    arr = np.asarray(mask)
    if arr.ndim == 3:
        # Collapse channels: identical channels -> one; else max keeps distinct IDs.
        if arr.shape[2] >= 1 and np.all(arr == arr[:, :, :1]):
            arr = arr[:, :, 0]
        else:
            arr = arr.max(axis=2)
    if arr.ndim != 2:
        raise ValueError(f"mask must be 2D (or HxWxC); got shape {np.asarray(mask).shape}")

    # Round to integer labels (PNG masks are integer-valued but may load as float).
    lab_img = np.rint(arr).astype(np.int64)
    if lab_img.max() <= 0:
        return []

    nonzero_vals = np.unique(lab_img[lab_img > 0])

    # Binary mask (single foreground value) -> 8-connected components.
    if nonzero_vals.size <= 1:
        structure = np.ones((3, 3), dtype=int)  # 8-connectivity
        labeled, n_comp = ndimage.label(lab_img > 0, structure=structure)
        instances = [labeled == k for k in range(1, n_comp + 1)]
    else:
        # Instance-ID mask -> one mask per unique nonzero value.
        instances = [lab_img == v for v in nonzero_vals]

    if min_area > 0:
        instances = [m for m in instances if int(m.sum()) >= int(min_area)]
    return instances


def instance_bbox(instance_mask: np.ndarray) -> Box:
    """Inclusive ``(top, left, bottom, right)`` bbox of a boolean instance mask.

    Args:
        instance_mask: Boolean (or truthy) ``[H, W]`` array with >= 1 True pixel.

    Returns:
        ``(top, left, bottom, right)`` = ``(rows.min, cols.min, rows.max,
        cols.max)``, inclusive — matching ``cc_box_detector.get_bbox``.

    Raises:
        ValueError: if the mask is empty (no True pixel).
    """
    m = np.asarray(instance_mask).astype(bool)
    rows = np.any(m, axis=1)
    cols = np.any(m, axis=0)
    if not rows.any() or not cols.any():
        raise ValueError("instance_bbox: empty instance mask (no True pixels)")
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return (int(rmin), int(cmin), int(rmax), int(cmax))


def masks_to_boxes(mask: np.ndarray, min_area: int = 0) -> List[Box]:
    """Full GT-mask -> list of inclusive ``(t,l,b,r)`` boxes.

    Combines :func:`mask_to_instances` and :func:`instance_bbox`. Instances with
    no pixels are skipped defensively (should not happen for label-derived masks).

    Args:
        mask: GT mask, ``[H, W]`` or ``[H, W, C]`` (instance-ID or binary).
        min_area: Drop instances smaller than this many pixels (see
            :func:`mask_to_instances`). Use ~20 to suppress binary-mask noise
            fragments; 0 = no filter.

    Returns:
        List of ``(top, left, bottom, right)`` boxes (one per kept instance).
    """
    boxes: List[Box] = []
    for inst in mask_to_instances(mask, min_area=min_area):
        if not inst.any():
            continue
        boxes.append(instance_bbox(inst))
    return boxes


# ---------------------------------------------------------------------------
# Clamping
# ---------------------------------------------------------------------------
def clamp_box(box: Box, n_freq: int, duration: int) -> Box:
    """Clamp a box to ``[0, n_freq-1]`` (rows/freq) x ``[0, duration-1]`` (cols/time).

    Mirrors the clamp range used by ``cc_box_detector._dilate_clamp``: the right
    edge is clamped to ``duration-1`` (NOT ``n_time-1``) so prompts never extend
    into the right-edge zero padding. Also enforces ``top <= bottom`` and
    ``left <= right`` after clamping.

    Args:
        box: ``(top, left, bottom, right)`` inclusive.
        n_freq: Number of frequency rows ``F`` (so valid rows are ``[0, F-1]``).
        duration: Valid time frames (so valid cols are ``[0, duration-1]``).

    Returns:
        Clamped ``(top, left, bottom, right)``.
    """
    t, l, b, r = box
    hi_row = max(0, int(n_freq) - 1)
    hi_col = max(0, int(duration) - 1)
    t = int(min(max(0, int(t)), hi_row))
    b = int(min(max(0, int(b)), hi_row))
    l = int(min(max(0, int(l)), hi_col))
    r = int(min(max(0, int(r)), hi_col))
    if b < t:
        t, b = b, t
    if r < l:
        l, r = r, l
    return (t, l, b, r)


# ---------------------------------------------------------------------------
# Format converters (YOLO + SAM need these)
# ---------------------------------------------------------------------------
def tlbr_to_xywhn(box: Box, W: int, H: int) -> Tuple[float, float, float, float]:
    """Convert inclusive ``(t,l,b,r)`` -> YOLO normalized ``(xc, yc, w, h)``.

    Ultralytics/YOLO label files use normalized box centre + size, with x along
    the IMAGE WIDTH (columns/time) and y along the IMAGE HEIGHT (rows/freq).
    Because the detector box is inclusive, a 1-pixel box has width/height = 1
    pixel (``right - left + 1``), so we add 1 before normalizing.

    Args:
        box: ``(top, left, bottom, right)`` inclusive, in pixels.
        W: Image width  (number of columns / time frames).
        H: Image height (number of rows / frequency bins).

    Returns:
        ``(x_center, y_center, width, height)`` all normalized to ``[0, 1]``.
    """
    t, l, b, r = box
    W = float(W)
    H = float(H)
    box_w = (r - l + 1)
    box_h = (b - t + 1)
    xc = (l + r + 1) / 2.0 / W   # centre of inclusive span [l, r]
    yc = (t + b + 1) / 2.0 / H
    return (xc, yc, box_w / W, box_h / H)


def tlbr_to_xyxy(box: Box) -> Tuple[int, int, int, int]:
    """Convert detector ``(top, left, bottom, right)`` -> SAM XYXY ``(x0, y0, x1, y1)``.

    Identical semantics to ``cc_box_detector.box_to_xyxy``: the SINGLE place x and
    y are swapped for SAM. Detector boxes are ``(y0, x0, y1, x1)``; SAM wants
    ``(x0, y0, x1, y1)``.

    Args:
        box: ``(top, left, bottom, right)`` inclusive.

    Returns:
        ``(x0, y0, x1, y1)`` inclusive, ready for ``SAM2ImagePredictor.predict``.
    """
    t, l, b, r = box
    return (int(l), int(t), int(r), int(b))


# ---------------------------------------------------------------------------
# Canonical spectrogram -> uint8 image rendering
# ---------------------------------------------------------------------------
def render_spec_image(spec: np.ndarray, colormap: str = "viridis") -> np.ndarray:
    """Min-max normalize a ``[F, T]`` spec and render it to a uint8 ``HxWx3`` image.

    This is the CANONICAL rendering reused by BOTH the GT exporter and YOLO
    inference, so the pixels a model sees at train time and test time are
    byte-for-byte identical. (The U-Net path consumes the spec tensor directly
    and does not strictly need this, but uses the same normalization for
    consistency.) Reuse this function rather than re-rendering ad hoc.

    Normalization is global min-max over the array passed in. NOTE: the caller is
    responsible for slicing to the valid ``duration`` and for the ``np.flipud``
    orientation (see module docstring) BEFORE calling this — render the exact
    array you want pixels for.

    Args:
        spec: Float array ``[F, T]`` (freq=rows, time=cols). Non-finite values are
            treated as the array minimum.
        colormap: ``"viridis"`` (default) or ``"gray"``/``"grayscale"``
            (channel-replicated grayscale). The viridis path reproduces spec_gen's
            EXACT box-prompt rendering — ``matplotlib.get_cmap('viridis')(norm)`` —
            so the pixels a detector sees at inference match the pre-rendered
            training images and the SAM image. (Falls back to cv2's COLORMAP_VIRIDIS,
            then grayscale, if matplotlib is unavailable; the two viridis LUTs differ
            by < JPEG noise.)

    Returns:
        ``uint8`` array of shape ``[F, T, 3]`` in RGB channel order.
    """
    arr = np.asarray(spec, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"render_spec_image expects a 2D [F, T] spec; got {arr.shape}")
    arr = np.where(np.isfinite(arr), arr, np.nan)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.uint8)
    lo = float(finite.min())
    hi = float(finite.max())
    arr = np.where(np.isfinite(arr), arr, lo)
    if hi - lo > 0:
        norm = (arr - lo) / (hi - lo)
    else:
        norm = np.zeros_like(arr)
    u8 = np.clip(norm * 255.0, 0, 255).astype(np.uint8)

    cm = colormap.lower()
    if cm in ("gray", "grayscale", "grey"):
        return np.repeat(u8[:, :, None], 3, axis=2)

    if cm == "viridis":
        # Preferred: matplotlib viridis applied to the FLOAT min-max-normed spec,
        # byte-for-byte identical to spec_gen's box-prompt rendering
        #   image = (plt.get_cmap('viridis')(spec_norm)[:, :, :3] * 255).astype(uint8)
        # (boxprompt_utils.process_single_spec_boxprompt). This is what the SAM image
        # and the pre-rendered training set use, so train/inference pixels match.
        try:
            try:
                from matplotlib import colormaps as _mpl_colormaps  # mpl >= 3.6
                _cmap = _mpl_colormaps["viridis"]
            except Exception:
                from matplotlib.cm import get_cmap as _get_cmap  # older mpl
                _cmap = _get_cmap("viridis")
            return (np.asarray(_cmap(norm))[:, :, :3] * 255).astype(np.uint8)
        except Exception:
            pass
        # Fallback: cv2's COLORMAP_VIRIDIS (differs from mpl by < JPEG noise).
        try:
            import cv2  # lazy: keep this module importable without cv2
            bgr = cv2.applyColorMap(u8, cv2.COLORMAP_VIRIDIS)  # uint8 HxWx3, BGR
            return bgr[:, :, ::-1].copy()  # -> RGB
        except Exception:
            # Last resort: grayscale if neither matplotlib nor cv2 is available.
            return np.repeat(u8[:, :, None], 3, axis=2)

    raise ValueError(f"Unknown colormap {colormap!r}; use 'viridis' or 'gray'.")
