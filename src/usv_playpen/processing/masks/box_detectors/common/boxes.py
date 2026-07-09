# ABOUTME: Pure numpy box utilities: clamping, format converters (YOLO xywhn / SAM xyxy),
# ABOUTME: and canonical spec->uint8 rendering.
"""Shared, dependency-light box / rendering helpers for box_detectors.

NO torch, NO ultralytics, NO segmentation_models_pytorch here — only numpy
and (optionally, for viridis) cv2. Keeping this module light means every
detector (exporter, U-Net, YOLO) and the unit tests can import it in any
cluster env or on the GPU-less dev host.

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

from typing import Tuple

import numpy as np

# Inclusive (top, left, bottom, right) = (y0, x0, y1, x1). Rows=freq, cols=time.
Box = Tuple[int, int, int, int]


# Clamping
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


# Format converters (YOLO + SAM need these)
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


# Canonical spectrogram -> uint8 image rendering
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
