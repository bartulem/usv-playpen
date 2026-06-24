# ABOUTME: YOLO11 inference adapter: detect_boxes_yolo(spec, duration, model, cfg) -> [(t,l,b,r), ...],
# ABOUTME: rendering the flipped working spec exactly like the exporter; xyxy->inclusive (t,l,b,r); clamp.
"""Inference adapter for the fine-tuned YOLO11 USV box detector.

Public surface
--------------
* :func:`load_model` — load an ultralytics YOLO ``best.pt`` once (lazy import).
* :func:`detect_boxes_yolo` — the detector function matching the box_detectors
  contract::

      detect_boxes_yolo(spec, duration, model=None, cfg=None)
          -> list[(top, left, bottom, right)]

  * ``spec`` — float ``[F, T]`` (freq=rows, time=cols), F=T=128 typically.
  * Columns ``>= duration`` are zero padding and are excluded.
  * Boxes are inclusive ``(t, l, b, r) == (y0, x0, y1, x1)`` (NOT xyxy) clamped to
    ``[0, F-1]`` x ``[0, duration-1]`` — identical convention to
    ``cc_box_detector.detect_boxes`` / :func:`common.boxes.clamp_box`.

Orientation (MUST match the exporter)
-------------------------------------
At inference the existing box-prompt path operates on ``np.flipud(working_spec)``
— the orientation SAM sees. The GT exporter renders each spec with
:func:`common.boxes.render_spec_image` AFTER ``np.flipud`` and writes labels in
that flipped space. So here we likewise **flip first**, render with the SAME
:func:`render_spec_image`, run YOLO, and return boxes in flipped space. Downstream
``box_to_xyxy`` + SAM then behave exactly as for the CC detector.

>>> CRITICAL CAVEAT — NMS can delete the calls box-prompt exists to keep separate <<<
YOLO applies Non-Maximum Suppression to its raw detections. Two USVs that overlap
in time but are stacked in frequency (or a long call partly overlapping a short
one) can produce two boxes with high IoU — and NMS will SUPPRESS one of them.
That is precisely the case box-prompted SAM2 exists to handle (one box per call,
kept separate). To mitigate, this adapter exposes:
  * ``iou`` (NMS IoU threshold) — RAISE it (e.g. 0.7-0.9) so only near-duplicate
    boxes are merged and genuinely-stacked calls survive. Default here is 0.7,
    higher than ultralytics' 0.7/0.45-ish defaults-for-detection intent, to favor
    RECALL of overlapping calls.
  * ``agnostic_nms`` — single class anyway, but kept explicit.
  * ``max_det`` — generous cap (default 300) so dense sessions aren't truncated.
**This is the #1 thing to validate** when adopting the YOLO detector: compare
per-spec box counts and stacked-call recall against the CC baseline. If YOLO
drops stacked calls, raise ``iou`` further or fall back to ``cc`` for those specs.

Lazy imports
------------
``ultralytics`` and ``cv2`` (via ``render_spec_image``) are imported lazily so this
module ``py_compile``s and imports in ``samv2_env`` (no ultralytics). ``numpy`` is
top-level.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

# Inclusive (top, left, bottom, right) = (y0, x0, y1, x1). Rows=freq, cols=time.
Box = Tuple[int, int, int, int]


# ---------------------------------------------------------------------------
# Robust import of the shared box helpers (common.boxes)
# ---------------------------------------------------------------------------
def _load_box_helpers():
    """Import ``render_spec_image``, ``tlbr_to_xyxy``, ``clamp_box`` from common.boxes.

    Works both as a package (``box_detectors.common.boxes``) and when this file is
    run from an odd cwd: falls back to putting the box_detectors PARENT dir on
    ``sys.path`` and importing ``box_detectors.common.boxes``.
    """
    try:
        from box_detectors.common.boxes import (  # type: ignore
            render_spec_image,
            tlbr_to_xyxy,
            clamp_box,
        )
        return render_spec_image, tlbr_to_xyxy, clamp_box
    except Exception:
        pass
    try:
        # Relative import when imported as box_detectors.yolo.infer.
        from ..common.boxes import (  # type: ignore
            render_spec_image,
            tlbr_to_xyxy,
            clamp_box,
        )
        return render_spec_image, tlbr_to_xyxy, clamp_box
    except Exception:
        pass
    # Last resort: add the parent of box_detectors/ to sys.path.
    here = os.path.dirname(os.path.abspath(__file__))          # .../box_detectors/yolo
    pkg_parent = os.path.dirname(os.path.dirname(here))        # parent of box_detectors
    if pkg_parent not in sys.path:
        sys.path.insert(0, pkg_parent)
    from box_detectors.common.boxes import (  # type: ignore
        render_spec_image,
        tlbr_to_xyxy,
        clamp_box,
    )
    return render_spec_image, tlbr_to_xyxy, clamp_box


# ---------------------------------------------------------------------------
# Inference config
# ---------------------------------------------------------------------------
@dataclass
class YoloDetectorConfig:
    """Knobs for :func:`detect_boxes_yolo` (mirrors ultralytics ``predict`` args).

    The defaults bias toward RECALL of overlapping/stacked USVs (the failure mode
    box-prompt cares about): a high NMS ``iou`` and a generous ``max_det``.
    """

    conf: float = 0.25            # confidence threshold (lower => more recall)
    iou: float = 0.7             # NMS IoU; RAISE to keep stacked calls (see module doc)
    agnostic_nms: bool = True     # class-agnostic NMS (single class anyway)
    max_det: int = 300            # cap on detections per spec
    imgsz: int = 128              # MUST match the rendered image size (native 128)
    colormap: str = "viridis"     # MUST match the exporter's render_spec_image colormap
    device: Optional[str] = None  # "0"/"cpu"/None(auto)
    half: bool = False            # fp16 inference (GPU only)
    verbose: bool = False         # silence ultralytics per-call logging
    # Optional: clamp very-tall full-frame false boxes, etc. Reserved for future.
    extra_predict_kwargs: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(weights: str, device: Optional[str] = None):
    """Load a fine-tuned ultralytics YOLO model from a ``best.pt`` checkpoint.

    Args:
        weights: Path to the trained checkpoint (e.g.
            ``.../runs/usv_yolo11n/weights/best.pt``). Required.
        device: ``"0"``, ``"cpu"``, or ``None`` (ultralytics auto-selects). If given,
            the model is moved to that device.

    Returns:
        An ``ultralytics.YOLO`` model ready for ``predict``.

    Raises:
        ImportError: if ultralytics is not installed (lazy import; off-cluster).
        FileNotFoundError: if ``weights`` does not exist.
    """
    try:
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover - exercised only off-cluster
        raise ImportError(
            "ultralytics is required for YOLO inference but is not installed in "
            "this environment. Install it on the cluster GPU env "
            "(`pip install ultralytics`); samv2_env does NOT have it."
        ) from exc
    if not os.path.exists(weights):
        raise FileNotFoundError(
            f"YOLO weights not found: {weights!r}. Train first "
            "(box_detectors.yolo.train) and point at {project}/{name}/weights/best.pt."
        )
    model = YOLO(weights)
    if device is not None:
        try:
            model.to(device)
        except Exception:
            # ultralytics also accepts device per-predict; ignore move failures.
            pass
    return model


# ---------------------------------------------------------------------------
# The detector function (box_detectors contract)
# ---------------------------------------------------------------------------
def detect_boxes_yolo(
    spec: np.ndarray,
    duration: int,
    model=None,
    cfg: Optional[YoloDetectorConfig] = None,
) -> List[Box]:
    """Detect one box per USV with a fine-tuned YOLO11 model.

    Mirrors the box_detectors detector signature and the orientation/clamping
    behavior of ``cc_box_detector.detect_boxes``.

    Pipeline:
      1. Flip the spec (``np.flipud``) to the SAM/exporter orientation, slice to the
         valid ``duration`` columns (drop right-edge zero padding from the image).
      2. Render to a uint8 RGB image with the SAME
         :func:`common.boxes.render_spec_image` the exporter used (pixel-exact
         train/test match).
      3. Run YOLO ``predict`` (with the recall-biased NMS config).
      4. Convert each xyxy detection -> inclusive ``(t, l, b, r)`` (x<->y swap),
         clamp to ``[0, F-1]`` x ``[0, duration-1]``, and drop boxes that fall
         entirely in the zero-pad region (``left >= duration``).

    Args:
        spec: Float ``[F, T]`` (freq=rows, time=cols). NOT pre-flipped — the flip is
            done here, exactly as ``cc_box_detector`` does it.
        duration: Valid time frames (active columns). Columns ``>= duration`` are
            zero padding.
        model: A loaded ultralytics ``YOLO`` model (from :func:`load_model`). If
            ``None``, ``cfg.extra_predict_kwargs['weights']`` (or a ``weights`` attr)
            must let us load one; otherwise a ValueError is raised — pass a model.
        cfg: :class:`YoloDetectorConfig`; defaults if ``None``.

    Returns:
        List of inclusive ``(top, left, bottom, right)`` boxes in flipped space,
        clamped. Empty list if nothing is detected or the spec is degenerate.
    """
    if cfg is None:
        cfg = YoloDetectorConfig()

    render_spec_image, _tlbr_to_xyxy, clamp_box = _load_box_helpers()

    spec = np.asarray(spec, dtype=np.float64)
    if spec.ndim != 2:
        raise ValueError(f"detect_boxes_yolo expects a 2D [F, T] spec; got {spec.shape}")
    n_freq, n_time = spec.shape

    # Clamp duration into [1, n_time] (defensive; matches cc_box_detector._clip_duration).
    d = int(duration)
    d = max(1, min(d, n_time))
    if d < 1:
        return []

    if model is None:
        # Allow a weights path passed via cfg for convenience, else demand a model.
        weights = cfg.extra_predict_kwargs.get("weights")
        if weights is None:
            raise ValueError(
                "detect_boxes_yolo requires a loaded `model` (from load_model). "
                "Pass model=load_model(weights) or put weights in "
                "cfg.extra_predict_kwargs['weights']."
            )
        model = load_model(weights, device=cfg.device)

    # 1. Flip to SAM/exporter orientation and slice off right zero-padding.
    #    (We feed only the valid [F, d] window to the model so pad pixels can't
    #     produce spurious boxes; coordinates returned are already in this window.)
    flipped = np.flipud(spec)            # [F, T]
    window = flipped[:, :d]              # [F, d]

    # 2. Render EXACTLY like the exporter (same function, same colormap).
    img = render_spec_image(window, colormap=cfg.colormap)  # uint8 [F, d, 3] RGB

    # 3. YOLO predict with recall-biased NMS.
    predict_kwargs = dict(
        source=img,
        imgsz=int(cfg.imgsz),
        conf=float(cfg.conf),
        iou=float(cfg.iou),
        agnostic_nms=bool(cfg.agnostic_nms),
        max_det=int(cfg.max_det),
        half=bool(cfg.half),
        verbose=bool(cfg.verbose),
    )
    if cfg.device is not None:
        predict_kwargs["device"] = cfg.device
    # Allow advanced overrides but never let a stray "weights" key reach predict().
    for k, v in cfg.extra_predict_kwargs.items():
        if k == "weights":
            continue
        predict_kwargs[k] = v

    results = model.predict(**predict_kwargs)
    if not results:
        return []
    res = results[0]

    boxes_xyxy = _extract_xyxy(res)  # [(x0,y0,x1,y1) float in IMAGE pixels], img = [F, d]
    if boxes_xyxy is None or len(boxes_xyxy) == 0:
        return []

    out: List[Box] = []
    for (x0, y0, x1, y1) in boxes_xyxy:
        # ultralytics xyxy: x along width (=time/cols, our window has d cols),
        # y along height (=freq/rows). Convert to inclusive integer pixel extents.
        left = int(np.floor(min(x0, x1)))
        right = int(np.ceil(max(x0, x1))) - 1   # inclusive right edge
        top = int(np.floor(min(y0, y1)))
        bottom = int(np.ceil(max(y0, y1))) - 1  # inclusive bottom edge
        if right < left:
            right = left
        if bottom < top:
            bottom = top

        # Drop boxes that lie entirely in the (now-removed) zero-pad region.
        # Since we cropped to [:, :d], any valid box has left < d; guard anyway.
        if left >= d:
            continue

        # Clamp to [0, F-1] x [0, d-1] (same range as cc_box_detector._dilate_clamp /
        # common.boxes.clamp_box). clamp_box also re-orders if needed.
        box = clamp_box((top, left, bottom, right), n_freq=n_freq, duration=d)
        out.append(box)

    return out


def _extract_xyxy(result):
    """Pull an (N, 4) array of xyxy boxes out of an ultralytics Results object.

    Returns a numpy float array or ``None``. Handles torch tensors (lazy: torch may
    not be needed) by going through ``.cpu().numpy()`` when available.
    """
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return None
    xyxy = getattr(boxes, "xyxy", None)
    if xyxy is None:
        return None
    # xyxy may be a torch.Tensor or already array-like.
    if hasattr(xyxy, "detach"):
        xyxy = xyxy.detach()
    if hasattr(xyxy, "cpu"):
        xyxy = xyxy.cpu()
    if hasattr(xyxy, "numpy"):
        xyxy = xyxy.numpy()
    arr = np.asarray(xyxy, dtype=np.float64)
    if arr.size == 0:
        return np.zeros((0, 4), dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr[:, :4]
