# ABOUTME: Box-prompt SAM2 inference for USV spectrograms -- detector boxes -> SAM2ImagePredictor.predict(box).
# ABOUTME: The "boxprompt" alternative to the grid/AMG path in specgen.masks.segment; same output schema.

"""Box-prompt SAM2 mask generation for the spec_gen_full_pipeline Step 2.

This is the more sophisticated alternative to the grid path
(``SAM2AutomaticMaskGenerator``). It is selected from ``specgen.masks.segment`` via
``--method boxprompt`` and ported from ``sam2_pred/box_prompt/`` (see that directory's
``README.md`` for the validated design rationale and a standalone debugging/diagnostic
tool with paired-row example grids).

Pipeline per spectrogram:
  1. flipud + min-max normalize + viridis -> RGB uint8   (identical to the grid path)
  2. ``cc_box_detector.detect_boxes`` on the flipped working spec -> instance boxes
  3. ``predictor.set_image(image)`` ONCE, then loop ``predict(box=...)`` per box
     (the Hiera image encoder runs once; only the prompt-encoder+decoder rerun per box)
  4. select the ``argmax(iou_predictions)`` mask from ``multimask_output=True``
  5. emit ONE ``{'segmentation', 'area', 'bbox'}`` dict per box -- NO 'any'-overlap merge

Why not the grid path: AMG over-proposes then patches up with an ``'any'``-overlap merge
that re-fuses temporally overlapping USVs -- exactly the calls we want kept separate.
Box-prompting emits one instance per detected box, so overlapping calls stay distinct.

Output schema is identical to the grid path: a list of ``{'segmentation', 'area', 'bbox'}``
dicts per spec index (plus ``iou_pred``/``box`` for QA), so downstream
``mmmmb.mask_utils.load_sam_masks`` / ``sam_to_binary_mask`` and ``vae_inference.py`` keep
working unmodified.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import ndimage
from tqdm import tqdm

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from .sam_utils import get_bbox
from .cc_box_detector import BoxDetectorConfig, detect_boxes, box_to_xyxy
from ._common_memory import cleanup_memory, log_memory_usage


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_predictor(model_cfg: str, checkpoint: str, device: torch.device,
                    logger: logging.Logger) -> SAM2ImagePredictor:
    """Build a SAM2ImagePredictor around the (fine-tuned) checkpoint.

    Args:
        model_cfg: SAM2 model config path (relative to the SAM2 model dir).
        checkpoint: SAM2 checkpoint path (relative to the SAM2 model dir).
        device: Compute device.
        logger: Logger instance.

    Returns:
        SAM2ImagePredictor: Ready for ``set_image`` / ``predict``.
    """
    logger.info(f"Loading SAM2 model: {checkpoint}")
    logger.info(f"Model config: {model_cfg}")
    sam2 = build_sam2(model_cfg, checkpoint, device=device, apply_postprocessing=False)
    predictor = SAM2ImagePredictor(sam2)
    logger.info("SAM2ImagePredictor initialized")
    return predictor


# ---------------------------------------------------------------------------
# Per-spectrogram box-prompt inference
# ---------------------------------------------------------------------------
_CC8 = np.ones((3, 3), dtype=int)  # 8-connectivity


def _split_mask_components(seg: np.ndarray, min_px: int) -> List[np.ndarray]:
    """Split a boolean mask into 8-connected components >= ``min_px``.

    Returns ``[seg]`` unchanged if the mask is a single component (the common case),
    so a connected USV is never artificially split -- only genuinely disconnected blobs
    captured under one box become separate instances.
    """
    lab, n = ndimage.label(seg, structure=_CC8)
    if n <= 1:
        return [seg]
    subs = [(lab == c) for c in range(1, n + 1)]
    subs = [s for s in subs if int(s.sum()) >= min_px]
    return subs if subs else [seg]


def _box_mask(box, shape) -> np.ndarray:
    """Boolean mask of an inclusive (top, left, bottom, right) box."""
    t, l, b, r = box
    m = np.zeros(shape, dtype=bool)
    m[t:b + 1, l:r + 1] = True
    return m


def _box_area(box) -> int:
    """Pixel area of an inclusive (top, left, bottom, right) box."""
    t, l, b, r = box
    return (b - t + 1) * (r - l + 1)


def _uncovered_frac(mask: np.ndarray, covered: np.ndarray) -> float:
    """Fraction of ``mask`` pixels not already in ``covered``."""
    a = int(mask.sum())
    if a == 0:
        return 0.0
    return 1.0 - int((mask & covered).sum()) / a


def merge_overlapping_instances(masks: List[Dict], iou_thresh: float = 0.5,
                                containment_thresh: float = 0.8) -> List[Dict]:
    """Conservatively merge instance masks that occupy essentially the SAME region.

    Two instances are unioned iff their IoU >= ``iou_thresh`` OR their containment
    (intersection / smaller area) >= ``containment_thresh``. This targets over-segmentation
    (near-duplicate masks from overlapping boxes or residual rounds) while LEAVING two
    genuinely distinct calls that merely cross -- they share few pixels relative to their
    size, so both ratios stay low. Disjoint instances (no shared pixel) are never merged.
    Iterates to a fixed point.
    """
    if len(masks) <= 1:
        return masks

    items = [{"seg": np.asarray(m["segmentation"], dtype=bool),
              "iou_pred": float(m.get("iou_pred", 0.0)),
              "boxes": ([m["box"]] if m.get("box") is not None else [])}
             for m in masks]

    changed = True
    while changed and len(items) > 1:
        changed = False
        n = len(items)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = items[i]["seg"], items[j]["seg"]
                inter = int(np.logical_and(a, b).sum())
                if inter == 0:
                    continue
                aa, bb = int(a.sum()), int(b.sum())
                iou = inter / (aa + bb - inter) if (aa + bb - inter) else 0.0
                contain = inter / min(aa, bb) if min(aa, bb) else 0.0
                if iou >= iou_thresh or contain >= containment_thresh:
                    items[i]["seg"] = np.logical_or(a, b)
                    items[i]["iou_pred"] = max(items[i]["iou_pred"], items[j]["iou_pred"])
                    items[i]["boxes"] += items[j]["boxes"]
                    items.pop(j)
                    changed = True
                    break
            if changed:
                break

    out: List[Dict] = []
    for it in items:
        seg = it["seg"]
        out.append({
            "segmentation": seg,
            "area": int(seg.sum()),
            "bbox": get_bbox(seg),
            "iou_pred": it["iou_pred"],
            "box": it["boxes"][0] if it["boxes"] else None,
            "n_merged": len(it["boxes"]),
        })
    return out


def process_single_spec_boxprompt(
    spec: np.ndarray,
    duration: int,
    predictor: SAM2ImagePredictor,
    detector_cfg: Optional[BoxDetectorConfig],
    cmap,
    multimask_output: bool = True,
    iou_floor: float = 0.70,
    drop_below_iou: bool = False,
    tiny_mask_floor_px: int = 12,
    split_disconnected: bool = True,
    max_iters: int = 1,
    merge_instances: bool = True,
    merge_iou: float = 0.5,
    merge_containment: float = 0.8,
    detect_fn=None,
    mask_intensity_floor: float = 0.0,
    min_box_area: int = 0,
    logger: Optional[logging.Logger] = None,
) -> List[Dict]:
    """Detect boxes, prompt SAM2 per box, return one instance mask dict per kept box.

    Returns a list of ``{'segmentation': bool[H,W], 'area': int, 'bbox': (t,l,b,r),
    'iou_pred': float, 'box': (t,l,b,r)}`` dicts. Only ``segmentation`` (and optionally
    ``area``) is read downstream; the extra keys aid debugging/QA.

    Two levers split a single box that actually contains several discrete parts:

    * ``split_disconnected`` (default True): if a returned SAM mask has multiple
      8-connected components (each >= ``tiny_mask_floor_px``), emit each as its own
      instance -- so two disconnected blobs under one box become two instances.
    * ``max_iters`` > 1: iterative residual prompting. After a pass, the already-segmented
      pixels are suppressed in the detector's input and boxes are re-detected on the
      remainder, then prompted again -- recovering parts the first masks did not cover.
      The image is encoded once; only re-detection + extra predict() calls repeat.

    By default (``drop_below_iou=False``) a mask is kept even if its predicted IoU is
    below ``iou_floor``; set ``drop_below_iou=True`` to discard those. Near-empty masks
    (< ``tiny_mask_floor_px``) are always dropped.
    """
    if logger is None:
        logger = logging.getLogger("sam2_processing")

    height, width = spec.shape

    # --- working region (mirror the grid path's zero-padded crop) ---
    if duration <= width:
        working_spec = np.zeros((height, duration), dtype=spec.dtype)
        working_spec[:, :duration] = spec[:, :duration]
    else:
        working_spec = spec

    # --- flip + normalize + viridis -> RGB uint8 (identical to the grid path) ---
    flipped_spec = np.flipud(working_spec)
    spec_min, spec_max = flipped_spec.min(), flipped_spec.max()
    if spec_max > spec_min:
        spec_norm = (flipped_spec - spec_min) / (spec_max - spec_min)
    else:
        spec_norm = np.zeros_like(flipped_spec)
    image = (cmap(spec_norm)[:, :, :3] * 255).astype(np.uint8)

    img_w = image.shape[1]
    eff_duration = min(int(duration), img_w)

    # --- detect first-pass boxes on the SAME flipped space SAM sees ---
    # detect_fn (learned detector, e.g. yolo) takes the UNFLIPPED working spec, flips +
    # renders internally, and returns flipped-space (t,l,b,r) -- same convention as the cc
    # detect_boxes. It runs a SINGLE pass (no residual re-detection). The cc baseline
    # (detect_fn=None) keeps the iterative residual prompting.
    if detect_fn is not None:
        boxes0 = [tuple(int(v) for v in b) for b in detect_fn(working_spec, eff_duration)]
        eff_max_iters = 1
    else:
        boxes0 = detect_boxes(flipped_spec, eff_duration, detector_cfg)
        eff_max_iters = max(1, max_iters)
    # Box-area gate (pre-SAM): drop boxes smaller than min_box_area. Together with the YOLO
    # confidence threshold this blocks unwanted small segments — conf kills low-confidence
    # boxes, min_box_area kills small ones (incl. confident-but-tiny specks). Bounding-box
    # area, complementary to the cc detector's component-area `min_area_px`.
    if min_box_area > 0:
        boxes0 = [b for b in boxes0 if _box_area(b) >= min_box_area]
    if not boxes0:
        return []

    # --- encode the image ONCE, then prompt (optionally iterating on the residual) ---
    predictor.set_image(image)

    masks_out: List[Dict] = []
    covered = np.zeros(image.shape[:2], dtype=bool)
    n_low_iou = 0

    for it in range(eff_max_iters):
        if it == 0:
            boxes = boxes0
        else:
            # Suppress already-segmented pixels so the detector finds only the remainder.
            resid = flipped_spec.copy()
            resid[covered] = spec_min
            cand = detect_boxes(resid, eff_duration, detector_cfg)
            if min_box_area > 0:
                cand = [b for b in cand if _box_area(b) >= min_box_area]
            boxes = [b for b in cand
                     if _uncovered_frac(_box_mask(b, image.shape[:2]), covered) >= 0.5]
        if not boxes:
            break

        new_count = 0
        for box in boxes:
            # SINGLE x/y swap site: detector (t,l,b,r) -> SAM XYXY (x0,y0,x1,y1).
            xyxy = np.array(box_to_xyxy(box), dtype=np.float32)
            masks, iou_preds, _ = predictor.predict(box=xyxy, multimask_output=multimask_output)
            best = int(np.argmax(iou_preds))
            best_iou = float(iou_preds[best])
            seg = masks[best].astype(bool)

            # Stringency: keep only the bright call core, dropping faint harmonic bands /
            # diffuse tails below the intensity floor (in the SAME normalized space as the
            # image). Re-splitting then separates a harmonic that detaches from the core.
            if mask_intensity_floor > 0.0:
                seg = seg & (spec_norm >= mask_intensity_floor)

            subs = _split_mask_components(seg, tiny_mask_floor_px) if split_disconnected else [seg]
            for sub in subs:
                area = int(sub.sum())
                if area < tiny_mask_floor_px:
                    continue
                # In residual rounds, skip a sub-mask that just re-covers known pixels.
                if it > 0 and _uncovered_frac(sub, covered) < 0.5:
                    continue
                if best_iou < iou_floor:
                    n_low_iou += 1
                    if drop_below_iou:
                        continue
                masks_out.append({
                    "segmentation": sub,
                    "area": area,
                    "bbox": get_bbox(sub),   # (top,left,bottom,right), consistent w/ grid path
                    "iou_pred": best_iou,
                    "box": box,              # detector prompt box, for QA
                })
                covered |= sub
                new_count += 1

        if new_count == 0:
            break

    predictor.reset_predictor()  # free cached image embedding before next spec

    # Conservative post-merge: fuse near-duplicate / contained instances (over-seg),
    # keep genuinely distinct crossing calls separate.
    if merge_instances and len(masks_out) > 1:
        masks_out = merge_overlapping_instances(masks_out, merge_iou, merge_containment)

    if n_low_iou:
        logger.debug(f"{n_low_iou} mask(s) below iou_floor={iou_floor} "
                     f"({'dropped' if drop_below_iou else 'kept'})")

    return masks_out


def process_session_batch_boxprompt(
    session_name: str,
    specs: np.ndarray,
    durations: np.ndarray,
    predictor: SAM2ImagePredictor,
    detector_cfg: Optional[BoxDetectorConfig],
    cmap,
    duration_min: int = 10,
    batch_size: int = 12,
    multimask_output: bool = True,
    iou_floor: float = 0.70,
    drop_below_iou: bool = False,
    split_disconnected: bool = True,
    max_iters: int = 1,
    merge_instances: bool = True,
    merge_iou: float = 0.5,
    merge_containment: float = 0.8,
    detect_fn=None,
    mask_intensity_floor: float = 0.0,
    tiny_mask_floor_px: int = 12,
    min_box_area: int = 0,
    logger: Optional[logging.Logger] = None,
) -> Tuple[str, Dict]:
    """Process all spectrograms in a session (mirrors the grid path's process_session_batch).

    Args:
        session_name: Name of the session being processed.
        specs: Array of spectrograms (num_specs, height, width).
        durations: Array of valid time frames per spectrogram.
        predictor: SAM2ImagePredictor instance.
        detector_cfg: Connected-component box detector config.
        cmap: Matplotlib colormap (viridis) for RGB conversion.
        duration_min: Minimum duration threshold for processing.
        batch_size: Number of spectrograms to process before memory cleanup.
        multimask_output / iou_floor / drop_below_iou / split_disconnected / max_iters /
        merge_instances / merge_iou / merge_containment: see ``process_single_spec_boxprompt``.
        logger: Logger instance.

    Returns:
        Tuple[str, Dict]: Session name and dictionary mapping spec indices to masks.
    """
    if logger is None:
        logger = logging.getLogger("sam2_processing")

    processed_masks: Dict[int, List[Dict]] = {}

    valid_indices = [i for i, d in enumerate(durations) if d >= duration_min]
    logger.info(f"Processing {len(valid_indices)} valid specs out of {len(specs)} total")

    invalid_count = 0
    for i, duration in enumerate(durations):
        if duration < duration_min:
            invalid_count += 1
            processed_masks[i] = []
    logger.info(f"Found {invalid_count} invalid specs out of {len(durations)} total")

    total_batches = (len(valid_indices) + batch_size - 1) // batch_size
    for batch_idx in tqdm(range(total_batches), desc="Processing batches"):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(valid_indices))
        for i in valid_indices[start:end]:
            try:
                processed_masks[i] = process_single_spec_boxprompt(
                    specs[i], int(durations[i]), predictor, detector_cfg, cmap,
                    multimask_output=multimask_output, iou_floor=iou_floor,
                    drop_below_iou=drop_below_iou, split_disconnected=split_disconnected,
                    max_iters=max_iters, merge_instances=merge_instances,
                    merge_iou=merge_iou, merge_containment=merge_containment,
                    detect_fn=detect_fn, mask_intensity_floor=mask_intensity_floor,
                    tiny_mask_floor_px=tiny_mask_floor_px, min_box_area=min_box_area,
                    logger=logger,
                )
            except Exception as e:
                logger.error(f"Error processing spec {i}: {e}")
                processed_masks[i] = []

        if batch_idx % 5 == 0:
            cleanup_memory(logger, force_gpu_cleanup=(batch_idx % 20 == 0))
        if batch_idx % 50 == 0 and batch_idx > 0:
            log_memory_usage(logger, f"after batch {batch_idx}/{total_batches}")

    return session_name, processed_masks


def build_detector_cfg(args) -> BoxDetectorConfig:
    """Build a BoxDetectorConfig from parsed CLI args (boxprompt detector knobs).

    Field precedence (low -> high): BoxDetectorConfig defaults < the mapped CLI knobs
    (``pad_*``/``merge_*``/``min_area_px``/``strict_traces``, already preset-adjusted in
    cli.py) < ``detector_extra`` (a named preset's non-flag thresholds like ``PCT``) <
    ``detector_config_json`` (explicit power-user override file).
    """
    kwargs = dict(
        pad_time=args.pad_time,
        pad_freq=args.pad_freq,
        merge_time_gap=args.merge_time_gap,
        merge_freq_gap=args.merge_freq_gap,
        min_area_px=args.min_area_px,
        strict_traces=args.strict_traces,
    )
    # Named-preset detector fields not exposed as CLI flags (PCT, ABS_FLOOR, min_mean_int, ...).
    kwargs.update(getattr(args, "detector_extra", None) or {})
    # Explicit JSON file overrides everything.
    if getattr(args, "detector_config_json", None):
        import json
        with open(args.detector_config_json) as fh:
            kwargs.update(json.load(fh))
    return BoxDetectorConfig(**kwargs)


def boxprompt_config_summary(args, detector_cfg: Optional[BoxDetectorConfig]) -> Dict:
    """Assemble the box-prompt config dict stored in the pkl / YAML (analog of sam2_config)."""
    det = getattr(args, "detector", "cc")
    summary = {
        "method": "boxprompt",
        "detector_type": det,
        "multimask_output": not args.single_mask,
        "iou_floor": args.iou_floor,
        "drop_below_iou": args.drop_below_iou,
        "split_disconnected": not args.no_split_masks,
        "max_iters": args.max_iters,
        "merge_instances": not args.no_merge_instances,
        "merge_iou": args.merge_iou,
        "merge_containment": args.merge_containment,
        "mask_intensity_floor": getattr(args, "mask_intensity_floor", 0.0),
        "tiny_mask_floor_px": getattr(args, "tiny_mask_floor_px", 12),
        "min_box_area": getattr(args, "min_box_area", 0),
    }
    if det == "yolo":
        summary["yolo"] = {
            "weights": getattr(args, "yolo_weights", None),
            "conf": getattr(args, "yolo_conf", None),
            "iou": getattr(args, "yolo_iou", None),
            "imgsz": getattr(args, "yolo_imgsz", None),
        }
        # Keep the legacy key populated so downstream readers of "detector" don't KeyError.
        summary["detector"] = summary["yolo"]
    else:
        summary["detector"] = dict(detector_cfg.__dict__) if detector_cfg else {}
    return summary
