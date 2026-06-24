# ABOUTME: Connected-component USV box detector -- the prompt source for box-prompted SAM2.
# ABOUTME: Threshold -> ndimage.label -> per-component bbox -> time-gap merge -> asymmetric dilate/clamp.
# ABOUTME: Now lives in the specgen package (specgen.masks). Ported from the in-subproject
# ABOUTME: cc_box_detector.py, itself a copy of sam2_pred/box_prompt/cc_box_detector.py.

"""Connected-component box detector for USV spectrograms.

This is the *prompt source* for the box-prompt SAM2 path (see
``specgen.masks.boxprompt_utils``). Given a single spectrogram and its valid
``duration`` (active time frames), it returns one bounding box per detected
USV candidate. Those boxes are then fed, one at a time, to
``SAM2ImagePredictor.predict(box=...)``.

Design provenance
-----------------
The threshold -> label -> per-component-bbox core is **ported** (not imported, to keep
``sam2_pred`` dependency-light and runnable in ``samv2_env``) from the frozen
no-mask detector at ``train_ava_hydra/src/noise_filter/features.py``
(``morphological_trace_features``, Stages 0-2) and its ``DetectorConfig`` defaults at
``train_ava_hydra/src/noise_filter/config.py``. Only the geometry needed to emit
boxes is kept; the noise *scoring* stages (3-5) are dropped.

Key differences from the noise-filter source, justified by the box-prompt design:
  * The noise-filter per-component trace filters (elongation / fill / max-height) are
    tuned to isolate clean horizontal tonal strips for *noise scoring*. For *prompting*
    we want RECALL (a missed component => no box => no mask, with no recovery), so those
    shape gates are DISABLED by default here and exposed as opt-in (``strict_traces``).
  * A second **time-gap box merge** unions components that are close in time AND
    overlap/near in frequency -- joining a single call broken into pieces by the
    threshold -- while deliberately keeping frequency-stacked, time-overlapping
    distinct calls SEPARATE (this is what fixes the overlap case).
  * Boxes are **asymmetrically dilated** (more along time than frequency, since USVs are
    horizontally elongated) and clamped to ``[0, H-1]`` x ``[0, duration-1]`` so prompts
    never extend into the right-edge zero padding.

Coordinate convention
----------------------
Boxes are returned as ``(top, left, bottom, right)`` with INCLUSIVE bottom/right,
matching ``simple_mask_merging.get_bbox`` = ``(rows.min, cols.min, rows.max, cols.max)``.
This is (y0, x0, y1, x1). SAM2's ``predict(box=...)`` wants XYXY = (x0, y0, x1, y1);
the swap is done ONLY at the predict() call site in the inference script (the repo has
a history of tuple-index-shift bugs, so the conversion lives in exactly one place).

Pure numpy / scipy.ndimage / skimage -- no torch -- so this module is unit-testable in
``samv2_env`` without a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy import ndimage
from skimage.filters import threshold_otsu

Box = Tuple[int, int, int, int]  # (top, left, bottom, right), inclusive


# ---------------------------------------------------------------------------
# Config (ported subset of noise_filter.config.DetectorConfig + box params)
# ---------------------------------------------------------------------------
@dataclass
class BoxDetectorConfig:
    """Thresholds for the connected-component box detector.

    Stage 0-2 fields mirror ``noise_filter.config.DetectorConfig`` defaults so the
    detector behaves like the validated reference. The ``merge_*`` / ``pad_*`` /
    ``strict_*`` fields are box-prompt additions.
    """

    # --- numerical hygiene ---
    EPS: float = 1e-6
    PTP_EPS: float = 1e-3  # min peak-to-peak of signal region before "degenerate"

    # --- Stage 0: degenerate gate ---
    D_MIN: int = 4         # minimum valid duration (columns) to attempt detection
    MIN_FG_PIX: int = 20   # minimum foreground (S > 0) pixel count

    # --- Stage 1: pre-conditioning ---
    NORM_PCTL: Tuple[float, float] = (2.0, 98.0)  # robust contrast normalization
    TIME_SMOOTH: int = 3   # uniform_filter1d window along TIME axis only (never freq)
    ROW_BG_PCTL: float = 15.0  # per-row baseline percentile (must be < 50)

    # --- Stage 2: binarization threshold ---
    OTSU_MIN_RANGE: float = 0.05  # min dynamic range of foreground to run Otsu
    PCT: float = 90.0      # percentile floor on the Otsu threshold
    ABS_FLOOR: float = 0.05  # absolute floor on the binarization threshold

    # --- per-component KEEP filters (loosened for recall vs. noise_filter) ---
    min_area_px: int = 20          # drop small diffuse-noise components. 12 was too low and
    #   admitted faint blobs as false positives; 20 trims the noisy tail (specs with >=5
    #   boxes ~1.9% -> ~0.1%) while keeping ~99% of specs with >=1 box. Raise to 25-30 for
    #   stricter, at the cost of ~1-2% more empty specs. (Tuned on 20251111_151902.)
    min_mean_int: float = 0.05     # drop very faint components (R_ridge mean). NOTE: on tested
    #   data the noise was small-area not low-intensity, so min_area_px is the active knob.
    # The following horizontal-tonal-strip gates are DISABLED by default (None) to
    # maximize recall; set strict_traces=True to apply the noise_filter values.
    strict_traces: bool = False
    min_elong: float = 2.0         # used only if strict_traces
    max_height_f: int = 18         # used only if strict_traces
    min_fill: float = 0.60         # used only if strict_traces
    min_width_t: int = 5           # used only if strict_traces

    # --- saturation guard (skip pathological all-bright specs) ---
    max_area_frac: float = 0.35    # if binarized foreground exceeds this frac -> no boxes

    # --- time-gap box merge ---
    merge_time_gap: int = 6        # union boxes whose column gap <= this (frames).
    #   ~12 ms at hop=512 / sr=250 kHz (~2.05 ms/frame); 10-15 ms ~= 5-7 frames.
    merge_freq_gap: int = 2        # ...AND whose row intervals overlap or are within this.
    #   Small, so frequency-stacked distinct calls (large freq gap) stay SEPARATE.

    # --- asymmetric dilation (USVs are horizontally elongated) ---
    pad_time: int = 8              # pad along time (cols); MedSAM jitter is 0-20px, kept conservative
    pad_freq: int = 4              # pad along freq (rows)

    def __post_init__(self) -> None:
        if isinstance(self.NORM_PCTL, list):
            self.NORM_PCTL = tuple(self.NORM_PCTL)
        assert 0.0 <= self.ROW_BG_PCTL < 50.0, "ROW_BG_PCTL must be in [0, 50)"
        assert self.TIME_SMOOTH >= 1, "TIME_SMOOTH must be >= 1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _clip_duration(duration, n_time: int) -> int:
    """Clamp a (possibly noisy) duration to ``[1, n_time]``."""
    d = int(duration)
    return max(1, min(d, n_time))


def _sanitize(arr: np.ndarray) -> np.ndarray:
    """Replace non-finite pixels with 0 and return a float64 copy."""
    out = np.asarray(arr, dtype=np.float64)
    if not np.all(np.isfinite(out)):
        out = np.where(np.isfinite(out), out, 0.0)
    return out


def ridge_map(spec: np.ndarray, duration: int, cfg: BoxDetectorConfig
              ) -> Optional[np.ndarray]:
    """Stages 0-1: degenerate gate + robust normalize + row-bg subtract + ridge.

    Ported from ``noise_filter.features.morphological_trace_features``.

    Args:
        spec: Float array ``[F, T]`` (freq = rows, time = cols). Pass it in the SAME
            orientation the boxes / SAM masks should live in (see module docstring).
        duration: Valid time frames.
        cfg: BoxDetectorConfig.

    Returns:
        ``R_ridge`` over the signal region ``[F, d]``, or ``None`` if the spec fails
        the degenerate gate (caller should emit zero boxes).
    """
    spec = np.asarray(spec)
    n_freq, n_time = spec.shape[-2], spec.shape[-1]
    d = _clip_duration(duration, n_time)

    # Hard-slice to the signal region so right zero-padding never enters detection.
    S = _sanitize(spec[:, :d])  # [F, d]

    # ---- Stage 0: degenerate gate ----
    ptp = float(S.max() - S.min())
    n_fg = int(np.count_nonzero(S > 0))
    if (d < cfg.D_MIN) or (ptp < cfg.PTP_EPS) or (n_fg < cfg.MIN_FG_PIX):
        return None

    EPS = cfg.EPS

    # ---- Stage 1: pre-conditioning ----
    p_lo, p_hi = cfg.NORM_PCTL
    p2, p98 = np.percentile(S, [float(p_lo), float(p_hi)])
    Sn = np.clip((S - p2) / (p98 - p2 + EPS), 0.0, 1.0)

    # Row background subtraction (cancels per-frequency baseline / broadband).
    row_bg = np.percentile(Sn, float(cfg.ROW_BG_PCTL), axis=1, keepdims=True)
    R_bs = np.clip(Sn - row_bg, 0.0, None)

    # Horizontal ridge emphasis: smooth along TIME only (axis=1), never freq.
    R_ridge = ndimage.uniform_filter1d(
        R_bs, size=int(cfg.TIME_SMOOTH), axis=1, mode="nearest"
    )
    return R_ridge


def _label_boxes(R_ridge: np.ndarray, cfg: BoxDetectorConfig) -> List[Box]:
    """Stage 2: binarize -> 8-connected label -> per-component boxes (filtered).

    Returns inclusive ``(top, left, bottom, right)`` boxes in ``R_ridge`` coords.
    """
    fg_vals = R_ridge[R_ridge > 0]
    if (fg_vals.size < cfg.MIN_FG_PIX) or (
        (fg_vals.max() - fg_vals.min()) < cfg.OTSU_MIN_RANGE
    ):
        return []

    try:
        otsu_thr = float(threshold_otsu(fg_vals))
    except Exception:  # threshold_otsu robustness fallback
        otsu_thr = float(np.median(fg_vals))
    thr = max(otsu_thr, float(np.percentile(fg_vals, cfg.PCT)), float(cfg.ABS_FLOOR))

    B = R_ridge > thr

    # Saturation guard: pathological all-bright spec -> emit nothing.
    if B.mean() > cfg.max_area_frac:
        return []

    structure = np.ones((3, 3), dtype=int)  # 8-connectivity
    lab, n_comp = ndimage.label(B, structure=structure)
    if n_comp == 0:
        return []

    areas = np.bincount(lab.ravel(), minlength=n_comp + 1)
    slices = ndimage.find_objects(lab)
    comp_idx = np.arange(1, n_comp + 1)
    mean_ints = np.atleast_1d(
        np.asarray(ndimage.mean(R_ridge, labels=lab, index=comp_idx), dtype=np.float64)
    )

    boxes: List[Box] = []
    for k in range(n_comp):
        sl = slices[k]
        if sl is None:
            continue
        row_sl, col_sl = sl
        area = float(areas[k + 1])
        h = float(row_sl.stop - row_sl.start)  # freq extent
        w = float(col_sl.stop - col_sl.start)  # time extent
        m_int = float(mean_ints[k])

        # Recall-first filters (always on).
        if area < cfg.min_area_px or m_int < cfg.min_mean_int:
            continue

        # Optional strict horizontal-tonal-strip gates (noise_filter parity).
        if cfg.strict_traces:
            fill = area / (h * w) if (h * w) > 0 else 0.0
            elong = w / max(h, 1.0)
            if not (
                elong >= cfg.min_elong
                and h <= cfg.max_height_f
                and fill >= cfg.min_fill
                and w >= cfg.min_width_t
            ):
                continue

        # Inclusive (top, left, bottom, right), matching get_bbox.
        boxes.append((row_sl.start, col_sl.start, row_sl.stop - 1, col_sl.stop - 1))

    return boxes


def _intervals_near(a0: int, a1: int, b0: int, b1: int, gap: int) -> bool:
    """True if inclusive intervals [a0,a1],[b0,b1] overlap or are within ``gap``."""
    if a1 >= b0 and b1 >= a0:  # overlap
        return True
    return (b0 - a1 <= gap) and (a0 - b1 <= gap)  # one of the two gaps is small & >=0


def merge_time_gaps(boxes: List[Box], cfg: BoxDetectorConfig) -> List[Box]:
    """Union boxes close in TIME and overlapping/near in FREQUENCY.

    Joins a single call fragmented by the threshold, while keeping
    frequency-stacked, time-overlapping distinct calls SEPARATE (large freq gap =>
    not merged). Iterates to a fixed point.
    """
    if len(boxes) <= 1:
        return list(boxes)

    cur = list(boxes)
    while True:
        merged_any = False
        n = len(cur)
        for i in range(n):
            for j in range(i + 1, n):
                t0, l0, b0, r0 = cur[i]
                t1, l1, b1, r1 = cur[j]
                time_near = _intervals_near(l0, r0, l1, r1, cfg.merge_time_gap)
                freq_near = _intervals_near(t0, b0, t1, b1, cfg.merge_freq_gap)
                if time_near and freq_near:
                    cur[i] = (min(t0, t1), min(l0, l1), max(b0, b1), max(r0, r1))
                    cur.pop(j)
                    merged_any = True
                    break
            if merged_any:
                break
        if not merged_any:
            return cur


def _dilate_clamp(box: Box, cfg: BoxDetectorConfig, n_freq: int, duration: int) -> Box:
    """Asymmetrically pad a box and clamp to ``[0, n_freq-1]`` x ``[0, duration-1]``."""
    t, l, b, r = box
    t = max(0, t - cfg.pad_freq)
    b = min(n_freq - 1, b + cfg.pad_freq)
    l = max(0, l - cfg.pad_time)
    # Right edge clamped to duration-1, NOT n_time-1: cols >= duration are zero padding.
    r = min(duration - 1, r + cfg.pad_time)
    return (t, l, b, r)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def detect_boxes(spec: np.ndarray, duration: int,
                 cfg: Optional[BoxDetectorConfig] = None) -> List[Box]:
    """Detect one box per USV candidate in a single spectrogram.

    Args:
        spec: Float array ``[F, T]`` (freq = rows, time = cols), in the SAME
            orientation the output boxes/masks should live in. For the box-prompt
            inference path this is the FLIPPED (``np.flipud``) working spec, so boxes
            match the flipped viridis image SAM sees.
        duration: Valid time frames (active columns).
        cfg: BoxDetectorConfig (defaults if None).

    Returns:
        List of inclusive ``(top, left, bottom, right)`` boxes, time-gap merged and
        asymmetrically dilated/clamped. Empty list if the spec is degenerate or has
        no detectable components.
    """
    if cfg is None:
        cfg = BoxDetectorConfig()

    n_freq, n_time = spec.shape[-2], spec.shape[-1]
    d = _clip_duration(duration, n_time)

    R_ridge = ridge_map(spec, d, cfg)
    if R_ridge is None:
        return []

    raw_boxes = _label_boxes(R_ridge, cfg)
    if not raw_boxes:
        return []

    merged = merge_time_gaps(raw_boxes, cfg)
    return [_dilate_clamp(bx, cfg, n_freq, d) for bx in merged]


def box_to_xyxy(box: Box) -> Tuple[int, int, int, int]:
    """Convert detector ``(top, left, bottom, right)`` -> SAM XYXY ``(x0, y0, x1, y1)``.

    NOTE: this is the SINGLE place x/y are swapped for SAM. Detector boxes are
    (y0, x0, y1, x1); SAM wants (x0, y0, x1, y1). Keep this isolated -- the repo has
    a history of tuple-index-shift bugs.
    """
    t, l, b, r = box
    return (l, t, r, b)
