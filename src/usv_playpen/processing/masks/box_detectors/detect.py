# ABOUTME: Unified detector dispatcher: get_detector(name, **kw) -> callable(spec, duration) -> [(t,l,b,r)],
# ABOUTME: wrapping the cc / unet / yolo detectors behind ONE call signature for spec_gen integration.
"""Unified box-detector dispatcher for the box_detectors subproject.

This is the single entry point a consumer (e.g. an integrated
``spec_gen_full_pipeline`` boxprompt path, or an evaluation harness) uses to get a
detector without caring which backend it is:

    from box_detectors.detect import get_detector
    detect = get_detector("yolo", weights="/.../best.pt")
    boxes = detect(spec, duration)   # -> [(top, left, bottom, right), ...]

``get_detector(name, **kw)`` returns a **callable** with the uniform signature::

    detect(spec, duration) -> list[(top, left, bottom, right)]

for ``name`` in ``{"cc", "unet", "yolo"}``. The returned callable is a thin closure
that already holds the loaded model + config, so the caller never threads a
``model=`` / ``cfg=`` argument through — all three detectors look identical at the
call site.

Box / orientation contract (shared by all three)
-------------------------------------------------
* Output boxes are inclusive ``(top, left, bottom, right) == (y0, x0, y1, x1)`` —
  rows are FREQUENCY, columns are TIME — the SAME convention as
  ``spec_gen_full_pipeline/src/specgen/masks/cc_box_detector.detect_boxes``. NOT
  xyxy; the swap to SAM XYXY happens downstream (``box_to_xyxy``).
* Each detector receives the UNFLIPPED working spec and does its own
  ``np.flipud`` internally (matching ``cc_box_detector``), so the caller passes the
  exact same array to every backend. For ``"cc"`` the caller must therefore pass
  the working spec UNFLIPPED — see the wrapper note below, which mirrors
  ``cc_box_detector`` exactly.

Lazy / light imports
--------------------
Nothing heavy is imported at module load: ``cc`` pulls in
``specgen.masks.cc_box_detector`` (pure numpy/scipy/skimage) only when built;
``unet`` pulls in torch + smp only when built; ``yolo`` pulls in ultralytics only
when built. So ``import box_detectors.detect`` and ``py_compile`` succeed in
``samv2_env`` (no smp / no ultralytics) and on the GPU-less dev host.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import numpy as np

# Inclusive (top, left, bottom, right) = (y0, x0, y1, x1). Rows=freq, cols=time.
Box = Tuple[int, int, int, int]
Detector = Callable[[np.ndarray, int], List[Box]]

# NOTE (vendored copy): in the standalone ``specgen`` package the cc geometry and the
# yolo adapter are reachable as ordinary intra-package imports
# (``specgen.masks.cc_box_detector`` and ``..yolo.infer``), so the original sys.path
# hacks (``_SPECGEN_SRC`` / ``_ensure_box_detectors_on_path``) are dropped here. The
# ``unet`` backend is NOT vendored — use the standalone box_detectors subproject for it.

VALID_NAMES = ("cc", "unet", "yolo")


# ---------------------------------------------------------------------------
# Per-backend builders (each returns a closure detect(spec, duration) -> boxes)
# ---------------------------------------------------------------------------
def _build_cc(
    cfg=None,
    detector_extra: Optional[dict] = None,
    **detector_kwargs,
) -> Detector:
    """Build the connected-component baseline detector closure.

    Wraps ``specgen.masks.cc_box_detector.detect_boxes`` so it shares the uniform
    ``detect(spec, duration)`` signature. IMPORTANT: ``cc_box_detector.detect_boxes``
    does NOT flip internally — it expects to be called on the SAME orientation the
    boxes should live in, and in the existing pipeline its caller passes the
    *already-flipped* working spec. To keep ALL three detectors identical at the
    call site (caller passes the UNFLIPPED working spec; the detector flips
    internally), THIS wrapper performs the ``np.flipud`` itself before delegating to
    ``detect_boxes``. So the cc, unet, and yolo wrappers all take an unflipped spec
    and return flipped-space ``(t,l,b,r)`` boxes — interchangeable behind
    :func:`get_detector`.

    Args:
        cfg: An existing ``BoxDetectorConfig`` to use as-is. If None, one is built
            from ``detector_kwargs`` + ``detector_extra``.
        detector_extra: Extra ``BoxDetectorConfig`` fields (e.g. ``{"PCT": 92.0}``)
            merged on top of ``detector_kwargs`` (named-preset style).
        **detector_kwargs: ``BoxDetectorConfig`` field overrides
            (``pad_time``, ``merge_time_gap``, ``min_area_px``, ``strict_traces``, ...).

    Returns:
        ``detect(spec, duration) -> list[(t,l,b,r)]``.
    """
    from ..cc_box_detector import BoxDetectorConfig, detect_boxes  # type: ignore

    if cfg is None:
        kwargs = dict(detector_kwargs)
        kwargs.update(detector_extra or {})
        cfg = BoxDetectorConfig(**kwargs)

    def detect(spec: np.ndarray, duration: int) -> List[Box]:
        spec = np.asarray(spec)
        # Flip once here so this wrapper takes the UNFLIPPED working spec, matching
        # the unet / yolo wrappers (which also flip internally). The cc detector then
        # runs in flipped space and returns flipped-space (t,l,b,r) boxes.
        flipped = np.flipud(spec)
        n_time = flipped.shape[-1]
        d = max(1, min(int(duration), n_time))
        return list(detect_boxes(flipped, d, cfg))

    return detect


def _build_unet(
    weights: str,
    device: Optional[str] = None,
    config_yaml: Optional[str] = None,
    cfg=None,
    **cfg_overrides,
) -> Detector:
    """Build the U-Net detector closure (loads the checkpoint ONCE).

    Args:
        weights: Path to the trained U-Net ``.pt`` (from ``unet/train.py``). Required.
        device: ``"cuda"`` / ``"cpu"`` / specific; defaults to cuda-if-available.
        config_yaml: Optional path to ``unet/config.yaml`` to read ``prob_thresh`` +
            the geometry ``detector`` block. If None, defaults are used (or ``cfg``).
        cfg: An explicit ``UNetInferConfig`` (overrides ``config_yaml``).
        **cfg_overrides: Field overrides applied on top of the resolved cfg
            (e.g. ``prob_thresh=0.4``, ``device="cuda"``).

    Returns:
        ``detect(spec, duration) -> list[(t,l,b,r)]`` bound to the loaded model.

    Raises:
        ValueError: if ``weights`` is not provided.
    """
    # The U-Net backend is intentionally NOT vendored into specgen (it pulls in
    # segmentation_models_pytorch and is unused by the boxprompt path). specgen only
    # ever calls get_detector("yolo", ...) / ("cc", ...). For unet, use the full
    # box_detectors subproject from the MMMmB monorepo.
    raise NotImplementedError(
        "The 'unet' detector is not bundled in the standalone specgen copy. Use the "
        "box_detectors subproject (box_detectors.unet) in the MMMmB monorepo, or pick "
        "--detector yolo / cc."
    )


def _build_yolo(
    weights: str,
    device: Optional[str] = None,
    cfg=None,
    **cfg_overrides,
) -> Detector:
    """Build the YOLO detector closure (loads the ultralytics model ONCE).

    Args:
        weights: Path to the fine-tuned YOLO ``best.pt`` (from ``yolo/train.py``).
            Required.
        device: ``"0"`` / ``"cpu"`` / None (ultralytics auto-selects).
        cfg: An explicit ``YoloDetectorConfig`` (else one is built from defaults).
        **cfg_overrides: Field overrides applied on top of the cfg
            (e.g. ``conf=0.2``, ``iou=0.8``, ``max_det=500``).

    Returns:
        ``detect(spec, duration) -> list[(t,l,b,r)]`` bound to the loaded model.

    Raises:
        ValueError: if ``weights`` is not provided.
    """
    if not weights:
        raise ValueError(
            "get_detector('yolo', ...) requires weights=<path to trained best.pt> "
            "(train via box_detectors.yolo.train first)."
        )
    from .yolo.infer import (  # type: ignore
        YoloDetectorConfig,
        detect_boxes_yolo,
        load_model,
    )

    if cfg is None:
        cfg = YoloDetectorConfig()
    for k, v in (cfg_overrides or {}).items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    if device is not None:
        cfg.device = device

    model = load_model(weights, device=cfg.device)

    def detect(spec: np.ndarray, duration: int) -> List[Box]:
        return list(detect_boxes_yolo(spec, duration, model=model, cfg=cfg))

    return detect


# ---------------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------------
def get_detector(name: str, weights: Optional[str] = None, **kw) -> Detector:
    """Return a unified ``detect(spec, duration) -> [(t,l,b,r)]`` callable.

    All three backends are exposed behind ONE signature so a caller can swap
    ``--detector {cc,unet,yolo}`` without touching the rest of the pipeline. The
    learned detectors load their checkpoint exactly once (here), and the returned
    closure holds the model + config.

    Args:
        name: ``"cc"`` (unlearned baseline; no weights needed), ``"unet"``, or
            ``"yolo"`` (both require ``weights``).
        weights: Path to the trained checkpoint for ``unet`` / ``yolo``. Ignored for
            ``cc``.
        **kw: Backend-specific options, forwarded to the builder:

            * ``cc``: ``cfg`` (a ``BoxDetectorConfig``), ``detector_extra`` (dict of
              extra config fields), and/or any ``BoxDetectorConfig`` field
              (``pad_time``, ``merge_time_gap``, ``min_area_px``, ``strict_traces``...).
            * ``unet``: ``device``, ``config_yaml`` (path to ``unet/config.yaml``),
              ``cfg`` (a ``UNetInferConfig``), and ``UNetInferConfig`` field
              overrides (``prob_thresh``, ...).
            * ``yolo``: ``device``, ``cfg`` (a ``YoloDetectorConfig``), and
              ``YoloDetectorConfig`` field overrides (``conf``, ``iou``, ``max_det``...).

    Returns:
        A callable ``detect(spec, duration) -> list[(top, left, bottom, right)]``.
        ``spec`` is the UNFLIPPED working spec ``[F, T]``; the detector flips
        internally (matching ``cc_box_detector``). Columns ``>= duration`` are zero
        padding and excluded; boxes are clamped to ``[0, F-1]`` x ``[0, duration-1]``.

    Raises:
        ValueError: for an unknown ``name``, or a missing ``weights`` for unet/yolo.
    """
    key = str(name).strip().lower()
    if key == "cc":
        return _build_cc(**kw)
    if key == "unet":
        return _build_unet(weights=weights, **kw)
    if key == "yolo":
        return _build_yolo(weights=weights, **kw)
    raise ValueError(
        f"unknown detector {name!r}; expected one of {VALID_NAMES}."
    )
