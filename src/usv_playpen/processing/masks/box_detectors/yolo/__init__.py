# ABOUTME: YOLO11 (ultralytics) USV box detector subpackage: train.py fine-tunes from yolo11n.pt,
# ABOUTME: infer.py exposes detect_boxes_yolo(spec, duration, model, cfg) -> [(t,l,b,r), ...].
"""YOLO11 transfer-learning USV box detector.

The easy fine-tuning entry point: start from COCO-pretrained ``yolo11n.pt`` and
fine-tune on the exporter's single-class dataset, then call
:func:`box_detectors.yolo.infer.detect_boxes_yolo` at inference.

``ultralytics`` is imported LAZILY inside ``train.py`` / ``infer.py`` function
bodies, so importing this subpackage (or ``py_compile``-ing it) does NOT require
ultralytics — important for the dev/test env (``samv2_env``) which lacks it.
"""

__all__ = ["train", "infer"]
