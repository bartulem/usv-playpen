# ABOUTME: YOLO11 (ultralytics) USV box detector subpackage: infer.py exposes
# ABOUTME: detect_boxes_yolo(spec, duration, model, cfg) -> [(t,l,b,r), ...].
"""YOLO11 USV box detector (inference).

:func:`box_detectors.yolo.infer.detect_boxes_yolo` runs a YOLO11 model
(fine-tuned from COCO-pretrained ``yolo11n.pt`` on the exporter's single-class
dataset) at inference. Training itself is driven by ``ultralytics`` directly on
the dataset produced by ``processing/export_yolo_dataset.py`` — there is no
vendored ``train`` module here.

``ultralytics`` is imported LAZILY inside ``infer.py`` function bodies, so
importing this subpackage (or ``py_compile``-ing it) does NOT require
ultralytics — important for the dev/test env (``samv2_env``) which lacks it.
"""

__all__ = ["infer"]
