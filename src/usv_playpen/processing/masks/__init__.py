# ABOUTME: Vendored SAM2/YOLO USV mask-segmentation kernels (from the specgen
# ABOUTME: spec_gen_full_pipeline package), driven by processing.generate_masks.
"""Vendored mask-segmentation kernels.

This subpackage is a near-verbatim vendor of the segmentation core from the
``specgen`` ``spec_gen_full_pipeline`` project (box-prompt SAM2 + the YOLO / CC
box detectors). It is kept close to the original so it tracks that validated
implementation; the usv-playpen-native orchestration (consolidated-H5 I/O,
settings, the CLI) lives in ``processing/generate_masks.py``.

The heavy ``torch`` / ``sam2`` / ``ultralytics`` dependencies are imported only
when these modules are actually used, so importing the package name is cheap.
"""
from __future__ import annotations
