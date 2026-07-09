# ABOUTME: Shared helpers for box_detectors: box clamping, format converters,
# ABOUTME: and the canonical spectrogram->uint8 image rendering reused by all detectors.
"""Shared, dependency-light helpers for the box_detectors subproject.

Everything in :mod:`box_detectors.common.boxes` is pure numpy (no torch, no
ultralytics, no segmentation_models_pytorch) so it imports cleanly in any of the
cluster envs and is unit-testable on the dev host.
"""

from . import boxes  # noqa: F401

__all__ = ["boxes"]
