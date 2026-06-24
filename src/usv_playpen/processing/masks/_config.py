# ABOUTME: Minimal config shim for the vendored mask kernels: locates the bundled
# ABOUTME: mask-preset YAMLs. SAM2/YOLO model paths come from usv-playpen settings.
"""Tiny replacement for the parts of ``specgen.config`` the vendored mask
kernels touch.

The kernels only ever ask the config layer for the bundled mask-preset
catalogs; everything else (SAM2 checkpoint, YOLO weights, the spectrogram
source) is injected by ``processing/generate_masks.py`` from usv-playpen's
``processing_settings.json`` rather than resolved here.
"""
from __future__ import annotations

import pathlib

_CONFIGS_DIR = pathlib.Path(__file__).parent / "configs"


def load_grid_presets() -> dict:
    """Return the SAM2 AMG (grid) preset catalog from the bundled YAML."""
    import yaml  # lazy: only the grid path needs YAML

    with (_CONFIGS_DIR / "mask_grid_presets.yaml").open("r") as preset_file:
        return yaml.safe_load(preset_file)


def load_boxprompt_presets() -> dict:
    """Return the box-prompt preset catalog from the bundled YAML."""
    import yaml

    with (_CONFIGS_DIR / "mask_boxprompt_presets.yaml").open("r") as preset_file:
        return yaml.safe_load(preset_file)
