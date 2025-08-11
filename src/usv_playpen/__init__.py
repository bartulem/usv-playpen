"""
Copyright (c) 2025 Bartul Mimica. All rights reserved.

usv_playpen: GUI/CLI to conduct, process, and analyze experiments w/ multichannel e-phys, audio, and video acquisition.
"""

from __future__ import annotations

# Get the path to the configuration directory, it should be in the package directory
import pathlib

from ._version import version as __version__

config_dir = pathlib.Path(__file__).parent / "_config"

__all__ = ["__version__"]
