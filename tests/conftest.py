"""
Test configuration and fixtures.
"""

from __future__ import annotations

import pytest
import toml

from usv_playpen import config_dir


@pytest.fixture(scope="session")
def behavioral_experiments_settings():
    """Load behavioral experiments settings TOML file as a pytest fixture."""
    return toml.load(config_dir / "behavioral_experiments_settings.toml")
