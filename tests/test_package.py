from __future__ import annotations

import importlib.metadata

import usv_playpen as m


def test_version():
    assert importlib.metadata.version("usv_playpen") == m.__version__
