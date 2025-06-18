"""
@author: bartulem
Utility functions for precise waiting in both GUI and non-GUI contexts.
"""

from __future__ import annotations

import time

from PyQt6.QtCore import QCoreApplication
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication


def is_gui_context() -> bool:
    """
    Description
    -----------
    Checks if the current context is a GUI context by verifying if a QApplication instance exists.
    -----------

    Parameters
    ----------
    ----------

    Returns
    ----------
    (bool)
        True if a QApplication instance exists, indicating a GUI context; False otherwise.
    ----------
    """

    return isinstance(QCoreApplication.instance(), QApplication)


def smart_wait(app_context_bool: bool = None, seconds: float = None) -> None:
    """
    Description
    -----------
    Waits for a specified number of seconds, using QTest.qWait if a QApplication instance is available,
    -----------

    Parameters
    ----------
    app_context_bool (bool)
        If True, indicates that the function should use QTest.qWait for waiting. If False or None, it will use time.sleep.
    seconds (float)
        The number of seconds to wait. This can be a fractional value for more precise timing.
    ----------

    Returns
    ----------
    (None)
    ----------
    """

    if app_context_bool:
        QTest.qWait(int(seconds * 1000))
    else:
        time.sleep(seconds)
