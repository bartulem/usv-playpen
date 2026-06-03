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

    Parameters
    ----------
    None

    Returns
    -------
    (bool)
        True if a QApplication instance exists, indicating a GUI context; False otherwise.
    """

    return isinstance(QCoreApplication.instance(), QApplication)


def smart_wait(app_context_bool: bool = False, seconds: float = 0) -> None:
    """
    Description
    -----------
    Waits for ``seconds`` seconds. When ``app_context_bool`` is True the wait is
    performed with ``QTest.qWait`` (which keeps the Qt event loop responsive in a
    GUI context); otherwise it uses ``time.sleep``. The mechanism is chosen purely
    by ``app_context_bool``, not by probing for a live QApplication. A negative
    ``seconds`` is clamped to 0 so both branches behave consistently (``time.sleep``
    would otherwise raise on a negative value).

    Parameters
    ----------
    app_context_bool (bool)
        If True, wait via ``QTest.qWait`` (GUI context); if False, via
        ``time.sleep``. Callers typically pass the result of ``is_gui_context()``.
    seconds (float)
        The number of seconds to wait; may be fractional for more precise timing.
        Values below 0 are treated as 0.

    Returns
    -------
    (None)
    """

    seconds = max(0.0, seconds)
    if app_context_bool:
        QTest.qWait(int(seconds * 1000))
    else:
        time.sleep(seconds)
