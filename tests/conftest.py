from __future__ import annotations

import os
import sys
import warnings

# If running under WSL
# Ensure Qt uses a headless platform during tests (no X server required).
# This must be set before the first QApplication is created.
if sys.platform.startswith("linux") and "microsoft" in os.uname().release.lower():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
