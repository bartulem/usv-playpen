"""
@author: bartulem
Test processing module.
"""

from __future__ import annotations

import subprocess


def test_video_processing():
    assert subprocess.Popen(args="ffmpeg -version", shell=True).wait() == 0, (
        "FFMPEG not functional from command line."
    )


def test_audio_processing():
    assert subprocess.Popen(args="static_sox --version", shell=True).wait() == 0, (
        "SOX not functional from command line."
    )
