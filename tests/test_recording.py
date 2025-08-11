"""
@author: bartulem
Test recording module.
"""

from __future__ import annotations

import configparser
from pathlib import Path

import motifapi

from usv_playpen import config_dir


def test_recording_video_software():
    # test video recording capabilities
    video_config = configparser.ConfigParser()
    video_config.read(config_dir / "motif_config.ini")

    try:
        motif_api = motifapi.MotifApi(
            video_config["motif"]["master_ip_address"], video_config["motif"]["api"]
        )
        motif_success = True
    except motifapi.api.MotifError:
        motif_success = False

    assert motif_success, "Motif not operational on this PC."


def test_recording_audio_sync_software(behavioral_experiments_settings):
    # test audio / sync_software presence
    config_file_status = (config_dir / "avisoft_config.ini").is_file() and (
        config_dir / "coolterm_config.stc"
    ).is_file()
    assert config_file_status, "Audio/sync software config files missing."

    avisoft_exe = (
        Path(behavioral_experiments_settings["avisoft_recorder_exe"]) / "rec_usgh.exe"
    )
    coolterm_exe = (
        Path(behavioral_experiments_settings["coolterm_basedirectory"]) / "CoolTerm.exe"
    )

    assert coolterm_exe.is_file(), "CoolTerm executable not found."
    assert avisoft_exe.is_file(), "AviSoft recorder executable not found."
