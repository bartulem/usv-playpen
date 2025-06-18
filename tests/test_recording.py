"""
@author: bartulem
Test recording module.
"""

from __future__ import annotations

import configparser
import os
import pathlib
import sys

import motifapi
import toml

from usv_playpen.send_email import Messenger

config_dir_global = pathlib.Path(__file__).parent.parent / "src" / "_config"
exp_settings_dict = toml.load(
    f"{config_dir_global}{os.sep}behavioral_experiments_settings.toml"
)


# def test_recording_send_email():
#     try:
#         email_receiver = [str(email_address)]
#     except IndexError:
#         print("Error: Missing e-mail argument. Please provide it.")
#
#     # test email sending
#     try:
#         Messenger(
#             receivers=email_receiver, exp_settings_dict=exp_settings_dict
#         ).send_message(
#             subject="Test",
#             message="This is a 165B recording test email. Please do not reply.",
#         )
#         email_success = True
#     except Exception:
#         email_success = False
#
#     assert email_success, "165B not able to send e-mails"


def test_recording_video_software():
    # test video recording capabilities
    video_config = configparser.ConfigParser()
    video_config.read(f"{config_dir_global}{os.sep}motif_config.ini")

    try:
        motif_api = motifapi.MotifApi(
            video_config["motif"]["master_ip_address"], video_config["motif"]["api"]
        )
        motif_success = True
    except motifapi.api.MotifError:
        motif_success = False

    assert motif_success, "Motif not operational on this PC."


def test_recording_audio_sync_software():
    # test audio / sync_software presence
    config_file_status = os.path.isfile(
        f"{config_dir_global}{os.sep}avisoft_config.ini"
    ) and os.path.isfile(f"{config_dir_global}{os.sep}coolterm_config.stc")
    assert config_file_status, "Audio/sync software config files missing."

    software_status = os.path.isfile(
        f"{exp_settings_dict['avisoft_recorder_exe']}{os.sep}rec_usgh.exe"
    ) and os.path.isfile(
        f"{exp_settings_dict['coolterm_basedirectory']}{os.sep}CoolTerm.exe"
    )

    assert software_status, "Audio / SYNC software not ready for use."
