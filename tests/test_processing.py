"""
@author: bartulem
Test processing module.
"""

from __future__ import annotations

import os
import pathlib
import subprocess

import toml

from usv_playpen.send_email import Messenger

if os.name == "nt":
    command_addition = "cmd /c "
    shell_usage_bool = False
else:
    command_addition = ""
    shell_usage_bool = True

config_dir_global = pathlib.Path(__file__).parent.parent / "src" / "_config"
esd = toml.load(f"{config_dir_global}{os.sep}behavioral_experiments_settings.toml")


# def test_send_email():
#     try:
#         email_receiver = [str(email_address)]
#     except IndexError:
#         print("Error: Missing e-mail argument. Please provide it.")
#
#     # test email sending
#     try:
#         Messenger(
#             receivers=email_receiver, exp_settings_dict=esd
#         ).send_message(
#             subject="Test",
#             message="This is a 165B processing test email. Please do not reply.",
#         )
#         email_success = True
#     except Exception:
#         email_success = False
#
#     assert email_success, "165B not able to send e-mails."


def test_video_processing():
    # test video processing capabilities
    try:
        subprocess.Popen(
            args=f"""{command_addition}ffmpeg -version""",
            shell=shell_usage_bool,
        ).wait()
        video_processing_success = True
    except subprocess.CalledProcessError as e:
        print(e)
        video_processing_success = False

    assert video_processing_success, "FFMPEG not functional from command line."


def test_audio_processing():
    try:
        subprocess.Popen(
            args=f"""{command_addition}sox --version""",
            shell=shell_usage_bool,
        ).wait()
        audio_processing_success = True
    except subprocess.CalledProcessError as e:
        print(e)
        audio_processing_success = False

    assert audio_processing_success, "SOX not functional from command line."
