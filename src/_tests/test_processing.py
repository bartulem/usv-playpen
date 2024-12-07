"""
@author: bartulem
Test processing module.
"""

import os
import platform
import subprocess
import sys
import toml
import unittest
from ..send_email import Messenger
from .. import usv_playpen_gui as usv_playpen_gui


class TestProcessing(unittest.TestCase):

    if os.name == 'nt':
        command_addition = 'cmd /c '
        shell_usage_bool = False
    else:
        command_addition = ''
        shell_usage_bool = True

    if platform.system() == 'Windows':
        config_dir_global = 'C:\\experiment_running_docs'
    elif platform.system() == 'Linux':
        config_di_global = f'/mnt/falkner/Bartul/PC_transfer/experiment_running_docs'
    else:
        config_dir_global = f'/Volumes/falkner/Bartul/PC_transfer/experiment_running_docs'

    esd = toml.load(f"{config_dir_global}{os.sep}behavioral_experiments_settings.toml")

    def test_send_email(self):

        # test email sending
        try:
            Messenger(receivers=usv_playpen_gui.email_list_global.split(","),
                      exp_settings_dict=self.esd).send_message(subject="Test", message="This is a 165B processing test email. Please do not reply.")
            email_success = True
        except Exception:
            email_success = False

        self.assertTrue(expr=email_success, msg="165B not able to send e-mails.")

    def test_video_processing(self):

        # test video processing capabilities
        try:
            subprocess.Popen(f'''{self.command_addition}ffmpeg -version''',
                             shell=self.shell_usage_bool).wait()
            video_processing_success = True
        except subprocess.CalledProcessError as e:
            print(e)
            video_processing_success = False

        self.assertTrue(expr=video_processing_success, msg="FFMPEG not functional from command line.")

    def test_audio_processing(self):

        try:
            subprocess.Popen(f'''{self.command_addition}sox --version''',
                             shell=self.shell_usage_bool).wait()
            audio_processing_success = True
        except subprocess.CalledProcessError as e:
            print(e)
            audio_processing_success = False

        self.assertTrue(expr=audio_processing_success, msg="SOX not functional from command line.")


def main():
    unittest.main()

if __name__ == '__main__':
    main()
