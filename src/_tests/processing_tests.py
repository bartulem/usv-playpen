"""
@author: bartulem
Testing processing module.
"""

import os
import subprocess
import sys
import toml
import unittest
sys.path.append('..')
from src.send_email import Messenger
from src import usv_playpen_gui as usv_playpen_gui


class TestProcessing(unittest.TestCase):

    if os.name == 'nt':
        command_addition = 'cmd /c '
        shell_usage_bool = False
    else:
        command_addition = ''
        shell_usage_bool = True

    esd = toml.load(f"{usv_playpen_gui.config_dir_global}{os.sep}behavioral_experiments_settings.toml")


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


if __name__ == '__main__':
    unittest.main()
