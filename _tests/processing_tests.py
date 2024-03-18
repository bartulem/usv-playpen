"""
@author: bartulem
Testing processing module.
"""

import os
import subprocess
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import toml
import unittest
from send_email import Messenger
import usv_playpen_gui as usv_playpen_gui


class TestProcessing(unittest.TestCase):

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
            subprocess.Popen(f'''cmd /c "ffmpeg -version"''').wait()
            video_processing_success = True
        except subprocess.CalledProcessError as e:
            print(e)
            video_processing_success = False

        self.assertTrue(expr=video_processing_success, msg="FFMPEG not functional from command line.")

    def test_audio_processing(self):

        try:
            subprocess.Popen(f'''cmd /c "sox --version"''').wait()
            audio_processing_success = True
        except subprocess.CalledProcessError as e:
            print(e)
            audio_processing_success = False

        self.assertTrue(expr=audio_processing_success, msg="SOX not functional from command line.")


if __name__ == '__main__':
    unittest.main()
