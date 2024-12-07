"""
@author: bartulem
Test recording module.
"""

import configparser
import motifapi
import os
import platform
import sys
import toml
import unittest
from ..send_email import Messenger


class TestRecording(unittest.TestCase):

    if platform.system() == 'Windows':
        config_dir_global = 'C:\\experiment_running_docs'
    elif platform.system() == 'Linux':
        config_dir_global = f'/mnt/falkner/Bartul/PC_transfer/experiment_running_docs'
    else:
        config_dir_global = f'/Volumes/falkner/Bartul/PC_transfer/experiment_running_docs'

    exp_settings_dict = toml.load(f"{config_dir_global}{os.sep}behavioral_experiments_settings.toml")

    def test_recording_send_email(self):

        try:
            email_receiver = [str(self.email_address)]
        except IndexError:
            print("Error: Missing e-mail argument. Please provide it.")

        # test email sending
        try:
            Messenger(receivers=email_receiver,
                      exp_settings_dict=self.exp_settings_dict).send_message(subject="Test", message="This is a 165B recording test email. Please do not reply.")
            email_success = True
        except Exception:
            email_success = False

        self.assertTrue(expr=email_success, msg="165B not able to send e-mails")

    def test_recording_video_software(self):

        # test video recording capabilities
        video_config = configparser.ConfigParser()
        video_config.read(f"{self.config_dir_global}{os.sep}motif_config.ini")

        try:
            self.motif_api = motifapi.MotifApi(video_config['motif']['master_ip_address'], video_config['motif']['api'])
            motif_success = True
        except motifapi.api.MotifError:
            motif_success = False

        self.assertTrue(expr=motif_success, msg="Motif not operational on this PC.")

    def test_recording_audio_sync_software(self):

        # test audi0 / sync_software presence
        config_file_status = (os.path.isfile(f"{self.config_dir_global}{os.sep}avisoft_config.ini") and
                              os.path.isfile(f"{self.config_dir_global}{os.sep}coolterm_config.stc"))

        software_status = (os.path.isfile(f"{self.exp_settings_dict['avisoft_recorder_exe']}{os.sep}rec_usgh.exe") and
                           os.path.isfile(f"{self.exp_settings_dict['coolterm_basedirectory']}{os.sep}CoolTerm.exe"))

        self.assertTrue(expr=config_file_status and software_status, msg="Audio / SYNC software and/or config files not ready for use.")


if __name__ == '__main__':
    TestRecording.email_address = sys.argv.pop()
    unittest.main()
