"""
@author: bartulem
Testing recording module.
"""

import configparser
import motifapi
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import toml
import unittest
from send_email import Messenger
import usv_playpen_gui as usv_playpen_gui


class TestRecording(unittest.TestCase):

    exp_settings_dict = toml.load(f"{usv_playpen_gui.config_dir_global}{os.sep}behavioral_experiments_settings.toml")

    def test_recording_send_email(self):

        # test email sending
        try:
            Messenger(receivers=usv_playpen_gui.email_list_global.split(","),
                      exp_settings_dict=self.exp_settings_dict).send_message(subject="Test", message="This is a 165B recording test email. Please do not reply.")
            email_success = True
        except Exception:
            email_success = False

        self.assertTrue(expr=email_success, msg="165B not able to send e-mails")

    def test_recording_video_software(self):

        # test video recording capabilities
        video_config = configparser.ConfigParser()
        video_config.read(f"{self.exp_settings_dict['config_settings_directory']}{os.sep}motif_config.ini")

        try:
            self.motif_api = motifapi.MotifApi(video_config['motif']['master_ip_address'], video_config['motif']['api'])
            motif_success = True
        except motifapi.api.MotifError:
            motif_success = False

        self.assertTrue(expr=motif_success, msg="Motif not operational on this PC.")

    def test_recording_audio_sync_software(self):

        # test audi0 / sync_software presence
        try:
            config_files_exist = (os.path.isfile(f"{self.exp_settings_dict['config_settings_directory']}{os.sep}avisoft_config.ini") and
                                  os.path.isfile(f"{self.exp_settings_dict['config_settings_directory']}{os.sep}coolterm_config.stc"))
            if not config_files_exist:
                config_file_status = False
            else:
                with open(f"{self.exp_settings_dict['config_settings_directory']}{os.sep}coolterm_config.stc") as coolterm_config_file:
                    for line in coolterm_config_file:
                        key, value = line.strip().split("=")
                        if key == 'CapturePath' and value == 'D:\CoolTermWin\Data':
                            config_sync_status = True
                            break
                    else:
                        config_file_status = False

                audio_config = configparser.ConfigParser()
                audio_config.read(f"{self.exp_settings_dict['config_settings_directory']}{os.sep}avisoft_config.ini")
                if audio_config['Configuration']['basedirectory'] == 'C:\\Users\\bmimica\\Documents\\Avisoft Bioacoustics\\':
                    config_audio_status = True
                else:
                    config_audio_status = False

                if config_audio_status and config_sync_status:
                    config_file_status = True
                else:
                    config_file_status = False
        except Exception:
            config_file_status = False

        software_status = (os.path.isfile(f"{self.exp_settings_dict['avisoft_recorder_exe']}{os.sep}rec_usgh.exe") and
                           os.path.isfile(f"{self.exp_settings_dict['coolterm_basedirectory']}{os.sep}CoolTerm.exe"))

        self.assertTrue(expr=config_file_status and software_status, msg="Audio / SYNC software and/or config files not ready for use.")


if __name__ == '__main__':
    unittest.main()