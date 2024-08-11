"""
@author: bartulem
Code to run experiments with Motif/Avisoft.
"""

from PyQt6.QtTest import QTest
import configparser
import datetime
import glob
import os
import pathlib
import shutil
import subprocess
import sys
import webbrowser
import motifapi
from send_email import Messenger


class ExperimentController:

    def __init__(self, email_receivers=None, exp_settings_dict=None, message_output=None):

        self.api = None
        self.camera_serial_num = None
        self.config_1 = None
        self.avisoft_recording = None

        if email_receivers is None:
            self.email_receivers = None
        else:
            self.email_receivers = email_receivers

        if exp_settings_dict is None:
            self.exp_settings_dict = None
        else:
            self.exp_settings_dict = exp_settings_dict

        if message_output is None:
            self.message_output = print
        else:
            self.message_output = message_output

    def get_connection_params(self):
        """
        Description
        ----------
        This method gets the IP address and API key to run Motif remotely.
        ----------

        Parameters
        ----------
        ----------

        Returns
        ----------
        master_ip_address (str)
            IP address of the master PC.
        api_key (str)
            API key to run Motif.
        ----------
        """

        config = configparser.ConfigParser()

        if not os.path.exists(f"{self.exp_settings_dict['config_settings_directory']}"
                              f"{os.sep}motif_config.ini"):
            self.message_output("Motif config file not found. Try again!")
            sys.exit()
        else:
            config.read(f"{self.exp_settings_dict['config_settings_directory']}"
                        f"{os.sep}motif_config.ini")
            return config['motif']['master_ip_address'], config['motif']['api']

    def get_custom_dir_names(self, now=None):
        """
        Description
        ----------
        This method creates directories where recording files are copied.
        ----------

        Parameters
        ----------
        now (str)
            Contains year, month, day, hour, minute, second, and microsecond.
        ----------

        Returns
        ----------
        total_dir_name_linux (str)
            Directory location in Linux coordinates.
        total_dir_name_windows (str)
            Directory location in Windows coordinates.
        ----------
        """

        dt = datetime.datetime.fromtimestamp(now)

        sub_dir_name = f"{dt.year:04d}{dt.month:02d}{dt.day:02d}_{dt.hour:02d}{dt.minute:02d}{dt.second:02d}"
        start_hour_min_sec = f"{dt.hour:02d}:{dt.minute:02d}:{dt.second:02d}"

        total_dir_name_linux = []
        total_dir_name_windows = []

        for lin_directory in self.exp_settings_dict['recording_files_destination_linux']:
            total_dir_name_linux.append(f"{lin_directory}/{sub_dir_name}")

        for win_directory in self.exp_settings_dict['recording_files_destination_win']:
            total_dir_name_windows.append(f"{win_directory}\\{sub_dir_name}")

        return start_hour_min_sec, total_dir_name_linux, total_dir_name_windows

    def check_camera_vitals(self, camera_fr=None):
        ip_address, api_key = self.get_connection_params()

        try:
            api = motifapi.MotifApi(ip_address, api_key)
        except motifapi.api.MotifError:
            self.message_output('Motif not running or reachable. Check hardware and connections.')
            sys.exit()
        else:
            available_cameras = api.call('cameras')['cameras']
            self.camera_serial_num = [camera_dict['serial'] for camera_dict in available_cameras]
            self.message_output(f"The system is running Motif v{api.call('version')['software']} "
                                f"and {len(available_cameras)} camera(s) is/are online: {self.camera_serial_num}")

            if len(self.camera_serial_num) != self.exp_settings_dict['video']['general']['expected_camera_num']:
                self.message_output(f"The number of connected cameras ({len(self.camera_serial_num)}) does not match the expected "
                                    f"number of connected cameras, which is {self.exp_settings_dict['video']['general']['expected_camera_num']}.")
                self.message_output(self.exp_settings_dict['continue_key'])
                sys.exit()

            # configure cameras
            for serial_num in self.exp_settings_dict['video']['general']['expected_cameras']:
                api.call(f'camera/{serial_num}/configure',
                         ExposureTime=self.exp_settings_dict['video']['cameras_config'][serial_num]['exposure_time'],
                         Gain=self.exp_settings_dict['video']['cameras_config'][serial_num]['gain'])

            # frame rate has to be same for all
            if len(self.exp_settings_dict['video']['general']['expected_cameras']) == 1:
                api.call(f"camera/{self.exp_settings_dict['video']['general']['expected_cameras'][0]}/configure", AcquisitionFrameRate=camera_fr)
                self.message_output(f"The camera frame rate is set to {camera_fr} fps for {self.exp_settings_dict['video']['general']['expected_cameras'][0]}.")
            else:
                api.call('cameras/configure', MotifMulticamFrameRate=camera_fr)
                self.message_output(f"The camera frame rate is set to {camera_fr} fps for all available cameras.")

            # monitor recording via browser
            if self.exp_settings_dict['video']['general']['monitor_recording']:
                if self.exp_settings_dict['video']['general']['monitor_specific_camera']:
                    meta = api.call(f"camera/{self.exp_settings_dict['video']['general']['specific_camera_serial']}")
                    webbrowser.open(meta['camera_info']['stream']['preview']['url'])
                else:
                    webbrowser.open(self.exp_settings_dict['video']['general']['monitor_url'])

            # pause for N seconds
            QTest.qWait(1000)

            self.api = api

    def conduct_tracking_calibration(self):
        self.message_output(f"Video calibration started at {datetime.datetime.now().hour:02d}:{datetime.datetime.now().minute:02d}.{datetime.datetime.now().second:02d}")

        self.check_camera_vitals(camera_fr=self.exp_settings_dict['video']['general']['calibration_frame_rate'])

        # calibrate tracking cameras
        self.api.call('recording/start',
                      filename='calibration',
                      duration=self.exp_settings_dict['calibration_duration'] * 60,
                      codec=self.exp_settings_dict['video']['general']['recording_codec'])
        QTest.qWait(1000*((self.exp_settings_dict['calibration_duration'] * 60) + 5))

        self.message_output(f"Video calibration completed at {datetime.datetime.now().hour:02d}:{datetime.datetime.now().minute:02d}.{datetime.datetime.now().second:02d}")

    def modify_audio_file(self):
        changes = 0
        if self.exp_settings_dict['modify_audio_config_file']:

            self.config_1 = configparser.ConfigParser()

            self.config_1.read(f"{self.exp_settings_dict['config_settings_directory']}"
                               f"{os.sep}avisoft_config.ini")

            if str(self.exp_settings_dict['avisoft_basedirectory']) != self.config_1['Configuration']['basedirectory']:
                self.config_1['Configuration']['basedirectory'] = self.exp_settings_dict['avisoft_basedirectory']
                changes += 1

            for mic_num in range(self.exp_settings_dict['audio']['total_mic_number']):
                if mic_num in self.exp_settings_dict['audio']['used_mics']:
                    if self.config_1['Configuration'][f"active{mic_num}"] != str(1):
                        self.config_1['Configuration'][f"active{mic_num}"] = str(1)
                        changes += 1
                else:
                    if self.config_1['Configuration'][f"active{mic_num}"] != str(0):
                        self.config_1['Configuration'][f"active{mic_num}"] = str(0)
                        changes += 1

            if self.exp_settings_dict['audio']['general']['total'] == 0:
                if f"{(self.exp_settings_dict['video_session_duration'] + .41)}" != self.config_1['MaxFileSize']['minutes']:
                    self.config_1['MaxFileSize']['minutes'] = f"{(self.exp_settings_dict['video_session_duration'] + .41)}"
                    changes += 1
            else:
                max_file_size = (self.exp_settings_dict['video_session_duration'] + .36) / (self.exp_settings_dict['video_session_duration'] / 5)
                if self.config_1['MaxFileSize']['minutes'] != str(max_file_size):
                    self.config_1['MaxFileSize']['minutes'] = str(max_file_size)
                    changes += 1

            if self.config_1['Configuration']['configfilename'] != f"{self.exp_settings_dict['avisoft_basedirectory']}Configurations{os.sep}RECORDER_USGH{os.sep}avisoft_config.ini":
                self.config_1['Configuration']['configfilename'] = f"{self.exp_settings_dict['avisoft_basedirectory']}Configurations{os.sep}RECORDER_USGH{os.sep}avisoft_config.ini"

            for general_key in self.exp_settings_dict['audio']['general'].keys():
                if str(self.exp_settings_dict['audio']['general'][general_key]) != self.config_1['Configuration'][general_key]:
                    self.config_1['Configuration'][general_key] = str(self.exp_settings_dict['audio']['general'][general_key])
                    changes += 1
                    if general_key == 'total':
                        if self.exp_settings_dict['audio']['general']['total'] == 0:
                            if self.config_1['FileNameMode']['channelname'] != str(0):
                                self.config_1['FileNameMode']['channelname'] = str(0)
                        elif self.exp_settings_dict['audio']['general']['total'] == 1:
                            if self.config_1['FileNameMode']['channelname'] != str(1):
                                self.config_1['FileNameMode']['channelname'] = str(1)

            if self.exp_settings_dict['audio']['mics_config']['triggertype'] == 41:
                timer_duration = self.exp_settings_dict['video_session_duration'] + .36
                if str(timer_duration * 60) != self.config_1['Configuration']['timer']:
                    self.config_1['Configuration']['timer'] = str(timer_duration * 60)
                    changes += 1

            for screen_pos_key in self.exp_settings_dict['audio']['screen_position'].keys():
                if str(self.exp_settings_dict['audio']['screen_position'][screen_pos_key]) != self.config_1['MainWindowPos'][screen_pos_key]:
                    self.config_1['MainWindowPos'][screen_pos_key] = str(self.exp_settings_dict['audio']['screen_position'][screen_pos_key])
                    changes += 1

            for device_num in range(self.exp_settings_dict['audio']['total_device_num']):
                for device_setting_key in self.exp_settings_dict['audio']['devices'].keys():
                    if device_setting_key != "usghflags":
                        if str(self.exp_settings_dict['audio']['devices'][device_setting_key]) != self.config_1['Configuration'][f"{device_setting_key}{device_num}"]:
                            self.config_1['Configuration'][f"{device_setting_key}{device_num}"] = str(self.exp_settings_dict['audio']['devices'][device_setting_key])
                    else:
                        if device_num == 0:
                            self.config_1['Configuration'][f"{device_setting_key}{device_num}"] = str(self.exp_settings_dict['audio']['devices'][device_setting_key])
                        else:
                            self.config_1['Configuration'][f"{device_setting_key}{device_num}"] = str(self.exp_settings_dict['audio']['devices'][device_setting_key] - 2)

            for mic_num in range(self.exp_settings_dict['audio']['total_mic_number']):
                for mic_spec_key in self.exp_settings_dict['audio']['mics_config'].keys():
                    if (mic_num in self.exp_settings_dict['audio']['used_mics'] and f"{mic_spec_key}{mic_num}" not in self.config_1['Configuration'].keys()) or \
                            (mic_num in self.exp_settings_dict['audio']['used_mics'] and str(self.exp_settings_dict['audio']['mics_config'][mic_spec_key]) != self.config_1['Configuration'][f"{mic_spec_key}{mic_num}"]):
                        if mic_spec_key == 'name':
                            self.config_1['Configuration'][f"{mic_spec_key}{mic_num}"] = f"ch{mic_num + 1}"
                            changes += 1
                        elif mic_spec_key == 'deviceid' or mic_spec_key == 'id':
                            if mic_num < 12:
                                self.config_1['Configuration'][f"{mic_spec_key}{mic_num}"] = '0'
                                changes += 1
                            else:
                                self.config_1['Configuration'][f"{mic_spec_key}{mic_num}"] = '1'
                                changes += 1
                        elif mic_spec_key == 'channel':
                            if mic_num < 12:
                                self.config_1['Configuration'][f"{mic_spec_key}{mic_num}"] = f"{mic_num}"
                                changes += 1
                            else:
                                self.config_1['Configuration'][f"{mic_spec_key}{mic_num}"] = f"{mic_num - 12}"
                                changes += 1
                        else:
                            self.config_1['Configuration'][f"{mic_spec_key}{mic_num}"] = str(self.exp_settings_dict['audio']['mics_config'][mic_spec_key])
                            changes += 1

            for mic_num in range(self.exp_settings_dict['audio']['total_mic_number']):
                for monitor_key in self.exp_settings_dict['audio']['monitor'].keys():
                    if (mic_num in self.exp_settings_dict['audio']['used_mics'] and f"{monitor_key}{mic_num}" not in self.config_1['Monitor'].keys()) or \
                            (mic_num in self.exp_settings_dict['audio']['used_mics'] and str(self.exp_settings_dict['audio']['monitor'][monitor_key]) != self.config_1['Monitor'][f"{monitor_key}{mic_num}"]):
                        self.config_1['Monitor'][f"{monitor_key}{mic_num}"] = str(self.exp_settings_dict['audio']['monitor'][monitor_key])
                        changes += 1

            for mic_num in range(self.exp_settings_dict['audio']['total_mic_number']):
                for call_key in self.exp_settings_dict['audio']['call'].keys():
                    if (mic_num in self.exp_settings_dict['audio']['used_mics'] and f"{call_key}{mic_num}" not in self.config_1['Call'].keys()) or \
                            (mic_num in self.exp_settings_dict['audio']['used_mics'] and str(self.exp_settings_dict['audio']['call'][call_key]) != self.config_1['Call'][f"{call_key}{mic_num}"]):
                        self.config_1['Call'][f"{call_key}{mic_num}"] = str(self.exp_settings_dict['audio']['call'][call_key])
                        changes += 1

        if changes > 0:
            self.message_output(f"{changes} lines changed in the avisoft_config_file!")
            with open(f"{self.exp_settings_dict['config_settings_directory']}{os.sep}avisoft_config.ini", 'w') as configfile:
                self.config_1.write(configfile, space_around_delimiters=False)

            if os.path.isfile(f"{self.exp_settings_dict['avisoft_basedirectory']}Configurations{os.sep}RECORDER_USGH{os.sep}avisoft_config.ini"):
                shutil.copy(f"{self.exp_settings_dict['config_settings_directory']}{os.sep}avisoft_config.ini",
                            f"{self.exp_settings_dict['avisoft_basedirectory']}Configurations{os.sep}RECORDER_USGH{os.sep}avisoft_config.ini")
            else:
                shutil.copy(f"{self.exp_settings_dict['config_settings_directory']}{os.sep}avisoft_config.ini",
                            f"{self.exp_settings_dict['avisoft_basedirectory']}Configurations{os.sep}RECORDER_USGH")

            # pause for N seconds
            QTest.qWait(5000)

        else:
            if not os.path.isfile(f"{self.exp_settings_dict['avisoft_basedirectory']}Configurations{os.sep}RECORDER_USGH{os.sep}avisoft_config.ini"):
                shutil.copy(f"{self.exp_settings_dict['config_settings_directory']}{os.sep}avisoft_config.ini",
                            f"{self.exp_settings_dict['avisoft_basedirectory']}Configurations{os.sep}RECORDER_USGH")

            # pause for N seconds
            QTest.qWait(5000)

    def conduct_behavioral_recording(self):
        """
        Description
        ----------
        This method checks whether the system is ready for recording and if so,
        conducts a recording with the designated parameters, and copies the recorded
        files to the network drive (see below for details).

        NB: the data cannot be acquired until Motif, Avisoft USGH recorder and CoolTerm
        have been installed and configured for recording.
        ----------

        Parameters
        ----------
        Contains, among others in the .toml file, the following set of parameters
            conduct_audio_recording (bool)
                To record USV signals with Avisoft software or not; defaults to False.
            expected_camera_num (int)
                The expected number of recording cameras; defaults to 5.
            calibration_frame_rate (int/float)
                The calibration sampling rate; defaults to 10 (fps).
            recording_frame_rate (int/float)
                The recording sampling rate; defaults to 150 (fps).
            recording_duration (int/float)
                Recording duration; defaults to 20 (minutes).
            recording_codec (str)
                Video codec to use; defaults to "lq".
            monitor_recording (bool)
                To monitor or not to monitor the recording; defaults to True.
            monitor_specific_camera (bool)
                To monitor only one specific camera view; defaults to False.
            specific_camera_serial (str)
                The serial number of the specific camera to monitor; defaults to "21372315".
            monitor_url (str)
                URL for monitoring all recording cameras
            copy_files_location (str)
                Location to copy newly recorded files to; defaults to "/home/.../Data".
            delete_post_copy (bool)
                Delete files post copy; defaults to False.
            metadata (dict)
                Metadata to write to file.
            cameras_config (dict)
                Color of each camera, exposure time and gain.
        ----------

        Returns
        ----------
        Directory structure w/ "audio", "sync" and "video" subdirectories.
        The "audio" subdirectory contains "original" and "original_mc" subdirectories,
        the "sync" subdirectory contains .txt files (serial monitor output),
        and the "video" subdirectory contains individual camera video files (.mp4),
        each in its own subdirectory.
        ----------
        """

        Messenger(message_output=self.message_output,
                  receivers=self.email_receivers,
                  exp_settings_dict=self.exp_settings_dict).send_message(subject="Audio PC in 165B is busy, do NOT attempt to remote in!",
                                                                         message=f"Experiment in progress, started at "
                                                                                 f"{datetime.datetime.now().hour:02d}:{datetime.datetime.now().minute:02d}.{datetime.datetime.now().second:02d} "
                                                                                 f"and run by @{self.exp_settings_dict['video']['metadata']['experimenter']}. "
                                                                                 f"You will be notified upon completion. \n \n ***This is an automatic e-mail, please do NOT respond.***")

        # start capturing sync LEDS
        if not os.path.isfile(f"{self.exp_settings_dict['coolterm_basedirectory']}{os.sep}Connection_settings{os.sep}coolterm_config.stc"):
            shutil.copy(f"{self.exp_settings_dict['config_settings_directory']}{os.sep}coolterm_config.stc",
                        f"{self.exp_settings_dict['coolterm_basedirectory']}{os.sep}Connection_settings{os.sep}coolterm_config.stc")
        sync_leds_capture = subprocess.Popen(args=f'''cmd /c "{self.exp_settings_dict['coolterm_basedirectory']}{os.sep}Connection_settings{os.sep}coolterm_config.stc"''',
                                             stdout=subprocess.PIPE)

        self.check_camera_vitals(camera_fr=self.exp_settings_dict['video']['general']['recording_frame_rate'])

        # modify audio config file
        if self.exp_settings_dict['conduct_audio_recording']:
            self.modify_audio_file()

        start_hour_min_sec, total_dir_name_linux, total_dir_name_windows = self.get_custom_dir_names(now=self.api.call('schedule')['now'])

        for directory in total_dir_name_windows:
            pathlib.Path(f"{directory}{os.sep}video").mkdir(parents=True, exist_ok=True)
            pathlib.Path(f"{directory}{os.sep}sync").mkdir(parents=True, exist_ok=True)
            if self.exp_settings_dict['conduct_audio_recording']:
                if self.exp_settings_dict['audio']['general']['total'] == 0:
                    pathlib.Path(f"{directory}{os.sep}audio{os.sep}original").mkdir(parents=True, exist_ok=True)
                else:
                    pathlib.Path(f"{directory}{os.sep}audio{os.sep}original").mkdir(parents=True, exist_ok=True)
                    pathlib.Path(f"{directory}{os.sep}audio{os.sep}original_mc").mkdir(parents=True, exist_ok=True)

        # start recording audio
        if self.exp_settings_dict['conduct_audio_recording']:
            # run command to start audio recording and keep executing the rest of the script
            if os.path.exists(f"{self.exp_settings_dict['avisoft_basedirectory']}{os.sep}Configurations{os.sep}RECORDER_USGH{os.sep}avisoft_config.ini"):
                self.avisoft_recording = subprocess.Popen(args=f'''cmd /c ""rec_usgh.exe" /CFG=avisoft_config.ini /AUT"''',
                                                          stdout=subprocess.PIPE,
                                                          cwd=self.exp_settings_dict['avisoft_recorder_exe'])
                self.message_output(f"Recording in progress since {start_hour_min_sec}, it will last {self.exp_settings_dict['video_session_duration'] + .36} minute(s). Please be patient.")

                # pause for N seconds
                QTest.qWait(10000)

        # record video data
        if len(self.exp_settings_dict['video']['general']['expected_cameras']) == 1:
            self.message_output(f"You chose to conduct the recording with one camera only (camera serial num: {self.exp_settings_dict['video']['general']['expected_cameras'][0]}).")
            self.api.call(f"camera/{self.exp_settings_dict['video']['general']['expected_cameras'][0]}/recording/start",
                          duration=self.exp_settings_dict['video_session_duration'] * 60,
                          codec=self.exp_settings_dict['video']['general']['recording_codec'],
                          metadata=self.exp_settings_dict['video']['metadata'])
        else:
            self.api.call('recording/start',
                          duration=self.exp_settings_dict['video_session_duration'] * 60,
                          codec=self.exp_settings_dict['video']['general']['recording_codec'],
                          metadata=self.exp_settings_dict['video']['metadata'])

        if not self.exp_settings_dict['conduct_audio_recording']:
            self.message_output(f"Video recording in progress since {start_hour_min_sec}, it will last {self.exp_settings_dict['video_session_duration']} minute(s). Please be patient.")

        # wait until cameras have finished recording
        # pause for N extra seconds so audio is done, too
        QTest.qWait(1000*(10 + (self.exp_settings_dict['video_session_duration'] * 60)))

        self.message_output(f"Recording fully completed at {datetime.datetime.now().hour:02d}:{datetime.datetime.now().minute:02d}.{datetime.datetime.now().second:02d}.")

        # pause for N seconds
        QTest.qWait(10000)

        # close sync LED capture
        sync_leds_capture.terminate()
        subprocess.Popen(f'''cmd /c "taskkill /IM CoolTerm.exe /T /F 1>nul 2>&1"''').wait()

        self.message_output(f"Copying audio/video files started at: {datetime.datetime.now().hour:02d}:{datetime.datetime.now().minute:02d}.{datetime.datetime.now().second:02d}")

        # copy sync file to file server(s)
        last_modified_sync_file = max(glob.glob(f"{self.exp_settings_dict['coolterm_basedirectory']}{os.sep}Data{os.sep}*.txt"), key=os.path.getctime)
        for win_dir_idx, win_dir in enumerate(total_dir_name_windows):
            if not win_dir_idx == len(total_dir_name_windows) - 1:
                shutil.copy(src=last_modified_sync_file,
                            dst=f"{win_dir}{os.sep}sync{os.sep}{last_modified_sync_file.split(os.sep)[-1]}")
            else:
                shutil.move(src=last_modified_sync_file,
                            dst=f"{win_dir}{os.sep}sync{os.sep}{last_modified_sync_file.split(os.sep)[-1]}")

        # copy video files to file server(s)
        if len(self.exp_settings_dict['video']['general']['expected_cameras']) == 1:
            copy_video_command = f"camera/{self.exp_settings_dict['video']['general']['expected_cameras'][0]}/recordings/copy_all"
        else:
            copy_video_command = 'recordings/copy_all'

        self.api.call(copy_video_command,
                      delete_after=self.exp_settings_dict['video']['general']['delete_post_copy'],
                      location=f"{total_dir_name_linux[0]}/video")

        # copy audio files to file server(s)
        if self.exp_settings_dict['conduct_audio_recording']:
            audio_copy_subprocesses = []
            if self.exp_settings_dict['audio']['general']['total'] == 0:
                for mic_idx in self.exp_settings_dict['audio']['used_mics']:
                    last_modified_audio_file = max(glob.glob(f"{self.exp_settings_dict['avisoft_basedirectory']}{os.sep}ch{mic_idx + 1}{os.sep}*.wav"),
                                                   key=os.path.getctime)
                    single_audio_copy_subp = subprocess.Popen(f'''cmd /c move "{last_modified_audio_file.split(os.sep)[-1]}" "{total_dir_name_windows[0]}{os.sep}audio{os.sep}original{os.sep}ch{mic_idx + 1}_{last_modified_audio_file.split(os.sep)[-1]}"''',
                                                              cwd=f"{self.exp_settings_dict['avisoft_basedirectory']}{os.sep}ch{mic_idx + 1}",
                                                              stdout=subprocess.DEVNULL,
                                                              stderr=subprocess.STDOUT,
                                                              shell=False)
                    audio_copy_subprocesses.append(single_audio_copy_subp)
            else:
                device_id = ['m', 's']
                for mic_pos_idx, mic_idx in enumerate([0, 12]):
                    audio_file_list = glob.glob(f"{self.exp_settings_dict['avisoft_basedirectory']}{os.sep}ch{mic_idx + 1}{os.sep}*.wav")
                    for aud_file in audio_file_list:
                        multi_audio_copy_subp = subprocess.Popen(f'''cmd /c move "{aud_file.split(os.sep)[-1]}" "{total_dir_name_windows[0]}{os.sep}audio{os.sep}original_mc{os.sep}{device_id[mic_pos_idx]}_{aud_file.split(os.sep)[-1]}"''',
                                                                 cwd=f"{self.exp_settings_dict['avisoft_basedirectory']}{os.sep}ch{mic_idx + 1}",
                                                                 stdout=subprocess.DEVNULL,
                                                                 stderr=subprocess.STDOUT,
                                                                 shell=False)
                        audio_copy_subprocesses.append(multi_audio_copy_subp)

            while True:
                status_poll = [query_subp.poll() for query_subp in audio_copy_subprocesses]
                if any(elem is None for elem in status_poll):
                    QTest.qWait(2000)
                else:
                    break

        # ensure the video is done copying before proceeding
        while any(self.api.is_copying(_sn) for _sn in self.camera_serial_num):
            QTest.qWait(2000)

        # copy the video and audio directories to the backup network drive(s)
        if len(total_dir_name_windows) > 1:
            directory_copy_subprocesses = []
            for win_dir_idx, win_dir in enumerate(total_dir_name_windows[1:]):
                for dir_type in ['audio', 'video']:
                    directory_copy_subp = subprocess.Popen(f'''cmd /c xcopy "{total_dir_name_windows[0]}{os.sep}{dir_type}" "{win_dir}{os.sep}{dir_type}" /E /H /Q /Y''',
                                                           cwd=f"{total_dir_name_windows[0]}{os.sep}{dir_type}",
                                                           stdout=subprocess.DEVNULL,
                                                           stderr=subprocess.STDOUT,
                                                           shell=False)
                    directory_copy_subprocesses.append(directory_copy_subp)

            while True:
                status_poll = [query_subp.poll() for query_subp in directory_copy_subprocesses]
                if any(elem is None for elem in status_poll):
                    QTest.qWait(2000)
                else:
                    break

        self.message_output(f"Copying audio/video files finished at: {datetime.datetime.now().hour:02d}:{datetime.datetime.now().minute:02d}.{datetime.datetime.now().second:02d}")

        Messenger(message_output=self.message_output,
                  receivers=self.email_receivers,
                  no_receivers_notification=False,
                  exp_settings_dict=self.exp_settings_dict).send_message(subject="Audio PC in 165B is available again, recording has been completed.",
                                                                         message=f"Thank you for your patience, recording by @{self.exp_settings_dict['video']['metadata']['experimenter']} was completed at "
                                                                                 f"{datetime.datetime.now().hour:02d}:{datetime.datetime.now().minute:02d}.{datetime.datetime.now().second:02d}. "
                                                                                 f"You will be notified about further experiments "
                                                                                 f"should they occur. \n \n ***This is an automatic e-mail, please do NOT respond.***")
