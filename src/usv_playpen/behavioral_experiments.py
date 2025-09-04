"""
@author: bartulem
Runs experiments with Avisoft/CoolTerm/Loopbio software.
"""

import configparser
import datetime
import glob
import math
import os
import paramiko
import shutil
import subprocess
import sys
import toml
import webbrowser
import motifapi
from .cli_utils import *
from .send_email import Messenger
from .time_utils import *


def count_last_recording_dropouts(log_file_path: str,
                                  log_file_ch : str) -> int | None:
    """
    Description
    ----------
    Reads a log file, identifies the last recording session, and counts the number of dropouts.
    ----------

    Parameters
    ----------
    log_file_path (str)
        The directory of the log file (typically Avisoft basedirectory).
    log_file_ch (str)
        The channel number for which the log file is being read (e.g., 'ch1').
        This is used to construct the full path to the log file.
    ----------

    Returns
    -------
    (int | None)
       The number of dropouts in the last recording, or 0 if no recording is found,
       or None if the chX.log file cannot be found.
    -------
    """

    try:
        with open(f"{log_file_path}{log_file_ch}{os.sep}{log_file_ch}.log", 'r') as log_txt_file:
            content = log_txt_file.read()
    except FileNotFoundError:
        return None

    recordings = content.split(f"{log_file_path}{log_file_ch}{os.sep}")

    # filter out any empty strings that may result from the split
    recordings = [rec for rec in recordings if rec.strip()]

    if not recordings:
        return 0

    # get the last recording block
    last_recording = recordings[-1]

    # count the occurrences of "dropout" in the last recording
    dropout_count = last_recording.count('dropout')

    return dropout_count


class ExperimentController:

    def __init__(self, email_receivers: list = None,
                 exp_settings_dict: dict = None,
                 message_output: callable = None) -> None:

        """
        Initializes the ExperimentController class.

        Parameter
        ---------
        exp_settings_dict (dict)
            Experiment settings; defaults to None.
        email_receivers (list)
            Email receivers; defaults to None.
        message_output (function)
            Defines output messages; defaults to None.

        Returns
        -------
        -------
        """

        self.api = None
        self.camera_serial_num = None
        self.config_1 = None

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

        self.app_context_bool = is_gui_context()

    def check_ethernet_connection(self) -> None:
        """
        Description
        -----------
        Checks if the specified Ethernet adapter is connected;
        if not, it re-enables the connection.
        -----------

        Parameters
        ----------
        ----------

        Returns
        -------
        -------
        """

        # check if the Ethernet adapter is connected
        ethernet_command_list = ["powershell", "-Command", f'''& {{ (Get-NetAdapter -Name "{self.exp_settings_dict['ethernet_network']}").Status -eq "Up" }}''']
        ethernet_status = subprocess.run(args=ethernet_command_list, capture_output=True, text=True, check=True, encoding='utf-8')

        # reconnect if not connected
        if ethernet_status.returncode == 0:
            ethernet_status_output_text = ethernet_status.stdout.strip()
            if ethernet_status_output_text.lower() == 'false':
                subprocess.Popen(args=f'''cmd /c netsh interface set interface "{self.exp_settings_dict['ethernet_network']}" enable''').wait()

                # pause for N seconds
                smart_wait(app_context_bool=self.app_context_bool, seconds=15)

    def remount_cup_drives_on_windows(self) -> None:
        """
        Description
        -----------
        Checks if the specified Ethernet adapter is connected; if not,
        reconnects and attempts to remount CUP drives (if they are not mounted).
        -----------

        Parameters
        ----------
        ----------

        Returns
        -------
        -------
        """

        cup_username, cup_password = self.get_cup_mount_params()

        drives_to_mount = {"F:": r"\\cup\falkner", "M:": r"\\cup\murthy"}

        self.check_ethernet_connection()

        # check if all CUP drives are mounted and remount CUP drives if they are not
        mounted_drives_command_list = ["powershell", "-Command", '''& { gdr -PSProvider "FileSystem" | Select-Object -ExpandProperty Name }''']
        mounted_drives_status = subprocess.run(args=mounted_drives_command_list, capture_output=True, text=True, check=True, encoding='utf-8')

        if mounted_drives_status.returncode == 0:
            mounted_drives = sorted(list(set(mounted_drives_status.stdout.strip().split())))

            for drive_letter_with_colon, path in drives_to_mount.items():
                drive_letter_only = drive_letter_with_colon.replace(":", "")
                if drive_letter_only not in mounted_drives:
                    subprocess.Popen(args=f'''cmd /c net use {drive_letter_with_colon.lower()} {path} /user:{cup_username}@princeton.edu {cup_password} /persistent:yes''',
                                     stdout=subprocess.DEVNULL,
                                     stderr=subprocess.STDOUT).wait()
                    self.message_output(f"[**Local mount check**]'{drive_letter_with_colon}' has now been mounted on this PC.")
                else:
                    self.message_output(f"[**Local mount check**]'{drive_letter_with_colon}' is already mounted on this PC.")


    def check_remote_mount(self,
                           hostname: str = None,
                           port: int = None,
                           username: str = None,
                           password: str = None,
                           mount_path: str = None) -> bool:
        """
        Description
        ----------
        Connects to a remote host via SSH and checks if a path is a valid mount point.
        ----------

        Parameters
        ----------
        hostname (str)
            The IP address or hostname of the remote machine.
        port (int)
            The SSH port (usually 22).
        username (str)
            The username for the SSH connection.
        password (str)
            The password for the SSH connection.
        mount_path (str)
            The absolute path of the mount point to check.
        ----------

        Returns
        -------
        (bool)
            True if the path is a mount point, False otherwise.
        """

        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        try:
            client.connect(hostname=hostname, port=port, username=username, password=password, timeout=10)

            command = f"python3 -c \"import os; print(os.path.ismount('{mount_path}'))\""

            stdin, stdout, stderr = client.exec_command(command)
            result = stdout.read().decode('utf-8').strip()
            error = stderr.read().decode('utf-8').strip()

            if error:
                self.message_output(f"[**Remote mount check**] On {hostname}, an error occurred on the remote machine: {error}")
                return False
            elif result == "True":
                self.message_output(f"[**Remote mount check**] On {hostname} '{mount_path}' is a mounted filesystem.")
                return True
            elif result == "False":
                self.message_output(f"[**Remote mount check**] On {hostname}, '{mount_path}' exists but is not a mount point, or the path does not exist.")
                return False
            else:
                self.message_output(f"[**Remote mount check**] On {hostname}, received an unexpected result: {result}")
                return False

        except paramiko.AuthenticationException:
            self.message_output(f"[**Remote mount check**] On {hostname}, authentication failed. Please check your username and password.")
            return False
        except paramiko.SSHException as e:
            self.message_output(f"[**Remote mount check**] On {hostname}, SSH connection error: {e}")
            return False
        except Exception as e:
            self.message_output(f"[**Remote mount check**] On {hostname}, n unexpected error occurred: {e}")
            return False
        finally:
            client.close()

    def get_cpu_affinity_mask(self) -> str:
        """
        Description
        ----------
        This method gets the CPU affinity mask for the Avisoft software.
        ----------

        Parameters
        ----------
        ----------

        Returns
        -------
        (str)
            CPU affinity mask.
        """

        bitmask = 0
        for cpu in self.exp_settings_dict['audio']['cpu_affinity']:
            bitmask |= 1 << cpu
        return f"0x{bitmask:X}"

    def get_cup_mount_params(self) -> tuple:
        """
        Description
        ----------
        This method gets the username and password for cup mounting.
        ----------

        Parameters
        ----------
        ----------

        Returns
        ----------
        username (str), password (str)
            Username and password for cup mounting.
        ----------
        """

        config = configparser.ConfigParser()

        if not os.path.exists(f"{self.exp_settings_dict['credentials_directory']}{os.sep}cup_config.ini"):
            self.message_output("Cup config file not found. Try again!")
            smart_wait(app_context_bool=self.app_context_bool, seconds=10)
            sys.exit()
        else:
            config.read(f"{self.exp_settings_dict['credentials_directory']}{os.sep}cup_config.ini")
            return config['cup']['username'], config['cup']['password']

    def get_connection_params(self) -> tuple:
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
        master_ip_address (str), api_key (str), second_ip_address (str),
        ssh_port (str), ssh_username (str), ssh_password (str)
            IP address of the main PC, API key to run Motif remotely, IP address of the second PC,
            SSH port, SSH username, and SSH password.
        ----------
        """

        config = configparser.ConfigParser()



        if not os.path.exists(f"{self.exp_settings_dict['credentials_directory']}{os.sep}motif_config.ini"):
            self.message_output("Motif config file not found. Try again!")
            smart_wait(app_context_bool=self.app_context_bool, seconds=10)
            sys.exit()
        else:
            config.read(f"{self.exp_settings_dict['credentials_directory']}{os.sep}motif_config.ini")
            return (config['motif']['master_ip_address'], config['motif']['second_ip_address'],
                    config['motif']['ssh_port'], config['motif']['ssh_username'],
                    config['motif']['ssh_password'], config['motif']['api'])

    def get_custom_dir_names(self, now: float = None) -> tuple:
        """
        Description
        ----------
        This method creates directories where recording files are copied.
        ----------

        Parameters
        ----------
        now (float)
            Contains year, month, day, hour, minute, second, and microsecond.
        ----------

        Returns
        ----------
        total_dir_name_linux (str), total_dir_name_windows (str)
            Directory location in Linux and Window coordinates.
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

    def check_camera_vitals(self, camera_fr: int | float = None) -> None:
        """
        Description
        ----------
        This method checks whether Motif is operational.
        ----------

        Parameters
        ----------
        camera_fr (int / float)
            Camera sampling rate; defaults to None.
        ----------

        Returns
        ----------
        ----------
        """

        ip_address, second_ip_address, ssh_port, ssh_username, ssh_password, api_key  = self.get_connection_params()

        # check the existence and functionality of mount points on both tracking computers
        for ip_address_pc in [ip_address, second_ip_address]:
            for lin_directory in self.exp_settings_dict['recording_files_destination_linux']:
                lin_dir_elements = lin_directory.split('/')[0:3]
                mount_check_bool = self.check_remote_mount(hostname=ip_address_pc, port=int(ssh_port), username=ssh_username, password=ssh_password, mount_path='/'.join(lin_dir_elements))
                if not mount_check_bool:
                    self.message_output(f"Mount point {lin_directory} on {ip_address} is not valid. Please fix the issue and try again.")
                    smart_wait(app_context_bool=self.app_context_bool, seconds=10)
                    sys.exit()

        try:
            api = motifapi.MotifApi(ip_address, api_key)
        except motifapi.api.MotifError:
            self.message_output('Motif not running or reachable. Check hardware and connections.')
            smart_wait(app_context_bool=self.app_context_bool, seconds=10)
            sys.exit()
        else:
            if 1 < len(self.exp_settings_dict['video']['general']['expected_cameras']) < 5:
                temp_camera_serial_num = [camera_dict['serial'] for camera_dict in api.call('cameras')['cameras']]

                # connect cameras if necessary
                if len(list(set(self.exp_settings_dict['video']['general']['expected_cameras'])-set(temp_camera_serial_num))) > 0:
                    for camera_serial in list(set(self.exp_settings_dict['video']['general']['expected_cameras'])-set(temp_camera_serial_num)):
                        api.call(f'multicam/connect_camera/{camera_serial}')

                # disconnect cameras if necessary
                if len(list(set(temp_camera_serial_num) - set(self.exp_settings_dict['video']['general']['expected_cameras']))) > 0:
                    for camera_serial in list(set(temp_camera_serial_num) - set(self.exp_settings_dict['video']['general']['expected_cameras'])):
                        api.call(f'multicam/disconnect_camera/{camera_serial}')

            available_cameras = api.call('cameras')['cameras']
            self.camera_serial_num = [camera_dict['serial'] for camera_dict in available_cameras]
            self.message_output(f"The system is running Motif v{api.call('version')['software']} "
                                f"and {len(available_cameras)} camera(s) is/are online: {self.camera_serial_num}")

            # configure cameras
            for serial_num in self.exp_settings_dict['video']['general']['expected_cameras']:
                api.call(f'camera/{serial_num}/configure',
                         ExposureTime=self.exp_settings_dict['video']['cameras_config'][serial_num]['exposure_time'],
                         Gain=self.exp_settings_dict['video']['cameras_config'][serial_num]['gain'])

            # the frame rate has to be the same for all
            if len(self.exp_settings_dict['video']['general']['expected_cameras']) == 1:
                api.call(f"camera/{self.exp_settings_dict['video']['general']['expected_cameras'][0]}/configure", AcquisitionFrameRate=camera_fr)
                self.message_output(f"The camera frame rate is set to {camera_fr} fps for {self.exp_settings_dict['video']['general']['expected_cameras'][0]}.")
            else:
                api.call(f'cameras/configure', MotifMulticamFrameRate=camera_fr)
                self.message_output(f"The camera frame rate is set to {camera_fr} fps for all chosen cameras.")

            # monitor recording via browser
            if self.exp_settings_dict['video']['general']['monitor_recording']:
                if self.exp_settings_dict['video']['general']['monitor_specific_camera']:
                    meta = api.call(f"camera/{self.exp_settings_dict['video']['general']['specific_camera_serial']}")
                    webbrowser.open(meta['camera_info']['stream']['preview']['url'])
                else:
                    webbrowser.open(self.exp_settings_dict['video']['general']['monitor_url'])

            # pause for N seconds
            smart_wait(app_context_bool=self.app_context_bool, seconds=10)

            self.api = api

    def conduct_tracking_calibration(self) -> None:
        """
        Description
        ----------
        This method conducts tracking calibration.
        ----------

        Parameters
        ----------
        ----------

        Returns
        ----------
        ----------
        """

        self.message_output(f"Video calibration started at {datetime.datetime.now().hour:02d}:{datetime.datetime.now().minute:02d}.{datetime.datetime.now().second:02d}")

        self.check_camera_vitals(camera_fr=self.exp_settings_dict['video']['general']['calibration_frame_rate'])

        # calibrate tracking cameras
        self.api.call('recording/start',
                      filename='calibration',
                      duration=self.exp_settings_dict['calibration_duration'] * 60,
                      codec=self.exp_settings_dict['video']['general']['recording_codec'])
        smart_wait(app_context_bool=self.app_context_bool, seconds=(self.exp_settings_dict['calibration_duration'] * 60) + 5)

        self.message_output(f"Video calibration completed at {datetime.datetime.now().hour:02d}:{datetime.datetime.now().minute:02d}.{datetime.datetime.now().second:02d}")

    def modify_audio_file(self) -> None:
        """
        Description
        ----------
        This method modifies the Avisoft config file.
        ----------

        Parameters
        ----------
        ----------

        Returns
        ----------
        ----------
        """

        changes = 0

        self.config_1 = configparser.ConfigParser()
        self.config_1.read(os.path.join(os.path.dirname(os.path.abspath(__file__)), '_config/avisoft_config.ini'))

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
            if not math.isclose ((self.exp_settings_dict['video_session_duration'] + .41), float(self.config_1['MaxFileSize']['minutes'])):
                self.config_1['MaxFileSize']['minutes'] = f"{(self.exp_settings_dict['video_session_duration'] + .41)}"
                changes += 1
        else:
            max_file_size = (self.exp_settings_dict['video_session_duration'] + .36) / (self.exp_settings_dict['video_session_duration'] / 5)
            if not math.isclose(float(self.config_1['MaxFileSize']['minutes']), max_file_size):
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
            if not math.isclose((timer_duration * 60), float(self.config_1['Configuration']['timer'])):
                self.config_1['Configuration']['timer'] = str(timer_duration * 60)
                changes += 1

        for screen_pos_key in self.exp_settings_dict['audio']['screen_position'].keys():
            if str(self.exp_settings_dict['audio']['screen_position'][screen_pos_key]) != self.config_1['MainWindowPos'][screen_pos_key]:
                self.config_1['MainWindowPos'][screen_pos_key] = str(self.exp_settings_dict['audio']['screen_position'][screen_pos_key])
                changes += 1

        for device_num in range(self.exp_settings_dict['audio']['total_device_num']):
            for device_setting_key in self.exp_settings_dict['audio']['devices'].keys():
                if device_setting_key != "usghflags":
                    if not math.isclose(self.exp_settings_dict['audio']['devices'][device_setting_key], float(self.config_1['Configuration'][f"{device_setting_key}{device_num}"])):
                        self.config_1['Configuration'][f"{device_setting_key}{device_num}"] = str(self.exp_settings_dict['audio']['devices'][device_setting_key])
                        changes += 1
                else:
                    if device_num == 0:
                        if not math.isclose(self.exp_settings_dict['audio']['devices'][device_setting_key], float(self.config_1['Configuration'][f"{device_setting_key}{device_num}"])):
                            self.config_1['Configuration'][f"{device_setting_key}{device_num}"] = str(self.exp_settings_dict['audio']['devices'][device_setting_key])
                            changes += 1
                    else:
                        if self.exp_settings_dict['audio']['devices'][device_setting_key] == 1574:
                            if not math.isclose(self.exp_settings_dict['audio']['devices'][device_setting_key], float(self.config_1['Configuration'][f"{device_setting_key}{device_num}"])):
                                self.config_1['Configuration'][f"{device_setting_key}{device_num}"] = str(self.exp_settings_dict['audio']['devices'][device_setting_key] - 2)
                                changes += 1
                        else:
                            if not math.isclose(self.exp_settings_dict['audio']['devices'][device_setting_key], float(self.config_1['Configuration'][f"{device_setting_key}{device_num}"])):
                                self.config_1['Configuration'][f"{device_setting_key}{device_num}"] = str(self.exp_settings_dict['audio']['devices'][device_setting_key])
                                changes += 1

        for mic_num in range(self.exp_settings_dict['audio']['total_mic_number']):
            for mic_spec_key in self.exp_settings_dict['audio']['mics_config'].keys():
                if mic_spec_key in ['name', 'deviceid', 'id', 'channel', 'ditctime']:
                    if (mic_num in self.exp_settings_dict['audio']['used_mics'] and f"{mic_spec_key}{mic_num}" not in self.config_1['Configuration'].keys()) or \
                            (mic_num in self.exp_settings_dict['audio']['used_mics'] and str(self.exp_settings_dict['audio']['mics_config'][mic_spec_key]) != self.config_1['Configuration'][f"{mic_spec_key}{mic_num}"]):
                        if mic_spec_key == 'name':
                            if self.config_1['Configuration'][f"{mic_spec_key}{mic_num}"] != f"ch{mic_num + 1}":
                                self.config_1['Configuration'][f"{mic_spec_key}{mic_num}"] = f"ch{mic_num + 1}"
                                changes += 1
                        elif mic_spec_key == 'deviceid' or mic_spec_key == 'id':
                            if mic_num < 12:
                                if self.config_1['Configuration'][f"{mic_spec_key}{mic_num}"] != '0':
                                    self.config_1['Configuration'][f"{mic_spec_key}{mic_num}"] = '0'
                                    changes += 1
                            else:
                                if self.config_1['Configuration'][f"{mic_spec_key}{mic_num}"] != '1':
                                    self.config_1['Configuration'][f"{mic_spec_key}{mic_num}"] = '1'
                                    changes += 1
                        elif mic_spec_key == 'channel':
                            if mic_num < 12:
                                if self.config_1['Configuration'][f"{mic_spec_key}{mic_num}"] != f"{mic_num}":
                                    self.config_1['Configuration'][f"{mic_spec_key}{mic_num}"] = f"{mic_num}"
                                    changes += 1
                            else:
                                if self.config_1['Configuration'][f"{mic_spec_key}{mic_num}"] != f"{mic_num - 12}":
                                    self.config_1['Configuration'][f"{mic_spec_key}{mic_num}"] = f"{mic_num - 12}"
                                    changes += 1
                        else:
                            if self.config_1['Configuration'][f"{mic_spec_key}"] != str(self.exp_settings_dict['audio']['mics_config'][mic_spec_key]):
                                self.config_1['Configuration'][f"{mic_spec_key}"] = str(self.exp_settings_dict['audio']['mics_config'][mic_spec_key])
                                changes += 1
                else:
                    if (mic_num in self.exp_settings_dict['audio']['used_mics'] and f"{mic_spec_key}{mic_num}" not in self.config_1['Configuration'].keys()) or \
                            (mic_num in self.exp_settings_dict['audio']['used_mics'] and not math.isclose(self.exp_settings_dict['audio']['mics_config'][mic_spec_key], float(self.config_1['Configuration'][f"{mic_spec_key}{mic_num}"]))):
                        self.config_1['Configuration'][f"{mic_spec_key}{mic_num}"] = str(self.exp_settings_dict['audio']['mics_config'][mic_spec_key])
                        changes += 1

        for mic_num in range(self.exp_settings_dict['audio']['total_mic_number']):
            for monitor_key in self.exp_settings_dict['audio']['monitor'].keys():
                if (mic_num in self.exp_settings_dict['audio']['used_mics'] and f"{monitor_key}{mic_num}" not in self.config_1['Monitor'].keys()) or \
                        (mic_num in self.exp_settings_dict['audio']['used_mics'] and not math.isclose(self.exp_settings_dict['audio']['monitor'][monitor_key], float(self.config_1['Monitor'][f"{monitor_key}{mic_num}"]))):
                    self.config_1['Monitor'][f"{monitor_key}{mic_num}"] = str(self.exp_settings_dict['audio']['monitor'][monitor_key])
                    changes += 1

        for mic_num in range(self.exp_settings_dict['audio']['total_mic_number']):
            for call_key in self.exp_settings_dict['audio']['call'].keys():
                if (mic_num in self.exp_settings_dict['audio']['used_mics'] and f"{call_key}{mic_num}" not in self.config_1['Call'].keys()) or \
                        (mic_num in self.exp_settings_dict['audio']['used_mics'] and not math.isclose(self.exp_settings_dict['audio']['call'][call_key], float(self.config_1['Call'][f"{call_key}{mic_num}"]))):
                    self.config_1['Call'][f"{call_key}{mic_num}"] = str(self.exp_settings_dict['audio']['call'][call_key])
                    changes += 1

        if changes > 0:
            self.message_output(f"{changes} lines changed in the avisoft_config_file!")
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '_config/avisoft_config.ini'), 'w') as configfile:
                self.config_1.write(configfile, space_around_delimiters=False)

        shutil.copy(src=os.path.join(os.path.dirname(os.path.abspath(__file__)), '_config/avisoft_config.ini'),
                    dst=f"{self.exp_settings_dict['avisoft_basedirectory']}Configurations{os.sep}RECORDER_USGH{os.sep}avisoft_config.ini")

        smart_wait(app_context_bool=self.app_context_bool, seconds=2)

    def conduct_behavioral_recording(self) -> None:
        """
        Description
        ----------
        This method checks whether the system is ready for recording and if so,
        conducts a recording with the designated parameters and moves the recorded
        files to the network drive (see below for details).

        NB: the data cannot be acquired until Motif, Avisoft USGH recorder, and CoolTerm
        have been installed and/or configured for recording.
        ----------

        Parameters
        ----------
        ----------

        Returns
        ----------
        Directory structure w/ "audio", "sync" and "video" subdirectories.
        The "audio" subdirectory contains "original" and "original_mc" subdirectories,
        the "sync" subdirectory contains .txt files (serial monitor output),
        and the "video" subdirectory contains individual camera videos (.mp4) and metadata files,
        each in its own subdirectory.
        ----------
        """

        Messenger(message_output=self.message_output,
                  receivers=self.email_receivers,
                  credentials_file=f"{self.exp_settings_dict['credentials_directory']}{os.sep}email_config_record.ini",
                  exp_settings_dict=self.exp_settings_dict).send_message(subject="Audio PC in 165B is busy, do NOT attempt to remote in!",
                                                                         message=f"Experiment in progress, started at "
                                                                                 f"{datetime.datetime.now().hour:02d}:{datetime.datetime.now().minute:02d}.{datetime.datetime.now().second:02d} "
                                                                                 f"and run by @{self.exp_settings_dict['video']['metadata']['experimenter']}. "
                                                                                 f"You will be notified upon completion. \n \n ***This is an automatic e-mail, please do NOT respond.***")
        # reconnect to ethernet if it is off
        self.check_ethernet_connection()

        # start capturing sync LEDS
        if not os.path.isfile(f"{self.exp_settings_dict['coolterm_basedirectory']}{os.sep}Connection_settings{os.sep}coolterm_config.stc"):
            shutil.copy(src=os.path.join(os.path.dirname(os.path.abspath(__file__)), '_config/coolterm_config.ini'),
                        dst=f"{self.exp_settings_dict['coolterm_basedirectory']}{os.sep}Connection_settings{os.sep}coolterm_config.stc")
        subprocess.Popen(args=f'''cmd /c "start /MIN {self.exp_settings_dict['coolterm_basedirectory']}{os.sep}Connection_settings{os.sep}coolterm_config.stc"''',
                         stdout=subprocess.PIPE)

        self.check_camera_vitals(camera_fr=self.exp_settings_dict['video']['general']['recording_frame_rate'])

        # modify audio config file
        if self.exp_settings_dict['conduct_audio_recording']:
            self.modify_audio_file()

        start_hour_min_sec, total_dir_name_linux, total_dir_name_windows = self.get_custom_dir_names(now=self.api.call('schedule')['now'])

        # start recording audio
        if self.exp_settings_dict['conduct_audio_recording']:
            self.message_output(f"Audio recording in progress since {start_hour_min_sec}, it will last {round(self.exp_settings_dict['video_session_duration'] + .36, 2)} minute(s). Please be patient.")

            avisoft_recorder_program_name = self.exp_settings_dict['avisoft_recorder_program_name']

            cpu_priority = self.exp_settings_dict['audio']['cpu_priority']

            affinity_arg = ""
            if self.exp_settings_dict['audio']['cpu_affinity']:
                cpu_affinity_mask = self.get_cpu_affinity_mask()
                affinity_arg = f" /affinity {cpu_affinity_mask}"

            # run command to start Avisoft Recorder and keep executing the rest of the script
            if os.path.exists(f"{self.exp_settings_dict['avisoft_basedirectory']}{os.sep}Configurations{os.sep}RECORDER_USGH{os.sep}avisoft_config.ini"):

                # run Avisoft as Administrator
                run_avisoft_command = f"Start-Process -FilePath '{avisoft_recorder_program_name}' -ArgumentList '/CFG=avisoft_config.ini', '/AUT' -Verb RunAs"

                # add priority/affinity only if they exist
                if run_avisoft_command or (affinity_arg and affinity_arg.strip()):
                    run_avisoft_command += "; Start-Sleep 2; $proc = Get-Process 'rec_usgh' -ErrorAction SilentlyContinue; if ($proc) {"

                    if cpu_priority:
                        run_avisoft_command += f" $proc.PriorityClass = '{cpu_priority}';"

                    if affinity_arg and affinity_arg.strip():
                        run_avisoft_command += f" $proc.ProcessorAffinity = {affinity_arg};"

                    run_avisoft_command += " }"

                subprocess.Popen(args=f'''powershell -Command "{run_avisoft_command}"''', stdout=subprocess.PIPE, cwd=self.exp_settings_dict['avisoft_recorder_exe'])

                # pause for N seconds
                smart_wait(app_context_bool=self.app_context_bool, seconds=10)

                # check if Avisoft Recorder is running and exit GUI if not
                is_running = False
                is_frozen = False
                cmd_result = subprocess.run(args=f'''tasklist /FI "IMAGENAME eq {avisoft_recorder_program_name}"''',
                                            capture_output=True, text=True, check=False, shell=True)
                if cmd_result.returncode == 0 and avisoft_recorder_program_name.lower() in cmd_result.stdout.lower():
                    # check if Avisoft Recorder is frozen and exit GUI if so
                    cmd_result_frozen = subprocess.run(args=f'''tasklist /NH /FI "IMAGENAME eq {avisoft_recorder_program_name}" /FI "STATUS eq Not Responding''',
                                                       capture_output=True, text=True, check=False, shell=True)
                    if not (cmd_result_frozen.returncode == 0 and avisoft_recorder_program_name.lower() in cmd_result_frozen.stdout.lower()):
                        is_running = True
                    else:
                        is_frozen = True
                if not is_running:
                    subprocess.Popen(f'''cmd /c "taskkill /IM CoolTerm.exe /T /F 1>nul 2>&1"''').wait()
                    if is_frozen:
                        subprocess.Popen(f'''cmd /c "taskkill /IM {avisoft_recorder_program_name} /T /F 1>nul 2>&1"''').wait()
                    print("Aborted experiment as Avisoft Recorder was not running or was frozen.")
                    sys.exit(1)

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

        self.message_output(f"Video recording in progress since {datetime.datetime.now().hour:02d}:{datetime.datetime.now().minute:02d}.{datetime.datetime.now().second:02d}, it will last {round(self.exp_settings_dict['video_session_duration'], 2)} minute(s). Please be patient.")

        if self.exp_settings_dict['disable_ethernet']:
            subprocess.Popen(args=f'''cmd /c netsh interface set interface "{self.exp_settings_dict['ethernet_network']}" disable''',
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.STDOUT).wait()
            self.message_output(f"Ethernet DISCONNECTED at {datetime.datetime.now().hour:02d}:{datetime.datetime.now().minute:02d}.{datetime.datetime.now().second:02d}.")

        # wait until cameras have finished recording
        # pause for N extra seconds so audio is done, too
        smart_wait(app_context_bool=self.app_context_bool, seconds=25 + (self.exp_settings_dict['video_session_duration'] * 60))

        self.message_output(f"Recording fully completed at {datetime.datetime.now().hour:02d}:{datetime.datetime.now().minute:02d}.{datetime.datetime.now().second:02d}.")

        # close serial monitor sync LED capture
        subprocess.Popen(f'''cmd /c "taskkill /IM CoolTerm.exe /T /F 1>nul 2>&1"''').wait()

        if self.exp_settings_dict['disable_ethernet']:
            subprocess.Popen(args=f'''cmd /c netsh interface set interface "{self.exp_settings_dict['ethernet_network']}" enable''',
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.STDOUT).wait()
            self.message_output(f"Ethernet RECONNECTED at {datetime.datetime.now().hour:02d}:{datetime.datetime.now().minute:02d}.{datetime.datetime.now().second:02d}.")
            smart_wait(app_context_bool=self.app_context_bool, seconds=20)

        # remount CUP drives if necessary
        self.remount_cup_drives_on_windows()

        pathlib.Path(f"{total_dir_name_windows[0]}{os.sep}video").mkdir(parents=True, exist_ok=True)
        pathlib.Path(f"{total_dir_name_windows[0]}{os.sep}sync").mkdir(parents=True, exist_ok=True)
        if self.exp_settings_dict['conduct_audio_recording']:
            if self.exp_settings_dict['audio']['general']['total'] == 0:
                pathlib.Path(f"{total_dir_name_windows[0]}{os.sep}audio{os.sep}original").mkdir(parents=True, exist_ok=True)
            else:
                pathlib.Path(f"{total_dir_name_windows[0]}{os.sep}audio{os.sep}original").mkdir(parents=True, exist_ok=True)
                pathlib.Path(f"{total_dir_name_windows[0]}{os.sep}audio{os.sep}original_mc").mkdir(parents=True, exist_ok=True)

        self.message_output(f"Transferring audio/video files started at: {datetime.datetime.now().hour:02d}:{datetime.datetime.now().minute:02d}.{datetime.datetime.now().second:02d}")
        smart_wait(app_context_bool=self.app_context_bool, seconds=2)

        # move video file(s) to primary file server
        if len(self.exp_settings_dict['video']['general']['expected_cameras']) == 1:
            copy_video_command = f"camera/{self.exp_settings_dict['video']['general']['expected_cameras'][0]}/recordings/copy_all"
        else:
            copy_video_command = 'recordings/copy_all'

        self.api.call(copy_video_command,
                      delete_after=self.exp_settings_dict['video']['general']['delete_post_copy'],
                      location=f"{total_dir_name_linux[0]}/video")

        # move audio file(s) to primary file server
        if self.exp_settings_dict['conduct_audio_recording']:
            audio_copy_subprocesses = []
            if self.exp_settings_dict['audio']['general']['total'] == 0:
                for mic_idx in self.exp_settings_dict['audio']['used_mics']:
                    last_modified_audio_file = max(glob.glob(f"{self.exp_settings_dict['avisoft_basedirectory']}{os.sep}ch{mic_idx + 1}{os.sep}*.wav"),
                                                   key=os.path.getctime)
                    single_audio_copy_subp = subprocess.Popen(args=f'''cmd /c move "{last_modified_audio_file.split(os.sep)[-1]}" "{total_dir_name_windows[0]}{os.sep}audio{os.sep}original{os.sep}ch{mic_idx + 1}_{last_modified_audio_file.split(os.sep)[-1]}"''',
                                                              cwd=f"{self.exp_settings_dict['avisoft_basedirectory']}{os.sep}ch{mic_idx + 1}",
                                                              stdout=subprocess.DEVNULL,
                                                              stderr=subprocess.STDOUT,
                                                              shell=False)
                    audio_copy_subprocesses.append(single_audio_copy_subp)
            else:
                relevant_file_count = max(1, int(round((self.exp_settings_dict['video_session_duration']+.36) / 5.09)))
                device_id = ['m', 's']
                for mic_pos_idx, mic_idx in enumerate([0, 12]):
                    audio_file_list = sorted(glob.glob(f"{self.exp_settings_dict['avisoft_basedirectory']}{os.sep}ch{mic_idx + 1}{os.sep}*.wav"), key=os.path.getctime, reverse=True)[:relevant_file_count]
                    for aud_file in audio_file_list:
                        multi_audio_copy_subp = subprocess.Popen(args=f'''cmd /c move "{aud_file.split(os.sep)[-1]}" "{total_dir_name_windows[0]}{os.sep}audio{os.sep}original_mc{os.sep}{device_id[mic_pos_idx]}_{aud_file.split(os.sep)[-1]}"''',
                                                                 cwd=f"{self.exp_settings_dict['avisoft_basedirectory']}{os.sep}ch{mic_idx + 1}",
                                                                 stdout=subprocess.DEVNULL,
                                                                 stderr=subprocess.STDOUT,
                                                                 shell=False)
                        audio_copy_subprocesses.append(multi_audio_copy_subp)

            while True:
                status_poll = [query_subp.poll() for query_subp in audio_copy_subprocesses]
                if any(elem is None for elem in status_poll):
                    smart_wait(app_context_bool=self.app_context_bool, seconds=1)
                else:
                    break

        # move last modified sync file to primary file server
        last_modified_sync_file = max(glob.glob(f"{self.exp_settings_dict['coolterm_basedirectory']}{os.sep}Data{os.sep}*.txt"), key=os.path.getctime)
        shutil.move(src=last_modified_sync_file,
                    dst=f"{total_dir_name_windows[0]}{os.sep}sync{os.sep}{last_modified_sync_file.split(os.sep)[-1]}")

        # ensure the video is done copying before proceeding
        while any(self.api.is_copying(_sn) for _sn in self.camera_serial_num):
            smart_wait(app_context_bool=self.app_context_bool, seconds=1)

        # copy the audio, sync and video directories to the backup network drive(s)
        if len(total_dir_name_windows) > 1:
            for win_dir_idx, win_dir in enumerate(total_dir_name_windows[1:]):
                subprocess.Popen(args=f'''cmd /c robocopy "{total_dir_name_windows[0]}" "{win_dir}" /MIR''',
                                 cwd=f"{total_dir_name_windows[0]}",
                                 stdout=subprocess.DEVNULL,
                                 stderr=subprocess.STDOUT,
                                 shell=False)

        # check number of dropouts in audio recordings
        if self.exp_settings_dict['conduct_audio_recording'] and self.exp_settings_dict['audio']['devices']['usghflags'] != 1574:

            audio_triggerbox_sync_info_dict = {device: {'start_first_recorded_frame': 0, 'end_last_recorded_frame': 0, 'largest_break_duration': 0,
                                                        'duration_samples': 0, 'duration_seconds': 0, 'audio_tracking_diff_seconds': 0, 'num_dropouts': 0} for device in ['m', 's']}

            for log_ch, log_device in zip(['ch1', 'ch13'], ['m', 's']):
                dropout_count = count_last_recording_dropouts(log_file_path=self.exp_settings_dict['avisoft_basedirectory'],
                                                              log_file_ch=log_ch)

                audio_triggerbox_sync_info_dict[log_device]['num_dropouts'] = dropout_count

                if dropout_count is None:
                    self.message_output(f"Could not determine the number of dropouts for {log_device} device, please check the log file.")
                else:
                    if dropout_count > 0:
                        self.message_output(f"[***Important!***] Number of dropouts registered on {log_device} device: {dropout_count}.")
                    else:
                        self.message_output(f"Number of dropouts registered on {log_device} device: {dropout_count}.")

            with open(f"{total_dir_name_windows[0]}{os.sep}audio{os.sep}audio_triggerbox_sync_info.json", 'w') as audio_dict_outfile:
                json.dump(audio_triggerbox_sync_info_dict, audio_dict_outfile, indent=4)

        self.message_output(f"Transferring audio/video files finished at: {datetime.datetime.now().hour:02d}:{datetime.datetime.now().minute:02d}.{datetime.datetime.now().second:02d}")

        Messenger(message_output=self.message_output,
                  receivers=self.email_receivers,
                  no_receivers_notification=False,
                  credentials_file=f"{self.exp_settings_dict['credentials_directory']}{os.sep}email_config_record.ini",
                  exp_settings_dict=self.exp_settings_dict).send_message(subject="Audio PC in 165B is available again, recording has been completed.",
                                                                         message=f"Thank you for your patience, recording by @{self.exp_settings_dict['video']['metadata']['experimenter']} was completed at "
                                                                                 f"{datetime.datetime.now().hour:02d}:{datetime.datetime.now().minute:02d}.{datetime.datetime.now().second:02d}. "
                                                                                 f"You will be notified about further experiments "
                                                                                 f"should they occur. \n \n ***This is an automatic e-mail, please do NOT respond.***")

@click.command(name="conduct-calibration")
@click.option('--set', 'overrides', multiple=True, help='Override a setting, e.g., --set calibration_duration=10')
def conduct_calibration_cli( overrides):
    """
    Description
    ----------
    A command-line tool to perform a tracking camera calibration.
    ----------

    Parameters
    ----------
    ----------

    Returns
    ----------
    ----------
    """

    with open((pathlib.Path(__file__).parent / '_config/behavioral_experiments_settings.toml'), 'r') as f:
        exp_settings_dict = toml.load(f)

    if len(overrides) > 0:
        exp_settings_dict = override_toml_values(overrides=overrides, exp_settings_dict=exp_settings_dict)

    ExperimentController(exp_settings_dict=exp_settings_dict).conduct_tracking_calibration()

@click.command(name="conduct-recording")
@click.option('--set', 'overrides', multiple=True, help='Override a setting, e.g., --set video.metadata.notes="Test run"')
def conduct_recording_cli(overrides):
    """
    Description
    ----------
    A command-line tool to conduct a behavioral recording session.
    ----------

    Parameters
    ----------
    ----------

    Returns
    ----------
    ----------
    """

    with open((pathlib.Path(__file__).parent / '_config/behavioral_experiments_settings.toml'), 'r') as f:
        exp_settings_dict = toml.load(f)

    if len(overrides) > 0:
        exp_settings_dict = override_toml_values(overrides=overrides, exp_settings_dict=exp_settings_dict)

    ExperimentController(exp_settings_dict=exp_settings_dict).conduct_behavioral_recording()
