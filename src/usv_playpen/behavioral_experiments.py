"""
@author: bartulem
Runs experiments with Avisoft/CoolTerm/Loopbio software.
"""

from __future__ import annotations

import concurrent.futures
import configparser
import datetime
import json
import math
import os
import pathlib
import shutil
import subprocess
import sys
import webbrowser
from collections.abc import Callable
from importlib import metadata
from pathlib import Path

import click
import motifapi
import paramiko
import toml
import yaml

from .cli_utils import override_toml_values
from .os_utils import newest_match_or_raise, wait_for_subprocesses
from .send_email import Messenger
from .time_utils import is_gui_context, smart_wait
from .yaml_utils import SmartDumper


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
        with open(pathlib.Path(log_file_path) / log_file_ch / f"{log_file_ch}.log", 'r') as log_txt_file:
            content = log_txt_file.read()
    except FileNotFoundError:
        return None

    recordings = content.split(str(pathlib.Path(log_file_path) / log_file_ch) + os.sep)

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
                 metadata_settings: dict = None,
                 message_output: Callable | None = None) -> None:

        """
        Initializes the ExperimentController class.

        Parameters
        ----------
        exp_settings_dict (dict)
            Experiment settings; defaults to None.
        email_receivers (list)
            Email receivers; defaults to None.
        metadata_settings (dict)
            Metadata settings; defaults to None.
        message_output (function)
            Defines output messages; defaults to None.

        Returns
        -------
        -------
        """

        self.api = None
        self.camera_serial_num = None
        self.config_1 = None

        self.email_receivers = email_receivers
        self.exp_settings_dict = exp_settings_dict
        self.metadata_settings = metadata_settings if metadata_settings is not None else {}
        self.message_output = message_output if message_output is not None else print

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

        ethernet_name = self.exp_settings_dict['ethernet_network']

        status_check_command = [
            "powershell", "-Command",
            f'(Get-NetAdapter -Name "{ethernet_name}").Status'
        ]

        status_result = subprocess.run(
            status_check_command,
            capture_output=True,
            text=True,
            check=True,
            encoding='utf-8'
        )
        adapter_status = status_result.stdout.strip()

        if adapter_status.lower() != 'up':
            self.message_output(f"Adapter '{ethernet_name}' is not Up. Attempting to enable...")

            enable_command = [
                "netsh", "interface", "set", "interface",
                ethernet_name,
                "enable"
            ]

            subprocess.run(
                enable_command,
                capture_output=True,
                text=True,
                check=True,
                encoding='utf-8'
            )

            max_wait_seconds = 20
            poll_interval_seconds = 5
            num_attempts = max(1, max_wait_seconds // poll_interval_seconds)

            for attempt in range(num_attempts):
                status_result = subprocess.run(
                    status_check_command,
                    capture_output=True,
                    text=True,
                    check=True,
                    encoding='utf-8'
                )
                current_status = status_result.stdout.strip()

                if current_status.lower() == 'up':
                    self.message_output(f"Success! Adapter '{ethernet_name}' is now Up.")
                    smart_wait(app_context_bool=self.app_context_bool, seconds=10)
                    break

                if attempt < num_attempts - 1:
                    smart_wait(app_context_bool=self.app_context_bool, seconds=poll_interval_seconds)
            else:
                self.message_output(f"Timeout: Adapter '{ethernet_name}' did not come Up after {max_wait_seconds} seconds.")
                sys.exit(1)

    def verify_avisoft_is_recording(self,
                                    warmup_s: int = 3,
                                    poll_interval_s: int = 3,
                                    max_wait_s: int = 12) -> bool:
        """
        Description
        -----------
        Verifies that Avisoft Recorder is actually producing audio to disk
        shortly after launch. Instead of relying on Windows' 'STATUS eq Not
        Responding' tasklist heuristic (which tracks UI-thread responsiveness
        and misses audio-thread / USB-device freezes, as well as cases where a
        modal dialog has eaten the UI), this watches the per-channel .wav files
        under 'avisoft_basedirectory/chN/' and declares success only when every
        required channel directory has shown real byte growth.

        Behavior:
        1. Waits 'warmup_s' seconds so Avisoft has a chance to create its first
           .wav in each chN directory.
        2. Snapshots the newest .wav file and its size in every watched channel
           directory as the baseline (None / 0 bytes if Avisoft has not created
           one yet).
        3. Polls every 'poll_interval_s' seconds up to a total of 'max_wait_s'
           seconds, marking a channel directory 'verified' as soon as either
           (a) a new .wav file appears relative to the baseline, or
           (b) the newest .wav grew in size relative to the baseline.
        4. Returns True only if all channel directories got verified inside the
           polling window.

        Parameters
        ----------
        warmup_s (int)
            Seconds to wait after Avisoft launch before the first size snapshot.
        poll_interval_s (int)
            Seconds between successive polls of the .wav files.
        max_wait_s (int)
            Total budget, in seconds, for the verification window after warmup.

        Returns
        -------
        is_recording (bool)
            True if every watched chN directory has produced byte-level progress
            within the window; False otherwise.
        """

        base = pathlib.Path(self.exp_settings_dict['avisoft_basedirectory'])
        if self.exp_settings_dict['audio']['general']['total'] == 0:
            channel_dirs = [base / f"ch{mic_idx + 1}"
                            for mic_idx in self.exp_settings_dict['audio']['used_mics']]
        else:
            channel_dirs = [base / f"ch{mic_idx + 1}" for mic_idx in [0, 12]]

        if not channel_dirs:
            self.message_output("[**Avisoft health check**] No channel directories resolved to watch — skipping verification.")
            return True

        smart_wait(app_context_bool=self.app_context_bool, seconds=warmup_s)

        def newest_wav(ch_dir: pathlib.Path):
            """
            Returns the (path, size) of the newest .wav file in ch_dir, or
            (None, 0) if the directory is missing or contains no .wav files.
            Stat errors (file vanished mid-glob, permission denied) are
            treated the same as 'no file'.
            """
            if not ch_dir.is_dir():
                return None, 0
            try:
                wavs = list(ch_dir.glob('*.wav'))
            except OSError:
                return None, 0
            if not wavs:
                return None, 0
            try:
                newest = max(wavs, key=lambda p: p.stat().st_ctime)
                return newest, newest.stat().st_size
            except (OSError, ValueError):
                return None, 0

        baselines = {ch_dir: newest_wav(ch_dir) for ch_dir in channel_dirs}

        verified = set()
        elapsed = 0
        while elapsed < max_wait_s and len(verified) < len(channel_dirs):
            smart_wait(app_context_bool=self.app_context_bool, seconds=poll_interval_s)
            elapsed += poll_interval_s

            for ch_dir in channel_dirs:
                if ch_dir in verified:
                    continue
                baseline_file, baseline_size = baselines[ch_dir]
                curr_file, curr_size = newest_wav(ch_dir)
                if curr_file is None:
                    continue
                if curr_file != baseline_file:
                    verified.add(ch_dir)
                    self.message_output(f"[**Avisoft health check**] {ch_dir.name}: new recording file detected ({curr_file.name}, {curr_size} bytes).")
                elif curr_size > baseline_size:
                    verified.add(ch_dir)
                    self.message_output(f"[**Avisoft health check**] {ch_dir.name}: recording file is growing ({curr_file.name}, {curr_size} bytes).")

        if len(verified) < len(channel_dirs):
            unverified = [ch_dir.name for ch_dir in channel_dirs if ch_dir not in verified]
            self.message_output(f"[**Avisoft health check**] FAILED: no byte growth detected in channel dir(s) {unverified} within {warmup_s + max_wait_s} s of Avisoft launch.")
            return False

        self.message_output(f"[**Avisoft health check**] All {len(channel_dirs)} watched channel dir(s) are producing audio.")
        return True

    def purge_cup_connections_on_windows(self) -> None:
        """
        Description
        -----------
        Tears down every existing SMB connection to the CUP file server on the
        local Windows machine. This is used both as a standalone pre-recording
        cleanup step (to guarantee a clean mount state before a session starts)
        and as the first phase of the post-recording remount routine.

        After Ethernet disconnect/reconnect cycles, or when the machine has
        previously been used by a different university account, Windows SMB
        sessions to \\cup can become stale. The mapped drive letters (e.g., F:,
        M:) may remain in the registry as zombies with expired credentials, and
        cached Kerberos tickets may still hold credentials from a prior user.
        Mounting on top of those stale entries fails with System error 1219
        ("Multiple connections to a server or shared resource by the same user,
        using more than one user name, are not allowed.").

        This method:
        1. Deletes every known drive-letter and UNC mapping to \\cup\\falkner
           and \\cup\\murthy (regardless of which the current user needs).
        2. Deletes the server-level session to \\cup itself.
        3. Purges cached Kerberos tickets via 'klist purge'.
        4. Sleeps briefly so Windows fully tears the SMB sessions down.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        all_possible_drives = {"F:": r"\\cup\falkner", "M:": r"\\cup\murthy"}

        # Always purge ALL connections to \\cup, regardless of which drives the
        # current user needs. This prevents System error 1219 when switching
        # between users with different credentials on the same machine.
        # Use shell=False and argv lists so drive-letter / UNC values are never
        # interpolated into a cmd.exe string.
        for letter, unc in all_possible_drives.items():
            subprocess.run(["net", "use", letter, "/delete", "/y"], shell=False,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(["net", "use", unc, "/delete", "/y"], shell=False,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Kill the server-level session itself, not just individual share mappings
        subprocess.run(["net", "use", r"\\cup", "/delete", "/y"], shell=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Clear any cached Kerberos tickets that may hold stale auth
        subprocess.run(["klist", "purge"], shell=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Brief pause so Windows fully tears down the SMB sessions
        smart_wait(app_context_bool=self.app_context_bool, seconds=3)

    def remount_cup_drives_on_windows(self) -> None:
        """
        Description
        -----------
        Checks if the specified Ethernet adapter is connected. If not, it re-enables
        the connection. It then verifies the status of required network drives (CUP)
        and remounts them with fresh credentials if they are stale or missing.

        After Ethernet disconnect/reconnect cycles, Windows SMB sessions become stale.
        The mapped drive letter (e.g., F:) may remain in the registry as a zombie with
        expired credentials. This method:
        1. Verifies Ethernet connectivity.
        2. Waits for the SMB/network subsystem to stabilize.
        3. Purges ALL existing connections to \\cup to prevent System error 1219
           when switching between users with different credentials.
        4. Probes each needed drive with a short timeout to detect stale sessions.
        5. Remounts with fresh credentials.
        6. Verifies the new mount is accessible, with retries.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        cup_username, cup_password = self.get_cup_mount_params()

        all_possible_drives = {"F:": r"\\cup\falkner", "M:": r"\\cup\murthy"}
        needed_letters = {path.split("\\")[0] for path in self.exp_settings_dict['recording_files_destination_win']}
        drives_to_mount = {letter: unc for letter, unc in all_possible_drives.items()
                           if letter in needed_letters}

        self.check_ethernet_connection()

        # Wait for SMB/network services to stabilize after Ethernet reconnection.
        # The adapter being "Up" doesn't mean SMB sessions can be established yet.
        self.message_output("[**Local mount check**] Waiting for network services to stabilize...")
        smart_wait(app_context_bool=self.app_context_bool, seconds=5)

        # Purge every existing connection to \\cup before mounting so we do not
        # collide with stale sessions from a previous user or a prior Ethernet
        # cycle (System error 1219 otherwise).
        self.purge_cup_connections_on_windows()

        # Mount only the drives the current user needs
        for drive_letter, unc_path in drives_to_mount.items():

            max_retries = 3
            mounted_successfully = False

            for attempt in range(1, max_retries + 1):
                self.message_output(f"[**Local mount check**] Mounting {drive_letter} -> {unc_path} (attempt {attempt}/{max_retries})...")

                # argv list keeps the password (user-supplied) out of a cmd.exe
                # string — shell metacharacters in the password cannot break the
                # command, and it never appears in a process listing as a single
                # flat command line.
                mount_result = subprocess.run(
                    [
                        "net", "use", drive_letter, unc_path,
                        cup_password,
                        f"/user:{cup_username}@princeton.edu",
                        "/persistent:yes",
                    ],
                    shell=False,
                    capture_output=True,
                    text=True
                )

                if mount_result.returncode != 0:
                    self.message_output(f"[**Local mount check**] 'net use' returned error: {mount_result.stderr.strip()}")

                    # If "already in use" or "multiple connections", try removing again
                    if "already" in mount_result.stderr.lower() or "multiple" in mount_result.stderr.lower():
                        subprocess.run(["net", "use", drive_letter, "/delete", "/y"], shell=False,
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        subprocess.run(["net", "use", unc_path, "/delete", "/y"], shell=False,
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        subprocess.run(["net", "use", r"\\cup", "/delete", "/y"], shell=False,
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        smart_wait(app_context_bool=self.app_context_bool, seconds=3)
                        continue

                    if attempt < max_retries:
                        smart_wait(app_context_bool=self.app_context_bool, seconds=5)
                        continue

                # Verify the mount is actually accessible (not just mapped)
                smart_wait(app_context_bool=self.app_context_bool, seconds=2)

                try:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(lambda dl: pathlib.Path(f"{dl}\\").is_dir(), drive_letter)
                        is_accessible = future.result(timeout=10)

                    if is_accessible:
                        self.message_output(f"[**Local mount check**] '{drive_letter}' has been successfully mounted and verified.")
                        mounted_successfully = True
                        break
                    else:
                        self.message_output(f"[**Local mount check**] '{drive_letter}' was mapped but path verification failed.")
                except concurrent.futures.TimeoutError:
                    self.message_output(f"[**Local mount check**] '{drive_letter}' verification timed out.")

                if attempt < max_retries:
                    smart_wait(app_context_bool=self.app_context_bool, seconds=5)

            if not mounted_successfully:
                self.message_output(f"[**CRITICAL ERROR**] Failed to mount {drive_letter} after {max_retries} attempts. "
                                    f"Recording files may not be saved to the network drive.")

    def check_remote_mount(self,
                           hostname: str,
                           port: int,
                           username: str,
                           password: str,
                           mount_path: str) -> bool:
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
            self.message_output(f"[**Remote mount check**] On {hostname}, an unexpected error occurred: {e}")
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

        if not (pathlib.Path(self.exp_settings_dict['credentials_directory']) / 'cup_config.ini').exists():
            print("Cup config file not found. Try again!")
            sys.exit(1)
        else:
            config.read(pathlib.Path(self.exp_settings_dict['credentials_directory']) / 'cup_config.ini')
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

        if not (pathlib.Path(self.exp_settings_dict['credentials_directory']) / 'motif_config.ini').exists():
            print("Motif config file not found. Try again!")
            sys.exit(1)
        else:
            config.read(pathlib.Path(self.exp_settings_dict['credentials_directory']) / 'motif_config.ini')
            return (config['motif']['master_ip_address'], config['motif']['second_ip_address'],
                    config['motif']['ssh_port'], config['motif']['ssh_username'],
                    config['motif']['ssh_password'], config['motif']['api'])

    def get_custom_dir_names(self, now: float) -> tuple:
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
        start_hour_min_sec (str), total_dir_name_linux (str), total_dir_name_windows (str), sub_dir_name (str)
            Start time of recording, directory location in Linux and Window coordinates, and name of session directory
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

        return start_hour_min_sec, total_dir_name_linux, total_dir_name_windows, sub_dir_name

    def check_camera_vitals(self, camera_fr: int | float) -> None:
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
                    print(f"Mount point {lin_directory} on {ip_address} is not valid. Please fix the issue and try again.")
                    sys.exit(1)

        try:
            api = motifapi.MotifApi(ip_address, api_key)
        except motifapi.api.MotifError:
            print('Motif not running or reachable. Check hardware and connections.')
            sys.exit(1)
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

        self.message_output(f"Video calibration started at {datetime.datetime.now().hour:02d}:{datetime.datetime.now().minute:02d}:{datetime.datetime.now().second:02d}")

        self.check_camera_vitals(camera_fr=self.exp_settings_dict['video']['general']['calibration_frame_rate'])

        # calibrate tracking cameras
        self.api.call('recording/start',
                      filename='calibration',
                      duration=self.exp_settings_dict['calibration_duration'] * 60,
                      codec=self.exp_settings_dict['video']['general']['recording_codec'])
        smart_wait(app_context_bool=self.app_context_bool, seconds=(self.exp_settings_dict['calibration_duration'] * 60) + 5)

        self.message_output(f"Video calibration completed at {datetime.datetime.now().hour:02d}:{datetime.datetime.now().minute:02d}:{datetime.datetime.now().second:02d}")

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
        self.config_1.read(pathlib.Path(__file__).parent / '_config/avisoft_config.ini')

        if f"{str(self.exp_settings_dict['avisoft_basedirectory'])}{os.sep}" != self.config_1['Configuration']['basedirectory']:
            self.config_1['Configuration']['basedirectory'] = f"{self.exp_settings_dict['avisoft_basedirectory']}{os.sep}"
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
            if not math.isclose(float(self.config_1['MaxFileSize']['minutes']), 5.09):
                self.config_1['MaxFileSize']['minutes'] = str(5.09)
                changes += 1

        if self.config_1['Configuration']['configfilename'] != str(pathlib.Path(self.exp_settings_dict['avisoft_config_directory']) / 'avisoft_config.ini'):
            self.config_1['Configuration']['configfilename'] = str(pathlib.Path(self.exp_settings_dict['avisoft_config_directory']) / 'avisoft_config.ini')

        if self.config_1['Info']['configfilename'] != str(pathlib.Path(self.exp_settings_dict['avisoft_config_directory']) / 'avisoft_config.ini'):
            self.config_1['Info']['configfilename'] = str(pathlib.Path(self.exp_settings_dict['avisoft_config_directory']) / 'avisoft_config.ini')

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
                    if (mic_num in self.exp_settings_dict['audio']['used_mics'] and f"{mic_spec_key}{mic_num}" not in self.config_1['Configuration']) or \
                            (mic_num in self.exp_settings_dict['audio']['used_mics'] and str(self.exp_settings_dict['audio']['mics_config'][mic_spec_key]) != self.config_1['Configuration'][f"{mic_spec_key}{mic_num}"]):
                        if mic_spec_key == 'name':
                            if self.config_1['Configuration'][f"{mic_spec_key}{mic_num}"] != f"ch{mic_num + 1}":
                                self.config_1['Configuration'][f"{mic_spec_key}{mic_num}"] = f"ch{mic_num + 1}"
                                changes += 1
                        elif mic_spec_key in ('deviceid', 'id'):
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
                    if (mic_num in self.exp_settings_dict['audio']['used_mics'] and f"{mic_spec_key}{mic_num}" not in self.config_1['Configuration']) or \
                            (mic_num in self.exp_settings_dict['audio']['used_mics'] and not math.isclose(self.exp_settings_dict['audio']['mics_config'][mic_spec_key], float(self.config_1['Configuration'][f"{mic_spec_key}{mic_num}"]))):
                        self.config_1['Configuration'][f"{mic_spec_key}{mic_num}"] = str(self.exp_settings_dict['audio']['mics_config'][mic_spec_key])
                        changes += 1

        for mic_num in range(self.exp_settings_dict['audio']['total_mic_number']):
            for monitor_key in self.exp_settings_dict['audio']['monitor'].keys():
                if (mic_num in self.exp_settings_dict['audio']['used_mics'] and f"{monitor_key}{mic_num}" not in self.config_1['Monitor']) or \
                        (mic_num in self.exp_settings_dict['audio']['used_mics'] and not math.isclose(self.exp_settings_dict['audio']['monitor'][monitor_key], float(self.config_1['Monitor'][f"{monitor_key}{mic_num}"]))):
                    self.config_1['Monitor'][f"{monitor_key}{mic_num}"] = str(self.exp_settings_dict['audio']['monitor'][monitor_key])
                    changes += 1

        for mic_num in range(self.exp_settings_dict['audio']['total_mic_number']):
            for call_key in self.exp_settings_dict['audio']['call'].keys():
                if (mic_num in self.exp_settings_dict['audio']['used_mics'] and f"{call_key}{mic_num}" not in self.config_1['Call']) or \
                        (mic_num in self.exp_settings_dict['audio']['used_mics'] and not math.isclose(self.exp_settings_dict['audio']['call'][call_key], float(self.config_1['Call'][f"{call_key}{mic_num}"]))):
                    self.config_1['Call'][f"{call_key}{mic_num}"] = str(self.exp_settings_dict['audio']['call'][call_key])
                    changes += 1

        if changes > 0:
            self.message_output(f"{changes} lines changed in the avisoft_config_file!")
            with open(pathlib.Path(__file__).parent / '_config/avisoft_config.ini', 'w') as configfile:
                self.config_1.write(configfile, space_around_delimiters=False)

        shutil.copy(src=pathlib.Path(__file__).parent / '_config/avisoft_config.ini',
                    dst=pathlib.Path(self.exp_settings_dict['avisoft_config_directory']) / 'avisoft_config.ini')

        smart_wait(app_context_bool=self.app_context_bool, seconds=2)

    def conduct_behavioral_recording(self) -> dict:
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
        updated_metadata (dict)
            Updated metadata after recording.

        Directory structure w/ "audio", "sync" and "video" subdirectories.
        The "audio" subdirectory contains "original" and "original_mc" subdirectories,
        the "sync" subdirectory contains .txt files (serial monitor output),
        and the "video" subdirectory contains individual camera videos (.mp4) and metadata files,
        each in its own subdirectory.
        ----------
        """

        Messenger(message_output=self.message_output,
                  receivers=self.email_receivers,
                  credentials_file=pathlib.Path(self.exp_settings_dict['credentials_directory']) / 'email_config.ini',
                  exp_settings_dict=self.exp_settings_dict).send_message(subject="Audio PC in 165B is busy, do NOT attempt to remote in!",
                                                                         message=f"Experiment in progress, started at "
                                                                                 f"{datetime.datetime.now().hour:02d}:{datetime.datetime.now().minute:02d}:{datetime.datetime.now().second:02d} "
                                                                                 f"and run by @{self.exp_settings_dict['experimenter']}. "
                                                                                 f"You will be notified upon completion. \n \n ***This is an automatic e-mail, please do NOT respond.***")
        # reconnect to ethernet if it is off
        self.check_ethernet_connection()

        # Purge any stale \\cup sessions/credentials left over from previous
        # users or Ethernet cycles before recording starts. We do NOT remount
        # here: the drives are remounted only at the end of the recording via
        # remount_cup_drives_on_windows(), once transfer to the file server
        # actually needs to happen.
        self.purge_cup_connections_on_windows()

        # start capturing sync LEDS
        if not (pathlib.Path(self.exp_settings_dict['coolterm_basedirectory']) / 'Connection_settings' / 'coolterm_config.stc').is_file():
            shutil.copy(src=pathlib.Path(__file__).parent / '_config' / 'coolterm_config.stc',
                        dst=pathlib.Path(self.exp_settings_dict['coolterm_basedirectory']) / 'Connection_settings' / 'coolterm_config.stc')

        coolterm_config_path = str(pathlib.Path(self.exp_settings_dict['coolterm_basedirectory']) / 'Connection_settings' / 'coolterm_config.stc')

        subprocess.Popen(
            args=['powershell', '-Command', f"Start-Process -FilePath '{coolterm_config_path}' -WindowStyle Minimized"],
            stdout=subprocess.PIPE
        )

        self.check_camera_vitals(camera_fr=self.exp_settings_dict['video']['general']['recording_frame_rate'])

        # modify audio config file
        if self.exp_settings_dict['conduct_audio_recording']:
            self.modify_audio_file()

        start_hour_min_sec, total_dir_name_linux, total_dir_name_windows, session_id = self.get_custom_dir_names(now=self.api.call('schedule')['now'])

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
            if (pathlib.Path(self.exp_settings_dict['avisoft_config_directory']) / 'avisoft_config.ini').exists():

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

                subprocess.Popen(args=f'''powershell -Command "{run_avisoft_command}"''', stdout=subprocess.PIPE, cwd=self.exp_settings_dict['avisoft_recorder_exe_directory'])

                # Verify Avisoft is actually producing audio by watching the
                # per-channel .wav files grow on disk. This replaces the old
                # 'tasklist STATUS eq Not Responding' heuristic, which tracked
                # only UI-thread responsiveness and missed audio-thread / USB
                # freezes as well as modal-dialog hangs.
                if not self.verify_avisoft_is_recording():
                    subprocess.Popen(['powershell', '-Command', "Stop-Process -Name 'CoolTerm' -Force -ErrorAction SilentlyContinue"]).wait()
                    subprocess.Popen(['powershell', '-Command', f"Stop-Process -Name '{avisoft_recorder_program_name}' -Force -ErrorAction SilentlyContinue"]).wait()
                    self.message_output("Aborted experiment: Avisoft Recorder did not produce audio bytes after launch.")
                    print("Aborted experiment: Avisoft Recorder did not produce audio bytes after launch.")
                    sys.exit(1)

        # record video data
        if len(self.exp_settings_dict['video']['general']['expected_cameras']) == 1:
            self.message_output(f"You chose to conduct the recording with one camera only (camera serial num: {self.exp_settings_dict['video']['general']['expected_cameras'][0]}).")
            self.api.call(f"camera/{self.exp_settings_dict['video']['general']['expected_cameras'][0]}/recording/start",
                          duration=self.exp_settings_dict['video_session_duration'] * 60,
                          codec=self.exp_settings_dict['video']['general']['recording_codec'])
        else:
            self.api.call('recording/start',
                          duration=self.exp_settings_dict['video_session_duration'] * 60,
                          codec=self.exp_settings_dict['video']['general']['recording_codec'])

        self.message_output(f"Video recording in progress since {datetime.datetime.now().hour:02d}:{datetime.datetime.now().minute:02d}:{datetime.datetime.now().second:02d}, it will last {round(self.exp_settings_dict['video_session_duration'], 2)} minute(s). Please be patient.")

        if self.exp_settings_dict['disable_ethernet']:

            subprocess.Popen(
                args=['powershell', '-Command', f"Disable-NetAdapter -Name '{self.exp_settings_dict['ethernet_network']}' -Confirm:$false"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT
            ).wait()

            self.message_output(f"Ethernet DISCONNECTED at {datetime.datetime.now().hour:02d}:{datetime.datetime.now().minute:02d}:{datetime.datetime.now().second:02d}.")

        # wait until cameras have finished recording
        # pause for N extra seconds so audio is done, too
        smart_wait(app_context_bool=self.app_context_bool, seconds=25 + (self.exp_settings_dict['video_session_duration'] * 60))

        self.message_output(f"Recording fully completed at {datetime.datetime.now().hour:02d}:{datetime.datetime.now().minute:02d}:{datetime.datetime.now().second:02d}.")

        # close serial monitor sync LED capture
        subprocess.Popen(['powershell', '-Command', "Stop-Process -Name 'CoolTerm' -Force -ErrorAction SilentlyContinue"]).wait()

        if self.exp_settings_dict['disable_ethernet']:

            subprocess.Popen(
                args=['powershell', '-Command', f"Enable-NetAdapter -Name '{self.exp_settings_dict['ethernet_network']}' -Confirm:$false"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT
            ).wait()

            self.message_output(f"Ethernet RECONNECTED at {datetime.datetime.now().hour:02d}:{datetime.datetime.now().minute:02d}:{datetime.datetime.now().second:02d}.")

            smart_wait(app_context_bool=self.app_context_bool, seconds=20)

        # remount CUP drives if necessary
        self.remount_cup_drives_on_windows()

        (pathlib.Path(total_dir_name_windows[0]) / 'video').mkdir(parents=True, exist_ok=True)
        (pathlib.Path(total_dir_name_windows[0]) / 'sync').mkdir(parents=True, exist_ok=True)
        if self.exp_settings_dict['conduct_audio_recording']:
            if self.exp_settings_dict['audio']['general']['total'] == 0:
                (pathlib.Path(total_dir_name_windows[0]) / 'audio' / 'original').mkdir(parents=True, exist_ok=True)
            else:
                (pathlib.Path(total_dir_name_windows[0]) / 'audio' / 'original').mkdir(parents=True, exist_ok=True)
                (pathlib.Path(total_dir_name_windows[0]) / 'audio' / 'original_mc').mkdir(parents=True, exist_ok=True)

        self.message_output(f"Transferring audio/video files started at: {datetime.datetime.now().hour:02d}:{datetime.datetime.now().minute:02d}:{datetime.datetime.now().second:02d}")
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
                    ch_dir = pathlib.Path(self.exp_settings_dict['avisoft_basedirectory']) / f"ch{mic_idx + 1}"
                    try:
                        last_modified_audio_file = newest_match_or_raise(
                            root=ch_dir,
                            pattern='*.wav',
                            label=f"most recent Avisoft .wav in ch{mic_idx + 1}",
                        )
                    except FileNotFoundError as e:
                        self.message_output(f"Skipping ch{mic_idx + 1} audio move: {e}")
                        continue

                    full_destination_path = str(pathlib.Path(total_dir_name_windows[0]) / 'audio' / 'original' / f"ch{mic_idx + 1}_{last_modified_audio_file.name}")
                    move_file_ps_command = f"Move-Item -Path '{last_modified_audio_file.name}' -Destination '{full_destination_path}' -ErrorAction SilentlyContinue"

                    single_audio_copy_subp = subprocess.Popen(
                        args=['powershell', '-Command', move_file_ps_command],
                        cwd=ch_dir,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.STDOUT,
                        shell=False
                    )

                    audio_copy_subprocesses.append(single_audio_copy_subp)
            else:
                relevant_file_count = max(1, int(math.ceil((self.exp_settings_dict['video_session_duration']+.36) / 5.09)))
                device_id = ['m', 's']
                for mic_pos_idx, mic_idx in enumerate([0, 12]):
                    ch_dir = pathlib.Path(self.exp_settings_dict['avisoft_basedirectory']) / f"ch{mic_idx + 1}"
                    audio_file_list = sorted(ch_dir.glob('*.wav'), key=lambda p: p.stat().st_ctime, reverse=True)[:relevant_file_count]
                    if not audio_file_list:
                        self.message_output(
                            f"Skipping ch{mic_idx + 1} multichannel audio move: "
                            f"no .wav files found under '{ch_dir}' (expected {relevant_file_count})."
                        )
                        continue
                    for aud_file in audio_file_list:

                        full_destination_path = str(pathlib.Path(total_dir_name_windows[0]) / 'audio' / 'original_mc' / f"{device_id[mic_pos_idx]}_{aud_file.name}")
                        move_file_ps_command = f"Move-Item -Path '{aud_file.name}' -Destination '{full_destination_path}' -ErrorAction SilentlyContinue"

                        multi_audio_copy_subp = subprocess.Popen(
                            args=['powershell', '-Command', move_file_ps_command],
                            cwd=ch_dir,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.STDOUT,
                            shell=False
                        )

                        audio_copy_subprocesses.append(multi_audio_copy_subp)

            # 30-minute budget for moving the audio files; network copy to
            # \\cup typically completes in seconds, so a timeout this long
            # only triggers when something is genuinely wrong (mount lost,
            # permission loop, disk full). Non-zero exit codes are logged
            # but do NOT abort here — the sync file move below and the
            # metadata dump still need to run so the user has a partial
            # recording on disk.
            wait_for_subprocesses(
                subps=audio_copy_subprocesses,
                max_seconds=30 * 60,
                label="audio file move",
                poll_interval_s=1,
                message_output=self.message_output,
                raise_on_nonzero=False,
                raise_on_timeout=False,
            )

        # move last modified sync file to primary file server
        coolterm_data_dir = pathlib.Path(self.exp_settings_dict['coolterm_basedirectory']) / 'Data'
        try:
            last_modified_sync_file = newest_match_or_raise(
                root=coolterm_data_dir,
                pattern='*.txt',
                label="most recent CoolTerm sync .txt",
            )
            shutil.move(src=last_modified_sync_file,
                        dst=pathlib.Path(total_dir_name_windows[0]) / 'sync' / last_modified_sync_file.name)
        except FileNotFoundError as e:
            self.message_output(f"Sync file move skipped: {e}")

        # ensure the video is done copying before proceeding
        while any(self.api.is_copying(_sn) for _sn in self.camera_serial_num):
            smart_wait(app_context_bool=self.app_context_bool, seconds=1)

        # copy metadata file to the file server(s)
        if self.metadata_settings:
            self.metadata_settings['Session']['session_id'] = session_id
            self.metadata_settings['Environment']['playpen_version'] = f"v{metadata.version('usv-playpen').split('.dev')[0]}"
            if 'usv_playpen_recording_version' in self.metadata_settings.get('Session', {}):
                self.metadata_settings['Session']['usv_playpen_recording_version'] = f"v{metadata.version('usv-playpen').split('.dev')[0]}"

            destination_path = Path(total_dir_name_windows[0]) / f"{session_id}_metadata.yaml"

            with open(destination_path, 'w') as f:
                yaml.dump(
                    self.metadata_settings,
                    f,
                    Dumper=SmartDumper,
                    default_flow_style=False,
                    sort_keys=False,
                    indent=2
                )

        # copy the audio, sync and video directories to the backup network drive(s)
        if len(total_dir_name_windows) > 1:
            for win_dir_idx, win_dir in enumerate(total_dir_name_windows[1:]):
                subprocess.Popen(
                    args=['robocopy', total_dir_name_windows[0], win_dir, '/MIR'],
                    cwd=total_dir_name_windows[0],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT,
                    shell=False
                )

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

            with open(pathlib.Path(total_dir_name_windows[0]) / 'audio' / 'audio_triggerbox_sync_info.json', 'w') as audio_dict_outfile:
                json.dump(audio_triggerbox_sync_info_dict, audio_dict_outfile, indent=4)

        self.message_output(f"Transferring audio/video files finished at: {datetime.datetime.now().hour:02d}:{datetime.datetime.now().minute:02d}:{datetime.datetime.now().second:02d}")

        Messenger(message_output=self.message_output,
                  receivers=self.email_receivers,
                  no_receivers_notification=False,
                  credentials_file=pathlib.Path(self.exp_settings_dict['credentials_directory']) / 'email_config.ini',
                  exp_settings_dict=self.exp_settings_dict).send_message(subject="Audio PC in 165B is available again, recording has been completed.",
                                                                         message=f"Thank you for your patience, recording by @{self.exp_settings_dict['experimenter']} was completed at "
                                                                                 f"{datetime.datetime.now().hour:02d}:{datetime.datetime.now().minute:02d}:{datetime.datetime.now().second:02d}. "
                                                                                 f"You will be notified about further experiments "
                                                                                 f"should they occur. \n \n ***This is an automatic e-mail, please do NOT respond.***")
        return self.metadata_settings


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
@click.option('--set', 'overrides', multiple=True, help='Override a setting, e.g., --set video_session_duration=20')
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

    metadata_path = pathlib.Path(__file__).parent / '_config/_metadata.yaml'
    with open(metadata_path, 'r') as f:
        metadata_settings = yaml.safe_load(f)

    if len(overrides) > 0:
        exp_settings_dict = override_toml_values(overrides=overrides, exp_settings_dict=exp_settings_dict)

    controller = ExperimentController(
        exp_settings_dict=exp_settings_dict,
        metadata_settings=metadata_settings
    )

    updated_metadata = controller.conduct_behavioral_recording()

    if updated_metadata:
        with open(metadata_path, 'w') as f:
            yaml.dump(
                updated_metadata,
                f,
                Dumper=SmartDumper,
                default_flow_style=False,
                sort_keys=False,
                indent=2
            )
