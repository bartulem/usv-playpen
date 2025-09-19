"""
@author: bartulem
Test recording module.
"""

import pytest
import os
import configparser
import paramiko
from unittest.mock import MagicMock, call, mock_open
from usv_playpen.behavioral_experiments import ExperimentController, count_last_recording_dropouts

@pytest.fixture
def default_exp_settings():
    """Provides a default experiment settings dictionary for tests."""

    return {
        'ethernet_network': 'Ethernet 4',
        'credentials_directory': '/fake/credentials',
        'avisoft_basedirectory': 'C:\\Avisoft\\',
        'coolterm_basedirectory': 'C:\\CoolTerm',
        'recording_files_destination_linux': ['/mnt/falkner/data'],
        'recording_files_destination_win': ['F:\\data'],
        'video_session_duration': 0.2,
        'conduct_audio_recording': True,
        'disable_ethernet': True,
        'avisoft_recorder_program_name': 'RECORDER_USGH.EXE',
        'avisoft_recorder_exe': 'C:\\Avisoft\\RECORDER_USGH',
        'audio': {
            'cpu_affinity': [2, 3],
            'cpu_priority': 'HIGH',
            'devices': {'usghflags': 1572}
        },
        'video': {
            'general': {
                'expected_cameras': ['SN12345'],
                'recording_codec': 'H264',
                'delete_post_copy': True
            },
            'metadata': {
                'experimenter': 'TestBot',
                'notes': 'This is a simulated test run.'
            }
        }
    }


@pytest.fixture
def controller(default_exp_settings):
    """Provides an ExperimentController instance with default settings."""

    return ExperimentController(exp_settings_dict=default_exp_settings)

def test_count_dropouts_file_not_found():
    """Test that count_last_recording_dropouts returns None if the file doesn't exist."""

    assert count_last_recording_dropouts('/nonexistent/path', 'ch1') is None

def test_count_dropouts_parsing(mocker):
    """Tests the dropout counting logic with a simulated log file."""

    log_content = (
        "Recording started: C:\\Avisoft\\ch1\\rec1.wav\n"
        "Some info here.\n"
        "dropout detected\n"
        "Recording stopped.\n"
        "Recording started: C:\\Avisoft\\ch1\\rec2.wav\n"
        "Everything is fine.\n"
        "Recording stopped.\n"
        "Recording started: C:\\Avisoft\\ch1\\rec3.wav\n"
        "A problem occurred: dropout\n"
        "Another problem: dropout\n"
        "Yet another problem: dropout\n"
        "Recording stopped.\n"
    )
    # mock the built-in `open` function to return our fake log content
    mocker.patch("builtins.open", mock_open(read_data=log_content))

    assert count_last_recording_dropouts("C:\\Avisoft\\", "ch1") == 3


def test_count_dropouts_no_dropouts(mocker):
    """Tests the case where the last recording has no dropouts."""

    log_content = (
        "Recording started: C:\\Avisoft\\ch1\\rec1.wav\n"
        "dropout detected\n"
        "Recording started: C:\\Avisoft\\ch1\\rec2.wav\n"
        "No problems here.\n"
    )
    mocker.patch("builtins.open", mock_open(read_data=log_content))
    assert count_last_recording_dropouts("C:\\Avisoft\\", "ch1") == 0


def test_get_cpu_affinity_mask(controller):
    """Tests the CPU affinity mask calculation."""

    controller.exp_settings_dict['audio']['cpu_affinity'] = [0, 1, 4]
    # 2^0 + 2^1 + 2^4 = 1 + 2 + 16 = 19, which is 0x13 in hex
    assert controller.get_cpu_affinity_mask() == '0x13'


def test_get_connection_params(mocker, tmp_path, controller):
    """Tests reading of connection parameters from a simulated config file."""

    # tmp_path is a pytest fixture that provides a temporary directory
    config_dir = tmp_path
    controller.exp_settings_dict['credentials_directory'] = str(config_dir)
    config_file = config_dir / "motif_config.ini"

    # create a fake config file
    config = configparser.ConfigParser()
    config['motif'] = {
        'master_ip_address': '192.168.1.10',
        'second_ip_address': '192.168.1.11',
        'ssh_port': '22',
        'ssh_username': 'user',
        'ssh_password': 'pw',
        'api': 'fake_api_key'
    }
    with open(config_file, 'w') as f:
        config.write(f)

    params = controller.get_connection_params()
    assert params == ('192.168.1.10', '192.168.1.11', '22', 'user', 'pw', 'fake_api_key')


def test_check_remote_mount_success(mocker, controller):
    """Simulates a successful remote mount check."""

    # mock the entire paramiko SSHClient
    mock_ssh_client = MagicMock()
    mocker.patch('paramiko.SSHClient', return_value=mock_ssh_client)

    # simulate the output of the remote command
    mock_stdin, mock_stdout, mock_stderr = MagicMock(), MagicMock(), MagicMock()
    mock_stdout.read.return_value = b'True'
    mock_stderr.read.return_value = b''
    mock_ssh_client.exec_command.return_value = (mock_stdin, mock_stdout, mock_stderr)

    result = controller.check_remote_mount('host', 22, 'user', 'pw', '/mnt/data')

    mock_ssh_client.connect.assert_called_with(hostname='host', port=22, username='user', password='pw', timeout=10)
    assert result is True


def test_check_remote_mount_auth_failure(mocker, controller):
    """Simulates an authentication failure during SSH connection."""

    mock_ssh_client = MagicMock()
    # make the connect method raise an exception
    mock_ssh_client.connect.side_effect = paramiko.AuthenticationException("Auth failed")
    mocker.patch('paramiko.SSHClient', return_value=mock_ssh_client)

    result = controller.check_remote_mount('host', 22, 'user', 'pw', '/mnt/data')
    assert result is False


def test_conduct_behavioral_recording_simulation(mocker, controller):
    """
    A comprehensive simulation of the `conduct_behavioral_recording` method.
    This test mocks all external interactions to verify the internal logic.
    """

    # mock methods within the class itself
    mocker.patch.object(controller, 'check_ethernet_connection')
    mocker.patch.object(controller, 'modify_audio_file')
    mocker.patch.object(controller, 'remount_cup_drives_on_windows')
    mocker.patch.object(controller, 'get_custom_dir_names', return_value=(
        '12:00:00',
        ['/linux/path/20250901_120000'],
        ['F:\\data\\20250901_120000']
    ))

    # mock external libraries and functions
    mock_messenger_cls = mocker.patch('experiment_runner.Messenger')
    mock_subprocess_popen = mocker.patch('subprocess.Popen')
    mock_subprocess_run = mocker.patch('subprocess.run')
    mock_shutil_move = mocker.patch('shutil.move')
    mock_pathlib_mkdir = mocker.patch('pathlib.Path.mkdir')
    mocker.patch('glob.glob', return_value=['C:\\CoolTerm\\Data\\fake_sync_file.txt', 'C:\\Avisoft\\ch1\\fake_audio.wav'])
    mocker.patch('os.path.getctime', return_value=12345.678)
    mock_count_dropouts = mocker.patch('experiment_runner.count_last_recording_dropouts', return_value=0)
    mock_json_dump = mocker.patch('json.dump')
    mocker.patch('experiment_runner.smart_wait')  # Prevent tests from actually waiting

    # mock the Motif API
    mock_api = MagicMock()
    mock_api.call.return_value = {'now': 1756728000.0}  # A fake timestamp
    mock_api.is_copying.return_value = False  # Simulate that copying finishes instantly
    controller.api = mock_api  # Manually set the api attribute
    controller.camera_serial_num = ['SN12345']

    # simulate the output of `tasklist` to show Avisoft is running
    mock_tasklist_result = MagicMock()
    mock_tasklist_result.returncode = 0
    mock_tasklist_result.stdout = "RECORDER_USGH.EXE"
    mock_subprocess_run.return_value = mock_tasklist_result

    controller.conduct_behavioral_recording()

    # check that initial setup methods were called
    controller.check_ethernet_connection.assert_called_once()
    controller.modify_audio_file.assert_called_once()

    # check that processes were started (CoolTerm and Avisoft Recorder)
    popen_calls = mock_subprocess_popen.call_args_list
    assert any('coolterm_config.stc' in str(c) for c in popen_calls)
    assert any('RECORDER_USGH.EXE' in str(c) and '/AUT' in str(c) for c in popen_calls)

    # check that Motif video recording was started
    mock_api.call.assert_any_call(
        f"camera/{controller.exp_settings_dict['video']['general']['expected_cameras'][0]}/recording/start",
        duration=pytest.approx(0.2 * 60),
        codec='H264',
        metadata=controller.exp_settings_dict['video']['metadata']
    )

    # check that ethernet was disabled and then re-enabled
    assert any('disable' in str(c) for c in popen_calls)
    assert any('enable' in str(c) for c in popen_calls)

    # check that directories were created
    mock_pathlib_mkdir.assert_has_calls([
        call(parents=True, exist_ok=True)
    ], any_order=True)

    # check that Motif file copying was initiated
    mock_api.call.assert_any_call(
        'camera/SN12345/recordings/copy_all',
        delete_after=True,
        location='/linux/path/20250901_120000/video'
    )

    # check that audio and sync files were moved
    mock_shutil_move.assert_called_once()
    # the Popen call for moving audio files
    assert any('move' in str(c) and 'fake_audio.wav' in str(c) for c in popen_calls)

    # check that dropout analysis was performed
    mock_count_dropouts.assert_called_once()
    mock_json_dump.assert_called_once()

    # check that start and end emails were sent
    assert mock_messenger_cls.return_value.send_message.call_count == 2
    send_message_calls = mock_messenger_cls.return_value.send_message.call_args_list
    assert 'is busy' in send_message_calls[0].kwargs['subject']
    assert 'is available again' in send_message_calls[1].kwargs['subject']
