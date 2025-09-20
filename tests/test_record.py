"""
@author: bartulem
Test recording module.
"""

import pytest
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

