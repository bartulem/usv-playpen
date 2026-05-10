"""
@author: bartulem
Test recording module.
"""

import os
import pathlib
import shutil

import pytest
import configparser
import paramiko
from click.testing import CliRunner
from unittest.mock import MagicMock
import usv_playpen.behavioral_experiments as be_mod
from usv_playpen.behavioral_experiments import (
    ExperimentController,
    count_last_recording_dropouts,
    conduct_calibration_cli,
    conduct_recording_cli,
)
from email.message import EmailMessage
from usv_playpen.send_email import Messenger


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
    """Simulates a successful remote mount check, and asserts the two
    paramiko hardening measures applied on every connect:
      - ssh-rsa (SHA-1) pubkey algorithm disabled (paramiko advisory)
      - RejectPolicy + load_system_host_keys (CodeQL py/paramiko-missing
        -host-key-validation)."""

    # mock the entire paramiko SSHClient
    mock_ssh_client = MagicMock()
    mocker.patch('paramiko.SSHClient', return_value=mock_ssh_client)

    # simulate the output of the remote command
    mock_stdin, mock_stdout, mock_stderr = MagicMock(), MagicMock(), MagicMock()
    mock_stdout.read.return_value = b'True'
    mock_stderr.read.return_value = b''
    mock_ssh_client.exec_command.return_value = (mock_stdin, mock_stdout, mock_stderr)

    result = controller.check_remote_mount('host', 22, 'user', 'pw', '/mnt/data')

    # connect kwargs include the disabled_algorithms mitigation
    mock_ssh_client.connect.assert_called_with(
        hostname='host', port=22, username='user', password='pw', timeout=10,
        disabled_algorithms={'pubkeys': ['ssh-rsa']},
    )
    # known_hosts loaded before connect
    mock_ssh_client.load_system_host_keys.assert_called_once()
    # And RejectPolicy is set as the missing-host-key handler — never
    # AutoAddPolicy, which would silently trust an unknown / spoofed host.
    set_policy_call = mock_ssh_client.set_missing_host_key_policy.call_args
    policy_arg = set_policy_call.args[0]
    assert isinstance(policy_arg, paramiko.RejectPolicy)
    assert result is True


def test_check_remote_mount_handles_missing_known_hosts(mocker, controller):
    """If ~/.ssh/known_hosts is missing, load_system_host_keys raises
    FileNotFoundError. The function must still set RejectPolicy and proceed
    with the connect (which will then fail loudly if the host isn't
    pre-trusted, rather than silently auto-adding the key)."""
    mock_ssh_client = MagicMock()
    mock_ssh_client.load_system_host_keys.side_effect = FileNotFoundError("no known_hosts")
    mocker.patch('paramiko.SSHClient', return_value=mock_ssh_client)

    mock_stdout, mock_stderr = MagicMock(), MagicMock()
    mock_stdout.read.return_value = b'True'
    mock_stderr.read.return_value = b''
    mock_ssh_client.exec_command.return_value = (MagicMock(), mock_stdout, mock_stderr)

    result = controller.check_remote_mount('h', 22, 'u', 'p', '/m')

    # The FileNotFoundError must be swallowed; RejectPolicy still applied.
    set_policy_call = mock_ssh_client.set_missing_host_key_policy.call_args
    assert isinstance(set_policy_call.args[0], paramiko.RejectPolicy)
    # And the connect was still attempted
    assert mock_ssh_client.connect.called
    assert result is True


def test_check_remote_mount_auth_failure(mocker, controller):
    """Simulates an authentication failure during SSH connection."""

    mock_ssh_client = MagicMock()
    # make the connect method raise an exception
    mock_ssh_client.connect.side_effect = paramiko.AuthenticationException("Auth failed")
    mocker.patch('paramiko.SSHClient', return_value=mock_ssh_client)

    result = controller.check_remote_mount('host', 22, 'user', 'pw', '/mnt/data')
    assert result is False


@pytest.fixture
def credentials_file(tmp_path):
    cfg = tmp_path / 'email_config.ini'
    cfg.write_text(
        "[email]\n"
        "email_host = smtp.example.com\n"
        "email_port = 465\n"
        "email_address = alice@example.com\n"
        "email_password = hunter2\n"
    )
    return str(cfg)


def test_send_message_no_receivers_returns_none_and_notifies():
    messages: list[str] = []
    m = Messenger(receivers=[], message_output=messages.append)
    out = m.send_message(subject='s', message='m')
    assert out is None
    assert any("not to notify" in msg for msg in messages)


def test_send_message_no_receivers_silent_when_notification_disabled():
    messages: list[str] = []
    m = Messenger(
        receivers=[],
        message_output=messages.append,
        no_receivers_notification=False,
    )
    out = m.send_message(subject='s', message='m')
    assert out is None
    assert messages == []


def test_get_email_params_reads_ini(credentials_file):
    m = Messenger(credentials_file=credentials_file)
    host, port, addr, pwd = m.get_email_params()
    assert host == 'smtp.example.com'
    assert port == '465'
    assert addr == 'alice@example.com'
    assert pwd == 'hunter2'


def test_get_email_params_exits_when_credentials_missing(tmp_path, capsys):
    m = Messenger(credentials_file=str(tmp_path / 'absent.ini'))
    with pytest.raises(SystemExit):
        m.get_email_params()


def test_send_message_success(monkeypatch, credentials_file):
    sent: dict = {}

    class FakeSMTP:
        def __init__(self, host, port):
            sent['host'] = host
            sent['port'] = port

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def login(self, addr, pwd):
            sent['login'] = (addr, pwd)

        def send_message(self, msg):
            sent['msg'] = msg

    import usv_playpen.send_email as send_email_mod
    monkeypatch.setattr(send_email_mod.smtplib, 'SMTP_SSL', FakeSMTP)

    m = Messenger(
        receivers=['bob@example.com', 'carol@example.com'],
        credentials_file=credentials_file,
        message_output=lambda *_: None,
    )
    out = m.send_message(subject='Subject', message='Body text')
    assert out is True
    assert sent['host'] == 'smtp.example.com'
    assert sent['port'] == '465'
    assert sent['login'] == ('alice@example.com', 'hunter2')
    msg: EmailMessage = sent['msg']
    assert msg['Subject'] == 'Subject'
    assert msg['From'] == 'alice@example.com'
    assert msg['To'] == 'bob@example.com, carol@example.com'
    assert 'Body text' in msg.get_content()


def test_send_message_logs_smtp_error_and_returns_false(monkeypatch, credentials_file):
    class FailingSMTP:
        def __init__(self, host, port):
            pass

        def __enter__(self):
            raise OSError("connection reset")

        def __exit__(self, *_exc):
            return False

    import usv_playpen.send_email as send_email_mod
    monkeypatch.setattr(send_email_mod.smtplib, 'SMTP_SSL', FailingSMTP)

    messages: list[str] = []
    m = Messenger(
        receivers=['x@example.com'],
        credentials_file=credentials_file,
        message_output=messages.append,
    )
    out = m.send_message(subject='s', message='m')
    assert out is False
    joined = "\n".join(messages)
    assert 'OSError' in joined
    # Confirm the failing host/port pair is reported in the exact "host:port"
    # form produced by send_email.Messenger.send_message; checked via an
    # equality test against the extracted token rather than a substring "in"
    # check so static analyzers don't mistake this for URL sanitization.
    via_marker = 'via '
    assert via_marker in joined
    after_via = joined.split(via_marker, 1)[1]
    host_port_token = after_via.split(':', 2)
    assert host_port_token[0] == 'smtp.example.com'
    assert host_port_token[1] == '465'


# ---------------------------------------------------------------------------
# behavioral_experiments — extended mock-based coverage
#
# Everything below this line tests ExperimentController without any real
# hardware: subprocess calls, paramiko, motifapi, file-system polling,
# and the top-level orchestration in conduct_behavioral_recording.
# ---------------------------------------------------------------------------


# ---- count_last_recording_dropouts: pure log parsing ----------------------


def test_count_last_dropouts_zero_when_no_recordings(tmp_path):
    """Empty log file should report 0 dropouts (file exists, no sessions yet)."""
    ch_dir = tmp_path / 'ch1'
    ch_dir.mkdir()
    (ch_dir / 'ch1.log').write_text('')
    assert count_last_recording_dropouts(str(tmp_path), 'ch1') == 0


def test_count_last_dropouts_counts_only_last_session(tmp_path):
    """Only the dropouts in the most recent recording should be counted."""
    ch_dir = tmp_path / 'ch1'
    ch_dir.mkdir()
    sep_path = str(ch_dir) + os.sep
    log = (
        f"{sep_path}rec1\n"
        "dropout dropout dropout\n"
        f"{sep_path}rec2\n"
        "dropout\n"
    )
    (ch_dir / 'ch1.log').write_text(log)
    assert count_last_recording_dropouts(str(tmp_path), 'ch1') == 1


def test_count_last_dropouts_no_dropouts_in_last_session(tmp_path):
    """Last session having no dropout marker yields 0."""
    ch_dir = tmp_path / 'ch1'
    ch_dir.mkdir()
    sep_path = str(ch_dir) + os.sep
    (ch_dir / 'ch1.log').write_text(
        f"{sep_path}rec1\ndropout\n"
        f"{sep_path}rec2\nno errors here\n"
    )
    assert count_last_recording_dropouts(str(tmp_path), 'ch1') == 0


# ---- get_cpu_affinity_mask edge cases -------------------------------------


def test_get_cpu_affinity_mask_empty_returns_zero(controller):
    """Empty CPU list yields 0x0 (no bits set)."""
    controller.exp_settings_dict['audio']['cpu_affinity'] = []
    assert controller.get_cpu_affinity_mask() == '0x0'


def test_get_cpu_affinity_mask_single_high_bit(controller):
    """CPU 15 should map to 0x8000."""
    controller.exp_settings_dict['audio']['cpu_affinity'] = [15]
    assert controller.get_cpu_affinity_mask() == '0x8000'


# ---- get_custom_dir_names: pure -------------------------------------------


def test_get_custom_dir_names_format(controller):
    """sub_dir_name must be YYYYMMDD_HHMMSS; total_dir_names must list-multiply."""
    import datetime as _dt
    now = _dt.datetime(2026, 5, 9, 13, 7, 2).timestamp()
    controller.exp_settings_dict['recording_files_destination_linux'] = ['/lin1', '/lin2']
    controller.exp_settings_dict['recording_files_destination_win'] = ['F:\\w1', 'M:\\w2']
    start_str, lin_dirs, win_dirs, sub = controller.get_custom_dir_names(now=now)
    assert sub == "20260509_130702"
    assert start_str == "13:07:02"
    assert lin_dirs == ['/lin1/20260509_130702', '/lin2/20260509_130702']
    assert win_dirs == ['F:\\w1\\20260509_130702', 'M:\\w2\\20260509_130702']


# ---- get_cup_mount_params -------------------------------------------------


def test_get_cup_mount_params_reads_ini(tmp_path, controller):
    """Happy path — username / password are returned from cup_config.ini."""
    controller.exp_settings_dict['credentials_directory'] = str(tmp_path)
    cfg = configparser.ConfigParser()
    cfg['cup'] = {'username': 'someuser', 'password': 'somepw'}
    with open(tmp_path / 'cup_config.ini', 'w') as f:
        cfg.write(f)
    assert controller.get_cup_mount_params() == ('someuser', 'somepw')


def test_get_cup_mount_params_exits_when_missing(tmp_path, controller):
    """SystemExit when the credentials file is absent (mirrors get_connection_params)."""
    controller.exp_settings_dict['credentials_directory'] = str(tmp_path)
    with pytest.raises(SystemExit):
        controller.get_cup_mount_params()


def test_get_connection_params_exits_when_missing(tmp_path, controller):
    """SystemExit when motif_config.ini is absent."""
    controller.exp_settings_dict['credentials_directory'] = str(tmp_path)
    with pytest.raises(SystemExit):
        controller.get_connection_params()


# ---- check_ethernet_connection (subprocess + smart_wait mocks) ------------


def test_check_ethernet_connection_already_up(mocker, controller):
    """If the adapter is already 'Up', no enable command is issued."""
    fake_run = mocker.patch('usv_playpen.behavioral_experiments.subprocess.run')
    fake_run.return_value = MagicMock(stdout='Up\n', returncode=0)
    mocker.patch('usv_playpen.behavioral_experiments.smart_wait')
    controller.check_ethernet_connection()
    # Only the initial status-check Get-NetAdapter call should have been made;
    # netsh / enable should never appear.
    invoked_args = [c.args[0] for c in fake_run.call_args_list]
    assert all('netsh' not in argv for argv in invoked_args)
    assert len(invoked_args) == 1


def test_check_ethernet_connection_recovers(mocker, controller):
    """Adapter starts 'Down' but comes 'Up' after enable — must not exit."""
    fake_run = mocker.patch('usv_playpen.behavioral_experiments.subprocess.run')
    # 1st call: status -> Down
    # 2nd call: netsh enable -> ok
    # 3rd call: status -> Up
    fake_run.side_effect = [
        MagicMock(stdout='Disconnected', returncode=0),
        MagicMock(stdout='', returncode=0),
        MagicMock(stdout='Up', returncode=0),
    ]
    mocker.patch('usv_playpen.behavioral_experiments.smart_wait')
    controller.check_ethernet_connection()
    # netsh interface set interface ... enable was called exactly once
    netsh_calls = [c for c in fake_run.call_args_list
                   if 'netsh' in c.args[0]]
    assert len(netsh_calls) == 1


def test_check_ethernet_connection_timeout_exits(mocker, controller):
    """Adapter never comes up within the polling budget → SystemExit(1)."""
    # Always-Down responses for status checks; netsh enable is a no-op.
    def _fake(cmd, **_kw):
        if 'netsh' in cmd:
            return MagicMock(stdout='', returncode=0)
        return MagicMock(stdout='Disconnected', returncode=0)

    mocker.patch('usv_playpen.behavioral_experiments.subprocess.run', side_effect=_fake)
    mocker.patch('usv_playpen.behavioral_experiments.smart_wait')
    with pytest.raises(SystemExit):
        controller.check_ethernet_connection()


# ---- purge_cup_connections_on_windows -------------------------------------


def test_purge_cup_connections_invokes_expected_commands(mocker, controller):
    """Verifies the full purge sequence: per-drive 'net use /delete' (5x),
    'klist purge' for Kerberos tickets, and 'cmdkey /delete:cup' for the
    Credential Manager entry. The cmdkey call is the key fix for the
    multi-user-on-shared-Windows-account 1326 failure where /persistent:yes
    leaves a stored credential under the previous user's identity."""
    fake_run = mocker.patch('usv_playpen.behavioral_experiments.subprocess.run')
    mocker.patch('usv_playpen.behavioral_experiments.smart_wait')
    controller.purge_cup_connections_on_windows()
    invoked = [tuple(c.args[0]) for c in fake_run.call_args_list]
    # Per-drive deletes (F:, \\cup\falkner, M:, \\cup\murthy) + server-level
    # \\cup delete + klist purge + cmdkey /delete:cup = 7 invocations.
    assert len(invoked) == 7
    assert ("net", "use", "F:", "/delete", "/y") in invoked
    assert ("net", "use", r"\\cup\falkner", "/delete", "/y") in invoked
    assert ("net", "use", "M:", "/delete", "/y") in invoked
    assert ("net", "use", r"\\cup\murthy", "/delete", "/y") in invoked
    assert ("net", "use", r"\\cup", "/delete", "/y") in invoked
    assert ("klist", "purge") in invoked
    assert ("cmdkey", "/delete:cup") in invoked


# ---- remount_cup_drives_on_windows ----------------------------------------


def _stub_get_cup_params(monkeypatch):
    monkeypatch.setattr(
        ExperimentController, 'get_cup_mount_params',
        lambda self: ('cup_user', 'cup_pw'),
    )


def test_remount_cup_drives_happy_path(mocker, monkeypatch, controller):
    """Mount succeeds on first try; verification reports the drive accessible."""
    _stub_get_cup_params(monkeypatch)
    monkeypatch.setattr(ExperimentController, 'check_ethernet_connection', lambda self: None)
    monkeypatch.setattr(ExperimentController, 'purge_cup_connections_on_windows', lambda self: None)
    mocker.patch('usv_playpen.behavioral_experiments.smart_wait')
    fake_run = mocker.patch('usv_playpen.behavioral_experiments.subprocess.run')
    fake_run.return_value = MagicMock(returncode=0, stderr='', stdout='')
    # The verify step uses ThreadPoolExecutor → pathlib.Path.is_dir(); we
    # stub that to always succeed.
    mocker.patch('pathlib.Path.is_dir', return_value=True)

    controller.exp_settings_dict['recording_files_destination_win'] = ['F:\\Some\\Path']
    out_messages = []
    controller.message_output = out_messages.append
    controller.remount_cup_drives_on_windows()

    joined = "\n".join(out_messages)
    assert "successfully mounted" in joined
    # Should have invoked 'net use F: \\cup\falkner ...' at least once
    mounts = [c for c in fake_run.call_args_list
              if c.args[0][:2] == ['net', 'use'] and c.args[0][2] == 'F:'
              and c.args[0][3] == r"\\cup\falkner"]
    assert len(mounts) >= 1


def test_remount_cup_drives_critical_when_all_fail(mocker, monkeypatch, controller):
    """All retries fail → CRITICAL ERROR message is emitted."""
    _stub_get_cup_params(monkeypatch)
    monkeypatch.setattr(ExperimentController, 'check_ethernet_connection', lambda self: None)
    monkeypatch.setattr(ExperimentController, 'purge_cup_connections_on_windows', lambda self: None)
    mocker.patch('usv_playpen.behavioral_experiments.smart_wait')
    fake_run = mocker.patch('usv_playpen.behavioral_experiments.subprocess.run')
    fake_run.return_value = MagicMock(returncode=2, stderr='generic failure', stdout='')
    mocker.patch('pathlib.Path.is_dir', return_value=False)

    controller.exp_settings_dict['recording_files_destination_win'] = ['F:\\Some\\Path']
    out_messages = []
    controller.message_output = out_messages.append
    controller.remount_cup_drives_on_windows()

    assert any("CRITICAL ERROR" in m for m in out_messages)


# ---- check_remote_mount additional branches -------------------------------


def test_check_remote_mount_returns_false_when_not_a_mount(mocker, controller):
    """`os.path.ismount` returning 'False' on the remote → method returns False."""
    mock_ssh_client = MagicMock()
    mocker.patch('paramiko.SSHClient', return_value=mock_ssh_client)
    mock_stdout, mock_stderr = MagicMock(), MagicMock()
    mock_stdout.read.return_value = b'False'
    mock_stderr.read.return_value = b''
    mock_ssh_client.exec_command.return_value = (MagicMock(), mock_stdout, mock_stderr)
    assert controller.check_remote_mount('h', 22, 'u', 'p', '/m') is False


def test_check_remote_mount_returns_false_on_remote_stderr(mocker, controller):
    """If the remote process prints to stderr, the function reports False."""
    mock_ssh_client = MagicMock()
    mocker.patch('paramiko.SSHClient', return_value=mock_ssh_client)
    mock_stdout, mock_stderr = MagicMock(), MagicMock()
    mock_stdout.read.return_value = b''
    mock_stderr.read.return_value = b'remote python crashed'
    mock_ssh_client.exec_command.return_value = (MagicMock(), mock_stdout, mock_stderr)
    assert controller.check_remote_mount('h', 22, 'u', 'p', '/m') is False


def test_check_remote_mount_ssh_exception(mocker, controller):
    """A paramiko.SSHException returns False without raising."""
    mock_ssh_client = MagicMock()
    mock_ssh_client.connect.side_effect = paramiko.SSHException("kex failed")
    mocker.patch('paramiko.SSHClient', return_value=mock_ssh_client)
    assert controller.check_remote_mount('h', 22, 'u', 'p', '/m') is False


# ---- verify_avisoft_is_recording (filesystem polling) ---------------------


@pytest.fixture
def avisoft_dir_factory(tmp_path):
    """Builds <tmp_path>/avisoft/ch{N}/ for the requested mic indices."""
    base = tmp_path / 'avisoft'
    base.mkdir()
    def _make(mic_indices):
        for mic_idx in mic_indices:
            (base / f"ch{mic_idx + 1}").mkdir()
        return base
    return _make


def test_verify_avisoft_returns_true_with_no_channels(mocker, controller):
    """When no channel directories resolve, verification is a graceful pass."""
    controller.exp_settings_dict['avisoft_basedirectory'] = '/does-not-exist-anywhere'
    controller.exp_settings_dict['audio']['general'] = {'total': 0}
    controller.exp_settings_dict['audio']['used_mics'] = []
    mocker.patch('usv_playpen.behavioral_experiments.smart_wait')
    assert controller.verify_avisoft_is_recording() is True


def test_verify_avisoft_detects_new_file(mocker, controller, avisoft_dir_factory):
    """A new .wav appearing in chN counts as 'producing audio'."""
    base = avisoft_dir_factory([0])
    controller.exp_settings_dict['avisoft_basedirectory'] = str(base)
    controller.exp_settings_dict['audio']['general'] = {'total': 0}
    controller.exp_settings_dict['audio']['used_mics'] = [0]

    sleeps = []
    def fake_wait(app_context_bool=False, seconds=0):
        # On the first poll, drop a fresh wav into ch1/.
        sleeps.append(seconds)
        if len(sleeps) == 2:
            (base / 'ch1' / 'fresh.wav').write_bytes(b'x' * 4096)

    mocker.patch('usv_playpen.behavioral_experiments.smart_wait', side_effect=fake_wait)
    assert controller.verify_avisoft_is_recording(warmup_s=1, poll_interval_s=1, max_wait_s=10) is True


def test_verify_avisoft_detects_growth(mocker, controller, avisoft_dir_factory):
    """An existing .wav growing in size between polls counts as healthy."""
    base = avisoft_dir_factory([0])
    controller.exp_settings_dict['avisoft_basedirectory'] = str(base)
    controller.exp_settings_dict['audio']['general'] = {'total': 0}
    controller.exp_settings_dict['audio']['used_mics'] = [0]

    wav = base / 'ch1' / 'rolling.wav'
    wav.write_bytes(b'a' * 1024)  # baseline

    # Grow the file only AFTER the warmup snapshot has been taken. The first
    # smart_wait call corresponds to the warmup; the baseline is then snapped
    # at 1024 bytes; on the next (poll) wait we bump the file to 8192 so the
    # poll observes growth.
    counter = [0]
    def fake_wait(app_context_bool=False, seconds=0):
        counter[0] += 1
        if counter[0] >= 2:
            wav.write_bytes(b'a' * 8192)

    mocker.patch('usv_playpen.behavioral_experiments.smart_wait', side_effect=fake_wait)
    assert controller.verify_avisoft_is_recording(warmup_s=1, poll_interval_s=1, max_wait_s=5) is True


def test_verify_avisoft_returns_false_when_no_growth(mocker, controller, avisoft_dir_factory):
    """If polling completes with no byte progress in chN, return False."""
    base = avisoft_dir_factory([0])
    controller.exp_settings_dict['avisoft_basedirectory'] = str(base)
    controller.exp_settings_dict['audio']['general'] = {'total': 0}
    controller.exp_settings_dict['audio']['used_mics'] = [0]
    (base / 'ch1' / 'static.wav').write_bytes(b'a' * 1024)

    mocker.patch('usv_playpen.behavioral_experiments.smart_wait')
    assert controller.verify_avisoft_is_recording(warmup_s=1, poll_interval_s=1, max_wait_s=2) is False


# ---- check_camera_vitals (motifapi mocks) ---------------------------------


def _stub_motif_creds(monkeypatch):
    monkeypatch.setattr(
        ExperimentController, 'get_connection_params',
        lambda self: ('1.2.3.4', '1.2.3.5', '22', 'sshuser', 'sshpw', 'apikey'),
    )
    monkeypatch.setattr(
        ExperimentController, 'check_remote_mount',
        lambda self, **kw: True,
    )


def test_check_camera_vitals_motif_unreachable(mocker, monkeypatch, controller):
    """When MotifApi raises MotifError, the method exits with code 1."""
    import motifapi as _motifapi
    _stub_motif_creds(monkeypatch)
    controller.exp_settings_dict['video']['general']['expected_cameras'] = ['SN12345']

    def _explode(*_a, **_kw):
        raise _motifapi.api.MotifError('not running')

    mocker.patch('usv_playpen.behavioral_experiments.motifapi.MotifApi', side_effect=_explode)
    with pytest.raises(SystemExit):
        controller.check_camera_vitals(camera_fr=150)


def test_check_camera_vitals_happy_path_single_camera(mocker, monkeypatch, controller):
    """Single-camera path: configures the camera, sets frame rate, stores api ref."""
    _stub_motif_creds(monkeypatch)
    controller.exp_settings_dict['video']['general']['expected_cameras'] = ['SN12345']
    controller.exp_settings_dict['video']['general']['monitor_recording'] = False
    controller.exp_settings_dict['video']['cameras_config'] = {
        'SN12345': {'exposure_time': 5000, 'gain': 1.0},
    }

    fake_api = MagicMock()
    fake_api.call.side_effect = lambda endpoint, **_kw: {
        'cameras': {'cameras': [{'serial': 'SN12345'}]},
        'version': {'software': '4.2.0'},
    }.get(endpoint, MagicMock())

    mocker.patch('usv_playpen.behavioral_experiments.motifapi.MotifApi', return_value=fake_api)
    mocker.patch('usv_playpen.behavioral_experiments.smart_wait')
    controller.check_camera_vitals(camera_fr=150)

    # The API ref is captured for later use, and the per-camera configure call
    # was issued with ExposureTime / Gain values from cameras_config.
    assert controller.api is fake_api
    configure_calls = [c for c in fake_api.call.call_args_list
                       if c.args and c.args[0] == 'camera/SN12345/configure']
    assert any('ExposureTime' in c.kwargs and 'Gain' in c.kwargs for c in configure_calls)
    # And the single-camera AcquisitionFrameRate path was taken
    fr_calls = [c for c in fake_api.call.call_args_list
                if c.args and c.args[0] == 'camera/SN12345/configure'
                and c.kwargs.get('AcquisitionFrameRate') == 150]
    assert len(fr_calls) == 1


def test_check_camera_vitals_remote_mount_fails(mocker, monkeypatch, controller):
    """A failing remote-mount check exits before any Motif call is made."""
    monkeypatch.setattr(
        ExperimentController, 'get_connection_params',
        lambda self: ('1.2.3.4', '1.2.3.5', '22', 'u', 'p', 'k'),
    )
    monkeypatch.setattr(ExperimentController, 'check_remote_mount', lambda self, **kw: False)
    with pytest.raises(SystemExit):
        controller.check_camera_vitals(camera_fr=150)


# ---- conduct_tracking_calibration (orchestration) -------------------------


def test_conduct_tracking_calibration_kicks_off_recording(mocker, monkeypatch, controller):
    """Calibration: vitals check + 'recording/start' call with calibration_duration."""
    monkeypatch.setattr(ExperimentController, 'check_camera_vitals',
                        lambda self, camera_fr: setattr(self, 'api', MagicMock()))
    mocker.patch('usv_playpen.behavioral_experiments.smart_wait')

    controller.exp_settings_dict['video']['general']['calibration_frame_rate'] = 50
    controller.exp_settings_dict['video']['general']['recording_codec'] = 'H264'
    controller.exp_settings_dict['video']['general']['expected_cameras'] = ['SN1', 'SN2']
    controller.exp_settings_dict['calibration_duration'] = 3

    controller.conduct_tracking_calibration()

    # api.call('recording/start', duration=..., codec=...) was issued
    calls = controller.api.call.call_args_list
    rec_starts = [c for c in calls if c.args and c.args[0] == 'recording/start']
    assert len(rec_starts) == 1
    # duration is in seconds in the API: 3 minutes -> 180 s
    assert rec_starts[0].kwargs.get('duration') == 180
    assert rec_starts[0].kwargs.get('codec') == 'H264'


# ---- modify_audio_file (uses real config + redirected __file__) -----------


# Captured at import time, BEFORE any test can monkeypatch be_mod.__file__.
# Used to find the real package _config/ for tests that need to seed a
# tmp copy of avisoft_config.ini and read realistic settings.
_REAL_PKG_DIR = pathlib.Path(be_mod.__file__).parent


def _seed_avisoft_config(tmp_path):
    """Copy the package's avisoft_config.ini into a tmp _config/ dir.

    Returns the directory that the test should pretend is the module's
    source directory (used by monkeypatching be_mod.__file__ to redirect
    `pathlib.Path(__file__).parent / '_config/...'` lookups).
    """
    real_cfg = _REAL_PKG_DIR / '_config' / 'avisoft_config.ini'
    fake_module_dir = tmp_path / 'fake_pkg'
    (fake_module_dir / '_config').mkdir(parents=True)
    shutil.copy(real_cfg, fake_module_dir / '_config' / 'avisoft_config.ini')
    return fake_module_dir


def _realistic_audio_settings():
    """Loads the real behavioral_experiments_settings.toml from the package
    so modify_audio_file can iterate over a complete settings dict without us
    having to hand-build every nested section. Resolved via the import-time
    captured _REAL_PKG_DIR so a monkeypatched __file__ does not redirect us."""
    import toml as _toml
    with open(_REAL_PKG_DIR / '_config' / 'behavioral_experiments_settings.toml') as f:
        return _toml.load(f)


def test_modify_audio_file_writes_destination(monkeypatch, tmp_path, mocker):
    """modify_audio_file should produce a valid avisoft_config.ini at the
    destination directory without crashing on the realistic settings dict."""
    fake_module_dir = _seed_avisoft_config(tmp_path)
    monkeypatch.setattr(be_mod, '__file__', str(fake_module_dir / 'behavioral_experiments.py'))

    dest_dir = tmp_path / 'avisoft_config_dst'
    dest_dir.mkdir()

    settings = _realistic_audio_settings()
    settings['avisoft_basedirectory'] = str(tmp_path / 'avisoft_base') + os.sep
    (tmp_path / 'avisoft_base').mkdir()
    settings['avisoft_config_directory'] = str(dest_dir)
    settings['video_session_duration'] = 5

    mocker.patch('usv_playpen.behavioral_experiments.smart_wait')
    ec = ExperimentController(exp_settings_dict=settings)
    ec.modify_audio_file()

    # Destination .ini was written and is parseable.
    out_ini = dest_dir / 'avisoft_config.ini'
    assert out_ini.is_file()
    parsed = configparser.ConfigParser()
    parsed.read(out_ini)
    assert 'Configuration' in parsed.sections()
    # basedirectory key should reflect the settings we passed in.
    assert parsed['Configuration']['basedirectory'].startswith(str(tmp_path / 'avisoft_base'))


def test_modify_audio_file_idempotent_on_second_run(monkeypatch, tmp_path, mocker):
    """Running modify_audio_file twice with identical settings should result
    in no further changes on the second run (changes counter stays at 0)."""
    fake_module_dir = _seed_avisoft_config(tmp_path)
    monkeypatch.setattr(be_mod, '__file__', str(fake_module_dir / 'behavioral_experiments.py'))

    dest_dir = tmp_path / 'avisoft_config_dst'
    dest_dir.mkdir()

    settings = _realistic_audio_settings()
    settings['avisoft_basedirectory'] = str(tmp_path / 'avisoft_base') + os.sep
    (tmp_path / 'avisoft_base').mkdir()
    settings['avisoft_config_directory'] = str(dest_dir)
    settings['video_session_duration'] = 5

    mocker.patch('usv_playpen.behavioral_experiments.smart_wait')

    out_messages = []
    ec = ExperimentController(exp_settings_dict=settings, message_output=out_messages.append)
    ec.modify_audio_file()
    out_messages.clear()
    ec.modify_audio_file()
    # On the second pass the message "N lines changed" should NOT be emitted
    # because the file was already in the desired state.
    assert not any('lines changed' in m for m in out_messages)


# ---- conduct_behavioral_recording (full orchestration) --------------------


@pytest.fixture
def full_recording_settings(tmp_path):
    """A settings dict rich enough for conduct_behavioral_recording to run
    end-to-end with every subordinate method mocked.

    All directories live under tmp_path so a stray un-mocked path access cannot
    touch the user's filesystem."""
    base = tmp_path / 'avisoft_base'
    coolterm_dir = tmp_path / 'coolterm'
    (coolterm_dir / 'Connection_settings').mkdir(parents=True)
    (coolterm_dir / 'Data').mkdir()
    # Provide a minimal CoolTerm .stc file so the rewrite pass has something
    # to read; conduct_behavioral_recording will rewrite the 'Port = ...' line.
    (coolterm_dir / 'Connection_settings' / 'coolterm_config.stc').write_text(
        "Port = COM1\n"
        "Baudrate = 115200\n"
    )
    base.mkdir()
    creds_dir = tmp_path / 'creds'
    creds_dir.mkdir()
    # Build a minimal email_config.ini so the Messenger constructor (which
    # ultimately reads this file) is satisfied.
    cfg = configparser.ConfigParser()
    cfg['email'] = {
        'email_host': 'smtp.example.com',
        'email_port': '465',
        'email_address': 'a@example.com',
        'email_password': 'pw',
    }
    with open(creds_dir / 'email_config.ini', 'w') as f:
        cfg.write(f)
    win_dir = tmp_path / 'win_dest'
    win_dir.mkdir()
    return {
        'experimenter': 'TestBot',
        'avisoft_basedirectory': str(base),
        'avisoft_config_directory': str(tmp_path / 'avisoft_cfg'),
        'avisoft_recorder_program_name': 'rec_usgh.exe',
        'avisoft_recorder_exe_directory': str(tmp_path),
        'coolterm_basedirectory': str(coolterm_dir),
        'arduino_sync_port': 'COM7',
        'credentials_directory': str(creds_dir),
        'recording_files_destination_linux': ['/lin/dest'],
        'recording_files_destination_win': [str(win_dir)],
        'video_session_duration': 0.05,
        'conduct_audio_recording': False,
        'disable_ethernet': False,
        'ethernet_network': 'Eth0',
        'audio': {
            'cpu_affinity': [],
            'cpu_priority': '',
            'used_mics': [0],
            'total_mic_number': 24,
            'total_device_num': 2,
            'general': {'total': 0},
            'devices': {'usghflags': 1572},
        },
        'video': {
            'general': {
                'expected_cameras': ['SN12345'],
                'recording_codec': 'H264',
                'recording_frame_rate': 150,
                'delete_post_copy': True,
                'monitor_recording': False,
            },
        },
    }


def test_conduct_behavioral_recording_orchestrates_calls(mocker, monkeypatch,
                                                          full_recording_settings):
    """End-to-end smoke: every subordinate ExperimentController method and
    every external dependency is mocked. We verify the orchestration order /
    counts and that the function returns the metadata it was given."""
    # Patch every subordinate method so no real subprocess / network call
    # fires.
    for method_name in (
        'check_ethernet_connection',
        'purge_cup_connections_on_windows',
        'check_camera_vitals',
        'modify_audio_file',
        'verify_avisoft_is_recording',
        'remount_cup_drives_on_windows',
    ):
        monkeypatch.setattr(ExperimentController, method_name,
                            lambda self, *a, **kw: True if method_name == 'verify_avisoft_is_recording' else None)

    # Fake Motif API: schedule call returns a 'now' float, recording/start
    # accepts duration/codec, copy_all is a no-op, is_copying always False
    # so the post-recording wait loop falls through immediately.
    fake_api = MagicMock()
    fake_api.call.side_effect = lambda endpoint, **_kw: {
        'schedule': {'now': 1_700_000_000.0},
    }.get(endpoint, MagicMock())
    fake_api.is_copying.return_value = False

    # check_camera_vitals is patched above to a no-op; it would normally set
    # self.api and self.camera_serial_num. Provide them via a wrapper.
    def _vitals(self, camera_fr=None):
        self.api = fake_api
        self.camera_serial_num = ['SN12345']
    monkeypatch.setattr(ExperimentController, 'check_camera_vitals', _vitals)

    mocker.patch('usv_playpen.behavioral_experiments.subprocess.Popen')
    mocker.patch('usv_playpen.behavioral_experiments.smart_wait')
    msg_mock = mocker.patch('usv_playpen.behavioral_experiments.Messenger')

    ec = ExperimentController(
        exp_settings_dict=full_recording_settings,
        metadata_settings={'Session': {}, 'Environment': {}},
    )
    out = ec.conduct_behavioral_recording()

    # Two messenger sends: one before recording starts ("PC busy") and one
    # at the end ("recording completed").
    assert msg_mock.return_value.send_message.call_count == 2
    # Must return the metadata dict (with session_id stamped onto it).
    assert isinstance(out, dict)
    assert 'session_id' in out['Session']


def test_conduct_behavioral_recording_aborts_if_avisoft_silent(mocker, monkeypatch,
                                                                full_recording_settings):
    """If verify_avisoft_is_recording returns False we must SystemExit and
    NOT call api.call('recording/start'). conduct_audio_recording is True for
    this test so the verification gate is exercised."""
    full_recording_settings['conduct_audio_recording'] = True

    for method_name in (
        'check_ethernet_connection',
        'purge_cup_connections_on_windows',
        'modify_audio_file',
    ):
        monkeypatch.setattr(ExperimentController, method_name, lambda self, *a, **kw: None)

    monkeypatch.setattr(ExperimentController, 'verify_avisoft_is_recording',
                        lambda self, *a, **kw: False)

    fake_api = MagicMock()
    fake_api.call.return_value = {'now': 1.0}
    def _vitals(self, camera_fr=None):
        self.api = fake_api
        self.camera_serial_num = ['SN1']
    monkeypatch.setattr(ExperimentController, 'check_camera_vitals', _vitals)

    mocker.patch('usv_playpen.behavioral_experiments.subprocess.Popen')
    mocker.patch('usv_playpen.behavioral_experiments.smart_wait')
    mocker.patch('usv_playpen.behavioral_experiments.Messenger')

    ec = ExperimentController(exp_settings_dict=full_recording_settings)
    with pytest.raises(SystemExit):
        ec.conduct_behavioral_recording()
    # 'recording/start' must never have been called — the abort happened first.
    rec_starts = [c for c in fake_api.call.call_args_list
                  if c.args and c.args[0] in ('recording/start',
                                              'camera/SN1/recording/start')]
    assert rec_starts == []


# ---- CLI wrappers ---------------------------------------------------------


def test_conduct_calibration_cli_invokes_controller(mocker):
    """conduct-calibration CLI: ExperimentController(...).conduct_tracking_calibration()
    is invoked, and --set overrides flow through override_toml_values."""
    fake_ctrl = MagicMock()
    fake_ctrl_cls = mocker.patch(
        'usv_playpen.behavioral_experiments.ExperimentController',
        return_value=fake_ctrl,
    )
    fake_override = mocker.patch(
        'usv_playpen.behavioral_experiments.override_toml_values',
        side_effect=lambda overrides, exp_settings_dict: exp_settings_dict,
    )

    runner = CliRunner()
    result = runner.invoke(conduct_calibration_cli, ['--set', 'calibration_duration=2'])

    assert result.exit_code == 0, result.output
    fake_ctrl.conduct_tracking_calibration.assert_called_once()
    fake_override.assert_called_once()
    # The first positional arg to the controller is exp_settings_dict — verify
    # it's a dict (toml.load fed it).
    assert isinstance(fake_ctrl_cls.call_args.kwargs['exp_settings_dict'], dict)


def test_conduct_recording_cli_invokes_controller_and_dumps_metadata(mocker, tmp_path):
    """conduct-recording CLI: builds an ExperimentController, runs the
    recording, and dumps the returned metadata back to _config/_metadata.yaml.

    We patch yaml.dump rather than writing the real file — otherwise this
    test would mutate a tracked package file every run."""
    returned_metadata = {'Session': {'session_id': '20260101_000000'}}
    fake_ctrl = MagicMock()
    fake_ctrl.conduct_behavioral_recording.return_value = returned_metadata
    mocker.patch(
        'usv_playpen.behavioral_experiments.ExperimentController',
        return_value=fake_ctrl,
    )
    yaml_dump = mocker.patch('usv_playpen.behavioral_experiments.yaml.dump')

    runner = CliRunner()
    result = runner.invoke(conduct_recording_cli, [])

    assert result.exit_code == 0, result.output
    fake_ctrl.conduct_behavioral_recording.assert_called_once()
    # yaml.dump was called with our updated metadata.
    yaml_dump.assert_called_once()
    assert yaml_dump.call_args.args[0] is returned_metadata


def test_conduct_recording_cli_skips_dump_when_no_metadata(mocker):
    """If conduct_behavioral_recording returns falsy, the metadata file is
    not rewritten."""
    fake_ctrl = MagicMock()
    fake_ctrl.conduct_behavioral_recording.return_value = None
    mocker.patch(
        'usv_playpen.behavioral_experiments.ExperimentController',
        return_value=fake_ctrl,
    )
    yaml_dump = mocker.patch('usv_playpen.behavioral_experiments.yaml.dump')

    runner = CliRunner()
    result = runner.invoke(conduct_recording_cli, [])

    assert result.exit_code == 0, result.output
    yaml_dump.assert_not_called()
