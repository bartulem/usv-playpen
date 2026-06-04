"""
Tests for usv_playpen.send_email.Messenger.send_message's documented
True/False/None contract -- in particular that a missing credentials file is now
reported and returns False (via get_email_params raising FileNotFoundError,
which send_message catches) instead of calling sys.exit and killing the process.
"""

import pytest

from usv_playpen import send_email
from usv_playpen.send_email import Messenger


@pytest.fixture
def creds_file(tmp_path):
    p = tmp_path / "email_config.ini"
    p.write_text(
        "[email]\n"
        "email_host = smtp.test\n"
        "email_port = 465\n"
        "email_address = test@lab.org\n"
        "email_password = secret\n"
    )
    return str(p)


class FakeSMTP:
    """Context-manager SMTP_SSL stand-in that records the sent message."""
    last_msg = None

    def __init__(self, host=None, port=None):
        self.host = host
        self.port = port

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def login(self, address, password):
        pass

    def send_message(self, msg):
        FakeSMTP.last_msg = msg


def test_send_message_no_receivers_returns_none():
    msgr = Messenger(receivers=[], message_output=lambda *_: None)
    assert msgr.send_message(subject="s", message="m") is None


def test_send_message_missing_creds_returns_false_and_logs(tmp_path):
    logs = []
    msgr = Messenger(receivers=["a@b.org"], message_output=logs.append,
                     credentials_file=str(tmp_path / "absent.ini"))
    assert msgr.send_message(subject="s", message="m") is False
    assert any("FileNotFoundError" in m for m in logs)


def test_send_message_happy_path_returns_true(monkeypatch, creds_file):
    monkeypatch.setattr(send_email.smtplib, "SMTP_SSL", FakeSMTP)
    msgr = Messenger(receivers=["a@b.org"], message_output=lambda *_: None,
                     credentials_file=creds_file)
    assert msgr.send_message(subject="hello", message="body") is True
    assert FakeSMTP.last_msg["Subject"] == "hello"


def test_send_message_smtp_error_returns_false_and_logs_host_port(monkeypatch, creds_file):
    def boom(*args, **kwargs):
        raise ConnectionError("SMTP down")
    monkeypatch.setattr(send_email.smtplib, "SMTP_SSL", boom)
    logs = []
    msgr = Messenger(receivers=["a@b.org"], message_output=logs.append,
                     credentials_file=creds_file)
    assert msgr.send_message(subject="s", message="m") is False
    assert any("smtp.test:465" in m for m in logs)


def test_get_email_params_raises_on_missing_file(tmp_path):
    msgr = Messenger(credentials_file=str(tmp_path / "nope.ini"))
    with pytest.raises(FileNotFoundError):
        msgr.get_email_params()
