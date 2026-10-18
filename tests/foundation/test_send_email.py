"""
@author: bartulem
Tests for usv_playpen.send_email.Messenger.send_message's documented
True/False/None contract -- in particular that a missing credentials file is now
reported and returns False (via get_email_params raising FileNotFoundError,
which send_message catches) instead of calling sys.exit and killing the process.

Coverage: send_message with no receivers (None), a missing credentials file
(False + logged), a successful SMTP send (True), an SMTP connection error
(False + host:port logged), and get_email_params raising on a missing file.
"""

from __future__ import annotations

import pytest

from usv_playpen import send_email
from usv_playpen.send_email import Messenger


@pytest.fixture
def creds_file(tmp_path):
    """
    Description
    -----------
    Writes a minimal, valid email credentials INI file so send_message can
    reach the SMTP stage without touching the real lab credentials.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Pytest temporary directory the INI file is written into.

    Returns
    -------
    creds_path (str)
        Path to the written email_config.ini (host smtp.test, port 465).
    """
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
        """Store the host and port the code under test connects to."""
        self.host = host
        self.port = port

    def __enter__(self):
        """Return the fake itself, mirroring `with smtplib.SMTP_SSL(...) as server`."""
        return self

    def __exit__(self, *exc):
        """Do not swallow exceptions raised inside the with-block."""
        return False

    def login(self, address, password):
        """Accept any credentials; authentication is not under test."""

    def send_message(self, msg):
        """Record the outgoing message on the class so the test can inspect it."""
        FakeSMTP.last_msg = msg


def test_send_message_no_receivers_returns_none():
    """With no receivers there is nothing to send; None (not False) tells callers nothing was attempted."""
    msgr = Messenger(receivers=[], message_output=lambda *_: None)
    assert msgr.send_message(subject="s", message="m") is None


def test_send_message_missing_creds_returns_false_and_logs(tmp_path):
    """A missing credentials file must be reported and return False instead of calling sys.exit mid-pipeline."""
    logs = []
    msgr = Messenger(receivers=["a@b.org"], message_output=logs.append,
                     credentials_file=str(tmp_path / "absent.ini"))
    assert msgr.send_message(subject="s", message="m") is False
    assert any("FileNotFoundError" in m for m in logs)


def test_send_message_happy_path_returns_true(monkeypatch, creds_file):
    """A successful send returns True and the message reaches SMTP with the requested subject."""
    monkeypatch.setattr(send_email.smtplib, "SMTP_SSL", FakeSMTP)
    msgr = Messenger(receivers=["a@b.org"], message_output=lambda *_: None,
                     credentials_file=creds_file)
    assert msgr.send_message(subject="hello", message="body") is True
    assert FakeSMTP.last_msg["Subject"] == "hello"


def test_send_message_smtp_error_returns_false_and_logs_host_port(monkeypatch, creds_file):
    """An SMTP connection failure returns False and logs host:port, so a bad server config is diagnosable."""
    def boom(*_args, **_kwargs):
        """Stand in for SMTP_SSL and fail the connection immediately."""
        error_message = "SMTP down"
        raise ConnectionError(error_message)
    monkeypatch.setattr(send_email.smtplib, "SMTP_SSL", boom)
    logs = []
    msgr = Messenger(receivers=["a@b.org"], message_output=logs.append,
                     credentials_file=creds_file)
    assert msgr.send_message(subject="s", message="m") is False
    assert any("smtp.test:465" in m for m in logs)


def test_get_email_params_raises_on_missing_file(tmp_path):
    """get_email_params must raise FileNotFoundError so send_message can catch it and report False."""
    msgr = Messenger(credentials_file=str(tmp_path / "nope.ini"))
    with pytest.raises(FileNotFoundError):
        msgr.get_email_params()
