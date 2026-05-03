"""
@author: bartulem
Send e-mail notifying users the PC is busy.
"""

from __future__ import annotations

import configparser
import smtplib
import sys
from collections.abc import Callable
from email.message import EmailMessage
from pathlib import Path


class Messenger:
    def __init__(
        self,
        receivers: list | None = None,
        exp_settings_dict: dict | None = None,
        message_output: Callable | None = None,
        no_receivers_notification: bool = True,
        credentials_file: str = ''
    ) -> None:
        """
        Initializes the Messenger class.

        Parameters
        receivers (list)
            Root directories for data; defaults to None.
        exp_settings_dict (dict)
            Processing parameters; defaults to None.
        message_output (function)
            Defines output messages; defaults to None.
        no_receivers_notification (bool)
            Notify if no receivers are set; defaults to True.
        credentials_file (str)
            Path to credentials file.

        Returns
        """

        self.credentials_file = credentials_file or ''
        self.receivers = receivers if receivers is not None else []
        self.exp_settings_dict = exp_settings_dict
        self.message_output = message_output or print
        self.no_receivers_notification = no_receivers_notification

    def get_email_params(self) -> tuple:
        """
        Description
        This method gets the lab e-mail address and password to send a message.

        Parameters

        Returns
        email_host (str), email_port (str), email_address (str), email_password (str)
            Lab e-mail address and password.
        """

        config = configparser.ConfigParser()

        if not Path(self.credentials_file).is_file():
            print(self.credentials_file)  # noqa: T201
            print("E-mail config file not found. Try again!")  # noqa: T201
            sys.exit(1)
        else:
            config.read(self.credentials_file)
            return config['email']['email_host'], config['email']['email_port'], config['email']['email_address'], config['email']['email_password']

    def send_message(self, subject: str | None = None, message: str | None = None) -> bool | None:
        """
        Description
        This method sends e-mails about 165B PC usage.

        Failure reporting:
        * Returns True when the message was handed off to the SMTP server.
        * Returns False when an error occurred while attempting delivery; the
          exception class, exception message, and SMTP host/port are logged to
          self.message_output so the failure is visible in the GUI console and
          in any log that mirrors it. This distinguishes a real delivery
          failure from the 'no receivers configured' no-op, which returns None.
        * Returns None when the receiver list is empty (no-op).

        Parameters
        subject (str)
            E-mail subject field.
        message (str)
            Text to send.

        Returns
        outcome (bool or None)
            True on successful handoff, False on delivery error, None when
            there are no receivers configured.
        """

        if len(self.receivers) > 0:
            email_host = None
            email_port = None
            try:
                email_host, email_port, email_address, email_password = self.get_email_params()

                if email_address is None or email_password is None:
                    # no email address or password
                    self.message_output(
                        "Did you set the e-mail address and password correctly?"
                    )
                    sys.exit()

                # create email
                msg = EmailMessage()
                msg["Subject"] = subject
                msg["From"] = email_address
                msg["To"] = ", ".join(self.receivers)
                msg.set_content(message)

                # send email
                with smtplib.SMTP_SSL(host=email_host, port=email_port) as smtp:
                    smtp.login(email_address, email_password)
                    smtp.send_message(msg)
                return True
            except Exception as e:
                # Surface the concrete failure mode (auth, DNS, TLS handshake,
                # connection reset, etc.) so a silent mail outage does not go
                # unnoticed for days. The exception class name plus the SMTP
                # host:port gives the on-call enough to diagnose.
                host_info = f"{email_host}:{email_port}" if email_host else "<unresolved host>"
                self.message_output(
                    f"Error occurred during an attempt to send e-mail via {host_info}: "
                    f"{type(e).__name__}: {e}"
                )
                return False
        elif self.no_receivers_notification:
            self.message_output(
                "You chose not to notify anyone via e-mail about PC usage."
            )

        return None
