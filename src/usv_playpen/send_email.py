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

        Parameter
        ---------
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
        -------
        -------
        """

        self.credentials_file = credentials_file or ''
        self.receivers = receivers if receivers is not None else []
        self.exp_settings_dict = exp_settings_dict
        self.message_output = message_output or print
        self.no_receivers_notification = no_receivers_notification

    def get_email_params(self) -> tuple:
        """
        Description
        ----------
        This method gets the lab e-mail address and password to send a message.
        ----------

        Parameters
        ----------
        ----------

        Returns
        ----------
        email_host (str), email_port (str), email_address (str), email_password (str)
            Lab e-mail address and password.
        ----------
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
        ----------
        This method sends e-mails about 165B PC usage.
        ----------

        Parameters
        ----------
        subject (str)
            E-mail subject field.
        message (str)
            Text to send.
        ----------

        Returns
        ----------
        ----------
        """

        if len(self.receivers) > 0:
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
                self.message_output("Error occurred during an attempt to send e-mail.")
                self.message_output(str(e))
        elif self.no_receivers_notification:
            self.message_output(
                "You chose not to notify anyone via e-mail about PC usage."
            )

        return None
