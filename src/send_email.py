"""
@author: bartulem
Code to send e-mail notifying users the PC is busy.
"""

import configparser
import os
import smtplib
import sys
from email.message import EmailMessage


class Messenger:

    def __init__(self, receivers=None, exp_settings_dict=None,
                 message_output=None, no_receivers_notification=True):
        if receivers is None:
            self.receivers = ['']
        else:
            self.receivers = receivers

        if exp_settings_dict is None:
            self.exp_settings_dict = None
        else:
            self.exp_settings_dict = exp_settings_dict

        if message_output is None:
            self.message_output = print
        else:
            self.message_output = message_output

        self.no_receivers_notification = no_receivers_notification

    def get_email_params(self):
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
        email_address (str)
            Lab e-mail address.
        email_password (str)
            Lab e-mail password.
        ----------
        """

        config = configparser.ConfigParser()

        if not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), '_config/email_config.ini')):
            self.message_output("E-mail config file not found. Try again!")
            sys.exit()
        else:
            config.read(os.path.join(os.path.dirname(os.path.abspath(__file__)), '_config/email_config.ini'))
            return config['email']['email_address'], config['email']['email_password']

    def send_message(self, subject, message):
        """
        Description
        ----------
        This method sends e-mails about 165B PC usage.
        ----------

        Parameters
        ----------
        receivers
            List of e-mail addresses to notify.
        subject
            E-mail subject field.
        message
            Text to send.
        ----------

        Returns
        ----------
        ----------
        """

        if len(self.receivers) > 0:
            try:
                email_address, email_password = self.get_email_params()

                if email_address is None or email_password is None:
                    # no email address or password
                    self.message_output("Did you set the e-mail address and password correctly?")
                    sys.exit()

                # create email
                msg = EmailMessage()
                msg['Subject'] = subject
                msg['From'] = email_address
                msg['To'] = ', '.join(self.receivers)
                msg.set_content(message)

                # send email
                with smtplib.SMTP_SSL(host='smtp.gmail.com', port=465) as smtp:
                    smtp.login(email_address, email_password)
                    smtp.send_message(msg)
                return True
            except Exception as e:
                self.message_output("Error occurred during an attempt to send e-mail.")
                self.message_output(str(e))
        else:
            if self.no_receivers_notification:
                self.message_output("You chose not to notify anyone via e-mail about PC usage.")
