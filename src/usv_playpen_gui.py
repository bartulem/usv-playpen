"""
@author: bartulem
GUI to run behavioral experiments, data processing and analyses.
"""

import ast
import ctypes
import json
import os
import platform
import sys
from functools import partial
from pathlib import Path
import time
import toml
from PyQt6.QtCore import (
    Qt
)
from PyQt6.QtGui import (
    QFont,
    QFontDatabase,
    QGuiApplication,
    QIcon,
    QPainter,
    QPixmap
)
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSlider,
    QSplashScreen,
    QTextEdit,
    QWidget,
)
from .behavioral_experiments import ExperimentController
from .preprocess_data import Stylist

os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"

if os.name == 'nt':
    my_app_id = 'mycompany.myproduct.subproduct.version'
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(my_app_id)

app_name = 'USV Playpen v0.7.6'

basedir = os.path.dirname(__file__)
background_img = f'{basedir}{os.sep}img{os.sep}background_img.png'
lab_icon = f'{basedir}{os.sep}img{os.sep}lab.png'
splash_icon = f'{basedir}{os.sep}img{os.sep}uncle_stefan.png'
process_icon = f'{basedir}{os.sep}img{os.sep}process.png'
record_icon = f'{basedir}{os.sep}img{os.sep}record.png'
analyze_icon = f'{basedir}{os.sep}img{os.sep}analyze.png'
visualize_icon = f'{basedir}{os.sep}img{os.sep}plot.png'
previous_icon = f'{basedir}{os.sep}img{os.sep}previous.png'
next_icon = f'{basedir}{os.sep}img{os.sep}next.png'
main_icon = f'{basedir}{os.sep}img{os.sep}main.png'
calibrate_icon = f'{basedir}{os.sep}img{os.sep}calibrate.png'


class Main(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

    def paintEvent(self, event):
        paint_main = QPainter(self)
        paint_main.drawPixmap(self.rect(), QPixmap(f'{background_img}'))
        QWidget.paintEvent(self, event)


class Record(QWidget):
    def __init__(self, parent=Main):
        super(Record, self).__init__(parent)


class AudioSettings(QWidget):
    def __init__(self, parent=Main):
        super(AudioSettings, self).__init__(parent)


class VideoSettings(QWidget):
    def __init__(self, parent=Main):
        super(VideoSettings, self).__init__(parent)


class ConductRecording(QWidget):
    def __init__(self, parent=Main):
        super(ConductRecording, self).__init__(parent)


class ProcessSettings(QWidget):
    def __init__(self, parent=Main):
        super(ProcessSettings, self).__init__(parent)


class ConductProcess(QWidget):
    def __init__(self, parent=Main):
        super(ConductProcess, self).__init__(parent)


class USVPlaypenWindow(QMainWindow):
    """Main window of GUI."""

    def __init__(self, **kwargs):
        super().__init__()

        font_file_loc = QFontDatabase.addApplicationFont(f'{basedir}{os.sep}fonts{os.sep}segoeui.ttf')
        self.font_id = QFontDatabase.applicationFontFamilies(font_file_loc)[0]

        for attr, value in kwargs.items():
            setattr(self, attr, value)

        self.boolean_list = ['Yes', 'No']

        self.exp_settings_dict = toml.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), '_config/behavioral_experiments_settings.toml'))

        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '_parameter_settings/processing_settings.json'), 'r') as process_json_file:
            self.processing_input_dict = json.load(process_json_file)

        self.main_window()

    def main_window(self):
        self.Main = Main(self)
        self.setCentralWidget(self.Main)
        self.setFixedSize(420, 500)
        self._location_on_the_screen()
        self.setWindowFlag(Qt.WindowType.WindowMaximizeButtonHint, on=False)
        self.setWindowTitle(f'{app_name}')

        exp_id_label = QLabel('Experimenter:', self.Main)
        exp_id_label.setFont(QFont(self.font_id, 10))
        exp_id_label.setStyleSheet('QLabel { font-weight: bold;}')
        exp_id_label.move(120, 329)
        self.exp_id_list = sorted(self.exp_settings_dict['experimenter_list'], key=lambda x: x == self.exp_id, reverse=True)
        self.exp_id_cb = QComboBox(self.Main)
        self.exp_id_cb.addItems(self.exp_id_list)
        self.exp_id_cb.setStyleSheet('QComboBox { width: 60px; height: 24px}')
        self.exp_id_cb.activated.connect(partial(self._combo_box_prior_name, variable_id='exp_id'))
        self.exp_id_cb.move(215, 325)

        self._create_buttons_main()

    def record_one(self):
        self.Record = Record(self)
        self.setWindowTitle(f'{app_name} (Record > Select config directories and set basic parameters)')
        self.setCentralWidget(self.Record)
        record_one_x, record_one_y = (725, 510)
        self.setFixedSize(record_one_x, record_one_y)

        title_label = QLabel('Please select appropriate directories (with config files or executables in them)', self.Record)
        title_label.setFont(QFont(self.font_id, 13))
        title_label.setStyleSheet('QLabel { font-weight: bold;}')
        title_label.move(5, 10)

        avisoft_exe_dir_label = QLabel('Avisoft Recorder directory:', self.Record)
        avisoft_exe_dir_label.setFont(QFont(self.font_id, 12))
        avisoft_exe_dir_label.move(5, 40)
        self.recorder_settings_edit = QLineEdit(self.avisoft_rec_dir_global, self.Record)
        self.recorder_settings_edit.setFont(QFont(self.font_id, 10))
        self.recorder_settings_edit.setStyleSheet('QLineEdit { width: 400px; }')
        self.recorder_settings_edit.move(220, 40)
        recorder_dir_btn = QPushButton('Browse', self.Record)
        recorder_dir_btn.setFont(QFont(self.font_id, 8))
        recorder_dir_btn.move(625, 40)
        recorder_dir_btn.setStyleSheet('QPushButton { min-width: 64px; min-height: 12px; max-width: 64px; max-height: 13px; }')
        self.recorder_dir_btn_clicked_flag = False
        recorder_dir_btn.clicked.connect(self._open_recorder_dialog)

        avisoft_base_dir_label = QLabel('Avisoft base directory:', self.Record)
        avisoft_base_dir_label.setFont(QFont(self.font_id, 12))
        avisoft_base_dir_label.move(5, 70)
        self.avisoft_base_edit = QLineEdit(self.avisoft_base_dir_global, self.Record)
        self.avisoft_base_edit.setFont(QFont(self.font_id, 10))
        self.avisoft_base_edit.setStyleSheet('QLineEdit { width: 400px; }')
        self.avisoft_base_edit.move(220, 70)
        avisoft_base_dir_btn = QPushButton('Browse', self.Record)
        avisoft_base_dir_btn.setFont(QFont(self.font_id, 8))
        avisoft_base_dir_btn.move(625, 70)
        avisoft_base_dir_btn.setStyleSheet('QPushButton { min-width: 64px; min-height: 12px; max-width: 64px; max-height: 13px; }')
        self.avisoft_base_dir_btn_clicked_flag = False
        avisoft_base_dir_btn.clicked.connect(self._open_avisoft_dialog)

        coolterm_base_dir_label = QLabel('CoolTerm base directory:', self.Record)
        coolterm_base_dir_label.setFont(QFont(self.font_id, 12))
        coolterm_base_dir_label.move(5, 100)
        self.coolterm_base_edit = QLineEdit(self.coolterm_base_dir_global, self.Record)
        self.coolterm_base_edit.setFont(QFont(self.font_id, 10))
        self.coolterm_base_edit.setStyleSheet('QLineEdit { width: 400px; }')
        self.coolterm_base_edit.move(220, 100)
        coolterm_base_dir_btn = QPushButton('Browse', self.Record)
        coolterm_base_dir_btn.setFont(QFont(self.font_id, 8))
        coolterm_base_dir_btn.move(625, 100)
        coolterm_base_dir_btn.setStyleSheet('QPushButton { min-width: 64px; min-height: 12px; max-width: 64px; max-height: 13px; }')
        self.coolterm_base_dir_btn_clicked_flag = False
        coolterm_base_dir_btn.clicked.connect(self._open_coolterm_dialog)

        # recording files destination directories (across OS)
        recording_files_destination_linux_label = QLabel('File destination(s) Linux:', self.Record)
        recording_files_destination_linux_label.setFont(QFont(self.font_id, 12))
        recording_files_destination_linux_label.move(5, 130)
        self.recording_files_destination_linux = QLineEdit(self.destination_linux_global, self.Record)
        self.recording_files_destination_linux.setFont(QFont(self.font_id, 10))
        self.recording_files_destination_linux.setStyleSheet('QLineEdit { width: 490px; }')
        self.recording_files_destination_linux.move(220, 130)

        recording_files_destination_windows_label = QLabel('File destination(s) Windows:', self.Record)
        recording_files_destination_windows_label.setFont(QFont(self.font_id, 12))
        recording_files_destination_windows_label.move(5, 160)
        self.recording_files_destination_windows = QLineEdit(self.destination_win_global, self.Record)
        self.recording_files_destination_windows.setFont(QFont(self.font_id, 10))
        self.recording_files_destination_windows.setStyleSheet('QLineEdit { width: 490px; }')
        self.recording_files_destination_windows.move(220, 160)

        # set main recording parameters
        parameters_label = QLabel('Please set main recording parameters', self.Record)
        parameters_label.setFont(QFont(self.font_id, 13))
        parameters_label.setStyleSheet('QLabel { font-weight: bold;}')
        parameters_label.move(5, 200)

        conduct_audio_label = QLabel('Conduct AUDIO recording:', self.Record)
        conduct_audio_label.setFont(QFont(self.font_id, 11))
        conduct_audio_label.setStyleSheet('QLabel { color: #F58025; font-weight: bold;}')
        conduct_audio_label.move(5, 235)
        self.conduct_audio_cb_list = [x for _, x in sorted(zip([self.exp_settings_dict['conduct_audio_recording'], not self.exp_settings_dict['conduct_audio_recording']], self.boolean_list), reverse=True)]
        self.conduct_audio_cb = QComboBox(self.Record)
        self.conduct_audio_cb.addItems(self.conduct_audio_cb_list)
        self.conduct_audio_cb.setStyleSheet('QComboBox { width: 465px; }')
        self.conduct_audio_cb.activated.connect(partial(self._combo_box_prior_true if self.conduct_audio_cb_list[0] == 'Yes' else self._combo_box_prior_false, variable_id='conduct_audio_cb_bool'))
        self.conduct_audio_cb.move(220, 235)

        conduct_tracking_cal_label = QLabel('Conduct VIDEO calibration:', self.Record)
        conduct_tracking_cal_label.setFont(QFont(self.font_id, 11))
        conduct_tracking_cal_label.setStyleSheet('QLabel { color: #F58025; font-weight: bold;}')
        conduct_tracking_cal_label.move(5, 265)
        self.conduct_tracking_calibration_cb_list = [x for _, x in sorted(zip([self.exp_settings_dict['conduct_tracking_calibration'], not self.exp_settings_dict['conduct_tracking_calibration']], self.boolean_list), reverse=True)]
        self.conduct_tracking_calibration_cb = QComboBox(self.Record)
        self.conduct_tracking_calibration_cb.addItems(self.conduct_tracking_calibration_cb_list)
        self.conduct_tracking_calibration_cb.setStyleSheet('QComboBox { width: 465px; }')
        self.conduct_tracking_calibration_cb.activated.connect(partial(self._combo_box_prior_true if self.conduct_tracking_calibration_cb_list[0] == 'Yes' else self._combo_box_prior_false, variable_id='conduct_tracking_calibration_cb_bool'))
        self.conduct_tracking_calibration_cb.move(220, 265)

        disable_ethernet_label = QLabel('Disable ethernet connection:', self.Record)
        disable_ethernet_label.setFont(QFont(self.font_id, 11))
        disable_ethernet_label.setStyleSheet('QLabel { color: #F58025; font-weight: bold;}')
        disable_ethernet_label.move(5, 295)
        self.disable_ethernet_cb_list = [x for _, x in sorted(zip([self.exp_settings_dict['disable_ethernet'], not self.exp_settings_dict['disable_ethernet']], self.boolean_list), reverse=True)]
        self.disable_ethernet_cb = QComboBox(self.Record)
        self.disable_ethernet_cb.addItems(self.disable_ethernet_cb_list)
        self.disable_ethernet_cb.setStyleSheet('QComboBox { width: 465px; }')
        self.disable_ethernet_cb.activated.connect(partial(self._combo_box_prior_true if self.disable_ethernet_cb_list[0] == 'Yes' else self._combo_box_prior_false, variable_id='disable_ethernet_cb_bool'))
        self.disable_ethernet_cb.move(220, 295)

        video_duration_label = QLabel('Video session duration (min):', self.Record)
        video_duration_label.setFont(QFont(self.font_id, 12))
        video_duration_label.move(5, 325)
        self.video_session_duration = QLineEdit(f"{self.exp_settings_dict['video_session_duration']}", self.Record)
        self.video_session_duration.setFont(QFont(self.font_id, 10))
        self.video_session_duration.setStyleSheet('QLineEdit { width: 490px; }')
        self.video_session_duration.move(220, 325)

        cal_duration_label = QLabel('Calibration duration (min):', self.Record)
        cal_duration_label.setFont(QFont(self.font_id, 12))
        cal_duration_label.move(5, 355)
        self.calibration_session_duration = QLineEdit(f"{self.exp_settings_dict['calibration_duration']}", self.Record)
        self.calibration_session_duration.setFont(QFont(self.font_id, 10))
        self.calibration_session_duration.setStyleSheet('QLineEdit { width: 490px; }')
        self.calibration_session_duration.move(220, 355)

        ethernet_network_label = QLabel('Ethernet network ID:', self.Record)
        ethernet_network_label.setFont(QFont(self.font_id, 12))
        ethernet_network_label.move(5, 385)
        self.ethernet_network = QLineEdit(f"{self.exp_settings_dict['ethernet_network']}", self.Record)
        self.ethernet_network.setFont(QFont(self.font_id, 10))
        self.ethernet_network.setStyleSheet('QLineEdit { width: 490px; }')
        self.ethernet_network.move(220, 385)

        email_notification_label = QLabel('Notify e-mail(s) of PC usage:', self.Record)
        email_notification_label.setFont(QFont(self.font_id, 12))
        email_notification_label.move(5, 415)
        self.email_recipients = QLineEdit('', self.Record)
        self.email_recipients.setFont(QFont(self.font_id, 10))
        self.email_recipients.setStyleSheet('QLineEdit { width: 490px; }')
        self.email_recipients.move(220, 415)

        self._create_buttons_record(seq=0, class_option=self.Record,
                                    button_pos_y=record_one_y-35, next_button_x_pos=record_one_x-100)

    def record_two(self):
        self.AudioSettings = AudioSettings(self)
        self.setWindowTitle(f'{app_name} (Record > Audio Settings)')
        self.setCentralWidget(self.AudioSettings)
        record_two_x, record_two_y = (875, 875)
        self.setFixedSize(record_two_x, record_two_y)

        gas_label = QLabel('General audio recording settings', self.AudioSettings)
        gas_label.setFont(QFont(self.font_id, 13))
        gas_label.setStyleSheet('QLabel { font-weight: bold;}')
        gas_label.move(5, 10)

        # change usghflags to '1862' for NO SYNC audio mode

        self.default_audio_settings = {'name': f"{self.exp_settings_dict['audio']['mics_config']['name']}", 'id': f"{self.exp_settings_dict['audio']['mics_config']['id']}",
                                       'typech': f"{self.exp_settings_dict['audio']['mics_config']['typech']}", 'deviceid': f"{self.exp_settings_dict['audio']['mics_config']['deviceid']}",
                                       'channel': f"{self.exp_settings_dict['audio']['mics_config']['channel']}", 'gain': f"{self.exp_settings_dict['audio']['mics_config']['gain']}",
                                       'fullscalespl': f"{self.exp_settings_dict['audio']['mics_config']['fullscalespl']}", 'triggertype': f"{self.exp_settings_dict['audio']['mics_config']['triggertype']}",
                                       'toggle': f"{self.exp_settings_dict['audio']['mics_config']['toggle']}", 'invert': f"{self.exp_settings_dict['audio']['mics_config']['invert']}",
                                       'ditc': f"{self.exp_settings_dict['audio']['mics_config']['ditc']}", 'ditctime': self.exp_settings_dict['audio']['mics_config']['ditctime'],
                                       'whistletracking': f"{self.exp_settings_dict['audio']['mics_config']['whistletracking']}", 'wtbcf': f"{self.exp_settings_dict['audio']['mics_config']['wtbcf']}",
                                       'wtmaxchange': f"{self.exp_settings_dict['audio']['mics_config']['wtmaxchange']}", 'wtmaxchange2_': f"{self.exp_settings_dict['audio']['mics_config']['wtmaxchange2_']}",
                                       'wtminchange': f"{self.exp_settings_dict['audio']['mics_config']['wtminchange']}", 'wtminchange2_': f"{self.exp_settings_dict['audio']['mics_config']['wtminchange2_']}",
                                       'wtoutside': f"{self.exp_settings_dict['audio']['mics_config']['wtoutside']}", 'hilowratioenable': f"{self.exp_settings_dict['audio']['mics_config']['hilowratioenable']}",
                                       'hilowratio': f"{self.exp_settings_dict['audio']['mics_config']['hilowratio']}", 'hilowratiofc': f"{self.exp_settings_dict['audio']['mics_config']['hilowratiofc']}",
                                       'wtslope': f"{self.exp_settings_dict['audio']['mics_config']['wtslope']}", 'wtlevel': f"{self.exp_settings_dict['audio']['mics_config']['wtlevel']}",
                                       'wtmindurtotal': f"{self.exp_settings_dict['audio']['mics_config']['wtmindurtotal']}", 'wtmindur': f"{self.exp_settings_dict['audio']['mics_config']['wtmindur']}",
                                       'wtmindur2_': f"{self.exp_settings_dict['audio']['mics_config']['wtmindur2_']}", 'wtholdtime': f"{self.exp_settings_dict['audio']['mics_config']['wtholdtime']}",
                                       'wtmonotonic': f"{self.exp_settings_dict['audio']['mics_config']['wtmonotonic']}", 'wtbmaxdur': f"{self.exp_settings_dict['audio']['mics_config']['wtbmaxdur']}",
                                       'rejectwind': f"{self.exp_settings_dict['audio']['mics_config']['rejectwind']}", 'rwentropy': f"{self.exp_settings_dict['audio']['mics_config']['rwentropy']}",
                                       'rwfpegel': f"{self.exp_settings_dict['audio']['mics_config']['rwfpegel']}", 'rwdutycycle': f"{self.exp_settings_dict['audio']['mics_config']['rwdutycycle']}",
                                       'rwtimeconstant': f"{self.exp_settings_dict['audio']['mics_config']['rwtimeconstant']}", 'rwholdtime': f"{self.exp_settings_dict['audio']['mics_config']['rwholdtime']}",
                                       'fpegel': f"{self.exp_settings_dict['audio']['mics_config']['fpegel']}", 'energy': f"{self.exp_settings_dict['audio']['mics_config']['energy']}",
                                       'frange': f"{self.exp_settings_dict['audio']['mics_config']['frange']}", 'entropyb': f"{self.exp_settings_dict['audio']['mics_config']['entropyb']}",
                                       'entropy': f"{self.exp_settings_dict['audio']['mics_config']['entropy']}", 'increment': f"{self.exp_settings_dict['audio']['mics_config']['increment']}",
                                       'fu': f"{self.exp_settings_dict['audio']['mics_config']['fu']}", 'fo': f"{self.exp_settings_dict['audio']['mics_config']['fo']}",
                                       'pretrigger': f"{self.exp_settings_dict['audio']['mics_config']['pretrigger']}", 'mint': f"{self.exp_settings_dict['audio']['mics_config']['mint']}",
                                       'minst': f"{self.exp_settings_dict['audio']['mics_config']['minst']}", 'fhold': f"{self.exp_settings_dict['audio']['mics_config']['fhold']}",

                                       'logfileno': f"{self.exp_settings_dict['audio']['call']['logfileno']}", 'groupno': f"{self.exp_settings_dict['audio']['call']['groupno']}", 'callno': f"{self.exp_settings_dict['audio']['call']['callno']}",

                                       'timeconstant': f"{self.exp_settings_dict['audio']['monitor']['timeconstant']}", 'timeexpansion': f"{self.exp_settings_dict['audio']['monitor']['timeexpansion']}",
                                       'startstop': f"{self.exp_settings_dict['audio']['monitor']['startstop']}", 'sayf': f"{self.exp_settings_dict['audio']['monitor']['sayf']}",
                                       'over': f"{self.exp_settings_dict['audio']['monitor']['over']}", 'delay': f"{self.exp_settings_dict['audio']['monitor']['delay']}",
                                       'center': f"{self.exp_settings_dict['audio']['monitor']['center']}", 'bandwidth': f"{self.exp_settings_dict['audio']['monitor']['bandwidth']}", 'fd': f"{self.exp_settings_dict['audio']['monitor']['fd']}",
                                       'decimation': f"{self.exp_settings_dict['audio']['monitor']['decimation']}", 'device': f"{self.exp_settings_dict['audio']['monitor']['device']}", 'mode': f"{self.exp_settings_dict['audio']['monitor']['mode']}",

                                       'outfovertaps': f"{self.exp_settings_dict['audio']['devices']['outfovertaps']}", 'outfoverabtast': f"{self.exp_settings_dict['audio']['devices']['outfoverabtast']}",
                                       'outformat': f"{self.exp_settings_dict['audio']['devices']['outformat']}", 'outfabtast': f"{self.exp_settings_dict['audio']['devices']['outfabtast']}",
                                       'outdeviceid': f"{self.exp_settings_dict['audio']['devices']['outdeviceid']}",'outtype': f"{self.exp_settings_dict['audio']['devices']['outtype']}",
                                       'usghflags': f"{self.exp_settings_dict['audio']['devices']['usghflags']}", 'diff': f"{self.exp_settings_dict['audio']['devices']['diff']}",
                                       'format': f"{self.exp_settings_dict['audio']['devices']['format']}", 'type': f"{self.exp_settings_dict['audio']['devices']['type']}",
                                       'nbrwavehdr': f"{self.exp_settings_dict['audio']['devices']['nbrwavehdr']}", 'devbuffer': f"{self.exp_settings_dict['audio']['devices']['devbuffer']}",
                                       'ntaps': f"{self.exp_settings_dict['audio']['devices']['ntaps']}", 'filtercutoff': f"{self.exp_settings_dict['audio']['devices']['filtercutoff']}",
                                       'filter': f"{self.exp_settings_dict['audio']['devices']['filter']}", 'fabtast': f"{self.exp_settings_dict['audio']['devices']['fabtast']}",

                                       'y2': f"{self.exp_settings_dict['audio']['screen_position']['y2']}", 'x2': f"{self.exp_settings_dict['audio']['screen_position']['x2']}",
                                       'y1': f"{self.exp_settings_dict['audio']['screen_position']['y1']}", 'x1': f"{self.exp_settings_dict['audio']['screen_position']['x1']}",

                                       'display': f"{self.exp_settings_dict['audio']['general']['display']}", 'dcolumns': f"{self.exp_settings_dict['audio']['general']['dcolumns']}",
                                       'total': f"{self.exp_settings_dict['audio']['general']['total']}", 'dispspectrogramcontrast': f"{self.exp_settings_dict['audio']['general']['dispspectrogramcontrast']}",
                                       'disprangespectrogram': f"{self.exp_settings_dict['audio']['general']['disprangespectrogram']}", 'disprangeamplitude': f"{self.exp_settings_dict['audio']['general']['disprangeamplitude']}",
                                       'disprangewaveform': f"{self.exp_settings_dict['audio']['general']['disprangewaveform']}", 'fftlength': f"{self.exp_settings_dict['audio']['general']['fftlength']}",
                                       'usvmonitoringflags': f"{self.exp_settings_dict['audio']['general']['usvmonitoringflags']}",

                                       'total_mic_number': f"{self.exp_settings_dict['audio']['total_mic_number']}", 'total_device_num': f"{self.exp_settings_dict['audio']['total_device_num']}",
                                       'used_mics': ','.join([str(x) for x in self.exp_settings_dict['audio']['used_mics']]),
                                       'cpu_priority': self.exp_settings_dict['audio']['cpu_priority'],
                                       'cpu_affinity': ','.join([str(x) for x in self.exp_settings_dict['audio']['cpu_affinity']])}

        row_start_position_label = 5
        row_start_position_line_edit = 120
        row_counter = 0
        column_counter = 0
        for audio_idx, (audio_attr, audio_value) in enumerate(self.default_audio_settings.items()):
            setting_label = QLabel(f'{audio_attr}:', self.AudioSettings)
            setting_label.setFont(QFont(self.font_id, 12))
            setting_label.move(row_start_position_label, 45+(row_counter*30))
            setattr(self, audio_attr, QLineEdit(audio_value, self.AudioSettings))
            getattr(self, audio_attr).setFixedWidth(50)
            getattr(self, audio_attr).setFont(QFont(self.font_id, 10))
            getattr(self, audio_attr).move(row_start_position_line_edit, 45+(row_counter*30))
            if (audio_idx + 1) % 25 == 0:
                column_counter += 1
                row_counter = 0
                row_start_position_label += 200
                if column_counter < 3:
                    row_start_position_line_edit += 200
                else:
                    row_start_position_line_edit += 275
            else:
                row_counter += 1

        self._create_buttons_record(seq=1, class_option=self.AudioSettings,
                                    button_pos_y=record_two_y-35, next_button_x_pos=record_two_x-100)

    def record_three(self):
        self.VideoSettings = VideoSettings(self)
        self.setWindowTitle(f'{app_name} (Record > Video Settings)')
        self.setCentralWidget(self.VideoSettings)
        record_three_x, record_three_y = (980, 900)
        self.setFixedSize(record_three_x, record_three_y)

        gvs_label = QLabel('General video recording settings', self.VideoSettings)
        gvs_label.setFont(QFont(self.font_id, 13))
        gvs_label.setStyleSheet('QLabel { font-weight: bold;}')
        gvs_label.move(5, 10)

        browser_label = QLabel('Browser:', self.VideoSettings)
        browser_label.setFont(QFont(self.font_id, 12))
        browser_label.move(5, 40)
        self.browser = QLineEdit(self.exp_settings_dict['video']['general']['browser'], self.VideoSettings)
        self.browser.setFont(QFont(self.font_id, 10))
        self.browser.setStyleSheet('QLineEdit { width: 300px; }')
        self.browser.move(160, 40)

        use_cam_label = QLabel('Camera(s) to use:', self.VideoSettings)
        use_cam_label.setFont(QFont(self.font_id, 12))
        use_cam_label.move(5, 70)
        self.expected_cameras = QLineEdit(','.join(self.exp_settings_dict['video']['general']['expected_cameras']), self.VideoSettings)
        self.expected_cameras.setFont(QFont(self.font_id, 10))
        self.expected_cameras.setStyleSheet('QLineEdit { width: 300px; }')
        self.expected_cameras.move(160, 70)

        rec_codec_label = QLabel('Recording codec:', self.VideoSettings)
        rec_codec_label.setFont(QFont(self.font_id, 12))
        rec_codec_label.move(5, 100)
        self.recording_codec_list = sorted(['hq', 'mq', 'lq'], key=lambda x: x == self.exp_settings_dict['video']['general']['recording_codec'], reverse=True)
        self.recording_codec_cb = QComboBox(self.VideoSettings)
        self.recording_codec_cb.addItems(self.recording_codec_list)
        self.recording_codec_cb.setStyleSheet('QComboBox { width: 272px; }')
        self.recording_codec_cb.activated.connect(partial(self._combo_box_prior_codec, variable_id='recording_codec'))
        self.recording_codec_cb.move(160, 100)

        monitor_rec_label = QLabel('Monitor recording:', self.VideoSettings)
        monitor_rec_label.setFont(QFont(self.font_id, 11))
        monitor_rec_label.setStyleSheet('QLabel { color: #F58025; font-weight: bold;}')
        monitor_rec_label.move(5, 130)
        self.monitor_recording_cb_list = [x for _, x in sorted(zip([self.exp_settings_dict['video']['general']['monitor_recording'], not self.exp_settings_dict['video']['general']['monitor_recording']], self.boolean_list), reverse=True)]
        self.monitor_recording_cb = QComboBox(self.VideoSettings)
        self.monitor_recording_cb.addItems(self.monitor_recording_cb_list)
        self.monitor_recording_cb.setStyleSheet('QComboBox { width: 272px; }')
        self.monitor_recording_cb.activated.connect(partial(self._combo_box_prior_true if self.monitor_recording_cb_list[0] == 'Yes' else self._combo_box_prior_false, variable_id='monitor_recording_cb_bool'))
        self.monitor_recording_cb.move(160, 130)

        monitor_cam_label = QLabel('Monitor ONE camera:', self.VideoSettings)
        monitor_cam_label.setFont(QFont(self.font_id, 11))
        monitor_cam_label.setStyleSheet('QLabel { color: #F58025; font-weight: bold;}')
        monitor_cam_label.move(5, 160)
        self.monitor_specific_camera_cb_list = [x for _, x in sorted(zip([self.exp_settings_dict['video']['general']['monitor_specific_camera'], not self.exp_settings_dict['video']['general']['monitor_specific_camera']], self.boolean_list), reverse=True)]
        self.monitor_specific_camera_cb = QComboBox(self.VideoSettings)
        self.monitor_specific_camera_cb.addItems(self.monitor_specific_camera_cb_list)
        self.monitor_specific_camera_cb.setStyleSheet('QComboBox { width: 272px; }')
        self.monitor_specific_camera_cb.activated.connect(partial(self._combo_box_prior_true if self.monitor_specific_camera_cb_list[0] == 'Yes' else self._combo_box_prior_false, variable_id='monitor_specific_camera_cb_bool'))
        self.monitor_specific_camera_cb.move(160, 160)

        specific_camera_serial_label = QLabel('ONE camera serial:', self.VideoSettings)
        specific_camera_serial_label.setFont(QFont(self.font_id, 12))
        specific_camera_serial_label.move(5, 190)
        self.specific_camera_serial = QLineEdit(self.exp_settings_dict['video']['general']['specific_camera_serial'], self.VideoSettings)
        self.specific_camera_serial.setFont(QFont(self.font_id, 10))
        self.specific_camera_serial.setStyleSheet('QLineEdit { width: 300px; }')
        self.specific_camera_serial.move(160, 190)

        delete_post_copy_label = QLabel('Delete post copy:', self.VideoSettings)
        delete_post_copy_label.setFont(QFont(self.font_id, 11))
        delete_post_copy_label.setStyleSheet('QLabel { color: #F58025; font-weight: bold;}')
        delete_post_copy_label.move(5, 220)
        self.delete_post_copy_cb_list = [x for _, x in sorted(zip([self.exp_settings_dict['video']['general']['delete_post_copy'], not self.exp_settings_dict['video']['general']['delete_post_copy']], self.boolean_list), reverse=True)]
        self.delete_post_copy_cb = QComboBox(self.VideoSettings)
        self.delete_post_copy_cb.addItems(self.delete_post_copy_cb_list )
        self.delete_post_copy_cb.setStyleSheet('QComboBox { width: 272px; }')
        self.delete_post_copy_cb.activated.connect(partial(self._combo_box_prior_true if self.delete_post_copy_cb_list[0] == 'Yes' else self._combo_box_prior_false, variable_id='delete_post_copy_cb_bool'))
        self.delete_post_copy_cb.move(160, 220)

        self.cal_fr_label = QLabel('Calibration (10 fps):', self.VideoSettings)
        self.cal_fr_label.setFixedWidth(150)
        self.cal_fr_label.setFont(QFont(self.font_id, 12))
        self.cal_fr_label.move(5, 250)
        self.calibration_frame_rate = QSlider(Qt.Orientation.Horizontal, self.VideoSettings)
        self.calibration_frame_rate.setFixedWidth(150)
        self.calibration_frame_rate.move(160, 255)
        self.calibration_frame_rate.setRange(10, 150)
        self.calibration_frame_rate.setValue(self.exp_settings_dict['video']['general']['calibration_frame_rate'])
        self.calibration_frame_rate.valueChanged.connect(self._update_cal_fr_label)

        self.fr_label = QLabel('Recording (150 fps):', self.VideoSettings)
        self.fr_label.setFixedWidth(150)
        self.fr_label.setFont(QFont(self.font_id, 12))
        self.fr_label.move(5, 280)
        self.cameras_frame_rate = QSlider(Qt.Orientation.Horizontal, self.VideoSettings)
        self.cameras_frame_rate.setFixedWidth(150)
        self.cameras_frame_rate.move(160, 285)
        self.cameras_frame_rate.setRange(10, 150)
        self.cameras_frame_rate.setValue(self.exp_settings_dict['video']['general']['recording_frame_rate'])
        self.cameras_frame_rate.valueChanged.connect(self._update_fr_label)

        pcs_label = QLabel('Particular camera settings', self.VideoSettings)
        pcs_label.setFont(QFont(self.font_id, 13))
        pcs_label.setStyleSheet('QLabel { font-weight: bold;}')
        pcs_label.move(5, 325)

        camera_colors_global = ['white', 'orange', 'red', 'cyan', 'yellow']
        for cam_idx, cam in enumerate(self.exp_settings_dict['video']['general']['expected_cameras']):
            self._create_sliders_general(camera_id=cam, camera_color=camera_colors_global[cam_idx], y_start=355+(cam_idx*90))

        vm_label = QLabel('Metadata', self.VideoSettings)
        vm_label.setFont(QFont(self.font_id, 13))
        vm_label.setStyleSheet('QLabel { font-weight: bold;}')
        vm_label.move(495, 10)

        institution_label = QLabel('Institution:', self.VideoSettings)
        institution_label.setFont(QFont(self.font_id, 12))
        institution_label.move(495, 40)
        self.institution_entry = QLineEdit(f"{self.exp_settings_dict['video']['metadata']['institution']}", self.VideoSettings)
        self.institution_entry.setFont(QFont(self.font_id, 10))
        self.institution_entry.setStyleSheet('QLineEdit { width: 95px; }')
        self.institution_entry.move(630, 40)

        laboratory_label = QLabel('Laboratory:', self.VideoSettings)
        laboratory_label.setFont(QFont(self.font_id, 12))
        laboratory_label.move(750, 40)
        self.laboratory_entry = QLineEdit(f"{self.exp_settings_dict['video']['metadata']['laboratory']}", self.VideoSettings)
        self.laboratory_entry.setFont(QFont(self.font_id, 10))
        self.laboratory_entry.setStyleSheet('QLineEdit { width: 95px; }')
        self.laboratory_entry.move(875, 40)

        experimenter_label = QLabel('Experimenter:', self.VideoSettings)
        experimenter_label.setFont(QFont(self.font_id, 12))
        experimenter_label.move(495, 70)
        self.experimenter_entry = QLineEdit(f'{self.exp_id}', self.VideoSettings)
        self.experimenter_entry.setFont(QFont(self.font_id, 10))
        self.experimenter_entry.setStyleSheet('QLineEdit { width: 95px; }')
        self.experimenter_entry.move(630, 70)

        mice_num_label = QLabel('Animal count:', self.VideoSettings)
        mice_num_label.setFont(QFont(self.font_id, 12))
        mice_num_label.move(750, 70)
        self.mice_num_entry = QLineEdit(f"{self.exp_settings_dict['video']['metadata']['mice_num']}", self.VideoSettings)
        self.mice_num_entry.setFont(QFont(self.font_id, 10))
        self.mice_num_entry.setStyleSheet('QLineEdit { width: 95px; }')
        self.mice_num_entry.move(875, 70)

        vacant_arena_label = QLabel('Vacant arena:', self.VideoSettings)
        vacant_arena_label.setFont(QFont(self.font_id, 12))
        vacant_arena_label.move(495, 100)
        self.vacant_arena_cb_list = [x for _, x in sorted(zip([self.exp_settings_dict['video']['metadata']['vacant_arena'], not self.exp_settings_dict['video']['metadata']['vacant_arena']], self.boolean_list), reverse=True)]
        self.vacant_arena_cb = QComboBox(self.VideoSettings)
        self.vacant_arena_cb.addItems(self.vacant_arena_cb_list)
        self.vacant_arena_cb.setStyleSheet('QComboBox { width: 68px; }')
        self.vacant_arena_cb.activated.connect(partial(self._combo_box_prior_true if self.vacant_arena_cb_list[0] == 'Yes' else self._combo_box_prior_false, variable_id='vacant_arena_cb_bool'))
        self.vacant_arena_cb.move(630, 100)

        ambient_light_label = QLabel('Ambient light:', self.VideoSettings)
        ambient_light_label.setFont(QFont(self.font_id, 12))
        ambient_light_label.move(750, 100)
        self.ambient_light_cb_list = [x for _, x in sorted(zip([self.exp_settings_dict['video']['metadata']['ambient_light'], not self.exp_settings_dict['video']['metadata']['ambient_light']], self.boolean_list), reverse=True)]
        self.ambient_light_cb = QComboBox(self.VideoSettings)
        self.ambient_light_cb.addItems(self.ambient_light_cb_list)
        self.ambient_light_cb.setStyleSheet('QComboBox { width: 68px; }')
        self.ambient_light_cb.activated.connect(partial(self._combo_box_prior_true if self.ambient_light_cb_list[0] == 'Yes' else self._combo_box_prior_false, variable_id='ambient_light_cb_bool'))
        self.ambient_light_cb.move(875, 100)

        record_brain_label = QLabel('Record brain:', self.VideoSettings)
        record_brain_label.setFont(QFont(self.font_id, 12))
        record_brain_label.move(495, 130)
        self.record_brain_cb_list = [x for _, x in sorted(zip([self.exp_settings_dict['video']['metadata']['record_brain'], not self.exp_settings_dict['video']['metadata']['record_brain']], self.boolean_list), reverse=True)]
        self.record_brain_cb = QComboBox(self.VideoSettings)
        self.record_brain_cb.addItems(self.record_brain_cb_list)
        self.record_brain_cb.setStyleSheet('QComboBox { width: 68px; }')
        self.record_brain_cb.activated.connect(partial(self._combo_box_prior_true if self.record_brain_cb_list[0] == 'Yes' else self._combo_box_prior_false, variable_id='record_brain_cb_bool'))
        self.record_brain_cb.move(630, 130)

        usv_playback_label = QLabel('USV playback:', self.VideoSettings)
        usv_playback_label.setFont(QFont(self.font_id, 12))
        usv_playback_label.move(750, 130)
        self.usv_playback_cb_list = [x for _, x in sorted(zip([self.exp_settings_dict['video']['metadata']['usv_playback'], not self.exp_settings_dict['video']['metadata']['usv_playback']], self.boolean_list), reverse=True)]
        self.usv_playback_cb = QComboBox(self.VideoSettings)
        self.usv_playback_cb.addItems(self.usv_playback_cb_list)
        self.usv_playback_cb.setStyleSheet('QComboBox { width: 68px; }')
        self.usv_playback_cb.activated.connect(partial(self._combo_box_prior_true if self.usv_playback_cb_list[0] == 'Yes' else self._combo_box_prior_false, variable_id='usv_playback_cb_bool'))
        self.usv_playback_cb.move(875, 130)

        chemogenetics_label = QLabel('Chemogenetics:', self.VideoSettings)
        chemogenetics_label.setFont(QFont(self.font_id, 12))
        chemogenetics_label.move(495, 160)
        self.chemogenetics_cb_list = [x for _, x in sorted(zip([self.exp_settings_dict['video']['metadata']['chemogenetics'], not self.exp_settings_dict['video']['metadata']['chemogenetics']], self.boolean_list), reverse=True)]
        self.chemogenetics_cb = QComboBox(self.VideoSettings)
        self.chemogenetics_cb.addItems(self.chemogenetics_cb_list)
        self.chemogenetics_cb.setStyleSheet('QComboBox { width: 68px; }')
        self.chemogenetics_cb.activated.connect(partial(self._combo_box_prior_true if self.chemogenetics_cb_list[0] == 'Yes' else self._combo_box_prior_false, variable_id='chemogenetics_cb_bool'))
        self.chemogenetics_cb.move(630, 160)

        optogenetics_label = QLabel('Optogenetics:', self.VideoSettings)
        optogenetics_label.setFont(QFont(self.font_id, 12))
        optogenetics_label.move(750, 160)
        self.optogenetics_cb_list = [x for _, x in sorted(zip([self.exp_settings_dict['video']['metadata']['optogenetics'], not self.exp_settings_dict['video']['metadata']['optogenetics']], self.boolean_list), reverse=True)]
        self.optogenetics_cb = QComboBox(self.VideoSettings)
        self.optogenetics_cb.addItems(self.optogenetics_cb_list)
        self.optogenetics_cb.setStyleSheet('QComboBox { width: 68px; }')
        self.optogenetics_cb.activated.connect(partial(self._combo_box_prior_true if self.optogenetics_cb_list[0] == 'Yes' else self._combo_box_prior_false, variable_id='optogenetics_cb_bool'))
        self.optogenetics_cb.move(875, 160)

        brain_lesion_label = QLabel('Brain lesion:', self.VideoSettings)
        brain_lesion_label.setFont(QFont(self.font_id, 12))
        brain_lesion_label.move(495, 190)
        self.brain_lesion_cb_list = [x for _, x in sorted(zip([self.exp_settings_dict['video']['metadata']['brain_lesion'], not self.exp_settings_dict['video']['metadata']['brain_lesion']], self.boolean_list), reverse=True)]
        self.brain_lesion_cb = QComboBox(self.VideoSettings)
        self.brain_lesion_cb.addItems(self.brain_lesion_cb_list)
        self.brain_lesion_cb.setStyleSheet('QComboBox { width: 68px; }')
        self.brain_lesion_cb.activated.connect(partial(self._combo_box_prior_true if self.brain_lesion_cb_list[0] == 'Yes' else self._combo_box_prior_false, variable_id='brain_lesion_cb_bool'))
        self.brain_lesion_cb.move(630, 190)

        devocalization_label = QLabel('Devocalization:', self.VideoSettings)
        devocalization_label.setFont(QFont(self.font_id, 12))
        devocalization_label.move(750, 190)
        self.devocalization_cb_list = [x for _, x in sorted(zip([self.exp_settings_dict['video']['metadata']['devocalization'], not self.exp_settings_dict['video']['metadata']['devocalization']], self.boolean_list), reverse=True)]
        self.devocalization_cb = QComboBox(self.VideoSettings)
        self.devocalization_cb.addItems(self.devocalization_cb_list)
        self.devocalization_cb.setStyleSheet('QComboBox { width: 68px; }')
        self.devocalization_cb.activated.connect(partial(self._combo_box_prior_true if self.devocalization_cb_list[0] == 'Yes' else self._combo_box_prior_false, variable_id='devocalization_cb_bool'))
        self.devocalization_cb.move(875, 190)

        female_urine_label = QLabel('Female urine:', self.VideoSettings)
        female_urine_label.setFont(QFont(self.font_id, 12))
        female_urine_label.move(495, 220)
        self.female_urine_cb_list = [x for _, x in sorted(zip([self.exp_settings_dict['video']['metadata']['female_urine'], not self.exp_settings_dict['video']['metadata']['female_urine']], self.boolean_list), reverse=True)]
        self.female_urine_cb = QComboBox(self.VideoSettings)
        self.female_urine_cb.addItems(self.female_urine_cb_list)
        self.female_urine_cb.setStyleSheet('QComboBox { width: 68px; }')
        self.female_urine_cb.activated.connect(partial(self._combo_box_prior_true if self.female_urine_cb_list[0] == 'Yes' else self._combo_box_prior_false, variable_id='female_urine_cb_bool'))
        self.female_urine_cb.move(630, 220)

        female_bedding_label = QLabel('Female bedding:', self.VideoSettings)
        female_bedding_label.setFont(QFont(self.font_id, 12))
        female_bedding_label.move(750, 220)
        self.female_bedding_cb_list = [x for _, x in sorted(zip([self.exp_settings_dict['video']['metadata']['female_bedding'], not self.exp_settings_dict['video']['metadata']['female_bedding']], self.boolean_list), reverse=True)]
        self.female_bedding_cb = QComboBox(self.VideoSettings)
        self.female_bedding_cb.addItems(self.female_bedding_cb_list)
        self.female_bedding_cb.setStyleSheet('QComboBox { width: 68px; }')
        self.female_bedding_cb.activated.connect(partial(self._combo_box_prior_true if self.female_bedding_cb_list[0] == 'Yes' else self._combo_box_prior_false, variable_id='female_bedding_cb_bool'))
        self.female_bedding_cb.move(875, 220)

        species_label = QLabel('Species:', self.VideoSettings)
        species_label.setFont(QFont(self.font_id, 12))
        species_label.move(495, 250)
        self.species_entry = QLineEdit(f"{self.exp_settings_dict['video']['metadata']['species']}", self.VideoSettings)
        self.species_entry.setFont(QFont(self.font_id, 10))
        self.species_entry.setStyleSheet('QLineEdit { width: 402px; }')
        self.species_entry.move(570, 250)

        strain_label = QLabel('Strain-GT:', self.VideoSettings)
        strain_label.setFont(QFont(self.font_id, 12))
        strain_label.move(495, 280)
        self.strain_entry = QLineEdit(f"{self.exp_settings_dict['video']['metadata']['strain']}", self.VideoSettings)
        self.strain_entry.setFont(QFont(self.font_id, 10))
        self.strain_entry.setStyleSheet('QLineEdit { width: 402px; }')
        self.strain_entry.move(570, 280)

        cage_label = QLabel('Cage:', self.VideoSettings)
        cage_label.setFont(QFont(self.font_id, 12))
        cage_label.move(495, 310)
        self.cage_entry = QLineEdit(f"{self.exp_settings_dict['video']['metadata']['cage']}", self.VideoSettings)
        self.cage_entry.setFont(QFont(self.font_id, 10))
        self.cage_entry.setStyleSheet('QLineEdit { width: 402px; }')
        self.cage_entry.move(570, 310)

        subject_label = QLabel('Subject:', self.VideoSettings)
        subject_label.setFont(QFont(self.font_id, 12))
        subject_label.move(495, 340)
        self.subject_entry = QLineEdit(f"{self.exp_settings_dict['video']['metadata']['subject']}", self.VideoSettings)
        self.subject_entry.setFont(QFont(self.font_id, 10))
        self.subject_entry.setStyleSheet('QLineEdit { width: 402px; }')
        self.subject_entry.move(570, 340)

        dob_label = QLabel('DOB:', self.VideoSettings)
        dob_label.setFont(QFont(self.font_id, 12))
        dob_label.move(495, 370)
        self.dob_entry = QLineEdit(f"{self.exp_settings_dict['video']['metadata']['dob']}", self.VideoSettings)
        self.dob_entry.setFont(QFont(self.font_id, 10))
        self.dob_entry.setStyleSheet('QLineEdit { width: 402px; }')
        self.dob_entry.move(570, 370)

        sex_label = QLabel('Sex:', self.VideoSettings)
        sex_label.setFont(QFont(self.font_id, 12))
        sex_label.move(495, 400)
        self.sex_entry = QLineEdit(f"{self.exp_settings_dict['video']['metadata']['sex']}", self.VideoSettings)
        self.sex_entry.setFont(QFont(self.font_id, 10))
        self.sex_entry.setStyleSheet('QLineEdit { width: 402px; }')
        self.sex_entry.move(570, 400)

        weight_label = QLabel('Weight:', self.VideoSettings)
        weight_label.setFont(QFont(self.font_id, 12))
        weight_label.move(495, 430)
        self.weight_entry = QLineEdit(f"{self.exp_settings_dict['video']['metadata']['weight']}", self.VideoSettings)
        self.weight_entry.setFont(QFont(self.font_id, 10))
        self.weight_entry.setStyleSheet('QLineEdit { width: 402px; }')
        self.weight_entry.move(570, 430)

        housing_label = QLabel('Housing:', self.VideoSettings)
        housing_label.setFont(QFont(self.font_id, 12))
        housing_label.move(495, 460)
        self.housing_entry = QLineEdit(f"{self.exp_settings_dict['video']['metadata']['housing']}", self.VideoSettings)
        self.housing_entry.setFont(QFont(self.font_id, 10))
        self.housing_entry.setStyleSheet('QLineEdit { width: 402px; }')
        self.housing_entry.move(570, 460)

        implant_label = QLabel('Implant:', self.VideoSettings)
        implant_label.setFont(QFont(self.font_id, 12))
        implant_label.move(495, 490)
        self.implant_entry = QLineEdit(f"{self.exp_settings_dict['video']['metadata']['implant']}", self.VideoSettings)
        self.implant_entry.setFont(QFont(self.font_id, 10))
        self.implant_entry.setStyleSheet('QLineEdit { width: 402px; }')
        self.implant_entry.move(570, 490)

        virus_label = QLabel('Virus:', self.VideoSettings)
        virus_label.setFont(QFont(self.font_id, 12))
        virus_label.move(495, 520)
        self.virus_entry = QLineEdit(f"{self.exp_settings_dict['video']['metadata']['virus']}", self.VideoSettings)
        self.virus_entry.setFont(QFont(self.font_id, 10))
        self.virus_entry.setStyleSheet('QLineEdit { width: 402px; }')
        self.virus_entry.move(570, 520)

        notes_label = QLabel('Notes:', self.VideoSettings)
        notes_label.setFont(QFont(self.font_id, 12))
        notes_label.move(495, 550)
        self.notes_entry = QTextEdit(f"{self.exp_settings_dict['video']['metadata']['notes']}", self.VideoSettings)
        self.notes_entry.setFont(QFont(self.font_id, 10))
        self.notes_entry.move(570, 550)
        self.notes_entry.setFixedSize(403, 290)

        self._create_buttons_record(seq=2, class_option=self.VideoSettings,
                                    button_pos_y=record_three_y - 35, next_button_x_pos=record_three_x - 100)

    def record_four(self):
        self.ConductRecording = ConductRecording(self)
        self.setWindowTitle(f'{app_name} (Conduct recording)')
        self.setCentralWidget(self.ConductRecording)
        record_four_x, record_four_y = (480, 560)
        self.setFixedSize(record_four_x, record_four_y)

        self.txt_edit = QPlainTextEdit(self.ConductRecording)
        self.txt_edit.move(5, 5)
        self.txt_edit.setFixedSize(465, 500)
        self.txt_edit.setReadOnly(True)

        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '_config/behavioral_experiments_settings.toml'), 'w') as updated_toml_file:
            toml.dump(self.exp_settings_dict, updated_toml_file)

        self.run_exp = ExperimentController(message_output=self._message,
                                            email_receivers=self.email_recipients,
                                            exp_settings_dict=self.exp_settings_dict)

        self._create_buttons_record(seq=3, class_option=self.ConductRecording,
                                    button_pos_y=record_four_y - 35, next_button_x_pos=record_four_x - 100)

    def process_one(self):
        self.ProcessSettings = ProcessSettings(self)
        self.setWindowTitle(f'{app_name} (Process recordings > Settings)')
        self.setCentralWidget(self.ProcessSettings)
        record_four_x, record_four_y = (1020, 935)
        self.setFixedSize(record_four_x, record_four_y)

        # column 1

        processing_dir_label = QLabel('(*) Root directories for processing', self.ProcessSettings)
        processing_dir_label.setFont(QFont(self.font_id, 13))
        processing_dir_label.setStyleSheet('QLabel { font-weight: bold;}')
        processing_dir_label.move(50, 10)
        self.processing_dir_edit = QTextEdit('', self.ProcessSettings)
        self.processing_dir_edit.setFont(QFont(self.font_id, 10))
        self.processing_dir_edit.move(10, 40)
        self.processing_dir_edit.setFixedSize(350, 320)

        sleap_conda_label = QLabel('SLEAP conda environment name:', self.ProcessSettings)
        sleap_conda_label.setFont(QFont(self.font_id, 12))
        sleap_conda_label.move(10, 365)
        self.sleap_conda = QLineEdit(f"{self.processing_input_dict['anipose_operations']['ConvertTo3D']['sleap_file_conversion']['sleap_conda_env_name']}", self.ProcessSettings)
        self.sleap_conda.setFont(QFont(self.font_id, 10))
        self.sleap_conda.setStyleSheet('QLineEdit { width: 85px; }')
        self.sleap_conda.move(275, 365)

        self.centroid_model_edit = QLineEdit(f"{self.processing_input_dict['prepare_cluster_job']['centroid_model_path']}", self.ProcessSettings)
        self.centroid_model_edit.setPlaceholderText('SLEAP centroid model directory')
        self.centroid_model_edit.setFont(QFont(self.font_id, 10))
        self.centroid_model_edit.setStyleSheet('QLineEdit { width: 260px; }')
        self.centroid_model_edit.move(10, 395)
        centroid_model_btn = QPushButton('Browse', self.ProcessSettings)
        centroid_model_btn.setFont(QFont(self.font_id, 8))
        centroid_model_btn.move(275, 395)
        centroid_model_btn.setStyleSheet('QPushButton { min-width: 64px; min-height: 12px; max-width: 64px; max-height: 13px; }')
        self.centroid_model_btn_clicked_flag = False
        centroid_model_btn.clicked.connect(self._open_centroid_dialog)

        self.centered_instance_model_edit = QLineEdit(f"{self.processing_input_dict['prepare_cluster_job']['centered_instance_model_path']}", self.ProcessSettings)
        self.centered_instance_model_edit.setPlaceholderText('SLEAP centered instance model directory')
        self.centered_instance_model_edit.setFont(QFont(self.font_id, 10))
        self.centered_instance_model_edit.setStyleSheet('QLineEdit { width: 260px; }')
        self.centered_instance_model_edit.move(10, 425)
        centered_instance_btn = QPushButton('Browse', self.ProcessSettings)
        centered_instance_btn.setFont(QFont(self.font_id, 8))
        centered_instance_btn.move(275, 425)
        centered_instance_btn.setStyleSheet('QPushButton { min-width: 64px; min-height: 12px; max-width: 64px; max-height: 13px; }')
        self.centered_instance_btn_btn_clicked_flag = False
        centered_instance_btn.clicked.connect(self._open_centered_instance_dialog)

        self.inference_root_dir_edit = QLineEdit(self.sleap_inference_dir_global, self.ProcessSettings)
        self.inference_root_dir_edit.setPlaceholderText('SLEAP inference directory')
        self.inference_root_dir_edit.setFont(QFont(self.font_id, 10))
        self.inference_root_dir_edit.setStyleSheet('QLineEdit { width: 260px; }')
        self.inference_root_dir_edit.move(10, 455)
        inference_root_dir_btn = QPushButton('Browse', self.ProcessSettings)
        inference_root_dir_btn.setFont(QFont(self.font_id, 8))
        inference_root_dir_btn.move(275, 455)
        inference_root_dir_btn.setStyleSheet('QPushButton { min-width: 64px; min-height: 12px; max-width: 64px; max-height: 13px; }')
        self.inference_root_dir_btn_clicked_flag = False
        inference_root_dir_btn.clicked.connect(self._open_inference_root_dialog)

        self.calibration_file_loc_edit = QLineEdit('', self.ProcessSettings)
        self.calibration_file_loc_edit.setPlaceholderText('Tracking calibration root directory')
        self.calibration_file_loc_edit.setFont(QFont(self.font_id, 10))
        self.calibration_file_loc_edit.setStyleSheet('QLineEdit { width: 260px; }')
        self.calibration_file_loc_edit.move(10, 485)
        calibration_file_loc_btn = QPushButton('Browse', self.ProcessSettings)
        calibration_file_loc_btn.setFont(QFont(self.font_id, 8))
        calibration_file_loc_btn.move(275, 485)
        calibration_file_loc_btn.setStyleSheet('QPushButton { min-width: 64px; min-height: 12px; max-width: 64px; max-height: 12px; }')
        self.calibration_file_loc_btn_clicked_flag = False
        calibration_file_loc_btn.clicked.connect(self._open_anipose_calibration_dialog)

        das_conda_label = QLabel('DAS conda environment name:', self.ProcessSettings)
        das_conda_label.setFont(QFont(self.font_id, 12))
        das_conda_label.move(10, 515)
        self.das_conda = QLineEdit(self.processing_input_dict['usv_inference']['FindMouseVocalizations']['das_command_line_inference']['das_conda_env_name'], self.ProcessSettings)
        self.das_conda.setFont(QFont(self.font_id, 10))
        self.das_conda.setStyleSheet('QLineEdit { width: 85px; }')
        self.das_conda.move(275, 515)

        self.das_model_dir_edit = QLineEdit(self.das_model_dir_global, self.ProcessSettings)
        self.das_model_dir_edit.setPlaceholderText('DAS model directory')
        self.das_model_dir_edit.setFont(QFont(self.font_id, 10))
        self.das_model_dir_edit.setStyleSheet('QLineEdit { width: 260px; }')
        self.das_model_dir_edit.move(10, 545)
        das_model_dir_btn = QPushButton('Browse', self.ProcessSettings)
        das_model_dir_btn.setFont(QFont(self.font_id, 8))
        das_model_dir_btn.move(275, 545)
        das_model_dir_btn.setStyleSheet('QPushButton { min-width: 64px; min-height: 12px; max-width: 64px; max-height: 12px; }')
        self.das_model_dir_btn_clicked_flag = False
        das_model_dir_btn.clicked.connect(self._open_das_model_dialog)

        das_model_base_label = QLabel('DAS model base (timestamp):', self.ProcessSettings)
        das_model_base_label.setFont(QFont(self.font_id, 12))
        das_model_base_label.move(10, 575)
        self.das_model_base = QLineEdit(self.processing_input_dict['usv_inference']['FindMouseVocalizations']['das_command_line_inference']['model_name_base'], self.ProcessSettings)
        self.das_model_base.setFont(QFont(self.font_id, 10))
        self.das_model_base.setStyleSheet('QLineEdit { width: 135px; }')
        self.das_model_base.move(225, 575)

        pc_usage_process_label = QLabel('Notify e-mail(s) of PC usage:', self.ProcessSettings)
        pc_usage_process_label.setFont(QFont(self.font_id, 12))
        pc_usage_process_label.move(10, 605)
        self.pc_usage_process = QLineEdit('', self.ProcessSettings)
        self.pc_usage_process.setFont(QFont(self.font_id, 10))
        self.pc_usage_process.setStyleSheet('QLineEdit { width: 135px; }')
        self.pc_usage_process.move(225, 605)

        processing_pc_label = QLabel('Processing PC of choice:', self.ProcessSettings)
        processing_pc_label.setFont(QFont(self.font_id, 12))
        processing_pc_label.move(10, 635)
        self.loaded_processing_pc_list = sorted(self.processing_input_dict['send_email']['Messenger']['processing_pc_list'], key=lambda x: x == self.processing_input_dict['send_email']['Messenger']['processing_pc_choice'], reverse=True)
        self.processing_pc_cb = QComboBox(self.ProcessSettings)
        self.processing_pc_cb.addItems(self.loaded_processing_pc_list)
        self.processing_pc_cb.setStyleSheet('QComboBox { width: 107px; }')
        self.processing_pc_cb.activated.connect(partial(self._combo_box_prior_processing_pc_choice, variable_id='processing_pc_choice'))
        self.processing_pc_cb.move(225, 635)

        gvs_label = QLabel('Video processing settings', self.ProcessSettings)
        gvs_label.setFont(QFont(self.font_id, 13))
        gvs_label.setStyleSheet('QLabel { font-weight: bold;}')
        gvs_label.move(10, 675)

        conduct_video_concatenation_label = QLabel('Conduct video concatenation:', self.ProcessSettings)
        conduct_video_concatenation_label.setFont(QFont(self.font_id, 11))
        conduct_video_concatenation_label.setStyleSheet('QLabel { color: #F58025; font-weight: bold;}')
        conduct_video_concatenation_label.move(10, 705)
        self.conduct_video_concatenation_cb = QComboBox(self.ProcessSettings)
        self.conduct_video_concatenation_cb.addItems(['No', 'Yes'])
        self.conduct_video_concatenation_cb.setStyleSheet('QComboBox { width: 105px; }')
        self.conduct_video_concatenation_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='conduct_video_concatenation_cb_bool'))
        self.conduct_video_concatenation_cb.move(225, 705)

        conduct_video_fps_change_cb_label = QLabel('Conduct video re-encoding:', self.ProcessSettings)
        conduct_video_fps_change_cb_label.setFont(QFont(self.font_id, 11))
        conduct_video_fps_change_cb_label.setStyleSheet('QLabel { color: #F58025; font-weight: bold;}')
        conduct_video_fps_change_cb_label.move(10, 735)
        self.conduct_video_fps_change_cb = QComboBox(self.ProcessSettings)
        self.conduct_video_fps_change_cb.addItems(['No', 'Yes'])
        self.conduct_video_fps_change_cb.setStyleSheet('QComboBox { width: 105px; }')
        self.conduct_video_fps_change_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='conduct_video_fps_change_cb_bool'))
        self.conduct_video_fps_change_cb.move(225, 735)

        conversion_target_file_label = QLabel('Concatenated video name:', self.ProcessSettings)
        conversion_target_file_label.setFont(QFont(self.font_id, 12))
        conversion_target_file_label.move(10, 765)
        self.conversion_target_file = QLineEdit(self.processing_input_dict['modify_files']['Operator']['rectify_video_fps']['conversion_target_file'], self.ProcessSettings)
        self.conversion_target_file.setFont(QFont(self.font_id, 10))
        self.conversion_target_file.setStyleSheet('QLineEdit { width: 135px; }')
        self.conversion_target_file.move(225, 765)

        constant_rate_factor_label = QLabel('FFMPEG rate factor (-crf):', self.ProcessSettings)
        constant_rate_factor_label.setFont(QFont(self.font_id, 12))
        constant_rate_factor_label.move(10, 795)
        self.constant_rate_factor = QLineEdit(f"{self.processing_input_dict['modify_files']['Operator']['rectify_video_fps']['constant_rate_factor']}", self.ProcessSettings)
        self.constant_rate_factor.setFont(QFont(self.font_id, 10))
        self.constant_rate_factor.setStyleSheet('QLineEdit { width: 135px; }')
        self.constant_rate_factor.move(225, 795)

        encoding_preset_label = QLabel('FFMPEG encoding preset:', self.ProcessSettings)
        encoding_preset_label.setFont(QFont(self.font_id, 12))
        encoding_preset_label.move(10, 825)
        self.encoding_preset = QLineEdit(self.processing_input_dict['modify_files']['Operator']['rectify_video_fps']['encoding_preset'], self.ProcessSettings)
        self.encoding_preset.setFont(QFont(self.font_id, 10))
        self.encoding_preset.setStyleSheet('QLineEdit { width: 135px; }')
        self.encoding_preset.move(225, 825)

        delete_con_file_cb_label = QLabel('Delete concatenated video:', self.ProcessSettings)
        delete_con_file_cb_label.setFont(QFont(self.font_id, 12))
        delete_con_file_cb_label.move(10, 855)
        self.delete_con_file_cb = QComboBox(self.ProcessSettings)
        self.delete_con_file_cb.addItems(['Yes', 'No'])
        self.delete_con_file_cb.setStyleSheet('QComboBox { width: 105px; }')
        self.delete_con_file_cb.activated.connect(partial(self._combo_box_prior_true, variable_id='delete_con_file_cb_bool'))
        self.delete_con_file_cb.move(225, 855)

        # column 2

        column_two_x1 = 380
        column_two_x2 = 570

        gas_label = QLabel('Audio processing settings', self.ProcessSettings)
        gas_label.setFont(QFont(self.font_id, 13))
        gas_label.setStyleSheet('QLabel { font-weight: bold;}')
        gas_label.move(column_two_x1, 10)

        conduct_multichannel_conversion_cb_label = QLabel('Convert to single-ch files:', self.ProcessSettings)
        conduct_multichannel_conversion_cb_label.setFont(QFont(self.font_id, 11))
        conduct_multichannel_conversion_cb_label.setStyleSheet('QLabel { color: #F58025; font-weight: bold;}')
        conduct_multichannel_conversion_cb_label.move(column_two_x1, 40)
        self.conduct_multichannel_conversion_cb = QComboBox(self.ProcessSettings)
        self.conduct_multichannel_conversion_cb.addItems(['No', 'Yes'])
        self.conduct_multichannel_conversion_cb.setStyleSheet('QComboBox { width: 80px; }')
        self.conduct_multichannel_conversion_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='conduct_multichannel_conversion_cb_bool'))
        self.conduct_multichannel_conversion_cb.move(column_two_x2, 40)

        crop_wav_cam_cb_label = QLabel('Crop AUDIO (to VIDEO):', self.ProcessSettings)
        crop_wav_cam_cb_label.setFont(QFont(self.font_id, 11))
        crop_wav_cam_cb_label.setStyleSheet('QLabel { color: #F58025; font-weight: bold;}')
        crop_wav_cam_cb_label.move(column_two_x1, 70)
        self.crop_wav_cam_cb = QComboBox(self.ProcessSettings)
        self.crop_wav_cam_cb.addItems(['No', 'Yes'])
        self.crop_wav_cam_cb.setStyleSheet('QComboBox { width: 80px; }')
        self.crop_wav_cam_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='crop_wav_cam_cb_bool'))
        self.crop_wav_cam_cb.move(column_two_x2, 70)

        device_receiving_input_cb_label = QLabel('Trgbox-USGH device(s):', self.ProcessSettings)
        device_receiving_input_cb_label.setFont(QFont(self.font_id, 12))
        device_receiving_input_cb_label.move(column_two_x1, 100)
        self.device_receiving_input_cb = QComboBox(self.ProcessSettings)
        self.device_receiving_input_cb.addItems(['m', 's', 'both'])
        self.device_receiving_input_cb.setStyleSheet('QComboBox { width: 80px; }')
        self.device_receiving_input_cb.activated.connect(partial(self._combo_box_prior_audio_device_camera_input, variable_id='device_receiving_input'))
        self.device_receiving_input_cb.move(column_two_x2, 100)

        ch_receiving_input_label = QLabel('Trgbox-USGH ch (1-12):', self.ProcessSettings)
        ch_receiving_input_label.setFont(QFont(self.font_id, 12))
        ch_receiving_input_label.move(column_two_x1, 130)
        self.ch_receiving_input = QLineEdit(f"{self.processing_input_dict['synchronize_files']['Synchronizer']['crop_wav_files_to_video']['ch_receiving_input']}", self.ProcessSettings)
        self.ch_receiving_input.setFont(QFont(self.font_id, 10))
        self.ch_receiving_input.setStyleSheet('QLineEdit { width: 108px; }')
        self.ch_receiving_input.move(column_two_x2, 130)

        conduct_hpss_cb_label = QLabel('Conduct HPSS (slow!):', self.ProcessSettings)
        conduct_hpss_cb_label.setFont(QFont(self.font_id, 11))
        conduct_hpss_cb_label.setStyleSheet('QLabel { color: #F58025; font-weight: bold;}')
        conduct_hpss_cb_label.move(column_two_x1, 160)
        self.conduct_hpss_cb = QComboBox(self.ProcessSettings)
        self.conduct_hpss_cb.addItems(['No', 'Yes'])
        self.conduct_hpss_cb.setStyleSheet('QComboBox { width: 80px; }')
        self.conduct_hpss_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='conduct_hpss_cb_bool'))
        self.conduct_hpss_cb.move(column_two_x2, 160)

        stft_label = QLabel('STFT window & hop size:', self.ProcessSettings)
        stft_label.setFont(QFont(self.font_id, 12))
        stft_label.move(column_two_x1, 190)
        self.stft_window_hop = QLineEdit(','.join([str(x) for x in self.processing_input_dict['modify_files']['Operator']['hpss_audio']['stft_window_length_hop_size']]), self.ProcessSettings)
        self.stft_window_hop.setFont(QFont(self.font_id, 10))
        self.stft_window_hop.setStyleSheet('QLineEdit { width: 108px; }')
        self.stft_window_hop.move(column_two_x2, 190)

        hpss_kernel_size_label = QLabel('HPSS kernel size:', self.ProcessSettings)
        hpss_kernel_size_label.setFont(QFont(self.font_id, 12))
        hpss_kernel_size_label.move(column_two_x1, 220)
        self.hpss_kernel_size = QLineEdit(','.join([str(x) for x in self.processing_input_dict['modify_files']['Operator']['hpss_audio']['kernel_size']]), self.ProcessSettings)
        self.hpss_kernel_size.setFont(QFont(self.font_id, 10))
        self.hpss_kernel_size.setStyleSheet('QLineEdit { width: 108px; }')
        self.hpss_kernel_size.move(column_two_x2, 220)

        hpss_power_label = QLabel('HPSS power:', self.ProcessSettings)
        hpss_power_label.setFont(QFont(self.font_id, 12))
        hpss_power_label.move(column_two_x1, 250)
        self.hpss_power = QLineEdit(f"{self.processing_input_dict['modify_files']['Operator']['hpss_audio']['hpss_power']}", self.ProcessSettings)
        self.hpss_power.setFont(QFont(self.font_id, 10))
        self.hpss_power.setStyleSheet('QLineEdit { width: 108px; }')
        self.hpss_power.move(column_two_x2, 250)

        hpss_margin_label = QLabel('HPSS margin:', self.ProcessSettings)
        hpss_margin_label.setFont(QFont(self.font_id, 12))
        hpss_margin_label.move(column_two_x1, 280)
        self.hpss_margin = QLineEdit(','.join([str(x) for x in self.processing_input_dict['modify_files']['Operator']['hpss_audio']['margin']]), self.ProcessSettings)
        self.hpss_margin.setFont(QFont(self.font_id, 10))
        self.hpss_margin.setStyleSheet('QLineEdit { width: 108px; }')
        self.hpss_margin.move(column_two_x2, 280)

        filter_audio_cb_label = QLabel('Filter individual audio files:', self.ProcessSettings)
        filter_audio_cb_label.setFont(QFont(self.font_id, 11))
        filter_audio_cb_label.setStyleSheet('QLabel { color: #F58025; font-weight: bold;}')
        filter_audio_cb_label.move(column_two_x1, 310)
        self.filter_audio_cb = QComboBox(self.ProcessSettings)
        self.filter_audio_cb.addItems(['No', 'Yes'])
        self.filter_audio_cb.setStyleSheet('QComboBox { width: 80px; }')
        self.filter_audio_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='filter_audio_cb_bool'))
        self.filter_audio_cb.move(column_two_x2, 310)

        filter_freq_bounds_label = QLabel('Filter freq bounds (Hz):', self.ProcessSettings)
        filter_freq_bounds_label.setFont(QFont(self.font_id, 12))
        filter_freq_bounds_label.move(column_two_x1, 340)
        self.filter_freq_bounds = QLineEdit(','.join([str(x) for x in self.processing_input_dict['modify_files']['Operator']['filter_audio_files']['filter_freq_bounds']]), self.ProcessSettings)
        self.filter_freq_bounds.setFont(QFont(self.font_id, 10))
        self.filter_freq_bounds.setStyleSheet('QLineEdit { width: 108px; }')
        self.filter_freq_bounds.move(column_two_x2, 340)

        filter_dirs_label = QLabel('Folder(s) to filter:', self.ProcessSettings)
        filter_dirs_label.setFont(QFont(self.font_id, 12))
        filter_dirs_label.move(column_two_x1, 370)
        self.filter_dirs = QLineEdit(','.join([str(x) for x in self.processing_input_dict['modify_files']['Operator']['filter_audio_files']['filter_dirs']]), self.ProcessSettings)
        self.filter_dirs.setFont(QFont(self.font_id, 10))
        self.filter_dirs.setStyleSheet('QLineEdit { width: 108px; }')
        self.filter_dirs.move(column_two_x2, 370)

        conc_audio_cb_label = QLabel('Concatenate to MEMMAP:', self.ProcessSettings)
        conc_audio_cb_label.setFont(QFont(self.font_id, 11))
        conc_audio_cb_label.setStyleSheet('QLabel { color: #F58025; font-weight: bold;}')
        conc_audio_cb_label.move(column_two_x1, 400)
        self.conc_audio_cb = QComboBox(self.ProcessSettings)
        self.conc_audio_cb.addItems(['No', 'Yes'])
        self.conc_audio_cb.setStyleSheet('QComboBox { width: 80px; }')
        self.conc_audio_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='conc_audio_cb_bool'))
        self.conc_audio_cb.move(column_two_x2, 400)

        concat_dirs_label = QLabel('Folder(s) to concatenate:', self.ProcessSettings)
        concat_dirs_label.setFont(QFont(self.font_id, 12))
        concat_dirs_label.move(column_two_x1, 430)
        self.concat_dirs = QLineEdit(','.join([str(x) for x in self.processing_input_dict['modify_files']['Operator']['concatenate_audio_files']['concat_dirs']]), self.ProcessSettings)
        self.concat_dirs.setFont(QFont(self.font_id, 10))
        self.concat_dirs.setStyleSheet('QLineEdit { width: 108px; }')
        self.concat_dirs.move(column_two_x2, 430)

        av_sync_label = QLabel('Synchronization between A/V files', self.ProcessSettings)
        av_sync_label.setFont(QFont(self.font_id, 13))
        av_sync_label.setStyleSheet('QLabel { font-weight: bold;}')
        av_sync_label.move(column_two_x1, 470)

        conduct_sync_cb_label = QLabel('Conduct A/V sync check:', self.ProcessSettings)
        conduct_sync_cb_label.setFont(QFont(self.font_id, 11))
        conduct_sync_cb_label.setStyleSheet('QLabel { color: #F58025; font-weight: bold;}')
        conduct_sync_cb_label.move(column_two_x1, 500)
        self.conduct_sync_cb = QComboBox(self.ProcessSettings)
        self.conduct_sync_cb.addItems(['No', 'Yes'])
        self.conduct_sync_cb.setStyleSheet('QComboBox { width: 80px; }')
        self.conduct_sync_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='conduct_sync_cb_bool'))
        self.conduct_sync_cb.move(column_two_x2, 500)

        phidget_extra_data_camera_label = QLabel('Phidget(s) camera serial:', self.ProcessSettings)
        phidget_extra_data_camera_label.setFont(QFont(self.font_id, 12))
        phidget_extra_data_camera_label.move(column_two_x1, 530)
        self.phidget_extra_data_camera = QLineEdit(f"{self.processing_input_dict['extract_phidget_data']['Gatherer']['prepare_data_for_analyses']['extra_data_camera']}", self.ProcessSettings)
        self.phidget_extra_data_camera.setFont(QFont(self.font_id, 10))
        self.phidget_extra_data_camera.setStyleSheet('QLineEdit { width: 108px; }')
        self.phidget_extra_data_camera.move(column_two_x2, 530)

        a_ch_receiving_input_label = QLabel('Arduino-USGH ch (1-12):', self.ProcessSettings)
        a_ch_receiving_input_label.setFont(QFont(self.font_id, 12))
        a_ch_receiving_input_label.move(column_two_x1, 560)
        self.a_ch_receiving_input = QLineEdit(f"{self.processing_input_dict['synchronize_files']['Synchronizer']['find_audio_sync_trains']['ch_receiving_input']}", self.ProcessSettings)
        self.a_ch_receiving_input.setFont(QFont(self.font_id, 10))
        self.a_ch_receiving_input.setStyleSheet('QLineEdit { width: 108px; }')
        self.a_ch_receiving_input.move(column_two_x2, 560)

        v_camera_serial_num_label = QLabel('Sync camera serial num(s):', self.ProcessSettings)
        v_camera_serial_num_label.setFont(QFont(self.font_id, 12))
        v_camera_serial_num_label.move(column_two_x1, 590)
        self.v_camera_serial_num = QLineEdit(','.join([str(x) for x in self.processing_input_dict['synchronize_files']['Synchronizer']['find_video_sync_trains']['camera_serial_num']]), self.ProcessSettings)
        self.v_camera_serial_num.setFont(QFont(self.font_id, 10))
        self.v_camera_serial_num.setStyleSheet('QLineEdit { width: 108px; }')
        self.v_camera_serial_num.move(column_two_x2, 590)

        ev_sync_label = QLabel('Neural data processing settings', self.ProcessSettings)
        ev_sync_label.setFont(QFont(self.font_id, 13))
        ev_sync_label.setStyleSheet('QLabel { font-weight: bold;}')
        ev_sync_label.move(column_two_x1, 630)

        conduct_nv_sync_cb_label = QLabel('Conduct E/V sync check:', self.ProcessSettings)
        conduct_nv_sync_cb_label.setFont(QFont(self.font_id, 11))
        conduct_nv_sync_cb_label.setStyleSheet('QLabel { color: #F58025; font-weight: bold;}')
        conduct_nv_sync_cb_label.move(column_two_x1, 660)
        self.conduct_nv_sync_cb = QComboBox(self.ProcessSettings)
        self.conduct_nv_sync_cb.addItems(['No', 'Yes'])
        self.conduct_nv_sync_cb.setStyleSheet('QComboBox { width: 80px; }')
        self.conduct_nv_sync_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='conduct_nv_sync_cb_bool'))
        self.conduct_nv_sync_cb.move(column_two_x2, 660)

        conduct_ephys_file_chaining_label = QLabel('Conduct e-phys concat:', self.ProcessSettings)
        conduct_ephys_file_chaining_label.setFont(QFont(self.font_id, 11))
        conduct_ephys_file_chaining_label.setStyleSheet('QLabel { color: #F58025; font-weight: bold;}')
        conduct_ephys_file_chaining_label.move(column_two_x1, 690)
        self.conduct_ephys_file_chaining_cb = QComboBox(self.ProcessSettings)
        self.conduct_ephys_file_chaining_cb.addItems(['No', 'Yes'])
        self.conduct_ephys_file_chaining_cb.setStyleSheet('QComboBox { width: 80px; }')
        self.conduct_ephys_file_chaining_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='conduct_ephys_file_chaining_cb_bool'))
        self.conduct_ephys_file_chaining_cb.move(column_two_x2, 690)

        split_cluster_spikes_cb_label = QLabel('Split clusters to sessions:', self.ProcessSettings)
        split_cluster_spikes_cb_label.setFont(QFont(self.font_id, 11))
        split_cluster_spikes_cb_label.setStyleSheet('QLabel { color: #F58025; font-weight: bold;}')
        split_cluster_spikes_cb_label.move(column_two_x1, 720)
        self.split_cluster_spikes_cb = QComboBox(self.ProcessSettings)
        self.split_cluster_spikes_cb.addItems(['No', 'Yes'])
        self.split_cluster_spikes_cb.setStyleSheet('QComboBox { width: 80px; }')
        self.split_cluster_spikes_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='split_cluster_spikes_cb_bool'))
        self.split_cluster_spikes_cb.move(column_two_x2, 720)

        npx_file_type_cb_label = QLabel('Rec file format (ap | lf):', self.ProcessSettings)
        npx_file_type_cb_label.setFont(QFont(self.font_id, 12))
        npx_file_type_cb_label.move(column_two_x1, 750)
        self.npx_file_type_cb = QComboBox(self.ProcessSettings)
        self.npx_file_type_cb.addItems(['ap', 'lf'])
        self.npx_file_type_cb.setStyleSheet('QComboBox { width: 80px; }')
        self.npx_file_type_cb.activated.connect(partial(self._combo_box_prior_npx_file_type, variable_id='npx_file_type'))
        self.npx_file_type_cb.move(column_two_x2, 750)

        npx_ms_divergence_tolerance_label = QLabel('Divergence tolerance (ms):', self.ProcessSettings)
        npx_ms_divergence_tolerance_label.setFont(QFont(self.font_id, 12))
        npx_ms_divergence_tolerance_label.move(column_two_x1, 780)
        self.npx_ms_divergence_tolerance = QLineEdit(f"{self.processing_input_dict['synchronize_files']['Synchronizer']['validate_ephys_video_sync']['npx_ms_divergence_tolerance']}", self.ProcessSettings)
        self.npx_ms_divergence_tolerance.setFont(QFont(self.font_id, 10))
        self.npx_ms_divergence_tolerance.setStyleSheet('QLineEdit { width: 108px; }')
        self.npx_ms_divergence_tolerance.move(column_two_x2, 780)

        min_spike_num_label = QLabel('Min num of spikes:', self.ProcessSettings)
        min_spike_num_label.setFont(QFont(self.font_id, 12))
        min_spike_num_label.move(column_two_x1, 810)
        self.min_spike_num = QLineEdit(f"{self.processing_input_dict['modify_files']['Operator']['get_spike_times']['min_spike_num']}", self.ProcessSettings)
        self.min_spike_num.setFont(QFont(self.font_id, 10))
        self.min_spike_num.setStyleSheet('QLineEdit { width: 108px; }')
        self.min_spike_num.move(column_two_x2, 810)

        # column 3
        column_three_x1 = 700
        column_three_x2 = 900

        anipose_operations_label = QLabel('SLEAP / DAS operations', self.ProcessSettings)
        anipose_operations_label.setFont(QFont(self.font_id, 13))
        anipose_operations_label.setStyleSheet('QLabel { font-weight: bold;}')
        anipose_operations_label.move(column_three_x1, 10)

        sleap_cluster_cb_label = QLabel('Prepare SLEAP cluster job:', self.ProcessSettings)
        sleap_cluster_cb_label.setFont(QFont(self.font_id, 11))
        sleap_cluster_cb_label.setStyleSheet('QLabel { color: #F58025; font-weight: bold;}')
        sleap_cluster_cb_label.move(column_three_x1, 40)
        self.sleap_cluster_cb = QComboBox(self.ProcessSettings)
        self.sleap_cluster_cb.addItems(['No', 'Yes'])
        self.sleap_cluster_cb.setStyleSheet('QComboBox { width: 80px; }')
        self.sleap_cluster_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='sleap_cluster_cb_bool'))
        self.sleap_cluster_cb.move(column_three_x2, 40)

        sleap_file_conversion_cb_label = QLabel('Conduct SLP-H5 conversion:', self.ProcessSettings)
        sleap_file_conversion_cb_label.setFont(QFont(self.font_id, 11))
        sleap_file_conversion_cb_label.setStyleSheet('QLabel { color: #F58025; font-weight: bold;}')
        sleap_file_conversion_cb_label.move(column_three_x1, 70)
        self.sleap_file_conversion_cb = QComboBox(self.ProcessSettings)
        self.sleap_file_conversion_cb.addItems(['No', 'Yes'])
        self.sleap_file_conversion_cb.setStyleSheet('QComboBox { width: 80px; }')
        self.sleap_file_conversion_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='sleap_file_conversion_cb_bool'))
        self.sleap_file_conversion_cb.move(column_three_x2, 70)

        anipose_calibration_cb_label = QLabel('Conduct AP calibration:', self.ProcessSettings)
        anipose_calibration_cb_label.setFont(QFont(self.font_id, 11))
        anipose_calibration_cb_label.setStyleSheet('QLabel { color: #F58025; font-weight: bold;}')
        anipose_calibration_cb_label.move(column_three_x1, 100)
        self.anipose_calibration_cb = QComboBox(self.ProcessSettings)
        self.anipose_calibration_cb.addItems(['No', 'Yes'])
        self.anipose_calibration_cb.setStyleSheet('QComboBox { width: 80px; }')
        self.anipose_calibration_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='anipose_calibration_cb_bool'))
        self.anipose_calibration_cb.move(column_three_x2, 100)

        board_provided_cb_label = QLabel('Calibration board provided:', self.ProcessSettings)
        board_provided_cb_label.setFont(QFont(self.font_id, 12))
        board_provided_cb_label.move(column_three_x1, 130)
        self.board_provided_cb = QComboBox(self.ProcessSettings)
        self.board_provided_cb.addItems(['No', 'Yes'])
        self.board_provided_cb.setStyleSheet('QComboBox { width: 80px; }')
        self.board_provided_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='board_provided_cb_bool'))
        self.board_provided_cb.move(column_three_x2, 130)

        anipose_triangulation_cb_label = QLabel('Conduct AP triangulation:', self.ProcessSettings)
        anipose_triangulation_cb_label.setFont(QFont(self.font_id, 11))
        anipose_triangulation_cb_label.setStyleSheet('QLabel { color: #F58025; font-weight: bold;}')
        anipose_triangulation_cb_label.move(column_three_x1, 160)
        self.anipose_triangulation_cb = QComboBox(self.ProcessSettings)
        self.anipose_triangulation_cb.addItems(['No', 'Yes'])
        self.anipose_triangulation_cb.setStyleSheet('QComboBox { width: 80px; }')
        self.anipose_triangulation_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='anipose_triangulation_cb_bool'))
        self.anipose_triangulation_cb.move(column_three_x2, 160)

        triangulate_arena_points_cb_label = QLabel('Triangulate arena nodes:', self.ProcessSettings)
        triangulate_arena_points_cb_label.setFont(QFont(self.font_id, 12))
        triangulate_arena_points_cb_label.move(column_three_x1, 190)
        self.triangulate_arena_points_cb = QComboBox(self.ProcessSettings)
        self.triangulate_arena_points_cb.addItems(['No', 'Yes'])
        self.triangulate_arena_points_cb.setStyleSheet('QComboBox { width: 80px; }')
        self.triangulate_arena_points_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='triangulate_arena_points_cb_bool'))
        self.triangulate_arena_points_cb.move(column_three_x2, 190)

        display_progress_cb_label = QLabel('Display progress:', self.ProcessSettings)
        display_progress_cb_label.setFont(QFont(self.font_id, 12))
        display_progress_cb_label.move(column_three_x1, 220)
        self.display_progress_cb = QComboBox(self.ProcessSettings)
        self.display_progress_cb.addItems(['Yes', 'No'])
        self.display_progress_cb.setStyleSheet('QComboBox { width: 80px; }')
        self.display_progress_cb.activated.connect(partial(self._combo_box_prior_true, variable_id='display_progress_cb_bool'))
        self.display_progress_cb.move(column_three_x2, 220)

        frame_restriction_label = QLabel('Frame restriction:', self.ProcessSettings)
        frame_restriction_label.setFont(QFont(self.font_id, 12))
        frame_restriction_label.move(column_three_x1, 250)
        frame_restriction_input = '' if self.processing_input_dict['anipose_operations']['ConvertTo3D']['conduct_anipose_triangulation']['frame_restriction'] is None else ','.join([str(x) for x in self.processing_input_dict['anipose_operations']['ConvertTo3D']['conduct_anipose_triangulation']['frame_restriction']])
        self.frame_restriction = QLineEdit(frame_restriction_input, self.ProcessSettings)
        self.frame_restriction.setFont(QFont(self.font_id, 10))
        self.frame_restriction.setStyleSheet('QLineEdit { width: 108px; }')
        self.frame_restriction.move(column_three_x2, 250)

        excluded_views_label = QLabel('Excluded camera views:', self.ProcessSettings)
        excluded_views_label.setFont(QFont(self.font_id, 12))
        excluded_views_label.move(column_three_x1, 280)
        self.excluded_views = QLineEdit(','.join([str(x) for x in self.processing_input_dict['anipose_operations']['ConvertTo3D']['conduct_anipose_triangulation']['excluded_views']]), self.ProcessSettings)
        self.excluded_views.setFont(QFont(self.font_id, 10))
        self.excluded_views.setStyleSheet('QLineEdit { width: 108px; }')
        self.excluded_views.move(column_three_x2, 280)

        ransac_cb_label = QLabel('Ransac:', self.ProcessSettings)
        ransac_cb_label.setFont(QFont(self.font_id, 12))
        ransac_cb_label.move(column_three_x1, 310)
        self.ransac_cb = QComboBox(self.ProcessSettings)
        self.ransac_cb.addItems(['No', 'Yes'])
        self.ransac_cb.setStyleSheet('QComboBox { width: 80px; }')
        self.ransac_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='ransac_cb_bool'))
        self.ransac_cb.move(column_three_x2, 310)

        rigid_body_constraints_label = QLabel('Rigid body constraints:', self.ProcessSettings)
        rigid_body_constraints_label.setFont(QFont(self.font_id, 12))
        rigid_body_constraints_label.move(column_three_x1, 340)
        self.rigid_body_constraints = QLineEdit('', self.ProcessSettings)
        self.rigid_body_constraints.setFont(QFont(self.font_id, 10))
        self.rigid_body_constraints.setStyleSheet('QLineEdit { width: 108px; }')
        self.rigid_body_constraints.move(column_three_x2, 340)

        weak_body_constraints_label = QLabel('Weak body constraints:', self.ProcessSettings)
        weak_body_constraints_label.setFont(QFont(self.font_id, 12))
        weak_body_constraints_label.move(column_three_x1, 370)
        self.weak_body_constraints = QLineEdit('', self.ProcessSettings)
        self.weak_body_constraints.setFont(QFont(self.font_id, 10))
        self.weak_body_constraints.setStyleSheet('QLineEdit { width: 108px; }')
        self.weak_body_constraints.move(column_three_x2, 370)

        smooth_scale_label = QLabel('Smoothing scale:', self.ProcessSettings)
        smooth_scale_label.setFont(QFont(self.font_id, 12))
        smooth_scale_label.move(column_three_x1, 400)
        self.smooth_scale = QLineEdit(f"{self.processing_input_dict['anipose_operations']['ConvertTo3D']['conduct_anipose_triangulation']['smooth_scale']}", self.ProcessSettings)
        self.smooth_scale.setFont(QFont(self.font_id, 10))
        self.smooth_scale.setStyleSheet('QLineEdit { width: 108px; }')
        self.smooth_scale.move(column_three_x2, 400)

        weight_rigid_label = QLabel('Rigid constraints weight:', self.ProcessSettings)
        weight_rigid_label.setFont(QFont(self.font_id, 12))
        weight_rigid_label.move(column_three_x1, 430)
        self.weight_rigid = QLineEdit(f"{self.processing_input_dict['anipose_operations']['ConvertTo3D']['conduct_anipose_triangulation']['weight_rigid']}", self.ProcessSettings)
        self.weight_rigid.setFont(QFont(self.font_id, 10))
        self.weight_rigid.setStyleSheet('QLineEdit { width: 108px; }')
        self.weight_rigid.move(column_three_x2, 430)

        weight_weak_label = QLabel('Weak constraints weight:', self.ProcessSettings)
        weight_weak_label.setFont(QFont(self.font_id, 12))
        weight_weak_label.move(column_three_x1, 460)
        self.weight_weak = QLineEdit(f"{self.processing_input_dict['anipose_operations']['ConvertTo3D']['conduct_anipose_triangulation']['weight_weak']}", self.ProcessSettings)
        self.weight_weak.setFont(QFont(self.font_id, 10))
        self.weight_weak.setStyleSheet('QLineEdit { width: 108px; }')
        self.weight_weak.move(column_three_x2, 460)

        reprojection_error_threshold_label = QLabel('Reproject error threshold:', self.ProcessSettings)
        reprojection_error_threshold_label.setFont(QFont(self.font_id, 12))
        reprojection_error_threshold_label.move(column_three_x1, 490)
        self.reprojection_error_threshold = QLineEdit(f"{self.processing_input_dict['anipose_operations']['ConvertTo3D']['conduct_anipose_triangulation']['reprojection_error_threshold']}", self.ProcessSettings)
        self.reprojection_error_threshold.setFont(QFont(self.font_id, 10))
        self.reprojection_error_threshold.setStyleSheet('QLineEdit { width: 108px; }')
        self.reprojection_error_threshold.move(column_three_x2, 490)

        regularization_function_label = QLabel('Regularization:', self.ProcessSettings)
        regularization_function_label.setFont(QFont(self.font_id, 12))
        regularization_function_label.move(column_three_x1, 520)
        self.regularization_function = QLineEdit(f"{self.processing_input_dict['anipose_operations']['ConvertTo3D']['conduct_anipose_triangulation']['regularization_function']}", self.ProcessSettings)
        self.regularization_function.setFont(QFont(self.font_id, 10))
        self.regularization_function.setStyleSheet('QLineEdit { width: 108px; }')
        self.regularization_function.move(column_three_x2, 520)

        n_deriv_smooth_label = QLabel('Derivation kernel order:', self.ProcessSettings)
        n_deriv_smooth_label.setFont(QFont(self.font_id, 12))
        n_deriv_smooth_label.move(column_three_x1, 550)
        self.n_deriv_smooth = QLineEdit(f"{self.processing_input_dict['anipose_operations']['ConvertTo3D']['conduct_anipose_triangulation']['n_deriv_smooth']}", self.ProcessSettings)
        self.n_deriv_smooth.setFont(QFont(self.font_id, 10))
        self.n_deriv_smooth.setStyleSheet('QLineEdit { width: 108px; }')
        self.n_deriv_smooth.move(column_three_x2, 550)

        translate_rotate_metric_label = QLabel('Conduct coordinate change:', self.ProcessSettings)
        translate_rotate_metric_label.setFont(QFont(self.font_id, 11))
        translate_rotate_metric_label.setStyleSheet('QLabel { color: #F58025; font-weight: bold;}')
        translate_rotate_metric_label.move(column_three_x1, 580)
        self.translate_rotate_metric_cb = QComboBox(self.ProcessSettings)
        self.translate_rotate_metric_cb.addItems(['No', 'Yes'])
        self.translate_rotate_metric_cb.setStyleSheet('QComboBox { width: 80px; }')
        self.translate_rotate_metric_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='translate_rotate_metric_cb_bool'))
        self.translate_rotate_metric_cb.move(column_three_x2, 580)

        static_reference_len_label = QLabel('Static reference length (m):', self.ProcessSettings)
        static_reference_len_label.setFont(QFont(self.font_id, 12))
        static_reference_len_label.move(column_three_x1, 610)
        self.static_reference_len = QLineEdit(f"{self.processing_input_dict['anipose_operations']['ConvertTo3D']['translate_rotate_metric']['static_reference_len']}", self.ProcessSettings)
        self.static_reference_len.setFont(QFont(self.font_id, 10))
        self.static_reference_len.setStyleSheet('QLineEdit { width: 108px; }')
        self.static_reference_len.move(column_three_x2, 610)

        save_transformed_data_cb_label = QLabel('Save transformation type:', self.ProcessSettings)
        save_transformed_data_cb_label.setFont(QFont(self.font_id, 12))
        save_transformed_data_cb_label.move(column_three_x1, 640)
        self.save_transformed_data_cb = QComboBox(self.ProcessSettings)
        self.save_transformed_data_cb.addItems(['animal', 'arena'])
        self.save_transformed_data_cb.setStyleSheet('QComboBox { width: 80px; }')
        self.save_transformed_data_cb.activated.connect(partial(self._combo_box_prior_transformed_tracking_data, variable_id='save_transformed_data'))
        self.save_transformed_data_cb.move(column_three_x2, 640)

        delete_original_h5_cb_label = QLabel('Delete original .h5:', self.ProcessSettings)
        delete_original_h5_cb_label.setFont(QFont(self.font_id, 12))
        delete_original_h5_cb_label.move(column_three_x1, 670)
        self.delete_original_h5_cb = QComboBox(self.ProcessSettings)
        self.delete_original_h5_cb.addItems(['Yes', 'No'])
        self.delete_original_h5_cb.setStyleSheet('QComboBox { width: 80px; }')
        self.delete_original_h5_cb.activated.connect(partial(self._combo_box_prior_true, variable_id='delete_original_h5_cb_bool'))
        self.delete_original_h5_cb.move(column_three_x2, 670)

        das_inference_cb_label = QLabel('Detect USVs:', self.ProcessSettings)
        das_inference_cb_label.setFont(QFont(self.font_id, 11))
        das_inference_cb_label.setStyleSheet('QLabel { color: #F58025; font-weight: bold;}')
        das_inference_cb_label.move(column_three_x1, 700)
        self.das_inference_cb = QComboBox(self.ProcessSettings)
        self.das_inference_cb.addItems(['No', 'Yes'])
        self.das_inference_cb.setStyleSheet('QComboBox { width: 80px; }')
        self.das_inference_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='das_inference_cb_bool'))
        self.das_inference_cb.move(column_three_x2, 700)

        segment_confidence_threshold_label = QLabel('DAS confidence threshold:', self.ProcessSettings)
        segment_confidence_threshold_label.setFont(QFont(self.font_id, 12))
        segment_confidence_threshold_label.move(column_three_x1, 730)
        self.segment_confidence_threshold = QLineEdit(f"{self.processing_input_dict['usv_inference']['FindMouseVocalizations']['das_command_line_inference']['segment_confidence_threshold']}", self.ProcessSettings)
        self.segment_confidence_threshold.setFont(QFont(self.font_id, 10))
        self.segment_confidence_threshold.setStyleSheet('QLineEdit { width: 108px; }')
        self.segment_confidence_threshold.move(column_three_x2, 730)

        segment_minlen_label = QLabel('USV min duration (s):', self.ProcessSettings)
        segment_minlen_label.setFont(QFont(self.font_id, 12))
        segment_minlen_label.move(column_three_x1, 760)
        self.segment_minlen = QLineEdit(f"{self.processing_input_dict['usv_inference']['FindMouseVocalizations']['das_command_line_inference']['segment_minlen']}", self.ProcessSettings)
        self.segment_minlen.setFont(QFont(self.font_id, 10))
        self.segment_minlen.setStyleSheet('QLineEdit { width: 108px; }')
        self.segment_minlen.move(column_three_x2, 760)

        segment_fillgap_label = QLabel('Fill gaps shorter than (s):', self.ProcessSettings)
        segment_fillgap_label.setFont(QFont(self.font_id, 12))
        segment_fillgap_label.move(column_three_x1, 790)
        self.segment_fillgap = QLineEdit(f"{self.processing_input_dict['usv_inference']['FindMouseVocalizations']['das_command_line_inference']['segment_fillgap']}", self.ProcessSettings)
        self.segment_fillgap.setFont(QFont(self.font_id, 10))
        self.segment_fillgap.setStyleSheet('QLineEdit { width: 108px; }')
        self.segment_fillgap.move(column_three_x2, 790)

        das_output_type_label = QLabel('Inference output file type:', self.ProcessSettings)
        das_output_type_label.setFont(QFont(self.font_id, 12))
        das_output_type_label.move(column_three_x1, 820)
        self.das_output_type = QLineEdit(f"{self.processing_input_dict['usv_inference']['FindMouseVocalizations']['das_command_line_inference']['output_file_type']}", self.ProcessSettings)
        self.das_output_type.setFont(QFont(self.font_id, 10))
        self.das_output_type.setStyleSheet('QLineEdit { width: 108px; }')
        self.das_output_type.move(column_three_x2, 820)

        das_summary_cb_label = QLabel('Summarize DAS output:', self.ProcessSettings)
        das_summary_cb_label.setFont(QFont(self.font_id, 11))
        das_summary_cb_label.setStyleSheet('QLabel { color: #F58025; font-weight: bold;}')
        das_summary_cb_label.move(column_three_x1, 850)
        self.das_summary_cb = QComboBox(self.ProcessSettings)
        self.das_summary_cb.addItems(['No', 'Yes'])
        self.das_summary_cb.setStyleSheet('QComboBox { width: 80px; }')
        self.das_summary_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='das_summary_cb_bool'))
        self.das_summary_cb.move(column_three_x2, 850)

        self._create_buttons_process(seq=0, class_option=self.ProcessSettings,
                                     button_pos_y=record_four_y - 35, next_button_x_pos=record_four_x - 100)

    def process_two(self):
        self.ConductProcess = ConductProcess(self)
        self.setWindowTitle(f'{app_name} (Conduct Processing)')
        self.setCentralWidget(self.ConductProcess)
        record_four_x, record_four_y = (870, 1000)
        self.setFixedSize(record_four_x, record_four_y)

        self.txt_edit_process = QPlainTextEdit(self.ConductProcess)
        self.txt_edit_process.move(5, 5)
        self.txt_edit_process.setFixedSize(855, 940)
        self.txt_edit_process.setReadOnly(True)

        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '_config/behavioral_experiments_settings.toml'), 'w') as updated_toml_file:
            toml.dump(self.exp_settings_dict, updated_toml_file)

        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '_parameter_settings/processing_settings.json'), 'w') as processing_settings_file:
            json.dump(self.processing_input_dict, processing_settings_file, indent=2)

        self.run_processing = Stylist(message_output=self._process_message,
                                      input_parameter_dict=self.processing_input_dict,
                                      root_directories=self.processing_input_dict['preprocess_data']['root_directories'],
                                      exp_settings_dict=self.exp_settings_dict)

        self._create_buttons_process(seq=1, class_option=self.ConductProcess,
                                     button_pos_y=record_four_y - 35, next_button_x_pos=record_four_x - 100)


    def _save_process_labels_func(self):
        qlabel_strings = ['conversion_target_file', 'constant_rate_factor', 'encoding_preset', 'ch_receiving_input',
                          'a_ch_receiving_input', 'pc_usage_process', 'min_spike_num', 'phidget_extra_data_camera',
                          'npx_ms_divergence_tolerance', 'hpss_power', 'sleap_conda', 'n_deriv_smooth',
                          'static_reference_len', 'das_conda', 'das_model_base', 'das_output_type', 'smooth_scale',
                          'weight_rigid', 'weight_weak', 'reprojection_error_threshold', 'regularization_function',
                          'segment_confidence_threshold', 'segment_minlen', 'segment_fillgap']
        lists_in_string = ['v_camera_serial_num', 'filter_dirs', 'concat_dirs', 'stft_window_hop', 'hpss_kernel_size',
                           'hpss_margin', 'filter_freq_bounds', 'frame_restriction', 'excluded_views',
                           'rigid_body_constraints', 'weak_body_constraints']

        for one_elem_str in qlabel_strings:
            if type(self.__dict__[one_elem_str]) != str:
                self.__dict__[one_elem_str] = self.__dict__[one_elem_str].text()

        for one_elem_lst in lists_in_string:
            if type(self.__dict__[one_elem_lst]) != str:
                self.__dict__[one_elem_lst] = self.__dict__[one_elem_lst].text().split(',')

        if os.name == 'nt':
            self.processing_dir_edit = self.processing_dir_edit.toPlainText().replace(os.sep, '\\')
        else:
            self.processing_dir_edit = self.processing_dir_edit.toPlainText()

        if len(self.processing_dir_edit) == 0:
            self.processing_dir_edit = []
        else:
            self.processing_dir_edit = self.processing_dir_edit.split('\n')

        if len(self.pc_usage_process) == 0:
            self.pc_usage_process = []
        else:
            self.pc_usage_process = self.pc_usage_process.split(',')

        if not self.inference_root_dir_btn_clicked_flag:
            self.processing_input_dict['prepare_cluster_job']['inference_root_dir'] = self.inference_root_dir_edit.text()

        if not self.centroid_model_btn_clicked_flag:
            self.processing_input_dict['prepare_cluster_job']['centroid_model_path'] = self.centroid_model_edit.text()

        if not self.centered_instance_btn_btn_clicked_flag:
            self.processing_input_dict['prepare_cluster_job']['centered_instance_model_path'] = self.centered_instance_model_edit.text()

        if not self.calibration_file_loc_btn_clicked_flag:
            self.processing_input_dict['anipose_operations']['ConvertTo3D']['conduct_anipose_triangulation']['calibration_file_loc'] = self.calibration_file_loc_edit.text()
            self.processing_input_dict['anipose_operations']['ConvertTo3D']['translate_rotate_metric']['original_arena_file_loc'] = self.calibration_file_loc_edit.text()

        if not self.das_model_dir_btn_clicked_flag:
            self.processing_input_dict['usv_inference']['FindMouseVocalizations']['das_command_line_inference']['model_directory'] = self.das_model_dir_edit.text()

        self.processing_input_dict['synchronize_files']['Synchronizer']['crop_wav_files_to_video']['device_receiving_input'] = str(getattr(self, 'device_receiving_input'))
        self.device_receiving_input = 'm'

        self.processing_input_dict['send_email']['Messenger']['processing_pc_choice'] = str(getattr(self, 'processing_pc_choice'))

        self.processing_input_dict['synchronize_files']['Synchronizer']['validate_ephys_video_sync']['npx_file_type'] = str(getattr(self, 'npx_file_type'))
        self.npx_file_type = 'ap'

        self.processing_input_dict['modify_files']['Operator']['rectify_video_fps']['conversion_target_file'] = self.conversion_target_file
        self.processing_input_dict['modify_files']['Operator']['rectify_video_fps']['constant_rate_factor'] = int(round(ast.literal_eval(self.constant_rate_factor)))
        self.processing_input_dict['modify_files']['Operator']['rectify_video_fps']['encoding_preset'] = self.encoding_preset
        self.processing_input_dict['synchronize_files']['Synchronizer']['crop_wav_files_to_video']['ch_receiving_input'] = int(ast.literal_eval(self.ch_receiving_input))
        self.processing_input_dict['modify_files']['Operator']['filter_audio_files']['filter_freq_bounds'] = [int(ast.literal_eval(freq_bound)) for freq_bound in self.filter_freq_bounds]
        self.processing_input_dict['modify_files']['Operator']['hpss_audio']['stft_window_length_hop_size'] = [int(ast.literal_eval(stft_value)) for stft_value in self.stft_window_hop]
        self.processing_input_dict['modify_files']['Operator']['hpss_audio']['kernel_size'] = tuple([int(ast.literal_eval(kernel_value)) for kernel_value in self.hpss_kernel_size])
        self.processing_input_dict['modify_files']['Operator']['hpss_audio']['hpss_power'] = float(ast.literal_eval(self.hpss_power))
        self.processing_input_dict['modify_files']['Operator']['hpss_audio']['margin'] = tuple([int(ast.literal_eval(margin_value)) for margin_value in self.hpss_margin])
        self.processing_input_dict['modify_files']['Operator']['get_spike_times']['min_spike_num'] = int(ast.literal_eval(self.min_spike_num))
        self.processing_input_dict['synchronize_files']['Synchronizer']['find_audio_sync_trains']['ch_receiving_input'] = int(ast.literal_eval(self.a_ch_receiving_input))
        self.processing_input_dict['synchronize_files']['Synchronizer']['validate_ephys_video_sync']['npx_ms_divergence_tolerance'] = float(ast.literal_eval(self.npx_ms_divergence_tolerance))
        self.processing_input_dict['extract_phidget_data']['Gatherer']['prepare_data_for_analyses']['extra_data_camera'] = self.phidget_extra_data_camera

        self.processing_input_dict['preprocess_data']['root_directories'] = self.processing_dir_edit
        self.processing_input_dict['send_email']['Messenger']['send_message']['receivers'] = self.pc_usage_process
        self.processing_input_dict['synchronize_files']['Synchronizer']['find_video_sync_trains']['camera_serial_num'] = self.v_camera_serial_num
        self.processing_input_dict['modify_files']['Operator']['filter_audio_files']['filter_dirs'] = self.filter_dirs
        self.processing_input_dict['modify_files']['Operator']['concatenate_audio_files']['concat_dirs'] = self.concat_dirs

        self.processing_input_dict['usv_inference']['FindMouseVocalizations']['das_command_line_inference']['das_conda_env_name'] = self.das_conda
        self.processing_input_dict['usv_inference']['FindMouseVocalizations']['das_command_line_inference']['model_name_base'] = self.das_model_base
        self.processing_input_dict['usv_inference']['FindMouseVocalizations']['das_command_line_inference']['output_file_type'] = self.das_output_type

        self.processing_input_dict['usv_inference']['FindMouseVocalizations']['das_command_line_inference']['segment_confidence_threshold'] = float(ast.literal_eval(self.segment_confidence_threshold))
        self.processing_input_dict['usv_inference']['FindMouseVocalizations']['das_command_line_inference']['segment_minlen'] = float(ast.literal_eval(self.segment_minlen))
        self.processing_input_dict['usv_inference']['FindMouseVocalizations']['das_command_line_inference']['segment_fillgap'] = float(ast.literal_eval(self.segment_fillgap))

        self.processing_input_dict['anipose_operations']['ConvertTo3D']['sleap_file_conversion']['sleap_conda_env_name'] = self.sleap_conda
        if self.frame_restriction == ['']:
            self.processing_input_dict['anipose_operations']['ConvertTo3D']['conduct_anipose_triangulation']['frame_restriction'] = None
        else:
            self.processing_input_dict['anipose_operations']['ConvertTo3D']['conduct_anipose_triangulation']['frame_restriction'] = [int(ast.literal_eval(fr_value)) for fr_value in self.frame_restriction]
        if self.excluded_views == ['']:
            self.processing_input_dict['anipose_operations']['ConvertTo3D']['conduct_anipose_triangulation']['excluded_views'] = []
        else:
            self.processing_input_dict['anipose_operations']['ConvertTo3D']['conduct_anipose_triangulation']['excluded_views'] = self.excluded_views
        if self.rigid_body_constraints == ['']:
            self.processing_input_dict['anipose_operations']['ConvertTo3D']['conduct_anipose_triangulation']['rigid_body_constraints'] = []
        else:
            self.processing_input_dict['anipose_operations']['ConvertTo3D']['conduct_anipose_triangulation']['rigid_body_constraints'] = list(ast.literal_eval(self.rigid_body_constraints))
        if self.weak_body_constraints == ['']:
            self.processing_input_dict['anipose_operations']['ConvertTo3D']['conduct_anipose_triangulation']['weak_body_constraints'] = []
        else:
            self.processing_input_dict['anipose_operations']['ConvertTo3D']['conduct_anipose_triangulation']['weak_body_constraints'] = list(ast.literal_eval(self.weak_body_constraints))

        self.processing_input_dict['anipose_operations']['ConvertTo3D']['conduct_anipose_triangulation']['smooth_scale'] = int(ast.literal_eval(self.smooth_scale))
        self.processing_input_dict['anipose_operations']['ConvertTo3D']['conduct_anipose_triangulation']['weight_rigid'] = int(ast.literal_eval(self.weight_rigid))
        self.processing_input_dict['anipose_operations']['ConvertTo3D']['conduct_anipose_triangulation']['weight_weak'] = int(ast.literal_eval(self.weight_weak))
        self.processing_input_dict['anipose_operations']['ConvertTo3D']['conduct_anipose_triangulation']['reprojection_error_threshold'] = int(ast.literal_eval(self.reprojection_error_threshold))
        self.processing_input_dict['anipose_operations']['ConvertTo3D']['conduct_anipose_triangulation']['regularization_function'] = self.regularization_function
        self.processing_input_dict['anipose_operations']['ConvertTo3D']['conduct_anipose_triangulation']['n_deriv_smooth'] = int(ast.literal_eval(self.n_deriv_smooth))
        self.processing_input_dict['anipose_operations']['ConvertTo3D']['translate_rotate_metric']['static_reference_len'] = float(ast.literal_eval(self.static_reference_len))

        self.processing_input_dict['processing_booleans']['conduct_video_concatenation'] = self.conduct_video_concatenation_cb_bool
        self.conduct_video_concatenation_cb_bool = False
        self.processing_input_dict['processing_booleans']['conduct_video_fps_change'] = self.conduct_video_fps_change_cb_bool
        self.conduct_video_fps_change_cb_bool = False
        self.processing_input_dict['modify_files']['Operator']['rectify_video_fps']['delete_old_file'] = self.delete_con_file_cb_bool
        self.delete_con_file_cb_bool = True
        self.processing_input_dict['processing_booleans']['conduct_audio_multichannel_to_single_ch'] = self.conduct_multichannel_conversion_cb_bool
        self.conduct_multichannel_conversion_cb_bool = False
        self.processing_input_dict['processing_booleans']['conduct_audio_cropping'] = self.crop_wav_cam_cb_bool
        self.crop_wav_cam_cb_bool = False
        self.processing_input_dict['processing_booleans']['conduct_audio_to_mmap'] = self.conc_audio_cb_bool
        self.conc_audio_cb_bool = False
        self.processing_input_dict['processing_booleans']['conduct_audio_filtering'] = self.filter_audio_cb_bool
        self.filter_audio_cb_bool = False
        self.processing_input_dict['processing_booleans']['conduct_hpss'] = self.conduct_hpss_cb_bool
        self.conduct_hpss_cb_bool = False
        self.processing_input_dict['processing_booleans']['conduct_audio_video_sync'] = self.conduct_sync_cb_bool
        self.conduct_sync_cb_bool = False
        self.processing_input_dict['processing_booleans']['conduct_ephys_video_sync'] = self.conduct_nv_sync_cb_bool
        self.conduct_nv_sync_cb_bool = False
        self.processing_input_dict['processing_booleans']['conduct_ephys_file_chaining'] = self.conduct_ephys_file_chaining_cb_bool
        self.conduct_ephys_file_chaining_cb_bool = False
        self.processing_input_dict['processing_booleans']['split_cluster_spikes'] = self.split_cluster_spikes_cb_bool
        self.split_cluster_spikes_cb_bool = False
        self.processing_input_dict['processing_booleans']['prepare_sleap_cluster'] = self.sleap_cluster_cb_bool
        self.sleap_cluster_cb_bool = False
        self.processing_input_dict['processing_booleans']['sleap_h5_conversion'] = self.sleap_file_conversion_cb_bool
        self.sleap_file_conversion_cb_bool = False
        self.processing_input_dict['processing_booleans']['anipose_calibration'] = self.anipose_calibration_cb_bool
        self.anipose_calibration_cb_bool = False
        self.processing_input_dict['anipose_operations']['ConvertTo3D']['conduct_anipose_calibration']['board_provided_bool'] = self.board_provided_cb_bool
        self.board_provided_cb_bool = False
        self.processing_input_dict['processing_booleans']['anipose_triangulation'] = self.anipose_triangulation_cb_bool
        self.anipose_triangulation_cb_bool = False
        self.processing_input_dict['anipose_operations']['ConvertTo3D']['conduct_anipose_triangulation']['triangulate_arena_points_bool'] = self.triangulate_arena_points_cb_bool
        self.triangulate_arena_points_cb_bool = False
        self.processing_input_dict['anipose_operations']['ConvertTo3D']['conduct_anipose_triangulation']['display_progress_bool'] = self.display_progress_cb_bool
        self.display_progress_cb_bool = True
        self.processing_input_dict['anipose_operations']['ConvertTo3D']['conduct_anipose_triangulation']['ransac_bool'] = self.ransac_cb_bool
        self.ransac_cb_bool = False
        self.processing_input_dict['processing_booleans']['anipose_trm'] = self.translate_rotate_metric_cb_bool
        self.translate_rotate_metric_cb_bool = False
        self.processing_input_dict['anipose_operations']['ConvertTo3D']['translate_rotate_metric']['save_transformed_data'] = self.save_transformed_data
        self.save_transformed_data = "animal"
        self.processing_input_dict['anipose_operations']['ConvertTo3D']['translate_rotate_metric']['delete_original_h5'] = self.delete_original_h5_cb_bool
        self.delete_original_h5_cb_bool = True
        self.processing_input_dict['processing_booleans']['das_infer'] = self.das_inference_cb_bool
        self.das_inference_cb_bool = False
        self.processing_input_dict['processing_booleans']['das_summarize'] = self.das_summary_cb_bool
        self.das_summary_cb_bool = False

    def _save_record_one_labels_func(self):
        if type(self.recording_files_destination_linux) != str:
            self.recording_files_destination_linux = self.recording_files_destination_linux.text().split(',')
        if type(self.recording_files_destination_windows) != str:
            self.recording_files_destination_windows = self.recording_files_destination_windows.text().split(',')
        if type(self.email_recipients) != str:
            self.email_recipients = self.email_recipients.text()
        if type(self.video_session_duration) != str:
            self.video_session_duration = self.video_session_duration.text()
        if type(self.calibration_session_duration) != str:
            self.calibration_session_duration = self.calibration_session_duration.text()
        if type(self.ethernet_network) != str:
            self.ethernet_network = self.ethernet_network.text()

        if len(self.email_recipients) == 0:
            self.email_recipients = []
        else:
            self.email_recipients = self.email_recipients.split(',')

        self.exp_settings_dict['recording_files_destination_linux'] = self.recording_files_destination_linux
        self.exp_settings_dict['recording_files_destination_win'] = self.recording_files_destination_windows
        self.exp_settings_dict['video_session_duration'] = ast.literal_eval(self.video_session_duration)
        self.exp_settings_dict['calibration_duration'] = ast.literal_eval(self.calibration_session_duration)
        self.exp_settings_dict['ethernet_network'] = self.ethernet_network

        self.exp_settings_dict['conduct_tracking_calibration'] = self.conduct_tracking_calibration_cb_bool
        self.exp_settings_dict['conduct_audio_recording'] = self.conduct_audio_cb_bool
        self.exp_settings_dict['disable_ethernet'] = self.disable_ethernet_cb_bool

        if not self.recorder_dir_btn_clicked_flag:
            self.exp_settings_dict['avisoft_recorder_exe'] = self.recorder_settings_edit.text()

        if not self.avisoft_base_dir_btn_clicked_flag:
            self.exp_settings_dict['avisoft_basedirectory'] = self.avisoft_base_edit.text()

        if not self.coolterm_base_dir_btn_clicked_flag:
            self.exp_settings_dict['coolterm_basedirectory'] = self.coolterm_base_edit.text()

    def _save_record_two_labels_func(self):
        for variable in self.default_audio_settings.keys():
            if variable in self.exp_settings_dict['audio'].keys():
                if variable == 'used_mics' or variable == 'cpu_affinity':
                    self.exp_settings_dict['audio'][f'{variable}'] = [int(x) for x in getattr(self, variable).text().split(',')]
                elif variable == 'cpu_priority':
                    self.exp_settings_dict['audio'][f'{variable}'] = getattr(self, variable).text()
                else:
                    self.exp_settings_dict['audio'][f'{variable}'] = ast.literal_eval(getattr(self, variable).text())
            elif variable in self.exp_settings_dict['audio']['general'].keys():
                self.exp_settings_dict['audio']['general'][f'{variable}'] = ast.literal_eval(getattr(self, variable).text())
            elif variable in self.exp_settings_dict['audio']['screen_position'].keys():
                self.exp_settings_dict['audio']['screen_position'][f'{variable}'] = ast.literal_eval(getattr(self, variable).text())
            elif variable in self.exp_settings_dict['audio']['devices'].keys():
                self.exp_settings_dict['audio']['devices'][f'{variable}'] = ast.literal_eval(getattr(self, variable).text())
            elif variable in self.exp_settings_dict['audio']['mics_config'].keys():
                if variable == 'ditctime':
                    self.exp_settings_dict['audio']['mics_config'][f'{variable}'] = getattr(self, variable).text()
                else:
                    self.exp_settings_dict['audio']['mics_config'][f'{variable}'] = ast.literal_eval(getattr(self, variable).text())
            elif variable in self.exp_settings_dict['audio']['monitor'].keys():
                self.exp_settings_dict['audio']['monitor'][f'{variable}'] = ast.literal_eval(getattr(self, variable).text())
            elif variable in self.exp_settings_dict['audio']['call'].keys():
                self.exp_settings_dict['audio']['call'][f'{variable}'] = ast.literal_eval(getattr(self, variable).text())

    def _save_record_three_labels_func(self):
        video_dict_keys = ['browser', 'expected_cameras', 'recording_codec', 'specific_camera_serial',
                           'institution_entry', 'laboratory_entry', 'experimenter_entry', 'mice_num_entry',
                           'species_entry', 'strain_entry', 'cage_entry', 'subject_entry', 'dob_entry',
                           'sex_entry', 'weight_entry', 'housing_entry', 'implant_entry', 'virus_entry', 'notes_entry']

        self.exp_settings_dict['video']['general']['monitor_recording'] = self.monitor_recording_cb_bool
        self.exp_settings_dict['video']['general']['monitor_specific_camera'] = self.monitor_specific_camera_cb_bool
        self.exp_settings_dict['video']['general']['delete_post_copy'] = self.delete_post_copy_cb_bool

        self.exp_settings_dict['video']['metadata']['vacant_arena'] = self.vacant_arena_cb_bool
        self.exp_settings_dict['video']['metadata']['ambient_light'] = self.ambient_light_cb_bool
        self.exp_settings_dict['video']['metadata']['record_brain'] = self.record_brain_cb_bool
        self.exp_settings_dict['video']['metadata']['usv_playback'] = self.usv_playback_cb_bool
        self.exp_settings_dict['video']['metadata']['chemogenetics'] = self.chemogenetics_cb_bool
        self.exp_settings_dict['video']['metadata']['optogenetics'] = self.optogenetics_cb_bool
        self.exp_settings_dict['video']['metadata']['brain_lesion'] = self.brain_lesion_cb_bool
        self.exp_settings_dict['video']['metadata']['devocalization'] = self.devocalization_cb_bool
        self.exp_settings_dict['video']['metadata']['female_urine'] = self.female_urine_cb_bool
        self.exp_settings_dict['video']['metadata']['female_bedding'] = self.female_bedding_cb_bool

        self.exp_settings_dict['video']['general']['recording_frame_rate'] = self.cameras_frame_rate.value()
        self.exp_settings_dict['video']['general']['calibration_frame_rate'] = self.calibration_frame_rate.value()

        self.exp_settings_dict['video']['cameras_config']['21372316']['exposure_time'] = self.exposure_time_21372316.value()
        self.exp_settings_dict['video']['cameras_config']['21372316']['gain'] = self.gain_21372316.value()
        self.exp_settings_dict['video']['cameras_config']['21372315']['exposure_time'] = self.exposure_time_21372315.value()
        self.exp_settings_dict['video']['cameras_config']['21372315']['gain'] = self.gain_21372315.value()
        self.exp_settings_dict['video']['cameras_config']['21369048']['exposure_time'] = self.exposure_time_21369048.value()
        self.exp_settings_dict['video']['cameras_config']['21369048']['gain'] = self.gain_21369048.value()
        self.exp_settings_dict['video']['cameras_config']['22085397']['exposure_time'] = self.exposure_time_22085397.value()
        self.exp_settings_dict['video']['cameras_config']['22085397']['gain'] = self.gain_22085397.value()
        self.exp_settings_dict['video']['cameras_config']['21241563']['exposure_time'] = self.exposure_time_21241563.value()
        self.exp_settings_dict['video']['cameras_config']['21241563']['gain'] = self.gain_21241563.value()

        for variable in video_dict_keys:
            if variable != 'expected_cameras':
                if variable == 'recording_codec':
                    self.exp_settings_dict['video']['general'][variable] = str(getattr(self, variable))
                elif variable == 'browser' or variable == 'specific_camera_serial':
                    self.exp_settings_dict['video']['general'][variable] = getattr(self, variable).text()
                elif variable == 'notes_entry':
                    self.exp_settings_dict['video']['metadata'][variable[:-6]] = getattr(self, variable).toPlainText()
                else:
                    self.exp_settings_dict['video']['metadata'][variable[:-6]] = getattr(self, variable).text()
            else:
                self.expected_cameras = self.expected_cameras.text()
                self.exp_settings_dict['video']['general'][variable] = self.expected_cameras.split(',')

    def _save_variables_based_on_exp_id(self):

        self.config_dir_global = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_config')

        if platform.system() == 'Windows':
            self.das_model_dir_global = f'F:\\{self.exp_id}\\DAS\\model_2024-03-25'
            self.sleap_inference_dir_global = f'F:\\{self.exp_id}\\SLEAP\\inference'
        elif platform.system() == 'Linux':
            self.das_model_dir_global = f'/mnt/falkner/{self.exp_id}/DAS/model_2024-03-25'
            self.sleap_inference_dir_global = f'/mnt/falkner/{self.exp_id}/SLEAP/inference'
        else:
            self.das_model_dir_global = f'/Volumes/falkner/{self.exp_id}/DAS/model_2024-03-25'
            self.sleap_inference_dir_global = f'/Volumes/falkner/{self.exp_id}/SLEAP/inference'

        self.avisoft_rec_dir_global = self.exp_settings_dict['avisoft_recorder_exe']
        self.avisoft_base_dir_global = self.exp_settings_dict['avisoft_basedirectory']
        self.coolterm_base_dir_global = self.exp_settings_dict['coolterm_basedirectory']
        self.destination_linux_global = ','.join(self.exp_settings_dict['recording_files_destination_linux'])
        self.destination_win_global =  ','.join(self.exp_settings_dict['recording_files_destination_win'])

        self.processing_input_dict['send_email']['Messenger']['experimenter'] = f'{self.exp_id}'

    def _combo_box_prior_transformed_tracking_data(self, index, variable_id=None):
        if index == 0:
            self.__dict__[variable_id] = 'animal'
        else:
            self.__dict__[variable_id] = 'arena'

    def _combo_box_prior_name(self, index, variable_id=None):
        for idx in range(len(self.exp_id_list)):
            if index == idx:
                self.__dict__[variable_id] = self.exp_id_list[idx]
                break

    def _combo_box_prior_codec(self, index, variable_id=None):
        for idx in range(len(self.recording_codec_list)):
            if index == idx:
                self.__dict__[variable_id] = self.recording_codec_list[idx]
                break

    def _combo_box_prior_processing_pc_choice(self, index, variable_id=None):
        for idx in range(len(self.loaded_processing_pc_list)):
            if index == idx:
                self.__dict__[variable_id] = self.loaded_processing_pc_list[idx]
                break

    def _combo_box_prior_audio_device_camera_input(self, index, variable_id=None):
        if index == 0:
            self.__dict__[variable_id] = 'm'
        elif index == 1:
            self.__dict__[variable_id] = 's'
        else:
            self.__dict__[variable_id] = 'both'

    def _combo_box_prior_npx_file_type(self, index, variable_id=None):
        if index == 0:
            self.__dict__[variable_id] = 'ap'
        else:
            self.__dict__[variable_id] = 'lf'

    def _combo_box_prior_true(self, index, variable_id=None):
        if index == 1:
            self.__dict__[variable_id] = False
        else:
            self.__dict__[variable_id] = True

    def _combo_box_prior_false(self, index, variable_id=None):
        if index == 1:
            self.__dict__[variable_id] = True
        else:
            self.__dict__[variable_id] = False

    def _update_exposure_time_label(self, value, variable_id=None):
        self.__dict__[variable_id].setText(f'exp time ({str(value)} μs):')

    def _update_gain_label(self, value, variable_id=None):
        self.__dict__[variable_id].setText(f'digital gain ({str(value)} dB):')

    def _update_fr_label(self, value):
        self.fr_label.setText(f'Recording ({str(value)} fps):')

    def _update_cal_fr_label(self, value):
        self.cal_fr_label.setText(f'Calibration ({str(value)} fps):')

    def _create_sliders_general(self, camera_id=None, camera_color=None, y_start=None):

        specific_camera_label = QLabel(f'Camera {camera_id} ({camera_color})', self.VideoSettings)
        specific_camera_label.setStyleSheet('QLabel { font-weight: bold;}')
        specific_camera_label.setFont(QFont(self.font_id, 12))
        specific_camera_label.move(5, y_start)

        self.__dict__[f'exposure_time_{camera_id}_label'] = QLabel('exp time (2500 μs)', self.VideoSettings)
        self.__dict__[f'exposure_time_{camera_id}_label'].setFixedWidth(150)
        self.__dict__[f'exposure_time_{camera_id}_label'].setFont(QFont(self.font_id, 10))
        self.__dict__[f'exposure_time_{camera_id}_label'].move(25, y_start+30)
        self.__dict__[f'exposure_time_{camera_id}'] = QSlider(Qt.Orientation.Horizontal, self.VideoSettings)
        self.__dict__[f'exposure_time_{camera_id}'].setFixedWidth(150)
        self.__dict__[f'exposure_time_{camera_id}'].setRange(500, 30000)
        self.__dict__[f'exposure_time_{camera_id}'].setValue(2500)
        self.__dict__[f'exposure_time_{camera_id}'].move(5, y_start+60)
        self.__dict__[f'exposure_time_{camera_id}'].valueChanged.connect(partial(self._update_exposure_time_label, variable_id=f'exposure_time_{camera_id}_label'))

        self.__dict__[f'gain_{camera_id}_label'] = QLabel('digital gain (0 dB)', self.VideoSettings)
        self.__dict__[f'gain_{camera_id}_label'].setFixedWidth(150)
        self.__dict__[f'gain_{camera_id}_label'].setFont(QFont(self.font_id, 10))
        self.__dict__[f'gain_{camera_id}_label'].move(200, y_start+30)
        self.__dict__[f'gain_{camera_id}'] = QSlider(Qt.Orientation.Horizontal, self.VideoSettings)
        self.__dict__[f'gain_{camera_id}'].setFixedWidth(150)
        self.__dict__[f'gain_{camera_id}'].setRange(0, 15)
        self.__dict__[f'gain_{camera_id}'].setValue(0)
        self.__dict__[f'gain_{camera_id}'].move(180, y_start+60)
        self.__dict__[f'gain_{camera_id}'].valueChanged.connect(partial(self._update_gain_label, variable_id=f'gain_{camera_id}_label'))


    def _create_buttons_main(self):
        self.button_map = {'Process': QPushButton(QIcon(process_icon), 'Process', self.Main),
                           'Record': QPushButton(QIcon(record_icon), 'Record', self.Main),
                           'Analyze': QPushButton(QIcon(analyze_icon), 'Analyze', self.Main),
                           'Visualize': QPushButton(QIcon(visualize_icon), 'Visualize', self.Main)}

        self.button_map['Record'].move(120, 370)
        self.button_map['Record'].setFont(QFont(self.font_id, 8))
        self.button_map['Record'].clicked.connect(self._save_variables_based_on_exp_id)
        self.button_map['Record'].clicked.connect(self.record_one)

        self.button_map['Process'].move(215, 370)
        self.button_map['Process'].setFont(QFont(self.font_id, 8))
        self.button_map['Process'].clicked.connect(self._save_variables_based_on_exp_id)
        self.button_map['Process'].clicked.connect(self.process_one)

        self.button_map['Analyze'].move(120, 405)
        self.button_map['Analyze'].setFont(QFont(self.font_id, 8))

        self.button_map['Visualize'].move(215, 405)
        self.button_map['Visualize'].setFont(QFont(self.font_id, 8))

    def _create_buttons_record(self, seq, class_option, button_pos_y, next_button_x_pos):
        if seq == 0:
            previous_win = self.main_window
            next_win_connect = [self._save_record_one_labels_func, self.record_two]
        elif seq == 1:
            previous_win = self.record_one
            next_win_connect = [self._save_record_two_labels_func, self.record_three]
        elif seq == 2:
            previous_win = self.record_two
            next_win_connect = [self._save_record_three_labels_func, self.main_window,
                                self.record_four]
        else:
            previous_win = self.record_three
            next_win_connect = []

        self.button_map = {'Previous': QPushButton(QIcon(previous_icon), 'Previous', class_option),
                           'Main': QPushButton(QIcon(main_icon), 'Main', class_option)}

        self.button_map['Previous'].move(5, button_pos_y)
        self.button_map['Previous'].setFont(QFont(self.font_id, 8))
        self.button_map['Previous'].clicked.connect(previous_win)

        self.button_map['Main'].move(100, button_pos_y)
        self.button_map['Main'].setFont(QFont(self.font_id, 8))
        self.button_map['Main'].clicked.connect(self.main_window)

        if len(next_win_connect) > 0:
            self.button_map['Next'] = QPushButton(QIcon(next_icon), 'Next', class_option)
            self.button_map['Next'].move(next_button_x_pos, button_pos_y)
            self.button_map['Next'].setFont(QFont(self.font_id, 8))
            for one_connection in next_win_connect:
                self.button_map['Next'].clicked.connect(one_connection)
        else:
            if self.exp_settings_dict['conduct_tracking_calibration']:
                self.button_map['Calibrate'] = QPushButton(QIcon(calibrate_icon), 'Calibrate', class_option)
                self.button_map['Calibrate'].move(next_button_x_pos-95, button_pos_y)
                self.button_map['Calibrate'].setFont(QFont(self.font_id, 8))
                self.button_map['Calibrate'].clicked.connect(self._disable_other_buttons)
                self.button_map['Calibrate'].clicked.connect(self._start_calibration)
                self.button_map['Calibrate'].clicked.connect(self._enable_other_buttons_post_cal)

            self.button_map['Record'] = QPushButton(QIcon(record_icon), 'Record', class_option)
            self.button_map['Record'].move(next_button_x_pos, button_pos_y)
            self.button_map['Record'].setFont(QFont(self.font_id, 8))
            self.button_map['Record'].clicked.connect(self._disable_other_buttons)
            self.button_map['Record'].clicked.connect(self._start_recording)
            self.button_map['Record'].clicked.connect(self._enable_other_buttons_post_rec)

    def _create_buttons_process(self, seq, class_option, button_pos_y, next_button_x_pos):
        if seq == 0:
            previous_win = self.main_window
            next_win_connect = [self._save_process_labels_func, self.process_two]
        else:
            previous_win = self.process_one

        self.button_map = {'Previous': QPushButton(QIcon(previous_icon), 'Previous', class_option),
                           'Main': QPushButton(QIcon(main_icon), 'Main', class_option)}

        self.button_map['Previous'].move(5, button_pos_y)
        self.button_map['Previous'].setFont(QFont(self.font_id, 8))
        self.button_map['Previous'].clicked.connect(previous_win)

        self.button_map['Main'].move(100, button_pos_y)
        self.button_map['Main'].setFont(QFont(self.font_id, 8))
        self.button_map['Main'].clicked.connect(self.main_window)

        if seq == 0:
            self.button_map['Next'] = QPushButton(QIcon(next_icon), 'Next', class_option)
            self.button_map['Next'].move(next_button_x_pos, button_pos_y)
            self.button_map['Next'].setFont(QFont(self.font_id, 8))
            for one_connection in next_win_connect:
                self.button_map['Next'].clicked.connect(one_connection)
        else:
            self.button_map['Process'] = QPushButton(QIcon(process_icon), 'Process', class_option)
            self.button_map['Process'].move(next_button_x_pos, button_pos_y)
            self.button_map['Process'].setFont(QFont(self.font_id, 8))
            self.button_map['Process'].clicked.connect(self._disable_process_buttons)
            self.button_map['Process'].clicked.connect(self._start_processing)
            self.button_map['Process'].clicked.connect(self._enable_process_buttons)

    def _start_processing(self):
        self.run_processing.prepare_data_for_analyses()

    def _start_calibration(self):
        self.run_exp.conduct_tracking_calibration()

    def _start_recording(self):
        self.run_exp.conduct_behavioral_recording()

    def _enable_process_buttons(self):
        self.button_map['Previous'].setEnabled(True)
        self.button_map['Main'].setEnabled(True)
        self.button_map['Process'].setEnabled(False)

    def _disable_process_buttons(self):
        self.button_map['Previous'].setEnabled(False)
        self.button_map['Main'].setEnabled(False)
        self.button_map['Process'].setEnabled(False)

    def _enable_other_buttons_post_cal(self):
        self.button_map['Main'].setEnabled(True)
        self.button_map['Record'].setEnabled(True)

    def _enable_other_buttons_post_rec(self):
        self.button_map['Main'].setEnabled(True)

    def _disable_other_buttons(self):
        self.button_map['Previous'].setEnabled(False)
        self.button_map['Main'].setEnabled(False)
        self.button_map['Record'].setEnabled(False)
        if self.exp_settings_dict['conduct_tracking_calibration']:
            self.button_map['Calibrate'].setEnabled(False)

    def _open_centroid_dialog(self):
        self.centroid_model_btn_clicked_flag = True
        centroid_dir_name = QFileDialog.getExistingDirectory(
            self,
            'Select SLEAP centroid model directory',
            f'')
        if centroid_dir_name:
            centroid_dir_name_path = Path(centroid_dir_name)
            self.centroid_model_edit.setText(str(centroid_dir_name_path))
            if os.name == 'nt':
                self.processing_input_dict['prepare_cluster_job']['centroid_model_path'] = str(centroid_dir_name_path).replace(os.sep, '\\') + '\\'
            else:
                self.processing_input_dict['prepare_cluster_job']['centroid_model_path'] = str(centroid_dir_name_path)
        else:
            self.processing_input_dict['prepare_cluster_job']['centroid_model_path'] = self.centroid_model_edit.text()

    def _open_centered_instance_dialog(self):
        self.centered_instance_btn_btn_clicked_flag = True
        centered_instance_dir_name = QFileDialog.getExistingDirectory(
            self,
            'Select SLEAP centered instance model directory',
            f'')
        if centered_instance_dir_name:
            centered_instance_dir_name_path = Path(centered_instance_dir_name)
            self.centered_instance_model_edit.setText(str(centered_instance_dir_name_path))
            if os.name == 'nt':
                self.processing_input_dict['prepare_cluster_job']['centered_instance_model_path'] = str(centered_instance_dir_name_path).replace(os.sep, '\\') + '\\'
            else:
                self.processing_input_dict['prepare_cluster_job']['centered_instance_model_path'] = str(centered_instance_dir_name_path)
        else:
            self.processing_input_dict['prepare_cluster_job']['centered_instance_model_path'] = self.centered_instance_model_edit.text()

    def _open_inference_root_dialog(self):
        self.inference_root_dir_btn_clicked_flag = True
        inference_root_dir_name = QFileDialog.getExistingDirectory(
            self,
            'Select SLEAP inference directory',
            f'')
        if inference_root_dir_name:
            inference_root_dir_name_path = Path(inference_root_dir_name)
            self.inference_root_dir_edit.setText(str(inference_root_dir_name_path))
            if os.name == 'nt':
                self.processing_input_dict['prepare_cluster_job']['inference_root_dir'] = str(inference_root_dir_name_path).replace(os.sep, '\\') + '\\'
            else:
                self.processing_input_dict['prepare_cluster_job']['inference_root_dir'] = str(inference_root_dir_name_path)
        else:
            self.processing_input_dict['prepare_cluster_job']['inference_root_dir'] = self.inference_root_dir_edit.text()

    def _open_recorder_dialog(self):
        self.recorder_dir_btn_clicked_flag = True
        recorder_dir_name = QFileDialog.getExistingDirectory(
            self,
            'Select Avisoft Recorder USGH directory',
            f'{self.avisoft_rec_dir_global}')
        if recorder_dir_name:
            recorder_dir_name_path = Path(recorder_dir_name)
            self.recorder_settings_edit.setText(str(recorder_dir_name_path))
            if os.name == 'nt':
                self.exp_settings_dict['avisoft_recorder_exe'] = str(recorder_dir_name_path).replace(os.sep, '\\')
            else:
                self.exp_settings_dict['avisoft_recorder_exe'] = str(recorder_dir_name_path)
        else:
            self.exp_settings_dict['avisoft_recorder_exe'] = f'{self.avisoft_rec_dir_global}'

    def _open_avisoft_dialog(self):
        self.avisoft_base_dir_btn_clicked_flag = True
        avisoft_dir_name = QFileDialog.getExistingDirectory(
            self,
            'Select Avisoft base directory',
            f'{self.avisoft_base_dir_global}')
        if avisoft_dir_name:
            avisoft_dir_name_path = Path(avisoft_dir_name)
            self.avisoft_base_edit.setText(str(avisoft_dir_name_path))
            if os.name == 'nt':
                self.exp_settings_dict['avisoft_basedirectory'] = str(avisoft_dir_name_path).replace(os.sep, '\\') + '\\'
            else:
                self.exp_settings_dict['avisoft_basedirectory'] = str(avisoft_dir_name_path)
        else:
            self.exp_settings_dict['avisoft_basedirectory'] = f'{self.avisoft_base_dir_global}'

    def _open_coolterm_dialog(self):
        self.coolterm_base_dir_btn_clicked_flag = True
        coolterm_dir_name = QFileDialog.getExistingDirectory(
            self,
            'Select Coolterm base directory',
            f'{self.coolterm_base_dir_global}')
        if coolterm_dir_name:
            coolterm_dir_name_path = Path(coolterm_dir_name)
            self.coolterm_base_edit.setText(str(coolterm_dir_name_path))
            if os.name == 'nt':
                self.exp_settings_dict['coolterm_basedirectory'] = str(coolterm_dir_name_path).replace(os.sep, '\\')
            else:
                self.exp_settings_dict['coolterm_basedirectory'] = str(coolterm_dir_name_path)
        else:
            self.exp_settings_dict['coolterm_basedirectory'] = f'{self.coolterm_base_dir_global}'

    def _open_anipose_calibration_dialog(self):
        self.calibration_file_loc_btn_clicked_flag = True
        anipose_cal_dir_name = QFileDialog.getExistingDirectory(
            self,
            'Select Anipose calibration root directory',
            '')
        if anipose_cal_dir_name:
            anipose_cal_dir_name_path = Path(anipose_cal_dir_name)
            self.calibration_file_loc_edit.setText(str(anipose_cal_dir_name_path))
            if os.name == 'nt':
                self.processing_input_dict['anipose_operations']['ConvertTo3D']['conduct_anipose_triangulation']['calibration_file_loc'] = str(anipose_cal_dir_name_path).replace(os.sep, '\\')
                self.processing_input_dict['anipose_operations']['ConvertTo3D']['translate_rotate_metric']['original_arena_file_loc'] = str(anipose_cal_dir_name_path).replace(os.sep, '\\')
            else:
                self.processing_input_dict['anipose_operations']['ConvertTo3D']['conduct_anipose_triangulation']['calibration_file_loc'] = str(anipose_cal_dir_name_path)
                self.processing_input_dict['anipose_operations']['ConvertTo3D']['translate_rotate_metric']['original_arena_file_loc'] = str(anipose_cal_dir_name_path)
        else:
            self.processing_input_dict['anipose_operations']['ConvertTo3D']['conduct_anipose_triangulation']['calibration_file_loc'] = self.calibration_file_loc_edit.text()
            self.processing_input_dict['anipose_operations']['ConvertTo3D']['translate_rotate_metric']['original_arena_file_loc'] = self.calibration_file_loc_edit.text()

    def _open_das_model_dialog(self):
        self.das_model_dir_btn_clicked_flag = True
        das_model_dir_name = QFileDialog.getExistingDirectory(
            self,
            'Select DAS model directory',
            '')
        if das_model_dir_name:
            das_model_dir_name_path = Path(das_model_dir_name)
            self.das_model_dir_edit.setText(str(das_model_dir_name_path))
            if os.name == 'nt':
                self.processing_input_dict['usv_inference']['FindMouseVocalizations']['das_command_line_inference']['model_directory'] = str(das_model_dir_name_path).replace(os.sep, '\\')
            else:
                self.processing_input_dict['usv_inference']['FindMouseVocalizations']['das_command_line_inference']['model_directory'] = str(das_model_dir_name_path)
        else:
            self.processing_input_dict['usv_inference']['FindMouseVocalizations']['das_command_line_inference']['model_directory'] = self.das_model_dir_edit.text()

    def _location_on_the_screen(self):
        top_left_point = QGuiApplication.primaryScreen().availableGeometry().topLeft()
        self.move(top_left_point)

    def _message(self, s):
        self.txt_edit.appendPlainText(s)

    def _process_message(self, s):
        self.txt_edit_process.appendPlainText(s)


def main():
    # Handle high resolution displays:
    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    usv_playpen_app = QApplication([])

    usv_playpen_app.setStyle('Fusion')
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '_config/gui_style_sheet.css') , 'r') as file:
        usv_playpen_app.setStyleSheet(file.read())

    usv_playpen_app.setWindowIcon(QIcon(lab_icon))

    _toml = toml.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), '_config/behavioral_experiments_settings.toml'))
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '_parameter_settings/processing_settings.json'), 'r') as process_json_file:
        _processing_settings = json.load(process_json_file)

    splash = QSplashScreen(QPixmap(splash_icon))
    progress_bar = QProgressBar(splash)
    progress_bar.setGeometry(340, 75, 250, 10)
    splash.show()
    for i in range(0, 101):
        progress_bar.setValue(i)
        t = time.time()
        while time.time() < t + 0.025:
            usv_playpen_app.processEvents()

    initial_values_dict = {'exp_id': _toml['video']['metadata']['experimenter'],
                           'conduct_audio_cb_bool': _toml['conduct_audio_recording'], 'conduct_tracking_calibration_cb_bool': _toml['conduct_tracking_calibration'],
                           'disable_ethernet_cb_bool': _toml['disable_ethernet'], 'monitor_recording_cb_bool': _toml['video']['general']['monitor_recording'],
                           'monitor_specific_camera_cb_bool': _toml['video']['general']['monitor_specific_camera'], 'delete_post_copy_cb_bool': _toml['video']['general']['delete_post_copy'],
                           'vacant_arena_cb_bool': _toml['video']['metadata']['vacant_arena'], 'ambient_light_cb_bool': _toml['video']['metadata']['ambient_light'], 'record_brain_cb_bool': _toml['video']['metadata']['record_brain'],
                           'usv_playback_cb_bool': _toml['video']['metadata']['usv_playback'], 'chemogenetics_cb_bool': _toml['video']['metadata']['chemogenetics'], 'optogenetics_cb_bool': _toml['video']['metadata']['optogenetics'],
                           'brain_lesion_cb_bool': _toml['video']['metadata']['brain_lesion'], 'devocalization_cb_bool': _toml['video']['metadata']['devocalization'], 'female_urine_cb_bool': _toml['video']['metadata']['female_urine'],
                           'female_bedding_cb_bool': _toml['video']['metadata']['female_bedding'], 'recording_codec': _toml['video']['general']['recording_codec'],
                           'inference_root_dir_btn_clicked_flag': False, 'centroid_model_btn_clicked_flag': False,  'centered_instance_btn_btn_clicked_flag': False,
                           'calibration_file_loc_btn_clicked_flag': False, 'das_model_dir_btn_clicked_flag': False,
                           'recorder_dir_btn_clicked_flag': False, 'avisoft_base_dir_btn_clicked_flag': False, 'coolterm_base_dir_btn_clicked_flag': False,
                           'processing_pc_choice': _processing_settings['send_email']['Messenger']['processing_pc_choice'],
                           'npx_file_type': 'ap', 'device_receiving_input': 'm', 'save_transformed_data': 'animal',
                           'conduct_video_concatenation_cb_bool': False, 'conduct_video_fps_change_cb_bool': False,
                           'conduct_multichannel_conversion_cb_bool': False, 'crop_wav_cam_cb_bool': False, 'conc_audio_cb_bool': False, 'filter_audio_cb_bool': False,
                           'conduct_sync_cb_bool': False, 'conduct_hpss_cb_bool': False, 'conduct_ephys_file_chaining_cb_bool': False,
                           'conduct_nv_sync_cb_bool': False, 'split_cluster_spikes_cb_bool': False, 'anipose_calibration_cb_bool': False,
                           'sleap_file_conversion_cb_bool': False, 'anipose_triangulation_cb_bool': False, 'translate_rotate_metric_cb_bool': False,
                           'sleap_cluster_cb_bool': False, 'das_inference_cb_bool': False, 'das_summary_cb_bool': False, 'delete_con_file_cb_bool': True,
                           'board_provided_cb_bool': False, 'triangulate_arena_points_cb_bool': False,
                           'display_progress_cb_bool': True, 'ransac_cb_bool': False, 'delete_original_h5_cb_bool': True}

    usv_playpen_window = USVPlaypenWindow(**initial_values_dict)

    splash.finish(usv_playpen_window)

    usv_playpen_window.show()

    sys.exit(usv_playpen_app.exec())


if __name__ == "__main__":
    main()
