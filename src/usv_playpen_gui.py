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
import toml
from PyQt6.QtCore import (
    Qt, QEvent
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
    QPushButton,
    QSlider,
    QSplashScreen,
    QTextEdit,
    QWidget,
)
from PyQt6.QtTest import QTest
from .analyze_data import Analyst
from .visualize_data import Visualizer
from .behavioral_experiments import ExperimentController
from .preprocess_data import Stylist

os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"

if os.name == 'nt':
    my_app_id = 'mycompany.myproduct.subproduct.version'
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(my_app_id)

app_name = 'USV Playpen v0.8.1'

basedir = os.path.dirname(__file__)
background_img = f'{basedir}{os.sep}img{os.sep}background_img.png'
gui_icon = f'{basedir}{os.sep}img{os.sep}gui_icon.png'
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
    def __init__(self, parent: QWidget = None) -> None:
        """
        Initializes the Main class.

        Parameters
        ----------
        parent (QWidget)
            Parent widget; defaults to None.

        Returns
        -------
        -------
        """
        super().__init__(parent)

    def paintEvent(self, event: QEvent = None) -> None:
        """
        Put background on GUI.

        Parameters
        ----------
        event (QEvent)
            Event to be painted.

        Returns
        -------
        -------
        """
        paint_main = QPainter(self)
        paint_main.drawPixmap(self.rect(), QPixmap(f'{background_img}'))
        QWidget.paintEvent(self, event)


class Record(QWidget):
    def __init__(self, parent: QWidget = Main) -> None:
        """
        Initializes the Record class.

        Parameters
        ----------
        parent (QWidget)
            Parent widget; defaults to Main.

        Returns
        -------
        -------
        """
        super(Record, self).__init__(parent)


class AudioSettings(QWidget):
    def __init__(self, parent: QWidget = Main) -> None:
        """
        Initializes the AudioSettings class.

        Parameters
        ----------
        parent (QWidget)
            Parent widget; defaults to Main.

        Returns
        -------
        -------
        """
        super(AudioSettings, self).__init__(parent)


class VideoSettings(QWidget):
    def __init__(self, parent: QWidget = Main) -> None:
        """
        Initializes the VideoSettings class.

        Parameters
        ----------
        parent (QWidget)
            Parent widget; defaults to Main.

        Returns
        -------
        -------
        """
        super(VideoSettings, self).__init__(parent)

class ConductRecording(QWidget):
    def __init__(self, parent: QWidget = Main) -> None:
        """
        Initializes the ConductRecording class.

        Parameters
        ----------
        parent (QWidget)
            Parent widget; defaults to Main.

        Returns
        -------
        -------
        """
        super(ConductRecording, self).__init__(parent)

class ProcessSettings(QWidget):
    def __init__(self, parent: QWidget = Main) -> None:
        """
        Initializes the ProcessSettings class.

        Parameters
        ----------
        parent (QWidget)
            Parent widget; defaults to Main.

        Returns
        -------
        -------
        """
        super(ProcessSettings, self).__init__(parent)

class ConductProcess(QWidget):
    def __init__(self, parent: QWidget = Main) -> None:
        """
        Initializes the ConductProcess class.

        Parameters
        ----------
        parent (QWidget)
            Parent widget; defaults to Main.

        Returns
        -------
        -------
        """
        super(ConductProcess, self).__init__(parent)

class AnalysesSettings(QWidget):
    def __init__(self, parent: QWidget = Main) -> None:
        """
        Initializes the AnalysesSettings class.

        Parameters
        ----------
        parent (QWidget)
            Parent widget; defaults to Main.

        Returns
        -------
        -------
        """
        super(AnalysesSettings, self).__init__(parent)

class ConductAnalyses(QWidget):
    def __init__(self, parent: QWidget = Main) -> None:
        """
        Initializes the ConductAnalyses class.

        Parameters
        ----------
        parent (QWidget)
            Parent widget; defaults to Main.

        Returns
        -------
        -------
        """
        super(ConductAnalyses, self).__init__(parent)

class VisualizationsSettings(QWidget):
    def __init__(self, parent: QWidget = Main) -> None:
        """
        Initializes the VisualizationsSettings class.

        Parameters
        ----------
        parent (QWidget)
            Parent widget; defaults to Main.

        Returns
        -------
        -------
        """
        super(VisualizationsSettings, self).__init__(parent)

class ConductVisualizations(QWidget):
    def __init__(self, parent: QWidget = Main) -> None:
        """
        Initializes the ConductVisualizations class.

        Parameters
        ----------
        parent (QWidget)
            Parent widget; defaults to Main.

        Returns
        -------
        -------
        """
        super(ConductVisualizations, self).__init__(parent)


# noinspection PyUnresolvedReferences,PyTypeChecker
class USVPlaypenWindow(QMainWindow):
    """ Main window of usv-playpen GUI """

    def __init__(self, **kwargs) -> None:
        """
        Initializes the USVPlaypenWindow class.

        Parameters
        ----------
        ----------

        Returns
        -------
        -------
        """
        super().__init__()

        font_file_loc = QFontDatabase.addApplicationFont(f'{basedir}{os.sep}fonts{os.sep}segoeui.ttf')
        self.font_id = QFontDatabase.applicationFontFamilies(font_file_loc)[0]

        if platform.system() == 'Darwin':
            self.font_size_increase = 4
        else:
            self.font_size_increase = 0

        for attr, value in kwargs.items():
            setattr(self, attr, value)

        self.boolean_list = ['Yes', 'No']

        self.main_window()

    def main_window(self) -> None:
        """
        Initializes the usv-playpen Main window.

        Parameters
        ----------
        ----------

        Returns
        -------
        -------
        """

        self.Main = Main(self)
        self.setCentralWidget(self.Main)
        self.setFixedSize(420, 500)
        self._location_on_the_screen()
        self.setWindowFlag(Qt.WindowType.WindowMaximizeButtonHint, on=False)
        self.setWindowTitle(f'{app_name}')

        self.exp_settings_dict = toml.load(Path(__file__).parent / '_config/behavioral_experiments_settings.toml')

        with open((Path(__file__).parent / '_parameter_settings/processing_settings.json'), 'r') as process_json_file:
            self.processing_input_dict = json.load(process_json_file)

        with open((Path(__file__).parent / '_parameter_settings/analyses_settings.json'), 'r') as analyses_json_file:
            self.analyses_input_dict = json.load(analyses_json_file)

        with open((Path(__file__).parent / '_parameter_settings/visualizations_settings.json'), 'r') as visualizations_json_file:
            self.visualizations_input_dict = json.load(visualizations_json_file)

        exp_id_label = QLabel('Experimenter:', self.Main)
        exp_id_label.setFont(QFont(self.font_id, 10+self.font_size_increase))
        exp_id_label.setStyleSheet('QLabel { font-weight: bold;}')
        exp_id_label.move(120, 329)
        self.exp_id_list = sorted(self.exp_settings_dict['experimenter_list'], key=lambda x: x == self.exp_id, reverse=True)
        self.exp_id_cb = QComboBox(self.Main)
        self.exp_id_cb.addItems(self.exp_id_list)
        self.exp_id_cb.setStyleSheet('QComboBox { width: 60px; height: 24px}')
        self.exp_id_cb.activated.connect(partial(self._combo_box_prior_name, variable_id='exp_id'))
        self.exp_id_cb.move(215, 325)

        self._create_buttons_main()

    def record_one(self) -> None:
        """
        Initializes the usv-playpen Record One window.

        Parameters
        ----------
        ----------

        Returns
        -------
        -------
        """

        self.Record = Record(self)
        self.setWindowTitle(f'{app_name} (Record > Select config directories and set basic parameters)')
        self.setCentralWidget(self.Record)
        record_one_x, record_one_y = (725, 510)
        self.setFixedSize(record_one_x, record_one_y)

        title_label = QLabel('Please select appropriate directories (with config files or executables in them)', self.Record)
        title_label.setFont(QFont(self.font_id, 13+self.font_size_increase))
        title_label.setStyleSheet('QLabel { font-weight: bold;}')
        title_label.move(5, 10)

        avisoft_exe_dir_label = QLabel('Avisoft Recorder directory:', self.Record)
        avisoft_exe_dir_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        avisoft_exe_dir_label.move(5, 40)
        self.recorder_settings_edit = QLineEdit(self.avisoft_rec_dir_global, self.Record)
        self.recorder_settings_edit.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.recorder_settings_edit.setStyleSheet('QLineEdit { width: 400px; }')
        self.recorder_settings_edit.move(220, 40)
        recorder_dir_btn = QPushButton('Browse', self.Record)
        recorder_dir_btn.setFont(QFont(self.font_id, 8+self.font_size_increase))
        recorder_dir_btn.move(625, 40)
        recorder_dir_btn.setStyleSheet('QPushButton { min-width: 64px; min-height: 12px; max-width: 64px; max-height: 13px; }')
        self.recorder_dir_btn_clicked_flag = False
        recorder_dir_btn.clicked.connect(self._open_recorder_dialog)

        avisoft_base_dir_label = QLabel('Avisoft base directory:', self.Record)
        avisoft_base_dir_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        avisoft_base_dir_label.move(5, 70)
        self.avisoft_base_edit = QLineEdit(self.avisoft_base_dir_global, self.Record)
        self.avisoft_base_edit.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.avisoft_base_edit.setStyleSheet('QLineEdit { width: 400px; }')
        self.avisoft_base_edit.move(220, 70)
        avisoft_base_dir_btn = QPushButton('Browse', self.Record)
        avisoft_base_dir_btn.setFont(QFont(self.font_id, 8+self.font_size_increase))
        avisoft_base_dir_btn.move(625, 70)
        avisoft_base_dir_btn.setStyleSheet('QPushButton { min-width: 64px; min-height: 12px; max-width: 64px; max-height: 13px; }')
        self.avisoft_base_dir_btn_clicked_flag = False
        avisoft_base_dir_btn.clicked.connect(self._open_avisoft_dialog)

        coolterm_base_dir_label = QLabel('CoolTerm base directory:', self.Record)
        coolterm_base_dir_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        coolterm_base_dir_label.move(5, 100)
        self.coolterm_base_edit = QLineEdit(self.coolterm_base_dir_global, self.Record)
        self.coolterm_base_edit.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.coolterm_base_edit.setStyleSheet('QLineEdit { width: 400px; }')
        self.coolterm_base_edit.move(220, 100)
        coolterm_base_dir_btn = QPushButton('Browse', self.Record)
        coolterm_base_dir_btn.setFont(QFont(self.font_id, 8+self.font_size_increase))
        coolterm_base_dir_btn.move(625, 100)
        coolterm_base_dir_btn.setStyleSheet('QPushButton { min-width: 64px; min-height: 12px; max-width: 64px; max-height: 13px; }')
        self.coolterm_base_dir_btn_clicked_flag = False
        coolterm_base_dir_btn.clicked.connect(self._open_coolterm_dialog)

        # recording files destination directories (across OS)
        recording_files_destination_linux_label = QLabel('File destination(s) Linux:', self.Record)
        recording_files_destination_linux_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        recording_files_destination_linux_label.move(5, 130)
        self.recording_files_destination_linux = QLineEdit(self.destination_linux_global, self.Record)
        self.recording_files_destination_linux.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.recording_files_destination_linux.setStyleSheet('QLineEdit { width: 490px; }')
        self.recording_files_destination_linux.move(220, 130)

        recording_files_destination_windows_label = QLabel('File destination(s) Windows:', self.Record)
        recording_files_destination_windows_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        recording_files_destination_windows_label.move(5, 160)
        self.recording_files_destination_windows = QLineEdit(self.destination_win_global, self.Record)
        self.recording_files_destination_windows.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.recording_files_destination_windows.setStyleSheet('QLineEdit { width: 490px; }')
        self.recording_files_destination_windows.move(220, 160)

        # set main recording parameters
        parameters_label = QLabel('Please set main recording parameters', self.Record)
        parameters_label.setFont(QFont(self.font_id, 13+self.font_size_increase))
        parameters_label.setStyleSheet('QLabel { font-weight: bold;}')
        parameters_label.move(5, 200)

        conduct_audio_label = QLabel('Conduct AUDIO recording:', self.Record)
        conduct_audio_label.setFont(QFont(self.font_id, 11+self.font_size_increase))
        conduct_audio_label.setStyleSheet('QLabel { color: #F58025; font-weight: bold;}')
        conduct_audio_label.move(5, 235)
        self.conduct_audio_cb_list = [x for _, x in sorted(zip([self.exp_settings_dict['conduct_audio_recording'], not self.exp_settings_dict['conduct_audio_recording']], self.boolean_list), reverse=True)]
        self.conduct_audio_cb = QComboBox(self.Record)
        self.conduct_audio_cb.addItems(self.conduct_audio_cb_list)
        self.conduct_audio_cb.setStyleSheet('QComboBox { width: 465px; }')
        self.conduct_audio_cb.activated.connect(partial(self._combo_box_prior_true if self.conduct_audio_cb_list[0] == 'Yes' else self._combo_box_prior_false, variable_id='conduct_audio_cb_bool'))
        self.conduct_audio_cb.move(220, 235)

        conduct_tracking_cal_label = QLabel('Conduct VIDEO calibration:', self.Record)
        conduct_tracking_cal_label.setFont(QFont(self.font_id, 11+self.font_size_increase))
        conduct_tracking_cal_label.setStyleSheet('QLabel { color: #F58025; font-weight: bold;}')
        conduct_tracking_cal_label.move(5, 265)
        self.conduct_tracking_calibration_cb_list = [x for _, x in sorted(zip([self.exp_settings_dict['conduct_tracking_calibration'], not self.exp_settings_dict['conduct_tracking_calibration']], self.boolean_list), reverse=True)]
        self.conduct_tracking_calibration_cb = QComboBox(self.Record)
        self.conduct_tracking_calibration_cb.addItems(self.conduct_tracking_calibration_cb_list)
        self.conduct_tracking_calibration_cb.setStyleSheet('QComboBox { width: 465px; }')
        self.conduct_tracking_calibration_cb.activated.connect(partial(self._combo_box_prior_true if self.conduct_tracking_calibration_cb_list[0] == 'Yes' else self._combo_box_prior_false, variable_id='conduct_tracking_calibration_cb_bool'))
        self.conduct_tracking_calibration_cb.move(220, 265)

        disable_ethernet_label = QLabel('Disable ethernet connection:', self.Record)
        disable_ethernet_label.setFont(QFont(self.font_id, 11+self.font_size_increase))
        disable_ethernet_label.setStyleSheet('QLabel { color: #F58025; font-weight: bold;}')
        disable_ethernet_label.move(5, 295)
        self.disable_ethernet_cb_list = [x for _, x in sorted(zip([self.exp_settings_dict['disable_ethernet'], not self.exp_settings_dict['disable_ethernet']], self.boolean_list), reverse=True)]
        self.disable_ethernet_cb = QComboBox(self.Record)
        self.disable_ethernet_cb.addItems(self.disable_ethernet_cb_list)
        self.disable_ethernet_cb.setStyleSheet('QComboBox { width: 465px; }')
        self.disable_ethernet_cb.activated.connect(partial(self._combo_box_prior_true if self.disable_ethernet_cb_list[0] == 'Yes' else self._combo_box_prior_false, variable_id='disable_ethernet_cb_bool'))
        self.disable_ethernet_cb.move(220, 295)

        video_duration_label = QLabel('Video session duration (min):', self.Record)
        video_duration_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        video_duration_label.move(5, 325)
        self.video_session_duration = QLineEdit(f"{self.exp_settings_dict['video_session_duration']}", self.Record)
        self.video_session_duration.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.video_session_duration.setStyleSheet('QLineEdit { width: 490px; }')
        self.video_session_duration.move(220, 325)

        cal_duration_label = QLabel('Calibration duration (min):', self.Record)
        cal_duration_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        cal_duration_label.move(5, 355)
        self.calibration_session_duration = QLineEdit(f"{self.exp_settings_dict['calibration_duration']}", self.Record)
        self.calibration_session_duration.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.calibration_session_duration.setStyleSheet('QLineEdit { width: 490px; }')
        self.calibration_session_duration.move(220, 355)

        ethernet_network_label = QLabel('Ethernet network ID:', self.Record)
        ethernet_network_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        ethernet_network_label.move(5, 385)
        self.ethernet_network = QLineEdit(f"{self.exp_settings_dict['ethernet_network']}", self.Record)
        self.ethernet_network.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.ethernet_network.setStyleSheet('QLineEdit { width: 490px; }')
        self.ethernet_network.move(220, 385)

        email_notification_label = QLabel('Notify e-mail(s) of PC usage:', self.Record)
        email_notification_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        email_notification_label.move(5, 415)
        self.email_recipients = QLineEdit('', self.Record)
        self.email_recipients.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.email_recipients.setStyleSheet('QLineEdit { width: 490px; }')
        self.email_recipients.move(220, 415)

        self._create_buttons_record(seq=0, class_option=self.Record,
                                    button_pos_y=record_one_y-35, next_button_x_pos=record_one_x-100)

    def record_two(self) -> None:
        """
        Initializes the usv-playpen Record Two window.

        Parameters
        ----------
        ----------

        Returns
        -------
        -------
        """

        self.AudioSettings = AudioSettings(self)
        self.setWindowTitle(f'{app_name} (Record > Audio Settings)')
        self.setCentralWidget(self.AudioSettings)
        record_two_x, record_two_y = (875, 875)
        self.setFixedSize(record_two_x, record_two_y)

        gas_label = QLabel('General audio recording settings', self.AudioSettings)
        gas_label.setFont(QFont(self.font_id, 13+self.font_size_increase))
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
            setting_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
            setting_label.move(row_start_position_label, 45+(row_counter*30))
            setattr(self, audio_attr, QLineEdit(audio_value, self.AudioSettings))
            getattr(self, audio_attr).setFixedWidth(50)
            getattr(self, audio_attr).setFont(QFont(self.font_id, 10+self.font_size_increase))
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

    def record_three(self) -> None:
        """
        Initializes the usv-playpen Record Three window.

        Parameters
        ----------
        ----------

        Returns
        -------
        -------
        """
        self.VideoSettings = VideoSettings(self)
        self.setWindowTitle(f'{app_name} (Record > Video Settings)')
        self.setCentralWidget(self.VideoSettings)
        record_three_x, record_three_y = (980, 900)
        self.setFixedSize(record_three_x, record_three_y)

        gvs_label = QLabel('General video recording settings', self.VideoSettings)
        gvs_label.setFont(QFont(self.font_id, 13+self.font_size_increase))
        gvs_label.setStyleSheet('QLabel { font-weight: bold;}')
        gvs_label.move(5, 10)

        browser_label = QLabel('Browser:', self.VideoSettings)
        browser_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        browser_label.move(5, 40)
        self.browser = QLineEdit(self.exp_settings_dict['video']['general']['browser'], self.VideoSettings)
        self.browser.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.browser.setStyleSheet('QLineEdit { width: 300px; }')
        self.browser.move(160, 40)

        use_cam_label = QLabel('Camera(s) to use:', self.VideoSettings)
        use_cam_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        use_cam_label.move(5, 70)
        self.expected_cameras = QLineEdit(','.join(self.exp_settings_dict['video']['general']['expected_cameras']), self.VideoSettings)
        self.expected_cameras.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.expected_cameras.setStyleSheet('QLineEdit { width: 300px; }')
        self.expected_cameras.move(160, 70)

        """
        'nvenc-fast-yuv420_A' : '-preset','fast','-qmin','15','-qmax','15'
        'nvenc-fast-yuv420_B' : '-preset','fast','-qmin','15','-qmax','18'
        'nvenc-ll-yuv420'     : '-preset', 'lossless', '-pix_fmt', 'yuv420p'
        """

        rec_codec_label = QLabel('Recording codec:', self.VideoSettings)
        rec_codec_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        rec_codec_label.move(5, 100)
        self.recording_codec_list = sorted(['hq', 'hq-fast', 'mq', 'lq', 'nvenc-fast-yuv420_A',
                                            'nvenc-fast-yuv420_B','nvenc-ll-yuv420'], key=lambda x: x == self.recording_codec, reverse=True)
        self.recording_codec_cb = QComboBox(self.VideoSettings)
        self.recording_codec_cb.addItems(self.recording_codec_list)
        self.recording_codec_cb.setStyleSheet('QComboBox { width: 272px; }')
        self.recording_codec_cb.activated.connect(partial(self._combo_box_prior_codec, variable_id='recording_codec'))
        self.recording_codec_cb.move(160, 100)

        monitor_rec_label = QLabel('Monitor recording:', self.VideoSettings)
        monitor_rec_label.setFont(QFont(self.font_id, 11+self.font_size_increase))
        monitor_rec_label.setStyleSheet('QLabel { color: #F58025; font-weight: bold;}')
        monitor_rec_label.move(5, 130)
        self.monitor_recording_cb_list = [x for _, x in sorted(zip([self.exp_settings_dict['video']['general']['monitor_recording'], not self.exp_settings_dict['video']['general']['monitor_recording']], self.boolean_list), reverse=True)]
        self.monitor_recording_cb = QComboBox(self.VideoSettings)
        self.monitor_recording_cb.addItems(self.monitor_recording_cb_list)
        self.monitor_recording_cb.setStyleSheet('QComboBox { width: 272px; }')
        self.monitor_recording_cb.activated.connect(partial(self._combo_box_prior_true if self.monitor_recording_cb_list[0] == 'Yes' else self._combo_box_prior_false, variable_id='monitor_recording_cb_bool'))
        self.monitor_recording_cb.move(160, 130)

        monitor_cam_label = QLabel('Monitor ONE camera:', self.VideoSettings)
        monitor_cam_label.setFont(QFont(self.font_id, 11+self.font_size_increase))
        monitor_cam_label.setStyleSheet('QLabel { color: #F58025; font-weight: bold;}')
        monitor_cam_label.move(5, 160)
        self.monitor_specific_camera_cb_list = [x for _, x in sorted(zip([self.exp_settings_dict['video']['general']['monitor_specific_camera'], not self.exp_settings_dict['video']['general']['monitor_specific_camera']], self.boolean_list), reverse=True)]
        self.monitor_specific_camera_cb = QComboBox(self.VideoSettings)
        self.monitor_specific_camera_cb.addItems(self.monitor_specific_camera_cb_list)
        self.monitor_specific_camera_cb.setStyleSheet('QComboBox { width: 272px; }')
        self.monitor_specific_camera_cb.activated.connect(partial(self._combo_box_prior_true if self.monitor_specific_camera_cb_list[0] == 'Yes' else self._combo_box_prior_false, variable_id='monitor_specific_camera_cb_bool'))
        self.monitor_specific_camera_cb.move(160, 160)

        specific_camera_serial_label = QLabel('ONE camera serial:', self.VideoSettings)
        specific_camera_serial_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        specific_camera_serial_label.move(5, 190)
        self.specific_camera_serial = QLineEdit(self.exp_settings_dict['video']['general']['specific_camera_serial'], self.VideoSettings)
        self.specific_camera_serial.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.specific_camera_serial.setStyleSheet('QLineEdit { width: 300px; }')
        self.specific_camera_serial.move(160, 190)

        delete_post_copy_label = QLabel('Delete post copy:', self.VideoSettings)
        delete_post_copy_label.setFont(QFont(self.font_id, 11+self.font_size_increase))
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
        self.cal_fr_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        self.cal_fr_label.move(5, 250)
        self.calibration_frame_rate = QSlider(Qt.Orientation.Horizontal, self.VideoSettings)
        self.calibration_frame_rate.setFixedWidth(150)
        self.calibration_frame_rate.move(160, 255)
        self.calibration_frame_rate.setRange(10, 150)
        self.calibration_frame_rate.setValue(self.exp_settings_dict['video']['general']['calibration_frame_rate'])
        self.calibration_frame_rate.valueChanged.connect(self._update_cal_fr_label)

        self.fr_label = QLabel('Recording (150 fps):', self.VideoSettings)
        self.fr_label.setFixedWidth(150)
        self.fr_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        self.fr_label.move(5, 280)
        self.cameras_frame_rate = QSlider(Qt.Orientation.Horizontal, self.VideoSettings)
        self.cameras_frame_rate.setFixedWidth(150)
        self.cameras_frame_rate.move(160, 285)
        self.cameras_frame_rate.setRange(10, 150)
        self.cameras_frame_rate.setValue(self.exp_settings_dict['video']['general']['recording_frame_rate'])
        self.cameras_frame_rate.valueChanged.connect(self._update_fr_label)

        pcs_label = QLabel('Particular camera settings', self.VideoSettings)
        pcs_label.setFont(QFont(self.font_id, 13+self.font_size_increase))
        pcs_label.setStyleSheet('QLabel { font-weight: bold;}')
        pcs_label.move(5, 325)

        camera_colors_global = ['white', 'orange', 'red', 'cyan', 'yellow']
        for cam_idx, cam in enumerate(self.exp_settings_dict['video']['general']['expected_cameras']):
            self._create_sliders_general(camera_id=cam, camera_color=camera_colors_global[cam_idx], y_start=355+(cam_idx*90))

        vm_label = QLabel('Metadata', self.VideoSettings)
        vm_label.setFont(QFont(self.font_id, 13+self.font_size_increase))
        vm_label.setStyleSheet('QLabel { font-weight: bold;}')
        vm_label.move(495, 10)

        institution_label = QLabel('Institution:', self.VideoSettings)
        institution_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        institution_label.move(495, 40)
        self.institution_entry = QLineEdit(f"{self.exp_settings_dict['video']['metadata']['institution']}", self.VideoSettings)
        self.institution_entry.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.institution_entry.setStyleSheet('QLineEdit { width: 95px; }')
        self.institution_entry.move(630, 40)

        laboratory_label = QLabel('Laboratory:', self.VideoSettings)
        laboratory_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        laboratory_label.move(750, 40)
        self.laboratory_entry = QLineEdit(f"{self.exp_settings_dict['video']['metadata']['laboratory']}", self.VideoSettings)
        self.laboratory_entry.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.laboratory_entry.setStyleSheet('QLineEdit { width: 95px; }')
        self.laboratory_entry.move(875, 40)

        experimenter_label = QLabel('Experimenter:', self.VideoSettings)
        experimenter_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        experimenter_label.move(495, 70)
        self.experimenter_entry = QLineEdit(f'{self.exp_id}', self.VideoSettings)
        self.experimenter_entry.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.experimenter_entry.setStyleSheet('QLineEdit { width: 95px; }')
        self.experimenter_entry.move(630, 70)

        mice_num_label = QLabel('Animal count:', self.VideoSettings)
        mice_num_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        mice_num_label.move(750, 70)
        self.mice_num_entry = QLineEdit(f"{self.exp_settings_dict['video']['metadata']['mice_num']}", self.VideoSettings)
        self.mice_num_entry.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.mice_num_entry.setStyleSheet('QLineEdit { width: 95px; }')
        self.mice_num_entry.move(875, 70)

        vacant_arena_label = QLabel('Vacant arena:', self.VideoSettings)
        vacant_arena_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        vacant_arena_label.move(495, 100)
        self.vacant_arena_cb_list = [x for _, x in sorted(zip([self.exp_settings_dict['video']['metadata']['vacant_arena'], not self.exp_settings_dict['video']['metadata']['vacant_arena']], self.boolean_list), reverse=True)]
        self.vacant_arena_cb = QComboBox(self.VideoSettings)
        self.vacant_arena_cb.addItems(self.vacant_arena_cb_list)
        self.vacant_arena_cb.setStyleSheet('QComboBox { width: 68px; }')
        self.vacant_arena_cb.activated.connect(partial(self._combo_box_prior_true if self.vacant_arena_cb_list[0] == 'Yes' else self._combo_box_prior_false, variable_id='vacant_arena_cb_bool'))
        self.vacant_arena_cb.move(630, 100)

        ambient_light_label = QLabel('Ambient light:', self.VideoSettings)
        ambient_light_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        ambient_light_label.move(750, 100)
        self.ambient_light_cb_list = [x for _, x in sorted(zip([self.exp_settings_dict['video']['metadata']['ambient_light'], not self.exp_settings_dict['video']['metadata']['ambient_light']], self.boolean_list), reverse=True)]
        self.ambient_light_cb = QComboBox(self.VideoSettings)
        self.ambient_light_cb.addItems(self.ambient_light_cb_list)
        self.ambient_light_cb.setStyleSheet('QComboBox { width: 68px; }')
        self.ambient_light_cb.activated.connect(partial(self._combo_box_prior_true if self.ambient_light_cb_list[0] == 'Yes' else self._combo_box_prior_false, variable_id='ambient_light_cb_bool'))
        self.ambient_light_cb.move(875, 100)

        record_brain_label = QLabel('Record brain:', self.VideoSettings)
        record_brain_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        record_brain_label.move(495, 130)
        self.record_brain_cb_list = [x for _, x in sorted(zip([self.exp_settings_dict['video']['metadata']['record_brain'], not self.exp_settings_dict['video']['metadata']['record_brain']], self.boolean_list), reverse=True)]
        self.record_brain_cb = QComboBox(self.VideoSettings)
        self.record_brain_cb.addItems(self.record_brain_cb_list)
        self.record_brain_cb.setStyleSheet('QComboBox { width: 68px; }')
        self.record_brain_cb.activated.connect(partial(self._combo_box_prior_true if self.record_brain_cb_list[0] == 'Yes' else self._combo_box_prior_false, variable_id='record_brain_cb_bool'))
        self.record_brain_cb.move(630, 130)

        usv_playback_label = QLabel('USV playback:', self.VideoSettings)
        usv_playback_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        usv_playback_label.move(750, 130)
        self.usv_playback_cb_list = [x for _, x in sorted(zip([self.exp_settings_dict['video']['metadata']['usv_playback'], not self.exp_settings_dict['video']['metadata']['usv_playback']], self.boolean_list), reverse=True)]
        self.usv_playback_cb = QComboBox(self.VideoSettings)
        self.usv_playback_cb.addItems(self.usv_playback_cb_list)
        self.usv_playback_cb.setStyleSheet('QComboBox { width: 68px; }')
        self.usv_playback_cb.activated.connect(partial(self._combo_box_prior_true if self.usv_playback_cb_list[0] == 'Yes' else self._combo_box_prior_false, variable_id='usv_playback_cb_bool'))
        self.usv_playback_cb.move(875, 130)

        chemogenetics_label = QLabel('Chemogenetics:', self.VideoSettings)
        chemogenetics_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        chemogenetics_label.move(495, 160)
        self.chemogenetics_cb_list = [x for _, x in sorted(zip([self.exp_settings_dict['video']['metadata']['chemogenetics'], not self.exp_settings_dict['video']['metadata']['chemogenetics']], self.boolean_list), reverse=True)]
        self.chemogenetics_cb = QComboBox(self.VideoSettings)
        self.chemogenetics_cb.addItems(self.chemogenetics_cb_list)
        self.chemogenetics_cb.setStyleSheet('QComboBox { width: 68px; }')
        self.chemogenetics_cb.activated.connect(partial(self._combo_box_prior_true if self.chemogenetics_cb_list[0] == 'Yes' else self._combo_box_prior_false, variable_id='chemogenetics_cb_bool'))
        self.chemogenetics_cb.move(630, 160)

        optogenetics_label = QLabel('Optogenetics:', self.VideoSettings)
        optogenetics_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        optogenetics_label.move(750, 160)
        self.optogenetics_cb_list = [x for _, x in sorted(zip([self.exp_settings_dict['video']['metadata']['optogenetics'], not self.exp_settings_dict['video']['metadata']['optogenetics']], self.boolean_list), reverse=True)]
        self.optogenetics_cb = QComboBox(self.VideoSettings)
        self.optogenetics_cb.addItems(self.optogenetics_cb_list)
        self.optogenetics_cb.setStyleSheet('QComboBox { width: 68px; }')
        self.optogenetics_cb.activated.connect(partial(self._combo_box_prior_true if self.optogenetics_cb_list[0] == 'Yes' else self._combo_box_prior_false, variable_id='optogenetics_cb_bool'))
        self.optogenetics_cb.move(875, 160)

        brain_lesion_label = QLabel('Brain lesion:', self.VideoSettings)
        brain_lesion_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        brain_lesion_label.move(495, 190)
        self.brain_lesion_cb_list = [x for _, x in sorted(zip([self.exp_settings_dict['video']['metadata']['brain_lesion'], not self.exp_settings_dict['video']['metadata']['brain_lesion']], self.boolean_list), reverse=True)]
        self.brain_lesion_cb = QComboBox(self.VideoSettings)
        self.brain_lesion_cb.addItems(self.brain_lesion_cb_list)
        self.brain_lesion_cb.setStyleSheet('QComboBox { width: 68px; }')
        self.brain_lesion_cb.activated.connect(partial(self._combo_box_prior_true if self.brain_lesion_cb_list[0] == 'Yes' else self._combo_box_prior_false, variable_id='brain_lesion_cb_bool'))
        self.brain_lesion_cb.move(630, 190)

        devocalization_label = QLabel('Devocalization:', self.VideoSettings)
        devocalization_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        devocalization_label.move(750, 190)
        self.devocalization_cb_list = [x for _, x in sorted(zip([self.exp_settings_dict['video']['metadata']['devocalization'], not self.exp_settings_dict['video']['metadata']['devocalization']], self.boolean_list), reverse=True)]
        self.devocalization_cb = QComboBox(self.VideoSettings)
        self.devocalization_cb.addItems(self.devocalization_cb_list)
        self.devocalization_cb.setStyleSheet('QComboBox { width: 68px; }')
        self.devocalization_cb.activated.connect(partial(self._combo_box_prior_true if self.devocalization_cb_list[0] == 'Yes' else self._combo_box_prior_false, variable_id='devocalization_cb_bool'))
        self.devocalization_cb.move(875, 190)

        female_urine_label = QLabel('Female urine:', self.VideoSettings)
        female_urine_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        female_urine_label.move(495, 220)
        self.female_urine_cb_list = [x for _, x in sorted(zip([self.exp_settings_dict['video']['metadata']['female_urine'], not self.exp_settings_dict['video']['metadata']['female_urine']], self.boolean_list), reverse=True)]
        self.female_urine_cb = QComboBox(self.VideoSettings)
        self.female_urine_cb.addItems(self.female_urine_cb_list)
        self.female_urine_cb.setStyleSheet('QComboBox { width: 68px; }')
        self.female_urine_cb.activated.connect(partial(self._combo_box_prior_true if self.female_urine_cb_list[0] == 'Yes' else self._combo_box_prior_false, variable_id='female_urine_cb_bool'))
        self.female_urine_cb.move(630, 220)

        female_bedding_label = QLabel('Female bedding:', self.VideoSettings)
        female_bedding_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        female_bedding_label.move(750, 220)
        self.female_bedding_cb_list = [x for _, x in sorted(zip([self.exp_settings_dict['video']['metadata']['female_bedding'], not self.exp_settings_dict['video']['metadata']['female_bedding']], self.boolean_list), reverse=True)]
        self.female_bedding_cb = QComboBox(self.VideoSettings)
        self.female_bedding_cb.addItems(self.female_bedding_cb_list)
        self.female_bedding_cb.setStyleSheet('QComboBox { width: 68px; }')
        self.female_bedding_cb.activated.connect(partial(self._combo_box_prior_true if self.female_bedding_cb_list[0] == 'Yes' else self._combo_box_prior_false, variable_id='female_bedding_cb_bool'))
        self.female_bedding_cb.move(875, 220)

        species_label = QLabel('Species:', self.VideoSettings)
        species_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        species_label.move(495, 250)
        self.species_entry = QLineEdit(f"{self.exp_settings_dict['video']['metadata']['species']}", self.VideoSettings)
        self.species_entry.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.species_entry.setStyleSheet('QLineEdit { width: 402px; }')
        self.species_entry.move(570, 250)

        strain_label = QLabel('Strain-GT:', self.VideoSettings)
        strain_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        strain_label.move(495, 280)
        self.strain_entry = QLineEdit(f"{self.exp_settings_dict['video']['metadata']['strain']}", self.VideoSettings)
        self.strain_entry.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.strain_entry.setStyleSheet('QLineEdit { width: 402px; }')
        self.strain_entry.move(570, 280)

        cage_label = QLabel('Cage:', self.VideoSettings)
        cage_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        cage_label.move(495, 310)
        self.cage_entry = QLineEdit(f"{self.exp_settings_dict['video']['metadata']['cage']}", self.VideoSettings)
        self.cage_entry.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.cage_entry.setStyleSheet('QLineEdit { width: 402px; }')
        self.cage_entry.move(570, 310)

        subject_label = QLabel('Subject:', self.VideoSettings)
        subject_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        subject_label.move(495, 340)
        self.subject_entry = QLineEdit(f"{self.exp_settings_dict['video']['metadata']['subject']}", self.VideoSettings)
        self.subject_entry.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.subject_entry.setStyleSheet('QLineEdit { width: 402px; }')
        self.subject_entry.move(570, 340)

        dob_label = QLabel('DOB:', self.VideoSettings)
        dob_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        dob_label.move(495, 370)
        self.dob_entry = QLineEdit(f"{self.exp_settings_dict['video']['metadata']['dob']}", self.VideoSettings)
        self.dob_entry.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.dob_entry.setStyleSheet('QLineEdit { width: 402px; }')
        self.dob_entry.move(570, 370)

        sex_label = QLabel('Sex:', self.VideoSettings)
        sex_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        sex_label.move(495, 400)
        self.sex_entry = QLineEdit(f"{self.exp_settings_dict['video']['metadata']['sex']}", self.VideoSettings)
        self.sex_entry.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.sex_entry.setStyleSheet('QLineEdit { width: 402px; }')
        self.sex_entry.move(570, 400)

        weight_label = QLabel('Weight:', self.VideoSettings)
        weight_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        weight_label.move(495, 430)
        self.weight_entry = QLineEdit(f"{self.exp_settings_dict['video']['metadata']['weight']}", self.VideoSettings)
        self.weight_entry.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.weight_entry.setStyleSheet('QLineEdit { width: 402px; }')
        self.weight_entry.move(570, 430)

        housing_label = QLabel('Housing:', self.VideoSettings)
        housing_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        housing_label.move(495, 460)
        self.housing_entry = QLineEdit(f"{self.exp_settings_dict['video']['metadata']['housing']}", self.VideoSettings)
        self.housing_entry.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.housing_entry.setStyleSheet('QLineEdit { width: 402px; }')
        self.housing_entry.move(570, 460)

        implant_label = QLabel('Implant:', self.VideoSettings)
        implant_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        implant_label.move(495, 490)
        self.implant_entry = QLineEdit(f"{self.exp_settings_dict['video']['metadata']['implant']}", self.VideoSettings)
        self.implant_entry.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.implant_entry.setStyleSheet('QLineEdit { width: 402px; }')
        self.implant_entry.move(570, 490)

        virus_label = QLabel('Virus:', self.VideoSettings)
        virus_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        virus_label.move(495, 520)
        self.virus_entry = QLineEdit(f"{self.exp_settings_dict['video']['metadata']['virus']}", self.VideoSettings)
        self.virus_entry.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.virus_entry.setStyleSheet('QLineEdit { width: 402px; }')
        self.virus_entry.move(570, 520)

        notes_label = QLabel('Notes:', self.VideoSettings)
        notes_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        notes_label.move(495, 550)
        self.notes_entry = QTextEdit(f"{self.exp_settings_dict['video']['metadata']['notes']}", self.VideoSettings)
        self.notes_entry.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.notes_entry.move(570, 550)
        self.notes_entry.setFixedSize(403, 290)

        self._create_buttons_record(seq=2, class_option=self.VideoSettings,
                                    button_pos_y=record_three_y - 35, next_button_x_pos=record_three_x - 100)

    def record_four(self) -> None:
        """
        Initializes the usv-playpen Record Four window.

        Parameters
        ----------
        ----------

        Returns
        -------
        -------
        """

        self.ConductRecording = ConductRecording(self)
        self.setWindowTitle(f'{app_name} (Conduct recording)')
        self.setCentralWidget(self.ConductRecording)
        record_four_x, record_four_y = (480, 560)
        self.setFixedSize(record_four_x, record_four_y)

        self.txt_edit = QPlainTextEdit(self.ConductRecording)
        self.txt_edit.move(5, 5)
        self.txt_edit.setFixedSize(465, 500)
        self.txt_edit.setReadOnly(True)

        with open((Path(__file__).parent / '_config/behavioral_experiments_settings.toml'), 'w') as updated_toml_file:
            toml.dump(self.exp_settings_dict, updated_toml_file)

        self.run_exp = ExperimentController(message_output=self._message,
                                            email_receivers=self.email_recipients,
                                            exp_settings_dict=self.exp_settings_dict)

        self._create_buttons_record(seq=3, class_option=self.ConductRecording,
                                    button_pos_y=record_four_y - 35, next_button_x_pos=record_four_x - 100)

    def process_one(self) -> None:
        """
        Initializes the usv-playpen Process One window.

        Parameters
        ----------
        ----------

        Returns
        -------
        -------
        """

        self.ProcessSettings = ProcessSettings(self)
        self.setWindowTitle(f'{app_name} (Process recordings > Settings)')
        self.setCentralWidget(self.ProcessSettings)
        record_four_x, record_four_y = (1080, 935)
        self.setFixedSize(record_four_x, record_four_y)

        # column 1

        processing_dir_label = QLabel('(*) Root directories', self.ProcessSettings)
        processing_dir_label.setFont(QFont(self.font_id, 13+self.font_size_increase))
        processing_dir_label.setStyleSheet('QLabel { font-weight: bold;}')
        processing_dir_label.move(85, 10)
        self.processing_dir_edit = QTextEdit('', self.ProcessSettings)
        self.processing_dir_edit.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.processing_dir_edit.move(10, 40)
        self.processing_dir_edit.setFixedSize(295, 320)

        exp_codes_dir_label = QLabel('ExCode', self.ProcessSettings)
        exp_codes_dir_label.setFont(QFont(self.font_id, 13+self.font_size_increase))
        exp_codes_dir_label.setStyleSheet('QLabel { font-weight: bold;}')
        exp_codes_dir_label.move(330, 10)
        self.exp_codes_edit = QTextEdit('', self.ProcessSettings)
        self.exp_codes_edit.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.exp_codes_edit.move(310, 40)
        self.exp_codes_edit.setFixedSize(100, 320)

        sleap_conda_label = QLabel('SLEAP conda environment name:', self.ProcessSettings)
        sleap_conda_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        sleap_conda_label.move(10, 365)
        self.sleap_conda = QLineEdit(f"{self.processing_input_dict['anipose_operations']['ConvertTo3D']['sleap_file_conversion']['sleap_conda_env_name']}", self.ProcessSettings)
        self.sleap_conda.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.sleap_conda.setStyleSheet('QLineEdit { width: 98px; }')
        self.sleap_conda.move(310, 365)

        self.centroid_model_edit = QLineEdit(f"{self.processing_input_dict['prepare_cluster_job']['centroid_model_path']}", self.ProcessSettings)
        self.centroid_model_edit.setPlaceholderText('SLEAP centroid model directory')
        self.centroid_model_edit.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.centroid_model_edit.setStyleSheet('QLineEdit { width: 295px; }')
        self.centroid_model_edit.move(10, 395)
        centroid_model_btn = QPushButton('Browse', self.ProcessSettings)
        centroid_model_btn.setFont(QFont(self.font_id, 8+self.font_size_increase))
        centroid_model_btn.move(310, 395)
        centroid_model_btn.setStyleSheet('QPushButton { min-width: 77px; min-height: 12px; max-width: 77px; max-height: 13px; }')
        self.centroid_model_btn_clicked_flag = False
        centroid_model_btn.clicked.connect(self._open_centroid_dialog)

        self.centered_instance_model_edit = QLineEdit(f"{self.processing_input_dict['prepare_cluster_job']['centered_instance_model_path']}", self.ProcessSettings)
        self.centered_instance_model_edit.setPlaceholderText('SLEAP centered instance model directory')
        self.centered_instance_model_edit.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.centered_instance_model_edit.setStyleSheet('QLineEdit { width: 295px; }')
        self.centered_instance_model_edit.move(10, 425)
        centered_instance_btn = QPushButton('Browse', self.ProcessSettings)
        centered_instance_btn.setFont(QFont(self.font_id, 8+self.font_size_increase))
        centered_instance_btn.move(310, 425)
        centered_instance_btn.setStyleSheet('QPushButton { min-width: 77px; min-height: 12px; max-width: 77px; max-height: 13px; }')
        self.centered_instance_btn_btn_clicked_flag = False
        centered_instance_btn.clicked.connect(self._open_centered_instance_dialog)

        self.inference_root_dir_edit = QLineEdit(self.sleap_inference_dir_global, self.ProcessSettings)
        self.inference_root_dir_edit.setPlaceholderText('SLEAP inference directory')
        self.inference_root_dir_edit.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.inference_root_dir_edit.setStyleSheet('QLineEdit { width: 295px; }')
        self.inference_root_dir_edit.move(10, 455)
        inference_root_dir_btn = QPushButton('Browse', self.ProcessSettings)
        inference_root_dir_btn.setFont(QFont(self.font_id, 8+self.font_size_increase))
        inference_root_dir_btn.move(310, 455)
        inference_root_dir_btn.setStyleSheet('QPushButton { min-width: 77px; min-height: 12px; max-width: 77px; max-height: 13px; }')
        self.inference_root_dir_btn_clicked_flag = False
        inference_root_dir_btn.clicked.connect(self._open_inference_root_dialog)

        self.calibration_file_loc_edit = QLineEdit('', self.ProcessSettings)
        self.calibration_file_loc_edit.setPlaceholderText('Tracking calibration root directory')
        self.calibration_file_loc_edit.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.calibration_file_loc_edit.setStyleSheet('QLineEdit { width: 295px; }')
        self.calibration_file_loc_edit.move(10, 485)
        calibration_file_loc_btn = QPushButton('Browse', self.ProcessSettings)
        calibration_file_loc_btn.setFont(QFont(self.font_id, 8+self.font_size_increase))
        calibration_file_loc_btn.move(310, 485)
        calibration_file_loc_btn.setStyleSheet('QPushButton { min-width: 77px; min-height: 12px; max-width: 77px; max-height: 12px; }')
        self.calibration_file_loc_btn_clicked_flag = False
        calibration_file_loc_btn.clicked.connect(self._open_anipose_calibration_dialog)

        das_conda_label = QLabel('DAS conda environment name:', self.ProcessSettings)
        das_conda_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        das_conda_label.move(10, 515)
        self.das_conda = QLineEdit(self.processing_input_dict['usv_inference']['FindMouseVocalizations']['das_command_line_inference']['das_conda_env_name'], self.ProcessSettings)
        self.das_conda.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.das_conda.setStyleSheet('QLineEdit { width: 98px; }')
        self.das_conda.move(310, 515)

        self.das_model_dir_edit = QLineEdit(self.das_model_dir_global, self.ProcessSettings)
        self.das_model_dir_edit.setPlaceholderText('DAS model directory')
        self.das_model_dir_edit.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.das_model_dir_edit.setStyleSheet('QLineEdit { width: 295px; }')
        self.das_model_dir_edit.move(10, 545)
        das_model_dir_btn = QPushButton('Browse', self.ProcessSettings)
        das_model_dir_btn.setFont(QFont(self.font_id, 8+self.font_size_increase))
        das_model_dir_btn.move(310, 545)
        das_model_dir_btn.setStyleSheet('QPushButton { min-width: 77px; min-height: 12px; max-width: 77px; max-height: 12px; }')
        self.das_model_dir_btn_clicked_flag = False
        das_model_dir_btn.clicked.connect(self._open_das_model_dialog)

        das_model_base_label = QLabel('DAS model base (timestamp):', self.ProcessSettings)
        das_model_base_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        das_model_base_label.move(10, 575)
        self.das_model_base = QLineEdit(self.processing_input_dict['usv_inference']['FindMouseVocalizations']['das_command_line_inference']['model_name_base'], self.ProcessSettings)
        self.das_model_base.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.das_model_base.setStyleSheet('QLineEdit { width: 183px; }')
        self.das_model_base.move(225, 575)

        pc_usage_process_label = QLabel('Notify e-mail(s) of PC usage:', self.ProcessSettings)
        pc_usage_process_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        pc_usage_process_label.move(10, 605)
        self.pc_usage_process = QLineEdit('', self.ProcessSettings)
        self.pc_usage_process.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.pc_usage_process.setStyleSheet('QLineEdit { width: 183px; }')
        self.pc_usage_process.move(225, 605)

        processing_pc_label = QLabel('Processing PC of choice:', self.ProcessSettings)
        processing_pc_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        processing_pc_label.move(10, 635)
        self.loaded_processing_pc_list = sorted(self.processing_input_dict['send_email']['Messenger']['processing_pc_list'], key=lambda x: x == self.processing_input_dict['send_email']['Messenger']['processing_pc_choice'], reverse=True)
        self.processing_pc_cb = QComboBox(self.ProcessSettings)
        self.processing_pc_cb.addItems(self.loaded_processing_pc_list)
        self.processing_pc_cb.setStyleSheet('QComboBox { width: 155px; }')
        self.processing_pc_cb.activated.connect(partial(self._combo_box_prior_processing_pc_choice, variable_id='processing_pc_choice'))
        self.processing_pc_cb.move(225, 635)

        gvs_label = QLabel('Video processing settings', self.ProcessSettings)
        gvs_label.setFont(QFont(self.font_id, 13+self.font_size_increase))
        gvs_label.setStyleSheet('QLabel { font-weight: bold;}')
        gvs_label.move(10, 675)

        conduct_video_concatenation_label = QLabel('Conduct video concatenation:', self.ProcessSettings)
        conduct_video_concatenation_label.setFont(QFont(self.font_id, 11+self.font_size_increase))
        conduct_video_concatenation_label.setStyleSheet('QLabel { color: #F58025; font-weight: bold;}')
        conduct_video_concatenation_label.move(10, 705)
        self.conduct_video_concatenation_cb = QComboBox(self.ProcessSettings)
        self.conduct_video_concatenation_cb.addItems(['No', 'Yes'])
        self.conduct_video_concatenation_cb.setStyleSheet('QComboBox { width: 105px; }')
        self.conduct_video_concatenation_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='conduct_video_concatenation_cb_bool'))
        self.conduct_video_concatenation_cb.move(225, 705)

        conduct_video_fps_change_cb_label = QLabel('Conduct video re-encoding:', self.ProcessSettings)
        conduct_video_fps_change_cb_label.setFont(QFont(self.font_id, 11+self.font_size_increase))
        conduct_video_fps_change_cb_label.setStyleSheet('QLabel { color: #F58025; font-weight: bold;}')
        conduct_video_fps_change_cb_label.move(10, 735)
        self.conduct_video_fps_change_cb = QComboBox(self.ProcessSettings)
        self.conduct_video_fps_change_cb.addItems(['No', 'Yes'])
        self.conduct_video_fps_change_cb.setStyleSheet('QComboBox { width: 105px; }')
        self.conduct_video_fps_change_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='conduct_video_fps_change_cb_bool'))
        self.conduct_video_fps_change_cb.move(225, 735)

        conversion_target_file_label = QLabel('Concatenated video name:', self.ProcessSettings)
        conversion_target_file_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        conversion_target_file_label.move(10, 765)
        self.conversion_target_file = QLineEdit(self.processing_input_dict['modify_files']['Operator']['rectify_video_fps']['conversion_target_file'], self.ProcessSettings)
        self.conversion_target_file.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.conversion_target_file.setStyleSheet('QLineEdit { width: 135px; }')
        self.conversion_target_file.move(225, 765)

        constant_rate_factor_label = QLabel('FFMPEG encoding crf (0–51):', self.ProcessSettings)
        constant_rate_factor_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        constant_rate_factor_label.move(10, 795)
        self.constant_rate_factor = QLineEdit(f"{self.processing_input_dict['modify_files']['Operator']['rectify_video_fps']['constant_rate_factor']}", self.ProcessSettings)
        self.constant_rate_factor.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.constant_rate_factor.setStyleSheet('QLineEdit { width: 135px; }')
        self.constant_rate_factor.move(225, 795)

        encoding_preset_label = QLabel('FFMPEG encoding preset:', self.ProcessSettings)
        encoding_preset_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        encoding_preset_label.move(10, 825)
        self.encoding_preset_list = sorted(['veryfast', 'faster', 'fast', 'medium', 'slow', 'slower', 'veryslow'], key=lambda x: x == self.encoding_preset, reverse=True)
        self.encoding_preset_cb = QComboBox(self.ProcessSettings)
        self.encoding_preset_cb.addItems([str(encode_preset_item) for encode_preset_item in self.encoding_preset_list])
        self.encoding_preset_cb.setStyleSheet('QComboBox { width: 105px; }')
        self.encoding_preset_cb.activated.connect(partial(self._combo_box_encoding_preset, variable_id='encoding_preset'))
        self.encoding_preset_cb.move(225, 825)

        delete_con_file_cb_label = QLabel('Delete concatenated video:', self.ProcessSettings)
        delete_con_file_cb_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        delete_con_file_cb_label.move(10, 855)
        self.delete_con_file_cb = QComboBox(self.ProcessSettings)
        self.delete_con_file_cb.addItems(['Yes', 'No'])
        self.delete_con_file_cb.setStyleSheet('QComboBox { width: 105px; }')
        self.delete_con_file_cb.activated.connect(partial(self._combo_box_prior_true, variable_id='delete_con_file_cb_bool'))
        self.delete_con_file_cb.move(225, 855)

        # column 2

        column_two_x1 = 440
        column_two_x2 = 630

        gas_label = QLabel('Audio processing settings', self.ProcessSettings)
        gas_label.setFont(QFont(self.font_id, 13+self.font_size_increase))
        gas_label.setStyleSheet('QLabel { font-weight: bold;}')
        gas_label.move(column_two_x1, 10)

        conduct_multichannel_conversion_cb_label = QLabel('Convert to single-ch files:', self.ProcessSettings)
        conduct_multichannel_conversion_cb_label.setFont(QFont(self.font_id, 11+self.font_size_increase))
        conduct_multichannel_conversion_cb_label.setStyleSheet('QLabel { color: #F58025; font-weight: bold;}')
        conduct_multichannel_conversion_cb_label.move(column_two_x1, 40)
        self.conduct_multichannel_conversion_cb = QComboBox(self.ProcessSettings)
        self.conduct_multichannel_conversion_cb.addItems(['No', 'Yes'])
        self.conduct_multichannel_conversion_cb.setStyleSheet('QComboBox { width: 80px; }')
        self.conduct_multichannel_conversion_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='conduct_multichannel_conversion_cb_bool'))
        self.conduct_multichannel_conversion_cb.move(column_two_x2, 40)

        crop_wav_cam_cb_label = QLabel('Crop AUDIO (to VIDEO):', self.ProcessSettings)
        crop_wav_cam_cb_label.setFont(QFont(self.font_id, 11+self.font_size_increase))
        crop_wav_cam_cb_label.setStyleSheet('QLabel { color: #F58025; font-weight: bold;}')
        crop_wav_cam_cb_label.move(column_two_x1, 70)
        self.crop_wav_cam_cb = QComboBox(self.ProcessSettings)
        self.crop_wav_cam_cb.addItems(['No', 'Yes'])
        self.crop_wav_cam_cb.setStyleSheet('QComboBox { width: 80px; }')
        self.crop_wav_cam_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='crop_wav_cam_cb_bool'))
        self.crop_wav_cam_cb.move(column_two_x2, 70)

        device_receiving_input_cb_label = QLabel('Trgbox-USGH device(s):', self.ProcessSettings)
        device_receiving_input_cb_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        device_receiving_input_cb_label.move(column_two_x1, 100)
        self.device_receiving_input_cb = QComboBox(self.ProcessSettings)
        self.device_receiving_input_cb.addItems(['m', 's', 'both'])
        self.device_receiving_input_cb.setStyleSheet('QComboBox { width: 80px; }')
        self.device_receiving_input_cb.activated.connect(partial(self._combo_box_prior_audio_device_camera_input, variable_id='device_receiving_input'))
        self.device_receiving_input_cb.move(column_two_x2, 100)

        ch_receiving_input_label = QLabel('Trgbox-USGH ch (1-12):', self.ProcessSettings)
        ch_receiving_input_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        ch_receiving_input_label.move(column_two_x1, 130)
        self.ch_receiving_input = QLineEdit(f"{self.processing_input_dict['synchronize_files']['Synchronizer']['crop_wav_files_to_video']['ch_receiving_input']}", self.ProcessSettings)
        self.ch_receiving_input.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.ch_receiving_input.setStyleSheet('QLineEdit { width: 108px; }')
        self.ch_receiving_input.move(column_two_x2, 130)

        conduct_hpss_cb_label = QLabel('Conduct HPSS (slow!):', self.ProcessSettings)
        conduct_hpss_cb_label.setFont(QFont(self.font_id, 11+self.font_size_increase))
        conduct_hpss_cb_label.setStyleSheet('QLabel { color: #F58025; font-weight: bold;}')
        conduct_hpss_cb_label.move(column_two_x1, 160)
        self.conduct_hpss_cb = QComboBox(self.ProcessSettings)
        self.conduct_hpss_cb.addItems(['No', 'Yes'])
        self.conduct_hpss_cb.setStyleSheet('QComboBox { width: 80px; }')
        self.conduct_hpss_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='conduct_hpss_cb_bool'))
        self.conduct_hpss_cb.move(column_two_x2, 160)

        stft_label = QLabel('STFT window & hop size:', self.ProcessSettings)
        stft_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        stft_label.move(column_two_x1, 190)
        self.stft_window_hop = QLineEdit(','.join([str(x) for x in self.processing_input_dict['modify_files']['Operator']['hpss_audio']['stft_window_length_hop_size']]), self.ProcessSettings)
        self.stft_window_hop.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.stft_window_hop.setStyleSheet('QLineEdit { width: 108px; }')
        self.stft_window_hop.move(column_two_x2, 190)

        hpss_kernel_size_label = QLabel('HPSS kernel size:', self.ProcessSettings)
        hpss_kernel_size_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        hpss_kernel_size_label.move(column_two_x1, 220)
        self.hpss_kernel_size = QLineEdit(','.join([str(x) for x in self.processing_input_dict['modify_files']['Operator']['hpss_audio']['kernel_size']]), self.ProcessSettings)
        self.hpss_kernel_size.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.hpss_kernel_size.setStyleSheet('QLineEdit { width: 108px; }')
        self.hpss_kernel_size.move(column_two_x2, 220)

        hpss_power_label = QLabel('HPSS power:', self.ProcessSettings)
        hpss_power_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        hpss_power_label.move(column_two_x1, 250)
        self.hpss_power = QLineEdit(f"{self.processing_input_dict['modify_files']['Operator']['hpss_audio']['hpss_power']}", self.ProcessSettings)
        self.hpss_power.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.hpss_power.setStyleSheet('QLineEdit { width: 108px; }')
        self.hpss_power.move(column_two_x2, 250)

        hpss_margin_label = QLabel('HPSS margin:', self.ProcessSettings)
        hpss_margin_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        hpss_margin_label.move(column_two_x1, 280)
        self.hpss_margin = QLineEdit(','.join([str(x) for x in self.processing_input_dict['modify_files']['Operator']['hpss_audio']['margin']]), self.ProcessSettings)
        self.hpss_margin.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.hpss_margin.setStyleSheet('QLineEdit { width: 108px; }')
        self.hpss_margin.move(column_two_x2, 280)

        filter_audio_cb_label = QLabel('Filter individual audio files:', self.ProcessSettings)
        filter_audio_cb_label.setFont(QFont(self.font_id, 11+self.font_size_increase))
        filter_audio_cb_label.setStyleSheet('QLabel { color: #F58025; font-weight: bold;}')
        filter_audio_cb_label.move(column_two_x1, 310)
        self.filter_audio_cb = QComboBox(self.ProcessSettings)
        self.filter_audio_cb.addItems(['No', 'Yes'])
        self.filter_audio_cb.setStyleSheet('QComboBox { width: 80px; }')
        self.filter_audio_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='filter_audio_cb_bool'))
        self.filter_audio_cb.move(column_two_x2, 310)

        filter_freq_bounds_label = QLabel('Filter freq bounds (Hz):', self.ProcessSettings)
        filter_freq_bounds_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        filter_freq_bounds_label.move(column_two_x1, 340)
        self.filter_freq_bounds = QLineEdit(','.join([str(x) for x in self.processing_input_dict['modify_files']['Operator']['filter_audio_files']['filter_freq_bounds']]), self.ProcessSettings)
        self.filter_freq_bounds.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.filter_freq_bounds.setStyleSheet('QLineEdit { width: 108px; }')
        self.filter_freq_bounds.move(column_two_x2, 340)

        filter_dirs_label = QLabel('Folder(s) to filter:', self.ProcessSettings)
        filter_dirs_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        filter_dirs_label.move(column_two_x1, 370)
        self.filter_dirs = QLineEdit(','.join([str(x) for x in self.processing_input_dict['modify_files']['Operator']['filter_audio_files']['filter_dirs']]), self.ProcessSettings)
        self.filter_dirs.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.filter_dirs.setStyleSheet('QLineEdit { width: 108px; }')
        self.filter_dirs.move(column_two_x2, 370)

        conc_audio_cb_label = QLabel('Concatenate to MEMMAP:', self.ProcessSettings)
        conc_audio_cb_label.setFont(QFont(self.font_id, 11+self.font_size_increase))
        conc_audio_cb_label.setStyleSheet('QLabel { color: #F58025; font-weight: bold;}')
        conc_audio_cb_label.move(column_two_x1, 400)
        self.conc_audio_cb = QComboBox(self.ProcessSettings)
        self.conc_audio_cb.addItems(['No', 'Yes'])
        self.conc_audio_cb.setStyleSheet('QComboBox { width: 80px; }')
        self.conc_audio_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='conc_audio_cb_bool'))
        self.conc_audio_cb.move(column_two_x2, 400)

        concat_dirs_label = QLabel('Folder(s) to concatenate:', self.ProcessSettings)
        concat_dirs_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        concat_dirs_label.move(column_two_x1, 430)
        self.concat_dirs = QLineEdit(','.join([str(x) for x in self.processing_input_dict['modify_files']['Operator']['concatenate_audio_files']['concat_dirs']]), self.ProcessSettings)
        self.concat_dirs.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.concat_dirs.setStyleSheet('QLineEdit { width: 108px; }')
        self.concat_dirs.move(column_two_x2, 430)

        av_sync_label = QLabel('Synchronization between A/V files', self.ProcessSettings)
        av_sync_label.setFont(QFont(self.font_id, 13+self.font_size_increase))
        av_sync_label.setStyleSheet('QLabel { font-weight: bold;}')
        av_sync_label.move(column_two_x1, 470)

        conduct_sync_cb_label = QLabel('Conduct A/V sync check:', self.ProcessSettings)
        conduct_sync_cb_label.setFont(QFont(self.font_id, 11+self.font_size_increase))
        conduct_sync_cb_label.setStyleSheet('QLabel { color: #F58025; font-weight: bold;}')
        conduct_sync_cb_label.move(column_two_x1, 500)
        self.conduct_sync_cb = QComboBox(self.ProcessSettings)
        self.conduct_sync_cb.addItems(['No', 'Yes'])
        self.conduct_sync_cb.setStyleSheet('QComboBox { width: 80px; }')
        self.conduct_sync_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='conduct_sync_cb_bool'))
        self.conduct_sync_cb.move(column_two_x2, 500)

        phidget_extra_data_camera_label = QLabel('Phidget(s) camera serial:', self.ProcessSettings)
        phidget_extra_data_camera_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        phidget_extra_data_camera_label.move(column_two_x1, 530)
        self.phidget_extra_data_camera = QLineEdit(f"{self.processing_input_dict['extract_phidget_data']['Gatherer']['prepare_data_for_analyses']['extra_data_camera']}", self.ProcessSettings)
        self.phidget_extra_data_camera.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.phidget_extra_data_camera.setStyleSheet('QLineEdit { width: 108px; }')
        self.phidget_extra_data_camera.move(column_two_x2, 530)

        a_ch_receiving_input_label = QLabel('Arduino-USGH ch (1-12):', self.ProcessSettings)
        a_ch_receiving_input_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        a_ch_receiving_input_label.move(column_two_x1, 560)
        self.a_ch_receiving_input = QLineEdit(f"{self.processing_input_dict['synchronize_files']['Synchronizer']['find_audio_sync_trains']['ch_receiving_input']}", self.ProcessSettings)
        self.a_ch_receiving_input.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.a_ch_receiving_input.setStyleSheet('QLineEdit { width: 108px; }')
        self.a_ch_receiving_input.move(column_two_x2, 560)

        v_camera_serial_num_label = QLabel('Sync camera serial num(s):', self.ProcessSettings)
        v_camera_serial_num_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        v_camera_serial_num_label.move(column_two_x1, 590)
        self.v_camera_serial_num = QLineEdit(','.join([str(x) for x in self.processing_input_dict['synchronize_files']['Synchronizer']['find_video_sync_trains']['camera_serial_num']]), self.ProcessSettings)
        self.v_camera_serial_num.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.v_camera_serial_num.setStyleSheet('QLineEdit { width: 108px; }')
        self.v_camera_serial_num.move(column_two_x2, 590)

        ev_sync_label = QLabel('Neural data processing settings', self.ProcessSettings)
        ev_sync_label.setFont(QFont(self.font_id, 13+self.font_size_increase))
        ev_sync_label.setStyleSheet('QLabel { font-weight: bold;}')
        ev_sync_label.move(column_two_x1, 630)

        conduct_nv_sync_cb_label = QLabel('Conduct E/V sync check:', self.ProcessSettings)
        conduct_nv_sync_cb_label.setFont(QFont(self.font_id, 11+self.font_size_increase))
        conduct_nv_sync_cb_label.setStyleSheet('QLabel { color: #F58025; font-weight: bold;}')
        conduct_nv_sync_cb_label.move(column_two_x1, 660)
        self.conduct_nv_sync_cb = QComboBox(self.ProcessSettings)
        self.conduct_nv_sync_cb.addItems(['No', 'Yes'])
        self.conduct_nv_sync_cb.setStyleSheet('QComboBox { width: 80px; }')
        self.conduct_nv_sync_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='conduct_nv_sync_cb_bool'))
        self.conduct_nv_sync_cb.move(column_two_x2, 660)

        conduct_ephys_file_chaining_label = QLabel('Conduct e-phys concat:', self.ProcessSettings)
        conduct_ephys_file_chaining_label.setFont(QFont(self.font_id, 11+self.font_size_increase))
        conduct_ephys_file_chaining_label.setStyleSheet('QLabel { color: #F58025; font-weight: bold;}')
        conduct_ephys_file_chaining_label.move(column_two_x1, 690)
        self.conduct_ephys_file_chaining_cb = QComboBox(self.ProcessSettings)
        self.conduct_ephys_file_chaining_cb.addItems(['No', 'Yes'])
        self.conduct_ephys_file_chaining_cb.setStyleSheet('QComboBox { width: 80px; }')
        self.conduct_ephys_file_chaining_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='conduct_ephys_file_chaining_cb_bool'))
        self.conduct_ephys_file_chaining_cb.move(column_two_x2, 690)

        split_cluster_spikes_cb_label = QLabel('Split clusters to sessions:', self.ProcessSettings)
        split_cluster_spikes_cb_label.setFont(QFont(self.font_id, 11+self.font_size_increase))
        split_cluster_spikes_cb_label.setStyleSheet('QLabel { color: #F58025; font-weight: bold;}')
        split_cluster_spikes_cb_label.move(column_two_x1, 720)
        self.split_cluster_spikes_cb = QComboBox(self.ProcessSettings)
        self.split_cluster_spikes_cb.addItems(['No', 'Yes'])
        self.split_cluster_spikes_cb.setStyleSheet('QComboBox { width: 80px; }')
        self.split_cluster_spikes_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='split_cluster_spikes_cb_bool'))
        self.split_cluster_spikes_cb.move(column_two_x2, 720)

        npx_file_type_cb_label = QLabel('Rec file format (ap | lf):', self.ProcessSettings)
        npx_file_type_cb_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        npx_file_type_cb_label.move(column_two_x1, 750)
        self.npx_file_type_cb = QComboBox(self.ProcessSettings)
        self.npx_file_type_cb.addItems(['ap', 'lf'])
        self.npx_file_type_cb.setStyleSheet('QComboBox { width: 80px; }')
        self.npx_file_type_cb.activated.connect(partial(self._combo_box_prior_npx_file_type, variable_id='npx_file_type'))
        self.npx_file_type_cb.move(column_two_x2, 750)

        npx_ms_divergence_tolerance_label = QLabel('Divergence tolerance (ms):', self.ProcessSettings)
        npx_ms_divergence_tolerance_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        npx_ms_divergence_tolerance_label.move(column_two_x1, 780)
        self.npx_ms_divergence_tolerance = QLineEdit(f"{self.processing_input_dict['synchronize_files']['Synchronizer']['validate_ephys_video_sync']['npx_ms_divergence_tolerance']}", self.ProcessSettings)
        self.npx_ms_divergence_tolerance.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.npx_ms_divergence_tolerance.setStyleSheet('QLineEdit { width: 108px; }')
        self.npx_ms_divergence_tolerance.move(column_two_x2, 780)

        min_spike_num_label = QLabel('Min num of spikes:', self.ProcessSettings)
        min_spike_num_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        min_spike_num_label.move(column_two_x1, 810)
        self.min_spike_num = QLineEdit(f"{self.processing_input_dict['modify_files']['Operator']['get_spike_times']['min_spike_num']}", self.ProcessSettings)
        self.min_spike_num.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.min_spike_num.setStyleSheet('QLineEdit { width: 108px; }')
        self.min_spike_num.move(column_two_x2, 810)

        # column 3
        column_three_x1 = 760
        column_three_x2 = 960

        anipose_operations_label = QLabel('SLEAP / DAS operations', self.ProcessSettings)
        anipose_operations_label.setFont(QFont(self.font_id, 13+self.font_size_increase))
        anipose_operations_label.setStyleSheet('QLabel { font-weight: bold;}')
        anipose_operations_label.move(column_three_x1, 10)

        sleap_cluster_cb_label = QLabel('Prepare SLEAP cluster job:', self.ProcessSettings)
        sleap_cluster_cb_label.setFont(QFont(self.font_id, 11+self.font_size_increase))
        sleap_cluster_cb_label.setStyleSheet('QLabel { color: #F58025; font-weight: bold;}')
        sleap_cluster_cb_label.move(column_three_x1, 40)
        self.sleap_cluster_cb = QComboBox(self.ProcessSettings)
        self.sleap_cluster_cb.addItems(['No', 'Yes'])
        self.sleap_cluster_cb.setStyleSheet('QComboBox { width: 80px; }')
        self.sleap_cluster_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='sleap_cluster_cb_bool'))
        self.sleap_cluster_cb.move(column_three_x2, 40)

        sleap_file_conversion_cb_label = QLabel('Conduct SLP-H5 conversion:', self.ProcessSettings)
        sleap_file_conversion_cb_label.setFont(QFont(self.font_id, 11+self.font_size_increase))
        sleap_file_conversion_cb_label.setStyleSheet('QLabel { color: #F58025; font-weight: bold;}')
        sleap_file_conversion_cb_label.move(column_three_x1, 70)
        self.sleap_file_conversion_cb = QComboBox(self.ProcessSettings)
        self.sleap_file_conversion_cb.addItems(['No', 'Yes'])
        self.sleap_file_conversion_cb.setStyleSheet('QComboBox { width: 80px; }')
        self.sleap_file_conversion_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='sleap_file_conversion_cb_bool'))
        self.sleap_file_conversion_cb.move(column_three_x2, 70)

        anipose_calibration_cb_label = QLabel('Conduct AP calibration:', self.ProcessSettings)
        anipose_calibration_cb_label.setFont(QFont(self.font_id, 11+self.font_size_increase))
        anipose_calibration_cb_label.setStyleSheet('QLabel { color: #F58025; font-weight: bold;}')
        anipose_calibration_cb_label.move(column_three_x1, 100)
        self.anipose_calibration_cb = QComboBox(self.ProcessSettings)
        self.anipose_calibration_cb.addItems(['No', 'Yes'])
        self.anipose_calibration_cb.setStyleSheet('QComboBox { width: 80px; }')
        self.anipose_calibration_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='anipose_calibration_cb_bool'))
        self.anipose_calibration_cb.move(column_three_x2, 100)

        board_provided_cb_label = QLabel('Calibration board provided:', self.ProcessSettings)
        board_provided_cb_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        board_provided_cb_label.move(column_three_x1, 130)
        self.board_provided_cb = QComboBox(self.ProcessSettings)
        self.board_provided_cb.addItems(['No', 'Yes'])
        self.board_provided_cb.setStyleSheet('QComboBox { width: 80px; }')
        self.board_provided_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='board_provided_cb_bool'))
        self.board_provided_cb.move(column_three_x2, 130)

        anipose_triangulation_cb_label = QLabel('Conduct AP triangulation:', self.ProcessSettings)
        anipose_triangulation_cb_label.setFont(QFont(self.font_id, 11+self.font_size_increase))
        anipose_triangulation_cb_label.setStyleSheet('QLabel { color: #F58025; font-weight: bold;}')
        anipose_triangulation_cb_label.move(column_three_x1, 160)
        self.anipose_triangulation_cb = QComboBox(self.ProcessSettings)
        self.anipose_triangulation_cb.addItems(['No', 'Yes'])
        self.anipose_triangulation_cb.setStyleSheet('QComboBox { width: 80px; }')
        self.anipose_triangulation_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='anipose_triangulation_cb_bool'))
        self.anipose_triangulation_cb.move(column_three_x2, 160)

        triangulate_arena_points_cb_label = QLabel('Triangulate arena nodes:', self.ProcessSettings)
        triangulate_arena_points_cb_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        triangulate_arena_points_cb_label.move(column_three_x1, 190)
        self.triangulate_arena_points_cb = QComboBox(self.ProcessSettings)
        self.triangulate_arena_points_cb.addItems(['No', 'Yes'])
        self.triangulate_arena_points_cb.setStyleSheet('QComboBox { width: 80px; }')
        self.triangulate_arena_points_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='triangulate_arena_points_cb_bool'))
        self.triangulate_arena_points_cb.move(column_three_x2, 190)

        display_progress_cb_label = QLabel('Display progress:', self.ProcessSettings)
        display_progress_cb_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        display_progress_cb_label.move(column_three_x1, 220)
        self.display_progress_cb = QComboBox(self.ProcessSettings)
        self.display_progress_cb.addItems(['Yes', 'No'])
        self.display_progress_cb.setStyleSheet('QComboBox { width: 80px; }')
        self.display_progress_cb.activated.connect(partial(self._combo_box_prior_true, variable_id='display_progress_cb_bool'))
        self.display_progress_cb.move(column_three_x2, 220)

        frame_restriction_label = QLabel('Frame restriction:', self.ProcessSettings)
        frame_restriction_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        frame_restriction_label.move(column_three_x1, 250)
        frame_restriction_input = '' if self.processing_input_dict['anipose_operations']['ConvertTo3D']['conduct_anipose_triangulation']['frame_restriction'] is None else ','.join([str(x) for x in self.processing_input_dict['anipose_operations']['ConvertTo3D']['conduct_anipose_triangulation']['frame_restriction']])
        self.frame_restriction = QLineEdit(frame_restriction_input, self.ProcessSettings)
        self.frame_restriction.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.frame_restriction.setStyleSheet('QLineEdit { width: 108px; }')
        self.frame_restriction.move(column_three_x2, 250)

        excluded_views_label = QLabel('Excluded camera views:', self.ProcessSettings)
        excluded_views_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        excluded_views_label.move(column_three_x1, 280)
        self.excluded_views = QLineEdit(','.join([str(x) for x in self.processing_input_dict['anipose_operations']['ConvertTo3D']['conduct_anipose_triangulation']['excluded_views']]), self.ProcessSettings)
        self.excluded_views.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.excluded_views.setStyleSheet('QLineEdit { width: 108px; }')
        self.excluded_views.move(column_three_x2, 280)

        ransac_cb_label = QLabel('Ransac:', self.ProcessSettings)
        ransac_cb_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        ransac_cb_label.move(column_three_x1, 310)
        self.ransac_cb = QComboBox(self.ProcessSettings)
        self.ransac_cb.addItems(['No', 'Yes'])
        self.ransac_cb.setStyleSheet('QComboBox { width: 80px; }')
        self.ransac_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='ransac_cb_bool'))
        self.ransac_cb.move(column_three_x2, 310)

        rigid_body_constraints_label = QLabel('Rigid body constraints:', self.ProcessSettings)
        rigid_body_constraints_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        rigid_body_constraints_label.move(column_three_x1, 340)
        self.rigid_body_constraints = QLineEdit('', self.ProcessSettings)
        self.rigid_body_constraints.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.rigid_body_constraints.setStyleSheet('QLineEdit { width: 108px; }')
        self.rigid_body_constraints.move(column_three_x2, 340)

        weak_body_constraints_label = QLabel('Weak body constraints:', self.ProcessSettings)
        weak_body_constraints_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        weak_body_constraints_label.move(column_three_x1, 370)
        self.weak_body_constraints = QLineEdit('', self.ProcessSettings)
        self.weak_body_constraints.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.weak_body_constraints.setStyleSheet('QLineEdit { width: 108px; }')
        self.weak_body_constraints.move(column_three_x2, 370)

        smooth_scale_label = QLabel('Smoothing scale:', self.ProcessSettings)
        smooth_scale_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        smooth_scale_label.move(column_three_x1, 400)
        self.smooth_scale = QLineEdit(f"{self.processing_input_dict['anipose_operations']['ConvertTo3D']['conduct_anipose_triangulation']['smooth_scale']}", self.ProcessSettings)
        self.smooth_scale.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.smooth_scale.setStyleSheet('QLineEdit { width: 108px; }')
        self.smooth_scale.move(column_three_x2, 400)

        weight_rigid_label = QLabel('Rigid constraints weight:', self.ProcessSettings)
        weight_rigid_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        weight_rigid_label.move(column_three_x1, 430)
        self.weight_rigid = QLineEdit(f"{self.processing_input_dict['anipose_operations']['ConvertTo3D']['conduct_anipose_triangulation']['weight_rigid']}", self.ProcessSettings)
        self.weight_rigid.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.weight_rigid.setStyleSheet('QLineEdit { width: 108px; }')
        self.weight_rigid.move(column_three_x2, 430)

        weight_weak_label = QLabel('Weak constraints weight:', self.ProcessSettings)
        weight_weak_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        weight_weak_label.move(column_three_x1, 460)
        self.weight_weak = QLineEdit(f"{self.processing_input_dict['anipose_operations']['ConvertTo3D']['conduct_anipose_triangulation']['weight_weak']}", self.ProcessSettings)
        self.weight_weak.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.weight_weak.setStyleSheet('QLineEdit { width: 108px; }')
        self.weight_weak.move(column_three_x2, 460)

        reprojection_error_threshold_label = QLabel('Reproject error threshold:', self.ProcessSettings)
        reprojection_error_threshold_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        reprojection_error_threshold_label.move(column_three_x1, 490)
        self.reprojection_error_threshold = QLineEdit(f"{self.processing_input_dict['anipose_operations']['ConvertTo3D']['conduct_anipose_triangulation']['reprojection_error_threshold']}", self.ProcessSettings)
        self.reprojection_error_threshold.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.reprojection_error_threshold.setStyleSheet('QLineEdit { width: 108px; }')
        self.reprojection_error_threshold.move(column_three_x2, 490)

        regularization_function_label = QLabel('Regularization:', self.ProcessSettings)
        regularization_function_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        regularization_function_label.move(column_three_x1, 520)
        self.regularization_function = QLineEdit(f"{self.processing_input_dict['anipose_operations']['ConvertTo3D']['conduct_anipose_triangulation']['regularization_function']}", self.ProcessSettings)
        self.regularization_function.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.regularization_function.setStyleSheet('QLineEdit { width: 108px; }')
        self.regularization_function.move(column_three_x2, 520)

        n_deriv_smooth_label = QLabel('Derivation kernel order:', self.ProcessSettings)
        n_deriv_smooth_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        n_deriv_smooth_label.move(column_three_x1, 550)
        self.n_deriv_smooth = QLineEdit(f"{self.processing_input_dict['anipose_operations']['ConvertTo3D']['conduct_anipose_triangulation']['n_deriv_smooth']}", self.ProcessSettings)
        self.n_deriv_smooth.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.n_deriv_smooth.setStyleSheet('QLineEdit { width: 108px; }')
        self.n_deriv_smooth.move(column_three_x2, 550)

        translate_rotate_metric_label = QLabel('Re-coordinate (ExCode!):', self.ProcessSettings)
        translate_rotate_metric_label.setFont(QFont(self.font_id, 11+self.font_size_increase))
        translate_rotate_metric_label.setStyleSheet('QLabel { color: #F58025; font-weight: bold;}')
        translate_rotate_metric_label.move(column_three_x1, 580)
        self.translate_rotate_metric_cb = QComboBox(self.ProcessSettings)
        self.translate_rotate_metric_cb.addItems(['No', 'Yes'])
        self.translate_rotate_metric_cb.setStyleSheet('QComboBox { width: 80px; }')
        self.translate_rotate_metric_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='translate_rotate_metric_cb_bool'))
        self.translate_rotate_metric_cb.move(column_three_x2, 580)

        static_reference_len_label = QLabel('Static reference length (m):', self.ProcessSettings)
        static_reference_len_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        static_reference_len_label.move(column_three_x1, 610)
        self.static_reference_len = QLineEdit(f"{self.processing_input_dict['anipose_operations']['ConvertTo3D']['translate_rotate_metric']['static_reference_len']}", self.ProcessSettings)
        self.static_reference_len.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.static_reference_len.setStyleSheet('QLineEdit { width: 108px; }')
        self.static_reference_len.move(column_three_x2, 610)

        save_transformed_data_cb_label = QLabel('Save transformation type:', self.ProcessSettings)
        save_transformed_data_cb_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        save_transformed_data_cb_label.move(column_three_x1, 640)
        self.save_transformed_data_cb = QComboBox(self.ProcessSettings)
        self.save_transformed_data_cb.addItems(['animal', 'arena'])
        self.save_transformed_data_cb.setStyleSheet('QComboBox { width: 80px; }')
        self.save_transformed_data_cb.activated.connect(partial(self._combo_box_prior_transformed_tracking_data, variable_id='save_transformed_data'))
        self.save_transformed_data_cb.move(column_three_x2, 640)

        delete_original_h5_cb_label = QLabel('Delete original .h5:', self.ProcessSettings)
        delete_original_h5_cb_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        delete_original_h5_cb_label.move(column_three_x1, 670)
        self.delete_original_h5_cb = QComboBox(self.ProcessSettings)
        self.delete_original_h5_cb.addItems(['Yes', 'No'])
        self.delete_original_h5_cb.setStyleSheet('QComboBox { width: 80px; }')
        self.delete_original_h5_cb.activated.connect(partial(self._combo_box_prior_true, variable_id='delete_original_h5_cb_bool'))
        self.delete_original_h5_cb.move(column_three_x2, 670)

        das_inference_cb_label = QLabel('Detect USVs:', self.ProcessSettings)
        das_inference_cb_label.setFont(QFont(self.font_id, 11+self.font_size_increase))
        das_inference_cb_label.setStyleSheet('QLabel { color: #F58025; font-weight: bold;}')
        das_inference_cb_label.move(column_three_x1, 700)
        self.das_inference_cb = QComboBox(self.ProcessSettings)
        self.das_inference_cb.addItems(['No', 'Yes'])
        self.das_inference_cb.setStyleSheet('QComboBox { width: 80px; }')
        self.das_inference_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='das_inference_cb_bool'))
        self.das_inference_cb.move(column_three_x2, 700)

        segment_confidence_threshold_label = QLabel('DAS confidence threshold:', self.ProcessSettings)
        segment_confidence_threshold_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        segment_confidence_threshold_label.move(column_three_x1, 730)
        self.segment_confidence_threshold = QLineEdit(f"{self.processing_input_dict['usv_inference']['FindMouseVocalizations']['das_command_line_inference']['segment_confidence_threshold']}", self.ProcessSettings)
        self.segment_confidence_threshold.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.segment_confidence_threshold.setStyleSheet('QLineEdit { width: 108px; }')
        self.segment_confidence_threshold.move(column_three_x2, 730)

        segment_minlen_label = QLabel('USV min duration (s):', self.ProcessSettings)
        segment_minlen_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        segment_minlen_label.move(column_three_x1, 760)
        self.segment_minlen = QLineEdit(f"{self.processing_input_dict['usv_inference']['FindMouseVocalizations']['das_command_line_inference']['segment_minlen']}", self.ProcessSettings)
        self.segment_minlen.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.segment_minlen.setStyleSheet('QLineEdit { width: 108px; }')
        self.segment_minlen.move(column_three_x2, 760)

        segment_fillgap_label = QLabel('Fill gaps shorter than (s):', self.ProcessSettings)
        segment_fillgap_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        segment_fillgap_label.move(column_three_x1, 790)
        self.segment_fillgap = QLineEdit(f"{self.processing_input_dict['usv_inference']['FindMouseVocalizations']['das_command_line_inference']['segment_fillgap']}", self.ProcessSettings)
        self.segment_fillgap.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.segment_fillgap.setStyleSheet('QLineEdit { width: 108px; }')
        self.segment_fillgap.move(column_three_x2, 790)

        das_output_type_label = QLabel('Inference output file type:', self.ProcessSettings)
        das_output_type_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        das_output_type_label.move(column_three_x1, 820)
        self.das_output_type = QLineEdit(f"{self.processing_input_dict['usv_inference']['FindMouseVocalizations']['das_command_line_inference']['output_file_type']}", self.ProcessSettings)
        self.das_output_type.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.das_output_type.setStyleSheet('QLineEdit { width: 108px; }')
        self.das_output_type.move(column_three_x2, 820)

        das_summary_cb_label = QLabel('Summarize DAS output:', self.ProcessSettings)
        das_summary_cb_label.setFont(QFont(self.font_id, 11+self.font_size_increase))
        das_summary_cb_label.setStyleSheet('QLabel { color: #F58025; font-weight: bold;}')
        das_summary_cb_label.move(column_three_x1, 850)
        self.das_summary_cb = QComboBox(self.ProcessSettings)
        self.das_summary_cb.addItems(['No', 'Yes'])
        self.das_summary_cb.setStyleSheet('QComboBox { width: 80px; }')
        self.das_summary_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='das_summary_cb_bool'))
        self.das_summary_cb.move(column_three_x2, 850)

        self._create_buttons_process(seq=0, class_option=self.ProcessSettings,
                                     button_pos_y=record_four_y - 35, next_button_x_pos=record_four_x - 100)

    def process_two(self) -> None:
        """
        Initializes the usv-playpen Process Two window.

        Parameters
        ----------
        ----------

        Returns
        -------
        -------
        """

        self.ConductProcess = ConductProcess(self)
        self.setWindowTitle(f'{app_name} (Conduct Processing)')
        self.setCentralWidget(self.ConductProcess)
        record_four_x, record_four_y = (870, 1000)
        self.setFixedSize(record_four_x, record_four_y)

        self.txt_edit_process = QPlainTextEdit(self.ConductProcess)
        self.txt_edit_process.move(5, 5)
        self.txt_edit_process.setFixedSize(855, 940)
        self.txt_edit_process.setReadOnly(True)

        with open((Path(__file__).parent / '_config/behavioral_experiments_settings.toml'), 'w') as updated_toml_file:
            toml.dump(self.exp_settings_dict, updated_toml_file)

        with open((Path(__file__).parent / '_parameter_settings/processing_settings.json'), 'w') as processing_settings_file:
            json.dump(self.processing_input_dict, processing_settings_file, indent=2)

        self.run_processing = Stylist(message_output=self._process_message,
                                      input_parameter_dict=self.processing_input_dict,
                                      root_directories=self.processing_input_dict['preprocess_data']['root_directories'],
                                      exp_settings_dict=self.exp_settings_dict)

        self._create_buttons_process(seq=1, class_option=self.ConductProcess,
                                     button_pos_y=record_four_y - 35, next_button_x_pos=record_four_x - 100)

    def analyze_one(self) -> None:
        """
        Initializes the usv-playpen Analyze One window.

        Parameters
        ----------
        ----------

        Returns
        -------
        -------
        """
        self.AnalysesSettings = AnalysesSettings(self)
        self.setWindowTitle(f'{app_name} (Analyze data > Settings)')
        self.setCentralWidget(self.AnalysesSettings)
        analyze_one_x, analyze_one_y = (370, 935)
        self.setFixedSize(analyze_one_x, analyze_one_y)

        analyses_dir_label = QLabel('(*) Root directories for analyses', self.AnalysesSettings)
        analyses_dir_label.setFont(QFont(self.font_id, 13+self.font_size_increase))
        analyses_dir_label.setStyleSheet('QLabel { font-weight: bold;}')
        analyses_dir_label.move(50, 10)
        self.analyses_dir_edit = QTextEdit('', self.AnalysesSettings)
        self.analyses_dir_edit.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.analyses_dir_edit.move(10, 40)
        self.analyses_dir_edit.setFixedSize(350, 320)

        pc_usage_analyses_label = QLabel('Notify e-mail(s) of PC usage:', self.AnalysesSettings)
        pc_usage_analyses_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        pc_usage_analyses_label.move(10, 365)
        self.pc_usage_analyses = QLineEdit('', self.AnalysesSettings)
        self.pc_usage_analyses.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.pc_usage_analyses.setStyleSheet('QLineEdit { width: 135px; }')
        self.pc_usage_analyses.move(225, 365)

        analyses_pc_label = QLabel('Analyses PC of choice:', self.AnalysesSettings)
        analyses_pc_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        analyses_pc_label.move(10, 395)
        self.loaded_analyses_pc_list = sorted(self.analyses_input_dict['send_email']['analyses_pc_list'], key=lambda x: x == self.analyses_input_dict['send_email']['analyses_pc_choice'], reverse=True)
        self.analyses_pc_cb = QComboBox(self.AnalysesSettings)
        self.analyses_pc_cb.addItems(self.loaded_analyses_pc_list)
        self.analyses_pc_cb.setStyleSheet('QComboBox { width: 107px; }')
        self.analyses_pc_cb.activated.connect(partial(self._combo_box_prior_analyses_pc_choice, variable_id='analyses_pc_choice'))
        self.analyses_pc_cb.move(225, 395)

        da_label = QLabel('Select data analysis', self.AnalysesSettings)
        da_label.setFont(QFont(self.font_id, 13+self.font_size_increase))
        da_label.setStyleSheet('QLabel { font-weight: bold;}')
        da_label.move(10, 435)

        compute_behavioral_features_label = QLabel('Compute 3D behavioral features:', self.AnalysesSettings)
        compute_behavioral_features_label.setFont(QFont(self.font_id, 11+self.font_size_increase))
        compute_behavioral_features_label.setStyleSheet('QLabel { color: #F58025; font-weight: bold;}')
        compute_behavioral_features_label.move(10, 465)
        self.compute_behavioral_features_cb = QComboBox(self.AnalysesSettings)
        self.compute_behavioral_features_cb.addItems(['No', 'Yes'])
        self.compute_behavioral_features_cb.setStyleSheet('QComboBox { width: 57px; }')
        self.compute_behavioral_features_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='compute_behavioral_features_cb_bool'))
        self.compute_behavioral_features_cb.move(275, 465)

        calculate_neuronal_tuning_curves_label = QLabel('Compute 3D feature tuning curves:', self.AnalysesSettings)
        calculate_neuronal_tuning_curves_label.setFont(QFont(self.font_id, 11 + self.font_size_increase))
        calculate_neuronal_tuning_curves_label.setStyleSheet('QLabel { color: #F58025; font-weight: bold;}')
        calculate_neuronal_tuning_curves_label.move(10, 495)
        self.calculate_neuronal_tuning_curves_cb = QComboBox(self.AnalysesSettings)
        self.calculate_neuronal_tuning_curves_cb.addItems(['No', 'Yes'])
        self.calculate_neuronal_tuning_curves_cb.setStyleSheet('QComboBox { width: 57px; }')
        self.calculate_neuronal_tuning_curves_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='calculate_neuronal_tuning_curves_cb_bool'))
        self.calculate_neuronal_tuning_curves_cb.move(275, 495)

        create_usv_playback_wav_label = QLabel('Create USV playback .WAV file:', self.AnalysesSettings)
        create_usv_playback_wav_label.setFont(QFont(self.font_id, 11 + self.font_size_increase))
        create_usv_playback_wav_label.setStyleSheet('QLabel { color: #F58025; font-weight: bold;}')
        create_usv_playback_wav_label.move(10, 525)
        self.create_usv_playback_wav_cb = QComboBox(self.AnalysesSettings)
        self.create_usv_playback_wav_cb.addItems(['No', 'Yes'])
        self.create_usv_playback_wav_cb.setStyleSheet('QComboBox { width: 57px; }')
        self.create_usv_playback_wav_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='create_usv_playback_wav_cb_bool'))
        self.create_usv_playback_wav_cb.move(275, 525)

        num_usv_files_label = QLabel('Total number of USV playback files:', self.AnalysesSettings)
        num_usv_files_label.setFont(QFont(self.font_id, 12 + self.font_size_increase))
        num_usv_files_label.move(10, 555)
        self.num_usv_files = QLineEdit(f"{self.analyses_input_dict['create_usv_playback_wav']['num_usv_files']}", self.AnalysesSettings)
        self.num_usv_files.setFont(QFont(self.font_id, 10 + self.font_size_increase))
        self.num_usv_files.setStyleSheet('QLineEdit { width: 85px; }')
        self.num_usv_files.move(275, 555)

        total_usv_number_label = QLabel('Total number os USVs per file:', self.AnalysesSettings)
        total_usv_number_label.setFont(QFont(self.font_id, 12 + self.font_size_increase))
        total_usv_number_label.move(10, 585)
        self.total_usv_number = QLineEdit(f"{self.analyses_input_dict['create_usv_playback_wav']['total_usv_number']}", self.AnalysesSettings)
        self.total_usv_number.setFont(QFont(self.font_id, 10 + self.font_size_increase))
        self.total_usv_number.setStyleSheet('QLineEdit { width: 85px; }')
        self.total_usv_number.move(275, 585)

        ipi_duration_label = QLabel('Fixed silence between USVs (s):', self.AnalysesSettings)
        ipi_duration_label.setFont(QFont(self.font_id, 12 + self.font_size_increase))
        ipi_duration_label.move(10, 615)
        self.ipi_duration = QLineEdit(f"{self.analyses_input_dict['create_usv_playback_wav']['ipi_duration']}", self.AnalysesSettings)
        self.ipi_duration.setFont(QFont(self.font_id, 10 + self.font_size_increase))
        self.ipi_duration.setStyleSheet('QLineEdit { width: 85px; }')
        self.ipi_duration.move(275, 615)

        frequency_shift_audio_segment_label = QLabel('Frequency-shift audio segment:', self.AnalysesSettings)
        frequency_shift_audio_segment_label.setFont(QFont(self.font_id, 11 + self.font_size_increase))
        frequency_shift_audio_segment_label.setStyleSheet('QLabel { color: #F58025; font-weight: bold;}')
        frequency_shift_audio_segment_label.move(10, 645)
        self.frequency_shift_audio_segment_cb = QComboBox(self.AnalysesSettings)
        self.frequency_shift_audio_segment_cb.addItems(['No', 'Yes'])
        self.frequency_shift_audio_segment_cb.setStyleSheet('QComboBox { width: 57px; }')
        self.frequency_shift_audio_segment_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='frequency_shift_audio_segment_cb_bool'))
        self.frequency_shift_audio_segment_cb.move(275, 645)

        frequency_shift_audio_dir_label = QLabel('.WAV audio subdirectory of choice:', self.AnalysesSettings)
        frequency_shift_audio_dir_label.setFont(QFont(self.font_id, 12 + self.font_size_increase))
        frequency_shift_audio_dir_label.move(10, 675)
        self.frequency_shift_audio_dir_list = sorted(['cropped_to_video', 'hpss', 'hpss_filtered'], key=lambda x: x == self.analyses_input_dict['frequency_shift_audio_segment']['fs_audio_dir'], reverse=True)
        self.frequency_shift_audio_dir_cb = QComboBox(self.AnalysesSettings)
        self.frequency_shift_audio_dir_cb.addItems(self.frequency_shift_audio_dir_list)
        self.frequency_shift_audio_dir_cb.setStyleSheet('QComboBox { width: 57px; }')
        self.frequency_shift_audio_dir_cb.activated.connect(partial(self._combo_box_fs_audio_dir, variable_id='fs_audio_dir'))
        self.frequency_shift_audio_dir_cb.move(275, 675)

        frequency_shift_device_id_label = QLabel('Recording device identity (m|s):', self.AnalysesSettings)
        frequency_shift_device_id_label.setFont(QFont(self.font_id, 12 + self.font_size_increase))
        frequency_shift_device_id_label.move(10, 705)
        self.frequency_shift_device_id_list = sorted(['m', 's'], key=lambda x: x == self.analyses_input_dict['frequency_shift_audio_segment']['fs_device_id'], reverse=True)
        self.frequency_shift_device_id_cb = QComboBox(self.AnalysesSettings)
        self.frequency_shift_device_id_cb.addItems(self.frequency_shift_device_id_list)
        self.frequency_shift_device_id_cb.setStyleSheet('QComboBox { width: 57px; }')
        self.frequency_shift_device_id_cb.activated.connect(partial(self._combo_box_fs_device_id, variable_id='fs_device_id'))
        self.frequency_shift_device_id_cb.move(275, 705)

        frequency_shift_channel_id_label = QLabel('Recording device channel (1-12):', self.AnalysesSettings)
        frequency_shift_channel_id_label.setFont(QFont(self.font_id, 12 + self.font_size_increase))
        frequency_shift_channel_id_label.move(10, 735)
        self.frequency_shift_channel_id_list = sorted(list(range(1, 13)), key=lambda x: x == self.analyses_input_dict['frequency_shift_audio_segment']['fs_channel_id'], reverse=True)
        self.frequency_shift_channel_id_cb = QComboBox(self.AnalysesSettings)
        self.frequency_shift_channel_id_cb.addItems([str(ch_id_item) for ch_id_item in self.frequency_shift_channel_id_list])
        self.frequency_shift_channel_id_cb.setStyleSheet('QComboBox { width: 57px; }')
        self.frequency_shift_channel_id_cb.activated.connect(partial(self._combo_box_fs_channel_id, variable_id='fs_channel_id'))
        self.frequency_shift_channel_id_cb.move(275, 735)

        fs_sequence_start_label = QLabel('Start of audio sequence (s):', self.AnalysesSettings)
        fs_sequence_start_label.setFont(QFont(self.font_id, 12 + self.font_size_increase))
        fs_sequence_start_label.move(10, 765)
        self.fs_sequence_start = QLineEdit(f"{self.analyses_input_dict['frequency_shift_audio_segment']['fs_sequence_start']}", self.AnalysesSettings)
        self.fs_sequence_start.setFont(QFont(self.font_id, 10 + self.font_size_increase))
        self.fs_sequence_start.setStyleSheet('QLineEdit { width: 85px; }')
        self.fs_sequence_start.move(275, 765)

        fs_sequence_duration_label = QLabel('Total duration of audio sequence (s):', self.AnalysesSettings)
        fs_sequence_duration_label.setFont(QFont(self.font_id, 12 + self.font_size_increase))
        fs_sequence_duration_label.move(10, 795)
        self.fs_sequence_duration = QLineEdit(f"{self.analyses_input_dict['frequency_shift_audio_segment']['fs_sequence_duration']}", self.AnalysesSettings)
        self.fs_sequence_duration.setFont(QFont(self.font_id, 10 + self.font_size_increase))
        self.fs_sequence_duration.setStyleSheet('QLineEdit { width: 85px; }')
        self.fs_sequence_duration.move(275, 795)

        fs_octave_shift_label = QLabel('Octave shift (direction and quantity):', self.AnalysesSettings)
        fs_octave_shift_label.setFont(QFont(self.font_id, 12 + self.font_size_increase))
        fs_octave_shift_label.move(10, 825)
        self.fs_octave_shift = QLineEdit(f"{self.analyses_input_dict['frequency_shift_audio_segment']['fs_octave_shift']}", self.AnalysesSettings)
        self.fs_octave_shift.setFont(QFont(self.font_id, 10 + self.font_size_increase))
        self.fs_octave_shift.setStyleSheet('QLineEdit { width: 85px; }')
        self.fs_octave_shift.move(275, 825)

        volume_adjust_audio_segment_label = QLabel('Volume-adjust audio segment:', self.AnalysesSettings)
        volume_adjust_audio_segment_label.setFont(QFont(self.font_id, 11 + self.font_size_increase))
        volume_adjust_audio_segment_label.move(10, 855)
        self.volume_adjust_audio_segment_cb = QComboBox(self.AnalysesSettings)
        self.volume_adjust_audio_segment_cb.addItems(['Yes', 'No'])
        self.volume_adjust_audio_segment_cb.setStyleSheet('QComboBox { width: 57px; }')
        self.volume_adjust_audio_segment_cb.activated.connect(partial(self._combo_box_prior_true, variable_id='volume_adjust_audio_segment_cb_bool'))
        self.volume_adjust_audio_segment_cb.move(275, 855)

        self._create_buttons_analyze(seq=0, class_option=self.AnalysesSettings,
                                     button_pos_y=analyze_one_y - 35, next_button_x_pos=analyze_one_x - 100)

    def analyze_two(self) -> None:
        """
        Initializes the usv-playpen Analyze Two window.

        Parameters
        ----------
        ----------

        Returns
        -------
        -------
        """

        self.ConductAnalyses = ConductAnalyses(self)
        self.setWindowTitle(f'{app_name} (Conduct Analyses)')
        self.setCentralWidget(self.ConductAnalyses)
        analyze_two_x, analyze_two_y = (870, 800)
        self.setFixedSize(analyze_two_x, analyze_two_y)

        self.txt_edit_analyze = QPlainTextEdit(self.ConductAnalyses)
        self.txt_edit_analyze.move(5, 5)
        self.txt_edit_analyze.setFixedSize(855, 740)
        self.txt_edit_analyze.setReadOnly(True)

        with open((Path(__file__).parent / '_parameter_settings/analyses_settings.json'), 'w') as analyses_settings_file:
            json.dump(self.analyses_input_dict, analyses_settings_file, indent=2)

        self.run_analyses = Analyst(message_output=self._analyses_message,
                                    input_parameter_dict=self.analyses_input_dict,
                                    root_directories=self.analyses_input_dict['analyze_data']['root_directories'])

        self._create_buttons_analyze(seq=1, class_option=self.ConductAnalyses,
                                     button_pos_y=analyze_two_y - 35, next_button_x_pos=analyze_two_x - 100)

    def _save_analyses_labels_func(self) -> None:
        """
        Transfers Analyses variables to analyses_settings dictionary.

        Parameters
        ----------
        ----------

        Returns
        -------
        -------
        """

        if os.name == 'nt':
            self.analyses_dir_edit = self.analyses_dir_edit.toPlainText().replace(os.sep, '\\')
        else:
            self.analyses_dir_edit = self.analyses_dir_edit.toPlainText()

        if len(self.analyses_dir_edit) == 0:
            self.analyses_dir_edit = []
        else:
            self.analyses_dir_edit = self.analyses_dir_edit.split('\n')

        self.analyses_input_dict['analyze_data']['root_directories'] = self.analyses_dir_edit

        self.analyses_input_dict['send_email']['experimenter'] = f'{self.exp_id}'
        self.analyses_input_dict['send_email']['analyses_pc_choice'] = str(getattr(self, 'analyses_pc_choice'))

        self.pc_usage_analyses = self.pc_usage_analyses.text()
        if len(self.pc_usage_analyses) == 0:
            self.pc_usage_analyses = []
        else:
            self.pc_usage_analyses = self.pc_usage_analyses.split(',')
        self.analyses_input_dict['send_email']['send_message']['receivers'] = self.pc_usage_analyses

        self.analyses_input_dict['analyses_booleans']['compute_behavioral_features_bool'] = self.compute_behavioral_features_cb_bool
        self.compute_behavioral_features_cb_bool = False

        self.analyses_input_dict['analyses_booleans']['compute_behavioral_tuning_bool'] = self.calculate_neuronal_tuning_curves_cb_bool
        self.calculate_neuronal_tuning_curves_cb_bool = False

        self.analyses_input_dict['analyses_booleans']['create_usv_playback_wav_bool'] = self.create_usv_playback_wav_cb_bool
        self.create_usv_playback_wav_cb_bool = False

        self.analyses_input_dict['analyses_booleans']['frequency_shift_audio_segment_bool'] = self.frequency_shift_audio_segment_cb_bool
        self.frequency_shift_audio_segment_cb_bool = False

        self.analyses_input_dict['create_usv_playback_wav']['num_usv_files'] = int(ast.literal_eval(self.num_usv_files.text()))
        self.analyses_input_dict['create_usv_playback_wav']['total_usv_number'] = int(ast.literal_eval(self.total_usv_number.text()))
        self.analyses_input_dict['create_usv_playback_wav']['ipi_duration'] = float(ast.literal_eval(self.ipi_duration.text()))

        self.analyses_input_dict['frequency_shift_audio_segment']['fs_audio_dir'] = self.fs_audio_dir
        self.analyses_input_dict['frequency_shift_audio_segment']['fs_device_id'] = self.fs_device_id
        self.analyses_input_dict['frequency_shift_audio_segment']['fs_channel_id'] = self.fs_channel_id

        self.analyses_input_dict['frequency_shift_audio_segment']['fs_sequence_start'] = float(ast.literal_eval(self.fs_sequence_start.text()))
        self.analyses_input_dict['frequency_shift_audio_segment']['fs_sequence_duration'] = float(ast.literal_eval(self.fs_sequence_duration.text()))
        self.analyses_input_dict['frequency_shift_audio_segment']['fs_octave_shift'] = int(ast.literal_eval(self.fs_octave_shift.text()))

        self.analyses_input_dict['frequency_shift_audio_segment']['fs_volume_adjustment'] = self.volume_adjust_audio_segment_cb_bool
        self.volume_adjust_audio_segment_cb_bool = True

    def visualize_one(self) -> None:
        """
        Initializes the usv-playpen Visualize One window.

        Parameters
        ----------
        ----------

        Returns
        -------
        -------
        """

        self.VisualizationsSettings = VisualizationsSettings(self)
        self.setWindowTitle(f'{app_name} (Visualize data > Settings)')
        self.setCentralWidget(self.VisualizationsSettings)
        visualize_one_x, visualize_one_y = (770, 740)
        self.setFixedSize(visualize_one_x, visualize_one_y)

        visualizations_dir_label = QLabel('(*) Root directories for visualizations', self.VisualizationsSettings)
        visualizations_dir_label.setFont(QFont(self.font_id, 13+self.font_size_increase))
        visualizations_dir_label.setStyleSheet('QLabel { font-weight: bold;}')
        visualizations_dir_label.move(50, 10)
        self.visualizations_dir_edit = QTextEdit('', self.VisualizationsSettings)
        self.visualizations_dir_edit.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.visualizations_dir_edit.move(10, 40)
        self.visualizations_dir_edit.setFixedSize(350, 320)

        pc_usage_visualizations_label = QLabel('Notify e-mail(s) of PC usage:', self.VisualizationsSettings)
        pc_usage_visualizations_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        pc_usage_visualizations_label.move(10, 365)
        self.pc_usage_visualizations = QLineEdit('', self.VisualizationsSettings)
        self.pc_usage_visualizations.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.pc_usage_visualizations.setStyleSheet('QLineEdit { width: 135px; }')
        self.pc_usage_visualizations.move(225, 365)

        visualizations_pc_label = QLabel('Visualizations PC of choice:', self.VisualizationsSettings)
        visualizations_pc_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        visualizations_pc_label.move(10, 395)
        self.loaded_visualizations_pc_list = sorted(self.visualizations_input_dict['send_email']['visualizations_pc_list'], key=lambda x: x == self.visualizations_input_dict['send_email']['visualizations_pc_choice'], reverse=True)
        self.visualizations_pc_cb = QComboBox(self.VisualizationsSettings)
        self.visualizations_pc_cb.addItems(self.loaded_visualizations_pc_list)
        self.visualizations_pc_cb.setStyleSheet('QComboBox { width: 107px; }')
        self.visualizations_pc_cb.activated.connect(partial(self._combo_box_prior_visualizations_pc_choice, variable_id='visualizations_pc_choice'))
        self.visualizations_pc_cb.move(225, 395)

        dv_label = QLabel('Select data visualization', self.VisualizationsSettings)
        dv_label.setFont(QFont(self.font_id, 13+self.font_size_increase))
        dv_label.setStyleSheet('QLabel { font-weight: bold;}')
        dv_label.move(10, 435)

        plot_behavioral_features_label = QLabel('Plot 3D behavioral tuning curves:', self.VisualizationsSettings)
        plot_behavioral_features_label.setFont(QFont(self.font_id, 11 + self.font_size_increase))
        plot_behavioral_features_label.setStyleSheet('QLabel { color: #F58025; font-weight: bold;}')
        plot_behavioral_features_label.move(10, 465)
        self.plot_behavioral_features_cb = QComboBox(self.VisualizationsSettings)
        self.plot_behavioral_features_cb.addItems(['No', 'Yes'])
        self.plot_behavioral_features_cb.setStyleSheet('QComboBox { width: 57px; }')
        self.plot_behavioral_features_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='plot_behavioral_tuning_cb_bool'))
        self.plot_behavioral_features_cb.move(275, 465)

        smoothing_sd_label = QLabel('Ratemap smoothing sigma (bins):', self.VisualizationsSettings)
        smoothing_sd_label.setFont(QFont(self.font_id, 12 + self.font_size_increase))
        smoothing_sd_label.move(10, 495)
        self.smoothing_sd = QLineEdit(f"{self.visualizations_input_dict['neuronal_tuning_figures']['smoothing_sd']}", self.VisualizationsSettings)
        self.smoothing_sd.setFont(QFont(self.font_id, 10 + self.font_size_increase))
        self.smoothing_sd.setStyleSheet('QLineEdit { width: 85px; }')
        self.smoothing_sd.move(275, 495)

        occ_threshold_label = QLabel('Minimal occupancy allowed (s):', self.VisualizationsSettings)
        occ_threshold_label.setFont(QFont(self.font_id, 12 + self.font_size_increase))
        occ_threshold_label.move(10, 525)
        self.occ_threshold = QLineEdit(f"{self.visualizations_input_dict['neuronal_tuning_figures']['occ_threshold']}", self.VisualizationsSettings)
        self.occ_threshold.setFont(QFont(self.font_id, 10 + self.font_size_increase))
        self.occ_threshold.setStyleSheet('QLineEdit { width: 85px; }')
        self.occ_threshold.move(275, 525)

        vis_col_two_x1, vis_col_two_x2 = 380, 670

        make_behavioral_video_label = QLabel('Visualize 3D behavior (figure/video):', self.VisualizationsSettings)
        make_behavioral_video_label.setFont(QFont(self.font_id, 11 + self.font_size_increase))
        make_behavioral_video_label.setStyleSheet('QLabel { color: #F58025; font-weight: bold;}')
        make_behavioral_video_label.move(vis_col_two_x1, 40)
        self.make_behavioral_video_cb = QComboBox(self.VisualizationsSettings)
        self.make_behavioral_video_cb.addItems(['No', 'Yes'])
        self.make_behavioral_video_cb.setStyleSheet('QComboBox { width: 57px; }')
        self.make_behavioral_video_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='make_behavioral_video_cb_bool'))
        self.make_behavioral_video_cb.move(vis_col_two_x2, 40)

        self.arena_root_directory_edit = QLineEdit(f"{self.visualizations_input_dict['make_behavioral_videos']['arena_directory']}", self.VisualizationsSettings)
        self.arena_root_directory_edit.setPlaceholderText('Arena tracking root directory')
        self.arena_root_directory_edit.setFont(QFont(self.font_id, 10 + self.font_size_increase))
        self.arena_root_directory_edit.setStyleSheet('QLineEdit { width: 285px; }')
        self.arena_root_directory_edit.move(vis_col_two_x1, 70)
        arena_root_directory_btn = QPushButton('Browse', self.VisualizationsSettings)
        arena_root_directory_btn.setFont(QFont(self.font_id, 8 + self.font_size_increase))
        arena_root_directory_btn.move(vis_col_two_x2, 70)
        arena_root_directory_btn.setStyleSheet('QPushButton { min-width: 65px; min-height: 12px; max-width: 656px; max-height: 12px; }')
        self.arena_root_directory_btn_clicked_flag = False
        arena_root_directory_btn.clicked.connect(self._open_arena_tracking_dialog)

        self.speaker_audio_file_edit = QLineEdit(f"{self.visualizations_input_dict['make_behavioral_videos']['speaker_audio_file']}", self.VisualizationsSettings)
        self.speaker_audio_file_edit.setPlaceholderText('Speaker playback file')
        self.speaker_audio_file_edit.setFont(QFont(self.font_id, 10 + self.font_size_increase))
        self.speaker_audio_file_edit.setStyleSheet('QLineEdit { width: 285px; }')
        self.speaker_audio_file_edit.move(vis_col_two_x1, 100)
        speaker_audio_file_btn = QPushButton('Browse', self.VisualizationsSettings)
        speaker_audio_file_btn.setFont(QFont(self.font_id, 8 + self.font_size_increase))
        speaker_audio_file_btn.move(vis_col_two_x2, 100)
        speaker_audio_file_btn.setStyleSheet('QPushButton { min-width: 65px; min-height: 12px; max-width: 656px; max-height: 12px; }')
        self.speaker_audio_file_btn_clicked_flag = False
        speaker_audio_file_btn.clicked.connect(self._speaker_playback_file_dialog)

        self.sequence_audio_file_edit = QLineEdit('', self.VisualizationsSettings)
        self.sequence_audio_file_edit.setPlaceholderText('Audible USV sequence file')
        self.sequence_audio_file_edit.setFont(QFont(self.font_id, 10 + self.font_size_increase))
        self.sequence_audio_file_edit.setStyleSheet('QLineEdit { width: 285px; }')
        self.sequence_audio_file_edit.move(vis_col_two_x1, 130)
        sequence_audio_file_btn = QPushButton('Browse', self.VisualizationsSettings)
        sequence_audio_file_btn.setFont(QFont(self.font_id, 8 + self.font_size_increase))
        sequence_audio_file_btn.move(vis_col_two_x2, 130)
        sequence_audio_file_btn.setStyleSheet('QPushButton { min-width: 65px; min-height: 12px; max-width: 656px; max-height: 12px; }')
        self.sequence_audio_file_btn_clicked_flag = False
        sequence_audio_file_btn.clicked.connect(self._sequence_playback_file_dialog)

        visualization_type_cb_label = QLabel('Create data animation (or else figure):', self.VisualizationsSettings)
        visualization_type_cb_label.setFont(QFont(self.font_id, 12 + self.font_size_increase))
        visualization_type_cb_label.move(vis_col_two_x1, 160)
        self.visualization_type_cb = QComboBox(self.VisualizationsSettings)
        self.visualization_type_cb.addItems(['No', 'Yes'])
        self.visualization_type_cb.setStyleSheet('QComboBox { width: 57px; }')
        self.visualization_type_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='visualization_type_cb_bool'))
        self.visualization_type_cb.move(vis_col_two_x2, 160)

        save_fig_format_cb_label = QLabel('Save created figure to file in format:', self.VisualizationsSettings)
        save_fig_format_cb_label.setFont(QFont(self.font_id, 12 + self.font_size_increase))
        save_fig_format_cb_label.move(vis_col_two_x1, 190)
        self.fig_format_list = sorted(['eps', 'pdf', 'png', 'svg'], key=lambda x: x == self.visualizations_input_dict['make_behavioral_videos']['general_figure_specs']['fig_format'], reverse=True)
        self.save_fig_format_cb = QComboBox(self.VisualizationsSettings)
        self.save_fig_format_cb.addItems(self.fig_format_list)
        self.save_fig_format_cb.setStyleSheet('QComboBox { width: 57px; }')
        self.save_fig_format_cb.activated.connect(partial(self._combo_box_figure_format, variable_id='fig_format'))
        self.save_fig_format_cb.move(vis_col_two_x2, 190)

        video_start_time_label = QLabel('Animation/figure start time (s):', self.VisualizationsSettings)
        video_start_time_label.setFont(QFont(self.font_id, 12 + self.font_size_increase))
        video_start_time_label.move(vis_col_two_x1, 220)
        self.video_start_time = QLineEdit(f"{self.visualizations_input_dict['make_behavioral_videos']['video_start_time']}", self.VisualizationsSettings)
        self.video_start_time.setFont(QFont(self.font_id, 10 + self.font_size_increase))
        self.video_start_time.setStyleSheet('QLineEdit { width: 85px; }')
        self.video_start_time.move(vis_col_two_x2, 220)

        video_duration_label = QLabel('Total animation duration (s):', self.VisualizationsSettings)
        video_duration_label.setFont(QFont(self.font_id, 12 + self.font_size_increase))
        video_duration_label.move(vis_col_two_x1, 250)
        self.video_duration = QLineEdit(f"{self.visualizations_input_dict['make_behavioral_videos']['video_duration']}", self.VisualizationsSettings)
        self.video_duration.setFont(QFont(self.font_id, 10 + self.font_size_increase))
        self.video_duration.setStyleSheet('QLineEdit { width: 85px; }')
        self.video_duration.move(vis_col_two_x2, 250)

        plot_theme_cb_label = QLabel('Animation/figure background theme:', self.VisualizationsSettings)
        plot_theme_cb_label.setFont(QFont(self.font_id, 12 + self.font_size_increase))
        plot_theme_cb_label.move(vis_col_two_x1, 280)
        self.plot_theme_list = sorted(['light', 'dark'], key=lambda x: x == self.visualizations_input_dict['make_behavioral_videos']['plot_theme'], reverse=True)
        self.plot_theme_cb = QComboBox(self.VisualizationsSettings)
        self.plot_theme_cb.addItems(self.plot_theme_list)
        self.plot_theme_cb.setStyleSheet('QComboBox { width: 57px; }')
        self.plot_theme_cb.activated.connect(partial(self._combo_box_plot_theme, variable_id='plot_theme'))
        self.plot_theme_cb.move(vis_col_two_x2, 280)

        view_angle_cb_label = QLabel('Animation/figure viewing angle:', self.VisualizationsSettings)
        view_angle_cb_label.setFont(QFont(self.font_id, 12 + self.font_size_increase))
        view_angle_cb_label.move(vis_col_two_x1, 310)
        self.view_angle_list = sorted(['top', 'side'], key=lambda x: x == self.visualizations_input_dict['make_behavioral_videos']['view_angle'], reverse=True)
        self.view_angle_cb = QComboBox(self.VisualizationsSettings)
        self.view_angle_cb.addItems(self.view_angle_list)
        self.view_angle_cb.setStyleSheet('QComboBox { width: 57px; }')
        self.view_angle_cb.activated.connect(partial(self._combo_box_view_angle, variable_id='view_angle'))
        self.view_angle_cb.move(vis_col_two_x2, 310)

        side_azimuth_start_label = QLabel('Side view azimuth angle (°):', self.VisualizationsSettings)
        side_azimuth_start_label.setFont(QFont(self.font_id, 12 + self.font_size_increase))
        side_azimuth_start_label.move(vis_col_two_x1, 340)
        self.side_azimuth_start = QLineEdit(f"{self.visualizations_input_dict['make_behavioral_videos']['side_azimuth_start']}", self.VisualizationsSettings)
        self.side_azimuth_start.setFont(QFont(self.font_id, 10 + self.font_size_increase))
        self.side_azimuth_start.setStyleSheet('QLineEdit { width: 85px; }')
        self.side_azimuth_start.move(vis_col_two_x2, 340)

        rotate_side_view_label = QLabel('Rotate side view in animation:', self.VisualizationsSettings)
        rotate_side_view_label.setFont(QFont(self.font_id, 12 + self.font_size_increase))
        rotate_side_view_label.move(vis_col_two_x1, 370)
        self.rotate_side_view_cb = QComboBox(self.VisualizationsSettings)
        self.rotate_side_view_cb.addItems(['No', 'Yes'])
        self.rotate_side_view_cb.setStyleSheet('QComboBox { width: 57px; }')
        self.rotate_side_view_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='rotate_side_view_bool'))
        self.rotate_side_view_cb.move(vis_col_two_x2, 370)

        rotation_speed_label = QLabel('Rotation speed of side view (°/s):', self.VisualizationsSettings)
        rotation_speed_label.setFont(QFont(self.font_id, 12 + self.font_size_increase))
        rotation_speed_label.move(vis_col_two_x1, 400)
        self.rotation_speed = QLineEdit(f"{self.visualizations_input_dict['make_behavioral_videos']['rotation_speed']}", self.VisualizationsSettings)
        self.rotation_speed.setFont(QFont(self.font_id, 10 + self.font_size_increase))
        self.rotation_speed.setStyleSheet('QLineEdit { width: 85px; }')
        self.rotation_speed.move(vis_col_two_x2, 400)

        track_history_label = QLabel('Plot history of single body node:', self.VisualizationsSettings)
        track_history_label.setFont(QFont(self.font_id, 12 + self.font_size_increase))
        track_history_label.move(vis_col_two_x1, 430)
        self.track_history_cb = QComboBox(self.VisualizationsSettings)
        self.track_history_cb.addItems(['No', 'Yes'])
        self.track_history_cb.setStyleSheet('QComboBox { width: 57px; }')
        self.track_history_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='history_cb_bool'))
        self.track_history_cb.move(vis_col_two_x2, 430)

        plot_playback_speaker_label = QLabel('Plot tracking of playback speaker:', self.VisualizationsSettings)
        plot_playback_speaker_label.setFont(QFont(self.font_id, 12 + self.font_size_increase))
        plot_playback_speaker_label.move(vis_col_two_x1, 460)
        self.plot_playback_speaker_cb = QComboBox(self.VisualizationsSettings)
        self.plot_playback_speaker_cb.addItems(['No', 'Yes'])
        self.plot_playback_speaker_cb.setStyleSheet('QComboBox { width: 57px; }')
        self.plot_playback_speaker_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='speaker_cb_bool'))
        self.plot_playback_speaker_cb.move(vis_col_two_x2, 460)

        spectrogram_bool_label = QLabel('Plot audio spectrogram (>30 kHz):', self.VisualizationsSettings)
        spectrogram_bool_label.setFont(QFont(self.font_id, 12 + self.font_size_increase))
        spectrogram_bool_label.move(vis_col_two_x1, 490)
        self.spectrogram_bool_cb = QComboBox(self.VisualizationsSettings)
        self.spectrogram_bool_cb.addItems(['No', 'Yes'])
        self.spectrogram_bool_cb.setStyleSheet('QComboBox { width: 57px; }')
        self.spectrogram_bool_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='spectrogram_cb_bool'))
        self.spectrogram_bool_cb.move(vis_col_two_x2, 490)

        spectrogram_ch_label = QLabel('Audio spectrogram channel (0-23):', self.VisualizationsSettings)
        spectrogram_ch_label.setFont(QFont(self.font_id, 12 + self.font_size_increase))
        spectrogram_ch_label.move(vis_col_two_x1, 520)
        self.spectrogram_ch_list = sorted(list(range(24)), key=lambda x: x == self.visualizations_input_dict['make_behavioral_videos']['spectrogram_ch'], reverse=True)
        self.spectrogram_ch_cb = QComboBox(self.VisualizationsSettings)
        self.spectrogram_ch_cb.addItems([str(spec_ch_item) for spec_ch_item in self.spectrogram_ch_list])
        self.spectrogram_ch_cb.setStyleSheet('QComboBox { width: 57px; }')
        self.spectrogram_ch_cb.activated.connect(partial(self._combo_box_spectrogram_ch, variable_id='spectrogram_ch'))
        self.spectrogram_ch_cb.move(vis_col_two_x2, 520)

        raster_plot_bool_label = QLabel('Plot neural cluster activity raster:', self.VisualizationsSettings)
        raster_plot_bool_label.setFont(QFont(self.font_id, 12 + self.font_size_increase))
        raster_plot_bool_label.move(vis_col_two_x1, 550)
        self.raster_plot_bool_cb = QComboBox(self.VisualizationsSettings)
        self.raster_plot_bool_cb.addItems(['No', 'Yes'])
        self.raster_plot_bool_cb.setStyleSheet('QComboBox { width: 57px; }')
        self.raster_plot_bool_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='raster_plot_cb_bool'))
        self.raster_plot_bool_cb.move(vis_col_two_x2, 550)

        raster_special_units_label = QLabel('Emphasize specific unit(s) in raster plot:', self.VisualizationsSettings)
        raster_special_units_label.setFont(QFont(self.font_id, 12 + self.font_size_increase))
        raster_special_units_label.move(vis_col_two_x1, 580)
        self.raster_special_units = QLineEdit(','.join(map(str, self.visualizations_input_dict['make_behavioral_videos']['raster_special_units'])), self.VisualizationsSettings)
        self.raster_special_units.setFont(QFont(self.font_id, 10 + self.font_size_increase))
        self.raster_special_units.setStyleSheet('QLineEdit { width: 85px; }')
        self.raster_special_units.move(vis_col_two_x2, 580)

        spike_sound_bool_label = QLabel('Create a spike .WAV for special unit(s):', self.VisualizationsSettings)
        spike_sound_bool_label.setFont(QFont(self.font_id, 12 + self.font_size_increase))
        spike_sound_bool_label.move(vis_col_two_x1, 610)
        self.spike_sound_bool_cb = QComboBox(self.VisualizationsSettings)
        self.spike_sound_bool_cb.addItems(['No', 'Yes'])
        self.spike_sound_bool_cb.setStyleSheet('QComboBox { width: 57px; }')
        self.spike_sound_bool_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='spike_sound_cb_bool'))
        self.spike_sound_bool_cb.move(vis_col_two_x2, 610)

        beh_features_bool_label = QLabel('Plot 3D behavioral/social features:', self.VisualizationsSettings)
        beh_features_bool_label.setFont(QFont(self.font_id, 12 + self.font_size_increase))
        beh_features_bool_label.move(vis_col_two_x1, 640)
        self.beh_features_bool_cb = QComboBox(self.VisualizationsSettings)
        self.beh_features_bool_cb.addItems(['No', 'Yes'])
        self.beh_features_bool_cb.setStyleSheet('QComboBox { width: 57px; }')
        self.beh_features_bool_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='beh_features_cb_bool'))
        self.beh_features_bool_cb.move(vis_col_two_x2, 640)

        self._create_buttons_visualize(seq=0, class_option=self.VisualizationsSettings,
                                       button_pos_y=visualize_one_y - 35, next_button_x_pos=visualize_one_x - 100)

    def visualize_two(self) -> None:
        """
        Initializes the usv-playpen Visualize Two window.

        Parameters
        ----------
        ----------

        Returns
        -------
        -------
        """

        self.ConductVisualizations = ConductVisualizations(self)
        self.setWindowTitle(f'{app_name} (Conduct Visualizations)')
        self.setCentralWidget(self.ConductVisualizations)
        visualize_two_x, visualize_two_y = (870, 800)
        self.setFixedSize(visualize_two_x, visualize_two_y)

        self.txt_edit_visualize = QPlainTextEdit(self.ConductVisualizations)
        self.txt_edit_visualize.move(5, 5)
        self.txt_edit_visualize.setFixedSize(855, 740)
        self.txt_edit_visualize.setReadOnly(True)

        with open((Path(__file__).parent / '_parameter_settings/visualizations_settings.json'), 'w') as visualizations_settings_file:
            json.dump(self.visualizations_input_dict, visualizations_settings_file, indent=2)

        self.run_visualizations = Visualizer(message_output=self._visualizations_message,
                                             input_parameter_dict=self.visualizations_input_dict,
                                             root_directories=self.visualizations_input_dict['visualize_data']['root_directories'])

        self._create_buttons_visualize(seq=1, class_option=self.ConductVisualizations,
                                       button_pos_y=visualize_two_y - 35, next_button_x_pos=visualize_two_x - 100)

    def _save_visualizations_labels_func(self) -> None:
        """
        Transfers Visualize variables to visualizations_settings dictionary.

        Parameters
        ----------
        ----------

        Returns
        -------
        -------
        """

        if os.name == 'nt':
            self.visualizations_dir_edit = self.visualizations_dir_edit.toPlainText().replace(os.sep, '\\')
        else:
            self.visualizations_dir_edit = self.visualizations_dir_edit.toPlainText()

        if len(self.visualizations_dir_edit) == 0:
            self.visualizations_dir_edit = []
        else:
            self.visualizations_dir_edit = self.visualizations_dir_edit.split('\n')

        self.visualizations_input_dict['visualize_data']['root_directories'] = self.visualizations_dir_edit

        self.visualizations_input_dict['neuronal_tuning_figures']['smoothing_sd'] = float(ast.literal_eval(self.smoothing_sd.text()))
        self.visualizations_input_dict['neuronal_tuning_figures']['occ_threshold'] = float(ast.literal_eval(self.occ_threshold.text()))

        self.visualizations_input_dict['send_email']['experimenter'] = f'{self.exp_id}'
        self.visualizations_input_dict['send_email']['visualizations_pc_choice'] = str(getattr(self, 'visualizations_pc_choice'))

        self.pc_usage_visualizations = self.pc_usage_visualizations.text()
        if len(self.pc_usage_visualizations) == 0:
            self.pc_usage_visualizations = []
        else:
            self.pc_usage_visualizations = self.pc_usage_visualizations.split(',')
        self.visualizations_input_dict['send_email']['send_message']['receivers'] = self.pc_usage_visualizations

        self.visualizations_input_dict['visualize_booleans']['make_behavioral_tuning_figures_bool'] = self.plot_behavioral_tuning_cb_bool
        self.plot_behavioral_tuning_cb_bool = False

        self.visualizations_input_dict['visualize_booleans']['make_behavioral_videos_bool'] = self.make_behavioral_video_cb_bool
        self.make_behavioral_video_cb_bool = False

        if not self.arena_root_directory_btn_clicked_flag:
            self.visualizations_input_dict['make_behavioral_videos']['arena_directory'] = self.arena_root_directory_edit.text()

        if not self.speaker_audio_file_btn_clicked_flag:
            self.visualizations_input_dict['make_behavioral_videos']['speaker_audio_file'] = self.speaker_audio_file_edit.text()

        if not self.sequence_audio_file_btn_clicked_flag:
            self.visualizations_input_dict['make_behavioral_videos']['sequence_audio_file'] = self.sequence_audio_file_edit.text()

        self.visualizations_input_dict['make_behavioral_videos']['animate_bool'] = self.visualization_type_cb_bool
        self.visualization_type_cb_bool = False

        self.visualizations_input_dict['make_behavioral_videos']['video_start_time'] = float(ast.literal_eval(self.video_start_time.text()))
        self.visualizations_input_dict['make_behavioral_videos']['video_duration'] = float(ast.literal_eval(self.video_duration.text()))

        self.visualizations_input_dict['make_behavioral_videos']['plot_theme'] = self.plot_theme

        self.visualizations_input_dict['make_behavioral_videos']['general_figure_specs']['fig_format'] = self.fig_format
        self.visualizations_input_dict['make_behavioral_videos']['view_angle'] = self.view_angle

        self.visualizations_input_dict['make_behavioral_videos']['side_azimuth_start'] = int(ast.literal_eval(self.side_azimuth_start.text()))

        self.visualizations_input_dict['make_behavioral_videos']['rotate_side_view_bool'] = self.rotate_side_view_bool
        self.rotate_side_view_bool = False

        self.visualizations_input_dict['make_behavioral_videos']['rotation_speed'] = int(ast.literal_eval(self.rotation_speed.text()))

        self.visualizations_input_dict['make_behavioral_videos']['history_bool'] = self.history_cb_bool
        self.history_cb_bool = False

        self.visualizations_input_dict['make_behavioral_videos']['speaker_bool'] = self.speaker_cb_bool
        self.speaker_cb_bool = False

        self.visualizations_input_dict['make_behavioral_videos']['spectrogram_bool'] = self.spectrogram_cb_bool
        self.spectrogram_cb_bool = False

        self.visualizations_input_dict['make_behavioral_videos']['spectrogram_ch'] = int(self.spectrogram_ch)

        self.visualizations_input_dict['make_behavioral_videos']['raster_plot_bool'] = self.raster_plot_cb_bool
        self.raster_plot_cb_bool = False

        self.visualizations_input_dict['make_behavioral_videos']['raster_special_units'] = self.raster_special_units.text().split(',')

        self.visualizations_input_dict['make_behavioral_videos']['spike_sound_bool'] = self.spike_sound_cb_bool
        self.spike_sound_cb_bool = False

        self.visualizations_input_dict['make_behavioral_videos']['beh_features_bool'] = self.beh_features_cb_bool
        self.beh_features_cb_bool = False


    def _save_process_labels_func(self) -> None:
        """
        Transfers Processing variables to processing_settings dictionary.

        Parameters
        ----------
        ----------

        Returns
        -------
        -------
        """

        qlabel_strings = ['conversion_target_file', 'constant_rate_factor', 'ch_receiving_input',
                          'a_ch_receiving_input', 'pc_usage_process', 'min_spike_num', 'phidget_extra_data_camera',
                          'npx_ms_divergence_tolerance', 'hpss_power', 'sleap_conda', 'n_deriv_smooth',
                          'das_conda', 'das_model_base', 'das_output_type', 'smooth_scale', 'static_reference_len',
                          'weight_rigid', 'weight_weak', 'reprojection_error_threshold', 'regularization_function',
                          'segment_confidence_threshold', 'segment_minlen', 'segment_fillgap',
                          'rigid_body_constraints', 'weak_body_constraints']
        lists_in_string = ['v_camera_serial_num', 'filter_dirs', 'concat_dirs', 'stft_window_hop', 'hpss_kernel_size',
                           'hpss_margin', 'filter_freq_bounds', 'frame_restriction', 'excluded_views']

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

        if os.name == 'nt':
            self.exp_codes_edit = self.exp_codes_edit.toPlainText().replace(os.sep, '\\')
        else:
            self.exp_codes_edit = self.exp_codes_edit.toPlainText()

        if len(self.exp_codes_edit) == 0:
            self.exp_codes_edit = []
        else:
            self.exp_codes_edit = self.exp_codes_edit.split('\n')

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
        self.processing_input_dict['modify_files']['Operator']['rectify_video_fps']['encoding_preset'] = str(getattr(self, 'encoding_preset'))
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
        self.processing_input_dict['anipose_operations']['ConvertTo3D']['translate_rotate_metric']['experimental_codes'] = self.exp_codes_edit
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
        if self.rigid_body_constraints == '':
            self.processing_input_dict['anipose_operations']['ConvertTo3D']['conduct_anipose_triangulation']['rigid_body_constraints'] = []
        else:
            self.processing_input_dict['anipose_operations']['ConvertTo3D']['conduct_anipose_triangulation']['rigid_body_constraints'] = list(ast.literal_eval(self.rigid_body_constraints))
        if self.weak_body_constraints == '':
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

    def _save_record_one_labels_func(self) -> None:
        """
        Transfers Recording One settings to exp_settings dictionary.

        Parameters
        ----------
        ----------

        Returns
        -------
        -------
        """
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

    def _save_record_two_labels_func(self) -> None:
        """
        Transfers Recording Two settings to exp_settings dictionary.

        Parameters
        ----------
        ----------

        Returns
        -------
        -------
        """

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

    def _save_record_three_labels_func(self) -> None:
        """
        Transfers Recording Three settings to exp_settings dictionary.

        Parameters
        ----------
        ----------

        Returns
        -------
        -------
        """

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

    def _save_variables_based_on_exp_id(self) -> None:
        """
        Update all variable dependent on exp_id (e.g., fileserver path).

        Parameters
        ----------
        ----------

        Returns
        -------
        -------
        """

        self.config_dir_global = Path(__file__).parent / '_config'

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
        self.analyses_input_dict['send_email']['experimenter'] = f'{self.exp_id}'

    def _combo_box_fs_channel_id(self,
                                 index: int,
                                 variable_id: str = None) -> None:
        """
        Update frequency shift combo box.

        Parameters
        ----------
        index (int)
            Index of selected choice (completes automatically).
        variable_id (str)
            Attribute to be created based on the choice.
        ----------

        Returns
        -------
        -------
        """

        for idx in range(len(self.frequency_shift_channel_id_list)):
            if index == idx:
                self.__dict__[variable_id] = self.frequency_shift_channel_id_list[idx]
                break

    def _combo_box_fs_device_id(self,
                                index: int,
                                variable_id: str = None) -> None:
        """
        Frequency shift Avisoft device combo box.

        Parameters
        ----------
        index (int)
            Index of selected choice (completes automatically).
        variable_id (str)
            Attribute to be created based on the choice.
        ----------

        Returns
        -------
        -------
        """

        for idx in range(len(self.frequency_shift_device_id_list)):
            if index == idx:
                self.__dict__[variable_id] = self.frequency_shift_device_id_list[idx]
                break

    def _combo_box_fs_audio_dir(self,
                                index: int,
                                variable_id: str = None) -> None:
        """
        Frequency shift audio directory combo box.

        Parameters
        ----------
        index (int)
            Index of selected choice (completes automatically).
        variable_id (str)
            Attribute to be created based on the choice.
        ----------

        Returns
        -------
        -------
        """

        for idx in range(len(self.frequency_shift_audio_dir_list)):
            if index == idx:
                self.__dict__[variable_id] = self.frequency_shift_audio_dir_list[idx]
                break

    def _combo_box_spectrogram_ch(self,
                                  index: int,
                                  variable_id: str = None) -> None:
        """
        Spectrogram channel combo box.

        Parameters
        ----------
        index (int)
            Index of selected choice (completes automatically).
        variable_id (str)
            Attribute to be created based on the choice.
        ----------

        Returns
        -------
        -------
        """

        for idx in range(len(self.spectrogram_ch_list)):
            if index == idx:
                self.__dict__[variable_id] = self.spectrogram_ch_list[idx]
                break

    def _combo_box_view_angle(self,
                              index: int,
                              variable_id: str = None) -> None:
        """
        View angle combo box.

        Parameters
        ----------
        index (int)
            Index of selected choice (completes automatically).
        variable_id (str)
            Attribute to be created based on the choice.
        ----------

        Returns
        -------
        -------
        """

        for idx in range(len(self.view_angle_list)):
            if index == idx:
                self.__dict__[variable_id] = self.view_angle_list[idx]
                break

    def _combo_box_figure_format(self,
                                 index: int,
                                 variable_id: str = None) -> None:
        """
        Figure format combo box.

        Parameters
        ----------
         index (int)
            Index of selected choice (completes automatically).
        variable_id (str)
            Attribute to be created based on the choice.
        ----------

        Returns
        -------
        -------
        """

        for idx in range(len(self.fig_format_list)):
            if index == idx:
                self.__dict__[variable_id] = self.fig_format_list[idx]
                break

    def _combo_box_plot_theme(self,
                              index: int,
                              variable_id: str = None) -> None:
        """
        Plot theme combo box.

        Parameters
        ----------
        index (int)
            Index of selected choice (completes automatically).
        variable_id (str)
            Attribute to be created based on the choice.
        ----------

        Returns
        -------
        -------
        """

        for idx in range(len(self.plot_theme_list)):
            if index == idx:
                self.__dict__[variable_id] = self.plot_theme_list[idx]
                break

    def _combo_box_prior_transformed_tracking_data(self,
                                                   index: int,
                                                   variable_id: str = None) -> None:
        """
        Anipose transformation type combo box.

        Parameters
        ----------
         index (int)
            Index of selected choice (completes automatically).
        variable_id (str)
            Attribute to be created based on the choice.
        ----------

        Returns
        -------
        -------
        """

        if index == 0:
            self.__dict__[variable_id] = 'animal'
        else:
            self.__dict__[variable_id] = 'arena'

    def _combo_box_encoding_preset(self,
                                   index: int,
                                   variable_id: str = None) -> None:
        """
        Encoding preset combo box.

        Parameters
        ----------
         index (int)
            Index of selected choice (completes automatically).
        variable_id (str)
            Attribute to be created based on the choice.
        ----------

        Returns
        -------
        -------
        """

        for idx in range(len(self.encoding_preset_list)):
            if index == idx:
                self.__dict__[variable_id] = self.encoding_preset_list[idx]
                break

    def _combo_box_prior_name(self,
                              index: int,
                              variable_id: str = None) -> None:
        """
        Experimenter name combo box.

        Parameters
        ----------
         index (int)
            Index of selected choice (completes automatically).
        variable_id (str)
            Attribute to be created based on the choice.
        ----------

        Returns
        -------
        -------
        """

        for idx in range(len(self.exp_id_list)):
            if index == idx:
                self.__dict__[variable_id] = self.exp_id_list[idx]
                break

    def _combo_box_prior_codec(self,
                              index: int,
                              variable_id: str = None) -> None:
        """
        Recording codec combo box.

        Parameters
        ----------
         index (int)
            Index of selected choice (completes automatically).
        variable_id (str)
            Attribute to be created based on the choice.
        ----------

        Returns
        -------
        -------
        """

        for idx in range(len(self.recording_codec_list)):
            if index == idx:
                self.__dict__[variable_id] = self.recording_codec_list[idx]
                break

    def _combo_box_prior_processing_pc_choice(self,
                                              index: int,
                                              variable_id: str = None) -> None:
        """
        Processing PC of choice combo box.

        Parameters
        ----------
         index (int)
            Index of selected choice (completes automatically).
        variable_id (str)
            Attribute to be created based on the choice.
        ----------

        Returns
        -------
        -------
        """

        for idx in range(len(self.loaded_processing_pc_list)):
            if index == idx:
                self.__dict__[variable_id] = self.loaded_processing_pc_list[idx]
                break

    def _combo_box_prior_analyses_pc_choice(self,
                                            index: int,
                                            variable_id: str = None) -> None:
        """
        Analyses PC of choice combo box.

        Parameters
        ----------
         index (int)
            Index of selected choice (completes automatically).
        variable_id (str)
            Attribute to be created based on the choice.
        ----------

        Returns
        -------
        -------
        """

        for idx in range(len(self.loaded_analyses_pc_list)):
            if index == idx:
                self.__dict__[variable_id] = self.loaded_analyses_pc_list[idx]
                break

    def _combo_box_prior_visualizations_pc_choice(self,
                                                  index: int,
                                                  variable_id: str = None) -> None:
        """
        Visualizations PC of choice combo box.

        Parameters
        ----------
         index (int)
            Index of selected choice (completes automatically).
        variable_id (str)
            Attribute to be created based on the choice.
        ----------

        Returns
        -------
        -------
        """

        for idx in range(len(self.loaded_visualizations_pc_list)):
            if index == idx:
                self.__dict__[variable_id] = self.loaded_visualizations_pc_list[idx]
                break

    def _combo_box_prior_audio_device_camera_input(self,
                                                   index: int,
                                                   variable_id: str = None) -> None:
        """
        Audio device camera input combo box.

        Parameters
        ----------
         index (int)
            Index of selected choice (completes automatically).
        variable_id (str)
            Attribute to be created based on the choice.
        ----------

        Returns
        -------
        -------
        """

        if index == 0:
            self.__dict__[variable_id] = 'm'
        elif index == 1:
            self.__dict__[variable_id] = 's'
        else:
            self.__dict__[variable_id] = 'both'

    def _combo_box_prior_npx_file_type(self,
                                       index: int,
                                       variable_id: str = None) -> None:
        """
        Neuropixels file type combo box.

        Parameters
        ----------
        index (int)
            Index of selected choice (completes automatically).
        variable_id (str)
            Attribute to be created based on the choice.
        ----------

        Returns
        -------
        -------
        """

        if index == 0:
            self.__dict__[variable_id] = 'ap'
        else:
            self.__dict__[variable_id] = 'lf'

    def _combo_box_prior_true(self,
                              index: int,
                              variable_id: str = None) -> None:
        """
        Boolean (prior True) combo box.

        Parameters
        ----------
        index (int)
            Index of selected choice (completes automatically).
        variable_id (str)
            Attribute to be created based on the choice.
        ----------

        Returns
        -------
        -------
        """

        if index == 1:
            self.__dict__[variable_id] = False
        else:
            self.__dict__[variable_id] = True

    def _combo_box_prior_false(self,
                               index: int,
                               variable_id: str = None) -> None:
        """
        Boolean (prior False) combo box.

        Parameters
        ----------
        index (int)
            Index of selected choice (completes automatically).
        variable_id (str)
            Attribute to be created based on the choice.
        ----------

        Returns
        -------
        -------
        """

        if index == 1:
            self.__dict__[variable_id] = True
        else:
            self.__dict__[variable_id] = False

    def _update_exposure_time_label(self, value: int, variable_id: str = None) -> None:
        """
        Updates camera exposure time label.

        Parameters
        ----------
        value (int)
            Exposure time (in μs, completes automatically).
        variable_id (str)
            Attribute to be created based on the choice.
        ----------

        Returns
        -------
        -------
        """

        self.__dict__[variable_id].setText(f'exp time ({str(value)} μs):')

    def _update_gain_label(self, value: int, variable_id: str = None) -> None:
        """
        Updates camera digital gain label.

        Parameters
        ----------
        value (int)
            Digital gain (in dB, completes automatically).
        variable_id (str)
            Attribute to be created based on the choice.
        ----------

        Returns
        -------
        -------
        """

        self.__dict__[variable_id].setText(f'digital gain ({str(value)} dB):')

    def _update_fr_label(self, value: int) -> None:
        """
        Updates camera sampling rate for recording sessions.

        Parameters
        ----------
        value (int)
            Recording rate (in fps, completes automatically).
        ----------

        Returns
        -------
        -------
        """

        self.fr_label.setText(f'Recording ({str(value)} fps):')

    def _update_cal_fr_label(self, value: int) -> None:
        """
        Updates camera sampling rate for calibration sessions.

        Parameters
        ----------
        value (int)
            Recording rate (in fps, completes automatically).
        ----------

        Returns
        -------
        -------
        """

        self.cal_fr_label.setText(f'Calibration ({str(value)} fps):')

    def _create_sliders_general(self, camera_id: str = None, camera_color: str = None, y_start: int = None) -> None:
        """
        Creates sliders for camera exposure time and digital gain.

        Parameters
        ----------
        camera_id (str)
            Camera ID (e.g., 21372316).
        camera_color (str)
            Camera label color (e.g., green).
        y_start (int)
            Starting y position for the camera settings.
        ----------

        Returns
        -------
        -------
        """

        specific_camera_label = QLabel(f'Camera {camera_id} ({camera_color})', self.VideoSettings)
        specific_camera_label.setStyleSheet('QLabel { font-weight: bold;}')
        specific_camera_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        specific_camera_label.move(5, y_start)

        self.__dict__[f'exposure_time_{camera_id}_label'] = QLabel('exp time (2500 μs)', self.VideoSettings)
        self.__dict__[f'exposure_time_{camera_id}_label'].setFixedWidth(150)
        self.__dict__[f'exposure_time_{camera_id}_label'].setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.__dict__[f'exposure_time_{camera_id}_label'].move(25, y_start+30)
        self.__dict__[f'exposure_time_{camera_id}'] = QSlider(Qt.Orientation.Horizontal, self.VideoSettings)
        self.__dict__[f'exposure_time_{camera_id}'].setFixedWidth(150)
        self.__dict__[f'exposure_time_{camera_id}'].setRange(500, 30000)
        self.__dict__[f'exposure_time_{camera_id}'].setValue(self.__dict__[f"{camera_id}_et"])
        self.__dict__[f'exposure_time_{camera_id}'].move(5, y_start+60)
        self.__dict__[f'exposure_time_{camera_id}'].valueChanged.connect(partial(self._update_exposure_time_label, variable_id=f'exposure_time_{camera_id}_label'))

        self.__dict__[f'gain_{camera_id}_label'] = QLabel('digital gain (0 dB)', self.VideoSettings)
        self.__dict__[f'gain_{camera_id}_label'].setFixedWidth(150)
        self.__dict__[f'gain_{camera_id}_label'].setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.__dict__[f'gain_{camera_id}_label'].move(200, y_start+30)
        self.__dict__[f'gain_{camera_id}'] = QSlider(Qt.Orientation.Horizontal, self.VideoSettings)
        self.__dict__[f'gain_{camera_id}'].setFixedWidth(150)
        self.__dict__[f'gain_{camera_id}'].setRange(0, 15)
        self.__dict__[f'gain_{camera_id}'].setValue(self.__dict__[f"{camera_id}_dg"])
        self.__dict__[f'gain_{camera_id}'].move(180, y_start+60)
        self.__dict__[f'gain_{camera_id}'].valueChanged.connect(partial(self._update_gain_label, variable_id=f'gain_{camera_id}_label'))


    def _create_buttons_main(self) -> None:
        """
        Creates buttons for Main window.

        Parameters
        ----------
        ----------

        Returns
        -------
        -------
        """

        self.button_map = {'Process': QPushButton(QIcon(process_icon), 'Process', self.Main),
                           'Record': QPushButton(QIcon(record_icon), 'Record', self.Main),
                           'Analyze': QPushButton(QIcon(analyze_icon), 'Analyze', self.Main),
                           'Visualize': QPushButton(QIcon(visualize_icon), 'Visualize', self.Main)}

        self.button_map['Record'].move(120, 370)
        self.button_map['Record'].setFont(QFont(self.font_id, 8+self.font_size_increase))
        self.button_map['Record'].clicked.connect(self._save_variables_based_on_exp_id)
        self.button_map['Record'].clicked.connect(self.record_one)

        self.button_map['Process'].move(215, 370)
        self.button_map['Process'].setFont(QFont(self.font_id, 8+self.font_size_increase))
        self.button_map['Process'].clicked.connect(self._save_variables_based_on_exp_id)
        self.button_map['Process'].clicked.connect(self.process_one)

        self.button_map['Analyze'].move(120, 405)
        self.button_map['Analyze'].setFont(QFont(self.font_id, 8+self.font_size_increase))
        self.button_map['Analyze'].clicked.connect(self.analyze_one)

        self.button_map['Visualize'].move(215, 405)
        self.button_map['Visualize'].setFont(QFont(self.font_id, 8+self.font_size_increase))
        self.button_map['Visualize'].clicked.connect(self.visualize_one)

    def _create_buttons_record(self,
                               seq: int = None,
                               class_option: str = None,
                               button_pos_y: int = None,
                               next_button_x_pos: int = None) -> None:
        """
        Creates buttons for Record windows.

        Parameters
        ----------
        seq (int)
            Sequence number of the window.
        class_option (str)
            Class option for the button.
        button_pos_y (int)
            Y position for the button.
        next_button_x_pos (int)
            X position for the next button.
        ----------

        Returns
        -------
        -------
        """

        if seq == 0:
            previous_win = self.main_window
            next_win_connect = [self._save_record_one_labels_func, self.record_two]
        elif seq == 1:
            previous_win = self.record_one
            next_win_connect = [self._save_record_two_labels_func, self.record_three]
        elif seq == 2:
            previous_win = self.record_two
            next_win_connect = [self._save_record_three_labels_func, self.record_four]
        else:
            previous_win = self.record_three
            next_win_connect = []

        self.button_map = {'Previous': QPushButton(QIcon(previous_icon), 'Previous', class_option),
                           'Main': QPushButton(QIcon(main_icon), 'Main', class_option)}

        self.button_map['Previous'].move(5, button_pos_y)
        self.button_map['Previous'].setFont(QFont(self.font_id, 8+self.font_size_increase))
        self.button_map['Previous'].clicked.connect(previous_win)

        self.button_map['Main'].move(100, button_pos_y)
        self.button_map['Main'].setFont(QFont(self.font_id, 8+self.font_size_increase))
        self.button_map['Main'].clicked.connect(self.main_window)

        if len(next_win_connect) > 0:
            self.button_map['Next'] = QPushButton(QIcon(next_icon), 'Next', class_option)
            self.button_map['Next'].move(next_button_x_pos, button_pos_y)
            self.button_map['Next'].setFont(QFont(self.font_id, 8+self.font_size_increase))
            for one_connection in next_win_connect:
                self.button_map['Next'].clicked.connect(one_connection)
        else:
            if self.exp_settings_dict['conduct_tracking_calibration']:
                self.button_map['Calibrate'] = QPushButton(QIcon(calibrate_icon), 'Calibrate', class_option)
                self.button_map['Calibrate'].move(next_button_x_pos-95, button_pos_y)
                self.button_map['Calibrate'].setFont(QFont(self.font_id, 8+self.font_size_increase))
                self.button_map['Calibrate'].clicked.connect(self._disable_other_buttons)
                self.button_map['Calibrate'].clicked.connect(self._start_calibration)
                self.button_map['Calibrate'].clicked.connect(self._enable_other_buttons_post_cal)

            self.button_map['Record'] = QPushButton(QIcon(record_icon), 'Record', class_option)
            self.button_map['Record'].move(next_button_x_pos, button_pos_y)
            self.button_map['Record'].setFont(QFont(self.font_id, 8+self.font_size_increase))
            self.button_map['Record'].clicked.connect(self._disable_other_buttons)
            self.button_map['Record'].clicked.connect(self._start_recording)
            self.button_map['Record'].clicked.connect(self._enable_other_buttons_post_rec)

    def _create_buttons_process(self,
                                seq: int = None,
                                class_option: str = None,
                                button_pos_y: int = None,
                                next_button_x_pos: int = None) -> None:
        """
        Creates buttons for Process windows.

        Parameters
        ----------
        seq (int)
            Sequence number of the window.
        class_option (str)
            Class option for the button.
        button_pos_y (int)
            Y position for the button.
        next_button_x_pos (int)
            X position for the next button.
        ----------

        Returns
        -------
        -------
        """

        if seq == 0:
            previous_win = self.main_window
            next_win_connect = [self._save_process_labels_func, self.process_two]
        else:
            previous_win = self.process_one

        self.button_map = {'Previous': QPushButton(QIcon(previous_icon), 'Previous', class_option),
                           'Main': QPushButton(QIcon(main_icon), 'Main', class_option)}

        self.button_map['Previous'].move(5, button_pos_y)
        self.button_map['Previous'].setFont(QFont(self.font_id, 8+self.font_size_increase))
        self.button_map['Previous'].clicked.connect(previous_win)

        self.button_map['Main'].move(100, button_pos_y)
        self.button_map['Main'].setFont(QFont(self.font_id, 8+self.font_size_increase))
        self.button_map['Main'].clicked.connect(self.main_window)

        if seq == 0:
            self.button_map['Next'] = QPushButton(QIcon(next_icon), 'Next', class_option)
            self.button_map['Next'].move(next_button_x_pos, button_pos_y)
            self.button_map['Next'].setFont(QFont(self.font_id, 8+self.font_size_increase))
            for one_connection in next_win_connect:
                self.button_map['Next'].clicked.connect(one_connection)
        else:
            self.button_map['Process'] = QPushButton(QIcon(process_icon), 'Process', class_option)
            self.button_map['Process'].move(next_button_x_pos, button_pos_y)
            self.button_map['Process'].setFont(QFont(self.font_id, 8+self.font_size_increase))
            self.button_map['Process'].clicked.connect(self._disable_process_buttons)
            self.button_map['Process'].clicked.connect(self._start_processing)
            self.button_map['Process'].clicked.connect(self._enable_process_buttons)

    def _create_buttons_analyze(self,
                                seq: int = None,
                                class_option: str = None,
                                button_pos_y: int = None,
                                next_button_x_pos: int = None) -> None:
        """
        Creates buttons for Analyses windows.

        Parameters
        ----------
        seq (int)
            Sequence number of the window.
        class_option (str)
            Class option for the button.
        button_pos_y (int)
            Y position for the button.
        next_button_x_pos (int)
            X position for the next button.
        ----------

        Returns
        -------
        -------
        """

        if seq == 0:
            previous_win = self.main_window
            next_window_connect = [self._save_analyses_labels_func, self.analyze_two]
        else:
            previous_win = self.analyze_one

        self.button_map = {'Previous': QPushButton(QIcon(previous_icon), 'Previous', class_option),
                           'Main': QPushButton(QIcon(main_icon), 'Main', class_option)}

        self.button_map['Previous'].move(5, button_pos_y)
        self.button_map['Previous'].setFont(QFont(self.font_id, 8+self.font_size_increase))
        self.button_map['Previous'].clicked.connect(previous_win)

        self.button_map['Main'].move(100, button_pos_y)
        self.button_map['Main'].setFont(QFont(self.font_id, 8+self.font_size_increase))
        self.button_map['Main'].clicked.connect(self.main_window)

        if seq == 0:
            self.button_map['Next'] = QPushButton(QIcon(next_icon), 'Next', class_option)
            self.button_map['Next'].move(next_button_x_pos, button_pos_y)
            self.button_map['Next'].setFont(QFont(self.font_id, 8+self.font_size_increase))
            for one_connection in next_window_connect:
                self.button_map['Next'].clicked.connect(one_connection)
        else:
            self.button_map['Analyze'] = QPushButton(QIcon(analyze_icon), 'Analyze', class_option)
            self.button_map['Analyze'].move(next_button_x_pos, button_pos_y)
            self.button_map['Analyze'].setFont(QFont(self.font_id, 8+self.font_size_increase))
            self.button_map['Analyze'].clicked.connect(self._disable_analyze_buttons)
            self.button_map['Analyze'].clicked.connect(self._start_analyses)
            self.button_map['Analyze'].clicked.connect(self._enable_analyze_buttons)

    def _create_buttons_visualize(self,
                                  seq: int = None,
                                  class_option: str = None,
                                  button_pos_y: int = None,
                                  next_button_x_pos: int = None) -> None:
        """
        Creates buttons for Visualize windows.

        Parameters
        ----------
        seq (int)
            Sequence number of the window.
        class_option (str)
            Class option for the button.
        button_pos_y (int)
            Y position for the button.
        next_button_x_pos (int)
            X position for the next button.
        ----------

        Returns
        -------
        -------
        """

        if seq == 0:
            previous_win = self.main_window
            next_window_connect = [self._save_visualizations_labels_func, self.visualize_two]
        else:
            previous_win = self.visualize_one

        self.button_map = {'Previous': QPushButton(QIcon(previous_icon), 'Previous', class_option),
                           'Main': QPushButton(QIcon(main_icon), 'Main', class_option)}

        self.button_map['Previous'].move(5, button_pos_y)
        self.button_map['Previous'].setFont(QFont(self.font_id, 8+self.font_size_increase))
        self.button_map['Previous'].clicked.connect(previous_win)

        self.button_map['Main'].move(100, button_pos_y)
        self.button_map['Main'].setFont(QFont(self.font_id, 8+self.font_size_increase))
        self.button_map['Main'].clicked.connect(self.main_window)

        if seq == 0:
            self.button_map['Next'] = QPushButton(QIcon(next_icon), 'Next', class_option)
            self.button_map['Next'].move(next_button_x_pos, button_pos_y)
            self.button_map['Next'].setFont(QFont(self.font_id, 8+self.font_size_increase))
            for one_connection in next_window_connect:
                self.button_map['Next'].clicked.connect(one_connection)
        else:
            self.button_map['Visualize'] = QPushButton(QIcon(visualize_icon), 'Visualize', class_option)
            self.button_map['Visualize'].move(next_button_x_pos, button_pos_y)
            self.button_map['Visualize'].setFont(QFont(self.font_id, 8+self.font_size_increase))
            self.button_map['Visualize'].clicked.connect(self._disable_visualize_buttons)
            self.button_map['Visualize'].clicked.connect(self._start_visualizations)
            self.button_map['Visualize'].clicked.connect(self._enable_visualize_buttons)

    def _start_visualizations(self) -> None:
        """
        Runs visualizations.

        Parameters
        ----------
        ----------

        Returns
        -------
        -------
        """

        self.run_visualizations.visualize_data()

    def _start_analyses(self) -> None:
        """
        Runs analyses.

        Parameters
        ----------
        ----------

        Returns
        -------
        -------
        """

        self.run_analyses.analyze_data()

    def _start_processing(self) -> None:
        """
        Runs processing.

        Parameters
        ----------
        ----------

        Returns
        -------
        -------
        """

        self.run_processing.prepare_data_for_analyses()

    def _start_calibration(self) -> None:
        """
        Runs calibration.

        Parameters
        ----------
        ----------

        Returns
        -------
        -------
        """

        self.run_exp.conduct_tracking_calibration()

    def _start_recording(self) -> None:
        """
        Runs recording.

        Parameters
        ----------
        ----------

        Returns
        -------
        -------
        """

        self.run_exp.conduct_behavioral_recording()

    def _enable_visualize_buttons(self) -> None:
        """
        Enables visualize buttons.

        Parameters
        ----------
        ----------

        Returns
        -------
        -------
        """

        self.button_map['Previous'].setEnabled(True)
        self.button_map['Main'].setEnabled(True)
        self.button_map['Visualize'].setEnabled(False)

    def _disable_visualize_buttons(self) -> None:
        """
        Disables visualize buttons.

        Parameters
        ----------
        ----------

        Returns
        -------
        -------
        """

        self.button_map['Previous'].setEnabled(False)
        self.button_map['Main'].setEnabled(False)
        self.button_map['Visualize'].setEnabled(False)

    def _enable_analyze_buttons(self) -> None:
        """
        Enables analyze buttons.

        Parameters
        ----------
        ----------

        Returns
        -------
        -------
        """

        self.button_map['Previous'].setEnabled(True)
        self.button_map['Main'].setEnabled(True)
        self.button_map['Analyze'].setEnabled(False)

    def _disable_analyze_buttons(self) -> None:
        """
        Disables analyze buttons.

        Parameters
        ----------
        ----------

        Returns
        -------
        -------
        """

        self.button_map['Previous'].setEnabled(False)
        self.button_map['Main'].setEnabled(False)
        self.button_map['Analyze'].setEnabled(False)

    def _enable_process_buttons(self) -> None:
        """
        Enables process buttons.

        Parameters
        ----------
        ----------

        Returns
        -------
        -------
        """

        self.button_map['Previous'].setEnabled(True)
        self.button_map['Main'].setEnabled(True)
        self.button_map['Process'].setEnabled(False)

    def _disable_process_buttons(self) -> None:
        """
        Disables process buttons.

        Parameters
        ----------
        ----------

        Returns
        -------
        -------
        """

        self.button_map['Previous'].setEnabled(False)
        self.button_map['Main'].setEnabled(False)
        self.button_map['Process'].setEnabled(False)

    def _enable_other_buttons_post_cal(self) -> None:
        """
        Enables buttons after calibration.

        Parameters
        ----------
        ----------

        Returns
        -------
        -------
        """

        self.button_map['Main'].setEnabled(True)
        self.button_map['Record'].setEnabled(True)

    def _enable_other_buttons_post_rec(self):
        """
        Enables buttons after recording.

        Parameters
        ----------
        ----------

        Returns
        -------
        -------
        """

        self.button_map['Main'].setEnabled(True)

    def _disable_other_buttons(self) -> None:
        """
        Disables all buttons.

        Parameters
        ----------
        ----------

        Returns
        -------
        -------
        """

        self.button_map['Previous'].setEnabled(False)
        self.button_map['Main'].setEnabled(False)
        self.button_map['Record'].setEnabled(False)
        if self.exp_settings_dict['conduct_tracking_calibration']:
            self.button_map['Calibrate'].setEnabled(False)

    def _sequence_playback_file_dialog(self) -> None:
        """
        Creates dialog for audible audio sequence.

        Parameters
        ----------
        ----------

        Returns
        -------
        -------
        """

        self.sequence_audio_file_btn_clicked_flag = True
        sequence_audio_file_name = QFileDialog.getOpenFileNames(
            self,
            'Select audible USV sequence .WAV file',
            '',
            'Wave Files (*.wav)')
        if sequence_audio_file_name:
            sequence_audio_file_name_path = Path(sequence_audio_file_name[0][0])
            self.sequence_audio_file_edit.setText(str(sequence_audio_file_name_path))
            if os.name == 'nt':
                self.visualizations_input_dict['make_behavioral_videos']['sequence_audio_file'] = str(sequence_audio_file_name_path).replace(os.sep, '\\')
            else:
                self.visualizations_input_dict['make_behavioral_videos']['sequence_audio_file'] = str(sequence_audio_file_name_path)
        else:
            self.visualizations_input_dict['make_behavioral_videos']['sequence_audio_file'] = self.sequence_audio_file_edit.text()

    def _speaker_playback_file_dialog(self) -> None:
        """
        Creates dialog for speaker playback.

        Parameters
        ----------
        ----------

        Returns
        -------
        -------
        """

        self.speaker_audio_file_btn_clicked_flag = True
        speaker_audio_file_name = QFileDialog.getOpenFileNames(
            self,
            'Select playback speaker .WAV file',
            '',
            'Wave Files (*.wav)')
        if speaker_audio_file_name:
            speaker_audio_file_name_path = Path(speaker_audio_file_name[0][0])
            self.speaker_audio_file_edit.setText(str(speaker_audio_file_name_path))
            if os.name == 'nt':
                self.visualizations_input_dict['make_behavioral_videos']['speaker_audio_file'] = str(speaker_audio_file_name_path).replace(os.sep, '\\')
            else:
                self.visualizations_input_dict['make_behavioral_videos']['speaker_audio_file'] = str(speaker_audio_file_name_path)
        else:
            self.visualizations_input_dict['make_behavioral_videos']['speaker_audio_file'] = self.speaker_audio_file_edit.text()

    def _open_arena_tracking_dialog(self) -> None:
        """
        Creates dialog for arena tracking.

        Parameters
        ----------
        ----------

        Returns
        -------
        -------
        """

        self.arena_root_directory_btn_clicked_flag = True
        arena_tracking_dir_name = QFileDialog.getExistingDirectory(
            self,
            'Select arena tracking root directory',
            '')
        if arena_tracking_dir_name:
            arena_tracking_dir_name_path = Path(arena_tracking_dir_name)
            self.arena_root_directory_edit.setText(str(arena_tracking_dir_name_path))
            if os.name == 'nt':
                self.visualizations_input_dict['make_behavioral_videos']['arena_directory'] = str(arena_tracking_dir_name_path).replace(os.sep, '\\')
            else:
                self.visualizations_input_dict['make_behavioral_videos']['arena_directory'] = str(arena_tracking_dir_name_path)
        else:
            self.visualizations_input_dict['make_behavioral_videos']['arena_directory'] = self.arena_root_directory_edit.text()

    def _open_centroid_dialog(self) -> None:
        """
        Creates dialog for SLEAP centroid model.

        Parameters
        ----------
        ----------

        Returns
        -------
        -------
        """

        self.centroid_model_btn_clicked_flag = True
        centroid_dir_name = QFileDialog.getExistingDirectory(
            self,
            'Select SLEAP centroid model directory',
            '')
        if centroid_dir_name:
            centroid_dir_name_path = Path(centroid_dir_name)
            self.centroid_model_edit.setText(str(centroid_dir_name_path))
            if os.name == 'nt':
                self.processing_input_dict['prepare_cluster_job']['centroid_model_path'] = str(centroid_dir_name_path).replace(os.sep, '\\') + '\\'
            else:
                self.processing_input_dict['prepare_cluster_job']['centroid_model_path'] = str(centroid_dir_name_path)
        else:
            self.processing_input_dict['prepare_cluster_job']['centroid_model_path'] = self.centroid_model_edit.text()

    def _open_centered_instance_dialog(self) -> None:
        """
        Creates dialog for SLEAP centered instance model.

        Parameters
        ----------
        ----------

        Returns
        -------
        -------
        """

        self.centered_instance_btn_btn_clicked_flag = True
        centered_instance_dir_name = QFileDialog.getExistingDirectory(
            self,
            'Select SLEAP centered instance model directory',
            '')
        if centered_instance_dir_name:
            centered_instance_dir_name_path = Path(centered_instance_dir_name)
            self.centered_instance_model_edit.setText(str(centered_instance_dir_name_path))
            if os.name == 'nt':
                self.processing_input_dict['prepare_cluster_job']['centered_instance_model_path'] = str(centered_instance_dir_name_path).replace(os.sep, '\\') + '\\'
            else:
                self.processing_input_dict['prepare_cluster_job']['centered_instance_model_path'] = str(centered_instance_dir_name_path)
        else:
            self.processing_input_dict['prepare_cluster_job']['centered_instance_model_path'] = self.centered_instance_model_edit.text()

    def _open_inference_root_dialog(self) -> None:
        """
        Creates dialog for SLEAP inference directory.

        Parameters
        ----------
        ----------

        Returns
        -------
        -------
        """

        self.inference_root_dir_btn_clicked_flag = True
        inference_root_dir_name = QFileDialog.getExistingDirectory(
            self,
            'Select SLEAP inference directory',
            '')
        if inference_root_dir_name:
            inference_root_dir_name_path = Path(inference_root_dir_name)
            self.inference_root_dir_edit.setText(str(inference_root_dir_name_path))
            if os.name == 'nt':
                self.processing_input_dict['prepare_cluster_job']['inference_root_dir'] = str(inference_root_dir_name_path).replace(os.sep, '\\') + '\\'
            else:
                self.processing_input_dict['prepare_cluster_job']['inference_root_dir'] = str(inference_root_dir_name_path)
        else:
            self.processing_input_dict['prepare_cluster_job']['inference_root_dir'] = self.inference_root_dir_edit.text()

    def _open_recorder_dialog(self) -> None:
        """
        Creates dialog for Avisoft USGH Recorder directory.

        Parameters
        ----------
        ----------

        Returns
        -------
        -------
        """

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

    def _open_avisoft_dialog(self) -> None:
        """
        Creates dialog for Avisoft base directory.

        Parameters
        ----------
        ----------

        Returns
        -------
        -------
        """

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

    def _open_coolterm_dialog(self) -> None:
        """
        Creates dialog for Coolterm directory.

        Parameters
        ----------
        ----------

        Returns
        -------
        -------
        """

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

    def _open_anipose_calibration_dialog(self) -> None:
        """
        Creates dialog for Anipose calibration directory.

        Parameters
        ----------
        ----------

        Returns
        -------
        -------
        """

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

    def _open_das_model_dialog(self) -> None:
        """
        Creates dialog for DAS model directory.

        Parameters
        ----------
        ----------

        Returns
        -------
        -------
        """

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

    def _location_on_the_screen(self) -> None:
        """
        Places GUI in top left corner of screen.

        Parameters
        ----------
        ----------

        Returns
        -------
        -------
        """

        top_left_point = QGuiApplication.primaryScreen().availableGeometry().topLeft()
        self.move(top_left_point)

    def _message(self, s: str) -> None:
        """
        Creates messages displayed during recording.

        Parameters
        ----------
        ----------

        Returns
        -------
        -------
        """

        self.txt_edit.appendPlainText(s)

    def _analyses_message(self, s: str) -> None:
        """
        Creates messages displayed during analyses.

        Parameters
        ----------
        ----------

        Returns
        -------
        -------
        """

        self.txt_edit_analyze.appendPlainText(s)

    def _visualizations_message(self, s: str) -> None:
        """
        Creates messages displayed during visualizations.

        Parameters
        ----------
        ----------

        Returns
        -------
        -------
        """

        self.txt_edit_visualize.appendPlainText(s)

    def _process_message(self, s: str) -> None:
        """
        Creates messages displayed during processing.

        Parameters
        ----------
        ----------

        Returns
        -------
        -------
        """

        self.txt_edit_process.appendPlainText(s)


def main() -> None:
    """
    Creates GUI application.

    Parameters
    ----------
    ----------

    Returns
    -------
    -------
    """

    # Handle high resolution displays:
    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, on=True)
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, on=True)

    usv_playpen_app = QApplication([])

    usv_playpen_app.setStyle('Fusion')

    with open((Path(__file__).parent / '_config/gui_style_sheet.css') , 'r') as file:
        usv_playpen_app.setStyleSheet(file.read())

    usv_playpen_app.setWindowIcon(QIcon(gui_icon))

    _toml = toml.load(Path(__file__).parent / '_config/behavioral_experiments_settings.toml')

    with open((Path(__file__).parent / '_parameter_settings/processing_settings.json'), 'r') as processing_json_file:
        processing_input_dict = json.load(processing_json_file)

    with open((Path(__file__).parent / '_parameter_settings/analyses_settings.json'), 'r') as analyses_json_file:
        analyses_input_dict = json.load(analyses_json_file)

    with open((Path(__file__).parent / '_parameter_settings/visualizations_settings.json'), 'r') as visualizations_json_file:
        visualizations_input_dict = json.load(visualizations_json_file)


    splash = QSplashScreen(QPixmap(splash_icon))
    splash.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
    splash.show()
    QTest.qWait(5000)

    initial_values_dict = {'exp_id': _toml['video']['metadata']['experimenter'],
                           'conduct_audio_cb_bool': _toml['conduct_audio_recording'], 'conduct_tracking_calibration_cb_bool': _toml['conduct_tracking_calibration'],
                           'disable_ethernet_cb_bool': _toml['disable_ethernet'], 'monitor_recording_cb_bool': _toml['video']['general']['monitor_recording'],
                           'monitor_specific_camera_cb_bool': _toml['video']['general']['monitor_specific_camera'], 'delete_post_copy_cb_bool': _toml['video']['general']['delete_post_copy'],
                           'vacant_arena_cb_bool': _toml['video']['metadata']['vacant_arena'], 'ambient_light_cb_bool': _toml['video']['metadata']['ambient_light'], 'record_brain_cb_bool': _toml['video']['metadata']['record_brain'],
                           'usv_playback_cb_bool': _toml['video']['metadata']['usv_playback'], 'chemogenetics_cb_bool': _toml['video']['metadata']['chemogenetics'], 'optogenetics_cb_bool': _toml['video']['metadata']['optogenetics'],
                           'brain_lesion_cb_bool': _toml['video']['metadata']['brain_lesion'], 'devocalization_cb_bool': _toml['video']['metadata']['devocalization'], 'female_urine_cb_bool': _toml['video']['metadata']['female_urine'],
                           'female_bedding_cb_bool': _toml['video']['metadata']['female_bedding'], 'recording_codec': _toml['video']['general']['recording_codec'],
                           '21372315_et': _toml['video']['cameras_config']['21372315']['exposure_time'], '21372315_dg': _toml['video']['cameras_config']['21372315']['gain'],
                           '21372316_et': _toml['video']['cameras_config']['21372316']['exposure_time'], '21372316_dg': _toml['video']['cameras_config']['21372316']['gain'],
                           '21369048_et': _toml['video']['cameras_config']['21369048']['exposure_time'], '21369048_dg': _toml['video']['cameras_config']['21369048']['gain'],
                           '22085397_et': _toml['video']['cameras_config']['22085397']['exposure_time'], '22085397_dg': _toml['video']['cameras_config']['22085397']['gain'],
                           '21241563_et': _toml['video']['cameras_config']['21241563']['exposure_time'], '21241563_dg': _toml['video']['cameras_config']['21241563']['gain'],
                           'inference_root_dir_btn_clicked_flag': False, 'centroid_model_btn_clicked_flag': False,  'centered_instance_btn_btn_clicked_flag': False,
                           'calibration_file_loc_btn_clicked_flag': False, 'das_model_dir_btn_clicked_flag': False,
                           'recorder_dir_btn_clicked_flag': False, 'avisoft_base_dir_btn_clicked_flag': False, 'coolterm_base_dir_btn_clicked_flag': False,
                           'npx_file_type': 'ap', 'device_receiving_input': 'm', 'save_transformed_data': 'animal',
                           'conduct_video_concatenation_cb_bool': False, 'conduct_video_fps_change_cb_bool': False,
                           'conduct_multichannel_conversion_cb_bool': False, 'crop_wav_cam_cb_bool': False, 'conc_audio_cb_bool': False, 'filter_audio_cb_bool': False,
                           'conduct_sync_cb_bool': False, 'conduct_hpss_cb_bool': False, 'conduct_ephys_file_chaining_cb_bool': False,
                           'conduct_nv_sync_cb_bool': False, 'split_cluster_spikes_cb_bool': False, 'anipose_calibration_cb_bool': False,
                           'sleap_file_conversion_cb_bool': False, 'anipose_triangulation_cb_bool': False, 'translate_rotate_metric_cb_bool': False,
                           'sleap_cluster_cb_bool': False, 'das_inference_cb_bool': False, 'das_summary_cb_bool': False, 'delete_con_file_cb_bool': True,
                           'board_provided_cb_bool': False, 'triangulate_arena_points_cb_bool': False,
                           'display_progress_cb_bool': True, 'ransac_cb_bool': False, 'delete_original_h5_cb_bool': True,
                           'compute_behavioral_features_cb_bool': False, 'plot_behavioral_tuning_cb_bool': False, 'make_behavioral_video_cb_bool': False,
                           'visualization_type_cb_bool': False, 'plot_theme': visualizations_input_dict['make_behavioral_videos']['plot_theme'],
                           'fig_format': visualizations_input_dict['make_behavioral_videos']['general_figure_specs']['fig_format'], 'view_angle': visualizations_input_dict['make_behavioral_videos']['view_angle'],
                           'rotate_side_view_bool': False, 'history_cb_bool': False, 'speaker_cb_bool': False, 'spectrogram_cb_bool': False,
                           'spectrogram_ch': visualizations_input_dict['make_behavioral_videos']['spectrogram_ch'], 'raster_plot_cb_bool': False, 'spike_sound_cb_bool': False,
                           'beh_features_cb_bool': False, 'calculate_neuronal_tuning_curves_cb_bool': False, 'create_usv_playback_wav_cb_bool': False, 'frequency_shift_audio_segment_cb_bool': False,
                           'fs_audio_dir': analyses_input_dict['frequency_shift_audio_segment']['fs_audio_dir'], 'fs_device_id': analyses_input_dict['frequency_shift_audio_segment']['fs_device_id'],
                           'fs_channel_id': analyses_input_dict['frequency_shift_audio_segment']['fs_channel_id'], 'volume_adjust_audio_segment_cb_bool': True,
                           'visualizations_pc_choice': visualizations_input_dict['send_email']['visualizations_pc_choice'], 'analyses_pc_choice': analyses_input_dict['send_email']['analyses_pc_choice'],
                           'processing_pc_choice': processing_input_dict['send_email']['Messenger']['processing_pc_choice'], 'encoding_preset': processing_input_dict['modify_files']['Operator']['rectify_video_fps']['encoding_preset']}

    usv_playpen_window = USVPlaypenWindow(**initial_values_dict)

    splash.finish(usv_playpen_window)

    usv_playpen_window.show()

    sys.exit(usv_playpen_app.exec())


if __name__ == "__main__":
    main()
