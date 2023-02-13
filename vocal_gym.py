"""
@author: bartulem
GUI to run behavioral experiments.
"""

import ast
import configparser
import datetime
import os
from pathlib import Path
import sys
import threading
import time

from behavioral_experiments import ExperimentController, _loop_time

from PyQt6.QtCore import (
    Qt
)
from PyQt6.QtGui import (
    QGuiApplication,
    QIcon,
    QPainter,
    QPixmap
)
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QGridLayout,
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
import toml

basedir = os.path.dirname(__file__)
background_img = f"{basedir}{os.sep}img{os.sep}background_img.png"
background_img_2 = f"{basedir}{os.sep}img{os.sep}background_img_2.png"
background_img_3 = f"{basedir}{os.sep}img{os.sep}background_img_3.png"
lab_icon = f"{basedir}{os.sep}img{os.sep}lab.png"
splash_icon = f"{basedir}{os.sep}img{os.sep}uncle_stefan-min.png"
process_icon = f"{basedir}{os.sep}img{os.sep}process.png"
record_icon = f"{basedir}{os.sep}img{os.sep}record.png"
previous_icon = f"{basedir}{os.sep}img{os.sep}previous.png"
next_icon = f"{basedir}{os.sep}img{os.sep}next.png"
main_icon = f"{basedir}{os.sep}img{os.sep}main.png"
calibrate_icon = f"{basedir}{os.sep}img{os.sep}calibrate.png"


class Main(QWidget):
    def __init__(self, parent=None):
        super(Main, self).__init__(parent)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawPixmap(self.rect(), QPixmap(f"{background_img}"))
        QWidget.paintEvent(self, event)


class Record(QWidget):
    def __init__(self, parent=Main):
        super(Record, self).__init__(parent)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawPixmap(self.rect(), QPixmap(f"{background_img}"))
        QWidget.paintEvent(self, event)


class AudioSettings(QWidget):
    def __init__(self, parent=Main):
        super(AudioSettings, self).__init__(parent)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawPixmap(self.rect(), QPixmap(f"{background_img_2}"))
        QWidget.paintEvent(self, event)


class VideoSettings(QWidget):
    def __init__(self, parent=Main):
        super(VideoSettings, self).__init__(parent)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawPixmap(self.rect(), QPixmap(f"{background_img_3}"))
        QWidget.paintEvent(self, event)


class ConductRecording(QWidget):
    def __init__(self, parent=Main):
        super(ConductRecording, self).__init__(parent)


class ProcessSettings(QWidget):
    def __init__(self, parent=Main):
        super(ProcessSettings, self).__init__(parent)


class HyperlinkLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__()
        self.setOpenExternalLinks(True)
        self.setParent(parent)


class VocalGymWindow(QMainWindow):
    """Main window (GUI or view)."""

    def __init__(self):
        super().__init__()
        self.processing_dir_edit = None
        self.processing_title_label = None
        self.ProcessSettings = None
        self.experiment_time_sec = 0
        self.run_exp = None
        self.email_recipients = None
        self.experiment_progress_bar = None
        self.txt_edit = None
        self.gain_21241563_label = None
        self.gain_21241563 = None
        self.exposure_time_21241563_label = None
        self.exposure_time_21241563 = None
        self.gain_22085397_label = None
        self.gain_22085397 = None
        self.exposure_time_22085397_label = None
        self.exposure_time_22085397 = None
        self.gain_21369048_label = None
        self.gain_21369048 = None
        self.exposure_time_21369048_label = None
        self.exposure_time_21369048 = None
        self.gain_21372315_label = None
        self.gain_21372315 = None
        self.exposure_time_21372315_label = None
        self.exposure_time_21372315 = None
        self.gain_21372316_label = None
        self.gain_21372316 = None
        self.exposure_time_21372316_label = None
        self.exposure_time_21372316 = None
        self.pcs_label = None
        self.other = None
        self.housing_m2 = None
        self.dob_m2 = None
        self.sex_m2 = None
        self.genotype_m2 = None
        self.mouse_ID_m2 = None
        self.cage_ID_m2 = None
        self.housing_m1 = None
        self.dob_m1 = None
        self.sex_m1 = None
        self.genotype_m1 = None
        self.mouse_ID_m1 = None
        self.cage_ID_m1 = None
        self.mice_num = None
        self.experimenter = None
        self.vm_label = None
        self.fr_label = None
        self.cameras_frame_rate = None
        self.gvs_label = None
        self.parameters_label = None
        self.label = None
        self.title_label = None
        self.power_plan = None
        self.link_sox = None
        self.link_coolterm = None
        self.link_arduino = None
        self.link_ffmpeg = None
        self.link_chrome = None
        self.link_avisoft = None
        self.monitor_recording_cb_bool = True
        self.monitor_specific_camera_cb_bool = False
        self.delete_post_copy_cb_bool = True
        self.ConductRecording = None
        self.delete_post_copy_cb = None
        self.monitor_specific_camera_cb = None
        self.monitor_recording_cb = None
        self.recording_codec = None
        self.specific_camera_serial = None
        self.expected_camera_num = None
        self.browser = None
        self.name = None
        self.id = None
        self.typech = None
        self.deviceid = None
        self.channel = None
        self.gain = None
        self.fullscalespl = None
        self.triggertype = None
        self.toggle = None
        self.invert = None
        self.ditc = None
        self.ditctime = None
        self.whistletracking = None
        self.wtbcf = None
        self.wtmaxchange = None
        self.wtmaxchange2_ = None
        self.wtminchange = None
        self.wtminchange2_ = None
        self.wtoutside = None
        self.hilowratioenable = None
        self.hilowratio = None
        self.hilowratiofc = None
        self.wtslope = None
        self.wtlevel = None
        self.wtmindurtotal = None
        self.wtmindur = None
        self.wtmindur2_ = None
        self.wtholdtime = None
        self.wtmonotonic = None
        self.wtbmaxdur = None
        self.rejectwind = None
        self.rwentropy = None
        self.rwfpegel = None
        self.rwdutycycle = None
        self.rwtimeconstant = None
        self.rwholdtime = None
        self.fpegel = None
        self.energy = None
        self.frange = None
        self.entropyb = None
        self.entropy = None
        self.increment = None
        self.fu = None
        self.fo = None
        self.pretrigger = None
        self.mint = None
        self.minst = None
        self.fhold = None
        self.logfileno = None
        self.groupno = None
        self.callno = None
        self.timeconstant = None
        self.timeexpansion = None
        self.startstop = None
        self.sayf = None
        self.over = None
        self.delay = None
        self.center = None
        self.bandwidth = None
        self.fd = None
        self.decimation = None
        self.device = None
        self.mode = None
        self.outfovertaps = None
        self.outfoverabtast = None
        self.outformat = None
        self.outfabtast = None
        self.outdeviceid = None
        self.outtype = None
        self.usghflags = None
        self.diff = None
        self.format = None
        self.type = None
        self.nbrwavehdr = None
        self.devbuffer = None
        self.ntaps = None
        self.filtercutoff = None
        self.filter = None
        self.fabtast = None
        self.y2 = None
        self.x2 = None
        self.y1 = None
        self.x1 = None
        self.fftlength = None
        self.usvmonitoringflags = None
        self.dispspectrogramcontrast = None
        self.disprangespectrogram = None
        self.disprangeamplitude = None
        self.disprangewaveform = None
        self.total = None
        self.dcolumns = None
        self.display = None
        self.used_mics = None
        self.total_mic_number = None
        self.total_device_num = None
        self.first_col_labels_2 = None
        self.first_col_labels_1 = None
        self.second_col_labels = None
        self.VideoSettings = None
        self.second_col_variable_names = None
        self.settings_dict = {'general': {'config_settings_directory': '',
                                          'avisoft_recorder_exe': '',
                                          'avisoft_basedirectory': '',
                                          'coolterm_basedirectory': ''}, 'audio': {}, 'video': {}}
        self.calibration_session_duration = None
        self.conduct_tracking_calibration_cb = None
        self.conduct_audio_cb = None
        self.conduct_audio_cb_bool = True
        self.conduct_tracking_calibration_cb_bool = False
        self.modify_audio_config = False
        self.video_session_duration = None
        self.recording_files_destination_windows = None
        self.recording_files_destination_linux = None
        self.coolterm_base_edit = None
        self.avisoft_base_edit = None
        self.recorder_settings_edit = None
        self.dir_settings_edit = None
        self.textEdit = None
        self.generalLayout = None
        self.AudioSettings = None
        self.Record = None
        self.Main = None
        self.setWindowIcon(QIcon(lab_icon))
        self.main_window()

    def main_window(self):
        self.Main = Main(self)
        self.setCentralWidget(self.Main)
        self._location_on_the_screen()
        self.generalLayout = QGridLayout()
        self.Main.setLayout(self.generalLayout)
        self.setWindowFlag(Qt.WindowType.WindowMaximizeButtonHint, False)
        self.setWindowTitle("PNI Vocal Gymnasium")

        link_template = "<a href=\'{0}\'>{1}</a>"

        self.link_avisoft = HyperlinkLabel()
        self.link_avisoft = link_template.format('https://www.avisoft.com/downloads/', 'Avisoft-Recorder USGH')
        self.link_chrome = HyperlinkLabel()
        self.link_chrome = link_template.format('https://www.google.com/chrome/', 'Chrome')
        self.link_ffmpeg = HyperlinkLabel()
        self.link_ffmpeg = link_template.format('https://www.gyan.dev/ffmpeg/builds/', 'FFMPEG')
        self.link_arduino = HyperlinkLabel()
        self.link_arduino = link_template.format('https://www.arduino.cc/en/software', 'Arduino Software (IDE)')
        self.link_coolterm = HyperlinkLabel()
        self.link_coolterm = link_template.format('https://coolterm.en.lo4d.com/windows', 'CoolTerm')
        self.link_sox = HyperlinkLabel()
        self.link_sox = link_template.format('https://sourceforge.net/projects/sox/', 'Sox')
        self.link_pip = HyperlinkLabel()
        self.link_pip = link_template.format('https://www.geeksforgeeks.org/how-to-install-pip-on-windows/', 'pip')
        self.link_git = HyperlinkLabel()
        self.link_git = link_template.format('https://git-scm.com/download/win', 'git')
        self.link_git_path = HyperlinkLabel()
        self.link_git_path = link_template.format('https://linuxhint.com/add-git-to-path-windows/', 'PATH')
        self.power_plan = HyperlinkLabel()
        self.power_plan = link_template.format('https://www.howtogeek.com/240840/should-you-use-the-balanced-power-saver-or-high-performance-power-plan-on-windows/', 'power plan')
        self.label = QLabel(f"<br>Thank you for using the PNI Vocal Gymnasium."
                            f"<br><br>To ensure quality recordings/data, make sure you install the following (prior to further usage): "
                            f"<br><br>(1) " + self.link_avisoft + " ≥4.4.14"
                            f"<br>(2) " + self.link_chrome + " (or other) web browser"
                            f"<br>(3) " + self.link_ffmpeg + " (and add it to PATH)"
                            f"<br>(4) " + self.link_arduino +
                            f"<br>(5) " + self.link_coolterm +
                            f"<br>(6) " + self.link_sox + " (and add it to PATH)"
                            f"<br>(7) " + self.link_pip + " (and add it to PATH)"
                            f"<br>(8) " + self.link_git + " (and add it to " + self.link_git_path + ")"
                            f"<br><br> Change the Windows " + self.power_plan + " to 'High performance'."
                            f"<br><br> Contact the author for Arduino/Coolterm instructions and necessary configuration files."
                            f"<br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br>"
                            f"<br> © github/bartulem {datetime.date.today().strftime('%Y')}")
        self.label.setOpenExternalLinks(True)
        self.generalLayout.addWidget(self.label, 0, 0, alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self._create_buttons_main()

    def record_one(self):
        self.Record = Record(self)
        self.setWindowTitle("PNI Vocal Gymnasium (Record > Select directories and set basic parameters)")
        self.setCentralWidget(self.Record)
        self.generalLayout = QGridLayout()
        self.Record.setLayout(self.generalLayout)

        self.title_label = QLabel("Please select appropriate directories (w/ config files or executables in them)")
        self.title_label.setStyleSheet("QLabel { font-weight: bold;}")
        self.generalLayout.addWidget(self.title_label, 0, 0, 0, 3, alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        settings_dir_btn = QPushButton('Browse')
        settings_dir_btn.setStyleSheet('QPushButton { min-width: 64px; min-height: 12px; max-width: 64px; max-height: 13px;}')
        settings_dir_btn.clicked.connect(self._open_settings_dialog)
        self.dir_settings_edit = QLineEdit()

        self.generalLayout.addWidget(QLabel('settings file (*.toml) directory:'), 3, 0, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.dir_settings_edit, 3, 1, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(settings_dir_btn, 3, 2, alignment=Qt.AlignmentFlag.AlignTop)

        recorder_dir_btn = QPushButton('Browse')
        recorder_dir_btn.setStyleSheet('QPushButton { min-width: 64px; min-height: 12px; max-width: 64px; max-height: 13px;}')
        recorder_dir_btn.clicked.connect(self._open_recorder_dialog)
        self.recorder_settings_edit = QLineEdit()

        self.generalLayout.addWidget(QLabel('Avisoft Recorder (usgh.exe) directory:'), 4, 0, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.recorder_settings_edit, 4, 1, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(recorder_dir_btn, 4, 2, alignment=Qt.AlignmentFlag.AlignTop)

        avisoft_base_dir_btn = QPushButton('Browse')
        avisoft_base_dir_btn.setStyleSheet('QPushButton { min-width: 64px; min-height: 12px; max-width: 64px; max-height: 13px;}')
        avisoft_base_dir_btn.clicked.connect(self._open_avisoft_dialog)
        self.avisoft_base_edit = QLineEdit()

        self.generalLayout.addWidget(QLabel('Avisoft Bioacoustics base directory:'), 5, 0, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.avisoft_base_edit, 5, 1, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(avisoft_base_dir_btn, 5, 2, alignment=Qt.AlignmentFlag.AlignTop)

        coolterm_base_dir_btn = QPushButton('Browse')
        coolterm_base_dir_btn.setStyleSheet('QPushButton { min-width: 64px; min-height: 12px; max-width: 64px; max-height: 13px;}')
        coolterm_base_dir_btn.clicked.connect(self._open_coolterm_dialog)
        self.coolterm_base_edit = QLineEdit()

        self.generalLayout.addWidget(QLabel('CoolTerm base directory:'), 6, 0, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.coolterm_base_edit, 6, 1, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(coolterm_base_dir_btn, 6, 2, alignment=Qt.AlignmentFlag.AlignTop)

        # recording files destination directories (across OS)
        self.recording_files_destination_linux = QLineEdit("/home/labadmin/falkner/Bartul/Data,/home/labadmin/murthy/Bartul/Data")
        self.generalLayout.addWidget(QLabel('recording file destinations (Linux):'), 9, 0, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.recording_files_destination_linux, 9, 1, 9, 2, alignment=Qt.AlignmentFlag.AlignTop)

        self.recording_files_destination_windows = QLineEdit("F:\\Bartul\\Data,M:\\Bartul\\Data")
        self.generalLayout.addWidget(QLabel('recording file destinations (Windows):'), 10, 0, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.recording_files_destination_windows, 10, 1, 10, 2, alignment=Qt.AlignmentFlag.AlignTop)

        # set main recording parameters
        self.parameters_label = QLabel("Please set main recording parameters")
        self.parameters_label.setStyleSheet("QLabel { font-weight: bold;}")
        self.generalLayout.addWidget(self.parameters_label, 13, 0, 13, 3, alignment=Qt.AlignmentFlag.AlignTop)
        self.video_session_duration = QLineEdit("20")
        self.generalLayout.addWidget(QLabel('video session duration (min):'), 16, 0, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.video_session_duration, 16, 1, 16, 2, alignment=Qt.AlignmentFlag.AlignTop)

        self.conduct_audio_cb = QComboBox()
        self.conduct_audio_cb.addItems(['Yes', 'No'])
        self.conduct_audio_cb.activated.connect(self._conduct_audio_combo_box_activated)
        self.generalLayout.addWidget(QLabel('conduct audio recording:'), 17, 0, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.conduct_audio_cb, 17, 1, alignment=Qt.AlignmentFlag.AlignTop)

        # tracking calibration settings
        self.conduct_tracking_calibration_cb = QComboBox()
        self.conduct_tracking_calibration_cb.addItems(['No', 'Yes'])
        self.conduct_tracking_calibration_cb.activated.connect(self._conduct_tracking_calibration_combo_box_activated)
        self.generalLayout.addWidget(QLabel('conduct video calibration:'), 18, 0, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.conduct_tracking_calibration_cb, 18, 1, alignment=Qt.AlignmentFlag.AlignTop)

        self.calibration_session_duration = QLineEdit("4")
        self.generalLayout.addWidget(QLabel('calibration session duration (min):'), 19, 0, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.calibration_session_duration, 19, 1, 19, 2, alignment=Qt.AlignmentFlag.AlignTop)

        self.email_recipients = QLineEdit("bmimica@princeton.edu")
        self.generalLayout.addWidget(QLabel('notify the following addresses about PC usage:'), 20, 0, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.email_recipients, 20, 1, 20, 2, alignment=Qt.AlignmentFlag.AlignTop)

        self._create_buttons_record(seq=0)

    def record_two(self):
        self.AudioSettings = AudioSettings(self)
        self.setWindowTitle("PNI Vocal Gymnasium (Record > Audio Settings)")
        self.setCentralWidget(self.AudioSettings)
        self.generalLayout = QGridLayout()
        self.generalLayout.setSpacing(1)
        self.AudioSettings.setLayout(self.generalLayout)

        self.name = QLineEdit("999")
        self.id = QLineEdit("999")
        self.typech = QLineEdit("13")
        self.deviceid = QLineEdit("999")
        self.channel = QLineEdit("999")
        self.gain = QLineEdit("1")
        self.fullscalespl = QLineEdit("0.0")
        self.triggertype = QLineEdit("41")
        self.toggle = QLineEdit("0")
        self.invert = QLineEdit("0")
        self.ditc = QLineEdit("0")
        self.ditctime = QLineEdit("00:00:00:00")
        self.whistletracking = QLineEdit("0")
        self.wtbcf = QLineEdit("0")
        self.wtmaxchange = QLineEdit("3")
        self.wtmaxchange2_ = QLineEdit("0")
        self.wtminchange = QLineEdit("-10")
        self.wtminchange2_ = QLineEdit("0")
        self.wtoutside = QLineEdit("0")
        self.hilowratioenable = QLineEdit("0")
        self.hilowratio = QLineEdit("2.0")
        self.hilowratiofc = QLineEdit("15000.0")
        self.wtslope = QLineEdit("0")
        self.wtlevel = QLineEdit("0")
        self.wtmindurtotal = QLineEdit("0.0")
        self.wtmindur = QLineEdit("0.005")
        self.wtmindur2_ = QLineEdit("0.0")
        self.wtholdtime = QLineEdit("0.02")
        self.wtmonotonic = QLineEdit("1")
        self.wtbmaxdur = QLineEdit("0")
        self.rejectwind = QLineEdit("0")
        self.rwentropy = QLineEdit("0.5")
        self.rwfpegel = QLineEdit("2.5")
        self.rwdutycycle = QLineEdit("0.2")
        self.rwtimeconstant = QLineEdit("2.0")
        self.rwholdtime = QLineEdit("10.0")
        self.fpegel = QLineEdit("5.0")
        self.energy = QLineEdit("0")
        self.frange = QLineEdit("1")
        self.entropyb = QLineEdit("0")
        self.entropy = QLineEdit("0.35")
        self.increment = QLineEdit("1")
        self.fu = QLineEdit("0.0")
        self.fo = QLineEdit("250000.0")
        self.pretrigger = QLineEdit("0.5")
        self.mint = QLineEdit("0.0")
        self.minst = QLineEdit("0.0")
        self.fhold = QLineEdit("0.5")
        self.logfileno = QLineEdit("0")
        self.groupno = QLineEdit("0")
        self.callno = QLineEdit("0")
        self.timeconstant = QLineEdit("0.003")
        self.timeexpansion = QLineEdit("0")
        self.startstop = QLineEdit("0")
        self.sayf = QLineEdit("2")
        self.over = QLineEdit("0")
        self.delay = QLineEdit("0.0")
        self.center = QLineEdit("40000")
        self.bandwidth = QLineEdit("5")
        self.fd = QLineEdit("5")
        self.decimation = QLineEdit("-1")
        self.device = QLineEdit("0")
        self.mode = QLineEdit("0")
        self.outfovertaps = QLineEdit("32")
        self.outfoverabtast = QLineEdit("2000000")
        self.outformat = QLineEdit("2")
        self.outfabtast = QLineEdit("-22050")
        self.outdeviceid = QLineEdit("0")
        self.outtype = QLineEdit("7")
        self.usghflags = QLineEdit("1574")
        self.diff = QLineEdit("0")
        self.format = QLineEdit("1")
        self.type = QLineEdit("0")
        self.nbrwavehdr = QLineEdit("32")
        self.devbuffer = QLineEdit("0.032")
        self.ntaps = QLineEdit("32")
        self.filtercutoff = QLineEdit("15.0")
        self.filter = QLineEdit("0")
        self.fabtast = QLineEdit("250000")
        self.y2 = QLineEdit("1315")
        self.x2 = QLineEdit("2563")
        self.y1 = QLineEdit("3")
        self.x1 = QLineEdit("1378")
        self.fftlength = QLineEdit("256")
        self.usvmonitoringflags = QLineEdit("9136")
        self.dispspectrogramcontrast = QLineEdit("0.0")
        self.disprangespectrogram = QLineEdit("250.0")
        self.disprangeamplitude = QLineEdit("100.0")
        self.disprangewaveform = QLineEdit("100.0")
        self.total = QLineEdit("1")
        self.dcolumns = QLineEdit("3")
        self.display = QLineEdit("2")
        self.used_mics = QLineEdit("0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23")
        self.total_mic_number = QLineEdit("24")
        self.total_device_num = QLineEdit("2")

        self.first_col_labels_1 = ["total_device_num:", "total_mic_number:", "used_mics:",
                                   "display:", "dcolumns:", "total:",
                                   "disprangewaveform:", "disprangeamplitude:", "disprangespectrogram:",
                                   "dispspectrogramcontrast:", "usvmonitoringflags:", "fftlength:",
                                   "x1:", "y1:", "x2:", "y2:"]

        for fc_idx_1, fc_item_1 in enumerate(self.first_col_labels_1):
            getattr(self, fc_item_1[:-1]).setFixedWidth(150)
            self.generalLayout.addWidget(QLabel(self.first_col_labels_1[fc_idx_1]), 5 + fc_idx_1, 0, alignment=Qt.AlignmentFlag.AlignTop)
            self.generalLayout.addWidget(getattr(self, fc_item_1[:-1]), 5 + fc_idx_1, 1, 5 + fc_idx_1, 2, alignment=Qt.AlignmentFlag.AlignTop)

        self.first_col_labels_2 = ["fabtast:", "filter:", "filtercutoff:",
                                   "ntaps:", "devbuffer:", "nbrwavehdr:",
                                   "type:", "format:", "diff:",
                                   "usghflags:", "outtype:", "outdeviceid:",
                                   "outfabtast:", "outformat:", "outfoverabtast:",
                                   "outfovertaps:", "mode:", "device:",
                                   "decimation:",
                                   "fd:", "bandwidth:", "center:",
                                   "delay:", "over:", "sayf:",
                                   "startstop:", "timeexpansion:", "timeconstant:",
                                   "callno:", "groupno:", "logfileno:"]

        for fc_idx_2, fc_item_2 in enumerate(self.first_col_labels_2):
            getattr(self, fc_item_2[:-1]).setFixedWidth(150)
            self.generalLayout.addWidget(QLabel(self.first_col_labels_2[fc_idx_2]), 22 + fc_idx_2, 0, alignment=Qt.AlignmentFlag.AlignTop)
            self.generalLayout.addWidget(getattr(self, fc_item_2[:-1]), 22 + fc_idx_2, 1, 22 + fc_idx_2, 2, alignment=Qt.AlignmentFlag.AlignTop)

        self.second_col_labels = ["name:", "id:", "typech:",
                                  "deviceid:", "channel:", "gain:",
                                  "fullscalespl:", "triggertype:", "toggle:",
                                  "invert:", "ditc:", "ditctime:",
                                  "whistletracking:", "wtbcf:", "wtmaxchange:",
                                  "wtmaxchange2_:", "wtminchange:", "wtminchange2_:",
                                  "wtoutside:", "hilowratioenable:", "hilowratio:",
                                  "hilowratiofc:", "wtslope:", "wtlevel:",
                                  "wtmindurtotal:", "wtmindur:", "wtmindur2_:",
                                  "wtholdtime:", "wtmonotonic:", "wtbmaxdur:",
                                  "rejectwind:", "rwentropy:", "rwfpegel:",
                                  "rwdutycycle:", "rwtimeconstant:", "rwholdtime:",
                                  "fpegel:", "energy:", "frange:",
                                  "entropyb:", "entropy:", "increment:",
                                  "fu:", "fo:", "pretrigger:",
                                  "mint:", "minst:", "fhold:"]

        for fc_idx_3, fc_item_3 in enumerate(self.second_col_labels):
            getattr(self, fc_item_3[:-1]).setFixedWidth(150)
            self.generalLayout.addWidget(QLabel(self.second_col_labels[fc_idx_3]), 5 + fc_idx_3, 55, alignment=Qt.AlignmentFlag.AlignTop)
            self.generalLayout.addWidget(getattr(self, fc_item_3[:-1]), 5 + fc_idx_3, 57, alignment=Qt.AlignmentFlag.AlignTop)

        self._create_buttons_record(seq=1)

    def record_three(self):
        self.VideoSettings = VideoSettings(self)
        self.setWindowTitle("PNI Vocal Gymnasium (Record > Video Settings)")
        self.setCentralWidget(self.VideoSettings)
        self.generalLayout = QGridLayout()
        self.VideoSettings.setLayout(self.generalLayout)

        self.gvs_label = QLabel("General video settings")
        self.gvs_label.setStyleSheet("QLabel { font-weight: bold;}")
        self.generalLayout.addWidget(self.gvs_label,
                                     0, 0, alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        self.browser = QLineEdit("chrome")
        self.generalLayout.addWidget(QLabel('browser:'), 2, 0, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.browser, 2, 1, alignment=Qt.AlignmentFlag.AlignTop)

        self.expected_camera_num = QLineEdit("5")
        self.generalLayout.addWidget(QLabel('expected_camera_num:'), 3, 0, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.expected_camera_num, 3, 1, alignment=Qt.AlignmentFlag.AlignTop)

        self.specific_camera_serial = QLineEdit("21372315")
        self.generalLayout.addWidget(QLabel('specific_camera_serial:'), 4, 0, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.specific_camera_serial, 4, 1, alignment=Qt.AlignmentFlag.AlignTop)

        self.recording_codec = QLineEdit("lq")
        self.generalLayout.addWidget(QLabel('recording_codec:'), 5, 0, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.recording_codec, 5, 1, alignment=Qt.AlignmentFlag.AlignTop)

        self.monitor_recording_cb = QComboBox()
        self.monitor_recording_cb.addItems(['Yes', 'No'])
        self.monitor_recording_cb.activated.connect(self._monitor_recording_combo_box_activated)
        self.generalLayout.addWidget(QLabel('monitor_recording:'), 6, 0, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.monitor_recording_cb, 6, 1, alignment=Qt.AlignmentFlag.AlignTop)

        self.monitor_specific_camera_cb = QComboBox()
        self.monitor_specific_camera_cb.addItems(['No', 'Yes'])
        self.monitor_specific_camera_cb.activated.connect(self._monitor_specific_camera_combo_box_activated)
        self.generalLayout.addWidget(QLabel('monitor_specific_camera:'), 7, 0, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.monitor_specific_camera_cb, 7, 1, alignment=Qt.AlignmentFlag.AlignTop)

        self.delete_post_copy_cb = QComboBox()
        self.delete_post_copy_cb.addItems(['Yes', 'No'])
        self.delete_post_copy_cb.activated.connect(self._delete_post_copy_combo_box_activated)
        self.generalLayout.addWidget(QLabel('delete_post_copy:'), 8, 0, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.delete_post_copy_cb, 8, 1, alignment=Qt.AlignmentFlag.AlignTop)

        self.cameras_frame_rate = QSlider(Qt.Orientation.Horizontal)
        self.cameras_frame_rate.setRange(10, 150)
        self.cameras_frame_rate.setValue(150)
        self.cameras_frame_rate.valueChanged.connect(self._update_fr_label)

        self.fr_label = QLabel('camera freq (150 fps):')
        self.fr_label.setFixedWidth(150)

        self.generalLayout.addWidget(self.fr_label, 9, 0, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.cameras_frame_rate, 9, 1, alignment=Qt.AlignmentFlag.AlignVCenter)

        self.vm_label = QLabel("Video metadata")
        self.vm_label.setStyleSheet("QLabel { font-weight: bold;}")
        self.generalLayout.addWidget(self.vm_label,
                                     11, 0, alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        self.experimenter = QLineEdit("bartulem")
        self.generalLayout.addWidget(QLabel('experimenter:'), 12, 0, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.experimenter, 12, 1, alignment=Qt.AlignmentFlag.AlignTop)

        self.mice_num = QLineEdit("2")
        self.generalLayout.addWidget(QLabel('mice_num:'), 13, 0, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.mice_num, 13, 1, alignment=Qt.AlignmentFlag.AlignTop)

        self.cage_ID_m1 = QLineEdit("")
        self.generalLayout.addWidget(QLabel('cage_ID_m1:'), 14, 0, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.cage_ID_m1, 14, 1, alignment=Qt.AlignmentFlag.AlignTop)

        self.mouse_ID_m1 = QLineEdit("")
        self.generalLayout.addWidget(QLabel('mouse_ID_m1:'), 15, 0, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.mouse_ID_m1, 15, 1, alignment=Qt.AlignmentFlag.AlignTop)

        self.genotype_m1 = QLineEdit("CD1-WT")
        self.generalLayout.addWidget(QLabel('genotype_m1:'), 16, 0, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.genotype_m1, 16, 1, alignment=Qt.AlignmentFlag.AlignTop)

        self.sex_m1 = QLineEdit("")
        self.generalLayout.addWidget(QLabel('sex_m1:'), 17, 0, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.sex_m1, 17, 1, alignment=Qt.AlignmentFlag.AlignTop)

        self.dob_m1 = QLineEdit("")
        self.generalLayout.addWidget(QLabel('DOB_m1:'), 18, 0, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.dob_m1, 18, 1, alignment=Qt.AlignmentFlag.AlignTop)

        self.housing_m1 = QLineEdit("group")
        self.generalLayout.addWidget(QLabel('housing_m1:'), 19, 0, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.housing_m1, 19, 1, alignment=Qt.AlignmentFlag.AlignTop)

        self.cage_ID_m2 = QLineEdit("")
        self.generalLayout.addWidget(QLabel('cage_ID_m2:'), 20, 0, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.cage_ID_m2, 20, 1, alignment=Qt.AlignmentFlag.AlignTop)

        self.mouse_ID_m2 = QLineEdit("")
        self.generalLayout.addWidget(QLabel('mouse_ID_m2:'), 21, 0, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.mouse_ID_m2, 21, 1, alignment=Qt.AlignmentFlag.AlignTop)

        self.genotype_m2 = QLineEdit("CD1-WT")
        self.generalLayout.addWidget(QLabel('genotype_m2:'), 22, 0, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.genotype_m2, 22, 1, alignment=Qt.AlignmentFlag.AlignTop)

        self.sex_m2 = QLineEdit("")
        self.generalLayout.addWidget(QLabel('sex_m2:'), 23, 0, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.sex_m2, 23, 1, alignment=Qt.AlignmentFlag.AlignTop)

        self.dob_m2 = QLineEdit("")
        self.generalLayout.addWidget(QLabel('dob_m2:'), 24, 0, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.dob_m2, 24, 1, alignment=Qt.AlignmentFlag.AlignTop)

        self.housing_m2 = QLineEdit("group")
        self.generalLayout.addWidget(QLabel('housing_m2:'), 25, 0, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.housing_m2, 25, 1, alignment=Qt.AlignmentFlag.AlignTop)

        self.other = QLineEdit("")
        self.generalLayout.addWidget(QLabel('other information:'), 26, 0, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.other, 26, 1, alignment=Qt.AlignmentFlag.AlignTop)

        self.generalLayout.addWidget(QLabel("         "), 5, 6, alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.pcs_label = QLabel("Particular camera settings")
        self.pcs_label.setStyleSheet("QLabel { font-weight: bold;}")
        self.generalLayout.addWidget(self.pcs_label,
                                     0, 7, alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        self.generalLayout.addWidget(QLabel("Camera 21372316 (orange) settings"),
                                     2, 7, 2, 9, alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        self.exposure_time_21372316 = QSlider(Qt.Orientation.Horizontal)
        self.exposure_time_21372316.setRange(500, 30000)
        self.exposure_time_21372316.setValue(2500)
        self.exposure_time_21372316.valueChanged.connect(self._update_exposure_time_21372316_label)
        self.exposure_time_21372316_label = QLabel('exp time (2500 μs):')
        self.exposure_time_21372316_label.setFixedWidth(150)
        self.exposure_time_21372316_label.setAlignment(Qt.AlignmentFlag.AlignLeft |
                                                       Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.exposure_time_21372316_label, 3, 7)
        self.generalLayout.addWidget(self.exposure_time_21372316, 3, 8, alignment=Qt.AlignmentFlag.AlignVCenter)

        self.gain_21372316 = QSlider(Qt.Orientation.Horizontal)
        self.gain_21372316.setRange(0, 15)
        self.gain_21372316.setValue(0)
        self.gain_21372316.valueChanged.connect(self._update_gain_21372316_label)
        self.gain_21372316_label = QLabel('digital gain (0 dB):')
        self.gain_21372316_label.setFixedWidth(150)
        self.gain_21372316_label.setAlignment(Qt.AlignmentFlag.AlignLeft |
                                              Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.gain_21372316_label, 4, 7)
        self.generalLayout.addWidget(self.gain_21372316, 4, 8, alignment=Qt.AlignmentFlag.AlignVCenter)

        self.generalLayout.addWidget(QLabel("Camera 21372315 (white) settings"),
                                     6, 7, 6, 9, alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        self.exposure_time_21372315 = QSlider(Qt.Orientation.Horizontal)
        self.exposure_time_21372315.setRange(500, 30000)
        self.exposure_time_21372315.setValue(2500)
        self.exposure_time_21372315.valueChanged.connect(self._update_exposure_time_21372315_label)
        self.exposure_time_21372315_label = QLabel('exp time (2500 μs):')
        self.exposure_time_21372315_label.setFixedWidth(150)
        self.exposure_time_21372315_label.setAlignment(Qt.AlignmentFlag.AlignLeft |
                                                       Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.exposure_time_21372315_label, 7, 7)
        self.generalLayout.addWidget(self.exposure_time_21372315, 7, 8, alignment=Qt.AlignmentFlag.AlignVCenter)

        self.gain_21372315 = QSlider(Qt.Orientation.Horizontal)
        self.gain_21372315.setRange(0, 15)
        self.gain_21372315.setValue(0)
        self.gain_21372315.valueChanged.connect(self._update_gain_21372315_label)
        self.gain_21372315_label = QLabel('digital gain (0 dB):')
        self.gain_21372315_label.setFixedWidth(150)
        self.gain_21372315_label.setAlignment(Qt.AlignmentFlag.AlignLeft |
                                              Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.gain_21372315_label, 8, 7)
        self.generalLayout.addWidget(self.gain_21372315, 8, 8, alignment=Qt.AlignmentFlag.AlignVCenter)

        self.generalLayout.addWidget(QLabel("Camera 21369048 (red) settings"),
                                     10, 7, 10, 9, alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        self.exposure_time_21369048 = QSlider(Qt.Orientation.Horizontal)
        self.exposure_time_21369048.setRange(500, 30000)
        self.exposure_time_21369048.setValue(2500)
        self.exposure_time_21369048.valueChanged.connect(self._update_exposure_time_21369048_label)
        self.exposure_time_21369048_label = QLabel('exp time (2500 μs):')
        self.exposure_time_21369048_label.setFixedWidth(150)
        self.exposure_time_21369048_label.setAlignment(Qt.AlignmentFlag.AlignLeft |
                                                       Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.exposure_time_21369048_label, 11, 7)
        self.generalLayout.addWidget(self.exposure_time_21369048, 11, 8, alignment=Qt.AlignmentFlag.AlignVCenter)

        self.gain_21369048 = QSlider(Qt.Orientation.Horizontal)
        self.gain_21369048.setRange(0, 15)
        self.gain_21369048.setValue(0)
        self.gain_21369048.valueChanged.connect(self._update_gain_21369048_label)
        self.gain_21369048_label = QLabel('digital gain (0 dB):')
        self.gain_21369048_label.setFixedWidth(150)
        self.gain_21369048_label.setAlignment(Qt.AlignmentFlag.AlignLeft |
                                              Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.gain_21369048_label, 12, 7)
        self.generalLayout.addWidget(self.gain_21369048, 12, 8, alignment=Qt.AlignmentFlag.AlignVCenter)

        self.generalLayout.addWidget(QLabel("Camera 22085397 (cyan) settings"),
                                     14, 7, 14, 9, alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        self.exposure_time_22085397 = QSlider(Qt.Orientation.Horizontal)
        self.exposure_time_22085397.setRange(500, 30000)
        self.exposure_time_22085397.setValue(2500)
        self.exposure_time_22085397.valueChanged.connect(self._update_exposure_time_22085397_label)
        self.exposure_time_22085397_label = QLabel('exp time (2500 μs):')
        self.exposure_time_22085397_label.setFixedWidth(150)
        self.exposure_time_22085397_label.setAlignment(Qt.AlignmentFlag.AlignLeft |
                                                       Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.exposure_time_22085397_label, 15, 7)
        self.generalLayout.addWidget(self.exposure_time_22085397, 15, 8, alignment=Qt.AlignmentFlag.AlignVCenter)

        self.gain_22085397 = QSlider(Qt.Orientation.Horizontal)
        self.gain_22085397.setRange(0, 15)
        self.gain_22085397.setValue(0)
        self.gain_22085397.valueChanged.connect(self._update_gain_22085397_label)
        self.gain_22085397_label = QLabel('digital gain (0 dB):')
        self.gain_22085397_label.setFixedWidth(150)
        self.gain_22085397_label.setAlignment(Qt.AlignmentFlag.AlignLeft |
                                              Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.gain_22085397_label, 16, 7)
        self.generalLayout.addWidget(self.gain_22085397, 16, 8, alignment=Qt.AlignmentFlag.AlignVCenter)

        self.generalLayout.addWidget(QLabel("Camera 21241563 (yellow) settings"),
                                     18, 7, 18, 9, alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        self.exposure_time_21241563 = QSlider(Qt.Orientation.Horizontal)
        self.exposure_time_21241563.setRange(500, 30000)
        self.exposure_time_21241563.setValue(2500)
        self.exposure_time_21241563.valueChanged.connect(self._update_exposure_time_21241563_label)
        self.exposure_time_21241563_label = QLabel('exp time (2500 μs):')
        self.exposure_time_21241563_label.setFixedWidth(150)
        self.exposure_time_21241563_label.setAlignment(Qt.AlignmentFlag.AlignLeft |
                                                       Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.exposure_time_21241563_label, 19, 7)
        self.generalLayout.addWidget(self.exposure_time_21241563, 19, 8, alignment=Qt.AlignmentFlag.AlignVCenter)

        self.gain_21241563 = QSlider(Qt.Orientation.Horizontal)
        self.gain_21241563.setRange(0, 15)
        self.gain_21241563.setValue(0)
        self.gain_21241563.valueChanged.connect(self._update_gain_21241563_label)
        self.gain_21241563_label = QLabel('digital gain (0 dB):')
        self.gain_21241563_label.setFixedWidth(150)
        self.gain_21241563_label.setAlignment(Qt.AlignmentFlag.AlignLeft |
                                              Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.gain_21241563_label, 20, 7)
        self.generalLayout.addWidget(self.gain_21241563, 20, 8, alignment=Qt.AlignmentFlag.AlignVCenter)

        self._create_buttons_record(seq=2)

    def record_four(self):
        self.ConductRecording = ConductRecording(self)
        self.generalLayout = QGridLayout()
        self.ConductRecording.setLayout(self.generalLayout)
        self.setWindowTitle("PNI Vocal Gymnasium (Conduct recording)")
        self.setCentralWidget(self.ConductRecording)

        self.txt_edit = QPlainTextEdit()
        self.txt_edit.setReadOnly(True)
        self.generalLayout.addWidget(self.txt_edit, 0, 0,
                                     alignment=Qt.AlignmentFlag.AlignTop)

        self.experiment_progress_bar = QProgressBar()
        self.generalLayout.addWidget(self.experiment_progress_bar, 1, 0,
                                     alignment=Qt.AlignmentFlag.AlignTop)
        self.experiment_progress_bar.setStyleSheet("QProgressBar { min-width: 1030px; min-height: 30px;}")

        self._save_modified_values_to_toml()

        exp_settings_dict_final = toml.load(f"{self.settings_dict['general']['config_settings_directory']}{os.sep}behavioral_experiments_settings.toml")
        self.run_exp = ExperimentController(message_output=self.message,
                                            email_receivers=self.email_recipients,
                                            exp_settings_dict=exp_settings_dict_final)

        self._create_buttons_record(seq=3)

    def process_one(self):
        self.ProcessSettings = ProcessSettings(self)
        self.generalLayout = QGridLayout()
        self.ProcessSettings.setLayout(self.generalLayout)
        self.setWindowTitle("PNI Vocal Gymnasium (Process recordings > Settings)")
        self.setCentralWidget(self.ProcessSettings)

        # select all directories for processing
        self.processing_dir_edit = QTextEdit('[]')
        self.generalLayout.addWidget(QLabel('(*) directories for processing'), 0, 0, alignment=Qt.AlignmentFlag.AlignCenter |Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.processing_dir_edit, 1, 0, 10, 1, alignment=Qt.AlignmentFlag.AlignTop)

        # set other parameters
        self.gvs_label = QLabel("Video processing settings")
        self.gvs_label.setStyleSheet("QLabel { font-weight: bold;}")
        self.generalLayout.addWidget(self.gvs_label,
                                     11, 0, alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        self.conduct_video_concatenation_cb = QComboBox()
        self.conduct_video_concatenation_cb.setStyleSheet("QComboBox { min-width: 80px;}")
        self.conduct_video_concatenation_cb.addItems(['Yes', 'No'])
        self.conduct_video_concatenation_cb.activated.connect(self._conduct_video_concatenation_combo_box_activated)
        self.generalLayout.addWidget(QLabel('conduct video concatenation:'), 12, 0, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.conduct_video_concatenation_cb, 12, 0, alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop)

        self.concatenate_cam_serial_num = QLineEdit('21241563,21369048,21372315,21372316,22085397')
        self.concatenate_cam_serial_num.setStyleSheet("QLineEdit { min-width: 200px;}")
        self.generalLayout.addWidget(QLabel('cam_serial_num:'), 13, 0, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.concatenate_cam_serial_num, 13, 0, alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop)

        self.concatenate_video_ext = QLineEdit('mp4')
        self.concatenate_video_ext.setStyleSheet("QLineEdit { min-width: 200px;}")
        self.generalLayout.addWidget(QLabel('video_extension:'), 14, 0, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.concatenate_video_ext, 14, 0, alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop)

        self.concatenated_video_name = QLineEdit('concatenated_temp')
        self.concatenated_video_name.setStyleSheet("QLineEdit { min-width: 200px;}")
        self.generalLayout.addWidget(QLabel('con_video_name:'), 15, 0, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.concatenated_video_name, 15, 0, alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop)


        # self.generalLayout.addWidget(QLabel("         "), 5, 2, alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.general_line_example = QLineEdit('["21241563", "21369048", "21372315", "21372316", "22085397"]')
        self.generalLayout.addWidget(QLabel('camera_serial_num:'), 0, 3, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.general_line_example, 0, 4, 0, 4, alignment=Qt.AlignmentFlag.AlignTop)

        # self.generalLayout.addWidget(QLabel("         "), 5, 6, alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.general_line_example_two = QLineEdit('["21241563", "21369048", "21372315", "21372316", "22085397"]')
        self.generalLayout.addWidget(QLabel('camera_serial_num:'), 0, 20, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.general_line_example_two, 0, 21, 0, 21, alignment=Qt.AlignmentFlag.AlignTop)

        self._create_buttons_process(seq=0)

    def _save_modified_values_to_toml(self):
        self.exp_settings_dict = toml.load(f"{self.settings_dict['general']['config_settings_directory']}{os.sep}behavioral_experiments_settings.toml")

        if self.exp_settings_dict["avisoft_recorder_exe"] != self.settings_dict['general']['avisoft_recorder_exe']:
            self.exp_settings_dict["avisoft_recorder_exe"] = self.settings_dict['general']['avisoft_recorder_exe']

        if self.exp_settings_dict["avisoft_basedirectory"] != self.settings_dict['general']['avisoft_basedirectory']:
            self.exp_settings_dict["avisoft_basedirectory"] = self.settings_dict['general']['avisoft_basedirectory']

        if self.exp_settings_dict["coolterm_basedirectory"] != self.settings_dict['general']['coolterm_basedirectory']:
            self.exp_settings_dict["coolterm_basedirectory"] = self.settings_dict['general']['coolterm_basedirectory']

        if self.exp_settings_dict["recording_files_destination_linux"] != self.settings_dict['general']['recording_files_destination_linux']:
            self.exp_settings_dict["recording_files_destination_linux"] = self.settings_dict['general']['recording_files_destination_linux']

        if self.exp_settings_dict["recording_files_destination_win"] != self.settings_dict['general']['recording_files_destination_win']:
            self.exp_settings_dict["recording_files_destination_win"] = self.settings_dict['general']['recording_files_destination_win']

        if self.exp_settings_dict["conduct_tracking_calibration"] != self.settings_dict['general']['conduct_tracking_calibration']:
            self.exp_settings_dict["conduct_tracking_calibration"] = self.settings_dict['general']['conduct_tracking_calibration']

        if self.exp_settings_dict["calibration_duration"] != ast.literal_eval(self.settings_dict['general']['calibration_duration']):
            self.exp_settings_dict["calibration_duration"] = ast.literal_eval(self.settings_dict['general']['calibration_duration'])

        if self.exp_settings_dict["conduct_audio_recording"] != self.settings_dict['general']['conduct_audio_recording']:
            self.exp_settings_dict["conduct_audio_recording"] = self.settings_dict['general']['conduct_audio_recording']

        if self.exp_settings_dict["video_session_duration"] != ast.literal_eval(self.settings_dict['general']['video_session_duration']):
            self.exp_settings_dict["video_session_duration"] = ast.literal_eval(self.settings_dict['general']['video_session_duration'])

        self.experiment_time_sec += ((ast.literal_eval(self.settings_dict['general']['video_session_duration']) + 0.36) * 60)

        for audio_key in self.settings_dict['audio'].keys():
            if audio_key in self.exp_settings_dict['audio'].keys():
                if audio_key != 'used_mics':
                    if self.exp_settings_dict['audio'][audio_key] != ast.literal_eval(self.settings_dict['audio'][audio_key]):
                        self.exp_settings_dict['audio'][audio_key] = ast.literal_eval(self.settings_dict['audio'][audio_key])
                        self.modify_audio_config = True
                else:
                    used_mics_temp = [int(mic) for mic in self.settings_dict['audio'][audio_key].split(',')]
                    if self.exp_settings_dict['audio'][audio_key] != used_mics_temp:
                        self.exp_settings_dict['audio'][audio_key] = used_mics_temp
                        self.modify_audio_config = True

            elif audio_key in self.exp_settings_dict['audio']['general'].keys() and \
                    self.exp_settings_dict['audio']['general'][audio_key] != ast.literal_eval(self.settings_dict['audio'][audio_key]):
                self.exp_settings_dict['audio']['general'][audio_key] = ast.literal_eval(self.settings_dict['audio'][audio_key])
                self.modify_audio_config = True

            elif audio_key in self.exp_settings_dict['audio']['screen_position'].keys() and \
                    self.exp_settings_dict['audio']['screen_position'][audio_key] != ast.literal_eval(self.settings_dict['audio'][audio_key]):
                self.exp_settings_dict['audio']['screen_position'][audio_key] = ast.literal_eval(self.settings_dict['audio'][audio_key])
                self.modify_audio_config = True

            elif audio_key in self.exp_settings_dict['audio']['devices'].keys() and \
                    self.exp_settings_dict['audio']['devices'][audio_key] != ast.literal_eval(self.settings_dict['audio'][audio_key]):
                self.exp_settings_dict['audio']['devices'][audio_key] = ast.literal_eval(self.settings_dict['audio'][audio_key])
                self.modify_audio_config = True

            elif audio_key in self.exp_settings_dict['audio']['mics_config'].keys():
                if audio_key != 'ditctime':
                    if self.exp_settings_dict['audio']['mics_config'][audio_key] != ast.literal_eval(self.settings_dict['audio'][audio_key]):
                        self.exp_settings_dict['audio']['mics_config'][audio_key] = ast.literal_eval(self.settings_dict['audio'][audio_key])
                        if audio_key not in ["name", "id", "typech", "deviceid", "channel"]:
                            self.modify_audio_config = True
                else:
                    if self.exp_settings_dict['audio']['mics_config'][audio_key] != self.settings_dict['audio'][audio_key]:
                        self.exp_settings_dict['audio']['mics_config'][audio_key] = self.settings_dict['audio'][audio_key]
                        self.modify_audio_config = True

            elif audio_key in self.exp_settings_dict['audio']['monitor'].keys() and \
                    self.exp_settings_dict['audio']['monitor'][audio_key] != ast.literal_eval(self.settings_dict['audio'][audio_key]):
                self.exp_settings_dict['audio']['monitor'][audio_key] = ast.literal_eval(self.settings_dict['audio'][audio_key])
                self.modify_audio_config = True

            elif audio_key in self.exp_settings_dict['audio']['call'].keys() and \
                    self.exp_settings_dict['audio']['call'][audio_key] != ast.literal_eval(self.settings_dict['audio'][audio_key]):
                self.exp_settings_dict['audio']['call'][audio_key] = ast.literal_eval(self.settings_dict['audio'][audio_key])
                self.modify_audio_config = True

        audio_config = configparser.ConfigParser()
        audio_config.read(f"{self.settings_dict['general']['config_settings_directory']}"
                          f"{os.sep}avisoft_config.ini")
        if float(audio_config['Configuration']['timer']) != (float(self.settings_dict['general']['video_session_duration']) + .36) * 60:
            self.modify_audio_config = True

        if self.exp_settings_dict["modify_audio_config_file"] != self.modify_audio_config:
            self.exp_settings_dict["modify_audio_config_file"] = self.modify_audio_config

        for video_key in self.settings_dict['video'].keys():
            if video_key in self.exp_settings_dict['video']['general'].keys():
                if video_key in ['browser', 'recording_codec', 'specific_camera_serial', 'monitor_recording',
                                 'monitor_specific_camera', 'delete_post_copy']:
                    if self.exp_settings_dict['video']['general'][video_key] != self.settings_dict['video'][video_key]:
                        self.exp_settings_dict['video']['general'][video_key] = self.settings_dict['video'][video_key]
                elif video_key == 'expected_camera_num':
                    if self.exp_settings_dict['video']['general'][video_key] != ast.literal_eval(self.settings_dict['video'][video_key]):
                        self.exp_settings_dict['video']['general'][video_key] = ast.literal_eval(self.settings_dict['video'][video_key])
                else:
                    if self.exp_settings_dict['video']['general'][video_key] != self.settings_dict['video'][video_key]:
                        self.exp_settings_dict['video']['general'][video_key] = self.settings_dict['video'][video_key]
            elif video_key in self.exp_settings_dict['video']['metadata'].keys() and \
                    self.exp_settings_dict['video']['metadata'][video_key] != self.settings_dict['video'][video_key]:
                self.exp_settings_dict['video']['metadata'][video_key] = self.settings_dict['video'][video_key]
            else:
                for camera_id in ['21372316', '21372315', '21369048', '22085397', '21241563']:
                    if camera_id in video_key:
                        if 'gain' in video_key:
                            if self.exp_settings_dict['video']['cameras_config'][camera_id]['gain'] != self.settings_dict['video'][f'gain_{camera_id}']:
                                self.exp_settings_dict['video']['cameras_config'][camera_id]['gain'] = self.settings_dict['video'][f'gain_{camera_id}']
                        else:
                            if self.exp_settings_dict['video']['cameras_config'][camera_id]['exposure_time'] != self.settings_dict['video'][f'exposure_time_{camera_id}']:
                                self.exp_settings_dict['video']['cameras_config'][camera_id]['exposure_time'] = self.settings_dict['video'][f'exposure_time_{camera_id}']

        with open(f"{self.settings_dict['general']['config_settings_directory']}{os.sep}behavioral_experiments_settings.toml", 'w') as toml_file:
            toml.dump(self.exp_settings_dict, toml_file)

        self.message(f"Updating the configuration .toml file completed at: "
                     f"{datetime.datetime.now().hour:02d}:{datetime.datetime.now().minute:02d}.{datetime.datetime.now().second:02d}.")

    def _move_progress_bar(self):
        time_to_sleep = int(round(self.experiment_time_sec * 10))
        for i in range(101):
            _loop_time(delay_time=time_to_sleep)
            self.experiment_progress_bar.setValue(i)
        self.experiment_time_sec = 0

    def message(self, s):
        self.txt_edit.appendPlainText(s)

    def _save_record_three_labels_func(self):
        video_dict_keys = ["browser", "expected_camera_num", "recording_codec", "specific_camera_serial",
                           "experimenter", "mice_num", "cage_ID_m1", "mouse_ID_m1", "genotype_m1", "sex_m1", "dob_m1",
                           "housing_m1", "cage_ID_m2", "mouse_ID_m2", "genotype_m2", "sex_m2", "dob_m2", "housing_m2", "other"]

        self.settings_dict['video']['monitor_recording'] = self.monitor_recording_cb_bool
        self.monitor_recording_cb_bool = True
        self.settings_dict['video']['monitor_specific_camera'] = self.monitor_specific_camera_cb_bool
        self.monitor_specific_camera_cb_bool = False
        self.settings_dict['video']['delete_post_copy'] = self.delete_post_copy_cb_bool
        self.delete_post_copy_cb_bool = True

        self.settings_dict['video']['cameras_frame_rate'] = self.cameras_frame_rate.value()

        self.settings_dict['video']['exposure_time_21372316'] = self.exposure_time_21372316.value()
        self.settings_dict['video']['gain_21372316'] = self.gain_21372316.value()
        self.settings_dict['video']['exposure_time_21372315'] = self.exposure_time_21372315.value()
        self.settings_dict['video']['gain_21372315'] = self.gain_21372315.value()
        self.settings_dict['video']['exposure_time_21369048'] = self.exposure_time_21369048.value()
        self.settings_dict['video']['gain_21369048'] = self.gain_21369048.value()
        self.settings_dict['video']['exposure_time_22085397'] = self.exposure_time_22085397.value()
        self.settings_dict['video']['gain_22085397'] = self.gain_22085397.value()
        self.settings_dict['video']['exposure_time_21241563'] = self.exposure_time_21241563.value()
        self.settings_dict['video']['gain_21241563'] = self.gain_21241563.value()

        for variable in video_dict_keys:
            self.settings_dict['video'][variable] = getattr(self, variable).text()

    def _update_exposure_time_21241563_label(self, value):
        self.exposure_time_21241563_label.setText(f"exp time ({str(value)} μs):")

    def _update_gain_21241563_label(self, value):
        self.gain_21241563_label.setText(f"digital gain ({str(value)} dB):")

    def _update_exposure_time_22085397_label(self, value):
        self.exposure_time_22085397_label.setText(f"exp time ({str(value)} μs):")

    def _update_gain_22085397_label(self, value):
        self.gain_22085397_label.setText(f"digital gain ({str(value)} dB):")

    def _update_exposure_time_21369048_label(self, value):
        self.exposure_time_21369048_label.setText(f"exp time ({str(value)} μs):")

    def _update_gain_21369048_label(self, value):
        self.gain_21369048_label.setText(f"digital gain ({str(value)} dB):")

    def _update_exposure_time_21372315_label(self, value):
        self.exposure_time_21372315_label.setText(f"exp time ({str(value)} μs):")

    def _update_gain_21372315_label(self, value):
        self.gain_21372315_label.setText(f"digital gain ({str(value)} dB):")

    def _update_exposure_time_21372316_label(self, value):
        self.exposure_time_21372316_label.setText(f"exp time ({str(value)} μs):")

    def _update_gain_21372316_label(self, value):
        self.gain_21372316_label.setText(f"digital gain ({str(value)} dB):")

    def _update_fr_label(self, value):
        self.fr_label.setText(f"camera freq ({str(value)} fps):")

    def _save_record_two_labels_func(self):
        for variable in self.first_col_labels_1 + self.first_col_labels_2 + self.second_col_labels:
            self.settings_dict['audio'][f'{variable[:-1]}'] = getattr(self, variable[:-1]).text()

    def _create_buttons_process(self, seq):
        if seq == 0:
            previous_win = self.main_window
            next_win_connect = []

        self.button_map = {"Previous": QPushButton(QIcon(previous_icon), "Previous")}
        self.button_map["Previous"].clicked.connect(previous_win)
        self.generalLayout.addWidget(self.button_map["Previous"], 95, 0, alignment=Qt.AlignmentFlag.AlignLeft)

        self.button_map["Main"] = QPushButton(QIcon(main_icon), "Main")
        self.button_map["Main"].clicked.connect(self.main_window)
        self.generalLayout.addWidget(self.button_map["Main"], 95, 71)

        self.button_map["Next"] = QPushButton(QIcon(next_icon), "Next")
        self.button_map["Next"].setEnabled(False)
        # for one_connection in next_win_connect:
        #     self.button_map["Next"].clicked.connect(one_connection)
        self.generalLayout.addWidget(self.button_map["Next"], 95, 58)

    def _create_buttons_main(self):
        self.button_map = {"Process": QPushButton(QIcon(process_icon), "Process")}
        self.button_map["Process"].clicked.connect(self.process_one)
        self.generalLayout.addWidget(self.button_map["Process"], 57, 2)

        self.button_map["Record"] = QPushButton(QIcon(record_icon), "Record")
        self.button_map["Record"].clicked.connect(self.record_one)
        self.generalLayout.addWidget(self.button_map["Record"], 57, 1, alignment=Qt.AlignmentFlag.AlignRight)

    def _create_buttons_record(self, seq):
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

        self.button_map = {"Previous": QPushButton(QIcon(previous_icon), "Previous")}
        self.button_map["Previous"].clicked.connect(previous_win)
        self.generalLayout.addWidget(self.button_map["Previous"], 95, 0, alignment=Qt.AlignmentFlag.AlignLeft)

        self.button_map["Main"] = QPushButton(QIcon(main_icon), "Main")
        self.button_map["Main"].clicked.connect(self.main_window)
        self.generalLayout.addWidget(self.button_map["Main"], 95, 71)

        if len(next_win_connect) > 0:
            self.button_map["Next"] = QPushButton(QIcon(next_icon), "Next")
            for one_connection in next_win_connect:
                self.button_map["Next"].clicked.connect(one_connection)
            self.generalLayout.addWidget(self.button_map["Next"], 95, 58)
        else:
            if self.settings_dict['general']['conduct_tracking_calibration']:
                self.button_map["Calibrate"] = QPushButton(QIcon(calibrate_icon), "Calibrate")
                self.button_map["Calibrate"].clicked.connect(self._disable_other_buttons)
                self.button_map["Calibrate"].clicked.connect(self._start_calibration)
                self.button_map["Calibrate"].clicked.connect(self._enable_other_buttons_post_cal)
                self.generalLayout.addWidget(self.button_map["Calibrate"], 95, 45)

            self.button_map["Record"] = QPushButton(QIcon(record_icon), "Record")
            self.button_map["Record"].clicked.connect(self._disable_other_buttons)
            self.button_map["Record"].clicked.connect(self._start_recording)
            self.button_map["Record"].clicked.connect(self._enable_other_buttons_post_rec)
            self.generalLayout.addWidget(self.button_map["Record"], 95, 58)

    def _start_calibration(self):
        cal_thread = threading.Thread(self.run_exp.conduct_tracking_calibration())
        cal_thread.start()

    def _start_recording(self):
        prog_bar_thread = threading.Thread(target=self._move_progress_bar)
        prog_bar_thread.start()

        rec_thread = threading.Thread(self.run_exp.conduct_behavioral_recording())
        rec_thread.start()

    def _enable_other_buttons_post_cal(self):
        self.button_map["Main"].setEnabled(True)
        self.button_map["Record"].setEnabled(True)

    def _enable_other_buttons_post_rec(self):
        self.button_map["Main"].setEnabled(True)

    def _disable_other_buttons(self):
        self.button_map["Previous"].setEnabled(False)
        self.button_map["Main"].setEnabled(False)
        self.button_map["Record"].setEnabled(False)
        if self.settings_dict['general']['conduct_tracking_calibration']:
            self.button_map["Calibrate"].setEnabled(False)

    def _open_settings_dialog(self):
        settings_dir_name = QFileDialog.getExistingDirectory(
            self,
            "Select settings directory",
            "C:\\experiment_running_docs")
        if settings_dir_name:
            settings_dir_name_path = Path(settings_dir_name)
            self.dir_settings_edit.setText(str(settings_dir_name_path))
            self.settings_dict['general']['config_settings_directory'] = str(settings_dir_name_path).replace(os.sep, '\\')
        else:
            self.settings_dict['general']['config_settings_directory'] = ''

    def _open_recorder_dialog(self):
        recorder_dir_name = QFileDialog.getExistingDirectory(
            self,
            "Select Avisoft Recorder USGH directory",
            "C:\\Program Files (x86)\\Avisoft Bioacoustics\\RECORDER USGH")
        if recorder_dir_name:
            recorder_dir_name_path = Path(recorder_dir_name)
            self.recorder_settings_edit.setText(str(recorder_dir_name_path))
            self.settings_dict['general']['avisoft_recorder_exe'] = str(recorder_dir_name_path).replace(os.sep, '\\')
        else:
            self.settings_dict['general']['avisoft_recorder_exe'] = ''

    def _open_avisoft_dialog(self):
        avisoft_dir_name = QFileDialog.getExistingDirectory(
            self,
            "Select Avisoft base directory",
            "C:\\Users\\bmimica\\Documents\\Avisoft Bioacoustics")
        if avisoft_dir_name:
            avisoft_dir_name_path = Path(avisoft_dir_name)
            self.avisoft_base_edit.setText(str(avisoft_dir_name_path))
            self.settings_dict['general']['avisoft_basedirectory'] = str(avisoft_dir_name_path).replace(os.sep, '\\') + '\\'
        else:
            self.settings_dict['general']['avisoft_basedirectory'] = ''

    def _open_coolterm_dialog(self):
        coolterm_dir_name = QFileDialog.getExistingDirectory(
            self,
            "Select Coolterm base directory",
            "D:\\CoolTermWin")
        if coolterm_dir_name:
            coolterm_dir_name_path = Path(coolterm_dir_name)
            self.coolterm_base_edit.setText(str(coolterm_dir_name_path))
            self.settings_dict['general']['coolterm_basedirectory'] = str(coolterm_dir_name_path).replace(os.sep, '\\')
        else:
            self.settings_dict['general']['coolterm_basedirectory'] = ''

    def _monitor_recording_combo_box_activated(self, index):
        if index == 1:
            self.monitor_recording_cb_bool = False
        else:
            self.monitor_recording_cb_bool = True

    def _monitor_specific_camera_combo_box_activated(self, index):
        if index == 1:
            self.monitor_specific_camera_cb_bool = True
        else:
            self.monitor_specific_camera_cb_bool = False

    def _delete_post_copy_combo_box_activated(self, index):
        if index == 1:
            self.delete_post_copy_cb_bool = False
        else:
            self.delete_post_copy_cb_bool = True

    def _conduct_audio_combo_box_activated(self, index):
        if index == 1:
            self.conduct_audio_cb_bool = False
        else:
            self.conduct_audio_cb_bool = True

    def _conduct_tracking_calibration_combo_box_activated(self, index):
        if index == 1:
            self.conduct_tracking_calibration_cb_bool = True
        else:
            self.conduct_tracking_calibration_cb_bool = False

    def _conduct_video_concatenation_combo_box_activated(self, index):
        if index == 1:
            self.conduct_video_concatenation_cb_bool = False
        else:
            self.conduct_video_concatenation_cb_bool = True

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

        if len(self.email_recipients) == 0:
            self.email_recipients = []
        else:
            self.email_recipients = self.email_recipients.split(',')

        self.settings_dict['general']['recording_files_destination_linux'] = self.recording_files_destination_linux
        self.settings_dict['general']['recording_files_destination_win'] = self.recording_files_destination_windows
        self.settings_dict['general']['video_session_duration'] = self.video_session_duration
        self.settings_dict['general']['calibration_duration'] = self.calibration_session_duration

        self.settings_dict['general']['conduct_tracking_calibration'] = self.conduct_tracking_calibration_cb_bool
        self.conduct_tracking_calibration_cb_bool = False
        self.settings_dict['general']['conduct_audio_recording'] = self.conduct_audio_cb_bool
        self.conduct_audio_cb_bool = True

    def _location_on_the_screen(self):
        top_left_point = QGuiApplication.primaryScreen().availableGeometry().topLeft()
        self.move(top_left_point)


def main():
    """Vocal Gym's main function."""
    vocal_gym_app = QApplication([])

    vocal_gym_app.setStyle('Fusion')
    with open('vgg_style_sheet.css', 'r') as file:
        vocal_gym_app.setStyleSheet(file.read())

    # splash = QSplashScreen(QPixmap(splash_icon))
    # progress_bar = QProgressBar(splash)
    # progress_bar.setGeometry(0, 0, 610, 20)
    # splash.show()
    # for i in range(0, 101):
    #     progress_bar.setValue(i)
    #     t = time.time()
    #     while time.time() < t + 0.1:
    #         vocal_gym_app.processEvents()

    vocal_gym_window = VocalGymWindow()
    # splash.finish(vocal_gym_window)
    vocal_gym_window.show()

    sys.exit(vocal_gym_app.exec())


if __name__ == "__main__":
    main()
