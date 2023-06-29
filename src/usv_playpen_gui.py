"""
@author: bartulem
GUI to run behavioral experiments.
"""

import ast
import configparser
import ctypes
import datetime
import os
import sys
import threading
from functools import partial
from pathlib import Path
import time
import toml
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

from behavioral_experiments import ExperimentController, _loop_time
from preprocess_data import Stylist

if os.name == 'nt':
    my_app_id = 'mycompany.myproduct.subproduct.version'
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(my_app_id)

app_name = 'USV Playpen'
experimenter_id = 'bartulem'
email_list_global = ''
config_dir_global = 'C:\\experiment_running_docs'
avisoft_rec_dir_global = 'C:\\Program Files (x86)\\Avisoft Bioacoustics\\RECORDER USGH'
avisoft_base_dir_global = 'C:\\Users\\bmimica\\Documents\\Avisoft Bioacoustics\\'
coolterm_base_dir_global = 'D:\\CoolTermWin'
destination_linux_global = '/home/labadmin/falkner/Bartul/Data,/home/labadmin/murthy/Bartul/Data'
destination_win_global = 'F:\\Bartul\\Data,M:\\Bartul\\Data'
camera_ids_global = ['21372315', '21372316', '21369048', '22085397', '21241563']
camera_colors_global = ['white', 'orange', 'red', 'cyan', 'yellow']

basedir = os.path.dirname(__file__)
background_img = f'{basedir}{os.sep}img{os.sep}background_img.png'
background_img_2 = f'{basedir}{os.sep}img{os.sep}background_img_2.png'
background_process = f'{basedir}{os.sep}img{os.sep}process_background.png'
lab_icon = f'{basedir}{os.sep}img{os.sep}lab.png'
splash_icon = f'{basedir}{os.sep}img{os.sep}uncle_stefan.png'
process_icon = f'{basedir}{os.sep}img{os.sep}process.png'
record_icon = f'{basedir}{os.sep}img{os.sep}record.png'
previous_icon = f'{basedir}{os.sep}img{os.sep}previous.png'
next_icon = f'{basedir}{os.sep}img{os.sep}next.png'
main_icon = f'{basedir}{os.sep}img{os.sep}main.png'
calibrate_icon = f'{basedir}{os.sep}img{os.sep}calibrate.png'


class Main(QWidget):
    def __init__(self, parent=None):
        super(Main, self).__init__(parent)

    def paintEvent(self, event):
        paint_main = QPainter(self)
        paint_main.drawPixmap(self.rect(), QPixmap(f'{background_img}'))
        QWidget.paintEvent(self, event)


class Record(QWidget):
    def __init__(self, parent=Main):
        super(Record, self).__init__(parent)

    def paintEvent(self, event):
        paint_record = QPainter(self)
        paint_record.drawPixmap(self.rect(), QPixmap(f'{background_img}'))
        QWidget.paintEvent(self, event)


class AudioSettings(QWidget):
    def __init__(self, parent=Main):
        super(AudioSettings, self).__init__(parent)


class VideoSettings(QWidget):
    def __init__(self, parent=Main):
        super(VideoSettings, self).__init__(parent)

    def paintEvent(self, event):
        paint_video = QPainter(self)
        paint_video.drawPixmap(self.rect(), QPixmap(f'{background_img_2}'))
        QWidget.paintEvent(self, event)


class ConductRecording(QWidget):
    def __init__(self, parent=Main):
        super(ConductRecording, self).__init__(parent)


class ProcessSettings(QWidget):
    def __init__(self, parent=Main):
        super(ProcessSettings, self).__init__(parent)

    def paintEvent(self, event):
        paint_process = QPainter(self)
        paint_process.drawPixmap(self.rect(), QPixmap(f'{background_process}'))
        QWidget.paintEvent(self, event)


class ConductProcess(QWidget):
    def __init__(self, parent=Main):
        super(ConductProcess, self).__init__(parent)


class HyperlinkLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__()
        self.setOpenExternalLinks(True)
        self.setParent(parent)


class USVPlaypenWindow(QMainWindow):
    """Main window of GUI."""

    def __init__(self, **kwargs):
        super().__init__()
        self.setWindowIcon(QIcon(lab_icon))
        self.main_window()

        for attr, value in kwargs.items():
            setattr(self, attr, value)

        self.settings_dict = {'general': {'config_settings_directory': '',
                                          'avisoft_recorder_exe': '',
                                          'avisoft_basedirectory': '',
                                          'coolterm_basedirectory': ''}, 'audio': {}, 'video': {'expected_camera_num': len(camera_ids_global)}}
        self.processing_input_dict = {'processing_booleans': {
                                        'conduct_video_concatenation': True,
                                        'conduct_video_fps_change': True,
                                        'conduct_audio_multichannel_to_single_ch': True,
                                        'conduct_audio_cropping': True,
                                        'conduct_audio_to_mmap': True,
                                        'conduct_audio_filtering': True,
                                        'conduct_audio_video_sync': True,
                                        'conduct_phidget_data_extraction': True,
                                        'plot_sync_data': True},
                                      'preprocess_data': {
                                        'root_directories': []},
                                      'extract_phidget_data': {
                                        'Gatherer': {
                                          'prepare_data_for_analyses': {
                                            'extra_data_camera': '22085397',
                                            'sorting_key': 'sensor_time'}}},
                                      'file_loader': {
                                        'DataLoader': {
                                            'wave_data_loc': [''],
                                            'load_wavefile_data': {
                                                'library': 'scipy',
                                                'conditional_arg': []}}},
                                      'file_manipulation': {
                                        'Operator': {
                                          'concatenate_audio_files': {
                                            'audio_format': 'wav',
                                            'concat_type': 'vstack'},
                                          'filter_audio_files': {
                                            'audio_format': 'wav',
                                            'freq_hp': 2000,
                                            'freq_lp': 0},
                                          'concatenate_video_files': {
                                            'camera_serial_num': ['21241563', '21369048', '21372315', '21372316', '22085397'],
                                            'video_extension': 'mp4',
                                            'concatenated_video_name': 'concatenated_temp'},
                                          'rectify_video_fps': {
                                            'camera_serial_num': ['21241563', '21369048', '21372315', '21372316', '22085397'],
                                            'conversion_target_file': 'concatenated_temp',
                                            'video_extension': 'mp4',
                                            'calibration_fps': 10,
                                            'recording_fps': 150,
                                            'delete_old_file': True}}},
                                      'file_writer': {
                                        'DataWriter': {
                                          'wave_write_loc': '',
                                          'write_wavefile_data': {
                                            'library': 'scipy',
                                            'file_name': 'square_tone_repeats',
                                            'sampling_rate': 250000}}},
                                      'preprocessing_plot': {
                                        'SummaryPlotter': {
                                          'preprocessing_summary': {}}},
                                      'random_pulses': {
                                        'generate_truly_random_seed': {
                                          'dtype': 'uint16',
                                          'array_len': 1}},
                                      'send_email': {
                                        'Messenger': {
                                          'experimenter': f'{experimenter_id}',
                                          'toml_file_loc': '',
                                          'send_message': {
                                            'receivers': []}}},
                                      'synchronize_files': {
                                        'Synchronizer': {
                                          'find_audio_sync_trains': {
                                            'ch_receiving_input': 2,
                                            'sync_pulse_duration': 0.25,
                                            'break_proportion_threshold': 0.001,
                                            'ttl_proportion_threshold': 0.4},
                                          'find_video_sync_trains': {
                                            'camera_serial_num': ['21372315'],
                                            'led_px_version': 'current',
                                            'led_px_dev': 10,
                                            'video_extension': 'mp4',
                                            'mm_dtype': 'np.uint8',
                                            'relative_intensity_threshold': 0.35,
                                            'camera_fps': 150,
                                            'sync_pulse_duration': 0.25,
                                            'millisecond_divergence_tolerance': 10},
                                          'crop_wav_files_to_video': {
                                            'camera_serial_num': ['21241563', '21369048', '21372315', '21372316', '22085397'],
                                            'ch_receiving_input': 1,
                                            'ttl_pulse_duration': 0.00011333,
                                            'ttl_proportion_threshold': 1.0}}}}

    def main_window(self):
        # general settings
        self.Main = Main(self)
        self.setCentralWidget(self.Main)
        self._location_on_the_screen()
        self.generalLayout = QGridLayout()
        self.Main.setLayout(self.generalLayout)
        self.setWindowFlag(Qt.WindowType.WindowMaximizeButtonHint, False)
        self.setWindowTitle(f'{app_name}')

        # add text w/ links
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
        self.power_plan = HyperlinkLabel()
        self.power_plan = link_template.format('https://www.howtogeek.com/240840/should-you-use-the-balanced-power-saver-or-high-performance-power-plan-on-windows/', 'power plan')
        self.label = QLabel(f"<br>Thank you for using the {app_name}."
                            f"<br><br>To ensure quality recordings/data, make sure you install the following (prior to further usage): "
                            f"<br><br>(1) " + self.link_avisoft + " ≥4.4.14"
                            f"<br>(2) " + self.link_chrome + " (or other) web browser"
                            f"<br>(3) " + self.link_ffmpeg + " (and add it to PATH)"
                            f"<br>(4) " + self.link_arduino +
                            f"<br>(5) " + self.link_coolterm +
                            f"<br>(6) " + self.link_sox + " (and add it to PATH)"
                            f"<br><br> Change the Windows " + self.power_plan + " to 'High performance'."
                            f"<br><br> Contact the author for Arduino/Coolterm instructions and necessary configuration files.")
        self.label.setOpenExternalLinks(True)
        self.generalLayout.addWidget(self.label, 0, 0, alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        self._create_buttons_main()

    def record_one(self):
        # general settings
        self.Record = Record(self)
        self.setWindowTitle(f'{app_name} (Record > Select directories and set basic parameters)')
        self.setCentralWidget(self.Record)
        self.generalLayout = QGridLayout()
        self.Record.setLayout(self.generalLayout)

        self.title_label = QLabel('Please select appropriate directories (w/ config files or executables in them)')
        self.title_label.setStyleSheet('QLabel { font-weight: bold;}')
        self.generalLayout.addWidget(self.title_label, 0, 0, 0, 3, alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        settings_dir_btn = QPushButton('Browse')
        settings_dir_btn.setStyleSheet('QPushButton { min-width: 64px; min-height: 12px; max-width: 64px; max-height: 13px;}')
        self.settings_dict['general']['config_settings_directory'] = f'{config_dir_global}'
        self.processing_input_dict['send_email']['Messenger']['toml_file_loc'] = f'{config_dir_global}'
        settings_dir_btn.clicked.connect(self._open_settings_dialog)
        self.dir_settings_edit = QLineEdit(config_dir_global)

        self.generalLayout.addWidget(QLabel('settings file (*.toml) directory:'), 3, 0, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.dir_settings_edit, 3, 1, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(settings_dir_btn, 3, 2, alignment=Qt.AlignmentFlag.AlignTop)

        recorder_dir_btn = QPushButton('Browse')
        recorder_dir_btn.setStyleSheet('QPushButton { min-width: 64px; min-height: 12px; max-width: 64px; max-height: 13px;}')
        self.settings_dict['general']['avisoft_recorder_exe'] = f'{avisoft_rec_dir_global}'
        recorder_dir_btn.clicked.connect(self._open_recorder_dialog)
        self.recorder_settings_edit = QLineEdit(avisoft_rec_dir_global)

        self.generalLayout.addWidget(QLabel('Avisoft Recorder (usgh.exe) directory:'), 4, 0, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.recorder_settings_edit, 4, 1, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(recorder_dir_btn, 4, 2, alignment=Qt.AlignmentFlag.AlignTop)

        avisoft_base_dir_btn = QPushButton('Browse')
        avisoft_base_dir_btn.setStyleSheet('QPushButton { min-width: 64px; min-height: 12px; max-width: 64px; max-height: 13px;}')
        self.settings_dict['general']['avisoft_basedirectory'] = f'{avisoft_base_dir_global}'
        avisoft_base_dir_btn.clicked.connect(self._open_avisoft_dialog)
        self.avisoft_base_edit = QLineEdit(avisoft_base_dir_global)

        self.generalLayout.addWidget(QLabel('Avisoft Bioacoustics base directory:'), 5, 0, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.avisoft_base_edit, 5, 1, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(avisoft_base_dir_btn, 5, 2, alignment=Qt.AlignmentFlag.AlignTop)

        coolterm_base_dir_btn = QPushButton('Browse')
        coolterm_base_dir_btn.setStyleSheet('QPushButton { min-width: 64px; min-height: 12px; max-width: 64px; max-height: 13px;}')
        self.settings_dict['general']['coolterm_basedirectory'] = f'{coolterm_base_dir_global}'
        coolterm_base_dir_btn.clicked.connect(self._open_coolterm_dialog)
        self.coolterm_base_edit = QLineEdit(coolterm_base_dir_global)

        self.generalLayout.addWidget(QLabel('CoolTerm base directory:'), 6, 0, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.coolterm_base_edit, 6, 1, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(coolterm_base_dir_btn, 6, 2, alignment=Qt.AlignmentFlag.AlignTop)

        # recording files destination directories (across OS)
        self.recording_files_destination_linux = QLineEdit(f'{destination_linux_global}')
        self.generalLayout.addWidget(QLabel('recording file destinations (Linux):'), 9, 0, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.recording_files_destination_linux, 9, 1, 9, 2, alignment=Qt.AlignmentFlag.AlignTop)

        self.recording_files_destination_windows = QLineEdit(f'{destination_win_global}')
        self.generalLayout.addWidget(QLabel('recording file destinations (Windows):'), 10, 0, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.recording_files_destination_windows, 10, 1, 10, 2, alignment=Qt.AlignmentFlag.AlignTop)

        # set main recording parameters
        self.parameters_label = QLabel('Please set main recording parameters')
        self.parameters_label.setStyleSheet('QLabel { font-weight: bold;}')
        self.generalLayout.addWidget(self.parameters_label, 13, 0, 13, 3, alignment=Qt.AlignmentFlag.AlignTop)
        self.video_session_duration = QLineEdit('20')
        self.generalLayout.addWidget(QLabel('video session duration (min):'), 16, 0, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.video_session_duration, 16, 1, 16, 2, alignment=Qt.AlignmentFlag.AlignTop)

        self.conduct_audio_cb = QComboBox()
        self.conduct_audio_cb.addItems(['Yes', 'No'])
        self.conduct_audio_cb.activated.connect(partial(self._combo_box_prior_true, variable_id='conduct_audio_cb_bool'))
        self.generalLayout.addWidget(QLabel('conduct audio recording:'), 17, 0, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.conduct_audio_cb, 17, 1, alignment=Qt.AlignmentFlag.AlignTop)

        # tracking calibration settings
        self.conduct_tracking_calibration_cb = QComboBox()
        self.conduct_tracking_calibration_cb.addItems(['No', 'Yes'])
        self.conduct_tracking_calibration_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='conduct_tracking_calibration_cb_bool'))
        self.generalLayout.addWidget(QLabel('conduct video calibration:'), 18, 0, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.conduct_tracking_calibration_cb, 18, 1, alignment=Qt.AlignmentFlag.AlignTop)

        self.calibration_session_duration = QLineEdit('4')
        self.generalLayout.addWidget(QLabel('calibration session duration (min):'), 19, 0, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.calibration_session_duration, 19, 1, 19, 2, alignment=Qt.AlignmentFlag.AlignTop)

        self.email_recipients = QLineEdit(f'{email_list_global}')
        self.generalLayout.addWidget(QLabel('notify the following addresses about PC usage:'), 20, 0, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.email_recipients, 20, 1, 20, 2, alignment=Qt.AlignmentFlag.AlignTop)

        self._create_buttons_record(seq=0)

    def record_two(self):
        # general settings
        self.AudioSettings = AudioSettings(self)
        self.setWindowTitle(f'{app_name} (Record > Audio Settings)')
        self.setCentralWidget(self.AudioSettings)
        self.generalLayout = QGridLayout()
        self.generalLayout.setSpacing(2)
        self.AudioSettings.setLayout(self.generalLayout)

        default_audio_settings = {'name': '999', 'id': '999', 'typech': '13', 'deviceid': '999',
                                  'channel': '999', 'gain': '1', 'fullscalespl': '0.0', 'triggertype': '41',
                                  'toggle': '0', 'invert': '0', 'ditc': '0', 'ditctime': '00:00:00:00',
                                  'whistletracking': '0', 'wtbcf': '0', 'wtmaxchange': '3', 'wtmaxchange2_': '0',
                                  'wtminchange': '-10', 'wtminchange2_': '0', 'wtoutside': '0', 'hilowratioenable': '0',
                                  'hilowratio': '2.0', 'hilowratiofc': '15000.0', 'wtslope': '0', 'wtlevel': '0',
                                  'wtmindurtotal': '0.0', 'wtmindur': '0.005', 'wtmindur2_': '0.0', 'wtholdtime': '0.02',
                                  'wtmonotonic': '1', 'wtbmaxdur': '0', 'rejectwind': '0', 'rwentropy': '0.5',
                                  'rwfpegel': '2.5', 'rwdutycycle': '0.2', 'rwtimeconstant': '2.0', 'rwholdtime': '10.0',
                                  'fpegel': '5.0', 'energy': '0', 'frange': '1', 'entropyb': '0',
                                  'entropy': '0.35', 'increment': '1', 'fu': '0.0', 'fo': '250000.0',
                                  'pretrigger': '0.5', 'mint': '0.0', 'minst': '0.0', 'fhold': '0.5',
                                  'logfileno': '0', 'groupno': '0', 'callno': '0', 'timeconstant': '0.003',
                                  'timeexpansion': '0', 'startstop': '0', 'sayf': '2', 'over': '0',
                                  'delay': '0.0', 'center': '40000', 'bandwidth': '5', 'fd': '5',
                                  'decimation': '-1', 'device': '0', 'mode': '0', 'outfovertaps': '32',
                                  'outfoverabtast': '2000000', 'outformat': '2', 'outfabtast': '-22050', 'outdeviceid': '0',
                                  'outtype': '7', 'usghflags': '1574', 'diff': '0', 'format': '1',
                                  'type': '0', 'nbrwavehdr': '32', 'devbuffer': '0.032', 'ntaps': '32',
                                  'filtercutoff': '15.0', 'filter': '0', 'fabtast': '250000', 'y2': '1315',
                                  'x2': '2563', 'y1': '3', 'x1': '1378', 'fftlength': '256',
                                  'usvmonitoringflags': '9136', 'dispspectrogramcontrast': '0.0', 'disprangespectrogram': '250.0',
                                  'disprangeamplitude': '100.0', 'disprangewaveform': '100.0', 'total': '1', 'dcolumns': '3',
                                  'display': '2', 'total_mic_number': '24', 'total_device_num': '2',
                                  'used_mics': '0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23'}

        for audio_attr, audio_value in default_audio_settings.items():
            setattr(self, audio_attr, QLineEdit(audio_value))

        self.first_col_labels_1 = ['total_device_num:', 'total_mic_number:', 'used_mics:',
                                   'display:', 'dcolumns:', 'total:',
                                   'disprangewaveform:', 'disprangeamplitude:', 'disprangespectrogram:',
                                   'dispspectrogramcontrast:', 'usvmonitoringflags:', 'fftlength:',
                                   'x1:', 'y1:', 'x2:', 'y2:']

        for fc_idx_1, fc_item_1 in enumerate(self.first_col_labels_1):
            getattr(self, fc_item_1[:-1]).setFixedWidth(150)
            self.generalLayout.addWidget(QLabel(self.first_col_labels_1[fc_idx_1]), 5 + fc_idx_1, 0, alignment=Qt.AlignmentFlag.AlignTop)
            self.generalLayout.addWidget(getattr(self, fc_item_1[:-1]), 5 + fc_idx_1, 1, 5 + fc_idx_1, 2, alignment=Qt.AlignmentFlag.AlignTop)

        self.first_col_labels_2 = ['fabtast:', 'filter:', 'filtercutoff:',
                                   'ntaps:', 'devbuffer:', 'nbrwavehdr:',
                                   'type:', 'format:', 'diff:',
                                   'usghflags:', 'outtype:', 'outdeviceid:',
                                   'outfabtast:', 'outformat:', 'outfoverabtast:',
                                   'outfovertaps:', 'mode:', 'device:',
                                   'decimation:',
                                   'fd:', 'bandwidth:', 'center:',
                                   'delay:', 'over:', 'sayf:',
                                   'startstop:', 'timeexpansion:', 'timeconstant:',
                                   'callno:', 'groupno:', 'logfileno:']

        for fc_idx_2, fc_item_2 in enumerate(self.first_col_labels_2):
            getattr(self, fc_item_2[:-1]).setFixedWidth(150)
            self.generalLayout.addWidget(QLabel(self.first_col_labels_2[fc_idx_2]), 22 + fc_idx_2, 0, alignment=Qt.AlignmentFlag.AlignTop)
            self.generalLayout.addWidget(getattr(self, fc_item_2[:-1]), 22 + fc_idx_2, 1, 22 + fc_idx_2, 2, alignment=Qt.AlignmentFlag.AlignTop)

        self.second_col_labels = ['name:', 'id:', 'typech:',
                                  'deviceid:', 'channel:', 'gain:',
                                  'fullscalespl:', 'triggertype:', 'toggle:',
                                  'invert:', 'ditc:', 'ditctime:',
                                  'whistletracking:', 'wtbcf:', 'wtmaxchange:',
                                  'wtmaxchange2_:', 'wtminchange:', 'wtminchange2_:',
                                  'wtoutside:', 'hilowratioenable:', 'hilowratio:',
                                  'hilowratiofc:', 'wtslope:', 'wtlevel:',
                                  'wtmindurtotal:', 'wtmindur:', 'wtmindur2_:',
                                  'wtholdtime:', 'wtmonotonic:', 'wtbmaxdur:',
                                  'rejectwind:', 'rwentropy:', 'rwfpegel:',
                                  'rwdutycycle:', 'rwtimeconstant:', 'rwholdtime:',
                                  'fpegel:', 'energy:', 'frange:',
                                  'entropyb:', 'entropy:', 'increment:',
                                  'fu:', 'fo:', 'pretrigger:',
                                  'mint:', 'minst:', 'fhold:']

        for fc_idx_3, fc_item_3 in enumerate(self.second_col_labels):
            getattr(self, fc_item_3[:-1]).setFixedWidth(150)
            self.generalLayout.addWidget(QLabel(self.second_col_labels[fc_idx_3]), 5 + fc_idx_3, 55, alignment=Qt.AlignmentFlag.AlignTop)
            self.generalLayout.addWidget(getattr(self, fc_item_3[:-1]), 5 + fc_idx_3, 57, alignment=Qt.AlignmentFlag.AlignTop)

        self._create_buttons_record(seq=1)

    def record_three(self):
        # general settings
        self.VideoSettings = VideoSettings(self)
        self.setWindowTitle(f'{app_name} (Record > Video Settings)')
        self.setCentralWidget(self.VideoSettings)
        self.generalLayout = QGridLayout()
        self.VideoSettings.setLayout(self.generalLayout)

        self.gvs_label = QLabel('General video settings')
        self.gvs_label.setStyleSheet('QLabel { font-weight: bold;}')
        self.generalLayout.addWidget(self.gvs_label,
                                     0, 0, alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        self.browser = QLineEdit('chrome')
        self.generalLayout.addWidget(QLabel('browser:'), 2, 0, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.browser, 2, 1, alignment=Qt.AlignmentFlag.AlignTop)

        self.expected_cameras = QLineEdit('21372315,21372316,21369048,22085397,21241563')
        self.generalLayout.addWidget(QLabel('cameras to use:'), 3, 0, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.expected_cameras, 3, 1, alignment=Qt.AlignmentFlag.AlignTop)

        self.specific_camera_serial = QLineEdit('21372315')
        self.generalLayout.addWidget(QLabel('specific camera serial:'), 4, 0, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.specific_camera_serial, 4, 1, alignment=Qt.AlignmentFlag.AlignTop)

        self.recording_codec = QLineEdit('lq')
        self.generalLayout.addWidget(QLabel('recording codec:'), 5, 0, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.recording_codec, 5, 1, alignment=Qt.AlignmentFlag.AlignTop)

        self.monitor_recording_cb = QComboBox()
        self.monitor_recording_cb.addItems(['Yes', 'No'])
        self.monitor_recording_cb.activated.connect(partial(self._combo_box_prior_true, variable_id='monitor_recording_cb_bool'))
        self.generalLayout.addWidget(QLabel('monitor recording:'), 6, 0, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.monitor_recording_cb, 6, 1, alignment=Qt.AlignmentFlag.AlignTop)

        self.monitor_specific_camera_cb = QComboBox()
        self.monitor_specific_camera_cb.addItems(['No', 'Yes'])
        self.monitor_specific_camera_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='monitor_specific_camera_cb_bool'))
        self.generalLayout.addWidget(QLabel('monitor specific camera:'), 7, 0, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.monitor_specific_camera_cb, 7, 1, alignment=Qt.AlignmentFlag.AlignTop)

        self.delete_post_copy_cb = QComboBox()
        self.delete_post_copy_cb.addItems(['Yes', 'No'])
        self.delete_post_copy_cb.activated.connect(partial(self._combo_box_prior_true, variable_id='delete_post_copy_cb_bool'))
        self.generalLayout.addWidget(QLabel('delete post copy:'), 8, 0, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.delete_post_copy_cb, 8, 1, alignment=Qt.AlignmentFlag.AlignTop)

        self.calibration_frame_rate = QSlider(Qt.Orientation.Horizontal)
        self.calibration_frame_rate.setRange(10, 150)
        self.calibration_frame_rate.setValue(10)
        self.calibration_frame_rate.valueChanged.connect(self._update_cal_fr_label)

        self.cal_fr_label = QLabel('calibration (10 fps):')
        self.cal_fr_label.setFixedWidth(150)

        self.generalLayout.addWidget(self.cal_fr_label, 9, 0, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.calibration_frame_rate, 9, 1, alignment=Qt.AlignmentFlag.AlignVCenter)

        self.cameras_frame_rate = QSlider(Qt.Orientation.Horizontal)
        self.cameras_frame_rate.setRange(10, 150)
        self.cameras_frame_rate.setValue(150)
        self.cameras_frame_rate.valueChanged.connect(self._update_fr_label)

        self.fr_label = QLabel('recording (150 fps):')
        self.fr_label.setFixedWidth(150)

        self.generalLayout.addWidget(self.fr_label, 10, 0, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.cameras_frame_rate, 10, 1, alignment=Qt.AlignmentFlag.AlignVCenter)

        self.vm_label = QLabel('Video metadata')
        self.vm_label.setStyleSheet('QLabel { font-weight: bold;}')
        self.generalLayout.addWidget(self.vm_label,
                                     12, 0, alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        self.experimenter = QLineEdit(f'{experimenter_id}')
        self.generalLayout.addWidget(QLabel('experimenter:'), 13, 0, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.experimenter, 13, 1, alignment=Qt.AlignmentFlag.AlignTop)

        self.mice_num = QLineEdit('2')
        self.generalLayout.addWidget(QLabel('mice_num:'), 14, 0, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.mice_num, 14, 1, alignment=Qt.AlignmentFlag.AlignTop)

        self.cage_ID_m1 = QLineEdit('')
        self.generalLayout.addWidget(QLabel('cage_ID_m1:'), 15, 0, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.cage_ID_m1, 15, 1, alignment=Qt.AlignmentFlag.AlignTop)

        self.mouse_ID_m1 = QLineEdit('')
        self.generalLayout.addWidget(QLabel('mouse_ID_m1:'), 16, 0, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.mouse_ID_m1, 16, 1, alignment=Qt.AlignmentFlag.AlignTop)

        self.genotype_m1 = QLineEdit('CD1-WT')
        self.generalLayout.addWidget(QLabel('genotype_m1:'), 17, 0, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.genotype_m1, 17, 1, alignment=Qt.AlignmentFlag.AlignTop)

        self.sex_m1 = QLineEdit('')
        self.generalLayout.addWidget(QLabel('sex_m1:'), 18, 0, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.sex_m1, 18, 1, alignment=Qt.AlignmentFlag.AlignTop)

        self.dob_m1 = QLineEdit('')
        self.generalLayout.addWidget(QLabel('DOB_m1:'), 19, 0, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.dob_m1, 19, 1, alignment=Qt.AlignmentFlag.AlignTop)

        self.housing_m1 = QLineEdit('group')
        self.generalLayout.addWidget(QLabel('housing_m1:'), 20, 0, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.housing_m1, 20, 1, alignment=Qt.AlignmentFlag.AlignTop)

        self.cage_ID_m2 = QLineEdit('')
        self.generalLayout.addWidget(QLabel('cage_ID_m2:'), 21, 0, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.cage_ID_m2, 21, 1, alignment=Qt.AlignmentFlag.AlignTop)

        self.mouse_ID_m2 = QLineEdit('')
        self.generalLayout.addWidget(QLabel('mouse_ID_m2:'), 22, 0, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.mouse_ID_m2, 22, 1, alignment=Qt.AlignmentFlag.AlignTop)

        self.genotype_m2 = QLineEdit('CD1-WT')
        self.generalLayout.addWidget(QLabel('genotype_m2:'), 23, 0, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.genotype_m2, 23, 1, alignment=Qt.AlignmentFlag.AlignTop)

        self.sex_m2 = QLineEdit('')
        self.generalLayout.addWidget(QLabel('sex_m2:'), 24, 0, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.sex_m2, 24, 1, alignment=Qt.AlignmentFlag.AlignTop)

        self.dob_m2 = QLineEdit('')
        self.generalLayout.addWidget(QLabel('DOB_m2:'), 25, 0, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.dob_m2, 25, 1, alignment=Qt.AlignmentFlag.AlignTop)

        self.housing_m2 = QLineEdit('group')
        self.generalLayout.addWidget(QLabel('housing_m2:'), 26, 0, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.housing_m2, 26, 1, alignment=Qt.AlignmentFlag.AlignTop)

        self.other = QLineEdit('')
        self.generalLayout.addWidget(QLabel('other information:'), 27, 0, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.other, 27, 1, alignment=Qt.AlignmentFlag.AlignTop)

        self.generalLayout.addWidget(QLabel('         '), 5, 6, alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.pcs_label = QLabel('Particular camera settings')
        self.pcs_label.setStyleSheet('QLabel { font-weight: bold;}')
        self.generalLayout.addWidget(self.pcs_label,
                                     0, 7, alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        camera_screen_positions = [(2, 3, 4), (6, 7, 8), (10, 11, 12), (14, 15, 16), (18, 19, 20)]
        for cam_idx, cam in enumerate(camera_ids_global):
            self._create_sliders_general(camera_id=cam, camera_color=camera_colors_global[cam_idx], x_pos_tuple=camera_screen_positions[cam_idx])

        self._create_buttons_record(seq=2)

    def record_four(self):
        self.ConductRecording = ConductRecording(self)
        self.generalLayout = QGridLayout()
        self.ConductRecording.setLayout(self.generalLayout)
        self.setWindowTitle(f'{app_name} (Conduct recording)')
        self.setCentralWidget(self.ConductRecording)

        self.txt_edit = QPlainTextEdit()
        self.txt_edit.setReadOnly(True)
        self.generalLayout.addWidget(self.txt_edit, 0, 0,
                                     alignment=Qt.AlignmentFlag.AlignTop)

        self.experiment_progress_bar = QProgressBar()
        self.generalLayout.addWidget(self.experiment_progress_bar, 1, 0,
                                     alignment=Qt.AlignmentFlag.AlignTop)
        self.experiment_progress_bar.setStyleSheet('QProgressBar { min-width: 1030px; min-height: 30px;}')

        self._save_modified_values_to_toml()

        exp_settings_dict_final = toml.load(f"{self.settings_dict['general']['config_settings_directory']}{os.sep}behavioral_experiments_settings.toml")
        self.run_exp = ExperimentController(message_output=self._message,
                                            email_receivers=self.email_recipients,
                                            exp_settings_dict=exp_settings_dict_final)

        self._create_buttons_record(seq=3)

    def process_one(self):
        self.ProcessSettings = ProcessSettings(self)
        self.generalLayout = QGridLayout()
        self.ProcessSettings.setLayout(self.generalLayout)
        self.setWindowTitle(f'{app_name} (Process recordings > Settings)')
        self.setCentralWidget(self.ProcessSettings)

        # select all directories for processing
        self.processing_dir_edit = QTextEdit('')
        self.processing_dir_edit.setFixedSize(350, 380)
        self.generalLayout.addWidget(QLabel('(*) root directories for processing'), 0, 0, alignment=Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.processing_dir_edit, 1, 0, 17, 1, alignment=Qt.AlignmentFlag.AlignTop)

        settings_dir_btn = QPushButton('Browse')
        settings_dir_btn.setStyleSheet('QPushButton { min-width: 42px; min-height: 9px; max-width: 42px; max-height: 9px;}')
        self.settings_dict['general']['config_settings_directory'] = f'{config_dir_global}'
        self.processing_input_dict['send_email']['Messenger']['toml_file_loc'] = f'{config_dir_global}'
        settings_dir_btn.clicked.connect(self._open_settings_dialog)
        self.dir_settings_edit = QLineEdit(config_dir_global)
        self.dir_settings_edit.setStyleSheet('QLineEdit { min-width: 200px; max-width: 200px; min-height: 18px; max-height: 18px; }')

        self.generalLayout.addWidget(QLabel('config loc:'), 19, 0, alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.dir_settings_edit, 19, 0, alignment=Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom)
        self.generalLayout.addWidget(settings_dir_btn, 19, 0, alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop)

        # set other parameters
        self.pc_usage_process = QLineEdit(f'{email_list_global}')
        self.pc_usage_process.setStyleSheet('QLineEdit { min-width: 200px; min-height: 22px; max-height: 22px; }')
        self.generalLayout.addWidget(QLabel('notify PC usage:'), 20, 0, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.pc_usage_process, 20, 0, alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop)

        self.plot_sync_results_cb = QComboBox()
        self.plot_sync_results_cb.setStyleSheet('QComboBox { min-width: 80px; min-height: 20px; max-height: 20px; }')
        self.plot_sync_results_cb.addItems(['Yes', 'No'])
        self.plot_sync_results_cb.activated.connect(partial(self._combo_box_prior_true, variable_id='plot_sync_results_cb_bool'))
        self.generalLayout.addWidget(QLabel('plot sync results:'), 21, 0, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.plot_sync_results_cb, 21, 0, alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop)

        self.phidget_settings_label = QLabel('Phidget processing settings')
        self.phidget_settings_label.setStyleSheet('QLabel { font-weight: bold; }')
        self.generalLayout.addWidget(self.phidget_settings_label,
                                     23, 0, alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        self.process_phidget_cb = QComboBox()
        self.process_phidget_cb.setStyleSheet('QComboBox { min-width: 80px; min-height: 20px; max-height: 20px; }')
        self.process_phidget_cb.addItems(['Yes', 'No'])
        self.process_phidget_cb.activated.connect(partial(self._combo_box_prior_true, variable_id='process_phidget_cb_bool'))
        self.generalLayout.addWidget(QLabel('process phidget data:'), 24, 0, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.process_phidget_cb, 24, 0, alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop)

        self.phidget_extra_data_camera = QLineEdit('22085397')
        self.phidget_extra_data_camera.setStyleSheet('QLineEdit { min-width: 200px; min-height: 22px; max-height: 22px; }')
        self.generalLayout.addWidget(QLabel('extra data cam:'), 25, 0, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.phidget_extra_data_camera, 25, 0, alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop)

        self.phidget_sorting_key = QLineEdit('sensor_time')
        self.phidget_sorting_key.setStyleSheet('QLineEdit { min-width: 200px; min-height: 22px; max-height: 22px; }')
        self.generalLayout.addWidget(QLabel('sorting key:'), 26, 0, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.phidget_sorting_key, 26, 0, alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop)

        self.gvs_label = QLabel('Video processing settings')
        self.gvs_label.setStyleSheet('QLabel { font-weight: bold; }')
        self.generalLayout.addWidget(self.gvs_label,
                                     28, 0, alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        self.vs_conc_label = QLabel('Concatenation')
        self.vs_conc_label.setStyleSheet('QLabel { font-weight: bold; }')
        self.generalLayout.addWidget(self.vs_conc_label,
                                     29, 0, alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        self.conduct_video_concatenation_cb = QComboBox()
        self.conduct_video_concatenation_cb.setStyleSheet('QComboBox { min-width: 80px; min-height: 20px; max-height: 20px; }')
        self.conduct_video_concatenation_cb.addItems(['Yes', 'No'])
        self.conduct_video_concatenation_cb.activated.connect(partial(self._combo_box_prior_true, variable_id='conduct_video_concatenation_cb_bool'))
        self.generalLayout.addWidget(QLabel('conduct video concatenation:'), 30, 0, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.conduct_video_concatenation_cb, 30, 0, alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop)

        self.concatenate_cam_serial_num = QLineEdit('21241563,21369048,21372315,21372316,22085397')
        self.concatenate_cam_serial_num.setStyleSheet('QLineEdit { min-width: 200px; min-height: 22px; max-height: 22px; }')
        self.generalLayout.addWidget(QLabel('cam serial num:'), 31, 0, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.concatenate_cam_serial_num, 31, 0, alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop)

        self.concatenate_video_ext = QLineEdit('mp4')
        self.concatenate_video_ext.setStyleSheet('QLineEdit { min-width: 200px; min-height: 22px; max-height: 22px; }')
        self.generalLayout.addWidget(QLabel('video extension:'), 32, 0, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.concatenate_video_ext, 32, 0, alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop)

        self.concatenated_video_name = QLineEdit('concatenated_temp')
        self.concatenated_video_name.setStyleSheet('QLineEdit { min-width: 200px; min-height: 22px; max-height: 22px; }')
        self.generalLayout.addWidget(QLabel('con video name:'), 33, 0, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.concatenated_video_name, 33, 0, alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop)

        self.vs_fps_change_label = QLabel('FPS change')
        self.vs_fps_change_label.setStyleSheet('QLabel { font-weight: bold; }')
        self.generalLayout.addWidget(self.vs_fps_change_label,
                                     35, 0, alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        self.conduct_video_fps_change_cb = QComboBox()
        self.conduct_video_fps_change_cb.setStyleSheet('QComboBox { min-width: 80px; min-height: 20px; max-height: 20px; }')
        self.conduct_video_fps_change_cb.addItems(['Yes', 'No'])
        self.conduct_video_fps_change_cb.activated.connect(partial(self._combo_box_prior_true, variable_id='conduct_video_fps_change_cb_bool'))
        self.generalLayout.addWidget(QLabel('conduct video fps change:'), 36, 0, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.conduct_video_fps_change_cb, 36, 0, alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop)

        self.change_fps_cam_serial_num = QLineEdit('21241563,21369048,21372315,21372316,22085397')
        self.change_fps_cam_serial_num.setStyleSheet('QLineEdit { min-width: 200px; min-height: 22px; max-height: 22px; }')
        self.generalLayout.addWidget(QLabel('cam serial num:'), 37, 0, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.change_fps_cam_serial_num, 37, 0, alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop)

        self.conversion_target_file = QLineEdit('concatenated_temp')
        self.conversion_target_file.setStyleSheet('QLineEdit { min-width: 200px; min-height: 22px; max-height: 22px; }')
        self.generalLayout.addWidget(QLabel('con target file:'), 38, 0, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.conversion_target_file, 38, 0, alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop)

        self.conversion_vid_ext = QLineEdit('mp4')
        self.conversion_vid_ext.setStyleSheet('QLineEdit { min-width: 200px; min-height: 22px; max-height: 22px; }')
        self.generalLayout.addWidget(QLabel('video extension:'), 39, 0, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.conversion_vid_ext, 39, 0, alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop)

        self.calibration_fps = QLineEdit('10')
        self.calibration_fps.setStyleSheet('QLineEdit { min-width: 200px; min-height: 22px; max-height: 22px; }')
        self.generalLayout.addWidget(QLabel('calibration (fps):'), 40, 0, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.calibration_fps, 40, 0, alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop)

        self.recording_fps = QLineEdit('150')
        self.recording_fps.setStyleSheet('QLineEdit { min-width: 200px; min-height: 22px; max-height: 22px; }')
        self.generalLayout.addWidget(QLabel('recording (fps):'), 41, 0, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.recording_fps, 41, 0, alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop)

        self.delete_con_file_cb = QComboBox()
        self.delete_con_file_cb.setStyleSheet('QComboBox { min-width: 80px; min-height: 20px; max-height: 20px; }')
        self.delete_con_file_cb.addItems(['Yes', 'No'])
        self.delete_con_file_cb.activated.connect(partial(self._combo_box_prior_true, variable_id='delete_con_file_cb_bool'))
        self.generalLayout.addWidget(QLabel('delete concatenated files:'), 42, 0, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.delete_con_file_cb, 42, 0, alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop)

        self.generalLayout.addWidget(QLabel("  "), 45, 2, alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.gas_label = QLabel('Audio processing settings')
        self.gas_label.setStyleSheet('QLabel { font-weight: bold; }')
        self.generalLayout.addWidget(self.gas_label,
                                     0, 3, alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        self.conduct_multichannel_conversion_cb = QComboBox()
        self.conduct_multichannel_conversion_cb.setStyleSheet('QComboBox { min-width: 80px; min-height: 20px; max-height: 20px; }')
        self.conduct_multichannel_conversion_cb.addItems(['Yes', 'No'])
        self.conduct_multichannel_conversion_cb.activated.connect(partial(self._combo_box_prior_true, variable_id='conduct_multichannel_conversion_cb_bool'))
        self.generalLayout.addWidget(QLabel('multi-ch to single-ch:'), 1, 3, 1, 3, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.conduct_multichannel_conversion_cb, 1, 4, 1, 5, alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        self.crop_wav_label = QLabel('Crop WAV files to video')
        self.crop_wav_label.setStyleSheet('QLabel { font-weight: bold; }')
        self.generalLayout.addWidget(self.crop_wav_label,
                                     3, 3, alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        self.crop_wav_cam_cb = QComboBox()
        self.crop_wav_cam_cb.setStyleSheet('QComboBox { min-width: 80px; min-height: 20px; max-height: 20px; }')
        self.crop_wav_cam_cb.addItems(['Yes', 'No'])
        self.crop_wav_cam_cb.activated.connect(partial(self._combo_box_prior_true, variable_id='crop_wav_cam_cb_bool'))
        self.generalLayout.addWidget(QLabel('crop .wav files to match video:'), 4, 3, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.crop_wav_cam_cb, 4, 4, 4, 5, alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        self.crop_wav_cam_serial_num = QLineEdit('21241563,21369048,21372315,21372316,22085397')
        self.crop_wav_cam_serial_num.setStyleSheet('QLineEdit { min-width: 200px; min-height: 22px; max-height: 22px; }')
        self.generalLayout.addWidget(QLabel('used camera serial numbers:'), 5, 3, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.crop_wav_cam_serial_num, 5, 4, 5, 5, alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        self.ch_receiving_input = QLineEdit('1')
        self.ch_receiving_input.setStyleSheet('QLineEdit { min-width: 200px; min-height: 22px; max-height: 22px; }')
        self.generalLayout.addWidget(QLabel('channel receiving Motif input:'), 6, 3, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.ch_receiving_input, 6, 4, 6, 5, alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        self.cam_ttl_duration = QLineEdit('0.00011333')
        self.cam_ttl_duration.setStyleSheet('QLineEdit { min-width: 200px; min-height: 22px; max-height: 22px; }')
        self.generalLayout.addWidget(QLabel('camera TTL duration (s):'), 7, 3, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.cam_ttl_duration, 7, 4, 7, 5, alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        self.ttl_proportion_threshold = QLineEdit('1.0')
        self.ttl_proportion_threshold.setStyleSheet('QLineEdit { min-width: 200px; min-height: 22px; max-height: 22px; }')
        self.generalLayout.addWidget(QLabel('TTL proportion threshold:'), 8, 3, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.ttl_proportion_threshold, 8, 4, 8, 5, alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        self.conc_audio_label = QLabel('Concatenate audio files')
        self.conc_audio_label.setStyleSheet('QLabel { font-weight: bold; }')
        self.generalLayout.addWidget(self.conc_audio_label,
                                     10, 3, alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        self.conc_audio_cb = QComboBox()
        self.conc_audio_cb.setStyleSheet('QComboBox { min-width: 80px; min-height: 20px; max-height: 20px; }')
        self.conc_audio_cb.addItems(['Yes', 'No'])
        self.conc_audio_cb.activated.connect(partial(self._combo_box_prior_true, variable_id='conc_audio_cb_bool'))
        self.generalLayout.addWidget(QLabel('concatenate audio memmap:'), 11, 3, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.conc_audio_cb, 11, 4, 11, 5, alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        self.audio_format = QLineEdit('wav')
        self.audio_format.setStyleSheet('QLineEdit { min-width: 200px; min-height: 22px; max-height: 22px; }')
        self.generalLayout.addWidget(QLabel('audio file format:'), 12, 3, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.audio_format, 12, 4, 12, 5, alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        self.concat_type = QLineEdit('vstack')
        self.concat_type.setStyleSheet('QLineEdit { min-width: 200px; min-height: 22px; max-height: 22px; }')
        self.generalLayout.addWidget(QLabel('concatenation type:'), 13, 3, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.concat_type, 13, 4, 13, 5, alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        self.audio_filter_label = QLabel('Band-pass filter audio files')
        self.audio_filter_label.setStyleSheet('QLabel { font-weight: bold; }')
        self.generalLayout.addWidget(self.audio_filter_label,
                                     15, 3, alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        self.filter_audio_cb = QComboBox()
        self.filter_audio_cb.setStyleSheet('QComboBox { min-width: 80px; min-height: 20px; max-height: 20px; }')
        self.filter_audio_cb.addItems(['Yes', 'No'])
        self.filter_audio_cb.activated.connect(partial(self._combo_box_prior_true, variable_id='filter_audio_cb_bool'))
        self.generalLayout.addWidget(QLabel('filter cropped .wav:'), 16, 3, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.filter_audio_cb, 16, 4, 16, 5, alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        self.audio_filter_format = QLineEdit('wav')
        self.audio_filter_format.setStyleSheet('QLineEdit { min-width: 200px; min-height: 22px; max-height: 22px; }')
        self.generalLayout.addWidget(QLabel('audio file format:'), 17, 3, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.audio_filter_format, 17, 4, 17, 5, alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        self.freq_hp = QLineEdit('2000')
        self.freq_hp.setStyleSheet('QLineEdit { min-width: 200px; min-height: 22px; max-height: 22px; }')
        self.generalLayout.addWidget(QLabel('top feq cutoff (Hz):'), 18, 3, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.freq_hp, 18, 4, 18, 5, alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        self.freq_lp = QLineEdit('0')
        self.freq_lp.setStyleSheet('QLineEdit { min-width: 200px; min-height: 22px; max-height: 22px; }')
        self.generalLayout.addWidget(QLabel('bottom freq cutoff (Hz):'), 19, 3, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.freq_lp, 19, 4, 19, 5, alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        self.av_sync_label = QLabel('Check sync btw a/v files')
        self.av_sync_label.setStyleSheet('QLabel { font-weight: bold; }')
        self.generalLayout.addWidget(self.av_sync_label,
                                     21, 3, alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        self.conduct_sync_cb = QComboBox()
        self.conduct_sync_cb.setStyleSheet('QComboBox { min-width: 80px; min-height: 20px; max-height: 20px; }')
        self.conduct_sync_cb.addItems(['Yes', 'No'])
        self.conduct_sync_cb.activated.connect(partial(self._combo_box_prior_true, variable_id='conduct_sync_cb_bool'))
        self.generalLayout.addWidget(QLabel('conduct a/v sync check:'), 22, 3, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.conduct_sync_cb, 22, 4, 22, 5, alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        self.a_sync_label = QLabel('Find audio sync trains')
        self.a_sync_label.setStyleSheet('QLabel { font-weight: bold; }')
        self.generalLayout.addWidget(self.a_sync_label,
                                     24, 3, alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        self.a_ch_receiving_input = QLineEdit('2')
        self.a_ch_receiving_input.setStyleSheet('QLineEdit { min-width: 200px; min-height: 22px; max-height: 22px; }')
        self.generalLayout.addWidget(QLabel('channel receiving sync input:'), 25, 3, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.a_ch_receiving_input, 25, 4, 25, 5, alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        self.a_sync_pulse_duration = QLineEdit('0.25')
        self.a_sync_pulse_duration.setStyleSheet('QLineEdit { min-width: 200px; min-height: 22px; max-height: 22px; }')
        self.generalLayout.addWidget(QLabel('sync pulse duration (s):'), 26, 3, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.a_sync_pulse_duration, 26, 4, 26, 5, alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        self.a_break_proportion_threshold = QLineEdit('0.001')
        self.a_break_proportion_threshold.setStyleSheet('QLineEdit { min-width: 200px; min-height: 22px; max-height: 22px; }')
        self.generalLayout.addWidget(QLabel('break proportion threshold:'), 27, 3, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.a_break_proportion_threshold, 27, 4, 27, 5, alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        self.a_ttl_proportion_threshold = QLineEdit('0.4')
        self.a_ttl_proportion_threshold.setStyleSheet('QLineEdit { min-width: 200px; min-height: 22px; max-height: 22px; }')
        self.generalLayout.addWidget(QLabel('TTL proportion threshold:'), 28, 3, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.a_ttl_proportion_threshold, 28, 4, 28, 5, alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        self.v_sync_label = QLabel('Find video sync trains')
        self.v_sync_label.setStyleSheet('QLabel { font-weight: bold; }')
        self.generalLayout.addWidget(self.v_sync_label,
                                     30, 3, alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        self.v_camera_serial_num = QLineEdit('21372315')
        self.v_camera_serial_num.setStyleSheet('QLineEdit { min-width: 200px; min-height: 22px; max-height: 22px; }')
        self.generalLayout.addWidget(QLabel('sync cam serial num:'), 31, 3, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.v_camera_serial_num, 31, 4, 31, 5, alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        self.v_led_px_version = QLineEdit('current')
        self.v_led_px_version.setStyleSheet('QLineEdit { min-width: 200px; min-height: 22px; max-height: 22px; }')
        self.generalLayout.addWidget(QLabel('LED px version:'), 32, 3, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.v_led_px_version, 32, 4, 32, 5, alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        self.v_led_px_dev = QLineEdit('10')
        self.v_led_px_dev.setStyleSheet('QLineEdit { min-width: 200px; min-height: 22px; max-height: 22px; }')
        self.generalLayout.addWidget(QLabel('LED deviation (px):'), 33, 3, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.v_led_px_dev, 33, 4, 33, 5, alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        self.v_video_extension = QLineEdit('mp4')
        self.v_video_extension.setStyleSheet('QLineEdit { min-width: 200px; min-height: 22px; max-height: 22px; }')
        self.generalLayout.addWidget(QLabel('video extension:'), 34, 3, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.v_video_extension, 34, 4, 34, 5, alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        self.v_mm_dtype = QLineEdit('np.uint8')
        self.v_mm_dtype.setStyleSheet('QLineEdit { min-width: 200px; min-height: 22px; max-height: 22px; }')
        self.generalLayout.addWidget(QLabel('memmap dtype:'), 35, 3, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.v_mm_dtype, 35, 4, 35, 5, alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        self.v_relative_intensity_threshold = QLineEdit('0.35')
        self.v_relative_intensity_threshold.setStyleSheet('QLineEdit { min-width: 200px; min-height: 22px; max-height: 22px; }')
        self.generalLayout.addWidget(QLabel('rel intensity threshold:'), 36, 3, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.v_relative_intensity_threshold, 36, 4, 36, 5, alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        self.v_camera_fps = QLineEdit('150')
        self.v_camera_fps.setStyleSheet('QLineEdit { min-width: 200px; min-height: 22px; max-height: 22px; }')
        self.generalLayout.addWidget(QLabel('camera fr (fps):'), 37, 3, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.v_camera_fps, 37, 4, 37, 5, alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        self.v_sync_pulse_duration = QLineEdit('0.25')
        self.v_sync_pulse_duration.setStyleSheet('QLineEdit { min-width: 200px; min-height: 22px; max-height: 22px; }')
        self.generalLayout.addWidget(QLabel('sync pulse duration (s):'), 38, 3, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.v_sync_pulse_duration, 38, 4, 38, 5, alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        self.v_millisecond_divergence_tolerance = QLineEdit('10')
        self.v_millisecond_divergence_tolerance.setStyleSheet('QLineEdit { min-width: 200px; min-height: 22px; max-height: 22px; }')
        self.generalLayout.addWidget(QLabel('divergence tolerance (ms):'), 39, 3, alignment=Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.v_millisecond_divergence_tolerance, 39, 4, 39, 5, alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        self._create_buttons_process(seq=0)

    def process_two(self):
        self.ConductProcess = ConductProcess(self)
        self.generalLayout = QGridLayout()
        self.ConductProcess.setLayout(self.generalLayout)
        self.setWindowTitle(f'{app_name} (Conduct Processing)')
        self.setCentralWidget(self.ConductProcess)

        self.txt_edit_process = QPlainTextEdit()
        self.txt_edit_process.setReadOnly(True)
        self.generalLayout.addWidget(self.txt_edit_process, 0, 0,
                                     alignment=Qt.AlignmentFlag.AlignTop)

        exp_settings_dict_final = toml.load(f"{self.processing_input_dict['send_email']['Messenger']['toml_file_loc']}{os.sep}behavioral_experiments_settings.toml")

        self.run_processing = Stylist(message_output=self._process_message,
                                      input_parameter_dict=self.processing_input_dict,
                                      root_directories=self.processing_input_dict['preprocess_data']['root_directories'],
                                      exp_settings_dict=exp_settings_dict_final)

        self._create_buttons_process(seq=1)

    def _save_modified_values_to_toml(self):
        self.exp_settings_dict = toml.load(f"{self.settings_dict['general']['config_settings_directory']}{os.sep}behavioral_experiments_settings.toml")

        audio_config_temp = configparser.ConfigParser()
        audio_config_temp.read(f"{self.settings_dict['general']['config_settings_directory']}"
                               f"{os.sep}avisoft_config.ini")

        if audio_config_temp['Configuration']['basedirectory'] != self.settings_dict['general']['avisoft_basedirectory'] \
                or audio_config_temp['Configuration']['configfilename'] != f"{self.settings_dict['general']['avisoft_basedirectory']}Configurations{os.sep}RECORDER_USGH{os.sep}avisoft_config.ini":
            self.modify_audio_config = True

        if self.exp_settings_dict['config_settings_directory'] != self.settings_dict['general']['config_settings_directory']:
            self.exp_settings_dict['config_settings_directory'] = self.settings_dict['general']['config_settings_directory']

        if self.exp_settings_dict['avisoft_recorder_exe'] != self.settings_dict['general']['avisoft_recorder_exe']:
            self.exp_settings_dict['avisoft_recorder_exe'] = self.settings_dict['general']['avisoft_recorder_exe']

        if self.exp_settings_dict['avisoft_basedirectory'] != self.settings_dict['general']['avisoft_basedirectory']:
            self.exp_settings_dict['avisoft_basedirectory'] = self.settings_dict['general']['avisoft_basedirectory']
            self.modify_audio_config = True

        if self.exp_settings_dict['coolterm_basedirectory'] != self.settings_dict['general']['coolterm_basedirectory']:
            self.exp_settings_dict['coolterm_basedirectory'] = self.settings_dict['general']['coolterm_basedirectory']

        if self.exp_settings_dict['recording_files_destination_linux'] != self.settings_dict['general']['recording_files_destination_linux']:
            self.exp_settings_dict['recording_files_destination_linux'] = self.settings_dict['general']['recording_files_destination_linux']

        if self.exp_settings_dict['recording_files_destination_win'] != self.settings_dict['general']['recording_files_destination_win']:
            self.exp_settings_dict['recording_files_destination_win'] = self.settings_dict['general']['recording_files_destination_win']

        if self.exp_settings_dict['conduct_tracking_calibration'] != self.settings_dict['general']['conduct_tracking_calibration']:
            self.exp_settings_dict['conduct_tracking_calibration'] = self.settings_dict['general']['conduct_tracking_calibration']

        if self.exp_settings_dict['calibration_duration'] != ast.literal_eval(self.settings_dict['general']['calibration_duration']):
            self.exp_settings_dict['calibration_duration'] = ast.literal_eval(self.settings_dict['general']['calibration_duration'])

        if self.exp_settings_dict['conduct_audio_recording'] != self.settings_dict['general']['conduct_audio_recording']:
            self.exp_settings_dict['conduct_audio_recording'] = self.settings_dict['general']['conduct_audio_recording']

        if self.exp_settings_dict['video_session_duration'] != ast.literal_eval(self.settings_dict['general']['video_session_duration']):
            self.exp_settings_dict['video_session_duration'] = ast.literal_eval(self.settings_dict['general']['video_session_duration'])

        if self.settings_dict['general']['conduct_audio_recording']:
            self.experiment_time_sec += ((ast.literal_eval(self.settings_dict['general']['video_session_duration']) + 0.36) * 60)
        else:
            self.experiment_time_sec += ((ast.literal_eval(self.settings_dict['general']['video_session_duration']) + 0.26) * 60)

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

        if self.exp_settings_dict['modify_audio_config_file'] != self.modify_audio_config:
            self.exp_settings_dict['modify_audio_config_file'] = self.modify_audio_config

        for video_key in self.settings_dict['video'].keys():
            if video_key in self.exp_settings_dict['video']['general'].keys():
                if video_key in ['browser', 'expected_cameras', 'recording_codec', 'specific_camera_serial', 'monitor_recording',
                                 'monitor_specific_camera', 'delete_post_copy']:
                    if self.exp_settings_dict['video']['general'][video_key] != self.settings_dict['video'][video_key]:
                        self.exp_settings_dict['video']['general'][video_key] = self.settings_dict['video'][video_key]
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

        self._message(f"Updating the configuration .toml file completed at: "
                      f"{datetime.datetime.now().hour:02d}:{datetime.datetime.now().minute:02d}.{datetime.datetime.now().second:02d}.")

    def _save_process_labels_func(self):
        qlabel_strings = ['concatenate_video_ext', 'concatenated_video_name', 'conversion_target_file', 'conversion_vid_ext', 'calibration_fps', 'recording_fps',
                          'ch_receiving_input', 'cam_ttl_duration', 'ttl_proportion_threshold', 'audio_filter_format', 'freq_hp', 'freq_lp',
                          'a_ch_receiving_input', 'a_sync_pulse_duration', 'a_break_proportion_threshold', 'a_ttl_proportion_threshold',
                          'pc_usage_process', 'v_millisecond_divergence_tolerance', 'v_sync_pulse_duration', 'v_camera_fps',
                          'v_relative_intensity_threshold', 'v_mm_dtype', 'v_video_extension', 'v_led_px_dev', 'v_led_px_version',
                          'phidget_extra_data_camera', 'phidget_sorting_key', 'audio_format', 'concat_type']
        lists_in_string = ['concatenate_cam_serial_num', 'change_fps_cam_serial_num', 'crop_wav_cam_serial_num', 'v_camera_serial_num']

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

        self.processing_input_dict['file_manipulation']['Operator']['concatenate_video_files']['video_extension'] = self.concatenate_video_ext
        self.processing_input_dict['file_manipulation']['Operator']['concatenate_video_files']['concatenated_video_name'] = self.concatenated_video_name
        self.processing_input_dict['file_manipulation']['Operator']['rectify_video_fps']['conversion_target_file'] = self.conversion_target_file
        self.processing_input_dict['file_manipulation']['Operator']['rectify_video_fps']['video_extension'] = self.conversion_vid_ext
        self.processing_input_dict['file_manipulation']['Operator']['rectify_video_fps']['calibration_fps'] = int(round(ast.literal_eval(self.calibration_fps)))
        self.processing_input_dict['file_manipulation']['Operator']['rectify_video_fps']['recording_fps'] = int(round(ast.literal_eval(self.recording_fps)))
        self.processing_input_dict['synchronize_files']['Synchronizer']['crop_wav_files_to_video']['ch_receiving_input'] = int(ast.literal_eval(self.ch_receiving_input))
        self.processing_input_dict['synchronize_files']['Synchronizer']['crop_wav_files_to_video']['ttl_pulse_duration'] = float(ast.literal_eval(self.cam_ttl_duration))
        self.processing_input_dict['synchronize_files']['Synchronizer']['crop_wav_files_to_video']['ttl_proportion_threshold'] = float(ast.literal_eval(self.ttl_proportion_threshold))
        self.processing_input_dict['file_manipulation']['Operator']['concatenate_audio_files']['audio_format'] = self.audio_format
        self.processing_input_dict['file_manipulation']['Operator']['concatenate_audio_files']['concat_type'] = self.concat_type
        self.processing_input_dict['file_manipulation']['Operator']['filter_audio_files']['audio_format'] = self.audio_filter_format
        self.processing_input_dict['file_manipulation']['Operator']['filter_audio_files']['freq_hp'] = int(ast.literal_eval(self.freq_hp))
        self.processing_input_dict['file_manipulation']['Operator']['filter_audio_files']['freq_lp'] = int(ast.literal_eval(self.freq_lp))
        self.processing_input_dict['synchronize_files']['Synchronizer']['find_audio_sync_trains']['ch_receiving_input'] = int(ast.literal_eval(self.a_ch_receiving_input))
        self.processing_input_dict['synchronize_files']['Synchronizer']['find_audio_sync_trains']['sync_pulse_duration'] = float(ast.literal_eval(self.a_sync_pulse_duration))
        self.processing_input_dict['synchronize_files']['Synchronizer']['find_audio_sync_trains']['break_proportion_threshold'] = float(ast.literal_eval(self.a_break_proportion_threshold))
        self.processing_input_dict['synchronize_files']['Synchronizer']['find_audio_sync_trains']['ttl_proportion_threshold'] = float(ast.literal_eval(self.a_ttl_proportion_threshold))
        self.processing_input_dict['synchronize_files']['Synchronizer']['find_video_sync_trains']['led_px_version'] = self.v_led_px_version
        self.processing_input_dict['synchronize_files']['Synchronizer']['find_video_sync_trains']['led_px_dev'] = int(ast.literal_eval(self.v_led_px_dev))
        self.processing_input_dict['synchronize_files']['Synchronizer']['find_video_sync_trains']['video_extension'] = self.v_video_extension
        self.processing_input_dict['synchronize_files']['Synchronizer']['find_video_sync_trains']['mm_dtype'] = self.v_mm_dtype
        self.processing_input_dict['synchronize_files']['Synchronizer']['find_video_sync_trains']['relative_intensity_threshold'] = float(ast.literal_eval(self.v_relative_intensity_threshold))
        self.processing_input_dict['synchronize_files']['Synchronizer']['find_video_sync_trains']['camera_fps'] = int(round(ast.literal_eval(self.v_camera_fps)))
        self.processing_input_dict['synchronize_files']['Synchronizer']['find_video_sync_trains']['sync_pulse_duration'] = float(ast.literal_eval(self.v_sync_pulse_duration))
        self.processing_input_dict['synchronize_files']['Synchronizer']['find_video_sync_trains']['millisecond_divergence_tolerance'] = int(ast.literal_eval(self.v_millisecond_divergence_tolerance))
        self.processing_input_dict['extract_phidget_data']['Gatherer']['prepare_data_for_analyses']['extra_data_camera'] = self.phidget_extra_data_camera
        self.processing_input_dict['extract_phidget_data']['Gatherer']['prepare_data_for_analyses']['sorting_key'] = self.phidget_sorting_key

        self.processing_input_dict['preprocess_data']['root_directories'] = self.processing_dir_edit
        self.processing_input_dict['file_manipulation']['Operator']['concatenate_video_files']['camera_serial_num'] = self.concatenate_cam_serial_num
        self.processing_input_dict['file_manipulation']['Operator']['rectify_video_fps']['camera_serial_num'] = self.change_fps_cam_serial_num
        self.processing_input_dict['synchronize_files']['Synchronizer']['crop_wav_files_to_video']['camera_serial_num'] = self.crop_wav_cam_serial_num
        self.processing_input_dict['send_email']['Messenger']['send_message']['receivers'] = self.pc_usage_process
        self.processing_input_dict['synchronize_files']['Synchronizer']['find_video_sync_trains']['camera_serial_num'] = self.v_camera_serial_num

        self.processing_input_dict['processing_booleans']['conduct_video_concatenation'] = self.conduct_video_concatenation_cb_bool
        self.conduct_video_concatenation_cb_bool = True
        self.processing_input_dict['processing_booleans']['conduct_video_fps_change'] = self.conduct_video_fps_change_cb_bool
        self.conduct_video_fps_change_cb_bool = True
        self.processing_input_dict['file_manipulation']['Operator']['rectify_video_fps']['delete_old_file'] = self.delete_con_file_cb_bool
        self.delete_con_file_cb_bool = True
        self.processing_input_dict['processing_booleans']['conduct_audio_multichannel_to_single_ch'] = self.conduct_multichannel_conversion_cb_bool
        self.conduct_multichannel_conversion_cb_bool = True
        self.processing_input_dict['processing_booleans']['conduct_audio_cropping'] = self.crop_wav_cam_cb_bool
        self.crop_wav_cam_cb_bool = True
        self.processing_input_dict['processing_booleans']['conduct_audio_to_mmap'] = self.conc_audio_cb_bool
        self.conc_audio_cb_bool = True
        self.processing_input_dict['processing_booleans']['conduct_audio_filtering'] = self.filter_audio_cb_bool
        self.filter_audio_cb_bool = True
        self.processing_input_dict['processing_booleans']['conduct_audio_video_sync'] = self.conduct_sync_cb_bool
        self.conduct_sync_cb_bool = True
        self.processing_input_dict['processing_booleans']['plot_sync_data'] = self.plot_sync_results_cb_bool
        self.plot_sync_results_cb_bool = True
        self.processing_input_dict['processing_booleans']['conduct_phidget_data_extraction'] = self.process_phidget_cb_bool
        self.process_phidget_cb_bool = True

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

    def _save_record_two_labels_func(self):
        for variable in self.first_col_labels_1 + self.first_col_labels_2 + self.second_col_labels:
            self.settings_dict['audio'][f'{variable[:-1]}'] = getattr(self, variable[:-1]).text()

    def _save_record_three_labels_func(self):
        video_dict_keys = ['browser', 'expected_cameras', 'recording_codec', 'specific_camera_serial',
                           'experimenter', 'mice_num', 'cage_ID_m1', 'mouse_ID_m1', 'genotype_m1', 'sex_m1', 'dob_m1',
                           'housing_m1', 'cage_ID_m2', 'mouse_ID_m2', 'genotype_m2', 'sex_m2', 'dob_m2', 'housing_m2', 'other']

        self.settings_dict['video']['monitor_recording'] = self.monitor_recording_cb_bool
        self.monitor_recording_cb_bool = True
        self.settings_dict['video']['monitor_specific_camera'] = self.monitor_specific_camera_cb_bool
        self.monitor_specific_camera_cb_bool = False
        self.settings_dict['video']['delete_post_copy'] = self.delete_post_copy_cb_bool
        self.delete_post_copy_cb_bool = True

        self.settings_dict['video']['recording_frame_rate'] = self.cameras_frame_rate.value()
        self.settings_dict['video']['calibration_frame_rate'] = self.calibration_frame_rate.value()

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
            if variable != 'expected_cameras':
                self.settings_dict['video'][variable] = getattr(self, variable).text()
            else:
                self.expected_cameras = self.expected_cameras.text()
                self.settings_dict['video'][variable] = self.expected_cameras.split(',')

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
        self.fr_label.setText(f'recording ({str(value)} fps):')

    def _update_cal_fr_label(self, value):
        self.cal_fr_label.setText(f'calibration ({str(value)} fps):')

    def _create_sliders_general(self, camera_id=None, camera_color=None, x_pos_tuple=None):
        self.generalLayout.addWidget(QLabel(f'Camera {camera_id} ({camera_color}) settings'),
                                     x_pos_tuple[0], 7, x_pos_tuple[0], 9,
                                     alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        self.__dict__[f'exposure_time_{camera_id}'] = QSlider(Qt.Orientation.Horizontal)
        self.__dict__[f'exposure_time_{camera_id}'].setRange(500, 30000)
        self.__dict__[f'exposure_time_{camera_id}'].setValue(2500)
        self.__dict__[f'exposure_time_{camera_id}'].valueChanged.connect(partial(self._update_exposure_time_label, variable_id=f'exposure_time_{camera_id}_label'))
        self.__dict__[f'exposure_time_{camera_id}_label'] = QLabel('exp time (2500 μs):')
        self.__dict__[f'exposure_time_{camera_id}_label'].setFixedWidth(150)
        self.__dict__[f'exposure_time_{camera_id}_label'].setAlignment(Qt.AlignmentFlag.AlignLeft |
                                                                       Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.__dict__[f'exposure_time_{camera_id}_label'], x_pos_tuple[1], 7)
        self.generalLayout.addWidget(self.__dict__[f'exposure_time_{camera_id}'], x_pos_tuple[1], 8, alignment=Qt.AlignmentFlag.AlignVCenter)

        self.__dict__[f'gain_{camera_id}'] = QSlider(Qt.Orientation.Horizontal)
        self.__dict__[f'gain_{camera_id}'].setRange(0, 15)
        self.__dict__[f'gain_{camera_id}'].setValue(0)
        self.__dict__[f'gain_{camera_id}'].valueChanged.connect(partial(self._update_gain_label, variable_id=f'gain_{camera_id}_label'))
        self.__dict__[f'gain_{camera_id}_label'] = QLabel('digital gain (0 dB):')
        self.__dict__[f'gain_{camera_id}_label'].setFixedWidth(150)
        self.__dict__[f'gain_{camera_id}_label'].setAlignment(Qt.AlignmentFlag.AlignLeft |
                                                              Qt.AlignmentFlag.AlignTop)
        self.generalLayout.addWidget(self.__dict__[f'gain_{camera_id}_label'], x_pos_tuple[2], 7)
        self.generalLayout.addWidget(self.__dict__[f'gain_{camera_id}'], x_pos_tuple[2], 8, alignment=Qt.AlignmentFlag.AlignVCenter)

    def _create_buttons_main(self):
        self.button_map = {'Process': QPushButton(QIcon(process_icon), 'Process')}
        self.button_map['Process'].clicked.connect(self.process_one)
        self.generalLayout.addWidget(self.button_map['Process'], 57, 2)

        self.button_map['Record'] = QPushButton(QIcon(record_icon), 'Record')
        self.button_map['Record'].clicked.connect(self.record_one)
        self.generalLayout.addWidget(self.button_map['Record'], 57, 1, alignment=Qt.AlignmentFlag.AlignRight)

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

        self.button_map = {'Previous': QPushButton(QIcon(previous_icon), 'Previous')}
        self.button_map['Previous'].clicked.connect(previous_win)
        self.generalLayout.addWidget(self.button_map['Previous'], 98, 0, alignment=Qt.AlignmentFlag.AlignLeft)

        self.button_map['Main'] = QPushButton(QIcon(main_icon), 'Main')
        self.button_map['Main'].clicked.connect(self.main_window)
        self.generalLayout.addWidget(self.button_map['Main'], 98, 71)

        if len(next_win_connect) > 0:
            self.button_map['Next'] = QPushButton(QIcon(next_icon), 'Next')
            for one_connection in next_win_connect:
                self.button_map['Next'].clicked.connect(one_connection)
            self.generalLayout.addWidget(self.button_map['Next'], 98, 58)
        else:
            if self.settings_dict['general']['conduct_tracking_calibration']:
                self.button_map['Calibrate'] = QPushButton(QIcon(calibrate_icon), 'Calibrate')
                self.button_map['Calibrate'].clicked.connect(self._disable_other_buttons)
                self.button_map['Calibrate'].clicked.connect(self._start_calibration)
                self.button_map['Calibrate'].clicked.connect(self._enable_other_buttons_post_cal)
                self.generalLayout.addWidget(self.button_map['Calibrate'], 98, 45)

            self.button_map['Record'] = QPushButton(QIcon(record_icon), 'Record')
            self.button_map['Record'].clicked.connect(self._disable_other_buttons)
            self.button_map['Record'].clicked.connect(self._start_recording)
            self.button_map['Record'].clicked.connect(self._enable_other_buttons_post_rec)
            self.generalLayout.addWidget(self.button_map['Record'], 98, 58)

    def _create_buttons_process(self, seq):
        if seq == 0:
            previous_win = self.main_window
            next_win_connect = [self._save_process_labels_func, self.process_two]
        else:
            previous_win = self.process_one

        self.button_map = {'Previous': QPushButton(QIcon(previous_icon), 'Previous')}
        self.button_map['Previous'].clicked.connect(previous_win)
        self.generalLayout.addWidget(self.button_map['Previous'], 75, 0, alignment=Qt.AlignmentFlag.AlignLeft)

        self.button_map['Main'] = QPushButton(QIcon(main_icon), 'Main')
        self.button_map['Main'].clicked.connect(self.main_window)
        self.generalLayout.addWidget(self.button_map['Main'], 75, 71)

        if seq == 0:
            self.button_map['Next'] = QPushButton(QIcon(next_icon), 'Next')
            for one_connection in next_win_connect:
                self.button_map["Next"].clicked.connect(one_connection)
            self.generalLayout.addWidget(self.button_map['Next'], 75, 58)
        else:
            self.button_map['Process'] = QPushButton(QIcon(process_icon), 'Process')
            self.button_map['Process'].clicked.connect(self._disable_process_buttons)
            self.button_map['Process'].clicked.connect(self._start_processing)
            self.button_map['Process'].clicked.connect(self._enable_process_buttons)
            self.generalLayout.addWidget(self.button_map['Process'], 75, 58)

    def _start_processing(self):
        self.run_processing.prepare_data_for_analyses()

    def _start_calibration(self):
        self.cal_thread = threading.Thread(self.run_exp.conduct_tracking_calibration())
        self.cal_thread.start()

    def _start_recording(self):
        self.prog_bar_thread = threading.Thread(target=self._move_progress_bar)
        self.prog_bar_thread.start()

        self.rec_thread = threading.Thread(self.run_exp.conduct_behavioral_recording())
        self.rec_thread.start()

    def _enable_process_buttons(self):
        self.button_map['Previous'].setEnabled(True)
        self.button_map['Main'].setEnabled(True)
        self.button_map['Process'].setEnabled(True)

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
        if self.settings_dict['general']['conduct_tracking_calibration']:
            self.button_map['Calibrate'].setEnabled(False)

    def _open_settings_dialog(self):
        settings_dir_name = QFileDialog.getExistingDirectory(
            self,
            'Select settings directory',
            f'{config_dir_global}')
        if settings_dir_name:
            settings_dir_name_path = Path(settings_dir_name)
            self.dir_settings_edit.setText(str(settings_dir_name_path))
            if os.name == 'nt':
                self.settings_dict['general']['config_settings_directory'] = str(settings_dir_name_path).replace(os.sep, '\\')
                self.processing_input_dict['send_email']['Messenger']['toml_file_loc'] = str(settings_dir_name_path).replace(os.sep, '\\')
            else:
                self.settings_dict['general']['config_settings_directory'] = str(settings_dir_name_path)
                self.processing_input_dict['send_email']['Messenger']['toml_file_loc'] = str(settings_dir_name_path)
        else:
            self.settings_dict['general']['config_settings_directory'] = f'{config_dir_global}'
            self.processing_input_dict['send_email']['Messenger']['toml_file_loc'] = f'{config_dir_global}'

    def _open_recorder_dialog(self):
        recorder_dir_name = QFileDialog.getExistingDirectory(
            self,
            'Select Avisoft Recorder USGH directory',
            f'{avisoft_rec_dir_global}')
        if recorder_dir_name:
            recorder_dir_name_path = Path(recorder_dir_name)
            self.recorder_settings_edit.setText(str(recorder_dir_name_path))
            if os.name == 'nt':
                self.settings_dict['general']['avisoft_recorder_exe'] = str(recorder_dir_name_path).replace(os.sep, '\\')
            else:
                self.settings_dict['general']['avisoft_recorder_exe'] = str(recorder_dir_name_path)
        else:
            self.settings_dict['general']['avisoft_recorder_exe'] = f'{avisoft_rec_dir_global}'

    def _open_avisoft_dialog(self):
        avisoft_dir_name = QFileDialog.getExistingDirectory(
            self,
            'Select Avisoft base directory',
            f'{avisoft_base_dir_global}')
        if avisoft_dir_name:
            avisoft_dir_name_path = Path(avisoft_dir_name)
            self.avisoft_base_edit.setText(str(avisoft_dir_name_path))
            if os.name == 'nt':
                self.settings_dict['general']['avisoft_basedirectory'] = str(avisoft_dir_name_path).replace(os.sep, '\\') + '\\'
            else:
                self.settings_dict['general']['avisoft_basedirectory'] = str(avisoft_dir_name_path)
        else:
            self.settings_dict['general']['avisoft_basedirectory'] = f'{avisoft_base_dir_global}'

    def _open_coolterm_dialog(self):
        coolterm_dir_name = QFileDialog.getExistingDirectory(
            self,
            'Select Coolterm base directory',
            f'{coolterm_base_dir_global}')
        if coolterm_dir_name:
            coolterm_dir_name_path = Path(coolterm_dir_name)
            self.coolterm_base_edit.setText(str(coolterm_dir_name_path))
            if os.name == 'nt':
                self.settings_dict['general']['coolterm_basedirectory'] = str(coolterm_dir_name_path).replace(os.sep, '\\')
            else:
                self.settings_dict['general']['coolterm_basedirectory'] = str(coolterm_dir_name_path)
        else:
            self.settings_dict['general']['coolterm_basedirectory'] = f'{coolterm_base_dir_global}'

    def _location_on_the_screen(self):
        top_left_point = QGuiApplication.primaryScreen().availableGeometry().topLeft()
        self.move(top_left_point)

    def _move_progress_bar(self):
        time_to_sleep = int(round(self.experiment_time_sec * 10))
        for i in range(101):
            _loop_time(delay_time=time_to_sleep)
            self.experiment_progress_bar.setValue(i)
        self.experiment_time_sec = 0

    def _message(self, s):
        self.txt_edit.appendPlainText(s)

    def _process_message(self, s):
        self.txt_edit_process.appendPlainText(s)


def main():
    usv_playpen_app = QApplication([])

    # usv_playpen_app.setStyle('Fusion')
    with open('gui_style_sheet.css', 'r') as file:
        usv_playpen_app.setStyleSheet(file.read())

    splash = QSplashScreen(QPixmap(splash_icon))
    progress_bar = QProgressBar(splash)
    progress_bar.setGeometry(0, 0, 610, 20)
    splash.show()
    for i in range(0, 101):
        progress_bar.setValue(i)
        t = time.time()
        while time.time() < t + 0.025:
            usv_playpen_app.processEvents()

    initial_values_dict = {'experiment_time_sec': 0, 'monitor_recording_cb_bool': True, 'monitor_specific_camera_cb_bool': False, 'delete_post_copy_cb_bool': True,
                           'conduct_audio_cb_bool': True, 'conduct_tracking_calibration_cb_bool': False, 'modify_audio_config': False, 'conduct_video_concatenation_cb_bool': True,
                           'conduct_video_fps_change_cb_bool': True, 'delete_con_file_cb_bool': True, 'conduct_multichannel_conversion_cb_bool': True, 'crop_wav_cam_cb_bool': True,
                           'conc_audio_cb_bool': True, 'filter_audio_cb_bool': True, 'conduct_sync_cb_bool': True, 'plot_sync_results_cb_bool': True, 'process_phidget_cb_bool': True}

    usv_playpen_window = USVPlaypenWindow(**initial_values_dict)
    splash.finish(usv_playpen_window)
    usv_playpen_window.show()

    sys.exit(usv_playpen_app.exec())


if __name__ == "__main__":
    main()

