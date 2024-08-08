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
from PyQt6.QtTest import QTest
from behavioral_experiments import ExperimentController
from preprocess_data import Stylist

os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"

if os.name == 'nt':
    my_app_id = 'mycompany.myproduct.subproduct.version'
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(my_app_id)

app_name = 'USV Playpen v0.3.8'
experimenter_id = 'bartulem'
cup_directory_name = 'Bartul'
email_list_global = ''
config_dir_global = 'C:\\experiment_running_docs'
avisoft_rec_dir_global = 'C:\\Program Files (x86)\\Avisoft Bioacoustics\\RECORDER USGH'
avisoft_base_dir_global = 'C:\\Users\\bartulem\\Documents\\Avisoft Bioacoustics\\'
coolterm_base_dir_global = 'D:\\CoolTermWin'
destination_linux_global = f'/home/labadmin/falkner/{cup_directory_name}/Data,/home/labadmin/murthy/{cup_directory_name}/Data'
destination_win_global = f'F:\\{cup_directory_name}\\Data,M:\\{cup_directory_name}\\Data'
camera_ids_global = ['21372315', '21372316', '21369048', '22085397', '21241563']
camera_colors_global = ['white', 'orange', 'red', 'cyan', 'yellow']
gui_font_global = 'segoeui.ttf'

basedir = os.path.dirname(__file__)
background_img = f'{basedir}{os.sep}img{os.sep}background_img.png'
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

        font_file_loc = QFontDatabase.addApplicationFont(f'{basedir}{os.sep}fonts{os.sep}{gui_font_global}')
        self.font_id = QFontDatabase.applicationFontFamilies(font_file_loc)[0]

        for attr, value in kwargs.items():
            setattr(self, attr, value)

        self.settings_dict = {'general': {'config_settings_directory': f'{config_dir_global}',
                                          'avisoft_recorder_exe': f'{avisoft_rec_dir_global}',
                                          'avisoft_basedirectory': f'{avisoft_base_dir_global}',
                                          'coolterm_basedirectory': f'{coolterm_base_dir_global}'},
                              'audio': {},
                              'video': {'expected_camera_num': len(camera_ids_global)}}

        self.processing_input_dict = {'processing_booleans': {
            'conduct_video_concatenation': False,
            'conduct_video_fps_change': False,
            'conduct_audio_multichannel_to_single_ch': False,
            'conduct_audio_cropping': False,
            'conduct_audio_to_mmap': False,
            'conduct_audio_filtering': False,
            'conduct_hpss': False,
            'conduct_audio_video_sync': False,
            'conduct_ephys_video_sync': False,
            'conduct_ephys_file_chaining': False,
            'split_cluster_spikes': False,
            'sleap_h5_conversion': False,
            'anipose_calibration': False,
            'anipose_triangulation': False,
            'anipose_trm': False,
            'das_infer': False},
            'anipose_operations': {
                'ConvertTo3D': {
                    'sleap_file_conversion': {
                        'sleap_conda_env_name': 'sleap1.3.3'},
                    'conduct_anipose_calibration': {
                        'board_provided_bool': False,
                        'board_xy': [8, 11],
                        'square_len': 16.25,
                        'marker_len_bits': [12.75, 4],
                        'dict_size': 1000,
                        'img_width_height': [2100, 2970]},
                    'conduct_anipose_triangulation': {
                        'calibration_file_loc': '',
                        'triangulate_arena_points_bool': False,
                        'frame_restriction': None,
                        'excluded_views': [],
                        'display_progress_bool': False,
                        'rigid_body_constraints': [],
                        'weak_body_constraints': [],
                        'scale_smooth_len_weak': [3, 4, 1],
                        'reprojection_error_loss': [5, 'l2'],
                        'n_deriv_smooth': 2},
                    'translate_rotate_metric': {
                        'original_arena_file_loc': '',
                        'save_arena_data_bool': False,
                        'save_mouse_data_bool': True,
                        'delete_original_h5': True,
                        'static_reference_len': 0.615}}},
            'preprocess_data': {
                'root_directories': []},
            'extract_phidget_data': {
                'Gatherer': {
                    'prepare_data_for_analyses': {
                        'extra_data_camera': '22085397'}}},
            'file_loader': {
                'DataLoader': {
                    'wave_data_loc': [''],
                    'load_wavefile_data': {
                        'library': 'scipy',
                        'conditional_arg': []}}},
            'file_manipulation': {
                'Operator': {
                    'get_spike_times': {
                        'min_spike_num': 0,
                        'kilosort_version': '4'},
                    'concatenate_audio_files': {
                        'audio_format': 'wav',
                        'concat_dirs': ['hpss_filtered']},
                    'hpss_audio': {
                        'stft_window_length_hop_size': [512, 128],
                        'kernel_size': (5, 60),
                        'hpss_power': 4.0,
                        'margin': (4, 1)},
                    'filter_audio_files': {
                        'audio_format': 'wav',
                        'filter_dirs': ['hpss'],
                        'filter_freq_bounds': [0, 30000]},
                    'concatenate_video_files': {
                        'camera_serial_num': ['21241563', '21369048', '21372315', '21372316', '22085397'],
                        'video_extension': 'mp4',
                        'concatenated_video_name': 'concatenated_temp'},
                    'rectify_video_fps': {
                        'camera_serial_num': ['21241563', '21369048', '21372315', '21372316', '22085397'],
                        'conversion_target_file': 'concatenated_temp',
                        'video_extension': 'mp4',
                        'constant_rate_factor': 16,
                        'encoding_preset': 'veryfast',
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
                    'toml_file_loc': f'{config_dir_global}',
                    'send_message': {
                        'receivers': []}}},
            'synchronize_files': {
                'Synchronizer': {
                    'validate_ephys_video_sync': {
                        'npx_file_type': 'ap',
                        'npx_ms_divergence_tolerance': 10
                    },
                    'find_audio_sync_trains': {
                        'ch_receiving_input': 2},
                    'find_video_sync_trains': {
                        'camera_serial_num': ['21372315'],
                        'led_px_version': 'current',
                        'led_px_dev': 10,
                        'video_extension': 'mp4',
                        'relative_intensity_threshold': 0.35,
                        'millisecond_divergence_tolerance': 10},
                    'crop_wav_files_to_video': {
                        'device_receiving_input': 'm',
                        'ch_receiving_input': 1}}},
            'usv_inference': {
                'FindMouseVocalizations': {
                    'das_command_line_inference': {
                        'das_conda_env_name': 'das',
                        'model_directory': '',
                        'model_name_base': '',
                        'output_file_type': 'csv',
                        'segment_tmf': [0.5, 0.015, 0.015]}}}}

        self.main_window()

    def main_window(self):
        self.Main = Main(self)
        self.setCentralWidget(self.Main)
        self.setFixedSize(420, 500)
        self._location_on_the_screen()
        self.setWindowFlag(Qt.WindowType.WindowMaximizeButtonHint, False)
        self.setWindowTitle(f'{app_name}')

        self._create_buttons_main()

    def record_one(self):
        self.Record = Record(self)
        self.setWindowTitle(f'{app_name} (Record > Select config directories and set basic parameters)')
        self.setCentralWidget(self.Record)
        record_one_x, record_one_y = (725, 500)
        self.setFixedSize(record_one_x, record_one_y)

        title_label = QLabel('Please select appropriate directories (w/ config files or executables in them)', self.Record)
        title_label.setFont(QFont(self.font_id, 13))
        title_label.setStyleSheet('QLabel { font-weight: bold;}')
        title_label.move(5, 10)

        settings_dir_label = QLabel('Settings file (*.toml) directory:', self.Record)
        settings_dir_label.setFont(QFont(self.font_id, 12))
        settings_dir_label.move(5, 40)
        self.dir_settings_edit = QLineEdit(config_dir_global, self.Record)
        self.dir_settings_edit.setFont(QFont(self.font_id, 10))
        self.dir_settings_edit.setStyleSheet('QLineEdit { width: 400px; }')
        self.dir_settings_edit.move(220, 40)
        settings_dir_btn = QPushButton('Browse', self.Record)
        settings_dir_btn.setFont(QFont(self.font_id, 8))
        settings_dir_btn.move(625, 40)
        settings_dir_btn.setStyleSheet('QPushButton { min-width: 64px; min-height: 12px; max-width: 64px; max-height: 13px; }')
        settings_dir_btn.clicked.connect(self._open_settings_dialog)

        avisoft_exe_dir_label = QLabel('Audio Rec (usgh.exe) directory:', self.Record)
        avisoft_exe_dir_label.setFont(QFont(self.font_id, 12))
        avisoft_exe_dir_label.move(5, 70)
        self.recorder_settings_edit = QLineEdit(avisoft_rec_dir_global, self.Record)
        self.recorder_settings_edit.setFont(QFont(self.font_id, 10))
        self.recorder_settings_edit.setStyleSheet('QLineEdit { width: 400px; }')
        self.recorder_settings_edit.move(220, 70)
        recorder_dir_btn = QPushButton('Browse', self.Record)
        recorder_dir_btn.setFont(QFont(self.font_id, 8))
        recorder_dir_btn.move(625, 70)
        recorder_dir_btn.setStyleSheet('QPushButton { min-width: 64px; min-height: 12px; max-width: 64px; max-height: 13px; }')
        recorder_dir_btn.clicked.connect(self._open_recorder_dialog)

        avisoft_base_dir_label = QLabel('Avisoft Bioacoustics directory:', self.Record)
        avisoft_base_dir_label.setFont(QFont(self.font_id, 12))
        avisoft_base_dir_label.move(5, 100)
        self.avisoft_base_edit = QLineEdit(avisoft_base_dir_global, self.Record)
        self.avisoft_base_edit.setFont(QFont(self.font_id, 10))
        self.avisoft_base_edit.setStyleSheet('QLineEdit { width: 400px; }')
        self.avisoft_base_edit.move(220, 100)
        avisoft_base_dir_btn = QPushButton('Browse', self.Record)
        avisoft_base_dir_btn.setFont(QFont(self.font_id, 8))
        avisoft_base_dir_btn.move(625, 100)
        avisoft_base_dir_btn.setStyleSheet('QPushButton { min-width: 64px; min-height: 12px; max-width: 64px; max-height: 13px; }')
        avisoft_base_dir_btn.clicked.connect(self._open_avisoft_dialog)

        coolterm_base_dir_label = QLabel('CoolTerm base directory:', self.Record)
        coolterm_base_dir_label.setFont(QFont(self.font_id, 12))
        coolterm_base_dir_label.move(5, 130)
        self.coolterm_base_edit = QLineEdit(coolterm_base_dir_global, self.Record)
        self.coolterm_base_edit.setFont(QFont(self.font_id, 10))
        self.coolterm_base_edit.setStyleSheet('QLineEdit { width: 400px; }')
        self.coolterm_base_edit.move(220, 130)
        coolterm_base_dir_btn = QPushButton('Browse', self.Record)
        coolterm_base_dir_btn.setFont(QFont(self.font_id, 8))
        coolterm_base_dir_btn.move(625, 130)
        coolterm_base_dir_btn.setStyleSheet('QPushButton { min-width: 64px; min-height: 12px; max-width: 64px; max-height: 13px; }')
        coolterm_base_dir_btn.clicked.connect(self._open_coolterm_dialog)

        # recording files destination directories (across OS)
        recording_files_destination_linux_label = QLabel('File destination(s) Linux:', self.Record)
        recording_files_destination_linux_label.setFont(QFont(self.font_id, 12))
        recording_files_destination_linux_label.move(5, 160)
        self.recording_files_destination_linux = QLineEdit(f'{destination_linux_global}', self.Record)
        self.recording_files_destination_linux.setFont(QFont(self.font_id, 10))
        self.recording_files_destination_linux.setStyleSheet('QLineEdit { width: 490px; }')
        self.recording_files_destination_linux.move(220, 160)

        recording_files_destination_windows_label = QLabel('File destination(s) Windows:', self.Record)
        recording_files_destination_windows_label.setFont(QFont(self.font_id, 12))
        recording_files_destination_windows_label.move(5, 190)
        self.recording_files_destination_windows = QLineEdit(f'{destination_win_global}', self.Record)
        self.recording_files_destination_windows.setFont(QFont(self.font_id, 10))
        self.recording_files_destination_windows.setStyleSheet('QLineEdit { width: 490px; }')
        self.recording_files_destination_windows.move(220, 190)

        # set main recording parameters
        parameters_label = QLabel('Please set main recording parameters', self.Record)
        parameters_label.setFont(QFont(self.font_id, 13))
        parameters_label.setStyleSheet('QLabel { font-weight: bold;}')
        parameters_label.move(5, 230)

        conduct_audio_label = QLabel('Conduct AUDIO recording:', self.Record)
        conduct_audio_label.setFont(QFont(self.font_id, 12))
        conduct_audio_label.setStyleSheet('QLabel { color: #F58025; }')
        conduct_audio_label.move(5, 260)
        self.conduct_audio_cb = QComboBox(self.Record)
        self.conduct_audio_cb.addItems(['Yes', 'No'])
        self.conduct_audio_cb.setStyleSheet('QComboBox { width: 465px; }')
        self.conduct_audio_cb.activated.connect(partial(self._combo_box_prior_true, variable_id='conduct_audio_cb_bool'))
        self.conduct_audio_cb.move(220, 260)

        conduct_tracking_cal_label = QLabel('Conduct VIDEO calibration:', self.Record)
        conduct_tracking_cal_label.setFont(QFont(self.font_id, 12))
        conduct_tracking_cal_label.setStyleSheet('QLabel { color: #F58025; }')
        conduct_tracking_cal_label.move(5, 290)
        self.conduct_tracking_calibration_cb = QComboBox(self.Record)
        self.conduct_tracking_calibration_cb.addItems(['No', 'Yes'])
        self.conduct_tracking_calibration_cb.setStyleSheet('QComboBox { width: 465px; }')
        self.conduct_tracking_calibration_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='conduct_tracking_calibration_cb_bool'))
        self.conduct_tracking_calibration_cb.move(220, 290)

        cal_duration_label = QLabel('Calibration duration (min):', self.Record)
        cal_duration_label.setFont(QFont(self.font_id, 12))
        cal_duration_label.move(5, 320)
        self.calibration_session_duration = QLineEdit('5', self.Record)
        self.calibration_session_duration.setFont(QFont(self.font_id, 10))
        self.calibration_session_duration.setStyleSheet('QLineEdit { width: 490px; }')
        self.calibration_session_duration.move(220, 320)

        video_duration_label = QLabel('Video session duration (min):', self.Record)
        video_duration_label.setFont(QFont(self.font_id, 12))
        video_duration_label.move(5, 350)
        self.video_session_duration = QLineEdit('20', self.Record)
        self.video_session_duration.setFont(QFont(self.font_id, 10))
        self.video_session_duration.setStyleSheet('QLineEdit { width: 490px; }')
        self.video_session_duration.move(220, 350)

        email_notification_label = QLabel('Notify e-mail(s) of PC usage:', self.Record)
        email_notification_label.setFont(QFont(self.font_id, 12))
        email_notification_label.move(5, 380)
        self.email_recipients = QLineEdit(f'{email_list_global}', self.Record)
        self.email_recipients.setFont(QFont(self.font_id, 10))
        self.email_recipients.setStyleSheet('QLineEdit { width: 490px; }')
        self.email_recipients.move(220, 380)

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

        self.default_audio_settings = {'name': '999', 'id': '999', 'typech': '13', 'deviceid': '999',
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
                                       'filtercutoff': '15.0', 'filter': '0', 'fabtast': '250000', 'y2': '1322',
                                       'x2': '2557', 'y1': '10', 'x1': '1653', 'fftlength': '256',
                                       'usvmonitoringflags': '9136', 'dispspectrogramcontrast': '0.0', 'disprangespectrogram': '250.0',
                                       'disprangeamplitude': '100.0', 'disprangewaveform': '100.0', 'total': '1', 'dcolumns': '3',
                                       'display': '2', 'total_mic_number': '24', 'total_device_num': '2',
                                       'used_mics': '0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23'}

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
        record_three_x, record_three_y = (950, 900)
        self.setFixedSize(record_three_x, record_three_y)

        gvs_label = QLabel('General video recording settings', self.VideoSettings)
        gvs_label.setFont(QFont(self.font_id, 13))
        gvs_label.setStyleSheet('QLabel { font-weight: bold;}')
        gvs_label.move(5, 10)

        browser_label = QLabel('Browser:', self.VideoSettings)
        browser_label.setFont(QFont(self.font_id, 12))
        browser_label.move(5, 40)
        self.browser = QLineEdit('chrome', self.VideoSettings)
        self.browser.setFont(QFont(self.font_id, 10))
        self.browser.setStyleSheet('QLineEdit { width: 300px; }')
        self.browser.move(160, 40)

        use_cam_label = QLabel('Camera(s) to use:', self.VideoSettings)
        use_cam_label.setFont(QFont(self.font_id, 12))
        use_cam_label.move(5, 70)
        self.expected_cameras = QLineEdit(','.join(camera_ids_global), self.VideoSettings)
        self.expected_cameras.setFont(QFont(self.font_id, 10))
        self.expected_cameras.setStyleSheet('QLineEdit { width: 300px; }')
        self.expected_cameras.move(160, 70)

        rec_codec_label = QLabel('Recording codec:', self.VideoSettings)
        rec_codec_label.setFont(QFont(self.font_id, 12))
        rec_codec_label.move(5, 100)
        self.recording_codec_cb = QComboBox(self.VideoSettings)
        self.recording_codec_cb.addItems(['hq', 'mq', 'lq'])
        self.recording_codec_cb.setStyleSheet('QComboBox { width: 265px; }')
        self.recording_codec_cb.activated.connect(partial(self._combo_box_prior_codec, variable_id='recording_codec'))
        self.recording_codec_cb.move(160, 100)

        monitor_rec_label = QLabel('Monitor recording:', self.VideoSettings)
        monitor_rec_label.setFont(QFont(self.font_id, 12))
        monitor_rec_label.setStyleSheet('QLabel { color: #F58025; }')
        monitor_rec_label.move(5, 130)
        self.monitor_recording_cb = QComboBox(self.VideoSettings)
        self.monitor_recording_cb.addItems(['No', 'Yes'])
        self.monitor_recording_cb.setStyleSheet('QComboBox { width: 265px; }')
        self.monitor_recording_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='monitor_recording_cb_bool'))
        self.monitor_recording_cb.move(160, 130)

        monitor_cam_label = QLabel('Monitor ONE camera:', self.VideoSettings)
        monitor_cam_label.setFont(QFont(self.font_id, 12))
        monitor_cam_label.setStyleSheet('QLabel { color: #F58025; }')
        monitor_cam_label.move(5, 160)
        self.monitor_specific_camera_cb = QComboBox(self.VideoSettings)
        self.monitor_specific_camera_cb.addItems(['No', 'Yes'])
        self.monitor_specific_camera_cb.setStyleSheet('QComboBox { width: 265px; }')
        self.monitor_specific_camera_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='monitor_specific_camera_cb_bool'))
        self.monitor_specific_camera_cb.move(160, 160)

        specific_camera_serial_label = QLabel('ONE camera serial:', self.VideoSettings)
        specific_camera_serial_label.setFont(QFont(self.font_id, 12))
        specific_camera_serial_label.move(5, 190)
        self.specific_camera_serial = QLineEdit('21372315', self.VideoSettings)
        self.specific_camera_serial.setFont(QFont(self.font_id, 10))
        self.specific_camera_serial.setStyleSheet('QLineEdit { width: 300px; }')
        self.specific_camera_serial.move(160, 190)

        delete_post_copy_label = QLabel('Delete post copy:', self.VideoSettings)
        delete_post_copy_label.setFont(QFont(self.font_id, 12))
        delete_post_copy_label.setStyleSheet('QLabel { color: #F58025; }')
        delete_post_copy_label.move(5, 220)
        self.delete_post_copy_cb = QComboBox(self.VideoSettings)
        self.delete_post_copy_cb.addItems(['Yes', 'No'])
        self.delete_post_copy_cb.setStyleSheet('QComboBox { width: 265px; }')
        self.delete_post_copy_cb.activated.connect(partial(self._combo_box_prior_true, variable_id='delete_post_copy_cb_bool'))
        self.delete_post_copy_cb.move(160, 220)

        self.cal_fr_label = QLabel('Calibration (10 fps):', self.VideoSettings)
        self.cal_fr_label.setFixedWidth(150)
        self.cal_fr_label.setFont(QFont(self.font_id, 12))
        self.cal_fr_label.move(5, 250)
        self.calibration_frame_rate = QSlider(Qt.Orientation.Horizontal, self.VideoSettings)
        self.calibration_frame_rate.setFixedWidth(150)
        self.calibration_frame_rate.move(160, 255)
        self.calibration_frame_rate.setRange(10, 150)
        self.calibration_frame_rate.setValue(10)
        self.calibration_frame_rate.valueChanged.connect(self._update_cal_fr_label)

        self.fr_label = QLabel('Recording (150 fps):', self.VideoSettings)
        self.fr_label.setFixedWidth(150)
        self.fr_label.setFont(QFont(self.font_id, 12))
        self.fr_label.move(5, 280)
        self.cameras_frame_rate = QSlider(Qt.Orientation.Horizontal, self.VideoSettings)
        self.cameras_frame_rate.setFixedWidth(150)
        self.cameras_frame_rate.move(160, 285)
        self.cameras_frame_rate.setRange(10, 150)
        self.cameras_frame_rate.setValue(150)
        self.cameras_frame_rate.valueChanged.connect(self._update_fr_label)

        pcs_label = QLabel('Particular camera settings', self.VideoSettings)
        pcs_label.setFont(QFont(self.font_id, 13))
        pcs_label.setStyleSheet('QLabel { font-weight: bold;}')
        pcs_label.move(5, 325)

        for cam_idx, cam in enumerate(camera_ids_global):
            self._create_sliders_general(camera_id=cam, camera_color=camera_colors_global[cam_idx], y_start=355+(cam_idx*90))

        vm_label = QLabel('Video metadata', self.VideoSettings)
        vm_label.setFont(QFont(self.font_id, 13))
        vm_label.setStyleSheet('QLabel { font-weight: bold;}')
        vm_label.move(515, 10)

        experimenter_label = QLabel('Experimenter:', self.VideoSettings)
        experimenter_label.setFont(QFont(self.font_id, 12))
        experimenter_label.move(515, 40)
        self.experimenter = QLineEdit(f'{experimenter_id}', self.VideoSettings)
        self.experimenter.setFont(QFont(self.font_id, 10))
        self.experimenter.setStyleSheet('QLineEdit { width: 300px; }')
        self.experimenter.move(620, 40)

        mice_num_label = QLabel('Mouse count:', self.VideoSettings)
        mice_num_label.setFont(QFont(self.font_id, 12))
        mice_num_label.move(515, 70)
        self.mice_num = QLineEdit('2', self.VideoSettings)
        self.mice_num.setFont(QFont(self.font_id, 10))
        self.mice_num.setStyleSheet('QLineEdit { width: 300px; }')
        self.mice_num.move(620, 70)

        cage_ID_m1_label = QLabel('#1 Cage ID:', self.VideoSettings)
        cage_ID_m1_label.setFont(QFont(self.font_id, 12))
        cage_ID_m1_label.move(515, 100)
        self.cage_ID_m1 = QLineEdit('', self.VideoSettings)
        self.cage_ID_m1.setFont(QFont(self.font_id, 10))
        self.cage_ID_m1.setStyleSheet('QLineEdit { width: 300px; }')
        self.cage_ID_m1.move(620, 100)

        mouse_ID_m1_label = QLabel('#1 Mouse ID:', self.VideoSettings)
        mouse_ID_m1_label.setFont(QFont(self.font_id, 12))
        mouse_ID_m1_label.move(515, 130)
        self.mouse_ID_m1 = QLineEdit('', self.VideoSettings)
        self.mouse_ID_m1.setFont(QFont(self.font_id, 10))
        self.mouse_ID_m1.setStyleSheet('QLineEdit { width: 300px; }')
        self.mouse_ID_m1.move(620, 130)

        genotype_m1_label = QLabel('#1 Genotype:', self.VideoSettings)
        genotype_m1_label.setFont(QFont(self.font_id, 12))
        genotype_m1_label.move(515, 160)
        self.genotype_m1 = QLineEdit('CD1-WT', self.VideoSettings)
        self.genotype_m1.setFont(QFont(self.font_id, 10))
        self.genotype_m1.setStyleSheet('QLineEdit { width: 300px; }')
        self.genotype_m1.move(620, 160)

        sex_m1_label = QLabel('#1 Sex:', self.VideoSettings)
        sex_m1_label.setFont(QFont(self.font_id, 12))
        sex_m1_label.move(515, 190)
        self.sex_m1 = QLineEdit('male', self.VideoSettings)
        self.sex_m1.setFont(QFont(self.font_id, 10))
        self.sex_m1.setStyleSheet('QLineEdit { width: 300px; }')
        self.sex_m1.move(620, 190)

        DOB_m1_label = QLabel('#1 DOB:', self.VideoSettings)
        DOB_m1_label.setFont(QFont(self.font_id, 12))
        DOB_m1_label.move(515, 220)
        self.DOB_m1 = QLineEdit('', self.VideoSettings)
        self.DOB_m1.setFont(QFont(self.font_id, 10))
        self.DOB_m1.setStyleSheet('QLineEdit { width: 300px; }')
        self.DOB_m1.move(620, 220)

        housing_m1_label = QLabel('#1 Housing:', self.VideoSettings)
        housing_m1_label.setFont(QFont(self.font_id, 12))
        housing_m1_label.move(515, 250)
        self.housing_m1 = QLineEdit('group', self.VideoSettings)
        self.housing_m1.setFont(QFont(self.font_id, 10))
        self.housing_m1.setStyleSheet('QLineEdit { width: 300px; }')
        self.housing_m1.move(620, 250)

        cage_ID_m2_label = QLabel('#2 Cage ID:', self.VideoSettings)
        cage_ID_m2_label.setFont(QFont(self.font_id, 12))
        cage_ID_m2_label.move(515, 280)
        self.cage_ID_m2 = QLineEdit('', self.VideoSettings)
        self.cage_ID_m2.setFont(QFont(self.font_id, 10))
        self.cage_ID_m2.setStyleSheet('QLineEdit { width: 300px; }')
        self.cage_ID_m2.move(620, 280)

        mouse_ID_m2_label = QLabel('#2 Mouse ID:', self.VideoSettings)
        mouse_ID_m2_label.setFont(QFont(self.font_id, 12))
        mouse_ID_m2_label.move(515, 310)
        self.mouse_ID_m2 = QLineEdit('', self.VideoSettings)
        self.mouse_ID_m2.setFont(QFont(self.font_id, 10))
        self.mouse_ID_m2.setStyleSheet('QLineEdit { width: 300px; }')
        self.mouse_ID_m2.move(620, 310)

        genotype_m2_label = QLabel('#2 Genotype:', self.VideoSettings)
        genotype_m2_label.setFont(QFont(self.font_id, 12))
        genotype_m2_label.move(515, 340)
        self.genotype_m2 = QLineEdit('CD1-WT', self.VideoSettings)
        self.genotype_m2.setFont(QFont(self.font_id, 10))
        self.genotype_m2.setStyleSheet('QLineEdit { width: 300px; }')
        self.genotype_m2.move(620, 340)

        sex_m2_label = QLabel('#2 Sex:', self.VideoSettings)
        sex_m2_label.setFont(QFont(self.font_id, 12))
        sex_m2_label.move(515, 370)
        self.sex_m2 = QLineEdit('female', self.VideoSettings)
        self.sex_m2.setFont(QFont(self.font_id, 10))
        self.sex_m2.setStyleSheet('QLineEdit { width: 300px; }')
        self.sex_m2.move(620, 370)

        DOB_m2_label = QLabel('#2 DOB:', self.VideoSettings)
        DOB_m2_label.setFont(QFont(self.font_id, 12))
        DOB_m2_label.move(515, 400)
        self.DOB_m2 = QLineEdit('', self.VideoSettings)
        self.DOB_m2.setFont(QFont(self.font_id, 10))
        self.DOB_m2.setStyleSheet('QLineEdit { width: 300px; }')
        self.DOB_m2.move(620, 400)

        housing_m2_label = QLabel('#2 Housing:', self.VideoSettings)
        housing_m2_label.setFont(QFont(self.font_id, 12))
        housing_m2_label.move(515, 430)
        self.housing_m2 = QLineEdit('group', self.VideoSettings)
        self.housing_m2.setFont(QFont(self.font_id, 10))
        self.housing_m2.setStyleSheet('QLineEdit { width: 300px; }')
        self.housing_m2.move(620, 430)

        other_label = QLabel('Other info:', self.VideoSettings)
        other_label.setFont(QFont(self.font_id, 12))
        other_label.move(515, 460)
        self.other = QTextEdit('Lorem ipsum dolor sit amet', self.VideoSettings)
        self.other.setFont(QFont(self.font_id, 10))
        self.other.move(620, 460)
        self.other.setFixedSize(302, 390)

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

        self._save_modified_values_to_toml(run_exp_bool=True, message_func=self._message)

        exp_settings_dict_final = toml.load(f"{self.settings_dict['general']['config_settings_directory']}{os.sep}behavioral_experiments_settings.toml")
        self.run_exp = ExperimentController(message_output=self._message,
                                            email_receivers=self.email_recipients,
                                            exp_settings_dict=exp_settings_dict_final)

        self._create_buttons_record(seq=3, class_option=self.ConductRecording,
                                    button_pos_y=record_four_y - 35, next_button_x_pos=record_four_x - 100)

    def process_one(self):
        self.ProcessSettings = ProcessSettings(self)
        self.setWindowTitle(f'{app_name} (Process recordings > Settings)')
        self.setCentralWidget(self.ProcessSettings)
        record_four_x, record_four_y = (1070, 1100)
        self.setFixedSize(record_four_x, record_four_y)

        # select all directories for processing
        processing_dir_label = QLabel('(*) Root directories for processing', self.ProcessSettings)
        processing_dir_label.setFont(QFont(self.font_id, 13))
        processing_dir_label.setStyleSheet('QLabel { font-weight: bold;}')
        processing_dir_label.move(50, 10)
        self.processing_dir_edit = QTextEdit('', self.ProcessSettings)
        self.processing_dir_edit.setFont(QFont(self.font_id, 10))
        self.processing_dir_edit.move(10, 40)
        self.processing_dir_edit.setFixedSize(350, 380)

        self.dir_settings_edit = QLineEdit(config_dir_global, self.ProcessSettings)
        self.dir_settings_edit.setFont(QFont(self.font_id, 10))
        self.dir_settings_edit.setStyleSheet('QLineEdit { width: 260px; }')
        self.dir_settings_edit.move(10, 425)
        settings_dir_btn = QPushButton('Browse', self.ProcessSettings)
        settings_dir_btn.setFont(QFont(self.font_id, 8))
        settings_dir_btn.move(275, 425)
        settings_dir_btn.setStyleSheet('QPushButton { min-width: 64px; min-height: 12px; max-width: 64px; max-height: 13px; }')
        settings_dir_btn.clicked.connect(self._open_settings_dialog)

        pc_usage_process_label = QLabel('Notify e-mail(s) of PC usage:', self.ProcessSettings)
        pc_usage_process_label.setFont(QFont(self.font_id, 12))
        pc_usage_process_label.move(10, 455)
        self.pc_usage_process = QLineEdit(f'{email_list_global}', self.ProcessSettings)
        self.pc_usage_process.setFont(QFont(self.font_id, 10))
        self.pc_usage_process.setStyleSheet('QLineEdit { width: 150px; }')
        self.pc_usage_process.move(210, 455)

        # set other parameters

        gvs_label = QLabel('Video processing settings', self.ProcessSettings)
        gvs_label.setFont(QFont(self.font_id, 13))
        gvs_label.setStyleSheet('QLabel { font-weight: bold;}')
        gvs_label.move(10, 500)

        conduct_video_concatenation_label = QLabel('Conduct video concatenation:', self.ProcessSettings)
        conduct_video_concatenation_label.setFont(QFont(self.font_id, 12))
        conduct_video_concatenation_label.setStyleSheet('QLabel { color: #F58025; }')
        conduct_video_concatenation_label.move(10, 530)
        self.conduct_video_concatenation_cb = QComboBox(self.ProcessSettings)
        self.conduct_video_concatenation_cb.addItems(['No', 'Yes'])
        self.conduct_video_concatenation_cb.setStyleSheet('QComboBox { width: 105px; }')
        self.conduct_video_concatenation_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='conduct_video_concatenation_cb_bool'))
        self.conduct_video_concatenation_cb.move(225, 530)

        concatenate_cam_serial_num_label = QLabel('Camera serial num(s):', self.ProcessSettings)
        concatenate_cam_serial_num_label.setFont(QFont(self.font_id, 12))
        concatenate_cam_serial_num_label.move(10, 560)
        self.concatenate_cam_serial_num = QLineEdit(','.join(camera_ids_global), self.ProcessSettings)
        self.concatenate_cam_serial_num.setFont(QFont(self.font_id, 10))
        self.concatenate_cam_serial_num.setStyleSheet('QLineEdit { width: 195px; }')
        self.concatenate_cam_serial_num.move(165, 560)

        concatenated_video_name_label = QLabel('Chained video name:', self.ProcessSettings)
        concatenated_video_name_label.setFont(QFont(self.font_id, 12))
        concatenated_video_name_label.move(10, 590)
        self.concatenated_video_name = QLineEdit('concatenated_temp', self.ProcessSettings)
        self.concatenated_video_name.setFont(QFont(self.font_id, 10))
        self.concatenated_video_name.setStyleSheet('QLineEdit { width: 195px; }')
        self.concatenated_video_name.move(165, 590)

        concatenate_video_ext_label = QLabel('Video file format:', self.ProcessSettings)
        concatenate_video_ext_label.setFont(QFont(self.font_id, 12))
        concatenate_video_ext_label.move(10, 620)
        self.concatenate_video_ext = QLineEdit('mp4', self.ProcessSettings)
        self.concatenate_video_ext.setFont(QFont(self.font_id, 10))
        self.concatenate_video_ext.setStyleSheet('QLineEdit { width: 195px; }')
        self.concatenate_video_ext.move(165, 620)

        conduct_video_fps_change_cb_label = QLabel('Conduct video re-encoding:', self.ProcessSettings)
        conduct_video_fps_change_cb_label.setFont(QFont(self.font_id, 12))
        conduct_video_fps_change_cb_label.setStyleSheet('QLabel { color: #F58025; }')
        conduct_video_fps_change_cb_label.move(10, 650)
        self.conduct_video_fps_change_cb = QComboBox(self.ProcessSettings)
        self.conduct_video_fps_change_cb.addItems(['No', 'Yes'])
        self.conduct_video_fps_change_cb.setStyleSheet('QComboBox { width: 105px; }')
        self.conduct_video_fps_change_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='conduct_video_fps_change_cb_bool'))
        self.conduct_video_fps_change_cb.move(225, 650)

        change_fps_cam_serial_num_label = QLabel('Camera serial num(s):', self.ProcessSettings)
        change_fps_cam_serial_num_label.setFont(QFont(self.font_id, 12))
        change_fps_cam_serial_num_label.move(10, 680)
        self.change_fps_cam_serial_num = QLineEdit(','.join(camera_ids_global), self.ProcessSettings)
        self.change_fps_cam_serial_num.setFont(QFont(self.font_id, 10))
        self.change_fps_cam_serial_num.setStyleSheet('QLineEdit { width: 195px; }')
        self.change_fps_cam_serial_num.move(165, 680)

        conversion_target_file_label = QLabel('Chained video name:', self.ProcessSettings)
        conversion_target_file_label.setFont(QFont(self.font_id, 12))
        conversion_target_file_label.move(10, 710)
        self.conversion_target_file = QLineEdit('concatenated_temp', self.ProcessSettings)
        self.conversion_target_file.setFont(QFont(self.font_id, 10))
        self.conversion_target_file.setStyleSheet('QLineEdit { width: 195px; }')
        self.conversion_target_file.move(165, 710)

        conversion_vid_ext_label = QLabel('Video file format:', self.ProcessSettings)
        conversion_vid_ext_label.setFont(QFont(self.font_id, 12))
        conversion_vid_ext_label.move(10, 740)
        self.conversion_vid_ext = QLineEdit('mp4', self.ProcessSettings)
        self.conversion_vid_ext.setFont(QFont(self.font_id, 10))
        self.conversion_vid_ext.setStyleSheet('QLineEdit { width: 195px; }')
        self.conversion_vid_ext.move(165, 740)

        constant_rate_factor_label = QLabel('Rate factor (-crf):', self.ProcessSettings)
        constant_rate_factor_label.setFont(QFont(self.font_id, 12))
        constant_rate_factor_label.move(10, 770)
        self.constant_rate_factor = QLineEdit('16', self.ProcessSettings)
        self.constant_rate_factor.setFont(QFont(self.font_id, 10))
        self.constant_rate_factor.setStyleSheet('QLineEdit { width: 195px; }')
        self.constant_rate_factor.move(165, 770)

        encoding_preset_label = QLabel('Encoding preset:', self.ProcessSettings)
        encoding_preset_label.setFont(QFont(self.font_id, 12))
        encoding_preset_label.move(10, 800)
        self.encoding_preset = QLineEdit('veryfast', self.ProcessSettings)
        self.encoding_preset.setFont(QFont(self.font_id, 10))
        self.encoding_preset.setStyleSheet('QLineEdit { width: 195px; }')
        self.encoding_preset.move(165, 800)

        delete_con_file_cb_label = QLabel('Delete chained video file:', self.ProcessSettings)
        delete_con_file_cb_label.setFont(QFont(self.font_id, 12))
        delete_con_file_cb_label.move(10, 830)
        self.delete_con_file_cb = QComboBox(self.ProcessSettings)
        self.delete_con_file_cb.addItems(['Yes', 'No'])
        self.delete_con_file_cb.setStyleSheet('QComboBox { width: 105px; }')
        self.delete_con_file_cb.activated.connect(partial(self._combo_box_prior_true, variable_id='delete_con_file_cb_bool'))
        self.delete_con_file_cb.move(225, 830)

        gas_label = QLabel('Audio processing settings', self.ProcessSettings)
        gas_label.setFont(QFont(self.font_id, 13))
        gas_label.setStyleSheet('QLabel { font-weight: bold;}')
        gas_label.move(10, 870)

        conduct_multichannel_conversion_cb_label = QLabel('Convert multi-ch to single-ch files:', self.ProcessSettings)
        conduct_multichannel_conversion_cb_label.setFont(QFont(self.font_id, 12))
        conduct_multichannel_conversion_cb_label.setStyleSheet('QLabel { color: #F58025; }')
        conduct_multichannel_conversion_cb_label.move(10, 900)
        self.conduct_multichannel_conversion_cb = QComboBox(self.ProcessSettings)
        self.conduct_multichannel_conversion_cb.addItems(['No', 'Yes'])
        self.conduct_multichannel_conversion_cb.setStyleSheet('QComboBox { width: 80px; }')
        self.conduct_multichannel_conversion_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='conduct_multichannel_conversion_cb_bool'))
        self.conduct_multichannel_conversion_cb.move(250, 900)

        crop_wav_cam_cb_label = QLabel('Crop AUDIO files to match VIDEO:', self.ProcessSettings)
        crop_wav_cam_cb_label.setFont(QFont(self.font_id, 12))
        crop_wav_cam_cb_label.setStyleSheet('QLabel { color: #F58025; }')
        crop_wav_cam_cb_label.move(10, 930)
        self.crop_wav_cam_cb = QComboBox(self.ProcessSettings)
        self.crop_wav_cam_cb.addItems(['No', 'Yes'])
        self.crop_wav_cam_cb.setStyleSheet('QComboBox { width: 80px; }')
        self.crop_wav_cam_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='crop_wav_cam_cb_bool'))
        self.crop_wav_cam_cb.move(250, 930)

        device_receiving_input_cb_label = QLabel('Trgbox-USGH device (m | s):', self.ProcessSettings)
        device_receiving_input_cb_label.setFont(QFont(self.font_id, 12))
        device_receiving_input_cb_label.move(10, 960)
        self.device_receiving_input_cb = QComboBox(self.ProcessSettings)
        self.device_receiving_input_cb.addItems(['m', 's'])
        self.device_receiving_input_cb.setStyleSheet('QComboBox { width: 80px; }')
        self.device_receiving_input_cb.activated.connect(partial(self._combo_box_prior_audio_device_camera_input, variable_id='device_receiving_input'))
        self.device_receiving_input_cb.move(250, 960)

        ch_receiving_input_label = QLabel('Trgbox-USGH ch (1-12):', self.ProcessSettings)
        ch_receiving_input_label.setFont(QFont(self.font_id, 12))
        ch_receiving_input_label.move(10, 990)
        self.ch_receiving_input = QLineEdit('1', self.ProcessSettings)
        self.ch_receiving_input.setFont(QFont(self.font_id, 10))
        self.ch_receiving_input.setStyleSheet('QLineEdit { width: 108px; }')
        self.ch_receiving_input.move(250, 990)

        # column 2

        av_sync_label = QLabel('Synchronization between A/V files', self.ProcessSettings)
        av_sync_label.setFont(QFont(self.font_id, 13))
        av_sync_label.setStyleSheet('QLabel { font-weight: bold;}')
        av_sync_label.move(400, 10)

        conduct_sync_cb_label = QLabel('Conduct A/V sync check:', self.ProcessSettings)
        conduct_sync_cb_label.setFont(QFont(self.font_id, 12))
        conduct_sync_cb_label.setStyleSheet('QLabel { color: #F58025; }')
        conduct_sync_cb_label.move(400, 40)
        self.conduct_sync_cb = QComboBox(self.ProcessSettings)
        self.conduct_sync_cb.addItems(['No', 'Yes'])
        self.conduct_sync_cb.setStyleSheet('QComboBox { width: 80px; }')
        self.conduct_sync_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='conduct_sync_cb_bool'))
        self.conduct_sync_cb.move(590, 40)

        phidget_extra_data_camera_label = QLabel('Phidget data camera:', self.ProcessSettings)
        phidget_extra_data_camera_label.setFont(QFont(self.font_id, 12))
        phidget_extra_data_camera_label.move(400, 70)
        self.phidget_extra_data_camera = QLineEdit('22085397', self.ProcessSettings)
        self.phidget_extra_data_camera.setFont(QFont(self.font_id, 10))
        self.phidget_extra_data_camera.setStyleSheet('QLineEdit { width: 108px; }')
        self.phidget_extra_data_camera.move(590, 70)

        a_ch_receiving_input_label = QLabel('Arduino-USGH ch (1-12):', self.ProcessSettings)
        a_ch_receiving_input_label.setFont(QFont(self.font_id, 12))
        a_ch_receiving_input_label.move(400, 100)
        self.a_ch_receiving_input = QLineEdit('2', self.ProcessSettings)
        self.a_ch_receiving_input.setFont(QFont(self.font_id, 10))
        self.a_ch_receiving_input.setStyleSheet('QLineEdit { width: 108px; }')
        self.a_ch_receiving_input.move(590, 100)

        v_camera_serial_num_label = QLabel('Sync camera serial(s):', self.ProcessSettings)
        v_camera_serial_num_label.setFont(QFont(self.font_id, 12))
        v_camera_serial_num_label.move(400, 130)
        self.v_camera_serial_num = QLineEdit('21372315', self.ProcessSettings)
        self.v_camera_serial_num.setFont(QFont(self.font_id, 10))
        self.v_camera_serial_num.setStyleSheet('QLineEdit { width: 108px; }')
        self.v_camera_serial_num.move(590, 130)

        v_video_extension_label = QLabel('Video file format:', self.ProcessSettings)
        v_video_extension_label.setFont(QFont(self.font_id, 12))
        v_video_extension_label.move(400, 160)
        self.v_video_extension = QLineEdit('mp4', self.ProcessSettings)
        self.v_video_extension.setFont(QFont(self.font_id, 10))
        self.v_video_extension.setStyleSheet('QLineEdit { width: 108px; }')
        self.v_video_extension.move(590, 160)

        v_led_px_version_label = QLabel('LED px version:', self.ProcessSettings)
        v_led_px_version_label.setFont(QFont(self.font_id, 12))
        v_led_px_version_label.move(400, 190)
        self.v_led_px_version = QLineEdit('current', self.ProcessSettings)
        self.v_led_px_version.setFont(QFont(self.font_id, 10))
        self.v_led_px_version.setStyleSheet('QLineEdit { width: 108px; }')
        self.v_led_px_version.move(590, 190)

        v_led_px_dev_label = QLabel('LED deviation (px):', self.ProcessSettings)
        v_led_px_dev_label.setFont(QFont(self.font_id, 12))
        v_led_px_dev_label.move(400, 220)
        self.v_led_px_dev = QLineEdit('10', self.ProcessSettings)
        self.v_led_px_dev.setFont(QFont(self.font_id, 10))
        self.v_led_px_dev.setStyleSheet('QLineEdit { width: 108px; }')
        self.v_led_px_dev.move(590, 220)

        v_relative_intensity_threshold_label = QLabel('Rel intensity threshold:', self.ProcessSettings)
        v_relative_intensity_threshold_label.setFont(QFont(self.font_id, 12))
        v_relative_intensity_threshold_label.move(400, 250)
        self.v_relative_intensity_threshold = QLineEdit('0.6', self.ProcessSettings)
        self.v_relative_intensity_threshold.setFont(QFont(self.font_id, 10))
        self.v_relative_intensity_threshold.setStyleSheet('QLineEdit { width: 108px; }')
        self.v_relative_intensity_threshold.move(590, 250)

        v_millisecond_divergence_tolerance_label = QLabel('Divergence tolerance (ms):', self.ProcessSettings)
        v_millisecond_divergence_tolerance_label.setFont(QFont(self.font_id, 12))
        v_millisecond_divergence_tolerance_label.move(400, 280)
        self.v_millisecond_divergence_tolerance = QLineEdit('10', self.ProcessSettings)
        self.v_millisecond_divergence_tolerance.setFont(QFont(self.font_id, 10))
        self.v_millisecond_divergence_tolerance.setStyleSheet('QLineEdit { width: 108px; }')
        self.v_millisecond_divergence_tolerance.move(590, 280)

        ev_sync_label = QLabel('Neural data processing settings', self.ProcessSettings)
        ev_sync_label.setFont(QFont(self.font_id, 13))
        ev_sync_label.setStyleSheet('QLabel { font-weight: bold;}')
        ev_sync_label.move(400, 320)

        conduct_ephys_file_chaining_label = QLabel('Conduct e-phys file concat:', self.ProcessSettings)
        conduct_ephys_file_chaining_label.setFont(QFont(self.font_id, 12))
        conduct_ephys_file_chaining_label.setStyleSheet('QLabel { color: #F58025; }')
        conduct_ephys_file_chaining_label.move(400, 350)
        self.conduct_ephys_file_chaining_cb = QComboBox(self.ProcessSettings)
        self.conduct_ephys_file_chaining_cb.addItems(['No', 'Yes'])
        self.conduct_ephys_file_chaining_cb.setStyleSheet('QComboBox { width: 80px; }')
        self.conduct_ephys_file_chaining_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='conduct_ephys_file_chaining_cb_bool'))
        self.conduct_ephys_file_chaining_cb.move(590, 350)

        conduct_nv_sync_cb_label = QLabel('Conduct E/V sync check:', self.ProcessSettings)
        conduct_nv_sync_cb_label.setFont(QFont(self.font_id, 12))
        conduct_nv_sync_cb_label.setStyleSheet('QLabel { color: #F58025; }')
        conduct_nv_sync_cb_label.move(400, 380)
        self.conduct_nv_sync_cb = QComboBox(self.ProcessSettings)
        self.conduct_nv_sync_cb.addItems(['No', 'Yes'])
        self.conduct_nv_sync_cb.setStyleSheet('QComboBox { width: 80px; }')
        self.conduct_nv_sync_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='conduct_nv_sync_cb_bool'))
        self.conduct_nv_sync_cb.move(590, 380)

        split_cluster_spikes_cb_label = QLabel('Split clusters to sessions:', self.ProcessSettings)
        split_cluster_spikes_cb_label.setFont(QFont(self.font_id, 12))
        split_cluster_spikes_cb_label.setStyleSheet('QLabel { color: #F58025; }')
        split_cluster_spikes_cb_label.move(400, 410)
        self.split_cluster_spikes_cb = QComboBox(self.ProcessSettings)
        self.split_cluster_spikes_cb.addItems(['No', 'Yes'])
        self.split_cluster_spikes_cb.setStyleSheet('QComboBox { width: 80px; }')
        self.split_cluster_spikes_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='split_cluster_spikes_cb_bool'))
        self.split_cluster_spikes_cb.move(590, 410)

        npx_file_type_cb_label = QLabel('Rec file format (ap | lf):', self.ProcessSettings)
        npx_file_type_cb_label.setFont(QFont(self.font_id, 12))
        npx_file_type_cb_label.move(400, 440)
        self.npx_file_type_cb = QComboBox(self.ProcessSettings)
        self.npx_file_type_cb.addItems(['ap', 'lf'])
        self.npx_file_type_cb.setStyleSheet('QComboBox { width: 80px; }')
        self.npx_file_type_cb.activated.connect(partial(self._combo_box_prior_npx_file_type, variable_id='npx_file_type'))
        self.npx_file_type_cb.move(590, 440)

        npx_ms_divergence_tolerance_label = QLabel('Divergence tolerance (ms):', self.ProcessSettings)
        npx_ms_divergence_tolerance_label.setFont(QFont(self.font_id, 12))
        npx_ms_divergence_tolerance_label.move(400, 470)
        self.npx_ms_divergence_tolerance = QLineEdit('10', self.ProcessSettings)
        self.npx_ms_divergence_tolerance.setFont(QFont(self.font_id, 10))
        self.npx_ms_divergence_tolerance.setStyleSheet('QLineEdit { width: 108px; }')
        self.npx_ms_divergence_tolerance.move(590, 470)

        kilosort_version_cb_label = QLabel('Kilosort version:', self.ProcessSettings)
        kilosort_version_cb_label.setFont(QFont(self.font_id, 12))
        kilosort_version_cb_label.move(400, 500)
        self.kilosort_version_cb = QComboBox(self.ProcessSettings)
        self.kilosort_version_cb.addItems(['4', '3', '2.5', '2'])
        self.kilosort_version_cb.setStyleSheet('QComboBox { width: 80px; }')
        self.kilosort_version_cb.activated.connect(partial(self._combo_box_kilosort_version, variable_id='kilosort_version'))
        self.kilosort_version_cb.move(590, 500)

        min_spike_num_label = QLabel('Min num of spikes:', self.ProcessSettings)
        min_spike_num_label.setFont(QFont(self.font_id, 12))
        min_spike_num_label.move(400, 530)
        self.min_spike_num = QLineEdit('0', self.ProcessSettings)
        self.min_spike_num.setFont(QFont(self.font_id, 10))
        self.min_spike_num.setStyleSheet('QLineEdit { width: 108px; }')
        self.min_spike_num.move(590, 530)

        hpss_label = QLabel('Harmonic-percussive source separation', self.ProcessSettings)
        hpss_label.setFont(QFont(self.font_id, 13))
        hpss_label.setStyleSheet('QLabel { font-weight: bold;}')
        hpss_label.move(400, 570)

        conduct_hpss_cb_label = QLabel('Conduct HPSS (serial):', self.ProcessSettings)
        conduct_hpss_cb_label.setFont(QFont(self.font_id, 12))
        conduct_hpss_cb_label.setStyleSheet('QLabel { color: #F58025; }')
        conduct_hpss_cb_label.move(400, 600)
        self.conduct_hpss_cb = QComboBox(self.ProcessSettings)
        self.conduct_hpss_cb.addItems(['No', 'Yes'])
        self.conduct_hpss_cb.setStyleSheet('QComboBox { width: 80px; }')
        self.conduct_hpss_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='conduct_hpss_cb_bool'))
        self.conduct_hpss_cb.move(590, 600)

        stft_label = QLabel('STFT window and hop size:', self.ProcessSettings)
        stft_label.setFont(QFont(self.font_id, 12))
        stft_label.move(400, 630)
        self.stft_window_hop = QLineEdit('512,128', self.ProcessSettings)
        self.stft_window_hop.setFont(QFont(self.font_id, 10))
        self.stft_window_hop.setStyleSheet('QLineEdit { width: 108px; }')
        self.stft_window_hop.move(590, 630)

        hpss_kernel_size_label = QLabel('HPSS kernel size:', self.ProcessSettings)
        hpss_kernel_size_label.setFont(QFont(self.font_id, 12))
        hpss_kernel_size_label.move(400, 660)
        self.hpss_kernel_size = QLineEdit('5,60', self.ProcessSettings)
        self.hpss_kernel_size.setFont(QFont(self.font_id, 10))
        self.hpss_kernel_size.setStyleSheet('QLineEdit { width: 108px; }')
        self.hpss_kernel_size.move(590, 660)

        hpss_power_label = QLabel('HPSS power:', self.ProcessSettings)
        hpss_power_label.setFont(QFont(self.font_id, 12))
        hpss_power_label.move(400, 690)
        self.hpss_power = QLineEdit('4.0', self.ProcessSettings)
        self.hpss_power.setFont(QFont(self.font_id, 10))
        self.hpss_power.setStyleSheet('QLineEdit { width: 108px; }')
        self.hpss_power.move(590, 690)

        hpss_margin_label = QLabel('HPSS margin:', self.ProcessSettings)
        hpss_margin_label.setFont(QFont(self.font_id, 12))
        hpss_margin_label.move(400, 720)
        self.hpss_margin = QLineEdit('4,1', self.ProcessSettings)
        self.hpss_margin.setFont(QFont(self.font_id, 10))
        self.hpss_margin.setStyleSheet('QLineEdit { width: 108px; }')
        self.hpss_margin.move(590, 720)

        audio_filter_label = QLabel('Band-pass filter audio files', self.ProcessSettings)
        audio_filter_label.setFont(QFont(self.font_id, 13))
        audio_filter_label.setStyleSheet('QLabel { font-weight: bold;}')
        audio_filter_label.move(400, 760)

        filter_audio_cb_label = QLabel('Filter individual audio files:', self.ProcessSettings)
        filter_audio_cb_label.setFont(QFont(self.font_id, 12))
        filter_audio_cb_label.setStyleSheet('QLabel { color: #F58025; }')
        filter_audio_cb_label.move(400, 790)
        self.filter_audio_cb = QComboBox(self.ProcessSettings)
        self.filter_audio_cb.addItems(['No', 'Yes'])
        self.filter_audio_cb.setStyleSheet('QComboBox { width: 80px; }')
        self.filter_audio_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='filter_audio_cb_bool'))
        self.filter_audio_cb.move(590, 790)

        audio_filter_format_label = QLabel('Audio file format:', self.ProcessSettings)
        audio_filter_format_label.setFont(QFont(self.font_id, 12))
        audio_filter_format_label.move(400, 820)
        self.audio_filter_format = QLineEdit('wav', self.ProcessSettings)
        self.audio_filter_format.setFont(QFont(self.font_id, 10))
        self.audio_filter_format.setStyleSheet('QLineEdit { width: 108px; }')
        self.audio_filter_format.move(590, 820)

        filter_freq_bounds_label = QLabel('Filter freq bounds (Hz):', self.ProcessSettings)
        filter_freq_bounds_label.setFont(QFont(self.font_id, 12))
        filter_freq_bounds_label.move(400, 850)
        self.filter_freq_bounds = QLineEdit('0, 30000', self.ProcessSettings)
        self.filter_freq_bounds.setFont(QFont(self.font_id, 10))
        self.filter_freq_bounds.setStyleSheet('QLineEdit { width: 108px; }')
        self.filter_freq_bounds.move(590, 850)

        filter_dirs_label = QLabel('Folder(s) to filter:', self.ProcessSettings)
        filter_dirs_label.setFont(QFont(self.font_id, 12))
        filter_dirs_label.move(400, 880)
        self.filter_dirs = QLineEdit('hpss', self.ProcessSettings)
        self.filter_dirs.setFont(QFont(self.font_id, 10))
        self.filter_dirs.setStyleSheet('QLineEdit { width: 108px; }')
        self.filter_dirs.move(590, 880)

        conc_audio_label = QLabel('Concatenate audio files', self.ProcessSettings)
        conc_audio_label.setFont(QFont(self.font_id, 13))
        conc_audio_label.setStyleSheet('QLabel { font-weight: bold;}')
        conc_audio_label.move(400, 920)

        conc_audio_cb_label = QLabel('Concatenate to MEMMAP:', self.ProcessSettings)
        conc_audio_cb_label.setFont(QFont(self.font_id, 12))
        conc_audio_cb_label.setStyleSheet('QLabel { color: #F58025; }')
        conc_audio_cb_label.move(400, 950)
        self.conc_audio_cb = QComboBox(self.ProcessSettings)
        self.conc_audio_cb.addItems(['No', 'Yes'])
        self.conc_audio_cb.setStyleSheet('QComboBox { width: 80px; }')
        self.conc_audio_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='conc_audio_cb_bool'))
        self.conc_audio_cb.move(590, 950)

        audio_format_label = QLabel('Audio file format:', self.ProcessSettings)
        audio_format_label.setFont(QFont(self.font_id, 12))
        audio_format_label.move(400, 980)
        self.audio_format = QLineEdit('wav', self.ProcessSettings)
        self.audio_format.setFont(QFont(self.font_id, 10))
        self.audio_format.setStyleSheet('QLineEdit { width: 108px; }')
        self.audio_format.move(590, 980)

        concat_dirs_label = QLabel('Folder(s) to concatenate:', self.ProcessSettings)
        concat_dirs_label.setFont(QFont(self.font_id, 12))
        concat_dirs_label.move(400, 1010)
        self.concat_dirs = QLineEdit('hpss_filtered', self.ProcessSettings)
        self.concat_dirs.setFont(QFont(self.font_id, 10))
        self.concat_dirs.setStyleSheet('QLineEdit { width: 108px; }')
        self.concat_dirs.move(590, 1010)

        # column 3

        anipose_operations_label = QLabel('Anipose (3D) operations', self.ProcessSettings)
        anipose_operations_label.setFont(QFont(self.font_id, 13))
        anipose_operations_label.setStyleSheet('QLabel { font-weight: bold;}')
        anipose_operations_label.move(740, 10)

        sleap_file_conversion_cb_label = QLabel('Conduct SLP-H5 conversion:', self.ProcessSettings)
        sleap_file_conversion_cb_label.setFont(QFont(self.font_id, 12))
        sleap_file_conversion_cb_label.setStyleSheet('QLabel { color: #F58025; }')
        sleap_file_conversion_cb_label.move(740, 40)
        self.sleap_file_conversion_cb = QComboBox(self.ProcessSettings)
        self.sleap_file_conversion_cb.addItems(['No', 'Yes'])
        self.sleap_file_conversion_cb.setStyleSheet('QComboBox { width: 80px; }')
        self.sleap_file_conversion_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='sleap_file_conversion_cb_bool'))
        self.sleap_file_conversion_cb.move(940, 40)

        sleap_conda_label = QLabel('SLEAP conda env name:', self.ProcessSettings)
        sleap_conda_label.setFont(QFont(self.font_id, 12))
        sleap_conda_label.move(740, 70)
        self.sleap_conda = QLineEdit('sleap1.3.3', self.ProcessSettings)
        self.sleap_conda.setFont(QFont(self.font_id, 10))
        self.sleap_conda.setStyleSheet('QLineEdit { width: 108px; }')
        self.sleap_conda.move(940, 70)

        anipose_calibration_cb_label = QLabel('Conduct AP calibration:', self.ProcessSettings)
        anipose_calibration_cb_label.setFont(QFont(self.font_id, 12))
        anipose_calibration_cb_label.setStyleSheet('QLabel { color: #F58025; }')
        anipose_calibration_cb_label.move(740, 100)
        self.anipose_calibration_cb = QComboBox(self.ProcessSettings)
        self.anipose_calibration_cb.addItems(['No', 'Yes'])
        self.anipose_calibration_cb.setStyleSheet('QComboBox { width: 80px; }')
        self.anipose_calibration_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='anipose_calibration_cb_bool'))
        self.anipose_calibration_cb.move(940, 100)

        board_provided_cb_label = QLabel('Calibration board provided:', self.ProcessSettings)
        board_provided_cb_label.setFont(QFont(self.font_id, 12))
        board_provided_cb_label.move(740, 130)
        self.board_provided_cb = QComboBox(self.ProcessSettings)
        self.board_provided_cb.addItems(['No', 'Yes'])
        self.board_provided_cb.setStyleSheet('QComboBox { width: 80px; }')
        self.board_provided_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='board_provided_cb_bool'))
        self.board_provided_cb.move(940, 130)

        board_xy_label = QLabel('Calibration board XY:', self.ProcessSettings)
        board_xy_label.setFont(QFont(self.font_id, 12))
        board_xy_label.move(740, 160)
        self.board_xy = QLineEdit('8,11', self.ProcessSettings)
        self.board_xy.setFont(QFont(self.font_id, 10))
        self.board_xy.setStyleSheet('QLineEdit { width: 108px; }')
        self.board_xy.move(940, 160)

        square_len_label = QLabel('Calibration square length:', self.ProcessSettings)
        square_len_label.setFont(QFont(self.font_id, 12))
        square_len_label.move(740, 190)
        self.square_len = QLineEdit('16.25', self.ProcessSettings)
        self.square_len.setFont(QFont(self.font_id, 10))
        self.square_len.setStyleSheet('QLineEdit { width: 108px; }')
        self.square_len.move(940, 190)

        marker_len_bits_label = QLabel('Calibration markers (bits):', self.ProcessSettings)
        marker_len_bits_label.setFont(QFont(self.font_id, 12))
        marker_len_bits_label.move(740, 220)
        self.marker_len_bits = QLineEdit('12.75,4', self.ProcessSettings)
        self.marker_len_bits.setFont(QFont(self.font_id, 10))
        self.marker_len_bits.setStyleSheet('QLineEdit { width: 108px; }')
        self.marker_len_bits.move(940, 220)

        dict_size_label = QLabel('Dict size:', self.ProcessSettings)
        dict_size_label.setFont(QFont(self.font_id, 12))
        dict_size_label.move(740, 250)
        self.dict_size = QLineEdit('1000', self.ProcessSettings)
        self.dict_size.setFont(QFont(self.font_id, 10))
        self.dict_size.setStyleSheet('QLineEdit { width: 108px; }')
        self.dict_size.move(940, 250)

        img_width_height_label = QLabel('Image width / height:', self.ProcessSettings)
        img_width_height_label.setFont(QFont(self.font_id, 12))
        img_width_height_label.move(740, 280)
        self.img_width_height = QLineEdit('2100,2970', self.ProcessSettings)
        self.img_width_height.setFont(QFont(self.font_id, 10))
        self.img_width_height.setStyleSheet('QLineEdit { width: 108px; }')
        self.img_width_height.move(940, 280)

        anipose_triangulation_cb_label = QLabel('Conduct AP triangulation:', self.ProcessSettings)
        anipose_triangulation_cb_label.setFont(QFont(self.font_id, 12))
        anipose_triangulation_cb_label.setStyleSheet('QLabel { color: #F58025; }')
        anipose_triangulation_cb_label.move(740, 310)
        self.anipose_triangulation_cb = QComboBox(self.ProcessSettings)
        self.anipose_triangulation_cb.addItems(['No', 'Yes'])
        self.anipose_triangulation_cb.setStyleSheet('QComboBox { width: 80px; }')
        self.anipose_triangulation_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='anipose_triangulation_cb_bool'))
        self.anipose_triangulation_cb.move(940, 310)

        self.calibration_file_loc_edit = QLineEdit('', self.ProcessSettings)
        self.calibration_file_loc_edit.setPlaceholderText('tracking calibration root directory')
        self.calibration_file_loc_edit.setFont(QFont(self.font_id, 10))
        self.calibration_file_loc_edit.setStyleSheet('QLineEdit { width: 220px; }')
        self.calibration_file_loc_edit.move(740, 340)
        calibration_file_loc_btn = QPushButton('Browse', self.ProcessSettings)
        calibration_file_loc_btn.setFont(QFont(self.font_id, 8))
        calibration_file_loc_btn.move(965, 340)
        calibration_file_loc_btn.setStyleSheet('QPushButton { min-width: 64px; min-height: 12px; max-width: 64px; max-height: 12px; }')
        calibration_file_loc_btn.clicked.connect(self._open_anipose_calibration_dialog)

        triangulate_arena_points_cb_label = QLabel('Triangulate arena nodes:', self.ProcessSettings)
        triangulate_arena_points_cb_label.setFont(QFont(self.font_id, 12))
        triangulate_arena_points_cb_label.move(740, 370)
        self.triangulate_arena_points_cb = QComboBox(self.ProcessSettings)
        self.triangulate_arena_points_cb.addItems(['No', 'Yes'])
        self.triangulate_arena_points_cb.setStyleSheet('QComboBox { width: 80px; }')
        self.triangulate_arena_points_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='triangulate_arena_points_cb_bool'))
        self.triangulate_arena_points_cb.move(940, 370)

        display_progress_cb_label = QLabel('Display progress:', self.ProcessSettings)
        display_progress_cb_label.setFont(QFont(self.font_id, 12))
        display_progress_cb_label.move(740, 400)
        self.display_progress_cb = QComboBox(self.ProcessSettings)
        self.display_progress_cb.addItems(['No', 'Yes'])
        self.display_progress_cb.setStyleSheet('QComboBox { width: 80px; }')
        self.display_progress_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='display_progress_cb_bool'))
        self.display_progress_cb.move(940, 400)

        frame_restriction_label = QLabel('Frame restriction:', self.ProcessSettings)
        frame_restriction_label.setFont(QFont(self.font_id, 12))
        frame_restriction_label.move(740, 430)
        self.frame_restriction = QLineEdit('', self.ProcessSettings)
        self.frame_restriction.setFont(QFont(self.font_id, 10))
        self.frame_restriction.setStyleSheet('QLineEdit { width: 108px; }')
        self.frame_restriction.move(940, 430)

        excluded_views_label = QLabel('Excluded camera views:', self.ProcessSettings)
        excluded_views_label.setFont(QFont(self.font_id, 12))
        excluded_views_label.move(740, 460)
        self.excluded_views = QLineEdit('', self.ProcessSettings)
        self.excluded_views.setFont(QFont(self.font_id, 10))
        self.excluded_views.setStyleSheet('QLineEdit { width: 108px; }')
        self.excluded_views.move(940, 460)

        rigid_body_constraints_label = QLabel('Rigid body constraints:', self.ProcessSettings)
        rigid_body_constraints_label.setFont(QFont(self.font_id, 12))
        rigid_body_constraints_label.move(740, 490)
        self.rigid_body_constraints = QLineEdit('', self.ProcessSettings)
        self.rigid_body_constraints.setFont(QFont(self.font_id, 10))
        self.rigid_body_constraints.setStyleSheet('QLineEdit { width: 108px; }')
        self.rigid_body_constraints.move(940, 490)

        weak_body_constraints_label = QLabel('Weak body constraints:', self.ProcessSettings)
        weak_body_constraints_label.setFont(QFont(self.font_id, 12))
        weak_body_constraints_label.move(740, 520)
        self.weak_body_constraints = QLineEdit('', self.ProcessSettings)
        self.weak_body_constraints.setFont(QFont(self.font_id, 10))
        self.weak_body_constraints.setStyleSheet('QLineEdit { width: 108px; }')
        self.weak_body_constraints.move(940, 520)

        scale_smooth_len_weak_label = QLabel('Scale smooth / length:', self.ProcessSettings)
        scale_smooth_len_weak_label.setFont(QFont(self.font_id, 12))
        scale_smooth_len_weak_label.move(740, 550)
        self.scale_smooth_len_weak = QLineEdit('3,4,1', self.ProcessSettings)
        self.scale_smooth_len_weak.setFont(QFont(self.font_id, 10))
        self.scale_smooth_len_weak.setStyleSheet('QLineEdit { width: 108px; }')
        self.scale_smooth_len_weak.move(940, 550)

        reprojection_error_loss_label = QLabel('Reproject error / loss:', self.ProcessSettings)
        reprojection_error_loss_label.setFont(QFont(self.font_id, 12))
        reprojection_error_loss_label.move(740, 580)
        self.reprojection_error_loss = QLineEdit("5,'l2'", self.ProcessSettings)
        self.reprojection_error_loss.setFont(QFont(self.font_id, 10))
        self.reprojection_error_loss.setStyleSheet('QLineEdit { width: 108px; }')
        self.reprojection_error_loss.move(940, 580)

        n_deriv_smooth_label = QLabel('Derivation smoothing (n):', self.ProcessSettings)
        n_deriv_smooth_label.setFont(QFont(self.font_id, 12))
        n_deriv_smooth_label.move(740, 580)
        self.n_deriv_smooth = QLineEdit('2', self.ProcessSettings)
        self.n_deriv_smooth.setFont(QFont(self.font_id, 10))
        self.n_deriv_smooth.setStyleSheet('QLineEdit { width: 108px; }')
        self.n_deriv_smooth.move(940, 580)

        translate_rotate_metric_label = QLabel('Conduct coordinate change:', self.ProcessSettings)
        translate_rotate_metric_label.setFont(QFont(self.font_id, 12))
        translate_rotate_metric_label.setStyleSheet('QLabel { color: #F58025; }')
        translate_rotate_metric_label.move(740, 610)
        self.translate_rotate_metric_cb = QComboBox(self.ProcessSettings)
        self.translate_rotate_metric_cb.addItems(['No', 'Yes'])
        self.translate_rotate_metric_cb.setStyleSheet('QComboBox { width: 80px; }')
        self.translate_rotate_metric_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='translate_rotate_metric_cb_bool'))
        self.translate_rotate_metric_cb.move(940, 610)

        self.original_arena_file_loc_edit = QLineEdit('', self.ProcessSettings)
        self.original_arena_file_loc_edit.setPlaceholderText('points3d.h5 arena root directory')
        self.original_arena_file_loc_edit.setFont(QFont(self.font_id, 10))
        self.original_arena_file_loc_edit.setStyleSheet('QLineEdit { width: 220px; }')
        self.original_arena_file_loc_edit.move(740, 640)
        original_arena_file_loc_btn = QPushButton('Browse', self.ProcessSettings)
        original_arena_file_loc_btn.setFont(QFont(self.font_id, 8))
        original_arena_file_loc_btn.move(965, 640)
        original_arena_file_loc_btn.setStyleSheet('QPushButton { min-width: 64px; min-height: 12px; max-width: 64px; max-height: 12px; }')
        original_arena_file_loc_btn.clicked.connect(self._open_original_arena_dialog)

        save_arena_data_cb_label = QLabel('Save arena transformation:', self.ProcessSettings)
        save_arena_data_cb_label.setFont(QFont(self.font_id, 12))
        save_arena_data_cb_label.move(740, 670)
        self.save_arena_data_cb = QComboBox(self.ProcessSettings)
        self.save_arena_data_cb.addItems(['No', 'Yes'])
        self.save_arena_data_cb.setStyleSheet('QComboBox { width: 80px; }')
        self.save_arena_data_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='save_arena_data_cb_bool'))
        self.save_arena_data_cb.move(940, 670)

        save_mouse_data_cb_label = QLabel('Save mouse transformation:', self.ProcessSettings)
        save_mouse_data_cb_label.setFont(QFont(self.font_id, 12))
        save_mouse_data_cb_label.move(740, 700)
        self.save_mouse_data_cb = QComboBox(self.ProcessSettings)
        self.save_mouse_data_cb.addItems(['Yes', 'No'])
        self.save_mouse_data_cb.setStyleSheet('QComboBox { width: 80px; }')
        self.save_mouse_data_cb.activated.connect(partial(self._combo_box_prior_true, variable_id='save_mouse_data_cb_bool'))
        self.save_mouse_data_cb.move(940, 700)

        delete_original_h5_cb_label = QLabel('Delete original .h5:', self.ProcessSettings)
        delete_original_h5_cb_label.setFont(QFont(self.font_id, 12))
        delete_original_h5_cb_label.move(740, 730)
        self.delete_original_h5_cb = QComboBox(self.ProcessSettings)
        self.delete_original_h5_cb.addItems(['Yes', 'No'])
        self.delete_original_h5_cb.setStyleSheet('QComboBox { width: 80px; }')
        self.delete_original_h5_cb.activated.connect(partial(self._combo_box_prior_true, variable_id='delete_original_h5_cb_bool'))
        self.delete_original_h5_cb.move(940, 730)

        static_reference_len_label = QLabel('Static reference length (m):', self.ProcessSettings)
        static_reference_len_label.setFont(QFont(self.font_id, 12))
        static_reference_len_label.move(740, 760)
        self.static_reference_len = QLineEdit('0.615', self.ProcessSettings)
        self.static_reference_len.setFont(QFont(self.font_id, 10))
        self.static_reference_len.setStyleSheet('QLineEdit { width: 108px; }')
        self.static_reference_len.move(940, 760)

        usv_detection_label = QLabel('Vocalization detection', self.ProcessSettings)
        usv_detection_label.setFont(QFont(self.font_id, 13))
        usv_detection_label.setStyleSheet('QLabel { font-weight: bold;}')
        usv_detection_label.move(740, 800)

        das_inference_cb_label = QLabel('Conduct inference (serial):', self.ProcessSettings)
        das_inference_cb_label.setFont(QFont(self.font_id, 12))
        das_inference_cb_label.setStyleSheet('QLabel { color: #F58025; }')
        das_inference_cb_label.move(740, 830)
        self.das_inference_cb = QComboBox(self.ProcessSettings)
        self.das_inference_cb.addItems(['No', 'Yes'])
        self.das_inference_cb.setStyleSheet('QComboBox { width: 80px; }')
        self.das_inference_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='das_inference_cb_bool'))
        self.das_inference_cb.move(940, 830)

        das_conda_label = QLabel('DAS conda env name:', self.ProcessSettings)
        das_conda_label.setFont(QFont(self.font_id, 12))
        das_conda_label.move(740, 860)
        self.das_conda = QLineEdit('das', self.ProcessSettings)
        self.das_conda.setFont(QFont(self.font_id, 10))
        self.das_conda.setStyleSheet('QLineEdit { width: 108px; }')
        self.das_conda.move(940, 860)

        self.das_model_dir_edit = QLineEdit('', self.ProcessSettings)
        self.das_model_dir_edit.setPlaceholderText('trained DAS model directory')
        self.das_model_dir_edit.setFont(QFont(self.font_id, 10))
        self.das_model_dir_edit.setStyleSheet('QLineEdit { width: 220px; }')
        self.das_model_dir_edit.move(740, 890)
        das_model_dir_btn = QPushButton('Browse', self.ProcessSettings)
        das_model_dir_btn.setFont(QFont(self.font_id, 8))
        das_model_dir_btn.move(965, 890)
        das_model_dir_btn.setStyleSheet('QPushButton { min-width: 64px; min-height: 12px; max-width: 64px; max-height: 12px; }')
        das_model_dir_btn.clicked.connect(self._open_das_model_dialog)

        das_model_base_label = QLabel('DAS model base:', self.ProcessSettings)
        das_model_base_label.setFont(QFont(self.font_id, 12))
        das_model_base_label.move(740, 920)
        self.das_model_base = QLineEdit('', self.ProcessSettings)
        self.das_model_base.setFont(QFont(self.font_id, 10))
        self.das_model_base.setStyleSheet('QLineEdit { width: 108px; }')
        self.das_model_base.move(940, 920)

        das_output_type_label = QLabel('Inference output file type:', self.ProcessSettings)
        das_output_type_label.setFont(QFont(self.font_id, 12))
        das_output_type_label.move(740, 950)
        self.das_output_type = QLineEdit('csv', self.ProcessSettings)
        self.das_output_type.setFont(QFont(self.font_id, 10))
        self.das_output_type.setStyleSheet('QLineEdit { width: 108px; }')
        self.das_output_type.move(940, 950)

        das_segment_tmf_label = QLabel('USV segment cutoff params:', self.ProcessSettings)
        das_segment_tmf_label.setFont(QFont(self.font_id, 12))
        das_segment_tmf_label.move(740, 980)
        self.das_segment_tmf = QLineEdit('0.5,0.015,0.015', self.ProcessSettings)
        self.das_segment_tmf.setFont(QFont(self.font_id, 10))
        self.das_segment_tmf.setStyleSheet('QLineEdit { width: 108px; }')
        self.das_segment_tmf.move(940, 980)

        self._create_buttons_process(seq=0, class_option=self.ProcessSettings,
                                     button_pos_y=record_four_y - 35, next_button_x_pos=record_four_x - 100)

    def process_two(self):
        self.ConductProcess = ConductProcess(self)
        self.setWindowTitle(f'{app_name} (Conduct Processing)')
        self.setCentralWidget(self.ConductProcess)
        record_four_x, record_four_y = (820, 1100)
        self.setFixedSize(record_four_x, record_four_y)

        self.txt_edit_process = QPlainTextEdit(self.ConductProcess)
        self.txt_edit_process.move(5, 5)
        self.txt_edit_process.setFixedSize(805, 1040)
        self.txt_edit_process.setReadOnly(True)

        self._save_modified_values_to_toml(run_exp_bool=False, message_func=self._process_message)

        exp_settings_dict_final = toml.load(f"{self.processing_input_dict['send_email']['Messenger']['toml_file_loc']}{os.sep}behavioral_experiments_settings.toml")

        self.run_processing = Stylist(message_output=self._process_message,
                                      input_parameter_dict=self.processing_input_dict,
                                      root_directories=self.processing_input_dict['preprocess_data']['root_directories'],
                                      exp_settings_dict=exp_settings_dict_final)

        self._create_buttons_process(seq=1, class_option=self.ConductProcess,
                                     button_pos_y=record_four_y - 35, next_button_x_pos=record_four_x - 100)

    def _save_modified_values_to_toml(self, run_exp_bool=True, message_func=None):
        self.exp_settings_dict = toml.load(f"{self.settings_dict['general']['config_settings_directory']}{os.sep}behavioral_experiments_settings.toml")

        if self.exp_settings_dict['config_settings_directory'] != self.settings_dict['general']['config_settings_directory']:
            self.exp_settings_dict['config_settings_directory'] = self.settings_dict['general']['config_settings_directory']

        if run_exp_bool:
            audio_config_temp = configparser.ConfigParser()
            audio_config_temp.read(f"{self.settings_dict['general']['config_settings_directory']}"
                                   f"{os.sep}avisoft_config.ini")

            if audio_config_temp['Configuration']['basedirectory'] != self.settings_dict['general']['avisoft_basedirectory'] \
                    or audio_config_temp['Configuration']['configfilename'] != f"{self.settings_dict['general']['avisoft_basedirectory']}Configurations{os.sep}RECORDER_USGH{os.sep}avisoft_config.ini":
                self.modify_audio_config = True

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
                elif video_key in self.exp_settings_dict['video']['metadata'].keys():
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

        message_func(f"Updating the configuration .toml file completed at: "
                     f"{datetime.datetime.now().hour:02d}:{datetime.datetime.now().minute:02d}.{datetime.datetime.now().second:02d}.")

    def _save_process_labels_func(self):
        qlabel_strings = ['concatenate_video_ext', 'concatenated_video_name', 'conversion_target_file', 'conversion_vid_ext',
                          'constant_rate_factor', 'encoding_preset', 'ch_receiving_input', 'audio_filter_format',
                          'a_ch_receiving_input', 'pc_usage_process', 'v_millisecond_divergence_tolerance', 'min_spike_num',
                          'v_relative_intensity_threshold', 'v_video_extension', 'v_led_px_dev', 'v_led_px_version',
                          'phidget_extra_data_camera', 'audio_format', 'npx_ms_divergence_tolerance', 'hpss_power', 'sleap_conda',
                          'square_len', 'dict_size', 'n_deriv_smooth', 'static_reference_len', 'das_conda', 'das_model_base',
                          'das_output_type']
        lists_in_string = ['concatenate_cam_serial_num', 'change_fps_cam_serial_num', 'v_camera_serial_num', 'filter_dirs', 'concat_dirs',
                           'stft_window_hop', 'hpss_kernel_size', 'hpss_margin', 'filter_freq_bounds', 'board_xy', 'marker_len_bits',
                           'img_width_height', 'frame_restriction', 'excluded_views', 'rigid_body_constraints', 'weak_body_constraints',
                           'scale_smooth_len_weak', 'reprojection_error_loss', 'das_segment_tmf']

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

        self.processing_input_dict['anipose_operations']['ConvertTo3D']['translate_rotate_metric']['original_arena_file_loc'] = self.original_arena_file_loc_edit.text()
        self.processing_input_dict['anipose_operations']['ConvertTo3D']['conduct_anipose_triangulation']['calibration_file_loc'] = self.calibration_file_loc_edit.text()
        self.processing_input_dict['usv_inference']['FindMouseVocalizations']['das_command_line_inference']['model_directory'] = self.das_model_dir_edit.text()

        self.processing_input_dict['synchronize_files']['Synchronizer']['crop_wav_files_to_video']['device_receiving_input'] = str(getattr(self, 'device_receiving_input'))
        self.device_receiving_input = 'm'

        self.processing_input_dict['synchronize_files']['Synchronizer']['validate_ephys_video_sync']['npx_file_type'] = str(getattr(self, 'npx_file_type'))
        self.npx_file_type = 'ap'

        self.processing_input_dict['file_manipulation']['Operator']['get_spike_times']['kilosort_version'] = str(getattr(self, 'kilosort_version'))
        self.kilosort_version = '4'

        self.processing_input_dict['file_manipulation']['Operator']['concatenate_video_files']['video_extension'] = self.concatenate_video_ext
        self.processing_input_dict['file_manipulation']['Operator']['concatenate_video_files']['concatenated_video_name'] = self.concatenated_video_name
        self.processing_input_dict['file_manipulation']['Operator']['rectify_video_fps']['conversion_target_file'] = self.conversion_target_file
        self.processing_input_dict['file_manipulation']['Operator']['rectify_video_fps']['video_extension'] = self.conversion_vid_ext
        self.processing_input_dict['file_manipulation']['Operator']['rectify_video_fps']['constant_rate_factor'] = int(round(ast.literal_eval(self.constant_rate_factor)))
        self.processing_input_dict['file_manipulation']['Operator']['rectify_video_fps']['encoding_preset'] = self.encoding_preset
        self.processing_input_dict['synchronize_files']['Synchronizer']['crop_wav_files_to_video']['ch_receiving_input'] = int(ast.literal_eval(self.ch_receiving_input))
        self.processing_input_dict['file_manipulation']['Operator']['concatenate_audio_files']['audio_format'] = self.audio_format
        self.processing_input_dict['file_manipulation']['Operator']['filter_audio_files']['audio_format'] = self.audio_filter_format
        self.processing_input_dict['file_manipulation']['Operator']['filter_audio_files']['filter_freq_bounds'] = [int(ast.literal_eval(freq_bound)) for freq_bound in self.filter_freq_bounds]
        self.processing_input_dict['file_manipulation']['Operator']['hpss_audio']['stft_window_length_hop_size'] = [int(ast.literal_eval(stft_value)) for stft_value in self.stft_window_hop]
        self.processing_input_dict['file_manipulation']['Operator']['hpss_audio']['kernel_size'] = tuple([int(ast.literal_eval(kernel_value)) for kernel_value in self.hpss_kernel_size])
        self.processing_input_dict['file_manipulation']['Operator']['hpss_audio']['hpss_power'] = float(ast.literal_eval(self.hpss_power))
        self.processing_input_dict['file_manipulation']['Operator']['hpss_audio']['margin'] = tuple([int(ast.literal_eval(margin_value)) for margin_value in self.hpss_margin])
        self.processing_input_dict['file_manipulation']['Operator']['get_spike_times']['min_spike_num'] = int(ast.literal_eval(self.min_spike_num))
        self.processing_input_dict['synchronize_files']['Synchronizer']['find_audio_sync_trains']['ch_receiving_input'] = int(ast.literal_eval(self.a_ch_receiving_input))
        self.processing_input_dict['synchronize_files']['Synchronizer']['find_video_sync_trains']['led_px_version'] = self.v_led_px_version
        self.processing_input_dict['synchronize_files']['Synchronizer']['find_video_sync_trains']['led_px_dev'] = int(ast.literal_eval(self.v_led_px_dev))
        self.processing_input_dict['synchronize_files']['Synchronizer']['find_video_sync_trains']['video_extension'] = self.v_video_extension
        self.processing_input_dict['synchronize_files']['Synchronizer']['find_video_sync_trains']['relative_intensity_threshold'] = float(ast.literal_eval(self.v_relative_intensity_threshold))
        self.processing_input_dict['synchronize_files']['Synchronizer']['find_video_sync_trains']['millisecond_divergence_tolerance'] = int(ast.literal_eval(self.v_millisecond_divergence_tolerance))
        self.processing_input_dict['synchronize_files']['Synchronizer']['validate_ephys_video_sync']['npx_ms_divergence_tolerance'] = float(ast.literal_eval(self.npx_ms_divergence_tolerance))
        self.processing_input_dict['extract_phidget_data']['Gatherer']['prepare_data_for_analyses']['extra_data_camera'] = self.phidget_extra_data_camera

        self.processing_input_dict['preprocess_data']['root_directories'] = self.processing_dir_edit
        self.processing_input_dict['file_manipulation']['Operator']['concatenate_video_files']['camera_serial_num'] = self.concatenate_cam_serial_num
        self.processing_input_dict['file_manipulation']['Operator']['rectify_video_fps']['camera_serial_num'] = self.change_fps_cam_serial_num
        self.processing_input_dict['send_email']['Messenger']['send_message']['receivers'] = self.pc_usage_process
        self.processing_input_dict['synchronize_files']['Synchronizer']['find_video_sync_trains']['camera_serial_num'] = self.v_camera_serial_num
        self.processing_input_dict['file_manipulation']['Operator']['filter_audio_files']['filter_dirs'] = self.filter_dirs
        self.processing_input_dict['file_manipulation']['Operator']['concatenate_audio_files']['concat_dirs'] = self.concat_dirs

        self.processing_input_dict['usv_inference']['FindMouseVocalizations']['das_command_line_inference']['das_conda_env_name'] = self.das_conda
        self.processing_input_dict['usv_inference']['FindMouseVocalizations']['das_command_line_inference']['model_name_base'] = self.das_model_base
        self.processing_input_dict['usv_inference']['FindMouseVocalizations']['das_command_line_inference']['output_file_type'] = self.das_output_type
        self.processing_input_dict['usv_inference']['FindMouseVocalizations']['das_command_line_inference']['segment_tmf'] = [float(ast.literal_eval(tmf_value)) for tmf_value in self.das_segment_tmf]

        self.processing_input_dict['anipose_operations']['ConvertTo3D']['sleap_file_conversion']['sleap_conda_env_name'] = self.sleap_conda
        self.processing_input_dict['anipose_operations']['ConvertTo3D']['conduct_anipose_calibration']['board_xy'] = [ast.literal_eval(xy_value) for xy_value in self.board_xy]
        self.processing_input_dict['anipose_operations']['ConvertTo3D']['conduct_anipose_calibration']['square_len'] = float(ast.literal_eval(self.square_len))
        self.processing_input_dict['anipose_operations']['ConvertTo3D']['conduct_anipose_calibration']['marker_len_bits'] = [ast.literal_eval(mlb_value) for mlb_value in self.marker_len_bits]
        self.processing_input_dict['anipose_operations']['ConvertTo3D']['conduct_anipose_calibration']['dict_size'] = ast.literal_eval(self.dict_size)
        self.processing_input_dict['anipose_operations']['ConvertTo3D']['conduct_anipose_calibration']['img_width_height'] = [ast.literal_eval(iwh_value) for iwh_value in self.img_width_height]
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
        self.processing_input_dict['anipose_operations']['ConvertTo3D']['conduct_anipose_triangulation']['scale_smooth_len_weak'] = [ast.literal_eval(sslw_value) for sslw_value in self.scale_smooth_len_weak]
        self.processing_input_dict['anipose_operations']['ConvertTo3D']['conduct_anipose_triangulation']['reprojection_error_loss'] = [ast.literal_eval(rel_value) for rel_value in self.reprojection_error_loss]
        self.processing_input_dict['anipose_operations']['ConvertTo3D']['conduct_anipose_triangulation']['n_deriv_smooth'] = int(ast.literal_eval(self.n_deriv_smooth))
        self.processing_input_dict['anipose_operations']['ConvertTo3D']['translate_rotate_metric']['static_reference_len'] = float(ast.literal_eval(self.static_reference_len))

        self.processing_input_dict['processing_booleans']['conduct_video_concatenation'] = self.conduct_video_concatenation_cb_bool
        self.conduct_video_concatenation_cb_bool = False
        self.processing_input_dict['processing_booleans']['conduct_video_fps_change'] = self.conduct_video_fps_change_cb_bool
        self.conduct_video_fps_change_cb_bool = False
        self.processing_input_dict['file_manipulation']['Operator']['rectify_video_fps']['delete_old_file'] = self.delete_con_file_cb_bool
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
        self.display_progress_cb_bool = False
        self.processing_input_dict['processing_booleans']['anipose_trm'] = self.translate_rotate_metric_cb_bool
        self.translate_rotate_metric_cb_bool = False
        self.processing_input_dict['anipose_operations']['ConvertTo3D']['translate_rotate_metric']['save_arena_data_bool'] = self.save_arena_data_cb_bool
        self.save_arena_data_cb_bool = False
        self.processing_input_dict['anipose_operations']['ConvertTo3D']['translate_rotate_metric']['save_mouse_data_bool'] = self.save_mouse_data_cb_bool
        self.save_mouse_data_cb_bool = True
        self.processing_input_dict['anipose_operations']['ConvertTo3D']['translate_rotate_metric']['delete_original_h5'] = self.delete_original_h5_cb_bool
        self.delete_original_h5_cb_bool = True
        self.processing_input_dict['processing_booleans']['das_infer'] = self.das_inference_cb_bool
        self.das_inference_cb_bool = False

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
        for variable in self.default_audio_settings.keys():
            self.settings_dict['audio'][f'{variable}'] = getattr(self, variable).text()

    def _save_record_three_labels_func(self):
        video_dict_keys = ['browser', 'expected_cameras', 'recording_codec', 'specific_camera_serial',
                           'experimenter', 'mice_num', 'cage_ID_m1', 'mouse_ID_m1', 'genotype_m1', 'sex_m1', 'DOB_m1',
                           'housing_m1', 'cage_ID_m2', 'mouse_ID_m2', 'genotype_m2', 'sex_m2', 'DOB_m2', 'housing_m2', 'other']

        self.settings_dict['video']['monitor_recording'] = self.monitor_recording_cb_bool
        self.monitor_recording_cb_bool = False
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
                if variable == 'recording_codec':
                    self.settings_dict['video'][variable] = str(getattr(self, variable))
                    self.recording_codec = 'hq'
                elif variable == 'other':
                    self.settings_dict['video'][variable] = getattr(self, variable).toPlainText()
                else:
                    self.settings_dict['video'][variable] = getattr(self, variable).text()
            else:
                self.expected_cameras = self.expected_cameras.text()
                self.settings_dict['video'][variable] = self.expected_cameras.split(',')

    def _combo_box_kilosort_version(self, index, variable_id=None):
        if index == 0:
            self.__dict__[variable_id] = '4'
        elif index == 1:
            self.__dict__[variable_id] = '3'
        elif index == 2:
            self.__dict__[variable_id] = '2.5'
        else:
            self.__dict__[variable_id] = '2'

    def _combo_box_prior_codec(self, index, variable_id=None):
        if index == 0:
            self.__dict__[variable_id] = 'hq'
        elif index == 1:
            self.__dict__[variable_id] = 'mq'
        else:
            self.__dict__[variable_id] = 'lq'

    def _combo_box_prior_audio_device_camera_input(self, index, variable_id=None):
        if index == 0:
            self.__dict__[variable_id] = 'm'
        else:
            self.__dict__[variable_id] = 's'

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
                           'Record': QPushButton(QIcon(record_icon), 'Record', self.Main)}

        self.button_map['Record'].move(120, 370)
        self.button_map['Record'].setFont(QFont(self.font_id, 8))
        self.button_map['Record'].clicked.connect(self.record_one)

        self.button_map['Process'].move(215, 370)
        self.button_map['Process'].setFont(QFont(self.font_id, 8))
        self.button_map['Process'].clicked.connect(self.process_one)

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
            if self.settings_dict['general']['conduct_tracking_calibration']:
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
        if self.settings_dict['general']['conduct_tracking_calibration']:
            self.button_map['Calibrate'].setEnabled(False)

    def _open_settings_dialog(self):
        # TODO: check whether this does what it is supposed to, doesn't seem to change directories
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

    def _open_anipose_calibration_dialog(self):
        anipose_cal_dir_name = QFileDialog.getExistingDirectory(
            self,
            'Select Anipose calibration root directory',
            '')
        if anipose_cal_dir_name:
            anipose_cal_dir_name_path = Path(anipose_cal_dir_name)
            self.calibration_file_loc_edit.setText(str(anipose_cal_dir_name_path))
            if os.name == 'nt':
                self.processing_input_dict['anipose_operations']['ConvertTo3D']['conduct_anipose_triangulation']['calibration_file_loc'] = str(anipose_cal_dir_name_path).replace(os.sep, '\\')
            else:
                self.processing_input_dict['anipose_operations']['ConvertTo3D']['conduct_anipose_triangulation']['calibration_file_loc'] = str(anipose_cal_dir_name_path)
        else:
            self.processing_input_dict['anipose_operations']['ConvertTo3D']['conduct_anipose_triangulation']['calibration_file_loc'] = ''

    def _open_original_arena_dialog(self):
        original_arena_dir_name = QFileDialog.getExistingDirectory(
            self,
            'Select arena *points3d.h5 root directory',
            '')
        if original_arena_dir_name:
            original_aren_dir_name_path = Path(original_arena_dir_name)
            self.original_arena_file_loc_edit.setText(str(original_aren_dir_name_path))
            if os.name == 'nt':
                self.processing_input_dict['anipose_operations']['ConvertTo3D']['translate_rotate_metric']['original_arena_file_loc'] = str(original_aren_dir_name_path).replace(os.sep, '\\')
            else:
                self.processing_input_dict['anipose_operations']['ConvertTo3D']['translate_rotate_metric']['original_arena_file_loc'] = str(original_aren_dir_name_path)
        else:
            self.processing_input_dict['anipose_operations']['ConvertTo3D']['translate_rotate_metric']['original_arena_file_loc'] = ''

    def _open_das_model_dialog(self):
        das_model_dir_name = QFileDialog.getExistingDirectory(
            self,
            'Select arena *points3d.h5 root directory',
            '')
        if das_model_dir_name:
            das_model_dir_name_path = Path(das_model_dir_name)
            self.das_model_dir_edit.setText(str(das_model_dir_name_path))
            if os.name == 'nt':
                self.processing_input_dict['usv_inference']['FindMouseVocalizations']['das_command_line_inference']['model_directory'] = str(das_model_dir_name_path).replace(os.sep, '\\')
            else:
                self.processing_input_dict['usv_inference']['FindMouseVocalizations']['das_command_line_inference']['model_directory'] = str(das_model_dir_name_path)
        else:
            self.processing_input_dict['usv_inference']['FindMouseVocalizations']['das_command_line_inference']['model_directory'] = ''

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
    with open('gui_style_sheet.css', 'r') as file:
        usv_playpen_app.setStyleSheet(file.read())

    usv_playpen_app.setWindowIcon(QIcon(lab_icon))

    splash = QSplashScreen(QPixmap(splash_icon))
    progress_bar = QProgressBar(splash)
    progress_bar.setGeometry(340, 75, 250, 10)
    splash.show()
    for i in range(0, 101):
        progress_bar.setValue(i)
        t = time.time()
        while time.time() < t + 0.025:
            usv_playpen_app.processEvents()

    initial_values_dict = {'experiment_time_sec': 0, 'monitor_recording_cb_bool': False, 'monitor_specific_camera_cb_bool': False, 'delete_post_copy_cb_bool': True,
                           'conduct_audio_cb_bool': True, 'conduct_tracking_calibration_cb_bool': False, 'modify_audio_config': False, 'conduct_video_concatenation_cb_bool': False,
                           'conduct_video_fps_change_cb_bool': False, 'delete_con_file_cb_bool': True, 'conduct_multichannel_conversion_cb_bool': False, 'crop_wav_cam_cb_bool': False,
                           'conc_audio_cb_bool': False, 'filter_audio_cb_bool': False, 'conduct_sync_cb_bool': False, 'conduct_nv_sync_cb_bool': False, 'recording_codec': 'hq',
                           'npx_file_type': 'ap', 'device_receiving_input': 'm', 'kilosort_version': '4', 'conduct_hpss_cb_bool': False, 'conduct_ephys_file_chaining_cb_bool': False,
                           'split_cluster_spikes_cb_bool': False, 'anipose_calibration_cb_bool': False, 'sleap_file_conversion_cb_bool': False, 'board_provided_cb_bool': False,
                           'anipose_triangulation_cb_bool': False, 'triangulate_arena_points_cb_bool': False, 'display_progress_cb_bool': False, 'translate_rotate_metric_cb_bool': False,
                           'save_arena_data_cb_bool': False, 'save_mouse_data_cb_bool': True, 'delete_original_h5_cb_bool': True, 'das_inference_cb_bool': False}

    usv_playpen_window = USVPlaypenWindow(**initial_values_dict)

    splash.finish(usv_playpen_window)

    usv_playpen_window.show()

    sys.exit(usv_playpen_app.exec())


if __name__ == "__main__":
    main()
