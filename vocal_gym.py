"""
@author: bartulem
GUI to run behavioral experiments.
"""

import os
import sys
import time

from pathlib import Path

from PyQt6.QtCore import (
    QRect,
    Qt
)
from PyQt6.QtGui import (
    QAction,
    QFont,
    QIcon,
    QPixmap
)
from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QFileDialog,
    QGridLayout,
    QLabel,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QSplashScreen,
    QTextEdit,
    QWidget,
)

basedir = os.path.dirname(__file__)
lab_icon = f"{basedir}{os.sep}img{os.sep}lab.png"
splash_icon = f"{basedir}{os.sep}img{os.sep}uncle_stefan-min.png"
process_icon = f"{basedir}{os.sep}img{os.sep}process.png"
record_icon = f"{basedir}{os.sep}img{os.sep}record.png"

window_width = 550
window_height = 350
display_height = 35
button_width = 80
button_height = 30


class Main(QWidget):
    def __init__(self, parent=None):
        super(Main, self).__init__(parent)


class Record(QWidget):
    def __init__(self, parent=None):
        super(Record, self).__init__(parent)


class VocalGymWindow(QMainWindow):
    """Main window (GUI or view)."""

    def __init__(self):
        super().__init__()
        self.textEdit = None
        self.generalLayout = None
        self.Record = None
        self.label = None
        self.Main = None
        self.setFixedSize(window_width, window_height)
        self.setWindowIcon(QIcon(lab_icon))
        self.main_window()

    def main_window(self):
        self.Main = Main(self)
        self.label = QLabel()
        self.setWindowTitle("PNI Vocal Gymnasium")
        self.setCentralWidget(self.Main)
        self.generalLayout = QHBoxLayout()
        self.Main.setLayout(self.generalLayout)
        self.label.setText("Thank you for using the PNI Vocal Gymnasium. "
                           "\n\nTo ensure quality recordings, make sure you install the following (prior to further usage): "
                           "\n\n(1) Avisoft-Recorder USGH ≥4.3.07"
                           "\n(2) Chrome (or other) web browser"
                           "\n(3) FFMPEG (and add it to PATH)"
                           "\n(4) CoolTerm"
                           "\n(4) Sox (and add it to PATH)"
                           "\n\n\n\n\n\n\n\n\n© github/bartulem 2023")
        self.label.setStyleSheet("background-color: #FFFFFF")
        self.label.setGeometry(0, 0, 300, 100)
        self.label.setWordWrap(True)
        self.generalLayout.addWidget(self.label, alignment=Qt.AlignmentFlag.AlignTop)
        self._create_buttons_main()

    def record_one(self):
        self.Record = Record(self)
        self.setWindowTitle("PNI Vocal Gymnasium (Record)")
        self.setCentralWidget(self.Record)
        self.generalLayout = QHBoxLayout()
        self.Record.setLayout(self.generalLayout)
        self._create_buttons_record()

        home_dir = str(Path.home())
        file_name = QFileDialog.getOpenFileName(self, 'Open file', home_dir)

    def _create_buttons_main(self):
        self.button_map = {}
        buttons_layout = QHBoxLayout()

        self.button_map["Process"] = QPushButton(QIcon(process_icon), "Process")
        self.button_map["Process"].setFixedSize(button_width, button_height)
        self.button_map["Process"].setEnabled(False)
        buttons_layout.addWidget(self.button_map["Process"], alignment=Qt.AlignmentFlag.AlignBottom)

        self.button_map["Record"] = QPushButton(QIcon(record_icon), "Record")
        self.button_map["Record"].setFixedSize(button_width, button_height)
        self.button_map["Record"].clicked.connect(self.record_one)
        buttons_layout.addWidget(self.button_map["Record"], alignment=Qt.AlignmentFlag.AlignBottom)

        self.generalLayout.addLayout(buttons_layout)

    def _create_buttons_record(self):
        self.button_map = {}
        buttons_layout = QHBoxLayout()

        self.button_map["Main"] = QPushButton(QIcon(process_icon), "Main")
        self.button_map["Main"].setFixedSize(button_width, button_height)
        self.button_map["Main"].clicked.connect(self.main_window)
        buttons_layout.addWidget(self.button_map["Main"], alignment=Qt.AlignmentFlag.AlignBottom)

        self.button_map["Previous"] = QPushButton(QIcon(process_icon), "Previous")
        self.button_map["Previous"].setFixedSize(button_width, button_height)
        self.button_map["Previous"].clicked.connect(self.main_window)
        buttons_layout.addWidget(self.button_map["Previous"], alignment=Qt.AlignmentFlag.AlignBottom)

        self.button_map["Next"] = QPushButton(QIcon(process_icon), "Next")
        self.button_map["Next"].setFixedSize(button_width, button_height)
        self.button_map["Next"].setEnabled(False)
        buttons_layout.addWidget(self.button_map["Next"], alignment=Qt.AlignmentFlag.AlignBottom)

        self.generalLayout.addLayout(buttons_layout)


def main():
    """Vocal Gym's main function."""
    vocal_gym_app = QApplication([])

    # splash = QSplashScreen(QPixmap(splash_icon))
    # progress_bar = QProgressBar(splash)
    # progress_bar.setGeometry(0, 0, 610, 20)
    # progress_bar.setStyleSheet("QProgressBar::chunk "
    #                            "{"
    #                            "background-color: #F58025;"
    #                            "}")
    # splash.show()
    # for i in range(0, 100):
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
