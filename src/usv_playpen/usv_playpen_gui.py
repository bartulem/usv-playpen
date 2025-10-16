"""
@author: bartulem
GUI to run behavioral experiments, data processing and analyses.
"""

import ast
import configparser
import copy
import ctypes
import json
import numbers
import os
import platform
import sys
from functools import partial
from importlib import metadata
from pathlib import Path
import platformdirs
import re
import toml
import yaml
from PyQt6.QtCore import (
    Qt, QEvent, QRegularExpression
)
from PyQt6.QtGui import (
    QFont,
    QFontDatabase,
    QGuiApplication,
    QIcon,
    QPainter,
    QPixmap,
    QSyntaxHighlighter,
    QTextCharFormat,
    QColor
)
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QCompleter,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSlider,
    QSplashScreen,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from PyQt6.QtTest import QTest
from .analyze_data import Analyst
from .behavioral_experiments import ExperimentController
from .os_utils import configure_path
from .preprocess_data import Stylist
from .visualize_data import Visualizer

# do NOT print logs (debug, warnings, or otherwise) from the qt.qpa.window category.
os.environ["QT_LOGGING_RULES"] = "qt.qpa.window=false"

# automatically scale your UI based on the screen’s DPI (helpful on high-DPI monitors)
os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"

# controls how Windows represents your app in the taskbar, etc. (important for icon)
if os.name == 'nt':
    my_app_id = 'mycompany.myproduct.subproduct.version'
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(my_app_id)

# app_name = f"USV Playpen v{metadata.version('usv-playpen').split('.dev')[0]}"
app_name = "USV Playpen"

basedir = os.path.dirname(__file__)
background_img = f'{basedir}{os.sep}img{os.sep}background_img.png'
gui_icon = f'{basedir}{os.sep}img{os.sep}gui_icon.png'
password_icon = f'{basedir}{os.sep}img{os.sep}password.png'
save_icon = f'{basedir}{os.sep}img{os.sep}save.png'
splash_icon = f'{basedir}{os.sep}img{os.sep}uncle_stefan.png'
process_icon = f'{basedir}{os.sep}img{os.sep}process.png'
record_icon = f'{basedir}{os.sep}img{os.sep}record.png'
analyze_icon = f'{basedir}{os.sep}img{os.sep}analyze.png'
visualize_icon = f'{basedir}{os.sep}img{os.sep}plot.png'
previous_icon = f'{basedir}{os.sep}img{os.sep}previous.png'
next_icon = f'{basedir}{os.sep}img{os.sep}next.png'
main_icon = f'{basedir}{os.sep}img{os.sep}main.png'
calibrate_icon = f'{basedir}{os.sep}img{os.sep}calibrate.png'
add_icon = f'{basedir}{os.sep}img{os.sep}add.png'
remove_icon = f'{basedir}{os.sep}img{os.sep}remove.png'
accept_icon = f'{basedir}{os.sep}img{os.sep}accept.png'
cancel_icon = f'{basedir}{os.sep}img{os.sep}cancel.png'
clear_icon = f'{basedir}{os.sep}img{os.sep}clear.png'

# Custom Dumper to format lists in flow style (e.g., [1, 2, 3])
# while keeping dictionaries in block style for overall readability.
class SmartDumper(yaml.Dumper):
    def represent_list(self, data):
        is_simple_list = all(isinstance(item, (str, numbers.Number, bool)) or item is None for item in data)

        if is_simple_list:
            return self.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)
        else:
            return self.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=False)

SmartDumper.add_representer(list, SmartDumper.represent_list)

class YamlHighlighter(QSyntaxHighlighter):
    """
    A syntax highlighter for basic YAML files that colors keys.
    """
    def __init__(self, parent=None):
        super(YamlHighlighter, self).__init__(parent)

        self.highlighting_rules = []

        key_format = QTextCharFormat()
        key_format.setForeground(QColor("#F58025"))
        key_format.setFontWeight(QFont.Weight.Bold)

        key_pattern = QRegularExpression(r"^\s*[A-Za-z_][A-Za-z0-9_]*:")
        self.highlighting_rules.append((key_pattern, key_format))

    def highlightBlock(self, text: str) -> None:
        """
        This method is called by Qt for each line of text to apply formatting.

        Parameters
        ----------
        ----------

        Returns
        -------
        -------
        """
        for pattern, format in self.highlighting_rules:
            match_iterator = pattern.globalMatch(text)
            while match_iterator.hasNext():
                match = match_iterator.next()
                self.setFormat(match.capturedStart(), match.capturedLength(), format)


class ChemoDialog(QDialog):
    def __init__(self, parent=None, subject: dict = None):
        super().__init__(parent)
        self.parent_window = parent
        self.subject = subject
        self.is_edit_mode = (subject is not None)
        self.intervention_key = 'chemogenetics'

        self.setWindowTitle("Edit Chemogenetic Intervention" if self.is_edit_mode else "Add Chemogenetic Intervention")
        self.setMinimumWidth(500)

        placeholders = {
            'virus_vendor': 'addgene',
            'virus_name': 'pAAV-CaMKIIa-hM4D(Gi)-mCherry',
            'virus_concentration': 'e.g., 1.0e12 vg/mL',
            'virus_injection_date': 'YYYY-MM-DD',
            'target_area': 'lPAG',
            'agonist_vendor': 'HelloBio',
            'agonist_name': 'CNO',
            'agonist_concentration': 'e.g., 5',
            'agonist_injection_time': 'HH:MM'
        }

        layout = QVBoxLayout(self)
        self.form_layout = QFormLayout()
        self.form_layout.setHorizontalSpacing(25)
        self.fields = {}

        self.subject_combo = QComboBox()
        if self.is_edit_mode:
            self.subject_combo.addItem(self.subject.get('subject_id'))
            self.subject_combo.setEnabled(False)
        else:
            self.subject_combo.addItem("--- Select Subject ---")
            subject_ids = [s.get('subject_id', '') for s in self.parent_window.metadata_settings.get('Subjects', [])]
            self.subject_combo.addItems(subject_ids)
        self.form_layout.addRow("Subject:", self.subject_combo)

        self.intervention_type_combo = QComboBox()
        self.intervention_type_combo.addItems(['excitatory', 'inhibitory'])
        self.form_layout.addRow("Intervention Type:", self.intervention_type_combo)
        self.fields['name'] = self.intervention_type_combo

        line_edit_fields_part1 = [
            'virus_vendor', 'virus_name', 'virus_lot', 'virus_concentration',
            'virus_injection_date', 'target_area',
        ]
        for key in line_edit_fields_part1:
            widget = QLineEdit()
            widget.setPlaceholderText(placeholders.get(key, ""))
            self.form_layout.addRow(f"{key.replace('_', ' ').title()}:", widget)
            self.fields[key] = widget

        self.target_hemisphere_combo = QComboBox()
        self.target_hemisphere_combo.addItems(['left', 'right', 'bilateral'])
        self.form_layout.addRow("Target Hemisphere:", self.target_hemisphere_combo)
        self.fields['target_hemisphere'] = self.target_hemisphere_combo

        line_edit_fields_part2 = [
            'agonist_vendor', 'agonist_name', 'agonist_lot', 'agonist_concentration',
            'agonist_injection_time'
        ]
        for key in line_edit_fields_part2:
            widget = QLineEdit()
            widget.setPlaceholderText(placeholders.get(key, ""))
            self.form_layout.addRow(f"{key.replace('_', ' ').title()}:", widget)
            self.fields[key] = widget

        self.agonist_injection_type_combo = QComboBox()
        self.agonist_injection_type_combo.addItems(['subcutaneous', 'intraperitoneal'])
        self.form_layout.addRow("Agonist Injection Type:", self.agonist_injection_type_combo)
        self.fields['agonist_injection_type'] = self.agonist_injection_type_combo

        layout.addLayout(self.form_layout)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self.save_and_accept)
        button_box.rejected.connect(self.reject)

        cancel_button = button_box.button(QDialogButtonBox.StandardButton.Cancel)
        if cancel_button:
            cancel_button.setIcon(QIcon(cancel_icon))

        self.ok_button = button_box.button(QDialogButtonBox.StandardButton.Ok)
        if self.ok_button:
            self.ok_button.setIcon(QIcon(accept_icon))

        if self.is_edit_mode:
            delete_button = button_box.addButton("Delete", QDialogButtonBox.ButtonRole.DestructiveRole)
            delete_button.clicked.connect(self.delete_intervention)

        layout.addWidget(button_box)

        if self.is_edit_mode:
            self.ok_button.setEnabled(True)
        else:
            self.ok_button.setEnabled(False)
            self.subject_combo.currentIndexChanged.connect(lambda index: self.ok_button.setEnabled(index > 0))

        if self.is_edit_mode:
            self.populate_form()

    def populate_form(self):
        """
        Fills the form widgets with existing intervention data.

        Parameters
        ----------
        ----------

        Returns
        -------
        -------
        """

        interv_data = self.subject.get('interventions', {}).get(self.intervention_key, {})
        for key, widget in self.fields.items():
            value = interv_data.get(key, "")
            if isinstance(widget, QComboBox):
                widget.setCurrentText(str(value))
            else:
                widget.setText(str(value))

    def delete_intervention(self):
        """
        Deletes this intervention from the subject.

        Parameters
        ----------
        ----------

        Returns
        -------
        -------
        """

        reply = QMessageBox.question(self, 'Confirm Delete',
                                     f"Are you sure you want to delete the {self.intervention_key} intervention for subject {self.subject.get('subject_id')}?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)

        if reply == QMessageBox.StandardButton.Yes:
            if 'interventions' in self.subject and self.intervention_key in self.subject['interventions']:
                del self.subject['interventions'][self.intervention_key]
                self.parent_window._save_metadata_to_yaml()
                self.parent_window._update_subject_in_repository(self.subject)
            self.accept()

    def save_and_accept(self):
        """
        Gathers data from the mixed widgets, saves, and closes.

        Parameters
        ----------
        ----------

        Returns
        -------
        -------
        """

        selected_id = self.subject_combo.currentText()
        target_subject = next((s for s in self.parent_window.metadata_settings['Subjects'] if s.get('subject_id') == selected_id), None)

        if not target_subject:
            return

        intervention_data = {}
        for key, widget in self.fields.items():
            if isinstance(widget, QComboBox):
                intervention_data[key] = widget.currentText()
            else:  # It's a QLineEdit
                intervention_data[key] = widget.text()

        if 'interventions' not in target_subject or not isinstance(target_subject.get('interventions'), dict):
            target_subject['interventions'] = {}
        target_subject['interventions']['chemogenetics'] = intervention_data

        self.parent_window._save_metadata_to_yaml()
        self.parent_window._update_subject_in_repository(target_subject)
        self.accept()


class EphysDialog(QDialog):
    """A custom dialog for E-phys Interventions with edit/delete functionality."""
    def __init__(self, parent=None, subject: dict = None):
        super().__init__(parent)
        self.parent_window = parent
        self.subject = subject
        self.is_edit_mode = (subject is not None)
        self.intervention_key = 'electrophysiology'

        self.setWindowTitle("Edit E-phys Intervention" if self.is_edit_mode else "Add E-phys Intervention")
        self.setMinimumWidth(500)

        placeholders = {
            'surgery_date': '2025-09-29',
            'cable_sn': 'XSE0048-005413',
            'hs_sn': '23280190',
            'hs_sr': '30000.207531380755',
            'probe_model': 'NP2013',
            'probe_sn': '22420013594,22420015871',
            'target_area': 'lPAG',
            'software_version': 'v20240620-phase30'
        }

        layout = QVBoxLayout(self)
        self.form_layout = QFormLayout()
        self.form_layout.setHorizontalSpacing(25)
        self.fields = {}

        self.subject_combo = QComboBox()
        if self.is_edit_mode:
            self.subject_combo.addItem(self.subject.get('subject_id'))
            self.subject_combo.setEnabled(False)
        else:
            self.subject_combo.addItem("--- Select Subject ---")
            subject_ids = [s.get('subject_id', '') for s in self.parent_window.metadata_settings.get('Subjects', [])]
            self.subject_combo.addItems(subject_ids)
        self.form_layout.addRow("Subject:", self.subject_combo)

        ephys_setups = self.parent_window.equipment_settings_dict.get('ephys', {})
        self.setup_combo = QComboBox()
        self.setup_combo.addItems(ephys_setups.keys())
        self.form_layout.addRow("Setup:", self.setup_combo)
        self.fields['name'] = self.setup_combo

        line_edit_fields_part1 = {'surgery_date': 'Surgery Date:', 'cable_sn': 'Cable SN:', 'hs_sn': 'Headstage SN:',
                                  'hs_sr': 'Headstage SR:', 'probe_model': 'Probe Model:', 'probe_sn': 'Probe SN:'}
        for key, label_text in line_edit_fields_part1.items():
            widget = QLineEdit()
            widget.setPlaceholderText(placeholders.get(key, ""))
            self.form_layout.addRow(label_text, widget)
            self.fields[key] = widget

        self.probe_reused_combo = QComboBox()
        self.probe_reused_combo.addItems(['Yes', 'No'])
        self.form_layout.addRow("Probe Reused:", self.probe_reused_combo)
        self.fields['probe_reused'] = self.probe_reused_combo

        self.target_area_edit = QLineEdit()
        self.form_layout.addRow("Target Area:", self.target_area_edit)
        self.fields['target_area'] = self.target_area_edit

        self.hemisphere_combo = QComboBox()
        self.hemisphere_combo.addItems(['left', 'right', 'bilateral'])
        self.form_layout.addRow("Target Hemisphere:", self.hemisphere_combo)
        self.fields['target_hemisphere'] = self.hemisphere_combo

        self.software_version_edit = QLineEdit()
        self.form_layout.addRow("Software Version:", self.software_version_edit)
        self.fields['software_version'] = self.software_version_edit

        layout.addLayout(self.form_layout)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self.save_and_accept)
        button_box.rejected.connect(self.reject)

        cancel_button = button_box.button(QDialogButtonBox.StandardButton.Cancel)
        if cancel_button:
            cancel_button.setIcon(QIcon(cancel_icon))
        self.ok_button = button_box.button(QDialogButtonBox.StandardButton.Ok)
        if self.ok_button:
            self.ok_button.setIcon(QIcon(accept_icon))
        if self.is_edit_mode:
            delete_button = button_box.addButton("Delete", QDialogButtonBox.ButtonRole.DestructiveRole)
            delete_button.clicked.connect(self.delete_intervention)
        layout.addWidget(button_box)

        if self.is_edit_mode:
            self.ok_button.setEnabled(True)
        else:
            self.ok_button.setEnabled(False)
            self.subject_combo.currentIndexChanged.connect(lambda index: self.ok_button.setEnabled(index > 0))

        if self.is_edit_mode:
            self.populate_form()

    def populate_form(self):
        interv_data = self.subject.get('interventions', {}).get(self.intervention_key, {})
        for key, widget in self.fields.items():
            value = interv_data.get(key, "")
            if key == 'probe_reused':
                widget.setCurrentText('Yes' if value else 'No')
            elif key == 'probe_sn' and isinstance(value, list):
                widget.setText(', '.join(map(str, value)))
            elif isinstance(widget, QComboBox):
                widget.setCurrentText(str(value))
            else:
                widget.setText(str(value))

    def delete_intervention(self):
        reply = QMessageBox.question(self, 'Confirm Delete',
                                     f"Are you sure you want to delete the {self.intervention_key} intervention for subject {self.subject.get('subject_id')}?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            if 'interventions' in self.subject and self.intervention_key in self.subject['interventions']:
                del self.subject['interventions'][self.intervention_key]
                self.parent_window._save_metadata_to_yaml()
                self.parent_window._update_subject_in_repository(self.subject)

            self.accept()

    def save_and_accept(self):
        selected_id = self.subject_combo.currentText()
        target_subject = self.subject if self.is_edit_mode else next((s for s in self.parent_window.metadata_settings['Subjects'] if s.get('subject_id') == selected_id), None)
        if not target_subject: return

        intervention_data = {}
        for key, widget in self.fields.items():
            intervention_data[key] = widget.currentText() if isinstance(widget, QComboBox) else widget.text()

        intervention_data['probe_reused'] = (intervention_data['probe_reused'] == 'Yes')
        probe_sn_str = intervention_data.get('probe_sn', '')
        if probe_sn_str:
            final_sn_list = [item.strip() for item in probe_sn_str.split(',') if item.strip()]
            intervention_data['probe_sn'] = [int(sn) if sn.isdigit() else sn for sn in final_sn_list]
        else:
            intervention_data['probe_sn'] = []

        if 'interventions' not in target_subject: target_subject['interventions'] = {}
        target_subject['interventions'][self.intervention_key] = intervention_data

        self.parent_window._save_metadata_to_yaml()
        self.parent_window._update_subject_in_repository(target_subject)
        self.accept()


class LesionDialog(QDialog):
    """A custom dialog for Lesion Interventions with edit/delete functionality."""
    def __init__(self, parent=None, subject: dict = None):
        super().__init__(parent)
        self.parent_window = parent
        self.subject = subject
        self.is_edit_mode = (subject is not None)
        self.intervention_key = 'lesion'

        self.setWindowTitle("Edit Lesion Intervention" if self.is_edit_mode else "Add Lesion Intervention")
        self.setMinimumWidth(500)

        placeholders = {
            'name': 'caspase',
            'virus_vendor': 'addgene',
            'virus_name': 'pAAV5-flex-taCasp3-TEVp',
            'virus_injection_date': '2025-08-15',
            'target_area': 'lPAG'
        }

        layout = QVBoxLayout(self)
        self.form_layout = QFormLayout()
        self.form_layout.setHorizontalSpacing(25)
        self.fields = {}

        self.subject_combo = QComboBox()
        if self.is_edit_mode:
            self.subject_combo.addItem(self.subject.get('subject_id'))
            self.subject_combo.setEnabled(False)
        else:
            self.subject_combo.addItem("--- Select Subject ---")
            subject_ids = [s.get('subject_id', '') for s in self.parent_window.metadata_settings.get('Subjects', [])]
            self.subject_combo.addItems(subject_ids)
        self.form_layout.addRow("Subject:", self.subject_combo)

        line_edit_fields = ['name', 'virus_vendor', 'virus_name', 'virus_lot', 'virus_concentration', 'virus_injection_date', 'target_area']
        for key in line_edit_fields:
            widget = QLineEdit()
            widget.setPlaceholderText(placeholders.get(key, ""))
            self.form_layout.addRow(f"{key.replace('_', ' ').title()}:", widget)
            self.fields[key] = widget

        self.target_hemisphere_combo = QComboBox()
        self.target_hemisphere_combo.addItems(['left', 'right', 'bilateral'])
        self.form_layout.addRow("Target Hemisphere:", self.target_hemisphere_combo)
        self.fields['target_hemisphere'] = self.target_hemisphere_combo

        layout.addLayout(self.form_layout)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self.save_and_accept)
        button_box.rejected.connect(self.reject)

        cancel_button = button_box.button(QDialogButtonBox.StandardButton.Cancel)
        if cancel_button:
            cancel_button.setIcon(QIcon(cancel_icon))
        self.ok_button = button_box.button(QDialogButtonBox.StandardButton.Ok)
        if self.ok_button:
            self.ok_button.setIcon(QIcon(accept_icon))
        if self.is_edit_mode:
            delete_button = button_box.addButton("Delete", QDialogButtonBox.ButtonRole.DestructiveRole)
            delete_button.clicked.connect(self.delete_intervention)
        layout.addWidget(button_box)

        if self.is_edit_mode:
            self.ok_button.setEnabled(True)
        else:
            self.ok_button.setEnabled(False)
            self.subject_combo.currentIndexChanged.connect(lambda index: self.ok_button.setEnabled(index > 0))

        if self.is_edit_mode:
            self.populate_form()

    def populate_form(self):
        interv_data = self.subject.get('interventions', {}).get(self.intervention_key, {})
        for key, widget in self.fields.items():
            value = interv_data.get(key, "")
            if isinstance(widget, QComboBox):
                widget.setCurrentText(str(value))
            else:
                widget.setText(str(value))

    def delete_intervention(self):
        reply = QMessageBox.question(self, 'Confirm Delete',
                                     f"Are you sure you want to delete the {self.intervention_key} intervention for subject {self.subject.get('subject_id')}?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            if 'interventions' in self.subject and self.intervention_key in self.subject['interventions']:
                del self.subject['interventions'][self.intervention_key]
                self.parent_window._update_subject_in_repository(self.subject)
                self.parent_window._save_metadata_to_yaml()
            self.accept()

    def save_and_accept(self):
        selected_id = self.subject_combo.currentText()
        target_subject = self.subject if self.is_edit_mode else next((s for s in self.parent_window.metadata_settings['Subjects'] if s.get('subject_id') == selected_id), None)
        if not target_subject: return

        intervention_data = {}
        for key, widget in self.fields.items():
            intervention_data[key] = widget.currentText() if isinstance(widget, QComboBox) else widget.text()

        if 'interventions' not in target_subject: target_subject['interventions'] = {}
        target_subject['interventions'][self.intervention_key] = intervention_data

        self.parent_window._save_metadata_to_yaml()
        self.parent_window._update_subject_in_repository(target_subject)
        self.accept()


class OptoDialog(QDialog):
    """A custom dialog for Optogenetic Interventions with edit/delete functionality."""

    def __init__(self, parent=None, subject: dict = None):
        super().__init__(parent)
        self.parent_window = parent
        self.subject = subject
        self.is_edit_mode = (subject is not None)
        self.intervention_key = 'optogenetics'

        self.setWindowTitle("Edit Optogenetic Intervention" if self.is_edit_mode else "Add Optogenetic Intervention")
        self.setMinimumWidth(500)

        placeholders = {
            'virus_vendor': 'addgene',
            'virus_name': 'AAV-CAG-ChR2-GFP',
            'virus_injection_date': '2025-09-29',
            'virus_target_area': 'lPAG',
            'fiber_implantation_date': '2025-09-29',
            'fiber_target_area': 'lPAG',
            'stimulation_power': '2 mW',
            'stimulation_frequency': '10 Hz',
            'stimulation_duty_cycle': '0.5'
        }

        layout = QVBoxLayout(self)
        self.form_layout = QFormLayout()
        self.form_layout.setHorizontalSpacing(25)
        self.fields = {}

        self.subject_combo = QComboBox()
        if self.is_edit_mode:
            self.subject_combo.addItem(self.subject.get('subject_id'))
            self.subject_combo.setEnabled(False)
        else:
            self.subject_combo.addItem("--- Select Subject ---")
            subject_ids = [s.get('subject_id', '') for s in self.parent_window.metadata_settings.get('Subjects', [])]
            self.subject_combo.addItems(subject_ids)
        self.form_layout.addRow("Subject:", self.subject_combo)

        opto_setups = self.parent_window.equipment_settings_dict.get('opto', {})
        self.setup_combo = QComboBox()
        self.setup_combo.addItems(opto_setups.keys())
        self.form_layout.addRow("Setup:", self.setup_combo)
        self.fields['name'] = self.setup_combo

        self.intervention_type_combo = QComboBox()
        self.intervention_type_combo.addItems(['excitatory', 'inhibitory'])
        self.form_layout.addRow("Intervention Type:", self.intervention_type_combo)
        self.fields['intervention_type'] = self.intervention_type_combo

        for key in ['virus_vendor', 'virus_name', 'virus_lot', 'virus_concentration', 'virus_injection_date', 'virus_target_area']:
            widget = QLineEdit()
            widget.setPlaceholderText(placeholders.get(key, ""))
            self.form_layout.addRow(f"{key.replace('_', ' ').title()}:", widget)
            self.fields[key] = widget

        self.virus_hemisphere_combo = QComboBox()
        self.virus_hemisphere_combo.addItems(['left', 'right', 'bilateral'])
        self.form_layout.addRow("Virus Target Hemisphere:", self.virus_hemisphere_combo)
        self.fields['virus_target_hemisphere'] = self.virus_hemisphere_combo

        for key in ['fiber_implantation_date', 'fiber_target_area']:
            widget = QLineEdit()
            widget.setPlaceholderText(placeholders.get(key, ""))
            self.form_layout.addRow(f"{key.replace('_', ' ').title()}:", widget)
            self.fields[key] = widget

        self.fiber_hemisphere_combo = QComboBox()
        self.fiber_hemisphere_combo.addItems(['left', 'right', 'bilateral'])
        self.form_layout.addRow("Fiber Target Hemisphere:", self.fiber_hemisphere_combo)
        self.fields['fiber_target_hemisphere'] = self.fiber_hemisphere_combo

        for key in ['stimulation_power', 'stimulation_frequency', 'stimulation_duty_cycle']:
            widget = QLineEdit()
            widget.setPlaceholderText(placeholders.get(key, ""))
            self.form_layout.addRow(f"{key.replace('_', ' ').title()}:", widget)
            self.fields[key] = widget

        layout.addLayout(self.form_layout)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self.save_and_accept)
        button_box.rejected.connect(self.reject)

        cancel_button = button_box.button(QDialogButtonBox.StandardButton.Cancel)
        if cancel_button:
            cancel_button.setIcon(QIcon(cancel_icon))
        self.ok_button = button_box.button(QDialogButtonBox.StandardButton.Ok)
        if self.ok_button:
            self.ok_button.setIcon(QIcon(accept_icon))
        if self.is_edit_mode:
            delete_button = button_box.addButton("Delete", QDialogButtonBox.ButtonRole.DestructiveRole)
            delete_button.clicked.connect(self.delete_intervention)
        layout.addWidget(button_box)

        if self.is_edit_mode:
            self.ok_button.setEnabled(True)
        else:
            self.ok_button.setEnabled(False)
            self.subject_combo.currentIndexChanged.connect(lambda index: self.ok_button.setEnabled(index > 0))

        if self.is_edit_mode:
            self.populate_form()

    def populate_form(self):
        interv_data = self.subject.get('interventions', {}).get(self.intervention_key, {})
        for key, widget in self.fields.items():
            value = interv_data.get(key, "")
            if isinstance(widget, QComboBox):
                widget.setCurrentText(str(value))
            else:
                widget.setText(str(value))

    def delete_intervention(self):
        reply = QMessageBox.question(self, 'Confirm Delete',
                                     f"Are you sure you want to delete the {self.intervention_key} intervention for subject {self.subject.get('subject_id')}?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            if 'interventions' in self.subject and self.intervention_key in self.subject['interventions']:
                del self.subject['interventions'][self.intervention_key]
                self.parent_window._save_metadata_to_yaml()
                self.parent_window._update_subject_in_repository(self.subject)
            self.accept()

    def save_and_accept(self):
        selected_id = self.subject_combo.currentText()
        target_subject = self.subject if self.is_edit_mode else next((s for s in self.parent_window.metadata_settings['Subjects'] if s.get('subject_id') == selected_id), None)
        if not target_subject: return

        intervention_data = {}
        for key, widget in self.fields.items():
            intervention_data[key] = widget.currentText() if isinstance(widget, QComboBox) else widget.text()

        if 'interventions' not in target_subject: target_subject['interventions'] = {}
        target_subject['interventions'][self.intervention_key] = intervention_data

        self.parent_window._save_metadata_to_yaml()
        self.parent_window._update_subject_in_repository(target_subject)
        self.accept()

def replace_name_in_path(experimenter_list: list = None,
                         recording_files_destinations: list = None,
                         exp_id : str = None) -> str:
    """
    Replace the name in the path with the new experimenter name.

    Parameters
    ----------
    experimenter_list (list)
        Path to be modified.
    recording_files_destinations (list)
        New name to be used.
    exp_id (str)
        Experiment ID to be inserted.

    Returns
    -------
    global_destination (str)
        Path with correct exp_id name inserted.
    """

    if any([name in loc for name in experimenter_list for loc in recording_files_destinations]):
        revised_recording_files_destination_win = []
        for path, name_in_path in zip(recording_files_destinations,
                                      [name for name in experimenter_list for loc in recording_files_destinations if name in loc]):
            old_name_path = path
            name_span = re.search(pattern=name_in_path, string=old_name_path).span()
            revised_recording_files_destination_win.append(old_name_path[:name_span[0]] + exp_id + old_name_path[name_span[1]:])
        global_destination = ','.join(revised_recording_files_destination_win)
    else:
        global_destination = ','.join(recording_files_destinations)

    return global_destination


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

class Credentials(QWidget):
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
        super(Credentials, self).__init__(parent)

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

        self.remove_subject_buttons = []
        self.edit_intervention_widgets = []
        self.subject_form_widgets = {}
        self.active_subject_id = None
        self._is_clearing_form = False

        self.subject_repository = []

        config_dir = Path(platformdirs.user_config_dir(appname='usv_playpen', appauthor='lab'))
        self.subject_repo_path = config_dir / 'subject_presets.json'

        self.main_window()

    def _open_chemo_dialog(self):
        ChemoDialog(self).exec()
        self._update_subject_ui()

    def _open_ephys_dialog(self):
        EphysDialog(self).exec()
        self._update_subject_ui()

    def _open_lesion_dialog(self):
        LesionDialog(self).exec()
        self._update_subject_ui()

    def _open_opto_dialog(self):
        OptoDialog(self).exec()
        self._update_subject_ui()

    def _load_subject_repository(self):
        """
        Loads the master list of subjects from the JSON repository file.
        If the file doesn't exist or is corrupted, it starts with an empty list.

        Parameters
        ----------
        ----------

        Returns
        -------
        -------
        """

        try:
            if self.subject_repo_path.exists():
                with open(self.subject_repo_path, 'r') as f:
                    self.subject_repository = json.load(f)
                    if not isinstance(self.subject_repository, list):
                        self.subject_repository = []
            else:
                self.subject_repository = []
        except (json.JSONDecodeError, FileNotFoundError):
            self.subject_repository = []

    def _save_subject_to_repository(self):
        """
        Saves the current state of self.subject_repository to the JSON file.

        Parameters
        ----------
        ----------

        Returns
        -------
        -------
        """

        try:
            self.subject_repo_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.subject_repo_path, 'w') as f:
                json.dump(self.subject_repository, f, indent=4)
        except Exception as e:
            print(f"Error saving subject repository: {e}")

    def _on_subject_form_changed(self):
        """
        Live-updates the active subject's data whenever a form field is changed.

        Parameters
        ----------
        ----------

        Returns
        -------
        -------
        """

        if self._is_clearing_form:
            return

        if not self.active_subject_id:
            return

        target_subject = next((s for s in self.metadata_settings.get('Subjects', []) if s.get('subject_id') == self.active_subject_id), None)
        if not target_subject:
            return

        current_form_data = {}
        for key, widget in self.subject_form_widgets.items():
            if isinstance(widget, QComboBox):
                current_form_data[key] = widget.currentText()
            else:
                current_form_data[key] = widget.text()

        try:
            current_form_data['weight'] = ast.literal_eval(current_form_data['weight'])
        except (ValueError, SyntaxError):
            pass

        target_subject.update(current_form_data)

        self._update_subject_in_repository(target_subject)
        self._save_metadata_to_yaml()

    def _on_subject_selected_from_completer(self, subject_id: str):
        """
        Triggered when a subject is selected from the autocomplete dropdown.
        Finds the full subject data, autofills the form, adds the subject to the
        current session, and refreshes the UI.

        Parameters
        ----------
        ----------

        Returns
        -------
        -------
        """

        subject_data = None
        for subject in self.subject_repository:
            if subject.get('subject_id') == subject_id:
                subject_data = subject
                break

        if not subject_data:
            return

        for key, value in subject_data.items():
            if key in self.subject_form_widgets:
                widget = self.subject_form_widgets[key]
                if isinstance(widget, QComboBox):
                    widget.setCurrentText(str(value))
                else:
                    widget.setText(str(value))

        self.active_subject_id = subject_id

        current_session_ids = {s.get('subject_id') for s in self.metadata_settings.get('Subjects', []) if s}
        if subject_id in current_session_ids:
            QMessageBox.information(
                self,
                "Subject Already in Session",
                f"The subject '{subject_id}' is already in this session. The form has been filled for live editing."
            )

            self.add_subject_btn.setEnabled(False)
            return

        subject_to_add = copy.deepcopy(subject_data)

        if 'Subjects' not in self.metadata_settings or not isinstance(self.metadata_settings['Subjects'], list):
            self.metadata_settings['Subjects'] = []
        self.metadata_settings['Subjects'].append(subject_to_add)

        self._save_metadata_to_yaml()
        self._update_subject_ui()

        self.add_subject_btn.setEnabled(False)

    def _update_subject_in_repository(self, subject_data_to_save: dict):
        """
        Finds a subject in the master repository by its ID and updates it,
        or adds it if it's a new subject. Then, saves the repository to disk.

        Parameters
        ----------
        ----------

        Returns
        -------
        -------
        """

        subject_id = subject_data_to_save.get('subject_id')
        if not subject_id:
            return

        existing_subject_index = -1
        for i, subject in enumerate(self.subject_repository):
            if subject.get('subject_id') == subject_id:
                existing_subject_index = i
                break

        if existing_subject_index != -1:
            self.subject_repository[existing_subject_index] = subject_data_to_save
        else:
            self.subject_repository.append(subject_data_to_save)

        self._save_subject_to_repository()

    def _clear_subject_form(self) -> None:
        """
        Clears all QLineEdit fields in the 'Add Subject' form.

        Parameters
        ----------
        ----------

        Returns
        -------
        -------
        """

        self._is_clearing_form = True

        for widget in self.subject_form_widgets.values():
            if isinstance(widget, QLineEdit):
                widget.clear()
            elif isinstance(widget, QComboBox):
                widget.setCurrentIndex(0)

        self._is_clearing_form = False

        self.active_subject_id = None

        self._validate_subject_form()


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

        self._load_subject_repository()

        self.exp_settings_dict = toml.load(Path(__file__).parent / '_config/behavioral_experiments_settings.toml')

        self.equipment_settings_dict = toml.load(Path(__file__).parent / '_config/equipment.toml')

        with open(Path(__file__).parent / '_config/_metadata.yaml', 'r') as metadata_file:
            self.metadata_settings = yaml.safe_load(metadata_file)

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

    def credentials_window(self) -> None:
        """
        Initializes the usv-playpen Main window.

        Parameters
        ----------
        ----------

        Returns
        -------
        -------
        """

        self.Credentials = Credentials(self)
        self.setWindowTitle(f'{app_name} (Set credentials)')
        self.setCentralWidget(self.Credentials)
        self.setFixedSize(420, 500)

        credentials_label = QLabel('Please insert information to be saved to the credential files', self.Credentials)
        credentials_label.setFont(QFont(self.font_id, 10 + self.font_size_increase))
        credentials_label.setStyleSheet('QLabel { font-weight: bold;}')
        credentials_label.move(5, 10)

        credentials_save_dir_label = QLabel('Save directory:', self.Credentials)
        credentials_save_dir_label.setFont(QFont(self.font_id, 12 + self.font_size_increase))
        credentials_save_dir_label.move(5, 40)
        self.credentials_save_dir_edit = QLineEdit(f"{platformdirs.user_config_dir(appname='usv_playpen', appauthor='lab')}", self.Credentials)
        self.credentials_save_dir_edit.setPlaceholderText('Save credentials directory')
        self.credentials_save_dir_edit.setFont(QFont(self.font_id, 10 + self.font_size_increase))
        self.credentials_save_dir_edit.setStyleSheet('QLineEdit { width: 235px; }')
        self.credentials_save_dir_edit.move(115, 40)
        credentials_save_dir_btn = QPushButton('Browse', self.Credentials)
        credentials_save_dir_btn.setFont(QFont(self.font_id, 8 + self.font_size_increase))
        credentials_save_dir_btn.move(355, 40)
        credentials_save_dir_btn.setStyleSheet('QPushButton { min-width: 36px; min-height: 12px; max-width: 36px; max-height: 13px; }')
        credentials_save_dir_dialog = partial(self._open_directory_dialog, self.credentials_save_dir_edit, 'Select credentials directory')
        credentials_save_dir_btn.clicked.connect(credentials_save_dir_dialog)

        credentials_label = QLabel('E-MAIL credentials', self.Credentials)
        credentials_label.setFont(QFont(self.font_id, 10 + self.font_size_increase))
        credentials_label.setStyleSheet('QLabel { color: #F58025; font-weight: bold;}')
        credentials_label.move(5, 70)

        email_host_label = QLabel('[provider]:', self.Credentials)
        email_host_label.setFont(QFont(self.font_id, 10 + self.font_size_increase))
        email_host_label.setStyleSheet('QLabel { font-weight: bold;}')
        email_host_label.move(5, 95)
        self.email_host = QLineEdit("smtp.gmail.com", self.Credentials)
        self.email_host.setFont(QFont(self.font_id, 10 + self.font_size_increase))
        self.email_host.setStyleSheet('QLineEdit { height: 15px; width: 335px; }')
        self.email_host.move(80, 95)

        email_port_label = QLabel('[port num]:', self.Credentials)
        email_port_label.setFont(QFont(self.font_id, 10 + self.font_size_increase))
        email_port_label.setStyleSheet('QLabel { font-weight: bold;}')
        email_port_label.move(5, 120)
        self.email_port = QLineEdit("465", self.Credentials)
        self.email_port.setFont(QFont(self.font_id, 10 + self.font_size_increase))
        self.email_port.setStyleSheet('QLineEdit { height: 15px; width: 335px; }')
        self.email_port.move(80, 120)

        email_address_label = QLabel('[address]:', self.Credentials)
        email_address_label.setFont(QFont(self.font_id, 10 + self.font_size_increase))
        email_address_label.setStyleSheet('QLabel { font-weight: bold;}')
        email_address_label.move(5, 145)
        self.email_address = QLineEdit("165b.pni@gmail.com", self.Credentials)
        self.email_address.setFont(QFont(self.font_id, 10 + self.font_size_increase))
        self.email_address.setStyleSheet('QLineEdit { height: 15px; width: 335px; }')
        self.email_address.move(80, 145)

        email_password_label = QLabel('[password]:', self.Credentials)
        email_password_label.setFont(QFont(self.font_id, 10 + self.font_size_increase))
        email_password_label.setStyleSheet('QLabel { font-weight: bold;}')
        email_password_label.move(5, 170)
        self.email_password = QLineEdit("XXX", self.Credentials)
        self.email_password.setFont(QFont(self.font_id, 10 + self.font_size_increase))
        self.email_password.setStyleSheet('QLineEdit { height: 15px; width: 335px; }')
        self.email_password.move(80, 170)

        credentials_label = QLabel('UNIVERSITY credentials', self.Credentials)
        credentials_label.setFont(QFont(self.font_id, 10 + self.font_size_increase))
        credentials_label.setStyleSheet('QLabel { color: #F58025; font-weight: bold;}')
        credentials_label.move(5, 200)

        university_username_label = QLabel('[username]:', self.Credentials)
        university_username_label.setFont(QFont(self.font_id, 10 + self.font_size_increase))
        university_username_label.setStyleSheet('QLabel { font-weight: bold;}')
        university_username_label.move(5, 225)
        self.university_username = QLineEdit("nsurname", self.Credentials)
        self.university_username.setFont(QFont(self.font_id, 10 + self.font_size_increase))
        self.university_username.setStyleSheet('QLineEdit { height: 15px; width: 335px; }')
        self.university_username.move(80, 225)

        university_password_label = QLabel('[password]:', self.Credentials)
        university_password_label.setFont(QFont(self.font_id, 10 + self.font_size_increase))
        university_password_label.setStyleSheet('QLabel { font-weight: bold;}')
        university_password_label.move(5, 250)
        self.university_password = QLineEdit("XXX", self.Credentials)
        self.university_password.setFont(QFont(self.font_id, 10 + self.font_size_increase))
        self.university_password.setStyleSheet('QLineEdit { height: 15px; width: 335px; }')
        self.university_password.move(80, 250)

        credentials_label = QLabel('MOTIF credentials', self.Credentials)
        credentials_label.setFont(QFont(self.font_id, 10 + self.font_size_increase))
        credentials_label.setStyleSheet('QLabel { color: #F58025; font-weight: bold;}')
        credentials_label.move(5, 280)

        motif_master_ip_label = QLabel('[primary ip]:', self.Credentials)
        motif_master_ip_label.setFont(QFont(self.font_id, 10 + self.font_size_increase))
        motif_master_ip_label.setStyleSheet('QLabel { font-weight: bold;}')
        motif_master_ip_label.move(5, 305)
        self.motif_master_ip = QLineEdit("10.241.1.205", self.Credentials)
        self.motif_master_ip.setFont(QFont(self.font_id, 10 + self.font_size_increase))
        self.motif_master_ip.setStyleSheet('QLineEdit { height: 15px; width: 335px; }')
        self.motif_master_ip.move(80, 305)

        motif_second_ip_label = QLabel('[second ip]:', self.Credentials)
        motif_second_ip_label.setFont(QFont(self.font_id, 10 + self.font_size_increase))
        motif_second_ip_label.setStyleSheet('QLabel { font-weight: bold;}')
        motif_second_ip_label.move(5, 330)
        self.motif_second_ip = QLineEdit("10.241.1.183", self.Credentials)
        self.motif_second_ip.setFont(QFont(self.font_id, 10 + self.font_size_increase))
        self.motif_second_ip.setStyleSheet('QLineEdit { height: 15px; width: 335px; }')
        self.motif_second_ip.move(80, 330)

        motif_ssh_port_label = QLabel('[ssh port]:', self.Credentials)
        motif_ssh_port_label.setFont(QFont(self.font_id, 10 + self.font_size_increase))
        motif_ssh_port_label.setStyleSheet('QLabel { font-weight: bold;}')
        motif_ssh_port_label.move(5, 355)
        self.motif_ssh_port = QLineEdit("22", self.Credentials)
        self.motif_ssh_port.setFont(QFont(self.font_id, 10 + self.font_size_increase))
        self.motif_ssh_port.setStyleSheet('QLineEdit { height: 15px; width: 335px; }')
        self.motif_ssh_port.move(80, 355)

        motif_username_label = QLabel('[username]:', self.Credentials)
        motif_username_label.setFont(QFont(self.font_id, 10 + self.font_size_increase))
        motif_username_label.setStyleSheet('QLabel { font-weight: bold;}')
        motif_username_label.move(5, 380)
        self.motif_username = QLineEdit("labadmin", self.Credentials)
        self.motif_username.setFont(QFont(self.font_id, 10 + self.font_size_increase))
        self.motif_username.setStyleSheet('QLineEdit { height: 15px; width: 335px; }')
        self.motif_username.move(80, 380)

        motif_password_label = QLabel('[password]:', self.Credentials)
        motif_password_label.setFont(QFont(self.font_id, 10 + self.font_size_increase))
        motif_password_label.setStyleSheet('QLabel { font-weight: bold;}')
        motif_password_label.move(5, 405)
        self.motif_password = QLineEdit("XXX", self.Credentials)
        self.motif_password.setFont(QFont(self.font_id, 10 + self.font_size_increase))
        self.motif_password.setStyleSheet('QLineEdit { height: 15px; width: 335px; }')
        self.motif_password.move(80, 405)

        motif_api_label = QLabel('[api key]:', self.Credentials)
        motif_api_label.setFont(QFont(self.font_id, 10 + self.font_size_increase))
        motif_api_label.setStyleSheet('QLabel { font-weight: bold;}')
        motif_api_label.move(5, 430)
        self.motif_api = QLineEdit("XXX", self.Credentials)
        self.motif_api.setFont(QFont(self.font_id, 10 + self.font_size_increase))
        self.motif_api.setStyleSheet('QLineEdit { height: 15px; width: 335px; }')
        self.motif_api.move(80, 430)

        self._create_buttons_credentials(class_option=self.Credentials,
                                         button_pos_y=465,
                                         next_button_x_pos=320)

    def _update_subject_ui(self):
        """
        Redraws remove buttons and enables/disables intervention buttons, using separate
        lists for each UI section to prevent redraw conflicts.

        Parameters
        ----------
        ----------

        Returns
        -------
        -------
        """

        for button in self.remove_subject_buttons:
            button.deleteLater()
        self.remove_subject_buttons.clear()

        for widget in self.edit_intervention_widgets:
            widget.deleteLater()
        self.edit_intervention_widgets.clear()

        subjects = self.metadata_settings.get('Subjects', [])
        if subjects is None:
            subjects = []

        start_y_remove = 993
        start_x_remove = 448
        items_per_row_remove = 4
        horizontal_spacing_remove = 90
        vertical_spacing_remove = 30

        for i, subject in enumerate(subjects):
            subject_id = subject.get('subject_id', f'Index {i}')
            row, col = divmod(i, items_per_row_remove)
            button_x = start_x_remove + (col * horizontal_spacing_remove)
            button_y = start_y_remove + (row * vertical_spacing_remove)

            button = QPushButton(QIcon(remove_icon), f"{subject_id}", self.VideoSettings)
            button.setStyleSheet('QPushButton { min-width: 65px; min-height: 12px; max-width: 65px; max-height: 13px; }')
            button.move(button_x, button_y)
            button.clicked.connect(partial(self._remove_subject, index_to_remove=i))
            button.show()
            self.remove_subject_buttons.append(button)

        y_pos_interv_buttons = 920
        x_pos_interv_buttons = 10
        abbreviation_map = {
            'chemogenetics': 'chemo', 'electrophysiology': 'ephys',
            'lesion': 'lesion', 'optogenetics': 'opto'
        }

        button_index = 0
        if subjects:
            for subject in subjects:
                if 'interventions' in subject and isinstance(subject['interventions'], dict):
                    for interv_key in subject['interventions']:
                        subject_id = subject.get('subject_id', 'Unknown')
                        row, col = divmod(button_index, 3)
                        button_x = x_pos_interv_buttons + (col * 130)
                        button_y = y_pos_interv_buttons + (row * 30)

                        short_name = abbreviation_map.get(interv_key, interv_key)
                        btn_text = f"{short_name} ({subject_id})"

                        button = QPushButton(btn_text, self.VideoSettings)
                        button.setStyleSheet('QPushButton { min-width: 105px; min-height: 12px; max-width: 105px; max-height: 13px; }')
                        button.move(button_x, button_y)
                        button.clicked.connect(partial(self._open_edit_intervention_dialog, subject_id=subject_id, intervention_key=interv_key))
                        button.show()
                        self.edit_intervention_widgets.append(button)
                        button_index += 1

        has_subjects = bool(subjects)
        self.add_chemo_btn.setEnabled(has_subjects)
        self.add_ephys_btn.setEnabled(has_subjects)
        self.add_lesion_btn.setEnabled(has_subjects)
        self.add_opto_btn.setEnabled(has_subjects)

    def _open_edit_intervention_dialog(self, subject_id: str, intervention_key: str):
        """
        Opens the correct dialog in 'edit mode' for the selected intervention.

        Parameters
        ----------
        ----------

        Returns
        -------
        -------
        """

        target_subject = next((s for s in self.metadata_settings['Subjects'] if s.get('subject_id') == subject_id), None)
        if not target_subject:
            return

        if intervention_key == 'chemogenetics':
            dialog = ChemoDialog(self, subject=target_subject)
            dialog.exec()
        elif intervention_key == 'electrophysiology':
            dialog = EphysDialog(self, subject=target_subject)
            dialog.exec()
        elif intervention_key == 'lesion':
            dialog = LesionDialog(self, subject=target_subject)
            dialog.exec()
        elif intervention_key == 'optogenetics':
            dialog = OptoDialog(self, subject=target_subject)
            dialog.exec()

        self._update_subject_ui()

    def _add_subject(self) -> None:
        """
        Gathers data from the subject QLineEdits, adds it to the metadata,
        saves the file, and clears the fields.

        Parameters
        ----------
        ----------

        Returns
        -------
        -------
        """

        new_subject_id = self.subject_form_widgets['subject_id'].text().strip()
        if not new_subject_id:
            return

        subject_data = {}
        for key, widget in self.subject_form_widgets.items():
            if isinstance(widget, QComboBox):
                subject_data[key] = widget.currentText()
            else:
                subject_data[key] = widget.text()

        try:
            subject_data['weight'] = ast.literal_eval(subject_data['weight'])
        except (ValueError, SyntaxError):
            pass

        target_subject_in_session = next((s for s in self.metadata_settings.get('Subjects', []) if s.get('subject_id') == new_subject_id), None)
        if target_subject_in_session:
            target_subject_in_session.update(subject_data)
        else:
            if 'Subjects' not in self.metadata_settings: self.metadata_settings['Subjects'] = []
            self.metadata_settings['Subjects'].append(subject_data)

        self._update_subject_in_repository(subject_data)
        self._save_metadata_to_yaml()
        self._update_subject_ui()

        self.active_subject_id = new_subject_id

    def _remove_subject(self, index_to_remove: int) -> None:
        """
        Removes a subject from the metadata list by its index.

        Parameters
        ----------
        ----------

        Returns
        -------
        -------
        """

        subjects = self.metadata_settings.get('Subjects', [])
        if subjects and 0 <= index_to_remove < len(subjects):
            subjects.pop(index_to_remove)
            self._save_metadata_to_yaml()
            self._update_subject_ui()

    def _redraw_remove_subject_buttons(self) -> None:
        """
        Clears and redraws the list of 'Remove Subject' buttons based on the
        current subjects in the metadata.

        Parameters
        ----------
        ----------

        Returns
        -------
        -------
        """

        for button in self.remove_subject_buttons:
            button.deleteLater()
        self.remove_subject_buttons.clear()

        start_x = 10
        start_y = 900
        horizontal_spacing = 145
        vertical_spacing = 20
        buttons_per_row = 3

        subjects = self.metadata_settings.get('Subjects', [])
        if subjects is None:
            return

        for i, subject in enumerate(subjects):
            subject_id = subject.get('subject_id', f'Index {i}')
            row, col = divmod(i, buttons_per_row)
            button_x = start_x + (col * horizontal_spacing)
            button_y = start_y + (row * vertical_spacing)

            button = QPushButton(f"Remove '{subject_id}'", self.VideoSettings)
            button.setStyleSheet('QPushButton { min-width: 120px; min-height: 12px; max-width: 120px; max-height: 13px; background-color: #552222;}')
            button.move(button_x, button_y)
            button.clicked.connect(partial(self._remove_subject, index_to_remove=i))
            button.show()
            self.remove_subject_buttons.append(button)

    def _on_equipment_checkbox_toggled(self, state: int, equipment_key: str) -> None:
        """
        Handles the toggling of an equipment checkbox.

        Adds or removes the corresponding equipment data from the main metadata
        dictionary based on the checkbox state.

        Parameters
        ----------
        state (int)
            The new state of the checkbox (provided by the stateChanged signal).
        equipment_key (str)
            The key of the equipment from the .toml file (e.g., 'video.Loopbio').
        ----------

        Returns
        -------
        -------
        """

        yaml_key = equipment_key.replace('.', '_')
        is_checked = (state == Qt.CheckState.Checked.value)

        if 'Equipment' not in self.metadata_settings or self.metadata_settings['Equipment'] is None:
            self.metadata_settings['Equipment'] = {}

        if is_checked:
            category, device_name = equipment_key.split('.')
            self.metadata_settings['Equipment'][yaml_key] = self.equipment_settings_dict[category][device_name]
        else:
            if yaml_key in self.metadata_settings['Equipment']:
                del self.metadata_settings['Equipment'][yaml_key]

        self._save_metadata_to_yaml()

    def _validate_subject_form(self) -> None:
        """
        Checks if required subject fields are filled and toggles the 'Add Subject' button.

        Parameters
        ----------
        ----------

        Returns
        -------
        -------
        """

        required_fields = ['subject_id', 'species', 'genotype_strain']
        if not all(hasattr(self, f"{key}_edit") for key in required_fields):
            return

        is_valid = all([
            self.subject_id_edit.text().strip(),
            self.species_edit.text().strip(),
            self.genotype_strain_edit.text().strip()
        ])

        self.add_subject_btn.setEnabled(is_valid)

    def _update_metadata_preview(self) -> None:
        """
        Updates the metadata preview pane with the current state of self.metadata_settings.

        Parameters
        ----------
        ----------

        Returns
        -------
        -------
        """

        if hasattr(self, 'metadata_preview_edit'):
            preview_text = yaml.dump(
                self.metadata_settings,
                Dumper=SmartDumper,
                default_flow_style=False,
                sort_keys=False,
                indent=2
            )
            self.metadata_preview_edit.setPlainText(preview_text)

    def _save_metadata_to_yaml(self) -> None:
        """
        Saves the current state of the metadata UI to _metadata.yaml.

        Parameters
        ----------
        ----------

        Returns
        -------
        -------
        """

        if not hasattr(self, 'institution_edit'):
            return

        session = self.metadata_settings['Session']
        session['institution'] = self.institution_edit.text()
        session['lab'] = self.lab_edit.text()
        session['experimenter'] = self.experimenter_edit.text()

        session['ambient_light'] = getattr(self, 'ambient_light_bool', False)

        session['session_experiment_code'] = self.session_experiment_code_edit.text()
        session['calibration_session'] = self.calibration_session_edit.text()
        session['session_usv_playback_file'] = self.session_usv_playback_file_edit.text()
        session['session_description'] = self.session_description_edit.text()

        keywords_str = self.keywords_edit.text()
        session['keywords'] = [k.strip() for k in keywords_str.split(',') if k.strip()]
        session['notes'] = self.notes_edit.text()

        metadata_path = Path(__file__).parent / '_config/_metadata.yaml'
        with open(metadata_path, 'w') as metadata_file:
            yaml.dump(
                self.metadata_settings,
                metadata_file,
                Dumper=SmartDumper,
                default_flow_style=False,
                sort_keys=False,
                indent=2
            )

        self._update_metadata_preview()

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
        record_one_x, record_one_y = (725, 560)
        self.setFixedSize(record_one_x, record_one_y)

        title_label = QLabel('Please select appropriate directories (with config files or executables in them)', self.Record)
        title_label.setFont(QFont(self.font_id, 13+self.font_size_increase))
        title_label.setStyleSheet('QLabel { font-weight: bold;}')
        title_label.move(5, 10)

        avisoft_exe_dir_label = QLabel('Avisoft Recorder directory:', self.Record)
        avisoft_exe_dir_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        avisoft_exe_dir_label.move(5, 40)
        self.recorder_settings_edit = QLineEdit(self.avisoft_rec_dir_global, self.Record)
        self.recorder_settings_edit.setPlaceholderText('Select Avisoft Recorder USGH directory')
        update_avisoft_recorder_dir = partial(self._update_nested_dict_value, self.exp_settings_dict, ('avisoft_recorder_exe_directory',))
        self.recorder_settings_edit.textChanged.connect(update_avisoft_recorder_dir)
        self.recorder_settings_edit.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.recorder_settings_edit.setStyleSheet('QLineEdit { width: 400px; }')
        self.recorder_settings_edit.move(220, 40)
        recorder_dir_btn = QPushButton('Browse', self.Record)
        recorder_dir_btn.setFont(QFont(self.font_id, 8+self.font_size_increase))
        recorder_dir_btn.move(625, 40)
        recorder_dir_btn.setStyleSheet('QPushButton { min-width: 64px; min-height: 12px; max-width: 64px; max-height: 13px; }')
        open_avisoft_recorder_dir_dialog = partial(self._open_directory_dialog, self.recorder_settings_edit, 'Select Avisoft Recorder directory')
        recorder_dir_btn.clicked.connect(open_avisoft_recorder_dir_dialog)

        avisoft_base_dir_label = QLabel('Avisoft base directory:', self.Record)
        avisoft_base_dir_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        avisoft_base_dir_label.move(5, 70)
        self.avisoft_base_edit = QLineEdit(self.avisoft_base_dir_global, self.Record)
        self.avisoft_base_edit.setPlaceholderText('Select Avisoft base directory')
        update_avisoft_base_dir = partial(self._update_nested_dict_value, self.exp_settings_dict, ('avisoft_basedirectory',))
        self.avisoft_base_edit.textChanged.connect(update_avisoft_base_dir)
        self.avisoft_base_edit.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.avisoft_base_edit.setStyleSheet('QLineEdit { width: 400px; }')
        self.avisoft_base_edit.move(220, 70)
        avisoft_base_dir_btn = QPushButton('Browse', self.Record)
        avisoft_base_dir_btn.setFont(QFont(self.font_id, 8+self.font_size_increase))
        avisoft_base_dir_btn.move(625, 70)
        avisoft_base_dir_btn.setStyleSheet('QPushButton { min-width: 64px; min-height: 12px; max-width: 64px; max-height: 13px; }')
        open_avisoft_base_dir_dialog = partial(self._open_directory_dialog, self.avisoft_base_edit, 'Select Avisoft sase directory')
        avisoft_base_dir_btn.clicked.connect(open_avisoft_base_dir_dialog)

        avisoft_config_dir_label = QLabel('Avisoft config directory:', self.Record)
        avisoft_config_dir_label.setFont(QFont(self.font_id, 12 + self.font_size_increase))
        avisoft_config_dir_label.move(5, 100)
        self.avisoft_config_edit = QLineEdit(self.avisoft_config_dir_global, self.Record)
        self.avisoft_config_edit.setPlaceholderText('Select Avisoft config directory (must be on C:\\ drive!)')
        update_avisoft_config_dir = partial(self._update_nested_dict_value, self.exp_settings_dict, ('avisoft_config_directory',))
        self.avisoft_config_edit.textChanged.connect(update_avisoft_config_dir)
        self.avisoft_config_edit.setFont(QFont(self.font_id, 10 + self.font_size_increase))
        self.avisoft_config_edit.setStyleSheet('QLineEdit { width: 400px; }')
        self.avisoft_config_edit.move(220, 100)
        avisoft_config_dir_btn = QPushButton('Browse', self.Record)
        avisoft_config_dir_btn.setFont(QFont(self.font_id, 8 + self.font_size_increase))
        avisoft_config_dir_btn.move(625, 100)
        avisoft_config_dir_btn.setStyleSheet('QPushButton { min-width: 64px; min-height: 12px; max-width: 64px; max-height: 13px; }')
        open_avisoft_config_dir_dialog = partial(self._open_directory_dialog, self.avisoft_config_edit, 'Select Avisoft config directory')
        avisoft_config_dir_btn.clicked.connect(open_avisoft_config_dir_dialog)

        coolterm_base_dir_label = QLabel('CoolTerm base directory:', self.Record)
        coolterm_base_dir_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        coolterm_base_dir_label.move(5, 130)
        self.coolterm_base_edit = QLineEdit(self.coolterm_base_dir_global, self.Record)
        self.coolterm_base_edit.setPlaceholderText('Select Coolterm base directory')
        update_coolterm_base_dir = partial(self._update_nested_dict_value, self.exp_settings_dict, ('coolterm_basedirectory',))
        self.coolterm_base_edit.textChanged.connect(update_coolterm_base_dir)
        self.coolterm_base_edit.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.coolterm_base_edit.setStyleSheet('QLineEdit { width: 400px; }')
        self.coolterm_base_edit.move(220, 130)
        coolterm_base_dir_btn = QPushButton('Browse', self.Record)
        coolterm_base_dir_btn.setFont(QFont(self.font_id, 8+self.font_size_increase))
        coolterm_base_dir_btn.move(625, 130)
        coolterm_base_dir_btn.setStyleSheet('QPushButton { min-width: 64px; min-height: 12px; max-width: 64px; max-height: 13px; }')
        open_coolterm_base_dir_dialog = partial(self._open_directory_dialog, self.coolterm_base_edit, 'Select Coolterm directory')
        coolterm_base_dir_btn.clicked.connect(open_coolterm_base_dir_dialog)

        recording_credentials_dir_label = QLabel('Credentials directory:', self.Record)
        recording_credentials_dir_label.setFont(QFont(self.font_id, 12 + self.font_size_increase))
        recording_credentials_dir_label.move(5, 160)
        self.recording_credentials_dir_edit = QLineEdit(self.recording_credentials_dir_global, self.Record)
        self.recording_credentials_dir_edit.setPlaceholderText('Credentials directory')
        update_recording_credentials_dir = partial(self._update_nested_dict_value, self.exp_settings_dict, ('credentials_directory',))
        self.recording_credentials_dir_edit.textChanged.connect(update_recording_credentials_dir)
        self.recording_credentials_dir_edit.setFont(QFont(self.font_id, 10 + self.font_size_increase))
        self.recording_credentials_dir_edit.setStyleSheet('QLineEdit { width: 400px; }')
        self.recording_credentials_dir_edit.move(220, 160)
        recording_credentials_dir_btn = QPushButton('Browse', self.Record)
        recording_credentials_dir_btn.setFont(QFont(self.font_id, 8 + self.font_size_increase))
        recording_credentials_dir_btn.move(625, 160)
        recording_credentials_dir_btn.setStyleSheet('QPushButton { min-width: 64px; min-height: 12px; max-width: 64px; max-height: 13px; }')
        open_recording_credentials_dir_dialog = partial(self._open_directory_dialog, self.recording_credentials_dir_edit, 'Select credentials directory')
        recording_credentials_dir_btn.clicked.connect(open_recording_credentials_dir_dialog)

        # recording files destination directories (across OS)
        recording_files_destinations_label = QLabel('Select all desirable lab CUP destinations for your files:', self.Record)
        recording_files_destinations_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        recording_files_destinations_label.move(5, 190)

        for lab_idx, lab in enumerate(self.exp_settings_dict['recording_files_destinations_all']):
            self._create_checkbox_file_destinations(lab_id=lab, x_start=5+(lab_idx*365))

        # set main recording parameters
        parameters_label = QLabel('Please set main recording parameters', self.Record)
        parameters_label.setFont(QFont(self.font_id, 13+self.font_size_increase))
        parameters_label.setStyleSheet('QLabel { font-weight: bold;}')
        parameters_label.move(5, 270)

        conduct_audio_label = QLabel('Conduct AUDIO recording:', self.Record)
        conduct_audio_label.setFont(QFont(self.font_id, 11+self.font_size_increase))
        conduct_audio_label.setStyleSheet('QLabel { color: #F58025; font-weight: bold;}')
        conduct_audio_label.move(5, 300)
        self.conduct_audio_cb_list = [x for _, x in sorted(zip([self.exp_settings_dict['conduct_audio_recording'], not self.exp_settings_dict['conduct_audio_recording']], self.boolean_list), reverse=True)]
        self.conduct_audio_cb = QComboBox(self.Record)
        self.conduct_audio_cb.addItems(self.conduct_audio_cb_list)
        self.conduct_audio_cb.setStyleSheet('QComboBox { width: 465px; }')
        self.conduct_audio_cb.activated.connect(partial(self._combo_box_prior_true if self.conduct_audio_cb_list[0] == 'Yes' else self._combo_box_prior_false, variable_id='conduct_audio_cb_bool'))
        self.conduct_audio_cb.move(220, 300)

        conduct_tracking_cal_label = QLabel('Conduct VIDEO calibration:', self.Record)
        conduct_tracking_cal_label.setFont(QFont(self.font_id, 11+self.font_size_increase))
        conduct_tracking_cal_label.setStyleSheet('QLabel { color: #F58025; font-weight: bold;}')
        conduct_tracking_cal_label.move(5, 330)
        self.conduct_tracking_calibration_cb_list = [x for _, x in sorted(zip([self.exp_settings_dict['conduct_tracking_calibration'], not self.exp_settings_dict['conduct_tracking_calibration']], self.boolean_list), reverse=True)]
        self.conduct_tracking_calibration_cb = QComboBox(self.Record)
        self.conduct_tracking_calibration_cb.addItems(self.conduct_tracking_calibration_cb_list)
        self.conduct_tracking_calibration_cb.setStyleSheet('QComboBox { width: 465px; }')
        self.conduct_tracking_calibration_cb.activated.connect(partial(self._combo_box_prior_true if self.conduct_tracking_calibration_cb_list[0] == 'Yes' else self._combo_box_prior_false, variable_id='conduct_tracking_calibration_cb_bool'))
        self.conduct_tracking_calibration_cb.move(220, 330)

        disable_ethernet_label = QLabel('Disable ethernet connection:', self.Record)
        disable_ethernet_label.setFont(QFont(self.font_id, 11+self.font_size_increase))
        disable_ethernet_label.setStyleSheet('QLabel { color: #F58025; font-weight: bold;}')
        disable_ethernet_label.move(5, 360)
        self.disable_ethernet_cb_list = [x for _, x in sorted(zip([self.exp_settings_dict['disable_ethernet'], not self.exp_settings_dict['disable_ethernet']], self.boolean_list), reverse=True)]
        self.disable_ethernet_cb = QComboBox(self.Record)
        self.disable_ethernet_cb.addItems(self.disable_ethernet_cb_list)
        self.disable_ethernet_cb.setStyleSheet('QComboBox { width: 465px; }')
        self.disable_ethernet_cb.activated.connect(partial(self._combo_box_prior_true if self.disable_ethernet_cb_list[0] == 'Yes' else self._combo_box_prior_false, variable_id='disable_ethernet_cb_bool'))
        self.disable_ethernet_cb.move(220, 360)

        video_duration_label = QLabel('Video session duration (min):', self.Record)
        video_duration_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        video_duration_label.move(5, 390)
        self.video_session_duration = QLineEdit(f"{self.exp_settings_dict['video_session_duration']}", self.Record)
        self.video_session_duration.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.video_session_duration.setStyleSheet('QLineEdit { width: 493px; }')
        self.video_session_duration.move(220, 390)

        cal_duration_label = QLabel('Calibration duration (min):', self.Record)
        cal_duration_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        cal_duration_label.move(5, 420)
        self.calibration_session_duration = QLineEdit(f"{self.exp_settings_dict['calibration_duration']}", self.Record)
        self.calibration_session_duration.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.calibration_session_duration.setStyleSheet('QLineEdit { width: 493px; }')
        self.calibration_session_duration.move(220, 420)

        ethernet_network_label = QLabel('Ethernet network ID:', self.Record)
        ethernet_network_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        ethernet_network_label.move(5, 450)
        self.ethernet_network = QLineEdit(f"{self.exp_settings_dict['ethernet_network']}", self.Record)
        self.ethernet_network.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.ethernet_network.setStyleSheet('QLineEdit { width: 493px; }')
        self.ethernet_network.move(220, 450)

        email_notification_label = QLabel('Notify e-mail(s) of PC usage:', self.Record)
        email_notification_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        email_notification_label.move(5, 480)
        self.email_recipients = QLineEdit('', self.Record)
        self.email_recipients.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.email_recipients.setStyleSheet('QLineEdit { width: 493px; }')
        self.email_recipients.move(220, 480)

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
        self.VideoSettings = VideoSettings(self)
        self.setWindowTitle(f'{app_name} (Record > Audio and Video Settings)')
        self.setCentralWidget(self.VideoSettings)
        record_two_x, record_two_y = (450, 530)
        self.setFixedSize(record_two_x, record_two_y)

        gas_label = QLabel('Audio settings', self.VideoSettings)
        gas_label.setFont(QFont(self.font_id, 13 + self.font_size_increase))
        gas_label.setStyleSheet('QLabel { font-weight: bold;}')
        gas_label.move(5, 10)

        usgh_sync_label = QLabel('USGH devices synchronized:', self.VideoSettings)
        usgh_sync_label.setFont(QFont(self.font_id, 12 + self.font_size_increase))
        usgh_sync_label.move(5, 40)
        self.usgh_devices_sync_cb_list = [x for _, x in sorted(zip([self.usgh_devices_sync_cb_bool, not self.usgh_devices_sync_cb_bool], self.boolean_list), reverse=True)]
        self.usgh_devices_sync_cb = QComboBox(self.VideoSettings)
        self.usgh_devices_sync_cb.addItems(self.usgh_devices_sync_cb_list)
        self.usgh_devices_sync_cb.setStyleSheet('QComboBox { width: 120px; }')
        self.usgh_devices_sync_cb.activated.connect(partial(self._combo_box_prior_true if self.usgh_devices_sync_cb_list[0] == 'Yes' else self._combo_box_prior_false, variable_id='usgh_devices_sync_cb_bool'))
        self.usgh_devices_sync_cb.move(260, 40)

        usgh_sr_label = QLabel('USGH devices sampling rate (Hz):', self.VideoSettings)
        usgh_sr_label.setFont(QFont(self.font_id, 12 + self.font_size_increase))
        usgh_sr_label.move(5, 70)
        self.usgh_sr_cb_list = sorted(['250000', '300000'], key=lambda x: x == str(self.exp_settings_dict['audio']['devices']['fabtast']), reverse=True)
        self.usgh_sr_cb = QComboBox(self.VideoSettings)
        self.usgh_sr_cb.addItems(self.usgh_sr_cb_list)
        self.usgh_sr_cb.setStyleSheet('QComboBox { width: 120px; }')
        self.usgh_sr_cb.activated.connect(partial(self._combo_box_usgh_sr, variable_id='usgh_sr'))
        self.usgh_sr_cb.move(260, 70)

        cpu_priority_label = QLabel('CPU priority (USGH Recorder):', self.VideoSettings)
        cpu_priority_label.setFont(QFont(self.font_id, 12 + self.font_size_increase))
        cpu_priority_label.move(5, 100)
        self.cpu_priority_cb_list = sorted(['', 'Realtime', 'High', 'Above normal', 'Normal', 'Below normal', 'Low'], key=lambda x: x == str(self.exp_settings_dict['audio']['cpu_priority']), reverse=True)
        self.cpu_priority_cb = QComboBox(self.VideoSettings)
        self.cpu_priority_cb.addItems(self.cpu_priority_cb_list)
        self.cpu_priority_cb.setStyleSheet('QComboBox { width: 120px; }')
        self.cpu_priority_cb.activated.connect(partial(self._combo_box_cpu_priority, variable_id='cpu_priority'))
        self.cpu_priority_cb.move(260, 100)

        cpu_affinity_label = QLabel('CPU affinity (USGH Recorder):', self.VideoSettings)
        cpu_affinity_label.setFont(QFont(self.font_id, 12 + self.font_size_increase))
        cpu_affinity_label.move(5, 130)
        self.cpu_affinity_edit = QLineEdit(','.join([str(x) for x in self.exp_settings_dict['audio']['cpu_affinity']]), self.VideoSettings)
        self.cpu_affinity_edit.setFixedWidth(150)
        self.cpu_affinity_edit.setFont(QFont(self.font_id, 10 + self.font_size_increase))
        self.cpu_affinity_edit.move(260, 130)

        gvs_label = QLabel('Video settings', self.VideoSettings)
        gvs_label.setFont(QFont(self.font_id, 13+self.font_size_increase))
        gvs_label.setStyleSheet('QLabel { font-weight: bold;}')
        gvs_label.move(5, 170)

        use_cam_label = QLabel('Select camera(s) you wish to use for behavioral recording:', self.VideoSettings)
        use_cam_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        use_cam_label.move(5, 200)

        for cam_idx, cam in enumerate(self.exp_settings_dict['video']['general']['available_cameras']):
            self._create_checkbox_general(camera_id=cam, x_start=5+(cam_idx*90))

        """
        'nvenc-fast-yuv420_A' : '-preset','fast','-qmin','15','-qmax','15'
        'nvenc-fast-yuv420_B' : '-preset','fast','-qmin','15','-qmax','18'
        'nvenc-ll-yuv420'     : '-preset', 'lossless', '-pix_fmt', 'yuv420p'
        """

        rec_codec_label = QLabel('Recording codec:', self.VideoSettings)
        rec_codec_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        rec_codec_label.move(5, 260)
        self.recording_codec_list = sorted(['hq', 'hq-fast', 'mq', 'lq', 'nvenc-fast-yuv420_A',
                                            'nvenc-fast-yuv420_B','nvenc-ll-yuv420'], key=lambda x: x == self.recording_codec, reverse=True)
        self.recording_codec_cb = QComboBox(self.VideoSettings)
        self.recording_codec_cb.addItems(self.recording_codec_list)
        self.recording_codec_cb.setStyleSheet('QComboBox { width: 120px; }')
        self.recording_codec_cb.activated.connect(partial(self._combo_box_prior_codec, variable_id='recording_codec'))
        self.recording_codec_cb.move(160, 260)

        monitor_rec_label = QLabel('Monitor recording:', self.VideoSettings)
        monitor_rec_label.setFont(QFont(self.font_id, 11+self.font_size_increase))
        monitor_rec_label.setStyleSheet('QLabel { color: #F58025; font-weight: bold;}')
        monitor_rec_label.move(5, 290)
        self.monitor_recording_cb_list = [x for _, x in sorted(zip([self.exp_settings_dict['video']['general']['monitor_recording'], not self.exp_settings_dict['video']['general']['monitor_recording']], self.boolean_list), reverse=True)]
        self.monitor_recording_cb = QComboBox(self.VideoSettings)
        self.monitor_recording_cb.addItems(self.monitor_recording_cb_list)
        self.monitor_recording_cb.setStyleSheet('QComboBox { width: 120px; }')
        self.monitor_recording_cb.activated.connect(partial(self._combo_box_prior_true if self.monitor_recording_cb_list[0] == 'Yes' else self._combo_box_prior_false, variable_id='monitor_recording_cb_bool'))
        self.monitor_recording_cb.move(160, 290)

        monitor_cam_label = QLabel('Monitor ONE camera:', self.VideoSettings)
        monitor_cam_label.setFont(QFont(self.font_id, 11+self.font_size_increase))
        monitor_cam_label.setStyleSheet('QLabel { color: #F58025; font-weight: bold;}')
        monitor_cam_label.move(5, 320)
        self.monitor_specific_camera_cb_list = [x for _, x in sorted(zip([self.exp_settings_dict['video']['general']['monitor_specific_camera'], not self.exp_settings_dict['video']['general']['monitor_specific_camera']], self.boolean_list), reverse=True)]
        self.monitor_specific_camera_cb = QComboBox(self.VideoSettings)
        self.monitor_specific_camera_cb.addItems(self.monitor_specific_camera_cb_list)
        self.monitor_specific_camera_cb.setStyleSheet('QComboBox { width: 120px; }')
        self.monitor_specific_camera_cb.activated.connect(partial(self._combo_box_prior_true if self.monitor_specific_camera_cb_list[0] == 'Yes' else self._combo_box_prior_false, variable_id='monitor_specific_camera_cb_bool'))
        self.monitor_specific_camera_cb.move(160, 320)

        specific_camera_serial_label = QLabel('ONE camera serial:', self.VideoSettings)
        specific_camera_serial_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        specific_camera_serial_label.move(5, 350)
        self.specific_camera_serial_list = sorted(self.exp_settings_dict['video']['general']['expected_cameras'], key=lambda x: x == self.exp_settings_dict['video']['general']['specific_camera_serial'], reverse=True)
        self.specific_camera_serial_cb = QComboBox(self.VideoSettings)
        self.specific_camera_serial_cb.addItems(self.specific_camera_serial_list)
        self.specific_camera_serial_cb.setStyleSheet('QComboBox { width: 120px; }')
        self.specific_camera_serial_cb.activated.connect(partial(self._combo_box_specific_camera, variable_id='specific_camera_serial'))
        self.specific_camera_serial_cb.move(160, 350)

        delete_post_copy_label = QLabel('Delete post copy:', self.VideoSettings)
        delete_post_copy_label.setFont(QFont(self.font_id, 11+self.font_size_increase))
        delete_post_copy_label.setStyleSheet('QLabel { color: #F58025; font-weight: bold;}')
        delete_post_copy_label.move(5, 380)
        self.delete_post_copy_cb_list = [x for _, x in sorted(zip([self.exp_settings_dict['video']['general']['delete_post_copy'], not self.exp_settings_dict['video']['general']['delete_post_copy']], self.boolean_list), reverse=True)]
        self.delete_post_copy_cb = QComboBox(self.VideoSettings)
        self.delete_post_copy_cb.addItems(self.delete_post_copy_cb_list )
        self.delete_post_copy_cb.setStyleSheet('QComboBox { width: 120px; }')
        self.delete_post_copy_cb.activated.connect(partial(self._combo_box_prior_true if self.delete_post_copy_cb_list[0] == 'Yes' else self._combo_box_prior_false, variable_id='delete_post_copy_cb_bool'))
        self.delete_post_copy_cb.move(160, 380)

        self.cal_fr_label = QLabel('Calibration (10 fps):', self.VideoSettings)
        self.cal_fr_label.setFixedWidth(150)
        self.cal_fr_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        self.cal_fr_label.move(5, 410)
        self.calibration_frame_rate = QSlider(Qt.Orientation.Horizontal, self.VideoSettings)
        self.calibration_frame_rate.setFixedWidth(150)
        self.calibration_frame_rate.move(160, 415)
        self.calibration_frame_rate.setRange(10, 150)
        self.calibration_frame_rate.setValue(self.exp_settings_dict['video']['general']['calibration_frame_rate'])
        self.calibration_frame_rate.valueChanged.connect(self._update_cal_fr_label)

        self.fr_label = QLabel('Recording (150 fps):', self.VideoSettings)
        self.fr_label.setFixedWidth(150)
        self.fr_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        self.fr_label.move(5, 440)
        self.cameras_frame_rate = QSlider(Qt.Orientation.Horizontal, self.VideoSettings)
        self.cameras_frame_rate.setFixedWidth(150)
        self.cameras_frame_rate.move(160, 445)
        self.cameras_frame_rate.setRange(10, 150)
        self.cameras_frame_rate.setValue(self.exp_settings_dict['video']['general']['recording_frame_rate'])
        self.cameras_frame_rate.valueChanged.connect(self._update_fr_label)

        self._create_buttons_record(seq=1, class_option=self.VideoSettings,
                                    button_pos_y=record_two_y - 35, next_button_x_pos=record_two_x - 100)

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

        self.remove_subject_buttons.clear()
        self.edit_intervention_widgets.clear()

        self.VideoSettings = VideoSettings(self)
        self.setWindowTitle(f'{app_name} (Record > Metadata)')
        self.setCentralWidget(self.VideoSettings)
        record_three_x, record_three_y = (950, 1050)
        self.setFixedSize(record_three_x, record_three_y)

        y_pos = 10
        x_label = 10
        x_widget = 160
        widget_width = 250

        title_label = QLabel('Session Metadata', self.VideoSettings)
        title_label.setFont(QFont(self.font_id, 13 + self.font_size_increase))
        title_label.setStyleSheet('QLabel { font-weight: bold;}')
        title_label.move(x_label, y_pos)
        y_pos += 30

        institution_label = QLabel('Institution:', self.VideoSettings)
        institution_label.setFont(QFont(self.font_id, 12 + self.font_size_increase))
        institution_label.move(x_label, y_pos)
        self.institution_edit = QLineEdit(self.metadata_settings['Session']['institution'], self.VideoSettings)
        self.institution_edit.setFont(QFont(self.font_id, 10 + self.font_size_increase))
        self.institution_edit.setFixedWidth(widget_width)
        self.institution_edit.move(x_widget, y_pos)
        self.institution_edit.textChanged.connect(self._save_metadata_to_yaml)
        y_pos += 30

        lab_label = QLabel('Lab:', self.VideoSettings)
        lab_label.setFont(QFont(self.font_id, 12 + self.font_size_increase))
        lab_label.move(x_label, y_pos)
        self.lab_edit = QLineEdit(self.metadata_settings['Session']['lab'], self.VideoSettings)
        self.lab_edit.setFont(QFont(self.font_id, 10 + self.font_size_increase))
        self.lab_edit.setFixedWidth(widget_width)
        self.lab_edit.move(x_widget, y_pos)
        self.lab_edit.textChanged.connect(self._save_metadata_to_yaml)
        y_pos += 30

        experimenter_label = QLabel('Experimenter:', self.VideoSettings)
        experimenter_label.setFont(QFont(self.font_id, 12 + self.font_size_increase))
        experimenter_label.move(x_label, y_pos)
        self.experimenter_edit = QLineEdit(self.exp_id, self.VideoSettings)
        self.experimenter_edit.setFont(QFont(self.font_id, 10 + self.font_size_increase))
        self.experimenter_edit.setFixedWidth(widget_width)
        self.experimenter_edit.move(x_widget, y_pos)
        self.experimenter_edit.textChanged.connect(self._save_metadata_to_yaml)
        y_pos += 30

        ambient_light_label = QLabel('Ambient Light:', self.VideoSettings)
        ambient_light_label.setFont(QFont(self.font_id, 11 + self.font_size_increase))
        ambient_light_label.move(x_label, y_pos)

        self.ambient_light_bool = self.metadata_settings['Session'].get('ambient_light', False)
        on_off_list = ['on', 'off'] if self.ambient_light_bool else ['off', 'on']

        self.ambient_light_cb = QComboBox(self.VideoSettings)
        self.ambient_light_cb.addItems(on_off_list)
        self.ambient_light_cb.setStyleSheet(f'QComboBox {{ width: 220px; }}')
        self.ambient_light_cb.move(x_widget, y_pos)

        def on_ambient_light_changed(index):
            self.ambient_light_bool = (self.ambient_light_cb.currentText() == 'on')
            self._save_metadata_to_yaml()

        self.ambient_light_cb.currentIndexChanged.connect(on_ambient_light_changed)
        y_pos += 30

        exp_code_label = QLabel('Experiment Code:', self.VideoSettings)
        exp_code_label.setFont(QFont(self.font_id, 12 + self.font_size_increase))
        exp_code_label.move(x_label, y_pos)
        self.session_experiment_code_edit = QLineEdit(self.metadata_settings['Session']['session_experiment_code'], self.VideoSettings)
        self.session_experiment_code_edit.setPlaceholderText('e.g., ECL2MSFSd')
        self.session_experiment_code_edit.setFont(QFont(self.font_id, 10 + self.font_size_increase))
        self.session_experiment_code_edit.setFixedWidth(widget_width)
        self.session_experiment_code_edit.move(x_widget, y_pos)
        self.session_experiment_code_edit.textChanged.connect(self._save_metadata_to_yaml)
        y_pos += 30

        calibration_label = QLabel('Calibration Session:', self.VideoSettings)
        calibration_label.setFont(QFont(self.font_id, 12 + self.font_size_increase))
        calibration_label.move(x_label, y_pos)
        self.calibration_session_edit = QLineEdit(self.metadata_settings['Session']['calibration_session'], self.VideoSettings)
        self.calibration_session_edit.setPlaceholderText('Tracking calibration session')
        self.calibration_session_edit.setFont(QFont(self.font_id, 10 + self.font_size_increase))
        self.calibration_session_edit.setFixedWidth(widget_width)
        self.calibration_session_edit.move(x_widget, y_pos)
        self.calibration_session_edit.textChanged.connect(self._save_metadata_to_yaml)
        y_pos += 30

        playback_file_label = QLabel('USV Playback File:', self.VideoSettings)
        playback_file_label.setFont(QFont(self.font_id, 12 + self.font_size_increase))
        playback_file_label.move(x_label, y_pos)
        self.session_usv_playback_file_edit = QLineEdit(self.metadata_settings['Session']['session_usv_playback_file'], self.VideoSettings)
        self.session_usv_playback_file_edit.setPlaceholderText('Playback .wav file (if any)')
        self.session_usv_playback_file_edit.setFont(QFont(self.font_id, 10 + self.font_size_increase))
        self.session_usv_playback_file_edit.setFixedWidth(widget_width)
        self.session_usv_playback_file_edit.move(x_widget, y_pos)
        self.session_usv_playback_file_edit.textChanged.connect(self._save_metadata_to_yaml)
        y_pos += 30

        description_label = QLabel('Session Description:', self.VideoSettings)
        description_label.setFont(QFont(self.font_id, 12 + self.font_size_increase))
        description_label.move(x_label, y_pos)
        self.session_description_edit = QLineEdit(self.metadata_settings['Session']['session_description'], self.VideoSettings)
        self.session_description_edit.setFont(QFont(self.font_id, 10 + self.font_size_increase))
        self.session_description_edit.setFixedWidth(widget_width)
        self.session_description_edit.move(x_widget, y_pos)
        self.session_description_edit.textChanged.connect(self._save_metadata_to_yaml)
        y_pos += 30

        keywords_label = QLabel('Keywords:', self.VideoSettings)
        keywords_label.setFont(QFont(self.font_id, 12 + self.font_size_increase))
        keywords_label.move(x_label, y_pos)
        self.keywords_edit = QLineEdit(','.join(self.metadata_settings['Session']['keywords']), self.VideoSettings)
        self.keywords_edit.setPlaceholderText('Comma-separated, e.g., social, e-phys, etc.')
        self.keywords_edit.setFont(QFont(self.font_id, 10 + self.font_size_increase))
        self.keywords_edit.setFixedWidth(widget_width)
        self.keywords_edit.move(x_widget, y_pos)
        self.keywords_edit.textChanged.connect(self._save_metadata_to_yaml)
        y_pos += 30

        notes_label = QLabel('Extra Notes:', self.VideoSettings)
        notes_label.setFont(QFont(self.font_id, 12 + self.font_size_increase))
        notes_label.move(x_label, y_pos)
        self.notes_edit = QLineEdit(self.metadata_settings['Session']['notes'], self.VideoSettings)
        self.notes_edit.setFont(QFont(self.font_id, 10 + self.font_size_increase))
        self.notes_edit.setFixedWidth(widget_width)
        self.notes_edit.move(x_widget, y_pos)
        self.notes_edit.textChanged.connect(self._save_metadata_to_yaml)
        y_pos += 30

        # Equipment
        y_pos += 10
        equipment_title_label = QLabel('Equipment', self.VideoSettings)
        equipment_title_label.setFont(QFont(self.font_id, 13 + self.font_size_increase))
        equipment_title_label.setStyleSheet('QLabel { font-weight: bold;}')
        equipment_title_label.move(x_label, y_pos)
        y_pos += 40

        equipment_items = []
        for category, devices in self.equipment_settings_dict.items():
            for device_name in devices.keys():
                equipment_items.append(f"{category}.{device_name}")

        items_per_row = 3
        horizontal_spacing = 135
        vertical_spacing = 30

        for i, equipment_key in enumerate(equipment_items):

            row = i // items_per_row
            col = i % items_per_row

            checkbox_x = x_label + (col * horizontal_spacing)
            checkbox_y = y_pos + (row * vertical_spacing)

            category, device_name = equipment_key.split('.')
            label_text = f"{category} [{device_name}]"
            yaml_key = equipment_key.replace('.', '_')
            equipment_dict = self.metadata_settings.get('Equipment')
            is_checked = bool(equipment_dict and yaml_key in equipment_dict)

            checkbox = QCheckBox(label_text, self.VideoSettings)
            checkbox.setFont(QFont(self.font_id, 11 + self.font_size_increase))
            checkbox.setStyleSheet("""QCheckBox {spacing: 5px;}
                                      QCheckBox::indicator {border: 1px solid grey; width: 15px; height: 15px; border-radius: 7.5px;}
                                      QCheckBox::indicator:checked {background-color: #F58025;}""")
            checkbox.setChecked(is_checked)

            checkbox.move(checkbox_x, checkbox_y)

            checkbox.stateChanged.connect(partial(self._on_equipment_checkbox_toggled, equipment_key=equipment_key))
            setattr(self, f"equip_cb_{yaml_key}", checkbox)

        num_rows = (len(equipment_items) + items_per_row - 1) // items_per_row
        y_pos += num_rows * vertical_spacing

        # Subjects
        y_pos += 20
        subjects_title_label = QLabel('Add Subjects', self.VideoSettings)
        subjects_title_label.setFont(QFont(self.font_id, 13 + self.font_size_increase))
        subjects_title_label.setStyleSheet('QLabel { font-weight: bold;}')
        subjects_title_label.move(x_label, y_pos)
        subjects_y_pos = y_pos
        y_pos += 30

        subject_fields = {
            'subject_id': ('Subject ID:', '181181_0'), 'species': ('Species:', 'mus musculus'),
            'genotype_strain': ('Genotype-Strain:', 'CD1-WT'), 'sex': ('Sex:', None),
            'dob': ('DOB:', '2025-10-12'), 'weight': ('Weight:', '35'), 'housing': ('Housing:', None),
            'estrous_stage': ('Estrous Stage:', 'E'), 'estrous_sample_time': ('Estrous Sample Time:', '18:25')
        }

        required_fields = ['subject_id', 'species', 'genotype_strain']

        self.subject_form_widgets.clear()

        for key, (label_text, placeholder) in subject_fields.items():
            label = QLabel(label_text, self.VideoSettings)
            label.setFont(QFont(self.font_id, 12 + self.font_size_increase))
            label.move(x_label, y_pos)

            if key == 'sex':
                widget = QComboBox(self.VideoSettings)
                widget.addItems(['male', 'female'])
            elif key == 'housing':
                widget = QComboBox(self.VideoSettings)
                widget.addItems(['group', 'single'])
            elif key == 'estrous_stage':
                widget = QComboBox(self.VideoSettings)
                widget.addItems(['N/A', 'proestrus', 'estrus', 'metestrus', 'diestrus'])
            else:
                widget = QLineEdit('', self.VideoSettings)
                widget.setPlaceholderText(placeholder)

            widget.setFont(QFont(self.font_id, 10 + self.font_size_increase))
            widget.setFixedWidth(230)
            widget.move(180, y_pos)

            self.subject_form_widgets[key] = widget
            setattr(self, f"{key}_edit", widget)

            if key in required_fields and isinstance(widget, QLineEdit):
                widget.textChanged.connect(self._validate_subject_form)

            if isinstance(widget, QLineEdit):
                widget.textChanged.connect(self._on_subject_form_changed)
            elif isinstance(widget, QComboBox):
                widget.currentIndexChanged.connect(self._on_subject_form_changed)


            y_pos += 30

        subject_ids_from_repo = [s.get('subject_id', '') for s in self.subject_repository if s.get('subject_id')]

        # create a QCompleter with the list of IDs
        completer = QCompleter(subject_ids_from_repo, self)
        completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        completer.setFilterMode(Qt.MatchFlag.MatchContains)

        if hasattr(self, 'subject_id_edit'):
            self.subject_id_edit.setCompleter(completer)

        completer.activated.connect(self._on_subject_selected_from_completer)

        clear_subject_form_btn = QPushButton(QIcon(clear_icon), "Clear Form", self.VideoSettings)
        clear_subject_form_btn.setFont(QFont(self.font_id, 10 + self.font_size_increase))
        clear_subject_form_btn.setStyleSheet('QPushButton { min-width: 90px; min-height: 12px; max-width: 90px; max-height: 13px; }')
        clear_subject_form_btn.move(180, subjects_y_pos)
        clear_subject_form_btn.clicked.connect(self._clear_subject_form)

        self.add_subject_btn = QPushButton(QIcon(add_icon), 'Add Subject', self.VideoSettings)
        self.add_subject_btn.setFont(QFont(self.font_id, 10 + self.font_size_increase))
        self.add_subject_btn.setStyleSheet('QPushButton { min-width: 90px; min-height: 12px; max-width: 90px; max-height: 13px; }')
        self.add_subject_btn.move(297, subjects_y_pos)
        self.add_subject_btn.clicked.connect(self._add_subject)
        self.add_subject_btn.setEnabled(False)

        # intervention for subject
        y_pos += 10
        self.add_chemo_btn = QPushButton(QIcon(add_icon), "chemo", self.VideoSettings)
        self.add_ephys_btn = QPushButton(QIcon(add_icon), "e-phys", self.VideoSettings)
        self.add_lesion_btn = QPushButton(QIcon(add_icon), "lesion", self.VideoSettings)
        self.add_opto_btn = QPushButton(QIcon(add_icon), "opto", self.VideoSettings)

        buttons = [self.add_chemo_btn, self.add_ephys_btn, self.add_lesion_btn, self.add_opto_btn]
        callbacks = [self._open_chemo_dialog, self._open_ephys_dialog, self._open_lesion_dialog, self._open_opto_dialog]

        items_per_row = 4
        horizontal_spacing = 100
        vertical_spacing = 20

        for i, (button, callback) in enumerate(zip(buttons, callbacks)):

            row, col = divmod(i, items_per_row)

            button_x = x_label + (col * horizontal_spacing)
            button_y = y_pos + (row * vertical_spacing)

            button.setStyleSheet('QPushButton { min-width: 70px; min-height: 12px; max-width: 70px; max-height: 13px; }')

            button.move(button_x, button_y)

            button.setEnabled(False)
            button.clicked.connect(callback)

        num_rows = (len(buttons) + items_per_row - 1) // items_per_row
        y_pos += num_rows * vertical_spacing

        # edit interventions title
        edit_interv_title_y = 910
        edit_interv_title_x = 10

        edit_interv_title = QLabel("Edit Subject Interventions", self.VideoSettings)
        edit_interv_title.setFont(QFont(self.font_id, 13 + self.font_size_increase))
        edit_interv_title.setStyleSheet('QLabel { font-weight: bold;}')
        edit_interv_title.move(edit_interv_title_x, edit_interv_title_y - 30)

        # metadata preview
        preview_x = record_three_x - 500
        preview_width = 490
        preview_height = record_three_y - 100

        preview_label = QLabel('Live Preview (_metadata.yaml)', self.VideoSettings)
        preview_label.setFont(QFont(self.font_id, 13 + self.font_size_increase))
        preview_label.setStyleSheet('QLabel { font-weight: bold;}')
        preview_label.move(preview_x, 10)

        self.metadata_preview_edit = QPlainTextEdit(self.VideoSettings)
        self.metadata_preview_edit.setReadOnly(True)
        self.metadata_preview_edit.setFixedSize(preview_width, preview_height)
        self.metadata_preview_edit.move(preview_x, 40)
        self.metadata_preview_edit.setStyleSheet("background-color: #2b2b2b; color: #f8f8f2;")
        self.highlighter = YamlHighlighter(self.metadata_preview_edit.document())

        self._update_subject_ui()
        self._update_metadata_preview()

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
                                            exp_settings_dict=self.exp_settings_dict,
                                            metadata_settings=self.metadata_settings)

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
        process_one_x, process_one_y = (1080, 935)
        self.setFixedSize(process_one_x, process_one_y)

        # column 1

        processing_dir_label = QLabel('(*) Root directories', self.ProcessSettings)
        processing_dir_label.setFont(QFont(self.font_id, 13+self.font_size_increase))
        processing_dir_label.setStyleSheet('QLabel { font-weight: bold;}')
        processing_dir_label.move(85, 10)
        self.processing_dir_edit = QTextEdit('', self.ProcessSettings)
        self.processing_dir_edit.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.processing_dir_edit.move(10, 40)
        self.processing_dir_edit.setFixedSize(295, 290)

        exp_codes_dir_label = QLabel('ExCode', self.ProcessSettings)
        exp_codes_dir_label.setFont(QFont(self.font_id, 13+self.font_size_increase))
        exp_codes_dir_label.setStyleSheet('QLabel { font-weight: bold;}')
        exp_codes_dir_label.move(330, 10)
        self.exp_codes_edit = QTextEdit('', self.ProcessSettings)
        self.exp_codes_edit.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.exp_codes_edit.move(310, 40)
        self.exp_codes_edit.setFixedSize(100, 290)

        self.processing_credentials_dir_edit = QLineEdit(f"{self.processing_input_dict['credentials_directory']}", self.ProcessSettings)
        self.processing_credentials_dir_edit.setPlaceholderText('Credentials directory')
        update_credentials = partial(self._update_nested_dict_value, self.processing_input_dict, ('credentials_directory',))
        self.processing_credentials_dir_edit.textChanged.connect(update_credentials)
        self.processing_credentials_dir_edit.setFont(QFont(self.font_id, 10 + self.font_size_increase))
        self.processing_credentials_dir_edit.setStyleSheet('QLineEdit { width: 290px; }')
        self.processing_credentials_dir_edit.move(10, 335)
        processing_credentials_dir_btn = QPushButton('Browse', self.ProcessSettings)
        processing_credentials_dir_btn.setFont(QFont(self.font_id, 8 + self.font_size_increase))
        processing_credentials_dir_btn.move(310, 335)
        processing_credentials_dir_btn.setStyleSheet('QPushButton { min-width: 77px; min-height: 12px; max-width: 77px; max-height: 13px; }')
        open_dialog = partial(self._open_directory_dialog, self.processing_credentials_dir_edit, 'Select Credentials Directory')
        processing_credentials_dir_btn.clicked.connect(open_dialog)

        self.centroid_model_edit = QLineEdit(f"{self.processing_input_dict['prepare_cluster_job']['centroid_model_path']}", self.ProcessSettings)
        update_centroid_model_dir = partial(self._update_nested_dict_value, self.processing_input_dict, ('prepare_cluster_job', 'centroid_model_path'))
        self.centroid_model_edit.textChanged.connect(update_centroid_model_dir)
        self.centroid_model_edit.setPlaceholderText('SLEAP centroid model directory')
        self.centroid_model_edit.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.centroid_model_edit.setStyleSheet('QLineEdit { width: 295px; }')
        self.centroid_model_edit.move(10, 365)
        centroid_model_btn = QPushButton('Browse', self.ProcessSettings)
        centroid_model_btn.setFont(QFont(self.font_id, 8+self.font_size_increase))
        centroid_model_btn.move(310, 365)
        centroid_model_btn.setStyleSheet('QPushButton { min-width: 77px; min-height: 12px; max-width: 77px; max-height: 13px; }')
        centroid_model_dir_dialog = partial(self._open_directory_dialog, self.centroid_model_edit, 'Select SLEAP centroid model directory')
        centroid_model_btn.clicked.connect(centroid_model_dir_dialog)

        self.centered_instance_model_edit = QLineEdit(f"{self.processing_input_dict['prepare_cluster_job']['centered_instance_model_path']}", self.ProcessSettings)
        update_centered_instance_dir = partial(self._update_nested_dict_value, self.processing_input_dict, ('prepare_cluster_job', 'centered_instance_model_path'))
        self.centered_instance_model_edit.textChanged.connect(update_centered_instance_dir)
        self.centered_instance_model_edit.setPlaceholderText('SLEAP centered instance model directory')
        self.centered_instance_model_edit.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.centered_instance_model_edit.setStyleSheet('QLineEdit { width: 295px; }')
        self.centered_instance_model_edit.move(10, 395)
        centered_instance_btn = QPushButton('Browse', self.ProcessSettings)
        centered_instance_btn.setFont(QFont(self.font_id, 8+self.font_size_increase))
        centered_instance_btn.move(310, 395)
        centered_instance_btn.setStyleSheet('QPushButton { min-width: 77px; min-height: 12px; max-width: 77px; max-height: 13px; }')
        open_centered_instance_dir_dialog = partial(self._open_directory_dialog, self.centered_instance_model_edit, 'Select SLEAP centered instance directory')
        centered_instance_btn.clicked.connect(open_centered_instance_dir_dialog)

        self.inference_root_dir_edit = QLineEdit(self.sleap_inference_dir_global, self.ProcessSettings)
        update_inference_root_dir = partial(self._update_nested_dict_value, self.processing_input_dict, ('prepare_cluster_job', 'inference_root_dir'))
        self.inference_root_dir_edit.textChanged.connect(update_inference_root_dir)
        self.inference_root_dir_edit.setPlaceholderText('SLEAP inference directory')
        self.inference_root_dir_edit.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.inference_root_dir_edit.setStyleSheet('QLineEdit { width: 295px; }')
        self.inference_root_dir_edit.move(10, 425)
        inference_root_dir_btn = QPushButton('Browse', self.ProcessSettings)
        inference_root_dir_btn.setFont(QFont(self.font_id, 8+self.font_size_increase))
        inference_root_dir_btn.move(310, 425)
        inference_root_dir_btn.setStyleSheet('QPushButton { min-width: 77px; min-height: 12px; max-width: 77px; max-height: 13px; }')
        inference_root_dir_dialog = partial(self._open_directory_dialog, self.inference_root_dir_edit, 'Select SLEAP inference directory')
        inference_root_dir_btn.clicked.connect(inference_root_dir_dialog)

        self.calibration_file_loc_edit = QLineEdit('', self.ProcessSettings)
        update_calibration_file_loc = partial(self._update_nested_dict_value, self.processing_input_dict, ('anipose_operations', 'ConvertTo3D', 'translate_rotate_metric', 'original_arena_file_loc'))
        self.calibration_file_loc_edit.textChanged.connect(update_calibration_file_loc)
        update_calibration_file_loc_2 = partial(self._update_nested_dict_value, self.processing_input_dict, ('anipose_operations', 'ConvertTo3D', 'conduct_anipose_triangulation', 'calibration_file_loc'))
        self.calibration_file_loc_edit.textChanged.connect(update_calibration_file_loc_2)
        self.calibration_file_loc_edit.setPlaceholderText('Tracking calibration / Arena root directory')
        self.calibration_file_loc_edit.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.calibration_file_loc_edit.setStyleSheet('QLineEdit { width: 295px; }')
        self.calibration_file_loc_edit.move(10, 455)
        calibration_file_loc_btn = QPushButton('Browse', self.ProcessSettings)
        calibration_file_loc_btn.setFont(QFont(self.font_id, 8+self.font_size_increase))
        calibration_file_loc_btn.move(310, 455)
        calibration_file_loc_btn.setStyleSheet('QPushButton { min-width: 77px; min-height: 12px; max-width: 77px; max-height: 12px; }')
        calibration_file_loc_dialog = partial(self._open_directory_dialog, self.calibration_file_loc_edit, 'Select calibration/arena root directory')
        calibration_file_loc_btn.clicked.connect(calibration_file_loc_dialog)

        das_conda_label = QLabel('DAS conda environment name:', self.ProcessSettings)
        das_conda_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        das_conda_label.move(10, 485)
        self.das_conda = QLineEdit(self.processing_input_dict['usv_inference']['FindMouseVocalizations']['das_command_line_inference']['das_conda_env_name'], self.ProcessSettings)
        self.das_conda.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.das_conda.setStyleSheet('QLineEdit { width: 98px; }')
        self.das_conda.move(310, 485)

        self.das_model_dir_edit = QLineEdit(self.das_model_dir_global, self.ProcessSettings)
        update_das_model_dir = partial(self._update_nested_dict_value, self.processing_input_dict, ('usv_inference', 'FindMouseVocalizations', 'das_command_line_inference', 'das_model_directory'))
        self.das_model_dir_edit.textChanged.connect(update_das_model_dir)
        self.das_model_dir_edit.setPlaceholderText('DAS model directory')
        self.das_model_dir_edit.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.das_model_dir_edit.setStyleSheet('QLineEdit { width: 295px; }')
        self.das_model_dir_edit.move(10, 515)
        das_model_dir_btn = QPushButton('Browse', self.ProcessSettings)
        das_model_dir_btn.setFont(QFont(self.font_id, 8+self.font_size_increase))
        das_model_dir_btn.move(310, 515)
        das_model_dir_btn.setStyleSheet('QPushButton { min-width: 77px; min-height: 12px; max-width: 77px; max-height: 12px; }')
        open_das_model_dir_dialog = partial(self._open_directory_dialog, self.das_model_dir_edit, 'Select DAS model directory')
        das_model_dir_btn.clicked.connect(open_das_model_dir_dialog)

        das_model_base_label = QLabel('DAS model base (timestamp):', self.ProcessSettings)
        das_model_base_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        das_model_base_label.move(10, 545)
        self.das_model_base = QLineEdit(self.processing_input_dict['usv_inference']['FindMouseVocalizations']['das_command_line_inference']['model_name_base'], self.ProcessSettings)
        self.das_model_base.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.das_model_base.setStyleSheet('QLineEdit { width: 183px; }')
        self.das_model_base.move(225, 545)

        vcl_conda_label = QLabel('Vocalocator conda environment name:', self.ProcessSettings)
        vcl_conda_label.setFont(QFont(self.font_id, 12 + self.font_size_increase))
        vcl_conda_label.move(10, 575)
        self.vcl_conda = QLineEdit(self.processing_input_dict['vocalocator']['vcl_conda_env_name'], self.ProcessSettings)
        self.vcl_conda.setFont(QFont(self.font_id, 10 + self.font_size_increase))
        self.vcl_conda.setStyleSheet('QLineEdit { width: 98px; }')
        self.vcl_conda.move(310, 575)

        self.vcl_model_dir_edit = QLineEdit(self.vcl_model_dir_global, self.ProcessSettings)
        update_vcl_model_dir = partial(self._update_nested_dict_value, self.processing_input_dict, ('vocalocator', 'vcl_model_directory'))
        self.vcl_model_dir_edit.textChanged.connect(update_vcl_model_dir)
        self.vcl_model_dir_edit.setPlaceholderText('Vocalocator model directory')
        self.vcl_model_dir_edit.setFont(QFont(self.font_id, 10 + self.font_size_increase))
        self.vcl_model_dir_edit.setStyleSheet('QLineEdit { width: 295px; }')
        self.vcl_model_dir_edit.move(10, 605)
        vcl_model_dir_btn = QPushButton('Browse', self.ProcessSettings)
        vcl_model_dir_btn.setFont(QFont(self.font_id, 8 + self.font_size_increase))
        vcl_model_dir_btn.move(310, 605)
        vcl_model_dir_btn.setStyleSheet('QPushButton { min-width: 77px; min-height: 12px; max-width: 77px; max-height: 12px; }')
        vcl_model_dir_dialog = partial(self._open_directory_dialog, self.vcl_model_dir_edit, 'Select VCL model directory')
        vcl_model_dir_btn.clicked.connect(vcl_model_dir_dialog)

        pc_usage_process_label = QLabel('Notify e-mail(s) of PC usage:', self.ProcessSettings)
        pc_usage_process_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        pc_usage_process_label.move(10, 635)
        self.pc_usage_process = QLineEdit('', self.ProcessSettings)
        self.pc_usage_process.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.pc_usage_process.setStyleSheet('QLineEdit { width: 183px; }')
        self.pc_usage_process.move(225, 635)

        processing_pc_label = QLabel('Processing PC of choice:', self.ProcessSettings)
        processing_pc_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        processing_pc_label.move(10, 665)
        self.loaded_processing_pc_list = sorted(self.processing_input_dict['send_email']['Messenger']['processing_pc_list'], key=lambda x: x == self.processing_input_dict['send_email']['Messenger']['processing_pc_choice'], reverse=True)
        self.processing_pc_cb = QComboBox(self.ProcessSettings)
        self.processing_pc_cb.addItems(self.loaded_processing_pc_list)
        self.processing_pc_cb.setStyleSheet('QComboBox { width: 155px; }')
        self.processing_pc_cb.activated.connect(partial(self._combo_box_prior_processing_pc_choice, variable_id='processing_pc_choice'))
        self.processing_pc_cb.move(225, 665)

        ev_sync_label = QLabel('E-PHYS processing settings', self.ProcessSettings)
        ev_sync_label.setFont(QFont(self.font_id, 13 + self.font_size_increase))
        ev_sync_label.setStyleSheet('QLabel { font-weight: bold;}')
        ev_sync_label.move(10, 705)

        conduct_nv_sync_cb_label = QLabel('Run E/V sync check:', self.ProcessSettings)
        conduct_nv_sync_cb_label.setFont(QFont(self.font_id, 11 + self.font_size_increase))
        conduct_nv_sync_cb_label.setStyleSheet('QLabel { color: #F58025; font-weight: bold;}')
        conduct_nv_sync_cb_label.move(10, 735)
        self.conduct_nv_sync_cb = QComboBox(self.ProcessSettings)
        self.conduct_nv_sync_cb.addItems(['No', 'Yes'])
        self.conduct_nv_sync_cb.setStyleSheet('QComboBox { width: 80px; }')
        self.conduct_nv_sync_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='conduct_nv_sync_cb_bool'))
        self.conduct_nv_sync_cb.move(225, 735)

        conduct_ephys_file_chaining_label = QLabel('Concatenate e-phys files:', self.ProcessSettings)
        conduct_ephys_file_chaining_label.setFont(QFont(self.font_id, 11 + self.font_size_increase))
        conduct_ephys_file_chaining_label.setStyleSheet('QLabel { color: #F58025; font-weight: bold;}')
        conduct_ephys_file_chaining_label.move(10, 765)
        self.conduct_ephys_file_chaining_cb = QComboBox(self.ProcessSettings)
        self.conduct_ephys_file_chaining_cb.addItems(['No', 'Yes'])
        self.conduct_ephys_file_chaining_cb.setStyleSheet('QComboBox { width: 80px; }')
        self.conduct_ephys_file_chaining_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='conduct_ephys_file_chaining_cb_bool'))
        self.conduct_ephys_file_chaining_cb.move(225, 765)

        split_cluster_spikes_cb_label = QLabel('Split clusters to sessions:', self.ProcessSettings)
        split_cluster_spikes_cb_label.setFont(QFont(self.font_id, 11 + self.font_size_increase))
        split_cluster_spikes_cb_label.setStyleSheet('QLabel { color: #F58025; font-weight: bold;}')
        split_cluster_spikes_cb_label.move(10, 795)
        self.split_cluster_spikes_cb = QComboBox(self.ProcessSettings)
        self.split_cluster_spikes_cb.addItems(['No', 'Yes'])
        self.split_cluster_spikes_cb.setStyleSheet('QComboBox { width: 80px; }')
        self.split_cluster_spikes_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='split_cluster_spikes_cb_bool'))
        self.split_cluster_spikes_cb.move(225, 795)

        min_spike_num_label = QLabel('Min num of spikes:', self.ProcessSettings)
        min_spike_num_label.setFont(QFont(self.font_id, 12 + self.font_size_increase))
        min_spike_num_label.move(10, 825)
        self.min_spike_num = QLineEdit(f"{self.processing_input_dict['modify_files']['Operator']['get_spike_times']['min_spike_num']}", self.ProcessSettings)
        self.min_spike_num.setFont(QFont(self.font_id, 10 + self.font_size_increase))
        self.min_spike_num.setStyleSheet('QLineEdit { width: 108px; }')
        self.min_spike_num.move(225, 825)

        # column 2
        column_two_x1 = 440
        column_two_x2 = 630

        gvs_label = QLabel('VIDEO processing settings', self.ProcessSettings)
        gvs_label.setFont(QFont(self.font_id, 13 + self.font_size_increase))
        gvs_label.setStyleSheet('QLabel { font-weight: bold;}')
        gvs_label.move(column_two_x1, 10)

        conduct_video_concatenation_label = QLabel('Run video concatenation:', self.ProcessSettings)
        conduct_video_concatenation_label.setFont(QFont(self.font_id, 11 + self.font_size_increase))
        conduct_video_concatenation_label.setStyleSheet('QLabel { color: #F58025; font-weight: bold;}')
        conduct_video_concatenation_label.move(column_two_x1, 40)
        self.conduct_video_concatenation_cb = QComboBox(self.ProcessSettings)
        self.conduct_video_concatenation_cb.addItems(['No', 'Yes'])
        self.conduct_video_concatenation_cb.setStyleSheet('QComboBox { width: 80px; }')
        self.conduct_video_concatenation_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='conduct_video_concatenation_cb_bool'))
        self.conduct_video_concatenation_cb.move(column_two_x2, 40)

        conduct_video_fps_change_cb_label = QLabel('Run video re-encoding:', self.ProcessSettings)
        conduct_video_fps_change_cb_label.setFont(QFont(self.font_id, 11 + self.font_size_increase))
        conduct_video_fps_change_cb_label.setStyleSheet('QLabel { color: #F58025; font-weight: bold;}')
        conduct_video_fps_change_cb_label.move(column_two_x1, 70)
        self.conduct_video_fps_change_cb = QComboBox(self.ProcessSettings)
        self.conduct_video_fps_change_cb.addItems(['No', 'Yes'])
        self.conduct_video_fps_change_cb.setStyleSheet('QComboBox { width: 80px; }')
        self.conduct_video_fps_change_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='conduct_video_fps_change_cb_bool'))
        self.conduct_video_fps_change_cb.move(column_two_x2, 70)

        conversion_target_file_label = QLabel('Concatenation name:', self.ProcessSettings)
        conversion_target_file_label.setFont(QFont(self.font_id, 12 + self.font_size_increase))
        conversion_target_file_label.move(column_two_x1, 100)
        self.conversion_target_file = QLineEdit(self.processing_input_dict['modify_files']['Operator']['rectify_video_fps']['conversion_target_file'], self.ProcessSettings)
        self.conversion_target_file.setFont(QFont(self.font_id, 10 + self.font_size_increase))
        self.conversion_target_file.setStyleSheet('QLineEdit { width: 108px; }')
        self.conversion_target_file.move(column_two_x2, 100)

        constant_rate_factor_label = QLabel('FFMPEG crf (0–51):', self.ProcessSettings)
        constant_rate_factor_label.setFont(QFont(self.font_id, 12 + self.font_size_increase))
        constant_rate_factor_label.move(column_two_x1, 130)
        self.constant_rate_factor = QLineEdit(f"{self.processing_input_dict['modify_files']['Operator']['rectify_video_fps']['constant_rate_factor']}", self.ProcessSettings)
        self.constant_rate_factor.setFont(QFont(self.font_id, 10 + self.font_size_increase))
        self.constant_rate_factor.setStyleSheet('QLineEdit { width: 108px; }')
        self.constant_rate_factor.move(column_two_x2, 130)

        encoding_preset_label = QLabel('FFMPEG preset:', self.ProcessSettings)
        encoding_preset_label.setFont(QFont(self.font_id, 12 + self.font_size_increase))
        encoding_preset_label.move(column_two_x1, 160)
        self.encoding_preset_list = sorted(['veryfast', 'faster', 'fast', 'medium', 'slow', 'slower', 'veryslow'], key=lambda x: x == self.encoding_preset, reverse=True)
        self.encoding_preset_cb = QComboBox(self.ProcessSettings)
        self.encoding_preset_cb.addItems([str(encode_preset_item) for encode_preset_item in self.encoding_preset_list])
        self.encoding_preset_cb.setStyleSheet('QComboBox { width: 80px; }')
        self.encoding_preset_cb.activated.connect(partial(self._combo_box_encoding_preset, variable_id='encoding_preset'))
        self.encoding_preset_cb.move(column_two_x2, 160)

        delete_con_file_cb_label = QLabel('Purge concatenated files:', self.ProcessSettings)
        delete_con_file_cb_label.setFont(QFont(self.font_id, 12 + self.font_size_increase))
        delete_con_file_cb_label.move(column_two_x1, 190)
        self.delete_con_file_cb = QComboBox(self.ProcessSettings)
        self.delete_con_file_cb.addItems(['Yes', 'No'])
        self.delete_con_file_cb.setStyleSheet('QComboBox { width: 80px; }')
        self.delete_con_file_cb.activated.connect(partial(self._combo_box_prior_true, variable_id='delete_con_file_cb_bool'))
        self.delete_con_file_cb.move(column_two_x2, 190)

        sleap_cluster_cb_label = QLabel('Prepare SLEAP cluster job:', self.ProcessSettings)
        sleap_cluster_cb_label.setFont(QFont(self.font_id, 11 + self.font_size_increase))
        sleap_cluster_cb_label.setStyleSheet('QLabel { color: #F58025; font-weight: bold;}')
        sleap_cluster_cb_label.move(column_two_x1, 220)
        self.sleap_cluster_cb = QComboBox(self.ProcessSettings)
        self.sleap_cluster_cb.addItems(['No', 'Yes'])
        self.sleap_cluster_cb.setStyleSheet('QComboBox { width: 80px; }')
        self.sleap_cluster_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='sleap_cluster_cb_bool'))
        self.sleap_cluster_cb.move(column_two_x2, 220)

        sleap_file_conversion_cb_label = QLabel('Run SLP-H5 conversion:', self.ProcessSettings)
        sleap_file_conversion_cb_label.setFont(QFont(self.font_id, 11 + self.font_size_increase))
        sleap_file_conversion_cb_label.setStyleSheet('QLabel { color: #F58025; font-weight: bold;}')
        sleap_file_conversion_cb_label.move(column_two_x1, 250)
        self.sleap_file_conversion_cb = QComboBox(self.ProcessSettings)
        self.sleap_file_conversion_cb.addItems(['No', 'Yes'])
        self.sleap_file_conversion_cb.setStyleSheet('QComboBox { width: 80px; }')
        self.sleap_file_conversion_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='sleap_file_conversion_cb_bool'))
        self.sleap_file_conversion_cb.move(column_two_x2, 250)

        anipose_calibration_cb_label = QLabel('Run AP calibration:', self.ProcessSettings)
        anipose_calibration_cb_label.setFont(QFont(self.font_id, 11 + self.font_size_increase))
        anipose_calibration_cb_label.setStyleSheet('QLabel { color: #F58025; font-weight: bold;}')
        anipose_calibration_cb_label.move(column_two_x1, 280)
        self.anipose_calibration_cb = QComboBox(self.ProcessSettings)
        self.anipose_calibration_cb.addItems(['No', 'Yes'])
        self.anipose_calibration_cb.setStyleSheet('QComboBox { width: 80px; }')
        self.anipose_calibration_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='anipose_calibration_cb_bool'))
        self.anipose_calibration_cb.move(column_two_x2, 280)

        board_provided_cb_label = QLabel('ChArUco board provided:', self.ProcessSettings)
        board_provided_cb_label.setFont(QFont(self.font_id, 12 + self.font_size_increase))
        board_provided_cb_label.move(column_two_x1, 310)
        self.board_provided_cb = QComboBox(self.ProcessSettings)
        self.board_provided_cb.addItems(['No', 'Yes'])
        self.board_provided_cb.setStyleSheet('QComboBox { width: 80px; }')
        self.board_provided_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='board_provided_cb_bool'))
        self.board_provided_cb.move(column_two_x2, 310)

        anipose_triangulation_cb_label = QLabel('Run AP triangulation:', self.ProcessSettings)
        anipose_triangulation_cb_label.setFont(QFont(self.font_id, 11 + self.font_size_increase))
        anipose_triangulation_cb_label.setStyleSheet('QLabel { color: #F58025; font-weight: bold;}')
        anipose_triangulation_cb_label.move(column_two_x1, 340)
        self.anipose_triangulation_cb = QComboBox(self.ProcessSettings)
        self.anipose_triangulation_cb.addItems(['No', 'Yes'])
        self.anipose_triangulation_cb.setStyleSheet('QComboBox { width: 80px; }')
        self.anipose_triangulation_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='anipose_triangulation_cb_bool'))
        self.anipose_triangulation_cb.move(column_two_x2, 340)

        triangulate_arena_points_cb_label = QLabel('Triangulate arena nodes:', self.ProcessSettings)
        triangulate_arena_points_cb_label.setFont(QFont(self.font_id, 12 + self.font_size_increase))
        triangulate_arena_points_cb_label.move(column_two_x1, 370)
        self.triangulate_arena_points_cb = QComboBox(self.ProcessSettings)
        self.triangulate_arena_points_cb.addItems(['No', 'Yes'])
        self.triangulate_arena_points_cb.setStyleSheet('QComboBox { width: 80px; }')
        self.triangulate_arena_points_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='triangulate_arena_points_cb_bool'))
        self.triangulate_arena_points_cb.move(column_two_x2, 370)

        display_progress_cb_label = QLabel('Display progress:', self.ProcessSettings)
        display_progress_cb_label.setFont(QFont(self.font_id, 12 + self.font_size_increase))
        display_progress_cb_label.move(column_two_x1, 400)
        self.display_progress_cb = QComboBox(self.ProcessSettings)
        self.display_progress_cb.addItems(['Yes', 'No'])
        self.display_progress_cb.setStyleSheet('QComboBox { width: 80px; }')
        self.display_progress_cb.activated.connect(partial(self._combo_box_prior_true, variable_id='display_progress_cb_bool'))
        self.display_progress_cb.move(column_two_x2, 400)

        frame_restriction_label = QLabel('Frame restriction:', self.ProcessSettings)
        frame_restriction_label.setFont(QFont(self.font_id, 12 + self.font_size_increase))
        frame_restriction_label.move(column_two_x1, 430)
        frame_restriction_input = '' if self.processing_input_dict['anipose_operations']['ConvertTo3D']['conduct_anipose_triangulation']['frame_restriction'] is None else ','.join([str(x) for x in self.processing_input_dict['anipose_operations']['ConvertTo3D']['conduct_anipose_triangulation']['frame_restriction']])
        self.frame_restriction = QLineEdit(frame_restriction_input, self.ProcessSettings)
        self.frame_restriction.setFont(QFont(self.font_id, 10 + self.font_size_increase))
        self.frame_restriction.setStyleSheet('QLineEdit { width: 108px; }')
        self.frame_restriction.move(column_two_x2, 430)

        excluded_views_label = QLabel('Excluded camera views:', self.ProcessSettings)
        excluded_views_label.setFont(QFont(self.font_id, 12 + self.font_size_increase))
        excluded_views_label.move(column_two_x1, 460)
        self.excluded_views = QLineEdit(','.join([str(x) for x in self.processing_input_dict['anipose_operations']['ConvertTo3D']['conduct_anipose_triangulation']['excluded_views']]), self.ProcessSettings)
        self.excluded_views.setFont(QFont(self.font_id, 10 + self.font_size_increase))
        self.excluded_views.setStyleSheet('QLineEdit { width: 108px; }')
        self.excluded_views.move(column_two_x2, 460)

        ransac_cb_label = QLabel('Ransac:', self.ProcessSettings)
        ransac_cb_label.setFont(QFont(self.font_id, 12 + self.font_size_increase))
        ransac_cb_label.move(column_two_x1, 490)
        self.ransac_cb = QComboBox(self.ProcessSettings)
        self.ransac_cb.addItems(['No', 'Yes'])
        self.ransac_cb.setStyleSheet('QComboBox { width: 80px; }')
        self.ransac_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='ransac_cb_bool'))
        self.ransac_cb.move(column_two_x2, 490)

        rigid_body_constraints_label = QLabel('Rigid body constraints:', self.ProcessSettings)
        rigid_body_constraints_label.setFont(QFont(self.font_id, 12 + self.font_size_increase))
        rigid_body_constraints_label.move(column_two_x1, 520)
        self.rigid_body_constraints = QLineEdit('', self.ProcessSettings)
        self.rigid_body_constraints.setFont(QFont(self.font_id, 10 + self.font_size_increase))
        self.rigid_body_constraints.setStyleSheet('QLineEdit { width: 108px; }')
        self.rigid_body_constraints.move(column_two_x2, 520)

        weak_body_constraints_label = QLabel('Weak body constraints:', self.ProcessSettings)
        weak_body_constraints_label.setFont(QFont(self.font_id, 12 + self.font_size_increase))
        weak_body_constraints_label.move(column_two_x1, 550)
        self.weak_body_constraints = QLineEdit('', self.ProcessSettings)
        self.weak_body_constraints.setFont(QFont(self.font_id, 10 + self.font_size_increase))
        self.weak_body_constraints.setStyleSheet('QLineEdit { width: 108px; }')
        self.weak_body_constraints.move(column_two_x2, 550)

        smooth_scale_label = QLabel('Smoothing scale:', self.ProcessSettings)
        smooth_scale_label.setFont(QFont(self.font_id, 12 + self.font_size_increase))
        smooth_scale_label.move(column_two_x1, 580)
        self.smooth_scale = QLineEdit(f"{self.processing_input_dict['anipose_operations']['ConvertTo3D']['conduct_anipose_triangulation']['smooth_scale']}", self.ProcessSettings)
        self.smooth_scale.setFont(QFont(self.font_id, 10 + self.font_size_increase))
        self.smooth_scale.setStyleSheet('QLineEdit { width: 108px; }')
        self.smooth_scale.move(column_two_x2, 580)

        weight_rigid_label = QLabel('Rigid constraints weight:', self.ProcessSettings)
        weight_rigid_label.setFont(QFont(self.font_id, 12 + self.font_size_increase))
        weight_rigid_label.move(column_two_x1, 610)
        self.weight_rigid = QLineEdit(f"{self.processing_input_dict['anipose_operations']['ConvertTo3D']['conduct_anipose_triangulation']['weight_rigid']}", self.ProcessSettings)
        self.weight_rigid.setFont(QFont(self.font_id, 10 + self.font_size_increase))
        self.weight_rigid.setStyleSheet('QLineEdit { width: 108px; }')
        self.weight_rigid.move(column_two_x2, 610)

        weight_weak_label = QLabel('Weak constraints weight:', self.ProcessSettings)
        weight_weak_label.setFont(QFont(self.font_id, 12 + self.font_size_increase))
        weight_weak_label.move(column_two_x1, 640)
        self.weight_weak = QLineEdit(f"{self.processing_input_dict['anipose_operations']['ConvertTo3D']['conduct_anipose_triangulation']['weight_weak']}", self.ProcessSettings)
        self.weight_weak.setFont(QFont(self.font_id, 10 + self.font_size_increase))
        self.weight_weak.setStyleSheet('QLineEdit { width: 108px; }')
        self.weight_weak.move(column_two_x2, 640)

        reprojection_error_threshold_label = QLabel('Reproject error threshold:', self.ProcessSettings)
        reprojection_error_threshold_label.setFont(QFont(self.font_id, 12 + self.font_size_increase))
        reprojection_error_threshold_label.move(column_two_x1, 670)
        self.reprojection_error_threshold = QLineEdit(f"{self.processing_input_dict['anipose_operations']['ConvertTo3D']['conduct_anipose_triangulation']['reprojection_error_threshold']}", self.ProcessSettings)
        self.reprojection_error_threshold.setFont(QFont(self.font_id, 10 + self.font_size_increase))
        self.reprojection_error_threshold.setStyleSheet('QLineEdit { width: 108px; }')
        self.reprojection_error_threshold.move(column_two_x2, 670)

        regularization_function_label = QLabel('Regularization:', self.ProcessSettings)
        regularization_function_label.setFont(QFont(self.font_id, 12 + self.font_size_increase))
        regularization_function_label.move(column_two_x1, 700)
        self.regularization_function = QLineEdit(f"{self.processing_input_dict['anipose_operations']['ConvertTo3D']['conduct_anipose_triangulation']['regularization_function']}", self.ProcessSettings)
        self.regularization_function.setFont(QFont(self.font_id, 10 + self.font_size_increase))
        self.regularization_function.setStyleSheet('QLineEdit { width: 108px; }')
        self.regularization_function.move(column_two_x2, 700)

        n_deriv_smooth_label = QLabel('Derivation kernel order:', self.ProcessSettings)
        n_deriv_smooth_label.setFont(QFont(self.font_id, 12 + self.font_size_increase))
        n_deriv_smooth_label.move(column_two_x1, 730)
        self.n_deriv_smooth = QLineEdit(f"{self.processing_input_dict['anipose_operations']['ConvertTo3D']['conduct_anipose_triangulation']['n_deriv_smooth']}", self.ProcessSettings)
        self.n_deriv_smooth.setFont(QFont(self.font_id, 10 + self.font_size_increase))
        self.n_deriv_smooth.setStyleSheet('QLineEdit { width: 108px; }')
        self.n_deriv_smooth.move(column_two_x2, 730)

        translate_rotate_metric_label = QLabel('Re-coordinate (ExCode!):', self.ProcessSettings)
        translate_rotate_metric_label.setFont(QFont(self.font_id, 11 + self.font_size_increase))
        translate_rotate_metric_label.setStyleSheet('QLabel { color: #F58025; font-weight: bold;}')
        translate_rotate_metric_label.move(column_two_x1, 760)
        self.translate_rotate_metric_cb = QComboBox(self.ProcessSettings)
        self.translate_rotate_metric_cb.addItems(['No', 'Yes'])
        self.translate_rotate_metric_cb.setStyleSheet('QComboBox { width: 80px; }')
        self.translate_rotate_metric_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='translate_rotate_metric_cb_bool'))
        self.translate_rotate_metric_cb.move(column_two_x2, 760)

        static_reference_len_label = QLabel('Static reference (m):', self.ProcessSettings)
        static_reference_len_label.setFont(QFont(self.font_id, 12 + self.font_size_increase))
        static_reference_len_label.move(column_two_x1, 790)
        self.static_reference_len = QLineEdit(f"{self.processing_input_dict['anipose_operations']['ConvertTo3D']['translate_rotate_metric']['static_reference_len']}", self.ProcessSettings)
        self.static_reference_len.setFont(QFont(self.font_id, 10 + self.font_size_increase))
        self.static_reference_len.setStyleSheet('QLineEdit { width: 108px; }')
        self.static_reference_len.move(column_two_x2, 790)

        save_transformed_data_cb_label = QLabel('Save transformation type:', self.ProcessSettings)
        save_transformed_data_cb_label.setFont(QFont(self.font_id, 12 + self.font_size_increase))
        save_transformed_data_cb_label.move(column_two_x1, 820)
        self.save_transformed_data_cb = QComboBox(self.ProcessSettings)
        self.save_transformed_data_cb.addItems(['animal', 'arena'])
        self.save_transformed_data_cb.setStyleSheet('QComboBox { width: 80px; }')
        self.save_transformed_data_cb.activated.connect(partial(self._combo_box_prior_transformed_tracking_data, variable_id='save_transformed_data'))
        self.save_transformed_data_cb.move(column_two_x2, 820)

        delete_original_h5_cb_label = QLabel('Delete original .h5:', self.ProcessSettings)
        delete_original_h5_cb_label.setFont(QFont(self.font_id, 12 + self.font_size_increase))
        delete_original_h5_cb_label.move(column_two_x1, 850)
        self.delete_original_h5_cb = QComboBox(self.ProcessSettings)
        self.delete_original_h5_cb.addItems(['Yes', 'No'])
        self.delete_original_h5_cb.setStyleSheet('QComboBox { width: 80px; }')
        self.delete_original_h5_cb.activated.connect(partial(self._combo_box_prior_true, variable_id='delete_original_h5_cb_bool'))
        self.delete_original_h5_cb.move(column_two_x2, 850)

        # column 3
        column_three_x1 = 760
        column_three_x2 = 960

        gas_label = QLabel('AUDIO processing settings', self.ProcessSettings)
        gas_label.setFont(QFont(self.font_id, 13+self.font_size_increase))
        gas_label.setStyleSheet('QLabel { font-weight: bold;}')
        gas_label.move(column_three_x1, 10)

        conduct_multichannel_conversion_cb_label = QLabel('Convert to single-ch files:', self.ProcessSettings)
        conduct_multichannel_conversion_cb_label.setFont(QFont(self.font_id, 11+self.font_size_increase))
        conduct_multichannel_conversion_cb_label.setStyleSheet('QLabel { color: #F58025; font-weight: bold;}')
        conduct_multichannel_conversion_cb_label.move(column_three_x1, 40)
        self.conduct_multichannel_conversion_cb = QComboBox(self.ProcessSettings)
        self.conduct_multichannel_conversion_cb.addItems(['No', 'Yes'])
        self.conduct_multichannel_conversion_cb.setStyleSheet('QComboBox { width: 80px; }')
        self.conduct_multichannel_conversion_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='conduct_multichannel_conversion_cb_bool'))
        self.conduct_multichannel_conversion_cb.move(column_three_x2, 40)

        crop_wav_cam_cb_label = QLabel('Crop AUDIO (to VIDEO):', self.ProcessSettings)
        crop_wav_cam_cb_label.setFont(QFont(self.font_id, 11+self.font_size_increase))
        crop_wav_cam_cb_label.setStyleSheet('QLabel { color: #F58025; font-weight: bold;}')
        crop_wav_cam_cb_label.move(column_three_x1, 70)
        self.crop_wav_cam_cb = QComboBox(self.ProcessSettings)
        self.crop_wav_cam_cb.addItems(['No', 'Yes'])
        self.crop_wav_cam_cb.setStyleSheet('QComboBox { width: 80px; }')
        self.crop_wav_cam_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='crop_wav_cam_cb_bool'))
        self.crop_wav_cam_cb.move(column_three_x2, 70)

        device_receiving_input_cb_label = QLabel('Trgbox-USGH device(s):', self.ProcessSettings)
        device_receiving_input_cb_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        device_receiving_input_cb_label.move(column_three_x1, 100)
        self.usgh_device_receiving_input_list = sorted(['m', 's', 'both'], key=lambda x: x == self.processing_input_dict['synchronize_files']['Synchronizer']['crop_wav_files_to_video']['device_receiving_input'], reverse=True)
        self.device_receiving_input_cb = QComboBox(self.ProcessSettings)
        self.device_receiving_input_cb.addItems(self.usgh_device_receiving_input_list)
        self.device_receiving_input_cb.setStyleSheet('QComboBox { width: 80px; }')
        self.device_receiving_input_cb.activated.connect(partial(self._combo_box_prior_audio_device_camera_input, variable_id='device_receiving_input'))
        self.device_receiving_input_cb.move(column_three_x2, 100)

        ch_receiving_input_label = QLabel('Trgbox-USGH ch (1-12):', self.ProcessSettings)
        ch_receiving_input_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        ch_receiving_input_label.move(column_three_x1, 130)
        self.ch_receiving_input = QLineEdit(f"{self.processing_input_dict['synchronize_files']['Synchronizer']['crop_wav_files_to_video']['triggerbox_ch_receiving_input']}", self.ProcessSettings)
        self.ch_receiving_input.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.ch_receiving_input.setStyleSheet('QLineEdit { width: 108px; }')
        self.ch_receiving_input.move(column_three_x2, 130)

        conduct_hpss_cb_label = QLabel('Run HPSS (slow!):', self.ProcessSettings)
        conduct_hpss_cb_label.setFont(QFont(self.font_id, 11+self.font_size_increase))
        conduct_hpss_cb_label.setStyleSheet('QLabel { color: #F58025; font-weight: bold;}')
        conduct_hpss_cb_label.move(column_three_x1, 160)
        self.conduct_hpss_cb = QComboBox(self.ProcessSettings)
        self.conduct_hpss_cb.addItems(['No', 'Yes'])
        self.conduct_hpss_cb.setStyleSheet('QComboBox { width: 80px; }')
        self.conduct_hpss_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='conduct_hpss_cb_bool'))
        self.conduct_hpss_cb.move(column_three_x2, 160)

        stft_label = QLabel('STFT window & hop size:', self.ProcessSettings)
        stft_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        stft_label.move(column_three_x1, 190)
        self.stft_window_hop = QLineEdit(','.join([str(x) for x in self.processing_input_dict['modify_files']['Operator']['hpss_audio']['stft_window_length_hop_size']]), self.ProcessSettings)
        self.stft_window_hop.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.stft_window_hop.setStyleSheet('QLineEdit { width: 108px; }')
        self.stft_window_hop.move(column_three_x2, 190)

        hpss_kernel_size_label = QLabel('HPSS kernel size:', self.ProcessSettings)
        hpss_kernel_size_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        hpss_kernel_size_label.move(column_three_x1, 220)
        self.hpss_kernel_size = QLineEdit(','.join([str(x) for x in self.processing_input_dict['modify_files']['Operator']['hpss_audio']['kernel_size']]), self.ProcessSettings)
        self.hpss_kernel_size.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.hpss_kernel_size.setStyleSheet('QLineEdit { width: 108px; }')
        self.hpss_kernel_size.move(column_three_x2, 220)

        hpss_power_label = QLabel('HPSS power:', self.ProcessSettings)
        hpss_power_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        hpss_power_label.move(column_three_x1, 250)
        self.hpss_power = QLineEdit(f"{self.processing_input_dict['modify_files']['Operator']['hpss_audio']['hpss_power']}", self.ProcessSettings)
        self.hpss_power.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.hpss_power.setStyleSheet('QLineEdit { width: 108px; }')
        self.hpss_power.move(column_three_x2, 250)

        hpss_margin_label = QLabel('HPSS margin:', self.ProcessSettings)
        hpss_margin_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        hpss_margin_label.move(column_three_x1, 280)
        self.hpss_margin = QLineEdit(','.join([str(x) for x in self.processing_input_dict['modify_files']['Operator']['hpss_audio']['margin']]), self.ProcessSettings)
        self.hpss_margin.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.hpss_margin.setStyleSheet('QLineEdit { width: 108px; }')
        self.hpss_margin.move(column_three_x2, 280)

        filter_audio_cb_label = QLabel('Filter audio files:', self.ProcessSettings)
        filter_audio_cb_label.setFont(QFont(self.font_id, 11+self.font_size_increase))
        filter_audio_cb_label.setStyleSheet('QLabel { color: #F58025; font-weight: bold;}')
        filter_audio_cb_label.move(column_three_x1, 310)
        self.filter_audio_cb = QComboBox(self.ProcessSettings)
        self.filter_audio_cb.addItems(['No', 'Yes'])
        self.filter_audio_cb.setStyleSheet('QComboBox { width: 80px; }')
        self.filter_audio_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='filter_audio_cb_bool'))
        self.filter_audio_cb.move(column_three_x2, 310)

        filter_freq_bounds_label = QLabel('Filter freq bounds (Hz):', self.ProcessSettings)
        filter_freq_bounds_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        filter_freq_bounds_label.move(column_three_x1, 340)
        self.filter_freq_bounds = QLineEdit(','.join([str(x) for x in self.processing_input_dict['modify_files']['Operator']['filter_audio_files']['filter_freq_bounds']]), self.ProcessSettings)
        self.filter_freq_bounds.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.filter_freq_bounds.setStyleSheet('QLineEdit { width: 108px; }')
        self.filter_freq_bounds.move(column_three_x2, 340)

        filter_dirs_label = QLabel('Folder(s) to filter:', self.ProcessSettings)
        filter_dirs_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        filter_dirs_label.move(column_three_x1, 370)
        self.filter_dirs = QLineEdit(','.join([str(x) for x in self.processing_input_dict['modify_files']['Operator']['filter_audio_files']['filter_dirs']]), self.ProcessSettings)
        self.filter_dirs.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.filter_dirs.setStyleSheet('QLineEdit { width: 108px; }')
        self.filter_dirs.move(column_three_x2, 370)

        conc_audio_cb_label = QLabel('Concatenate to MEMMAP:', self.ProcessSettings)
        conc_audio_cb_label.setFont(QFont(self.font_id, 11+self.font_size_increase))
        conc_audio_cb_label.setStyleSheet('QLabel { color: #F58025; font-weight: bold;}')
        conc_audio_cb_label.move(column_three_x1, 400)
        self.conc_audio_cb = QComboBox(self.ProcessSettings)
        self.conc_audio_cb.addItems(['No', 'Yes'])
        self.conc_audio_cb.setStyleSheet('QComboBox { width: 80px; }')
        self.conc_audio_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='conc_audio_cb_bool'))
        self.conc_audio_cb.move(column_three_x2, 400)

        concat_dirs_label = QLabel('Folder(s) to concatenate:', self.ProcessSettings)
        concat_dirs_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        concat_dirs_label.move(column_three_x1, 430)
        self.concat_dirs = QLineEdit(','.join([str(x) for x in self.processing_input_dict['modify_files']['Operator']['concatenate_audio_files']['concat_dirs']]), self.ProcessSettings)
        self.concat_dirs.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.concat_dirs.setStyleSheet('QLineEdit { width: 108px; }')
        self.concat_dirs.move(column_three_x2, 430)

        das_inference_cb_label = QLabel('Run DAS inference:', self.ProcessSettings)
        das_inference_cb_label.setFont(QFont(self.font_id, 11+self.font_size_increase))
        das_inference_cb_label.setStyleSheet('QLabel { color: #F58025; font-weight: bold;}')
        das_inference_cb_label.move(column_three_x1, 460)
        self.das_inference_cb = QComboBox(self.ProcessSettings)
        self.das_inference_cb.addItems(['No', 'Yes'])
        self.das_inference_cb.setStyleSheet('QComboBox { width: 80px; }')
        self.das_inference_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='das_inference_cb_bool'))
        self.das_inference_cb.move(column_three_x2, 460)

        segment_confidence_threshold_label = QLabel('DAS confidence threshold:', self.ProcessSettings)
        segment_confidence_threshold_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        segment_confidence_threshold_label.move(column_three_x1, 490)
        self.segment_confidence_threshold = QLineEdit(f"{self.processing_input_dict['usv_inference']['FindMouseVocalizations']['das_command_line_inference']['segment_confidence_threshold']}", self.ProcessSettings)
        self.segment_confidence_threshold.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.segment_confidence_threshold.setStyleSheet('QLineEdit { width: 108px; }')
        self.segment_confidence_threshold.move(column_three_x2, 490)

        segment_minlen_label = QLabel('USV min duration (s):', self.ProcessSettings)
        segment_minlen_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        segment_minlen_label.move(column_three_x1, 520)
        self.segment_minlen = QLineEdit(f"{self.processing_input_dict['usv_inference']['FindMouseVocalizations']['das_command_line_inference']['segment_minlen']}", self.ProcessSettings)
        self.segment_minlen.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.segment_minlen.setStyleSheet('QLineEdit { width: 108px; }')
        self.segment_minlen.move(column_three_x2, 520)

        segment_fillgap_label = QLabel('Fill gaps shorter than (s):', self.ProcessSettings)
        segment_fillgap_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        segment_fillgap_label.move(column_three_x1, 550)
        self.segment_fillgap = QLineEdit(f"{self.processing_input_dict['usv_inference']['FindMouseVocalizations']['das_command_line_inference']['segment_fillgap']}", self.ProcessSettings)
        self.segment_fillgap.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.segment_fillgap.setStyleSheet('QLineEdit { width: 108px; }')
        self.segment_fillgap.move(column_three_x2, 550)

        das_output_type_label = QLabel('Inference output file type:', self.ProcessSettings)
        das_output_type_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        das_output_type_label.move(column_three_x1, 580)
        self.das_output_type = QLineEdit(f"{self.processing_input_dict['usv_inference']['FindMouseVocalizations']['das_command_line_inference']['output_file_type']}", self.ProcessSettings)
        self.das_output_type.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.das_output_type.setStyleSheet('QLineEdit { width: 108px; }')
        self.das_output_type.move(column_three_x2, 580)

        das_summary_cb_label = QLabel('Curate DAS outputs:', self.ProcessSettings)
        das_summary_cb_label.setFont(QFont(self.font_id, 11+self.font_size_increase))
        das_summary_cb_label.setStyleSheet('QLabel { color: #F58025; font-weight: bold;}')
        das_summary_cb_label.move(column_three_x1, 610)
        self.das_summary_cb = QComboBox(self.ProcessSettings)
        self.das_summary_cb.addItems(['No', 'Yes'])
        self.das_summary_cb.setStyleSheet('QComboBox { width: 80px; }')
        self.das_summary_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='das_summary_cb_bool'))
        self.das_summary_cb.move(column_three_x2, 610)

        prepare_assign_usv_cb_label = QLabel('Prepare USV assignment:', self.ProcessSettings)
        prepare_assign_usv_cb_label.setFont(QFont(self.font_id, 11 + self.font_size_increase))
        prepare_assign_usv_cb_label.setStyleSheet('QLabel { color: #F58025; font-weight: bold;}')
        prepare_assign_usv_cb_label.move(column_three_x1, 640)
        self.prepare_assign_usv_cb = QComboBox(self.ProcessSettings)
        self.prepare_assign_usv_cb.addItems(['No', 'Yes'])
        self.prepare_assign_usv_cb.setStyleSheet('QComboBox { width: 80px; }')
        self.prepare_assign_usv_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='prepare_assign_usv_cb_bool'))
        self.prepare_assign_usv_cb.move(column_three_x2, 640)

        assign_usv_cb_label = QLabel('Run USV assignment:', self.ProcessSettings)
        assign_usv_cb_label.setFont(QFont(self.font_id, 11 + self.font_size_increase))
        assign_usv_cb_label.setStyleSheet('QLabel { color: #F58025; font-weight: bold;}')
        assign_usv_cb_label.move(column_three_x1, 670)
        self.assign_usv_cb = QComboBox(self.ProcessSettings)
        self.assign_usv_cb.addItems(['No', 'Yes'])
        self.assign_usv_cb.setStyleSheet('QComboBox { width: 80px; }')
        self.assign_usv_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='assign_usv_cb_bool'))
        self.assign_usv_cb.move(column_three_x2, 670)

        assign_type_cb_label = QLabel('Assignment type:', self.ProcessSettings)
        assign_type_cb_label.setFont(QFont(self.font_id, 11 + self.font_size_increase))
        assign_type_cb_label.move(column_three_x1, 700)
        self.assign_type_list = sorted(['vcl', 'vcl-ssl'], key=lambda x: x == self.vcl_version, reverse=True)
        self.assign_type_cb = QComboBox(self.ProcessSettings)
        self.assign_type_cb.addItems([str(assign_item) for assign_item in self.assign_type_list])
        self.assign_type_cb.setStyleSheet('QComboBox { width: 80px; }')
        self.assign_type_cb.activated.connect(partial(self._combo_box_vcl_version, variable_id='vcl_version'))
        self.assign_type_cb.move(column_three_x2, 700)

        av_sync_label = QLabel('Synchronization between A/V files', self.ProcessSettings)
        av_sync_label.setFont(QFont(self.font_id, 13 + self.font_size_increase))
        av_sync_label.setStyleSheet('QLabel { font-weight: bold;}')
        av_sync_label.move(column_three_x1, 740)

        conduct_sync_cb_label = QLabel('Run A/V sync check:', self.ProcessSettings)
        conduct_sync_cb_label.setFont(QFont(self.font_id, 11 + self.font_size_increase))
        conduct_sync_cb_label.setStyleSheet('QLabel { color: #F58025; font-weight: bold;}')
        conduct_sync_cb_label.move(column_three_x1, 770)
        self.conduct_sync_cb = QComboBox(self.ProcessSettings)
        self.conduct_sync_cb.addItems(['No', 'Yes'])
        self.conduct_sync_cb.setStyleSheet('QComboBox { width: 80px; }')
        self.conduct_sync_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='conduct_sync_cb_bool'))
        self.conduct_sync_cb.move(column_three_x2, 770)

        phidget_extra_data_camera_label = QLabel('Phidget(s) camera serial:', self.ProcessSettings)
        phidget_extra_data_camera_label.setFont(QFont(self.font_id, 12 + self.font_size_increase))
        phidget_extra_data_camera_label.move(column_three_x1, 800)
        self.phidget_extra_data_camera = QLineEdit(f"{self.processing_input_dict['extract_phidget_data']['Gatherer']['prepare_data_for_analyses']['extra_data_camera']}", self.ProcessSettings)
        self.phidget_extra_data_camera.setFont(QFont(self.font_id, 10 + self.font_size_increase))
        self.phidget_extra_data_camera.setStyleSheet('QLineEdit { width: 108px; }')
        self.phidget_extra_data_camera.move(column_three_x2, 800)

        a_ch_receiving_input_label = QLabel('Arduino-USGH ch (1-12):', self.ProcessSettings)
        a_ch_receiving_input_label.setFont(QFont(self.font_id, 12 + self.font_size_increase))
        a_ch_receiving_input_label.move(column_three_x1, 830)
        self.a_ch_receiving_input = QLineEdit(f"{self.processing_input_dict['synchronize_files']['Synchronizer']['find_audio_sync_trains']['sync_ch_receiving_input']}", self.ProcessSettings)
        self.a_ch_receiving_input.setFont(QFont(self.font_id, 10 + self.font_size_increase))
        self.a_ch_receiving_input.setStyleSheet('QLineEdit { width: 108px; }')
        self.a_ch_receiving_input.move(column_three_x2, 830)

        v_camera_serial_num_label = QLabel('Sync camera serial num(s):', self.ProcessSettings)
        v_camera_serial_num_label.setFont(QFont(self.font_id, 12 + self.font_size_increase))
        v_camera_serial_num_label.move(column_three_x1, 860)
        self.v_camera_serial_num = QLineEdit(','.join([str(x) for x in self.processing_input_dict['synchronize_files']['Synchronizer']['find_video_sync_trains']['sync_camera_serial_num']]), self.ProcessSettings)
        self.v_camera_serial_num.setFont(QFont(self.font_id, 10 + self.font_size_increase))
        self.v_camera_serial_num.setStyleSheet('QLineEdit { width: 108px; }')
        self.v_camera_serial_num.move(column_three_x2, 860)

        self._create_buttons_process(seq=0, class_option=self.ProcessSettings,
                                     button_pos_y=process_one_y - 35, next_button_x_pos=process_one_x - 100)

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
        process_two_x, process_two_y = (870, 1000)
        self.setFixedSize(process_two_x, process_two_y)

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
                                     button_pos_y=process_two_y - 35, next_button_x_pos=process_two_x - 100)

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
        analyze_one_x, analyze_one_y = (770, 615)
        self.setFixedSize(analyze_one_x, analyze_one_y)

        analyses_dir_label = QLabel('(*) Root directories for analyses', self.AnalysesSettings)
        analyses_dir_label.setFont(QFont(self.font_id, 13+self.font_size_increase))
        analyses_dir_label.setStyleSheet('QLabel { font-weight: bold;}')
        analyses_dir_label.move(50, 10)
        self.analyses_dir_edit = QTextEdit('', self.AnalysesSettings)
        self.analyses_dir_edit.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.analyses_dir_edit.move(10, 40)
        self.analyses_dir_edit.setFixedSize(350, 320)

        self.analyses_credentials_dir_edit = QLineEdit(f"{self.analyses_input_dict['credentials_directory']}", self.AnalysesSettings)
        self.analyses_credentials_dir_edit.setPlaceholderText('Credentials directory')
        update_analyses_credentials_dir = partial(self._update_nested_dict_value, self.analyses_input_dict, ('credentials_directory',))
        self.analyses_credentials_dir_edit.textChanged.connect(update_analyses_credentials_dir)
        self.analyses_credentials_dir_edit.setFont(QFont(self.font_id, 10 + self.font_size_increase))
        self.analyses_credentials_dir_edit.setStyleSheet('QLineEdit { width: 285px; }')
        self.analyses_credentials_dir_edit.move(10, 365)
        analyses_credentials_dir_btn = QPushButton('Browse', self.AnalysesSettings)
        analyses_credentials_dir_btn.setFont(QFont(self.font_id, 8 + self.font_size_increase))
        analyses_credentials_dir_btn.move(300, 365)
        analyses_credentials_dir_btn.setStyleSheet('QPushButton { min-width: 36px; min-height: 12px; max-width: 36px; max-height: 13px; }')
        open_analyses_credentials_dir_dialog = partial(self._open_directory_dialog, self.analyses_credentials_dir_edit, 'Select credentials directory')
        analyses_credentials_dir_btn.clicked.connect(open_analyses_credentials_dir_dialog)

        pc_usage_analyses_label = QLabel('Notify e-mail(s) of PC usage:', self.AnalysesSettings)
        pc_usage_analyses_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        pc_usage_analyses_label.move(10, 395)
        self.pc_usage_analyses = QLineEdit('', self.AnalysesSettings)
        self.pc_usage_analyses.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.pc_usage_analyses.setStyleSheet('QLineEdit { width: 135px; }')
        self.pc_usage_analyses.move(225, 395)

        analyses_pc_label = QLabel('Analyses PC of choice:', self.AnalysesSettings)
        analyses_pc_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        analyses_pc_label.move(10, 425)
        self.loaded_analyses_pc_list = sorted(self.analyses_input_dict['send_email']['analyses_pc_list'], key=lambda x: x == self.analyses_input_dict['send_email']['analyses_pc_choice'], reverse=True)
        self.analyses_pc_cb = QComboBox(self.AnalysesSettings)
        self.analyses_pc_cb.addItems(self.loaded_analyses_pc_list)
        self.analyses_pc_cb.setStyleSheet('QComboBox { width: 107px; }')
        self.analyses_pc_cb.activated.connect(partial(self._combo_box_prior_analyses_pc_choice, variable_id='analyses_pc_choice'))
        self.analyses_pc_cb.move(225, 425)

        da_label = QLabel('Select data analysis', self.AnalysesSettings)
        da_label.setFont(QFont(self.font_id, 13+self.font_size_increase))
        da_label.setStyleSheet('QLabel { font-weight: bold;}')
        da_label.move(10, 465)

        compute_behavioral_features_label = QLabel('Compute 3D behavioral features:', self.AnalysesSettings)
        compute_behavioral_features_label.setFont(QFont(self.font_id, 11+self.font_size_increase))
        compute_behavioral_features_label.setStyleSheet('QLabel { color: #F58025; font-weight: bold;}')
        compute_behavioral_features_label.move(10, 495)
        self.compute_behavioral_features_cb = QComboBox(self.AnalysesSettings)
        self.compute_behavioral_features_cb.addItems(['No', 'Yes'])
        self.compute_behavioral_features_cb.setStyleSheet('QComboBox { width: 57px; }')
        self.compute_behavioral_features_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='compute_behavioral_features_cb_bool'))
        self.compute_behavioral_features_cb.move(275, 495)

        calculate_neuronal_tuning_curves_label = QLabel('Compute 3D feature tuning curves:', self.AnalysesSettings)
        calculate_neuronal_tuning_curves_label.setFont(QFont(self.font_id, 11 + self.font_size_increase))
        calculate_neuronal_tuning_curves_label.setStyleSheet('QLabel { color: #F58025; font-weight: bold;}')
        calculate_neuronal_tuning_curves_label.move(10, 525)
        self.calculate_neuronal_tuning_curves_cb = QComboBox(self.AnalysesSettings)
        self.calculate_neuronal_tuning_curves_cb.addItems(['No', 'Yes'])
        self.calculate_neuronal_tuning_curves_cb.setStyleSheet('QComboBox { width: 57px; }')
        self.calculate_neuronal_tuning_curves_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='calculate_neuronal_tuning_curves_cb_bool'))
        self.calculate_neuronal_tuning_curves_cb.move(275, 525)

        analyses_col_two_x1, analyses_col_two_x2 = 380, 645

        frequency_shift_audio_segment_label = QLabel('Frequency-shift audio segment:', self.AnalysesSettings)
        frequency_shift_audio_segment_label.setFont(QFont(self.font_id, 11 + self.font_size_increase))
        frequency_shift_audio_segment_label.setStyleSheet('QLabel { color: #F58025; font-weight: bold;}')
        frequency_shift_audio_segment_label.move(analyses_col_two_x1, 40)
        self.frequency_shift_audio_segment_cb = QComboBox(self.AnalysesSettings)
        self.frequency_shift_audio_segment_cb.addItems(['No', 'Yes'])
        self.frequency_shift_audio_segment_cb.setStyleSheet('QComboBox { width: 57px; }')
        self.frequency_shift_audio_segment_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='frequency_shift_audio_segment_cb_bool'))
        self.frequency_shift_audio_segment_cb.move(analyses_col_two_x2, 40)

        frequency_shift_audio_dir_label = QLabel('WAV audio subdirectory of choice:', self.AnalysesSettings)
        frequency_shift_audio_dir_label.setFont(QFont(self.font_id, 12 + self.font_size_increase))
        frequency_shift_audio_dir_label.move(analyses_col_two_x1, 70)
        self.frequency_shift_audio_dir_list = sorted(['cropped_to_video', 'hpss', 'hpss_filtered'], key=lambda x: x == self.analyses_input_dict['frequency_shift_audio_segment']['fs_audio_dir'], reverse=True)
        self.frequency_shift_audio_dir_cb = QComboBox(self.AnalysesSettings)
        self.frequency_shift_audio_dir_cb.addItems(self.frequency_shift_audio_dir_list)
        self.frequency_shift_audio_dir_cb.setStyleSheet('QComboBox { width: 57px; }')
        self.frequency_shift_audio_dir_cb.activated.connect(partial(self._combo_box_fs_audio_dir, variable_id='fs_audio_dir'))
        self.frequency_shift_audio_dir_cb.move(analyses_col_two_x2, 70)

        frequency_shift_device_id_label = QLabel('Recording device identity (m|s):', self.AnalysesSettings)
        frequency_shift_device_id_label.setFont(QFont(self.font_id, 12 + self.font_size_increase))
        frequency_shift_device_id_label.move(analyses_col_two_x1, 100)
        self.frequency_shift_device_id_list = sorted(['m', 's'], key=lambda x: x == self.analyses_input_dict['frequency_shift_audio_segment']['fs_device_id'], reverse=True)
        self.frequency_shift_device_id_cb = QComboBox(self.AnalysesSettings)
        self.frequency_shift_device_id_cb.addItems(self.frequency_shift_device_id_list)
        self.frequency_shift_device_id_cb.setStyleSheet('QComboBox { width: 57px; }')
        self.frequency_shift_device_id_cb.activated.connect(partial(self._combo_box_fs_device_id, variable_id='fs_device_id'))
        self.frequency_shift_device_id_cb.move(analyses_col_two_x2, 100)

        frequency_shift_channel_id_label = QLabel('Recording device channel (1-12):', self.AnalysesSettings)
        frequency_shift_channel_id_label.setFont(QFont(self.font_id, 12 + self.font_size_increase))
        frequency_shift_channel_id_label.move(analyses_col_two_x1, 130)
        self.frequency_shift_channel_id_list = sorted(list(range(1, 13)), key=lambda x: x == self.analyses_input_dict['frequency_shift_audio_segment']['fs_channel_id'], reverse=True)
        self.frequency_shift_channel_id_cb = QComboBox(self.AnalysesSettings)
        self.frequency_shift_channel_id_cb.addItems([str(ch_id_item) for ch_id_item in self.frequency_shift_channel_id_list])
        self.frequency_shift_channel_id_cb.setStyleSheet('QComboBox { width: 57px; }')
        self.frequency_shift_channel_id_cb.activated.connect(partial(self._combo_box_fs_channel_id, variable_id='fs_channel_id'))
        self.frequency_shift_channel_id_cb.move(analyses_col_two_x2, 130)

        fs_sequence_start_label = QLabel('Start of audio sequence (s):', self.AnalysesSettings)
        fs_sequence_start_label.setFont(QFont(self.font_id, 12 + self.font_size_increase))
        fs_sequence_start_label.move(analyses_col_two_x1, 160)
        self.fs_sequence_start = QLineEdit(f"{self.analyses_input_dict['frequency_shift_audio_segment']['fs_sequence_start']}", self.AnalysesSettings)
        self.fs_sequence_start.setFont(QFont(self.font_id, 10 + self.font_size_increase))
        self.fs_sequence_start.setStyleSheet('QLineEdit { width: 85px; }')
        self.fs_sequence_start.move(analyses_col_two_x2, 160)

        fs_sequence_duration_label = QLabel('Total duration of audio sequence (s):', self.AnalysesSettings)
        fs_sequence_duration_label.setFont(QFont(self.font_id, 12 + self.font_size_increase))
        fs_sequence_duration_label.move(analyses_col_two_x1, 190)
        self.fs_sequence_duration = QLineEdit(f"{self.analyses_input_dict['frequency_shift_audio_segment']['fs_sequence_duration']}", self.AnalysesSettings)
        self.fs_sequence_duration.setFont(QFont(self.font_id, 10 + self.font_size_increase))
        self.fs_sequence_duration.setStyleSheet('QLineEdit { width: 85px; }')
        self.fs_sequence_duration.move(analyses_col_two_x2, 190)

        fs_octave_shift_label = QLabel('Octave shift (direction and quantity):', self.AnalysesSettings)
        fs_octave_shift_label.setFont(QFont(self.font_id, 12 + self.font_size_increase))
        fs_octave_shift_label.move(analyses_col_two_x1, 220)
        self.fs_octave_shift = QLineEdit(f"{self.analyses_input_dict['frequency_shift_audio_segment']['fs_octave_shift']}", self.AnalysesSettings)
        self.fs_octave_shift.setFont(QFont(self.font_id, 10 + self.font_size_increase))
        self.fs_octave_shift.setStyleSheet('QLineEdit { width: 85px; }')
        self.fs_octave_shift.move(analyses_col_two_x2, 220)

        volume_adjust_audio_segment_label = QLabel('Volume-adjust audio segment:', self.AnalysesSettings)
        volume_adjust_audio_segment_label.setFont(QFont(self.font_id, 11 + self.font_size_increase))
        volume_adjust_audio_segment_label.move(analyses_col_two_x1, 250)
        self.volume_adjust_audio_segment_cb = QComboBox(self.AnalysesSettings)
        self.volume_adjust_audio_segment_cb.addItems(['Yes', 'No'])
        self.volume_adjust_audio_segment_cb.setStyleSheet('QComboBox { width: 57px; }')
        self.volume_adjust_audio_segment_cb.activated.connect(partial(self._combo_box_prior_true, variable_id='volume_adjust_audio_segment_cb_bool'))
        self.volume_adjust_audio_segment_cb.move(analyses_col_two_x2, 250)

        create_usv_playback_wav_label = QLabel('Create artificial playback .WAV file:', self.AnalysesSettings)
        create_usv_playback_wav_label.setFont(QFont(self.font_id, 11 + self.font_size_increase))
        create_usv_playback_wav_label.setStyleSheet('QLabel { color: #F58025; font-weight: bold;}')
        create_usv_playback_wav_label.move(analyses_col_two_x1, 280)
        self.create_usv_playback_wav_cb = QComboBox(self.AnalysesSettings)
        self.create_usv_playback_wav_cb.addItems(['No', 'Yes'])
        self.create_usv_playback_wav_cb.setStyleSheet('QComboBox { width: 57px; }')
        self.create_usv_playback_wav_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='create_usv_playback_wav_cb_bool'))
        self.create_usv_playback_wav_cb.move(analyses_col_two_x2, 280)

        num_usv_files_label = QLabel('Number of artificial playback files:', self.AnalysesSettings)
        num_usv_files_label.setFont(QFont(self.font_id, 12 + self.font_size_increase))
        num_usv_files_label.move(analyses_col_two_x1, 310)
        self.num_usv_files = QLineEdit(f"{self.analyses_input_dict['create_usv_playback_wav']['num_usv_files']}", self.AnalysesSettings)
        self.num_usv_files.setFont(QFont(self.font_id, 10 + self.font_size_increase))
        self.num_usv_files.setStyleSheet('QLineEdit { width: 85px; }')
        self.num_usv_files.move(analyses_col_two_x2, 310)

        total_usv_number_label = QLabel('Total number os USVs per file:', self.AnalysesSettings)
        total_usv_number_label.setFont(QFont(self.font_id, 12 + self.font_size_increase))
        total_usv_number_label.move(analyses_col_two_x1, 340)
        self.total_usv_number = QLineEdit(f"{self.analyses_input_dict['create_usv_playback_wav']['total_usv_number']}", self.AnalysesSettings)
        self.total_usv_number.setFont(QFont(self.font_id, 10 + self.font_size_increase))
        self.total_usv_number.setStyleSheet('QLineEdit { width: 85px; }')
        self.total_usv_number.move(analyses_col_two_x2, 340)

        ipi_duration_label = QLabel('Fixed silence between USVs (s):', self.AnalysesSettings)
        ipi_duration_label.setFont(QFont(self.font_id, 12 + self.font_size_increase))
        ipi_duration_label.move(analyses_col_two_x1, 370)
        self.ipi_duration = QLineEdit(f"{self.analyses_input_dict['create_usv_playback_wav']['ipi_duration']}", self.AnalysesSettings)
        self.ipi_duration.setFont(QFont(self.font_id, 10 + self.font_size_increase))
        self.ipi_duration.setStyleSheet('QLineEdit { width: 85px; }')
        self.ipi_duration.move(analyses_col_two_x2, 370)

        create_naturalistic_usv_playback_wav_label = QLabel('Create naturalistic playback .WAV file:', self.AnalysesSettings)
        create_naturalistic_usv_playback_wav_label.setFont(QFont(self.font_id, 11 + self.font_size_increase))
        create_naturalistic_usv_playback_wav_label.setStyleSheet('QLabel { color: #F58025; font-weight: bold;}')
        create_naturalistic_usv_playback_wav_label.move(analyses_col_two_x1, 400)
        self.create_naturalistic_usv_playback_wav_cb = QComboBox(self.AnalysesSettings)
        self.create_naturalistic_usv_playback_wav_cb.addItems(['No', 'Yes'])
        self.create_naturalistic_usv_playback_wav_cb.setStyleSheet('QComboBox { width: 57px; }')
        self.create_naturalistic_usv_playback_wav_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='create_naturalistic_usv_playback_wav_cb_bool'))
        self.create_naturalistic_usv_playback_wav_cb.move(analyses_col_two_x2, 400)

        num_naturalistic_usv_files_label = QLabel('Number of naturalistic playback files:', self.AnalysesSettings)
        num_naturalistic_usv_files_label.setFont(QFont(self.font_id, 12 + self.font_size_increase))
        num_naturalistic_usv_files_label.move(analyses_col_two_x1, 430)
        self.num_naturalistic_usv_files = QLineEdit(f"{self.analyses_input_dict['create_naturalistic_usv_playback_wav']['num_naturalistic_usv_files']}", self.AnalysesSettings)
        self.num_naturalistic_usv_files.setFont(QFont(self.font_id, 10 + self.font_size_increase))
        self.num_naturalistic_usv_files.setStyleSheet('QLineEdit { width: 85px; }')
        self.num_naturalistic_usv_files.move(analyses_col_two_x2, 430)

        total_playback_file_duration_label = QLabel('Total playback file duration (s):', self.AnalysesSettings)
        total_playback_file_duration_label.setFont(QFont(self.font_id, 12 + self.font_size_increase))
        total_playback_file_duration_label.move(analyses_col_two_x1, 460)
        self.total_playback_file_duration = QLineEdit(f"{self.analyses_input_dict['create_naturalistic_usv_playback_wav']['total_acceptable_naturalistic_playback_time']}", self.AnalysesSettings)
        self.total_playback_file_duration.setFont(QFont(self.font_id, 10 + self.font_size_increase))
        self.total_playback_file_duration.setStyleSheet('QLineEdit { width: 85px; }')
        self.total_playback_file_duration.move(analyses_col_two_x2, 460)

        preferred_mouse_sex_label = QLabel('Sex that produced the vocalizations:', self.AnalysesSettings)
        preferred_mouse_sex_label.setFont(QFont(self.font_id, 12 + self.font_size_increase))
        preferred_mouse_sex_label.move(analyses_col_two_x1, 490)
        self.preferred_mouse_sex_list = sorted(['combined', 'male', 'female'], key=lambda x: x == self.analyses_input_dict['create_naturalistic_usv_playback_wav']['naturalistic_playback_snippets_dir_prefix'], reverse=True)
        self.preferred_mouse_sex_cb = QComboBox(self.AnalysesSettings)
        self.preferred_mouse_sex_cb.addItems(self.preferred_mouse_sex_list)
        self.preferred_mouse_sex_cb.setStyleSheet('QComboBox { width: 57px; }')
        self.preferred_mouse_sex_cb.activated.connect(partial(self._combo_box_preferred_mouse_sex, variable_id='preferred_mouse_sex'))
        self.preferred_mouse_sex_cb.move(analyses_col_two_x2, 490)

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

        self.analyses_input_dict['analyses_booleans']['create_naturalistic_usv_playback_wav_bool'] = self.create_naturalistic_usv_playback_wav_cb_bool
        self.create_naturalistic_usv_playback_wav_cb_bool = False

        self.analyses_input_dict['analyses_booleans']['frequency_shift_audio_segment_bool'] = self.frequency_shift_audio_segment_cb_bool
        self.frequency_shift_audio_segment_cb_bool = False

        self.analyses_input_dict['create_usv_playback_wav']['num_usv_files'] = int(ast.literal_eval(self.num_usv_files.text()))
        self.analyses_input_dict['create_usv_playback_wav']['total_usv_number'] = int(ast.literal_eval(self.total_usv_number.text()))
        self.analyses_input_dict['create_usv_playback_wav']['ipi_duration'] = float(ast.literal_eval(self.ipi_duration.text()))

        self.analyses_input_dict['create_naturalistic_usv_playback_wav']['num_naturalistic_usv_files'] = int(ast.literal_eval(self.num_naturalistic_usv_files.text()))
        self.analyses_input_dict['create_naturalistic_usv_playback_wav']['total_acceptable_naturalistic_playback_time'] = int(ast.literal_eval(self.total_playback_file_duration.text()))
        self.analyses_input_dict['create_naturalistic_usv_playback_wav']['naturalistic_playback_snippets_dir_prefix'] = self.preferred_mouse_sex

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

        self.visualizations_credentials_dir_edit = QLineEdit(f"{self.visualizations_input_dict['credentials_directory']}", self.VisualizationsSettings)
        update_visualizations_credentials_dir = partial(self._update_nested_dict_value, self.visualizations_input_dict, ('credentials_directory',))
        self.visualizations_credentials_dir_edit.textChanged.connect(update_visualizations_credentials_dir)
        self.visualizations_credentials_dir_edit.setPlaceholderText('Credentials directory')
        self.visualizations_credentials_dir_edit.setFont(QFont(self.font_id, 10 + self.font_size_increase))
        self.visualizations_credentials_dir_edit.setStyleSheet('QLineEdit { width: 285px; }')
        self.visualizations_credentials_dir_edit.move(10, 365)
        visualizations_credentials_dir_btn = QPushButton('Browse', self.VisualizationsSettings)
        visualizations_credentials_dir_btn.setFont(QFont(self.font_id, 8 + self.font_size_increase))
        visualizations_credentials_dir_btn.move(300, 365)
        visualizations_credentials_dir_btn.setStyleSheet('QPushButton { min-width: 36px; min-height: 12px; max-width: 36px; max-height: 13px; }')
        open_visualizations_credentials_dir_dialog = partial(self._open_directory_dialog, self.visualizations_credentials_dir_edit, 'Select credentials directory')
        visualizations_credentials_dir_btn.clicked.connect(open_visualizations_credentials_dir_dialog)

        pc_usage_visualizations_label = QLabel('Notify e-mail(s) of PC usage:', self.VisualizationsSettings)
        pc_usage_visualizations_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        pc_usage_visualizations_label.move(10, 395)
        self.pc_usage_visualizations = QLineEdit('', self.VisualizationsSettings)
        self.pc_usage_visualizations.setFont(QFont(self.font_id, 10+self.font_size_increase))
        self.pc_usage_visualizations.setStyleSheet('QLineEdit { width: 135px; }')
        self.pc_usage_visualizations.move(225, 395)

        visualizations_pc_label = QLabel('Visualizations PC of choice:', self.VisualizationsSettings)
        visualizations_pc_label.setFont(QFont(self.font_id, 12+self.font_size_increase))
        visualizations_pc_label.move(10, 425)
        self.loaded_visualizations_pc_list = sorted(self.visualizations_input_dict['send_email']['visualizations_pc_list'], key=lambda x: x == self.visualizations_input_dict['send_email']['visualizations_pc_choice'], reverse=True)
        self.visualizations_pc_cb = QComboBox(self.VisualizationsSettings)
        self.visualizations_pc_cb.addItems(self.loaded_visualizations_pc_list)
        self.visualizations_pc_cb.setStyleSheet('QComboBox { width: 107px; }')
        self.visualizations_pc_cb.activated.connect(partial(self._combo_box_prior_visualizations_pc_choice, variable_id='visualizations_pc_choice'))
        self.visualizations_pc_cb.move(225, 425)

        dv_label = QLabel('Select data visualization', self.VisualizationsSettings)
        dv_label.setFont(QFont(self.font_id, 13+self.font_size_increase))
        dv_label.setStyleSheet('QLabel { font-weight: bold;}')
        dv_label.move(10, 465)

        plot_behavioral_features_label = QLabel('Plot 3D behavioral tuning curves:', self.VisualizationsSettings)
        plot_behavioral_features_label.setFont(QFont(self.font_id, 11 + self.font_size_increase))
        plot_behavioral_features_label.setStyleSheet('QLabel { color: #F58025; font-weight: bold;}')
        plot_behavioral_features_label.move(10, 495)
        self.plot_behavioral_features_cb = QComboBox(self.VisualizationsSettings)
        self.plot_behavioral_features_cb.addItems(['No', 'Yes'])
        self.plot_behavioral_features_cb.setStyleSheet('QComboBox { width: 57px; }')
        self.plot_behavioral_features_cb.activated.connect(partial(self._combo_box_prior_false, variable_id='plot_behavioral_tuning_cb_bool'))
        self.plot_behavioral_features_cb.move(275, 495)

        smoothing_sd_label = QLabel('Ratemap smoothing sigma (bins):', self.VisualizationsSettings)
        smoothing_sd_label.setFont(QFont(self.font_id, 12 + self.font_size_increase))
        smoothing_sd_label.move(10, 525)
        self.smoothing_sd = QLineEdit(f"{self.visualizations_input_dict['neuronal_tuning_figures']['smoothing_sd']}", self.VisualizationsSettings)
        self.smoothing_sd.setFont(QFont(self.font_id, 10 + self.font_size_increase))
        self.smoothing_sd.setStyleSheet('QLineEdit { width: 85px; }')
        self.smoothing_sd.move(275, 525)

        occ_threshold_label = QLabel('Minimal occupancy allowed (s):', self.VisualizationsSettings)
        occ_threshold_label.setFont(QFont(self.font_id, 12 + self.font_size_increase))
        occ_threshold_label.move(10, 555)
        self.occ_threshold = QLineEdit(f"{self.visualizations_input_dict['neuronal_tuning_figures']['occ_threshold']}", self.VisualizationsSettings)
        self.occ_threshold.setFont(QFont(self.font_id, 10 + self.font_size_increase))
        self.occ_threshold.setStyleSheet('QLineEdit { width: 85px; }')
        self.occ_threshold.move(275, 555)

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
        update_arena_root_dir = partial(self._update_nested_dict_value, self.visualizations_input_dict, ('make_behavioral_videos', 'arena_directory'))
        self.arena_root_directory_edit.textChanged.connect(update_arena_root_dir)
        self.arena_root_directory_edit.setPlaceholderText('Arena tracking root directory')
        self.arena_root_directory_edit.setFont(QFont(self.font_id, 10 + self.font_size_increase))
        self.arena_root_directory_edit.setStyleSheet('QLineEdit { width: 285px; }')
        self.arena_root_directory_edit.move(vis_col_two_x1, 70)
        arena_root_directory_btn = QPushButton('Browse', self.VisualizationsSettings)
        arena_root_directory_btn.setFont(QFont(self.font_id, 8 + self.font_size_increase))
        arena_root_directory_btn.move(vis_col_two_x2, 70)
        arena_root_directory_btn.setStyleSheet('QPushButton { min-width: 65px; min-height: 12px; max-width: 656px; max-height: 12px; }')
        open_arena_root_dir_dialog = partial(self._open_directory_dialog, self.arena_root_directory_edit, 'Select arena tracking root directory')
        arena_root_directory_btn.clicked.connect(open_arena_root_dir_dialog)

        self.speaker_audio_file_edit = QLineEdit(f"{self.visualizations_input_dict['make_behavioral_videos']['speaker_audio_file']}", self.VisualizationsSettings)
        update_speaker_audio_file_edit = partial(self._update_nested_dict_value, self.visualizations_input_dict, ('make_behavioral_videos', 'speaker_audio_file'))
        self.speaker_audio_file_edit.textChanged.connect(update_speaker_audio_file_edit)
        self.speaker_audio_file_edit.setPlaceholderText('Speaker playback file')
        self.speaker_audio_file_edit.setFont(QFont(self.font_id, 10 + self.font_size_increase))
        self.speaker_audio_file_edit.setStyleSheet('QLineEdit { width: 285px; }')
        self.speaker_audio_file_edit.move(vis_col_two_x1, 100)
        speaker_audio_file_btn = QPushButton('Browse', self.VisualizationsSettings)
        speaker_audio_file_btn.setFont(QFont(self.font_id, 8 + self.font_size_increase))
        speaker_audio_file_btn.move(vis_col_two_x2, 100)
        speaker_audio_file_btn.setStyleSheet('QPushButton { min-width: 65px; min-height: 12px; max-width: 656px; max-height: 12px; }')
        speaker_audio_file_dialog = partial(self._open_file_dialog, self.speaker_audio_file_edit, 'Select speaker audio file', 'Wave Files (*.wav)')
        speaker_audio_file_btn.clicked.connect(speaker_audio_file_dialog)

        self.sequence_audio_file_edit = QLineEdit('', self.VisualizationsSettings)
        update_sequence_audio_file_edit = partial(self._update_nested_dict_value, self.visualizations_input_dict, ('make_behavioral_videos', 'sequence_audio_file'))
        self.sequence_audio_file_edit.textChanged.connect(update_sequence_audio_file_edit)
        self.sequence_audio_file_edit.setPlaceholderText('Audible USV sequence file')
        self.sequence_audio_file_edit.setFont(QFont(self.font_id, 10 + self.font_size_increase))
        self.sequence_audio_file_edit.setStyleSheet('QLineEdit { width: 285px; }')
        self.sequence_audio_file_edit.move(vis_col_two_x1, 130)
        sequence_audio_file_btn = QPushButton('Browse', self.VisualizationsSettings)
        sequence_audio_file_btn.setFont(QFont(self.font_id, 8 + self.font_size_increase))
        sequence_audio_file_btn.move(vis_col_two_x2, 130)
        sequence_audio_file_btn.setStyleSheet('QPushButton { min-width: 65px; min-height: 12px; max-width: 656px; max-height: 12px; }')
        sequence_audio_file_dialog = partial(self._open_file_dialog, self.sequence_audio_file_edit, 'Select audible sequence file', 'Wave Files (*.wav)')
        sequence_audio_file_btn.clicked.connect(sequence_audio_file_dialog)

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
                          'hpss_power', 'n_deriv_smooth','das_conda', 'das_model_base', 'das_output_type',
                          'smooth_scale', 'static_reference_len', 'weight_rigid', 'weight_weak',
                          'reprojection_error_threshold', 'regularization_function',
                          'segment_confidence_threshold', 'segment_minlen', 'segment_fillgap',
                          'rigid_body_constraints', 'weak_body_constraints', 'vcl_conda']
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

        self.processing_input_dict['vocalocator']['vcl_version'] = str(getattr(self, 'vcl_version'))
        self.processing_input_dict['vocalocator']['vcl_conda_env_name'] = self.vcl_conda

        self.processing_input_dict['synchronize_files']['Synchronizer']['crop_wav_files_to_video']['device_receiving_input'] = str(getattr(self, 'device_receiving_input'))
        self.device_receiving_input = self.processing_input_dict['synchronize_files']['Synchronizer']['crop_wav_files_to_video']['device_receiving_input']

        self.processing_input_dict['send_email']['Messenger']['processing_pc_choice'] = str(getattr(self, 'processing_pc_choice'))

        self.processing_input_dict['modify_files']['Operator']['rectify_video_fps']['conversion_target_file'] = self.conversion_target_file
        self.processing_input_dict['modify_files']['Operator']['rectify_video_fps']['constant_rate_factor'] = int(round(ast.literal_eval(self.constant_rate_factor)))
        self.processing_input_dict['modify_files']['Operator']['rectify_video_fps']['encoding_preset'] = str(getattr(self, 'encoding_preset'))
        self.processing_input_dict['synchronize_files']['Synchronizer']['crop_wav_files_to_video']['triggerbox_ch_receiving_input'] = int(ast.literal_eval(self.ch_receiving_input))
        self.processing_input_dict['modify_files']['Operator']['filter_audio_files']['filter_freq_bounds'] = [int(ast.literal_eval(freq_bound)) for freq_bound in self.filter_freq_bounds]
        self.processing_input_dict['modify_files']['Operator']['hpss_audio']['stft_window_length_hop_size'] = [int(ast.literal_eval(stft_value)) for stft_value in self.stft_window_hop]
        self.processing_input_dict['modify_files']['Operator']['hpss_audio']['kernel_size'] = tuple([int(ast.literal_eval(kernel_value)) for kernel_value in self.hpss_kernel_size])
        self.processing_input_dict['modify_files']['Operator']['hpss_audio']['hpss_power'] = float(ast.literal_eval(self.hpss_power))
        self.processing_input_dict['modify_files']['Operator']['hpss_audio']['margin'] = tuple([int(ast.literal_eval(margin_value)) for margin_value in self.hpss_margin])
        self.processing_input_dict['modify_files']['Operator']['get_spike_times']['min_spike_num'] = int(ast.literal_eval(self.min_spike_num))
        self.processing_input_dict['synchronize_files']['Synchronizer']['find_audio_sync_trains']['sync_ch_receiving_input'] = int(ast.literal_eval(self.a_ch_receiving_input))
        self.processing_input_dict['extract_phidget_data']['Gatherer']['prepare_data_for_analyses']['extra_data_camera'] = self.phidget_extra_data_camera

        self.processing_input_dict['preprocess_data']['root_directories'] = self.processing_dir_edit
        self.processing_input_dict['anipose_operations']['ConvertTo3D']['translate_rotate_metric']['experimental_codes'] = self.exp_codes_edit
        self.processing_input_dict['send_email']['Messenger']['send_message']['receivers'] = self.pc_usage_process
        self.processing_input_dict['synchronize_files']['Synchronizer']['find_video_sync_trains']['sync_camera_serial_num'] = self.v_camera_serial_num
        self.processing_input_dict['modify_files']['Operator']['filter_audio_files']['filter_dirs'] = self.filter_dirs
        self.processing_input_dict['modify_files']['Operator']['concatenate_audio_files']['concat_dirs'] = self.concat_dirs

        self.processing_input_dict['usv_inference']['FindMouseVocalizations']['das_command_line_inference']['das_conda_env_name'] = self.das_conda
        self.processing_input_dict['usv_inference']['FindMouseVocalizations']['das_command_line_inference']['model_name_base'] = self.das_model_base
        self.processing_input_dict['usv_inference']['FindMouseVocalizations']['das_command_line_inference']['output_file_type'] = self.das_output_type

        self.processing_input_dict['usv_inference']['FindMouseVocalizations']['das_command_line_inference']['segment_confidence_threshold'] = float(ast.literal_eval(self.segment_confidence_threshold))
        self.processing_input_dict['usv_inference']['FindMouseVocalizations']['das_command_line_inference']['segment_minlen'] = float(ast.literal_eval(self.segment_minlen))
        self.processing_input_dict['usv_inference']['FindMouseVocalizations']['das_command_line_inference']['segment_fillgap'] = float(ast.literal_eval(self.segment_fillgap))

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
        self.processing_input_dict['processing_booleans']['prepare_assign_vocalizations'] = self.prepare_assign_usv_cb_bool
        self.prepare_assign_usv_cb_bool = False
        self.processing_input_dict['processing_booleans']['assign_vocalizations'] = self.assign_usv_cb_bool
        self.assign_usv_cb_bool = False

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

        if self.falkner_checkbox_bool:
            if not f"/mnt/falkner/{self.exp_id}/Data" in self.destination_linux_global:
                self.destination_linux_global.insert(0, f"/mnt/falkner/{self.exp_id}/Data")
            if not f"F:\\{self.exp_id}\\Data" in self.destination_win_global:
                self.destination_win_global.insert(0, f"F:\\{self.exp_id}\\Data")
        else:
            if f"/mnt/falkner/{self.exp_id}/Data" in self.destination_linux_global:
                self.destination_linux_global.pop(self.destination_linux_global.index(f"/mnt/falkner/{self.exp_id}/Data"))
                if not f"/mnt/murthy/{self.exp_id}/Data" in self.destination_linux_global:
                    self.destination_linux_global.insert(0, f"/mnt/murthy/{self.exp_id}/Data")
            if f"F:\\{self.exp_id}\\Data" in self.destination_win_global:
                self.destination_win_global.pop(self.destination_win_global.index(f"F:\\{self.exp_id}\\Data"))
                if not f"M:\\{self.exp_id}\\Data" in self.destination_win_global:
                    self.destination_win_global.insert(0, f"M:\\{self.exp_id}\\Data")

        if self.murthy_checkbox_bool:
            if not f"M:\\{self.exp_id}\\Data" in self.destination_win_global:
                self.destination_win_global.insert(1, f"M:\\{self.exp_id}\\Data")
        else:
            if f"/mnt/murthy/{self.exp_id}/Data" in self.destination_linux_global:
                self.destination_linux_global.pop(self.destination_linux_global.index(f"/mnt/murthy/{self.exp_id}/Data"))
                if not f"/mnt/falkner/{self.exp_id}/Data" in self.destination_linux_global:
                    self.destination_linux_global.insert(0, f"/mnt/falkner/{self.exp_id}/Data")
            if f"M:\\{self.exp_id}\\Data" in self.destination_win_global:
                self.destination_win_global.pop(self.destination_win_global.index(f"M:\\{self.exp_id}\\Data"))
                if not f"F:\\{self.exp_id}\\Data" in self.destination_win_global:
                    self.destination_win_global.insert(0, f"F:\\{self.exp_id}\\Data")

        self.exp_settings_dict['recording_files_destination_linux'] = self.destination_linux_global
        self.exp_settings_dict['recording_files_destination_win'] = self.destination_win_global

        self.exp_settings_dict['video_session_duration'] = ast.literal_eval(self.video_session_duration)
        self.exp_settings_dict['calibration_duration'] = ast.literal_eval(self.calibration_session_duration)
        self.exp_settings_dict['ethernet_network'] = self.ethernet_network

        self.exp_settings_dict['conduct_tracking_calibration'] = self.conduct_tracking_calibration_cb_bool
        self.exp_settings_dict['conduct_audio_recording'] = self.conduct_audio_cb_bool
        self.exp_settings_dict['disable_ethernet'] = self.disable_ethernet_cb_bool

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

        video_dict_keys = ['expected_cameras', 'recording_codec']

        if self.usgh_devices_sync_cb_bool:
            self.exp_settings_dict['audio']['devices']['usghflags'] = 1574
            self.exp_settings_dict['audio']['usgh_devices_sync'] = True
        else:
            self.exp_settings_dict['audio']['devices']['usghflags'] = 1862
            self.exp_settings_dict['audio']['usgh_devices_sync'] = False

        if self.usgh_sr == '300000':
            self.exp_settings_dict['audio']['devices']['fabtast'] = 300000
            self.exp_settings_dict['audio']['mics_config']['fo'] = 300000.0
        else:
            self.exp_settings_dict['audio']['devices']['fabtast'] = 250000
            self.exp_settings_dict['audio']['mics_config']['fo'] = 250000.0

        self.exp_settings_dict['audio']['cpu_priority'] = self.cpu_priority

        if not self.cpu_affinity_edit.text() == '':
            self.exp_settings_dict['audio']['cpu_affinity'] = [int(x) for x in self.cpu_affinity_edit.text().split(',')]
        else:
            self.exp_settings_dict['audio']['cpu_affinity'] = []

        self.exp_settings_dict['video']['general']['monitor_recording'] = self.monitor_recording_cb_bool
        self.exp_settings_dict['video']['general']['monitor_specific_camera'] = self.monitor_specific_camera_cb_bool
        self.exp_settings_dict['video']['general']['specific_camera_serial'] = self.specific_camera_serial
        self.exp_settings_dict['video']['general']['delete_post_copy'] = self.delete_post_copy_cb_bool

        self.exp_settings_dict['video']['general']['recording_frame_rate'] = self.cameras_frame_rate.value()
        self.exp_settings_dict['video']['general']['calibration_frame_rate'] = self.calibration_frame_rate.value()

        for variable in video_dict_keys:
            if variable == 'recording_codec':
                self.exp_settings_dict['video']['general'][variable] = str(getattr(self, variable))
            else:
                expected_cameras_list = self.exp_settings_dict['video']['general'][variable]
                for camera_id in self.exp_settings_dict['video']['general']['available_cameras']:
                    rec_status_bool = self.__dict__[f'{camera_id}_rec_checkbox_bool']
                    if rec_status_bool and camera_id not in expected_cameras_list:
                        expected_cameras_list.insert(self.exp_settings_dict['video']['general']['available_cameras'].index(camera_id), camera_id)
                    if not rec_status_bool and camera_id in expected_cameras_list:
                        expected_cameras_list.pop(expected_cameras_list.index(camera_id))
                self.exp_settings_dict['video']['general'][variable] = expected_cameras_list

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

        das_model_dir = configure_path(self.processing_input_dict['usv_inference']['FindMouseVocalizations']['das_command_line_inference']['das_model_directory'])
        sleap_inference_dir = configure_path(self.processing_input_dict['prepare_cluster_job']['inference_root_dir'])
        vcl_model_dir = configure_path(self.processing_input_dict['vocalocator']['vcl_model_directory'])

        self.das_model_dir_global = replace_name_in_path(experimenter_list=self.exp_settings_dict['experimenter_list'],
                                                         recording_files_destinations=[das_model_dir],
                                                         exp_id=self.exp_id)

        self.sleap_inference_dir_global = replace_name_in_path(experimenter_list=self.exp_settings_dict['experimenter_list'],
                                                               recording_files_destinations=[sleap_inference_dir],
                                                               exp_id=self.exp_id)

        self.vcl_model_dir_global = replace_name_in_path(experimenter_list=self.exp_settings_dict['experimenter_list'],
                                                         recording_files_destinations=[vcl_model_dir],
                                                         exp_id=self.exp_id)

        self.avisoft_rec_dir_global = self.exp_settings_dict['avisoft_recorder_exe_directory']
        self.avisoft_base_dir_global = self.exp_settings_dict['avisoft_basedirectory']
        self.avisoft_config_dir_global = self.exp_settings_dict['avisoft_config_directory']
        self.coolterm_base_dir_global = self.exp_settings_dict['coolterm_basedirectory']
        self.recording_credentials_dir_global = f"{platformdirs.user_config_dir(appname='usv_playpen', appauthor='lab')}{os.sep}.credentials_{self.exp_id}"
        self.exp_settings_dict['credentials_directory'] = self.recording_credentials_dir_global

        self.destination_linux_global = replace_name_in_path(experimenter_list=self.exp_settings_dict['experimenter_list'],
                                                             recording_files_destinations=self.exp_settings_dict['recording_files_destination_linux'],
                                                             exp_id=self.exp_id).split(',')

        self.destination_win_global = replace_name_in_path(experimenter_list=self.exp_settings_dict['experimenter_list'],
                                                           recording_files_destinations=self.exp_settings_dict['recording_files_destination_win'],
                                                           exp_id=self.exp_id).split(',')

        self.processing_input_dict['send_email']['Messenger']['experimenter'] = f'{self.exp_id}'
        self.analyses_input_dict['send_email']['experimenter'] = f'{self.exp_id}'
        self.exp_settings_dict['experimenter'] = f'{self.exp_id}'

    def _combo_box_cpu_priority(self,
                                index: int,
                                variable_id: str = None) -> None:
        """
        CPU priority combo box.

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

        for idx in range(len(self.cpu_priority_cb_list)):
            if index == idx:
                self.__dict__[variable_id] = self.cpu_priority_cb_list[idx]
                break

    def _combo_box_usgh_sr(self,
                           index: int,
                           variable_id: str = None) -> None:
        """
        USGH devices sampling rate (can be 250 kHz or 300 kHz).

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

        for idx in range(len(self.usgh_sr_cb_list)):
            if index == idx:
                self.__dict__[variable_id] = self.usgh_sr_cb_list[idx]
                break

    def _combo_box_specific_camera(self,
                                   index: int,
                                   variable_id: str = None) -> None:
        """
        Specific monitoring camera serial number combo box.

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

        for idx in range(len(self.specific_camera_serial_list)):
            if index == idx:
                self.__dict__[variable_id] = self.specific_camera_serial_list[idx]
                break

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


    def _combo_box_preferred_mouse_sex(self,
                                       index: int,
                                       variable_id: str = None) -> None:
        """
        Preferred sex of the mouse which produced USVs.

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

        for idx in range(len(self.preferred_mouse_sex_list)):
            if index == idx:
                self.__dict__[variable_id] = self.preferred_mouse_sex_list[idx]
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

    def _combo_box_vcl_version(self,
                               index: int,
                               variable_id: str = None) -> None:
        """
        USV assignment vocalocator version.

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

        for idx in range(len(self.assign_type_list)):
            if index == idx:
                self.__dict__[variable_id] = self.assign_type_list[idx]
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

    def _checkbox_state_read(self, checkbox_id: str = None, variable_id: str = None) -> None:
        """
        Reads the state of a checkbox and updates the corresponding variable.

        Parameters
        ----------
        checkbox_id (str)
            Name of the checkbox.
        variable_id (str)
            Name of the variable to be updated.
        ----------

        Returns
        -------
        -------
        """

        if self.__dict__[checkbox_id].isChecked():
            self.__dict__[variable_id] = True
        else:
            self.__dict__[variable_id] = False

    def _create_checkbox_general(self,
                                 camera_id: str = None,
                                 x_start: int = None) -> None:
        """
        Creates checkbox for choosing cameras to record with.

        Parameters
        ----------
        camera_id (str)
            Camera ID (e.g., 21372316).
        x_start (int)
            Starting x position for the camera settings.
        ----------

        Returns
        -------
        -------
        """

        self.__dict__[f'{camera_id}_rec_checkbox'] = QCheckBox(self.VideoSettings, text=camera_id)
        self.__dict__[f'{camera_id}_rec_checkbox'].setStyleSheet("""QCheckBox {spacing: 2px;}
                                                                    QCheckBox::indicator {border: 1px solid grey; width: 15px; height: 15px; border-radius: 7.5px;}
                                                                    QCheckBox::indicator:checked {background-color: #F58025;}""")
        self.__dict__[f'{camera_id}_rec_checkbox'].setChecked(self.__dict__[f'{camera_id}_rec_checkbox_bool'])
        self.__dict__[f'{camera_id}_rec_checkbox'].setFont(QFont(self.font_id, 10 + self.font_size_increase))
        self.__dict__[f'{camera_id}_rec_checkbox'].move(x_start, 230)
        self.__dict__[f'{camera_id}_rec_checkbox'].stateChanged.connect(partial(self._checkbox_state_read, checkbox_id=f'{camera_id}_rec_checkbox', variable_id=f'{camera_id}_rec_checkbox_bool'))

    def _create_checkbox_file_destinations(self,
                                           lab_id: str = None,
                                           x_start: int = None) -> None:
        """
        Creates checkbox for file server destinations for recorded files.

        Parameters
        ----------
        lab_id (str)
            Lab ID (e.g., Falkner).
        x_start (int)
            Starting x position for the camera settings.
        ----------

        Returns
        -------
        -------
        """

        self.__dict__[f'{lab_id}_checkbox'] = QCheckBox(self.Record, text=f"{lab_id.capitalize()} (/cup/{lab_id}/{self.exp_id}/Data/)")
        self.__dict__[f'{lab_id}_checkbox'].setStyleSheet("""QCheckBox {spacing: 5px;}
                                                             QCheckBox::indicator {border: 2px solid grey; width: 15px; height: 15px; border-radius: 7.5px;}
                                                             QCheckBox::indicator:checked {background-color: #F58025;}""")
        self.__dict__[f'{lab_id}_checkbox'].setChecked(self.__dict__[f'{lab_id}_checkbox_bool'])
        self.__dict__[f'{lab_id}_checkbox'].setFont(QFont(self.font_id, 14 + self.font_size_increase))
        self.__dict__[f'{lab_id}_checkbox'].move(x_start, 220)
        self.__dict__[f'{lab_id}_checkbox'].stateChanged.connect(partial(self._checkbox_state_read, checkbox_id=f'{lab_id}_checkbox', variable_id=f'{lab_id}_checkbox_bool'))

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

        for idx in range(len(self.usgh_device_receiving_input_list)):
            if index == idx:
                self.__dict__[variable_id] = self.usgh_device_receiving_input_list[idx]
                break

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
        Creates buttons for the Main window.

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

        self.login_button = QPushButton(QIcon(password_icon), "", self.Main)
        self.login_button.setToolTip("login credentials")
        self.login_button.setObjectName("loginButton")
        self.login_button.setFixedSize(32, 32)
        self.login_button.move(385, 2)
        self.login_button.clicked.connect(self.credentials_window)

    def _create_buttons_credentials(self,
                                    class_option: str = None,
                                    button_pos_y: int = None,
                                    next_button_x_pos: int = None) -> None:

        """
        Creates buttons for Credentials window.

        Parameters
        ----------
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

        self.button_map = {'Previous': QPushButton(QIcon(previous_icon), 'Previous', class_option),
                           'Main': QPushButton(QIcon(main_icon), 'Main', class_option),
                           'Save': QPushButton(QIcon(save_icon), 'Save', class_option)}

        self.button_map['Previous'].move(5, button_pos_y)
        self.button_map['Previous'].setFont(QFont(self.font_id, 8+self.font_size_increase))
        self.button_map['Previous'].clicked.connect(self.main_window)

        self.button_map['Main'].move(100, button_pos_y)
        self.button_map['Main'].setFont(QFont(self.font_id, 8+self.font_size_increase))
        self.button_map['Main'].clicked.connect(self.main_window)

        self.button_map['Save'].move(next_button_x_pos, button_pos_y)
        self.button_map['Save'].setFont(QFont(self.font_id, 8 + self.font_size_increase))
        self.button_map['Save'].clicked.connect(self._start_saving)


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
            next_win_connect = [self.record_four]
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

    def _start_saving(self) -> None:
        """
        Saves credential files:
        - e-mail credentials file
        - CUP credentials file
        - Motif credentials file.

        Parameters
        ----------
        ----------

        Returns
        -------
        -------
        """

        # hide credentials directory
        credentials_dir = Path(f"{self.credentials_save_dir_edit.text()}{os.sep}.credentials_{self.exp_id}")
        credentials_dir.mkdir(parents=True, exist_ok=True)

        if platform.system() == 'Windows':
            file_attribute_hidden = 0x02
            ctypes.windll.kernel32.SetFileAttributesW(str(credentials_dir), file_attribute_hidden)

        motif_config = configparser.ConfigParser()
        motif_config['motif'] = {
            'master_ip_address': f"{self.motif_master_ip.text()}",
            'second_ip_address': f"{self.motif_second_ip.text()}",
            'ssh_port': f"{self.motif_ssh_port.text()}",
            'ssh_username': f"{self.motif_username.text()}",
            'ssh_password': f"{self.motif_password.text()}",
            'api': f"{self.motif_api.text()}"}
        with open(f"{credentials_dir}{os.sep}motif_config.ini", mode='w') as motif_configfile:
            motif_config.write(motif_configfile, space_around_delimiters=False)

        university_config = configparser.ConfigParser()
        university_config['cup'] = {
            'username': f"{self.university_username.text()}",
            'password': f"{self.university_password.text()}"}
        with open(f"{credentials_dir}{os.sep}cup_config.ini", mode='w') as university_configfile:
            university_config.write(university_configfile, space_around_delimiters=False)

        emai_config = configparser.ConfigParser()
        emai_config['email'] = {
            'email_host': f"{self.email_host.text()}",
            'email_port': f"{self.email_port.text()}",
            'email_address': f"{self.email_address.text()}",
            'email_password': f"{self.email_password.text()}"}
        with open(f"{credentials_dir}{os.sep}email_config.ini", mode='w') as email_configfile:
            emai_config.write(email_configfile, space_around_delimiters=False)

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

        updated_metadata = self.run_exp.conduct_behavioral_recording()

        if updated_metadata:
            self.metadata_settings = updated_metadata
            self._save_metadata_to_yaml()

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

    def _update_nested_dict_value(self,
                                  base_dict: dict = None,
                                  keys_path: tuple = None,
                                  text: str = None) -> None:
        """
        Updates a value in a nested dictionary using a sequence of keys.

        Parameters
        ----------
        base_dict (dict)
            Dictonary to be updated.
        keys_path (tuple)
            Sequence of keys leading to the target value.
        text (str)
            New value to set.
        ----------

        Returns
        -------
        -------
        """

        # navigate to the second-to-last dictionary
        current_level = base_dict
        for key in keys_path[:-1]:
            current_level = current_level[key]

        # get the final key for setting the value
        final_key = keys_path[-1]

        # update the value in the final dictionary
        current_level[final_key] = text

    def _open_directory_dialog(self,
                               target_line_edit: QLineEdit,
                               dialog_title: str = None,
                               start_dir: str = None) -> None:
        """
        Opens a directory dialog and updates a QLineEdit.

        Parameters
        ----------
        target_line_ediT (QLineEdit)
            QLineEdit to be updated with the selected directory.
        dialog_title (str)
            Title of the dialog window.
        start_dir (str)
            Starting directory for the dialog.
        ----------

        Returns
        -------
        -------
        """

        initial_path = start_dir or target_line_edit.text()

        if not os.path.isdir(initial_path):
            initial_path = os.path.expanduser('~')

        directory_name = QFileDialog.getExistingDirectory(
            self,
            dialog_title,
            initial_path
        )

        if directory_name:
            target_line_edit.setText(directory_name)

    def _open_file_dialog(self,
                          target_line_edit: QLineEdit,
                          dialog_title: str = None,
                          file_filter: str = None,
                          start_dir: str = None) -> None:
        """
        A general-purpose method to open a file selection dialog
        and update a target QLineEdit widget.

        Parameters
        ----------
        target_line_edit (QLineEdit)
            The line edit widget to update with the selected file path.
        dialog_title (str)
            The title for the file dialog window.
        file_filter (str)
            The filter for file types (e.g., 'Wave Files (*.wav)').
        start_dir (str, optional)
            An optional directory to start the dialog in. Defaults to the
            directory of the file currently in the line edit.

        Returns
        -------
        -------
        """

        if start_dir:
            initial_path = start_dir
        else:
            current_file = target_line_edit.text()
            initial_path = os.path.dirname(current_file)

        if not os.path.isdir(initial_path):
            initial_path = os.path.expanduser('~')

        file_name, _ = QFileDialog.getOpenFileName(
            self,
            dialog_title,
            initial_path,
            file_filter
        )

        if file_name:
            target_line_edit.setText(file_name)

    def _location_on_the_screen(self) -> None:
        """
        Places GUI in the top-left corner of screen.

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


def initialize_main_window(no_splash: bool = False) -> QMainWindow:
    """
    Initialize the main GUI window and return the object.

    Parameters
    ----------
    no_splash: Whether to display the splash screen or not.

    Returns
    -------
    The intialized GUI windows.
    """

    # Handle high-resolution displays:
    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, on=True)
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, on=True)

    # Reuse existing app if present
    usv_playpen_app = QApplication.instance()
    if usv_playpen_app is None:
        usv_playpen_app = QApplication(sys.argv)

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

    splash = None
    if not no_splash:
        splash = QSplashScreen(QPixmap(splash_icon))
        splash.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        splash.show()
        QTest.qWait(2500)

    initial_values_dict = {'exp_id': _toml['experimenter'],
                           'conduct_audio_cb_bool': _toml['conduct_audio_recording'], 'conduct_tracking_calibration_cb_bool': _toml['conduct_tracking_calibration'],
                           'disable_ethernet_cb_bool': _toml['disable_ethernet'], 'monitor_recording_cb_bool': _toml['video']['general']['monitor_recording'],
                           'monitor_specific_camera_cb_bool': _toml['video']['general']['monitor_specific_camera'], 'delete_post_copy_cb_bool': _toml['video']['general']['delete_post_copy'],
                           'recording_codec': _toml['video']['general']['recording_codec'],
                           'device_receiving_input': processing_input_dict['synchronize_files']['Synchronizer']['crop_wav_files_to_video']['device_receiving_input'],
                           'save_transformed_data': processing_input_dict['anipose_operations']['ConvertTo3D']['translate_rotate_metric']['save_transformed_data'],
                           'usgh_devices_sync_cb_bool': _toml['audio']['usgh_devices_sync'], 'usgh_sr': str(_toml['audio']['devices']['fabtast']), 'cpu_priority': _toml['audio']['cpu_priority'],
                           'conduct_video_concatenation_cb_bool': False, 'conduct_video_fps_change_cb_bool': False,
                           'conduct_multichannel_conversion_cb_bool': False, 'crop_wav_cam_cb_bool': False, 'conc_audio_cb_bool': False, 'filter_audio_cb_bool': False,
                           'conduct_sync_cb_bool': False, 'conduct_hpss_cb_bool': False, 'conduct_ephys_file_chaining_cb_bool': False,
                           'conduct_nv_sync_cb_bool': False, 'split_cluster_spikes_cb_bool': False, 'anipose_calibration_cb_bool': False,
                           'sleap_file_conversion_cb_bool': False, 'anipose_triangulation_cb_bool': False, 'translate_rotate_metric_cb_bool': False,
                           'sleap_cluster_cb_bool': False, 'das_inference_cb_bool': False, 'das_summary_cb_bool': False, 'assign_usv_cb_bool': False,
                           'prepare_assign_usv_cb_bool': False, 'delete_con_file_cb_bool': True, 'board_provided_cb_bool': False, 'triangulate_arena_points_cb_bool': False,
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
                           'processing_pc_choice': processing_input_dict['send_email']['Messenger']['processing_pc_choice'], 'encoding_preset': processing_input_dict['modify_files']['Operator']['rectify_video_fps']['encoding_preset'],
                           'vcl_version': processing_input_dict['vocalocator']['vcl_version'], 'specific_camera_serial': _toml['video']['general']['specific_camera_serial'],
                           '21372315_rec_checkbox_bool': True if '21372315' in _toml['video']['general']['expected_cameras'] else False,
                           '21372316_rec_checkbox_bool': True if '21372316' in _toml['video']['general']['expected_cameras'] else False,
                           '21369048_rec_checkbox_bool': True if '21369048' in _toml['video']['general']['expected_cameras'] else False,
                           '22085397_rec_checkbox_bool': True if '22085397' in _toml['video']['general']['expected_cameras'] else False,
                           '21241563_rec_checkbox_bool': True if '21241563' in _toml['video']['general']['expected_cameras'] else False,
                           'falkner_checkbox_bool': any("F:" in cup_path for cup_path in _toml['recording_files_destination_win']),
                           'murthy_checkbox_bool': any("M:" in cup_path for cup_path in _toml['recording_files_destination_win']),
                           'create_naturalistic_usv_playback_wav_cb_bool': False, 'preferred_mouse_sex': analyses_input_dict['create_naturalistic_usv_playback_wav']['naturalistic_playback_snippets_dir_prefix']}

    usv_playpen_window = USVPlaypenWindow(**initial_values_dict)

    if splash is not None:
        splash.finish(usv_playpen_window)

    return usv_playpen_app, usv_playpen_window


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

    app, window = initialize_main_window()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
