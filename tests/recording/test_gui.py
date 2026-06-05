import pytest

from importlib import metadata
from pathlib import Path

import yaml
from PyQt6.QtCore import Qt  # Added import

from usv_playpen import usv_playpen_gui

@pytest.fixture
def app(qtbot):
    app_obj, window = usv_playpen_gui.initialize_main_window(no_splash=True)
    qtbot.addWidget(window)
    return window


@pytest.fixture
def preserve_settings_toml():
    """Back up and restore the package behavioral_experiments_settings.toml.

    A GUI 'save' exercised under test writes that file via toml.dump, which
    strips comments and reformats it -- churning the tracked package config in
    the repo. Restoring it after the test keeps the working tree clean.
    """
    settings_path = Path(usv_playpen_gui.__file__).parent / '_config' / 'behavioral_experiments_settings.toml'
    original = settings_path.read_text()
    yield
    settings_path.write_text(original)


@pytest.fixture
def preserve_all_settings():
    """Back up and restore every package settings file the GUI's label-save
    handlers may rewrite while navigating between windows (the .toml plus the
    three _parameter_settings JSONs), so window-navigation tests never churn
    the tracked configs."""
    pkg = Path(usv_playpen_gui.__file__).parent
    paths = [
        pkg / '_config' / 'behavioral_experiments_settings.toml',
        pkg / '_parameter_settings' / 'analyses_settings.json',
        pkg / '_parameter_settings' / 'processing_settings.json',
        pkg / '_parameter_settings' / 'visualizations_settings.json',
    ]
    originals = {p: p.read_text() for p in paths}
    yield
    for p, text in originals.items():
        p.write_text(text)

def test_main_window_initialization(app):
    expected_version = metadata.version('usv-playpen').split('.dev')[0].split('.post')[0]
    assert app.windowTitle() == f"USV Playpen v{expected_version}"
    assert app.exp_id_cb.currentText() == "Bartul"

def test_navigation_to_record_window(app, qtbot):
    qtbot.mouseClick(app.button_map['Record'], Qt.MouseButton.LeftButton)
    assert "Record > Select config" in app.windowTitle()
    assert app.recorder_settings_edit is not None

def test_record_one_settings_are_saved(app, qtbot, preserve_settings_toml):
    qtbot.mouseClick(app.button_map['Record'], Qt.MouseButton.LeftButton)

    new_duration = "25.5"

    # Ensure the line edit is cleared (either approach works)
    # Approach 1 (simplest):
    app.video_session_duration.clear()

    # Alternatively (robust): select all then delete
    # qtbot.mouseClick(app.video_session_duration, Qt.MouseButton.LeftButton)
    # qtbot.keyClick(app.video_session_duration, Qt.Key_A, modifier=Qt.KeyboardModifier.ControlModifier)
    # qtbot.keyClick(app.video_session_duration, Qt.Key_Delete)

    qtbot.keyClicks(app.video_session_duration, new_duration)

    assert app.exp_settings_dict['video_session_duration'] != 25.5
    qtbot.mouseClick(app.button_map['Next'], Qt.MouseButton.LeftButton)
    assert app.exp_settings_dict['video_session_duration'] == 25.5


def test_bundled_metadata_yaml_is_a_dict():
    """
    Description
    -----------
    Guards against silent corruption / accidental truncation of the
    bundled ``_config/_metadata.yaml``. If the file is empty (or anything
    other than a top-level mapping with the ``Session`` / ``Equipment``
    blocks), ``yaml.safe_load`` returns ``None`` and the GUI later
    crashes when ``record_three`` does ``self.metadata_settings['Session']
    ['institution']``. We caught exactly this regression after commit
    ff8eef1 shipped an empty file in v0.10.1 / v0.10.2.

    Parameters
    ----------

    Returns
    -------
    None
    """

    metadata_yaml_path = (
        Path(usv_playpen_gui.__file__).parent / '_config' / '_metadata.yaml'
    )
    with open(metadata_yaml_path, 'r') as metadata_file:
        loaded = yaml.safe_load(metadata_file)
    assert isinstance(loaded, dict), (
        f"{metadata_yaml_path} did not parse to a dict (got {type(loaded).__name__}); "
        "the bundled metadata template must not be empty."
    )
    assert 'Session' in loaded and isinstance(loaded['Session'], dict)
    assert 'Equipment' in loaded and isinstance(loaded['Equipment'], dict)
    for required_session_key in (
        'institution', 'lab', 'session_experiment_code',
        'calibration_session', 'session_usv_playback_file',
        'session_description', 'keywords', 'notes',
    ):
        assert required_session_key in loaded['Session'], (
            f"Session.{required_session_key} missing from bundled metadata YAML"
        )


def test_navigation_to_record_metadata_window(app, qtbot):
    """
    Description
    -----------
    Click ``Record`` → ``Next`` → ``Next`` to walk all the way to the
    ``record_three`` (Metadata) window and assert that the Session
    metadata widgets were constructed from the bundled
    ``_config/_metadata.yaml``. Had this test existed before commit
    ff8eef1, it would have flagged the empty-YAML regression that
    crashed the GUI on the recording PC with
    ``TypeError: 'NoneType' object is not subscriptable``.

    Parameters
    ----------

    Returns
    -------
    None
    """

    qtbot.mouseClick(app.button_map['Record'], Qt.MouseButton.LeftButton)
    assert "Record > Select config" in app.windowTitle()

    qtbot.mouseClick(app.button_map['Next'], Qt.MouseButton.LeftButton)
    assert "Record > Audio and Video Settings" in app.windowTitle()

    qtbot.mouseClick(app.button_map['Next'], Qt.MouseButton.LeftButton)
    assert "Record > Metadata" in app.windowTitle()

    assert isinstance(app.metadata_settings, dict)
    assert hasattr(app, 'institution_edit')
    assert app.institution_edit.text() == \
        app.metadata_settings['Session']['institution']
    assert hasattr(app, 'lab_edit')
    assert app.lab_edit.text() == app.metadata_settings['Session']['lab']


# def test_start_recording_button_calls_backend(app, qtbot):
#     qtbot.mouseClick(app.button_map['Record'], Qt.MouseButton.LeftButton)
#     qtbot.mouseClick(app.button_map['Next'], Qt.MouseButton.LeftButton)
#     qtbot.mouseClick(app.button_map['Next'], Qt.MouseButton.LeftButton)
#     qtbot.mouseClick(app.button_map['Next'], Qt.MouseButton.LeftButton)
#     record_button = app.button_map['Record']
#     qtbot.mouseClick(record_button, Qt.MouseButton.LeftButton)
#
# def test_start_processing_button_calls_backend(app, qtbot):
#     qtbot.mouseClick(app.button_map['Process'], Qt.MouseButton.LeftButton)
#     qtbot.keyClicks(app.processing_dir_edit, "/my/test/directory")
#     qtbot.mouseClick(app.button_map['Next'], Qt.MouseButton.LeftButton)
#     process_button = app.button_map['Process']
#     qtbot.mouseClick(process_button, Qt.MouseButton.LeftButton)


def test_navigation_to_record_conduct_window(app, qtbot, preserve_all_settings):
    """Record -> Next -> Next -> Next walks record_one -> record_four,
    building the 'Conduct recording' window (and its widget tree) end-to-end."""
    qtbot.mouseClick(app.button_map['Record'], Qt.MouseButton.LeftButton)
    assert "Record > Select config" in app.windowTitle()
    qtbot.mouseClick(app.button_map['Next'], Qt.MouseButton.LeftButton)
    assert "Record > Audio and Video Settings" in app.windowTitle()
    qtbot.mouseClick(app.button_map['Next'], Qt.MouseButton.LeftButton)
    assert "Record > Metadata" in app.windowTitle()
    qtbot.mouseClick(app.button_map['Next'], Qt.MouseButton.LeftButton)
    assert "Conduct recording" in app.windowTitle()


def test_navigation_to_process_windows(app, qtbot, preserve_all_settings):
    """Process builds the settings window; Next builds the 'Conduct
    Processing' window — covering process_one and process_two."""
    qtbot.mouseClick(app.button_map['Process'], Qt.MouseButton.LeftButton)
    assert "Process recordings > Settings" in app.windowTitle()
    qtbot.mouseClick(app.button_map['Next'], Qt.MouseButton.LeftButton)
    assert "Conduct Processing" in app.windowTitle()


def test_navigation_to_analyze_windows(app, qtbot, preserve_all_settings):
    """Analyze builds the settings window; Next builds the 'Conduct
    Analyses' window — covering analyze_one and analyze_two."""
    qtbot.mouseClick(app.button_map['Analyze'], Qt.MouseButton.LeftButton)
    assert "Analyze data > Settings" in app.windowTitle()
    qtbot.mouseClick(app.button_map['Next'], Qt.MouseButton.LeftButton)
    assert "Conduct Analyses" in app.windowTitle()


def test_navigation_to_visualize_windows(app, qtbot, preserve_all_settings):
    """Visualize builds the settings window; Next builds the 'Conduct
    Visualizations' window — covering visualize_one and visualize_two."""
    qtbot.mouseClick(app.button_map['Visualize'], Qt.MouseButton.LeftButton)
    assert "Visualize data > Settings" in app.windowTitle()
    qtbot.mouseClick(app.button_map['Next'], Qt.MouseButton.LeftButton)
    assert "Conduct Visualizations" in app.windowTitle()


def test_navigation_to_credentials_window(app, qtbot):
    """The login button opens the credentials window — covering
    credentials_window's widget construction."""
    qtbot.mouseClick(app.login_button, Qt.MouseButton.LeftButton)
    assert "Set credentials" in app.windowTitle()
    assert hasattr(app, 'email_address')


from unittest.mock import patch  # noqa: E402

from usv_playpen.usv_playpen_gui import (  # noqa: E402
    ChemoDialog, EphysDialog, LesionDialog, OptoDialog,
)

_INTERVENTION_DIALOGS = [
    (ChemoDialog, "chemogenetics"),
    (EphysDialog, "electrophysiology"),
    (LesionDialog, "lesion"),
    (OptoDialog, "optogenetics"),
]


@pytest.fixture
def app_with_subject(app):
    """Main window primed with a single metadata subject and the YAML/repo
    persistence side-effects stubbed out, so the intervention dialogs can be
    saved/deleted without touching disk."""
    app.metadata_settings = {"Subjects": [{"subject_id": "M", "interventions": {}}]}
    app._save_metadata_to_yaml = lambda: None
    app._update_subject_in_repository = lambda s: None
    return app


@pytest.mark.parametrize("dialog_cls,key", _INTERVENTION_DIALOGS)
def test_intervention_dialog_add_mode_saves(app_with_subject, qtbot, dialog_cls, key):
    """Each intervention dialog must build its add-mode form, collect the mixed
    combo/line-edit field values on OK, and write them into the chosen
    subject's interventions block under the dialog's own key."""
    dlg = dialog_cls(parent=app_with_subject, subject=None)
    qtbot.addWidget(dlg)
    assert "Add" in dlg.windowTitle()
    dlg.subject_combo.setCurrentText("M")
    dlg.save_and_accept()
    interventions = app_with_subject.metadata_settings["Subjects"][0]["interventions"]
    assert key in interventions
    assert isinstance(interventions[key], dict)


@pytest.mark.parametrize("dialog_cls,key", _INTERVENTION_DIALOGS)
def test_intervention_dialog_edit_mode_populates_and_deletes(
    app_with_subject, qtbot, dialog_cls, key,
):
    """In edit mode each dialog locks the subject selector, pre-populates the
    form from the existing intervention, and (on confirmed delete) removes the
    intervention from the subject."""
    subject = {"subject_id": "M", "interventions": {key: {"name": "excitatory"}}}
    app_with_subject.metadata_settings["Subjects"][0] = subject
    dlg = dialog_cls(parent=app_with_subject, subject=subject)
    qtbot.addWidget(dlg)
    assert "Edit" in dlg.windowTitle()
    assert not dlg.subject_combo.isEnabled()

    with patch(
        "usv_playpen.usv_playpen_gui.QMessageBox.question",
        return_value=usv_playpen_gui.QMessageBox.StandardButton.Yes,
    ):
        dlg.delete_intervention()
    assert key not in subject["interventions"]
