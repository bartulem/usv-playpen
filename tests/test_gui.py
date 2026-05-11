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

def test_main_window_initialization(app):
    expected_version = metadata.version('usv-playpen').split('.dev')[0].split('.post')[0]
    assert app.windowTitle() == f"USV Playpen v{expected_version}"
    assert app.exp_id_cb.currentText() == "Bartul"

def test_navigation_to_record_window(app, qtbot):
    qtbot.mouseClick(app.button_map['Record'], Qt.MouseButton.LeftButton)
    assert "Record > Select config" in app.windowTitle()
    assert app.recorder_settings_edit is not None

# python
def test_record_one_settings_are_saved(app, qtbot):
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
