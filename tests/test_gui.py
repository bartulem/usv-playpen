import pytest

from importlib import metadata
from PyQt6.QtCore import Qt  # Added import

from usv_playpen import usv_playpen_gui

@pytest.fixture
def app(qtbot):
    app_obj, window = usv_playpen_gui.initialize_main_window(no_splash=True)
    qtbot.addWidget(window)
    return window

def test_main_window_initialization(app):
    assert app.windowTitle() == f"USV Playpen v{metadata.version('usv-playpen').split('.dev')[0]}"
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
