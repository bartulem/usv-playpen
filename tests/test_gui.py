"""
@author: bartulem
Test GUI.
"""

import pytest
from PyQt6.QtWidgets import QApplication
from unittest.mock import patch, MagicMock
from usv_playpen.usv_playpen_gui import USVPlaypenWindow

@pytest.fixture
def mock_configs(mocker):
    """Mocks the loading of all configuration files."""

    mocker.patch('toml.load', return_value={
        'video': {
            'metadata': {'experimenter': 'default_user'},
            'general': {'available_cameras': [], 'monitor_recording': False, 'monitor_specific_camera': False, 'delete_post_copy': True, 'recording_codec': 'hq', 'specific_camera_serial': '', 'calibration_frame_rate': 10, 'recording_frame_rate': 150},
            'cameras_config': {}
        },
        'conduct_audio_recording': True, 'conduct_tracking_calibration': False, 'disable_ethernet': True,
        'experimenter_list': ['default_user', 'another_user'],
        'recording_files_destinations_all': [], 'recording_files_destination_win': [], 'recording_files_destination_linux': []
    })
    # Mock json.load to return predictable dictionaries
    mocker.patch('json.load', side_effect=[
        {"visualize_data": {"root_directories": []}, "make_behavioral_videos": {"arena_directory": "", "speaker_audio_file": ""}},  # Mock for visualizations_settings.json
        {"analyze_data": {"root_directories": []}},  # Mock for analyses_settings.json
        {"preprocess_data": {"root_directories": []}},  # Mock for processing_settings.json
    ])

@pytest.fixture
def mock_backend_classes(mocker):
    """Mocks the backend classes (ExperimentController, Analyst, etc.) that the GUI calls."""

    mocked_classes = {
        'ExperimentController': mocker.patch('usv_playpen.usv_playpen_gui.ExperimentController'),
        'Stylist': mocker.patch('usv_playpen.usv_playpen_gui.Stylist'),
        'Analyst': mocker.patch('usv_playpen.usv_playpen_gui.Analyst'),
        'Visualizer': mocker.patch('usv_playpen.usv_playpen_gui.Visualizer'),
    }
    return mocked_classes

@pytest.fixture
def app(qtbot, mock_configs, mock_backend_classes):
    """
    Creates an instance of the main GUI window for testing.
    The `qtbot` fixture is provided by pytest-qt.
    """

    # the splash screen can cause issues in a test environment, so we mock it.
    with patch('usv_playpen.usv_playpen_gui.QSplashScreen'):
        test_app = USVPlaypenWindow()
        qtbot.addWidget(test_app)
        return test_app

def test_main_window_initialization(app):
    """
    Tests if the main window initializes correctly.
    """

    assert app.windowTitle() == "USV Playpen v0.1.0"
    assert app.exp_id_cb.currentText() == "default_user"


def test_navigation_to_record_window(app, qtbot):
    """
    Tests the button click to navigate from the main menu to the first record screen.
    """

    # simulate a click on the 'Record' button
    qtbot.mouseClick(app.button_map['Record'], 'left')

    # after clicking, the central widget should be an instance of the Record class
    # and the window title should have changed.
    assert "Record > Select config" in app.windowTitle()
    # check if a widget from the new screen exists to confirm navigation
    assert app.recorder_settings_edit is not None


def test_record_one_settings_are_saved(app, qtbot):
    """
    Tests that user input on the 'record_one' screen correctly updates the settings dictionary
    when the 'Next' button is clicked.
    """

    # navigate to the record screen
    qtbot.mouseClick(app.button_map['Record'], 'left')

    # simulate user input
    new_duration = "25.5"
    qtbot.keyClicks(app.video_session_duration, new_duration)

    # check that the dict is NOT updated yet
    assert app.exp_settings_dict['video_session_duration'] != 25.5

    # click the 'Next' button to trigger the save function
    qtbot.mouseClick(app.button_map['Next'], 'left')

    # assert that the internal settings dictionary has been updated
    assert app.exp_settings_dict['video_session_duration'] == 25.5
    assert "Record > Audio Settings" in app.windowTitle()  # Also check we navigated


def test_start_recording_button_calls_backend(app, qtbot, mock_backend_classes):
    """
    Tests that clicking the final 'Record' button correctly initializes and calls
    the `ExperimentController` backend class.
    """

    # navigate through the GUI to the final recording screen
    qtbot.mouseClick(app.button_map['Record'], 'left')  # Main -> Record 1
    qtbot.mouseClick(app.button_map['Next'], 'left')  # Record 1 -> Record 2 (Audio)
    qtbot.mouseClick(app.button_map['Next'], 'left')  # Record 2 -> Record 3 (Video)
    qtbot.mouseClick(app.button_map['Next'], 'left')  # Record 3 -> Record 4 (Conduct)

    # at this point, the "Record" button should be visible.
    record_button = app.button_map['Record']

    # simulate a click
    qtbot.mouseClick(record_button, 'left')

    # assertions
    mock_controller = mock_backend_classes['ExperimentController']

    # was the ExperimentController class initialized?
    mock_controller.assert_called_once()

    # was the `conduct_behavioral_recording` method called on the instance?
    mock_controller.return_value.conduct_behavioral_recording.assert_called_once()


def test_start_processing_button_calls_backend(app, qtbot, mock_backend_classes):
    """
    Tests that clicking the final 'Process' button correctly initializes and calls
    the `Stylist` backend class.
    """

    # navigate to the processing screen
    qtbot.mouseClick(app.button_map['Process'], 'left')  # Main -> Process 1

    # simulate user input (e.g., adding a directory to process)
    qtbot.keyClicks(app.processing_dir_edit, "/my/test/directory")

    # navigate to the final screen
    qtbot.mouseClick(app.button_map['Next'], 'left')  # Process 1 -> Process 2 (Conduct)

    # click the 'Process' button
    process_button = app.button_map['Process']
    qtbot.mouseClick(process_button, 'left')

    # assertions
    mock_stylist = mock_backend_classes['Stylist']

    # was the Stylist class initialized?
    mock_stylist.assert_called_once()

    # check if it was initialized with the correct directory from user input
    # `call_args.kwargs` is a dictionary of the keyword arguments passed to the mock
    init_kwargs = mock_stylist.call_args.kwargs
    assert init_kwargs['root_directories'] == ["/my/test/directory"]

    # was the main processing method called?
    mock_stylist.return_value.prepare_data_for_analyses.assert_called_once()
