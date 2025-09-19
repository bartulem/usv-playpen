"""
@author: bartulem
Test visualizations module.
"""

import pytest
from unittest.mock import patch
from usv_playpen.visualize_data import Visualizer

@pytest.fixture
def mock_settings():
    """Provides a mocked visualizations_settings dictionary for tests."""

    settings = {
        "visualize_booleans": {
            "make_behavioral_tuning_figures_bool": False,
            "make_behavioral_videos_bool": False,
        },
        "credentials_directory": "/fake/credentials",
        "send_email": {
            "send_message": {"receivers": []},
            "visualizations_pc_choice": "Test PC",
            "experimenter": "Tester"
        },
        "neuronal_tuning_figures": {},
        "make_behavioral_videos": {
            "arena_directory": "/fake/arena/dir",
            "speaker_audio_file": "/fake/speaker.wav",
        }
    }
    return settings


@pytest.fixture
def mock_dependencies(mocker):
    """Mocks all external class dependencies for the Visualizer class."""

    mocked_classes = {
        'RatemapFigureMaker': mocker.patch('usv_playpen.visualize_data.RatemapFigureMaker'),
        'Create3DVideo': mocker.patch('usv_playpen.visualize_data.Create3DVideo'),
        'Messenger': mocker.patch('usv_playpen.visualize_data.Messenger'),
    }
    return mocked_classes

def test_visualize_data_no_booleans_true(mock_settings, mock_dependencies):
    """
    Tests that if all boolean flags are False, no visualization methods are called.
    """
    visualizer = Visualizer(
        input_parameter_dict=mock_settings,
        root_directories=['/fake/dir1']
    )
    visualizer.visualize_data()

    # check that only the Messenger was called to send start/end emails
    assert mock_dependencies['Messenger'].return_value.send_message.call_count == 2

    # ensure no other visualization classes were even initialized
    for name, mock_class in mock_dependencies.items():
        if name != 'Messenger':
            assert mock_class.call_count == 0


def test_make_behavioral_tuning_figures_logic(mock_settings, mock_dependencies):
    """
    Tests that `RatemapFigureMaker.neuronal_tuning_figures` is called when the flag is True.
    """

    mock_settings['visualize_booleans']['make_behavioral_tuning_figures_bool'] = True
    root_dirs = ['/fake/dir1', '/fake/dir2']

    # Act: Run the main method
    visualizer = Visualizer(
        input_parameter_dict=mock_settings,
        root_directories=root_dirs
    )
    visualizer.visualize_data()

    # check that the RatemapFigureMaker class was initialized for each directory
    mock_ratemap_maker = mock_dependencies['RatemapFigureMaker']
    assert mock_ratemap_maker.call_count == len(root_dirs)

    # check that the correct method was called for each instance
    mock_ratemap_maker.return_value.neuronal_tuning_figures.assert_called()
    assert mock_ratemap_maker.return_value.neuronal_tuning_figures.call_count == len(root_dirs)


def test_make_behavioral_videos_logic(mock_settings, mock_dependencies):
    """
    Tests that `Create3DVideo.visualize_in_video` is called when the flag is True.
    """
    mock_settings['visualize_booleans']['make_behavioral_videos_bool'] = True
    root_dirs = ['/fake/dir1']

    visualizer = Visualizer(
        input_parameter_dict=mock_settings,
        root_directories=root_dirs
    )
    visualizer.visualize_data()

    mock_video_creator = mock_dependencies['Create3DVideo']
    assert mock_video_creator.call_count == len(root_dirs)
    mock_video_creator.return_value.visualize_in_video.assert_called_once()


def test_multiple_tasks_are_called(mock_settings, mock_dependencies):
    """
    Tests that when multiple boolean flags are True, all corresponding tasks are executed.
    """
    mock_settings['visualize_booleans']['make_behavioral_tuning_figures_bool'] = True
    mock_settings['visualize_booleans']['make_behavioral_videos_bool'] = True
    root_dirs = ['/fake/dir1', '/fake/dir2']

    visualizer = Visualizer(
        input_parameter_dict=mock_settings,
        root_directories=root_dirs
    )
    visualizer.visualize_data()

    # check RatemapFigureMaker calls
    mock_ratemap_maker = mock_dependencies['RatemapFigureMaker']
    assert mock_ratemap_maker.call_count == len(root_dirs)
    assert mock_ratemap_maker.return_value.neuronal_tuning_figures.call_count == len(root_dirs)

    # check Create3DVideo calls
    mock_video_creator = mock_dependencies['Create3DVideo']
    assert mock_video_creator.call_count == len(root_dirs)
    assert mock_video_creator.return_value.visualize_in_video.call_count == len(root_dirs)
