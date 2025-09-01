"""
@author: bartulem
Test processing module.
"""

import pytest
import json
import shutil
import subprocess
from unittest.mock import MagicMock, patch
from usv_playpen.process_data import Stylist

FFMPEG_INSTALLED = shutil.which("ffmpeg") is not None
SOX_INSTALLED = shutil.which("static_sox") is not None

@pytest.fixture
def mock_settings():
    """Provides a mocked processing_settings dictionary for tests."""

    settings = {
        "processing_booleans": {f"conduct_{i}": False for i in [
            "video_concatenation", "video_fps_change", "audio_multichannel_to_single_ch",
            "audio_cropping", "audio_to_mmap", "audio_filtering", "hpss",
            "audio_video_sync", "ephys_video_sync", "ephys_file_chaining",
            "split_cluster_spikes", "prepare_sleap_cluster", "sleap_h5_conversion",
            "anipose_calibration", "anipose_triangulation", "anipose_trm",
            "das_infer", "das_summarize", "prepare_assign_vocalizations", "assign_vocalizations"
        ]},
        "credentials_directory": "/fake/credentials",
        "send_email": {"Messenger": {"send_message": {"receivers": []}, "processing_pc_choice": "Test PC", "experimenter": "Tester"}},
        "anipose_operations": {"ConvertTo3D": {"conduct_anipose_triangulation": {"triangulate_arena_points_bool": False}, "translate_rotate_metric": {"experimental_codes": []}}},
        "vocalocator": {"vcl_version": "vcl-ssl"}
    }
    return settings

@pytest.fixture
def mock_dependencies(mocker):
    """Mocks all external class dependencies for the Stylist class."""
    mocked_classes = {
        'ConvertTo3D': mocker.patch('usv_playpen.process_data.ConvertTo3D'),
        'Vocalocator': mocker.patch('usv_playpen.process_data.Vocalocator'),
        'FindMouseVocalizations': mocker.patch('usv_playpen.process_data.FindMouseVocalizations'),
        'Gatherer': mocker.patch('usv_playpen.process_data.Gatherer'),
        'Operator': mocker.patch('usv_playpen.process_data.Operator'),
        'PrepareClusterJob': mocker.patch('usv_playpen.process_data.PrepareClusterJob'),
        'SummaryPlotter': mocker.patch('usv_playpen.process_data.SummaryPlotter'),
        'Messenger': mocker.patch('usv_playpen.process_data.Messenger'),
        'Synchronizer': mocker.patch('usv_playpen.process_data.Synchronizer'),
    }
    mocked_classes['Gatherer'].return_value.prepare_data_for_analyses.return_value = {}
    mocked_classes['Synchronizer'].return_value.find_audio_sync_trains.return_value = {}
    return mocked_classes

@pytest.mark.skipif(not FFMPEG_INSTALLED, reason="ffmpeg executable not found in PATH")
def test_environment_ffmpeg_is_functional():
    """Checks if the ffmpeg command runs successfully."""
    assert subprocess.Popen(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).wait() == 0, \
        "FFMPEG failed to execute. Check installation."

@pytest.mark.skipif(not SOX_INSTALLED, reason="static_sox executable not found in PATH")
def test_environment_sox_is_functional():
    assert subprocess.Popen(["static_sox", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).wait() == 0, \
        "SOX failed to execute. Check installation."

def test_prepare_data_no_booleans_true(mock_settings, mock_dependencies):
    """
    Tests that if all boolean flags are False, no processing methods are called.
    """
    stylist = Stylist(
        input_parameter_dict=mock_settings,
        root_directories=['/fake/dir1']
    )
    stylist.prepare_data_for_analyses()

    assert mock_dependencies['Messenger'].return_value.send_message.call_count == 2

    for name, mock_class in mock_dependencies.items():
        if name != 'Messenger':
            assert mock_class.call_count == 0


def test_single_directory_video_concatenation(mock_settings, mock_dependencies):
    """
    Tests that `Operator.concatenate_video_files` is called when the flag is True.
    """
    mock_settings['processing_booleans']['conduct_video_concatenation'] = True

    stylist = Stylist(
        input_parameter_dict=mock_settings,
        root_directories=['/fake/dir1']
    )
    stylist.prepare_data_for_analyses()

    mock_operator = mock_dependencies['Operator']
    assert mock_operator.call_count == 1
    mock_operator.return_value.concatenate_video_files.assert_called_once()
    mock_operator.return_value.rectify_video_fps.assert_not_called()


def test_multiple_directory_looping(mock_settings, mock_dependencies):
    """
    Tests that per-directory tasks initialize the worker class for each directory.
    """
    mock_settings['processing_booleans']['anipose_calibration'] = True
    root_dirs = ['/fake/dir1', '/fake/dir2']

    stylist = Stylist(
        input_parameter_dict=mock_settings,
        root_directories=root_dirs
    )
    stylist.prepare_data_for_analyses()

    mock_converter = mock_dependencies['ConvertTo3D']
    assert mock_converter.call_count == len(root_dirs)
    assert mock_converter.return_value.conduct_anipose_calibration.call_count == len(root_dirs)


def test_audio_video_sync_chain(mock_settings, mock_dependencies):
    """
    Tests the logic for `conduct_audio_video_sync`, which involves multiple classes.
    """
    mock_settings['processing_booleans']['conduct_audio_video_sync'] = True

    stylist = Stylist(
        input_parameter_dict=mock_settings,
        root_directories=['/fake/dir1']
    )
    stylist.prepare_data_for_analyses()

    mock_gatherer = mock_dependencies['Gatherer']
    mock_synchronizer = mock_dependencies['Synchronizer']
    mock_plotter = mock_dependencies['SummaryPlotter']

    mock_gatherer.return_value.prepare_data_for_analyses.assert_called_once()
    mock_synchronizer.return_value.find_audio_sync_trains.assert_called_once()
    mock_plotter.return_value.preprocessing_summary.assert_called_once()


def test_ephys_chaining_logic(mock_settings, mock_dependencies):
    """
    Tests the special "all directories at once" logic for e-phys file chaining.
    """
    mock_settings['processing_booleans']['conduct_ephys_file_chaining'] = True
    root_dirs = ['/fake/dir1', '/fake/dir2']  # Multiple directories

    stylist = Stylist(
        input_parameter_dict=mock_settings,
        root_directories=root_dirs
    )
    stylist.prepare_data_for_analyses()

    # Operator should be initialized ONLY ONCE with the full list of directories
    mock_operator = mock_dependencies['Operator']
    assert mock_operator.call_count == 1

    # check that it was initialized with the list, not a single string
    init_kwargs = mock_operator.call_args.kwargs
    assert init_kwargs['root_directory'] == root_dirs

    mock_operator.return_value.concatenate_binary_files.assert_called_once()


def test_vocalocator_version_routing(mock_settings, mock_dependencies):
    """
    Tests that the correct Vocalocator method is called based on the 'vcl_version' setting.
    """
    mock_settings['processing_booleans']['assign_vocalizations'] = True

    # case 1: Test the 'vcl-ssl' path (the default in your settings)
    mock_settings['vocalocator']['vcl_version'] = 'vcl-ssl'
    stylist_ssl = Stylist(input_parameter_dict=mock_settings, root_directories=['/fake/dir1'])
    stylist_ssl.prepare_data_for_analyses()

    mock_vocalocator = mock_dependencies['Vocalocator']
    mock_vocalocator.return_value.run_vocalocator_ssl.assert_called_once()
    mock_vocalocator.return_value.run_vocalocator.assert_not_called()

    # reset mock for the next case
    mock_vocalocator.reset_mock()

    # case 2: Test the 'vcl' path
    mock_settings['vocalocator']['vcl_version'] = 'vcl'
    stylist_vcl = Stylist(input_parameter_dict=mock_settings, root_directories=['/fake/dir1'])
    stylist_vcl.prepare_data_for_analyses()

    mock_vocalocator.return_value.run_vocalocator.assert_called_once()
    mock_vocalocator.return_value.run_vocalocator_ssl.assert_not_called()
