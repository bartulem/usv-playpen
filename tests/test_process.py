"""
@author: bartulem
Test processing module.
"""

import pytest
import json
import shutil
import subprocess
import numpy as np
from unittest.mock import MagicMock, patch
from usv_playpen.preprocess_data import Stylist
from usv_playpen.synchronize_files import Synchronizer

FFMPEG_INSTALLED = shutil.which("ffmpeg") is not None
SOX_INSTALLED = shutil.which("static_sox") is not None


@pytest.fixture
def mock_dependencies(mocker):
    """Mocks all external class dependencies for the Stylist class."""

    mocked_classes = {
        'ConvertTo3D': mocker.patch('usv_playpen.preprocess_data.ConvertTo3D'),
        'Vocalocator': mocker.patch('usv_playpen.preprocess_data.Vocalocator'),
        'FindMouseVocalizations': mocker.patch('usv_playpen.preprocess_data.FindMouseVocalizations'),
        'Gatherer': mocker.patch('usv_playpen.preprocess_data.Gatherer'),
        'Operator': mocker.patch('usv_playpen.preprocess_data.Operator'),
        'PrepareClusterJob': mocker.patch('usv_playpen.preprocess_data.PrepareClusterJob'),
        'SummaryPlotter': mocker.patch('usv_playpen.preprocess_data.SummaryPlotter'),
        'Messenger': mocker.patch('usv_playpen.preprocess_data.Messenger'),
        'Synchronizer': mocker.patch('usv_playpen.preprocess_data.Synchronizer'),
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


@pytest.fixture
def processing_settings(tmp_path):
    """Loads processing_settings.json from the usv_playpen/_parameter_settings directory using pathlib."""
    from pathlib import Path
    import json
    import usv_playpen
    package_dir = Path(usv_playpen.__file__).parent
    settings_path = package_dir / '_parameter_settings' / 'processing_settings.json'
    with settings_path.open('r') as f:
        settings = json.load(f)
    return settings


def test_single_directory_video_concatenation(processing_settings, mock_dependencies, tmp_path):
    """
    Tests that `Operator.concatenate_video_files` is called when the flag is True.
    """
    processing_settings['processing_booleans']['conduct_video_concatenation'] = True
    stylist = Stylist(
        input_parameter_dict=processing_settings,
        root_directories=[str(tmp_path)]
    )
    stylist.prepare_data_for_analyses()
    mock_operator = mock_dependencies['Operator']
    assert mock_operator.call_count == 1
    mock_operator.return_value.concatenate_video_files.assert_called_once()
    mock_operator.return_value.rectify_video_fps.assert_not_called()


def test_multiple_directory_looping(processing_settings, mock_dependencies, tmp_path):
    """
    Tests that per-directory tasks initialize the worker class for each directory.
    """
    processing_settings['processing_booleans']['anipose_calibration'] = True
    root_dirs = [str(tmp_path), str(tmp_path)]
    stylist = Stylist(
        input_parameter_dict=processing_settings,
        root_directories=root_dirs
    )
    stylist.prepare_data_for_analyses()
    mock_converter = mock_dependencies['ConvertTo3D']
    assert mock_converter.call_count == len(root_dirs)
    assert mock_converter.return_value.conduct_anipose_calibration.call_count == len(root_dirs)


def test_audio_video_sync_chain(processing_settings, mock_dependencies, tmp_path):
    """
    Tests the logic for `conduct_audio_video_sync`, which involves multiple classes.
    """
    processing_settings['processing_booleans']['conduct_audio_video_sync'] = True
    stylist = Stylist(
        input_parameter_dict=processing_settings,
        root_directories=[str(tmp_path)]
    )
    stylist.prepare_data_for_analyses()
    mock_gatherer = mock_dependencies['Gatherer']
    mock_synchronizer = mock_dependencies['Synchronizer']
    mock_plotter = mock_dependencies['SummaryPlotter']
    mock_gatherer.return_value.prepare_data_for_analyses.assert_called_once()
    mock_synchronizer.return_value.find_audio_sync_trains.assert_called_once()
    mock_plotter.return_value.preprocessing_summary.assert_called_once()


def test_ephys_chaining_logic(processing_settings, mock_dependencies, tmp_path):
    """
    Tests the special "all directories at once" logic for e-phys file chaining.
    """
    processing_settings['processing_booleans']['conduct_ephys_file_chaining'] = True
    root_dirs = [str(tmp_path), str(tmp_path)]
    stylist = Stylist(
        input_parameter_dict=processing_settings,
        root_directories=root_dirs
    )
    stylist.prepare_data_for_analyses()
    mock_operator = mock_dependencies['Operator']
    assert mock_operator.call_count == 1
    init_kwargs = mock_operator.call_args.kwargs
    assert init_kwargs['root_directory'] == root_dirs
    mock_operator.return_value.concatenate_binary_files.assert_called_once()


def test_vocalocator_version_routing(processing_settings, mock_dependencies, tmp_path):
    """
    Tests that the correct Vocalocator method is called based on the 'vcl_version' setting.
    """
    processing_settings['processing_booleans']['assign_vocalizations'] = True
    # case 1: Test the 'vcl-ssl' path (the default in your settings)
    processing_settings['vocalocator']['vcl_version'] = 'vcl-ssl'
    stylist_ssl = Stylist(input_parameter_dict=processing_settings, root_directories=[str(tmp_path)])
    stylist_ssl.prepare_data_for_analyses()
    mock_vocalocator = mock_dependencies['Vocalocator']
    mock_vocalocator.return_value.run_vocalocator_ssl.assert_called_once()
    mock_vocalocator.return_value.run_vocalocator.assert_not_called()
    mock_vocalocator.reset_mock()
    # case 2: Test the 'vcl' path
    processing_settings['vocalocator']['vcl_version'] = 'vcl'
    stylist_vcl = Stylist(input_parameter_dict=processing_settings, root_directories=[str(tmp_path)])
    stylist_vcl.prepare_data_for_analyses()
    mock_vocalocator.return_value.run_vocalocator.assert_called_once()
    mock_vocalocator.return_value.run_vocalocator_ssl.assert_not_called()


@pytest.fixture
def mock_sync_settings():
    """Provides a default settings dictionary for Synchronizer tests."""

    return {
        'synchronize_files': {
            'Synchronizer': {
                'find_audio_sync_trains': {
                    'sync_ch_receiving_input': 1,
                    'sync_camera_serial_num': ['CAM123'],
                    'extract_exact_video_frame_times_bool': False,
                    'millisecond_divergence_tolerance': 12
                }
            }
        }
    }


@pytest.fixture
def synchronizer_instance(tmp_path, mock_sync_settings):
    """Creates a Synchronizer instance with a temporary file structure."""

    # create fake directory structure
    root_dir = tmp_path
    (root_dir / "audio" / "cropped_to_video").mkdir(parents=True)
    (root_dir / "video").mkdir(parents=True)
    (root_dir / "sync").mkdir(parents=True)

    # create a fake audio file for DataLoader to find
    (root_dir / "audio" / "cropped_to_video" / "test_ch01.wav").touch()

    # create a fake camera frame count JSON
    frame_count_data = {
        'total_frame_number_least': 1000,
        'total_video_time_least': 10.0,
        'CAM123': (1000, 100.0)  # (frame_count, fps)
    }
    with open(root_dir / "video" / "camera_counts.json", 'w') as f:
        json.dump(frame_count_data, f)

    return Synchronizer(root_directory=str(root_dir), input_parameter_dict=mock_sync_settings)


def test_find_ipi_intervals_static_method():
    """
    Tests the core logic of the `find_ipi_intervals` static method.
    """

    # arrange: create a fake audio signal with sync pulses
    sr = 250000  # 250 kHz sampling rate
    signal = np.zeros(sr, dtype=np.int16)  # 1 second of silence

    # create two pulses:
    # pulse 1: starts at sample 1000, duration 250 samples (1 ms)
    signal[1000:1250] = 1
    # pulse 2: starts at sample 5000, duration 500 samples (2 ms)
    signal[5000:5500] = 1

    # act: Run the method
    ipi_durations_ms, audio_ipi_start_samples = Synchronizer.find_ipi_intervals(
        sound_array=signal,
        audio_sr_rate=sr
    )

    # check the results
    assert np.array_equal(audio_ipi_start_samples, np.array([1250]))
    assert np.allclose(ipi_durations_ms, np.array([((5000-1250)/sr)*1000]))

