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
from usv_playpen.process_data import Stylist
from usv_playpen.synchronize_files import Synchronizer

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
    assert np.array_equal(audio_ipi_start_samples, np.array([1000, 5000]))
    assert np.allclose(ipi_durations_ms, np.array([1.0, 2.0]))


def test_find_audio_sync_trains_logic(synchronizer_instance, mocker):
    """
    Tests the orchestration logic of the `find_audio_sync_trains` method.
    """

    # mock DataLoader to return our fake audio signal
    mock_dataloader_instance = MagicMock()
    fake_audio_signal = np.zeros(2500000, dtype=np.int16)  # 10 seconds
    fake_audio_signal[250000:250250] = 1  # A 1ms pulse at 1s
    mock_dataloader_instance.load_wavefile_data.return_value = {
        'test_ch01.wav': {'wav_data': fake_audio_signal, 'sampling_rate': 250000}
    }
    mocker.patch('usv_playpen.synchronize_files.DataLoader', return_value=mock_dataloader_instance)

    # mock the internal call to `find_video_sync_trains`
    fake_video_starts = np.array([100])  # 1 pulse found, starting at frame 100
    fake_video_ipi_dict = {'CAM123': np.array([1.0])}  # 1ms duration
    mocker.patch.object(synchronizer_instance, 'find_video_sync_trains', return_value=(fake_video_starts, fake_video_ipi_dict))

    # run the main method
    result_dict = synchronizer_instance.find_audio_sync_trains()

    # the method should find that the audio and video sync trains match.
    # audio pulse at 1s. Video pulse at frame 100 / 100fps = 1s. Discrepancy should be ~0.
    discrepancy_ms = result_dict['test_ch01']['ipi_discrepancy_ms']
    assert np.isclose(discrepancy_ms[0], 0.0, atol=1e-9)
    assert np.array_equal(result_dict['test_ch01']['video_ipi_start_frames'], fake_video_starts)
