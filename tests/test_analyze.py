"""
@author: bartulem
Test analyses module.
"""

import pytest
from unittest.mock import patch
from usv_playpen.analyze_data import Analyst

@pytest.fixture
def mock_settings():
    """Provides a mocked analyses_settings dictionary for tests."""
    settings = {
        "analyses_booleans": {
            "compute_behavioral_features_bool": False,
            "compute_behavioral_tuning_bool": False,
            "create_usv_playback_wav_bool": False,
            "create_naturalistic_usv_playback_wav_bool": False,
            "frequency_shift_audio_segment_bool": False
        },
        "credentials_directory": "/fake/credentials",
        "send_email": {
            "send_message": {"receivers": []},
            "analyses_pc_choice": "Test PC",
            "experimenter": "Tester"
        },
        "create_usv_playback_wav": {},
        "create_naturalistic_usv_playback_wav": {},
        "compute_behavioral_features": {},
        "calculate_neuronal_tuning_curves": {},
        "frequency_shift_audio_segment": {}
    }
    return settings

@pytest.fixture
def mock_dependencies(mocker):
    """Mocks all external class dependencies for the Analyst class."""
    mocked_classes = {
        'FeatureZoo': mocker.patch('usv_playpen.analyze_data.FeatureZoo'),
        'NeuronalTuning': mocker.patch('usv_playpen.analyze_data.NeuronalTuning'),
        'AudioGenerator': mocker.patch('usv_playpen.analyze_data.AudioGenerator'),
        'Messenger': mocker.patch('usv_playpen.analyze_data.Messenger'),
    }
    return mocked_classes

def test_analyze_data_no_booleans_true(mock_settings, mock_dependencies):
    """
    Tests that if all boolean flags are False, no analysis methods are called.
    """
    analyst = Analyst(
        input_parameter_dict=mock_settings,
        root_directories=['/fake/dir1']
    )
    analyst.analyze_data()

    # check that only the Messenger was called to send start/end emails
    assert mock_dependencies['Messenger'].return_value.send_message.call_count == 2

    # ensure no other analysis classes were even initialized
    for name, mock_class in mock_dependencies.items():
        if name != 'Messenger':
            assert mock_class.call_count == 0


def test_compute_behavioral_features_logic(mock_settings, mock_dependencies):
    """
    Tests that `FeatureZoo.save_behavioral_features_to_file` is called when the flag is True.
    """
    mock_settings['analyses_booleans']['compute_behavioral_features_bool'] = True
    root_dirs = ['/fake/dir1', '/fake/dir2']

    analyst = Analyst(
        input_parameter_dict=mock_settings,
        root_directories=root_dirs
    )
    analyst.analyze_data()

    # check that the FeatureZoo class was initialized for each directory
    mock_feature_zoo = mock_dependencies['FeatureZoo']
    assert mock_feature_zoo.call_count == len(root_dirs)

    # check that the correct method was called for each instance
    mock_feature_zoo.return_value.save_behavioral_features_to_file.assert_called()
    assert mock_feature_zoo.return_value.save_behavioral_features_to_file.call_count == len(root_dirs)


def test_compute_tuning_curves_logic(mock_settings, mock_dependencies):
    """
    Tests that `NeuronalTuning.calculate_neuronal_tuning_curves` is called when the flag is True.
    """
    mock_settings['analyses_booleans']['compute_behavioral_tuning_bool'] = True
    root_dirs = ['/fake/dir1']

    analyst = Analyst(
        input_parameter_dict=mock_settings,
        root_directories=root_dirs
    )
    analyst.analyze_data()

    mock_neuronal_tuning = mock_dependencies['NeuronalTuning']
    assert mock_neuronal_tuning.call_count == len(root_dirs)
    mock_neuronal_tuning.return_value.calculate_neuronal_tuning_curves.assert_called_once()


def test_create_usv_playback_wav_routing(mock_settings, mock_dependencies):
    """
    Tests that the correct AudioGenerator method is called based on which playback flag is set.
    This analysis step is unique because it runs outside the main directory loop.
    """
    mock_settings['analyses_booleans']['create_usv_playback_wav_bool'] = True

    analyst = Analyst(
        input_parameter_dict=mock_settings,
        root_directories=[]
    )
    analyst.analyze_data()

    mock_audio_generator = mock_dependencies['AudioGenerator']

    # It should be called once, and call the standard playback method
    assert mock_audio_generator.call_count == 1
    mock_audio_generator.return_value.create_usv_playback_wav.assert_called_once()
    mock_audio_generator.return_value.create_naturalistic_usv_playback_wav.assert_not_called()


def test_create_naturalistic_usv_playback_wav_routing(mock_settings, mock_dependencies):
    """
    Tests the routing for the naturalistic playback generation.
    """
    # set the 'naturalistic' flag to True and the other playback flag to False
    mock_settings['analyses_booleans']['create_usv_playback_wav_bool'] = False
    mock_settings['analyses_booleans']['create_naturalistic_usv_playback_wav_bool'] = True

    analyst = Analyst(
        input_parameter_dict=mock_settings,
        root_directories=[]
    )
    analyst.analyze_data()

    mock_audio_generator = mock_dependencies['AudioGenerator']

    # it should be called once, and call the naturalistic playback method
    assert mock_audio_generator.call_count == 1
    mock_audio_generator.return_value.create_naturalistic_usv_playback_wav.assert_called_once()
    mock_audio_generator.return_value.create_usv_playback_wav.assert_not_called()


def test_frequency_shift_logic(mock_settings, mock_dependencies):
    """
    Tests that `AudioGenerator.frequency_shift_audio_segment` is called correctly.
    """
    mock_settings['analyses_booleans']['frequency_shift_audio_segment_bool'] = True
    root_dirs = ['/fake/dir1', '/fake/dir2']

    analyst = Analyst(
        input_parameter_dict=mock_settings,
        root_directories=root_dirs
    )
    analyst.analyze_data()

    mock_audio_generator = mock_dependencies['AudioGenerator']

    # should be initialized once for each directory
    assert mock_audio_generator.call_count == len(root_dirs)

    # the method should be called once for each instance
    mock_audio_generator.return_value.frequency_shift_audio_segment.assert_called()
    assert mock_audio_generator.return_value.frequency_shift_audio_segment.call_count == len(root_dirs)