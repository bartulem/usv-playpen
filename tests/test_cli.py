"""
@author: bartulem
Test CLI.
"""

import pytest
from click.testing import CliRunner
from unittest.mock import patch
from usv_playpen.analyze_data import (
    generate_beh_features_cli,
    generate_usv_playback_cli
)
from usv_playpen.process_data import (
    concatenate_video_files_cli,
    rectify_video_fps_cli,
    das_infer_cli,
    concatenate_ephys_files_cli
)
from usv_playpen.visualize_data import visualize_3D_data_cli

@pytest.fixture
def runner():
    """Provides a CliRunner instance for invoking commands."""
    return CliRunner()


def test_generate_beh_features_cli_success(runner, mocker):
    """
    Tests that the 'generate-beh-features' command successfully calls the
    FeatureZoo backend with correctly parsed arguments.
    """

    # mock the backend class that the CLI is supposed to call
    mock_feature_zoo = mocker.patch('usv_playpen.analyze_data.FeatureZoo')

    # mock the settings loader to avoid file dependencies
    mocker.patch(
        'usv_playpen.cli_utils.modify_settings_json_for_cli',
        return_value={'compute_behavioral_features': {'head_points': ['Nose', 'Ear_L', 'Ear_R', 'Head']}}
    )

    # use the runner to invoke the command with arguments
    result = runner.invoke(generate_beh_features_cli, [
        '--root-directory', '/fake/path',
        '--head-points', 'Nose', 'Ear_L', 'Ear_R', 'Head'
    ])

    # assert that the command exited successfully (exit code 0)
    assert result.exit_code == 0
    assert result.exception is None

    # assert that the backend class was initialized correctly
    mock_feature_zoo.assert_called_once()

    # check that the main method on the instance was called
    mock_feature_zoo.return_value.save_behavioral_features_to_file.assert_called_once()


def test_generate_usv_playback_cli_success(runner, mocker):
    """
    Tests the 'generate-usv-playback' command with a required option.
    """

    # mock the backend class
    mock_audio_generator = mocker.patch('usv_playpen.analyze_data.AudioGenerator')

    # mock the settings loader
    mocker.patch(
        'usv_playpen.cli_utils.modify_settings_json_for_cli',
        return_value={'create_usv_playback_wav': {'some_setting': 'default'}}
    )

    # invoke the command
    result = runner.invoke(generate_usv_playback_cli, ['--exp-id', 'TestExp'])

    # assert success
    assert result.exit_code == 0
    assert result.exception is None

    # assert that AudioGenerator was initialized with the correct 'exp_id'
    mock_audio_generator.assert_called_once()
    init_kwargs = mock_audio_generator.call_args.kwargs
    assert init_kwargs['exp_id'] == 'TestExp'

    # assert the correct method was called
    mock_audio_generator.return_value.create_usv_playback_wav.assert_called_once()


def test_cli_fails_with_missing_required_option(runner):
    """
    Tests that a command fails correctly if a required option is missing.
    """
    # invoke the command WITHOUT the required '--root-directory'
    result = runner.invoke(generate_beh_features_cli, [])

    # assert that the command failed (non-zero exit code)
    assert result.exit_code != 0

    # assert that the error message tells the user what's missing
    assert "Missing option '--root-directory'" in result.output


def test_visualize_3d_data_cli_success(runner, mocker):
    """
    Tests that the 'generate-viz' command successfully calls the
    Create3DVideo backend with correctly parsed arguments.
    """

    # mock the backend class that the CLI is supposed to call
    mock_create_3d_video = mocker.patch('usv_playpen.visualize_data.Create3DVideo')

    # mock the settings helper to isolate the CLI logic
    # have it return a dictionary containing the animate_bool flag
    mocker.patch(
        'usv_playpen.visualize_data.modify_settings_json_for_cli',
        return_value={'make_behavioral_videos': {'animate_bool': True}}
    )

    # use the runner to invoke the command with arguments, including a flag
    result = runner.invoke(visualize_3D_data_cli, [
        '--root-directory', '/fake/session',
        '--arena-directory', '/fake/arena',
        '--exp-id', 'TestExp',
        '--animate'
    ])

    # assert that the command ran successfully
    assert result.exit_code == 0
    assert result.exception is None

    # assert that the backend class was initialized with the correct arguments
    mock_create_3d_video.assert_called_once()

    init_kwargs = mock_create_3d_video.call_args.kwargs
    assert init_kwargs['root_directory'] == '/fake/session'
    assert init_kwargs['arena_directory'] == '/fake/arena'
    assert init_kwargs['exp_id'] == 'TestExp'

    # check that the boolean flag was correctly passed inside the settings dict
    assert init_kwargs['visualizations_parameter_dict']['make_behavioral_videos']['animate_bool'] is True

    # assert that the main method on the instance was called
    mock_create_3d_video.return_value.visualize_in_video.assert_called_once()

def test_concatenate_video_cli_success(runner, mocker):
    """
    Tests the 'concatenate-video-files' command with required and multiple options.
    """

    # mock the backend Operator class and the settings helper
    mock_operator = mocker.patch('usv_playpen.process_data.Operator')
    mocker.patch(
        'usv_playpen.process_data.modify_settings_json_for_cli',
        return_value={'modify_files': {'Operator': {'concatenate_video_files': {
            'concatenate_camera_serial_num': ['SN123', 'SN456']
        }}}}
    )

    # use the runner to invoke the command
    result = runner.invoke(concatenate_video_files_cli, [
        '--root-directory', '/fake/path',
        '--camera-serial', 'SN123',
        '--camera-serial', 'SN456',
        '--output-name', 'test_video'
    ])

    # assert that the command exited successfully
    assert result.exit_code == 0
    assert result.exception is None

    # assert that the backend Operator class was initialized correctly
    mock_operator.assert_called_once()
    init_kwargs = mock_operator.call_args.kwargs
    assert init_kwargs['root_directory'] == '/fake/path'

    # assert that the main method on the instance was called
    mock_operator.return_value.concatenate_video_files.assert_called_once()


def test_rectify_video_cli_with_flag(runner, mocker):
    """
    Tests the 'rectify-video-fps' command with a boolean flag.
    """
    mock_operator = mocker.patch('usv_playpen.process_data.Operator')
    mocker.patch(
        'usv_playpen.process_data.modify_settings_json_for_cli',
        return_value={'modify_files': {'Operator': {'rectify_video_fps': {}}}}
    )

    # invoke the command with the '--delete-old-file' flag
    result = runner.invoke(rectify_video_fps_cli, [
        '--root-directory', '/fake/path',
        '--delete-old-file'
    ])

    assert result.exit_code == 0
    assert result.exception is None

    # check that the `rectify_video_fps` method was called
    mock_operator.return_value.rectify_video_fps.assert_called_once()


def test_das_infer_cli_success(runner, mocker):
    """
    Tests the 'das-infer' command.
    """
    mock_das = mocker.patch('usv_playpen.process_data.FindMouseVocalizations')
    mocker.patch(
        'usv_playpen.process_data.modify_settings_json_for_cli',
        return_value={'usv_inference': {'FindMouseVocalizations': {}}}
    )

    result = runner.invoke(das_infer_cli, ['--root-directory', '/fake/path'])

    assert result.exit_code == 0
    assert result.exception is None

    # assert that the backend class was initialized with the correct root directory
    mock_das.assert_called_once()
    init_kwargs = mock_das.call_args.kwargs
    assert init_kwargs['root_directory'] == '/fake/path'

    mock_das.return_value.das_command_line_inference.assert_called_once()


def test_concatenate_ephys_cli_parses_list(runner, mocker):
    """
    Tests that the 'concatenate-ephys-files' command correctly parses a
    comma-separated string into a list of directories.
    """
    mock_operator = mocker.patch('usv_playpen.process_data.Operator')

    # mock pathlib.Path.is_dir to always return True to simplify the test
    mocker.patch('pathlib.Path.is_dir', return_value=True)

    # comma-separated list of paths
    dirs_string = '/fake/dir1, /fake/dir2, /fake/dir3'

    result = runner.invoke(concatenate_ephys_files_cli, ['--root-directories', dirs_string])

    assert result.exit_code == 0
    assert result.exception is None

    # check that Operator was initialized with a correctly parsed list
    mock_operator.assert_called_once()
    init_kwargs = mock_operator.call_args.kwargs

    # the CLI function should strip whitespace and split by comma
    expected_list = ['/fake/dir1', '/fake/dir2', '/fake/dir3']
    assert init_kwargs['root_directory'] == expected_list

    mock_operator.return_value.concatenate_binary_files.assert_called_once()


def test_cli_fails_with_missing_required_directory(runner):
    """
    Tests that a command fails correctly if a required directory path is missing.
    """
    # invoke a command that requires '--root-directory' without providing it
    result = runner.invoke(concatenate_video_files_cli, [])

    # assert that the command failed (non-zero exit code)
    assert result.exit_code != 0

    # assert that the error message tells the user what's missing
    assert "Missing option '--root-directory'" in result.output
