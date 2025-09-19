"""
@author: bartulem
Test CLI.
"""

import pytest
from click.testing import CliRunner
from unittest.mock import patch
from usv_playpen.analyze_data import (
    generate_beh_features_cli,
    generate_usv_playback_cli,
    generate_rm_files_cli
)
from usv_playpen.preprocess_data import (
    concatenate_video_files_cli,
    rectify_video_fps_cli,
    # das_infer_cli,
    # concatenate_ephys_files_cli,
    crop_wav_files_to_video_cli,
    av_sync_check_cli
)
from usv_playpen.visualize_data import (
    visualize_3D_data_cli,
    generate_rm_figures_cli
)

@pytest.fixture
def runner():
    """Provides a CliRunner instance for invoking commands."""
    return CliRunner()

def test_generate_beh_features_cli_success(runner, mocker, tmp_path):
    """
    Tests the 'generate-beh-features' command successfully calls the
    FeatureZoo backend with correctly parsed arguments.
    """

    mock_feature_zoo = mocker.patch('usv_playpen.analyze_data.FeatureZoo')
    mocker.patch(
        'usv_playpen.cli_utils.modify_settings_json_for_cli',
        return_value={'compute_behavioral_features': {'head_points': ['Nose', 'Ear_L', 'Ear_R', 'Head']}}
    )

    result = runner.invoke(generate_beh_features_cli, [
        '--root-directory', str(tmp_path),
        '--head-points', 'Nose', 'Ear_L', 'Ear_R', 'Head'
    ])

    assert result.exit_code == 0, f"CLI failed: {result.output}"
    assert result.exception is None, f"Exception occurred: {result.output}"
    mock_feature_zoo.assert_called_once()
    mock_feature_zoo.return_value.save_behavioral_features_to_file.assert_called_once()


def test_generate_usv_playback_cli_success(runner, mocker):
    """
    Tests the 'generate-usv-playback' command with a required option.
    """

    mock_audio_generator = mocker.patch('usv_playpen.analyze_data.AudioGenerator')
    mocker.patch(
        'usv_playpen.cli_utils.modify_settings_json_for_cli',
        return_value={'create_usv_playback_wav': {'some_setting': 'default'}}
    )

    result = runner.invoke(generate_usv_playback_cli, ['--exp-id', 'TestExp'])

    assert result.exit_code == 0, f"CLI failed: {result.output}"
    assert result.exception is None, f"Exception occurred: {result.output}"
    mock_audio_generator.assert_called_once()
    init_kwargs = mock_audio_generator.call_args.kwargs
    assert init_kwargs['exp_id'] == 'TestExp'
    mock_audio_generator.return_value.create_usv_playback_wav.assert_called_once()


def test_generate_rm_files_cli_success(runner, mocker, tmp_path):
    """
    Tests the 'generate-rm' (rate map) command.
    """

    mock_tuning = mocker.patch('usv_playpen.analyze_data.NeuronalTuning')
    mocker.patch(
        'usv_playpen.cli_utils.modify_settings_json_for_cli',
        return_value={'calculate_neuronal_tuning_curves': {}}
    )

    result = runner.invoke(generate_rm_files_cli, [
        '--root-directory', str(tmp_path),
        '--n-shuffles', '500'
    ])

    assert result.exit_code == 0, f"CLI failed: {result.output}"
    assert result.exception is None, f"Exception occurred: {result.output}"
    mock_tuning.assert_called_once()
    mock_tuning.return_value.calculate_neuronal_tuning_curves.assert_called_once()

def test_visualize_3d_data_cli_success(runner, mocker, tmp_path):
    """
    Tests that the 'generate-viz' command successfully calls the
    Create3DVideo backend with correctly parsed arguments.
    """

    mock_create_3d_video = mocker.patch('usv_playpen.visualize_data.Create3DVideo')
    mocker.patch(
        'usv_playpen.visualize_data.modify_settings_json_for_cli',
        return_value={'make_behavioral_videos': {'animate_bool': True}}
    )

    result = runner.invoke(visualize_3D_data_cli, [
        '--root-directory', str(tmp_path),
        '--arena-directory', str(tmp_path),
        '--exp-id', 'TestExp',
        '--animate'
    ])

    assert result.exit_code == 0, f"CLI failed: {result.output}"
    assert result.exception is None, f"Exception occurred: {result.output}"
    mock_create_3d_video.assert_called_once()
    init_kwargs = mock_create_3d_video.call_args.kwargs
    assert init_kwargs['root_directory'] == str(tmp_path)
    assert init_kwargs['arena_directory'] == str(tmp_path)
    assert init_kwargs['exp_id'] == 'TestExp'
    assert init_kwargs['visualizations_parameter_dict']['make_behavioral_videos']['animate_bool'] is True
    mock_create_3d_video.return_value.visualize_in_video.assert_called_once()


def test_generate_rm_figures_cli_success(runner, mocker, tmp_path):
    """
    Tests the 'generate-rm-figs' (rate map figures) command.
    """

    mock_ratemap_maker = mocker.patch('usv_playpen.visualize_data.RatemapFigureMaker')
    mocker.patch(
        'usv_playpen.visualize_data.modify_settings_json_for_cli',
        return_value={'neuronal_tuning_figures': {}}
    )

    result = runner.invoke(generate_rm_figures_cli, [
        '--root-directory', str(tmp_path),
        '--smoothing-sd', '1.5'
    ])

    assert result.exit_code == 0, f"CLI failed: {result.output}"
    assert result.exception is None, f"Exception occurred: {result.output}"
    mock_ratemap_maker.assert_called_once()
    mock_ratemap_maker.return_value.neuronal_tuning_figures.assert_called_once()

def test_concatenate_video_cli_success(runner, mocker, tmp_path):
    """
    Tests the 'concatenate-video-files' command with required and multiple options.
    """

    mock_operator = mocker.patch('usv_playpen.preprocess_data.Operator')
    mocker.patch(
        'usv_playpen.preprocess_data.modify_settings_json_for_cli',
        return_value={'modify_files': {'Operator': {}}}
    )

    result = runner.invoke(concatenate_video_files_cli, [
        '--root-directory', str(tmp_path),
        '--camera-serial', 'SN123',
        '--camera-serial', 'SN456',
        '--output-name', 'test_video'
    ])

    assert result.exit_code == 0, f"CLI failed: {result.output}"
    assert result.exception is None, f"Exception occurred: {result.output}"
    mock_operator.assert_called_once_with(
        root_directory=str(tmp_path),
        input_parameter_dict={'modify_files': {'Operator': {}}}
    )
    mock_operator.return_value.concatenate_video_files.assert_called_once()


def test_rectify_video_cli_with_flag(runner, mocker, tmp_path):
    """
    Tests the 'rectify-video-fps' command with a boolean flag.
    """

    mock_operator = mocker.patch('usv_playpen.preprocess_data.Operator')
    mocker.patch(
        'usv_playpen.preprocess_data.modify_settings_json_for_cli',
        return_value={'modify_files': {'Operator': {'rectify_video_fps': {}}}}
    )

    result = runner.invoke(rectify_video_fps_cli, [
        '--root-directory', str(tmp_path),
        '--delete-old-file'
    ])

    assert result.exit_code == 0, f"CLI failed: {result.output}"
    assert result.exception is None, f"Exception occurred: {result.output}"
    mock_operator.return_value.rectify_video_fps.assert_called_once()


def test_crop_wav_cli_success(runner, mocker, tmp_path):
    """
    Tests the 'crop-wav-files' command.
    """

    mock_synchronizer = mocker.patch('usv_playpen.preprocess_data.Synchronizer')
    mocker.patch(
        'usv_playpen.preprocess_data.modify_settings_json_for_cli',
        return_value={'synchronize_files': {'Synchronizer': {}}}
    )

    result = runner.invoke(crop_wav_files_to_video_cli, [
        '--root-directory', str(tmp_path),
        '--trigger-device', 'both'
    ])

    assert result.exit_code == 0, f"CLI failed: {result.output}"
    assert result.exception is None, f"Exception occurred: {result.output}"
    mock_synchronizer.assert_called_once()
    mock_synchronizer.return_value.crop_wav_files_to_video.assert_called_once()

def test_av_sync_check_cli_success(runner, mocker, tmp_path):
    """
    Tests the 'av-sync-check' command which calls multiple backend classes.
    """

    mock_gatherer = mocker.patch('usv_playpen.preprocess_data.Gatherer')
    mock_synchronizer = mocker.patch('usv_playpen.preprocess_data.Synchronizer')
    mock_plotter = mocker.patch('usv_playpen.preprocess_data.SummaryPlotter')
    mocker.patch('usv_playpen.preprocess_data.modify_settings_json_for_cli', return_value={})

    result = runner.invoke(av_sync_check_cli, ['--root-directory', str(tmp_path)])

    assert result.exit_code == 0, f"CLI failed: {result.output}"
    assert result.exception is None, f"Exception occurred: {result.output}"

    # check that all three backend classes were called in order
    mock_gatherer.return_value.prepare_data_for_analyses.assert_called_once()
    mock_synchronizer.return_value.find_audio_sync_trains.assert_called_once()
    mock_plotter.return_value.preprocessing_summary.assert_called_once()

def test_das_infer_cli_success(runner, mocker, tmp_path):
    """
    Tests the 'das-infer' command.
    """

    mock_das = mocker.patch('usv_playpen.preprocess_data.FindMouseVocalizations')
    mocker.patch(
        'usv_playpen.preprocess_data.modify_settings_json_for_cli',
        return_value={'usv_inference': {'FindMouseVocalizations': {}}}
    )

    result = runner.invoke(das_infer_cli, ['--root-directory', str(tmp_path)])

    assert result.exit_code == 0, f"CLI failed: {result.output}"
    assert result.exception is None, f"Exception occurred: {result.output}"
    mock_das.assert_called_once()
    init_kwargs = mock_das.call_args.kwargs
    assert init_kwargs['root_directory'] == str(tmp_path)
    mock_das.return_value.das_command_line_inference.assert_called_once()

def test_concatenate_ephys_cli_parses_list(runner, mocker, tmp_path):
    """
    Tests that the 'concatenate-ephys-files' command correctly parses a
    comma-separated string into a list of directories.
    """

    mock_operator = mocker.patch('usv_playpen.preprocess_data.Operator')
    mocker.patch('pathlib.Path.is_dir', return_value=True)

    dirs_string = f'{str(tmp_path)}, {str(tmp_path)}, {str(tmp_path)}'
    result = runner.invoke(concatenate_ephys_files_cli, ['--root-directories', dirs_string])

    assert result.exit_code == 0, f"CLI failed: {result.output}"
    assert result.exception is None, f"Exception occurred: {result.output}"
    mock_operator.assert_called_once()
    init_kwargs = mock_operator.call_args.kwargs
    expected_list = [str(tmp_path), str(tmp_path), str(tmp_path)]
    assert init_kwargs['root_directory'] == expected_list
    mock_operator.return_value.concatenate_binary_files.assert_called_once()

def test_cli_fails_with_missing_required_directory(runner):
    """
    Tests that a command fails correctly if a required directory path is missing.
    """

    result = runner.invoke(concatenate_video_files_cli, [])

    assert result.exit_code != 0, f"CLI should fail but succeeded: {result.output}"
    assert "Missing option '--root-directory'" in result.output, f"Expected error not found: {result.output}"
