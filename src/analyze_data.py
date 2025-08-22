"""
@author: bartulem
Conduct analyses of choice on the data of choice (on the PC of choice).
"""

from datetime import datetime
import traceback
import warnings
from click.core import ParameterSource
from .cli_utils import *
from .send_email import Messenger
from .analyses.compute_behavioral_features import FeatureZoo
from .analyses.compute_behavioral_tuning_curves import NeuronalTuning
from .analyses.generate_audio_files import AudioGenerator

warnings.simplefilter('ignore')


class Analyst:

    def __init__(self, input_parameter_dict: dict = None,
                 root_directories: list = None,
                 message_output: callable = None) -> None:

        """
        Initializes the Analyst class.

        Parameter
        ---------
        root_directories (list)
            Root directories for data; defaults to None.
        input_parameter_dict (dict)
            Analyses parameters; defaults to None.
        message_output (function)
            Defines output messages; defaults to None.

        Returns
        -------
        -------
        """

        if root_directories is None:
            with open((pathlib.Path(__file__).parent / '_parameter_settings/analyses_settings.json'), 'r') as json_file:
                self.root_directories = json.load(json_file)['analyze_data']['root_directories']
        else:
            self.root_directories = root_directories

        if input_parameter_dict is None:
            with open((pathlib.Path(__file__).parent / '_parameter_settings/analyses_settings.json'), 'r') as json_file:
                self.input_parameter_dict = json.load(json_file)
        else:
            self.input_parameter_dict = input_parameter_dict

        if message_output is None:
            self.message_output = print
        else:
            self.message_output = message_output

    def analyze_data(self) -> None:
        """
        Description
        ----------
        This method performs the following analyses:
        (1) computes behavioral features and plots their distributions
        (2) computes behavioral tuning curves
        (3) generates playback WAV files
        (4) frequency shifts audio segments
        ----------

        Parameters
        ----------
        ----------

        Returns
        ----------
        ----------
        """

        Messenger(message_output=self.message_output,
                  receivers=self.input_parameter_dict['send_email']['send_message']['receivers'],
                  exp_settings_dict=None).send_message(subject=f"{self.input_parameter_dict['send_email']['analyses_pc_choice']} PC is busy, do NOT attempt to remote in!",
                                                       message=f"Data analyses in progress, started at "
                                                               f"{datetime.now().hour:02d}:{datetime.now().minute:02d}.{datetime.now().second:02d} "
                                                               f"and run by @{self.input_parameter_dict['send_email']['experimenter']}. "
                                                               f"You will be notified upon completion. \n \n ***This is an automatic e-mail, please do NOT respond.***")

        # # # create USV playback WAV files
        if self.input_parameter_dict['analyses_booleans']['create_usv_playback_wav_bool'] or self.input_parameter_dict['analyses_booleans']['create_naturalistic_usv_playback_wav_bool']:
            if self.input_parameter_dict['analyses_booleans']['create_usv_playback_wav_bool']:
                AudioGenerator(exp_id=self.input_parameter_dict['send_email']['experimenter'],
                               create_playback_settings_dict=self.input_parameter_dict['create_usv_playback_wav'],
                               message_output=self.message_output).create_usv_playback_wav()
            else:
                AudioGenerator(exp_id=self.input_parameter_dict['send_email']['experimenter'],
                               create_playback_settings_dict=self.input_parameter_dict['create_naturalistic_usv_playback_wav'],
                               message_output=self.message_output).create_naturalistic_usv_playback_wav()

        for one_directory in self.root_directories:
            try:
                self.message_output(f"Analyzing data in {one_directory} started at: "
                                    f"{datetime.now().hour:02d}:{datetime.now().minute:02d}.{datetime.now().second:02d}.")

                # # # compute behavioral features and plot their distributions
                if self.input_parameter_dict['analyses_booleans']['compute_behavioral_features_bool']:
                    FeatureZoo(root_directory=one_directory,
                               behavioral_parameters_dict=self.input_parameter_dict['compute_behavioral_features'],
                               message_output=self.message_output).save_behavioral_features_to_file()

                # # # compute behavioral tuning curves
                if self.input_parameter_dict['analyses_booleans']['compute_behavioral_tuning_bool']:
                    NeuronalTuning(root_directory=one_directory,
                                   tuning_parameters_dict=self.input_parameter_dict['calculate_neuronal_tuning_curves'],
                                   message_output=self.message_output).calculate_neuronal_tuning_curves()

                # # # frequency shift audio segments
                if self.input_parameter_dict['analyses_booleans']['frequency_shift_audio_segment_bool']:
                    AudioGenerator(root_directory=one_directory,
                                   freq_shift_settings_dict=self.input_parameter_dict['frequency_shift_audio_segment'],
                                   message_output=self.message_output).frequency_shift_audio_segment()

                self.message_output(f"Analyzing data in {one_directory} finished at: "
                                    f"{datetime.now().hour:02d}:{datetime.now().minute:02d}.{datetime.now().second:02d}.")

            except (OSError, RuntimeError, TypeError, IndexError, IOError, EOFError, TimeoutError, NameError, KeyError, ValueError, AttributeError):
                self.message_output(traceback.format_exc())

        Messenger(message_output=self.message_output,
                  no_receivers_notification=False,
                  receivers=self.input_parameter_dict['send_email']['send_message']['receivers'],
                  exp_settings_dict=None).send_message(subject=f"{self.input_parameter_dict['send_email']['analyses_pc_choice']} PC is available again, analyses have been completed",
                                                       message=f"Data analyses have been completed at "
                                                               f"{datetime.now().hour:02d}:{datetime.now().minute:02d}.{datetime.now().second:02d} "
                                                               f"by @{self.input_parameter_dict['send_email']['experimenter']}. "
                                                               f"You will be notified about further PC usage "
                                                               f"should it occur. \n \n ***This is an automatic e-mail, please do NOT respond.***")


@click.command(name='generate-usv-playback')
@click.option('--exp-id', 'experimenter', type=str, required=True, help='Experimenter ID.')
@click.option('--num-usv-files', 'num_usv_files', type=int, default=None, required=False, help='Number of WAV files to create.')
@click.option('--total-usv-number', 'total_usv_number', type=int, default=None, required=False, help='Total number of USVs to distribute across files.')
@click.option('--ipi-duration', 'ipi_duration', type=float, default=None, required=False, help='Inter-USV-interval duration (in s).')
@click.option('--wav-sampling-rate', 'wav_sampling_rate', type=int, default=None, required=False, help='Sampling rate for the output WAV file (in kHz).')
@click.option('--playback-snippets-dir', 'playback_snippets_dir', type=str, default=None, required=False, help='Directory of USV playback snippets.')
@click.pass_context
def generate_usv_playback_cli(ctx, exp_id, **kwargs) -> None:
    """
    Description
    ----------
    A command-line tool to generate USV playback WAV files.
    ----------

    Parameters
    ----------
    ----------

    Returns
    ----------
    ----------
    """

    provided_params = [key for key in kwargs if ctx.get_parameter_source(key) == ParameterSource.COMMANDLINE]

    if not provided_params:
        AudioGenerator().create_usv_playback_wav()
    else:
        analyses_settings_parameter_dict = modify_settings_json_for_cli(ctx=ctx,
                                                                        provided_params=provided_params,
                                                                        settings_dict='analyses_settings')
        AudioGenerator(exp_id=exp_id,
                       create_playback_settings_dict=analyses_settings_parameter_dict['create_usv_playback_wav'],
                       message_output=print).create_usv_playback_wav()

@click.command(name='generate-naturalistic-usv-playback')
@click.option('--exp-id', 'experimenter', type=str, required=True, help='Experimenter ID.')
@click.option('--num-naturalistic-usv-files', 'num_naturalistic_usv_files', type=int, default=None, required=False, help='Number of WAV files to create.')
@click.option('--naturalistic-wav-sampling-rate', 'naturalistic_wav_sampling_rate', type=int, default=None, required=False, help='Sampling rate for the output WAV file (in kHz).')
@click.option('--naturalistic-playback-snippets-dir-prefix', 'naturalistic_playback_snippets_dir_prefix', type=str, default=None, required=False, help='Prefix of directory of the naturalistic USV playback snippets.')
@click.option('--total-playback-time', 'total_acceptable_naturalistic_playback_time', type=int, default=None, required=False, help='Maximum amount of seconds for the duration of the naturalistic playback file.')
@click.option('--inter-seq-interval-dist', 'inter_seq_interval_distribution', type=str, default=None, help='JSON string for inter-sequence interval distribution.')
@click.option('--usv-seq-length-dist', 'usv_seq_length_distribution', type=str, default=None, help='JSON string for USV sequence length distribution.')
@click.option('--inter-usv-interval-dist', 'inter_usv_interval_distribution', type=str, default=None, help='JSON string for inter-USV interval distribution.')
@click.pass_context
def generate_naturalistic_usv_playback_cli(ctx, exp_id, **kwargs) -> None:
    """
    Description
    ----------
    A command-line tool to generate USV playback WAV files.
    ----------

    Parameters
    ----------
    ----------

    Returns
    ----------
    ----------
    """

    for key, value in kwargs.items():
        if isinstance(value, str) and value.strip().startswith('{'):
            try:
                ctx.params[key] = json.loads(value)
            except json.JSONDecodeError:
                # raise an error if the JSON is invalid
                raise click.BadParameter(message=f"Option '--{key.replace('_', '-')}' has invalid JSON.", param_hint=key)

    provided_params = [key for key in kwargs if ctx.get_parameter_source(key) == ParameterSource.COMMANDLINE]

    if not provided_params:
        AudioGenerator().create_usv_playback_wav()
    else:
        analyses_settings_parameter_dict = modify_settings_json_for_cli(ctx=ctx,
                                                                        provided_params=provided_params,
                                                                        settings_dict='analyses_settings')
        AudioGenerator(exp_id=exp_id,
                       create_playback_settings_dict=analyses_settings_parameter_dict['create_naturalistic_usv_playback_wav'],
                       message_output=print).create_naturalistic_usv_playback_wav()

@click.command(name='generate-rm')
@click.option('--root-directory', type=click.Path(exists=True, file_okay=False, dir_okay=True), default=None, required=True, help='Session root directory path.')
@click.option('--temporal-offsets', 'temporal_offsets', multiple=True, type=int, default=None, required=False, help='Spike-behavior offsets to consider (in s).')
@click.option('--n-shuffles', 'n_shuffles', type=int, default=None, required=False, help='Number of shuffles.')
@click.option('--total-bin-num', 'total_bin_num', type=int, default=None, required=False, help='Total number of bins for 1D tuning curves.')
@click.option('--n-spatial-bins', 'n_spatial_bins', type=int, default=None, required=False, help='Number of spatial bins.')
@click.option('--spatial-scale-cm', 'spatial_scale_cm', type=int, default=None, required=False, help='Spatial extent of the arena (in cm).')
@click.pass_context
def generate_rm_files_cli(ctx, root_directory, **kwargs) -> None:
    """
    Description
    ----------
    A command-line tool to calculate behavioral tuning curves.
    ----------

    Parameters
    ----------
    ----------

    Returns
    ----------
    ----------
    """

    provided_params = [key for key in kwargs if ctx.get_parameter_source(key) == ParameterSource.COMMANDLINE]

    analyses_settings_parameter_dict = modify_settings_json_for_cli(ctx=ctx,
                                                                    provided_params=provided_params,
                                                                    settings_dict='analyses_settings')
    NeuronalTuning(root_directory=root_directory,
                   tuning_parameters_dict=analyses_settings_parameter_dict['calculate_neuronal_tuning_curves'],
                   message_output=print).calculate_neuronal_tuning_curves()

@click.command(name='generate-beh-features')
@click.option('--root-directory', type=click.Path(exists=True, file_okay=False, dir_okay=True), default=None, required=True, help='Session root directory path.')
@click.option('--head-points', 'head_points', nargs=4, type=str, default=None, required=False, help='Skeleton head nodes.')
@click.option('--tail-points', 'tail_points',  nargs=5, type=str, default=None, required=False, help='Skeleton tail nodes.')
@click.option('--back-root-points', 'back_root_points', nargs=3, type=str, default=None, required=False, help='Skeleton back nodes.')
@click.option('--derivative-bins', 'derivative_bins', multiple=True, type=str, default=None, required=False, help='Number of bins for derivative calculation.')
@click.pass_context
def generate_beh_features_cli(ctx, root_directory, **kwargs) -> None:
    """
    Description
    ----------
    A command-line tool to compute 3D behavioral features.
    ----------

    Parameters
    ----------
    ----------

    Returns
    ----------
    ----------
    """

    parameters_lists = ['head_points', 'tail_points', 'back_root_points']

    provided_params = [key for key in kwargs if ctx.get_parameter_source(key) == ParameterSource.COMMANDLINE]

    analyses_settings_parameter_dict = modify_settings_json_for_cli(ctx=ctx,
                                                                    parameters_lists=parameters_lists,
                                                                    provided_params=provided_params,
                                                                    settings_dict='analyses_settings')
    FeatureZoo(root_directory=root_directory,
               behavioral_parameters_dict=analyses_settings_parameter_dict['compute_behavioral_features'],
               message_output=print).save_behavioral_features_to_file()
