"""
@author: bartulem
Conduct analyses of choice on the data of choice (on the PC of choice).
"""

from __future__ import annotations

import json
import pathlib
import traceback
import warnings
from collections.abc import Callable
from datetime import datetime

import click
from click.core import ParameterSource

from ..cli_utils import modify_settings_json_for_cli
from ..os_utils import configure_path
from ..send_email import Messenger
from .compute_behavioral_features import FeatureZoo
from .compute_inter_usv_interval_distributions import InterUSVIntervalCalculator
from .compute_neuronal_tuning_curves import NeuronalTuning
from .generate_audio_files import AudioGenerator


class Analyst:

    def __init__(self, input_parameter_dict: dict = None,
                 root_directories: list = None,
                 message_output: Callable | None = None) -> None:

        """
        Description
        -----------
        Initializes the Analyst class.

        Parameters
        ----------
        root_directories (list)
            Root directories for data; defaults to None.
        input_parameter_dict (dict)
            Analyses parameters; defaults to None.
        message_output (function)
            Defines output messages; defaults to None.

        Returns
        -------
        None
        """

        if input_parameter_dict is None or root_directories is None:
            with open(pathlib.Path(__file__).parent.parent / '_parameter_settings/analyses_settings.json') as json_file:
                _settings = json.load(json_file)

        self.root_directories = root_directories if root_directories is not None else _settings['analyze_data']['root_directories']
        self.input_parameter_dict = input_parameter_dict if input_parameter_dict is not None else _settings
        self.message_output = message_output if message_output is not None else print

    def analyze_data(self) -> None:
        """
        Description
        -----------
        This method performs the following analyses (listed in execution
        order):
        (1) computes inter-USV-interval distributions across one or more
            session lists
        (2) generates (regular and/or naturalistic) USV playback WAV files
        (3) computes behavioral features and plots their distributions
        (4) computes per-cluster neuronal tuning curves (behavioral + vocal)
        (5) frequency shifts audio segments

        Parameters
        ----------

        Returns
        -------
        None
        """

        # Benign numpy/scipy RuntimeWarnings raised during analysis -- all-NaN
        # slices, divide-by-zero in sparse-bin ratios, and scipy's
        # ConstantInputWarning/NearConstantInputWarning (RuntimeWarning
        # subclasses) from constant-input pearsonr/spearmanr -- carry no
        # actionable information. Suppress only that category, scoped to the run
        # via catch_warnings, instead of the old module-level
        # simplefilter('ignore') that silenced every warning process-wide (GUI
        # included) the moment this module was imported. Genuinely informative
        # warnings (DeprecationWarning, ConvergenceWarning, ...) still surface.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            self._analyze_data_impl()

    def _analyze_data_impl(self) -> None:
        """
        Description
        -----------
        Runs the configured analyses. Wrapped by ``analyze_data``, which scopes
        RuntimeWarning suppression around the whole run; see that method's
        docstring for the list of analyses performed.

        Parameters
        ----------

        Returns
        -------
        None
        """

        Messenger(message_output=self.message_output,
                  receivers=self.input_parameter_dict['send_email']['send_message']['receivers'],
                  credentials_file=pathlib.Path(configure_path(self.input_parameter_dict['credentials_directory'])) / 'email_config.ini',
                  exp_settings_dict=None).send_message(subject=f"{self.input_parameter_dict['send_email']['analyses_pc_choice']} PC is busy, do NOT attempt to remote in!",
                                                       message=f"Data analyses in progress, started at "
                                                               f"{datetime.now().hour:02d}:{datetime.now().minute:02d}:{datetime.now().second:02d} "
                                                               f"and run by @{self.input_parameter_dict['send_email']['experimenter']}. "
                                                               f"You will be notified upon completion. \n \n ***This is an automatic e-mail, please do NOT respond.***")

        # The "PC available again" completion e-mail is sent from a `finally`
        # so that whoever is waiting on the notification is always released --
        # even when an exception type OUTSIDE the per-directory `except` tuple
        # (e.g. KeyboardInterrupt, a settings KeyError raised by the pre-loop
        # IVI/playback steps, or any other unexpected error) propagates out of
        # the analysis body. `failed_directories` is initialised before the
        # `try` so the `finally` can report status regardless of where the run
        # stopped.
        failed_directories = []
        try:
            # # # compute inter-vocalization-interval distributions across one or more session lists
            if self.input_parameter_dict['analyses_booleans']['compute_inter_usv_interval_distributions_bool']:
                InterUSVIntervalCalculator(input_parameter_dict=self.input_parameter_dict,
                              message_output=self.message_output).save_inter_usv_interval_distributions_to_file()

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
                                        f"{datetime.now().hour:02d}:{datetime.now().minute:02d}:{datetime.now().second:02d}.")

                    # # # compute behavioral features and plot their distributions
                    if self.input_parameter_dict['analyses_booleans']['compute_behavioral_features_bool']:
                        FeatureZoo(root_directory=one_directory,
                                   behavioral_parameters_dict=self.input_parameter_dict['compute_behavioral_features'],
                                   message_output=self.message_output).save_behavioral_features_to_file()

                    # # # compute neuronal tuning curves (behavioral and vocal in a single pass)
                    if self.input_parameter_dict['analyses_booleans']['compute_neuronal_tuning_bool']:
                        NeuronalTuning(root_directory=one_directory,
                                       tuning_parameters_dict=self.input_parameter_dict['calculate_neuronal_tuning_curves'],
                                       message_output=self.message_output).calculate_neuronal_tuning_curves()

                    # # # frequency shift audio segments
                    if self.input_parameter_dict['analyses_booleans']['frequency_shift_audio_segment_bool']:
                        AudioGenerator(root_directory=one_directory,
                                       freq_shift_settings_dict=self.input_parameter_dict['frequency_shift_audio_segment'],
                                       message_output=self.message_output).frequency_shift_audio_segment()

                    self.message_output(f"Analyzing data in {one_directory} finished at: "
                                        f"{datetime.now().hour:02d}:{datetime.now().minute:02d}:{datetime.now().second:02d}.")

                except (OSError, RuntimeError, TypeError, IndexError, IOError, EOFError, TimeoutError, NameError, KeyError, ValueError, AttributeError) as exc:
                    self.message_output(traceback.format_exc())
                    failed_directories.append((one_directory, f"{type(exc).__name__}: {exc}"))
        finally:
            # The completion e-mail must report failures honestly: previously it
            # always announced success even when every directory had errored and
            # been swallowed by the except above, so a silent total failure looked
            # like a clean run to whoever was waiting on the notification.
            if failed_directories:
                failure_summary = "\n".join(f"  - {failed_dir}: {failure_reason}"
                                            for failed_dir, failure_reason in failed_directories)
                completion_subject = (f"{self.input_parameter_dict['send_email']['analyses_pc_choice']} PC is available again, "
                                      f"analyses completed with {len(failed_directories)} failure(s)")
                completion_message = (f"Data analyses finished at "
                                      f"{datetime.now().hour:02d}:{datetime.now().minute:02d}:{datetime.now().second:02d} "
                                      f"by @{self.input_parameter_dict['send_email']['experimenter']}, but "
                                      f"{len(failed_directories)} of {len(self.root_directories)} director(ies) failed:\n"
                                      f"{failure_summary}\n\n"
                                      f"See the run log / traceback for details. \n \n "
                                      f"***This is an automatic e-mail, please do NOT respond.***")
            else:
                completion_subject = (f"{self.input_parameter_dict['send_email']['analyses_pc_choice']} PC is available again, "
                                      f"analyses have been completed")
                completion_message = (f"Data analyses have been completed at "
                                      f"{datetime.now().hour:02d}:{datetime.now().minute:02d}:{datetime.now().second:02d} "
                                      f"by @{self.input_parameter_dict['send_email']['experimenter']}. "
                                      f"You will be notified about further PC usage "
                                      f"should it occur. \n \n ***This is an automatic e-mail, please do NOT respond.***")

            Messenger(message_output=self.message_output,
                      no_receivers_notification=False,
                      receivers=self.input_parameter_dict['send_email']['send_message']['receivers'],
                      credentials_file=pathlib.Path(configure_path(self.input_parameter_dict['credentials_directory'])) / 'email_config.ini',
                      exp_settings_dict=None).send_message(subject=completion_subject, message=completion_message)


@click.command(name='generate-usv-playback')
@click.option('--exp-id', type=str, required=True, help='Experimenter ID.')
@click.option('--num-usv-files', 'num_usv_files', type=int, default=None, required=False, help='Number of WAV files to create.')
@click.option('--total-usv-number', 'total_usv_number', type=int, default=None, required=False, help='Total number of USVs to distribute across files.')
@click.option('--ipi-duration', 'ipi_duration', type=float, default=None, required=False, help='Inter-USV-interval duration (in s).')
@click.option('--wav-sampling-rate', 'wav_sampling_rate', type=int, default=None, required=False, help='Sampling rate for the output WAV file (in kHz).')
@click.option('--playback-snippets-dir', 'playback_snippets_dir', type=str, default=None, required=False, help='Directory of USV playback snippets.')
@click.pass_context
def generate_usv_playback_cli(ctx, exp_id, **kwargs) -> None:
    """
    Description
    -----------
    A command-line tool to generate USV playback WAV files.

    Parameters
    ----------

    Returns
    -------
    None
    """

    provided_params = [key for key in kwargs if ctx.get_parameter_source(key) == ParameterSource.COMMANDLINE]

    analyses_settings_parameter_dict = modify_settings_json_for_cli(ctx=ctx,
                                                                    provided_params=provided_params,
                                                                    settings_dict='analyses_settings')
    AudioGenerator(exp_id=exp_id,
                   create_playback_settings_dict=analyses_settings_parameter_dict['create_usv_playback_wav'],
                   message_output=print).create_usv_playback_wav()

@click.command(name='generate-naturalistic-usv-playback')
@click.option('--exp-id', type=str, required=True, help='Experimenter ID.')
@click.option('--num-naturalistic-usv-files', 'num_naturalistic_usv_files', type=int, default=None, required=False, help='Number of WAV files to create.')
@click.option('--naturalistic-wav-sampling-rate', 'naturalistic_wav_sampling_rate', type=int, default=None, required=False, help='Sampling rate for the output WAV file (in kHz).')
@click.option('--naturalistic-playback-snippets-dir-prefix', 'naturalistic_playback_snippets_dir_prefix', type=str, default=None, required=False, help='Prefix of directory of the naturalistic USV playback snippets.')
@click.option('--total-playback-time', 'total_acceptable_naturalistic_playback_time', type=int, default=None, required=False, help='Maximum amount of seconds for the duration of the naturalistic playback file.')
@click.pass_context
def generate_naturalistic_usv_playback_cli(ctx, exp_id, **kwargs) -> None:
    """
    Description
    -----------
    A command-line tool to generate naturalistic USV playback WAV files.

    Parameters
    ----------

    Returns
    -------
    None
    """

    provided_params = [key for key in kwargs if ctx.get_parameter_source(key) == click.core.ParameterSource.COMMANDLINE]

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
@click.option('--peth-window-seconds', 'peth_window_seconds', nargs=2, type=float, default=None, required=False, help='Pre-USV PETH window [start stop] (in s).')
@click.option('--peth-bin-seconds', 'peth_bin_seconds', type=float, default=None, required=False, help='PETH bin width (in s).')
@click.option('--bout-quiet-seconds', 'bout_quiet_seconds', type=float, default=None, required=False, help='Inter-bout silence required to define a new bout (in s).')
@click.option('--n-usv-min-self', 'n_usv_min_self', type=int, default=None, required=False, help='Minimum self-side USV count to compute self plots.')
@click.option('--n-usv-min-partner', 'n_usv_min_partner', type=int, default=None, required=False, help='Minimum partner-side USV count to compute partner plots.')
@click.option('--n-usv-min-category', 'n_usv_min_category', type=int, default=None, required=False, help='Minimum per-category USV count to retain that category.')
@click.option('--include-partner-tuning/--no-include-partner-tuning', 'include_partner_vocalization_tuning_bool', default=None, required=False, help='If set, also compute partner-side vocal tuning when partner threshold is met.')
@click.option('--behavioral-min-occupancy-seconds', 'behavioral_min_occupancy_seconds', type=float, default=None, required=False, help='Minimum behavioral occupancy per bin (in s) for that bin to be rendered in the 1D feature line plots; persisted into behavioral_metadata of each cluster pkl.')
@click.option('--smoothing-sd', 'smoothing_sd', type=float, default=None, required=False, help='Standard deviation (in bins) of the Gaussian smoothing applied to ratemaps and shuffle distributions; 0 disables smoothing.')
@click.pass_context
def generate_rm_files_cli(ctx, root_directory, **kwargs) -> None:
    """
    Description
    -----------
    A command-line tool to compute neuronal tuning curves: behavioral
    (spike-vs-3D-feature ratemaps) and vocal (Q1 pre-USV PETH, Q2 within-USV
    continuous-property tuning, Q3a within-USV categorical, Q3b per-category
    PETH). Each subset is produced if its corresponding input is present in
    the session: behavioral runs when the `*_behavioral_features.csv` is
    present, vocal runs when the `*_usv_summary.csv` is present. Sessions
    missing both inputs return cleanly without producing any tuning files.

    Parameters
    ----------

    Returns
    -------
    None
    """

    parameters_lists = ['temporal_offsets', 'peth_window_seconds']

    provided_params = [key for key in kwargs if ctx.get_parameter_source(key) == ParameterSource.COMMANDLINE]

    analyses_settings_parameter_dict = modify_settings_json_for_cli(ctx=ctx,
                                                                    parameters_lists=parameters_lists,
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
    -----------
    A command-line tool to compute 3D behavioral features.

    Parameters
    ----------

    Returns
    -------
    None
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

@click.command(name='generate-usv-interval-distributions')
@click.option('--session-list', 'session_lists', type=click.Path(exists=True, file_okay=True, dir_okay=False), multiple=True, required=False, help='Path to a text file containing session root directories (one per line). Repeatable.')
@click.option('--output-directory', 'output_directory', type=click.Path(file_okay=False, dir_okay=True), default=None, required=False, help='Directory in which to write the consolidated usv_interval_analysis_<YYYYMMDD>_<HHMMSS>.h5 archive.')
@click.option('--noise-col-id', 'noise_col_id', type=str, default=None, required=False, help='Name of the noise classification column in the USV summary CSV.')
@click.option('--noise-categories', 'noise_categories', multiple=True, type=int, default=None, required=False, help='Integer label(s) in noise_col_id that mark a USV as noise.')
@click.option('--fit-gmm/--no-fit-gmm', 'fit_gmm', default=None, required=False, help='Whether to run the GMM sweep after inter-USV interval extraction.')
@click.option('--n-components-min', 'n_components_min', type=int, default=None, required=False, help='Minimum number of GMM components.')
@click.option('--n-components-max', 'n_components_max', type=int, default=None, required=False, help='Maximum number of GMM components.')
@click.option('--n-repeats', 'n_repeats', type=int, default=None, required=False, help='Number of EM-init repeats per (key, n_components).')
@click.option('--max-modes-reported', 'max_modes_reported', type=int, default=None, required=False, help='Maximum number of mixture modes recorded per fit.')
@click.option('--random-seed-base', 'random_seed_base', type=int, default=None, required=False, help='Base seed; rep r uses random_seed_base + r.')
@click.option('--cv-n-folds', 'cv_n_folds', type=int, default=None, required=False, help='Number of K-fold splits used by cross-validated log-likelihood.')
@click.option('--cv-n-init', 'cv_n_init', type=int, default=None, required=False, help='Number of EM restarts per fold during cross-validation.')
@click.option('--gmm-n-init', 'gmm_n_init', type=int, default=None, required=False, help='Number of EM restarts per in-sample GMM fit.')
@click.option('--gmm-reg-covar', 'gmm_reg_covar', type=float, default=None, required=False, help='Regularisation added to GMM covariances (sklearn reg_covar).')
@click.option('--tau', 'tau', type=float, default=None, required=False, help='Posterior threshold for the LEFT component when computing inter-component decision boundaries; 0.5 = standard Bayes boundary.')
@click.option('--figures-directory', 'figures_directory', type=click.Path(file_okay=False, dir_okay=True), default=None, required=False, help='Directory where the inter-USV interval notebook saves figures (used by downstream plotting; not consumed by the analysis CLI itself).')
@click.option('--model-class', 'model_class', type=click.Choice(['gauss', 't'], case_sensitive=False), default=None, required=False, help='Mixture model class: "gauss" (log-Gaussian mixture, classical) or "t" (Student-t mixture in log-space, recommended for inter-USV interval bout structure because one heavy-tailed t-component absorbs the long-pause tail without inflating the component count).')
@click.option('--bootstrap-lrt-B', 'bootstrap_lrt_B', type=int, default=None, required=False, help='Number of bootstrap replicates per pairwise LRT (McLachlan 1987). Defaults to JSON value (1000); reduce to ~100-200 only for fast-iteration debugging.')
@click.option('--bootstrap-lrt-n-subsample', 'bootstrap_lrt_n_subsample', type=int, default=None, required=False, help='Subsample size used for both observed and bootstrap fits in the LRT, so the LR statistic is on the same N scale. Defaults to JSON value (15000).')
@click.option('--bootstrap-lrt-alpha', 'bootstrap_lrt_alpha', type=float, default=None, required=False, help='Significance threshold for the step-up LRT decision rule. Defaults to JSON value (0.05).')
@click.option('--bootstrap-lrt-bonferroni/--no-bootstrap-lrt-bonferroni', 'bootstrap_lrt_bonferroni', default=None, required=False, help='If set, divide alpha by the number of pairwise tests (per key) before applying the step-up rule.')
@click.pass_context
def generate_usv_interval_distributions_cli(ctx, **kwargs) -> None:
    """
    Description
    -----------
    A command-line tool to compute inter-vocalization-interval (inter-USV interval)
    distributions across one or more session lists, and (optionally)
    sweep a 1D GMM on the pooled log-inter-USV intervals.

    Parameters
    ----------

    Returns
    -------
    None
    """

    parameters_lists = ['session_lists', 'noise_categories']

    provided_params = [key for key in kwargs if ctx.get_parameter_source(key) == ParameterSource.COMMANDLINE]

    analyses_settings_parameter_dict = modify_settings_json_for_cli(ctx=ctx,
                                                                    parameters_lists=parameters_lists,
                                                                    provided_params=provided_params,
                                                                    settings_dict='analyses_settings')
    InterUSVIntervalCalculator(input_parameter_dict=analyses_settings_parameter_dict,
                  message_output=print).save_inter_usv_interval_distributions_to_file()
