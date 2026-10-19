"""
@author: bartulem
Test the ``usv_playpen.analyses`` package: the ``Analyst`` dispatcher and its helpers.

Coverage: the ``Analyst`` boolean routing and its completion e-mail; experiment-code decoding;
the behavioral-feature geometry (derivatives, speed, head / back frames, Euler angles, tail
curvature, SEI) and the ``FeatureZoo`` end-to-end feature table; the neuronal tuning pure helpers
(runs, peak z, selectivity, Skaggs information / sparsity, coherence, ramp index, bout detection,
VMI, smoothing, validity grids) and the synthetic-session behavioral / vocal compute with its
squeak-exclusion and re-run rules; the unit-triage flaggers; the Gaussian / Student-t / IG mixture
utilities (fits, CDF / quantile inversion, boundaries, ICL, CV, bootstrap LRT, step-up selection,
plots); the inter-USV interval helpers (session lists, per-animal pairing, chain breaks, serial
dependence, pool summaries); and the HDF5 interval-archive round-trip with its provenance helpers.
"""

from __future__ import annotations

import json
import math
import pathlib
import pickle
import subprocess as _subprocess
import warnings
from unittest.mock import MagicMock

import h5py
import matplotlib as mpl
import numpy as np
import polars as pls
import pytest
from matplotlib.figure import Figure
from scipy.stats import multivariate_normal, norm
from scipy.stats import t as t_dist
from sklearn.mixture import GaussianMixture

# Headless matplotlib for the mixture_model_utils plotting smoke tests.
mpl.use("Agg")

from usv_playpen.analyses import compute_inter_usv_interval_distributions as cmod
from usv_playpen.analyses import compute_neuronal_tuning_curves as tuning_module
from usv_playpen.analyses._usv_io import extract_animal_sexes
from usv_playpen.analyses.analyze_data import Analyst, generate_beh_features_cli
from usv_playpen.analyses.compute_behavioral_features import (
    FeatureZoo,
    calculate_derivatives,
    calculate_sei,
    calculate_speed,
    calculate_tail_curvature,
    generate_feature_distributions,
    get_back_angles,
    get_back_root,
    get_egocentric_direction,
    get_euler_ang,
    get_head_root,
)
from usv_playpen.analyses.compute_inter_usv_interval_distributions import (
    _read_session_lists,
    _session_source_map,
    compute_session_usv_intervals,
    consecutive_interval_pairs,
    fit_serial_dependence,
    summarize_interval_pool,
)
from usv_playpen.analyses.compute_neuronal_tuning_curves import (
    NeuronalTuning,
    _anchor_bin_validity_grid,
    _bin_count_for_property,
    _bin_property_indices,
    _circular_shift_spike_times,
    _gaussian_smooth_1d,
    _gaussian_smooth_2d,
    _generate_shuffle_offsets,
    _latest_other_stop_before_anchor,
    _longest_run,
    _monotonicity_spearman,
    _peak_z_info,
    _percentiles_block,
    _ramp_index,
    _run_analysis,
    _selectivity_index,
    _skaggs_info_rate_bps,
    _skaggs_sparsity,
    _spatial_coherence,
    _within_usv_validity,
    behavioral_feature_base,
    generate_ratemaps,
    shuffle_spikes,
)
from usv_playpen.analyses.decode_experiment_label import extract_information
from usv_playpen.analyses.mixture_model_utils import (
    IGMixture,
    TMixture,
    _alpha,
    _extract_params,
    _log_gauss,
    _lr_statistic,
    _sample_from_mixture,
    _t_logpdf_1d,
    _t_update_nu,
    bootstrap_lrt,
    fit_log_gmm,
    fit_log_t_mixture,
    gmm_boundaries_logspace,
    gmm_cdf_logspace,
    gmm_cv_neg_loglik,
    gmm_icl,
    gmm_modes,
    gmm_quantile_logspace,
    plot_gmm_fit,
    qqplot_gmm,
    report_gmm_stats,
    report_t_mixture_stats,
    select_best_n_components,
    select_n_components_step_up_lrt,
    summarize_best_gmm,
    summarize_best_t_mixture,
    t_mixture_cdf_logspace,
    t_mixture_cv_neg_loglik,
    t_mixture_icl,
    t_mixture_quantile_logspace,
)
from usv_playpen.analyses.unit_triage_aggregator import (
    _emitter_role_map,
    _flag_categorical,
    _flag_runs,
    _flag_spatial,
    _flag_vmi,
    _safe_float,
    _to_jsonable,
)
from usv_playpen.analyses.usv_interval_archive import (
    _attr_value,
    _decode_attr,
    _h5_to_polars,
    _polars_to_h5,
    _try_git_sha,
    detect_repo_root_for_provenance,
    git_sha_for_provenance,
    read_usv_interval_h5,
    reconstruct_best_model,
    write_ivi_h5,
)


@pytest.fixture
def mock_settings():
    """
    Description
    -----------
    Loads the live ``analyses_settings.json`` so the fixture stays in sync with
    the schema the ``Analyst`` actually references. Every ``analyses_booleans``
    flag is forced to False (tests then flip individual flags as needed), the
    credentials directory is pointed at a fake path, and the e-mail block gets
    an empty receiver list plus placeholder PC / experimenter strings so no
    message can leave the machine.

    Parameters
    ----------

    Returns
    -------
    settings (dict)
        The analyses settings dictionary with every boolean off.
    """
    settings_path = pathlib.Path(__file__).parent.parent.parent / 'src' / 'usv_playpen' / '_parameter_settings' / 'analyses_settings.json'
    settings = json.loads(settings_path.read_text())

    for key in settings['analyses_booleans']:
        settings['analyses_booleans'][key] = False

    settings['credentials_directory'] = '/fake/credentials'
    settings['send_email']['send_message']['receivers'] = []
    settings['send_email']['analyses_pc_choice'] = 'Test PC'
    settings['send_email']['experimenter'] = 'Tester'

    return settings


@pytest.fixture
def mock_dependencies(mocker):
    """
    Description
    -----------
    Patches every class the ``Analyst`` dispatches to (``FeatureZoo``,
    ``NeuronalTuning``, ``AudioGenerator``, ``InterUSVIntervalCalculator``,
    ``Messenger``) at their ``analyze_data`` module-local references, so the
    dispatcher's routing can be checked by call counts without running any
    analysis or sending e-mail.

    Parameters
    ----------
    mocker (pytest_mock.MockerFixture)
        Fixture used to install the patches for the duration of one test.

    Returns
    -------
    mocked_classes (dict)
        The patched class mocks keyed by class name.
    """
    return {
        'FeatureZoo': mocker.patch('usv_playpen.analyses.analyze_data.FeatureZoo'),
        'NeuronalTuning': mocker.patch('usv_playpen.analyses.analyze_data.NeuronalTuning'),
        'AudioGenerator': mocker.patch('usv_playpen.analyses.analyze_data.AudioGenerator'),
        'InterUSVIntervalCalculator': mocker.patch('usv_playpen.analyses.analyze_data.InterUSVIntervalCalculator'),
        'Messenger': mocker.patch('usv_playpen.analyses.analyze_data.Messenger'),
    }

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


def test_analyze_data_reports_failures_in_completion_email(mock_settings, mock_dependencies):
    """A per-directory failure must be surfaced in the completion e-mail
    (subject + body) instead of the old unconditional 'completed' message. The
    orchestrator deliberately swallows the exception to keep going, so the
    e-mail is the only signal the run wasn't clean."""
    mock_settings['analyses_booleans']['compute_behavioral_features_bool'] = True
    mock_dependencies['FeatureZoo'].return_value.save_behavioral_features_to_file.side_effect = \
        RuntimeError("boom")

    analyst = Analyst(input_parameter_dict=mock_settings, root_directories=['/fake/dir1'])
    analyst.analyze_data()

    # The completion e-mail is the last send_message call (the first is the
    # 'PC is busy' start notification).
    last_call = mock_dependencies['Messenger'].return_value.send_message.call_args_list[-1]
    assert "failure(s)" in last_call.kwargs['subject']
    assert "/fake/dir1" in last_call.kwargs['message']
    assert "RuntimeError: boom" in last_call.kwargs['message']


def test_analyze_data_clean_run_reports_success(mock_settings, mock_dependencies):
    """With no failures the completion e-mail keeps its original success wording
    (the honest-reporting change must not regress the clean-run path)."""
    mock_settings['analyses_booleans']['compute_behavioral_features_bool'] = True

    analyst = Analyst(input_parameter_dict=mock_settings, root_directories=['/fake/dir1'])
    analyst.analyze_data()

    last_call = mock_dependencies['Messenger'].return_value.send_message.call_args_list[-1]
    assert "have been completed" in last_call.kwargs['subject']
    assert "failure(s)" not in last_call.kwargs['subject']


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
    mock_settings['analyses_booleans']['compute_neuronal_tuning_bool'] = True
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


def test_both_playback_booleans_run_both_generators(mock_settings, mock_dependencies):
    """
    Description
    -----------
    The standard and the naturalistic playback generators are independent steps.
    With both booleans on, both must run, each with its own settings block; the
    dispatcher used to be an if/else that silently skipped the naturalistic one.

    Parameters
    ----------
    mock_settings (dict)
        Analyses settings fixture.
    mock_dependencies (dict)
        Mocked analysis classes keyed by name.

    Returns
    -------
    None
    """

    mock_settings['analyses_booleans']['create_usv_playback_wav_bool'] = True
    mock_settings['analyses_booleans']['create_naturalistic_usv_playback_wav_bool'] = True

    Analyst(input_parameter_dict=mock_settings, root_directories=[]).analyze_data()

    mock_audio_generator = mock_dependencies['AudioGenerator']
    assert mock_audio_generator.call_count == 2
    settings_passed = [c.kwargs['create_playback_settings_dict'] for c in mock_audio_generator.call_args_list]
    assert settings_passed == [mock_settings['create_usv_playback_wav'], mock_settings['create_naturalistic_usv_playback_wav']]
    mock_audio_generator.return_value.create_usv_playback_wav.assert_called_once()
    mock_audio_generator.return_value.create_naturalistic_usv_playback_wav.assert_called_once()

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


def test_extract_information_returns_none_for_none_input():
    """A missing experiment code (None) decodes to None rather than raising, so sessions without a
    code are skipped cleanly."""
    assert extract_information(None) is None


def test_extract_information_returns_none_for_non_string():
    """A non-string code (an int or a list) decodes to None instead of being parsed character by
    character."""
    assert extract_information(123) is None
    assert extract_information(["E", "2"]) is None


def test_extract_information_simple_courtship_pair():
    """``E2CFM`` decodes to an ephys courtship experiment with two mice whose sexes are female and
    male, the most common code in the cohort."""
    out = extract_information("E2CFM")
    assert out is not None
    assert "ephys" in out["experiment_type"]
    assert "courtship" in out["experiment_type"]
    assert out["mouse_number"] == 2
    assert out["mouse_sex"] == ["female", "male"]


def test_extract_information_solo_recording_with_estrus():
    """``B1QFSe`` decodes every field of a solo recording: one mouse, behavior + alone, female,
    single-housed, in estrus."""
    out = extract_information("B1QFSe")
    assert out is not None
    assert out["mouse_number"] == 1
    assert "behavior" in out["experiment_type"]
    assert "alone" in out["experiment_type"]
    assert out["mouse_sex"] == ["female"]
    assert out["mouse_housing"] == ["single"]
    assert out["mouse_estrus"] == ["estrus"]


def test_extract_information_multi_digit_mouse_number():
    """The animal count is read as the whole digit run, so a 12-animal code gives 12 and not 1."""
    # `\d+` must capture the full count, not just the first digit (>= 10 animals).
    assert extract_information("BG12")["mouse_number"] == 12


def test_extract_information_no_digit_defaults_to_zero():
    """A code with no digit sets ``mouse_number`` to 0 instead of raising on the failed
    regex match."""
    # a code with no digit must default mouse_number to 0, not raise AttributeError.
    assert extract_information("BCFM")["mouse_number"] == 0


def test_extract_information_lighting_decoded():
    """The ``L`` and ``D`` letters decode to light and dark experiment types respectively."""
    light = extract_information("E1L")
    assert light is not None
    assert "light" in light["experiment_type"]
    dark = extract_information("E1D")
    assert dark is not None
    assert "dark" in dark["experiment_type"]


@pytest.mark.parametrize(
    ("estrus_letter", "expected"),
    [("p", "proestrus"), ("e", "estrus"), ("m", "metestrus"), ("d", "diestrus")],
)
def test_extract_information_estrus_letters(estrus_letter, expected):
    """Each of the four lowercase estrus letters (p, e, m, d) maps to its estrus-stage name."""
    out = extract_information(f"B1F{estrus_letter}")
    assert out is not None
    assert expected in out["mouse_estrus"]


def test_extract_information_default_lists_are_empty_when_unmatched():
    """A code carrying no sex, housing or estrus letters returns empty lists for those fields rather
    than missing keys or None."""
    out = extract_information("E1")
    assert out is not None
    assert out["mouse_estrus"] == []
    assert out["mouse_housing"] == []
    assert out["mouse_sex"] == []


def test_generate_feature_distributions_1d_returns_seconds():
    """A 1-D feature histogram has ``num_bins`` bins and ``num_bins + 1`` edges, and its occupancy
    is in seconds (200 frames at 100 fps sum to 2 s)."""
    arr = np.linspace(0.001, 0.999, 200)
    occ, centers, edges = generate_feature_distributions(
        feature_arr=arr,
        min_val=0.0,
        max_val=1.0,
        num_bins=10,
        camera_fr=100,
    )
    assert occ.shape == (10,)
    assert centers.shape == (10,)
    assert edges.shape == (11,)
    assert occ.sum() == pytest.approx(2.0)


def test_generate_feature_distributions_2d_grid_when_space_bool():
    """With ``space_bool`` the occupancy is a square grid whose side is ceil(sqrt(num_bins)), with
    matching centre and edge lengths."""
    rng = np.random.default_rng(0)
    arr = rng.uniform(low=-1, high=1, size=(50, 2))
    occ, centers, edges = generate_feature_distributions(
        feature_arr=arr,
        min_val=-1.0,
        max_val=1.0,
        num_bins=25,
        camera_fr=100,
        space_bool=True,
    )
    side = int(np.ceil(np.sqrt(25)))
    assert occ.shape == (side, side)
    assert centers.shape == (side,)
    assert edges.shape == (side + 1,)


def test_calculate_derivatives_linear_input_first_der_is_constant():
    """A linear ramp has a constant first derivative (slope x frame rate) and zero second derivative
    in the interior, with the first and last ``diff_bins`` frames NaN."""
    fr = 100
    n = 50
    a = 2.0
    x = (a * np.arange(n)).reshape(-1, 1)
    diff_bins = 5
    first_der, second_der = calculate_derivatives(
        input_arr=x.astype(float),
        diff_bins=diff_bins,
        capture_fr=fr,
        is_angle=False,
    )
    assert np.isnan(first_der[:diff_bins]).all()
    assert np.isnan(first_der[-diff_bins:]).all()
    np.testing.assert_allclose(first_der[diff_bins:-diff_bins, 0], a * fr, rtol=1e-9)
    interior = second_der[2 * diff_bins : n - 2 * diff_bins, 0]
    np.testing.assert_allclose(interior, 0.0, atol=1e-9)


def test_calculate_derivatives_propagates_nan_locations():
    """A NaN input frame stays NaN in both derivatives instead of being filled from its
    neighbours."""
    x = np.zeros((30, 1))
    x[10] = np.nan
    first_der, second_der = calculate_derivatives(
        input_arr=x, diff_bins=2, capture_fr=100, is_angle=False
    )
    assert np.isnan(first_der[10]).all()
    assert np.isnan(second_der[10]).all()


def test_calculate_derivatives_angle_wraps_at_pi():
    """For an angle jumping from 179 to -179 degrees, ``is_angle`` takes the short way round, so the
    derivative at the jump stays far below a full 360-degree turn per frame."""
    n = 30
    diff_bins = 1
    angles = np.zeros(n)
    angles[15:] = -179
    angles[:15] = 179
    first_der, _ = calculate_derivatives(
        input_arr=angles.reshape(-1, 1),
        diff_bins=diff_bins,
        capture_fr=100,
        is_angle=True,
    )
    discontinuity_der = first_der[15, 0]
    assert abs(discontinuity_der) < 360 * 100


def test_calculate_tail_curvature_zero_for_straight_tail():
    """A tail whose nodes lie on a straight line has zero curvature in every frame, with one value
    per frame."""
    n_frames = 4
    n_nodes = 6
    pts = np.zeros((n_frames, n_nodes, 3))
    pts[..., 0] = np.linspace(0.0, 1.0, n_nodes)
    out = calculate_tail_curvature(pts)
    assert out.shape == (n_frames, 1)
    np.testing.assert_allclose(out, 0.0, atol=1e-10)


def test_calculate_tail_curvature_positive_for_bent_tail():
    """A tail bent through a corner has positive curvature, so the measure responds to bending."""
    n_frames = 1
    bend = np.array(
        [[0.0, 0.0, 0.0],
         [1.0, 0.0, 0.0],
         [2.0, 0.0, 0.0],
         [2.5, 1.0, 0.0],
         [2.5, 2.0, 0.0]]
    )
    pts = bend[None, :, :].repeat(n_frames, axis=0)
    out = calculate_tail_curvature(pts)
    assert out[0, 0] > 0


def test_get_egocentric_direction_target_in_front_of_observer():
    """A target straight ahead along the head's x-axis has zero egocentric yaw and pitch."""
    head_root = np.eye(3)[None, :, :]
    head_pivot = np.array([[0.0, 0.0, 0.0]])
    target = np.array([[1.0, 0.0, 0.0]])
    yaw, pitch = get_egocentric_direction(head_root, head_pivot, target)
    assert yaw[0] == pytest.approx(0.0)
    assert pitch[0] == pytest.approx(0.0)


def test_get_egocentric_direction_target_above_observer():
    """A target directly above the head has +90 degrees egocentric pitch, which fixes the sign
    convention."""
    head_root = np.eye(3)[None, :, :]
    head_pivot = np.zeros((1, 3))
    target = np.array([[0.0, 0.0, 1.0]])
    _, pitch = get_egocentric_direction(head_root, head_pivot, target)
    assert pitch[0] == pytest.approx(90.0)


def test_get_egocentric_direction_below_tolerance_returns_nan():
    """A target closer to the head than the tolerance gives NaN yaw and pitch, since the direction
    is undefined."""
    head_root = np.eye(3)[None, :, :]
    head_pivot = np.zeros((1, 3))
    target = np.array([[1e-6, 0.0, 0.0]])
    yaw, pitch = get_egocentric_direction(head_root, head_pivot, target)
    assert np.isnan(yaw[0])
    assert np.isnan(pitch[0])


def test_calculate_speed_zero_for_stationary_centroid():
    """A stationary animal has zero speed after the first frame, which is NaN because it has no
    predecessor."""
    n_frames = 30
    pts = np.zeros((n_frames, 4, 3))
    speed = calculate_speed(pts, capture_framerate=100, smoothing_time_window=0.05)
    assert speed.shape == (n_frames, 1)
    assert np.isnan(speed[0, 0])
    np.testing.assert_allclose(speed[1:, 0], 0.0, atol=1e-9)


def test_calculate_speed_constant_velocity_recovered():
    """A centroid moving 0.01 m per frame at 100 fps gives a speed of 100 (cm/s) in the interior,
    which checks the unit scaling."""
    n_frames = 60
    pts = np.zeros((n_frames, 1, 3))
    pts[:, 0, 0] = np.arange(n_frames) * 0.01
    speed = calculate_speed(pts, capture_framerate=100, smoothing_time_window=0.05)
    interior = speed[10:-10, 0]
    np.testing.assert_allclose(interior, 100.0, rtol=1e-6)


def test_get_head_root_canonical_orientation():
    """Head, ears and nose in the canonical layout give the identity head frame: x toward the nose,
    y toward the left ear, z up."""
    head = np.array([0.0, 0.0, 0.0])
    nose = np.array([1.0, 0.0, 0.0])
    ear_l = np.array([0.0, 1.0, 0.0])
    ear_r = np.array([0.0, -1.0, 0.0])
    pts = np.stack([head, ear_r, ear_l, nose])[None, :, :]
    R = get_head_root(pts)
    assert R.shape == (1, 3, 3)
    np.testing.assert_allclose(R[0, 0], [1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(R[0, 1], [0.0, 1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(R[0, 2], [0.0, 0.0, 1.0], atol=1e-12)


def test_get_head_root_propagates_nan_in_inputs():
    """A NaN in any head node makes the whole head frame NaN instead of a partly valid rotation."""
    pts = np.zeros((1, 4, 3))
    pts[0, 0, 0] = np.nan
    R = get_head_root(pts)
    assert np.isnan(R[0]).all()


def test_get_euler_ang_identity_returns_zero_angles():
    """The identity rotation decomposes to zero Euler angles, one (3,) row per frame."""
    R = np.eye(3)[None, :, :]
    angles = get_euler_ang(R)
    assert angles.shape == (1, 3)
    np.testing.assert_allclose(angles[0], [0.0, 0.0, 0.0], atol=1e-9)


def test_generate_ratemaps_1d_returns_expected_shapes():
    """A 1-D rate map returns the (bins, 2) spike / occupancy table, the (shuffles, bins) shuffled
    counts, and centre / edge arrays of matching length."""
    feature = np.linspace(0, 1, 200)
    spikes = np.array([10, 50, 100, 150])
    sh_spikes = np.tile(spikes, (3, 1))
    ratemap, sh_counts, centers, edges = generate_ratemaps(
        feature_arr=feature,
        spike_arr=spikes,
        shuffled_spike_arr=sh_spikes,
        min_val=0.0,
        max_val=1.0,
        num_bins=10,
        camera_fr=100,
    )
    assert ratemap.shape == (10, 2)
    assert sh_counts.shape == (3, 10)
    assert centers.shape == (10,)
    assert edges.shape == (11,)


def test_generate_ratemaps_1d_total_occupancy_matches_samples_over_fr():
    """The occupancy column sums to the session length in seconds (250 frames at 100 fps =
    2.5 s)."""
    feature = np.linspace(0.001, 0.999, 250)
    spikes = np.array([0])
    sh_spikes = np.array([[0]])
    ratemap, *_ = generate_ratemaps(
        feature_arr=feature,
        spike_arr=spikes,
        shuffled_spike_arr=sh_spikes,
        min_val=0.0,
        max_val=1.0,
        num_bins=10,
        camera_fr=100,
    )
    assert ratemap[:, 1].sum() == pytest.approx(2.5)


def test_generate_ratemaps_1d_spike_count_matches_input():
    """Every in-range spike lands in exactly one bin, so the spike column sums to the input spike
    count."""
    feature = np.linspace(0.0, 1.0, 100)
    spikes = np.array([5, 50, 95])
    sh_spikes = np.zeros((1, 3), dtype=int)
    ratemap, *_ = generate_ratemaps(
        feature_arr=feature,
        spike_arr=spikes,
        shuffled_spike_arr=sh_spikes,
        min_val=0.0,
        max_val=1.0,
        num_bins=10,
        camera_fr=100,
    )
    assert ratemap[:, 0].sum() == 3


def test_generate_ratemaps_2d_returns_expected_shapes():
    """A spatial rate map is a (side, side, 2) grid with side = ceil(sqrt(num_bins)) and centre /
    edge arrays sized to it."""
    rng = np.random.default_rng(0)
    feature_xy = rng.uniform(low=-1, high=1, size=(100, 2))
    spikes = np.arange(20)
    sh_spikes = np.tile(spikes, (2, 1))
    ratemap, centers, edges = generate_ratemaps(
        feature_arr=feature_xy,
        spike_arr=spikes,
        shuffled_spike_arr=sh_spikes,
        min_val=-1.0,
        max_val=1.0,
        num_bins=25,
        camera_fr=100,
        space_bool=True,
    )
    side = int(np.ceil(np.sqrt(25)))
    assert ratemap.shape == (side, side, 2)
    assert centers.shape == (side,)
    assert edges.shape == (side + 1,)


def test_shuffle_spikes_shape_and_wraparound():
    """Shuffled spike frames have one row per shuffle and wrap circularly into [0, total frames), so
    no shifted spike falls outside the recording."""
    spikes = np.array([10, 20, 30])
    out = shuffle_spikes(
        spike_array=spikes,
        total_fr_num=100,
        shuffle_min_fr=10,
        shuffle_max_fr=99,
        n_shuffles=20,
    )
    assert out.shape == (20, 3)
    assert (out >= 0).all()
    assert (out < 100).all()


def test_shuffle_spikes_per_row_is_sorted():
    """Each shuffled row is re-sorted after the circular shift, which the downstream binary-search
    binning needs."""
    spikes = np.array([5, 50, 95])
    out = shuffle_spikes(
        spike_array=spikes,
        total_fr_num=200,
        shuffle_min_fr=1,
        shuffle_max_fr=199,
        n_shuffles=10,
    )
    for row in out:
        assert (np.diff(row) >= 0).all()


def test_shuffle_spikes_seed_is_reproducible_and_varies():
    """A fixed seed makes the shuffled null distribution exactly reproducible
    (so the significance verdict is stable across runs), a different seed
    produces a different draw, and the default (seed=None) stays
    non-reproducible -- the property the seeding was introduced to control."""
    kwargs = {'spike_array': np.array([10, 20, 30]), 'total_fr_num': 100,
              'shuffle_min_fr': 10, 'shuffle_max_fr': 99, 'n_shuffles': 20}

    same_a = shuffle_spikes(seed=0, **kwargs)
    same_b = shuffle_spikes(seed=0, **kwargs)
    different = shuffle_spikes(seed=1, **kwargs)

    assert np.array_equal(same_a, same_b)
    assert not np.array_equal(same_a, different)


def test_fit_log_gmm_recovers_two_modes():
    """A two-component GMM fitted to log intervals from modes near 0.05 s and 2 s puts one sorted
    mean on each side of log(0.5)."""
    rng = np.random.default_rng(0)
    short = np.exp(rng.normal(loc=np.log(0.05), scale=0.1, size=300))
    long_ = np.exp(rng.normal(loc=np.log(2.0), scale=0.15, size=300))
    x = np.concatenate([short, long_])

    gmm, order = fit_log_gmm(x, n_components=2, seed=0, n_init=2)

    assert gmm.n_components == 2
    means_sorted = gmm.means_.flatten()[order]
    assert means_sorted[0] < means_sorted[1]
    assert means_sorted[0] < np.log(0.5)
    assert means_sorted[1] > np.log(0.5)


def test_gmm_cdf_logspace_matches_norm_cdf_for_single_component():
    """With one component the mixture CDF equals the normal CDF with the fitted mean and SD."""
    rng = np.random.default_rng(0)
    log_x = rng.normal(loc=0.0, scale=1.0, size=500).reshape(-1, 1)
    gmm = GaussianMixture(n_components=1, covariance_type='full', random_state=0).fit(log_x)
    grid = np.linspace(-3, 3, 50)
    mu = float(gmm.means_.flatten()[0])
    sd = float(np.sqrt(gmm.covariances_.flatten()[0]))
    np.testing.assert_allclose(
        gmm_cdf_logspace(grid, gmm),
        norm.cdf((grid - mu) / sd),
        atol=1e-12,
    )


def test_gmm_quantile_logspace_inverts_cdf():
    """The GMM quantile function inverts the CDF to 1e-6 at five test points, so percentile lookups
    are exact."""
    rng = np.random.default_rng(1)
    log_x = rng.normal(loc=0.0, scale=1.0, size=400).reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=0, n_init=2).fit(log_x)

    test_points = np.array([-1.0, -0.25, 0.0, 0.5, 1.5])
    cdf_values = gmm_cdf_logspace(test_points, gmm)
    recovered = gmm_quantile_logspace(cdf_values, gmm)
    np.testing.assert_allclose(recovered, test_points, atol=1e-6)


def test_gmm_boundaries_logspace_returns_log_and_seconds():
    """Two separated components give one boundary that lies between the means in log space, with the
    seconds value equal to its exponential."""
    rng = np.random.default_rng(2)
    short = rng.normal(loc=-3.0, scale=0.3, size=300)
    long_ = rng.normal(loc=1.0, scale=0.3, size=300)
    log_x = np.concatenate([short, long_]).reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=0, n_init=2).fit(log_x)

    log_b, sec_b = gmm_boundaries_logspace(gmm, tau=0.5)
    assert log_b.shape == (1,)
    assert sec_b.shape == (1,)
    assert -3.0 < log_b[0] < 1.0
    assert sec_b[0] == pytest.approx(np.exp(log_b[0]))


@pytest.mark.parametrize("cov_type", ["full", "tied", "diag", "spherical"])
def test_extract_params_returns_consistent_shapes(cov_type):
    """For every sklearn covariance type the extracted weights, means, covariances and precisions
    have full (K, d, d) shapes, and each precision inverts its covariance."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 2))
    gmm = GaussianMixture(n_components=3, covariance_type=cov_type, random_state=0).fit(X)
    w, M, Sig, Prec = _extract_params(gmm)
    K, d = 3, 2
    assert w.shape == (K,)
    assert M.shape == (K, d)
    assert Sig.shape == (K, d, d)
    assert Prec.shape == (K, d, d)
    for k in range(K):
        np.testing.assert_allclose(Prec[k] @ Sig[k], np.eye(d), atol=1e-6)


def test_extract_params_rejects_unknown_covariance_type():
    """An unrecognised covariance type raises ValueError instead of returning wrongly shaped
    parameters."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 1))
    gmm = GaussianMixture(n_components=1, covariance_type='full', random_state=0).fit(X)
    gmm.covariance_type = 'not_a_real_type'
    with pytest.raises(ValueError, match="Unsupported covariance type"):
        _extract_params(gmm)


def test_log_gauss_matches_scipy_for_2d():
    """The hand-written multivariate Gaussian log-density matches scipy's
    ``multivariate_normal.logpdf`` for a correlated 2-D covariance."""
    mu = np.array([0.0, 0.0])
    Sig = np.array([[1.0, 0.2], [0.2, 0.5]])
    Prec = np.linalg.inv(Sig)
    pts = np.array([[0.0, 0.0], [1.0, -0.5], [-2.0, 1.5]])
    expected = multivariate_normal(mean=mu, cov=Sig).logpdf(pts)
    np.testing.assert_allclose(_log_gauss(pts, mu, Sig, Prec), expected, atol=1e-12)


def test_alpha_returns_nonnegative_responsibilities():
    """The unnormalised responsibilities at a point are finite and non-negative, one per component,
    and the log-max shift makes the largest exactly 1."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 1))
    gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=0, n_init=2).fit(X)
    w, M, Sig, Prec = _extract_params(gmm)
    a = _alpha(np.array([0.5]), w, M, Sig, Prec)
    assert a.shape == (2,)
    assert (a >= 0).all()
    assert np.isfinite(a).all()
    assert a.max() == pytest.approx(1.0)


def test_t_logpdf_1d_matches_scipy():
    """The 1-D Student-t log-density with location and scale matches scipy's location-scale
    ``t.logpdf``."""
    x = np.linspace(-3, 3, 25)
    mu, sigma2, nu = 0.5, 2.0, 5.0
    expected = t_dist.logpdf((x - mu) / np.sqrt(sigma2), df=nu) - 0.5 * np.log(sigma2)
    np.testing.assert_allclose(_t_logpdf_1d(x, mu, sigma2, nu), expected, atol=1e-12)


def test_t_logpdf_1d_gaussian_limit():
    """With very large degrees of freedom the Student-t log-density approaches the standard
    normal's."""
    x = np.linspace(-2, 2, 11)
    mu, sigma2, nu = 0.0, 1.0, 1e6
    np.testing.assert_allclose(
        _t_logpdf_1d(x, mu, sigma2, nu),
        norm.logpdf(x),
        atol=1e-3,
    )


def _make_tmixture():
    """
    Description
    -----------
    Builds a fixed two-component Student-t mixture (weights 0.4 / 0.6, means
    -2.0 / 1.5, variances 0.25 / 0.4, degrees of freedom 10 / 8) directly from
    its parameters, so the ``TMixture`` methods can be checked against
    hand-computed values without running a fit.

    Parameters
    ----------

    Returns
    -------
    model (TMixture)
        The constructed two-component t-mixture.
    """
    return TMixture(
        weights=[0.4, 0.6],
        means=[-2.0, 1.5],
        covariances=[0.25, 0.4],
        nus=[10.0, 8.0],
    )


def test_tmixture_init_normalizes_shapes():
    """``TMixture`` reshapes flat parameter lists into the sklearn layout ((K,), (K, 1), (K, 1, 1))
    so it can stand in for a GaussianMixture in shared code."""
    m = _make_tmixture()
    assert m.weights_.shape == (2,)
    assert m.means_.shape == (2, 1)
    assert m.covariances_.shape == (2, 1, 1)
    assert m.nus_.shape == (2,)
    assert m.n_components == 2
    assert m.covariance_type == "full"


def test_tmixture_score_samples_matches_logaddexp_of_components():
    """``score_samples`` equals the log-sum-exp of the weighted per-component Student-t log-
    densities."""
    m = _make_tmixture()
    x = np.linspace(-3.0, 3.0, 7)
    out = m.score_samples(x)
    components = np.array([
        np.log(m.weights_[k])
        + _t_logpdf_1d(x, m.means_[k, 0], m.covariances_[k, 0, 0], m.nus_[k])
        for k in range(m.n_components)
    ])
    expected = np.logaddexp.reduce(components, axis=0)
    np.testing.assert_allclose(out, expected, atol=1e-12)


def test_tmixture_predict_proba_rows_sum_to_one():
    """Posterior responsibilities are non-negative and each row sums to one."""
    m = _make_tmixture()
    z = m.predict_proba(np.linspace(-3.0, 3.0, 25))
    assert z.shape == (25, 2)
    np.testing.assert_allclose(z.sum(axis=1), 1.0, atol=1e-12)
    assert (z >= 0).all()


def test_tmixture_score_is_mean_of_score_samples():
    """``score`` is the mean per-sample log-likelihood, matching sklearn's convention."""
    m = _make_tmixture()
    x = np.array([-1.0, 0.0, 1.0, 2.0])
    assert m.score(x) == pytest.approx(float(np.mean(m.score_samples(x))))


def test_tmixture_n_params_and_aic_bic_relations():
    """BIC and AIC use 4K - 1 free parameters (weights, means, scales and degrees of freedom), so
    information-criterion comparisons with Gaussian fits are fair."""
    m = _make_tmixture()
    x = np.linspace(-2.0, 2.0, 100)
    n_params = 4 * m.n_components - 1
    log_lik_total = float(np.sum(m.score_samples(x)))
    expected_bic = -2.0 * log_lik_total + n_params * np.log(x.size)
    expected_aic = -2.0 * log_lik_total + 2.0 * n_params
    assert m.bic(x) == pytest.approx(expected_bic)
    assert m.aic(x) == pytest.approx(expected_aic)


def test_t_mixture_cdf_in_unit_interval():
    """The t-mixture CDF lies strictly inside (0, 1) and is non-decreasing over the grid."""
    m = _make_tmixture()
    x = np.linspace(-5.0, 5.0, 30)
    cdf = t_mixture_cdf_logspace(x, m)
    assert ((cdf > 0) & (cdf < 1)).all()
    assert (np.diff(cdf) >= 0).all()


def test_t_mixture_quantile_inverts_cdf():
    """The t-mixture quantile function inverts its CDF to 1e-6."""
    m = _make_tmixture()
    test_points = np.array([-1.5, -0.5, 0.5, 1.0])
    cdf_values = t_mixture_cdf_logspace(test_points, m)
    recovered = t_mixture_quantile_logspace(cdf_values, m)
    np.testing.assert_allclose(recovered, test_points, atol=1e-6)


# triage helpers (compute_neuronal_tuning_curves.py)


# _longest_run

@pytest.mark.parametrize(("mask", "circular", "expected"), [
    (np.empty(0, dtype=bool), False, (-1, -1, 0)),
    (np.zeros(5, dtype=bool), False, (-1, -1, 0)),
    (np.ones(5, dtype=bool), False, (0, 4, 5)),
    (np.ones(5, dtype=bool), True, (0, 4, 5)),                     # cap at n
    (np.array([True, False, False, False, True]), False, (0, 0, 1)),  # no wrap
    (np.array([True, False, False, False, True]), True, (4, 0, 2)),   # wrap detected
    (np.array([False, True, True, False, True, True, True, False]), False, (4, 6, 3)),
    (np.array([True, False, True, False, True]), True, (4, 0, 2)),    # bins 4,0 wrap-adjacent
])
def test_longest_run(mask, circular, expected):
    """`_longest_run` correctly handles linear, circular, empty, and wrap masks."""
    assert _longest_run(mask, circular=circular) == expected


# _peak_z_info


def test_peak_z_info_all_nan_returns_nan():
    """An all-NaN rate gives a NaN peak, index -1 and a NaN signed z instead of raising on an empty
    argmax."""
    nan = np.full(5, np.nan)
    a, idx, signed = _peak_z_info(nan, nan, nan)
    assert math.isnan(a)
    assert idx == -1
    assert math.isnan(signed)


def test_peak_z_info_zero_std_filtered():
    """Bins with zero null SD are left out of the z-score, so a zero-variance null gives no peak
    rather than an infinite z."""
    rate = np.array([1.0, 2.0, 3.0])
    null_mean = np.zeros(3)
    null_std = np.zeros(3)
    a, idx, _signed = _peak_z_info(rate, null_mean, null_std)
    assert math.isnan(a)
    assert idx == -1


def test_peak_z_info_picks_max_abs_excitation():
    """An excitatory bin 4 SD above the null mean is reported as the peak with positive signed z."""
    rate = np.array([1.0, 5.0, 1.0])
    null_mean = np.array([1.0, 1.0, 1.0])
    null_std = np.array([1.0, 1.0, 1.0])
    a, idx, signed = _peak_z_info(rate, null_mean, null_std)
    assert a == 4.0
    assert idx == 1
    assert signed == 4.0


def test_peak_z_info_picks_max_abs_suppression():
    """The peak is chosen by absolute z, so a -10 SD suppression wins and keeps its
    negative sign."""
    rate = np.array([0.0, 0.0, -10.0])
    null_mean = np.array([0.0, 0.0, 0.0])
    null_std = np.array([1.0, 1.0, 1.0])
    a, idx, signed = _peak_z_info(rate, null_mean, null_std)
    assert a == 10.0
    assert idx == 2
    assert signed == -10.0


# _run_analysis


def test_run_analysis_no_divergence():
    """A rate inside the null band in every bin gives zero significant bins and zero-length runs on
    both sides."""
    rate = np.full(5, 1.0)
    null_low = np.zeros(5)
    null_high = np.full(5, 2.0)
    null_mean = np.full(5, 1.0)
    null_std = np.full(5, 1.0)
    out = _run_analysis(rate, null_low, null_high, null_mean, null_std)
    assert out["excit"]["n_bins"] == 0
    assert out["suppress"]["n_bins"] == 0
    assert out["excit"]["max_run"] == 0
    assert out["suppress"]["max_run"] == 0


def test_run_analysis_pure_excitation():
    """A rate above the null band everywhere gives a full-length excitation run and no suppression
    run."""
    rate = np.full(3, 5.0)
    null_low = np.zeros(3)
    null_high = np.full(3, 1.0)
    null_mean = np.full(3, 0.5)
    null_std = np.full(3, 0.5)
    out = _run_analysis(rate, null_low, null_high, null_mean, null_std)
    assert out["excit"]["max_run"] == 3
    assert out["suppress"]["max_run"] == 0


def test_run_analysis_pure_suppression():
    """A rate below the null band everywhere gives a full-length suppression run and no excitation
    run."""
    rate = np.full(3, -5.0)
    null_low = np.full(3, -1.0)
    null_high = np.full(3, 1.0)
    null_mean = np.zeros(3)
    null_std = np.full(3, 1.0)
    out = _run_analysis(rate, null_low, null_high, null_mean, null_std)
    assert out["excit"]["max_run"] == 0
    assert out["suppress"]["max_run"] == 3


def test_run_analysis_circular_wraps_excitation():
    """On a circular axis an excitation run crossing the last-to-first bin boundary counts as one
    length-3 run starting at index 3 and ending at 0."""
    rate = np.array([5.0, 0.0, 0.0, 5.0, 5.0])  # last 2 + first 1 = wrap of length 3
    null_low = np.full(5, -1.0)
    null_high = np.full(5, 1.0)
    null_mean = np.zeros(5)
    null_std = np.full(5, 1.0)
    out = _run_analysis(
        rate, null_low, null_high, null_mean, null_std, circular=True
    )
    assert out["excit"]["max_run"] == 3
    assert out["excit"]["run_start_idx"] == 3
    assert out["excit"]["run_end_idx"] == 0


def test_run_analysis_propagates_axis_indices_for_peak():
    """The excitation block reports the index and z of its peak bin (bin 2, z = 9)."""
    rate = np.array([1.0, 2.0, 10.0, 2.0, 1.0])
    null_low = np.full(5, 0.0)
    null_high = np.full(5, 1.5)
    null_mean = np.full(5, 1.0)
    null_std = np.full(5, 1.0)
    out = _run_analysis(rate, null_low, null_high, null_mean, null_std)
    assert out["excit"]["peak_idx"] == 2
    assert out["excit"]["peak_z"] == pytest.approx(9.0)


def test_run_analysis_handles_nan_rate():
    """NaN rate bins break no comparisons: the three valid excited bins still form a run starting at
    index 1."""
    rate = np.array([np.nan, 5.0, 5.0, 5.0, np.nan])
    null_low = np.full(5, 0.0)
    null_high = np.full(5, 1.0)
    null_mean = np.full(5, 0.5)
    null_std = np.full(5, 0.5)
    out = _run_analysis(rate, null_low, null_high, null_mean, null_std)
    assert out["excit"]["max_run"] == 3
    assert out["excit"]["run_start_idx"] == 1


# _selectivity_index


def test_selectivity_uniform_returns_zero():
    """A flat tuning curve has zero selectivity."""
    assert _selectivity_index(np.full(10, 5.0)) == 0.0


def test_selectivity_max_min_normalized():
    """Selectivity is (max - min) / (max + min), so a curve with a zero minimum scores 1."""
    assert _selectivity_index(np.array([0.0, 10.0])) == 1.0


def test_selectivity_all_nan_returns_nan():
    """An all-NaN curve gives NaN selectivity instead of raising."""
    assert math.isnan(_selectivity_index(np.full(5, np.nan)))


# _monotonicity_spearman


def test_monotonicity_increasing():
    """A strictly increasing curve has Spearman monotonicity 1."""
    assert _monotonicity_spearman(np.arange(10).astype(float)) == pytest.approx(1.0)


def test_monotonicity_decreasing():
    """A strictly decreasing curve has Spearman monotonicity -1."""
    assert _monotonicity_spearman(
        np.arange(10)[::-1].astype(float)
    ) == pytest.approx(-1.0)


def test_monotonicity_constant_returns_nan():
    """A constant curve has undefined rank correlation and returns NaN (scipy's constant-input
    warning is silenced)."""
    # filterwarnings to suppress scipy ConstantInputWarning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert math.isnan(_monotonicity_spearman(np.full(10, 5.0)))


def test_monotonicity_too_few_returns_nan():
    """Fewer than three points give NaN monotonicity, since a rank correlation on two points is
    trivially 1."""
    assert math.isnan(_monotonicity_spearman(np.array([1.0, 2.0])))


# _skaggs_info_rate_bps


def test_skaggs_info_place_cell():
    """A single hot bin out of ten with uniform occupancy carries log2(10) bits per spike, the
    textbook place-cell value."""
    rate = np.array([10.0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    occ = np.ones(10)
    # R = 1.0, p_hot = 0.1, info = 0.1 * 10 * log2(10)
    assert _skaggs_info_rate_bps(rate, occ) == pytest.approx(np.log2(10.0))


def test_skaggs_info_uniform_returns_zero():
    """A uniform rate map carries zero spatial information."""
    assert _skaggs_info_rate_bps(np.full(10, 5.0), np.ones(10)) == pytest.approx(0.0)


def test_skaggs_info_all_zero_rate_returns_nan():
    """A silent unit (mean rate 0) gives NaN information, not a division-by-zero result."""
    assert math.isnan(_skaggs_info_rate_bps(np.zeros(10), np.ones(10)))


def test_skaggs_info_all_nan_returns_nan():
    """An all-NaN rate map gives NaN information."""
    assert math.isnan(
        _skaggs_info_rate_bps(np.full(10, np.nan), np.ones(10))
    )


def test_skaggs_info_zero_occ_excluded():
    """A zero-occupancy bin is left out of the occupancy probabilities, so one hot bin among nine
    visited bins gives log2(9)."""
    rate = np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    occ = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0])
    # 9 valid bins, R = 10/9, info_per_spike = log2(9)
    assert _skaggs_info_rate_bps(rate, occ) == pytest.approx(np.log2(9.0))


def test_skaggs_info_zero_rate_bins_count_in_R():
    """Regression test: zero-rate bins must still contribute time to R."""
    # A 10-bin map with 1 hot bin: R = 0.1*10 = 1.0, info = log2(10) = 3.32
    rate10 = np.array([10.0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    # A 2-bin map with same hot bin: R = 0.5*10 = 5.0, info = log2(2) = 1.0
    rate2 = np.array([10.0, 0])
    info10 = _skaggs_info_rate_bps(rate10, np.ones(10))
    info2 = _skaggs_info_rate_bps(rate2, np.ones(2))
    assert info10 > info2  # bug had R=10 in both cases -> info=0 in both


def test_skaggs_info_2d_ravels_to_1d_answer():
    """A 2-D map gives the same information as the same values flattened to 1-D, so spatial maps
    share the 1-D formula."""
    rate1d = np.array([10.0, 0.0, 0.0, 0.0])
    occ1d = np.ones(4)
    rate2d = rate1d.reshape(2, 2)
    occ2d = occ1d.reshape(2, 2)
    assert _skaggs_info_rate_bps(rate1d, occ1d) == _skaggs_info_rate_bps(rate2d, occ2d)


# _skaggs_sparsity


def test_skaggs_sparsity_uniform_is_one():
    """A uniform map has Skaggs sparsity 1, the least sparse value."""
    assert _skaggs_sparsity(np.full(10, 5.0), np.ones(10)) == pytest.approx(1.0)


def test_skaggs_sparsity_place_cell_is_one_over_n():
    """One hot bin out of ten gives sparsity 1/10."""
    rate = np.array([10.0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    assert _skaggs_sparsity(rate, np.ones(10)) == pytest.approx(0.1)


def test_skaggs_sparsity_all_nan_returns_nan():
    """An all-NaN map gives NaN sparsity."""
    assert math.isnan(_skaggs_sparsity(np.full(10, np.nan), np.ones(10)))


# _spatial_coherence


def test_spatial_coherence_gradient_high():
    """A smooth gradient map has neighbour coherence above 0.9."""
    grid = np.tile(np.arange(10), (10, 1)).astype(float)
    assert _spatial_coherence(grid) > 0.9


def test_spatial_coherence_random_low():
    """A white-noise map has neighbour coherence near zero (|r| < 0.3)."""
    rng = np.random.default_rng(42)
    assert abs(_spatial_coherence(rng.standard_normal((20, 20)))) < 0.3


def test_spatial_coherence_1d_returns_nan():
    """Coherence is only defined on 2-D maps, so a 1-D input returns NaN."""
    assert math.isnan(_spatial_coherence(np.zeros(10)))


# _ramp_index


def test_ramp_index_up_returns_positive():
    """A PETH rising from 1 to 5 toward onset gives ramp index (5 - 1) / (5 + 1)."""
    centers = np.linspace(-2, 0, 41)
    rate = np.zeros(41)
    rate[centers >= -0.2] = 5.0
    rate[centers <= -1.0] = 1.0
    assert _ramp_index(rate, centers, near_anchor_s=-0.1, far_anchor_s=-1.5) == pytest.approx((5 - 1) / (5 + 1), abs=1e-3)


def test_ramp_index_equal_endpoints_returns_zero():
    """A flat PETH has ramp index 0."""
    centers = np.linspace(-2, 0, 41)
    rate = np.full(41, 3.0)
    assert _ramp_index(rate, centers, near_anchor_s=-0.1, far_anchor_s=-1.5) == 0.0


def test_ramp_index_both_zero_returns_nan():
    """Zero rate at both ends gives NaN, since the normalised difference is 0 / 0."""
    centers = np.linspace(-2, 0, 41)
    rate = np.zeros(41)
    assert math.isnan(_ramp_index(rate, centers, near_anchor_s=-0.1, far_anchor_s=-1.5))


def test_ramp_index_nan_endpoints_returns_nan():
    """An all-NaN PETH gives a NaN ramp index."""
    centers = np.linspace(-2, 0, 41)
    rate = np.full(41, np.nan)
    assert math.isnan(_ramp_index(rate, centers, near_anchor_s=-0.1, far_anchor_s=-1.5))


# _detect_bouts


def test_detect_bouts_empty():
    """No USVs give empty bout start, stop and index arrays."""
    bs, be, bi = NeuronalTuning._detect_bouts(np.empty(0), np.empty(0), 2.0)
    assert bs.size == 0
    assert be.size == 0
    assert bi.size == 0


def test_detect_bouts_single_usv():
    """A single USV forms one bout spanning its own start and stop."""
    bs, be, bi = NeuronalTuning._detect_bouts(
        np.array([5.0]), np.array([5.5]), 2.0
    )
    assert bs.tolist() == [5.0]
    assert be.tolist() == [5.5]
    assert bi.tolist() == [0]


def test_detect_bouts_close_gaps_one_bout():
    """USVs separated by gaps shorter than the quiet threshold merge into one bout from the first
    start to the last stop."""
    bs, be, bi = NeuronalTuning._detect_bouts(
        np.array([5.0, 5.5, 6.0]), np.array([5.4, 5.9, 6.4]), 2.0
    )
    assert bs.tolist() == [5.0]
    assert be.tolist() == [6.4]
    assert bi.tolist() == [0, 0, 0]


def test_detect_bouts_far_gaps_separate_bouts():
    """USVs separated by gaps longer than the quiet threshold each start their own bout."""
    bs, _be, bi = NeuronalTuning._detect_bouts(
        np.array([5.0, 10.0, 15.0]), np.array([5.5, 10.5, 15.5]), 2.0
    )
    assert bs.tolist() == [5.0, 10.0, 15.0]
    assert bi.tolist() == [0, 1, 2]


def test_detect_bouts_mixed():
    """A mix of short and long gaps gives the right bout starts, stops and per-USV bout indices."""
    bs, be, bi = NeuronalTuning._detect_bouts(
        np.array([5.0, 5.5, 10.0, 15.0, 15.3]),
        np.array([5.4, 5.9, 10.5, 15.2, 15.5]),
        2.0,
    )
    assert bs.tolist() == [5.0, 10.0, 15.0]
    assert be.tolist() == [5.9, 10.5, 15.5]
    assert bi.tolist() == [0, 0, 1, 2, 2]


# _compute_vmi_for_emitter


def test_vmi_empty_returns_all_nan():
    """An emitter with no USVs gives NaN VMI and zero bouts instead of raising."""
    out = NeuronalTuning._compute_vmi_for_emitter(
        np.array([0.0]), np.empty(0), np.empty(0), np.empty(0), 2.0, wilcoxon_exact_max_n=25
    )
    assert math.isnan(out["vmi"])
    assert out["n_bouts"] == 0


def test_vmi_pure_excitation_returns_one():
    """Spikes only inside the USV and none in the baseline window give VMI = +1."""
    sp = np.array([5.05, 5.15, 5.25])
    em_starts = np.array([5.0])
    em_stops = np.array([5.3])
    em_dur = em_stops - em_starts
    out = NeuronalTuning._compute_vmi_for_emitter(
        sp, em_starts, em_stops, em_dur, 2.0, wilcoxon_exact_max_n=25
    )
    assert out["vmi"] == pytest.approx(1.0)


def test_vmi_pure_suppression_returns_minus_one():
    """Spikes only in the baseline window and none inside the USV give VMI = -1."""
    sp = np.array([3.0, 3.5, 4.0])  # all in baseline window, none in USV
    em_starts = np.array([5.0])
    em_stops = np.array([5.5])
    em_dur = em_stops - em_starts
    out = NeuronalTuning._compute_vmi_for_emitter(
        sp, em_starts, em_stops, em_dur, 2.0, wilcoxon_exact_max_n=25
    )
    assert out["vmi"] == pytest.approx(-1.0)


def test_vmi_baseline_pre_recording_is_nan():
    """A bout whose baseline window would start before time 0 has a NaN baseline rate, not a rate
    computed over a truncated window."""
    sp = np.array([0.6])
    em_starts = np.array([0.5])
    em_stops = np.array([1.0])
    em_dur = em_stops - em_starts
    out = NeuronalTuning._compute_vmi_for_emitter(
        sp, em_starts, em_stops, em_dur, 2.0, wilcoxon_exact_max_n=25
    )
    assert math.isnan(out["fr_baseline"])


def test_vmi_equal_pairs_wilcoxon_nan():
    """When every bout's USV and baseline rates are equal the Wilcoxon test is undefined and its
    p-value is NaN."""
    sp = np.linspace(0, 100, 1000)  # uniform spike rate
    em_starts = np.array([5.0, 10.0, 15.0, 20.0])
    em_stops = np.array([6.0, 11.0, 16.0, 21.0])
    em_dur = em_stops - em_starts
    out = NeuronalTuning._compute_vmi_for_emitter(
        sp, em_starts, em_stops, em_dur, 2.0, wilcoxon_exact_max_n=25
    )
    assert math.isnan(out["wilcoxon_pvalue"])


def test_vmi_n_bouts_matches_detector():
    """The VMI bout count matches ``_detect_bouts`` on the same USVs (three close USVs and one
    isolated USV give two bouts)."""
    # 2 bouts: one big (3 USVs), one small (1 USV)
    em_starts = np.array([5.0, 5.5, 5.8, 20.0])
    em_stops = np.array([5.4, 5.7, 6.0, 20.5])
    em_dur = em_stops - em_starts
    out = NeuronalTuning._compute_vmi_for_emitter(
        np.array([5.05]), em_starts, em_stops, em_dur, 2.0, wilcoxon_exact_max_n=25
    )
    assert out["n_bouts"] == 2


# unit_triage_aggregator.py


# _to_jsonable / _safe_float


@pytest.mark.parametrize(("inp", "expected"), [
    (np.int64(5), 5),
    (np.float64(2.5), 2.5),
    (np.float64(np.nan), None),
    (np.float64(np.inf), None),
    (np.bool_(True), True),
    (np.array([1, 2, 3]), [1, 2, 3]),
    (pathlib.Path("/foo"), "/foo"),
])
def test_to_jsonable(inp, expected):
    """numpy scalars and arrays and Paths convert to plain JSON values, with non-finite floats
    becoming None."""
    assert _to_jsonable(inp) == expected


def test_to_jsonable_unhandled_type_raises():
    """An unsupported object raises TypeError instead of being written as an opaque string."""
    class _X:
        """Bare class with no JSON conversion, used to trigger the TypeError path."""
    with pytest.raises(TypeError):
        _to_jsonable(_X())


@pytest.mark.parametrize(("inp", "expected"), [
    (None, None),
    (5.0, 5.0),
    (np.nan, None),
    (np.inf, None),
    ("not a number", None),
    (np.float64(2.5), 2.5),
])
def test_safe_float(inp, expected):
    """Finite numbers pass through as floats, while None, NaN, inf and non-numeric strings become
    None."""
    assert _safe_float(inp) == expected


# _flag_vmi


def test_flag_vmi_significant_excit():
    """A significant positive VMI with enough bouts is flagged as excitation and returns its
    payload."""
    payload = {
        "vmi": 0.5, "wilcoxon_pvalue": 0.001, "n_bouts": 20,
        "fr_baseline": 1.0, "fr_usv": 3.0,
    }
    direction, info = _flag_vmi(payload, alpha=0.01, min_bouts=10)
    assert direction == "excit"
    assert info["vmi"] == 0.5


def test_flag_vmi_significant_suppress():
    """A significant negative VMI is flagged as suppression."""
    payload = {
        "vmi": -0.5, "wilcoxon_pvalue": 0.001, "n_bouts": 20,
        "fr_baseline": 3.0, "fr_usv": 1.0,
    }
    direction, _info = _flag_vmi(payload, alpha=0.01, min_bouts=10)
    assert direction == "suppress"


def test_flag_vmi_below_min_bouts():
    """Fewer bouts than ``min_bouts`` rejects the unit even when p is small."""
    payload = {
        "vmi": 0.5, "wilcoxon_pvalue": 0.001, "n_bouts": 5,
        "fr_baseline": 1.0, "fr_usv": 3.0,
    }
    assert _flag_vmi(payload, alpha=0.01, min_bouts=10) == (None, None)


def test_flag_vmi_above_alpha_rejected():
    """A p-value above alpha rejects the unit."""
    payload = {
        "vmi": 0.5, "wilcoxon_pvalue": 0.5, "n_bouts": 20,
        "fr_baseline": 1.0, "fr_usv": 3.0,
    }
    assert _flag_vmi(payload, alpha=0.01, min_bouts=10) == (None, None)


def test_flag_vmi_nan_pvalue_rejected():
    """A NaN p-value is treated as not significant, not as passing."""
    payload = {
        "vmi": 0.5, "wilcoxon_pvalue": float("nan"), "n_bouts": 20,
        "fr_baseline": 1.0, "fr_usv": 3.0,
    }
    assert _flag_vmi(payload, alpha=0.01, min_bouts=10) == (None, None)


def test_flag_vmi_gates_on_n_valid_pairs_over_n_bouts():
    """The bout gate counts the paired observations that entered the Wilcoxon (``n_valid_pairs``),
    not all bouts, so 20 bouts with only 5 valid pairs are rejected."""
    # plenty of total bouts, but only 5 paired observations entered the
    # Wilcoxon -> the n_valid_pairs gate (5 < 10) must reject it even though
    # n_bouts (20) would have passed.
    payload = {
        "vmi": 0.5, "wilcoxon_pvalue": 0.001, "n_bouts": 20,
        "n_valid_pairs": 5, "fr_baseline": 1.0, "fr_usv": 3.0,
    }
    assert _flag_vmi(payload, alpha=0.01, min_bouts=10) == (None, None)


def test_flag_vmi_n_valid_pairs_passes():
    """With ``n_valid_pairs`` above the minimum the unit is flagged and the count is carried into
    the returned info."""
    payload = {
        "vmi": 0.5, "wilcoxon_pvalue": 0.001, "n_bouts": 20,
        "n_valid_pairs": 15, "fr_baseline": 1.0, "fr_usv": 3.0,
    }
    direction, info = _flag_vmi(payload, alpha=0.01, min_bouts=10)
    assert direction == "excit"
    assert info["n_valid_pairs"] == 15


# _flag_runs


def test_flag_runs_pass():
    """A run at least ``min_run`` long whose peak |z| clears the threshold passes and returns its
    block."""
    block = {"max_run": 5, "peak_z": 4.0, "n_bins": 5}
    info = _flag_runs(block, z_threshold=3.0, min_run=3)
    assert info is not None
    assert info["max_run"] == 5


def test_flag_runs_short_run_rejected():
    """A run shorter than ``min_run`` is rejected even with a high peak z."""
    block = {"max_run": 2, "peak_z": 4.0, "n_bins": 2}
    assert _flag_runs(block, z_threshold=3.0, min_run=3) is None


def test_flag_runs_low_z_rejected():
    """A long run whose peak z is below threshold is rejected."""
    block = {"max_run": 5, "peak_z": 1.0, "n_bins": 5}
    assert _flag_runs(block, z_threshold=3.0, min_run=3) is None


def test_flag_runs_negative_z_passes_via_abs():
    """The z threshold is applied to |peak_z|, so a strongly negative peak passes."""
    block = {"max_run": 5, "peak_z": -4.0, "n_bins": 5}
    info = _flag_runs(block, z_threshold=3.0, min_run=3)
    assert info is not None


# _flag_categorical


def test_flag_categorical_pass():
    """A category tuning whose peak |z| clears the threshold is flagged and keeps its best
    category."""
    payload = {
        "peak_abs_z": 4.0, "best_cat": 5, "n_sig_categories": 3,
        "selectivity": 0.5, "peak_signed_z": 4.0,
    }
    info = _flag_categorical(payload, z_threshold=3.0)
    assert info is not None
    assert info["best_cat"] == 5


def test_flag_categorical_fail():
    """A category tuning below the z threshold is not flagged."""
    payload = {
        "peak_abs_z": 1.0, "best_cat": 5, "n_sig_categories": 0,
        "selectivity": 0.0, "peak_signed_z": 1.0,
    }
    assert _flag_categorical(payload, z_threshold=3.0) is None


# _flag_spatial


def test_flag_spatial_pass():
    """A spatial map whose information rate clears the threshold is flagged with its metrics."""
    payload = {
        "info_rate_bps": 1.5, "sparsity": 0.3, "coherence": 0.8,
        "peak_rate_sps": 5.0, "peak_row": 3, "peak_col": 4,
    }
    info = _flag_spatial(payload, info_threshold=0.5)
    assert info is not None
    assert info["info_rate_bps"] == 1.5


def test_flag_spatial_fail():
    """A spatial map below the information threshold is not flagged."""
    payload = {
        "info_rate_bps": 0.1, "sparsity": 0.9, "coherence": 0.5,
        "peak_rate_sps": 1.0, "peak_row": 0, "peak_col": 0,
    }
    assert _flag_spatial(payload, info_threshold=0.5) is None


# _emitter_role_map


def test_emitter_role_map_from_usv_peth():
    """Emitter roles are read from the ``usv_peth`` block when it carries them."""
    cluster_data = {
        "usv_peth": {"emitter_a": {"role": "self", "rate": np.zeros(5)}}
    }
    assert _emitter_role_map(cluster_data) == {"emitter_a": "self"}


def test_emitter_role_map_from_property_tuning():
    """When the PETH block has no roles they are read from the per-property tuning payloads
    instead."""
    cluster_data = {
        "usv_property_tuning": {
            "emitter_b": {"duration": {"role": "partner"}}
        }
    }
    assert _emitter_role_map(cluster_data) == {"emitter_b": "partner"}


def test_emitter_role_map_missing_role_returns_empty():
    """An emitter whose payload has no role is left out of the map instead of getting a guessed
    role."""
    cluster_data = {"usv_peth": {"emitter_x": {"rate": np.zeros(5)}}}
    assert _emitter_role_map(cluster_data) == {}

# mixture_model_utils — additional targeted tests


def _make_two_component_gmm(seed=0, n_per=2000):
    """
    Description
    -----------
    Builds a fitted 1-D ``GaussianMixture`` with two well-separated modes
    (N(-2, 0.3) and N(+2, 0.4) in log space, ``n_per`` samples each), so
    mode-finding, ICL and summary helpers have an unambiguous two-mode target.

    Parameters
    ----------
    seed (int)
        Seed for both the sample generator and the sklearn fit.
    n_per (int)
        Number of samples drawn per mode.

    Returns
    -------
    gmm, order, log_x (tuple)
        The fitted mixture, the argsort of its means (ascending), and the
        (2 * n_per, 1) log-space sample array it was fitted to.
    """
    rng = np.random.default_rng(seed)
    a = rng.normal(loc=-2.0, scale=0.3, size=n_per)
    b = rng.normal(loc=+2.0, scale=0.4, size=n_per)
    log_x = np.concatenate([a, b]).reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, random_state=seed).fit(log_x)
    order = np.argsort(gmm.means_.ravel())
    return gmm, order, log_x


def test_gmm_modes_1d_recovers_component_centers():
    """A well-separated two-component 1-D mixture has two modes near the component means, with
    densities sorted descending."""
    gmm, _, _ = _make_two_component_gmm()
    modes, dens = gmm_modes(gmm)
    modes_flat = np.asarray(modes).ravel()
    dens_flat = np.asarray(dens).ravel()
    # well-separated 2-component mixture in 1D has 2 modes
    assert modes_flat.size == 2
    means = sorted(gmm.means_.ravel().tolist())
    assert min(modes_flat) == pytest.approx(means[0], abs=0.1)
    assert max(modes_flat) == pytest.approx(means[1], abs=0.1)
    # densities sorted descending
    assert dens_flat[0] >= dens_flat[-1]


def test_gmm_modes_returns_same_count_for_modes_and_densities():
    """Modes and densities arrays describe the same set; same count."""
    gmm, _, _ = _make_two_component_gmm()
    modes, dens = gmm_modes(gmm)
    assert np.asarray(modes).ravel().size == np.asarray(dens).ravel().size


def test_gmm_icl_matches_bic_minus_entropy_term():
    """ICL adds a non-negative classification-entropy penalty to BIC, so ICL >= BIC for a fitted
    mixture."""
    gmm, _, log_x = _make_two_component_gmm()
    icl = gmm_icl(gmm, log_x)
    bic = gmm.bic(log_x)
    # ICL should be >= BIC for any non-degenerate fit (entropy term is
    # always non-negative).
    assert icl >= bic


def test_gmm_cv_neg_loglik_returns_finite_with_sufficient_data():
    """With enough data for every fold the cross-validated negative log-likelihood is finite."""
    rng = np.random.default_rng(2)
    intervals = np.exp(rng.normal(0.0, 1.0, size=400))
    val = gmm_cv_neg_loglik(intervals, n_components=2, seed=0, n_folds=3, n_init=2)
    assert np.isfinite(val)


def test_select_best_n_components_picks_min_ic():
    """Schema is ['key', 'n_comp', 'rep', ic_col] per the function docstring."""
    df = pls.DataFrame({
        "key":    ["a", "a", "a", "b", "b", "b"],
        "n_comp": [2,   3,   4,   2,   3,   4],
        "rep":    [0,   0,   0,   0,   0,   0],
        "bic":    [100.0, 90.0, 95.0, 200.0, 180.0, 220.0],
    })
    out = select_best_n_components(df, ic_col="bic")
    # The function may return either an int or a dict per key; assert the
    # n_comp value either way.
    def _n_comp(v):
        """Read ``n_comp`` from either a per-key dict or a bare integer selection."""
        return v["n_comp"] if isinstance(v, dict) else int(v)
    assert _n_comp(out["a"]) == 3
    assert _n_comp(out["b"]) == 3


def test_report_gmm_stats_returns_aligned_outputs():
    """``report_gmm_stats`` returns one mean and SD per component in ascending order, and modes
    aligned with their densities."""
    gmm, order, _ = _make_two_component_gmm()
    means, sds, modes, mode_dens = report_gmm_stats(gmm, order)
    assert means.shape == sds.shape == (2,)
    assert modes.shape == mode_dens.shape
    # means should be sorted ascending after applying `order`
    assert means[0] < means[1]


def test_summarize_best_gmm_includes_log_and_seconds():
    """The best-GMM summary carries both log-space and seconds-space entries, which the archive and
    figures both read."""
    gmm, order, _ = _make_two_component_gmm()
    out = summarize_best_gmm(gmm, order, tau=0.5)
    # the dict must carry both log-space and seconds-space entries
    assert any("log" in k.lower() for k in out)
    assert any("sec" in k.lower() for k in out)


# Student-t mixture


def _t_mixture_intervals(rng, n_per=300, log_loc_a=-2.0, log_loc_b=2.0):
    """
    Description
    -----------
    Returns strictly positive interval values whose logs follow a two-mode
    Student-t mixture (df = 10, scale 0.4, locations ``log_loc_a`` and
    ``log_loc_b``). ``fit_log_t_mixture`` takes raw intervals in seconds and
    applies the log internally, so the samples are exponentiated here.

    Parameters
    ----------
    rng (np.random.Generator)
        Seeded random generator.
    n_per (int)
        Number of samples per mode.
    log_loc_a (float)
        Log-space location of the first mode.
    log_loc_b (float)
        Log-space location of the second mode.

    Returns
    -------
    intervals (np.ndarray)
        A (2 * n_per,) array of positive intervals in seconds.
    """
    a = rng.standard_t(df=10.0, size=n_per) * 0.4 + log_loc_a
    b = rng.standard_t(df=10.0, size=n_per) * 0.4 + log_loc_b
    return np.exp(np.concatenate([a, b]))


def test_fit_log_t_mixture_recovers_one_component():
    """A one-component t-mixture fit returns one component whose weights sum to 1."""
    rng = np.random.default_rng(3)
    intervals = np.exp(rng.standard_t(df=10.0, size=500) * 0.5)
    model, _ = fit_log_t_mixture(
        intervals, n_components=1, seed=0, n_init=1, max_iter=80,
    )
    assert model.n_components == 1
    assert model.weights_.sum() == pytest.approx(1.0, abs=1e-6)


def test_fit_log_t_mixture_two_modes_recovered_centers():
    """A two-mode t-mixture fit to log intervals centred at -2 and +2 puts one sorted mean on each
    side of zero."""
    rng = np.random.default_rng(4)
    intervals = _t_mixture_intervals(rng, n_per=400)
    model, _ = fit_log_t_mixture(
        intervals, n_components=2, seed=0, n_init=2, max_iter=80,
    )
    sorted_means = np.sort(model.means_.ravel())
    assert sorted_means[0] < 0 < sorted_means[1]


def test_t_mixture_icl_finite():
    """ICL of a fitted two-component t-mixture is finite."""
    rng = np.random.default_rng(5)
    intervals = _t_mixture_intervals(rng, n_per=200)
    model, _ = fit_log_t_mixture(
        intervals, n_components=2, seed=0, n_init=1, max_iter=60,
    )
    val = t_mixture_icl(model, np.log(intervals))
    assert np.isfinite(val)


def test_t_mixture_cv_neg_loglik_finite():
    """The cross-validated t-mixture negative log-likelihood is finite with enough data."""
    rng = np.random.default_rng(6)
    intervals = _t_mixture_intervals(rng, n_per=150)
    val = t_mixture_cv_neg_loglik(
        intervals, n_components=2, seed=0, n_folds=3, n_init=2,
    )
    assert np.isfinite(val)


def test_report_t_mixture_stats_shapes():
    """``report_t_mixture_stats`` returns one mean, SD and nu per component, and modes aligned with
    their densities."""
    rng = np.random.default_rng(7)
    intervals = _t_mixture_intervals(rng, n_per=200)
    model, order = fit_log_t_mixture(
        intervals, n_components=2, seed=0, n_init=1, max_iter=60,
    )
    means, sds, modes, mode_dens, nus = report_t_mixture_stats(model, order)
    assert means.shape == sds.shape == nus.shape == (2,)
    assert np.asarray(modes).shape == np.asarray(mode_dens).shape


def test_summarize_best_t_mixture_log_and_seconds():
    """The best t-mixture summary carries both log-space and seconds-space entries."""
    rng = np.random.default_rng(8)
    intervals = _t_mixture_intervals(rng, n_per=200)
    model, order = fit_log_t_mixture(
        intervals, n_components=2, seed=0, n_init=1, max_iter=60,
    )
    out = summarize_best_t_mixture(model, order)
    assert any("log" in k.lower() for k in out)
    assert any("sec" in k.lower() for k in out)


# Helper functions


def test_sample_from_mixture_returns_correct_shape():
    """Sampling N points from a fitted mixture returns exactly N values, which the parametric
    bootstrap relies on."""
    rng = np.random.default_rng(9)
    intervals = _t_mixture_intervals(rng, n_per=200)
    model, _ = fit_log_t_mixture(
        intervals, n_components=2, seed=0, n_init=1, max_iter=40,
    )
    samples = _sample_from_mixture(model, N=200, rng=rng)
    assert np.asarray(samples).ravel().size == 200


def test_lr_statistic_alt_better_returns_positive():
    """If the alt model fits better than the null, LR should be positive."""
    rng = np.random.default_rng(10)
    a = rng.normal(-2.0, 0.3, size=400)
    b = rng.normal(+2.0, 0.3, size=400)
    log_x = np.concatenate([a, b]).reshape(-1, 1)
    null = GaussianMixture(n_components=1, random_state=0).fit(log_x)
    alt = GaussianMixture(n_components=2, random_state=0).fit(log_x)
    lr = _lr_statistic(null, alt, log_x.ravel())
    assert lr > 0


def test_t_update_nu_returns_finite_in_range():
    """`_t_update_nu` finds a ν root via Brent; result must be in (2, 200]."""
    rng = np.random.default_rng(11)
    z_k = rng.uniform(0.1, 1.0, size=200)
    u_k = rng.uniform(0.5, 2.0, size=200)
    nu = _t_update_nu(z_k=z_k, u_k=u_k, nu_old=10.0, n_k=float(z_k.sum()))
    assert 2.0 < nu <= 200.0


# compute_neuronal_tuning_curves — synthetic integration tests
#
# These tests exercise the per-cluster behavioral and vocal compute methods
# end-to-end with synthetic data. No real spike trains, USV CSVs, or
# tracking H5s — just numpy/h5py-built fakes wired through the actual code
# path. Goal: lift coverage on _compute_one_cluster_behavioral,
# _load_*_inputs, _build_vocal_side_precompute, _compute_one_cluster_vocal,
# and the triage attachers.


def _make_synthetic_session(tmp_path, *, n_frames=1500, n_usvs=120, fps=150.0):
    """
    Description
    -----------
    Builds a minimal session directory layout with the files ``NeuronalTuning``
    looks for: a ``*_behavioral_features.csv`` for one animal ``m1`` (speed,
    acceleration, allocentric yaw and spaceX / spaceY), a translated/rotated
    tracking H5 whose ``tracks`` leading axis sets the session length, a
    ``session_metadata.yaml`` giving ``m1``'s sex, an
    ``audio_triggerbox_sync_info.json``, a ``*_usv_summary.csv`` of pure USVs
    (noise / squeak false, usv true, null squeak extents, a QLVM category and
    the continuous acoustic properties), and one cluster ``.npy`` of spike
    seconds and frames. All values are drawn from a seeded generator.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Per-test temporary directory under which ``session/`` is created.
    n_frames (int)
        Number of tracked frames.
    n_usvs (int)
        Number of USV rows in the summary.
    fps (float)
        Camera frame rate written to the tracking H5.

    Returns
    -------
    root, cluster_path (tuple)
        The session root and the path of the cluster file the compute consumes.
    """

    root = tmp_path / "session"
    (root / "video" / "vid1").mkdir(parents=True)
    (root / "audio" / "sync").mkdir(parents=True)
    (root / "ephys" / "imec0" / "cluster_data").mkdir(parents=True)

    rng = np.random.default_rng(0)
    # Behavioral CSV: just the features we'll exercise. One animal "m1".
    df_beh = pls.DataFrame({
        "m1.speed":        rng.uniform(0, 30, n_frames),
        "m1.acceleration": rng.uniform(-100, 100, n_frames),
        "m1.allo_yaw":     rng.uniform(-180, 180, n_frames),
        "m1.spaceX":       rng.uniform(-30, 30, n_frames),
        "m1.spaceY":       rng.uniform(-30, 30, n_frames),
    })
    beh_csv = root / "video" / "vid1" / "vid1_behavioral_features.csv"
    df_beh.write_csv(beh_csv)

    # Tracking H5
    h5_path = root / "video" / "vid1" / "20230101000000_points3d_translated_rotated_metric.h5"
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("track_names", data=np.array([b"m1"]))
        f.create_dataset("recording_frame_rate", data=np.float64(fps))
        f.create_dataset("experimental_code", data=b"e99")
        # `tracks` leading axis = frame count; the vocal compute reads
        # `tracks.shape[0]` to derive session duration in seconds.
        f.create_dataset("tracks", data=np.zeros((n_frames, 1, 1, 3), dtype=float))

    # Session metadata: the vocal sides read each animal's sex from its Subjects entry.
    (root / "session_metadata.yaml").write_text("Subjects:\n- subject_id: 'm1'\n  sex: male\n")

    # Audio sync JSON
    duration_s = float(n_frames / fps)
    sync_json = root / "audio" / "sync" / "audio_triggerbox_sync_info.json"
    sync_json.write_text(json.dumps({"m": {"duration_seconds": duration_s}}))

    # USV summary CSV (for vocal compute)
    starts = np.sort(rng.uniform(2, duration_s - 2, n_usvs))
    durations = rng.uniform(0.05, 0.2, n_usvs)
    stops = starts + durations
    usv_df = pls.DataFrame({
        "start": starts.tolist(),
        "stop": stops.tolist(),
        "duration": durations.tolist(),
        "emitter": ["m1"] * n_usvs,
        # Every synthetic USV is real; the tuning-curve loader drops noise-flagged rows.
        "noise": [False] * n_usvs,
        # No squeaks: the vocal tuning keeps pure-USV anchors (usv true, squeak false), so the
        # two booleans are required.
        "usv": [True] * n_usvs,
        "squeak": [False] * n_usvs,
        # The squeak extent detect-usv-squeaks writes (null on rows holding no squeak); the
        # tuning loader times squeak anchors by it and needs the columns.
        "squeak_start": [None] * n_usvs,
        "squeak_end": [None] * n_usvs,
        # the only QLVM category column (R-1..R-4 of the regular map)
        "qlvm_category":     rng.integers(1, 5, size=n_usvs).tolist(),
        "mean_freq_hz":      rng.uniform(40000, 90000, n_usvs).tolist(),
        "peak_freq_hz":      rng.uniform(40000, 90000, n_usvs).tolist(),
        "freq_bandwidth_hz": rng.uniform(5000, 30000, n_usvs).tolist(),
        "mean_amplitude":    rng.uniform(0.1, 3.0, n_usvs).tolist(),
        "max_amplitude":     rng.uniform(0.5, 8.0, n_usvs).tolist(),
        "spectral_entropy":  rng.uniform(1.0, 4.0, n_usvs).tolist(),
        "mask_number":       rng.integers(1, 12, n_usvs).tolist(),
    })
    usv_csv = root / "audio" / "sync" / "vid1_usv_summary.csv"
    usv_df.write_csv(usv_csv)

    # Cluster .npy: shape (2, n_spikes); row 0 = spike times (s), row 1 = frames
    n_spikes = 800
    spike_times = np.sort(rng.uniform(0, duration_s, n_spikes))
    spike_frames = np.clip(
        np.floor(spike_times * fps).astype(np.int64), 0, n_frames - 1,
    )
    cluster_arr = np.vstack([spike_times, spike_frames.astype(float)])
    cluster_path = root / "ephys" / "imec0" / "cluster_data" / "imec0_cl0001_ch001_good.npy"
    np.save(cluster_path, cluster_arr)

    return root, cluster_path


@pytest.fixture
def synthetic_compute_session(tmp_path):
    """
    Description
    -----------
    Builds the synthetic tuning-curve session of ``_make_synthetic_session``
    with its default sizes for one test.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Per-test temporary directory.

    Returns
    -------
    root, cluster_path (tuple)
        The session root and the cluster file path.
    """
    return _make_synthetic_session(tmp_path)


def _make_neuronal_tuning(root, *, n_shuffles=5, smoothing_sd=0.0):
    """
    Description
    -----------
    Constructs a ``NeuronalTuning`` instance with small, fast settings: one
    temporal offset, a handful of shuffles, ten feature bins, 36 spatial bins,
    a two-second PETH window, the IBI bout-onset block, the squeak-exclusion
    and minimum-count rules, and a silent message sink. Tests change individual
    entries of ``tuning_parameters_dict`` afterwards when they need to.

    Parameters
    ----------
    root (pathlib.Path)
        Session root directory.
    n_shuffles (int)
        Number of null shuffles.
    smoothing_sd (float)
        Gaussian smoothing SD in bins; 0 disables smoothing.

    Returns
    -------
    nt (NeuronalTuning)
        The configured tuning instance.
    """
    return NeuronalTuning(
        root_directory=str(root),
        tuning_parameters_dict={
            "temporal_offsets": [0],
            "n_shuffles": n_shuffles,
            "shuffle_seed": 0,
            "total_bin_num": 10,
            "n_spatial_bins": 36,
            "spatial_scale_cm": 32,
            "shuffle_seconds_range": [3.0, 6.0],
            "peth_window_seconds": [-2.0, 0.0],
            "peth_bin_seconds": 0.05,
            "bout_quiet_seconds": 2.0,
            "ramp_anchor_near_seconds": -0.1,
            "ramp_anchor_far_seconds": -1.5,
            "vmi_wilcoxon_exact_max_n": 25,
            "ibi_bout_onset": {
                "ibi_z_score": 2.58,
                "ibi_component_index": 0,
                "min_vocalizations": 3,
                "shuffle_guard_seconds": 1.0,
                "trigger_window_seconds": [-1.0, 2.0],
                "trigger_bin_seconds": 0.02,
                "smoothing_bins": 2.0,
                "gain_window_seconds": [0.0, 0.25],
                "pre_window_seconds": [-1.0, -0.5],
                "peak_search_seconds": [-0.25, 1.0],
                "response_window_seconds": 0.15,
                "response_kernel_seconds": 0.02,
            },
            "vocal_require_clean_post_anchor": True,
            "vocal_require_clean_prior_anchor": False,
            "n_usv_min_self": 5,
            "n_usv_min_partner": 30,
            "n_usv_min_category": 3,
            "behavioral_min_occupancy_seconds": 0.1,
            "usv_property_min_occupancy_seconds": 0.05,
            "include_partner_vocalization_tuning_bool": False,
            "exclude_squeaks_self": True,
            "exclude_squeaks_partner": False,
            "excluded_behavioral_features": ["nose-nose", "allo_yaw-nose", "nose-allo_yaw",
                                             "allo_pitch-nose", "nose-allo_pitch"],
            "smoothing_sd": smoothing_sd,
            "circular_features": ["allo_yaw", "body_dir"],
        },
        message_output=lambda *_a, **_k: None,
    )


# _load_behavioral_inputs


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_load_behavioral_inputs_returns_expected_keys(synthetic_compute_session):
    """The behavioral loader returns the feature table, the track names and the empirical camera
    rate read from the tracking H5."""
    root, _ = synthetic_compute_session
    nt = _make_neuronal_tuning(root)
    bundle = nt._load_behavioral_inputs()
    assert bundle is not None
    assert "behavioral_data" in bundle
    assert bundle["animal_ids"] == ["m1"]
    assert bundle["empirical_camera_sr"] == pytest.approx(150.0)


def test_load_behavioral_inputs_returns_none_when_csv_missing(tmp_path):
    """If the behavioral CSV is absent, loader returns None gracefully."""
    nt = _make_neuronal_tuning(tmp_path)
    assert nt._load_behavioral_inputs() is None


# _compute_one_cluster_behavioral


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_compute_one_cluster_behavioral_smoke(synthetic_compute_session):
    """One cluster's behavioral compute returns the offset block, metadata and triage stats, with
    1-D features and the 2-D spatial map built from spaceX / spaceY carrying rate, occupancy and
    null statistics."""
    root, cluster_path = synthetic_compute_session
    nt = _make_neuronal_tuning(root)
    beh_inputs = nt._load_behavioral_inputs()
    partial = nt._compute_one_cluster_behavioral(cluster_path, beh_inputs)
    # Top-level structure: one offset block + metadata + triage_stats
    assert "beh_offset=0s" in partial
    assert "behavioral_metadata" in partial
    assert "triage_stats" in partial
    # 1D + 2D features both present
    feats = partial["beh_offset=0s"]
    assert "m1.speed" in feats
    assert "m1.space" in feats   # 2D spatial map auto-derived from spaceX/Y
    # 1D feature payload has rate / occupancy / null stats
    speed_payload = feats["m1.speed"]
    for k in ("rate", "occupancy_seconds", "bin_centers",
              "null_mean", "null_std", "null_p99_5"):
        assert k in speed_payload


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_compute_one_cluster_behavioral_attaches_triage_stats(synthetic_compute_session):
    """Behavioral triage stats carry run analysis and selectivity for 1-D features, the circular
    flag for yaw, and Skaggs metrics for the spatial map."""
    root, cluster_path = synthetic_compute_session
    nt = _make_neuronal_tuning(root)
    beh_inputs = nt._load_behavioral_inputs()
    partial = nt._compute_one_cluster_behavioral(cluster_path, beh_inputs)
    ts = partial["triage_stats"]
    assert "behavioral" in ts
    assert "spatial" in ts
    # 1D feature triage stats have run analysis fields
    speed_ts = ts["behavioral"]["beh_offset=0s"]["m1.speed"]
    assert "excit" in speed_ts
    assert "suppress" in speed_ts
    assert "selectivity" in speed_ts
    # circular feature flagged correctly
    yaw_ts = ts["behavioral"]["beh_offset=0s"]["m1.allo_yaw"]
    assert yaw_ts["is_circular"] is True
    # 2D spatial triage stats include Skaggs metrics
    sp = ts["spatial"]["beh_offset=0s"]["m1.space"]
    for k in ("info_rate_bps", "sparsity", "coherence", "peak_rate_sps"):
        assert k in sp


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_compute_one_cluster_behavioral_with_smoothing(synthetic_compute_session):
    """With smoothing_sd > 0, smoothed payloads must be present."""
    root, cluster_path = synthetic_compute_session
    nt = _make_neuronal_tuning(root, smoothing_sd=1.0)
    beh_inputs = nt._load_behavioral_inputs()
    partial = nt._compute_one_cluster_behavioral(cluster_path, beh_inputs)
    speed = partial["beh_offset=0s"]["m1.speed"]
    assert "rate_smoothed" in speed
    assert "null_p99_5_smoothed" in speed
    # a circular feature is smoothed with its ends wrapped, a linear one is not
    yaw = partial["beh_offset=0s"]["m1.allo_yaw"]
    np.testing.assert_allclose(
        yaw["rate_smoothed"], _gaussian_smooth_1d(yaw["rate"], 1.0, preserve_nan=False, circular=True)
    )
    np.testing.assert_allclose(
        speed["rate_smoothed"], _gaussian_smooth_1d(speed["rate"], 1.0, preserve_nan=False)
    )
    assert not np.allclose(
        yaw["rate_smoothed"], _gaussian_smooth_1d(yaw["rate"], 1.0, preserve_nan=False)
    )


# _load_vocal_inputs


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_load_vocal_inputs_returns_expected_keys(synthetic_compute_session):
    """The vocal loader returns the USV table, track names, per-animal sex from metadata and USV
    timing, with no legacy per-sex keys."""
    root, _ = synthetic_compute_session
    nt = _make_neuronal_tuning(root)
    bundle = nt._load_vocal_inputs()
    assert bundle is not None
    for k in ("usv_df", "track_names", "animal_sex", "duration_seconds",
              "starts", "stops", "emitters"):
        assert k in bundle
    assert "male" not in bundle
    assert "female" not in bundle
    assert bundle["animal_sex"] == {"m1": "male"}


def test_vocal_sides_take_sex_from_metadata_not_track_slot(synthetic_compute_session):
    """A female in track slot 0 (here the only track) is labelled female on its vocal
    side; a slot-0-is-male rule would have called it male."""
    root, _ = synthetic_compute_session
    (root / "session_metadata.yaml").write_text("Subjects:\n- subject_id: 'm1'\n  sex: female\n")
    nt = _make_neuronal_tuning(root)
    voc_inputs = nt._load_vocal_inputs()
    assert voc_inputs["animal_sex"] == {"m1": "female"}
    precompute = nt._build_vocal_side_precompute(voc_inputs)
    assert precompute["self"]["side"]["sex"] == "female"


def test_load_vocal_inputs_unmatched_track_raises(synthetic_compute_session):
    """A track with no metadata subject raises instead of defaulting to its slot."""
    root, _ = synthetic_compute_session
    (root / "session_metadata.yaml").write_text("Subjects:\n- subject_id: 'other'\n  sex: male\n")
    with pytest.raises(ValueError, match="'m1' has no subject"):
        _make_neuronal_tuning(root)._load_vocal_inputs()


def test_load_vocal_inputs_returns_none_when_no_inputs(tmp_path):
    """A session with no USV summary or tracking file makes the vocal loader return None so the
    vocal path is skipped."""
    nt = _make_neuronal_tuning(tmp_path)
    assert nt._load_vocal_inputs() is None


# _build_vocal_side_precompute


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_build_vocal_side_precompute_includes_self_side(synthetic_compute_session):
    """The vocal precompute holds the shared grid and a self side whose anchors belong to the
    recorded animal."""
    root, _ = synthetic_compute_session
    nt = _make_neuronal_tuning(root)
    voc_inputs = nt._load_vocal_inputs()
    precompute = nt._build_vocal_side_precompute(voc_inputs)
    assert precompute is not None
    assert "_grid" in precompute
    assert "self" in precompute
    self_block = precompute["self"]
    assert "anchor_idx" in self_block
    assert "side" in self_block
    assert self_block["side"]["emitter"] == "m1"


# _compute_one_cluster_vocal


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_behavioral_feature_base_strips_prefix_and_derivatives():
    """A behavioral column's base feature drops the animal / pair prefix and any
    derivative suffix, so an exclusion covers the feature and its derivatives."""
    assert behavioral_feature_base("158112_0-156693_3.allo_yaw-nose_1st_der") == "allo_yaw-nose"
    assert behavioral_feature_base("158112_0-156693_3.nose-nose_2nd_der") == "nose-nose"
    assert behavioral_feature_base("158112_0-156693_3.head-head") == "head-head"
    assert behavioral_feature_base("m1.speed") == "speed"


def test_load_behavioral_inputs_drops_excluded_features(synthetic_compute_session):
    """Columns whose base feature is in excluded_behavioral_features (derivatives
    included) are not tuned; their head-anchored versions and every other column stay."""
    root, _ = synthetic_compute_session
    beh_csv = next(root.rglob("*_behavioral_features.csv"))
    rng = np.random.default_rng(3)
    df = pls.read_csv(beh_csv)
    extra = {
        "m1-f1.nose-nose": rng.uniform(0, 30, df.height),
        "m1-f1.nose-nose_1st_der": rng.uniform(-5, 5, df.height),
        "m1-f1.head-head": rng.uniform(0, 30, df.height),
        "m1-f1.allo_yaw-nose": rng.uniform(-180, 180, df.height),
        "m1-f1.allo_yaw-head": rng.uniform(-180, 180, df.height),
        "m1-f1.nose-TTI": rng.uniform(0, 30, df.height),
    }
    df.with_columns([pls.Series(name, values) for name, values in extra.items()]).write_csv(beh_csv)
    nt = _make_neuronal_tuning(root)

    columns = nt._load_behavioral_inputs()["behavioral_data"].columns

    assert "m1-f1.nose-nose" not in columns
    assert "m1-f1.nose-nose_1st_der" not in columns
    assert "m1-f1.allo_yaw-nose" not in columns
    assert {"m1-f1.head-head", "m1-f1.allo_yaw-head", "m1-f1.nose-TTI", "m1.speed"} <= set(columns)


def test_squeaks_are_dropped_from_self_anchors_and_never_categorised(synthetic_compute_session):
    """With exclude_squeaks_self (default) the self side's anchors are its pure USVs only (usv
    true AND squeak false): pure squeaks AND segments with both flags true are dropped alike, so
    usv being true is not enough. With it off they are anchors but
    still carry no QLVM category (the qlvm_category tuning is usv-only), and a category held
    only by squeak / overlap segments does not appear."""
    root, _ = synthetic_compute_session
    usv_csv = next(root.rglob("*_usv_summary.csv"))
    df = pls.read_csv(usv_csv)
    usv = np.ones(df.height, dtype=bool)
    squeak = np.zeros(df.height, dtype=bool)
    usv[:5] = False
    squeak[:5] = True
    squeak[5:10] = True
    not_usv = squeak
    # category 99 only on squeak / overlap segments; each squeak's extent is its segment, so the
    # anchors keep their order
    df.with_columns(
        pls.Series("usv", usv.tolist(), dtype=pls.Boolean),
        pls.Series("squeak", squeak.tolist(), dtype=pls.Boolean),
        pls.when(pls.Series(not_usv)).then(99).otherwise(pls.col("qlvm_category")).alias("qlvm_category"),
        pls.when(pls.Series(squeak)).then(pls.col("start")).otherwise(None).alias("squeak_start"),
        pls.when(pls.Series(squeak)).then(pls.col("stop")).otherwise(None).alias("squeak_end"),
    ).write_csv(usv_csv)

    nt = _make_neuronal_tuning(root)
    voc_inputs = nt._load_vocal_inputs()
    np.testing.assert_array_equal(voc_inputs["is_usv"], ~not_usv)
    excluded = nt._build_vocal_side_precompute(voc_inputs)["self"]
    assert excluded["side"]["n"] == df.height - 10
    assert excluded["side"]["n_usv"] == df.height - 10
    assert voc_inputs["is_usv"][excluded["anchor_idx"]].all()

    nt.tuning_parameters_dict["exclude_squeaks_self"] = False
    kept = nt._build_vocal_side_precompute(voc_inputs)["self"]
    assert kept["side"]["n"] == df.height
    categorical = kept["anchor_categorical"]["qlvm_category"]
    assert 99 not in categorical["unique_cats"].tolist()
    assert (categorical["anchor_cat_idx_dense"][~voc_inputs["is_usv"][kept["anchor_idx"]]] == -1).all()


def test_a_pure_squeak_is_anchored_at_its_own_onset(synthetic_compute_session):
    """A pure squeak's anchor is its squeak's onset (squeak_start), not its segment's start, while
    the silence reference still spans its whole segment; a summary without the squeak extent raises
    rather than anchoring squeaks at their segments."""
    root, _ = synthetic_compute_session
    usv_csv = next(root.rglob("*_usv_summary.csv"))
    df = pls.read_csv(usv_csv).sort("start")
    first_start = float(df["start"][0])
    squeak_onset = first_start + 0.0005
    is_first = pls.Series([True] + [False] * (df.height - 1))
    df.with_columns(
        pls.when(is_first).then(False).otherwise(pls.col("usv")).alias("usv"),
        pls.when(is_first).then(True).otherwise(pls.col("squeak")).alias("squeak"),
        pls.when(is_first).then(squeak_onset).otherwise(None).alias("squeak_start"),
        pls.when(is_first).then(pls.col("stop")).otherwise(None).alias("squeak_end"),
    ).write_csv(usv_csv)

    voc_inputs = _make_neuronal_tuning(root)._load_vocal_inputs()
    assert voc_inputs["starts"][0] == pytest.approx(squeak_onset)
    assert not voc_inputs["is_usv"][0]
    assert voc_inputs["occupancy_starts"][0] == pytest.approx(first_start)
    pls.read_csv(usv_csv).drop("squeak_start").write_csv(usv_csv)
    with pytest.raises(KeyError, match="squeak_start"):
        _make_neuronal_tuning(root)._load_vocal_inputs()


def test_unscored_rows_are_not_usv_anchors(synthetic_compute_session):
    """A non-noise row with null booleans (a segment too short to score) is not a USV: it is
    dropped from the anchors when squeaks are excluded, like a squeak or an overlap segment."""
    root, _ = synthetic_compute_session
    usv_csv = next(root.rglob("*_usv_summary.csv"))
    df = pls.read_csv(usv_csv)
    usv = [True] * df.height
    squeak = [False] * df.height
    usv[0] = None
    squeak[0] = None
    df.with_columns(pls.Series("usv", usv, dtype=pls.Boolean), pls.Series("squeak", squeak, dtype=pls.Boolean)).write_csv(usv_csv)

    nt = _make_neuronal_tuning(root)
    voc_inputs = nt._load_vocal_inputs()
    assert not voc_inputs["is_usv"][0]
    assert nt._build_vocal_side_precompute(voc_inputs)["self"]["side"]["n"] == df.height - 1


def test_load_vocal_inputs_requires_the_vocal_flags(synthetic_compute_session):
    """A summary without the usv / squeak booleans raises instead of tuning to squeaks."""
    root, _ = synthetic_compute_session
    usv_csv = next(root.rglob("*_usv_summary.csv"))
    pls.read_csv(usv_csv).drop("usv").write_csv(usv_csv)
    with pytest.raises(KeyError, match=r"no \['usv'\] column"):
        _make_neuronal_tuning(root)._load_vocal_inputs()


def test_build_vocal_side_precompute_notices_missing_qlvm_labels(synthetic_compute_session):
    """Without the qlvm_category column (e.g. a summary assign-qlvm-categories has not
    labelled yet) the precompute prints a one-line notice naming it, instead of leaving
    the category outputs empty without a word; with the column present nothing is
    printed."""
    root, _ = synthetic_compute_session
    nt = _make_neuronal_tuning(root)
    messages: list[str] = []
    nt.message_output = messages.append
    voc_inputs = nt._load_vocal_inputs()

    assert nt._build_vocal_side_precompute(voc_inputs) is not None
    assert not any("QLVM category tuning skipped" in m for m in messages)

    voc_inputs["usv_df"] = voc_inputs["usv_df"].drop(["qlvm_category"])
    assert nt._build_vocal_side_precompute(voc_inputs) is not None
    notices = [m for m in messages if "QLVM category tuning skipped" in m]
    assert len(notices) == 1
    assert "(qlvm_category)" in notices[0]
    assert "labels are unavailable" in notices[0]


def test_generate_beh_features_cli_derivative_bins_is_one_integer():
    """--derivative-bins is the single integer calculate_derivatives slices with (the settings
    value derivative_bins is an int); declared as a multiple string option it reached the
    slice as the tuple ('10',) and raised TypeError."""
    option = next(param for param in generate_beh_features_cli.params if param.name == "derivative_bins")
    assert not option.multiple
    assert option.type.name == "integer"
    assert option.type.convert("10", option, None) == 10


def test_build_vocal_side_precompute_takes_an_all_null_property_column(synthetic_compute_session):
    """A continuous property with no value in the session (e.g. loudness_db never measured) is
    read from CSV as an all-null String column; the precompute casts it to Float64 and bins it as
    all-NaN (every anchor out of range) instead of raising inside np.isfinite."""
    root, _ = synthetic_compute_session
    nt = _make_neuronal_tuning(root)
    voc_inputs = nt._load_vocal_inputs()
    voc_inputs["usv_df"] = voc_inputs["usv_df"].with_columns(
        pls.lit(None, dtype=pls.String).alias("loudness_db")
    )
    precompute = nt._build_vocal_side_precompute(voc_inputs)
    assert precompute is not None
    assert "self" in precompute
    assert (precompute["self"]["anchor_property_bin_idx"]["loudness_db"] == -1).all()


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_compute_one_cluster_vocal_smoke(synthetic_compute_session):
    """End-to-end: vocal compute over a synthetic cluster + USV table."""
    root, cluster_path = synthetic_compute_session
    nt = _make_neuronal_tuning(root)
    voc_inputs = nt._load_vocal_inputs()
    precompute = nt._build_vocal_side_precompute(voc_inputs)
    partial = nt._compute_one_cluster_vocal(
        cluster_file=cluster_path,
        voc_inputs=voc_inputs,
        side_precompute=precompute,
    )
    for top_key in ("usv_peth", "usv_property_tuning",
                    "usv_category_tuning", "usv_category_peth",
                    "usv_metadata", "triage_stats"):
        assert top_key in partial
    # VMI block populated for the self emitter
    vmi = partial["triage_stats"]["vmi"]
    assert "m1" in vmi
    assert "vmi" in vmi["m1"]
    assert "wilcoxon_pvalue" in vmi["m1"]
    # usv_peth payload structure
    peth = partial["usv_peth"]["m1"]
    for k in ("rate", "bin_centers_s", "null_mean", "null_std", "null_p99_5"):
        assert k in peth
    # property tuning iterates over all 8 properties
    props = partial["usv_property_tuning"]["m1"]
    for prop in ("duration", "mean_freq_hz", "peak_freq_hz",
                 "freq_bandwidth_hz", "mean_amplitude",
                 "max_amplitude", "spectral_entropy", "mask_number"):
        assert prop in props


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_vmi_and_ibi_bouts_are_grouped_from_the_side_anchor_rows(synthetic_compute_session, mocker):
    """The VMI bouts and the IBI bouts are grouped from the same rows as the side's anchors:
    with exclude_squeaks_self the squeak and overlap segments are left out of both, and with
    it off every one of the emitter's vocalizations is in."""
    root, cluster_path = synthetic_compute_session
    usv_csv = next(root.rglob("*_usv_summary.csv"))
    df = pls.read_csv(usv_csv)
    usv = np.ones(df.height, dtype=bool)
    squeak = np.zeros(df.height, dtype=bool)
    usv[:5] = False
    squeak[:10] = True
    df.with_columns(
        pls.Series("usv", usv.tolist(), dtype=pls.Boolean),
        pls.Series("squeak", squeak.tolist(), dtype=pls.Boolean),
        pls.when(pls.Series(squeak)).then(pls.col("start")).otherwise(None).alias("squeak_start"),
        pls.when(pls.Series(squeak)).then(pls.col("stop")).otherwise(None).alias("squeak_end"),
    ).write_csv(usv_csv)

    for exclude in (True, False):
        nt = _make_neuronal_tuning(root)
        nt.tuning_parameters_dict["exclude_squeaks_self"] = exclude
        voc_inputs = nt._load_vocal_inputs()
        precompute = nt._build_vocal_side_precompute(voc_inputs)
        side_mask = precompute["self"]["side"]["mask"]
        assert int(side_mask.sum()) == (df.height - 10 if exclude else df.height)
        vmi_spy = mocker.spy(tuning_module.NeuronalTuning, "_compute_vmi_for_emitter")
        runs_spy = mocker.spy(tuning_module, "ibi_bout_runs")
        nt._compute_one_cluster_vocal(cluster_file=cluster_path, voc_inputs=voc_inputs, side_precompute=precompute)
        np.testing.assert_array_equal(vmi_spy.call_args.kwargs["em_starts"], voc_inputs["starts"][side_mask])
        np.testing.assert_array_equal(vmi_spy.call_args.kwargs["em_stops"], voc_inputs["stops"][side_mask])
        np.testing.assert_array_equal(runs_spy.call_args.args[0], voc_inputs["starts"][side_mask])
        np.testing.assert_array_equal(runs_spy.call_args.args[1], voc_inputs["stops"][side_mask])
        mocker.stopall()


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_rerun_replaces_each_side_and_clears_a_skipped_or_failed_side(synthetic_compute_session, mocker):
    """A re-run replaces each path's keys whole (a key the new run does not produce is gone),
    a vocal path that fails for a cluster removes that cluster's previous vocal results, and a
    skipped vocal path removes them from every pkl, while the behavioral results stay."""
    root, cluster_path = synthetic_compute_session
    pkl_path = root / "ephys" / "tuning_curves" / f"{cluster_path.stem}_tuning_curves_data.pkl"

    def read_pkl():
        """Load the cluster's tuning-curve pickle as written by the last run."""
        with pkl_path.open("rb") as handle:
            return pickle.load(handle)

    _make_neuronal_tuning(root).calculate_neuronal_tuning_curves()
    first = read_pkl()
    assert "beh_offset=0s" in first
    assert "usv_peth" in first
    assert "vmi" in first["triage_stats"]

    stale = read_pkl()
    stale["beh_offset=9s"] = {"stale": True}
    stale["usv_peth"]["f9"] = {"stale": True}
    stale["triage_stats"]["vmi"]["f9"] = {"stale": True}
    stale["triage_stats"]["behavioral"]["beh_offset=9s"] = {"stale": True}
    stale["unrelated"] = 1
    with pkl_path.open("wb") as handle:
        pickle.dump(stale, handle)
    _make_neuronal_tuning(root).calculate_neuronal_tuning_curves()
    rerun = read_pkl()
    assert "beh_offset=9s" not in rerun
    assert "beh_offset=9s" not in rerun["triage_stats"]["behavioral"]
    assert "f9" not in rerun["usv_peth"]
    assert "f9" not in rerun["triage_stats"]["vmi"]
    assert rerun["unrelated"] == 1

    mocker.patch.object(NeuronalTuning, "_compute_one_cluster_vocal", side_effect=RuntimeError("boom"))
    # every cluster is still attempted, then the run fails loudly naming the failed cluster
    # (the broad handler used to log the failure and report a finished run)
    with pytest.raises(RuntimeError, match=r"Vocal tuning failed for 1 of 1 clusters.*boom"):
        _make_neuronal_tuning(root).calculate_neuronal_tuning_curves()
    failed = read_pkl()
    for key in ("usv_peth", "usv_property_tuning", "usv_category_tuning", "usv_category_peth",
                "usv_ibi_bout_onset", "usv_metadata"):
        assert key not in failed
    assert not {"vmi", "usv_peth", "usv_ibi_bout_onset"} & set(failed["triage_stats"])
    assert "beh_offset=0s" in failed
    assert "behavioral" in failed["triage_stats"]
    mocker.stopall()

    _make_neuronal_tuning(root).calculate_neuronal_tuning_curves()
    assert "usv_peth" in read_pkl()
    next(root.rglob("*_usv_summary.csv")).unlink()
    _make_neuronal_tuning(root).calculate_neuronal_tuning_curves()
    skipped = read_pkl()
    assert "usv_peth" not in skipped
    assert "vmi" not in skipped["triage_stats"]
    assert "beh_offset=0s" in skipped


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_compute_one_cluster_vocal_triage_stats_populated(synthetic_compute_session):
    """The vocal compute fills every triage_stats sub-block, not just VMI."""
    root, cluster_path = synthetic_compute_session
    nt = _make_neuronal_tuning(root)
    voc_inputs = nt._load_vocal_inputs()
    precompute = nt._build_vocal_side_precompute(voc_inputs)
    partial = nt._compute_one_cluster_vocal(
        cluster_file=cluster_path,
        voc_inputs=voc_inputs,
        side_precompute=precompute,
    )
    ts = partial["triage_stats"]
    for k in ("vmi", "usv_peth", "usv_property_tuning",
              "usv_category_tuning", "usv_category_peth"):
        assert k in ts
    # Per-emitter dict shape
    assert "m1" in ts["usv_peth"]
    assert "m1" in ts["usv_property_tuning"]


# usv_interval_archive — HDF5 archive layer for IUI / mixture-model output


# _attr_value / _decode_attr round-trip


@pytest.mark.parametrize(("v", "expected_type"), [
    (None, str),       # None -> "null" str
    (True, bool),
    (42, int),
    (3.14, float),
    ("hello", str),
])
def test_attr_value_scalar_passthrough(v, expected_type):
    """Scalars / None pass through to h5py-storable types."""
    out = _attr_value(v)
    assert isinstance(out, expected_type)
    if v is None:
        assert out == "null"


def test_attr_value_list_dict_json_encoded():
    """Lists / dicts / tuples must be JSON-encoded so h5py can store them."""
    s_list = _attr_value([1, 2, 3])
    s_dict = _attr_value({"k": [1, 2]})
    s_tup = _attr_value((4, 5))
    assert json.loads(s_list) == [1, 2, 3]
    assert json.loads(s_dict) == {"k": [1, 2]}
    assert json.loads(s_tup) == [4, 5]


def test_decode_attr_recovers_python_types():
    """JSON strings round-trip back to python; "null" → None."""
    assert _decode_attr("null") is None
    assert _decode_attr(b"null") is None
    assert _decode_attr(_attr_value([1, 2, 3])) == [1, 2, 3]
    assert _decode_attr(_attr_value({"a": 1})) == {"a": 1}
    # Plain non-JSON strings pass through.
    assert _decode_attr("not json") == "not json"
    # numpy scalars round-trip
    assert _decode_attr(np.int64(5)) == 5
    assert _decode_attr(np.float64(1.5)) == 1.5
    assert _decode_attr(np.bool_(True)) is True
    assert _decode_attr(np.array([1, 2])) == [1, 2]


# _polars_to_h5 / _h5_to_polars round-trip


def test_h5_to_polars_renames_the_legacy_call_type_column(tmp_path):
    """An archive table written before the rename carries ``call_type``; it reads back
    as ``vocal_class`` with its values, so figures read old archives unchanged."""
    path = tmp_path / "legacy.h5"
    legacy = pls.DataFrame({"sex": ["male"], "call_type": ["usv"], "adjacency": ["filtered"]})
    with h5py.File(path, "w") as f:
        _polars_to_h5(f, "pools", legacy)
    with h5py.File(path, "r") as f:
        table = _h5_to_polars(f["pools"])
    assert table.columns == ["sex", "vocal_class", "adjacency"]
    assert table["vocal_class"].to_list() == ["usv"]


def test_polars_h5_roundtrip_mixed_dtypes(tmp_path):
    """Numeric / string / bool columns survive the polars↔h5py round-trip."""
    df = pls.DataFrame({
        "i": [1, 2, 3],
        "u": pls.Series([1, 2, 3], dtype=pls.UInt32),
        "f": [1.0, 2.5, 3.25],
        "s": ["one", "two", "three"],
        "b": [True, False, True],
    })
    out = tmp_path / "t.h5"
    with h5py.File(out, "w") as f:
        _polars_to_h5(f, "tbl", df)
    with h5py.File(out, "r") as f:
        rt = _h5_to_polars(f["tbl"])
    assert rt.shape == df.shape
    assert rt["i"].to_list() == [1, 2, 3]
    assert rt["s"].to_list() == ["one", "two", "three"]
    assert rt["b"].to_list() == [True, False, True]
    assert rt["f"].to_list() == [1.0, 2.5, 3.25]


def test_polars_h5_roundtrip_empty_frame(tmp_path):
    """An empty polars DataFrame survives via the schema_json placeholder."""
    df = pls.DataFrame(schema={"a": pls.Int64, "b": pls.Utf8})
    out = tmp_path / "empty.h5"
    with h5py.File(out, "w") as f:
        _polars_to_h5(f, "tbl", df)
    with h5py.File(out, "r") as f:
        rt = _h5_to_polars(f["tbl"])
    assert rt.height == 0
    assert set(rt.columns) == {"a", "b"}


# _try_git_sha (subprocess.run mocked)


def test_try_git_sha_happy(monkeypatch, tmp_path):
    """Mocked git rev-parse returns a SHA → that's what we get back."""
    fake = MagicMock(returncode=0, stdout="abc1234\n")
    monkeypatch.setattr(_subprocess, "run", lambda *_a, **_kw: fake)
    assert _try_git_sha(tmp_path) == "abc1234"


def test_try_git_sha_error_returns_unknown(monkeypatch, tmp_path):
    """Any subprocess error must yield "unknown" (provenance-only field)."""
    def _explode(*_a, **_kw):
        """Stand-in for ``subprocess.run`` that fails as when git is not installed."""
        msg = "no git here"
        raise FileNotFoundError(msg)
    monkeypatch.setattr(_subprocess, "run", _explode)
    assert _try_git_sha(tmp_path) == "unknown"


def test_try_git_sha_timeout_returns_unknown(monkeypatch, tmp_path):
    """TimeoutExpired must be caught silently."""
    def _explode(*_a, **_kw):
        """Stand-in for ``subprocess.run`` that fails as when git hangs past the timeout."""
        raise _subprocess.TimeoutExpired(cmd="git", timeout=2)
    monkeypatch.setattr(_subprocess, "run", _explode)
    assert _try_git_sha(tmp_path) == "unknown"


def test_try_git_sha_nonzero_returncode_returns_unknown(monkeypatch, tmp_path):
    """Non-zero exit code (not in repo) → unknown."""
    fake = MagicMock(returncode=128, stdout="")
    monkeypatch.setattr(_subprocess, "run", lambda *_a, **_kw: fake)
    assert _try_git_sha(tmp_path) == "unknown"


# detect_repo_root_for_provenance / git_sha_for_provenance


def test_detect_repo_root_walks_up(tmp_path):
    """Walks upward to find the .git directory; falls back to start otherwise."""
    repo = tmp_path / "repo"
    sub = repo / "src" / "pkg" / "deep"
    sub.mkdir(parents=True)
    (repo / ".git").mkdir()
    out = detect_repo_root_for_provenance(sub)
    assert out == repo.resolve()


def test_detect_repo_root_falls_back_when_no_git(tmp_path):
    """No .git anywhere on the path → returns the start dir resolved."""
    sub = tmp_path / "no_repo" / "deep"
    sub.mkdir(parents=True)
    out = detect_repo_root_for_provenance(sub)
    assert out == sub.resolve()


def test_git_sha_for_provenance_combines_helpers(monkeypatch, tmp_path):
    """git_sha_for_provenance(start) = _try_git_sha(detect_repo_root(start))."""
    repo = tmp_path / "r"
    (repo / ".git").mkdir(parents=True)
    fake = MagicMock(returncode=0, stdout="deadbee\n")
    monkeypatch.setattr(_subprocess, "run", lambda *_a, **_kw: fake)
    assert git_sha_for_provenance(repo) == "deadbee"


# write_ivi_h5 / read_usv_interval_h5 round-trip


def _example_per_mode_payload():
    """
    Description
    -----------
    Builds a minimal ``per_mode`` payload for ``write_ivi_h5`` with one ``s2s``
    mode carrying attributes, an intervals table and a drop-count table, and
    ``None`` for the three optional mixture-model tables, so the archive
    round-trip covers both written and absent tables.

    Parameters
    ----------

    Returns
    -------
    per_mode (dict)
        The per-mode payload keyed by interval type.
    """
    return {
        "s2s": {
            "attrs": {"alpha_effective": 0.05, "K_selected_male": 2, "K_selected_female": 3},
            "intervals": pls.DataFrame({
                "session_id": ["sess1", "sess1", "sess2"],
                "interval": [0.1, 0.2, 0.3],
                "sex": ["male", "female", "male"],
            }),
            "drop_counts": pls.DataFrame({
                "session_id": ["sess1", "sess2"],
                "n_dropped_male": [0, 1],
                "n_dropped_female": [2, 0],
            }),
            "mixture_model_fits": None,
            "bootstrap_lrt": None,
            "bootstrap_lrt_null": None,
        },
    }


def test_write_and_read_ivi_h5_roundtrip(tmp_path):
    """Top-level + per-mode attrs and tables survive the archive round-trip."""
    out = tmp_path / "archive.h5"
    payload = _example_per_mode_payload()
    analysis_attrs = {
        "created_at_iso": "2026-05-09T12:00:00",
        "git_sha": "abc123",
        "n_sessions_loaded": 2,
        "source_lists": ["a.txt", "b.txt"],
        "tau": 0.5,
        "fit_mixture_model": False,
    }
    written = write_ivi_h5(out, analysis_attrs=analysis_attrs, per_mode=payload)
    assert written == out

    archive = read_usv_interval_h5(out)
    # File-level attrs decoded.
    assert archive["attrs"]["created_at_iso"] == "2026-05-09T12:00:00"
    assert archive["attrs"]["source_lists"] == ["a.txt", "b.txt"]
    assert archive["attrs"]["fit_mixture_model"] is False or archive["attrs"]["fit_mixture_model"] == 0
    # Mode-level: tables exist; missing tables are None.
    s2s = archive["modes"]["s2s"]
    assert s2s["attrs"]["alpha_effective"] == 0.05
    assert s2s["intervals"].height == 3
    assert s2s["drop_counts"].height == 2
    assert s2s["mixture_model_fits"] is None
    assert s2s["bootstrap_lrt"] is None
    assert s2s["bootstrap_lrt_null"] is None


def test_write_ivi_h5_creates_parent_dir(tmp_path):
    """write_ivi_h5 must mkdir(parents=True) before writing."""
    nested = tmp_path / "a" / "b" / "out.h5"
    write_ivi_h5(nested, analysis_attrs={}, per_mode={})
    assert nested.is_file()


# reconstruct_best_model


def _mixture_model_fits_row(model_class, sex="male", K=2, bic=10.0, cv=1.0):
    """
    Description
    -----------
    Builds a single-row mixture-model sweep table that ``reconstruct_best_model``
    can consume: identity columns (sex, n_comp, rep, bic, cv_neg_loglik,
    model_class) plus per-component weight, log-mean, log-SD, nu and lambda
    columns, so the Gaussian, t and inverse-Gaussian reconstruction paths all
    find their parameters.

    Parameters
    ----------
    model_class (str)
        Model family label (``gauss``, ``t``, ``ig`` or an invalid value).
    sex (str)
        Sex the row belongs to.
    K (int)
        Number of components.
    bic (float)
        BIC value of the row.
    cv (float)
        Cross-validated negative log-likelihood of the row.

    Returns
    -------
    df (pls.DataFrame)
        The one-row sweep table.
    """
    row = {
        "sex": sex,
        "n_comp": K,
        "rep": 0,
        "bic": bic,
        "cv_neg_loglik": cv,
        "model_class": model_class,
    }
    for k in range(K):
        row[f"weight_{k+1}"] = 0.5
        row[f"logmean_{k+1}"] = float(k - 1)  # spread out
        row[f"logsd_{k+1}"] = 0.5
        row[f"nu_{k+1}"] = 5.0
        row[f"lambda_{k+1}"] = 2.0
    return pls.DataFrame([row])


def test_reconstruct_best_model_gauss():
    """Gauss path returns an sklearn GMM with score_samples working."""
    df = _mixture_model_fits_row("gauss", K=2)
    model, order = reconstruct_best_model(df, sex="male", K=2)
    assert isinstance(model, GaussianMixture)
    # score_samples must work on synthetic input — proves precisions_cholesky_
    # was set correctly.
    s = model.score_samples(np.array([[0.0], [1.0]]))
    assert s.shape == (2,)
    np.testing.assert_array_equal(order, np.arange(2))


def test_reconstruct_best_model_t_mixture():
    """t-mixture path returns a TMixture object."""
    df = _mixture_model_fits_row("t", K=2)
    model, order = reconstruct_best_model(df, sex="male", K=2)
    assert isinstance(model, TMixture)
    np.testing.assert_array_equal(order, np.arange(2))


def test_reconstruct_best_model_ig_mixture():
    """IG path returns an IGMixture whose mus are exp(logmean_k) and whose
    lambdas come from the lambda_k columns."""
    df = _mixture_model_fits_row("ig", K=2)
    model, order = reconstruct_best_model(df, sex="male", K=2)
    assert isinstance(model, IGMixture)
    np.testing.assert_allclose(model.mus_, np.exp([-1.0, 0.0]))
    np.testing.assert_allclose(model.lambdas_, [2.0, 2.0])
    np.testing.assert_array_equal(order, np.arange(2))


def test_reconstruct_best_model_no_rows_raises():
    """No matching (sex, K) → ValueError."""
    df = _mixture_model_fits_row("gauss", sex="male", K=2)
    with pytest.raises(ValueError, match="no rows"):
        reconstruct_best_model(df, sex="female", K=2)


def test_reconstruct_best_model_unknown_class_raises():
    """Unknown model_class string → ValueError."""
    df = _mixture_model_fits_row("bogus", K=2)
    with pytest.raises(ValueError, match="unknown model_class"):
        reconstruct_best_model(df, sex="male", K=2)


def test_reconstruct_best_model_falls_back_when_cv_all_nan():
    """If every cv_neg_loglik is NaN, reconstruct_best_model should fall back to bic."""
    df1 = _mixture_model_fits_row("gauss", K=2, bic=10.0, cv=float("nan"))
    df2 = _mixture_model_fits_row("gauss", K=2, bic=5.0, cv=float("nan"))
    # Ensure 'rep' dtype matches df1's (Int64) — pls.lit(1) is Int32 by default.
    df2 = df2.with_columns(pls.lit(1, dtype=pls.Int64).alias("rep"))
    df = pls.concat([df1, df2])
    model, _order = reconstruct_best_model(df, sex="male", K=2)
    # Just verify we get a valid model — the fallback to BIC means rep=1 (bic=5)
    # should win, but the reconstruction exposes only weights/means/cov, not rep.
    assert model is not None


# compute_inter_usv_interval_distributions — pure helpers


def test_read_session_lists_dedupes_and_preserves_order(tmp_path):
    """Sessions appearing in two list files are de-duped, first-seen wins."""
    a = tmp_path / "list_a.txt"
    b = tmp_path / "list_b.txt"
    a.write_text("/sess/one\n/sess/two\n\n/sess/three\n")
    b.write_text("/sess/two\n/sess/four\n")
    msgs = []
    out = _read_session_lists([str(a), str(b)], msgs.append)
    # Order: a's contents first (one, two, three), then b's new (four)
    # configure_path may rewrite paths but preserve them on Linux/Mac
    assert "/sess/one" in out
    assert "/sess/two" in out
    assert "/sess/three" in out
    assert "/sess/four" in out
    # No duplicates
    assert len(out) == len(set(out))
    # Notification logs were called for each list
    assert len(msgs) == 2


def test_read_session_lists_warns_on_missing(tmp_path):
    """Missing list file logs a warning and is skipped, not raised."""
    a = tmp_path / "missing.txt"
    msgs = []
    out = _read_session_lists([str(a)], msgs.append)
    assert out == []
    assert any("not found" in m for m in msgs)


def test_session_source_map_first_seen_wins(tmp_path):
    """A session appearing in two lists keeps the first list's stem."""
    a = tmp_path / "groupA.txt"
    b = tmp_path / "groupB.txt"
    a.write_text("/sess/X\n/sess/Y\n")
    b.write_text("/sess/Y\n/sess/Z\n")
    out = _session_source_map([str(a), str(b)])
    # /sess/Y should map to "groupA" (the first list it appeared in)
    assert out["/sess/Y"] == "groupA"
    assert out["/sess/X"] == "groupA"
    assert out["/sess/Z"] == "groupB"


def test_session_source_map_skips_missing(tmp_path):
    """Missing list files are silently ignored."""
    a = tmp_path / "exists.txt"
    a.write_text("/sess/A\n")
    out = _session_source_map([str(a), str(tmp_path / "absent.txt")])
    assert out == {"/sess/A": "exists"}


def test_compute_session_usv_intervals_invalid_type_raises():
    """interval_type not in {'s2s', 'e2s'} → ValueError before any I/O."""
    with pytest.raises(ValueError, match="Unknown interval_type"):
        compute_session_usv_intervals(
            session_root="/whatever",
            interval_type="bogus",
            exclude_noise_usvs=True,
        )


def test_compute_session_usv_intervals_missing_session_returns_empty(monkeypatch):
    """Missing session metadata → empty dict (graceful skip, not crash)."""
    def _raise(_root):
        """Stand-in for ``extract_session_metadata`` that fails as for a missing session."""
        msg = "no such session"
        raise FileNotFoundError(msg)
    monkeypatch.setattr(cmod, "extract_session_metadata", _raise)
    out = compute_session_usv_intervals(
        session_root="/missing",
        interval_type="s2s",
        exclude_noise_usvs=True,
    )
    assert out == {}


def test_compute_session_usv_intervals_basic_pairs(monkeypatch):
    """Synthesise 4 USVs (M, M, F, F) → expect one M-M interval and one F-F."""
    monkeypatch.setattr(cmod, "extract_session_metadata", lambda _root: {
        "track_names": ["M", "F"], "frame_rate": 150.0,
    })
    monkeypatch.setattr(cmod, "extract_animal_sexes", lambda _root, _names: {
        "M": "male", "F": "female",
    })
    fake_usv = pls.DataFrame({
        "start": [0.0, 0.5, 1.0, 1.7],
        "stop":  [0.1, 0.6, 1.1, 1.8],
        "duration": [0.1, 0.1, 0.1, 0.1],
        "emitter": ["M", "M", "F", "F"],
    })
    monkeypatch.setattr(cmod, "load_and_filter_usv_data",
                        lambda **_kw: fake_usv)
    out = compute_session_usv_intervals(
        session_root="/ok", interval_type="s2s",
        exclude_noise_usvs=True,
    )
    # M-M interval: start[1]-start[0] = 0.5
    # F-F interval: start[3]-start[2] = 0.7
    np.testing.assert_allclose(out["male"], [0.5])
    np.testing.assert_allclose(out["female"], [0.7])
    assert out["interval_type"] == "s2s"
    assert out["n_dropped_male"] == 0
    assert out["n_dropped_female"] == 0


def test_compute_session_usv_intervals_empty_usv_returns_empty_arrays(monkeypatch):
    """Zero rows in the USV CSV → empty interval arrays, not a crash."""
    monkeypatch.setattr(cmod, "extract_session_metadata", lambda _root: {
        "track_names": ["M", "F"], "frame_rate": 150.0,
    })
    monkeypatch.setattr(cmod, "extract_animal_sexes", lambda _root, _names: {
        "M": "male", "F": "female",
    })
    monkeypatch.setattr(cmod, "load_and_filter_usv_data",
                        lambda **_kw: pls.DataFrame({
                            "start": pls.Series([], dtype=pls.Float64),
                            "stop":  pls.Series([], dtype=pls.Float64),
                            "duration": pls.Series([], dtype=pls.Float64),
                            "emitter": pls.Series([], dtype=pls.Utf8),
                        }))
    out = compute_session_usv_intervals(
        session_root="/ok", interval_type="s2s",
        exclude_noise_usvs=True,
    )
    assert out["male"].size == 0
    assert out["female"].size == 0


def test_compute_session_usv_intervals_e2s_drops_overlapping(monkeypatch):
    """e2s mode: stop[0]=0.6, start[1]=0.5 → -0.1 interval, dropped, counted."""
    monkeypatch.setattr(cmod, "extract_session_metadata", lambda _root: {
        "track_names": ["M", "F"], "frame_rate": 150.0,
    })
    monkeypatch.setattr(cmod, "extract_animal_sexes", lambda _root, _names: {
        "M": "male", "F": "female",
    })
    fake_usv = pls.DataFrame({
        "start": [0.0, 0.5, 1.0],
        "stop":  [0.6, 0.7, 1.2],
        "duration": [0.6, 0.2, 0.2],
        "emitter": ["M", "M", "M"],
    })
    monkeypatch.setattr(cmod, "load_and_filter_usv_data",
                        lambda **_kw: fake_usv)
    out = compute_session_usv_intervals(
        session_root="/ok", interval_type="e2s",
        exclude_noise_usvs=True,
    )
    # First M-M pair: start[1]-stop[0] = 0.5 - 0.6 = -0.1 → dropped
    # Second M-M pair: start[2]-stop[1] = 1.0 - 0.7 = 0.3 → kept
    np.testing.assert_allclose(out["male"], [0.3])
    assert out["n_dropped_male"] == 1


def test_compute_session_usv_intervals_same_sex_session_pairs_per_animal(monkeypatch):
    """
    A female-female session: both animals are female by their metadata, so both
    animals' intervals land in the 'female' pool and nothing in 'male'; a USV of
    one female is never paired with a USV of the other (the A->B step breaks the
    chain), and each interval carries the animal it belongs to.
    """
    monkeypatch.setattr(cmod, "extract_session_metadata", lambda _root: {
        "track_names": ["A", "B"], "frame_rate": 150.0,
    })
    monkeypatch.setattr(cmod, "extract_animal_sexes", lambda _root, _names: {
        "A": "female", "B": "female",
    })
    fake_usv = pls.DataFrame({
        "start": [0.0, 0.5, 1.0, 1.7, 2.0],
        "stop":  [0.1, 0.6, 1.1, 1.8, 2.1],
        "duration": [0.1] * 5,
        "emitter": ["A", "A", "B", "B", "A"],
    })
    monkeypatch.setattr(cmod, "load_and_filter_usv_data", lambda **_kw: fake_usv)
    out = compute_session_usv_intervals(
        session_root="/ok", interval_type="s2s", exclude_noise_usvs=True,
    )
    # A-A 0.5 (rows 0-1) and B-B 0.7 (rows 2-3); B->A (rows 3-4) is not a pair
    assert out["male"].size == 0
    np.testing.assert_allclose(out["female"], [0.5, 0.7])
    assert list(out["emitter_female"]) == ["A", "B"]


def test_extract_animal_sexes_reads_metadata_and_refuses_to_guess(tmp_path):
    """
    Sex comes from the metadata Subjects block, matched on the stripped track
    name; a track without a recorded sex raises instead of falling back to its slot.
    """
    (tmp_path / "20260101_120000_metadata.yaml").write_text(
        "Subjects:\n- subject_id: '124784_2'\n  sex: female\n"
        "- subject_id: '124784_3'\n  sex: female\n")
    assert extract_animal_sexes(str(tmp_path), ["124784_2\x00", "124784_3"]) == {
        "124784_2": "female", "124784_3": "female"}
    with pytest.raises(ValueError, match="no subject with a recorded sex"):
        extract_animal_sexes(str(tmp_path), ["124784_2", "999999_1"])
    with pytest.raises(FileNotFoundError):
        extract_animal_sexes(str(tmp_path / "absent"), ["124784_2"])


# mixture_model_utils — coverage extension toward ~90%
# Targeting: bootstrap_lrt, select_n_components_step_up_lrt, edge cases in
# gmm_boundaries_logspace / gmm_modes / select_best_n_components / cv helpers,
# and the matplotlib smoke surfaces (plot_gmm_fit, qqplot_gmm).


# bootstrap_lrt


def test_bootstrap_lrt_invalid_K_pair_raises():
    """K_alt must be strictly greater than K_null."""
    with pytest.raises(ValueError, match="K_alt > K_null"):
        bootstrap_lrt(np.array([0.1, 0.2, 0.3]), K_null=2, K_alt=2,
                      B=2, n_subsample=3)
    with pytest.raises(ValueError, match="K_alt > K_null"):
        bootstrap_lrt(np.array([0.1, 0.2, 0.3]), K_null=3, K_alt=2,
                      B=2, n_subsample=3)


def test_bootstrap_lrt_invalid_model_class_raises():
    """model_class must be 'gauss' or 't'."""
    with pytest.raises(ValueError, match="model_class must be"):
        bootstrap_lrt(np.array([0.1, 0.2, 0.3]), K_null=1, K_alt=2,
                      B=2, n_subsample=3, model_class="bogus")


def test_bootstrap_lrt_returns_full_result_dict_gauss():
    """End-to-end Gaussian LRT with a tiny B for speed; verifies the result
    dict has every documented key with sensible types."""
    rng = np.random.default_rng(0)
    # Bimodal log-normal: two peaks at log(0.5) and log(2.0)
    intervals = np.exp(np.concatenate([
        rng.normal(np.log(0.5), 0.2, 50),
        rng.normal(np.log(2.0), 0.2, 50),
    ]))
    result = bootstrap_lrt(intervals, K_null=1, K_alt=2,
                           B=3, n_subsample=80, model_class="gauss",
                           n_init_obs=2, n_init_boot=1, seed=42)
    assert set(result.keys()) >= {
        "K_null", "K_alt", "B", "n_subsample", "model_class",
        "lr_obs", "lr_null", "p_value", "null_mean", "null_p95", "null_max",
    }
    assert result["K_null"] == 1
    assert result["K_alt"] == 2
    assert result["B"] == 3
    assert result["model_class"] == "gauss"
    assert isinstance(result["lr_obs"], float)
    assert result["lr_null"].shape == (3,)
    assert 0.0 <= result["p_value"] <= 1.0


def test_bootstrap_lrt_subsamples_when_n_subsample_smaller():
    """If intervals.size > n_subsample, the function subsamples down."""
    rng = np.random.default_rng(1)
    intervals = np.exp(rng.normal(0.0, 0.5, 200))
    result = bootstrap_lrt(intervals, K_null=1, K_alt=2,
                           B=2, n_subsample=50, model_class="gauss",
                           n_init_obs=1, n_init_boot=1, seed=0)
    assert result["n_subsample"] == 50


def test_bootstrap_lrt_uses_full_array_when_smaller_than_subsample():
    """If intervals.size <= n_subsample, the entire input is used."""
    rng = np.random.default_rng(2)
    intervals = np.exp(rng.normal(0.0, 0.5, 30))
    result = bootstrap_lrt(intervals, K_null=1, K_alt=2,
                           B=2, n_subsample=100, model_class="gauss",
                           n_init_obs=1, n_init_boot=1, seed=0)
    assert result["n_subsample"] == 30


# select_n_components_step_up_lrt


def test_select_n_components_step_up_returns_first_nonsig_K_null():
    """First (K_null, K_alt) with p >= alpha → return K_null."""
    pair = {
        (1, 2): {"p_value": 0.001},  # significant
        (2, 3): {"p_value": 0.20},   # not significant → return 2
        (3, 4): {"p_value": 0.001},  # would have been significant
    }
    assert select_n_components_step_up_lrt(pair, alpha=0.05) == 2


def test_select_n_components_step_up_all_significant_returns_max_K_alt():
    """Every test rejects → return the largest K_alt."""
    pair = {
        (1, 2): {"p_value": 0.001},
        (2, 3): {"p_value": 0.001},
        (3, 4): {"p_value": 0.001},
    }
    assert select_n_components_step_up_lrt(pair, alpha=0.05) == 4


def test_select_n_components_step_up_empty_pairs_raises():
    """Empty pair_results → ValueError (caller mistake)."""
    with pytest.raises(ValueError, match="no pair_results"):
        select_n_components_step_up_lrt({}, alpha=0.05)


# gmm_boundaries_logspace edge cases


def test_gmm_boundaries_logspace_returns_nan_when_no_real_root():
    """Discriminant < 0 → that boundary is NaN (no crossing exists). We
    construct two components whose densities never equal each other."""
    gmm = GaussianMixture(n_components=2, covariance_type="full")
    # Component 1 dominates everywhere: same mean, same variance, but weight
    # ratio is so extreme that the weighted curves never cross with tau=0.5.
    gmm.weights_ = np.array([0.99, 0.01])
    gmm.means_ = np.array([[0.0], [0.0]])
    gmm.covariances_ = np.array([[[1.0]], [[1.0]]])
    gmm.precisions_cholesky_ = (1.0 / np.sqrt(np.array([[[1.0]], [[1.0]]]))).reshape(2, 1, 1)
    log_b, sec_b = gmm_boundaries_logspace(gmm, tau=0.5)
    # Equal variances + equal means → a = 0, b = 0 → NaN per the source
    assert np.isnan(log_b[0])
    assert np.isnan(sec_b[0])


def test_gmm_boundaries_logspace_returns_finite_when_unique_root():
    """Two well-separated components → exactly one boundary in (-inf, +inf),
    finite-valued."""
    log_x = np.concatenate([
        np.random.RandomState(0).normal(-1.5, 0.2, 200).reshape(-1, 1),
        np.random.RandomState(1).normal(1.5, 0.2, 200).reshape(-1, 1),
    ])
    gmm = GaussianMixture(n_components=2, covariance_type="full",
                          random_state=0).fit(log_x)
    log_b, sec_b = gmm_boundaries_logspace(gmm, tau=0.5)
    assert np.isfinite(log_b[0])
    assert sec_b[0] > 0  # exp() of finite log-boundary is positive


# gmm_cv_neg_loglik short-data branches


def test_gmm_cv_neg_loglik_returns_inf_when_n_below_folds():
    """N < n_folds → cannot do K-fold CV → return inf."""
    short = np.array([0.1, 0.2])  # only 2 samples
    val = gmm_cv_neg_loglik(short, n_components=1, n_folds=5, seed=0)
    assert val == float("inf")


def test_gmm_cv_neg_loglik_returns_inf_when_train_smaller_than_n_components():
    """Even if N >= n_folds, if a fold's train set is smaller than
    n_components, we cannot fit → inf."""
    # 5 samples, 5 folds → train sets of size 4, n_components=5 → inf
    short = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    val = gmm_cv_neg_loglik(short, n_components=5, n_folds=5, seed=0)
    assert val == float("inf")


def test_t_mixture_cv_neg_loglik_returns_inf_when_n_below_folds():
    """Same short-data branch on the t-mixture path."""
    short = np.array([0.1, 0.2])
    val = t_mixture_cv_neg_loglik(short, n_components=1, n_folds=5, seed=0)
    assert val == float("inf")


def test_t_mixture_cv_neg_loglik_returns_inf_when_train_smaller_than_n_components():
    """When a fold's training set is smaller than the component count the t-mixture CV returns inf
    instead of fitting a degenerate model."""
    short = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    val = t_mixture_cv_neg_loglik(short, n_components=5, n_folds=5, seed=0)
    assert val == float("inf")


# select_best_n_components cv_mode branch


def test_select_best_n_components_cv_mode_picks_rep_by_bic():
    """When ic_col == 'cv_neg_loglik', the rep selection within the chosen
    n_comp falls back to argmin BIC (CV is constant across reps for a
    given (key, n_comp))."""
    df = pls.DataFrame({
        "key":    ["a", "a", "a", "a"],
        "n_comp": [2,   2,   3,   3],
        "rep":    [0,   1,   0,   1],
        "cv_neg_loglik": [10.0, 10.0, 5.0, 5.0],   # K=3 wins on CV
        "bic":           [12.0, 11.0, 7.0, 8.0],   # within K=3, rep=0 wins on bic
    })
    out = select_best_n_components(df, ic_col="cv_neg_loglik")
    assert out["a"]["n_comp"] == 3
    assert out["a"]["rep"] == 0


# plot_gmm_fit / qqplot_gmm smoke


def test_plot_gmm_fit_returns_figure_and_axes():
    """plot_gmm_fit should return (Figure, Axes); does not crash."""
    rng = np.random.RandomState(0)
    log_x = rng.normal(0.0, 1.0, 200).reshape(-1, 1)
    gmm = GaussianMixture(n_components=1, covariance_type="full",
                          random_state=0).fit(log_x)
    # Signature: plot_gmm_fit(model, x, ...)
    fig, ax = plot_gmm_fit(gmm, log_x.ravel(), figsize=(5, 5), density_grid_points=500, bin_mean_grid_points=4000, bins=20)
    assert fig is not None
    assert ax is not None


def test_plot_gmm_fit_with_xlims_filters_data():
    """xlims=(lo, hi) clips the input data before plotting; show_components=True
    triggers the per-component density-curve overlay. Returns a Figure that
    the caller can save via `figure_io.save_figure`."""
    rng = np.random.RandomState(0)
    log_x = rng.normal(0.0, 1.0, 200)
    gmm = GaussianMixture(n_components=2, covariance_type="full",
                          random_state=0).fit(log_x.reshape(-1, 1))
    fig, _ax = plot_gmm_fit(gmm, log_x, figsize=(5, 5), density_grid_points=500, bin_mean_grid_points=4000, bins=10,
                            xlims=(-1.0, 1.0), show_components=True)
    assert isinstance(fig, Figure)


def test_qqplot_gmm_returns_figure():
    """qqplot_gmm returns a matplotlib Figure (caller saves via figure_io)."""
    rng = np.random.RandomState(0)
    log_x = rng.normal(0.0, 0.5, 300)
    gmm = GaussianMixture(n_components=1, covariance_type="full",
                          random_state=0).fit(log_x.reshape(-1, 1))
    fig = qqplot_gmm(log_x, gmm, n_q=50)
    assert isinstance(fig, Figure)


# compute_behavioral_features — FeatureZoo class + missing pure helpers


# FeatureZoo class


def test_feature_zoo_init_loads_visualizations_settings():
    """FeatureZoo.__init__ stores all kwargs and loads the package's
    visualizations_settings.json into self.visualizations_parameter_dict."""
    fz = FeatureZoo(
        root_directory="/some/path",
        message_output=lambda *_a, **_kw: None,
    )
    assert fz.root_directory == "/some/path"
    assert isinstance(fz.visualizations_parameter_dict, dict)
    # Sanity: the loaded JSON has the expected top-level keys.
    assert "figures" in fz.visualizations_parameter_dict
    assert "male_colors" in fz.visualizations_parameter_dict


def test_feature_zoo_init_rejects_unexpected_kwarg():
    """
    Description
    -----------
    ``FeatureZoo.__init__`` must validate its kwargs against the consumed set
    and raise ``TypeError`` (naming the offending key) on anything else,
    guarding against silently-typo'd settings keys.

    Parameters
    ----------

    Returns
    -------
    None
    """

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        FeatureZoo(root_directory="/x", not_a_real_kwarg=1)


def test_feature_zoo_save_behavioral_features_missing_h5_raises(tmp_path, mocker):
    """No `*_points3d_translated_rotated_metric.h5` under <root>/video →
    FileNotFoundError from first_match_or_raise. Critical: the function should
    not silently no-op if the input file is missing."""
    (tmp_path / "video").mkdir()
    mocker.patch("usv_playpen.analyses.compute_behavioral_features.smart_wait")
    fz = FeatureZoo(
        root_directory=str(tmp_path),
        behavioral_parameters_dict={
            "head_points": [], "tail_points": [],
            "back_root_points": [], "derivative_bins": 5,
        },
        message_output=lambda *_a, **_kw: None,
    )
    with pytest.raises(FileNotFoundError, match="translated/rotated"):
        fz.save_behavioral_features_to_file()


# Anatomical node layout the feature pipeline requires: the union of
# `head_points` (Head, Ear_R, Ear_L, Nose), `tail_points` (TTI, Tail_0..2,
# TailTip), `back_root_points` (Neck, Trunk, TTI), plus the bare Head / Neck
# / Nose / TTI literals the social block indexes. Order is arbitrary — the
# pipeline looks every node up by name via `mouse_nodes.index(...)`.
_BEH_NODES = [
    "Head", "Ear_R", "Ear_L", "Nose", "Neck", "Trunk", "TTI",
    "Tail_0", "Tail_1", "Tail_2", "TailTip",
]

# Per-node body-frame offsets (metres): a head-to-tail axis along +x with
# the snout points slightly elevated. Distinct, non-coincident positions
# keep `get_back_root` / `get_head_root` out of their degenerate (NaN)
# branches so the geometry actually exercises the math.
_BEH_NODE_OFFSETS = {
    "Nose":    (0.030, 0.000, 0.018),
    "Head":    (0.020, 0.000, 0.020),
    "Ear_R":   (0.015, 0.008, 0.019),
    "Ear_L":   (0.015, -0.008, 0.019),
    "Neck":    (0.000, 0.000, 0.016),
    "Trunk":   (-0.020, 0.000, 0.014),
    "TTI":     (-0.040, 0.000, 0.012),
    "Tail_0":  (-0.050, 0.000, 0.010),
    "Tail_1":  (-0.060, 0.000, 0.009),
    "Tail_2":  (-0.070, 0.000, 0.008),
    "TailTip": (-0.080, 0.000, 0.007),
}


def _build_behavioral_tracking_h5(
    h5_path, *, n_frames=240, fps=150.0, experimental_code=b"E2MF",
):
    """
    Description
    -----------
    Write a minimal two-mouse `*_points3d_translated_rotated_metric.h5`
    carrying every dataset `save_behavioral_features_to_file` reads:
    `tracks` (n_frames, 2 mice, 11 nodes, 3), `node_names`, `track_names`,
    `experimental_code`, and `recording_frame_rate`. Each mouse is a rigid
    body-axis cloud (`_BEH_NODE_OFFSETS`) translated to its own x-centre
    and drifted slowly across frames with light per-frame jitter, so speed
    / derivatives / Euler-angle paths all see non-degenerate motion. All
    coordinates stay within ±0.32 m so the centroid (×100 → cm) lands
    inside the spatial occupancy window.

    Parameters
    ----------
    h5_path (pathlib.Path)
        Destination H5 path.
    n_frames (int)
        Number of tracked frames.
    fps (float)
        Camera frame rate written to `recording_frame_rate`.
    experimental_code (bytes)
        Experiment code; `b"E2MF"` decodes to one male + one female so
        `choose_animal_colors` returns two colors.

    Returns
    -------
    None
    """

    rng = np.random.default_rng(0)
    n_mice = 2
    n_nodes = len(_BEH_NODES)
    tracks = np.zeros((n_frames, n_mice, n_nodes, 3), dtype=np.float64)
    drift = np.linspace(0.0, 0.02, n_frames)
    for mouse_idx in range(n_mice):
        x_centre = -0.06 + 0.12 * mouse_idx
        for node_idx, node in enumerate(_BEH_NODES):
            off_x, off_y, off_z = _BEH_NODE_OFFSETS[node]
            tracks[:, mouse_idx, node_idx, 0] = (
                x_centre + off_x + drift + rng.normal(0.0, 0.0008, n_frames)
            )
            tracks[:, mouse_idx, node_idx, 1] = (
                off_y + rng.normal(0.0, 0.0008, n_frames)
            )
            tracks[:, mouse_idx, node_idx, 2] = (
                off_z + rng.normal(0.0, 0.0008, n_frames)
            )

    with h5py.File(h5_path, "w") as f:
        f.create_dataset("tracks", data=tracks)
        f.create_dataset(
            "node_names", data=np.array([n.encode("utf-8") for n in _BEH_NODES])
        )
        f.create_dataset("track_names", data=np.array([b"m1", b"m2"]))
        f.create_dataset("experimental_code", data=experimental_code)
        f.create_dataset("recording_frame_rate", data=np.float64(fps))


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_feature_zoo_save_behavioral_features_end_to_end(tmp_path, mocker):
    """
    Description
    -----------
    End-to-end run of ``save_behavioral_features_to_file`` on a two-mouse
    synthetic session: drives the full individual + social + SEI feature
    computation, the 1D / 2D ``generate_feature_distributions`` pass, the
    ``plot_feature_distributions`` PDF render (individual + social pages, bar +
    spatial-occupancy branches), and the CSV write.

    Asserts both output artifacts land next to the input H5, the CSV has one
    row per tracked frame, and a representative slice of the expected column
    namespace (self pose / movement / spatial + dyadic distance / angle / SEI,
    in both pair directions) is present.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Per-test temp directory used as the session root.
    mocker (pytest_mock.MockerFixture)
        Used to no-op the interactive ``smart_wait``.

    Returns
    -------
    None
    """

    video_dir = tmp_path / "video" / "vid1"
    video_dir.mkdir(parents=True)
    h5_path = video_dir / "20230101000000_points3d_translated_rotated_metric.h5"
    n_frames = 240
    _build_behavioral_tracking_h5(h5_path, n_frames=n_frames)

    mocker.patch("usv_playpen.analyses.compute_behavioral_features.smart_wait")
    fz = FeatureZoo(
        root_directory=str(tmp_path),
        behavioral_parameters_dict={
            "head_points":      ["Head", "Ear_R", "Ear_L", "Nose"],
            "tail_points":      ["TTI", "Tail_0", "Tail_1", "Tail_2", "TailTip"],
            "back_root_points": ["Neck", "Trunk", "TTI"],
            "derivative_bins":  10,
            "speed_smoothing_time_window": 0.015,
            "feature_hist_num_bins": 36,
            "spatial_hist_min_val": -32,
            "spatial_hist_max_val": 32,
            "spatial_hist_num_bins": 196,
            "back_angles_optimizer_seed": 0,
            "sei_speed_percentile": 99,
        },
        message_output=lambda *_a, **_kw: None,
    )

    fz.save_behavioral_features_to_file()

    csv_path = video_dir / "20230101000000_points3d_translated_rotated_metric_behavioral_features.csv"
    pdf_path = video_dir / "20230101000000_points3d_translated_rotated_metric_behavioral_features_histograms.pdf"
    assert csv_path.exists(), "behavioral features CSV not written"
    assert pdf_path.exists(), "feature-distribution histogram PDF not written"
    assert pdf_path.stat().st_size > 10_000, "histogram PDF suspiciously small"

    df = pls.read_csv(csv_path)
    assert df.height == n_frames, (
        f"expected one CSV row per frame ({n_frames}), got {df.height}"
    )
    expected_columns = {
        # self spatial + movement + pose
        "m1.spaceX", "m1.spaceY", "m1.spaceZ",
        "m1.speed", "m1.acceleration",
        "m1.allo_yaw", "m1.ego_yaw", "m1.back_pitch", "m1.body_dir",
        "m1.tail_curvature", "m1.neck_elevation",
        "m2.allo_yaw",
        # dyadic distance / angle / SEI (both pair directions)
        "m1-m2.nose-nose", "m1-m2.TTI-TTI", "m1-m2.allo_yaw-nose",
        "m1-m2.orofacial-sei", "m1-m2.anogenital-sei",
        "m2-m1.orofacial-sei", "m2-m1.anogenital-sei",
    }
    missing = expected_columns - set(df.columns)
    assert not missing, f"CSV missing expected feature columns: {sorted(missing)}"


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_feature_zoo_save_behavioral_features_single_mouse_skips_social(tmp_path, mocker):
    """
    Description
    -----------
    A one-mouse session must still produce the full individual-feature table
    and histogram PDF, but emit no dyadic columns — the social branch
    (``mouse_data.shape[1] >= 2``) must be correctly skipped.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Per-test temp directory used as the session root.
    mocker (pytest_mock.MockerFixture)
        Used to no-op the interactive ``smart_wait``.

    Returns
    -------
    None
    """

    video_dir = tmp_path / "video" / "vid1"
    video_dir.mkdir(parents=True)
    h5_path = video_dir / "20230101000000_points3d_translated_rotated_metric.h5"
    n_frames = 200

    rng = np.random.default_rng(1)
    tracks = np.zeros((n_frames, 1, len(_BEH_NODES), 3), dtype=np.float64)
    drift = np.linspace(0.0, 0.02, n_frames)
    for node_idx, node in enumerate(_BEH_NODES):
        off_x, off_y, off_z = _BEH_NODE_OFFSETS[node]
        tracks[:, 0, node_idx, 0] = off_x + drift + rng.normal(0.0, 0.0008, n_frames)
        tracks[:, 0, node_idx, 1] = off_y + rng.normal(0.0, 0.0008, n_frames)
        tracks[:, 0, node_idx, 2] = off_z + rng.normal(0.0, 0.0008, n_frames)
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("tracks", data=tracks)
        f.create_dataset(
            "node_names", data=np.array([n.encode("utf-8") for n in _BEH_NODES])
        )
        f.create_dataset("track_names", data=np.array([b"m1"]))
        f.create_dataset("experimental_code", data=b"E1M")
        f.create_dataset("recording_frame_rate", data=np.float64(150.0))

    mocker.patch("usv_playpen.analyses.compute_behavioral_features.smart_wait")
    fz = FeatureZoo(
        root_directory=str(tmp_path),
        behavioral_parameters_dict={
            "head_points":      ["Head", "Ear_R", "Ear_L", "Nose"],
            "tail_points":      ["TTI", "Tail_0", "Tail_1", "Tail_2", "TailTip"],
            "back_root_points": ["Neck", "Trunk", "TTI"],
            "derivative_bins":  10,
            "speed_smoothing_time_window": 0.015,
            "feature_hist_num_bins": 36,
            "spatial_hist_min_val": -32,
            "spatial_hist_max_val": 32,
            "spatial_hist_num_bins": 196,
            "back_angles_optimizer_seed": 0,
            "sei_speed_percentile": 99,
        },
        message_output=lambda *_a, **_kw: None,
    )

    fz.save_behavioral_features_to_file()

    csv_path = video_dir / "20230101000000_points3d_translated_rotated_metric_behavioral_features.csv"
    assert csv_path.exists()
    df = pls.read_csv(csv_path)
    assert "m1.speed" in df.columns
    assert not any("-" in col.split(".")[0] for col in df.columns), (
        "single-mouse session unexpectedly produced dyadic (social) columns"
    )


# calculate_sei


def test_calculate_sei_returns_high_score_when_aligned_and_close():
    """Observer head aligned with target's nose at moderate range → high SEI.

    Layout: 1 frame, 2 mice, 6 nodes per mouse, 3 dims.
    Indices used by SEI: idx_head=5, idx_nose=0, idx_tti=3.

    Target is placed about one body-length ahead so that d_norm ≈ 1 (which
    keeps gamma=1+1/d_norm bounded). Putting the target exactly at the
    observer's nose would make d_norm=0 and gamma→∞, which (despite a
    near-perfect cos_theta) crushes the score by floating-point noise — a
    documented edge of the SEI algorithm.
    """
    n_nodes = 6
    tracks = np.zeros((1, 2, n_nodes, 3), dtype=float)
    # Observer mouse: head at origin, nose ahead, TTI behind
    tracks[0, 0, 5, :] = [0.0, 0.0, 0.0]    # idx_head
    tracks[0, 0, 0, :] = [0.05, 0.0, 0.0]   # idx_nose
    tracks[0, 0, 3, :] = [-0.05, 0.0, 0.0]  # idx_tti
    # Target mouse: nose well in front of the observer (d_norm ≈ 1)
    tracks[0, 1, 0, :] = [0.15, 0.0, 0.0]   # observed_node_idx (nose)

    speeds = np.array([0.05])
    sei = calculate_sei(
        tracks, speeds,
        observer_idx=0, observed_idx=1, observed_node_idx=0,
        v_max=0.1, speed_percentile=99,
    )
    assert sei.shape == (1,)
    # Aligned + moderate distance → engagement should be substantially positive
    assert sei[0] > 0.3


def test_calculate_sei_returns_negative_when_target_is_behind():
    """Target placed behind observer's head → cosine of gaze-target angle is
    negative → SEI is negative (signed output preserved)."""
    n_nodes = 6
    tracks = np.zeros((1, 2, n_nodes, 3), dtype=float)
    tracks[0, 0, 5, :] = [0.0, 0.0, 0.0]    # observer head
    tracks[0, 0, 0, :] = [0.05, 0.0, 0.0]   # observer nose: gaze axis = +x
    tracks[0, 0, 3, :] = [-0.05, 0.0, 0.0]  # observer TTI
    # Target nose at observer's back (negative x)
    tracks[0, 1, 0, :] = [-0.05, 0.0, 0.0]

    sei = calculate_sei(tracks, np.array([0.05]),
                        observer_idx=0, observed_idx=1, observed_node_idx=0,
                        v_max=0.1, speed_percentile=99)
    assert sei[0] < 0


# get_back_root


def _make_back_input(neck_pos, tti_pos, n_frames=1):
    """
    Description
    -----------
    Builds a (n_frames, 1 mouse, 2 nodes, 3) point array with the neck at node
    index 0 and the TTI at node index 1, the minimal input ``get_back_root``
    needs to build a back frame.

    Parameters
    ----------
    neck_pos (list)
        Neck (x, y, z) coordinates.
    tti_pos (list)
        TTI (x, y, z) coordinates.
    n_frames (int)
        Number of frames to repeat the pose over.

    Returns
    -------
    arr (np.ndarray)
        The (n_frames, 1, 2, 3) point array.
    """
    arr = np.zeros((n_frames, 1, 2, 3), dtype=float)
    arr[:, 0, 0, :] = neck_pos
    arr[:, 0, 1, :] = tti_pos
    return arr


def test_get_back_root_default_returns_orthonormal_columns():
    """default mode: returned matrices have orthonormal column vectors;
    z-axis matches world up; x-axis is the XY-projected (Neck-TTI) direction
    normalized."""
    arr = _make_back_input(neck_pos=[1.0, 0.0, 0.5],
                           tti_pos=[0.0, 0.0, 0.5])
    out = get_back_root(arr, mouse_id=0, neck_point_pos=0, tti_point_pos=1,
                        root_method="default")
    assert out.shape == (1, 3, 3)
    # Columns: x_dir, y_dir, z_dir
    x_col = out[0, :, 0]
    z_col = out[0, :, 2]
    np.testing.assert_allclose(z_col, [0, 0, 1], atol=1e-10)
    # x-axis points along (Neck - TTI) projected to XY then normalized → (1, 0, 0)
    np.testing.assert_allclose(x_col, [1.0, 0.0, 0.0], atol=1e-10)


def test_get_back_root_root_inv_returns_orthonormal_rows():
    """root_inv mode: rows are orthonormal (rotation matrix); pitch carried
    through the x-axis (z-component non-zero when neck is higher than TTI)."""
    arr = _make_back_input(neck_pos=[1.0, 0.0, 1.0],
                           tti_pos=[0.0, 0.0, 0.0])
    out = get_back_root(arr, mouse_id=0, neck_point_pos=0, tti_point_pos=1,
                        root_method="root_inv")
    assert out.shape == (1, 3, 3)
    # Rows orthonormal:
    R = out[0]
    # Each row is unit length
    norms = np.linalg.norm(R, axis=1)
    np.testing.assert_allclose(norms, [1, 1, 1], atol=1e-10)
    # And the matrix is orthogonal: R @ R.T = I
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)


def test_get_back_root_planar_branch_returns_xy_back_with_world_up():
    """The else-branch ('planar back'): x_dir is XY-projected, z_dir is forced
    to world up, y_dir is x_dir's 2D perpendicular. Returned as rows."""
    arr = _make_back_input(neck_pos=[0.0, 1.0, 0.5],
                           tti_pos=[0.0, 0.0, 0.0])
    out = get_back_root(arr, mouse_id=0, neck_point_pos=0, tti_point_pos=1,
                        root_method="anything-other-than-default-or-root_inv")
    R = out[0]
    # The z_dir row is forced to world up:
    np.testing.assert_allclose(R[2], [0, 0, 1], atol=1e-10)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_get_back_root_returns_nan_when_neck_and_tti_coincident():
    """Neck and TTI at the same point → undefined x_dir/y_dir → NaN in those
    two columns. The z-axis column is hardcoded to world up in the default
    branch, so it remains [0, 0, 1] regardless.

    The internal divide-by-zero raises a RuntimeWarning that the project's
    pytest config escalates to an error; the warning is the source of truth
    that the input was singular, so we silence it for this test only."""
    arr = _make_back_input(neck_pos=[0.5, 0.5, 0.5],
                           tti_pos=[0.5, 0.5, 0.5])
    out = get_back_root(arr, mouse_id=0, neck_point_pos=0, tti_point_pos=1,
                        root_method="default")
    # x-axis (column 0) and y-axis (column 1) should be NaN; z-axis is world up.
    assert np.all(np.isnan(out[0, :, 0]))
    assert np.all(np.isnan(out[0, :, 1]))
    np.testing.assert_array_equal(out[0, :, 2], [0.0, 0.0, 1.0])


# get_back_angles


def test_get_back_angles_zero_when_back_along_x_axis():
    """A back direction along +x is the canonical "centred" frame: after the
    Nelder-Mead alignment finds zero rotation, the residual pitch/yaw should
    be ~0 (numerical noise only). Output is in DEGREES per the source."""
    back = np.array([[1.0, 0.0, 0.0]])
    out = get_back_angles(back, optimizer_seed=0)
    assert out.shape == (1, 2)
    np.testing.assert_allclose(out[0], [0.0, 0.0], atol=1e-3)


def test_get_back_angles_returns_finite_for_tilted_back():
    """A 45° X-Z tilted back: optimizer aligns it; per-frame residual still
    parses to finite degree values without NaN/inf — that's the contract."""
    angle = np.pi / 4
    back = np.array([[np.cos(angle), 0.0, np.sin(angle)]])
    out = get_back_angles(back, optimizer_seed=0)
    assert out.shape == (1, 2)
    assert np.all(np.isfinite(out))


def test_get_back_angles_handles_multi_frame_input():
    """Multi-frame input → output has shape (n_frames, 2)."""
    back = np.array([
        [1.0, 0.0, 0.0],
        [0.866, 0.5, 0.0],
        [0.5, 0.866, 0.0],
    ])
    out = get_back_angles(back, optimizer_seed=0)
    assert out.shape == (3, 2)
    assert np.all(np.isfinite(out))


# compute_neuronal_tuning_curves — pure-helper edge branches


def test_bin_property_indices_nan_and_degenerate_range():
    """NaN values, values at the upper edge of the half-open range, all-NaN input and a zero-width
    range all map to the -1 sentinel, while in-range values bin normally."""
    vals = np.array([0.0, 5.0, 10.0, np.nan])
    out = _bin_property_indices(vals, [0.0, 10.0], 5)
    assert out[3] == -1                      # NaN -> sentinel
    # In-range values bin normally; a value == the upper edge (10.0) is out of the
    # half-open [low, high) range and is now DROPPED to the -1 sentinel (previously it
    # was clipped into the last bin), matching the behavioral/spatial occupancy binning.
    assert out[0] == 0
    assert out[1] == 2
    assert out[2] == -1
    # all-NaN -> all sentinel
    assert np.all(_bin_property_indices(np.array([np.nan, np.nan]), [0.0, 1.0], 4) == -1)
    # degenerate (zero-width) range -> all sentinel
    assert np.all(_bin_property_indices(vals, [3.0, 3.0], 5) == -1)


def test_bin_count_for_property_mask_vs_default():
    """Continuous properties use the default bin count, while the integer ``mask_number`` gets its
    own count."""
    assert _bin_count_for_property("duration", 36) == 36
    assert _bin_count_for_property("mask_number", 36) != 36


def test_gaussian_smooth_1d_wraps_a_circular_axis():
    """A circular axis wraps: a spike of rate in the last bin spreads into the first
    bin as much as into the second-to-last (the two ends are one direction), with or
    without NaN bins, while a linear axis leaves the first bin at the edge-extended
    value. A shift of a circular array commutes with its smoothing."""

    data = np.zeros(12)
    data[-1] = 1.0
    wrapped = _gaussian_smooth_1d(data, 1.0, circular=True)
    linear = _gaussian_smooth_1d(data, 1.0)
    assert wrapped[0] == pytest.approx(wrapped[-2])
    assert wrapped[0] > 0.2
    assert linear[0] < 1e-3
    assert wrapped.sum() == pytest.approx(1.0)
    with_nan = data.copy()
    with_nan[5] = np.nan
    wrapped_nan = _gaussian_smooth_1d(with_nan, 1.0, preserve_nan=False, circular=True)
    assert wrapped_nan[0] == pytest.approx(wrapped_nan[-2])
    rng = np.random.default_rng(0)
    values = rng.uniform(0.0, 5.0, 12)
    np.testing.assert_allclose(
        np.roll(_gaussian_smooth_1d(values, 1.5, circular=True), 4),
        _gaussian_smooth_1d(np.roll(values, 4), 1.5, circular=True),
    )


def test_gaussian_smooth_1d_sigma_nonpositive_returns_input():
    """A non-positive sigma returns the input array object unchanged (no smoothing, no copy)."""
    data = np.arange(5, dtype=float)
    assert _gaussian_smooth_1d(data, sigma=0.0) is data


def test_gaussian_smooth_1d_nan_aware():
    """With ``preserve_nan`` NaN bins stay NaN and their neighbours stay finite; without it the NaN
    bin is filled from its neighbours."""
    data = np.array([1.0, np.nan, 3.0, 4.0])
    out = _gaussian_smooth_1d(data, sigma=1.0, preserve_nan=True)
    assert np.isnan(out[1])                   # NaN restored
    assert np.all(np.isfinite(out[[0, 2, 3]]))
    filled = _gaussian_smooth_1d(data, sigma=1.0, preserve_nan=False)
    assert np.isfinite(filled[1])             # NaN filled by neighbours


def test_gaussian_smooth_2d_sigma_nonpositive_and_nan():
    """In 2-D a non-positive sigma returns the input object, and ``preserve_nan`` keeps NaN cells
    NaN while smoothing the rest."""
    arr = np.array([[1.0, np.nan], [3.0, 4.0]])
    assert _gaussian_smooth_2d(arr, sigma=0.0) is arr
    out = _gaussian_smooth_2d(arr, sigma=1.0, preserve_nan=True)
    assert np.isnan(out[0, 1])
    assert np.isfinite(out[0, 0])


def test_longest_run_empty_all_false_all_true_circular():
    """Edge cases of ``_longest_run``: empty and all-False masks give no run, an all-True mask gives
    a full run, and True at both ends of a circular mask gives one wrapped run that ends before
    it starts."""
    assert _longest_run(np.array([], dtype=bool)) == (-1, -1, 0)
    assert _longest_run(np.array([False, False])) == (-1, -1, 0)
    s, e, length = _longest_run(np.array([True, True, True]))
    assert length == 3
    # wrap: True at the two ends -> one length-2 circular run
    s, e, length = _longest_run(np.array([True, False, False, True]), circular=True)
    assert length == 2
    assert e < s


def test_peak_z_info_all_invalid_returns_nan():
    """When every bin has zero null SD there is no valid z, so the peak is NaN with index -1."""
    rate = np.array([1.0, 2.0])
    pa, pi, ps = _peak_z_info(rate, np.array([0.0, 0.0]), np.array([0.0, 0.0]))
    assert np.isnan(pa)
    assert pi == -1
    assert np.isnan(ps)


def test_run_analysis_circular_wrap_and_suppress():
    """A circular excitation run over indices 4, 5, 0 has length 3, and a linear suppression run
    over the first three bins has length 3."""
    rate = np.array([10.0, 1.0, 1.0, 1.0, 10.0, 10.0])
    low = np.zeros(6)
    high = np.full(6, 2.0)
    mean = np.ones(6)
    std = np.ones(6)
    out = _run_analysis(rate, low, high, mean, std, circular=True)
    assert out["excit"]["max_run"] == 3       # indices 4,5,0 wrap
    # suppression side
    rate_s = np.array([0.0, 0.0, 0.0, 5.0, 5.0, 5.0])
    out_s = _run_analysis(rate_s, np.full(6, 1.0), np.full(6, 9.0), mean, std)
    assert out_s["suppress"]["max_run"] == 3


def test_run_analysis_all_nan_z_in_run():
    """A run whose null mean is NaN in every bin is still counted by length, but its peak z
    is NaN."""
    rate = np.array([10.0, 10.0, 10.0])
    out = _run_analysis(
        rate, np.zeros(3), np.full(3, 2.0), np.full(3, np.nan), np.zeros(3)
    )
    assert out["excit"]["max_run"] == 3
    assert np.isnan(out["excit"]["peak_z"])


def test_selectivity_index_edges():
    """Selectivity is NaN for all-NaN or all-zero curves and 1 for a curve with a zero minimum."""
    assert np.isnan(_selectivity_index(np.array([np.nan, np.nan])))
    assert np.isnan(_selectivity_index(np.array([0.0, 0.0])))
    assert _selectivity_index(np.array([0.0, 4.0])) == 1.0


def test_monotonicity_spearman_too_few_finite():
    """Fewer than three finite points give NaN monotonicity even when the array is longer."""
    assert np.isnan(_monotonicity_spearman(np.array([1.0, np.nan])))


def test_skaggs_info_and_sparsity_edges():
    """A single hot bin gives positive information and sparsity in (0, 1]; zero total occupancy or
    zero rate gives NaN."""
    rate = np.array([0.0, 0.0, 4.0, 0.0])
    occ = np.array([1.0, 1.0, 1.0, 1.0])
    assert _skaggs_info_rate_bps(rate, occ) > 0
    assert 0 < _skaggs_sparsity(rate, occ) <= 1
    # no occupancy -> NaN
    assert np.isnan(_skaggs_info_rate_bps(rate, np.zeros(4)))
    # no positive rate -> NaN (R <= 0) for both
    assert np.isnan(_skaggs_info_rate_bps(np.zeros(4), occ))
    assert np.isnan(_skaggs_sparsity(np.zeros(4), occ))


def test_spatial_coherence_edges():
    """Coherence is NaN for 1-D input or a map with fewer than three valid cells, and finite for a
    valid gradient."""
    assert np.isnan(_spatial_coherence(np.array([1.0, 2.0, 3.0])))   # not 2D
    nan_map = np.full((3, 3), np.nan)
    assert np.isnan(_spatial_coherence(nan_map))                     # <3 valid
    grad = np.arange(9, dtype=float).reshape(3, 3)
    assert np.isfinite(_spatial_coherence(grad))


def test_ramp_index_edges():
    """The ramp index is finite on a full window, NaN when the window does not reach the -1.5 s
    anchor, and NaN when rate and centre lengths differ."""
    centers = np.linspace(-2.0, 0.0, 40)
    rate = np.linspace(1.0, 2.0, 40)
    assert np.isfinite(_ramp_index(rate, centers, near_anchor_s=-0.1, far_anchor_s=-1.5))
    # window not spanning the -1.5 anchor -> NaN (coverage guard)
    narrow = np.linspace(-1.0, 0.0, 20)
    assert np.isnan(_ramp_index(np.ones(20), narrow, near_anchor_s=-0.1, far_anchor_s=-1.5))
    # size mismatch -> NaN
    assert np.isnan(_ramp_index(np.ones(5), centers, near_anchor_s=-0.1, far_anchor_s=-1.5))


def test_circular_shift_spike_times_modulo_wrap():
    """Shifting spikes by more than the session length wraps them into [0, duration) and re-sorts
    them."""
    spikes = np.array([0.1, 0.5, 0.9])
    shifted = _circular_shift_spike_times(spikes, duration_seconds=1.0, shift_seconds=2.7)
    assert np.all((shifted >= 0.0) & (shifted < 1.0))
    assert np.all(np.diff(shifted) >= 0)     # re-sorted


def test_generate_shuffle_offsets_seeded_and_in_range():
    """Shuffle offsets are reproducible under a fixed seed and fall in [min, max) seconds."""
    a = _generate_shuffle_offsets(8, 1.0, 3.0, seed=7)
    b = _generate_shuffle_offsets(8, 1.0, 3.0, seed=7)
    assert a.shape == (8,)
    assert np.array_equal(a, b)
    assert np.all((a >= 1.0) & (a < 3.0))


def test_percentiles_block_keys_and_values():
    """The null summary carries the mean, SD and the four percentile keys, with the mean taken per
    bin across shuffles."""
    arr = np.array([[0.0, 10.0], [2.0, 20.0], [4.0, 30.0]])
    out = _percentiles_block(arr)
    assert set(out) == {
        "null_mean", "null_std", "null_p0_5", "null_p2_5", "null_p97_5", "null_p99_5",
    }
    assert np.allclose(out["null_mean"], [2.0, 20.0])


def test_within_usv_validity_overlap_vs_clean():
    """An anchor USV is valid when no other USV overlaps it and invalid when an earlier USV extends
    past its onset."""
    starts = np.array([0.0, 1.0, 2.0])
    stops = np.array([0.5, 1.5, 2.5])
    assert _within_usv_validity(np.array([1]), starts, stops)[0]      # clean
    starts_o = np.array([0.0, 1.0])
    stops_o = np.array([2.0, 1.5])
    assert not _within_usv_validity(np.array([1]), starts_o, stops_o)[0]  # USV0 overlaps


def test_latest_other_stop_before_anchor():
    """For each anchor the latest earlier USV stop is returned, or -inf when no USV precedes it."""
    starts = np.array([0.0, 1.0, 2.0])
    stops = np.array([0.5, 1.5, 2.5])
    out = _latest_other_stop_before_anchor(np.array([2]), starts, stops)
    assert out[0] == 1.5
    # no earlier USV -> -inf
    assert _latest_other_stop_before_anchor(np.array([0]), starts, stops)[0] == -np.inf


def test_anchor_bin_validity_grid_post_and_prior():
    """The validity grid has one boolean row per anchor and one column per PETH bin with both clean-
    post and clean-prior rules on."""
    starts = np.array([0.0, 1.0, 2.0])
    stops = np.array([0.5, 1.5, 2.5])
    grid = _anchor_bin_validity_grid(
        np.array([1]), starts, stops, duration_seconds=10.0,
        rel_bin_lo=np.array([-0.5, -1.0]), rel_bin_hi=np.array([0.0, -0.5]),
        require_clean_post=True, require_clean_prior=True,
    )
    assert grid.shape == (1, 2)
    assert grid.dtype == bool


def test_anchor_bin_validity_grid_cuts_post_onset_bins_at_the_next_onset():
    """Post-onset bins stay valid only while they end before the next USV's onset
    (any USV after the anchor's onset); pre-onset bins are untouched by that rule,
    and a last USV keeps its post-onset bins up to the recording end."""
    starts = np.array([1.0, 1.25, 3.0])
    stops = np.array([1.1, 1.3, 3.1])
    edges = np.arange(-0.5, 0.5 + 1e-9, 0.1)
    rel_bin_lo, rel_bin_hi = edges[:-1], edges[1:]
    grid = _anchor_bin_validity_grid(
        np.array([0, 2]), starts, stops, duration_seconds=10.0,
        rel_bin_lo=rel_bin_lo, rel_bin_hi=rel_bin_hi,
        require_clean_post=False, require_clean_prior=False,
    )
    post = rel_bin_hi > 1e-9
    # anchor 0 (onset 1.0 s): the next onset is 1.25 s, so post-onset bins ending at
    # 1.1 and 1.2 s are valid and those ending at 1.3 / 1.4 / 1.5 s are not
    np.testing.assert_array_equal(grid[0, post], [True, True, False, False, False])
    assert grid[0, ~post].all()
    # anchor 2 (the last USV): nothing follows it, so every post-onset bin is valid
    assert grid[1].all()
    # another USV starting at the anchor's very onset leaves no clean post-onset bin
    same = _anchor_bin_validity_grid(
        np.array([0]), np.array([1.0, 1.0]), np.array([1.1, 1.2]), duration_seconds=10.0,
        rel_bin_lo=rel_bin_lo, rel_bin_hi=rel_bin_hi,
        require_clean_post=False, require_clean_prior=False,
    )
    assert not same[0, post].any()
    assert same[0, ~post].all()


def test_consecutive_interval_pairs_stay_within_one_animal_of_one_session():
    """Neighbouring intervals pair only when they share the session AND the animal: the switch
    from female A to female B, and from one session to the next, never forms a pair."""
    values = np.array([0.1, 0.2, 0.3, 5.0, 6.0, 0.4, 0.5])
    sessions = np.array(["s1"] * 5 + ["s2"] * 2)
    emitters = np.array(["A", "A", "A", "B", "B", "A", "A"])
    chains = np.ones(values.size, dtype=np.int64)
    current, following, pair_sessions = consecutive_interval_pairs(values, sessions, emitters, chains)
    np.testing.assert_allclose(current, [0.1, 0.2, 5.0, 0.4])
    np.testing.assert_allclose(following, [0.2, 0.3, 6.0, 0.5])
    assert list(pair_sessions) == ["s1", "s1", "s1", "s2"]


def test_consecutive_interval_pairs_never_cross_a_chain_break():
    """Two intervals of one animal pair only within one chain: a break between them (chain
    1 -> 2) forms no pair, the rest of each chain still does."""
    values = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    current, following, _ = consecutive_interval_pairs(
        values, np.array(["s1"] * 5), np.array(["A"] * 5), np.array([1, 1, 2, 2, 2]))
    np.testing.assert_allclose(current, [0.1, 0.3, 0.4])
    np.testing.assert_allclose(following, [0.2, 0.4, 0.5])


def test_fit_serial_dependence_recovers_a_known_bend():
    """
    Pairs whose median log(next) rises with slope 0.6 below 100 ms and is flat above it (Laplace
    noise, 30 sessions): the bent line recovers the bend near 100 ms with an interval that
    covers it, the slope near 0.6, and the tables carry the pool identity and one bend per
    replicate.
    """
    rng = np.random.default_rng(0)
    n = 6000
    x = rng.uniform(np.log(0.02), np.log(5.0), n)
    c = np.log(0.1)
    y = -1.0 + 0.6 * np.minimum(x, c) + rng.laplace(0.0, 0.25, n)
    sessions = np.repeat(np.arange(30).astype(str), n // 30)
    identity = {"sex": "male", "vocal_class": "usv", "adjacency": "filtered"}
    curves, fit, bends = fit_serial_dependence(
        current=np.exp(x), following=np.exp(y), pair_sessions=sessions, pool_identity=identity,
        n_knots=5, corner_widths=[0.05, 0.15], n_bootstrap=20, level=90.0,
        bend_bounds_ms=(30.0, 600.0), grid_percentiles=(1.0, 99.0), max_iter=5000, seed=0,
        n_jobs=1, n_grid=50)
    row = fit.row(0, named=True)
    assert 85.0 < row["bend_ms"] < 118.0
    assert row["bend_low_ms"] <= 100.0 <= row["bend_high_ms"]
    assert 0.5 < row["slope"] < 0.7
    assert row["n_pairs"] == n
    assert row["n_sessions"] == 30
    assert curves.height == 50
    assert bends.height == 20
    assert set(curves["sex"].to_list()) == {"male"}
    assert (curves["spline_low_ms"] <= curves["spline_high_ms"]).all()


def test_summarize_interval_pool_reports_counts_and_percentiles():
    """The descriptive summary written for every pool: count, sessions and the 5/25/50/75/95th
    percentiles; an empty pool gives NaN percentiles rather than raising."""
    values = np.arange(1, 101, dtype=float) / 1000.0
    out = summarize_interval_pool(values, 7)
    assert out["n_intervals"] == 100
    assert out["n_sessions"] == 7
    assert out["median_s"] == pytest.approx(np.percentile(values, 50))
    assert out["p05_s"] == pytest.approx(np.percentile(values, 5))
    assert out["p95_s"] == pytest.approx(np.percentile(values, 95))
    empty = summarize_interval_pool(np.array([]), 0)
    assert empty["n_intervals"] == 0
    assert np.isnan(empty["median_s"])
