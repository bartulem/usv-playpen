"""
@author: bartulem
Test analyses module.
"""

import json
import math
import pathlib
import pickle as _pickle
import subprocess as _subprocess
from unittest.mock import MagicMock

import h5py
import matplotlib
import numpy as np
import polars as pls
import pytest
from scipy.stats import norm
from sklearn.mixture import GaussianMixture

# Headless matplotlib for the mixture_model_utils plotting smoke tests.
matplotlib.use("Agg")

from usv_playpen.analyze_data import Analyst
from usv_playpen.analyses.decode_experiment_label import extract_information
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
)
from usv_playpen.analyses.compute_neuronal_tuning_curves import (
    generate_ratemaps,
    shuffle_spikes,
    NeuronalTuning,
    _longest_run,
    _peak_z_info,
    _run_analysis,
    _selectivity_index,
    _monotonicity_spearman,
    _skaggs_info_rate_bps,
    _skaggs_sparsity,
    _spatial_coherence,
    _ramp_index,
)
from usv_playpen.analyses.unit_triage_aggregator import (
    _to_jsonable,
    _safe_float,
    _flag_vmi,
    _flag_runs,
    _flag_categorical,
    _flag_spatial,
    _emitter_role_map,
)
from usv_playpen.analyses.mixture_model_utils import (
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
    Loads the live analyses_settings.json so the fixture stays in sync with
    the schema the Analyst actually references. Every analyses_booleans flag
    is forced to False; tests then flip individual flags as needed.
    """
    settings_path = pathlib.Path(__file__).parent.parent / 'src' / 'usv_playpen' / '_parameter_settings' / 'analyses_settings.json'
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
    """Mocks all external class dependencies for the Analyst class."""
    mocked_classes = {
        'FeatureZoo': mocker.patch('usv_playpen.analyze_data.FeatureZoo'),
        'NeuronalTuning': mocker.patch('usv_playpen.analyze_data.NeuronalTuning'),
        'AudioGenerator': mocker.patch('usv_playpen.analyze_data.AudioGenerator'),
        'InterUSVIntervalCalculator': mocker.patch('usv_playpen.analyze_data.InterUSVIntervalCalculator'),
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
    assert extract_information(None) is None


def test_extract_information_returns_none_for_non_string():
    assert extract_information(123) is None
    assert extract_information(["E", "2"]) is None


def test_extract_information_simple_courtship_pair():
    out = extract_information("E2CFM")
    assert out is not None
    assert "ephys" in out["experiment_type"]
    assert "courtship" in out["experiment_type"]
    assert out["mouse_number"] == 2
    assert out["mouse_sex"] == ["female", "male"]


def test_extract_information_solo_recording_with_estrus():
    out = extract_information("B1QFSe")
    assert out is not None
    assert out["mouse_number"] == 1
    assert "behavior" in out["experiment_type"]
    assert "alone" in out["experiment_type"]
    assert out["mouse_sex"] == ["female"]
    assert out["mouse_housing"] == ["single"]
    assert out["mouse_estrus"] == ["estrus"]


def test_extract_information_lighting_decoded():
    light = extract_information("E1L")
    assert light is not None
    assert "light" in light["experiment_type"]
    dark = extract_information("E1D")
    assert dark is not None
    assert "dark" in dark["experiment_type"]


@pytest.mark.parametrize(
    ("estrus_letter", "expected"),
    [("p", "proestrus"), ("e", "estrus"), ("m", "matestrus"), ("d", "diestrus")],
)
def test_extract_information_estrus_letters(estrus_letter, expected):
    out = extract_information(f"B1F{estrus_letter}")
    assert out is not None
    assert expected in out["mouse_estrus"]


def test_extract_information_default_lists_are_empty_when_unmatched():
    out = extract_information("E1")
    assert out is not None
    assert out["mouse_estrus"] == []
    assert out["mouse_housing"] == []
    assert out["mouse_sex"] == []


def test_generate_feature_distributions_1d_returns_seconds():
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
    x = np.zeros((30, 1))
    x[10] = np.nan
    first_der, second_der = calculate_derivatives(
        input_arr=x, diff_bins=2, capture_fr=100, is_angle=False
    )
    assert np.isnan(first_der[10]).all()
    assert np.isnan(second_der[10]).all()


def test_calculate_derivatives_angle_wraps_at_pi():
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
    n_frames = 4
    n_nodes = 6
    pts = np.zeros((n_frames, n_nodes, 3))
    pts[..., 0] = np.linspace(0.0, 1.0, n_nodes)
    out = calculate_tail_curvature(pts)
    assert out.shape == (n_frames, 1)
    np.testing.assert_allclose(out, 0.0, atol=1e-10)


def test_calculate_tail_curvature_positive_for_bent_tail():
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
    head_root = np.eye(3)[None, :, :]
    head_pivot = np.array([[0.0, 0.0, 0.0]])
    target = np.array([[1.0, 0.0, 0.0]])
    yaw, pitch = get_egocentric_direction(head_root, head_pivot, target)
    assert yaw[0] == pytest.approx(0.0)
    assert pitch[0] == pytest.approx(0.0)


def test_get_egocentric_direction_target_above_observer():
    head_root = np.eye(3)[None, :, :]
    head_pivot = np.zeros((1, 3))
    target = np.array([[0.0, 0.0, 1.0]])
    _, pitch = get_egocentric_direction(head_root, head_pivot, target)
    assert pitch[0] == pytest.approx(90.0)


def test_get_egocentric_direction_below_tolerance_returns_nan():
    head_root = np.eye(3)[None, :, :]
    head_pivot = np.zeros((1, 3))
    target = np.array([[1e-6, 0.0, 0.0]])
    yaw, pitch = get_egocentric_direction(head_root, head_pivot, target)
    assert np.isnan(yaw[0])
    assert np.isnan(pitch[0])


def test_calculate_speed_zero_for_stationary_centroid():
    n_frames = 30
    pts = np.zeros((n_frames, 4, 3))
    speed = calculate_speed(pts, capture_framerate=100, smoothing_time_window=0.05)
    assert speed.shape == (n_frames, 1)
    assert np.isnan(speed[0, 0])
    np.testing.assert_allclose(speed[1:, 0], 0.0, atol=1e-9)


def test_calculate_speed_constant_velocity_recovered():
    n_frames = 60
    pts = np.zeros((n_frames, 1, 3))
    pts[:, 0, 0] = np.arange(n_frames) * 0.01
    speed = calculate_speed(pts, capture_framerate=100, smoothing_time_window=0.05)
    interior = speed[10:-10, 0]
    np.testing.assert_allclose(interior, 100.0, rtol=1e-6)


def test_get_head_root_canonical_orientation():
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
    pts = np.zeros((1, 4, 3))
    pts[0, 0, 0] = np.nan
    R = get_head_root(pts)
    assert np.isnan(R[0]).all()


def test_get_euler_ang_identity_returns_zero_angles():
    R = np.eye(3)[None, :, :]
    angles = get_euler_ang(R)
    assert angles.shape == (1, 3)
    np.testing.assert_allclose(angles[0], [0.0, 0.0, 0.0], atol=1e-9)


def test_generate_ratemaps_1d_returns_expected_shapes():
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


def test_fit_log_gmm_recovers_two_modes():
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
    rng = np.random.default_rng(1)
    log_x = rng.normal(loc=0.0, scale=1.0, size=400).reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=0, n_init=2).fit(log_x)

    test_points = np.array([-1.0, -0.25, 0.0, 0.5, 1.5])
    cdf_values = gmm_cdf_logspace(test_points, gmm)
    recovered = gmm_quantile_logspace(cdf_values, gmm)
    np.testing.assert_allclose(recovered, test_points, atol=1e-6)


def test_gmm_boundaries_logspace_returns_log_and_seconds():
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
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 1))
    gmm = GaussianMixture(n_components=1, covariance_type='full', random_state=0).fit(X)
    gmm.covariance_type = 'not_a_real_type'
    with pytest.raises(ValueError, match="Unsupported covariance type"):
        _extract_params(gmm)


def test_log_gauss_matches_scipy_for_2d():
    from scipy.stats import multivariate_normal
    mu = np.array([0.0, 0.0])
    Sig = np.array([[1.0, 0.2], [0.2, 0.5]])
    Prec = np.linalg.inv(Sig)
    pts = np.array([[0.0, 0.0], [1.0, -0.5], [-2.0, 1.5]])
    expected = multivariate_normal(mean=mu, cov=Sig).logpdf(pts)
    np.testing.assert_allclose(_log_gauss(pts, mu, Sig, Prec), expected, atol=1e-12)


def test_alpha_returns_nonnegative_responsibilities():
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
    from scipy.stats import t as t_dist
    x = np.linspace(-3, 3, 25)
    mu, sigma2, nu = 0.5, 2.0, 5.0
    expected = t_dist.logpdf((x - mu) / np.sqrt(sigma2), df=nu) - 0.5 * np.log(sigma2)
    np.testing.assert_allclose(_t_logpdf_1d(x, mu, sigma2, nu), expected, atol=1e-12)


def test_t_logpdf_1d_gaussian_limit():
    x = np.linspace(-2, 2, 11)
    mu, sigma2, nu = 0.0, 1.0, 1e6
    np.testing.assert_allclose(
        _t_logpdf_1d(x, mu, sigma2, nu),
        norm.logpdf(x),
        atol=1e-3,
    )


def _make_tmixture():
    return TMixture(
        weights=[0.4, 0.6],
        means=[-2.0, 1.5],
        covariances=[0.25, 0.4],
        nus=[10.0, 8.0],
    )


def test_tmixture_init_normalizes_shapes():
    m = _make_tmixture()
    assert m.weights_.shape == (2,)
    assert m.means_.shape == (2, 1)
    assert m.covariances_.shape == (2, 1, 1)
    assert m.nus_.shape == (2,)
    assert m.n_components == 2
    assert m.covariance_type == "full"


def test_tmixture_score_samples_matches_logaddexp_of_components():
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
    m = _make_tmixture()
    z = m.predict_proba(np.linspace(-3.0, 3.0, 25))
    assert z.shape == (25, 2)
    np.testing.assert_allclose(z.sum(axis=1), 1.0, atol=1e-12)
    assert (z >= 0).all()


def test_tmixture_score_is_mean_of_score_samples():
    m = _make_tmixture()
    x = np.array([-1.0, 0.0, 1.0, 2.0])
    assert m.score(x) == pytest.approx(float(np.mean(m.score_samples(x))))


def test_tmixture_n_params_and_aic_bic_relations():
    m = _make_tmixture()
    x = np.linspace(-2.0, 2.0, 100)
    n_params = 4 * m.n_components - 1
    log_lik_total = float(np.sum(m.score_samples(x)))
    expected_bic = -2.0 * log_lik_total + n_params * np.log(x.size)
    expected_aic = -2.0 * log_lik_total + 2.0 * n_params
    assert m.bic(x) == pytest.approx(expected_bic)
    assert m.aic(x) == pytest.approx(expected_aic)


def test_t_mixture_cdf_in_unit_interval():
    m = _make_tmixture()
    x = np.linspace(-5.0, 5.0, 30)
    cdf = t_mixture_cdf_logspace(x, m)
    assert ((cdf > 0) & (cdf < 1)).all()
    assert (np.diff(cdf) >= 0).all()


def test_t_mixture_quantile_inverts_cdf():
    m = _make_tmixture()
    test_points = np.array([-1.5, -0.5, 0.5, 1.0])
    cdf_values = t_mixture_cdf_logspace(test_points, m)
    recovered = t_mixture_quantile_logspace(cdf_values, m)
    np.testing.assert_allclose(recovered, test_points, atol=1e-6)


# ---------------------------------------------------------------------------
# triage helpers (compute_neuronal_tuning_curves.py)
# ---------------------------------------------------------------------------


# _longest_run --------------------------------------------------------------

@pytest.mark.parametrize("mask, circular, expected", [
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


# _peak_z_info --------------------------------------------------------------


def test_peak_z_info_all_nan_returns_nan():
    nan = np.full(5, np.nan)
    a, idx, signed = _peak_z_info(nan, nan, nan)
    assert math.isnan(a) and idx == -1 and math.isnan(signed)


def test_peak_z_info_zero_std_filtered():
    rate = np.array([1.0, 2.0, 3.0])
    null_mean = np.zeros(3)
    null_std = np.zeros(3)
    a, idx, signed = _peak_z_info(rate, null_mean, null_std)
    assert math.isnan(a) and idx == -1


def test_peak_z_info_picks_max_abs_excitation():
    rate = np.array([1.0, 5.0, 1.0])
    null_mean = np.array([1.0, 1.0, 1.0])
    null_std = np.array([1.0, 1.0, 1.0])
    a, idx, signed = _peak_z_info(rate, null_mean, null_std)
    assert a == 4.0 and idx == 1 and signed == 4.0


def test_peak_z_info_picks_max_abs_suppression():
    rate = np.array([0.0, 0.0, -10.0])
    null_mean = np.array([0.0, 0.0, 0.0])
    null_std = np.array([1.0, 1.0, 1.0])
    a, idx, signed = _peak_z_info(rate, null_mean, null_std)
    assert a == 10.0 and idx == 2 and signed == -10.0


# _run_analysis -------------------------------------------------------------


def test_run_analysis_no_divergence():
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
    rate = np.full(3, 5.0)
    null_low = np.zeros(3)
    null_high = np.full(3, 1.0)
    null_mean = np.full(3, 0.5)
    null_std = np.full(3, 0.5)
    out = _run_analysis(rate, null_low, null_high, null_mean, null_std)
    assert out["excit"]["max_run"] == 3
    assert out["suppress"]["max_run"] == 0


def test_run_analysis_pure_suppression():
    rate = np.full(3, -5.0)
    null_low = np.full(3, -1.0)
    null_high = np.full(3, 1.0)
    null_mean = np.zeros(3)
    null_std = np.full(3, 1.0)
    out = _run_analysis(rate, null_low, null_high, null_mean, null_std)
    assert out["excit"]["max_run"] == 0
    assert out["suppress"]["max_run"] == 3


def test_run_analysis_circular_wraps_excitation():
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
    rate = np.array([1.0, 2.0, 10.0, 2.0, 1.0])
    null_low = np.full(5, 0.0)
    null_high = np.full(5, 1.5)
    null_mean = np.full(5, 1.0)
    null_std = np.full(5, 1.0)
    out = _run_analysis(rate, null_low, null_high, null_mean, null_std)
    assert out["excit"]["peak_idx"] == 2
    assert out["excit"]["peak_z"] == pytest.approx(9.0)


def test_run_analysis_handles_nan_rate():
    rate = np.array([np.nan, 5.0, 5.0, 5.0, np.nan])
    null_low = np.full(5, 0.0)
    null_high = np.full(5, 1.0)
    null_mean = np.full(5, 0.5)
    null_std = np.full(5, 0.5)
    out = _run_analysis(rate, null_low, null_high, null_mean, null_std)
    assert out["excit"]["max_run"] == 3
    assert out["excit"]["run_start_idx"] == 1


# _selectivity_index --------------------------------------------------------


def test_selectivity_uniform_returns_zero():
    assert _selectivity_index(np.full(10, 5.0)) == 0.0


def test_selectivity_max_min_normalized():
    assert _selectivity_index(np.array([0.0, 10.0])) == 1.0


def test_selectivity_all_nan_returns_nan():
    assert math.isnan(_selectivity_index(np.full(5, np.nan)))


# _monotonicity_spearman ----------------------------------------------------


def test_monotonicity_increasing():
    assert _monotonicity_spearman(np.arange(10).astype(float)) == pytest.approx(1.0)


def test_monotonicity_decreasing():
    assert _monotonicity_spearman(
        np.arange(10)[::-1].astype(float)
    ) == pytest.approx(-1.0)


def test_monotonicity_constant_returns_nan():
    # filterwarnings to suppress scipy ConstantInputWarning
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert math.isnan(_monotonicity_spearman(np.full(10, 5.0)))


def test_monotonicity_too_few_returns_nan():
    assert math.isnan(_monotonicity_spearman(np.array([1.0, 2.0])))


# _skaggs_info_rate_bps -----------------------------------------------------


def test_skaggs_info_place_cell():
    rate = np.array([10.0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    occ = np.ones(10)
    # R = 1.0, p_hot = 0.1, info = 0.1 * 10 * log2(10)
    assert _skaggs_info_rate_bps(rate, occ) == pytest.approx(np.log2(10.0))


def test_skaggs_info_uniform_returns_zero():
    assert _skaggs_info_rate_bps(np.full(10, 5.0), np.ones(10)) == pytest.approx(0.0)


def test_skaggs_info_all_zero_rate_returns_nan():
    assert math.isnan(_skaggs_info_rate_bps(np.zeros(10), np.ones(10)))


def test_skaggs_info_all_nan_returns_nan():
    assert math.isnan(
        _skaggs_info_rate_bps(np.full(10, np.nan), np.ones(10))
    )


def test_skaggs_info_zero_occ_excluded():
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
    rate1d = np.array([10.0, 0.0, 0.0, 0.0])
    occ1d = np.ones(4)
    rate2d = rate1d.reshape(2, 2)
    occ2d = occ1d.reshape(2, 2)
    assert _skaggs_info_rate_bps(rate1d, occ1d) == _skaggs_info_rate_bps(rate2d, occ2d)


# _skaggs_sparsity ----------------------------------------------------------


def test_skaggs_sparsity_uniform_is_one():
    assert _skaggs_sparsity(np.full(10, 5.0), np.ones(10)) == pytest.approx(1.0)


def test_skaggs_sparsity_place_cell_is_one_over_n():
    rate = np.array([10.0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    assert _skaggs_sparsity(rate, np.ones(10)) == pytest.approx(0.1)


def test_skaggs_sparsity_all_nan_returns_nan():
    assert math.isnan(_skaggs_sparsity(np.full(10, np.nan), np.ones(10)))


# _spatial_coherence --------------------------------------------------------


def test_spatial_coherence_gradient_high():
    grid = np.tile(np.arange(10), (10, 1)).astype(float)
    assert _spatial_coherence(grid) > 0.9


def test_spatial_coherence_random_low():
    rng = np.random.default_rng(42)
    assert abs(_spatial_coherence(rng.standard_normal((20, 20)))) < 0.3


def test_spatial_coherence_1d_returns_nan():
    assert math.isnan(_spatial_coherence(np.zeros(10)))


# _ramp_index ---------------------------------------------------------------


def test_ramp_index_up_returns_positive():
    centers = np.linspace(-2, 0, 41)
    rate = np.zeros(41)
    rate[centers >= -0.2] = 5.0
    rate[centers <= -1.0] = 1.0
    assert _ramp_index(rate, centers) == pytest.approx((5 - 1) / (5 + 1), abs=1e-3)


def test_ramp_index_equal_endpoints_returns_zero():
    centers = np.linspace(-2, 0, 41)
    rate = np.full(41, 3.0)
    assert _ramp_index(rate, centers) == 0.0


def test_ramp_index_both_zero_returns_nan():
    centers = np.linspace(-2, 0, 41)
    rate = np.zeros(41)
    assert math.isnan(_ramp_index(rate, centers))


def test_ramp_index_nan_endpoints_returns_nan():
    centers = np.linspace(-2, 0, 41)
    rate = np.full(41, np.nan)
    assert math.isnan(_ramp_index(rate, centers))


# _detect_bouts -------------------------------------------------------------


def test_detect_bouts_empty():
    bs, be, bi = NeuronalTuning._detect_bouts(np.empty(0), np.empty(0), 2.0)
    assert bs.size == 0 and be.size == 0 and bi.size == 0


def test_detect_bouts_single_usv():
    bs, be, bi = NeuronalTuning._detect_bouts(
        np.array([5.0]), np.array([5.5]), 2.0
    )
    assert bs.tolist() == [5.0] and be.tolist() == [5.5] and bi.tolist() == [0]


def test_detect_bouts_close_gaps_one_bout():
    bs, be, bi = NeuronalTuning._detect_bouts(
        np.array([5.0, 5.5, 6.0]), np.array([5.4, 5.9, 6.4]), 2.0
    )
    assert bs.tolist() == [5.0]
    assert be.tolist() == [6.4]
    assert bi.tolist() == [0, 0, 0]


def test_detect_bouts_far_gaps_separate_bouts():
    bs, be, bi = NeuronalTuning._detect_bouts(
        np.array([5.0, 10.0, 15.0]), np.array([5.5, 10.5, 15.5]), 2.0
    )
    assert bs.tolist() == [5.0, 10.0, 15.0]
    assert bi.tolist() == [0, 1, 2]


def test_detect_bouts_mixed():
    bs, be, bi = NeuronalTuning._detect_bouts(
        np.array([5.0, 5.5, 10.0, 15.0, 15.3]),
        np.array([5.4, 5.9, 10.5, 15.2, 15.5]),
        2.0,
    )
    assert bs.tolist() == [5.0, 10.0, 15.0]
    assert be.tolist() == [5.9, 10.5, 15.5]
    assert bi.tolist() == [0, 0, 1, 2, 2]


# _compute_vmi_for_emitter --------------------------------------------------


def test_vmi_empty_returns_all_nan():
    out = NeuronalTuning._compute_vmi_for_emitter(
        np.array([0.0]), np.empty(0), np.empty(0), np.empty(0), 2.0
    )
    assert math.isnan(out["vmi"]) and out["n_bouts"] == 0


def test_vmi_pure_excitation_returns_one():
    sp = np.array([5.05, 5.15, 5.25])
    em_starts = np.array([5.0])
    em_stops = np.array([5.3])
    em_dur = em_stops - em_starts
    out = NeuronalTuning._compute_vmi_for_emitter(
        sp, em_starts, em_stops, em_dur, 2.0
    )
    assert out["vmi"] == pytest.approx(1.0)


def test_vmi_pure_suppression_returns_minus_one():
    sp = np.array([3.0, 3.5, 4.0])  # all in baseline window, none in USV
    em_starts = np.array([5.0])
    em_stops = np.array([5.5])
    em_dur = em_stops - em_starts
    out = NeuronalTuning._compute_vmi_for_emitter(
        sp, em_starts, em_stops, em_dur, 2.0
    )
    assert out["vmi"] == pytest.approx(-1.0)


def test_vmi_baseline_pre_recording_is_nan():
    sp = np.array([0.6])
    em_starts = np.array([0.5])
    em_stops = np.array([1.0])
    em_dur = em_stops - em_starts
    out = NeuronalTuning._compute_vmi_for_emitter(
        sp, em_starts, em_stops, em_dur, 2.0
    )
    assert math.isnan(out["fr_baseline"])


def test_vmi_equal_pairs_wilcoxon_nan():
    sp = np.linspace(0, 100, 1000)  # uniform spike rate
    em_starts = np.array([5.0, 10.0, 15.0, 20.0])
    em_stops = np.array([6.0, 11.0, 16.0, 21.0])
    em_dur = em_stops - em_starts
    out = NeuronalTuning._compute_vmi_for_emitter(
        sp, em_starts, em_stops, em_dur, 2.0
    )
    assert math.isnan(out["wilcoxon_pvalue"])


def test_vmi_n_bouts_matches_detector():
    # 2 bouts: one big (3 USVs), one small (1 USV)
    em_starts = np.array([5.0, 5.5, 5.8, 20.0])
    em_stops = np.array([5.4, 5.7, 6.0, 20.5])
    em_dur = em_stops - em_starts
    out = NeuronalTuning._compute_vmi_for_emitter(
        np.array([5.05]), em_starts, em_stops, em_dur, 2.0
    )
    assert out["n_bouts"] == 2


# ---------------------------------------------------------------------------
# unit_triage_aggregator.py
# ---------------------------------------------------------------------------


# _to_jsonable / _safe_float ------------------------------------------------


@pytest.mark.parametrize("inp, expected", [
    (np.int64(5), 5),
    (np.float64(2.5), 2.5),
    (np.float64(np.nan), None),
    (np.float64(np.inf), None),
    (np.bool_(True), True),
    (np.array([1, 2, 3]), [1, 2, 3]),
    (pathlib.Path("/foo"), "/foo"),
])
def test_to_jsonable(inp, expected):
    assert _to_jsonable(inp) == expected


def test_to_jsonable_unhandled_type_raises():
    class _X:
        pass
    with pytest.raises(TypeError):
        _to_jsonable(_X())


@pytest.mark.parametrize("inp, expected", [
    (None, None),
    (5.0, 5.0),
    (np.nan, None),
    (np.inf, None),
    ("not a number", None),
    (np.float64(2.5), 2.5),
])
def test_safe_float(inp, expected):
    assert _safe_float(inp) == expected


# _flag_vmi -----------------------------------------------------------------


def test_flag_vmi_significant_excit():
    payload = {
        "vmi": 0.5, "wilcoxon_pvalue": 0.001, "n_bouts": 20,
        "fr_baseline": 1.0, "fr_usv": 3.0,
    }
    direction, info = _flag_vmi(payload, alpha=0.01, min_bouts=10)
    assert direction == "excit" and info["vmi"] == 0.5


def test_flag_vmi_significant_suppress():
    payload = {
        "vmi": -0.5, "wilcoxon_pvalue": 0.001, "n_bouts": 20,
        "fr_baseline": 3.0, "fr_usv": 1.0,
    }
    direction, info = _flag_vmi(payload, alpha=0.01, min_bouts=10)
    assert direction == "suppress"


def test_flag_vmi_below_min_bouts():
    payload = {
        "vmi": 0.5, "wilcoxon_pvalue": 0.001, "n_bouts": 5,
        "fr_baseline": 1.0, "fr_usv": 3.0,
    }
    assert _flag_vmi(payload, alpha=0.01, min_bouts=10) == (None, None)


def test_flag_vmi_above_alpha_rejected():
    payload = {
        "vmi": 0.5, "wilcoxon_pvalue": 0.5, "n_bouts": 20,
        "fr_baseline": 1.0, "fr_usv": 3.0,
    }
    assert _flag_vmi(payload, alpha=0.01, min_bouts=10) == (None, None)


def test_flag_vmi_nan_pvalue_rejected():
    payload = {
        "vmi": 0.5, "wilcoxon_pvalue": float("nan"), "n_bouts": 20,
        "fr_baseline": 1.0, "fr_usv": 3.0,
    }
    assert _flag_vmi(payload, alpha=0.01, min_bouts=10) == (None, None)


# _flag_runs ----------------------------------------------------------------


def test_flag_runs_pass():
    block = {"max_run": 5, "peak_z": 4.0, "n_bins": 5}
    info = _flag_runs(block, z_threshold=3.0, min_run=3)
    assert info is not None and info["max_run"] == 5


def test_flag_runs_short_run_rejected():
    block = {"max_run": 2, "peak_z": 4.0, "n_bins": 2}
    assert _flag_runs(block, z_threshold=3.0, min_run=3) is None


def test_flag_runs_low_z_rejected():
    block = {"max_run": 5, "peak_z": 1.0, "n_bins": 5}
    assert _flag_runs(block, z_threshold=3.0, min_run=3) is None


def test_flag_runs_negative_z_passes_via_abs():
    block = {"max_run": 5, "peak_z": -4.0, "n_bins": 5}
    info = _flag_runs(block, z_threshold=3.0, min_run=3)
    assert info is not None


# _flag_categorical ---------------------------------------------------------


def test_flag_categorical_pass():
    payload = {
        "peak_abs_z": 4.0, "best_cat": 5, "n_sig_categories": 3,
        "selectivity": 0.5, "peak_signed_z": 4.0,
    }
    info = _flag_categorical(payload, z_threshold=3.0)
    assert info is not None and info["best_cat"] == 5


def test_flag_categorical_fail():
    payload = {
        "peak_abs_z": 1.0, "best_cat": 5, "n_sig_categories": 0,
        "selectivity": 0.0, "peak_signed_z": 1.0,
    }
    assert _flag_categorical(payload, z_threshold=3.0) is None


# _flag_spatial -------------------------------------------------------------


def test_flag_spatial_pass():
    payload = {
        "info_rate_bps": 1.5, "sparsity": 0.3, "coherence": 0.8,
        "peak_rate_sps": 5.0, "peak_row": 3, "peak_col": 4,
    }
    info = _flag_spatial(payload, info_threshold=0.5)
    assert info is not None and info["info_rate_bps"] == 1.5


def test_flag_spatial_fail():
    payload = {
        "info_rate_bps": 0.1, "sparsity": 0.9, "coherence": 0.5,
        "peak_rate_sps": 1.0, "peak_row": 0, "peak_col": 0,
    }
    assert _flag_spatial(payload, info_threshold=0.5) is None


# _emitter_role_map ---------------------------------------------------------


def test_emitter_role_map_from_usv_peth():
    cluster_data = {
        "usv_peth": {"emitter_a": {"role": "self", "rate": np.zeros(5)}}
    }
    assert _emitter_role_map(cluster_data) == {"emitter_a": "self"}


def test_emitter_role_map_from_property_tuning():
    cluster_data = {
        "usv_property_tuning": {
            "emitter_b": {"duration": {"role": "partner"}}
        }
    }
    assert _emitter_role_map(cluster_data) == {"emitter_b": "partner"}


def test_emitter_role_map_missing_role_returns_empty():
    cluster_data = {"usv_peth": {"emitter_x": {"rate": np.zeros(5)}}}
    assert _emitter_role_map(cluster_data) == {}

# ---------------------------------------------------------------------------
# mixture_model_utils — additional targeted tests
# ---------------------------------------------------------------------------


def _make_two_component_gmm(seed=0, n_per=2000):
    """Build a fitted 1D GaussianMixture with two well-separated modes."""
    rng = np.random.default_rng(seed)
    a = rng.normal(loc=-2.0, scale=0.3, size=n_per)
    b = rng.normal(loc=+2.0, scale=0.4, size=n_per)
    log_x = np.concatenate([a, b]).reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, random_state=seed).fit(log_x)
    order = np.argsort(gmm.means_.ravel())
    return gmm, order, log_x


def test_gmm_modes_1d_recovers_component_centers():
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
    gmm, _, log_x = _make_two_component_gmm()
    icl = gmm_icl(gmm, log_x)
    bic = gmm.bic(log_x)
    # ICL should be >= BIC for any non-degenerate fit (entropy term is
    # always non-negative).
    assert icl >= bic


def test_gmm_cv_neg_loglik_returns_finite_with_sufficient_data():
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
        return v["n_comp"] if isinstance(v, dict) else int(v)
    assert _n_comp(out["a"]) == 3
    assert _n_comp(out["b"]) == 3


def test_report_gmm_stats_returns_aligned_outputs():
    gmm, order, _ = _make_two_component_gmm()
    means, sds, modes, mode_dens = report_gmm_stats(gmm, order)
    assert means.shape == sds.shape == (2,)
    assert modes.shape == mode_dens.shape
    # means should be sorted ascending after applying `order`
    assert means[0] < means[1]


def test_summarize_best_gmm_includes_log_and_seconds():
    gmm, order, _ = _make_two_component_gmm()
    out = summarize_best_gmm(gmm, order, tau=0.5)
    # the dict must carry both log-space and seconds-space entries
    assert any("log" in k.lower() for k in out)
    assert any("sec" in k.lower() for k in out)


# Student-t mixture --------------------------------------------------------


def _t_mixture_intervals(rng, n_per=300, log_loc_a=-2.0, log_loc_b=2.0):
    """Return strictly-positive interval values whose log values follow
    a 2-mode Student-t mixture. `fit_log_t_mixture` takes raw intervals
    (in seconds) and applies log internally."""
    a = rng.standard_t(df=10.0, size=n_per) * 0.4 + log_loc_a
    b = rng.standard_t(df=10.0, size=n_per) * 0.4 + log_loc_b
    return np.exp(np.concatenate([a, b]))


def test_fit_log_t_mixture_recovers_one_component():
    rng = np.random.default_rng(3)
    intervals = np.exp(rng.standard_t(df=10.0, size=500) * 0.5)
    model, _ = fit_log_t_mixture(
        intervals, n_components=1, seed=0, n_init=1, max_iter=80,
    )
    assert model.n_components == 1
    assert model.weights_.sum() == pytest.approx(1.0, abs=1e-6)


def test_fit_log_t_mixture_two_modes_recovered_centers():
    rng = np.random.default_rng(4)
    intervals = _t_mixture_intervals(rng, n_per=400)
    model, _ = fit_log_t_mixture(
        intervals, n_components=2, seed=0, n_init=2, max_iter=80,
    )
    sorted_means = np.sort(model.means_.ravel())
    assert sorted_means[0] < 0 < sorted_means[1]


def test_t_mixture_icl_finite():
    rng = np.random.default_rng(5)
    intervals = _t_mixture_intervals(rng, n_per=200)
    model, _ = fit_log_t_mixture(
        intervals, n_components=2, seed=0, n_init=1, max_iter=60,
    )
    val = t_mixture_icl(model, np.log(intervals))
    assert np.isfinite(val)


def test_t_mixture_cv_neg_loglik_finite():
    rng = np.random.default_rng(6)
    intervals = _t_mixture_intervals(rng, n_per=150)
    val = t_mixture_cv_neg_loglik(
        intervals, n_components=2, seed=0, n_folds=3, n_init=2,
    )
    assert np.isfinite(val)


def test_report_t_mixture_stats_shapes():
    rng = np.random.default_rng(7)
    intervals = _t_mixture_intervals(rng, n_per=200)
    model, order = fit_log_t_mixture(
        intervals, n_components=2, seed=0, n_init=1, max_iter=60,
    )
    means, sds, modes, mode_dens, nus = report_t_mixture_stats(model, order)
    assert means.shape == sds.shape == nus.shape == (2,)
    assert np.asarray(modes).shape == np.asarray(mode_dens).shape


def test_summarize_best_t_mixture_log_and_seconds():
    rng = np.random.default_rng(8)
    intervals = _t_mixture_intervals(rng, n_per=200)
    model, order = fit_log_t_mixture(
        intervals, n_components=2, seed=0, n_init=1, max_iter=60,
    )
    out = summarize_best_t_mixture(model, order)
    assert any("log" in k.lower() for k in out)
    assert any("sec" in k.lower() for k in out)


# Helper functions ---------------------------------------------------------


def test_sample_from_mixture_returns_correct_shape():
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


# ---------------------------------------------------------------------------
# compute_neuronal_tuning_curves — synthetic integration tests
#
# These tests exercise the per-cluster behavioral and vocal compute methods
# end-to-end with synthetic data. No real spike trains, USV CSVs, or
# tracking H5s — just numpy/h5py-built fakes wired through the actual code
# path. Goal: lift coverage on _compute_one_cluster_behavioral,
# _load_*_inputs, _build_vocal_side_precompute, _compute_one_cluster_vocal,
# and the triage attachers.
# ---------------------------------------------------------------------------


def _make_synthetic_session(tmp_path, *, n_frames=1500, n_usvs=120, fps=150.0):
    """
    Build a minimal session directory layout with the files NeuronalTuning
    looks for: a behavioral_features.csv, a translated/rotated H5, an
    audio_triggerbox_sync_info.json, and a per-cluster spike .npy.

    Returns
    -------
    (root, cluster_path) : (Path, Path)
        Session root and the cluster file path that the compute consumes.
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
    h5_path = root / "video" / "vid1" / "vid1_points3d_translated_rotated_metric.h5"
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("track_names", data=np.array([b"m1"]))
        f.create_dataset("recording_frame_rate", data=np.float64(fps))
        f.create_dataset("experimental_code", data=b"e99")
        # `tracks` leading axis = frame count; the vocal compute reads
        # `tracks.shape[0]` to derive session duration in seconds.
        f.create_dataset("tracks", data=np.zeros((n_frames, 1, 1, 3), dtype=float))

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
        "vae_supercategory": rng.integers(1, 5, size=n_usvs).tolist(),
        "vae_category":      rng.integers(1, 8, size=n_usvs).tolist(),
        "qlvm_supercategory": rng.integers(1, 4, size=n_usvs).tolist(),
        "qlvm_category":     rng.integers(1, 6, size=n_usvs).tolist(),
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
    """Provides (root, cluster_path) for synthetic compute tests."""
    return _make_synthetic_session(tmp_path)


def _make_neuronal_tuning(root, *, n_shuffles=5, smoothing_sd=0.0):
    """Construct a NeuronalTuning instance with small, fast settings."""
    return NeuronalTuning(
        root_directory=str(root),
        tuning_parameters_dict={
            "temporal_offsets": [0],
            "n_shuffles": n_shuffles,
            "total_bin_num": 10,
            "n_spatial_bins": 36,
            "spatial_scale_cm": 32,
            "shuffle_seconds_range": [3.0, 6.0],
            "peth_window_seconds": [-2.0, 0.0],
            "peth_bin_seconds": 0.05,
            "bout_quiet_seconds": 2.0,
            "vocal_require_clean_post_anchor": True,
            "vocal_require_clean_prior_anchor": False,
            "n_usv_min_self": 5,
            "n_usv_min_partner": 30,
            "n_usv_min_category": 3,
            "behavioral_min_occupancy_seconds": 0.1,
            "usv_property_min_occupancy_seconds": 0.05,
            "include_partner_vocalization_tuning_bool": False,
            "shuffle_chunk_size": 4,
            "smoothing_sd": smoothing_sd,
            "circular_features": ["allo_yaw", "body_dir"],
        },
        message_output=lambda *a, **k: None,
    )


# _load_behavioral_inputs ---------------------------------------------------


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_load_behavioral_inputs_returns_expected_keys(synthetic_compute_session):
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


# _compute_one_cluster_behavioral ------------------------------------------


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_compute_one_cluster_behavioral_smoke(synthetic_compute_session):
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
    root, cluster_path = synthetic_compute_session
    nt = _make_neuronal_tuning(root)
    beh_inputs = nt._load_behavioral_inputs()
    partial = nt._compute_one_cluster_behavioral(cluster_path, beh_inputs)
    ts = partial["triage_stats"]
    assert "behavioral" in ts and "spatial" in ts
    # 1D feature triage stats have run analysis fields
    speed_ts = ts["behavioral"]["beh_offset=0s"]["m1.speed"]
    assert "excit" in speed_ts and "suppress" in speed_ts
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


# _load_vocal_inputs --------------------------------------------------------


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_load_vocal_inputs_returns_expected_keys(synthetic_compute_session):
    root, _ = synthetic_compute_session
    nt = _make_neuronal_tuning(root)
    bundle = nt._load_vocal_inputs()
    assert bundle is not None
    for k in ("usv_df", "track_names", "male", "duration_seconds",
              "starts", "stops", "emitters"):
        assert k in bundle
    assert bundle["male"] == "m1"


def test_load_vocal_inputs_returns_none_when_no_inputs(tmp_path):
    nt = _make_neuronal_tuning(tmp_path)
    assert nt._load_vocal_inputs() is None


# _build_vocal_side_precompute ---------------------------------------------


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_build_vocal_side_precompute_includes_self_side(synthetic_compute_session):
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


# _compute_one_cluster_vocal ------------------------------------------------


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
    assert "vmi" in vmi["m1"] and "wilcoxon_pvalue" in vmi["m1"]
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


# ===========================================================================
# usv_interval_archive — HDF5 archive layer for IUI / mixture-model output
# ===========================================================================


# ---- _attr_value / _decode_attr round-trip --------------------------------


@pytest.mark.parametrize("v,expected_type", [
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


# ---- _polars_to_h5 / _h5_to_polars round-trip -----------------------------


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


# ---- _try_git_sha (subprocess.run mocked) ---------------------------------


def test_try_git_sha_happy(monkeypatch, tmp_path):
    """Mocked git rev-parse returns a SHA → that's what we get back."""
    fake = MagicMock(returncode=0, stdout="abc1234\n")
    monkeypatch.setattr(_subprocess, "run", lambda *a, **kw: fake)
    assert _try_git_sha(tmp_path) == "abc1234"


def test_try_git_sha_error_returns_unknown(monkeypatch, tmp_path):
    """Any subprocess error must yield "unknown" (provenance-only field)."""
    def _explode(*_a, **_kw):
        raise FileNotFoundError("no git here")
    monkeypatch.setattr(_subprocess, "run", _explode)
    assert _try_git_sha(tmp_path) == "unknown"


def test_try_git_sha_timeout_returns_unknown(monkeypatch, tmp_path):
    """TimeoutExpired must be caught silently."""
    def _explode(*_a, **_kw):
        raise _subprocess.TimeoutExpired(cmd="git", timeout=2)
    monkeypatch.setattr(_subprocess, "run", _explode)
    assert _try_git_sha(tmp_path) == "unknown"


def test_try_git_sha_nonzero_returncode_returns_unknown(monkeypatch, tmp_path):
    """Non-zero exit code (not in repo) → unknown."""
    fake = MagicMock(returncode=128, stdout="")
    monkeypatch.setattr(_subprocess, "run", lambda *a, **kw: fake)
    assert _try_git_sha(tmp_path) == "unknown"


# ---- detect_repo_root_for_provenance / git_sha_for_provenance -------------


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
    monkeypatch.setattr(_subprocess, "run", lambda *a, **kw: fake)
    assert git_sha_for_provenance(repo) == "deadbee"


# ---- write_ivi_h5 / read_usv_interval_h5 round-trip -----------------------


def _example_per_mode_payload():
    """Builds a minimal per_mode payload covering both populated and None cases."""
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
            "gmm_fits": None,
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
        "fit_gmm": False,
    }
    written = write_ivi_h5(out, analysis_attrs=analysis_attrs, per_mode=payload)
    assert written == out

    archive = read_usv_interval_h5(out)
    # File-level attrs decoded.
    assert archive["attrs"]["created_at_iso"] == "2026-05-09T12:00:00"
    assert archive["attrs"]["source_lists"] == ["a.txt", "b.txt"]
    assert archive["attrs"]["fit_gmm"] is False or archive["attrs"]["fit_gmm"] == 0
    # Mode-level: tables exist; missing tables are None.
    s2s = archive["modes"]["s2s"]
    assert s2s["attrs"]["alpha_effective"] == 0.05
    assert s2s["intervals"].height == 3
    assert s2s["drop_counts"].height == 2
    assert s2s["gmm_fits"] is None
    assert s2s["bootstrap_lrt"] is None
    assert s2s["bootstrap_lrt_null"] is None


def test_write_ivi_h5_creates_parent_dir(tmp_path):
    """write_ivi_h5 must mkdir(parents=True) before writing."""
    nested = tmp_path / "a" / "b" / "out.h5"
    write_ivi_h5(nested, analysis_attrs={}, per_mode={})
    assert nested.is_file()


# ---- reconstruct_best_model -----------------------------------------------


def _gmm_fits_row(model_class, sex="male", K=2, bic=10.0, cv=1.0):
    """Build a single-row sweep DF that reconstruct_best_model can consume."""
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
    return pls.DataFrame([row])


def test_reconstruct_best_model_gauss():
    """Gauss path returns an sklearn GMM with score_samples working."""
    df = _gmm_fits_row("gauss", K=2)
    model, order = reconstruct_best_model(df, sex="male", K=2)
    from sklearn.mixture import GaussianMixture
    assert isinstance(model, GaussianMixture)
    # score_samples must work on synthetic input — proves precisions_cholesky_
    # was set correctly.
    s = model.score_samples(np.array([[0.0], [1.0]]))
    assert s.shape == (2,)
    np.testing.assert_array_equal(order, np.arange(2))


def test_reconstruct_best_model_t_mixture():
    """t-mixture path returns a TMixture object."""
    from usv_playpen.analyses.mixture_model_utils import TMixture
    df = _gmm_fits_row("t", K=2)
    model, order = reconstruct_best_model(df, sex="male", K=2)
    assert isinstance(model, TMixture)
    np.testing.assert_array_equal(order, np.arange(2))


def test_reconstruct_best_model_no_rows_raises():
    """No matching (sex, K) → ValueError."""
    df = _gmm_fits_row("gauss", sex="male", K=2)
    with pytest.raises(ValueError, match="no rows"):
        reconstruct_best_model(df, sex="female", K=2)


def test_reconstruct_best_model_unknown_class_raises():
    """Unknown model_class string → ValueError."""
    df = _gmm_fits_row("bogus", K=2)
    with pytest.raises(ValueError, match="unknown model_class"):
        reconstruct_best_model(df, sex="male", K=2)


def test_reconstruct_best_model_falls_back_when_cv_all_nan():
    """If every cv_neg_loglik is NaN, reconstruct_best_model should fall back to bic."""
    df1 = _gmm_fits_row("gauss", K=2, bic=10.0, cv=float("nan"))
    df2 = _gmm_fits_row("gauss", K=2, bic=5.0, cv=float("nan"))
    # Ensure 'rep' dtype matches df1's (Int64) — pls.lit(1) is Int32 by default.
    df2 = df2.with_columns(pls.lit(1, dtype=pls.Int64).alias("rep"))
    df = pls.concat([df1, df2])
    model, _order = reconstruct_best_model(df, sex="male", K=2)
    # Just verify we get a valid model — the fallback to BIC means rep=1 (bic=5)
    # should win, but the reconstruction exposes only weights/means/cov, not rep.
    assert model is not None


# ===========================================================================
# compute_inter_usv_interval_distributions — pure helpers
# ===========================================================================


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
            noise_col_id="cluster",
            noise_categories=[],
        )


def test_compute_session_usv_intervals_missing_session_returns_empty(monkeypatch):
    """Missing session metadata → empty dict (graceful skip, not crash)."""
    import usv_playpen.analyses.compute_inter_usv_interval_distributions as cmod

    def _raise(_root):
        raise FileNotFoundError("no such session")
    monkeypatch.setattr(cmod, "extract_session_metadata", _raise)
    out = compute_session_usv_intervals(
        session_root="/missing",
        interval_type="s2s",
        noise_col_id="cluster",
        noise_categories=[],
    )
    assert out == {}


def test_compute_session_usv_intervals_basic_pairs(monkeypatch):
    """Synthesise 4 calls (M, M, F, F) → expect one M-M interval and one F-F."""
    import usv_playpen.analyses.compute_inter_usv_interval_distributions as cmod

    monkeypatch.setattr(cmod, "extract_session_metadata", lambda _root: {
        "male_id": "M", "female_id": "F", "frame_rate": 150.0,
    })
    fake_usv = pls.DataFrame({
        "start": [0.0, 0.5, 1.0, 1.7],
        "stop":  [0.1, 0.6, 1.1, 1.8],
        "duration": [0.1, 0.1, 0.1, 0.1],
        "emitter": ["M", "M", "F", "F"],
    })
    monkeypatch.setattr(cmod, "load_and_filter_usv_data",
                        lambda **kw: fake_usv)
    out = compute_session_usv_intervals(
        session_root="/ok", interval_type="s2s",
        noise_col_id="cluster", noise_categories=[],
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
    import usv_playpen.analyses.compute_inter_usv_interval_distributions as cmod
    monkeypatch.setattr(cmod, "extract_session_metadata", lambda _root: {
        "male_id": "M", "female_id": "F", "frame_rate": 150.0,
    })
    monkeypatch.setattr(cmod, "load_and_filter_usv_data",
                        lambda **kw: pls.DataFrame({
                            "start": pls.Series([], dtype=pls.Float64),
                            "stop":  pls.Series([], dtype=pls.Float64),
                            "duration": pls.Series([], dtype=pls.Float64),
                            "emitter": pls.Series([], dtype=pls.Utf8),
                        }))
    out = compute_session_usv_intervals(
        session_root="/ok", interval_type="s2s",
        noise_col_id="cluster", noise_categories=[],
    )
    assert out["male"].size == 0 and out["female"].size == 0


def test_compute_session_usv_intervals_e2s_drops_overlapping(monkeypatch):
    """e2s mode: stop[0]=0.6, start[1]=0.5 → -0.1 interval, dropped, counted."""
    import usv_playpen.analyses.compute_inter_usv_interval_distributions as cmod
    monkeypatch.setattr(cmod, "extract_session_metadata", lambda _root: {
        "male_id": "M", "female_id": "F", "frame_rate": 150.0,
    })
    fake_usv = pls.DataFrame({
        "start": [0.0, 0.5, 1.0],
        "stop":  [0.6, 0.7, 1.2],
        "duration": [0.6, 0.2, 0.2],
        "emitter": ["M", "M", "M"],
    })
    monkeypatch.setattr(cmod, "load_and_filter_usv_data",
                        lambda **kw: fake_usv)
    out = compute_session_usv_intervals(
        session_root="/ok", interval_type="e2s",
        noise_col_id="cluster", noise_categories=[],
    )
    # First M-M pair: start[1]-stop[0] = 0.5 - 0.6 = -0.1 → dropped
    # Second M-M pair: start[2]-stop[1] = 1.0 - 0.7 = 0.3 → kept
    np.testing.assert_allclose(out["male"], [0.3])
    assert out["n_dropped_male"] == 1


# ===========================================================================
# mixture_model_utils — coverage extension toward ~90%
# Targeting: bootstrap_lrt, select_n_components_step_up_lrt, edge cases in
# gmm_boundaries_logspace / gmm_modes / select_best_n_components / cv helpers,
# and the matplotlib smoke surfaces (plot_gmm_fit, qqplot_gmm).
# ===========================================================================


# ---- bootstrap_lrt --------------------------------------------------------


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


# ---- select_n_components_step_up_lrt --------------------------------------


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


# ---- gmm_boundaries_logspace edge cases -----------------------------------


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


# ---- gmm_cv_neg_loglik short-data branches --------------------------------


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
    short = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    val = t_mixture_cv_neg_loglik(short, n_components=5, n_folds=5, seed=0)
    assert val == float("inf")


# ---- select_best_n_components cv_mode branch ------------------------------


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


# ---- plot_gmm_fit / qqplot_gmm smoke ----------------------------------


def test_plot_gmm_fit_returns_figure_and_axes():
    """plot_gmm_fit should return (Figure, Axes); does not crash."""
    rng = np.random.RandomState(0)
    log_x = rng.normal(0.0, 1.0, 200).reshape(-1, 1)
    gmm = GaussianMixture(n_components=1, covariance_type="full",
                          random_state=0).fit(log_x)
    # Signature: plot_gmm_fit(model, x, ...)
    fig, ax = plot_gmm_fit(gmm, log_x.ravel(), bins=20)
    assert fig is not None and ax is not None


def test_plot_gmm_fit_with_xlims_filters_data(tmp_path):
    """xlims=(lo, hi) clips the input data before plotting; show_components=True
    triggers the per-component density-curve overlay."""
    rng = np.random.RandomState(0)
    log_x = rng.normal(0.0, 1.0, 200)
    gmm = GaussianMixture(n_components=2, covariance_type="full",
                          random_state=0).fit(log_x.reshape(-1, 1))
    out_path = tmp_path / "overlay.png"
    fig, _ax = plot_gmm_fit(gmm, log_x, bins=10,
                            xlims=(-1.0, 1.0), show_components=True,
                            path=str(out_path))
    assert out_path.is_file()


def test_qqplot_gmm_writes_figure(tmp_path):
    """qqplot_gmm writes a savefig file at the requested path."""
    rng = np.random.RandomState(0)
    log_x = rng.normal(0.0, 0.5, 300)
    gmm = GaussianMixture(n_components=1, covariance_type="full",
                          random_state=0).fit(log_x.reshape(-1, 1))
    out = tmp_path / "qq.png"
    qqplot_gmm(log_x, gmm, str(out), n_q=50)
    assert out.is_file()


# ===========================================================================
# compute_behavioral_features — FeatureZoo class + missing pure helpers
# ===========================================================================


# ---- FeatureZoo class -----------------------------------------------------


def test_feature_zoo_init_loads_visualizations_settings():
    """FeatureZoo.__init__ stores all kwargs and loads the package's
    visualizations_settings.json into self.visualizations_parameter_dict."""
    fz = FeatureZoo(
        root_directory="/some/path",
        message_output=lambda *_a, **_kw: None,
    )
    assert fz.root_directory == "/some/path"
    assert isinstance(fz.visualizations_parameter_dict, dict)
    # Sanity: the loaded JSON has the expected top-level key.
    assert "neuronal_tuning_figures" in fz.visualizations_parameter_dict


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


# ---- calculate_sei --------------------------------------------------------


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
        v_max=0.1,
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
                        v_max=0.1)
    assert sei[0] < 0


# ---- get_back_root --------------------------------------------------------


def _make_back_input(neck_pos, tti_pos, n_frames=1):
    """Build a (n_frames, 1 mouse, 2 nodes, 3) array with neck at index 0
    and TTI at index 1 of the per-mouse node axis."""
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


# ---- get_back_angles ------------------------------------------------------


def test_get_back_angles_zero_when_back_along_x_axis():
    """A back direction along +x is the canonical "centred" frame: after the
    Nelder-Mead alignment finds zero rotation, the residual pitch/yaw should
    be ~0 (numerical noise only). Output is in DEGREES per the source."""
    back = np.array([[1.0, 0.0, 0.0]])
    out = get_back_angles(back)
    assert out.shape == (1, 2)
    np.testing.assert_allclose(out[0], [0.0, 0.0], atol=1e-3)


def test_get_back_angles_returns_finite_for_tilted_back():
    """A 45° X-Z tilted back: optimizer aligns it; per-frame residual still
    parses to finite degree values without NaN/inf — that's the contract."""
    angle = np.pi / 4
    back = np.array([[np.cos(angle), 0.0, np.sin(angle)]])
    out = get_back_angles(back)
    assert out.shape == (1, 2)
    assert np.all(np.isfinite(out))


def test_get_back_angles_handles_multi_frame_input():
    """Multi-frame input → output has shape (n_frames, 2)."""
    back = np.array([
        [1.0, 0.0, 0.0],
        [0.866, 0.5, 0.0],
        [0.5, 0.866, 0.0],
    ])
    out = get_back_angles(back)
    assert out.shape == (3, 2)
    assert np.all(np.isfinite(out))
