"""
@author: bartulem
Test analyses module.
"""

import json
import pathlib
import pytest
from usv_playpen.analyze_data import Analyst
from usv_playpen.analyses.decode_experiment_label import extract_information
import numpy as np
from usv_playpen.analyses.compute_behavioral_features import (
    calculate_derivatives,
    calculate_speed,
    calculate_tail_curvature,
    generate_feature_distributions,
    get_egocentric_direction,
    get_euler_ang,
    get_head_root,
)
from usv_playpen.analyses.compute_behavioral_tuning_curves import (
    generate_ratemaps,
    shuffle_spikes,
)
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
from usv_playpen.analyses.gmm_utils import (
    TMixture,
    _alpha,
    _extract_params,
    _log_gauss,
    _t_logpdf_1d,
    fit_log_gmm,
    gmm_boundaries_logspace,
    gmm_cdf_logspace,
    gmm_quantile_logspace,
    t_mixture_cdf_logspace,
    t_mixture_quantile_logspace,
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
