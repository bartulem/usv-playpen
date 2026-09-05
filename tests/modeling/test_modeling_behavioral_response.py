"""
@author: bartulem
Unit tests for ``usv_playpen.modeling.modeling_behavioral_response``
— the event-anchored extraction that contrasts female behaviour after male
vocal bouts against comparable inter-bout silence.

The invariants worth pinning are the ones that would silently corrupt the
contrast rather than crash it. Coverage:

* ``bout_offset_anchors`` — the anchor sits at the bout's OFFSET, a bout whose
  successor arrives inside the silence window is dropped, and an anchor with
  no room for its history or forward window is dropped.
* ``inter_bout_quiet_anchors`` — one anchor per usable gap, every anchor has a
  clean history AND a clean forward window, placement is reproducible from the
  seed, and a gap too short yields nothing rather than a bad anchor.
* ``summarise_history`` — window means are exact and NaN-aware, and a window
  with no finite sample yields NaN rather than zero.
* ``forward_window_mean`` — the window is strictly forward of the anchor, and
  out-of-bounds samples are EXCLUDED rather than clamped.
* ``_response_likelihood`` — derived from the feature's post-fold support, so a
  signed feature can never be handed to a Gamma likelihood.
* ``BehavioralResponsePipeline`` — seconds convert to frames on the configured
  camera grid, the response bins tile the window exactly, and a degenerate
  configuration is rejected loudly.
* The shipped ``behavioral_response`` settings block — every key the pipeline
  reads by name exists.
* ``duration_tercile_labels`` / ``build_design_matrix`` / ``fit_contrast`` — the
  contrast half: bands hold equal numbers of BOUTS, quiet rows load no band, a
  planted step and dose-response are recovered, the false-positive rate is near
  nominal, session clustering widens the interval, and rank-deficient or
  unusable designs raise rather than returning a meaningless fit.
"""

from __future__ import annotations

import importlib.resources
import json

import numpy as np
import pytest

from usv_playpen.modeling.modeling_behavioral_response import (
    BehavioralResponsePipeline,
    bout_offset_anchors,
    build_design_matrix,
    duration_tercile_labels,
    fit_contrast,
    forward_window_mean,
    inter_bout_quiet_anchors,
    summarise_history,
)

CAMERA_FPS = 150.0
HISTORY_FRAMES = 600      # 4 s
LOOKAHEAD_FRAMES = 75     # 0.5 s
SESSION_FRAMES = 150 * 60


def _shipped_settings() -> dict:
    """
    Loads the settings block the package actually ships.

    Parameters
    ----------
    None

    Returns
    -------
    settings : dict
        Parsed ``modeling_settings.json``.
    """

    resource = (importlib.resources.files('usv_playpen')
                / '_parameter_settings' / 'modeling_settings.json')
    return json.loads(resource.read_text())


class TestBoutOffsetAnchors:
    """Anchors must sit at bout ends and leave the forward window clean."""

    def test_anchor_is_the_offset_not_the_onset(self):
        """Anchoring on the onset would put the bout inside the target window."""

        frames, _ = bout_offset_anchors(
            np.array([10.0]), np.array([0.5]), CAMERA_FPS, SESSION_FRAMES,
            HISTORY_FRAMES, LOOKAHEAD_FRAMES, silence_seconds=0.5)

        assert frames.tolist() == [int(round(10.5 * CAMERA_FPS))]

    def test_a_bout_followed_too_soon_is_dropped(self):
        """Otherwise the forward window would contain the next bout."""

        onsets = np.array([10.0, 10.7])
        durations = np.array([0.5, 0.3])
        frames, _ = bout_offset_anchors(
            onsets, durations, CAMERA_FPS, SESSION_FRAMES,
            HISTORY_FRAMES, LOOKAHEAD_FRAMES, silence_seconds=0.5)

        # 10.5 has only 0.2 s before the next onset; 11.0 has no successor.
        assert frames.tolist() == [int(round(11.0 * CAMERA_FPS))]

    def test_durations_stay_aligned_with_surviving_anchors(self):
        """A misaligned duration would mislabel every bout's dose."""

        onsets = np.array([10.0, 10.7, 30.0])
        durations = np.array([0.5, 0.3, 1.25])
        frames, kept = bout_offset_anchors(
            onsets, durations, CAMERA_FPS, SESSION_FRAMES,
            HISTORY_FRAMES, LOOKAHEAD_FRAMES, silence_seconds=0.5)

        assert frames.size == kept.size == 2
        assert kept.tolist() == [0.3, 1.25]

    def test_a_bout_without_room_for_its_history_is_dropped(self):
        """A truncated history would silently compare unequal windows."""

        frames, _ = bout_offset_anchors(
            np.array([1.0]), np.array([0.1]), CAMERA_FPS, SESSION_FRAMES,
            HISTORY_FRAMES, LOOKAHEAD_FRAMES, silence_seconds=0.5)

        assert frames.size == 0

    def test_no_bouts_yields_empty_arrays_rather_than_raising(self):
        """A silent session is ordinary, not exceptional."""

        frames, kept = bout_offset_anchors(
            np.empty(0), np.empty(0), CAMERA_FPS, SESSION_FRAMES,
            HISTORY_FRAMES, LOOKAHEAD_FRAMES, silence_seconds=0.5)

        assert frames.size == 0 and kept.size == 0


class TestInterBoutQuietAnchors:
    """The silent condition must be genuinely silent on both sides."""

    @staticmethod
    def _bouts() -> tuple[np.ndarray, np.ndarray]:
        """
        Three bouts leaving one long gap, one short gap and one long gap.

        Parameters
        ----------
        None

        Returns
        -------
        onsets, durations : tuple of np.ndarray
            Bout onsets and durations in seconds.
        """

        return np.array([10.0, 20.0, 21.0, 40.0]), np.array([0.5, 0.5, 0.3, 0.4])

    def test_one_anchor_per_usable_gap(self):
        """Tiling long gaps would let a few silences dominate the condition."""

        onsets, durations = self._bouts()
        anchors = inter_bout_quiet_anchors(
            onsets, durations, CAMERA_FPS, SESSION_FRAMES, HISTORY_FRAMES,
            LOOKAHEAD_FRAMES, np.random.default_rng(0))

        # 10.5->20 and 21.3->40 are usable; 20.5->21 is far too short.
        assert anchors.size == 2

    def test_every_anchor_has_clean_history_and_forward_window(self):
        """A quiet anchor touching a call is not a silent observation."""

        onsets, durations = self._bouts()
        offsets = onsets + durations
        anchors = inter_bout_quiet_anchors(
            onsets, durations, CAMERA_FPS, SESSION_FRAMES, HISTORY_FRAMES,
            LOOKAHEAD_FRAMES, np.random.default_rng(3))

        history_seconds = HISTORY_FRAMES / CAMERA_FPS
        forward_seconds = LOOKAHEAD_FRAMES / CAMERA_FPS
        for anchor in anchors / CAMERA_FPS:
            overlaps = [
                (anchor - history_seconds) < stop and start < (anchor + forward_seconds)
                for start, stop in zip(onsets, offsets, strict=True)
            ]
            assert not any(overlaps)

    def test_placement_is_reproducible_from_the_seed(self):
        """A run must be repeatable, and the seed is the only source of jitter."""

        onsets, durations = self._bouts()
        first = inter_bout_quiet_anchors(
            onsets, durations, CAMERA_FPS, SESSION_FRAMES, HISTORY_FRAMES,
            LOOKAHEAD_FRAMES, np.random.default_rng(7))
        second = inter_bout_quiet_anchors(
            onsets, durations, CAMERA_FPS, SESSION_FRAMES, HISTORY_FRAMES,
            LOOKAHEAD_FRAMES, np.random.default_rng(7))

        assert first.tolist() == second.tolist()

    def test_gaps_too_short_yield_nothing(self):
        """Better no silent row than one contaminated by a call."""

        anchors = inter_bout_quiet_anchors(
            np.array([10.0, 12.0]), np.array([0.2, 0.2]), CAMERA_FPS, SESSION_FRAMES,
            HISTORY_FRAMES, LOOKAHEAD_FRAMES, np.random.default_rng(0))

        assert anchors.size == 0

    def test_a_single_bout_has_no_gap(self):
        """A gap needs two bouts to bracket it."""

        anchors = inter_bout_quiet_anchors(
            np.array([10.0]), np.array([0.2]), CAMERA_FPS, SESSION_FRAMES,
            HISTORY_FRAMES, LOOKAHEAD_FRAMES, np.random.default_rng(0))

        assert anchors.size == 0


class TestSummariseHistory:
    """Covariates must describe the window BEFORE the anchor, exactly."""

    def test_means_are_exact_and_backward_looking(self):
        """A forward-looking covariate would leak the response into the control."""

        values = np.arange(1000, dtype=float)
        summaries = summarise_history(values, np.array([500]), [10, 100])

        assert summaries[0, 0] == pytest.approx(np.mean(np.arange(490, 500)))
        assert summaries[0, 1] == pytest.approx(np.mean(np.arange(400, 500)))

    def test_nan_samples_are_ignored_not_propagated(self):
        """Out-of-bounds frames are nulled upstream and must not void the window."""

        values = np.arange(1000, dtype=float)
        values[495:500] = np.nan
        summaries = summarise_history(values, np.array([500]), [10])

        assert summaries[0, 0] == pytest.approx(np.mean(np.arange(490, 495)))

    def test_an_all_nan_window_yields_nan(self):
        """The caller must be able to see that a row has no covariate."""

        values = np.full(1000, np.nan)
        summaries = summarise_history(values, np.array([500]), [10])

        assert np.isnan(summaries[0, 0])


class TestForwardWindowMean:
    """The response window must be forward-only and bound-respecting."""

    def test_window_is_strictly_forward_of_the_anchor(self):
        """Overlapping the history would make the row predict its own input."""

        values = np.concatenate([np.zeros(100), np.full(100, 5.0)])
        mean = forward_window_mean(values, np.array([100]), 50, 0.0, 54.0)

        assert mean[0] == pytest.approx(5.0)

    def test_out_of_bounds_samples_are_excluded_not_clamped(self):
        """Clamping would enter a fabricated boundary value as an observation."""

        values = np.array([1.0, 2.0, 1e7, 3.0, 4.0])
        mean = forward_window_mean(values, np.array([0]), 5, 0.0, 54.0)

        assert mean[0] == pytest.approx(2.5)

    def test_a_window_with_no_in_bounds_sample_yields_nan(self):
        """Such a row carries no response and must be droppable."""

        values = np.full(10, 1e7)
        mean = forward_window_mean(values, np.array([0]), 5, 0.0, 54.0)

        assert np.isnan(mean[0])


class TestShippedSettingsBlock:
    """The pipeline reads these keys by name, so they must exist."""

    def test_every_key_the_pipeline_reads_is_present(self):
        """A missing key should fail here rather than mid-extraction."""

        block = _shipped_settings()['behavioral_response']
        expected = {'response_mouse_index', 'response_features', 'history_seconds',
                    'target_window_seconds', 'target_bin_seconds',
                    'post_bout_silence_seconds', 'covariate_summary_seconds',
                    'duration_n_bins'}

        assert expected <= set(block)

    def test_response_features_are_all_known_kinematic_features(self):
        """A typo here would surface only once extraction reached that session."""

        settings = _shipped_settings()
        block = settings['behavioral_response']

        assert set(block['response_features']) <= set(settings['kinematic_features']['egocentric'])

    def test_the_mouse_index_is_an_absolute_slot(self):
        """Role strings would have to be read against another key to mean anything."""

        block = _shipped_settings()['behavioral_response']

        assert block['response_mouse_index'] in (0, 1)


class TestPipelineGeometry:
    """Seconds become frames on the configured grid, or the run stops."""

    @staticmethod
    def _settings(**overrides) -> dict:
        """
        Shipped settings with the response block overridden.

        Parameters
        ----------
        **overrides
            Keys to replace inside ``behavioral_response``.

        Returns
        -------
        settings : dict
            Modified settings.
        """

        settings = _shipped_settings()
        settings['behavioral_response'].update(overrides)
        return settings

    def test_seconds_convert_on_the_camera_grid(self):
        """A wrong conversion would silently change every window width."""

        pipeline = BehavioralResponsePipeline(modeling_settings_dict=self._settings())
        fps = pipeline.modeling_settings['io']['camera_sampling_rate']

        assert pipeline.response_history_frames == int(np.floor(fps * 4.0))
        assert pipeline.response_window_frames == int(np.floor(fps * 0.5))

    def test_response_bins_tile_the_window_exactly(self):
        """A floored bin width would leave the window's tail outside the curve."""

        pipeline = BehavioralResponsePipeline(modeling_settings_dict=self._settings())
        widths = np.diff(pipeline.response_bin_edges)

        assert widths.sum() == pipeline.response_window_frames
        assert pipeline.n_response_bins == 10

    def test_a_degenerate_window_is_rejected(self):
        """Sub-frame windows would produce empty targets, not small ones."""

        with pytest.raises(ValueError, match='at least one frame'):
            BehavioralResponsePipeline(
                modeling_settings_dict=self._settings(target_window_seconds=1e-6))

    def test_a_bin_wider_than_the_window_is_rejected(self):
        """The time course must fit inside the window it resolves."""

        with pytest.raises(ValueError, match='exceeds'):
            BehavioralResponsePipeline(
                modeling_settings_dict=self._settings(target_bin_seconds=5.0))

    def test_the_predictor_index_is_derived_from_the_response_index(self):
        """Setting both by hand is how they end up contradicting each other."""

        pipeline = BehavioralResponsePipeline(
            modeling_settings_dict=self._settings(response_mouse_index=1))

        assert pipeline.modeling_settings['model_params']['model_predictor_mouse_index'] == 0

    def test_deriving_the_predictor_does_not_touch_the_shared_block(self):
        """The five vocal pipelines read that same block."""

        settings = self._settings(response_mouse_index=1)
        original = settings['model_params']['model_predictor_mouse_index']
        BehavioralResponsePipeline(modeling_settings_dict=settings)

        assert settings['model_params']['model_predictor_mouse_index'] == original


class TestDerivedLikelihood:
    """A signed feature must never reach a Gamma likelihood."""

    @staticmethod
    def _pipeline() -> BehavioralResponsePipeline:
        """
        Builds a pipeline on the shipped settings.

        Parameters
        ----------
        None

        Returns
        -------
        pipeline : BehavioralResponsePipeline
            Pipeline instance.
        """

        return BehavioralResponsePipeline(modeling_settings_dict=_shipped_settings())

    def test_signed_features_get_gaussian(self):
        """Gamma would discard every negative row without saying so."""

        pipeline = self._pipeline()

        assert pipeline._response_likelihood('allo_pitch') == 'gaussian'
        assert pipeline._response_likelihood('back_pitch') == 'gaussian'

    def test_non_negative_features_get_gamma(self):
        """Speed is positive and right-skewed, which is what Gamma is for."""

        pipeline = self._pipeline()

        assert pipeline._response_likelihood('speed') == 'gamma'
        assert pipeline._response_likelihood('neck_elevation') == 'gamma'

    def test_folded_features_get_gamma_despite_signed_bounds(self):
        """The magnitude fold maps a signed angle onto a non-negative support."""

        pipeline = self._pipeline()

        assert pipeline._response_fold_label('ego_yaw') == 'smooth_abs'
        assert pipeline._response_likelihood('ego_yaw') == 'gamma'
        assert pipeline._response_fold_label('allo_roll') == 'abs'
        assert pipeline._response_likelihood('allo_roll') == 'gamma'

    def test_an_unfolded_feature_reports_no_fold(self):
        """The provenance must say which branch actually fired."""

        assert self._pipeline()._response_fold_label('speed') == 'none'


def _synthetic_rows(n_sessions: int = 40,
                    per_session: int = 60,
                    seed: int = 0) -> dict:
    """
    Builds a clustered anchor table with known structure.

    Parameters
    ----------
    n_sessions : int
        Number of sessions, the clustering unit.
    per_session : int
        Anchors per session.
    seed : int
        Seed for the generator.

    Returns
    -------
    rows : dict
        ``session_ids``, ``is_vocal``, ``duration``, ``covariates``,
        ``covariate_labels`` and the per-row session offset used to build them.
    """

    rng = np.random.default_rng(seed)
    n_rows = n_sessions * per_session
    session_ids = np.repeat([f's{index:02d}' for index in range(n_sessions)], per_session)
    is_vocal = (rng.random(n_rows) < 0.6).astype(float)
    duration = np.where(is_vocal > 0.0, rng.gamma(2.0, 0.25, n_rows), np.nan)
    covariates = rng.normal(size=(n_rows, 4))
    session_offset = np.repeat(rng.normal(0.0, 0.15, n_sessions), per_session)
    return {
        'session_ids': session_ids,
        'is_vocal': is_vocal,
        'duration': duration,
        'covariates': covariates,
        'covariate_labels': [f'cov{index}' for index in range(4)],
        'session_offset': session_offset,
        'rng': rng,
    }


class TestDurationTercileLabels:
    """Duration bands must describe bouts, not anchors."""

    def test_bands_hold_equal_numbers_of_bouts(self):
        """Equal-count bands are what make the steps comparable."""

        rows = _synthetic_rows()
        band, _ = duration_tercile_labels(rows['duration'], rows['is_vocal'], 3)
        counts = [int((band == index).sum()) for index in range(3)]

        assert max(counts) - min(counts) <= 1

    def test_quiet_rows_are_not_banded(self):
        """A quiet row has no duration, so it belongs to no band."""

        rows = _synthetic_rows()
        band, _ = duration_tercile_labels(rows['duration'], rows['is_vocal'], 3)

        assert np.all(band[rows['is_vocal'] == 0.0] == -1)
        assert np.all(band[rows['is_vocal'] > 0.0] >= 0)

    def test_edges_span_the_observed_durations(self):
        """Cut points outside the data would leave a band empty."""

        rows = _synthetic_rows()
        _, edges = duration_tercile_labels(rows['duration'], rows['is_vocal'], 3)
        vocal_durations = rows['duration'][rows['is_vocal'] > 0.0]

        assert edges[0] == pytest.approx(vocal_durations.min())
        assert edges[-1] == pytest.approx(vocal_durations.max())

    def test_too_few_distinct_durations_raises(self):
        """Silently collapsing bands would misreport the dose-response."""

        is_vocal = np.array([1.0, 1.0, 0.0])
        duration = np.array([0.2, 0.2, np.nan])

        with pytest.raises(ValueError, match='distinct bout durations'):
            duration_tercile_labels(duration, is_vocal, 3)


class TestBuildDesignMatrix:
    """The design must keep silence and each duration band separable."""

    def test_quiet_rows_are_zero_on_every_band_column(self):
        """A quiet row loading a band would blur the contrast it defines."""

        rows = _synthetic_rows()
        band, _ = duration_tercile_labels(rows['duration'], rows['is_vocal'], 3)
        design, _ = build_design_matrix(rows['covariates'], rows['is_vocal'], band, 3,
                                        rows['covariate_labels'])
        quiet = rows['is_vocal'] == 0.0

        assert np.all(design[quiet, 1:4] == 0.0)

    def test_each_vocal_row_loads_exactly_one_band(self):
        """Two bands on one row would double-count that bout."""

        rows = _synthetic_rows()
        band, _ = duration_tercile_labels(rows['duration'], rows['is_vocal'], 3)
        design, _ = build_design_matrix(rows['covariates'], rows['is_vocal'], band, 3,
                                        rows['covariate_labels'])
        vocal = rows['is_vocal'] > 0.0

        assert np.all(design[vocal, 1:4].sum(axis=1) == 1.0)

    def test_labels_line_up_with_columns(self):
        """Misaligned labels would attribute a coefficient to the wrong term."""

        rows = _synthetic_rows()
        band, _ = duration_tercile_labels(rows['duration'], rows['is_vocal'], 3)
        design, labels = build_design_matrix(rows['covariates'], rows['is_vocal'], band, 3,
                                             rows['covariate_labels'])

        assert len(labels) == design.shape[1]
        assert labels[0] == 'intercept'
        assert labels[1:4] == ['vocal_duration_band_0', 'vocal_duration_band_1',
                               'vocal_duration_band_2']
        assert labels[4:] == rows['covariate_labels']


class TestFitContrast:
    """A coefficient that is wrong still looks like a coefficient."""

    @staticmethod
    def _fit(planted_step: float, planted_slope: float, seed: int = 0) -> tuple:
        """
        Plants a known effect and fits it back.

        Parameters
        ----------
        planted_step : float
            Log-scale step applied to every vocal row.
        planted_slope : float
            Log-scale change per second of bout duration.
        seed : int
            Seed for the generator.

        Returns
        -------
        fit, edges : tuple
            The fit results and the duration band edges.
        """

        rows = _synthetic_rows(seed=seed)
        rng = rows['rng']
        eta = (1.0
               + planted_step * rows['is_vocal']
               + planted_slope * np.nan_to_num(rows['duration'])
               + 0.5 * rows['covariates'][:, 0]
               + rows['session_offset'])
        target = rng.gamma(shape=20.0, scale=np.exp(eta) / 20.0)

        band, edges = duration_tercile_labels(rows['duration'], rows['is_vocal'], 3)
        design, labels = build_design_matrix(rows['covariates'], rows['is_vocal'], band, 3,
                                             rows['covariate_labels'])
        fit = fit_contrast(target, design, labels, rows['session_ids'], 'gamma')
        return fit, edges

    def test_a_planted_step_is_recovered(self):
        """The headline number must track the truth, not merely be significant."""

        fit, _ = self._fit(planted_step=0.30, planted_slope=0.0)
        term = fit['terms']['vocal_duration_band_0']

        assert term['coefficient'] == pytest.approx(0.30, abs=0.08)
        assert term['p_value'] < 1e-6

    def test_a_planted_dose_response_comes_back_monotone(self):
        """Non-monotone bands would misstate question 2 entirely."""

        fit, _ = self._fit(planted_step=0.30, planted_slope=0.20)
        betas = [fit['terms'][f'vocal_duration_band_{band}']['coefficient']
                 for band in range(3)]

        assert betas[0] < betas[1] < betas[2]

    def test_the_false_positive_rate_is_near_nominal(self):
        """A single ns fit proves nothing; the REJECTION RATE is the calibration.

        Asserting one seed comes back ns would be flaky by construction -- three
        bands at alpha 0.05 reject on roughly one seed in seven -- and would pass
        by luck rather than by the test being calibrated.
        """

        rejections, total = 0, 0
        for seed in range(20):
            fit, _ = self._fit(planted_step=0.0, planted_slope=0.0, seed=seed)
            for band in range(3):
                rejections += fit['terms'][f'vocal_duration_band_{band}']['p_value'] < 0.05
                total += 1

        # 60 tests at a true 5% rate: P(>= 9 rejections) is under 2%.
        assert rejections / total < 0.15, f'{rejections}/{total} rejected under the null'

    def test_clustering_widens_the_interval_when_the_data_cluster(self):
        """Naive errors would be several times too narrow on real sessions."""

        rng = np.random.default_rng(1)
        n_sessions, per_session = 40, 60
        n_rows = n_sessions * per_session
        session_ids = np.repeat([f's{i:02d}' for i in range(n_sessions)], per_session)
        covariates = rng.normal(size=(n_rows, 2))
        # `vocal` varies mostly BETWEEN sessions: the regime clustering exists for.
        session_rate = np.repeat(rng.random(n_sessions), per_session)
        is_vocal = (rng.random(n_rows) < session_rate).astype(float)
        band = np.where(is_vocal > 0.0, 0, -1)
        eta = 1.0 + 0.4 * covariates[:, 0] + np.repeat(rng.normal(0.0, 0.6, n_sessions),
                                                       per_session)
        target = rng.gamma(shape=20.0, scale=np.exp(eta) / 20.0)

        design, labels = build_design_matrix(covariates, is_vocal, band, 1, ['c0', 'c1'])
        clustered = fit_contrast(target, design, labels, session_ids, 'gamma')
        naive = fit_contrast(target, design, labels, np.arange(n_rows).astype(str), 'gamma')

        clustered_se = clustered['terms']['vocal_duration_band_0']['std_error']
        naive_se = naive['terms']['vocal_duration_band_0']['std_error']
        assert clustered_se > 2.0 * naive_se

    def test_non_finite_rows_are_dropped_and_counted(self):
        """A silently shrinking sample is how an underpowered fit looks fine."""

        rows = _synthetic_rows()
        rng = rows['rng']
        target = rng.gamma(shape=20.0, scale=np.exp(1.0) / 20.0, size=rows['is_vocal'].size)
        target[:25] = np.nan

        band, _ = duration_tercile_labels(rows['duration'], rows['is_vocal'], 3)
        design, labels = build_design_matrix(rows['covariates'], rows['is_vocal'], band, 3,
                                             rows['covariate_labels'])
        fit = fit_contrast(target, design, labels, rows['session_ids'], 'gamma')

        assert fit['n_rows_dropped'] == 25
        assert fit['n_rows_fitted'] == target.size - 25

    def test_a_rank_deficient_design_raises_with_the_offending_columns(self):
        """statsmodels would otherwise raise a bare 'Singular matrix'."""

        rows = _synthetic_rows(n_sessions=4, per_session=5)
        band = np.full(rows['is_vocal'].size, -1)          # no vocal row loads any band
        design, labels = build_design_matrix(rows['covariates'], np.zeros_like(rows['is_vocal']),
                                             band, 2, rows['covariate_labels'])
        target = np.abs(rows['covariates'][:, 0]) + 1.0

        with pytest.raises(ValueError, match='rank-deficient'):
            fit_contrast(target, design, labels, rows['session_ids'], 'gamma')

    def test_an_unknown_likelihood_raises(self):
        """A typo must not silently fall through to a default family."""

        rows = _synthetic_rows(n_sessions=4, per_session=10)
        band, _ = duration_tercile_labels(rows['duration'], rows['is_vocal'], 2)
        design, labels = build_design_matrix(rows['covariates'], rows['is_vocal'], band, 2,
                                             rows['covariate_labels'])

        with pytest.raises(ValueError, match="must be 'gamma' or 'gaussian'"):
            fit_contrast(np.ones(design.shape[0]), design, labels, rows['session_ids'], 'poisson')

    def test_gaussian_accepts_negative_targets(self):
        """Signed features exist precisely because Gamma cannot take them."""

        rows = _synthetic_rows()
        rng = rows['rng']
        target = (0.5 * rows['is_vocal'] + rows['covariates'][:, 0]
                  + rng.normal(0.0, 1.0, rows['is_vocal'].size))

        band, _ = duration_tercile_labels(rows['duration'], rows['is_vocal'], 3)
        design, labels = build_design_matrix(rows['covariates'], rows['is_vocal'], band, 3,
                                             rows['covariate_labels'])
        fit = fit_contrast(target, design, labels, rows['session_ids'], 'gaussian')

        assert np.any(target < 0.0)
        assert fit['n_rows_dropped'] == 0
