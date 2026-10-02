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
* ``deep_silence_anchors`` — every anchor is at least the configured margin
  from any bout, anchors are stride-spaced so rows do not overlap, and a session
  with no long-enough stretch yields nothing.
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
* ``select_covariate_columns`` — the fit-time choice of which animal's pre-anchor
  kinematics adjust the contrast; dyadic columns survive every choice.
* ``duration_tercile_labels`` / ``build_design_matrix`` / ``fit_contrast`` — the
  contrast half: bands hold equal numbers of BOUTS, quiet rows load no band, a
  planted step and dose-response are recovered, the false-positive rate is near
  nominal, session clustering widens the interval, and rank-deficient or
  unusable designs raise rather than returning a meaningless fit.
* ``BehavioralResponsePipeline.extract_and_save_modeling_input_data`` — the
  REAL extraction run end-to-end on a synthetic on-disk session tree (built
  with ``tests/modeling/_synth.py``): the published pickle's row table is
  aligned across every array, carries the three condition levels, and its
  provenance counts agree with the rows; sessions whose predictor never calls,
  or whose response is never in bounds, are dropped; degenerate cohorts raise.
* ``behavioral_response_contrast`` — run on that extracted pickle under every
  covariate transform, both controls, every covariate set, the legacy
  two-condition artifact layout, and the misconfigurations it must refuse.
* ``MatchedDivergencePipeline.extract_and_save_matched_divergence`` — the
  matched-pair artifact on the same tree: curve widths, pair bookkeeping, the
  baseline balance matching exists to buy, and the all-rejected caliper case.

Warning policy
--------------
The project runs pytest with ``filterwarnings = ["error"]``. The modeling
import chain pulls ``optax`` -> a one-time JAX ``DeprecationWarning``, so the
module import below is wrapped in a ``warnings.catch_warnings`` block that
ignores ``DeprecationWarning`` during import. The synthetic extractions below
were checked to run warning-free, so no per-test marker is needed.
"""

from __future__ import annotations

import copy
import importlib.resources
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import polars as pls
import pytest

from scipy import stats

from tests.modeling._synth import (
    build_modeling_settings,
    build_session_tree,
    write_session_list_file,
)

# The modeling import chain pulls optax -> a one-time JAX DeprecationWarning.
# Guard the import so collection does not trip ``filterwarnings = ["error"]``.
with warnings.catch_warnings():
    warnings.simplefilter('ignore', DeprecationWarning)
    from usv_playpen.modeling.modeling_behavioral_response import (
        BehavioralResponsePipeline,
        MatchedDivergencePipeline,
        _all_usv_times,
        behavioral_response_contrast,
        bout_offset_anchors,
        build_continuous_design_matrix,
        build_design_matrix,
        deep_silence_anchors,
        derived_covariate_transform,
        duration_tercile_labels,
        fit_contrast,
        forward_window_mean,
        forward_clean_times,
        inter_bout_quiet_anchors,
        normal_scores,
        select_covariate_columns,
        summarise_history,
        variance_explained_by_vocal_terms,
        yeo_johnson_covariates,
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

    def test_non_negative_features_get_lognormal(self):
        """The primary estimand is multiplicative but on the geometric mean."""

        pipeline = self._pipeline()

        assert pipeline._response_likelihood('speed') == 'lognormal'
        assert pipeline._response_likelihood('neck_elevation') == 'lognormal'

    def test_folded_features_get_lognormal_despite_signed_bounds(self):
        """The magnitude fold maps a signed angle onto a non-negative support."""

        pipeline = self._pipeline()

        assert pipeline._response_fold_label('ego_yaw') == 'smooth_abs'
        assert pipeline._response_likelihood('ego_yaw') == 'lognormal'
        assert pipeline._response_fold_label('allo_roll') == 'abs'
        assert pipeline._response_likelihood('allo_roll') == 'lognormal'

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

        with pytest.raises(ValueError, match="must be 'lognormal', 'gamma' or 'gaussian'"):
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


class TestDeepSilenceAnchors:
    """The second control: stretches where he has not called for a long while."""

    @staticmethod
    def _bouts() -> tuple[np.ndarray, np.ndarray]:
        """
        Two bouts far apart, leaving one long silent stretch between them.

        Parameters
        ----------
        None

        Returns
        -------
        onsets, durations : tuple of np.ndarray
            Bout onsets and durations in seconds.
        """

        return np.array([10.0, 120.0]), np.array([0.5, 0.5])

    def test_every_anchor_clears_the_margin_from_any_bout(self):
        """An anchor inside the margin is not deep silence, whatever it is."""

        onsets, durations = self._bouts()
        anchors = deep_silence_anchors(
            onsets, durations, CAMERA_FPS, 150 * 200, HISTORY_FRAMES, LOOKAHEAD_FRAMES,
            margin_seconds=10.0, stride_frames=HISTORY_FRAMES)

        assert anchors.size > 0
        for anchor in anchors / CAMERA_FPS:
            assert np.all(np.abs(onsets - anchor) > 10.0)
            assert np.all(np.abs((onsets + durations) - anchor) > 10.0)

    def test_anchors_are_stride_spaced_so_rows_do_not_overlap(self):
        """Overlapping rows would be counted as independent when they are not."""

        onsets, durations = self._bouts()
        anchors = deep_silence_anchors(
            onsets, durations, CAMERA_FPS, 150 * 200, HISTORY_FRAMES, LOOKAHEAD_FRAMES,
            margin_seconds=10.0, stride_frames=HISTORY_FRAMES)

        assert np.all(np.diff(anchors) >= HISTORY_FRAMES - 1)

    def test_a_widely_spaced_margin_can_exclude_everything(self):
        """A margin longer than the session must yield nothing, not a bad anchor."""

        onsets, durations = self._bouts()
        anchors = deep_silence_anchors(
            onsets, durations, CAMERA_FPS, 150 * 200, HISTORY_FRAMES, LOOKAHEAD_FRAMES,
            margin_seconds=500.0, stride_frames=HISTORY_FRAMES)

        assert anchors.size == 0

    def test_no_bouts_yields_nothing(self):
        """With no bouts there is no vocal condition to control for."""

        anchors = deep_silence_anchors(
            np.empty(0), np.empty(0), CAMERA_FPS, 150 * 200, HISTORY_FRAMES,
            LOOKAHEAD_FRAMES, margin_seconds=10.0, stride_frames=HISTORY_FRAMES)

        assert anchors.size == 0

    def test_it_yields_different_anchors_than_the_inter_bout_control(self):
        """The two controls must not silently be the same rows."""

        onsets, durations = self._bouts()
        deep = deep_silence_anchors(
            onsets, durations, CAMERA_FPS, 150 * 200, HISTORY_FRAMES, LOOKAHEAD_FRAMES,
            margin_seconds=10.0, stride_frames=HISTORY_FRAMES)
        inter = inter_bout_quiet_anchors(
            onsets, durations, CAMERA_FPS, 150 * 200, HISTORY_FRAMES, LOOKAHEAD_FRAMES,
            np.random.default_rng(0))

        assert deep.size > inter.size


class TestContinuousDesignMatrix:
    """The primary design: one vocal step plus one duration slope."""

    @staticmethod
    def _rows(seed: int = 0):
        """
        Builds vocal and control rows with skewed durations.

        Parameters
        ----------
        seed : int
            Seed for the generator.

        Returns
        -------
        covariates, is_vocal, duration : tuple of np.ndarray
            Covariates, condition indicator and per-row bout duration.
        """

        rng = np.random.default_rng(seed)
        n = 600
        is_vocal = (rng.random(n) < 0.6).astype(float)
        duration = np.where(is_vocal > 0.0, rng.lognormal(-0.9, 0.7, n), np.nan)
        return rng.normal(size=(n, 2)), is_vocal, duration

    def test_control_rows_are_zero_on_both_vocal_terms(self):
        """A control row loading either term would blur the contrast."""

        cov, is_vocal, duration = self._rows()
        design, labels = build_continuous_design_matrix(cov, is_vocal, duration, ['c0', 'c1'])
        control = is_vocal == 0.0

        assert labels[1:3] == ['vocal', 'vocal_x_log_duration']
        assert np.all(design[control, 1] == 0.0)
        assert np.all(design[control, 2] == 0.0)

    def test_the_duration_slope_is_centred_so_vocal_is_the_average_bout(self):
        """Otherwise the step would be the effect of a zero-length bout."""

        cov, is_vocal, duration = self._rows()
        design, _ = build_continuous_design_matrix(cov, is_vocal, duration, ['c0', 'c1'])

        assert design[is_vocal > 0.0, 2].mean() == pytest.approx(0.0, abs=1e-9)
        assert design[is_vocal > 0.0, 2].std() == pytest.approx(1.0, abs=1e-9)

    def test_duration_enters_on_the_log_scale(self):
        """A raw linear term would be dominated by a few multi-second bouts."""

        cov, is_vocal, duration = self._rows()
        design, _ = build_continuous_design_matrix(cov, is_vocal, duration, ['c0', 'c1'])
        vocal = is_vocal > 0.0
        expected = np.log(duration[vocal])
        expected = (expected - expected.mean()) / expected.std()

        assert np.allclose(design[vocal, 2], expected)

    def test_identical_durations_raise_rather_than_producing_a_dead_column(self):
        """A constant column would make the slope unidentifiable."""

        cov, is_vocal, _ = self._rows()
        duration = np.where(is_vocal > 0.0, 0.4, np.nan)

        with pytest.raises(ValueError, match='not identifiable'):
            build_continuous_design_matrix(cov, is_vocal, duration, ['c0', 'c1'])

    def test_both_planted_terms_are_recovered(self):
        """The step and the slope must be separable, not traded off."""

        rng = np.random.default_rng(3)
        n = 4000
        sessions = np.repeat([f's{i:02d}' for i in range(40)], 100)
        is_vocal = (rng.random(n) < 0.6).astype(float)
        duration = np.where(is_vocal > 0.0, rng.lognormal(-0.9, 0.7, n), np.nan)
        cov = rng.normal(size=(n, 2))
        z = np.zeros(n)
        logs = np.log(duration[is_vocal > 0.0])
        z[is_vocal > 0.0] = (logs - logs.mean()) / logs.std()
        target = np.exp(1.0 + 0.15 * is_vocal + 0.10 * is_vocal * z
                        + 0.4 * cov[:, 0] + rng.normal(0.0, 0.6, n))

        design, labels = build_continuous_design_matrix(cov, is_vocal, duration, ['c0', 'c1'])
        fit = fit_contrast(target, design, labels, sessions, 'lognormal')

        assert fit['terms']['vocal']['coefficient'] == pytest.approx(0.15, abs=0.05)
        assert fit['terms']['vocal_x_log_duration']['coefficient'] == pytest.approx(0.10, abs=0.04)


class TestLognormalLikelihood:
    """The primary estimand: multiplicative, on the geometric mean."""

    def test_it_resists_a_tail_that_defeats_the_gamma_mean(self):
        """This is the whole reason lognormal is primary rather than Gamma."""

        rng = np.random.default_rng(0)
        n = 3000
        sessions = np.repeat([f's{i:02d}' for i in range(30)], 100)
        is_vocal = (rng.random(n) < 0.5).astype(float)
        cov = rng.normal(size=(n, 2))
        band = np.where(is_vocal > 0.0, 0, -1)
        target = np.exp(1.0 + 0.182 * is_vocal + 0.4 * cov[:, 0] + rng.normal(0.0, 0.8, n))
        target[rng.integers(0, n, 30)] *= 50.0        # a few wild anchors

        design, labels = build_design_matrix(cov, is_vocal, band, 1, ['c0', 'c1'])
        lognormal = fit_contrast(target, design, labels, sessions, 'lognormal')
        gamma = fit_contrast(target, design, labels, sessions, 'gamma')

        assert lognormal['terms']['vocal_duration_band_0']['p_value'] < 1e-6
        assert gamma['terms']['vocal_duration_band_0']['p_value'] > 0.05

    def test_non_positive_targets_are_dropped_under_lognormal(self):
        """log(y) is undefined at zero, so those rows cannot be fitted."""

        rng = np.random.default_rng(1)
        n = 600
        sessions = np.repeat([f's{i:02d}' for i in range(20)], 30)
        is_vocal = (rng.random(n) < 0.5).astype(float)
        cov = rng.normal(size=(n, 2))
        band = np.where(is_vocal > 0.0, 0, -1)
        target = np.exp(rng.normal(1.0, 0.4, n))
        target[:12] = 0.0

        design, labels = build_design_matrix(cov, is_vocal, band, 1, ['c0', 'c1'])
        fit = fit_contrast(target, design, labels, sessions, 'lognormal')

        assert fit['n_rows_dropped'] == 12


class TestNormalScores:
    """Rank-based covariate transform."""

    def test_ranking_is_invariant_to_any_monotone_transform(self):
        # This is why the transform can be applied to the stored z-scores without
        # re-extracting: it sees only the ordering.
        generator = np.random.default_rng(0)
        raw = generator.lognormal(0.0, 1.0, 400).reshape(-1, 1)
        assert np.allclose(normal_scores(raw), normal_scores(np.log(raw)))

    def test_output_is_standard_normal_shaped(self):
        generator = np.random.default_rng(1)
        transformed = normal_scores(generator.lognormal(0.0, 1.5, 2000).reshape(-1, 1))
        assert abs(float(stats.skew(transformed[:, 0]))) < 0.1

    def test_non_finite_entries_survive_as_nan(self):
        values = np.array([[1.0], [np.nan], [3.0]])
        transformed = normal_scores(values)
        assert np.isnan(transformed[1, 0])
        assert np.all(np.isfinite(transformed[[0, 2], 0]))


class TestYeoJohnsonCovariates:
    """Per-column power transform fitted by maximum likelihood."""

    @staticmethod
    def _z_scored(values: np.ndarray) -> tuple:
        return (values - values.mean()) / values.std(), {
            'mean': float(values.mean()), 'std': float(values.std())}

    def test_a_lognormal_column_is_straightened(self):
        generator = np.random.default_rng(2)
        native = generator.lognormal(0.5, 0.8, 3000)
        column, scaling = self._z_scored(native)
        transformed, lambdas = yeo_johnson_covariates(
            column.reshape(-1, 1), ['self.speed__mean_0.5s'], {'speed': scaling})
        assert abs(float(stats.skew(native))) > 1.5
        assert abs(float(stats.skew(transformed[:, 0]))) < 0.2
        assert lambdas['self.speed__mean_0.5s'] < 0.5

    def test_a_symmetric_column_is_left_essentially_alone(self):
        # The support rule became unnecessary rather than being replaced: a signed
        # feature returns lambda near the identity without being told it is signed.
        generator = np.random.default_rng(3)
        native = generator.normal(0.0, 10.0, 3000)
        column, scaling = self._z_scored(native)
        _, lambdas = yeo_johnson_covariates(
            column.reshape(-1, 1), ['self.back_pitch__mean_4s'], {'back_pitch': scaling})
        assert 0.8 < lambdas['self.back_pitch__mean_4s'] < 1.2

    def test_a_near_zero_narrow_column_does_not_explode(self):
        # A plain log on this shape drives the skew past -30; the lambda = 0 branch
        # is log(x + 1), which is what keeps it finite.
        generator = np.random.default_rng(4)
        native = np.clip(generator.normal(4.0, 1.0, 3000), 0.01, None)
        column, scaling = self._z_scored(native)
        transformed, _ = yeo_johnson_covariates(
            column.reshape(-1, 1), ['self.neck_elevation__mean_4s'],
            {'neck_elevation': scaling})
        assert abs(float(stats.skew(transformed[:, 0]))) < 0.5

    def test_output_is_standardised(self):
        generator = np.random.default_rng(5)
        native = generator.lognormal(0.0, 1.0, 500)
        column, scaling = self._z_scored(native)
        transformed, _ = yeo_johnson_covariates(
            column.reshape(-1, 1), ['self.speed__mean_4s'], {'speed': scaling})
        assert abs(float(transformed[:, 0].mean())) < 1e-9
        assert abs(float(transformed[:, 0].std()) - 1.0) < 1e-9

    def test_a_missing_scaling_entry_raises_rather_than_guessing(self):
        with pytest.raises(KeyError, match='re-extracted'):
            yeo_johnson_covariates(np.zeros((5, 1)), ['self.speed__mean_4s'], {})


class TestVarianceExplainedByVocalTerms:
    """Decomposition stored beside the coefficients."""

    def test_an_unrelated_vocal_block_adds_nothing(self):
        generator = np.random.default_rng(6)
        covariate = generator.normal(size=800)
        design = np.column_stack([np.ones(800), generator.integers(0, 2, 800).astype(float),
                                  np.zeros(800), covariate])
        target = np.exp(0.7 * covariate + 0.1 * generator.normal(size=800))
        result = variance_explained_by_vocal_terms(
            target=target, full_design=design, n_contrast_terms=3, likelihood='lognormal')
        assert result['vocal_share_percent'] < 1.0
        assert result['r_squared_full'] > 0.9

    def test_a_planted_vocal_effect_takes_a_visible_share(self):
        generator = np.random.default_rng(7)
        treated = generator.integers(0, 2, 800).astype(float)
        design = np.column_stack([np.ones(800), treated, np.zeros(800),
                                  generator.normal(size=800)])
        target = np.exp(2.0 * treated + 0.1 * generator.normal(size=800))
        result = variance_explained_by_vocal_terms(
            target=target, full_design=design, n_contrast_terms=3, likelihood='lognormal')
        assert result['vocal_share_percent'] > 50.0


class TestForwardCleanTimes:
    """All-animal silence test used by the matched-divergence anchors."""

    def test_a_call_starting_inside_the_window_disqualifies(self):
        mask = forward_clean_times(np.array([10.0]), np.array([12.0]), np.array([12.1]), 4.0)
        assert not bool(mask[0])

    def test_a_call_starting_after_the_window_does_not(self):
        mask = forward_clean_times(np.array([10.0]), np.array([15.0]), np.array([15.1]), 4.0)
        assert bool(mask[0])

    def test_a_call_straddling_the_moment_disqualifies(self):
        # A starts-only test would pass this: the call began before the anchor and
        # is still running at it.
        mask = forward_clean_times(np.array([10.0]), np.array([9.0]), np.array([11.0]), 4.0)
        assert not bool(mask[0])

    def test_a_session_without_calls_is_entirely_clean(self):
        mask = forward_clean_times(np.array([1.0, 2.0]), np.empty(0), np.empty(0), 4.0)
        assert mask.all()


class TestMatchedDivergencePipeline:
    """Settings wiring for the design-based counterpart to the contrast."""

    @staticmethod
    def _settings() -> dict:
        return _shipped_settings()

    def test_windows_convert_to_frames_at_the_configured_rate(self):
        settings = self._settings()
        pipeline = MatchedDivergencePipeline(modeling_settings_dict=settings)
        rate = settings['io']['camera_sampling_rate']
        block = settings['behavioral_response']['matched_divergence']
        for name in ('pre', 'post_clean', 'window', 'pre_window',
                     'baseline_start', 'baseline_end', 'silence_grid'):
            expected = int(np.floor(rate * block[f'{name}_seconds']))
            assert pipeline.divergence_frames[name] == expected

    def test_a_matching_window_that_does_not_precede_the_anchor_raises(self):
        # The window must END before the anchor: with it running up to the anchor
        # the two arms still differed at t=0 in the direction of the effect,
        # because the divergence is already under way inside it.
        settings = self._settings()
        settings['behavioral_response']['matched_divergence']['baseline_start_seconds'] = 0.5
        settings['behavioral_response']['matched_divergence']['baseline_end_seconds'] = 4.0
        with pytest.raises(ValueError, match='must exceed'):
            MatchedDivergencePipeline(modeling_settings_dict=settings)

    def test_it_inherits_the_response_features_and_likelihood_rule(self):
        pipeline = MatchedDivergencePipeline(modeling_settings_dict=self._settings())
        assert pipeline._response_likelihood('speed') == 'lognormal'
        assert pipeline._response_likelihood('back_pitch') == 'gaussian'


class TestSelectCovariateColumns:
    """The fit-time choice of which animal's covariates adjust the contrast."""

    labels = ['self.speed__mean_0.5s', 'self.speed__mean_4s', 'other.speed__mean_0.5s',
              'other.neck_elevation__mean_4s', 'nose-nose__mean_0.5s', 'nose-allo_yaw__mean_4s']

    def _covariates(self) -> np.ndarray:
        return np.arange(3 * len(self.labels), dtype=float).reshape(3, len(self.labels))

    def test_both_keeps_every_column_in_order(self):
        selected, kept = select_covariate_columns(self._covariates(), self.labels, 'both')
        assert kept == self.labels
        np.testing.assert_array_equal(selected, self._covariates())

    def test_self_drops_the_partner_and_keeps_dyadic(self):
        selected, kept = select_covariate_columns(self._covariates(), self.labels, 'self')
        assert kept == ['self.speed__mean_0.5s', 'self.speed__mean_4s',
                        'nose-nose__mean_0.5s', 'nose-allo_yaw__mean_4s']
        np.testing.assert_array_equal(selected, self._covariates()[:, [0, 1, 4, 5]])

    def test_partner_drops_self_and_keeps_dyadic(self):
        selected, kept = select_covariate_columns(self._covariates(), self.labels, 'partner')
        assert kept == ['other.speed__mean_0.5s', 'other.neck_elevation__mean_4s',
                        'nose-nose__mean_0.5s', 'nose-allo_yaw__mean_4s']
        np.testing.assert_array_equal(selected, self._covariates()[:, [2, 3, 4, 5]])

    def test_unknown_choice_is_rejected_by_name(self):
        with pytest.raises(ValueError, match='covariate_features'):
            select_covariate_columns(self._covariates(), self.labels, 'female')

    def test_shipped_settings_carry_a_valid_choice(self):
        settings_path = (Path(__file__).resolve().parents[2] / 'src' / 'usv_playpen'
                         / '_parameter_settings' / 'modeling_settings.json')
        with settings_path.open('r') as handle:
            choice = json.load(handle)['behavioral_response']['covariate_features']
        assert choice in ('self', 'partner', 'both')


# Geometry of the synthetic end-to-end cohort. The pipeline windows are shrunk
# from the shipped 4 s / 0.5 s so a 240 s session at 30 fps holds dozens of
# anchors of every kind: 1 s of history is 30 frames, the 0.5 s forward window
# is 15 frames, and 0.1 s bins cut it into 5 bins of exactly 3 frames each.
SYNTH_FPS = 30.0
SYNTH_FRAMES = int(SYNTH_FPS * 240)
SYNTH_SESSIONS = 5
SYNTH_HISTORY_SECONDS = 1.0
SYNTH_HISTORY_FRAMES = 30
SYNTH_WINDOW_FRAMES = 15
SYNTH_N_BINS = 5
SYNTH_SUMMARY_SECONDS = [0.2, 1.0]
SYNTH_RESPONSE_FEATURES = ['speed', 'neck_elevation']
# One dyadic feature, so the role-neutral covariate mapping also meets a column
# that belongs to neither animal and must keep its bare name.
SYNTH_DYADIC_FEATURES = ['nose-nose']
# The `_synth` feature tables are unit sinusoids around zero, so half of every
# trace would sit below the [0, ...] bound of `speed` and `neck_elevation` and be
# dropped as out of range. Shifting every column up keeps each trace inside its
# bounds, so every target is finite and strictly positive.
SYNTH_FEATURE_OFFSET = 3.0
# Matched-divergence windows: 1 s either side of the anchor gives a 60-frame
# stored curve whose anchor sits at index 30; the baseline is [-1.0, -0.2) s.
SYNTH_DIVERGENCE_BLOCK = {
    'pre_seconds': 1.0,
    'post_clean_seconds': 1.0,
    'window_seconds': 1.0,
    'pre_window_seconds': 1.0,
    'baseline_start_seconds': 1.0,
    'baseline_end_seconds': 0.2,
    'silence_grid_seconds': 0.5,
    'match_caliper_sd': 0.5,
}


def _write_variable_bout_summary(session_root: Path,
                                 male_name: str,
                                 female_name: str,
                                 seed: int,
                                 silent_male: bool = False) -> Path:
    """
    Overwrites a session's USV summary with bouts of VARYING length.

    The shared ``_synth.build_usv_summary_csv`` packs the same number of
    syllables into every bout, so every bout has the same duration; the
    duration terciles and the log-duration slope of the contrast then cannot be
    built at all (both raise on identical durations). This writes the male's
    bouts with 2..6 syllables in rotation (durations 0.04 s to 0.16 s at a
    0.03 s syllable pitch, well inside the ~0.116 s inter-bout threshold the
    shipped male mixture model yields), separated by seeded 6-16 s gaps so
    every inter-bout gap can hold a 1 s history plus a 0.5 s forward window,
    and the longest gaps clear the 3 s deep-silence margin. Two sparse female
    calls are added so the all-animal silence rule of the matched-divergence
    pipeline has partner calls to respect.

    Parameters
    ----------
    session_root : pathlib.Path
        Session directory whose ``audio/*_usv_summary.csv`` is replaced.
    male_name, female_name : str
        Emitter names, matching the session's track file.
    seed : int
        Seed for the gap lengths, so sessions differ but are reproducible.
    silent_male : bool
        When True the male emits nothing, so the session has no predictor bout.

    Returns
    -------
    summary_path : pathlib.Path
        The rewritten summary file.
    """

    rng = np.random.default_rng(seed)
    emitters: list[str] = []
    starts: list[float] = []
    stops: list[float] = []

    bout_start, bout_index = 5.0, 0
    while not silent_male and bout_start < 225.0:
        for syllable in range(2 + bout_index % 5):
            onset = bout_start + syllable * 0.03
            emitters.append(male_name)
            starts.append(round(onset, 6))
            stops.append(round(onset + 0.01, 6))
        bout_index += 1
        bout_start += 6.0 + 10.0 * float(rng.random())

    for onset in (50.3, 130.7):
        emitters.append(female_name)
        starts.append(onset)
        stops.append(onset + 0.01)

    order = np.argsort(starts)
    n_rows = len(starts)
    summary_path = next((session_root / 'audio').glob('*_usv_summary.csv'))
    pls.DataFrame({
        'emitter': [emitters[index] for index in order],
        'start': [starts[index] for index in order],
        'stop': [stops[index] for index in order],
        'noise': [False] * n_rows,
        'squeak': [False] * n_rows,
        'qlvm_supercategory': [1] * n_rows,
        'qlvm_category': [1] * n_rows,
        'mask_number': [2] * n_rows,
        'qlvm1': [0.0] * n_rows,
        'qlvm2': [0.0] * n_rows,
    }).write_csv(file=summary_path)
    return summary_path


def _shift_feature_table(session_root: Path,
                         offset: float,
                         out_of_bounds_columns: list[str]) -> None:
    """
    Shifts every behavioral-feature column into its bounds, or out of them.

    Parameters
    ----------
    session_root : pathlib.Path
        Session directory whose behavioral-feature CSV is rewritten in place.
    offset : float
        Constant added to every column, lifting the ``_synth`` sinusoids above
        the zero lower bound of the non-negative features.
    out_of_bounds_columns : list of str
        Columns overwritten with a constant ``-5.0``, below the zero lower bound,
        so no sample of them is ever in range.

    Returns
    -------
    None
    """

    csv_path = next((session_root / 'video').glob('*_behavioral_features.csv'))
    table = pls.read_csv(source=csv_path)
    table = table.with_columns([(pls.col(column) + offset).alias(column) for column in table.columns])
    if out_of_bounds_columns:
        table = table.with_columns([pls.lit(-5.0).alias(column) for column in out_of_bounds_columns])
    table.write_csv(file=csv_path)


def _build_response_cohort(base_dir: Path,
                           n_sessions: int = SYNTH_SESSIONS,
                           silent_male_sessions: tuple[int, ...] = (),
                           out_of_bounds_sessions: tuple[int, ...] = ()) -> tuple[dict, Path]:
    """
    Builds a synthetic cohort on disk plus the settings that point at it.

    The session tree comes from ``_synth.build_session_tree`` (feature CSV,
    track H5, USV summary per session); each session's feature table is then
    shifted into bounds and its USV summary replaced with variable-length male
    bouts. The settings are ``_synth.build_modeling_settings`` with the
    ``behavioral_response`` block and its ``matched_divergence`` sub-block
    shrunk to the synthetic geometry.

    Parameters
    ----------
    base_dir : pathlib.Path
        Scratch directory everything is written under.
    n_sessions : int
        Number of sessions to build.
    silent_male_sessions : tuple of int
        Session indices whose male never calls (no predictor bout).
    out_of_bounds_sessions : tuple of int
        Session indices whose female ``speed`` is never inside its bounds, so
        no anchor in them has a usable ``speed`` target.

    Returns
    -------
    settings, save_dir : tuple
        The modeling settings, and the directory the pipelines publish into.
    """

    session_roots = build_session_tree(
        base_dir=base_dir / 'sessions',
        n_sessions=n_sessions,
        n_frames=SYNTH_FRAMES,
        camera_fps=SYNTH_FPS,
        filter_history=SYNTH_HISTORY_SECONDS,
        dyadic_features=list(SYNTH_DYADIC_FEATURES),
        n_bouts=4,
    )
    for index, session_root in enumerate(session_roots):
        male_name, female_name = f"s{index}_m_male", f"s{index}_m_female"
        _shift_feature_table(
            session_root=session_root,
            offset=SYNTH_FEATURE_OFFSET,
            out_of_bounds_columns=([f"{female_name}.speed"]
                                   if index in out_of_bounds_sessions else []),
        )
        _write_variable_bout_summary(
            session_root=session_root,
            male_name=male_name,
            female_name=female_name,
            seed=index,
            silent_male=index in silent_male_sessions,
        )

    list_file = write_session_list_file(session_roots, base_dir / 'session_list.txt')
    save_dir = base_dir / 'out'
    settings = build_modeling_settings(
        session_list_file=list_file,
        save_directory=save_dir,
        camera_sampling_rate=SYNTH_FPS,
        filter_history=SYNTH_HISTORY_SECONDS,
        dyadic_features=list(SYNTH_DYADIC_FEATURES),
    )
    response_block = settings['behavioral_response']
    response_block['response_mouse_index'] = 1
    response_block['response_features'] = list(SYNTH_RESPONSE_FEATURES)
    response_block['history_seconds'] = SYNTH_HISTORY_SECONDS
    response_block['target_window_seconds'] = 0.5
    response_block['target_bin_seconds'] = 0.1
    response_block['post_bout_silence_seconds'] = 0.5
    response_block['deep_silence_margin_seconds'] = 3.0
    response_block['covariate_summary_seconds'] = list(SYNTH_SUMMARY_SECONDS)
    response_block['duration_n_bins'] = 3
    response_block['matched_divergence'].update(SYNTH_DIVERGENCE_BLOCK)
    return settings, save_dir


def _write_contrast_settings(settings: dict, path: Path, **overrides) -> Path:
    """
    Writes the settings JSON ``behavioral_response_contrast`` reads at fit time.

    The contrast takes ``covariate_transform`` and ``covariate_features`` from a
    settings FILE rather than from the artifact, so each test writes its own.

    Parameters
    ----------
    settings : dict
        Base modeling settings; deep-copied, never mutated.
    path : pathlib.Path
        Destination JSON file.
    **overrides
        Keys replaced inside ``behavioral_response``.

    Returns
    -------
    path : pathlib.Path
        The file written.
    """

    written = copy.deepcopy(settings)
    written['behavioral_response'].update(overrides)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as handle:
        json.dump(written, handle)
    return path


def _load_single_pickle(directory: Path, pattern: str) -> tuple[Path, dict]:
    """
    Loads the one pickle matching a glob, asserting there is exactly one.

    Parameters
    ----------
    directory : pathlib.Path
        Directory to search.
    pattern : str
        Glob the pickle must match.

    Returns
    -------
    path, payload : tuple
        The pickle's path and its unpickled content.
    """

    matches = sorted(directory.glob(pattern))
    assert len(matches) == 1, f"expected exactly one '{pattern}' in {directory}, got {matches}"
    with matches[0].open('rb') as handle:
        return matches[0], pickle.load(handle)


def _rewrite_artifact(artifact: dict, path: Path) -> Path:
    """
    Pickles a (modified) artifact to a new path.

    Parameters
    ----------
    artifact : dict
        Artifact to write.
    path : pathlib.Path
        Destination file.

    Returns
    -------
    path : pathlib.Path
        The file written.
    """

    with path.open('wb') as handle:
        pickle.dump(artifact, handle)
    return path


@pytest.fixture(scope='module')
def extracted_cohort(tmp_path_factory) -> dict:
    """
    Runs the real behavioral-response extraction once for the whole module.

    Five ordinary sessions plus a sixth whose male never calls, so the shared
    artifact also carries the evidence that a predictor-silent session is
    dropped. Extraction is the slow step, so every contrast test reuses this
    pickle rather than re-reading the cohort.

    Parameters
    ----------
    tmp_path_factory : pytest.TempPathFactory
        Source of the module-scoped scratch directory.

    Returns
    -------
    cohort : dict
        ``settings``, ``pickle_path`` and the loaded ``artifact``.
    """

    base_dir = tmp_path_factory.mktemp('behavioral_response')
    settings, save_dir = _build_response_cohort(
        base_dir=base_dir, n_sessions=SYNTH_SESSIONS + 1,
        silent_male_sessions=(SYNTH_SESSIONS,))
    BehavioralResponsePipeline(modeling_settings_dict=copy.deepcopy(settings)
                               ).extract_and_save_modeling_input_data()
    pickle_path, artifact = _load_single_pickle(
        save_dir, 'modeling_behavioral_response_m1_allfeatures_*.pkl')
    return {'settings': settings, 'pickle_path': pickle_path, 'artifact': artifact}


class TestBehavioralResponseExtraction:
    """The real extraction, end-to-end on a synthetic on-disk cohort."""

    def test_the_artifact_holds_every_documented_key(self, extracted_cohort):
        """
        The contrast and every figure read the artifact by key, so a missing
        key would surface only downstream. Checks the full documented key set,
        the reserved ``_input_metadata`` block, and that the stored condition
        levels are the three codes the contrast selects rows with.
        """

        artifact = extracted_cohort['artifact']

        assert set(artifact) == {
            'covariates', 'covariate_labels', 'covariate_scaling', 'covariate_non_negative',
            'target', 'target_bins', 'response_features', 'response_likelihoods',
            'is_vocal', 'condition', 'condition_levels', 'bout_duration', 'session_ids',
            '_input_metadata'}
        assert artifact['condition_levels'] == {
            'after_bout': 1.0, 'inter_bout_silence': 0.0, 'deep_silence': 2.0}
        assert artifact['response_features'] == SYNTH_RESPONSE_FEATURES
        assert artifact['response_likelihoods'] == {'speed': 'lognormal',
                                                     'neck_elevation': 'lognormal'}

    def test_every_array_carries_the_same_rows(self, extracted_cohort):
        """
        Every array is positionally paired into one design, so a length or
        width mismatch would silently misalign covariates and targets. Checks
        the row count of every per-row array, the feature axis of both targets,
        and that the bin axis has one entry per configured response bin.
        """

        artifact = extracted_cohort['artifact']
        n_rows = artifact['target'].shape[0]

        assert n_rows > 0
        assert artifact['target'].shape == (n_rows, len(SYNTH_RESPONSE_FEATURES))
        assert artifact['target_bins'].shape == (n_rows, len(SYNTH_RESPONSE_FEATURES), SYNTH_N_BINS)
        assert artifact['covariates'].shape == (n_rows, len(artifact['covariate_labels']))
        for key in ('is_vocal', 'condition', 'bout_duration', 'session_ids'):
            assert artifact[key].shape == (n_rows,)

    def test_covariate_labels_are_role_neutral_summaries(self, extracted_cohort):
        """
        Raw column names carry per-session mouse ids, so without the ``self.`` /
        ``other.`` mapping no two sessions would share a covariate. Checks that
        every label is one of the two roles crossed with the two synthetic
        egocentric features and the two summary widths, that the dyadic
        feature keeps its bare, animal-free name, and that the pooled scaling
        and support flags cover every base feature the labels name.
        """

        artifact = extracted_cohort['artifact']
        expected = sorted(
            [f"{role}.{feature}__mean_{seconds:g}s"
             for role in ('self', 'other')
             for feature in ('speed', 'neck_elevation')
             for seconds in SYNTH_SUMMARY_SECONDS]
            + [f"{feature}__mean_{seconds:g}s"
               for feature in SYNTH_DYADIC_FEATURES
               for seconds in SYNTH_SUMMARY_SECONDS])

        assert sorted(artifact['covariate_labels']) == expected
        assert set(artifact['covariate_scaling']) == {'speed', 'neck_elevation', 'nose-nose'}
        # Every base has a zero lower bound, so every one is loggable at fit time.
        assert artifact['covariate_non_negative'] == {
            'speed': True, 'neck_elevation': True, 'nose-nose': True}

    def test_condition_codes_agree_with_the_vocal_flag_and_durations(self, extracted_cohort):
        """
        ``is_vocal`` and ``condition`` encode the same split twice, and only
        bout rows have a duration; any disagreement would move rows between
        arms. Checks all three levels are populated, ``is_vocal`` is exactly the
        after-bout rows, durations are finite and positive on those rows and NaN
        on both controls, and at least three distinct durations exist so the
        terciles can be built.
        """

        artifact = extracted_cohort['artifact']
        condition = artifact['condition']
        vocal = condition == 1.0

        assert set(np.unique(condition).tolist()) == {0.0, 1.0, 2.0}
        np.testing.assert_array_equal(artifact['is_vocal'], vocal.astype(float))
        assert np.all(np.isfinite(artifact['bout_duration'][vocal]))
        assert np.all(artifact['bout_duration'][vocal] > 0.0)
        assert np.all(np.isnan(artifact['bout_duration'][~vocal]))
        assert np.unique(artifact['bout_duration'][vocal]).size >= 3

    def test_targets_are_finite_positive_and_bins_tile_the_window(self, extracted_cohort):
        """
        With every sample inside its bounds the targets must be finite and
        strictly positive (the lognormal likelihood logs them), and because the
        five bins are equal 3-frame slices of the 15-frame window, the mean of
        the bin means must reproduce the window mean exactly -- the check that
        the time course and the headline number describe the same samples.
        """

        artifact = extracted_cohort['artifact']

        assert np.all(np.isfinite(artifact['target']))
        assert np.all(artifact['target'] > 0.0)
        np.testing.assert_allclose(artifact['target_bins'].mean(axis=2), artifact['target'])

    def test_a_session_whose_predictor_never_calls_is_dropped(self, extracted_cohort):
        """
        The sixth session's male is silent. Sessions are dropped on the
        PREDICTOR having no bouts, so it must be absent from the rows and from
        the provenance, while all five calling sessions contribute.
        """

        artifact = extracted_cohort['artifact']
        contributing = sorted(np.unique(artifact['session_ids']).tolist())

        assert contributing == [f'session_{index}' for index in range(SYNTH_SESSIONS)]
        assert f'session_{SYNTH_SESSIONS}' not in artifact['_input_metadata']['session_ids']

    def test_provenance_counts_agree_with_the_rows(self, extracted_cohort):
        """
        ``_save_extracted_data`` backfills the metadata counts once anchors
        exist; stale placeholders would misreport the sample. Checks the row
        totals per condition, the per-session anchor counts, the derived
        predictor index, and the frame geometry recorded for the fit.
        """

        artifact = extracted_cohort['artifact']
        metadata = artifact['_input_metadata']
        condition = artifact['condition']
        per_session = metadata['n_events_per_session']

        assert metadata['n_rows'] == condition.size
        assert metadata['n_vocal_rows'] == int(np.sum(condition == 1.0))
        assert metadata['n_inter_bout_rows'] == int(np.sum(condition == 0.0))
        assert metadata['n_deep_silence_rows'] == int(np.sum(condition == 2.0))
        assert metadata['n_quiet_rows'] == condition.size - metadata['n_vocal_rows']
        assert metadata['n_sessions_used'] == SYNTH_SESSIONS
        assert sum(sum(counts.values()) for counts in per_session.values()) == condition.size
        for session, counts in per_session.items():
            in_session = artifact['session_ids'] == session
            assert counts['vocal'] == int(np.sum(condition[in_session] == 1.0))

        analysis = metadata['analysis_specific']
        assert analysis['derived_predictor_mouse_index'] == 0
        assert analysis['target_window_frames'] == SYNTH_WINDOW_FRAMES
        assert analysis['n_response_bins'] == SYNTH_N_BINS
        assert analysis['covariate_labels'] == artifact['covariate_labels']
        assert analysis['response_folds'] == {'speed': 'none', 'neck_elevation': 'none'}

    def test_an_unknown_response_feature_raises_before_loading(self, tmp_path):
        """
        A response feature outside ``kinematic_features.egocentric`` would only
        fail once extraction reached the first session; it must be refused up
        front, by name.
        """

        settings, _ = _build_response_cohort(base_dir=tmp_path, n_sessions=1)
        settings['behavioral_response']['response_features'] = ['speed', 'not_a_feature']

        with pytest.raises(ValueError, match='not_a_feature'):
            BehavioralResponsePipeline(modeling_settings_dict=settings
                                       ).extract_and_save_modeling_input_data()

    def test_a_cohort_whose_predictor_never_calls_raises(self, tmp_path):
        """
        When every session's male is silent, every session is dropped before
        column selection; the extraction must stop with an explanation of what
        to check rather than fail obscurely on an empty design.
        """

        settings, save_dir = _build_response_cohort(
            base_dir=tmp_path, n_sessions=2, silent_male_sessions=(0, 1))

        with pytest.raises(RuntimeError, match='No session survived'):
            BehavioralResponsePipeline(modeling_settings_dict=settings
                                       ).extract_and_save_modeling_input_data()
        assert not list(save_dir.glob('modeling_behavioral_response_*.pkl'))

    def test_a_session_with_no_in_bounds_response_is_dropped(self, tmp_path):
        """
        An anchor with no usable target for ANY feature carries no information.
        Session 0's female speed is entirely out of bounds and ``speed`` is the
        only response, so every one of its anchors is unusable and the session
        must vanish from the rows while the others still contribute.
        """

        settings, save_dir = _build_response_cohort(
            base_dir=tmp_path, n_sessions=3, out_of_bounds_sessions=(0,))
        settings['behavioral_response']['response_features'] = ['speed']
        BehavioralResponsePipeline(modeling_settings_dict=settings
                                   ).extract_and_save_modeling_input_data()
        _, artifact = _load_single_pickle(save_dir, 'modeling_behavioral_response_*.pkl')

        assert sorted(np.unique(artifact['session_ids']).tolist()) == ['session_1', 'session_2']
        assert artifact['target'].shape[1] == 1

    def test_a_cohort_with_no_usable_anchor_raises(self, tmp_path):
        """
        When no session keeps a single anchor there is nothing to write; the
        extraction must raise rather than publish an empty artifact that would
        fail obscurely at fit time.
        """

        settings, save_dir = _build_response_cohort(
            base_dir=tmp_path, n_sessions=2, out_of_bounds_sessions=(0, 1))
        settings['behavioral_response']['response_features'] = ['speed']

        with pytest.raises(RuntimeError, match='No anchor survived'):
            BehavioralResponsePipeline(modeling_settings_dict=settings
                                       ).extract_and_save_modeling_input_data()
        assert not list(save_dir.glob('modeling_behavioral_response_*.pkl'))


class TestSaveExtractedDataValidation:
    """A misaligned design must be refused, never written."""

    @staticmethod
    def _arrays(n_rows: int = 6) -> dict:
        """
        Builds a consistent set of keyword arguments for ``_save_extracted_data``.

        Parameters
        ----------
        n_rows : int
            Rows in every per-row array.

        Returns
        -------
        kwargs : dict
            Arguments that pass validation as they stand.
        """

        return {
            'covariates': np.zeros((n_rows, 2)),
            'covariate_labels': ['self.speed__mean_0.2s', 'self.speed__mean_1s'],
            'covariate_scaling': {'speed': {'mean': 0.0, 'std': 1.0}},
            'covariate_non_negative': {'speed': True},
            'target': np.ones((n_rows, 1)),
            'target_bins': np.ones((n_rows, 1, 2)),
            'response_features': ['speed'],
            'response_likelihoods': {'speed': 'lognormal'},
            'is_vocal': np.zeros(n_rows),
            'condition': np.zeros(n_rows),
            'bout_duration': np.full(n_rows, np.nan),
            'session_ids': np.array(['s0'] * n_rows, dtype=object),
            'input_metadata': {'analysis_specific': {}},
            'anchors_per_session': {'s0': {'vocal': 0, 'inter_bout': n_rows, 'deep_silence': 0}},
            'fname': 'never_written.pkl',
        }

    @pytest.mark.parametrize(
        ('override', 'message'),
        [
            ({'response_features': ['speed', 'neck_elevation']}, 'feature columns'),
            ({'condition': np.zeros(5)}, "Row-count mismatch.*`condition`"),
            ({'session_ids': np.array(['s0'] * 7, dtype=object)}, "Row-count mismatch.*`session_ids`"),
            ({'covariate_labels': ['self.speed__mean_0.2s']}, 'labels were built'),
        ],
        ids=['target_width', 'condition_rows', 'session_rows', 'covariate_labels'],
    )
    def test_each_misalignment_raises_and_writes_nothing(self, tmp_path, override, message):
        """
        Every array is positionally paired into one design, so each kind of
        mismatch -- the target's feature width, a per-row array's length, the
        covariate label count -- must raise a ``ValueError`` naming the problem
        before anything reaches ``io.save_directory``.
        """

        settings = _shipped_settings()
        settings['io']['save_directory'] = str(tmp_path)
        pipeline = BehavioralResponsePipeline(modeling_settings_dict=settings)
        kwargs = self._arrays()
        kwargs.update(override)

        with pytest.raises(ValueError, match=message):
            pipeline._save_extracted_data(**kwargs)
        assert not (tmp_path / 'never_written.pkl').exists()


class TestResponseColumnAndTargets:
    """The responder's column is found by slot, and its target is folded."""

    @staticmethod
    def _pipeline(**overrides) -> BehavioralResponsePipeline:
        """
        Builds a pipeline on the shipped settings.

        Parameters
        ----------
        **overrides
            Keys replaced inside ``behavioral_response``.

        Returns
        -------
        pipeline : BehavioralResponsePipeline
            Pipeline instance.
        """

        settings = _shipped_settings()
        settings['behavioral_response'].update(overrides)
        return BehavioralResponsePipeline(modeling_settings_dict=settings)

    def test_a_non_binary_mouse_index_is_rejected(self):
        """
        The response index is an absolute slot, 0 or 1; anything else cannot
        name a mouse and would derive a nonsensical predictor index.
        """

        with pytest.raises(ValueError, match='must be 0 \\(male\\) or 1'):
            self._pipeline(response_mouse_index=2)

    def test_the_column_is_named_by_absolute_slot(self):
        """
        Slot 1 must resolve to the second track name whatever the column
        order, since role keys can only be read against another setting.
        """

        column = self._pipeline()._resolve_response_column(
            session_df_columns=['m_a.speed', 'm_b.speed'], mouse_names=['m_a', 'm_b'],
            response_feature='speed')

        assert column == 'm_b.speed'

    def test_a_slot_beyond_the_session_mice_raises(self):
        """
        A single-mouse session has no slot 1; indexing it would raise a bare
        ``IndexError`` without saying which setting is at fault.
        """

        with pytest.raises(ValueError, match='outside the 1 mouse slots'):
            self._pipeline()._resolve_response_column(
                session_df_columns=['m_a.speed'], mouse_names=['m_a'], response_feature='speed')

    def test_a_missing_column_names_the_csv_mice(self):
        """
        When the track H5 and the feature CSV disagree on mouse identity, the
        error must list the CSV's mouse labels (dyadic columns excluded) so the
        stale labels can be corrected on disk.
        """

        with pytest.raises(KeyError, match=r"\['m_x', 'm_y'\]"):
            self._pipeline()._resolve_response_column(
                session_df_columns=['m_x.speed', 'm_y.speed', 'm_x-m_y.nose-nose'],
                mouse_names=['m_a', 'm_b'], response_feature='speed')

    def test_a_smooth_abs_response_is_folded_before_averaging(self):
        """
        ``ego_yaw`` is a ``smooth_abs_features`` entry (epsilon 1.0 shipped), so
        a constant -3 trace must average to ``sqrt(9 + 1)`` in the window and in
        every bin, never to a signed residual.
        """

        pipeline = self._pipeline()
        window, bins = pipeline._response_target_values(
            raw_values=np.full(400, -3.0), anchor_frames=np.array([100, 200]),
            response_feature='ego_yaw')

        np.testing.assert_allclose(window, np.sqrt(10.0))
        assert bins.shape == (2, pipeline.n_response_bins)
        np.testing.assert_allclose(bins, np.sqrt(10.0))

    def test_an_abs_response_is_folded_before_averaging(self):
        """
        ``allo_roll`` is an ``abs_features`` entry, so a trace alternating
        between -2 and +2 must average to 2 rather than cancelling to 0.
        """

        values = np.tile([-2.0, 2.0], 200)
        window, bins = self._pipeline()._response_target_values(
            raw_values=values, anchor_frames=np.array([100]), response_feature='allo_roll')

        np.testing.assert_allclose(window, 2.0)
        np.testing.assert_allclose(bins, 2.0)

    def test_an_unfolded_response_keeps_its_sign(self):
        """
        ``allo_pitch`` receives no fold, so a signed constant stays signed --
        the reason such a feature is fitted with a Gaussian likelihood.
        """

        window, _ = self._pipeline()._response_target_values(
            raw_values=np.full(400, -3.0), anchor_frames=np.array([100]),
            response_feature='allo_pitch')

        np.testing.assert_allclose(window, -3.0)


class TestBehavioralResponseContrast:
    """The contrast, run on the pickle the real extraction wrote."""

    @pytest.mark.parametrize('transform', ['yeo_johnson', 'derived', 'normal_scores', 'linear'])
    def test_every_covariate_transform_yields_a_complete_result(
            self, extracted_cohort, tmp_path, transform):
        """
        Each fit-time covariate transform must run on a real artifact and
        return the full result structure: the primary continuous-design fit
        with both vocal terms, the Gamma mean-scale refit (the features are
        lognormal), the descriptive variance split, the tercile fit, one
        time-course fit per bin, and a per-session log ratio for each of the
        five sessions. The result pickle must be published too.
        """

        settings_path = _write_contrast_settings(
            extracted_cohort['settings'], tmp_path / 'settings.json',
            covariate_transform=transform, covariate_features='both')
        results = behavioral_response_contrast(
            input_pickle_path=extracted_cohort['pickle_path'],
            output_directory=tmp_path / 'contrast', settings_path=settings_path)

        assert results['continuous_labels'][:3] == ['intercept', 'vocal', 'vocal_x_log_duration']
        assert results['term_labels'][1:4] == ['vocal_duration_band_0', 'vocal_duration_band_1',
                                               'vocal_duration_band_2']
        assert results['duration_edges'].size == 4
        assert results['response_features'] == SYNTH_RESPONSE_FEATURES
        for feature in SYNTH_RESPONSE_FEATURES:
            block = results['per_feature'][feature]
            assert block['likelihood'] == 'lognormal'
            assert {'vocal', 'vocal_x_log_duration'} <= set(block['window']['terms'])
            assert block['window']['n_sessions'] == SYNTH_SESSIONS
            assert block['mean_scale']['likelihood'] == 'gamma'
            assert 0.0 <= block['variance_explained']['r_squared_full'] <= 1.0
            assert [fit['bin_index'] for fit in block['time_course']] == list(range(SYNTH_N_BINS))
            assert block['per_session_log_ratio'].shape == (SYNTH_SESSIONS,)
        _load_single_pickle(tmp_path / 'contrast', 'behavioral_response_contrast_inter_bout_silence_*.pkl')

    @pytest.mark.parametrize(('control', 'level'), [('inter_bout_silence', 0.0), ('deep_silence', 2.0)])
    def test_only_the_bout_rows_and_the_chosen_control_are_fitted(
            self, extracted_cohort, tmp_path, control, level):
        """
        The other control's rows would otherwise enter as untreated and blend
        the two comparisons. The fitted row count must equal the after-bout
        rows plus the chosen control's rows, for either control.
        """

        artifact = extracted_cohort['artifact']
        settings_path = _write_contrast_settings(
            extracted_cohort['settings'], tmp_path / 'settings.json', covariate_transform='linear')
        results = behavioral_response_contrast(
            input_pickle_path=extracted_cohort['pickle_path'],
            output_directory=tmp_path / 'contrast', settings_path=settings_path, control=control)
        expected = int(np.sum(artifact['condition'] == 1.0) + np.sum(artifact['condition'] == level))

        assert results['control'] == control
        assert results['n_rows'] == expected
        assert results['per_feature']['speed']['window']['n_rows_fitted'] == expected

    def test_the_self_covariate_set_drops_the_partner_columns(self, extracted_cohort, tmp_path):
        """
        Which animal's kinematics adjust the contrast is a fit-time choice read
        from settings; under ``'self'`` no ``other.`` column may reach the design,
        while the dyadic columns, which belong to neither animal, stay in.
        """

        settings_path = _write_contrast_settings(
            extracted_cohort['settings'], tmp_path / 'settings.json',
            covariate_transform='linear', covariate_features='self')
        results = behavioral_response_contrast(
            input_pickle_path=extracted_cohort['pickle_path'],
            output_directory=tmp_path / 'contrast', settings_path=settings_path)

        assert results['covariate_features'] == 'self'
        assert results['covariate_labels']
        assert not any(label.startswith('other.') for label in results['covariate_labels'])
        assert {'nose-nose__mean_0.2s', 'nose-nose__mean_1s'} <= set(results['covariate_labels'])
        assert any(label.startswith('self.') for label in results['covariate_labels'])
        assert results['continuous_labels'][3:] == results['covariate_labels']

    def test_a_gaussian_feature_skips_the_mean_scale_refit(self, extracted_cohort, tmp_path):
        """
        A signed feature has no Gamma mean to report, so ``mean_scale`` must be
        ``None`` and the variance split must run on the raw response. The
        artifact's likelihood for ``neck_elevation`` is switched to Gaussian to
        reach that branch on real extracted rows.
        """

        artifact = copy.deepcopy(extracted_cohort['artifact'])
        artifact['response_likelihoods']['neck_elevation'] = 'gaussian'
        pickle_path = _rewrite_artifact(artifact, tmp_path / 'gaussian.pkl')
        settings_path = _write_contrast_settings(
            extracted_cohort['settings'], tmp_path / 'settings.json', covariate_transform='linear')
        results = behavioral_response_contrast(
            input_pickle_path=pickle_path, output_directory=tmp_path / 'contrast',
            settings_path=settings_path)

        block = results['per_feature']['neck_elevation']
        assert block['likelihood'] == 'gaussian'
        assert block['mean_scale'] is None
        assert block['window']['likelihood'] == 'gaussian'
        assert 0.0 <= block['variance_explained']['r_squared_full'] <= 1.0

    def test_a_legacy_two_condition_artifact_is_still_read(self, extracted_cohort, tmp_path):
        """
        Artifacts written before the deep-silence control carry only
        ``is_vocal``. They must still fit against inter-bout silence (every
        non-vocal row then counts as that control), and asking them for deep
        silence must be refused by name.
        """

        artifact = copy.deepcopy(extracted_cohort['artifact'])
        keep = artifact['condition'] != 2.0
        for key in ('covariates', 'target', 'target_bins', 'is_vocal', 'bout_duration',
                    'session_ids'):
            artifact[key] = artifact[key][keep]
        del artifact['condition']
        del artifact['condition_levels']
        pickle_path = _rewrite_artifact(artifact, tmp_path / 'legacy.pkl')
        settings_path = _write_contrast_settings(
            extracted_cohort['settings'], tmp_path / 'settings.json', covariate_transform='linear')

        results = behavioral_response_contrast(
            input_pickle_path=pickle_path, output_directory=tmp_path / 'contrast',
            settings_path=settings_path)
        assert results['n_rows'] == int(keep.sum())

        with pytest.raises(ValueError, match="got 'deep_silence'"):
            behavioral_response_contrast(
                input_pickle_path=pickle_path, output_directory=tmp_path / 'contrast',
                settings_path=settings_path, control='deep_silence')

    def test_an_unknown_control_is_rejected(self, extracted_cohort, tmp_path):
        """
        A typo in ``control`` must not silently fall back to either silence;
        the error lists the valid controls.
        """

        settings_path = _write_contrast_settings(
            extracted_cohort['settings'], tmp_path / 'settings.json')

        with pytest.raises(ValueError, match=r'deep_silence.*inter_bout_silence'):
            behavioral_response_contrast(
                input_pickle_path=extracted_cohort['pickle_path'],
                output_directory=tmp_path / 'contrast', settings_path=settings_path,
                control='quiet')

    def test_a_control_with_no_rows_is_rejected(self, extracted_cohort, tmp_path):
        """
        A valid control level with zero rows would leave the contrast with
        nothing to compare against; it must raise rather than fit an
        unidentifiable design.
        """

        artifact = copy.deepcopy(extracted_cohort['artifact'])
        artifact['condition'] = np.where(artifact['condition'] == 2.0, 0.0, artifact['condition'])
        pickle_path = _rewrite_artifact(artifact, tmp_path / 'no_deep.pkl')
        settings_path = _write_contrast_settings(
            extracted_cohort['settings'], tmp_path / 'settings.json')

        with pytest.raises(ValueError, match="no 'deep_silence' rows"):
            behavioral_response_contrast(
                input_pickle_path=pickle_path, output_directory=tmp_path / 'contrast',
                settings_path=settings_path, control='deep_silence')

    def test_an_unknown_covariate_transform_is_rejected(self, extracted_cohort, tmp_path):
        """
        The transform is read from settings at fit time; an unrecognised value
        must raise rather than silently fitting untransformed covariates.
        """

        settings_path = _write_contrast_settings(
            extracted_cohort['settings'], tmp_path / 'settings.json', covariate_transform='sqrt')

        with pytest.raises(ValueError, match="got 'sqrt'"):
            behavioral_response_contrast(
                input_pickle_path=extracted_cohort['pickle_path'],
                output_directory=tmp_path / 'contrast', settings_path=settings_path)


class TestMatchedDivergenceExtraction:
    """The design-based counterpart, end-to-end on the same synthetic cohort."""

    @staticmethod
    def _extract(tmp_path: Path, caliper_sd: float) -> dict:
        """
        Builds a cohort (with one predictor-silent session) and runs the real
        matched-divergence extraction on it.

        Parameters
        ----------
        tmp_path : pathlib.Path
            Scratch directory.
        caliper_sd : float
            ``match_caliper_sd`` for the run.

        Returns
        -------
        artifact : dict
            The unpickled divergence artifact.
        """

        settings, save_dir = _build_response_cohort(
            base_dir=tmp_path, n_sessions=4, silent_male_sessions=(3,))
        settings['behavioral_response']['matched_divergence']['match_caliper_sd'] = caliper_sd
        # `save_dir` does not exist yet: the pipeline must create it itself.
        assert not save_dir.exists()
        MatchedDivergencePipeline(modeling_settings_dict=settings
                                  ).extract_and_save_matched_divergence()
        return _load_single_pickle(save_dir, 'matched_divergence_*.pkl')[1]

    def test_the_artifact_pairs_equal_width_curves(self, tmp_path):
        """
        Each stored pair is one vocal curve and the mean of its controls, both
        spanning ``[-pre_window, +window)`` = 60 frames with the anchor at index
        30. Checks the key set, the curve shapes, that session and control-count
        bookkeeping line up with the pairs, and that the silent-male session
        contributes no pair.
        """

        artifact = self._extract(tmp_path, caliper_sd=0.5)

        assert artifact['features'] == SYNTH_RESPONSE_FEATURES
        assert artifact['anchor_index'] == 30
        assert artifact['camera_fps'] == SYNTH_FPS
        assert artifact['parameters']['match_caliper_sd'] == 0.5
        assert set(artifact['pooled_scaling']) == set(SYNTH_RESPONSE_FEATURES)
        for feature in SYNTH_RESPONSE_FEATURES:
            vocal = artifact['pairs'][feature]['vocal']
            silence = artifact['pairs'][feature]['silence']
            assert vocal.shape[0] > 0
            assert vocal.shape == silence.shape == (vocal.shape[0], 60)
            assert len(artifact['pair_sessions'][feature]) == vocal.shape[0]
            assert artifact['pair_control_counts'][feature].shape == (vocal.shape[0],)
            assert np.all(artifact['pair_control_counts'][feature] >= 1)
            assert 'session_3' not in artifact['pair_sessions'][feature]
            assert artifact['caliper_rejected'][feature] == 0

    def test_matched_pairs_start_level_on_the_baseline(self, tmp_path):
        """
        The whole design rests on the two arms starting level: every control
        averaged into a pair lies within ``caliper_sd`` x (the session's
        control-baseline spread, about 1 on these pooled z-scores) of the vocal
        anchor's baseline, so the pair's baseline gap -- over the stored
        ``[-1.0, -0.2)`` s window, indices 0..23 -- must stay inside the caliper.
        """

        artifact = self._extract(tmp_path, caliper_sd=0.5)

        for feature in SYNTH_RESPONSE_FEATURES:
            vocal_baseline = artifact['pairs'][feature]['vocal'][:, :24].mean(axis=1)
            silence_baseline = artifact['pairs'][feature]['silence'][:, :24].mean(axis=1)
            assert np.max(np.abs(vocal_baseline - silence_baseline)) < 0.5

    def test_a_zero_caliper_rejects_every_anchor_and_still_publishes(self, tmp_path):
        """
        With a zero caliper no control can match a vocal baseline exactly, so
        every anchor is counted as rejected. The artifact must still be written,
        with empty ``(0, 60)`` curve blocks rather than a failed ``vstack``, so
        the failure is visible in the counts rather than as a crash.
        """

        artifact = self._extract(tmp_path, caliper_sd=0.0)

        for feature in SYNTH_RESPONSE_FEATURES:
            assert artifact['pairs'][feature]['vocal'].shape == (0, 60)
            assert artifact['pairs'][feature]['silence'].shape == (0, 60)
            assert artifact['pair_control_counts'][feature].size == 0
            assert artifact['caliper_rejected'][feature] > 0


class TestAllUsvTimes:
    """Every call in the session counts against silence, sorted by start."""

    def test_times_come_back_sorted_by_start_with_stops_aligned(self, tmp_path):
        """
        ``forward_clean_times`` assumes starts sorted ascending with each stop
        still paired to its own start; an unsorted table must be reordered as
        pairs, and every emitter's calls kept.
        """

        audio = tmp_path / 'session' / 'audio'
        audio.mkdir(parents=True)
        pls.DataFrame({'emitter': ['b', 'a', 'b'], 'start': [3.0, 1.0, 2.0],
                       'stop': [3.5, 1.5, 2.5]}).write_csv(file=audio / 'x_usv_summary.csv')

        starts, stops = _all_usv_times(tmp_path / 'session', ',')

        np.testing.assert_array_equal(starts, [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(stops, [1.5, 2.5, 3.5])

    def test_a_session_without_a_summary_yields_empty_arrays(self, tmp_path):
        """
        A session with no summary table has no calls on record; the caller
        skips it on empty arrays rather than crashing on a missing file.
        """

        (tmp_path / 'session' / 'audio').mkdir(parents=True)
        starts, stops = _all_usv_times(tmp_path / 'session', ',')

        assert starts.size == 0
        assert stops.size == 0


class TestHelperEdgeBranches:
    """Degenerate inputs the fit-time helpers must handle explicitly."""

    def test_forward_clean_times_on_no_candidates_is_empty(self):
        """
        No candidate times is an ordinary outcome of a short session; the mask
        must be an empty boolean array, not an indexing error.
        """

        mask = forward_clean_times(np.empty(0), np.array([1.0]), np.array([1.1]), 4.0)

        assert mask.dtype == bool
        assert mask.size == 0

    def test_normal_scores_leaves_an_all_nan_column_nan(self):
        """
        A column with no finite entry has nothing to rank; it must stay NaN
        (so the finite filter drops it visibly) while other columns transform.
        """

        values = np.column_stack([np.full(4, np.nan), np.arange(4.0)])
        transformed = normal_scores(values)

        assert np.all(np.isnan(transformed[:, 0]))
        assert np.all(np.isfinite(transformed[:, 1]))

    def test_derived_transform_logs_only_non_negative_columns(self):
        """
        The derived rule logs a non-negative base feature on its native scale
        and re-standardises it, and passes a signed one through untouched;
        the logged names are reported.
        """

        native = np.array([1.0, 2.0, 4.0, 8.0])
        scaling = {'speed': {'mean': float(native.mean()), 'std': float(native.std())},
                   'allo_pitch': {'mean': 0.0, 'std': 1.0}}
        z_scored = (native - native.mean()) / native.std()
        signed = np.array([-1.0, 0.5, 0.0, 2.0])
        transformed, logged = derived_covariate_transform(
            np.column_stack([z_scored, signed]),
            ['self.speed__mean_1s', 'other.allo_pitch__mean_1s'],
            scaling, {'speed': True, 'allo_pitch': False})

        expected = np.log(native)
        np.testing.assert_allclose(transformed[:, 0], (expected - expected.mean()) / expected.std())
        np.testing.assert_array_equal(transformed[:, 1], signed)
        assert logged == ['self.speed__mean_1s']

    def test_derived_transform_turns_non_positive_values_into_nan(self):
        """
        A log is undefined at or below zero; such values become NaN for the
        caller's finite filter rather than being offset into a fabricated value.
        """

        scaling = {'speed': {'mean': 0.0, 'std': 1.0}}
        transformed, _ = derived_covariate_transform(
            np.array([[0.0], [1.0], [np.e]]), ['self.speed__mean_1s'], scaling, {'speed': True})

        assert np.isnan(transformed[0, 0])
        assert np.all(np.isfinite(transformed[1:, 0]))

    def test_derived_transform_refuses_an_unrecorded_base_feature(self):
        """
        An artifact without the stored statistics cannot be returned to its
        native scale; the transform must raise rather than log a z-score.
        """

        with pytest.raises(KeyError, match='re-extracted'):
            derived_covariate_transform(
                np.zeros((3, 1)), ['self.speed__mean_1s'], {}, {'speed': True})

    def test_yeo_johnson_leaves_a_constant_column_at_the_identity(self):
        """
        A constant column has no likelihood to maximise; its power must be
        reported as the identity (1.0) and the column left untouched.
        """

        transformed, lambdas = yeo_johnson_covariates(
            np.full((5, 1), 0.3), ['self.speed__mean_1s'], {'speed': {'mean': 2.0, 'std': 1.0}})

        assert lambdas == {'self.speed__mean_1s': 1.0}
        np.testing.assert_array_equal(transformed, np.full((5, 1), 0.3))

    def test_the_continuous_design_needs_a_vocal_row(self):
        """
        Without a single vocal row there is no contrast and no duration to
        centre; the design builder must say so rather than dividing by zero.
        """

        with pytest.raises(ValueError, match='No vocal rows'):
            build_continuous_design_matrix(
                np.zeros((4, 1)), np.zeros(4), np.full(4, np.nan), ['c0'])

    def test_fit_contrast_refuses_when_no_row_is_finite(self):
        """
        When every row is lost to the finite filter the error must say how many
        targets were non-finite, instead of statsmodels failing on an empty fit.
        """

        design = np.column_stack([np.ones(6), np.arange(6.0)])

        with pytest.raises(ValueError, match='6 of 6 rows have a non-finite target'):
            fit_contrast(np.full(6, np.nan), design, ['intercept', 'x'],
                         np.array(['s0'] * 6), 'gaussian')

    def test_variance_split_on_the_raw_scale_for_a_gaussian_feature(self):
        """
        A Gaussian feature is decomposed on ``y`` itself, so negative targets
        must be kept; a planted vocal effect must then take a visible share.
        """

        generator = np.random.default_rng(8)
        treated = generator.integers(0, 2, 400).astype(float)
        design = np.column_stack([np.ones(400), treated, np.zeros(400),
                                  generator.normal(size=400)])
        target = 3.0 * treated - 1.5 + 0.1 * generator.normal(size=400)
        result = variance_explained_by_vocal_terms(
            target=target, full_design=design, n_contrast_terms=3, likelihood='gaussian')

        assert np.any(target < 0.0)
        assert result['vocal_share_percent'] > 50.0
