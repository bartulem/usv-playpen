"""
@author: bartulem
Unit tests for ``usv_playpen.modeling.behavioral_response_contrast``
— the GLM that asks whether a male vocal bout, and its duration, change female
behaviour relative to comparable inter-bout silence.

This module produces coefficients that go straight into a claim, so the tests
are weighted toward the failures that would still produce a plausible number.
Coverage:

* ``duration_tercile_labels`` — bands hold equal numbers of BOUTS, quiet rows are
  excluded rather than banded, and too few distinct durations raises.
* ``build_design_matrix`` — quiet rows are zero on every band column, each vocal
  row loads exactly one band, and labels line up with columns.
* ``fit_contrast`` — a planted step is recovered with the right sign and size, a
  planted dose-response comes back monotone, nothing planted yields nothing
  significant, session clustering widens the interval when the data actually
  cluster, and rank-deficient or empty designs raise rather than returning a
  meaningless fit.
"""

from __future__ import annotations

import numpy as np
import pytest

from usv_playpen.modeling.behavioral_response_contrast import (
    build_design_matrix,
    duration_tercile_labels,
    fit_contrast,
)


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
