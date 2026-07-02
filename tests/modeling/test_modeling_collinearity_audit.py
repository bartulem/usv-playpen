"""
@author: bartulem
Unit tests for the pure statistical helpers in
``usv_playpen.modeling.modeling_collinearity_audit``.

The collinearity / timescale audit drives whether two behavioural
predictors are treated as redundant and how many independent samples a
trace carries. The numerically load-bearing pieces are small, pure NumPy
functions with textbook closed forms, so the tests pin those forms
directly: VIF blows up to infinity for an exactly collinear or constant
column and sits near 1 for orthogonal columns; the FFT autocorrelation is
normalised to ``acf[0] == 1``; Sokal's integrated time is bounded below by
1; and the binary trace renderers light up exactly the frames an event
occupies.
"""

from __future__ import annotations

import pickle

import numpy as np
import polars as pl
import pytest

from usv_playpen.modeling.modeling_collinearity_audit import (
    _binary_event_trace,
    _build_event_summary_matrix,
    _first_crossing_below,
    _flagged_pairs,
    _integrated_autocorr_time,
    _per_session_acf,
    _vif_from_design,
    audit_predictor_collinearity,
    audit_predictor_timescales,
)


# _vif_from_design


class TestVifFromDesign:

    def test_orthogonal_columns_have_vif_near_one(self):
        """Independent (near-orthogonal) columns carry essentially no
        shared variance, so every VIF sits close to 1."""

        rng = np.random.default_rng(0)
        X = rng.standard_normal((5000, 3))
        vif = _vif_from_design(X)
        assert vif.shape == (3,)
        assert np.all(vif < 1.5)
        assert np.all(vif >= 1.0 - 1e-9)

    def test_exact_linear_combination_blows_up(self):
        """A column that is an exact linear combination of the others
        drives the VIF toward infinity (capped at the ``1/1e-12`` ceiling
        the implementation uses)."""

        rng = np.random.default_rng(1)
        c0 = rng.standard_normal(2000)
        c1 = rng.standard_normal(2000)
        c2 = c0 + c1  # exact collinearity
        X = np.column_stack([c0, c1, c2])
        vif = _vif_from_design(X)
        assert np.max(vif) > 1e6

    def test_constant_column_is_infinite(self):
        """A zero-variance column has undefined VIF and is reported as
        ``inf`` so it surfaces in the summary table."""

        rng = np.random.default_rng(2)
        X = np.column_stack([rng.standard_normal(1000),
                             np.ones(1000)])
        vif = _vif_from_design(X)
        assert np.isinf(vif[1])

    def test_underdetermined_returns_nan(self):
        """When samples do not exceed features + 1 the system is
        under-determined and every VIF is NaN."""

        X = np.ones((3, 4))
        vif = _vif_from_design(X)
        assert vif.shape == (4,)
        assert np.all(np.isnan(vif))


# _flagged_pairs


class TestFlaggedPairs:

    def test_tiers_and_sorting(self):
        """Pairs at/above the exclude threshold are tier ``'exclude'``,
        those at/above the concern threshold ``'concern'``; weaker pairs
        are dropped and the survivors come back sorted by descending
        absolute correlation."""

        names = ['a', 'b', 'c']
        rho = np.array([
            [1.00, 0.90, 0.10],
            [0.90, 1.00, 0.75],
            [0.10, 0.75, 1.00],
        ])
        pairs = _flagged_pairs(rho, names, concern_thresh=0.7, exclude_thresh=0.85)
        assert [(p[0], p[1], p[3]) for p in pairs] == [
            ('a', 'b', 'exclude'),
            ('b', 'c', 'concern'),
        ]
        # Sorted by descending |rho|.
        assert abs(pairs[0][2]) >= abs(pairs[1][2])

    def test_non_finite_entries_skipped(self):
        """NaN correlations cannot be flagged and are silently ignored."""

        names = ['a', 'b']
        rho = np.array([[1.0, np.nan], [np.nan, 1.0]])
        assert _flagged_pairs(rho, names) == []

    def test_negative_correlation_flagged_by_magnitude(self):
        """A strong negative correlation is flagged on its magnitude and
        keeps its signed value in the tuple."""

        names = ['a', 'b']
        rho = np.array([[1.0, -0.92], [-0.92, 1.0]])
        pairs = _flagged_pairs(rho, names)
        assert len(pairs) == 1
        assert pairs[0][2] == pytest.approx(-0.92)
        assert pairs[0][3] == 'exclude'


# _per_session_acf


class TestPerSessionAcf:

    def test_length_and_lag_zero_normalisation(self):
        """The ACF spans ``max_lag + 1`` lags and is normalised so that
        ``acf[0] == 1`` for any non-constant input."""

        rng = np.random.default_rng(3)
        trace = rng.standard_normal(500)
        acf = _per_session_acf(trace, max_lag_frames=20)
        assert acf.shape == (21,)
        assert acf[0] == pytest.approx(1.0)

    def test_constant_trace_is_nan(self):
        """A constant trace has zero variance, so the ACF is undefined and
        returned as all-NaN."""

        acf = _per_session_acf(np.full(100, 3.0), max_lag_frames=10)
        assert acf.shape == (11,)
        assert np.all(np.isnan(acf))

    def test_autocorrelated_series_decays_from_one(self):
        """A smooth (cumulative-sum) series stays positively correlated at
        short lags: ``acf[1]`` is high and below ``acf[0]``."""

        rng = np.random.default_rng(4)
        trace = np.cumsum(rng.standard_normal(1000))
        acf = _per_session_acf(trace, max_lag_frames=30)
        assert acf[1] > 0.9
        assert acf[1] <= acf[0]


# _integrated_autocorr_time


class TestIntegratedAutocorrTime:

    def test_white_noise_is_at_least_one(self):
        """Only positive lags up to the first non-positive crossing are
        summed, so the integrated time is bounded below by 1."""

        rng = np.random.default_rng(5)
        acf = _per_session_acf(rng.standard_normal(2000), max_lag_frames=50)
        tau = _integrated_autocorr_time(acf)
        assert tau >= 1.0

    def test_autocorrelated_series_exceeds_white_noise(self):
        """A strongly autocorrelated series carries fewer independent
        samples, so its integrated time is well above 1."""

        rng = np.random.default_rng(6)
        acf = _per_session_acf(np.cumsum(rng.standard_normal(2000)),
                               max_lag_frames=100)
        assert _integrated_autocorr_time(acf) > 5.0

    def test_degenerate_acf_is_nan(self):
        """An ACF containing NaN is degenerate and returns NaN."""

        assert np.isnan(_integrated_autocorr_time(np.array([1.0, np.nan, 0.2])))


# _first_crossing_below


class TestFirstCrossingBelow:

    def test_returns_first_lag_under_threshold(self):
        """The first lag whose ACF dips below the threshold is returned."""

        acf = np.array([1.0, 0.8, 0.5, 0.3, 0.1])
        assert _first_crossing_below(acf, threshold=0.4) == 3.0

    def test_no_crossing_returns_nan(self):
        """If the ACF never dips below the threshold within the window the
        result is NaN."""

        acf = np.array([1.0, 0.9, 0.85, 0.8])
        assert np.isnan(_first_crossing_below(acf, threshold=0.5))

    def test_non_finite_acf_returns_nan(self):
        """A NaN-bearing ACF short-circuits to NaN."""

        assert np.isnan(_first_crossing_below(np.array([1.0, np.nan]), threshold=0.5))


# _binary_event_trace


class TestBinaryEventTrace:

    def test_lights_up_onset_frames(self):
        """Each event onset sets exactly its floor(t * fps) frame to 1."""

        trace = _binary_event_trace(np.array([0.0, 1.0, 1.95]), n_frames=20, fps=10.0)
        assert trace.shape == (20,)
        assert trace.dtype == np.float32
        assert trace[0] == 1.0
        assert trace[10] == 1.0
        assert trace[19] == 1.0  # floor(1.95 * 10) == 19
        assert trace.sum() == 3.0

    def test_empty_event_list_is_all_zero(self):
        """No events -> an all-zero trace of the requested length."""

        assert not _binary_event_trace(np.array([]), n_frames=15, fps=10.0).any()

    def test_out_of_range_events_dropped(self):
        """Events landing outside ``[0, n_frames)`` are discarded, not
        clipped onto the edge frames."""

        trace = _binary_event_trace(np.array([-1.0, 100.0]), n_frames=10, fps=10.0)
        assert not trace.any()


# Synthetic-fixture builders shared by the orchestrator tests


def _make_session_df(n_frames: int, seed: int) -> pl.DataFrame:
    """
    Builds one synthetic per-session feature DataFrame in the exact column
    naming the audits expect.

    The frame carries two ``self``-mappable columns (``m0.speed``,
    ``m0.accel``), one ``other``-mappable column (``m1.speed``), one
    generic dyadic column (``nose-nose``), and one digit-suffixed embedding
    column (``m0.embedding.3``) that the suffix-is-digit guard must skip.
    ``m0.accel`` is built as a noisy multiple of ``m0.speed`` so the two
    carry a strong but not perfect Spearman correlation, giving the
    collinearity audit a non-trivial off-diagonal entry to report.

    Parameters
    ----------
    n_frames : int
        Number of per-frame rows in the returned DataFrame.
    seed : int
        Seed for the per-session noise so every session differs.

    Returns
    -------
    pl.DataFrame
        A ``(n_frames, 5)`` DataFrame whose column names follow the
        ``{mouse}.{feat}`` / ``dyad`` conventions of the real loader.
    """

    rng = np.random.default_rng(seed)
    speed = rng.standard_normal(n_frames).astype(np.float64)
    accel = 3.0 * speed + 0.05 * rng.standard_normal(n_frames)
    other_speed = rng.standard_normal(n_frames).astype(np.float64)
    nose = rng.standard_normal(n_frames).astype(np.float64)
    embedding = rng.standard_normal(n_frames).astype(np.float64)
    return pl.DataFrame({
        'm0.speed': speed,
        'm0.accel': accel,
        'm1.speed': other_speed,
        'nose-nose': nose,
        'm0.embedding.3': embedding,
    })


# _build_event_summary_matrix


class TestBuildEventSummaryMatrix:

    def test_generic_naming_shapes_and_digit_skip(self):
        """The summary matrix renames target/predictor columns to
        ``self.*`` / ``other.*``, keeps generic dyadic columns verbatim,
        drops digit-suffixed embedding columns, and returns one row per
        in-bounds event onset pooled across sessions."""

        fps = 10.0
        history_frames = 5
        beh = {
            's0': _make_session_df(n_frames=200, seed=1),
            's1': _make_session_df(n_frames=200, seed=2),
        }
        names = {'s0': ['m0', 'm1'], 's1': ['m0', 'm1']}
        cam = {'s0': fps, 's1': fps}
        # Every onset window [round(t*fps) - 5, round(t*fps)] fits inside.
        events = {
            's0': np.array([1.0, 2.0, 3.0, 4.0]),
            's1': np.array([5.0, 6.0, 7.0]),
        }
        feats, X = _build_event_summary_matrix(
            processed_beh_dict=beh,
            event_times_per_session=events,
            mouse_names_dict=names,
            target_idx=0,
            predictor_idx=1,
            history_frames=history_frames,
            camera_fps_dict=cam,
        )
        assert feats == ['nose-nose', 'other.speed', 'self.accel', 'self.speed']
        assert X.shape == (7, 4)
        assert X.dtype == np.float32

    def test_out_of_bounds_events_dropped(self):
        """Events whose pre-event window underflows frame 0 or whose onset
        overshoots the recording are skipped, so only the in-bounds onsets
        contribute rows."""

        fps = 10.0
        beh = {'s0': _make_session_df(n_frames=100, seed=3)}
        names = {'s0': ['m0', 'm1']}
        cam = {'s0': fps}
        # t=0.1 -> end=1, start=-4 (underflow); t=20.0 -> end=200 > 100
        # (overshoot); t=5.0 -> end=50, start=45 (valid).
        events = {'s0': np.array([0.1, 20.0, 5.0])}
        _feats, X = _build_event_summary_matrix(
            processed_beh_dict=beh,
            event_times_per_session=events,
            mouse_names_dict=names,
            target_idx=0,
            predictor_idx=1,
            history_frames=5,
            camera_fps_dict=cam,
        )
        assert X.shape == (1, 4)

    def test_feature_absent_from_one_session_is_dropped(self):
        """A feature present in only some contributing sessions cannot
        column-stack rectangularly, so it is dropped while the features
        shared by all contributing sessions survive."""

        fps = 10.0
        df_full = _make_session_df(n_frames=120, seed=4)
        df_partial = df_full.drop('nose-nose')  # missing dyadic feature
        beh = {'s0': df_full, 's1': df_partial}
        names = {'s0': ['m0', 'm1'], 's1': ['m0', 'm1']}
        cam = {'s0': fps, 's1': fps}
        events = {'s0': np.array([2.0, 3.0]), 's1': np.array([4.0, 5.0])}
        feats, X = _build_event_summary_matrix(
            processed_beh_dict=beh,
            event_times_per_session=events,
            mouse_names_dict=names,
            target_idx=0,
            predictor_idx=1,
            history_frames=5,
            camera_fps_dict=cam,
        )
        assert 'nose-nose' not in feats
        assert X.shape == (4, 3)

    def test_nan_window_is_zero_filled(self):
        """A window containing NaN is mean-imputed to zero before averaging
        rather than poisoning the row with NaN, so the produced row stays
        finite."""

        fps = 10.0
        df = _make_session_df(n_frames=100, seed=20)
        col = df['m0.speed'].to_numpy().copy()
        col[44:50] = np.nan  # falls inside the [45, 50] window of t=5.0
        df = df.with_columns(pl.Series('m0.speed', col))
        beh = {'s0': df}
        names = {'s0': ['m0', 'm1']}
        cam = {'s0': fps}
        events = {'s0': np.array([5.0])}
        _feats, X = _build_event_summary_matrix(
            processed_beh_dict=beh,
            event_times_per_session=events,
            mouse_names_dict=names,
            target_idx=0,
            predictor_idx=1,
            history_frames=5,
            camera_fps_dict=cam,
        )
        assert np.all(np.isfinite(X))

    def test_no_common_feature_returns_empty(self):
        """If two contributing sessions share no feature at all, no feature
        is present for every session, so the builder returns empty."""

        fps = 10.0
        df0 = pl.DataFrame({'m0.speed': np.arange(100.0)})
        df1 = pl.DataFrame({'m0.accel': np.arange(100.0)})
        beh = {'s0': df0, 's1': df1}
        names = {'s0': ['m0', 'm1'], 's1': ['m0', 'm1']}
        cam = {'s0': fps, 's1': fps}
        events = {'s0': np.array([5.0]), 's1': np.array([6.0])}
        feats, X = _build_event_summary_matrix(
            processed_beh_dict=beh,
            event_times_per_session=events,
            mouse_names_dict=names,
            target_idx=0,
            predictor_idx=1,
            history_frames=5,
            camera_fps_dict=cam,
        )
        assert feats == []
        assert X.shape == (0, 0)

    def test_no_inbounds_events_returns_empty(self):
        """When no session contributes any in-bounds event the builder
        short-circuits to an empty feature list and a ``(0, 0)`` matrix."""

        fps = 10.0
        beh = {'s0': _make_session_df(n_frames=50, seed=5)}
        names = {'s0': ['m0', 'm1']}
        cam = {'s0': fps}
        events = {'s0': np.array([100.0])}  # onset frame 1000 > 50
        feats, X = _build_event_summary_matrix(
            processed_beh_dict=beh,
            event_times_per_session=events,
            mouse_names_dict=names,
            target_idx=0,
            predictor_idx=1,
            history_frames=5,
            camera_fps_dict=cam,
        )
        assert feats == []
        assert X.shape == (0, 0)

    def test_session_without_events_or_names_skipped(self):
        """Sessions missing from the event-times or mouse-names maps, and
        sessions with an empty event array, contribute no rows."""

        fps = 10.0
        beh = {
            's_no_events': _make_session_df(n_frames=100, seed=6),
            's_empty': _make_session_df(n_frames=100, seed=7),
            's_no_names': _make_session_df(n_frames=100, seed=8),
            's_good': _make_session_df(n_frames=100, seed=9),
        }
        names = {
            's_no_events': ['m0', 'm1'],
            's_empty': ['m0', 'm1'],
            's_good': ['m0', 'm1'],
        }
        cam = {k: fps for k in beh}
        events = {
            's_empty': np.array([]),
            's_no_names': np.array([5.0]),
            's_good': np.array([5.0, 6.0]),
        }
        _feats, X = _build_event_summary_matrix(
            processed_beh_dict=beh,
            event_times_per_session=events,
            mouse_names_dict=names,
            target_idx=0,
            predictor_idx=1,
            history_frames=5,
            camera_fps_dict=cam,
        )
        # Only 's_good' contributes its two in-bounds onsets.
        assert X.shape == (2, 4)


# audit_predictor_collinearity


class TestAuditPredictorCollinearity:

    def test_full_payload_structure_and_artifact(self, tmp_path):
        """A populated audit returns square Spearman/Pearson matrices, a
        per-feature VIF vector, a finite condition number, and persists an
        identical payload to ``save_path``; the strongly correlated
        ``self.speed`` / ``self.accel`` pair is flagged."""

        fps = 10.0
        beh = {
            's0': _make_session_df(n_frames=400, seed=11),
            's1': _make_session_df(n_frames=400, seed=12),
        }
        names = {'s0': ['m0', 'm1'], 's1': ['m0', 'm1']}
        cam = {'s0': fps, 's1': fps}
        events = {
            's0': np.arange(1.0, 30.0, 0.5),
            's1': np.arange(1.0, 30.0, 0.5),
        }
        save_path = tmp_path / 'sub' / 'collinearity.pkl'
        payload = audit_predictor_collinearity(
            processed_beh_dict=beh,
            event_times_per_session=events,
            mouse_names_dict=names,
            target_idx=0,
            predictor_idx=1,
            history_frames=5,
            camera_fps_dict=cam,
            save_path=str(save_path),
            source_pickle='input.pkl',
            input_metadata={'pipeline': 'vocal_onsets'},
        )
        n_feat = len(payload['features'])
        assert n_feat == 4
        assert payload['spearman_rho'].shape == (n_feat, n_feat)
        assert payload['pearson_rho'].shape == (n_feat, n_feat)
        assert payload['vif'].shape == (n_feat,)
        assert np.isfinite(payload['condition_number'])
        assert payload['n_events'] > 0
        assert payload['source_pickle'] == 'input.pkl'
        assert payload['_input_metadata'] == {'pipeline': 'vocal_onsets'}
        flagged_names = {frozenset((f1, f2)) for f1, f2, *_ in payload['flagged_pairs']}
        assert frozenset(('self.speed', 'self.accel')) in flagged_names
        # Artifact on disk matches the returned payload feature list.
        with save_path.open('rb') as fh:
            on_disk = pickle.load(fh)
        assert on_disk['features'] == payload['features']

    def test_empty_matrix_embeds_metadata(self, tmp_path):
        """The empty-summary-matrix early branch embeds a supplied
        ``input_metadata`` block under the reserved key."""

        fps = 10.0
        beh = {'s0': _make_session_df(n_frames=40, seed=40)}
        names = {'s0': ['m0', 'm1']}
        cam = {'s0': fps}
        events = {'s0': np.array([500.0])}
        payload = audit_predictor_collinearity(
            processed_beh_dict=beh,
            event_times_per_session=events,
            mouse_names_dict=names,
            target_idx=0,
            predictor_idx=1,
            history_frames=5,
            camera_fps_dict=cam,
            save_path=str(tmp_path / 'em.pkl'),
            source_pickle='input.pkl',
            input_metadata={'pipeline': 'bout'},
        )
        assert payload['_input_metadata'] == {'pipeline': 'bout'}

    def test_empty_matrix_payload(self, tmp_path):
        """With no in-bounds events the audit writes the documented
        empty-matrix payload (zero events, NaN condition number, no
        flagged pairs)."""

        fps = 10.0
        beh = {'s0': _make_session_df(n_frames=40, seed=13)}
        names = {'s0': ['m0', 'm1']}
        cam = {'s0': fps}
        events = {'s0': np.array([500.0])}  # out of bounds
        save_path = tmp_path / 'empty.pkl'
        payload = audit_predictor_collinearity(
            processed_beh_dict=beh,
            event_times_per_session=events,
            mouse_names_dict=names,
            target_idx=0,
            predictor_idx=1,
            history_frames=5,
            camera_fps_dict=cam,
            save_path=str(save_path),
            source_pickle='input.pkl',
        )
        assert payload['features'] == []
        assert payload['n_events'] == 0
        assert payload['flagged_pairs'] == []
        assert np.isnan(payload['condition_number'])
        assert save_path.exists()
        assert '_input_metadata' not in payload

    def test_zero_variance_feature_dropped(self, tmp_path):
        """A constant feature is dropped before the correlation/VIF step
        and does not appear in the surviving feature list."""

        fps = 10.0
        df = _make_session_df(n_frames=400, seed=14)
        df = df.with_columns(pl.lit(7.0).alias('m0.speed'))  # constant self col
        beh = {'s0': df}
        names = {'s0': ['m0', 'm1']}
        cam = {'s0': fps}
        events = {'s0': np.arange(1.0, 30.0, 0.5)}
        save_path = tmp_path / 'zerovar.pkl'
        payload = audit_predictor_collinearity(
            processed_beh_dict=beh,
            event_times_per_session=events,
            mouse_names_dict=names,
            target_idx=0,
            predictor_idx=1,
            history_frames=5,
            camera_fps_dict=cam,
            save_path=str(save_path),
            source_pickle='input.pkl',
        )
        assert 'self.speed' not in payload['features']
        assert len(payload['features']) == 3

    def test_moderate_vif_tier(self, tmp_path):
        """A feature with moderate (5 < VIF <= 10) collinearity surfaces in
        the audit with a finite VIF in that band, exercising the
        intermediate VIF-tagging branch of the stdout summary."""

        fps = 10.0
        rng = np.random.default_rng(50)
        n = 3000
        speed = rng.standard_normal(n)
        # accel shares ~85% of speed's variance -> R^2 ~ 0.85 -> VIF ~ 6.7.
        accel = speed + np.sqrt(1.0 / 0.85 - 1.0) * rng.standard_normal(n)
        df = pl.DataFrame({
            'm0.speed': speed,
            'm0.accel': accel,
            'm1.speed': rng.standard_normal(n),
        })
        beh = {'s0': df}
        names = {'s0': ['m0', 'm1']}
        cam = {'s0': fps}
        # Dense events so the event-pooled design has plenty of rows.
        events = {'s0': np.arange(1.0, 290.0, 0.2)}
        payload = audit_predictor_collinearity(
            processed_beh_dict=beh,
            event_times_per_session=events,
            mouse_names_dict=names,
            target_idx=0,
            predictor_idx=1,
            history_frames=3,
            camera_fps_dict=cam,
            save_path=str(tmp_path / 'modvif.pkl'),
            source_pickle='input.pkl',
        )
        vif = payload['vif']
        # At least one feature sits in the intermediate 5 < VIF <= 10 band.
        assert np.any((vif > 5.0) & (vif <= 10.0))

    def test_all_features_zero_variance(self, tmp_path):
        """If every surviving feature is constant the audit reports empty
        correlation matrices and a NaN condition number rather than
        crashing on a degenerate design."""

        fps = 10.0
        n = 400
        df = pl.DataFrame({
            'm0.speed': np.full(n, 1.0),
            'm0.accel': np.full(n, 2.0),
            'm1.speed': np.full(n, 3.0),
            'nose-nose': np.full(n, 4.0),
        })
        beh = {'s0': df}
        names = {'s0': ['m0', 'm1']}
        cam = {'s0': fps}
        events = {'s0': np.arange(1.0, 30.0, 0.5)}
        save_path = tmp_path / 'allconst.pkl'
        payload = audit_predictor_collinearity(
            processed_beh_dict=beh,
            event_times_per_session=events,
            mouse_names_dict=names,
            target_idx=0,
            predictor_idx=1,
            history_frames=5,
            camera_fps_dict=cam,
            save_path=str(save_path),
            source_pickle='input.pkl',
        )
        assert payload['features'] == []
        assert payload['spearman_rho'].shape == (0, 0)
        assert payload['vif'].shape == (0,)
        assert np.isnan(payload['condition_number'])
        assert payload['flagged_pairs'] == []
        # n_events is still the pooled row count, not zeroed.
        assert payload['n_events'] > 0

    def test_two_feature_spearman_fallback(self, tmp_path):
        """With exactly two surviving features ``scipy.stats.spearmanr``
        returns a scalar; the audit must still produce a ``(2, 2)``
        Spearman matrix via the rankdata fallback."""

        fps = 10.0
        df = _make_session_df(n_frames=400, seed=30)
        # Keep only two non-digit, non-constant feature columns.
        df = df.select(['m0.speed', 'm1.speed'])
        beh = {'s0': df}
        names = {'s0': ['m0', 'm1']}
        cam = {'s0': fps}
        events = {'s0': np.arange(1.0, 30.0, 0.5)}
        save_path = tmp_path / 'twofeat.pkl'
        payload = audit_predictor_collinearity(
            processed_beh_dict=beh,
            event_times_per_session=events,
            mouse_names_dict=names,
            target_idx=0,
            predictor_idx=1,
            history_frames=5,
            camera_fps_dict=cam,
            save_path=str(save_path),
            source_pickle='input.pkl',
        )
        assert payload['features'] == ['other.speed', 'self.speed']
        assert payload['spearman_rho'].shape == (2, 2)
        np.testing.assert_allclose(np.diag(payload['spearman_rho']), 1.0, atol=1e-5)


# audit_predictor_timescales


def _make_timescale_inputs(n_frames: int, fps: float, n_sessions: int):
    """
    Assembles a complete, minimal input set for the timescale audit.

    Each session gets a synthetic feature DataFrame, a sparse bout-onset
    time array (the audit's sole ``Y`` source), and a per-USV
    ``(starts, stops)`` interval pair (the IBI-percentile source). All
    sessions share one fps so the reported lag-in-seconds axis is
    unambiguous.

    Parameters
    ----------
    n_frames : int
        Per-session recording length in frames.
    fps : float
        Shared camera sampling rate.
    n_sessions : int
        Number of synthetic sessions to generate.

    Returns
    -------
    tuple
        ``(processed_beh_dict, mouse_names_dict, camera_fps_dict,
        onset_times_per_session, event_intervals_per_session,
        event_times_per_session)`` ready to splat into the audit.
    """

    beh, names, cam, bouts, intervals, events = {}, {}, {}, {}, {}, {}
    duration_s = n_frames / fps
    for k in range(n_sessions):
        sid = f's{k}'
        beh[sid] = _make_session_df(n_frames=n_frames, seed=100 + k)
        names[sid] = ['m0', 'm1']
        cam[sid] = fps
        onset_times = np.arange(0.5, duration_s - 0.5, 0.7) + 0.01 * k
        bouts[sid] = onset_times
        starts = onset_times
        stops = onset_times + 0.05
        intervals[sid] = (starts, stops)
        events[sid] = onset_times
    return beh, names, cam, bouts, intervals, events


@pytest.mark.filterwarnings('ignore:Mean of empty slice:RuntimeWarning')
@pytest.mark.filterwarnings(
    'ignore:invalid value encountered in scalar divide:RuntimeWarning')
class TestAuditPredictorTimescales:

    def test_raises_when_max_lag_exceeds_shuffle_floor(self, tmp_path):
        """A config where max_lag_seconds > shuffle_range_seconds[0] would make the
        signal-correlation null's slice start index negative (silently wrapping and
        corrupting the null); the audit raises a clear ValueError up front instead."""
        fps = 10.0
        beh, names, cam, bouts, intervals, _events = _make_timescale_inputs(
            n_frames=800, fps=fps, n_sessions=3,
        )
        with pytest.raises(ValueError, match="must be >= max_lag_frames"):
            audit_predictor_timescales(
                processed_beh_dict=beh,
                mouse_names_dict=names,
                target_idx=0,
                predictor_idx=1,
                configured_filter_history=1.0,
                camera_fps_dict=cam,
                max_lag_seconds=5.0,                  # 50 frames @ 10 fps
                n_shuffles=8,
                ibi_thresholds={'male': 0.25, 'female': 0.30},
                save_path=str(tmp_path / 'ts' / 'timescales.pkl'),
                source_pickle='input.pkl',
                random_seed=0,
                input_metadata={'pipeline': 'vocal_onsets'},
                shuffle_range_seconds=(3.0, 6.0),     # floor 30 frames < 50 -> invalid
                event_intervals_per_session=intervals,
                onset_times_per_session=bouts,
            )

    def test_full_payload_structure_and_artifact(self, tmp_path):
        """A populated timescale audit returns ACF and signal-correlation
        blocks with the documented shapes, the symmetric vs positive lag
        grids, populated IBI percentiles, and persists an identical payload
        to disk."""

        fps = 10.0
        n_frames = 800
        max_lag_seconds = 2.0
        beh, names, cam, bouts, intervals, events = _make_timescale_inputs(
            n_frames=n_frames, fps=fps, n_sessions=3,
        )
        save_path = tmp_path / 'ts' / 'timescales.pkl'
        payload = audit_predictor_timescales(
            processed_beh_dict=beh,
            mouse_names_dict=names,
            target_idx=0,
            predictor_idx=1,
            configured_filter_history=1.0,
            camera_fps_dict=cam,
            max_lag_seconds=max_lag_seconds,
            n_shuffles=8,
            ibi_thresholds={'male': 0.25, 'female': 0.30},
            save_path=str(save_path),
            source_pickle='input.pkl',
            random_seed=0,
            input_metadata={'pipeline': 'vocal_onsets'},
            shuffle_range_seconds=(3.0, 5.0),
            event_intervals_per_session=intervals,
            onset_times_per_session=bouts,
        )

        n_feat = len(payload['features'])
        assert n_feat == 4  # self.speed, self.accel, other.speed, nose-nose
        max_lag_frames = int(np.ceil(max_lag_seconds * fps))
        n_acf_lags = max_lag_frames + 1
        n_signal_lags = 2 * max_lag_frames + 1

        assert payload['acf_lags_frames'].shape == (n_acf_lags,)
        assert payload['acf_median'].shape == (n_feat, n_acf_lags)
        assert payload['acf_p25'].shape == (n_feat, n_acf_lags)
        assert payload['acf_null_mean'].shape == (n_feat, n_acf_lags)
        assert payload['tau_acf_integrated'].shape == (n_feat,)
        # acf[0] is normalised to 1 for every non-constant feature.
        np.testing.assert_allclose(payload['acf_median'][:, 0], 1.0, atol=1e-5)

        assert payload['signal_lags_frames'].shape == (n_signal_lags,)
        assert payload['signal_lags_frames'][0] == -max_lag_frames
        assert payload['signal_lags_frames'][-1] == max_lag_frames
        assert payload['rho_signal'].shape == (n_feat, n_signal_lags)
        assert payload['rho_signal_per_session_sem'].shape == (n_feat, n_signal_lags)
        assert payload['rho_signal_null_p99_5'].shape == (n_feat, n_signal_lags)

        assert payload['ibi_thresholds'] == {'male': 0.25, 'female': 0.30}
        assert np.isfinite(payload['ibi_empirical_pcts']['p90'])
        assert payload['n_sessions'] == 3
        assert payload['n_bouts'] > 0
        assert payload['n_usvs'] > 0
        assert payload['_input_metadata'] == {'pipeline': 'vocal_onsets'}

        with save_path.open('rb') as fh:
            on_disk = pickle.load(fh)
        assert on_disk['features'] == payload['features']
        np.testing.assert_array_equal(
            on_disk['signal_lags_frames'], payload['signal_lags_frames'])

    def test_circular_shift_null_reproducible_across_calls(self, tmp_path):
        """The within-session circular-shift null is seeded by
        ``random_seed``, so two audits run with the same seed produce
        identical null bands while a different seed perturbs them."""

        fps = 10.0
        beh, names, cam, bouts, intervals, events = _make_timescale_inputs(
            n_frames=800, fps=fps, n_sessions=3,
        )
        common = dict(
            processed_beh_dict=beh,
            mouse_names_dict=names,
            target_idx=0,
            predictor_idx=1,
            configured_filter_history=1.0,
            camera_fps_dict=cam,
            max_lag_seconds=2.0,
            n_shuffles=12,
            ibi_thresholds={'male': 0.25, 'female': 0.30},
            source_pickle='input.pkl',
            shuffle_range_seconds=(3.0, 5.0),
            event_intervals_per_session=intervals,
            onset_times_per_session=bouts,
        )
        p_a = audit_predictor_timescales(
            save_path=str(tmp_path / 'a.pkl'), random_seed=7, **common)
        p_b = audit_predictor_timescales(
            save_path=str(tmp_path / 'b.pkl'), random_seed=7, **common)
        p_c = audit_predictor_timescales(
            save_path=str(tmp_path / 'c.pkl'), random_seed=99, **common)

        np.testing.assert_array_equal(
            p_a['rho_signal_null_mean'], p_b['rho_signal_null_mean'])
        np.testing.assert_array_equal(
            p_a['acf_null_mean'], p_b['acf_null_mean'])
        # The actual (non-shuffled) line is deterministic regardless of seed.
        np.testing.assert_array_equal(p_a['rho_signal'], p_c['rho_signal'])
        # A different seed moves the null band.
        assert not np.allclose(
            p_a['rho_signal_null_mean'], p_c['rho_signal_null_mean'],
            equal_nan=True)

    def test_negative_lag_direction_bout_leads_feature(self, tmp_path):
        """When a predictor is a delayed copy of the bout-onset train the
        signal-correlation peak lands at a negative lag (bout precedes
        feature), exercising the ``bout leads feature`` headline branch."""

        fps = 10.0
        n_frames = 800
        delay_frames = 8  # feature lags the bout train by 0.8 s
        beh, names, cam, bouts, intervals, events = _make_timescale_inputs(
            n_frames=n_frames, fps=fps, n_sessions=2,
        )
        # Overwrite one feature with a delayed copy of the bout impulse
        # train so corr(feature[t], Y[t - delay]) peaks at a negative lag.
        for sid in beh:
            y = _binary_event_trace(bouts[sid], n_frames, fps)
            delayed = np.zeros(n_frames, dtype=np.float64)
            delayed[delay_frames:] = y[:n_frames - delay_frames]
            delayed += 1e-3 * np.arange(n_frames)  # break the all-zero tail
            beh[sid] = beh[sid].with_columns(pl.Series('m0.speed', delayed))
        payload = audit_predictor_timescales(
            processed_beh_dict=beh,
            mouse_names_dict=names,
            target_idx=0,
            predictor_idx=1,
            configured_filter_history=1.0,
            camera_fps_dict=cam,
            max_lag_seconds=2.0,
            n_shuffles=6,
            ibi_thresholds={'male': 0.25, 'female': 0.30},
            save_path=str(tmp_path / 'neg_lag.pkl'),
            source_pickle='input.pkl',
            shuffle_range_seconds=(3.0, 5.0),
            event_intervals_per_session=intervals,
            onset_times_per_session=bouts,
        )
        idx = payload['features'].index('self.speed')
        row = payload['rho_signal'][idx]
        peak_lag_frames = payload['signal_lags_frames'][int(np.nanargmax(np.abs(row)))]
        assert peak_lag_frames < 0

    def test_zero_lag_direction_simultaneous(self, tmp_path):
        """A predictor equal to the bout-onset train peaks at lag 0, so the
        headline reports the ``simultaneous`` direction branch."""

        fps = 10.0
        n_frames = 800
        beh, names, cam, bouts, intervals, events = _make_timescale_inputs(
            n_frames=n_frames, fps=fps, n_sessions=2,
        )
        for sid in beh:
            y = _binary_event_trace(bouts[sid], n_frames, fps).astype(np.float64)
            y += 1e-3 * np.arange(n_frames)  # break the all-zero tail
            beh[sid] = beh[sid].with_columns(pl.Series('m0.speed', y))
        payload = audit_predictor_timescales(
            processed_beh_dict=beh,
            mouse_names_dict=names,
            target_idx=0,
            predictor_idx=1,
            configured_filter_history=1.0,
            camera_fps_dict=cam,
            max_lag_seconds=2.0,
            n_shuffles=6,
            ibi_thresholds={'male': 0.25, 'female': 0.30},
            save_path=str(tmp_path / 'zero_lag.pkl'),
            source_pickle='input.pkl',
            shuffle_range_seconds=(3.0, 5.0),
            event_intervals_per_session=intervals,
            onset_times_per_session=bouts,
        )
        idx = payload['features'].index('self.speed')
        row = payload['rho_signal'][idx]
        peak_lag_frames = payload['signal_lags_frames'][int(np.nanargmax(np.abs(row)))]
        assert peak_lag_frames == 0

    def test_missing_bout_onsets_raises(self, tmp_path):
        """Omitting ``onset_times_per_session`` removes the audit's
        sole ``Y`` source and must raise ``ValueError``."""

        fps = 10.0
        beh, names, cam, _, intervals, events = _make_timescale_inputs(
            n_frames=400, fps=fps, n_sessions=2,
        )
        with pytest.raises(ValueError, match='onset_times_per_session'):
            audit_predictor_timescales(
                processed_beh_dict=beh,
                mouse_names_dict=names,
                target_idx=0,
                predictor_idx=1,
                configured_filter_history=1.0,
                camera_fps_dict=cam,
                max_lag_seconds=2.0,
                n_shuffles=4,
                ibi_thresholds={'male': 0.25, 'female': 0.30},
                save_path=str(tmp_path / 'x.pkl'),
                source_pickle='input.pkl',
                event_intervals_per_session=intervals,
                onset_times_per_session=None,
            )

    def test_missing_event_intervals_raises(self, tmp_path):
        """Omitting ``event_intervals_per_session`` removes the
        IBI-percentile source and must raise ``ValueError``."""

        fps = 10.0
        beh, names, cam, bouts, _, events = _make_timescale_inputs(
            n_frames=400, fps=fps, n_sessions=2,
        )
        with pytest.raises(ValueError, match='event_intervals_per_session'):
            audit_predictor_timescales(
                processed_beh_dict=beh,
                mouse_names_dict=names,
                target_idx=0,
                predictor_idx=1,
                configured_filter_history=1.0,
                camera_fps_dict=cam,
                max_lag_seconds=2.0,
                n_shuffles=4,
                ibi_thresholds={'male': 0.25, 'female': 0.30},
                save_path=str(tmp_path / 'x.pkl'),
                source_pickle='input.pkl',
                event_intervals_per_session=None,
                onset_times_per_session=bouts,
            )

    def test_empty_input_payload(self, tmp_path):
        """An empty ``processed_beh_dict`` yields the documented empty
        timescale payload (no features, NaN IBI percentiles, zero
        sessions) and still writes an artifact."""

        save_path = tmp_path / 'empty_ts.pkl'
        payload = audit_predictor_timescales(
            processed_beh_dict={},
            mouse_names_dict={},
            target_idx=0,
            predictor_idx=1,
            configured_filter_history=1.0,
            camera_fps_dict={},
            max_lag_seconds=2.0,
            n_shuffles=4,
            ibi_thresholds={'male': 0.25, 'female': 0.30},
            save_path=str(save_path),
            source_pickle='input.pkl',
            event_intervals_per_session={},
            onset_times_per_session={},
        )
        assert payload['features'] == []
        assert payload['n_sessions'] == 0
        assert np.isnan(payload['ibi_empirical_pcts']['p90'])
        assert save_path.exists()

    def test_empty_input_embeds_metadata(self, tmp_path):
        """The empty-input early-return path still embeds a supplied
        ``input_metadata`` block under the reserved key for provenance."""

        save_path = tmp_path / 'empty_meta.pkl'
        payload = audit_predictor_timescales(
            processed_beh_dict={},
            mouse_names_dict={},
            target_idx=0,
            predictor_idx=1,
            configured_filter_history=1.0,
            camera_fps_dict={},
            max_lag_seconds=2.0,
            n_shuffles=4,
            ibi_thresholds={'male': 0.25, 'female': 0.30},
            save_path=str(save_path),
            source_pickle='input.pkl',
            input_metadata={'pipeline': 'manifold'},
            event_intervals_per_session={},
            onset_times_per_session={},
        )
        assert payload['_input_metadata'] == {'pipeline': 'manifold'}

    def test_valid_sessions_without_ibi_intervals(self, tmp_path):
        """Sessions can carry bout onsets (so signal correlation runs) yet
        supply no per-USV intervals; the IBI percentiles then fall back to
        NaN while bouts are still counted, and a constant feature exercises
        the zero-denominator skip in the cross-correlation loop."""

        fps = 10.0
        n_frames = 800
        beh, names, cam, bouts, _, events = _make_timescale_inputs(
            n_frames=n_frames, fps=fps, n_sessions=2,
        )
        # Make one feature constant in a single session so its
        # cross-correlation denom <= 0 there (the zero-denom skip), while
        # the feature stays finite in the other session so the headline's
        # per-feature nanargmax has a valid lag to report.
        beh['s0'] = beh['s0'].with_columns(pl.lit(0.0).alias('m1.speed'))
        save_path = tmp_path / 'no_ibi.pkl'
        payload = audit_predictor_timescales(
            processed_beh_dict=beh,
            mouse_names_dict=names,
            target_idx=0,
            predictor_idx=1,
            configured_filter_history=1.0,
            camera_fps_dict=cam,
            max_lag_seconds=2.0,
            n_shuffles=6,
            ibi_thresholds={'male': 0.25, 'female': 0.30},
            save_path=str(save_path),
            source_pickle='input.pkl',
            shuffle_range_seconds=(3.0, 5.0),
            event_intervals_per_session={},  # required key, no intervals
            onset_times_per_session=bouts,
        )
        assert payload['n_bouts'] > 0
        assert payload['n_usvs'] == 0
        assert np.isnan(payload['ibi_empirical_pcts']['p50'])
        # The feature is constant in only one session, so its cohort-mean
        # row is finite (the all-session-constant case crashes the headline
        # nanargmax — a separate source-level limitation, not exercised here).
        const_idx = payload['features'].index('other.speed')
        assert np.any(np.isfinite(payload['rho_signal'][const_idx]))

    def test_session_missing_names_skipped_and_mixed_fps(self, tmp_path):
        """A session absent from ``mouse_names_dict`` is skipped during the
        generic-naming pass, and a cohort with heterogeneous fps still runs
        (reporting the lag axis at one fps), exercising the name-skip and
        mixed-fps branches."""

        beh, names, cam, bouts, intervals, events = _make_timescale_inputs(
            n_frames=800, fps=10.0, n_sessions=2,
        )
        # Add a third session that has feature data but no mouse-names entry.
        beh['s_extra'] = _make_session_df(n_frames=800, seed=200)
        cam['s_extra'] = 10.0
        # Perturb one session's fps so the cohort is heterogeneous.
        cam['s1'] = 12.0
        payload = audit_predictor_timescales(
            processed_beh_dict=beh,
            mouse_names_dict=names,  # no 's_extra' key
            target_idx=0,
            predictor_idx=1,
            configured_filter_history=1.0,
            camera_fps_dict=cam,
            max_lag_seconds=2.0,
            n_shuffles=4,
            ibi_thresholds={'male': 0.25, 'female': 0.30},
            save_path=str(tmp_path / 'mixed.pkl'),
            source_pickle='input.pkl',
            shuffle_range_seconds=(3.0, 5.0),
            event_intervals_per_session=intervals,
            onset_times_per_session=bouts,
        )
        # 's_extra' contributed no generic features (no names) -> only the
        # two named sessions populate session_blocks.
        assert payload['n_sessions'] == 2
        assert len(payload['features']) == 4

    def test_feature_absent_from_some_sessions(self, tmp_path):
        """A feature present in only one session leaves its missing-session
        ACF/cross-correlation slabs as NaN (exercising the
        feature-not-in-session skips) while still producing a finite
        summary for the session that has it."""

        fps = 10.0
        beh, names, cam, bouts, intervals, events = _make_timescale_inputs(
            n_frames=800, fps=fps, n_sessions=2,
        )
        # Drop the dyadic feature from one session only.
        beh['s1'] = beh['s1'].drop('nose-nose')
        payload = audit_predictor_timescales(
            processed_beh_dict=beh,
            mouse_names_dict=names,
            target_idx=0,
            predictor_idx=1,
            configured_filter_history=1.0,
            camera_fps_dict=cam,
            max_lag_seconds=2.0,
            n_shuffles=6,
            ibi_thresholds={'male': 0.25, 'female': 0.30},
            save_path=str(tmp_path / 'absent.pkl'),
            source_pickle='input.pkl',
            shuffle_range_seconds=(3.0, 5.0),
            event_intervals_per_session=intervals,
            onset_times_per_session=bouts,
        )
        assert 'nose-nose' in payload['features']
        idx = payload['features'].index('nose-nose')
        # acf[0] from the one session that has it is still 1.
        assert payload['acf_median'][idx, 0] == pytest.approx(1.0, abs=1e-5)

    def test_featureless_session_skipped_in_valid_loop(self, tmp_path):
        """A session whose DataFrame holds only digit-suffixed columns
        contributes an empty feature block; even with bout onsets present
        it is skipped by the ``not per_feature`` guard in the
        valid-session selection, while the real session still computes."""

        fps = 10.0
        n_frames = 800
        beh, names, cam, bouts, intervals, events = _make_timescale_inputs(
            n_frames=n_frames, fps=fps, n_sessions=1,
        )
        # Featureless session: only a digit-suffixed embedding column.
        beh['s_blank'] = pl.DataFrame({
            'm0.embedding.0': np.zeros(n_frames, dtype=np.float64),
        })
        names['s_blank'] = ['m0', 'm1']
        cam['s_blank'] = fps
        bouts['s_blank'] = bouts['s0']
        intervals['s_blank'] = intervals['s0']
        events['s_blank'] = events['s0']
        payload = audit_predictor_timescales(
            processed_beh_dict=beh,
            mouse_names_dict=names,
            target_idx=0,
            predictor_idx=1,
            configured_filter_history=1.0,
            camera_fps_dict=cam,
            max_lag_seconds=2.0,
            n_shuffles=4,
            ibi_thresholds={'male': 0.25, 'female': 0.30},
            save_path=str(tmp_path / 'blank.pkl'),
            source_pickle='input.pkl',
            shuffle_range_seconds=(3.0, 5.0),
            event_intervals_per_session=intervals,
            onset_times_per_session=bouts,
        )
        # 's_blank' adds no features; the real session's 4 features remain.
        assert len(payload['features']) == 4
        assert payload['n_sessions'] == 2

    def test_no_valid_sessions_with_bout_onsets(self, tmp_path):
        """When features exist but no session carries bout onsets, the
        signal-correlation block falls back to all-NaN arrays of the
        correct shape while the ACF block is still computed."""

        fps = 10.0
        beh, names, cam, _, _, events = _make_timescale_inputs(
            n_frames=400, fps=fps, n_sessions=2,
        )
        save_path = tmp_path / 'no_valid.pkl'
        payload = audit_predictor_timescales(
            processed_beh_dict=beh,
            mouse_names_dict=names,
            target_idx=0,
            predictor_idx=1,
            configured_filter_history=1.0,
            camera_fps_dict=cam,
            max_lag_seconds=2.0,
            n_shuffles=4,
            ibi_thresholds={'male': 0.25, 'female': 0.30},
            save_path=str(save_path),
            source_pickle='input.pkl',
            event_intervals_per_session={},  # required but empty
            onset_times_per_session={},  # no session has Y
        )
        n_feat = len(payload['features'])
        max_lag_frames = int(np.ceil(2.0 * fps))
        n_signal_lags = 2 * max_lag_frames + 1
        assert n_feat == 4
        assert payload['rho_signal'].shape == (n_feat, n_signal_lags)
        assert np.all(np.isnan(payload['rho_signal']))
        assert payload['n_bouts'] == 0
        # ACF block is still real.
        assert np.isfinite(payload['acf_median'][:, 0]).any()
