"""
@author: bartulem
Unit tests for ``usv_playpen.modeling.load_input_files``: the pure temporal
helpers and the on-disk ``find_*`` event loaders.

The pure helpers carry the load-bearing temporal logic and are checked against
closed forms and exact values; the ``find_*`` loaders are driven on tiny
synthetic session trees written under ``tmp_path`` in the exact on-disk layout
the production code reads (USV summary CSV, behavioural-feature CSV, tracking
file), so their event sets, loop control and console messages are asserted
without real recordings.

Coverage:

* the IBI-threshold formulas (Gaussian-mixture and inverse-Gaussian), the
  USV-timestamp-to-trace renderer, the vocal-class traces, occupancy
  rasterisation and smoothing edges;
* the "clean" (no-USV) epoch tiler, including the guarantee that every tiled
  epoch's history window avoids the forbidden zone around each USV, and the
  history-is-dirty check;
* the USV summary-table reader, the pickle loader and the behavioural-feature
  loader (including its skip paths);
* vocalization-to-bout grouping and the bout-offset pairing rules;
* ``find_onset_epochs``, ``find_usv_categories``, ``find_variable_length_bouts``
  and the bout-offset epochs: exact event sets, sex-specific thresholds, loop
  control and messages;
* the USV-category label requirement and the vocal-setting messages.
"""

from __future__ import annotations

import json
import pickle
import re
from typing import ClassVar

import h5py
import numpy as np
import polars as pls
import pytest
from scipy import integrate
from scipy.stats import norm

from tests.modeling._synth import _SETTINGS_JSON
from usv_playpen.modeling.load_input_files import (
    _bout_offset_events,
    _build_binary_occupancy,
    _calculate_ibi_threshold,
    _calculate_ibi_threshold_ig,
    _generate_vocal_trace,
    _get_clean_tiled_epochs,
    _group_vocalizations_into_bouts,
    _history_is_dirty,
    _smooth_occupancy,
    build_vocal_predictor_signals,
    find_onset_epochs,
    find_usv_categories,
    find_variable_length_bouts,
    is_pooled_vocal_signal,
    is_vocal_signal,
    load_behavioral_feature_data,
    load_pickle_modeling_data,
    mixture_component_count,
    mixture_ibi_threshold,
    onset_target_tag,
    read_usv_summary_table,
    require_labels_for_vocal_predictors,
    require_usv_category_column,
    validate_vocal_class_settings,
    vocal_trace_name,
)


def test_read_usv_summary_table_infers_types_from_every_row_and_warns_on_no_emitter(tmp_path, capsys):
    """A category label empty in the first 100 rows is still read as an integer (default
    100-row inference would give String, and ``col == k`` would raise), usv_id keeps its
    padding, and an all-empty emitter column prints a warning naming vcl-assign."""
    n_rows = 150
    summary_path = tmp_path / 's_usv_summary.csv'
    pls.DataFrame({
        'usv_id': [f"{i:06d}" for i in range(n_rows)],
        'start': np.arange(n_rows, dtype=float),
        'emitter': pls.Series([None] * n_rows, dtype=pls.String),
        'qlvm_category': [None] * 120 + [2] * 30,
    }).write_csv(summary_path)
    table = read_usv_summary_table(summary_path, ',')
    assert table['qlvm_category'].dtype == pls.Int64
    assert table.filter(pls.col('qlvm_category') == 2).height == 30
    assert table['usv_id'][0] == '000000'
    assert 'no assigned emitter' in capsys.readouterr().out

# The message every label-dependent path raises when its label column is null or absent.
_LABELS_UNAVAILABLE = 'QLVM category labels are not available'


# Shared synthetic-session helpers for the on-disk ``find_*`` loaders.


def _write_usv_summary(session_root, rows: dict, csv_sep: str = ',') -> None:
    """
    Description
    -----------
    Writes a synthetic ``*_usv_summary.csv`` under ``<session_root>/audio`` so
    the disk-reading loaders (``find_onset_epochs``, ``find_usv_categories``,
    ``find_variable_length_bouts``) can ingest it through their polars
    ``read_csv`` glob (``audio/**/*_usv_summary.csv``).

    Parameters
    ----------
    session_root (pathlib.Path)
        The session ROOT directory; the ``audio`` subdir is created if absent.
    rows (dict)
        Column-name -> value-list mapping passed straight to
        ``polars.DataFrame``; must include at least the ``emitter``, ``start``
        and ``stop`` columns the loaders rely on. An all-False ``noise`` column and
        pure-USV ``usv`` (all True) / ``squeak`` (all False) columns are added when the
        caller does not supply them, since the loaders filter on them and raise when any
        is missing, and so is the squeak extent ``squeak_start`` / ``squeak_end``: the
        segment's own ``start`` / ``stop`` on every ``squeak`` row (so the squeak times
        equal the segment times unless a test sets them) and null elsewhere.
    csv_sep (str)
        Field separator written into the CSV (mirrors the loaders' ``csv_sep``).

    Returns
    -------
    None
    """

    audio_dir = session_root / 'audio'
    audio_dir.mkdir(parents=True, exist_ok=True)
    csv_path = audio_dir / f'{session_root.name}_usv_summary.csv'
    if 'noise' not in rows:
        rows = {**rows, 'noise': [False] * len(next(iter(rows.values())))}
    if 'usv' not in rows and 'squeak' not in rows:
        n_rows = len(next(iter(rows.values())))
        rows = {**rows, 'usv': [True] * n_rows, 'squeak': [False] * n_rows}
    if 'squeak_start' not in rows and 'squeak_end' not in rows:
        bearing = [flag is True for flag in rows['squeak']]
        rows = {
            **rows,
            'squeak_start': [start if flag else None for start, flag in zip(rows['start'], bearing, strict=True)],
            'squeak_end': [stop if flag else None for stop, flag in zip(rows['stop'], bearing, strict=True)],
        }
    pls.DataFrame(rows).write_csv(file=csv_path, separator=csv_sep)


def _mixture_model_params() -> dict:
    """
    Description
    -----------
    Returns a minimal sex-specific mixture-model parameter dict of the shape the loaders
    require (``{'male': {'means': [...], 'sds': [...]}, 'female': {...}}``).

    The log-space mean/sd are chosen so the resulting inter-bout-interval (IBI)
    threshold ``exp(mean + z * sd)`` is roughly one second, which cleanly
    separates the closely-spaced (in-bout) and widely-spaced (between-bout)
    synthetic USVs used by the tests.

    Parameters
    ----------

    Returns
    -------
    params (dict)
        A ``{'male': ..., 'female': ...}`` mixture-model parameter dictionary.
    """

    return {'male': {'means': [0.0], 'sds': [0.0]},
            'female': {'means': [0.0], 'sds': [0.0]}}


def _features_df(n_frames: int):
    """
    Description
    -----------
    Builds a tiny polars DataFrame standing in for a session's behavioral
    feature table; only its row count (``shape[0]``) is consumed by the loaders
    to derive the session duration in frames.

    Parameters
    ----------
    n_frames (int)
        Desired number of frames (rows) in the synthetic feature table.

    Returns
    -------
    features (polars.DataFrame)
        A single-column DataFrame with ``n_frames`` rows.
    """

    return pls.DataFrame({'frame': np.arange(n_frames)})


# _calculate_ibi_threshold


class TestCalculateIbiThreshold:
    """The log-normal IBI cutoff is the closed form ``exp(mu + z * sd)`` and grows with z, since
    every bout grouping downstream inherits this threshold."""

    def test_log_normal_formula(self):
        """The threshold is ``exp(mu + z * sd)``."""

        assert _calculate_ibi_threshold(0.0, 1.0, 2.58) == pytest.approx(np.exp(2.58))
        assert _calculate_ibi_threshold(1.0, 0.0, 2.58) == pytest.approx(np.exp(1.0))

    def test_monotone_increasing_in_z(self):
        """A larger z-score yields a larger (more permissive) cutoff."""

        lo = _calculate_ibi_threshold(0.0, 0.5, 1.0)
        hi = _calculate_ibi_threshold(0.0, 0.5, 3.0)
        assert hi > lo


class TestCalculateIbiThresholdIg:
    """The inverse-Gaussian IBI cutoff is the IG quantile of the z tail, checked against an
    independently integrated density so a parametrisation slip in the src cannot pass."""

    @staticmethod
    def _ig_cdf(x: float, mu: float, lam: float) -> float:
        """
        Description
        -----------
        Evaluates the inverse-Gaussian CDF at ``x`` by numerically integrating the
        density ``sqrt(lam / (2 pi t^3)) exp(-lam (t - mu)^2 / (2 mu^2 t))`` written
        out here, independent of scipy's parametrization.

        Parameters
        ----------
        x (float)
            Upper integration limit (seconds).
        mu (float)
            Inverse-Gaussian mean.
        lam (float)
            Inverse-Gaussian shape parameter.

        Returns
        -------
        cdf (float)
            ``P(T <= x)`` under the inverse-Gaussian distribution.
        """

        def density(t):
            """
            Description
            -----------
            The inverse-Gaussian density at ``t`` for the enclosing ``mu`` and ``lam``.

            Parameters
            ----------
            t (float)
                Evaluation point (seconds, > 0).

            Returns
            -------
            pdf (float)
                The density value at ``t``.
            """
            return np.sqrt(lam / (2.0 * np.pi * t ** 3)) * np.exp(-lam * (t - mu) ** 2 / (2.0 * mu ** 2 * t))

        return integrate.quad(density, 0.0, x, limit=200)[0]

    @pytest.mark.parametrize(("mu", "lam", "z"), [(0.08, 0.3, 2.58), (0.5, 2.0, 1.0), (0.02, 0.01, 1.96)])
    def test_threshold_is_the_ig_quantile_of_the_z_tail(self, mu, lam, z):
        """The threshold is the quantile ``Phi(z)`` of IG(mean ``mu``, shape ``lam``):
        the IG CDF at the threshold equals ``Phi(z)`` (so mean and shape are not
        swapped and the quantile level is the upper-tail mapping of ``z``)."""

        threshold = _calculate_ibi_threshold_ig(mu, lam, z)
        assert self._ig_cdf(threshold, mu, lam) == pytest.approx(norm.cdf(z), abs=1e-6)

    def test_monotone_in_z_and_mean(self):
        """A larger z-score or a larger component mean gives a longer cutoff."""

        assert _calculate_ibi_threshold_ig(0.1, 0.5, 3.0) > _calculate_ibi_threshold_ig(0.1, 0.5, 1.0)
        assert _calculate_ibi_threshold_ig(0.2, 0.5, 2.58) > _calculate_ibi_threshold_ig(0.1, 0.5, 2.58)


class TestMixtureIbiThreshold:
    """One helper reads either mixture-block form, so every reader of ``mixture_model_params``
    gives the same threshold the loaders use."""

    LOGNORMAL: ClassVar[dict] = {'means': [-2.5, -1.0], 'sds': [0.3, 0.4]}
    INVERSE_GAUSSIAN: ClassVar[dict] = {'model_class': 'ig', 'mus': [0.1, 0.4], 'lambdas': [1.0, 2.0]}

    def test_component_count_reads_the_block_form(self):
        """A log-normal block counts ``means``, an inverse-Gaussian block counts ``mus``."""

        assert mixture_component_count(self.LOGNORMAL) == 2
        assert mixture_component_count({'model_class': 'ig', 'mus': [0.1], 'lambdas': [1.0]}) == 1

    def test_lognormal_block_matches_the_lognormal_threshold(self):
        """A log-normal block gives exactly ``exp(mean + z * sd)`` of the chosen component."""

        assert mixture_ibi_threshold(self.LOGNORMAL, 1, 2.58) == pytest.approx(
            float(_calculate_ibi_threshold(-1.0, 0.4, 2.58)))

    def test_inverse_gaussian_block_matches_the_ig_threshold(self):
        """An inverse-Gaussian block gives the ``Phi(z)`` quantile of the chosen component,
        where reading ``means`` would raise KeyError."""

        assert mixture_ibi_threshold(self.INVERSE_GAUSSIAN, 1, 2.58) == pytest.approx(
            _calculate_ibi_threshold_ig(0.4, 2.0, 2.58))

    @pytest.mark.parametrize('index', [-1, 2])
    def test_an_out_of_range_index_raises(self, index):
        """A negative index would wrap to the last component, and one past the end has no component."""

        for block in (self.LOGNORMAL, self.INVERSE_GAUSSIAN):
            with pytest.raises(ValueError, match="Invalid mixture_model_component_index"):
                mixture_ibi_threshold(block, index, 2.58)


# _generate_vocal_trace


class TestGenerateVocalTrace:
    """The USV-timestamp renderer fills the right frame spans (binary) and keeps the trace
    shape when smoothed, since every vocal predictor is built from this trace."""

    def test_binary_interval_fill(self):
        """Without smoothing the trace is binary and 1 across the
        floor(start)..ceil(stop) frame span."""

        trace = _generate_vocal_trace(np.array([0.0]), np.array([0.5]),
                                      duration_frames=20, fps=10.0)
        assert trace.shape == (20,)
        assert set(np.unique(trace)).issubset({0.0, 1.0})
        assert trace[:5].sum() == 5.0
        assert trace[5:].sum() == 0.0

    def test_subframe_usv_marks_one_frame(self):
        """A sub-frame USV still lights up a single frame."""

        trace = _generate_vocal_trace(np.array([0.0]), np.array([0.0]),
                                      duration_frames=10, fps=10.0)
        assert trace[0] == 1.0
        assert trace.sum() == 1.0

    def test_out_of_range_clipped(self):
        """Indices beyond the session are clipped, never raising."""

        trace = _generate_vocal_trace(np.array([100.0]), np.array([101.0]),
                                      duration_frames=10, fps=10.0)
        assert trace.shape == (10,)
        assert not trace.any()

    @pytest.mark.filterwarnings("ignore::astropy.utils.exceptions.AstropyUserWarning")
    def test_smoothing_spreads_mass_below_unit_peak(self):
        """With ``smooth_sd > 0`` the trace becomes a smoothed density:
        the unit spike is spread out so the peak drops below 1 while the
        shape is preserved."""

        trace = _generate_vocal_trace(np.array([1.0]), np.array([1.05]),
                                      duration_frames=40, fps=10.0, smooth_sd=2.0)
        assert trace.shape == (40,)
        assert 0.0 < trace.max() < 1.0
        # Mass is spread to neighbouring frames around the original spike.
        assert np.count_nonzero(trace > 1e-6) > 1


# _get_clean_tiled_epochs


class TestGetCleanTiledEpochs:
    """The clean (no-USV) tiler covers the silent session and never lets a window's history
    overlap the forbidden zone around a USV, which is what makes the negatives clean."""

    def test_no_usv_tiles_whole_session(self):
        """With no USVs the entire session is tiled into window end-times,
        the first window starting at the session start."""

        ts = _get_clean_tiled_epochs(np.array([]), np.array([]),
                                     filter_history=1.0, session_duration_sec=10.0)
        # First end-time is filter_history == 1.0: the window [0, 1] fits.
        assert ts.min() == pytest.approx(1.0)
        assert ts.max() <= 10.0 + 1e-9
        # Regular 1.0-spaced tiling.
        assert np.allclose(np.diff(ts), 1.0)

    def test_windows_avoid_forbidden_zone(self):
        """Every returned end-time ``t`` defines a history window
        ``[t - filter_history, t]`` that must not overlap the USV
        ``[start, stop]`` (abutting it at either edge is allowed, exactly as
        for a positive onset under ``_history_is_dirty``)."""

        fh = 1.0
        usv_start, usv_stop = 50.0, 50.5
        ts = _get_clean_tiled_epochs(np.array([usv_start]), np.array([usv_stop]),
                                     filter_history=fh, session_duration_sec=100.0)
        for t in ts:
            win_lo, win_hi = t - fh, t
            # Window is entirely before or entirely after the USV.
            assert win_hi <= usv_start + 1e-9 or win_lo >= usv_stop - 1e-9, (
                f"window [{win_lo}, {win_hi}] overlaps the USV "
                f"[{usv_start}, {usv_stop}]"
            )
        # The first window after the USV starts right where it stopped.
        assert usv_stop + fh in ts.tolist()

    def test_short_clean_zone_still_hosts_its_first_window_end(self):
        """A clean zone of window ENDS shorter than ``filter_history`` still
        holds the end at its start: the window ending there has a full
        ``filter_history`` of silence behind it."""

        # Two USVs at 1.0-1.1 and 1.4-1.5 s (fh 1 s, 2.6 s session): the zone
        # [1.0, 1.0] gives the window [0, 1] that ends where the first USV
        # starts, and the trailing zone [2.5, 2.6] gives the window [1.5, 2.5]
        # that starts where the last USV stops.
        ts = _get_clean_tiled_epochs(np.array([1.0, 1.4]), np.array([1.1, 1.5]),
                                     filter_history=1.0, session_duration_sec=2.6)
        np.testing.assert_allclose(ts, [1.0, 2.5])


# load_pickle_modeling_data


class TestLoadPickleModelingData:
    """The modeling pickle loader returns the stored dict unchanged."""

    def test_round_trip(self, tmp_path):
        """A pickled dict is returned verbatim."""

        payload = {'feat': np.arange(5), 'meta': {'x': 1}}
        p = tmp_path / 'modeling.pkl'
        with p.open('wb') as fh:
            pickle.dump(payload, fh)
        out = load_pickle_modeling_data(str(p))
        assert out['meta'] == {'x': 1}
        np.testing.assert_array_equal(out['feat'], np.arange(5))


# load_behavioral_feature_data


class TestLoadBehavioralFeatureData:
    """The behavioural-feature loader reads the features CSV, the recording frame rate and the
    track names from the session's ``video`` subtree."""

    def test_reads_features_fps_and_track_names(self, tmp_path):
        """A session whose ``video`` subtree holds the behavioral-features CSV
        and the metric track H5 yields the per-session feature DataFrame, the
        recording frame rate (decoded from the H5 scalar) and the decoded
        track-name list."""

        sess = tmp_path / 'sess_A'
        video_dir = sess / 'video'
        video_dir.mkdir(parents=True)

        features_csv = video_dir / '20230101000000_points3d_translated_rotated_metric_behavioral_features.csv'
        pls.DataFrame({'speed': [0.1, 0.2, 0.3], 'angle': [1.0, 2.0, 3.0]}).write_csv(file=features_csv)

        track_h5 = video_dir / '20230101000000_points3d_translated_rotated_metric.h5'
        with h5py.File(name=track_h5, mode='w') as h5f:
            h5f['recording_frame_rate'] = 150.0
            h5f['track_names'] = np.array([b'male', b'female'])

        beh, fps, names = load_behavioral_feature_data(behavior_file_paths=[str(sess)])

        assert list(beh.keys()) == ['sess_A']
        assert beh['sess_A'].shape == (3, 2)
        assert fps['sess_A'] == pytest.approx(150.0)
        assert names['sess_A'] == ['male', 'female']


# find_onset_epochs


class TestFindBoutEpochs:
    """``find_onset_epochs`` on synthetic session trees: the bout / individual / state modes
    and their event sets, vocal output types, noise filtering, target-category and
    target-class selection, label requirements and the skip / error paths."""

    def _build(self, tmp_path, rows: dict, n_frames: int = 600, fps: float = 100.0):
        """
        Description
        -----------
        Builds a one-session tree (``sess_B``) holding a synthetic USV summary and
        returns the common loader kwargs for the ``find_onset_epochs`` tests below.

        Parameters
        ----------
        tmp_path (pathlib.Path)
            The test's temporary directory; the session root is created under it.
        rows (dict)
            Column-name -> value-list mapping for the USV summary, forwarded to
            ``_write_usv_summary``.
        n_frames (int)
            Number of frames in the synthetic behavioural-feature table.
        fps (float)
            Camera frame rate recorded for the session.

        Returns
        -------
        kwargs (dict)
            ``root_directories``, ``mouse_ids_dict``, ``camera_fps_dict`` and
            ``features_dict`` keyed for the single session.
        """

        sess = tmp_path / 'sess_B'
        _write_usv_summary(sess, rows)
        return {
            'root_directories': [str(sess)],
            'mouse_ids_dict': {'sess_B': ['male', 'female']},
            'camera_fps_dict': {'sess_B': fps},
            'features_dict': {'sess_B': _features_df(n_frames)},
        }

    def test_bout_mode_positive_and_negative_events(self, tmp_path):
        """In 'bout_onset' mode a clean cluster of >= ``min_usv_per_bout`` syllables
        becomes one positive onset, and the silent tail tiles into negative
        (no-USV) onsets."""

        rows = {
            'emitter': ['male', 'male', 'male'],
            'start': [2.0, 2.1, 2.2],
            'stop': [2.05, 2.15, 2.25],
            'usv_category': [1, 1, 1],
        }
        kwargs = self._build(tmp_path, rows)
        out = find_onset_epochs(prediction_mode='bout_onset', filter_history=1.0,
                               usv_bout_time=0.5, min_usv_per_bout=2,
                               proportion_smoothing_sd=None, mixture_model_params=_mixture_model_params(),
                               **kwargs)

        male = out['sess_B']['male']
        # The single clean bout starts at 2.0 (> filter_history, no prior USV).
        np.testing.assert_allclose(male['positive_events'], [2.0])
        # Negative events exist and none of their 1 s windows overlaps the bout.
        assert male['negative_events'].size > 0
        negatives = male['negative_events']
        assert np.all((negatives <= rows['start'][0]) | (negatives - 1.0 >= rows['stop'][-1]))
        # Female (no USVs) has empty positive events but tiled negatives.
        assert out['sess_B']['female']['positive_events'].size == 0
        assert out['sess_B']['female']['negative_events'].size > 0

    def test_bout_mode_rejects_small_and_early_bouts(self, tmp_path):
        """A bout below ``min_usv_per_bout`` or starting at/under
        ``filter_history`` yields no positive events."""

        rows = {
            'emitter': ['male', 'male'],
            'start': [0.5, 2.0],
            'stop': [0.55, 2.05],
            'usv_category': [1, 1],
        }
        kwargs = self._build(tmp_path, rows)
        out = find_onset_epochs(prediction_mode='bout_onset', filter_history=1.0,
                               usv_bout_time=0.5, min_usv_per_bout=3,
                               proportion_smoothing_sd=None, mixture_model_params=_mixture_model_params(),
                               **kwargs)
        assert out['sess_B']['male']['positive_events'].size == 0

    def test_individual_mode_positive_is_filtered_onsets(self, tmp_path):
        """In 'individual' mode every USV onset past ``filter_history`` is a
        positive event; negatives are the tiled clean epochs."""

        rows = {
            'emitter': ['male', 'male', 'male'],
            'start': [0.5, 2.0, 3.0],
            'stop': [0.55, 2.05, 3.05],
            'usv_category': [1, 2, 1],
        }
        kwargs = self._build(tmp_path, rows)
        out = find_onset_epochs(prediction_mode='individual', filter_history=1.0,
                               usv_bout_time=0.5, min_usv_per_bout=2,
                               proportion_smoothing_sd=None, mixture_model_params=_mixture_model_params(),
                               **kwargs)
        male = out['sess_B']['male']
        # 0.5 is dropped (<= filter_history); 2.0 and 3.0 survive.
        np.testing.assert_allclose(male['positive_events'], [2.0, 3.0])
        assert male['negative_events'].size > 0

    def test_individual_mode_negatives_keep_the_future_window_silent(self, tmp_path):
        """In 'individual' mode a negative (no-USV) onset must be followed by
        ``usv_bout_time`` of silence, as in 'bout_onset' mode: the tile at 4.6 s
        has a clean history but a USV starts 20 ms later, so it is dropped while
        the tile at 3.6 s (next USV a full second away) survives."""

        rows = {
            'emitter': ['male', 'male'],
            'start': [1.5, 4.62],
            'stop': [1.6, 4.7],
            'usv_category': [1, 1],
        }
        kwargs = self._build(tmp_path, rows)
        out = find_onset_epochs(prediction_mode='individual', filter_history=1.0,
                               usv_bout_time=0.5, min_usv_per_bout=2,
                               proportion_smoothing_sd=None, mixture_model_params=_mixture_model_params(),
                               **kwargs)
        male = out['sess_B']['male']
        np.testing.assert_allclose(male['positive_events'], [1.5, 4.62])
        # Window ends tile each clean zone from its start: [1.0, 1.5] -> 1.0,
        # [2.6, 4.62] -> 2.6, 3.6, 4.6 (dropped) and [5.7, 6.0] -> 5.7.
        np.testing.assert_allclose(male['negative_events'], [1.0, 2.6, 3.6, 5.7])

    def test_state_mode_partitions_grid_by_occupancy(self, tmp_path):
        """In 'state' mode the regular time grid is split into vocalizing
        (positive) and silent (negative) onsets by per-frame occupancy."""

        rows = {
            'emitter': ['male'],
            'start': [2.0],
            'stop': [2.9],
            'usv_category': [1],
        }
        kwargs = self._build(tmp_path, rows, n_frames=600, fps=100.0)
        out = find_onset_epochs(prediction_mode='state', filter_history=1.0,
                               usv_bout_time=0.5, min_usv_per_bout=2,
                               proportion_smoothing_sd=None, mixture_model_params=_mixture_model_params(),
                               **kwargs)
        male = out['sess_B']['male']
        # Grid is at 1.0 spacing; the t=2.0 grid point falls inside the USV.
        assert 2.0 in male['positive_events']
        assert male['negative_events'].size > 0

    @pytest.mark.filterwarnings("ignore::astropy.utils.exceptions.AstropyUserWarning")
    def test_vocal_output_all_rate_emits_pooled_and_category_signals(self, tmp_path):
        """``vocal_output_type='all_rate'`` writes both the pooled 'usv_rate'
        trace and one 'usv_cat_X' trace per observed category."""

        rows = {
            'emitter': ['male', 'male'],
            'start': [2.0, 3.0],
            'stop': [2.05, 3.05],
            'usv_category': [1, 4],
        }
        kwargs = self._build(tmp_path, rows)
        out = find_onset_epochs(prediction_mode='bout_onset', filter_history=1.0,
                               usv_bout_time=0.5, min_usv_per_bout=2,
                               proportion_smoothing_sd=2.0, mixture_model_params=_mixture_model_params(),
                               vocal_output_type='all_rate',
                               **kwargs)
        signals = out['sess_B']['male']['continuous_vocal_signals']
        assert 'usv_rate' in signals
        assert 'usv_cat_1' in signals
        assert 'usv_cat_4' in signals
        assert signals['usv_rate'].shape == (600,)

    def test_pooled_binary_output(self, tmp_path):
        """``vocal_output_type='pooled_binary'`` writes a binary 'usv_event'
        trace and no per-category traces."""

        rows = {
            'emitter': ['male', 'male'],
            'start': [2.0, 2.1],
            'stop': [2.05, 2.15],
            'usv_category': [1, 1],
        }
        kwargs = self._build(tmp_path, rows)
        out = find_onset_epochs(prediction_mode='bout_onset', filter_history=1.0,
                               usv_bout_time=0.5, min_usv_per_bout=2,
                               proportion_smoothing_sd=None, mixture_model_params=_mixture_model_params(),
                               vocal_output_type='pooled_binary',
                               **kwargs)
        signals = out['sess_B']['male']['continuous_vocal_signals']
        assert 'usv_event' in signals
        assert set(np.unique(signals['usv_event'])).issubset({0.0, 1.0})

    def test_noise_flagged_rows_filtered_out(self, tmp_path):
        """Rows the noise classifier flagged are dropped before any sampling."""

        rows = {
            'emitter': ['male', 'male', 'male'],
            'start': [2.0, 2.1, 2.2],
            'stop': [2.05, 2.15, 2.25],
            'usv_category': [1, 1, 1],
            'noise': [True, True, True],
        }
        kwargs = self._build(tmp_path, rows)
        out = find_onset_epochs(prediction_mode='individual', filter_history=1.0,
                               usv_bout_time=0.5, min_usv_per_bout=2,
                               proportion_smoothing_sd=None, mixture_model_params=_mixture_model_params(),
                               exclude_noise_usvs=True,
                               **kwargs)
        # All male USVs were noise -> no positive events, empty start array.
        assert out['sess_B']['male']['start'].size == 0
        assert out['sess_B']['male']['positive_events'].size == 0

    def test_target_category_filters_positives_individual(self, tmp_path):
        """In 'individual' mode, ``target_category`` restricts the positive
        onset events to USVs of that category; onsets of other categories are
        excluded (the broadband-vocalization use case)."""

        rows = {
            'emitter': ['male', 'male', 'male'],
            'start': [2.0, 3.0, 4.0],
            'stop': [2.05, 3.05, 4.05],
            'usv_category': [3, 2, 3],
        }
        kwargs = self._build(tmp_path, rows)
        out = find_onset_epochs(prediction_mode='individual', filter_history=1.0,
                               usv_bout_time=0.5, min_usv_per_bout=2,
                               proportion_smoothing_sd=None, mixture_model_params=_mixture_model_params(),
                               target_category=3,
                               **kwargs)
        male = out['sess_B']['male']
        # Only the category-3 onsets (2.0, 4.0) survive; the cat-2 onset (3.0) is gone.
        np.testing.assert_allclose(male['positive_events'], [2.0, 4.0])

    def test_target_category_negatives_unchanged(self, tmp_path):
        """Negative (silent-epoch) events are sampled over ALL USVs and so are
        identical whether or not a ``target_category`` filter is applied."""

        rows = {
            'emitter': ['male', 'male', 'male'],
            'start': [2.0, 3.0, 4.0],
            'stop': [2.05, 3.05, 4.05],
            'usv_category': [3, 2, 3],
        }
        kwargs = self._build(tmp_path, rows)
        common = {'prediction_mode': 'individual', 'filter_history': 1.0,
                      'usv_bout_time': 0.5, 'min_usv_per_bout': 2,
                      'proportion_smoothing_sd': None, 'mixture_model_params': _mixture_model_params()}
        # Both calls read the same (read-only) session tree.
        out_all = find_onset_epochs(target_category=None, **common, **kwargs)
        out_cat = find_onset_epochs(target_category=3, **common, **kwargs)
        np.testing.assert_allclose(out_all['sess_B']['male']['negative_events'],
                                   out_cat['sess_B']['male']['negative_events'])

    def test_target_category_none_pools_all_categories(self, tmp_path):
        """``target_category=None`` (the default) reproduces the original
        all-USV behavior: every onset past ``filter_history`` is positive."""

        rows = {
            'emitter': ['male', 'male', 'male'],
            'start': [2.0, 3.0, 4.0],
            'stop': [2.05, 3.05, 4.05],
            'usv_category': [3, 2, 3],
        }
        kwargs = self._build(tmp_path, rows)
        out = find_onset_epochs(prediction_mode='individual', filter_history=1.0,
                               usv_bout_time=0.5, min_usv_per_bout=2,
                               proportion_smoothing_sd=None, mixture_model_params=_mixture_model_params(),
                               target_category=None,
                               **kwargs)
        np.testing.assert_allclose(out['sess_B']['male']['positive_events'], [2.0, 3.0, 4.0])

    def test_target_classes_select_the_positive_onsets(self, tmp_path):
        """The ``usv`` / ``squeak`` booleans decide which onsets are positive: each target
        list keeps the rows of its classes -- "usv" pure USVs (usv true, squeak false),
        "squeak" pure squeaks, "overlap" both flags true -- and null flags are in no class;
        each positive is timed as the event its list asks about: the pure squeak at its own
        onset (3.01 s, not its segment's 3.0 s), the overlap segment at its squeak's onset
        (5.02 s) in a squeak-only list and at the earlier of the two (its USV's 5.0 s)
        otherwise; the silent-epoch negatives are the same for every list, because silence is
        checked against every vocalization of the mouse."""

        rows = {
            'emitter': ['male', 'male', 'male', 'male', 'male'],
            'start': [2.0, 3.0, 4.0, 5.0, 6.0],
            'stop': [2.05, 3.05, 4.05, 5.05, 6.05],
            'usv_category': [1, 1, 1, 1, 1],
            'usv': [True, False, True, True, None],
            'squeak': [False, True, False, True, None],
            'squeak_start': [None, 3.01, None, 5.02, None],
            'squeak_end': [None, 3.04, None, 5.06, None],
        }
        kwargs = self._build(tmp_path, rows)
        common = {'prediction_mode': 'individual', 'filter_history': 1.0,
                      'usv_bout_time': 0.5, 'min_usv_per_bout': 2,
                      'proportion_smoothing_sd': None, 'mixture_model_params': _mixture_model_params()}
        expected = {
            ('usv',): [2.0, 4.0],
            ('usv', 'overlap'): [2.0, 4.0, 5.0],
            ('squeak',): [3.01],
            ('squeak', 'overlap'): [3.01, 5.02],
            ('overlap',): [5.0],
            ('usv', 'squeak'): [2.0, 3.01, 4.0],
            ('usv', 'squeak', 'overlap'): [2.0, 3.01, 4.0, 5.0],
        }
        out = {classes: find_onset_epochs(target_classes=list(classes), **common, **kwargs)['sess_B']['male']
               for classes in expected}
        for classes, positives in expected.items():
            np.testing.assert_allclose(out[classes]['positive_events'], positives)
            np.testing.assert_allclose(out[classes]['negative_events'], out[('usv',)]['negative_events'])
        # the default is ['usv']
        default = find_onset_epochs(**common, **kwargs)['sess_B']['male']
        np.testing.assert_allclose(default['positive_events'], [2.0, 4.0])

    def test_overlap_segments_join_bouts_when_targeted(self, tmp_path):
        """In a bout mode an overlap segment (usv and squeak true) between two pure USVs is
        a bout member under ["usv", "overlap"], so the three form one bout with one
        onset; under the default ["usv"] it is not a target and the two pure USVs, now
        farther apart than the inter-bout threshold (1 s for these parameters), are two
        bouts of one USV each, below the two-USV floor. The negatives do not depend on
        the target classes."""

        rows = {
            'emitter': ['male', 'male', 'male'],
            'start': [2.0, 2.7, 3.4],
            'stop': [2.01, 2.71, 3.41],
            'usv_category': [1, 1, 1],
            'usv': [True, True, True],
            'squeak': [False, True, False],
        }
        kwargs = self._build(tmp_path, rows)
        common = {'prediction_mode': 'bout_onset', 'filter_history': 1.0, 'usv_bout_time': 0.5,
                      'min_usv_per_bout': 2, 'proportion_smoothing_sd': None,
                      'mixture_model_params': _mixture_model_params()}
        with_overlap = find_onset_epochs(target_classes=['usv', 'overlap'], **common, **kwargs)['sess_B']['male']
        pure = find_onset_epochs(target_classes=['usv'], **common, **kwargs)['sess_B']['male']
        np.testing.assert_allclose(with_overlap['positive_events'], [2.0])
        assert len(pure['positive_events']) == 0
        np.testing.assert_allclose(with_overlap['negative_events'], pure['negative_events'])

    @pytest.mark.parametrize(('emitter', 'squeak_start', 'kept'), [
        ('male', 1.5, False),     # the mouse's own squeak 0.5 s before the onset
        (None, 1.5, False),       # a vocalization assigned to no animal
        ('male', 0.5, True),      # own squeak, but ended before the 1 s history window
        ('female', 1.5, True),    # the partner's squeak: a predictor, not part of silence
    ])
    def test_bout_history_is_clean_of_every_class(self, tmp_path, emitter, squeak_start, kept):
        """A 'bout_onset' positive's clean history obeys the negatives' rule: no
        vocalization of the target mouse of ANY class (not only the target classes) and
        none assigned to no animal may fall in the window before the onset, so a squeak
        can no longer sit in a positive's history while every negative is free of
        squeaks. The partner's vocalizations do not count, for positives and negatives
        alike."""

        rows = {
            'emitter': [emitter, 'male', 'male'],
            'start': [squeak_start, 2.0, 2.3],
            'stop': [squeak_start + 0.05, 2.05, 2.35],
            'usv_category': [1, 1, 1],
            'usv': [False, True, True],
            'squeak': [True, False, False],
        }
        kwargs = self._build(tmp_path, rows)
        male = find_onset_epochs(prediction_mode='bout_onset', filter_history=1.0, usv_bout_time=0.5,
                                 min_usv_per_bout=2, proportion_smoothing_sd=None,
                                 mixture_model_params=_mixture_model_params(), target_classes=['usv'],
                                 **kwargs)['sess_B']['male']
        if kept:
            np.testing.assert_allclose(male['positive_events'], [2.0])
        else:
            assert male['positive_events'].size == 0

    def test_target_classes_refuse_what_they_cannot_do(self, tmp_path):
        """A target holding pure squeaks in a bout mode, an unknown or repeated class, an
        empty list, and a summary without the ``usv`` / ``squeak`` columns all raise
        rather than silently falling back to every vocalization."""

        rows = {
            'emitter': ['male', 'male'],
            'start': [2.0, 3.0],
            'stop': [2.05, 3.05],
            'usv_category': [1, 1],
            'usv': [True, False],
            'squeak': [False, True],
        }
        kwargs = self._build(tmp_path, rows)
        common = {'filter_history': 1.0, 'usv_bout_time': 0.5, 'min_usv_per_bout': 2,
                      'proportion_smoothing_sd': None, 'mixture_model_params': _mixture_model_params()}
        for mode in ('bout_onset', 'bout_offset'):
            with pytest.raises(ValueError, match="bouts are grouped from"):
                find_onset_epochs(prediction_mode=mode, target_classes=['squeak', 'overlap'], **common, **kwargs)
        with pytest.raises(ValueError, match="must hold distinct classes"):
            find_onset_epochs(prediction_mode='individual', target_classes=['chirp'], **common, **kwargs)
        with pytest.raises(ValueError, match="must hold distinct classes"):
            find_onset_epochs(prediction_mode='individual', target_classes=['usv', 'usv'], **common, **kwargs)
        with pytest.raises(ValueError, match="non-empty list"):
            find_onset_epochs(prediction_mode='individual', target_classes=[], **common, **kwargs)
        out = find_onset_epochs(prediction_mode='individual', target_classes=['squeak'], **common, **kwargs)
        np.testing.assert_allclose(out['sess_B']['male']['positive_events'], [3.0])

        bare = tmp_path / 'bare'
        bare_kwargs = self._build(bare, {'emitter': ['male'], 'start': [2.0], 'stop': [2.05],
                                         'usv_category': [1]})
        csv_path = next((bare / 'sess_B' / 'audio').glob('*_usv_summary.csv'))
        pls.read_csv(csv_path).drop('usv', 'squeak').write_csv(csv_path)
        with pytest.raises(ValueError, match=r"no \['usv', 'squeak'\] columns"):
            find_onset_epochs(prediction_mode='individual', target_classes=['usv'], **common, **bare_kwargs)

    @pytest.mark.filterwarnings("ignore::astropy.utils.exceptions.AstropyUserWarning")
    def test_target_category_predictors_unaffected(self, tmp_path):
        """A ``target_category`` filter restricts the positive onsets but never
        leaks into the predictor vocal signals: the non-target category's
        'usv_cat_X' trace and the pooled 'usv_rate' are still emitted over the
        mouse's full USV set."""

        rows = {
            'emitter': ['male', 'male'],
            'start': [2.0, 3.0],
            'stop': [2.05, 3.05],
            'usv_category': [3, 4],
        }
        kwargs = self._build(tmp_path, rows)
        out = find_onset_epochs(prediction_mode='individual', filter_history=1.0,
                               usv_bout_time=0.5, min_usv_per_bout=2,
                               proportion_smoothing_sd=2.0, mixture_model_params=_mixture_model_params(),
                               vocal_output_type='all_rate', target_category=3,
                               **kwargs)
        male = out['sess_B']['male']
        # Positives restricted to the cat-3 onset ...
        np.testing.assert_allclose(male['positive_events'], [2.0])
        # ... yet the predictor signals still carry the non-target cat-4 trace
        # and a pooled rate computed over BOTH USVs.
        signals = male['continuous_vocal_signals']
        assert 'usv_cat_3' in signals
        assert 'usv_cat_4' in signals
        assert signals['usv_rate'].sum() > 0
        assert male['usv_count'].sum() > 0

    def test_target_category_ignored_in_bout_mode(self, tmp_path):
        """``target_category`` is honoured only in 'individual' mode; in 'bout_onset'
        mode it is ignored (all categories pooled), so the positive events match
        the unfiltered bout result."""

        rows = {
            'emitter': ['male', 'male', 'male'],
            'start': [2.0, 2.1, 2.2],
            'stop': [2.05, 2.15, 2.25],
            'usv_category': [3, 2, 3],
        }
        kwargs = self._build(tmp_path, rows)
        common = {'prediction_mode': 'bout_onset', 'filter_history': 1.0, 'usv_bout_time': 0.5,
                      'min_usv_per_bout': 2, 'proportion_smoothing_sd': None,
                      'mixture_model_params': _mixture_model_params()}
        out_all = find_onset_epochs(target_category=None, **common, **kwargs)
        out_cat = find_onset_epochs(target_category=3, **common, **kwargs)
        np.testing.assert_allclose(out_all['sess_B']['male']['positive_events'],
                                   out_cat['sess_B']['male']['positive_events'])
        # The pooled 3-syllable cluster still forms exactly one bout onset.
        assert out_cat['sess_B']['male']['positive_events'].size == 1

    def test_target_category_missing_column_raises(self, tmp_path):
        """If the requested category column is absent from the summary, the
        single-category onset target cannot be applied: the loader raises the
        labels-unavailable error naming the setting (it used to fall back to all
        USVs with only a printed warning)."""

        rows = {
            'emitter': ['male', 'male'],
            'start': [2.0, 3.0],
            'stop': [2.05, 3.05],
        }
        kwargs = self._build(tmp_path, rows)
        with pytest.raises(ValueError, match=_LABELS_UNAVAILABLE) as excinfo:
            find_onset_epochs(prediction_mode='individual', filter_history=1.0,
                              usv_bout_time=0.5, min_usv_per_bout=2,
                              proportion_smoothing_sd=None, mixture_model_params=_mixture_model_params(),
                              category_column='qlvm_category', target_category=3,
                              **kwargs)
        assert 'onset_target_category 3' in str(excinfo.value)
        assert "'qlvm_category'" in str(excinfo.value)

    def test_target_category_null_column_raises(self, tmp_path):
        """A null category column (the label-free setting) with a single-category
        onset target raises before any summary is read."""

        kwargs = self._build(tmp_path, {'emitter': ['male'], 'start': [2.0], 'stop': [2.05]})
        with pytest.raises(ValueError, match=_LABELS_UNAVAILABLE):
            find_onset_epochs(prediction_mode='individual', filter_history=1.0,
                              usv_bout_time=0.5, min_usv_per_bout=2,
                              proportion_smoothing_sd=None, mixture_model_params=_mixture_model_params(),
                              category_column=None, target_category=3,
                              **kwargs)

    @pytest.mark.parametrize('vocal_output_type', ['categories_rate', 'all_rate'])
    @pytest.mark.parametrize('category_column', [None, 'absent_label_column'])
    def test_category_predictors_without_labels_raise(self, tmp_path, vocal_output_type, category_column):
        """Per-category predictor traces with no label column (null setting, or a
        column the summary lacks) raise the labels-unavailable error naming
        ``vocal_predictor_type`` instead of silently building no category trace."""

        rows = {
            'emitter': ['male', 'male'],
            'start': [2.0, 3.0],
            'stop': [2.05, 3.05],
            'qlvm1': [0.1, 0.2],
            'qlvm2': [0.3, 0.4],
        }
        kwargs = self._build(tmp_path, rows)
        with pytest.raises(ValueError, match=_LABELS_UNAVAILABLE) as excinfo:
            find_onset_epochs(prediction_mode='bout_onset', filter_history=1.0,
                              usv_bout_time=0.5, min_usv_per_bout=2,
                              proportion_smoothing_sd=None, mixture_model_params=_mixture_model_params(),
                              vocal_output_type=vocal_output_type,
                              category_column=category_column, **kwargs)
        assert f"vocal_predictor_type '{vocal_output_type}'" in str(excinfo.value)

    @pytest.mark.parametrize('vocal_output_type', ['pooled_rate', 'pooled_binary'])
    def test_pooled_predictors_need_no_category_column(self, tmp_path, vocal_output_type):
        """The pooled predictor modes run with a null category column."""

        rows = {
            'emitter': ['male', 'male'],
            'start': [2.0, 3.0],
            'stop': [2.05, 3.05],
        }
        kwargs = self._build(tmp_path, rows)
        out = find_onset_epochs(prediction_mode='bout_onset', filter_history=1.0,
                                usv_bout_time=0.5, min_usv_per_bout=2,
                                proportion_smoothing_sd=2.0, mixture_model_params=_mixture_model_params(),
                                vocal_output_type=vocal_output_type,
                                category_column=None, **kwargs)
        signals = out['sess_B']['male']['continuous_vocal_signals']
        expected = 'usv_event' if vocal_output_type == 'pooled_binary' else 'usv_rate'
        assert set(signals) == {expected}

    def test_missing_summary_csv_skips_session(self, tmp_path, capsys):
        """A session ROOT with no ``audio/*_usv_summary.csv`` is skipped with a
        warning, leaving its dict entry empty."""

        sess = tmp_path / 'sess_empty'
        (sess / 'audio').mkdir(parents=True)
        out = find_onset_epochs(root_directories=[str(sess)],
                               mouse_ids_dict={'sess_empty': ['male', 'female']},
                               camera_fps_dict={'sess_empty': 100.0},
                               features_dict={'sess_empty': _features_df(100)},
                               prediction_mode='bout_onset', filter_history=1.0,
                               usv_bout_time=0.5, min_usv_per_bout=2,
                               proportion_smoothing_sd=None, mixture_model_params=_mixture_model_params())
        assert out['sess_empty'] == {}
        assert 'No USV summary' in capsys.readouterr().out

    def test_unregistered_session_skipped(self, tmp_path, capsys):
        """A session present on disk but absent from ``mouse_ids_dict`` is
        skipped with a warning."""

        sess = tmp_path / 'sess_C'
        _write_usv_summary(sess, {
            'emitter': ['male'], 'start': [2.0], 'stop': [2.05],
            'usv_category': [1],
        })
        out = find_onset_epochs(root_directories=[str(sess)],
                               mouse_ids_dict={},
                               camera_fps_dict={'sess_C': 100.0},
                               features_dict={'sess_C': _features_df(100)},
                               prediction_mode='bout_onset', filter_history=1.0,
                               usv_bout_time=0.5, min_usv_per_bout=2,
                               proportion_smoothing_sd=None, mixture_model_params=_mixture_model_params())
        assert out['sess_C'] == {}
        assert 'No mouse names registered' in capsys.readouterr().out

    def test_unknown_prediction_mode_raises(self, tmp_path):
        """An unsupported ``prediction_mode`` raises ``ValueError``."""

        kwargs = self._build(tmp_path, {
            'emitter': ['male'], 'start': [2.0], 'stop': [2.05],
            'usv_category': [1],
        })
        with pytest.raises(ValueError, match='Unknown prediction_mode'):
            find_onset_epochs(prediction_mode='bogus', filter_history=1.0,
                             usv_bout_time=0.5, min_usv_per_bout=2,
                             proportion_smoothing_sd=None, mixture_model_params=_mixture_model_params(),
                             **kwargs)

    def test_invalid_mixture_model_component_index_raises(self, tmp_path):
        """An out-of-range or negative ``mixture_model_component_index`` raises ``ValueError``
        (a negative index would wrap to the last component)."""

        kwargs = self._build(tmp_path, {
            'emitter': ['male'], 'start': [2.0], 'stop': [2.05],
            'usv_category': [1],
        })
        for bad_index in (5, -1):
            with pytest.raises(ValueError, match='Invalid mixture_model_component_index'):
                find_onset_epochs(prediction_mode='bout_onset', filter_history=1.0,
                                 usv_bout_time=0.5, min_usv_per_bout=2,
                                 mixture_model_component_index=bad_index,
                                 proportion_smoothing_sd=None, mixture_model_params=_mixture_model_params(),
                                 **kwargs)

    def test_bout_mode_rejects_bout_with_dirty_history(self, tmp_path):
        """A multi-syllable bout whose immediately-preceding USV is within
        ``filter_history`` is rejected (dirty pre-bout history)."""

        rows = {
            'emitter': ['male', 'male', 'male'],
            'start': [2.0, 2.5, 2.6],
            'stop': [2.05, 2.55, 2.65],
            'usv_category': [1, 1, 1],
        }
        kwargs = self._build(tmp_path, rows)
        # IBI threshold ~ exp(log(0.3)) so the first USV breaks into its own
        # single-syllable bout and the [2.5, 2.6] bout has a prior USV at 2.05,
        # only 0.45 s away (< filter_history = 1.0) -> rejected.
        mixture_model = {'male': {'means': [np.log(0.3)], 'sds': [0.0]},
               'female': {'means': [np.log(0.3)], 'sds': [0.0]}}
        out = find_onset_epochs(prediction_mode='bout_onset', filter_history=1.0,
                               usv_bout_time=0.5, min_usv_per_bout=2,
                               proportion_smoothing_sd=None, mixture_model_params=mixture_model,
                               **kwargs)
        assert out['sess_B']['male']['positive_events'].size == 0

    @pytest.mark.filterwarnings("ignore::astropy.utils.exceptions.AstropyUserWarning")
    def test_categories_rate_skips_non_integer_category(self, tmp_path):
        """A non-integer category label is skipped during per-category signal
        generation (no 'usv_cat_*' key is created for it)."""

        rows = {
            'emitter': ['male', 'male'],
            'start': [2.0, 3.0],
            'stop': [2.05, 3.05],
            'usv_category': ['noise', 'noise'],
        }
        kwargs = self._build(tmp_path, rows)
        out = find_onset_epochs(prediction_mode='individual', filter_history=1.0,
                               usv_bout_time=0.5, min_usv_per_bout=2,
                               proportion_smoothing_sd=2.0, mixture_model_params=_mixture_model_params(),
                               vocal_output_type='categories_rate',
                               **kwargs)
        signals = out['sess_B']['male']['continuous_vocal_signals']
        assert not any(k.startswith('usv_cat_') for k in signals)

    def test_bout_mode_single_usv_bout(self, tmp_path):
        """A mouse with exactly one USV exercises the single-syllable bout
        index branch; with ``min_usv_per_bout=1`` it becomes one positive
        onset."""

        rows = {
            'emitter': ['male'], 'start': [2.0], 'stop': [2.05],
            'usv_category': [1],
        }
        kwargs = self._build(tmp_path, rows)
        out = find_onset_epochs(prediction_mode='bout_onset', filter_history=1.0,
                               usv_bout_time=0.5, min_usv_per_bout=1,
                               proportion_smoothing_sd=None, mixture_model_params=_mixture_model_params(),
                               **kwargs)
        np.testing.assert_allclose(out['sess_B']['male']['positive_events'], [2.0])

    def test_state_mode_empty_when_no_valid_samples(self, tmp_path, capsys):
        """When the grid yields no valid frame samples (``filter_history``
        exceeds session duration), 'state' mode returns empty event arrays."""

        rows = {
            'emitter': ['male'], 'start': [0.1], 'stop': [0.15],
            'usv_category': [1],
        }
        # n_frames=50 at 100 fps -> 0.5 s session; filter_history=2.0 > 0.5.
        kwargs = self._build(tmp_path, rows, n_frames=50, fps=100.0)
        out = find_onset_epochs(prediction_mode='state', filter_history=2.0,
                               usv_bout_time=0.5, min_usv_per_bout=2,
                               proportion_smoothing_sd=None, mixture_model_params=_mixture_model_params(),
                               **kwargs)
        male = out['sess_B']['male']
        assert male['positive_events'].size == 0
        assert male['negative_events'].size == 0
        assert 'No valid' in capsys.readouterr().out


# find_usv_categories


class TestFindUsvCategories:
    """``find_usv_categories`` on synthetic session trees: target-versus-other splitting,
    per-category streams and the handling of unknown or null categories."""

    def _kwargs(self, tmp_path, rows: dict, n_frames: int = 600, fps: float = 100.0):
        """
        Description
        -----------
        Builds a one-session tree (``sess_D``) holding a synthetic USV summary and
        returns the common loader kwargs for the ``find_usv_categories`` tests below.

        Parameters
        ----------
        tmp_path (pathlib.Path)
            The test's temporary directory; the session root is created under it.
        rows (dict)
            Column-name -> value-list mapping for the USV summary, forwarded to
            ``_write_usv_summary``.
        n_frames (int)
            Number of frames in the synthetic behavioural-feature table.
        fps (float)
            Camera frame rate recorded for the session.

        Returns
        -------
        kwargs (dict)
            ``root_directories``, ``mouse_ids_dict``, ``camera_fps_dict`` and
            ``features_dict`` keyed for the single session.
        """

        sess = tmp_path / 'sess_D'
        _write_usv_summary(sess, rows)
        return {
            'root_directories': [str(sess)],
            'mouse_ids_dict': {'sess_D': ['male', 'female']},
            'camera_fps_dict': {'sess_D': fps},
            'features_dict': {'sess_D': _features_df(n_frames)},
        }

    def test_target_vs_other_split(self, tmp_path):
        """With ``target_category`` set, USVs split into target and other event
        arrays and per-category streams are recorded."""

        rows = {
            'emitter': ['male', 'male', 'male'],
            'start': [2.0, 3.0, 4.0],
            'stop': [2.05, 3.05, 4.05],
            'usv_category': [1, 2, 1],
        }
        out = find_usv_categories(target_category=1, filter_history=1.0,
                                  **self._kwargs(tmp_path, rows))
        male = out['sess_D']['male']
        np.testing.assert_allclose(male['target_events'], [2.0, 4.0])
        np.testing.assert_allclose(male['other_events'], [3.0])
        assert set(male['events_by_category'].keys()) == {1, 2}
        assert 1 in male['category_streams']

    def test_predictor_traces_include_the_history_period(self, tmp_path):
        """The vocal predictor traces cover the whole session: a USV inside the history
        period (0.5 s, before ``filter_history`` = 1.0 s) is no event target but still appears
        in the trace, so an event soon after the history period sees it in its history window."""

        rows = {
            'emitter': ['male', 'male'],
            'start': [0.5, 3.0],
            'stop': [0.55, 3.05],
            'usv_category': [2, 4],
        }
        out = find_usv_categories(target_category=None, filter_history=1.0,
                                  vocal_output_type='pooled_binary', proportion_smoothing_sd=None,
                                  **self._kwargs(tmp_path, rows))
        male = out['sess_D']['male']
        assert set(male['events_by_category'].keys()) == {4}
        trace = male['continuous_vocal_signals']['usv_event']
        assert trace[50] > 0
        assert trace[300] > 0

    def test_multinomial_mode_events_by_category(self, tmp_path):
        """With ``target_category=None`` the function populates
        ``events_by_category`` for every category and leaves target/other
        unset."""

        rows = {
            'emitter': ['male', 'male'],
            'start': [2.0, 3.0],
            'stop': [2.05, 3.05],
            'usv_category': [2, 4],
        }
        out = find_usv_categories(target_category=None, filter_history=1.0,
                                  **self._kwargs(tmp_path, rows))
        male = out['sess_D']['male']
        assert male['target_events'] is None
        assert set(male['events_by_category'].keys()) == {2, 4}

    @pytest.mark.filterwarnings("ignore::astropy.utils.exceptions.AstropyUserWarning")
    @pytest.mark.parametrize("prefix", ["qlvm", "qlvm_duration", "qlvm_entropy"])
    def test_continuous_targets_from_manifold_columns(self, tmp_path, prefix):
        """When ``manifold_column_names`` are present, continuous onsets, stacked
        targets and the region label array are written. The label is the regular
        map's ``qlvm_category`` for EVERY map, the conditional ones included (the
        categories are defined on the regular map and label the USV); a
        ``<prefix>_category`` / ``<prefix>_supercategory`` column of the manifold's
        own prefix is never read."""

        rows = {
            'emitter': ['male', 'male'],
            'start': [2.0, 3.0],
            'stop': [2.05, 3.05],
            'usv_category': [1, 2],
            f'{prefix}1': [0.1, 0.2],
            f'{prefix}2': [0.3, 0.4],
            'qlvm_category': [3, 4],
        }
        if prefix != 'qlvm':
            rows[f'{prefix}_category'] = [20, 21]
            rows[f'{prefix}_supercategory'] = [10, 11]
        out = find_usv_categories(target_category=None, filter_history=1.0,
                                  vocal_output_type='all_rate',
                                  proportion_smoothing_sd=2.0,
                                  manifold_column_names=[f'{prefix}1', f'{prefix}2'],
                                  **self._kwargs(tmp_path, rows))
        male = out['sess_D']['male']
        np.testing.assert_allclose(male['continuous_onsets'], [2.0, 3.0])
        assert male['continuous_targets'].shape == (2, 2)
        assert 'continuous_supercategory' not in male
        np.testing.assert_allclose(male['continuous_category'], [3, 4])
        assert 'usv_rate' in male['continuous_vocal_signals']
        assert 'usv_cat_1' in male['continuous_vocal_signals']

    @pytest.mark.filterwarnings("ignore::astropy.utils.exceptions.AstropyUserWarning")
    def test_squeak_map_onsets_are_the_squeaks_own(self, tmp_path):
        """The squeak map's onsets are the squeaks' own (``squeak_start``): the pure
        squeak at 3.02 s rather than its segment's 3.0 s, the overlap segment at its
        squeak's 3.5 s rather than its USV's 2.5 s, so the two swap order and the targets
        follow their rows; the pure USV has no squeak coordinates and is dropped. A
        summary without ``squeak_start`` raises."""

        rows = {
            'emitter': ['male', 'male', 'male'],
            'start': [2.0, 2.5, 3.0],
            'stop': [2.05, 2.55, 3.05],
            'usv_category': [1, 1, 1],
            'usv': [True, True, False],
            'squeak': [False, True, True],
            'squeak_start': [None, 3.5, 3.02],
            'squeak_end': [None, 3.6, 3.04],
            'qlvm_squeak1': [None, 0.2, 0.1],
            'qlvm_squeak2': [None, 0.4, 0.3],
        }
        common = {'target_category': None, 'filter_history': 1.0, 'category_column': None,
                  'manifold_column_names': ['qlvm_squeak1', 'qlvm_squeak2']}
        male = find_usv_categories(**common, **self._kwargs(tmp_path, rows))['sess_D']['male']
        np.testing.assert_allclose(male['continuous_onsets'], [3.02, 3.5])
        np.testing.assert_allclose(male['continuous_targets'], [[0.1, 0.3], [0.2, 0.4]])
        kwargs = self._kwargs(tmp_path / 'no_extent', rows)
        csv_path = tmp_path / 'no_extent' / 'sess_D' / 'audio' / 'sess_D_usv_summary.csv'
        pls.read_csv(csv_path).drop('squeak_start', 'squeak_end').write_csv(csv_path)
        with pytest.raises(KeyError, match="squeak_start"):
            find_usv_categories(**common, **kwargs)

    @pytest.mark.filterwarnings("ignore::astropy.utils.exceptions.AstropyUserWarning")
    def test_continuous_targets_without_qlvm_category_carry_no_labels(self, tmp_path):
        """A summary whose categories were not assigned yet (no ``qlvm_category``)
        yields manifold targets without a region label array."""

        rows = {
            'emitter': ['male', 'male'],
            'start': [2.0, 3.0],
            'stop': [2.05, 3.05],
            'usv_category': [1, 2],
            'qlvm_duration1': [0.1, 0.2],
            'qlvm_duration2': [0.3, 0.4],
        }
        out = find_usv_categories(target_category=None, filter_history=1.0,
                                  manifold_column_names=['qlvm_duration1', 'qlvm_duration2'],
                                  **self._kwargs(tmp_path, rows))
        male = out['sess_D']['male']
        assert male['continuous_targets'].shape == (2, 2)
        assert 'continuous_category' not in male

    def test_missing_category_column_raises(self, tmp_path):
        """A CSV lacking ``category_column`` raises the labels-unavailable
        ``ValueError`` naming the column and the file."""

        rows = {
            'emitter': ['male'], 'start': [2.0], 'stop': [2.05],
        }
        with pytest.raises(ValueError, match=_LABELS_UNAVAILABLE) as excinfo:
            find_usv_categories(target_category=1, filter_history=1.0,
                                **self._kwargs(tmp_path, rows))
        assert "'usv_category'" in str(excinfo.value)
        assert 'sess_D_usv_summary.csv' in str(excinfo.value)

    @pytest.mark.parametrize('target_category', [None, 3])
    def test_categorical_paths_raise_on_null_column(self, tmp_path, target_category):
        """The multinomial (no target) and binomial (target) category paths need
        labels: a null category column raises before any session is read."""

        rows = {'emitter': ['male'], 'start': [2.0], 'stop': [2.05]}
        with pytest.raises(ValueError, match=_LABELS_UNAVAILABLE):
            find_usv_categories(target_category=target_category, filter_history=1.0,
                                category_column=None, **self._kwargs(tmp_path, rows))

    def test_manifold_path_without_category_column(self, tmp_path):
        """On the continuous manifold path a null category column is allowed: the
        manifold targets are returned, with no category packet
        and no category events."""

        rows = {
            'emitter': ['male', 'male'],
            'start': [2.0, 3.0],
            'stop': [2.05, 3.05],
            'qlvm1': [0.1, 0.2],
            'qlvm2': [0.3, 0.4],
        }
        out = find_usv_categories(target_category=None, filter_history=1.0,
                                  vocal_output_type='pooled_rate', proportion_smoothing_sd=2.0,
                                  category_column=None, manifold_column_names=['qlvm1', 'qlvm2'],
                                  **self._kwargs(tmp_path, rows))
        male = out['sess_D']['male']
        np.testing.assert_allclose(male['continuous_onsets'], [2.0, 3.0])
        np.testing.assert_allclose(male['continuous_targets'], [[0.1, 0.3], [0.2, 0.4]])
        assert 'continuous_supercategory' not in male
        assert 'continuous_category' not in male
        assert male['events_by_category'] == {}
        assert male['category_streams'] == {}
        assert set(male['continuous_vocal_signals']) == {'usv_rate'}

    def test_manifold_path_category_predictors_still_need_labels(self, tmp_path):
        """Even on the manifold path, per-category predictor traces need a label
        column."""

        rows = {'emitter': ['male'], 'start': [2.0], 'stop': [2.05], 'qlvm1': [0.1], 'qlvm2': [0.3]}
        with pytest.raises(ValueError, match="vocal_predictor_type 'categories_rate'"):
            find_usv_categories(target_category=None, filter_history=1.0,
                                vocal_output_type='categories_rate', category_column=None,
                                manifold_column_names=['qlvm1', 'qlvm2'],
                                **self._kwargs(tmp_path, rows))

    def test_manifold_path_drops_unplaced_usvs(self, tmp_path, capsys):
        """USVs with null / NaN manifold coordinates (not placed by the
        embedding) are dropped from onsets, targets and label arrays, and the
        per-session and total counts are printed, so no NaN reaches the
        inverse-density KDE."""

        rows = {
            'emitter': ['male', 'male', 'male', 'male'],
            'start': [2.0, 3.0, 4.0, 5.0],
            'stop': [2.05, 3.05, 4.05, 5.05],
            'qlvm1': [0.1, None, 0.3, 0.4],
            'qlvm2': [0.5, 0.6, None, 0.8],
            'qlvm_category': [1, 2, 3, 4],
        }
        out = find_usv_categories(target_category=None, filter_history=1.0,
                                  category_column=None, manifold_column_names=['qlvm1', 'qlvm2'],
                                  **self._kwargs(tmp_path, rows))
        male = out['sess_D']['male']
        np.testing.assert_allclose(male['continuous_onsets'], [2.0, 5.0])
        np.testing.assert_allclose(male['continuous_targets'], [[0.1, 0.5], [0.4, 0.8]])
        assert np.isfinite(male['continuous_targets']).all()
        np.testing.assert_allclose(male['continuous_category'], [1, 4])
        printed = capsys.readouterr().out
        assert 'sess_D (male): dropped 2 of 4 vocalizations with null/NaN manifold coordinates' in printed
        assert 'dropped 2 vocalizations with null/NaN coordinates in total' in printed

    def test_manifold_path_all_null_column(self, tmp_path):
        """A manifold column that is null for every USV (read by CSV inference
        as text) leaves no placed USV rather than crashing."""

        rows = {
            'emitter': ['male', 'male'],
            'start': [2.0, 3.0],
            'stop': [2.05, 3.05],
            'qlvm1': [None, None],
            'qlvm2': [None, None],
        }
        out = find_usv_categories(target_category=None, filter_history=1.0,
                                  category_column=None, manifold_column_names=['qlvm1', 'qlvm2'],
                                  **self._kwargs(tmp_path, rows))
        male = out['sess_D']['male']
        assert male['continuous_onsets'].size == 0
        assert male['continuous_targets'].shape == (0, 2)

    def test_filter_history_removes_early_usvs(self, tmp_path):
        """USVs starting at/under ``filter_history`` are discarded, leaving a
        mouse with only late USVs in the output."""

        rows = {
            'emitter': ['male', 'male'],
            'start': [0.5, 3.0],
            'stop': [0.55, 3.05],
            'usv_category': [1, 1],
        }
        out = find_usv_categories(target_category=1, filter_history=1.0,
                                  **self._kwargs(tmp_path, rows))
        np.testing.assert_allclose(out['sess_D']['male']['target_events'], [3.0])

    def test_pooled_binary_signal(self, tmp_path):
        """``vocal_output_type='pooled_binary'`` writes a binary 'usv_event'
        aggregate trace."""

        rows = {
            'emitter': ['male', 'male'],
            'start': [2.0, 3.0],
            'stop': [2.05, 3.05],
            'usv_category': [1, 1],
        }
        out = find_usv_categories(target_category=None, filter_history=1.0,
                                  vocal_output_type='pooled_binary',
                                  **self._kwargs(tmp_path, rows))
        signals = out['sess_D']['male']['continuous_vocal_signals']
        assert 'usv_event' in signals
        assert set(np.unique(signals['usv_event'])).issubset({0.0, 1.0})

    def test_unregistered_session_skipped(self, tmp_path, capsys):
        """A session absent from ``mouse_ids_dict`` is skipped with a
        warning."""

        rows = {
            'emitter': ['male'], 'start': [2.0], 'stop': [2.05],
            'usv_category': [1],
        }
        kwargs = self._kwargs(tmp_path, rows)
        kwargs['mouse_ids_dict'] = {}
        out = find_usv_categories(target_category=1, filter_history=1.0, **kwargs)
        assert out['sess_D'] == {}
        assert 'No mouse names registered' in capsys.readouterr().out

    def test_missing_features_dict_skips_session(self, tmp_path, capsys):
        """A session with no feature data (no duration) is skipped with a
        warning."""

        rows = {
            'emitter': ['male'], 'start': [2.0], 'stop': [2.05],
            'usv_category': [1],
        }
        kwargs = self._kwargs(tmp_path, rows)
        kwargs['features_dict'] = {}
        out = find_usv_categories(target_category=1, filter_history=1.0, **kwargs)
        assert out['sess_D'] == {}
        assert 'No feature data' in capsys.readouterr().out

    def test_missing_summary_csv_skips_session(self, tmp_path, capsys):
        """A session lacking the USV summary CSV is skipped with a warning."""

        sess = tmp_path / 'sess_G'
        (sess / 'audio').mkdir(parents=True)
        out = find_usv_categories(
            root_directories=[str(sess)],
            mouse_ids_dict={'sess_G': ['male', 'female']},
            camera_fps_dict={'sess_G': 100.0},
            features_dict={'sess_G': _features_df(100)},
            target_category=1, filter_history=1.0)
        assert out['sess_G'] == {}
        assert 'No USV summary' in capsys.readouterr().out

    def test_all_usvs_filtered_leaves_default_dict(self, tmp_path):
        """A mouse whose USVs are all before ``filter_history`` is left with
        the default (empty) per-mouse dict (no events_by_category)."""

        rows = {
            'emitter': ['male'], 'start': [0.3], 'stop': [0.35],
            'usv_category': [1],
        }
        out = find_usv_categories(target_category=1, filter_history=1.0,
                                  **self._kwargs(tmp_path, rows))
        assert out['sess_D']['male']['events_by_category'] == {}
        assert out['sess_D']['male']['target_events'] is None

    def test_noise_flagged_rows_filtered_globally(self, tmp_path):
        """USVs the noise classifier flagged are removed before any per-mouse processing."""

        rows = {
            'emitter': ['male', 'male'],
            'start': [2.0, 3.0], 'stop': [2.05, 3.05],
            'usv_category': [1, 2], 'noise': [True, False],
        }
        out = find_usv_categories(target_category=None, filter_history=1.0,
                                  exclude_noise_usvs=True,
                                  **self._kwargs(tmp_path, rows))
        male = out['sess_D']['male']
        # Only the row the classifier left alone (usv_category 2) survives.
        assert set(male['events_by_category'].keys()) == {2}

    @pytest.mark.filterwarnings("ignore::astropy.utils.exceptions.AstropyUserWarning")
    def test_categories_rate_skips_non_integer_category(self, tmp_path):
        """A non-integer category is skipped in the per-category signal loop so
        no 'usv_cat_*' key is created for it."""

        rows = {
            'emitter': ['male', 'male'],
            'start': [2.0, 3.0], 'stop': [2.05, 3.05],
            'usv_category': ['p', 'p'],
        }
        out = find_usv_categories(target_category=None, filter_history=1.0,
                                  vocal_output_type='categories_rate',
                                  proportion_smoothing_sd=2.0,
                                  **self._kwargs(tmp_path, rows))
        signals = out['sess_D']['male']['continuous_vocal_signals']
        assert not any(k.startswith('usv_cat_') for k in signals)

    def test_non_integer_category_skipped_in_events(self, tmp_path):
        """Non-integer category labels are skipped when populating
        ``events_by_category`` (but still appear in raw ``category_streams``)."""

        rows = {
            'emitter': ['male'], 'start': [2.0], 'stop': [2.05],
            'usv_category': ['weird'],
        }
        out = find_usv_categories(target_category=None, filter_history=1.0,
                                  **self._kwargs(tmp_path, rows))
        male = out['sess_D']['male']
        assert male['events_by_category'] == {}
        assert 'weird' in male['category_streams']


# find_variable_length_bouts


class TestFindVariableLengthBouts:
    """``find_variable_length_bouts`` on synthetic session trees: bout onsets, durations,
    counts, mask complexity and the vocal output types."""

    def _kwargs(self, tmp_path, rows: dict, n_frames: int = 600, fps: float = 100.0):
        """
        Description
        -----------
        Builds a one-session tree (``sess_E``) holding a synthetic USV summary and
        returns the common loader kwargs for the ``find_variable_length_bouts`` tests below.

        Parameters
        ----------
        tmp_path (pathlib.Path)
            The test's temporary directory; the session root is created under it.
        rows (dict)
            Column-name -> value-list mapping for the USV summary, forwarded to
            ``_write_usv_summary``.
        n_frames (int)
            Number of frames in the synthetic behavioural-feature table.
        fps (float)
            Camera frame rate recorded for the session.

        Returns
        -------
        kwargs (dict)
            ``root_directories``, ``mouse_ids_dict``, ``camera_fps_dict`` and
            ``features_dict`` keyed for the single session.
        """

        sess = tmp_path / 'sess_E'
        _write_usv_summary(sess, rows)
        return {
            'root_directories': [str(sess)],
            'mouse_ids_dict': {'sess_E': ['male', 'female']},
            'camera_fps_dict': {'sess_E': fps},
            'features_dict': {'sess_E': _features_df(n_frames)},
        }

    def test_single_bout_metrics(self, tmp_path):
        """A clean cluster of three close syllables forms one bout whose
        onset, duration, syllable count and mask complexity are computed."""

        rows = {
            'emitter': ['male', 'male', 'male'],
            'start': [2.0, 2.1, 2.2],
            'stop': [2.05, 2.15, 2.25],
            'usv_category': [1, 1, 1],
            'mask_number': [2, 3, 5],
        }
        out = find_variable_length_bouts(min_vocalizations=2, filter_history=1.0,
                                         mixture_model_params=_mixture_model_params(),
                                         **self._kwargs(tmp_path, rows))
        male = out['sess_E']['male']
        np.testing.assert_allclose(male['bout_onsets'], [2.0])
        np.testing.assert_allclose(male['bout_durations'], [2.25 - 2.0])
        np.testing.assert_allclose(male['bout_syllable_counts'], [3])
        np.testing.assert_allclose(male['total_mask_complexity'], [10])
        np.testing.assert_allclose(male['mean_mask_complexity'], [10 / 3])

    def test_bouts_are_grouped_from_the_target_classes(self, tmp_path):
        """Bouts are grouped from the target classes only, as in the onset pipeline: an
        overlap segment between two USV groups bridges them into one bout only when
        overlaps are targets. Under the default ["usv"] the groups (gap 1.0 s, at the 1 s
        threshold of these parameters) are separate bouts and the lone last USV falls
        below the two-USV floor; under ["usv", "overlap"] all four form one bout. A
        target holding pure squeaks is refused."""

        rows = {
            'emitter': ['male', 'male', 'male', 'male'],
            'start': [2.0, 2.3, 2.8, 3.3],
            'stop': [2.0, 2.3, 2.8, 3.3],
            'usv_category': [1, 1, 1, 1],
            'mask_number': [1, 1, 1, 1],
            'usv': [True, True, True, True],
            'squeak': [False, False, True, False],
        }
        common = {'min_vocalizations': 2, 'filter_history': 1.0, 'mixture_model_params': _mixture_model_params()}
        pure = find_variable_length_bouts(target_classes=['usv'], **common, **self._kwargs(tmp_path, rows))['sess_E']['male']
        np.testing.assert_allclose(pure['bout_onsets'], [2.0])
        np.testing.assert_allclose(pure['bout_durations'], [0.3])
        np.testing.assert_allclose(pure['bout_syllable_counts'], [2])
        with_overlap = find_variable_length_bouts(target_classes=['usv', 'overlap'], **common,
                                                  **self._kwargs(tmp_path / 'o', rows))['sess_E']['male']
        np.testing.assert_allclose(with_overlap['bout_onsets'], [2.0])
        np.testing.assert_allclose(with_overlap['bout_durations'], [1.3])
        np.testing.assert_allclose(with_overlap['bout_syllable_counts'], [4])
        with pytest.raises(ValueError, match="bouts are grouped from"):
            find_variable_length_bouts(target_classes=['usv', 'squeak'], **common, **self._kwargs(tmp_path / 's', rows))

    def test_below_min_vocalizations_dropped(self, tmp_path):
        """A bout with fewer than ``min_vocalizations`` syllables is dropped,
        yielding empty bout arrays."""

        rows = {
            'emitter': ['male'],
            'start': [2.0], 'stop': [2.05],
            'usv_category': [1], 'mask_number': [1],
        }
        out = find_variable_length_bouts(min_vocalizations=2, filter_history=1.0,
                                         mixture_model_params=_mixture_model_params(),
                                         **self._kwargs(tmp_path, rows))
        assert out['sess_E']['male']['bout_onsets'].size == 0

    def test_missing_mask_column_uses_unit_complexity(self, tmp_path, capsys):
        """When ``mask_number`` is absent the loader warns and falls back to
        unit mask complexity (count == total complexity)."""

        rows = {
            'emitter': ['male', 'male'],
            'start': [2.0, 2.1], 'stop': [2.05, 2.15],
            'usv_category': [1, 1],
        }
        out = find_variable_length_bouts(min_vocalizations=2, filter_history=1.0,
                                         mixture_model_params=_mixture_model_params(),
                                         **self._kwargs(tmp_path, rows))
        assert 'mask_number' in capsys.readouterr().out
        np.testing.assert_allclose(out['sess_E']['male']['total_mask_complexity'], [2])

    @pytest.mark.filterwarnings("ignore::astropy.utils.exceptions.AstropyUserWarning")
    def test_all_rate_signals_and_noise_filter(self, tmp_path):
        """``vocal_output_type='all_rate'`` writes pooled and per-category
        signals, and the noise-flagged rows are dropped first."""

        rows = {
            'emitter': ['male', 'male', 'male'],
            'start': [2.0, 2.1, 2.2],
            'stop': [2.05, 2.15, 2.25],
            'usv_category': [1, 1, 4],
            'noise': [False, False, True],
            'mask_number': [1, 1, 1],
        }
        out = find_variable_length_bouts(min_vocalizations=2, filter_history=1.0,
                                         mixture_model_params=_mixture_model_params(),
                                         proportion_smoothing_sd=2.0,
                                         vocal_output_type='all_rate',
                                         exclude_noise_usvs=True,
                                         **self._kwargs(tmp_path, rows))
        signals = out['sess_E']['male']['continuous_vocal_signals']
        assert 'usv_rate' in signals
        assert 'usv_cat_1' in signals
        # The row the classifier flagged (usv_category 4) is excluded.
        assert 'usv_cat_4' not in signals

    def test_ibi_threshold_recorded(self, tmp_path):
        """The dynamically-computed IBI threshold is stored per mouse."""

        rows = {
            'emitter': ['male'], 'start': [2.0], 'stop': [2.05],
            'usv_category': [1], 'mask_number': [1],
        }
        out = find_variable_length_bouts(min_vocalizations=2, filter_history=1.0,
                                         mixture_model_params=_mixture_model_params(),
                                         **self._kwargs(tmp_path, rows))
        # exp(0 + 2.58 * 0) == 1.0 for the chosen mixture-model params.
        assert out['sess_E']['male']['ibi_threshold_used'] == pytest.approx(1.0)

    def test_invalid_mixture_model_component_index_raises(self, tmp_path):
        """An out-of-range or negative ``mixture_model_component_index`` raises ``ValueError``
        (a negative index would wrap to the last component)."""

        rows = {
            'emitter': ['male'], 'start': [2.0], 'stop': [2.05],
            'usv_category': [1], 'mask_number': [1],
        }
        for bad_index in (9, -1):
            with pytest.raises(ValueError, match='Invalid mixture_model_component_index'):
                find_variable_length_bouts(min_vocalizations=2, filter_history=1.0,
                                           mixture_model_component_index=bad_index,
                                           mixture_model_params=_mixture_model_params(),
                                           **self._kwargs(tmp_path / str(bad_index), rows))

    def test_missing_summary_csv_skips_session(self, tmp_path, capsys):
        """A session lacking the USV summary CSV is skipped with a warning."""

        sess = tmp_path / 'sess_F'
        (sess / 'audio').mkdir(parents=True)
        out = find_variable_length_bouts(
            root_directories=[str(sess)],
            mouse_ids_dict={'sess_F': ['male', 'female']},
            camera_fps_dict={'sess_F': 100.0},
            features_dict={'sess_F': _features_df(100)},
            min_vocalizations=2, filter_history=1.0, mixture_model_params=_mixture_model_params())
        assert out['sess_F'] == {}
        assert 'No USV summary' in capsys.readouterr().out

    def test_unregistered_session_skipped(self, tmp_path, capsys):
        """A session absent from ``mouse_ids_dict`` is skipped with a
        warning."""

        rows = {
            'emitter': ['male'], 'start': [2.0], 'stop': [2.05],
            'usv_category': [1], 'mask_number': [1],
        }
        kwargs = self._kwargs(tmp_path, rows)
        kwargs['mouse_ids_dict'] = {}
        out = find_variable_length_bouts(min_vocalizations=2, filter_history=1.0,
                                         mixture_model_params=_mixture_model_params(), **kwargs)
        assert out['sess_E'] == {}
        assert 'No mouse names registered' in capsys.readouterr().out

    def test_pooled_binary_signal(self, tmp_path):
        """``vocal_output_type='pooled_binary'`` writes a binary 'usv_event'
        aggregate trace."""

        rows = {
            'emitter': ['male', 'male'],
            'start': [2.0, 2.1], 'stop': [2.05, 2.15],
            'usv_category': [1, 1],
            'mask_number': [1, 1],
        }
        out = find_variable_length_bouts(min_vocalizations=2, filter_history=1.0,
                                         mixture_model_params=_mixture_model_params(),
                                         vocal_output_type='pooled_binary',
                                         **self._kwargs(tmp_path, rows))
        signals = out['sess_E']['male']['continuous_vocal_signals']
        assert 'usv_event' in signals
        assert set(np.unique(signals['usv_event'])).issubset({0.0, 1.0})

    @pytest.mark.filterwarnings("ignore::astropy.utils.exceptions.AstropyUserWarning")
    def test_categories_rate_skips_non_integer_category(self, tmp_path):
        """A non-integer category label is skipped during per-category signal
        generation."""

        rows = {
            'emitter': ['male', 'male'],
            'start': [2.0, 2.1], 'stop': [2.05, 2.15],
            'usv_category': ['x', 'x'],
            'mask_number': [1, 1],
        }
        out = find_variable_length_bouts(min_vocalizations=2, filter_history=1.0,
                                         mixture_model_params=_mixture_model_params(),
                                         proportion_smoothing_sd=2.0,
                                         vocal_output_type='categories_rate',
                                         **self._kwargs(tmp_path, rows))
        signals = out['sess_E']['male']['continuous_vocal_signals']
        assert not any(k.startswith('usv_cat_') for k in signals)


    @pytest.mark.parametrize('category_column', [None, 'absent_label_column'])
    def test_category_predictors_without_labels_raise(self, tmp_path, category_column):
        """Bout loading with 'categories_rate' and no label column (null, or
        absent from the summary) raises the labels-unavailable error."""

        rows = {
            'emitter': ['male', 'male'],
            'start': [2.0, 2.1],
            'stop': [2.05, 2.15],
        }
        with pytest.raises(ValueError, match=_LABELS_UNAVAILABLE):
            find_variable_length_bouts(min_vocalizations=2, filter_history=1.0,
                                       mixture_model_params=_mixture_model_params(),
                                       vocal_output_type='categories_rate', category_column=category_column,
                                       **self._kwargs(tmp_path, rows))

    def test_pooled_rate_needs_no_category_column(self, tmp_path):
        """Bout loading with 'pooled_rate' runs with a null category column."""

        rows = {
            'emitter': ['male', 'male'],
            'start': [2.0, 2.1],
            'stop': [2.05, 2.15],
        }
        out = find_variable_length_bouts(min_vocalizations=2, filter_history=1.0,
                                         mixture_model_params=_mixture_model_params(),
                                         vocal_output_type='pooled_rate', category_column=None,
                                         **self._kwargs(tmp_path, rows))
        assert set(out['sess_E']['male']['continuous_vocal_signals']) == {'usv_rate'}


class TestBoutOffsetEpochs:
    """``'bout_offset'`` mode: the END of a bout against an interior USV.

    Two bouts under the ~1 s IBI threshold of ``_mixture_model_params``: a short
    one (three USVs over 0.25 s ending at 2.25 s) and a long one (six USVs
    over 1.05 s ending at 7.05 s). Positions are measured from the bout's first
    USV start to the USV's offset.
    """

    def _build(self, tmp_path, rows: dict, n_frames: int = 1000, fps: float = 100.0):
        """
        Description
        -----------
        Builds a one-session tree (``sess_O``) holding a synthetic USV summary and
        returns the common loader kwargs for the ``bout_offset`` tests below.

        Parameters
        ----------
        tmp_path (pathlib.Path)
            The test's temporary directory; the session root is created under it.
        rows (dict)
            Column-name -> value-list mapping for the USV summary, forwarded to
            ``_write_usv_summary``.
        n_frames (int)
            Number of frames in the synthetic behavioural-feature table.
        fps (float)
            Camera frame rate recorded for the session.

        Returns
        -------
        kwargs (dict)
            ``root_directories``, ``mouse_ids_dict``, ``camera_fps_dict`` and
            ``features_dict`` keyed for the single session.
        """
        sess = tmp_path / 'sess_O'
        _write_usv_summary(sess, rows)
        return {
            'root_directories': [str(sess)],
            'mouse_ids_dict': {'sess_O': ['male', 'female']},
            'camera_fps_dict': {'sess_O': fps},
            'features_dict': {'sess_O': _features_df(n_frames)},
        }

    @staticmethod
    def _two_bouts() -> dict:
        """
        Description
        -----------
        The USV summary rows for the class's two male bouts: three USVs starting at
        2.0-2.2 s and six starting at 6.0-7.0 s, each 50 ms long and category 1.

        Returns
        -------
        rows (dict)
            Column-name -> value-list mapping for ``_write_usv_summary``.
        """
        starts = [2.0, 2.1, 2.2, 6.0, 6.2, 6.4, 6.6, 6.8, 7.0]
        return {
            'emitter': ['male'] * len(starts),
            'start': starts,
            'stop': [round(t + 0.05, 6) for t in starts],
            'usv_category': [1] * len(starts),
        }

    def _run(self, tmp_path, rows, **offset_kwargs):
        """
        Description
        -----------
        Runs ``find_onset_epochs`` in ``'bout_offset'`` mode on a one-session tree
        built from ``rows`` with the class's default pairing settings (cross-bout,
        0.5 s minimum singing left, 0.10 s position tolerance, at most 3 negatives
        per bout), any of which ``offset_kwargs`` overrides.

        Parameters
        ----------
        tmp_path (pathlib.Path)
            The test's temporary directory.
        rows (dict)
            Column-name -> value-list mapping for the USV summary.
        **offset_kwargs
            Overrides for the bout-offset pairing keywords.

        Returns
        -------
        male (dict)
            The male's entry of the loader output (``positive_events``,
            ``negative_events`` and the rest).
        """
        kwargs = self._build(tmp_path, rows)
        base = {'negative_scheme': 'cross_bout', 'min_singing_after_negative': 0.5,
                'time_since_bout_onset_tolerance': 0.10, 'max_negatives_per_bout': 3}
        base.update(offset_kwargs)
        return find_onset_epochs(prediction_mode='bout_offset', filter_history=1.0,
                                 usv_bout_time=0.5, min_usv_per_bout=2,
                                 proportion_smoothing_sd=None, mixture_model_params=_mixture_model_params(),
                                 **base, **kwargs)['sess_O']['male']

    def test_cross_bout_pairs_positions_across_bouts_and_drops_unmatched(self, tmp_path):
        """The short bout's end (position 0.25 s) is matched to the long bout's
        interior USV at position 0.25 s (offset 6.25 s, with 0.8 s of singing
        left); the long bout's end (position 1.05 s) finds no candidate in the
        short bout and is dropped, so the groups stay position-matched."""

        out = self._run(tmp_path, self._two_bouts())
        np.testing.assert_allclose(out['positive_events'], [2.25])
        np.testing.assert_allclose(out['negative_events'], [6.25])

    def test_cross_bout_respects_minimum_singing_left(self, tmp_path):
        """Raising the minimum singing left past what any interior USV offers
        leaves no candidates and therefore no pairs."""

        out = self._run(tmp_path, self._two_bouts(), min_singing_after_negative=2.0)
        assert out['positive_events'].size == 0
        assert out['negative_events'].size == 0

    def test_cross_bout_tolerance_and_cap(self, tmp_path):
        """Three short bouts at the same position all want the long bout's
        candidates. With a wide tolerance and no cap all three pair up (to the
        candidates at 0.05, 0.25 and 0.45 s); with a cap of 2 per bout only two
        do, and no negative ever comes from a positive's own bout."""

        starts = [2.0, 2.1, 2.2, 4.0, 4.1, 4.2, 6.0, 6.1, 6.2, 8.0, 8.2, 8.4, 8.6, 8.8, 9.0]
        rows = {'emitter': ['male'] * len(starts), 'start': starts,
                'stop': [round(t + 0.05, 6) for t in starts],
                'usv_category': [1] * len(starts)}
        kwargs = {'n_frames': 1200}
        sess_kwargs = self._build(tmp_path, rows, **kwargs)
        common = {'prediction_mode': 'bout_offset', 'filter_history': 1.0, 'usv_bout_time': 0.5, 'min_usv_per_bout': 2,
                      'proportion_smoothing_sd': None, 'mixture_model_params': _mixture_model_params(),
                      'min_singing_after_negative': 0.5, 'time_since_bout_onset_tolerance': 0.30}
        loose = find_onset_epochs(negative_scheme='cross_bout', max_negatives_per_bout=None, **common, **sess_kwargs)['sess_O']['male']
        assert loose['positive_events'].size == 3
        assert set(np.round(loose['negative_events'], 2)) == {8.05, 8.25, 8.45}
        capped = find_onset_epochs(negative_scheme='cross_bout', max_negatives_per_bout=2, **common, **sess_kwargs)['sess_O']['male']
        assert capped['positive_events'].size == 2
        assert capped['negative_events'].size == 2
        assert np.all(capped['negative_events'] > 8.0)

    def test_within_bout_uses_own_interior_usv_and_skips_short_bouts(self, tmp_path):
        """Within the long bout the negative is the latest interior USV with
        at least 0.5 s of singing left (offset 6.45 s, 0.6 s before the end at
        7.05 s); the short bout holds no such USV and contributes nothing."""

        out = self._run(tmp_path, self._two_bouts(), negative_scheme='within_bout')
        np.testing.assert_allclose(out['positive_events'], [7.05])
        np.testing.assert_allclose(out['negative_events'], [6.45])

    def test_positives_and_negatives_never_share_a_usv(self, tmp_path):
        """Under both negative schemes no USV offset is both a positive and a negative, and
        the groups are equal in size, so the contrast compares disjoint, matched events."""

        for scheme in ('cross_bout', 'within_bout'):
            out = self._run(tmp_path, self._two_bouts(), negative_scheme=scheme)
            assert not set(out['positive_events']) & set(out['negative_events'])
            assert out['positive_events'].size == out['negative_events'].size

    def test_unknown_scheme_raises(self, tmp_path):
        """An unrecognised ``negative_scheme`` raises naming the setting instead of silently
        producing no negatives."""

        with pytest.raises(ValueError, match='negative_scheme'):
            self._run(tmp_path, self._two_bouts(), negative_scheme='sideways')

    def test_female_without_usvs_is_empty_in_both_groups(self, tmp_path):
        """A mouse with no USVs gets empty positive and negative arrays (not a missing key),
        so downstream pooling sees every listed mouse."""

        kwargs = self._build(tmp_path, self._two_bouts())
        out = find_onset_epochs(prediction_mode='bout_offset', filter_history=1.0, usv_bout_time=0.5, min_usv_per_bout=2,
                                proportion_smoothing_sd=None, mixture_model_params=_mixture_model_params(),
                                negative_scheme='cross_bout', min_singing_after_negative=0.5,
                                time_since_bout_onset_tolerance=0.1, max_negatives_per_bout=3, **kwargs)
        assert out['sess_O']['female']['positive_events'].size == 0
        assert out['sess_O']['female']['negative_events'].size == 0


class TestVocalClassTraces:
    """One trace per predictor list, named by what it holds; the target tag; the
    settings checks; and the helpers that recognise vocal columns."""

    @staticmethod
    def _rows() -> pls.DataFrame:
        """
        Description
        -----------
        Five segments at 0.1 s steps: pure USV (category 1), pure squeak, pure USV
        (category 2), overlap (category 1), and an unscored row (null flags). The squeaks'
        own extents differ from their segments: the pure squeak sounds at 0.220-0.225 s
        (frame 22) and the overlap segment's squeak at 0.450-0.455 s (frame 45), after the
        segment's USV onset (0.40 s, frame 40).

        Returns
        -------
        rows (polars.DataFrame)
            The five-row USV summary table.
        """

        return pls.DataFrame({
            'start': [0.10, 0.20, 0.30, 0.40, 0.50],
            'stop': [0.11, 0.21, 0.31, 0.41, 0.51],
            'qlvm_category': [1, None, 2, 1, 1],
            'usv': [True, False, True, True, None],
            'squeak': [False, True, False, True, None],
            'squeak_start': [None, 0.220, None, 0.450, None],
            'squeak_end': [None, 0.225, None, 0.455, None],
        })

    @pytest.mark.parametrize(('classes', 'name'), [
        (['usv'], 'usv'), (['usv', 'overlap'], 'usv'),
        (['squeak'], 'squeak'), (['squeak', 'overlap'], 'squeak'),
        (['overlap'], 'overlap'),
        (['usv', 'squeak'], 'all_vocalizations'), (['usv', 'squeak', 'overlap'], 'all_vocalizations'),
    ])
    def test_trace_name(self, classes, name):
        """A list holding USVs is a USV trace, one holding squeaks a squeak trace, overlap
        alone an overlap trace, and USVs with squeaks every vocalization."""

        assert vocal_trace_name(classes) == name

    def test_pooled_traces_hold_their_classes_only(self):
        """``pooled_binary`` marks exactly the frames of each list's rows (100 fps: frame
        10 is 0.10 s), a pure squeak at its own extent (frame 22, not its segment's frame
        20); overlap rows count in a USV list at their segment (frame 40); unscored rows in
        none."""

        signals = build_vocal_predictor_signals(
            self._rows(), [['usv', 'overlap'], ['squeak']], 'pooled_binary', 'qlvm_category',
            duration_frames=60, fps=100.0, smooth_sd=None, source='s')
        assert sorted(signals) == ['squeak_event', 'usv_event']
        assert np.flatnonzero(signals['usv_event']).tolist() == [10, 30, 40]
        assert np.flatnonzero(signals['squeak_event']).tolist() == [22]

    def test_categories_split_squeak_free_lists_only(self):
        """``categories_rate`` splits a list without pure squeaks by QLVM category and
        gives a list with pure squeaks one pooled trace, every squeak at its own extent
        (the overlap segment's squeak at frame 45 in a squeak list, its USV at frame 40 in
        a USV list); ``all_rate`` adds the pooled trace to the split one; an empty list
        still gives an all-zero pooled trace."""

        rows = self._rows()
        cats = build_vocal_predictor_signals(rows, [['usv', 'overlap'], ['squeak', 'overlap']], 'categories_rate',
                                             'qlvm_category', duration_frames=60, fps=100.0, smooth_sd=None, source='s')
        assert sorted(cats) == ['squeak_rate', 'usv_cat_1', 'usv_cat_2']
        assert np.flatnonzero(cats['usv_cat_1']).tolist() == [10, 40]
        assert np.flatnonzero(cats['squeak_rate']).tolist() == [22, 45]
        both = build_vocal_predictor_signals(rows, [['usv']], 'all_rate', 'qlvm_category',
                                             duration_frames=60, fps=100.0, smooth_sd=None, source='s')
        assert sorted(both) == ['usv_cat_1', 'usv_cat_2', 'usv_rate']
        empty = build_vocal_predictor_signals(rows.filter(pls.col('squeak') == False), [['squeak']], 'pooled_rate',  # noqa: E712
                                              'qlvm_category', duration_frames=60, fps=100.0, smooth_sd=None, source='s')
        assert list(empty) == ['squeak_rate']
        assert not empty['squeak_rate'].any()
        assert build_vocal_predictor_signals(rows, [['usv']], None, None, 60, 100.0, None, 's') == {}

    def test_overlap_alone_spans_the_union(self):
        """An ``["overlap"]`` list asks about neither class alone, so the overlap segment
        spans the union of its segment and its squeak extent (0.40-0.455 s, frames 40-45)."""

        signals = build_vocal_predictor_signals(
            self._rows(), [['overlap']], 'pooled_binary', 'qlvm_category',
            duration_frames=60, fps=100.0, smooth_sd=None, source='s')
        assert np.flatnonzero(signals['overlap_event']).tolist() == [40, 41, 42, 43, 44, 45]

    def test_squeak_list_without_the_extent_raises(self):
        """A list that times squeaks by their extent fails loudly on a summary without
        ``squeak_start`` / ``squeak_end``; a USV-only list does not need them."""

        rows = self._rows().drop('squeak_start', 'squeak_end')
        with pytest.raises(KeyError, match="squeak_start"):
            build_vocal_predictor_signals(rows, [['squeak']], 'pooled_binary', 'qlvm_category',
                                          duration_frames=60, fps=100.0, smooth_sd=None, source='s')
        signals = build_vocal_predictor_signals(rows, [['usv', 'overlap']], 'pooled_binary', 'qlvm_category',
                                                duration_frames=60, fps=100.0, smooth_sd=None, source='s')
        assert np.flatnonzero(signals['usv_event']).tolist() == [10, 30, 40]

    def test_predictor_lists_are_checked(self):
        """Lists may share classes, but two lists giving one trace name, an unknown or
        repeated class, and an empty list are refused."""

        validate_vocal_class_settings(None, [['usv', 'overlap'], ['squeak', 'overlap']], None)
        with pytest.raises(ValueError, match="give the trace name"):
            validate_vocal_class_settings(None, [['usv'], ['usv', 'overlap']], None)
        with pytest.raises(ValueError, match="distinct classes"):
            validate_vocal_class_settings(None, [['usv', 'chirp']], None)
        with pytest.raises(ValueError, match="non-empty list"):
            validate_vocal_class_settings(None, [[]], None)
        with pytest.raises(ValueError, match="non-empty list"):
            validate_vocal_class_settings(None, [], None)
        with pytest.raises(ValueError, match="give the trace name"):
            require_labels_for_vocal_predictors({'vocal_predictor_type': 'pooled_rate',
                                                 'vocal_predictor_classes': [['squeak'], ['squeak', 'overlap']],
                                                 'usv_category_column_name': None})

    def test_target_tag(self):
        """The default ["usv"] adds nothing to a file name; any other list adds its
        classes in a fixed order."""

        assert onset_target_tag(['usv']) == ''
        assert onset_target_tag(['overlap', 'usv']) == '_usv+overlap'
        assert onset_target_tag(['squeak']) == '_squeak'

    @pytest.mark.parametrize(('column', 'vocal', 'pooled'), [
        ('other.usv_rate', True, True), ('self.squeak_event', True, True),
        ('other.all_vocalizations_rate', True, True), ('other.overlap_cat_3', True, False),
        ('usv_cat_0', True, False), ('other.speed', False, False), ('other.usv_cat_x', False, False),
        ('other.squeaky_rate', False, False),
    ])
    def test_vocal_column_recognition(self, column, vocal, pooled):
        """Vocal columns are '<name>_rate' / '<name>_event' / '<name>_cat_<int>' with a
        known trace name; only the first two are pooled."""

        assert is_vocal_signal(column) is vocal
        assert is_pooled_vocal_signal(column) is pooled


class TestCategoryLabelRequirement:
    """The settings-level label checks the pipelines run before loading anything."""

    @pytest.mark.parametrize('category_column', [None, ''])
    def test_null_column_raises_naming_the_setting(self, category_column):
        """A null / empty column raises the labels-unavailable error naming the
        setting and the purpose."""

        with pytest.raises(ValueError, match=_LABELS_UNAVAILABLE) as excinfo:
            require_usv_category_column(category_column, 'The multinomial USV category model')
        assert 'vocal_features.usv_category_column_name' in str(excinfo.value)
        assert 'The multinomial USV category model' in str(excinfo.value)

    def test_column_absent_from_summary_raises(self):
        """A set column missing from one summary's columns raises, naming the file."""

        with pytest.raises(ValueError, match=r'absent from s\.csv'):
            require_usv_category_column('absent_label_column', 'x', summary_columns=['qlvm1'], source='s.csv')

    def test_present_column_passes(self):
        """A set column present in the summary passes silently."""

        require_usv_category_column('qlvm_category', 'x', summary_columns=['qlvm_category'], source='s.csv')

    @pytest.mark.parametrize(('predictor_type', 'raises'), [
        ('categories_rate', True), ('all_rate', True),
        ('pooled_rate', False), ('pooled_binary', False), (None, False),
    ])
    def test_vocal_predictor_settings_check(self, predictor_type, raises):
        """Only the per-category predictor modes need a label column."""

        voc = {'vocal_predictor_type': predictor_type, 'vocal_predictor_classes': [['usv']],
               'usv_category_column_name': None}
        if raises:
            with pytest.raises(ValueError, match=_LABELS_UNAVAILABLE):
                require_labels_for_vocal_predictors(voc)
        else:
            require_labels_for_vocal_predictors(voc)

    def test_shipped_vocal_defaults_use_the_qlvm_category(self):
        """The usv_summary.csv files carry the QLVM category (qlvm_category, R-1..R-k of
        the regular map), so the shipped settings build the per-category rate
        predictors over qlvm_category, and the settings-level
        label check passes on them; the check (and its error) still guards a summary
        that genuinely lacks the column."""

        with _SETTINGS_JSON.open() as fh:
            voc = json.load(fh)['vocal_features']
        assert voc['vocal_predictor_type'] == 'categories_rate'
        assert voc['usv_category_column_name'] == 'qlvm_category'
        require_labels_for_vocal_predictors(voc)
        with pytest.raises(ValueError, match=r'absent from s\.csv'):
            require_usv_category_column(voc['usv_category_column_name'], 'x', summary_columns=['qlvm1'], source='s.csv')


# Exact-value tests of the pure helpers: tiling boundaries, occupancy rasterisation,
# smoothing edges, bout grouping and the bout-offset pairing rules.


def _write_behavior_session(session_root, with_features: bool, with_track: bool, csv_sep: str = ',') -> None:
    """
    Description
    -----------
    Writes a synthetic behavioral session under ``<session_root>/video`` for
    ``load_behavioral_feature_data``: a three-row, two-column features CSV and/or a
    metric track H5 carrying a 150 fps ``recording_frame_rate`` and the track names
    ``male`` / ``female``. Either file can be left out to build the "missing input"
    sessions the loader must skip.

    Parameters
    ----------
    session_root (pathlib.Path)
        The session ROOT directory; the ``video`` subdir is created.
    with_features (bool)
        Whether to write the ``*_points3d_translated_rotated_metric_behavioral_features.csv``.
    with_track (bool)
        Whether to write the ``*_points3d_translated_rotated_metric.h5``.
    csv_sep (str)
        Field separator of the features CSV.

    Returns
    -------
    None
    """

    video_dir = session_root / 'video'
    video_dir.mkdir(parents=True)
    if with_features:
        pls.DataFrame({'speed': [0.1, 0.2, 0.3], 'angle': [1.0, 2.0, 3.0]}).write_csv(
            file=video_dir / '20230101000000_points3d_translated_rotated_metric_behavioral_features.csv', separator=csv_sep)
    if with_track:
        with h5py.File(name=video_dir / '20230101000000_points3d_translated_rotated_metric.h5', mode='w') as h5f:
            h5f['recording_frame_rate'] = 150.0
            h5f['track_names'] = np.array([b'male', b'female'])


class TestLoadBehavioralFeatureDataSkips:
    """A session missing either behavioural input is skipped with a warning while the loader
    carries on, so one incomplete session cannot abort a cohort load."""

    def test_missing_inputs_skip_only_their_session(self, tmp_path, capsys):
        """A session without the features CSV and a session without the track H5 are
        each skipped with a warning naming them (either file missing is enough, the
        glob lookup never raises), and the loader carries on to the next session: a
        valid session listed after both is still read, with its ';'-separated CSV
        split by ``csv_sep`` into its two columns."""

        no_features = tmp_path / 'sess_no_features'
        no_track = tmp_path / 'sess_no_track'
        valid = tmp_path / 'sess_valid'
        _write_behavior_session(no_features, with_features=False, with_track=True)
        _write_behavior_session(no_track, with_features=True, with_track=False)
        _write_behavior_session(valid, with_features=True, with_track=True, csv_sep=';')

        beh, fps, names = load_behavioral_feature_data(
            behavior_file_paths=[str(no_features), str(no_track), str(valid)], csv_sep=';')

        assert list(beh) == ['sess_valid']
        assert list(fps) == ['sess_valid']
        assert list(names) == ['sess_valid']
        assert beh['sess_valid'].columns == ['speed', 'angle']
        assert beh['sess_valid'].shape == (3, 2)
        assert fps['sess_valid'] == 150.0
        printed = capsys.readouterr().out
        assert 'Missing behavioral feature/track file for sess_no_features' in printed
        assert 'Missing behavioral feature/track file for sess_no_track' in printed


class TestUsvSummaryTableWarning:
    """The USV summary reader honours ``csv_sep`` and warns about missing emitters only when
    an emitter column exists, has rows and is entirely null."""

    def test_separator_and_emitter_warning_cases(self, tmp_path, capsys):
        """The summary is split by ``csv_sep`` (a ';' file gives its three columns), and
        the no-emitter warning fires only when an ``emitter`` column exists, the table
        has rows and every emitter is null: an assigned emitter, a table without the
        column and a header-only table all read silently."""

        assigned = tmp_path / 'assigned_usv_summary.csv'
        pls.DataFrame({'usv_id': ['0001', '0002'], 'start': [1.0, 2.0], 'emitter': ['male', None]}).write_csv(
            assigned, separator=';')
        table = read_usv_summary_table(assigned, ';')
        assert table.columns == ['usv_id', 'start', 'emitter']
        assert table['usv_id'].to_list() == ['0001', '0002']

        no_column = tmp_path / 'no_column_usv_summary.csv'
        pls.DataFrame({'usv_id': ['0001'], 'start': [1.0]}).write_csv(no_column)
        assert read_usv_summary_table(no_column, ',').height == 1

        header_only = tmp_path / 'header_only_usv_summary.csv'
        header_only.write_text('usv_id,start,emitter\n')
        assert read_usv_summary_table(header_only, ',').height == 0

        assert 'no assigned emitter' not in capsys.readouterr().out


class TestVocalSettingMessages:
    """The vocal-predictor settings errors name the exact offending lists or values, so a
    misconfiguration is diagnosable from the message alone."""

    def test_trace_name_clash_names_the_clashing_lists(self):
        """The duplicate-trace error names exactly the lists that clash (the two USV
        lists), not the list that gives its own trace name (the squeak list)."""

        with pytest.raises(ValueError, match=re.escape("entries [['usv'], ['usv', 'overlap']] all give the trace name(s) ['usv']")):
            validate_vocal_class_settings(None, [['usv'], ['usv', 'overlap'], ['squeak']], None)

    @pytest.mark.parametrize('column', ['other.x.usv_rate', 'a.b.squeak_event', 'a.b.usv_cat_1'])
    def test_only_the_first_dot_separates_the_animal(self, column):
        """Only the text before the FIRST '.' is the animal: ``'other.x.usv_rate'`` has
        the bare signal ``'x.usv_rate'``, which is no vocal trace."""

        assert is_vocal_signal(column) is False
        assert is_pooled_vocal_signal(column) is False

    def test_unknown_predictor_type_is_named(self):
        """An unknown ``vocal_predictor_type`` raises naming the bad value."""

        rows = pls.DataFrame({'start': [0.1], 'stop': [0.2], 'usv': [True], 'squeak': [False]})
        with pytest.raises(ValueError, match=re.escape("Unknown vocal_features.vocal_predictor_type 'bogus'")):
            build_vocal_predictor_signals(rows, [['usv']], 'bogus', None, 10, 10.0, None, 's')

    def test_label_requirement_names_the_predictor_type(self):
        """The settings-level label check names the predictor type that needs labels."""

        with pytest.raises(ValueError, match=re.escape("vocal_features.vocal_predictor_type 'categories_rate' needs")):
            require_labels_for_vocal_predictors({'vocal_predictor_type': 'categories_rate',
                                                 'vocal_predictor_classes': [['usv']],
                                                 'usv_category_column_name': None})


class TestPredictorSignalsExact:
    """``build_vocal_predictor_signals`` produces exact per-frame values for each predictor
    type on a hand-placed USV table."""

    @staticmethod
    def _rows() -> pls.DataFrame:
        """
        Description
        -----------
        Four pure USVs at 100 fps frames 10, 20, 30 and 40: categories 1, null, 2 and
        1; no squeaks.

        Returns
        -------
        rows (polars.DataFrame)
            The four-row USV summary table.
        """

        return pls.DataFrame({
            'start': [0.10, 0.20, 0.30, 0.40],
            'stop': [0.11, 0.21, 0.31, 0.41],
            'qlvm_category': [1, None, 2, 1],
            'usv': [True, True, True, True],
            'squeak': [False, False, False, False],
            'squeak_start': pls.Series([None] * 4, dtype=pls.Float64),
            'squeak_end': pls.Series([None] * 4, dtype=pls.Float64),
        })

    @pytest.mark.filterwarnings("ignore::astropy.utils.exceptions.AstropyUserWarning")
    def test_rate_traces_are_smoothed_with_the_given_sd(self):
        """``all_rate`` smooths the pooled trace and every category trace with the
        configured sd: each equals ``_generate_vocal_trace`` of its rows at that sd. A
        null category is skipped without ending the category loop (categories 1 and 2
        still get their traces although the null one sorts first)."""

        rows = self._rows()
        signals = build_vocal_predictor_signals(rows, [['usv']], 'all_rate', 'qlvm_category',
                                                duration_frames=60, fps=100.0, smooth_sd=2.0, source='s')
        assert sorted(signals) == ['usv_cat_1', 'usv_cat_2', 'usv_rate']
        np.testing.assert_allclose(signals['usv_rate'], _generate_vocal_trace(
            np.array([0.10, 0.20, 0.30, 0.40]), np.array([0.11, 0.21, 0.31, 0.41]), 60, 100.0, smooth_sd=2.0))
        np.testing.assert_allclose(signals['usv_cat_1'], _generate_vocal_trace(
            np.array([0.10, 0.40]), np.array([0.11, 0.41]), 60, 100.0, smooth_sd=2.0))
        np.testing.assert_allclose(signals['usv_cat_2'], _generate_vocal_trace(
            np.array([0.30]), np.array([0.31]), 60, 100.0, smooth_sd=2.0))
        assert signals['usv_rate'].max() < 1.0

    def test_single_row_still_gets_its_category_trace(self):
        """One USV is enough for a per-category trace."""

        rows = self._rows().head(1)
        signals = build_vocal_predictor_signals(rows, [['usv']], 'categories_rate', 'qlvm_category',
                                                duration_frames=60, fps=100.0, smooth_sd=None, source='s')
        assert sorted(signals) == ['usv_cat_1']
        assert np.flatnonzero(signals['usv_cat_1']).tolist() == [10]

    def test_errors_name_the_source(self):
        """A table without the class flags, or a squeak list without the squeak extent,
        raises naming the source the caller passed."""

        with pytest.raises(KeyError, match="summary_X has no"):
            build_vocal_predictor_signals(self._rows().drop('usv', 'squeak'), [['usv']], 'pooled_binary', None,
                                          60, 100.0, None, 'summary_X')
        with pytest.raises(KeyError, match="summary_X has no"):
            build_vocal_predictor_signals(self._rows().drop('squeak_start', 'squeak_end'), [['squeak']],
                                          'pooled_binary', None, 60, 100.0, None, 'summary_X')


class TestCleanTiledEpochsExact:
    """Exact tiling boundaries of the clean-epoch tiler: unsorted, overlapping forbidden zones
    merge, and every clean gap is tiled from its start."""

    def test_unsorted_overlapping_zones_merge_and_every_gap_is_tiled(self):
        """Unsorted USVs at 5.0, 1.5, 2.0 and 8.0 s (fh 1 s, 12 s session) give the
        forbidden window-end zones (1.5, 3.2), (5.0, 6.5) and (8.0, 9.1) after sorting
        and merging the two overlapping early ones; the clean zones are tiled from
        their starts: [1.0, 1.5] -> 1.0, [3.2, 5.0] -> 3.2, 4.2, [6.5, 8.0] -> 6.5, 7.5
        and [9.1, 12] -> 9.1, 10.1, 11.1."""

        ts = _get_clean_tiled_epochs(np.array([5.0, 1.5, 2.0, 8.0]), np.array([5.5, 1.6, 2.2, 8.1]),
                                     filter_history=1.0, session_duration_sec=12.0)
        np.testing.assert_allclose(ts, [1.0, 3.2, 4.2, 6.5, 7.5, 9.1, 10.1, 11.1])

    @pytest.mark.parametrize(('filter_history', 'duration', 'expected'), [
        (1.0, 2.0, [1.0, 2.0]),             # the first window starts at the session start
        (1.0, 3.0, [1.0, 2.0, 3.0]),        # the session end is a valid window end
        (0.5, 2.0, [0.5, 1.0, 1.5, 2.0]),   # windows step by filter_history, not by 1 s
    ])
    def test_tiling_boundaries_without_usvs(self, filter_history, duration, expected):
        """Without USVs the clean zone is [filter_history, duration] and its windows
        end at every multiple of ``filter_history`` from ``filter_history`` up to
        and including the session end."""

        ts = _get_clean_tiled_epochs(np.array([]), np.array([]),
                                     filter_history=filter_history, session_duration_sec=duration)
        np.testing.assert_allclose(ts, expected)


class TestHistoryIsDirty:
    """``_history_is_dirty`` treats the window edge as exclusive and ignores the onset's own
    vocalization, so positives and negatives share one boundary convention."""

    def test_window_edge_is_exclusive_and_the_onset_itself_is_excluded(self):
        """A reference vocalization ending exactly at ``onset - filter_history`` abuts
        the window without touching it (the negatives' forbidden zone of window ends is
        open at stop + filter_history, so both classes share the boundary); one ending
        just after it is in the window; one starting AT the onset is the onset's own."""

        assert _history_is_dirty(3.0, np.array([1.5]), np.array([2.0]), 1.0) is False
        assert _history_is_dirty(3.0, np.array([1.5]), np.array([2.01]), 1.0) is True
        assert _history_is_dirty(3.0, np.array([1.5]), np.array([1.99]), 1.0) is False
        assert _history_is_dirty(3.0, np.array([3.0]), np.array([3.5]), 1.0) is False

    def test_negatives_start_where_a_clean_positive_may(self):
        """
        Regression: one USV at 10.0-10.1 s, fh 1 s. A positive onset is clean
        from 11.1 s on (``_history_is_dirty``), and the first negative window after the
        USV must end there too, not one more ``filter_history`` later (12.1 s).
        """

        starts, stops = np.array([10.0]), np.array([10.1])
        negatives = _get_clean_tiled_epochs(starts, stops, 1.0, 20.0)
        after = negatives[negatives > 10.1]

        assert after[0] == pytest.approx(11.1)
        assert _history_is_dirty(11.1, starts, stops, 1.0) is False
        assert _history_is_dirty(11.09, starts, stops, 1.0) is True
        for t in negatives:
            assert _history_is_dirty(t, starts, stops, 1.0) is False


class TestOccupancyAndSmoothing:
    """Occupancy rasterisation clips events at the session edges and smoothing preserves the
    trace at its edges, so no event leaks or wraps around the trace."""

    def test_event_starting_before_the_session_is_clipped_to_frame_zero(self):
        """An event from -0.5 s to 0.3 s (10 fps) covers frames 0..2: its start index is
        clipped to 0 instead of wrapping to the end of the trace."""

        trace = _build_binary_occupancy(np.array([-0.5]), np.array([0.3]), duration_frames=10, fps=10.0)
        assert trace.tolist() == [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def test_smoothing_extends_the_edges_and_keeps_nans(self):
        """The edges are extended (an all-ones trace stays all ones, no zero padding),
        and a NaN frame stays NaN while its neighbours are smoothed around it."""

        np.testing.assert_allclose(_smooth_occupancy(np.ones(10), 2.0), np.ones(10))
        trace = np.zeros(10)
        trace[3] = np.nan
        trace[5] = 1.0
        smoothed = _smooth_occupancy(trace, 1.0)
        assert np.isnan(smoothed[3])
        assert np.isfinite(np.delete(smoothed, 3)).all()
        assert smoothed[5] < 1.0

    @pytest.mark.filterwarnings("ignore::astropy.utils.exceptions.AstropyUserWarning")
    def test_zero_sd_is_binary_and_sub_unit_sd_smooths(self):
        """``smooth_sd=0`` returns the binary trace; any positive sd, also one below 1
        frame, smooths it."""

        starts, stops = np.array([1.0]), np.array([1.05])
        binary = _generate_vocal_trace(starts, stops, 40, 10.0)
        np.testing.assert_array_equal(_generate_vocal_trace(starts, stops, 40, 10.0, smooth_sd=0), binary)
        half = _generate_vocal_trace(starts, stops, 40, 10.0, smooth_sd=0.5)
        np.testing.assert_allclose(half, _smooth_occupancy(binary, 0.5))
        assert half[10] < 1.0


class TestGroupVocalizationsIntoBouts:
    """Vocalization-to-bout grouping returns integer index arrays and splits a bout at a gap
    equal to the threshold."""

    def test_empty_single_and_threshold_gap(self):
        """No vocalizations give empty INTEGER index arrays (usable as indices); one
        vocalization is one bout (0, 0); a gap exactly equal to the threshold starts a
        new bout."""

        empty_starts, empty_ends = _group_vocalizations_into_bouts(np.array([]), np.array([]), 1.0)
        assert empty_starts.dtype.kind == 'i'
        assert empty_ends.dtype.kind == 'i'
        assert empty_starts.size == 0
        assert empty_ends.size == 0
        one_starts, one_ends = _group_vocalizations_into_bouts(np.array([2.0]), np.array([2.5]), 1.0)
        assert one_starts.tolist() == [0]
        assert one_ends.tolist() == [0]
        split_starts, split_ends = _group_vocalizations_into_bouts(np.array([0.0, 1.5, 1.75]),
                                                                   np.array([0.5, 1.6, 2.0]), 1.0)
        assert split_starts.tolist() == [0, 1]
        assert split_ends.tolist() == [0, 2]


class TestBoutOffsetPairingRules:
    """``_bout_offset_events`` on explicit bout indices, with binary-exact times where
    a boundary is tested."""

    def test_min_usv_per_bout_counts_each_bout(self):
        """With a three-vocalization floor the three-vocalization bout (0..2) counts and
        the two-vocalization bout (3..4) does not: within the first, the negative is the
        latest interior offset (0.75 s) and the positive its end (1.25 s)."""

        starts = np.array([0.0, 0.5, 1.0, 10.0, 10.5])
        stops = np.array([0.25, 0.75, 1.25, 10.25, 10.75])
        positives, negatives = _bout_offset_events(starts, stops, np.array([0, 3]), np.array([2, 4]),
                                                   min_usv_per_bout=3, negative_scheme='within_bout',
                                                   min_singing_after_negative=0.0,
                                                   time_since_bout_onset_tolerance=None, max_negatives_per_bout=None)
        assert positives.tolist() == [1.25]
        assert negatives.tolist() == [0.75]

    def test_within_bout_minimum_singing_is_inclusive(self):
        """An interior offset exactly ``min_singing_after_negative`` before the bout's end
        (1.0 - 0.5 = 0.5 s) is eligible and, being the latest, is the negative."""

        positives, negatives = _bout_offset_events(np.array([0.0, 0.3, 0.75]), np.array([0.25, 0.5, 1.0]),
                                                   np.array([0]), np.array([2]), min_usv_per_bout=2,
                                                   negative_scheme='within_bout', min_singing_after_negative=0.5,
                                                   time_since_bout_onset_tolerance=None, max_negatives_per_bout=None)
        assert positives.tolist() == [1.0]
        assert negatives.tolist() == [0.5]

    def test_cross_bout_singing_and_tolerance_bounds_are_inclusive(self):
        """Bout A's interior offset at 0.5 s has exactly 0.5 s of singing left and sits at
        position 0.5 s, exactly bout B's end position, so with a zero tolerance it is
        B's negative; A's own end (position 1.0 s) finds no partner in B."""

        starts = np.array([0.0, 0.25, 0.5, 10.0, 10.25])
        stops = np.array([0.125, 0.5, 1.0, 10.125, 10.5])
        positives, negatives = _bout_offset_events(starts, stops, np.array([0, 3]), np.array([2, 4]),
                                                   min_usv_per_bout=2, negative_scheme='cross_bout',
                                                   min_singing_after_negative=0.5,
                                                   time_since_bout_onset_tolerance=0.0, max_negatives_per_bout=None)
        assert positives.tolist() == [10.5]
        assert negatives.tolist() == [0.5]

    @pytest.mark.parametrize(('scheme', 'expected_positives', 'expected_negatives'), [
        ('cross_bout', [10.3], [0.1]),
        ('within_bout', [1.1], [0.6]),
    ])
    def test_candidates_come_from_their_own_bout_only(self, scheme, expected_positives, expected_negatives):
        """Bout B (two vocalizations 0.2 s apart) has no interior vocalization with 0.5 s
        of singing left, so it offers no negative: under ``cross_bout`` its end pairs
        with A's nearest candidate (0.1 s) and A's end stays unpaired; under
        ``within_bout`` only A contributes. B never borrows A's vocalizations."""

        starts = np.array([0.0, 0.5, 1.0, 10.0, 10.2])
        stops = np.array([0.1, 0.6, 1.1, 10.1, 10.3])
        positives, negatives = _bout_offset_events(starts, stops, np.array([0, 3]), np.array([2, 4]),
                                                   min_usv_per_bout=2, negative_scheme=scheme,
                                                   min_singing_after_negative=0.5,
                                                   time_since_bout_onset_tolerance=100.0, max_negatives_per_bout=None)
        np.testing.assert_allclose(np.sort(positives), expected_positives)
        np.testing.assert_allclose(np.sort(negatives), expected_negatives)

    def test_cross_bout_takes_the_nearest_candidate(self):
        """Bout B ends at position 0.55 s; of A's candidates (0.1 s and 0.6 s) the nearer
        is 0.6 s, not the first. A's end then takes B's only candidate (10.05 s)."""

        starts = np.array([0.0, 0.5, 1.0, 10.0, 10.5])
        stops = np.array([0.1, 0.6, 1.1, 10.05, 10.55])
        positives, negatives = _bout_offset_events(starts, stops, np.array([0, 3]), np.array([2, 4]),
                                                   min_usv_per_bout=2, negative_scheme='cross_bout',
                                                   min_singing_after_negative=0.4,
                                                   time_since_bout_onset_tolerance=100.0, max_negatives_per_bout=None)
        np.testing.assert_allclose(np.sort(positives), [1.1, 10.55])
        np.testing.assert_allclose(np.sort(negatives), [0.6, 10.05])

    def test_cross_bout_with_a_single_candidate(self):
        """One candidate in the whole session (A's first offset, the only one with 0.8 s
        of singing left) is still used."""

        starts = np.array([0.0, 0.5, 1.0, 10.0, 10.2])
        stops = np.array([0.1, 0.6, 1.1, 10.1, 10.3])
        positives, negatives = _bout_offset_events(starts, stops, np.array([0, 3]), np.array([2, 4]),
                                                   min_usv_per_bout=2, negative_scheme='cross_bout',
                                                   min_singing_after_negative=0.8,
                                                   time_since_bout_onset_tolerance=100.0, max_negatives_per_bout=None)
        np.testing.assert_allclose(positives, [10.3])
        np.testing.assert_allclose(negatives, [0.1])

    def test_unmatched_positive_does_not_stop_later_ones(self):
        """Positions in order: B (0.3 s) finds no candidate within 0.05 s and is skipped;
        C (1.1 s) still pairs with A's interior offset at position 1.1 s (1.1 s); A
        (1.6 s) finds none."""

        starts = np.array([0.0, 0.5, 1.0, 1.5, 10.0, 10.2, 20.0, 20.5, 21.0])
        stops = np.array([0.1, 0.6, 1.1, 1.6, 10.1, 10.3, 20.1, 20.6, 21.1])
        positives, negatives = _bout_offset_events(starts, stops, np.array([0, 4, 6]), np.array([3, 5, 8]),
                                                   min_usv_per_bout=2, negative_scheme='cross_bout',
                                                   min_singing_after_negative=0.5,
                                                   time_since_bout_onset_tolerance=0.05, max_negatives_per_bout=None)
        np.testing.assert_allclose(positives, [21.1])
        np.testing.assert_allclose(negatives, [1.1])


# find_onset_epochs: exact event sets, loop control, sex-specific thresholds and messages


def _onset_kwargs(tmp_path, rows: dict, session: str = 'sess_B', n_frames: int = 600, fps: float = 100.0) -> dict:
    """
    Description
    -----------
    Writes one synthetic session ``<tmp_path>/<session>`` (male and female) and returns
    the session kwargs every loader takes: ``root_directories``, ``mouse_ids_dict``,
    ``camera_fps_dict`` and ``features_dict``.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Parent directory of the session.
    rows (dict)
        USV summary columns (see ``_write_usv_summary``).
    session (str)
        Session id (the directory name).
    n_frames (int)
        Session length in camera frames.
    fps (float)
        Camera frame rate.

    Returns
    -------
    kwargs (dict)
        The four session kwargs.
    """

    sess = tmp_path / session
    _write_usv_summary(sess, rows)
    return {
        'root_directories': [str(sess)],
        'mouse_ids_dict': {session: ['male', 'female']},
        'camera_fps_dict': {session: fps},
        'features_dict': {session: _features_df(n_frames)},
    }


def _drop_summary_columns(tmp_path, session: str, *columns: str) -> None:
    """
    Description
    -----------
    Removes columns from a session's synthetic USV summary written by
    ``_write_usv_summary`` (e.g. the class flags or the squeak extent) to build a
    summary an older pipeline stage would have left.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Parent directory of the session.
    session (str)
        Session id (the directory name).
    *columns (str)
        Columns to drop.

    Returns
    -------
    None
    """

    csv_path = tmp_path / session / 'audio' / f'{session}_usv_summary.csv'
    pls.read_csv(csv_path).drop(*columns).write_csv(csv_path)


# The sampling settings the exact-value onset tests share (overridden per test where needed).
_ONSET_COMMON = {'filter_history': 1.0, 'usv_bout_time': 0.5, 'min_usv_per_bout': 2, 'proportion_smoothing_sd': None}


class TestOnsetEpochsExact:
    """``find_onset_epochs`` exact event sets, loop control, sex-specific thresholds and
    console messages on hand-placed USVs."""


    def test_bout_onset_negatives_are_exact(self, tmp_path):
        """Three male USVs from 2.0 to 2.25 s (6 s session, fh 1 s): the forbidden zone
        of window ends is (2.0, 3.25), so the tiled windows end at 1.0, 2.0, 3.25, 4.25
        and 5.25 s, and the one at 2.0 s is dropped because a USV starts within its
        0.5 s future. The silent female tiles the whole session: 1, 2, 3, 4, 5, 6 s."""

        rows = {'emitter': ['male'] * 3, 'start': [2.0, 2.1, 2.2], 'stop': [2.05, 2.15, 2.25]}
        out = find_onset_epochs(prediction_mode='bout_onset', mixture_model_params=_mixture_model_params(),
                                **_ONSET_COMMON, **_onset_kwargs(tmp_path, rows))['sess_B']
        np.testing.assert_allclose(out['male']['positive_events'], [2.0])
        np.testing.assert_allclose(out['male']['negative_events'], [1.0, 3.25, 4.25, 5.25])
        np.testing.assert_allclose(out['female']['negative_events'], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    def test_bout_onset_rejections_do_not_stop_the_bout_loop(self, tmp_path):
        """Four male bouts in time order (1 s threshold): one starting exactly at
        ``filter_history`` (rejected, the bound is exclusive), a one-USV bout (below
        the floor), one whose history holds a vocalization assigned to no animal
        (dirty), and a clean one at 7.0 s, which is the only positive."""

        rows = {
            'emitter': ['male', 'male', 'male', None, 'male', 'male', 'male', 'male'],
            'start': [1.0, 1.1, 3.0, 4.6, 5.0, 5.1, 7.0, 7.1],
            'stop': [1.05, 1.15, 3.05, 4.65, 5.05, 5.15, 7.05, 7.15],
        }
        out = find_onset_epochs(prediction_mode='bout_onset', mixture_model_params=_mixture_model_params(),
                                **_ONSET_COMMON, **_onset_kwargs(tmp_path, rows, n_frames=900))
        np.testing.assert_allclose(out['sess_B']['male']['positive_events'], [7.0])

    def test_individual_mode_bound_and_negatives(self, tmp_path):
        """'individual' mode excludes an onset exactly at ``filter_history`` (1.0 s) and
        keeps 2.0 s; the forbidden zone of window ends (1.0, 3.05) leaves windows ending
        at 1.0 s (dropped: the 1.0 s USV starts in its future), 3.05, 4.05 and 5.05 s.
        Without a target category it needs no label column."""

        rows = {'emitter': ['male', 'male'], 'start': [1.0, 2.0], 'stop': [1.05, 2.05]}
        out = find_onset_epochs(prediction_mode='individual', mixture_model_params=_mixture_model_params(),
                                category_column=None, target_category=None,
                                **_ONSET_COMMON, **_onset_kwargs(tmp_path, rows))
        male = out['sess_B']['male']
        np.testing.assert_allclose(male['positive_events'], [2.0])
        np.testing.assert_allclose(male['negative_events'], [3.05, 4.05, 5.05])

    @pytest.mark.parametrize(('n_frames', 'fps', 'filter_history', 'start', 'stop', 'positives', 'negatives'), [
        (600, 100.0, 1.0, 2.0, 2.9, [2.0], [1.0, 3.0, 4.0, 5.0]),
        (21, 10.0, 0.5, 1.0, 1.04, [1.0], [0.5, 1.5, 2.0]),
        (15, 10.0, 1.0, 1.0, 1.05, [1.0], []),
    ])
    def test_state_mode_grid(self, tmp_path, n_frames, fps, filter_history, start, stop, positives, negatives):
        """'state' mode samples every ``filter_history`` from ``filter_history`` on, keeps
        the grid points whose frame lies in the session (the last frame, 20 of 21,
        included) and labels them by the target occupancy; a grid of a single point
        still yields its label."""

        rows = {'emitter': ['male'], 'start': [start], 'stop': [stop]}
        common = dict(_ONSET_COMMON, filter_history=filter_history)
        out = find_onset_epochs(prediction_mode='state', mixture_model_params=_mixture_model_params(),
                                **common, **_onset_kwargs(tmp_path, rows, n_frames=n_frames, fps=fps))
        male = out['sess_B']['male']
        np.testing.assert_allclose(male['positive_events'], positives)
        np.testing.assert_allclose(male['negative_events'], negatives)

    @pytest.mark.filterwarnings("ignore::astropy.utils.exceptions.AstropyUserWarning")
    @pytest.mark.parametrize('smoothing_sd', [None, 0, 0.5, 2.0])
    def test_usv_rate_and_predictor_smoothing(self, tmp_path, smoothing_sd):
        """``usv_rate`` is ``usv_count`` smoothed with ``proportion_smoothing_sd`` when it
        is positive (also below one frame) and ``usv_count`` itself when it is None or
        0; the pooled predictor trace is smoothed with the same sd. The one-frame USV
        (frame 200) makes any smoothing lower the peak below 1."""

        rows = {'emitter': ['male'], 'start': [2.0], 'stop': [2.005]}
        common = dict(_ONSET_COMMON, proportion_smoothing_sd=smoothing_sd)
        male = find_onset_epochs(prediction_mode='individual', mixture_model_params=_mixture_model_params(),
                                 vocal_output_type='pooled_rate', **common,
                                 **_onset_kwargs(tmp_path, rows))['sess_B']['male']
        binary = _generate_vocal_trace(np.array([2.0]), np.array([2.005]), 600, 100.0)
        assert np.flatnonzero(binary).tolist() == [200]
        np.testing.assert_array_equal(male['usv_count'], binary)
        if smoothing_sd:
            expected = _smooth_occupancy(binary, smoothing_sd)
            assert male['usv_rate'].max() < 1.0
        else:
            expected = binary
        np.testing.assert_allclose(male['usv_rate'], expected)
        np.testing.assert_allclose(male['continuous_vocal_signals']['usv_rate'], expected)

    def test_skipped_sessions_do_not_stop_the_session_loop(self, tmp_path, capsys):
        """A session without a summary and a session missing from ``mouse_ids_dict``,
        both listed first, are skipped with their warnings and the valid session after
        them is still loaded."""

        (tmp_path / 'a_nocsv' / 'audio').mkdir(parents=True)
        _write_usv_summary(tmp_path / 'b_unreg', {'emitter': ['male'], 'start': [2.0], 'stop': [2.05]})
        _write_usv_summary(tmp_path / 'c_valid', {'emitter': ['male'], 'start': [3.0], 'stop': [3.05]})
        out = find_onset_epochs(root_directories=[str(tmp_path / name) for name in ('a_nocsv', 'b_unreg', 'c_valid')],
                                mouse_ids_dict={'a_nocsv': ['male', 'female'], 'c_valid': ['male', 'female']},
                                camera_fps_dict=dict.fromkeys(('a_nocsv', 'b_unreg', 'c_valid'), 100.0),
                                features_dict={name: _features_df(600) for name in ('a_nocsv', 'b_unreg', 'c_valid')},
                                prediction_mode='individual', mixture_model_params=_mixture_model_params(),
                                **_ONSET_COMMON)
        assert out['a_nocsv'] == {}
        assert out['b_unreg'] == {}
        np.testing.assert_allclose(out['c_valid']['male']['positive_events'], [3.0])
        printed = capsys.readouterr().out
        assert 'No USV summary found for a_nocsv' in printed
        assert 'No mouse names registered for b_unreg' in printed

    def test_each_sex_uses_its_own_mixture_parameters(self, tmp_path):
        """The first track (male) groups bouts with the male threshold (0.3 s) and the
        second (female) with the female one (1 s): the same 0.45 s gap splits the male's
        pair into two one-USV bouts (no positive) and keeps the female's pair as one
        bout (positive at 4.0 s)."""

        rows = {'emitter': ['male', 'male', 'female', 'female'],
                'start': [2.0, 2.5, 4.0, 4.5], 'stop': [2.05, 2.55, 4.05, 4.55]}
        params = {'male': {'means': [np.log(0.3)], 'sds': [0.0]}, 'female': {'means': [0.0], 'sds': [0.0]}}
        out = find_onset_epochs(prediction_mode='bout_onset', mixture_model_params=params,
                                **_ONSET_COMMON, **_onset_kwargs(tmp_path, rows))['sess_B']
        assert out['male']['positive_events'].size == 0
        np.testing.assert_allclose(out['female']['positive_events'], [4.0])

    def test_inverse_gaussian_threshold_groups_the_bouts(self, tmp_path):
        """An inverse-Gaussian block (``model_class`` 'ig', mean 0.1 s, shape 1 s) sets a
        0.21 s threshold: gaps of 0.15 s join and the 0.30 s gap splits, so
        'within_bout' offsets give two bouts (ends 2.25 and 2.8 s, negatives 2.05 and
        2.6 s) where the 1 s lognormal default would give one."""

        rows = {'emitter': ['male'] * 4, 'start': [2.0, 2.2, 2.55, 2.75], 'stop': [2.05, 2.25, 2.6, 2.8]}
        params = {'male': {'model_class': 'ig', 'mus': [0.1], 'lambdas': [1.0]},
                  'female': {'model_class': 'ig', 'mus': [0.1], 'lambdas': [1.0]}}
        assert 0.15 < _calculate_ibi_threshold_ig(0.1, 1.0, 2.58) < 0.30
        male = find_onset_epochs(prediction_mode='bout_offset', mixture_model_params=params,
                                 negative_scheme='within_bout', min_singing_after_negative=0.0,
                                 time_since_bout_onset_tolerance=None, max_negatives_per_bout=None,
                                 **_ONSET_COMMON, **_onset_kwargs(tmp_path, rows))['sess_B']['male']
        np.testing.assert_allclose(male['positive_events'], [2.25, 2.8])
        np.testing.assert_allclose(male['negative_events'], [2.05, 2.6])

    @pytest.mark.parametrize(('params', 'sex'), [
        ({'male': {'means': [], 'sds': []}, 'female': {'means': [0.0], 'sds': [0.0]}}, 'male'),
        ({'male': {'means': [0.0], 'sds': [0.0]}, 'female': {'means': [], 'sds': []}}, 'female'),
    ])
    def test_invalid_component_index_names_the_sex(self, tmp_path, params, sex):
        """A component index the block lacks raises naming the sex whose block it is."""

        rows = {'emitter': ['male'], 'start': [2.0], 'stop': [2.05]}
        with pytest.raises(ValueError, match=f"Invalid mixture_model_component_index 0 for {sex}"):
            find_onset_epochs(prediction_mode='bout_onset', mixture_model_params=params,
                              **_ONSET_COMMON, **_onset_kwargs(tmp_path, rows))

    def test_predictor_lists_are_validated(self, tmp_path):
        """With vocal predictors on, two predictor lists giving one trace name are
        refused before any session is read."""

        rows = {'emitter': ['male'], 'start': [2.0], 'stop': [2.05]}
        with pytest.raises(ValueError, match="give the trace name"):
            find_onset_epochs(prediction_mode='individual', mixture_model_params=_mixture_model_params(),
                              vocal_output_type='pooled_rate', predictor_classes=[['usv'], ['usv', 'overlap']],
                              **_ONSET_COMMON, **_onset_kwargs(tmp_path, rows))

    def test_messages_name_the_summary(self, tmp_path, capsys):
        """The absent-label error, the noise-drop report and the missing-squeak-extent
        errors (of the target timing and of the silence reference) all name the
        session's summary file."""

        rows = {'emitter': ['male', 'male', 'male'], 'start': [2.0, 3.0, 4.0], 'stop': [2.05, 3.05, 4.05],
                'noise': [False, True, False]}
        kwargs = _onset_kwargs(tmp_path, rows)
        with pytest.raises(ValueError, match=_LABELS_UNAVAILABLE) as excinfo:
            find_onset_epochs(prediction_mode='individual', mixture_model_params=_mixture_model_params(),
                              category_column='qlvm_category', target_category=3, **_ONSET_COMMON, **kwargs)
        assert 'sess_B_usv_summary.csv' in str(excinfo.value)
        male = find_onset_epochs(prediction_mode='individual', mixture_model_params=_mixture_model_params(),
                                 **_ONSET_COMMON, **kwargs)['sess_B']['male']
        np.testing.assert_allclose(male['positive_events'], [2.0, 4.0])
        assert 'sess_B_usv_summary.csv: dropped 1 noise segment(s) of 3.' in capsys.readouterr().out
        _drop_summary_columns(tmp_path, 'sess_B', 'squeak_start', 'squeak_end')
        for target in (['squeak'], ['usv']):
            with pytest.raises(KeyError, match=re.escape("sess_B_usv_summary.csv has no")):
                find_onset_epochs(prediction_mode='individual', mixture_model_params=_mixture_model_params(),
                                  target_classes=target, **_ONSET_COMMON, **kwargs)


# find_usv_categories: exact event sets, loop control, counts and messages


class TestUsvCategoriesExact:
    """``find_usv_categories`` exact event sets, loop control, counts and console messages on
    hand-placed USVs."""

    @pytest.mark.filterwarnings("ignore::astropy.utils.exceptions.AstropyUserWarning")
    def test_events_streams_and_traces_are_exact(self, tmp_path):
        """The male's only USV starts exactly at ``filter_history`` and is dropped (the
        bound is exclusive), leaving him the default dict, while the female after him is
        still processed: her null-category USV is skipped in ``events_by_category``
        without ending the loop and has no category stream; categories 1 and 2 carry
        their own onsets and stop times; her pooled trace (history period included) is
        smoothed with ``proportion_smoothing_sd``. The stream loop compares the label
        column with the null label, which polars flags with a UserWarning (filtered
        here); the comparison matches no row, so the null label gets no stream."""

        rows = {
            'emitter': ['male', 'female', 'female', 'female', 'female'],
            'start': [1.0, 2.0, 3.0, 4.0, 5.0],
            'stop': [1.05, 2.05, 3.05, 4.05, 5.05],
            'usv_category': [1, None, 2, 1, 2],
        }
        out = find_usv_categories(target_category=None, filter_history=1.0, vocal_output_type='pooled_rate',
                                  proportion_smoothing_sd=2.0, **_onset_kwargs(tmp_path, rows, session='sess_D'))
        male = out['sess_D']['male']
        assert male['events_by_category'] == {}
        assert male['category_streams'] == {}
        assert male['continuous_vocal_signals'] == {}
        female = out['sess_D']['female']
        assert sorted(female['events_by_category']) == [1, 2]
        np.testing.assert_allclose(female['events_by_category'][1], [4.0])
        np.testing.assert_allclose(female['events_by_category'][2], [3.0, 5.0])
        assert sorted(female['category_streams']) == [1, 2]
        np.testing.assert_allclose(female['category_streams'][1]['start'], [4.0])
        np.testing.assert_allclose(female['category_streams'][1]['stop'], [4.05])
        np.testing.assert_allclose(female['category_streams'][2]['start'], [3.0, 5.0])
        np.testing.assert_allclose(female['category_streams'][2]['stop'], [3.05, 5.05])
        np.testing.assert_allclose(female['continuous_vocal_signals']['usv_rate'], _generate_vocal_trace(
            np.array([2.0, 3.0, 4.0, 5.0]), np.array([2.05, 3.05, 4.05, 5.05]), 600, 100.0, smooth_sd=2.0))

    def test_skipped_sessions_do_not_stop_the_session_loop(self, tmp_path):
        """Sessions without a summary, without registered mice and without feature data,
        all listed first, are skipped and the valid session after them is loaded."""

        names = ('a_nocsv', 'b_unreg', 'c_nofeat', 'd_valid')
        (tmp_path / 'a_nocsv' / 'audio').mkdir(parents=True)
        for name in names[1:]:
            _write_usv_summary(tmp_path / name, {'emitter': ['male'], 'start': [2.0], 'stop': [2.05],
                                                 'usv_category': [3]})
        out = find_usv_categories(root_directories=[str(tmp_path / name) for name in names],
                                  mouse_ids_dict={name: ['male', 'female'] for name in names if name != 'b_unreg'},
                                  camera_fps_dict=dict.fromkeys(names, 100.0),
                                  features_dict={name: _features_df(600) for name in names if name != 'c_nofeat'},
                                  target_category=None, filter_history=1.0)
        assert out['a_nocsv'] == {}
        assert out['b_unreg'] == {}
        assert out['c_nofeat'] == {}
        np.testing.assert_allclose(out['d_valid']['male']['events_by_category'][3], [2.0])

    def test_unplaced_counts_per_pair_and_in_total(self, tmp_path, capsys):
        """Unplaced vocalizations are counted per session-mouse pair (male 1 of 3, female
        2 of 3) and summed over every pair (3 in 2 pairs); a session whose vocalizations
        are all placed adds no line and no pair."""

        rows_p = {
            'emitter': ['male', 'male', 'male', 'female', 'female', 'female'],
            'start': [2.0, 3.0, 4.0, 2.5, 3.5, 4.5],
            'stop': [2.05, 3.05, 4.05, 2.55, 3.55, 4.55],
            'qlvm1': [0.1, None, 0.3, None, 0.2, None],
            'qlvm2': [0.4, 0.5, 0.6, 0.1, 0.2, 0.3],
        }
        rows_q = {'emitter': ['male', 'female'], 'start': [2.0, 3.0], 'stop': [2.05, 3.05],
                  'qlvm1': [0.1, 0.2], 'qlvm2': [0.3, 0.4]}
        kwargs_p = _onset_kwargs(tmp_path, rows_p, session='sess_P')
        kwargs_q = _onset_kwargs(tmp_path, rows_q, session='sess_Q')
        kwargs = {key: ({**kwargs_p[key], **kwargs_q[key]} if isinstance(kwargs_p[key], dict)
                        else kwargs_p[key] + kwargs_q[key]) for key in kwargs_p}
        out = find_usv_categories(target_category=None, filter_history=1.0, category_column=None,
                                  manifold_column_names=['qlvm1', 'qlvm2'], **kwargs)
        np.testing.assert_allclose(out['sess_P']['male']['continuous_onsets'], [2.0, 4.0])
        np.testing.assert_allclose(out['sess_P']['female']['continuous_onsets'], [3.5])
        printed = capsys.readouterr().out
        assert 'sess_P (male): dropped 1 of 3 vocalizations' in printed
        assert 'sess_P (female): dropped 2 of 3 vocalizations' in printed
        assert 'sess_Q' not in printed
        assert 'dropped 3 vocalizations with null/NaN coordinates in total (2 session-mouse pairs affected).' in printed

    def test_text_coordinates_are_unplaced_not_fatal(self, tmp_path):
        """A manifold column CSV inference reads as text (an 'NA' marker) casts leniently:
        the marked vocalization is unplaced and dropped, the rest keep their targets."""

        rows = {'emitter': ['male'] * 3, 'start': [2.0, 3.0, 4.0], 'stop': [2.05, 3.05, 4.05],
                'qlvm1': ['0.1', 'NA', '0.3'], 'qlvm2': [0.4, 0.5, 0.6]}
        male = find_usv_categories(target_category=None, filter_history=1.0, category_column=None,
                                   manifold_column_names=['qlvm1', 'qlvm2'],
                                   **_onset_kwargs(tmp_path, rows, session='sess_D'))['sess_D']['male']
        np.testing.assert_allclose(male['continuous_onsets'], [2.0, 4.0])
        np.testing.assert_allclose(male['continuous_targets'], [[0.1, 0.4], [0.3, 0.6]])

    def test_squeak_map_refilters_the_history_on_the_squeak_onset(self, tmp_path):
        """On the squeak map the history bound applies to the squeak's own onset: a squeak
        whose extent starts exactly at ``filter_history`` (1.0 s) is dropped even though
        its segment starts later (1.2 s)."""

        rows = {'emitter': ['male', 'male'], 'start': [1.2, 2.0], 'stop': [1.3, 2.1],
                'usv': [False, False], 'squeak': [True, True],
                'squeak_start': [1.0, 2.02], 'squeak_end': [1.3, 2.08],
                'qlvm_squeak1': [0.1, 0.2], 'qlvm_squeak2': [0.3, 0.4]}
        male = find_usv_categories(target_category=None, filter_history=1.0, category_column=None,
                                   manifold_column_names=['qlvm_squeak1', 'qlvm_squeak2'],
                                   **_onset_kwargs(tmp_path, rows, session='sess_D'))['sess_D']['male']
        np.testing.assert_allclose(male['continuous_onsets'], [2.02])
        np.testing.assert_allclose(male['continuous_targets'], [[0.2, 0.4]])

    def test_purposes_and_sources_are_named(self, tmp_path, capsys):
        """Every error names what needs the labels or the flags: the category models, the
        binomial target, the configured label column, the pair whose summary lacks the
        class flags or the squeak extent; the noise-drop report names the pair; and
        clashing predictor lists are refused."""

        rows = {'emitter': ['male', 'male'], 'start': [2.0, 3.0], 'stop': [2.05, 3.05],
                'qlvm1': [0.1, 0.2], 'qlvm2': [0.3, 0.4], 'noise': [False, True]}
        kwargs = _onset_kwargs(tmp_path, rows, session='sess_D')
        with pytest.raises(ValueError, match=re.escape("The USV category models (multinomial / binomial) needs")):
            find_usv_categories(target_category=None, filter_history=1.0, category_column=None, **kwargs)
        with pytest.raises(ValueError, match="The binomial target category 3 needs"):
            find_usv_categories(target_category=3, filter_history=1.0, category_column=None,
                                manifold_column_names=['qlvm1', 'qlvm2'], **kwargs)
        with pytest.raises(ValueError, match=re.escape("vocal_features.usv_category_column_name needs the column 'absent'")):
            find_usv_categories(target_category=None, filter_history=1.0, category_column='absent',
                                manifold_column_names=['qlvm1', 'qlvm2'], **kwargs)
        with pytest.raises(ValueError, match="give the trace name"):
            find_usv_categories(target_category=None, filter_history=1.0, category_column=None,
                                manifold_column_names=['qlvm1', 'qlvm2'], vocal_output_type='pooled_rate',
                                predictor_classes=[['usv'], ['usv', 'overlap']], **kwargs)
        male = find_usv_categories(target_category=None, filter_history=1.0, category_column=None,
                                   manifold_column_names=['qlvm1', 'qlvm2'], **kwargs)['sess_D']['male']
        np.testing.assert_allclose(male['continuous_onsets'], [2.0])
        assert 'sess_D (male): dropped 1 noise segment(s) of 2.' in capsys.readouterr().out
        _drop_summary_columns(tmp_path, 'sess_D', 'usv', 'squeak')
        with pytest.raises(KeyError, match=r"sess_D \(male\) has no"):
            find_usv_categories(target_category=None, filter_history=1.0, category_column=None,
                                manifold_column_names=['qlvm1', 'qlvm2'], vocal_output_type='pooled_rate', **kwargs)

    def test_squeak_map_without_extent_names_the_pair(self, tmp_path):
        """The squeak map on a summary without the squeak extent raises naming the pair."""

        rows = {'emitter': ['male'], 'start': [2.0], 'stop': [2.05], 'usv': [False], 'squeak': [True],
                'qlvm_squeak1': [0.1], 'qlvm_squeak2': [0.2]}
        kwargs = _onset_kwargs(tmp_path, rows, session='sess_D')
        _drop_summary_columns(tmp_path, 'sess_D', 'squeak_start', 'squeak_end')
        with pytest.raises(KeyError, match=r"sess_D \(male\) has no"):
            find_usv_categories(target_category=None, filter_history=1.0, category_column=None,
                                manifold_column_names=['qlvm_squeak1', 'qlvm_squeak2'], **kwargs)


# find_variable_length_bouts: exact bouts, loop control, sex-specific thresholds and messages


class TestVariableLengthBoutsExact:
    """``find_variable_length_bouts`` exact bouts, loop control, sex-specific thresholds and
    console messages on hand-placed USVs."""

    def test_bout_metrics_and_rejections_are_exact(self, tmp_path):
        """Four male bouts (1 s threshold): one starting exactly at ``filter_history``
        (rejected, exclusive bound), a one-USV bout (below the floor), and two valid ones
        whose onsets, durations, counts and mask sums cover their own USVs only (masks
        1+2+3 and 4+5, not the next bout's first)."""

        rows = {
            'emitter': ['male'] * 8,
            'start': [1.0, 1.1, 3.0, 5.0, 5.1, 5.2, 7.0, 7.1],
            'stop': [1.05, 1.15, 3.05, 5.05, 5.15, 5.25, 7.05, 7.15],
            'mask_number': [9, 9, 9, 1, 2, 3, 4, 5],
        }
        male = find_variable_length_bouts(min_vocalizations=2, filter_history=1.0,
                                          mixture_model_params=_mixture_model_params(),
                                          **_onset_kwargs(tmp_path, rows, session='sess_E', n_frames=900))['sess_E']['male']
        np.testing.assert_allclose(male['bout_onsets'], [5.0, 7.0])
        np.testing.assert_allclose(male['bout_durations'], [0.25, 0.15])
        assert male['bout_syllable_counts'].tolist() == [3, 2]
        assert male['total_mask_complexity'].tolist() == [6, 9]
        np.testing.assert_allclose(male['mean_mask_complexity'], [2.0, 4.5])
        assert isinstance(male['ibi_threshold_used'], float)
        assert not isinstance(male['ibi_threshold_used'], np.ndarray)

    def test_bout_offsets_are_the_stored_stop_bit_for_bit(self, tmp_path):
        """The offset is the last USV's stored stop, not ``onset + duration``: for this
        bout (1.069411 -> 3.23396 s) the rebuilt sum falls one ulp before the stop, which
        made the bout's own last USV mark it 'not silent' in the matched divergence."""

        rows = {
            'emitter': ['male'] * 4,
            'start': [1.069411, 1.8, 2.5, 3.1],
            'stop': [1.15, 1.9, 2.6, 3.23396],
            'mask_number': [1, 1, 1, 1],
        }
        male = find_variable_length_bouts(min_vocalizations=2, filter_history=1.0,
                                          mixture_model_params=_mixture_model_params(),
                                          **_onset_kwargs(tmp_path, rows, session='sess_E'))['sess_E']['male']
        assert male['bout_onsets'][0] + male['bout_durations'][0] < 3.23396
        assert male['bout_offsets'].tolist() == [3.23396]

    def test_single_usv_bout(self, tmp_path):
        """A single USV with a one-vocalization floor is one bout of count 1."""

        rows = {'emitter': ['male'], 'start': [2.0], 'stop': [2.05], 'mask_number': [3]}
        male = find_variable_length_bouts(min_vocalizations=1, filter_history=1.0,
                                          mixture_model_params=_mixture_model_params(),
                                          **_onset_kwargs(tmp_path, rows, session='sess_E'))['sess_E']['male']
        np.testing.assert_allclose(male['bout_onsets'], [2.0])
        assert male['bout_syllable_counts'].tolist() == [1]
        assert male['total_mask_complexity'].tolist() == [3]

    def test_mouse_without_targets_does_not_stop_the_mouse_loop(self, tmp_path):
        """A male holding only squeaks (no target USVs) is skipped and the female after
        him still gets her bout."""

        rows = {'emitter': ['male', 'female', 'female'], 'start': [2.0, 3.0, 3.1], 'stop': [2.05, 3.05, 3.15],
                'usv': [False, True, True], 'squeak': [True, False, False]}
        out = find_variable_length_bouts(min_vocalizations=2, filter_history=1.0,
                                         mixture_model_params=_mixture_model_params(),
                                         **_onset_kwargs(tmp_path, rows, session='sess_E'))['sess_E']
        assert out['male']['bout_onsets'] == []
        np.testing.assert_allclose(out['female']['bout_onsets'], [3.0])

    def test_skipped_sessions_do_not_stop_the_session_loop(self, tmp_path):
        """A session without a summary and one missing from ``mouse_ids_dict``, listed
        first, are skipped and the valid session after them is loaded."""

        names = ('a_nocsv', 'b_unreg', 'c_valid')
        (tmp_path / 'a_nocsv' / 'audio').mkdir(parents=True)
        for name in names[1:]:
            _write_usv_summary(tmp_path / name, {'emitter': ['male', 'male'], 'start': [2.0, 2.1],
                                                 'stop': [2.05, 2.15]})
        out = find_variable_length_bouts(root_directories=[str(tmp_path / name) for name in names],
                                         mouse_ids_dict={name: ['male', 'female'] for name in names if name != 'b_unreg'},
                                         camera_fps_dict=dict.fromkeys(names, 100.0),
                                         features_dict={name: _features_df(600) for name in names},
                                         min_vocalizations=2, filter_history=1.0,
                                         mixture_model_params=_mixture_model_params())
        assert out['a_nocsv'] == {}
        assert out['b_unreg'] == {}
        np.testing.assert_allclose(out['c_valid']['male']['bout_onsets'], [2.0])

    def test_each_sex_uses_its_own_lognormal_threshold(self, tmp_path):
        """The male's threshold comes from the male block (exp(log 0.3) = 0.3 s), the
        female's from the female block (1 s)."""

        rows = {'emitter': ['male'], 'start': [2.0], 'stop': [2.05]}
        params = {'male': {'means': [np.log(0.3)], 'sds': [0.0]}, 'female': {'means': [0.0], 'sds': [0.0]}}
        out = find_variable_length_bouts(min_vocalizations=2, filter_history=1.0, mixture_model_params=params,
                                         **_onset_kwargs(tmp_path, rows, session='sess_E'))['sess_E']
        assert out['male']['ibi_threshold_used'] == pytest.approx(0.3)
        assert out['female']['ibi_threshold_used'] == pytest.approx(1.0)

    def test_inverse_gaussian_threshold_per_sex(self, tmp_path):
        """Inverse-Gaussian blocks give each sex the IG quantile of its own component
        (male mean 0.1 s / shape 1 s ~ 0.21 s, female 0.05 s / 0.5 s ~ 0.105 s), and the
        male's 0.15 s gaps join while his 0.30 s gap splits."""

        rows = {'emitter': ['male'] * 4, 'start': [5.0, 5.2, 5.55, 5.75], 'stop': [5.05, 5.25, 5.6, 5.8]}
        params = {'male': {'model_class': 'ig', 'mus': [0.1], 'lambdas': [1.0]},
                  'female': {'model_class': 'ig', 'mus': [0.05], 'lambdas': [0.5]}}
        out = find_variable_length_bouts(min_vocalizations=2, filter_history=1.0, mixture_model_params=params,
                                         **_onset_kwargs(tmp_path, rows, session='sess_E'))['sess_E']
        assert out['male']['ibi_threshold_used'] == pytest.approx(_calculate_ibi_threshold_ig(0.1, 1.0, 2.58))
        assert out['female']['ibi_threshold_used'] == pytest.approx(_calculate_ibi_threshold_ig(0.05, 0.5, 2.58))
        np.testing.assert_allclose(out['male']['bout_onsets'], [5.0, 5.55])

    @pytest.mark.parametrize(('params', 'sex'), [
        ({'male': {'means': [], 'sds': []}, 'female': {'means': [0.0], 'sds': [0.0]}}, 'male'),
        ({'male': {'means': [0.0], 'sds': [0.0]}, 'female': {'means': [], 'sds': []}}, 'female'),
    ])
    def test_invalid_component_index_names_the_sex(self, tmp_path, params, sex):
        """A component index the block lacks raises naming the sex whose block it is."""

        rows = {'emitter': ['male'], 'start': [2.0], 'stop': [2.05]}
        with pytest.raises(ValueError, match=f"Invalid mixture_model_component_index 0 for {sex}"):
            find_variable_length_bouts(min_vocalizations=2, filter_history=1.0, mixture_model_params=params,
                                       **_onset_kwargs(tmp_path, rows, session='sess_E'))

    @pytest.mark.filterwarnings("ignore::astropy.utils.exceptions.AstropyUserWarning")
    def test_predictor_traces_span_the_session_and_are_smoothed(self, tmp_path):
        """The predictor traces have one value per camera frame (600, the feature table's
        row count), the binary one marks frames 200..205 of the USV at 2.0-2.05 s, and
        the rate one is smoothed with ``proportion_smoothing_sd``."""

        rows = {'emitter': ['male'], 'start': [2.0], 'stop': [2.05]}
        kwargs = _onset_kwargs(tmp_path, rows, session='sess_E')
        binary = find_variable_length_bouts(min_vocalizations=2, filter_history=1.0,
                                            mixture_model_params=_mixture_model_params(),
                                            vocal_output_type='pooled_binary', **kwargs)['sess_E']['male']
        assert np.flatnonzero(binary['continuous_vocal_signals']['usv_event']).tolist() == [200, 201, 202, 203, 204]
        assert binary['continuous_vocal_signals']['usv_event'].shape == (600,)
        rate = find_variable_length_bouts(min_vocalizations=2, filter_history=1.0, proportion_smoothing_sd=2.0,
                                          mixture_model_params=_mixture_model_params(),
                                          vocal_output_type='pooled_rate', **kwargs)['sess_E']['male']
        np.testing.assert_allclose(rate['continuous_vocal_signals']['usv_rate'], _generate_vocal_trace(
            np.array([2.0]), np.array([2.05]), 600, 100.0, smooth_sd=2.0))

    def test_purposes_and_sources_are_named(self, tmp_path, capsys):
        """Every error names its purpose or source: the predictor type that needs labels
        (before and after a summary is read, the latter with the file), clashing
        predictor lists, the pair whose summary lacks the class flags (whether the
        predictor traces or the bout grouping reads them first) or the squeak extent;
        the noise-drop report names the pair."""

        rows = {'emitter': ['male', 'male', 'male'], 'start': [2.0, 2.1, 2.2], 'stop': [2.05, 2.15, 2.25],
                'noise': [False, True, False]}
        kwargs = _onset_kwargs(tmp_path, rows, session='sess_E')
        common = {'min_vocalizations': 2, 'filter_history': 1.0, 'mixture_model_params': _mixture_model_params()}
        with pytest.raises(ValueError, match=re.escape("vocal_features.vocal_predictor_type 'categories_rate' needs a per-USV")):
            find_variable_length_bouts(vocal_output_type='categories_rate', category_column=None, **common, **kwargs)
        with pytest.raises(ValueError, match="vocal_predictor_type 'categories_rate' needs the column 'absent'") as excinfo:
            find_variable_length_bouts(vocal_output_type='categories_rate', category_column='absent', **common, **kwargs)
        assert 'sess_E_usv_summary.csv' in str(excinfo.value)
        with pytest.raises(ValueError, match="give the trace name"):
            find_variable_length_bouts(vocal_output_type='pooled_rate', predictor_classes=[['usv'], ['usv', 'overlap']],
                                       **common, **kwargs)
        male = find_variable_length_bouts(**common, **kwargs)['sess_E']['male']
        np.testing.assert_allclose(male['bout_onsets'], [2.0])
        assert 'sess_E (male): dropped 1 noise segment(s) of 3.' in capsys.readouterr().out
        _drop_summary_columns(tmp_path, 'sess_E', 'squeak_start', 'squeak_end')
        with pytest.raises(KeyError, match=r"sess_E \(male\) has no"):
            find_variable_length_bouts(target_classes=['overlap'], **common, **kwargs)
        _drop_summary_columns(tmp_path, 'sess_E', 'usv', 'squeak')
        for vocal_output_type in (None, 'pooled_rate'):
            with pytest.raises(KeyError, match=r"sess_E \(male\) has no"):
                find_variable_length_bouts(vocal_output_type=vocal_output_type, **common, **kwargs)
