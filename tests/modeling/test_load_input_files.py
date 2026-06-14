"""
@author: bartulem
Unit tests for the pure helpers in
``usv_playpen.modeling.load_input_files``: the IBI-threshold formula, the
USV-timestamp-to-trace renderer, the "clean" (no-USV) epoch tiler, and the
pickle loader.

The big ``find_*`` orchestrators require on-disk session trees and are out
of scope here; these four helpers are pure and carry the load-bearing
temporal logic. The tests assert the closed-form IBI cutoff, the binary /
smoothed trace shapes, and — most importantly — that every tiled clean
epoch's history window avoids the forbidden zone around each USV.
"""

from __future__ import annotations

import pickle

import h5py
import numpy as np
import polars as pls
import pytest

from usv_playpen.modeling.load_input_files import (
    _calculate_ibi_threshold,
    _generate_vocal_trace,
    _get_clean_tiled_epochs,
    find_bout_epochs,
    find_usv_categories,
    find_variable_length_bouts,
    load_behavioral_feature_data,
    load_pickle_modeling_data,
)


# Shared synthetic-session helpers for the on-disk ``find_*`` loaders.


def _write_usv_summary(session_root, rows: dict, csv_sep: str = ',') -> None:
    """
    Description
    -----------
    Writes a synthetic ``*_usv_summary.csv`` under ``<session_root>/audio`` so
    the disk-reading loaders (``find_bout_epochs``, ``find_usv_categories``,
    ``find_variable_length_bouts``) can ingest it through their polars
    ``read_csv`` glob (``audio/**/*_usv_summary.csv``).

    Parameters
    ----------
    session_root (pathlib.Path)
        The session ROOT directory; the ``audio`` subdir is created if absent.
    rows (dict)
        Column-name -> value-list mapping passed straight to
        ``polars.DataFrame``; must include at least the ``emitter``, ``start``
        and ``stop`` columns the loaders rely on.
    csv_sep (str)
        Field separator written into the CSV (mirrors the loaders' ``csv_sep``).

    Returns
    -------
    None
    """

    audio_dir = session_root / 'audio'
    audio_dir.mkdir(parents=True, exist_ok=True)
    csv_path = audio_dir / f'{session_root.name}_usv_summary.csv'
    pls.DataFrame(rows).write_csv(file=csv_path, separator=csv_sep)


def _gmm_params() -> dict:
    """
    Description
    -----------
    Returns a minimal sex-specific GMM parameter dict of the shape the loaders
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
        A ``{'male': ..., 'female': ...}`` GMM parameter dictionary.
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

    def test_log_normal_formula(self):
        """The threshold is ``exp(mu + z * sd)``."""

        assert _calculate_ibi_threshold(0.0, 1.0, 2.58) == pytest.approx(np.exp(2.58))
        assert _calculate_ibi_threshold(1.0, 0.0, 2.58) == pytest.approx(np.exp(1.0))

    def test_monotone_increasing_in_z(self):
        """A larger z-score yields a larger (more permissive) cutoff."""

        lo = _calculate_ibi_threshold(0.0, 0.5, 1.0)
        hi = _calculate_ibi_threshold(0.0, 0.5, 3.0)
        assert hi > lo


# _generate_vocal_trace


class TestGenerateVocalTrace:

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

    def test_no_usv_tiles_whole_session(self):
        """With no USVs the entire session (after the initial
        ``filter_history`` warm-up) is tiled into window end-times."""

        ts = _get_clean_tiled_epochs(np.array([]), np.array([]),
                                     filter_history=1.0, session_duration_sec=10.0)
        # First end-time is at session_start + filter_history == 2.0.
        assert ts.min() == pytest.approx(2.0)
        assert ts.max() <= 10.0 + 1e-9
        # Regular 1.0-spaced tiling.
        assert np.allclose(np.diff(ts), 1.0)

    def test_windows_avoid_forbidden_zone(self):
        """Every returned end-time ``t`` defines a history window
        ``[t - filter_history, t]`` that must not overlap the forbidden
        zone ``[start, stop + filter_history]`` around a USV."""

        fh = 1.0
        usv_start, usv_stop = 50.0, 50.5
        ts = _get_clean_tiled_epochs(np.array([usv_start]), np.array([usv_stop]),
                                     filter_history=fh, session_duration_sec=100.0)
        forbidden_lo = usv_start
        forbidden_hi = usv_stop + fh
        for t in ts:
            win_lo, win_hi = t - fh, t
            # Window is entirely before or entirely after the forbidden zone.
            assert win_hi <= forbidden_lo + 1e-9 or win_lo >= forbidden_hi - 1e-9, (
                f"window [{win_lo}, {win_hi}] overlaps forbidden "
                f"[{forbidden_lo}, {forbidden_hi}]"
            )

    def test_short_clean_gap_is_dropped(self):
        """A clean gap shorter than ``filter_history`` cannot host a
        window and yields no end-times for that gap."""

        # Two USVs leaving only a tiny clean gap between them; the whole
        # session is otherwise forbidden.
        ts = _get_clean_tiled_epochs(np.array([1.0, 1.4]), np.array([1.1, 1.5]),
                                     filter_history=1.0, session_duration_sec=2.6)
        # session_duration 2.6, last forbidden end = 1.5 + 1.0 = 2.5; the
        # trailing clean gap [2.5, 2.6] is shorter than filter_history -> empty.
        assert ts.size == 0


# load_pickle_modeling_data


class TestLoadPickleModelingData:

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

    def test_reads_features_fps_and_track_names(self, tmp_path):
        """A session whose ``video`` subtree holds the behavioral-features CSV
        and the metric track H5 yields the per-session feature DataFrame, the
        recording frame rate (decoded from the H5 scalar) and the decoded
        track-name list."""

        sess = tmp_path / 'sess_A'
        video_dir = sess / 'video'
        video_dir.mkdir(parents=True)

        features_csv = video_dir / 'x_points3d_translated_rotated_metric_behavioral_features.csv'
        pls.DataFrame({'speed': [0.1, 0.2, 0.3], 'angle': [1.0, 2.0, 3.0]}).write_csv(file=features_csv)

        track_h5 = video_dir / 'x_points3d_translated_rotated_metric.h5'
        with h5py.File(name=track_h5, mode='w') as h5f:
            h5f['recording_frame_rate'] = 150.0
            h5f['track_names'] = np.array([b'male', b'female'])

        beh, fps, names = load_behavioral_feature_data(behavior_file_paths=[str(sess)])

        assert list(beh.keys()) == ['sess_A']
        assert beh['sess_A'].shape == (3, 2)
        assert fps['sess_A'] == pytest.approx(150.0)
        assert names['sess_A'] == ['male', 'female']


# find_bout_epochs


class TestFindBoutEpochs:

    def _build(self, tmp_path, rows: dict, n_frames: int = 600, fps: float = 100.0):
        """Builds a one-session tree from ``rows`` and returns the kwargs
        triplet (mouse_ids/camera_fps/features dicts) plus the root list that
        every ``find_bout_epochs`` test below feeds in."""

        sess = tmp_path / 'sess_B'
        _write_usv_summary(sess, rows)
        return {
            'root_directories': [str(sess)],
            'mouse_ids_dict': {'sess_B': ['male', 'female']},
            'camera_fps_dict': {'sess_B': fps},
            'features_dict': {'sess_B': _features_df(n_frames)},
        }

    def test_bout_mode_positive_and_negative_events(self, tmp_path):
        """In 'bout' mode a clean cluster of >= ``min_usv_per_bout`` syllables
        becomes one positive onset, and the silent tail tiles into negative
        (no-USV) onsets."""

        rows = {
            'emitter': ['male', 'male', 'male'],
            'start': [2.0, 2.1, 2.2],
            'stop': [2.05, 2.15, 2.25],
            'usv_category': [1, 1, 1],
            'usv_supercategory': [1, 1, 1],
        }
        kwargs = self._build(tmp_path, rows)
        out = find_bout_epochs(prediction_mode='bout', filter_history=1.0,
                               usv_bout_time=0.5, min_usv_per_bout=2,
                               proportion_smoothing_sd=None, gmm_params=_gmm_params(),
                               **kwargs)

        male = out['sess_B']['male']
        # The single clean bout starts at 2.0 (> filter_history, no prior USV).
        np.testing.assert_allclose(male['positive_events'], [2.0])
        # Negative events exist and are all silent onsets after the bout zone.
        assert male['negative_events'].size > 0
        assert np.all(male['negative_events'] > rows['stop'][-1])
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
            'usv_supercategory': [1, 1],
        }
        kwargs = self._build(tmp_path, rows)
        out = find_bout_epochs(prediction_mode='bout', filter_history=1.0,
                               usv_bout_time=0.5, min_usv_per_bout=3,
                               proportion_smoothing_sd=None, gmm_params=_gmm_params(),
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
            'usv_supercategory': [1, 1, 1],
        }
        kwargs = self._build(tmp_path, rows)
        out = find_bout_epochs(prediction_mode='individual', filter_history=1.0,
                               usv_bout_time=0.5, min_usv_per_bout=2,
                               proportion_smoothing_sd=None, gmm_params=_gmm_params(),
                               **kwargs)
        male = out['sess_B']['male']
        # 0.5 is dropped (<= filter_history); 2.0 and 3.0 survive.
        np.testing.assert_allclose(male['positive_events'], [2.0, 3.0])
        assert male['negative_events'].size > 0

    def test_state_mode_partitions_grid_by_occupancy(self, tmp_path):
        """In 'state' mode the regular time grid is split into vocalizing
        (positive) and silent (negative) onsets by per-frame occupancy."""

        rows = {
            'emitter': ['male'],
            'start': [2.0],
            'stop': [2.9],
            'usv_category': [1],
            'usv_supercategory': [1],
        }
        kwargs = self._build(tmp_path, rows, n_frames=600, fps=100.0)
        out = find_bout_epochs(prediction_mode='state', filter_history=1.0,
                               usv_bout_time=0.5, min_usv_per_bout=2,
                               proportion_smoothing_sd=None, gmm_params=_gmm_params(),
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
            'usv_category': [1, 7],
            'usv_supercategory': [1, 1],
        }
        kwargs = self._build(tmp_path, rows)
        out = find_bout_epochs(prediction_mode='bout', filter_history=1.0,
                               usv_bout_time=0.5, min_usv_per_bout=2,
                               proportion_smoothing_sd=2.0, gmm_params=_gmm_params(),
                               vocal_output_type='all_rate',
                               **kwargs)
        signals = out['sess_B']['male']['continuous_vocal_signals']
        assert 'usv_rate' in signals
        assert 'usv_cat_1' in signals
        assert 'usv_cat_7' in signals
        assert signals['usv_rate'].shape == (600,)

    def test_pooled_binary_output(self, tmp_path):
        """``vocal_output_type='pooled_binary'`` writes a binary 'usv_event'
        trace and no per-category traces."""

        rows = {
            'emitter': ['male', 'male'],
            'start': [2.0, 2.1],
            'stop': [2.05, 2.15],
            'usv_category': [1, 1],
            'usv_supercategory': [1, 1],
        }
        kwargs = self._build(tmp_path, rows)
        out = find_bout_epochs(prediction_mode='bout', filter_history=1.0,
                               usv_bout_time=0.5, min_usv_per_bout=2,
                               proportion_smoothing_sd=None, gmm_params=_gmm_params(),
                               vocal_output_type='pooled_binary',
                               **kwargs)
        signals = out['sess_B']['male']['continuous_vocal_signals']
        assert 'usv_event' in signals
        assert set(np.unique(signals['usv_event'])).issubset({0.0, 1.0})

    def test_noise_categories_filtered_out(self, tmp_path):
        """Rows whose ``usv_supercategory`` is in ``noise_vocal_categories``
        are dropped before any sampling."""

        rows = {
            'emitter': ['male', 'male', 'male'],
            'start': [2.0, 2.1, 2.2],
            'stop': [2.05, 2.15, 2.25],
            'usv_category': [1, 1, 1],
            'usv_supercategory': [0, 0, 0],
        }
        kwargs = self._build(tmp_path, rows)
        out = find_bout_epochs(prediction_mode='individual', filter_history=1.0,
                               usv_bout_time=0.5, min_usv_per_bout=2,
                               proportion_smoothing_sd=None, gmm_params=_gmm_params(),
                               noise_vocal_categories=[0],
                               **kwargs)
        # All male USVs were noise -> no positive events, empty start array.
        assert out['sess_B']['male']['start'].size == 0
        assert out['sess_B']['male']['positive_events'].size == 0

    def test_missing_summary_csv_skips_session(self, tmp_path, capsys):
        """A session ROOT with no ``audio/*_usv_summary.csv`` is skipped with a
        warning, leaving its dict entry empty."""

        sess = tmp_path / 'sess_empty'
        (sess / 'audio').mkdir(parents=True)
        out = find_bout_epochs(root_directories=[str(sess)],
                               mouse_ids_dict={'sess_empty': ['male', 'female']},
                               camera_fps_dict={'sess_empty': 100.0},
                               features_dict={'sess_empty': _features_df(100)},
                               prediction_mode='bout', filter_history=1.0,
                               usv_bout_time=0.5, min_usv_per_bout=2,
                               proportion_smoothing_sd=None, gmm_params=_gmm_params())
        assert out['sess_empty'] == {}
        assert 'No USV summary' in capsys.readouterr().out

    def test_unregistered_session_skipped(self, tmp_path, capsys):
        """A session present on disk but absent from ``mouse_ids_dict`` is
        skipped with a warning."""

        sess = tmp_path / 'sess_C'
        _write_usv_summary(sess, {
            'emitter': ['male'], 'start': [2.0], 'stop': [2.05],
            'usv_category': [1], 'usv_supercategory': [1],
        })
        out = find_bout_epochs(root_directories=[str(sess)],
                               mouse_ids_dict={},
                               camera_fps_dict={'sess_C': 100.0},
                               features_dict={'sess_C': _features_df(100)},
                               prediction_mode='bout', filter_history=1.0,
                               usv_bout_time=0.5, min_usv_per_bout=2,
                               proportion_smoothing_sd=None, gmm_params=_gmm_params())
        assert out['sess_C'] == {}
        assert 'No mouse names registered' in capsys.readouterr().out

    def test_unknown_prediction_mode_raises(self, tmp_path):
        """An unsupported ``prediction_mode`` raises ``ValueError``."""

        kwargs = self._build(tmp_path, {
            'emitter': ['male'], 'start': [2.0], 'stop': [2.05],
            'usv_category': [1], 'usv_supercategory': [1],
        })
        with pytest.raises(ValueError, match='Unknown prediction_mode'):
            find_bout_epochs(prediction_mode='bogus', filter_history=1.0,
                             usv_bout_time=0.5, min_usv_per_bout=2,
                             proportion_smoothing_sd=None, gmm_params=_gmm_params(),
                             **kwargs)

    def test_invalid_gmm_component_index_raises(self, tmp_path):
        """An out-of-range ``gmm_component_index`` raises ``ValueError``."""

        kwargs = self._build(tmp_path, {
            'emitter': ['male'], 'start': [2.0], 'stop': [2.05],
            'usv_category': [1], 'usv_supercategory': [1],
        })
        with pytest.raises(ValueError, match='Invalid gmm_component_index'):
            find_bout_epochs(prediction_mode='bout', filter_history=1.0,
                             usv_bout_time=0.5, min_usv_per_bout=2,
                             gmm_component_index=5,
                             proportion_smoothing_sd=None, gmm_params=_gmm_params(),
                             **kwargs)

    def test_bout_mode_rejects_bout_with_dirty_history(self, tmp_path):
        """A multi-syllable bout whose immediately-preceding USV is within
        ``filter_history`` is rejected (dirty pre-bout history)."""

        rows = {
            'emitter': ['male', 'male', 'male'],
            'start': [2.0, 2.5, 2.6],
            'stop': [2.05, 2.55, 2.65],
            'usv_category': [1, 1, 1],
            'usv_supercategory': [1, 1, 1],
        }
        kwargs = self._build(tmp_path, rows)
        # IBI threshold ~ exp(log(0.3)) so the first USV breaks into its own
        # single-syllable bout and the [2.5, 2.6] bout has a prior USV at 2.05,
        # only 0.45 s away (< filter_history = 1.0) -> rejected.
        gmm = {'male': {'means': [np.log(0.3)], 'sds': [0.0]},
               'female': {'means': [np.log(0.3)], 'sds': [0.0]}}
        out = find_bout_epochs(prediction_mode='bout', filter_history=1.0,
                               usv_bout_time=0.5, min_usv_per_bout=2,
                               proportion_smoothing_sd=None, gmm_params=gmm,
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
            'usv_supercategory': [1, 1],
        }
        kwargs = self._build(tmp_path, rows)
        out = find_bout_epochs(prediction_mode='individual', filter_history=1.0,
                               usv_bout_time=0.5, min_usv_per_bout=2,
                               proportion_smoothing_sd=2.0, gmm_params=_gmm_params(),
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
            'usv_category': [1], 'usv_supercategory': [1],
        }
        kwargs = self._build(tmp_path, rows)
        out = find_bout_epochs(prediction_mode='bout', filter_history=1.0,
                               usv_bout_time=0.5, min_usv_per_bout=1,
                               proportion_smoothing_sd=None, gmm_params=_gmm_params(),
                               **kwargs)
        np.testing.assert_allclose(out['sess_B']['male']['positive_events'], [2.0])

    def test_state_mode_empty_when_no_valid_samples(self, tmp_path, capsys):
        """When the grid yields no valid frame samples (``filter_history``
        exceeds session duration), 'state' mode returns empty event arrays."""

        rows = {
            'emitter': ['male'], 'start': [0.1], 'stop': [0.15],
            'usv_category': [1], 'usv_supercategory': [1],
        }
        # n_frames=50 at 100 fps -> 0.5 s session; filter_history=2.0 > 0.5.
        kwargs = self._build(tmp_path, rows, n_frames=50, fps=100.0)
        out = find_bout_epochs(prediction_mode='state', filter_history=2.0,
                               usv_bout_time=0.5, min_usv_per_bout=2,
                               proportion_smoothing_sd=None, gmm_params=_gmm_params(),
                               **kwargs)
        male = out['sess_B']['male']
        assert male['positive_events'].size == 0
        assert male['negative_events'].size == 0
        assert 'No valid' in capsys.readouterr().out


# find_usv_categories


class TestFindUsvCategories:

    def _kwargs(self, tmp_path, rows: dict, n_frames: int = 600, fps: float = 100.0):
        """Builds a one-session tree and returns the common loader kwargs for
        the ``find_usv_categories`` tests below."""

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
            'usv_supercategory': [1, 1, 1],
        }
        out = find_usv_categories(target_category=1, filter_history=1.0,
                                  **self._kwargs(tmp_path, rows))
        male = out['sess_D']['male']
        np.testing.assert_allclose(male['target_events'], [2.0, 4.0])
        np.testing.assert_allclose(male['other_events'], [3.0])
        assert set(male['events_by_category'].keys()) == {1, 2}
        assert 1 in male['category_streams']

    def test_multinomial_mode_events_by_category(self, tmp_path):
        """With ``target_category=None`` the function populates
        ``events_by_category`` for every category and leaves target/other
        unset."""

        rows = {
            'emitter': ['male', 'male'],
            'start': [2.0, 3.0],
            'stop': [2.05, 3.05],
            'usv_category': [5, 8],
            'usv_supercategory': [1, 1],
        }
        out = find_usv_categories(target_category=None, filter_history=1.0,
                                  **self._kwargs(tmp_path, rows))
        male = out['sess_D']['male']
        assert male['target_events'] is None
        assert set(male['events_by_category'].keys()) == {5, 8}

    @pytest.mark.filterwarnings("ignore::astropy.utils.exceptions.AstropyUserWarning")
    def test_continuous_targets_from_manifold_columns(self, tmp_path):
        """When ``manifold_column_names`` are present, continuous onsets,
        stacked targets and the derived super/category label arrays are
        written."""

        rows = {
            'emitter': ['male', 'male'],
            'start': [2.0, 3.0],
            'stop': [2.05, 3.05],
            'usv_category': [1, 2],
            'usv_supercategory': [1, 1],
            'vae_umap1': [0.1, 0.2],
            'vae_umap2': [0.3, 0.4],
            'vae_supercategory': [10, 11],
            'vae_category': [20, 21],
        }
        out = find_usv_categories(target_category=None, filter_history=1.0,
                                  vocal_output_type='all_rate',
                                  proportion_smoothing_sd=2.0,
                                  manifold_column_names=['vae_umap1', 'vae_umap2'],
                                  **self._kwargs(tmp_path, rows))
        male = out['sess_D']['male']
        np.testing.assert_allclose(male['continuous_onsets'], [2.0, 3.0])
        assert male['continuous_targets'].shape == (2, 2)
        np.testing.assert_allclose(male['continuous_supercategory'], [10, 11])
        np.testing.assert_allclose(male['continuous_category'], [20, 21])
        assert 'usv_rate' in male['continuous_vocal_signals']
        assert 'usv_cat_1' in male['continuous_vocal_signals']

    def test_missing_category_column_raises(self, tmp_path):
        """A CSV lacking ``category_column`` raises ``ValueError``."""

        rows = {
            'emitter': ['male'], 'start': [2.0], 'stop': [2.05],
            'usv_supercategory': [1],
        }
        with pytest.raises(ValueError, match='missing'):
            find_usv_categories(target_category=1, filter_history=1.0,
                                **self._kwargs(tmp_path, rows))

    def test_filter_history_removes_early_usvs(self, tmp_path):
        """USVs starting at/under ``filter_history`` are discarded, leaving a
        mouse with only late USVs in the output."""

        rows = {
            'emitter': ['male', 'male'],
            'start': [0.5, 3.0],
            'stop': [0.55, 3.05],
            'usv_category': [1, 1],
            'usv_supercategory': [1, 1],
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
            'usv_supercategory': [1, 1],
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
            'usv_category': [1], 'usv_supercategory': [1],
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
            'usv_category': [1], 'usv_supercategory': [1],
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
            'usv_category': [1], 'usv_supercategory': [1],
        }
        out = find_usv_categories(target_category=1, filter_history=1.0,
                                  **self._kwargs(tmp_path, rows))
        assert out['sess_D']['male']['events_by_category'] == {}
        assert out['sess_D']['male']['target_events'] is None

    def test_noise_categories_filtered_globally(self, tmp_path):
        """USVs whose ``usv_supercategory`` is in ``noise_vocal_categories``
        are removed before any per-mouse processing."""

        rows = {
            'emitter': ['male', 'male'],
            'start': [2.0, 3.0], 'stop': [2.05, 3.05],
            'usv_category': [1, 2], 'usv_supercategory': [0, 1],
        }
        out = find_usv_categories(target_category=None, filter_history=1.0,
                                  noise_vocal_categories=[0],
                                  **self._kwargs(tmp_path, rows))
        male = out['sess_D']['male']
        # Only the supercategory-1 USV (usv_category 2) survives.
        assert set(male['events_by_category'].keys()) == {2}

    @pytest.mark.filterwarnings("ignore::astropy.utils.exceptions.AstropyUserWarning")
    def test_categories_rate_skips_non_integer_category(self, tmp_path):
        """A non-integer category is skipped in the per-category signal loop so
        no 'usv_cat_*' key is created for it."""

        rows = {
            'emitter': ['male', 'male'],
            'start': [2.0, 3.0], 'stop': [2.05, 3.05],
            'usv_category': ['p', 'p'], 'usv_supercategory': [1, 1],
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
            'usv_category': ['weird'], 'usv_supercategory': [1],
        }
        out = find_usv_categories(target_category=None, filter_history=1.0,
                                  **self._kwargs(tmp_path, rows))
        male = out['sess_D']['male']
        assert male['events_by_category'] == {}
        assert 'weird' in male['category_streams']


# find_variable_length_bouts


class TestFindVariableLengthBouts:

    def _kwargs(self, tmp_path, rows: dict, n_frames: int = 600, fps: float = 100.0):
        """Builds a one-session tree and returns the common loader kwargs for
        the ``find_variable_length_bouts`` tests below."""

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
            'usv_supercategory': [1, 1, 1],
            'mask_number': [2, 3, 5],
        }
        out = find_variable_length_bouts(min_vocalizations=2, filter_history=1.0,
                                         gmm_params=_gmm_params(),
                                         **self._kwargs(tmp_path, rows))
        male = out['sess_E']['male']
        np.testing.assert_allclose(male['bout_onsets'], [2.0])
        np.testing.assert_allclose(male['bout_durations'], [2.25 - 2.0])
        np.testing.assert_allclose(male['bout_syllable_counts'], [3])
        np.testing.assert_allclose(male['total_mask_complexity'], [10])
        np.testing.assert_allclose(male['mean_mask_complexity'], [10 / 3])

    def test_below_min_vocalizations_dropped(self, tmp_path):
        """A bout with fewer than ``min_vocalizations`` syllables is dropped,
        yielding empty bout arrays."""

        rows = {
            'emitter': ['male'],
            'start': [2.0], 'stop': [2.05],
            'usv_category': [1], 'usv_supercategory': [1], 'mask_number': [1],
        }
        out = find_variable_length_bouts(min_vocalizations=2, filter_history=1.0,
                                         gmm_params=_gmm_params(),
                                         **self._kwargs(tmp_path, rows))
        assert out['sess_E']['male']['bout_onsets'].size == 0

    def test_missing_mask_column_uses_unit_complexity(self, tmp_path, capsys):
        """When ``mask_number`` is absent the loader warns and falls back to
        unit mask complexity (count == total complexity)."""

        rows = {
            'emitter': ['male', 'male'],
            'start': [2.0, 2.1], 'stop': [2.05, 2.15],
            'usv_category': [1, 1], 'usv_supercategory': [1, 1],
        }
        out = find_variable_length_bouts(min_vocalizations=2, filter_history=1.0,
                                         gmm_params=_gmm_params(),
                                         **self._kwargs(tmp_path, rows))
        assert 'mask_number' in capsys.readouterr().out
        np.testing.assert_allclose(out['sess_E']['male']['total_mask_complexity'], [2])

    @pytest.mark.filterwarnings("ignore::astropy.utils.exceptions.AstropyUserWarning")
    def test_all_rate_signals_and_noise_filter(self, tmp_path):
        """``vocal_output_type='all_rate'`` writes pooled and per-category
        signals, and ``noise_vocal_categories`` drops the noise rows first."""

        rows = {
            'emitter': ['male', 'male', 'male'],
            'start': [2.0, 2.1, 2.2],
            'stop': [2.05, 2.15, 2.25],
            'usv_category': [1, 1, 4],
            'usv_supercategory': [1, 1, 0],
            'mask_number': [1, 1, 1],
        }
        out = find_variable_length_bouts(min_vocalizations=2, filter_history=1.0,
                                         gmm_params=_gmm_params(),
                                         proportion_smoothing_sd=2.0,
                                         vocal_output_type='all_rate',
                                         noise_vocal_categories=[0],
                                         **self._kwargs(tmp_path, rows))
        signals = out['sess_E']['male']['continuous_vocal_signals']
        assert 'usv_rate' in signals
        assert 'usv_cat_1' in signals
        # The noise category (supercategory 0 -> usv_category 4) is excluded.
        assert 'usv_cat_4' not in signals

    def test_ibi_threshold_recorded(self, tmp_path):
        """The dynamically-computed IBI threshold is stored per mouse."""

        rows = {
            'emitter': ['male'], 'start': [2.0], 'stop': [2.05],
            'usv_category': [1], 'usv_supercategory': [1], 'mask_number': [1],
        }
        out = find_variable_length_bouts(min_vocalizations=2, filter_history=1.0,
                                         gmm_params=_gmm_params(),
                                         **self._kwargs(tmp_path, rows))
        # exp(0 + 2.58 * 0) == 1.0 for the chosen GMM params.
        assert out['sess_E']['male']['ibi_threshold_used'] == pytest.approx(1.0)

    def test_invalid_gmm_component_index_raises(self, tmp_path):
        """An out-of-range ``gmm_component_index`` raises ``ValueError``."""

        rows = {
            'emitter': ['male'], 'start': [2.0], 'stop': [2.05],
            'usv_category': [1], 'usv_supercategory': [1], 'mask_number': [1],
        }
        with pytest.raises(ValueError, match='Invalid gmm_component_index'):
            find_variable_length_bouts(min_vocalizations=2, filter_history=1.0,
                                       gmm_component_index=9, gmm_params=_gmm_params(),
                                       **self._kwargs(tmp_path, rows))

    def test_missing_summary_csv_skips_session(self, tmp_path, capsys):
        """A session lacking the USV summary CSV is skipped with a warning."""

        sess = tmp_path / 'sess_F'
        (sess / 'audio').mkdir(parents=True)
        out = find_variable_length_bouts(
            root_directories=[str(sess)],
            mouse_ids_dict={'sess_F': ['male', 'female']},
            camera_fps_dict={'sess_F': 100.0},
            features_dict={'sess_F': _features_df(100)},
            min_vocalizations=2, filter_history=1.0, gmm_params=_gmm_params())
        assert out['sess_F'] == {}
        assert 'No USV summary' in capsys.readouterr().out

    def test_unregistered_session_skipped(self, tmp_path, capsys):
        """A session absent from ``mouse_ids_dict`` is skipped with a
        warning."""

        rows = {
            'emitter': ['male'], 'start': [2.0], 'stop': [2.05],
            'usv_category': [1], 'usv_supercategory': [1], 'mask_number': [1],
        }
        kwargs = self._kwargs(tmp_path, rows)
        kwargs['mouse_ids_dict'] = {}
        out = find_variable_length_bouts(min_vocalizations=2, filter_history=1.0,
                                         gmm_params=_gmm_params(), **kwargs)
        assert out['sess_E'] == {}
        assert 'No mouse names registered' in capsys.readouterr().out

    def test_pooled_binary_signal(self, tmp_path):
        """``vocal_output_type='pooled_binary'`` writes a binary 'usv_event'
        aggregate trace."""

        rows = {
            'emitter': ['male', 'male'],
            'start': [2.0, 2.1], 'stop': [2.05, 2.15],
            'usv_category': [1, 1], 'usv_supercategory': [1, 1],
            'mask_number': [1, 1],
        }
        out = find_variable_length_bouts(min_vocalizations=2, filter_history=1.0,
                                         gmm_params=_gmm_params(),
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
            'usv_category': ['x', 'x'], 'usv_supercategory': [1, 1],
            'mask_number': [1, 1],
        }
        out = find_variable_length_bouts(min_vocalizations=2, filter_history=1.0,
                                         gmm_params=_gmm_params(),
                                         proportion_smoothing_sd=2.0,
                                         vocal_output_type='categories_rate',
                                         **self._kwargs(tmp_path, rows))
        signals = out['sess_E']['male']['continuous_vocal_signals']
        assert not any(k.startswith('usv_cat_') for k in signals)
