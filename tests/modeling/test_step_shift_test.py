"""
@author: bartulem
Tests of the per-step shift test of forward feature selection and its plumbing.

Covered:

* the window helpers the extractions and the test share (``valid_window_starts``,
  ``cut_history_windows``, ``circular_shift_trace``) and the trace sidecar
  (``write_feature_traces`` / ``read_feature_trace`` / ``verify_windows_against_traces``);
* the test's arithmetic (``shift_refit_count``, ``bonferroni_permutation_p``,
  ``draw_session_shift_lags``, ``macro_log_likelihood``, ``pool_worker_count``);
* ``SelectionPool`` (serial path, early stop, fork pool on Linux);
* ``run_step_shift_test`` on a toy refit: acceptance with a real signal, early rejection of a
  null feature, lag reproducibility, partial-file resume and fingerprint mismatch;
* the onset selector's guards (``_require_shift_test_inputs``, ``_require_step_test_rule``,
  ``_quarantine_stale_step_files``, ``_assemble_fold_metrics``, ``_step_test_settings``);
* an end-to-end ``vocal_onset_model_selection`` on a tiny synthetic cohort whose windows are cut
  from stored traces, including a resume and the refusal of a pickle without the sidecar.
"""

from __future__ import annotations

import importlib
import json
import os
import pickle
import subprocess
import sys
import textwrap
import warnings
from pathlib import Path

import h5py
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from tests.modeling._synth import (
    build_modeling_settings,
    build_session_tree,
    write_session_list_file,
)
from usv_playpen.modeling import modeling_utils as mu
from usv_playpen.modeling.modeling_metadata import SCHEMA_VERSIONS

# The modeling import chain pulls optax -> a one-time JAX DeprecationWarning; the
# suite escalates warnings to errors, so the import is guarded here as the other
# selection tests do.
with warnings.catch_warnings():
    warnings.simplefilter('ignore', DeprecationWarning)
    from usv_playpen.modeling import model_selection as ms
    from usv_playpen.modeling import step_shift_test as sst
    from usv_playpen.modeling.model_selection import vocal_onset_model_selection

# Toy cohort geometry of the end-to-end test: 6-frame windows at 12 fps, four sessions
# of 600 frames, one of them reserved as held-out.
CAMERA_FPS = 12.0
FILTER_HISTORY = 0.5
HISTORY_FRAMES = int(np.floor(CAMERA_FPS * FILTER_HISTORY))
SESSION_FRAMES = 600
N_SESSIONS = 4
MIN_LAG_SECONDS = 2.0
ONSET_SLOTS = {'usv': 'usv_feature_arr', 'no_usv': 'no_usv_feature_arr'}


# ---------------------------------------------------------------------------------
# toy data builders
# ---------------------------------------------------------------------------------

def _toy_sessions(rng: np.random.Generator,
                  n_sessions: int = 3,
                  n_frames: int = 3000,
                  history_frames: int = 20,
                  n_events: int = 40,
                  signal: float = 1.5) -> tuple[dict, dict, dict]:
    """
    Description
    -----------
    Builds the trace / start-frame / window triple of one feature over a few sessions,
    with a class signal written INTO the trace: a positive bump over the history
    window of every positive event, so the windows cut at the stored starts separate
    the classes while a shifted trace does not.

    Parameters
    ----------
    rng (np.random.Generator)
        Seeded generator.
    n_sessions (int)
        Sessions to build.
    n_frames (int)
        Trace length per session.
    history_frames (int)
        Window length.
    n_events (int)
        Events per session (half positive, half negative), windows non-overlapping.
    signal (float)
        Bump height added over positive windows (0 = a feature carrying no signal).

    Returns
    -------
    traces (dict)
        ``{session_id: (n_frames,) trace}``.
    starts (dict)
        ``{session_id: {'usv': starts, 'no_usv': starts}}``.
    windows (dict)
        ``{session_id: {'usv_feature_arr': windows, 'no_usv_feature_arr': windows}}``.
    """

    traces, starts, windows = {}, {}, {}
    spacing = history_frames + 4
    for s_idx in range(n_sessions):
        session_id = f'session_{s_idx}'
        trace = rng.standard_normal(n_frames)
        candidate_starts = np.arange(history_frames, n_frames - history_frames - spacing, spacing)
        chosen = np.sort(rng.choice(candidate_starts, size=n_events, replace=False))
        labels = np.zeros(n_events, dtype=int)
        labels[rng.choice(n_events, size=n_events // 2, replace=False)] = 1
        for start, label in zip(chosen, labels, strict=True):
            if label == 1:
                trace[start:start + history_frames] += signal
        pos, neg = chosen[labels == 1], chosen[labels == 0]
        traces[session_id] = trace
        starts[session_id] = {'usv': pos, 'no_usv': neg}
        windows[session_id] = {
            'usv_feature_arr': mu.cut_history_windows(trace, pos, history_frames),
            'no_usv_feature_arr': mu.cut_history_windows(trace, neg, history_frames),
        }
    return traces, starts, windows


def _toy_refit_factory(windows: dict):
    """
    Description
    -----------
    Returns the ``refit_and_score`` callable of the toy shift test: a logistic
    regression on the window mean, scored by its in-sample mean Bernoulli
    log-likelihood per event, reading the stored windows unless an override dict is
    given.

    Parameters
    ----------
    windows (dict)
        Stored ``{session_id: {array_key: windows}}``.

    Returns
    -------
    refit_and_score (callable)
        ``refit_and_score(override: dict | None) -> float``.
    """

    def refit_and_score(override: dict | None = None) -> float:
        """
        Description
        -----------
        Fits the toy model on the stored or overridden windows and returns the
        in-sample mean log-likelihood per event.

        Parameters
        ----------
        override (dict | None)
            Per-session replacement windows.

        Returns
        -------
        score (float)
            Mean Bernoulli log-likelihood per event.
        """

        source = windows if override is None else override
        X_parts, y_parts = [], []
        for session_id in sorted(source):
            for array_key, label in (('usv_feature_arr', 1), ('no_usv_feature_arr', 0)):
                arr = source[session_id][array_key]
                X_parts.append(arr.mean(axis=1)[:, None])
                y_parts.append(np.full(arr.shape[0], label))
        X, y = np.concatenate(X_parts), np.concatenate(y_parts)
        model = LogisticRegression(C=1.0).fit(X, y)
        p = np.clip(model.predict_proba(X)[:, 1], 1e-15, 1 - 1e-15)
        return float(np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    return refit_and_score


def _constant_model_score(starts: dict) -> float:
    """
    Description
    -----------
    Mean log-likelihood per event of the constant model at the class prior of the toy
    cohort (the incumbent of the first step).

    Parameters
    ----------
    starts (dict)
        ``{session_id: {'usv': starts, 'no_usv': starts}}``.

    Returns
    -------
    score (float)
        The constant model's mean log-likelihood.
    """

    y = np.concatenate([np.concatenate((np.ones(len(e['usv'])), np.zeros(len(e['no_usv']))))
                        for e in starts.values()])
    prior = float(y.mean())
    return float(np.mean(y * np.log(prior) + (1 - y) * np.log(1 - prior)))


def _run_toy_shift_test(tmp_path: Path, *, traces: dict, starts: dict, windows: dict,
                        history_frames: int, n_workers: int = 1, m_candidates: int = 3,
                        alpha: float = 0.05, min_lag_frames: int = 100, random_seed: int = 0,
                        step_index: int = 1, checkpoint_every: int = 5,
                        refit_and_score=None, partial_name: str = 'step_1_null_partial.pkl') -> dict:
    """
    Description
    -----------
    Runs ``run_step_shift_test`` on the toy feature with the toy refit and returns its
    record.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Scratch directory for the partial file.
    traces (dict)
        Toy traces.
    starts (dict)
        Toy start frames.
    windows (dict)
        Toy stored windows.
    history_frames (int)
        Window length.
    n_workers (int)
        Pool size.
    m_candidates (int)
        Bonferroni factor.
    alpha (float)
        Level.
    min_lag_frames (int)
        Lag floor.
    random_seed (int)
        Seed.
    step_index (int)
        Step.
    checkpoint_every (int)
        Partial-write cadence.
    refit_and_score (callable | None)
        Replacement refit (default: the toy logistic refit).
    partial_name (str)
        Partial file name under ``tmp_path``.

    Returns
    -------
    record (dict)
        The ``step_test`` block.
    """

    refit = refit_and_score if refit_and_score is not None else _toy_refit_factory(windows)
    sst.set_pool_context()
    return sst.run_step_shift_test(
        n_workers=n_workers,
        refit_and_score=refit,
        incumbent_features=[],
        incumbent_score=_constant_model_score(starts),
        winner='toy',
        observed_score=refit(None),
        statistic_name='toy_log_likelihood',
        winner_traces=traces,
        window_starts=starts,
        slots=ONSET_SLOTS,
        history_frames=history_frames,
        m_candidates=m_candidates,
        alpha=alpha,
        min_lag_frames=min_lag_frames,
        random_seed=random_seed,
        step_index=step_index,
        checkpoint_every=checkpoint_every,
        partial_path=tmp_path / partial_name,
        label='toy',
    )


# ---------------------------------------------------------------------------------
# window helpers and the trace sidecar
# ---------------------------------------------------------------------------------

class TestWindowHelpers:
    """``valid_window_starts``, ``cut_history_windows`` and ``circular_shift_trace``."""

    def test_valid_window_starts_drops_edge_events(self):
        """Events whose window would start before frame 0 or end past the trace are dropped."""

        starts, mask = mu.valid_window_starts(np.array([3, 6, 10, 11]), history_frames=6, n_frames=10)
        assert starts.tolist() == [0, 4]
        assert mask.tolist() == [False, True, True, False]

    def test_cut_history_windows_matches_slice_loop_with_nan_fill(self):
        """The gathered windows equal a per-event slice with NaN replaced by 0, in the trace's dtype."""

        trace = np.arange(30, dtype=float)
        trace[7] = np.nan
        starts = np.array([0, 5, 24])
        windows = mu.cut_history_windows(trace, starts, 6)
        expected = np.stack([np.nan_to_num(trace[s:s + 6], nan=0.0) for s in starts])
        assert windows.shape == (3, 6)
        assert np.array_equal(windows, expected)
        assert windows.dtype == trace.dtype
        assert windows[1, 2] == 0.0

    def test_cut_history_windows_empty_and_out_of_range(self):
        """No start gives a ``(0, H)`` array; a start outside the trace raises."""

        trace = np.arange(20, dtype=float)
        assert mu.cut_history_windows(trace, np.array([], dtype=int), 6).shape == (0, 6)
        with pytest.raises(ValueError, match='outside the trace'):
            mu.cut_history_windows(trace, np.array([15]), 6)
        with pytest.raises(ValueError, match='outside the trace'):
            mu.cut_history_windows(trace, np.array([-1]), 6)

    def test_circular_shift_trace_is_np_roll(self):
        """Frame ``t`` of the shifted trace is frame ``t - lag`` of the input, wrapping at the end."""

        trace = np.arange(10, dtype=float)
        shifted = mu.circular_shift_trace(trace, 3)
        assert np.array_equal(shifted, np.roll(trace, 3))
        assert shifted[3] == trace[0]
        assert shifted[0] == trace[7]
        assert np.array_equal(mu.circular_shift_trace(trace, 0), trace)


class TestTraceSidecar:
    """``write_feature_traces`` / ``read_feature_trace`` / ``verify_windows_against_traces``."""

    def test_round_trip_and_attributes(self, tmp_path):
        """Traces and window starts come back bit-exact per feature, session and slot, with the attributes."""

        rng = np.random.default_rng(1)
        traces = {'self.speed': {'a': rng.standard_normal(50), 'b': rng.standard_normal(70)},
                  'head-head': {'a': rng.standard_normal(50), 'b': rng.standard_normal(70)}}
        window_starts = {'a': {'usv': [3, 20], 'no_usv': [], 'n_frames': 50, 'fps': 12.0},
                         'b': {'usv': np.array([5]), 'no_usv': np.array([30, 41, 52]), 'n_frames': 70, 'fps': 12.0}}
        path = tmp_path / 'traces.h5'
        mu.write_feature_traces(path, traces, window_starts)
        for feature, by_session in traces.items():
            for session_id, column in by_session.items():
                assert np.array_equal(mu.read_feature_trace(path, feature, session_id), column)
        with h5py.File(path, 'r') as handle:
            assert handle['head-head']['b'].attrs['n_frames'] == 70
            assert handle['head-head']['b'].attrs['fps'] == 12.0
            assert set(handle[mu.WINDOW_STARTS_GROUP]) == {'a', 'b'}
        read_back = mu.read_window_starts(path)
        assert set(read_back) == {'a', 'b'}
        for session_id, entry in window_starts.items():
            assert set(read_back[session_id]) == {'usv', 'no_usv', 'n_frames', 'fps'}
            assert read_back[session_id]['n_frames'] == entry['n_frames']
            assert read_back[session_id]['fps'] == entry['fps']
            for slot in ('usv', 'no_usv'):
                assert read_back[session_id][slot].dtype == np.int64
                assert np.array_equal(read_back[session_id][slot], np.asarray(entry[slot], dtype=np.int64))
        # the pickle's metadata block never carries the starts: the sidecar alone does
        assert isinstance(read_back['a']['usv'], np.ndarray)

    def test_write_refuses_reserved_key_and_unknown_session(self, tmp_path):
        """A feature named like the start group, or a trace of a session without starts, is refused."""

        starts = {'a': {'X': [1], 'n_frames': 10, 'fps': 12.0}}
        with pytest.raises(ValueError, match='reserved'):
            mu.write_feature_traces(tmp_path / 'r.h5', {mu.WINDOW_STARTS_GROUP: {'a': np.zeros(10)}}, starts)
        with pytest.raises(ValueError, match='no window starts'):
            mu.write_feature_traces(tmp_path / 'u.h5', {'f': {'a': np.zeros(10), 'b': np.zeros(10)}}, starts)

    def test_read_window_starts_refuses_sidecar_without_starts(self, tmp_path):
        """A sidecar holding traces only (written before the starts moved into it) is refused."""

        path = tmp_path / 'old.h5'
        with h5py.File(path, 'w') as handle:
            handle.create_group('f').create_dataset('a', data=np.zeros(10))
        with pytest.raises(ValueError, match='predates'):
            mu.read_window_starts(path)

    def test_verify_passes_on_consistent_data(self, tmp_path):
        """Windows cut from the stored traces at the stored starts verify at lag zero."""

        rng = np.random.default_rng(2)
        traces, starts, windows = _toy_sessions(rng, n_sessions=2, n_frames=500, history_frames=8, n_events=10)
        path = tmp_path / 'traces.h5'
        mu.write_feature_traces(path, {'f': traces},
                                {s: {**starts[s], 'n_frames': len(t), 'fps': 12.0} for s, t in traces.items()})
        mu.verify_windows_against_traces({'f': windows}, path, starts, ONSET_SLOTS, 8,
                                         features=['f'], sessions=sorted(traces))

    def test_verify_raises_on_modified_window_and_count_mismatch(self, tmp_path):
        """A changed stored window, or starts without windows, abort the check."""

        rng = np.random.default_rng(3)
        traces, starts, windows = _toy_sessions(rng, n_sessions=2, n_frames=500, history_frames=8, n_events=10)
        path = tmp_path / 'traces.h5'
        mu.write_feature_traces(path, {'f': traces},
                                {s: {**starts[s], 'n_frames': len(t), 'fps': 12.0} for s, t in traces.items()})
        tampered = {'f': {s: {k: v.copy() for k, v in e.items()} for s, e in windows.items()}}
        tampered['f']['session_0']['usv_feature_arr'][0, 0] += 1e-6
        with pytest.raises(ValueError, match='do not match the stored windows'):
            mu.verify_windows_against_traces(tampered, path, starts, ONSET_SLOTS, 8,
                                             features=['f'], sessions=['session_0', 'session_1'])
        emptied = {'f': {s: {k: v.copy() for k, v in e.items()} for s, e in windows.items()}}
        emptied['f']['session_1']['no_usv_feature_arr'] = np.empty((0, 8))
        with pytest.raises(ValueError, match='stores no windows'):
            mu.verify_windows_against_traces(emptied, path, starts, ONSET_SLOTS, 8,
                                             features=['f'], sessions=['session_0', 'session_1'])


# ---------------------------------------------------------------------------------
# the test's arithmetic
# ---------------------------------------------------------------------------------

class TestShiftArithmetic:
    """``shift_refit_count``, ``bonferroni_permutation_p``, ``draw_session_shift_lags``,
    ``macro_log_likelihood`` and ``pool_worker_count``."""

    @pytest.mark.parametrize(('m', 'alpha', 'expected'), [(15, 0.01, 1500), (9, 0.01, 900), (15, 0.05, 300)])
    def test_shift_refit_count(self, m, alpha, expected):
        """``B = floor(m / alpha)``: passes only with zero exceedances, never with one."""

        n_refits = mu.shift_refit_count(m, alpha)
        assert n_refits == expected
        assert m * 1 / (n_refits + 1) < alpha
        assert m * 2 / (n_refits + 1) > alpha

    def test_shift_refit_count_rejects_bad_inputs(self):
        """``m < 1`` or a level outside ``(0, 1)`` raise."""

        with pytest.raises(ValueError, match='need m >= 1'):
            mu.shift_refit_count(0, 0.05)
        with pytest.raises(ValueError, match='need m >= 1'):
            mu.shift_refit_count(3, 1.0)

    def test_bonferroni_permutation_p(self):
        """``p = (1 + k) / (n + 1)``, ``p_adj = min(1, m p)``."""

        p, p_adj = mu.bonferroni_permutation_p(0, 1500, 15)
        assert p == pytest.approx(1 / 1501)
        assert p_adj == pytest.approx(15 / 1501)
        assert p_adj < 0.01
        p, p_adj = mu.bonferroni_permutation_p(2, 10, 3)
        assert p == pytest.approx(3 / 11)
        assert p_adj == pytest.approx(9 / 11)
        assert mu.bonferroni_permutation_p(9, 9, 5)[1] == 1.0

    def test_draw_session_shift_lags_range_seed_and_independence(self):
        """Lags lie in ``[min_lag, T - min_lag]``, reproduce from the seed and differ across sessions."""

        lengths = {'a': 100, 'b': 100, 'c': 1000}
        lags_1 = mu.draw_session_shift_lags(np.random.default_rng([0, 1, 2]), lengths, 10)
        lags_2 = mu.draw_session_shift_lags(np.random.default_rng([0, 1, 2]), lengths, 10)
        assert lags_1 == lags_2
        assert set(lags_1) == set(lengths)
        for session_id, n_frames in lengths.items():
            assert 10 <= lags_1[session_id] <= n_frames - 10
        draws = [mu.draw_session_shift_lags(np.random.default_rng(seed), {'a': 100, 'b': 100}, 10) for seed in range(50)]
        assert any(d['a'] != d['b'] for d in draws)
        assert len({d['a'] for d in draws}) > 1

    def test_draw_session_shift_lags_raises_on_short_session(self):
        """A session shorter than ``2 x min_lag`` has no admissible lag."""

        with pytest.raises(ValueError, match='no admissible lag'):
            mu.draw_session_shift_lags(np.random.default_rng(0), {'short': 15}, 10)
        with pytest.raises(ValueError, match='min_lag_frames must be'):
            mu.draw_session_shift_lags(np.random.default_rng(0), {'a': 100}, 0)

    def test_macro_log_likelihood_weights_classes_equally(self):
        """A rare class contributes its own class mean, as much as the common class."""

        y = np.array([0, 0, 0, 1])
        proba = np.array([[0.8, 0.2], [0.5, 0.5], [0.2, 0.8], [0.1, 0.9]])
        classes = np.array([0, 1])
        class_0 = np.mean(np.log([0.8, 0.5, 0.2]))
        class_1 = np.log(0.9)
        expected = 0.5 * (class_0 + class_1)
        assert mu.macro_log_likelihood(y, proba, classes) == pytest.approx(expected)
        plain = np.mean(np.log([0.8, 0.5, 0.2, 0.9]))
        assert mu.macro_log_likelihood(y, proba, classes) != pytest.approx(plain)

    def test_macro_log_likelihood_rejects_shape_mismatch(self):
        """The probability matrix must have one column per class."""

        with pytest.raises(ValueError, match='y_proba must be'):
            mu.macro_log_likelihood(np.array([0, 1]), np.array([[0.5], [0.5]]), np.array([0, 1]))

    def test_pool_worker_count_caps(self, monkeypatch):
        """Requested workers, the memory cap and the SLURM environment are all honoured."""

        monkeypatch.delenv('SLURM_CPUS_PER_TASK', raising=False)
        monkeypatch.delenv('SLURM_MEM_PER_NODE', raising=False)
        assert mu.pool_worker_count(4, n_terms=3, memory_gb=100.0, peak_gb_base=2.0, peak_gb_per_term=4.0) == 4
        # 2 + 4 * 9 = 38 GB per fit: a 64 GB budget holds one.
        assert mu.pool_worker_count(12, n_terms=9, memory_gb=64.0, peak_gb_base=2.0, peak_gb_per_term=4.0) == 1
        # 2 + 4 * 3 = 14 GB per fit: 64 GB holds four.
        assert mu.pool_worker_count(12, n_terms=3, memory_gb=64.0, peak_gb_base=2.0, peak_gb_per_term=4.0) == 4
        # Never below one, even when a single fit exceeds the budget.
        assert mu.pool_worker_count(12, n_terms=9, memory_gb=8.0, peak_gb_base=2.0, peak_gb_per_term=4.0) == 1
        # Off the cluster, 0 means the machine's physical memory, never "no cap":
        # a per-fit peak larger than the machine leaves one worker.
        physical_gb = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 1024.0 ** 3
        assert mu.pool_worker_count(64, n_terms=1, memory_gb=0.0, peak_gb_base=0.0,
                                    peak_gb_per_term=2.0 * physical_gb) == 1
        assert mu.pool_worker_count(64, n_terms=1, memory_gb=0.0, peak_gb_base=0.0,
                                    peak_gb_per_term=physical_gb / 3.5) == 3
        monkeypatch.setenv('SLURM_CPUS_PER_TASK', '6')
        monkeypatch.setenv('SLURM_MEM_PER_NODE', str(96 * 1024))
        assert mu.pool_worker_count(0, n_terms=3, memory_gb=0.0, peak_gb_base=2.0, peak_gb_per_term=4.0) == 6
        assert mu.pool_worker_count(0, n_terms=9, memory_gb=0.0, peak_gb_base=2.0, peak_gb_per_term=4.0) == 2


# ---------------------------------------------------------------------------------
# the pool
# ---------------------------------------------------------------------------------

def _square(x: int) -> int:
    """
    Description
    -----------
    Toy pool task.

    Parameters
    ----------
    x (int)
        Input.

    Returns
    -------
    y (int)
        ``x * x``.
    """

    return x * x


def _square_with_pid(x: int) -> tuple:
    """
    Description
    -----------
    Toy pool task reporting the process it ran in.

    Parameters
    ----------
    x (int)
        Input.

    Returns
    -------
    result (tuple)
        ``(x * x, pid)``.
    """

    return x * x, os.getpid()


class TestSelectionPool:
    """``SelectionPool`` on the serial path and, on Linux, the fork pool."""

    def test_serial_map_keeps_order_and_runs_in_process(self):
        """With one worker every task runs in the calling process, results in order."""

        sst.set_pool_context(square=_square_with_pid)
        pool = sst.SelectionPool(1)
        results = pool.map('square', [(i,) for i in range(6)])
        pool.close()
        assert [r[0] for r in results] == [i * i for i in range(6)]
        assert {r[1] for r in results} == {os.getpid()}

    def test_run_until_stops_at_first_true(self):
        """The serial path hands results over in order and stops at the first True."""

        sst.set_pool_context(square=_square)
        seen = []

        def _on_result(value: int) -> bool:
            """
            Description
            -----------
            Records a result and asks to stop once it reaches 9.

            Parameters
            ----------
            value (int)
                A task result.

            Returns
            -------
            stop (bool)
                True when ``value >= 9``.
            """

            seen.append(value)
            return value >= 9

        pool = sst.SelectionPool(1)
        pool.run_until('square', [(i,) for i in range(10)], _on_result)
        pool.close()
        assert seen == [0, 1, 4, 9]

    @pytest.mark.skipif(sys.platform != 'linux', reason='fork-based pool is exercised on Linux only')
    @pytest.mark.filterwarnings('ignore:This process:DeprecationWarning')
    def test_fork_pool_matches_serial(self):
        """
        Two forked workers return the serial results, in order, from other processes.
        Runs in a fresh interpreter: in a process where JAX has already initialized a
        backend the pool falls back to the serial path by design, which is not what is
        under test here.
        """

        script = textwrap.dedent('''
            import json
            import os

            from usv_playpen.modeling import step_shift_test as sst


            def square_with_pid(i):
                return i * i, os.getpid()


            sst.set_pool_context(square=square_with_pid)
            serial = sst.SelectionPool(1).map('square', [(i,) for i in range(8)])
            pool = sst.SelectionPool(2)
            pooled = pool.map('square', [(i,) for i in range(8)])
            pool.close()
            seen = []
            pool = sst.SelectionPool(2)
            pool.run_until('square', [(i,) for i in range(40)], lambda r: seen.append(r[0]) or r[0] >= 25)
            pool.close()
            print(json.dumps({'serial': [r[0] for r in serial], 'pooled': [r[0] for r in pooled],
                              'pids': sorted({r[1] for r in pooled}), 'parent': os.getpid(),
                              'seen': seen, 'workers': pool.n_workers}))
        ''')
        completed = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True,
                                   check=True, timeout=180)
        result = json.loads(completed.stdout.strip().splitlines()[-1])
        assert result['workers'] == 2
        assert result['pooled'] == result['serial'] == [i * i for i in range(8)]
        assert result['parent'] not in result['pids']
        assert 25 in result['seen']
        assert len(result['seen']) < 40

    def test_pool_runs_serially_after_jax_initialized(self, monkeypatch, capsys):
        """Once JAX has installed its at-fork handler here, a multi-worker pool degrades to the serial path and says so."""

        bridge = importlib.import_module('jax._src.xla_bridge')
        monkeypatch.setattr(bridge, '_at_fork_handler_installed', True)
        assert sst.jax_backend_initialized() is True
        pool = sst.SelectionPool(3)
        assert pool.n_workers == 1
        assert 'serially' in capsys.readouterr().out
        sst.set_pool_context(square=_square_with_pid)
        assert pool.map('square', [(3,)]) == [(9, os.getpid())]
        pool.close()


# ---------------------------------------------------------------------------------
# the driver on a toy refit
# ---------------------------------------------------------------------------------

class TestRunStepShiftTest:
    """``run_step_shift_test`` on the toy feature."""

    def test_signal_feature_is_accepted_with_all_draws(self, tmp_path):
        """A feature whose windows carry the class signal passes with ``k == 0`` after all ``B`` draws."""

        traces, starts, windows = _toy_sessions(np.random.default_rng(10))
        record = _run_toy_shift_test(tmp_path, traces=traces, starts=starts, windows=windows, history_frames=20)
        assert record['rule'] == sst.STEP_TEST_RULE
        assert record['n_refits'] == mu.shift_refit_count(3, 0.05) == 60
        assert record['accepted'] is True
        assert record['early_stop'] is False
        assert record['k_exceed'] == 0
        assert record['n_draws_run'] == record['n_refits']
        assert record['t_obs'] > record['t_null'].max()
        assert record['p'] == pytest.approx(1 / 61)
        assert record['p_adjusted'] == pytest.approx(3 / 61)
        assert record['lags'].shape == (60, 3)
        assert record['draw_indices'].tolist() == list(range(60))
        assert record['session_order'] == sorted(traces)
        assert record['refit_seconds'].shape == (60,)
        assert not (tmp_path / 'step_1_null_partial.pkl').exists()
        for key in ('statistic', 'winner', 'incumbent_features', 'm_candidates', 'alpha',
                    'score_incumbent', 'score_winner', 'min_lag_frames', 'seed', 'wall_seconds'):
            assert key in record

    def test_null_feature_is_rejected_early(self, tmp_path):
        """A feature without signal is rejected at the first exceedance, before ``B`` draws."""

        traces, starts, windows = _toy_sessions(np.random.default_rng(11), signal=0.0)
        record = _run_toy_shift_test(tmp_path, traces=traces, starts=starts, windows=windows,
                                     history_frames=20, random_seed=11)
        assert record['accepted'] is False
        assert record['early_stop'] is True
        assert record['k_exceed'] >= 1
        assert record['n_draws_run'] < record['n_refits']
        assert record['p_adjusted'] >= 0.05

    def test_recorded_lags_match_the_seeded_draw(self, tmp_path):
        """Draw ``b`` uses the lags of ``default_rng([seed, step, b])`` over the sorted sessions."""

        traces, starts, windows = _toy_sessions(np.random.default_rng(12))
        record = _run_toy_shift_test(tmp_path, traces=traces, starts=starts, windows=windows,
                                     history_frames=20, random_seed=7, step_index=3,
                                     partial_name='step_3_null_partial.pkl')
        lengths = {s: len(traces[s]) for s in record['session_order']}
        for row, draw_index in enumerate(record['draw_indices']):
            expected = mu.draw_session_shift_lags(np.random.default_rng([7, 3, int(draw_index)]), lengths, 100)
            assert record['lags'][row].tolist() == [expected[s] for s in record['session_order']]
            assert all(100 <= lag <= lengths[s] - 100 for s, lag in expected.items())

    def test_partial_file_resume_and_cleanup(self, tmp_path):
        """A run interrupted after a checkpoint reuses its finished draws, then removes the partial file."""

        traces, starts, windows = _toy_sessions(np.random.default_rng(13))
        base_refit = _toy_refit_factory(windows)
        calls = {'n': 0}

        def _failing_refit(override: dict | None = None) -> float:
            """
            Description
            -----------
            The toy refit that raises on its 8th null call (after the second checkpoint).

            Parameters
            ----------
            override (dict | None)
                Replacement windows (None on the observed refit).

            Returns
            -------
            score (float)
                The toy score.
            """

            if override is not None:
                calls['n'] += 1
                if calls['n'] == 8:
                    msg = 'simulated job kill'
                    raise RuntimeError(msg)
            return base_refit(override)

        partial = tmp_path / 'step_1_null_partial.pkl'
        with pytest.raises(RuntimeError, match='simulated job kill'):
            _run_toy_shift_test(tmp_path, traces=traces, starts=starts, windows=windows, history_frames=20,
                                checkpoint_every=5, refit_and_score=_failing_refit)
        assert partial.is_file()
        with partial.open('rb') as handle:
            saved = pickle.load(handle)
        assert saved['fingerprint']['rule'] == sst.STEP_TEST_RULE
        assert sorted(saved['draws']) == [0, 1, 2, 3, 4]
        saved_gains = {b: saved['draws'][b][1] for b in saved['draws']}

        record = _run_toy_shift_test(tmp_path, traces=traces, starts=starts, windows=windows,
                                     history_frames=20, checkpoint_every=5)
        assert not partial.exists()
        assert record['n_draws_run'] == record['n_refits'] == 60
        for b, gain in saved_gains.items():
            assert record['t_null'][b] == pytest.approx(gain)
        fresh = _run_toy_shift_test(tmp_path, traces=traces, starts=starts, windows=windows,
                                    history_frames=20, checkpoint_every=5)
        assert np.allclose(fresh['t_null'], record['t_null'])
        assert np.array_equal(fresh['lags'], record['lags'])

    def test_partial_file_with_another_fingerprint_is_ignored(self, tmp_path, capsys):
        """A partial file from another winner is not reused; every draw is run."""

        traces, starts, windows = _toy_sessions(np.random.default_rng(14))
        partial = tmp_path / 'step_1_null_partial.pkl'
        with partial.open('wb') as handle:
            pickle.dump({'fingerprint': {'rule': sst.STEP_TEST_RULE, 'winner': 'someone_else'},
                         'draws': {0: (np.zeros(3, dtype=int), 99.0, 0.0)}}, handle)
        record = _run_toy_shift_test(tmp_path, traces=traces, starts=starts, windows=windows, history_frames=20)
        assert 'ignored' in capsys.readouterr().out
        assert record['n_draws_run'] == 60
        assert record['t_null'].max() < 99.0
        assert not partial.exists()


# ---------------------------------------------------------------------------------
# selector guards
# ---------------------------------------------------------------------------------

def _settings_for_guards() -> dict:
    """
    Description
    -----------
    A minimal settings dict holding the six step-test keys.

    Returns
    -------
    settings (dict)
        ``{'model_params': {...}}``.
    """

    return {'model_params': {
        'selection_step_null_min_lag_seconds': 2.0,
        'selection_step_null_checkpoint_every': 50,
        'selection_pool_workers': 0,
        'selection_pool_memory_gb': 0,
        'selection_fit_peak_gb_base': 2,
        'selection_fit_peak_gb_per_term': 4,
    }}


class TestSelectorGuards:
    """``_step_test_settings``, ``_require_shift_test_inputs``, ``_require_step_test_rule``,
    ``_quarantine_stale_step_files`` and ``_assemble_fold_metrics``."""

    def test_step_test_settings_reads_keys_and_rounds_lag(self):
        """All six knobs are read and the lag floor is rounded to frames at the camera rate."""

        cfg = ms._step_test_settings(_settings_for_guards(), 12.0)
        assert cfg == {
            'min_lag_seconds': 2.0, 'min_lag_frames': 24, 'checkpoint_every': 50, 'pool_workers': 0,
            'pool_memory_gb': 0.0, 'fit_peak_gb_base': 2.0, 'fit_peak_gb_per_term': 4.0,
        }
        settings = _settings_for_guards()
        settings['model_params']['selection_step_null_min_lag_seconds'] = 0.1
        assert ms._step_test_settings(settings, 12.0)['min_lag_frames'] == 1

    def test_require_shift_test_inputs(self, tmp_path):
        """A missing path, a missing sidecar, a sidecar without starts and an fps mismatch each raise."""

        with pytest.raises(ValueError, match='traces_path'):
            ms._require_shift_test_inputs(None, 'old.pkl', 12.0)
        with pytest.raises(ValueError, match='traces_path'):
            ms._require_shift_test_inputs({'analysis_type': 'onset'}, 'old.pkl', 12.0)
        with pytest.raises(ValueError, match='not on disk'):
            ms._require_shift_test_inputs({'traces_path': str(tmp_path / 'absent.h5')}, 'x.pkl', 12.0)
        traces_only = tmp_path / 'traces_only.h5'
        with h5py.File(traces_only, 'w') as handle:
            handle.create_group('f').create_dataset('s', data=np.zeros(10))
        with pytest.raises(ValueError, match='predates'):
            ms._require_shift_test_inputs({'traces_path': str(traces_only)}, 'x.pkl', 12.0)
        mismatch = tmp_path / 'mismatch.h5'
        mu.write_feature_traces(mismatch, {'f': {'s': np.zeros(10)}},
                                {'s': {'usv': [], 'no_usv': [], 'n_frames': 10, 'fps': 30.0}})
        with pytest.raises(ValueError, match='fps'):
            ms._require_shift_test_inputs({'traces_path': str(mismatch)}, 'x.pkl', 12.0)
        sidecar = tmp_path / 'traces.h5'
        starts = {'s': {'usv': [2, 5], 'no_usv': [], 'n_frames': 10, 'fps': 12.0}}
        mu.write_feature_traces(sidecar, {'f': {'s': np.zeros(10)}}, starts)
        traces_path, window_starts = ms._require_shift_test_inputs({'traces_path': str(sidecar)}, 'x.pkl', 12.0)
        assert traces_path == str(sidecar)
        assert set(window_starts) == {'s'}
        assert np.array_equal(window_starts['s']['usv'], [2, 5])
        assert window_starts['s']['no_usv'].size == 0
        assert window_starts['s']['n_frames'] == 10
        assert window_starts['s']['fps'] == 12.0

    def test_require_step_test_rule(self, tmp_path):
        """A checkpoint without the rule tag, or with another rule, is refused; the right rule passes."""

        path = tmp_path / 'prefix_3.pkl'
        with pytest.raises(ValueError, match='written under step rule None'):
            ms._require_step_test_rule({'candidates_summary': {}}, path)
        with pytest.raises(ValueError, match='paired_1se'):
            ms._require_step_test_rule({'_run_metadata': {'extra_knobs': {'step_test': {'rule': 'paired_1se'}}}}, path)
        ms._require_step_test_rule({'_run_metadata': {'extra_knobs': {'step_test': {'rule': sst.STEP_TEST_RULE}}}}, path)

    def test_quarantine_moves_step_and_partial_files_only(self, tmp_path):
        """Step files and ``<prefix><k>_null_partial.pkl`` are moved; unrelated files stay."""

        prefix = 'model_selection_x_step_'
        for name in (f'{prefix}0.pkl', f'{prefix}1.pkl', f'{prefix}1_null_partial.pkl',
                     'other_run_step_0.pkl', f'{prefix}notes.txt'):
            (tmp_path / name).write_bytes(b'x')
        stale_dir = ms._quarantine_stale_step_files(tmp_path, prefix)
        assert stale_dir is not None
        assert stale_dir.is_dir()
        moved = sorted(p.name for p in stale_dir.iterdir())
        assert moved == sorted([f'{prefix}0.pkl', f'{prefix}1.pkl', f'{prefix}1_null_partial.pkl'])
        assert (tmp_path / 'other_run_step_0.pkl').is_file()
        assert (tmp_path / f'{prefix}notes.txt').is_file()
        assert ms._quarantine_stale_step_files(tmp_path, prefix) is None

    def test_assemble_fold_metrics_placeholders(self, capsys):
        """Failed folds are logged and contribute NaN (a NaN ``(2, 2)`` matrix for the confusion matrix)."""

        ok = {'ok': True, 'll': 0.5, 'confusion_matrix': np.eye(2)}
        bad = {'ok': False, 'error': 'ValueError: boom'}
        metrics = ms._assemble_fold_metrics([ok, bad, ok], 'self.speed', ('ll', 'confusion_matrix'))
        assert len(metrics['ll']) == 3
        assert np.isnan(metrics['ll'][1])
        assert metrics['confusion_matrix'][1].shape == (2, 2)
        assert np.isnan(metrics['confusion_matrix'][1]).all()
        assert np.array_equal(metrics['confusion_matrix'][2], np.eye(2))
        assert 'Error fitting self.speed (Fold 1): ValueError: boom' in capsys.readouterr().out


# ---------------------------------------------------------------------------------
# end-to-end onset selection
# ---------------------------------------------------------------------------------

def _onset_input_metadata(session_ids: list, held_out: list, feature_names: list,
                          traces_path: Path | None) -> dict:
    """
    Description
    -----------
    The ``_input_metadata`` block of the toy onset pickle, in the shape the onset
    extraction writes (schema version 3 with the sidecar fields, or without them to
    mimic an older pickle).

    Parameters
    ----------
    session_ids (list)
        All sessions.
    held_out (list)
        Reserved sessions.
    feature_names (list)
        Generic feature keys.
    traces_path (pathlib.Path | None)
        Sidecar path, or None to omit the sidecar field.

    Returns
    -------
    input_md (dict)
        The metadata block.
    """

    input_md = {
        '_schema_version': SCHEMA_VERSIONS['input'],
        'analysis_type': 'onset',
        'analysis_tag': 'bout_onset',
        'session_ids': list(session_ids),
        'held_out_session_ids': list(held_out),
        'feature_zoo_kept': list(feature_names),
        'filter_history_frames': HISTORY_FRAMES,
        'camera_sampling_rate_hz': CAMERA_FPS,
    }
    if traces_path is not None:
        input_md['traces_path'] = str(traces_path)
    return input_md


def _build_onset_cohort(tmp_path: Path, feature_names: list, with_sidecar: bool = True,
                        seed: int = 0) -> tuple[str, str, list]:
    """
    Description
    -----------
    Writes a toy onset modeling pickle whose windows are cut from stored traces at
    stored start frames (plus the HDF5 sidecar), and a univariate ranking that lets
    every feature through the screen. The first feature carries a strong class signal,
    the second a weaker one, the third none.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Scratch directory.
    feature_names (list)
        Three generic feature keys.
    with_sidecar (bool)
        False writes the pickle without ``traces_path`` (an old pickle).
    seed (int)
        Seed of the toy data.

    Returns
    -------
    input_pkl (str)
        The modeling pickle.
    ranking_pkl (str)
        The univariate ranking pickle.
    session_ids (list)
        The sessions.
    """

    rng = np.random.default_rng(seed)
    session_ids = [f'session_{i}' for i in range(N_SESSIONS)]
    signals = [2.5, 1.2, 0.0]
    spacing = HISTORY_FRAMES + 2
    starts_by_session, traces_by_feature, artifact = {}, {}, {}
    for session_id in session_ids:
        candidate_starts = np.arange(HISTORY_FRAMES, SESSION_FRAMES - spacing, spacing)
        chosen = np.sort(rng.choice(candidate_starts, size=60, replace=False))
        labels = np.zeros(60, dtype=int)
        labels[rng.choice(60, size=24, replace=False)] = 1
        starts_by_session[session_id] = {
            'usv': chosen[labels == 1], 'no_usv': chosen[labels == 0],
            'n_frames': SESSION_FRAMES, 'fps': CAMERA_FPS,
        }
        for feature, signal in zip(feature_names, signals, strict=True):
            trace = rng.standard_normal(SESSION_FRAMES)
            for start in chosen[labels == 1]:
                trace[start:start + HISTORY_FRAMES] += signal
            traces_by_feature.setdefault(feature, {})[session_id] = trace
            artifact.setdefault(feature, {})[session_id] = {
                'usv_feature_arr': mu.cut_history_windows(trace, chosen[labels == 1], HISTORY_FRAMES),
                'no_usv_feature_arr': mu.cut_history_windows(trace, chosen[labels == 0], HISTORY_FRAMES),
            }
    traces_path = tmp_path / 'modeling_input_traces.h5'
    mu.write_feature_traces(traces_path, traces_by_feature, starts_by_session)
    artifact['_input_metadata'] = _onset_input_metadata(
        session_ids, [session_ids[-1]], feature_names, traces_path if with_sidecar else None,
    )
    input_pkl = tmp_path / 'modeling_input.pkl'
    with input_pkl.open('wb') as handle:
        pickle.dump(artifact, handle)
    ranking = {feature: {'actual': {'ll': np.full(4, 0.20 + 0.01 * i)}, 'null': {'ll': np.full(4, 0.69)}}
               for i, feature in enumerate(feature_names)}
    ranking_pkl = tmp_path / 'univariate_combined.pkl'
    with ranking_pkl.open('wb') as handle:
        pickle.dump(ranking, handle)
    return str(input_pkl), str(ranking_pkl), session_ids


def _onset_settings(tmp_path: Path) -> tuple[dict, str]:
    """
    Description
    -----------
    Shrunk settings of the end-to-end test: pygam engine, three session folds, a
    0.2 level (so ``B`` is 15 / 10 / 5 over the three steps), one serial worker, a 2 s
    lag floor and the toy camera rate; serialized next to the data.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Scratch directory.

    Returns
    -------
    settings (dict)
        The settings dict.
    settings_path (str)
        Its JSON path.
    """

    session_roots = build_session_tree(base_dir=tmp_path / 'sessions', n_sessions=2, n_frames=300,
                                       camera_fps=CAMERA_FPS, filter_history=FILTER_HISTORY,
                                       n_bouts=2, usv_per_bout=2)
    list_file = write_session_list_file(session_roots, tmp_path / 'session_list.txt')
    settings = build_modeling_settings(
        session_list_file=list_file, save_directory=tmp_path / 'out', camera_sampling_rate=CAMERA_FPS,
        filter_history=FILTER_HISTORY, model_engine='pygam', split_strategy='session', split_num=3,
        test_proportion=0.34,
    )
    settings['model_params']['usv_bout_time'] = FILTER_HISTORY
    settings['model_params']['selection_p_val'] = 0.2
    settings['model_params']['selection_step_null_min_lag_seconds'] = MIN_LAG_SECONDS
    settings['model_params']['selection_step_null_checkpoint_every'] = 4
    settings['model_params']['selection_pool_workers'] = 1
    settings['model_validation']['held_out_test_proportion'] = 0.25
    settings_path = tmp_path / 'settings.json'
    settings_path.write_text(json.dumps(settings))
    return settings, str(settings_path)


def _load_steps(ms_dir: Path) -> list:
    """
    Description
    -----------
    Loads every step pickle of a selection directory in step order.

    Parameters
    ----------
    ms_dir (pathlib.Path)
        The selection output directory.

    Returns
    -------
    steps (list)
        The loaded step dicts.
    """

    paths = sorted(ms_dir.glob('model_selection_*_step_*.pkl'),
                   key=lambda p: int(p.stem.rsplit('step_', 1)[1]))
    steps = []
    for path in paths:
        with path.open('rb') as handle:
            steps.append(pickle.load(handle))
    return steps


STEP_TEST_KEYS = ('rule', 'statistic', 'statistic_family', 'winner', 'incumbent_features', 'm_candidates',
                  'alpha', 'n_refits', 'n_draws_run', 'k_exceed', 'p', 'p_adjusted', 'accepted',
                  'early_stop', 't_obs', 't_null', 'draw_indices', 'lags', 'session_order',
                  'min_lag_frames', 'seed', 'score_incumbent', 'score_winner', 'refit_seconds', 'wall_seconds')


class TestOnsetSelectionEndToEnd:
    """``vocal_onset_model_selection`` under the shift test on a toy cohort."""

    @pytest.mark.filterwarnings('ignore:Bitwise inversion:DeprecationWarning')
    def test_selection_runs_tests_every_step_and_resumes(self, tmp_path, capsys):
        """
        Every written step carries a complete ``step_test`` block under the recorded rule;
        the strong first feature is accepted with zero exceedances after all its draws; a
        rejecting step, if any, records no selected feature; finalization fields exist;
        a second call resumes and reproduces the final feature set.
        """

        feature_names = ['self.speed', 'other.speed', 'self.neck_elevation']
        input_pkl, ranking_pkl, session_ids = _build_onset_cohort(tmp_path, feature_names)
        _, settings_path = _onset_settings(tmp_path)
        ms_dir = tmp_path / 'model_selection'
        ms_dir.mkdir()

        vocal_onset_model_selection(
            univariate_results_path=ranking_pkl, input_data_path=input_pkl, settings_path=settings_path,
            output_directory=str(ms_dir), use_top_rank_as_anchor=False, p_val=0.2,
        )
        first_out = capsys.readouterr().out
        assert 'Shift-test inputs verified' in first_out

        steps = _load_steps(ms_dir)
        assert steps, 'no step pickle written'
        assert all('step_test' in step for step in steps)
        for step in steps:
            assert step['_run_metadata']['extra_knobs']['step_test']['rule'] == sst.STEP_TEST_RULE
            assert step['_run_metadata']['extra_knobs']['step_test']['min_lag_frames'] == int(MIN_LAG_SECONDS * CAMERA_FPS)
            if step['step_test'] is not None:
                assert all(key in step['step_test'] for key in STEP_TEST_KEYS)
                assert step['step_test']['rule'] == sst.STEP_TEST_RULE
                assert step['step_test']['statistic'] == 'bernoulli_log_likelihood_per_event'
                assert step['step_test']['n_refits'] == mu.shift_refit_count(step['step_test']['m_candidates'], 0.2)
                assert set(step['step_test']['session_order']) == set(session_ids[:-1])
        assert not list(ms_dir.glob('*_null_partial.pkl'))

        first = steps[0]
        assert first['step_test']['winner'] == feature_names[0]
        assert first['step_test']['m_candidates'] == 3
        assert first['step_test']['incumbent_features'] == []
        assert first['step_test']['score_incumbent'] == pytest.approx(-np.log(2))
        assert first['step_test']['accepted'] is True
        assert first['step_test']['k_exceed'] == 0
        assert first['step_test']['n_draws_run'] == first['step_test']['n_refits'] == 15
        assert first['selected_feature'] == feature_names[0]
        assert first['current_features'] == []

        for step in steps:
            test = step['step_test']
            if step['selected_feature'] is None:
                assert test is None or test['accepted'] is False
            else:
                assert test['accepted'] is True
                assert test['k_exceed'] == 0
                assert test['n_draws_run'] == test['n_refits']
                assert test['winner'] == step['selected_feature']
                assert test['incumbent_features'] == step['current_features']
        assert len(steps) <= len(feature_names)
        if len(steps) < len(feature_names):
            assert steps[-1]['selected_feature'] is None

        final = steps[-1]
        assert 'final_model_features' in final
        assert 'filter_shapes' in final
        assert 'heldout' in final
        accepted = [step['selected_feature'] for step in steps if step['selected_feature'] is not None]
        assert final['final_model_features'] == accepted
        assert accepted[0] == feature_names[0]
        if accepted:
            assert final['heldout'] is not None
            assert final['heldout']['n_held_out_sessions'] == 1
            assert len(final['filter_shapes']) >= 1

        vocal_onset_model_selection(
            univariate_results_path=ranking_pkl, input_data_path=input_pkl, settings_path=settings_path,
            output_directory=str(ms_dir), use_top_rank_as_anchor=False, p_val=0.2,
        )
        assert '[RESUME]' in capsys.readouterr().out
        resumed = _load_steps(ms_dir)
        assert len(resumed) == len(steps)
        assert resumed[-1]['final_model_features'] == final['final_model_features']

    @pytest.mark.filterwarnings('ignore:Bitwise inversion:DeprecationWarning')
    def test_forced_anchor_is_tested(self, tmp_path):
        """With ``--anchor`` the top-ranked feature is tested against every screened feature."""

        feature_names = ['self.speed', 'other.speed', 'self.neck_elevation']
        input_pkl, ranking_pkl, _ = _build_onset_cohort(tmp_path, feature_names, seed=5)
        _, settings_path = _onset_settings(tmp_path)
        ms_dir = tmp_path / 'model_selection'
        ms_dir.mkdir()
        vocal_onset_model_selection(
            univariate_results_path=ranking_pkl, input_data_path=input_pkl, settings_path=settings_path,
            output_directory=str(ms_dir), use_top_rank_as_anchor=True, p_val=0.2,
        )
        steps = _load_steps(ms_dir)
        anchor = steps[0]
        assert anchor['step_idx'] == 0
        assert anchor['step_test']['winner'] == feature_names[0]
        assert anchor['step_test']['m_candidates'] == 3
        assert anchor['step_test']['accepted'] is True
        assert anchor['selected_feature'] == feature_names[0]
        assert anchor['current_features'] == [feature_names[0]]
        assert anchor['candidates_summary'][feature_names[0]]['mean_ll'] < np.log(2)

    def test_pickle_without_sidecar_is_refused(self, tmp_path):
        """A modeling pickle extracted before the sidecar existed stops the run with an instruction."""

        feature_names = ['self.speed', 'other.speed', 'self.neck_elevation']
        input_pkl, ranking_pkl, _ = _build_onset_cohort(tmp_path, feature_names, with_sidecar=False)
        _, settings_path = _onset_settings(tmp_path)
        ms_dir = tmp_path / 'model_selection'
        ms_dir.mkdir()
        with pytest.raises(ValueError, match='Rebuild the modeling input pickle'):
            vocal_onset_model_selection(
                univariate_results_path=ranking_pkl, input_data_path=input_pkl, settings_path=settings_path,
                output_directory=str(ms_dir), use_top_rank_as_anchor=False, p_val=0.2,
            )
        assert not list(ms_dir.glob('*.pkl'))
