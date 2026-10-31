"""
@author: bartulem

Per-step significance test of forward feature selection, and the process pool the
behavioral selectors fit through.

The test (Nevjen & Dunn 2024, eLife Reviewed Preprint 92517, their cyclic-shift
permutation with Bonferroni correction): at every step the cross-validated winner W
among the m remaining candidates is refit together with the incumbent model M on the
development draw, and the statistic is the in-sample gain T = L(M + W) - L(M) of the
pipeline's log-likelihood per event. The null refits M + W after circularly shifting
W's feature trace within every development session by a random lag of at least
``min_lag_frames`` and re-cutting its history windows at the stored start frames, so
the feature keeps its distribution and autocorrelation but loses its alignment with
the events. With ``B = floor(m / alpha)`` draws the Bonferroni-corrected p-value
``m * (1 + k) / (B + 1)`` can fall below ``alpha`` only when no draw reaches the
observed gain, so the step is decided by the first exceedance: it ends the step as a
rejection, and an accepted step has run all ``B`` draws.

The draws are independent and run through ``SelectionPool``, a fork-based process
pool whose workers read the (large, read-only) data from a module-level context
set before the pool is created, so nothing but small results is pickled. The same
pool runs the selectors' candidate x fold fits.
"""

from __future__ import annotations

import multiprocessing
import pickle
import sys
import time
from collections.abc import Callable, Iterable
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from ..os_utils import atomic_output_path
from .modeling_utils import (
    bonferroni_permutation_p,
    circular_shift_trace,
    cut_history_windows,
    draw_session_shift_lags,
    shift_refit_count,
)

#: Name of the step rule recorded in every step pickle's ``step_test`` block and in
#: ``_run_metadata['step_test']``; a resume refuses a checkpoint written under another.
STEP_TEST_RULE = 'shift_refit_bonferroni'

#: Name of the statistic family recorded with the rule (the per-pipeline quantity is
#: named in ``step_test['statistic']``).
STEP_TEST_STATISTIC_FAMILY = 'in_sample_log_likelihood_per_event'

#: Read-only data and callables the pool workers use. Filled by ``set_pool_context``
#: in the parent BEFORE ``SelectionPool`` forks, so every worker inherits it.
_POOL_CONTEXT: dict = {}


def set_pool_context(**context) -> None:
    """
    Description
    -----------
    Replaces the module-level pool context with ``context``. Call it in the parent
    process before creating a ``SelectionPool``: the pool forks its workers at
    creation, and a forked worker sees exactly the context the parent held at that
    moment (copy-on-write, nothing pickled). Entries are usually large arrays and
    the callables the workers run (``'step_refit'``, ``'candidate_fold'``).

    Parameters
    ----------
    **context
        Named entries to store.

    Returns
    -------
    None
    """

    _POOL_CONTEXT.clear()
    _POOL_CONTEXT.update(context)


def pool_context(key: str):
    """
    Description
    -----------
    Returns one entry of the pool context (in a worker: the entry inherited at fork).

    Parameters
    ----------
    key (str)
        Entry name.

    Returns
    -------
    value
        The stored entry.
    """

    return _POOL_CONTEXT[key]


def _worker_initializer() -> None:
    """
    Description
    -----------
    Runs once in every forked worker: caps the BLAS / OpenMP thread pools at one
    thread, so ``n`` workers use ``n`` cores instead of each spawning every core's
    worth of threads (pygam's dense QR gains nothing from threads: 110 s at 4 threads
    against 142 s at 1, measured).

    Returns
    -------
    None
    """

    threadpool_limits(limits=1)


def _run_pool_task(kind: str, args: tuple):
    """
    Description
    -----------
    Executes one task in a worker: looks the callable ``kind`` up in the inherited
    pool context and applies it to ``args``. Module-level so the pool can pickle the
    reference to it; the data stays in the context.

    Parameters
    ----------
    kind (str)
        Context key of the callable.
    args (tuple)
        Positional arguments.

    Returns
    -------
    result
        Whatever the callable returns (must be picklable).
    """

    return _POOL_CONTEXT[kind](*args)


def jax_backend_initialized() -> bool:
    """
    Description
    -----------
    Whether JAX has initialized a backend in this process. JAX then registers an
    at-fork handler that warns that a fork of its (multithreaded) process will likely
    deadlock; the pool reads that flag instead of importing JAX, so a process that never
    touched JAX pays nothing.

    Returns
    -------
    initialized (bool)
        True once ``jax._src.xla_bridge`` has installed its at-fork handler.
    """

    bridge = sys.modules.get('jax._src.xla_bridge')
    return bool(bridge is not None and getattr(bridge, '_at_fork_handler_installed', False))


class SelectionPool:
    """
    Fork-based process pool for the selection fits.

    With ``n_workers == 1`` every task runs in the calling process, in order, with no
    pool at all (the serial path the tests and the lab PC use); otherwise tasks run in
    a ``ProcessPoolExecutor`` forked from the current process (unless JAX has already
    initialized a backend in it, after which a fork can deadlock: the pool then runs
    serially and says so), each worker capped at
    one BLAS thread. Tasks are ``(kind, args)`` pairs resolved against the pool
    context set by ``set_pool_context`` before the pool was created.

    Parameters
    ----------
    n_workers (int)
        Number of worker processes (``>= 1``).
    """

    def __init__(self, n_workers: int) -> None:
        self.n_workers = max(1, int(n_workers))
        if self.n_workers > 1 and jax_backend_initialized():
            print("  [pool] JAX has already initialized a backend in this process; forking after that "
                  "can deadlock, so the fits run serially in this process instead.")
            self.n_workers = 1
        self._executor = None
        if self.n_workers > 1:
            self._executor = ProcessPoolExecutor(
                max_workers=self.n_workers,
                mp_context=multiprocessing.get_context('fork'),
                initializer=_worker_initializer,
            )

    def map(self, kind: str, arg_tuples: Iterable[tuple]) -> list:
        """
        Description
        -----------
        Runs every task and returns the results in submission order.

        Parameters
        ----------
        kind (str)
            Context key of the callable.
        arg_tuples (Iterable[tuple])
            One argument tuple per task.

        Returns
        -------
        results (list)
            One result per task, in order.
        """

        arg_tuples = list(arg_tuples)
        if self._executor is None:
            return [_POOL_CONTEXT[kind](*args) for args in arg_tuples]
        futures = [self._executor.submit(_run_pool_task, kind, args) for args in arg_tuples]
        return [future.result() for future in futures]

    def run_until(self,
                  kind: str,
                  arg_tuples: Iterable[tuple],
                  on_result: Callable[[object], bool]) -> None:
        """
        Description
        -----------
        Runs tasks and hands every result to ``on_result`` as it completes; stops
        submitting new tasks once ``on_result`` returns True (tasks already running
        finish and their results are also delivered, cancelled ones are not). Keeps
        at most ``2 x n_workers`` tasks in flight.

        Parameters
        ----------
        kind (str)
            Context key of the callable.
        arg_tuples (Iterable[tuple])
            Argument tuples, consumed lazily.
        on_result (Callable[[object], bool])
            Receives each result; returns True to stop.

        Returns
        -------
        None
        """

        pending_args = iter(arg_tuples)
        if self._executor is None:
            for args in pending_args:
                if on_result(_POOL_CONTEXT[kind](*args)):
                    return
            return
        in_flight = set()
        stop = False
        max_in_flight = 2 * self.n_workers
        while True:
            while not stop and len(in_flight) < max_in_flight:
                try:
                    args = next(pending_args)
                except StopIteration:
                    break
                in_flight.add(self._executor.submit(_run_pool_task, kind, args))
            if not in_flight:
                return
            done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in done:
                if on_result(future.result()):
                    stop = True
            if stop:
                for future in in_flight:
                    future.cancel()
                still_running = [future for future in in_flight if not future.cancelled()]
                for future in still_running:
                    on_result(future.result())
                return

    def close(self) -> None:
        """
        Description
        -----------
        Shuts the executor down (no-op on the serial path).

        Returns
        -------
        None
        """

        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None


def shifted_windows_for_lags(winner_traces: dict,
                             window_starts: dict,
                             slots: dict,
                             lags: dict,
                             history_frames: int) -> dict:
    """
    Description
    -----------
    Re-cuts the winner feature's history windows of every session from its
    circularly shifted trace: the replacement feature dict a null refit uses in place
    of the stored windows.

    Parameters
    ----------
    winner_traces (dict)
        ``{session_id: (n_frames,) trace}`` of the winner feature.
    window_starts (dict)
        ``{session_id: {slot_name: (n,) start frames}}``.
    slots (dict)
        ``{slot_name: array_key}``: which pickle key each start list fills
        (``{'usv': 'usv_feature_arr', 'no_usv': 'no_usv_feature_arr'}`` for the onset
        pipeline, ``{'X': 'X'}`` for the per-session single-array pipelines).
    lags (dict)
        ``{session_id: lag_frames}``.
    history_frames (int)
        Window length in frames.

    Returns
    -------
    override (dict)
        ``{session_id: {array_key: (n, history_frames) windows}}``.
    """

    override = {}
    for session_id, column in winner_traces.items():
        shifted = circular_shift_trace(column, lags[session_id])
        override[session_id] = {
            array_key: cut_history_windows(shifted, window_starts[session_id][slot_name], history_frames)
            for slot_name, array_key in slots.items()
        }
    return override


def _shift_draw(draw_index: int) -> tuple:
    """
    Description
    -----------
    One null draw, run in a worker: draws the per-session lags from the seed
    ``[random_seed, step_index, draw_index]`` (so a resumed run redraws the same lags),
    re-cuts the winner's windows from the shifted traces, refits through the
    selector's ``'step_refit'`` callable and returns the gain over the incumbent.

    Parameters
    ----------
    draw_index (int)
        Index of the draw (``0 .. B-1``).

    Returns
    -------
    draw_index (int)
        Echoed.
    lags (np.ndarray)
        ``(n_sessions,)`` lags in the sorted session order.
    gain (float)
        ``L(M + W_shifted) - L(M)``.
    seconds (float)
        Wall time of the refit.
    """

    spec = _POOL_CONTEXT['step_shift_spec']
    rng = np.random.default_rng([spec['random_seed'], spec['step_index'], int(draw_index)])
    lags = draw_session_shift_lags(rng, spec['session_lengths'], spec['min_lag_frames'])
    override = shifted_windows_for_lags(
        spec['winner_traces'], spec['window_starts'], spec['slots'], lags, spec['history_frames'],
    )
    started = time.perf_counter()
    score = _POOL_CONTEXT['step_refit'](override)
    seconds = float(time.perf_counter() - started)
    lag_vector = np.array([lags[session_id] for session_id in spec['session_order']], dtype=int)
    return int(draw_index), lag_vector, float(score) - float(spec['incumbent_score']), seconds


def _partial_fingerprint(spec: dict) -> dict:
    """
    Description
    -----------
    The identity of a step's null run, stored with its partial checkpoint and compared
    on resume: a partial file from another step, winner, candidate count, level, lag
    floor or seed is ignored.

    Parameters
    ----------
    spec (dict)
        The step-shift specification.

    Returns
    -------
    fingerprint (dict)
        Plain scalars and lists.
    """

    return {
        'rule': STEP_TEST_RULE,
        'step_index': int(spec['step_index']),
        'incumbent_features': list(spec['incumbent_features']),
        'winner': spec['winner'],
        'm_candidates': int(spec['m_candidates']),
        'n_refits': int(spec['n_refits']),
        'alpha': float(spec['alpha']),
        'min_lag_frames': int(spec['min_lag_frames']),
        'random_seed': int(spec['random_seed']),
        'session_order': list(spec['session_order']),
        'incumbent_score': float(spec['incumbent_score']),
        'observed_score': float(spec['observed_score']),
    }


def _load_partial(partial_path: Path, fingerprint: dict) -> dict:
    """
    Description
    -----------
    Loads the finished draws of an interrupted null run when its fingerprint matches.

    Parameters
    ----------
    partial_path (Path)
        The partial checkpoint file.
    fingerprint (dict)
        Expected identity.

    Returns
    -------
    draws (dict)
        ``{draw_index: (lags, gain, seconds)}``; empty when there is no usable file.
    """

    if not partial_path.is_file():
        return {}
    try:
        with partial_path.open('rb') as handle:
            payload = pickle.load(handle)
    except (OSError, pickle.UnpicklingError, EOFError) as exc:
        print(f"    [step test] partial null file {partial_path.name} unreadable ({exc}); starting the draws over.")
        return {}
    if payload['fingerprint'] != fingerprint:
        print(f"    [step test] partial null file {partial_path.name} belongs to another step/winner; ignored.")
        return {}
    draws = payload['draws']
    print(f"    [step test] resumed {len(draws)} finished null draw(s) from {partial_path.name}.")
    return draws


def _save_partial(partial_path: Path, fingerprint: dict, draws: dict) -> None:
    """
    Description
    -----------
    Writes the finished draws of a running null atomically.

    Parameters
    ----------
    partial_path (Path)
        Destination.
    fingerprint (dict)
        The run's identity.
    draws (dict)
        ``{draw_index: (lags, gain, seconds)}``.

    Returns
    -------
    None
    """

    with atomic_output_path(partial_path) as tmp_path, Path(tmp_path).open('wb') as handle:
        pickle.dump({'fingerprint': fingerprint, 'draws': draws}, handle)


def run_step_shift_test(*,
                        n_workers: int,
                        refit_and_score: Callable[[dict | None], float],
                        incumbent_features: list,
                        incumbent_score: float,
                        winner: str,
                        observed_score: float,
                        statistic_name: str,
                        winner_traces: dict,
                        window_starts: dict,
                        slots: dict,
                        history_frames: int,
                        m_candidates: int,
                        alpha: float,
                        min_lag_frames: int,
                        random_seed: int,
                        step_index: int,
                        checkpoint_every: int,
                        partial_path: Path,
                        label: str) -> dict:
    """
    Description
    -----------
    Runs the per-step shift test for one winner and returns its record.

    ``refit_and_score(override)`` is the selector's development refit of M + W: with
    ``override=None`` it uses the stored windows of W, otherwise the per-session
    windows in ``override`` (``{session: {array_key: windows}}``) replace them. It
    returns the pipeline's in-sample log-likelihood-type score. The caller has
    already computed ``incumbent_score = L(M)`` and ``observed_score = L(M + W)`` with
    it (through the same pool context), so the observed gain is
    ``observed_score - incumbent_score``.

    Draw ``b`` (``0 .. B-1``, ``B = floor(m / alpha)``) seeds
    ``[random_seed, step_index, b]``, shifts every development session's trace of W
    by its own lag in ``[min_lag, T_s - min_lag]``, re-cuts W's windows at the
    stored starts, refits and records the gain. Draws run through ``pool``; the first
    draw whose gain reaches the observed one stops the submission of new draws
    (rejection), accepted steps run all ``B``. Finished draws are checkpointed every
    ``checkpoint_every`` draws to ``partial_path`` and reused by a resumed run; the
    file is removed once the step is decided.

    Parameters
    ----------
    n_workers (int)
        Worker processes for the draws. The pool is created here, after the step's
        traces, starts and the refit callable are stored in the pool context, so the
        forked workers inherit them; it is closed before returning.
    refit_and_score (Callable)
        See above.
    incumbent_features (list)
        Features of M (for the record and the partial fingerprint).
    incumbent_score (float)
        ``L(M)``.
    winner (str)
        The candidate under test.
    observed_score (float)
        ``L(M + W)``.
    statistic_name (str)
        Name of the per-event score (``'bernoulli_log_likelihood'``, ...).
    winner_traces (dict)
        ``{session_id: trace}`` of W over the development sessions.
    window_starts (dict)
        ``{session_id: {slot_name: starts}}``.
    slots (dict)
        ``{slot_name: array_key}``.
    history_frames (int)
        Window length.
    m_candidates (int)
        Candidates at this step (the Bonferroni factor).
    alpha (float)
        Level.
    min_lag_frames (int)
        Lag floor in frames.
    random_seed (int)
        The run's seed.
    step_index (int)
        The step.
    checkpoint_every (int)
        Draws between partial writes.
    partial_path (Path)
        Partial checkpoint file.
    label (str)
        Log prefix.

    Returns
    -------
    record (dict)
        The ``step_test`` block: ``rule``, ``statistic``, ``winner``, ``m_candidates``,
        ``alpha``, ``n_refits`` (B), ``n_draws_run``, ``k_exceed``, ``p``,
        ``p_adjusted``, ``accepted``, ``early_stop``, ``t_obs``, ``t_null``
        (``(n_draws_run,)`` in draw order), ``draw_indices``, ``lags``
        (``(n_draws_run, n_sessions)``), ``session_order``, ``min_lag_frames``,
        ``seed``, ``score_incumbent``, ``score_winner``, ``incumbent_features``,
        ``refit_seconds`` (``(n_draws_run,)``), ``wall_seconds``.
    """

    started = time.perf_counter()
    n_refits = shift_refit_count(m_candidates, alpha)
    session_order = sorted(winner_traces.keys())
    spec = {
        'random_seed': int(random_seed),
        'step_index': int(step_index),
        'session_lengths': {session_id: int(np.asarray(winner_traces[session_id]).shape[0]) for session_id in session_order},
        'session_order': session_order,
        'min_lag_frames': int(min_lag_frames),
        'winner_traces': winner_traces,
        'window_starts': window_starts,
        'slots': slots,
        'history_frames': int(history_frames),
        'incumbent_score': float(incumbent_score),
        'observed_score': float(observed_score),
        'incumbent_features': list(incumbent_features),
        'winner': winner,
        'm_candidates': int(m_candidates),
        'n_refits': int(n_refits),
        'alpha': float(alpha),
    }
    _POOL_CONTEXT['step_shift_spec'] = spec
    _POOL_CONTEXT['step_refit'] = refit_and_score
    t_obs = float(observed_score) - float(incumbent_score)
    fingerprint = _partial_fingerprint(spec)
    draws = _load_partial(partial_path, fingerprint)

    print(f"    [step test] {label}: winner {winner!r} vs incumbent {list(incumbent_features)}; "
          f"T_obs = {t_obs:+.6f}; m = {m_candidates}, alpha = {alpha}, B = {n_refits}; "
          f"pool of {max(1, int(n_workers))} worker(s).")

    state = {'k': sum(1 for (_, gain, _) in draws.values() if gain >= t_obs), 'since_save': 0}

    def _on_result(result) -> bool:
        """
        Description
        -----------
        Records one finished draw, checkpoints when due and reports whether the step is
        already decided (first exceedance).

        Parameters
        ----------
        result (tuple)
            ``(draw_index, lags, gain, seconds)`` from ``_shift_draw``.

        Returns
        -------
        stop (bool)
            True once a draw reached the observed gain.
        """

        draw_index, lag_vector, gain, seconds = result
        draws[draw_index] = (lag_vector, gain, seconds)
        state['since_save'] += 1
        if gain >= t_obs:
            state['k'] += 1
        if state['since_save'] >= int(checkpoint_every):
            _save_partial(partial_path, fingerprint, draws)
            state['since_save'] = 0
            print(f"    [step test] {label}: {len(draws)}/{n_refits} draws done, "
                  f"{state['k']} at/above T_obs, last refit {seconds:.0f} s.")
        return state['k'] >= 1

    if state['k'] == 0:
        remaining = [(b,) for b in range(n_refits) if b not in draws]
        _POOL_CONTEXT['_shift_draw'] = _shift_draw
        pool = SelectionPool(n_workers)
        try:
            pool.run_until('_shift_draw', remaining, _on_result)
        finally:
            pool.close()

    order = sorted(draws)
    t_null = np.array([draws[b][1] for b in order], dtype=float)
    lags = np.array([draws[b][0] for b in order], dtype=int).reshape(len(order), len(session_order))
    seconds = np.array([draws[b][2] for b in order], dtype=float)
    n_draws_run = len(order)
    k_exceed = int(np.sum(t_null >= t_obs))
    p_value, p_adjusted = bonferroni_permutation_p(k_exceed, n_draws_run, m_candidates)
    accepted = bool(k_exceed == 0 and n_draws_run == n_refits)
    early_stop = bool(k_exceed >= 1 and n_draws_run < n_refits)
    if partial_path.is_file():
        partial_path.unlink()
    _POOL_CONTEXT.pop('step_shift_spec', None)
    _POOL_CONTEXT.pop('step_refit', None)
    _POOL_CONTEXT.pop('_shift_draw', None)
    decision = 'ACCEPT' if accepted else 'REJECT'
    print(f"    [step test] {label}: {decision} -- {k_exceed} of {n_draws_run} draws at/above T_obs "
          f"(max null gain {np.max(t_null) if n_draws_run else float('nan'):+.6f}); "
          f"p = {p_value:.5f}, Bonferroni x{m_candidates} = {p_adjusted:.5f} vs alpha {alpha}.")
    return {
        'rule': STEP_TEST_RULE,
        'statistic': statistic_name,
        'statistic_family': STEP_TEST_STATISTIC_FAMILY,
        'winner': winner,
        'incumbent_features': list(incumbent_features),
        'm_candidates': int(m_candidates),
        'alpha': float(alpha),
        'n_refits': int(n_refits),
        'n_draws_run': int(n_draws_run),
        'k_exceed': k_exceed,
        'p': p_value,
        'p_adjusted': p_adjusted,
        'accepted': accepted,
        'early_stop': early_stop,
        't_obs': t_obs,
        't_null': t_null,
        'draw_indices': np.array(order, dtype=int),
        'lags': lags,
        'session_order': list(session_order),
        'min_lag_frames': int(min_lag_frames),
        'seed': [int(random_seed), int(step_index)],
        'score_incumbent': float(incumbent_score),
        'score_winner': float(observed_score),
        'refit_seconds': seconds,
        'wall_seconds': float(time.perf_counter() - started),
    }
