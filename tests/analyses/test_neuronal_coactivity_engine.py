"""
@author: bartulem
Tests for ``usv_playpen.analyses.neuronal_coactivity_engine``.

Covers the deterministic building blocks (binary-search snippet counts,
the three coactivity metrics, the joint circular shift, sliding-window
sweep) against hand-computed values, and the five significance routines
(bootstrap / circular-shuffle / chained-circular-shuffle / label-
permutation / onset-sampling). The significance tests assert that the
``seed`` parameter makes results reproducible — locking in the N2 seed
plumbing so coactivity significance is no longer dependent on global
RNG entropy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from usv_playpen.analyses import neuronal_coactivity_engine as engine


def _neural_data():
    """
    Description
    -----------
    Three neurons with irregular (seeded-random) spike trains spanning
    a 100 s session, sorted as the engine requires. Irregular rates
    keep per-window spike counts variable across trials so the
    correlation metrics stay finite (no all-NaN degenerate matrices).

    Returns
    -------
    dict[str, numpy.ndarray]
        Spike-time arrays keyed by neuron id.
    """

    rng = np.random.default_rng(0)
    return {
        "n0": np.sort(rng.uniform(0.0, 100.0, 40)),
        "n1": np.sort(rng.uniform(0.0, 100.0, 30)),
        "n2": np.sort(rng.uniform(0.0, 100.0, 50)),
    }


def test_extract_snippet_matrix_counts_by_binary_search():
    """
    Description
    -----------
    Each cell counts the spikes falling in ``[onset, onset + window]``
    via the searchsorted left/right pair. For neuron ``a`` with spikes
    ``[0.1, 0.5, 0.9, 1.5]`` and window 0.5, onset 0 captures
    ``{0.1, 0.5}`` (2) and onset 1.0 captures ``{1.5}`` (1).
    """

    neural_data = {
        "a": np.array([0.1, 0.5, 0.9, 1.5]),
        "b": np.array([0.2, 0.6]),
    }
    matrix = engine.extract_snippet_matrix(np.array([0.0, 1.0]), neural_data, 0.5)
    np.testing.assert_array_equal(matrix, [[2, 1], [1, 0]])


def test_extract_snippet_matrix_no_trials_returns_empty_columns():
    """
    Description
    -----------
    With zero onsets the function short-circuits to a
    ``(n_neurons, 0)`` zero matrix.
    """

    matrix = engine.extract_snippet_matrix(np.array([]), _neural_data(), 1.0)
    assert matrix.shape == (3, 0)


def test_compute_coactivity_metrics_too_few_neurons_is_nan():
    """
    Description
    -----------
    Pairwise correlation is undefined for a single neuron, so all three
    metrics are NaN when the matrix has fewer than two rows.
    """

    metrics = engine.compute_coactivity_metrics(np.array([[1.0, 2.0, 3.0]]))
    assert all(np.isnan(metrics[k]) for k in ("r_sc", "similarity", "pop_corr"))


def test_compute_coactivity_metrics_perfectly_correlated_rows():
    """
    Description
    -----------
    Two perfectly correlated neurons (row 2 = 2 × row 1) give a mean
    pairwise spike-count correlation of exactly 1, and the metric dict
    exposes the three expected keys.
    """

    matrix = np.array([[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0]])
    metrics = engine.compute_coactivity_metrics(matrix)
    assert set(metrics) == {"r_sc", "similarity", "pop_corr"}
    assert metrics["r_sc"] == pytest.approx(1.0)


def test_apply_circular_shift_wraps_and_resorts():
    """
    Description
    -----------
    Every spike is shifted by a constant and wrapped modulo the session
    duration, then re-sorted: ``[1, 5, 9] + 3 (mod 10) = [4, 8, 2]``
    sorts to ``[2, 4, 8]``.
    """

    shifted = engine.apply_circular_shift({"a": np.array([1.0, 5.0, 9.0])}, 3.0, 10.0)
    np.testing.assert_array_equal(shifted["a"], [2.0, 4.0, 8.0])


@pytest.mark.filterwarnings("ignore:Mean of empty slice:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
def test_compute_sliding_coactivity_shapes_and_time_bins():
    """
    Description
    -----------
    The sliding sweep returns ``time_bins`` at multiples of the step
    plus per-bin ``r_sc`` / ``similarity`` arrays, all of length
    ``n_steps``.
    """

    out = engine.compute_sliding_coactivity(
        np.array([10.0, 30.0, 50.0]), _neural_data(),
        window_s=5.0, step_s=2.0, n_steps=4,
    )
    np.testing.assert_array_equal(out["time_bins"], [0.0, 2.0, 4.0, 6.0])
    assert out["r_sc"].shape == (4,)
    assert out["similarity"].shape == (4,)


@pytest.mark.filterwarnings("ignore:Mean of empty slice:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
def test_bootstrap_distribution_is_seed_reproducible():
    """
    Description
    -----------
    Two bootstrap runs with the same ``seed`` produce identical
    per-metric distributions, while a different seed diverges — the N2
    reproducibility guarantee.
    """

    matrix = engine.extract_snippet_matrix(
        np.array([10.0, 30.0, 50.0, 70.0, 90.0]), _neural_data(), 5.0,
    )
    a = engine.bootstrap_coactivity_distribution(matrix, n_target=5, n_iterations=16, seed=42)
    b = engine.bootstrap_coactivity_distribution(matrix, n_target=5, n_iterations=16, seed=42)
    c = engine.bootstrap_coactivity_distribution(matrix, n_target=5, n_iterations=16, seed=43)
    for key in ("r_sc", "similarity", "pop_corr"):
        np.testing.assert_array_equal(a[key], b[key])
    assert not np.array_equal(a["r_sc"], c["r_sc"])


@pytest.mark.filterwarnings("ignore:Mean of empty slice:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
def test_circular_shuffle_is_seed_reproducible():
    """
    Description
    -----------
    The joint circular shuffle yields identical null distributions for
    equal seeds and length-``n_shuffles`` arrays for every metric.
    """

    onsets = np.array([10.0, 30.0, 50.0, 70.0, 90.0])
    kwargs = dict(total_duration=100.0, window_s=5.0, n_shuffles=12)
    a = engine.perform_circular_shuffle(onsets, _neural_data(), seed=7, **kwargs)
    b = engine.perform_circular_shuffle(onsets, _neural_data(), seed=7, **kwargs)
    c = engine.perform_circular_shuffle(onsets, _neural_data(), seed=8, **kwargs)
    for key in ("r_sc", "similarity", "pop_corr"):
        np.testing.assert_array_equal(a[key], b[key])
        assert a[key].shape == (12,)
    # A different seed must produce a different null (at least one metric),
    # otherwise the seed is not actually driving the shuffle.
    assert any(
        not np.array_equal(a[key], c[key])
        for key in ("r_sc", "similarity", "pop_corr")
    )


@pytest.mark.filterwarnings("ignore:Mean of empty slice:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
def test_chained_circular_shuffle_is_seed_reproducible():
    """
    Description
    -----------
    The multi-session chained shuffle concatenates per-session shuffled
    matrices and is reproducible under a fixed ``seed``.
    """

    onsets = np.array([10.0, 30.0, 50.0, 70.0, 90.0])
    nd = _neural_data()
    kwargs = dict(
        session_onsets=[onsets, onsets],
        session_neural_data=[nd, nd],
        session_durations=[100.0, 100.0],
        window_s=5.0,
        n_shuffles=10,
    )
    a = engine.perform_chained_circular_shuffle(seed=3, **kwargs)
    b = engine.perform_chained_circular_shuffle(seed=3, **kwargs)
    c = engine.perform_chained_circular_shuffle(seed=4, **kwargs)
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])
    # A different seed must diverge on at least one metric.
    assert any(not np.array_equal(a[key], c[key]) for key in a)


@pytest.mark.filterwarnings("ignore:Mean of empty slice:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:Degrees of freedom <= 0:RuntimeWarning")
def test_label_permutation_test_structure_and_reproducibility():
    """
    Description
    -----------
    The label-permutation test returns, per metric, the observed delta,
    the null array, the two p-values and a z-score; the null is
    reproducible under a fixed ``seed``.
    """

    nd = _neural_data()
    counts_a = engine.extract_snippet_matrix(np.array([10.0, 30.0, 50.0]), nd, 5.0)
    counts_b = engine.extract_snippet_matrix(np.array([20.0, 40.0, 60.0, 80.0]), nd, 5.0)
    a = engine.perform_label_permutation_test(counts_a, counts_b, n_permutations=20, seed=11)
    b = engine.perform_label_permutation_test(counts_a, counts_b, n_permutations=20, seed=11)
    c = engine.perform_label_permutation_test(counts_a, counts_b, n_permutations=20, seed=12)
    for key in ("r_sc", "similarity", "pop_corr"):
        entry = a[key]
        assert set(entry) >= {
            "observed_delta", "null", "p_a_gt_b", "p_two_tailed", "z_score",
        }
        assert entry["null"].shape == (20,)
        assert 0.0 <= entry["p_a_gt_b"] <= 1.0
        np.testing.assert_array_equal(entry["null"], b[key]["null"])
    # A different seed must produce a different null on at least one metric.
    assert any(
        not np.array_equal(a[key]["null"], c[key]["null"])
        for key in ("r_sc", "similarity", "pop_corr")
    )


def _sessions_for_sampling():
    """
    Description
    -----------
    Two sessions, each exposing a ``group`` table with a ``start``
    column — the structure :func:`sample_onsets_across_sessions`
    consumes.

    Returns
    -------
    list[dict]
        Session dicts pooling 10 candidate onsets.
    """

    return [
        {"group": pd.DataFrame({"start": [1.0, 2.0, 3.0, 4.0, 5.0]})},
        {"group": pd.DataFrame({"start": [6.0, 7.0, 8.0, 9.0, 10.0]})},
    ]


def test_sample_onsets_across_sessions_is_seed_reproducible():
    """
    Description
    -----------
    Sampling the same total count with the same ``seed`` redistributes
    the identical onsets back into the same per-session buckets, and
    each bucket is returned sorted.
    """

    sessions = _sessions_for_sampling()
    a = engine.sample_onsets_across_sessions(sessions, "group", n_total=4, seed=5)
    b = engine.sample_onsets_across_sessions(sessions, "group", n_total=4, seed=5)
    assert len(a) == 2
    for arr_a, arr_b in zip(a, b):
        np.testing.assert_array_equal(arr_a, arr_b)
        np.testing.assert_array_equal(arr_a, np.sort(arr_a))
    assert sum(arr.size for arr in a) == 4
    # A different seed must redistribute the onsets differently. The pool is
    # small, so guard against a coincidental match by requiring at least one
    # of several alternative seeds to diverge.
    a_flat = np.concatenate([np.sort(arr) for arr in a])
    assert any(
        not np.array_equal(
            a_flat,
            np.concatenate([
                np.sort(arr)
                for arr in engine.sample_onsets_across_sessions(
                    sessions, "group", n_total=4, seed=alt_seed
                )
            ]),
        )
        for alt_seed in (6, 7, 8, 9)
    )


def test_sample_onsets_across_sessions_rejects_oversized_request():
    """
    Description
    -----------
    Requesting more onsets than the pooled total raises ``ValueError``
    naming the available and target counts.
    """

    with pytest.raises(ValueError, match="less than target N"):
        engine.sample_onsets_across_sessions(_sessions_for_sampling(), "group", n_total=99)
