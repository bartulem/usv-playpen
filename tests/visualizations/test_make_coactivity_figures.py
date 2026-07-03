"""
@author: bartulem
Smoke tests for make_coactivity_figures: every plotter returns a matplotlib
Figure with the expected axis count (or None for the empty case), the summary
builders emit their labelled sections, and one integration test feeds real
neuronal_coactivity_engine output through the figure / summary layer to catch
any drift in the result-object contract between the two modules.
"""
from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import polars as pls
import pytest

from usv_playpen.analyses import neuronal_coactivity_engine as engine
from usv_playpen.visualizations import make_coactivity_figures as mcf

_METRICS = ("r_sc", "similarity", "pop_corr")
_FEATURES = ("rms", "mean_freq_hz", "peak_freq_hz", "freq_bandwidth_hz")
_FEATURE_LABELS = {
    "rms": "RMS amplitude (a.u.)",
    "mean_freq_hz": "mean frequency (Hz)",
    "peak_freq_hz": "peak frequency (Hz)",
    "freq_bandwidth_hz": "frequency bandwidth (Hz)",
}


def _acoustics(rng, mean):
    """Fresh per-feature array dict, one independent random draw per feature."""
    out = {}
    for feature in _FEATURES:
        out[feature] = rng.normal(mean, 1.0, 100)
    return out


def _results():
    """
    Description
    -----------
    A synthetic :func:`run_group_comparison`-shaped result dict with random
    bootstrap / null distributions, a permutation block and two per-session rows.

    Returns
    -------
    dict
        The result object the null-distribution plot and group-comparison summary
        consume.
    """

    rng = np.random.default_rng(0)
    boot_a, boot_b, null_a, null_b, perm = {}, {}, {}, {}, {}
    for metric in _METRICS:
        boot_a[metric] = rng.normal(0.2, 0.05, 50)
        boot_b[metric] = rng.normal(0.1, 0.05, 50)
        null_a[metric] = rng.normal(0.0, 0.05, 100)
        null_b[metric] = rng.normal(0.0, 0.05, 100)
        perm[metric] = {
            "observed_delta": 0.1, "null_mean": 0.0, "p_a_gt_b": 0.02,
            "p_two_tailed": 0.04, "z_score": 1.9,
        }
    per_session = []
    for i in range(2):
        per_session.append({
            "session_id": f"sess{i}", "n_a": 12, "n_b": 9,
            "metrics_a": {metric: 0.3 for metric in _METRICS},
            "metrics_b": {metric: 0.1 for metric in _METRICS},
            "deltas": {metric: 0.2 for metric in _METRICS},
        })
    return {
        "boot_a": boot_a, "boot_b": boot_b,
        "chained_null_a": null_a, "chained_null_b": null_b,
        "perm": perm, "per_session": per_session, "bootstrap_n": 50,
    }


def test_plot_null_distributions_returns_3x2_figure():
    """The null-distribution plot is a 3-metric x 2-group grid (6 axes)."""
    fig = mcf.plot_null_distributions(_results(), category_column="qlvm_supercategory", group_a_ids=[1], group_b_ids=[7])
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 6
    plt.close(fig)


def test_plot_per_session_pop_corr_figure_and_empty():
    """One panel per session with a null block; ``None`` when no row carries a null."""
    rng = np.random.default_rng(1)
    rows = []
    for i in range(3):
        rows.append({
            "session_id": f"sess{i}", "n_a": 10, "n_b": 8,
            "metrics_a": {"pop_corr": 0.4}, "metrics_b": {"pop_corr": 0.2},
            "null": {"pop_corr": rng.normal(0.0, 0.05, 80)},
        })
    fig = mcf.plot_per_session_pop_corr(rows, chosen_animal="178621_2", category_column="qlvm_supercategory")
    assert isinstance(fig, plt.Figure)
    assert len([ax for ax in fig.axes if ax.get_visible()]) >= 3
    plt.close(fig)
    assert mcf.plot_per_session_pop_corr([{"session_id": "x", "metrics_a": {}, "metrics_b": {}}], chosen_animal="a", category_column="c") is None


def test_plot_acoustic_confound_returns_figure():
    """One subplot per acoustic feature."""
    rng = np.random.default_rng(2)
    fig = mcf.plot_acoustic_confound(
        _acoustics(rng, 0.0), _acoustics(rng, 0.5),
        features=_FEATURES, feature_labels=_FEATURE_LABELS, chosen_animal="178621_2",
    )
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == len(_FEATURES)
    plt.close(fig)


def test_plot_cross_animal_slope_returns_figure():
    """The slope plot renders one figure for a per-animal result mapping."""
    cross = {
        "m0": {"pop_a": 0.4, "pop_b": 0.2, "p_two": 0.01},
        "m1": {"pop_a": 0.1, "pop_b": 0.3, "p_two": 0.20},
    }
    fig = mcf.plot_cross_animal_slope(cross, category_column="qlvm_supercategory", group_a_ids=[1], group_b_ids=[7])
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_amplitude_stratified_figure_and_all_nan():
    """Renders with valid bins, and still returns a figure when every bin is empty."""
    rows = [
        {"lo": 0.1, "hi": 0.2, "n_a": 20, "n_b": 18, "pop_a": 0.3, "pop_b": 0.2, "p_two": 0.04},
        {"lo": 0.2, "hi": 0.4, "n_a": 22, "n_b": 20, "pop_a": 0.25, "pop_b": 0.22, "p_two": 0.30},
    ]
    fig = mcf.plot_amplitude_stratified(rows, 0.28, 0.21, chosen_animal="178621_2")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
    empty = [{"lo": 0.1, "hi": 0.2, "n_a": 2, "n_b": 1, "pop_a": np.nan, "pop_b": np.nan, "p_two": np.nan}]
    fig2 = mcf.plot_amplitude_stratified(empty, np.nan, np.nan, chosen_animal="178621_2")
    assert isinstance(fig2, plt.Figure)
    plt.close(fig2)


def test_summaries_contain_labelled_sections():
    """The three summary builders emit their key labelled rows / headers."""
    group = mcf.summarize_group_comparison(_results(), label_a="complex", label_b="simple")
    assert "DIRECT PERMUTATION" in group
    assert "vs CHAINED NULL" in group
    assert "complex" in group

    rng = np.random.default_rng(3)
    acoustic = mcf.summarize_acoustic_confound(_acoustics(rng, 0.0), _acoustics(rng, 0.5), features=_FEATURES, chosen_animal="178621_2")
    assert "Acoustic confound check" in acoustic
    assert "Cohen's d" in acoustic
    assert "rms" in acoustic

    rows = [{"lo": 0.1, "hi": 0.2, "n_a": 20, "n_b": 18, "pop_a": 0.3, "pop_b": 0.2, "p_two": 0.04}]
    strat = mcf.summarize_amplitude_stratified(rows, 0.28, 0.21, chosen_animal="178621_2", n_bins=5)
    assert "Amplitude-stratified pop_corr" in strat
    assert "unstratified" in strat


def _sessions():
    """
    Description
    -----------
    Two sessions sharing three-neuron spike data with group-A / group-B onset
    tables — enough for a real engine run feeding the figure layer.

    Returns
    -------
    list[dict]
    """

    rng = np.random.default_rng(0)
    neural = {}
    for i in range(3):
        neural[f"n{i}"] = np.sort(rng.uniform(0.0, 100.0, 40))
    onsets_a = np.array([5.0, 15.0, 25.0, 35.0, 45.0])
    onsets_b = np.array([10.0, 20.0, 30.0, 40.0])
    return [
        {
            "session_id": "s0", "neural_data": neural, "total_duration": 100.0,
            "group_a_df": pls.DataFrame({"start": onsets_a}), "group_b_df": pls.DataFrame({"start": onsets_b}),
        },
        {
            "session_id": "s1", "neural_data": neural, "total_duration": 100.0,
            "group_a_df": pls.DataFrame({"start": onsets_a + 1.0}), "group_b_df": pls.DataFrame({"start": onsets_b + 1.0}),
        },
    ]


@pytest.mark.filterwarnings("ignore:Mean of empty slice:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:Degrees of freedom <= 0:RuntimeWarning")
def test_engine_results_feed_figures():
    """
    Description
    -----------
    Real engine output flows through the figure and summary layer without a
    result-object contract mismatch: ``run_group_comparison`` -> null-distribution
    plot + group summary, and ``per_session_group_metrics(n_shuffles=...)`` ->
    per-session plot.
    """

    sessions = _sessions()
    results = engine.run_group_comparison(
        sessions, window_s=1.0, bootstrap_n=6, n_boot=16, n_shuffles=8, n_permutations=16, seed=0,
    )
    fig = mcf.plot_null_distributions(results, category_column="qlvm_supercategory", group_a_ids=[1], group_b_ids=[7])
    assert len(fig.axes) == 6
    plt.close(fig)
    assert "DIRECT PERMUTATION" in mcf.summarize_group_comparison(results)

    rows = engine.per_session_group_metrics(sessions, window_s=1.0, n_shuffles=8, seed=0)
    fig2 = mcf.plot_per_session_pop_corr(rows, chosen_animal="s", category_column="qlvm_supercategory")
    assert fig2 is not None
    plt.close(fig2)
