"""
@author: bartulem
Unit tests for the pure data-builder / transform helpers in
``usv_playpen.visualizations.modeling_plots`` (the modeling-figure module
that lives under ``visualizations/``, hence its tests mirror it here
rather than under ``tests/modeling/``).

Only the non-drawing helpers are covered — the colour-pastelisation,
the predictor-feature group classifier, the centred rolling mean, the
consecutive-run locator, and the seeded jitter sampler. The 7k lines of
matplotlib figure code are out of scope. The matplotlib backend is forced
to ``Agg`` before import so loading the module never tries to open a
window.
"""

from __future__ import annotations

import pickle
import warnings

import numpy as np
import pytest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# modeling_plots pulls in the modeling stack (optax-backed), whose import
# emits a one-time JAX DeprecationWarning that `filterwarnings = ["error"]`
# would otherwise promote to a collection error.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from usv_playpen.visualizations.modeling_plots import (
        _FIGURE_DPI,
        _FIGURE_SEED,
        _GLOBAL_CMAP,
        _VIZ_SETTINGS,
        DeepResultsVisualizer,
        _classify_predictor_feature,
        _last_bin_of_consecutive_run,
        _rolling_mean_1d,
        plot_collinearity_audit,
        plot_feature_ranking,
        plot_manifold_multivariate_filters,
        plot_manifold_selection_trajectory,
        plot_model_selection_results,
        plot_multinomial_multivariate_filters,
        plot_multinomial_selection_diagnosis,
        plot_multinomial_selection_trajectory,
        plot_raw_feature_difference,
        plot_significant_filters,
        plot_significant_filters_grid,
        plot_timescale_audit,
        plot_timescale_audit_per_feature,
        plot_univariate_multinomial_filters_grid,
        plot_univariate_multinomial_performance,
    )


# figure resolution / seed / cmap derive from the shared `figures` block


def test_figure_dpi_seed_cmap_derive_from_settings():
    """The module-level figure resolution, RNG seed, and colormap are read from
    the shared ``figures`` block of ``visualizations_settings.json`` rather than
    hard-coded, so every ``plt.subplots(...)`` / ``savefig(...)`` / estimator call
    honours the single configured source."""

    assert _FIGURE_DPI == _VIZ_SETTINGS["figures"]["dpi"]
    assert _FIGURE_SEED == _VIZ_SETTINGS["figures"]["seed"]
    assert _GLOBAL_CMAP == _VIZ_SETTINGS["figures"]["cmap"]


# _classify_predictor_feature


class TestClassifyPredictorFeature:

    @pytest.mark.parametrize("fname,group", [
        ('self.speed', 0),
        ('self.neck_elevation_1st_der', 0),
        ('orofacial-sei', 1),
        ('anogenital-sei_2nd_der', 1),   # derivative tail stripped -> SEI group
        ('other.speed', 2),
        ('nose-nose', 3),
        ('allo_yaw-nose', 3),
        ('pooled_usv_rate', 3),          # unprefixed catch-all -> social/dyadic
    ])
    def test_group_assignment(self, fname, group):
        """Each feature name maps to its documented presentation group."""

        assert _classify_predictor_feature(fname) == group


# _rolling_mean_1d


class TestRollingMean1d:

    def test_window_le_one_returns_input(self):
        """A window of 1 (or less) is a no-op smoother."""

        arr = np.array([1.0, 5.0, 2.0])
        np.testing.assert_allclose(_rolling_mean_1d(arr, 1), arr)

    def test_constant_array_unchanged(self):
        """Smoothing a constant array returns the same constant."""

        arr = np.full(20, 3.0)
        out = _rolling_mean_1d(arr, 5)
        np.testing.assert_allclose(out, np.full(20, 3.0), atol=1e-6)

    def test_reduces_variance_of_noisy_signal(self):
        """A centred rolling mean lowers the variance of a noisy signal
        while preserving its length."""

        rng = np.random.default_rng(0)
        arr = rng.standard_normal(200)
        out = _rolling_mean_1d(arr, 9)
        assert out.shape == arr.shape
        assert np.var(out) < np.var(arr)

    def test_even_window_bumped_to_odd(self):
        """An even window is bumped to the next odd size (centred); the
        output is finite and length-preserving."""

        arr = np.arange(11, dtype=float)
        out = _rolling_mean_1d(arr, 4)
        assert out.shape == arr.shape
        assert np.all(np.isfinite(out))

    def test_window_ge_size_returns_mean_fill(self):
        """A window at least as large as the array collapses to the global
        mean."""

        arr = np.array([0.0, 2.0, 4.0, 6.0])
        out = _rolling_mean_1d(arr, 10)
        np.testing.assert_allclose(out, np.full(4, 3.0), atol=1e-6)


# _last_bin_of_consecutive_run


class TestLastBinOfConsecutiveRun:

    def test_finds_latest_run_end(self):
        """The latest index ending a run of ``run_length`` True bins is
        returned."""

        mask = np.array([1, 1, 1, 0, 1, 1, 0], dtype=bool)
        # Runs of length 2 end at indices 1, 2, and 5; the latest is 5.
        assert _last_bin_of_consecutive_run(mask, run_length=2) == 5

    def test_no_qualifying_run_returns_none(self):
        """When no run of the required length exists, ``None`` is
        returned."""

        mask = np.array([1, 0, 1, 0, 1], dtype=bool)
        assert _last_bin_of_consecutive_run(mask, run_length=2) is None

    def test_non_positive_run_length_returns_none(self):
        """A non-positive run length is rejected with ``None``."""

        assert _last_bin_of_consecutive_run(np.array([1, 1, 1], dtype=bool), 0) is None

    def test_run_longer_than_array_returns_none(self):
        """A required run longer than the array cannot match."""

        assert _last_bin_of_consecutive_run(np.array([1, 1], dtype=bool), 5) is None


# Figure-emission smoke tests for the plotting functions.
#
# Each test below builds a minimal synthetic artifact that satisfies the
# exact schema a given plotter consumes, pickles it under `tmp_path`,
# calls the plotter with its save flag set + `output_dir`/`save_dir`
# pointed at `tmp_path`, and asserts the plotter returns without error
# and writes an output file into `tmp_path`. The goal is to exercise the
# drawing code paths (figure emission), not pixel-level correctness, so
# the synthetic data is shaped — not realistic. The matplotlib backend
# is forced to ``Agg`` at module import so nothing opens a window;
# ``plt.close('all')`` runs after every test (autouse fixture below) so
# the per-process figure count never trips the "More than 20 figures"
# warning across tests.


@pytest.fixture(autouse=True)
def _close_all_figures():
    """
    Close every open matplotlib figure after each test.

    The plotters call ``plt.show()`` (a no-op under the ``Agg`` backend)
    but never close the non-saved figures themselves, so without this
    teardown the open-figure count accumulates across tests and matplotlib
    raises the "More than 20 figures have been opened" RuntimeWarning —
    which ``filterwarnings = ["error"]`` would promote to a failure.
    """

    yield
    plt.close('all')


def _univariate_feature_entry(rng, sig: bool, lower_is_better: bool,
                              n_folds: int = 8, n_time: int = 12) -> dict:
    """
    Build one univariate feature-result entry for the linear/GAM
    ranking + filter plotters.

    The entry mirrors the per-feature schema those plotters read:
    ``{'actual': {metric: per-fold array, 'filter_shapes': (folds, time)},
    'shuffled': {metric: per-fold null array}}``. The ``actual`` metric
    values are pushed clearly above (or, for lower-is-better metrics,
    below) the shuffled null so the feature reads as significant when
    ``sig`` is True and clearly inside the null otherwise.

    Parameters
    ----------
    rng : numpy.random.Generator
        Seeded generator for reproducible synthetic values.
    sig : bool
        Whether the feature should clear the Bonferroni-corrected
        significance threshold for the metrics tested here.
    lower_is_better : bool
        When True the metric direction is inverted (e.g. ``ll``): the
        ``actual`` distribution is placed *below* the null for a
        significant feature.
    n_folds : int, default 8
        Number of cross-validation folds (length of each metric array).
    n_time : int, default 12
        Number of filter time bins (columns of ``filter_shapes``).

    Returns
    -------
    dict
        A single feature entry with ``actual`` and ``shuffled`` blocks.
    """

    null = rng.uniform(0.45, 0.55, size=n_folds)
    if lower_is_better:
        actual = (rng.uniform(0.05, 0.15, size=n_folds) if sig
                  else rng.uniform(0.45, 0.55, size=n_folds))
    else:
        actual = (rng.uniform(0.85, 0.95, size=n_folds) if sig
                  else rng.uniform(0.45, 0.55, size=n_folds))
    filter_shapes = rng.standard_normal(size=(n_folds, n_time))
    return {
        'actual': {
            'auc': actual.copy(),
            'll': actual.copy(),
            'explained_deviance': actual.copy(),
            'spearman_r': rng.uniform(0.1, 0.4, size=n_folds),
            'filter_shapes': filter_shapes,
        },
        'shuffled': {
            'auc': null.copy(),
            'll': null.copy(),
            'explained_deviance': null.copy(),
            'spearman_r': rng.uniform(-0.1, 0.1, size=n_folds),
        },
    }


def _write_univariate_pickle(tmp_path, rng) -> str:
    """
    Write a synthetic consolidated-univariate-style pickle to `tmp_path`.

    The artifact name carries the ``_male_`` token so the plotters pick
    the male/female self/other colour split. It contains five feature
    entries. Four span every colour branch the plotters classify on:
    a ``self.*`` feature, an ``other.*`` feature, a ``-sei`` engagement
    feature, and a ``nose-nose`` dyadic feature; these four are made
    significant so the filter plots draw a panel for each. A fifth
    ``other.acceleration`` entry is intentionally not significant so the
    neutral-colour branch in the ranking plot is exercised too.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    rng : numpy.random.Generator
        Seeded generator.

    Returns
    -------
    str
        Absolute path to the written pickle (filename contains
        ``_male_``).
    """

    data = {
        'self.speed': _univariate_feature_entry(rng, sig=True, lower_is_better=False),
        'other.speed': _univariate_feature_entry(rng, sig=True, lower_is_better=False),
        'orofacial-sei': _univariate_feature_entry(rng, sig=True, lower_is_better=False),
        'nose-nose': _univariate_feature_entry(rng, sig=True, lower_is_better=False),
        # An intentionally not-significant feature so the neutral-colour
        # branch in the ranking plot is exercised too.
        'other.acceleration': _univariate_feature_entry(rng, sig=False, lower_is_better=False),
    }
    out = tmp_path / "onsets_bout_male_hist4.0s_results.pkl"
    with out.open('wb') as fh:
        pickle.dump(data, fh)
    return str(out)


@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive:UserWarning")
class TestPlotFeatureRanking:
    """Figure-emission tests for ``plot_feature_ranking``."""

    @pytest.mark.filterwarnings("ignore:Tight layout:UserWarning")
    def test_writes_two_ranking_svgs(self, tmp_path):
        """With ``save_plot=True`` and an explicit ``output_dir`` the
        function writes one ranking SVG per configured metric (primary
        evaluation metric + secondary metric)."""

        rng = np.random.default_rng(7)
        pkl = _write_univariate_pickle(tmp_path, rng)
        out_dir = tmp_path / "ranking_out"
        out_dir.mkdir()
        plot_feature_ranking(
            results_file_loc=pkl,
            evaluation_metric='auc',
            secondary_metric='spearman_r',
            save_plot=True,
            output_dir=str(out_dir),
        )
        svgs = list(out_dir.glob("*_ranking.svg"))
        assert len(svgs) == 2

    @pytest.mark.filterwarnings("ignore:Tight layout:UserWarning")
    def test_runs_without_saving(self, tmp_path):
        """Default ``save_plot=False`` returns None and writes nothing."""

        rng = np.random.default_rng(8)
        pkl = _write_univariate_pickle(tmp_path, rng)
        assert plot_feature_ranking(results_file_loc=pkl, evaluation_metric='auc') is None


@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive:UserWarning")
class TestPlotSignificantFilters:
    """Figure-emission tests for ``plot_significant_filters``."""

    def test_writes_per_feature_filter_svgs(self, tmp_path):
        """Each significant feature yields one per-feature filter SVG."""

        rng = np.random.default_rng(11)
        pkl = _write_univariate_pickle(tmp_path, rng)
        out_dir = tmp_path / "filters_out"
        out_dir.mkdir()
        plot_significant_filters(
            results_file_loc=pkl,
            metric='auc',
            save_plot=True,
            output_dir=str(out_dir),
        )
        svgs = list(out_dir.glob("*_filter_*.svg"))
        assert len(svgs) >= 1


@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive:UserWarning")
class TestPlotSignificantFiltersGrid:
    """Figure-emission tests for ``plot_significant_filters_grid``."""

    @pytest.mark.filterwarnings("ignore:Tight layout:UserWarning")
    def test_writes_filter_grid_svg(self, tmp_path):
        """A populated significant-feature set writes a single filter-grid
        SVG into the output directory."""

        rng = np.random.default_rng(13)
        pkl = _write_univariate_pickle(tmp_path, rng)
        out_dir = tmp_path / "grid_out"
        out_dir.mkdir()
        plot_significant_filters_grid(
            results_file_loc=pkl,
            metric='auc',
            save_plot=True,
            output_dir=str(out_dir),
        )
        assert len(list(out_dir.glob("*_filter_grid.svg"))) == 1


def _selection_step(rng, feature_names, selected, n_folds: int = 6,
                    n_time: int = 12, with_filters: bool = False,
                    baseline: float | None = None) -> dict:
    """
    Build one forward-selection step dict for ``plot_model_selection_results``.

    The step mirrors the consolidated ``selection_*.pkl`` step schema the
    plotter reads: a ``candidates`` mapping ``feature -> {'ll': per-fold,
    'auc': per-fold, 'explained_deviance': per-fold}``, the
    ``selected_feature`` (``None`` marks a rejection step), an optional
    ``baseline_score`` (the chance NLL on the first step), and an optional
    ``filter_shapes`` block (list of per-fold ``{feature: array}`` dicts)
    used by the final-model filter grid.

    Parameters
    ----------
    rng : numpy.random.Generator
        Seeded generator.
    feature_names : list of str
        Candidate feature names evaluated at this step.
    selected : str or None
        The winning feature for this step; ``None`` for a rejection step.
    n_folds : int, default 6
        Cross-validation fold count.
    n_time : int, default 12
        Filter time-bin count (only used when ``with_filters`` is True).
    with_filters : bool, default False
        When True the step carries a ``filter_shapes`` list-of-fold-dicts
        block keyed by ``selected``-plus-anchor features.
    baseline : float or None, default None
        When not None, stored under ``baseline_score`` (step 0 only).

    Returns
    -------
    dict
        A single selection-step dict.
    """

    candidates = {}
    for f in feature_names:
        candidates[f] = {
            'll': rng.uniform(0.2, 0.6, size=n_folds),
            'auc': rng.uniform(0.55, 0.8, size=n_folds),
            'explained_deviance': rng.uniform(0.05, 0.3, size=n_folds),
        }
    step = {'candidates': candidates, 'selected_feature': selected}
    if baseline is not None:
        step['baseline_score'] = float(baseline)
    if with_filters and selected is not None:
        folds = []
        for _ in range(n_folds):
            folds.append({selected: rng.standard_normal(n_time)})
        step['filter_shapes'] = folds
    return step


def _write_selection_pickle(tmp_path, rng) -> str:
    """
    Write a synthetic consolidated ``selection_*.pkl`` to `tmp_path`.

    The artifact has a ``steps`` list (two accepted steps + one rejection
    step) so the trajectory plot draws accepted bars, the rejected-row
    branch, and the right-panel univariate/final bars; the last accepted
    step carries ``filter_shapes`` so the final-model filter grid renders.
    The filename carries ``_male_`` for the colour split and the
    ``selection_`` prefix so ``load_selection_results`` accepts it.

    Returns
    -------
    str
        Absolute path to the written pickle.
    """

    feats = ['self.speed', 'other.speed', 'nose-nose']
    steps = [
        _selection_step(rng, feats, selected='self.speed',
                        baseline=float(np.log(2.0))),
        _selection_step(rng, ['other.speed', 'nose-nose'],
                        selected='other.speed', with_filters=True),
        # Rejection step: no winner accepted.
        _selection_step(rng, ['nose-nose'], selected=None),
    ]
    out = tmp_path / "selection_onsets_bout_male_hist4.0s.pkl"
    with out.open('wb') as fh:
        pickle.dump({'steps': steps}, fh)
    return str(out)


@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive:UserWarning")
class TestPlotModelSelectionResults:
    """Figure-emission tests for ``plot_model_selection_results``."""

    @pytest.mark.filterwarnings("ignore:Tight layout:UserWarning")
    def test_writes_trajectory_and_filter_grid(self, tmp_path):
        """A two-accepted-step + rejection selection set writes both the
        trajectory SVG and the final-model filter-grid SVG."""

        rng = np.random.default_rng(17)
        pkl = _write_selection_pickle(tmp_path, rng)
        out_dir = tmp_path / "sel_out"
        out_dir.mkdir()
        plot_model_selection_results(
            selection_results_path=pkl,
            metric_secondary='auc',
            save_plots=True,
            output_dir=str(out_dir),
        )
        assert (out_dir / "model_selection_trajectory.svg").is_file()
        assert (out_dir / "model_selection_final_model_filters.svg").is_file()

    @pytest.mark.filterwarnings("ignore:Tight layout:UserWarning")
    def test_label_overrides_accepted(self, tmp_path):
        """Presentation label overrides are accepted without error."""

        rng = np.random.default_rng(18)
        pkl = _write_selection_pickle(tmp_path, rng)
        out_dir = tmp_path / "sel_out2"
        out_dir.mkdir()
        plot_model_selection_results(
            selection_results_path=pkl,
            save_plots=True,
            output_dir=str(out_dir),
            feature_label_overrides={'self.speed': 'Own speed'},
        )
        assert (out_dir / "model_selection_trajectory.svg").is_file()


def _multinomial_feature_entry(rng, n_classes: int, sig: bool,
                               n_folds: int = 6, n_samples: int = 40) -> dict:
    """
    Build one multinomial feature-result entry.

    Schema (as read by ``plot_univariate_multinomial_performance``):
    ``{'actual': {'folds': {'metrics': {metric: per-fold array},
    'y_true': [per-fold label arrays], 'y_pred': [...]}, 'classes': [...]},
    'null': {...}}``. ``actual`` ``ll`` is pushed below the null when
    ``sig`` so the feature clears the lower-is-better significance test.

    Parameters
    ----------
    rng : numpy.random.Generator
        Seeded generator.
    n_classes : int
        Number of vocal categories (confusion-matrix dimension).
    sig : bool
        Whether the feature should read as significant on ``ll``.
    n_folds : int, default 6
        Cross-validation fold count.
    n_samples : int, default 40
        Per-fold sample count for the y_true / y_pred arrays.

    Returns
    -------
    dict
        A single multinomial feature entry.
    """

    classes = [f"cat{c}" for c in range(n_classes)]

    def _block(is_actual: bool) -> dict:
        if is_actual and sig:
            ll = rng.uniform(0.1, 0.3, size=n_folds)
        else:
            ll = rng.uniform(0.6, 0.9, size=n_folds)
        # Draw labels from `classes` (not bare ints) so y_true / y_pred share the
        # dtype of the class labels, as they do in the real pipeline (both come
        # from the model's string category labels); the confusion matrices are
        # now built with labels=class_names, which requires this consistency.
        y_true = [rng.choice(classes, size=n_samples) for _ in range(n_folds)]
        y_pred = [rng.choice(classes, size=n_samples) for _ in range(n_folds)]
        # Per-fold weight tensor (n_folds, n_classes, n_time_bins) consumed
        # by the multinomial filter-grid heatmap plotter.
        weights = rng.standard_normal(size=(n_folds, n_classes, 16))
        return {
            'folds': {
                'metrics': {
                    'll': ll,
                    'score': rng.uniform(0.4, 0.7, size=n_folds),
                    'auc': rng.uniform(0.55, 0.85, size=n_folds),
                },
                'y_true': y_true,
                'y_pred': y_pred,
                'weights': weights,
            },
            'classes': classes,
        }

    return {'actual': _block(True), 'null': _block(False)}


@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive:UserWarning")
class TestPlotUnivariateMultinomialPerformance:
    """Figure-emission tests for ``plot_univariate_multinomial_performance``."""

    @pytest.mark.filterwarnings("ignore:Tight layout:UserWarning")
    @pytest.mark.filterwarnings("ignore:This figure includes Axes:UserWarning")
    def test_writes_ranking_and_confusion_trio(self, tmp_path):
        """A significant top feature writes both the ranking SVG and the
        confusion-trio SVG."""

        rng = np.random.default_rng(23)
        data = {
            'self.speed': _multinomial_feature_entry(rng, n_classes=3, sig=True),
            'other.speed': _multinomial_feature_entry(rng, n_classes=3, sig=False),
            'nose-nose': _multinomial_feature_entry(rng, n_classes=3, sig=True),
        }
        pkl = tmp_path / "multinomial_male_hist4.0s.pkl"
        with pkl.open('wb') as fh:
            pickle.dump(data, fh)
        out_dir = tmp_path / "mn_out"
        out_dir.mkdir()
        plot_univariate_multinomial_performance(
            results_file_loc=str(pkl),
            save_plot=True,
            output_dir=str(out_dir),
        )
        assert len(list(out_dir.glob("*_ranking.svg"))) == 1
        assert len(list(out_dir.glob("*_confusion_trio.svg"))) == 1

    @pytest.mark.filterwarnings("ignore:Tight layout:UserWarning")
    @pytest.mark.filterwarnings("ignore:This figure includes Axes:UserWarning")
    def test_confusion_trio_when_null_omits_a_category(self, tmp_path):
        """Regression: the null model predicting only the majority class — with a
        category that never appears as a true label — must not crash the
        actual-minus-null subtraction. Without labels=class_names the 'actual'
        matrix (which sees the extra predicted category) and the 'null' matrix
        would be different shapes and broadcast-error."""

        rng = np.random.default_rng(7)
        classes = ["cat0", "cat1", "cat2"]
        n_folds, n = 6, 30

        def _block(actual: bool) -> dict:
            ll = rng.uniform(0.1, 0.3, n_folds) if actual else rng.uniform(0.6, 0.9, n_folds)
            # True labels never include 'cat2'. The ACTUAL model still predicts it
            # sometimes; the NULL model predicts only the majority 'cat0'. So
            # without labels= the actual matrix is 3x3 and the null matrix 2x2.
            y_true = [rng.choice(["cat0", "cat1"], size=n) for _ in range(n_folds)]
            y_pred = ([rng.choice(classes, size=n) for _ in range(n_folds)] if actual
                      else [np.array(["cat0"] * n) for _ in range(n_folds)])
            return {
                'folds': {
                    'metrics': {'ll': ll,
                                'score': rng.uniform(0.4, 0.7, n_folds),
                                'auc': rng.uniform(0.55, 0.85, n_folds)},
                    'y_true': y_true, 'y_pred': y_pred,
                    'weights': rng.standard_normal((n_folds, len(classes), 16)),
                },
                'classes': classes,
            }

        data = {'self.speed': {'actual': _block(True), 'null': _block(False)}}
        pkl = tmp_path / "multinomial_male_hist4.0s.pkl"
        with pkl.open('wb') as fh:
            pickle.dump(data, fh)
        out_dir = tmp_path / "omit_out"
        out_dir.mkdir()
        plot_univariate_multinomial_performance(
            results_file_loc=str(pkl), save_plot=True, output_dir=str(out_dir),
        )
        assert len(list(out_dir.glob("*_confusion_trio.svg"))) == 1


@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive:UserWarning")
class TestPlotRawFeatureDifference:
    """Figure-emission tests for ``plot_raw_feature_difference``."""

    @pytest.mark.filterwarnings("ignore:Tight layout:UserWarning")
    def test_bout_onset_mode_writes_avg_and_heatmap(self, tmp_path):
        """A Bout-Onset-shaped input pickle (``usv_feature_arr`` /
        ``no_usv_feature_arr`` per session) writes the bootstrap-average
        SVG and the heatmap SVG."""

        rng = np.random.default_rng(29)
        n_time = 60
        feature_key = 'self.speed'
        data = {
            feature_key: {
                'session_a': {
                    'usv_feature_arr': rng.standard_normal((30, n_time)),
                    'no_usv_feature_arr': rng.standard_normal((40, n_time)),
                },
                'session_b': {
                    'usv_feature_arr': rng.standard_normal((25, n_time)),
                    'no_usv_feature_arr': rng.standard_normal((35, n_time)),
                },
            }
        }
        pkl = tmp_path / "bout_onset_input.pkl"
        with pkl.open('wb') as fh:
            pickle.dump(data, fh)
        out_dir = tmp_path / "raw_out"
        out_dir.mkdir()
        plot_raw_feature_difference(
            pickle_file_path=str(pkl),
            feature_key=feature_key,
            subset_fraction=0.5,
            n_bootstraps=20,
            save_plots=True,
            output_dir=str(out_dir),
        )
        assert len(list(out_dir.glob("*avg_bootstrap.svg"))) == 1
        assert len(list(out_dir.glob("*heatmap.svg"))) == 1

    def test_empty_condition_skips_without_crash(self, tmp_path):
        """A feature where one condition contributes zero epochs must skip the plot
        (log a message and return) rather than crash in np.nanmin / np.random.choice
        ('cannot take a larger sample than population when replace=False')."""
        n_time = 60
        feature_key = 'self.speed'
        rng = np.random.default_rng(31)
        data = {
            feature_key: {
                'session_a': {
                    'usv_feature_arr': np.empty((0, n_time)),   # zero target epochs
                    'no_usv_feature_arr': rng.standard_normal((35, n_time)),
                },
            }
        }
        pkl = tmp_path / "empty_cond_input.pkl"
        with pkl.open('wb') as fh:
            pickle.dump(data, fh)
        out_dir = tmp_path / "raw_empty_out"
        out_dir.mkdir()
        # Must not raise; the degenerate feature is skipped so no SVGs are written.
        plot_raw_feature_difference(
            pickle_file_path=str(pkl),
            feature_key=feature_key,
            subset_fraction=0.5,
            n_bootstraps=20,
            save_plots=True,
            output_dir=str(out_dir),
        )
        assert list(out_dir.glob("*.svg")) == []

    @pytest.mark.filterwarnings("ignore:Tight layout:UserWarning")
    def test_vocal_categories_mode_writes_avg_and_heatmap(self, tmp_path):
        """A Vocal-Categories-shaped input pickle (``target_feature_arr`` /
        ``other_feature_arr`` per session) selects the target-vs-other mode
        branch and still writes the bootstrap-average and heatmap SVGs."""

        rng = np.random.default_rng(30)
        n_time = 60
        feature_key = 'self.speed'
        data = {
            feature_key: {
                'session_a': {
                    'target_feature_arr': rng.standard_normal((30, n_time)),
                    'other_feature_arr': rng.standard_normal((40, n_time)),
                },
                'session_b': {
                    'target_feature_arr': rng.standard_normal((25, n_time)),
                    'other_feature_arr': rng.standard_normal((35, n_time)),
                },
            }
        }
        pkl = tmp_path / "vocal_categories_input.pkl"
        with pkl.open('wb') as fh:
            pickle.dump(data, fh)
        out_dir = tmp_path / "raw_cat_out"
        out_dir.mkdir()
        plot_raw_feature_difference(
            pickle_file_path=str(pkl),
            feature_key=feature_key,
            subset_fraction=0.5,
            n_bootstraps=20,
            save_plots=True,
            output_dir=str(out_dir),
        )
        assert len(list(out_dir.glob("*avg_bootstrap.svg"))) == 1
        assert len(list(out_dir.glob("*heatmap.svg"))) == 1

    def test_missing_feature_key_returns_without_drawing(self, tmp_path):
        """A ``feature_key`` absent from the pickle short-circuits with a
        message and writes nothing."""

        rng = np.random.default_rng(31)
        data = {
            'self.speed': {
                'session_a': {
                    'usv_feature_arr': rng.standard_normal((10, 40)),
                    'no_usv_feature_arr': rng.standard_normal((10, 40)),
                },
            }
        }
        pkl = tmp_path / "missing_key_input.pkl"
        with pkl.open('wb') as fh:
            pickle.dump(data, fh)
        out_dir = tmp_path / "raw_missing_out"
        out_dir.mkdir()
        plot_raw_feature_difference(
            pickle_file_path=str(pkl),
            feature_key='not.a.feature',
            save_plots=True,
            output_dir=str(out_dir),
        )
        assert list(out_dir.glob("*.svg")) == []


def _write_collinearity_pickle(tmp_path, rng, n_features: int = 5) -> str:
    """
    Write a synthetic ``_collinearity.pkl`` audit artifact to `tmp_path`.

    Schema (as read by ``plot_collinearity_audit``): ``features``,
    ``spearman_rho`` (symmetric ``(n, n)`` with unit diagonal), ``vif``
    (length-``n``, one entry set to ``+inf`` to exercise the singular-
    design branch), ``flagged_pairs``, ``n_events``, ``condition_number``,
    and ``source_pickle`` (its ``_male_`` token drives the colour split).
    The feature names span the self / other / sei / dyadic groups so the
    group-ordering + per-group colouring code runs.

    Returns
    -------
    str
        Absolute path to the written pickle.
    """

    features = ['self.speed', 'other.speed', 'orofacial-sei',
                'nose-nose', 'self.acceleration'][:n_features]
    n = len(features)
    rho = rng.uniform(-0.9, 0.9, size=(n, n))
    rho = (rho + rho.T) / 2.0
    np.fill_diagonal(rho, 1.0)
    vif = rng.uniform(1.0, 8.0, size=n)
    vif[0] = np.inf  # exercise the inf / singular-design annotation branch
    payload = {
        'features': features,
        'spearman_rho': rho,
        'vif': vif,
        'flagged_pairs': [(features[0], features[1], 0.85)],
        'n_events': 500,
        'condition_number': 42.0,
        'source_pickle': 'onsets_bout_male_hist4.0s_results.pkl',
    }
    out = tmp_path / "onsets_bout_male_collinearity.pkl"
    with out.open('wb') as fh:
        pickle.dump(payload, fh)
    return str(out)


@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive:UserWarning")
class TestPlotCollinearityAudit:
    """Figure-emission tests for ``plot_collinearity_audit``."""

    @pytest.mark.filterwarnings("ignore:Tight layout:UserWarning")
    def test_writes_figure_and_returns_summary(self, tmp_path):
        """The audit writes a figure into ``save_dir`` and returns the
        documented summary dict (feature / flagged counts, condition
        number, finite-VIF mean / median)."""

        rng = np.random.default_rng(31)
        pkl = _write_collinearity_pickle(tmp_path, rng)
        save_dir = tmp_path / "coll_out"
        save_dir.mkdir()
        result = plot_collinearity_audit(
            audit_pkl_path=pkl,
            save_dir=str(save_dir),
            save_plot_bool=True,
            plot_format='svg',
        )
        assert result['n_features'] == 5
        assert result['n_flagged'] == 1
        assert result['condition_number'] == pytest.approx(42.0)
        assert result['figure_path'].endswith('.svg')
        assert len(list(save_dir.glob("*.svg"))) == 1

    def test_no_save_returns_empty_path(self, tmp_path):
        """With ``save_plot_bool=False`` no figure is written and the
        returned ``figure_path`` is empty."""

        rng = np.random.default_rng(32)
        pkl = _write_collinearity_pickle(tmp_path, rng)
        result = plot_collinearity_audit(audit_pkl_path=pkl, save_plot_bool=False)
        assert result['figure_path'] == ''

    def test_empty_features_short_circuits(self, tmp_path):
        """A zero-feature audit returns the documented empty summary
        without drawing."""

        payload = {
            'features': [],
            'spearman_rho': np.zeros((0, 0)),
            'vif': np.zeros((0,)),
            'flagged_pairs': [],
            'n_events': 0,
            'condition_number': 1.0,
            'source_pickle': 'x_male_.pkl',
        }
        pkl = tmp_path / "empty_collinearity.pkl"
        with pkl.open('wb') as fh:
            pickle.dump(payload, fh)
        result = plot_collinearity_audit(audit_pkl_path=str(pkl),
                                         save_plot_bool=False)
        assert result['n_features'] == 0
        assert result['figure_path'] == ''


def _write_timescale_pickle(tmp_path, rng, n_features: int = 4,
                            n_acf_lags: int = 80,
                            n_signal_lags: int = 161) -> str:
    """
    Write a synthetic ``_timescales.pkl`` audit artifact to `tmp_path`.

    Builds every key both ``_require_timescale_payload`` and the two
    timescale plotters consume: the ACF block (median / IQR / circular-
    shift null mean+percentiles on positive lags) and the symmetric
    cross-correlation block (per-session mean ρ + SEM, circular-shift
    null mean+percentiles). The ACF median is forced clearly above its
    upper null on the early lags so the ``acf_run_length``-bin run marker
    fires for every feature, and the cross-correlation curve is given a
    sustained positive-lag bump above its upper null so the XC marker
    fires too — populating both cohort-summary panels.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    rng : numpy.random.Generator
        Seeded generator.
    n_features : int, default 4
        Feature count (spans self / other / sei / dyadic groups).
    n_acf_lags : int, default 80
        Number of positive ACF lag bins.
    n_signal_lags : int, default 161
        Number of symmetric cross-correlation lag bins (odd so zero lag
        is centred).

    Returns
    -------
    str
        Absolute path to the written pickle.
    """

    features = ['self.speed', 'other.speed', 'orofacial-sei', 'nose-nose'][:n_features]
    n = len(features)

    acf_lags = np.linspace(0.0, 4.0, n_acf_lags)
    # Decaying ACF that stays above the (near-zero) null for the early
    # lags so the consecutive-run marker has something to find.
    base_acf = np.exp(-acf_lags / 1.5)[None, :].repeat(n, axis=0)
    acf_med = np.clip(base_acf + rng.normal(0, 0.01, size=(n, n_acf_lags)), 0.0, 1.0)
    acf_p25 = np.clip(acf_med - 0.05, 0.0, 1.0)
    acf_p75 = np.clip(acf_med + 0.05, 0.0, 1.0)
    acf_null_mean = np.full((n, n_acf_lags), 0.01)
    acf_null_lo = np.full((n, n_acf_lags), -0.02)
    acf_null_hi = np.full((n, n_acf_lags), 0.05)

    signal_lags = np.linspace(-4.0, 4.0, n_signal_lags)
    # Sustained positive bump on the positive-lag side (>= signal floor)
    # so the cross-correlation outer-run marker fires.
    rho_signal = np.zeros((n, n_signal_lags))
    pos_mask = (signal_lags >= 0.6) & (signal_lags <= 2.5)
    rho_signal[:, pos_mask] = 0.30
    rho_signal += rng.normal(0, 0.005, size=(n, n_signal_lags))
    rho_signal_sem = np.full((n, n_signal_lags), 0.01)
    rho_signal_null_mean = np.zeros((n, n_signal_lags))
    rho_signal_null_lo = np.full((n, n_signal_lags), -0.05)
    rho_signal_null_hi = np.full((n, n_signal_lags), 0.05)

    payload = {
        'features': features,
        'acf_lags_seconds': acf_lags,
        'acf_median': acf_med,
        'acf_p25': acf_p25,
        'acf_p75': acf_p75,
        'acf_null_mean': acf_null_mean,
        'acf_null_p0_5': acf_null_lo,
        'acf_null_p99_5': acf_null_hi,
        'rho_signal': rho_signal,
        'rho_signal_per_session_sem': rho_signal_sem,
        'signal_lags_seconds': signal_lags,
        'rho_signal_null_mean': rho_signal_null_mean,
        'rho_signal_null_p0_5': rho_signal_null_lo,
        'rho_signal_null_p99_5': rho_signal_null_hi,
        'signal_floor_seconds': 0.5,
        'signal_min_run_seconds': 0.2,
        'configured_filter_history': 4.0,
        'source_pickle': 'onsets_bout_male_hist4.0s_results.pkl',
    }
    out = tmp_path / "onsets_bout_male_timescales.pkl"
    with out.open('wb') as fh:
        pickle.dump(payload, fh)
    return str(out)


@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive:UserWarning")
class TestPlotTimescaleAudit:
    """Figure-emission tests for the cohort-summary timescale plot."""

    @pytest.mark.filterwarnings("ignore:Tight layout:UserWarning")
    def test_writes_cohort_summary_figure(self, tmp_path):
        """The cohort summary writes a figure and reports a positive ACF
        feature count (every synthetic feature clears the marker)."""

        rng = np.random.default_rng(37)
        pkl = _write_timescale_pickle(tmp_path, rng)
        save_dir = tmp_path / "ts_out"
        save_dir.mkdir()
        result = plot_timescale_audit(
            timescale_pkl_path=pkl,
            save_dir=str(save_dir),
            save_plot_bool=True,
        )
        assert result['n_features_acf'] >= 1
        assert result['figure_path'].endswith('.svg')
        assert len(list(save_dir.glob("*.svg"))) == 1

    def test_rejects_non_timescale_pickle(self, tmp_path):
        """A pickle missing the canonical timescale keys raises a clear
        ``ValueError``."""

        bad = tmp_path / "not_timescales.pkl"
        with bad.open('wb') as fh:
            pickle.dump({'features': ['a']}, fh)
        with pytest.raises(ValueError, match='does not look like a timescale-audit pickle'):
            plot_timescale_audit(timescale_pkl_path=str(bad), save_plot_bool=False)


@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive:UserWarning")
class TestPlotTimescaleAuditPerFeature:
    """Figure-emission tests for the per-feature timescale grid."""

    @pytest.mark.filterwarnings("ignore:Tight layout:UserWarning")
    def test_writes_per_feature_grid(self, tmp_path):
        """The per-feature small-multiples grid writes a figure and reports
        the feature count back."""

        rng = np.random.default_rng(41)
        pkl = _write_timescale_pickle(tmp_path, rng)
        save_dir = tmp_path / "tspf_out"
        save_dir.mkdir()
        result = plot_timescale_audit_per_feature(
            timescale_pkl_path=pkl,
            save_dir=str(save_dir),
            save_plot_bool=True,
        )
        assert result['n_features'] == 4
        assert result['figure_path'].endswith('.svg')
        assert len(list(save_dir.glob("*.svg"))) == 1


@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive:UserWarning")
class TestPlotUnivariateMultinomialFiltersGrid:
    """Figure-emission tests for ``plot_univariate_multinomial_filters_grid``."""

    @pytest.mark.filterwarnings("ignore:Tight layout:UserWarning")
    def test_writes_filter_grid_svg(self, tmp_path):
        """At least one significant multinomial feature writes a single
        weight-heatmap grid SVG."""

        rng = np.random.default_rng(43)
        data = {
            'self.speed': _multinomial_feature_entry(rng, n_classes=3, sig=True),
            'nose-nose': _multinomial_feature_entry(rng, n_classes=3, sig=True),
            'other.speed': _multinomial_feature_entry(rng, n_classes=3, sig=False),
        }
        pkl = tmp_path / "multinomial_male_hist4.0s.pkl"
        with pkl.open('wb') as fh:
            pickle.dump(data, fh)
        out_dir = tmp_path / "mngrid_out"
        out_dir.mkdir()
        plot_univariate_multinomial_filters_grid(
            results_file_loc=str(pkl),
            evaluation_metric='ll',
            save_plot=True,
            output_dir=str(out_dir),
        )
        assert len(list(out_dir.glob("*_multinomial_filters_grid.svg"))) == 1


def _multinomial_candidate(rng, n_classes: int, n_folds: int = 6) -> dict:
    """
    Build one multinomial selection candidate-summary entry.

    Mirrors the ``candidates_summary[feature]`` schema the multinomial
    trajectory plotter reads: ``{'folds': {'metrics': {metric: per-fold
    array}}, 'classes': [...]}``.

    Parameters
    ----------
    rng : numpy.random.Generator
        Seeded generator.
    n_classes : int
        Number of USV categories (sets the secondary-metric chance floor).
    n_folds : int, default 6
        Cross-validation fold count.

    Returns
    -------
    dict
        A single candidate-summary entry.
    """

    return {
        'folds': {
            'metrics': {
                'auc': rng.uniform(0.55, 0.85, size=n_folds),
                'score': rng.uniform(0.4, 0.7, size=n_folds),
                'll': rng.uniform(0.2, 0.6, size=n_folds),
            },
        },
        'classes': [f"cat{c}" for c in range(n_classes)],
    }


def _write_multinomial_selection_pickle(tmp_path, rng, n_classes: int = 3) -> str:
    """
    Write a synthetic consolidated multinomial ``selection_*.pkl`` to
    `tmp_path`.

    The artifact has a ``steps`` list of two accepted steps (each step
    carries a ``candidates_summary`` keyed by feature) plus one rejection
    step (``selected_feature`` is ``None``) so the plotter exercises the
    accepted-bar, rejected-row, and best-univariate-bar code paths. The
    filename carries ``_male_`` for the colour split and ``selection_`` so
    ``load_selection_results`` accepts it.

    Returns
    -------
    str
        Absolute path to the written pickle.
    """

    feats = ['self.speed', 'other.speed', 'nose-nose']
    step0 = {
        'selected_feature': 'self.speed',
        'candidates_summary': {f: _multinomial_candidate(rng, n_classes) for f in feats},
    }
    step1 = {
        'selected_feature': 'other.speed',
        'candidates_summary': {
            f: _multinomial_candidate(rng, n_classes)
            for f in ['other.speed', 'nose-nose']
        },
    }
    step2 = {
        'selected_feature': None,
        'candidates_summary': {'nose-nose': _multinomial_candidate(rng, n_classes)},
    }
    out = tmp_path / "selection_multinomial_male_hist4.0s.pkl"
    with out.open('wb') as fh:
        pickle.dump({'steps': [step0, step1, step2]}, fh)
    return str(out)


@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive:UserWarning")
class TestPlotMultinomialSelectionTrajectory:
    """Figure-emission tests for ``plot_multinomial_selection_trajectory``."""

    @pytest.mark.filterwarnings("ignore:Tight layout:UserWarning")
    def test_writes_trajectory_svg(self, tmp_path):
        """An accepted-steps-plus-rejection multinomial selection set writes
        a single trajectory SVG named for the cohort + primary metric."""

        rng = np.random.default_rng(47)
        pkl = _write_multinomial_selection_pickle(tmp_path, rng)
        out_dir = tmp_path / "mntraj_out"
        out_dir.mkdir()
        plot_multinomial_selection_trajectory(
            selection_results_path=pkl,
            metric_primary='auc',
            metric_secondary='score',
            save_plot=True,
            output_dir=str(out_dir),
        )
        assert len(list(out_dir.glob("multinomial_selection_trajectory_*_auc.svg"))) == 1


def _manifold_candidate(rng, n_folds: int = 6) -> dict:
    """
    Build one manifold (continuous-vocal-manifold) selection candidate-
    summary entry.

    Mirrors the ``candidates_summary[feature]`` schema the manifold
    trajectory plotter reads: ``{'folds': {'metrics': {metric: per-fold
    array}}}`` with the 2-D regression metrics it scores on
    (``r2_spatial``, ``pearson_y``, ``euclidean_mae``).

    Parameters
    ----------
    rng : numpy.random.Generator
        Seeded generator.
    n_folds : int, default 6
        Cross-validation fold count.

    Returns
    -------
    dict
        A single candidate-summary entry.
    """

    return {
        'folds': {
            'metrics': {
                'r2_spatial': rng.uniform(0.05, 0.4, size=n_folds),
                'pearson_y': rng.uniform(0.1, 0.5, size=n_folds),
                'euclidean_mae': rng.uniform(0.5, 1.5, size=n_folds),
            },
        },
    }


def _write_manifold_selection_pickle(tmp_path, rng) -> str:
    """
    Write a synthetic consolidated manifold ``selection_*.pkl`` to
    `tmp_path`.

    Two accepted steps plus one rejection step, each carrying a
    ``candidates_summary`` keyed by feature, so the trajectory plotter
    exercises the accepted-bar, rejected-row, and best-univariate-bar
    paths. Filename carries ``_male_`` for the colour split and the
    ``selection_`` prefix so ``load_selection_results`` accepts it.

    Returns
    -------
    str
        Absolute path to the written pickle.
    """

    feats = ['self.speed', 'other.speed', 'nose-nose']
    step0 = {
        'selected_feature': 'self.speed',
        'candidates_summary': {f: _manifold_candidate(rng) for f in feats},
    }
    step1 = {
        'selected_feature': 'other.speed',
        'candidates_summary': {
            f: _manifold_candidate(rng) for f in ['other.speed', 'nose-nose']
        },
    }
    step2 = {
        'selected_feature': None,
        'candidates_summary': {'nose-nose': _manifold_candidate(rng)},
    }
    out = tmp_path / "selection_manifold_male_hist4.0s.pkl"
    with out.open('wb') as fh:
        pickle.dump({'steps': [step0, step1, step2]}, fh)
    return str(out)


@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive:UserWarning")
class TestPlotManifoldSelectionTrajectory:
    """Figure-emission tests for ``plot_manifold_selection_trajectory``."""

    @pytest.mark.filterwarnings("ignore:Tight layout:UserWarning")
    def test_writes_trajectory_svg_higher_is_better(self, tmp_path):
        """The default higher-is-better primary metric (``r2_spatial``)
        writes a single trajectory SVG named for the cohort + metric."""

        rng = np.random.default_rng(53)
        pkl = _write_manifold_selection_pickle(tmp_path, rng)
        out_dir = tmp_path / "mftraj_out"
        out_dir.mkdir()
        plot_manifold_selection_trajectory(
            selection_results_path=pkl,
            metric_primary='r2_spatial',
            metric_secondary='pearson_y',
            save_plot=True,
            output_dir=str(out_dir),
        )
        assert len(list(out_dir.glob("manifold_selection_trajectory_*_r2_spatial.svg"))) == 1

    @pytest.mark.filterwarnings("ignore:Tight layout:UserWarning")
    def test_writes_trajectory_svg_lower_is_better(self, tmp_path):
        """A lower-is-better error primary metric (``euclidean_mae``)
        exercises the inverted-axis branch and still writes one SVG."""

        rng = np.random.default_rng(54)
        pkl = _write_manifold_selection_pickle(tmp_path, rng)
        out_dir = tmp_path / "mftraj_out2"
        out_dir.mkdir()
        plot_manifold_selection_trajectory(
            selection_results_path=pkl,
            metric_primary='euclidean_mae',
            metric_secondary='pearson_y',
            save_plot=True,
            output_dir=str(out_dir),
        )
        assert len(list(out_dir.glob("manifold_selection_trajectory_*_euclidean_mae.svg"))) == 1


def _write_multinomial_multivariate_pickle(tmp_path, rng, n_classes: int = 3,
                                           n_features: int = 3,
                                           n_time: int = 12) -> str:
    """
    Write a synthetic consolidated multinomial ``selection_*.pkl`` whose
    final accepted step carries a pre-finalized ``weights_reshaped``
    tensor, for ``plot_multinomial_multivariate_filters``.

    The final step exposes the documented finalized-matrix schema:
    ``weights_reshaped`` (Folds × Classes × Features × Time — the plotter
    ``nanmean``-averages across the fold axis), ``final_model_features``
    (length-``n_features``), and ``classes``. A trailing rejection step is
    appended so the "last step that selected a feature" scan is exercised.
    Filename carries ``_male_`` + ``selection_`` so the loader / colour
    split work.

    Returns
    -------
    str
        Absolute path to the written pickle.
    """

    feats = ['self.speed', 'other.speed', 'nose-nose'][:n_features]
    classes = [f"cat{c}" for c in range(n_classes)]
    n_folds = 5
    accepted = {
        'selected_feature': feats[-1],
        'current_features': feats[:-1],
        'weights_reshaped': rng.standard_normal((n_folds, n_classes, n_features, n_time)),
        'final_model_features': feats,
        'classes': classes,
    }
    rejection = {'selected_feature': None, 'candidates_summary': {}}
    out = tmp_path / "selection_multinomial_male_mv.pkl"
    with out.open('wb') as fh:
        pickle.dump({'steps': [accepted, rejection]}, fh)
    return str(out)


@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive:UserWarning")
class TestPlotMultinomialMultivariateFilters:
    """Figure-emission tests for ``plot_multinomial_multivariate_filters``."""

    @pytest.mark.filterwarnings("ignore:Tight layout:UserWarning")
    def test_writes_final_filters_svg(self, tmp_path):
        """A finalized-matrix final step writes the per-feature weight-atlas
        SVG into the output directory."""

        rng = np.random.default_rng(57)
        pkl = _write_multinomial_multivariate_pickle(tmp_path, rng)
        out_dir = tmp_path / "mnmv_out"
        out_dir.mkdir()
        plot_multinomial_multivariate_filters(
            selection_results_path=pkl,
            save_plot=True,
            output_dir=str(out_dir),
        )
        assert len(list(out_dir.glob("model_selection_multinomial_usv_category_*_filters_final.svg"))) == 1


def _write_manifold_multivariate_pickle(tmp_path, rng, n_features: int = 2,
                                        n_time: int = 12, n_folds: int = 5) -> str:
    """
    Write a synthetic consolidated manifold ``selection_*.pkl`` whose
    final accepted step carries the raw bivariate weight block, for
    ``plot_manifold_multivariate_filters``.

    The final accepted step exposes ``selected_feature``,
    ``current_features`` (the non-anchor features), and
    ``candidates_summary[winner]['folds']['weights']`` shaped
    ``(n_folds, n_features * n_time, 2)`` — the 2 being the manifold
    output dimension. A trailing rejection step is appended.

    Returns
    -------
    str
        Absolute path to the written pickle.
    """

    feats = ['self.speed', 'other.speed', 'nose-nose'][:n_features]
    winner = feats[-1]
    current = feats[:-1]
    # `features` inside the plotter = current + [winner]; weight columns
    # must equal len(features) * n_time.
    n_total = len(feats) * n_time
    weights = rng.standard_normal((n_folds, n_total, 2))
    accepted = {
        'selected_feature': winner,
        'current_features': current,
        'candidates_summary': {
            winner: {'folds': {'weights': weights}},
        },
    }
    rejection = {'selected_feature': None, 'candidates_summary': {}}
    out = tmp_path / "selection_manifold_male_mv.pkl"
    with out.open('wb') as fh:
        pickle.dump({'steps': [accepted, rejection]}, fh)
    return str(out)


@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive:UserWarning")
class TestPlotManifoldMultivariateFilters:
    """Figure-emission tests for ``plot_manifold_multivariate_filters``."""

    @pytest.mark.filterwarnings("ignore:Tight layout:UserWarning")
    def test_writes_final_filters_svg(self, tmp_path):
        """The reshaped bivariate weight block renders a per-feature
        manifold-x / manifold-y filter atlas SVG."""

        rng = np.random.default_rng(59)
        pkl = _write_manifold_multivariate_pickle(tmp_path, rng)
        out_dir = tmp_path / "mfmv_out"
        out_dir.mkdir()
        plot_manifold_multivariate_filters(
            selection_results_path=pkl,
            save_plot=True,
            output_dir=str(out_dir),
        )
        assert len(list(out_dir.glob("model_selection_manifold_*_filters_final.svg"))) == 1


def _write_multinomial_diagnosis_pickle(tmp_path, rng, n_classes: int = 3,
                                        n_folds: int = 5,
                                        n_samples: int = 60) -> str:
    """
    Write a synthetic consolidated multinomial ``selection_*.pkl`` whose
    final accepted step carries pooled held-out predictions for
    ``plot_multinomial_selection_diagnosis``.

    The final step's ``candidates_summary[winner]['folds']`` holds the
    per-fold ``y_true`` (label arrays), ``y_probs`` (``(n_samples, K)``
    row-stochastic arrays), and ``confusion_matrix`` (``(K, K)``) lists the
    diagnosis reads, plus ``classes`` and ``final_model_features``. Every
    fold's confusion matrix is given a strong diagonal so per-class recall
    is well-defined; ``y_probs`` are normalised to sum to 1 per row so the
    pairwise binary-AUC computation has valid scores.

    Returns
    -------
    str
        Absolute path to the written pickle.
    """

    classes = [f"cat{c}" for c in range(n_classes)]
    feats = ['self.speed', 'other.speed']

    y_true_folds = []
    y_probs_folds = []
    cm_folds = []
    for _ in range(n_folds):
        y_true = rng.integers(0, n_classes, size=n_samples)
        probs = rng.uniform(0.1, 1.0, size=(n_samples, n_classes))
        # Nudge the true-class probability up so AUC scores are non-trivial.
        probs[np.arange(n_samples), y_true] += 1.0
        probs = probs / probs.sum(axis=1, keepdims=True)
        cm = np.zeros((n_classes, n_classes), dtype=float)
        for c in range(n_classes):
            cm[c, c] = rng.integers(8, 15)
            for d in range(n_classes):
                if d != c:
                    cm[c, d] = rng.integers(0, 4)
        y_true_folds.append(y_true)
        y_probs_folds.append(probs)
        cm_folds.append(cm)

    final_step = {
        'selected_feature': 'other.speed',
        'final_model_features': feats,
        'candidates_summary': {
            'other.speed': {
                'classes': classes,
                'folds': {
                    'y_true': y_true_folds,
                    'y_probs': y_probs_folds,
                    'confusion_matrix': cm_folds,
                },
            },
        },
    }
    out = tmp_path / "selection_multinomial_male_diag.pkl"
    with out.open('wb') as fh:
        pickle.dump({'steps': [final_step]}, fh)
    return str(out)


@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive:UserWarning")
class TestPlotMultinomialSelectionDiagnosis:
    """Figure-emission tests for ``plot_multinomial_selection_diagnosis``."""

    @pytest.mark.filterwarnings("ignore:Tight layout:UserWarning")
    def test_writes_auc_and_recall_figures(self, tmp_path):
        """Pooled held-out predictions yield both the pairwise-binary-AUC
        SVG and the per-class-recall SVG."""

        rng = np.random.default_rng(61)
        pkl = _write_multinomial_diagnosis_pickle(tmp_path, rng)
        out_dir = tmp_path / "mndiag_out"
        out_dir.mkdir()
        plot_multinomial_selection_diagnosis(
            selection_results_path=pkl,
            save_plot=True,
            output_dir=str(out_dir),
        )
        assert len(list(out_dir.glob("multinomial_pairwise_auc_*.svg"))) == 1
        assert len(list(out_dir.glob("multinomial_per_class_recall_*.svg"))) == 1

    @pytest.mark.filterwarnings("ignore:Tight layout:UserWarning")
    def test_label_overrides_accepted(self, tmp_path):
        """Presentation label overrides are threaded into the final-model
        title without error and both figures are still written."""

        rng = np.random.default_rng(62)
        pkl = _write_multinomial_diagnosis_pickle(tmp_path, rng)
        out_dir = tmp_path / "mndiag_lbl_out"
        out_dir.mkdir()
        plot_multinomial_selection_diagnosis(
            selection_results_path=pkl,
            save_plot=True,
            output_dir=str(out_dir),
            feature_label_overrides={'self.speed': 'Own speed'},
        )
        assert len(list(out_dir.glob("multinomial_pairwise_auc_*.svg"))) == 1

    def test_empty_prediction_pools_short_circuit(self, tmp_path):
        """A final step whose every fold carries empty ``y_true`` /
        ``y_probs`` arrays exercises the ``_concat_finite``-returns-None
        guard: the diagnosis returns without drawing either figure."""

        classes = ['cat0', 'cat1', 'cat2']
        n_folds = 4
        final_step = {
            'selected_feature': 'other.speed',
            'final_model_features': ['self.speed', 'other.speed'],
            'candidates_summary': {
                'other.speed': {
                    'classes': classes,
                    'folds': {
                        'y_true': [np.empty((0,), dtype=int) for _ in range(n_folds)],
                        'y_probs': [np.empty((0, 3)) for _ in range(n_folds)],
                        'confusion_matrix': [np.zeros((3, 3)) for _ in range(n_folds)],
                    },
                },
            },
        }
        pkl = tmp_path / "selection_multinomial_male_emptydiag.pkl"
        with pkl.open('wb') as fh:
            pickle.dump({'steps': [final_step]}, fh)
        out_dir = tmp_path / "mndiag_empty_out"
        out_dir.mkdir()
        plot_multinomial_selection_diagnosis(
            selection_results_path=str(pkl),
            save_plot=True,
            output_dir=str(out_dir),
        )
        assert list(out_dir.glob("*.svg")) == []


def _write_cnn_results_pickle(tmp_path, rng, n_folds: int = 8,
                              n_features: int = 5, n_bins: int = 20) -> str:
    """
    Write a synthetic CNN-manifold results pickle for ``DeepResultsVisualizer``.

    The artifact carries the two top-level blocks the lightweight
    interpretation methods read: ``metadata`` (``features_list``,
    ``n_time_bins``, ``save_dir``) and the per-method data blocks
    ``cross_validation`` (a list of fold dicts each with ``error_actual``
    and ``error_null_model_free`` plus the per-sample manifold arrays
    ``Y_true`` / ``Y_pred_actual`` / ``Y_pred_null_model_free`` (each
    ``(n_points, 2)``) consumed by the spatial-error methods) and
    ``feature_importance`` (``ranked_features`` / ``means`` / ``stds``,
    used by ``plot_feature_importance``). The actual errors / predictions
    are placed closer to ground truth than the null-free ones so the model
    reads as skillful. The filename carries ``male`` so the ``__init__``
    sex/colour detection picks the male colour.

    Returns
    -------
    str
        Absolute path to the written pickle.
    """

    features = [f"self.feat{i}" for i in range(n_features)]
    cv_folds = []
    n_points = 80
    for _ in range(n_folds):
        null_err = rng.uniform(2.0, 4.0)
        act_err = null_err * rng.uniform(0.4, 0.7)
        # 2-D manifold ground truth + predictions: the actual model sits
        # nearer ground truth (small jitter) than the null model (large
        # jitter), so the error-landscape skill contrast is non-trivial.
        y_true = rng.uniform(-5.0, 5.0, size=(n_points, 2))
        y_pred_actual = y_true + rng.normal(0.0, 0.3, size=(n_points, 2))
        y_pred_null = y_true + rng.normal(0.0, 1.5, size=(n_points, 2))
        cv_folds.append({
            'error_actual': float(act_err),
            'error_null_model_free': float(null_err),
            'Y_true': y_true,
            'Y_pred_actual': y_pred_actual,
            'Y_pred_null_model_free': y_pred_null,
        })

    means = {f: float(rng.uniform(0.0, 0.5)) for f in features}
    stds = {f: float(rng.uniform(0.02, 0.1)) for f in features}
    # Include the knockoff probe so the explicit knockoff-filter branch runs.
    means['knockoff_probe'] = 0.01
    stds['knockoff_probe'] = 0.01
    ranked = sorted(features, key=lambda f: means[f], reverse=True) + ['knockoff_probe']

    # Pre-computed saliency entry for one circular region centred on the
    # data cloud (so a generous radius captures >= 3 in-region samples).
    # ``contrastive_saliency`` is (n_avg, n_features, n_bins); the plotter
    # averages over axis 0 to get the (features × time) heatmap.
    saliency_maps = {
        'region_0': {
            'centroid': np.array([0.0, 0.0]),
            'radius': 8.0,
            'contrastive_saliency': rng.standard_normal((4, n_features, n_bins)),
        },
    }

    data = {
        'metadata': {
            'features_list': features,
            'n_time_bins': n_bins,
            'save_dir': str(tmp_path / "cnn_default_save"),
        },
        'cross_validation': cv_folds,
        'feature_importance': {
            'ranked_features': ranked,
            'means': means,
            'stds': stds,
            'best_fold_idx': 0,
        },
        'saliency_maps': saliency_maps,
    }
    out = tmp_path / "cnn_manifold_male_results.pkl"
    with out.open('wb') as fh:
        pickle.dump(data, fh)
    return str(out)


@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive:UserWarning")
class TestDeepResultsVisualizer:
    """Figure-emission tests for the lightweight ``DeepResultsVisualizer``
    interpretation methods (permutation test + feature importance)."""

    @pytest.mark.filterwarnings("ignore:Tight layout:UserWarning")
    def test_permutation_test_writes_svg(self, tmp_path):
        """The bootstrapped permutation-test broken-axis figure writes a
        single SVG into the override output directory."""

        rng = np.random.default_rng(67)
        pkl = _write_cnn_results_pickle(tmp_path, rng)
        viz = DeepResultsVisualizer(
            results_pkl_path=pkl,
            modeling_settings={"model_params": {"random_seed": 0}},
            visualization_settings={},
        )
        out_dir = tmp_path / "cnn_perm_out"
        out_dir.mkdir()
        viz.plot_permutation_test(
            n_bootstraps=50,
            save_plot=True,
            output_dir=str(out_dir),
        )
        assert len(list(out_dir.glob("cnn_permutation_test_*.svg"))) == 1

    @pytest.mark.filterwarnings("ignore:Tight layout:UserWarning")
    def test_feature_importance_writes_svg(self, tmp_path):
        """The SNR-weighted global feature-importance bar figure writes a
        single SVG into the override output directory."""

        rng = np.random.default_rng(68)
        pkl = _write_cnn_results_pickle(tmp_path, rng)
        viz = DeepResultsVisualizer(
            results_pkl_path=pkl,
            modeling_settings={},
            visualization_settings={},
        )
        out_dir = tmp_path / "cnn_imp_out"
        out_dir.mkdir()
        viz.plot_feature_importance(
            save_plot=True,
            output_dir=str(out_dir),
        )
        assert len(list(out_dir.glob("cnn_feature_importance_*.svg"))) == 1

    @pytest.mark.filterwarnings("ignore:Tight layout:UserWarning")
    def test_error_landscape_writes_svg(self, tmp_path):
        """The two-panel hexbin error-landscape / error-reduction figure
        writes a single SVG (uses the per-fold manifold prediction arrays
        and the wrap-aware ``pairwise_distance`` helper)."""

        rng = np.random.default_rng(69)
        pkl = _write_cnn_results_pickle(tmp_path, rng)
        viz = DeepResultsVisualizer(
            results_pkl_path=pkl,
            modeling_settings={},
            visualization_settings={},
        )
        out_dir = tmp_path / "cnn_land_out"
        out_dir.mkdir()
        viz.plot_error_landscape(
            gridsize=10,
            save_plot=True,
            output_dir=str(out_dir),
        )
        assert len(list(out_dir.glob("cnn_error_landscape_*.svg"))) == 1

    @pytest.mark.filterwarnings("ignore:Tight layout:UserWarning")
    @pytest.mark.filterwarnings("ignore:Glyph:UserWarning")
    @pytest.mark.filterwarnings(
        "ignore:The get_cmap function was deprecated"
        ":matplotlib._api.deprecation.MatplotlibDeprecationWarning"
    )
    def test_spatial_precision_grid_density_writes_svg(self, tmp_path):
        """The tiled per-region density precision grid (KMeans-guided patch
        selection + per-patch prediction KDE) writes a single SVG."""

        rng = np.random.default_rng(70)
        pkl = _write_cnn_results_pickle(tmp_path, rng)
        viz = DeepResultsVisualizer(
            results_pkl_path=pkl,
            modeling_settings={},
            visualization_settings={},
        )
        out_dir = tmp_path / "cnn_grid_out"
        out_dir.mkdir()
        viz.plot_spatial_precision_grid(
            plot_type='density',
            n_patches=4,
            patch_size=8.0,
            min_samples=5,
            save_plot=True,
            output_dir=str(out_dir),
        )
        assert len(list(out_dir.glob("cnn_precision_grid_*_density.svg"))) == 1

    @pytest.mark.filterwarnings("ignore:Tight layout:UserWarning")
    @pytest.mark.filterwarnings("ignore:Glyph:UserWarning")
    @pytest.mark.filterwarnings(
        "ignore:The get_cmap function was deprecated"
        ":matplotlib._api.deprecation.MatplotlibDeprecationWarning"
    )
    def test_regional_saliency_inset_writes_svg(self, tmp_path):
        """The two-panel regional saliency inset (manifold-context KDE +
        contrastive saliency heatmap with error inset) writes a single SVG
        for a stored circular saliency region."""

        rng = np.random.default_rng(71)
        pkl = _write_cnn_results_pickle(tmp_path, rng)
        viz = DeepResultsVisualizer(
            results_pkl_path=pkl,
            modeling_settings={},
            visualization_settings={},
        )
        out_dir = tmp_path / "cnn_sal_out"
        out_dir.mkdir()
        viz.plot_regional_saliency_inset(
            region_key='region_0',
            prediction_plot_type='contour',
            save_plot=True,
            output_dir=str(out_dir),
        )
        assert len(list(out_dir.glob("cnn_regional_saliency_region_0_*.svg"))) == 1

    @pytest.mark.filterwarnings("ignore:Tight layout:UserWarning")
    @pytest.mark.filterwarnings("ignore:Glyph:UserWarning")
    @pytest.mark.filterwarnings(
        "ignore:The get_cmap function was deprecated"
        ":matplotlib._api.deprecation.MatplotlibDeprecationWarning"
    )
    @pytest.mark.parametrize("prediction_plot_type", ['density', 'hexbin', 'scatter'])
    def test_regional_saliency_inset_prediction_plot_types(
            self, tmp_path, prediction_plot_type):
        """The alternate ``prediction_plot_type`` branches (the ``density``
        KDE-image branch, the ``hexbin`` branch, and the catch-all
        ``scatter`` fallback) each render and write a single SVG.

        The default ``contour`` path is exercised by
        ``test_regional_saliency_inset_writes_svg`` above; this parametrised
        case fills in the remaining three prediction-overlay code paths."""

        rng = np.random.default_rng(72)
        pkl = _write_cnn_results_pickle(tmp_path, rng)
        viz = DeepResultsVisualizer(
            results_pkl_path=pkl,
            modeling_settings={},
            visualization_settings={},
        )
        out_dir = tmp_path / f"cnn_sal_{prediction_plot_type}_out"
        out_dir.mkdir()
        viz.plot_regional_saliency_inset(
            region_key='region_0',
            prediction_plot_type=prediction_plot_type,
            save_plot=True,
            output_dir=str(out_dir),
        )
        assert len(list(out_dir.glob("cnn_regional_saliency_region_0_*.svg"))) == 1

    @pytest.mark.filterwarnings("ignore:Tight layout:UserWarning")
    @pytest.mark.filterwarnings("ignore:Glyph:UserWarning")
    @pytest.mark.filterwarnings(
        "ignore:The get_cmap function was deprecated"
        ":matplotlib._api.deprecation.MatplotlibDeprecationWarning"
    )
    def test_spatial_precision_grid_euclidean_nn_fallback(self, tmp_path):
        """A precision grid with a ``min_samples`` threshold no K-means
        centre patch can meet drives every centre through the wrap-aware
        nearest-neighbour snap-back branch. The ``patch_size`` is kept wide
        enough that the patch re-centred on the snapped data point still
        captures multiple points (so the per-patch KDE is non-singular)."""

        rng = np.random.default_rng(73)
        pkl = _write_cnn_results_pickle(tmp_path, rng)
        viz = DeepResultsVisualizer(
            results_pkl_path=pkl,
            modeling_settings={},
            visualization_settings={},
        )
        out_dir = tmp_path / "cnn_grid_nn_out"
        out_dir.mkdir()
        viz.plot_spatial_precision_grid(
            plot_type='density',
            n_patches=4,
            patch_size=4.0,
            # An 80-point cloud spread over ~[-5, 5]^2 has no 4.0-side
            # patch holding 500 samples, so every K-means centre falls
            # into the nearest-neighbour fallback; the re-centred 4.0-side
            # patch around a real point still captures several neighbours.
            min_samples=500,
            save_plot=True,
            output_dir=str(out_dir),
        )
        assert len(list(out_dir.glob("cnn_precision_grid_*_density.svg"))) == 1


def _write_cnn_torus_results_pickle(tmp_path, rng, n_folds: int = 6,
                                    n_features: int = 5, n_bins: int = 20,
                                    period: float = 6.2831853) -> str:
    """
    Write a synthetic torus-manifold CNN results pickle for
    ``DeepResultsVisualizer``.

    Identical in spirit to ``_write_cnn_results_pickle`` but tags
    ``metadata`` with ``manifold_metric='torus'`` and a positive
    ``manifold_period`` so every distance computation routes through the
    wrap-aware ``pairwise_distance(..., metric='torus', period=...)``
    path. All 2-D manifold coordinates (``Y_true`` / ``Y_pred_actual`` /
    ``Y_pred_null_model_free``) are sampled inside the unit cell
    ``[0, period)^2`` so the wrap-aware masks, the tiled KDE, and the
    circular-mean bias all operate on in-cell data. The actual model's
    predictions are placed nearer ground truth than the null model's so
    the torus error-landscape skill contrast is non-trivial. The
    saliency region is centred mid-cell with a radius wide enough to
    capture >= 3 in-region samples. The filename carries ``male`` for the
    sex/colour detection.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    rng : numpy.random.Generator
        Seeded generator for reproducible synthetic values.
    n_folds : int, default 6
        Cross-validation fold count.
    n_features : int, default 5
        Predictor-feature count.
    n_bins : int, default 20
        Number of filter time bins.
    period : float, default 2*pi
        Positive per-axis period of the torus manifold.

    Returns
    -------
    str
        Absolute path to the written pickle (filename contains ``male``
        and ``torus``).
    """

    features = [f"self.feat{i}" for i in range(n_features)]
    cv_folds = []
    n_points = 120
    for _ in range(n_folds):
        null_err = rng.uniform(2.0, 4.0)
        act_err = null_err * rng.uniform(0.4, 0.7)
        # In-cell ground truth on the torus; predictions wrapped back
        # into [0, period) after adding metric-appropriate jitter.
        y_true = rng.uniform(0.0, period, size=(n_points, 2))
        y_pred_actual = (y_true + rng.normal(0.0, 0.3, size=(n_points, 2))) % period
        y_pred_null = (y_true + rng.normal(0.0, 1.5, size=(n_points, 2))) % period
        cv_folds.append({
            'error_actual': float(act_err),
            'error_null_model_free': float(null_err),
            'Y_true': y_true,
            'Y_pred_actual': y_pred_actual,
            'Y_pred_null_model_free': y_pred_null,
        })

    means = {f: float(rng.uniform(0.0, 0.5)) for f in features}
    stds = {f: float(rng.uniform(0.02, 0.1)) for f in features}
    means['knockoff_probe'] = 0.01
    stds['knockoff_probe'] = 0.01
    ranked = sorted(features, key=lambda f: means[f], reverse=True) + ['knockoff_probe']

    # Region centred mid-cell with a generous radius so the in-region
    # mask (wrap-aware) captures plenty of samples.
    saliency_maps = {
        'region_0': {
            'centroid': np.array([period / 2.0, period / 2.0]),
            'radius': period / 3.0,
            'contrastive_saliency': rng.standard_normal((4, n_features, n_bins)),
        },
    }

    data = {
        'metadata': {
            'features_list': features,
            'n_time_bins': n_bins,
            'save_dir': str(tmp_path / "cnn_torus_default_save"),
            'manifold_metric': 'torus',
            'manifold_period': float(period),
        },
        'cross_validation': cv_folds,
        'feature_importance': {
            'ranked_features': ranked,
            'means': means,
            'stds': stds,
            'best_fold_idx': 0,
        },
        'saliency_maps': saliency_maps,
    }
    out = tmp_path / "cnn_manifold_torus_male_results.pkl"
    with out.open('wb') as fh:
        pickle.dump(data, fh)
    return str(out)


@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive:UserWarning")
class TestDeepResultsVisualizerTorus:
    """Figure-emission tests for the torus-manifold (QLVM) branches of the
    ``DeepResultsVisualizer`` spatial methods.

    These mirror the Euclidean ``DeepResultsVisualizer`` tests but feed a
    pickle whose ``metadata`` declares ``manifold_metric='torus'`` so the
    wrap-aware code paths -- the ``_plot_spatial_precision_grid_torus``
    private renderer, the 9-tiled circle / unit-cell axis branches of the
    regional-saliency inset, and the ``QLVM`` dim-prefix branch of the
    error landscape -- all execute."""

    @pytest.mark.filterwarnings("ignore:Tight layout:UserWarning")
    @pytest.mark.filterwarnings("ignore:Glyph:UserWarning")
    @pytest.mark.filterwarnings(
        "ignore:The get_cmap function was deprecated"
        ":matplotlib._api.deprecation.MatplotlibDeprecationWarning"
    )
    @pytest.mark.parametrize("plot_type", ['density', 'contour'])
    def test_spatial_precision_grid_torus_writes_svg(self, tmp_path, plot_type):
        """The torus precision grid renders a uniform wrap-aware patch grid
        (tiled KDE + circular-mean bias) and writes one SVG per
        ``plot_type`` (the ``density`` imshow and ``contour`` styles)."""

        rng = np.random.default_rng(74)
        pkl = _write_cnn_torus_results_pickle(tmp_path, rng)
        viz = DeepResultsVisualizer(
            results_pkl_path=pkl,
            modeling_settings={},
            visualization_settings={},
        )
        out_dir = tmp_path / f"cnn_torus_grid_{plot_type}_out"
        out_dir.mkdir()
        viz.plot_spatial_precision_grid(
            plot_type=plot_type,
            grid_shape=(3, 3),
            min_samples=3,
            save_plot=True,
            output_dir=str(out_dir),
        )
        assert len(list(out_dir.glob(f"cnn_precision_grid_*_torus_{plot_type}.svg"))) == 1

    @pytest.mark.filterwarnings("ignore:Tight layout:UserWarning")
    @pytest.mark.filterwarnings("ignore:Glyph:UserWarning")
    @pytest.mark.filterwarnings(
        "ignore:The get_cmap function was deprecated"
        ":matplotlib._api.deprecation.MatplotlibDeprecationWarning"
    )
    def test_spatial_precision_grid_torus_sparse_patch_branch(self, tmp_path):
        """A ``min_samples`` threshold no patch can clear drives every torus
        panel through the ``only N pts`` short-circuit branch (the patch is
        outlined but no KDE is fit) and still writes a single SVG."""

        rng = np.random.default_rng(75)
        pkl = _write_cnn_torus_results_pickle(tmp_path, rng)
        viz = DeepResultsVisualizer(
            results_pkl_path=pkl,
            modeling_settings={},
            visualization_settings={},
        )
        out_dir = tmp_path / "cnn_torus_grid_sparse_out"
        out_dir.mkdir()
        viz.plot_spatial_precision_grid(
            plot_type='density',
            grid_shape=(3, 3),
            patch_size=0.05,  # tiny patch -> almost no in-patch samples
            min_samples=10_000,
            save_plot=True,
            output_dir=str(out_dir),
        )
        assert len(list(out_dir.glob("cnn_precision_grid_*_torus_density.svg"))) == 1

    @pytest.mark.filterwarnings("ignore:Tight layout:UserWarning")
    def test_error_landscape_torus_writes_svg(self, tmp_path):
        """The two-panel hexbin error landscape on a torus manifold uses the
        wrap-aware per-sample error magnitudes and the ``QLVM`` axis-label
        branch, and writes a single SVG."""

        rng = np.random.default_rng(76)
        pkl = _write_cnn_torus_results_pickle(tmp_path, rng)
        viz = DeepResultsVisualizer(
            results_pkl_path=pkl,
            modeling_settings={},
            visualization_settings={},
        )
        out_dir = tmp_path / "cnn_torus_land_out"
        out_dir.mkdir()
        viz.plot_error_landscape(
            gridsize=10,
            save_plot=True,
            output_dir=str(out_dir),
        )
        assert len(list(out_dir.glob("cnn_error_landscape_*.svg"))) == 1

    @pytest.mark.filterwarnings("ignore:Tight layout:UserWarning")
    @pytest.mark.filterwarnings("ignore:Glyph:UserWarning")
    @pytest.mark.filterwarnings(
        "ignore:The get_cmap function was deprecated"
        ":matplotlib._api.deprecation.MatplotlibDeprecationWarning"
    )
    @pytest.mark.parametrize("prediction_plot_type", ['contour', 'density'])
    def test_regional_saliency_inset_torus_writes_svg(
            self, tmp_path, prediction_plot_type):
        """The regional-saliency inset on a torus manifold draws the 9-tiled
        wrapped region circle, fixes the axes to the unit cell, labels the
        axes ``QLVM Dimension N``, and writes a single SVG."""

        rng = np.random.default_rng(77)
        pkl = _write_cnn_torus_results_pickle(tmp_path, rng)
        viz = DeepResultsVisualizer(
            results_pkl_path=pkl,
            modeling_settings={},
            visualization_settings={},
        )
        out_dir = tmp_path / f"cnn_torus_sal_{prediction_plot_type}_out"
        out_dir.mkdir()
        viz.plot_regional_saliency_inset(
            region_key='region_0',
            prediction_plot_type=prediction_plot_type,
            save_plot=True,
            output_dir=str(out_dir),
        )
        assert len(list(out_dir.glob("cnn_regional_saliency_region_0_*.svg"))) == 1
