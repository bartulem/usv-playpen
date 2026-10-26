"""
@author: bartulem
Coverage: the cohort-scale behaviour control -- its embedding, its acceptance rule, and its guards.

Every regression test here pins a failure that actually occurred. The control was refitted per unit on
~2,000 USVs against thousands of columns and scored BELOW a constant fitted on the same USVs, so the added
score it produced was the neuron measured against nothing; the forward selection was written from prose
rather than from the fixed implementation and reproduced a known defect, selecting one feature while an
unselected set scored better; and a control silently out of step with its settings would change every
added score without changing a single reported configuration value.

The cohort-scale stages are covered by KNOWN-ANSWER tests on a tiny synthetic cohort (four recording days,
forty USVs each, three features at two lags) in which one feature ("signal") drives the first torus
coordinate and the other two are pure noise: the day holdout must never let a day into its own training
set, the fitted control must recover the planted coefficient and beat the no-behaviour baseline, the
forward selection must pick the planted feature and nothing when nothing is planted, and the cohort
assembly must stack sessions in order, record each USV's day and refuse an empty or inconsistent cohort.
The process pools are replaced by an in-process serial executor so every worker body runs (and is
measured) inside the test process.
"""

from __future__ import annotations

import json
import pathlib
import pickle
import re
from typing import ClassVar

import h5py
import numpy as np
import polars as pls
import pytest

from usv_playpen.modeling.manifold_metric import (
    inverse_region_frequency_weights,
    macro_von_mises_logscore,
)
from usv_playpen.modeling.modeling_metadata import compute_settings_sha256
from usv_playpen.neural_modeling import behaviour_control_model as control_module
from usv_playpen.neural_modeling.behaviour_control_model import (
    EMBEDDING_COLUMNS,
    apply_behaviour_control,
    assemble_cohort,
    control_estimator,
    control_predictions,
    feature_columns,
    fit_control_models,
    held_out_by_day,
    load_control_model,
    paired_gain,
    pooled_macro,
    select_control_features,
    session_vocalization_design,
    torus_embedding,
    upgrade_legacy_counts,
    write_control_model,
)
from usv_playpen.neural_modeling.neural_design_assembly import (
    onset_history_anchor_frames,
)
from usv_playpen.neural_modeling.neural_nested_decoding import reduced_model_features

SETTINGS_PATH = (pathlib.Path(__file__).resolve().parents[2] / "src" / "usv_playpen"
                 / "_parameter_settings" / "neural_modeling_settings.json")


@pytest.fixture
def settings() -> dict:
    """The shipped settings, read fresh so a test cannot mutate another's copy."""
    with SETTINGS_PATH.open() as handle:
        return json.load(handle)


@pytest.fixture
def artifact() -> dict:
    """A two-feature, five-lag control with one held-out day."""
    generator = np.random.default_rng(0)
    return {"features": ["nose-nose", "other.speed"], "n_lags": 5, "settings_sha256": "abc123",
            "coefficients": {"20250919": {"coef": generator.normal(size=(10, 4)),
                                          "intercept": np.zeros(4)}}}


SYNTHETIC_FEATURES = ["signal", "noise_a", "noise_b"]
SYNTHETIC_DAYS = ["20250101", "20250102", "20250103", "20250104"]
SYNTHETIC_N_LAGS = 2


class SerialExecutor:
    """
    Description
    -----------
    An in-process stand-in for ``concurrent.futures.ProcessPoolExecutor``.

    The cohort stages map their per-session and per-candidate work over a SPAWN-context process pool.
    Spawning re-imports the package in every child, which costs seconds per test and, worse, runs the
    worker bodies (``_assemble_one``, ``_score_candidate``) outside the measured process, so coverage
    never sees them. This executor runs ``map`` serially in the calling process, preserving the input
    order exactly as ``ProcessPoolExecutor.map`` does, and records the constructor arguments so a test
    can check that ``n_workers`` and the pool context were actually passed through.

    Parameters
    ----------
    max_workers (int)
        The worker count the caller asked for; recorded in ``SerialExecutor.requested_workers``.
    mp_context (object)
        The multiprocessing context the caller passed; recorded in ``SerialExecutor.contexts``.

    Returns
    -------
    executor (SerialExecutor)
        A context manager whose ``map`` returns a list of results in input order.
    """

    requested_workers: ClassVar[list] = []
    contexts: ClassVar[list] = []

    def __init__(self, max_workers: int, mp_context: object) -> None:
        """
        Description
        -----------
        Record the requested worker count and context on the class, so the test can read them after the
        executor has gone out of scope.

        Parameters
        ----------
        max_workers (int)
            The worker count the caller asked for.
        mp_context (object)
            The multiprocessing context the caller passed.

        Returns
        -------
        None
        """
        SerialExecutor.requested_workers.append(max_workers)
        SerialExecutor.contexts.append(mp_context)

    def __enter__(self) -> SerialExecutor:
        """
        Description
        -----------
        Enter the ``with`` block.

        Parameters
        ----------
        None

        Returns
        -------
        executor (SerialExecutor)
            This instance.
        """
        return self

    def __exit__(self, *_exc_info: object) -> bool:
        """
        Description
        -----------
        Leave the ``with`` block without suppressing any exception raised inside it.

        Parameters
        ----------
        *_exc_info (object)
            The exception type, value and traceback (all None on a clean exit); unused.

        Returns
        -------
        suppress (bool)
            Always False, so exceptions propagate exactly as they would from a real pool.
        """
        return False

    def map(self, function, iterable) -> list:
        """
        Description
        -----------
        Apply ``function`` to every item of ``iterable`` in order, in this process.

        Parameters
        ----------
        function (callable)
            The worker body.
        iterable (Iterable)
            The task arguments.

        Returns
        -------
        results (list)
            ``[function(item) for item in iterable]``.
        """
        return [function(item) for item in iterable]


@pytest.fixture
def serial_pools(monkeypatch) -> type:
    """
    Description
    -----------
    Replace the module's process pool with :class:`SerialExecutor` and its pool context with a sentinel
    string, clearing the executor's recorded arguments first so each test reads only its own pools.

    Parameters
    ----------
    monkeypatch (pytest.MonkeyPatch)
        Patches ``ProcessPoolExecutor`` and ``worker_process_context`` on the module under test.

    Returns
    -------
    executor_class (type)
        :class:`SerialExecutor`, whose ``requested_workers`` and ``contexts`` lists the test may inspect.
    """
    SerialExecutor.requested_workers = []
    SerialExecutor.contexts = []
    monkeypatch.setattr(control_module, "ProcessPoolExecutor", SerialExecutor)
    monkeypatch.setattr(control_module, "worker_process_context", lambda: "serial-context")
    return SerialExecutor


def synthetic_cohort(seed: int, planted: bool, per_day: int = 40) -> dict:
    """
    Description
    -----------
    A tiny cohort, in the exact shape ``assemble_cohort`` returns, with a KNOWN behaviour-to-position map.

    Each USV carries three features (``signal``, ``noise_a``, ``noise_b``) drawn uniformly on [-1, 1];
    each feature's value is repeated over ``SYNTHETIC_N_LAGS`` lag columns (feature-major, the layout
    ``feature_columns`` indexes) with small independent jitter (sd 0.05). When ``planted`` is True the
    first torus coordinate is ``0.25 + 0.15 * signal`` plus noise of sd 0.02, so ``signal`` drives
    position and the two noise features carry nothing; when False the first coordinate is ``0.25`` plus
    the same noise and no feature carries anything. The second coordinate is ``0.5`` plus noise of sd
    0.02 in both cases. All USVs share one acoustic region, and the days are ``SYNTHETIC_DAYS`` with
    ``per_day`` USVs each, enough to clear ``min_region_events`` (20) inside every day.

    Parameters
    ----------
    seed (int)
        Seed of the ``numpy`` generator, so every draw is reproducible.
    planted (bool)
        Whether ``signal`` drives the first torus coordinate.
    per_day (int)
        USVs per recording day.

    Returns
    -------
    cohort (dict)
        ``design`` (float32, ``(n, 3 * n_lags)``), ``positions`` (``(n, 2)`` in [0, 1)), ``regions``,
        ``days``, ``session_ids``, ``feature_names``, ``skipped`` (empty), plus ``signal`` (the
        undisturbed per-USV signal value, for building expected answers).
    """
    generator = np.random.default_rng(seed)
    n_rows = len(SYNTHETIC_DAYS) * per_day
    base = generator.uniform(-1.0, 1.0, (n_rows, len(SYNTHETIC_FEATURES)))
    design = np.repeat(base, SYNTHETIC_N_LAGS, axis=1) + generator.normal(
        0.0, 0.05, (n_rows, len(SYNTHETIC_FEATURES) * SYNTHETIC_N_LAGS)
    )
    drive = 0.15 * base[:, 0] if planted else 0.0
    first = (0.25 + drive + generator.normal(0.0, 0.02, n_rows)) % 1.0
    second = (0.5 + generator.normal(0.0, 0.02, n_rows)) % 1.0
    return {
        "design": design.astype(np.float32),
        "positions": np.column_stack([first, second]),
        "regions": np.ones(n_rows),
        "days": np.repeat(SYNTHETIC_DAYS, per_day),
        "session_ids": [f"{day}_120000" for day in SYNTHETIC_DAYS],
        "feature_names": list(SYNTHETIC_FEATURES),
        "skipped": {"20250105_120000": "3 tracked animals"},
        "signal": base[:, 0],
    }


def circular_error(predicted: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """
    Description
    -----------
    Wrap-aware absolute error per USV on the unit torus, taking the shorter way round on each axis and
    returning the larger of the two axis errors.

    Parameters
    ----------
    predicted (np.ndarray)
        ``(n, 2)`` predicted torus coordinates in [0, 1).
    truth (np.ndarray)
        ``(n, 2)`` true torus coordinates in [0, 1).

    Returns
    -------
    error (np.ndarray)
        ``(n,)`` per-USV error in torus units, in [0, 0.5].
    """
    difference = np.abs(predicted - truth) % 1.0
    return np.minimum(difference, 1.0 - difference).max(axis=1)


def two_region_cohort(seed: int) -> dict:
    """
    Description
    -----------
    The planted synthetic cohort with TWO acoustic regions of unequal size and unequal predictability, so
    that every place a region label is consumed (the inverse-region-frequency fit weights and the macro
    score's equal-weight average over regions) changes the answer if the labels are dropped.

    Each of the four days carries 80 USVs: the first 55 are region 1 and the last 25 region 2, so both
    regions clear ``min_region_events`` (20) inside every day while region 1 outnumbers region 2 by more
    than two to one (the fit weights are therefore not uniform). Region 2's first torus coordinate gets
    extra noise of sd 0.08 on top of the planted map, so the two regions are predicted with different
    accuracy and the macro (equal-weight) score differs from the pooled (per-USV) score.

    Parameters
    ----------
    seed (int)
        Seed of both the base cohort and the extra region-2 noise.

    Returns
    -------
    cohort (dict)
        As :func:`synthetic_cohort` with ``per_day=80``, its ``regions`` replaced by the two labels and
        its region-2 positions perturbed.
    """
    cohort = synthetic_cohort(seed=seed, planted=True, per_day=80)
    generator = np.random.default_rng(seed + 100)
    regions = np.tile(np.r_[np.ones(55), np.full(25, 2.0)], len(SYNTHETIC_DAYS))
    positions = cohort["positions"].copy()
    rare = regions == 2.0
    positions[rare, 0] = (positions[rare, 0] + generator.normal(0.0, 0.08, int(rare.sum()))) % 1.0
    cohort["regions"] = regions
    cohort["positions"] = positions
    return cohort


def macro_reference(predicted: np.ndarray, positions: np.ndarray, regions: np.ndarray | None) -> float:
    """
    Description
    -----------
    The macro von Mises log-score computed straight from ``manifold_metric`` with the shipped torus
    period (1.0) and minimum region size (20), independently of the module's ``pooled_macro`` wrapper,
    so a test can state the score a correctly wired call must produce.

    Parameters
    ----------
    predicted (np.ndarray)
        ``(n, 2)`` predicted torus positions.
    positions (np.ndarray)
        ``(n, 2)`` true torus positions.
    regions (np.ndarray | None)
        Region label per USV, or None for the single-region (pooled per-USV) score.

    Returns
    -------
    score (float)
        The macro (or, with ``regions=None``, pooled) von Mises log-score.
    """
    return float(macro_von_mises_logscore(predicted, positions, regions, metric="torus", period=1.0,
                                          min_region_events=20))



class TestTorusEmbedding:
    """The predicted POSITION has to become a predictor without a discontinuity at the wrap."""

    def test_pairs_are_unit_vectors(self) -> None:
        """Each dimension contributes a cosine/sine pair, so each pair has unit norm."""
        generator = np.random.default_rng(1)
        embedded = torus_embedding(generator.random((50, 2)))
        assert embedded.shape == (50, 4)
        assert np.allclose(embedded[:, 0] ** 2 + embedded[:, 1] ** 2, 1.0)
        assert np.allclose(embedded[:, 2] ** 2 + embedded[:, 3] ** 2, 1.0)

    def test_the_wrap_is_continuous(self) -> None:
        """0.99 and 0.01 are neighbours on a torus; a raw coordinate would make them opposites."""
        near_wrap = torus_embedding(np.array([[0.99, 0.5], [0.01, 0.5]]))
        far_apart = torus_embedding(np.array([[0.25, 0.5], [0.75, 0.5]]))
        assert np.linalg.norm(near_wrap[0] - near_wrap[1]) < np.linalg.norm(far_apart[0] - far_apart[1])

    def test_known_coordinates_map_to_known_angles(self) -> None:
        """
        Description
        -----------
        A coordinate ``c`` on the unit period is the angle ``2 * pi * c``, so (0.25, 0.0) must embed as
        ``(cos 90deg, sin 90deg, cos 0, sin 0) = (0, 1, 1, 0)`` and (0.5, 0.125) as
        ``(-1, 0, cos 45deg, sin 45deg)``. The unit-norm and wrap tests above hold for ANY angle map, so
        only exact values pin the scaling.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        embedded = torus_embedding(np.array([[0.25, 0.0], [0.5, 0.125]]))
        half = np.sqrt(0.5)
        np.testing.assert_allclose(embedded, [[0.0, 1.0, 1.0, 0.0], [-1.0, 0.0, half, half]], atol=1e-12)


class TestPairedGain:
    """The acceptance rule's pairing, which this project has now lost TWICE."""

    def test_pairing_survives_a_between_day_spread_that_swamps_the_gain(self) -> None:
        """Days differ by ~0.25 in absolute score while a step gains ~0.013.

        Comparing the gain to the UNPAIRED standard error of the raw scores therefore rejects every
        feature after the first, and the selected model then scores worse than an unselected one --
        which is exactly how the defect was spotted. Differencing within day removes the day effect.
        """
        days = np.array([-3.42, -3.50, -3.55, -3.60, -3.67, -3.45, -3.52, -3.58, -3.63])
        candidate = days + 0.013
        unpaired = float(np.std(days, ddof=1) / np.sqrt(days.size))
        gain, error = paired_gain(candidate, days)
        assert unpaired > 0.013, "the between-day spread must swamp the gain, or the test proves nothing"
        assert gain == pytest.approx(0.013)
        assert (gain - error) > 0.0

    def test_a_genuinely_noisy_candidate_is_rejected(self) -> None:
        """Pairing must not accept everything: an inconsistent improvement still fails."""
        generator = np.random.default_rng(2)
        days = np.linspace(-3.67, -3.42, 9)
        gain, error = paired_gain(days + generator.normal(0.0, 0.05, days.size), days)
        assert (gain - error) <= 0.0

    def test_all_nan_differences_do_not_raise(self) -> None:
        """A candidate that scored NaN everywhere returns NaN rather than exploding."""
        gain, error = paired_gain(np.full(4, np.nan), np.zeros(4))
        assert np.isnan(gain)
        assert np.isnan(error)

    def test_a_single_finite_day_returns_its_difference_with_zero_error(self) -> None:
        """
        Description
        -----------
        When only one day yields a finite difference, the gain is that one difference and the standard
        error is defined as 0.0 (a sample standard deviation of one value would be NaN under ``ddof=1``).
        Here three of four days are NaN and the surviving day improves by exactly 0.2.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        candidate = np.array([np.nan, -3.3, np.nan, np.nan])
        incumbent = np.array([-3.4, -3.5, np.nan, -3.6])
        gain, error = paired_gain(candidate, incumbent)
        assert gain == pytest.approx(0.2)
        assert error == 0.0

    def test_the_error_is_the_sample_standard_error_of_the_paired_differences(self) -> None:
        """
        Description
        -----------
        Known answers for the standard error ``std(d, ddof=1) / sqrt(n)`` of the paired differences ``d``.
        With two finite days the differences are (0.1, 0.3): mean 0.2, sample sd ``sqrt(0.02)``, standard
        error exactly 0.1 -- two days already have a defined sample sd, so the error must NOT fall back to
        the one-day 0.0. With four days the differences are (0.1, 0.2, 0.3, 0.6): mean 0.3, squared
        deviations summing to 0.14, so the standard error is ``sqrt(0.14 / 3) / 2``. Either a population
        sd (``ddof=0``), a multiplied rather than divided ``sqrt(n)``, or a moved one-day cut-off gives a
        different number.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        gain, error = paired_gain(np.array([-3.4, -3.1]), np.array([-3.5, -3.4]))
        assert gain == pytest.approx(0.2)
        assert error == pytest.approx(0.1)
        incumbent = np.array([-3.5, -3.6, -3.4, -3.7])
        gain, error = paired_gain(incumbent + np.array([0.1, 0.2, 0.3, 0.6]), incumbent)
        assert gain == pytest.approx(0.3)
        assert error == pytest.approx(np.sqrt(0.14 / 3.0) / 2.0)


class TestFeatureColumns:
    """Column selection has to honour the ASKED order, not the design's."""

    def test_columns_follow_the_requested_order(self) -> None:
        """Asking for ``["c", "a"]`` at three lags must return c's lag columns before a's, so the control's
        weights line up with the order it was fitted in, not the design's."""
        assert feature_columns(["a", "b", "c"], ["c", "a"], 3).tolist() == [6, 7, 8, 0, 1, 2]

    def test_an_absent_feature_raises(self) -> None:
        """A feature missing from the design must raise ``KeyError`` ("not in the assembled design") rather than
        select the wrong columns."""
        with pytest.raises(KeyError, match="not in the assembled design"):
            feature_columns(["a", "b"], ["z"], 3)


class TestControlEstimator:
    """The cohort control must carry the run's OWN penalties and period, not the estimator's defaults."""

    def test_the_settings_penalties_and_period_reach_the_estimator(self, settings: dict) -> None:
        """
        Description
        -----------
        The shipped ``lambda_smooth`` (1.0), ``l2_reg`` (0.01), ``smoothness_derivative_order`` (1) and
        torus period (1.0) all coincide with ``NestedTorusRegression``'s constructor defaults, so a value
        that never reaches the estimator is invisible under the shipped settings. Here every one is set
        off its default (3.0, 0.2, 2 and 2.0), and the estimator must report exactly those, must be a
        REDUCED model (no neural columns), and must be shaped as one "feature" of
        ``n_features * n_lags`` bins (three features at four lags: 12) for the parent's column check.
        These four are the arguments the closed-form solve actually consumes, so they set the fitted
        coefficients.

        Parameters
        ----------
        settings (dict)
            The shipped settings, adjusted off the estimator defaults.

        Returns
        -------
        None
        """
        block = settings["nested_vocal_decoding"]
        block["lambda_smooth"] = 3.0
        block["l2_reg"] = 0.2
        block["smoothness_derivative_order"] = 2
        settings["vocalization_settings"]["torus_period"] = 2.0
        model = control_estimator(3, 4, settings)
        assert model.lambda_smooth == 3.0
        assert model.l2_reg == 0.2
        assert model.smoothness_derivative_order == 2
        assert model.period == 2.0
        assert model.n_neural_columns == 0
        assert model.n_behaviour_features == 3
        assert model.n_lags == 4
        assert model.n_time_bins == 12


class TestPooledMacro:
    """The control is scored by the nested decoding's MACRO statistic, which needs the region labels."""

    def test_the_score_is_the_region_macro_not_the_pooled_mean(self, settings: dict) -> None:
        """
        Description
        -----------
        On the two-region cohort, with region 2 predicted worse than region 1 and outnumbered by it, the
        macro score (equal weight per region) sits below the pooled per-USV mean. ``pooled_macro`` must
        return the macro value computed directly from ``macro_von_mises_logscore`` with the region labels,
        and the guard checks the two statistics really are apart here, so the test can tell them apart.

        Parameters
        ----------
        settings (dict)
            The shipped settings (period 1.0, ``min_region_events`` 20).

        Returns
        -------
        None
        """
        cohort = two_region_cohort(seed=3)
        predicted = held_out_by_day(cohort["design"], cohort["positions"], cohort["regions"], cohort["days"],
                                    settings, SYNTHETIC_N_LAGS)
        macro = macro_reference(predicted, cohort["positions"], cohort["regions"])
        pooled = macro_reference(predicted, cohort["positions"], None)
        assert abs(macro - pooled) > 0.05, "the regions must differ, or the test proves nothing"
        assert pooled_macro(predicted, cohort["positions"], cohort["regions"], settings) == pytest.approx(
            macro, abs=1e-12)


class TestApplyBehaviourControl:
    """The swap that keeps the nesting and changes only the representation."""

    def test_the_block_becomes_four_columns_at_one_lag(self, settings: dict, artifact: dict) -> None:
        """Four coefficients are fitted on the unit; the behaviour comes from the cohort."""
        generator = np.random.default_rng(3)
        design = {"behaviour": generator.normal(size=(20, 10)),
                  "feature_names": ["nose-nose", "other.speed"],
                  "neural": generator.normal(size=(20, 1))}
        controlled, n_lags = apply_behaviour_control(design, artifact, "20250919", settings)
        assert n_lags == 1
        assert controlled["behaviour"].shape == (20, 4)
        assert controlled["feature_names"] == list(EMBEDDING_COLUMNS)
        assert controlled["behaviour_control"]["held_out_day"] == "20250919"
        assert controlled["behaviour_control"]["settings_sha256"] == "abc123"

    def test_the_provenance_record_carries_what_the_dispatcher_reads(self, settings: dict,
                                                                     artifact: dict) -> None:
        """
        Description
        -----------
        The dispatcher reads ``behaviour_control["features"]`` into every banked result and
        ``behaviour_control["predicted_positions"]`` to tune the behaviour prior, so the record must hold
        exactly those keys (plus the held-out day and settings hash): the control's features in order, and
        the predicted positions that ``control_predictions`` gives for the same block and day, which are
        also the positions the four embedding columns encode.

        Parameters
        ----------
        settings (dict)
            The shipped settings.
        artifact (dict)
            The two-feature, five-lag control fixture.

        Returns
        -------
        None
        """
        generator = np.random.default_rng(5)
        design = {"behaviour": generator.normal(size=(12, 10)), "feature_names": ["nose-nose", "other.speed"],
                  "neural": np.zeros((12, 1))}
        controlled, _n_lags = apply_behaviour_control(design, artifact, "20250919", settings)
        record = controlled["behaviour_control"]
        assert set(record) == {"features", "held_out_day", "predicted_positions", "settings_sha256"}
        assert record["features"] == ["nose-nose", "other.speed"]
        expected = control_predictions(artifact, "20250919", design["behaviour"], settings)
        np.testing.assert_array_equal(record["predicted_positions"], expected)
        np.testing.assert_allclose(controlled["behaviour"], torus_embedding(expected))

    def test_the_original_design_is_not_mutated(self, settings: dict, artifact: dict) -> None:
        """The caller keeps the raw block; an in-place swap would corrupt anything else holding it."""
        generator = np.random.default_rng(4)
        design = {"behaviour": generator.normal(size=(20, 10)),
                  "feature_names": ["nose-nose", "other.speed"], "neural": np.zeros((20, 1))}
        apply_behaviour_control(design, artifact, "20250919", settings)
        assert design["behaviour"].shape == (20, 10)
        assert design["feature_names"] == ["nose-nose", "other.speed"]

    def test_a_unit_whose_day_the_control_saw_is_refused(self, settings: dict,
                                                         artifact: dict) -> None:
        """The unit's own day must be held out of the control, or the reduced score is in-sample."""
        design = {"behaviour": np.zeros((5, 10)), "feature_names": ["nose-nose", "other.speed"],
                  "neural": np.zeros((5, 1))}
        with pytest.raises(KeyError, match="no fold for day"):
            apply_behaviour_control(design, artifact, "20250923", settings)

    def test_a_block_built_from_other_features_is_refused(self, settings: dict,
                                                          artifact: dict) -> None:
        """Columns that do not match the control's shape would be multiplied by the wrong weights."""
        design = {"behaviour": np.zeros((5, 7)), "feature_names": ["nose-nose"],
                  "neural": np.zeros((5, 1))}
        with pytest.raises(ValueError, match="must be built from the control's own features"):
            apply_behaviour_control(design, artifact, "20250919", settings)


class TestControlPredictions:
    """The stored coefficients must reproduce the fitted model's own predictions."""

    def test_a_known_linear_map_is_recovered(self, settings: dict) -> None:
        """coef_ and intercept_ are all predict() consumes, which is why only those are stored."""
        stored = {"features": ["nose-nose"], "n_lags": 2, "settings_sha256": "x",
                  "coefficients": {"20250919": {"coef": np.array([[1.0, 0.0, 0.0, 0.0],
                                                                  [0.0, 0.0, 1.0, 0.0]]),
                                                "intercept": np.array([0.0, 1.0, 0.0, 1.0])}}}
        predicted = control_predictions(stored, "20250919", np.array([[1.0, 1.0]]), settings)
        assert predicted.shape == (1, 2)
        assert np.all((predicted >= 0.0) & (predicted < 1.0)), "a torus coordinate must be in [0, 1)"


class TestLoadControlModel:
    """A control out of step with its settings changes every added score and no reported value."""

    def test_drifted_features_raise(self, settings: dict, artifact: dict, tmp_path) -> None:
        """A stored control whose features differ from the settings' reduced-model features must raise
        ``ValueError`` ("behaviour control drift") instead of being applied."""
        path = tmp_path / "control.pkl"
        write_control_model(artifact, str(path))
        settings["nested_vocal_decoding"]["behaviour_control_model_path"] = str(path)
        settings["nested_vocal_decoding"]["reduced_model_features"] = ["nose-nose"]
        with pytest.raises(ValueError, match="behaviour control drift"):
            load_control_model(settings)

    def test_matching_features_load(self, settings: dict, artifact: dict, tmp_path) -> None:
        """A stored control whose features match the settings must load and return those features, so the
        drift guard does not refuse a valid control."""
        path = tmp_path / "control.pkl"
        write_control_model(artifact, str(path))
        settings["nested_vocal_decoding"]["behaviour_control_model_path"] = str(path)
        settings["nested_vocal_decoding"]["reduced_model_features"] = list(
            artifact["features"])
        assert load_control_model(settings)["features"] == artifact["features"]

    def test_the_control_is_written_into_a_directory_that_does_not_exist_yet(self, artifact: dict,
                                                                              tmp_path) -> None:
        """
        Description
        -----------
        The artifact's destination sits several directory levels below anything that exists, as a fresh
        results tree does; writing must create every missing parent and the pickle must load back equal
        to what was written.

        Parameters
        ----------
        artifact (dict)
            The two-feature control fixture.
        tmp_path (pathlib.Path)
            Root under which the missing directories are created.

        Returns
        -------
        None
        """
        path = tmp_path / "results" / "behaviour" / "control.pkl"
        write_control_model(artifact, str(path))
        with path.open("rb") as handle:
            loaded = pickle.load(handle)
        assert loaded["features"] == artifact["features"]
        np.testing.assert_array_equal(loaded["coefficients"]["20250919"]["coef"],
                                      artifact["coefficients"]["20250919"]["coef"])

    def test_a_missing_control_names_the_dispatcher(self, settings: dict, tmp_path) -> None:
        """A missing control file must raise ``FileNotFoundError`` naming the behaviour-control dispatcher, so the
        reader knows which job produces it."""
        settings["nested_vocal_decoding"]["behaviour_control_model_path"] = str(
            tmp_path / "absent.pkl")
        with pytest.raises(FileNotFoundError, match="behaviour-control dispatcher"):
            load_control_model(settings)


class TestReducedModelFeaturesRepoint:
    """Either artifact shape may define the feature set, and both are checked against the freeze.

    A behavioural selection's FEATURE SET is adopted while its COEFFICIENTS never are -- the control
    is always refitted in domain, which is what the non-transferring fitted map requires. So the
    reader accepts both shapes and the drift assertion is what guards meaning, not the schema.
    """

    def test_a_behavioural_selection_feature_set_is_adopted(self, settings: dict,
                                                            tmp_path) -> None:
        """The last forward step's `current_features` is the selected set."""
        path = tmp_path / "behavioural.pkl"
        chosen = ["allo_yaw-head", "self.neck_elevation", "head-head"]
        path.write_bytes(pickle.dumps({"steps": [{"current_features": chosen[:1]},
                                                 {"current_features": chosen}]}))
        settings["data"]["behaviour_selection_result_path"] = str(path)
        settings["nested_vocal_decoding"]["reduced_model_features"] = chosen
        assert reduced_model_features(settings) == chosen

    def test_a_behavioural_selection_is_still_drift_checked(self, settings: dict,
                                                            tmp_path) -> None:
        """Adopting a feature set must not let a re-selection silently redefine the control."""
        path = tmp_path / "behavioural.pkl"
        path.write_bytes(pickle.dumps({"steps": [{"current_features": ["head-head"]}]}))
        settings["data"]["behaviour_selection_result_path"] = str(path)
        settings["nested_vocal_decoding"]["reduced_model_features"] = ["nose-nose"]
        with pytest.raises(ValueError, match="behaviour selection drift"):
            reduced_model_features(settings)

    def test_an_unrecognised_artifact_is_refused(self, settings: dict, tmp_path) -> None:
        """Neither shape present would otherwise raise an unhelpful KeyError deep in the reader."""
        path = tmp_path / "mystery.pkl"
        path.write_bytes(pickle.dumps({"something_else": ["a"]}))
        settings["data"]["behaviour_selection_result_path"] = str(path)
        with pytest.raises(ValueError, match="neither a cohort selection result"):
            reduced_model_features(settings)

    def test_a_cohort_selection_is_read_and_checked(self, settings: dict, tmp_path) -> None:
        """A cohort selection result's ``selected_features`` must be read and, when it matches the frozen
        settings list, returned unchanged."""
        path = tmp_path / "selection.pkl"
        chosen = ["nose-nose", "other.speed", "allo_yaw-nose"]
        path.write_bytes(pickle.dumps({"selected_features": chosen}))
        settings["data"]["behaviour_selection_result_path"] = str(path)
        settings["nested_vocal_decoding"]["reduced_model_features"] = chosen
        assert reduced_model_features(settings) == chosen

    def test_drift_between_file_and_settings_raises(self, settings: dict, tmp_path) -> None:
        """Freezing alone lets the control drift; reading alone lets a re-run redefine the claim."""
        path = tmp_path / "selection.pkl"
        path.write_bytes(pickle.dumps({"selected_features": ["nose-nose"]}))
        settings["data"]["behaviour_selection_result_path"] = str(path)
        settings["nested_vocal_decoding"]["reduced_model_features"] = ["other.speed"]
        with pytest.raises(ValueError, match="behaviour selection drift"):
            reduced_model_features(settings)


class TestSessionVocalizationDesign:
    """The control trains on EXACTLY the rows the per-unit analysis scores: the analysed vocalization set
    of the configured emitter, guarded against intrusions by the guard set, with every history ending at
    the frame before onset -- the frame the nested decoding builds the unit-side histories at."""

    SESSION, FPS, N_FRAMES = "20250101_000000", 10.0, 2000

    def _write(self, root: pathlib.Path, usvs: list, first_coordinates: list | None = None) -> None:
        """
        Description
        -----------
        Write a session's tracking-H5 header (``male``/``female`` tracks) and a USV summary carrying the
        torus-position, region and classifier columns the control reads.

        Parameters
        ----------
        root (pathlib.Path)
            The data root.
        usvs (list[tuple])
            ``(start, stop, emitter, noise, squeak)`` per row.
        first_coordinates (list | None)
            The ``qlvm1`` value per row (NaN marks a missing position); None writes 0.2 on every row.

        Returns
        -------
        None
            The files are written to disk.
        """
        video_dir = root / self.SESSION / "video" / f"{self.SESSION}_cam"
        video_dir.mkdir(parents=True)
        with h5py.File(video_dir / f"{self.SESSION}_points3d_translated_rotated_metric.h5", "w") as handle:
            handle["tracks"] = np.zeros((self.N_FRAMES, 2, 15, 3), dtype=np.float64)
            handle["recording_frame_rate"] = self.FPS
            handle.create_dataset("track_names", data=np.array([b"male", b"female"], dtype="S6"))
        audio_dir = root / self.SESSION / "audio"
        audio_dir.mkdir(parents=True)
        pls.DataFrame({"usv_id": list(range(len(usvs))),
                       "start": [row[0] for row in usvs], "stop": [row[1] for row in usvs],
                       "emitter": [row[2] for row in usvs],
                       "qlvm1": [0.2] * len(usvs) if first_coordinates is None else first_coordinates,
                       "qlvm2": [0.4] * len(usvs),
                       "qlvm_category": [1.0] * len(usvs),
                       "noise": [row[3] for row in usvs], "squeak": [row[4] for row in usvs],
                       }).write_csv(audio_dir / f"{self.SESSION}_usv_summary.csv")

    def _design(self, monkeypatch, tmp_path, settings: dict, emitter: str) -> dict:
        """
        Description
        -----------
        Run ``session_vocalization_design`` on the synthetic session with a one-feature kinematic series
        equal to the frame index, so a design column reveals the frame it was read at.

        Parameters
        ----------
        monkeypatch (pytest.MonkeyPatch)
            Patches the kinematic loader.
        tmp_path (pathlib.Path)
            The data root.
        settings (dict)
            The shipped settings, adjusted in place for the synthetic session.
        emitter (str)
            ``vocalization_settings.vocal_emitter``.

        Returns
        -------
        block (dict)
            As ``session_vocalization_design`` returns it.
        """
        self._write(tmp_path, [(50.0, 50.1, "male", False, False),
                               (60.0, 60.1, "male", True, False),     # noise: never a vocalization
                               (70.0, 70.1, "male", False, True),     # squeak: a different vocal type
                               (80.0, 80.1, "female", False, False),
                               (99.97, 99.99, "male", True, False),   # noise just before 100 s
                               (100.0, 100.1, "male", False, False)])
        settings["data_roots"]["data_root"] = str(tmp_path)
        settings["data_sufficiency"]["min_emitter_usvs_per_session"] = 1
        settings["vocalization_settings"]["vocal_emitter"] = emitter
        frames = pls.DataFrame({"f": np.arange(self.N_FRAMES, dtype=np.float64)})
        monkeypatch.setattr(control_module, "build_zscored_feature_frames",
                            lambda **_kwargs: ({self.SESSION: frames}, {self.SESSION: self.FPS}, {}, {}))
        return session_vocalization_design(str(tmp_path / self.SESSION), settings, 5)

    def test_noise_and_squeak_rows_are_not_trained_on(self, monkeypatch, tmp_path, settings: dict) -> None:
        """
        Repro basis: the control read the raw summary, so a noise row and a squeak row became training
        rows, and a noise row vetoed the vocalization after it as an "intrusion". It now keeps exactly
        the analysed set (no noise, no squeaks), guarded by the guard set (no noise).
        """
        block = self._design(monkeypatch, tmp_path, settings, "self")
        np.testing.assert_allclose(block["vocalization_start"], [50.0, 100.0])
        assert block["n_kept"] == 2

    def test_vocal_emitter_all_trains_on_every_emitter(self, monkeypatch, tmp_path, settings: dict) -> None:
        """``'all'`` means no emitter filter; it used to index the empty name list and raise."""
        block = self._design(monkeypatch, tmp_path, settings, "all")
        np.testing.assert_allclose(block["vocalization_start"], [50.0, 80.0, 100.0])

    def test_the_history_ends_at_the_frame_the_nested_decoding_applies_it_at(
            self, monkeypatch, tmp_path, settings: dict) -> None:
        """The fit anchors are ``onset_history_anchor_frames`` -- the frame before onset -- the same frames
        ``nested_design`` builds the unit-side histories at."""
        block = self._design(monkeypatch, tmp_path, settings, "self")
        expected = onset_history_anchor_frames(block["vocalization_start"], self.FPS)
        np.testing.assert_array_equal(expected, [499, 999])
        np.testing.assert_allclose(block["design"][:, -1], expected)

    def test_every_filter_boundary_and_the_returned_block(self, monkeypatch, tmp_path, settings: dict) -> None:
        """
        Description
        -----------
        One session built so that each filter decides exactly one row at its boundary. The recorded male
        emits six analysed vocalizations and the per-session minimum is set to exactly six, so the session
        must be kept (the minimum is inclusive). With ``n_lags = 500`` at 10 fps and a kinematic series of
        1,000 frames (the frame index, with frame 10 missing):

        - 49.90 s (anchor 498): one frame short of a full history -- dropped;
        - 50.00 s (anchor 499 = ``n_lags - 1``): exactly a full history -- KEPT;
        - 60.00 s: its ``qlvm1`` is missing -- dropped, and ONLY that row;
        - 70.12 s: a female vocalization (70.00-70.10 s) overlaps its 0.05 s pre-window [70.07, 70.12) --
          dropped as intruded;
        - 90.00 s (anchor 899): a female vocalization ENDS exactly at its window opening (89.95 s), which
          touches but does not overlap the window -- KEPT;
        - 100.10 s (anchor 1000 = the series length): its history would end past the session -- dropped.

        The kinematic loader must be asked for exactly this session, the configured kinematic features and
        the recorded animal (the first track). The returned block must carry exactly the documented keys:
        the session id, a float32 design whose last column is each kept anchor and whose frame-10 column is
        0 (a missing kinematic sample is filled with zero), the kept positions and regions, the feature
        names, the count and the onsets.

        Parameters
        ----------
        monkeypatch (pytest.MonkeyPatch)
            Patches the kinematic loader with a strict fake.
        tmp_path (pathlib.Path)
            The data root.
        settings (dict)
            The shipped settings, adjusted for the synthetic session.

        Returns
        -------
        None
        """
        nan = float("nan")
        rows = [(49.9, 49.92, "male", False, False, 0.2), (50.0, 50.1, "male", False, False, 0.3),
                (60.0, 60.1, "male", False, False, nan), (70.0, 70.1, "female", False, False, 0.2),
                (70.12, 70.2, "male", False, False, 0.2), (89.9, 89.95, "female", False, False, 0.2),
                (90.0, 90.1, "male", False, False, 0.6), (100.1, 100.2, "male", False, False, 0.2)]
        self._write(tmp_path, [row[:5] for row in rows], first_coordinates=[row[5] for row in rows])
        settings["data_roots"]["data_root"] = str(tmp_path)
        settings["data_sufficiency"]["min_emitter_usvs_per_session"] = 6
        settings["vocalization_settings"]["vocal_emitter"] = "self"
        series = np.arange(1000, dtype=np.float64)
        series[10] = np.nan
        frames = pls.DataFrame({"f": series})
        session_directory = str(tmp_path / self.SESSION)

        def strict_loader(*, session_dirs: list, kinematic_features: object, recorded_mouse_id: str) -> tuple:
            """
            Description
            -----------
            Keyword-only stand-in for ``build_zscored_feature_frames`` that checks what it is asked for and
            returns the one-feature frame-index series.

            Parameters
            ----------
            session_dirs (list)
                Must be exactly ``[session_directory]``.
            kinematic_features (object)
                Must be the settings' ``kinematic_features`` block.
            recorded_mouse_id (str)
                Must be ``"male"``, the first track.

            Returns
            -------
            loaded (tuple)
                ``(frames, camera_fps, names, suffixes)`` keyed by the session id.
            """
            assert session_dirs == [session_directory]
            assert kinematic_features == settings["kinematic_features"]
            assert recorded_mouse_id == "male"
            return {self.SESSION: frames}, {self.SESSION: self.FPS}, {}, {}

        monkeypatch.setattr(control_module, "build_zscored_feature_frames", strict_loader)
        block = session_vocalization_design(session_directory, settings, 500)
        assert set(block) == {"session_id", "design", "positions", "regions", "feature_names", "n_kept",
                              "vocalization_start"}
        assert block["session_id"] == self.SESSION
        np.testing.assert_allclose(block["vocalization_start"], [50.0, 90.0])
        assert block["n_kept"] == 2
        assert block["design"].dtype == np.float32
        assert block["design"].shape == (2, 500)
        np.testing.assert_array_equal(block["design"][:, -1], [499.0, 899.0])
        assert block["design"][0, 10] == 0.0
        np.testing.assert_allclose(block["positions"], [[0.3, 0.4], [0.6, 0.4]])
        np.testing.assert_array_equal(block["regions"], [1.0, 1.0])
        assert block["feature_names"] == ["f"]

    def _write_tracking_only(self, root: pathlib.Path, track_names: list) -> None:
        """
        Description
        -----------
        Write ONLY the session's tracking-H5 header, with as many tracks as ``track_names`` names and no
        USV summary, for the skip paths that are decided before (or by) the summary read.

        Parameters
        ----------
        root (pathlib.Path)
            The data root.
        track_names (list[bytes])
            The ``track_names`` to store, one per tracked animal.

        Returns
        -------
        None
            The file is written to disk.
        """
        video_dir = root / self.SESSION / "video" / f"{self.SESSION}_cam"
        video_dir.mkdir(parents=True)
        with h5py.File(video_dir / f"{self.SESSION}_points3d_translated_rotated_metric.h5", "w") as handle:
            handle["tracks"] = np.zeros((self.N_FRAMES, len(track_names), 15, 3), dtype=np.float64)
            handle["recording_frame_rate"] = self.FPS
            handle.create_dataset("track_names", data=np.array(track_names, dtype="S6"))

    def test_a_session_without_exactly_two_animals_is_skipped(self, tmp_path, settings: dict) -> None:
        """
        Description
        -----------
        The emitter mapping needs a recorded animal and ONE partner, so a three-animal session must come
        back as a skip whose reason names the track count, before any summary is read (none exists here,
        so reaching the summary read would raise a different reason).

        Parameters
        ----------
        tmp_path (pathlib.Path)
            The data root.
        settings (dict)
            The shipped settings, pointed at the synthetic data root.

        Returns
        -------
        None
        """
        self._write_tracking_only(tmp_path, [b"male", b"female", b"third"])
        settings["data_roots"]["data_root"] = str(tmp_path)
        block = session_vocalization_design(str(tmp_path / self.SESSION), settings, 5)
        assert block == {"session_id": self.SESSION, "reason": "3 tracked animals"}

    def test_a_session_without_a_summary_is_skipped_not_raised(self, tmp_path, settings: dict) -> None:
        """
        Description
        -----------
        A session with tracking but no USV summary is a property of that one session, so it must be
        skipped with the loader's ``FileNotFoundError`` message as the reason rather than abort the
        cohort assembly.

        Parameters
        ----------
        tmp_path (pathlib.Path)
            The data root.
        settings (dict)
            The shipped settings, pointed at the synthetic data root.

        Returns
        -------
        None
        """
        self._write_tracking_only(tmp_path, [b"male", b"female"])
        settings["data_roots"]["data_root"] = str(tmp_path)
        block = session_vocalization_design(str(tmp_path / self.SESSION), settings, 5)
        assert set(block) == {"session_id", "reason"}
        assert block["reason"].startswith("usv_summary not found")

    def test_a_summary_missing_the_configured_columns_raises(self, tmp_path, settings: dict) -> None:
        """
        Description
        -----------
        A summary lacking a configured torus column hits every session alike, so it must RAISE a
        ``KeyError`` naming exactly the absent column and the two settings to reconcile, instead of being
        skipped (which is how this control once went silently unfitted).

        Parameters
        ----------
        tmp_path (pathlib.Path)
            The data root; the summary is rewritten without ``qlvm2``.
        settings (dict)
            The shipped settings, pointed at the synthetic data root.

        Returns
        -------
        None
        """
        self._write(tmp_path, [(50.0, 50.1, "male", False, False)])
        summary = tmp_path / self.SESSION / "audio" / f"{self.SESSION}_usv_summary.csv"
        pls.read_csv(summary).drop("qlvm2").write_csv(summary)
        settings["data_roots"]["data_root"] = str(tmp_path)
        with pytest.raises(KeyError, match=re.escape("usv_summary lacks ['qlvm2']")):
            session_vocalization_design(str(tmp_path / self.SESSION), settings, 5)

    def test_too_few_emitter_vocalizations_are_skipped_with_the_count(self, tmp_path, settings: dict) -> None:
        """
        Description
        -----------
        The session carries four male rows, of which one is noise and one a squeak, plus one female row,
        so exactly two analysed vocalizations from the recorded male remain. A per-session minimum of
        three must skip it with the reason ``"2 emitter vocalizations"`` -- the count AFTER the analysed-set
        and emitter filters, not the raw row count (5) -- and before any kinematics are loaded (none are
        patched here, so loading them would fail).

        Parameters
        ----------
        tmp_path (pathlib.Path)
            The data root.
        settings (dict)
            The shipped settings, pointed at the synthetic data root with a minimum of 3.

        Returns
        -------
        None
        """
        self._write(tmp_path, [(50.0, 50.1, "male", False, False),
                               (60.0, 60.1, "male", True, False),
                               (70.0, 70.1, "male", False, True),
                               (80.0, 80.1, "female", False, False),
                               (100.0, 100.1, "male", False, False)])
        settings["data_roots"]["data_root"] = str(tmp_path)
        settings["data_sufficiency"]["min_emitter_usvs_per_session"] = 3
        settings["vocalization_settings"]["vocal_emitter"] = "self"
        block = session_vocalization_design(str(tmp_path / self.SESSION), settings, 5)
        assert block == {"session_id": self.SESSION, "reason": "2 emitter vocalizations"}

    def test_a_history_longer_than_any_onset_leaves_nothing(
            self, monkeypatch, tmp_path, settings: dict) -> None:
        """
        Description
        -----------
        The surviving onsets sit at anchor frames 499 and 999; a 2,000-frame history therefore runs off the
        start of the session for both, so nothing survives and the session is skipped with that reason
        rather than ``lagged_design`` raising on a negative start index.

        Parameters
        ----------
        monkeypatch (pytest.MonkeyPatch)
            Patches the kinematic loader.
        tmp_path (pathlib.Path)
            The data root.
        settings (dict)
            The shipped settings, adjusted for the synthetic session.

        Returns
        -------
        None
        """
        self._write(tmp_path, [(50.0, 50.1, "male", False, False), (100.0, 100.1, "male", False, False)])
        settings["data_roots"]["data_root"] = str(tmp_path)
        settings["data_sufficiency"]["min_emitter_usvs_per_session"] = 1
        frames = pls.DataFrame({"f": np.arange(self.N_FRAMES, dtype=np.float64)})
        monkeypatch.setattr(control_module, "build_zscored_feature_frames",
                            lambda **_kwargs: ({self.SESSION: frames}, {self.SESSION: self.FPS}, {}, {}))
        block = session_vocalization_design(str(tmp_path / self.SESSION), settings, self.N_FRAMES)
        assert block == {"session_id": self.SESSION, "reason": "no vocalization survives the filters"}


class TestLegacyCountKeys:
    """The stored count was renamed from ``n_calls`` to ``n_vocalizations``; older artifacts still read."""

    def test_an_old_artifact_is_read_under_the_new_names(self, settings: dict, artifact: dict,
                                                       tmp_path) -> None:
        """A control written before the rename (``n_calls`` at the top level and per day) loads with
        ``n_vocalizations`` in both places and no ``n_calls`` left."""
        old = {**artifact, "n_calls": 120, "per_day": {"20250919": {"n_calls": 120, "behaviour": -3.4,
                                                                     "no_behaviour": -3.5}}}
        path = tmp_path / "control.pkl"
        write_control_model(old, str(path))
        settings["nested_vocal_decoding"]["behaviour_control_model_path"] = str(path)
        settings["nested_vocal_decoding"]["reduced_model_features"] = list(
            artifact["features"])
        loaded = load_control_model(settings)
        assert loaded["n_vocalizations"] == 120
        assert loaded["per_day"]["20250919"]["n_vocalizations"] == 120
        assert "n_calls" not in loaded
        assert "n_calls" not in loaded["per_day"]["20250919"]

    def test_a_current_record_passes_through_unchanged_and_unmutated(self) -> None:
        """A record already on the new names is returned equal, and the input is never mutated."""
        record = {"n_vocalizations": 5, "per_day": {"d": {"n_vocalizations": 5}}}
        assert upgrade_legacy_counts(record) == record
        legacy = {"n_calls": 7}
        upgrade_legacy_counts(legacy)
        assert legacy == {"n_calls": 7}


class RecordingEstimator:
    """
    Description
    -----------
    A stand-in for the control estimator that RECORDS what each fold was fitted and asked to predict.

    The day holdout is the property under test, and it is invisible in a real estimator's output. This
    one stores, per instance, the rows it was trained on (read back from a design column that carries
    each USV's row index), the sample weights it received, and the rows it predicted; it predicts the
    plain mean of its training positions, so the expected held-out prediction for a day is computable by
    hand as the mean over the OTHER days.

    Parameters
    ----------
    n_features (int)
        Feature count the caller derived from the design; recorded for the test.
    n_lags (int)
        Lag count the caller passed; recorded for the test.
    registry (list)
        Shared list every new instance appends itself to.

    Returns
    -------
    estimator (RecordingEstimator)
        An unfitted recorder.
    """

    def __init__(self, n_features: int, n_lags: int, registry: list) -> None:
        """
        Description
        -----------
        Record the shape arguments and register the instance.

        Parameters
        ----------
        n_features (int)
            Feature count the caller derived from the design.
        n_lags (int)
            Lag count the caller passed.
        registry (list)
            Shared list this instance appends itself to.

        Returns
        -------
        None
        """
        self.shape = (n_features, n_lags)
        self.train_rows = None
        self.test_rows = None
        self.sample_weight = None
        self.mean_position = None
        self.snap = None
        registry.append(self)

    def fit(self, design: np.ndarray, positions: np.ndarray, sample_weight: np.ndarray) -> RecordingEstimator:
        """
        Description
        -----------
        Store the training rows (design column 0 holds each USV's row index), the weights, and the mean
        training position.

        Parameters
        ----------
        design (np.ndarray)
            Training design; column 0 is the row index.
        positions (np.ndarray)
            ``(n_train, 2)`` training positions.
        sample_weight (np.ndarray)
            Per-row training weights.

        Returns
        -------
        self (RecordingEstimator)
            The fitted recorder.
        """
        self.train_rows = design[:, 0].astype(np.int64)
        self.sample_weight = np.asarray(sample_weight, dtype=np.float64)
        self.mean_position = positions.mean(axis=0)
        return self

    def predict(self, design: np.ndarray, snap: bool) -> np.ndarray:
        """
        Description
        -----------
        Store the predicted rows and the ``snap`` flag, and predict the mean training position for every
        row.

        Parameters
        ----------
        design (np.ndarray)
            Test design; column 0 is the row index.
        snap (bool)
            The caller's snap flag; recorded so the test can confirm predictions are unsnapped.

        Returns
        -------
        predicted (np.ndarray)
            ``(n_test, 2)`` copies of the mean training position.
        """
        self.test_rows = design[:, 0].astype(np.int64)
        self.snap = snap
        return np.tile(self.mean_position, (design.shape[0], 1))


class TestHeldOutByDay:
    """Every day is predicted exactly once, by a model that never saw that day."""

    def test_no_day_is_ever_in_its_own_training_set(self, monkeypatch, settings: dict) -> None:
        """
        Description
        -----------
        Three days of unequal size (3, 4 and 5 USVs) and two regions of unequal size. A recording estimator
        replaces the real one, and the design's first column carries each row's index. The test checks,
        fold by fold, that: (i) one fold is fitted per unique day; (ii) its training rows are EXACTLY the
        rows of the other days and its test rows exactly the held-out day's, so train and test are disjoint
        and the test sets partition every row once; (iii) the weights it receives are the cohort-wide
        inverse-region-frequency weights restricted to the training rows (computed on the whole cohort,
        not refitted per fold); (iv) the feature count passed is ``n_columns // n_lags`` and predictions
        are requested unsnapped; and (v) each held-out row's prediction equals the hand-computed mean of
        the other days' positions.

        Parameters
        ----------
        monkeypatch (pytest.MonkeyPatch)
            Replaces ``control_estimator`` with the recorder factory.
        settings (dict)
            The shipped settings (unused by the recorder, passed through by the function).

        Returns
        -------
        None
        """
        days = np.array(["20250103"] * 3 + ["20250101"] * 4 + ["20250102"] * 5)
        n_rows = days.size
        design = np.column_stack([np.arange(n_rows, dtype=np.float64), np.zeros(n_rows),
                                  np.zeros(n_rows), np.zeros(n_rows)])
        positions = np.column_stack([np.linspace(0.05, 0.6, n_rows), np.linspace(0.3, 0.4, n_rows)])
        regions = np.array([1.0, 2.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0])
        registry: list = []
        monkeypatch.setattr(control_module, "control_estimator",
                            lambda n_features, n_lags, _settings: RecordingEstimator(n_features, n_lags,
                                                                                    registry))
        predicted = held_out_by_day(design, positions, regions, days, settings, n_lags=2)

        assert len(registry) == 3
        weights = inverse_region_frequency_weights(regions)
        seen_test_rows = []
        for fold in registry:
            held_out_days = set(days[fold.test_rows])
            assert len(held_out_days) == 1
            held_out = held_out_days.pop()
            np.testing.assert_array_equal(fold.test_rows, np.flatnonzero(days == held_out))
            np.testing.assert_array_equal(fold.train_rows, np.flatnonzero(days != held_out))
            assert held_out not in set(days[fold.train_rows])
            np.testing.assert_allclose(fold.sample_weight, weights[fold.train_rows])
            assert fold.shape == (2, 2)
            assert fold.snap is False
            expected = positions[days != held_out].mean(axis=0)
            np.testing.assert_allclose(predicted[days == held_out], np.tile(expected, ((days == held_out).sum(), 1)))
            seen_test_rows.extend(fold.test_rows.tolist())
        assert sorted(seen_test_rows) == list(range(n_rows))
        assert np.isfinite(predicted).all()

    def test_a_planted_feature_is_predicted_out_of_day(self, settings: dict) -> None:
        """
        Description
        -----------
        With the real estimator on the planted synthetic cohort, the held-out predictions from the full
        design must track the truth closely (median wrap-aware error under 0.03 torus units, against a
        noise sd of 0.02 and a signal range of 0.3), while predictions from an all-zero design -- the
        no-behaviour baseline -- can only recover the training mean and so miss by far more (median error
        above 0.05, the mean absolute value of ``0.15 * U(-1, 1)`` being 0.075). The held-out macro score
        must order the two the same way.

        Parameters
        ----------
        settings (dict)
            The shipped settings (closed-form ridge, so the fit is fast).

        Returns
        -------
        None
        """
        cohort = synthetic_cohort(seed=0, planted=True)
        predicted = held_out_by_day(cohort["design"], cohort["positions"], cohort["regions"], cohort["days"],
                                    settings, SYNTHETIC_N_LAGS)
        blank = held_out_by_day(np.zeros_like(cohort["design"][:, :SYNTHETIC_N_LAGS]), cohort["positions"],
                                cohort["regions"], cohort["days"], settings, SYNTHETIC_N_LAGS)
        assert np.median(circular_error(predicted, cohort["positions"])) < 0.03
        assert np.median(circular_error(blank, cohort["positions"])) > 0.05
        assert (pooled_macro(predicted, cohort["positions"], cohort["regions"], settings)
                > pooled_macro(blank, cohort["positions"], cohort["regions"], settings))


class TestFitControlModels:
    """The per-day fit must recover the planted map and report what it was fitted on."""

    def test_the_planted_coefficient_is_recovered_in_every_fold(self, settings: dict) -> None:
        """
        Description
        -----------
        The first torus coordinate is ``0.25 + 0.15 * signal``, so its cosine embedding is
        ``-sin(0.3 * pi * signal)``: decreasing in ``signal``. Each fold's coefficients for the cosine output
        on the two ``signal`` lag columns are first-order-smoothed and must therefore be near-equal, and
        their SUM must match the slope of an ordinary least-squares line of that cosine on the mean of the
        two lag columns, fitted on the same training days (within 5%: the ridge and smoothness penalties
        shrink it only slightly at this sample size). The noise feature's cosine coefficients must be an
        order of magnitude smaller. The features are requested as ``["noise_a", "signal"]``, so the
        signal's coefficients must sit in ROWS 2-3 -- the requested order, not the cohort's.

        Parameters
        ----------
        settings (dict)
            The shipped settings.

        Returns
        -------
        None
        """
        cohort = synthetic_cohort(seed=0, planted=True)
        artifact = fit_control_models(cohort, ["noise_a", "signal"], settings, SYNTHETIC_N_LAGS)
        assert sorted(artifact["coefficients"]) == SYNTHETIC_DAYS
        cosine = np.cos(2.0 * np.pi * cohort["positions"][:, 0])
        signal_columns = cohort["design"][:, 0:2].astype(np.float64).mean(axis=1)
        for day, fold in artifact["coefficients"].items():
            assert fold["coef"].shape == (4, 4)
            assert fold["intercept"].shape == (4,)
            train = cohort["days"] != day
            slope = np.polyfit(signal_columns[train], cosine[train], 1)[0]
            signal_coef = fold["coef"][2:4, 0]
            assert slope < -0.5
            assert signal_coef.sum() == pytest.approx(slope, rel=0.05)
            assert signal_coef[0] == pytest.approx(signal_coef[1], abs=0.05)
            assert np.abs(fold["coef"][0:2, 0]).max() < 0.1 * np.abs(signal_coef).max()

    def test_the_artifact_reports_scores_counts_and_provenance(self, settings: dict) -> None:
        """
        Description
        -----------
        The artifact must carry: the requested features and lag count; one per-day entry per day holding
        that day's USV count (40) and a behaviour score that BEATS that day's no-behaviour score (the
        planted feature is real); a pooled held-out score above the pooled no-behaviour score; the total
        USV count (160); the cohort's session ids and skip reasons copied through; and the SHA-256 of the
        exact settings it was fitted under, so a drifted control is detectable.

        Parameters
        ----------
        settings (dict)
            The shipped settings.

        Returns
        -------
        None
        """
        cohort = synthetic_cohort(seed=0, planted=True)
        artifact = fit_control_models(cohort, ["signal"], settings, SYNTHETIC_N_LAGS)
        assert artifact["features"] == ["signal"]
        assert artifact["n_lags"] == SYNTHETIC_N_LAGS
        assert sorted(artifact["per_day"]) == SYNTHETIC_DAYS
        for scores in artifact["per_day"].values():
            assert scores["n_vocalizations"] == 40
            assert scores["behaviour"] > scores["no_behaviour"]
        assert artifact["held_out_score"] > artifact["no_behaviour_score"]
        assert artifact["n_vocalizations"] == 160
        assert artifact["session_ids"] == cohort["session_ids"]
        assert artifact["skipped"] == cohort["skipped"]
        assert artifact["settings_sha256"] == compute_settings_sha256(settings)

    def test_the_stored_folds_reproduce_the_held_out_predictions(self, settings: dict) -> None:
        """
        Description
        -----------
        Only ``coef_`` and ``intercept_`` are stored, on the claim that they are all ``predict`` consumes.
        Re-predicting each day from its stored fold through ``control_predictions`` and re-scoring must
        therefore give back the artifact's own per-day behaviour score to numerical precision.

        Parameters
        ----------
        settings (dict)
            The shipped settings.

        Returns
        -------
        None
        """
        cohort = synthetic_cohort(seed=1, planted=True)
        artifact = fit_control_models(cohort, ["signal", "noise_b"], settings, SYNTHETIC_N_LAGS)
        columns = feature_columns(cohort["feature_names"], ["signal", "noise_b"], SYNTHETIC_N_LAGS)
        for day in SYNTHETIC_DAYS:
            rows = cohort["days"] == day
            predicted = control_predictions(artifact, day, cohort["design"][rows][:, columns], settings)
            rescored = pooled_macro(predicted, cohort["positions"][rows], cohort["regions"][rows], settings)
            assert rescored == pytest.approx(artifact["per_day"][day]["behaviour"], abs=1e-9)


    def test_each_fold_is_the_weighted_fit_on_the_other_days_and_scored_by_region(self,
                                                                               settings: dict) -> None:
        """
        Description
        -----------
        On the two-region cohort, every stored fold must equal -- to numerical precision -- a control
        fitted on the OTHER three days with the cohort-wide inverse-region-frequency weights restricted to
        those days. A fold fitted on its own day, or fitted unweighted, gives different coefficients
        (region 1 outnumbers region 2 by 55 to 25 per day, so the weights are far from uniform). The scores
        must be the region MACRO: each day's behaviour score is the macro score of that fold's predictions
        on that day, each day's no-behaviour score the macro score of the all-zero-design held-out
        predictions, and the two pooled scores the macro over every USV, all computed with the region
        labels directly from ``macro_von_mises_logscore``. The guard confirms the pooled-per-USV score
        differs here, so a dropped label is detectable.

        Parameters
        ----------
        settings (dict)
            The shipped settings.

        Returns
        -------
        None
        """
        cohort = two_region_cohort(seed=4)
        positions, regions, days = cohort["positions"], cohort["regions"], cohort["days"]
        artifact = fit_control_models(cohort, ["signal"], settings, SYNTHETIC_N_LAGS)
        design = cohort["design"][:, 0:SYNTHETIC_N_LAGS].astype(np.float64)
        weights = inverse_region_frequency_weights(regions)
        assert np.unique(weights).size == 2
        predicted = np.full_like(positions, np.nan)
        for day in SYNTHETIC_DAYS:
            train, test = days != day, days == day
            reference = control_estimator(1, SYNTHETIC_N_LAGS, settings)
            reference.fit(design[train], positions[train], sample_weight=weights[train])
            np.testing.assert_allclose(artifact["coefficients"][day]["coef"], reference.coef_, rtol=1e-10,
                                       atol=1e-12)
            np.testing.assert_allclose(artifact["coefficients"][day]["intercept"], reference.intercept_,
                                       rtol=1e-10, atol=1e-12)
            predicted[test] = reference.predict(design[test], snap=False)
        blank = held_out_by_day(np.zeros_like(design), positions, regions, days, settings, SYNTHETIC_N_LAGS)
        for day in SYNTHETIC_DAYS:
            rows = days == day
            assert artifact["per_day"][day]["behaviour"] == pytest.approx(
                macro_reference(predicted[rows], positions[rows], regions[rows]), abs=1e-9)
            assert artifact["per_day"][day]["no_behaviour"] == pytest.approx(
                macro_reference(blank[rows], positions[rows], regions[rows]), abs=1e-9)
        assert artifact["held_out_score"] == pytest.approx(macro_reference(predicted, positions, regions), abs=1e-9)
        assert artifact["no_behaviour_score"] == pytest.approx(macro_reference(blank, positions, regions), abs=1e-9)
        assert abs(macro_reference(predicted, positions, None) - artifact["held_out_score"]) > 0.05
        assert abs(macro_reference(blank, positions, None) - artifact["no_behaviour_score"]) > 0.05


class StubbedScores:
    """
    Description
    -----------
    A lookup that stands in for ``_score_candidate`` so the forward selection's ranking and acceptance
    rule can be driven with EXACT per-day scores.

    The real scorer fits and scores a model; its numbers are never round, so the boundaries of the
    acceptance rule (a gain less error of exactly zero, a gain above one nat) cannot be hit with it. This
    stub returns, for each candidate tuple, the per-day scores it was given; an unlisted candidate raises
    ``KeyError`` so an unexpected selection path fails loudly.

    Parameters
    ----------
    table (dict)
        ``{candidate_tuple: per_day_scores}``.

    Returns
    -------
    stub (StubbedScores)
        A callable with ``_score_candidate``'s signature.
    """

    def __init__(self, table: dict) -> None:
        """
        Description
        -----------
        Store the candidate-to-scores table.

        Parameters
        ----------
        table (dict)
            ``{candidate_tuple: per_day_scores}``.

        Returns
        -------
        None
        """
        self.table = table

    def __call__(self, arguments: tuple) -> tuple:
        """
        Description
        -----------
        Return the candidate, its tabled per-day scores, and their mean as the pooled score.

        Parameters
        ----------
        arguments (tuple)
            ``_score_candidate``'s argument tuple; only its last element (the candidate) is read.

        Returns
        -------
        result (tuple)
            ``(candidate, per_day, pooled)``.
        """
        candidate = arguments[-1]
        per_day = np.asarray(self.table[candidate], dtype=np.float64)
        return candidate, per_day, float(per_day.mean())


class TestSelectControlFeatures:
    """Forward selection must find a planted feature and refuse to invent one from noise."""

    def _select(self, tmp_path: pathlib.Path, settings: dict, planted: bool) -> dict:
        """
        Description
        -----------
        Build the seed-0 synthetic cohort, store its design as the ``.npy`` the workers memory-map, and run
        ``select_control_features`` with ``noise_a`` as the external reference and two requested workers.

        Parameters
        ----------
        tmp_path (pathlib.Path)
            Where the design ``.npy`` is written.
        settings (dict)
            The shipped settings.
        planted (bool)
            Whether ``signal`` drives position in the synthetic cohort.

        Returns
        -------
        selection (dict)
            As ``select_control_features`` returns it.
        """
        cohort = synthetic_cohort(seed=0, planted=planted)
        design_path = tmp_path / "design.npy"
        np.save(design_path, cohort["design"])
        return select_control_features(cohort, settings, SYNTHETIC_N_LAGS, 2, str(design_path),
                                       external_reference=["noise_a"])

    def test_the_planted_feature_is_selected_and_the_noise_is_not(self, tmp_path, settings: dict,
                                                                 serial_pools: type) -> None:
        """
        Description
        -----------
        With ``signal`` driving position, the first step must accept ``signal`` with a paired gain that
        clears its own standard error, the second step must REJECT the better of the two noise features
        (gain less error not above zero) and stop, and the selection must be exactly ``["signal"]``. The
        selected model's pooled score must beat the no-behaviour score, the external reference (a noise
        feature chosen "elsewhere") must score below the selected model, and the bookkeeping (days, count,
        sessions, skips, settings hash) must describe the cohort it ran on. Each step opens one pool with
        the requested two workers and the module's pool context.

        Parameters
        ----------
        tmp_path (pathlib.Path)
            Where the design ``.npy`` is written.
        settings (dict)
            The shipped settings.
        serial_pools (type)
            The in-process executor class, whose recorded pool arguments are checked.

        Returns
        -------
        None
        """
        selection = self._select(tmp_path, settings, planted=True)
        assert selection["selected_features"] == ["signal"]
        assert [step["feature"] for step in selection["per_step"]] == ["signal"]
        first = selection["per_step"][0]
        assert first["paired_gain"] - first["standard_error"] > 0.0
        assert first["pooled"] == selection["pooled_score"]
        assert selection["rejected"]["feature"] in {"noise_a", "noise_b"}
        assert selection["rejected"]["paired_gain"] - selection["rejected"]["standard_error"] <= 0.0
        assert selection["pooled_score"] > selection["no_behaviour_score"]
        assert selection["external_reference"]["features"] == ["noise_a"]
        assert selection["external_reference"]["pooled"] < selection["pooled_score"]
        assert selection["days"] == SYNTHETIC_DAYS
        assert selection["n_vocalizations"] == 160
        assert selection["session_ids"] == [f"{day}_120000" for day in SYNTHETIC_DAYS]
        assert selection["skipped"] == {"20250105_120000": "3 tracked animals"}
        assert selection["settings_sha256"] == compute_settings_sha256(settings)
        assert serial_pools.requested_workers == [2, 2]
        assert serial_pools.contexts == ["serial-context", "serial-context"]

    def test_pure_noise_selects_nothing(self, tmp_path, settings: dict, serial_pools: type) -> None:
        """
        Description
        -----------
        With no feature carrying position, the very first step must be rejected, so the selection is EMPTY,
        no step is recorded, and the reported pooled score is the no-behaviour score itself (nothing was
        added to it). Only one pool is opened, because the loop stops at the first rejection.

        Parameters
        ----------
        tmp_path (pathlib.Path)
            Where the design ``.npy`` is written.
        settings (dict)
            The shipped settings.
        serial_pools (type)
            The in-process executor class, whose recorded pool arguments are checked.

        Returns
        -------
        None
        """
        selection = self._select(tmp_path, settings, planted=False)
        assert selection["selected_features"] == []
        assert selection["per_step"] == []
        assert selection["rejected"]["feature"] in set(SYNTHETIC_FEATURES)
        assert selection["rejected"]["paired_gain"] - selection["rejected"]["standard_error"] <= 0.0
        assert selection["pooled_score"] == selection["no_behaviour_score"]
        assert serial_pools.requested_workers == [2]

    def test_no_external_reference_records_none(self, tmp_path, settings: dict, serial_pools: type) -> None:
        """
        Description
        -----------
        Without an external reference the result must record ``None`` for it rather than score an empty
        feature set; the selection itself is unaffected (still exactly ``["signal"]``).

        Parameters
        ----------
        tmp_path (pathlib.Path)
            Where the design ``.npy`` is written.
        settings (dict)
            The shipped settings.
        serial_pools (type)
            The in-process executor class (patched in; not inspected here).

        Returns
        -------
        None
        """
        cohort = synthetic_cohort(seed=0, planted=True)
        design_path = tmp_path / "design.npy"
        np.save(design_path, cohort["design"])
        selection = select_control_features(cohort, settings, SYNTHETIC_N_LAGS, 1, str(design_path))
        assert selection["external_reference"] is None
        assert selection["selected_features"] == ["signal"]
        assert serial_pools.requested_workers == [1, 1]


    def test_accepting_every_feature_ends_the_loop_without_a_rejection(self, tmp_path, settings: dict,
                                                                     serial_pools: type) -> None:
        """
        Description
        -----------
        A cohort whose ONLY feature is the planted one accepts it at the first step and then runs out of
        candidates, so the loop must end on exhaustion: the selection is ``["signal"]``, ``rejected`` is
        ``None`` (nothing was turned down), and exactly one pool was opened.

        Parameters
        ----------
        tmp_path (pathlib.Path)
            Where the design ``.npy`` is written.
        settings (dict)
            The shipped settings.
        serial_pools (type)
            The in-process executor class, whose recorded pool arguments are checked.

        Returns
        -------
        None
        """
        cohort = synthetic_cohort(seed=0, planted=True)
        cohort["design"] = np.ascontiguousarray(cohort["design"][:, :SYNTHETIC_N_LAGS])
        cohort["feature_names"] = ["signal"]
        design_path = tmp_path / "design.npy"
        np.save(design_path, cohort["design"])
        selection = select_control_features(cohort, settings, SYNTHETIC_N_LAGS, 1, str(design_path))
        assert selection["selected_features"] == ["signal"]
        assert selection["rejected"] is None
        assert selection["pooled_score"] == selection["per_step"][0]["pooled"]
        assert serial_pools.requested_workers == [1]


    @staticmethod
    def _stubbed_selection(monkeypatch, settings: dict, canonical: list, table: dict) -> dict:
        """
        Description
        -----------
        Run ``select_control_features`` with every model fit replaced, so the per-day scores are EXACTLY
        those tabled: the no-behaviour baseline is 0.0 on every day (``held_out_by_day`` returns zeros
        and ``pooled_macro`` returns 0.0), and each candidate's per-day scores come from
        :class:`StubbedScores`. The paired gain of a candidate is therefore the mean of its tabled scores
        and its error their sample standard error.

        Parameters
        ----------
        monkeypatch (pytest.MonkeyPatch)
            Patches the scorer, the baseline and the score wrapper on the module under test.
        settings (dict)
            The shipped settings (only hashed).
        canonical (list)
            The cohort's feature names, in canonical order.
        table (dict)
            ``{candidate_tuple: per_day_scores}`` over four days.

        Returns
        -------
        selection (dict)
            As ``select_control_features`` returns it.
        """
        monkeypatch.setattr(control_module, "_score_candidate", StubbedScores(table))
        monkeypatch.setattr(control_module, "held_out_by_day",
                            lambda _blank, positions, *_rest: np.zeros_like(positions))
        monkeypatch.setattr(control_module, "pooled_macro", lambda *_args: 0.0)
        n_rows = 4 * len(SYNTHETIC_DAYS)
        cohort = {"design": np.zeros((n_rows, len(canonical) * SYNTHETIC_N_LAGS), dtype=np.float32),
                  "positions": np.full((n_rows, 2), 0.5), "regions": np.ones(n_rows),
                  "days": np.repeat(SYNTHETIC_DAYS, 4), "feature_names": list(canonical),
                  "session_ids": ["s"], "skipped": {}}
        return select_control_features(cohort, settings, SYNTHETIC_N_LAGS, 1, "unused.npy")

    @pytest.mark.usefixtures("serial_pools")
    def test_the_largest_gain_leads_not_the_smallest_error(self, monkeypatch, settings: dict) -> None:
        """
        Description
        -----------
        Two candidates both clear the rule at the first step: ``a`` gains 0.5 with a tiny error, ``b``
        gains only 0.3 with a large error. Ranking is by GAIN, so ``a`` must be taken first; ``b`` is then
        offered on top of ``a`` and scores exactly as ``a`` did (zero paired gain), so it is rejected.

        Parameters
        ----------
        monkeypatch (pytest.MonkeyPatch)
            Patches the scorers.
        settings (dict)
            The shipped settings.

        Returns
        -------
        None
        """
        a_scores = [0.5, 0.51, 0.49, 0.5]
        selection = self._stubbed_selection(monkeypatch, settings, ["a", "b"], {
            ("a",): a_scores, ("b",): [0.1, 0.5, 0.1, 0.5], ("a", "b"): a_scores})
        assert [step["feature"] for step in selection["per_step"]] == ["a"]
        assert selection["per_step"][0]["paired_gain"] == pytest.approx(0.5)
        assert selection["rejected"]["feature"] == "b"
        assert selection["rejected"]["paired_gain"] == pytest.approx(0.0)

    @pytest.mark.parametrize(("per_day", "accepted"), [
        pytest.param([0.0, 0.0, 0.0, 0.0], False, id="gain_less_error_exactly_zero"),
        pytest.param([-0.1, 0.1, -0.2, 0.1], False, id="negative_gain_inside_its_error"),
        pytest.param([0.5, 0.5, 0.5, 0.5], True, id="half_a_nat_with_no_spread"),
    ])
    @pytest.mark.usefixtures("serial_pools")
    def test_the_acceptance_boundary(self, monkeypatch, settings: dict, per_day: list, accepted: bool) -> None:
        """
        Description
        -----------
        The rule accepts a candidate only when its paired gain LESS its own standard error is strictly
        above zero. Three exact cases for a lone feature against a 0.0 baseline: a gain of 0 with error 0
        sits ON the boundary and must be rejected; a gain of -0.025 whose error (about 0.066) exceeds its
        magnitude must be rejected (adding the error instead of subtracting it would accept it); a gain of
        0.5 with no spread must be accepted (the bar is zero, not one nat).

        Parameters
        ----------
        monkeypatch (pytest.MonkeyPatch)
            Patches the scorers.
        settings (dict)
            The shipped settings.
        per_day (list)
            The lone candidate's per-day scores.
        accepted (bool)
            Whether the rule must accept it.

        Returns
        -------
        None
        """
        selection = self._stubbed_selection(monkeypatch, settings, ["a"], {("a",): per_day})
        assert selection["selected_features"] == (["a"] if accepted else [])
        assert (selection["rejected"] is None) == accepted

    @pytest.mark.usefixtures("serial_pools")
    def test_the_baseline_and_gains_are_region_macro_scores(self, tmp_path, settings: dict) -> None:
        """
        Description
        -----------
        On the two-region cohort the no-behaviour score, each day's incumbent score and each candidate's
        per-day and pooled scores must all be REGION MACRO scores. The expected values are rebuilt
        independently: the all-zero-design and ``signal``-only held-out predictions from
        ``held_out_by_day``, scored per day and pooled with the region labels straight from
        ``macro_von_mises_logscore``. The first accepted step must be ``signal`` with exactly that paired
        gain and pooled score, and the reported no-behaviour score must be the macro one; scoring without
        the labels would shift every one of them (the guard confirms by how much).

        Parameters
        ----------
        tmp_path (pathlib.Path)
            Where the design ``.npy`` is written.
        settings (dict)
            The shipped settings.

        Returns
        -------
        None
        """
        cohort = two_region_cohort(seed=5)
        positions, regions, days = cohort["positions"], cohort["regions"], cohort["days"]
        design_path = tmp_path / "design.npy"
        np.save(design_path, cohort["design"])
        selection = select_control_features(cohort, settings, SYNTHETIC_N_LAGS, 1, str(design_path))

        blank = held_out_by_day(np.zeros_like(cohort["design"][:, :SYNTHETIC_N_LAGS]), positions, regions,
                                days, settings, SYNTHETIC_N_LAGS)
        signal = held_out_by_day(cohort["design"][:, :SYNTHETIC_N_LAGS], positions, regions, days, settings,
                                 SYNTHETIC_N_LAGS)
        incumbent = np.array([macro_reference(blank[days == day], positions[days == day], regions[days == day])
                              for day in SYNTHETIC_DAYS])
        candidate = np.array([macro_reference(signal[days == day], positions[days == day], regions[days == day])
                              for day in SYNTHETIC_DAYS])
        expected_gain, expected_error = paired_gain(candidate, incumbent)
        assert selection["no_behaviour_score"] == pytest.approx(macro_reference(blank, positions, regions),
                                                                abs=1e-9)
        first = selection["per_step"][0]
        assert first["feature"] == "signal"
        assert first["paired_gain"] == pytest.approx(expected_gain, abs=1e-9)
        assert first["standard_error"] == pytest.approx(expected_error, abs=1e-9)
        assert first["pooled"] == pytest.approx(macro_reference(signal, positions, regions), abs=1e-9)
        assert abs(macro_reference(blank, positions, None) - selection["no_behaviour_score"]) > 0.05

    def test_a_candidate_is_scored_from_the_mapped_design_by_region(self, tmp_path, settings: dict) -> None:
        """
        Description
        -----------
        The worker body maps the stored design, takes the candidate's columns, fits and predicts held out
        by day, and scores. Its per-day and pooled scores for ``("noise_a", "signal")`` must equal the
        region macro scores of ``held_out_by_day`` on the same columns, and it must hand back the candidate
        it was given, unchanged.

        Parameters
        ----------
        tmp_path (pathlib.Path)
            Where the design ``.npy`` is written.
        settings (dict)
            The shipped settings.

        Returns
        -------
        None
        """
        cohort = two_region_cohort(seed=6)
        positions, regions, days = cohort["positions"], cohort["regions"], cohort["days"]
        design_path = tmp_path / "design.npy"
        np.save(design_path, cohort["design"])
        columns = feature_columns(cohort["feature_names"], ["noise_a", "signal"], SYNTHETIC_N_LAGS)
        candidate, per_day, pooled = control_module._score_candidate(
            (str(design_path), columns, positions, regions, days, settings, SYNTHETIC_N_LAGS, 2,
             ("noise_a", "signal")))
        predicted = held_out_by_day(cohort["design"][:, columns], positions, regions, days, settings,
                                    SYNTHETIC_N_LAGS)
        assert candidate == ("noise_a", "signal")
        np.testing.assert_allclose(per_day, [macro_reference(predicted[days == day], positions[days == day],
                                                             regions[days == day]) for day in SYNTHETIC_DAYS],
                                   atol=1e-9)
        assert pooled == pytest.approx(macro_reference(predicted, positions, regions), abs=1e-9)


class TestAssembleCohort:
    """Per-session blocks are stacked in order, each USV keeps its day, and a bad cohort is refused."""

    @staticmethod
    def _block(session_id: str, n_kept: int, value: float, names: list) -> dict:
        """
        Description
        -----------
        A per-session block in the shape ``session_vocalization_design`` returns, filled with a constant so
        the stacked rows reveal which session they came from.

        Parameters
        ----------
        session_id (str)
            The session directory basename; its first eight characters are the day.
        n_kept (int)
            USVs the session contributes.
        value (float)
            Constant filling the design, positions and regions.
        names (list)
            The resolved feature names.

        Returns
        -------
        block (dict)
            ``session_id``, ``design`` (``(n_kept, 2)``), ``positions``, ``regions``, ``feature_names``,
            ``n_kept`` and ``vocalization_start``.
        """
        return {"session_id": session_id, "design": np.full((n_kept, 2), value, dtype=np.float32),
                "positions": np.full((n_kept, 2), value / 10.0), "regions": np.full(n_kept, value),
                "feature_names": list(names), "n_kept": n_kept, "vocalization_start": np.arange(n_kept, dtype=float)}

    def _patch(self, monkeypatch, blocks: dict) -> None:
        """
        Description
        -----------
        Replace ``session_vocalization_design`` with a lookup into ``blocks`` keyed by session directory,
        checking along the way that the worker shim passes the settings and lag count through unchanged.

        Parameters
        ----------
        monkeypatch (pytest.MonkeyPatch)
            Patches the per-session assembler on the module under test.
        blocks (dict)
            ``{session_directory: block}``.

        Returns
        -------
        None
        """
        def lookup(session_directory: str, run_settings: dict, n_lags: int) -> dict:
            """
            Description
            -----------
            Return the prepared block for ``session_directory`` after checking the forwarded arguments.

            Parameters
            ----------
            session_directory (str)
                The session directory the shim forwarded.
            run_settings (dict)
                The settings the shim forwarded; must be the sentinel dict.
            n_lags (int)
                The lag count the shim forwarded; must be 7.

            Returns
            -------
            block (dict)
                ``blocks[session_directory]``.
            """
            assert run_settings == {"sentinel": True}
            assert n_lags == 7
            return blocks[session_directory]

        monkeypatch.setattr(control_module, "session_vocalization_design", lookup)

    def test_blocks_are_stacked_in_order_with_days_and_skips(self, monkeypatch, serial_pools: type) -> None:
        """
        Description
        -----------
        Three listed sessions: two contribute 2 and 3 USVs (two different days), one is skipped with a
        reason. The result must stack the two contributing blocks in LISTED order (2 rows of 1.0 then 3 of
        2.0), give every USV the first eight characters of its session id as its day, list only the
        contributing session ids, carry the skip reason under the skipped session's id, keep the shared
        feature names, and open one pool with the requested worker count.

        Parameters
        ----------
        monkeypatch (pytest.MonkeyPatch)
            Patches the per-session assembler.
        serial_pools (type)
            The in-process executor class, whose recorded pool arguments are checked.

        Returns
        -------
        None
        """
        names = ["self.speed", "head-head"]
        self._patch(monkeypatch, {
            "/data/20250101_100000": self._block("20250101_100000", 2, 1.0, names),
            "/data/20250102_100000": {"session_id": "20250102_100000", "reason": "3 tracked animals"},
            "/data/20250103_100000": self._block("20250103_100000", 3, 2.0, names)})
        cohort = assemble_cohort(["/data/20250101_100000", "/data/20250102_100000", "/data/20250103_100000"],
                                 {"sentinel": True}, 7, 3)
        np.testing.assert_array_equal(cohort["design"][:, 0], [1.0, 1.0, 2.0, 2.0, 2.0])
        np.testing.assert_allclose(cohort["positions"][:, 1], [0.1, 0.1, 0.2, 0.2, 0.2])
        np.testing.assert_array_equal(cohort["regions"], [1.0, 1.0, 2.0, 2.0, 2.0])
        assert cohort["days"].tolist() == ["20250101"] * 2 + ["20250103"] * 3
        assert cohort["session_ids"] == ["20250101_100000", "20250103_100000"]
        assert cohort["skipped"] == {"20250102_100000": "3 tracked animals"}
        assert cohort["feature_names"] == names
        assert serial_pools.requested_workers == [3]
        assert serial_pools.contexts == ["serial-context"]

    def test_a_cohort_with_no_contributing_session_is_refused(self, monkeypatch, serial_pools: type) -> None:
        """
        Description
        -----------
        When every listed session is skipped there is nothing to fit, so the assembly must raise
        ``ValueError`` saying no session contributed, rather than return an empty design for the fit to
        fail on later.

        Parameters
        ----------
        monkeypatch (pytest.MonkeyPatch)
            Patches the per-session assembler.
        serial_pools (type)
            The in-process executor class (patched in; not inspected here).

        Returns
        -------
        None
        """
        self._patch(monkeypatch, {"/data/20250101_100000": {"session_id": "20250101_100000",
                                                            "reason": "0 emitter vocalizations"}})
        with pytest.raises(ValueError, match="no session contributed USVs"):
            assemble_cohort(["/data/20250101_100000"], {"sentinel": True}, 7, 1)
        assert serial_pools.requested_workers == [1]

    def test_sessions_resolving_different_features_are_refused(self, monkeypatch, serial_pools: type) -> None:
        """
        Description
        -----------
        Two sessions resolving the same features in a DIFFERENT order would stack design columns that do
        not line up, so the assembly must raise ``ValueError`` naming the offending (second) session.

        Parameters
        ----------
        monkeypatch (pytest.MonkeyPatch)
            Patches the per-session assembler.
        serial_pools (type)
            The in-process executor class, whose recorded pool arguments are checked.

        Returns
        -------
        None
        """
        self._patch(monkeypatch, {
            "/data/20250101_100000": self._block("20250101_100000", 2, 1.0, ["a", "b"]),
            "/data/20250102_100000": self._block("20250102_100000", 2, 2.0, ["b", "a"])})
        with pytest.raises(ValueError, match="20250102_100000 resolved a different feature set"):
            assemble_cohort(["/data/20250101_100000", "/data/20250102_100000"], {"sentinel": True}, 7, 1)
        assert serial_pools.requested_workers == [1]
