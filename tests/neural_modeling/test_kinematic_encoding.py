"""
@author: bartulem
Unit tests for the selection, null and readout functions of ``kinematic_encoding``.

Coverage: the per-feature screen (``screen_features``), greedy forward selection (``forward_select``), the
frozen-filter shift null (``frozen_null_scores``), the shape descriptors of a fitted filter
(``filter_descriptors``), the spike-triggered average (``spike_triggered_average``), and the guards and
corner paths of ``fit_quiet_model`` that the other test files leave untouched. Every test is built
around a synthetic case whose answer is known in advance: a Bernoulli unit driven by planted filters on
some feature columns and blind to the others, a unit whose tuning flips sign between sessions, labels
that are independent of the frozen predictor, a hand-built filter, and spikes planted at a known lag.
The settings are the shipped ``neural_modeling_settings.json``, deep-copied and shrunk (fewer splines,
fewer shuffles, a shorter guard band, a liberal screen alpha matched to the small shuffle count) so
each test runs in a few seconds.
"""

from __future__ import annotations

import copy
import json
import pathlib

import numpy as np
import pytest

from usv_playpen.modeling.jax_group_elastic_net import GroupElasticNetGLM
from usv_playpen.modeling.modeling_bases_functions import gam_bspline_basis
from usv_playpen.neural_modeling import kinematic_encoding
from usv_playpen.neural_modeling.deviance_metrics import (
    frozen_model_null_scores,
    frozen_model_score,
)
from usv_playpen.neural_modeling.kinematic_encoding import (
    FrameFilterModel,
    block_resample_anchors,
    filter_band,
    filter_descriptors,
    fit_quiet_model,
    forward_select,
    frozen_null_scores,
    lag_basis,
    linear_predictor_at_frames,
    quiet_block_sessions,
    screen_features,
    spike_triggered_average,
)
from usv_playpen.neural_modeling.neural_design_assembly import (
    lagged_design,
    spike_labels_at_frames,
    subsample_quiet_anchors,
)
from usv_playpen.neural_modeling.shift_null_inference import (
    sample_circular_shift,
    shifted_spike_frames,
)

SETTINGS_PATH = (
    pathlib.Path(__file__).resolve().parents[2]
    / "src"
    / "usv_playpen"
    / "_parameter_settings"
    / "neural_modeling_settings.json"
)
FPS = 30.0
N_LAGS = 15
N_FRAMES = 9000
SCREEN_SHUFFLES = 40
FEATURE_NAMES = ["planted_strong", "planted_weak", "noise"]
NULL_CALIBRATION = {"scheme": "crossfit", "n_blocks": 5}


def _silent(*_args) -> None:
    """
    Description
    -----------
    Message sink that discards every progress line, for tests that do not inspect the output.

    Parameters
    ----------
    *_args (tuple)
        Whatever the caller would have printed; ignored.

    Returns
    -------
    None
    """


def _settings() -> dict:
    """
    Description
    -----------
    Load the shipped neural-modeling settings and shrink them for fast synthetic tests.

    The temporal basis keeps its shipped type (``"bspline"`` with the spike-end correction) but uses six
    splines over the 15-lag history; the solver cap drops to 2,000 iterations; the shift-null guard band
    shrinks to 10 s so a 300 s synthetic session has room for it; the null calibration runs exact
    (``null_calibration_bins`` 0) because the binned path is meant for sessions far longer than these;
    and the screen draws ``SCREEN_SHUFFLES`` shuffles with ``screen_alpha`` raised to 0.3. The last pair
    goes together: with 40 draws an empirical p floors at 1/41 = 0.0244, and the Bonferroni threshold
    ``screen_alpha / n_features`` = 0.3 / 3 = 0.1 sits above that floor, so a genuinely driven feature
    can pass while the threshold still rejects most of the null.

    Parameters
    ----------
    None

    Returns
    -------
    settings (dict)
        A deep copy of the shipped settings with the overrides above applied.
    """
    with SETTINGS_PATH.open() as handle:
        settings = copy.deepcopy(json.load(handle))
    encoding = settings["kinematic_encoding"]
    encoding["temporal_basis"]["n_splines"] = 6
    encoding["max_train_total"] = 20000
    encoding["chunk_rows"] = 1000
    encoding["solver"]["max_iter"] = 2000
    encoding["solver"]["null_calibration_bins"] = 0
    encoding["feature_selection"]["screen_alpha"] = 0.3
    settings["null"]["screen_n_shuffles"] = SCREEN_SHUFFLES
    settings["null"]["shuffle_guard_seconds"] = 10.0
    return settings


def _planted_session(
    seed: int, gains: list, n_frames: int = N_FRAMES, baseline: float = -3.0
) -> dict:
    """
    Description
    -----------
    Build one synthetic quiet session whose spikes are Bernoulli draws from a known logistic model.

    Every feature column is smoothed white noise (a 10-frame boxcar), z-scored, and independent of the
    others. The log-odds are ``baseline + sum_j gains[j] * (w * x_j)``, where ``w`` is the same causal
    filter for every feature: weight 1/5 on each of the five frames from one to five frames before the
    anchor, zero elsewhere. A zero gain therefore makes that column a pure-noise feature the unit is
    blind to; a negative gain inverts the tuning.

    Parameters
    ----------
    seed (int)
        Seed for ``np.random.default_rng``.
    gains (list)
        One log-odds gain per feature column.
    n_frames (int)
        Session length in frames; defaults to ``N_FRAMES``.
    baseline (float)
        Log-odds baseline; defaults to -3.0 (about a 5 % spike probability per frame).

    Returns
    -------
    session (dict)
        ``feature_time_series`` ``(n_frames, len(gains))``, ``n_frames``, ``fps``, ``spike_frames`` and
        ``quiet`` anchor frames (every frame with a full history, minus the last five).
    """
    generator = np.random.default_rng(seed)
    kernel = np.ones(10) / 10.0
    features = np.empty((n_frames, len(gains)))
    for column in range(len(gains)):
        smooth = np.convolve(generator.normal(size=n_frames + 20), kernel, mode="same")[
            10 : n_frames + 10
        ]
        features[:, column] = (smooth - smooth.mean()) / smooth.std()
    planted = np.zeros(N_LAGS)
    planted[-6:-1] = 1.0 / 5.0
    eta = np.full(n_frames, baseline)
    for column, gain in enumerate(gains):
        eta = (
            eta
            + gain
            * np.convolve(features[:, column], planted[::-1], mode="full")[:n_frames]
        )
    spikes = np.flatnonzero(generator.random(n_frames) < 1.0 / (1.0 + np.exp(-eta)))
    return {
        "feature_time_series": features,
        "n_frames": n_frames,
        "fps": FPS,
        "spike_frames": spikes,
        "quiet": np.arange(N_LAGS, n_frames - 5),
    }


@pytest.fixture(scope="module")
def planted_pool() -> dict:
    """
    Description
    -----------
    Three pool sessions of one unit driven strongly by column 0 (gain 1.0), weakly by column 1 (gain 0.6)
    and not at all by column 2 (gain 0), so the screen and selection have a known answer: columns 0 and 1
    belong in the model, in that order, and column 2 does not.

    Parameters
    ----------
    None

    Returns
    -------
    pool (dict)
        ``{"s0": session, "s1": session, "s2": session}`` built by ``_planted_session``.
    """
    return {
        f"s{index}": _planted_session(10 + index, [1.0, 0.6, 0.0]) for index in range(3)
    }


@pytest.fixture(scope="module")
def screened(planted_pool) -> dict:
    """
    Description
    -----------
    Run ``screen_features`` once on the planted pool and keep both its rows and its progress lines, so
    several tests can check different properties of the same screen without repeating the fits.

    Parameters
    ----------
    planted_pool (dict)
        The module-scoped planted sessions.

    Returns
    -------
    result (dict)
        ``rows`` (the screen's return value) and ``messages`` (every line it reported).
    """
    messages = []
    rows = screen_features(
        planted_pool,
        list(planted_pool),
        N_LAGS,
        np.random.default_rng(0),
        _settings(),
        FEATURE_NAMES,
        message_output=messages.append,
    )
    return {"rows": rows, "messages": messages}


@pytest.fixture(scope="module")
def selected(planted_pool) -> dict:
    """
    Description
    -----------
    Run ``forward_select`` once on the planted pool with every column offered as a survivor, keeping the
    selection, the path and the progress lines.

    Parameters
    ----------
    planted_pool (dict)
        The module-scoped planted sessions.

    Returns
    -------
    result (dict)
        ``selected`` (accepted column indices), ``path`` (one dict per step) and ``messages``.
    """
    messages = []
    chosen, path = forward_select(
        planted_pool,
        list(planted_pool),
        [0, 1, 2],
        N_LAGS,
        np.random.default_rng(0),
        _settings(),
        FEATURE_NAMES,
        message_output=messages.append,
    )
    return {"selected": chosen, "path": path, "messages": messages}


class TestScreenFeatures:
    """The per-feature screen: each column fitted alone under the inner split, judged against its own
    frozen-filter null, a positive mean score and a positive calibration slope."""

    def test_the_planted_features_pass_and_the_blind_feature_fails(self, screened):
        """Columns 0 and 1 drive the unit, column 2 does not: the screen must pass exactly the first two,
        with positive held-out scores and slopes, and fail the third with a p-value above threshold."""
        rows = screened["rows"]
        assert [row["survived"] for row in rows] == [True, True, False]
        for row in rows[:2]:
            assert row["score"] > 0.01
            assert row["slope"] > 0.0
        assert rows[0]["score"] > rows[1]["score"]
        threshold = (
            _settings()["kinematic_encoding"]["feature_selection"]["screen_alpha"] / 3
        )
        assert rows[2]["p"] > threshold

    def test_a_driven_feature_sits_on_the_empirical_floor(self, screened):
        """A strongly driven feature beats every one of its null draws, so its p-value is exactly the
        empirical floor 1 / (n + 1) and it is flagged as at the floor."""
        row = screened["rows"][0]
        assert row["p"] == pytest.approx(1.0 / (SCREEN_SHUFFLES + 1))
        assert row["at_floor"] is True

    def test_rows_report_the_fold_values_they_were_decided_on(self, screened):
        """Each row carries one score and one slope per inner fold (three pool sessions, three folds), and
        the decided score and slope are exactly the means of those per-fold values."""
        for index, row in enumerate(screened["rows"]):
            assert row["feature"] == index
            assert row["name"] == FEATURE_NAMES[index]
            assert len(row["fold_scores"]) == 3
            assert len(row["fold_slopes"]) == 3
            assert row["score"] == pytest.approx(np.mean(row["fold_scores"]))
            assert row["slope"] == pytest.approx(np.mean(row["fold_slopes"]))
            assert row["converged"] is True

    def test_one_progress_line_per_feature_with_its_verdict(self, screened):
        """The screen reports one line per feature naming it and its verdict, so a run log shows which
        features passed without reading the artifact."""
        lines = [
            line for line in screened["messages"] if line.lstrip().startswith("screen ")
        ]
        assert len(lines) == 3
        for line, row, name in zip(lines, screened["rows"], FEATURE_NAMES, strict=True):
            assert name in line
            assert ("PASS" in line) == row["survived"]

    @pytest.mark.parametrize(
        ("require_positive_slope", "expected_survival"), [(True, False), (False, True)]
    )
    def test_a_tuning_that_flips_between_sessions_is_caught_only_by_the_slope_gate(
        self, require_positive_slope, expected_survival
    ):
        """
        Description
        -----------
        The calibrated score grows with the square of the slope and is blind to its sign. One pool session
        is driven by +feature and the other by -feature, so on both inner folds the filter learned on one
        session predicts the other BACKWARDS: the score is still clearly positive and beats its null, but
        the slope is negative. With the slope gate on the feature must fail; with it off it passes. The
        gate is the only thing that separates the two outcomes.

        Parameters
        ----------
        require_positive_slope (bool)
            Value given to ``screen_require_positive_slope``.
        expected_survival (bool)
            Whether the feature should survive under that setting.

        Returns
        -------
        None
        """
        pool = {
            "plus": _planted_session(30, [1.0]),
            "minus": _planted_session(31, [-1.0]),
        }
        settings = _settings()
        settings["kinematic_encoding"]["feature_selection"][
            "screen_require_positive_slope"
        ] = require_positive_slope
        settings["kinematic_encoding"]["feature_selection"]["screen_alpha"] = 0.1
        (row,) = screen_features(
            pool,
            ["plus", "minus"],
            N_LAGS,
            np.random.default_rng(0),
            settings,
            ["flipped"],
            message_output=_silent,
        )
        assert row["score"] > 0.05
        assert row["slope"] < -0.5
        assert all(slope < 0.0 for slope in row["fold_slopes"])
        assert row["p"] < 0.1
        assert row["survived"] is expected_survival


class TestForwardSelect:
    """Greedy forward selection on pool quiet anchors under the paired 1SE acceptance rule."""

    def test_the_planted_features_are_selected_strongest_first(self, selected):
        """The strongly driven column anchors the model, the weakly driven one is accepted next, and the
        blind column is rejected, which ends the search."""
        assert selected["selected"] == [0, 1]
        assert [step["decision"] for step in selected["path"]] == [
            "ANCHOR",
            "ACCEPT",
            "REJECT",
        ]
        assert [step["candidate"] for step in selected["path"]] == FEATURE_NAMES
        assert [step["step"] for step in selected["path"]] == [0, 1, 2]

    def test_each_decision_follows_the_paired_improvement_and_its_se(self, selected):
        """The accepted step improves on its incumbent by more than the SE of the paired per-fold
        differences, the rejected one does not, and both are recomputable from the stored fold scores."""
        anchor, accepted, rejected = selected["path"]
        assert np.isnan(anchor["improvement"])
        assert np.isnan(anchor["standard_error"])
        for step, incumbent in ((accepted, anchor), (rejected, accepted)):
            differences = np.asarray(step["fold_scores"]) - np.asarray(
                incumbent["fold_scores"]
            )
            assert step["improvement"] == pytest.approx(differences.mean())
            assert step["standard_error"] == pytest.approx(
                differences.std(ddof=1) / np.sqrt(differences.size)
            )
            assert step["mean"] == pytest.approx(np.mean(step["fold_scores"]))
        assert accepted["improvement"] > accepted["standard_error"] > 0.0
        assert rejected["improvement"] <= rejected["standard_error"]
        assert accepted["mean"] > anchor["mean"] > 0.0

    def test_progress_lines_name_each_step(self, selected):
        """One line per path step, carrying the step's decision and candidate."""
        assert len(selected["messages"]) == len(selected["path"])
        for line, step in zip(selected["messages"], selected["path"], strict=True):
            assert step["decision"] in line
            assert step["candidate"] in line

    def test_a_pure_noise_unit_selects_nothing(self):
        """A unit blind to every column has no feature with a positive mean held-out score, so selection
        returns an empty model and an empty path rather than anchoring on noise."""
        pool = {
            f"s{index}": _planted_session(40 + index, [0.0, 0.0, 0.0])
            for index in range(3)
        }
        chosen, path = forward_select(
            pool,
            list(pool),
            [0, 1, 2],
            N_LAGS,
            np.random.default_rng(0),
            _settings(),
            FEATURE_NAMES,
            message_output=_silent,
        )
        assert chosen == []
        assert path == []

    def test_a_single_survivor_is_anchored_without_a_search(self, planted_pool):
        """With one survivor there is nothing to add: it is anchored and the path holds only the anchor."""
        chosen, path = forward_select(
            planted_pool,
            list(planted_pool),
            [1],
            N_LAGS,
            np.random.default_rng(0),
            _settings(),
            FEATURE_NAMES,
            message_output=_silent,
        )
        assert chosen == [1]
        assert len(path) == 1
        assert path[0]["decision"] == "ANCHOR"
        assert path[0]["candidate"] == "planted_weak"
        assert path[0]["mean"] > 0.0


class TestFrozenNullScores:
    """The shift null of a frozen predictor: circularly shift the spikes, relabel, re-score, never refit."""

    N_NULL_FRAMES = 3000
    GUARD = 10.0

    @classmethod
    def _case(cls, driven):
        """
        Description
        -----------
        Build a frozen predictor and a spike train over a 100 s session, either independent of each other
        or with the spikes driven by the predictor.

        Parameters
        ----------
        driven (bool)
            When True the spikes follow ``logistic(-3 + eta)``; when False they are a constant 5 % per
            frame, independent of ``eta``.

        Returns
        -------
        case (tuple)
            ``(eta, frames, spike_frames)``: a standard-normal predictor at every frame, the frames
            themselves, and the spike-frame train.
        """
        generator = np.random.default_rng(1)
        eta = generator.normal(size=cls.N_NULL_FRAMES)
        probability = (
            1.0 / (1.0 + np.exp(-(-3.0 + eta)))
            if driven
            else np.full(cls.N_NULL_FRAMES, 0.05)
        )
        spikes = np.flatnonzero(generator.random(cls.N_NULL_FRAMES) < probability)
        return eta, np.arange(cls.N_NULL_FRAMES), spikes

    def _null(self, eta, frames, spikes, n_shuffles, seed, block_megabytes=32.0):
        """
        Description
        -----------
        Call ``frozen_null_scores`` with this class's session length, guard and calibration.

        Parameters
        ----------
        eta (np.ndarray)
            Frozen predictor.
        frames (np.ndarray)
            Scored frames.
        spikes (np.ndarray)
            Spike-frame train.
        n_shuffles (int)
            Number of draws.
        seed (int)
            Seed for the generator the shifts are drawn from.
        block_megabytes (float)
            Memory budget for one block of draws' labels.

        Returns
        -------
        null (np.ndarray)
            ``n_shuffles`` null scores.
        """
        return frozen_null_scores(
            eta,
            frames,
            spikes,
            self.N_NULL_FRAMES,
            FPS,
            n_shuffles,
            np.random.default_rng(seed),
            self.GUARD,
            30,
            0,
            NULL_CALIBRATION,
            block_megabytes,
        )

    def test_it_is_the_frozen_score_of_each_shifted_train(self):
        """Each draw is the frozen-model score of the labels of ONE circularly shifted spike train, the
        shifts drawn in order from the generator: rebuilding that by hand gives the same numbers."""
        eta, frames, spikes = self._case(driven=False)
        null = self._null(eta, frames, spikes, 12, seed=2)
        generator = np.random.default_rng(2)
        shifts = [
            sample_circular_shift(generator, self.N_NULL_FRAMES, FPS, self.GUARD)
            for _ in range(12)
        ]
        labels = np.column_stack(
            [
                spike_labels_at_frames(
                    shifted_spike_frames(spikes, shift, FPS, self.N_NULL_FRAMES),
                    frames,
                    self.N_NULL_FRAMES,
                )
                for shift in shifts
            ]
        )
        expected = frozen_model_null_scores(eta, labels, 30, 0, NULL_CALIBRATION)
        assert null.shape == (12,)
        np.testing.assert_allclose(null, expected)

    def test_splitting_the_draws_into_memory_blocks_changes_nothing(self):
        """The block size is a memory budget, not a statistical choice: forcing blocks of three draws must
        give exactly the scores of one block holding all of them, including a short final block."""
        eta, frames, spikes = self._case(driven=False)
        whole = self._null(eta, frames, spikes, 10, seed=3)
        blocked = self._null(
            eta, frames, spikes, 10, seed=3, block_megabytes=frames.size * 8 * 3 / (1024 * 1024)
        )
        np.testing.assert_allclose(blocked, whole)

    def test_an_unrelated_predictor_has_a_null_centred_on_zero(self):
        """When the spikes are independent of the predictor every shifted train is too, so the
        cross-fitted explained deviance of each draw is noise around zero: the null mean is within a few
        standard errors of zero and no draw strays far from it."""
        eta, frames, spikes = self._case(driven=False)
        null = self._null(eta, frames, spikes, 60, seed=4)
        assert np.all(np.isfinite(null))
        assert abs(null.mean()) < 4.0 * null.std(ddof=1) / np.sqrt(null.size) + 1e-4
        assert np.max(np.abs(null)) < 0.01

    def test_a_driven_unit_scores_far_above_its_own_null(self):
        """Shifting destroys the alignment between spikes and predictor, so even for a strongly driven
        unit the null stays near zero while the unshifted score is an order of magnitude beyond every
        draw."""
        eta, frames, spikes = self._case(driven=True)
        labels = spike_labels_at_frames(spikes, frames, self.N_NULL_FRAMES)
        observed, slope = frozen_model_score(
            eta,
            labels,
            np.zeros(labels.size, dtype=np.int64),
            30,
            "bernoulli",
            1.0,
            NULL_CALIBRATION,
        )
        null = self._null(eta, frames, spikes, 40, seed=5)
        assert slope > 0.5
        assert observed > 0.05
        assert observed > 10.0 * np.max(np.abs(null))


class TestFitQuietModelCornerPaths:
    """Guards and corner paths of ``fit_quiet_model`` not reached by the screen or selection tests."""

    @staticmethod
    def _raw_encoding():
        """
        Description
        -----------
        The shrunk shipped ``kinematic_encoding`` block switched to the raw per-frame path, where the
        per-session baselines are backfitted by Newton steps rather than carried as indicator columns.

        Parameters
        ----------
        None

        Returns
        -------
        encoding (dict)
            A ``kinematic_encoding`` block with ``temporal_basis.type`` set to ``"none"`` and the
            first-difference smoothness lowered from 1000 to 10, at which these small raw fits converge
            well inside the 2,000-iteration cap (at 1000 they exhaust it).
        """
        encoding = _settings()["kinematic_encoding"]
        encoding["temporal_basis"]["type"] = "none"
        encoding["raw_lag_penalties"]["lambda_smooth"] = 10.0
        return encoding

    def test_a_subsampling_correction_other_than_the_log_prior_offset_is_refused(self):
        """The case-control offset is the only correction implemented; declaring importance weighting
        must raise rather than run the offset under another name."""
        encoding = _settings()["kinematic_encoding"]
        encoding["subsampling_correction"] = "importance_weights"
        with pytest.raises(ValueError, match="log-prior offset"):
            fit_quiet_model(
                {"a": _planted_session(50, [1.0], n_frames=600)},
                ["a"],
                [0],
                N_LAGS,
                np.random.default_rng(0),
                encoding,
                message_output=_silent,
            )

    def test_more_sessions_than_one_indicator_group_can_hold_is_refused(self):
        """On the spline path the k - 1 session indicators share one column group of basis width. Four
        splines under the spike-end correction leave three columns, so five training sessions (four
        indicators) cannot be represented and must raise instead of silently dropping a baseline."""
        encoding = _settings()["kinematic_encoding"]
        encoding["temporal_basis"]["n_splines"] = 4
        sessions = {
            f"s{index}": _planted_session(60 + index, [1.0], n_frames=600)
            for index in range(5)
        }
        with pytest.raises(ValueError, match="indicator columns"):
            fit_quiet_model(
                sessions,
                list(sessions),
                [0],
                N_LAGS,
                np.random.default_rng(0),
                encoding,
                message_output=_silent,
            )

    def test_one_newton_step_per_sweep_reaches_the_same_backfit(self):
        """The baselines are a fixed point of the sweeps, so capping the inner Newton loop at one step
        (which never meets its step tolerance) only slows the approach: the filter and intercept match the
        fit that runs Newton to convergence inside every sweep."""
        sessions = {
            "a": _planted_session(70, [1.0], n_frames=2000, baseline=-3.0),
            "b": _planted_session(71, [1.0], n_frames=2000, baseline=-2.0),
        }
        fits = {}
        for name, steps in (("one_step", 1), ("converged", 25)):
            encoding = self._raw_encoding()
            encoding["session_intercepts"]["newton_max_steps"] = steps
            encoding["session_intercepts"]["max_sweeps"] = 50
            encoding["session_intercepts"]["tolerance"] = 1e-5
            fits[name], _rate = fit_quiet_model(
                sessions,
                ["a", "b"],
                [0],
                N_LAGS,
                np.random.default_rng(0),
                encoding,
                message_output=_silent,
            )
            assert fits[name].converged_
        np.testing.assert_allclose(
            fits["one_step"].coef_, fits["converged"].coef_, rtol=1e-4, atol=1e-6
        )
        assert fits["one_step"].intercept_ == pytest.approx(
            fits["converged"].intercept_, abs=1e-5
        )

    def test_a_training_session_without_quiet_anchors_contributes_nothing(self):
        """A training session with no quiet anchors has no rows, so its Newton curvature is zero and its
        baseline is left alone; it carries no weight in the returned intercept. The fit must therefore be
        identical to fitting the other session alone, rather than failing on an empty block."""
        session = _planted_session(80, [1.0])
        empty = {**session, "quiet": np.array([], dtype=np.int64)}
        alone, rate_alone = fit_quiet_model(
            {"a": session},
            ["a"],
            [0],
            N_LAGS,
            np.random.default_rng(0),
            self._raw_encoding(),
            message_output=_silent,
        )
        paired, rate_paired = fit_quiet_model(
            {"a": session, "empty": empty},
            ["a", "empty"],
            [0],
            N_LAGS,
            np.random.default_rng(0),
            self._raw_encoding(),
            message_output=_silent,
        )
        np.testing.assert_allclose(np.asarray(paired.coef_), np.asarray(alone.coef_))
        assert paired.intercept_ == pytest.approx(alone.intercept_)
        assert rate_paired == pytest.approx(rate_alone)


class TestFilterDescriptors:
    """Shape summaries of a fitted filter, checked against values worked out by hand."""

    def test_a_hand_built_filter_gives_the_worked_out_descriptors(self):
        """
        Description
        -----------
        Five lags at 10 frames per second, lag age counting back from the LAST column (age 0.4, 0.3, 0.2,
        0.1, 0.0 s). For ``[0, 0, 1, 3, 0]``: the largest weight is 3 at age 0.1 s; norm sqrt(1 + 9);
        roughness is the norm of the differences ``[0, 1, 2, -3]``, sqrt(14); the centre of mass of
        ``|w|`` is (1 * 0.2 + 3 * 0.1) / 4 = 0.125 s.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        (row,) = filter_descriptors(
            np.array([0.0, 0.0, 1.0, 3.0, 0.0]), ["speed"], 5, 10.0
        )
        assert row["feature"] == "speed"
        assert row["peak_lag_seconds"] == pytest.approx(0.1)
        assert row["peak_weight"] == pytest.approx(3.0)
        assert row["norm"] == pytest.approx(np.sqrt(10.0))
        assert row["roughness"] == pytest.approx(np.sqrt(14.0))
        assert row["centre_of_mass_seconds"] == pytest.approx(0.125)

    def test_the_peak_is_by_magnitude_and_keeps_its_sign(self):
        """A negative lobe larger than the positive one is the peak, reported with its negative weight,
        and the blocks are split feature-major so the second feature's descriptors are its own."""
        coefficients = np.array([-4.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 2.0])
        first, second = filter_descriptors(coefficients, ["a", "b"], 5, 10.0)
        assert first["peak_weight"] == pytest.approx(-4.0)
        assert first["peak_lag_seconds"] == pytest.approx(0.4)
        assert first["centre_of_mass_seconds"] == pytest.approx(
            (4.0 * 0.4 + 1.0 * 0.0) / 5.0
        )
        assert second["feature"] == "b"
        assert second["peak_weight"] == pytest.approx(2.0)
        assert second["peak_lag_seconds"] == pytest.approx(0.0)
        assert second["centre_of_mass_seconds"] == pytest.approx(0.0)

    def test_an_all_zero_filter_has_no_centre_of_mass(self):
        """A filter with no weight has no defined centre of mass, which is reported as NaN rather than a
        division by zero; its norm and roughness are zero."""
        (row,) = filter_descriptors(np.zeros(4), ["flat"], 4, 10.0)
        assert np.isnan(row["centre_of_mass_seconds"])
        assert row["norm"] == 0.0
        assert row["roughness"] == 0.0


class TestSpikeTriggeredAverage:
    """The spike-triggered average over the history window, on quiet anchors."""

    LAG = 4
    N_STA_LAGS = 10

    @classmethod
    def _session(cls, seed):
        """
        Description
        -----------
        Build a session whose spikes fire exactly ``LAG`` frames after the white-noise feature exceeds 1,
        so the triggered average must peak at the column ``LAG`` frames before the anchor.

        Parameters
        ----------
        seed (int)
            Seed for ``np.random.default_rng``.

        Returns
        -------
        session (dict)
            ``feature_time_series`` ``(2000, 2)`` (column 1 is an unrelated white-noise feature),
            ``n_frames``, ``spike_frames`` and ``quiet`` anchor frames.
        """
        generator = np.random.default_rng(seed)
        n_frames = 2000
        features = generator.normal(size=(n_frames, 2))
        spikes = np.flatnonzero(features[:, 0] > 1.0) + cls.LAG
        spikes = spikes[spikes < n_frames]
        return {
            "feature_time_series": features,
            "n_frames": n_frames,
            "spike_frames": spikes,
            "quiet": np.arange(cls.N_STA_LAGS, n_frames),
        }

    def test_the_planted_lag_is_recovered(self):
        """The triggered average of the driving feature peaks at the column ``LAG`` frames before the
        anchor (the design puts the anchor in the LAST column) and sits above 1 there, the mean of a
        standard normal above 1; the unrelated feature shows no such peak."""
        sessions = {"a": self._session(0), "b": self._session(1)}
        sta = spike_triggered_average(
            sessions, ["a", "b"], [0, 1], self.N_STA_LAGS, 500
        )
        assert sta["difference"].shape == (2, self.N_STA_LAGS)
        assert int(np.argmax(sta["difference"][0])) == self.N_STA_LAGS - 1 - self.LAG
        assert sta["difference"][0, self.N_STA_LAGS - 1 - self.LAG] > 1.2
        assert np.max(np.abs(sta["difference"][1])) < 0.25

    def test_it_equals_the_direct_average_and_counts_every_anchor(self):
        """The chunked accumulation equals the mean of the lagged design over spike anchors minus its mean
        over all anchors, pooled over sessions, whatever the chunk size; the counts are the pooled
        totals."""
        sessions = {"a": self._session(2), "b": self._session(3)}
        designs, labels = [], []
        for session in sessions.values():
            designs.append(
                lagged_design(
                    session["feature_time_series"][:, [0, 1]],
                    session["quiet"],
                    self.N_STA_LAGS,
                )
            )
            labels.append(
                spike_labels_at_frames(
                    session["spike_frames"], session["quiet"], session["n_frames"]
                )
            )
        design, label = np.vstack(designs), np.concatenate(labels)
        triggered = design[label > 0.5].mean(axis=0).reshape(2, self.N_STA_LAGS)
        baseline = design.mean(axis=0).reshape(2, self.N_STA_LAGS)
        for chunk_rows in (7, 100_000):
            sta = spike_triggered_average(
                sessions, ["a", "b"], [0, 1], self.N_STA_LAGS, chunk_rows
            )
            np.testing.assert_allclose(sta["triggered"], triggered)
            np.testing.assert_allclose(sta["baseline"], baseline)
            np.testing.assert_allclose(sta["difference"], triggered - baseline)
            assert sta["n_spikes"] == int(label.sum())
            assert sta["n_anchors"] == label.size

    def test_a_session_without_spikes_leaves_the_triggered_average_at_zero(self):
        """With no spikes the triggered sum is divided by max(0, 1) rather than zero, so it stays exactly
        zero and the difference is minus the baseline."""
        session = {**self._session(4), "spike_frames": np.array([], dtype=np.int64)}
        sta = spike_triggered_average({"a": session}, ["a"], [0], self.N_STA_LAGS, 300)
        assert sta["n_spikes"] == 0
        assert np.all(sta["triggered"] == 0.0)
        np.testing.assert_allclose(sta["difference"], -sta["baseline"])


def _recording_solver(log: list) -> type:
    """
    Description
    -----------
    Build a ``GroupElasticNetGLM`` subclass that behaves exactly like the solver but keeps a copy of what
    every fit saw and returned, so a test can check how ``fit_quiet_model`` assembled the design, the
    offsets and the returned intercept from the solver's own output. The intercept is copied at the end
    of ``fit`` because the raw path later overwrites ``intercept_`` on the solver object it returns.

    Parameters
    ----------
    log (list)
        List that receives one dict per fit: ``x``, ``offset``, ``coef`` and ``intercept``.

    Returns
    -------
    recording_class (type)
        The recording subclass, to be monkeypatched over ``kinematic_encoding.GroupElasticNetGLM``.
    """

    class RecordingSolver(GroupElasticNetGLM):
        """A ``GroupElasticNetGLM`` that appends a copy of each fit's inputs and outputs to ``log``."""

        def fit(self, x_design, y, sample_weight=None, offset=None):
            """
            Description
            -----------
            Run the real fit, then record copies of the design, the offset, the coefficients and the
            intercept it produced.

            Parameters
            ----------
            x_design (np.ndarray)
                Design handed to the solver.
            y (np.ndarray)
                Labels handed to the solver.
            sample_weight (np.ndarray | None)
                Passed through unchanged.
            offset (np.ndarray | None)
                Offset handed to the solver.

            Returns
            -------
            self (RecordingSolver)
                The fitted solver, as ``GroupElasticNetGLM.fit`` returns it.
            """
            super().fit(x_design, y, sample_weight=sample_weight, offset=offset)
            log.append(
                {
                    "x": np.array(x_design, dtype=np.float64),
                    "offset": np.array(offset, dtype=np.float64),
                    "coef": np.array(self.coef_, dtype=np.float64),
                    "intercept": float(self.intercept_),
                }
            )
            return self

    return RecordingSolver


def _retained_anchor_count(
    session: dict, negatives_per_positive: int, rng, cap: int
) -> int:
    """
    Description
    -----------
    Number of anchors ``subsample_quiet_anchors`` keeps for one session, i.e. the number of design rows
    that session contributes to a quiet fit, drawn from the same generator in the same order the fit
    draws it.

    Parameters
    ----------
    session (dict)
        A session built by ``_planted_session``.
    negatives_per_positive (int)
        The ``negatives_per_positive`` setting.
    rng (np.random.Generator)
        Generator in the state the fit would hand to this session.
    cap (int)
        The per-session anchor cap.

    Returns
    -------
    n_rows (int)
        Retained anchors.
    """
    anchors = subsample_quiet_anchors(
        session["quiet"],
        session["spike_frames"],
        session["n_frames"],
        negatives_per_positive,
        rng,
        cap,
    )[0]
    return int(anchors.size)


class TestFrameFilterModel:
    """The frame-space model object every consumer of a spline fit receives."""

    def test_the_constructor_stores_canonical_types(self):
        """
        Description
        -----------
        Whatever types the caller hands in, the model stores float64 weights, a float intercept, a bool
        convergence flag and an int iteration count, so integer weights cannot turn the linear predictor
        into integer arithmetic and the iteration count is a plain number in progress lines and artifacts.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        model = FrameFilterModel([1, 0, 2], np.float32(0.5), 1, np.int64(7))
        assert model.coef_.dtype == np.float64
        np.testing.assert_array_equal(model.coef_, [1.0, 0.0, 2.0])
        assert isinstance(model.intercept_, float)
        assert model.intercept_ == 0.5
        assert model.converged_ is True
        assert isinstance(model.n_iter_, int)
        assert model.n_iter_ == 7

    def test_the_offset_is_added_to_the_linear_predictor(self):
        """
        Description
        -----------
        Weights ``[1, 0, 2]`` and intercept 0.5 on rows ``[1, 2, 3]`` and ``[0, 1, -1]`` give
        ``[1 + 6 + 0.5, -2 + 0.5] = [7.5, -1.5]``; an offset of ``[1, -2]`` is ADDED row by row, giving
        ``[8.5, -3.5]``.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        model = FrameFilterModel(np.array([1.0, 0.0, 2.0]), 0.5, True, 1)
        design = np.array([[1.0, 2.0, 3.0], [0.0, 1.0, -1.0]])
        np.testing.assert_allclose(model.predict_eta(design), [7.5, -1.5])
        np.testing.assert_allclose(
            model.predict_eta(design, offset=[1.0, -2.0]), [8.5, -3.5]
        )


class TestLagBasisKnownAnswer:
    """The lag-ridge scaling of the spline basis, checked column by column against the formula."""

    @pytest.mark.parametrize(
        ("n_lags", "n_splines", "alpha", "correction"),
        [
            (15, 6, 0.5, "none"),
            (15, 6, 100.0, "drop_outside_and_rescale"),
            (2, 4, 0.5, "none"),
        ],
    )
    def test_each_column_is_scaled_by_its_lag_fraction(
        self, n_lags, n_splines, alpha, correction
    ):
        """
        Description
        -----------
        Every column of the B-spline basis is divided by ``sqrt(1 + alpha * f**2)``, where ``f`` is the
        column's centre of mass counted back from the LAST lag (the spike's frame) as a fraction of
        ``n_lags - 1``, and the whole basis is then divided by its mean column sum. The expected basis is
        rebuilt from ``gam_bspline_basis`` with that formula. The cases cover a ridge weaker than one
        (``alpha`` 0.5), the shipped ridge on the spike-end-corrected basis, and a two-lag window, where
        the four columns of the cubic basis sit at centres 0, 0.2, 0.8 and 1 frames, so ``f`` is
        ``[1, 0.8, 0.2, 0]`` with a denominator of exactly one frame.

        Parameters
        ----------
        n_lags (int)
            History length in frames.
        n_splines (int)
            Number of splines.
        alpha (float)
            The ``l2_lag_alpha`` lag-ridge strength.
        correction (str)
            The ``spike_end_correction`` setting.

        Returns
        -------
        None
        """
        settings = {
            "type": "bspline",
            "n_splines": n_splines,
            "spline_order": 3,
            "l2_lag_alpha": alpha,
            "spike_end_correction": correction,
        }
        raw = np.asarray(gam_bspline_basis(n_lags, n_splines, 3), dtype=np.float64)
        if correction == "drop_outside_and_rescale":
            raw = raw[:, :-1]
            raw = raw * np.sqrt(raw.sum(axis=0).max() / raw.sum(axis=0))[None, :]
        centres = (raw * np.arange(n_lags)[:, None]).sum(axis=0) / raw.sum(axis=0)
        fraction = (n_lags - 1.0 - centres) / (n_lags - 1.0)
        if n_lags == 2:
            np.testing.assert_allclose(fraction, [1.0, 0.8, 0.2, 0.0], atol=1e-12)
        scaled = raw / np.sqrt(1.0 + alpha * fraction**2)[None, :]
        expected = scaled / scaled.sum(axis=0).mean()
        np.testing.assert_allclose(lag_basis(n_lags, settings), expected, rtol=1e-12)


class TestQuietBlockSessions:
    """Contiguous quiet blocks presented as pseudo-sessions, on anchor sets small enough to work by hand."""

    def test_blocks_too_short_for_the_gap_are_skipped_and_the_rest_trimmed(self):
        """
        Description
        -----------
        Seven anchors (given unsorted) in three blocks: ``linspace(0, 7, 4)`` cuts them at 0, 2, 4, 7 into
        ``[0, 1]``, ``[2, 3]`` and ``[4, 5, 6]``. With a gap of one frame a block must hold more than two
        anchors, so the first two are skipped WITHOUT ending the loop, and the third is trimmed by one at
        each end to ``[5]``. The pseudo-session keeps every other key of the session.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        session = {"quiet": np.array([6, 0, 3, 1, 5, 2, 4]), "n_frames": 50}
        blocks = quiet_block_sessions(session, "s", 3, 1)
        assert list(blocks) == ["s#block2"]
        np.testing.assert_array_equal(blocks["s#block2"]["quiet"], [5])
        assert blocks["s#block2"]["n_frames"] == 50

    def test_a_zero_gap_keeps_every_block_whole(self):
        """
        Description
        -----------
        With no gap nothing is trimmed: five anchors in two blocks are cut at 0, 2, 5 and come back as
        ``[0, 1]`` and ``[2, 3, 4]``.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        blocks = quiet_block_sessions({"quiet": np.arange(5)}, "s", 2, 0)
        assert list(blocks) == ["s#block0", "s#block1"]
        np.testing.assert_array_equal(blocks["s#block0"]["quiet"], [0, 1])
        np.testing.assert_array_equal(blocks["s#block1"]["quiet"], [2, 3, 4])


class TestFitQuietModelArithmetic:
    """The bookkeeping of ``fit_quiet_model``: the anchor cap, the base rate, the session baselines and the
    intercept it returns."""

    @pytest.mark.parametrize(
        ("max_train_total", "n_sessions", "expected_cap"),
        [(1000, 1, 1000), (1000, 2, 500), (1, 3, 1)],
    )
    def test_the_global_cap_is_divided_across_training_sessions(
        self, monkeypatch, max_train_total, n_sessions, expected_cap
    ):
        """
        Description
        -----------
        The design is assembled from every training session at once, so each session's subsample is
        capped at ``max_train_total / n_sessions`` anchors (floored, and never below one). Every session
        must receive exactly that cap: 1000 for one session, 500 each for two, and 1 for three sessions
        under a global cap of 1 (where the floor of one third would otherwise be zero).

        Parameters
        ----------
        monkeypatch (pytest.MonkeyPatch)
            Wraps ``subsample_quiet_anchors`` to record the cap it receives.
        max_train_total (int)
            The global anchor cap.
        n_sessions (int)
            Number of training sessions.
        expected_cap (int)
            The per-session cap each session must receive.

        Returns
        -------
        None
        """
        caps = []
        real_subsample = kinematic_encoding.subsample_quiet_anchors

        def recording_subsample(
            quiet_frames,
            spike_frames,
            n_frames,
            negatives_per_positive,
            rng,
            max_total=None,
        ):
            """
            Description
            -----------
            Record the cap, then subsample exactly as the real function does.

            Parameters
            ----------
            quiet_frames (np.ndarray)
                Candidate quiet anchors.
            spike_frames (np.ndarray)
                Spike frames.
            n_frames (int)
                Session frame count.
            negatives_per_positive (int)
                Class ratio.
            rng (np.random.Generator)
                Generator.
            max_total (int | None)
                The cap under test.

            Returns
            -------
            result (tuple)
                What ``subsample_quiet_anchors`` returns.
            """
            caps.append(max_total)
            return real_subsample(
                quiet_frames,
                spike_frames,
                n_frames,
                negatives_per_positive,
                rng,
                max_total,
            )

        monkeypatch.setattr(
            kinematic_encoding, "subsample_quiet_anchors", recording_subsample
        )
        encoding = _settings()["kinematic_encoding"]
        encoding["max_train_total"] = max_train_total
        sessions = {
            f"s{index}": _planted_session(110 + index, [1.0], n_frames=600)
            for index in range(n_sessions)
        }
        fit_quiet_model(
            sessions,
            list(sessions),
            [0],
            N_LAGS,
            np.random.default_rng(0),
            encoding,
            message_output=_silent,
        )
        assert caps == [expected_cap] * n_sessions

    def test_the_base_rate_is_the_pooled_rate_before_subsampling(self):
        """
        Description
        -----------
        The returned base rate is the fraction of ALL quiet anchors of the training sessions that carry a
        spike, counted before the negatives are thinned: total quiet spikes over total quiet anchors,
        pooled across a 5 % session and a roughly 20 % session.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        sessions = {
            "a": _planted_session(120, [1.0], n_frames=2000, baseline=-3.0),
            "b": _planted_session(121, [1.0], n_frames=2000, baseline=-1.5),
        }
        n_spikes = sum(
            spike_labels_at_frames(
                session["spike_frames"], session["quiet"], session["n_frames"]
            ).sum()
            for session in sessions.values()
        )
        n_anchors = sum(session["quiet"].size for session in sessions.values())
        _model, base_rate = fit_quiet_model(
            sessions,
            ["a", "b"],
            [0],
            N_LAGS,
            np.random.default_rng(0),
            _settings()["kinematic_encoding"],
            message_output=_silent,
        )
        assert base_rate == pytest.approx(n_spikes / n_anchors, rel=1e-12)

    def test_a_rate_of_one_is_clipped_just_below_one(self):
        """
        Description
        -----------
        A training set of one quiet anchor that carries a spike has a raw rate of 1/1. The base rate is
        clipped to ``1 - 1e-6`` so that its log-odds stay finite downstream.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        session = _planted_session(122, [1.0], n_frames=600)
        spiking = session["spike_frames"][session["spike_frames"] >= N_LAGS][:1]
        encoding = _settings()["kinematic_encoding"]
        encoding["solver"]["max_iter"] = 50
        _model, base_rate = fit_quiet_model(
            {"a": {**session, "quiet": spiking}},
            ["a"],
            [0],
            N_LAGS,
            np.random.default_rng(0),
            encoding,
            message_output=_silent,
        )
        assert base_rate == 1.0 - 1e-6

    def test_spline_session_indicators_and_the_anchor_weighted_intercept(
        self, monkeypatch
    ):
        """
        Description
        -----------
        On the spline path two training sessions get ONE indicator column (session ``b``, relative to the
        first session ``a`` whose level the solver intercept carries), appended as an extra column group
        that holds ``indicator_scale`` on ``b``'s rows in its first column and zero everywhere else. The
        baseline difference is the indicator coefficient times the scale; the planted difference is
        -1.0 - (-3.0) = 2.0. The returned intercept is the solver intercept plus the ANCHOR-weighted mean
        of the baselines ``[0, delta_b]``, the weights being each session's number of design rows, which
        are unequal here because the faster session keeps more of its frames. Two features are fitted so
        the indicator group is the last of three groups and not the only non-filter row.

        Parameters
        ----------
        monkeypatch (pytest.MonkeyPatch)
            Replaces the solver with the recording subclass.

        Returns
        -------
        None
        """
        log = []
        monkeypatch.setattr(
            kinematic_encoding, "GroupElasticNetGLM", _recording_solver(log)
        )
        encoding = _settings()["kinematic_encoding"]
        sessions = {
            "a": _planted_session(130, [1.0, 0.6], n_frames=3000, baseline=-3.0),
            "b": _planted_session(131, [1.0, 0.6], n_frames=3000, baseline=-1.0),
        }
        model, _rate = fit_quiet_model(
            sessions,
            ["a", "b"],
            [0, 1],
            N_LAGS,
            np.random.default_rng(0),
            encoding,
            message_output=_silent,
        )
        replay = np.random.default_rng(0)
        cap = encoding["max_train_total"] // 2
        n_a = _retained_anchor_count(
            sessions["a"], encoding["negatives_per_positive"], replay, cap
        )
        n_b = _retained_anchor_count(
            sessions["b"], encoding["negatives_per_positive"], replay, cap
        )
        (record,) = log
        n_columns = lag_basis(N_LAGS, encoding["temporal_basis"]).shape[1]
        scale = encoding["session_intercepts"]["indicator_scale"]
        expected_indicators = np.zeros((n_a + n_b, n_columns))
        expected_indicators[n_a:, 0] = scale
        np.testing.assert_array_equal(
            record["x"][:, 2 * n_columns :], expected_indicators
        )
        delta_b = record["coef"].reshape(3, n_columns)[2, 0] * scale
        assert delta_b == pytest.approx(2.0, abs=0.5)
        assert n_a != n_b
        assert model.intercept_ == pytest.approx(
            record["intercept"] + delta_b * n_b / (n_a + n_b), rel=1e-12
        )

    def test_as_many_indicators_as_the_group_holds_are_accepted(self):
        """
        Description
        -----------
        The indicator group is refused only when the k - 1 indicators do NOT fit in it. Four splines under
        the spike-end correction leave three columns, and four training sessions need exactly three
        indicators, so the fit must run and return one filter of ``n_lags`` weights.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        encoding = _settings()["kinematic_encoding"]
        encoding["temporal_basis"]["n_splines"] = 4
        sessions = {
            f"s{index}": _planted_session(125 + index, [1.0], n_frames=600)
            for index in range(4)
        }
        model, _rate = fit_quiet_model(
            sessions,
            list(sessions),
            [0],
            N_LAGS,
            np.random.default_rng(0),
            encoding,
            message_output=_silent,
        )
        assert np.asarray(model.coef_).shape == (N_LAGS,)

    @staticmethod
    def _raw_backfit_encoding(negatives_per_positive: int) -> dict:
        """
        Description
        -----------
        The raw per-frame encoding of ``TestFitQuietModelCornerPaths`` with a chosen class ratio, so the
        case-control offsets of the sessions can be made non-zero.

        Parameters
        ----------
        negatives_per_positive (int)
            Majority-to-minority ratio of the subsample.

        Returns
        -------
        encoding (dict)
            The raw-path ``kinematic_encoding`` block.
        """
        encoding = TestFitQuietModelCornerPaths._raw_encoding()
        encoding["negatives_per_positive"] = negatives_per_positive
        return encoding

    def test_raw_backfit_recovers_the_planted_baseline_difference(self, monkeypatch):
        """
        Description
        -----------
        On the raw path the baselines are backfitted: the first fit runs with the case-control offsets
        alone, and every later fit adds each session's baseline to its rows. The difference between the
        last and the first offsets is therefore ``delta[owner]``: exactly zero on the first session's
        rows (its level is the intercept's) and one constant on the second session's, which must recover
        the planted -2.0 - (-3.0) = 1.0. Both sessions are thinned (two negatives per positive) so their
        case-control offsets are non-zero and a sign error in them would move the baseline. The returned
        intercept is the final solver intercept plus the anchor-weighted mean baseline.

        Parameters
        ----------
        monkeypatch (pytest.MonkeyPatch)
            Replaces the solver with the recording subclass.

        Returns
        -------
        None
        """
        log = []
        monkeypatch.setattr(
            kinematic_encoding, "GroupElasticNetGLM", _recording_solver(log)
        )
        encoding = self._raw_backfit_encoding(2)
        sessions = {
            "a": _planted_session(140, [1.0], n_frames=3000, baseline=-3.0),
            "b": _planted_session(141, [1.0], n_frames=3000, baseline=-2.0),
        }
        model, _rate = fit_quiet_model(
            sessions,
            ["a", "b"],
            [0],
            N_LAGS,
            np.random.default_rng(0),
            encoding,
            message_output=_silent,
        )
        replay = np.random.default_rng(0)
        cap = encoding["max_train_total"] // 2
        n_a = _retained_anchor_count(sessions["a"], 2, replay, cap)
        n_b = _retained_anchor_count(sessions["b"], 2, replay, cap)
        assert len(log) > 1
        shift = log[-1]["offset"] - log[0]["offset"]
        assert shift.size == n_a + n_b
        assert np.all(shift[:n_a] == 0.0)
        delta_b = float(shift[n_a])
        np.testing.assert_allclose(shift[n_a:], delta_b)
        assert delta_b == pytest.approx(1.0, abs=0.35)
        assert model.intercept_ == pytest.approx(
            log[-1]["intercept"] + delta_b * n_b / (n_a + n_b), rel=1e-12
        )

    def test_a_tiny_session_still_gets_its_own_baseline(self, monkeypatch):
        """
        Description
        -----------
        The Newton update of a baseline stops only when the curvature ``sum p (1 - p)`` is exactly zero
        (an empty session). A session of eight quiet anchors, four of them spikes, has a curvature well
        below one at the starting baseline of a 5 % session, yet its baseline must still move to fit its
        50 % rate: about logit(0.5) - logit(0.05) = 2.9 above the first session's.

        Parameters
        ----------
        monkeypatch (pytest.MonkeyPatch)
            Replaces the solver with the recording subclass.

        Returns
        -------
        None
        """
        log = []
        monkeypatch.setattr(
            kinematic_encoding, "GroupElasticNetGLM", _recording_solver(log)
        )
        large = _planted_session(150, [1.0], n_frames=3000, baseline=-3.0)
        tiny = _planted_session(151, [1.0], n_frames=3000, baseline=-3.0)
        usable_spikes = tiny["spike_frames"][tiny["spike_frames"] >= N_LAGS]
        silent_frames = np.setdiff1d(np.arange(N_LAGS, 3000), tiny["spike_frames"])
        tiny = {
            **tiny,
            "quiet": np.sort(np.concatenate([usable_spikes[:4], silent_frames[:4]])),
        }
        fit_quiet_model(
            {"large": large, "tiny": tiny},
            ["large", "tiny"],
            [0],
            N_LAGS,
            np.random.default_rng(0),
            self._raw_backfit_encoding(10),
            message_output=_silent,
        )
        shift = log[-1]["offset"] - log[0]["offset"]
        assert shift[-1] > 1.5

    @pytest.mark.parametrize("path", ["raw", "spline"])
    def test_the_group_penalty_shrinkage_is_not_refitted_away(self, path):
        """
        Description
        -----------
        The solver is built with the relaxed refit OFF: whatever group penalty the settings declare
        shrinks the returned filters, and a selected group is not refitted without it. One session, two
        features (a driven one and a blind one): at a group penalty of 0.05 the blind feature is zeroed
        and the driven filter keeps clearly less than 90 % of its norm at a group penalty of zero. A
        relaxed refit would restore the unshrunk norm. The spline case also has two features on a single
        session, where the filters are mapped back to frames without any indicator group.

        Parameters
        ----------
        path (str)
            ``"raw"`` for one weight per frame, ``"spline"`` for the shipped basis.

        Returns
        -------
        None
        """
        session = _planted_session(160, [1.0, 0.0], n_frames=3000)
        norms = {}
        for lambda_group in (0.0, 0.05):
            if path == "raw":
                encoding = TestFitQuietModelCornerPaths._raw_encoding()
                encoding["raw_lag_penalties"]["lambda_group"] = lambda_group
            else:
                encoding = _settings()["kinematic_encoding"]
                encoding["temporal_basis"]["lambda_group"] = lambda_group
            model, _rate = fit_quiet_model(
                {"a": session},
                ["a"],
                [0, 1],
                N_LAGS,
                np.random.default_rng(0),
                encoding,
                message_output=_silent,
            )
            assert np.asarray(model.coef_).size == 2 * N_LAGS
            norms[lambda_group] = np.linalg.norm(
                np.asarray(model.coef_).reshape(2, N_LAGS), axis=1
            )
        assert norms[0.05][1] == 0.0
        assert norms[0.05][0] < 0.9 * norms[0.0][0]


class TestLinearPredictorAtFrames:
    """The chunked linear predictor that the frozen nulls reuse."""

    @pytest.mark.parametrize("chunk_rows", [1, 7, 1000])
    def test_every_frame_matches_the_unchunked_predictor(self, chunk_rows):
        """
        Description
        -----------
        Chunking is a memory budget only: at every chunk size, including chunks of one frame and one chunk
        holding everything, each frame's value (the first one included) equals the model's prediction on
        the full lagged design of the requested columns, in the requested order.

        Parameters
        ----------
        chunk_rows (int)
            Frames per chunk.

        Returns
        -------
        None
        """
        generator = np.random.default_rng(3)
        features = generator.normal(size=(120, 3))
        model = FrameFilterModel(generator.normal(size=2 * N_LAGS), -0.5, True, 1)
        frames = np.arange(N_LAGS, 120, 3)
        expected = model.predict_eta(lagged_design(features[:, [2, 0]], frames, N_LAGS))
        values = linear_predictor_at_frames(
            model, features, [2, 0], frames, N_LAGS, chunk_rows
        )
        np.testing.assert_allclose(values, expected, rtol=1e-12)


class TestScreenAndSelectionScoring:
    """The scoring rule, threshold and convergence reporting of the screen and of forward selection."""

    def test_the_first_fold_score_is_the_bernoulli_frozen_score(
        self, planted_pool, screened, selected
    ):
        """
        Description
        -----------
        The first thing both the screen and selection do with a fresh generator is fit column 0 on the
        first inner fold (train on ``s1`` and ``s2``, validate on ``s0``) and score it by the Bernoulli
        frozen-model rule. Replaying exactly that fit and score must reproduce the first stored fold score
        of the screen's column-0 row (and its slope) and of selection's anchor step.

        Parameters
        ----------
        planted_pool (dict)
            The module-scoped planted sessions.
        screened (dict)
            The module-scoped screen result.
        selected (dict)
            The module-scoped selection result.

        Returns
        -------
        None
        """
        settings = _settings()
        encoding = settings["kinematic_encoding"]
        model, _rate = fit_quiet_model(
            planted_pool,
            ["s1", "s2"],
            [0],
            N_LAGS,
            np.random.default_rng(0),
            encoding,
            message_output=_silent,
        )
        session = planted_pool["s0"]
        eta = linear_predictor_at_frames(
            model,
            session["feature_time_series"],
            [0],
            session["quiet"],
            N_LAGS,
            encoding["chunk_rows"],
        )
        labels = spike_labels_at_frames(
            session["spike_frames"], session["quiet"], session["n_frames"]
        )
        score, slope = frozen_model_score(
            eta,
            labels,
            np.zeros(labels.size, dtype=np.int64),
            encoding["solver"]["calibration_steps"],
            "bernoulli",
            1.0,
            encoding["calibration"],
        )
        assert screened["rows"][0]["fold_scores"][0] == pytest.approx(score, rel=1e-9)
        assert screened["rows"][0]["fold_slopes"][0] == pytest.approx(slope, rel=1e-9)
        assert selected["path"][0]["candidate"] == "planted_strong"
        assert selected["path"][0]["fold_scores"][0] == pytest.approx(score, rel=1e-9)

    def test_the_screen_threshold_is_alpha_over_features_and_strict(self):
        """
        Description
        -----------
        Two features, 40 shuffles and ``screen_alpha = 2 / 41`` put the Bonferroni threshold at
        ``(2 / 41) / 2 = 1 / 41``, exactly the empirical floor of 40 draws. The driven column beats every
        draw, so its p-value IS that floor, and since a feature passes only when ``p < threshold`` it must
        fail -- although its score and slope are clearly positive.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        pool = {
            "a": _planted_session(170, [1.0, 0.0]),
            "b": _planted_session(171, [1.0, 0.0]),
        }
        settings = _settings()
        settings["kinematic_encoding"]["feature_selection"]["screen_alpha"] = 2.0 / (
            SCREEN_SHUFFLES + 1
        )
        driven, _blind = screen_features(
            pool,
            ["a", "b"],
            N_LAGS,
            np.random.default_rng(0),
            settings,
            ["driven", "blind"],
            message_output=_silent,
        )
        assert driven["score"] > 0.01
        assert driven["slope"] > 0.0
        assert driven["p"] == 1.0 / (SCREEN_SHUFFLES + 1)
        assert driven["survived"] is False

    @pytest.mark.parametrize("stage", ["screen", "select"])
    def test_an_unconverged_fit_is_reported_to_the_given_sink(self, stage):
        """
        Description
        -----------
        A fit that exhausts its iteration cap must be announced where the caller collects progress, not
        on standard output and not by crashing. With the solver capped at one iteration, every inner-fold
        fit (two folds of a two-session pool) is unconverged, so the sink handed to the screen or to
        selection must receive one non-convergence warning per fit.

        Parameters
        ----------
        stage (str)
            ``"screen"`` runs ``screen_features``, ``"select"`` runs ``forward_select``.

        Returns
        -------
        None
        """
        pool = {
            "a": _planted_session(180, [1.0], n_frames=3000),
            "b": _planted_session(181, [1.0], n_frames=3000),
        }
        settings = _settings()
        settings["kinematic_encoding"]["solver"]["max_iter"] = 1
        messages = []
        if stage == "screen":
            (row,) = screen_features(
                pool,
                ["a", "b"],
                N_LAGS,
                np.random.default_rng(0),
                settings,
                ["driven"],
                message_output=messages.append,
            )
            assert row["converged"] is False
        else:
            forward_select(
                pool,
                ["a", "b"],
                [0],
                N_LAGS,
                np.random.default_rng(0),
                settings,
                ["driven"],
                message_output=messages.append,
            )
        warnings = [line for line in messages if "FIT DID NOT CONVERGE" in line]
        assert len(warnings) == 2

    def test_every_converged_step_is_recorded_as_converged(self, selected):
        """
        Description
        -----------
        Each step of the selection path carries a ``converged`` entry, stored in the artifact, that is
        True exactly when every inner-fold fit behind it converged. On the planted pool every fit
        converges, so every step must record ``True``.

        Parameters
        ----------
        selected (dict)
            The module-scoped selection result.

        Returns
        -------
        None
        """
        assert [step["converged"] for step in selected["path"]] == [True, True, True]

    def test_one_unconverged_fold_marks_the_step_unconverged(
        self, planted_pool, monkeypatch
    ):
        """
        Description
        -----------
        A step is converged only if ALL of its inner-fold fits converged. Marking the fit of one fold
        (trained on ``s1``, validated on ``s0``) as unconverged while the other fold converges must flag
        the anchor step unconverged.

        Parameters
        ----------
        planted_pool (dict)
            The module-scoped planted sessions.
        monkeypatch (pytest.MonkeyPatch)
            Wraps ``fit_quiet_model`` to mark one fold's fit unconverged.

        Returns
        -------
        None
        """
        real_fit = kinematic_encoding.fit_quiet_model

        def one_fold_unconverged(
            per_session,
            train_session_ids,
            feature_indices,
            n_lags,
            rng,
            encoding_settings,
            message_output=print,
        ):
            """
            Description
            -----------
            Fit exactly as the real function does, then mark the fit unconverged when it was trained on
            ``s1`` alone.

            Parameters
            ----------
            per_session (dict)
                Session data.
            train_session_ids (list)
                Training sessions.
            feature_indices (list)
                Feature columns.
            n_lags (int)
                History length.
            rng (np.random.Generator)
                Generator.
            encoding_settings (dict)
                Encoding settings.
            message_output (Callable)
                Progress sink.

            Returns
            -------
            result (tuple)
                The real ``(estimator, base_rate)``, with the estimator possibly marked unconverged.
            """
            model, rate = real_fit(
                per_session,
                train_session_ids,
                feature_indices,
                n_lags,
                rng,
                encoding_settings,
                message_output,
            )
            if list(train_session_ids) == ["s1"]:
                model.converged_ = False
            return model, rate

        monkeypatch.setattr(kinematic_encoding, "fit_quiet_model", one_fold_unconverged)
        pool = {"s0": planted_pool["s0"], "s1": planted_pool["s1"]}
        chosen, path = forward_select(
            pool,
            ["s0", "s1"],
            [0],
            N_LAGS,
            np.random.default_rng(0),
            _settings(),
            FEATURE_NAMES,
            message_output=_silent,
        )
        assert chosen == [0]
        assert path[0]["converged"] is False

    def test_the_anchor_is_the_best_survivor_not_the_first(self, planted_pool):
        """
        Description
        -----------
        The anchor is the survivor with the highest mean held-out score, whatever order the survivors are
        offered in: offering the weak column first must still anchor on the strong one and then accept the
        weak one.

        Parameters
        ----------
        planted_pool (dict)
            The module-scoped planted sessions.

        Returns
        -------
        None
        """
        pool = {"s0": planted_pool["s0"], "s1": planted_pool["s1"]}
        chosen, path = forward_select(
            pool,
            ["s0", "s1"],
            [1, 0],
            N_LAGS,
            np.random.default_rng(0),
            _settings(),
            FEATURE_NAMES,
            message_output=_silent,
        )
        assert chosen[0] == 0
        assert path[0]["candidate"] == "planted_strong"

    def test_a_search_with_no_finite_candidate_stops_and_returns(
        self, planted_pool, monkeypatch
    ):
        """
        Description
        -----------
        When no remaining candidate has a finite mean score, the search ends and selection still returns
        its model and path. Making every multi-feature predictor constant (a predictor with no spread
        scores NaN) leaves the anchor without a scorable candidate, so the result must be the anchor
        alone with a one-step path.

        Parameters
        ----------
        planted_pool (dict)
            The module-scoped planted sessions.
        monkeypatch (pytest.MonkeyPatch)
            Wraps ``linear_predictor_at_frames`` to flatten every multi-feature predictor.

        Returns
        -------
        None
        """
        real_predictor = kinematic_encoding.linear_predictor_at_frames

        def flat_for_candidates(
            estimator, feature_time_series, feature_indices, frames, n_lags, chunk_rows
        ):
            """
            Description
            -----------
            Return the real predictor for single-feature models and a constant zero predictor for any
            model with more than one feature.

            Parameters
            ----------
            estimator (object)
                Fitted model.
            feature_time_series (np.ndarray)
                Predictor time series.
            feature_indices (list)
                Feature columns.
            frames (np.ndarray)
                Frames to evaluate.
            n_lags (int)
                History length.
            chunk_rows (int)
                Frames per chunk.

            Returns
            -------
            eta (np.ndarray)
                Linear predictor at ``frames``.
            """
            if len(feature_indices) > 1:
                return np.zeros(frames.size)
            return real_predictor(
                estimator,
                feature_time_series,
                feature_indices,
                frames,
                n_lags,
                chunk_rows,
            )

        monkeypatch.setattr(
            kinematic_encoding, "linear_predictor_at_frames", flat_for_candidates
        )
        pool = {"s0": planted_pool["s0"], "s1": planted_pool["s1"]}
        chosen, path = forward_select(
            pool,
            ["s0", "s1"],
            [0, 1],
            N_LAGS,
            np.random.default_rng(0),
            _settings(),
            FEATURE_NAMES,
            message_output=_silent,
        )
        assert chosen == [0]
        assert len(path) == 1
        assert path[0]["decision"] == "ANCHOR"


class TestBlockResampleAnchors:
    """Block resampling of quiet anchors, checked against draws replayed from the same generator."""

    @pytest.mark.parametrize(("block_frames", "seed"), [(3, 7), (1, 8)])
    def test_the_draw_is_the_replayed_block_bootstrap(self, block_frames, seed):
        """
        Description
        -----------
        Twenty anchors (every second frame) are covered by ``ceil(20 / width)`` blocks whose starts are
        drawn uniformly from ``0 .. 20 - width``; each block is ``width`` consecutive anchors, the
        concatenation is cut to 20 and sorted. Replaying the starts from a generator with the same seed
        must give the same anchors. A block of one anchor is the ordinary bootstrap: twenty starts in
        ``0 .. 19``.

        Parameters
        ----------
        block_frames (int)
            Anchors per block.
        seed (int)
            Generator seed.

        Returns
        -------
        None
        """
        anchors = np.arange(0, 40, 2)
        width = block_frames
        starts = np.random.default_rng(seed).integers(
            0, anchors.size - width + 1, size=int(np.ceil(anchors.size / width))
        )
        expected = np.sort(
            np.concatenate([anchors[start : start + width] for start in starts])[
                : anchors.size
            ]
        )
        drawn = block_resample_anchors(
            anchors[::-1], block_frames, np.random.default_rng(seed)
        )
        np.testing.assert_array_equal(drawn, expected)

    def test_fewer_anchors_than_one_block_come_back_whole(self):
        """
        Description
        -----------
        When the block is longer than the anchor set, the only possible start is the first anchor, so
        every draw returns the whole set unchanged, whatever the generator.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        anchors = np.arange(5)
        for seed in range(20):
            drawn = block_resample_anchors(anchors, 10, np.random.default_rng(seed))
            np.testing.assert_array_equal(drawn, anchors)


class TestFilterBandReplay:
    """The filter band, rebuilt draw by draw from the same generator."""

    def test_each_draw_is_a_refit_on_block_resampled_anchors(self):
        """
        Description
        -----------
        Each band draw block-resamples every training session's quiet anchors (blocks of ``n_lags``) and
        refits the frozen feature set on them, drawing from one shared generator. Replaying that sequence
        by hand must reproduce every stored filter and intercept; ``n_converged`` counts the converged
        refits (all three here), and ``low`` / ``high`` are the per-lag 2.5th and 97.5th percentiles of the
        stored filters.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        sessions = {
            "a": _planted_session(190, [1.0], n_frames=2000),
            "b": _planted_session(191, [1.0], n_frames=2000),
        }
        encoding = _settings()["kinematic_encoding"]
        band = filter_band(
            sessions,
            ["a", "b"],
            [0],
            N_LAGS,
            np.random.default_rng(5),
            encoding,
            3,
            message_output=_silent,
        )
        replay = np.random.default_rng(5)
        n_converged = 0
        for draw in range(3):
            resampled = {
                session_id: {
                    **sessions[session_id],
                    "quiet": block_resample_anchors(
                        sessions[session_id]["quiet"], N_LAGS, replay
                    ),
                }
                for session_id in ("a", "b")
            }
            model, _rate = fit_quiet_model(
                resampled,
                ["a", "b"],
                [0],
                N_LAGS,
                replay,
                encoding,
                message_output=_silent,
            )
            np.testing.assert_allclose(
                band["filters"][draw], model.coef_.reshape(1, N_LAGS), rtol=1e-9
            )
            assert band["intercepts"][draw] == pytest.approx(model.intercept_, rel=1e-9)
            n_converged += int(model.converged_)
        assert n_converged == 3
        assert band["n_converged"] == n_converged
        np.testing.assert_allclose(
            band["low"], np.percentile(band["filters"], 2.5, axis=0)
        )
        np.testing.assert_allclose(
            band["high"], np.percentile(band["filters"], 97.5, axis=0)
        )
        assert band["low"].shape == (1, N_LAGS)


class TestReadoutEdgeCases:
    """Edge cases of the filter descriptors and the spike-triggered average."""

    def test_a_filter_with_total_weight_below_one_has_a_centre_of_mass(self):
        """
        Description
        -----------
        Any non-zero filter has a centre of mass, however small its weights. For ``[0, 0.25, 0.5, 0]`` at
        10 frames per second (ages 0.3, 0.2, 0.1, 0.0 s) it is (0.25 * 0.2 + 0.5 * 0.1) / 0.75 = 0.1333 s.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        (row,) = filter_descriptors(np.array([0.0, 0.25, 0.5, 0.0]), ["small"], 4, 10.0)
        assert row["centre_of_mass_seconds"] == pytest.approx(0.1 / 0.75)

    def test_one_spiking_anchor_gives_its_own_design_row(self):
        """
        Description
        -----------
        With a single quiet anchor that carries a spike, both averages are that anchor's lagged design row
        divided by one: the triggered average and the baseline equal the row, and their difference is
        zero.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        features = np.arange(40, dtype=np.float64).reshape(20, 2)
        session = {
            "feature_time_series": features,
            "n_frames": 20,
            "spike_frames": np.array([12]),
            "quiet": np.array([12]),
        }
        row = lagged_design(features, np.array([12]), 5).reshape(2, 5)
        sta = spike_triggered_average({"a": session}, ["a"], [0, 1], 5, 100)
        np.testing.assert_allclose(sta["triggered"], row)
        np.testing.assert_allclose(sta["baseline"], row)
        np.testing.assert_allclose(sta["difference"], np.zeros((2, 5)))
        assert sta["n_spikes"] == 1
        assert sta["n_anchors"] == 1
