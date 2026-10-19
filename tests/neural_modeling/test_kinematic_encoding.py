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

from usv_playpen.neural_modeling import kinematic_encoding
from usv_playpen.neural_modeling.deviance_metrics import (
    frozen_model_null_scores,
    frozen_model_score,
)
from usv_playpen.neural_modeling.kinematic_encoding import (
    filter_descriptors,
    fit_quiet_model,
    forward_select,
    frozen_null_scores,
    screen_features,
    spike_triggered_average,
)
from usv_playpen.neural_modeling.neural_design_assembly import (
    lagged_design,
    spike_labels_at_frames,
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

    def _null(self, eta, frames, spikes, n_shuffles, seed):
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

    def test_splitting_the_draws_into_memory_blocks_changes_nothing(self, monkeypatch):
        """The block size is a memory budget, not a statistical choice: forcing blocks of three draws must
        give exactly the scores of one block holding all of them, including a short final block."""
        eta, frames, spikes = self._case(driven=False)
        whole = self._null(eta, frames, spikes, 10, seed=3)
        monkeypatch.setattr(
            kinematic_encoding, "_NULL_BLOCK_BYTES", frames.size * 8 * 3
        )
        blocked = self._null(eta, frames, spikes, 10, seed=3)
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
        encoding["significance_model"]["lambda_smooth"] = 10.0
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
