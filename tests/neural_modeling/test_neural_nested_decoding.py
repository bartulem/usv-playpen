"""
@author: bartulem
Coverage: claim 3's block-diagonal penalty and the behaviour control it is fitted against.

The penalty is the piece that makes a mixed design legal -- five behavioural features wanting 600 lags
of smoothing beside one neural column that must receive ridge only -- so these tests pin its SHAPE
(zeros off the behaviour blocks, zero on the neural diagonal) and its IDENTITY to the parent's
operator, since a second copy of the finite-difference operator could drift from the one P1 selected
its features under. They also pin the reduced model to P1's own result file rather than to a literal.
"""

from __future__ import annotations

import copy
import json
import pathlib
import pickle
import warnings
from typing import ClassVar

import numpy as np
import pytest

# Importing the estimator pulls in the JAX/Optax parent class, which sets a flag JAX v0.9 has
# deprecated; that one-time DeprecationWarning fires at import (collection) time, where the project's
# `filterwarnings = ["error"]` would turn it into a hard collection error. Suppressed just for this
# import, matching `tests/modeling/test_manifold_torus_regression.py` -- it is a third-party
# transition warning, not a defect under test.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from usv_playpen.modeling.manifold_torus_regression import (
        SmoothTorusManifoldRegression,
    )
    from usv_playpen.neural_modeling.neural_nested_decoding import (
        NestedTorusRegression,
        reduced_model_features,
    )

SETTINGS_PATH = (pathlib.Path(__file__).resolve().parents[2] / "src" / "usv_playpen"
                 / "_parameter_settings" / "neural_modeling_settings.json")


def _settings() -> dict:
    with SETTINGS_PATH.open() as handle:
        return json.load(handle)


class TestBlockDiagonalPenalty:
    """One temporal block per behavioural feature, and NOTHING on the neural column."""

    def test_the_neural_column_is_unsmoothed(self):
        est = NestedTorusRegression(n_behaviour_features=3, n_lags=10, n_neural_columns=1)
        penalty = est._smoothness_penalty()
        assert penalty.shape == (31, 31)
        assert penalty[30, 30] == 0.0
        assert np.abs(penalty[:30, 30]).max() == 0.0      # no cross terms either

    def test_features_do_not_smooth_into_each_other(self):
        """A single 600-lag block spanning two features would tie the end of one filter to the start
        of the next -- adjacent in the design, unrelated in time."""
        est = NestedTorusRegression(n_behaviour_features=2, n_lags=8, n_neural_columns=0)
        penalty = est._smoothness_penalty()
        assert penalty.shape == (16, 16)
        assert np.abs(penalty[:8, 8:]).max() == 0.0

    @pytest.mark.parametrize("order", [1, 2])
    def test_the_block_is_the_parents_operator_not_a_second_copy(self, order):
        """Including the reflective (Neumann) boundary rows the parent adds at order 2. A rebuilt
        operator could drift from the one the behaviour control was selected under."""
        parent = SmoothTorusManifoldRegression(n_features=1, n_time_bins=12,
                                               smoothness_derivative_order=order, metric="torus")
        nested = NestedTorusRegression(n_behaviour_features=1, n_lags=12, n_neural_columns=0,
                                       smoothness_derivative_order=order)
        np.testing.assert_allclose(nested._smoothness_penalty(), parent._smoothness_penalty())

    def test_the_reduced_model_penalty_has_no_neural_block(self):
        est = NestedTorusRegression(n_behaviour_features=5, n_lags=20, n_neural_columns=0)
        assert est._smoothness_penalty().shape == (100, 100)

    def test_the_shipped_shape_is_five_features_by_six_hundred_lags_plus_one(self):
        est = NestedTorusRegression(n_behaviour_features=5, n_lags=600, n_neural_columns=1)
        assert est._smoothness_penalty().shape == (3001, 3001)


class TestTheEstimatorFits:
    """The parent hard-checks column count against `n_features * n_time_bins`, so a mixed design only
    fits at all if this class reports its shape the way it does."""

    @staticmethod
    def _design(n=240, n_features=3, n_lags=8, seed=0, neural_signal=0.0):
        rng = np.random.default_rng(seed)
        behaviour = rng.normal(size=(n, n_features * n_lags))
        neural = rng.normal(size=(n, 1))
        angle = (0.6 * behaviour[:, 0] + 0.4 * behaviour[:, n_lags]
                 + neural_signal * neural[:, 0] + 0.05 * rng.normal(size=n))
        y = np.column_stack([angle % 1.0, (0.5 * angle) % 1.0])
        return behaviour, neural, y

    def test_a_mixed_design_fits(self):
        behaviour, neural, y = self._design()
        est = NestedTorusRegression(n_behaviour_features=3, n_lags=8, n_neural_columns=1)
        est.fit(np.hstack([behaviour, neural]), y)
        assert est.coef_.shape == (25, 4)          # 4-D torus embedding
        assert est.converged_

    def test_a_wrong_column_count_is_still_refused(self):
        behaviour, _neural, y = self._design()
        est = NestedTorusRegression(n_behaviour_features=3, n_lags=8, n_neural_columns=1)
        with pytest.raises(ValueError, match="columns"):
            est.fit(behaviour, y)                  # the neural column is missing

    def test_the_neural_column_still_receives_the_ridge(self):
        """Zero SMOOTHNESS must not mean zero regularisation -- `l2_reg` reaches every column through
        the parent's `+ l2_reg * I`, so a larger ridge must shrink the neural coefficient."""
        behaviour, neural, y = self._design(neural_signal=1.0)
        design = np.hstack([behaviour, neural])
        weak = NestedTorusRegression(n_behaviour_features=3, n_lags=8, n_neural_columns=1,
                                     l2_reg=1e-4).fit(design, y)
        strong = NestedTorusRegression(n_behaviour_features=3, n_lags=8, n_neural_columns=1,
                                       l2_reg=1e3).fit(design, y)
        assert np.abs(strong.coef_[-1]).sum() < np.abs(weak.coef_[-1]).sum()

    def test_the_full_model_tracks_a_neuron_the_behaviour_cannot_supply(self):
        """The synthetic gate for the nesting: when the target depends on a neural column that is
        independent of the behaviour block, the full model must fit better in-sample than the reduced
        one. (Out-of-sample scoring is the statistic; this only pins the mechanics.)"""
        behaviour, neural, y = self._design(neural_signal=2.0, seed=3)
        reduced = NestedTorusRegression(n_behaviour_features=3, n_lags=8,
                                        n_neural_columns=0).fit(behaviour, y)
        full = NestedTorusRegression(n_behaviour_features=3, n_lags=8,
                                     n_neural_columns=1).fit(np.hstack([behaviour, neural]), y)
        residual = lambda est, x: float(np.mean((est.predict(x, snap=False) - y) ** 2))  # noqa: E731
        assert residual(full, np.hstack([behaviour, neural])) < residual(reduced, behaviour)


class TestTheBehaviourControl:
    """Read P1's selection AND assert it against the frozen list -- reading alone lets a P1 re-run
    silently redefine 'beyond behaviour'; freezing alone lets the control drift from the selection."""

    FROZEN: ClassVar[list[str]] = ["self.neck_elevation", "nose-nose", "allo_yaw-nose",
                                   "self.back_pitch", "self.allo_roll"]

    @staticmethod
    def _fake_selection(tmp_path, features):
        path = tmp_path / "selection.pkl"
        with path.open("wb") as handle:
            pickle.dump({"steps": [{"final_model_features": list(features)}]}, handle)
        return path

    def test_the_shipped_settings_freeze_p1s_five_features(self):
        assert _settings()["nested_position_decoding"]["reduced_model_features"] == self.FROZEN

    def test_a_matching_file_returns_the_selection_order(self, tmp_path):
        settings = _settings()
        settings["data"]["behaviour_selection_result_path"] = str(
            self._fake_selection(tmp_path, self.FROZEN))
        assert reduced_model_features(settings) == self.FROZEN

    def test_drift_fails_loudly_rather_than_being_absorbed(self, tmp_path):
        settings = _settings()
        settings["data"]["behaviour_selection_result_path"] = str(
            self._fake_selection(tmp_path, [*self.FROZEN, "self.speed"]))
        with pytest.raises(ValueError, match="behaviour selection drift"):
            reduced_model_features(settings)

    def test_a_missing_file_raises_unless_explicitly_tolerated(self, tmp_path):
        settings = _settings()
        settings["data"]["behaviour_selection_result_path"] = str(tmp_path / "absent.pkl")
        with pytest.raises(FileNotFoundError, match="behaviour selection result not found"):
            reduced_model_features(settings)
        assert reduced_model_features(settings, require_selection_file=False) == self.FROZEN

    def test_a_present_file_is_checked_even_when_the_file_is_optional(self, tmp_path):
        """`require_selection_file=False` tolerates ABSENCE, never disagreement."""
        settings = _settings()
        settings["data"]["behaviour_selection_result_path"] = str(
            self._fake_selection(tmp_path, ["something.else"]))
        with pytest.raises(ValueError, match="behaviour selection drift"):
            reduced_model_features(settings, require_selection_file=False)

    def test_the_settings_do_not_mutate(self, tmp_path):
        settings = _settings()
        original = copy.deepcopy(settings["nested_position_decoding"]["reduced_model_features"])
        settings["data"]["behaviour_selection_result_path"] = str(
            self._fake_selection(tmp_path, self.FROZEN))
        reduced_model_features(settings)
        assert settings["nested_position_decoding"]["reduced_model_features"] == original
