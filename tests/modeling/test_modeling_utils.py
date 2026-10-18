"""
@author: bartulem
Unit tests for the pure helpers in ``usv_playpen.modeling.modeling_utils``:
the classification / regression metric wrappers, the two-class pooling and
balancing array ops, history unrolling, the forward-selection 1SE margin,
and the mouse-role / kinematic column selectors.

The metrics are checked against their textbook extremes (perfect / chance
/ degenerate), the array ops against shape and label invariants (with the
global RNG seeded for the down-sampling draws), the 1SE margin against the
paired/unpaired case it exists to separate, and the column selectors against
small synthetic column lists exercising the directional dyadic folding rules
documented in the source.

Coverage:

* classification / probabilistic metrics (multi-class Brier, ECE, safe MCC and
  confusion matrix, canonical class alignment) and regression metrics;
* the two-class array ops (session pooling, balancing, concatenation with
  labels, history unrolling, train/test shuffling, bounded test proportion);
* the paired 1SE forward-selection improvement;
* mouse-role resolution and kinematic / vocal-signal column selection;
* session-list loading, empty-event session detection, column harmonisation,
  cross-session z-scoring and the predictor-audit orchestration (including its
  keyword forwarding);
* the standardised console-output formatters and the univariate headline;
* the seeded session holdout, held-out ids from metadata, development /
  held-out masks, session-grouped inner CV and the manifold tag segment;
* mutation-round known-answer checks across the core helpers.
"""

from __future__ import annotations

from datetime import datetime
from itertools import pairwise
from typing import ClassVar

import numpy as np
import polars as pls
import pytest
from sklearn.model_selection import StratifiedGroupKFold

import usv_playpen.modeling.modeling_utils as mu
from usv_playpen.modeling.modeling_utils import (
    align_probs_to_canonical,
    balance_two_class_arrays,
    bounded_test_proportion,
    brier_score_multi,
    build_vocal_signal_columns,
    concat_two_class_with_labels,
    development_heldout_masks,
    expected_calibration_error,
    extract_univariate_headline,
    format_run_header,
    format_run_summary,
    format_selection_step,
    format_split_line,
    harmonize_session_columns,
    held_out_session_ids_from_metadata,
    identify_empty_event_sessions,
    manifold_tag_segment,
    mean_absolute_error_1d,
    paired_one_se_improvement,
    pearson_r_safe,
    pool_session_arrays,
    prepare_modeling_sessions,
    resolve_mouse_roles,
    root_mean_squared_error,
    run_predictor_audits,
    safe_confusion_matrix,
    safe_matthews_corrcoef,
    seeded_session_holdout,
    select_kinematic_columns,
    shuffle_train_test_arrays,
    spearman_r_safe,
    unroll_history_matrix,
    zscore_features_across_sessions,
)

# Probabilistic / classification metrics


class TestBrierScoreMulti:
    """The multi-class Brier score hits its textbook extremes (0 for perfect predictions, the
    closed form for uniform ones), since it is reported as a calibration headline."""

    def test_perfect_predictions_score_zero(self):
        """One-hot probabilities matching the labels give a Brier score of
        exactly 0."""

        classes = np.array([0, 1, 2])
        y_true = np.array([0, 1, 2, 0])
        proba = np.eye(3)[y_true]
        assert brier_score_multi(y_true, proba, classes) == pytest.approx(0.0)

    def test_uniform_predictions_match_closed_form(self):
        """A uniform K-class prediction has Brier score ``(K-1)/K`` per
        sample (here K=3 -> 2/3)."""

        classes = np.array([0, 1, 2])
        y_true = np.array([0, 1, 2])
        proba = np.full((3, 3), 1.0 / 3.0)
        assert brier_score_multi(y_true, proba, classes) == pytest.approx(2.0 / 3.0)


class TestExpectedCalibrationError:
    """ECE is 0 for confident correct predictions, positive for overconfident wrong ones, and
    refuses a non-2-D probability array instead of mis-binning it."""

    def test_perfectly_calibrated_confident_is_zero(self):
        """Confident (prob 1) and always-correct predictions are perfectly
        calibrated -> ECE 0."""

        y_true = np.array([0, 1, 0, 1])
        y_pred = y_true.copy()
        proba = np.eye(2)[y_true]
        assert expected_calibration_error(y_true, y_pred, proba) == pytest.approx(0.0)

    def test_overconfident_wrong_is_positive(self):
        """High-confidence but always-wrong predictions are badly
        miscalibrated -> ECE near 1."""

        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([1, 1, 1, 1])
        proba = np.tile([0.0, 1.0], (4, 1))
        assert expected_calibration_error(y_true, y_pred, proba) == pytest.approx(1.0)

    def test_non_2d_proba_raises(self):
        """A 1-D probability array is rejected with ``ValueError``."""

        with pytest.raises(ValueError, match=r'requires y_proba of shape \(N, K\)'):
            expected_calibration_error(np.array([0]), np.array([0]), np.array([1.0]))


class TestSafeMatthews:
    """The safe MCC is 1 on perfect agreement and 0 (not NaN) when predictions collapse to one
    class, so a degenerate fold cannot poison an average."""

    def test_perfect_agreement_is_one(self):
        """Identical label vectors give MCC == 1."""

        y = np.array([0, 1, 0, 1, 1])
        assert safe_matthews_corrcoef(y, y) == pytest.approx(1.0)

    def test_single_class_collapse_is_zero(self):
        """When predictions collapse to one class MCC is undefined;
        scikit-learn returns 0 and the wrapper preserves that."""

        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 0, 0, 0])
        assert safe_matthews_corrcoef(y_true, y_pred) == pytest.approx(0.0)


class TestSafeConfusionMatrix:
    """The safe confusion matrix keeps the canonical shape when a class is absent from a fold,
    so per-fold matrices can be summed."""

    def test_canonical_shape_with_missing_class(self):
        """Passing explicit ``labels`` forces the canonical (K, K) shape
        even when a class is never observed in this fold."""

        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        cm = safe_confusion_matrix(y_true, y_pred, labels=np.array([0, 1, 2]))
        assert cm.shape == (3, 3)
        # Perfect prediction -> mass only on the diagonal of observed classes.
        assert cm[0, 0] == 2
        assert cm[1, 1] == 2
        assert cm[2].sum() == 0


class TestAlignProbsToCanonical:
    """Fold probabilities are realigned to the canonical class order (absent classes zero-filled)
    and a non-ascending canonical order is rejected rather than silently shuffling columns."""

    def test_missing_class_column_filled_with_zeros(self):
        """A fold missing class 1 yields a canonical matrix whose class-1
        column is all zeros and whose other columns carry the model
        output."""

        probs = np.array([[0.6, 0.4], [0.3, 0.7]])  # model trained on classes 0, 2
        out = align_probs_to_canonical(probs, model_classes=np.array([0, 2]),
                                       canonical_classes=np.array([0, 1, 2]))
        assert out.shape == (2, 3)
        np.testing.assert_allclose(out[:, 0], probs[:, 0])
        np.testing.assert_allclose(out[:, 2], probs[:, 1])
        assert np.all(out[:, 1] == 0.0)

    def test_unsorted_canonical_raises(self):
        """A non-ascending canonical ordering would silently shuffle
        columns, so it is rejected."""

        probs = np.array([[1.0]])
        with pytest.raises(ValueError, match='strictly ascending'):
            align_probs_to_canonical(probs, model_classes=np.array([2]),
                                     canonical_classes=np.array([2, 1, 0]))


# Regression metrics


class TestRegressionMetrics:
    """Pearson, Spearman, RMSE and MAE hit their textbook extremes, give NaN on constant input,
    and keep RMSE >= MAE."""

    def test_pearson_perfect_and_anti(self):
        """Pearson r is +1 for a perfect linear match and -1 for its
        negation."""

        y = np.linspace(0.0, 1.0, 50)
        assert pearson_r_safe(y, 2.0 * y + 1.0) == pytest.approx(1.0)
        assert pearson_r_safe(y, -y) == pytest.approx(-1.0)

    def test_pearson_constant_input_is_nan(self):
        """A zero-variance input has undefined correlation -> NaN."""

        y = np.linspace(0.0, 1.0, 10)
        assert np.isnan(pearson_r_safe(y, np.ones_like(y)))

    def test_spearman_monotonic_reversal_and_constant(self):
        """Spearman rho is +1 for a perfectly monotonic (even non-linear) match,
        -1 for its reversal, and NaN for a zero-variance input (the NaN-safe guard
        that avoids scipy's ConstantInputWarning under filterwarnings=error)."""

        y = np.linspace(0.0, 1.0, 50)
        assert spearman_r_safe(y, np.exp(y)) == pytest.approx(1.0)   # monotonic, non-linear
        assert spearman_r_safe(y, -y) == pytest.approx(-1.0)
        assert np.isnan(spearman_r_safe(y, np.ones_like(y)))         # constant -> NaN

    def test_rmse_and_mae_zero_on_match(self):
        """Perfect predictions give RMSE == MAE == 0."""

        y = np.array([1.0, 2.0, 3.0])
        assert root_mean_squared_error(y, y) == pytest.approx(0.0)
        assert mean_absolute_error_1d(y, y) == pytest.approx(0.0)

    def test_rmse_ge_mae(self):
        """RMSE penalises large residuals more, so for any residuals it is
        >= MAE."""

        y_true = np.array([0.0, 0.0, 0.0, 0.0])
        y_pred = np.array([0.0, 0.0, 0.0, 4.0])
        rmse = root_mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error_1d(y_true, y_pred)
        assert rmse == pytest.approx(2.0)   # sqrt(16/4)
        assert mae == pytest.approx(1.0)    # 4/4
        assert rmse >= mae


# Two-class array ops


class TestPoolSessionArrays:
    """Session pooling stacks the sessions that are present, skips missing ones and returns a
    zero-row placeholder for an empty class, so downstream shapes stay valid."""

    def test_pools_present_sessions_and_skips_missing(self):
        """Arrays are concatenated across the sessions present in
        ``feature_data``; sessions absent from the dict are skipped
        without error."""

        feature_data = {
            's1': {'pos': np.ones((2, 4)), 'neg': np.zeros((3, 4))},
            's2': {'pos': np.ones((1, 4)), 'neg': np.zeros((0, 4))},
        }
        X_pos, X_neg = pool_session_arrays(
            feature_data, ['s1', 's2', 's_missing'],
            pos_key='pos', neg_key='neg', n_frames=4,
        )
        assert X_pos.shape == (3, 4)   # 2 + 1
        assert X_neg.shape == (3, 4)   # 3 + 0

    def test_empty_class_returns_zero_row_placeholder(self):
        """A class with no usable data anywhere returns a ``(0, n_frames)``
        placeholder."""

        feature_data = {'s1': {'pos': np.empty((0, 5)), 'neg': np.ones((2, 5))}}
        X_pos, X_neg = pool_session_arrays(
            feature_data, ['s1'], pos_key='pos', neg_key='neg', n_frames=5,
        )
        assert X_pos.shape == (0, 5)
        assert X_neg.shape == (2, 5)


class TestBalanceTwoClassArrays:
    """Two-class balancing down-samples the majority class to the minority size, reproducibly
    under a seed, and passes empty input through."""

    def test_downsamples_majority_to_minority(self):
        """The majority class is subsampled down to the minority count;
        both outputs end up with ``min(n_pos, n_neg)`` rows."""

        X_pos = np.arange(100 * 3).reshape(100, 3).astype(float)
        X_neg = np.arange(10 * 3).reshape(10, 3).astype(float)
        bp, bn = balance_two_class_arrays(X_pos, X_neg, np.random.default_rng(0))
        assert bp.shape == (10, 3)
        assert bn.shape == (10, 3)

    def test_reproducible_under_seed(self):
        """The down-sampling draw comes from the caller's generator, so the same
        seed gives an identical subsample whatever the global RNG holds."""

        X_pos = np.arange(100 * 2).reshape(100, 2).astype(float)
        X_neg = np.zeros((10, 2))
        bp1, _ = balance_two_class_arrays(X_pos, X_neg, np.random.default_rng(7))
        np.random.seed(123)  # noqa: NPY002 -- global state must not matter
        bp2, _ = balance_two_class_arrays(X_pos, X_neg, np.random.default_rng(7))
        np.testing.assert_array_equal(bp1, bp2)

    def test_empty_input_returns_empty(self):
        """If one class is empty both outputs are empty with preserved
        column counts."""

        bp, bn = balance_two_class_arrays(np.empty((0, 4)), np.ones((5, 4)), np.random.default_rng(0))
        assert bp.shape == (0, 4)
        assert bn.shape == (0, 4)


class TestConcatTwoClassWithLabels:
    """Positive and negative arrays are stacked with 1 / 0 labels aligned to their rows."""

    def test_stacks_and_labels(self):
        """Positive rows come first with label 1, negatives after with
        label 0; the stacked feature matrix preserves that order."""

        X_pos = np.ones((2, 3))
        X_neg = np.zeros((3, 3))
        X, y = concat_two_class_with_labels(X_pos, X_neg)
        assert X.shape == (5, 3)
        np.testing.assert_array_equal(y, np.array([1.0, 1.0, 0.0, 0.0, 0.0]))


class TestUnrollHistoryMatrix:
    """History unrolling flattens ``(n_samples, n_frames)`` into one row per (sample, lag) with
    the lag index (default or custom) tiled into the time column."""

    def test_layout_and_time_column(self):
        """Unrolling ``(n_samples, n_frames)`` produces
        ``(n_samples * n_frames, 2)`` with row-major feature values in
        column 0 and tiled lag indices in column 1."""

        X = np.array([[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]])
        out = unroll_history_matrix(X)
        assert out.shape == (6, 2)
        assert out.dtype == np.float32
        np.testing.assert_array_equal(out[:, 0], X.ravel())
        np.testing.assert_array_equal(out[:, 1], np.array([0, 1, 2, 0, 1, 2]))

    def test_custom_time_indices(self):
        """An explicit ``time_indices`` array is tiled into column 1."""

        X = np.zeros((2, 2))
        out = unroll_history_matrix(X, time_indices=np.array([5.0, 9.0]))
        np.testing.assert_array_equal(out[:, 1], np.array([5.0, 9.0, 5.0, 9.0]))


class TestShuffleTrainTestArrays:
    """Train/test shuffling permutes rows without breaking the feature-label alignment."""

    def test_preserves_label_alignment_and_shapes(self):
        """Shuffling permutes rows but keeps each X row paired with its y
        label and leaves shapes unchanged."""

        X_train = np.arange(20).reshape(10, 2).astype(float)
        y_train = X_train[:, 0].copy()  # label encodes the row identity
        X_test = np.arange(8).reshape(4, 2).astype(float)
        y_test = X_test[:, 0].copy()
        Xtr, ytr, Xte, yte = shuffle_train_test_arrays(X_train, y_train, X_test, y_test, np.random.default_rng(1))
        assert Xtr.shape == X_train.shape
        assert Xte.shape == X_test.shape
        # Column-0 value still equals the paired label after the shuffle.
        np.testing.assert_array_equal(Xtr[:, 0], ytr)
        np.testing.assert_array_equal(Xte[:, 0], yte)


class TestBoundedTestProportion:
    """The test proportion is raised to a floor when too few sessions would be held out, kept
    when already sufficient, and returned unchanged with zero sessions."""

    def test_raises_floor_for_small_session_count(self):
        """A nominal 0.2 across 3 sessions would round to 0 test sessions,
        so the proportion is raised to ``1/3``."""

        assert bounded_test_proportion(0.2, n_sessions=3) == pytest.approx(1.0 / 3.0)

    def test_keeps_proportion_when_already_sufficient(self):
        """When the nominal proportion already yields >= 1 test session it
        is returned unchanged."""

        assert bounded_test_proportion(0.5, n_sessions=10) == pytest.approx(0.5)

    def test_zero_sessions_returns_raw(self):
        """With no sessions the caller short-circuits, so the raw value is
        returned untouched."""

        assert bounded_test_proportion(0.2, n_sessions=0) == pytest.approx(0.2)


class TestPairedOneSeImprovement:
    """The forward-selection 1SE margin compares the PAIRED per-fold improvement to its own SE:
    it accepts consistent gains the unpaired rule rejects, follows the metric's sign, drops
    failed folds pairwise and returns NaN when no fold is usable."""

    def test_consistent_improvement_is_accepted_where_unpaired_rejects(self):
        """The case the paired rule exists for: a candidate that improves every
        fold by the same amount has zero spread in its per-fold DIFFERENCE, so
        it clears the 1SE bar, while the SE of its own absolute scores -- driven
        by how hard each fold is -- would reject it."""

        incumbent = [0.60, 0.50, 0.70, 0.55, 0.65, 0.58, 0.62, 0.53, 0.68, 0.57]
        candidate = [x - 0.01 for x in incumbent]

        improvement, standard_error = paired_one_se_improvement(
            candidate, incumbent, higher_is_better=False)

        assert improvement == pytest.approx(0.01)
        assert standard_error == pytest.approx(0.0, abs=1e-12)
        assert improvement > standard_error

        unpaired = float(np.std(candidate, ddof=1) / np.sqrt(len(candidate)))
        assert improvement < unpaired

    def test_sign_convention_follows_higher_is_better(self):
        """A positive return always means the candidate is better, whichever
        direction the metric runs."""

        worse_on_a_lower_is_better_metric = paired_one_se_improvement(
            [0.7, 0.7], [0.6, 0.6], higher_is_better=False)[0]
        better_on_a_higher_is_better_metric = paired_one_se_improvement(
            [0.7, 0.7], [0.6, 0.6], higher_is_better=True)[0]

        assert worse_on_a_lower_is_better_metric == pytest.approx(-0.1)
        assert better_on_a_higher_is_better_metric == pytest.approx(0.1)

    def test_scalar_incumbent_reduces_to_the_unpaired_case(self):
        """Pairing against a constant baseline has the same spread as the
        candidate's own scores, so the scalar branch is exact rather than an
        approximation."""

        candidate = [0.61, 0.49, 0.72, 0.55]

        _, standard_error = paired_one_se_improvement(candidate, 0.5, higher_is_better=True)

        assert standard_error == pytest.approx(
            float(np.std(candidate, ddof=1) / np.sqrt(len(candidate))))

    def test_folds_are_dropped_pairwise_when_either_model_failed(self):
        """A fold carries no information about the difference unless both
        models scored it, so it is dropped as a pair."""

        improvement, standard_error = paired_one_se_improvement(
            [0.10, np.nan, 0.30], [0.05, 0.20, np.nan], higher_is_better=True)

        assert improvement == pytest.approx(0.05)
        assert standard_error == pytest.approx(0.0)

    def test_no_usable_fold_returns_nan_so_the_caller_rejects(self):
        """`np.isfinite(nan)` is False at every call site, so an all-failed
        candidate can never be accepted."""

        improvement, standard_error = paired_one_se_improvement(
            [np.nan, np.nan], [0.1, 0.2], higher_is_better=True)

        assert np.isnan(improvement)
        assert standard_error == pytest.approx(0.0)

    def test_mismatched_fold_counts_raise(self):
        """Folds are paired by position, so unequal lengths are a caller bug
        rather than something to silently truncate."""

        with pytest.raises(ValueError, match="one score per fold"):
            paired_one_se_improvement([0.1, 0.2, 0.3], [0.1, 0.2])


# Mouse-role / kinematic column selectors


class TestResolveMouseRoles:
    """The target mouse is the slot opposite the predictor index, with names swapped to match."""

    def test_target_is_opposite_slot(self):
        """The target index is the opposite two-mouse slot of the
        predictor, and names are looked up by slot."""

        settings = {'model_params': {'model_predictor_mouse_index': 0}}
        names = {'sess': ['m_male', 'm_female']}
        p_idx, t_idx, p_name, t_name = resolve_mouse_roles(settings, names, 'sess')
        assert (p_idx, t_idx) == (0, 1)
        assert p_name == 'm_male'
        assert t_name == 'm_female'

    def test_predictor_index_one(self):
        """Predictor slot 1 -> target slot 0, names swapped accordingly."""

        settings = {'model_params': {'model_predictor_mouse_index': 1}}
        names = {'sess': ['m_male', 'm_female']}
        p_idx, t_idx, p_name, t_name = resolve_mouse_roles(settings, names, 'sess')
        assert (p_idx, t_idx) == (1, 0)
        assert p_name == 'm_female'
        assert t_name == 'm_male'


class TestSelectKinematicColumns:
    """Kinematic column selection keeps egocentric features for both mice, adds derivatives only
    when flagged, and applies the directional dyadic-pose and engagement folding rules."""

    def _kin_settings(self, **overrides):
        """
        Description
        -----------
        A minimal kinematic-settings block (egocentric ``speed`` only, no dyadic
        features, symmetric dyadic pose, no derivatives) with ``overrides`` applied.

        Parameters
        ----------
        **overrides
            Keys of the kinematic-settings block to replace.

        Returns
        -------
        kin_settings (dict)
            The settings block passed to ``select_kinematic_columns``.
        """
        base = {
            'egocentric': ['speed'],
            'dyadic_pose': [],
            'dyadic_engagement': [],
            'dyadic_pose_symmetric': True,
            'include_1st_derivatives': False,
            'include_2nd_derivatives': False,
        }
        base.update(overrides)
        return base

    def test_egocentric_keeps_both_mice(self):
        """An egocentric base feature keeps both the target and predictor
        per-mouse columns when present."""

        cols = ['mA.speed', 'mB.speed', 'mA.other', 'unrelated']
        out = select_kinematic_columns(
            cols, target_name='mA', predictor_name='mB',
            kin_settings=self._kin_settings(), predictor_idx=0,
        )
        assert out == ['mA.speed', 'mB.speed']

    def test_egocentric_derivatives_added_when_flagged(self):
        """First-derivative columns are added when the flag is on and the
        column exists; speed derivatives are explicitly excluded."""

        cols = ['mA.snout_speed', 'mA.snout_speed_1st_der', 'mA.speed', 'mA.speed_1st_der']
        out = select_kinematic_columns(
            cols, target_name='mA', predictor_name='mB',
            kin_settings=self._kin_settings(
                egocentric=['snout_speed', 'speed'], include_1st_derivatives=True,
            ),
            predictor_idx=0,
        )
        assert 'mA.snout_speed_1st_der' in out      # non-speed derivative kept
        assert 'mA.speed_1st_der' not in out         # speed derivative excluded

    def test_dyadic_engagement_keeps_target_observer_direction(self):
        """For engagement features only the ``target-predictor`` (focal
        observer) orientation is retained; the reverse is dropped."""

        cols = ['mA-mB.orofacial-sei', 'mB-mA.orofacial-sei']
        out = select_kinematic_columns(
            cols, target_name='mA', predictor_name='mB',
            kin_settings=self._kin_settings(dyadic_engagement=['orofacial-sei']),
            predictor_idx=0,
        )
        assert out == ['mA-mB.orofacial-sei']

    def test_dyadic_pose_directional_drop(self):
        """With ``dyadic_pose_symmetric=False`` and predictor slot 0, the
        ``allo_yaw`` half is dropped per the documented directional rule."""

        cols = ['mA-mB.allo_yaw-tti', 'mA-mB.tti-allo_yaw']
        out = select_kinematic_columns(
            cols, target_name='mA', predictor_name='mB',
            kin_settings=self._kin_settings(
                dyadic_pose=['allo_yaw-tti', 'tti-allo_yaw'],
                dyadic_pose_symmetric=False,
            ),
            predictor_idx=0,
        )
        # predictor_idx == 0 drops suffixes whose first part is 'allo_yaw'.
        assert 'mA-mB.allo_yaw-tti' not in out
        assert 'mA-mB.tti-allo_yaw' in out

    def test_dyadic_pose_directional_drop_predictor_one(self):
        """With predictor slot != 0 the directional rule mirrors: the half
        whose *second* part is ``allo_yaw`` (or whose first part is ``TTI``)
        is dropped, exercising the ``predictor_idx != 0`` branch."""

        cols = ['mA-mB.allo_yaw-tti', 'mA-mB.tti-allo_yaw', 'mA-mB.TTI-foo',
                'mA-mB.foo-TTI']
        out = select_kinematic_columns(
            cols, target_name='mA', predictor_name='mB',
            kin_settings=self._kin_settings(
                dyadic_pose=['allo_yaw-tti', 'tti-allo_yaw', 'TTI-foo', 'foo-TTI'],
                dyadic_pose_symmetric=False,
            ),
            predictor_idx=1,
        )
        # predictor_idx != 0 -> drop when feat_parts[1] in allo angles or
        # feat_parts[0] == 'TTI'.
        assert 'mA-mB.tti-allo_yaw' not in out      # second part allo_yaw -> dropped
        assert 'mA-mB.allo_yaw-tti' in out          # first part allo_yaw -> kept
        assert 'mA-mB.TTI-foo' not in out           # first part TTI -> dropped
        assert 'mA-mB.foo-TTI' in out               # second part TTI -> kept

    def test_dyadic_pose_symmetric_keeps_both(self):
        """When ``dyadic_pose_symmetric`` is True both directional halves are
        retained regardless of predictor slot."""

        cols = ['mA-mB.allo_yaw-tti', 'mA-mB.tti-allo_yaw']
        out = select_kinematic_columns(
            cols, target_name='mA', predictor_name='mB',
            kin_settings=self._kin_settings(
                dyadic_pose=['allo_yaw-tti', 'tti-allo_yaw'],
                dyadic_pose_symmetric=True,
            ),
            predictor_idx=0,
        )
        assert 'mA-mB.allo_yaw-tti' in out
        assert 'mA-mB.tti-allo_yaw' in out

    def test_second_derivatives_added_when_flagged(self):
        """Second-derivative columns are added for non-speed egocentric
        features when ``include_2nd_derivatives`` is set and the column
        exists."""

        cols = ['mA.snout_speed', 'mA.snout_speed_2nd_der']
        out = select_kinematic_columns(
            cols, target_name='mA', predictor_name='mB',
            kin_settings=self._kin_settings(
                egocentric=['snout_speed'], include_2nd_derivatives=True,
            ),
            predictor_idx=0,
        )
        assert 'mA.snout_speed_2nd_der' in out

    def test_dyadic_pose_non_two_part_suffix_kept(self):
        """A dyadic-pose base feature whose suffix does not split into two
        parts on ``-`` bypasses the directional rule (``len != 2`` branch)
        and is kept even when asymmetric."""

        cols = ['mA-mB.singlepart']
        out = select_kinematic_columns(
            cols, target_name='mA', predictor_name='mB',
            kin_settings=self._kin_settings(
                dyadic_pose=['singlepart'],
                dyadic_pose_symmetric=False,
            ),
            predictor_idx=0,
        )
        assert out == ['mA-mB.singlepart']


# Session / RNG preparation


class TestPrepareModelingSessions:
    """The session-list loader skips blank lines, leaves the global RNG untouched and raises on a
    missing or empty file."""

    def _settings(self, seed, session_file):
        """
        Description
        -----------
        The two settings blocks ``prepare_modeling_sessions`` reads: the
        ``model_validation`` random seed and the ``io`` session-list path.

        Parameters
        ----------
        seed (int | None)
            The configured ``random_seed``.
        session_file (pathlib.Path)
            Path of the session-list text file.

        Returns
        -------
        settings (dict)
            The minimal modeling settings.
        """
        # `random_seed` moved from `model_params` into the `model_validation`
        # block; `prepare_modeling_sessions` reads it exclusively from there.
        return {'model_validation': {'random_seed': seed},
                'io': {'session_list_file': str(session_file)}}

    def test_loads_paths_and_leaves_the_global_rng_alone(self, tmp_path):
        """A session-list file with blank lines is read into a path list (blanks
        skipped), and the global RNG state is left exactly as it was, whatever the
        configured seed."""

        f = tmp_path / 'sessions.txt'
        f.write_text("/some/sessionA\n\n/some/sessionB\n")
        for seed in (123, None):
            state = np.random.get_state()  # noqa: NPY002 -- checking global state is untouched
            out = prepare_modeling_sessions(self._settings(seed, f))
            assert out == ['/some/sessionA', '/some/sessionB']
            after = np.random.get_state()  # noqa: NPY002
            assert all(np.array_equal(x, y) for x, y in zip(state, after, strict=True))

    def test_missing_file_raises_filenotfound(self, tmp_path):
        """A non-existent session-list file raises ``FileNotFoundError``."""

        with pytest.raises(FileNotFoundError):
            prepare_modeling_sessions(self._settings(0, tmp_path / 'nope.txt'))

    def test_empty_file_raises_valueerror(self, tmp_path):
        """A file with only blank lines yields no paths -> ``ValueError``."""

        f = tmp_path / 'blank.txt'
        f.write_text("\n   \n\n")
        with pytest.raises(ValueError, match='No valid session paths found'):
            prepare_modeling_sessions(self._settings(0, f))


# Vocal signal column construction


class TestBuildVocalSignalColumns:
    """Vocal predictor columns honour the self-exclusion and partner-only rules and skip mice
    absent from the vocal dict, so a mouse's own vocal trace never predicts itself."""

    def _usv_dict(self):
        """
        Description
        -----------
        A one-session vocal dict with a target mouse (rate, event and one category
        trace) and a partner mouse (rate and one category trace), three frames each.

        Returns
        -------
        usv_dict (dict)
            ``{session: {mouse: {'continuous_vocal_signals': {name: trace}}}}``.
        """
        return {
            'sess': {
                'mTarget': {'continuous_vocal_signals': {
                    'usv_rate': np.arange(3, dtype=np.float32),
                    'usv_event': np.ones(3, dtype=np.float32),
                    'usv_cat_0': np.array([1.0, 0.0, 1.0], dtype=np.float32),
                }},
                'mPartner': {'continuous_vocal_signals': {
                    'usv_rate': np.array([2.0, 2.0, 2.0], dtype=np.float32),
                    'usv_cat_0': np.array([0.0, 1.0, 0.0], dtype=np.float32),
                }},
            }
        }

    def test_falsy_predictor_type_returns_empty(self):
        """A falsy ``vocal_predictor_type`` short-circuits to empty lists."""

        cols, names = build_vocal_signal_columns(
            self._usv_dict(), 'sess', 'mTarget', 'mPartner',
            voc_settings={'vocal_predictor_type': None,
                          'vocal_predictor_partner_only': False},
        )
        assert cols == []
        assert names == []

    def test_self_exclusion_drops_rate_and_event(self):
        """For the target mouse the default ``usv_self_exclude`` keys
        (rate/event) are skipped, but per-category signals survive; the
        partner mouse keeps everything."""

        cols, names = build_vocal_signal_columns(
            self._usv_dict(), 'sess', 'mTarget', 'mPartner',
            voc_settings={'vocal_predictor_type': 'continuous',
                          'vocal_predictor_partner_only': False},
        )
        assert 'mTarget.usv_rate' not in names
        assert 'mTarget.usv_event' not in names
        assert 'mTarget.usv_cat_0' in names          # category survives
        assert 'mPartner.usv_rate' in names          # partner keeps rate
        assert 'mPartner.usv_cat_0' in names
        # The Series list mirrors the names list one-to-one.
        assert [s.name for s in cols] == names

    def test_self_exclusion_covers_every_pooled_trace_name(self):
        """Every pooled trace of the target mouse is left out whatever its name
        ('squeak_rate', 'overlap_event', 'all_vocalizations_rate'), its per-category
        traces of any name survive, and an extra ``usv_self_exclude`` key is left out
        on top; the partner keeps everything."""

        signals = {
            'squeak_rate': np.zeros(3), 'overlap_event': np.zeros(3), 'all_vocalizations_rate': np.zeros(3),
            'overlap_cat_2': np.zeros(3), 'usv_cat_1': np.zeros(3),
        }
        d = {'sess': {'mTarget': {'continuous_vocal_signals': dict(signals)},
                      'mPartner': {'continuous_vocal_signals': dict(signals)}}}
        _cols, names = build_vocal_signal_columns(
            d, 'sess', 'mTarget', 'mPartner',
            voc_settings={'vocal_predictor_type': 'pooled_rate', 'vocal_predictor_partner_only': False},
            usv_self_exclude=('usv_cat_1',),
        )
        assert [n for n in names if n.startswith('mTarget.')] == ['mTarget.overlap_cat_2']
        assert sorted(n for n in names if n.startswith('mPartner.')) == sorted(f"mPartner.{k}" for k in signals)

    def test_partner_only_emits_partner_signals_only(self):
        """``vocal_predictor_partner_only`` restricts emission to the partner
        mouse."""

        _cols, names = build_vocal_signal_columns(
            self._usv_dict(), 'sess', 'mTarget', 'mPartner',
            voc_settings={'vocal_predictor_type': 'continuous',
                          'vocal_predictor_partner_only': True},
        )
        assert all(n.startswith('mPartner.') for n in names)
        assert names  # non-empty

    def test_mouse_absent_from_dict_skipped(self):
        """A mouse missing from the session's USV entry is skipped without
        error (only the present mouse's signals appear)."""

        d = self._usv_dict()
        del d['sess']['mPartner']
        _cols, names = build_vocal_signal_columns(
            d, 'sess', 'mTarget', 'mPartner',
            voc_settings={'vocal_predictor_type': 'continuous',
                          'vocal_predictor_partner_only': True},
        )
        # partner_only=True but partner absent -> nothing emitted.
        assert names == []


# Empty-event session identification


class TestIdentifyEmptyEventSessions:
    """Sessions with a missing or empty event array are flagged (and reported) so they can be
    excluded before fitting."""

    def test_flags_missing_and_empty_sessions(self, capsys):
        """Sessions are flagged when absent from the USV dict, when the
        target mouse is absent, when the event key is missing, or when the
        event array is empty; a populated session is retained."""

        names = {
            's_ok': ['mM', 'mF'],
            's_no_session': ['mM', 'mF'],
            's_no_target': ['mM', 'mF'],
            's_no_key': ['mM', 'mF'],
            's_empty': ['mM', 'mF'],
        }
        usv = {
            's_ok': {'mF': {'positive_events': np.array([1.0, 2.0])}},
            's_no_target': {'mM': {'positive_events': np.array([1.0])}},
            's_no_key': {'mF': {'other_key': np.array([1.0])}},
            's_empty': {'mF': {'positive_events': np.array([])}},
        }
        removed = identify_empty_event_sessions(
            usv, names, target_idx=1, event_key='positive_events',
        )
        assert set(removed) == {
            's_no_session', 's_no_target', 's_no_key', 's_empty',
        }
        assert 's_ok' not in removed
        # Order is deterministic (mouse_names_dict iteration order).
        assert removed == ['s_no_session', 's_no_target', 's_no_key', 's_empty']
        _ = capsys.readouterr()  # drain prints


# Column harmonization


class TestHarmonizeSessionColumns:
    """Column harmonisation renames dyad-prefixed columns to their suffix, zero-fills columns a
    session lacks and gates USV columns by the existence map, so every session shares one
    column set."""

    def test_dyad_rename_and_zero_fill_and_suffixes(self):
        """Dyadic-prefixed columns are renamed to their suffix; non-USV
        ego columns are zero-filled across sessions; dyadic columns missing
        in a session are zero-filled; the returned suffix list is the sorted
        non-numeric union."""

        d = {
            's1': pls.DataFrame({
                'mM.speed': [1.0, 2.0],
                'mM-mF.nose-nose': [0.1, 0.2],   # dyadic -> renamed
            }),
            's2': pls.DataFrame({
                'mF.speed': [3.0, 4.0],
            }),
        }
        names = {'s1': ['mM', 'mF'], 's2': ['mM', 'mF']}
        out, suffixes = harmonize_session_columns(
            d, names, target_idx=1, predictor_idx=0,
        )
        # Dyadic column renamed to bare suffix in s1.
        assert 'nose-nose' in out['s1'].columns
        # s2 missing dyadic column -> zero-filled standalone.
        assert 'nose-nose' in out['s2'].columns
        assert out['s2']['nose-nose'].to_list() == [0.0, 0.0]
        # Non-USV ego suffix 'speed' filled for both roles everywhere.
        assert 'mM.speed' in out['s1'].columns
        assert 'mF.speed' in out['s1'].columns
        assert 'mM.speed' in out['s2'].columns
        assert 'mF.speed' in out['s2'].columns
        assert suffixes == sorted(suffixes)
        assert 'speed' in suffixes
        assert 'nose-nose' in suffixes

    def test_usv_gated_by_existence_map(self):
        """A ``usv_*`` ego suffix is only zero-filled where the matching
        generic role key was populated somewhere; an absent role is not
        spuriously filled."""

        # Only the partner ('other', slot 0 = mM) carries a usv_rate column.
        d = {
            's1': pls.DataFrame({'mM.usv_rate': [1.0, 2.0]}),
            's2': pls.DataFrame({'mM.speed': [3.0, 4.0]}),
        }
        names = {'s1': ['mM', 'mF'], 's2': ['mM', 'mF']}
        out, _suffixes = harmonize_session_columns(
            d, names, target_idx=1, predictor_idx=0,
        )
        # 'other.usv_rate' present in map -> partner (mM) gets zero-filled in s2.
        assert 'mM.usv_rate' in out['s2'].columns
        # 'self.usv_rate' never populated -> target (mF) NOT zero-filled.
        assert 'mF.usv_rate' not in out['s1'].columns
        assert 'mF.usv_rate' not in out['s2'].columns


# Cross-session z-scoring wrapper


class TestZscoreFeaturesAcrossSessions:
    """Cross-session z-scoring pools the sessions so the pooled feature has zero mean and unit
    sample spread."""

    def test_pools_and_zscores_in_place(self):
        """The wrapper forwards to the pooled z-scorer: a feature suffix is
        standardized against the across-session pooled mean/std (pooled mean
        becomes ~0)."""

        d = {
            's1': pls.DataFrame({'mA.speed': [0.0, 2.0, 4.0]}),
            's2': pls.DataFrame({'mB.speed': [6.0, 8.0, 10.0]}),
        }
        out = zscore_features_across_sessions(
            d, suffixes=['speed'], feature_bounds={'speed': (-1e9, 1e9)},
        )
        pooled = np.concatenate([
            out['s1']['mA.speed'].to_numpy(),
            out['s2']['mB.speed'].to_numpy(),
        ])
        assert np.nanmean(pooled) == pytest.approx(0.0, abs=1e-6)
        # Pooled spread is unit-scale (sample-std normalization, ddof=1).
        assert np.nanstd(pooled, ddof=1) == pytest.approx(1.0, abs=1e-6)


# Predictor audits orchestration


class TestRunPredictorAudits:
    """The predictor-audit orchestration runs the enabled audits with derived paths and event
    times, and keeps every audit failure non-fatal."""

    def _settings(self, collinearity=True, timescale=True):
        """
        Description
        -----------
        A minimal modeling-settings dict for ``run_predictor_audits``: the
        diagnostics block with each audit switchable, plus the model, validation and
        mixture-model parameters the event extraction reads.

        Parameters
        ----------
        collinearity (bool)
            Whether the collinearity audit is enabled.
        timescale (bool)
            Whether the timescale audit is enabled.

        Returns
        -------
        settings (dict)
            The settings dict.
        """
        return {
            'diagnostics': {
                'collinearity_audit': collinearity,
                'collinearity_concern_threshold': 0.7,
                'collinearity_exclude_threshold': 0.85,
                'timescale_audit': timescale,
                'timescale_max_lag_seconds': 5.0,
                'timescale_n_shuffles': 3,
                'timescale_shuffle_range': [20.0, 60.0],
                'timescale_signal_floor_seconds': 0.5,
                'timescale_signal_min_run_seconds': 0.2,
            },
            'model_params': {
                'filter_history': 1.0,
                'mixture_model_component_index': 0,
                'mixture_model_z_score': 1.0,
            },
            # `random_seed` moved into `model_validation`; the timescale audit's
            # shuffle RNG reads it exclusively from there.
            'model_validation': {
                'random_seed': 0,
            },
            'mixture_model_params': {
                'male': {'means': [0.1], 'sds': [0.05]},
                'female': {'means': [0.12], 'sds': [0.06]},
            },
        }

    def _inputs(self):
        """
        Description
        -----------
        A one-session audit input: a three-frame behavioural table, the female's two
        USV events, the mouse names and the camera frame rate.

        Returns
        -------
        inputs (tuple)
            ``(beh, usv, names, fps)`` dicts keyed by session.
        """
        beh = {'s1': pls.DataFrame({'mF.speed': [0.0, 1.0, 2.0]})}
        usv = {'s1': {'mF': {
            'positive_events': np.array([0.5, 1.5]),
            'start': np.array([0.5, 1.5]),
            'stop': np.array([0.6, 1.6]),
        }}}
        names = {'s1': ['mM', 'mF']}
        fps = {'s1': 30.0}
        return beh, usv, names, fps

    def test_returns_early_when_both_audits_disabled(self, mocker):
        """When neither audit is enabled the wrapper returns without invoking
        either audit function or touching disk."""

        coll = mocker.patch.object(mu, 'audit_predictor_collinearity')
        ts = mocker.patch.object(mu, 'audit_predictor_timescales')
        beh, usv, names, fps = self._inputs()
        run_predictor_audits(
            beh, usv, names, fps, target_idx=1, predictor_idx=0,
            history_frames=5, event_keys=['positive_events'],
            settings=self._settings(collinearity=False, timescale=False),
            save_dir='/unused', pickle_basename='in.pkl',
        )
        coll.assert_not_called()
        ts.assert_not_called()

    def test_invokes_both_audits_with_derived_paths(self, mocker, tmp_path):
        """Both audit functions are called once with artifact paths derived
        from the pickle basename and the pooled event-time / IBI-threshold
        inputs assembled by the wrapper."""

        coll = mocker.patch.object(mu, 'audit_predictor_collinearity')
        ts = mocker.patch.object(mu, 'audit_predictor_timescales')
        beh, usv, names, fps = self._inputs()
        run_predictor_audits(
            beh, usv, names, fps, target_idx=1, predictor_idx=0,
            history_frames=5, event_keys=['positive_events'],
            settings=self._settings(),
            save_dir=str(tmp_path), pickle_basename='myinput.pkl',
        )
        coll.assert_called_once()
        ts.assert_called_once()
        coll_kwargs = coll.call_args.kwargs
        assert coll_kwargs['save_path'].endswith('myinput_collinearity.pkl')
        assert 's1' in coll_kwargs['event_times_per_session']
        ts_kwargs = ts.call_args.kwargs
        assert ts_kwargs['save_path'].endswith('myinput_timescales.pkl')
        # IBI thresholds computed for both sexes from the mixture-model params.
        assert set(ts_kwargs['ibi_thresholds']) == {'male', 'female'}
        # Bout-onset Y trace built from default 'positive_events' key.
        assert 's1' in ts_kwargs['onset_times_per_session']

    def test_balance_event_keys_subsamples_per_session(self, mocker, tmp_path):
        """With ``balance_event_keys`` and two event keys of unequal size the
        pooled event-time array is down-sampled to twice the smaller key
        (one min-sized draw per key)."""

        coll = mocker.patch.object(mu, 'audit_predictor_collinearity')
        mocker.patch.object(mu, 'audit_predictor_timescales')
        beh, _, names, fps = self._inputs()
        usv = {'s1': {'mF': {
            'target_events': np.array([0.1, 0.2, 0.3, 0.4]),
            'other_events': np.array([0.9]),
            'start': np.array([0.1]),
            'stop': np.array([0.15]),
        }}}
        run_predictor_audits(
            beh, usv, names, fps, target_idx=1, predictor_idx=0,
            history_frames=5, event_keys=['target_events', 'other_events'],
            settings=self._settings(timescale=False),
            save_dir=str(tmp_path), pickle_basename='b.pkl',
            balance_event_keys=True,
            onset_event_key='target_events',
        )
        pooled = coll.call_args.kwargs['event_times_per_session']['s1']
        assert pooled.size == 2     # 1 (min) per key, two keys

    def test_precomputed_event_times_bypasses_extraction(self, mocker, tmp_path):
        """Supplying ``precomputed_event_times`` skips the per-mouse
        extraction loop and forwards the dict verbatim."""

        coll = mocker.patch.object(mu, 'audit_predictor_collinearity')
        mocker.patch.object(mu, 'audit_predictor_timescales')
        beh, usv, names, fps = self._inputs()
        pre = {'s1': np.array([9.0, 9.5, 9.9])}
        run_predictor_audits(
            beh, usv, names, fps, target_idx=1, predictor_idx=0,
            history_frames=5, event_keys=['positive_events'],
            settings=self._settings(timescale=False),
            save_dir=str(tmp_path), pickle_basename='p.pkl',
            precomputed_event_times=pre,
        )
        np.testing.assert_array_equal(
            coll.call_args.kwargs['event_times_per_session']['s1'], pre['s1'],
        )

    def test_collinearity_failure_is_non_fatal(self, mocker, tmp_path, capsys):
        """An exception raised inside the collinearity audit is caught and
        printed, not propagated."""

        mocker.patch.object(mu, 'audit_predictor_collinearity',
                            side_effect=RuntimeError("boom"))
        mocker.patch.object(mu, 'audit_predictor_timescales')
        beh, usv, names, fps = self._inputs()
        run_predictor_audits(
            beh, usv, names, fps, target_idx=1, predictor_idx=0,
            history_frames=5, event_keys=['positive_events'],
            settings=self._settings(timescale=False),
            save_dir=str(tmp_path), pickle_basename='f.pkl',
        )
        assert 'collinearity audit failed' in capsys.readouterr().out

    def test_missing_bout_onset_key_raises_into_nonfatal_path(self, mocker, tmp_path, capsys):
        """A ``onset_event_key`` absent from every target entry raises
        the universal-absence guard, which the timescale try/except converts
        into a non-fatal warning."""

        mocker.patch.object(mu, 'audit_predictor_collinearity')
        ts = mocker.patch.object(mu, 'audit_predictor_timescales')
        beh, usv, names, fps = self._inputs()
        run_predictor_audits(
            beh, usv, names, fps, target_idx=1, predictor_idx=0,
            history_frames=5, event_keys=['positive_events'],
            settings=self._settings(collinearity=False),
            save_dir=str(tmp_path), pickle_basename='g.pkl',
            onset_event_key='does_not_exist',
        )
        ts.assert_not_called()      # guard raised before the audit call
        assert 'timescale audit failed' in capsys.readouterr().out

    def test_precomputed_bout_onset_times_used(self, mocker, tmp_path):
        """``precomputed_onset_times`` bypasses the string-key lookup and
        is forwarded (sorted, empties dropped) as the timescale ``Y`` source."""

        mocker.patch.object(mu, 'audit_predictor_collinearity')
        ts = mocker.patch.object(mu, 'audit_predictor_timescales')
        beh, usv, names, fps = self._inputs()
        run_predictor_audits(
            beh, usv, names, fps, target_idx=1, predictor_idx=0,
            history_frames=5, event_keys=['positive_events'],
            settings=self._settings(collinearity=False),
            save_dir=str(tmp_path), pickle_basename='h.pkl',
            precomputed_onset_times={'s1': np.array([3.0, 1.0, 2.0]),
                                          's_empty': np.array([])},
        )
        bo = ts.call_args.kwargs['onset_times_per_session']
        np.testing.assert_array_equal(bo['s1'], np.array([1.0, 2.0, 3.0]))
        assert 's_empty' not in bo       # empty array dropped

    def test_skips_absent_and_empty_sessions_in_extraction(self, mocker, tmp_path):
        """Sessions absent from the USV dict, sessions whose target mouse is
        absent, missing/None/empty event arrays, and entries lacking
        start/stop are all skipped during the wrapper's extraction loops
        without contributing rows."""

        coll = mocker.patch.object(mu, 'audit_predictor_collinearity')
        mocker.patch.object(mu, 'audit_predictor_timescales')
        beh = {'s_ok': pls.DataFrame({'mF.speed': [0.0, 1.0]})}
        names = {
            's_no_session': ['mM', 'mF'],   # absent from usv_data_dict
            's_no_target': ['mM', 'mF'],    # target mouse absent
            's_none_arr': ['mM', 'mF'],     # event key present but None
            's_empty_arr': ['mM', 'mF'],    # event key present but empty
            's_no_startstop': ['mM', 'mF'], # has events but no start/stop
            's_ok': ['mM', 'mF'],
        }
        usv = {
            's_no_target': {'mM': {'positive_events': np.array([0.1])}},
            's_none_arr': {'mF': {'positive_events': None,
                                  'start': np.array([]), 'stop': np.array([])}},
            's_empty_arr': {'mF': {'positive_events': np.array([]),
                                   'start': np.array([0.1]), 'stop': np.array([])}},
            's_no_startstop': {'mF': {'positive_events': np.array([0.2])}},
            's_ok': {'mF': {'positive_events': np.array([0.5, 1.5]),
                            'start': np.array([0.5]), 'stop': np.array([0.6])}},
        }
        run_predictor_audits(
            beh, usv, names, {'s_ok': 30.0}, target_idx=1, predictor_idx=0,
            history_frames=5, event_keys=['positive_events'],
            settings=self._settings(timescale=False),
            save_dir=str(tmp_path), pickle_basename='sk.pkl',
        )
        evt = coll.call_args.kwargs['event_times_per_session']
        # Sessions with a non-empty `positive_events` array contribute to the
        # pooled event times; absent/no-target/None/empty-array sessions are
        # all skipped. (start/stop absence only affects the intervals dict.)
        assert set(evt) == {'s_no_startstop', 's_ok'}
        assert 's_no_session' not in evt
        assert 's_no_target' not in evt
        assert 's_none_arr' not in evt
        assert 's_empty_arr' not in evt

    def test_ibi_threshold_nan_when_mixture_model_index_out_of_range(self, mocker, tmp_path):
        """When ``mixture_model_component_index`` exceeds the per-sex component count the
        IBI threshold for that sex is NaN (the out-of-range ``else`` branch)."""

        mocker.patch.object(mu, 'audit_predictor_collinearity')
        ts = mocker.patch.object(mu, 'audit_predictor_timescales')
        beh, usv, names, fps = self._inputs()
        settings = self._settings(collinearity=False)
        settings['model_params']['mixture_model_component_index'] = 5   # out of range
        run_predictor_audits(
            beh, usv, names, fps, target_idx=1, predictor_idx=0,
            history_frames=5, event_keys=['positive_events'],
            settings=settings,
            save_dir=str(tmp_path), pickle_basename='nan.pkl',
        )
        thr = ts.call_args.kwargs['ibi_thresholds']
        assert np.isnan(thr['male'])
        assert np.isnan(thr['female'])

    @pytest.mark.parametrize(('bad_key', 'bad_val'), [
        ('timescale_shuffle_range', [60.0]),         # not 2-element
        ('timescale_shuffle_range', [60.0, 20.0]),   # min >= max
        ('timescale_signal_floor_seconds', -1.0),    # negative floor
        ('timescale_signal_min_run_seconds', 0.0),   # non-positive run
    ])
    def test_invalid_timescale_config_is_non_fatal(self, mocker, tmp_path, capsys,
                                                   bad_key, bad_val):
        """Each malformed timescale diagnostic config value raises a
        validation error that the timescale try/except converts into a
        non-fatal warning rather than calling the audit."""

        mocker.patch.object(mu, 'audit_predictor_collinearity')
        ts = mocker.patch.object(mu, 'audit_predictor_timescales')
        beh, usv, names, fps = self._inputs()
        settings = self._settings(collinearity=False)
        settings['diagnostics'][bad_key] = bad_val
        run_predictor_audits(
            beh, usv, names, fps, target_idx=1, predictor_idx=0,
            history_frames=5, event_keys=['positive_events'],
            settings=settings,
            save_dir=str(tmp_path), pickle_basename='bad.pkl',
        )
        ts.assert_not_called()
        assert 'timescale audit failed' in capsys.readouterr().out

    def test_bout_onset_extraction_skips_in_timescale_path(self, mocker, tmp_path):
        """The default (non-precomputed) bout-onset ``Y`` extraction loop
        skips sessions absent from the USV dict, sessions whose target mouse
        is absent, and sessions whose bout-onset key is empty, while keeping
        the populated session (so the universal-absence guard does not fire)."""

        mocker.patch.object(mu, 'audit_predictor_collinearity')
        ts = mocker.patch.object(mu, 'audit_predictor_timescales')
        beh = {'s_ok': pls.DataFrame({'mF.speed': [0.0, 1.0]})}
        names = {
            's_absent': ['mM', 'mF'],       # not in usv -> skip
            's_no_target': ['mM', 'mF'],    # target absent -> skip
            's_empty': ['mM', 'mF'],        # key present but empty -> skip
            's_ok': ['mM', 'mF'],
        }
        usv = {
            's_no_target': {'mM': {'positive_events': np.array([0.1]),
                                   'start': np.array([0.1]), 'stop': np.array([0.2])}},
            's_empty': {'mF': {'positive_events': np.array([]),
                               'start': np.array([0.1]), 'stop': np.array([0.2])}},
            's_ok': {'mF': {'positive_events': np.array([0.5, 1.5]),
                            'start': np.array([0.5]), 'stop': np.array([0.6])}},
        }
        run_predictor_audits(
            beh, usv, names, {'s_ok': 30.0}, target_idx=1, predictor_idx=0,
            history_frames=5, event_keys=['positive_events'],
            settings=self._settings(collinearity=False),
            save_dir=str(tmp_path), pickle_basename='ts.pkl',
        )
        ts.assert_called_once()
        bo = ts.call_args.kwargs['onset_times_per_session']
        assert set(bo) == {'s_ok'}


# Standardized modeling console-output formatters

def test_format_run_header_lists_files_and_config():
    """The header block names task/engine/feature, every input file, the output
    directory, and the split config, bracketed by `=` rules."""

    header = format_run_header(
        task='ONSET', engine='pygam', feature='self.speed',
        input_files={'input data': '/a/in.pkl', 'settings': '/a/s.json'},
        output_directory='/a/out', split_strategy='session', n_splits=10,
    )
    lines = header.splitlines()
    assert lines[0] == '=' * 70
    assert lines[-1] == '=' * 70
    assert 'task=ONSET' in header
    assert 'engine=pygam' in header
    assert 'feature=self.speed' in header
    assert 'split_strategy=session | n_splits=10' in header
    assert '  input data: /a/in.pkl' in lines
    assert '  settings: /a/s.json' in lines
    assert '  output_directory: /a/out' in lines


def test_format_split_line_one_and_two_strategies():
    """One combined split line renders each strategy's metrics in order, with a
    finite value at four decimals and non-finite as the literal `nan`."""

    line = format_split_line(3, 10, {
        'ACTUAL': {'AUC': 0.7201, 'LL': 0.65},
        'NULL': {'AUC': float('nan'), 'LL': 0.69},
    })
    assert line == ('Split 03/10 | ACTUAL AUC=0.7201 LL=0.6500 '
                    '| NULL AUC=nan LL=0.6900')


def test_format_run_summary_single_and_multi_metric():
    """The two-line summary reproduces the requested style: single-metric groups
    join with ', ', multi-metric groups join within a group with ' / ', and the
    save line is timestamped with the supplied timestamp."""

    ts = datetime(2026, 7, 18, 12, 0, 0)
    single = format_run_summary(
        label='self.speed [pygam]',
        metrics_by_strategy={'Mean Actual': {'AUC': 0.7468},
                             'Mean Null': {'AUC': 0.5174}},
        out_path='/a/f.pkl', timestamp=ts,
    )
    assert single.split('\n')[0] == (
        '--- Finished self.speed [pygam]. Mean Actual AUC: 0.7468, Mean Null AUC: 0.5174')
    assert single.split('\n')[1] == f'[{ts}] Success. Results saved to: /a/f.pkl'

    multi = format_run_summary(
        label='self.speed [jax]',
        metrics_by_strategy={'Mean Actual': {'bal-acc': 0.24, 'AUC': 0.62},
                             'Mean Null': {'bal-acc': 0.17, 'AUC': 0.50}},
        out_path='/a/f.pkl', timestamp=ts,
    )
    assert multi.split('\n')[0] == (
        '--- Finished self.speed [jax]. Mean Actual bal-acc: 0.2400 / AUC: 0.6200, '
        'Mean Null bal-acc: 0.1700 / AUC: 0.5000')


def test_format_selection_step_optional_fields():
    """The selection-step line renders only the non-empty fields, in order:
    stage, +feature, metrics, decision, detail."""

    assert format_selection_step('Step 03', feature='self.speed',
                                 metrics={'dcor_xy': 0.31}, decision='ACCEPT',
                                 detail='CI [+0.01, +0.05]') == (
        'Step 03 | +self.speed | dcor_xy=0.3100 | ACCEPT | CI [+0.01, +0.05]')
    assert format_selection_step('Baseline', metrics={'AUC': 0.5}) == 'Baseline | AUC=0.5000'
    assert format_selection_step('Step 02', decision='REJECT',
                                 detail='selection finished') == (
        'Step 02 | REJECT | selection finished')


def test_extract_univariate_headline_per_task():
    """The end-of-run extractor reduces each task's per-fold metrics to the
    mean-over-folds headline for every strategy present, NaN-guarded, and picks
    dcor_xy vs r2_spatial by geometry for the continuous task."""

    onset_res = {'actual': {'auc': [0.7, 0.8]}, 'null': {'auc': [0.5, 0.5]}}
    assert extract_univariate_headline('onset', onset_res) == {
        'Mean Actual': {'AUC': 0.75}, 'Mean Null': {'AUC': 0.5}}

    params_res = {'actual': {'explained_deviance': [0.1, 0.3]},
                  'null': {'explained_deviance': [0.0, 0.0]}}
    assert extract_univariate_headline('params', params_res) == {
        'Mean Actual': {'D^2': 0.2}, 'Mean Null': {'D^2': 0.0}}

    multi_res = {s: {'folds': {'metrics': {'score': [0.4], 'auc': [0.6]}}}
                 for s in ('actual', 'null', 'null_model_free')}
    multi_out = extract_univariate_headline('multinomial', multi_res)
    assert multi_out['Mean Actual'] == {'bal-acc': 0.4, 'AUC': 0.6}
    assert set(multi_out) == {'Mean Actual', 'Mean Null', 'Mean Null-MF'}

    # torus geometry: vm_logscore is finite -> it is the headline.
    torus_res = {s: {'folds': {'metrics': {'vm_logscore': [0.3], 'dcor_xy': [float('nan')], 'r2_spatial': [-0.1]}}}
                 for s in ('actual', 'null', 'null_model_free')}
    assert extract_univariate_headline('continuous', torus_res)['Mean Actual'] == {'vm_logscore': 0.3}

    # euclidean geometry: vm_logscore all-NaN, dcor_xy finite -> dcor_xy headline.
    euclid_dcor_res = {s: {'folds': {'metrics': {'vm_logscore': [float('nan')], 'dcor_xy': [0.25], 'r2_spatial': [0.1]}}}
                       for s in ('actual', 'null', 'null_model_free')}
    assert extract_univariate_headline('continuous', euclid_dcor_res)['Mean Actual'] == {'dcor_xy': 0.25}

    # both selection scores all-NaN -> r2_spatial is the headline.
    euclid_res = {s: {'folds': {'metrics': {'dcor_xy': [float('nan')], 'r2_spatial': [0.2]}}}
                  for s in ('actual', 'null', 'null_model_free')}
    assert extract_univariate_headline('continuous', euclid_res)['Mean Actual'] == {'r2_spatial': 0.2}

    # degenerate / minimal result dict -> empty summary, never raises.
    assert extract_univariate_headline('continuous', {'r2': 0.9}) == {}


class TestSeededSessionHoldout:
    """The deterministic session-grain held-out draw used to carve the reserved
    test block once at data-extraction time (`seeded_session_holdout`)."""

    def test_zero_proportion_reserves_nothing(self):
        """A proportion of 0 disables the holdout entirely (empty list), so the
        whole pipeline behaves exactly as it did before the mechanism existed."""
        assert seeded_session_holdout(['s3', 's1', 's2'], 0.0, 0) == []

    def test_count_is_floor_of_proportion(self):
        """The reserved count is floor(n_sessions * proportion)."""
        sessions = [f's{i}' for i in range(10)]
        assert len(seeded_session_holdout(sessions, 0.25, 0)) == 2  # floor(10 * 0.25)
        assert len(seeded_session_holdout(sessions, 0.1, 0)) == 1   # floor(10 * 0.1)

    def test_deterministic_and_order_invariant(self):
        """The same (sessions, proportion, seed) always yields the same sorted
        reserved set, regardless of the input order (the function sorts first)."""
        forward = seeded_session_holdout([f's{i}' for i in range(20)], 0.2, 7)
        shuffled = seeded_session_holdout([f's{i}' for i in reversed(range(20))], 0.2, 7)
        assert forward == shuffled
        assert forward == sorted(forward)

    def test_different_seed_can_differ(self):
        """Different seeds draw (in general) different reserved sets, confirming
        the seed actually drives the permutation."""
        a = seeded_session_holdout([f's{i}' for i in range(50)], 0.2, 0)
        b = seeded_session_holdout([f's{i}' for i in range(50)], 0.2, 1)
        assert a != b

    def test_reserved_is_subset_of_input(self):
        """Every reserved session is drawn from the input set (no fabrication)."""
        sessions = [f's{i}' for i in range(12)]
        reserved = seeded_session_holdout(sessions, 0.25, 3)
        assert set(reserved).issubset(set(sessions))

    def test_none_seed_coalesces_to_zero(self):
        """A None seed is treated as 0 rather than crashing the RNG."""
        assert seeded_session_holdout([f's{i}' for i in range(10)], 0.2, None) == \
            seeded_session_holdout([f's{i}' for i in range(10)], 0.2, 0)

    def test_out_of_range_proportion_raises(self):
        """Proportions outside [0, 1) are rejected."""
        with pytest.raises(ValueError, match=r'must be in \[0, 1\)'):
            seeded_session_holdout(['s0', 's1'], 1.0, 0)
        with pytest.raises(ValueError, match=r'must be in \[0, 1\)'):
            seeded_session_holdout(['s0', 's1'], -0.1, 0)

    def test_reserving_zero_raises(self):
        """A positive proportion too small to reserve even one whole session
        (floor rounds it to 0) leaves the holdout meaningless -> ValueError,
        rather than silently disabling it."""
        with pytest.raises(ValueError, match='must reserve at least one session'):
            seeded_session_holdout(['s0', 's1', 's2'], 0.1, 0)  # floor(3 * 0.1) == 0

    def test_positive_proportion_always_leaves_a_development_set(self):
        """For any proportion in (0, 1) that reserves >= 1 session, at least one
        development session always remains (floor(n * p) < n for p < 1), so a
        valid partition is produced and no exception is raised."""
        reserved = seeded_session_holdout(['s0', 's1'], 0.9, 0)  # floor(2 * 0.9) == 1
        assert len(reserved) == 1


class TestHeldOutSessionIdsFromMetadata:
    """Reading the reserved held-out list back from an `_input_metadata` block
    (`held_out_session_ids_from_metadata`), the single source of truth every
    downstream stage consults."""

    def test_present_field_returned_as_list_copy(self):
        """A present field is returned as an independent list copy."""
        md = {'held_out_session_ids': ['s1', 's2']}
        out = held_out_session_ids_from_metadata(md)
        assert out == ['s1', 's2']
        out.append('s3')
        assert md['held_out_session_ids'] == ['s1', 's2']  # not mutated

    def test_absent_field_returns_empty(self):
        """A schema-v1 pickle lacking the field yields [] (backward compatible)."""
        assert held_out_session_ids_from_metadata({'session_ids': ['s1']}) == []

    def test_none_value_returns_empty(self):
        """An explicit None value is treated as 'no holdout'."""
        assert held_out_session_ids_from_metadata({'held_out_session_ids': None}) == []

    def test_non_dict_returns_empty(self):
        """A non-dict argument (e.g. a caller passing {} for a missing block)
        never raises; it returns []."""
        assert held_out_session_ids_from_metadata(None) == []


class TestDevelopmentHeldoutMasks:
    """Splitting the per-event session-id array into development vs held-out
    boolean masks (`development_heldout_masks`)."""

    def test_empty_holdout_is_a_noop_identity(self):
        """With no reserved sessions the development mask is all-True and the
        held-out mask all-False, so `np.where(dev_mask)[0]` is the identity index
        map -- the property that makes every downstream dev-fold remap byte-
        identical to the pre-holdout behaviour."""
        groups = np.array(['s0', 's0', 's1', 's2', 's2'])
        dev_mask, held_mask = development_heldout_masks(groups, [])
        assert dev_mask.all()
        assert not held_mask.any()
        assert np.array_equal(np.where(dev_mask)[0], np.arange(len(groups)))

    def test_partition_by_session(self):
        """Rows whose session is reserved land in the held-out mask; all others in
        the development mask; the two masks are exact complements."""
        groups = np.array(['s0', 's1', 's1', 's2', 's3', 's3'])
        dev_mask, held_mask = development_heldout_masks(groups, ['s1', 's3'])
        assert np.array_equal(held_mask, np.array([False, True, True, False, True, True]))
        assert np.array_equal(dev_mask, ~held_mask)
        # No held-out session id survives in the development rows.
        assert set(groups[dev_mask].tolist()).isdisjoint({'s1', 's3'})

    def test_masks_are_complementary_and_full_length(self):
        """The masks always cover every row exactly once."""
        groups = np.array([f's{i % 4}' for i in range(20)])
        dev_mask, held_mask = development_heldout_masks(groups, ['s2'])
        assert len(dev_mask) == len(held_mask) == len(groups)
        assert np.array_equal(dev_mask, ~held_mask)
        assert int(dev_mask.sum() + held_mask.sum()) == len(groups)


@pytest.mark.parametrize(
    ("manifold_column_names", "expected"),
    [
        (["qlvm1", "qlvm2"], "qlvm"),
        (["qlvm_duration1", "qlvm_duration2"], "qlvm_duration"),
        (["qlvm_entropy1", "qlvm_entropy2"], "qlvm_entropy"),
        (["qlvm_bandwidth1", "qlvm_bandwidth2"], "qlvm_bandwidth"),
        (["qlvm_loudness1", "qlvm_loudness2"], "qlvm_loudness"),
    ],
)
def test_manifold_tag_segment_names_the_map(manifold_column_names, expected):
    """
    Description
    -----------
    The manifold analysis tag segment is the QLVM map the target columns come
    from (their prefix without the trailing digit), never the category column, so
    runs on different maps get different tags.

    Parameters
    ----------
    manifold_column_names (list[str])
        The two manifold columns of the run.
    expected (str)
        The map prefix the segment must be.

    Returns
    -------
    None
    """

    assert manifold_tag_segment(manifold_column_names) == expected


class TestSessionGroupedInnerCv:
    """The inner hyperparameter-search splitter holds out whole sessions."""

    def test_folds_hold_out_whole_sessions(self):
        """Every inner validation fold is a set of whole training sessions, capped at the
        session count; a single-session training set falls back to event folds."""

        groups = np.repeat(np.array(['a', 'b', 'c']), 4)
        cv = mu.session_grouped_inner_cv(groups, 5)
        assert cv.get_n_splits() == 3
        for train_rows, val_rows in cv.split(np.zeros(groups.size), groups=groups):
            assert not set(groups[train_rows]) & set(groups[val_rows])
        assert mu.session_grouped_inner_cv(np.repeat('a', 6), 5) == 5


class TestHeldOutSessionsNeverShapeTheScale:
    """Leakage property: the pooled z-score statistics come from the reference (development)
    sessions only, so arbitrary changes to a held-out session never move a development value."""

    @staticmethod
    def _sessions(held_out_values: np.ndarray) -> dict:
        """
        Description
        -----------
        Three development sessions of 50 random rows and one held-out session built
        from ``held_out_values``, each with ``self.speed`` and ``other.speed``.

        Parameters
        ----------
        held_out_values (np.ndarray)
            Values of the held-out session's ``self.speed`` (reversed for
            ``other.speed``).

        Returns
        -------
        sessions (dict)
            Session id -> ``polars.DataFrame``.
        """
        rng = np.random.default_rng(3)
        sessions = {
            f"dev_{i}": pls.DataFrame({"self.speed": rng.normal(5.0, 2.0, 50), "other.speed": rng.normal(1.0, 0.5, 50)})
            for i in range(3)
        }
        sessions["held"] = pls.DataFrame({"self.speed": held_out_values, "other.speed": held_out_values[::-1]})
        return sessions

    @pytest.mark.parametrize("seed", range(5))
    def test_development_values_ignore_the_held_out_session(self, seed):
        """Any held-out session content gives bit-identical development z-scores, and the
        statistics equal those of the development frames alone."""

        rng = np.random.default_rng(seed)
        reference = ["dev_0", "dev_1", "dev_2"]
        stats_a, stats_b = {}, {}
        a = zscore_features_across_sessions(self._sessions(rng.normal(0.0, 1.0, 40)), ["speed"], {"speed": (-1e9, 1e9)},
                                            stats_out=stats_a, reference_sessions=reference)
        b = zscore_features_across_sessions(self._sessions(rng.normal(1e3, 1e2, 40)), ["speed"], {"speed": (-1e9, 1e9)},
                                            stats_out=stats_b, reference_sessions=reference)
        for session in reference:
            assert a[session].equals(b[session])
        assert stats_a == stats_b
        pooled = np.concatenate([np.concatenate([self._sessions(np.zeros(1))[s][c].to_numpy()
                                                 for c in ("self.speed", "other.speed")]) for s in reference])
        assert stats_a["speed"]["mean"] == pytest.approx(float(pooled.mean()), rel=1e-5)


class TestCoreHelperKnownAnswers:
    """Known-answer checks added after mutation testing showed that a `break` in session pooling,
    sampling with replacement in class balancing, wrong calibration bins, a transposed sample count
    in class alignment and several boundary changes passed every existing test."""

    def test_pool_session_arrays_concatenates_every_listed_session(self):
        """Both classes are pooled over every listed session in list order along rows; empty or
        missing sessions are skipped; an empty class gives a (0, n_frames) block."""
        rng = np.random.default_rng(0)
        data = {f"s{i}": {"pos": rng.normal(size=(i + 1, 4)), "neg": rng.normal(size=(2 * i + 1, 4))} for i in range(3)}
        data["s_empty"] = {"pos": np.empty((0, 4)), "neg": np.empty((0, 4))}
        order = ["s2", "s_empty", "missing", "s0", "s1"]
        X_pos, X_neg = mu.pool_session_arrays(data, order, pos_key="pos", neg_key="neg", n_frames=4)
        np.testing.assert_array_equal(X_pos, np.concatenate([data[s]["pos"] for s in ("s2", "s0", "s1")]))
        np.testing.assert_array_equal(X_neg, np.concatenate([data[s]["neg"] for s in ("s2", "s0", "s1")]))
        X_pos, X_neg = mu.pool_session_arrays(data, ["s_empty"], pos_key="pos", neg_key="neg", n_frames=4)
        assert X_pos.shape == (0, 4)
        assert X_neg.shape == (0, 4)

    def test_balance_two_class_arrays_subsamples_without_replacement(self):
        """The majority class is cut to the minority size by drawing DISTINCT rows of itself; the
        minority class is returned whole; either class empty gives two empty blocks."""
        X_pos = np.arange(5 * 3, dtype=float).reshape(5, 3)
        X_neg = np.arange(100, 100 + 40 * 3, dtype=float).reshape(40, 3)
        rng = np.random.default_rng(0)
        bal_pos, bal_neg = mu.balance_two_class_arrays(X_pos, X_neg, rng)
        np.testing.assert_array_equal(bal_pos, X_pos)
        assert bal_neg.shape == (5, 3)
        rows = {tuple(row) for row in bal_neg}
        assert len(rows) == 5
        assert rows <= {tuple(row) for row in X_neg}
        empty_pos, empty_neg = mu.balance_two_class_arrays(np.empty((0, 3)), X_neg, rng)
        assert empty_pos.shape == (0, 3)
        assert empty_neg.shape == (0, 3)
        swapped_pos, swapped_neg = mu.balance_two_class_arrays(X_neg, X_pos, rng)
        np.testing.assert_array_equal(swapped_neg, X_pos)
        assert len({tuple(row) for row in swapped_pos}) == 5

    def test_expected_calibration_error_known_value(self):
        """ECE = sum over the n_bins equal-width confidence bins (lo, hi], the last closed at 1, of
        (n_bin / N) * |accuracy - mean confidence|."""
        proba = np.array([[0.95, 0.05], [0.85, 0.15], [0.55, 0.45], [0.6, 0.4], [1.0, 0.0], [0.3, 0.7]])
        y_pred = proba.argmax(axis=1)
        y_true = np.array([0, 1, 0, 1, 0, 1])
        conf = proba.max(axis=1)
        correct = (y_pred == y_true).astype(float)
        edges = np.linspace(0.0, 1.0, 11)
        expected = 0.0
        for k, (lo, hi) in enumerate(pairwise(edges)):
            in_bin = (conf > lo) & (conf <= hi + (1e-9 if k == 9 else 0.0))
            if in_bin.any():
                expected += in_bin.mean() * abs(correct[in_bin].mean() - conf[in_bin].mean())
        got = mu.expected_calibration_error(y_true, y_pred, proba, n_bins=10)
        assert got == pytest.approx(expected, rel=1e-12)
        assert got > 0.0
        assert mu.expected_calibration_error(np.array([0, 1]), np.array([0, 1]), np.array([[1.0, 0.0], [0.0, 1.0]]),
                                             n_bins=10) == pytest.approx(0.0)

    def test_align_probs_to_canonical_scatters_columns(self):
        """A model fit on a subset of the classes has its probability columns scattered into the
        canonical column order, zeros elsewhere, for a non-square (n_samples != n_classes) input."""
        probs = np.array([[0.2, 0.8], [0.6, 0.4], [0.5, 0.5]])
        out = mu.align_probs_to_canonical(probs, model_classes=[1, 3], canonical_classes=[0, 1, 2, 3])
        np.testing.assert_array_equal(out, [[0.0, 0.2, 0.0, 0.8], [0.0, 0.6, 0.0, 0.4], [0.0, 0.5, 0.0, 0.5]])
        with pytest.raises(ValueError, match="strictly ascending"):
            mu.align_probs_to_canonical(probs, model_classes=[1, 3], canonical_classes=[3, 1])
        with pytest.raises(ValueError, match="present in `canonical_classes`"):
            mu.align_probs_to_canonical(probs, model_classes=[1, 4], canonical_classes=[0, 1, 2, 3])

    def test_boundaries_of_the_selection_helpers(self):
        """Two folds give a finite paired SE (np.std ddof=1 / sqrt(2)); a held-out proportion of 1
        is refused; two sessions already split by session."""
        improvement, se = mu.paired_one_se_improvement([0.3, 0.5], [0.1, 0.2], higher_is_better=True)
        assert improvement == pytest.approx(0.25)
        assert se == pytest.approx(np.std([0.2, 0.3], ddof=1) / np.sqrt(2))
        with pytest.raises(ValueError, match=r"\[0, 1\)"):
            mu.seeded_session_holdout(session_ids=['a', 'b'], held_out_test_proportion=1.0, random_seed=0)
        cv = mu.session_grouped_inner_cv(np.array(['a', 'a', 'b', 'b']), 5)
        assert cv.get_n_splits() == 2


class TestFeatureSelectionKnownAnswers:
    """Known-answer checks for the predictor-column rules, added after mutation testing showed the
    derivative, asymmetric-pose and vocal-trace rules could be broken without any test noticing."""

    _COLUMNS: ClassVar[list[str]] = [
        'M.speed', 'M.speed_1st_der', 'M.neck_elevation', 'M.neck_elevation_1st_der',
        'M.neck_elevation_2nd_der', 'F.neck_elevation',
        'M-F.allo_yaw-nose', 'M-F.nose-allo_yaw', 'M-F.nose-nose', 'M-F.nose-nose_1st_der',
        'M-F.orofacial-sei', 'F-M.orofacial-sei',
    ]

    @staticmethod
    def _kin(symmetric=False, first=True, second=False):
        """
        Description
        -----------
        A kinematic-settings block over the class's column list, with the dyadic-pose
        symmetry and derivative switches exposed.

        Parameters
        ----------
        symmetric (bool)
            ``dyadic_pose_symmetric``.
        first (bool)
            ``include_1st_derivatives``.
        second (bool)
            ``include_2nd_derivatives``.

        Returns
        -------
        kin_settings (dict)
            The settings block passed to ``select_kinematic_columns``.
        """
        return {'egocentric': ['speed', 'neck_elevation'],
                'dyadic_pose': ['allo_yaw-nose', 'nose-allo_yaw', 'nose-nose'],
                'dyadic_engagement': ['orofacial-sei'], 'dyadic_pose_symmetric': symmetric,
                'include_1st_derivatives': first, 'include_2nd_derivatives': second}

    def test_select_kinematic_columns_rules(self):
        """Speed never gets derivatives; other features get the derivatives that are switched on
        and present; asymmetric dyadic-pose features keep only the side the predictor's slot
        allows; engagement keeps only the target-predictor dyad."""
        base = ['M-F.nose-nose', 'M-F.nose-nose_1st_der', 'M-F.orofacial-sei', 'M.neck_elevation',
                'M.neck_elevation_1st_der', 'M.speed', 'F.neck_elevation']
        got = mu.select_kinematic_columns(self._COLUMNS, 'M', 'F', self._kin(), predictor_idx=0)
        assert got == sorted([*base, 'M-F.nose-allo_yaw'])
        got = mu.select_kinematic_columns(self._COLUMNS, 'M', 'F', self._kin(), predictor_idx=1)
        assert got == sorted([*base, 'M-F.allo_yaw-nose'])
        got = mu.select_kinematic_columns(self._COLUMNS, 'M', 'F', self._kin(symmetric=True, second=True), predictor_idx=0)
        assert got == sorted([*base, 'M-F.allo_yaw-nose', 'M-F.nose-allo_yaw', 'M.neck_elevation_2nd_der'])
        got = mu.select_kinematic_columns(self._COLUMNS, 'M', 'F', self._kin(first=False), predictor_idx=0)
        assert got == sorted(['M-F.nose-nose', 'M-F.orofacial-sei', 'M.neck_elevation', 'M.speed',
                              'F.neck_elevation', 'M-F.nose-allo_yaw'])

    def test_build_vocal_signal_columns_emits_every_trace_with_its_values(self):
        """Every partner trace and every non-pooled target trace becomes a column carrying its own
        values; the target's pooled traces and excluded keys are dropped; partner-only drops the
        target entirely; no predictor type emits nothing."""
        signals = {
            'M': {'continuous_vocal_signals': {'usv_rate': np.array([1.0, 2.0]), 'usv_cat_1': np.array([3.0, 4.0]),
                                               'usv_cat_2': np.array([5.0, 6.0])}},
            'F': {'continuous_vocal_signals': {'usv_rate': np.array([7.0, 8.0]), 'squeak_rate': np.array([9.0, 10.0])}},
        }
        usv_data = {'s': signals}
        settings = {'vocal_predictor_type': 'all_rate', 'vocal_predictor_partner_only': False}
        cols, names = mu.build_vocal_signal_columns(usv_data, 's', 'M', 'F', settings, usv_self_exclude=('usv_cat_2',))
        assert names == ['M.usv_cat_1', 'F.usv_rate', 'F.squeak_rate']
        assert [col.to_list() for col in cols] == [[3.0, 4.0], [7.0, 8.0], [9.0, 10.0]]
        _, names = mu.build_vocal_signal_columns(usv_data, 's', 'M', 'F',
                                                 {**settings, 'vocal_predictor_partner_only': True})
        assert names == ['F.usv_rate', 'F.squeak_rate']
        assert mu.build_vocal_signal_columns(usv_data, 's', 'M', 'F', {**settings, 'vocal_predictor_type': None}) == ([], [])

    @pytest.mark.filterwarnings("ignore:An input array is constant:scipy.stats.ConstantInputWarning")
    def test_two_class_labels_and_constant_correlations(self):
        """Labels count ROWS of each class (non-square blocks); a constant truth (not only a constant
        prediction) makes Pearson and Spearman NaN."""
        X, y = mu.concat_two_class_with_labels(np.zeros((3, 5)), np.ones((2, 5)))
        assert X.shape == (5, 5)
        np.testing.assert_array_equal(y, [1, 1, 1, 0, 0])
        assert np.isnan(mu.pearson_r_safe(np.ones(5), np.arange(5.0)))
        assert np.isnan(mu.spearman_r_safe(np.ones(5), np.arange(5.0)))


class TestSecondMutationRoundKnownAnswers:
    """Known-answer checks for the changes the second mutation run (with the first round's tests
    in place) still let through: sampling the majority POSITIVES with replacement, a wrong ECE bin
    count or bin-edge inclusion, the acceleration / allo_pitch / TTI predictor rules, a `break`
    for a `continue` in the engagement loop, engagement derivatives, the dyad rename, the target
    side of the vocal existence map, single-row sessions in pooling, the one-session test
    proportion, the paired gate's default direction, the z-score wrapper's pass-through and the
    category / null headlines."""

    def test_balance_never_redraws_a_row_and_keeps_a_single_positive(self):
        """Whichever class is the majority, its rows are drawn without replacement (checked over
        many seeds with a 6-to-5 split, where drawing with replacement repeats a row 91% of the
        time); one positive against many negatives still gives one row per class."""
        small = np.arange(5 * 2, dtype=float).reshape(5, 2)
        large = np.arange(100, 100 + 6 * 2, dtype=float).reshape(6, 2)
        for seed in range(40):
            rng = np.random.default_rng(seed)
            bal_pos, bal_neg = mu.balance_two_class_arrays(large, small, rng)
            assert len({tuple(row) for row in bal_pos}) == 5
            np.testing.assert_array_equal(bal_neg, small)
            bal_pos, bal_neg = mu.balance_two_class_arrays(small, large, rng)
            assert len({tuple(row) for row in bal_neg}) == 5
        one_pos, few_neg = mu.balance_two_class_arrays(np.ones((1, 2)), large, rng)
        assert one_pos.shape == (1, 2)
        assert few_neg.shape == (1, 2)
        many_pos, one_neg = mu.balance_two_class_arrays(large, np.ones((1, 2)), rng)
        assert many_pos.shape == (1, 2)
        assert one_neg.shape == (1, 2)

    def test_expected_calibration_error_uses_n_bins_with_half_open_edges(self):
        """Confidences 0.62 / 0.68 share the bin (0.6, 0.7] with 10 bins (so a correct and a wrong
        prediction average out), but would fall in different bins with 8 or 11; a confidence of
        exactly 0.6 belongs to (0.5, 0.6] only. A single prediction is a valid ECE of
        |correct - confidence|."""
        proba = np.array([[0.62, 0.38], [0.68, 0.32], [0.6, 0.4], [0.9, 0.1]])
        y_pred = np.zeros(4, dtype=int)
        y_true = np.array([0, 1, 1, 0])
        # bins: (0.5,0.6] holds 0.6 (wrong) -> 0.25*|0-0.6|; (0.6,0.7] holds 0.62 (right) and 0.68
        # (wrong) -> 0.5*|0.5-0.65|; (0.8,0.9] holds 0.9 (right) -> 0.25*|1-0.9|
        expected = 0.25 * 0.6 + 0.5 * 0.15 + 0.25 * 0.1
        assert mu.expected_calibration_error(y_true, y_pred, proba, n_bins=10) == pytest.approx(expected)
        single = mu.expected_calibration_error(np.array([1]), np.array([0]), np.array([[0.7, 0.3]]), n_bins=10)
        assert single == pytest.approx(0.7)

    def test_select_kinematic_columns_acceleration_pitch_tti_and_engagement(self):
        """Acceleration never gets derivatives; allo_pitch is asymmetric like allo_yaw, and a TTI
        partner side follows the opposite slot rule; a non-target dyad listed BEFORE the target dyad
        does not stop the engagement search; engagement derivatives are kept when switched on."""
        columns = ['M.acceleration', 'M.acceleration_1st_der',
                   'M-F.allo_pitch-nose', 'M-F.nose-allo_pitch', 'M-F.nose-TTI', 'M-F.TTI-nose',
                   'F-M.orofacial-sei', 'M-F.orofacial-sei', 'M-F.orofacial-sei_1st_der']
        kin = {'egocentric': ['acceleration'],
               'dyadic_pose': ['allo_pitch-nose', 'nose-allo_pitch', 'nose-TTI', 'TTI-nose'],
               'dyadic_engagement': ['orofacial-sei'], 'dyadic_pose_symmetric': False,
               'include_1st_derivatives': True, 'include_2nd_derivatives': False}
        shared = ['M-F.orofacial-sei', 'M-F.orofacial-sei_1st_der', 'M.acceleration']
        got = mu.select_kinematic_columns(columns, 'M', 'F', kin, predictor_idx=0)
        assert got == sorted([*shared, 'M-F.nose-allo_pitch', 'M-F.TTI-nose'])
        got = mu.select_kinematic_columns(columns, 'M', 'F', kin, predictor_idx=1)
        assert got == sorted([*shared, 'M-F.allo_pitch-nose', 'M-F.nose-TTI'])

    def test_harmonize_renames_dyads_keeping_values_and_fills_the_target_side(self):
        """A dyad column is renamed to its bare suffix with its values (not replaced by a zero
        column next to the old name), and a vocal suffix present only for the TARGET somewhere is
        zero-filled for the target where missing, never for the predictor."""
        d = {
            's1': pls.DataFrame({'mF.speed': [1.0, 2.0], 'mM-mF.nose-nose': [0.1, 0.2], 'mF.usv_rate': [5.0, 6.0]}),
            's2': pls.DataFrame({'mF.speed': [3.0, 4.0], 'mM.speed': [7.0, 8.0]}),
        }
        names = {'s1': ['mM', 'mF'], 's2': ['mM', 'mF']}
        out, suffixes = harmonize_session_columns(d, names, target_idx=1, predictor_idx=0)
        assert out['s1']['nose-nose'].to_list() == [0.1, 0.2]
        assert 'mM-mF.nose-nose' not in out['s1'].columns
        assert out['s2']['mF.usv_rate'].to_list() == [0.0, 0.0]
        assert 'mM.usv_rate' not in out['s1'].columns
        assert 'mM.usv_rate' not in out['s2'].columns
        assert suffixes == ['nose-nose', 'speed', 'usv_rate']

    def test_pooling_keeps_single_row_sessions_and_one_session_needs_the_whole_set(self):
        """A session contributing a single row of either class is pooled (not dropped as empty);
        with one session the test proportion is 1 (the one session), with two it is at least 1/2."""
        data = {'a': {'pos': np.ones((1, 3)), 'neg': np.zeros((1, 3))}, 'b': {'pos': np.full((2, 3), 2.0), 'neg': None}}
        X_pos, X_neg = mu.pool_session_arrays(data, ['a', 'b'], pos_key='pos', neg_key='neg', n_frames=3)
        np.testing.assert_array_equal(X_pos, np.array([[1.0] * 3, [2.0] * 3, [2.0] * 3]))
        np.testing.assert_array_equal(X_neg, np.zeros((1, 3)))
        assert mu.bounded_test_proportion(0.2, n_sessions=1) == 1.0
        assert mu.bounded_test_proportion(0.2, n_sessions=2) == 0.5
        assert mu.bounded_test_proportion(0.2, n_sessions=0) == 0.2

    def test_paired_gate_defaults_to_higher_is_better(self):
        """Without `higher_is_better` the gate treats larger scores as better: a candidate scoring
        0.1 above the incumbent on every fold is a +0.1 improvement."""
        improvement, _ = mu.paired_one_se_improvement([0.6, 0.7, 0.8], [0.5, 0.6, 0.7])
        assert improvement == pytest.approx(0.1)

    def test_zscore_wrapper_passes_every_argument_through(self, mocker):
        """The cross-session z-score wrapper hands every argument, the absolute-value lists
        included, to zscore_different_sessions_together unchanged."""
        inner = mocker.patch.object(mu, 'zscore_different_sessions_together', return_value={'ok': 1})
        out = mu.zscore_features_across_sessions({'s': 1}, ['speed'], {'speed': (0, 1)}, abs_features=['a'],
                                                 smooth_abs_features={'b': 2}, stats_out={}, reference_sessions=['s'])
        assert out == {'ok': 1}
        assert inner.call_args.kwargs == {
            'data_dict': {'s': 1}, 'feature_lst': ['speed'], 'feature_bounds': {'speed': (0, 1)},
            'abs_features': ['a'], 'smooth_abs_features': {'b': 2}, 'stats_out': {}, 'reference_sessions': ['s']}

    def test_headline_for_the_category_task_and_the_continuous_nulls(self):
        """The category task reports AUC like the onset task, and the continuous task reports the
        same headline metric for its two null strategies."""
        res = {'actual': {'auc': [0.8, 0.6]}, 'null': {'auc': [0.5, 0.4]}}
        assert extract_univariate_headline('category', res) == {'Mean Actual': {'AUC': 0.7}, 'Mean Null': {'AUC': 0.45}}
        cont = {'actual': {'folds': {'metrics': {'vm_logscore': [0.3]}}},
                'null': {'folds': {'metrics': {'vm_logscore': [0.1]}}},
                'null_model_free': {'folds': {'metrics': {'vm_logscore': [-0.2]}}}}
        out = extract_univariate_headline('continuous', cont)
        assert out['Mean Null'] == {'vm_logscore': 0.1}
        assert out['Mean Null-MF'] == {'vm_logscore': -0.2}


class TestSessionLabelsThroughPooledTasks:
    """The pooled-row tasks (onset, category, bout parameters) carry each row's session so
    their nulls shuffle labels within sessions and their inner CVs hold out whole sessions."""

    @staticmethod
    def _data():
        """
        Description
        -----------
        Three sessions whose rows carry the session index in every column, with
        different class balances (s0: 4 pos / 1 neg, s1: 1 / 4, s2: 2 / 2).

        Returns
        -------
        data (dict)
            Session id -> ``{'pos': (n_pos, 3), 'neg': (n_neg, 3)}`` arrays.
        """
        counts = {"s0": (4, 1), "s1": (1, 4), "s2": (2, 2)}
        return {sess: {"pos": np.full((n_pos, 3), float(i)), "neg": np.full((n_neg, 3), float(i))}
                for i, (sess, (n_pos, n_neg)) in enumerate(counts.items())}

    def test_pooling_and_balancing_keep_every_rows_session(self):
        """Pooled session labels match the rows they label, and stay matched through the
        class balancing (whose draw comes from the given generator)."""
        data = self._data()
        X_pos, X_neg, g_pos, g_neg = mu.pool_session_arrays(data, ["s0", "s1", "s2"], pos_key="pos",
                                                            neg_key="neg", n_frames=3, return_groups=True)
        assert list(g_pos) == ["s0"] * 4 + ["s1"] + ["s2"] * 2
        assert list(g_neg) == ["s0"] + ["s1"] * 4 + ["s2"] * 2
        bp, bn, gp, gn = mu.balance_two_class_arrays(X_pos, X_neg, np.random.default_rng(3),
                                                     groups_pos=g_pos, groups_neg=g_neg)
        assert bp.shape[0] == bn.shape[0] == 7
        index = {"s0": 0.0, "s1": 1.0, "s2": 2.0}
        assert [index[g] for g in gp] == list(bp[:, 0])
        assert [index[g] for g in gn] == list(bn[:, 0])

    def test_within_group_permutation_keeps_each_sessions_labels(self):
        """Shuffling within sessions moves labels only among a session's own rows: every
        session keeps exactly its label counts, and over many draws every row of a session
        takes every value present in it."""
        groups = np.array(["a"] * 4 + ["b"] * 6)
        y = np.array([1, 1, 1, 0, 0, 0, 0, 0, 1, 0])
        for seed in range(20):
            perm = mu.within_group_permutation(groups, np.random.default_rng(seed))
            assert sorted(perm[:4]) == [0, 1, 2, 3]
            assert sorted(perm[4:]) == list(range(4, 10))
            assert y[perm][:4].sum() == 3
            assert y[perm][4:].sum() == 1

    def test_inner_cv_splits_hold_out_whole_sessions_with_both_classes(self):
        """The inner folds never put one session on both sides and keep both classes on each
        side; one session, or folds that would drop a class, fall back to event folds."""
        groups = np.repeat(np.array(["a", "b", "c", "d", "e"]), 6)
        y = np.tile(np.array([1, 1, 1, 0, 0, 0]), 5)
        splits = mu.classification_inner_cv_splits(y, groups, 5, 0)
        assert isinstance(splits, list)
        assert len(splits) == 5
        for train, validation in splits:
            assert not set(groups[train]) & set(groups[validation])
            assert set(y[train]) == {0, 1}
            assert set(y[validation]) == {0, 1}
        assert mu.classification_inner_cv_splits(y, np.full(30, "a"), 5, 0) == 5
        single_class_sessions = np.repeat(np.array(["a", "b"]), 15)
        assert mu.classification_inner_cv_splits(np.repeat([1, 0], 15), single_class_sessions, 5, 0) == 5


def _assert_same_kwargs(actual: dict, expected: dict) -> None:
    """
    Description
    -----------
    Asserts that the keyword arguments a spy recorded are exactly the expected
    ones: the same key set (so a dropped keyword fails) and equal values, where
    NumPy arrays compare element-wise, dicts and tuples compare recursively,
    floats compare with NaN equal to NaN, and every other value compares with
    ``==``.

    Parameters
    ----------
    actual (dict)
        The recorded keyword arguments (``spy.call_args.kwargs``).
    expected (dict)
        The keyword arguments the wrapper must have passed.

    Returns
    -------
    None
    """

    assert set(actual) == set(expected), (sorted(actual), sorted(expected))
    for key, expected_value in expected.items():
        _assert_same_value(actual[key], expected_value, key)


def _assert_same_value(actual, expected, where: str) -> None:
    """
    Description
    -----------
    Recursive value comparison behind ``_assert_same_kwargs``: arrays must match
    in shape, dtype kind and values; dicts must have the same keys and equal
    values; tuples must have the same length and equal items; floats must be
    equal, or both NaN; anything else must be ``==`` and of the same type. The
    very object expected (e.g. the caller's own dict of DataFrames) always passes.

    Parameters
    ----------
    actual (object)
        The recorded value.
    expected (object)
        The expected value.
    where (str)
        The keyword (and nested key path) being compared, for the failure message.

    Returns
    -------
    None
    """

    if actual is expected:
        return
    if isinstance(expected, np.ndarray):
        assert isinstance(actual, np.ndarray), where
        assert actual.dtype.kind == expected.dtype.kind, where
        np.testing.assert_array_equal(actual, expected, err_msg=where)
    elif isinstance(expected, dict):
        assert isinstance(actual, dict), where
        assert set(actual) == set(expected), where
        for sub_key in expected:
            _assert_same_value(actual[sub_key], expected[sub_key], f"{where}[{sub_key!r}]")
    elif isinstance(expected, tuple):
        assert isinstance(actual, tuple), where
        assert len(actual) == len(expected), where
        for i, (a_item, e_item) in enumerate(zip(actual, expected, strict=True)):
            _assert_same_value(a_item, e_item, f"{where}[{i}]")
    elif isinstance(expected, float):
        assert isinstance(actual, float), where
        assert (np.isnan(actual) and np.isnan(expected)) or actual == expected, where
    else:
        assert type(actual) is type(expected), where
        assert actual == expected, where


class TestThirdMutationRoundKnownAnswers:
    """Known-answer checks for the changes the third mutation run still let through: the
    second manifold column naming the tag, 3-part column names in the kinematic selector and
    the dyad rename, derivatives of speed / acceleration in the dyadic buckets, the vocal
    column loop's skips, the float32 zero-fill, None / single-row / 1-D empty arrays in
    pooling, group labels through balancing, cross-group shuffling, the inner-CV fold count
    and seed, the ECE bin count, bin-edge inclusion and top-bin tolerance, and the NaN paths
    of the safe metrics."""

    def test_manifold_tag_segment_reads_only_the_first_column(self):
        """The segment comes from the FIRST manifold column alone: a one-column list works
        and a second column with a different prefix never leaks into the tag."""

        assert manifold_tag_segment(['qlvm_duration1']) == 'qlvm_duration'
        assert manifold_tag_segment(['qlvm1', 'umap2']) == 'qlvm'

    def test_select_kinematic_columns_matches_the_last_dot_part(self):
        """Dyadic-pose and dyadic-engagement columns are matched on the LAST dot-separated
        part of the column name, so a 3-part name ``dyad.extra.feature`` is kept by its
        feature, while a column whose MIDDLE part equals the feature is not."""

        cols = ['mM-mF.extra.nose-nose', 'mM-mF.nose-nose.zzz',
                'mF-mM.extra.orofacial-sei', 'mF-mM.orofacial-sei.zzz']
        kin = {'egocentric': [], 'dyadic_pose': ['nose-nose'],
               'dyadic_engagement': ['orofacial-sei'], 'dyadic_pose_symmetric': True,
               'include_1st_derivatives': False, 'include_2nd_derivatives': False}
        out = select_kinematic_columns(cols, target_name='mF', predictor_name='mM',
                                       kin_settings=kin, predictor_idx=0)
        assert out == ['mF-mM.extra.orofacial-sei', 'mM-mF.extra.nose-nose']

    def test_dyadic_speed_and_acceleration_never_get_derivatives(self):
        """The speed / acceleration derivative exclusion applies in the dyadic buckets too:
        a dyadic-pose ``speed`` and a dyadic-engagement ``acceleration`` keep their base
        column but not their present ``_1st_der`` / ``_2nd_der`` columns, while a dyadic
        ``nose-nose`` keeps both of its derivatives."""

        cols = ['mM-mF.speed', 'mM-mF.speed_1st_der', 'mM-mF.speed_2nd_der',
                'mF-mM.acceleration', 'mF-mM.acceleration_1st_der', 'mF-mM.acceleration_2nd_der',
                'mM-mF.nose-nose', 'mM-mF.nose-nose_1st_der', 'mM-mF.nose-nose_2nd_der']
        kin = {'egocentric': [], 'dyadic_pose': ['speed', 'nose-nose'],
               'dyadic_engagement': ['acceleration'], 'dyadic_pose_symmetric': True,
               'include_1st_derivatives': True, 'include_2nd_derivatives': True}
        out = select_kinematic_columns(cols, target_name='mF', predictor_name='mM',
                                       kin_settings=kin, predictor_idx=0)
        assert out == ['mF-mM.acceleration', 'mM-mF.nose-nose', 'mM-mF.nose-nose_1st_der',
                       'mM-mF.nose-nose_2nd_der', 'mM-mF.speed']

    def test_vocal_columns_skip_and_keep_going(self):
        """A target mouse absent from the session does not stop the partner's columns, and a
        pooled target trace listed BEFORE a per-category trace does not stop the target's
        per-category trace from being emitted (skips, not stops)."""

        settings = {'vocal_predictor_type': 'continuous', 'vocal_predictor_partner_only': False}
        absent_target = {'sess': {'mPartner': {'continuous_vocal_signals': {
            'usv_rate': np.zeros(2)}}}}
        _cols, names = build_vocal_signal_columns(absent_target, 'sess', 'mTarget', 'mPartner',
                                                  voc_settings=settings)
        assert names == ['mPartner.usv_rate']
        pooled_first = {'sess': {
            'mTarget': {'continuous_vocal_signals': {'usv_rate': np.zeros(2), 'usv_cat_3': np.ones(2)}},
            'mPartner': {'continuous_vocal_signals': {'usv_cat_3': np.ones(2)}}}}
        _cols, names = build_vocal_signal_columns(pooled_first, 'sess', 'mTarget', 'mPartner',
                                                  voc_settings=settings)
        assert names == ['mTarget.usv_cat_3', 'mPartner.usv_cat_3']

    def test_harmonize_renames_by_the_dyad_prefix_and_last_part(self):
        """A dyad column with three dot-parts is renamed to its LAST part, and a per-mouse
        column whose suffix (not prefix) contains a dash is NOT renamed, since the rename is
        keyed on a dash in the part before the first dot."""

        d = {'s1': pls.DataFrame({'mM-mF.extra.nose-nose': [1.0, 2.0],
                                  'mF.orofacial-sei': [3.0, 4.0]})}
        out, _suffixes = harmonize_session_columns(d, {'s1': ['mM', 'mF']}, target_idx=1, predictor_idx=0)
        assert out['s1']['nose-nose'].to_list() == [1.0, 2.0]
        assert 'extra' not in out['s1'].columns
        assert out['s1']['mF.orofacial-sei'].to_list() == [3.0, 4.0]

    def test_harmonize_zero_fills_are_float32(self):
        """Every zero-filled column is float32 whichever rule filled it: a missing dyadic
        column, a missing non-vocal per-mouse column, and a missing vocal column whose role
        was populated elsewhere (float64 inputs, so the dtype comes from the fill)."""

        d = {'s1': pls.DataFrame({'nose-nose': [0.5, 0.6], 'mF.speed': [1.0, 2.0],
                                  'mF.usv_rate': [0.1, 0.2]}),
             's2': pls.DataFrame({'mF2.dummy': [9.0, 9.0]})}
        names = {'s1': ['mM', 'mF'], 's2': ['mM2', 'mF2']}
        out, _suffixes = harmonize_session_columns(d, names, target_idx=1, predictor_idx=0)
        s2 = out['s2']
        for filled in ('nose-nose', 'mF2.speed', 'mM2.speed', 'mF2.usv_rate'):
            assert s2[filled].dtype == pls.Float32, filled
            assert s2[filled].to_list() == [0.0, 0.0]
        assert 'mM2.usv_rate' not in s2.columns
        assert out['s1']['mM.speed'].dtype == pls.Float32

    def test_pool_skips_none_and_keeps_single_values(self):
        """A ``None`` class array is skipped while the other class of that session is kept,
        a 1x1 array (a single row of a single frame) is kept, and the session labels are
        object arrays holding each row's session."""

        data = {'s1': {'pos': None, 'neg': np.ones((2, 1))},
                's2': {'pos': np.array([[7.0]]), 'neg': None}}
        X_pos, X_neg, g_pos, g_neg = pool_session_arrays(data, ['s1', 's2'], pos_key='pos',
                                                         neg_key='neg', n_frames=1, return_groups=True)
        np.testing.assert_array_equal(X_pos, np.array([[7.0]]))
        np.testing.assert_array_equal(X_neg, np.ones((2, 1)))
        assert g_pos.dtype == object
        assert list(g_pos) == ['s2']
        assert g_neg.dtype == object
        assert list(g_neg) == ['s1', 's1']
        X_pos, X_neg = pool_session_arrays({'s1': {'pos': np.array([[7.0]]), 'neg': np.array([[3.0]])}},
                                           ['s1'], pos_key='pos', neg_key='neg', n_frames=1)
        np.testing.assert_array_equal(X_pos, np.array([[7.0]]))
        np.testing.assert_array_equal(X_neg, np.array([[3.0]]))

    def test_pool_drops_size_zero_arrays_of_any_shape(self):
        """A size-0 array is dropped whatever its shape (here 1-D ``np.array([])``, which
        cannot be stacked with 2-D rows), and a class that is empty everywhere returns a
        ``(0, n_frames)`` placeholder and a size-0 object label array."""

        data = {'s1': {'pos': np.array([]), 'neg': np.array([])},
                's2': {'pos': np.ones((2, 4)), 'neg': np.ones((1, 4))}}
        X_pos, X_neg, g_pos, g_neg = pool_session_arrays(data, ['s1', 's2'], pos_key='pos',
                                                         neg_key='neg', n_frames=4, return_groups=True)
        assert X_pos.shape == (2, 4)
        assert X_neg.shape == (1, 4)
        assert list(g_pos) == ['s2', 's2']
        assert list(g_neg) == ['s2']
        X_pos, X_neg, g_pos, g_neg = pool_session_arrays({'s1': {'pos': np.array([]), 'neg': None}}, ['s1'],
                                                         pos_key='pos', neg_key='neg', n_frames=4,
                                                         return_groups=True)
        assert X_pos.shape == (0, 4)
        assert X_neg.shape == (0, 4)
        assert g_pos.shape == (0,)
        assert g_pos.dtype == object
        assert g_neg.shape == (0,)
        assert g_neg.dtype == object

    def test_balance_one_empty_class_returns_empty_labels_without_drawing(self):
        """When one class is empty both classes come back empty with size-0 OBJECT label
        arrays (whatever dtype the labels came in) and the generator is left untouched."""

        rng = np.random.default_rng(0)
        out = balance_two_class_arrays(np.empty((0, 2)), np.ones((3, 2)), rng,
                                       groups_pos=np.array([], dtype='<U2'),
                                       groups_neg=np.array(['s1', 's1', 's2']))
        assert len(out) == 4
        bp, bn, gp, gn = out
        assert bp.shape == (0, 2)
        assert bn.shape == (0, 2)
        assert gp.shape == (0,)
        assert gp.dtype == object
        assert gn.shape == (0,)
        assert gn.dtype == object
        assert rng.random() == np.random.default_rng(0).random()

    def test_balance_labels_follow_subsampled_rows_and_need_both_label_arrays(self):
        """Labels are subsampled with their rows whichever class is the majority, and labels
        given for one class only are ignored (the 2-tuple comes back)."""

        X_big = np.arange(6, dtype=float).reshape(6, 1)
        g_big = np.array([f"s{int(v)}" for v in X_big[:, 0]], dtype=object)
        X_small = np.array([[10.0], [11.0]])
        g_small = np.array(['s10', 's11'], dtype=object)
        bp, bn, gp, gn = balance_two_class_arrays(X_big, X_small, np.random.default_rng(1),
                                                  groups_pos=g_big, groups_neg=g_small)
        assert list(gp) == [f"s{int(v)}" for v in bp[:, 0]]
        assert list(gn) == ['s10', 's11']
        bp, bn, gp, gn = balance_two_class_arrays(X_small, X_big, np.random.default_rng(1),
                                                  groups_pos=g_small, groups_neg=g_big)
        assert list(gn) == [f"s{int(v)}" for v in bn[:, 0]]
        assert gn.shape == (2,)
        out = balance_two_class_arrays(X_big, X_small, np.random.default_rng(1), groups_pos=g_big, groups_neg=None)
        assert len(out) == 2
        assert out[0].shape == (2, 1)

    def test_within_group_permutation_moves_rows_only_inside_their_group(self):
        """With three groups the permutation actually shuffles every group (it is not the
        identity) and never sends a row to another group's position."""

        groups = np.repeat(np.array(['a', 'b', 'c']), 10)
        perm = mu.within_group_permutation(groups, np.random.default_rng(0))
        np.testing.assert_array_equal(np.sort(perm), np.arange(30))
        np.testing.assert_array_equal(groups[perm], groups)
        for g in ('a', 'b', 'c'):
            rows = np.where(groups == g)[0]
            assert not np.array_equal(perm[rows], rows), g

    def test_inner_cv_two_sessions_seed_and_fold_count(self):
        """Two sessions are enough for session folds; the folds are exactly the seeded
        ``StratifiedGroupKFold`` split with ``min(n_folds, n_sessions)`` folds (sessions with
        different class balances, so the seed actually changes the assignment)."""

        groups2 = np.repeat(np.array(['a', 'b']), 4)
        y2 = np.tile(np.array([1, 0]), 4)
        splits = mu.classification_inner_cv_splits(y2, groups2, 5, 0)
        assert isinstance(splits, list)
        assert len(splits) == 2
        assert sorted(set(groups2[val]).pop() for _train, val in splits) == ['a', 'b']
        counts = [(3, 1), (1, 3), (3, 1), (1, 3), (2, 2), (2, 2), (3, 1), (1, 3), (4, 0), (0, 4), (2, 2), (3, 1)]
        groups = np.concatenate([[f"s{i:02d}"] * (n_pos + n_neg) for i, (n_pos, n_neg) in enumerate(counts)])
        y = np.concatenate([[1] * n_pos + [0] * n_neg for n_pos, n_neg in counts])
        splits = mu.classification_inner_cv_splits(y, groups, 3, 11)
        assert isinstance(splits, list)
        expected = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=11).split(
            np.zeros(len(y)), y, groups))
        assert len(splits) == 3
        for (train, val), (e_train, e_val) in zip(splits, expected, strict=True):
            np.testing.assert_array_equal(train, e_train)
            np.testing.assert_array_equal(val, e_val)

    def test_inner_cv_falls_back_when_only_a_validation_fold_misses_a_class(self):
        """A session holding only one class becomes a validation fold without the other class
        (while its training side still has both), which is enough to fall back to the int."""

        groups = np.repeat(np.array(['a', 'b', 'c']), 4)
        y = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1])
        assert mu.classification_inner_cv_splits(y, groups, 3, 0) == 3

    def test_ece_hand_computed_with_ten_bins(self):
        """Hand-computed ECE: confidences 0.52 / 0.58 share the (0.5, 0.6] bin (accuracy 0.5,
        mean confidence 0.55, gap 0.05) and 0.61 / 0.64 share (0.6, 0.7] (gap 0.125), so ECE
        = 0.5 * 0.05 + 0.5 * 0.125 = 0.0875; 8 or 11 bins would split a pair differently."""

        conf = np.array([0.52, 0.58, 0.61, 0.64])
        proba = np.column_stack([1.0 - conf, conf])
        y_pred = np.ones(4, dtype=int)
        y_true = np.array([1, 0, 1, 0])
        assert expected_calibration_error(y_true, y_pred, proba) == pytest.approx(0.0875, abs=1e-12)

    def test_ece_bin_edges_are_right_closed(self):
        """A confidence exactly on an inner bin edge belongs only to the bin it closes; one
        just above the edge (by less than the top bin's 1e-9 tolerance) belongs only to the
        next bin. Hand values: edge 0.6 correct + 0.7 wrong -> 0.5*0.4 + 0.5*0.7 = 0.55; a
        single correct confidence 0.6 + 5e-10 -> 0.4 - 5e-10."""

        edges = np.linspace(0.0, 1.0, 11)
        on_edge = edges[6]
        proba = np.array([[1.0 - on_edge, on_edge], [0.3, 0.7]])
        val = expected_calibration_error(np.array([1, 0]), np.array([1, 1]), proba)
        assert val == pytest.approx(0.5 * (1.0 - on_edge) + 0.5 * 0.7, abs=1e-12)
        above = edges[6] + 5e-10
        val = expected_calibration_error(np.array([1]), np.array([1]), np.array([[1.0 - above, above]]))
        assert val == pytest.approx(1.0 - above, abs=1e-12)

    def test_ece_top_bin_tolerance(self):
        """The top bin takes confidences up to 1 + 1e-9 inclusive (rounding just above 1)
        and nothing beyond: a wrong prediction at 1 + 1e-9 counts with gap 1 + 1e-9 and one
        at 1 + 1e-6 is unbinned but still counts in N, so ECE = (1 + 1e-9) / 2."""

        proba = np.array([[0.0, 1.0 + 1e-9], [0.0, 1.0 + 1e-6]])
        val = expected_calibration_error(np.array([0, 0]), np.array([1, 1]), proba)
        assert val == pytest.approx((1.0 + 1e-9) / 2.0, rel=1e-12, abs=0.0)
        val = expected_calibration_error(np.array([1]), np.array([1]), np.array([[0.0, 1.0]]))
        assert val == pytest.approx(0.0, abs=1e-12)

    def test_ece_empty_input_is_nan(self):
        """No samples -> NaN (the documented empty-input return)."""

        assert np.isnan(expected_calibration_error(np.array([]), np.array([]), np.empty((0, 2))))

    def test_safe_metric_nan_paths(self):
        """Inputs the libraries reject (mismatched lengths) give NaN from MCC, Pearson and
        Spearman, a constant y_true gives NaN, and an input whose standard deviation is
        exactly 1.0 is a valid (finite) correlation."""

        assert np.isnan(safe_matthews_corrcoef(np.array([0, 1]), np.array([0, 1, 1])))
        assert np.isnan(pearson_r_safe(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0])))
        assert np.isnan(spearman_r_safe(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0])))
        assert np.isnan(pearson_r_safe(np.ones(4), np.arange(4.0)))
        assert np.isnan(spearman_r_safe(np.ones(4), np.arange(4.0)))
        y_unit_sd = np.array([-1.0, -1.0, 1.0, 1.0])
        assert np.std(y_unit_sd) == 1.0
        y_pred = np.array([0.0, 1.0, 2.0, 4.0])
        assert pearson_r_safe(y_unit_sd, y_pred) == pytest.approx(np.corrcoef(y_unit_sd, y_pred)[0, 1])
        assert spearman_r_safe(y_unit_sd, y_pred) == pytest.approx(0.8944271909999159)

    def test_safe_fold_mean_paths(self):
        """The fold mean ignores NaNs (and ``None``, read as NaN), and an empty or all-NaN
        list gives NaN without a RuntimeWarning."""

        assert mu._safe_fold_mean([1.0, float('nan'), 3.0]) == pytest.approx(2.0)
        assert mu._safe_fold_mean([1.0, None, 3.0]) == pytest.approx(2.0)
        assert np.isnan(mu._safe_fold_mean([float('nan'), float('nan')]))
        assert np.isnan(mu._safe_fold_mean([]))

    def test_align_rejects_duplicates_and_keeps_dtype(self):
        """A canonical ordering with a repeated class is not strictly ascending and is
        rejected; the aligned matrix keeps the input probabilities' dtype (float32 here)."""

        with pytest.raises(ValueError, match='strictly ascending'):
            align_probs_to_canonical(np.array([[0.5, 0.5]]), model_classes=np.array([0, 1]),
                                     canonical_classes=np.array([0, 1, 1]))
        out = align_probs_to_canonical(np.array([[0.25, 0.75]], dtype=np.float32),
                                       model_classes=np.array([0, 2]), canonical_classes=np.array([0, 1, 2]))
        assert out.dtype == np.float32
        np.testing.assert_array_equal(out, np.array([[0.25, 0.0, 0.75]], dtype=np.float32))


class TestRunPredictorAuditsForwarding:
    """The audit wrapper's exact hand-off to both audits: every keyword it forwards, the event
    times, intervals and onset times it builds (and the sessions it skips on the way), the
    seeded per-session balancing, the IBI thresholds and the timescale settings validation."""

    @staticmethod
    def _settings(seed=None, collinearity=True, timescale=True):
        """
        Description
        -----------
        A complete diagnostics settings dictionary with distinct values for every knob the
        wrapper forwards.

        Parameters
        ----------
        seed (int | None)
            ``model_validation.random_seed``.
        collinearity (bool)
            Whether the collinearity audit is enabled.
        timescale (bool)
            Whether the timescale audit is enabled.

        Returns
        -------
        settings (dict)
            The settings dictionary.
        """

        return {
            'diagnostics': {
                'collinearity_audit': collinearity,
                'collinearity_concern_threshold': 0.7,
                'collinearity_exclude_threshold': 0.85,
                'timescale_audit': timescale,
                'timescale_max_lag_seconds': 5.0,
                'timescale_n_shuffles': 3,
                'timescale_shuffle_range': [20.0, 60.0],
                'timescale_signal_floor_seconds': 0.5,
                'timescale_signal_min_run_seconds': 0.2,
            },
            'model_params': {'filter_history': 1.5, 'mixture_model_component_index': 0,
                             'mixture_model_z_score': 1.0},
            'model_validation': {'random_seed': seed},
            'mixture_model_params': {'male': {'means': [0.1], 'sds': [0.05]},
                                     'female': {'means': [0.12], 'sds': [0.06]}},
        }

    @staticmethod
    def _skip_inputs():
        """
        Description
        -----------
        Sessions that every extraction loop must skip, listed BEFORE the one valid session
        (so a ``break`` in place of a ``continue`` loses the valid session): absent from the
        USV dict, target mouse absent, no ``stop``, empty ``start``, empty ``stop``; the valid
        session has a single start/stop pair and unsorted onsets.

        Returns
        -------
        inputs (tuple)
            ``(beh, usv, names, fps)``.
        """

        names = {sess: ['mM', 'mF'] for sess in
                 ('s_absent', 's_no_target', 's_no_stop', 's_empty_start', 's_empty_stop', 's_ok')}
        usv = {
            's_no_target': {'mM': {'positive_events': np.array([0.1]),
                                   'start': np.array([0.1]), 'stop': np.array([0.2])}},
            's_no_stop': {'mF': {'positive_events': np.array([]), 'start': np.array([0.1])}},
            's_empty_start': {'mF': {'start': np.array([]), 'stop': np.array([0.2])}},
            's_empty_stop': {'mF': {'start': np.array([0.1]), 'stop': np.array([])}},
            's_ok': {'mF': {'positive_events': np.array([1.5, 0.5]),
                            'start': np.array([0.5]), 'stop': np.array([0.6])}},
        }
        beh = {'s_ok': pls.DataFrame({'mF.speed': [0.0, 1.0, 2.0]})}
        fps = {'s_ok': 150.0}
        return beh, usv, names, fps

    def test_exact_keywords_and_built_inputs(self, mocker, tmp_path):
        """Both audits get exactly the expected keywords: the inputs passed through, the
        artifact paths, the thresholds, the IBI thresholds of component 0, the seed (None ->
        0), the validated timescale settings, and the event times / intervals / onsets built
        from the one valid session only (a missing event key is skipped, not a stop)."""

        coll = mocker.patch.object(mu, 'audit_predictor_collinearity')
        ts = mocker.patch.object(mu, 'audit_predictor_timescales')
        beh, usv, names, fps = self._skip_inputs()
        meta = {'cohort': 'unit-test'}
        run_predictor_audits(beh, usv, names, fps, target_idx=1, predictor_idx=0, history_frames=5,
                             event_keys=['missing_key', 'positive_events'], settings=self._settings(),
                             save_dir=str(tmp_path), pickle_basename='myinput.pkl', input_metadata=meta)
        coll.assert_called_once()
        ts.assert_called_once()
        assert coll.call_args.kwargs['processed_beh_dict'] is beh
        assert ts.call_args.kwargs['processed_beh_dict'] is beh
        _assert_same_kwargs(coll.call_args.kwargs, {
            'processed_beh_dict': beh,
            'event_times_per_session': {'s_ok': np.array([0.5, 1.5])},
            'mouse_names_dict': names,
            'target_idx': 1,
            'predictor_idx': 0,
            'history_frames': 5,
            'camera_fps_dict': fps,
            'save_path': str(tmp_path / 'myinput_collinearity.pkl'),
            'source_pickle': 'myinput.pkl',
            'concern_thresh': 0.7,
            'exclude_thresh': 0.85,
            'input_metadata': meta,
        })
        _assert_same_kwargs(ts.call_args.kwargs, {
            'processed_beh_dict': beh,
            'mouse_names_dict': names,
            'target_idx': 1,
            'predictor_idx': 0,
            'configured_filter_history': 1.5,
            'camera_fps_dict': fps,
            'max_lag_seconds': 5.0,
            'n_shuffles': 3,
            'ibi_thresholds': {'male': float(mu._calculate_ibi_threshold(0.1, 0.05, 1.0)),
                               'female': float(mu._calculate_ibi_threshold(0.12, 0.06, 1.0))},
            'save_path': str(tmp_path / 'myinput_timescales.pkl'),
            'source_pickle': 'myinput.pkl',
            'random_seed': 0,
            'input_metadata': meta,
            'shuffle_range_seconds': (20.0, 60.0),
            'event_intervals_per_session': {'s_ok': (np.array([0.5]), np.array([0.6]))},
            'onset_times_per_session': {'s_ok': np.array([0.5, 1.5])},
            'signal_floor_seconds': 0.5,
            'signal_min_run_seconds': 0.2,
        })

    def test_configured_seed_reaches_the_timescale_audit(self, mocker, tmp_path):
        """A configured ``random_seed`` (7) is forwarded as is."""

        mocker.patch.object(mu, 'audit_predictor_collinearity')
        ts = mocker.patch.object(mu, 'audit_predictor_timescales')
        beh, usv, names, fps = self._skip_inputs()
        run_predictor_audits(beh, usv, names, fps, target_idx=1, predictor_idx=0, history_frames=5,
                             event_keys=['positive_events'], settings=self._settings(seed=7),
                             save_dir=str(tmp_path), pickle_basename='s.pkl')
        assert ts.call_args.kwargs['random_seed'] == 7

    @staticmethod
    def _balance_inputs():
        """
        Description
        -----------
        One session whose target has 3 ``other_events`` and 8 ``target_events``.

        Returns
        -------
        inputs (tuple)
            ``(beh, usv, names, fps, small, large)``.
        """

        small = np.array([100.0, 101.0, 102.0])
        large = np.arange(8, dtype=float)
        usv = {'s1': {'mF': {'other_events': small, 'target_events': large,
                             'start': np.array([0.1]), 'stop': np.array([0.2])}}}
        beh = {'s1': pls.DataFrame({'mF.speed': [0.0, 1.0]})}
        return beh, usv, {'s1': ['mM', 'mF']}, {'s1': 30.0}, small, large

    @pytest.mark.parametrize(('seed', 'rng_seed'), [(7, 7), (None, 0), (3, 3)])
    def test_balanced_pool_is_the_seeded_draw(self, mocker, tmp_path, seed, rng_seed):
        """With ``balance_event_keys`` the smallest key (listed first) is kept whole WITHOUT
        consuming the generator, and the larger key is cut to its size by one draw without
        replacement from ``default_rng(random_seed)`` (``None`` -> 0); the pooled times are
        that draw, sorted."""

        coll = mocker.patch.object(mu, 'audit_predictor_collinearity')
        mocker.patch.object(mu, 'audit_predictor_timescales')
        beh, usv, names, fps, small, large = self._balance_inputs()
        run_predictor_audits(beh, usv, names, fps, target_idx=1, predictor_idx=0, history_frames=5,
                             event_keys=['other_events', 'target_events'],
                             settings=self._settings(seed=seed, timescale=False),
                             save_dir=str(tmp_path), pickle_basename='b.pkl', balance_event_keys=True)
        pick = np.random.default_rng(rng_seed).choice(8, size=3, replace=False)
        expected = np.sort(np.concatenate([small, large[pick]]))
        np.testing.assert_array_equal(coll.call_args.kwargs['event_times_per_session']['s1'], expected)

    def test_unbalanced_pool_keeps_every_event(self, mocker, tmp_path):
        """Without ``balance_event_keys`` every event of every key is pooled (sorted)."""

        coll = mocker.patch.object(mu, 'audit_predictor_collinearity')
        mocker.patch.object(mu, 'audit_predictor_timescales')
        beh, usv, names, fps, small, large = self._balance_inputs()
        run_predictor_audits(beh, usv, names, fps, target_idx=1, predictor_idx=0, history_frames=5,
                             event_keys=['other_events', 'target_events'],
                             settings=self._settings(timescale=False),
                             save_dir=str(tmp_path), pickle_basename='b.pkl')
        np.testing.assert_array_equal(coll.call_args.kwargs['event_times_per_session']['s1'],
                                      np.sort(np.concatenate([small, large])))

    def _run_timescale(self, mocker, tmp_path, settings, **extra):
        """
        Description
        -----------
        Runs the wrapper with only the timescale audit enabled on the skip inputs and
        returns the timescale spy.

        Parameters
        ----------
        mocker (pytest_mock.MockerFixture)
            The mocker fixture.
        tmp_path (pathlib.Path)
            The save directory.
        settings (dict)
            The settings dictionary.
        **extra
            Further keyword arguments for ``run_predictor_audits``.

        Returns
        -------
        ts (unittest.mock.MagicMock)
            The timescale-audit spy.
        """

        mocker.patch.object(mu, 'audit_predictor_collinearity')
        ts = mocker.patch.object(mu, 'audit_predictor_timescales')
        beh, usv, names, fps = self._skip_inputs()
        run_predictor_audits(beh, usv, names, fps, target_idx=1, predictor_idx=0, history_frames=5,
                             event_keys=['positive_events'], settings=settings,
                             save_dir=str(tmp_path), pickle_basename='t.pkl', **extra)
        return ts

    def test_ibi_threshold_component_index_bounds(self, mocker, tmp_path):
        """Component index 1 of two components gives that component's threshold; index 1 of
        one component (== the component count) is out of range and gives NaN, and the audit
        still runs."""

        settings = self._settings(timescale=True, collinearity=False)
        settings['model_params']['mixture_model_component_index'] = 1
        settings['mixture_model_params'] = {'male': {'means': [0.1, 0.3], 'sds': [0.05, 0.07]},
                                            'female': {'means': [0.12], 'sds': [0.06]}}
        ts = self._run_timescale(mocker, tmp_path, settings)
        ts.assert_called_once()
        thr = ts.call_args.kwargs['ibi_thresholds']
        assert thr['male'] == float(mu._calculate_ibi_threshold(0.3, 0.07, 1.0))
        assert np.isnan(thr['female'])

    @pytest.mark.parametrize(('bad_key', 'bad_val'), [
        ('timescale_shuffle_range', [20.0, 40.0, 60.0]),   # three elements
        ('timescale_shuffle_range', [0.0, 60.0]),          # min not > 0
        ('timescale_shuffle_range', [30.0, 30.0]),         # min == max
    ])
    def test_rejected_timescale_settings(self, mocker, tmp_path, capsys, bad_key, bad_val):
        """A three-element shuffle range, a zero minimum and an equal min / max are rejected
        (the audit is not run and the non-fatal warning is printed)."""

        settings = self._settings(collinearity=False)
        settings['diagnostics'][bad_key] = bad_val
        ts = self._run_timescale(mocker, tmp_path, settings)
        ts.assert_not_called()
        assert 'timescale audit failed' in capsys.readouterr().out

    @pytest.mark.parametrize(('key', 'val', 'kwarg', 'expected'), [
        ('timescale_shuffle_range', [0.5, 60.0], 'shuffle_range_seconds', (0.5, 60.0)),
        ('timescale_signal_floor_seconds', 0.0, 'signal_floor_seconds', 0.0),
    ])
    def test_accepted_boundary_timescale_settings(self, mocker, tmp_path, key, val, kwarg, expected):
        """A shuffle-range minimum below 1 s and a zero signal floor are valid and forwarded."""

        settings = self._settings(collinearity=False)
        settings['diagnostics'][key] = val
        ts = self._run_timescale(mocker, tmp_path, settings)
        ts.assert_called_once()
        assert ts.call_args.kwargs[kwarg] == expected

    def test_single_precomputed_onset_is_kept(self, mocker, tmp_path):
        """A precomputed onset array with ONE onset is a non-empty session and is kept."""

        ts = self._run_timescale(mocker, tmp_path, self._settings(collinearity=False),
                                 precomputed_onset_times={'s_ok': np.array([2.0])})
        np.testing.assert_array_equal(ts.call_args.kwargs['onset_times_per_session']['s_ok'], np.array([2.0]))

    def test_no_target_entries_is_not_a_wrong_key(self, mocker, tmp_path):
        """When no session carries a target-mouse entry at all, an empty onset set is not
        the wrong-key error: the audit runs with no onsets."""

        mocker.patch.object(mu, 'audit_predictor_collinearity')
        ts = mocker.patch.object(mu, 'audit_predictor_timescales')
        run_predictor_audits({}, {'s1': {'mM': {}}}, {'s1': ['mM', 'mF']}, {}, target_idx=1, predictor_idx=0,
                             history_frames=5, event_keys=['positive_events'],
                             settings=self._settings(collinearity=False),
                             save_dir=str(tmp_path), pickle_basename='e.pkl')
        ts.assert_called_once()
        assert ts.call_args.kwargs['onset_times_per_session'] == {}

    def test_wrong_key_message_counts_sessions_with_a_target(self, mocker, tmp_path, capsys):
        """The wrong-key warning names the key and the number of sessions that had a target
        entry (here 2 of 3)."""

        mocker.patch.object(mu, 'audit_predictor_collinearity')
        ts = mocker.patch.object(mu, 'audit_predictor_timescales')
        names = {'s1': ['mM', 'mF'], 's2': ['mM', 'mF'], 's3': ['mM', 'mF']}
        usv = {'s1': {'mF': {'start': np.array([0.1])}}, 's2': {'mF': {}}, 's3': {'mM': {}}}
        run_predictor_audits({}, usv, names, {}, target_idx=1, predictor_idx=0, history_frames=5,
                             event_keys=['positive_events'], settings=self._settings(collinearity=False),
                             save_dir=str(tmp_path), pickle_basename='w.pkl', onset_event_key='bout_onsets')
        ts.assert_not_called()
        out = capsys.readouterr().out
        assert "`onset_event_key='bout_onsets'`" in out
        assert 'found in any of the 2 session(s)' in out
