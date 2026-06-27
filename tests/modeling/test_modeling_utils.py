"""
@author: bartulem
Unit tests for the pure helpers in ``usv_playpen.modeling.modeling_utils``:
the classification / regression metric wrappers, the two-class pooling and
balancing array ops, history unrolling, and the mouse-role / kinematic
column selectors.

The metrics are checked against their textbook extremes (perfect / chance
/ degenerate), the array ops against shape and label invariants (with the
global RNG seeded for the down-sampling draws), and the column selectors
against small synthetic column lists exercising the directional dyadic
folding rules documented in the source.
"""

from __future__ import annotations

import numpy as np
import polars as pls
import pytest

import usv_playpen.modeling.modeling_utils as mu
from usv_playpen.modeling.modeling_utils import (
    align_probs_to_canonical,
    balance_two_class_arrays,
    bounded_test_proportion,
    brier_score_multi,
    build_vocal_signal_columns,
    concat_two_class_with_labels,
    expected_calibration_error,
    harmonize_session_columns,
    identify_empty_event_sessions,
    mean_absolute_error_1d,
    pearson_r_safe,
    pool_session_arrays,
    prepare_modeling_sessions,
    resolve_mouse_roles,
    root_mean_squared_error,
    run_predictor_audits,
    safe_confusion_matrix,
    safe_matthews_corrcoef,
    select_kinematic_columns,
    shuffle_train_test_arrays,
    unroll_history_matrix,
    zscore_features_across_sessions,
)


# Probabilistic / classification metrics


class TestBrierScoreMulti:

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

        with pytest.raises(ValueError):
            expected_calibration_error(np.array([0]), np.array([0]), np.array([1.0]))


class TestSafeMatthews:

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

    def test_canonical_shape_with_missing_class(self):
        """Passing explicit ``labels`` forces the canonical (K, K) shape
        even when a class is never observed in this fold."""

        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        cm = safe_confusion_matrix(y_true, y_pred, labels=np.array([0, 1, 2]))
        assert cm.shape == (3, 3)
        # Perfect prediction -> mass only on the diagonal of observed classes.
        assert cm[0, 0] == 2 and cm[1, 1] == 2
        assert cm[2].sum() == 0


class TestAlignProbsToCanonical:

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
        with pytest.raises(ValueError):
            align_probs_to_canonical(probs, model_classes=np.array([2]),
                                     canonical_classes=np.array([2, 1, 0]))


# Regression metrics


class TestRegressionMetrics:

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

    def test_downsamples_majority_to_minority(self):
        """The majority class is subsampled down to the minority count;
        both outputs end up with ``min(n_pos, n_neg)`` rows."""

        np.random.seed(0)
        X_pos = np.arange(100 * 3).reshape(100, 3).astype(float)
        X_neg = np.arange(10 * 3).reshape(10, 3).astype(float)
        bp, bn = balance_two_class_arrays(X_pos, X_neg)
        assert bp.shape == (10, 3)
        assert bn.shape == (10, 3)

    def test_reproducible_under_seed(self):
        """The down-sampling draw uses the global RNG, so re-seeding gives
        an identical subsample."""

        X_pos = np.arange(100 * 2).reshape(100, 2).astype(float)
        X_neg = np.zeros((10, 2))
        np.random.seed(7)
        bp1, _ = balance_two_class_arrays(X_pos, X_neg)
        np.random.seed(7)
        bp2, _ = balance_two_class_arrays(X_pos, X_neg)
        np.testing.assert_array_equal(bp1, bp2)

    def test_empty_input_returns_empty(self):
        """If one class is empty both outputs are empty with preserved
        column counts."""

        bp, bn = balance_two_class_arrays(np.empty((0, 4)), np.ones((5, 4)))
        assert bp.shape == (0, 4)
        assert bn.shape == (0, 4)


class TestConcatTwoClassWithLabels:

    def test_stacks_and_labels(self):
        """Positive rows come first with label 1, negatives after with
        label 0; the stacked feature matrix preserves that order."""

        X_pos = np.ones((2, 3))
        X_neg = np.zeros((3, 3))
        X, y = concat_two_class_with_labels(X_pos, X_neg)
        assert X.shape == (5, 3)
        np.testing.assert_array_equal(y, np.array([1.0, 1.0, 0.0, 0.0, 0.0]))


class TestUnrollHistoryMatrix:

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

    def test_preserves_label_alignment_and_shapes(self):
        """Shuffling permutes rows but keeps each X row paired with its y
        label and leaves shapes unchanged."""

        X_train = np.arange(20).reshape(10, 2).astype(float)
        y_train = X_train[:, 0].copy()  # label encodes the row identity
        X_test = np.arange(8).reshape(4, 2).astype(float)
        y_test = X_test[:, 0].copy()
        np.random.seed(1)
        Xtr, ytr, Xte, yte = shuffle_train_test_arrays(X_train, y_train, X_test, y_test)
        assert Xtr.shape == X_train.shape
        assert Xte.shape == X_test.shape
        # Column-0 value still equals the paired label after the shuffle.
        np.testing.assert_array_equal(Xtr[:, 0], ytr)
        np.testing.assert_array_equal(Xte[:, 0], yte)


class TestBoundedTestProportion:

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


# Mouse-role / kinematic column selectors


class TestResolveMouseRoles:

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

    def _kin_settings(self, **overrides):
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

    def _settings(self, seed, session_file):
        return {'model_params': {'random_seed': seed},
                'io': {'session_list_file': str(session_file)}}

    def test_loads_paths_and_seeds_rng(self, tmp_path):
        """A session-list file with blank lines is read into a path list
        (blanks skipped) and a non-None seed makes the global RNG
        deterministic."""

        f = tmp_path / 'sessions.txt'
        f.write_text("/some/sessionA\n\n/some/sessionB\n")
        out = prepare_modeling_sessions(self._settings(123, f))
        assert out == ['/some/sessionA', '/some/sessionB']
        # Seed was applied: next draw matches a freshly-seeded RNG.
        first = np.random.random()
        np.random.seed(123)
        # Advance past the same point prepare_modeling_sessions left the RNG.
        np.random.random()
        # Re-running prepare resets the seed identically.
        prepare_modeling_sessions(self._settings(123, f))
        assert np.random.random() == pytest.approx(first)

    def test_none_seed_branch(self, tmp_path):
        """A None seed takes the non-deterministic branch and still returns
        the loaded paths."""

        f = tmp_path / 's.txt'
        f.write_text("/x/y\n")
        out = prepare_modeling_sessions(self._settings(None, f))
        assert out == ['/x/y']

    def test_missing_file_raises_filenotfound(self, tmp_path):
        """A non-existent session-list file raises ``FileNotFoundError``."""

        with pytest.raises(FileNotFoundError):
            prepare_modeling_sessions(self._settings(0, tmp_path / 'nope.txt'))

    def test_empty_file_raises_valueerror(self, tmp_path):
        """A file with only blank lines yields no paths -> ``ValueError``."""

        f = tmp_path / 'blank.txt'
        f.write_text("\n   \n\n")
        with pytest.raises(ValueError):
            prepare_modeling_sessions(self._settings(0, f))


# Vocal signal column construction


class TestBuildVocalSignalColumns:

    def _usv_dict(self):
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
        """A falsy ``usv_predictor_type`` short-circuits to empty lists."""

        cols, names = build_vocal_signal_columns(
            self._usv_dict(), 'sess', 'mTarget', 'mPartner',
            voc_settings={'usv_predictor_type': None,
                          'usv_predictor_partner_only': False},
        )
        assert cols == [] and names == []

    def test_self_exclusion_drops_rate_and_event(self):
        """For the target mouse the default ``usv_self_exclude`` keys
        (rate/event) are skipped, but per-category signals survive; the
        partner mouse keeps everything."""

        cols, names = build_vocal_signal_columns(
            self._usv_dict(), 'sess', 'mTarget', 'mPartner',
            voc_settings={'usv_predictor_type': 'continuous',
                          'usv_predictor_partner_only': False},
        )
        assert 'mTarget.usv_rate' not in names
        assert 'mTarget.usv_event' not in names
        assert 'mTarget.usv_cat_0' in names          # category survives
        assert 'mPartner.usv_rate' in names          # partner keeps rate
        assert 'mPartner.usv_cat_0' in names
        # The Series list mirrors the names list one-to-one.
        assert [s.name for s in cols] == names

    def test_partner_only_emits_partner_signals_only(self):
        """``usv_predictor_partner_only`` restricts emission to the partner
        mouse."""

        _cols, names = build_vocal_signal_columns(
            self._usv_dict(), 'sess', 'mTarget', 'mPartner',
            voc_settings={'usv_predictor_type': 'continuous',
                          'usv_predictor_partner_only': True},
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
            voc_settings={'usv_predictor_type': 'continuous',
                          'usv_predictor_partner_only': True},
        )
        # partner_only=True but partner absent -> nothing emitted.
        assert names == []


# Empty-event session identification


class TestIdentifyEmptyEventSessions:

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
        assert 'mM.speed' in out['s1'].columns and 'mF.speed' in out['s1'].columns
        assert 'mM.speed' in out['s2'].columns and 'mF.speed' in out['s2'].columns
        assert suffixes == sorted(suffixes)
        assert 'speed' in suffixes and 'nose-nose' in suffixes

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

    def _settings(self, collinearity=True, timescale=True):
        return {
            'diagnostics': {
                'collinearity_audit': collinearity,
                'timescale_audit': timescale,
                'timescale_max_lag_seconds': 5.0,
                'timescale_n_shuffles': 3,
                'timescale_shuffle_range': [20.0, 60.0],
                'timescale_signal_floor_seconds': 0.5,
                'timescale_signal_min_run_seconds': 0.2,
            },
            'model_params': {
                'random_seed': 0,
                'filter_history': 1.0,
                'mixture_model_component_index': 0,
                'mixture_model_z_score': 1.0,
            },
            'mixture_model_params': {
                'male': {'means': [0.1], 'sds': [0.05]},
                'female': {'means': [0.12], 'sds': [0.06]},
            },
        }

    def _inputs(self):
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
        assert 's1' in ts_kwargs['bout_onset_times_per_session']

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
            bout_onset_event_key='target_events',
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
        """A ``bout_onset_event_key`` absent from every target entry raises
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
            bout_onset_event_key='does_not_exist',
        )
        ts.assert_not_called()      # guard raised before the audit call
        assert 'timescale audit failed' in capsys.readouterr().out

    def test_precomputed_bout_onset_times_used(self, mocker, tmp_path):
        """``precomputed_bout_onset_times`` bypasses the string-key lookup and
        is forwarded (sorted, empties dropped) as the timescale ``Y`` source."""

        mocker.patch.object(mu, 'audit_predictor_collinearity')
        ts = mocker.patch.object(mu, 'audit_predictor_timescales')
        beh, usv, names, fps = self._inputs()
        run_predictor_audits(
            beh, usv, names, fps, target_idx=1, predictor_idx=0,
            history_frames=5, event_keys=['positive_events'],
            settings=self._settings(collinearity=False),
            save_dir=str(tmp_path), pickle_basename='h.pkl',
            precomputed_bout_onset_times={'s1': np.array([3.0, 1.0, 2.0]),
                                          's_empty': np.array([])},
        )
        bo = ts.call_args.kwargs['bout_onset_times_per_session']
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
        assert np.isnan(thr['male']) and np.isnan(thr['female'])

    @pytest.mark.parametrize('bad_key,bad_val', [
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
        bo = ts.call_args.kwargs['bout_onset_times_per_session']
        assert set(bo) == {'s_ok'}
